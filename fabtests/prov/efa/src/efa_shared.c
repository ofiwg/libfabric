/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include <rdma/fi_cm.h>

#include <shared.h>
#include "efa_shared.h"

static struct option efa_extra_opts[] = {
	{"high-pps", no_argument, NULL, OPT_HIGH_PPS},
	{"post-list", required_argument, NULL, OPT_POST_LIST},
	{"num-eps", required_argument, NULL, OPT_NUM_EPS},
	{"sl-low-latency", no_argument, NULL, OPT_SL_LOW_LATENCY},
	{"eps-per-domain", required_argument, NULL, OPT_EPS_PER_DOMAIN},
	{"domains", required_argument, NULL, OPT_DOMAINS},
	{0, 0, 0, 0}
};

struct option *efa_long_opts;

void build_efa_long_opts(void)
{
	static bool initialized = false;
	int shared_cnt, i;
	int extra_cnt = sizeof(efa_extra_opts) / sizeof(efa_extra_opts[0]) - 1;

	if (initialized)
		return;

	for (shared_cnt = 0; long_opts[shared_cnt].name; shared_cnt++)
		;
	efa_long_opts = calloc(shared_cnt + extra_cnt + 1, sizeof(struct option));
	for (i = 0; i < extra_cnt; i++)
		efa_long_opts[i] = efa_extra_opts[i];
	for (i = 0; i < shared_cnt; i++)
		efa_long_opts[extra_cnt + i] = long_opts[i];

	initialized = true;
}

void efa_longopts_usage(void)
{
	FT_PRINT_OPTS_USAGE("--high-pps",
		"Enable FI_EFA_WR_HIGH_PPS flag on writes");
	FT_PRINT_OPTS_USAGE("--post-list <n>",
		"Batch n posts per doorbell using FI_MORE (default: 1)");
	FT_PRINT_OPTS_USAGE("-q <n>, --num-eps <n>",
		"Number of endpoints/QPs (default: 1)");
	FT_PRINT_OPTS_USAGE("--sl-low-latency",
		"Enable FI_TC_LOW_LATENCY on all endpoints used in the test");
	FT_PRINT_OPTS_USAGE("--eps-per-domain <n>",
		"Endpoints to place on each local domain (default: all on one)");
	FT_PRINT_OPTS_USAGE("--domains <d1,d2,...>",
		"Domain names to spread endpoints across; requires "
		"--eps-per-domain and excludes -d");
	ft_longopts_usage();
}

/**
 * efa_calc_peer_distribution - Distribute peers across workers
 * @my_id: ID of the local worker to calculate distribution for
 * @my_count: Total number of local workers
 * @peer_count: Total number of peers to distribute
 * @num_peers: Output - number of peers assigned to this worker
 * @peer_ids: Output - dynamically allocated array of peer IDs for this worker
 *
 * Distributes a given number of remote peers across local workers
 *
 * When peers <= workers, each worker gets one peer.
 * The index of the assigned peer is worker_id % total_peers
 *
 * When peers > workers, the peers are distributed in a
 * round-robin fashion. If the number of peers is not divisible by
 * the number of local workers, the remaining peers are distributed
 * to lower-numbered workers.
 *
 * Returns: FI_SUCCESS on success, -FI_ENOMEM on allocation failure
 *
 * Note: Caller must free the allocated peer_ids array
 */
int efa_calc_peer_distribution(int my_idx, int my_count, int peer_count,
			       int *num_peers, int **peer_ids)
{
	int n, i;

	if (peer_count <= my_count) {
		n = 1;
		*peer_ids = malloc(sizeof(int));
		if (!*peer_ids)
			return -FI_ENOMEM;
		(*peer_ids)[0] = my_idx % peer_count;
	} else {
		assert(my_count > 0);
		n = peer_count / my_count;
		if (my_idx < peer_count % my_count)
			n++;
		*peer_ids = malloc(n * sizeof(int));
		if (!*peer_ids)
			return -FI_ENOMEM;
		for (i = 0; i < n; i++)
			(*peer_ids)[i] = my_idx + i * my_count;
	}

	*num_peers = n;
	return FI_SUCCESS;
}

int efa_exchange_addrs_oob(int oob_sock, bool is_initiator,
			   struct fid_ep **local_eps, int local_n,
			   struct fid_av *av, fi_addr_t *remote_addrs,
			   int remote_n)
{
	char *local_buf = NULL, *remote_buf = NULL;
	size_t addrlen;
	int i, ret;

	local_buf = calloc(local_n, FT_MAX_CTRL_MSG);
	remote_buf = calloc(remote_n, FT_MAX_CTRL_MSG);
	if (!local_buf || !remote_buf) {
		ret = -FI_ENOMEM;
		goto out;
	}

	for (i = 0; i < local_n; i++) {
		addrlen = FT_MAX_CTRL_MSG;
		ret = fi_getname(&local_eps[i]->fid,
				 local_buf + (size_t) i * FT_MAX_CTRL_MSG,
				 &addrlen);
		if (ret) {
			FT_PRINTERR("fi_getname", ret);
			goto out;
		}
	}

	if (is_initiator) {
		ret = ft_sock_send(oob_sock, local_buf,
				   (size_t) local_n * FT_MAX_CTRL_MSG);
		if (ret)
			goto out;
		ret = ft_sock_recv(oob_sock, remote_buf,
				   (size_t) remote_n * FT_MAX_CTRL_MSG);
		if (ret)
			goto out;
	} else {
		ret = ft_sock_recv(oob_sock, remote_buf,
				   (size_t) remote_n * FT_MAX_CTRL_MSG);
		if (ret)
			goto out;
		ret = ft_sock_send(oob_sock, local_buf,
				   (size_t) local_n * FT_MAX_CTRL_MSG);
		if (ret)
			goto out;
	}

	for (i = 0; i < remote_n; i++) {
		ret = ft_av_insert(av,
				   remote_buf + (size_t) i * FT_MAX_CTRL_MSG,
				   1, &remote_addrs[i], 0, NULL);
		if (ret)
			goto out;
	}

	ret = FI_SUCCESS;
out:
	free(local_buf);
	free(remote_buf);
	return ret;
}
