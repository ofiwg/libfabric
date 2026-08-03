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

/*
 * Collect the distinct domain names the provider offers for @p hints. Returns a
 * NULL-terminated array of strdup'ed names, to be freed with
 * efa_free_domain_names(). fi_getinfo returns one fi_info per (domain, ep type,
 * ...) combination, so the same domain can appear several times; only the first
 * occurrence of each name is kept.
 */
static int efa_list_domains(struct fi_info *hints, char ***names, int *n_found)
{
	struct fi_info *dup, *info = NULL, *cur;
	char **arr = NULL;
	int n = 0, cnt = 0, i, ret;

	dup = fi_dupinfo(hints);
	if (!dup)
		return -FI_ENOMEM;

	/*
	 * Enumerate domains only: a specific name in hints would filter the
	 * list down to that one domain, which is the opposite of what we want.
	 */
	free(dup->domain_attr->name);
	dup->domain_attr->name = NULL;

	ret = fi_getinfo(ft_fiversion, NULL, NULL, 0, dup, &info);
	fi_freeinfo(dup);
	if (ret) {
		FT_PRINTERR("fi_getinfo", ret);
		return ret;
	}

	for (cur = info; cur; cur = cur->next)
		cnt++;

	/* +1 for the NULL terminator */
	arr = calloc(cnt + 1, sizeof(*arr));
	if (!arr) {
		ret = -FI_ENOMEM;
		goto out;
	}

	for (cur = info; cur; cur = cur->next) {
		bool seen = false;

		if (!cur->domain_attr || !cur->domain_attr->name)
			continue;

		for (i = 0; i < n; i++) {
			if (!strcmp(arr[i], cur->domain_attr->name)) {
				seen = true;
				break;
			}
		}
		if (seen)
			continue;

		arr[n] = strdup(cur->domain_attr->name);
		if (!arr[n]) {
			ret = -FI_ENOMEM;
			goto out_arr;
		}
		n++;
	}

	*names = arr;
	*n_found = n;
	ret = FI_SUCCESS;
	goto out;

out_arr:
	while (--n >= 0)
		free(arr[n]);
	free(arr);
out:
	fi_freeinfo(info);
	return ret;
}

void efa_free_domain_names(char **names)
{
	int i;

	if (!names)
		return;

	for (i = 0; names[i]; i++)
		free(names[i]);
	free(names);
}

/**
 * efa_select_domains - Pick the domains to spread endpoints across
 * @hints: fi_info hints describing the fabric/provider to enumerate
 * @want_n: number of domains needed
 * @explicit_csv: comma-separated domain names to use, or NULL to pick
 *                automatically from whatever the host offers
 * @names: Output - NULL-terminated array of @p want_n strdup'ed domain names
 *
 * The first @p want_n names are used; any extras are reported and ignored, so
 * one --domains list can serve several endpoint counts. Names given in
 * @p explicit_csv are validated against what the provider actually offers, so a
 * typo fails here with the list of real domains rather than later in fi_domain.
 *
 * Returns: FI_SUCCESS on success, -FI_ENODATA if fewer than @p want_n domains
 * are available, -FI_EINVAL if a listed name does not exist, or -FI_ENOMEM.
 *
 * Note: Caller must free @p names with efa_free_domain_names()
 */
int efa_select_domains(struct fi_info *hints, int want_n,
		       const char *explicit_csv, char ***names)
{
	char **avail = NULL, **want = NULL, **sel = NULL;
	size_t want_cnt = 0;
	int avail_n = 0, i, j, ret;

	ret = efa_list_domains(hints, &avail, &avail_n);
	if (ret)
		return ret;

	if (explicit_csv) {
		want = ft_split_and_alloc(explicit_csv, ",", &want_cnt);
		if (!want) {
			ret = -FI_ENOMEM;
			goto out;
		}

		if ((int) want_cnt < want_n) {
			FT_ERR("--domains lists %zu domain(s) but %d are needed",
			       want_cnt, want_n);
			ret = -FI_EINVAL;
			goto out;
		}

		/* Every name must exist, including the ones we will not use. */
		for (i = 0; i < (int) want_cnt; i++) {
			for (j = 0; j < avail_n; j++)
				if (!strcmp(want[i], avail[j]))
					break;
			if (j == avail_n) {
				FT_ERR("--domains: no such domain '%s'", want[i]);
				FT_ERR("available domains:");
				for (j = 0; j < avail_n; j++)
					FT_ERR("  %s", avail[j]);
				ret = -FI_EINVAL;
				goto out;
			}
		}

		if ((int) want_cnt > want_n)
			FT_INFO("--domains lists %zu domain(s), using the first "
				"%d", want_cnt, want_n);
	} else if (avail_n < want_n) {
		FT_ERR("found %d domain(s), need %d", avail_n, want_n);
		ret = -FI_ENODATA;
		goto out;
	}

	/* +1 for the NULL terminator */
	sel = calloc(want_n + 1, sizeof(*sel));
	if (!sel) {
		ret = -FI_ENOMEM;
		goto out;
	}

	for (i = 0; i < want_n; i++) {
		sel[i] = strdup(want ? want[i] : avail[i]);
		if (!sel[i]) {
			efa_free_domain_names(sel);
			ret = -FI_ENOMEM;
			goto out;
		}
	}

	for (i = 0; i < want_n; i++)
		FT_INFO("domain %d: %s", i, sel[i]);

	*names = sel;
	ret = FI_SUCCESS;
out:
	ft_free_string_array(want);
	efa_free_domain_names(avail);
	return ret;
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
