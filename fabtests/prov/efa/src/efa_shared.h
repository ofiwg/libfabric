/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef _EFA_SHARED_H
#define _EFA_SHARED_H

#include <getopt.h>
#include <rdma/fabric.h>

#define EFA_FABRIC_NAME	       "efa"
#define EFA_DIRECT_FABRIC_NAME "efa-direct"

#define EFA_INFO_TYPE_IS_RDM(_info)                                        \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && \
	 !strcasecmp(_info->fabric_attr->name, EFA_FABRIC_NAME))

#define EFA_INFO_TYPE_IS_DIRECT(_info)                                     \
	(_info && _info->ep_attr && (_info->ep_attr->type == FI_EP_RDM) && \
	 !strcasecmp(_info->fabric_attr->name, EFA_DIRECT_FABRIC_NAME))

enum {
	OPT_HIGH_PPS = 256,
	OPT_POST_LIST,
	OPT_NUM_EPS,
	OPT_SL_LOW_LATENCY,
	OPT_EPS_PER_DOMAIN,
	OPT_DOMAINS,
};

/*
 * Merged long options table combining the EFA-specific options
 * (e.g. --high-pps) with the shared fabtests long_opts (e.g. --no-rx-cq-data)
 */
extern struct option *efa_long_opts;

void build_efa_long_opts(void);

void efa_longopts_usage(void);

int efa_calc_peer_distribution(int my_idx, int my_count, int peer_count,
			       int *num_peers, int **peer_ids);

int efa_exchange_addrs_oob(int oob_sock, bool is_initiator,
			   struct fid_ep **local_eps, int local_n,
			   struct fid_av *av, fi_addr_t *remote_addrs,
			   int remote_n);

int efa_exchange_raw_addrs_oob(int oob_sock, bool is_initiator,
			       struct fid_ep **local_eps, int local_n,
			       char *remote_buf, int remote_n);

int efa_insert_raw_addrs(struct fid_av *av, const char *remote_buf,
			 int remote_n, fi_addr_t *remote_addrs);

int efa_select_domains(struct fi_info *hints, int want_n,
		       const char *explicit_csv, char ***names);

void efa_free_domain_names(char **names);

#endif /* _EFA_SHARED_H */
