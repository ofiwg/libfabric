/*
 * Copyright (c) 2022 ORNL. All rights reserved.
 * Copyright (c) Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef LNX_H
#define LNX_H

#include "ofi_lock.h"
#include "ofi_util.h"

#define LNX_MAX_LOCAL_EPS 	16
#define LNX_IOV_LIMIT 		4
#define LNX_SUB_ID_BITS 	8
#define LNX_MAX_SUB_ID 		((1ULL << LNX_SUB_ID_BITS) - 1)

#define lnx_ep_rx_flags(lnx_ep) ((lnx_ep)->le_ep.rx_op_flags)

struct lnx_match_attr {
	fi_addr_t lm_addr;
	uint64_t lm_tag;
	uint64_t lm_ignore;
};

struct lnx_queue {
	struct dlist_entry lq_queue;
	dlist_func_t *lq_match_func;
	ofi_spin_t lq_qlock;
	uint64_t lq_max;
	uint64_t lq_size;
	uint64_t lq_count;
	uint64_t lq_rolling_avg;
	uint64_t lq_rolling_sum;
};

struct lnx_qpair {
	struct lnx_queue lqp_recvq;
	struct lnx_queue lqp_unexq;
};

struct lnx_peer_srq {
	struct lnx_qpair lps_trecv;
	struct lnx_qpair lps_recv;
};

struct lnx_core_fabric {
	struct fi_info *cf_info;
	struct fid_fabric *cf_fabric;
};

struct lnx_core_domain {
	struct fid_domain *cd_domain;
	struct lnx_core_fabric *cd_fabric;
	struct fi_info *cd_info;
	uint64_t cd_num_sends;
};

struct lnx_core_av {
	/* on the lnx_av list */
	struct dlist_entry cav_peer_entry;
	struct fid_av *cav_av;
	struct lnx_core_domain *cav_domain;
	struct ofi_bufpool *cav_map;
	struct dlist_entry cav_endpoints;
};

struct lnx_t_traffic_stats {
	uint64_t st_num_tsend;
	uint64_t st_num_tsendv;
	uint64_t st_num_tsendmsg;
	uint64_t st_num_tsenddata;
	uint64_t st_num_tinject;
	uint64_t st_num_tinjectdata;
	uint64_t st_num_posted_recvs;
	uint64_t st_num_unexp_msgs;
};

struct lnx_core_ep {
	struct dlist_entry cep_av_entry;
	struct fid_peer_srx cep_srx;
	struct fid_ep *cep_ep;
	struct fid_ep *cep_srx_ep;
	struct lnx_core_domain *cep_domain;
	struct lnx_core_av *cep_cav;
	struct lnx_ep *cep_parent;
	struct lnx_t_traffic_stats cep_t_stats;
};

struct lnx_core_cq {
	struct fid_cq *cc_cq;
	struct lnx_core_domain *cc_domain;
};

struct lnx_peer_map {
	int map_count;
	ofi_atomic32_t map_rr;
	fi_addr_t map_addrs[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer_ep_map {
	struct lnx_core_ep **pem_eps;
	int pem_num_eps;
};

struct lnx_peer {
	fi_addr_t lp_addr;
	int lp_ep_count;
	int lp_av_count;
	ofi_atomic32_t lp_ep_rr;
	struct lnx_core_av **lp_avs;
	struct lnx_peer_ep_map *lp_src_eps;
	/* Both fields below are used to lock the peer to a specific local
	 * endpoint and remote peer address
	 */
	fi_addr_t lp_locked_core_addr;
	struct lnx_core_ep *lp_locked_cep;
};

struct lnx_av {
	struct util_av lav_av;
	int lav_max_count;
	struct lnx_domain *lav_domain;
	struct lnx_core_av *lav_core_avs;
};

struct lnx_mr {
	struct ofi_mr lm_mr;
	struct fi_mr_attr lm_attr;
	struct fid_mr *lm_core_mr;
	struct iovec lm_iov[LNX_IOV_LIMIT];
};

struct lnx_domain {
	struct util_domain ld_domain;
	struct ofi_bufpool *ld_mem_reg_bp;
	struct lnx_core_domain *ld_core_domains;
	size_t ld_iov_limit;
	int ld_num_doms;
	int ld_ep_idx;
};

enum lnx_multirail_selection {
	LNX_MR_SELECTION_PER_MSG = 0,
	LNX_MR_SELECTION_PER_PEER,
	LNX_MR_SELECTION_MAX,
};

struct lnx_ep {
	int le_idx;
	struct util_ep le_ep;
	struct lnx_core_ep *le_core_eps;
	struct ofi_bufpool *le_recv_bp;
	ofi_spin_t le_bplock;
	struct lnx_domain *le_domain;
	size_t le_fclass;
	struct lnx_peer_srq le_srq;
	struct lnx_av *le_lav;
	/* global round robin index */
	ofi_atomic32_t le_rr;
	enum lnx_multirail_selection le_mr;
};

struct lnx_cq {
	struct util_cq lcq_util_cq;
	struct lnx_core_cq *lcq_core_cqs;
	struct lnx_domain *lcq_lnx_domain;
};

struct lnx_fabric {
	struct util_fabric lf_util_fabric;
	struct lnx_core_fabric *lf_core_fabrics;
	bool lf_fab_setup_complete;
	int lf_num_fabs;
};

struct lnx_ep_addr {
	char lea_prov[FI_NAME_MAX];
	size_t lea_addr_size;
	char lea_addr[];
};

struct lnx_address {
	char la_hostname[FI_NAME_MAX];
	int la_ep_count;
	struct lnx_ep_addr la_addrs[];
};

struct lnx_rx_entry {
	struct dlist_entry entry;
	struct fi_peer_rx_entry rx_entry;
	struct iovec rx_iov[LNX_IOV_LIMIT];
	void *rx_desc[LNX_IOV_LIMIT];
	struct lnx_ep *rx_lep;
	struct lnx_core_ep *rx_cep;
	uint64_t rx_ignore;
	bool rx_global;
};

extern struct fi_ops_msg lnx_msg_ops;
extern struct fi_ops_tagged lnx_tagged_ops;
extern struct fi_ops_rma lnx_rma_ops;
extern struct fi_ops_atomic lnx_atomic_ops;
extern struct fi_ops_srx_owner lnx_srx_ops;

extern struct util_prov lnx_util_prov;
extern struct fi_provider lnx_prov;
extern struct ofi_bufpool *global_recv_bp;
extern ofi_spin_t global_bplock;

int lnx_setup_fabrics(char *name, struct lnx_fabric *lnx_fab, void *context);

int lnx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **dom, void *context);

int lnx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

struct lnx_peer *lnx_av_lookup_addr(struct lnx_av *av, fi_addr_t addr);

int lnx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context);

int lnx_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);

int lnx_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		   uint64_t flags, struct fid_mr **mr_fid);
int lnx_mr_regattr_core(struct lnx_core_domain *cd, void *desc,
			void **core_desc);

int lnx_process_recv(struct lnx_ep *lep, const struct iovec *iov, void *desc,
		     fi_addr_t addr, size_t count, uint64_t tag,
		     uint64_t ignore, void *context, uint64_t flags,
		     bool tagged);

static inline fi_addr_t lnx_encode_fi_addr(uint64_t primary_id, uint8_t sub_id)
{
	return (primary_id << 8) | sub_id;
}

static inline fi_addr_t lnx_decode_primary_id(uint64_t fi_addr)
{
	return (fi_addr == FI_ADDR_UNSPEC) ? fi_addr :
		fi_addr >> LNX_SUB_ID_BITS;
}

static inline uint8_t lnx_decode_sub_id(uint64_t fi_addr)
{
	return (fi_addr == FI_ADDR_UNSPEC) ? fi_addr : fi_addr & LNX_MAX_SUB_ID;
}

static inline fi_addr_t lnx_get_core_addr(struct lnx_core_ep *cep,
					  fi_addr_t addr)
{
	struct lnx_peer_map *map_addr;
	uint8_t idx = lnx_decode_sub_id(addr);
	fi_addr_t primary = lnx_decode_primary_id(addr);

	if (addr == FI_ADDR_UNSPEC)
		return addr;

	map_addr = ofi_bufpool_get_ibuf(cep->cep_cav->cav_map, primary);
	return map_addr->map_addrs[idx];
}

typedef int (*lnx_select_tx_ep)(
		struct lnx_ep *lep, fi_addr_t lnx_addr,
		struct lnx_core_ep **cep_out, fi_addr_t *core_addr);
extern lnx_select_tx_ep lnx_select_send_endpoints[LNX_MR_SELECTION_MAX];

typedef void (*lnx_set_tx_pair)(
		struct lnx_ep *lep, struct lnx_core_ep *cep, fi_addr_t addr);
extern lnx_set_tx_pair lnx_set_send_pair[LNX_MR_SELECTION_MAX];

#endif /* LNX_H */
