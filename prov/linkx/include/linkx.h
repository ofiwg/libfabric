/*
 * Copyright (c) 2022 ORNL. All rights reserved.
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

#ifndef LINKX_H
#define LINKX_H

#define LNX_MAX_LOCAL_EPS 16

struct local_prov_ep {
	bool lpe_local;
	char lpe_fabric_name[FI_NAME_MAX];
	struct fid_fabric *lpe_fabric;
	struct fid_domain *lpe_domain;
	struct fid_ep *lpe_ep;
	struct fid_av *lpe_av;
	struct fi_info *lpe_fi_info;
};

struct local_prov {
	struct dlist_entry lpv_entry;
	char lpv_prov_name[FI_NAME_MAX];
	int lpv_ep_count;
	struct local_prov_ep *lpv_prov_eps[LNX_MAX_LOCAL_EPS];
};

struct lnx_ep {
	struct util_ep le_ep;
	struct util_domain *le_domain;
	size_t le_fclass;
	/* TODO - add the shared queues here */
};

struct lnx_peer_entry {
	int lp_addr_count;
	int lp_ep_count;
	/* pointer to the local endpoint information to be used for
	 * communication with this peer.
	 *
	 * If the peer is on-node, then lp_endpoints[0] = shm &&
	 * lp_addr_count = 1
	 *
	 * if peer is off-node, then there could be up to LNX_MAX_LOCAL_EPS
	 * endpoints we can use to reach that peer.
	 */
	struct local_prov_entry *lp_endpoints[LNX_MAX_LOCAL_EPS];

	/* All addresses which we can reach that peer on */
	fi_addr_t lp_fi_addrs[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer_table {
	struct fid_av lpt_av_fid;
	size_t lpt_size;
	int lpt_count;
	struct util_domain *lpt_domain;
	/* an array of peer entries */
	struct lnx_peer_entry *lpt_entries;
};

extern struct dlist_entry local_prov_table;
extern struct util_prov lnx_util_prov;
extern struct fi_provider lnx_prov;

int lnx_getinfo(uint32_t version, const char *node, const char *service,
				uint64_t flags, const struct fi_info *hints,
				struct fi_info **info);

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

void lnx_fini(void);

int lnx_fabric_close(struct fid *fid);

int lnx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
					struct fid_domain **dom, void *context);

int lnx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
				struct fid_av **av, void *context);

static inline
int lnx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
				struct fid_cq **cq, void *context)
{
	return -FI_EOPNOTSUPP;
}

int lnx_endpoint(struct fid_domain *domain, struct fi_info *info,
				 struct fid_ep **ep, void *context);

int lnx_scalable_ep(struct fid_domain *domain, struct fi_info *info,
					struct fid_ep **ep, void *context);

#endif /* LINKX_H */
