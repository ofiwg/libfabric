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

/*
 * . Each endpoint LINKx manages will have an instance of this structure.
 * . The structure has a pointer to the general shared cq.
 *     . All events are written into that CQ.
 * . The structure has a pointer to the core cq which the core provider
 *   returns
 * . The structure has an instance of the peer_cq which is unique for
 *   handling communication with the constituent endpoint.
 */
struct lnx_peer_cq {
	struct util_cq *lpc_shared_cq;
	struct fid_peer_cq lpc_cq;
	struct fid_cq *lpc_core_cq;
};

struct local_prov_ep {
	bool lpe_local;
	char lpe_fabric_name[FI_NAME_MAX];
	struct fid_fabric *lpe_fabric;
	struct fid_domain *lpe_domain;
	struct fid_ep *lpe_ep;
	struct fid_av *lpe_av;
	struct lnx_peer_cq lpe_cq;
	struct fi_info *lpe_fi_info;
};

struct local_prov {
	struct dlist_entry lpv_entry;
	char lpv_prov_name[FI_NAME_MAX];
	int lpv_ep_count;
	struct local_prov_ep *lpv_prov_eps[LNX_MAX_LOCAL_EPS];
};

struct lnx_address_prov {
	char lap_prov[FI_NAME_MAX];
	/* an array of addresses of size count. */
	/* entry 0 is shm if available */
	/* array can't be larger than LNX_MAX_LOCAL_EPS */
	int lap_addr_count;
	/* size as specified by the provider */
	int lap_addr_size;
	/* payload */
	char lap_addrs[];
};

struct lnx_addresses {
	/* used to determine if the address is node local or node remote */
	char la_hostname[FI_NAME_MAX];
	/* number of providers <= LNX_MAX_LOCAL_EPS */
	int la_prov_count;
	struct lnx_address_prov la_addr_prov[];
};

struct lnx_local2peer_map {
	struct local_prov_ep *local_ep;
	int addr_count;
	fi_addr_t peer_addrs[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer_prov {
	/* provider name */
	char lpp_prov_name[FI_NAME_MAX];

	uint64_t lpp_flags;

	/* pointer to the local endpoint information to be used for
	 * communication with this peer.
	 *
	 * If the peer is on-node, then lp_endpoints[0] = shm
	 *
	 * if peer is off-node, then there could be up to LNX_MAX_LOCAL_EPS
	 * local endpoints we can use to reach that peer.
	 */
	struct local_prov *lpp_prov;

	/* each peer can be reached from any of the local provider endpoints
	 * on any of the addresses which are given to us. It's an N:N
	 * relationship
	 */
	struct lnx_local2peer_map *lpp_map[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer {
	/* true if peer can be reached over shared memory, false otherwise */
	bool lp_local;

	/* Each provider that we can reach the peer on will have an entry
	 * below. Each entry will contain all the local provider endpoints we
	 * can reach the peer on, as well as all the peer addresses on that
	 * provider.
	 *
	 * We can potentially multi-rail between the interfaces on the same
	 * provider, both local and remote.
	 *
	 * Or we can multi-rail across different providers. Although this
	 * might be more complicated due to the differences in provider
	 * capabilities.
	 */
	struct lnx_peer_prov *lp_provs[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer_table {
	struct util_av lpt_av;
	int lpt_max_count;
	int lpt_count;
	struct util_domain *lpt_domain;
	/* an array of peer entries */
	struct lnx_peer **lpt_entries;
};

struct lnx_ep {
	struct util_ep le_ep;
	struct util_domain *le_domain;
	size_t le_fclass;
	struct lnx_peer_table *le_peer_tbl;
	/* TODO - add the shared queues here */
};

struct lnx_mem_desc {
	struct fid_mr *core_mr[LNX_MAX_LOCAL_EPS];
	struct local_prov_ep *ep[LNX_MAX_LOCAL_EPS];
	fi_addr_t peer_addr[LNX_MAX_LOCAL_EPS];
};

extern struct dlist_entry local_prov_table;
extern struct util_prov lnx_util_prov;
extern struct fi_provider lnx_prov;
extern struct local_prov *shm_prov;
extern struct lnx_peer_table *lnx_peer_tbl;

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

int lnx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context);

int lnx_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);

int lnx_scalable_ep(struct fid_domain *domain, struct fi_info *info,
		    struct fid_ep **ep, void *context);

int lnx_cq2ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags);

static inline struct lnx_peer *
lnx_get_peer(struct lnx_peer **peers, fi_addr_t addr)
{
	if (!peers || addr == FI_ADDR_UNSPEC)
		return NULL;

	return peers[addr];
}

static inline
int lnx_select_send_pathway(struct lnx_peer *lp, struct lnx_mem_desc *desc,
			    struct local_prov_ep **cep, fi_addr_t *addr,
			    void **mem_desc)
{
	int idx = 0;
	struct lnx_local2peer_map *lpm;

	/* TODO this will need to be expanded to handle Multi-Rail. For now
	 * the assumption is that local peers can be reached on shm and remote
	 * peers have only one interface, hence indexing on 0 and 1
	 *
	 * If we did memory registration, then we've already figured out the
	 * pathway
	 */
	if (desc && desc->core_mr[0]) {
		*cep = desc->ep[0];
		*addr = desc->peer_addr[0];
		if (mem_desc)
			*mem_desc = desc->core_mr[0]->mem_desc;
		return 0;
	}

	if (!lp->lp_local)
		idx = 1;

	/* TODO when we support multi-rail we can have multiple maps */
	lpm = lp->lp_provs[idx]->lpp_map[0];

	*cep = lpm->local_ep;
	*addr = lpm->peer_addrs[0];
	if (mem_desc)
		*mem_desc = NULL;

	return 0;
}

static inline
int lnx_select_recv_pathway(struct lnx_peer *lp, struct lnx_mem_desc *desc,
			    struct local_prov_ep **cep, fi_addr_t *addr,
			    void **mem_desc)
{
	/* TODO for now keeping two different functions. The receive case will
	 * need to handle FI_ADDR_UNSPEC
	 */
	if (!lp)
		return -FI_ENOSYS;

	return lnx_select_send_pathway(lp, desc, cep, addr, mem_desc);
}

#endif /* LINKX_H */