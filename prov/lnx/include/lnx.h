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

#ifndef LNX_H
#define LNX_H

#define LNX_DEF_AV_SIZE 1024
#define LNX_MAX_LOCAL_EPS 16
#define LNX_IOV_LIMIT 4

#define lnx_ep_rx_flags(lnx_ep) ((lnx_ep)->le_ep.rx_op_flags)

struct local_prov_ep;

struct lnx_match_attr {
	fi_addr_t lm_addr;
	uint64_t lm_tag;
	uint64_t lm_ignore;
	struct lnx_peer *lm_peer;
	struct local_prov_ep *lm_cep;
};

struct lnx_peer_cq {
	struct lnx_cq *lpc_shared_cq;
	struct fid_peer_cq lpc_cq;
	struct fid_cq *lpc_core_cq;
};

struct lnx_queue {
	struct dlist_entry lq_queue;
	dlist_func_t *lq_match_func;
	ofi_spin_t lq_qlock;
};

struct lnx_qpair {
	struct lnx_queue lqp_recvq;
	struct lnx_queue lqp_unexq;
};

struct lnx_peer_srq {
	struct lnx_qpair lps_trecv;
	struct lnx_qpair lps_recv;
};

struct local_prov_ep {
	struct dlist_entry entry;
	bool lpe_local;
	char lpe_fabric_name[FI_NAME_MAX];
	struct fid_fabric *lpe_fabric;
	struct fid_domain *lpe_domain;
	struct fid_ep *lpe_ep;
	struct fid_ep **lpe_txc;
	struct fid_ep **lpe_rxc;
	struct fid_av *lpe_av;
	struct fid_ep *lpe_srx_ep;
	struct lnx_peer_cq lpe_cq;
	struct fi_info *lpe_fi_info;
	struct fid_peer_srx lpe_srx;
	struct ofi_bufpool *lpe_recv_bp;
	ofi_spin_t lpe_bplock;
	struct local_prov *lpe_parent;
};

struct lnx_rx_entry {
	/* the entry which will be passed to the core provider */
	struct fi_peer_rx_entry rx_entry;
	/* iovec to use to point to receive buffers */
	struct iovec rx_iov[LNX_IOV_LIMIT];
	/* desc array to be used to point to the descs passed by the user */
	void *rx_desc[LNX_IOV_LIMIT];
	/* peer we expect messages from.
	 * This is available if the receive request provided a source address.
	 * Otherwise it will be NULL
	 */
	struct lnx_peer *rx_peer;
	/* local prov endpoint receiving the message if this entry is
	 * added to the SUQ
	 */
	struct local_prov_ep *rx_cep;
	/* match information which will be given to us by the core provider */
	struct fi_peer_match_attr rx_match_info;
	/* ignore bit passed in by the user */
	uint64_t rx_ignore;
	/* which pool this rx_entry came from. It's either from the global
	 * pool or some core provider pool
	 */
	bool rx_global;
};

OFI_DECLARE_FREESTACK(struct lnx_rx_entry, lnx_recv_fs);

struct local_prov {
	struct dlist_entry lpv_entry;
	char lpv_prov_name[FI_NAME_MAX];
	int lpv_ep_count;
	struct dlist_entry lpv_prov_eps;
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
	struct dlist_entry entry;
	struct local_prov_ep *local_ep;
	int addr_count;
	fi_addr_t peer_addrs[LNX_MAX_LOCAL_EPS];
};

struct lnx_peer_prov {
	struct dlist_entry entry;

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
	struct dlist_entry lpp_map;
};

struct lnx_peer {
	/* true if peer can be reached over shared memory, false otherwise */
	bool lp_local;

	/* Each provider that we can reach the peer on will have an entry
	 * below. Each entry will contain all the local provider endpoints we
	 * can reach the peer through, as well as all the peer addresses on that
	 * provider.
	 *
	 * We can potentially multi-rail between the interfaces on the same
	 * provider, both local and remote.
	 *
	 * Or we can multi-rail across different providers. Although this
	 * might be more complicated due to the differences in provider
	 * capabilities.
	 */
	struct lnx_peer_prov *lp_shm_prov;
	struct dlist_entry lp_provs;
};

struct lnx_peer_table {
	struct util_av lpt_av;
	int lpt_max_count;
	int lpt_count;
	struct lnx_domain *lpt_domain;
	/* an array of peer entries */
	struct lnx_peer **lpt_entries;
};

struct lnx_ctx {
	struct dlist_entry ctx_head;
	int ctx_idx;
	struct lnx_ep *ctx_parent;
	struct fid_ep ctx_ep;
};

struct lnx_ep {
	struct util_ep le_ep;
	struct dlist_entry le_tx_ctx;
	struct dlist_entry le_rx_ctx;
	struct lnx_domain *le_domain;
	size_t le_fclass;
	struct lnx_peer_table *le_peer_tbl;
	struct lnx_peer_srq le_srq;
};

struct lnx_srx_context {
	struct lnx_ep *srx_lep;
	struct local_prov_ep *srx_cep;
};

struct lnx_mem_desc_prov {
	struct local_prov *prov;
	struct fid_mr *core_mr;
};

struct lnx_mem_desc {
	struct lnx_mem_desc_prov desc[LNX_MAX_LOCAL_EPS];
	int desc_count;
};

struct lnx_mr {
	struct ofi_mr mr;
	struct lnx_mem_desc desc;
};

struct lnx_domain {
	struct util_domain ld_domain;
	struct lnx_fabric *ld_fabric;
	bool ld_srx_supported;
	struct ofi_mr_cache ld_mr_cache;
};

struct lnx_cq {
	struct util_cq util_cq;
	struct lnx_domain *lnx_domain;
};

struct lnx_fabric {
	struct util_fabric	util_fabric;
	/* providers linked by this fabric */
	struct dlist_entry local_prov_table;
	/* memory registration buffer pool */
	struct ofi_bufpool *mem_reg_bp;
	/* shared memory provider used in this link */
	struct local_prov *shm_prov;
	/* peers associated with this link */
	struct lnx_peer_table *lnx_peer_tbl;
};

extern struct util_prov lnx_util_prov;
extern struct fi_provider lnx_prov;
extern struct ofi_bufpool *global_recv_bp;
extern ofi_spin_t global_bplock;

struct fi_info *lnx_get_link_by_dom(char *domain_name);

int lnx_getinfo(uint32_t version, const char *node, const char *service,
				uint64_t flags, const struct fi_info *hints,
				struct fi_info **info);

int lnx_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);
int lnx_setup_core_fabrics(char *name, struct lnx_fabric *lnx_fab,
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

int lnx_get_msg(struct fid_peer_srx *srx, struct fi_peer_match_attr *match,
		struct fi_peer_rx_entry **entry);
int lnx_get_tag(struct fid_peer_srx *srx, struct fi_peer_match_attr *match,
		struct fi_peer_rx_entry **entry);
int lnx_queue_msg(struct fi_peer_rx_entry *entry);
int lnx_queue_tag(struct fi_peer_rx_entry *entry);
void lnx_free_entry(struct fi_peer_rx_entry *entry);
void lnx_foreach_unspec_addr(struct fid_peer_srx *srx,
	fi_addr_t (*get_addr)(struct fi_peer_rx_entry *));

static inline struct lnx_peer *
lnx_get_peer(struct lnx_peer **peers, fi_addr_t addr)
{
	if (!peers || addr == FI_ADDR_UNSPEC)
		return NULL;

	return peers[addr];
}

static inline
void lnx_get_core_desc(struct lnx_mem_desc *desc, void **mem_desc)
{
	if (desc && desc->desc[0].core_mr) {
		if (mem_desc)
			*mem_desc = desc->desc[0].core_mr->mem_desc;
		return;
	}

	*mem_desc = NULL;
}

static inline
int lnx_create_mr(const struct iovec *iov, fi_addr_t addr,
		  struct lnx_domain *lnx_dom, struct ofi_mr_entry **mre)
{
	struct ofi_mr *mr;
	struct fi_mr_attr attr = {};
	struct fi_mr_attr cur_abi_attr;
	struct ofi_mr_info info = {};
	uint64_t flags = 0;
	int rc;

	attr.iov_count = 1;
	attr.mr_iov = iov;
	*mre = ofi_mr_cache_find(&lnx_dom->ld_mr_cache, &attr, 0);
	if (*mre) {
		mr = (struct ofi_mr *)(*mre)->data;
		goto out;
	}

	attr.iface = ofi_get_hmem_iface(iov->iov_base,
			&attr.device.reserved, &flags);
	info.iov = *iov;
	info.iface = attr.iface;
	rc = ofi_hmem_dev_register(attr.iface, iov->iov_base, iov->iov_len,
				   (uint64_t *) &attr.hmem_data);
	if (rc)
		return rc;

	rc = ofi_mr_cache_search(&lnx_dom->ld_mr_cache, &info, mre);
	if (rc) {
		ofi_hmem_dev_unregister(attr.iface, (uint64_t)attr.hmem_data);
		return rc;
	}

	mr = (struct ofi_mr *)(*mre)->data;
	ofi_mr_update_attr(lnx_dom->ld_domain.fabric->fabric_fid.api_version,
			   lnx_dom->ld_domain.info_domain_caps, &attr, &cur_abi_attr, 0);

	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = attr.context;
	mr->domain = &lnx_dom->ld_domain;
	mr->flags = flags;
	mr->iface = cur_abi_attr.iface;
	mr->device = cur_abi_attr.device.reserved;
	mr->hmem_data = cur_abi_attr.hmem_data;
	mr->mr_fid.mem_desc = (void*) mr;

out:
	return FI_SUCCESS;
}

static inline
int lnx_select_send_pathway(struct lnx_peer *lp, struct lnx_domain *lnx_dom,
			    struct lnx_mem_desc *desc, struct local_prov_ep **cep,
			    fi_addr_t *addr, const struct iovec *iov, size_t iov_count,
			    struct ofi_mr_entry **mre, void **mem_desc, uint64_t *rkey)
{
	int idx = 0;
	int rc;
	struct lnx_peer_prov *prov;
	struct lnx_local2peer_map *lpm;
	struct ofi_mr *mr = NULL;

	if (lp->lp_local) {
		prov = lp->lp_shm_prov;
	} else {
		prov = dlist_first_entry_or_null(
			&lp->lp_provs, struct lnx_peer_prov, entry);
		idx = 1;
	}

	/* TODO when we support multi-rail we can have multiple maps */
	lpm = dlist_first_entry_or_null(&prov->lpp_map,
					struct lnx_local2peer_map, entry);
	*addr = lpm->peer_addrs[0];

	/* TODO this will need to be expanded to handle Multi-Rail. For now
	 * the assumption is that local peers can be reached on shm and remote
	 * peers have only one interface, hence indexing on 0 and 1
	 *
	 * If we did memory registration, then we've already figured out the
	 * pathway
	 */
	if (desc && desc->desc[idx].core_mr) {
		*cep = dlist_first_entry_or_null(
				&desc->desc[idx].prov->lpv_prov_eps,
				struct local_prov_ep, entry);
		if (mem_desc)
			*mem_desc = fi_mr_desc(desc->desc[idx].core_mr);
		if (rkey)
			*rkey = fi_mr_key(desc->desc[idx].core_mr);
		return 0;
	}

	*cep = lpm->local_ep;
	if (mem_desc)
		*mem_desc = NULL;

	if (!lp->lp_local || !mem_desc || (mem_desc && *mem_desc) ||
	    !iov || (iov && iov->iov_base == NULL))
		return 0;

	/* Look up the address in the cache:
	 *  - if it's found then use the cached fid_mr
	 *     - This will include the iface, which is really all we need
	 *  - if it's not then lookup the iface, create the fid_mr and
	 *    cache it.
	 */
	rc = lnx_create_mr(iov, *addr, lnx_dom, mre);
	if (!rc && mre) {
		mr = (struct ofi_mr *)(*mre)->data;
		*mem_desc = mr->mr_fid.mem_desc;
	}

	return rc;
}

static inline
int lnx_select_recv_pathway(struct lnx_peer *lp, struct lnx_domain *lnx_dom,
			    struct lnx_mem_desc *desc, struct local_prov_ep **cep,
			    fi_addr_t *addr, const struct iovec *iov, size_t iov_count,
			    struct ofi_mr_entry **mre, void **mem_desc)
{
	/* if the src address is FI_ADDR_UNSPEC, then we'll need to trigger
	 * all core providers to listen for a receive, since we don't know
	 * which one will endup getting the message.
	 *
	 * For each core provider we're tracking, trigger the recv operation
	 * on it.
	 *
	 * if the src address is specified then we just need to select and
	 * exact core endpoint to trigger the recv on.
	 */
	if (!lp)
		return -FI_ENOSYS;

	return lnx_select_send_pathway(lp, lnx_dom, desc, cep, addr, iov,
				       iov_count, mre, mem_desc, NULL);
}

#endif /* LNX_H */
