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

#include "config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>

#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "ofi_str.h"
#include "ofi_prov.h"
#include "ofi_perf.h"
#include "ofi_hmem.h"
#include "rdma/fi_ext.h"
#include "lnx.h"

extern struct fi_ops_cm lnx_cm_ops;
extern struct fi_ops_msg lnx_msg_ops;
extern struct fi_ops_tagged lnx_tagged_ops;
extern struct fi_ops_rma lnx_rma_ops;
extern struct fi_ops_atomic lnx_atomic_ops;

static void lnx_init_ctx(struct fid_ep *ctx, size_t fclass);

static int lnx_close_ceps(struct local_prov *prov)
{
	int rc, frc = 0;
	struct local_prov_ep *ep;

	dlist_foreach_container(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry) {

		if (ep->lpe_srx.ep_fid.fid.context)
			free(ep->lpe_srx.ep_fid.fid.context);

		rc = fi_close(&ep->lpe_ep->fid);
		if (rc)
			frc = rc;
		ofi_bufpool_destroy(ep->lpe_recv_bp);
	}

	return frc;
}

int lnx_ep_close(struct fid *fid)
{
	int rc = 0;
	struct local_prov *entry;
	struct lnx_ep *ep;
	struct lnx_fabric *fabric;

	ep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	fabric = ep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table,
				struct local_prov,
				entry, lpv_entry) {
		lnx_close_ceps(entry);
		if (rc)
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Failed to close endpoint for %s\n",
				entry->lpv_prov_name);
	}

	ofi_endpoint_close(&ep->le_ep);
	free(ep);

	return rc;
}

static int lnx_enable_core_eps(struct lnx_ep *lep)
{
	int rc;
	struct local_prov *entry;
	struct local_prov_ep *ep;
	int srq_support = 1;
	struct lnx_fabric *fabric = lep->le_domain->ld_fabric;

	fi_param_get_bool(&lnx_prov, "use_srq", &srq_support);

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			if (srq_support) {
				rc = fi_ep_bind(ep->lpe_ep,
						&ep->lpe_srx_ep->fid, 0);
				if (rc) {
					FI_INFO(&lnx_prov, FI_LOG_CORE,
						"%s doesn't support SRX (%d)\n",
						ep->lpe_fabric_name, rc);
					return rc;
				}
			}

			rc = fi_enable(ep->lpe_ep);
			if (rc)
				return rc;
		}
	}

	return 0;
}

static int lnx_ep_control(struct fid *fid, int command, void *arg)
{
	struct lnx_ep *ep;
	int rc;

	ep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);

	switch (command) {
	case FI_ENABLE:
		if (ep->le_fclass == FI_CLASS_EP &&
		    ((ofi_needs_rx(ep->le_ep.caps) && !ep->le_ep.rx_cq) ||
		    (ofi_needs_tx(ep->le_ep.caps) && !ep->le_ep.tx_cq)))
			return -FI_ENOCQ;
		if (!ep->le_peer_tbl)
			return -FI_ENOAV;
		rc = lnx_enable_core_eps(ep);
		break;
	default:
		return -FI_ENOSYS;
	}

	return rc;
}

int lnx_cq_bind_core_prov(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct util_cq *cq;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	lep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	cq = container_of(bfid, struct util_cq, cq_fid.fid);
	fabric = lep->le_domain->ld_fabric;

	rc = ofi_ep_bind_cq(&lep->le_ep, cq, flags);
	if (rc)
		return rc;

	/* bind the core providers to their respective CQs */
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			rc = fi_ep_bind(ep->lpe_ep,
					&ep->lpe_cq.lpc_core_cq->fid, flags);
			if (rc)
				return rc;
		}
	}

	return 0;
}

static int lnx_ep_bind_core_prov(struct lnx_fabric *fabric, uint64_t flags)
{
	struct local_prov *entry;
	struct local_prov_ep *ep;
	int rc;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			rc = fi_ep_bind(ep->lpe_ep, &ep->lpe_av->fid, flags);
			if (rc)
				return rc;
		}
	}

	return rc;
}

static int
lnx_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int rc = 0;
	struct lnx_ep *ep;
	struct lnx_peer_table *peer_tbl;

	switch (fid->fclass) {
	case FI_CLASS_EP:	/* Standard EP */
	case FI_CLASS_SEP:	/* Scalable EP */
		ep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
		break;

	default:
		return -FI_EINVAL;
	}

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		return -FI_ENOSYS;

	case FI_CLASS_CQ:
		rc = lnx_cq_bind_core_prov(fid, bfid, flags);
		break;

	case FI_CLASS_CNTR:
		return -FI_ENOSYS;

	case FI_CLASS_AV:
		peer_tbl = container_of(bfid, struct lnx_peer_table,
					lpt_av.av_fid.fid);
		if (peer_tbl->lpt_domain != ep->le_domain)
			return -FI_EINVAL;
		ep->le_peer_tbl = peer_tbl;
		/* forward the bind to the core provider endpoints */
		rc = lnx_ep_bind_core_prov(ep->le_domain->ld_fabric, flags);
		break;

	case FI_CLASS_STX_CTX:	/* shared TX context */
		return -FI_ENOSYS;

	case FI_CLASS_SRX_CTX:	/* shared RX context */
		return -FI_ENOSYS;

	default:
		return -FI_EINVAL;
	}

	return rc;
}

int lnx_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct local_prov *entry;
	size_t size = sizeof(struct lnx_addresses);
	/* initial location to put the address */
	char ep_addr[FI_NAME_MAX];
	char *tmp = NULL;
	struct lnx_addresses *la;
	struct lnx_address_prov *lap;
	char hostname[FI_NAME_MAX];
	size_t prov_addrlen;
	size_t addrlen_list[LNX_MAX_LOCAL_EPS];
	int rc, j = 0;
	struct lnx_ep *lnx_ep;
	struct lnx_fabric *fabric;
	struct local_prov_ep *ep;

	lnx_ep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	fabric = lnx_ep->le_domain->ld_fabric;

	/* check the hostname and compare it to mine
	 * TODO: Is this good enough? or do we need a better way of
	 * determining if the address is local?
	 */
	rc = gethostname(hostname, FI_NAME_MAX);
	if (rc == -1) {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "failed to get hostname\n");
		return -FI_EPERM;
	}

	addrlen_list[0] = 0;

	/* calculate the size of the address */
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		size += sizeof(struct lnx_address_prov);
		prov_addrlen = 0;

		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			rc = fi_getname(&ep->lpe_ep->fid, (void*)ep_addr, &prov_addrlen);
			if (rc == -FI_ETOOSMALL) {
				size += prov_addrlen * entry->lpv_ep_count;
				addrlen_list[j] = prov_addrlen;
				j++;
				break;
			} else {
				return -FI_EINVAL;
			}
		}
	}

	if (!addr || *addrlen < size) {
		*addrlen = size;
		return -FI_ETOOSMALL;
	}

	la = addr;

	lap = (struct lnx_address_prov *)((char*)la + sizeof(*la));

	j = 0;
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
							entry, lpv_entry) {
		memcpy(lap->lap_prov, entry->lpv_prov_name, FI_NAME_MAX - 1);
		lap->lap_addr_count = entry->lpv_ep_count;
		lap->lap_addr_size = addrlen_list[j];

		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			tmp = (char*)lap + sizeof(*lap);

			rc = fi_getname(&ep->lpe_ep->fid, (void*)tmp, &addrlen_list[j]);
			if (rc)
				return rc;

			if (lap->lap_addr_size != addrlen_list[j])
				return -FI_EINVAL;

			tmp += addrlen_list[j];
		}

		lap = (struct lnx_address_prov *)tmp;
		j++;
	}

	la->la_prov_count = j;
	memcpy(la->la_hostname, hostname, FI_NAME_MAX - 1);

	return 0;
}

static ssize_t lnx_ep_cancel(fid_t fid, void *context)
{
	int rc = 0;
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		lep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
		break;
	case FI_CLASS_RX_CTX:
		ctx = container_of(fid, struct lnx_ctx, ctx_ep.fid);
		lep = ctx->ctx_parent;
		break;
	case FI_CLASS_TX_CTX:
		return -FI_ENOENT;
	default:
		return -FI_EINVAL;
	}

	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			rc = fi_cancel(&ep->lpe_ep->fid, context);
			if (rc == -FI_ENOSYS) {
				FI_WARN(&lnx_prov, FI_LOG_CORE,
				 "%s: Operation not supported by provider. "
				 "Ignoring\n", ep->lpe_fabric_name);
				rc = 0;
				continue;
			} else if (rc != FI_SUCCESS) {
				return rc;
			}
		}
	}

	return rc;
}

static int lnx_ep_setopt(fid_t fid, int level, int optname, const void *optval,
			  size_t optlen)
{
	int rc = 0;
	struct lnx_ep *lep;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	lep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			rc = fi_setopt(&ep->lpe_ep->fid, level, optname,
				       optval, optlen);
			if (rc == -FI_ENOSYS) {
				FI_WARN(&lnx_prov, FI_LOG_CORE,
				 "%s: Operation not supported by provider. "
				 "Ignoring\n", ep->lpe_fabric_name);
				rc = 0;
				continue;
			} else if (rc != FI_SUCCESS) {
				return rc;
			}
		}
	}

	return rc;
}


static int lnx_ep_txc(struct fid_ep *fid, int index, struct fi_tx_attr *attr,
		      struct fid_ep **tx_ep, void *context)
{
	int rc = 0;
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	ctx = calloc(sizeof(*ctx), 1);
	if (!ctx)
		return -FI_ENOMEM;

	lep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			if (index >= ep->lpe_fi_info->ep_attr->tx_ctx_cnt)
				continue;

			rc = fi_tx_context(ep->lpe_ep, index, attr,
					   &ep->lpe_txc[index], context);
			if (rc == -FI_ENOSYS) {
				FI_WARN(&lnx_prov, FI_LOG_CORE,
				 "%s: Operation not supported by provider. "
				 "Ignoring\n", ep->lpe_fabric_name);
				rc = 0;
				continue;
			} else if (rc != FI_SUCCESS) {
				return rc;
			}
		}
	}

	dlist_init(&ctx->ctx_head);
	ctx->ctx_idx = index;
	ctx->ctx_parent = lep;
	lnx_init_ctx(&ctx->ctx_ep, FI_CLASS_TX_CTX);
	dlist_insert_tail(&ctx->ctx_head, &lep->le_tx_ctx);
	/* set the callbacks for the transmit context */
	*tx_ep = &ctx->ctx_ep;

	return rc;
}

static int lnx_ep_rxc(struct fid_ep *fid, int index, struct fi_rx_attr *attr,
		      struct fid_ep **rx_ep, void *context)
{
	int rc = 0;
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	ctx = calloc(sizeof(*ctx), 1);
	if (!ctx)
		return -FI_ENOMEM;

	lep = container_of(fid, struct lnx_ep, le_ep.ep_fid.fid);
	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			if (index >= ep->lpe_fi_info->ep_attr->rx_ctx_cnt)
				continue;

			rc = fi_rx_context(ep->lpe_ep, index, attr,
					   &ep->lpe_rxc[index], context);
			if (rc == -FI_ENOSYS) {
				FI_WARN(&lnx_prov, FI_LOG_CORE,
					"%s: Operation not supported by provider. "
					"Ignoring\n", ep->lpe_fabric_name);
				rc = 0;
				continue;
			} else if (rc != FI_SUCCESS) {
				return rc;
			}
		}
	}

	dlist_init(&ctx->ctx_head);
	ctx->ctx_idx = index;
	ctx->ctx_parent = lep;
	lnx_init_ctx(&ctx->ctx_ep, FI_CLASS_RX_CTX);
	dlist_insert_tail(&ctx->ctx_head, &lep->le_rx_ctx);
	/* set the callbacks for the receive context */
	*rx_ep = &ctx->ctx_ep;

	return rc;
}

struct fi_ops_ep lnx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = lnx_ep_cancel,
	/* can't get opt, because there is no way to report multiple
	 * options for the different links */
	.getopt = fi_no_getopt,
	.setopt = lnx_ep_setopt,
	.tx_ctx = lnx_ep_txc,
	.rx_ctx = lnx_ep_rxc,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

struct fi_ops lnx_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_ep_close,
	.bind = lnx_ep_bind,
	.control = lnx_ep_control,
	.ops_open = fi_no_ops_open,
};

struct fi_ops_cm lnx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = lnx_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

static int lnx_open_eps(struct local_prov *prov, struct fi_info *info,
			void *context, size_t fclass, struct lnx_ep *lep)
{
	int rc = 0;
	struct local_prov_ep *ep;
	struct dlist_entry *tmp;
	struct ofi_bufpool_attr bp_attrs = {};
	struct lnx_srx_context *ctxt;

	ctxt = calloc(1, sizeof(*ctxt));
	if (!ctxt)
		return -FI_ENOMEM;

	dlist_foreach_container_safe(&prov->lpv_prov_eps,
		struct local_prov_ep, ep, entry, tmp) {
		if (fclass == FI_CLASS_EP) {
			rc = fi_endpoint(ep->lpe_domain, ep->lpe_fi_info,
					 &ep->lpe_ep, context);
		} else {
			/* update endpoint attributes with whatever is being
			 * passed from the application
			 */
			if (ep->lpe_fi_info && info) {
				ep->lpe_fi_info->ep_attr->tx_ctx_cnt =
					info->ep_attr->tx_ctx_cnt;
				ep->lpe_fi_info->ep_attr->rx_ctx_cnt =
					info->ep_attr->rx_ctx_cnt;
			}

			ep->lpe_txc = calloc(info->ep_attr->tx_ctx_cnt,
					sizeof(*ep->lpe_txc));
			ep->lpe_rxc = calloc(info->ep_attr->rx_ctx_cnt,
					sizeof(*ep->lpe_rxc));
			if (!ep->lpe_txc || !ep->lpe_rxc)
				return -FI_ENOMEM;

			rc = fi_scalable_ep(ep->lpe_domain, ep->lpe_fi_info,
					    &ep->lpe_ep, context);
		}
		if (rc)
			return rc;

		ctxt->srx_lep = lep;
		ctxt->srx_cep = ep;

		ep->lpe_srx.ep_fid.fid.context = ctxt;
		ep->lpe_srx.ep_fid.fid.fclass = FI_CLASS_SRX_CTX;
		ofi_spin_init(&ep->lpe_bplock);
		/* create a buffer pool for the receive requests */
		bp_attrs.size = sizeof(struct lnx_rx_entry);
		bp_attrs.alignment = 8;
		bp_attrs.max_cnt = UINT16_MAX;
		bp_attrs.chunk_cnt = 64;
		bp_attrs.flags = OFI_BUFPOOL_NO_TRACK;
		rc = ofi_bufpool_create_attr(&bp_attrs, &ep->lpe_recv_bp);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_FABRIC,
				"Failed to create receive buffer pool");
			return -FI_ENOMEM;
		}
	}

	return 0;
}

static void
lnx_ep_nosys_progress(struct util_ep *util_ep)
{
	assert(0);
}

static inline int
match_tag(uint64_t tag, uint64_t match_tag, uint64_t ignore)
{
	return ((tag | ignore) == (match_tag | ignore));
}

static inline bool
lnx_addr_match(fi_addr_t addr1, fi_addr_t addr2)
{
	return (addr1 == addr2);
}

static inline bool
lnx_search_addr_match(fi_addr_t cep_addr, struct lnx_peer_prov *lpp)
{
	struct lnx_local2peer_map *lpm;
	fi_addr_t peer_addr;
	int i;

	dlist_foreach_container(&lpp->lpp_map,
				struct lnx_local2peer_map,
				lpm, entry) {
		for (i = 0; i < LNX_MAX_LOCAL_EPS; i++) {
			peer_addr = lpm->peer_addrs[i];
			if (peer_addr == FI_ADDR_NOTAVAIL)
				break;
			if (lnx_addr_match(peer_addr, cep_addr))
				return true;
		}
	}

	return false;
}

static int lnx_match_common(uint64_t tag1, uint64_t tag2, uint64_t ignore,
		fi_addr_t cep_addr, fi_addr_t lnx_addr, struct lnx_peer *peer,
		struct local_prov_ep *cep)
{
	struct lnx_peer_prov *lpp;
	struct local_prov *lp;
	bool tmatch;

	/* if a request has no address specified it'll match against any
	 * rx_entry with a matching tag
	 *  or
	 * if an rx_entry has no address specified, it'll match against any
	 * request with a matching tag
	 *
	 * for non tagged messages tags will be set to TAG_ANY so they will
	 * always match and decision will be made on address only.
	 */
	tmatch = match_tag(tag1, tag2, ignore);
	if (!tmatch)
		return tmatch;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "tag1=%lx tag2=%lx ignore=%lx cep_addr=%lx lnx_addr=%lx tmatch=%d\n",
	       tag1, tag2, ignore, cep_addr, lnx_addr, tmatch);

	/* if we're requested to receive from any peer, then tag maching is
	 * enough. None tagged message will match irregardless.
	 */
	if (lnx_addr == FI_ADDR_UNSPEC)
		return tmatch;

	/* if the address is specified, then we should have a peer and
	 * a receiving core endpoint and a provider parent
	 */
	assert(peer && cep && cep->lpe_parent);

	lp = cep->lpe_parent;

	/* if this is a shm core provider, then only go through lnx
	 * shm provider
	 */
	if (cep->lpe_local)
		return lnx_search_addr_match(cep_addr, peer->lp_shm_prov);

	/* check if we already have a peer provider.
	 * A peer can receive messages from multiple providers, we need to
	 * find the provider which maps to the provider we're currently
	 * checking. The map looked up can have multiple addresses which
	 * we can receive from, so we need to check which one of those is
	 * the correct match.
	 *
	 * Note: we're trying to make this loop as efficient as possible,
	 * because it's executed on the message matching path, which is
	 * heavily hit.
	 *
	 * The theory is in most use cases:
	 *   - There will be only two providers to check
	 *   - Each provider will have 1 endpoint, and therefore only one map
	 *   - Each peer will only have 1 address.
	 *
	 */
	dlist_foreach_container(&peer->lp_provs,
			struct lnx_peer_prov, lpp, entry) {
		if (lpp->lpp_prov == lp)
			return lnx_search_addr_match(cep_addr, lpp);
	}

	return false;
}

static int lnx_match_unexq(struct dlist_entry *item, const void *args)
{
	/* this entry is placed on the SUQ via the lnx_get_tag() path
	 * and examined in the lnx_process_tag() path */
	struct lnx_match_attr *match_attr = (struct lnx_match_attr *) args;
	struct lnx_rx_entry *entry = (struct lnx_rx_entry *) item;
	struct lnx_peer *peer = match_attr->lm_peer;

	/* entry refers to the unexpected message received
	 * entry->rx_entry.tag will be the tag of the message or TAG_UNSPEC
	 * otherwise
	 *
	 * entry->rx_entry.addr will be the address of the peer which sent the
	 * message or ADDR_UNSPEC if the core provider didn't do a reverse
	 * lookup.
	 *
	 * entry->rx_cep will be set to the core endpoint which received the
	 * message.
	 *
	 * match_attr is filled in by the lnx_process_tag() and contains
	 * information passed to us by the application
	 *
	 * match_attr->lm_peer is the peer looked up via the addr passed by
	 * the application to LNX. It is NULL if the addr is ADDR_UNSPEC.
	 *
	 * match_attr->lm_tag, match_attr->lm_ignore are the tag and ignore
	 * bits passed by the application to LNX via the receive API.
	 *
	 * match_attr->lm_addr is the only significant if it's set to
	 * FI_ADDR_UNSPEC, otherwise it's not used in matching because it's
	 * the LNX level address and we need to compare the core level address.
	 */
	return lnx_match_common(entry->rx_entry.tag, match_attr->lm_tag,
			match_attr->lm_ignore, entry->rx_entry.addr,
			match_attr->lm_addr, peer, entry->rx_cep);
}

static int lnx_match_recvq(struct dlist_entry *item, const void *args)
{
	struct lnx_match_attr *match_attr = (struct lnx_match_attr *) args;
	/* this entry is placed on the recvq via the lnx_process_tag() path
	 * and examined in the lnx_get_tag() path */
	struct lnx_rx_entry *entry = (struct lnx_rx_entry *) item;

	/* entry refers to the receive request waiting for a message
	 * entry->rx_entry.tag is the tag passed in by the application.
	 *
	 * entry->rx_entry.addr is the address passed in by the application.
	 * This is the LNX level address. It's only significant if it's set
	 * to ADDR_UNSPEC. Otherwise, it has already been used to look up the
	 * peer.
	 *
	 * entry->rx_cep is always NULL in this case, as this will only be
	 * known when the message is received.
	 *
	 * entry->rx_peer is the LNX peer looked up if a valid address is
	 * given by the application, otherwise it's NULL.
	 *
	 * match_attr information is filled by the lnx_get_tag() callback and
	 * contains information passed to us by the core endpoint receiving
	 * the message.
	 *
	 * match_attr->rx_peer is not significant because at the lnx_get_tag()
	 * call there isn't enough information to find what the peer is.
	 *
	 * match_attr->lm_tag, match_attr->lm_ignore are the tag and ignore
	 * bits passed up by the core endpoint receiving the message.
	 *
	 * match_attr->lm_addr is the address of the peer which sent the
	 * message. Set if the core endpoint has done a reverse lookup,
	 * otherwise set to ADDR_UNSPEC.
	 *
	 * match_attr->lm_cep is the core endpoint which received the message.
	 */
	return lnx_match_common(entry->rx_entry.tag, match_attr->lm_tag,
			entry->rx_ignore, match_attr->lm_addr,
			entry->rx_entry.addr, entry->rx_peer, match_attr->lm_cep);
}

static inline int
lnx_init_queue(struct lnx_queue *q, dlist_func_t *match_func)
{
	int rc;

	rc = ofi_spin_init(&q->lq_qlock);
	if (rc)
		return rc;

	dlist_init(&q->lq_queue);

	q->lq_match_func = match_func;

	return 0;
}

static inline int
lnx_init_qpair(struct lnx_qpair *qpair, dlist_func_t *recvq_match_func,
			   dlist_func_t *unexq_match_func)
{
	int rc = 0;

	rc = lnx_init_queue(&qpair->lqp_recvq, recvq_match_func);
	if (rc)
		goto out;
	rc = lnx_init_queue(&qpair->lqp_unexq, unexq_match_func);
	if (rc)
		goto out;

out:
	return rc;
}

static inline int
lnx_init_srq(struct lnx_peer_srq *srq)
{
	int rc;

	rc = lnx_init_qpair(&srq->lps_trecv, lnx_match_recvq, lnx_match_unexq);
	if (rc)
		return rc;
	rc = lnx_init_qpair(&srq->lps_recv, lnx_match_recvq, lnx_match_unexq);
	if (rc)
		return rc;

	return rc;
}

static int lnx_get_ctx(struct local_prov_ep *ep, size_t fclass,
		       struct fid_ep ***ep_ctx, size_t *size)
{
	switch (fclass) {
	case FI_CLASS_RX_CTX:
		*ep_ctx = ep->lpe_rxc;
		*size = ep->lpe_fi_info->ep_attr->rx_ctx_cnt;
		break;
	case FI_CLASS_TX_CTX:
		*ep_ctx = ep->lpe_txc;
		*size = ep->lpe_fi_info->ep_attr->tx_ctx_cnt;
		break;
	default:
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static void lnx_close_ep_ctx(struct local_prov_ep *ep, size_t fclass)
{
	struct fid_ep **ep_ctx;
	size_t size;
	size_t i;
	int rc;

	rc = lnx_get_ctx(ep, fclass, &ep_ctx, &size);
	if (rc)
		return;

	for (i = 0; i < size; i++) {
		rc = fi_close(&ep_ctx[i]->fid);
		if (rc)
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Failed to close ep context %lu with %d\n",
				fclass, rc);
	}
}

static int lnx_ctx_close(struct fid *fid)
{
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	if (fid->fclass != FI_CLASS_RX_CTX &&
	    fid->fclass != FI_CLASS_TX_CTX)
		return -FI_EINVAL;

	ctx = container_of(fid, struct lnx_ctx, ctx_ep.fid);
	lep = ctx->ctx_parent;

	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
					struct local_prov_ep, ep, entry)
			lnx_close_ep_ctx(ep, fid->fclass);
	}

	return FI_SUCCESS;
}

static int lnx_ctx_bind_cq(struct local_prov_ep *ep, size_t fclass,
			   struct fid *bfid, uint64_t flags)
{
	struct fid_ep **ep_ctx;
	size_t size;
	size_t i;
	int rc;

	rc = lnx_get_ctx(ep, fclass, &ep_ctx, &size);
	if (rc)
		return rc;

	for (i = 0; i < size; i++) {
		rc = fi_ep_bind(ep_ctx[i], bfid, flags);
		if (rc)
			return rc;
	}

	return FI_SUCCESS;
}

static int
lnx_ctx_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	if (fid->fclass != FI_CLASS_RX_CTX &&
	    fid->fclass != FI_CLASS_TX_CTX)
		return -FI_EINVAL;

	ctx = container_of(fid, struct lnx_ctx, ctx_ep.fid);
	lep = ctx->ctx_parent;

	fabric = lep->le_domain->ld_fabric;

	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		dlist_foreach_container(&entry->lpv_prov_eps,
			struct local_prov_ep, ep, entry) {
			if (bfid->fclass == FI_CLASS_CQ)
				/* bind the context to the shared cq */
				rc = lnx_ctx_bind_cq(ep, fid->fclass,
						&ep->lpe_cq.lpc_core_cq->fid,
						flags);
			else
				return -FI_ENOSYS;

			if (rc)
				return rc;
		}
	}

	return FI_SUCCESS;
}

static int
lnx_enable_ctx_eps(struct local_prov_ep *ep, size_t fclass)
{
	struct fid_ep **ep_ctx;
	size_t size;
	size_t i;
	int rc;

	rc = lnx_get_ctx(ep, fclass, &ep_ctx, &size);
	if (rc)
		return rc;

	for (i = 0; i < size; i++) {
		rc = fi_enable(ep_ctx[i]);
		if (rc)
			return rc;
	}

	return FI_SUCCESS;
}

static int
lnx_ctx_control(struct fid *fid, int command, void *arg)
{
	int rc;
	struct lnx_ep *lep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;

	if (fid->fclass != FI_CLASS_RX_CTX &&
	    fid->fclass != FI_CLASS_TX_CTX)
		return -FI_EINVAL;

	ctx = container_of(fid, struct lnx_ctx, ctx_ep.fid);
	lep = ctx->ctx_parent;

	fabric = lep->le_domain->ld_fabric;

	switch (command) {
	case FI_ENABLE:
		if (!lep->le_peer_tbl)
			return -FI_ENOAV;
		dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
					entry, lpv_entry) {
			dlist_foreach_container(&entry->lpv_prov_eps,
				struct local_prov_ep, ep, entry) {
				rc = lnx_enable_ctx_eps(ep, fid->fclass);
				if (rc)
					return rc;
			}
		}
		break;
	default:
		return -FI_ENOSYS;
	}

	return rc;
}

static struct fi_ops lnx_ctx_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_ctx_close,
	.bind = lnx_ctx_bind,
	.control = lnx_ctx_control,
	.ops_open = fi_no_ops_open,
};

struct fi_ops_ep lnx_ctx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = lnx_ep_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void
lnx_init_ctx(struct fid_ep *ctx, size_t fclass)
{
	ctx->fid.fclass = fclass;
	ctx->fid.ops = &lnx_ctx_ops;
	ctx->ops = &lnx_ctx_ep_ops;
	ctx->msg = &lnx_msg_ops;
	ctx->tagged = &lnx_tagged_ops;
	ctx->rma = &lnx_rma_ops;
	ctx->atomic = &lnx_atomic_ops;
}

static int
lnx_alloc_endpoint(struct fid_domain *domain, struct fi_info *info,
		   struct lnx_ep **out_ep, void *context, size_t fclass)
{
	int rc;
	struct lnx_ep *ep;
	struct local_prov *entry;
	struct lnx_fabric *fabric;
	uint64_t mr_mode;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ep->le_fclass = fclass;
	ep->le_ep.ep_fid.fid.fclass = fclass;

	ep->le_ep.ep_fid.fid.ops = &lnx_ep_fi_ops;
	ep->le_ep.ep_fid.ops = &lnx_ep_ops;
	ep->le_ep.ep_fid.cm = &lnx_cm_ops;
	ep->le_ep.ep_fid.msg = &lnx_msg_ops;
	ep->le_ep.ep_fid.tagged = &lnx_tagged_ops;
	ep->le_ep.ep_fid.rma = &lnx_rma_ops;
	ep->le_ep.ep_fid.atomic = &lnx_atomic_ops;
	ep->le_domain = container_of(domain, struct lnx_domain,
				     ld_domain.domain_fid);
	lnx_init_srq(&ep->le_srq);

	dlist_init(&ep->le_rx_ctx);
	dlist_init(&ep->le_tx_ctx);

	fabric = ep->le_domain->ld_fabric;

	/* create all the core provider endpoints */
	dlist_foreach_container(&fabric->local_prov_table, struct local_prov,
				entry, lpv_entry) {
		rc = lnx_open_eps(entry, info, context, fclass, ep);
		if (rc) {
			FI_WARN(&lnx_prov, FI_LOG_CORE,
				"Failed to create ep for %s\n",
				entry->lpv_prov_name);
			goto fail;
		}
	}

	mr_mode = lnx_util_prov.info->domain_attr->mr_mode;
	lnx_util_prov.info->domain_attr->mr_mode = 0;
	rc = ofi_endpoint_init(domain, (const struct util_prov *)&lnx_util_prov,
			       (struct fi_info *)lnx_util_prov.info, &ep->le_ep,
			       context, lnx_ep_nosys_progress);
	if (rc)
		goto fail;

	lnx_util_prov.info->domain_attr->mr_mode = mr_mode;
	*out_ep = ep;

	return 0;

fail:
	free(ep);
	return rc;
}

int lnx_scalable_ep(struct fid_domain *domain, struct fi_info *info,
		    struct fid_ep **ep, void *context)
{
	int rc;
	struct lnx_ep *my_ep;

	rc = lnx_alloc_endpoint(domain, info, &my_ep, context, FI_CLASS_SEP);
	if (rc)
		return rc;

	*ep = &my_ep->le_ep.ep_fid;

	return 0;
}

int lnx_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	int rc;
	struct lnx_ep *my_ep;

	rc = lnx_alloc_endpoint(domain, info, &my_ep, context, FI_CLASS_EP);
	if (rc)
		return rc;

	*ep = &my_ep->le_ep.ep_fid;

	return 0;
}


