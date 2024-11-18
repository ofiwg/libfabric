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
#include "ofi_lock.h"
#include "rdma/fi_ext.h"
#include "ofi_iov.h"
#include "lnx.h"

int lnx_get_msg(struct fid_peer_srx *srx, struct fi_peer_match_attr *match,
		struct fi_peer_rx_entry **entry)
{
	return -FI_ENOSYS;
}

int lnx_queue_msg(struct fi_peer_rx_entry *entry)
{
	return -FI_ENOSYS;
}

void lnx_free_entry(struct fi_peer_rx_entry *entry)
{
	struct lnx_rx_entry *rx_entry = (struct lnx_rx_entry *) entry;
	ofi_spin_t *bplock;

	if (rx_entry->rx_global)
		bplock = &global_bplock;
	else
		bplock = &rx_entry->rx_cep->lpe_bplock;

	ofi_spin_lock(bplock);
	ofi_buf_free(rx_entry);
	ofi_spin_unlock(bplock);
}

static struct lnx_ep *lnx_get_lep(struct fid_ep *ep, struct lnx_ctx **ctx)
{
	struct lnx_ep *lep;

	if (ctx)
		*ctx = NULL;

	switch (ep->fid.fclass) {
	case FI_CLASS_RX_CTX:
	case FI_CLASS_TX_CTX:
		*ctx = container_of(ep, struct lnx_ctx, ctx_ep.fid);
		lep = (*ctx)->ctx_parent;
		break;
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		lep = container_of(ep, struct lnx_ep, le_ep.ep_fid.fid);
		break;
	default:
		lep = NULL;
	}

	return lep;
}

static struct fid_ep *lnx_get_core_ep(struct local_prov_ep *cep, int idx,
				      size_t fclass)
{
	switch (fclass) {
	case FI_CLASS_RX_CTX:
		return cep->lpe_rxc[idx];
	case FI_CLASS_TX_CTX:
		return cep->lpe_txc[idx];
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		return cep->lpe_ep;
	default:
		return NULL;
	}

	return NULL;
}

static void
lnx_init_rx_entry(struct lnx_rx_entry *entry, struct iovec *iov, void **desc,
		  size_t count, fi_addr_t addr, uint64_t tag,
		  uint64_t ignore, void *context, uint64_t flags)
{
	memcpy(&entry->rx_iov, iov, sizeof(*iov) * count);
	if (desc)
		memcpy(entry->rx_desc, desc, sizeof(*desc) * count);

	entry->rx_entry.iov = entry->rx_iov;
	entry->rx_entry.desc = entry->rx_desc;
	entry->rx_entry.count = count;
	entry->rx_entry.addr = addr;
	entry->rx_entry.context = context;
	entry->rx_entry.tag = tag;
	entry->rx_entry.flags = flags;
	entry->rx_ignore = ignore;
}

static struct lnx_rx_entry *
get_rx_entry(struct local_prov_ep *cep, struct iovec *iov, void **desc,
	size_t count, fi_addr_t addr, uint64_t tag,
	uint64_t ignore, void *context, uint64_t flags)
{
	struct lnx_rx_entry *rx_entry = NULL;
	ofi_spin_t *bplock;
	struct ofi_bufpool *bp;

	/* if lp is NULL, then we don't know where the message is going to
	 * come from, so allocate the rx_entry from a global pool
	 */
	if (!cep) {
		bp = global_recv_bp;
		bplock = &global_bplock;
	} else {
		bp = cep->lpe_recv_bp;
		bplock = &cep->lpe_bplock;
	}

	ofi_spin_lock(bplock);
	rx_entry = (struct lnx_rx_entry *)ofi_buf_alloc(bp);
	ofi_spin_unlock(bplock);
	if (rx_entry) {
		memset(rx_entry, 0, sizeof(*rx_entry));
		if (!cep)
			rx_entry->rx_global = true;
		rx_entry->rx_cep = cep;
		lnx_init_rx_entry(rx_entry, iov, desc, count, addr, tag,
				  ignore, context, flags);
	}

	return rx_entry;
}

static inline struct lnx_rx_entry *
lnx_remove_first_match(struct lnx_queue *q, struct lnx_match_attr *match)
{
	struct lnx_rx_entry *rx_entry;

	ofi_spin_lock(&q->lq_qlock);
	rx_entry = (struct lnx_rx_entry *) dlist_remove_first_match(
			&q->lq_queue, q->lq_match_func, match);
	ofi_spin_unlock(&q->lq_qlock);

	return rx_entry;
}

static inline void
lnx_insert_rx_entry(struct lnx_queue *q, struct lnx_rx_entry *entry)
{
	ofi_spin_lock(&q->lq_qlock);
	dlist_insert_tail((struct dlist_entry *)(&entry->rx_entry),
			  &q->lq_queue);
	ofi_spin_unlock(&q->lq_qlock);
}

int lnx_queue_tag(struct fi_peer_rx_entry *entry)
{
	struct lnx_rx_entry *rx_entry = (struct lnx_rx_entry *)entry;
	struct lnx_peer_srq *lnx_srq = (struct lnx_peer_srq*)entry->owner_context;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
		"addr = %lx tag = %lx ignore = 0 found\n",
		entry->addr, entry->tag);

	lnx_insert_rx_entry(&lnx_srq->lps_trecv.lqp_unexq, rx_entry);

	return 0;
}

int lnx_get_tag(struct fid_peer_srx *srx, struct fi_peer_match_attr *match,
		struct fi_peer_rx_entry **entry)
{
	struct lnx_match_attr match_attr;
	struct lnx_peer_srq *lnx_srq;
	struct local_prov_ep *cep;
	struct lnx_ep *lep;
	struct lnx_rx_entry *rx_entry;
	fi_addr_t addr = match->addr;
	struct lnx_srx_context *srx_ctxt;
	uint64_t tag = match->tag;
	int rc = 0;

	/* get the endpoint */
	cep = container_of(srx, struct local_prov_ep, lpe_srx);
	srx_ctxt = cep->lpe_srx.ep_fid.fid.context;
	cep = srx_ctxt->srx_cep;
	lep = srx_ctxt->srx_lep;
	lnx_srq = &lep->le_srq;

	/* The fi_addr_t is a generic address returned by the provider. It's usually
	 * just an index or id in their AV table. When I get it here, I could have
	 * duplicates if multiple providers are using the same scheme to
	 * insert in the AV table. I need to be able to identify the provider
	 * in this function so I'm able to correctly match this message to
	 * a possible rx entry on my receive queue. That's why we need to make
	 * sure we use the core endpoint as part of the matching key.
	 */
	memset(&match_attr, 0, sizeof(match_attr));

	match_attr.lm_addr = addr;
	match_attr.lm_ignore = 0;
	match_attr.lm_tag = tag;
	match_attr.lm_cep = cep;

	/*  1. Find a matching request to the message received.
	 *  2. Return the receive request.
	 *  3. If there are no matching requests, then create a new one
	 *     and return it to the core provider. The core provider will turn
	 *     around and tell us to queue it. Return -FI_ENOENT.
	 */
	rx_entry = lnx_remove_first_match(&lnx_srq->lps_trecv.lqp_recvq,
					  &match_attr);
	if (rx_entry) {
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "addr = %lx tag = %lx ignore = 0 found\n",
		       addr, tag);

		goto assign;
	}

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr = %lx tag = %lx ignore = 0 not found\n",
	       addr, tag);

	rx_entry = get_rx_entry(cep, NULL, NULL, 0, addr, tag, 0, NULL,
				lnx_ep_rx_flags(lep));
	if (!rx_entry) {
		rc = -FI_ENOMEM;
		goto out;
	}

	rx_entry->rx_match_info = *match;
	rx_entry->rx_entry.owner_context = lnx_srq;
	rx_entry->rx_entry.msg_size = match->msg_size;

	rc = -FI_ENOENT;

assign:
	rx_entry->rx_entry.msg_size = MIN(rx_entry->rx_entry.msg_size,
				      match->msg_size);
	*entry = &rx_entry->rx_entry;

out:
	return rc;
}

/*
 * if lp is NULL, then we're attempting to receive from any peer so
 * matching the tag is the only thing that matters.
 *
 * if lp != NULL, then we're attempting to receive from a particular
 * peer. This peer can have multiple endpoints serviced by different core
 * providers.
 *
 * Therefore when we check the unexpected queue, we need to check
 * if we received any messages from any of the peer's addresses. If we
 * find one, then we kick the core provider associated with that
 * address to receive the message.
 *
 * If nothing is found on the unexpected messages, then add a receive
 * request on the SRQ; happens in the lnx_process_recv()
 */
static int lnx_process_recv(struct lnx_ep *lep, struct iovec *iov, void **desc,
			fi_addr_t addr, size_t count, struct lnx_peer *lp, uint64_t tag,
			uint64_t ignore, void *context, uint64_t flags,
			bool tagged)
{
	struct lnx_peer_srq *lnx_srq = &lep->le_srq;
	struct local_prov_ep *cep;
	struct lnx_rx_entry *rx_entry;
	struct lnx_match_attr match_attr;
	int rc = 0;

	match_attr.lm_addr = addr;
	match_attr.lm_ignore = ignore;
	match_attr.lm_tag = tag;
	match_attr.lm_cep = NULL;
	match_attr.lm_peer = lp;

	/* if support is turned off, don't go down the SRQ path */
	if (!lep->le_domain->ld_srx_supported)
		return -FI_ENOSYS;

	rx_entry = lnx_remove_first_match(&lnx_srq->lps_trecv.lqp_unexq,
					  &match_attr);
	if (!rx_entry) {
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "addr=%lx tag=%lx ignore=%lx buf=%p len=%lx not found\n",
		       addr, tag, ignore, iov->iov_base, iov->iov_len);

		goto nomatch;
	}

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr=%lx tag=%lx ignore=%lx buf=%p len=%lx found\n",
	       addr, tag, ignore, iov->iov_base, iov->iov_len);

	cep = rx_entry->rx_cep;

	/* match is found in the unexpected queue. call into the core
	 * provider to complete this message
	 */
	lnx_init_rx_entry(rx_entry, iov, desc, count, addr, tag, ignore,
			  context, lnx_ep_rx_flags(lep));
	rx_entry->rx_entry.msg_size = MIN(ofi_total_iov_len(iov, count),
				      rx_entry->rx_entry.msg_size);
	if (tagged)
		rc = cep->lpe_srx.peer_ops->start_tag(&rx_entry->rx_entry);
	else
		rc = cep->lpe_srx.peer_ops->start_msg(&rx_entry->rx_entry);

	if (rc == -FI_EINPROGRESS) {
		/* this is telling me that more messages can match the same
		 * rx_entry. So keep it on the queue
		 */
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "addr = %lx tag = %lx ignore = %lx start_tag() in progress\n",
		       addr, tag, ignore);

		goto insert_recvq;
	} else if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_CORE, "start tag failed with %d\n", rc);
	}

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr = %lx tag = %lx ignore = %lx start_tag() success\n",
	       addr, tag, ignore);

	return 0;

nomatch:
	/* nothing on the unexpected queue, then allocate one and put it on
	 * the receive queue
	 */
	rx_entry = get_rx_entry(NULL, iov, desc, count, addr, tag, ignore,
				context, lnx_ep_rx_flags(lep));
	rx_entry->rx_entry.msg_size = ofi_total_iov_len(iov, count);
	if (!rx_entry) {
		rc = -FI_ENOMEM;
		goto out;
	}
	rx_entry->rx_peer = lp;

insert_recvq:
	lnx_insert_rx_entry(&lnx_srq->lps_trecv.lqp_recvq, rx_entry);

out:
	return rc;
}

ssize_t lnx_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep = NULL;
	fi_addr_t core_addr = FI_ADDR_UNSPEC;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct iovec iov = {.iov_base = buf, .iov_len = len};
	struct lnx_peer *lp;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	lnx_get_core_desc(desc, &mem_desc);

	/* addr is an index into the peer table.
	 * This gets us to a peer. Each peer can be reachable on
	 * multiple endpoints. Each endpoint has its own fi_addr_t which is
	 * core provider specific.
	 */
	lp = lnx_get_peer(peer_tbl->lpt_entries, src_addr);
	if (lp) {
		rc = lnx_select_recv_pathway(lp, lep->le_domain, desc, &cep,
					     &core_addr, &iov, 1, &mre, &mem_desc);
		if (rc)
			goto out;
	}

	rc = lnx_process_recv(lep, &iov, &mem_desc, src_addr, 1, lp, tag, ignore,
			      context, 0, true);
	if (rc == -FI_ENOSYS)
		goto do_recv;
	else if (rc)
		FI_WARN(&lnx_prov, FI_LOG_CORE, "lnx_process_recv failed with %d\n", rc);

	goto out;

do_recv:
	if (lp)
		rc = fi_trecv(cep->lpe_ep, buf, len, mem_desc, core_addr, tag, ignore, context);

out:
	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep = NULL;
	fi_addr_t core_addr = FI_ADDR_UNSPEC;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct lnx_peer *lp;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;
	lnx_get_core_desc(*desc, &mem_desc);

	lp = lnx_get_peer(peer_tbl->lpt_entries, src_addr);
	if (lp) {
		rc = lnx_select_recv_pathway(lp, lep->le_domain, *desc, &cep,
					     &core_addr, iov, count, &mre, &mem_desc);
		if (rc)
			goto out;
	}

	rc = lnx_process_recv(lep, (struct iovec *)iov, &mem_desc, src_addr,
			      1, lp, tag, ignore, context, 0, true);
	if (rc == -FI_ENOSYS)
		goto do_recv;

	goto out;

do_recv:
	if (lp)
		rc = fi_trecvv(cep->lpe_ep, iov, &mem_desc, count, core_addr, tag, ignore, context);

out:
	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
		     uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep = NULL;
	fi_addr_t core_addr = FI_ADDR_UNSPEC;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct lnx_peer *lp;
	struct fi_msg_tagged core_msg;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	lp = lnx_get_peer(peer_tbl->lpt_entries, msg->addr);
	if (lp) {
		rc = lnx_select_recv_pathway(lp, lep->le_domain, *msg->desc,
					&cep, &core_addr, msg->msg_iov,
					msg->iov_count, &mre, &mem_desc);
		if (rc)
			goto out;
	}
	lnx_get_core_desc(*msg->desc, &mem_desc);

	rc = lnx_process_recv(lep, (struct iovec *)msg->msg_iov, &mem_desc,
			msg->addr, msg->iov_count, lp, msg->tag, msg->ignore,
			msg->context, flags, true);
	if (rc == -FI_ENOSYS)
		goto do_recv;

	goto out;

do_recv:
	if (lp) {
		memcpy(&core_msg, msg, sizeof(*msg));

		core_msg.desc = mem_desc;
		core_msg.addr = core_addr;

		rc = fi_trecvmsg(cep->lpe_ep, &core_msg, flags);
	}

out:
	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*) buf, .iov_len = len};

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				     lep->le_domain, desc, &cep,
				     &core_addr, &iov, 1, &mre, &mem_desc, NULL);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx buf %p len %ld\n",
	       core_addr, tag, buf, len);

	rc = fi_tsend(cep->lpe_ep, buf, len, mem_desc, core_addr, tag, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	struct ofi_mr_entry *mre = NULL;
	void *mem_desc;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, (desc) ? *desc : NULL, &cep,
				&core_addr, iov, count, &mre, &mem_desc, NULL);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx\n", core_addr, tag);

	rc = fi_tsendv(cep->lpe_ep, iov, &mem_desc, count, core_addr, tag, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
		uint64_t flags)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct fi_msg_tagged core_msg;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[msg->addr],
				lep->le_domain,
				(msg->desc) ? *msg->desc : NULL, &cep,
				&core_addr, msg->msg_iov,
				msg->iov_count, &mre, &mem_desc, NULL);
	if (rc)
		return rc;

	memcpy(&core_msg, msg, sizeof(*msg));

	core_msg.desc = mem_desc;
	core_msg.addr = core_addr;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx\n", core_msg.addr, core_msg.tag);

	rc = fi_tsendmsg(cep->lpe_ep, &core_msg, flags);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tinject(struct fid_ep *ep, const void *buf, size_t len,
		fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, NULL, &cep,
				&core_addr, NULL, 0, &mre, NULL, NULL);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx buf %p len %ld\n",
	       core_addr, tag, buf, len);

	rc = fi_tinject(cep->lpe_ep, buf, len, core_addr, tag);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tsenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		uint64_t data, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = len};

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, desc, &cep,
				&core_addr, &iov, 1, &mre, &mem_desc, NULL);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx buf %p len %ld\n",
	       core_addr, tag, buf, len);

	rc = fi_tsenddata(cep->lpe_ep, buf, len, mem_desc,
			  data, core_addr, tag, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

ssize_t lnx_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	int rc;
	struct lnx_ep *lep;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	struct ofi_mr_entry *mre = NULL;

	lep = lnx_get_lep(ep, NULL);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				     lep->le_domain, NULL, &cep,
				     &core_addr, NULL, 0, &mre, NULL, NULL);
	if (rc)
		return rc;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx tag %lx buf %p len %ld\n",
	       core_addr, tag, buf, len);

	rc = fi_tinjectdata(cep->lpe_ep, buf, len, data, core_addr, tag);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

	return rc;
}

static inline ssize_t
lnx_rma_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *core_ep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	uint64_t rkey;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = len};

	lep = lnx_get_lep(ep, &ctx);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[src_addr],
				     lep->le_domain, desc, &cep,
				     &core_addr, &iov, 1, &mre, &mem_desc, &rkey);
	if (rc)
		goto out;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "rma read from %lx key %lx buf %p len %ld\n",
	       core_addr, key, buf, len);

	core_ep = lnx_get_core_ep(cep, ctx->ctx_idx, ep->fid.fclass);

	rc = fi_read(core_ep, buf, len, mem_desc,
		     core_addr, addr, key, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);
out:
	return rc;
}

static inline ssize_t
lnx_rma_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
	 fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *core_ep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	uint64_t rkey;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = len};

	lep = lnx_get_lep(ep, &ctx);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				     lep->le_domain, desc, &cep,
				&core_addr, &iov, 1, &mre, &mem_desc, &rkey);
	if (rc)
		goto out;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "rma write to %lx key %lx buf %p len %ld\n",
	       core_addr, key, buf, len);

	core_ep = lnx_get_core_ep(cep, ctx->ctx_idx, ep->fid.fclass);

	rc = fi_write(core_ep, buf, len, mem_desc,
		      core_addr, addr, key, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);
out:
	return rc;
}

static inline ssize_t
lnx_atomic_write(struct fid_ep *ep,
	  const void *buf, size_t count, void *desc,
	  fi_addr_t dest_addr,
	  uint64_t addr, uint64_t key,
	  enum fi_datatype datatype, enum fi_op op, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *core_ep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	uint64_t rkey;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = count};

	lep = lnx_get_lep(ep, &ctx);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, desc, &cep,
				&core_addr, &iov, 1, &mre, &mem_desc, &rkey);
	if (rc)
		goto out;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx\n", core_addr);

	core_ep = lnx_get_core_ep(cep, ctx->ctx_idx, ep->fid.fclass);

	rc = fi_atomic(core_ep, buf, count, mem_desc,
		      core_addr, addr, key, datatype, op, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);
out:
	return rc;
}

static inline ssize_t
lnx_atomic_readwrite(struct fid_ep *ep,
		const void *buf, size_t count, void *desc,
		void *result, void *result_desc,
		fi_addr_t dest_addr,
		uint64_t addr, uint64_t key,
		enum fi_datatype datatype, enum fi_op op, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *core_ep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	uint64_t rkey;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = count};

	lep = lnx_get_lep(ep, &ctx);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, result_desc, &cep, &core_addr, &iov, 1,
				&mre, &mem_desc, &rkey);
	if (rc)
		goto out;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx\n", core_addr);

	core_ep = lnx_get_core_ep(cep, ctx->ctx_idx, ep->fid.fclass);

	rc = fi_fetch_atomic(core_ep, buf, count, desc,
		      result, mem_desc, core_addr, addr, key,
		      datatype, op, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);
out:
	return rc;
}

static inline ssize_t
lnx_atomic_compwrite(struct fid_ep *ep,
		  const void *buf, size_t count, void *desc,
		  const void *compare, void *compare_desc,
		  void *result, void *result_desc,
		  fi_addr_t dest_addr,
		  uint64_t addr, uint64_t key,
		  enum fi_datatype datatype, enum fi_op op, void *context)
{
	int rc;
	struct lnx_ep *lep;
	struct fid_ep *core_ep;
	struct lnx_ctx *ctx;
	struct local_prov_ep *cep;
	fi_addr_t core_addr;
	struct lnx_peer_table *peer_tbl;
	void *mem_desc;
	uint64_t rkey;
	struct ofi_mr_entry *mre = NULL;
	struct iovec iov = {.iov_base = (void*)buf, .iov_len = count};

	lep = lnx_get_lep(ep, &ctx);
	if (!lep)
		return -FI_ENOSYS;

	peer_tbl = lep->le_peer_tbl;

	rc = lnx_select_send_pathway(peer_tbl->lpt_entries[dest_addr],
				lep->le_domain, result_desc, &cep, &core_addr, &iov, 1,
				&mre, &mem_desc, &rkey);
	if (rc)
		goto out;

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "sending to %lx\n", core_addr);

	core_ep = lnx_get_core_ep(cep, ctx->ctx_idx, ep->fid.fclass);

	rc = fi_compare_atomic(core_ep, buf, count, desc,
		      compare, compare_desc, result, mem_desc,
		      core_addr, addr, key, datatype, op, context);

	if (mre)
		ofi_mr_cache_delete(&lep->le_domain->ld_mr_cache, mre);

out:
	return rc;
}

struct fi_ops_tagged lnx_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = lnx_trecv,
	.recvv = lnx_trecvv,
	.recvmsg = lnx_trecvmsg,
	.send = lnx_tsend,
	.sendv = lnx_tsendv,
	.sendmsg = lnx_tsendmsg,
	.inject = lnx_tinject,
	.senddata = lnx_tsenddata,
	.injectdata = lnx_tinjectdata,
};

struct fi_ops_msg lnx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

struct fi_ops_rma lnx_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = lnx_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = lnx_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

struct fi_ops_atomic lnx_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = lnx_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = lnx_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = lnx_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
};


