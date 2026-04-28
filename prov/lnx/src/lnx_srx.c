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

#include "lnx.h"
#include "ofi_iov.h"

static int lnx_get_msg(struct fid_peer_srx *srx,
		       struct fi_peer_match_attr *match,
		       struct fi_peer_rx_entry **entry)
{
	return -FI_ENOSYS;
}

static int lnx_queue_msg(struct fi_peer_rx_entry *entry)
{
	return -FI_ENOSYS;
}

static void lnx_free_entry(struct fi_peer_rx_entry *entry)
{
	struct lnx_rx_entry *rx_entry;

	rx_entry = container_of(entry, struct lnx_rx_entry, rx_entry);

	ofi_genlock_lock(&rx_entry->rx_lep->le_ep.lock);
	ofi_buf_free(rx_entry);
	ofi_genlock_unlock(&rx_entry->rx_lep->le_ep.lock);
}

static void lnx_init_rx_entry(struct lnx_rx_entry *entry,
			      const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t addr,
			      uint64_t tag, uint64_t ignore, void *context,
			      uint64_t flags)
{
	if (iov)
		memcpy(&entry->rx_iov, iov, sizeof(*iov) * count);
	if (desc)
		memcpy(entry->rx_desc, desc, sizeof(*desc) * count);

	entry->rx_entry.iov = entry->rx_iov;
	entry->rx_entry.desc = entry->rx_desc;
	entry->rx_entry.count = count;
	entry->rx_entry.addr = addr;
	entry->rx_entry.context = context;
	entry->rx_entry.tag = tag;
	entry->rx_entry.flags = flags | FI_TAGGED | FI_RECV;
	entry->rx_ignore = ignore;
}

static struct lnx_rx_entry *get_rx_entry(struct lnx_ep *lep,
					 const struct iovec *iov, void **desc,
					 size_t count, fi_addr_t addr,
					 uint64_t tag, uint64_t ignore,
					 void *context, uint64_t flags)
{
	struct lnx_rx_entry *rx_entry = NULL;

	ofi_genlock_lock(&lep->le_ep.lock);
	rx_entry = (struct lnx_rx_entry *)ofi_buf_alloc(lep->le_recv_bp);
	ofi_genlock_unlock(&lep->le_ep.lock);
	if (!rx_entry)
		return NULL;

	rx_entry->rx_lep = lep;
	lnx_init_rx_entry(rx_entry, iov, desc, count, addr, tag,
				ignore, context, flags);

	return rx_entry;
}

static inline struct lnx_rx_entry *lnx_find_first_match(
			struct lnx_queue *q, struct lnx_match_attr *match)
{
	struct lnx_rx_entry *rx_entry;

	rx_entry = (struct lnx_rx_entry *) dlist_find_first_match(
			&q->lq_queue, q->lq_match_func, match);

	return rx_entry;
}

static inline void lnx_update_queue_stats(struct lnx_queue *q, bool dq)
{
	if (dq)
		q->lq_size--;
	else
		q->lq_size++;

	if (q->lq_size > q->lq_max)
		q->lq_max = q->lq_size;

	q->lq_rolling_sum += q->lq_size;
	q->lq_count++;
	q->lq_rolling_avg = q->lq_rolling_sum / q->lq_count;
}

static inline struct lnx_rx_entry *lnx_remove_first_match(
			struct lnx_queue *q, struct lnx_match_attr *match)
{
	struct lnx_rx_entry *rx_entry;

	rx_entry = (struct lnx_rx_entry *) dlist_remove_first_match(
			&q->lq_queue, q->lq_match_func, match);
	if (rx_entry)
		lnx_update_queue_stats(q, true);

	return rx_entry;
}

static inline void lnx_insert_rx_entry(struct lnx_queue *q,
				       struct lnx_rx_entry *entry)
{
	dlist_insert_tail(&entry->entry, &q->lq_queue);
	lnx_update_queue_stats(q, false);
}

static int lnx_queue_tag(struct fi_peer_rx_entry *entry)
{
	struct lnx_rx_entry *rx_entry;
	struct lnx_peer_srq *lnx_srq =
				(struct lnx_peer_srq*)entry->owner_context;

	rx_entry = container_of(entry, struct lnx_rx_entry, rx_entry);
	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr = %" PRIx64 " tag = %" PRIx64 " ignore = 0 found\n",
	       entry->addr, entry->tag);

	lnx_insert_rx_entry(&lnx_srq->lps_trecv.lqp_unexq, rx_entry);

	return 0;
}

static int lnx_get_tag(struct fid_peer_srx *srx,
		       struct fi_peer_match_attr *match,
		       struct fi_peer_rx_entry **entry)
{
	struct lnx_match_attr match_attr = {0};
	struct lnx_peer_srq *lnx_srq;
	struct lnx_core_ep *cep;
	struct lnx_ep *lep;
	struct lnx_rx_entry *rx_entry;
	fi_addr_t addr = match->addr;
	uint64_t tag = match->tag;
	int rc = 0;

	cep = srx->ep_fid.fid.context;
	lep = cep->cep_parent;
	lnx_srq = &lep->le_srq;

	match_attr.lm_addr = addr;
	match_attr.lm_tag = tag;

	rx_entry = lnx_remove_first_match(&lnx_srq->lps_trecv.lqp_recvq,
					  &match_attr);
	if (rx_entry) {
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "addr = %" PRIx64 " tag = %" PRIx64
		       " ignore = 0 found\n", match_attr.lm_addr, tag);

		goto assign;
	}

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr = %" PRIx64 " tag = %" PRIx64 " ignore = 0 not found\n",
	       match_attr.lm_addr, tag);

	rx_entry = get_rx_entry(lep, NULL, NULL, 0, addr, tag, 0, NULL,
				FI_TAGGED | FI_RECV);
	if (!rx_entry) {
		rc = -FI_ENOMEM;
		goto out;
	}

	rx_entry->rx_entry.owner_context = lnx_srq;
	rx_entry->rx_cep = cep;

	rc = -FI_ENOENT;

	cep->cep_t_stats.st_num_unexp_msgs++;

	goto finalize;

assign:
	lnx_set_send_pair[lep->le_mr](lep, cep, addr);

	cep->cep_t_stats.st_num_posted_recvs++;

	rx_entry->rx_entry.addr = lnx_get_core_addr(cep, addr);
	if (rx_entry->rx_entry.desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain,
					 rx_entry->rx_entry.desc,
					 rx_entry->rx_entry.count,
					 rx_entry->rx_entry.desc);
		if (rc)
			return rc;
	}
finalize:
	rx_entry->rx_entry.msg_size = match->msg_size;
	*entry = &rx_entry->rx_entry;

out:
	return rc;
}

static void lnx_update_msg_entries(
			struct lnx_qpair *qp,
			fi_addr_t (*get_addr)(struct fi_peer_rx_entry *))
{
	struct lnx_queue *q = &qp->lqp_unexq;
	struct lnx_rx_entry *rx_entry;
	struct dlist_entry *item;

	dlist_foreach(&q->lq_queue, item) {
		rx_entry = (struct lnx_rx_entry *) item;
		if (rx_entry->rx_entry.addr == FI_ADDR_UNSPEC)
			rx_entry->rx_entry.addr = get_addr(&rx_entry->rx_entry);
	}
}

static void lnx_foreach_unspec_addr(
			struct fid_peer_srx *srx,
			fi_addr_t (*get_addr)(struct fi_peer_rx_entry *))
{
	struct lnx_ep *lep;
	struct lnx_core_ep *cep;

	cep = (struct lnx_core_ep *) srx->ep_fid.fid.context;
	lep = cep->cep_parent;

	lnx_update_msg_entries(&lep->le_srq.lps_trecv, get_addr);
	lnx_update_msg_entries(&lep->le_srq.lps_recv, get_addr);
}

struct fi_ops_srx_owner lnx_srx_ops = {
	.size = sizeof(struct fi_ops_srx_owner),
	.get_msg = lnx_get_msg,
	.get_tag = lnx_get_tag,
	.queue_msg = lnx_queue_msg,
	.queue_tag = lnx_queue_tag,
	.free_entry = lnx_free_entry,
	.foreach_unspec_addr = lnx_foreach_unspec_addr,
};

static int lnx_discard(struct lnx_ep *lep, struct lnx_rx_entry *rx_entry,
		       void *context)
{
	struct lnx_core_ep *cep = rx_entry->rx_cep;
	int rc;

	rc = cep->cep_srx.peer_ops->discard_tag(&rx_entry->rx_entry);
	if (rc) {
		FI_WARN(&lnx_prov, FI_LOG_CORE,
			"Error discarding message from %s\n",
			cep->cep_domain->cd_info->fabric_attr->name);
	}

	rc = ofi_cq_write(lep->le_ep.rx_cq, context, rx_entry->rx_entry.flags,
			  rx_entry->rx_entry.msg_size, NULL,
			  rx_entry->rx_entry.cq_data, rx_entry->rx_entry.tag);

	dlist_remove(&rx_entry->entry);
	lnx_free_entry(&rx_entry->rx_entry);

	return rc;
}

static int lnx_peek(struct lnx_ep *lep, struct lnx_match_attr *match_attr,
		    void *context, uint64_t flags)
{
	int rc;
	struct lnx_rx_entry *rx_entry;
	struct lnx_peer_srq *lnx_srq = &lep->le_srq;

	rx_entry = lnx_find_first_match(&lnx_srq->lps_trecv.lqp_unexq,
					match_attr);
	if (!rx_entry) {
		FI_DBG(&lnx_prov, FI_LOG_CORE,
			"PEEK addr=%" PRIx64 " tag=%" PRIx64 " ignore=%"
			PRIx64 "\n", match_attr->lm_addr, match_attr->lm_tag,
			match_attr->lm_ignore);
		return ofi_cq_write_error_peek(lep->le_ep.rx_cq,
					       match_attr->lm_tag, context);
	}

	rc = ofi_cq_write(lep->le_ep.rx_cq, context,
			  rx_entry->rx_entry.flags,
			  rx_entry->rx_entry.msg_size, NULL,
			  rx_entry->rx_entry.cq_data,
			  rx_entry->rx_entry.tag);

	if (flags & FI_DISCARD) {
		rc = rx_entry->rx_cep->cep_srx.peer_ops->discard_tag(
							&rx_entry->rx_entry);
		dlist_remove(&rx_entry->entry);
		lnx_free_entry(&rx_entry->rx_entry);
		goto out;
	}

	if (flags & FI_CLAIM) {
		dlist_remove(&rx_entry->entry);
		((struct fi_context *)context)->internal[0] = rx_entry;
	}

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
int lnx_process_recv(struct lnx_ep *lep, const struct iovec *iov, void **desc,
		     fi_addr_t addr, size_t count, uint64_t tag,
		     uint64_t ignore, void *context, uint64_t flags,
		     bool tagged)
{
	struct lnx_peer_srq *lnx_srq = &lep->le_srq;
	struct lnx_rx_entry *rx_entry;
	struct lnx_match_attr match_attr;
	struct lnx_core_ep *cep;
	int rc = 0;
	fi_addr_t sub_addr, encoded_addr = lnx_encode_fi_addr(addr, 0);

	/* Matching format should always be in the encoded form */
	match_attr.lm_addr = (addr == FI_ADDR_UNSPEC) ||
			     !(lep->le_ep.caps & FI_DIRECTED_RECV) ?
			     FI_ADDR_UNSPEC : encoded_addr;
	match_attr.lm_ignore = ignore;
	match_attr.lm_tag = tag;

	if (flags & FI_PEEK)
		return lnx_peek(lep, &match_attr, context, flags);

	if (flags & FI_DISCARD) {
		rx_entry = (struct lnx_rx_entry *)
			(((struct fi_context *)context)->internal[0]);
		return lnx_discard(lep, rx_entry, context);
	}

	if (flags & FI_CLAIM) {
		rx_entry = (struct lnx_rx_entry *)
			   (((struct fi_context *)context)->internal[0]);
	} else {
		rx_entry = lnx_remove_first_match(&lnx_srq->lps_trecv.lqp_unexq,
						&match_attr);
		if (!rx_entry) {
			FI_DBG(&lnx_prov, FI_LOG_CORE,
			       "addr=%" PRIx64 " tag=%" PRIx64 " ignore=%"
			       PRIx64 " buf=%p len=%zu not found\n",
			       addr, tag, ignore, iov->iov_base, iov->iov_len);
			goto nomatch;
		}
	}

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr=%" PRIx64 " tag=%" PRIx64 " ignore=%" PRIx64
	       " buf=%p len=%zu found\n",
	       addr, tag, ignore, iov->iov_base, iov->iov_len);

	/* match is found in the unexpected queue. call into the core
	 * provider to complete this message
	 */
	cep = rx_entry->rx_cep;
	sub_addr = lnx_get_core_addr(cep, rx_entry->rx_entry.addr);
	lnx_init_rx_entry(rx_entry, iov, desc, count, sub_addr, tag, ignore,
			  context, rx_entry->rx_entry.flags | flags);
	rx_entry->rx_entry.msg_size = MIN(ofi_total_iov_len(iov, count),
				          rx_entry->rx_entry.msg_size);

	if (desc) {
		rc = lnx_mr_regattr_core(cep->cep_domain, desc, count,
					 rx_entry->rx_entry.desc);
		if (rc)
			return rc;
	}

	lnx_set_send_pair[lep->le_mr](lep, cep, match_attr.lm_addr);

	if (tagged)
		rc = cep->cep_srx.peer_ops->start_tag(&rx_entry->rx_entry);
	else
		rc = cep->cep_srx.peer_ops->start_msg(&rx_entry->rx_entry);

	if (rc == -FI_EINPROGRESS) {
		/* this is telling me that more messages can match the same
		 * rx_entry. So keep it on the queue
		 */
		FI_DBG(&lnx_prov, FI_LOG_CORE,
		       "addr = %" PRIx64 " tag = %" PRIx64 " ignore = %" PRIx64
		       " start_tag() in progress\n",
		       addr, tag, ignore);

		goto insert_recvq;
	}
	if (rc)
		FI_WARN(&lnx_prov, FI_LOG_CORE,
			"start tag failed with %d\n", rc);

	FI_DBG(&lnx_prov, FI_LOG_CORE,
	       "addr = %" PRIx64 " tag = %" PRIx64 " ignore = %" PRIx64
	       " start_tag() success\n",
	       addr, tag, ignore);

	return 0;

nomatch:
	/* nothing on the unexpected queue, then allocate one and put it on
	 * the receive queue
	 */
	rx_entry = get_rx_entry(lep, iov, desc, count, match_attr.lm_addr,
				tag, ignore, context, flags);
	if (!rx_entry) {
		rc = -FI_ENOMEM;
		goto out;
	}

insert_recvq:
	lnx_insert_rx_entry(&lnx_srq->lps_trecv.lqp_recvq, rx_entry);

out:
	return rc;
}
