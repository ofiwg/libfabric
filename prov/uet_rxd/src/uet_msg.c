/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <ofi_mem.h>
#include <ofi_iov.h>
#include "uet.h"

static int uet_match_unexp(struct dlist_entry *item, const void *arg)
{
	struct uet_match_attr *attr = (struct uet_match_attr *) arg;
	struct uet_unexp_msg *unexp_msg;

	unexp_msg = container_of(item, struct uet_unexp_msg, entry);

	if (!uet_match_addr(attr->peer, unexp_msg->base_hdr->peer))
		return 0;

	if (!unexp_msg->tag_hdr)
		return 1;

	return uet_match_tag(attr->tag, attr->ignore,
			     unexp_msg->tag_hdr->tag);
}

static struct uet_unexp_msg *uet_ep_check_unexp_list(struct dlist_entry *list,
				fi_addr_t addr, uint64_t tag, uint64_t ignore)
{
	struct uet_match_attr attr;
	struct dlist_entry *match;

	attr.peer = addr;
	attr.tag = tag;
	attr.ignore = ignore;

	match = dlist_find_first_match(list, &uet_match_unexp, &attr);
	if (!match)
		return NULL;

	FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Matched to unexp msg entry\n");

	return container_of(match, struct uet_unexp_msg, entry);
}

static void uet_progress_unexp_msg(struct uet_ep *ep, struct uet_x_entry *rx_entry,
				   struct uet_unexp_msg *unexp_msg)
{
	struct uet_pkt_entry *pkt_entry;
	uint64_t num_segs = 0;
	uint16_t curr_id = uet_peer(ep, unexp_msg->base_hdr->peer)->curr_rx_id;

	uet_progress_op(ep, rx_entry, unexp_msg->pkt_entry, unexp_msg->base_hdr,
			unexp_msg->sar_hdr, unexp_msg->tag_hdr,
			unexp_msg->data_hdr, NULL, NULL, &unexp_msg->msg,
			unexp_msg->msg_size);

	while (!dlist_empty(&unexp_msg->pkt_list)) {
		dlist_pop_front(&unexp_msg->pkt_list, struct uet_pkt_entry,
				pkt_entry, d_entry);
		uet_ep_recv_data(ep, rx_entry, (struct uet_data_pkt *)
				 (pkt_entry->pkt), pkt_entry->pkt_size);
		ofi_buf_free(pkt_entry);
		num_segs++;
	}

	if (uet_peer(ep, unexp_msg->base_hdr->peer)->curr_unexp) {
		if (!unexp_msg->sar_hdr || num_segs == unexp_msg->sar_hdr->num_segs - 1)
			uet_peer(ep, unexp_msg->base_hdr->peer)->curr_rx_id = curr_id;
		else
			uet_peer(ep, unexp_msg->base_hdr->peer)->curr_unexp = NULL;
	}

	uet_free_unexp_msg(unexp_msg);
}

static int uet_progress_unexp_list(struct uet_ep *ep,
				   struct dlist_entry *unexp_list,
				   struct dlist_entry *rx_list,
				   struct uet_x_entry *rx_entry)
{
	struct uet_x_entry *progress_entry, *dup_entry = NULL;
	struct uet_unexp_msg *unexp_msg;
	size_t total_size;

	while (!dlist_empty(unexp_list)) {
		unexp_msg = uet_ep_check_unexp_list(unexp_list, rx_entry->peer,
					rx_entry->cq_entry.tag, rx_entry->ignore);
		if (!unexp_msg)
			return 0;

		total_size = unexp_msg->sar_hdr ? unexp_msg->sar_hdr->size :
			     unexp_msg->msg_size;

		if (rx_entry->flags & UET_MULTI_RECV)
			dup_entry = uet_progress_multi_recv(ep, rx_entry, total_size);

		progress_entry = dup_entry ? dup_entry : rx_entry;
		progress_entry->cq_entry.len = MIN(rx_entry->cq_entry.len, total_size);
		uet_progress_unexp_msg(ep, progress_entry, unexp_msg);
		if (!dup_entry)
			return 1;
	}

	return 0;
}

static int uet_ep_discard_recv(struct uet_ep *uet_ep, void *context,
			       struct uet_unexp_msg *unexp_msg)
{
	uint64_t seq = unexp_msg->base_hdr->seq_no;
	int ret;

	assert(unexp_msg->tag_hdr);
	seq += unexp_msg->sar_hdr ? unexp_msg->sar_hdr->num_segs : 1;

	uet_peer(uet_ep, unexp_msg->base_hdr->peer)->rx_seq_no =
			MAX(seq, uet_peer(uet_ep,
				 unexp_msg->base_hdr->peer)->rx_seq_no);
	uet_ep_send_ack(uet_ep, unexp_msg->base_hdr->peer);

	ret = ofi_cq_write(uet_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			   0, NULL, unexp_msg->data_hdr ?
			   unexp_msg->data_hdr->cq_data : 0,
			   unexp_msg->tag_hdr->tag);

	uet_cleanup_unexp_msg(unexp_msg);

	return ret;
}

static int uet_peek_recv(struct uet_ep *uet_ep, fi_addr_t addr, uint64_t tag,
			 uint64_t ignore, void *context, uint64_t flags,
			 struct dlist_entry *unexp_list)
{
	struct uet_unexp_msg *unexp_msg;

	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	uet_ep_progress(&uet_ep->util_ep);
	ofi_genlock_lock(&uet_ep->util_ep.lock);

	unexp_msg = uet_ep_check_unexp_list(unexp_list, addr, tag, ignore);
	if (!unexp_msg) {
		FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Message not found\n");
		return ofi_cq_write_error_peek(uet_ep->util_ep.rx_cq, tag,
					       context);
	}
	FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Message found\n");

	assert(unexp_msg->tag_hdr);
	if (flags & FI_DISCARD)
		return uet_ep_discard_recv(uet_ep, context, unexp_msg);

	if (flags & FI_CLAIM) {
		FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Marking message for CLAIM\n");
		((struct fi_context *)context)->internal[0] = unexp_msg;
		dlist_remove(&unexp_msg->entry);
	}

	return ofi_cq_write(uet_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			    unexp_msg->sar_hdr ? unexp_msg->sar_hdr->size :
			    unexp_msg->msg_size, NULL, unexp_msg->data_hdr ?
			    unexp_msg->data_hdr->cq_data : 0,
			    unexp_msg->tag_hdr->tag);
}

ssize_t uet_ep_generic_recvmsg(struct uet_ep *uet_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t ignore, void *context, uint32_t op,
			       uint32_t uet_flags, uint64_t flags)
{
	ssize_t ret = 0;
	struct uet_x_entry *rx_entry;
	struct dlist_entry *unexp_list, *rx_list;
	struct uet_unexp_msg *unexp_msg;
	fi_addr_t uet_addr = UET_ADDR_INVALID;


	assert(iov_count <= UET_IOV_LIMIT);
	assert(!(uet_flags & UET_MULTI_RECV) || iov_count == 1);
	assert(!(flags & FI_PEEK) || op == UET_TAGGED);


	ofi_genlock_lock(&uet_ep->util_ep.lock);

	if (ofi_cirque_isfull(uet_ep->util_ep.rx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (op == UET_TAGGED) {
		unexp_list = &uet_ep->unexp_tag_list;
		rx_list = &uet_ep->rx_tag_list;
	} else {
		unexp_list = &uet_ep->unexp_list;
		rx_list = &uet_ep->rx_list;
	}

	if (uet_ep->util_ep.caps & FI_DIRECTED_RECV &&
	    addr != FI_ADDR_UNSPEC) {
		uet_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(uet_ep)->fi_addr_idx),
						     UET_IDX_OFFSET((int)addr));
	}

	if (flags & FI_PEEK) {
		ret = uet_peek_recv(uet_ep, uet_addr, tag, ignore, context, flags,
				    unexp_list);
		goto out;
	}
	if (!(flags & FI_DISCARD)) {

		rx_entry = uet_rx_entry_init(uet_ep, iov, iov_count, tag, ignore,
					     context, uet_addr, op, uet_flags);
		if (!rx_entry) {
			ret = -FI_EAGAIN;
		} else if (flags & FI_CLAIM) {
			FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Claiming message\n");
			unexp_msg = (struct uet_unexp_msg *)
				(((struct fi_context *) context)->internal[0]);
			uet_progress_unexp_msg(uet_ep, rx_entry, unexp_msg);
		} else if (!uet_progress_unexp_list(uet_ep, unexp_list,
			   rx_list, rx_entry)) {
			dlist_insert_tail(&rx_entry->entry, rx_list);
		}
		goto out;
	}

	assert(flags & FI_CLAIM);
	FI_DBG(&uet_prov, FI_LOG_EP_CTRL, "Discarding message\n");
	unexp_msg = (struct uet_unexp_msg *)
			(((struct fi_context *) context)->internal[0]);
	ret = uet_ep_discard_recv(uet_ep, context, unexp_msg);

out:
	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	return ret;
}

static ssize_t uet_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_ep_generic_recvmsg(ep, msg->msg_iov, msg->iov_count,
				      msg->addr, 0, ~0ULL, msg->context, UET_MSG,
				      uet_rx_flags(flags | ep->util_ep.rx_msg_flags),
				      flags);
}

static ssize_t uet_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			   fi_addr_t src_addr, void *context)
{
	struct uet_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	return uet_ep_generic_recvmsg(ep, &msg_iov, 1, src_addr, 0, ~0ULL, context,
				      UET_MSG, ep->rx_flags, 0);
}

static ssize_t uet_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t src_addr, void *context)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_ep_generic_recvmsg(ep, iov, count, src_addr,
				      0, ~0ULL, context, UET_MSG, ep->rx_flags, 0);
}

static struct uet_x_entry *uet_tx_entry_init_msg(struct uet_ep *ep, fi_addr_t addr,
					uint32_t op, const struct iovec *iov,
					size_t iov_count, uint64_t tag,
					uint64_t data, uint32_t flags, void *context)
{
	struct uet_x_entry *tx_entry;
	struct uet_domain *uet_domain = uet_ep_domain(ep);
	size_t max_inline;
	struct uet_base_hdr *base_hdr;
	void *ptr;

	tx_entry = uet_tx_entry_init_common(ep, addr, op, iov, iov_count,
					    tag, data, flags, context, &base_hdr, &ptr);
	if (!tx_entry)
		return NULL;

	max_inline = uet_domain->max_inline_msg;

	if (tx_entry->flags & UET_TAG_HDR) {
		max_inline -= sizeof(tx_entry->cq_entry.tag);
		uet_init_tag_hdr(&ptr, tx_entry);
	}
	uet_check_init_cq_data(&ptr, tx_entry, &max_inline);

	if (tx_entry->cq_entry.len > max_inline) {
		max_inline -= sizeof(struct uet_sar_hdr);
		tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len - max_inline,
						  uet_domain->max_seg_sz) + 1;
		uet_init_sar_hdr(&ptr, tx_entry, 0);
	} else {
		tx_entry->flags |= UET_INLINE;
		base_hdr->flags = (uint16_t) tx_entry->flags;
		tx_entry->num_segs = 1;
	}

	tx_entry->bytes_done = uet_init_msg(&ptr, tx_entry->iov,
					    tx_entry->iov_count,
					    tx_entry->cq_entry.len,
					    max_inline);

	tx_entry->pkt->pkt_size = uet_pkt_size(ep, base_hdr, ptr);

	return tx_entry;
}

ssize_t uet_ep_generic_inject(struct uet_ep *uet_ep, const struct iovec *iov,
			      size_t iov_count, fi_addr_t addr, uint64_t tag,
			      uint64_t data, uint32_t op, uint32_t uet_flags)
{
	struct uet_x_entry *tx_entry;
	ssize_t ret = -FI_EAGAIN;
	fi_addr_t uet_addr;

	assert(iov_count <= UET_IOV_LIMIT);
	assert(ofi_total_iov_len(iov, iov_count) <=
	       (size_t) uet_ep_domain(uet_ep)->max_inline_msg);

	ofi_genlock_lock(&uet_ep->util_ep.lock);

	if (ofi_cirque_isfull(uet_ep->util_ep.tx_cq->cirq))
		goto out;

	uet_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(uet_ep)->fi_addr_idx),
					     UET_IDX_OFFSET((int) addr));
	if (!uet_addr)
		goto out;

	ret = uet_send_rts_if_needed(uet_ep, uet_addr);
	if (ret)
		goto out;

	tx_entry = uet_tx_entry_init_msg(uet_ep, uet_addr, op, iov, iov_count,
					 tag, data, uet_flags, NULL);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (uet_peer(uet_ep, uet_addr)->peer_addr != UET_ADDR_INVALID)
		(void) uet_start_xfer(uet_ep, tx_entry);

out:
	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	return ret;
}

ssize_t uet_ep_generic_sendmsg(struct uet_ep *uet_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t data, void *context, uint32_t op,
			       uint32_t uet_flags)
{
	struct uet_x_entry *tx_entry;
	ssize_t ret = -FI_EAGAIN;
	fi_addr_t uet_addr;

	assert(iov_count <= UET_IOV_LIMIT);

	if (uet_flags & UET_INJECT)
		return uet_ep_generic_inject(uet_ep, iov, iov_count, addr, tag, 0,
					     op, uet_flags);

	ofi_genlock_lock(&uet_ep->util_ep.lock);

	if (ofi_cirque_isfull(uet_ep->util_ep.tx_cq->cirq))
		goto out;

	uet_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(uet_ep)->fi_addr_idx),
					     UET_IDX_OFFSET((int) addr));
	if (!uet_addr)
		goto out;

	ret = uet_send_rts_if_needed(uet_ep, uet_addr);
	if (ret)
		goto out;

	tx_entry = uet_tx_entry_init_msg(uet_ep, uet_addr, op, iov, iov_count,
					 tag, data, uet_flags, context);
	if (!tx_entry)
		goto out;

	if (uet_peer(uet_ep, uet_addr)->peer_addr == UET_ADDR_INVALID)
		goto out;

	ret = uet_start_xfer(uet_ep, tx_entry);
	if (ret && tx_entry->num_segs > 1)
		(void) uet_ep_post_data_pkts(uet_ep, tx_entry);

	ret = 0;
out:
	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	return ret;
}

static ssize_t uet_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_ep_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   UET_MSG, uet_tx_flags(flags |
				   ep->util_ep.tx_msg_flags));

}

static ssize_t uet_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_ep_generic_sendmsg(ep, iov, count, dest_addr, 0,
				      0, context, UET_MSG,
				      ep->tx_flags);
}

static ssize_t uet_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	struct uet_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return uet_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0,
				      0, context, UET_MSG,
				      ep->tx_flags);
}

static ssize_t uet_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
	struct uet_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return uet_ep_generic_inject(ep, &iov, 1, dest_addr, 0, 0, UET_MSG,
				     UET_NO_TX_COMP | UET_INJECT);
}

static ssize_t uet_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       void *context)
{
	struct uet_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return uet_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0, data, context,
				      UET_MSG, ep->tx_flags |
				      UET_REMOTE_CQ_DATA);
}

static ssize_t uet_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr)
{
	struct uet_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return uet_ep_generic_inject(ep, &iov, 1, dest_addr, 0, data, UET_MSG,
				     UET_NO_TX_COMP | UET_INJECT |
				     UET_REMOTE_CQ_DATA);
}

struct fi_ops_msg uet_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = uet_ep_recv,
	.recvv = uet_ep_recvv,
	.recvmsg = uet_ep_recvmsg,
	.send = uet_ep_send,
	.sendv = uet_ep_sendv,
	.sendmsg = uet_ep_sendmsg,
	.inject = uet_ep_inject,
	.senddata = uet_ep_senddata,
	.injectdata = uet_ep_injectdata,
};
