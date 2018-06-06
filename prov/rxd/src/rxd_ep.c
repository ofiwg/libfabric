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
#include "rxd.h"

static uint32_t rxd_flags(uint64_t fi_flags)
{
	uint32_t rxd_flags = 0;

	if (fi_flags & FI_REMOTE_CQ_DATA)
		rxd_flags |= RXD_REMOTE_CQ_DATA;
	if (fi_flags & FI_INJECT)
		rxd_flags |= RXD_INJECT;

	return rxd_flags;
}

struct rxd_pkt_entry *rxd_get_tx_pkt(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ep->do_local_mr ?
		    util_buf_alloc_ex(ep->tx_pkt_pool, &mr) :
		    util_buf_alloc(ep->tx_pkt_pool);

	pkt_entry->mr = (struct fid_mr *) mr;

	return pkt_entry;
}

static struct rxd_pkt_entry *rxd_get_rx_pkt(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ep->do_local_mr ?
		    util_buf_alloc_ex(ep->rx_pkt_pool, &mr) :
		    util_buf_alloc(ep->rx_pkt_pool);

	pkt_entry->mr = (struct fid_mr *) mr;

	return pkt_entry;
}

void rxd_release_tx_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt)
{
	util_buf_release(ep->tx_pkt_pool, pkt);
}

void rxd_release_rx_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt)
{
	util_buf_release(ep->rx_pkt_pool, pkt);
}

static int rxd_match_ctx(struct dlist_entry *item, const void *arg)
{
	struct rxd_x_entry *x_entry;

	x_entry = container_of(item, struct rxd_x_entry, entry);

	return (x_entry->cq_entry.op_context == arg);
}

static ssize_t rxd_ep_cancel(fid_t fid, void *context)
{
	struct rxd_ep *ep;
	struct dlist_entry *entry;
	struct rxd_x_entry *rx_entry;
	struct fi_cq_err_entry err_entry = {0};

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.lock);

	entry = dlist_remove_first_match(&ep->rx_list,
				&rxd_match_ctx, context);
	if (!entry)
		goto out;

	rx_entry = container_of(entry, struct rxd_x_entry, entry);

	rxd_rx_entry_free(ep, rx_entry);
	err_entry.op_context = rx_entry->cq_entry.op_context;
	err_entry.flags = (FI_MSG | FI_RECV);
	err_entry.err = FI_ECANCELED;
	err_entry.prov_errno = -FI_ECANCELED;
	rxd_cq_report_error(rxd_ep_rx_cq(ep), &err_entry);

out:
	fastlock_release(&ep->util_ep.lock);
	return 0;
}

static int rxd_ep_getopt(fid_t fid, int level, int optname,
		   void *optval, size_t *optlen)
{
	return -FI_ENOSYS;
}

static int rxd_ep_setopt(fid_t fid, int level, int optname,
		   const void *optval, size_t optlen)
{
	return -FI_ENOSYS;
}

struct fi_ops_ep rxd_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = rxd_ep_cancel,
	.getopt = rxd_ep_getopt,
	.setopt = rxd_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int rxd_match_unexp(struct dlist_entry *item, const void *arg)
{
	struct rxd_x_entry *rx_entry = (struct rxd_x_entry *) arg;
	struct rxd_pkt_entry *pkt_entry;

	pkt_entry = container_of(item, struct rxd_pkt_entry, d_entry);

	return rxd_match_addr(rx_entry->peer, pkt_entry->peer) &&
	       rxd_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     pkt_entry->pkt->ctrl.tag);
}

static void rxd_ep_check_unexp_msg_list(struct rxd_ep *ep, struct dlist_entry *list,
					struct rxd_x_entry *rx_entry)
{
	struct dlist_entry *match;
	struct rxd_pkt_entry *pkt_entry;

	match = dlist_remove_first_match(list, &rxd_match_unexp,
					 (void *) rx_entry);
	if (match) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "progressing unexp msg entry\n");
		dlist_remove(&rx_entry->entry);
		dlist_insert_tail(&rx_entry->entry, &ep->active_rx_list);

		pkt_entry = container_of(match, struct rxd_pkt_entry, d_entry);

		rxd_post_cts(ep, rx_entry, pkt_entry);

		rxd_release_rx_pkt(ep, pkt_entry);
		rxd_ep_post_buf(ep);
	}
}

static struct rxd_x_entry *rxd_rx_entry_init(struct rxd_ep *ep,
			const struct iovec *iov, size_t iov_count, uint64_t tag,
			uint64_t ignore, void *context, fi_addr_t addr,
			uint32_t op, uint64_t flags)
{
	struct rxd_x_entry *rx_entry;

	if (freestack_isempty(ep->rx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no-more rx entries\n");
		return NULL;
	}

	rx_entry = freestack_pop(ep->rx_fs);

	rx_entry->rx_id = rxd_rx_fs_index(ep->rx_fs, rx_entry);
	rx_entry->state = RXD_RTS;
	rx_entry->peer = addr;
	rx_entry->flags = rxd_flags(flags);
	rx_entry->bytes_done = 0;
	rx_entry->next_seg_no = 0;
	rx_entry->next_start = 0;
	rx_entry->window = 0;
	rx_entry->iov_count = iov_count;

	memcpy(rx_entry->iov, iov, sizeof(*rx_entry->iov) * iov_count);

	memset(&rx_entry->cq_entry, 0, sizeof(rx_entry->cq_entry));
	rx_entry->cq_entry.op_context = context;
	rx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	rx_entry->cq_entry.buf = iov[0].iov_base;
	rx_entry->cq_entry.flags = (FI_RECV | FI_MSG);
	if (op == ofi_op_tagged) {
		rx_entry->cq_entry.flags |= FI_TAGGED;
		rx_entry->cq_entry.tag = tag;
		rx_entry->ignore = ignore;
		dlist_insert_tail(&rx_entry->entry, &ep->rx_tag_list);
	} else {
		rx_entry->cq_entry.tag = 0;
		rx_entry->ignore = ~0;
		dlist_insert_tail(&rx_entry->entry, &ep->rx_list);
	}
	
	slist_init(&rx_entry->pkt_list);

	return rx_entry;
}

static ssize_t rxd_ep_generic_recvmsg(struct rxd_ep *rxd_ep, const struct iovec *iov,
				      size_t iov_count, fi_addr_t addr, uint64_t tag,
				      uint64_t ignore, void *context, uint32_t op,
				      uint64_t flags)
{
	ssize_t ret = 0;
	struct rxd_av *rxd_av;
	struct rxd_x_entry *rx_entry;
	struct dlist_entry *unexp_list;

	assert(iov_count <= RXD_IOV_LIMIT);

	rxd_av = rxd_ep_av(rxd_ep);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.rx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.rx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	rx_entry = rxd_rx_entry_init(rxd_ep, iov, iov_count, tag, ignore, context,
				(rxd_ep->util_ep.caps & FI_DIRECTED_RECV) ?
				rxd_av_dg_addr(rxd_av, addr) :
				FI_ADDR_UNSPEC, op, flags);
	if (!rx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	unexp_list = (op == ofi_op_tagged) ? &rxd_ep->unexp_tag_list :
		      &rxd_ep->unexp_list;
	if (!dlist_empty(unexp_list))
		rxd_ep_check_unexp_msg_list(rxd_ep, unexp_list, rx_entry);
out:
	fastlock_release(&rxd_ep->util_ep.rx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

static ssize_t rxd_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, msg->msg_iov, msg->iov_count,
				      msg->addr, 0, 0, msg->context, ofi_op_msg, flags);
}

static ssize_t rxd_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	return rxd_ep_generic_recvmsg(ep, &msg_iov, 1, src_addr, 0, 0, context,
				      ofi_op_msg, rxd_ep_rx_flags(ep));
}

static ssize_t rxd_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			     size_t count, fi_addr_t src_addr, void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, iov, count, src_addr,
				      0, 0, context, ofi_op_msg, rxd_ep_rx_flags(ep));
}

static inline void *rxd_mr_desc(struct fid_mr *mr, struct rxd_ep *ep)
{
	return (ep->do_local_mr) ? fi_mr_desc(mr) : NULL;
}

int rxd_ep_post_buf(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	ssize_t ret;

	pkt_entry = rxd_get_rx_pkt(ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	memset(pkt_entry, 0, sizeof(*pkt_entry));
	rxd_set_pkt(ep, pkt_entry);

	ret = fi_recv(ep->dg_ep, rxd_pkt_start(pkt_entry),
		      rxd_ep_domain(ep)->max_mtu_sz,
		      rxd_mr_desc(pkt_entry->mr, ep),
		      FI_ADDR_UNSPEC, &pkt_entry->context);
	if (ret) {
		rxd_release_rx_pkt(ep, pkt_entry);
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "failed to repost\n");
		return ret;
	}

	ep->posted_bufs++;
	slist_insert_tail(&pkt_entry->s_entry, &ep->rx_pkt_list);

	return 0;
}

static int rxd_ep_enable(struct rxd_ep *ep)
{
	size_t i;
	ssize_t ret;

	ret = fi_enable(ep->dg_ep);
	if (ret)
		return ret;

	fastlock_acquire(&ep->util_ep.lock);
	for (i = 0; i < ep->rx_size; i++) {
		ret = rxd_ep_post_buf(ep);
		if (ret)
			goto out;
	}
out:
	fastlock_release(&ep->util_ep.lock);
	return ret;
}

/*
 * Exponential back-off starting at 1ms, max 4s.
 */
void rxd_set_timeout(struct rxd_x_entry *x_entry)
{
	x_entry->retry_time = fi_gettime_ms() +
			      MIN(1 << (++x_entry->retry_cnt), 4000);
}

static void rxd_init_data_pkt(struct rxd_ep *ep,
			      struct rxd_x_entry *tx_entry,
			      struct rxd_pkt_entry *pkt_entry)
{
	uint32_t seg_size;

	seg_size = tx_entry->cq_entry.len - tx_entry->bytes_done;
	seg_size = MIN(tx_entry->seg_size, seg_size);

	rxd_set_pkt(ep, pkt_entry);
	pkt_entry->pkt->hdr.rx_id = tx_entry->rx_id;
	pkt_entry->pkt->hdr.seg_no = tx_entry->next_seg_no++;
	pkt_entry->pkt->hdr.tx_id = tx_entry->tx_id;
	pkt_entry->pkt->hdr.key = tx_entry->key;
	pkt_entry->pkt->hdr.peer = tx_entry->peer_x_addr;

	pkt_entry->pkt_size = ofi_copy_from_iov(&pkt_entry->pkt->data, seg_size,
						tx_entry->iov,
						tx_entry->iov_count,
						tx_entry->bytes_done);

	tx_entry->bytes_done += pkt_entry->pkt_size;
	if (tx_entry->bytes_done == tx_entry->cq_entry.len)
		pkt_entry->pkt->hdr.flags = RXD_LAST;
	else
		pkt_entry->pkt->hdr.flags = 0;

	pkt_entry->pkt_size += sizeof(struct rxd_pkt_hdr) + ep->prefix_size;
}

void rxd_init_ctrl_pkt(struct rxd_ep *ep, struct rxd_x_entry *x_entry,
		       struct rxd_pkt_entry *pkt_entry, uint32_t type)
{
	x_entry->retry_cnt = 0;
	rxd_set_pkt(ep, pkt_entry);
	pkt_entry->pkt->hdr.flags = x_entry->flags | RXD_CTRL;
	pkt_entry->pkt->hdr.tx_id = x_entry->tx_id;
	pkt_entry->pkt->hdr.rx_id = x_entry->rx_id;
	pkt_entry->pkt->hdr.key = x_entry->key;
	pkt_entry->pkt->hdr.peer = x_entry->peer;
	pkt_entry->pkt->ctrl.type = type;
	pkt_entry->pkt->ctrl.window = x_entry->window;
	pkt_entry->pkt->ctrl.version = RXD_PROTOCOL_VERSION;
	pkt_entry->pkt_size = RXD_CTRL_PKT_SIZE + ep->prefix_size;
}

struct rxd_x_entry *rxd_tx_entry_init(struct rxd_ep *ep, const struct iovec *iov,
				      size_t iov_count, uint64_t data, uint64_t tag,
				      void *context, fi_addr_t addr, uint32_t op,
				      uint64_t flags)
{
	struct rxd_x_entry *tx_entry;

	if (freestack_isempty(ep->tx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no-more tx entries\n");
		return NULL;
	}

	tx_entry = freestack_pop(ep->tx_fs);

	tx_entry->tx_id = rxd_tx_fs_index(ep->tx_fs, tx_entry);
	tx_entry->key = (++ep->key != ~0) ? ep->key : ++ep->key;

	tx_entry->state = RXD_RTS;
	tx_entry->peer = addr;
	tx_entry->flags = rxd_flags(flags);
	tx_entry->bytes_done = 0;
	tx_entry->next_seg_no = 0;
	tx_entry->next_start = 0;
	tx_entry->retry_cnt = 0;
	tx_entry->seg_size = rxd_ep_domain(ep)->max_seg_sz;
	tx_entry->iov_count = iov_count;
	memcpy(&tx_entry->iov[0], iov,
	       sizeof(*iov) * iov_count);

	memset(&tx_entry->cq_entry, 0, sizeof(tx_entry->cq_entry));
	if (flags & FI_REMOTE_CQ_DATA)
		tx_entry->cq_entry.data = data;
	tx_entry->cq_entry.op_context = context;
	tx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	tx_entry->cq_entry.buf = iov[0].iov_base;
	tx_entry->cq_entry.flags = (FI_TRANSMIT | FI_MSG);
	if (op == ofi_op_tagged) {
		tx_entry->cq_entry.flags |= FI_TAGGED;
		tx_entry->cq_entry.tag = tag;
	}

	tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len,
					  tx_entry->seg_size);
	tx_entry->window = MIN(tx_entry->num_segs, RXD_MAX_UNACKED);

	slist_init(&tx_entry->pkt_list);
	dlist_insert_tail(&tx_entry->entry, &ep->tx_list);

	return tx_entry;
}

void rxd_tx_entry_free(struct rxd_ep *ep, struct rxd_x_entry *tx_entry)
{
	rxd_ep_free_acked_pkts(ep, tx_entry, 0);
	tx_entry->state = RXD_FREE;
	tx_entry->key = ~0;
	dlist_remove(&tx_entry->entry);
	freestack_push(ep->tx_fs, tx_entry);
}

static ssize_t rxd_ep_post_data_msg(struct rxd_ep *ep,
				    struct rxd_x_entry *tx_entry, int try_send)
{
	struct rxd_pkt_entry *pkt_entry;

	pkt_entry = rxd_get_tx_pkt(ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	rxd_init_data_pkt(ep, tx_entry, pkt_entry);

	slist_insert_tail(&pkt_entry->s_entry, &tx_entry->pkt_list);
	return try_send ? rxd_ep_retry_pkt(ep, pkt_entry, tx_entry) : 0;
}

void rxd_ep_free_acked_pkts(struct rxd_ep *ep, struct rxd_x_entry *x_entry,
			    uint32_t last_acked)
{
	struct rxd_pkt_entry *pkt_entry;

	while (!slist_empty(&x_entry->pkt_list)) {
		pkt_entry = container_of(x_entry->pkt_list.head,
				   struct rxd_pkt_entry, s_entry);
		if (!rxd_is_ctrl_pkt(pkt_entry) &&
		    pkt_entry->pkt->hdr.seg_no > last_acked)
			break;
		slist_remove_head(&x_entry->pkt_list);
		rxd_release_tx_pkt(ep, pkt_entry);
	}
}

int rxd_ep_retry_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry,
		     struct rxd_x_entry *x_entry)
{
	return fi_send(ep->dg_ep, (const void *) rxd_pkt_start(pkt_entry),
		       pkt_entry->pkt_size, rxd_mr_desc(pkt_entry->mr, ep),
		       x_entry->peer, &pkt_entry->context);
}

void rxd_tx_entry_progress(struct rxd_ep *ep, struct rxd_x_entry *tx_entry,
			   int try_send)
{
	tx_entry->retry_cnt = 0;
	tx_entry->next_start += tx_entry->window;
	while (tx_entry->next_seg_no < tx_entry->next_start &&
	       tx_entry->bytes_done != tx_entry->cq_entry.len) {
		if (rxd_ep_post_data_msg(ep, tx_entry, try_send))
			break;
	}
	rxd_set_timeout(tx_entry);
}

static ssize_t rxd_ep_post_rts(struct rxd_ep *rxd_ep, struct rxd_x_entry *tx_entry)
{
	struct rxd_pkt_entry *pkt_entry;
	ssize_t ret;
	size_t addrlen;

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	rxd_init_ctrl_pkt(rxd_ep, tx_entry, pkt_entry, RXD_RTS);
	addrlen = RXD_NAME_LENGTH;
	memset(pkt_entry->pkt->source, 0, RXD_NAME_LENGTH);
	ret = fi_getname(&rxd_ep->dg_ep->fid, (void *) pkt_entry->pkt->source, &addrlen);
	if (ret) {
		rxd_release_tx_pkt(rxd_ep, pkt_entry);
		return ret;
	}

	pkt_entry->pkt->ctrl.size = tx_entry->cq_entry.len;
	pkt_entry->pkt->ctrl.data = tx_entry->cq_entry.data;
	pkt_entry->pkt->ctrl.tag = tx_entry->cq_entry.tag;
	pkt_entry->pkt->ctrl.seg_size = tx_entry->seg_size;

	if (tx_entry->cq_entry.flags & FI_TAGGED)
		pkt_entry->pkt->ctrl.op = ofi_op_tagged;
	else
		pkt_entry->pkt->ctrl.op = ofi_op_msg;

	slist_insert_tail(&pkt_entry->s_entry, &tx_entry->pkt_list);

	ret = rxd_ep_retry_pkt(rxd_ep, pkt_entry, tx_entry);
	rxd_set_timeout(tx_entry);

	return ret;
}

ssize_t rxd_ep_post_ack(struct rxd_ep *rxd_ep, struct rxd_x_entry *rx_entry)
{
	struct rxd_pkt_entry *pkt_entry;
	int ret;

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	rxd_init_ctrl_pkt(rxd_ep, rx_entry, pkt_entry, RXD_ACK);
	pkt_entry->pkt->hdr.seg_no = rx_entry->next_seg_no;

	pkt_entry->pkt->ctrl.window = rx_entry->window;

	ret = rxd_ep_retry_pkt(rxd_ep, pkt_entry, rx_entry);

	if (ret) {
		rxd_release_tx_pkt(rxd_ep, pkt_entry);
		return ret;
	}

	rx_entry->state = RXD_ACK;

	return 0;
}

static ssize_t rxd_ep_generic_inject(struct rxd_ep *rxd_ep, const struct iovec *iov,
				  size_t iov_count, fi_addr_t addr, uint64_t tag,
				  uint64_t data, uint32_t op, uint64_t flags)
{
	struct rxd_x_entry *tx_entry;
	fi_addr_t peer_addr;
	ssize_t ret;

	assert(ofi_total_iov_len(iov, iov_count) <= RXD_INJECT_SIZE);

	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), addr);

	fastlock_acquire(&rxd_ep->util_ep.lock);

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, data, tag, NULL,
				     peer_addr, op, flags | FI_INJECT);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (!(flags & (FI_COMPLETION | FI_INJECT_COMPLETE | FI_TRANSMIT_COMPLETE |
		FI_DELIVERY_COMPLETE)))
		tx_entry->flags |= RXD_NO_COMPLETION;

	ret = rxd_ep_post_rts(rxd_ep, tx_entry);
	if (ret) {
		rxd_tx_entry_free(rxd_ep, tx_entry);
		goto out;
	}

	rxd_tx_entry_progress(rxd_ep, tx_entry, 0);
out:
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

static ssize_t rxd_ep_generic_sendmsg(struct rxd_ep *rxd_ep, const struct iovec *iov,
				      size_t iov_count, fi_addr_t addr, uint64_t tag,
				      uint64_t data, void *context, uint32_t op,
				      uint64_t flags)
{
	struct rxd_x_entry *tx_entry;
	fi_addr_t peer_addr;
	ssize_t ret;

	assert(iov_count <= RXD_IOV_LIMIT);

	if (flags & FI_INJECT)
		return rxd_ep_generic_inject(rxd_ep, iov, iov_count, addr, tag, 0,
					     op, flags);

	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), addr);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.tx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, data, tag, context,
				     peer_addr, op, flags);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	ret = rxd_ep_post_rts(rxd_ep, tx_entry);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

static ssize_t rxd_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg, flags);

}

static ssize_t rxd_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, iov, count, dest_addr, 0,
				   0, context, ofi_op_msg, rxd_ep_tx_flags(ep));
}

static ssize_t rxd_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
			   fi_addr_t dest_addr, void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0,
				   0, context, ofi_op_msg, rxd_ep_tx_flags(ep));
}

static ssize_t rxd_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, 0, 0, ofi_op_msg, 0);
}

static ssize_t rxd_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
				uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0, data, context, ofi_op_msg,
				      FI_REMOTE_CQ_DATA);
}

static ssize_t rxd_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, 0, data, ofi_op_msg,
				     FI_REMOTE_CQ_DATA);
}

static struct fi_ops_msg rxd_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxd_ep_recv,
	.recvv = rxd_ep_recvv,
	.recvmsg = rxd_ep_recvmsg,
	.send = rxd_ep_send,
	.sendv = rxd_ep_sendv,
	.sendmsg = rxd_ep_sendmsg,
	.inject = rxd_ep_inject,
	.senddata = rxd_ep_senddata,
	.injectdata = rxd_ep_injectdata,
};

ssize_t rxd_ep_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return rxd_ep_generic_recvmsg(ep, &msg_iov, 1, src_addr, tag, ignore,
				      context, ofi_op_tagged, rxd_ep_tx_flags(ep));
}

ssize_t rxd_ep_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, iov, count, src_addr, tag, ignore,
				      context, ofi_op_tagged, rxd_ep_tx_flags(ep));
}

ssize_t rxd_ep_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, msg->msg_iov, msg->iov_count, msg->addr,
				      msg->tag, msg->ignore, msg->context,
				      ofi_op_tagged, flags);
}

ssize_t rxd_ep_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &msg_iov, 1, dest_addr, tag,
				      0, context, ofi_op_tagged,
				      rxd_ep_tx_flags(ep));
}

ssize_t rxd_ep_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
	void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, iov, count, dest_addr, tag,
				      0, context, ofi_op_tagged,
				      rxd_ep_tx_flags(ep));
}

ssize_t rxd_ep_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				      msg->addr, msg->tag, msg->data, msg->context,
				      ofi_op_tagged, flags);
}

ssize_t rxd_ep_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
		       fi_addr_t dest_addr, uint64_t tag)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, tag, 0,
				     ofi_op_tagged, 0);
}

ssize_t rxd_ep_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		         void *desc, uint64_t data, fi_addr_t dest_addr,
		         uint64_t tag, void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &iov, 1, dest_addr, tag, data, context,
				      ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

ssize_t rxd_ep_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			   uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, tag, data,
				     ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

static struct fi_ops_tagged rxd_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = rxd_ep_trecv,
	.recvv = rxd_ep_trecvv,
	.recvmsg = rxd_ep_trecvmsg,
	.send = rxd_ep_tsend,
	.sendv = rxd_ep_tsendv,
	.sendmsg = rxd_ep_tsendmsg,
	.inject = rxd_ep_tinject,
	.senddata = rxd_ep_tsenddata,
	.injectdata = rxd_ep_tinjectdata,
};

static void rxd_ep_free_res(struct rxd_ep *ep)
{

	if (ep->tx_fs)
		rxd_tx_fs_free(ep->tx_fs);

	if (ep->rx_fs)
		rxd_rx_fs_free(ep->rx_fs);

	util_buf_pool_destroy(ep->tx_pkt_pool);
	util_buf_pool_destroy(ep->rx_pkt_pool);
}

static int rxd_ep_close(struct fid *fid)
{
	int ret;
	struct rxd_ep *ep;
	struct rxd_pkt_entry *pkt_entry;
	struct slist_entry *entry;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);

	if (!dlist_empty(&ep->active_rx_list) || !dlist_empty(&ep->tx_list))
		return -FI_EBUSY;

	ret = fi_close(&ep->dg_ep->fid);
	if (ret)
		return ret;

	ret = fi_close(&ep->dg_cq->fid);
	if (ret)
		return ret;

	while (!slist_empty(&ep->rx_pkt_list)) {
		entry = slist_remove_head(&ep->rx_pkt_list);
		pkt_entry = container_of(entry, struct rxd_pkt_entry, s_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	while (!dlist_empty(&ep->unexp_list)) {
		dlist_pop_front(&ep->unexp_list, struct rxd_pkt_entry,
				pkt_entry, d_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	while (!dlist_empty(&ep->unexp_tag_list)) {
		dlist_pop_front(&ep->unexp_tag_list, struct rxd_pkt_entry,
				pkt_entry, d_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	if (ep->util_ep.tx_cq) {
		/* TODO: wait handling */
		fid_list_remove(&ep->util_ep.tx_cq->ep_list,
				&ep->util_ep.tx_cq->ep_list_lock,
				&ep->util_ep.ep_fid.fid);
	}

	if (ep->util_ep.rx_cq) {
		if (ep->util_ep.rx_cq != ep->util_ep.tx_cq) {
			/* TODO: wait handling */
			fid_list_remove(&ep->util_ep.rx_cq->ep_list,
					&ep->util_ep.rx_cq->ep_list_lock,
					&ep->util_ep.ep_fid.fid);
		}
	}

	rxd_ep_free_res(ep);
	ofi_endpoint_close(&ep->util_ep);
	free(ep);
	return 0;
}

static int rxd_ep_bind_cq(struct rxd_ep *ep, struct rxd_cq *cq, uint64_t flags)
{
	int ret;

	if (flags & ~(FI_TRANSMIT | FI_RECV)) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "unsupported flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && rxd_ep_tx_cq(ep)) ||
	    ((flags & FI_RECV) && rxd_ep_rx_cq(ep))) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "duplicate CQ binding\n");
		return -FI_EINVAL;
	}

	ret = fid_list_insert(&cq->util_cq.ep_list,
			      &cq->util_cq.ep_list_lock,
			      &ep->util_ep.ep_fid.fid);
	if (ret)
		return ret;

	if (flags & FI_TRANSMIT) {
		ep->util_ep.tx_cq = &cq->util_cq;
		ofi_atomic_inc32(&cq->util_cq.ref);
		/* TODO: wait handling */
	}

	if (flags & FI_RECV) {
		ep->util_ep.rx_cq = &cq->util_cq;
		ofi_atomic_inc32(&cq->util_cq.ref);
		/* TODO: wait handling */
	}

	return 0;
}

static int rxd_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxd_ep *ep;
	struct rxd_av *av;
	int ret = 0;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct rxd_av, util_av.av_fid.fid);
		ret = ofi_ep_bind_av(&ep->util_ep, &av->util_av);
		if (ret)
			return ret;

		ret = fi_ep_bind(ep->dg_ep, &av->dg_av->fid, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		ret = rxd_ep_bind_cq(ep, container_of(bfid, struct rxd_cq,
						       util_cq.cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
	case FI_CLASS_CNTR:
		return ofi_ep_bind(&ep->util_ep, bfid, flags);
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int rxd_ep_control(struct fid *fid, int command, void *arg)
{
	int ret;
	struct rxd_ep *ep;

	switch (command) {
	case FI_ENABLE:
		ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
		ret = rxd_ep_enable(ep);
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}
	return ret;
}

static struct fi_ops rxd_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_ep_close,
	.bind = rxd_ep_bind,
	.control = rxd_ep_control,
	.ops_open = fi_no_ops_open,
};

static int rxd_ep_cm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxd_ep *ep;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	return fi_setname(&ep->dg_ep->fid, addr, addrlen);
}

static int rxd_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxd_ep *ep;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	return fi_getname(&ep->dg_ep->fid, addr, addrlen);
}

struct fi_ops_cm rxd_ep_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxd_ep_cm_setname,
	.getname = rxd_ep_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static void rxd_ep_progress(struct util_ep *util_ep)
{
	struct dlist_entry *tx_item;
	struct slist_entry *pkt_item;
	struct fi_cq_err_entry err_entry = {0};
	struct rxd_x_entry *tx_entry;
	struct fi_cq_msg_entry cq_entry;
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_ep *ep;
	uint64_t current;
	ssize_t ret;
	int i;

	ep = container_of(util_ep, struct rxd_ep, util_ep);

	fastlock_acquire(&ep->util_ep.lock);
	for(ret = 1, i = 0;
	    ret > 0 && (!rxd_env.spin_count || i < rxd_env.spin_count);
	    i++) {
		ret = fi_cq_read(ep->dg_cq, &cq_entry, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (cq_entry.flags & FI_SEND)
			rxd_handle_send_comp(ep, &cq_entry);
		else if (cq_entry.flags & FI_RECV)
			rxd_handle_recv_comp(ep, &cq_entry);
		else
			assert(0);
	}

	current = fi_gettime_ms();

	if (rxd_env.ooo_rdm)
		goto out;

	dlist_foreach(&ep->tx_list, tx_item) {
		tx_entry = container_of(tx_item, struct rxd_x_entry, entry);

		if (current < tx_entry->retry_time)
			continue;

		if (tx_entry->retry_cnt > RXD_MAX_PKT_RETRY) {
			rxd_tx_entry_free(ep, tx_entry);
			err_entry.op_context = tx_entry->cq_entry.op_context;
			err_entry.flags = (FI_MSG | FI_SEND);
			err_entry.err = FI_ECONNREFUSED;
			err_entry.prov_errno = 0;
			rxd_cq_report_error(rxd_ep_tx_cq(ep), &err_entry);
		}

		//TODO - pkt list shouldn't be empty (progress/fail?)
		if (slist_empty(&tx_entry->pkt_list))
			continue;

		for (pkt_item = tx_entry->pkt_list.head; pkt_item;
		     pkt_item = pkt_item->next) {
			pkt_entry = container_of(pkt_item, struct rxd_pkt_entry,
						 s_entry);
			pkt_entry->pkt->hdr.flags |= RXD_RETRY;
			ret = rxd_ep_retry_pkt(ep, pkt_entry, tx_entry);
			if (ret || rxd_is_ctrl_pkt(pkt_entry))
				break;
		}
		rxd_set_timeout(tx_entry);
	}

out:
	while (ep->posted_bufs < ep->rx_size && !ret)
		ret = rxd_ep_post_buf(ep);

	fastlock_release(&ep->util_ep.lock);
}

static int rxd_buf_region_alloc_hndlr(void *pool_ctx, void *addr, size_t len,
				void **context)
{
	int ret;
	struct fid_mr *mr;
	struct rxd_domain *domain = pool_ctx;

	ret = fi_mr_reg(domain->dg_domain, addr, len,
			FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);
	*context = mr;
	return ret;
}

static void rxd_buf_region_free_hndlr(void *pool_ctx, void *context)
{
	fi_close((struct fid *) context);
}

int rxd_ep_init_res(struct rxd_ep *ep, struct fi_info *fi_info)
{
	int ret = util_buf_pool_create_ex(
		&ep->tx_pkt_pool,
		rxd_ep_domain(ep)->max_mtu_sz + sizeof(struct rxd_pkt_entry),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_TX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_ep_domain(ep));
	if (ret)
		return -FI_ENOMEM;

	ret = util_buf_pool_create_ex(
		&ep->rx_pkt_pool,
		rxd_ep_domain(ep)->max_mtu_sz + sizeof (struct rxd_pkt_entry),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_RX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_ep_domain(ep));
	if (ret)
		goto err;

	ep->tx_fs = rxd_tx_fs_create(ep->tx_size);
	if (!ep->tx_fs)
		goto err;
	ep->rx_fs = rxd_rx_fs_create(ep->rx_size);
	if (!ep->rx_fs)
		goto err;

	dlist_init(&ep->tx_list);
	dlist_init(&ep->rx_list);
	dlist_init(&ep->rx_tag_list);
	dlist_init(&ep->active_rx_list);
	dlist_init(&ep->unexp_list);
	dlist_init(&ep->unexp_tag_list);
	slist_init(&ep->rx_pkt_list);

	return 0;
err:
	if (ep->tx_pkt_pool)
		util_buf_pool_destroy(ep->tx_pkt_pool);

	if (ep->rx_pkt_pool)
		util_buf_pool_destroy(ep->rx_pkt_pool);

	if (ep->tx_fs)
		rxd_tx_fs_free(ep->tx_fs);

	if (ep->rx_fs)
		rxd_rx_fs_free(ep->rx_fs);

	return -FI_ENOMEM;
}

int rxd_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	struct fi_info *dg_info;
	struct rxd_domain *rxd_domain;
	struct rxd_ep *rxd_ep;
	struct fi_cq_attr cq_attr;
	int ret;

	rxd_ep = calloc(1, sizeof(*rxd_ep));
	if (!rxd_ep)
		return -FI_ENOMEM;

	rxd_domain = container_of(domain, struct rxd_domain,
				  util_domain.domain_fid);
	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = FI_WAIT_FD;

	ret = ofi_endpoint_init(domain, &rxd_util_prov, info, &rxd_ep->util_ep,
				context, rxd_ep_progress);
	if (ret)
		goto err1;

	ret = ofi_get_core_info(rxd_domain->util_domain.fabric->fabric_fid.api_version,
				NULL, NULL, 0, &rxd_util_prov, info,
				rxd_info_to_core, &dg_info);
	if (ret)
		goto err2;

	rxd_ep->do_local_mr = (rxd_domain->mr_mode & FI_MR_LOCAL) ? 1 : 0;

	ret = fi_endpoint(rxd_domain->dg_domain, dg_info, &rxd_ep->dg_ep, rxd_ep);
	cq_attr.size = dg_info->tx_attr->size + dg_info->rx_attr->size;
	fi_freeinfo(dg_info);
	if (ret)
		goto err2;

	ret = fi_cq_open(rxd_domain->dg_domain, &cq_attr, &rxd_ep->dg_cq, rxd_ep);
	if (ret)
		goto err3;

	ret = fi_ep_bind(rxd_ep->dg_ep, &rxd_ep->dg_cq->fid,
			 FI_TRANSMIT | FI_RECV);
	if (ret)
		goto err4;

	rxd_ep->rx_size = info->rx_attr->size;
	rxd_ep->tx_size = info->tx_attr->size;
	rxd_ep->prefix_size = info->ep_attr->msg_prefix_size;
	ret = rxd_ep_init_res(rxd_ep, info);
	if (ret)
		goto err4;

	rxd_ep->util_ep.ep_fid.fid.ops = &rxd_ep_fi_ops;
	rxd_ep->util_ep.ep_fid.cm = &rxd_ep_cm;
	rxd_ep->util_ep.ep_fid.ops = &rxd_ops_ep;
	rxd_ep->util_ep.ep_fid.msg = &rxd_ops_msg;
	rxd_ep->util_ep.ep_fid.tagged = &rxd_ops_tagged;

	*ep = &rxd_ep->util_ep.ep_fid;
	return 0;

err4:
	fi_close(&rxd_ep->dg_cq->fid);
err3:
	fi_close(&rxd_ep->dg_ep->fid);
err2:
	ofi_endpoint_close(&rxd_ep->util_ep);
err1:
	free(rxd_ep);
	return ret;
}
