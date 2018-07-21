/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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
#include <inttypes.h>

#include "ofi.h"
#include "ofi_iov.h"

#include "rxm.h"

static struct rxm_conn *rxm_key2conn(struct rxm_ep *rxm_ep, uint64_t key)
{
	struct util_cmap_handle *handle;
	handle = ofi_cmap_key2handle(rxm_ep->util_ep.cmap, key);
	if (!handle)
		return NULL;
	return container_of(handle, struct rxm_conn, handle);
}

static const char *rxm_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
		const void *err_data, char *buf, size_t len)
{
	struct util_cq *cq;
	struct rxm_ep *rxm_ep;
	struct fid_list_entry *fid_entry;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	fid_entry = container_of(cq->ep_list.next, struct fid_list_entry, entry);
	rxm_ep = container_of(fid_entry->fid, struct rxm_ep, util_ep.ep_fid);

	return fi_cq_strerror(rxm_ep->msg_cq, prov_errno, err_data, buf, len);
}

/* Get a match_iov derived from iov whose size matches given length */
static int rxm_match_iov(const struct iovec *iov, void **desc,
			 uint8_t count, uint64_t offset, size_t match_len,
			 struct rxm_iov *match_iov)
{
	uint8_t i;

	assert(count <= RXM_IOV_LIMIT);

	for (i = 0; i < count; i++) {
		if (offset >= iov[i].iov_len) {
			offset -= iov[i].iov_len;
			continue;
		}

		match_iov->iov[i].iov_base = (char *)iov[i].iov_base + offset;
		match_iov->iov[i].iov_len = MIN(iov[i].iov_len - offset, match_len);
		if (desc)
			match_iov->desc[i] = desc[i];

		match_len -= match_iov->iov[i].iov_len;
		if (!match_len)
			break;
		offset = 0;
	}

	if (match_len) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
			"Given iov size (%zu) < match_len (remained match_len = %zu)!\n",
			ofi_total_iov_len(iov, count), match_len);
		return -FI_ETOOSMALL;
	}

	match_iov->count = i + 1;
	return FI_SUCCESS;
}

static int rxm_finish_buf_recv(struct rxm_rx_buf *rx_buf)
{
	uint64_t flags = rx_buf->pkt.hdr.flags | FI_RECV;

	assert(rx_buf->pkt.hdr.size <= rx_buf->ep->sar_limit);

	if (rx_buf->pkt.hdr.size > rx_buf->ep->rxm_info->tx_attr->inject_size)
		flags |= FI_MORE;

	FI_DBG(&rxm_prov, FI_LOG_CQ, "writing buffered recv completion: "
	       "length: %" PRIu64 "\n", rx_buf->pkt.hdr.size);
	rx_buf->recv_context.ep = &rx_buf->ep->util_ep.ep_fid;

	return rxm_cq_write_recv_comp(rx_buf, &rx_buf->recv_context, flags,
				      rx_buf->pkt.hdr.size, rx_buf->pkt.data);
}

static int rxm_cq_write_error_trunc(struct rxm_rx_buf *rx_buf, size_t done_len)
{
	int ret;

	if (rx_buf->ep->util_ep.flags & OFI_CNTR_ENABLED)
		rxm_cntr_incerr(rx_buf->ep->util_ep.rx_cntr);

	FI_WARN(&rxm_prov, FI_LOG_CQ, "Message truncated: "
		"recv buf length: %zu message length: %" PRIu64 "\n",
		done_len, rx_buf->pkt.hdr.size);
	ret = ofi_cq_write_error_trunc(rx_buf->ep->util_ep.rx_cq,
				       rx_buf->recv_entry->context,
				       rx_buf->recv_entry->comp_flags |
				       rx_buf->pkt.hdr.flags,
				       rx_buf->pkt.hdr.size,
				       rx_buf->recv_entry->rxm_iov.iov[0].iov_base,
				       rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag,
				       rx_buf->pkt.hdr.size - done_len);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
			"Unable to write recv error CQ\n");
		return ret;
	}
	return 0;
}

static int rxm_finish_recv(struct rxm_rx_buf *rx_buf, size_t done_len)
{
	int ret;

	if (OFI_UNLIKELY(done_len < rx_buf->pkt.hdr.size)) {
		ret = rxm_cq_write_error_trunc(rx_buf, done_len);
		if (ret)
			return ret;
	} else {
		if (rx_buf->recv_entry->flags & FI_COMPLETION) {
			ret = rxm_cq_write_recv_comp(
					rx_buf, rx_buf->recv_entry->context,
					(rx_buf->recv_entry->comp_flags |
					 rx_buf->pkt.hdr.flags),
					rx_buf->pkt.hdr.size,
					rx_buf->recv_entry->rxm_iov.iov[0].iov_base);
			if (ret)
				return ret;
		}
		if (rx_buf->ep->util_ep.flags & OFI_CNTR_ENABLED)
			rxm_cntr_inc(rx_buf->ep->util_ep.rx_cntr);
	}

	rxm_enqueue_rx_buf_for_repost_check(rx_buf);

	if (rx_buf->recv_entry->flags & FI_MULTI_RECV) {
		struct rxm_iov rxm_iov;

		rx_buf->recv_entry->total_len -= rx_buf->pkt.hdr.size;

		if (rx_buf->recv_entry->total_len <= rx_buf->ep->min_multi_recv_size) {
			FI_DBG(&rxm_prov, FI_LOG_CQ,
			       "Buffer %p has been completely consumed. "
			       "Reporting Multi-Recv completion\n",
			       rx_buf->recv_entry->multi_recv_buf);
			ret = rxm_cq_write_recv_comp(rx_buf,
						     rx_buf->recv_entry->context,
						     FI_MULTI_RECV,
						     rx_buf->pkt.hdr.size,
						     rx_buf->recv_entry->multi_recv_buf);
			if (OFI_UNLIKELY(ret)) {
				FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to write FI_MULTI_RECV completion\n");
				return ret;
			}
			/* Since buffer is elapsed, release recv_entry */
			rxm_recv_entry_release(rx_buf->recv_entry->recv_queue,
					       rx_buf->recv_entry);
			return ret;
		}

		FI_DBG(&rxm_prov, FI_LOG_CQ,
		       "Repost Multi-Recv entry: "
		       "consumed len = %zu, remain len = %zu\n",
		       rx_buf->pkt.hdr.size,
		       rx_buf->recv_entry->total_len);

		rxm_iov = rx_buf->recv_entry->rxm_iov;
		ret = rxm_match_iov(rxm_iov.iov, rxm_iov.desc, rxm_iov.count,	/* prev iovecs */
				    rx_buf->pkt.hdr.size,			/* offset */
				    rx_buf->recv_entry->total_len,		/* match_len */
				    &rx_buf->recv_entry->rxm_iov);		/* match_iov */
		if (OFI_UNLIKELY(ret))
			return ret;

		return rxm_process_recv_entry(&rx_buf->ep->recv_queue, rx_buf->recv_entry);
	} else {
		rxm_recv_entry_release(rx_buf->recv_entry->recv_queue,
				       rx_buf->recv_entry);
	}

	return FI_SUCCESS;
}

static inline int rxm_finish_send(struct rxm_tx_entry *tx_entry)
{
	rxm_tx_buf_release(tx_entry->ep, tx_entry->tx_buf);
	return rxm_finish_send_nobuf(tx_entry);
}

static inline int rxm_finish_sar_segment_send(struct rxm_tx_buf *tx_buf)
{
	struct rxm_tx_entry *tx_entry = tx_buf->tx_entry;

	rxm_tx_buf_release(tx_entry->ep, tx_buf);
	/* If `segs_left` == 0, all segments of the message have been fully sent */
	if (!--tx_entry->segs_left) {
		return rxm_finish_send_nobuf(tx_entry);
	} else if (!dlist_empty(&tx_entry->deferred_tx_buf_list)) {
		ssize_t ret = 0;

		while (!dlist_empty(&tx_entry->deferred_tx_buf_list) && !ret) {
			tx_buf = container_of(tx_entry->deferred_tx_buf_list.next,
					      struct rxm_tx_buf, hdr.entry);
			ret = fi_send(tx_entry->conn->msg_ep, &tx_buf->pkt,
				      sizeof(tx_buf->pkt) + tx_buf->pkt.ctrl_hdr.seg_size,
				      tx_buf->hdr.desc, 0, tx_buf);
			if (!ret)
				dlist_remove(&tx_buf->hdr.entry);
		}
	}
	/* This branch takes true, when it's impossible to allocate TX buffer */
	if (OFI_UNLIKELY((tx_entry->msg_id == UINT64_MAX) &&
			 dlist_empty(&tx_entry->deferred_tx_buf_list))) {
		rxm_cq_write_error(tx_entry->ep->util_ep.tx_cq,
				   tx_entry->ep->util_ep.tx_cntr,
				   tx_entry->context, -FI_ENOMEM);
		rxm_tx_entry_release(&tx_entry->conn->send_queue, tx_entry);
	}
	return FI_SUCCESS;
}

static inline int rxm_finish_send_lmt_ack(struct rxm_rx_buf *rx_buf)
{
	RXM_LOG_STATE(FI_LOG_CQ, rx_buf->pkt, RXM_LMT_ACK_SENT, RXM_LMT_FINISH);
	rx_buf->hdr.state = RXM_LMT_FINISH;
	if (!rx_buf->ep->rxm_mr_local)
		rxm_ep_msg_mr_closev(rx_buf->mr, rx_buf->recv_entry->rxm_iov.count);
	return rxm_finish_recv(rx_buf, rx_buf->recv_entry->total_len);
}

static int rxm_lmt_tx_finish(struct rxm_tx_entry *tx_entry)
{
	int ret;

	RXM_LOG_STATE_TX(FI_LOG_CQ, tx_entry, RXM_LMT_FINISH);
	tx_entry->state = RXM_LMT_FINISH;

	if (!tx_entry->ep->rxm_mr_local)
		rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);

	ret = rxm_finish_send(tx_entry);
	if (ret)
		return ret;

	rxm_enqueue_rx_buf_for_repost_check(tx_entry->rx_buf);

	return ret;
}

static int rxm_lmt_handle_ack(struct rxm_rx_buf *rx_buf)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_conn *rxm_conn = rxm_key2conn(rx_buf->ep,
						 rx_buf->pkt.ctrl_hdr.conn_id);

	FI_DBG(&rxm_prov, FI_LOG_CQ, "Got ACK for msg_id: 0x%" PRIx64 "\n",
	       rx_buf->pkt.ctrl_hdr.msg_id);

	tx_entry = &rxm_conn->send_queue.fs->entry[rx_buf->pkt.ctrl_hdr.msg_id].buf;

	assert(tx_entry->tx_buf->pkt.ctrl_hdr.msg_id == rx_buf->pkt.ctrl_hdr.msg_id);

	tx_entry->rx_buf = rx_buf;

	if (tx_entry->state == RXM_LMT_ACK_WAIT) {
		return rxm_lmt_tx_finish(tx_entry);
	} else {
		assert(tx_entry->state == RXM_LMT_TX);
		RXM_LOG_STATE_TX(FI_LOG_CQ, tx_entry, RXM_LMT_ACK_RECVD);
		tx_entry->state = RXM_LMT_ACK_RECVD;
		return 0;
	}
}

static inline
ssize_t rxm_cq_handle_seg_data(struct rxm_rx_buf *rx_buf)
{
	uint64_t done_len = ofi_copy_to_iov(rx_buf->recv_entry->rxm_iov.iov,
					    rx_buf->recv_entry->rxm_iov.count,
					    rxm_sar_get_offset(&rx_buf->pkt.ctrl_hdr),
					    rx_buf->pkt.data,
					    rx_buf->pkt.ctrl_hdr.seg_size);
	rx_buf->recv_entry->total_recv_len += done_len;
	rx_buf->recv_entry->segs_rcvd++;
	/* Check that this isn't last segment */
	if (((rxm_sar_get_seg_type(&rx_buf->pkt.ctrl_hdr) != RXM_SAR_SEG_LAST) &&
	     (!rx_buf->recv_entry->last_seg_no ||
	      (rx_buf->recv_entry->last_seg_no != rx_buf->pkt.ctrl_hdr.seg_no))) &&
	/* and message isn't truncated */
	    (done_len == rx_buf->pkt.ctrl_hdr.seg_size)) {
		if (rx_buf->recv_entry->msg_id == UINT64_MAX) {
			rx_buf->conn = rxm_key2conn(rx_buf->ep,
						    rx_buf->pkt.ctrl_hdr.conn_id);
			rx_buf->recv_entry->msg_id = rx_buf->pkt.ctrl_hdr.msg_id;
			dlist_insert_tail(&rx_buf->recv_entry->sar_entry,
					  &rx_buf->conn->sar_rx_msg_list);
		}
		/* The RX buffer can be reposted for further re-use */
		rx_buf->recv_entry = NULL;
		rxm_enqueue_rx_buf_for_repost_check(rx_buf);
		return FI_SUCCESS;
	} else if ((rxm_sar_get_seg_type(&rx_buf->pkt.ctrl_hdr) == RXM_SAR_SEG_LAST) &&
		   (rx_buf->recv_entry->segs_rcvd != (rx_buf->pkt.ctrl_hdr.seg_no + 1))) {
		/* Handle case when the segment with the `RXM_SAR_SEG_LAST` flag
		 * is received, but not all segments were received yet. We should
		 * wait untill all segments will be received. */
		rx_buf->recv_entry->last_seg_no = rx_buf->pkt.ctrl_hdr.seg_no;
	}
	dlist_remove(&rx_buf->recv_entry->sar_entry);
	/* Mark rxm_recv_entry::msg_id as unknown for futher re-use */
	rx_buf->recv_entry->msg_id = UINT64_MAX;
	done_len = rx_buf->recv_entry->total_recv_len;
	rx_buf->recv_entry->total_recv_len = 0;
	rx_buf->recv_entry->segs_rcvd = 0;
	rx_buf->recv_entry->last_seg_no = 0;
	return rxm_finish_recv(rx_buf, done_len);
}

static inline
ssize_t rxm_cq_handle_large_data(struct rxm_rx_buf *rx_buf)
{
	size_t i, index = 0, offset = 0, count;
	struct iovec iov[RXM_IOV_LIMIT];
	void *desc[RXM_IOV_LIMIT];
	int ret = 0;

	if (!rx_buf->conn) {
		assert(rx_buf->ep->srx_ctx);
		rx_buf->conn = rxm_key2conn(rx_buf->ep,
					    rx_buf->pkt.ctrl_hdr.conn_id);
		if (OFI_UNLIKELY(!rx_buf->conn))
			return -FI_EOTHER;
	}
	assert(rx_buf->conn);

	FI_DBG(&rxm_prov, FI_LOG_CQ,
	       "Got incoming recv with msg_id: 0x%" PRIx64 "\n",
	       rx_buf->pkt.ctrl_hdr.msg_id);

	rx_buf->rma_iov = (struct rxm_rma_iov *)rx_buf->pkt.data;
	rx_buf->rma_iov_index = 0;

	if (!rx_buf->ep->rxm_mr_local) {
		ret = rxm_ep_msg_mr_regv(rx_buf->ep,
					 rx_buf->recv_entry->rxm_iov.iov,
					 rx_buf->recv_entry->rxm_iov.count,
					 FI_READ, rx_buf->mr);
		if (OFI_UNLIKELY(ret))
			return ret;

		for (i = 0; i < rx_buf->recv_entry->rxm_iov.count; i++)
			rx_buf->recv_entry->rxm_iov.desc[i] =
						fi_mr_desc(rx_buf->mr[i]);
	} else {
		for (i = 0; i < rx_buf->recv_entry->rxm_iov.count; i++)
			rx_buf->recv_entry->rxm_iov.desc[i] =
				fi_mr_desc(rx_buf->recv_entry->rxm_iov.desc[i]);
	}

	assert(rx_buf->rma_iov->count && (rx_buf->rma_iov->count <= RXM_IOV_LIMIT));

	RXM_LOG_STATE_RX(FI_LOG_CQ, rx_buf, RXM_LMT_READ);
	rx_buf->hdr.state = RXM_LMT_READ;

	for (i = 0; i < rx_buf->rma_iov->count; i++) {
		ret = ofi_copy_iov_desc(&iov[0], &desc[0], &count,
					&rx_buf->recv_entry->rxm_iov.iov[0],
					&rx_buf->recv_entry->rxm_iov.desc[0],
					rx_buf->recv_entry->rxm_iov.count,
					&index, &offset, rx_buf->rma_iov->iov[i].len);
		if (ret) {
			assert(ret == -FI_ETOOSMALL);
			return rxm_cq_write_error_trunc(
				rx_buf,
				ofi_total_iov_len(rx_buf->recv_entry->rxm_iov.iov,
						  rx_buf->recv_entry->rxm_iov.count));
		}
		ret = fi_readv(rx_buf->conn->msg_ep, iov, desc, count, 0,
			       rx_buf->rma_iov->iov[i].addr,
			       rx_buf->rma_iov->iov[i].key, rx_buf);
		if (ret) {
			rxm_cq_write_error(rx_buf->ep->util_ep.rx_cq,
					   rx_buf->ep->util_ep.rx_cntr,
					   rx_buf->recv_entry->context, ret);
			break;
		}
	}
	return ret;
}

static inline
ssize_t rxm_cq_handle_data(struct rxm_rx_buf *rx_buf)
{
	uint64_t done_len = ofi_copy_to_iov(rx_buf->recv_entry->rxm_iov.iov,
					    rx_buf->recv_entry->rxm_iov.count,
					    0, rx_buf->pkt.data,
					    rx_buf->pkt.hdr.size);
	return rxm_finish_recv(rx_buf, done_len);
}

ssize_t rxm_cq_handle_rx_buf(struct rxm_rx_buf *rx_buf)
{
	switch (rx_buf->pkt.ctrl_hdr.type) {
	case ofi_ctrl_data:
		return rxm_cq_handle_data(rx_buf);
	case ofi_ctrl_large_data:
		return rxm_cq_handle_large_data(rx_buf);
	case ofi_ctrl_seg_data:
		return rxm_cq_handle_seg_data(rx_buf);
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown message type\n");
		assert(0);
		return -FI_EINVAL;
	}
}

static inline ssize_t
rxm_cq_match_rx_buf(struct rxm_rx_buf *rx_buf,
		    struct rxm_recv_queue *recv_queue,
		    struct rxm_recv_match_attr *match_attr)
{
	struct dlist_entry *entry;
	struct rxm_ep *rxm_ep;
	struct fid_ep *msg_ep;

	rx_buf->ep->res_fastlock_acquire(&recv_queue->lock);
	entry = dlist_remove_first_match(&recv_queue->recv_list,
					 recv_queue->match_recv, match_attr);
	if (!entry) {
		RXM_DBG_ADDR_TAG(FI_LOG_CQ, "No matching recv found for "
				 "incoming msg", match_attr->addr,
				 match_attr->tag);
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Enqueueing msg to unexpected msg"
		       "queue\n");
		rx_buf->unexp_msg.addr = match_attr->addr;
		rx_buf->unexp_msg.tag = match_attr->tag;
		rx_buf->repost = 0;

		msg_ep = rx_buf->hdr.msg_ep;
		rxm_ep = rx_buf->ep;

		dlist_insert_tail(&rx_buf->unexp_msg.entry,
				  &recv_queue->unexp_msg_list);
		rx_buf->ep->res_fastlock_release(&recv_queue->lock);

		rx_buf = rxm_rx_buf_get(rxm_ep);
		if (!rx_buf)
			return -FI_ENOMEM;

		rx_buf->hdr.state = RXM_RX;
		rx_buf->hdr.msg_ep = msg_ep;
		rx_buf->repost = 1;

		rxm_enqueue_rx_buf_for_repost(rx_buf);
		return 0;
	}
	rx_buf->ep->res_fastlock_release(&recv_queue->lock);

	rx_buf->recv_entry = container_of(entry, struct rxm_recv_entry, entry);
	return rxm_cq_handle_rx_buf(rx_buf);
}

static inline ssize_t rxm_handle_recv_comp(struct rxm_rx_buf *rx_buf)
{
	struct rxm_recv_match_attr match_attr = {
		.addr = FI_ADDR_UNSPEC,
	};

	if (rx_buf->ep->rxm_info->caps & (FI_SOURCE | FI_DIRECTED_RECV)) {
		if (rx_buf->ep->srx_ctx)
			rx_buf->conn =
				rxm_key2conn(rx_buf->ep, rx_buf->pkt.ctrl_hdr.conn_id);
		if (OFI_UNLIKELY(!rx_buf->conn))
			return -FI_EOTHER;
		match_attr.addr = rx_buf->conn->handle.fi_addr;
	}

	if (rx_buf->ep->rxm_info->mode & FI_BUFFERED_RECV)
		return rxm_finish_buf_recv(rx_buf);

	switch(rx_buf->pkt.hdr.op) {
	case ofi_op_msg:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got MSG op\n");
		return rxm_cq_match_rx_buf(rx_buf, &rx_buf->ep->recv_queue,
					   &match_attr);
	case ofi_op_tagged:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got TAGGED op\n");
		match_attr.tag = rx_buf->pkt.hdr.tag;
		return rxm_cq_match_rx_buf(rx_buf, &rx_buf->ep->trecv_queue,
					   &match_attr);
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown op!\n");
		assert(0);
		return -FI_EINVAL;
	}
}

static int rxm_sar_match_msg_id(struct dlist_entry *item, const void *arg)
{
	uint64_t msg_id = *((uint64_t *)arg);
	struct rxm_recv_entry *recv_entry =
		container_of(item, struct rxm_recv_entry, sar_entry);
	return (msg_id == recv_entry->msg_id);
}

static inline
ssize_t rxm_sar_handle_segment(struct rxm_rx_buf *rx_buf)
{
	struct dlist_entry *sar_entry;

	rx_buf->conn = rxm_key2conn(rx_buf->ep,
				    rx_buf->pkt.ctrl_hdr.conn_id);
	if (OFI_UNLIKELY(!rx_buf->conn))
		return -FI_EOTHER;
	FI_DBG(&rxm_prov, FI_LOG_CQ,
	       "Got incoming recv with msg_id: 0x%" PRIx64 "for conn - %p\n",
	       rx_buf->pkt.ctrl_hdr.msg_id, rx_buf->conn);
	sar_entry = dlist_find_first_match(&rx_buf->conn->sar_rx_msg_list,
					   rxm_sar_match_msg_id,
					   &rx_buf->pkt.ctrl_hdr.msg_id);
	if (!sar_entry)
		return rxm_handle_recv_comp(rx_buf);
	rx_buf->recv_entry =
		container_of(sar_entry, struct rxm_recv_entry, sar_entry);
	return rxm_cq_handle_seg_data(rx_buf);
}

static ssize_t rxm_lmt_send_ack(struct rxm_rx_buf *rx_buf)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;
	ssize_t ret;

	assert(rx_buf->conn);

	tx_buf = rxm_tx_buf_get(rx_buf->ep, RXM_BUF_POOL_TX_ACK);
	if (OFI_UNLIKELY(!tx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "TX queue full!\n");
		return -FI_EAGAIN;
	}
	assert(tx_buf->pkt.ctrl_hdr.type == ofi_ctrl_ack);

	tx_entry = rxm_tx_entry_get(&rx_buf->conn->send_queue);
	if (OFI_UNLIKELY(!tx_entry)) {
		ret = -FI_EAGAIN;
		goto err1;
	}

	RXM_LOG_STATE(FI_LOG_CQ, rx_buf->pkt, RXM_LMT_READ, RXM_LMT_ACK_SENT);
	rx_buf->hdr.state = RXM_LMT_ACK_SENT;

	tx_entry->state 	= rx_buf->hdr.state;
	tx_entry->context 	= rx_buf;
	tx_entry->tx_buf 	= tx_buf;

	tx_buf->pkt.ctrl_hdr.conn_id 	= rx_buf->conn->handle.remote_key;
	tx_buf->pkt.ctrl_hdr.msg_id 	= rx_buf->pkt.ctrl_hdr.msg_id;

	ret = fi_send(rx_buf->conn->msg_ep, &tx_buf->pkt, sizeof(tx_buf->pkt),
		      tx_buf->hdr.desc, 0, tx_entry);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to send ACK\n");
		goto err2;
	}
	return 0;
err2:
	rxm_tx_entry_release(&rx_buf->conn->send_queue, tx_entry);
err1:
	rxm_tx_buf_release(rx_buf->ep, tx_buf);
	return ret;
}

static ssize_t rxm_lmt_send_ack_fast(struct rxm_rx_buf *rx_buf)
{
	struct rxm_pkt pkt;
	ssize_t ret;

	assert(rx_buf->conn);

	RXM_LOG_STATE(FI_LOG_CQ, rx_buf->pkt, RXM_LMT_READ, RXM_LMT_ACK_SENT);

	pkt.hdr.op		= ofi_op_msg;
	pkt.hdr.version		= OFI_OP_VERSION;
	pkt.ctrl_hdr.version	= RXM_CTRL_VERSION;
	pkt.ctrl_hdr.type	= ofi_ctrl_ack;
	pkt.ctrl_hdr.conn_id 	= rx_buf->conn->handle.remote_key;
	pkt.ctrl_hdr.msg_id 	= rx_buf->pkt.ctrl_hdr.msg_id;

	ret = fi_inject(rx_buf->conn->msg_ep, &pkt, sizeof(pkt), 0);
	if (OFI_UNLIKELY(ret)) {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
		       "fi_inject(ack pkt) for MSG provider failed\n");
		return ret;
	}

	return rxm_finish_send_lmt_ack(rx_buf);
}

static int rxm_handle_remote_write(struct rxm_ep *rxm_ep,
				   struct fi_cq_data_entry *comp)
{
	int ret;

	FI_DBG(&rxm_prov, FI_LOG_CQ, "writing remote write completion\n");
	ret = ofi_cq_write(rxm_ep->util_ep.rx_cq, NULL, comp->flags, 0, NULL,
			   comp->data, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to write remote write completion\n");
		return ret;
	}
	rxm_cntr_inc(rxm_ep->util_ep.rem_wr_cntr);
	if (comp->op_context)
		rxm_enqueue_rx_buf_for_repost_check(comp->op_context);
	return 0;
}

static ssize_t rxm_cq_handle_comp(struct rxm_ep *rxm_ep,
				  struct fi_cq_data_entry *comp)
{
	struct rxm_rx_buf *rx_buf = comp->op_context;
	struct rxm_tx_entry *tx_entry = comp->op_context;
	struct rxm_tx_buf *tx_buf = comp->op_context;

	/* Remote write events may not consume a posted recv so op context
	 * and hence state would be NULL */
	if (comp->flags & FI_REMOTE_WRITE)
		return rxm_handle_remote_write(rxm_ep, comp);

	switch (RXM_GET_PROTO_STATE(comp->op_context)) {
	case RXM_TX_NOBUF:
		assert(comp->flags & (FI_SEND | FI_WRITE | FI_READ));
		if (tx_entry->ep->msg_mr_local && !tx_entry->ep->rxm_mr_local)
			rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);
		return rxm_finish_send_nobuf(tx_entry);
	case RXM_TX:
		assert(comp->flags & FI_SEND);
		return rxm_finish_send(tx_entry);
	case RXM_SAR_TX:
		assert(comp->flags & FI_SEND);
		return rxm_finish_sar_segment_send(tx_buf);
	case RXM_TX_RMA:
		assert(comp->flags & (FI_WRITE | FI_READ));
		if (tx_entry->ep->msg_mr_local && !tx_entry->ep->rxm_mr_local)
			rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);
		rxm_rma_buf_release(rxm_ep, tx_entry->rma_buf);
		return rxm_finish_send_nobuf(tx_entry);
	case RXM_RX:
		assert(!(comp->flags & FI_REMOTE_READ));
		assert((rx_buf->pkt.hdr.version == OFI_OP_VERSION) &&
		       (rx_buf->pkt.ctrl_hdr.version == RXM_CTRL_VERSION));

		switch (rx_buf->pkt.ctrl_hdr.type) {
		case ofi_ctrl_data:
		case ofi_ctrl_large_data:
			return rxm_handle_recv_comp(rx_buf);
		case ofi_ctrl_ack:
			return rxm_lmt_handle_ack(rx_buf);
		case ofi_ctrl_seg_data:
			return rxm_sar_handle_segment(rx_buf);
		default:
			FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown message type\n");
			assert(0);
			return -FI_EINVAL;
		}
	case RXM_LMT_TX:
		assert(comp->flags & FI_SEND);
		RXM_LOG_STATE_TX(FI_LOG_CQ, tx_entry, RXM_LMT_ACK_WAIT);
		RXM_SET_PROTO_STATE(comp, RXM_LMT_ACK_WAIT);
		return 0;
	case RXM_LMT_ACK_RECVD:
		assert(comp->flags & FI_SEND);
		return rxm_lmt_tx_finish(tx_entry);
	case RXM_LMT_READ:
		assert(comp->flags & FI_READ);
		if (++rx_buf->rma_iov_index < rx_buf->rma_iov->count)
			return 0;
		else if (sizeof(rx_buf->pkt) > rxm_ep->msg_info->tx_attr->inject_size)
			return rxm_lmt_send_ack(rx_buf);
		else
			return rxm_lmt_send_ack_fast(rx_buf);
	case RXM_LMT_ACK_SENT:
		assert(comp->flags & FI_SEND);
		rx_buf = tx_entry->context;
		rxm_tx_buf_release(rx_buf->ep, tx_entry->tx_buf);
		rxm_tx_entry_release(&tx_entry->conn->send_queue, tx_entry);
		return rxm_finish_send_lmt_ack(rx_buf);
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Invalid state!\n");
		assert(0);
		return -FI_EOPBADSTATE;
	}
}

void rxm_cq_write_error(struct util_cq *cq, struct util_cntr *cntr,
			void *op_context, int err)
{
	struct fi_cq_err_entry err_entry = {0};
	err_entry.op_context = op_context;
	err_entry.prov_errno = err;
	err_entry.err = err;

	if (cntr)
		rxm_cntr_incerr(cntr);
	if (ofi_cq_write_error(cq, &err_entry)) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to ofi_cq_write_error\n");
		assert(0);
	}
}

static void rxm_cq_write_error_all(struct rxm_ep *rxm_ep, int err)
{
	struct fi_cq_err_entry err_entry = {0};
	ssize_t ret = 0;

	err_entry.prov_errno = err;
	err_entry.err = err;
	if (rxm_ep->util_ep.tx_cq) {
		ret = ofi_cq_write_error(rxm_ep->util_ep.tx_cq, &err_entry);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to ofi_cq_write_error\n");
			assert(0);
		}
	}
	if (rxm_ep->util_ep.rx_cq) {
		ret = ofi_cq_write_error(rxm_ep->util_ep.rx_cq, &err_entry);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to ofi_cq_write_error\n");
			assert(0);
		}
	}
	if (rxm_ep->util_ep.tx_cntr)
		rxm_cntr_incerr(rxm_ep->util_ep.tx_cntr);

	if (rxm_ep->util_ep.rx_cntr)
		rxm_cntr_incerr(rxm_ep->util_ep.rx_cntr);

	if (rxm_ep->util_ep.wr_cntr)
		rxm_cntr_incerr(rxm_ep->util_ep.wr_cntr);

	if (rxm_ep->util_ep.rd_cntr)
		rxm_cntr_incerr(rxm_ep->util_ep.rd_cntr);
}

static void rxm_cq_read_write_error(struct rxm_ep *rxm_ep)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;
	struct rxm_rx_buf *rx_buf;
	struct fi_cq_err_entry err_entry = {0};
	struct util_cq *util_cq;
	struct util_cntr *util_cntr = NULL;
	ssize_t ret;

	OFI_CQ_READERR(&rxm_prov, FI_LOG_CQ, rxm_ep->msg_cq, ret,
		       err_entry);
	if (ret < 0) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
			"Unable to fi_cq_readerr on msg cq\n");
		rxm_cq_write_error_all(rxm_ep, (int)ret);
		return;
	}

	tx_buf = (struct rxm_tx_buf *)err_entry.op_context;
	tx_entry = (struct rxm_tx_entry *)err_entry.op_context;
	rx_buf = (struct rxm_rx_buf *)err_entry.op_context;

	switch (RXM_GET_PROTO_STATE(err_entry.op_context)) {
	case RXM_SAR_TX:
		tx_entry = tx_buf->tx_entry;
		/* fall through */
	case RXM_TX:
	case RXM_LMT_TX:
		util_cq = tx_entry->ep->util_ep.tx_cq;
		if (tx_entry->ep->util_ep.flags & OFI_CNTR_ENABLED) {
			if (tx_entry->comp_flags & FI_SEND)
				util_cntr = tx_entry->ep->util_ep.tx_cntr;
			else if (tx_entry->comp_flags & FI_WRITE)
				util_cntr = tx_entry->ep->util_ep.wr_cntr;
			else
				util_cntr = tx_entry->ep->util_ep.rd_cntr;
		}
		break;
	case RXM_LMT_ACK_SENT:
		util_cq = tx_entry->ep->util_ep.rx_cq;
		util_cntr = tx_entry->ep->util_ep.rx_cntr;
		break;
	case RXM_RX:
	case RXM_LMT_READ:
		util_cq = rx_buf->ep->util_ep.rx_cq;
		util_cntr = rx_buf->ep->util_ep.rx_cntr;
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Invalid state!\n");
		FI_WARN(&rxm_prov, FI_LOG_CQ, "msg cq error info: %s\n",
			fi_cq_strerror(rxm_ep->msg_cq, err_entry.prov_errno,
				       err_entry.err_data, NULL, 0));
		rxm_cq_write_error_all(rxm_ep, -FI_EOPBADSTATE);
		return;
	}
	if (util_cntr)
		rxm_cntr_incerr(util_cntr);
	ret = ofi_cq_write_error(util_cq, &err_entry);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to ofi_cq_write_error\n");
		assert(0);
	}
}

static inline int rxm_ep_repost_buf(struct rxm_rx_buf *rx_buf)
{
	if (rx_buf->ep->srx_ctx)
		rx_buf->conn = NULL;
	rx_buf->hdr.state = RXM_RX;

	if (fi_recv(rx_buf->hdr.msg_ep, &rx_buf->pkt, rx_buf->ep->eager_pkt_size,
		    rx_buf->hdr.desc, FI_ADDR_UNSPEC, rx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to repost buf\n");
		return -FI_EAVAIL;
	}
	return FI_SUCCESS;
}

int rxm_ep_prepost_buf(struct rxm_ep *rxm_ep, struct fid_ep *msg_ep)
{
	struct rxm_rx_buf *rx_buf;
	int ret;
	size_t i;

	for (i = 0; i < rxm_ep->msg_info->rx_attr->size; i++) {
		rx_buf = rxm_rx_buf_get(rxm_ep);
		if (OFI_UNLIKELY(!rx_buf))
			return -FI_ENOMEM;

		rx_buf->hdr.state = RXM_RX;
		rx_buf->hdr.msg_ep = msg_ep;
		rx_buf->repost = 1;

		if (!rxm_ep->srx_ctx)
			rx_buf->conn = container_of(msg_ep->fid.context,
						    struct rxm_conn,
						    handle);
		ret = rxm_ep_repost_buf(rx_buf);
		if (ret) {
			rxm_rx_buf_release(rxm_ep, rx_buf);
			return ret;
		}
	}
	return 0;
}

static inline void rxm_cq_repost_rx_buffers(struct rxm_ep *rxm_ep)
{
	struct rxm_rx_buf *buf;
	rxm_ep->res_fastlock_acquire(&rxm_ep->util_ep.lock);
	while (!dlist_empty(&rxm_ep->repost_ready_list)) {
		dlist_pop_front(&rxm_ep->repost_ready_list, struct rxm_rx_buf,
				buf, repost_entry);
		(void) rxm_ep_repost_buf(buf);
	}
	rxm_ep->res_fastlock_release(&rxm_ep->util_ep.lock);
}

static inline ssize_t rxm_ep_read_msg_cq(struct rxm_ep *rxm_ep)
{	
	struct fi_cq_data_entry comp;
	ssize_t ret;

	ret = fi_cq_read(rxm_ep->msg_cq, &comp, 1);
	if (ret > 0) {
		// TODO handle errors internally and make this function return void.
		// We don't have enough info to write a good error entry to the CQ at
		// this point
		ret = rxm_cq_handle_comp(rxm_ep, &comp);
		if (OFI_UNLIKELY(ret)) {
			rxm_cq_write_error_all(rxm_ep, ret);
		} else {
			return 1;
		}
	} else if (ret < 0) {
		if (ret != -FI_EAGAIN) {
			if (ret == -FI_EAVAIL)
				rxm_cq_read_write_error(rxm_ep);
			else
				rxm_cq_write_error_all(rxm_ep, ret);
		}
	}
	return ret;
}

void rxm_ep_progress_one(struct util_ep *util_ep)
{
	struct rxm_ep *rxm_ep =
		container_of(util_ep, struct rxm_ep, util_ep);

	rxm_cq_repost_rx_buffers(rxm_ep);

	(void) rxm_ep_read_msg_cq(rxm_ep);

	if (OFI_UNLIKELY(!dlist_empty(&rxm_ep->conn_deferred_list)))
		rxm_ep_progress_deferred_list(rxm_ep);
}

void rxm_ep_progress_multi(struct util_ep *util_ep)
{
	struct rxm_ep *rxm_ep =
		container_of(util_ep, struct rxm_ep, util_ep);
	ssize_t ret;
	size_t comp_read = 0;

	rxm_cq_repost_rx_buffers(rxm_ep);

	do {
		ret = rxm_ep_read_msg_cq(rxm_ep);

		if (OFI_UNLIKELY(!dlist_empty(&rxm_ep->conn_deferred_list)))
			rxm_ep_progress_deferred_list(rxm_ep);
	} while ((++comp_read < rxm_ep->comp_per_progress) && (ret > 0));
}

static int rxm_cq_close(struct fid *fid)
{
	struct util_cq *util_cq;
	int ret, retv = 0;

	util_cq = container_of(fid, struct util_cq, cq_fid.fid);

	ret = ofi_cq_cleanup(util_cq);
	if (ret)
		retv = ret;

	free(util_cq);
	return retv;
}

static struct fi_ops rxm_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq rxm_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = ofi_cq_sread,
	.sreadfrom = ofi_cq_sreadfrom,
	.signal = ofi_cq_signal,
	.strerror = rxm_cq_strerror,
};

int rxm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	struct util_cq *util_cq;
	int ret;

	util_cq = calloc(1, sizeof(*util_cq));
	if (!util_cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&rxm_prov, domain, attr, util_cq, &ofi_cq_progress,
			context);
	if (ret)
		goto err1;

	*cq_fid = &util_cq->cq_fid;
	/* Override util_cq_fi_ops */
	(*cq_fid)->fid.ops = &rxm_cq_fi_ops;
	(*cq_fid)->ops = &rxm_cq_ops;
	return 0;
err1:
	free(util_cq);
	return ret;
}
