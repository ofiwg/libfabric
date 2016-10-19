/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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
#include "rxd.h"

static int rxd_cq_write_ctx(struct rxd_cq *cq,
			     struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_ctx_signal(struct rxd_cq *cq,
				    struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_ctx(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_msg(struct rxd_cq *cq,
			     struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_msg_signal(struct rxd_cq *cq,
				    struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_msg(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_data(struct rxd_cq *cq,
			      struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	comp->buf = cq_entry->buf;
	comp->data = cq_entry->data;
	cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_data_signal(struct rxd_cq *cq,
				     struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_data(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_tagged(struct rxd_cq *cq,
				struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		"report completion: %p\n", cq_entry->tag);

	comp = cirque_tail(cq->util_cq.cirq);
	*comp = *cq_entry;
	cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_tagged_signal(struct rxd_cq *cq,
				       struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_tagged(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_check_start_pkt_order(struct rxd_ep *ep, struct rxd_peer *peer,
				      struct ofi_ctrl_hdr *ctrl,
				      struct fi_cq_msg_entry *comp)
{
	uint64_t msg_id;
	msg_id = ctrl->msg_id >> RXD_MAX_TX_BITS;
	if (peer->exp_msg_id == msg_id)
		return RXD_PKT_ORDR_OK;

	return (peer->exp_msg_id > msg_id) ? RXD_PKT_ORDR_DUP : RXD_PKT_ORDR_UNEXP;
}

static int rxd_rx_entry_match(struct dlist_entry *item, const void *arg)
{
	const struct ofi_ctrl_hdr *ctrl = arg;
	struct rxd_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_rx_entry, entry);
	return (rx_entry->msg_id == ctrl->msg_id && rx_entry->peer == ctrl->conn_id);
}

static void rxd_handle_dup_datastart(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
				      struct rxd_rx_buf *rx_buf)
{
	struct dlist_entry *item;
	struct rxd_rx_entry *rx_entry;
	struct rxd_peer *peer;

	item = dlist_find_first_match(&ep->rx_entry_list,
				      rxd_rx_entry_match, ctrl);
	if (!item)
		return;

	FI_INFO(&rxd_prov, FI_LOG_EP_CTRL,
		"duplicate start-data: msg_id: %" PRIu64 ", seg_no: %d\n",
		ctrl->msg_id, ctrl->seg_no);

	rx_entry = container_of(item, struct rxd_rx_entry, entry);
	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);
	rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, rx_entry->window, rx_entry->key,
		       peer->conn_data, ctrl->conn_id);
	return;
}

int rxd_handle_conn_req(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
			 struct fi_cq_msg_entry *comp,
			 struct rxd_rx_buf *rx_buf)
{
	int ret;
	void *addr;
	size_t addrlen;
	uint64_t peer;
	struct rxd_pkt_data *pkt_data;
	struct rxd_peer *peer_info;

	rxd_ep_lock_if_required(ep);

	pkt_data = (struct rxd_pkt_data *) ctrl;
	addr = pkt_data->data;
	addrlen = ctrl->seg_size;

	ret = rxd_av_dg_reverse_lookup(ep->av, ctrl->rx_key, addr, addrlen, &peer);
	if (ret == -FI_ENODATA) {
		ret = rxd_av_insert_dg_av(ep->av, addr);
		assert(ret == 1);

		ret = rxd_av_dg_reverse_lookup(ep->av, ctrl->rx_key, addr, addrlen, &peer);
		assert(ret == 0);
	}

	peer_info = rxd_ep_getpeer_info(ep, peer);
	if (!peer_info->addr_published) {
		peer_info->addr_published = 1;
		peer_info->conn_initiated = 1;
		peer_info->conn_data = ctrl->conn_id;
		peer_info->exp_msg_id++;
	}

	rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_connresp, 0, ctrl->conn_id, peer, peer);
	rxd_ep_repost_buff(rx_buf);
	rxd_ep_unlock_if_required(ep);
	return ret;
}

int rxd_tx_pkt_match(struct dlist_entry *item, const void *arg)
{
	const struct ofi_ctrl_hdr *pkt_ctrl, *ack_ctrl = arg;
	struct rxd_pkt_meta *tx_pkt;

	tx_pkt = container_of(item, struct rxd_pkt_meta, entry);
	pkt_ctrl = (struct ofi_ctrl_hdr *) tx_pkt->pkt_data;
	return (ack_ctrl->seg_no == pkt_ctrl->seg_no) ? 1 : 0;
}

int rxd_handle_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
		    struct rxd_rx_buf *rx_buf)
{
	int ret = 0;
	uint64_t idx;
	struct rxd_tx_entry *tx_entry;
	struct dlist_entry *item;
	struct rxd_pkt_meta *pkt;

	rxd_ep_lock_if_required(ep);
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got ack: msg: %p - %d\n",
		ctrl->msg_id, ctrl->seg_no);

	idx = ctrl->msg_id & RXD_TX_IDX_BITS;
	tx_entry = &ep->tx_entry_fs->buf[idx];
	if (tx_entry->msg_id != ctrl->msg_id)
		goto out;

	item = dlist_find_first_match(&tx_entry->pkt_list, rxd_tx_pkt_match, ctrl);
	if (!item)
		goto out;

	pkt = container_of(item, struct rxd_pkt_meta, entry);
	switch (pkt->type) {

	case RXD_PKT_STRT:
	case RXD_PKT_DATA:
		ret = rxd_tx_entry_progress(ep, tx_entry, ctrl);
		break;

	case RXD_PKT_LAST:
		rxd_ep_free_acked_pkts(ep, tx_entry, ctrl->seg_no);
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "reporting TX completion : %p\n", tx_entry);
		if (tx_entry->op_type != RXD_TX_READ_REQ) {
			rxd_cq_report_tx_comp(ep->tx_cq, tx_entry);
			rxd_tx_entry_done(ep, tx_entry);
		}
		break;
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid pkt type\n");
		break;
	}
out:
	rxd_ep_repost_buff(rx_buf);
	rxd_ep_unlock_if_required(ep);
	return ret;
}

int rxd_handle_nack(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
		     struct rxd_rx_buf *rx_buf)
{
	int ret = 0;
	uint64_t idx;
	struct rxd_tx_entry *tx_entry;

	rxd_ep_lock_if_required(ep);
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got NACK: msg: %p - %d\n",
		ctrl->msg_id, ctrl->seg_no);

	idx = ctrl->msg_id & RXD_TX_IDX_BITS;
	tx_entry = &ep->tx_entry_fs->buf[idx];
	if (tx_entry->msg_id != ctrl->msg_id)
		goto out;

	ret = rxd_tx_entry_progress(ep, tx_entry, ctrl);
out:
	rxd_ep_repost_buff(rx_buf);
	rxd_ep_unlock_if_required(ep);
	return ret;
}

void rxd_handle_discard(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
			struct rxd_rx_buf *rx_buf)
{
	uint64_t idx;
	struct rxd_tx_entry *tx_entry;

	rxd_ep_lock_if_required(ep);
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got Reject: msg: %p - %d\n",
		ctrl->msg_id, ctrl->seg_no);

	idx = ctrl->msg_id & RXD_TX_IDX_BITS;
	tx_entry = &ep->tx_entry_fs->buf[idx];
	if (tx_entry->msg_id != ctrl->msg_id)
		goto out;

	rxd_tx_entry_discard(ep, tx_entry);
out:
	rxd_ep_repost_buff(rx_buf);
	rxd_ep_unlock_if_required(ep);
}

void rxd_tx_pkt_release(struct rxd_pkt_meta *pkt_meta)
{
	if (RXD_PKT_IS_COMPLETE(pkt_meta)) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Releasing buf: %p, num_out: %d\n",
		       pkt_meta, pkt_meta->ep->num_out);
		pkt_meta->ep->num_out--;
		util_buf_release(pkt_meta->ep->tx_pkt_pool, pkt_meta);
	}
}

void rxd_tx_entry_done(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	struct rxd_pkt_meta *pkt_meta;
	struct dlist_entry *item;

	while (!dlist_empty(&tx_entry->pkt_list)) {
		item = tx_entry->pkt_list.next;
		pkt_meta = container_of(item, struct rxd_pkt_meta, entry);
		dlist_remove(&pkt_meta->entry);
		RXD_PKT_MARK_REMOTE_ACK(pkt_meta);
		rxd_tx_pkt_release(pkt_meta);
	}
	rxd_tx_entry_release(ep, tx_entry);
}

static int rxd_conn_msg_match(struct dlist_entry *item, const void *arg)
{
	struct rxd_tx_entry *tx_entry;
	struct ofi_ctrl_hdr *ctrl = (struct ofi_ctrl_hdr *) arg;
	tx_entry = container_of(item, struct rxd_tx_entry, entry);
	return (tx_entry->op_type == RXD_TX_CONN &&
		tx_entry->peer == ctrl->rx_key);
}

void rxd_handle_connect_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
			     struct rxd_rx_buf *rx_buf)
{
	struct rxd_peer *peer;
	struct dlist_entry *match;
	struct rxd_tx_entry *tx_entry;

	rxd_ep_lock_if_required(ep);
	match = dlist_find_first_match(&ep->tx_entry_list, rxd_conn_msg_match, ctrl);
	if (!match)
		goto out;

	tx_entry = container_of(match, struct rxd_tx_entry, entry);
	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);
	peer->addr_published = 1;
	peer->conn_data = ctrl->conn_id;

	dlist_remove(match);
	rxd_tx_entry_done(ep, tx_entry);
out:
	rxd_ep_repost_buff(rx_buf);
	rxd_ep_unlock_if_required(ep);
}

static inline uint16_t rxd_get_window_sz(struct rxd_ep *ep, uint64_t rem)
{
	uint16_t num_pkts, avail;

	num_pkts = (rem +  RXD_MAX_DATA_PKT_SZ(ep) - 1) / RXD_MAX_DATA_PKT_SZ(ep);
	avail = MIN(ep->credits, num_pkts);
	return MIN(avail, RXD_MAX_RX_WIN);
}

struct rxd_rx_entry *rxd_get_rx_entry(struct rxd_ep *ep)
{
	struct rxd_rx_entry *rx_entry;
	if (freestack_isempty(ep->rx_entry_fs))
		return NULL;

	rx_entry = freestack_pop(ep->rx_entry_fs);
	rx_entry->key = rx_entry - &ep->rx_entry_fs->buf[0];
	dlist_init(&rx_entry->entry);
	dlist_init(&rx_entry->wait_entry);
	dlist_insert_tail(&rx_entry->entry, &ep->rx_entry_list);
	return rx_entry;
}

static void rxd_progress_wait_rx(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry)
{
	struct ofi_ctrl_hdr ctrl;

	rx_entry->window = rxd_get_window_sz(ep, rx_entry->op_hdr.size - rx_entry->done);

	if (!rx_entry->window)
		return;

	rx_entry->last_win_seg += rx_entry->window;
	dlist_remove(&rx_entry->wait_entry);

	ep->credits -= rx_entry->window;

	ctrl.msg_id = rx_entry->msg_id;
	ctrl.seg_no = rx_entry->exp_seg_no - 1;
	ctrl.conn_id = rx_entry->peer;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "rx-entry wait over [%p], window: %d\n",
		rx_entry->msg_id, rx_entry->window);
	rxd_ep_reply_ack(ep, &ctrl, ofi_ctrl_ack, rx_entry->window,
		       rx_entry->key, rx_entry->peer_info->conn_data,
		       ctrl.conn_id);
}

static void rxd_check_waiting_rx(struct rxd_ep *ep)
{
	struct dlist_entry *entry;
	struct rxd_rx_entry *rx_entry;

	if (!ep->credits)
		return;

	while(!dlist_empty(&ep->wait_rx_list) && ep->credits) {
		entry = ep->wait_rx_list.next;
		rx_entry = container_of(entry, struct rxd_rx_entry, wait_entry);
		rxd_progress_wait_rx(ep, rx_entry);
	}
}

void rxd_rx_entry_release(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry)
{
	rx_entry->key = -1;
	dlist_remove(&rx_entry->entry);
	freestack_push(ep->rx_entry_fs, rx_entry);

	if (ep->credits && !dlist_empty(&ep->wait_rx_list))
		rxd_check_waiting_rx(ep);
}

static int rxd_match_recv_entry(struct dlist_entry *item, const void *arg)
{
	struct rxd_rx_entry *rx_entry = (struct rxd_rx_entry *) arg;
	struct rxd_recv_entry *recv_entry;
	recv_entry = container_of(item, struct rxd_recv_entry, entry);
	return (recv_entry->msg.addr == FI_ADDR_UNSPEC ||
		rx_entry->source == FI_ADDR_UNSPEC ||
		recv_entry->msg.addr == rx_entry->source);
}

struct rxd_recv_entry *rxd_get_recv_entry(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry)
{
	struct dlist_entry *match;
	struct rxd_recv_entry *recv_entry;

	match = dlist_find_first_match(&ep->recv_list, &rxd_match_recv_entry,
				       (void *) rx_entry);
	if (!match) {
		/*todo: queue the pkt */
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "no matching recv entry\n");
		return NULL;
	}

	dlist_remove(match);
	recv_entry = container_of(match, struct rxd_recv_entry, entry);
	return recv_entry;
}

static int rxd_match_trecv_entry(struct dlist_entry *item, const void *arg)
{
	struct rxd_rx_entry *rx_entry = (struct rxd_rx_entry *) arg;
	struct rxd_trecv_entry *trecv_entry;

	trecv_entry = container_of(item, struct rxd_trecv_entry, entry);
	return ((trecv_entry->msg.tag | trecv_entry->msg.ignore) ==
		(rx_entry->op_hdr.tag | trecv_entry->msg.ignore) &&
                ((trecv_entry->msg.addr == FI_ADDR_UNSPEC) ||
		 (rx_entry->source == FI_ADDR_UNSPEC) ||
                 (trecv_entry->msg.addr == rx_entry->source)));
	return 0;
}

struct rxd_trecv_entry *rxd_get_trecv_entry(struct rxd_ep *ep,
					      struct rxd_rx_entry *rx_entry)
{
	struct dlist_entry *match;
	struct rxd_trecv_entry *trecv_entry;

	match = dlist_find_first_match(&ep->trecv_list, &rxd_match_trecv_entry,
				       (void *)rx_entry);
	if (!match) {
		/*todo: queue the pkt */
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "no matching trecv entry, tag: %p\n",
			rx_entry->op_hdr.tag);
		return NULL;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "matched - tag: %p\n", rx_entry->op_hdr.tag);

	dlist_remove(match);
	trecv_entry = container_of(match, struct rxd_trecv_entry, entry);
	trecv_entry->rx_entry = rx_entry;
	return trecv_entry;
}

void rxd_report_rx_comp(struct rxd_cq *cq, struct rxd_rx_entry *rx_entry)
{
	struct fi_cq_tagged_entry cq_entry = {0};

	/* todo: handle FI_COMPLETION */
	if (rx_entry->op_hdr.flags & OFI_REMOTE_CQ_DATA)
		cq_entry.flags |= FI_REMOTE_CQ_DATA;

	switch(rx_entry->op_hdr.op) {
	case ofi_op_msg:
		cq_entry.flags |= FI_RECV;
		cq_entry.op_context = rx_entry->recv->msg.context;
		cq_entry.len = rx_entry->done;
		cq_entry.buf = rx_entry->recv->iov[0].iov_base;
		cq_entry.data = rx_entry->op_hdr.data;
		break;

	case ofi_op_tagged:
		cq_entry.flags |= (FI_RECV | FI_TAGGED);
		cq_entry.op_context = rx_entry->trecv->msg.context;
		cq_entry.len = rx_entry->done;
		cq_entry.buf = rx_entry->trecv->iov[0].iov_base;
		cq_entry.data = rx_entry->op_hdr.data;
		cq_entry.tag = rx_entry->trecv->msg.tag;
		break;

	case ofi_op_atomic:
		cq_entry.flags |= FI_ATOMIC;
		break;

	case ofi_op_write:
		if (!(rx_entry->op_hdr.flags & OFI_REMOTE_CQ_DATA))
			return;

		cq_entry.flags |= (FI_RMA | FI_REMOTE_WRITE);
		cq_entry.op_context = rx_entry->trecv->msg.context;
		cq_entry.len = rx_entry->done;
		cq_entry.buf = rx_entry->write.iov[0].iov_base;
		cq_entry.data = rx_entry->op_hdr.data;
		break;

	case ofi_op_read_rsp:
		return;

	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type: %d\n",
			rx_entry->op_hdr.op);
		break;
	}

	cq->write_fn(cq, &cq_entry);
}

void rxd_cq_report_error(struct rxd_cq *cq, struct fi_cq_err_entry *err_entry)
{
	struct fi_cq_tagged_entry cq_entry = {0};
	struct util_cq_err_entry *entry = calloc(1, sizeof(*entry));
	if (!entry) {
		FI_WARN(&rxd_prov, FI_LOG_CQ,
			"out of memory, cannot report CQ error\n");
		return;
	}

	entry->err_entry = *err_entry;
	slist_insert_tail(&entry->list_entry, &cq->util_cq.err_list);
	cq_entry.flags = UTIL_FLAG_ERROR;
	cq->write_fn(cq, &cq_entry);
}

void rxd_cq_report_tx_comp(struct rxd_cq *cq, struct rxd_tx_entry *tx_entry)
{
	struct fi_cq_tagged_entry cq_entry = {0};

	/* todo: handle FI_COMPLETION */
	switch(tx_entry->op_type) {
	case RXD_TX_MSG:
		cq_entry.flags = (FI_TRANSMIT | FI_MSG);
		cq_entry.op_context = tx_entry->msg.msg.context;
		cq_entry.len = tx_entry->op_hdr.size;
		cq_entry.buf = tx_entry->msg.msg_iov[0].iov_base;
		cq_entry.data = tx_entry->op_hdr.data;
		break;

	case RXD_TX_TAG:
		cq_entry.flags = (FI_TRANSMIT | FI_TAGGED);
		cq_entry.op_context = tx_entry->tmsg.tmsg.context;
		cq_entry.len = tx_entry->op_hdr.size;
		cq_entry.buf = tx_entry->tmsg.msg_iov[0].iov_base;
		cq_entry.data = tx_entry->op_hdr.data;
		cq_entry.tag = tx_entry->tmsg.tmsg.tag;
		break;

	case RXD_TX_WRITE:
		cq_entry.flags = (FI_TRANSMIT | FI_RMA | FI_WRITE);
		cq_entry.op_context = tx_entry->write.msg.context;
		cq_entry.len = tx_entry->op_hdr.size;
		cq_entry.buf = tx_entry->write.msg.msg_iov[0].iov_base;
		cq_entry.data = tx_entry->op_hdr.data;
		break;

	case RXD_TX_READ_REQ:
		cq_entry.flags = (FI_TRANSMIT | FI_RMA | FI_READ);
		cq_entry.op_context = tx_entry->read_req.msg.context;
		cq_entry.len = tx_entry->op_hdr.size;
		cq_entry.buf = tx_entry->read_req.msg.msg_iov[0].iov_base;
		cq_entry.data = tx_entry->op_hdr.data;
		break;

	case RXD_TX_READ_RSP:
		return;

	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type\n");
		return;
	}

	cq->write_fn(cq, &cq_entry);
}

void rxd_ep_handle_data_msg(struct rxd_ep *ep, struct rxd_peer *peer,
			   struct rxd_rx_entry *rx_entry,
			   struct iovec *iov, size_t iov_count,
			   struct ofi_ctrl_hdr *ctrl, void *data,
			   struct rxd_rx_buf *rx_buf)
{

	uint64_t done;

	ep->credits++;
	done = rxd_ep_copy_iov_buf(iov, iov_count, data, ctrl->seg_size,
				   rx_entry->done, RXD_COPY_BUF_TO_IOV);
	rx_entry->done += done;
	rx_entry->window--;
	rx_entry->exp_seg_no++;

	if (done != ctrl->seg_size) {
		/* todo: generate truncation error */
		/* inform peer */
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "TODO: message truncated\n");
	}

	if (rx_entry->window == 0) {
		rx_entry->window = rxd_get_window_sz(ep, rx_entry->op_hdr.size - rx_entry->done);

		rx_entry->last_win_seg += rx_entry->window;
		ep->credits -= rx_entry->window;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "replying ack [%p] - %d\n",
			ctrl->msg_id, ctrl->seg_no);

		rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, rx_entry->window,
			       rx_entry->key, peer->conn_data, ctrl->conn_id);
	}

	if (rx_entry->op_hdr.size != rx_entry->done) {
		if (rx_entry->window == 0) {
			dlist_init(&rx_entry->wait_entry);
			dlist_insert_tail(&rx_entry->wait_entry, &ep->wait_rx_list);
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "rx-entry %p - %d enqueued\n",
				ctrl->msg_id, ctrl->seg_no);
		} else {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
				"rx_entry->op_hdr.size: %d, rx_entry->done: %d\n",
				rx_entry->op_hdr.size, rx_entry->done);
		}
		return;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "reporting RX completion event\n");
	rxd_report_rx_comp(ep->rx_cq, rx_entry);

	switch(rx_entry->op_hdr.op) {
	case ofi_op_msg:
		freestack_push(ep->recv_fs, rx_entry->recv);
		break;

	case ofi_op_tagged:
		freestack_push(ep->trecv_fs, rx_entry->trecv);
		break;

	case ofi_op_read_rsp:
		rxd_cq_report_tx_comp(ep->tx_cq, rx_entry->read_rsp.tx_entry);
		rxd_tx_entry_done(ep, rx_entry->read_rsp.tx_entry);
		break;

	default:
		break;
	}
	rxd_rx_entry_release(ep, rx_entry);
}

static int rxd_check_data_pkt_order(struct rxd_ep *ep,
				     struct rxd_peer *peer,
				     struct ofi_ctrl_hdr *ctrl,
				     struct rxd_rx_entry *rx_entry)
{
	if (rx_entry->msg_id != ctrl->msg_id)
		return RXD_PKT_ORDR_DUP;

	if (rx_entry->exp_seg_no == ctrl->seg_no)
		return RXD_PKT_ORDR_OK;

	return (rx_entry->exp_seg_no > ctrl->seg_no) ?
		RXD_PKT_ORDR_DUP : RXD_PKT_ORDR_UNEXP;
}

static inline void rxd_ep_enqueue_pkt(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
				       struct fi_cq_msg_entry *comp)
{
	struct rxd_unexp_cq_entry *unexp;
	if (comp->flags & RXD_UNEXP_ENTRY ||
	    ep->num_unexp_pkt > RXD_EP_MAX_UNEXP_PKT)
		return;

	unexp = util_buf_alloc(ep->rx_cq->unexp_pool);
	assert(unexp);
	unexp->cq_entry = *comp;
	unexp->cq_entry.flags |= RXD_UNEXP_ENTRY;

	dlist_init(&unexp->entry);
	dlist_insert_tail(&ep->rx_cq->unexp_list, &unexp->entry);
	FI_INFO(&rxd_prov, FI_LOG_EP_CTRL,
		"enqueuing unordered pkt: %p, seg_no: %d\n",
		ctrl->msg_id, ctrl->seg_no);
	ep->num_unexp_pkt++;
}

static inline void rxd_release_unexp_entry(struct rxd_cq *cq,
					    struct fi_cq_msg_entry *comp)
{
	struct rxd_unexp_cq_entry *unexp;
	unexp = container_of(comp, struct rxd_unexp_cq_entry, cq_entry);
	dlist_remove(&unexp->entry);
	util_buf_release(cq->unexp_pool, unexp);
}


static int rxd_match_unexp_msg(struct dlist_entry *item, const void *arg)
{
	const struct rxd_recv_entry *recv_entry = arg;
	struct rxd_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_rx_entry, unexp_entry);
	return (recv_entry->msg.addr == FI_ADDR_UNSPEC ||
		rx_entry->source == FI_ADDR_UNSPEC ||
		rx_entry->source == recv_entry->msg.addr);
}

void rxd_ep_check_unexp_msg_list(struct rxd_ep *ep, struct rxd_recv_entry *recv_entry)
{
	struct dlist_entry *match;
	struct rxd_rx_entry *rx_entry;
	struct rxd_pkt_data_start *pkt_start;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "ep->num_unexp_msg: %d\n", ep->num_unexp_msg);
	match = dlist_remove_first_match(&ep->unexp_msg_list, &rxd_match_unexp_msg,
					 (void *) recv_entry);
	if (match) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "progressing unexp msg entry\n");
		dlist_remove(&recv_entry->entry);
		ep->num_unexp_msg--;

		rx_entry = container_of(match, struct rxd_rx_entry, unexp_entry);
		rx_entry->recv = recv_entry;

		pkt_start = (struct rxd_pkt_data_start *) rx_entry->unexp_buf->buf;
		rxd_ep_handle_data_msg(ep, rx_entry->peer_info, rx_entry, rx_entry->recv->iov,
				     rx_entry->recv->msg.iov_count, &pkt_start->ctrl,
				     pkt_start->data, rx_entry->unexp_buf);
		rxd_ep_repost_buff(rx_entry->unexp_buf);
	}
}

static int rxd_match_unexp_tag(struct dlist_entry *item, const void *arg)
{
	const struct rxd_trecv_entry *trecv_entry = arg;
	struct rxd_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_rx_entry, unexp_entry);
	return ((trecv_entry->msg.tag | trecv_entry->msg.ignore) ==
		(rx_entry->op_hdr.tag | trecv_entry->msg.ignore) &&
		((trecv_entry->msg.addr == FI_ADDR_UNSPEC) ||
		 (rx_entry->source == FI_ADDR_UNSPEC) ||
		 (trecv_entry->msg.addr == rx_entry->source)));
}

void rxd_ep_check_unexp_tag_list(struct rxd_ep *ep, struct rxd_trecv_entry *trecv_entry)
{
	struct dlist_entry *match;
	struct rxd_rx_entry *rx_entry;
	struct rxd_pkt_data_start *pkt_start;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "ep->num_unexp_msg: %d\n", ep->num_unexp_msg);
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "ep->num_unexp_pkt: %d\n", ep->num_unexp_pkt);
	match = dlist_find_first_match(&ep->unexp_tag_list, &rxd_match_unexp_tag,
				       (void *) trecv_entry);
	if (match) {
		dlist_remove(match);
		dlist_remove(&trecv_entry->entry);
		ep->num_unexp_msg--;

		rx_entry = container_of(match, struct rxd_rx_entry, unexp_entry);
		rx_entry->trecv = trecv_entry;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "progressing unexp tagged recv [%p]\n",
			rx_entry->msg_id);

		pkt_start = (struct rxd_pkt_data_start *) rx_entry->unexp_buf->buf;
		rxd_ep_handle_data_msg(ep, rx_entry->peer_info, rx_entry, rx_entry->trecv->iov,
				     rx_entry->trecv->msg.iov_count, &pkt_start->ctrl,
				     pkt_start->data, rx_entry->unexp_buf);
		rxd_ep_repost_buff(rx_entry->unexp_buf);
	}
}

void rxd_handle_data(struct rxd_ep *ep, struct rxd_peer *peer,
		      struct ofi_ctrl_hdr *ctrl, struct fi_cq_msg_entry *comp,
		      struct rxd_rx_buf *rx_buf)
{
	int ret;
	struct rxd_rx_entry *rx_entry;
	struct rxd_tx_entry *tx_entry;
	struct rxd_pkt_data *pkt_data = (struct rxd_pkt_data *) ctrl;
	uint16_t win_sz;
	uint64_t curr_stamp;

	rxd_ep_lock_if_required(ep);
	rx_entry = &ep->rx_entry_fs->buf[ctrl->rx_key];

	ret = rxd_check_data_pkt_order(ep, peer, ctrl, rx_entry);
	if (ret == RXD_PKT_ORDR_DUP) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
			"duplicate pkt: %d expected:%d, rx-key:%d, ctrl_msg_id: %p\n",
			ctrl->seg_no, rx_entry->exp_seg_no, ctrl->rx_key, ctrl->msg_id);

		win_sz = (rx_entry->msg_id == ctrl->msg_id &&
			  rx_entry->last_win_seg == ctrl->seg_no) ? rx_entry->window : 0;
		rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, win_sz,
			       ctrl->rx_key, peer->conn_data, ctrl->conn_id);

		goto repost;
	} else if (ret == RXD_PKT_ORDR_UNEXP) {
		if (!(comp->flags & RXD_UNEXP_ENTRY)) {
			curr_stamp = fi_gettime_us();
			if (rx_entry->nack_stamp == 0 ||
			    (curr_stamp > rx_entry->nack_stamp &&
			     curr_stamp - rx_entry->nack_stamp > RXD_RETRY_TIMEOUT)) {

				FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
				       "unexpected pkt, sending NACK: %d\n", ctrl->seg_no);

				rx_entry->nack_stamp = curr_stamp;
				rxd_ep_reply_nack(ep, ctrl, rx_entry->exp_seg_no,
						ctrl->rx_key, peer->conn_data,
						ctrl->conn_id);
			}
			rxd_ep_enqueue_pkt(ep, ctrl, comp);
		}
		goto out;
	}

	rx_entry->nack_stamp = 0;
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "expected pkt: %d\n", ctrl->seg_no);
	switch (rx_entry->op_hdr.op) {
	case ofi_op_msg:
		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->recv->iov,
				     rx_entry->recv->msg.iov_count, ctrl,
				     pkt_data->data, rx_buf);
		break;

	case ofi_op_tagged:
		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->trecv->iov,
				     rx_entry->trecv->msg.iov_count, ctrl,
				     pkt_data->data, rx_buf);
		break;

	case ofi_op_write:
		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->write.iov,
				       rx_entry->op_hdr.iov_count, ctrl,
				       pkt_data->data, rx_buf);
		break;

	case ofi_op_read_rsp:
		tx_entry = rx_entry->read_rsp.tx_entry;
		rxd_ep_handle_data_msg(ep, peer, rx_entry, tx_entry->read_req.dst_iov,
				       tx_entry->read_req.msg.iov_count, ctrl,
				       pkt_data->data, rx_buf);
		break;

	case ofi_op_atomic:
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type\n");
	}

repost:
	if (comp->flags & RXD_UNEXP_ENTRY) {
		rxd_release_unexp_entry(ep->rx_cq, comp);
		ep->num_unexp_pkt--;
	}

	rxd_ep_repost_buff(rx_buf);
out:
	rxd_ep_unlock_if_required(ep);
}

void rxd_ep_handle_read_req(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			    struct rxd_peer *peer)
{
	int ret;

	dlist_init(&tx_entry->pkt_list);
	tx_entry->op_type = RXD_TX_READ_RSP;
	ret = rxd_ep_post_start_msg(ep, peer, ofi_op_read_rsp, tx_entry);
	if (ret)
		goto err;

	dlist_insert_tail(&tx_entry->entry, &ep->tx_entry_list);
	return;
err:
	rxd_tx_entry_release(ep, tx_entry);
	return;
}

int rxd_process_start_data(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry,
			   struct rxd_peer *peer, struct ofi_ctrl_hdr *ctrl,
			   struct fi_cq_msg_entry *comp,
			   struct rxd_rx_buf *rx_buf)
{
	uint64_t idx;
	int i, offset, ret;
	struct ofi_rma_iov *rma_iov;
	struct rxd_pkt_data_start *pkt_start;
	struct rxd_tx_entry *tx_entry;
	pkt_start = (struct rxd_pkt_data_start *) ctrl;

	switch (rx_entry->op_hdr.op) {
	case ofi_op_msg:
		rx_entry->recv = rxd_get_recv_entry(ep, rx_entry);
		if (!rx_entry->recv) {
			if (ep->num_unexp_msg < RXD_EP_MAX_UNEXP_MSG) {
				dlist_insert_tail(&rx_entry->unexp_entry, &ep->unexp_msg_list);
				rx_entry->unexp_buf = rx_buf;
				ep->num_unexp_msg++;
				return -FI_ENOENT;
			} else {
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "dropping msg\n");
				return -FI_ENOMEM;
			}
		}

		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->recv->iov,
				     rx_entry->recv->msg.iov_count, ctrl,
				     pkt_start->data, rx_buf);
		break;

	case ofi_op_tagged:
		rx_entry->trecv = rxd_get_trecv_entry(ep, rx_entry);
		if (!rx_entry->trecv) {
			if (ep->num_unexp_msg < RXD_EP_MAX_UNEXP_MSG) {
				dlist_insert_tail(&rx_entry->unexp_entry, &ep->unexp_tag_list);
				rx_entry->unexp_buf = rx_buf;
				ep->num_unexp_msg++;
				return -FI_ENOENT;
			} else {
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "dropping msg\n");
				return -FI_ENOMEM;
			}
		}

		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->trecv->iov,
				     rx_entry->trecv->msg.iov_count, ctrl,
				     pkt_start->data, rx_buf);
		break;

	case ofi_op_write:
		rma_iov = (struct ofi_rma_iov *) pkt_start->data;
		for (i = 0; i < rx_entry->op_hdr.iov_count; i++) {
			ret = rxd_mr_verify(ep->domain,
					    rma_iov[i].len,
					    (uintptr_t *) &rma_iov[i].addr,
					    rma_iov[i].key, FI_REMOTE_WRITE);
			if (ret) {
				/* todo: handle invalid key case */
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid key/access permissions\n");
				return -FI_EACCES;
			}

			rx_entry->write.iov[i].iov_base = (void *) (uintptr_t) rma_iov[i].addr;
			rx_entry->write.iov[i].iov_len = rma_iov[i].len;
		}

		offset = sizeof(struct ofi_rma_iov) * rx_entry->op_hdr.iov_count;
		ctrl->seg_size -= offset;
		rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->write.iov,
				       rx_entry->op_hdr.iov_count, ctrl,
				       pkt_start->data + offset, rx_buf);
		break;

	case ofi_op_read_req:
		rma_iov = (struct ofi_rma_iov *) pkt_start->data;
		tx_entry = rxd_tx_entry_acquire_fast(ep, peer);
		if (!tx_entry) {
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "no free tx-entry\n");
			return -FI_ENOMEM;
		}

		tx_entry->peer = rx_entry->peer;
		tx_entry->read_rsp.iov_count = rx_entry->op_hdr.iov_count;
		for (i = 0; i < rx_entry->op_hdr.iov_count; i++) {
			ret = rxd_mr_verify(ep->domain,
					    rma_iov[i].len,
					    (uintptr_t *) &rma_iov[i].addr,
					    rma_iov[i].key, FI_REMOTE_READ);
			if (ret) {
				/* todo: handle invalid key case */
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid key/access permissions\n");
				return -FI_EACCES;
			}

			tx_entry->read_rsp.src_iov[i].iov_base = (void *) (uintptr_t) rma_iov[i].addr;
			tx_entry->read_rsp.src_iov[i].iov_len = rma_iov[i].len;
		}
		tx_entry->read_rsp.peer_msg_id = ctrl->msg_id;
		rxd_ep_handle_read_req(ep, tx_entry, peer);
		rxd_rx_entry_release(ep, rx_entry);
		break;

	case ofi_op_read_rsp:
		idx = rx_entry->op_hdr.remote_idx & RXD_TX_IDX_BITS;
		tx_entry = &ep->tx_entry_fs->buf[idx];
		if (tx_entry->msg_id != rx_entry->op_hdr.remote_idx)
			return -FI_ENOMEM;

		rx_entry->read_rsp.tx_entry = tx_entry;
		rxd_ep_handle_data_msg(ep, peer, rx_entry, tx_entry->read_req.dst_iov,
				       tx_entry->read_req.msg.iov_count, ctrl,
				       pkt_start->data, rx_buf);
		break;

	case ofi_op_atomic:
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type\n");
		return -FI_EINVAL;
	}
	return 0;
}

void rxd_handle_start_data(struct rxd_ep *ep, struct rxd_peer *peer,
			   struct ofi_ctrl_hdr *ctrl,
			   struct fi_cq_msg_entry *comp,
			   struct rxd_rx_buf *rx_buf)
{
	int ret;
	struct rxd_rx_entry *rx_entry;
	struct rxd_pkt_data_start *pkt_start;
	pkt_start = (struct rxd_pkt_data_start *) ctrl;

	rxd_ep_lock_if_required(ep);
	if (pkt_start->op.version != OFI_OP_VERSION) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "op version mismatch\n");
		goto repost;
	}

	ret = rxd_check_start_pkt_order(ep, peer, ctrl, comp);
	if (ret == RXD_PKT_ORDR_DUP) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "duplicate pkt: %d\n", ctrl->seg_no);
		rxd_handle_dup_datastart(ep, ctrl, rx_buf);
		goto repost;
	} else if (ret == RXD_PKT_ORDR_UNEXP) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "unexpected pkt: %d\n", ctrl->seg_no);
		rxd_ep_enqueue_pkt(ep, ctrl, comp);
		goto out;
	}

	rx_entry = rxd_get_rx_entry(ep);
	if (!rx_entry)
		goto repost;

	rx_entry->peer_info = peer;
	rx_entry->op_hdr = pkt_start->op;
	rx_entry->exp_seg_no = 0;
	rx_entry->msg_id = ctrl->msg_id;
	rx_entry->done = 0;
	rx_entry->peer = ctrl->conn_id;
	rx_entry->source = (ep->caps & FI_DIRECTED_RECV) ?
		rxd_av_get_fi_addr(ep->av, ctrl->conn_id) : FI_ADDR_UNSPEC;
	rx_entry->window = 1;
	rx_entry->last_win_seg = 1;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Assign rx_entry :%d for  %p\n",
	       rx_entry->key, rx_entry->msg_id);

	ep->credits--;
	ret = rxd_process_start_data(ep, rx_entry, peer, ctrl, comp, rx_buf);
	if (ret == -FI_ENOMEM)
		rxd_rx_entry_release(ep, rx_entry);
	else if (ret == -FI_ENOENT) {
		peer->exp_msg_id++;

		/* reply ack, with win_sz = 0 */
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Sending wait-ACK [%p] - %d\n",
			ctrl->msg_id, ctrl->seg_no);
		goto out;
	} else {
		peer->exp_msg_id++;
	}

repost:
	if (comp->flags & RXD_UNEXP_ENTRY) {
		rxd_release_unexp_entry(ep->rx_cq, comp);
		ep->num_unexp_pkt--;
	}
	rxd_ep_repost_buff(rx_buf);
out:
	rxd_ep_unlock_if_required(ep);
	return;
}

void rxd_handle_recv_comp(struct rxd_cq *cq, struct fi_cq_msg_entry *comp,
			   int is_unexpected)
{
	struct rxd_ep *ep;
	struct ofi_ctrl_hdr *ctrl;
	struct rxd_rx_buf *rx_buf;
	struct rxd_peer *peer;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got recv completion\n");

	rx_buf = container_of(comp->op_context, struct rxd_rx_buf, context);
	ctrl = (struct ofi_ctrl_hdr *) rx_buf->buf;
	ep = rx_buf->ep;
	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);

	if (ctrl->type != ofi_ctrl_ack && ctrl->type != ofi_ctrl_nack) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		       "got data pkt - msg_id:[%p - %d], type: %d on buf %p [unexp: %d]\n",
		       ctrl->msg_id, ctrl->seg_no, ctrl->type, rx_buf, is_unexpected);
	} else {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		       "got ack pkt - msg_id:[%p - %d], type: %d on buf %p  [unexp: %d]\n",
		       ctrl->msg_id, ctrl->seg_no, ctrl->type, rx_buf, is_unexpected);
	}
	if (ctrl->version != OFI_CTRL_VERSION) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "ctrl version mismatch\n");
		return;
	}

	switch(ctrl->type) {
	case ofi_ctrl_connreq:
		rxd_handle_conn_req(ep, ctrl, comp, rx_buf);
		break;

	case ofi_ctrl_ack:
		rxd_handle_ack(ep, ctrl, rx_buf);
		break;

	case ofi_ctrl_nack:
		rxd_handle_nack(ep, ctrl, rx_buf);
		break;

	case ofi_ctrl_discard:
		rxd_handle_discard(ep, ctrl, rx_buf);
		break;

	case ofi_ctrl_connresp:
		rxd_handle_connect_ack(ep, ctrl, rx_buf);
		break;

	case ofi_ctrl_start_data:
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		       "start data msg for tx: %p\n", ctrl->msg_id);
		rxd_handle_start_data(ep, peer, ctrl, comp, rx_buf);
		break;

	case ofi_ctrl_data:
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
			"data msg for tx: %p, %d \n", ctrl->msg_id, ctrl->seg_no);
		rxd_handle_data(ep, peer, ctrl, comp, rx_buf);
		break;

	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"invalid ctrl type \n", ctrl->type);
	}

	rxd_ep_lock_if_required(ep);
	rxd_check_waiting_rx(ep);
	rxd_ep_unlock_if_required(ep);
	return;
}

static inline void rxd_handle_send_comp(struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_meta *pkt_meta;
	pkt_meta = container_of(comp->op_context, struct rxd_pkt_meta, context);

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Send completion for: %p\n", pkt_meta);
	rxd_ep_lock_if_required(pkt_meta->ep);
	RXD_PKT_MARK_LOCAL_ACK(pkt_meta);
	rxd_tx_pkt_release(pkt_meta);
	rxd_ep_unlock_if_required(pkt_meta->ep);
}

void rxd_cq_progress(struct util_cq *util_cq)
{
	ssize_t ret = 0;
	struct rxd_cq *cq;
	struct fi_cq_msg_entry cq_entry;
	struct dlist_entry *item, *next;
	struct rxd_unexp_cq_entry *unexp;

	cq = container_of(util_cq, struct rxd_cq, util_cq);
	fastlock_acquire(&cq->lock);

	do {
		ret = fi_cq_read(cq->dg_cq, &cq_entry, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (cq_entry.flags & FI_SEND) {
			rxd_handle_send_comp(&cq_entry);
		} else if (cq_entry.flags & FI_RECV) {
			rxd_handle_recv_comp(cq, &cq_entry, 0);
		} else
			assert (0);
	} while (ret > 0);

	for (item = cq->unexp_list.next; item != &cq->unexp_list;) {
		unexp = container_of(item, struct rxd_unexp_cq_entry, entry);
		next = item->next;
		rxd_handle_recv_comp(cq, &unexp->cq_entry, 1);
		item = next;
	}

	fastlock_release(&cq->lock);
}

static int rxd_cq_close(struct fid *fid)
{
	int ret;
	struct rxd_cq *cq;

	cq = container_of(fid, struct rxd_cq, util_cq.cq_fid.fid);

	fastlock_acquire(&cq->domain->lock);
	dlist_remove(&cq->dom_entry);
	fastlock_release(&cq->domain->lock);
	fastlock_destroy(&cq->lock);

	ret = fi_close(&cq->dg_cq->fid);
	if (ret)
		return ret;

	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;
	util_buf_pool_destroy(cq->unexp_pool);
	free(cq);
	return 0;
}

static struct fi_ops rxd_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int rxd_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct rxd_cq *cq;
	struct rxd_domain *rxd_domain;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&rxd_prov, domain, attr, &cq->util_cq,
			  &rxd_cq_progress, context);
	if (ret)
		goto err1;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
	case FI_CQ_FORMAT_CONTEXT:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_ctx_signal : rxd_cq_write_ctx;
		break;
	case FI_CQ_FORMAT_MSG:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_msg_signal : rxd_cq_write_msg;
		break;
	case FI_CQ_FORMAT_DATA:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_data_signal : rxd_cq_write_data;
		break;
	case FI_CQ_FORMAT_TAGGED:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_tagged_signal : rxd_cq_write_tagged;
		break;
	default:
		ret = -FI_EINVAL;
		goto err2;
	}

	rxd_domain = container_of(domain, struct rxd_domain, util_domain.domain_fid);
	attr->format = FI_CQ_FORMAT_MSG;
	ret = fi_cq_open(rxd_domain->dg_domain, attr, &cq->dg_cq, context);
	if (ret)
		goto err2;

	cq->unexp_pool = util_buf_pool_create(
		RXD_EP_MAX_UNEXP_PKT * sizeof (struct rxd_unexp_cq_entry),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_EP_MAX_UNEXP_PKT);
	if (!cq->unexp_pool) {
		ret = -FI_ENOMEM;
		goto err3;
	}

	dlist_init(&cq->dom_entry);
	dlist_init(&cq->unexp_list);
	fastlock_init(&cq->lock);

	fastlock_acquire(&rxd_domain->lock);
	dlist_insert_tail(&cq->dom_entry, &rxd_domain->cq_list);
	fastlock_release(&rxd_domain->lock);

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &rxd_cq_fi_ops;
	*cq_fid = &cq->util_cq.cq_fid;
	cq->domain = rxd_domain;
	return 0;

err3:
	ofi_cq_cleanup(&cq->util_cq);
err2:
	fi_close(&cq->dg_cq->fid);
err1:
	free(cq);
	return ret;
}
