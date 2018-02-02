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
#include <ofi_iov.h>
#include "rxd.h"

/*
 * All EPs use the same underlying datagram provider, so pick any and use its
 * associated CQ.
 */
static const char *rxd_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
		const void *err_data, char *buf, size_t len)
{
	struct fid_list_entry *fid_entry;
	struct util_ep *util_ep;
	struct rxd_cq *cq;
	struct rxd_ep *ep;
	const char *str;

	cq = container_of(cq_fid, struct rxd_cq, util_cq.cq_fid);

	fastlock_acquire(&cq->util_cq.ep_list_lock);
	assert(!dlist_empty(&cq->util_cq.ep_list));
	fid_entry = container_of(cq->util_cq.ep_list.next,
				struct fid_list_entry, entry);
	util_ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
	ep = container_of(util_ep, struct rxd_ep, util_ep);

	str = fi_cq_strerror(ep->dg_cq, prov_errno, err_data, buf, len);
	fastlock_release(&cq->util_cq.ep_list_lock);
	return str;
}

static int rxd_cq_write_ctx(struct rxd_cq *cq,
			     struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	ofi_cirque_commit(cq->util_cq.cirq);
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
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	ofi_cirque_commit(cq->util_cq.cirq);
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
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	comp->buf = cq_entry->buf;
	comp->data = cq_entry->data;
	ofi_cirque_commit(cq->util_cq.cirq);
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
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "report completion: %" PRIx64 "\n", cq_entry->tag);

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	*comp = *cq_entry;
	ofi_cirque_commit(cq->util_cq.cirq);
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
		return 0;

	return (peer->exp_msg_id > msg_id) ?
		-FI_EALREADY : -FI_EINVAL;
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

	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);
	item = dlist_find_first_match(&ep->rx_entry_list,
				      rxd_rx_entry_match, ctrl);
	if (!item) {
	      /* for small (1-packet) messages we may have situation
	       * when receiver completed operation and destroyed
	       * rx_entry, but ack is lost (not delivered to sender).
	       * in this case just send ack with zero window to
	       * allow sender complete operation on sender side */
	      rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, 0, UINT64_MAX,
			       peer->conn_data, ctrl->conn_id);
	      return;
	}

	FI_INFO(&rxd_prov, FI_LOG_EP_CTRL,
		"duplicate start-data: msg_id: %" PRIu64 ", seg_no: %d\n",
		ctrl->msg_id, ctrl->seg_no);

	rx_entry = container_of(item, struct rxd_rx_entry, entry);
	rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, rx_entry->credits, rx_entry->key,
		       peer->conn_data, ctrl->conn_id);
	return;
}

static void rxd_handle_conn_req(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
				struct fi_cq_msg_entry *comp,
				struct rxd_rx_buf *rx_buf)
{
	struct rxd_pkt_data *pkt_data;
	struct rxd_peer *peer_info;
	fi_addr_t dg_fiaddr;
	void *addr;
	int ret;

	FI_INFO(&rxd_prov, FI_LOG_EP_DATA,
	       "conn req - rx_key: %" PRIu64 "\n", ctrl->rx_key);

	pkt_data = (struct rxd_pkt_data *) ctrl;
	addr = pkt_data->data;
	if (ctrl->seg_size > RXD_MAX_DGRAM_ADDR) {
		FI_WARN(&rxd_prov, FI_LOG_EP_DATA, "addr too large\n");
		goto repost;
	}

	ret = rxd_av_insert_dg_addr(rxd_ep_av(ep), ctrl->rx_key, addr, &dg_fiaddr);
	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_EP_DATA, "failed to insert peer address\n");
		goto repost;
	}

	peer_info = rxd_ep_getpeer_info(ep, dg_fiaddr);
	if (peer_info->state != CMAP_CONNECTED) {
		peer_info->state = CMAP_CONNECTED;
		peer_info->conn_data = ctrl->conn_id;
		peer_info->exp_msg_id++;
	}

	rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_connresp, 0, ctrl->conn_id,
			 dg_fiaddr, dg_fiaddr);
repost:
	rxd_ep_repost_buff(rx_buf);
}

int rxd_tx_pkt_match(struct dlist_entry *item, const void *arg)
{
	const struct ofi_ctrl_hdr *pkt_ctrl, *ack_ctrl = arg;
	struct rxd_pkt_meta *tx_pkt;

	tx_pkt = container_of(item, struct rxd_pkt_meta, entry);
	pkt_ctrl = (struct ofi_ctrl_hdr *) tx_pkt->pkt_data;
	return (ack_ctrl->seg_no == pkt_ctrl->seg_no) ? 1 : 0;
}

static void rxd_handle_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
			   struct rxd_rx_buf *rx_buf)
{
	struct rxd_tx_entry *tx_entry;
	uint64_t idx;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "ack- msg_id: %" PRIu64 ", segno: %d, segsz: %d, buf: %p\n",
	       ctrl->msg_id, ctrl->seg_no, ctrl->seg_size, rx_buf);

	idx = ctrl->msg_id & RXD_TX_IDX_BITS;
	tx_entry = &ep->tx_entry_fs->buf[idx];
	if (tx_entry->msg_id != ctrl->msg_id)
		goto out;

	rxd_ep_free_acked_pkts(ep, tx_entry, ctrl->seg_no);
	if ((tx_entry->bytes_sent == tx_entry->op_hdr.size) &&
	    dlist_empty(&tx_entry->pkt_list)) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
			"reporting TX completion : %p\n", tx_entry);
		if (tx_entry->op_type != RXD_TX_READ_REQ) {
			rxd_cq_report_tx_comp(rxd_ep_tx_cq(ep), tx_entry);
			rxd_cntr_report_tx_comp(ep, tx_entry);
			rxd_tx_entry_free(ep, tx_entry);
		}
	} else {
		tx_entry->rx_key = ctrl->rx_key;
		/* do not allow reduce window size (on duplicate acks) */
		tx_entry->window = MAX(tx_entry->window, ctrl->seg_no + ctrl->seg_size);
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		       "ack- msg_id: %" PRIu64 ", window: %d\n",
		       ctrl->msg_id, tx_entry->window);
	}
out:
	rxd_ep_repost_buff(rx_buf);
}

/*
 * Discarded transfers were discarded by the receiving side, so we abort
 * transferring the rest of the data.  However, the completion is still
 * reported to the sender as successful.  This ensures that short and long
 * messages are treated the same, since short messages would be entirely
 * buffered at the receiver, with no notification that the application later
 * discarded the message.
 */
static void rxd_handle_discard(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
			       struct rxd_rx_buf *rx_buf)
{
	struct rxd_tx_entry *tx_entry;
	uint64_t idx;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "discard- msg_id: %" PRIu64 ", segno: %d\n",
	       ctrl->msg_id, ctrl->seg_no);

	idx = ctrl->msg_id & RXD_TX_IDX_BITS;
	tx_entry = &ep->tx_entry_fs->buf[idx];
	if (tx_entry->msg_id == ctrl->msg_id) {
		rxd_cq_report_tx_comp(rxd_ep_tx_cq(ep), tx_entry);
		rxd_cntr_report_tx_comp(ep, tx_entry);
		rxd_tx_entry_done(ep, tx_entry);
	}

	rxd_ep_repost_buff(rx_buf);
}

void rxd_tx_pkt_free(struct rxd_pkt_meta *pkt_meta)
{
	util_buf_release(pkt_meta->ep->tx_pkt_pool, pkt_meta);
}

void rxd_tx_entry_done(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	struct rxd_pkt_meta *pkt_meta;

	while (!dlist_empty(&tx_entry->pkt_list)) {
		pkt_meta = container_of(tx_entry->pkt_list.next,
					struct rxd_pkt_meta, entry);
		dlist_remove(&pkt_meta->entry);
		if (pkt_meta->flags & RXD_LOCAL_COMP)
			rxd_tx_pkt_free(pkt_meta);
		else
			pkt_meta->flags |= RXD_REMOTE_ACK;
	}
	rxd_tx_entry_free(ep, tx_entry);
}

static int rxd_conn_msg_match(struct dlist_entry *item, const void *arg)
{
	struct rxd_tx_entry *tx_entry;
	struct ofi_ctrl_hdr *ctrl = (struct ofi_ctrl_hdr *) arg;
	tx_entry = container_of(item, struct rxd_tx_entry, entry);
	return (tx_entry->op_type == RXD_TX_CONN &&
		tx_entry->peer == ctrl->rx_key);
}

static void rxd_handle_connect_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *ctrl,
				   struct rxd_rx_buf *rx_buf)
{
	struct rxd_peer *peer;
	struct dlist_entry *match;
	struct rxd_tx_entry *tx_entry;

	FI_INFO(&rxd_prov, FI_LOG_EP_CTRL,
		"connect ack- msg_id: %" PRIu64 ", segno: %d\n",
		ctrl->msg_id, ctrl->seg_no);

	match = dlist_find_first_match(&ep->tx_entry_list,
					rxd_conn_msg_match, ctrl);
	if (!match) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no matching connect\n");
		goto out;
	}

	tx_entry = container_of(match, struct rxd_tx_entry, entry);
	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);
	peer->state = CMAP_CONNECTED;
	peer->conn_data = ctrl->conn_id;

	dlist_remove(match);
	rxd_tx_entry_done(ep, tx_entry);
out:
	rxd_ep_repost_buff(rx_buf);
}

static void rxd_set_rx_credits(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry)
{
	size_t num_pkts, avail, size_left;

	size_left = rx_entry->op_hdr.size - rx_entry->done;
	num_pkts = (size_left + rxd_ep_domain(ep)->max_mtu_sz - 1) /
		    rxd_ep_domain(ep)->max_mtu_sz;
	avail = MIN(ep->credits, num_pkts);
	rx_entry->credits = MIN(avail, RXD_MAX_RX_CREDITS);
	rx_entry->last_win_seg += rx_entry->credits;
	ep->credits -= rx_entry->credits;
}

static struct rxd_rx_entry *rxd_rx_entry_alloc(struct rxd_ep *ep)
{
	struct rxd_rx_entry *rx_entry;

	if (freestack_isempty(ep->rx_entry_fs))
		return NULL;

	rx_entry = freestack_pop(ep->rx_entry_fs);
	rx_entry->key = rx_entry - &ep->rx_entry_fs->buf[0];
	dlist_insert_tail(&rx_entry->entry, &ep->rx_entry_list);
	return rx_entry;
}

static void rxd_progress_wait_rx(struct rxd_ep *ep,
				 struct rxd_rx_entry *rx_entry)
{
	struct ofi_ctrl_hdr ctrl;

	rxd_set_rx_credits(ep, rx_entry);
	if (!rx_entry->credits)
		return;

	dlist_remove(&rx_entry->wait_entry);

	ctrl.msg_id = rx_entry->msg_id;
	ctrl.seg_no = rx_entry->exp_seg_no - 1;
	ctrl.conn_id = rx_entry->peer;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "rx-entry wait over [%" PRIx64 "], credits: %d\n",
	       rx_entry->msg_id, rx_entry->credits);
	rxd_ep_reply_ack(ep, &ctrl, ofi_ctrl_ack, rx_entry->credits,
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

void rxd_rx_entry_free(struct rxd_ep *ep, struct rxd_rx_entry *rx_entry)
{
	rx_entry->key = -1;
	dlist_remove(&rx_entry->entry);
	freestack_push(ep->rx_entry_fs, rx_entry);

	if (ep->credits && !dlist_empty(&ep->wait_rx_list))
		rxd_check_waiting_rx(ep);
}

static int rxd_match_recv_entry(struct dlist_entry *item, const void *arg)
{
	const struct rxd_rx_entry *rx_entry = arg;
	struct rxd_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxd_recv_entry, entry);
	return (recv_entry->msg.addr == FI_ADDR_UNSPEC ||
		rx_entry->source == FI_ADDR_UNSPEC ||
		recv_entry->msg.addr == rx_entry->source);
}

struct rxd_recv_entry *rxd_get_recv_entry(struct rxd_ep *ep,
					  struct rxd_rx_entry *rx_entry)
{
	struct dlist_entry *match;
	struct rxd_recv_entry *recv_entry;

	match = dlist_find_first_match(&ep->recv_list, &rxd_match_recv_entry,
				       (void *) rx_entry);
	if (!match) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "no matching recv entry\n");
		return NULL;
	}

	dlist_remove(match);
	recv_entry = container_of(match, struct rxd_recv_entry, entry);
	return recv_entry;
}

static int rxd_match_trecv_entry(struct dlist_entry *item, const void *arg)
{
	const struct rxd_rx_entry *rx_entry = arg;
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
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
		       "no matching trecv entry, tag: %" PRIx64 "\n",
		       rx_entry->op_hdr.tag);
		return NULL;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "matched - tag: %" PRIx64 "\n",
	       rx_entry->op_hdr.tag);

	dlist_remove(match);
	trecv_entry = container_of(match, struct rxd_trecv_entry, entry);
	trecv_entry->rx_entry = rx_entry;
	return trecv_entry;
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
	struct fi_cq_tagged_entry cq_entry = {0};
	struct util_cntr *cntr = NULL;
	uint64_t done;
	struct rxd_cq *rxd_rx_cq = rxd_ep_rx_cq(ep);

	ep->credits++;
	done = ofi_copy_to_iov(iov, iov_count, rx_entry->done, data,
				ctrl->seg_size);
	rx_entry->done += done;
	rx_entry->credits--;
	rx_entry->exp_seg_no++;

	if (done != ctrl->seg_size) {
		/* todo: generate truncation error */
		/* inform peer */
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "TODO: message truncated\n");
	}

	if (rx_entry->credits == 0) {
		rxd_set_rx_credits(ep, rx_entry);

		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "replying ack [%" PRIx64 "] - %d\n",
		       ctrl->msg_id, ctrl->seg_no);

		rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, rx_entry->credits,
			       rx_entry->key, peer->conn_data, ctrl->conn_id);
	}

	if (rx_entry->op_hdr.size != rx_entry->done) {
		if (rx_entry->credits == 0) {
			dlist_init(&rx_entry->wait_entry);
			dlist_insert_tail(&rx_entry->wait_entry, &ep->wait_rx_list);
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "rx-entry %" PRIx64 " - %d enqueued\n",
				ctrl->msg_id, ctrl->seg_no);
		} else {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
			       "rx_entry->op_hdr.size: %" PRIu64 ", rx_entry->done: %" PRId64 "\n",
			       rx_entry->op_hdr.size,
			       rx_entry->done);
		}
		return;
	}

	/* todo: handle FI_COMPLETION for RX CQ comp */
	switch(rx_entry->op_hdr.op) {
	case ofi_op_msg:
		freestack_push(ep->recv_fs, rx_entry->recv);
		/* Handle cntr */
		cntr = ep->util_ep.rx_cntr;
		/* Handle CQ comp */
		cq_entry.flags |= FI_RECV;
		cq_entry.op_context = rx_entry->recv->msg.context;
		cq_entry.len = rx_entry->done;
		cq_entry.buf = rx_entry->recv->iov[0].iov_base;
		cq_entry.data = rx_entry->op_hdr.data;
		rxd_rx_cq->write_fn(rxd_rx_cq, &cq_entry);
		break;
	case ofi_op_tagged:
		freestack_push(ep->trecv_fs, rx_entry->trecv);
		/* Handle cntr */
		cntr = ep->util_ep.rx_cntr;
		/* Handle CQ comp */
		cq_entry.flags |= (FI_RECV | FI_TAGGED);
		cq_entry.op_context = rx_entry->trecv->msg.context;
		cq_entry.len = rx_entry->done;
		cq_entry.buf = rx_entry->trecv->iov[0].iov_base;
		cq_entry.data = rx_entry->op_hdr.data;
		cq_entry.tag = rx_entry->trecv->msg.tag;\
		rxd_rx_cq->write_fn(rxd_rx_cq, &cq_entry);
		break;
	case ofi_op_atomic:
		/* Handle cntr */
		cntr = ep->util_ep.rem_wr_cntr;
		/* Handle CQ comp */
		cq_entry.flags |= FI_ATOMIC;
		rxd_rx_cq->write_fn(rxd_rx_cq, &cq_entry);
		break;
	case ofi_op_write:
		/* Handle cntr */
		cntr = ep->util_ep.rem_wr_cntr;
		/* Handle CQ comp */
		if (rx_entry->op_hdr.flags & OFI_REMOTE_CQ_DATA) {
			cq_entry.flags |= (FI_RMA | FI_REMOTE_WRITE);
			cq_entry.op_context = rx_entry->trecv->msg.context;
			cq_entry.len = rx_entry->done;
			cq_entry.buf = rx_entry->write.iov[0].iov_base;
			cq_entry.data = rx_entry->op_hdr.data;
			rxd_rx_cq->write_fn(rxd_rx_cq, &cq_entry);
		}
		break;
	case ofi_op_read_rsp:
		rxd_cq_report_tx_comp(rxd_ep_tx_cq(ep), rx_entry->read_rsp.tx_entry);
		rxd_cntr_report_tx_comp(ep, rx_entry->read_rsp.tx_entry);
		rxd_tx_entry_done(ep, rx_entry->read_rsp.tx_entry);
		break;
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid op type: %d\n",
			rx_entry->op_hdr.op);
		break;
	}

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);

	rxd_rx_entry_free(ep, rx_entry);
}

static int rxd_check_data_pkt_order(struct rxd_ep *ep,
				     struct rxd_peer *peer,
				     struct ofi_ctrl_hdr *ctrl,
				     struct rxd_rx_entry *rx_entry)
{
	if ((rx_entry->msg_id == ctrl->msg_id) &&
	    (rx_entry->exp_seg_no == ctrl->seg_no))
		return 0;

	if ((rx_entry->msg_id != ctrl->msg_id) ||
	    (rx_entry->exp_seg_no > ctrl->seg_no))
		return -FI_EALREADY;

	return -FI_EINVAL;
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
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "progressing unexp tagged recv [%" PRIx64 "]\n",
		       rx_entry->msg_id);

		pkt_start = (struct rxd_pkt_data_start *) rx_entry->unexp_buf->buf;
		rxd_ep_handle_data_msg(ep, rx_entry->peer_info, rx_entry, rx_entry->trecv->iov,
				     rx_entry->trecv->msg.iov_count, &pkt_start->ctrl,
				     pkt_start->data, rx_entry->unexp_buf);
		rxd_ep_repost_buff(rx_entry->unexp_buf);
	}
}

static void rxd_handle_data(struct rxd_ep *ep, struct rxd_peer *peer,
			    struct ofi_ctrl_hdr *ctrl, struct fi_cq_msg_entry *comp,
			    struct rxd_rx_buf *rx_buf)
{
	struct rxd_rx_entry *rx_entry;
	struct rxd_tx_entry *tx_entry;
	struct rxd_pkt_data *pkt_data = (struct rxd_pkt_data *) ctrl;
	uint16_t credits;
	int ret;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "data pkt- msg_id: %" PRIu64 ", segno: %d, buf: %p\n",
	       ctrl->msg_id, ctrl->seg_no, rx_buf);

	rx_entry = &ep->rx_entry_fs->buf[ctrl->rx_key];

	ret = rxd_check_data_pkt_order(ep, peer, ctrl, rx_entry);
	if (ret) {
		if (ret == -FI_EALREADY) {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "duplicate pkt: %d "
			       "expected:%d, rx-key:%" PRId64 ", ctrl_msg_id: %" PRIx64 "\n",
			       ctrl->seg_no, rx_entry->exp_seg_no,
			       ctrl->rx_key,
			       ctrl->msg_id);

			credits = ((rx_entry->msg_id == ctrl->msg_id) &&
				  (rx_entry->last_win_seg == ctrl->seg_no)) ?
				  rx_entry->credits : 0;
			rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, credits,
				       ctrl->rx_key, peer->conn_data,
				       ctrl->conn_id);
			goto repost;
		} else {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "invalid pkt: segno: %d "
			       "expected:%d, rx-key:%" PRId64 ", ctrl_msg_id: %" PRIu64 ", "
			       "rx_entry_msg_id: %" PRIx64 "\n",
			       ctrl->seg_no, rx_entry->exp_seg_no,
			       ctrl->rx_key,
			       ctrl->msg_id, rx_entry->msg_id);
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "invalid pkt: "
			       "credits: %d, last win: %d\n",
			       rx_entry->credits, rx_entry->last_win_seg);
			credits = (rx_entry->msg_id == ctrl->msg_id) ?
				  rx_entry->last_win_seg - rx_entry->exp_seg_no : 0;
			rxd_ep_reply_ack(ep, ctrl, ofi_ctrl_ack, credits,
				       ctrl->rx_key, peer->conn_data,
				       ctrl->conn_id);
			goto repost;
		}
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
	rxd_ep_repost_buff(rx_buf);
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
			ret = rxd_mr_verify(rxd_ep_domain(ep),
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
		tx_entry = rxd_tx_entry_alloc(ep, peer, rx_entry->peer, 0,
						RXD_TX_READ_RSP);
		if (!tx_entry) {
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "no free tx-entry\n");
			return -FI_ENOMEM;
		}

		tx_entry->peer = rx_entry->peer;
		tx_entry->read_rsp.iov_count = rx_entry->op_hdr.iov_count;
		for (i = 0; i < rx_entry->op_hdr.iov_count; i++) {
			ret = rxd_mr_verify(rxd_ep_domain(ep),
					    rma_iov[i].len,
					    (uintptr_t *) &rma_iov[i].addr,
					    rma_iov[i].key, FI_REMOTE_READ);
			if (ret) {
				/* todo: handle invalid key case */
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "invalid key/access permissions\n");
				return -FI_EACCES;
			}

			tx_entry->read_rsp.src_iov[i].iov_base = (void *) (uintptr_t)
								rma_iov[i].addr;
			tx_entry->read_rsp.src_iov[i].iov_len = rma_iov[i].len;
		}
		tx_entry->read_rsp.peer_msg_id = ctrl->msg_id;
		ret = rxd_ep_start_xfer(ep, peer, ofi_op_read_rsp, tx_entry);
		if (ret)
			rxd_tx_entry_free(ep, tx_entry);
		rxd_rx_entry_free(ep, rx_entry);
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

static void rxd_handle_start_data(struct rxd_ep *ep, struct rxd_peer *peer,
				  struct ofi_ctrl_hdr *ctrl,
				  struct fi_cq_msg_entry *comp,
				  struct rxd_rx_buf *rx_buf)
{
	struct rxd_rx_entry *rx_entry;
	struct rxd_pkt_data_start *pkt_start;
	int ret;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "start data- msg_id: %" PRIu64 ", segno: %d, buf: %p\n",
	       ctrl->msg_id, ctrl->seg_no, rx_buf);

	pkt_start = (struct rxd_pkt_data_start *) ctrl;
	if (pkt_start->op.version != OFI_OP_VERSION) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "op version mismatch\n");
		goto repost;
	}

	ret = rxd_check_start_pkt_order(ep, peer, ctrl, comp);
	if (ret) {
		if (ret == -FI_EALREADY) {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "duplicate pkt: %d\n",
				ctrl->seg_no);
			rxd_handle_dup_datastart(ep, ctrl, rx_buf);
			goto repost;
		} else {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "unexpected pkt: %d\n",
				ctrl->seg_no);
			goto repost;
		}
	}

	rx_entry = rxd_rx_entry_alloc(ep);
	if (!rx_entry)
		goto repost;

	rx_entry->peer_info = peer;
	rx_entry->op_hdr = pkt_start->op;
	rx_entry->exp_seg_no = 0;
	rx_entry->msg_id = ctrl->msg_id;
	rx_entry->done = 0;
	rx_entry->peer = ctrl->conn_id;
	rx_entry->source = (ep->util_ep.caps & FI_DIRECTED_RECV) ?
		rxd_av_fi_addr(rxd_ep_av(ep), ctrl->conn_id) : FI_ADDR_UNSPEC;
	rx_entry->credits = 1;
	rx_entry->last_win_seg = 1;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Assign rx_entry :%" PRId64 " for %" PRIx64 "\n",
	       rx_entry->key, rx_entry->msg_id);

	ep->credits--;
	ret = rxd_process_start_data(ep, rx_entry, peer, ctrl, comp, rx_buf);
	if (ret == -FI_ENOMEM)
		rxd_rx_entry_free(ep, rx_entry);
	else if (ret == -FI_ENOENT) {
		peer->exp_msg_id++;

		/* reply ack, with no window = 0 */
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Sending wait-ACK [%" PRIx64 "] - %d\n",
		       ctrl->msg_id, ctrl->seg_no);
		goto out;
	} else {
		peer->exp_msg_id++;
	}

repost:
	rxd_ep_repost_buff(rx_buf);
out:
	assert(rxd_reposted_bufs);
	return;
}

void rxd_handle_recv_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct ofi_ctrl_hdr *ctrl;
	struct rxd_rx_buf *rx_buf;
	struct rxd_peer *peer;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got recv completion\n");

	assert(rxd_reposted_bufs);
	rxd_reposted_bufs--;

	rx_buf = container_of(comp->op_context, struct rxd_rx_buf, context);
	ctrl = (struct ofi_ctrl_hdr *) rx_buf->buf;
	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);

	if (ctrl->version != OFI_CTRL_VERSION) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "ctrl version mismatch\n");
		return;
	}

	switch (ctrl->type) {
	case ofi_ctrl_connreq:
		rxd_handle_conn_req(ep, ctrl, comp, rx_buf);
		break;
	case ofi_ctrl_ack:
		rxd_handle_ack(ep, ctrl, rx_buf);
		break;
	case ofi_ctrl_discard:
		rxd_handle_discard(ep, ctrl, rx_buf);
		break;
	case ofi_ctrl_connresp:
		rxd_handle_connect_ack(ep, ctrl, rx_buf);
		break;
	case ofi_ctrl_start_data:
		rxd_handle_start_data(ep, peer, ctrl, comp, rx_buf);
		break;
	case ofi_ctrl_data:
		rxd_handle_data(ep, peer, ctrl, comp, rx_buf);
		break;
	default:
		rxd_ep_repost_buff(rx_buf);
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"invalid ctrl type %u\n", ctrl->type);
	}

	rxd_check_waiting_rx(ep);
}

void rxd_handle_send_comp(struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_meta *pkt_meta;

	pkt_meta = container_of(comp->op_context, struct rxd_pkt_meta, context);
	if (pkt_meta->flags & (RXD_REMOTE_ACK | RXD_NOT_ACKED))
		rxd_tx_pkt_free(pkt_meta);
	else
		pkt_meta->flags |= RXD_LOCAL_COMP;
}

static int rxd_cq_close(struct fid *fid)
{
	int ret;
	struct rxd_cq *cq;

	cq = container_of(fid, struct rxd_cq, util_cq.cq_fid.fid);
	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;
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

static struct fi_ops_cq rxd_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = ofi_cq_sread,
	.sreadfrom = ofi_cq_sreadfrom,
	.signal = ofi_cq_signal,
	.strerror = rxd_cq_strerror,
};

int rxd_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct rxd_cq *cq;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&rxd_prov, domain, attr, &cq->util_cq,
			  &ofi_cq_progress, context);
	if (ret)
		goto free;

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
		goto cleanup;
	}

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &rxd_cq_fi_ops;
	(*cq_fid)->ops = &rxd_cq_ops;
	return 0;

cleanup:
	ofi_cq_cleanup(&cq->util_cq);
free:
	free(cq);
	return ret;
}
