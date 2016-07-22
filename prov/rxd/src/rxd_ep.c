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
#include <fi_mem.h>
#include "rxd.h"

static ssize_t	rxd_ep_cancel(fid_t fid, void *context)
{
	struct rxd_ep *ep;
	struct dlist_entry *entry, *next;
	struct rxd_recv_entry *recv_entry;
	struct rxd_trecv_entry *trecv_entry;
	struct fi_cq_err_entry err_entry = {0};

	ep = container_of(fid, struct rxd_ep, ep.fid);
	rxd_ep_lock_if_required(ep);
	for (entry = ep->recv_list.next; entry != &ep->recv_list; entry = next) {
		next = entry->next;
		recv_entry = container_of(entry, struct rxd_recv_entry, entry);
		if (recv_entry->msg.context != context)
			continue;

		dlist_remove(entry);
		err_entry.op_context = recv_entry->msg.context;
		err_entry.flags = (FI_MSG | FI_RECV);
		err_entry.err = FI_ECANCELED;
		err_entry.prov_errno = -FI_ECANCELED;
		rxd_cq_report_error(ep->rx_cq, &err_entry);
		goto out;
	}

	for (entry = ep->trecv_list.next; entry != &ep->trecv_list; entry = next) {
		next = entry->next;
		trecv_entry = container_of(entry, struct rxd_trecv_entry, entry);
		if (trecv_entry->msg.context != context)
			continue;

		dlist_remove(entry);
		err_entry.op_context = trecv_entry->msg.context;
		err_entry.flags = (FI_MSG | FI_RECV | FI_TAGGED);
		err_entry.tag = trecv_entry->msg.tag;
		err_entry.err = FI_ECANCELED;
		err_entry.prov_errno = -FI_ECANCELED;
		rxd_cq_report_error(ep->rx_cq, &err_entry);
		goto out;
	}

out:
	rxd_ep_unlock_if_required(ep);
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

static ssize_t rxd_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			       uint64_t flags)
{
	ssize_t ret = 0, i;
	struct rxd_ep *rxd_ep;
	struct rxd_recv_entry *recv_entry;

	rxd_ep = container_of(ep, struct rxd_ep, ep);

	rxd_ep_lock_if_required(rxd_ep);
	if (freestack_isempty(rxd_ep->recv_fs)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	recv_entry = freestack_pop(rxd_ep->recv_fs);
	recv_entry->msg = *msg;
	recv_entry->flags = flags;
	recv_entry->msg.addr = (rxd_ep->caps & FI_DIRECTED_RECV) ?
		recv_entry->msg.addr : FI_ADDR_UNSPEC;
	for (i = 0; i < msg->iov_count; i++) {
		recv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		recv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post recv: %u\n",
			msg->msg_iov[i].iov_len);
	}

	dlist_init(&recv_entry->entry);
	dlist_insert_tail(&recv_entry->entry, &rxd_ep->recv_list);

	if (!dlist_empty(&rxd_ep->unexp_msg_list)) {
		rxd_ep_check_unexp_msg_list(rxd_ep, recv_entry);
	}
out:
	rxd_ep_unlock_if_required(rxd_ep);
	return ret;
}

static ssize_t rxd_ep_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;
	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;
	return rxd_ep_recvmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			     size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;
	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;
	return rxd_ep_recvmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static inline void *rxd_mr_desc(struct fid_mr *mr, struct rxd_ep *ep)
{
	return (ep->do_local_mr) ? fi_mr_desc(mr) : NULL;
}

int rxd_ep_repost_buff(struct rxd_rx_buf *buf)
{
	int ret;
	ret = fi_recv(buf->ep->dg_ep, buf->buf, buf->ep->domain->max_mtu_sz,
		      rxd_mr_desc(buf->mr, buf->ep),
		      FI_ADDR_UNSPEC, &buf->context);
	if (ret)
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "failed to repost\n");
	return ret;
}

static int rxd_ep_set_conn_id(struct rxd_ep *ep)
{
	int ret;
	ep->name = calloc(1, ep->addrlen);
	if (!ep->name)
		return -FI_ENOMEM;

	ret = fi_getname(&ep->dg_ep->fid, ep->name, &ep->addrlen);
	if (ret)
		return -FI_EINVAL;

	ret = rxd_av_dg_reverse_lookup(ep->av, 0, ep->name, ep->addrlen,
					&ep->conn_data);
	if (!ret)
		ep->conn_data_set = 1;
	return 0;
}

int rxd_ep_enable(struct rxd_ep *ep)
{
	ssize_t i, ret;
	struct fid_mr *mr = NULL;
	struct rxd_rx_buf *rx_buf;

	ret = fi_enable(ep->dg_ep);
	if (ret)
		return ret;

	rxd_ep_lock_if_required(ep);
	ret = rxd_ep_set_conn_id(ep);
	if (ret)
		goto out;

	ep->credits = ep->rx_size;
	for (i = 0; i < ep->rx_size; i++) {
		rx_buf = ep->do_local_mr ?
			util_buf_get_ex(ep->rx_pkt_pool, (void **)&mr) :
			util_buf_get(ep->rx_pkt_pool);

		if (!rx_buf) {
			ret = -FI_ENOMEM;
			goto out;
		}

		rx_buf->mr = mr;
		rx_buf->ep = ep;
		ret = rxd_ep_repost_buff(rx_buf);
		if (ret)
			goto out;
		slist_insert_tail(&rx_buf->entry, &ep->rx_pkt_list);
	}

	fastlock_acquire(&ep->domain->lock);
	dlist_insert_tail(&ep->dom_entry, &ep->domain->ep_list);
	fastlock_release(&ep->domain->lock);
	ret = 0;
out:
	rxd_ep_unlock_if_required(ep);
	return ret;
}

struct rxd_peer *rxd_ep_getpeer_info(struct rxd_ep *ep, fi_addr_t addr)
{
	return &ep->peer_info[addr];
}

void rxd_ep_lock_if_required(struct rxd_ep *ep)
{
	/* todo: do locking based on threading model */
	fastlock_acquire(&ep->lock);
}

void rxd_ep_unlock_if_required(struct rxd_ep *ep)
{
	/* todo: do unlocking based on threading model */
	fastlock_release(&ep->lock);
}

size_t rxd_get_msg_len(const struct iovec *iov, size_t iov_count)
{
	size_t i, ret = 0;
	for (i = 0; i < iov_count; i++)
		ret += iov[i].iov_len;
	return ret;
}

static void rxd_init_ctrl_hdr(struct ofi_ctrl_hdr *ctrl,
			      uint8_t type, uint16_t seg_size,
			      uint32_t seg_no, uint64_t msg_id,
			      uint64_t rx_key, uint64_t source)
{
	ctrl->version = OFI_CTRL_VERSION;
	ctrl->type = type;
	ctrl->seg_size = seg_size;
	ctrl->seg_no = seg_no;
	ctrl->msg_id = msg_id;
	ctrl->rx_key = rx_key;
	ctrl->conn_id = source;
}

static void rxd_init_op_hdr(struct ofi_op_hdr *op, uint64_t data,
			    uint64_t msg_sz, uint8_t rx_index,
			    uint8_t op_type, uint64_t tag, uint32_t flags)
{
	op->version = OFI_OP_VERSION;
	op->rx_index = rx_index;
	op->op = op_type;
	op->op_data = 0; /* unused */
	op->flags = flags;
	op->size = msg_sz;
	op->data = data;
	op->tag = tag;
}

static uint32_t rxd_prepare_tx_flags(uint64_t flags)
{
	uint32_t tx_flags = 0;
	if (flags & FI_REMOTE_CQ_DATA)
		tx_flags = OFI_REMOTE_CQ_DATA;
	if (flags & FI_TRANSMIT_COMPLETE)
		tx_flags |= OFI_TRANSMIT_COMPLETE;
	if (flags & FI_DELIVERY_COMPLETE)
		tx_flags |= OFI_DELIVERY_COMPLETE;
	return tx_flags;
}

uint64_t rxd_ep_copy_iov_buf(const struct iovec *iov, size_t iov_count,
			     void *buf, uint64_t data_sz, uint64_t skip, int dir)
{
	int i;
	uint64_t rem, offset, len, iov_offset;
	offset = 0, rem = data_sz, iov_offset = 0;

	for (i = 0; i < iov_count; i++) {
		len = iov[i].iov_len;
		iov_offset = 0;

		if (skip) {
			iov_offset = MIN(skip, len);
			skip -= iov_offset;
			len -= iov_offset;
		}

		len = MIN(rem, len);
		if (dir == RXD_COPY_BUF_TO_IOV)
			memcpy((char *) iov[i].iov_base + iov_offset,
			       (char *) buf + offset, len);
		else if (dir == RXD_COPY_IOV_TO_BUF)
			memcpy((char *) buf + offset,
			       (char *) iov[i].iov_base + iov_offset, len);
		rem -= len, offset += len;
	}
	return offset;
}

struct rxd_pkt_meta *rxd_tx_pkt_acquire(struct rxd_ep *ep)
{
	struct rxd_pkt_meta *pkt_meta;
	struct fid_mr *mr = NULL;

	pkt_meta = ep->do_local_mr ?
		util_buf_alloc_ex(ep->tx_pkt_pool, (void **)&mr) :
		util_buf_alloc(ep->tx_pkt_pool);

	if (!pkt_meta) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "No free tx pkt\n");
		return NULL;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Acquired tx pkt: %p\n", pkt_meta);
	pkt_meta->ep = ep;
	pkt_meta->retries = 0;
	pkt_meta->mr = mr;
	pkt_meta->ref = 0;
	return pkt_meta;
}

ssize_t rxd_ep_post_data_msg(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	int ret;
	uint64_t data_sz;
	size_t done;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;
	struct rxd_peer *peer;

	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);
	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	data_sz = MIN(RXD_MAX_DATA_PKT_SZ(ep), tx_entry->op_hdr.size - tx_entry->done);

	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_data, data_sz, tx_entry->nxt_seg_no,
			   tx_entry->msg_id, tx_entry->rx_key, peer->conn_data);

	if (tx_entry->op_hdr.op == ofi_op_msg)
		done = rxd_ep_copy_iov_buf(tx_entry->msg.msg_iov,
					   tx_entry->msg.msg.iov_count,
					   pkt->data, data_sz, tx_entry->done,
					   RXD_COPY_IOV_TO_BUF);
	else
		done = rxd_ep_copy_iov_buf(tx_entry->tmsg.msg_iov,
					   tx_entry->tmsg.tmsg.iov_count,
					   pkt->data, data_sz, tx_entry->done,
					   RXD_COPY_IOV_TO_BUF);

	pkt_meta->tx_entry = tx_entry;
	pkt_meta->type = (tx_entry->op_hdr.size == tx_entry->done + done) ?
		RXD_PKT_LAST : RXD_PKT_DATA;
	pkt_meta->us_stamp = fi_gettime_us();

	ret = fi_send(ep->dg_ep, pkt, data_sz + RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep), tx_entry->peer, &pkt_meta->context);
	if (ret) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "send %d failed\n", pkt->ctrl.seg_no);
		util_buf_release(ep->tx_pkt_pool, pkt_meta);
		return ret;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sent data %p, %d [on buf: %p]\n",
		pkt->ctrl.msg_id, pkt->ctrl.seg_no, pkt_meta);
	tx_entry->done += done;
	tx_entry->win_sz--;
	tx_entry->nxt_seg_no++;
	tx_entry->num_unacked++;

	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);
	ep->num_out++;
	return 0;
}

void rxd_ep_free_acked_pkts(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			  uint32_t seg_no)
{
	struct dlist_entry *next, *curr;
	struct rxd_pkt_meta *pkt;
	struct ofi_ctrl_hdr *ctrl;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "freeing all [%p] pkts <= %d\n",
		tx_entry->msg_id, seg_no);
	for (curr = tx_entry->pkt_list.next; curr != &tx_entry->pkt_list;) {
		next = curr->next;
		pkt = container_of(curr, struct rxd_pkt_meta, entry);
		ctrl = (struct ofi_ctrl_hdr *) pkt->pkt_data;
		if (ctrl->seg_no <= seg_no) {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "freeing [%p] pkt:%d\n",
				tx_entry->msg_id, ctrl->seg_no);
			dlist_remove(curr);
			RXD_PKT_MARK_REMOTE_ACK(pkt);
			rxd_tx_pkt_release(pkt);
			tx_entry->num_unacked--;
		} else {
			break;
		}
		curr = next;
	}
}

void rxd_tx_entry_update_ts(struct rxd_tx_entry *tx_entry, uint32_t seg_no)
{
	struct dlist_entry *curr;
	struct rxd_pkt_meta *pkt;
	struct ofi_ctrl_hdr *ctrl;

	for (curr = tx_entry->pkt_list.next; curr != &tx_entry->pkt_list;
	     curr = curr->next) {
		pkt = container_of(curr, struct rxd_pkt_meta, entry);
		ctrl = (struct ofi_ctrl_hdr *) pkt->pkt_data;
		if (ctrl->seg_no == seg_no) {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "updating TS of [%p] pkt:%d\n",
				tx_entry->msg_id, seg_no);
			pkt->us_stamp += RXD_WAIT_TIMEOUT;
			break;
		}
	}
}

int rxd_progress_tx(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	int ret = 0;

	switch (tx_entry->op_hdr.op) {
	case ofi_op_msg:
	case ofi_op_tagged:
		ret = rxd_ep_post_data_msg(ep, tx_entry);
		break;
	default:
		break;
	}
	return ret;
}

void rxd_tx_entry_discard(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	rxd_cq_report_tx_comp(ep->tx_cq, tx_entry);
	rxd_tx_entry_done(ep, tx_entry);
}

void rxd_resend_pkt(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
		     uint32_t seg_no)
{
	struct dlist_entry *pkt_item;
	struct rxd_pkt_meta *pkt;
	struct ofi_ctrl_hdr *ctrl;
	uint64_t curr_stamp = fi_gettime_us();

	dlist_foreach(&tx_entry->pkt_list, pkt_item) {
		pkt = container_of(pkt_item, struct rxd_pkt_meta, entry);
		ctrl = (struct ofi_ctrl_hdr *)pkt->pkt_data;

		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "check pkt %d, %p\n",
			ctrl->seg_no, ctrl->msg_id);

		if (ctrl->seg_no == seg_no) {

			if (curr_stamp < pkt->us_stamp ||
			    (curr_stamp - pkt->us_stamp) <
			    (1 << (pkt->retries + 1)) * RXD_RETRY_TIMEOUT) {
				break;
			}

			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "resending pkt %d, %p\n",
				ctrl->seg_no, ctrl->msg_id);

			pkt->us_stamp = fi_gettime_us();
			rxd_ep_retry_pkt(ep, tx_entry, pkt);
			break;
		}
	}
}

int rxd_tx_entry_progress(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			   struct ofi_ctrl_hdr *ack)
{
	if (ack) {
		tx_entry->rx_key = ack->rx_key;
		tx_entry->win_sz += ack->seg_size;

		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "tx: %p [%p] - avail_win: %d\n",
			tx_entry, ack->msg_id, ack->seg_size);

		if (ack->type == ofi_ctrl_nack) {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got NACK for %d, %p\n",
				ack->seg_no, ack->msg_id);
			if (ack->seg_no > 0)
				rxd_ep_free_acked_pkts(ep, tx_entry, ack->seg_no - 1);
			rxd_resend_pkt(ep, tx_entry, ack->seg_no);
		} else {
			FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got ACK for %d, %p\n",
			       ack->seg_no, ack->msg_id);

			rxd_ep_free_acked_pkts(ep, tx_entry, ack->seg_no);
			if (ack->seg_size == 0 &&
			    tx_entry->done != tx_entry->op_hdr.size) {
				tx_entry->is_waiting = 1;
				tx_entry->retry_stamp = fi_gettime_us();
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
					"Marking [%p] as waiting\n", tx_entry->msg_id);
			}
		}
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "tx: %p [%p] - num_unacked: %d\n",
		tx_entry, tx_entry->msg_id, tx_entry->num_unacked);

	while (tx_entry->win_sz && tx_entry->num_unacked < RXD_MAX_UNACKED &&
	       tx_entry->done != tx_entry->op_hdr.size) {
		if (rxd_progress_tx(ep, tx_entry))
			break;
	}
	return 0;
}

int rxd_ep_reply_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		   uint8_t type, uint16_t seg_size, uint64_t rx_key,
		   uint64_t source, fi_addr_t dest)
{
	ssize_t ret;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;

	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	rxd_init_ctrl_hdr(&pkt->ctrl, type, seg_size, in_ctrl->seg_no,
			   in_ctrl->msg_id, rx_key, source);

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sending ack [%p] - %d, %d\n",
		in_ctrl->msg_id, in_ctrl->seg_no, seg_size);

	RXD_PKT_MARK_REMOTE_ACK(pkt_meta);
	pkt_meta->us_stamp = fi_gettime_us();
	ret = fi_send(ep->dg_ep, pkt, RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      dest, &pkt_meta->context);
	if (ret)
		goto err;
	ep->num_out++;
	return 0;
err:
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

int rxd_ep_reply_nack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		    uint32_t seg_no, uint64_t rx_key,
		    uint64_t source, fi_addr_t dest)
{
	ssize_t ret;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;

	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_nack, 0, seg_no,
			   in_ctrl->msg_id, rx_key, source);

	RXD_PKT_MARK_REMOTE_ACK(pkt_meta);
	pkt_meta->us_stamp = fi_gettime_us();
	ret = fi_send(ep->dg_ep, pkt, RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      dest, &pkt_meta->context);
	if (ret)
		goto err;
	ep->num_out++;
	return 0;
err:
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

int rxd_ep_reply_discard(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		       uint32_t seg_no, uint64_t rx_key,
		       uint64_t source, fi_addr_t dest)
{
	ssize_t ret;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;

	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_discard, 0, seg_no,
			   in_ctrl->msg_id, rx_key, source);

	RXD_PKT_MARK_REMOTE_ACK(pkt_meta);
	pkt_meta->us_stamp = fi_gettime_us();
	ret = fi_send(ep->dg_ep, pkt, RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      dest, &pkt_meta->context);
	if (ret)
		goto err;
	ep->num_out++;
	return 0;
err:
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

#define RXD_TX_ENTRY_ID(ep, tx_entry) (tx_entry - &ep->tx_entry_fs->buf[0])

ssize_t rxd_ep_post_start_msg(struct rxd_ep *ep, struct rxd_peer *peer,
			       uint8_t op, struct rxd_tx_entry *tx_entry)
{
	ssize_t ret;
	uint32_t flags;
	uint64_t msg_sz;
	uint64_t data_sz;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data_start *pkt;

	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt = (struct rxd_pkt_data_start *)pkt_meta->pkt_data;
	flags = rxd_prepare_tx_flags(tx_entry->flags);
	tx_entry->msg_id = RXD_TX_ID(peer->nxt_msg_id, RXD_TX_ENTRY_ID(ep, tx_entry));

	if (op == ofi_op_msg) {
		msg_sz = rxd_get_msg_len(tx_entry->msg.msg_iov, tx_entry->msg.msg.iov_count);
		data_sz = MIN(RXD_MAX_STRT_DATA_PKT_SZ(ep), msg_sz);
		rxd_init_op_hdr(&pkt->op, tx_entry->msg.msg.data, msg_sz, 0, op, 0, flags);
		tx_entry->done = rxd_ep_copy_iov_buf(tx_entry->msg.msg_iov,
						     tx_entry->msg.msg.iov_count,
						     pkt->data, data_sz, 0, RXD_COPY_IOV_TO_BUF);
	} else {
		msg_sz = rxd_get_msg_len(tx_entry->tmsg.msg_iov, tx_entry->tmsg.tmsg.iov_count);
		data_sz = MIN(RXD_MAX_STRT_DATA_PKT_SZ(ep), msg_sz);
		rxd_init_op_hdr(&pkt->op, tx_entry->tmsg.tmsg.data, msg_sz, 0, op,
				 tx_entry->tmsg.tmsg.tag,flags);
		tx_entry->done = rxd_ep_copy_iov_buf(tx_entry->tmsg.msg_iov,
						     tx_entry->tmsg.tmsg.iov_count,
						     pkt->data, data_sz, 0, RXD_COPY_IOV_TO_BUF);
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sent start %p, tag: %p, len: %d\n",
		       pkt->ctrl.msg_id, tx_entry->tmsg.tmsg.tag, msg_sz);
	}

	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_start_data, data_sz, 0,
			   tx_entry->msg_id, peer->conn_data, peer->conn_data);
	tx_entry->nxt_seg_no = 1;
	tx_entry->op_hdr = pkt->op;
	tx_entry->win_sz = 0;

	pkt_meta->tx_entry = tx_entry;
	pkt_meta->type = (tx_entry->op_hdr.size == tx_entry->done) ?
		RXD_PKT_LAST : RXD_PKT_DATA;

	pkt_meta->us_stamp = fi_gettime_us();
	ret = fi_send(ep->dg_ep, pkt, data_sz + RXD_START_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      tx_entry->peer, &pkt_meta->context);
	if (ret)
		goto err;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sent start %p, %d\n",
		pkt->ctrl.msg_id, pkt->ctrl.seg_no);
	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);
	peer->nxt_msg_id++;
	ep->num_out++;
	tx_entry->num_unacked++;
	return 0;
err:
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

uint64_t rxd_find_av_index(struct rxd_av *av, uint64_t start_idx,
			    const void *addr, size_t addrlen, int *p_found)
{
	void *tmp_addr;
	uint64_t idx = 0, count;
	size_t tmp_addrlen;

	if (p_found)
		*p_found = 0;

	tmp_addr = calloc(1, addrlen);
	if (!tmp_addr)
		return 0;

	for (idx = start_idx, count = 0; count < av->dg_av_used;
	     count++, idx = (idx+1) % av->dg_av_used) {
		tmp_addrlen = addrlen;
		if (fi_av_lookup(av->dg_av, idx, tmp_addr, &tmp_addrlen)) {
			idx = 0;
			goto out;
		}

		if (addrlen == tmp_addrlen &&
		    memcmp(tmp_addr, addr, addrlen) == 0) {
			if (p_found)
				*p_found = 1;
			goto out;
		}
	}
out:
	free(tmp_addr);
	return idx;
}

ssize_t rxd_ep_post_conn_msg(struct rxd_ep *ep, struct rxd_peer *peer,
			      fi_addr_t addr)
{
	ssize_t ret;
	size_t addrlen;
	uint16_t data_sz;
	struct rxd_pkt_data *pkt;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_tx_entry *tx_entry;

	if (peer->conn_initiated)
		return 0;

	tx_entry = rxd_tx_entry_acquire(ep, peer);
	if (!tx_entry)
		return -FI_EAGAIN;

	dlist_init(&tx_entry->pkt_list);
	tx_entry->op_type = RXD_TX_CONN;
	tx_entry->peer = addr;
	tx_entry->win_sz = 0;

	pkt_meta = rxd_tx_pkt_acquire(ep);
	if (!pkt_meta) {
		rxd_tx_entry_release(ep, tx_entry);
		return -FI_ENOMEM;
	}

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	addrlen = RXD_MAX_DATA_PKT_SZ(ep);
	ret = fi_getname(&ep->dg_ep->fid, pkt->data, &addrlen);
	assert(ret == 0);
	data_sz = (uint16_t) addrlen;
	tx_entry->msg_id = RXD_TX_ID(peer->nxt_msg_id, RXD_TX_ENTRY_ID(ep, tx_entry));

	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_connreq, data_sz, 0,
			   tx_entry->msg_id, ep->conn_data, addr);

	pkt_meta->us_stamp = fi_gettime_us();
	ret = fi_send(ep->dg_ep, pkt, data_sz + RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      addr, &pkt_meta->context);
	if (ret)
		goto err;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sent conn %p\n", pkt->ctrl.msg_id);
	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);
	dlist_insert_tail(&tx_entry->entry, &ep->tx_entry_list);
	peer->nxt_msg_id++;
	ep->num_out++;
	peer->conn_initiated = 1;
	return 0;
err:
	rxd_tx_entry_release(ep, tx_entry);
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

struct rxd_tx_entry *rxd_tx_entry_acquire(struct rxd_ep *ep, struct rxd_peer *peer)
{
	struct rxd_tx_entry *tx_entry;
	if (freestack_isempty(ep->tx_entry_fs) ||
	    peer->num_msg_out == RXD_MAX_OUT_TX_MSG)
		return NULL;

	peer->num_msg_out++;
	tx_entry = freestack_pop(ep->tx_entry_fs);
	tx_entry->num_unacked = 0;
	tx_entry->is_waiting = 0;
	dlist_init(&tx_entry->entry);
	return tx_entry;
}

void rxd_tx_entry_release(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	struct rxd_peer *peer;

	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);
	peer->num_msg_out--;
	dlist_remove(&tx_entry->entry);
	freestack_push(ep->tx_entry_fs, tx_entry);
}

static inline void rxd_copy_iov(const struct iovec *src_iov,
				 struct iovec *dst_iov, size_t iov_count)
{
	size_t i;
	for (i = 0; i < iov_count; i++)
		dst_iov[i] = src_iov[i];
}

static ssize_t rxd_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			       uint64_t flags)
{
	ssize_t ret;
	uint64_t peer_addr;
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	rxd_ep = container_of(ep, struct rxd_ep, ep);

	peer_addr = rxd_av_get_dg_addr(rxd_ep->av, msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	rxd_ep_lock_if_required(rxd_ep);
	if (!peer->addr_published) {
		ret = rxd_ep_post_conn_msg(rxd_ep, peer, peer_addr);
		ret = (ret) ? ret : -FI_EAGAIN;
		goto out;
	}

	tx_entry = rxd_tx_entry_acquire(rxd_ep, peer);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	dlist_init(&tx_entry->pkt_list);
	tx_entry->op_type = RXD_TX_MSG;
	tx_entry->msg.msg = *msg;
	tx_entry->flags = flags;
	tx_entry->peer = peer_addr;
	rxd_copy_iov(msg->msg_iov, &tx_entry->msg.msg_iov[0], msg->iov_count);

	ret = rxd_ep_post_start_msg(rxd_ep, peer, ofi_op_msg, tx_entry);
	if (ret)
		goto err;

	dlist_insert_tail(&tx_entry->entry, &rxd_ep->tx_entry_list);
out:
	rxd_ep_unlock_if_required(rxd_ep);
	return ret;
err:
	rxd_tx_entry_release(rxd_ep, tx_entry);
	goto out;
}

static ssize_t rxd_ep_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			    fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;
	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;

	return rxd_ep_sendmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			     size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;
	return rxd_ep_sendmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t	rxd_ep_inject(struct fid_ep *ep, const void *buf, size_t len,
			       fi_addr_t dest_addr)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.addr = dest_addr;

	return rxd_ep_sendmsg(ep, &msg, FI_INJECT |
			       RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
				uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.data = data;

	return rxd_ep_sendmsg(ep, &msg, FI_REMOTE_CQ_DATA | RXD_USE_OP_FLAGS);
}

static ssize_t	rxd_ep_injectdata(struct fid_ep *ep, const void *buf, size_t len,
				   uint64_t data, fi_addr_t dest_addr)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;

	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.data = data;

	return rxd_ep_sendmsg(ep, &msg, FI_REMOTE_CQ_DATA | FI_INJECT |
			       RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
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

static int rxd_peek_trecv(struct dlist_entry *item, const void *arg)
{
	const struct fi_msg_tagged *msg = (const struct fi_msg_tagged *) arg;
	struct rxd_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_rx_entry, unexp_entry);
	return ((rx_entry->op_hdr.tag | msg->ignore) ==
		(msg->tag | msg->ignore) &&
                ((rx_entry->source == FI_ADDR_UNSPEC) ||
		 (msg->addr == FI_ADDR_UNSPEC) ||
                 (rx_entry->source == msg->addr)));
}

static void rxd_trx_discard_recv(struct rxd_ep *ep,
				  struct rxd_rx_entry *rx_entry)
{
	struct rxd_rx_buf *rx_buf;
	struct ofi_ctrl_hdr *ctrl;
	struct rxd_peer *peer;

	rx_buf = rx_entry->unexp_buf;
	ctrl = (struct ofi_ctrl_hdr *) rx_buf->buf;
	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);

	dlist_remove(&rx_entry->unexp_entry);
	ep->num_unexp_msg--;

	rxd_ep_reply_discard(ep, ctrl, 0, ctrl->rx_key, peer->conn_data, ctrl->conn_id);
	rxd_rx_entry_release(ep, rx_entry);
	rxd_ep_repost_buff(rx_buf);
}

static ssize_t rxd_trx_peek_recv(struct rxd_ep *ep,
				  const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct dlist_entry *match;
	struct rxd_rx_entry *rx_entry;
	struct fi_cq_err_entry err_entry = {0};
	struct fi_cq_tagged_entry cq_entry = {0};
	struct fi_context *context;

	match = dlist_find_first_match(&ep->unexp_tag_list,
				       &rxd_peek_trecv, msg);
	if (!match) {
		err_entry.op_context = msg->context;
		err_entry.flags = (FI_MSG | FI_RECV | FI_TAGGED);
		err_entry.tag = msg->tag;
		err_entry.err = FI_ENOMSG;
		err_entry.prov_errno = -FI_ENOMSG;
		rxd_cq_report_error(ep->rx_cq, &err_entry);
		return 0;
	}

	rx_entry = container_of(match, struct rxd_rx_entry, unexp_entry);
	cq_entry.flags = (FI_MSG | FI_RECV | FI_TAGGED);
	cq_entry.op_context = msg->context;
	cq_entry.len = rx_entry->op_hdr.size;
	cq_entry.buf = NULL;
	cq_entry.data = rx_entry->op_hdr.data;
	cq_entry.tag = rx_entry->op_hdr.tag;

	if (flags & FI_CLAIM) {
		context = (struct fi_context *)msg->context;
		context->internal[0] = rx_entry;
		dlist_remove(match);
	} else if (flags & FI_DISCARD) {
		rxd_trx_discard_recv(ep, rx_entry);
	}

	ep->rx_cq->write_fn(ep->rx_cq, &cq_entry);
	return 0;
}

ssize_t rxd_trx_claim_recv(struct rxd_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	int ret = 0, i;
	struct fi_context *context;
	struct rxd_rx_entry *rx_entry;
	struct rxd_trecv_entry *trecv_entry;
	struct rxd_peer *peer;
	struct rxd_rx_buf *rx_buf;
	struct ofi_ctrl_hdr *ctrl;
	struct rxd_pkt_data_start *pkt_start;

	rxd_ep_lock_if_required(ep);
	if (freestack_isempty(ep->trecv_fs)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	trecv_entry = freestack_pop(ep->trecv_fs);
	trecv_entry->msg = *msg;
	trecv_entry->msg.addr = (ep->caps & FI_DIRECTED_RECV) ?
		msg->addr : FI_ADDR_UNSPEC;
	trecv_entry->flags = flags;
	for (i = 0; i < msg->iov_count; i++) {
		trecv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		trecv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post claim trecv: %u, tag: %p\n",
		       msg->msg_iov[i].iov_len, msg->tag);
	}

	context = (struct fi_context *) msg->context;
	rx_entry = context->internal[0];
	rx_entry->trecv = trecv_entry;

	rx_buf = rx_entry->unexp_buf;
	peer = rx_entry->peer_info;
	ctrl = (struct ofi_ctrl_hdr *) rx_buf->buf;
	pkt_start = (struct rxd_pkt_data_start *) ctrl;

	rxd_ep_handle_data_msg(ep, peer, rx_entry, rx_entry->trecv->iov,
			     rx_entry->trecv->msg.iov_count, ctrl,
			     pkt_start->data, rx_buf);
out:
	rxd_ep_unlock_if_required(ep);
	return ret;
}

ssize_t rxd_ep_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			 uint64_t flags)
{
	ssize_t ret = 0, i;
	struct rxd_ep *rxd_ep;
	struct rxd_trecv_entry *trecv_entry;

	rxd_ep = container_of(ep, struct rxd_ep, ep);
	rxd_ep_lock_if_required(rxd_ep);

	if (flags & FI_PEEK) {
		ret = rxd_trx_peek_recv(rxd_ep, msg, flags);
		goto out;
	} else if (flags & FI_CLAIM) {
		ret = rxd_trx_claim_recv(rxd_ep, msg, flags);
		goto out;
	}

	if (freestack_isempty(rxd_ep->trecv_fs)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	trecv_entry = freestack_pop(rxd_ep->trecv_fs);
	trecv_entry->msg = *msg;
	trecv_entry->msg.addr = (rxd_ep->caps & FI_DIRECTED_RECV) ?
		msg->addr : FI_ADDR_UNSPEC;
	trecv_entry->flags = flags;
	for (i = 0; i < msg->iov_count; i++) {
		trecv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		trecv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post trecv: %u, tag: %p\n",
			msg->msg_iov[i].iov_len, msg->tag);
	}
	dlist_init(&trecv_entry->entry);
	dlist_insert_tail(&trecv_entry->entry, &rxd_ep->trecv_list);

	if (!dlist_empty(&rxd_ep->unexp_tag_list)) {
		rxd_ep_check_unexp_tag_list(rxd_ep, trecv_entry);
	}
out:
	rxd_ep_unlock_if_required(rxd_ep);
	return ret;
}

static ssize_t rxd_ep_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr,
			    uint64_t tag, uint64_t ignore, void *context)
{
	struct fi_msg_tagged msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;
	msg.tag = tag;
	msg.ignore = ignore;
	msg.data = 0;
	return rxd_ep_trecvmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

ssize_t rxd_ep_trecvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t src_addr,
		       uint64_t tag, uint64_t ignore, void *context)
{
	struct fi_msg_tagged msg;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;
	msg.tag = tag;
	msg.ignore = ignore;
	msg.data = 0;
	return rxd_ep_trecvmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

ssize_t rxd_ep_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			 uint64_t flags)
{
	ssize_t ret;
	uint64_t peer_addr;
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	rxd_ep = container_of(ep, struct rxd_ep, ep);

	peer_addr = rxd_av_get_dg_addr(rxd_ep->av, msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	rxd_ep_lock_if_required(rxd_ep);
	if (!peer->addr_published) {
		ret = rxd_ep_post_conn_msg(rxd_ep, peer, peer_addr);
		ret = (ret) ? ret : -FI_EAGAIN;
		goto out;
	}

	tx_entry = rxd_tx_entry_acquire(rxd_ep, peer);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	dlist_init(&tx_entry->pkt_list);
	tx_entry->op_type = RXD_TX_TAG;
	tx_entry->tmsg.tmsg = *msg;
	tx_entry->flags = flags;
	tx_entry->peer = peer_addr;
	rxd_copy_iov(msg->msg_iov, &tx_entry->tmsg.msg_iov[0], msg->iov_count);

	ret = rxd_ep_post_start_msg(rxd_ep, peer, ofi_op_tagged, tx_entry);
	if (ret)
		goto err;

	dlist_insert_tail(&tx_entry->entry, &rxd_ep->tx_entry_list);
out:
	rxd_ep_unlock_if_required(rxd_ep);
	return ret;
err:
	rxd_tx_entry_release(rxd_ep, tx_entry);
	goto out;
}

ssize_t rxd_ep_tsend(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		      fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct fi_msg_tagged msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.tag = tag;

	return rxd_ep_tsendmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

ssize_t rxd_ep_tsendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct fi_msg_tagged msg;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.tag = tag;
	return rxd_ep_tsendmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

ssize_t	rxd_ep_tinject(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	struct fi_msg_tagged msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.tag = tag;
	return rxd_ep_tsendmsg(ep, &msg, FI_INJECT |
				RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
}

ssize_t rxd_ep_tsenddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			  uint64_t data, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct fi_msg_tagged msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.data = data;
	msg.tag = tag;

	return rxd_ep_tsendmsg(ep, &msg, FI_REMOTE_CQ_DATA | RXD_USE_OP_FLAGS);
}

ssize_t	rxd_ep_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
			    uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct fi_msg_tagged msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;

	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.data = data;
	msg.tag = tag;

	return rxd_ep_tsendmsg(ep, &msg, FI_REMOTE_CQ_DATA | FI_INJECT |
				RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
}

struct fi_ops_tagged rxd_ops_tagged = {
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

static void rxd_ep_free_buf_pools(struct rxd_ep *ep)
{
	util_buf_pool_destroy(ep->tx_pkt_pool);
	util_buf_pool_destroy(ep->rx_pkt_pool);

	if (ep->tx_entry_fs)
		rxd_tx_entry_fs_free(ep->tx_entry_fs);

	if (ep->rx_entry_fs)
		rxd_rx_entry_fs_free(ep->rx_entry_fs);

	if (ep->recv_fs)
		rxd_recv_fs_free(ep->recv_fs);

	if (ep->trecv_fs)
		rxd_trecv_fs_free(ep->trecv_fs);
}

static int rxd_ep_close(struct fid *fid)
{
	int ret;
	struct rxd_ep *ep;
	struct slist_entry *entry;
	struct rxd_rx_buf *buf;

	ep = container_of(fid, struct rxd_ep, ep.fid);
	while (ep->num_out) {
		rxd_cq_progress(&ep->tx_cq->util_cq);
	}

	ret = fi_close(&ep->dg_ep->fid);
	if (ret)
		return ret;

	fastlock_acquire(&ep->domain->lock);
	dlist_remove(&ep->dom_entry);
	fastlock_release(&ep->domain->lock);

	while(!slist_empty(&ep->rx_pkt_list)) {
		entry = slist_remove_head(&ep->rx_pkt_list);
		buf = container_of(entry, struct rxd_rx_buf, entry);
		util_buf_release(ep->rx_pkt_pool, buf);
	}

	if (ep->tx_cq)
		atomic_dec(&ep->tx_cq->util_cq.ref);

	if (ep->rx_cq)
		atomic_dec(&ep->rx_cq->util_cq.ref);

	atomic_dec(&ep->domain->util_domain.ref);
	fastlock_destroy(&ep->lock);
	rxd_ep_free_buf_pools(ep);
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

	if (((flags & FI_TRANSMIT) && ep->tx_cq) ||
	    ((flags & FI_RECV) && ep->rx_cq)) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "duplicate CQ binding\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		if (!ep->tx_cq && !ep->rx_cq) {
			ret = fi_ep_bind(ep->dg_ep, &cq->dg_cq->fid, FI_TRANSMIT | FI_RECV);
		if (ret)
			return ret;
		}

		ep->tx_cq = cq;
		atomic_inc(&cq->util_cq.ref);
	}

	if (flags & FI_RECV) {
		if (!ep->tx_cq && !ep->rx_cq) {
			ret = fi_ep_bind(ep->dg_ep, &cq->dg_cq->fid, FI_TRANSMIT | FI_RECV);
			if (ret)
				return ret;
		}

		ep->rx_cq = cq;
		atomic_inc(&cq->util_cq.ref);
	}
	return 0;
}

static int rxd_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxd_ep *ep;
	struct rxd_av *av;
	int ret = 0;

	ep = container_of(ep_fid, struct rxd_ep, ep.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		if (ep->av) {
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
			return -FI_EINVAL;
		}
		av = container_of(bfid, struct rxd_av, util_av.av_fid.fid);
		ret = fi_ep_bind(ep->dg_ep, &av->dg_av->fid, flags);
		if (ret)
			return ret;

		/* todo: handle case where AV is updated after binding */
		ep->peer_info = calloc(av->size, sizeof(struct rxd_peer));
		ep->max_peers = av->size;
		if (!ep->peer_info) {
			return -FI_ENOMEM;
		}

		ep->av = av;
		break;
	case FI_CLASS_CQ:
		ret = rxd_ep_bind_cq(ep, container_of(bfid, struct rxd_cq,
						       util_cq.cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
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
		ep = container_of(fid, struct rxd_ep, ep.fid);
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
	ep = container_of(fid, struct rxd_ep, ep.fid);
	return fi_setname(&ep->dg_ep->fid, addr, addrlen);
}

static int rxd_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxd_ep *ep;
	ep = container_of(fid, struct rxd_ep, ep.fid);
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
};

static int rxd_buf_region_alloc_hndlr(void *pool_ctx, void *addr, size_t len,
				void **context)
{
	int ret;
	struct fid_mr *mr;
	struct rxd_domain *domain = (struct rxd_domain *)pool_ctx;

	ret = fi_mr_reg(domain->dg_domain, addr, len,
			FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);
	*context = mr;
	return ret;
}

static void rxd_buf_region_free_hndlr(void *pool_ctx, void *context)
{
	fi_close((struct fid *) context);
}

int rxd_ep_create_buf_pools(struct rxd_ep *ep, struct fi_info *fi_info)
{
	ep->tx_pkt_pool = util_buf_pool_create_ex(
		ep->domain->max_mtu_sz + sizeof(struct rxd_pkt_meta),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_TX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		ep->domain);
	if (!ep->tx_pkt_pool)
		return -FI_ENOMEM;

	ep->rx_pkt_pool = util_buf_pool_create_ex(
		ep->domain->max_mtu_sz + sizeof (struct rxd_rx_buf),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_RX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		ep->domain);
	if (!ep->rx_pkt_pool)
		goto err;

	ep->tx_entry_fs = rxd_tx_entry_fs_create(1ULL << RXD_MAX_TX_BITS);
	if (!ep->tx_entry_fs)
		goto err;

	ep->rx_entry_fs = rxd_rx_entry_fs_create(1ULL << RXD_MAX_RX_BITS);
	if (!ep->rx_entry_fs)
		goto err;

	if (ep->caps & FI_MSG) {
		ep->recv_fs = rxd_recv_fs_create(ep->rx_size);
		dlist_init(&ep->recv_list);
		if (!ep->recv_fs)
			goto err;
	}

	if (ep->caps & FI_TAGGED) {
		ep->trecv_fs = rxd_trecv_fs_create(ep->rx_size);
		dlist_init(&ep->trecv_list);
		if (!ep->trecv_fs)
			goto err;
	}

	return 0;
err:
	if (ep->tx_pkt_pool)
		util_buf_pool_destroy(ep->tx_pkt_pool);

	if (ep->rx_pkt_pool)
		util_buf_pool_destroy(ep->rx_pkt_pool);

	if (ep->tx_entry_fs)
		rxd_tx_entry_fs_free(ep->tx_entry_fs);

	if (ep->rx_entry_fs)
		rxd_rx_entry_fs_free(ep->rx_entry_fs);

	if (ep->recv_fs)
		rxd_recv_fs_free(ep->recv_fs);

	if (ep->trecv_fs)
		rxd_trecv_fs_free(ep->trecv_fs);

	return -FI_ENOMEM;
}

int rxd_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context)
{
	int ret;
	struct fi_info *dg_info;
	struct rxd_ep *rxd_ep;
	struct rxd_domain *rxd_domain;

	ret = fi_check_info(&rxd_util_prov, info, FI_MATCH_PREFIX);
	if (ret)
		return ret;

	ret = ofix_getinfo(rxd_prov.version, NULL, NULL, 0, &rxd_util_prov,
			   info, rxd_alter_layer_info,
			   rxd_alter_base_info, 1, &dg_info);
	if (ret)
		return ret;

	rxd_domain = container_of(domain, struct rxd_domain, util_domain.domain_fid);
	rxd_ep = calloc(1, sizeof(*rxd_ep));
	if (!rxd_ep) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	rxd_ep->addrlen = (info->src_addr) ? info->src_addrlen : info->dest_addrlen;
	rxd_ep->do_local_mr = (rxd_domain->dg_mode & FI_LOCAL_MR) ? 1 : 0;
	ret = fi_endpoint(rxd_domain->dg_domain, dg_info, ep, context);
	if (ret)
		goto err2;

	rxd_ep->caps = info->caps;
	rxd_ep->domain = rxd_domain;
	rxd_ep->rx_size = info->rx_attr->size;
	ret = rxd_ep_create_buf_pools(rxd_ep, info);
	if (ret)
		goto err3;

	rxd_ep->dg_ep = *ep;
	rxd_ep->ep.fid.ops = &rxd_ep_fi_ops;
	rxd_ep->ep.cm = &rxd_ep_cm;
	rxd_ep->ep.ops = &rxd_ops_ep;
	rxd_ep->ep.msg = &rxd_ops_msg;
	rxd_ep->ep.tagged = &rxd_ops_tagged;

	dlist_init(&rxd_ep->tx_entry_list);
	dlist_init(&rxd_ep->rx_entry_list);
	dlist_init(&rxd_ep->wait_rx_list);
	dlist_init(&rxd_ep->unexp_msg_list);
	dlist_init(&rxd_ep->unexp_tag_list);
	slist_init(&rxd_ep->rx_pkt_list);
	fastlock_init(&rxd_ep->lock);

	dlist_init(&rxd_ep->dom_entry);
	atomic_inc(&rxd_ep->domain->util_domain.ref);

	*ep = &rxd_ep->ep;
	fi_freeinfo(dg_info);
	return 0;

err3:
	fi_close(&(*ep)->fid);
err2:
	free(rxd_ep);
err1:
	fi_freeinfo(dg_info);
	return ret;
}

int rxd_ep_retry_pkt(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
		   struct rxd_pkt_meta *pkt)
{
	int ret;
	struct ofi_ctrl_hdr *ctrl;

	ctrl = (struct ofi_ctrl_hdr *)pkt->pkt_data;
	if (pkt->retries > RXD_MAX_PKT_RETRY) {
		/* todo: report error */
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Pkt delivery failed\n", ctrl->seg_no);
		return -FI_EIO;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "retry packet : %2d, size: %d, tx_id :%p\n",
		ctrl->seg_no, ctrl->type == ofi_ctrl_start_data ?
		ctrl->seg_size + RXD_START_DATA_PKT_SZ :
		ctrl->seg_size + RXD_DATA_PKT_SZ,
		ctrl->msg_id);

	ret = fi_send(ep->dg_ep, ctrl,
		      ctrl->type == ofi_ctrl_start_data ?
		      ctrl->seg_size + RXD_START_DATA_PKT_SZ :
		      ctrl->seg_size + RXD_DATA_PKT_SZ,
		      rxd_mr_desc(pkt->mr, ep),
		      tx_entry->peer, &pkt->context);

	if (ret != -FI_EAGAIN)
		pkt->retries++;

	if (ret && ret != -FI_EAGAIN) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Pkt sent failed seg: %d, ret: %d\n",
			ctrl->seg_no, ret);
	}

	return ret;
}

void rxd_ep_progress(struct rxd_ep *ep)
{
	struct dlist_entry *tx_item, *pkt_item;
	struct rxd_tx_entry *tx_entry;
	struct rxd_pkt_meta *pkt;
	uint64_t curr_stamp;

	rxd_ep_lock_if_required(ep);
	curr_stamp = fi_gettime_us();
	dlist_foreach(&ep->tx_entry_list, tx_item) {

		tx_entry = container_of(tx_item, struct rxd_tx_entry, entry);

		if (tx_entry->win_sz)
			rxd_tx_entry_progress(ep, tx_entry, NULL);

		else if (tx_entry->is_waiting &&
			 (curr_stamp - tx_entry->retry_stamp > RXD_WAIT_TIMEOUT) &&
			 dlist_empty(&tx_entry->pkt_list)) {
			tx_entry->win_sz = 1;
			tx_entry->is_waiting = 0;

			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Progressing waiting entry [%p]\n",
				tx_entry->msg_id);

			rxd_tx_entry_progress(ep, tx_entry, NULL);
			tx_entry->retry_stamp = fi_gettime_us();
		}

		dlist_foreach(&tx_entry->pkt_list, pkt_item) {
			pkt = container_of(pkt_item, struct rxd_pkt_meta, entry);
			if (curr_stamp > pkt->us_stamp &&
			    curr_stamp - pkt->us_stamp >
			    (1 << (pkt->retries + 1)) * RXD_RETRY_TIMEOUT) {
				pkt->us_stamp = curr_stamp;
				rxd_ep_retry_pkt(ep, tx_entry, pkt);
			}
		}
	}
	rxd_ep_unlock_if_required(ep);
}
