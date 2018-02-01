/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

int rxd_progress_spin_count = 1000;
int rxd_reposted_bufs = 0;

static ssize_t rxd_ep_cancel(fid_t fid, void *context)
{
	struct rxd_ep *ep;
	struct dlist_entry *entry, *next;
	struct rxd_recv_entry *recv_entry;
	struct rxd_trecv_entry *trecv_entry;
	struct fi_cq_err_entry err_entry = {0};

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->lock);
	if (ep->util_ep.caps & FI_MSG) {
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
			rxd_cq_report_error(rxd_ep_rx_cq(ep), &err_entry);
			goto out;
		}
	}

	if (ep->util_ep.caps & FI_TAGGED) {
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
			rxd_cq_report_error(rxd_ep_rx_cq(ep), &err_entry);
			goto out;
		}
	}

out:
	fastlock_release(&ep->lock);
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
	ssize_t ret = 0;
	size_t i;
	struct rxd_ep *rxd_ep;
	struct rxd_recv_entry *recv_entry;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid.fid);

	fastlock_acquire(&rxd_ep->lock);
	if (freestack_isempty(rxd_ep->recv_fs)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	recv_entry = freestack_pop(rxd_ep->recv_fs);
	recv_entry->msg = *msg;
	recv_entry->flags = flags;
	recv_entry->msg.addr = (rxd_ep->util_ep.caps & FI_DIRECTED_RECV) ?
		recv_entry->msg.addr : FI_ADDR_UNSPEC;
	for (i = 0; i < msg->iov_count; i++) {
		recv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		recv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post recv: %zu\n",
		       msg->msg_iov[i].iov_len);
	}

	dlist_init(&recv_entry->entry);
	dlist_insert_tail(&recv_entry->entry, &rxd_ep->recv_list);

	if (!dlist_empty(&rxd_ep->unexp_msg_list)) {
		rxd_ep_check_unexp_msg_list(rxd_ep, recv_entry);
	}
out:
	fastlock_release(&rxd_ep->lock);
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
	ret = fi_recv(buf->ep->dg_ep, buf->buf, rxd_ep_domain(buf->ep)->max_mtu_sz,
		      rxd_mr_desc(buf->mr, buf->ep),
		      FI_ADDR_UNSPEC, &buf->context);
	if (ret)
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "failed to repost\n");
	else
		rxd_reposted_bufs++;
	return ret;
}

/*
 * See ofi_proto.h for how conn_data is being used.
 */
static uint64_t rxd_ep_conn_data(struct rxd_ep *ep)
{
	char name[RXD_MAX_DGRAM_ADDR];
	size_t addrlen;
	int ret;

	if (ep->conn_data_set)
		return ep->conn_data;

	addrlen = sizeof name;
	ret = fi_getname(&ep->dg_ep->fid, name, &addrlen);
	if (ret)
		return 0;

	ret = rxd_av_dg_reverse_lookup(rxd_ep_av(ep), 0, name, &ep->conn_data);
	if (!ret)
		ep->conn_data_set = 1;

	return ep->conn_data;
}

static int rxd_ep_enable(struct rxd_ep *ep)
{
	size_t i;
	ssize_t ret;
	void *mr = NULL;
	struct rxd_rx_buf *rx_buf;

	ret = fi_enable(ep->dg_ep);
	if (ret)
		return ret;

	fastlock_acquire(&ep->lock);
	ep->credits = ep->rx_size;
	for (i = 0; i < ep->rx_size; i++) {
		rx_buf = ep->do_local_mr ?
			util_buf_get_ex(ep->rx_pkt_pool, &mr) :
			util_buf_get(ep->rx_pkt_pool);

		if (!rx_buf) {
			ret = -FI_ENOMEM;
			goto out;
		}

		rx_buf->mr = (struct fid_mr *) mr;
		rx_buf->ep = ep;
		ret = rxd_ep_repost_buff(rx_buf);
		if (ret)
			goto out;
		slist_insert_tail(&rx_buf->entry, &ep->rx_pkt_list);
	}
out:
	fastlock_release(&ep->lock);
	return ret;
}

struct rxd_peer *rxd_ep_getpeer_info(struct rxd_ep *ep, fi_addr_t addr)
{
	return &ep->peer_info[addr];
}

/*
 * Exponential back-off starting at 1ms, max 4s.
 */
void rxd_set_timeout(struct rxd_tx_entry *tx_entry)
{
	tx_entry->retry_time = fi_gettime_ms() +
				MIN(1 << tx_entry->retry_cnt, 4000);
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

static uint32_t rxd_map_fi_flags(uint64_t fi_flags)
{
	uint32_t flags = 0;

	if (fi_flags & FI_REMOTE_CQ_DATA)
		flags = OFI_REMOTE_CQ_DATA;
	if (fi_flags & FI_TRANSMIT_COMPLETE)
		flags |= OFI_TRANSMIT_COMPLETE;
	if (fi_flags & FI_DELIVERY_COMPLETE)
		flags |= OFI_DELIVERY_COMPLETE;
	return flags;
}

static struct rxd_pkt_meta *rxd_tx_pkt_alloc(struct rxd_ep *ep)
{
	struct rxd_pkt_meta *pkt_meta;
	void *mr = NULL;

	pkt_meta = ep->do_local_mr ?
		util_buf_alloc_ex(ep->tx_pkt_pool, &mr) :
		util_buf_alloc(ep->tx_pkt_pool);

	if (!pkt_meta) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "No free tx pkt\n");
		return NULL;
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Acquired tx pkt: %p\n", pkt_meta);
	pkt_meta->ep = ep;
	pkt_meta->mr = (struct fid_mr *) mr;
	pkt_meta->flags = 0;
	return pkt_meta;
}

static uint64_t rxd_ep_start_seg_size(struct rxd_ep *ep, uint64_t msg_size)
{
	return MIN(rxd_ep_domain(ep)->max_mtu_sz -
		   sizeof(struct rxd_pkt_data_start), msg_size);
}

struct rxd_tx_entry *rxd_tx_entry_alloc(struct rxd_ep *ep,
	struct rxd_peer *peer, fi_addr_t addr, uint64_t flags, uint8_t op)
{
	struct rxd_tx_entry *tx_entry;

	if (freestack_isempty(ep->tx_entry_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no-more tx entries\n");
		return NULL;
	}

	if (peer->active_tx_cnt == RXD_MAX_PEER_TX)
		return NULL;

	peer->active_tx_cnt++;

	tx_entry = freestack_pop(ep->tx_entry_fs);
	tx_entry->peer = addr;
	tx_entry->flags = flags;
	tx_entry->bytes_sent = 0;
	tx_entry->seg_no = 0;
	tx_entry->window = 1;
	tx_entry->retry_cnt = 0;
	tx_entry->op_type = op;
	dlist_init(&tx_entry->pkt_list);
	return tx_entry;
}

void rxd_tx_entry_free(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	struct rxd_peer *peer;

	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);
	peer->active_tx_cnt--;
	/* reset ID to invalid state to avoid ID collision */
	tx_entry->msg_id = UINT64_MAX;
	dlist_remove(&tx_entry->entry);
	freestack_push(ep->tx_entry_fs, tx_entry);
}

static size_t rxd_copy_rma_iov(struct ofi_rma_iov *dst,
				const struct fi_rma_iov *src, size_t count)
{
	int i;

	for (i = 0; i < count; i++) {
		dst->addr = src->addr;
		dst->len = src->len;
		dst->key = src->key;
	}
	return sizeof(*dst) * count;
}

static uint64_t rxd_ep_copy_data(struct rxd_tx_entry *tx_entry,
				 char *buf, uint64_t size)
{
	const struct iovec *iov;
	size_t iov_count;

	switch(tx_entry->op_hdr.op) {
	case ofi_op_msg:
		iov = tx_entry->msg.msg_iov;
		iov_count = tx_entry->msg.msg.iov_count;
		break;
	case ofi_op_tagged:
		iov = tx_entry->tmsg.msg_iov;
		iov_count = tx_entry->tmsg.tmsg.iov_count;
		break;
	case ofi_op_write:
		iov = tx_entry->write.src_iov;
		iov_count = tx_entry->write.msg.iov_count;
		break;
	case ofi_op_read_rsp:
		iov = tx_entry->read_rsp.src_iov;
		iov_count = tx_entry->read_rsp.iov_count;
		break;
	default:
		return 0;
	}

	return ofi_copy_from_iov(buf, size, iov, iov_count, tx_entry->bytes_sent);
}

static void rxd_ep_init_data_pkt(struct rxd_ep *ep, struct rxd_peer *peer,
				 struct rxd_tx_entry *tx_entry,
				 struct rxd_pkt_data *pkt)
{
	uint16_t seg_size;

	seg_size = rxd_ep_domain(ep)->max_mtu_sz - sizeof(struct rxd_pkt_data);
	seg_size = MIN(seg_size, tx_entry->op_hdr.size - tx_entry->bytes_sent);

	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_data, seg_size, tx_entry->seg_no,
			   tx_entry->msg_id, tx_entry->rx_key, peer->conn_data);
	tx_entry->bytes_sent += rxd_ep_copy_data(tx_entry, pkt->data, seg_size);
	tx_entry->seg_no++;
}

static ssize_t rxd_ep_post_data_msg(struct rxd_ep *ep,
				    struct rxd_tx_entry *tx_entry)
{
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;
	struct rxd_peer *peer;
	int ret;

	peer = rxd_ep_getpeer_info(ep, tx_entry->peer);

	pkt_meta = rxd_tx_pkt_alloc(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt_meta->tx_entry = tx_entry;
	pkt = (struct rxd_pkt_data *) pkt_meta->pkt_data;
	rxd_ep_init_data_pkt(ep, peer, tx_entry, pkt);

	if (tx_entry->op_hdr.size == pkt->ctrl.seg_size)
		pkt_meta->flags |= RXD_PKT_LAST;

	ret = fi_send(ep->dg_ep, pkt, sizeof(*pkt) + pkt->ctrl.seg_size,
		      rxd_mr_desc(pkt_meta->mr, ep), tx_entry->peer,
		      &pkt_meta->context);
	if (ret) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "send %d failed\n",
		       pkt->ctrl.seg_no);
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "msg data %" PRIx64 ", seg %d\n",
	       pkt->ctrl.msg_id, pkt->ctrl.seg_no);
	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);

	return ret;
}

void rxd_ep_free_acked_pkts(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			    uint32_t last_acked)
{
	struct rxd_pkt_meta *pkt;
	struct ofi_ctrl_hdr *ctrl;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "freeing all [%" PRIx64 "] pkts < %d\n",
	       tx_entry->msg_id, last_acked);
	while (!dlist_empty(&tx_entry->pkt_list)) {

		pkt = container_of(tx_entry->pkt_list.next,
				   struct rxd_pkt_meta, entry);
		ctrl = (struct ofi_ctrl_hdr *) pkt->pkt_data;
		if (ctrl->seg_no >= last_acked)
			break;

		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "freeing [%" PRIx64 "] pkt:%d\n",
		       tx_entry->msg_id, ctrl->seg_no);
		dlist_remove(&pkt->entry);
		if (pkt->flags & RXD_LOCAL_COMP)
			rxd_tx_pkt_free(pkt);
		else
			pkt->flags |= RXD_REMOTE_ACK;
	};
}

static int rxd_ep_retry_pkt(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry,
			    struct rxd_pkt_meta *pkt)
{
	int ret;
	struct ofi_ctrl_hdr *ctrl;

	ctrl = (struct ofi_ctrl_hdr *)pkt->pkt_data;
//	if (pkt->retries > RXD_MAX_PKT_RETRY) {
//		/* todo: report error */
//		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Pkt delivery failed\n", ctrl->seg_no);
//		return -FI_EIO;
//	}
//
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "retry packet : %2d, size: %zd, tx_id :%" PRIx64 "\n",
	       ctrl->seg_no, ctrl->type == ofi_ctrl_start_data ?
	       ctrl->seg_size + sizeof(struct rxd_pkt_data_start) :
	       ctrl->seg_size + sizeof(struct rxd_pkt_data),
	       ctrl->msg_id);

	ret = fi_send(ep->dg_ep, ctrl,
		      ctrl->type == ofi_ctrl_start_data ?
		      ctrl->seg_size + sizeof(struct rxd_pkt_data_start) :
		      ctrl->seg_size + sizeof(struct rxd_pkt_data),
		      rxd_mr_desc(pkt->mr, ep),
		      tx_entry->peer, &pkt->context);

//	if (ret != -FI_EAGAIN)
//		pkt->retries++;

	if (ret && ret != -FI_EAGAIN) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "Pkt sent failed seg: %d, ret: %d\n",
			ctrl->seg_no, ret);
	}

	return ret;
}

//void rxd_resend_pkt(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry )
//{
//	struct dlist_entry *pkt_item;
//	struct rxd_pkt_meta *pkt;
//	struct ofi_ctrl_hdr *ctrl;
//
//	dlist_foreach(&tx_entry->pkt_list, pkt_item) {
//		pkt = container_of(pkt_item, struct rxd_pkt_meta, entry);
//		ctrl = (struct ofi_ctrl_hdr *) pkt->pkt_data;
//
//		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "resending pkt %d, %p\n",
//			ctrl->seg_no, ctrl->msg_id);
//
//		rxd_ep_retry_pkt(ep, tx_entry, pkt);
//	}
//}

void rxd_tx_entry_progress(struct rxd_ep *ep, struct rxd_tx_entry *tx_entry)
{
	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "tx: %p [%" PRIx64 "]\n",
		tx_entry, tx_entry->msg_id);

	while ((tx_entry->seg_no < tx_entry->window) &&
	       (tx_entry->bytes_sent != tx_entry->op_hdr.size)) {
		if (rxd_ep_post_data_msg(ep, tx_entry))
			break;
	}
	rxd_set_timeout(tx_entry);
}

int rxd_ep_reply_ack(struct rxd_ep *ep, struct ofi_ctrl_hdr *in_ctrl,
		   uint8_t type, uint16_t seg_size, uint64_t rx_key,
		   uint64_t source, fi_addr_t dest)
{
	ssize_t ret;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;
	struct rxd_rx_entry *rx_entry;

	pkt_meta = rxd_tx_pkt_alloc(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	rx_entry = (rx_key != UINT64_MAX) ? &ep->rx_entry_fs->buf[rx_key] : NULL;

	pkt = (struct rxd_pkt_data *)pkt_meta->pkt_data;
	rxd_init_ctrl_hdr(&pkt->ctrl, type, seg_size,
			  rx_entry ? rx_entry->exp_seg_no : 0,
			  in_ctrl->msg_id, rx_key, source);

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sending ack [%" PRIx64 "] - segno: %d, window: %d\n",
	       pkt->ctrl.msg_id, pkt->ctrl.seg_no, pkt->ctrl.seg_size);

	pkt_meta->flags = RXD_NOT_ACKED;
	ret = fi_send(ep->dg_ep, pkt, sizeof(struct rxd_pkt_data),
		      rxd_mr_desc(pkt_meta->mr, ep),
		      dest, &pkt_meta->context);
	if (ret)
		util_buf_release(ep->tx_pkt_pool, pkt_meta);

	return ret;
}

#define RXD_TX_ENTRY_ID(ep, tx_entry) (tx_entry - &ep->tx_entry_fs->buf[0])

static void rxd_ep_init_start_pkt(struct rxd_ep *ep, struct rxd_peer *peer,
				  uint8_t op, struct rxd_tx_entry *tx_entry,
				  struct rxd_pkt_data_start *pkt, uint32_t flags)
{
	uint64_t msg_size, iov_size;
	uint16_t seg_size;

	switch (op) {
	case ofi_op_msg:
		msg_size = ofi_total_iov_len(tx_entry->msg.msg_iov,
					     tx_entry->msg.msg.iov_count);
		seg_size = rxd_ep_start_seg_size(ep, msg_size);
		rxd_init_op_hdr(&pkt->op, tx_entry->msg.msg.data, msg_size, 0,
				op, 0, flags);
		break;
	case ofi_op_tagged:
		msg_size = ofi_total_iov_len(tx_entry->tmsg.msg_iov,
					     tx_entry->tmsg.tmsg.iov_count);
		seg_size = rxd_ep_start_seg_size(ep, msg_size);
		rxd_init_op_hdr(&pkt->op, tx_entry->tmsg.tmsg.data, msg_size, 0,
				op, tx_entry->tmsg.tmsg.tag, flags);
		break;
	case ofi_op_write:
		msg_size = ofi_total_iov_len(tx_entry->write.msg.msg_iov,
					     tx_entry->write.msg.iov_count);
		iov_size = rxd_copy_rma_iov((struct ofi_rma_iov *) pkt->data,
					    tx_entry->write.dst_iov,
					    tx_entry->write.msg.rma_iov_count);
		seg_size = MIN(rxd_ep_domain(ep)->max_mtu_sz -
			       sizeof(struct rxd_pkt_data_start) - iov_size, msg_size);
		rxd_init_op_hdr(&pkt->op, tx_entry->write.msg.data, msg_size, 0,
				op, 0, flags);
		pkt->op.iov_count = tx_entry->write.msg.rma_iov_count;
		break;
	case ofi_op_read_req:
		msg_size = ofi_total_iov_len(tx_entry->read_req.msg.msg_iov,
					     tx_entry->read_req.msg.iov_count);
		iov_size = rxd_copy_rma_iov((struct ofi_rma_iov *) pkt->data,
					    tx_entry->read_req.src_iov,
					    tx_entry->read_req.msg.rma_iov_count);
		seg_size = 0;
		rxd_init_op_hdr(&pkt->op, tx_entry->read_req.msg.data, msg_size,
				0, op, 0, flags);
		pkt->op.iov_count = tx_entry->read_req.msg.rma_iov_count;
		break;
	case ofi_op_read_rsp:
		msg_size = ofi_total_iov_len(tx_entry->read_rsp.src_iov,
					     tx_entry->read_rsp.iov_count);
		seg_size = rxd_ep_start_seg_size(ep, msg_size);
		rxd_init_op_hdr(&pkt->op, 0, msg_size, 0, op, 0, flags);
		pkt->op.remote_idx = tx_entry->read_rsp.peer_msg_id;
		break;
	default:
		seg_size = 0;
		assert(0);
	}

	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_start_data, seg_size, 0,
			   tx_entry->msg_id, peer->conn_data,
			   peer->conn_data);
	/* copy op header here because it is used in ep_copy_data call */
	tx_entry->op_hdr = pkt->op;
	tx_entry->bytes_sent = rxd_ep_copy_data(tx_entry, pkt->data,
						seg_size);
	tx_entry->seg_no++;
	assert(tx_entry->bytes_sent == seg_size);
}

ssize_t rxd_ep_start_xfer(struct rxd_ep *ep, struct rxd_peer *peer,
			  uint8_t op, struct rxd_tx_entry *tx_entry)
{
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data_start *pkt;
	uint32_t flags;
	ssize_t ret;

	pkt_meta = rxd_tx_pkt_alloc(ep);
	if (!pkt_meta)
		return -FI_ENOMEM;

	pkt_meta->tx_entry = tx_entry;
	pkt = (struct rxd_pkt_data_start *) pkt_meta->pkt_data;
	flags = rxd_map_fi_flags(tx_entry->flags);
	tx_entry->msg_id = RXD_TX_ID(peer->nxt_msg_id,
				     RXD_TX_ENTRY_ID(ep, tx_entry));

	rxd_ep_init_start_pkt(ep, peer, op, tx_entry, pkt, flags);

	if (tx_entry->op_hdr.size == pkt->ctrl.seg_size)
		pkt_meta->flags |= RXD_PKT_LAST;

	ret = fi_send(ep->dg_ep, pkt, sizeof(*pkt) + pkt->ctrl.seg_size,
		      rxd_mr_desc(pkt_meta->mr, ep),
		      tx_entry->peer, &pkt_meta->context);
	if (ret) {
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "send %d failed\n",
		       pkt->ctrl.seg_no);
	}

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "start msg %" PRIx64 ", size: %" PRIu64 "\n",
	       pkt->ctrl.msg_id, tx_entry->op_hdr.size);
	rxd_set_timeout(tx_entry);
	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);
	dlist_insert_tail(&tx_entry->entry, &ep->tx_entry_list);
	peer->nxt_msg_id++;

	return 0;
}

ssize_t rxd_ep_connect(struct rxd_ep *ep, struct rxd_peer *peer, fi_addr_t addr)
{
	struct rxd_pkt_data *pkt;
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_tx_entry *tx_entry;
	size_t addrlen;
	ssize_t ret;

	if (peer->state != CMAP_IDLE)
		return -FI_EALREADY;

	tx_entry = rxd_tx_entry_alloc(ep, peer, addr, 0, RXD_TX_CONN);
	if (!tx_entry)
		return -FI_EAGAIN;

	pkt_meta = rxd_tx_pkt_alloc(ep);
	if (!pkt_meta) {
		rxd_tx_entry_free(ep, tx_entry);
		return -FI_ENOMEM;
	}

	pkt = (struct rxd_pkt_data *) pkt_meta->pkt_data;
	addrlen = RXD_MAX_DGRAM_ADDR;
	ret = fi_getname(&ep->dg_ep->fid, pkt->data, &addrlen);
	if (ret)
		goto err;

	tx_entry->msg_id = RXD_TX_ID(peer->nxt_msg_id,
				     RXD_TX_ENTRY_ID(ep, tx_entry));
	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_connreq, (uint16_t) addrlen, 0,
			   tx_entry->msg_id, rxd_ep_conn_data(ep), addr);

	ret = fi_send(ep->dg_ep, pkt, addrlen + sizeof(struct rxd_pkt_data),
		      rxd_mr_desc(pkt_meta->mr, ep),
		      addr, &pkt_meta->context);
	if (ret)
		goto err;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "sent conn %" PRIx64 "\n",
	       pkt->ctrl.msg_id);
	rxd_set_timeout(tx_entry);
	dlist_insert_tail(&pkt_meta->entry, &tx_entry->pkt_list);
	dlist_insert_tail(&tx_entry->entry, &ep->tx_entry_list);
	peer->nxt_msg_id++;
	peer->state = CMAP_CONNREQ_SENT;
	return 0;
err:
	rxd_tx_entry_free(ep, tx_entry);
	util_buf_release(ep->tx_pkt_pool, pkt_meta);
	return ret;
}

static ssize_t rxd_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	fi_addr_t peer_addr;
	ssize_t ret;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid.fid);

	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	fastlock_acquire(&rxd_ep->lock);
	if (peer->state != CMAP_CONNECTED) {
		ret = rxd_ep_connect(rxd_ep, peer, peer_addr);
		fastlock_release(&rxd_ep->lock);
		if (ret == -FI_EALREADY) {
			rxd_ep->util_ep.progress(&rxd_ep->util_ep);
			ret = -FI_EAGAIN;
		}
		return ret ? ret : -FI_EAGAIN;
	}

	tx_entry = rxd_tx_entry_alloc(rxd_ep, peer, peer_addr, flags,
				      RXD_TX_MSG);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry->msg.msg = *msg;
	memcpy(&tx_entry->msg.msg_iov[0], msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	ret = rxd_ep_start_xfer(rxd_ep, peer, ofi_op_msg, tx_entry);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->lock);
	return ret;
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

static ssize_t rxd_ep_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			   fi_addr_t dest_addr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_sendv(ep, &iov, desc, 1, dest_addr, context);
}

static ssize_t rxd_ep_inject(struct fid_ep *ep, const void *buf, size_t len,
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

static ssize_t rxd_ep_injectdata(struct fid_ep *ep, const void *buf, size_t len,
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
	struct rxd_pkt_meta *pkt_meta;
	struct rxd_pkt_data *pkt;
	struct rxd_peer *peer;
	ssize_t ret;

	rx_buf = rx_entry->unexp_buf;
	ctrl = (struct ofi_ctrl_hdr *) rx_buf->buf;
	peer = rxd_ep_getpeer_info(ep, ctrl->conn_id);

	dlist_remove(&rx_entry->unexp_entry);
	ep->num_unexp_msg--;

	pkt_meta = rxd_tx_pkt_alloc(ep);
	if (!pkt_meta)
		goto out;

	pkt = (struct rxd_pkt_data *) pkt_meta->pkt_data;
	rxd_init_ctrl_hdr(&pkt->ctrl, ofi_ctrl_discard, 0, 0,
			   ctrl->msg_id, ctrl->rx_key, peer->conn_data);

	pkt_meta->flags = RXD_NOT_ACKED;
	ret = fi_send(ep->dg_ep, pkt, sizeof(struct rxd_pkt_data),
		      rxd_mr_desc(pkt_meta->mr, ep),
		      peer->fiaddr, &pkt_meta->context);
	if (ret)
		util_buf_release(ep->tx_pkt_pool, pkt_meta);

out:
	rxd_rx_entry_free(ep, rx_entry);
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
		rxd_cq_report_error(rxd_ep_rx_cq(ep), &err_entry);
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

	rxd_ep_rx_cq(ep)->write_fn(rxd_ep_rx_cq(ep), &cq_entry);
	return 0;
}

static ssize_t rxd_trx_claim_recv(struct rxd_ep *ep,
				  const struct fi_msg_tagged *msg, uint64_t flags)
{
	size_t i;
	struct fi_context *context;
	struct rxd_rx_entry *rx_entry;
	struct rxd_trecv_entry *trecv_entry;
	struct rxd_peer *peer;
	struct rxd_rx_buf *rx_buf;
	struct ofi_ctrl_hdr *ctrl;
	struct rxd_pkt_data_start *pkt_start;

	if (freestack_isempty(ep->trecv_fs))
		return -FI_EAGAIN;

	trecv_entry = freestack_pop(ep->trecv_fs);
	trecv_entry->msg = *msg;
	trecv_entry->msg.addr = (ep->util_ep.caps & FI_DIRECTED_RECV) ?
		msg->addr : FI_ADDR_UNSPEC;
	trecv_entry->flags = flags;
	for (i = 0; i < msg->iov_count; i++) {
		trecv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		trecv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post claim trecv: %zu, tag: %" PRIx64 "\n",
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
	return 0;
}

static ssize_t rxd_ep_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	ssize_t ret = 0;
	size_t i;
	struct rxd_ep *rxd_ep;
	struct rxd_trecv_entry *trecv_entry;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&rxd_ep->lock);

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
	trecv_entry->msg.addr = (rxd_ep->util_ep.caps & FI_DIRECTED_RECV) ?
		msg->addr : FI_ADDR_UNSPEC;
	trecv_entry->flags = flags;
	for (i = 0; i < msg->iov_count; i++) {
		trecv_entry->iov[i].iov_base = msg->msg_iov[i].iov_base;
		trecv_entry->iov[i].iov_len = msg->msg_iov[i].iov_len;
		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "post trecv: %zu, tag: %" PRIx64 "\n",
			msg->msg_iov[i].iov_len, msg->tag);
	}
	dlist_init(&trecv_entry->entry);
	dlist_insert_tail(&trecv_entry->entry, &rxd_ep->trecv_list);

	if (!dlist_empty(&rxd_ep->unexp_tag_list)) {
		rxd_ep_check_unexp_tag_list(rxd_ep, trecv_entry);
	}
out:
	fastlock_release(&rxd_ep->lock);
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

static ssize_t rxd_ep_trecvv(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t src_addr,
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

static ssize_t rxd_ep_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	fi_addr_t peer_addr;
	ssize_t ret;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid.fid);

	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	fastlock_acquire(&rxd_ep->lock);
	if (peer->state != CMAP_CONNECTED) {
		ret = rxd_ep_connect(rxd_ep, peer, peer_addr);
		fastlock_release(&rxd_ep->lock);
		if (ret == -FI_EALREADY) {
			rxd_ep->util_ep.progress(&rxd_ep->util_ep);
			ret = -FI_EAGAIN;
		}
		return ret ? ret : -FI_EAGAIN;
	}

	tx_entry = rxd_tx_entry_alloc(rxd_ep, peer, peer_addr, flags,
				      RXD_TX_TAG);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry->tmsg.tmsg = *msg;
	memcpy(&tx_entry->tmsg.msg_iov[0], msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	ret = rxd_ep_start_xfer(rxd_ep, peer, ofi_op_tagged, tx_entry);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->lock);
	return ret;
}

static ssize_t rxd_ep_tsend(struct fid_ep *ep, const void *buf, size_t len,
			    void *desc, fi_addr_t dest_addr, uint64_t tag,
			    void *context)
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

static ssize_t rxd_ep_tsendv(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
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

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	ret = fi_close(&ep->dg_ep->fid);
	if (ret)
		return ret;

	ret = fi_close(&ep->dg_cq->fid);
	if (ret)
		return ret;

	while(!slist_empty(&ep->rx_pkt_list)) {
		entry = slist_remove_head(&ep->rx_pkt_list);
		buf = container_of(entry, struct rxd_rx_buf, entry);
		util_buf_release(ep->rx_pkt_pool, buf);
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

	fastlock_destroy(&ep->lock);
	rxd_ep_free_buf_pools(ep);
	free(ep->peer_info);
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

		ep->peer_info = calloc(av->util_av.count, sizeof(struct rxd_peer));
		ep->max_peers = av->util_av.count;
		if (!ep->peer_info)
			return -FI_ENOMEM;
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
	struct dlist_entry *tx_item, *pkt_item;
	struct rxd_tx_entry *tx_entry;
	struct fi_cq_msg_entry cq_entry;
	struct rxd_pkt_meta *pkt;
	struct rxd_ep *ep;
	uint64_t cur_time;
	ssize_t ret;
	int i;

	ep = container_of(util_ep, struct rxd_ep, util_ep);

	fastlock_acquire(&ep->lock);
	for(ret = 1, i = 0;
	    ret > 0 && (!rxd_progress_spin_count || i < rxd_progress_spin_count);
	    i++) {
		ret = fi_cq_read(ep->dg_cq, &cq_entry, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (cq_entry.flags & FI_SEND)
			rxd_handle_send_comp(&cq_entry);
		else if (cq_entry.flags & FI_RECV)
			rxd_handle_recv_comp(ep, &cq_entry);
		else
			assert (0);
	}

	cur_time = fi_gettime_us();
	dlist_foreach(&ep->tx_entry_list, tx_item) {
		tx_entry = container_of(tx_item, struct rxd_tx_entry, entry);

		if (tx_entry->seg_no < tx_entry->window) {
			rxd_tx_entry_progress(ep, tx_entry);
		} else if ((tx_entry->retry_time > cur_time) /* &&
			 dlist_empty(&tx_entry->pkt_list)*/ ) {

			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Progressing waiting entry [%" PRIx64 "]\n",
				tx_entry->msg_id);

			rxd_tx_entry_progress(ep, tx_entry);
//			rxd_set_timeout(tx_entry);
		}

		dlist_foreach(&tx_entry->pkt_list, pkt_item) {
			pkt = container_of(pkt_item, struct rxd_pkt_meta, entry);
//			/* TODO: This if check is repeated.  Create a function
//			 * to perform this check, and figure out what the check
//			 * is actually doing with the bit-shift, multiply operation.
//			 */
//			if (curr_stamp > pkt->us_stamp &&
//			    curr_stamp - pkt->us_stamp >
//			    (((uint64_t) 1) << ((uint64_t) pkt->retries + 1)) *
//			     RXD_RETRY_TIMEOUT) {
//				pkt->us_stamp = curr_stamp;
				rxd_ep_retry_pkt(ep, tx_entry, pkt);
//			}
		}
	}
	fastlock_release(&ep->lock);
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

int rxd_ep_create_buf_pools(struct rxd_ep *ep, struct fi_info *fi_info)
{
	int ret = util_buf_pool_create_ex(
		&ep->tx_pkt_pool,
		rxd_ep_domain(ep)->max_mtu_sz + sizeof(struct rxd_pkt_meta),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_TX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_ep_domain(ep));
	if (ret)
		return -FI_ENOMEM;

	ret = util_buf_pool_create_ex(
		&ep->rx_pkt_pool,
		rxd_ep_domain(ep)->max_mtu_sz + sizeof (struct rxd_rx_buf),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_RX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_ep_domain(ep));
	if (ret)
		goto err;

	ep->tx_entry_fs = rxd_tx_entry_fs_create(1ULL << RXD_MAX_TX_BITS);
	if (!ep->tx_entry_fs)
		goto err;

	ep->rx_entry_fs = rxd_rx_entry_fs_create(1ULL << RXD_MAX_RX_BITS);
	if (!ep->rx_entry_fs)
		goto err;

	if (ep->util_ep.caps & FI_MSG) {
		ep->recv_fs = rxd_recv_fs_create(ep->rx_size);
		dlist_init(&ep->recv_list);
		if (!ep->recv_fs)
			goto err;
	}

	if (ep->util_ep.caps & FI_TAGGED) {
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
	ret = rxd_ep_create_buf_pools(rxd_ep, info);
	if (ret)
		goto err4;

	rxd_ep->util_ep.ep_fid.fid.ops = &rxd_ep_fi_ops;
	rxd_ep->util_ep.ep_fid.cm = &rxd_ep_cm;
	rxd_ep->util_ep.ep_fid.ops = &rxd_ops_ep;
	rxd_ep->util_ep.ep_fid.msg = &rxd_ops_msg;
	rxd_ep->util_ep.ep_fid.tagged = &rxd_ops_tagged;
	rxd_ep->util_ep.ep_fid.rma = &rxd_ops_rma;

	dlist_init(&rxd_ep->tx_entry_list);
	dlist_init(&rxd_ep->rx_entry_list);
	dlist_init(&rxd_ep->wait_rx_list);
	dlist_init(&rxd_ep->unexp_msg_list);
	dlist_init(&rxd_ep->unexp_tag_list);
	slist_init(&rxd_ep->rx_pkt_list);
	fastlock_init(&rxd_ep->lock);

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
