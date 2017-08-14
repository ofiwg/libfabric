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

#include "fi.h"
#include <fi_iov.h>

#include "rxm.h"

static struct rxm_conn *rxm_key2conn(struct rxm_ep *rxm_ep, uint64_t key)
{
	struct util_cmap_handle *handle;
	handle = ofi_cmap_key2handle(rxm_ep->util_ep.cmap, key);
	if (!handle) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Can't find handle!\n");
		return NULL;
	}
	if (handle->key != key) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
				"handle->key not matching with given key!\n");
		return NULL;
	}

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

#if ENABLE_DEBUG
static void rxm_cq_log_comp(uint64_t flags)
{
	flags &= (FI_SEND | FI_WRITE | FI_READ | FI_REMOTE_READ |
		  FI_REMOTE_WRITE);

	switch(flags) {
	case FI_SEND:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Reporting send completion\n");
		break;
	case FI_WRITE:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Reporting write completion\n");
		break;
	case FI_READ:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Reporting read completion\n");
		break;
	case FI_REMOTE_READ:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Reporting remote read completion\n");
		break;
	case FI_REMOTE_WRITE:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Reporting remote write completion\n");
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown completion\n");
	}
}
#else
static void rxm_cq_log_comp(uint64_t flags)
{
	// NOP
}
#endif

int rxm_cq_comp(struct util_cq *util_cq, void *context, uint64_t flags, size_t len,
		void *buf, uint64_t data, uint64_t tag)
{
	struct fi_cq_tagged_entry *comp;
	int ret = 0;

	fastlock_acquire(&util_cq->cq_lock);
	if (ofi_cirque_isfull(util_cq->cirq)) {
		FI_DBG(&rxm_prov, FI_LOG_CQ, "util_cq cirq is full!\n");
		ret = -FI_EAGAIN;
		goto out;
	}

	comp = ofi_cirque_tail(util_cq->cirq);
	comp->op_context = context;
	comp->flags = flags;
	comp->len = len;
	comp->buf = buf;
	comp->data = data;
	comp->tag = tag;
	ofi_cirque_commit(util_cq->cirq);
out:
	fastlock_release(&util_cq->cq_lock);
	return ret;
}

int rxm_finish_recv(struct rxm_rx_buf *rx_buf)
{
	int ret;

	if (rx_buf->recv_entry->flags & FI_COMPLETION) {
		FI_DBG(&rxm_prov, FI_LOG_CQ, "writing recv completion\n");
		ret = rxm_cq_comp(rx_buf->ep->util_ep.rx_cq,
				  rx_buf->recv_entry->context,
				  rx_buf->comp_flags | FI_RECV,
				  rx_buf->pkt.hdr.size, NULL,
				  rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to write recv completion\n");
			return ret;
		}
	}

	rxm_recv_entry_release(rx_buf->recv_queue, rx_buf->recv_entry);
	return rxm_ep_repost_buf(rx_buf);
}

static int rxm_finish_send_nobuf(struct rxm_tx_entry *tx_entry)
{
	int ret;

	if (tx_entry->flags & FI_COMPLETION) {
		ret = rxm_cq_comp(tx_entry->ep->util_ep.tx_cq, tx_entry->context,
				tx_entry->comp_flags, 0, NULL, 0, 0);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to report completion\n");
			return ret;
		}
		rxm_cq_log_comp(tx_entry->comp_flags);
	}
	rxm_tx_entry_release(&tx_entry->ep->send_queue, tx_entry);
	return 0;
}

int rxm_finish_send(struct rxm_tx_entry *tx_entry)
{
	rxm_buf_release(&tx_entry->ep->tx_pool, (struct rxm_buf *)tx_entry->tx_buf);
	return rxm_finish_send_nobuf(tx_entry);
}

/* Get a match_iov derived from iov whose size matches given length */
static int rxm_match_iov(const struct iovec *iov, void **desc,
			 uint8_t count, uint64_t offset, size_t match_len,
			 struct rxm_iov *match_iov)
{
	int i;

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
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Given iov size < match_len!\n");
		return -FI_ETOOSMALL;
	}

	match_iov->count = i + 1;
	return 0;
}

static int rxm_match_rma_iov(struct rxm_recv_entry *recv_entry,
			     struct rxm_rma_iov *rma_iov,
			     struct rxm_iov *match_iov)
{
	uint64_t offset = 0;
	size_t i, j;
	uint8_t count;
	int ret;

	assert(rma_iov->count <= RXM_IOV_LIMIT);

	for (i = 0, j = 0; i < rma_iov->count; ) {
		ret = rxm_match_iov(&recv_entry->iov[j], &recv_entry->desc[j],
				    recv_entry->count - j, offset,
				    rma_iov->iov[i].len, &match_iov[i]);
		if (ret)
			return ret;

		count = match_iov[i].count;
		offset = match_iov[i].iov[count - 1].iov_len;

		i++;
		j += count - 1;

		if (j >= recv_entry->count)
			break;
	}

	if (i < rma_iov->count) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "posted recv_entry size < "
			"rndv rma read size!\n");
		return -FI_ETOOSMALL;
	}

	return FI_SUCCESS;
}

static int rxm_lmt_rma_read(struct rxm_rx_buf *rx_buf)
{
	struct rxm_iov *match_iov = &rx_buf->match_iov[rx_buf->index];
	int ret;

	ret = fi_readv(rx_buf->conn->msg_ep, match_iov->iov, match_iov->desc,
		       match_iov->count, 0, rx_buf->rma_iov->iov[0].addr,
		       rx_buf->rma_iov->iov[0].key, rx_buf);
	if (ret)
		return ret;
	rx_buf->index++;
	return 0;
}

static int rxm_lmt_tx_finish(struct rxm_tx_entry *tx_entry)
{
	int ret;

	RXM_LOG_STATE_TX(FI_LOG_CQ, tx_entry, RXM_LMT_FINISH);
	tx_entry->state = RXM_LMT_FINISH;

	if (!OFI_CHECK_MR_LOCAL(tx_entry->ep->rxm_info->domain_attr->mr_mode))
		rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);

	tx_entry->comp_flags |= FI_SEND;
	ret = rxm_finish_send(tx_entry);
	if (ret)
		return ret;

	return rxm_ep_repost_buf(tx_entry->rx_buf);
}

static int rxm_lmt_handle_ack(struct rxm_rx_buf *rx_buf)
{
	struct rxm_tx_entry *tx_entry;
	int index;

	FI_DBG(&rxm_prov, FI_LOG_CQ, "Got ACK for msg_id: 0x%" PRIx64 "\n",
			rx_buf->pkt.ctrl_hdr.msg_id);

	fastlock_acquire(&rx_buf->ep->send_queue.lock);
	index = ofi_key2idx(&rx_buf->ep->send_queue.tx_key_idx,
			    rx_buf->pkt.ctrl_hdr.msg_id);
	tx_entry = &rx_buf->ep->send_queue.fs->buf[index];
	fastlock_release(&rx_buf->ep->send_queue.lock);

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

int rxm_cq_handle_data(struct rxm_rx_buf *rx_buf)
{
	struct rxm_iov mr_match_iov;
	size_t i, rma_total_len = 0;
	int ret;

	if (rx_buf->pkt.ctrl_hdr.type == ofi_ctrl_large_data) {
		if (!rx_buf->conn) {
			rx_buf->conn = rxm_key2conn(rx_buf->ep, rx_buf->pkt.ctrl_hdr.conn_id);
			if (!rx_buf->conn)
				return -FI_EOTHER;
		}

		FI_DBG(&rxm_prov, FI_LOG_CQ,
		       "Got incoming recv with msg_id: 0x%" PRIx64 "\n",
		       rx_buf->pkt.ctrl_hdr.msg_id);

		rx_buf->rma_iov = (struct rxm_rma_iov *)rx_buf->pkt.data;
		rx_buf->index = 0;

		for (i = 0; i < rx_buf->rma_iov->count; i++)
			rma_total_len += rx_buf->rma_iov->iov->len;

		if (rma_total_len > ofi_total_iov_len(rx_buf->recv_entry->iov,
				      rx_buf->recv_entry->count)) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Posted receive buffer size is not enough!\n");
			return -FI_ETRUNC; // TODO copy data and write to CQ error
		}

		if (!OFI_CHECK_MR_LOCAL(rx_buf->ep->rxm_info->domain_attr->mr_mode)) {
			ret = rxm_match_iov(rx_buf->recv_entry->iov,
					    rx_buf->recv_entry->desc,
					    rx_buf->recv_entry->count, 0,
					    rma_total_len, &mr_match_iov);

			ret = rxm_ep_msg_mr_regv(rx_buf->ep, rx_buf->recv_entry->iov,
						 rx_buf->recv_entry->count, FI_READ,
						 rx_buf->mr);
			if (ret)
				return ret;

			ret = rxm_ep_msg_mr_regv(rx_buf->ep, mr_match_iov.iov,
						 mr_match_iov.count, FI_WRITE,
						 rx_buf->mr);
			if (ret)
				return ret;

			rx_buf->recv_entry->count = mr_match_iov.count;

			for (i = 0; i < rx_buf->recv_entry->count; i++)
				rx_buf->recv_entry->desc[i] = rx_buf->mr[i];
		}

		for (i = 0; i < rx_buf->recv_entry->count; i++)
			rx_buf->recv_entry->desc[i] = fi_mr_desc(rx_buf->recv_entry->desc[i]);

		ret = rxm_match_rma_iov(rx_buf->recv_entry, rx_buf->rma_iov,
				    rx_buf->match_iov);
		if (ret)
			return ret;

		RXM_LOG_STATE_RX(FI_LOG_CQ, rx_buf, RXM_LMT_READ);
		rx_buf->hdr.state = RXM_LMT_READ;
		return rxm_lmt_rma_read(rx_buf);
	} else {
		ofi_copy_to_iov(rx_buf->recv_entry->iov, rx_buf->recv_entry->count, 0,
				rx_buf->pkt.data, rx_buf->pkt.hdr.size);
		return rxm_finish_recv(rx_buf);
	}
}

int rxm_handle_recv_comp(struct rxm_rx_buf *rx_buf)
{
	struct rxm_recv_match_attr match_attr = {0};
	struct dlist_entry *entry;
	struct rxm_recv_queue *recv_queue;
	struct util_cq *util_cq;

	util_cq = rx_buf->ep->util_ep.rx_cq;

	if ((rx_buf->ep->rxm_info->caps & FI_SOURCE) ||
			(rx_buf->ep->rxm_info->caps & FI_DIRECTED_RECV)) {
		if (!rx_buf->conn) {
			rx_buf->conn = rxm_key2conn(rx_buf->ep, rx_buf->pkt.ctrl_hdr.conn_id);
			if (!rx_buf->conn)
				return -FI_EOTHER;
		}
	}

	/* fi_addr would be FI_ADDR_UNSPEC if there is no corresponding AV entry */
	match_attr.addr = rx_buf->conn->handle.fi_addr;

	if (rx_buf->ep->rxm_info->caps & FI_SOURCE)
		util_cq->src[ofi_cirque_windex(util_cq->cirq)] = rx_buf->conn->handle.fi_addr;

	switch(rx_buf->pkt.hdr.op) {
	case ofi_op_msg:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got MSG op\n");
		recv_queue = &rx_buf->ep->recv_queue;
		rx_buf->comp_flags = FI_MSG;
		break;
	case ofi_op_tagged:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got TAGGED op\n");
		rx_buf->comp_flags = FI_TAGGED;
		match_attr.tag = rx_buf->pkt.hdr.tag;
		recv_queue = &rx_buf->ep->trecv_queue;
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown op!\n");
		assert(0);
		return -FI_EINVAL;
	}

	rx_buf->recv_queue = recv_queue;

	fastlock_acquire(&recv_queue->lock);
	entry = dlist_remove_first_match(&recv_queue->recv_list,
					 recv_queue->match_recv, &match_attr);
	if (!entry) {
		FI_DBG(&rxm_prov, FI_LOG_CQ,
				"No matching recv found. Enqueueing msg to unexpected queue\n");
		rx_buf->unexp_msg.addr = match_attr.addr;
		rx_buf->unexp_msg.tag = match_attr.tag;
		dlist_insert_tail(&rx_buf->unexp_msg.entry, &recv_queue->unexp_msg_list);
		fastlock_release(&recv_queue->lock);
		return 0;
	}
	fastlock_release(&recv_queue->lock);

	rx_buf->recv_entry = container_of(entry, struct rxm_recv_entry, entry);
	return rxm_cq_handle_data(rx_buf);
}

static int rxm_lmt_send_ack(struct rxm_rx_buf *rx_buf)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;
	int ret;

	assert(rx_buf->conn);

	tx_buf = (struct rxm_tx_buf *)rxm_buf_get(&rx_buf->ep->tx_pool);
	if (!tx_buf) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "TX queue full!\n");
		return -FI_EAGAIN;
	}

	if (!(tx_entry = rxm_tx_entry_get(&rx_buf->ep->send_queue))) {
		ret = -FI_EAGAIN;
		goto err1;
	}

	RXM_LOG_STATE(FI_LOG_CQ, rx_buf->pkt, RXM_LMT_READ, RXM_LMT_ACK_SENT);
	rx_buf->hdr.state = RXM_LMT_ACK_SENT;

	tx_entry->state 	= rx_buf->hdr.state;
	tx_entry->ep 		= rx_buf->ep;
	tx_entry->context 	= rx_buf;
	tx_entry->tx_buf 	= tx_buf;

	rxm_pkt_init(&tx_buf->pkt);
	tx_buf->pkt.ctrl_hdr.type 	= ofi_ctrl_ack;
	tx_buf->pkt.ctrl_hdr.conn_id 	= rx_buf->conn->handle.remote_key;
	tx_buf->pkt.ctrl_hdr.msg_id 	= rx_buf->pkt.ctrl_hdr.msg_id;
	tx_buf->pkt.hdr.op 		= rx_buf->pkt.hdr.op;

	ret = fi_send(rx_buf->conn->msg_ep, &tx_buf->pkt, sizeof(tx_buf->pkt),
		      tx_buf->hdr.desc, 0, tx_entry);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to send ACK\n");
		rx_buf->hdr.state = RXM_NONE;
		goto err2;
	}
	return 0;
err2:
	rxm_tx_entry_release(&rx_buf->ep->send_queue, tx_entry);
err1:
	rxm_buf_release(&rx_buf->ep->tx_pool, (struct rxm_buf *)tx_buf);
	return ret;
}

static int rxm_handle_remote_write(struct rxm_ep *rxm_ep,
				   struct fi_cq_tagged_entry *comp)
{
	int ret;

	FI_DBG(&rxm_prov, FI_LOG_CQ, "writing remote write completion\n");
	ret = rxm_cq_comp(rxm_ep->util_ep.rx_cq, NULL,
			  comp->flags, 0, NULL, comp->data, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to write remote write completion\n");
		return ret;
	}
	if (comp->op_context)
		return rxm_ep_repost_buf((struct rxm_rx_buf *)comp->op_context);
	return 0;
}

static int rxm_cq_handle_comp(struct rxm_ep *rxm_ep,
			      struct fi_cq_tagged_entry *comp)
{
	enum rxm_proto_state *state = comp->op_context;
	struct rxm_rx_buf *rx_buf = comp->op_context;
	struct rxm_tx_entry *tx_entry = comp->op_context;

	/* Remote write events may not consume a posted recv so op context
	 * and hence state would be NULL */
	if (comp->flags & FI_REMOTE_WRITE)
		return rxm_handle_remote_write(rxm_ep, comp);

	switch (*state) {
	case RXM_TX_NOBUF:
		assert(comp->flags & (FI_SEND | FI_WRITE | FI_READ));
		return rxm_finish_send_nobuf(tx_entry);
	case RXM_TX:
		assert(comp->flags & (FI_SEND | FI_WRITE));
		return rxm_finish_send(tx_entry);
	case RXM_RX:
		assert(!(comp->flags & FI_REMOTE_READ));

		if (rx_buf->pkt.ctrl_hdr.type == ofi_ctrl_ack)
			return rxm_lmt_handle_ack(rx_buf);
		else
			return rxm_handle_recv_comp(comp->op_context);
	case RXM_LMT_TX:
		assert(comp->flags & FI_SEND);
		RXM_LOG_STATE_TX(FI_LOG_CQ, tx_entry, RXM_LMT_ACK_WAIT);
		*state = RXM_LMT_ACK_WAIT;
		return 0;
	case RXM_LMT_ACK_RECVD:
		assert(comp->flags & FI_SEND);
		return rxm_lmt_tx_finish(tx_entry);
	case RXM_LMT_READ:
		assert(comp->flags & FI_READ);
		if (rx_buf->index < rx_buf->rma_iov->count)
			return rxm_lmt_rma_read(rx_buf);
		else
			return rxm_lmt_send_ack(rx_buf);
	case RXM_LMT_ACK_SENT:
		assert(comp->flags & FI_SEND);
		rx_buf = tx_entry->context;
		rxm_tx_entry_release(&tx_entry->ep->send_queue, tx_entry);
		rxm_buf_release(&rx_buf->ep->tx_pool, (struct rxm_buf *)tx_entry->tx_buf);

		RXM_LOG_STATE_RX(FI_LOG_CQ, rx_buf, RXM_LMT_FINISH);
		rx_buf->hdr.state = RXM_LMT_FINISH;
		if (!OFI_CHECK_MR_LOCAL(rx_buf->ep->rxm_info->domain_attr->mr_mode))
			rxm_ep_msg_mr_closev(rx_buf->mr, RXM_IOV_LIMIT);
		return rxm_finish_recv(rx_buf);
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Invalid state!\n");
		assert(0);
		return -FI_EOPBADSTATE;
	}
}

static ssize_t rxm_cq_write_error(struct fid_cq *msg_cq,
				  struct fi_cq_tagged_entry *comp,
				  int err)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_rx_buf *rx_buf;
	struct fi_cq_err_entry err_entry;
	struct util_cq *util_cq;
	void *op_context;
	ssize_t ret;

	op_context = comp->op_context;
	memset(&err_entry, 0, sizeof(err_entry));

	if (err == -FI_EAVAIL) {
		OFI_CQ_READERR(&rxm_prov, FI_LOG_CQ, msg_cq, ret, err_entry);
		if (ret < 0) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to fi_cq_readerr on msg cq\n");
			err_entry.prov_errno = ret;
			err = ret;
		} else {
			op_context = err_entry.op_context;
		}
	} else {
		err_entry.prov_errno = err;
	}
	switch (*(enum rxm_proto_state *)op_context) {
	case RXM_TX:
	case RXM_LMT_TX:
		tx_entry = (struct rxm_tx_entry *)op_context;
		util_cq = tx_entry->ep->util_ep.tx_cq;
		break;
	case RXM_LMT_ACK_SENT:
		tx_entry = (struct rxm_tx_entry *)op_context;
		util_cq = tx_entry->ep->util_ep.rx_cq;
		break;
	case RXM_RX:
	case RXM_LMT_READ:
		rx_buf = (struct rxm_rx_buf *)op_context;
		util_cq = rx_buf->ep->util_ep.rx_cq;
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Invalid state!\n");
		if (err == -FI_EAVAIL)
			FI_WARN(&rxm_prov, FI_LOG_CQ, "msg cq error info: %s\n",
				fi_cq_strerror(msg_cq, err_entry.prov_errno,
					       err_entry.err_data, NULL, 0));
		return -FI_EOPBADSTATE;
	}
	return ofi_cq_write_error(util_cq, &err_entry);
}

void rxm_cq_progress(struct rxm_ep *rxm_ep)
{
	struct fi_cq_tagged_entry comp;
	ssize_t ret, comp_read = 0;

	do {
		ret = fi_cq_read(rxm_ep->msg_cq, &comp, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (ret < 0)
			goto err;

		if (ret) {
			comp_read += ret;
			ret = rxm_cq_handle_comp(rxm_ep, &comp);
			if (ret)
				goto err;
		}
	} while (comp_read < rxm_ep->comp_per_progress);
	return;
err:
	if (rxm_cq_write_error(rxm_ep->msg_cq, &comp, ret))
		assert(0);
	return;
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
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
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
