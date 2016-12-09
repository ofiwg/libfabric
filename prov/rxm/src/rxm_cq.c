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

static int rxm_match_recv_entry(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *) arg;
	struct rxm_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxm_recv_entry, entry);
	return rxm_match_addr(recv_entry->addr, attr->addr);
}

static int rxm_match_recv_entry_tagged(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxm_recv_entry, entry);
	return rxm_match_addr(recv_entry->addr, attr->addr) &&
		rxm_match_tag(recv_entry->tag, recv_entry->ignore, attr->tag);
}

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

int rxm_cq_report_error(struct util_cq *util_cq, struct fi_cq_err_entry *err_entry)
{
	struct util_cq_err_entry *entry;
	struct fi_cq_tagged_entry *comp;

	entry = calloc(1, sizeof(*entry));
	if (!entry) {
		FI_WARN(&rxm_prov, FI_LOG_CQ,
				"Unable to allocate util_cq_err_entry\n");
		return -FI_ENOMEM;
	}

	entry->err_entry = *err_entry;
	fastlock_acquire(&util_cq->cq_lock);
	slist_insert_tail(&entry->list_entry, &util_cq->err_list);

	comp = ofi_cirque_tail(util_cq->cirq);
	comp->flags = UTIL_FLAG_ERROR;
	ofi_cirque_commit(util_cq->cirq);
	fastlock_release(&util_cq->cq_lock);

	return 0;
}

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
				FI_RECV, rx_buf->pkt.hdr.size, NULL,
				rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to write recv completion\n");
			return ret;
		}
	}

	freestack_push(rx_buf->recv_fs, rx_buf->recv_entry);
	return rxm_ep_repost_buf(rx_buf);
}

int rxm_finish_send(struct rxm_tx_entry *tx_entry)
{
	int ret;

	if (tx_entry->flags & FI_COMPLETION) {
		FI_DBG(&rxm_prov, FI_LOG_CQ, "writing send completion\n");
		ret = rxm_cq_comp(tx_entry->ep->util_ep.tx_cq, tx_entry->context,
				FI_SEND, 0, NULL, 0, 0);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to write send completion\n");
			return ret;
		}
	}
	util_buf_release(tx_entry->ep->tx_pool, tx_entry->pkt);
	freestack_push(tx_entry->ep->txe_fs, tx_entry);
	return 0;
}

/* Get an iov whose size matches given length */
static int rxm_match_iov(struct rxm_match_iov *match_iov, size_t len,
		struct rxm_iovx_entry *iovx)
{
	int i, j;

	for (i = match_iov->index, j = 0; i < match_iov->count; i++, j++) {
		iovx->iov[j].iov_base = (char *)match_iov->iov[i].iov_base + match_iov->offset;
		iovx->iov[j].iov_len = MIN(match_iov->iov[i].iov_len - match_iov->offset, len);
		iovx->desc[j] = match_iov->desc[i];

		len -= iovx->iov[j].iov_len;
		if (!len)
			break;
		match_iov->offset = 0;
	}

	if (len) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "iovx size < len\n");
		return -FI_EINVAL;
	}

	iovx->count = j + 1;
	match_iov->index = i;
	match_iov->offset += iovx->iov[j].iov_len;
	return 0;
}

static int rxm_lmt_rma_read(struct rxm_rx_buf *rx_buf)
{
	struct rxm_iovx_entry iovx;
	int i, ret;

	memset(&iovx, 0, sizeof(iovx));
	iovx.count = RXM_IOV_LIMIT;

	ret = rxm_match_iov(&rx_buf->match_iov, rx_buf->rma_iov->iov[rx_buf->index].len, &iovx);
	if (ret)
		return ret;

	for (i = 0; i < iovx.count; i++)
		iovx.desc[i] = fi_mr_desc(iovx.desc[i]);

	ret = fi_readv(rx_buf->conn->msg_ep, iovx.iov, iovx.desc, iovx.count, 0,
			rx_buf->rma_iov->iov[0].addr, rx_buf->rma_iov->iov[0].key, rx_buf);
	// TODO do any cleanup?
	if (ret)
		return ret;
	rx_buf->index++;
	return 0;
}

int rxm_cq_handle_ack(struct rxm_rx_buf *rx_buf)
{
	struct rxm_tx_entry *tx_entry;
	int ret, index;

	FI_DBG(&rxm_prov, FI_LOG_CQ, "Got ACK for msg_id: 0x" PRIx64 "\n",
			rx_buf->pkt.ctrl_hdr.msg_id);

	index = ofi_key2idx(&rx_buf->ep->tx_key_idx, rx_buf->pkt.ctrl_hdr.msg_id);
	tx_entry = &rx_buf->ep->txe_fs->buf[index];

	assert(tx_entry->msg_id == rx_buf->pkt.ctrl_hdr.msg_id);
	assert(tx_entry->state == RXM_LMT_ACK);

	FI_DBG(&rxm_prov, FI_LOG_CQ, "tx_entry->state -> RXM_LMT_FINISH\n");
	tx_entry->state = RXM_LMT_FINISH;

	ret = rxm_finish_send(tx_entry);
	if (ret)
		return ret;

	return rxm_ep_repost_buf(rx_buf);
}

int rxm_cq_handle_data(struct rxm_rx_buf *rx_buf)
{
	if (rx_buf->pkt.ctrl_hdr.type == ofi_ctrl_large_data) {
		if (!rx_buf->conn) {
			rx_buf->conn = rxm_key2conn(rx_buf->ep, rx_buf->pkt.ctrl_hdr.conn_id);
			if (!rx_buf->conn)
				return -FI_EOTHER;
		}

		FI_DBG(&rxm_prov, FI_LOG_CQ, "rx_buf->state -> RXM_LMT_START\n");
		rx_buf->state = RXM_LMT_START;

		memset(&rx_buf->match_iov, 0, sizeof(rx_buf->match_iov));
		rx_buf->match_iov.iov = rx_buf->recv_entry->iov;
		rx_buf->match_iov.desc = rx_buf->recv_entry->desc;
		rx_buf->match_iov.count = rx_buf->recv_entry->count;

		rx_buf->rma_iov = (struct rxm_rma_iov *)rx_buf->pkt.data;
		rx_buf->index = 0;

		return rxm_lmt_rma_read(rx_buf);
	} else {
		ofi_copy_iov_buf(rx_buf->recv_entry->iov, rx_buf->recv_entry->count, rx_buf->pkt.data,
			rx_buf->pkt.hdr.size, 0, OFI_COPY_BUF_TO_IOV);
		return rxm_finish_recv(rx_buf);
	}
}

int rxm_handle_recv_comp(struct rxm_rx_buf *rx_buf)
{
	struct rxm_recv_match_attr match_attr = {0};
	struct dlist_entry *entry;
	struct rxm_recv_queue *recv_queue;
	struct util_cq *util_cq;
	dlist_func_t *match;

	if (rx_buf->pkt.ctrl_hdr.type == ofi_ctrl_ack)
		return rxm_cq_handle_ack(rx_buf);

	util_cq = rx_buf->ep->util_ep.rx_cq;

	if ((rx_buf->ep->rxm_info->caps & FI_SOURCE) ||
			(rx_buf->ep->rxm_info->caps & FI_DIRECTED_RECV)) {
		if (!rx_buf->conn) {
			rx_buf->conn = rxm_key2conn(rx_buf->ep, rx_buf->pkt.ctrl_hdr.conn_id);
			if (!rx_buf->conn)
				return -FI_EOTHER;
		}
	}

	if (rx_buf->ep->rxm_info->caps & FI_DIRECTED_RECV)
		match_attr.addr = rx_buf->conn->handle.fi_addr;
	else
		match_attr.addr = FI_ADDR_UNSPEC;

	if (rx_buf->ep->rxm_info->caps & FI_SOURCE)
		util_cq->src[ofi_cirque_windex(util_cq->cirq)] = rx_buf->conn->handle.fi_addr;

	switch(rx_buf->pkt.hdr.op) {
	case ofi_op_msg:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got MSG op\n");
		recv_queue = &rx_buf->ep->recv_queue;
		match = rxm_match_recv_entry;
		break;
	case ofi_op_tagged:
		FI_DBG(&rxm_prov, FI_LOG_CQ, "Got TAGGED op\n");
		match_attr.tag = rx_buf->pkt.hdr.tag;
		recv_queue = &rx_buf->ep->trecv_queue;
		match = rxm_match_recv_entry_tagged;
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown op!\n");
		assert(0);
		return -FI_EINVAL;
	}

	rx_buf->recv_fs = recv_queue->recv_fs;

	entry = dlist_remove_first_match(&recv_queue->recv_list, match, &match_attr);
	if (!entry) {
		FI_DBG(&rxm_prov, FI_LOG_CQ,
				"No matching recv found. Enqueueing msg to unexpected queue\n");
		rx_buf->unexp_msg.addr = match_attr.addr;
		rx_buf->unexp_msg.tag = match_attr.tag;
		dlist_insert_tail(&rx_buf->unexp_msg.entry, &recv_queue->unexp_msg_list);
		return 0;
	}

	rx_buf->recv_entry = container_of(entry, struct rxm_recv_entry, entry);
	return rxm_cq_handle_data(rx_buf);
}

int rxm_handle_send_comp(void *op_context)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_rx_buf *rx_buf;
	int ret = 0;

	switch (*(enum rxm_ctx_type *)op_context) {
	case RXM_TX_ENTRY:
		tx_entry = (struct rxm_tx_entry *)op_context;
		if (tx_entry->pkt->ctrl_hdr.type == ofi_ctrl_large_data) {
			assert(tx_entry->state == RXM_LMT_START);
			FI_DBG(&rxm_prov, FI_LOG_CQ, "tx_entry->state -> RXM_LMT_ACK\n");
			tx_entry->state = RXM_LMT_ACK;
			return 0;
		}
		ret = rxm_finish_send(tx_entry);
		break;
	case RXM_RX_BUF:
		rx_buf = (struct rxm_rx_buf *)op_context;
		if (rx_buf->state != RXM_LMT_ACK) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"invalid state. expected: %s, found: %s\n",
					rxm_lmt_state_str[RXM_LMT_START],
					rxm_lmt_state_str[rx_buf->state]);
			return -FI_EOPBADSTATE;
		}
		FI_DBG(&rxm_prov, FI_LOG_CQ, "rx_buf->state -> RXM_LMT_FINISH\n");
		rx_buf->state = RXM_LMT_FINISH;
		ret = rxm_finish_recv(rx_buf);
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown entry type!\n");
		assert(0);
		return -FI_EOTHER;
	}

	return ret;
}

static int rxm_handle_read_comp(struct rxm_rx_buf *rx_buf)
{
	struct iovec iov;
	struct fi_msg msg;
	struct rxm_pkt pkt;
	int ret;

	if (rx_buf->pkt.ctrl_hdr.type == ofi_ctrl_large_data) {
		if (rx_buf->state != RXM_LMT_START) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"invalid state. expected: %s, found: %s\n",
					rxm_lmt_state_str[RXM_LMT_START],
					rxm_lmt_state_str[rx_buf->state]);
			return -FI_EOPBADSTATE;
		}
		assert(rx_buf->conn);

		if (rx_buf->index < rx_buf->rma_iov->count)
			return rxm_lmt_rma_read(rx_buf);

		rxm_pkt_init(&pkt);
		pkt.ctrl_hdr.type = ofi_ctrl_ack;
		pkt.ctrl_hdr.conn_id = rx_buf->conn->handle.remote_key;
		pkt.ctrl_hdr.msg_id = rx_buf->pkt.ctrl_hdr.msg_id;
		pkt.hdr.op = rx_buf->pkt.hdr.op;

		iov.iov_base = &pkt;
		iov.iov_len = sizeof(pkt);

		memset(&msg, 0, sizeof(msg));
		msg.msg_iov = &iov;
		msg.iov_count = 1;
		msg.context = rx_buf;

		FI_DBG(&rxm_prov, FI_LOG_CQ, "rx_buf->state -> RXM_LMT_ACK\n");
		rx_buf->state = RXM_LMT_ACK;

		ret = fi_sendmsg(rx_buf->conn->msg_ep, &msg, FI_INJECT);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_CQ, "Unable to send ACK\n");
			rx_buf->state = RXM_LMT_NONE;
			return ret;
		}
	}
	// TODO process app RMA read
	return 0;
}

static ssize_t rxm_cq_read(struct fid_cq *msg_cq, struct fi_cq_msg_entry *comp)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_rx_buf *rx_buf;
	struct fi_cq_err_entry err_entry;
	ssize_t ret;

	ret = fi_cq_read(msg_cq, comp, 1);
	if (ret == -FI_EAVAIL) {
		OFI_CQ_READERR(&rxm_prov, FI_LOG_CQ, msg_cq,
				ret, err_entry);
		if (ret < 0) {
			FI_WARN(&rxm_prov, FI_LOG_CQ,
					"Unable to fi_cq_readerr on msg cq\n");
			return ret;
		}
		switch (*(enum rxm_ctx_type *)comp->op_context) {
		case RXM_TX_ENTRY:
			tx_entry = (struct rxm_tx_entry *)comp->op_context;
			return rxm_cq_report_error(tx_entry->ep->util_ep.tx_cq, &err_entry);
		case RXM_RX_BUF:
			rx_buf = (struct rxm_rx_buf *)comp->op_context;
			return rxm_cq_report_error(rx_buf->ep->util_ep.rx_cq, &err_entry);
		default:
			FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown ctx type!\n");
			FI_WARN(&rxm_prov, FI_LOG_CQ, "msg cq readerr: %s\n",
					fi_cq_strerror(msg_cq, err_entry.prov_errno,
						err_entry.err_data, NULL, 0));
			assert(0);
			return err_entry.err;
		}
	}
	return ret;
}

void rxm_cq_progress(struct fid_cq *msg_cq)
{
	struct fi_cq_msg_entry comp;
	ssize_t ret = 0;

	do {
		ret = rxm_cq_read(msg_cq, &comp);
		if (ret < 0)
			goto err;

		if (comp.flags & FI_RECV) {
			ret = rxm_handle_recv_comp(comp.op_context);
			if (ret)
				goto err;
		} else if (comp.flags & FI_SEND) {
			ret = rxm_handle_send_comp(comp.op_context);
			if (ret)
				goto err;
		} else if (comp.flags & FI_READ) {
			ret = rxm_handle_read_comp(comp.op_context);
			if (ret)
				goto err;
		} else {
			FI_WARN(&rxm_prov, FI_LOG_CQ, "Unknown completion!\n");
			goto err;
		}
	} while (ret > 0);
	return;
err:
	// TODO report error on RXM EP/domain since EP/CQ is broken.
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
