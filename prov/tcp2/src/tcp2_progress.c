/*
 * Copyright (c) 2017-2022 Intel Corporation, Inc.  All rights reserved.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "tcp2.h"
#include <poll.h>

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <ofi_util.h>
#include <ofi_iov.h>


static ssize_t (*tcp2_start_op[ofi_op_write + 1])(struct tcp2_ep *ep);


static ssize_t tcp2_send_msg(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *tx_entry;
	ssize_t ret;
	size_t len;

	assert(ep->cur_tx.entry);
	tx_entry = ep->cur_tx.entry;
	ret = ofi_bsock_sendv(&ep->bsock, tx_entry->iov, tx_entry->iov_cnt,
			      &len);
	if (ret < 0 && ret != -FI_EINPROGRESS)
		return ret;

	if (ret == -FI_EINPROGRESS) {
		/* If a transfer generated multiple async sends, we only
		 * need to track the last async index to know when the entire
		 * transfer has completed.
		 */
		tx_entry->async_index = ep->bsock.async_index;
		tx_entry->ctrl_flags |= TCP2_ASYNC;
	} else {
		len = ret;
	}

	ep->cur_tx.data_left -= len;
	if (ep->cur_tx.data_left) {
		ofi_consume_iov(tx_entry->iov, &tx_entry->iov_cnt, len);
		return -FI_EAGAIN;
	}
	return FI_SUCCESS;
}

static ssize_t tcp2_recv_msg_data(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	ssize_t ret;

	if (!ep->cur_rx.data_left)
		return FI_SUCCESS;

	rx_entry = ep->cur_rx.entry;
	ret = ofi_bsock_recvv(&ep->bsock, rx_entry->iov, rx_entry->iov_cnt);
	if (ret < 0)
		return ret;

	ep->cur_rx.data_left -= ret;
	if (!ep->cur_rx.data_left)
		return FI_SUCCESS;

	ofi_consume_iov(rx_entry->iov, &rx_entry->iov_cnt, ret);
	if (!rx_entry->iov_cnt || !rx_entry->iov[0].iov_len)
		return -FI_ETRUNC;

	return -FI_EAGAIN;
}

static void tcp2_progress_tx(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *tx_entry;
	struct tcp2_cq *cq;
	ssize_t ret;

	assert(ofi_mutex_held(&ep->lock));
	while (ep->cur_tx.entry) {
		ret = tcp2_send_msg(ep);
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			goto update;

		tx_entry = ep->cur_tx.entry;
		cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);

		if (ret) {
			FI_WARN(&tcp2_prov, FI_LOG_DOMAIN, "msg send failed\n");
			tcp2_cntr_incerr(ep, tx_entry);
			tcp2_cq_report_error(&cq->util_cq, tx_entry, (int) -ret);
			tcp2_free_xfer(cq, tx_entry);
		} else if (tx_entry->ctrl_flags & TCP2_NEED_ACK) {
			/* A SW ack guarantees the peer received the data, so
			 * we can skip the async completion.
			 */
			slist_insert_tail(&tx_entry->entry,
					  &ep->need_ack_queue);
		} else if (tx_entry->ctrl_flags & TCP2_NEED_RESP) {
			// discard send but enable receive for completeion
			assert(tx_entry->resp_entry);
			tx_entry->resp_entry->ctrl_flags &= ~TCP2_INTERNAL_XFER;
			tcp2_free_xfer(cq, tx_entry);
		} else if ((tx_entry->ctrl_flags & TCP2_ASYNC) &&
			   (ofi_val32_gt(tx_entry->async_index,
					 ep->bsock.done_index))) {
			slist_insert_tail(&tx_entry->entry,
						&ep->async_queue);
		} else {
			ep->report_success(ep, &cq->util_cq, tx_entry);
			tcp2_free_xfer(cq, tx_entry);
		}

		if (!slist_empty(&ep->priority_queue)) {
			ep->cur_tx.entry = container_of(slist_remove_head(
							&ep->priority_queue),
					     struct tcp2_xfer_entry, entry);
			assert(ep->cur_tx.entry->ctrl_flags & TCP2_INTERNAL_XFER);
		} else if (!slist_empty(&ep->tx_queue)) {
			ep->cur_tx.entry = container_of(slist_remove_head(
							&ep->tx_queue),
					     struct tcp2_xfer_entry, entry);
			assert(!(ep->cur_tx.entry->ctrl_flags & TCP2_INTERNAL_XFER));
		} else {
			ep->cur_tx.entry = NULL;
			break;
		}

		ep->cur_tx.data_left = ep->cur_tx.entry->hdr.base_hdr.size;
		OFI_DBG_SET(ep->cur_tx.entry->hdr.base_hdr.id, ep->tx_id++);
		ep->hdr_bswap(&ep->cur_tx.entry->hdr.base_hdr);
	}

	/* Buffered data is sent first by tcp2_send_msg, but if we don't
	 * have other data to send, we need to try flushing any buffered data.
	 */
	(void) ofi_bsock_flush(&ep->bsock);
update:
	tcp2_update_poll(ep);
}

static int tcp2_queue_ack(struct tcp2_xfer_entry *rx_entry)
{
	struct tcp2_ep *ep;
	struct tcp2_cq *cq;
	struct tcp2_xfer_entry *resp;

	ep = rx_entry->ep;
	cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);

	resp = tcp2_alloc_xfer(cq);
	if (!resp)
		return -FI_ENOMEM;

	resp->iov[0].iov_base = (void *) &resp->hdr;
	resp->iov[0].iov_len = sizeof(resp->hdr.base_hdr);
	resp->iov_cnt = 1;

	resp->hdr.base_hdr.version = TCP2_HDR_VERSION;
	resp->hdr.base_hdr.op_data = TCP2_OP_ACK;
	resp->hdr.base_hdr.op = ofi_op_msg;
	resp->hdr.base_hdr.size = sizeof(resp->hdr.base_hdr);
	resp->hdr.base_hdr.hdr_size = (uint8_t) sizeof(resp->hdr.base_hdr);

	resp->ctrl_flags = TCP2_INTERNAL_XFER;
	resp->context = NULL;
	resp->ep = ep;

	tcp2_tx_queue_insert(ep, resp);
	return FI_SUCCESS;
}

static int tcp2_update_rx_iov(struct tcp2_xfer_entry *rx_entry)
{
	struct ofi_cq_rbuf_entry cq_entry;
	int ret;

	assert(tcp2_dynamic_rbuf(rx_entry->ep));

	cq_entry.ep_context = rx_entry->ep->util_ep.ep_fid.fid.context;
	cq_entry.op_context = rx_entry->context;
	cq_entry.flags = 0;
	cq_entry.len = rx_entry->hdr.base_hdr.size -
		       rx_entry->hdr.base_hdr.hdr_size;
	cq_entry.buf = rx_entry->mrecv_msg_start;
	tcp2_get_cq_info(rx_entry, &cq_entry.flags, &cq_entry.data,
			 &cq_entry.tag);

	rx_entry->iov_cnt = TCP2_IOV_LIMIT;
	ret = (int) tcp2_dynamic_rbuf(rx_entry->ep)->
		    get_rbuf(&cq_entry, &rx_entry->iov[0], &rx_entry->iov_cnt);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
			"get_rbuf callback failed %s\n",
			fi_strerror(-ret));
		return ret;
	}

	assert(rx_entry->iov_cnt <= TCP2_IOV_LIMIT);
	ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt,
			       rx_entry->ep->cur_rx.data_left);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
			"dynamically provided rbuf is too small\n");
		return ret;
	}

	return 0;
}

static ssize_t tcp2_process_recv(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	ssize_t ret;

	rx_entry = ep->cur_rx.entry;
retry:
	ret = tcp2_recv_msg_data(ep);
	if (ret) {
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			return ret;

		if (ret != -FI_ETRUNC)
			goto err;
		assert(rx_entry->ctrl_flags & TCP2_NEED_DYN_RBUF);
	}

	if (rx_entry->ctrl_flags & TCP2_NEED_DYN_RBUF) {
		ret = tcp2_update_rx_iov(rx_entry);
		if (ret)
			goto err;

		rx_entry->ctrl_flags &= ~TCP2_NEED_DYN_RBUF;
		goto retry;
	}

	if (rx_entry->hdr.base_hdr.flags & TCP2_DELIVERY_COMPLETE) {
		ret = tcp2_queue_ack(rx_entry);
		if (ret)
			goto err;
	}

	ep->report_success(ep, ep->util_ep.rx_cq, rx_entry);
	tcp2_free_rx(rx_entry);
	tcp2_reset_rx(ep);
	return 0;

err:
	FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
		"msg recv failed ret = %zd (%s)\n", ret, fi_strerror((int)-ret));
	tcp2_cntr_incerr(ep, rx_entry);
	tcp2_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	tcp2_free_rx(rx_entry);
	tcp2_reset_rx(ep);
	return ret;
}

static void tcp2_pmem_commit(struct tcp2_xfer_entry *rx_entry)
{
	struct ofi_rma_iov *rma_iov;
	size_t offset;
	int i;

	if (!ofi_pmem_commit)
		return ;

	if (rx_entry->hdr.base_hdr.flags & TCP2_REMOTE_CQ_DATA)
		offset = sizeof(rx_entry->hdr.base_hdr) + sizeof(uint64_t);
	else
		offset = sizeof(rx_entry->hdr.base_hdr);


	rma_iov = (struct ofi_rma_iov *)((uint8_t *)&rx_entry->hdr + offset);

	for (i = 0; i < rx_entry->hdr.base_hdr.rma_iov_cnt; i++) {
		(*ofi_pmem_commit)((const void *) (uintptr_t) rma_iov[i].addr,
				   rma_iov[i].len);
	}
}

static ssize_t tcp2_process_remote_write(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct tcp2_cq *cq;
	ssize_t ret;

	rx_entry = ep->cur_rx.entry;
	ret = tcp2_recv_msg_data(ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	cq = container_of(ep->util_ep.rx_cq, struct tcp2_cq, util_cq);
	if (ret)
		goto err;

	if (rx_entry->hdr.base_hdr.flags &
	    (TCP2_DELIVERY_COMPLETE | TCP2_COMMIT_COMPLETE)) {

		if (rx_entry->hdr.base_hdr.flags & TCP2_COMMIT_COMPLETE)
			tcp2_pmem_commit(rx_entry);

		ret = tcp2_queue_ack(rx_entry);
		if (ret)
			goto err;
	}

	ep->report_success(ep, ep->util_ep.rx_cq, rx_entry);
	tcp2_free_xfer(cq, rx_entry);
	tcp2_reset_rx(ep);
	return FI_SUCCESS;

err:
	FI_WARN(&tcp2_prov, FI_LOG_DOMAIN, "remote write failed %zd\n", ret);
	tcp2_free_xfer(cq, rx_entry);
	tcp2_reset_rx(ep);
	return ret;
}

static ssize_t tcp2_process_remote_read(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct tcp2_cq *cq;
	ssize_t ret;

	rx_entry = ep->cur_rx.entry;
	cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);

	ret = tcp2_recv_msg_data(ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_DOMAIN,
			"msg recv Failed ret = %zd\n", ret);
		tcp2_cntr_incerr(ep, rx_entry);
		tcp2_cq_report_error(&cq->util_cq, rx_entry, (int) -ret);
	} else {
		ep->report_success(ep, &cq->util_cq, rx_entry);
	}

	slist_remove_head(&rx_entry->ep->rma_read_queue);
	tcp2_free_xfer(cq, rx_entry);
	tcp2_reset_rx(ep);
	return ret;
}

static struct tcp2_xfer_entry *tcp2_get_rx_entry(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *xfer;
	struct tcp2_rx_ctx *srx;

	if (ep->srx_ctx) {
		srx = ep->srx_ctx;
		ofi_mutex_lock(&srx->lock);
		if (!slist_empty(&srx->rx_queue)) {
			xfer = container_of(slist_remove_head(&srx->rx_queue),
					    struct tcp2_xfer_entry, entry);
			xfer->cq_flags |= tcp2_rx_completion_flag(ep, 0);
		} else {
			xfer = NULL;
		}
		ofi_mutex_unlock(&ep->srx_ctx->lock);
	} else {
		assert(ofi_mutex_held(&ep->lock));
		if (!slist_empty(&ep->rx_queue)) {
			xfer = container_of(slist_remove_head(&ep->rx_queue),
					    struct tcp2_xfer_entry, entry);
			ep->rx_avail++;
		} else {
			xfer = NULL;
		}
	}

	return xfer;
}

static int tcp2_handle_ack(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *tx_entry;
	struct tcp2_cq *cq;

	if (ep->cur_rx.hdr.base_hdr.size !=
	    sizeof(ep->cur_rx.hdr.base_hdr))
		return -FI_EIO;

	assert(!slist_empty(&ep->need_ack_queue));
	tx_entry = container_of(slist_remove_head(&ep->need_ack_queue),
				struct tcp2_xfer_entry, entry);

	cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);
	ep->report_success(ep, ep->util_ep.tx_cq, tx_entry);
	tcp2_free_xfer(cq, tx_entry);
	tcp2_reset_rx(ep);
	return FI_SUCCESS;
}

static ssize_t tcp2_op_msg(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct tcp2_cur_rx *msg = &ep->cur_rx;
	size_t msg_len;
	ssize_t ret;

	if (msg->hdr.base_hdr.op_data == TCP2_OP_ACK)
		return tcp2_handle_ack(ep);

	msg_len = (msg->hdr.base_hdr.size - msg->hdr.base_hdr.hdr_size);

	rx_entry = tcp2_get_rx_entry(ep);
	if (!rx_entry)
		return -FI_EAGAIN;

	memcpy(&rx_entry->hdr, &msg->hdr,
	       (size_t) msg->hdr.base_hdr.hdr_size);
	rx_entry->ep = ep;
	rx_entry->mrecv_msg_start = rx_entry->iov[0].iov_base;

	if (tcp2_dynamic_rbuf(ep)) {
		rx_entry->ctrl_flags = TCP2_NEED_DYN_RBUF;

		if (msg->hdr.base_hdr.flags & TCP2_TAGGED) {
			/* Raw message, no rxm header */
			rx_entry->iov_cnt = 0;
		} else {
			/* Receiving only rxm header */
			assert(msg_len >= ofi_total_iov_len(rx_entry->iov,
							    rx_entry->iov_cnt));
		}
	} else {
		ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt,
				       msg_len);
		if (ret)
			goto truncate_err;
	}

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = tcp2_process_recv;
	return tcp2_process_recv(ep);

truncate_err:
	FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
		"posted rx buffer size is not big enough\n");
	tcp2_cntr_incerr(ep, rx_entry);
	tcp2_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	tcp2_free_rx(rx_entry);
	return ret;
}

static ssize_t tcp2_op_tagged(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct tcp2_cur_rx *msg = &ep->cur_rx;
	size_t msg_len;
	uint64_t tag;
	ssize_t ret;

	assert(ep->srx_ctx && !tcp2_dynamic_rbuf(ep));
	msg_len = (msg->hdr.base_hdr.size - msg->hdr.base_hdr.hdr_size);

	tag = (msg->hdr.base_hdr.flags & TCP2_REMOTE_CQ_DATA) ?
	      msg->hdr.tag_data_hdr.tag : msg->hdr.tag_hdr.tag;

	rx_entry = ep->srx_ctx->match_tag_rx(ep->srx_ctx, ep, tag);
	if (!rx_entry)
		return -FI_EAGAIN;

	rx_entry->cq_flags |= tcp2_rx_completion_flag(ep, 0);
	memcpy(&rx_entry->hdr, &msg->hdr,
	       (size_t) msg->hdr.base_hdr.hdr_size);
	rx_entry->ep = ep;

	ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt, msg_len);
	if (ret)
		goto truncate_err;

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = tcp2_process_recv;
	return tcp2_process_recv(ep);

truncate_err:
	FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
		"posted rx buffer size is not big enough\n");
	tcp2_cntr_incerr(ep, rx_entry);
	tcp2_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	tcp2_free_rx(rx_entry);
	return ret;
}

static ssize_t tcp2_op_read_req(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *resp;
	struct tcp2_cq *cq;
	struct ofi_rma_iov *rma_iov;
	ssize_t i, ret;

	cq = container_of(ep->util_ep.tx_cq, struct tcp2_cq, util_cq);
	resp = tcp2_alloc_xfer(cq);
	if (!resp)
		return -FI_ENOMEM;

	memcpy(&resp->hdr, &ep->cur_rx.hdr,
	       (size_t) ep->cur_rx.hdr.base_hdr.hdr_size);
	resp->hdr.base_hdr.op_data = 0;
	resp->ep = ep;

	resp->iov[0].iov_base = (void *) &resp->hdr;
	resp->iov[0].iov_len = sizeof(resp->hdr.base_hdr);

	rma_iov = (struct ofi_rma_iov *) ((uint8_t *)
		  &resp->hdr + sizeof(resp->hdr.base_hdr));

	resp->iov_cnt = 1 + resp->hdr.base_hdr.rma_iov_cnt;
	resp->hdr.base_hdr.size = resp->iov[0].iov_len;
	for (i = 0; i < resp->hdr.base_hdr.rma_iov_cnt; i++) {
		ret = ofi_mr_verify(&ep->util_ep.domain->mr_map, rma_iov[i].len,
				    (uintptr_t *) &rma_iov[i].addr,
				    rma_iov[i].key, FI_REMOTE_READ);
		if (ret) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			tcp2_free_xfer(cq, resp);
			return ret;
		}

		resp->iov[i + 1].iov_base = (void *) (uintptr_t)
					    rma_iov[i].addr;
		resp->iov[i + 1].iov_len = rma_iov[i].len;
		resp->hdr.base_hdr.size += resp->iov[i + 1].iov_len;
	}

	resp->hdr.base_hdr.op = ofi_op_read_rsp;
	resp->hdr.base_hdr.hdr_size = (uint8_t) sizeof(resp->hdr.base_hdr);

	resp->ctrl_flags = TCP2_INTERNAL_XFER;
	resp->context = NULL;

	tcp2_tx_queue_insert(ep, resp);
	tcp2_reset_rx(ep);
	return FI_SUCCESS;
}

static ssize_t tcp2_op_write(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct tcp2_cq *cq;
	struct ofi_rma_iov *rma_iov;
	ssize_t ret, i;

	cq = container_of(ep->util_ep.rx_cq, struct tcp2_cq, util_cq);
	rx_entry = tcp2_alloc_xfer(cq);
	if (!rx_entry)
		return -FI_ENOMEM;

	if (ep->cur_rx.hdr.base_hdr.flags & TCP2_REMOTE_CQ_DATA) {
		rx_entry->cq_flags = (FI_COMPLETION | FI_REMOTE_WRITE |
				      FI_REMOTE_CQ_DATA);
		rma_iov = (struct ofi_rma_iov *) ((uint8_t *) &rx_entry->hdr +
			   sizeof(rx_entry->hdr.cq_data_hdr));
	} else {
		rx_entry->ctrl_flags = TCP2_INTERNAL_XFER;
		rma_iov = (struct ofi_rma_iov *) ((uint8_t *) &rx_entry->hdr +
			  sizeof(rx_entry->hdr.base_hdr));
	}

	memcpy(&rx_entry->hdr, &ep->cur_rx.hdr,
	       (size_t) ep->cur_rx.hdr.base_hdr.hdr_size);
	rx_entry->hdr.base_hdr.op_data = 0;
	rx_entry->ep = ep;

	rx_entry->iov_cnt = rx_entry->hdr.base_hdr.rma_iov_cnt;
	for (i = 0; i < rx_entry->hdr.base_hdr.rma_iov_cnt; i++) {
		ret = ofi_mr_verify(&ep->util_ep.domain->mr_map, rma_iov[i].len,
				    (uintptr_t *) &rma_iov[i].addr,
				    rma_iov[i].key, FI_REMOTE_WRITE);
		if (ret) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			tcp2_free_xfer(cq, rx_entry);
			return ret;
		}
		rx_entry->iov[i].iov_base = (void *) (uintptr_t)
					      rma_iov[i].addr;
		rx_entry->iov[i].iov_len = rma_iov[i].len;
	}

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = tcp2_process_remote_write;
	return tcp2_process_remote_write(ep);
}

static ssize_t tcp2_op_read_rsp(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *rx_entry;
	struct slist_entry *entry;

	if (slist_empty(&ep->rma_read_queue))
		return -FI_EINVAL;

	entry = ep->rma_read_queue.head;
	rx_entry = container_of(entry, struct tcp2_xfer_entry, entry);

	memcpy(&rx_entry->hdr, &ep->cur_rx.hdr,
	       (size_t) ep->cur_rx.hdr.base_hdr.hdr_size);
	rx_entry->hdr.base_hdr.op_data = 0;

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = tcp2_process_remote_read;
	return tcp2_process_remote_read(ep);
}

static ssize_t tcp2_recv_hdr(struct tcp2_ep *ep)
{
	size_t len;
	void *buf;
	ssize_t ret;

	assert(ep->cur_rx.hdr_done < ep->cur_rx.hdr_len);

next_hdr:
	buf = (uint8_t *) &ep->cur_rx.hdr + ep->cur_rx.hdr_done;
	len = ep->cur_rx.hdr_len - ep->cur_rx.hdr_done;
	ret = ofi_bsock_recv(&ep->bsock, buf, len);
	if (ret < 0)
		return ret;

	ep->cur_rx.hdr_done += ret;
	if (ep->cur_rx.hdr_done == sizeof(ep->cur_rx.hdr.base_hdr)) {
		assert(ep->cur_rx.hdr_len == sizeof(ep->cur_rx.hdr.base_hdr));

		if (ep->cur_rx.hdr.base_hdr.hdr_size > TCP2_MAX_HDR) {
			FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
				"Payload offset is too large\n");
			return -FI_EIO;
		}
		ep->cur_rx.hdr_len = (size_t) ep->cur_rx.hdr.base_hdr.hdr_size;
		if (ep->cur_rx.hdr_done < ep->cur_rx.hdr_len)
			goto next_hdr;

	} else if (ep->cur_rx.hdr_done < ep->cur_rx.hdr_len) {
		return -FI_EAGAIN;
	}

	ep->hdr_bswap(&ep->cur_rx.hdr.base_hdr);
	assert(ep->cur_rx.hdr.base_hdr.id == ep->rx_id++);
	if (ep->cur_rx.hdr.base_hdr.op >= ARRAY_SIZE(tcp2_start_op)) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_DATA,
			"Received invalid opcode\n");
		return -FI_EIO;
	}

	ep->cur_rx.data_left = ep->cur_rx.hdr.base_hdr.size -
			       ep->cur_rx.hdr.base_hdr.hdr_size;
	ep->cur_rx.handler = tcp2_start_op[ep->cur_rx.hdr.base_hdr.op];

	return ep->cur_rx.handler(ep);
}

void tcp2_progress_rx(struct tcp2_ep *ep)
{
	ssize_t ret;

	assert(ofi_mutex_held(&ep->lock));
	do {
		if (ep->cur_rx.hdr_done < ep->cur_rx.hdr_len) {
			ret = tcp2_recv_hdr(ep);
		} else {
			ret = ep->cur_rx.handler(ep);
		}

	} while (!ret && ofi_bsock_readable(&ep->bsock));

	if (ret && !OFI_SOCK_TRY_SND_RCV_AGAIN(-ret)) {
		tcp2_ep_disable(ep, 0, NULL, 0);
	} else if (tcp2_active_wait(ep) && dlist_empty(&ep->progress_entry)) {
		dlist_insert_tail(&ep->progress_entry,
				  &tcp2_ep2_progress(ep)->active_wait_list);
		tcp2_signal_progress(tcp2_ep2_progress(ep));
	}
}

void tcp2_progress_async(struct tcp2_ep *ep)
{
	struct tcp2_xfer_entry *xfer;
	uint32_t done;

	assert(ofi_mutex_held(&ep->lock));
	done = ofi_bsock_async_done(&tcp2_prov, &ep->bsock);
	while (!slist_empty(&ep->async_queue)) {
		xfer = container_of(ep->async_queue.head,
				    struct tcp2_xfer_entry, entry);
		if (ofi_val32_gt(xfer->async_index, done))
			break;

		slist_remove_head(&ep->async_queue);
		ep->report_success(ep, ep->util_ep.tx_cq, xfer);
		tcp2_free_tx(xfer);
	}
}

static bool tcp2_tx_pending(struct tcp2_ep *ep)
{
	return ep->cur_tx.entry || ofi_bsock_tosend(&ep->bsock);
}

void tcp2_tx_queue_insert(struct tcp2_ep *ep,
			  struct tcp2_xfer_entry *tx_entry)
{
	assert(ofi_mutex_held(&tcp2_ep2_progress(ep)->lock));
	assert(ofi_mutex_held(&ep->lock));

	if (!ep->cur_tx.entry) {
		ep->cur_tx.entry = tx_entry;
		ep->cur_tx.data_left = tx_entry->hdr.base_hdr.size;
		OFI_DBG_SET(tx_entry->hdr.base_hdr.id, ep->tx_id++);
		ep->hdr_bswap(&tx_entry->hdr.base_hdr);
		tcp2_progress_tx(ep);
	} else if (tx_entry->ctrl_flags & TCP2_INTERNAL_XFER) {
		slist_insert_tail(&tx_entry->entry, &ep->priority_queue);
	} else {
		slist_insert_tail(&tx_entry->entry, &ep->tx_queue);
	}
}

static ssize_t (*tcp2_start_op[ofi_op_write + 1])(struct tcp2_ep *ep) = {
	[ofi_op_msg] = tcp2_op_msg,
	[ofi_op_tagged] = tcp2_op_tagged,
	[ofi_op_read_req] = tcp2_op_read_req,
	[ofi_op_read_rsp] = tcp2_op_read_rsp,
	[ofi_op_write] = tcp2_op_write,
};

static void tcp2_run_ep(struct tcp2_ep *ep, bool pin, bool pout, bool perr)
{
	assert(ofi_mutex_held(&tcp2_ep2_progress(ep)->lock));
	ofi_mutex_lock(&ep->lock);
	switch (ep->state) {
	case TCP2_CONNECTED:
		if (perr)
			tcp2_progress_async(ep);
		if (pin)
			tcp2_progress_rx(ep);
		if (pout)
			tcp2_progress_tx(ep);
		break;
	case TCP2_CONNECTING:
		tcp2_connect_done(ep);
		break;
	case TCP2_REQ_SENT:
		tcp2_req_done(ep);
		break;
	default:
		break;
	};
	ofi_mutex_unlock(&ep->lock);
}

static int
tcp2_epoll_wait(struct tcp2_progress *progress,
		struct ofi_epollfds_event *events, int max_events, int timeout)
{
	return ofi_epoll_wait(progress->epoll, events, max_events, timeout);
}

static int
tcp2_epoll_add(struct tcp2_progress *progress, int fd, uint32_t events,
	       void *context)
{
	return ofi_epoll_add(progress->epoll, fd, events, context);
}

static void
tcp2_epoll_mod(struct tcp2_progress *progress, int fd, uint32_t events,
	       void *context)
{
	(void) ofi_epoll_mod(progress->epoll, fd, events, context);
}

static int tcp2_epoll_del(struct tcp2_progress *progress, int fd)
{
	return ofi_epoll_del(progress->epoll, fd);
}

static void tcp2_epoll_close(struct tcp2_progress *progress)
{
	ofi_epoll_close(progress->epoll);
}

static int
tcp2_pollfds_wait(struct tcp2_progress *progress,
		  struct ofi_epollfds_event *events, int max_events, int timeout)
{
	return ofi_pollfds_wait(progress->pollfds, events, max_events, timeout);
}

static int
tcp2_pollfds_add(struct tcp2_progress *progress, int fd, uint32_t events,
		 void *context)
{
	return ofi_pollfds_add(progress->pollfds, fd, events, context);
}

static void
tcp2_pollfds_mod(struct tcp2_progress *progress, int fd, uint32_t events,
		 void *context)
{
	(void) ofi_pollfds_mod(progress->pollfds, fd, events, context);
}

static int tcp2_pollfds_del(struct tcp2_progress *progress, int fd)
{
	return ofi_pollfds_del(progress->pollfds, fd);
}

static void tcp2_pollfds_close(struct tcp2_progress *progress)
{
	ofi_pollfds_close(progress->pollfds);
}

static int tcp2_poll_create(struct tcp2_progress *progress, bool use_epoll)
{
	int ret;

	progress->use_epoll = use_epoll;
	if (use_epoll) {
		ret = ofi_epoll_create(&progress->epoll);
		progress->poll_wait = tcp2_epoll_wait;
		progress->poll_add = tcp2_epoll_add;
		progress->poll_mod = tcp2_epoll_mod;
		progress->poll_del = tcp2_epoll_del;
		progress->poll_close = tcp2_epoll_close;
	} else {
		ret = ofi_pollfds_create(&progress->pollfds);
		progress->poll_wait = tcp2_pollfds_wait;
		progress->poll_add = tcp2_pollfds_add;
		progress->poll_mod = tcp2_pollfds_mod;
		progress->poll_del = tcp2_pollfds_del;
		progress->poll_close = tcp2_pollfds_close;
	}

	return ret;
}

void tcp2_run_progress(struct tcp2_progress *progress, bool internal)
{
	struct ofi_epollfds_event events[TCP2_MAX_EVENTS];
	struct dlist_entry *item, *tmp;
	struct tcp2_ep *ep;
	struct fid *fid;
	int nfds, i;
	bool pin, pout, perr;

	ofi_mutex_lock(&progress->lock);
	dlist_foreach_safe(&progress->active_wait_list, item, tmp) {
		ep = container_of(item, struct tcp2_ep, progress_entry);

		ofi_mutex_lock(&ep->lock);

		if (tcp2_active_wait(ep)) {
			assert(ep->state == TCP2_CONNECTED);
			tcp2_progress_rx(ep);
		} else {
			dlist_remove_init(&ep->progress_entry);
		}
		ofi_mutex_unlock(&ep->lock);
	}

	nfds = progress->poll_wait(progress, events, TCP2_MAX_EVENTS, 0);
	if (nfds <= 0)
		goto unlock;

	for (i = 0; i < nfds; i++) {
		fid = events[i].data.ptr;
		assert(fid);

		pin = events[i].events & POLLIN;
		pout = events[i].events & POLLOUT;
		perr = events[i].events & POLLERR;

		switch (fid->fclass) {
		case FI_CLASS_EP:
			tcp2_run_ep(events[i].data.ptr, pin, pout, perr);
			break;
		case FI_CLASS_PEP:
			tcp2_accept_sock(events[i].data.ptr);
			break;
		case FI_CLASS_CONNREQ:
			tcp2_run_conn(events[i].data.ptr, pin, pout, perr);
			break;
		default:
			assert(fid->fclass == TCP2_CLASS_PROGRESS);
			/* Only allow the internal thread to clear the signal.
			 * This ensures that its poll set is up to date.
			 */
			if (internal)
				fd_signal_reset(&progress->signal);
			break;
		}
	}
unlock:
	ofi_mutex_unlock(&progress->lock);
}

void tcp2_progress_all(struct tcp2_fabric *fabric)
{
	struct tcp2_domain *domain;
	struct dlist_entry *item;

	ofi_mutex_lock(&fabric->util_fabric.lock);
	dlist_foreach(&fabric->util_fabric.domain_list, item) {
		domain = container_of(item, struct tcp2_domain,
				      util_domain.list_entry);
		tcp2_run_progress(&domain->progress, false);
	}

	ofi_mutex_unlock(&fabric->util_fabric.lock);

	tcp2_run_progress(&fabric->progress, false);
}

/* We start progress thread(s) if app requests blocking reads */
int tcp2_trywait(struct fid_fabric *fabric_fid, struct fid **fid, int count)
{
	return 0;
}

void tcp2_update_poll(struct tcp2_ep *ep)
{
	struct tcp2_progress *progress;
	uint32_t events;
	bool tx_pending;

	progress = tcp2_ep2_progress(ep);
	assert(ofi_mutex_held(&progress->lock));
	assert(ofi_mutex_held(&ep->lock));
	tx_pending = tcp2_tx_pending(ep);
	if ((tx_pending && ep->pollout_set) ||
	    (!tx_pending && !ep->pollout_set))
		return;

	ep->pollout_set = tx_pending;
	events = ep->pollout_set ? POLLIN | POLLOUT : POLLIN;

	progress->poll_mod(progress, ep->bsock.sock,
			   events, &ep->util_ep.ep_fid.fid);
	tcp2_signal_progress(progress);
}

/* If we're only using auto progress to drive transfers, we end up with an
 * unfortunate choice.  See the comment in the progress function about
 * waiting for the application to post a buffer.  If that situation occurs,
 * then we either need for the progress thread to spin until the application
 * posts the necessary receive buffer, or we block the thread.  However, if we
 * block the thread, there's no good way to wake-up the thread to resume
 * processing.  We could set some state that we check on every posted receive
 * operation and use that to signal the thread, but that introduces overhead
 * to every receive call.  As an alternative, we wake-up the thread
 * periodically, so it can check for progress.
 */
static void *tcp2_auto_progress(void *arg)
{
	struct tcp2_progress *progress = arg;
	struct ofi_epollfds_event event;
	int timeout, nfds;

	FI_INFO(&tcp2_prov, FI_LOG_DOMAIN, "progress thread starting\n");
	ofi_mutex_lock(&progress->lock);
	while (progress->auto_progress) {
		timeout = dlist_empty(&progress->active_wait_list) ? -1 : 1;
		ofi_mutex_unlock(&progress->lock);

		/* We can't hold the progress lock around waiting, or we
		 * can hang another thread trying to obtain the lock.  But
		 * the poll fds may change while we're waiting for an event.
		 * To avoid possibly processing an event for an object that
		 * we just removed from the poll fds, which could access freed
		 * memory, we must re-acquire the progress lock and re-read
		 * any queued events before processing it.
		 */
		nfds = progress->poll_wait(progress, &event, 1, timeout);

		if (nfds >= 0)
			tcp2_run_progress(progress, true);
		ofi_mutex_lock(&progress->lock);
	}
	ofi_mutex_unlock(&progress->lock);
	FI_INFO(&tcp2_prov, FI_LOG_DOMAIN, "progress thread exiting\n");
	return NULL;
}

int tcp2_monitor_sock(struct tcp2_progress *progress, SOCKET sock,
		      uint32_t events, struct fid *fid)
{
	int ret;

	assert(ofi_mutex_held(&progress->lock));
	ret = progress->poll_add(progress, sock, events, fid);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"Failed to add fd to progress\n");
	}
	return ret;
}

/* May be called from progress thread to disable endpoint. */
void tcp2_halt_sock(struct tcp2_progress *progress, SOCKET sock)
{
	int ret;

	assert(ofi_mutex_held(&progress->lock));
	ret = progress->poll_del(progress, sock);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_EP_CTRL,
			"Failed to del fd from progress\n");
	}
}

int tcp2_start_progress(struct tcp2_progress *progress)
{
	int ret;

	ofi_mutex_lock(&progress->lock);
	if (progress->auto_progress) {
		ret = 0;
		goto unlock;
	}

	progress->auto_progress = true;
	ret = pthread_create(&progress->thread, NULL, tcp2_auto_progress,
			     progress);
	if (ret) {
		FI_WARN(&tcp2_prov, FI_LOG_DOMAIN,
			"unable to start progress thread\n");
		progress->auto_progress = false;
		ret = -ret;
	}

unlock:
	ofi_mutex_unlock(&progress->lock);
	return ret;
}

int tcp2_start_all(struct tcp2_fabric *fabric)
{
	struct tcp2_domain *domain;
	struct dlist_entry *item;
	int ret;

	ret = tcp2_start_progress(&fabric->progress);
	if (ret)
		return ret;

	ofi_mutex_lock(&fabric->util_fabric.lock);
	dlist_foreach(&fabric->util_fabric.domain_list, item) {
		domain = container_of(item, struct tcp2_domain,
				      util_domain.list_entry);
		ret = tcp2_start_progress(&domain->progress);
		if (ret)
			break;
	}

	ofi_mutex_unlock(&fabric->util_fabric.lock);
	return ret;
}

void tcp2_stop_progress(struct tcp2_progress *progress)
{
	ofi_mutex_lock(&progress->lock);
	if (!progress->auto_progress) {
		ofi_mutex_unlock(&progress->lock);
		return;
	}

	progress->auto_progress = false;
	fd_signal_set(&progress->signal);
	ofi_mutex_unlock(&progress->lock);
	(void) pthread_join(progress->thread, NULL);
}

int tcp2_init_progress(struct tcp2_progress *progress, bool use_epoll)
{
	int ret;

	progress->fid.fclass = TCP2_CLASS_PROGRESS;
	progress->auto_progress = false;
	dlist_init(&progress->active_wait_list);

	ret = fd_signal_init(&progress->signal);
	if (ret)
		return ret;

	ret = ofi_mutex_init(&progress->lock);
	if (ret)
		goto free_sig;

	ret = tcp2_poll_create(progress, use_epoll);
	if (ret)
		goto destroy;

	ret = progress->poll_add(progress, progress->signal.fd[FI_READ_FD],
				 POLLIN, &progress->fid);
	if (ret) {
		progress->poll_close(progress);
		goto destroy;
	}

	return 0;

destroy:
	ofi_mutex_destroy(&progress->lock);
free_sig:
	fd_signal_free(&progress->signal);
	return ret;
}

void tcp2_close_progress(struct tcp2_progress *progress)
{
	assert(dlist_empty(&progress->active_wait_list));
	tcp2_stop_progress(progress);
	progress->poll_close(progress);
	ofi_mutex_destroy(&progress->lock);
	fd_signal_free(&progress->signal);
}
