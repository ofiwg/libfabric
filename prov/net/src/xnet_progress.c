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
#include "xnet.h"
#include <poll.h>

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <ofi_util.h>
#include <ofi_iov.h>


static ssize_t (*xnet_start_op[ofi_op_write + 1])(struct xnet_ep *ep);


static ssize_t xnet_send_msg(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *tx_entry;
	ssize_t ret;
	size_t len;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
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
		tx_entry->ctrl_flags |= XNET_ASYNC;
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

static ssize_t xnet_recv_msg_data(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
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

static void xnet_progress_tx(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *tx_entry;
	struct xnet_cq *cq;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	while (ep->cur_tx.entry) {
		ret = xnet_send_msg(ep);
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			goto update;

		tx_entry = ep->cur_tx.entry;
		cq = container_of(ep->util_ep.tx_cq, struct xnet_cq, util_cq);

		if (ret) {
			FI_WARN(&xnet_prov, FI_LOG_DOMAIN, "msg send failed\n");
			xnet_cntr_incerr(ep, tx_entry);
			xnet_cq_report_error(&cq->util_cq, tx_entry, (int) -ret);
			xnet_free_xfer(ep, tx_entry);
		} else if (tx_entry->ctrl_flags & XNET_NEED_ACK) {
			/* A SW ack guarantees the peer received the data, so
			 * we can skip the async completion.
			 */
			slist_insert_tail(&tx_entry->entry,
					  &ep->need_ack_queue);
		} else if (tx_entry->ctrl_flags & XNET_NEED_RESP) {
			// discard send but enable receive for completeion
			assert(tx_entry->resp_entry);
			tx_entry->resp_entry->ctrl_flags &= ~XNET_INTERNAL_XFER;
			xnet_free_xfer(ep, tx_entry);
		} else if ((tx_entry->ctrl_flags & XNET_ASYNC) &&
			   (ofi_val32_gt(tx_entry->async_index,
					 ep->bsock.done_index))) {
			slist_insert_tail(&tx_entry->entry,
						&ep->async_queue);
		} else {
			ep->report_success(ep, &cq->util_cq, tx_entry);
			xnet_free_xfer(ep, tx_entry);
		}

		if (!slist_empty(&ep->priority_queue)) {
			ep->cur_tx.entry = container_of(slist_remove_head(
							&ep->priority_queue),
					     struct xnet_xfer_entry, entry);
			assert(ep->cur_tx.entry->ctrl_flags & XNET_INTERNAL_XFER);
		} else if (!slist_empty(&ep->tx_queue)) {
			ep->cur_tx.entry = container_of(slist_remove_head(
							&ep->tx_queue),
					     struct xnet_xfer_entry, entry);
			assert(!(ep->cur_tx.entry->ctrl_flags & XNET_INTERNAL_XFER));
		} else {
			ep->cur_tx.entry = NULL;
			break;
		}

		ep->cur_tx.data_left = ep->cur_tx.entry->hdr.base_hdr.size;
		OFI_DBG_SET(ep->cur_tx.entry->hdr.base_hdr.id, ep->tx_id++);
		ep->hdr_bswap(&ep->cur_tx.entry->hdr.base_hdr);
	}

	/* Buffered data is sent first by xnet_send_msg, but if we don't
	 * have other data to send, we need to try flushing any buffered data.
	 */
	(void) ofi_bsock_flush(&ep->bsock);
update:
	xnet_update_poll(ep);
}

static int xnet_queue_ack(struct xnet_xfer_entry *rx_entry)
{
	struct xnet_xfer_entry *resp;

	assert(xnet_progress_locked(xnet_ep2_progress(rx_entry->ep)));
	resp = xnet_alloc_xfer(xnet_ep2_progress(rx_entry->ep));
	if (!resp)
		return -FI_ENOMEM;

	resp->iov[0].iov_base = (void *) &resp->hdr;
	resp->iov[0].iov_len = sizeof(resp->hdr.base_hdr);
	resp->iov_cnt = 1;

	resp->hdr.base_hdr.version = XNET_HDR_VERSION;
	resp->hdr.base_hdr.op_data = XNET_OP_ACK;
	resp->hdr.base_hdr.op = ofi_op_msg;
	resp->hdr.base_hdr.size = sizeof(resp->hdr.base_hdr);
	resp->hdr.base_hdr.hdr_size = (uint8_t) sizeof(resp->hdr.base_hdr);

	resp->ctrl_flags = XNET_INTERNAL_XFER;
	resp->context = NULL;
	resp->ep = rx_entry->ep;

	xnet_tx_queue_insert(rx_entry->ep, resp);
	return FI_SUCCESS;
}

static ssize_t xnet_process_recv(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	rx_entry = ep->cur_rx.entry;
	ret = xnet_recv_msg_data(ep);
	if (ret) {
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			return ret;

		goto err;
	}

	if (rx_entry->hdr.base_hdr.flags & XNET_DELIVERY_COMPLETE) {
		ret = xnet_queue_ack(rx_entry);
		if (ret)
			goto err;
	}

	ep->report_success(ep, ep->util_ep.rx_cq, rx_entry);
	xnet_free_xfer(ep, rx_entry);
	xnet_reset_rx(ep);
	return 0;

err:
	FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
		"msg recv failed ret = %zd (%s)\n", ret, fi_strerror((int)-ret));
	xnet_cntr_incerr(ep, rx_entry);
	xnet_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	xnet_free_xfer(ep, rx_entry);
	xnet_reset_rx(ep);
	return ret;
}

static void xnet_pmem_commit(struct xnet_xfer_entry *rx_entry)
{
	struct ofi_rma_iov *rma_iov;
	size_t offset;
	int i;

	assert(xnet_progress_locked(xnet_ep2_progress(rx_entry->ep)));
	if (!ofi_pmem_commit)
		return ;

	if (rx_entry->hdr.base_hdr.flags & XNET_REMOTE_CQ_DATA)
		offset = sizeof(rx_entry->hdr.base_hdr) + sizeof(uint64_t);
	else
		offset = sizeof(rx_entry->hdr.base_hdr);


	rma_iov = (struct ofi_rma_iov *)((uint8_t *)&rx_entry->hdr + offset);

	for (i = 0; i < rx_entry->hdr.base_hdr.rma_iov_cnt; i++) {
		(*ofi_pmem_commit)((const void *) (uintptr_t) rma_iov[i].addr,
				   rma_iov[i].len);
	}
}

static ssize_t xnet_process_remote_write(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	rx_entry = ep->cur_rx.entry;
	ret = xnet_recv_msg_data(ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	if (rx_entry->hdr.base_hdr.flags &
	    (XNET_DELIVERY_COMPLETE | XNET_COMMIT_COMPLETE)) {

		if (rx_entry->hdr.base_hdr.flags & XNET_COMMIT_COMPLETE)
			xnet_pmem_commit(rx_entry);

		ret = xnet_queue_ack(rx_entry);
		if (ret)
			goto err;
	}

	ep->report_success(ep, ep->util_ep.rx_cq, rx_entry);
	xnet_free_xfer(ep, rx_entry);
	xnet_reset_rx(ep);
	return FI_SUCCESS;

err:
	FI_WARN(&xnet_prov, FI_LOG_DOMAIN, "remote write failed %zd\n", ret);
	xnet_free_xfer(ep, rx_entry);
	xnet_reset_rx(ep);
	return ret;
}

static ssize_t xnet_process_remote_read(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	struct xnet_cq *cq;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	rx_entry = ep->cur_rx.entry;
	cq = container_of(ep->util_ep.tx_cq, struct xnet_cq, util_cq);

	ret = xnet_recv_msg_data(ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	if (ret) {
		FI_WARN(&xnet_prov, FI_LOG_DOMAIN,
			"msg recv Failed ret = %zd\n", ret);
		xnet_cntr_incerr(ep, rx_entry);
		xnet_cq_report_error(&cq->util_cq, rx_entry, (int) -ret);
	} else {
		ep->report_success(ep, &cq->util_cq, rx_entry);
	}

	slist_remove_head(&rx_entry->ep->rma_read_queue);
	xnet_free_xfer(ep, rx_entry);
	xnet_reset_rx(ep);
	return ret;
}

static struct xnet_xfer_entry *xnet_get_rx_entry(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *xfer;
	struct xnet_srx *srx;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	if (ep->srx) {
		srx = ep->srx;
		if (!slist_empty(&srx->rx_queue)) {
			xfer = container_of(slist_remove_head(&srx->rx_queue),
					    struct xnet_xfer_entry, entry);
			xfer->cq_flags |= xnet_rx_completion_flag(ep, 0);
		} else {
			xfer = NULL;
		}
	} else {
		if (!slist_empty(&ep->rx_queue)) {
			xfer = container_of(slist_remove_head(&ep->rx_queue),
					    struct xnet_xfer_entry, entry);
			ep->rx_avail++;
		} else {
			xfer = NULL;
		}
	}

	return xfer;
}

static int xnet_handle_ack(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *tx_entry;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	if (ep->cur_rx.hdr.base_hdr.size !=
	    sizeof(ep->cur_rx.hdr.base_hdr))
		return -FI_EIO;

	assert(!slist_empty(&ep->need_ack_queue));
	tx_entry = container_of(slist_remove_head(&ep->need_ack_queue),
				struct xnet_xfer_entry, entry);

	ep->report_success(ep, ep->util_ep.tx_cq, tx_entry);
	xnet_free_xfer(ep, tx_entry);
	xnet_reset_rx(ep);
	return FI_SUCCESS;
}

static ssize_t xnet_op_msg(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	struct xnet_cur_rx *msg = &ep->cur_rx;
	size_t msg_len;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	if (msg->hdr.base_hdr.op_data == XNET_OP_ACK)
		return xnet_handle_ack(ep);

	msg_len = (msg->hdr.base_hdr.size - msg->hdr.base_hdr.hdr_size);

	rx_entry = xnet_get_rx_entry(ep);
	if (!rx_entry)
		return -FI_EAGAIN;

	memcpy(&rx_entry->hdr, &msg->hdr,
	       (size_t) msg->hdr.base_hdr.hdr_size);
	rx_entry->ep = ep;
	rx_entry->mrecv_msg_start = rx_entry->iov[0].iov_base;

	ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt,
				msg_len);
	if (ret)
		goto truncate_err;

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = xnet_process_recv;
	return xnet_process_recv(ep);

truncate_err:
	FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
		"posted rx buffer size is not big enough\n");
	xnet_cntr_incerr(ep, rx_entry);
	xnet_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	xnet_free_xfer(ep, rx_entry);
	return ret;
}

static ssize_t xnet_op_tagged(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	struct xnet_cur_rx *msg = &ep->cur_rx;
	size_t msg_len;
	uint64_t tag;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	assert(ep->srx);
	msg_len = (msg->hdr.base_hdr.size - msg->hdr.base_hdr.hdr_size);

	tag = (msg->hdr.base_hdr.flags & XNET_REMOTE_CQ_DATA) ?
	      msg->hdr.tag_data_hdr.tag : msg->hdr.tag_hdr.tag;

	rx_entry = ep->srx->match_tag_rx(ep->srx, ep, tag);
	if (!rx_entry)
		return -FI_EAGAIN;

	rx_entry->cq_flags |= xnet_rx_completion_flag(ep, 0);
	memcpy(&rx_entry->hdr, &msg->hdr,
	       (size_t) msg->hdr.base_hdr.hdr_size);
	rx_entry->ep = ep;

	ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt, msg_len);
	if (ret)
		goto truncate_err;

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = xnet_process_recv;
	return xnet_process_recv(ep);

truncate_err:
	FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
		"posted rx buffer size is not big enough\n");
	xnet_cntr_incerr(ep, rx_entry);
	xnet_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, (int) -ret);
	xnet_free_xfer(ep, rx_entry);
	return ret;
}

static ssize_t xnet_op_read_req(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *resp;
	struct ofi_rma_iov *rma_iov;
	ssize_t i, ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	resp = xnet_alloc_xfer(xnet_ep2_progress(ep));
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
			FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			xnet_free_xfer(ep, resp);
			return ret;
		}

		resp->iov[i + 1].iov_base = (void *) (uintptr_t)
					    rma_iov[i].addr;
		resp->iov[i + 1].iov_len = rma_iov[i].len;
		resp->hdr.base_hdr.size += resp->iov[i + 1].iov_len;
	}

	resp->hdr.base_hdr.op = ofi_op_read_rsp;
	resp->hdr.base_hdr.hdr_size = (uint8_t) sizeof(resp->hdr.base_hdr);

	resp->ctrl_flags = XNET_INTERNAL_XFER;
	resp->context = NULL;

	xnet_tx_queue_insert(ep, resp);
	xnet_reset_rx(ep);
	return FI_SUCCESS;
}

static ssize_t xnet_op_write(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	struct ofi_rma_iov *rma_iov;
	ssize_t ret, i;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	rx_entry = xnet_alloc_xfer(xnet_ep2_progress(ep));
	if (!rx_entry)
		return -FI_ENOMEM;

	if (ep->cur_rx.hdr.base_hdr.flags & XNET_REMOTE_CQ_DATA) {
		rx_entry->cq_flags = (FI_COMPLETION | FI_REMOTE_WRITE |
				      FI_REMOTE_CQ_DATA);
		rma_iov = (struct ofi_rma_iov *) ((uint8_t *) &rx_entry->hdr +
			   sizeof(rx_entry->hdr.cq_data_hdr));
	} else {
		rx_entry->ctrl_flags = XNET_INTERNAL_XFER;
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
			FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			xnet_free_xfer(ep, rx_entry);
			return ret;
		}
		rx_entry->iov[i].iov_base = (void *) (uintptr_t)
					      rma_iov[i].addr;
		rx_entry->iov[i].iov_len = rma_iov[i].len;
	}

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = xnet_process_remote_write;
	return xnet_process_remote_write(ep);
}

static ssize_t xnet_op_read_rsp(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *rx_entry;
	struct slist_entry *entry;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	if (slist_empty(&ep->rma_read_queue))
		return -FI_EINVAL;

	entry = ep->rma_read_queue.head;
	rx_entry = container_of(entry, struct xnet_xfer_entry, entry);

	memcpy(&rx_entry->hdr, &ep->cur_rx.hdr,
	       (size_t) ep->cur_rx.hdr.base_hdr.hdr_size);
	rx_entry->hdr.base_hdr.op_data = 0;

	ep->cur_rx.entry = rx_entry;
	ep->cur_rx.handler = xnet_process_remote_read;
	return xnet_process_remote_read(ep);
}

static ssize_t xnet_recv_hdr(struct xnet_ep *ep)
{
	size_t len;
	void *buf;
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
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

		if (ep->cur_rx.hdr.base_hdr.hdr_size > XNET_MAX_HDR) {
			FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
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
	if (ep->cur_rx.hdr.base_hdr.op >= ARRAY_SIZE(xnet_start_op)) {
		FI_WARN(&xnet_prov, FI_LOG_EP_DATA,
			"Received invalid opcode\n");
		return -FI_EIO;
	}

	ep->cur_rx.data_left = ep->cur_rx.hdr.base_hdr.size -
			       ep->cur_rx.hdr.base_hdr.hdr_size;
	ep->cur_rx.handler = xnet_start_op[ep->cur_rx.hdr.base_hdr.op];

	return ep->cur_rx.handler(ep);
}

void xnet_progress_rx(struct xnet_ep *ep)
{
	ssize_t ret;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	do {
		if (ep->cur_rx.hdr_done < ep->cur_rx.hdr_len) {
			ret = xnet_recv_hdr(ep);
		} else {
			ret = ep->cur_rx.handler(ep);
		}

	} while (!ret && ofi_bsock_readable(&ep->bsock));

	if (ret && !OFI_SOCK_TRY_SND_RCV_AGAIN(-ret)) {
		xnet_ep_disable(ep, 0, NULL, 0);
	} else if (xnet_active_wait(ep) && dlist_empty(&ep->active_entry)) {
		dlist_insert_tail(&ep->active_entry,
				  &xnet_ep2_progress(ep)->active_wait_list);
		xnet_signal_progress(xnet_ep2_progress(ep));
	}
}

void xnet_progress_async(struct xnet_ep *ep)
{
	struct xnet_xfer_entry *xfer;
	uint32_t done;

	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	done = ofi_bsock_async_done(&xnet_prov, &ep->bsock);
	while (!slist_empty(&ep->async_queue)) {
		xfer = container_of(ep->async_queue.head,
				    struct xnet_xfer_entry, entry);
		if (ofi_val32_gt(xfer->async_index, done))
			break;

		slist_remove_head(&ep->async_queue);
		ep->report_success(ep, ep->util_ep.tx_cq, xfer);
		xnet_free_xfer(ep, xfer);
	}
}

static bool xnet_tx_pending(struct xnet_ep *ep)
{
	return ep->cur_tx.entry || ofi_bsock_tosend(&ep->bsock);
}

void xnet_tx_queue_insert(struct xnet_ep *ep,
			  struct xnet_xfer_entry *tx_entry)
{
	assert(xnet_progress_locked(xnet_ep2_progress(ep)));

	if (!ep->cur_tx.entry) {
		ep->cur_tx.entry = tx_entry;
		ep->cur_tx.data_left = tx_entry->hdr.base_hdr.size;
		OFI_DBG_SET(tx_entry->hdr.base_hdr.id, ep->tx_id++);
		ep->hdr_bswap(&tx_entry->hdr.base_hdr);
		xnet_progress_tx(ep);
	} else if (tx_entry->ctrl_flags & XNET_INTERNAL_XFER) {
		slist_insert_tail(&tx_entry->entry, &ep->priority_queue);
	} else {
		slist_insert_tail(&tx_entry->entry, &ep->tx_queue);
	}
}

static ssize_t (*xnet_start_op[ofi_op_write + 1])(struct xnet_ep *ep) = {
	[ofi_op_msg] = xnet_op_msg,
	[ofi_op_tagged] = xnet_op_tagged,
	[ofi_op_read_req] = xnet_op_read_req,
	[ofi_op_read_rsp] = xnet_op_read_rsp,
	[ofi_op_write] = xnet_op_write,
};

static void xnet_run_ep(struct xnet_ep *ep, bool pin, bool pout, bool perr)
{
	assert(xnet_progress_locked(xnet_ep2_progress(ep)));
	switch (ep->state) {
	case XNET_CONNECTED:
		if (perr)
			xnet_progress_async(ep);
		if (pin)
			xnet_progress_rx(ep);
		if (pout)
			xnet_progress_tx(ep);
		break;
	case XNET_CONNECTING:
		xnet_connect_done(ep);
		break;
	case XNET_REQ_SENT:
		xnet_req_done(ep);
		break;
	default:
		break;
	};
}

static int
xnet_epoll_wait(struct xnet_progress *progress,
		struct ofi_epollfds_event *events, int max_events, int timeout)
{
	return ofi_epoll_wait(progress->epoll, events, max_events, timeout);
}

static int
xnet_epoll_add(struct xnet_progress *progress, int fd, uint32_t events,
	       void *context)
{
	return ofi_epoll_add(progress->epoll, fd, events, context);
}

static void
xnet_epoll_mod(struct xnet_progress *progress, int fd, uint32_t events,
	       void *context)
{
	(void) ofi_epoll_mod(progress->epoll, fd, events, context);
}

static int xnet_epoll_del(struct xnet_progress *progress, int fd)
{
	return ofi_epoll_del(progress->epoll, fd);
}

static void xnet_epoll_close(struct xnet_progress *progress)
{
	ofi_epoll_close(progress->epoll);
}

static int
xnet_pollfds_wait(struct xnet_progress *progress,
		  struct ofi_epollfds_event *events, int max_events, int timeout)
{
	return ofi_pollfds_wait(progress->pollfds, events, max_events, timeout);
}

static int
xnet_pollfds_add(struct xnet_progress *progress, int fd, uint32_t events,
		 void *context)
{
	return ofi_pollfds_add(progress->pollfds, fd, events, context);
}

static void
xnet_pollfds_mod(struct xnet_progress *progress, int fd, uint32_t events,
		 void *context)
{
	(void) ofi_pollfds_mod(progress->pollfds, fd, events, context);
}

static int xnet_pollfds_del(struct xnet_progress *progress, int fd)
{
	return ofi_pollfds_del(progress->pollfds, fd);
}

static void xnet_pollfds_close(struct xnet_progress *progress)
{
	ofi_pollfds_close(progress->pollfds);
}

static int xnet_poll_create(struct xnet_progress *progress, bool use_epoll)
{
	int ret;

	progress->use_epoll = use_epoll;
	if (use_epoll) {
		ret = ofi_epoll_create(&progress->epoll);
		progress->poll_wait = xnet_epoll_wait;
		progress->poll_add = xnet_epoll_add;
		progress->poll_mod = xnet_epoll_mod;
		progress->poll_del = xnet_epoll_del;
		progress->poll_close = xnet_epoll_close;
	} else {
		ret = ofi_pollfds_create(&progress->pollfds);
		progress->poll_wait = xnet_pollfds_wait;
		progress->poll_add = xnet_pollfds_add;
		progress->poll_mod = xnet_pollfds_mod;
		progress->poll_del = xnet_pollfds_del;
		progress->poll_close = xnet_pollfds_close;
	}

	return ret;
}

void xnet_run_progress(struct xnet_progress *progress, bool internal)
{
	struct ofi_epollfds_event events[XNET_MAX_EVENTS];
	struct dlist_entry *item, *tmp;
	struct xnet_ep *ep;
	struct fid *fid;
	int nfds, i;
	bool pin, pout, perr;

	ofi_genlock_held(progress->active_lock);
	dlist_foreach_safe(&progress->active_wait_list, item, tmp) {
		ep = container_of(item, struct xnet_ep, active_entry);

		if (xnet_active_wait(ep)) {
			assert(ep->state == XNET_CONNECTED);
			xnet_progress_rx(ep);
		} else {
			dlist_remove_init(&ep->active_entry);
		}
	}

	nfds = progress->poll_wait(progress, events, XNET_MAX_EVENTS, 0);
	if (nfds <= 0)
		goto out;

	for (i = 0; i < nfds; i++) {
		fid = events[i].data.ptr;
		assert(fid);

		pin = events[i].events & POLLIN;
		pout = events[i].events & POLLOUT;
		perr = events[i].events & POLLERR;

		switch (fid->fclass) {
		case FI_CLASS_EP:
			xnet_run_ep(events[i].data.ptr, pin, pout, perr);
			break;
		case FI_CLASS_PEP:
			xnet_accept_sock(events[i].data.ptr);
			break;
		case FI_CLASS_CONNREQ:
			xnet_run_conn(events[i].data.ptr, pin, pout, perr);
			break;
		default:
			assert(fid->fclass == XNET_CLASS_PROGRESS);
			/* Only allow the internal thread to clear the signal.
			 * This ensures that its poll set is up to date.
			 */
			if (internal)
				fd_signal_reset(&progress->signal);
			break;
		}
	}
out:
	xnet_handle_events(progress);
}

void xnet_progress(struct xnet_progress *progress, bool internal)
{
	ofi_genlock_lock(progress->active_lock);
	xnet_run_progress(progress, internal);
	ofi_genlock_unlock(progress->active_lock);
}

void xnet_progress_all(struct xnet_fabric *fabric)
{
	struct xnet_domain *domain;
	struct dlist_entry *item;

	ofi_mutex_lock(&fabric->util_fabric.lock);
	dlist_foreach(&fabric->util_fabric.domain_list, item) {
		domain = container_of(item, struct xnet_domain,
				      util_domain.list_entry);
		xnet_progress(&domain->progress, false);
	}

	ofi_mutex_unlock(&fabric->util_fabric.lock);

	xnet_progress(&fabric->progress, false);
}

/* We start progress thread(s) if app requests blocking reads */
int xnet_trywait(struct fid_fabric *fabric_fid, struct fid **fid, int count)
{
	return 0;
}

void xnet_update_poll(struct xnet_ep *ep)
{
	struct xnet_progress *progress;
	uint32_t events;
	bool tx_pending;

	progress = xnet_ep2_progress(ep);
	assert(xnet_progress_locked(progress));
	tx_pending = xnet_tx_pending(ep);
	if ((tx_pending && ep->pollout_set) ||
	    (!tx_pending && !ep->pollout_set))
		return;

	ep->pollout_set = tx_pending;
	events = ep->pollout_set ? POLLIN | POLLOUT : POLLIN;

	progress->poll_mod(progress, ep->bsock.sock,
			   events, &ep->util_ep.ep_fid.fid);
	xnet_signal_progress(progress);
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
static void *xnet_auto_progress(void *arg)
{
	struct xnet_progress *progress = arg;
	struct ofi_epollfds_event event;
	int timeout, nfds;

	FI_INFO(&xnet_prov, FI_LOG_DOMAIN, "progress thread starting\n");
	ofi_genlock_lock(progress->active_lock);
	while (progress->auto_progress) {
		timeout = dlist_empty(&progress->active_wait_list) ? -1 : 1;
		ofi_genlock_unlock(progress->active_lock);

		/* We can't hold the progress lock around waiting, or we
		 * can hang another thread trying to obtain the lock.  But
		 * the poll fds may change while we're waiting for an event.
		 * To avoid possibly processing an event for an object that
		 * we just removed from the poll fds, which could access freed
		 * memory, we must re-acquire the progress lock and re-read
		 * any queued events before processing it.
		 */
		nfds = progress->poll_wait(progress, &event, 1, timeout);

		ofi_genlock_lock(progress->active_lock);
		if (nfds >= 0)
			xnet_run_progress(progress, true);
	}
	ofi_genlock_unlock(progress->active_lock);
	FI_INFO(&xnet_prov, FI_LOG_DOMAIN, "progress thread exiting\n");
	return NULL;
}

int xnet_monitor_sock(struct xnet_progress *progress, SOCKET sock,
		      uint32_t events, struct fid *fid)
{
	int ret;

	assert(xnet_progress_locked(progress));
	ret = progress->poll_add(progress, sock, events, fid);
	if (ret) {
		FI_WARN(&xnet_prov, FI_LOG_EP_CTRL,
			"Failed to add fd to progress\n");
	}
	return ret;
}

/* May be called from progress thread to disable endpoint. */
void xnet_halt_sock(struct xnet_progress *progress, SOCKET sock)
{
	int ret;

	assert(xnet_progress_locked(progress));
	ret = progress->poll_del(progress, sock);
	if (ret) {
		FI_WARN(&xnet_prov, FI_LOG_EP_CTRL,
			"Failed to del fd from progress\n");
	}
}

int xnet_start_progress(struct xnet_progress *progress)
{
	int ret;

	ofi_genlock_lock(progress->active_lock);
	if (progress->auto_progress) {
		ret = 0;
		goto unlock;
	}

	progress->auto_progress = true;
	ret = pthread_create(&progress->thread, NULL, xnet_auto_progress,
			     progress);
	if (ret) {
		FI_WARN(&xnet_prov, FI_LOG_DOMAIN,
			"unable to start progress thread\n");
		progress->auto_progress = false;
		ret = -ret;
	}

unlock:
	ofi_genlock_unlock(progress->active_lock);
	return ret;
}

int xnet_start_all(struct xnet_fabric *fabric)
{
	struct xnet_domain *domain;
	struct dlist_entry *item;
	int ret;

	ret = xnet_start_progress(&fabric->progress);
	if (ret)
		return ret;

	ofi_mutex_lock(&fabric->util_fabric.lock);
	dlist_foreach(&fabric->util_fabric.domain_list, item) {
		domain = container_of(item, struct xnet_domain,
				      util_domain.list_entry);
		ret = xnet_start_progress(&domain->progress);
		if (ret)
			break;
	}

	ofi_mutex_unlock(&fabric->util_fabric.lock);
	return ret;
}

void xnet_stop_progress(struct xnet_progress *progress)
{
	ofi_genlock_lock(progress->active_lock);
	if (!progress->auto_progress) {
		ofi_genlock_unlock(progress->active_lock);
		return;
	}

	progress->auto_progress = false;
	fd_signal_set(&progress->signal);
	ofi_genlock_unlock(progress->active_lock);
	(void) pthread_join(progress->thread, NULL);
}

/* Because we may need to start the progress thread to support blocking CQ
 * or EQ calls, we always need to enable an active lock, independent from
 * the threading model requested by the app.
 */
static int xnet_init_locks(struct xnet_progress *progress, struct fi_info *info)
{
	enum ofi_lock_type base_type, rdm_type;
	int ret;

	if (info && info->ep_attr && info->ep_attr->type == FI_EP_RDM) {
		base_type = OFI_LOCK_NONE;
		rdm_type = OFI_LOCK_MUTEX;
		progress->active_lock = &progress->rdm_lock;
	} else {
		base_type = OFI_LOCK_MUTEX;
		rdm_type = OFI_LOCK_NONE;
		progress->active_lock = &progress->lock;
	}

	ret = ofi_genlock_init(&progress->lock, base_type);
	if (ret)
		return ret;

	ret = ofi_genlock_init(&progress->rdm_lock, rdm_type);
	if (ret)
		ofi_genlock_destroy(&progress->lock);

	return ret;
}

int xnet_init_progress(struct xnet_progress *progress, struct fi_info *info)
{
	int ret;

	progress->fid.fclass = XNET_CLASS_PROGRESS;
	progress->auto_progress = false;
	dlist_init(&progress->active_wait_list);
	slist_init(&progress->event_list);

	ret = fd_signal_init(&progress->signal);
	if (ret)
		return ret;

	ret = xnet_init_locks(progress, info);
	if (ret)
		goto err1;

	ret = xnet_poll_create(progress, false);
	if (ret)
		goto err2;

	ret = ofi_bufpool_create(&progress->xfer_pool,
				 sizeof(struct xnet_xfer_entry), 16, 0,
				 1024, 0);
	if (ret)
		goto err3;

	ret = progress->poll_add(progress, progress->signal.fd[FI_READ_FD],
				 POLLIN, &progress->fid);
	if (ret)
		goto err4;

	return 0;

err4:
	ofi_bufpool_destroy(progress->xfer_pool);
err3:
	progress->poll_close(progress);
err2:
	ofi_genlock_destroy(&progress->rdm_lock);
	ofi_genlock_destroy(&progress->lock);
err1:
	fd_signal_free(&progress->signal);
	return ret;
}

void xnet_close_progress(struct xnet_progress *progress)
{
	assert(dlist_empty(&progress->active_wait_list));
	assert(slist_empty(&progress->event_list));
	xnet_stop_progress(progress);
	progress->poll_close(progress);
	ofi_bufpool_destroy(progress->xfer_pool);
	ofi_genlock_destroy(&progress->lock);
	ofi_genlock_destroy(&progress->rdm_lock);
	fd_signal_free(&progress->signal);
}
