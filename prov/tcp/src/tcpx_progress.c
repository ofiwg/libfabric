/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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
#include "tcpx.h"
#include <poll.h>

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <ofi_util.h>
#include <ofi_iov.h>


void tcpx_progress_tx(struct tcpx_ep *ep)
{
	struct tcpx_xfer_entry *tx_entry;
	struct tcpx_cq *cq;
	int ret;

	assert(fastlock_held(&ep->lock));
	if (!ep->cur_tx_entry) {
		(void) ofi_bsock_flush(&ep->bsock);
		return;
	}

	ret = tcpx_send_msg(ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	tx_entry = ep->cur_tx_entry;
	ep->hdr_bswap(&tx_entry->hdr.base_hdr);
	cq = container_of(ep->util_ep.tx_cq, struct tcpx_cq, util_cq);

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg send failed\n");
		tcpx_cq_report_error(&cq->util_cq, tx_entry, -ret);
		tcpx_xfer_entry_free(cq, tx_entry);
	} else {
		if (tx_entry->hdr.base_hdr.flags &
		    (TCPX_DELIVERY_COMPLETE | TCPX_COMMIT_COMPLETE)) {
			slist_insert_tail(&tx_entry->entry,
					  &ep->tx_rsp_pend_queue);
		} else {
			tcpx_cq_report_success(&cq->util_cq, tx_entry);
			tcpx_xfer_entry_free(cq, tx_entry);
		}
	}

	if (!slist_empty(&ep->tx_queue)) {
		ep->cur_tx_entry = container_of(slist_remove_head(&ep->tx_queue),
						struct tcpx_xfer_entry, entry);
		ep->rem_tx_len = ep->cur_tx_entry->hdr.base_hdr.size;
		ep->hdr_bswap(&ep->cur_tx_entry->hdr.base_hdr);
	} else {
		ep->cur_tx_entry = NULL;
	}
}

static int tcpx_prepare_rx_entry_resp(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_tx_cq;
	struct tcpx_xfer_entry *resp_entry;

	tcpx_tx_cq = container_of(rx_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);

	resp_entry = tcpx_xfer_entry_alloc(tcpx_tx_cq, TCPX_OP_MSG_RESP);
	if (!resp_entry)
		return -FI_EAGAIN;

	resp_entry->iov[0].iov_base = (void *) &resp_entry->hdr;
	resp_entry->iov[0].iov_len = sizeof(resp_entry->hdr.base_hdr);
	resp_entry->iov_cnt = 1;

	resp_entry->hdr.base_hdr.op = ofi_op_msg;
	resp_entry->hdr.base_hdr.size = sizeof(resp_entry->hdr.base_hdr);
	resp_entry->hdr.base_hdr.payload_off =
		(uint8_t)sizeof(resp_entry->hdr.base_hdr);

	resp_entry->flags = 0;
	resp_entry->context = NULL;
	resp_entry->ep = rx_entry->ep;

	tcpx_tx_queue_insert(resp_entry->ep, resp_entry);
	tcpx_cq_report_success(rx_entry->ep->util_ep.rx_cq, rx_entry);

	tcpx_rx_entry_free(rx_entry);
	return FI_SUCCESS;
}

static int tcpx_update_rx_iov(struct tcpx_xfer_entry *rx_entry)
{
	struct ofi_cq_rbuf_entry cq_entry;
	int ret;

	assert(tcpx_dynamic_rbuf(rx_entry->ep));

	cq_entry.ep_context = rx_entry->ep->util_ep.ep_fid.fid.context;
	cq_entry.op_context = rx_entry->context;
	cq_entry.flags = 0;
	cq_entry.len = rx_entry->hdr.base_hdr.size -
		       rx_entry->hdr.base_hdr.payload_off;
	cq_entry.buf = rx_entry->mrecv_msg_start;
	tcpx_get_cq_info(rx_entry, &cq_entry.flags, &cq_entry.data,
			 &cq_entry.tag);

	rx_entry->iov_cnt = TCPX_IOV_LIMIT;
	ret = (int) tcpx_dynamic_rbuf(rx_entry->ep)->
		    get_rbuf(&cq_entry, &rx_entry->iov[0], &rx_entry->iov_cnt);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			"get_rbuf callback failed %s\n",
			fi_strerror(-ret));
		return ret;
	}

	assert(rx_entry->iov_cnt <= TCPX_IOV_LIMIT);
	ret = ofi_truncate_iov(rx_entry->iov, &rx_entry->iov_cnt,
			       rx_entry->ep->rem_rx_len);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			"dynamically provided rbuf is too small\n");
		return ret;
	}

	return 0;
}

static int tcpx_process_recv(struct tcpx_xfer_entry *rx_entry)
{
	int ret;

retry:
	ret = tcpx_recv_msg_data(rx_entry->ep);
	if (ret) {
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			return ret;

		if (ret != -FI_ETRUNC)
			goto shutdown;
		assert(rx_entry->flags & TCPX_NEED_DYN_RBUF);
	}

	if (rx_entry->flags & TCPX_NEED_DYN_RBUF) {
		ret = tcpx_update_rx_iov(rx_entry);
		if (ret)
			goto shutdown;

		rx_entry->flags &= ~TCPX_NEED_DYN_RBUF;
		goto retry;
	}

	if (rx_entry->hdr.base_hdr.flags & OFI_DELIVERY_COMPLETE) {
		if (tcpx_prepare_rx_entry_resp(rx_entry))
			rx_entry->ep->cur_rx_proc_fn = tcpx_prepare_rx_entry_resp;
	} else {
		tcpx_cq_report_success(rx_entry->ep->util_ep.rx_cq, rx_entry);
		tcpx_rx_entry_free(rx_entry);
	}
	return 0;

shutdown:
	FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
		"msg recv failed ret = %d (%s)\n", ret, fi_strerror(-ret));
	tcpx_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, -ret);
	tcpx_rx_entry_free(rx_entry);
	return ret;
}

static int tcpx_prepare_rx_write_resp(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_rx_cq, *tcpx_tx_cq;
	struct tcpx_xfer_entry *resp_entry;

	tcpx_tx_cq = container_of(rx_entry->ep->util_ep.tx_cq,
				  struct tcpx_cq, util_cq);

	resp_entry = tcpx_xfer_entry_alloc(tcpx_tx_cq, TCPX_OP_MSG_RESP);
	if (!resp_entry)
		return -FI_EAGAIN;

	resp_entry->iov[0].iov_base = (void *) &resp_entry->hdr;
	resp_entry->iov[0].iov_len = sizeof(resp_entry->hdr.base_hdr);
	resp_entry->iov_cnt = 1;

	resp_entry->hdr.base_hdr.op = ofi_op_msg;
	resp_entry->hdr.base_hdr.size = sizeof(resp_entry->hdr.base_hdr);
	resp_entry->hdr.base_hdr.payload_off = (uint8_t)
						sizeof(resp_entry->hdr.base_hdr);

	resp_entry->flags &= ~FI_COMPLETION;
	resp_entry->context = NULL;
	resp_entry->ep = rx_entry->ep;
	tcpx_tx_queue_insert(resp_entry->ep, resp_entry);

	tcpx_cq_report_success(rx_entry->ep->util_ep.rx_cq, rx_entry);
	tcpx_rx_cq = container_of(rx_entry->ep->util_ep.rx_cq,
				  struct tcpx_cq, util_cq);
	tcpx_xfer_entry_free(tcpx_rx_cq, rx_entry);
	return FI_SUCCESS;
}

static void tcpx_pmem_commit(struct tcpx_xfer_entry *rx_entry)
{
	struct ofi_rma_iov *rma_iov;
	size_t offset;
	int i;

	if (!ofi_pmem_commit)
		return ;

	if (rx_entry->hdr.base_hdr.flags & TCPX_REMOTE_CQ_DATA)
		offset = sizeof(rx_entry->hdr.base_hdr) + sizeof(uint64_t);
	else
		offset = sizeof(rx_entry->hdr.base_hdr);


	rma_iov = (struct ofi_rma_iov *)((uint8_t *)&rx_entry->hdr + offset);

	for (i = 0; i < rx_entry->hdr.base_hdr.rma_iov_cnt; i++) {
		(*ofi_pmem_commit)((const void *) (uintptr_t) rma_iov[i].addr,
				   rma_iov[i].len);
	}
}

static int tcpx_process_remote_write(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret = FI_SUCCESS;

	ret = tcpx_recv_msg_data(rx_entry->ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"remote write Failed ret = %d\n",
			ret);

		tcpx_cq_report_error(rx_entry->ep->util_ep.rx_cq, rx_entry, -ret);
		tcpx_cq = container_of(rx_entry->ep->util_ep.rx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_free(tcpx_cq, rx_entry);

	} else if (rx_entry->hdr.base_hdr.flags &
		  (TCPX_DELIVERY_COMPLETE | TCPX_COMMIT_COMPLETE)) {

		if (rx_entry->hdr.base_hdr.flags & TCPX_COMMIT_COMPLETE)
			tcpx_pmem_commit(rx_entry);

		if (tcpx_prepare_rx_write_resp(rx_entry))
			rx_entry->ep->cur_rx_proc_fn = tcpx_prepare_rx_write_resp;
	} else {
		tcpx_cq_report_success(rx_entry->ep->util_ep.rx_cq, rx_entry);
		tcpx_cq = container_of(rx_entry->ep->util_ep.rx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_free(tcpx_cq, rx_entry);
	}
	return ret;
}

static int tcpx_process_remote_read(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret = FI_SUCCESS;

	ret = tcpx_recv_msg_data(rx_entry->ep);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return ret;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"msg recv Failed ret = %d\n", ret);
		tcpx_cq_report_error(rx_entry->ep->util_ep.tx_cq, rx_entry, -ret);
	} else {
		tcpx_cq_report_success(rx_entry->ep->util_ep.tx_cq, rx_entry);
	}

	slist_remove_head(&rx_entry->ep->rma_read_queue);
	tcpx_cq = container_of(rx_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_xfer_entry_free(tcpx_cq, rx_entry);
	return ret;
}

static void tcpx_copy_rma_iov_to_msg_iov(struct tcpx_xfer_entry *xfer_entry)
{
	struct ofi_rma_iov *rma_iov;
	size_t offset;
	int i;

	if (xfer_entry->hdr.base_hdr.flags & TCPX_REMOTE_CQ_DATA)
		offset = sizeof(xfer_entry->hdr.base_hdr) + sizeof(uint64_t);
	else
		offset = sizeof(xfer_entry->hdr.base_hdr);

	rma_iov = (struct ofi_rma_iov *) ((uint8_t *) &xfer_entry->hdr + offset);

	xfer_entry->iov_cnt = xfer_entry->hdr.base_hdr.rma_iov_cnt;
	for ( i = 0 ; i < xfer_entry->hdr.base_hdr.rma_iov_cnt; i++ ) {
		xfer_entry->iov[i].iov_base = (void *) rma_iov[i].addr;
		xfer_entry->iov[i].iov_len = rma_iov[i].len;
	}
}

static int tcpx_prepare_rx_remote_read_resp(struct tcpx_xfer_entry *resp_entry)
{
	struct ofi_rma_iov *rma_iov;
	int i;

	resp_entry->iov[0].iov_base = (void *) &resp_entry->hdr;
	resp_entry->iov[0].iov_len = sizeof(resp_entry->hdr.base_hdr);

	rma_iov = (struct ofi_rma_iov *) ((uint8_t *)
		  &resp_entry->hdr + sizeof(resp_entry->hdr.base_hdr));

	resp_entry->iov_cnt = 1 + resp_entry->hdr.base_hdr.rma_iov_cnt;
	resp_entry->hdr.base_hdr.size = resp_entry->iov[0].iov_len;
	for ( i = 0 ; i < resp_entry->hdr.base_hdr.rma_iov_cnt ; i++ ) {
		resp_entry->iov[i+1].iov_base =	(void *) (uintptr_t)rma_iov[i].addr;
		resp_entry->iov[i+1].iov_len = rma_iov[i].len;
		resp_entry->hdr.base_hdr.size += resp_entry->iov[i+1].iov_len;
	}

	resp_entry->hdr.base_hdr.op = ofi_op_read_rsp;
	resp_entry->hdr.base_hdr.payload_off = (uint8_t)
						sizeof(resp_entry->hdr.base_hdr);

	resp_entry->flags &= ~FI_COMPLETION;
	resp_entry->context = NULL;

	tcpx_tx_queue_insert(resp_entry->ep, resp_entry);
	resp_entry->ep->cur_rx_entry = NULL;
	return FI_SUCCESS;
}

static int tcpx_validate_rx_rma_data(struct tcpx_xfer_entry *rx_entry,
				     uint64_t access)
{
	struct ofi_mr_map *map = &rx_entry->ep->util_ep.domain->mr_map;
	struct ofi_rma_iov *rma_iov;
	size_t offset;
	int i, ret;

	if (rx_entry->hdr.base_hdr.flags & TCPX_REMOTE_CQ_DATA)
		offset = sizeof(rx_entry->hdr.base_hdr) + sizeof(uint64_t);
	else
		offset = sizeof(rx_entry->hdr.base_hdr);

	rma_iov = (struct ofi_rma_iov *) ((uint8_t *) &rx_entry->hdr + offset);

	for ( i = 0 ; i < rx_entry->hdr.base_hdr.rma_iov_cnt ; i++) {
		ret = ofi_mr_verify(map, rma_iov[i].len,
				    (uintptr_t *)&rma_iov[i].addr,
				    rma_iov[i].key, access);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			return -FI_EINVAL;
		}
	}
	return FI_SUCCESS;
}

int tcpx_op_invalid(struct tcpx_ep *tcpx_ep)
{
	return -FI_EINVAL;
}

static struct tcpx_xfer_entry *tcpx_rx_entry_alloc(struct tcpx_ep *ep)
{
	struct tcpx_xfer_entry *rx_entry;

	assert(fastlock_held(&ep->lock));
	if (slist_empty(&ep->rx_queue))
		return NULL;

	rx_entry = container_of(ep->rx_queue.head, struct tcpx_xfer_entry,
				entry);
	slist_remove_head(&ep->rx_queue);
	return rx_entry;
}

static void tcpx_rx_setup(struct tcpx_ep *ep, struct tcpx_xfer_entry *rx_entry,
			  tcpx_rx_process_fn_t process_fn)
{
	ep->cur_rx_entry = rx_entry;
	ep->cur_rx_proc_fn = process_fn;

	/* Reset to receive next message */
	ep->cur_rx_msg.hdr_len = sizeof(ep->cur_rx_msg.hdr.base_hdr);
	ep->cur_rx_msg.done_len = 0;
}

static int tcpx_handle_resp(struct tcpx_xfer_entry *tx_entry)
{
	struct tcpx_ep *ep;
	struct tcpx_cq *cq;

	ep = tx_entry->ep;
	cq = container_of(ep->util_ep.tx_cq, struct tcpx_cq, util_cq);
	tcpx_cq_report_success(ep->util_ep.tx_cq, tx_entry);

	slist_remove_head(&ep->tx_rsp_pend_queue);
	tcpx_xfer_entry_free(cq, tx_entry);
	return FI_SUCCESS;
}

int tcpx_op_msg(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *rx_entry;
	struct tcpx_cur_rx_msg *msg = &tcpx_ep->cur_rx_msg;
	size_t msg_len;
	int ret;

	if (msg->hdr.base_hdr.op_data == TCPX_OP_MSG_RESP) {
		/* response matches with original send */
		assert(!slist_empty(&tcpx_ep->tx_rsp_pend_queue));
		rx_entry = container_of(tcpx_ep->tx_rsp_pend_queue.head,
					struct tcpx_xfer_entry, entry);
		tcpx_rx_setup(tcpx_ep, rx_entry, tcpx_handle_resp);
		return FI_SUCCESS;
	}

	msg_len = (msg->hdr.base_hdr.size - msg->hdr.base_hdr.payload_off);

	if (tcpx_ep->srx_ctx) {
		rx_entry = tcpx_srx_entry_alloc(tcpx_ep->srx_ctx, tcpx_ep);
		if (!rx_entry)
			return -FI_EAGAIN;

		rx_entry->flags |= tcpx_ep->util_ep.rx_op_flags & FI_COMPLETION;
	} else {
		rx_entry = tcpx_rx_entry_alloc(tcpx_ep);
		if (!rx_entry)
			return -FI_EAGAIN;
	}

	memcpy(&rx_entry->hdr, &msg->hdr,
	       (size_t) msg->hdr.base_hdr.payload_off);
	rx_entry->ep = tcpx_ep;
	rx_entry->hdr.base_hdr.op_data = TCPX_OP_MSG_RECV;
	rx_entry->mrecv_msg_start = rx_entry->iov[0].iov_base;

	if (tcpx_dynamic_rbuf(tcpx_ep)) {
		rx_entry->flags |= TCPX_NEED_DYN_RBUF;

		if (msg->hdr.base_hdr.flags & TCPX_TAGGED) {
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

	tcpx_rx_setup(tcpx_ep, rx_entry, tcpx_process_recv);
	return FI_SUCCESS;

truncate_err:
	FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
		"posted rx buffer size is not big enough\n");
	tcpx_cq_report_error(rx_entry->ep->util_ep.rx_cq,
				rx_entry, -ret);
	tcpx_rx_entry_free(rx_entry);
	return ret;
}

int tcpx_op_read_req(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *rx_entry;
	struct tcpx_cq *tcpx_cq;
	int ret;

	/* The read request will generate a response once done,
	 * so the xfer_entry will become a transmit and returned
	 * to the tx cq buffer pool.
	 */
	tcpx_cq = container_of(tcpx_ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);

	rx_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_REMOTE_READ);
	if (!rx_entry)
		return -FI_EAGAIN;

	memcpy(&rx_entry->hdr, &tcpx_ep->cur_rx_msg.hdr,
	       (size_t) tcpx_ep->cur_rx_msg.hdr.base_hdr.payload_off);
	rx_entry->hdr.base_hdr.op_data = TCPX_OP_REMOTE_READ;
	rx_entry->ep = tcpx_ep;

	ret = tcpx_validate_rx_rma_data(rx_entry, FI_REMOTE_READ);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid rma data\n");
		tcpx_xfer_entry_free(tcpx_cq, rx_entry);
		return ret;
	}

	tcpx_rx_setup(tcpx_ep, rx_entry, tcpx_prepare_rx_remote_read_resp);
	return FI_SUCCESS;
}

int tcpx_op_write(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *rx_entry;
	struct tcpx_cq *tcpx_cq;
	int ret;

	tcpx_cq = container_of(tcpx_ep->util_ep.rx_cq,
			       struct tcpx_cq, util_cq);

	rx_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_REMOTE_WRITE);
	if (!rx_entry)
		return -FI_EAGAIN;

	rx_entry->flags = 0;
	if (tcpx_ep->cur_rx_msg.hdr.base_hdr.flags & TCPX_REMOTE_CQ_DATA)
		rx_entry->flags = (FI_COMPLETION | FI_REMOTE_WRITE);

	memcpy(&rx_entry->hdr, &tcpx_ep->cur_rx_msg.hdr,
	       (size_t) tcpx_ep->cur_rx_msg.hdr.base_hdr.payload_off);
	rx_entry->hdr.base_hdr.op_data = TCPX_OP_REMOTE_WRITE;
	rx_entry->ep = tcpx_ep;

	ret = tcpx_validate_rx_rma_data(rx_entry, FI_REMOTE_WRITE);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid rma data\n");
		tcpx_xfer_entry_free(tcpx_cq, rx_entry);
		return ret;
	}

	tcpx_copy_rma_iov_to_msg_iov(rx_entry);
	tcpx_rx_setup(tcpx_ep, rx_entry, tcpx_process_remote_write);
	return FI_SUCCESS;

}

int tcpx_op_read_rsp(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *rx_entry;
	struct slist_entry *entry;

	if (slist_empty(&tcpx_ep->rma_read_queue))
		return -FI_EINVAL;

	entry = tcpx_ep->rma_read_queue.head;
	rx_entry = container_of(entry, struct tcpx_xfer_entry,
				entry);

	memcpy(&rx_entry->hdr, &tcpx_ep->cur_rx_msg.hdr,
	       (size_t) tcpx_ep->cur_rx_msg.hdr.base_hdr.payload_off);
	rx_entry->hdr.base_hdr.op_data = TCPX_OP_READ_RSP;

	tcpx_rx_setup(tcpx_ep, rx_entry, tcpx_process_remote_read);
	return FI_SUCCESS;
}

static int tcpx_get_next_rx_hdr(struct tcpx_ep *ep)
{
	ssize_t ret;

	ret = tcpx_recv_hdr(ep);
	if (ret < 0)
		return (int) ret;

	ep->cur_rx_msg.done_len += ret;
	if (ep->cur_rx_msg.done_len >= sizeof(ep->cur_rx_msg.hdr.base_hdr)) {
		if (ep->cur_rx_msg.hdr.base_hdr.payload_off > TCPX_MAX_HDR) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
				"Payload offset is too large\n");
			return -FI_EIO;
		}
		ep->cur_rx_msg.hdr_len = (size_t) ep->cur_rx_msg.hdr.
						  base_hdr.payload_off;

		if (ep->cur_rx_msg.hdr_len > ep->cur_rx_msg.done_len) {
			ret = tcpx_recv_hdr(ep);
			if (ret < 0)
				return (int) ret;

			ep->cur_rx_msg.done_len += ret;
		}
	}

	if (ep->cur_rx_msg.done_len < ep->cur_rx_msg.hdr_len)
		return -FI_EAGAIN;

	ep->hdr_bswap(&ep->cur_rx_msg.hdr.base_hdr);
	ep->rem_rx_len = ep->cur_rx_msg.hdr.base_hdr.size -
			 ep->cur_rx_msg.hdr.base_hdr.payload_off;
	return FI_SUCCESS;
}

void tcpx_progress_rx(struct tcpx_ep *ep)
{
	int ret = 0;

	assert(fastlock_held(&ep->lock));
	do {
		if (!ep->cur_rx_entry) {
			if (ep->cur_rx_msg.done_len < ep->cur_rx_msg.hdr_len) {
				ret = tcpx_get_next_rx_hdr(ep);
				if (ret)
					break;
			}

			if (ep->cur_rx_msg.hdr.base_hdr.op >=
			    ARRAY_SIZE(ep->start_op)) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
					"Received invalid opcode\n");
				ret = -FI_EIO;
				break;
			}
			ret = ep->start_op[ep->cur_rx_msg.hdr.base_hdr.op](ep);
			if (ret)
				break;
		}
		assert(ep->cur_rx_proc_fn);
		ep->cur_rx_proc_fn(ep->cur_rx_entry);

	} while (ofi_bsock_readable(&ep->bsock));

	if (ret && !OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		tcpx_ep_disable(ep, 0);
}

int tcpx_try_func(void *util_ep)
{
	uint32_t events;
	struct util_wait_fd *wait_fd;
	struct tcpx_ep *ep;
	int ret;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);
	wait_fd = container_of(((struct util_ep *) util_ep)->tx_cq->wait,
			       struct util_wait_fd, util_wait);

	fastlock_acquire(&ep->lock);
	if (ep->cur_tx_entry && !ep->pollout_set) {
		ep->pollout_set = true;
		events = (wait_fd->util_wait.wait_obj == FI_WAIT_FD) ?
			 (OFI_EPOLL_IN | OFI_EPOLL_OUT) : (POLLIN | POLLOUT);
		goto epoll_mod;
	} else if (!ep->cur_tx_entry && ep->pollout_set) {
		ep->pollout_set = false;
		events = (wait_fd->util_wait.wait_obj == FI_WAIT_FD) ?
			 OFI_EPOLL_IN : POLLIN;
		goto epoll_mod;
	}
	fastlock_release(&ep->lock);
	return FI_SUCCESS;

epoll_mod:
	ret = (wait_fd->util_wait.wait_obj == FI_WAIT_FD) ?
	      ofi_epoll_mod(wait_fd->epoll_fd, ep->bsock.sock, events,
			    &ep->util_ep.ep_fid.fid) :
	      ofi_pollfds_mod(wait_fd->pollfds, ep->bsock.sock, events,
			      &ep->util_ep.ep_fid.fid);
	if (ret)
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			"epoll modify failed\n");
	fastlock_release(&ep->lock);
	return ret;
}

void tcpx_tx_queue_insert(struct tcpx_ep *ep,
			  struct tcpx_xfer_entry *tx_entry)
{
	struct util_wait *wait = ep->util_ep.tx_cq->wait;

	if (!ep->cur_tx_entry) {
		ep->cur_tx_entry = tx_entry;
		ep->rem_tx_len = tx_entry->hdr.base_hdr.size;
		ep->hdr_bswap(&tx_entry->hdr.base_hdr);
		tcpx_progress_tx(ep);

		if (!ep->cur_tx_entry && wait)
			wait->signal(wait);
	} else {
		slist_insert_tail(&tx_entry->entry, &ep->tx_queue);
	}
}
