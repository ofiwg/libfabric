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

static void tcpx_report_error(struct tcpx_ep *tcpx_ep, int err)
{
	struct fi_eq_err_entry err_entry = {0};

	err_entry.fid = &tcpx_ep->util_ep.ep_fid.fid;
	err_entry.context = tcpx_ep->util_ep.ep_fid.fid.context;
	err_entry.err = -err;

	fi_eq_write(&tcpx_ep->util_ep.eq->eq_fid, FI_NOTIFY,
		    &err_entry, sizeof(err_entry), UTIL_FLAG_ERROR);
}

int tcpx_ep_shutdown_report(struct tcpx_ep *ep, fid_t fid)
{
	struct fi_eq_cm_entry cm_entry = {0};
	ssize_t len;

	if (ep->cm_state == TCPX_EP_SHUTDOWN)
		return FI_SUCCESS;

	ep->cm_state = TCPX_EP_SHUTDOWN;

	cm_entry.fid = fid;
	len =  fi_eq_write(&ep->util_ep.eq->eq_fid, FI_SHUTDOWN,
			   &cm_entry, sizeof(cm_entry), 0);
	if (len < 0)
		return (int) len;

	return FI_SUCCESS;
}

void process_tx_entry(struct tcpx_xfer_entry *tx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_send_msg(tx_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg send failed\n");

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(tx_entry->ep,
					&tx_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(tx_entry->ep->util_ep.tx_cq,
				  tx_entry, ret);
	slist_remove_head(&tx_entry->ep->tx_queue);
	tcpx_cq = container_of(tx_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_xfer_entry_release(tcpx_cq, tx_entry);
}

static void process_rx_entry(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(rx_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(rx_entry->ep,
					&rx_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(rx_entry->ep->util_ep.rx_cq,
				  rx_entry, ret);
	slist_remove_head(&rx_entry->ep->rx_queue);
	tcpx_cq = container_of(rx_entry->ep->util_ep.rx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_xfer_entry_release(tcpx_cq, rx_entry);
}

static void process_rx_remote_write_entry(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(rx_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(rx_entry->ep,
					&rx_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(rx_entry->ep->util_ep.rx_cq,
				  rx_entry, ret);
	tcpx_cq = container_of(rx_entry->ep->util_ep.rx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_xfer_entry_release(tcpx_cq, rx_entry);

}

static void process_rx_read_entry(struct tcpx_xfer_entry *rx_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(rx_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(rx_entry->ep,
					&rx_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(rx_entry->ep->util_ep.tx_cq,
				  rx_entry, ret);
	slist_remove_head(&rx_entry->ep->rma_read_queue);
	tcpx_cq = container_of(rx_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_xfer_entry_release(tcpx_cq, rx_entry);
}

static void tcpx_copy_rma_iov_to_msg_iov(struct tcpx_xfer_entry *xfer_entry)
{
	int i;

	xfer_entry->msg_data.iov_cnt = xfer_entry->msg_hdr.rma_iov_cnt;
	for ( i = 0 ; i < xfer_entry->msg_hdr.rma_iov_cnt ; i++ ) {
		xfer_entry->msg_data.iov[i].iov_base =
			(void *) xfer_entry->msg_hdr.rma_iov[i].addr;
		xfer_entry->msg_data.iov[i].iov_len =
			xfer_entry->msg_hdr.rma_iov[i].len;
	}
}

static void tcpx_prepare_rx_remote_read_resp(struct tcpx_xfer_entry *resp_entry)
{
	int i;

	resp_entry->msg_data.iov[0].iov_base = (void *) &resp_entry->msg_hdr;
	resp_entry->msg_data.iov[0].iov_len = sizeof(resp_entry->msg_hdr);
	resp_entry->msg_data.iov_cnt = 1 + resp_entry->msg_hdr.rma_iov_cnt;

	resp_entry->msg_hdr.hdr.size = resp_entry->msg_data.iov[0].iov_len;
	for ( i = 0 ; i < resp_entry->msg_hdr.rma_iov_cnt ; i++ ) {
		resp_entry->msg_data.iov[i+1].iov_base =
			(void *) (uintptr_t)resp_entry->msg_hdr.rma_iov[i].addr;
		resp_entry->msg_data.iov[i+1].iov_len =
			resp_entry->msg_hdr.rma_iov[i].len;
		resp_entry->msg_hdr.hdr.size +=
			resp_entry->msg_data.iov[i+1].iov_len;
	}

	resp_entry->msg_hdr.hdr.op = ofi_op_read_rsp;
	resp_entry->msg_hdr.hdr.size =
		htonll(resp_entry->msg_hdr.hdr.size);

	resp_entry->flags |= TCPX_NO_COMPLETION;
	resp_entry->context = NULL;
	resp_entry->done_len = 0;

	tcpx_tx_queue_insert(resp_entry->ep, resp_entry);
}

static int tcpx_validate_rx_rma_data(struct tcpx_xfer_entry *rx_entry,
				     uint64_t access)
{
	struct ofi_mr_map *map = &rx_entry->ep->util_ep.domain->mr_map;
	struct fi_rma_iov *rma_iov = rx_entry->msg_hdr.rma_iov;
	int i, ret;

	for ( i = 0 ; i < rx_entry->msg_hdr.rma_iov_cnt ; i++) {
		ret = ofi_mr_map_verify(map,
					(uintptr_t *)&rma_iov[i].addr,
					rma_iov[i].len,
					rma_iov[i].key,
					access, NULL);
		if (ret) {
			FI_DBG(&tcpx_prov, FI_LOG_EP_DATA,
			       "invalid rma iov received\n");
			return -FI_EINVAL;
		}
	}
	return FI_SUCCESS;
}

static int tcpx_get_rx_entry(struct tcpx_rx_detect *rx_detect,
			     struct tcpx_xfer_entry **new_rx_entry)
{
	struct tcpx_xfer_entry *rx_entry;
	struct slist_entry *entry;
	struct tcpx_ep *tcpx_ep;
	struct tcpx_cq *tcpx_cq;
	int ret;

	tcpx_ep = container_of(rx_detect, struct tcpx_ep, rx_detect);
	tcpx_cq = container_of(tcpx_ep->util_ep.rx_cq, struct tcpx_cq,
			       util_cq);

	switch (rx_detect->hdr.hdr.op) {
	case ofi_op_msg:
		if (slist_empty(&tcpx_ep->rx_queue))
			return -FI_EAGAIN;

		entry = tcpx_ep->rx_queue.head;
		rx_entry = container_of(entry, struct tcpx_xfer_entry,
					entry);

		rx_entry->msg_hdr = rx_detect->hdr;
		rx_entry->msg_hdr.hdr.op_data = TCPX_OP_MSG_RECV;
		rx_entry->done_len = sizeof(rx_detect->hdr);

		if (ntohl(rx_detect->hdr.hdr.flags) & OFI_REMOTE_CQ_DATA)
			rx_entry->flags |= FI_REMOTE_CQ_DATA;

		ret = ofi_truncate_iov(rx_entry->msg_data.iov,
				       &rx_entry->msg_data.iov_cnt,
				       (ntohll(rx_entry->msg_hdr.hdr.size) -
					sizeof(rx_entry->msg_hdr)));
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"posted rx buffer size is not big enough\n");
			tcpx_cq_report_completion(rx_entry->ep->util_ep.rx_cq,
						  rx_entry, ret);
			slist_remove_head(&rx_entry->ep->rx_queue);
			tcpx_xfer_entry_release(tcpx_cq, rx_entry);
			return ret;
		}
		break;
	case ofi_op_read_req:
		rx_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_REMOTE_READ);
		if (!rx_entry)
			return -FI_EAGAIN;

		rx_entry->msg_hdr = rx_detect->hdr;
		rx_entry->msg_hdr.hdr.op_data =	TCPX_OP_REMOTE_READ;
		rx_entry->ep = tcpx_ep;
		rx_entry->done_len = sizeof(rx_detect->hdr);

		ret = tcpx_validate_rx_rma_data(rx_entry, FI_REMOTE_READ);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"invalid rma data\n");
			tcpx_xfer_entry_release(tcpx_cq, rx_entry);
			return ret;
		}
		break;
	case ofi_op_write:
		rx_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_REMOTE_WRITE);
		if (!rx_entry)
			return -FI_EAGAIN;

		if (ntohl(rx_detect->hdr.hdr.flags) & OFI_REMOTE_CQ_DATA) {
			rx_entry->flags |= (FI_REMOTE_CQ_DATA |
					    FI_REMOTE_WRITE);

		} else {
			rx_entry->flags = TCPX_NO_COMPLETION;
		}

		rx_entry->msg_hdr = rx_detect->hdr;
		rx_entry->msg_hdr.hdr.op_data = TCPX_OP_REMOTE_WRITE;
		rx_entry->ep = tcpx_ep;
		rx_entry->done_len = sizeof(rx_detect->hdr);

		ret = tcpx_validate_rx_rma_data(rx_entry, FI_REMOTE_WRITE);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"invalid rma data\n");
			tcpx_xfer_entry_release(tcpx_cq, rx_entry);
			return ret;
		}

		tcpx_copy_rma_iov_to_msg_iov(rx_entry);
		break;
	case ofi_op_read_rsp:
		if (slist_empty(&tcpx_ep->rma_read_queue))
			return -FI_EINVAL;

		entry = tcpx_ep->rma_read_queue.head;
		rx_entry = container_of(entry, struct tcpx_xfer_entry,
					entry);

		rx_entry->msg_hdr = rx_detect->hdr;
		rx_entry->msg_hdr.hdr.op_data = TCPX_OP_READ_RSP;
		rx_entry->done_len = sizeof(rx_detect->hdr);
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid op in hdr\n");
		return -FI_EINVAL;
	}
	rx_detect->done_len = 0;
	*new_rx_entry = rx_entry;
	return FI_SUCCESS;
}

static void tcpx_process_rx_msg(struct tcpx_ep *ep)
{
	int ret;

	if (!ep->cur_rx_entry) {
		ret = tcpx_recv_hdr(ep->conn_fd, &ep->rx_detect);
		if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
			return;

		if (ret)
			goto err1;

		ret = tcpx_get_rx_entry(&ep->rx_detect,
				      &ep->cur_rx_entry);
		if (ret == -FI_EAGAIN)
			return;

		if (ret)
			goto err2;
	}

	switch(ep->cur_rx_entry->msg_hdr.hdr.op_data){
	case TCPX_OP_MSG_RECV:
		process_rx_entry(ep->cur_rx_entry);
		break;
	case TCPX_OP_REMOTE_WRITE:
		process_rx_remote_write_entry(ep->cur_rx_entry);
		break;
	case TCPX_OP_READ_RSP:
		process_rx_read_entry(ep->cur_rx_entry);
		break;
 	case TCPX_OP_REMOTE_READ:
		tcpx_prepare_rx_remote_read_resp(ep->cur_rx_entry);
		ep->cur_rx_entry = NULL;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid op type\n");
		return;
	}
	return;
err2:
	tcpx_report_error(ep, ret);
	return;
err1:
	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(ep, &ep->util_ep.ep_fid.fid);
}

static void process_tx_queue(struct tcpx_ep *ep)
{
	struct tcpx_xfer_entry *tx_entry;
	struct slist_entry *entry;

	if (slist_empty(&ep->tx_queue))
		return;

	entry = ep->tx_queue.head;
	tx_entry = container_of(entry, struct tcpx_xfer_entry,
				entry);
	process_tx_entry(tx_entry);
}

void tcpx_ep_progress(struct tcpx_ep *ep)
{
	tcpx_process_rx_msg(ep);
	process_tx_queue(ep);
}

void tcpx_progress(struct util_ep *util_ep)
{
	struct tcpx_ep *ep;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);
	fastlock_acquire(&ep->lock);
	ep->progress_func(ep);
	fastlock_release(&ep->lock);
	return;
}

static int tcpx_try_func(void *util_ep)
{
	uint32_t events;
	struct util_wait_fd *wait_fd;
	struct tcpx_ep *ep;
	int ret;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);
	wait_fd = container_of(((struct util_ep *)util_ep)->rx_cq->wait,
			       struct util_wait_fd, util_wait);

	fastlock_acquire(&ep->lock);
	if (slist_empty(&ep->tx_queue) && !ep->send_ready_monitor) {
		ep->send_ready_monitor = true;
		events = FI_EPOLL_IN | FI_EPOLL_OUT;
		goto epoll_mod;
	} else if (!slist_empty(&ep->tx_queue) && ep->send_ready_monitor) {
		ep->send_ready_monitor = false;
		events = FI_EPOLL_IN;
		goto epoll_mod;
	}
	fastlock_release(&ep->lock);
	return FI_SUCCESS;

epoll_mod:
	ret = fi_epoll_mod(wait_fd->epoll_fd, ep->conn_fd, events, NULL);
	if (ret)
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			"invalid op type\n");
	fastlock_release(&ep->lock);
	return ret;
}

int tcpx_cq_wait_ep_add(struct tcpx_ep *ep)
{
	if (!ep->util_ep.rx_cq->wait)
		return FI_SUCCESS;

	return ofi_wait_fd_add(ep->util_ep.rx_cq->wait,
			       ep->conn_fd, FI_EPOLL_IN,
			       tcpx_try_func, (void *)&ep->util_ep,
			       NULL);
}

void tcpx_cq_wait_ep_del(struct tcpx_ep *ep)
{
	fastlock_acquire(&ep->lock);
	if (ep->cm_state == TCPX_EP_CONNECTING) {
		goto out;
	}

	if (ep->util_ep.rx_cq->wait) {
		ofi_wait_fd_del(ep->util_ep.rx_cq->wait, ep->conn_fd);
	}
out:
	fastlock_release(&ep->lock);
}

void tcpx_tx_queue_insert(struct tcpx_ep *tcpx_ep,
			  struct tcpx_xfer_entry *tx_entry)
{
	int empty;
	struct util_wait *wait = tcpx_ep->util_ep.tx_cq->wait;

	empty = slist_empty(&tcpx_ep->tx_queue);
	slist_insert_tail(&tx_entry->entry, &tcpx_ep->tx_queue);

	if (empty) {
		process_tx_entry(tx_entry);

		if (!slist_empty(&tcpx_ep->tx_queue) && wait)
			wait->signal(wait);
	}
}
