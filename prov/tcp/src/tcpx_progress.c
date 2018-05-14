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

int tcpx_ep_shutdown_report(struct tcpx_ep *ep, fid_t fid)
{
	struct fi_eq_cm_entry cm_entry = {0};
	ssize_t len;

	fastlock_acquire(&ep->cm_state_lock);
	if (ep->cm_state == TCPX_EP_SHUTDOWN) {
		fastlock_release(&ep->cm_state_lock);
		return FI_SUCCESS;
	}
	ep->cm_state = TCPX_EP_SHUTDOWN;
	fastlock_release(&ep->cm_state_lock);

	cm_entry.fid = fid;
	len =  fi_eq_write(&ep->util_ep.eq->eq_fid, FI_SHUTDOWN,
			   &cm_entry, sizeof(cm_entry), 0);
	if (len < 0)
		return (int) len;

	return FI_SUCCESS;
}

void process_tx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_send_msg(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg send failed\n");

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(pe_entry->ep,
					&pe_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(pe_entry->ep->util_ep.tx_cq,
				  pe_entry, ret);
	dlist_remove(&pe_entry->entry);
	tcpx_cq = container_of(pe_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_pe_entry_release(tcpx_cq, pe_entry);
}

static void process_rx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(pe_entry->ep,
					&pe_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(pe_entry->ep->util_ep.rx_cq,
				  pe_entry, ret);
	dlist_remove(&pe_entry->entry);
	tcpx_cq = container_of(pe_entry->ep->util_ep.rx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_pe_entry_release(tcpx_cq, pe_entry);
}

static void process_remote_write_entry(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(pe_entry->ep,
					&pe_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq = container_of(pe_entry->ep->util_ep.rx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_pe_entry_release(tcpx_cq, pe_entry);

}

static void process_read_entry(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_cq *tcpx_cq;
	int ret;

	ret = tcpx_recv_msg_data(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (!ret)
		goto done;

	FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);

	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(pe_entry->ep,
					&pe_entry->ep->util_ep.ep_fid.fid);
done:
	tcpx_cq_report_completion(pe_entry->ep->util_ep.tx_cq,
				  pe_entry, ret);
	dlist_remove(&pe_entry->entry);
	tcpx_cq = container_of(pe_entry->ep->util_ep.tx_cq,
			       struct tcpx_cq, util_cq);
	tcpx_pe_entry_release(tcpx_cq, pe_entry);
}

static void tcpx_copy_rma_iov_to_msg_iov(struct tcpx_pe_entry *pe_entry)
{
	int i;

	pe_entry->msg_data.iov_cnt = pe_entry->msg_hdr.rma_iov_cnt;
	for ( i = 0 ; i < pe_entry->msg_hdr.rma_iov_cnt ; i++ ) {
		pe_entry->msg_data.iov[i].iov_base =
			(void *) pe_entry->msg_hdr.rma_iov[i].addr;
		pe_entry->msg_data.iov[i].iov_len =
			pe_entry->msg_hdr.rma_iov[i].len;
	}
}

static void tcpx_prepare_remote_read_resp(struct tcpx_pe_entry *resp_entry)
{
	int i;

	resp_entry->msg_data.iov[0].iov_base = (void *) &resp_entry->msg_hdr;
	resp_entry->msg_data.iov[0].iov_len = sizeof(resp_entry->msg_hdr);
	resp_entry->msg_data.iov_cnt = 1 + resp_entry->msg_hdr.rma_iov_cnt;

	resp_entry->msg_hdr.hdr.size = resp_entry->msg_data.iov[0].iov_len;
	for ( i = 0 ; i < resp_entry->msg_hdr.rma_iov_cnt ; i++ ) {
		resp_entry->msg_data.iov[i+1].iov_base =
			(void *) resp_entry->msg_hdr.rma_iov[i].addr;
		resp_entry->msg_data.iov[i+1].iov_len =
			resp_entry->msg_hdr.rma_iov[i].len;
		resp_entry->msg_hdr.hdr.size +=
			resp_entry->msg_data.iov[i+1].iov_len;
	}

	resp_entry->msg_hdr.hdr.version = OFI_CTRL_VERSION;
	resp_entry->msg_hdr.hdr.op = ofi_op_read_rsp;
	resp_entry->msg_hdr.hdr.op_data = TCPX_OP_REMOTE_READ_RSP;
	resp_entry->msg_hdr.hdr.size =
		htonll(resp_entry->msg_hdr.hdr.size);

	resp_entry->flags |= TCPX_NO_COMPLETION;
	resp_entry->context = NULL;
	resp_entry->done_len = 0;

	dlist_insert_tail(&resp_entry->entry, &resp_entry->ep->tx_queue);
}

static int tcpx_validate_rx_rma_data(struct tcpx_pe_entry *pe_entry,
				     uint64_t access)
{
	struct ofi_mr_map *map = &pe_entry->ep->util_ep.domain->mr_map;
	struct fi_rma_iov *rma_iov = pe_entry->msg_hdr.rma_iov;
	int i, ret;

	for ( i = 0 ; i < pe_entry->msg_hdr.rma_iov_cnt ; i++) {
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

static int tcpx_match_read_rsp(struct dlist_entry *entry, const void *arg)
{
	struct tcpx_pe_entry *pe_entry;
	struct tcpx_rx_detect *rx_detect = (struct tcpx_rx_detect *) arg;

	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	return (pe_entry->msg_hdr.hdr.remote_idx ==
		ntohll(rx_detect->hdr.hdr.remote_idx))?1:0;
}

static int tcpx_get_rx_pe_entry(struct tcpx_rx_detect *rx_detect,
				struct tcpx_pe_entry **new_pe_entry)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;
	struct tcpx_ep *tcpx_ep;
	struct tcpx_cq *tcpx_cq;
	int ret;

	tcpx_ep = container_of(rx_detect, struct tcpx_ep, rx_detect);
	tcpx_cq = container_of(tcpx_ep->util_ep.rx_cq, struct tcpx_cq,
			       util_cq);

	switch (rx_detect->hdr.hdr.op) {
	case ofi_op_msg:
		if (dlist_empty(&tcpx_ep->rx_queue))
			return -FI_EAGAIN;

		entry = tcpx_ep->rx_queue.next;
		pe_entry = container_of(entry, struct tcpx_pe_entry,
					entry);

		pe_entry->msg_hdr = rx_detect->hdr;
		pe_entry->msg_hdr.hdr.op_data = TCPX_OP_MSG_RECV;
		pe_entry->done_len = sizeof(rx_detect->hdr);

		ret = ofi_truncate_iov(pe_entry->msg_data.iov,
				       &pe_entry->msg_data.iov_cnt,
				       (ntohll(pe_entry->msg_hdr.hdr.size) -
					sizeof(pe_entry->msg_hdr)));
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"posted rx buffer size is not big enough\n");
			tcpx_cq_report_completion(pe_entry->ep->util_ep.rx_cq,
						  pe_entry, ret);
			dlist_remove(&pe_entry->entry);
			tcpx_pe_entry_release(tcpx_cq, pe_entry);
			return ret;
		}
		break;
	case ofi_op_read_req:
		pe_entry = tcpx_pe_entry_alloc(tcpx_cq);
		if (!pe_entry)
			return -FI_EAGAIN;

		pe_entry->msg_hdr = rx_detect->hdr;
		pe_entry->msg_hdr.hdr.op_data =	TCPX_OP_REMOTE_READ_REQ;
		pe_entry->ep = tcpx_ep;
		pe_entry->flags = TCPX_NO_COMPLETION;
		pe_entry->done_len = sizeof(rx_detect->hdr);

		ret = tcpx_validate_rx_rma_data(pe_entry, FI_REMOTE_READ);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"invalid rma data\n");
			tcpx_pe_entry_release(tcpx_cq, pe_entry);
			return ret;
		}
		break;
	case ofi_op_write:
		pe_entry = tcpx_pe_entry_alloc(tcpx_cq);
		if (!pe_entry)
			return -FI_EAGAIN;

		pe_entry->msg_hdr = rx_detect->hdr;
		pe_entry->msg_hdr.hdr.op_data = TCPX_OP_REMOTE_WRITE;
		pe_entry->ep = tcpx_ep;
		pe_entry->flags = TCPX_NO_COMPLETION;
		pe_entry->done_len = sizeof(rx_detect->hdr);

		ret = tcpx_validate_rx_rma_data(pe_entry, FI_REMOTE_WRITE);
		if (ret) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"invalid rma data\n");
			tcpx_pe_entry_release(tcpx_cq, pe_entry);
			return ret;
		}

		tcpx_copy_rma_iov_to_msg_iov(pe_entry);
		break;
	case ofi_op_read_rsp:
		if (dlist_empty(&tcpx_ep->rma_list.list))
			return -FI_EINVAL;

		entry = dlist_find_first_match(&tcpx_ep->rma_list.list,
					       tcpx_match_read_rsp,
					       rx_detect);
		if (!entry) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"no matching RMA read for this response\n");
			return -FI_EINVAL;
		}

		pe_entry = container_of(entry, struct tcpx_pe_entry,
					entry);

		pe_entry->msg_hdr = rx_detect->hdr;
		pe_entry->msg_hdr.hdr.op_data = TCPX_OP_READ;
		pe_entry->done_len = sizeof(rx_detect->hdr);
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid op in hdr\n");
		return -FI_EINVAL;
	}
	rx_detect->done_len = 0;
	*new_pe_entry = pe_entry;
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
			goto err;

		if (tcpx_get_rx_pe_entry(&ep->rx_detect,
					 &ep->cur_rx_entry))
			return;
	}

	switch(ep->cur_rx_entry->msg_hdr.hdr.op_data){
	case TCPX_OP_MSG_RECV:
		process_rx_pe_entry(ep->cur_rx_entry);
		break;
	case TCPX_OP_REMOTE_WRITE:
		process_remote_write_entry(ep->cur_rx_entry);
		break;
	case TCPX_OP_READ:
		process_read_entry(ep->cur_rx_entry);
		break;
 	case TCPX_OP_REMOTE_READ_REQ:
		tcpx_prepare_remote_read_resp(ep->cur_rx_entry);
		ep->cur_rx_entry = NULL;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"invalid op type\n");
		return;
	}
	return;
err:
	if (ret == -FI_ENOTCONN)
		tcpx_ep_shutdown_report(ep, &ep->util_ep.ep_fid.fid);
}

static void process_tx_queue(struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;

	if (dlist_empty(&ep->tx_queue))
		return;

	entry = ep->tx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_tx_pe_entry(pe_entry);
}

void tcpx_progress(struct util_ep *util_ep)
{
	struct tcpx_ep *ep;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);
	fastlock_acquire(&ep->queue_lock);
	tcpx_process_rx_msg(ep);
	process_tx_queue(ep);
	fastlock_release(&ep->queue_lock);
	return;
}

static int tcpx_try_func(void *util_ep)
{
	/* nothing to do here. When endpoints
	   have incoming data, cq drives progress*/
	return FI_SUCCESS;
}

int tcpx_progress_ep_add(struct tcpx_ep *ep)
{
	if (!ep->util_ep.rx_cq->wait)
		return FI_SUCCESS;

	return ofi_wait_fd_add(ep->util_ep.rx_cq->wait,
			       ep->conn_fd, tcpx_try_func,
			       (void *)&ep->util_ep, NULL);
}

void tcpx_progress_ep_del(struct tcpx_ep *ep)
{
	fastlock_acquire(&ep->cm_state_lock);
	if (ep->cm_state == TCPX_EP_CONNECTING) {
		goto out;
	}

	if (ep->util_ep.rx_cq->wait) {
		ofi_wait_fd_del(ep->util_ep.rx_cq->wait, ep->conn_fd);
	}
out:
	fastlock_release(&ep->cm_state_lock);
}
