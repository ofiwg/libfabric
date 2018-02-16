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

int tcpx_progress_close(struct tcpx_progress *progress)
{
	util_buf_pool_destroy(progress->posted_rx_pool);
	util_buf_pool_destroy(progress->pe_entry_pool);
	fastlock_destroy(&progress->posted_rx_pool_lock);
	return FI_SUCCESS;
}

struct tcpx_pe_entry *pe_entry_alloc(struct tcpx_progress *progress)
{
	struct tcpx_pe_entry *pe_entry;

	pe_entry = util_buf_alloc(progress->pe_entry_pool);
	if (!pe_entry) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"failed to get buffer\n");
		return NULL;
	}
	memset(pe_entry, 0, sizeof(*pe_entry));
	return pe_entry;
}

static void report_pe_entry_completion(struct tcpx_pe_entry *pe_entry, int err)
{
	struct fi_cq_err_entry err_entry;
	struct tcpx_ep *ep = pe_entry->ep;
	struct util_cq *cq = NULL;
	struct util_cntr *cntr = NULL;

	if (pe_entry->flags & TCPX_NO_COMPLETION) {
		return;
	}

	switch (pe_entry->msg_hdr.op_data) {
	case TCPX_OP_MSG_SEND:
		cq = ep->util_ep.tx_cq;
		cntr = ep->util_ep.tx_cntr;
		break;
	case TCPX_OP_MSG_RECV:
		cq = ep->util_ep.rx_cq;
		cntr = ep->util_ep.rx_cntr;
		break;
	default:

		return;
	}

	if (cq && err) {
		err_entry.op_context = pe_entry->context;
		err_entry.flags = pe_entry->flags;
		err_entry.len = 0;
		err_entry.buf = NULL;
		err_entry.data = pe_entry->msg_hdr.data;
		err_entry.tag = 0;
		err_entry.olen = 0;
		err_entry.err = err;
		err_entry.prov_errno = errno;
		err_entry.err_data = NULL;
		err_entry.err_data_size = 0;

		ofi_cq_write_error(cq, &err_entry);
	} else if (cq) {
		ofi_cq_write(cq, pe_entry->context,
			     pe_entry->flags, 0, NULL,
			     pe_entry->msg_hdr.data, 0);
	}

	if (cntr && err) {
		cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
	}else if (cntr) {
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
	}
}

void pe_entry_release(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_domain *domain;

	domain = container_of(pe_entry->ep->util_ep.domain,
			      struct tcpx_domain, util_domain);

	memset(&pe_entry->msg_hdr, 0, sizeof(pe_entry->msg_hdr));
	dlist_remove(&pe_entry->entry);
	memset(pe_entry, 0, sizeof(*pe_entry));
	util_buf_release(domain->progress.pe_entry_pool, pe_entry);
}

static void process_tx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	uint64_t total_len = ntohll(pe_entry->msg_hdr.size);
	int ret;

	ret = tcpx_send_msg(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg send failed\n");
		report_pe_entry_completion(pe_entry, ret);
		pe_entry_release(pe_entry);
		return;
	}

	if (pe_entry->done_len == total_len) {
		report_pe_entry_completion(pe_entry, 0);
		pe_entry_release(pe_entry);
	}
}

void posted_rx_find(struct tcpx_pe_entry *pe_entry)
{
	struct dlist_entry *entry;
	struct tcpx_posted_rx *posted_rx;
	struct tcpx_domain *domain;
	struct tcpx_ep *ep;
	size_t data_len = 0;

	ep = pe_entry->ep;
	domain = container_of(ep->util_ep.domain,
			      struct tcpx_domain,
			      util_domain);
	fastlock_acquire(&ep->posted_rx_list_lock);
	dlist_foreach(&ep->posted_rx_list, entry) {
		posted_rx = container_of(entry, struct tcpx_posted_rx, entry);
		data_len = ofi_total_iov_len(posted_rx->msg_data.iov,
					     posted_rx->msg_data.iov_cnt);
		if (data_len >= ntohll(pe_entry->msg_hdr.size)-sizeof(pe_entry->msg_hdr)) {
			dlist_remove(entry);
			goto copy_to_pe_entry;
		}
	}
	fastlock_release(&ep->posted_rx_list_lock);
	return;

copy_to_pe_entry:
	pe_entry->flags = posted_rx->flags;
	pe_entry->context = posted_rx->context;
	pe_entry->msg_data.iov_cnt = posted_rx->msg_data.iov_cnt;
	memcpy(pe_entry->msg_data.iov, posted_rx->msg_data.iov,
	       TCPX_IOV_LIMIT*sizeof(pe_entry->msg_data.iov[0]));
	util_buf_release(domain->progress.posted_rx_pool, posted_rx);
	fastlock_release(&ep->posted_rx_list_lock);
}

static void process_rx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	int ret;

	ret = tcpx_recv_msg(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);
		report_pe_entry_completion(pe_entry, ret);
		pe_entry_release(pe_entry);
		return;
	}

	if (pe_entry->done_len &&
	    pe_entry->done_len == ntohll(pe_entry->msg_hdr.size)) {
		report_pe_entry_completion(pe_entry, 0);
		pe_entry_release(pe_entry);
	}
}

static void process_pe_lists(struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;

	if (dlist_empty(&ep->rx_queue))
		goto tx_pe_list;

	entry = ep->rx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_rx_pe_entry(pe_entry);

tx_pe_list:
	if (dlist_empty(&ep->tx_queue))
		return ;

	entry = ep->tx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_tx_pe_entry(pe_entry);
}

static void process_rx_requests(struct tcpx_ep *ep)

{
	struct tcpx_pe_entry *pe_entry;
	struct tcpx_domain *domain;

	domain = container_of(ep->util_ep.domain,
			      struct tcpx_domain,
			      util_domain);

	if (!dlist_empty(&ep->rx_queue))
		return;

	pe_entry = pe_entry_alloc(&domain->progress);
	if (!pe_entry) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
			"failed to allocate pe entry");
		return ;
	}
	pe_entry->ep = ep;
	dlist_insert_tail(&pe_entry->entry, &ep->rx_queue);
}

void tcpx_progress(struct util_ep *util_ep)
{
	struct tcpx_ep *ep;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);
	process_rx_requests(ep);
	process_pe_lists(ep);
	return;
}

int tcpx_progress_init(struct tcpx_progress *progress)
{
	int ret;

	ret = util_buf_pool_create(&progress->pe_entry_pool,
				   sizeof(struct tcpx_pe_entry),
				   16, 0, 1024);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"failed to create buffer pool\n");
		return ret;
	}

	fastlock_init(&progress->posted_rx_pool_lock);
	ret = util_buf_pool_create(&progress->posted_rx_pool,
				   sizeof(struct tcpx_posted_rx),
				   16, 0, 1024);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"failed to create buffer pool\n");
		goto err1;
	}
	return FI_SUCCESS;
err1:
	fastlock_destroy(&progress->posted_rx_pool_lock);
	util_buf_pool_destroy(progress->pe_entry_pool);
	return ret;
}
