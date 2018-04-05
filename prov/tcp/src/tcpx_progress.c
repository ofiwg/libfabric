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

static void process_tx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	uint64_t total_len = ntohll(pe_entry->msg_hdr.size);
	int ret;

	ret = tcpx_send_msg(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg send failed\n");
		tcpx_cq_report_completion(pe_entry->ep->util_ep.tx_cq,
					  pe_entry, ret);
		tcpx_pe_entry_release(pe_entry);
		return;
	}

	if (pe_entry->done_len == total_len) {
		tcpx_cq_report_completion(pe_entry->ep->util_ep.tx_cq,
					  pe_entry, 0);
		tcpx_pe_entry_release(pe_entry);
	}
}

static void process_rx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	int ret;

	ret = tcpx_recv_msg(pe_entry);
	if (OFI_SOCK_TRY_SND_RCV_AGAIN(-ret))
		return;

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN, "msg recv Failed ret = %d\n", ret);
		tcpx_cq_report_completion(pe_entry->ep->util_ep.rx_cq,
					  pe_entry, ret);
		tcpx_pe_entry_release(pe_entry);
		return;
	}

	if (pe_entry->done_len &&
	    pe_entry->done_len == ntohll(pe_entry->msg_hdr.size)) {
		tcpx_cq_report_completion(pe_entry->ep->util_ep.rx_cq,
					  pe_entry, 0);
		tcpx_pe_entry_release(pe_entry);
	}
}

static void process_rx_queue(struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;

	if (dlist_empty(&ep->rx_queue))
		return;

	entry = ep->rx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_rx_pe_entry(pe_entry);
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
	process_rx_queue(ep);
	process_tx_queue(ep);
	fastlock_release(&ep->queue_lock);
	return;
}
