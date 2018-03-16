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

void tcpx_auto_prog_mgr_close(struct auto_prog_mgr *mgr)
{
	struct tcpx_progress *progress;

	progress = container_of(mgr, struct tcpx_progress, auto_prog_mgr);
	mgr->run = 0;
	tcpx_progress_signal(progress);
	if (mgr->thread &&
	    pthread_join(mgr->thread, NULL)) {
		FI_DBG(&tcpx_prov, FI_LOG_DOMAIN,
		       "auto progress thread failed to join\n");
	}

	free(mgr->ctxs);
	fi_epoll_del(mgr->epoll_fd,mgr->signal.fd[FI_READ_FD]);
	fd_signal_free(&mgr->signal);
	fi_epoll_close(mgr->epoll_fd);
	fastlock_destroy(&mgr->lock);
}

int tcpx_progress_close(struct tcpx_progress *progress)
{
	if (progress->mode == FI_PROGRESS_AUTO) {
		tcpx_auto_prog_mgr_close(&progress->auto_prog_mgr);
	}

	util_buf_pool_destroy(progress->pe_entry_pool);
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

		if (cq->wait)
			ofi_cq_signal(&cq->cq_fid);
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

static void progress_rx_queue(struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;

	fastlock_acquire(&ep->queue_lock);
	if (dlist_empty(&ep->rx_queue)) {
		fastlock_release(&ep->queue_lock);
		return;
	}

	entry = ep->rx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_rx_pe_entry(pe_entry);
	fastlock_release(&ep->queue_lock);
}

static void progress_tx_queue(struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct dlist_entry *entry;

	fastlock_acquire(&ep->queue_lock);
	if (dlist_empty(&ep->tx_queue)) {
		fastlock_release(&ep->queue_lock);
		return;
	}

	entry = ep->tx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				entry);
	process_tx_pe_entry(pe_entry);
	fastlock_release(&ep->queue_lock);
}

void tcpx_manual_progress(struct util_ep *util_ep)
{
	struct tcpx_ep *ep;

	ep = container_of(util_ep, struct tcpx_ep, util_ep);

	progress_tx_queue(ep);
	progress_rx_queue(ep);
	return;
}

int tcpx_progress_ep_add(struct tcpx_progress *progress,
			 struct tcpx_ep *ep)
{
	void *new_ctxs;
	uint64_t new_size;
	int ret;


	ret = fid_list_insert(&progress->auto_prog_mgr.ep_list,
			      &progress->auto_prog_mgr.lock,
			      &ep->util_ep.ep_fid.fid);
	if (ret)
		return ret;

	fastlock_acquire(&progress->auto_prog_mgr.lock);
	ret = fi_epoll_add(progress->auto_prog_mgr.epoll_fd,
			   ep->conn_fd, FI_EPOLL_IN, ep);
	if (ret) {
		fastlock_release(&progress->auto_prog_mgr.lock);
		return ret;
	}

	progress->auto_prog_mgr.used++;
	if (progress->auto_prog_mgr.used > progress->auto_prog_mgr.ctxs_sz) {
		new_size = 2 * progress->auto_prog_mgr.ctxs_sz;
		new_ctxs = realloc(progress->auto_prog_mgr.ctxs, new_size *
				   sizeof(*progress->auto_prog_mgr.ctxs));
		if (!new_ctxs) {
			progress->auto_prog_mgr.ctxs = new_ctxs;
			progress->auto_prog_mgr.ctxs_sz = new_size;
		}
	}
	fastlock_release(&progress->auto_prog_mgr.lock);
	tcpx_progress_signal(progress);
	return FI_SUCCESS;
}

void tcpx_progress_ep_del(struct tcpx_progress *progress,
			 struct tcpx_ep *ep)
{
	fid_list_remove(&progress->auto_prog_mgr.ep_list,
			&progress->auto_prog_mgr.lock,
			&ep->util_ep.ep_fid.fid);

	fastlock_acquire(&progress->auto_prog_mgr.lock);
	fi_epoll_del(progress->auto_prog_mgr.epoll_fd, ep->conn_fd);
	progress->auto_prog_mgr.used--;
	fastlock_release(&progress->auto_prog_mgr.lock);
	tcpx_progress_signal(progress);
}

void tcpx_progress_signal(struct tcpx_progress *progress)
{
	if (progress->mode != FI_PROGRESS_AUTO)
		return;

	fd_signal_set(&progress->auto_prog_mgr.signal);
}

static int tcpx_auto_prog_wait_ok(struct auto_prog_mgr *mgr)
{
	struct dlist_entry *entry;
	struct tcpx_ep *ep;
	struct fid_list_entry *item;

	dlist_foreach(&mgr->ep_list,entry) {
		item = container_of(entry, struct fid_list_entry, entry);
		ep = container_of(item->fid, struct tcpx_ep,
				  util_ep.ep_fid.fid);

		if (!dlist_empty(&ep->tx_queue))
			return 0;
	}
	return 1;
}

static int tcpx_auto_prog_rx_queues(struct auto_prog_mgr *mgr)
{
	struct tcpx_ep *ep;
	int ret, i;

	ret = fi_epoll_wait(mgr->epoll_fd, mgr->ctxs,
			    MIN(mgr->used, mgr->ctxs_sz), 0);

	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"fi_epoll_wait failed\n");
		return -errno;
	}

	for (i = 0; i < ret ; i++) {
		if (mgr->ctxs[i] == NULL) {
			fd_signal_reset(&mgr->signal);
			continue;
		}
		ep = (struct tcpx_ep *) mgr->ctxs[i];
		progress_rx_queue(ep);
	}
	return FI_SUCCESS;
}

static void tcpx_auto_prog_tx_queues(struct auto_prog_mgr *mgr)
{
	struct dlist_entry *entry;
	struct tcpx_ep *ep;
	struct fid_list_entry *item;

	dlist_foreach(&mgr->ep_list,entry) {
		item = container_of(entry, struct fid_list_entry, entry);
		ep = container_of(item->fid, struct tcpx_ep,
				  util_ep.ep_fid.fid);
		progress_tx_queue(ep);
	}
}

static void *tcpx_auto_progress_thread(void *arg)
{
	struct auto_prog_mgr *mgr = (struct auto_prog_mgr *)arg;
	int ret;

	while(mgr->run) {
		fastlock_acquire(&mgr->lock);
		if (tcpx_auto_prog_wait_ok(mgr)) {
			fastlock_release(&mgr->lock);
			ret = fi_epoll_wait(mgr->epoll_fd, mgr->ctxs,
					    MIN(mgr->used, mgr->ctxs_sz), -1);
			if (ret < 0) {
				FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
					"fi_epoll_wait failed\n");
				goto err;
			}
			fastlock_acquire(&mgr->lock);
		}
		tcpx_auto_prog_tx_queues(mgr);
		tcpx_auto_prog_rx_queues(mgr);
		fastlock_release(&mgr->lock);
	}
err:
	return NULL;
}

static int tcpx_auto_prog_mgr_init(struct auto_prog_mgr *mgr)
{
	int ret ;

	ret = fastlock_init(&mgr->lock);
	if (ret)
		return ret;

	ret = fi_epoll_create(&mgr->epoll_fd);
	if (ret)
		goto err1;

	ret = fd_signal_init(&mgr->signal);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"signal init failed\n");
		goto err2;
	}

	ret = fi_epoll_add(mgr->epoll_fd,
			   mgr->signal.fd[FI_READ_FD],
			   FI_EPOLL_IN, NULL);
	if (ret)
		goto err3;

	mgr->used = 1;
	mgr->ctxs_sz = 1024;
	mgr->ctxs = calloc(mgr->ctxs_sz, sizeof(*mgr->ctxs));
	if (!mgr->ctxs) {
		ret = -FI_ENOMEM;
		goto err4;
	}
	dlist_init(&mgr->ep_list);

	mgr->run = 1;
	ret = pthread_create(&mgr->thread, 0,
			     tcpx_auto_progress_thread,
			     (void *)mgr);

	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"Failed creating auto progress thread");
		goto err5;
	}

	return FI_SUCCESS;
err5:
	free(mgr->ctxs);
err4:
	fi_epoll_del(mgr->epoll_fd, mgr->signal.fd[FI_READ_FD]);
err3:
	fd_signal_free(&mgr->signal);
err2:
	fi_epoll_close(mgr->epoll_fd);
err1:
	fastlock_destroy(&mgr->lock);
	return ret;
}

int tcpx_progress_init(struct tcpx_progress *progress)
{
	int ret;

	if (progress->mode == FI_PROGRESS_AUTO) {
		ret = tcpx_auto_prog_mgr_init(&progress->auto_prog_mgr);
		if (ret)
			return ret;
	}
	return util_buf_pool_create(&progress->pe_entry_pool,
				    sizeof(struct tcpx_pe_entry),
				    16, 0, 1024);
}
