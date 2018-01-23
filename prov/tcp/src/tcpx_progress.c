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

#include <prov.h>
#include "tcpx.h"
#include <poll.h>

#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <fi_util.h>

static int tcpx_progress_table_clean(struct tcpx_progress *progress)
{
	int i;
	struct dlist_entry *entry;
	struct tcpx_pe_entry *pe_entry;

	for (i = 0; i < TCPX_PE_MAX_ENTRIES; i++) {
		ofi_rbfree(&progress->pe_entry_table[i].comm_buf);
	}

	while(!dlist_empty(&progress->pool_list)) {
		entry = progress->pool_list.next;
		dlist_remove(entry);
		pe_entry = container_of(entry, struct tcpx_pe_entry, entry);
		ofi_rbfree(&pe_entry->comm_buf);
		util_buf_release(progress->pe_entry_pool, pe_entry);
	}

	return FI_SUCCESS;
}

int tcpx_progress_close(struct tcpx_domain *domain)
{
	struct tcpx_progress *progress = &domain->progress;

	if (!dlist_empty(&progress->ep_list)) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"All EPs are not removed from progress\n");
		return -FI_EBUSY;
	}

	progress->do_progress = 0;
	tcpx_progress_signal(progress);
	if (progress->progress_thread &&
	    pthread_join(progress->progress_thread, NULL)) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"progress thread failed to join\n");
	}

	fd_signal_free(&progress->signal);
	fastlock_destroy(&progress->signal_lock);
	fi_epoll_close(progress->epoll_set);
	tcpx_progress_table_clean(progress);
	util_buf_pool_destroy(progress->posted_rx_pool);
	util_buf_pool_destroy(progress->pe_entry_pool);
	pthread_mutex_destroy(&progress->ep_list_lock);
	return FI_SUCCESS;
}

void tcpx_progress_signal(struct tcpx_progress *progress)
{
	fastlock_acquire(&progress->signal_lock);
	fd_signal_set(&progress->signal);
	fastlock_release(&progress->signal_lock);
}

static struct tcpx_pe_entry *get_new_pe_entry(struct tcpx_progress *progress)
{
	struct dlist_entry *entry;
	struct tcpx_pe_entry *pe_entry;

	if (dlist_empty(&progress->free_list)) {
		pe_entry = util_buf_alloc(progress->pe_entry_pool);
		if (!pe_entry) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"failed to get buffer\n");
			return NULL;
		}
		memset(pe_entry, 0, sizeof(*pe_entry));
		pe_entry->is_pool_entry = 1;
		if (ofi_rbinit(&pe_entry->comm_buf, TCPX_PE_COMM_BUFF_SZ))
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"failed to init comm-cache\n");
		pe_entry->cache_sz = TCPX_PE_COMM_BUFF_SZ;
		dlist_insert_tail(&pe_entry->entry, &progress->pool_list);
	} else {
		entry = progress->free_list.next;
		dlist_remove(entry);
		dlist_insert_tail(entry, &progress->busy_list);
		pe_entry = container_of(entry, struct tcpx_pe_entry, entry);
		assert(ofi_rbempty(&pe_entry->comm_buf));
	}
	return pe_entry;
}

static inline ssize_t tcpx_pe_send_field(struct tcpx_pe_entry *pe_entry,
					 void *field, size_t field_len,
					 size_t field_offset)
{
	size_t field_rem_len, field_done_len;
	uint8_t *buf;
	int ret;

	if (pe_entry->done_len >= (field_offset + field_len))
		return 0;

	field_done_len = pe_entry->done_len - field_offset;
	field_rem_len = field_len - field_done_len;
	buf = (uint8_t *) field + field_done_len;

	ret = tcpx_comm_send(pe_entry, buf, field_rem_len);
	if (ret <= 0)
		return -1;

	pe_entry->done_len += ret;
	return (ret == field_rem_len) ? 0 : -1;
}

static inline ssize_t tcpx_pe_recv_field(struct tcpx_pe_entry *pe_entry,
					 void *field, size_t field_len,
					 size_t field_offset)
{
	size_t field_rem_len, field_done_len;
	uint8_t *buf;
	int ret;

	if (pe_entry->done_len >= (field_offset + field_len))
		return 0;

	field_done_len = pe_entry->done_len - field_offset;
	field_rem_len = field_len - field_done_len;
	buf = (uint8_t *) field + field_done_len;

	ret = tcpx_comm_recv(pe_entry, buf, field_rem_len);
	if (ret <= 0)
		return -1;

	pe_entry->done_len += ret;
	return (ret == field_rem_len) ? 0 : -1;
}

/* to do : needs to be verified */
static void report_pe_entry_completion(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_ep *ep = pe_entry->ep;

	if (pe_entry->flags & TCPX_NO_COMPLETION) {
		return;
	}

	switch (pe_entry->msg_hdr.op_data) {
	case TCPX_OP_MSG_SEND:
		if (ep->util_ep.tx_cq) {
			ofi_cq_write(ep->util_ep.tx_cq, pe_entry->context,
				     pe_entry->flags, 0, NULL,
				     pe_entry->msg_hdr.data, 0);
		}

		if (ep->util_ep.tx_cntr) {
			ep->util_ep.tx_cntr->cntr_fid.ops->add(&ep->util_ep.tx_cntr->cntr_fid, 1);
		}
		break;
	case TCPX_OP_MSG_RECV:
		if (ep->util_ep.rx_cq) {
			ofi_cq_write(ep->util_ep.rx_cq, pe_entry->context,
				     pe_entry->flags, 0, NULL,
				     pe_entry->msg_hdr.data, 0);
		}

		if (ep->util_ep.rx_cntr) {
			ep->util_ep.rx_cntr->cntr_fid.ops->add(&ep->util_ep.rx_cntr->cntr_fid, 1);
		}
		break;
	}
}

static void release_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	struct tcpx_domain *domain;

	domain = container_of(pe_entry->ep->util_ep.domain,
			      struct tcpx_domain, util_domain);
	assert(ofi_rbempty(&pe_entry->comm_buf));

	pe_entry->state = TCPX_XFER_IDLE;
	memset(&pe_entry->msg_hdr, 0, sizeof(pe_entry->msg_hdr));
	dlist_remove(&pe_entry->ctx_entry);
	ofi_rbreset(&pe_entry->comm_buf);
	pe_entry->ep = NULL;
	pe_entry->flags = 0;
	pe_entry->context = NULL;
	memset(pe_entry->inject, 0, sizeof(TCPX_MAX_INJECT_SZ));
	pe_entry->data_len = 0;
	pe_entry->done_len = 0;
	pe_entry->iov_cnt = 0;

	if (pe_entry->is_pool_entry) {
		ofi_rbfree(&pe_entry->comm_buf);
		dlist_remove(&pe_entry->entry);
		util_buf_release(domain->progress.pe_entry_pool, pe_entry);
		return;
	}
	dlist_remove(&pe_entry->entry);
	dlist_insert_tail(&pe_entry->entry, &domain->progress.free_list);
}

void tcpx_progress_posted_rx_cleanup(struct tcpx_ep *ep,
				    struct tcpx_progress *progress)
{
	struct dlist_entry *entry;
	struct tcpx_posted_rx *posted_rx;

	pthread_mutex_lock(&ep->posted_rx_list_lock);
	while (!dlist_empty(&ep->posted_rx_list)) {
		entry =  ep->posted_rx_list.next;
		posted_rx = container_of(entry, struct tcpx_posted_rx, entry);
		dlist_remove(entry);
		util_buf_release(progress->posted_rx_pool, posted_rx);
	}
	pthread_mutex_unlock(&ep->posted_rx_list_lock);
}

void tcpx_progress_pe_entry_cleanup(struct tcpx_ep *ep,
				    struct tcpx_progress *progress)
{
	struct dlist_entry *entry;
	struct tcpx_pe_entry *pe_entry;

	pthread_mutex_lock(&progress->ep_list_lock);
	while (!dlist_empty(&ep->rx_queue)) {
		entry = ep->rx_queue.next;
		pe_entry = container_of(entry, struct tcpx_pe_entry, ctx_entry);
		dlist_remove(entry);
		release_pe_entry(pe_entry);
	}

	while (!dlist_empty(&ep->rx_queue)) {
		entry = ep->rx_queue.next;
		pe_entry = container_of(entry, struct tcpx_pe_entry, ctx_entry);
		dlist_remove(entry);
		release_pe_entry(pe_entry);
	}
	pthread_mutex_unlock(&progress->ep_list_lock);
}

static void process_tx_pe_entry(struct tcpx_pe_entry *pe_entry)
{

	size_t field_offset;
	int i;

	if (pe_entry->state == TCPX_XFER_STARTED) {
		field_offset = 0;
		if (0 == tcpx_pe_send_field(pe_entry, &pe_entry->msg_hdr,
					    sizeof(pe_entry->msg_hdr),
					    field_offset)) {
			pe_entry->state = TCPX_XFER_HDR_SENT;
		}
	}

	if (pe_entry->state == TCPX_XFER_HDR_SENT) {
		field_offset = sizeof(pe_entry->msg_hdr);
		for (i = 0 ; i < pe_entry->iov_cnt ; i++) {
			if (0 != tcpx_pe_send_field(pe_entry,
						    (char *) (uintptr_t)pe_entry->iov[i].iov.addr,
						    pe_entry->iov[i].iov.len,
						    field_offset)) {
				break;
			}
			field_offset += pe_entry->iov[i].iov.len;
		}

		if (pe_entry->done_len == pe_entry->data_len + sizeof(pe_entry->msg_hdr))
			pe_entry->state = TCPX_XFER_FLUSH_COMM_BUF;
	}

	if (pe_entry->state == TCPX_XFER_FLUSH_COMM_BUF) {
		if (!ofi_rbempty(&pe_entry->comm_buf)) {
			tcpx_comm_flush(pe_entry);
		} else {
			pe_entry->state = TCPX_XFER_COMPLETE;
		}
	}

	if (pe_entry->state == TCPX_XFER_COMPLETE) {
		report_pe_entry_completion(pe_entry);
		release_pe_entry(pe_entry);
	}
}

static void find_matching_posted_rx(struct tcpx_ep *ep,
				   struct tcpx_pe_entry *pe_entry)
{
	struct dlist_entry *entry;
	struct tcpx_posted_rx *posted_rx;
	struct tcpx_domain *domain;

	domain = container_of(ep->util_ep.domain,
			      struct tcpx_domain,
			      util_domain);
	pthread_mutex_lock(&ep->posted_rx_list_lock);
	dlist_foreach(&ep->posted_rx_list, entry) {
		posted_rx = container_of(entry, struct tcpx_posted_rx, entry);
		if (posted_rx->data_len >= (pe_entry->msg_hdr.size-sizeof(pe_entry->msg_hdr))) {
			dlist_remove(entry);
			goto copy_to_pe_entry;
		}
	}
	pthread_mutex_unlock(&ep->posted_rx_list_lock);
	return;

copy_to_pe_entry:
	pe_entry->flags = posted_rx->flags;
	pe_entry->context = posted_rx->context;
	pe_entry->iov_cnt = posted_rx->iov_cnt;
	memcpy(pe_entry->iov, posted_rx->iov,
	       TCPX_IOV_LIMIT*sizeof(pe_entry->iov[0]));
	util_buf_release(domain->progress.posted_rx_pool, posted_rx);
	pthread_mutex_unlock(&ep->posted_rx_list_lock);
}

static void process_rx_pe_entry(struct tcpx_pe_entry *pe_entry)
{
	size_t field_offset;
	int i;

	if (pe_entry->state == TCPX_XFER_STARTED) {
		field_offset = 0;
		pe_entry->data_len = sizeof(pe_entry->msg_hdr);
		if (0 == tcpx_pe_recv_field(pe_entry, &pe_entry->msg_hdr,
					    sizeof(pe_entry->msg_hdr),
					    field_offset)) {
			pe_entry->msg_hdr.op_data = TCPX_OP_MSG_RECV;
			pe_entry->msg_hdr.flags = ntohl(pe_entry->msg_hdr.flags);
			pe_entry->msg_hdr.size = ntohl(pe_entry->msg_hdr.size);
			pe_entry->msg_hdr.data = ntohll(pe_entry->msg_hdr.data);
			pe_entry->msg_hdr.remote_idx = ntohll(pe_entry->msg_hdr.remote_idx);

			pe_entry->data_len = pe_entry->msg_hdr.size;
			find_matching_posted_rx(pe_entry->ep, pe_entry);
			pe_entry->state = TCPX_XFER_HDR_RECVD;
		}
	}
	if (pe_entry->state == TCPX_XFER_HDR_RECVD) {
		field_offset = sizeof(pe_entry->msg_hdr);
		for (i = 0 ; i < pe_entry->iov_cnt ; i++) {
			if (0 != tcpx_pe_recv_field(pe_entry,
						    (char *) (uintptr_t)pe_entry->iov[i].iov.addr,
						    pe_entry->iov[i].iov.len, field_offset)) {
				break;
			}
			field_offset += pe_entry->iov[i].iov.len;
		}

		if (pe_entry->done_len == pe_entry->msg_hdr.size) {
			pe_entry->state = TCPX_XFER_COMPLETE;
		}
	}

	if (pe_entry->state == TCPX_XFER_COMPLETE) {
		report_pe_entry_completion(pe_entry);
		ofi_rbreset(&pe_entry->comm_buf);
		release_pe_entry(pe_entry);
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
				ctx_entry);
	process_rx_pe_entry(pe_entry);

tx_pe_list:
	if (dlist_empty(&ep->tx_queue))
		return ;

	entry = ep->tx_queue.next;
	pe_entry = container_of(entry, struct tcpx_pe_entry,
				ctx_entry);
	process_tx_pe_entry(pe_entry);
}


static void process_rx_requests(struct tcpx_progress *progress)
{
	struct tcpx_pe_entry *pe_entry;
	struct tcpx_ep *ep;
	void *ep_contexts[TCPX_MAX_EPOLL_EVENTS];
	int i, num_fds;

	do {
		num_fds = fi_epoll_wait(progress->epoll_set,
					ep_contexts,
					TCPX_MAX_EPOLL_EVENTS,0);
		if (num_fds < 0)
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"epoll failed: %d\n", num_fds);
		if (num_fds < 1)
			return;

		for (i = 0 ; i < num_fds ; i++) {

			/* ignore signal fd ; already handled */
			if (!ep_contexts[i])
				continue;

			ep = (struct tcpx_ep *) ep_contexts[i];

			/* skip if there is pending pe entry already */
			if (!dlist_empty(&ep->rx_queue))
				continue;

			pe_entry = get_new_pe_entry(progress);
			if (!pe_entry) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
					"failed to allocate pe entry");
				return ;
			}

			pe_entry->state = TCPX_XFER_STARTED;
			pe_entry->ep = ep;

			dlist_insert_tail(&pe_entry->ctx_entry, &ep->rx_queue);
		}
	} while (num_fds == TCPX_MAX_EPOLL_EVENTS);
}

static void process_ep_tx_requests(struct tcpx_progress *progress,
				   struct tcpx_ep *ep)
{
	struct tcpx_pe_entry *pe_entry;
	struct tcpx_domain *tcpx_domain;
	int i;

	tcpx_domain = container_of(ep->util_ep.domain,
				   struct tcpx_domain,
				   util_domain);

	fastlock_acquire(&ep->rb_lock);
	while (!ofi_rbempty(&ep->rb)) {
		pe_entry = get_new_pe_entry(&tcpx_domain->progress);
		if (!pe_entry) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_DATA,
				"failed to allocate pe entry");
			goto out;
		}

		pe_entry->state = TCPX_XFER_STARTED;
		pe_entry->msg_hdr.version = OFI_CTRL_VERSION;
		pe_entry->msg_hdr.op = ofi_op_msg;

		ofi_rbread(&ep->rb, &pe_entry->msg_hdr.op_data,
			   sizeof(pe_entry->msg_hdr.op_data));
		ofi_rbread(&ep->rb, &pe_entry->iov_cnt,
			   sizeof(pe_entry->iov_cnt));
		ofi_rbread(&ep->rb, &pe_entry->flags,
			   sizeof(pe_entry->flags));
		ofi_rbread(&ep->rb, &pe_entry->context,
			   sizeof(pe_entry->context));
		ofi_rbread(&ep->rb, &pe_entry->data_len,
			   sizeof(pe_entry->data_len));

		if (pe_entry->flags & FI_REMOTE_CQ_DATA) {
			pe_entry->msg_hdr.flags |= OFI_REMOTE_CQ_DATA;
			ofi_rbread(&ep->rb, &pe_entry->msg_hdr.data,
				   sizeof(pe_entry->msg_hdr.data));
		}

		if (pe_entry->flags & FI_INJECT) {
			pe_entry->msg_hdr.flags |= FI_INJECT;
			ofi_rbread(&ep->rb, pe_entry->inject,
				   pe_entry->data_len);

			pe_entry->iov_cnt = 1;
			pe_entry->iov[0].iov.addr = (uintptr_t)pe_entry->inject;
			pe_entry->iov[0].iov.len = pe_entry->data_len;

		} else {
			for (i = 0; i < pe_entry->iov_cnt; i++) {
				ofi_rbread(&ep->rb, &pe_entry->iov[i],
					   sizeof(pe_entry->iov[i]));
			}
		}
		pe_entry->ep = ep;
		pe_entry->msg_hdr.flags = htonl(pe_entry->msg_hdr.flags);
		pe_entry->msg_hdr.data = htonll(pe_entry->msg_hdr.data);
		pe_entry->msg_hdr.size = htonl(pe_entry->data_len +
					       sizeof(pe_entry->msg_hdr));

		dlist_insert_tail(&pe_entry->ctx_entry, &ep->tx_queue);
	}
out:
	fastlock_release(&ep->rb_lock);
}

static int tcpx_progress_wait(struct tcpx_progress *progress)
{
	int ret;
	void *ep_context[1];

	ret = fi_epoll_wait(progress->epoll_set,
			    ep_context,1,-1);
	if (ret < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"Poll failed\n");
		return -errno;
	}

	fastlock_acquire(&progress->signal_lock);
	fd_signal_reset(&progress->signal);
	fastlock_release(&progress->signal_lock);

	return FI_SUCCESS;
}

static int tcpx_progress_wait_ok(struct tcpx_progress *progress)
{
	struct dlist_entry *ep_entry;
	struct tcpx_ep *ep;
	int ret;

	pthread_mutex_lock(&progress->ep_list_lock);
	if (dlist_empty(&progress->ep_list)) {
		ret = 1;
		goto out;
	}

	dlist_foreach(&progress->ep_list, ep_entry) {
		ep = container_of(ep_entry, struct tcpx_ep, ep_entry);
		if (!ofi_rbempty(&ep->rb) ||
		    !dlist_empty(&ep->tx_queue) ||
		    !dlist_empty(&ep->rx_queue)) {
			ret = 0;
			goto out;
		}
	}
	ret = 1;
out:
	pthread_mutex_unlock(&progress->ep_list_lock);
	return ret;
}

void *tcpx_progress_thread(void *data)
{
	struct tcpx_progress *progress;
	struct dlist_entry  *ep_entry;
	struct tcpx_ep *ep;

	progress = (struct tcpx_progress *) data;
	while (progress->do_progress) {
		if (tcpx_progress_wait_ok(progress)) {
			tcpx_progress_wait(progress);
		}

		process_rx_requests(progress);

		pthread_mutex_lock(&progress->ep_list_lock);

		if (dlist_empty(&progress->ep_list)) {
			pthread_mutex_unlock(&progress->ep_list_lock);
			continue;
		}

		dlist_foreach(&progress->ep_list, ep_entry) {
			ep = container_of(ep_entry, struct tcpx_ep,
					  ep_entry);

			process_ep_tx_requests(progress, ep);
			process_pe_lists(ep);
		}
		pthread_mutex_unlock(&progress->ep_list_lock);
	}
	return NULL;
}
int tcpx_progress_ep_remove(struct tcpx_ep *ep, struct tcpx_progress *progress)
{
	int ret;

	ret = fi_epoll_del(progress->epoll_set, ep->conn_fd);
	if (ret)
		return ret;

	pthread_mutex_lock(&progress->ep_list_lock);
	dlist_remove(&ep->ep_entry);
	pthread_mutex_unlock(&progress->ep_list_lock);
	tcpx_progress_signal(progress);
	return FI_SUCCESS;
}

int tcpx_progress_ep_add(struct tcpx_ep *ep, struct tcpx_progress *progress)
{
	int ret;

	ret = fi_epoll_add(progress->epoll_set, ep->conn_fd, ep);
	if (ret)
		return ret;

	pthread_mutex_lock(&progress->ep_list_lock);
	dlist_insert_tail(&ep->ep_entry, &progress->ep_list);
	pthread_mutex_unlock(&progress->ep_list_lock);

	tcpx_progress_signal(progress);
	return FI_SUCCESS;
}

static int tcpx_progress_table_init(struct tcpx_progress *progress)
{
	int i;

	memset(&progress->pe_entry_table, 0,
	       sizeof(struct tcpx_pe_entry) * TCPX_PE_MAX_ENTRIES);

	dlist_init(&progress->free_list);
	dlist_init(&progress->busy_list);
	dlist_init(&progress->pool_list);

	for (i = 0; i < TCPX_PE_MAX_ENTRIES; i++) {
		dlist_insert_head(&progress->pe_entry_table[i].entry, &progress->free_list);
		progress->pe_entry_table[i].cache_sz = TCPX_PE_COMM_BUFF_SZ;
		if (ofi_rbinit(&progress->pe_entry_table[i].comm_buf, TCPX_PE_COMM_BUFF_SZ)) {
			FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
				"failed to init comm-cache\n");
			goto err;
		}
	}
	FI_DBG(&tcpx_prov, FI_LOG_DOMAIN,
	       "Progress entry table init: OK\n");
	return FI_SUCCESS;
err:
	while(i--) {
		ofi_rbfree(&progress->pe_entry_table[i].comm_buf);
	}
	return FI_ENOMEM;
}

int tcpx_progress_init(struct tcpx_domain *domain,
		       struct tcpx_progress *progress)
{
	int ret;

	progress->domain = domain;
	ret = tcpx_progress_table_init(progress);
	if (ret)
		return ret;

	dlist_init(&progress->ep_list);
	pthread_mutex_init(&progress->ep_list_lock, NULL);

	progress->pe_entry_pool = util_buf_pool_create(sizeof(struct tcpx_pe_entry),
						 16, 0, 1024);
	if (!progress->pe_entry_pool) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"failed to create buffer pool\n");
	       	ret = -FI_ENOMEM;
		goto err1;
	}
	fastlock_init(&progress->posted_rx_pool_lock);
	progress->posted_rx_pool = util_buf_pool_create(sizeof(struct tcpx_posted_rx),
						       16, 0, 1024);
	if (!progress->posted_rx_pool) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,
			"failed to create buffer pool\n");
		ret = -FI_ENOMEM;
		goto err2;
	}

	fastlock_init(&progress->signal_lock);
	ret = fd_signal_init(&progress->signal);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"signal init failed\n");
		goto err3;
	}

	ret = fi_epoll_create(&progress->epoll_set);
	if (ret < 0)
	{
                FI_WARN(&tcpx_prov, FI_LOG_DOMAIN,"failed to create epoll set\n");
		ret = -errno;
                goto err4;
	}

	ret = fi_epoll_add(progress->epoll_set,
			   progress->signal.fd[FI_READ_FD], NULL);
	if (ret)
		goto err5;

	progress->do_progress = 1;
	if (pthread_create(&progress->progress_thread, NULL,
			   tcpx_progress_thread, (void *)progress)) {
		goto err5;
	}
	return FI_SUCCESS;
err5:
	fi_epoll_close(progress->epoll_set);
err4:
	fd_signal_free(&progress->signal);
	fastlock_destroy(&progress->signal_lock);
err3:
	util_buf_pool_destroy(progress->posted_rx_pool);
	fastlock_destroy(&progress->posted_rx_pool_lock);
err2:
	util_buf_pool_destroy(progress->pe_entry_pool);
err1:
	pthread_mutex_destroy(&progress->ep_list_lock);
	tcpx_progress_table_clean(progress);
	return ret;
}
