/*
 * Copyright (c) 2018 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <sys/types.h>
#include <ofi_util.h>
#include <unistd.h>
#include <ofi_iov.h>


static ssize_t
tcpx_srx_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		 uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = msg->context;
	recv_entry->iov_cnt = msg->iov_count;
	memcpy(&recv_entry->iov[0], msg->msg_iov,
	       msg->iov_count * sizeof(*msg->msg_iov));

	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

static ssize_t
tcpx_srx_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	      fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = 1;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

static ssize_t
tcpx_srx_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	       size_t count, fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = count;
	memcpy(&recv_entry->iov[0], iov, count * sizeof(*iov));

	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

struct fi_ops_msg tcpx_srx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = tcpx_srx_recv,
	.recvv = tcpx_srx_recvv,
	.recvmsg = tcpx_srx_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};


static ssize_t
tcpx_srx_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
		  uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = msg->tag;
	recv_entry->ignore = msg->ignore;
	recv_entry->ep = (void *) (uintptr_t) msg->addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->context = msg->context;
	recv_entry->iov_cnt = msg->iov_count;
	memcpy(&recv_entry->iov[0], msg->msg_iov,
	       msg->iov_count * sizeof(*msg->msg_iov));

	slist_insert_tail(&recv_entry->entry, &srx_ctx->tag_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

static ssize_t
tcpx_srx_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	       fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = tag;
	recv_entry->ignore = ignore;
	recv_entry->ep = (void *) (uintptr_t) src_addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = 1;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	slist_insert_tail(&recv_entry->entry, &srx_ctx->tag_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

static ssize_t
tcpx_srx_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t tag,
		uint64_t ignore, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;
	ssize_t ret = FI_SUCCESS;

	srx_ctx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = ofi_buf_alloc(srx_ctx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = tag;
	recv_entry->ignore = ignore;
	recv_entry->ep = (void *) (uintptr_t) src_addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = count;
	memcpy(&recv_entry->iov[0], iov, count * sizeof(*iov));

	slist_insert_tail(&recv_entry->entry, &srx_ctx->tag_queue);
unlock:
	fastlock_release(&srx_ctx->lock);
	return ret;
}

struct fi_ops_tagged tcpx_srx_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = tcpx_srx_trecv,
	.recvv = tcpx_srx_trecvv,
	.recvmsg = tcpx_srx_trecvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

struct tcpx_xfer_entry *
tcpx_match_tag(struct tcpx_rx_ctx *srx, struct tcpx_ep *ep, uint64_t tag)
{
	struct tcpx_xfer_entry *rx_entry;
	struct slist_entry *item, *prev;

	fastlock_acquire(&srx->lock);
	slist_foreach(&srx->tag_queue, item, prev) {
		rx_entry = container_of(item, struct tcpx_xfer_entry, entry);
		if (ofi_match_tag(rx_entry->tag, rx_entry->ignore, tag)) {
			slist_remove(&srx->tag_queue, item, prev);
			fastlock_release(&srx->lock);
			return rx_entry;
		}
	}
	fastlock_release(&srx->lock);

	return NULL;
}

struct tcpx_xfer_entry *
tcpx_match_tag_addr(struct tcpx_rx_ctx *srx, struct tcpx_ep *ep, uint64_t tag)
{
	struct tcpx_xfer_entry *rx_entry;
	struct slist_entry *item, *prev;

	fastlock_acquire(&srx->lock);
	slist_foreach(&srx->tag_queue, item, prev) {
		rx_entry = container_of(item, struct tcpx_xfer_entry, entry);
		if (ofi_match_tag(rx_entry->tag, rx_entry->ignore, tag) &&
		    ofi_match_addr((uintptr_t) rx_entry->ep, (uintptr_t) ep)) {
			slist_remove(&srx->tag_queue, item, prev);
			fastlock_release(&srx->lock);
			return rx_entry;
		}
	}
	fastlock_release(&srx->lock);

	return NULL;
}
