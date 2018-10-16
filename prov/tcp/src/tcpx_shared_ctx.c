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

static inline struct tcpx_xfer_entry *
tcpx_srx_ctx_rx_entry_alloc(struct tcpx_rx_ctx *srx_ctx)
{
	struct tcpx_xfer_entry *recv_entry;

	fastlock_acquire(&srx_ctx->lock);
	recv_entry = util_buf_alloc(srx_ctx->buf_pool);
	if (recv_entry)
		recv_entry->done_len = 0;

	fastlock_release(&srx_ctx->lock);
	return recv_entry;
}

static ssize_t tcpx_srx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
				uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;

	srx_ctx = container_of(ep, struct tcpx_rx_ctx, rx_fid);
	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	recv_entry = tcpx_srx_ctx_rx_entry_alloc(srx_ctx);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = msg->iov_count;
	memcpy(&recv_entry->msg_data.iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof(struct iovec));

	recv_entry->flags = flags | FI_MSG | FI_RECV;
	recv_entry->context = msg->context;

	fastlock_acquire(&srx_ctx->lock);
	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
	fastlock_release(&srx_ctx->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_srx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			     fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;

	srx_ctx = container_of(ep, struct tcpx_rx_ctx, rx_fid);

	recv_entry = tcpx_srx_ctx_rx_entry_alloc(srx_ctx);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = 1;
	recv_entry->msg_data.iov[0].iov_base = buf;
	recv_entry->msg_data.iov[0].iov_len = len;

	recv_entry->flags = FI_MSG | FI_RECV;
	recv_entry->context = context;

	fastlock_acquire(&srx_ctx->lock);
	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
	fastlock_release(&srx_ctx->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_srx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			      size_t count, fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx_ctx;

	srx_ctx = container_of(ep, struct tcpx_rx_ctx, rx_fid);
	assert(count <= TCPX_IOV_LIMIT);

	recv_entry = tcpx_srx_ctx_rx_entry_alloc(srx_ctx);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = count;
	memcpy(recv_entry->msg_data.iov, iov, count * sizeof(*iov));

	recv_entry->flags = FI_MSG | FI_RECV;
	recv_entry->context = context;

	fastlock_acquire(&srx_ctx->lock);
	slist_insert_tail(&recv_entry->entry, &srx_ctx->rx_queue);
	fastlock_release(&srx_ctx->lock);
	return FI_SUCCESS;
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
