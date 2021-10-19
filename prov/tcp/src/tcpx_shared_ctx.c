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
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = msg->context;
	recv_entry->iov_cnt = msg->iov_count;
	memcpy(&recv_entry->iov[0], msg->msg_iov,
	       msg->iov_count * sizeof(*msg->msg_iov));

	slist_insert_tail(&recv_entry->entry, &srx->rx_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static ssize_t
tcpx_srx_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	      fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = 1;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	slist_insert_tail(&recv_entry->entry, &srx->rx_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static ssize_t
tcpx_srx_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	       size_t count, fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->context = context;
	recv_entry->iov_cnt = count;
	memcpy(&recv_entry->iov[0], iov, count * sizeof(*iov));

	slist_insert_tail(&recv_entry->entry, &srx->rx_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static struct fi_ops_msg tcpx_srx_msg_ops = {
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
tcpx_srx_peek(struct tcpx_rx_ctx *srx, const struct fi_msg_tagged *msg,
	      uint64_t flags)
{
	struct fi_cq_err_entry err_entry = {0};

	err_entry.op_context = msg->context;
	err_entry.flags = FI_RECV | FI_TAGGED;
	err_entry.tag = msg->tag;
	err_entry.err = FI_ENOMSG;

	ofi_cq_write_error(&srx->cq->util_cq, &err_entry);
	return FI_SUCCESS;
}

static ssize_t
tcpx_srx_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
		  uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	if (flags & FI_PEEK)
		return tcpx_srx_peek(srx, msg, flags);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
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

	slist_insert_tail(&recv_entry->entry, &srx->tag_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static ssize_t
tcpx_srx_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	       fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
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

	slist_insert_tail(&recv_entry->entry, &srx->tag_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static ssize_t
tcpx_srx_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t tag,
		uint64_t ignore, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_rx_ctx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct tcpx_rx_ctx, rx_fid);
	assert(count <= TCPX_IOV_LIMIT);

	fastlock_acquire(&srx->lock);
	recv_entry = ofi_buf_alloc(srx->buf_pool);
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

	slist_insert_tail(&recv_entry->entry, &srx->tag_queue);
unlock:
	fastlock_release(&srx->lock);
	return ret;
}

static struct fi_ops_tagged tcpx_srx_tag_ops = {
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

static struct tcpx_xfer_entry *
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

static struct tcpx_xfer_entry *
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

int tcpx_srx_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcpx_rx_ctx *srx;

	if (flags != FI_RECV || bfid->fclass != FI_CLASS_CQ)
		return -FI_EINVAL;

	srx = container_of(fid, struct tcpx_rx_ctx, rx_fid.fid);
	srx->cq = container_of(bfid, struct tcpx_cq, util_cq.cq_fid.fid);
	ofi_atomic_inc32(&srx->cq->util_cq.ref);
	return FI_SUCCESS;
}

static int tcpx_srx_close(struct fid *fid)
{
	struct tcpx_rx_ctx *srx;
	struct slist_entry *entry;
	struct tcpx_xfer_entry *xfer_entry;

	srx = container_of(fid, struct tcpx_rx_ctx, rx_fid.fid);

	while (!slist_empty(&srx->rx_queue)) {
		entry = slist_remove_head(&srx->rx_queue);
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		if (srx->cq) {
			tcpx_cq_report_error(&srx->cq->util_cq, xfer_entry,
					      FI_ECANCELED);
		}
		ofi_buf_free(xfer_entry);
	}

	while (!slist_empty(&srx->tag_queue)) {
		entry = slist_remove_head(&srx->tag_queue);
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		if (srx->cq) {
			tcpx_cq_report_error(&srx->cq->util_cq, xfer_entry,
					      FI_ECANCELED);
		}
		ofi_buf_free(xfer_entry);
	}

	if (srx->cq)
		ofi_atomic_dec32(&srx->cq->util_cq.ref);
	ofi_bufpool_destroy(srx->buf_pool);
	fastlock_destroy(&srx->lock);
	free(srx);
	return FI_SUCCESS;
}

static struct fi_ops fi_ops_srx = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_srx_close,
	.bind = tcpx_srx_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int tcpx_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		     struct fid_ep **rx_ep, void *context)
{
	struct tcpx_rx_ctx *srx;
	int ret = FI_SUCCESS;

	srx = calloc(1, sizeof(*srx));
	if (!srx)
		return -FI_ENOMEM;

	srx->rx_fid.fid.fclass = FI_CLASS_SRX_CTX;
	srx->rx_fid.fid.context = context;
	srx->rx_fid.fid.ops = &fi_ops_srx;

	srx->rx_fid.msg = &tcpx_srx_msg_ops;
	srx->rx_fid.tagged = &tcpx_srx_tag_ops;
	slist_init(&srx->rx_queue);
	slist_init(&srx->tag_queue);

	ret = fastlock_init(&srx->lock);
	if (ret)
		goto err1;

	ret = ofi_bufpool_create(&srx->buf_pool,
				 sizeof(struct tcpx_xfer_entry),
				 16, attr->size, 1024, 0);
	if (ret)
		goto err2;

	srx->match_tag_rx = (attr->caps & FI_DIRECTED_RECV) ?
			    tcpx_match_tag_addr : tcpx_match_tag;
	srx->op_flags = attr->op_flags;
	*rx_ep = &srx->rx_fid;
	return FI_SUCCESS;
err2:
	fastlock_destroy(&srx->lock);
err1:
	free(srx);
	return ret;
}
