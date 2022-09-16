/*
 * Copyright (c) 2018-2022 Intel Corporation. All rights reserved.
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
#include "xnet.h"

#include <sys/types.h>
#include <ofi_util.h>
#include <unistd.h>
#include <ofi_iov.h>


static struct xnet_xfer_entry *
xnet_match_tag(struct xnet_srx *srx, struct xnet_ep *ep, uint64_t tag);


/* The rdm ep calls directly through to the srx calls, so we need to use the
 * progress active_lock for protection.
 */

static void
xnet_srx_msg(struct xnet_srx *srx, struct xnet_xfer_entry *recv_entry)
{
	struct xnet_progress *progress;
	struct xnet_ep *ep;

	progress = xnet_srx2_progress(srx);
	assert(xnet_progress_locked(progress));
	/* See comment with xnet_srx_tag(). */
	slist_insert_tail(&recv_entry->entry, &srx->rx_queue);

	if (!dlist_empty(&progress->unexp_msg_list)) {
		ep = container_of(progress->unexp_msg_list.next,
				  struct xnet_ep, unexp_entry);
		xnet_progress_rx(ep);
	}
}

static ssize_t
xnet_srx_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		 uint64_t flags)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);
	assert(msg->iov_count <= XNET_IOV_LIMIT);
	assert(!(flags & FI_MULTI_RECV) || msg->iov_count == 1);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->ctrl_flags = flags & FI_MULTI_RECV;
	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = msg->context;
	recv_entry->iov_cnt = msg->iov_count;
	if (msg->iov_count) {
		recv_entry->user_buf = msg->msg_iov[0].iov_base;
		memcpy(&recv_entry->iov[0], msg->msg_iov,
		       msg->iov_count * sizeof(*msg->msg_iov));
	}

	xnet_srx_msg(srx, recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static ssize_t
xnet_srx_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	      fi_addr_t src_addr, void *context)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->ctrl_flags = srx->op_flags & FI_MULTI_RECV;
	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = context;
	recv_entry->iov_cnt = 1;
	recv_entry->user_buf = buf;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	xnet_srx_msg(srx, recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static ssize_t
xnet_srx_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	       size_t count, fi_addr_t src_addr, void *context)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);
	assert(count <= XNET_IOV_LIMIT);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->ctrl_flags = srx->op_flags & FI_MULTI_RECV;
	recv_entry->cq_flags = FI_MSG | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = context;
	recv_entry->iov_cnt = count;
	if (count) {
		recv_entry->user_buf = iov[0].iov_base;
		memcpy(&recv_entry->iov[0], iov, count * sizeof(*iov));
	}

	xnet_srx_msg(srx, recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static struct fi_ops_msg xnet_srx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = xnet_srx_recv,
	.recvv = xnet_srx_recvv,
	.recvmsg = xnet_srx_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static void
xnet_srx_peek(struct xnet_srx *srx, const struct fi_msg_tagged *msg,
	      uint64_t flags)
{
	struct fi_cq_err_entry err_entry = {0};

	assert(xnet_progress_locked(xnet_srx2_progress(srx)));
	err_entry.op_context = msg->context;
	err_entry.flags = FI_RECV | FI_TAGGED;
	err_entry.tag = msg->tag;
	err_entry.err = FI_ENOMSG;

	ofi_cq_write_error(&srx->cq->util_cq, &err_entry);
}

/* It's possible that an endpoint may be waiting for the message being
 * posted (i.e. it has an unexpected message).  If so, kick off progress
 * to handle it immediately.
 *
 * Note that we go through the full flow of queuing the request and calling
 * the progress function to handle the message.  This is needed as there
 * may be another message stored on the buffered socket.  We need to process
 * any buffered data after completing this one to prevent hangs.
 */
static ssize_t
xnet_srx_tag(struct xnet_srx *srx, struct xnet_xfer_entry *recv_entry)
{
	struct xnet_progress *progress;
	struct xnet_ep *ep;
	struct slist *queue;

	progress = xnet_srx2_progress(srx);
	assert(xnet_progress_locked(progress));
	assert(srx->rdm);

	/* Always set and bump the tag_seq_no to help debugging */
	recv_entry->tag_seq_no = srx->tag_seq_no++;

	if ((srx->match_tag_rx == xnet_match_tag) ||
	    (recv_entry->src_addr == FI_ADDR_UNSPEC)) {
		slist_insert_tail(&recv_entry->entry, &srx->tag_queue);

		/* The message could match any endpoint waiting. */
		if (!dlist_empty(&progress->unexp_tag_list))
			xnet_progress_unexp(progress, &progress->unexp_tag_list);
	} else {
		queue = ofi_array_at(&srx->src_tag_queues, recv_entry->src_addr);
		if (!queue)
			return -FI_EAGAIN;

		slist_insert_tail(&recv_entry->entry, queue);

		ep = xnet_get_ep(srx->rdm, recv_entry->src_addr);
		if (ep) {
			xnet_active_ep(ep);
			if (xnet_has_unexp(ep)) {
				assert(!dlist_empty(&ep->unexp_entry));
				xnet_progress_rx(ep);
			}
		}
	}

	return 0;
}

static ssize_t
xnet_srx_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
		  uint64_t flags)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);
	assert(msg->iov_count <= XNET_IOV_LIMIT);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	if (flags & FI_PEEK) {
		xnet_srx_peek(srx, msg, flags);
		ret = 0;
		goto unlock;
	}

	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = msg->tag;
	recv_entry->ignore = msg->ignore;
	recv_entry->src_addr = msg->addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = msg->context;

	recv_entry->iov_cnt = msg->iov_count;
	if (msg->iov_count) {
		recv_entry->user_buf = msg->msg_iov[0].iov_base;
		memcpy(&recv_entry->iov[0], msg->msg_iov,
		       msg->iov_count * sizeof(*msg->msg_iov));
	}

	ret = xnet_srx_tag(srx, recv_entry);
	if (ret)
		ofi_buf_free(recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static ssize_t
xnet_srx_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	       fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = tag;
	recv_entry->ignore = ignore;
	recv_entry->src_addr = src_addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = context;
	recv_entry->user_buf = buf;
	recv_entry->iov_cnt = 1;
	recv_entry->iov[0].iov_base = buf;
	recv_entry->iov[0].iov_len = len;

	ret = xnet_srx_tag(srx, recv_entry);
	if (ret)
		ofi_buf_free(recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static ssize_t
xnet_srx_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, uint64_t tag,
		uint64_t ignore, void *context)
{
	struct xnet_xfer_entry *recv_entry;
	struct xnet_srx *srx;
	ssize_t ret;

	srx = container_of(ep_fid, struct xnet_srx, rx_fid);
	assert(count <= XNET_IOV_LIMIT);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	recv_entry = xnet_alloc_xfer(xnet_srx2_progress(srx));
	if (!recv_entry) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	recv_entry->tag = tag;
	recv_entry->ignore = ignore;
	recv_entry->src_addr = src_addr;
	recv_entry->cq_flags = FI_TAGGED | FI_RECV;
	recv_entry->cntr_inc = ofi_ep_rx_cntr_inc;
	recv_entry->context = context;

	recv_entry->iov_cnt = count;
	if (count) {
		recv_entry->user_buf = iov[0].iov_base;
		memcpy(&recv_entry->iov[0], iov, count * sizeof(*iov));
	}

	ret = xnet_srx_tag(srx, recv_entry);
	if (ret)
		ofi_buf_free(recv_entry);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);
	return ret;
}

static struct fi_ops_tagged xnet_srx_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = xnet_srx_trecv,
	.recvv = xnet_srx_trecvv,
	.recvmsg = xnet_srx_trecvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

static struct xnet_xfer_entry *
xnet_match_tag(struct xnet_srx *srx, struct xnet_ep *ep, uint64_t tag)
{
	struct xnet_xfer_entry *rx_entry;
	struct slist_entry *item, *prev;

	assert(xnet_progress_locked(xnet_srx2_progress(srx)));
	slist_foreach(&srx->tag_queue, item, prev) {
		rx_entry = container_of(item, struct xnet_xfer_entry, entry);
		if (ofi_match_tag(rx_entry->tag, rx_entry->ignore, tag)) {
			slist_remove(&srx->tag_queue, item, prev);
			return rx_entry;
		}
	}
	return NULL;
}

/* A matching receive could be found on either the any source queue or the
 * source matched queue.  However, most apps going through this path will
 * use source matching, with the any source queue being empty.  We optimize
 * for this case.
 */
static struct xnet_xfer_entry *
xnet_match_tag_addr(struct xnet_srx *srx, struct xnet_ep *ep, uint64_t tag)
{
	struct xnet_xfer_entry *rx_entry, *any_entry;
	struct slist *queue;
	struct slist_entry *any_item, *any_prev;
	struct slist_entry *item, *prev;

	assert(xnet_progress_locked(xnet_srx2_progress(srx)));
	queue = ofi_array_at(&srx->src_tag_queues, ep->peer->fi_addr);
	if (!queue)
		return xnet_match_tag(srx, ep, tag);

	slist_foreach(queue, item, prev) {
		rx_entry = container_of(item, struct xnet_xfer_entry, entry);
		if (ofi_match_tag(rx_entry->tag, rx_entry->ignore, tag))
			goto found;
	}

	return xnet_match_tag(srx, ep, tag);

found:
	/* We select from the any source queue if it matches and was posted
	 * earlier than our source based match.
	 */
	slist_foreach(&srx->tag_queue, any_item, any_prev) {
		any_entry = container_of(any_item, struct xnet_xfer_entry, entry);
		if (any_entry->tag_seq_no > rx_entry->tag_seq_no)
			break;

		if (ofi_match_tag(any_entry->tag, any_entry->ignore, tag)) {
			queue = &srx->tag_queue;
			rx_entry = any_entry;
			item = any_item;
			prev = any_prev;
			break;
		}
	}

	slist_remove(queue, item, prev);
	return rx_entry;
}

static bool
xnet_srx_cancel_rx(struct xnet_srx *srx, struct slist *queue, void *context)
{
	struct slist_entry *cur, *prev;
	struct xnet_xfer_entry *xfer_entry;

	assert(xnet_progress_locked(xnet_srx2_progress(srx)));
	slist_foreach(queue, cur, prev) {
		xfer_entry = container_of(cur, struct xnet_xfer_entry, entry);
		if (xfer_entry->context == context) {
			slist_remove(queue, cur, prev);
			xnet_cq_report_error(&srx->cq->util_cq, xfer_entry,
					     FI_ECANCELED);
			ofi_buf_free(xfer_entry);
			return true;
		}
	}

	return false;
}

static int
xnet_srx_cancel_src(struct ofi_dyn_arr *arr, void *list, void *context)
{
	struct xnet_srx *srx;
	struct slist *queue = list;

	srx = container_of(arr, struct xnet_srx, src_tag_queues);
	return (int) xnet_srx_cancel_rx(srx, queue, context);
}

static ssize_t xnet_srx_cancel(fid_t fid, void *context)
{
	struct xnet_srx *srx;

	srx = container_of(fid, struct xnet_srx, rx_fid.fid);

	ofi_genlock_lock(xnet_srx2_progress(srx)->active_lock);
	if (xnet_srx_cancel_rx(srx, &srx->tag_queue, context))
		goto unlock;

	if (xnet_srx_cancel_rx(srx, &srx->rx_queue, context))
		goto unlock;

	ofi_array_iter(&srx->src_tag_queues, context, xnet_srx_cancel_src);
unlock:
	ofi_genlock_unlock(xnet_srx2_progress(srx)->active_lock);

	return 0;
}

static struct fi_ops_ep xnet_srx_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = xnet_srx_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int xnet_srx_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct xnet_srx *srx;

	srx = container_of(fid, struct xnet_srx, rx_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		if (flags != FI_RECV)
			return -FI_EINVAL;

		srx->cq = container_of(bfid, struct xnet_cq, util_cq.cq_fid.fid);
		ofi_atomic_inc32(&srx->cq->util_cq.ref);
		break;
	case FI_CLASS_EP:
		if (flags != (FI_TAGGED | FI_MSG))
			return -FI_EINVAL;

		srx->rdm = container_of(bfid, struct xnet_rdm, util_ep.ep_fid.fid);
		break;
	default:
		return -FI_ENOSYS;
	}

	return FI_SUCCESS;
}

static void xnet_srx_cleanup(struct xnet_srx *srx, struct slist *queue)
{
	struct slist_entry *entry;
	struct xnet_xfer_entry *xfer_entry;

	while (!slist_empty(queue)) {
		entry = slist_remove_head(queue);
		xfer_entry = container_of(entry, struct xnet_xfer_entry, entry);
		if (srx->cq) {
			xnet_cq_report_error(&srx->cq->util_cq, xfer_entry,
					      FI_ECANCELED);
		}
		ofi_buf_free(xfer_entry);
	}
}

static int
xnet_srx_cleanup_arr(struct ofi_dyn_arr *arr, void *list, void *context)
{
	struct xnet_srx *srx = context;
	struct slist *queue = list;

	if (!slist_empty(queue))
		xnet_srx_cleanup(srx, queue);
	return 0;
}

static int xnet_srx_close(struct fid *fid)
{
	struct xnet_srx *srx;

	srx = container_of(fid, struct xnet_srx, rx_fid.fid);

	xnet_srx_cleanup(srx, &srx->rx_queue);
	xnet_srx_cleanup(srx, &srx->tag_queue);
	ofi_array_iter(&srx->src_tag_queues, srx, xnet_srx_cleanup_arr);
	ofi_array_destroy(&srx->src_tag_queues);

	if (srx->cq)
		ofi_atomic_dec32(&srx->cq->util_cq.ref);
	ofi_atomic_dec32(&srx->domain->util_domain.ref);
	free(srx);
	return FI_SUCCESS;
}

static struct fi_ops xnet_srx_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = xnet_srx_close,
	.bind = xnet_srx_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int xnet_srx_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		     struct fid_ep **rx_ep, void *context)
{
	struct xnet_srx *srx;

	srx = calloc(1, sizeof(*srx));
	if (!srx)
		return -FI_ENOMEM;

	srx->rx_fid.fid.fclass = FI_CLASS_SRX_CTX;
	srx->rx_fid.fid.context = context;
	srx->rx_fid.fid.ops = &xnet_srx_fid_ops;
	srx->rx_fid.ops = &xnet_srx_ops;

	srx->rx_fid.msg = &xnet_srx_msg_ops;
	srx->rx_fid.tagged = &xnet_srx_tag_ops;
	slist_init(&srx->rx_queue);
	slist_init(&srx->tag_queue);
	ofi_array_init(&srx->src_tag_queues, sizeof(struct slist), NULL);

	srx->domain = container_of(domain, struct xnet_domain,
				   util_domain.domain_fid);
	ofi_atomic_inc32(&srx->domain->util_domain.ref);
	srx->match_tag_rx = (attr->caps & FI_DIRECTED_RECV) ?
			    xnet_match_tag_addr : xnet_match_tag;
	srx->op_flags = attr->op_flags & FI_MULTI_RECV;
	srx->min_multi_recv_size = XNET_MIN_MULTI_RECV;
	*rx_ep = &srx->rx_fid;
	return FI_SUCCESS;
}
