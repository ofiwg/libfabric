/*
 * Copyright (c) 2023 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>

#include <ofi_enosys.h>
#include "ofi_iov.h"
#include <ofi_util.h>

static struct util_rx_entry *util_alloc_rx_entry(struct util_srx_ctx *srx)
{
	return (struct util_rx_entry *) ofi_buf_alloc(srx->rx_pool);
}

static inline struct iovec *util_srx_iov(struct util_rx_entry *rx_entry)
{
	return (struct iovec *) ((char *) rx_entry + sizeof(*rx_entry));
}

static inline void **util_srx_desc(struct util_srx_ctx *srx,
				   struct util_rx_entry *rx_entry)
{
	return (void **) ((char *) util_srx_iov(rx_entry) +
			(sizeof(struct iovec) * srx->iov_limit));
}

static void util_init_rx_entry(struct util_rx_entry *entry,
			       const struct iovec *iov, void **desc,
			       size_t count, fi_addr_t addr, void *context,
			       uint64_t tag, uint64_t flags)
{
	int i;

	for (i = 0; i < count; i++) {
		entry->peer_entry.iov[i] = iov[i];
		if (desc)
			entry->peer_entry.desc[i] = desc[i];
		else
			entry->peer_entry.desc[i] = NULL;
	}

	entry->peer_entry.count = count;
	entry->peer_entry.addr = addr;
	entry->peer_entry.context = context;
	entry->peer_entry.tag = tag;
	entry->peer_entry.flags = flags;
}

static struct util_rx_entry *util_get_recv_entry(struct util_srx_ctx *srx,
		const struct iovec *iov, void **desc, size_t count,
		fi_addr_t addr, void *context, uint64_t tag, uint64_t ignore,
		uint64_t flags)
{
	struct util_rx_entry *entry;

	entry = util_alloc_rx_entry(srx);
	if (!entry)
		return NULL;

	util_init_rx_entry(entry, iov, desc, count, addr, context, tag, flags);

	entry->peer_entry.owner_context = NULL;

	entry->multi_recv_ref = 0;
	entry->ignore = ignore;
	entry->seq_no = srx->rx_seq_no++;

	return entry;
}

static struct util_rx_entry *util_init_unexp(struct util_srx_ctx *srx,
			fi_addr_t addr, uint64_t size, uint64_t tag)
{
	struct util_rx_entry *util_entry;

	util_entry = util_alloc_rx_entry(srx);
	if (!util_entry)
		return NULL;

	util_entry->peer_entry.owner_context = NULL;
	util_entry->peer_entry.size = size;
	util_entry->peer_entry.addr = addr;
	util_entry->peer_entry.tag = tag;

	return util_entry;
}

static bool util_adjust_multi_recv(struct util_srx_ctx *srx,
		struct fi_peer_rx_entry *rx_entry, size_t len)
{
	size_t left;
	void *new_base;

	left = rx_entry->iov[0].iov_len - len;

	new_base = (void *) ((uintptr_t) rx_entry->iov[0].iov_base + len);
	rx_entry->iov[0].iov_len = left;
	rx_entry->iov[0].iov_base = new_base;
	rx_entry->size = left;

	return left < srx->min_multi_recv_size;
}

static struct util_rx_entry *util_process_multi_recv(struct util_srx_ctx *srx, struct slist *queue,
		fi_addr_t addr, size_t size, struct util_rx_entry *owner_entry)
{
	struct util_rx_entry *util_entry;

	util_entry = util_get_recv_entry(srx,
					 owner_entry->peer_entry.iov,
					 owner_entry->peer_entry.desc,
					 owner_entry->peer_entry.count, addr,
					 owner_entry->peer_entry.context,
					 owner_entry->peer_entry.tag,
					 owner_entry->ignore,
					 owner_entry->peer_entry.flags &
					 (~FI_MULTI_RECV));
	if (!util_entry)
		return NULL;

	if (util_adjust_multi_recv(srx, &owner_entry->peer_entry, size))
		slist_remove_head(queue);

	util_entry->peer_entry.owner_context = owner_entry;
	owner_entry->multi_recv_ref++;

	return util_entry;
}

static int util_match_msg(struct fid_peer_srx *srx, fi_addr_t addr, size_t size,
			  struct fi_peer_rx_entry **rx_entry)
{
	struct util_srx_ctx *srx_ctx;
	struct util_rx_entry *util_entry;
	struct slist_entry *head;
	int ret = FI_SUCCESS;
	
	srx_ctx = srx->ep_fid.fid.context;
	if (slist_empty(&srx_ctx->msg_queue)) {
		util_entry = util_init_unexp(srx_ctx, addr, size, 0);
		if (!util_entry)
			return -FI_ENOMEM;
		ret = -FI_ENOENT;
	} else {
		head = srx_ctx->msg_queue.head;
		util_entry = container_of(head, struct util_rx_entry, peer_entry);
		if (util_entry->peer_entry.flags & FI_MULTI_RECV) {
			util_entry = util_process_multi_recv(srx_ctx,
				&srx_ctx->msg_queue, addr, size, util_entry);
			if (!util_entry) {
				FI_WARN(&core_prov, FI_LOG_EP_CTRL,
					"cannot allocate multi receive buffer\n");
				return -FI_ENOMEM;
			}
		} else {
			(void) slist_remove_head(&srx_ctx->msg_queue);
		}
	}
	util_entry->peer_entry.srx = srx;
	*rx_entry = &util_entry->peer_entry;
	return ret;
}

static int util_get_msg(struct fid_peer_srx *srx, fi_addr_t addr,
		        size_t size, struct fi_peer_rx_entry **rx_entry)
{
	struct util_srx_ctx *srx_ctx;
	struct util_rx_entry *util_entry, *any_entry;
	struct slist *queue;
	int ret;

	srx_ctx = srx->ep_fid.fid.context;
	ofi_spin_lock(&srx_ctx->lock);

	queue = addr == FI_ADDR_UNSPEC ? NULL:
		ofi_array_at(&srx_ctx->src_recv_queues, addr);

	if (!queue || slist_empty(queue)) {
		ret = util_match_msg(srx, addr, size, rx_entry);
		goto out;
	}

	util_entry = container_of(queue->head, struct util_rx_entry, peer_entry);
	if (!slist_empty(&srx_ctx->msg_queue)) {
		any_entry = container_of(&srx_ctx->msg_queue.head,
					 struct util_rx_entry, peer_entry);
		if (any_entry->seq_no <= util_entry->seq_no) {
			queue = &srx_ctx->msg_queue;
			util_entry = any_entry;
		}
	}

	if (util_entry->peer_entry.flags & FI_MULTI_RECV) {
		util_entry = util_process_multi_recv(srx_ctx, queue, addr, size,
						     util_entry);
		if (!util_entry) {
			FI_WARN(&core_prov, FI_LOG_EP_CTRL,
				"cannot allocate multi receive buffer\n");
			ret = -FI_ENOMEM;
			goto out;
		}
	} else {
		(void) slist_remove_head(queue);
	}

	util_entry->peer_entry.srx = srx;
	*rx_entry = &util_entry->peer_entry;
	ret = FI_SUCCESS;
out:
	ofi_spin_unlock(&srx_ctx->lock);
	return ret;
}

static int util_match_tag(struct fid_peer_srx *srx, fi_addr_t addr,
			  uint64_t tag, struct fi_peer_rx_entry **rx_entry)
{
	struct util_srx_ctx *srx_ctx;
	struct util_rx_entry *util_entry;
	struct slist_entry *item, *prev;
	int ret = FI_SUCCESS;

	srx_ctx = srx->ep_fid.fid.context;
	slist_foreach(&srx_ctx->tag_queue, item, prev) {
		util_entry = container_of(item, struct util_rx_entry, peer_entry);
		if (ofi_match_tag(util_entry->peer_entry.tag, util_entry->ignore, tag)) {
			slist_remove(&srx_ctx->tag_queue, item, prev);
			goto out;
		}
	}

	util_entry = util_init_unexp(srx_ctx, addr, 0, tag);
	if (!util_entry)
		return -FI_ENOMEM;
	ret = -FI_ENOENT;
out:
	util_entry->peer_entry.srx = srx;
	*rx_entry = &util_entry->peer_entry;
	return ret;
}

static int util_get_tag(struct fid_peer_srx *srx, fi_addr_t addr,
			uint64_t tag, struct fi_peer_rx_entry **rx_entry)
{
	struct util_srx_ctx *srx_ctx;
	struct slist *queue;
	struct slist_entry *any_item, *any_prev;
	struct slist_entry *item, *prev;
	struct util_rx_entry *util_entry, *any_entry;
	int ret = FI_SUCCESS;

	srx_ctx = srx->ep_fid.fid.context;
	ofi_spin_lock(&srx_ctx->lock);

	queue = addr == FI_ADDR_UNSPEC ? &srx_ctx->tag_queue:
		ofi_array_at(&srx_ctx->src_trecv_queues, addr);

	if (!queue) {
		ret = util_match_tag(srx, addr, tag, rx_entry);
		goto out;
	}

	slist_foreach(queue, item, prev) {
		util_entry = container_of(item, struct util_rx_entry, peer_entry);
		if (ofi_match_tag(util_entry->peer_entry.tag, util_entry->ignore, tag))
			goto check_any;
	}
	ret = util_match_tag(srx, addr, tag, rx_entry);
	goto out;
check_any:
	slist_foreach(&srx_ctx->tag_queue, any_item, any_prev) {
		any_entry = container_of(any_item, struct util_rx_entry,
					 peer_entry);
		if (any_entry->seq_no > util_entry->seq_no)
			break;

		if (ofi_match_tag(any_entry->peer_entry.tag, any_entry->ignore, tag)) {
			queue = &srx_ctx->tag_queue;
			util_entry = any_entry;
			item = any_item;
			prev = any_prev;
			break;
		}
	}
	util_entry->peer_entry.srx = srx;
	*rx_entry = &util_entry->peer_entry;
	slist_remove(queue, item, prev);
out:
	ofi_spin_unlock(&srx_ctx->lock);
	return ret;
}

static int util_queue_msg(struct fi_peer_rx_entry *rx_entry)
{
	struct util_srx_ctx *srx_ctx = rx_entry->srx->ep_fid.fid.context;
	struct slist *queue;

	queue = rx_entry->addr == FI_ADDR_UNSPEC ? &srx_ctx->unexp_msg_queue :
		ofi_array_at(&srx_ctx->src_unexp_msg_queues, rx_entry->addr);
	ofi_spin_lock(&srx_ctx->lock);
	slist_insert_tail((struct slist_entry *) rx_entry, queue);
	ofi_spin_unlock(&srx_ctx->lock);
	return 0;
}

static int util_queue_tag(struct fi_peer_rx_entry *rx_entry)
{
	struct util_srx_ctx *srx_ctx = rx_entry->srx->ep_fid.fid.context;
	struct slist *queue;

	queue = rx_entry->addr == FI_ADDR_UNSPEC ? &srx_ctx->unexp_tag_queue :
		ofi_array_at(&srx_ctx->src_unexp_tag_queues, rx_entry->addr);
	ofi_spin_lock(&srx_ctx->lock);
	slist_insert_tail((struct slist_entry *) rx_entry, queue);
	ofi_spin_unlock(&srx_ctx->lock);
	return 0;
}

static void util_free_entry(struct fi_peer_rx_entry *entry)
{
	struct util_srx_ctx *srx;
	struct util_rx_entry *util_entry, *owner_entry;

	srx = (struct util_srx_ctx *) entry->srx->ep_fid.fid.context;

	ofi_spin_lock(&srx->lock);
	util_entry = container_of(entry, struct util_rx_entry, peer_entry);
	if (entry->owner_context) {
		owner_entry = (struct util_rx_entry *) entry->owner_context;
		if (!--owner_entry->multi_recv_ref &&
		    owner_entry->peer_entry.size < srx->min_multi_recv_size) {
			if (ofi_peer_cq_write(srx->cq,
					      owner_entry->peer_entry.context,
					      FI_MULTI_RECV, 0, NULL, 0, 0,
					      FI_ADDR_NOTAVAIL)) {
				FI_WARN(&core_prov, FI_LOG_EP_CTRL,
					"cannot write MULTI_RECV completion\n");
			}
			ofi_buf_free(owner_entry);
		}
	}
	ofi_buf_free(util_entry);
	ofi_spin_unlock(&srx->lock);
}

static struct fi_ops_srx_owner util_srx_owner_ops = {
	.size = sizeof(struct fi_ops_srx_owner),
	.get_msg = util_get_msg,
	.get_tag = util_get_tag,
	.queue_msg = util_queue_msg,
	.queue_tag = util_queue_tag,
	.free_entry = util_free_entry,
};

static struct util_rx_entry *util_match_unexp_msg(struct util_srx_ctx *srx,
			fi_addr_t addr)
{
	struct slist_entry *head;

	if (slist_empty(&srx->unexp_msg_queue))
		return NULL;

	head = slist_remove_head(&srx->unexp_msg_queue);
	return container_of(head, struct util_rx_entry, peer_entry);
}

static struct util_rx_entry *util_get_unexp_msg(struct util_srx_ctx *srx,
			fi_addr_t addr)
{
	struct util_rx_entry *util_entry, *any_entry;
	struct slist *queue;

	queue = addr == FI_ADDR_UNSPEC ? NULL:
		ofi_array_at(&srx->src_unexp_msg_queues, addr);

	if (!queue || slist_empty(queue))
		return util_match_unexp_msg(srx, addr);

	util_entry = container_of(queue->head, struct util_rx_entry, peer_entry);
	if (!slist_empty(&srx->unexp_msg_queue)) {
		any_entry = container_of(&srx->unexp_msg_queue.head,
					 struct util_rx_entry, peer_entry);
		if (any_entry->seq_no <= util_entry->seq_no) {
			queue = &srx->unexp_msg_queue;
			util_entry = any_entry;
		}
	}
	(void) slist_remove_head(queue);

	return util_entry;
}


static ssize_t util_generic_mrecv(struct util_srx_ctx *srx,
		const struct iovec *iov, void **desc, size_t iov_count,
		fi_addr_t addr, void *context, uint64_t flags)
{
	struct util_rx_entry *rx_entry, *mrecv_entry;
	struct slist *queue;
	bool buf_done = false;
	int ret;

	assert(flags & FI_MULTI_RECV && iov_count == 1);

	addr = srx->dir_recv ? addr : FI_ADDR_UNSPEC;
	queue = addr == FI_ADDR_UNSPEC ? &srx->msg_queue:
		ofi_array_at(&srx->src_recv_queues, addr);

	ofi_spin_lock(&srx->lock);
	mrecv_entry = util_get_recv_entry(srx, iov, desc, iov_count, addr,
					  context, 0, 0, flags);
	if (!mrecv_entry) {
		ret = -FI_ENOMEM;
		goto out;
	}
	mrecv_entry->peer_entry.size = ofi_total_iov_len(iov, iov_count);

	rx_entry = util_get_unexp_msg(srx, addr);
	while (rx_entry) {
		util_init_rx_entry(rx_entry, mrecv_entry->peer_entry.iov, desc,
				   iov_count, addr, context, 0,
				   flags & (~FI_MULTI_RECV));
		mrecv_entry->multi_recv_ref++;
		rx_entry->peer_entry.owner_context = mrecv_entry;

		if (util_adjust_multi_recv(srx, &mrecv_entry->peer_entry,
					   rx_entry->peer_entry.size))
			buf_done = true;

		ofi_spin_unlock(&srx->lock);
		ret = rx_entry->peer_entry.srx->peer_ops->start_msg(&rx_entry->peer_entry);
		if (ret || buf_done)
			return ret;

		ofi_spin_lock(&srx->lock);
		rx_entry = util_get_unexp_msg(srx, addr);
	}
	slist_insert_tail((struct slist_entry *)(&mrecv_entry->peer_entry), queue);
	ret = FI_SUCCESS;
out:
	ofi_spin_unlock(&srx->lock);
	return ret;
}

static struct util_rx_entry *util_match_unexp_tag(struct util_srx_ctx *srx,
			fi_addr_t addr, uint64_t tag, uint64_t ignore)
{
	struct util_rx_entry *util_entry;
	struct slist_entry *item, *prev;

	if (slist_empty(&srx->unexp_tag_queue))
		return NULL;

	slist_foreach(&srx->unexp_tag_queue, item, prev) {
		util_entry = container_of(item, struct util_rx_entry, peer_entry);
		if (ofi_match_tag(tag, ignore, util_entry->peer_entry.tag)) {
			slist_remove(&srx->tag_queue, item, prev);
			return util_entry;
		}
	}
	return NULL;
}

static struct util_rx_entry *util_get_unexp_tag(struct util_srx_ctx *srx,
			fi_addr_t addr, uint64_t tag, uint64_t ignore)
{
	struct slist *queue;
	struct slist_entry *any_item, *any_prev;
	struct slist_entry *item, *prev;
	struct util_rx_entry *util_entry, *any_entry;

	queue = addr == FI_ADDR_UNSPEC ? &srx->unexp_tag_queue:
		ofi_array_at(&srx->src_unexp_tag_queues, addr);

	if (!queue)
		return util_match_unexp_tag(srx, addr, tag, ignore);

	slist_foreach(queue, item, prev) {
		util_entry = container_of(item, struct util_rx_entry, peer_entry);
		if (ofi_match_tag(tag, ignore, util_entry->peer_entry.tag))
			goto check_any;
	}
	return util_match_unexp_tag(srx, addr, tag, ignore);

check_any:
	slist_foreach(&srx->unexp_tag_queue, any_item, any_prev) {
		any_entry = container_of(any_item, struct util_rx_entry,
					 peer_entry);
		if (any_entry->seq_no > util_entry->seq_no)
			break;

		if (ofi_match_tag(tag, ignore, any_entry->peer_entry.tag)) {
			queue = &srx->unexp_tag_queue;
			util_entry = any_entry;
			item = any_item;
			prev = any_prev;
			break;
		}
	}
	slist_remove(queue, item, prev);
	return util_entry;
}

ssize_t util_srx_generic_trecv(struct fid_ep *ep_fid, const struct iovec *iov,
			       void **desc, size_t iov_count, fi_addr_t addr,
			       void *context, uint64_t tag, uint64_t ignore,
			       uint64_t flags)
{
	struct util_srx_ctx *srx;
	struct util_rx_entry *rx_entry;
	struct slist *queue;
	int ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	assert(iov_count <= srx->iov_limit);
	addr = srx->dir_recv ? addr : FI_ADDR_UNSPEC;
	queue = addr == FI_ADDR_UNSPEC ? &srx->tag_queue:
		ofi_array_at(&srx->src_trecv_queues, addr);

	ofi_spin_lock(&srx->lock);
	rx_entry = util_get_unexp_tag(srx, addr, tag, ignore);
	if (!rx_entry) {
		rx_entry = util_get_recv_entry(srx, iov, desc, iov_count, addr,
					       context, tag, ignore, flags);
		if (!rx_entry)
			ret = -FI_ENOMEM;
		else
			slist_insert_tail((struct slist_entry *)
					  (&rx_entry->peer_entry), queue);
		ofi_spin_unlock(&srx->lock);
		return ret;
	}
	ofi_spin_unlock(&srx->lock);

	util_init_rx_entry(rx_entry, iov, desc, iov_count, addr, context, tag,
			   flags);

	return rx_entry->peer_entry.srx->peer_ops->start_tag(&rx_entry->peer_entry);
}

ssize_t util_srx_generic_recv(struct fid_ep *ep_fid, const struct iovec *iov,
			      void **desc, size_t iov_count, fi_addr_t addr,
			      void *context, uint64_t flags)
{
	struct util_srx_ctx *srx;
	struct util_rx_entry *rx_entry;
	struct slist *queue;
	int ret = FI_SUCCESS;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);
	if (flags & FI_MULTI_RECV)
		return util_generic_mrecv(srx, iov, desc, iov_count, addr,
					  context, flags);

	assert(iov_count <= srx->iov_limit);
	addr = srx->dir_recv ? addr : FI_ADDR_UNSPEC;

	queue = addr == FI_ADDR_UNSPEC ? &srx->msg_queue :
			ofi_array_at(&srx->src_recv_queues, addr);
	ofi_spin_lock(&srx->lock);
	rx_entry = util_get_unexp_msg(srx, addr);
	if (!rx_entry) {
		rx_entry = util_get_recv_entry(srx, iov, desc, iov_count, addr,
					       context, 0, 0, flags);
		if (!rx_entry)
			ret = -FI_ENOMEM;
		else
			slist_insert_tail((struct slist_entry *)
					  (&rx_entry->peer_entry), queue);
		ofi_spin_unlock(&srx->lock);
		return ret;
	}
	ofi_spin_unlock(&srx->lock);

	util_init_rx_entry(rx_entry, iov, desc, iov_count, addr, context, 0,
			   flags);

	return rx_entry->peer_entry.srx->peer_ops->start_msg(&rx_entry->peer_entry);
}

static ssize_t util_srx_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
				uint64_t flags)
{
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	return util_srx_generic_recv(ep_fid, msg->msg_iov, msg->desc,
				     msg->iov_count, msg->addr, msg->context,
				     flags | srx->rx_msg_flags);
}

static ssize_t util_srx_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      void *context)
{
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	return util_srx_generic_recv(ep_fid, iov, desc, count, src_addr,
				    context, srx->rx_op_flags);
}

static ssize_t util_srx_recv(struct fid_ep *ep_fid, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return util_srx_generic_recv(ep_fid, &iov, &desc, 1, src_addr, context,
				     srx->rx_op_flags);
}

struct fi_ops_msg util_srx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = util_srx_recv,
	.recvv = util_srx_recvv,
	.recvmsg = util_srx_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static ssize_t util_srx_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
			      void *desc, fi_addr_t src_addr, uint64_t tag,
			      uint64_t ignore, void *context)
{
	struct iovec iov;
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return util_srx_generic_trecv(ep_fid, &iov, &desc, 1, src_addr, context,
				      tag, ignore, srx->rx_op_flags);
}

static ssize_t util_srx_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
			       void **desc, size_t count, fi_addr_t src_addr,
			       uint64_t tag, uint64_t ignore, void *context)
{
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	return util_srx_generic_trecv(ep_fid, iov, desc, count, src_addr,
				      context, tag, ignore, srx->rx_op_flags);
}

static ssize_t util_srx_trecvmsg(struct fid_ep *ep_fid,
			const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	return util_srx_generic_trecv(ep_fid, msg->msg_iov, msg->desc,
				      msg->iov_count, msg->addr, msg->context,
				      msg->tag, msg->ignore,
				      flags | srx->rx_msg_flags);
}

struct fi_ops_tagged util_srx_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = util_srx_trecv,
	.recvv = util_srx_trecvv,
	.recvmsg = util_srx_trecvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

int util_srx_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct util_srx_ctx *srx;

	if (flags != FI_RECV || bfid->fclass != FI_CLASS_CQ)
		return -FI_EINVAL;

	srx = container_of(fid, struct util_srx_ctx, peer_srx.ep_fid.fid);
	srx->cq = container_of(bfid, struct util_cq, cq_fid.fid);
	ofi_atomic_inc32(&srx->cq->ref);
	return FI_SUCCESS;
}

static int util_cancel_entry(struct util_srx_ctx *srx, uint64_t flags,
			     struct util_rx_entry *rx_entry)
{
	struct fi_cq_err_entry err_entry;
	int ret;

	err_entry.op_context = rx_entry->peer_entry.context;
	err_entry.flags = flags;
	err_entry.tag = rx_entry->peer_entry.tag;
	err_entry.err = FI_ECANCELED;
	err_entry.prov_errno = -FI_ECANCELED;

	ret = ofi_peer_cq_write_error(srx->cq, &err_entry);
	ofi_buf_free(rx_entry);
	return ret ? ret : 1;
}

static int util_cleanup_queues(struct ofi_dyn_arr *arr, void *list,
			       void *context)
{
	struct util_srx_ctx *srx = context;
	struct slist *queue = list;
	struct slist_entry *item;
	struct util_rx_entry *rx_entry;
	uint64_t flags;

	//TODO pass/set this better
	flags = arr == &srx->src_trecv_queues ?
		FI_TAGGED | FI_RECV : FI_MSG | FI_RECV;

	while (!slist_empty(queue)) {
		item = slist_remove_head(queue);
		rx_entry = container_of(item, struct util_rx_entry, peer_entry);
		(void) util_cancel_entry(srx, flags, rx_entry);
	}
	return 0;
}

static int util_cleanup_msg_queues(struct ofi_dyn_arr *arr, void *list,
			          void *context)
{
	struct slist *queue = list;
	struct slist_entry *item;
	struct util_rx_entry *rx_entry;

	while (!slist_empty(queue)) {
		item = slist_remove_head(queue);
		rx_entry = container_of(item, struct util_rx_entry, peer_entry);
		rx_entry->peer_entry.srx->peer_ops->discard_msg(
							&rx_entry->peer_entry);
	}
	return 0;
}

static int util_cleanup_tag_queues(struct ofi_dyn_arr *arr, void *list,
			          void *context)
{
	struct slist *queue = list;
	struct slist_entry *item;
	struct util_rx_entry *rx_entry;

	while (!slist_empty(queue)) {
		item = slist_remove_head(queue);
		rx_entry = container_of(item, struct util_rx_entry, peer_entry);
		rx_entry->peer_entry.srx->peer_ops->discard_tag(
							&rx_entry->peer_entry);
	}
	return 0;
}

int util_srx_close(struct fid *fid)
{
	struct util_srx_ctx *srx;
	struct util_rx_entry *rx_entry;
	struct slist_entry *entry;

	srx = container_of(fid, struct util_srx_ctx, peer_srx.ep_fid.fid);
	if (!srx)
		return -FI_EINVAL;

	(void)ofi_array_iter(&srx->src_recv_queues, srx, util_cleanup_queues);
	(void)ofi_array_iter(&srx->src_trecv_queues, srx, util_cleanup_queues);
	ofi_array_destroy(&srx->src_recv_queues);
	ofi_array_destroy(&srx->src_trecv_queues);

	while (!slist_empty(&srx->msg_queue)) {
		entry = slist_remove_head(&srx->msg_queue);
		(void) util_cancel_entry(srx, FI_SEND | FI_MSG,
			container_of(entry, struct util_rx_entry, peer_entry));
	}
	
	while (!slist_empty(&srx->tag_queue)) {
		entry = slist_remove_head(&srx->tag_queue);
		(void) util_cancel_entry(srx, FI_SEND | FI_TAGGED,
			container_of(entry, struct util_rx_entry, peer_entry));
	}

	(void)ofi_array_iter(&srx->src_unexp_msg_queues, srx, 
			     util_cleanup_msg_queues);
	(void)ofi_array_iter(&srx->src_unexp_tag_queues, srx, 
			     util_cleanup_tag_queues);
	ofi_array_destroy(&srx->src_unexp_msg_queues);
	ofi_array_destroy(&srx->src_unexp_tag_queues);

	while (!slist_empty(&srx->unexp_msg_queue)) {
		entry = slist_remove_head(&srx->unexp_msg_queue);
		rx_entry = container_of(entry, struct util_rx_entry, peer_entry);
		rx_entry->peer_entry.srx->peer_ops->discard_msg(
							&rx_entry->peer_entry);
	}
	
	while (!slist_empty(&srx->unexp_tag_queue)) {
		entry = slist_remove_head(&srx->unexp_tag_queue);
		rx_entry = container_of(entry, struct util_rx_entry, peer_entry);
		rx_entry->peer_entry.srx->peer_ops->discard_tag(
							&rx_entry->peer_entry);
	}

	ofi_atomic_dec32(&srx->cq->ref);
	ofi_bufpool_destroy(srx->rx_pool);
	ofi_spin_destroy(&srx->lock);
	free(srx);

	return FI_SUCCESS;
}

static struct fi_ops util_srx_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = util_srx_close,
	.bind = util_srx_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static bool util_cancel_recv(struct util_srx_ctx *srx, struct slist *queue,
			     uint64_t flags, void *context)
{
	struct slist_entry *item, *prev;
	struct util_rx_entry *rx_entry;

	ofi_spin_lock(&srx->lock);
	slist_foreach(queue, item, prev) {
		rx_entry = container_of(item, struct util_rx_entry, peer_entry);
		if (rx_entry->peer_entry.context == context) {
			slist_remove(queue, item, prev);
			util_cancel_entry(srx, flags, rx_entry);
			ofi_spin_unlock(&srx->lock);
			return true;
		}
	}
	ofi_spin_unlock(&srx->lock);
	return false;
}

static int util_cancel_src(struct ofi_dyn_arr *arr, void *list, void *context)
{
	struct util_srx_ctx *srx;
	struct slist *queue = list;
	uint64_t flags;

	srx = container_of(arr, struct util_srx_ctx, src_trecv_queues);

	flags = arr == &srx->src_trecv_queues ?
		FI_TAGGED | FI_RECV : FI_MSG | FI_RECV;

	return (int) util_cancel_recv(srx, queue, flags, context);
}

static ssize_t util_srx_cancel(fid_t ep_fid, void *context)
{
	struct util_srx_ctx *srx;

	srx = container_of(ep_fid, struct util_srx_ctx, peer_srx.ep_fid);

	if (util_cancel_recv(srx, &srx->tag_queue, FI_TAGGED | FI_RECV,
			     context))
		return 0;

	if (util_cancel_recv(srx, &srx->msg_queue, FI_MSG | FI_RECV,
			     context))
		return 0;

	if (ofi_array_iter(&srx->src_trecv_queues, context, util_cancel_src))
		return 0;

	if (ofi_array_iter(&srx->src_recv_queues, context, util_cancel_src))
		return 0;

	return 0;
}

static int util_srx_getopt(fid_t fid, int level, int optname,
		           void *optval, size_t *optlen)
{
	struct util_srx_ctx *srx =
		container_of(fid, struct util_srx_ctx, peer_srx.ep_fid.fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	*(size_t *)optval = srx->min_multi_recv_size;
	*optlen = sizeof(size_t);

	return FI_SUCCESS;
}

static int util_srx_setopt(fid_t fid, int level, int optname,
		           const void *optval, size_t optlen)
{
	struct util_srx_ctx *srx =
		container_of(fid, struct util_srx_ctx, peer_srx.ep_fid.fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	srx->min_multi_recv_size = *(size_t *)optval;

	return FI_SUCCESS;
}

static struct fi_ops_ep util_srx_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = util_srx_cancel,
	.getopt = util_srx_getopt,
	.setopt = util_srx_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void util_rx_entry_init_fn(struct ofi_bufpool_region *region, void *buf)
{
	struct util_rx_entry *rx_entry = (struct util_rx_entry *) buf;
	struct util_srx_ctx *srx = (struct util_srx_ctx *)
					region->pool->attr.context;

	rx_entry->peer_entry.iov = util_srx_iov(rx_entry);
	rx_entry->peer_entry.desc = util_srx_desc(srx, rx_entry);
}

static void util_srx_init_list(struct ofi_dyn_arr *arr, void *item)
{
	slist_init((struct slist *) item);
}

int util_ep_srx_context(struct util_domain *domain, size_t rx_size,
			size_t iov_limit, size_t default_min_mr,
			struct fid_ep **rx_ep)
{
	struct util_srx_ctx *srx;
	struct ofi_bufpool_attr pool_attr;
	int ret = FI_SUCCESS;

	srx = calloc(1, sizeof(*srx));
	if (!srx)
		return -FI_ENOMEM;

	ret = ofi_spin_init(&srx->lock);
	if (ret)
		goto err;

	ofi_array_init(&srx->src_unexp_msg_queues, sizeof(struct slist),
		       util_srx_init_list);
	ofi_array_init(&srx->src_unexp_tag_queues, sizeof(struct slist),
		       util_srx_init_list);

	ofi_array_init(&srx->src_recv_queues, sizeof(struct slist),
		       util_srx_init_list);
	ofi_array_init(&srx->src_trecv_queues, sizeof(struct slist),
		       util_srx_init_list);

	slist_init(&srx->msg_queue);
	slist_init(&srx->tag_queue);
	slist_init(&srx->unexp_msg_queue);
	slist_init(&srx->unexp_tag_queue);
	srx->rx_seq_no = 0;

	pool_attr.size = sizeof(struct util_rx_entry) +
		(sizeof(struct iovec) + sizeof(void *)) * iov_limit;
	pool_attr.alignment = 16;
	pool_attr.max_cnt = 0,
	pool_attr.chunk_cnt = rx_size,
	pool_attr.alloc_fn = NULL;
	pool_attr.free_fn = NULL;
	pool_attr.init_fn = util_rx_entry_init_fn;
	pool_attr.context = srx;
	pool_attr.flags = OFI_BUFPOOL_NO_TRACK;
	ret = ofi_bufpool_create_attr(&pool_attr, &srx->rx_pool);
	if (ret)
		return ret;

	srx->min_multi_recv_size = default_min_mr;
	srx->iov_limit = iov_limit;
	srx->dir_recv = domain->info_domain_caps & FI_DIRECTED_RECV;

	srx->peer_srx.owner_ops = &util_srx_owner_ops;
	srx->peer_srx.peer_ops = NULL;

	srx->peer_srx.ep_fid.fid.fclass = FI_CLASS_SRX_CTX;
	srx->peer_srx.ep_fid.fid.context = srx;
	srx->peer_srx.ep_fid.fid.ops = &util_srx_fid_ops;
	srx->peer_srx.ep_fid.ops = &util_srx_ops;

	srx->peer_srx.ep_fid.msg = &util_srx_msg_ops;
	srx->peer_srx.ep_fid.tagged = &util_srx_tag_ops;
	*rx_ep = &srx->peer_srx.ep_fid;

	return FI_SUCCESS;

err:
	free(srx);
	return ret;
}
