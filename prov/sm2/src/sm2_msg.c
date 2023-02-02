/*
 * Copyright (c) 2013-2021 Intel Corporation. All rights reserved
 * (C) Copyright 2021 Amazon.com, Inc. or its affiliates.
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
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "sm2.h"

struct sm2_rx_entry *sm2_alloc_rx_entry(struct sm2_srx_ctx *srx)
{
	if (ofi_freestack_isempty(srx->recv_fs)) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"not enough space to post recv\n");
		return NULL;
	}

	return ofi_freestack_pop(srx->recv_fs);
}

void sm2_init_rx_entry(struct sm2_rx_entry *entry, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t addr,
		       void *context, uint64_t tag, uint64_t flags)
{
	memcpy(&entry->iov, iov, sizeof(*iov) * count);
	if (desc)
		memcpy(entry->desc, desc, sizeof(*desc) * count);

	entry->peer_entry.iov = entry->iov;
	entry->peer_entry.desc = entry->desc;
	entry->peer_entry.count = count;
	entry->peer_entry.addr = addr;
	entry->peer_entry.context = context;
	entry->peer_entry.tag = tag;
	entry->peer_entry.flags = flags;
}

struct sm2_rx_entry *sm2_get_recv_entry(struct sm2_srx_ctx *srx,
		const struct iovec *iov, void **desc, size_t count,
		fi_addr_t addr, void *context, uint64_t tag, uint64_t ignore,
		uint64_t flags)
{
	struct sm2_rx_entry *entry;

	entry = sm2_alloc_rx_entry(srx);
	if (!entry)
		return NULL;

	sm2_init_rx_entry(entry, iov, desc, count, addr,
			  context, tag, flags);

	entry->peer_entry.owner_context = NULL;

	entry->multi_recv_ref = 0;
	entry->ignore = ignore;
	entry->err = 0;

	return entry;
}

static ssize_t sm2_generic_mrecv(struct sm2_srx_ctx *srx,
		const struct iovec *iov, void **desc, size_t iov_count,
		fi_addr_t addr, void *context, uint64_t flags)
{
	struct sm2_match_attr match_attr;
	struct sm2_rx_entry *rx_entry, *mrecv_entry;
	struct dlist_entry *dlist_entry;
	bool buf_done = false;
	int ret;

	assert(flags & FI_MULTI_RECV && iov_count == 1);

	addr = srx->dir_recv ? addr : FI_ADDR_UNSPEC;
	match_attr.id = addr;

	ofi_spin_lock(&srx->lock);
	mrecv_entry = sm2_get_recv_entry(srx, iov, desc, iov_count, addr,
					 context, 0, 0, flags);
	if (!mrecv_entry) {
		ret = -FI_ENOMEM;
		goto out;
	}
	mrecv_entry->peer_entry.size = ofi_total_iov_len(iov, iov_count);

	dlist_entry = dlist_remove_first_match(&srx->unexp_msg_queue.list,
					       srx->unexp_msg_queue.match_func,
					       &match_attr);
	while (dlist_entry) {
		rx_entry = container_of(dlist_entry, struct sm2_rx_entry,
					peer_entry);
		sm2_init_rx_entry(rx_entry, mrecv_entry->peer_entry.iov, desc,
				  iov_count, addr, context, 0,
				  flags & (~FI_MULTI_RECV));
		mrecv_entry->multi_recv_ref++;
		rx_entry->peer_entry.owner_context = mrecv_entry;

		if (sm2_adjust_multi_recv(srx, &mrecv_entry->peer_entry,
					  rx_entry->peer_entry.size))
			buf_done = true;

		ofi_spin_unlock(&srx->lock);
		ret = srx->peer_srx.peer_ops->start_msg(&rx_entry->peer_entry);
		if (ret || buf_done)
			return ret;

		ofi_spin_lock(&srx->lock);
		dlist_entry = dlist_remove_first_match(&srx->unexp_msg_queue.list,
						       srx->unexp_msg_queue.match_func,
						       &match_attr);
	}

	dlist_insert_tail((struct dlist_entry *) (&mrecv_entry->peer_entry),
			  &srx->recv_queue.list);
	ret = FI_SUCCESS;
out:
	ofi_spin_unlock(&srx->lock);
	return ret;
}

static ssize_t sm2_generic_recv(struct sm2_srx_ctx *srx, const struct iovec *iov,
		void **desc, size_t iov_count, fi_addr_t addr, void *context,
		uint64_t tag, uint64_t ignore, uint64_t flags,
		struct sm2_queue *recv_queue, struct sm2_queue *unexp_queue)
{
	struct sm2_match_attr match_attr;
	struct sm2_rx_entry *rx_entry;
	struct dlist_entry *dlist_entry;
	int ret = FI_SUCCESS;

	if (flags & FI_MULTI_RECV) {
		assert(recv_queue != &srx->trecv_queue);
		return sm2_generic_mrecv(srx, iov, desc, iov_count, addr,
					 context, flags);
	}

	assert(iov_count <= SM2_IOV_LIMIT);

	addr = srx->dir_recv ? addr : FI_ADDR_UNSPEC;
	match_attr.id = addr;
	match_attr.ignore = ignore;
	match_attr.tag = tag;

	ofi_spin_lock(&srx->lock);
	dlist_entry = dlist_remove_first_match(&unexp_queue->list,
					       unexp_queue->match_func,
					       &match_attr);
	if (!dlist_entry) {
		rx_entry = sm2_get_recv_entry(srx, iov, desc, iov_count, addr,
					      context, tag, ignore, flags);
		if (!rx_entry)
			ret = -FI_ENOMEM;
		else
			dlist_insert_tail((struct dlist_entry *)
					  (&rx_entry->peer_entry),
					  &recv_queue->list);
		ofi_spin_unlock(&srx->lock);
		return ret;
	}
	ofi_spin_unlock(&srx->lock);

	rx_entry = container_of(dlist_entry, struct sm2_rx_entry, peer_entry);
	sm2_init_rx_entry(rx_entry, iov, desc, iov_count, addr, context,
			  tag, flags);

	return srx->peer_srx.peer_ops->start_msg(&rx_entry->peer_entry);
}

static ssize_t sm2_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_recv(sm2_get_sm2_srx(ep), msg->msg_iov, msg->desc,
				msg->iov_count, msg->addr, msg->context, 0, 0,
				flags | ep->util_ep.rx_msg_flags,
				&sm2_get_sm2_srx(ep)->recv_queue,
				&sm2_get_sm2_srx(ep)->unexp_msg_queue);
}

static ssize_t sm2_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t src_addr,
			 void *context)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_recv(sm2_get_sm2_srx(ep), iov, desc, count, src_addr,
				context, 0, 0, sm2_ep_rx_flags(ep),
				&sm2_get_sm2_srx(ep)->recv_queue,
				&sm2_get_sm2_srx(ep)->unexp_msg_queue);
}

static ssize_t sm2_recv(struct fid_ep *ep_fid, void *buf, size_t len,
			void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return sm2_generic_recv(sm2_get_sm2_srx(ep), &iov, &desc, 1, src_addr,
				context, 0, 0, sm2_ep_rx_flags(ep),
				&sm2_get_sm2_srx(ep)->recv_queue,
				&sm2_get_sm2_srx(ep)->unexp_msg_queue);
}

static ssize_t sm2_srx_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	return sm2_generic_recv(srx, msg->msg_iov, msg->desc, msg->iov_count,
				msg->addr, msg->context, 0, 0,
				flags | srx->rx_msg_flags, &srx->recv_queue,
				&srx->unexp_msg_queue);
}

static ssize_t sm2_srx_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			     size_t count, fi_addr_t src_addr, void *context)
{
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	return sm2_generic_recv(srx, iov, desc, count, src_addr, context, 0, 0,
				srx->rx_op_flags, &srx->recv_queue,
				&srx->unexp_msg_queue);
}

static ssize_t sm2_srx_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return sm2_generic_recv(srx, &iov, &desc, 1, src_addr, context, 0, 0,
				srx->rx_op_flags, &srx->recv_queue,
				&srx->unexp_msg_queue);
}

static ssize_t sm2_generic_sendmsg(struct sm2_ep *ep, const struct iovec *iov,
				   void **desc, size_t iov_count, fi_addr_t addr,
				   uint64_t tag, uint64_t data, void *context,
				   uint32_t op, uint64_t op_flags)
{
	struct sm2_region *peer_smr;
	enum fi_hmem_iface iface;
	uint64_t device;
	int64_t id, peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(iov_count <= SM2_IOV_LIMIT);

	id = sm2_verify_peer(ep, addr);
	if (id < 0)
		return -FI_EAGAIN;

	peer_id = sm2_peer_data(ep->region)[id].addr.id;
	peer_smr = sm2_peer_region(ep->region, id);

	pthread_spin_lock(&peer_smr->lock);
	if (!peer_smr->cmd_cnt || sm2_peer_data(ep->region)[id].sar_status) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	ofi_spin_lock(&ep->tx_lock);
	iface = sm2_get_mr_hmem_iface(ep->util_ep.domain, desc, &device);

	total_len = ofi_total_iov_len(iov, iov_count);
	assert(!(op_flags & FI_INJECT) || total_len <= SM2_INJECT_SIZE);

	ret = sm2_proto_ops[sm2_src_inject](ep, peer_smr, id, peer_id, op, tag, data, op_flags,
				   iface, device, iov, iov_count, total_len, context);
	if (ret)
		goto unlock_cq;

	sm2_signal(peer_smr);

	ret = sm2_complete_tx(ep, context, op, op_flags);
	if (ret) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
		goto unlock_cq;
	}

unlock_cq:
	ofi_spin_unlock(&ep->tx_lock);
unlock_region:
	pthread_spin_unlock(&peer_smr->lock);
	return ret;
}

static ssize_t sm2_send(struct fid_ep *ep_fid, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, void *context)
{
	struct sm2_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return sm2_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, 0,
				   0, context, ofi_op_msg, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t dest_addr, void *context)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_sendmsg(ep, iov, desc, count, dest_addr, 0,
				   0, context, ofi_op_msg, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			   uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg, flags | ep->util_ep.tx_msg_flags);
}

static ssize_t sm2_generic_inject(struct fid_ep *ep_fid, const void *buf,
				  size_t len, fi_addr_t dest_addr, uint64_t tag,
				  uint64_t data, uint32_t op, uint64_t op_flags)
{
	struct sm2_ep *ep;
	struct sm2_region *peer_smr;
	int64_t id, peer_id;
	ssize_t ret = 0;
	struct iovec msg_iov;

	assert(len <= SM2_INJECT_SIZE);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	id = sm2_verify_peer(ep, dest_addr);
	if (id < 0)
		return -FI_EAGAIN;

	peer_id = sm2_peer_data(ep->region)[id].addr.id;
	peer_smr = sm2_peer_region(ep->region, id);

	pthread_spin_lock(&peer_smr->lock);
	if (!peer_smr->cmd_cnt || sm2_peer_data(ep->region)[id].sar_status) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	ret = sm2_proto_ops[sm2_src_inject](ep, peer_smr, id, peer_id, op, tag, data,
			op_flags, FI_HMEM_SYSTEM, 0, &msg_iov, 1, len, NULL);

	assert(!ret);
	ofi_ep_tx_cntr_inc_func(&ep->util_ep, op);

	sm2_signal(peer_smr);
unlock:
	pthread_spin_unlock(&peer_smr->lock);

	return ret;
}

static ssize_t sm2_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			  fi_addr_t dest_addr)
{
	return sm2_generic_inject(ep_fid, buf, len, dest_addr, 0, 0,
				  ofi_op_msg, 0);
}

static ssize_t sm2_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			    void *desc, uint64_t data, fi_addr_t dest_addr,
			    void *context)
{
	struct sm2_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return sm2_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, 0, data,
				   context, ofi_op_msg,
				   FI_REMOTE_CQ_DATA | sm2_ep_tx_flags(ep));
}

static ssize_t sm2_injectdata(struct fid_ep *ep_fid, const void *buf,
			      size_t len, uint64_t data, fi_addr_t dest_addr)
{
	return sm2_generic_inject(ep_fid, buf, len, dest_addr, 0, data,
				  ofi_op_msg, FI_REMOTE_CQ_DATA);
}

struct fi_ops_msg sm2_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = sm2_recv,
	.recvv = sm2_recvv,
	.recvmsg = sm2_recvmsg,
	.send = sm2_send,
	.sendv = sm2_sendv,
	.sendmsg = sm2_sendmsg,
	.inject = sm2_inject,
	.senddata = sm2_senddata,
	.injectdata = sm2_injectdata,
};

struct fi_ops_msg sm2_no_recv_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = sm2_send,
	.sendv = sm2_sendv,
	.sendmsg = sm2_sendmsg,
	.inject = sm2_inject,
	.senddata = sm2_senddata,
	.injectdata = sm2_injectdata,
};

struct fi_ops_msg sm2_srx_msg_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = sm2_srx_recv,
	.recvv = sm2_srx_recvv,
	.recvmsg = sm2_srx_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static ssize_t sm2_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
			 void *desc, fi_addr_t src_addr, uint64_t tag,
			 uint64_t ignore, void *context)
{
	struct iovec iov;
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return sm2_generic_recv(sm2_get_sm2_srx(ep), &iov, &desc, 1, src_addr,
				context, tag, ignore, sm2_ep_rx_flags(ep),
				&sm2_get_sm2_srx(ep)->trecv_queue,
				&sm2_get_sm2_srx(ep)->unexp_tagged_queue);
}

static ssize_t sm2_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  uint64_t tag, uint64_t ignore, void *context)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_recv(sm2_get_sm2_srx(ep), iov, desc, count, src_addr,
				context, tag, ignore, sm2_ep_rx_flags(ep),
				&sm2_get_sm2_srx(ep)->trecv_queue,
				&sm2_get_sm2_srx(ep)->unexp_tagged_queue);
}

static ssize_t sm2_trecvmsg(struct fid_ep *ep_fid,
			    const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_recv(sm2_get_sm2_srx(ep), msg->msg_iov, msg->desc,
				msg->iov_count, msg->addr, msg->context,
				msg->tag, msg->ignore,
				flags | ep->util_ep.rx_msg_flags,
				&sm2_get_sm2_srx(ep)->trecv_queue,
				&sm2_get_sm2_srx(ep)->unexp_tagged_queue);
}

static ssize_t sm2_srx_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct iovec iov;
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return sm2_generic_recv(srx, &iov, &desc, 1, src_addr, context, tag,
				ignore, srx->rx_op_flags, &srx->trecv_queue,
				&srx->unexp_tagged_queue);
}

static ssize_t sm2_srx_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	return sm2_generic_recv(srx, iov, desc, count, src_addr, context, tag,
				ignore, srx->rx_op_flags, &srx->trecv_queue,
				&srx->unexp_tagged_queue);
}

static ssize_t sm2_srx_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct sm2_srx_ctx *srx;

	srx = container_of(ep_fid, struct sm2_srx_ctx, peer_srx.ep_fid);

	return sm2_generic_recv(srx, msg->msg_iov, msg->desc, msg->iov_count,
				msg->addr, msg->context, msg->tag, msg->ignore,
				flags | srx->rx_msg_flags, &srx->trecv_queue,
				&srx->unexp_tagged_queue);
}

static ssize_t sm2_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct sm2_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return sm2_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   sm2_ep_tx_flags(ep));
}

static ssize_t sm2_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
	void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_sendmsg(ep, iov, desc, count, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   sm2_ep_tx_flags(ep));
}

static ssize_t sm2_tsendmsg(struct fid_ep *ep_fid,
			    const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, msg->tag, msg->data, msg->context,
				   ofi_op_tagged,
				   flags | ep->util_ep.tx_msg_flags);
}

static ssize_t sm2_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			   fi_addr_t dest_addr, uint64_t tag)
{
	return sm2_generic_inject(ep_fid, buf, len, dest_addr, tag, 0,
				  ofi_op_tagged, 0);
}

static ssize_t sm2_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
{
	struct sm2_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return sm2_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, tag, data,
				   context, ofi_op_tagged,
				   FI_REMOTE_CQ_DATA | sm2_ep_tx_flags(ep));
}

static ssize_t sm2_tinjectdata(struct fid_ep *ep_fid, const void *buf,
			       size_t len, uint64_t data, fi_addr_t dest_addr,
			       uint64_t tag)
{
	return sm2_generic_inject(ep_fid, buf, len, dest_addr, tag, data,
				  ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

struct fi_ops_tagged sm2_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = sm2_trecv,
	.recvv = sm2_trecvv,
	.recvmsg = sm2_trecvmsg,
	.send = sm2_tsend,
	.sendv = sm2_tsendv,
	.sendmsg = sm2_tsendmsg,
	.inject = sm2_tinject,
	.senddata = sm2_tsenddata,
	.injectdata = sm2_tinjectdata,
};

struct fi_ops_tagged sm2_no_recv_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = sm2_tsend,
	.sendv = sm2_tsendv,
	.sendmsg = sm2_tsendmsg,
	.inject = sm2_tinject,
	.senddata = sm2_tsenddata,
	.injectdata = sm2_tinjectdata,
};

struct fi_ops_tagged sm2_srx_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = sm2_srx_trecv,
	.recvv = sm2_srx_trecvv,
	.recvmsg = sm2_srx_trecvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = sm2_tsendmsg,
	.inject = sm2_tinject,
	.senddata = sm2_tsenddata,
	.injectdata = sm2_tinjectdata,
};