/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include "config.h"

#include "fi_verbs.h"


static ssize_t
fi_ibv_msg_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct fi_ibv_wre *wre;
	struct ibv_sge *sge = NULL;
	size_t i;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	assert(_ep->rcq);

	fastlock_acquire(&_ep->wre_lock);
	wre = util_buf_alloc(_ep->wre_pool);
	if (OFI_UNLIKELY(!wre)) {
		fastlock_release(&_ep->wre_lock);
		return -FI_EAGAIN;
	}
	memset(wre, 0, sizeof(*wre));
	dlist_insert_tail(&wre->entry, &_ep->wre_list);
	fastlock_release(&_ep->wre_lock);

	wre->ep = _ep;
	wre->context = msg->context;
	wre->wr.type = IBV_RECV_WR;
	wre->wr.rwr.wr_id = (uintptr_t)wre;
	wre->wr.rwr.next = NULL;

	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t)msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t)msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t)(uintptr_t)(msg->desc[i]);
		}
	}
	wre->wr.rwr.sg_list = sge;
	wre->wr.rwr.num_sge = msg->iov_count;

	return FI_IBV_INVOKE_POST(recv, recv, _ep->id->qp, &wre->wr.rwr,
				  FI_IBV_RELEASE_WRE(_ep, wre));
}

static ssize_t
fi_ibv_msg_ep_recv(struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct fi_msg msg;

	iov.iov_base = buf;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_msg_ep_recvmsg(ep, &msg, 0);
}

static ssize_t
fi_ibv_msg_ep_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_msg_ep_recvmsg(ep, &msg, 0);
}

static ssize_t
fi_ibv_msg_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
		wr.imm_data = htonl((uint32_t)msg->data);
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_msg(ep, &wr, msg, flags);
}

static ssize_t
fi_ibv_msg_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	wr.opcode = IBV_WR_SEND;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_INJECT(ep, len) | VERBS_COMP(ep);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context);
}

static ssize_t
fi_ibv_msg_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		    void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_INJECT(ep, len) | VERBS_COMP(ep);

	return fi_ibv_send_buf(ep, &wr, buf, len, desc, context);
}

static ssize_t
fi_ibv_msg_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	wr.opcode = IBV_WR_SEND;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_send_iov(ep, &wr, iov, desc, count, context);
}

static ssize_t fi_ibv_msg_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	wr.opcode = IBV_WR_SEND;
	wr.send_flags = IBV_SEND_INLINE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf_inline(ep, &wr, buf, len);
}

static ssize_t fi_ibv_msg_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		    uint64_t data, fi_addr_t dest_addr)
{
	struct fi_ibv_msg_ep *ep;
	struct ibv_send_wr wr = { 0 };

	wr.opcode = IBV_WR_SEND_WITH_IMM;
	wr.imm_data = htonl((uint32_t)data);
	wr.send_flags = IBV_SEND_INLINE;

	ep = container_of(ep_fid, struct fi_ibv_msg_ep, ep_fid);

	return fi_ibv_send_buf_inline(ep, &wr, buf, len);
}

struct fi_ops_msg fi_ibv_msg_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_msg_ep_recv,
	.recvv = fi_ibv_msg_ep_recvv,
	.recvmsg = fi_ibv_msg_ep_recvmsg,
	.send = fi_ibv_msg_ep_send,
	.sendv = fi_ibv_msg_ep_sendv,
	.sendmsg = fi_ibv_msg_ep_sendmsg,
	.inject = fi_ibv_msg_ep_inject,
	.senddata = fi_ibv_msg_ep_senddata,
	.injectdata = fi_ibv_msg_ep_injectdata,
};

struct fi_ops_msg fi_ibv_msg_srq_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_ibv_msg_ep_send,
	.sendv = fi_ibv_msg_ep_sendv,
	.sendmsg = fi_ibv_msg_ep_sendmsg,
	.inject = fi_ibv_msg_ep_inject,
	.senddata = fi_ibv_msg_ep_senddata,
	.injectdata = fi_ibv_msg_ep_injectdata,
};
