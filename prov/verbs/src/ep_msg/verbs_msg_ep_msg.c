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

#include <prov/verbs/src/fi_verbs.h>

static ssize_t
fi_ibv_msg_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge *sge = NULL;
	ssize_t ret;
	size_t i;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}

	}
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count;

	ret = ibv_post_recv(_ep->id->qp, &wr, &bad);
	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
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
fi_ibv_msg_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i, len;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.num_sge = msg->iov_count;
	wr.send_flags = 0;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (len = 0, i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			len += sge[i].length;
		}
		if (VERBS_INJECT_FLAGS(_ep, len, flags)) {
			wr.send_flags |= IBV_SEND_INLINE;
		} else {
			for (i = 0; i < msg->iov_count; i++) {
				sge[i].lkey = (uint32_t)(uintptr_t)(msg->desc[i]);
			}
		}

		wr.sg_list = sge;
	}
	if (VERBS_COMP_FLAGS(_ep, flags))
		wr.send_flags |= IBV_SEND_SIGNALED;

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_SEND_WITH_IMM;
                wr.imm_data = htonl((uint32_t) msg->data);
	} else {
		wr.opcode = IBV_WR_SEND;
	}

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_send(struct fid_ep *ep, const void *buf, size_t len,
		void *desc, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct iovec iov;
	struct fi_msg msg;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
        return fi_ibv_msg_ep_sendmsg(ep, &msg, _ep->info->tx_attr->op_flags);
}

static ssize_t
fi_ibv_msg_ep_senddata(struct fid_ep *ep, const void *buf, size_t len,
		    void *desc, uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct iovec iov;
	struct fi_msg msg;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.data = data;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	return fi_ibv_msg_ep_sendmsg(ep, &msg,
                                        FI_REMOTE_CQ_DATA | _ep->info->tx_attr->op_flags);
}

static ssize_t
fi_ibv_msg_ep_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
        return fi_ibv_msg_ep_sendmsg(ep, &msg, _ep->info->tx_attr->op_flags);
}

struct fi_ops_msg fi_ibv_msg_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_msg_ep_recv,
	.recvv = fi_ibv_msg_ep_recvv,
	.recvmsg = fi_ibv_msg_ep_recvmsg,
	.send = fi_ibv_msg_ep_send,
	.sendv = fi_ibv_msg_ep_sendv,
	.sendmsg = fi_ibv_msg_ep_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_ibv_msg_ep_senddata,
	.injectdata = fi_no_msg_injectdata,
};
