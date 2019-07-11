/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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
#include "mlx.h"
#include "mlx_core.h"

static ssize_t mlx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	uint64_t ignore = MLX_AUX_TAG_MASK;
	struct fi_msg_tagged tmsg = {
		.msg_iov = msg->msg_iov,
		.desc = msg->desc,
		.iov_count = msg->iov_count,
		.addr = msg->addr,
		.tag = 0,
		.ignore = 0,
		.context = msg->context,
		.data = 0,
	};

	if (flags & (FI_REMOTE_CQ_DATA | FI_PEEK | FI_CLAIM)) {
		return -FI_EBADFLAGS;
	}

	if (msg->addr != FI_ADDR_UNSPEC) {
		return -FI_EADDRNOTAVAIL;
	}

	return mlx_do_recvmsg(ep, &tmsg, flags, tag, ignore, 1);
}

static ssize_t mlx_recv(struct fid_ep *ep, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	uint64_t ignore = MLX_AUX_TAG_MASK;

	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};

	struct fi_msg_tagged tmsg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = src_addr,
		.context = context,
		.data = 0,
	};
	if (src_addr != FI_ADDR_UNSPEC) {
		return -FI_EADDRNOTAVAIL;
	}

	return mlx_do_recvmsg(ep, &tmsg, 0, tag, ignore, 1);
}

static ssize_t mlx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t src_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	uint64_t ignore = MLX_AUX_TAG_MASK;

	struct fi_msg_tagged tmsg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.context = context,
		.data = 0,
	};
	if (src_addr != FI_ADDR_UNSPEC) {
		return -FI_EADDRNOTAVAIL;
	}

	return mlx_do_recvmsg(ep, &tmsg, 0, tag, ignore, 1);
}


ssize_t mlx_inject(struct fid_ep *ep, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	return mlx_do_inject(ep, buf, len, dest_addr, tag);
}

ssize_t mlx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
		uint64_t flags)
{
	uint64_t tag = MLX_EP_MSG_TAG;

	struct fi_msg_tagged tmsg = {
		.msg_iov = msg->msg_iov,
		.desc = msg->desc,
		.iov_count = msg->iov_count,
		.addr = msg->addr,
		.tag = 0,
		.ignore = 0,
		.context = msg->context,
		.data = 0,
	};

	if(flags & FI_REMOTE_CQ_DATA) {
		return -FI_EBADFLAGS;
	}

	return mlx_do_sendmsg(ep, &tmsg, flags, tag, 1);
}


static ssize_t mlx_send(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct fi_msg_tagged tmsg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = dest_addr,
		.context = context,
		.data = 0,
	};

	return mlx_do_sendmsg(ep, &tmsg, 0, tag, 1);
}

static ssize_t mlx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	struct fi_msg_tagged tmsg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.context = context,
		.data = 0,
	};

	return mlx_do_sendmsg(ep, &tmsg, 0, tag, 1);
}


struct fi_ops_msg mlx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = mlx_recv,
	.recvv = mlx_recvv,
	.recvmsg = mlx_recvmsg,
	.send = mlx_send,
	.sendv = mlx_sendv,
	.sendmsg = mlx_sendmsg,
	.inject = mlx_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};
