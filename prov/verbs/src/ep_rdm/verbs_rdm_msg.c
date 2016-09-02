/*
 * Copyright (c) 2013-2016 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#include "fi_enosys.h"

#include "verbs_rdm.h"

static ssize_t fi_ibv_rdm_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
				  uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t
fi_ibv_rdm_recvv(struct fid_ep *ep, const struct iovec *iov,
		 void **desc, size_t count, fi_addr_t src_addr,
		 void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	const struct fi_msg msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_recvmsg(ep, &msg,
		(ep_rdm->rx_selective_completion ? 0ULL : FI_COMPLETION));
}

static ssize_t
fi_ibv_rdm_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	const struct iovec iov = {
		.iov_base = buf,
		.iov_len = len
	};
	return fi_ibv_rdm_recvv(ep, &iov, &desc, 1, src_addr, context);
}

static ssize_t fi_ibv_rdm_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
				  uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t fi_ibv_rdm_sendv(struct fid_ep *ep, const struct iovec *iov,
				void **desc, size_t count, fi_addr_t dest_addr,
				void *context)
{
	struct fi_ibv_rdm_ep *ep_rdm =
		container_of(ep, struct fi_ibv_rdm_ep, ep_fid);

	const struct fi_msg msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.context = context,
		.data = 0
	};

	return fi_ibv_rdm_sendmsg(ep, &msg,
		(ep_rdm->tx_selective_completion ? 0ULL : FI_COMPLETION));
}

static ssize_t fi_ibv_rdm_send(struct fid_ep *ep, const void *buf, size_t len,
			       void *desc, fi_addr_t dest_addr, void *context)
{
	const struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len
	};
	return fi_ibv_rdm_sendv(ep, &iov, &desc, 1, dest_addr, context);
}

static ssize_t fi_ibv_rdm_inject(struct fid_ep *ep, const void *buf, size_t len,
				 fi_addr_t dest_addr)
{
	return -FI_ENOSYS;
}

static ssize_t fi_ibv_rdm_senddata(struct fid_ep *ep, const void *buf,
				   size_t len, void *desc, uint64_t data,
				   fi_addr_t dest_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t fi_ibv_rdm_injectdata(struct fid_ep *ep, const void *buf,
				     size_t len, uint64_t data,
				     fi_addr_t dest_addr)
{
	return -FI_ENOSYS;
}

static struct fi_ops_msg fi_ibv_rdm_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_rdm_recv,
	.recvv = fi_ibv_rdm_recvv,
	.recvmsg = fi_ibv_rdm_recvmsg,
	.send = fi_ibv_rdm_send,
	.sendv = fi_ibv_rdm_sendv,
	.sendmsg = fi_ibv_rdm_sendmsg,
	.inject = fi_ibv_rdm_inject,
	.senddata = fi_ibv_rdm_senddata,
	.injectdata = fi_ibv_rdm_injectdata
};

struct fi_ops_msg *fi_ibv_rdm_ep_ops_msg()
{
	return &fi_ibv_rdm_ep_msg_ops;
}
