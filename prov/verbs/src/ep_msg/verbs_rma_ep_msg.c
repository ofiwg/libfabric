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
fi_ibv_msg_ep_rma_write(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.send_flags = 0;
	if (VERBS_INJECT(_ep, len))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t count, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t len, i;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sge = alloca(count * sizeof(struct ibv_sge));

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = count;
	wr.opcode = IBV_WR_RDMA_WRITE;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	for (len = 0, i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		len += iov[i].iov_len;
                if (!(_ep->info->tx_attr->op_flags & FI_INJECT))
			sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}

	wr.send_flags = 0;
	if (VERBS_INJECT(_ep, len))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge=NULL;
	size_t i, len;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
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
	}
	if (VERBS_COMP_FLAGS(_ep, flags))
		wr.send_flags |= IBV_SEND_SIGNALED;
	wr.sg_list = sge;

	wr.opcode = IBV_WR_RDMA_WRITE;
	if (flags & FI_REMOTE_CQ_DATA) {
		wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
                wr.imm_data = htonl((uint32_t) msg->data);
	}

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_read(struct fid_ep *ep, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_COMP_READ(_ep) ? IBV_SEND_SIGNALED : 0;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		     size_t count, fi_addr_t src_addr,
		     uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i;

	sge = alloca(count * sizeof(struct ibv_sge));
	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = sge;
	wr.num_sge = count;
	wr.opcode = IBV_WR_RDMA_READ;
	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	for (i = 0; i < count; i++) {
		sge[i].addr = (uintptr_t) iov[i].iov_base;
		sge[i].length = (uint32_t) iov[i].iov_len;
		sge[i].lkey = (uint32_t) (uintptr_t) desc[i];
	}

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	wr.send_flags = VERBS_COMP_READ(_ep) ? IBV_SEND_SIGNALED : 0;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge *sge;
	size_t i;

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}
		wr.sg_list = sge;
	}
	wr.num_sge = msg->iov_count;
	wr.opcode = IBV_WR_RDMA_READ;

	wr.wr.rdma.remote_addr = msg->rma_iov->addr;
	wr.wr.rdma.rkey = (uint32_t) msg->rma_iov->key;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (VERBS_COMP_READ_FLAGS(_ep, flags)) {
		wr.send_flags = IBV_SEND_SIGNALED;
	} else {
		wr.send_flags = 0;
	}

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_rma_writedata(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, uint64_t data, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) len;
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	wr.send_flags = 0;
	if (VERBS_INJECT(_ep, len))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;
        wr.imm_data = htonl((uint32_t) data);

	wr.wr.rdma.remote_addr = addr;
	wr.wr.rdma.rkey = (uint32_t) key;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

struct fi_ops_rma fi_ibv_msg_ep_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_ibv_msg_ep_rma_read,
	.readv = fi_ibv_msg_ep_rma_readv,
	.readmsg = fi_ibv_msg_ep_rma_readmsg,
	.write = fi_ibv_msg_ep_rma_write,
	.writev = fi_ibv_msg_ep_rma_writev,
	.writemsg = fi_ibv_msg_ep_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_ibv_msg_ep_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};
