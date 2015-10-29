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
fi_ibv_msg_ep_atomic_write(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype, enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (op) {
	case FI_ATOMIC_WRITE:
		wr.opcode = IBV_WR_RDMA_WRITE;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}
	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	sge.addr = (uintptr_t) buf;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) desc;

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE;
	if (VERBS_INJECT(_ep, sizeof(uint64_t)))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_atomic_writev(struct fid_ep *ep,
                        const struct fi_ioc *iov, void **desc, size_t count,
                        fi_addr_t dest_addr, uint64_t addr, uint64_t key,
                        enum fi_datatype datatype, enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return fi_ibv_msg_ep_atomic_write(ep, iov->addr, count, desc[0],
			dest_addr, addr, key, datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_writemsg(struct fid_ep *ep,
                        const struct fi_msg_atomic *msg, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch (msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (msg->op) {
	case FI_ATOMIC_WRITE:
		if (flags & FI_REMOTE_CQ_DATA) {
			wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
			wr.imm_data = htonl((uint32_t) msg->data);
		} else {
			wr.opcode = IBV_WR_RDMA_WRITE;
		}
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}
	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	sge.addr = (uintptr_t) msg->msg_iov->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	if (!(flags & FI_INJECT)) {
		sge.lkey = (uint32_t) (uintptr_t) msg->desc[0];
	}

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE;
	if (VERBS_INJECT_FLAGS(_ep, sizeof(uint64_t), flags))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP_FLAGS(_ep, flags))
		wr.send_flags |= IBV_SEND_SIGNALED;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_atomic_readwrite(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, void *result, void *result_desc,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = addr;
		wr.wr.atomic.compare_add = (uintptr_t) buf;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;
		break;
	default:
		return -ENOSYS;
	}

	sge.addr = (uintptr_t) result;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (VERBS_INJECT(_ep, sizeof(uint64_t)))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_atomic_readwritev(struct fid_ep *ep, const struct fi_ioc *iov,
			void **desc, size_t count,
			struct fi_ioc *resultv, void **result_desc,
			size_t result_count, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

        return fi_ibv_msg_ep_atomic_readwrite(ep, iov->addr, count,
			desc[0], resultv->addr, result_desc[0],
			dest_addr, addr, key, datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_readwritemsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count, uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch (msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	switch (msg->op) {
	case FI_ATOMIC_READ:
		wr.opcode = IBV_WR_RDMA_READ;
		wr.wr.rdma.remote_addr = msg->rma_iov->addr;
		wr.wr.rdma.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	case FI_SUM:
		wr.opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
		wr.wr.atomic.remote_addr = msg->rma_iov->addr;
		wr.wr.atomic.compare_add = (uintptr_t) msg->addr;
		wr.wr.atomic.swap = 0;
		wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;
		break;
	default:
		return -ENOSYS;
	}

	sge.addr = (uintptr_t) resultv->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	if (!(flags & FI_INJECT)) {
		sge.lkey = (uint32_t) (uintptr_t) result_desc[0];
	}

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (VERBS_INJECT_FLAGS(_ep, sizeof(uint64_t), flags))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP_FLAGS(_ep, flags))
		wr.send_flags |= IBV_SEND_SIGNALED;
	if (flags & FI_REMOTE_CQ_DATA)
		wr.imm_data = htonl((uint32_t) msg->data);

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_atomic_compwrite(struct fid_ep *ep, const void *buf, size_t count,
			void *desc, const void *compare,
			void *compare_desc, void *result,
			void *result_desc,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key,
			enum fi_datatype datatype,
			enum fi_op op, void *context)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (op != FI_CSWAP)
		return -ENOSYS;

	if (count != 1)
		return -FI_E2BIG;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = addr;
	wr.wr.atomic.compare_add = (uintptr_t) compare;
	wr.wr.atomic.swap = (uintptr_t) buf;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) key;

	sge.addr = (uintptr_t) result;
	sge.length = (uint32_t) sizeof(uint64_t);
	sge.lkey = (uint32_t) (uintptr_t) result_desc;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (VERBS_INJECT(_ep, sizeof(uint64_t)))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP(_ep))
		wr.send_flags |= IBV_SEND_SIGNALED;

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static ssize_t
fi_ibv_msg_ep_atomic_compwritev(struct fid_ep *ep, const struct fi_ioc *iov,
				void **desc, size_t count,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv, void **result_desc,
				size_t result_count,
				fi_addr_t dest_addr, uint64_t addr, uint64_t key,
				enum fi_datatype datatype,
				enum fi_op op, void *context)
{
	if (iov->count != 1)
		return -FI_E2BIG;

	return fi_ibv_msg_ep_atomic_compwrite(ep, iov->addr, count, desc[0],
				comparev->addr, compare_desc[0], resultv->addr,
				result_desc[0], dest_addr, addr, key,
                        	datatype, op, context);
}

static ssize_t
fi_ibv_msg_ep_atomic_compwritemsg(struct fid_ep *ep,
				const struct fi_msg_atomic *msg,
				const struct fi_ioc *comparev,
				void **compare_desc, size_t compare_count,
				struct fi_ioc *resultv,
				void **result_desc, size_t result_count,
				uint64_t flags)
{
	struct fi_ibv_msg_ep *_ep;
	struct ibv_send_wr wr, *bad;
	struct ibv_sge sge;

	if (msg->op != FI_CSWAP)
		return -ENOSYS;

	if (msg->iov_count != 1 || msg->msg_iov->count != 1)
		return -FI_E2BIG;

	switch(msg->datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	wr.opcode = IBV_WR_ATOMIC_CMP_AND_SWP;
	wr.wr.atomic.remote_addr = msg->rma_iov->addr;
	wr.wr.atomic.compare_add = (uintptr_t) comparev->addr;
	wr.wr.atomic.swap = (uintptr_t) msg->addr;
	wr.wr.atomic.rkey = (uint32_t) (uintptr_t) msg->rma_iov->key;

	sge.addr = (uintptr_t) resultv->addr;
	sge.length = (uint32_t) sizeof(uint64_t);
	if (!(flags & FI_INJECT)) {
		sge.lkey = (uint32_t) (uintptr_t) result_desc[0];
	}

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	wr.send_flags = IBV_SEND_FENCE; 
	if (VERBS_INJECT_FLAGS(_ep, sizeof(uint64_t), flags))
		wr.send_flags |= IBV_SEND_INLINE;
	if (VERBS_COMP_FLAGS(_ep, flags))
		wr.send_flags |= IBV_SEND_SIGNALED;
	if (flags & FI_REMOTE_CQ_DATA)
		wr.imm_data = htonl((uint32_t) msg->data);

	return fi_ibv_post_send(_ep->id->qp, &wr, &bad);
}

static int
fi_ibv_msg_ep_atomic_writevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	switch (op) {
	case FI_ATOMIC_WRITE:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
fi_ibv_msg_ep_atomic_readwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	switch (op) {
	case FI_ATOMIC_READ:
	case FI_SUM:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}

static int
fi_ibv_msg_ep_atomic_compwritevalid(struct fid_ep *ep, enum fi_datatype datatype,
				enum fi_op op, size_t *count)
{
	if (op != FI_CSWAP)
		return -FI_ENOSYS;

	switch (datatype) {
	case FI_INT64:
	case FI_UINT64:
#if __BITS_PER_LONG == 64
	case FI_DOUBLE:
	case FI_FLOAT:
#endif
		break;
	default:
		return -FI_EINVAL;
	}

	if (count)
		*count = 1;
	return 0;
}


struct fi_ops_atomic fi_ibv_msg_ep_atomic_ops = {
	.size		= sizeof(struct fi_ops_atomic),
	.write		= fi_ibv_msg_ep_atomic_write,
	.writev		= fi_ibv_msg_ep_atomic_writev,
	.writemsg	= fi_ibv_msg_ep_atomic_writemsg,
	.inject		= fi_no_atomic_inject,
	.readwrite	= fi_ibv_msg_ep_atomic_readwrite,
	.readwritev	= fi_ibv_msg_ep_atomic_readwritev,
	.readwritemsg	= fi_ibv_msg_ep_atomic_readwritemsg,
	.compwrite	= fi_ibv_msg_ep_atomic_compwrite,
	.compwritev	= fi_ibv_msg_ep_atomic_compwritev,
	.compwritemsg	= fi_ibv_msg_ep_atomic_compwritemsg,
	.writevalid	= fi_ibv_msg_ep_atomic_writevalid,
	.readwritevalid	= fi_ibv_msg_ep_atomic_readwritevalid,
	.compwritevalid = fi_ibv_msg_ep_atomic_compwritevalid
};
