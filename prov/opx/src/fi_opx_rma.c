/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021 Cornelis Networks.
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
#include <ofi.h>

#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_rma.h"
#include "rdma/opx/fi_opx_eq.h"
#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_internal.h"

#include <ofi_enosys.h>
#include <errno.h>

/* Delivery completion callback for simple sends.
 * This is only used for simple sends but it is included here
 * due to the complex dependencies of all the inlined code.
 *
 * This differs from the other completion callback in that it adds
 * the message tag from the original message to the completion item.
 */
void fi_opx_delivery_complete(struct fi_opx_completion_counter *cc)
{
	if (cc->cntr) {
		ofi_atomic_inc64(&cc->cntr->std);
	}
	if (cc->cq && cc->context) {
		assert(cc->context);
		union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
		opx_context->next = NULL;
		opx_context->len = 0;
		opx_context->buf = NULL;
		opx_context->byte_counter = 0;
		opx_context->tag = cc->tag;

		fi_opx_cq_enqueue_completed(cc->cq, cc->context, 0);
	}
	ofi_buf_free(cc);
}

/* Delivery completion callback for atomics, RMA operations.
 *
 * This differs from the other completion callback in that it ignores
 * any message tag.
 */
void fi_opx_hit_zero(struct fi_opx_completion_counter *cc)
{
	if (cc->cntr) {
		ofi_atomic_inc64(&cc->cntr->std);
	}
	if (cc->cq && cc->context) {
		assert(cc->context);
		union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
		opx_context->next = NULL;
		opx_context->len = 0;
		opx_context->buf = NULL;
		opx_context->byte_counter = 0;
		opx_context->tag = 0;

		fi_opx_cq_enqueue_completed(cc->cq, cc->context, 0);
	}
	ofi_buf_free(cc);
}

inline int fi_opx_check_rma(struct fi_opx_ep *opx_ep)
{
	if (!opx_ep)
		return -FI_EINVAL;
	if (opx_ep->state != FI_OPX_EP_INITITALIZED_ENABLED)
		return -FI_EINVAL;

	const enum fi_av_type av_type = opx_ep->av->type;

	if (av_type == FI_AV_UNSPEC)
		return -FI_EINVAL;
	if (av_type == FI_AV_MAP && opx_ep->av->type != FI_AV_MAP)
		return -FI_EINVAL;
	if (av_type == FI_AV_TABLE && opx_ep->av->type != FI_AV_TABLE)
		return -FI_EINVAL;

	return 0;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_inject_write_internal(struct fid_ep *ep, const void *buf, size_t len,
				     fi_addr_t dst_addr, uint64_t addr_offset, uint64_t key,
				     int lock_required, const enum fi_av_type av_type,
				     const uint64_t caps,
				     const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}
	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[dst_addr].fi :
							       dst_addr };

	assert(dst_addr != FI_ADDR_UNSPEC);

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			opx_dst_addr.uid.lid,
			opx_dst_addr.hfi1_rx,
			opx_dst_addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME);
		return -FI_EAGAIN;
	}

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = len;
	cc->cntr = opx_ep->write_cntr;
	cc->cq = NULL;
	cc->context = NULL;
	cc->hit_zero = fi_opx_hit_zero;

	fi_opx_write_internal(opx_ep, buf, len, opx_dst_addr, addr_offset, key,
				NULL, cc, FI_VOID, FI_NOOP,
				opx_ep->tx->op_flags | FI_INJECT,
				lock_required, caps, reliability);

	return 0;
}

inline ssize_t fi_opx_inject_write_generic(struct fid_ep *ep, const void *buf, size_t len,
					   fi_addr_t dst_addr, uint64_t addr_offset, uint64_t key,
					   int lock_required, const enum fi_av_type av_type,
					   const uint64_t caps,
					   const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_inject_write_internal(ep, buf, len, dst_addr, addr_offset, key,
						  FI_OPX_LOCK_NOT_REQUIRED, av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		     fi_addr_t dst_addr, uint64_t addr_offset, uint64_t key,
		     void *context, int lock_required, const enum fi_av_type av_type,
		     const uint64_t caps,
		     const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}
	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[dst_addr].fi :
							       dst_addr };

	assert(dst_addr != FI_ADDR_UNSPEC);
	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = len;
	cc->cntr = opx_ep->write_cntr;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) || ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE)  == FI_DELIVERY_COMPLETE)) ? opx_ep->rx->cq : NULL;
	cc->context = context;
	union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_WRITE;

	cc->hit_zero = fi_opx_hit_zero;

	fi_opx_write_internal(opx_ep, buf, len, opx_dst_addr, addr_offset, key,
			(union fi_opx_context *)context, cc, FI_VOID, FI_NOOP, opx_ep->tx->op_flags,
			lock_required, caps, reliability);

	return 0;
}

inline ssize_t fi_opx_write_generic(struct fid_ep *ep, const void *buf, size_t len, void *desc,
				    fi_addr_t dst_addr, uint64_t addr_offset, uint64_t key,
				    void *context, int lock_required, const enum fi_av_type av_type,
				    const uint64_t caps,
				    const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_write(ep, buf, len, desc, dst_addr, addr_offset, key, context,
				  FI_OPX_LOCK_NOT_REQUIRED, av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_writev_internal(struct fid_ep *ep, const struct iovec *iov, void **desc,
			       size_t count, fi_addr_t dst_addr, uint64_t addr_offset,
			       uint64_t key, void *context, int lock_required,
			       const enum fi_av_type av_type, const uint64_t caps,
			       const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	assert(dst_addr != FI_ADDR_UNSPEC);
	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[dst_addr].fi :
							       dst_addr };

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	size_t index;
	cc->byte_counter = 0;
	for (index = 0; index < count; ++index) {
		cc->byte_counter += iov[index].iov_len;
	}
	cc->cntr = opx_ep->write_cntr;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) || ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE)  == FI_DELIVERY_COMPLETE)) ? opx_ep->rx->cq : NULL;
	cc->context = context;
	union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_WRITE;

	cc->hit_zero = fi_opx_hit_zero;

	for (index = 0; index < count; ++index) {
		size_t len = iov[index].iov_len;
		void *buf = iov[index].iov_base;
		fi_opx_write_internal(opx_ep, buf, len, opx_dst_addr, addr_offset, key,
							  (union fi_opx_context *)context, cc, FI_VOID, FI_NOOP, 0, lock_required, caps,
				      reliability);

		addr_offset += len;
	}

	return 0;
}

inline ssize_t fi_opx_writev_generic(struct fid_ep *ep, const struct iovec *iov, void **desc,
				     size_t count, fi_addr_t dst_addr, uint64_t addr_offset,
				     uint64_t key, void *context, int lock_required,
				     const enum fi_av_type av_type, const uint64_t caps,
				     const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_writev_internal(ep, iov, desc, count, dst_addr, addr_offset, key, context,
					    FI_OPX_LOCK_NOT_REQUIRED, av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_writemsg_internal(struct fid_ep *ep, const struct fi_msg_rma *msg,
			         uint64_t flags, int lock_required,
			         const enum fi_av_type av_type, const uint64_t caps,
			         const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep;
	opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	assert(msg->addr != FI_ADDR_UNSPEC);
	/* constant compile-time expression */
	const union fi_opx_addr opx_dst_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[msg->addr].fi :
							       msg->addr };

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	size_t index;
	cc->byte_counter = 0;
	for(index=0; index < msg->iov_count; index++) {
		cc->byte_counter += msg->msg_iov[index].iov_len;
	}

	cc->cntr = opx_ep->write_cntr;
	cc->cq = ((flags & FI_COMPLETION) == FI_COMPLETION) ? opx_ep->rx->cq : NULL;
	cc->context = msg->context;
	union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_WRITE;

	cc->hit_zero = fi_opx_hit_zero;

	size_t rma_iov_index = 0;
	const size_t rma_iov_count = msg->rma_iov_count;
	uint64_t rma_iov_bytes = msg->rma_iov[rma_iov_index].len;
	uint64_t rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
	uint64_t rma_iov_key = msg->rma_iov[rma_iov_index].key;

	size_t msg_iov_index = 0;
	const size_t msg_iov_count = msg->iov_count;
	uint64_t msg_iov_bytes = msg->msg_iov[msg_iov_index].iov_len;
	uintptr_t msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].iov_base;

	while (msg_iov_bytes != 0 && rma_iov_bytes != 0) {
		size_t len = (msg_iov_bytes <= rma_iov_bytes) ? msg_iov_bytes : rma_iov_bytes;
		fi_opx_write_internal(opx_ep, (void *)msg_iov_vaddr, len, opx_dst_addr,
				      rma_iov_addr, rma_iov_key, NULL, cc, FI_VOID, FI_NOOP, 0, lock_required, caps,
				      reliability);

		msg_iov_bytes -= len;
		msg_iov_vaddr += len;

		if ((msg_iov_bytes == 0) && ((msg_iov_index + 1) < msg_iov_count)) {
			++msg_iov_index;
			msg_iov_bytes = msg->msg_iov[msg_iov_index].iov_len;
			msg_iov_vaddr = (uintptr_t)msg->msg_iov[msg_iov_index].iov_base;
		}

		rma_iov_bytes -= len;
		rma_iov_addr += len;

		if ((rma_iov_bytes == 0) && ((rma_iov_index + 1) < rma_iov_count)) {
			++rma_iov_index;
			rma_iov_bytes = msg->rma_iov[rma_iov_index].len;
			rma_iov_addr = msg->rma_iov[rma_iov_index].addr;
			rma_iov_key = msg->rma_iov[rma_iov_index].key;
		}
	}

	return 0;
}

inline ssize_t fi_opx_writemsg_generic(struct fid_ep *ep, const struct fi_msg_rma *msg,
				       uint64_t flags, int lock_required,
				       const enum fi_av_type av_type, const uint64_t caps,
				       const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_writemsg_internal(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED,
					      av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_read_internal(struct fid_ep *ep, void *buf, size_t len, void *desc,
			     fi_addr_t src_addr, uint64_t addr_offset, uint64_t key,
			     void *context, int lock_required, const enum fi_av_type av_type,
			     const uint64_t caps, const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	struct iovec iov;
	iov.iov_base = buf;
	iov.iov_len = len;

	assert(src_addr != FI_ADDR_UNSPEC);
	const union fi_opx_addr opx_addr = { .fi = (av_type == FI_AV_TABLE) ?
							   opx_ep->tx->av_addr[src_addr].fi :
							   src_addr };

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	cc->byte_counter = len;
	cc->cntr = opx_ep->read_cntr;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) || ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE)  == FI_DELIVERY_COMPLETE)) ? opx_ep->rx->cq : NULL;
	cc->context = context;
	union fi_opx_context * opx_context = (union fi_opx_context *)cc->context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_READ;

	cc->hit_zero = fi_opx_hit_zero;

	fi_opx_readv_internal(opx_ep, &iov, 1, opx_addr, &addr_offset, &key,
			      (union fi_opx_context *)context, opx_ep->tx->op_flags, opx_ep->rx->cq,
						  opx_ep->read_cntr, cc, FI_VOID, FI_NOOP, FI_OPX_HFI_DPUT_OPCODE_GET, lock_required,
			      caps, reliability);

	return 0;
}

inline ssize_t fi_opx_read_generic(struct fid_ep *ep, void *buf, size_t len, void *desc,
				   fi_addr_t src_addr, uint64_t addr_offset, uint64_t key,
				   void *context, int lock_required, const enum fi_av_type av_type,
				   const uint64_t caps, const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_read_internal(ep, buf, len, desc, src_addr, addr_offset, key, context,
					  FI_OPX_LOCK_NOT_REQUIRED, av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		     size_t count, fi_addr_t src_addr, uint64_t addr_offset,
		     uint64_t key, void *context, int lock_required,
		     const enum fi_av_type av_type, const uint64_t caps,
		     const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	assert(src_addr != FI_ADDR_UNSPEC);
	const union fi_opx_addr opx_addr = { .fi = (av_type == FI_AV_TABLE) ?
							   opx_ep->tx->av_addr[src_addr].fi :
							   src_addr };

	union fi_opx_context *opx_context = (union fi_opx_context *)context;
	const uint64_t tx_op_flags = opx_ep->tx->op_flags;

	uint64_t addr_v[8] = { addr_offset, addr_offset, addr_offset, addr_offset,
			       addr_offset, addr_offset, addr_offset, addr_offset };
	uint64_t key_v[8] = { key, key, key, key, key, key, key, key };

	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	size_t index;
	cc->byte_counter = 0;
	for(index=0; index < count; index++) {
		cc->byte_counter += iov[index].iov_len;
	}
	cc->cntr = opx_ep->read_cntr;
	cc->cq = (((opx_ep->tx->op_flags & FI_COMPLETION) == FI_COMPLETION) || ((opx_ep->tx->op_flags & FI_DELIVERY_COMPLETE)  == FI_DELIVERY_COMPLETE)) ? opx_ep->rx->cq : NULL;
	cc->context = context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_READ;
	cc->hit_zero = fi_opx_hit_zero;

	/* max 8 descriptors (iovecs) per readv_internal */
	const size_t full_count = count >> 3;
	for (index = 0; index < full_count; index += 8) {
		fi_opx_readv_internal(opx_ep, &iov[index], 8, opx_addr, addr_v, key_v, NULL, 0,
							  NULL, NULL, cc, FI_VOID, FI_NOOP, FI_OPX_HFI_DPUT_OPCODE_GET, lock_required,
				      caps, reliability);
	}

	/* if 'partial_ndesc' is zero, the fi_opx_readv_internal() will fence */
	const size_t partial_ndesc = count & 0x07ull;
	fi_opx_readv_internal(opx_ep, &iov[index], partial_ndesc, opx_addr, addr_v, key_v,
			      opx_context, tx_op_flags, opx_ep->rx->cq, opx_ep->read_cntr, cc,
						 FI_VOID, FI_NOOP,
			      FI_OPX_HFI_DPUT_OPCODE_GET, lock_required, caps, reliability);

	return 0;
}

inline ssize_t fi_opx_readv_generic(struct fid_ep *ep, const struct iovec *iov, void **desc,
				    size_t count, fi_addr_t src_addr, uint64_t addr_offset,
				    uint64_t key, void *context, int lock_required,
				    const enum fi_av_type av_type, const uint64_t caps,
				    const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_readv(ep, iov, desc, count, src_addr, addr_offset, key, context,
				  FI_OPX_LOCK_NOT_REQUIRED, av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
ssize_t fi_opx_readmsg_internal(struct fid_ep *ep, const struct fi_msg_rma *msg,
			        uint64_t flags, int lock_required,
			        const enum fi_av_type av_type, const uint64_t caps,
			        const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	int ret = 0;
	ret = fi_opx_check_rma(opx_ep);
	if (ret)
		return ret;
#endif

	if (lock_required) {
		fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__);
		abort();
	}

	/* TODO - should this be a different cq than the one used by tsend, etc? */
	struct fi_opx_cq *cq = opx_ep->tx->cq;
	if (((cq != NULL) &&
	     ((cq->bflags & FI_SELECTIVE_COMPLETION) && (flags & FI_COMPLETION) == 0))) {
		cq = NULL;
	}

	union fi_opx_context *opx_context = (union fi_opx_context *)msg->context;
	const union fi_opx_addr opx_src_addr = { .fi = (av_type == FI_AV_TABLE) ?
							       opx_ep->tx->av_addr[msg->addr].fi :
							       msg->addr };

	/* for fi_read*(), the 'src' is the remote data */
	size_t src_iov_index = 0;
	const size_t src_iov_count = msg->rma_iov_count;
	uint64_t src_iov_bytes = msg->rma_iov[0].len;
	uint64_t src_iov_addr = msg->rma_iov[0].addr;
	uint64_t src_iov_key = msg->rma_iov[0].key;

	/* for fi_read*(), the 'dst' is the local data */
	size_t dst_iov_index = 0;
	const size_t dst_iov_count = msg->iov_count;
	uint64_t dst_iov_bytes = msg->msg_iov[0].iov_len;
	void *dst_iov_vaddr = msg->msg_iov[0].iov_base;

	size_t niov;
	struct iovec iov[8];
	uint64_t addr[8];
	uint64_t key[8];

	ssize_t index;
	struct fi_opx_completion_counter *cc = ofi_buf_alloc(opx_ep->rma_counter_pool);
	memset(cc, 0, sizeof(*cc));
	cc->byte_counter = 0;
	for(index=0; index < msg->iov_count; index++) {
		cc->byte_counter += msg->msg_iov[index].iov_len;
	}
#ifndef NDEBUG
	size_t totsize = 0;
	size_t totsize_issued = 0;
	for(index=0; index < msg->rma_iov_count; index++) {
		totsize += msg->rma_iov[index].len;
	}
	assert(totsize == cc->byte_counter);
#endif
	cc->cntr = opx_ep->read_cntr;
	cc->cq = ((flags & FI_COMPLETION) == FI_COMPLETION) ? opx_ep->rx->cq : NULL;
	cc->context = msg->context;
	if(opx_context && cc->cq) opx_context->flags = FI_RMA | FI_READ;

	cc->hit_zero = fi_opx_hit_zero;

	while (src_iov_index < src_iov_count) {
		for (niov = 0; niov < 8; ++niov) {
			const size_t len =
				(dst_iov_bytes <= src_iov_bytes) ? dst_iov_bytes : src_iov_bytes;
			iov[niov].iov_len = len;
			iov[niov].iov_base = dst_iov_vaddr;
			addr[niov] = src_iov_addr;
			key[niov] = src_iov_key;

			dst_iov_bytes -= len;
			src_iov_bytes -= len;

			if (src_iov_bytes == 0) {
				/* all done with this src rma iovec */

				if (src_iov_index == (src_iov_count - 1)) {
					/* this is the last src rma iovec .. perform
					 * read with completion processing and return
					 *
					 * the 'dst_iov_bytes' must be zero and it must
					 * be the last dst iovec as well */
					assert(dst_iov_bytes == 0);
					assert(dst_iov_index == (dst_iov_count - 1));
#ifndef NDEBUG
					for(index = 0; index < niov+1; index++) {
						totsize_issued += iov[index].iov_len;
					}
					assert(totsize_issued <= totsize);
#endif
					fi_opx_readv_internal(
						opx_ep, iov, niov + 1, opx_src_addr, addr, key,
						opx_context, flags, cq,
						opx_ep->read_cntr, /* enable_cq, enable_cntr */
						cc, FI_VOID, FI_NOOP, FI_OPX_HFI_DPUT_OPCODE_GET, lock_required, caps,
						reliability);

					return 0;

				} else {
					/* advance to next src rma iovec */
					++src_iov_index;
					src_iov_bytes = msg->rma_iov[src_iov_index].len;
					src_iov_addr = msg->rma_iov[src_iov_index].addr;
					src_iov_key = msg->rma_iov[src_iov_index].key;
				}
			} else {
				src_iov_addr += len;
			}

			if (dst_iov_bytes == 0) {
				/* all done with this dst iovec */

				if (dst_iov_index == (dst_iov_count - 1)) {
					/* this is the last dst iovec .. do nothing since
					 * the 'src_iov_bytes' must be zero and it must
					 * be the last src rma iovec as well */
					assert(src_iov_bytes == 0);
					assert(src_iov_index == (src_iov_count - 1));

					/* in fact, it should be impossible to get here */
					assert(0);
				} else {
					/* advance to next dst iovec */
					++dst_iov_index;
					dst_iov_bytes = msg->msg_iov[dst_iov_index].iov_len;
					dst_iov_vaddr = msg->msg_iov[dst_iov_index].iov_base;
				}
			} else {
				dst_iov_vaddr = (void *)((uintptr_t)dst_iov_vaddr + len);
			}

		} /* end for */
#ifndef NDEBUG
		for(index = 0; index < 8; index++) {
			totsize_issued += iov[index].iov_len;
		}
		assert(totsize_issued <= totsize);
#endif
		fi_opx_readv_internal(opx_ep, iov, 8, opx_src_addr, addr, key, NULL, 0, NULL,
				      NULL, /* disable_cq, disable_cntr */
							  cc, FI_VOID, FI_NOOP, FI_OPX_HFI_DPUT_OPCODE_GET, lock_required, caps,
				      reliability);

	} /* end while */

	/* should never get here */
	abort();

	return 0;
}

inline ssize_t fi_opx_readmsg_generic(struct fid_ep *ep, const struct fi_msg_rma *msg,
				      uint64_t flags, int lock_required,
				      const enum fi_av_type av_type, const uint64_t caps,
				      const enum ofi_reliability_kind reliability)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_readmsg_internal(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED,
					     av_type, caps, reliability);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

static inline ssize_t fi_opx_rma_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
				      fi_addr_t src_addr, uint64_t addr_offset, uint64_t key,
				      void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_read_generic(ep, buf, len, desc, src_addr, addr_offset, key, context,
					 FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static inline ssize_t fi_opx_rma_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
					 uint64_t flags)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_readmsg_generic(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static inline ssize_t fi_opx_rma_inject_write(struct fid_ep *ep, const void *buf, size_t len,
					      fi_addr_t dst_addr, uint64_t addr_offset,
					      uint64_t key)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_inject_write_internal(ep, buf, len, dst_addr, addr_offset, key,
						  FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static inline ssize_t fi_opx_rma_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
				       fi_addr_t dst_addr, uint64_t addr_offset, uint64_t key,
				       void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_write(ep, buf, len, desc, dst_addr, addr_offset, key, context,
				  FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static inline ssize_t fi_opx_rma_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
					size_t count, fi_addr_t dest_addr, uint64_t addr_offset,
					uint64_t key, void *context)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_writev_internal(ep, iov, desc, count, dest_addr, addr_offset, key, context,
					    FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static inline ssize_t fi_opx_rma_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
					  uint64_t flags)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const int lock_required = fi_opx_threading_lock_required(opx_ep->threading);
	const uint64_t caps = opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_writemsg_internal(ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED,
						OPX_AV, caps, OPX_RELIABILITY);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);
	return rc;
}

static struct fi_ops_rma fi_opx_ops_rma_default = {
	.size = sizeof(struct fi_ops_rma),
	.read = fi_opx_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_opx_rma_readmsg,
	.write = fi_opx_rma_write,
	.inject = fi_opx_rma_inject_write,
	.writev = fi_opx_rma_writev,
	.writemsg = fi_opx_rma_writemsg,
	.writedata = fi_no_rma_writedata,
};

int fi_opx_init_rma_ops(struct fid_ep *ep, struct fi_info *info)
{
	if (!ep || !info) {
		errno = FI_EINVAL;
		goto err;
	}

	return 0;
err:
	return -errno;
}

FI_OPX_RMA_SPECIALIZED_FUNC(FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, 0x0018000000000000ull, OPX_RELIABILITY)
FI_OPX_RMA_SPECIALIZED_FUNC(FI_OPX_LOCK_REQUIRED, OPX_AV, 0x0018000000000000ull, OPX_RELIABILITY)

#define FI_OPX_RMA_OPS_STRUCT_NAME(LOCK, AV, CAPS, RELIABILITY)                                    \
	FI_OPX_RMA_OPS_STRUCT_NAME_(LOCK, AV, CAPS, RELIABILITY)

#define FI_OPX_RMA_OPS_STRUCT_NAME_(LOCK, AV, CAPS, RELIABILITY)                                   \
	fi_opx_ops_rma_##LOCK##_##AV##_##CAPS##_##RELIABILITY

#define FI_OPX_RMA_OPS_STRUCT(LOCK, AV, CAPS, RELIABILITY)                                         \
	static struct fi_ops_rma FI_OPX_RMA_OPS_STRUCT_NAME(LOCK, AV, CAPS, RELIABILITY) = {       \
		.size = sizeof(struct fi_ops_rma),                                                 \
		.read = FI_OPX_RMA_SPECIALIZED_FUNC_NAME(read, LOCK, AV, CAPS, RELIABILITY),       \
		.readv = fi_no_rma_readv,                                                          \
		.readmsg = FI_OPX_RMA_SPECIALIZED_FUNC_NAME(readmsg, LOCK, AV, CAPS, RELIABILITY), \
		.write = FI_OPX_RMA_SPECIALIZED_FUNC_NAME(write, LOCK, AV, CAPS, RELIABILITY),     \
		.inject = FI_OPX_RMA_SPECIALIZED_FUNC_NAME(inject_write, LOCK, AV, CAPS,           \
							   RELIABILITY),                           \
		.writev = FI_OPX_RMA_SPECIALIZED_FUNC_NAME(writev, LOCK, AV, CAPS, RELIABILITY),   \
		.writemsg =                                                                        \
			FI_OPX_RMA_SPECIALIZED_FUNC_NAME(writemsg, LOCK, AV, CAPS, RELIABILITY),   \
		.writedata = fi_no_rma_writedata,                                                  \
	}

FI_OPX_RMA_OPS_STRUCT(FI_OPX_LOCK_NOT_REQUIRED, OPX_AV, 0x0018000000000000ull, OPX_RELIABILITY);
FI_OPX_RMA_OPS_STRUCT(FI_OPX_LOCK_REQUIRED, OPX_AV, 0x0018000000000000ull, OPX_RELIABILITY);

int fi_opx_enable_rma_ops(struct fid_ep *ep)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (!opx_ep || !opx_ep->domain) {
		errno = FI_EINVAL;
		goto err;
	}

	uint64_t caps = opx_ep->tx->caps;
	if (caps == 0) {
		caps = FI_OPX_BASE_CAPS;
	}
	if (!(caps & FI_RMA)) {
		/* rma ops not enabled on this endpoint */
		return 0;
	}

	const enum fi_threading threading = opx_ep->domain->threading;

	if (OFI_UNLIKELY(fi_opx_threading_unknown(threading))) {
		opx_ep->ep_fid.rma = &fi_opx_ops_rma_default;
		errno = FI_EINVAL;
		goto err;
	}

	const int lock_required = fi_opx_threading_lock_required(threading);
	if (!lock_required) {
		opx_ep->ep_fid.rma = &FI_OPX_RMA_OPS_STRUCT_NAME(FI_OPX_LOCK_NOT_REQUIRED,
								OPX_AV, 0x0018000000000000ull,
								OPX_RELIABILITY);
	} else {
		opx_ep->ep_fid.rma = &FI_OPX_RMA_OPS_STRUCT_NAME(FI_OPX_LOCK_REQUIRED,
								OPX_AV, 0x0018000000000000ull,
								OPX_RELIABILITY);

	}

	return 0;
err:
	return -errno;
}

int fi_opx_finalize_rma_ops(struct fid_ep *ep)
{
	return 0;
}

ssize_t fi_opx_write_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t len, void *desc,
				   fi_addr_t dest_addr, uint64_t addr_offset, uint64_t key,
				   void *context)
{
	return FI_OPX_RMA_SPECIALIZED_FUNC_NAME(write, OPX_LOCK, OPX_AV, 0x0018000000000000ull,
						OPX_RELIABILITY)(ep, buf, len, desc, dest_addr,
								 addr_offset, key, context);
}

ssize_t fi_opx_inject_write_FABRIC_DIRECT(struct fid_ep *ep, const void *buf, size_t len,
					  fi_addr_t dest_addr, uint64_t addr_offset, uint64_t key)
{
	return FI_OPX_RMA_SPECIALIZED_FUNC_NAME(inject_write, OPX_LOCK, OPX_AV,
						0x0018000000000000ull, OPX_RELIABILITY)(
		ep, buf, len, dest_addr, addr_offset, key);
}

ssize_t fi_opx_read_FABRIC_DIRECT(struct fid_ep *ep, void *buf, size_t len, void *desc,
				  fi_addr_t src_addr, uint64_t addr_offset, uint64_t key,
				  void *context)
{
	return FI_OPX_RMA_SPECIALIZED_FUNC_NAME(read, OPX_LOCK, OPX_AV, 0x0018000000000000ull,
						OPX_RELIABILITY)(ep, buf, len, desc, src_addr,
								 addr_offset, key, context);
}

ssize_t fi_opx_readmsg_FABRIC_DIRECT(struct fid_ep *ep, const struct fi_msg_rma *msg,
				     uint64_t flags)
{
	return FI_OPX_RMA_SPECIALIZED_FUNC_NAME(readmsg, OPX_LOCK, OPX_AV, 0x0018000000000000ull,
						OPX_RELIABILITY)(ep, msg, flags);
}
