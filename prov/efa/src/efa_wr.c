/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "efa_mr.h"
#include "efa_data_path_ops.h"
#include "efa_io_defs.h"
#include "efa_wr.h"
#include <rdma/fi_wr.h>

/**
 * @brief Initiate transmit work the provider has deferred
 *
 * Completes the work request chain of sends posted with FI_MORE, or on the
 * direct data path rings the send queue doorbell for the entries batched so
 * far.
 *
 * @param flags	reserved, must be 0
 *
 * @return 0 on success, or a negative fabric errno on fatal failure.
 */
int efa_wr_tx_flush(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	int err = 0;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;

	ofi_genlock_lock(&base_ep->util_ep.lock);

#if HAVE_EFA_DATA_PATH_DIRECT
	if (qp->data_path_direct_enabled) {
		if (qp->data_path_direct_qp.sq.num_wqe_pending)
			efa_data_path_direct_send_wr_ring_db(&qp->data_path_direct_qp.sq);
		ofi_genlock_unlock(&base_ep->util_ep.lock);
		return 0;
	}
#endif
	if (qp->base_ep->is_wr_started) {
		qp->base_ep->is_wr_started = false;
		/*
		 * No error is expected here: ibv_wr_complete does not fail for
		 * the doorbell write, and the work requests were already
		 * validated when they were posted.
		 */
		err = ibv_wr_complete(qp->ibv_qp_ex);
	}
	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err ? -err : 0;
}

/**
 * @brief Initiate receive work the provider has deferred
 *
 * Posts the receive work request chain accumulated by fi_recv with FI_MORE.
 * The receive queue has no deferred-doorbell state of its own: the underlying
 * post always rings the RQ doorbell, so once the staged chain is posted the
 * receives are fully initiated.
 *
 * @param flags	reserved, must be 0
 *
 * @return 0 on success, or a negative fabric errno on fatal failure.
 */
int efa_wr_rx_flush(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_base_ep *base_ep;
	struct ibv_recv_wr *bad_wr;
	int err = 0;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	if (base_ep->recv_wr_index > 0) {
		err = efa_qp_post_recv(base_ep->qp,
				       &base_ep->efa_recv_wr_vec[0].wr, &bad_wr);
		if (OFI_UNLIKELY(err))
			err = (err == ENOMEM) ? -FI_EAGAIN : -err;
		base_ep->recv_wr_index = 0;
	}

	ofi_genlock_unlock(&base_ep->util_ep.lock);

	return err;
}

#if HAVE_EFA_DATA_PATH_DIRECT
static inline int efa_wr_prepare_send(struct efa_base_ep *base_ep,
			       const struct fi_op_msg *op_msg,
			       struct efa_io_tx_wqe_128 *wqe)
{
	const struct fi_msg *msg = &op_msg->msg;
	struct efa_io_tx_meta_desc *meta_desc = &wqe->meta;
	struct efa_qp *qp = base_ep->qp;
	struct efa_av_entry *entry;
	struct ibv_sge sg_list[2];  /* efa device support up to 2 iov */
	struct ibv_data_buf inline_data_list[2];
	size_t iov_count = msg->iov_count;
	size_t len;
	int err;

	entry = efa_av_addr_to_entry(base_ep->av, msg->addr);
	assert(entry);

	assert(msg->iov_count <= base_ep->info->tx_attr->iov_limit);

	len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	if (qp->ibv_qp->qp_type == IBV_QPT_UD) {
		assert(msg->msg_iov[0].iov_len >=
		       base_ep->info->ep_attr->msg_prefix_size);
		len -= base_ep->info->ep_attr->msg_prefix_size;
	}

	assert(len <= base_ep->info->ep_attr->max_msg_size);

	efa_data_path_direct_set_ud_addr(meta_desc, entry->ah,
					 efa_av_entry_ep_addr(entry)->qpn,
					 efa_av_entry_ep_addr(entry)->qkey);

	efa_set_common_ctrl_flags_no_phase(meta_desc, EFA_IO_SEND);

	if (op_msg->flags & FI_REMOTE_CQ_DATA)
		efa_send_wr_set_imm_data(meta_desc, msg->data);

	if (len == 0) {
		efa_data_path_direct_set_inline_data(wqe, 0, inline_data_list);
		return 0;
	}

	err = efa_msg_use_inline(msg->desc, iov_count, len,
				 base_ep->inject_msg_size, op_msg->flags);
	if (OFI_UNLIKELY(err < 0))
		return err;
	if (err) {
		efa_msg_setup_inline_data_list(base_ep, msg->msg_iov, iov_count,
					       inline_data_list);
		efa_data_path_direct_set_inline_data(wqe, iov_count,
						     inline_data_list);
	} else {
		err = efa_msg_setup_sge_list(base_ep, msg->msg_iov, msg->desc,
					     iov_count, sg_list);
		if (OFI_UNLIKELY(err))
			return err;
		efa_data_path_direct_set_sgl(wqe->data.sgl, meta_desc, sg_list,
					     iov_count);
	}

	return 0;
}

/**
 * @brief Common prologue for RDMA read/write work request formatting
 *
 * Validates the RDMA operation, resolves the peer, and formats the parts of
 * the WQE shared by read and write: the UD address, the static control flags
 * (op type, without PHASE), and the remote memory address. The caller then
 * fills in the local buffers (SGL or, for write, inline data) and the transfer
 * length.
 *
 * @param base_ep		endpoint the transfer is issued on
 * @param op_rma		the RMA operation descriptor
 * @param wqe			work request being formatted
 * @param op_type		EFA_IO_RDMA_READ or EFA_IO_RDMA_WRITE
 * @param total_len[out]	total local transfer length
 * @return 0 on success, or a negative libfabric error code
 */
static inline int efa_wr_prepare_rdma_common(struct efa_base_ep *base_ep,
				      const struct fi_op_rma *op_rma,
				      struct efa_io_tx_wqe_128 *wqe,
				      enum efa_io_send_op_type op_type,
				      size_t *total_len)
{
	const struct fi_msg_rma *msg = &op_rma->msg;
	struct efa_io_tx_meta_desc *meta_desc = &wqe->meta;
	struct efa_av_entry *entry;

	if (OFI_UNLIKELY(msg->iov_count > EFA_IO_TX_DESC_NUM_RDMA_BUFS)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "EFA device doesn't support > %d iov for rdma "
			 "operations\n", EFA_IO_TX_DESC_NUM_RDMA_BUFS);
		return -FI_EINVAL;
	}

	assert(msg->rma_iov_count > 0 &&
	       msg->rma_iov_count <= base_ep->info->tx_attr->rma_iov_limit);

	*total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	assert(*total_len <= base_ep->domain->device->max_rdma_size);

	entry = efa_av_addr_to_entry(base_ep->av, msg->addr);
	assert(entry);

	efa_data_path_direct_set_ud_addr(meta_desc, entry->ah,
					 efa_av_entry_ep_addr(entry)->qpn,
					 efa_av_entry_ep_addr(entry)->qkey);

	efa_set_common_ctrl_flags_no_phase(meta_desc, op_type);

	efa_send_wr_set_rdma_addr(&wqe->data.rdma_req.remote_mem,
				  msg->rma_iov[0].key, msg->rma_iov[0].addr);

	return 0;
}

/**
 * @brief Fill an RDMA WQE's local memory from a scatter-gather list
 *
 * Sets the local SGE list and the transfer length in the remote memory
 * descriptor. The 0-byte case uses the domain's bounce buffer with a single
 * zero-length SGE. Shared by read and the non-inline write path.
 */
static inline int efa_wr_prepare_rdma_sgl(struct efa_base_ep *base_ep,
				   const struct fi_msg_rma *msg,
				   struct efa_io_tx_wqe_128 *wqe,
				   size_t total_len)
{
	struct efa_io_rdma_req_128 *rdma_req = &wqe->data.rdma_req;
	struct ibv_sge sg_list[EFA_IO_TX_DESC_NUM_RDMA_BUFS];
	size_t iov_count = msg->iov_count;
	int err;

	if (total_len == 0) {
		struct efa_domain *domain = base_ep->domain;

		sg_list[0].addr = (uintptr_t) domain->zero_byte_bounce_buf;
		sg_list[0].length = 0;
		sg_list[0].lkey = domain->zero_byte_bounce_buf_mr->lkey;
		iov_count = 1;
	} else {
		err = efa_msg_setup_sge_list(base_ep, msg->msg_iov, msg->desc,
					     iov_count, sg_list);
		if (OFI_UNLIKELY(err))
			return err;
	}

	rdma_req->remote_mem.length = efa_sge_total_bytes(sg_list, iov_count);
	efa_data_path_direct_set_sgl(rdma_req->local_mem, &wqe->meta, sg_list,
				     iov_count);

	return 0;
}

static inline int efa_wr_prepare_read(struct efa_base_ep *base_ep,
			       const struct fi_op_rma *op_rma,
			       struct efa_io_tx_wqe_128 *wqe)
{
	size_t total_len;
	int err;

	err = efa_wr_prepare_rdma_common(base_ep, op_rma, wqe, EFA_IO_RDMA_READ,
					 &total_len);
	if (OFI_UNLIKELY(err))
		return err;

	return efa_wr_prepare_rdma_sgl(base_ep, &op_rma->msg, wqe, total_len);
}

static inline int efa_wr_prepare_write(struct efa_base_ep *base_ep,
				const struct fi_op_rma *op_rma,
				struct efa_io_tx_wqe_128 *wqe)
{
	const struct fi_msg_rma *msg = &op_rma->msg;
	struct efa_io_tx_meta_desc *meta_desc = &wqe->meta;
	struct efa_io_rdma_req_128 *rdma_req = &wqe->data.rdma_req;
	struct ibv_data_buf inline_data_list[EFA_IO_TX_DESC_NUM_RDMA_BUFS];
	size_t total_len;
	int err;

	err = efa_wr_prepare_rdma_common(base_ep, op_rma, wqe,
					 EFA_IO_RDMA_WRITE, &total_len);
	if (OFI_UNLIKELY(err))
		return err;

	if (op_rma->flags & FI_REMOTE_CQ_DATA)
		efa_send_wr_set_imm_data(meta_desc, msg->data);

	if (op_rma->flags & FI_EFA_WR_HIGH_PPS)
		efa_send_wr_set_processing_hint_high_pps(meta_desc);

	if (total_len == 0)
		return efa_wr_prepare_rdma_sgl(base_ep, msg, wqe, total_len);

	err = efa_msg_use_inline(msg->desc, msg->iov_count, total_len,
				 base_ep->inject_rma_size, op_rma->flags);
	if (OFI_UNLIKELY(err < 0))
		return err;
	if (!err)
		return efa_wr_prepare_rdma_sgl(base_ep, msg, wqe, total_len);

	/*
	 * Inline RDMA write payload lives in the rdma_req inline area, not the
	 * top-level inline_data used by sends, so copy it here rather than via
	 * efa_data_path_direct_set_inline_data.
	 */
	efa_msg_setup_inline_data_list(base_ep, msg->msg_iov, msg->iov_count,
				       inline_data_list);
	assert(msg->iov_count == 1);
	memcpy(rdma_req->inline_data, inline_data_list[0].addr,
	       inline_data_list[0].length);
	rdma_req->remote_mem.length = inline_data_list[0].length;
	EFA_SET(&meta_desc->ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 1);
	meta_desc->length = inline_data_list[0].length;

	return 0;
}
#endif /* HAVE_EFA_DATA_PATH_DIRECT */

static int efa_wr_prepare(struct fid_ep *ep_fid, const struct fi_wr_attr *attr,
			  fi_wr wr, size_t *wr_len)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_base_ep *base_ep;
	size_t len;
	int err;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, op_type: %d, wr: %p, wr_len: %zu\n",
		ep_fid, attr->op_type, wr, *wr_len);

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	len = efa_wr_tx_size();

	if (!wr || *wr_len < len)
		return -FI_ETOOSMALL;

	memset(wr, 0, len);

	switch (attr->op_type) {
	case FI_OP_SEND:
		assert(attr->op.msg->ep == ep_fid);
		err = efa_wr_prepare_send(base_ep, attr->op.msg, wr);
		break;
	case FI_OP_READ:
		assert(attr->op.rma->ep == ep_fid);
		err = efa_wr_prepare_read(base_ep, attr->op.rma, wr);
		break;
	case FI_OP_WRITE:
		assert(attr->op.rma->ep == ep_fid);
		err = efa_wr_prepare_write(base_ep, attr->op.rma, wr);
		break;
	default:
		return -FI_ENOSYS;
	}

	if (OFI_UNLIKELY(err))
		return err;

	*wr_len = len;
	return 0;
#else
	return -FI_ENOSYS;
#endif
}

static ssize_t efa_wr_queue_tx(struct fid_ep *ep_fid, const fi_wr wr,
			       void *context)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	struct efa_data_path_direct_sq *sq;
	const struct efa_io_tx_wqe_128 *prepared = wr;
	struct efa_io_tx_wqe_128 local_wqe;
	int err;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, wr: %p, context: %lx\n", ep_fid, wr,
		(size_t) context);

	if (OFI_UNLIKELY(!wr))
		return -FI_EINVAL;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;
	sq = &qp->data_path_direct_qp.sq;

	assert(base_ep->context_mode != USE_CONTEXT2);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	err = efa_post_send_validate(qp);
	if (OFI_UNLIKELY(err)) {
		if (sq->num_wqe_pending)
			efa_data_path_direct_send_wr_ring_db(sq);
		ofi_genlock_unlock(&base_ep->util_ep.lock);
		return (err == ENOMEM) ? -FI_EAGAIN : -err;
	}

	if (!sq->num_wqe_pending)
		mmio_wc_start();

	if (sq->num_wqe_pending == sq->wq.max_batch) {
		efa_data_path_direct_send_wr_ring_db(sq);
		mmio_wc_start();
	}

	/*
	 * The prepared WR is const and may be queued concurrently, so copy it
	 * and patch only the queue slot state: the request id, which carries
	 * this queue call's context, and the PHASE bit, which tracks the ring
	 * position.
	 */
	local_wqe = *prepared;
	efa_set_sq_comp_wrid(&local_wqe.meta, &sq->wq, (uintptr_t) context);
	EFA_SET(&local_wqe.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, sq->wq.phase);

	efa_data_path_direct_send_wr_post(qp, sq, &local_wqe);

	efa_sq_advance_post_idx(sq);
	sq->num_wqe_pending++;

	ofi_genlock_unlock(&base_ep->util_ep.lock);
	return 0;
#else
	return -FI_ENOSYS;
#endif
}

static ssize_t efa_wr_queue_recv(struct fid_ep *ep_fid, const fi_wr wr,
				 void *context)
{
	return -FI_ENOSYS;
}

static ssize_t efa_wr_queue_trecv(struct fid_ep *ep_fid, const fi_wr wr,
				  void *context)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_addr(struct fid_ep *ep_fid, fi_wr wr,
			      enum fi_op_type op_type, fi_addr_t addr)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_iov(struct fid_ep *ep_fid, fi_wr wr,
			     enum fi_op_type op_type, const struct iovec *iov,
			     void **desc, size_t count)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_rma_iov(struct fid_ep *ep_fid, fi_wr wr,
				 enum fi_op_type op_type,
				 const struct fi_rma_iov *rma_iov, size_t count)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_tag(struct fid_ep *ep_fid, fi_wr wr,
			     enum fi_op_type op_type, uint64_t tag,
			     uint64_t ignore)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_data(struct fid_ep *ep_fid, fi_wr wr,
			      enum fi_op_type op_type, uint64_t data)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_flags(struct fid_ep *ep_fid, fi_wr wr,
			       enum fi_op_type op_type, uint64_t flags)
{
	return -FI_ENOSYS;
}

struct fi_ops_wr efa_wr_ops = {
	.size = sizeof(struct fi_ops_wr),
	.prepare = efa_wr_prepare,
	.queue_tx = efa_wr_queue_tx,
	.queue_recv = efa_wr_queue_recv,
	.queue_trecv = efa_wr_queue_trecv,
	.modify_addr = efa_wr_modify_addr,
	.modify_iov = efa_wr_modify_iov,
	.modify_rma_iov = efa_wr_modify_rma_iov,
	.modify_tag = efa_wr_modify_tag,
	.modify_data = efa_wr_modify_data,
	.modify_flags = efa_wr_modify_flags,
};

size_t efa_wr_tx_size(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return sizeof(struct efa_io_tx_wqe_128);
#else
	return 0;
#endif
}

size_t efa_wr_rx_size(size_t num_sge)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return num_sge * sizeof(struct efa_io_rx_desc);
#else
	return 0;
#endif
}
