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
#if HAVE_EFA_DATA_PATH_DIRECT
	else if (base_ep->qp->data_path_direct_enabled) {
		/*
		 * ring the RQ doorbell directly for descriptors staged by
		 * fi_wr_queue_recv
		 */
		struct efa_data_path_direct_rq *rq = &base_ep->qp->data_path_direct_qp.rq;
		efa_data_path_direct_rq_ring_doorbell(rq, rq->wq.pc);
	}
#endif

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
	if (OFI_UNLIKELY(!entry)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "WR send to fi_addr %lu which has no AV entry. Was it "
			 "removed by fi_av_remove?\n", msg->addr);
		return -FI_EINVAL;
	}

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
				 base_ep->qp->max_inline_send, op_msg->flags);
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
	if (OFI_UNLIKELY(!entry)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "WR RMA op to fi_addr %lu which has no AV entry. Was "
			 "it removed by fi_av_remove?\n", msg->addr);
		return -FI_EINVAL;
	}

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
	struct ibv_data_buf inline_data_list[EFA_IO_TX_DESC_NUM_RDMA_BUFS];
	struct efa_comp_action_wr sig = {0};
	size_t total_len;
	int err;

	if (op_rma->flags & FI_EFA_EXTENDED_MSG) {
		const struct fi_efa_op_rma *eop =
			(const struct fi_efa_op_rma *) op_rma;

		err = efa_comp_action_wr_init(
			base_ep, efa_comp_action_fields(eop), &sig);
		if (OFI_UNLIKELY(err))
			return err;
	}

	err = efa_wr_prepare_rdma_common(base_ep, op_rma, wqe,
					 EFA_IO_RDMA_WRITE, &total_len);
	if (OFI_UNLIKELY(err))
		return err;

	if (op_rma->flags & FI_REMOTE_CQ_DATA)
		efa_send_wr_set_imm_data(meta_desc, msg->data);

	if (op_rma->flags & FI_EFA_WR_HIGH_PPS)
		efa_send_wr_set_processing_hint_high_pps(meta_desc);

	if (total_len == 0) {
		err = efa_wr_prepare_rdma_sgl(base_ep, msg, wqe, total_len);
	} else {
		err = efa_msg_use_inline(msg->desc, msg->iov_count, total_len,
					 base_ep->qp->max_inline_write,
					 op_rma->flags);
		if (OFI_UNLIKELY(err < 0))
			return err;
		if (!err) {
			err = efa_wr_prepare_rdma_sgl(base_ep, msg, wqe,
						      total_len);
		} else {
			efa_msg_setup_inline_data_list(base_ep, msg->msg_iov,
						       msg->iov_count,
						       inline_data_list);
			assert(msg->iov_count == 1);
			efa_data_path_direct_set_rdma_inline_data(
				wqe, inline_data_list[0].addr,
				inline_data_list[0].length);
			err = 0;
		}
	}
	if (OFI_UNLIKELY(err))
		return err;

	if (sig.feature_bits)
		efa_data_path_direct_set_comp_actions(
			wqe, &base_ep->qp->data_path_direct_qp.sq, &sig);

	return 0;
}

/* Whether a prepared transmit work request carries completion actions. */
static inline bool efa_wr_tx_has_comp_actions(const struct efa_io_tx_wqe_128 *wqe)
{
	return EFA_GET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_LOCAL_ACTION_REQ) ||
	       EFA_GET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_REMOTE_ACTION_REQ);
}

static inline bool efa_wr_op_is_tx(enum fi_op_type op_type)
{
	return op_type == FI_OP_SEND || op_type == FI_OP_READ || op_type == FI_OP_WRITE;
}

/**
 * @brief Replace the local buffers of a formatted transmit work request
 *
 * Sends keep their buffer descriptors in the top level sgl, while RDMA keeps
 * them in the rdma_req alongside the remote address, so the target area depends
 * on the operation. An RDMA transfer length is the total of its local buffers,
 * so it is recomputed here; a send's length comes from the descriptors
 * themselves and needs no fixup.
 *
 */
static inline int efa_wr_setup_tx_sgl(struct efa_base_ep *base_ep,
				      struct efa_io_tx_wqe_128 *wqe,
				      enum fi_op_type op_type,
				      const struct iovec *iov, void **desc,
				      size_t count)
{
	struct efa_io_tx_meta_desc *meta = &wqe->meta;
	bool is_rdma = op_type == FI_OP_READ || op_type == FI_OP_WRITE;
	struct ibv_sge sg_list[EFA_IO_TX_DESC_NUM_BUFS];
	struct efa_io_tx_buf_desc *tx_bufs;
	int err;

	assert(!EFA_GET(&meta->ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG));

	if (is_rdma) {
		assert(count <= EFA_IO_TX_DESC_NUM_RDMA_BUFS);
		tx_bufs = wqe->data.rdma_req.local_mem;
	} else {
		assert(count <= EFA_IO_TX_DESC_NUM_BUFS);
		tx_bufs = wqe->data.sgl;
	}

	err = efa_msg_setup_sge_list(base_ep, iov, desc, count, sg_list);
	if (OFI_UNLIKELY(err))
		return err;

	efa_data_path_direct_set_sgl(tx_bufs, meta, sg_list, count);

	if (is_rdma)
		wqe->data.rdma_req.remote_mem.length =
			efa_sge_total_bytes(sg_list, count);

	return 0;
}

/**
 * @brief Format a transmit work request whose inline flag is set
 *
 * Continue using inline, or fall back to efa_wr_setup_tx_sgl() when inline does
 * not apply.
 */
static inline int efa_wr_setup_tx_inline_or_sgl(struct efa_base_ep *base_ep,
					 struct efa_io_tx_wqe_128 *wqe,
					 enum fi_op_type op_type,
					 const struct iovec *iov, void **desc,
					 size_t count)
{
	struct ibv_data_buf inline_data_list[EFA_IO_TX_DESC_NUM_BUFS];
	bool is_rdma = op_type == FI_OP_READ || op_type == FI_OP_WRITE;
	size_t inline_size;
	size_t len;

	assert(EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG));

	len = ofi_total_iov_len(iov, count);

	if (is_rdma) {
		assert(op_type == FI_OP_WRITE);
		assert(count <= EFA_IO_TX_DESC_NUM_RDMA_BUFS);
		inline_size = base_ep->qp->max_inline_write;
	} else {
		assert(count <= EFA_IO_TX_DESC_NUM_BUFS);
		inline_size = base_ep->qp->max_inline_send;
		if (count && base_ep->qp->ibv_qp->qp_type == IBV_QPT_UD) {
			if (OFI_UNLIKELY(iov[0].iov_len <
					 base_ep->info->ep_attr->msg_prefix_size))
				return -FI_EINVAL;
			len -= base_ep->info->ep_attr->msg_prefix_size;
		}
	}

	if (!efa_msg_use_inline(desc, count, len, inline_size, 0)) {
		EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 0);
		/*
		 * Clear the inline payload so unused SGL slots are zeroed. It is
		 * at most inline_size bytes, and for an RDMA write it follows the
		 * remote address, which the SGL keeps. inline_size, the QP's
		 * inline limit, leaves room for a completion-action block, so
		 * the block is untouched as well.
		 */
		memset(is_rdma ? wqe->data.rdma_req.inline_data :
				 wqe->data.inline_data,
		       0, inline_size);
		return efa_wr_setup_tx_sgl(base_ep, wqe, op_type, iov, desc,
					   count);
	}

	efa_msg_setup_inline_data_list(base_ep, iov, count, inline_data_list);

	if (is_rdma)
		efa_data_path_direct_set_rdma_inline_data(wqe, inline_data_list[0].addr, len);
	else
		efa_data_path_direct_set_inline_data(wqe, count, inline_data_list);

	return 0;
}

/**
 * @brief Format a receive work request
 *
 * The device consumes one receive queue descriptor per SGE, so a receive work
 * request is an array of them, one per iov entry. LAST marks the final
 * descriptor: it is what tells the device, and queue_recv, where the work
 * request ends, since the array itself carries no count. Only the request id is
 * left out, because it belongs to the queue slot rather than to the operation
 * and is patched in at queue time.
 */
static inline int efa_wr_setup_rx_buf(struct efa_base_ep *base_ep,
					struct efa_io_rx_desc *rx_buf,
					const struct iovec *iov, void **desc,
					size_t count)
{
	struct efa_mr *efa_mr;
	size_t i;

	assert(count <= base_ep->info->rx_attr->iov_limit);

	if (base_ep->qp->ibv_qp->qp_type == IBV_QPT_UD &&
	    OFI_UNLIKELY(iov[0].iov_len <
			 base_ep->info->ep_attr->msg_prefix_size)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "prefix not present on first iov, iov_len[%zu]\n",
			 iov[0].iov_len);
		return -FI_EINVAL;
	}

	/*
	 * Validate every descriptor before writing any of them to avoid failing partway
	 */
	for (i = 0; i < count; i++) {
		if (OFI_UNLIKELY(!desc || !desc[i])) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "EFA direct requires FI_MR_LOCAL but "
				 "application does not provide a valid desc\n");
			return -FI_EINVAL;
		}
	}

	for (i = 0; i < count; i++) {
		efa_mr = (struct efa_mr *) desc[i];

		efa_set_rx_buf(&rx_buf[i], (uintptr_t) iov[i].iov_base,
			       efa_mr->lkey, iov[i].iov_len,
			       i == count - 1);
	}

	return 0;
}

static inline int efa_wr_prepare_recv(struct efa_base_ep *base_ep,
				      const struct fi_op_msg *op_msg,
				      struct efa_io_rx_desc *rx_buf)
{
	const struct fi_msg *msg = &op_msg->msg;

	return efa_wr_setup_rx_buf(base_ep, rx_buf, msg->msg_iov, msg->desc,
				     msg->iov_count);
}

/* The operation flags of @attr, for the op types fi_wr_prepare accepts. */
static inline uint64_t efa_wr_attr_flags(const struct fi_wr_attr *attr)
{
	switch (attr->op_type) {
	case FI_OP_RECV:
	case FI_OP_SEND:
		return attr->op.msg->flags;
	case FI_OP_READ:
	case FI_OP_WRITE:
		return attr->op.rma->flags;
	default:
		return 0;
	}
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
	assert(base_ep->qp->data_path_direct_enabled);

	if (attr->op_type == FI_OP_RECV)
		len = efa_wr_rx_size(attr->op.msg->msg.iov_count);
	else
		len = efa_wr_tx_size();

	if (!wr || *wr_len < len)
		return -FI_ETOOSMALL;

	memset(wr, 0, len);

	/* Completion actions attach to RDMA writes only. */
	if (OFI_UNLIKELY(efa_wr_attr_flags(attr) & FI_EFA_EXTENDED_MSG) &&
	    attr->op_type != FI_OP_WRITE) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_EXTENDED_MSG applies to rma write, op_type %d "
			 "does not\n", attr->op_type);
		return -FI_EBADFLAGS;
	}

	switch (attr->op_type) {
	case FI_OP_RECV:
		assert(attr->op.msg->ep == ep_fid);
		err = efa_wr_prepare_recv(base_ep, attr->op.msg, wr);
		break;
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
	assert(qp->data_path_direct_enabled);
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
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_base_ep *base_ep;
	struct efa_qp *qp;
	struct efa_data_path_direct_rq *rq;
	struct efa_data_path_direct_wq *wq;
	const struct efa_io_rx_desc *prepared = wr;
	struct efa_io_rx_desc *rx_buf;
	uint16_t req_id;
	size_t i;
	int err;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, wr: %p, context: %lx\n", ep_fid, wr,
		(size_t) context);

	if (OFI_UNLIKELY(!wr))
		return -FI_EINVAL;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
	qp = base_ep->qp;
	assert(qp->data_path_direct_enabled);
	rq = &qp->data_path_direct_qp.rq;
	wq = &rq->wq;

	assert(base_ep->context_mode != USE_CONTEXT2);

	ofi_genlock_lock(&base_ep->util_ep.lock);

	err = efa_post_recv_validate(qp, NULL);
	if (OFI_UNLIKELY(err)) {
		efa_data_path_direct_rq_ring_doorbell(rq, wq->pc);
		ofi_genlock_unlock(&base_ep->util_ep.lock);
		return (err == ENOMEM) ? -FI_EAGAIN : -err;
	}

	req_id = efa_wq_get_dev_req_id(wq, (uintptr_t) context);
	wq->wqe_posted++;

	/*
	 * The prepared work request is const and may be queued concurrently, so
	 * copy each of its descriptors into the ring and patch only the request
	 * id, which carries this queue call's context and slot.
	 *
	 * The work request carries no num_sge, so LAST is what says
	 * where it ends.
	 */
	for (i = 0; i < base_ep->info->rx_attr->iov_limit; i++) {
		rx_buf = efa_rq_claim_next_desc(rq);
		*rx_buf = prepared[i];
		rx_buf->req_id = req_id;

		if (EFA_GET(&prepared[i].lkey_ctrl, EFA_IO_RX_DESC_LAST))
			break;
	}

	ofi_genlock_unlock(&base_ep->util_ep.lock);
	return 0;
#else
	return -FI_ENOSYS;
#endif
}

static ssize_t efa_wr_queue_trecv(struct fid_ep *ep_fid, const fi_wr wr,
				  void *context)
{
	return -FI_ENOSYS;
}

static int efa_wr_modify_addr(struct fid_ep *ep_fid, fi_wr wr,
			      enum fi_op_type op_type, fi_addr_t addr)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_base_ep *base_ep;
	struct efa_io_tx_wqe_128 *wqe = wr;
	struct efa_av_entry *entry;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, wr: %p, op_type: %d, addr: %lu\n",
		ep_fid, wr, op_type, addr);

	assert(wr);

	if (!efa_wr_op_is_tx(op_type))
		return 0;

	if (OFI_UNLIKELY(addr == FI_ADDR_UNSPEC))
		return -FI_EINVAL;

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	entry = efa_av_addr_to_entry(base_ep->av, addr);
	if (OFI_UNLIKELY(!entry))
		return -FI_EINVAL;

	efa_data_path_direct_set_ud_addr(&wqe->meta, entry->ah,
					 efa_av_entry_ep_addr(entry)->qpn,
					 efa_av_entry_ep_addr(entry)->qkey);
	return 0;
#else
	return -FI_ENOSYS;
#endif
}

static int efa_wr_modify_iov(struct fid_ep *ep_fid, fi_wr wr,
			     enum fi_op_type op_type, const struct iovec *iov,
			     void **desc, size_t count)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_io_tx_wqe_128 *wqe = wr;
	struct efa_base_ep *base_ep;

	EFA_DBG(FI_LOG_EP_DATA,
		"ep: %p, wr: %p, op_type: %d, iov_count: %zu\n", ep_fid, wr,
		op_type, count);

	assert(wr && (!count || (iov && desc)));

	base_ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	if (efa_wr_op_is_tx(op_type)) {
		if (EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG))
			return efa_wr_setup_tx_inline_or_sgl(base_ep, wqe, op_type,
						      iov, desc, count);

		return efa_wr_setup_tx_sgl(base_ep, wqe, op_type, iov, desc,
					   count);
	}
	return efa_wr_setup_rx_buf(base_ep, wr, iov, desc, count);
#else
	return -FI_ENOSYS;
#endif
}

static int efa_wr_modify_rma_iov(struct fid_ep *ep_fid, fi_wr wr,
				 enum fi_op_type op_type,
				 const struct fi_rma_iov *rma_iov, size_t count)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_io_tx_wqe_128 *wqe = wr;

	EFA_DBG(FI_LOG_EP_DATA,
		"ep: %p, wr: %p, op_type: %d, rma_iov_count: %zu\n", ep_fid, wr,
		op_type, count);

	assert(wr && rma_iov);

	if (OFI_UNLIKELY(op_type != FI_OP_READ && op_type != FI_OP_WRITE)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "only an rma work request has a remote buffer, "
			 "op_type %d does not\n", op_type);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(count != EFA_IO_TX_DESC_NUM_RDMA_BUFS)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "EFA device takes %d remote buffer for rma "
			 "operations, not %zu\n",
			 EFA_IO_TX_DESC_NUM_RDMA_BUFS, count);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(rma_iov[0].len !=
			 wqe->data.rdma_req.remote_mem.length)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "rma_iov length %zu does not match the work request's "
			 "length of %u, use fi_wr_modify_iov to change it\n",
			 rma_iov[0].len, wqe->data.rdma_req.remote_mem.length);
		return -FI_EINVAL;
	}

	efa_send_wr_set_rdma_addr(&wqe->data.rdma_req.remote_mem, rma_iov[0].key, rma_iov[0].addr);
	return 0;
#else
	return -FI_ENOSYS;
#endif
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
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_io_tx_wqe_128 *wqe = wr;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, wr: %p, op_type: %d, data: %lu\n",
		ep_fid, wr, op_type, data);

	assert(wr);

	if (OFI_UNLIKELY(!efa_wr_op_is_tx(op_type))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "only a transmit work request carries remote CQ data, "
			 "op_type %d does not\n", op_type);
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(!EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_HAS_IMM))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "work request was not prepared with FI_REMOTE_CQ_DATA, "
			 "so it carries no remote CQ data to set\n");
		return -FI_EINVAL;
	}

	efa_send_wr_set_imm_data(&wqe->meta, (__be32) data);
	return 0;
#else
	return -FI_ENOSYS;
#endif
}

#define EFA_WR_SUPPORTED_FLAGS \
	(FI_REMOTE_CQ_DATA | FI_INJECT | FI_COMPLETION | \
	 FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE | FI_EFA_WR_HIGH_PPS | \
	 FI_EFA_EXTENDED_MSG)

static int efa_wr_modify_flags(struct fid_ep *ep_fid, fi_wr wr,
			       enum fi_op_type op_type, uint64_t flags)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	struct efa_io_tx_wqe_128 *wqe = wr;

	EFA_DBG(FI_LOG_EP_DATA, "ep: %p, wr: %p, op_type: %d, flags: %lx\n",
		ep_fid, wr, op_type, flags);

	assert(wr);

	if (OFI_UNLIKELY(flags & ~EFA_WR_SUPPORTED_FLAGS)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "flags %lx cannot be applied to a formatted work "
			 "request\n", flags);
		return -FI_EBADFLAGS;
	}

	/*
	 * FI_EFA_EXTENDED_MSG selects the descriptor the work request was
	 * prepared from, and so whether it carries completion actions. It
	 * cannot be added, since there are no action fields to add, nor
	 * removed, so it must agree with the prepared work request.
	 */
	if (OFI_UNLIKELY((flags & FI_EFA_EXTENDED_MSG) &&
			 op_type != FI_OP_WRITE)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_EXTENDED_MSG applies to rma write, op_type %d "
			 "does not\n", op_type);
		return -FI_EINVAL;
	}

	/* a receive descriptor carries none of the remaining flags */
	if (!efa_wr_op_is_tx(op_type))
		return 0;

	if (OFI_UNLIKELY(!!(flags & FI_EFA_EXTENDED_MSG) !=
			 efa_wr_tx_has_comp_actions(wqe))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_EFA_EXTENDED_MSG must match whether the work "
			 "request was prepared with completion actions\n");
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY(
		    !!(flags & FI_REMOTE_CQ_DATA) !=
		    !!EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_HAS_IMM))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_REMOTE_CQ_DATA cannot be added to a work request "
			 "that was not prepared with it\n");
		return -FI_EINVAL;
	}

	if (OFI_UNLIKELY((flags & FI_INJECT) &&
			 !EFA_GET(&wqe->meta.ctrl1,
				  EFA_IO_TX_META_DESC_INLINE_MSG))) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "FI_INJECT cannot be added to a work request that was "
			 "not formatted with inline data\n");
		return -FI_EINVAL;
	}

	if (flags & FI_EFA_WR_HIGH_PPS) {
		if (OFI_UNLIKELY(op_type != FI_OP_WRITE)) {
			EFA_WARN(FI_LOG_EP_DATA,
				 "FI_EFA_WR_HIGH_PPS applies to rma write, "
				 "op_type %d does not\n", op_type);
			return -FI_EINVAL;
		}
		efa_send_wr_set_processing_hint_high_pps(&wqe->meta);
	} else {
		EFA_SET(&wqe->meta.ctrl3, EFA_IO_TX_META_DESC_PROCESSING_HINTS,
			0);
	}

	return 0;
#else
	return -FI_ENOSYS;
#endif
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
