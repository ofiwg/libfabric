#ifndef _EFA_CQDIRECT_INTERNAL_H
#define _EFA_CQDIRECT_INTERNAL_H

#include "config.h"

#if HAVE_EFA_CQ_DIRECT

#include "efa_cqdirect_structs.h"
#include "efa_mmio.h"

MAYBE_INLINE void efa_cqdirect_rq_ring_doorbell(struct efa_cqdirect_rq *rq,
						uint16_t pc)
{
	udma_to_device_barrier();
	mmio_write32(rq->wq.db, pc);
}

MAYBE_INLINE void efa_sq_ring_doorbell(struct efa_cqdirect_sq *sq, uint16_t pc)
{
	mmio_write32(sq->wq.db, pc);
}

MAYBE_INLINE uint32_t efa_wq_get_next_wrid_idx(struct efa_cqdirect_wq *wq,
					       uint64_t wr_id)
{
	uint32_t wrid_idx;

	/* Get the next wrid to be used from the index pool */
	wrid_idx = wq->wrid_idx_pool[wq->wrid_idx_pool_next];
	wq->wrid[wrid_idx] = wr_id;

	/* Will never overlap, as validate function succeeded */
	wq->wrid_idx_pool_next++;
	assert(wq->wrid_idx_pool_next <= wq->wqe_cnt);

	return wrid_idx;
}

MAYBE_INLINE enum ibv_wc_status to_ibv_status(enum efa_errno status)
{
	/* note: enum efa_errno status are precisely enum efa_io_comp_status. */
	switch (status) {
	case EFA_IO_COMP_STATUS_OK:
		return IBV_WC_SUCCESS;
	case EFA_IO_COMP_STATUS_FLUSHED:
		return IBV_WC_WR_FLUSH_ERR;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH:
		return IBV_WC_LOC_QP_OP_ERR;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY:
		return IBV_WC_LOC_PROT_ERR;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH:
		return IBV_WC_LOC_LEN_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT:
		return IBV_WC_REM_ABORT_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR:
		return IBV_WC_RNR_RETRY_EXC_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN:
		return IBV_WC_REM_INV_RD_REQ_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS:
		return IBV_WC_BAD_RESP_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH:
		return IBV_WC_REM_INV_REQ_ERR;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE:
		return IBV_WC_RESP_TIMEOUT_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS:
		return IBV_WC_REM_ACCESS_ERR;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER:
		return IBV_WC_REM_OP_ERR;
	default:
		return IBV_WC_GENERAL_ERR;
	}
}

MAYBE_INLINE int efa_cqe_is_pending(struct efa_io_cdesc_common *cqe_common,
				    int phase)
{
	return EFA_GET(&cqe_common->flags, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

MAYBE_INLINE struct efa_io_cdesc_common *
efa_sub_cq_get_cqe(struct efa_cqdirect_cq *cqd, int entry)
{
	return (struct efa_io_cdesc_common *) (cqd->buffer +
					       (entry * cqd->entry_size));
}

MAYBE_INLINE uint32_t
efa_cqdirect_get_current_index(struct efa_cqdirect_cq *cqdirect)
{
	return cqdirect->consumed_cnt & cqdirect->qmask;
}

MAYBE_INLINE struct efa_io_cdesc_common *
efa_cqdirect_next_sub_cqe_get(struct efa_cqdirect_cq *cqdirect)
{
	/* See: cq_next_sub_cqe_get */
	struct efa_io_cdesc_common *cqe;
	uint32_t current_index;

	current_index = efa_cqdirect_get_current_index(cqdirect);
	cqe = efa_sub_cq_get_cqe(cqdirect, current_index);
	if (efa_cqe_is_pending(cqe, cqdirect->phase)) {
		/* Do not read the rest of the completion entry before the
		 * phase bit has been validated.
		 */
		udma_from_device_barrier();
		cqdirect->consumed_cnt++;
		if (!efa_cqdirect_get_current_index(cqdirect))
			cqdirect->phase = 1 - cqdirect->phase;
		return cqe;
	}

	return NULL;
}

MAYBE_INLINE void efa_cqdirect_process_ex_cqe(struct efa_ibv_cq *ibv_cq,
					      struct efa_qp *qp)
{
	struct ibv_cq_ex *ibvcqx = ibv_cq->ibv_cq_ex;
	struct efa_io_cdesc_common *cqe = ibv_cq->cqdirect.cur_cqe;
	uint32_t wrid_idx;

	wrid_idx = cqe->req_id;

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) ==
	    EFA_IO_SEND_QUEUE) {
		ibv_cq->cqdirect.cur_wq = &qp->cqdirect_qp.sq.wq;
		ibvcqx->wr_id = ibv_cq->cqdirect.cur_wq->wrid[wrid_idx];
		ibvcqx->status = to_ibv_status(cqe->status);
	} else {
		ibv_cq->cqdirect.cur_wq =
			EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED) ?
				NULL :
				&qp->cqdirect_qp.rq.wq;
		ibvcqx->wr_id =
			EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED) ?
				0 :
				ibv_cq->cqdirect.cur_wq->wrid[wrid_idx];
		ibvcqx->status = to_ibv_status(cqe->status);
	}
}

MAYBE_INLINE void efa_wq_put_wrid_idx(struct efa_cqdirect_wq *wq,
				      uint32_t wrid_idx)
{
	ofi_genlock_lock(wq->wqlock);
	wq->wrid_idx_pool_next--;
	wq->wrid_idx_pool[wq->wrid_idx_pool_next] = wrid_idx;
	wq->wqe_completed++;
	ofi_genlock_unlock(wq->wqlock);
}

MAYBE_INLINE int efa_cqdirect_wq_initialize(struct efa_cqdirect_wq *wq,
					    uint32_t wqe_cnt,
					    struct ofi_genlock *wqlock)
{
	int i;

	wq->wqe_cnt = wqe_cnt;
	wq->desc_mask = wqe_cnt - 1;
	wq->pc = 0;

	wq->wrid = malloc(wq->wqe_cnt * sizeof(*wq->wrid));
	if (!wq->wrid)
		return ENOMEM;

	wq->wrid_idx_pool = malloc(wqe_cnt * sizeof(uint32_t));
	if (!wq->wrid_idx_pool) {
		free(wq->wrid);
		return ENOMEM;
	}

	/* Initialize the wrid free indexes pool. */
	for (i = 0; i < wqe_cnt; i++)
		wq->wrid_idx_pool[i] = i;

	wq->wqlock = wqlock;
	return 0;
}

MAYBE_INLINE void efa_cqdirect_wq_finalize(struct efa_cqdirect_wq *wq)
{
	if (wq->wrid) {
		free(wq->wrid);
		wq->wrid = NULL;
	}

	if (wq->wrid_idx_pool) {
		free(wq->wrid_idx_pool);
		wq->wrid_idx_pool = NULL;
	}
}

MAYBE_INLINE size_t efa_sge_total_bytes(const struct ibv_sge *sg_list,
					int num_sge)
{
	size_t bytes = 0;
	size_t i;

	for (i = 0; i < num_sge; i++)
		bytes += sg_list[i].length;

	return bytes;
}

MAYBE_INLINE void efa_set_tx_buf(struct efa_io_tx_buf_desc *tx_buf,
				 uint64_t addr, uint32_t lkey, uint32_t length)
{
	tx_buf->length = length;
	EFA_SET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
	tx_buf->buf_addr_lo = addr & 0xffffffff;
	tx_buf->buf_addr_hi = addr >> 32;
}

MAYBE_INLINE void efa_post_send_sgl(struct efa_io_tx_buf_desc *tx_bufs,
				    const struct ibv_sge *sg_list, int num_sge)
{
	const struct ibv_sge *sge;
	size_t i;

	for (i = 0; i < num_sge; i++) {
		sge = &sg_list[i];
		efa_set_tx_buf(&tx_bufs[i], sge->addr, sge->lkey, sge->length);
	}
}

MAYBE_INLINE int efa_post_send_validate(struct efa_qp *qp,
					unsigned int wr_flags)
{
	if (OFI_UNLIKELY(qp->cqdirect_qp.sq.wq.wqe_posted -
				 qp->cqdirect_qp.sq.wq.wqe_completed ==
			 qp->cqdirect_qp.sq.wq.wqe_cnt)) {
		EFA_DBG(FI_LOG_EP_DATA,
			"SQ[%u] is full wqe_posted[%u] wqe_completed[%u] "
			"wqe_cnt[%u]\n",
			qp->qp_num, qp->cqdirect_qp.sq.wq.wqe_posted,
			qp->cqdirect_qp.sq.wq.wqe_completed,
			qp->cqdirect_qp.sq.wq.wqe_cnt);
		return ENOMEM;
	}

	return 0;
}

MAYBE_INLINE int efa_post_recv_validate(struct efa_qp *qp,
					struct ibv_recv_wr *wr)
{
	if (OFI_UNLIKELY(qp->cqdirect_qp.rq.wq.wqe_posted -
				 qp->cqdirect_qp.rq.wq.wqe_completed ==
			 qp->cqdirect_qp.rq.wq.wqe_cnt)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "RQ[%u] is full wqe_posted[%u] wqe_completed[%u] "
			 "wqe_cnt[%u]\n",
			 qp->ibv_qp->qp_num, qp->cqdirect_qp.rq.wq.wqe_posted,
			 qp->cqdirect_qp.rq.wq.wqe_completed,
			 qp->cqdirect_qp.rq.wq.wqe_cnt);
		return ENOMEM;
	}

	return 0;
}

MAYBE_INLINE void efa_set_common_ctrl_flags(struct efa_io_tx_meta_desc *desc,
					    struct efa_cqdirect_sq *sq,
					    enum efa_io_send_op_type op_type)
{
	EFA_SET(&desc->ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&desc->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, op_type);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_PHASE, sq->wq.phase);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);
}

MAYBE_INLINE void efa_sq_advance_post_idx(struct efa_cqdirect_sq *sq)
{
	struct efa_cqdirect_wq *wq = &sq->wq;

	wq->wqe_posted++;
	wq->pc++;

	if (!(wq->pc & wq->desc_mask))
		wq->phase++;
}

MAYBE_INLINE void
efa_cqdirect_send_wr_set_imm_data(struct efa_io_tx_wqe *tx_wqe, __be32 imm_data)
{
	struct efa_io_tx_meta_desc *meta_desc;

	meta_desc = &tx_wqe->meta;
	meta_desc->immediate_data = be32toh(imm_data);
	EFA_SET(&meta_desc->ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
}

MAYBE_INLINE size_t
efa_buf_list_total_bytes(const struct ibv_data_buf *buf_list, size_t num_buf)
{
	size_t bytes = 0;
	size_t i;

	for (i = 0; i < num_buf; i++)
		bytes += buf_list[i].length;

	return bytes;
}

MAYBE_INLINE void efa_send_wr_set_imm_data(struct efa_io_tx_wqe *tx_wqe,
					   __be32 imm_data)
{
	struct efa_io_tx_meta_desc *meta_desc;

	meta_desc = &tx_wqe->meta;
	meta_desc->immediate_data = be32toh(imm_data);
	EFA_SET(&meta_desc->ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
}

MAYBE_INLINE void efa_send_wr_set_rdma_addr(struct efa_io_tx_wqe *tx_wqe,
					    uint32_t rkey, uint64_t remote_addr)
{
	struct efa_io_remote_mem_addr *remote_mem;

	remote_mem = &tx_wqe->data.rdma_req.remote_mem;
	remote_mem->rkey = rkey;
	remote_mem->buf_addr_lo = remote_addr & 0xFFFFFFFF;
	remote_mem->buf_addr_hi = remote_addr >> 32;
}

MAYBE_INLINE void efa_cqdirect_send_wr_post_working(struct efa_cqdirect_sq *sq,
						    bool force_doorbell)
{
	uint32_t sq_desc_idx;

	sq_desc_idx = (sq->wq.pc - 1) & sq->wq.desc_mask;
	mmio_memcpy_x64((struct efa_io_tx_wqe *) sq->desc + sq_desc_idx,
			&sq->curr_tx_wqe, sizeof(struct efa_io_tx_wqe));
	/* this routine only rings the doorbell if it must. */
	if (force_doorbell) {
		mmio_flush_writes();
		efa_sq_ring_doorbell(sq, sq->wq.pc);
		mmio_wc_start();
		sq->num_wqe_pending = 0;
	}
}

MAYBE_INLINE struct efa_io_tx_wqe *
efa_cqdirect_send_wr_common(struct efa_qp *qp, enum efa_io_send_op_type op_type)
{
	struct ibv_qp_ex *ibvqpx = qp->ibv_qp_ex;
	struct efa_cqdirect_qp *cqd_qp = &qp->cqdirect_qp;
	struct efa_cqdirect_sq *sq = &qp->cqdirect_qp.sq;
	struct efa_io_tx_meta_desc *meta_desc;
	int err;

	if (OFI_UNLIKELY(cqd_qp->wr_session_err))
		return NULL;

	// TODO: how is ibvqpx->wr_flags ever set?
	err = efa_post_send_validate(qp, ibvqpx->wr_flags);
	if (OFI_UNLIKELY(err)) {
		cqd_qp->wr_session_err = err;
		return NULL;
	}

	/* MODIFIED: if any are pending, copy out the previous one first: */
	if (sq->num_wqe_pending) {
		efa_cqdirect_send_wr_post_working(sq, false);
	}

	memset(&sq->curr_tx_wqe, 0, sizeof(sq->curr_tx_wqe));

	meta_desc = &sq->curr_tx_wqe.meta;
	efa_set_common_ctrl_flags(meta_desc, sq, op_type);
	meta_desc->req_id = efa_wq_get_next_wrid_idx(&sq->wq, ibvqpx->wr_id);

	/* advance index and change phase */
	efa_sq_advance_post_idx(sq);
	sq->num_wqe_pending++;
	return &sq->curr_tx_wqe;
}

#endif /* end of HAVE_CQ_DIRECT */
#endif /* end of _EFA_CQDIRECT_INTERNAL_H*/
