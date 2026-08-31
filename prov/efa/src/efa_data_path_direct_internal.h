/**
 * @file efa_data_path_direct_internal.h
 * @brief EFA Direct Data Path Internal Functions
 *
 * This header file contains the internal implementation functions for EFA's
 * direct data path operations. These functions provide low-level
 * access to hardware resources and implement the core logic for direct
 * completion queue processing.
 *
 * Key Components:
 * - Hardware doorbell operations for queue management
 * - Work request ID management and indexing
 * - Completion queue entry processing and validation
 * - Work queue initialization and cleanup
 * - Direct memory access utilities for hardware buffers
 *
 * All functions in this file are marked as EFA_ALWAYS_INLINE for optimal
 * performance in the critical path. These functions are designed to
 * minimize overhead and provide direct hardware access.
 *
 * @note This file is only compiled when HAVE_EFA_DATA_PATH_DIRECT is defined
 */

#ifndef _EFA_DATA_PATH_DIRECT_INTERNAL_H
#define _EFA_DATA_PATH_DIRECT_INTERNAL_H

#include "config.h"

#if HAVE_EFA_DATA_PATH_DIRECT

#include "efa_base_ep.h"
#include "efa_cq.h"
#include "efa_data_path_direct_structs.h"
#include "efa_errno.h"
#include "efa_mmio.h"
#include "efa_tp.h"

/* Compiler optimization hints for performance-critical functions */
#define EFA_ALWAYS_INLINE __attribute__((always_inline)) static inline

/**
 * @brief Ring the receive queue doorbell to notify hardware of new entries
 *
 * Writes the producer counter to the hardware doorbell register to notify
 * the EFA device that new receive descriptors are available for processing.
 * Includes a memory barrier to ensure all descriptor writes are visible
 * to the device before the doorbell is rung.
 *
 * @param rq Pointer to the direct receive queue structure
 * @param pc Producer counter value to write to doorbell
 */
EFA_ALWAYS_INLINE void
efa_data_path_direct_rq_ring_doorbell(struct efa_data_path_direct_rq *rq,
				      uint16_t pc)
{
	udma_to_device_barrier(); /* Ensure all writes are visible to device */
	mmio_write32(rq->wq.db, pc);
}

/**
 * @brief Ring the send queue doorbell to notify hardware of new entries
 *
 * Writes the producer counter to the hardware doorbell register to notify
 * the EFA device that new send descriptors are available for processing.
 *
 * @param sq Pointer to the direct send queue structure
 * @param pc Producer counter value to write to doorbell
 */
EFA_ALWAYS_INLINE void efa_sq_ring_doorbell(struct efa_data_path_direct_sq *sq,
					 uint16_t pc)
{
	mmio_write32(sq->wq.db, pc);
}

/**
 * @brief Allocate the next work request ID index from the pool
 *
 * Retrieves the next available index from the work request ID pool and
 * associates it with the provided work request ID. This enables tracking
 * of work requests for completion processing, especially important for
 * out-of-order completions.
 *
 * @param wq Pointer to the work queue structure
 * @param wr_id Work request ID to associate with the allocated index
 * @return The allocated index that can be used in hardware descriptors
 *
 * @note This function assumes the queue has available space (validated
 * elsewhere)
 */
EFA_ALWAYS_INLINE uint32_t
efa_wq_get_next_wrid_idx(struct efa_data_path_direct_wq *wq, uint64_t wr_id)
{
	uint32_t wrid_idx;

	/* Get the next wrid index to be used from the free index pool */
	wrid_idx = wq->wrid_idx_pool[wq->wrid_idx_pool_next];
	wq->wrid[wrid_idx] = wr_id;

	/* Advance to next pool entry - will never overflow as validate
	 * succeeded */
	wq->wrid_idx_pool_next++;
	assert(wq->wrid_idx_pool_next <= wq->wqe_cnt);

	return wrid_idx;
}

/**
 * @brief Allocate a device request ID with QP generation tag
 *
 * Wraps efa_wq_get_next_wrid_idx to produce a device-level request ID
 * that includes the QP generation in the upper bits. This allows stale
 * completions from destroyed QPs to be detected during CQ polling.
 *
 * @param wq Pointer to the work queue structure
 * @param wr_id Work request ID to associate with the allocated index
 * @return Device request ID combining wrid index and generation bits
 */
EFA_ALWAYS_INLINE uint32_t
efa_wq_get_dev_req_id(struct efa_data_path_direct_wq *wq, uint64_t wr_id)
{
	return efa_wq_get_next_wrid_idx(wq, wr_id) | wq->shifted_gen;
}

/**
 * @brief Set 64-bit request ID directly in the TX meta descriptor
 *
 * When 64-bit request ID mode is enabled, the wr_id is placed directly
 * in the descriptor (req_id + req_id_ex fields) instead of going through
 * the wrid index pool translation.
 *
 * @param md Pointer to TX meta descriptor
 * @param wr_id 64-bit work request ID to encode
 */
EFA_ALWAYS_INLINE void
efa_set_req_id_64(struct efa_io_tx_meta_desc *md, uint64_t wr_id)
{
	md->req_id = (uint16_t)wr_id;
	md->req_id_ex.w[0] = (uint16_t)(wr_id >> 16);
	md->req_id_ex.w[1] = (uint16_t)(wr_id >> 32);
	md->req_id_ex.w[2] = (uint16_t)(wr_id >> 48);
}

/**
 * @brief Set the work request ID in a send queue meta descriptor
 *
 * In 64-bit mode, places the wr_id directly in the descriptor.
 * Otherwise, allocates a pool index and stores the mapping.
 *
 * @param md Pointer to TX meta descriptor
 * @param wq Pointer to the send work queue
 * @param wr_id 64-bit work request ID
 */
EFA_ALWAYS_INLINE void
efa_set_sq_comp_wrid(struct efa_io_tx_meta_desc *md,
		     struct efa_data_path_direct_wq *wq, uint64_t wr_id)
{
	if (wq->req_id_64_bit)
		efa_set_req_id_64(md, wr_id);
	else
		md->req_id = efa_wq_get_dev_req_id(wq, wr_id);
}

/**
 * @brief Reconstruct 64-bit request ID from TX completion descriptor
 *
 * When 64-bit request ID mode is enabled, the wr_id is reconstructed
 * from the CQE's req_id and req_id_ex fields.
 *
 * @param tcqe Pointer to TX completion descriptor
 * @return Reconstructed 64-bit work request ID
 */
EFA_ALWAYS_INLINE uint64_t
efa_get_req_id_64(struct efa_io_tx_cdesc *tcqe)
{
	struct efa_io_req_id_ex *req_id_ex = &tcqe->req_id_ex;

	return (uint64_t)tcqe->common.req_id |
	       (uint64_t)req_id_ex->w[0] << 16 |
	       (uint64_t)req_id_ex->w[1] << 32 |
	       (uint64_t)req_id_ex->w[2] << 48;
}

/**
 * @brief Get the work request ID from a send queue completion
 *
 * In 64-bit mode, reconstructs the wr_id directly from the CQE.
 * Otherwise, looks up the wr_id from the wrid table using the device req_id.
 *
 * @param wq Pointer to the send work queue
 * @param cqe Pointer to the completion queue entry
 * @return The 64-bit work request ID
 */
EFA_ALWAYS_INLINE uint64_t
efa_get_sq_comp_wrid(struct efa_data_path_direct_wq *wq,
		     struct efa_io_cdesc_common *cqe)
{
	if (wq->req_id_64_bit) {
		struct efa_io_tx_cdesc *tcqe =
			container_of(cqe, struct efa_io_tx_cdesc, common);

		return efa_get_req_id_64(tcqe);
	}

	return wq->wrid[cqe->req_id & ~wq->gen_mask];
}

/**
 * @brief Convert EFA hardware completion status to IBV work completion status
 *
 * Translates EFA-specific completion status codes to standard InfiniBand
 * verbs work completion status codes. This mapping ensures compatibility
 * with applications expecting standard IBV status codes.
 *
 * @param status EFA hardware completion status code
 * @return Corresponding IBV work completion status code
 *
 * @note EFA errno values are identical to efa_io_comp_status values
 */
EFA_ALWAYS_INLINE enum ibv_wc_status to_ibv_status(enum efa_errno status)
{
	/* Note: enum efa_errno status codes are precisely enum
	 * efa_io_comp_status */
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
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_FEATURE_MISMATCH:
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

/**
 * @brief Check if a completion queue entry is ready for processing
 *
 * Determines if a completion queue entry has been written by hardware
 * by comparing its phase bit with the expected phase. The phase bit
 * alternates each time the completion queue wraps around.
 *
 * @param cqe_common Pointer to the completion queue entry
 * @param phase Expected phase bit value
 * @return 1 if the entry is ready, 0 otherwise
 */
EFA_ALWAYS_INLINE int efa_cqe_is_pending(struct efa_io_cdesc_common *cqe_common,
				     int phase)
{
	return EFA_GET(&cqe_common->flags, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

/**
 * @brief Get a completion queue entry at a specific index
 *
 * Calculates the address of a completion queue entry within the hardware
 * completion queue buffer based on the entry index and entry size.
 *
 * @param cqd Pointer to the direct completion queue structure
 * @param entry Index of the completion queue entry to retrieve
 * @return Pointer to the completion queue entry
 */
EFA_ALWAYS_INLINE struct efa_io_cdesc_common *
efa_device_cq_get_cqe(struct efa_data_path_direct_cq *cqd, int entry)
{
	return (struct efa_io_cdesc_common *)(cqd->buffer +
					      (entry * cqd->entry_size));
}

/**
 * @brief Get the current completion queue index
 *
 * Calculates the current index within the completion queue buffer by
 * masking the consumed count with the queue mask. This handles queue
 * wraparound automatically.
 *
 * @param data_path_direct Pointer to the direct completion queue structure
 * @return Current index within the completion queue
 */
EFA_ALWAYS_INLINE uint32_t
efa_data_path_direct_get_current_index(struct efa_data_path_direct_cq *data_path_direct)
{
	return data_path_direct->consumed_cnt & data_path_direct->qmask;
}

/**
 * @brief Get the next available completion queue entry
 *
 * Retrieves the next completion queue entry that is ready for processing.
 * Handles phase bit validation, queue wraparound, and ensures proper
 * memory ordering when reading completion entries from hardware.
 *
 * @param data_path_direct Pointer to the direct completion queue structure
 * @return Pointer to the next completion entry, or NULL if none available
 *
 * @note This function mirrors cq_next_device_cqe_get from rdma-core
 */
EFA_ALWAYS_INLINE struct efa_io_cdesc_common *
efa_data_path_direct_next_device_cqe_get(struct efa_data_path_direct_cq *data_path_direct)
{
	/* Mirror of cq_next_device_cqe_get from rdma-core */
	struct efa_io_cdesc_common *cqe;
	uint32_t current_index;

	current_index =
		efa_data_path_direct_get_current_index(data_path_direct);
	cqe = efa_device_cq_get_cqe(data_path_direct, current_index);
	if (efa_cqe_is_pending(cqe, data_path_direct->phase)) {
		/**
		 * Do not read the rest of the completion entry before the
		 * phase bit has been validated. This ensures we don't read
		 * stale data from a previous queue wrap.
		 */
		udma_from_device_barrier();
		data_path_direct->consumed_cnt++;
		/* Check for queue wraparound and flip phase if needed */
		if (!efa_data_path_direct_get_current_index(data_path_direct))
			data_path_direct->phase = 1 - data_path_direct->phase;
		return cqe;
	}

	return NULL; /* No completion available */
}

/**
 * @brief Process an extended completion queue entry
 *
 * Extracts completion information from a hardware completion queue entry
 * and populates the extended IBV completion queue structure. Handles both
 * send and receive completions, including unsolicited receives.
 *
 * @param ibv_cq Pointer to the IBV completion queue structure
 * @param qp Pointer to the EFA queue pair that generated the completion
 */
EFA_ALWAYS_INLINE void
efa_data_path_direct_process_ex_cqe(struct efa_ibv_cq *ibv_cq,
				    struct efa_qp *qp)
{
	struct ibv_cq_ex *ibvcqx = ibv_cq->ibv_cq_ex;
	struct efa_io_cdesc_common *cqe = ibv_cq->data_path_direct.cur_cqe;
	uint32_t wrid_idx;

	/* Handle send queue completions */
	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) ==
	    EFA_IO_SEND_QUEUE) {
		ibv_cq->data_path_direct.cur_wq =
			&qp->data_path_direct_qp.sq.wq;
		ibvcqx->wr_id = efa_get_sq_comp_wrid(
			ibv_cq->data_path_direct.cur_wq, cqe);
		ibvcqx->status = to_ibv_status(cqe->status);
	} else {
		/* Handle receive queue completions */
		/* Unsolicited receives don't have associated work requests */
		if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED)) {
			ibv_cq->data_path_direct.cur_wq = NULL;
			ibvcqx->wr_id = 0;
		} else {
			ibv_cq->data_path_direct.cur_wq =
				&qp->data_path_direct_qp.rq.wq;
			wrid_idx = cqe->req_id & ~ibv_cq->data_path_direct.cur_wq->gen_mask;
			ibvcqx->wr_id = ibv_cq->data_path_direct.cur_wq->wrid[wrid_idx];
		}
		ibvcqx->status = to_ibv_status(cqe->status);
	}

#if HAVE_LTTNG
	efa_data_path_direct_tracepoint_process_completion(qp, ibvcqx, cqe);
#endif /* HAVE_LTTNG */
}

/**
 * @brief Return a work request ID index to the free pool
 *
 * Returns a previously allocated work request ID index back to the free
 * pool for reuse. This is called when a work request completes and its
 * index is no longer needed. Thread-safe operation using the work queue lock.
 *
 * @param wq Pointer to the work queue structure
 * @param wrid_idx Work request ID index to return to the pool
 */
EFA_ALWAYS_INLINE void efa_wq_put_wrid_idx(struct efa_data_path_direct_wq *wq,
				       uint32_t wrid_idx)
{
	wq->wrid_idx_pool_next--; /* Move back in the pool */
	wq->wrid_idx_pool[wq->wrid_idx_pool_next] = wrid_idx; /* Return index */
}

/**
 * @brief Release a work queue entry after completion
 *
 * @param wq Pointer to the work queue structure
 */
EFA_ALWAYS_INLINE void
efa_wq_release_slot(struct efa_data_path_direct_wq *wq)
{
	int32_t available = ofi_atomic_inc32(&wq->wqe_available);

	assert(available <= wq->wqe_cnt);
	(void) available;
}

/**
 * @brief Consume an available work queue entry
 *
 * Posting paths serialize wqe_posted with the wqlock.
 *
 * @param wq Pointer to the work queue structure
 */
EFA_ALWAYS_INLINE void efa_wq_consume_slot(struct efa_data_path_direct_wq *wq)
{
	int32_t available;

	wq->wqe_posted++;
	available = ofi_atomic_dec32(&wq->wqe_available);
	assert(available >= 0);
	(void) available;
}


/**
 * @brief Finalize a CQE by releasing its work queue entry
 *
 * In 64-bit request ID mode, no WR-ID pool is shared with the posting path, so
 * the completion only needs to atomically release the work queue entry. Legacy
 * request IDs still require the work queue lock while returning the pool index.
 *
 * @param wq Pointer to the work queue structure
 * @param cqe Pointer to the completion queue entry
 */
EFA_ALWAYS_INLINE void efa_wq_cqe_finalize(struct efa_data_path_direct_wq *wq,
					    struct efa_io_cdesc_common *cqe)
{
	if (wq->req_id_64_bit) {
		efa_wq_release_slot(wq);
		return;
	}

	ofi_genlock_lock(wq->wqlock);
	efa_wq_put_wrid_idx(wq, cqe->req_id & ~wq->gen_mask);
	efa_wq_release_slot(wq);
	ofi_genlock_unlock(wq->wqlock);
}

/**
 * @brief Initialize a direct work queue structure
 *
 * Allocates and initializes the data structures needed for direct work queue
 * operations, including work request ID arrays and index pools. Sets up the
 * queue for efficient work request tracking and completion processing.
 *
 * @param wq Pointer to the work queue structure to initialize
 * @param wqe_cnt Number of work queue entries (must be power of 2)
 * @param wqlock Pointer to the lock for thread-safe operations
 * @return 0 on success, ENOMEM on allocation failure
 */
EFA_ALWAYS_INLINE int
efa_data_path_direct_wq_initialize(struct efa_data_path_direct_wq *wq,
				   uint32_t wqe_cnt, bool req_id_64_bit,
				   struct ofi_genlock *wqlock)
{
	int i;

	wq->wqe_cnt = wqe_cnt;
	wq->desc_mask = wqe_cnt - 1; /* Assumes wqe_cnt is power of 2 */
	wq->wqe_posted = 0;
	wq->pc = 0; /* Initialize producer counter */
	wq->req_id_64_bit = req_id_64_bit;
	ofi_atomic_initialize32(&wq->wqe_available, wqe_cnt);

	if (!req_id_64_bit) {
		/* Allocate work request ID array */
		wq->wrid = malloc(wq->wqe_cnt * sizeof(*wq->wrid));
		if (!wq->wrid)
			return ENOMEM;

		/* Allocate work request ID index pool */
		wq->wrid_idx_pool = malloc(wqe_cnt * sizeof(uint32_t));
		if (!wq->wrid_idx_pool) {
			free(wq->wrid);
			return ENOMEM;
		}

		/* Initialize the work request ID free index pool */
		for (i = 0; i < wqe_cnt; i++)
			wq->wrid_idx_pool[i] = i;
	}

	wq->wqlock = wqlock;
	return 0;
}

/**
 * @brief Finalize and cleanup a direct work queue structure
 *
 * Frees all memory allocated for work queue operations, including work
 * request ID arrays and index pools. Should be called during queue pair
 * destruction to prevent memory leaks.
 *
 * @param wq Pointer to the work queue structure to finalize
 */
EFA_ALWAYS_INLINE void
efa_data_path_direct_wq_finalize(struct efa_data_path_direct_wq *wq)
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

/**
 * @brief Calculate total bytes in a scatter-gather list
 *
 * Sums the lengths of all scatter-gather elements to determine the total
 * number of bytes represented by the scatter-gather list.
 *
 * @param sg_list Array of scatter-gather elements
 * @param num_sge Number of elements in the scatter-gather list
 * @return Total number of bytes across all elements
 */
EFA_ALWAYS_INLINE size_t efa_sge_total_bytes(const struct ibv_sge *sg_list,
					  int num_sge)
{
	size_t bytes = 0;
	size_t i;

	for (i = 0; i < num_sge; i++)
		bytes += sg_list[i].length;

	return bytes;
}

/**
 * @brief Set up a transmit buffer descriptor
 *
 * Initializes a hardware transmit buffer descriptor with the provided
 * address, memory key, and length. Splits the 64-bit address into
 * separate high and low 32-bit fields as required by the hardware.
 *
 * @param tx_buf Pointer to the transmit buffer descriptor
 * @param addr 64-bit buffer address
 * @param lkey Local memory key for the buffer
 * @param length Length of the buffer in bytes
 */
EFA_ALWAYS_INLINE void efa_set_tx_buf(struct efa_io_tx_buf_desc *tx_buf,
				   uint64_t addr, uint32_t lkey,
				   uint32_t length)
{
	tx_buf->length = length;
	EFA_SET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
	tx_buf->buf_addr_lo = addr & 0xffffffff;
	tx_buf->buf_addr_hi = addr >> 32;
}

/**
 * @brief Internal utility: Set SGE list and update metadata length
 * @param tx_bufs Array of hardware transmit buffer descriptors to populate
 * @param meta_desc Pointer to WQE metadata descriptor
 * @param sg_list Array of IBV scatter-gather elements
 * @param num_sge Number of scatter-gather elements to convert
 */
EFA_ALWAYS_INLINE void efa_data_path_direct_set_sgl(struct efa_io_tx_buf_desc *tx_bufs,
				      struct efa_io_tx_meta_desc *meta_desc,
				      const struct ibv_sge *sg_list,
				      int num_sge)
{
	const struct ibv_sge *sge;
	size_t i;

	for (i = 0; i < num_sge; i++) {
		sge = &sg_list[i];
		efa_set_tx_buf(&tx_bufs[i], sge->addr, sge->lkey, sge->length);
	}
	meta_desc->length = num_sge;
}

/**
 * @brief Validate that a send operation can be posted
 *
 * Checks if the send queue has available space for a new work request.
 * Prevents queue overflow by checking the available work queue entry count.
 *
 * @param qp Pointer to the EFA queue pair
 * @return 0 if send can be posted, ENOMEM if queue is full
 */
EFA_ALWAYS_INLINE int efa_post_send_validate(struct efa_qp *qp)
{
	struct efa_data_path_direct_wq *wq = &qp->data_path_direct_qp.sq.wq;
	int32_t wqe_available = ofi_atomic_get32(&wq->wqe_available);
	/* Check if send queue is full */
	if (OFI_UNLIKELY(wqe_available <= 0)) {
		EFA_DBG(FI_LOG_EP_DATA,
			"SQ[%u] is full wqe_posted[%u] wqe_completed[%u] "
			"wqe_cnt[%u]\n",
			qp->qp_num, wq->wqe_posted, wq->wqe_posted - (wq->wqe_cnt - wqe_available),
			wq->wqe_cnt);
		return ENOMEM;
	}

	return 0; /* Queue has space available */
}

/**
 * @brief Validate that a receive operation can be posted
 *
 * Checks if the receive queue has available space for a new work request.
 * Prevents queue overflow by checking the available work queue entry count.
 *
 * @param qp Pointer to the EFA queue pair
 * @param wr Pointer to the receive work request (currently unused)
 * @return 0 if receive can be posted, ENOMEM if queue is full
 */
EFA_ALWAYS_INLINE int efa_post_recv_validate(struct efa_qp *qp,
					  struct ibv_recv_wr *wr)
{
	struct efa_data_path_direct_wq *wq = &qp->data_path_direct_qp.rq.wq;
	int32_t wqe_available = ofi_atomic_get32(&wq->wqe_available);
	/* Check if receive queue is full */
	if (OFI_UNLIKELY(wqe_available <= 0)) {
		EFA_WARN(FI_LOG_EP_DATA,
			 "RQ[%u] is full wqe_posted[%u] wqe_completed[%u] "
			 "wqe_cnt[%u]\n",
			 qp->ibv_qp->qp_num, wq->wqe_posted, wq->wqe_posted - (wq->wqe_cnt - wqe_available),
			 wq->wqe_cnt);
		return ENOMEM;
	}

	return 0; /* Queue has space available */
}

/**
 * @brief Set common control flags for transmit work queue entries
 *
 * Initializes the standard control flags that are common to all transmit
 * operations, including operation type, phase bit, and completion request.
 *
 * @param desc Pointer to the transmit metadata descriptor
 * @param sq Pointer to the send queue structure
 * @param op_type Type of send operation (SEND, RDMA_READ, RDMA_WRITE, etc.)
 */
EFA_ALWAYS_INLINE void efa_set_common_ctrl_flags(struct efa_io_tx_meta_desc *desc,
					      struct efa_data_path_direct_sq *sq,
					      enum efa_io_send_op_type op_type)
{
	EFA_SET(&desc->ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&desc->ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, op_type);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_PHASE, sq->wq.phase);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&desc->ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);
}

/**
 * @brief Advance the send queue posting index
 *
 * Updates the send queue indices after posting a work request. Consumes a work
 * queue entry, increments the producer counter, and handles phase bit changes
 * when the queue wraps.
 *
 * @param sq Pointer to the send queue structure
 */
EFA_ALWAYS_INLINE void efa_sq_advance_post_idx(struct efa_data_path_direct_sq *sq)
{
	struct efa_data_path_direct_wq *wq = &sq->wq;

	efa_wq_consume_slot(wq);
	wq->pc++; /* Advance producer counter */

	/* Check for queue wraparound and advance phase if needed */
	if (!(wq->pc & wq->desc_mask))
		wq->phase++;
}

EFA_ALWAYS_INLINE void efa_send_wr_set_imm_data(struct efa_io_tx_meta_desc *meta_desc,
					     __be32 imm_data)
{
	meta_desc->immediate_data = be32toh(imm_data);
	EFA_SET(&meta_desc->ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
}

EFA_ALWAYS_INLINE void efa_send_wr_set_processing_hint_high_pps(struct efa_io_tx_meta_desc *meta_desc)
{
	EFA_SET(&meta_desc->ctrl3, EFA_IO_TX_META_DESC_PROCESSING_HINTS, EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE);
}


EFA_ALWAYS_INLINE void efa_send_wr_set_rdma_addr(struct efa_io_remote_mem_addr *remote_mem,
					      uint32_t rkey,
					      uint64_t remote_addr)
{
	remote_mem->rkey = rkey;
	remote_mem->buf_addr_lo = remote_addr & 0xFFFFFFFF;
	remote_mem->buf_addr_hi = remote_addr >> 32;
}

EFA_ALWAYS_INLINE void
efa_data_path_direct_send_wr_post(
		struct efa_qp *qp,
		struct efa_data_path_direct_sq *sq,
		void *wqe)
{
	uint32_t sq_desc_idx;
	uint64_t *src, *dst;

	/* Calculate target address in write-combined memory.
	 * Use byte-level arithmetic since wqe_size may be 64 or 128 bytes. */
	sq_desc_idx = sq->wq.pc & sq->wq.desc_mask;
	src = (uint64_t *)wqe;
	dst = (uint64_t *)((uint8_t *)sq->desc + sq_desc_idx * sq->wq.wqe_size);

	/*
	 * Use mmio_memcpy_x64 to copy the WQE to write-combined memory
	 * with proper 8-byte atomic stores. The wqe_size is either 64 or
	 * 128 bytes depending on whether wide WQE is enabled.
	 */
	mmio_memcpy_x64(dst, src, sq->wq.wqe_size);

#if HAVE_LTTNG
	efa_data_path_direct_tracepoint_post_send(qp, sq, wqe);
#endif
}

EFA_ALWAYS_INLINE void
efa_data_path_direct_send_wr_ring_db(struct efa_data_path_direct_sq *sq)
{
	mmio_flush_writes();
	efa_sq_ring_doorbell(sq, sq->wq.pc);
	sq->num_wqe_pending = 0;
}

/**
 * @brief Internal utility: Set UD addressing information in WQE metadata
 * @param meta_desc Pointer to WQE metadata descriptor
 * @param ah Address handle
 * @param remote_qpn Remote queue pair number
 * @param remote_qkey Remote queue key
 */
EFA_ALWAYS_INLINE void efa_data_path_direct_set_ud_addr(struct efa_io_tx_meta_desc *meta_desc,
                                                        struct efa_ah *ah,
                                                        uint32_t remote_qpn,
                                                        uint32_t remote_qkey)
{
	meta_desc->dest_qp_num = remote_qpn;
	meta_desc->ah = ah->ahn;
	meta_desc->qkey = remote_qkey;
}

/**
 * @brief Internal utility: Set inline data buffers in WQE inline data area
 * @param wqe Pointer to work queue entry
 * @param num_buf Number of data buffers
 * @param buf_list Array of data buffers
 */
EFA_ALWAYS_INLINE void efa_data_path_direct_set_inline_data(struct efa_io_tx_wqe_128 *wqe,
                                                            size_t num_buf,
                                                            const struct ibv_data_buf *buf_list)
{
	uint32_t total_length = 0;
	size_t i;

	for (i = 0; i < num_buf; i++) {
		memcpy(wqe->data.inline_data + total_length,
			   buf_list[i].addr, buf_list[i].length);
		total_length += buf_list[i].length;
	}

	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 1);
	wqe->meta.length = total_length;
}

#endif /* HAVE_EFA_DATA_PATH_DIRECT */
#endif /* _EFA_DATA_PATH_DIRECT_INTERNAL_H */
