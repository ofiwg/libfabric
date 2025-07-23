#include "cuda_kernel.cuh"
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "efa_io_defs.h"
#include <stdint.h>

#define BIT(nr) (1UL << (nr))

#define __bf_shf(x) (__builtin_ffsll(x) - 1)

#define FIELD_GET(_mask, _reg) \
	({ (typeof(_mask)) (((_reg) & (_mask)) >> __bf_shf(_mask)); })

#define FIELD_PREP(_mask, _val) \
	({ ((typeof(_mask)) (_val) << __bf_shf(_mask)) & (_mask); })

#define BITS_PER_LONG (8 * sizeof(long))

#define GENMASK(h, l) \
	(((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))

#define EFA_GET(ptr, mask) FIELD_GET(mask##_MASK, *(ptr))

#define EFA_SET(ptr, mask, value)                       \
	({                                              \
		typeof(ptr) _ptr = ptr;                 \
		*_ptr = (*_ptr & ~(mask##_MASK)) |      \
			FIELD_PREP(mask##_MASK, value); \
	})

// Helper device functions
__device__ uint32_t efa_sub_cq_get_current_index(const efa_sub_cq *sub_cq)
{
	return sub_cq->consumed_cnt & sub_cq->queue_mask;
}

__device__ int efa_cqe_is_pending(const efa_io_cdesc_common *cqe_common,
				  int phase)
{
	return EFA_GET(&cqe_common->flags, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

__device__ efa_io_cdesc_common *efa_sub_cq_get_cqe(efa_sub_cq *sub_cq,
						   int entry)
{
	return (efa_io_cdesc_common *) (sub_cq->buf +
					(entry * sub_cq->cqe_size));
}

__device__ efa_io_cdesc_common *cq_next_sub_cqe_get(efa_sub_cq *sub_cq)
{
	uint32_t current_index = efa_sub_cq_get_current_index(sub_cq);
	efa_io_cdesc_common *cqe = efa_sub_cq_get_cqe(sub_cq, current_index);

	if (efa_cqe_is_pending(cqe, sub_cq->phase)) {
		__threadfence_system();

		sub_cq->consumed_cnt++;
		if (!efa_sub_cq_get_current_index(sub_cq))
			sub_cq->phase = 1 - sub_cq->phase;
		return cqe;
	}

	return nullptr;
}

__device__ int efa_poll_sub_cq(efa_cq *cq, efa_sub_cq *sub_cq, ibv_wc *wc)
{
	efa_io_cdesc_common *cqe = cq_next_sub_cqe_get(sub_cq);
	if (!cqe)
		return 0;

	uint32_t wrid_idx = cqe->req_id;
	wc->status = cqe->status;
	wc->vendor_err = cqe->status;
	wc->wc_flags = 0;

	enum efa_io_send_op_type op_type = (enum efa_io_send_op_type) EFA_GET(
		&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE);

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) ==
	    EFA_IO_SEND_QUEUE) {
		// Handle send queue case
		wc->opcode = op_type;
	} else {
		// Handle receive queue case
		efa_io_rx_cdesc_ex *rcqe = (efa_io_rx_cdesc_ex *) cqe;
		wc->byte_len = rcqe->base.length;
		wc->opcode = op_type;
	}

	wc->wr_id = wrid_idx;
	return 1;
}

// Main CUDA kernel
__global__ void efa_poll_cq_kernel(efa_cq *cq, int nwc, ibv_wc *wc, int *result)
{
	*result = 0;

	// Poll sub CQs
	for (uint16_t sub_cq_idx = 0; sub_cq_idx < cq->num_sub_cqs;
	     sub_cq_idx++) {
		efa_sub_cq *sub_cq = &cq->sub_cq_arr[0];

		if (sub_cq_idx != 0)
			continue;

		int ret = efa_poll_sub_cq(cq, sub_cq, wc);
		if (ret) {
			atomicAdd(&cq->cc, 1);
			*result = ret;
			break;
		}
	}
}

__global__ void efa_post_send_kernel(efa_qp *qp, uint16_t ah,
				     uint16_t remote_qpn, uint32_t remote_qkey,
				     uint64_t addr, uint32_t length,
				     uint32_t lkey, int *result)
{
	__shared__ uint8_t wqe_buf[sizeof(struct efa_io_tx_wqe) + 64];
	struct efa_io_tx_wqe *wqe =
		(struct efa_io_tx_wqe *) ((uint64_t) (wqe_buf + 64 - 1) &
					  ~(64 - 1));
	uint32_t sq_desc_offset;

	wqe->meta.dest_qp_num = remote_qpn;
	wqe->meta.ah = ah;
	wqe->meta.qkey = remote_qkey;
	wqe->meta.req_id = 0;

	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, EFA_IO_SEND);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, qp->sq.wq.phase);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);

	wqe->meta.length = 1;
	EFA_SET(&wqe->data.sgl[0].lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
	wqe->data.sgl[0].length = length;
	wqe->data.sgl[0].buf_addr_lo = addr & 0xffffffff;
	wqe->data.sgl[0].buf_addr_hi = addr >> 32;

	sq_desc_offset = (qp->sq.wq.pc & qp->sq.wq.queue_mask) * sizeof(*wqe);
	memcpy(qp->sq.buf + sq_desc_offset, wqe, sizeof(*wqe));

	qp->sq.wq.wqes_posted++;
	qp->sq.wq.pc++;

	if (!(qp->sq.wq.pc & qp->sq.wq.queue_mask))
		qp->sq.wq.phase++;

	__threadfence_system();
	*qp->sq.wq.db = qp->sq.wq.pc;

	*result = 0;
}

__global__ void efa_post_recv_kernel(efa_qp *qp, uint64_t addr, uint32_t length,
				     uint32_t lkey, int *result)
{
	struct efa_io_rx_desc wqe = {0};
	uint32_t rq_desc_offset;

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_FIRST, 1);
	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LAST, 1);

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LKEY, lkey);
	wqe.buf_addr_lo = addr;
	wqe.buf_addr_hi = addr >> 32;
	wqe.length = length;

	/* Copy descriptor to RX ring */
	rq_desc_offset = (qp->rq.wq.pc & qp->rq.wq.queue_mask) * sizeof(wqe);
	memcpy(qp->rq.buf + rq_desc_offset, &wqe, sizeof(wqe));

	/* Wrap rx descriptor index */
	qp->rq.wq.pc++;
	if (!(qp->rq.wq.pc & qp->rq.wq.queue_mask))
		qp->rq.wq.phase++;

	__threadfence_system();

	*qp->rq.wq.db = qp->rq.wq.pc;

	*result = 0;
}

namespace cuda
{

	cudaStream_t create_stream()
	{
		cudaStream_t stream;
		cudaError_t err = cudaStreamCreate(&stream);
		if (err != cudaSuccess) {
			printf("Failed to create CUDA stream: %s\n",
			       cudaGetErrorString(err));
			return nullptr;
		}
		return stream;
	}

	void destroy_stream(cudaStream_t stream)
	{
		if (stream) {
			cudaStreamDestroy(stream);
		}
	}

	struct efa_cq *efagda_create_cuda_cq(void *device_buffer,
					     uint16_t num_sub_cqs,
					     uint32_t sub_cq_size,
					     uint32_t cqe_size)
	{
		cudaError_t cuda_err;

		// Allocate device CQ structure
		efa_cq *d_cq;
		cuda_err = cudaMalloc(&d_cq, sizeof(efa_cq));
		if (cuda_err != cudaSuccess) {
			printf("Failed to allocate device memory for cq: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		// Allocate and initialize sub_cq array
		efa_sub_cq *d_sub_cq_arr;
		cuda_err = cudaMalloc(&d_sub_cq_arr,
				      num_sub_cqs * sizeof(efa_sub_cq));
		if (cuda_err != cudaSuccess) {
			cudaFree(d_cq);
			printf("Failed to allocate device memory for "
			       "sub_cq_arr: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		// Initialize sub_cqs on host
		efa_sub_cq *h_sub_cqs = new efa_sub_cq[num_sub_cqs];
		uint8_t *curr_buf = (uint8_t *) device_buffer;
		size_t sub_buf_size = cqe_size * sub_cq_size;

		for (int i = 0; i < num_sub_cqs; i++) {
			h_sub_cqs[i].consumed_cnt = 0;
			h_sub_cqs[i].phase = 1;
			h_sub_cqs[i].buf = curr_buf;
			h_sub_cqs[i].queue_mask = sub_cq_size - 1;
			h_sub_cqs[i].cqe_size = cqe_size;
			h_sub_cqs[i].ref_cnt = 0;

			curr_buf += sub_buf_size;
		}

		// Copy sub_cqs to device
		cuda_err = cudaMemcpy(d_sub_cq_arr, h_sub_cqs,
				      num_sub_cqs * sizeof(efa_sub_cq),
				      cudaMemcpyHostToDevice);
		delete[] h_sub_cqs;

		if (cuda_err != cudaSuccess) {
			cudaFree(d_sub_cq_arr);
			cudaFree(d_cq);
			printf("Failed to copy sub_cqs to device: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		// Initialize and copy CQ structure
		efa_cq h_cq = {}; // Zero initialize
		h_cq.buf = (uint8_t *) device_buffer;
		h_cq.num_sub_cqs = num_sub_cqs;
		h_cq.sub_cq_arr = d_sub_cq_arr;
		h_cq.next_poll_idx = 0;
		h_cq.cc = 0;

		cuda_err = cudaMemcpy(d_cq, &h_cq, sizeof(efa_cq),
				      cudaMemcpyHostToDevice);
		if (cuda_err != cudaSuccess) {
			cudaFree(d_sub_cq_arr);
			cudaFree(d_cq);
			printf("Failed to copy cq to device: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		return d_cq;
	}

	void efagda_destroy_cuda_cq(efa_cq *d_cq)
	{
		if (d_cq) {
			// Get the CQ structure from device to get sub_cq_arr
			// pointer to free
			efa_cq h_cq;
			if (cudaMemcpy(&h_cq, d_cq, sizeof(efa_cq),
				       cudaMemcpyDeviceToHost) == cudaSuccess) {
				cudaFree(h_cq.sub_cq_arr);
			}
			cudaFree(d_cq);
		}
	}

	struct efa_qp *
	efagda_create_cuda_qp(uint8_t *sq_buffer, uint32_t sq_num_wqes,
			      uint32_t *sq_db, uint32_t sq_max_batch,
			      uint8_t *rq_buffer, uint32_t rq_num_wqes,
			      uint32_t *rq_db)
	{
		cudaError_t cuda_err;

		// Allocate device QP structure
		efa_qp *d_qp;
		cuda_err = cudaMalloc(&d_qp, sizeof(efa_qp));
		if (cuda_err != cudaSuccess) {
			printf("Failed to allocate device memory for qp: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		// Initialize QP structure on host
		efa_qp h_qp = {};

		// Initialize SQ
		h_qp.sq.buf = sq_buffer;
		h_qp.sq.wq.db = sq_db;
		h_qp.sq.wq.max_wqes = sq_num_wqes;
		h_qp.sq.wq.queue_mask =
			sq_num_wqes - 1; // Assuming max_wqes is power of 2
		h_qp.sq.wq.wqes_posted = 0;
		h_qp.sq.wq.wqes_completed = 0;
		h_qp.sq.wq.pc = 0;
		h_qp.sq.wq.phase = 0;

		h_qp.sq.max_batch_wr = sq_max_batch;
		// TODO: get from args or delete:
		h_qp.sq.max_inline_data = 32;
		h_qp.sq.max_rdma_sges = 2;

		// Initialize RQ
		h_qp.rq.buf = rq_buffer;
		h_qp.rq.wq.db = rq_db;
		h_qp.rq.wq.max_wqes = rq_num_wqes;
		h_qp.rq.wq.queue_mask =
			rq_num_wqes - 1; // Assuming max_wqes is power of 2
		h_qp.rq.wq.wqes_posted = 0;
		h_qp.rq.wq.wqes_completed = 0;
		h_qp.rq.wq.pc = 0;
		h_qp.rq.wq.phase = 1;

		// Copy QP structure to device
		cuda_err = cudaMemcpy(d_qp, &h_qp, sizeof(efa_qp),
				      cudaMemcpyHostToDevice);
		if (cuda_err != cudaSuccess) {
			cudaFree(d_qp);
			printf("Failed to copy qp to device: %s\n",
			       cudaGetErrorString(cuda_err));
			return nullptr;
		}

		return d_qp;
	}

	void efagda_destroy_cuda_qp(efa_qp *d_qp)
	{
		if (d_qp) {
			cudaFree(d_qp);
		}
	}

	int efagda_poll_cq(efa_cq *d_cq, int nwc, ibv_wc *h_wc,
			   cudaStream_t stream)
	{
		cudaError_t cuda_err;

		// Allocate device memory for work completions and results
		ibv_wc *d_wc;
		int *d_results;

		cuda_err = cudaMalloc(&d_wc, nwc * sizeof(ibv_wc));
		if (cuda_err != cudaSuccess) {
			printf("Failed to allocate device memory for wc: %s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		cuda_err = cudaMalloc(&d_results, nwc * sizeof(int));
		if (cuda_err != cudaSuccess) {
			cudaFree(d_wc);
			printf("Failed to allocate device memory for results: "
			       "%s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		// Launch kernel in the stream
		efa_poll_cq_kernel<<<1, 1, 0, stream>>>(d_cq, nwc, d_wc,
							d_results);

		// Check for kernel launch errors
		cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			cudaFree(d_wc);
			cudaFree(d_results);
			printf("Failed to launch kernel: %s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		// Copy results back using the stream
		cuda_err = cudaMemcpyAsync(h_wc, d_wc, nwc * sizeof(ibv_wc),
					   cudaMemcpyDeviceToHost, stream);
		if (cuda_err != cudaSuccess) {
			printf("Failed to copy wc back to host: %s\n",
			       cudaGetErrorString(cuda_err));
		}

		// Synchronize the stream to ensure copies are complete
		cudaStreamSynchronize(stream);

		// Count successful polls
		int *h_results = new int[nwc];
		cuda_err =
			cudaMemcpyAsync(h_results, d_results, nwc * sizeof(int),
					cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		int total = 0;
		if (cuda_err == cudaSuccess) {
			for (int i = 0; i < nwc; i++) {
				if (h_results[i] <= 0)
					break;
				total++;
			}
		}

		// Cleanup temporary allocations
		cudaFree(d_wc);
		cudaFree(d_results);
		delete[] h_results;

		return total;
	}

	int efagda_post_send(efa_qp *d_qp, uint16_t ah, uint16_t remote_qpn,
			     uint32_t remote_qkey, uint64_t addr,
			     uint32_t length, uint32_t lkey,
			     cudaStream_t stream)
	{
		cudaError_t cuda_err;
		int *d_result;
		int h_result;

		cuda_err = cudaMalloc(&d_result, sizeof(int));
		if (cuda_err != cudaSuccess) {
			printf("Failed to allocate device memory for result: "
			       "%s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		// Launch kernel in the stream
		efa_post_send_kernel<<<1, 1, 0, stream>>>(
			d_qp, ah, remote_qpn, remote_qkey, addr, length, lkey,
			d_result);

		// Synchronize the stream to ensure copies are complete
		cudaStreamSynchronize(stream);

		// Check for kernel launch errors
		cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			cudaFree(d_result);
			printf("Failed to launch send kernel: %s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}
		cuda_err = cudaMemcpyAsync(&h_result, d_result, sizeof(int),
					   cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		// Cleanup temporary allocations
		cudaFree(d_result);

		return h_result;
	}

	int efagda_post_recv(efa_qp *d_qp, uint64_t addr, uint32_t length,
			     uint32_t lkey, cudaStream_t stream)
	{
		cudaError_t cuda_err;
		int *d_result = nullptr;
		int h_result;

		cuda_err = cudaMalloc(&d_result, sizeof(int));
		if (cuda_err != cudaSuccess) {
			printf("Failed to allocate device memory for result: "
			       "%s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		// Launch kernel in the stream
		efa_post_recv_kernel<<<1, 1, 0, stream>>>(d_qp, addr, length,
							  lkey, d_result);

		// Check for kernel launch errors
		cuda_err = cudaGetLastError();
		if (cuda_err != cudaSuccess) {
			cudaFree(d_result);
			printf("Failed to launch recv kernel: %s\n",
			       cudaGetErrorString(cuda_err));
			return -1;
		}

		// Synchronize the stream to ensure copies are complete
		cudaStreamSynchronize(stream);

		cuda_err = cudaMemcpyAsync(&h_result, d_result, sizeof(int),
					   cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);

		// Cleanup temporary allocations
		cudaFree(d_result);

		return h_result;
	}

} // namespace cuda
