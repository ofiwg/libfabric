#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include "efagda.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cuda
{
	cudaStream_t create_stream();
	void destroy_stream(cudaStream_t stream);

	struct efa_cq *efagda_create_cuda_cq(void *device_buffer,
					     uint16_t num_sub_cqs,
					     uint32_t sub_cq_size,
					     uint32_t cqe_size);
	void efagda_destroy_cuda_cq(efa_cq *d_cq);
	struct efa_qp *
	efagda_create_cuda_qp(uint8_t *sq_buffer, uint32_t sq_num_wqes,
			      uint32_t *sq_db, uint32_t sq_max_batch,
			      uint8_t *rq_buffer, uint32_t rq_num_wqes,
			      uint32_t *rq_db);
	void efagda_destroy_cuda_qp(efa_qp *d_qp);
	int efagda_poll_cq(efa_cq *d_cq, int nwc, ibv_wc *h_wc,
			   cudaStream_t stream);

	int efagda_post_send(efa_qp *d_qp, uint16_t ah, uint16_t remote_qpn,
			     uint32_t remote_qkey, uint64_t addr,
			     uint32_t length, uint32_t lkey,
			     cudaStream_t stream);
	int efagda_post_recv(efa_qp *d_qp, uint64_t addr, uint32_t length,
			     uint32_t lkey, cudaStream_t stream);
} // namespace cuda

#endif
