#include "efagda.h"
#include "cuda_kernel.cuh"

struct CudaStreamHandle {
	cudaStream_t stream;
};

cuda_stream_t cuda_create_stream(void)
{
	cuda_stream_t handle = new CudaStreamHandle;
	handle->stream = cuda::create_stream();
	if (handle->stream == nullptr) {
		delete handle;
		return nullptr;
	}
	return handle;
}

void cuda_destroy_stream(cuda_stream_t stream)
{
	if (stream) {
		cuda::destroy_stream(stream->stream);
		delete stream;
	}
}

struct efa_cq *efagda_create_cuda_cq(void *device_buffer, uint16_t num_sub_cqs,
				     uint32_t sub_cq_size, uint32_t cqe_size)
{
	return cuda::efagda_create_cuda_cq(device_buffer, num_sub_cqs,
					   sub_cq_size, cqe_size);
}

void efagda_destroy_cuda_cq(efa_cq *d_cq)
{
	return cuda::efagda_destroy_cuda_cq(d_cq);
}

struct efa_qp *efagda_create_cuda_qp(uint8_t *sq_buffer, uint32_t sq_num_wqes,
				     uint32_t *sq_db, uint32_t sq_max_batch,
				     uint8_t *rq_buffer, uint32_t rq_num_wqes,
				     uint32_t *rq_db)
{
	return cuda::efagda_create_cuda_qp(sq_buffer, sq_num_wqes, sq_db,
					   sq_max_batch, rq_buffer, rq_num_wqes,
					   rq_db);
}

void efagda_destroy_cuda_qp(struct efa_qp *d_qp)
{
	cuda::efagda_destroy_cuda_qp(d_qp);
}

int efagda_poll_cq(struct efa_cq *d_cq, int nwc, struct ibv_wc *h_wc,
		   cuda_stream_t stream)
{
	return cuda::efagda_poll_cq(d_cq, nwc, h_wc, stream->stream);
}

int efagda_post_send(struct efa_qp *d_qp, uint16_t ah, uint16_t remote_qpn,
		     uint32_t remote_qkey, uint64_t addr, uint32_t length,
		     uint32_t lkey, cuda_stream_t stream)
{
	return cuda::efagda_post_send(d_qp, ah, remote_qpn, remote_qkey, addr,
				      length, lkey, stream->stream);
}

int efagda_post_recv(struct efa_qp *d_qp, uint64_t addr, uint32_t length,
		     uint32_t lkey, cuda_stream_t stream)
{
	return cuda::efagda_post_recv(d_qp, addr, length, lkey, stream->stream);
}
