/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

#include "efa_cuda_dp_impl.cuh"
#include "efa_gda_kernels.h"
#include <errno.h>
#include <assert.h>

#define CNTR_POLL_MAX_ITER 100000000

static __device__ int efa_cuda_cntr_poll(volatile uint64_t *cntr_ptr, uint64_t target)
{
	for (int i = 0; i < CNTR_POLL_MAX_ITER; i++) {
		if (*cntr_ptr >= target)
			return 0;
	}
	return -1;
}

__global__ void efagda_lat_send_kernel(
	efa_cuda_qp *qp,
	efa_cuda_cq *send_cq,
	efa_cuda_cq *recv_cq,
	volatile uint64_t *send_cntr_ptr,
	volatile uint64_t *recv_cntr_ptr,
	uint16_t ah,
	uint16_t remote_qpn,
	uint32_t remote_qkey,
	uint64_t recv_addr,
	uint32_t recv_length,
	uint32_t recv_lkey,
	uint64_t send_addr,
	uint32_t send_length,
	uint32_t send_lkey,
	int iters,
	int rx_depth,
	int machine_type)
{
	int scnt = 0;
	int rcnt = 0;
	void *cqe;
	struct efa_io_tx_wqe wr_buf;
	__shared__ efa_cuda_qp local_qp;
	__shared__ efa_cuda_cq local_send_cq;
	__shared__ efa_cuda_cq local_recv_cq;
	uint64_t send_cntr_start_val = 0;
	uint64_t recv_cntr_start_val = 0;

	local_qp = *qp;
	local_send_cq = *send_cq;
	local_recv_cq = *recv_cq;

	if (send_cntr_ptr)
		send_cntr_start_val = *send_cntr_ptr;
	if (recv_cntr_ptr)
		recv_cntr_start_val = *recv_cntr_ptr;

	/* Post initial receives */
	for (int i = 0; i < rx_depth; i++) {
		if (efa_cuda_post_recv_wr(&local_qp, recv_addr, recv_length,
					  recv_lkey))
			return;
	}
	efa_cuda_flush_rq_wrs(&local_qp);

	while (scnt < iters || rcnt < iters) {
		/* Poll for receive completion (except for first client send) */
		if (rcnt < iters && !(scnt < 1 && machine_type == 1)) {
			if (recv_cntr_ptr) {
				if (efa_cuda_cntr_poll(recv_cntr_ptr, recv_cntr_start_val + rcnt + 1)) {
					printf("recv hw cntr timeout at rcnt=%d, recv_cntr_ptr=%lu\n", rcnt, *recv_cntr_ptr);
					assert(0);
				}
			} else {
				do {
					cqe = efa_cuda_cq_poll(&local_recv_cq, 0);
				} while (!cqe);
			}

			rcnt++;
			efa_cuda_cq_pop(&local_recv_cq, 1);

			/* Repost receive */
			if (rcnt + rx_depth <= iters) {
				if (efa_cuda_post_recv_wr(&local_qp, recv_addr,
							  recv_length,
							  recv_lkey))
					return;
				efa_cuda_flush_rq_wrs(&local_qp);
			}
		}

		/* Send */
		if (scnt < iters) {
			scnt++;

			if (efa_cuda_start_sq_batch(&local_qp, 1))
				return;
			if (efa_cuda_init_send_wr(&wr_buf, scnt))
				return;
			efa_cuda_wr_set_remote(&wr_buf, ah, remote_qpn,
					       remote_qkey);
			if (efa_cuda_wr_set_sge(&wr_buf, send_lkey, send_addr,
						send_length))
				return;
			if (efa_cuda_sq_batch_place_wr(&local_qp, 0, &wr_buf))
				return;
			efa_cuda_flush_sq_wrs(&local_qp);

			/* Wait for send completion */
			if (send_cntr_ptr) {
				if (efa_cuda_cntr_poll(send_cntr_ptr, send_cntr_start_val + scnt)) {
					printf("send hw cntr timeout at scnt=%d, send_cntr_ptr=%lu\n", scnt, *send_cntr_ptr);
					assert(0);
				}
			} else {
				do {
					cqe = efa_cuda_cq_poll(&local_send_cq, 0);
				} while (!cqe);

				uint32_t err = efa_cuda_wc_read_vendor_err(cqe);
				if (err)
					printf("send comp err %d\n", err);
			}
			efa_cuda_cq_pop(&local_send_cq, 1);
		}
	}

	*qp = local_qp;
	*send_cq = local_send_cq;
	*recv_cq = local_recv_cq;
}

int efagda_run_lat_send(struct efa_cuda_qp *qp,
			struct efa_cuda_cq *send_cq,
			struct efa_cuda_cq *recv_cq,
			volatile uint64_t *send_cntr_ptr,
			volatile uint64_t *recv_cntr_ptr,
			uint16_t ah, uint16_t remote_qpn,
			uint32_t remote_qkey,
			uint64_t recv_addr, uint32_t recv_length,
			uint32_t recv_lkey,
			uint64_t send_addr, uint32_t send_length,
			uint32_t send_lkey,
			int iters, int rx_depth, int is_client,
			cudaStream_t stream)
{
	cudaError_t err;

	efagda_lat_send_kernel<<<1, 1, 0, stream>>>(
		qp, send_cq, recv_cq,
		send_cntr_ptr, recv_cntr_ptr,
		ah, remote_qpn, remote_qkey,
		recv_addr, recv_length, recv_lkey,
		send_addr, send_length, send_lkey,
		iters, rx_depth, is_client);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_lat_send: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_lat_send: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

__global__ void efagda_bw_kernel(
	efa_cuda_qp *qp,
	efa_cuda_cq *send_cq,
	volatile uint64_t *send_cntr_ptr,
	enum ibv_wr_opcode opcode,
	uint64_t send_addr,
	uint32_t send_length,
	uint32_t send_lkey,
	uint16_t ah,
	uint32_t remote_qpn,
	uint32_t remote_qkey,
	uint64_t remote_addr,
	uint32_t remote_rkey,
	int iters,
	int tx_depth)
{
	int scnt = 0;
	int ccnt = 0;
	void *cqe;
	struct efa_io_tx_wqe wr_buf;
	__shared__ efa_cuda_qp local_qp;
	__shared__ efa_cuda_cq local_send_cq;
	uint64_t send_cntr_start_val = 0;

	local_qp = *qp;
	local_send_cq = *send_cq;

	if (send_cntr_ptr) {
		send_cntr_start_val = *send_cntr_ptr;
	}

	while (scnt < iters || ccnt < iters) {
		/* Post writes up to tx_depth */
		while (scnt < iters && (scnt - ccnt) < tx_depth) {
			int ret;
			switch (opcode) {
			case IBV_WR_RDMA_WRITE:
				ret = efa_cuda_init_rdma_write_wr(&wr_buf,
					scnt, remote_rkey, remote_addr);
				break;
			case IBV_WR_RDMA_WRITE_WITH_IMM:
				ret = efa_cuda_init_rdma_write_imm_wr(&wr_buf,
					scnt, remote_rkey, remote_addr,
					0x12345678);
				break;
			case IBV_WR_RDMA_READ:
				ret = efa_cuda_init_rdma_read_wr(&wr_buf,
					scnt, remote_rkey, remote_addr);
				break;
			case IBV_WR_SEND:
				ret = efa_cuda_init_send_wr(&wr_buf, scnt);
				break;
			default:
				return;
			}
			if (ret)
				return;

			if (efa_cuda_wr_set_sge(&wr_buf, send_lkey, send_addr,
						send_length))
				return;
			efa_cuda_wr_set_remote(&wr_buf, ah, remote_qpn,
					       remote_qkey);

			if (efa_cuda_start_sq_batch(&local_qp, 1))
				return;
			if (efa_cuda_sq_batch_place_wr(&local_qp, 0, &wr_buf))
				return;
			efa_cuda_flush_sq_wrs(&local_qp);
			scnt++;
		}

		/* Poll completions */
		while (ccnt < scnt && (scnt == iters ||
		       (scnt - ccnt) >= tx_depth)) {
			if (send_cntr_ptr) {
				if (efa_cuda_cntr_poll(send_cntr_ptr, send_cntr_start_val + ccnt + 1)) {
					printf("bw send hw cntr timeout at ccnt=%d, send_cntr_ptr=%lu\n", ccnt, *send_cntr_ptr);
					assert(0);
				}
				efa_cuda_cq_pop(&local_send_cq, 1);
				ccnt++;
			} else {
				cqe = efa_cuda_cq_poll(&local_send_cq, 0);
				if (cqe) {
					if (((efa_io_cdesc_common *)cqe)->status != 0)
						printf("bw comp err %d\n",
						       ((efa_io_cdesc_common *)cqe)->status);
					efa_cuda_cq_pop(&local_send_cq, 1);
					ccnt++;
				}
			}
		}
	}

	*qp = local_qp;
	*send_cq = local_send_cq;
}

__global__ void efagda_bw_recv_kernel(
	efa_cuda_qp *qp,
	efa_cuda_cq *recv_cq,
	volatile uint64_t *recv_cntr_ptr,
	uint64_t recv_addr,
	uint32_t recv_length,
	uint32_t recv_lkey,
	int iters,
	int rx_depth)
{
	int rcnt = 0;
	void *cqe;
	__shared__ efa_cuda_qp local_qp;
	__shared__ efa_cuda_cq local_recv_cq;
	uint64_t recv_cntr_start_val = 0;

	local_qp = *qp;
	local_recv_cq = *recv_cq;

	if (recv_cntr_ptr)
		recv_cntr_start_val = *recv_cntr_ptr;

	/* Post initial receives */
	for (int i = 0; i < rx_depth; i++) {
		if (efa_cuda_post_recv_wr(&local_qp, recv_addr, recv_length,
					  recv_lkey))
			return;
	}
	efa_cuda_flush_rq_wrs(&local_qp);

	while (rcnt < iters) {
		if (recv_cntr_ptr) {
			if (efa_cuda_cntr_poll(recv_cntr_ptr, recv_cntr_start_val + (rcnt + 1))) {
				printf("bw recv hw cntr timeout at rcnt=%d, recv_cntr_ptr=%lu\n", rcnt, *recv_cntr_ptr);
				assert(0);
			}
		} else {
			do {
				cqe = efa_cuda_cq_poll(&local_recv_cq, 0);
			} while (!cqe);

			if (((efa_io_cdesc_common *)cqe)->status != 0)
				printf("bw recv err %d\n",
				       ((efa_io_cdesc_common *)cqe)->status);
		}

		rcnt++;
		efa_cuda_cq_pop(&local_recv_cq, 1);

		/* Repost receive */
		if (rcnt + rx_depth <= iters) {
			if (efa_cuda_post_recv_wr(&local_qp, recv_addr,
						  recv_length, recv_lkey))
				return;
			efa_cuda_flush_rq_wrs(&local_qp);
		}
	}

	*qp = local_qp;
	*recv_cq = local_recv_cq;
}

int efagda_run_bw(struct efa_cuda_qp *qp,
		  struct efa_cuda_cq *send_cq,
		  volatile uint64_t *send_cntr_ptr,
		  enum ibv_wr_opcode opcode,
		  uint64_t send_addr, uint32_t send_length, uint32_t send_lkey,
		  uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey,
		  uint64_t remote_addr, uint32_t remote_rkey,
		  int iters, int tx_depth,
		  cudaStream_t stream)
{
	cudaError_t err;

	efagda_bw_kernel<<<1, 1, 0, stream>>>(
		qp, send_cq, send_cntr_ptr, opcode,
		send_addr, send_length, send_lkey,
		ah, remote_qpn, remote_qkey,
		remote_addr, remote_rkey,
		iters, tx_depth);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_bw: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_bw: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

int efagda_run_bw_recv(struct efa_cuda_qp *qp,
		       struct efa_cuda_cq *recv_cq,
		       volatile uint64_t *recv_cntr_ptr,
		       uint64_t recv_addr, uint32_t recv_length,
		       uint32_t recv_lkey,
		       int iters, int rx_depth,
		       cudaStream_t stream)
{
	cudaError_t err;

	efagda_bw_recv_kernel<<<1, 1, 0, stream>>>(
		qp, recv_cq, recv_cntr_ptr,
		recv_addr, recv_length, recv_lkey,
		iters, rx_depth);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_bw_recv: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "efagda_run_bw_recv: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}
