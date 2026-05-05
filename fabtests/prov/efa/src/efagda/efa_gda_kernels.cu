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

__global__ void efagda_lat_send_kernel(
	efa_cuda_qp *qp,
	efa_cuda_cq *send_cq,
	efa_cuda_cq *recv_cq,
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

	local_qp = *qp;
	local_send_cq = *send_cq;
	local_recv_cq = *recv_cq;

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
			do {
				cqe = efa_cuda_cq_poll(&local_recv_cq, 0);
			} while (!cqe);

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
			do {
				cqe = efa_cuda_cq_poll(&local_send_cq, 0);
			} while (!cqe);

			uint32_t err = efa_cuda_wc_read_vendor_err(cqe);
			if (err)
				printf("send comp err %d\n", err);
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
		qp, send_cq, recv_cq, ah, remote_qpn, remote_qkey,
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
