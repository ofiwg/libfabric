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

#ifndef EFA_GDA_KERNELS_H
#define EFA_GDA_KERNELS_H

#include <stdint.h>
#include <cuda_runtime.h>
#include <efa_cuda_dp.h>
#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

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
			cudaStream_t stream);

int efagda_run_bw(struct efa_cuda_qp *qp,
		  struct efa_cuda_cq *send_cq,
		  enum ibv_wr_opcode opcode,
		  uint64_t send_addr, uint32_t send_length, uint32_t send_lkey,
		  uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey,
		  uint64_t remote_addr, uint32_t remote_rkey,
		  int iters, int tx_depth,
		  cudaStream_t stream);

int efagda_run_bw_recv(struct efa_cuda_qp *qp,
		       struct efa_cuda_cq *recv_cq,
		       uint64_t recv_addr, uint32_t recv_length,
		       uint32_t recv_lkey,
		       int iters, int rx_depth,
		       cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
