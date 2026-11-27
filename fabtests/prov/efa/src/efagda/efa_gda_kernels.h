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
#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The QP and CQ arguments are device pointers to queue structures that the
 * efa-dp-direct host library initialized for the layout version reported by
 * efagda_get_dp_version(). That is the version of the efa-dp-direct device
 * header these kernels were compiled against.
 */
void efagda_get_dp_version(int *major, int *minor, int *subminor);

int efagda_run_lat_send(void *qp,
			void *send_cq,
			void *recv_cq,
			volatile uint64_t *send_cntr_ptr,
			volatile uint64_t *recv_cntr_ptr,
			uint16_t ah, uint16_t remote_qpn,
			uint32_t remote_qkey,
			uint64_t recv_addr, uint32_t recv_length,
			uint32_t recv_lkey,
			uint64_t send_addr, uint32_t send_length,
			uint32_t send_lkey,
			int iters, int rx_depth, int is_client,
			cudaStream_t stream);

int efagda_run_bw(void *qp,
		  void *send_cq,
		  volatile uint64_t *send_cntr_ptr,
		  enum ibv_wr_opcode opcode,
		  uint64_t send_addr, uint32_t send_length, uint32_t send_lkey,
		  uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey,
		  uint64_t remote_addr, uint32_t remote_rkey,
		  int iters, int tx_depth,
		  cudaStream_t stream);

int efagda_run_bw_recv(void *qp,
		       void *recv_cq,
		       volatile uint64_t *recv_cntr_ptr,
		       uint64_t recv_addr, uint32_t recv_length,
		       uint32_t recv_lkey,
		       int iters, int rx_depth,
		       cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
