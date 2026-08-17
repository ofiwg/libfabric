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

#ifndef EFA_XPU_KERNELS_H
#define EFA_XPU_KERNELS_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include <rdma/fi_xpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * ft_efa_xpu_run_lat_send - Run a latency (ping-pong) send test on GPU.
 *
 * @xpu_ep:       Device-side XPU endpoint handle
 * @xpu_send_cq:  Device-side XPU send completion queue handle
 * @xpu_recv_cq:  Device-side XPU recv completion queue handle
 * @xpu_send_cntr: Device-side XPU send counter (or NULL to use CQ)
 * @xpu_recv_cntr: Device-side XPU recv counter (or NULL to use CQ)
 * @dest_addr:    Device-side AV address of remote peer
 * @dest_addr_size: Size of dest_addr buffer
 * @recv_buf:     Receive buffer address (device memory)
 * @recv_len:     Receive buffer length
 * @recv_desc:    Receive MR descriptor (device-side)
 * @recv_desc_size: Size of recv_desc
 * @send_buf:     Send buffer address (device memory)
 * @send_len:     Send buffer length
 * @send_desc:    Send MR descriptor (device-side)
 * @send_desc_size: Size of send_desc
 * @iters:        Number of iterations
 * @rx_depth:     Receive queue depth (pre-posted receives)
 * @is_client:    1 if this side sends first, 0 otherwise
 * @scope:        Cooperative scope the operations are issued at
 * @threads:      Threads in the block, split between the issuers of @scope
 * @stream:       CUDA stream for kernel launch
 *
 * Returns 0 on success, negative on failure.
 */
int ft_efa_xpu_run_lat_send(struct fid_xpu_ep *xpu_ep,
			struct fid_xpu_cq *xpu_send_cq,
			struct fid_xpu_cq *xpu_recv_cq,
			struct fid_xpu_cntr *xpu_send_cntr,
			struct fid_xpu_cntr *xpu_recv_cntr,
			void *dest_addr, size_t dest_addr_size,
			void *recv_buf, size_t recv_len,
			void *recv_desc, size_t recv_desc_size,
			void *send_buf, size_t send_len,
			void *send_desc, size_t send_desc_size,
			int iters, int rx_depth, int is_client,
			int scope, int threads,
			cudaStream_t stream);

/**
 * ft_efa_xpu_run_bw - Run a bandwidth (one-directional) test on GPU.
 *
 * @xpu_ep:       Device-side XPU endpoint handle
 * @xpu_send_cq:  Device-side XPU send CQ handle
 * @xpu_send_cntr: Device-side XPU send counter (or NULL to use CQ)
 * @opcode:       0=send, 1=write, 2=writedata (write with imm), 3=read
 * @send_buf:     Send buffer address (device memory)
 * @send_len:     Send buffer length
 * @send_desc:    Send MR descriptor (device-side)
 * @send_desc_size: Size of send_desc
 * @dest_addr:    Device-side AV address of remote peer
 * @dest_addr_size: Size of dest_addr
 * @remote_addr:  Remote memory address (for RDMA write/read)
 * @remote_key:   Remote memory key (for RDMA write/read)
 * @iters:        Number of iterations
 * @tx_depth:     Transmit queue depth
 * @scope:        Cooperative scope the operations are issued at
 * @threads:      Threads in the block, split between the issuers of @scope
 * @stream:       CUDA stream for kernel launch
 *
 * Returns 0 on success, negative on failure.
 */
int ft_efa_xpu_run_bw(struct fid_xpu_ep *xpu_ep,
		  struct fid_xpu_cq *xpu_send_cq,
		  struct fid_xpu_cntr *xpu_send_cntr,
		  int opcode,
		  void *send_buf, size_t send_len,
		  void *send_desc, size_t send_desc_size,
		  void *dest_addr, size_t dest_addr_size,
		  uint64_t remote_addr, uint64_t remote_key,
		  int iters, int tx_depth, int scope, int threads,
		  cudaStream_t stream);

/**
 * ft_efa_xpu_run_bw_recv - Run receiver side of bandwidth test on GPU.
 *
 * @xpu_ep:       Device-side XPU endpoint handle
 * @xpu_recv_cq:  Device-side XPU recv CQ handle
 * @xpu_recv_cntr: Device-side XPU recv counter (or NULL to use CQ)
 * @recv_buf:     Receive buffer address (device memory)
 * @recv_len:     Receive buffer length
 * @recv_desc:    Receive MR descriptor (device-side)
 * @recv_desc_size: Size of recv_desc
 * @iters:        Number of iterations
 * @rx_depth:     Receive queue depth
 * @scope:        Cooperative scope the operations are issued at
 * @threads:      Threads in the block, split between the issuers of @scope
 * @stream:       CUDA stream for kernel launch
 *
 * Returns 0 on success, negative on failure.
 */
int ft_efa_xpu_run_bw_recv(struct fid_xpu_ep *xpu_ep,
		       struct fid_xpu_cq *xpu_recv_cq,
		       struct fid_xpu_cntr *xpu_recv_cntr,
		       void *recv_buf, size_t recv_len,
		       void *recv_desc, size_t recv_desc_size,
		       int iters, int rx_depth, int scope, int threads,
		       cudaStream_t stream);

/**
 * ft_efa_xpu_run_scope_reject - Check that a scope this build cannot support is
 * refused from the device.
 *
 * Every device entry point that takes a scope is called once and has to return
 * -FI_EOPNOTSUPP without posting anything, so an unsupported scope is an error
 * the caller sees rather than a kernel that never finishes.
 *
 * @xpu_ep:       Device-side XPU endpoint handle
 * @xpu_cq:       Device-side XPU CQ handle
 * @buf:          Any registered buffer (nothing is transferred)
 * @len:          Length of @buf
 * @desc:         MR descriptor for @buf (device-side)
 * @dest_addr:    Device-side AV address of remote peer
 * @scope:        The scope expected to be refused
 * @stream:       CUDA stream for kernel launch
 *
 * Returns 0 if every entry point refused the scope, -1 otherwise.
 */
int ft_efa_xpu_run_scope_reject(struct fid_xpu_ep *xpu_ep,
			    struct fid_xpu_cq *xpu_cq,
			    void *buf, size_t len, void *desc,
			    void *dest_addr, int scope,
			    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* EFA_XPU_KERNELS_H */
