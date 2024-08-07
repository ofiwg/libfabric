/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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
#include "hmem_util.h"

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

struct cudart_funcs {
	cudaError_t (*cudaSetDevice)(int device);
	cudaError_t (*cudaMalloc)(void** devPtr, size_t size);
	cudaError_t (*cudaFree)(void* devPtr);
	cudaError_t (*cudaMemcpy)(void* dst, const void* src, size_t count,
						enum cudaMemcpyKind kind);
	const char* (*cudaGetErrorString)(cudaError_t error);
};

struct hmem_cuda_state {
	void *cudart_lib;
	struct cudart_funcs fns;
	bool available;
};

static struct hmem_cuda_state cuda_state;

#ifndef count_of
#define count_of(x)	\
	((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
#endif

#define LD_CUDARTFN(_fn_name) do{						\
	int __i;								\
	const char *__names[] = {#_fn_name "_v2", #_fn_name };			\
	for(__i = 0;								\
	    __i < count_of(__names) && cuda_state.fns._fn_name == NULL;		\
	    __i++)								\
		cuda_state.fns._fn_name = dlsym(cuda_state.cudart_lib, __names[__i]);		\
	if (cuda_state.fns._fn_name == NULL) {					\
		error("Couldn't find cuda function " #_fn_name);		\
		rc = -ENOENT;							\
		goto err_close_libcudart;					\
	}									\
} while(0);

int hmem_cuda_memcpy_d2h(void *dst, void *src, size_t len)
{
	cudaError_t ce;
	if (!cuda_state.available) {
		error("Cuda not initalized");
		return -EOPNOTSUPP;
	}

	ce = cuda_state.fns.cudaMemcpy(dst, src, len, cudaMemcpyDeviceToHost);
	if (ce != cudaSuccess) {
		error("cuda %d (%s)", ce, cuda_state.fns.cudaGetErrorString(ce));
		return -EOPNOTSUPP;
	}
	return 0;
}

int hmem_cuda_memcpy_h2d(void *dst, void *src, size_t len)
{
	cudaError_t ce;
	if (!cuda_state.available) {
		error("Cuda not initalized");
		return -EOPNOTSUPP;
	}

	ce = cuda_state.fns.cudaMemcpy(dst, src, len, cudaMemcpyHostToDevice);
	if (ce != cudaSuccess) {
		error("cuda %d (%s)", ce, cuda_state.fns.cudaGetErrorString(ce));
		return -EOPNOTSUPP;
	}
	return 0;
}

int hmem_cuda_alloc(void *uaddr, size_t len)
{
	cudaError_t curet;
	const char *errstr;
	if (!cuda_state.available) {
		error("Cuda not initalized");
		return -EOPNOTSUPP;
	}

	curet = cuda_state.fns.cudaMalloc(uaddr, len);
	if (curet != cudaSuccess) {
		errstr = cuda_state.fns.cudaGetErrorString(curet);
		error("Couldn't allocate Cuda Memory (%d: %s)",
				curet, errstr);
		return -ENOMEM;
	}
	return 0;
}

void hmem_cuda_free(void *uaddr)
{
	if (!cuda_state.available) {
		error("Cuda not initalized");
		return;
	}
	cuda_state.fns.cudaFree(uaddr);
}

int hmem_cuda_init(void)
{
	char *err_str, *libcudart_str = "libcudart.so";
	int rc;
	cudaError_t cuerrort;

	cuda_state.cudart_lib = dlopen(libcudart_str, RTLD_LAZY);
	if (!cuda_state.cudart_lib) {
		err_str = dlerror();
		warn("%s", err_str ? err_str : "unknown error");
		rc = -ENOENT;
		goto err_exit;
	}
	LD_CUDARTFN(cudaSetDevice);
	LD_CUDARTFN(cudaMalloc);
	LD_CUDARTFN(cudaFree);
	LD_CUDARTFN(cudaMemcpy);
	LD_CUDARTFN(cudaGetErrorString);

	cuerrort = cuda_state.fns.cudaSetDevice(0);

	if (cuerrort != cudaSuccess) {
		warn("Failed to set cuda device");
		rc = -ENODATA;
		goto err_close_libcudart;
	}

	cuda_state.available = true;

	return 0;

err_close_libcudart:
	dlclose(cuda_state.cudart_lib);
	cuda_state.cudart_lib = NULL;
err_exit:
	return rc;
}

void hmem_cuda_cleanup(void)
{
	if (!cuda_state.available) return;
	cuda_state.fns.cudaSetDevice = NULL;
	cuda_state.fns.cudaMalloc = NULL;
	cuda_state.fns.cudaFree = NULL;
	cuda_state.fns.cudaMemcpy = NULL;
	cuda_state.fns.cudaGetErrorString = NULL;
	dlclose(cuda_state.cudart_lib);
	cuda_state.cudart_lib = NULL;
	cuda_state.available = false;
}

int run_fi_hmem_cuda_tag_d2d(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct ep_params ep_params = { 0 };
	struct rank_info *pri = NULL;
	struct mr_params mr_params = { 0 };
	const size_t buffer_lens[] = {(1<<16), (1<<13), (1<<9), 32};
	uint64_t tags[] = {0xffff1001, 0xffff2002, 0xffff3003, 0xffff4004};
	size_t nmsg = sizeof(tags)/sizeof(*tags);

	assert(nmsg == sizeof(buffer_lens)/sizeof(*buffer_lens));

	TRACE(ri, util_init(ri));

	mr_params.skip_reg = true;
	mr_params.hmem_iface = FI_HMEM_CUDA;
	if (my_node == NODE_A) {
		mr_params.access = FI_SEND;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV;
		mr_params.seed = seed_node_b;
	}
	for (int i = 0; i < nmsg; i++) {
		mr_params.length = buffer_lens[i];
		mr_params.idx = i;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
				fi_tsend(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					 0, NULL, pri->ep_info[0].fi_addr,
					 tags[0], NULL),
				0);

		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = 0x0;
		wait_tx_cq_params.flags = FI_TAGGED | FI_SEND;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
	} else {
		INSIST_FI_EQ(ri,
				fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					 buffer_lens[0], NULL, FI_ADDR_UNSPEC, tags[0],
					 0xffff, NULL),
				0);

		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = 0x0;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = 0;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	for (int i = 0; i < nmsg; i++) {
		if (my_node == NODE_B) {
			INSIST_FI_EQ(ri,
				     fi_trecv(ri->ep_info[0].fid, ri->mr_info[i].uaddr,
					      buffer_lens[i], NULL, FI_ADDR_UNSPEC, tags[i],
					      0xffff, NULL),
				     0);
		}

		TRACE(ri, util_barrier(ri));
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
				     fi_tsend(ri->ep_info[0].fid, ri->mr_info[i].uaddr,
					      buffer_lens[i], NULL, pri->ep_info[0].fi_addr,
					      tags[i], NULL),
				     0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = 0x0;
			wait_tx_cq_params.flags = FI_TAGGED | FI_SEND;
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		} else {
			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = 0x0;
			wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

			verify_buf_params.mr_idx = i;
			verify_buf_params.length = buffer_lens[i];
			verify_buf_params.expected_seed = seed_node_a;
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

static void send_wait(struct rank_info *ri, struct rank_info *pri,
							   size_t buf_size, uint64_t tag)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };

	INSIST_FI_EQ(ri,
			fi_tsend(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					 buf_size, NULL, pri->ep_info[0].fi_addr, tag, NULL),
			0);

	wait_tx_cq_params.ep_idx = 0;
	wait_tx_cq_params.context_val = 0x0;
	wait_tx_cq_params.flags = FI_TAGGED | FI_SEND;
	TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
}
static void recv_wait_validate(struct rank_info *ri, struct rank_info *pri,
							   size_t buf_size, uint64_t tag)
{
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };

	INSIST_FI_EQ(ri,
			fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					 buf_size, NULL, FI_ADDR_UNSPEC, tag, 0xffff, NULL),
			0);

	wait_rx_cq_params.ep_idx = 0;
	wait_rx_cq_params.context_val = 0x0;
	wait_rx_cq_params.flags = FI_TAGGED | FI_RECV;
	TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

	verify_buf_params.mr_idx = 0;
	verify_buf_params.length = 0;
	verify_buf_params.expected_seed = seed_node_a;
	TRACE(ri, util_verify_buf(ri, &verify_buf_params));
}

int run_fi_hmem_cuda_sendrecv_d2d(struct rank_info *ri)
{
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	struct rank_info *pri = NULL;
	uint64_t tag = 0xbeef, buf_size = 1<<14;

	TRACE(ri, util_init(ri));

	mr_params.skip_reg = true;
	mr_params.hmem_iface = FI_HMEM_CUDA;
	if (my_node == NODE_A) {
		mr_params.access = FI_SEND | FI_RECV;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_SEND | FI_RECV;
		mr_params.seed = seed_node_b;
	}
	mr_params.length = buf_size;
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_A) {
		send_wait(ri, pri, buf_size, tag);
		recv_wait_validate(ri, pri, buf_size, tag);
	} else {
		recv_wait_validate(ri, pri, buf_size, tag);
		send_wait(ri, pri, buf_size, tag);
	}

	TRACE(ri, util_teardown(ri, pri));
	return 0;
}

#else
inline int hmem_cuda_init(void) { return -ENOSYS; }
void hmem_cuda_cleanup(void){}
int hmem_cuda_alloc(void *uaddr, size_t len){return -ENOSYS;}
void hmem_cuda_free(void *uaddr){}
int hmem_cuda_memcpy_h2d(void *dst, void *src, size_t len){return -ENOSYS;}
int hmem_cuda_memcpy_d2h(void *dst, void *src, size_t len){return -ENOSYS;}
#endif
