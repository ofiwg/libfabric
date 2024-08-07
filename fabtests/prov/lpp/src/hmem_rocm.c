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

#ifdef USE_ROCM
#define __HIP_PLATFORM_AMD__

#include <hip/hip_runtime_api.h>

struct rocm_funcs {
	hipError_t (*hipMalloc)(void **ptr, size_t size);
	void (*hipFree)(void *ptr);
	hipError_t (*hipMemcpy)(void *dst, const void *src, size_t sizeBytes,
							enum hipMemcpyKind kind);
	char * (*hipGetErrorString)(hipError_t hipError);
};

struct hmem_rocm_state {
	void *hip_lib;
	struct rocm_funcs fns;
	bool available;
};

static struct hmem_rocm_state rocm_state;

#define LD_ROCMFN(_fn_name) do{							\
	const char *__name = #_fn_name ;					\
	rocm_state.fns._fn_name = dlsym(rocm_state.hip_lib, __name);		\
	if (rocm_state.fns._fn_name == NULL) {					\
		error("Couldn't find rocm function " #_fn_name);		\
		rc = -ENOENT;							\
		goto err_close_libhip;						\
	}									\
} while(0);									\

int hmem_rocm_init(void)
{
	char *err_str, *libhip_str = "libamdhip64.so";
	int rc;
	rocm_state.hip_lib = dlopen(libhip_str , RTLD_LAZY);

	if(!rocm_state.hip_lib) {
		err_str = dlerror();
		warn("%s", err_str ? err_str : "unknown error");
		rc = -ENOENT;
		goto err_exit;
	}

	LD_ROCMFN(hipMalloc);
	LD_ROCMFN(hipFree);
	LD_ROCMFN(hipMemcpy);
	LD_ROCMFN(hipGetErrorString);

	rocm_state.available = true;
	return 0;

err_close_libhip:
	dlclose(rocm_state.hip_lib);
	rocm_state.hip_lib = NULL;
err_exit:
	return rc;
}

void hmem_rocm_cleanup(void)
{
	if(!rocm_state.available) return;
	rocm_state.fns.hipMalloc = NULL;
	rocm_state.fns.hipFree = NULL;
	rocm_state.fns.hipMemcpy = NULL;
	rocm_state.fns.hipGetErrorString = NULL;
	dlclose(rocm_state.hip_lib);
	rocm_state.hip_lib = NULL;
	rocm_state.available = false;
}

int hmem_rocm_memcpy_h2d(void *dst, const void *src, size_t len)
{
	hipError_t e;
	char *errstr;
	if (!rocm_state.available) {
		error("ROCM not initalized");
		return -EOPNOTSUPP;
	}

	e = rocm_state.fns.hipMemcpy(dst, src, len, hipMemcpyHostToDevice);
	if (e != hipSuccess) {
		errstr = rocm_state.fns.hipGetErrorString(e);
		error("Couldn't allocate ROCM memory (%d: %s)", e, errstr);
		return -ENOMEM;
	}
	return 0;
}

int hmem_rocm_memcpy_d2h(void *dst, const void *src, size_t len)
{
	hipError_t e;
	char *errstr;
	if (!rocm_state.available) {
		error("ROCM not initalized");
		return -EOPNOTSUPP;
	}

	e = rocm_state.fns.hipMemcpy(dst, src, len, hipMemcpyDeviceToHost);
	if (e != hipSuccess) {
		errstr = rocm_state.fns.hipGetErrorString(e);
		error("Couldn't allocate ROCM memory (%d: %s)", e, errstr);
		return -ENOMEM;
	}
	return 0;
}

int hmem_rocm_alloc(void *uaddr, size_t len)
{
	hipError_t e;
	char *errstr;
	if (!rocm_state.available) {
		error("ROCM not initalized");
		return -EOPNOTSUPP;
	}

	e = rocm_state.fns.hipMalloc(uaddr, len);
	if (e != hipSuccess) {
		errstr = rocm_state.fns.hipGetErrorString(e);
		error("Couldn't allocate ROCM memory (%d: %s)", e, errstr);
		return -ENOMEM;
	}
	return 0;
}

void hmem_rocm_free(void *uaddr)
{
	if (!rocm_state.available) {
		error("ROCM not initalized");
		return;
	}

	rocm_state.fns.hipFree(uaddr);
}


int run_fi_hmem_rocm_tag_d2d(struct rank_info *ri)
{
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct ep_params ep_params = { 0 };
	struct rank_info *pri = NULL;
	struct mr_params mr_params = { 0 };
	const size_t buffer_len = (1<<13);
	uint64_t tag = 0xffff1001;

	TRACE(ri, util_init(ri));

	mr_params.idx = 0;
	mr_params.length = buffer_len;
	mr_params.skip_reg = true;
	mr_params.hmem_iface = FI_HMEM_ROCR;
	if (my_node == NODE_A) {
		mr_params.access = FI_SEND;
		mr_params.seed = seed_node_a;
	} else {
		mr_params.access = FI_RECV;
		mr_params.seed = seed_node_b;
	}
	TRACE(ri, util_create_mr(ri, &mr_params));

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC, tag,
				     0xffff, NULL),
			     0);

	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		INSIST_FI_EQ(ri,
			     fi_tsend(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				      buffer_len, NULL, pri->ep_info[0].fi_addr,
				      tag, NULL),
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

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = buffer_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

#else

inline int hmem_rocm_init(void) { return -ENOSYS; }
void hmem_rocm_close(void){}
int hmem_rocm_alloc(void *uaddr, size_t len){return -ENOSYS;}
void hmem_rocm_free(void *uaddr){}
int hmem_rocm_memcpy_h2d(void *dst, const void *src, size_t len){return -ENOSYS;}
int hmem_rocm_memcpy_d2h(void *dst, const void *src, size_t len){return -ENOSYS;}

#endif
