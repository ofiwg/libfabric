/*
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP
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

#include "ofi_mr.h"

#ifdef HAVE_LIBCUDA

#include <cuda.h>
#include <cuda_runtime.h>

static int cuda_mm_subscribe(struct ofi_mem_monitor *notifier, const void *addr,
			     size_t len, union ofi_mr_hmem_info *hmem_info)
{
	CUresult ret;

	ret = cuPointerGetAttribute(&hmem_info->cuda_id,
				    CU_POINTER_ATTRIBUTE_BUFFER_ID,
				    (CUdeviceptr)addr);
	if (ret == CUDA_SUCCESS) {
		FI_DBG(&core_prov, FI_LOG_MR,
		       "Assigned CUDA buffer ID %lu to buffer %p\n",
		       hmem_info->cuda_id, addr);
		return FI_SUCCESS;
	}

	FI_WARN(&core_prov, FI_LOG_MR,
		"Failed to get CUDA buffer ID for buffer %p len %lu\n"
		"cuPointerGetAttribute() failed: %s:%s\n", addr, len,
		cudaGetErrorName(ret), cudaGetErrorString(ret));

	return -FI_EFAULT;
}

static void cuda_mm_unsubscribe(struct ofi_mem_monitor *notifier,
				const void *addr, size_t len,
				const union ofi_mr_hmem_info *hmem_info)
{
	/* no-op */
}

static bool cuda_mm_valid(struct ofi_mem_monitor *notifier,
			  const void *addr, size_t len,
			  const union ofi_mr_hmem_info *hmem_info)
{
	uint64_t id;
	CUresult ret;

	/* CUDA buffer IDs are associated for each CUDA monitor entry. If the
	 * device pages backing the device virtual address change, a different
	 * buffer ID is associated with this mapping.
	 */
	ret = cuPointerGetAttribute(&id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
				    (CUdeviceptr)addr);
	if (ret == CUDA_SUCCESS && hmem_info->cuda_id == id) {
		FI_DBG(&core_prov, FI_LOG_MR,
		       "CUDA buffer ID %lu still valid for buffer %p\n",
		       hmem_info->cuda_id, addr);
		return true;
	} else if (ret == CUDA_SUCCESS && hmem_info->cuda_id != id) {
		FI_DBG(&core_prov, FI_LOG_MR,
		       "CUDA buffer ID %lu invalid for buffer %p\n",
		       hmem_info->cuda_id, addr);
	} else {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed to get CUDA buffer ID for buffer %p len %lu\n"
			"cuPointerGetAttribute() failed: %s:%s\n", addr, len,
			cudaGetErrorName(ret), cudaGetErrorString(ret));
	}

	return false;
}

int cuda_monitor_start(void)
{
	/* no-op */
	return FI_SUCCESS;
}

#else

static int cuda_mm_subscribe(struct ofi_mem_monitor *notifier, const void *addr,
			     size_t len, union ofi_mr_hmem_info *hmem_info)
{
	return -FI_ENOSYS;
}

static void cuda_mm_unsubscribe(struct ofi_mem_monitor *notifier,
				const void *addr, size_t len,
				const union ofi_mr_hmem_info *hmem_info)
{
}

static bool cuda_mm_valid(struct ofi_mem_monitor *notifier,
			  const void *addr, size_t len,
			  const union ofi_mr_hmem_info *hmem_info)
{
	return false;
}

int cuda_monitor_start(void)
{
	return -FI_ENOSYS;
}

#endif /* HAVE_LIBCUDA */

static struct ofi_mem_monitor cuda_mm = {
	.iface = FI_HMEM_CUDA,
	.init = ofi_monitor_init,
	.cleanup = ofi_monitor_cleanup,
	.subscribe = cuda_mm_subscribe,
	.unsubscribe = cuda_mm_unsubscribe,
	.valid = cuda_mm_valid,
};

struct ofi_mem_monitor *cuda_monitor = &cuda_mm;

void cuda_monitor_stop(void)
{
	/* no-op */
}
