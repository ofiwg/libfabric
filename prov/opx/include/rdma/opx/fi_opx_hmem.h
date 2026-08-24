/*
 * Copyright (C) 2023-2026 by Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HMEM_H_
#define _FI_PROV_OPX_HMEM_H_

#include <assert.h>
#include <rdma/hfi/hfi1_user.h>
#include "rdma/opx/fi_opx_compiler.h"
#include "rdma/opx/fi_opx_rma_ops.h"
#include "rdma/opx/opx_tracer.h"
#include "ofi_hmem.h"

#define OPX_HMEM_NO_HANDLE		   (0)
#define OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET (-1L)

enum opx_hmem_return_code {
	OPX_HMEM_ERROR = -1,
	OPX_HMEM_SUCCESS,
	OPX_HMEM_ERROR_NOT_READY,
};

#define OPX_HMEM_MEMCPY_ASYNC_DTOD 1

#ifdef OPX_HMEM
#define OPX_HMEM_DEV_REG_SEND_THRESHOLD (opx_ep->domain->hmem_domain->devreg_copy_from_threshold)
#define OPX_HMEM_DEV_REG_RECV_THRESHOLD (opx_ep->domain->hmem_domain->devreg_copy_to_threshold)
#else
#define OPX_HMEM_DEV_REG_SEND_THRESHOLD (OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET)
#define OPX_HMEM_DEV_REG_RECV_THRESHOLD (OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET)
#endif

struct fi_opx_hmem_info {
	union {
		struct {
			uint64_t device;
		} gpu;
		struct {
			struct fi_opx_mr *opx_mr;
		} dmabuf;
	};
	uint64_t	   hmem_dev_reg_handle;
	enum fi_hmem_iface iface;
	uint8_t		   is_unified;
	uint8_t		   unused[3];
} __attribute__((__packed__)) __attribute__((aligned(8)));

OPX_COMPILE_TIME_ASSERT((sizeof(struct fi_opx_hmem_info) & 0x7) == 0,
			"sizeof(fi_opx_hmem_info) should be a multiple of 8");

__OPX_FORCE_INLINE__
uint64_t opx_hmem_get_attr_device(enum fi_hmem_iface iface, const struct fi_mr_attr *attr)
{
	if (!attr) {
		return 0UL;
	}

	switch (iface) {
	case FI_HMEM_CUDA:
		return attr->device.cuda;
	case FI_HMEM_ZE:
		return attr->device.ze;
	case FI_HMEM_ROCR:
		return attr->device.rocr;
	case FI_HMEM_SYSTEM:
	default:
		return 0UL;
	}
}

__OPX_FORCE_INLINE__
void opx_hmem_set_mr_device(struct fi_mr_attr *attr, enum fi_hmem_iface iface, uint64_t device)
{
	switch (iface) {
	case FI_HMEM_CUDA:
		attr->device.cuda = (int) device;
		break;
	case FI_HMEM_ZE:
		attr->device.ze = (int) device;
		break;
	case FI_HMEM_ROCR:
		attr->device.rocr = (int) device;
		break;
	default:
		attr->device.reserved = device;
	}
}

__OPX_FORCE_INLINE__
int opx_hmem_is_device_iface(enum fi_hmem_iface iface)
{
	return iface != FI_HMEM_SYSTEM;
}

__OPX_FORCE_INLINE__
enum fi_hmem_iface opx_hmem_get_ptr_iface(const void *ptr, uint64_t *device, uint64_t *is_unified)
{
	*device = 0UL;
#ifdef OPX_HMEM

	uint64_t	   hmem_flags  = 0UL;
	enum fi_hmem_iface iface       = ofi_get_hmem_iface(ptr, device, &hmem_flags);
	bool		   device_only = !!(hmem_flags & FI_HMEM_DEVICE_ONLY);

	*is_unified = (hmem_flags & FI_HMEM_HOST_ALLOC) || (opx_hmem_is_device_iface(iface) && !device_only);
	if (*is_unified) {
		*device = 0UL;
		return FI_HMEM_SYSTEM;
	}
	OPX_TRACE_HMEM_INSTANT_COND(iface != FI_HMEM_SYSTEM, OPX_TRACE_EVENT_HMEM_DETECT, (uint64_t) iface, *device);
	return iface;
#endif

	*is_unified = 0UL;
	return FI_HMEM_SYSTEM;
}

__OPX_FORCE_INLINE__
enum fi_hmem_iface opx_hmem_get_mr_iface(const struct fi_opx_mr *desc, uint64_t *device, uint64_t *handle)
{
#ifdef OPX_HMEM
	if (desc && !desc->hmem_unified) {
		*device = opx_hmem_get_attr_device(desc->attr.iface, &desc->attr);
		*handle = (uint64_t) desc->attr.hmem_data;
		return desc->attr.iface;
	}
#endif
	*device = 0ul;
	*handle = OPX_HMEM_NO_HANDLE;
	return FI_HMEM_SYSTEM;
}

__OPX_FORCE_INLINE__
int opx_copy_to_hmem(enum fi_hmem_iface iface, uint64_t device, uint64_t hmem_handle, void *dest, const void *src,
		     size_t len, int64_t threshold)
{
	// These functions should never be called for regular host memory.
	// Calling this function directly should only ever be done in code
	// paths where we know iface != FI_HMEM_SYSTEM. Otherwise, the
	// OPX_HMEM_COPY_* macros should be used
	assert(iface != FI_HMEM_SYSTEM);

	int ret;

	assert(hmem_handle == OPX_HMEM_NO_HANDLE || threshold != OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);

	OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_COPY_TO, len, 0);
	switch (iface) {
#if HAVE_CUDA
	case FI_HMEM_CUDA:
		if ((hmem_handle != 0) && (len <= threshold)) {
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_GDRCOPY_TO_DEV, len, 0);
			cuda_gdrcopy_to_dev(hmem_handle, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_GDRCOPY_TO_DEV, len, 0);
			ret = 0;
		} else {
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_CUDAMEMCPY_TO, len, 0);
			ret = (int) ofi_cudaMemcpy(dest, src, len, cudaMemcpyHostToDevice);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_CUDAMEMCPY_TO, len, 0);
		}
		break;
#endif

#if HAVE_ROCR
	case FI_HMEM_ROCR:
		if ((hmem_handle != 0) && (len <= threshold)) {
			/* Perform a device registered copy */
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_AMD_DEV_REG_TO, len, 0);
			ret = rocr_dev_reg_copy_to_hmem(hmem_handle, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_AMD_DEV_REG_TO, len, 0);
		} else {
			/* Perform standard rocr_memcopy*/
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_AMD_ROCR_TO, len, 0);
			ret = rocr_copy_to_dev(device, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_AMD_ROCR_TO, len, 0);
		}
		break;
#endif

	default:
		OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_OFI_COPY_TO, len, 0);
		ret = ofi_copy_to_hmem(iface, device, dest, src, len);
		OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_OFI_COPY_TO, len, 0);
		break;
	}

	OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_COPY_TO, len, (uint64_t) ret);
	return ret;
}

__OPX_FORCE_INLINE__
int opx_copy_from_hmem(enum fi_hmem_iface iface, uint64_t device, uint64_t hmem_handle, void *dest, const void *src,
		       size_t len, int64_t threshold)
{
	// These functions should never be called for regular host memory.
	// Calling this function directly should only ever be done in code
	// paths where we know iface != FI_HMEM_SYSTEM. Otherwise, the
	// OPX_HMEM_COPY_* macros should be used
	assert(iface != FI_HMEM_SYSTEM);

	int ret;

	assert(hmem_handle == OPX_HMEM_NO_HANDLE || threshold != OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);

	OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_COPY_FROM, len, 0);
	switch (iface) {
#if HAVE_CUDA
	case FI_HMEM_CUDA:
		if ((hmem_handle != 0) && (len <= threshold)) {
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_GDRCOPY_FROM_DEV, len, 0);
			cuda_gdrcopy_from_dev(hmem_handle, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_GDRCOPY_FROM_DEV, len, 0);
			ret = 0;
		} else {
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_CUDAMEMCPY_FROM, len, 0);
			ret = (int) ofi_cudaMemcpy(dest, src, len, cudaMemcpyDeviceToHost);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_CUDAMEMCPY_FROM, len, 0);
		}
		break;
#endif

#if HAVE_ROCR
	case FI_HMEM_ROCR:
		if ((hmem_handle != 0) && (len <= threshold)) {
			/* Perform a device registered copy */
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_AMD_DEV_REG_FROM, len, 0);
			ret = rocr_dev_reg_copy_from_hmem(hmem_handle, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_AMD_DEV_REG_FROM, len, 0);
		} else {
			/* Perform standard rocr_memcopy*/
			OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_AMD_ROCR_FROM, len, 0);
			ret = rocr_copy_from_dev(device, dest, src, len);
			OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_AMD_ROCR_FROM, len, 0);
		}
		break;
#endif

	default:
		OPX_TRACE_HMEM_BEGIN(OPX_TRACE_EVENT_HMEM_OFI_COPY_FROM, len, 0);
		ret = ofi_copy_from_hmem(iface, device, dest, src, len);
		OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_OFI_COPY_FROM, len, 0);
		break;
	}

	OPX_TRACE_HMEM_END_SUCCESS(OPX_TRACE_EVENT_HMEM_COPY_FROM, len, (uint64_t) ret);
	return ret;
}

__OPX_FORCE_INLINE__
unsigned opx_hmem_iov_init(const void *buf, const size_t len, const void *desc, struct fi_opx_hmem_iov *iov,
			   uint64_t *handle)
{
	iov->buf = (uintptr_t) buf;
	iov->len = len;
#ifdef OPX_HMEM
	uint64_t	   hmem_device;
	enum fi_hmem_iface hmem_iface;
	if (desc) {
		hmem_iface = opx_hmem_get_mr_iface(desc, &hmem_device, handle);
	} else {
		uint64_t is_unified __attribute__((__unused__));
		hmem_iface = opx_hmem_get_ptr_iface(buf, &hmem_device, &is_unified);
		*handle	   = OPX_HMEM_NO_HANDLE;
	}
	iov->iface  = hmem_iface;
	iov->device = hmem_device;
	return (hmem_iface != FI_HMEM_SYSTEM);
#else
	iov->iface  = FI_HMEM_SYSTEM;
	iov->device = 0ul;
	*handle	    = OPX_HMEM_NO_HANDLE;
	return 0;
#endif
}

/*
 * Indexed by fi_hmem_iface. The FI_HMEM_ZE entry's type 1 is Level Zero's own
 * dma-buf export, not HFI1_MEMINFO_TYPE_DMABUF.
 */
static const unsigned OPX_HMEM_KERN_MEM_TYPE[4] = {
#ifdef OPX_HMEM
	HFI1_MEMINFO_TYPE_SYSTEM, HFI1_MEMINFO_TYPE_NVIDIA, 2, /* HFI1_MEMINFO_TYPE_AMD */
	1						       /* FI_HMEM_ZE: legacy ZE dma-buf type */
#endif
};

/*
 * Indexed by the kernel meminfo type, a 4-bit field decoded out of a request, so
 * callers must bounds-check before indexing. There is deliberately no
 * HFI1_MEMINFO_TYPE_DMABUF entry: fi_hmem_iface names the owning vendor
 * runtime, and a dma-buf request does not identify one.
 */
#ifdef OPX_HMEM
static const unsigned OPX_HMEM_OFI_MEM_TYPE[] = {
	[HFI1_MEMINFO_TYPE_SYSTEM] = FI_HMEM_SYSTEM,
	[HFI1_MEMINFO_TYPE_NVIDIA] = FI_HMEM_CUDA,
	[HFI1_MEMINFO_TYPE_AMD]	   = FI_HMEM_ROCR,
	[1]			   = FI_HMEM_ZE, /* legacy ZE dma-buf type */
};
#endif

#ifdef OPX_HMEM
#define OPX_HMEM_COPY_FROM(dst, src, len, handle, threshold, src_iface, src_device)                  \
	do {                                                                                         \
		if (src_iface == FI_HMEM_SYSTEM) {                                                   \
			memcpy(dst, src, len);                                                       \
		} else {                                                                             \
			opx_copy_from_hmem(src_iface, src_device, handle, dst, src, len, threshold); \
		}                                                                                    \
	} while (0)

#define OPX_HMEM_COPY_TO(dst, src, len, handle, threshold, dst_iface, dst_device)                  \
	do {                                                                                       \
		if (dst_iface == FI_HMEM_SYSTEM) {                                                 \
			memcpy(dst, src, len);                                                     \
		} else {                                                                           \
			opx_copy_to_hmem(dst_iface, dst_device, handle, dst, src, len, threshold); \
		}                                                                                  \
	} while (0)

#define OPX_HMEM_ATOMIC_DISPATCH(src, dst, len, dt, op, dst_iface, dst_device)                            \
	do {                                                                                              \
		if (dst_iface == FI_HMEM_SYSTEM) {                                                        \
			fi_opx_rx_atomic_dispatch(src, dst, len, dt, op);                                 \
		} else {                                                                                  \
			uint8_t hmem_buf[OPX_HFI1_MAX_PKT_SIZE];                                          \
			opx_copy_from_hmem(dst_iface, dst_device, OPX_HMEM_NO_HANDLE, hmem_buf, dst, len, \
					   OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);                           \
			fi_opx_rx_atomic_dispatch(src, hmem_buf, len, dt, op);                            \
			opx_copy_to_hmem(dst_iface, dst_device, OPX_HMEM_NO_HANDLE, dst, hmem_buf, len,   \
					 OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);                             \
		}                                                                                         \
	} while (0)

#else

#define OPX_HMEM_COPY_FROM(dst, src, len, handle, threshold, src_iface, src_device) \
	do {                                                                        \
		memcpy(dst, src, len);                                              \
		(void) src_iface;                                                   \
	} while (0)

#define OPX_HMEM_COPY_TO(dst, src, len, handle, threshold, dst_iface, dst_device) \
	do {                                                                      \
		memcpy(dst, src, len);                                            \
		(void) dst_iface;                                                 \
	} while (0)

#define OPX_HMEM_ATOMIC_DISPATCH(src, dst, len, dt, op, dst_iface, dst_device) \
	do {                                                                   \
		fi_opx_rx_atomic_dispatch(src, dst, len, dt, op);              \
	} while (0)

#endif // OPX_HMEM

#if HAVE_CUDA
#ifdef OPX_HMEM
#include "rdma/opx/opx_hmem_domain.h"
#endif

__OPX_FORCE_INLINE__
void opx_hmem_cuda_dbg_trace(char *string, CUresult result)
{
	const char *error_string = NULL;

	ofi_cuGetErrorString(result, &error_string);
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "%s CUresult=%d (%s)\n", string, result,
		     error_string ? error_string : "unknown error");
}

__OPX_FORCE_INLINE__
void opx_hmem_cuda_warn_trace(char *string, CUresult result)
{
	const char *error_string = NULL;

	ofi_cuGetErrorString(result, &error_string);
	FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "%s CUresult=%d (%s)\n", string, result,
		error_string ? error_string : "unknown error");
}

__OPX_FORCE_INLINE__
void opx_hmem_cuda_stream_synchronize(CUstream stream)
{
	CUresult result = ofi_cuStreamSynchronize(stream);

	if (result) {
		opx_hmem_cuda_warn_trace("Error synchronizing the CUDA stream", result);
		abort();
	}
}

__OPX_FORCE_INLINE__
int opx_hmem_cuda_stream_create(CUstream *stream)
{
	CUresult result = ofi_cuStreamCreate(stream, CU_STREAM_NON_BLOCKING);

	if (result) {
		opx_hmem_cuda_dbg_trace("Error creating the CUDA stream", result);
		return OPX_HMEM_ERROR;
	}
	return OPX_HMEM_SUCCESS;
}

__OPX_FORCE_INLINE__
void opx_hmem_cuda_stream_destroy(CUstream stream)
{
	if (stream) {
		opx_hmem_cuda_stream_synchronize(stream);
		ofi_cuStreamDestroy(stream);
	}
}

__OPX_FORCE_INLINE__
int opx_hmem_cuda_event_create(CUevent *event)
{
	CUresult result = ofi_cuEventCreate(event, CU_EVENT_DISABLE_TIMING);

	if (result) {
		opx_hmem_cuda_dbg_trace("Error creating the CUDA event", result);
		return OPX_HMEM_ERROR;
	}
	return OPX_HMEM_SUCCESS;
}

__OPX_FORCE_INLINE__
void opx_hmem_cuda_event_destroy(CUevent *event)
{
	if (*event) {
		ofi_cuEventDestroy(*event);
		*event = NULL;
	}
}

__OPX_FORCE_INLINE__
int opx_hmem_cuda_event_record(CUevent event, CUstream stream)
{
	CUresult result = ofi_cuEventRecord(event, stream);

	if (result) {
		opx_hmem_cuda_dbg_trace("Error recording an event on the CUDA stream", result);
		return OPX_HMEM_ERROR;
	}
	return OPX_HMEM_SUCCESS;
}

__OPX_FORCE_INLINE__
int opx_hmem_cuda_event_synchronize(CUevent *event)
{
	CUresult result = ofi_cuEventSynchronize(*event);

	opx_hmem_cuda_event_destroy(event);
	if (result) {
		opx_hmem_cuda_dbg_trace("Error on CUDA event synchronize", result);
		abort();
	}
	return OPX_HMEM_SUCCESS;
}

__OPX_FORCE_INLINE__
enum opx_hmem_return_code opx_hmem_cuda_event_query(CUevent event)
{
	CUresult result = ofi_cuEventQuery(event);

	if (result == CUDA_SUCCESS) {
		return OPX_HMEM_SUCCESS;
	}
	if (result == CUDA_ERROR_NOT_READY) {
		return OPX_HMEM_ERROR_NOT_READY;
	}
	return OPX_HMEM_ERROR;
}

__OPX_FORCE_INLINE__
int opx_hmem_cuda_memcpy_async_DtoD(void *dst, const void *src, size_t size, CUstream stream)
{
	CUresult result = ofi_cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr) src, size, stream);

	if (result) {
		opx_hmem_cuda_dbg_trace("Error on the asynchronous CUDA copy", result);
		return OPX_HMEM_ERROR;
	}
	return OPX_HMEM_SUCCESS;
}

#ifdef OPX_HMEM
__OPX_FORCE_INLINE__
void opx_hmem_cuda_memcpy_async(uint64_t device, void *dst, const void *src, size_t size,
				struct opx_hmem_domain *domain, CUevent *event, int copy_type)
{
	CUevent new_event = NULL;
	int	ret;

	if (domain->cuda.stream == NULL) {
		ret = opx_hmem_cuda_stream_create(&domain->cuda.stream);
		if (ret) {
			goto err;
		}
	}

	ret = opx_hmem_cuda_event_create(&new_event);
	if (ret) {
		goto err;
	}

	assert(copy_type == OPX_HMEM_MEMCPY_ASYNC_DTOD);
	ret = opx_hmem_cuda_memcpy_async_DtoD(dst, src, size, domain->cuda.stream);
	if (ret) {
		opx_hmem_cuda_event_destroy(&new_event);
		goto err;
	}

	ret = opx_hmem_cuda_event_record(new_event, domain->cuda.stream);
	if (ret) {
		opx_hmem_cuda_stream_synchronize(domain->cuda.stream);
		opx_hmem_cuda_event_destroy(&new_event);
		goto err;
	}

	*event = new_event;
	return;

err:
	*event = NULL;
	ret    = ofi_copy_to_hmem(FI_HMEM_CUDA, device, dst, src, size);
	if (ret) {
		opx_hmem_cuda_dbg_trace("Error trying to synchronously copy", ret);
		abort();
	}
}
#endif
#endif

#endif
