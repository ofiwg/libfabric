/*
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP
 * (C) Copyright Amazon.com, Inc. or its affiliates.
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

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "ofi_hmem.h"
#include "ofi.h"

#if HAVE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#if ENABLE_CUDA_DLOPEN
#include <dlfcn.h>
#endif

/*
 * Convenience higher-order macros for enumerating CUDA driver/runtime API
 * function names
 */
#define CUDA_DRIVER_FUNCS_DEF(_)	\
	_(cuGetErrorName)		\
	_(cuGetErrorString)		\
	_(cuPointerGetAttribute)	\
	_(cuPointerSetAttribute)	\
	_(cuMemGetAddressRange)

#define CUDA_RUNTIME_FUNCS_DEF(_)	\
	_(cudaMemcpy)			\
	_(cudaDeviceSynchronize)	\
	_(cudaFree)			\
	_(cudaMalloc)			\
	_(cudaGetErrorName)		\
	_(cudaGetErrorString)		\
	_(cudaHostRegister)		\
	_(cudaHostUnregister)		\
	_(cudaGetDeviceCount)		\
	_(cudaGetDevice)		\
	_(cudaSetDevice)		\
	_(cudaIpcOpenMemHandle)		\
	_(cudaIpcGetMemHandle)		\
	_(cudaIpcCloseMemHandle)

static struct {
	int   device_count;
	bool  use_gdrcopy;
	bool  use_ipc;
	void *driver_handle;
	void *runtime_handle;
	enum cuda_xfer_setting xfer_setting;
} cuda_attr = {
	.device_count   = -1,
	.use_gdrcopy    = false,
	.use_ipc        = false,
	.driver_handle  = NULL,
	.runtime_handle = NULL,
	.xfer_setting   = CUDA_XFER_UNSPECIFIED
};

static struct {
	cudaError_t (*cudaMemcpy)(void *dst, const void *src, size_t size,
				  enum cudaMemcpyKind kind);
	cudaError_t (*cudaDeviceSynchronize)(void);
	cudaError_t (*cudaFree)(void* ptr);
	cudaError_t (*cudaMalloc)(void** ptr, size_t size);
	const char *(*cudaGetErrorName)(cudaError_t error);
	const char *(*cudaGetErrorString)(cudaError_t error);
	CUresult (*cuGetErrorName)(CUresult error, const char** pStr);
	CUresult (*cuGetErrorString)(CUresult error, const char** pStr);
	CUresult (*cuPointerGetAttribute)(void *data,
					  CUpointer_attribute attribute,
					  CUdeviceptr ptr);
	CUresult (*cuPointerSetAttribute)(const void *data,
					  CUpointer_attribute attribute,
					  CUdeviceptr ptr);
	CUresult (*cuMemGetAddressRange)( CUdeviceptr* pbase,
					  size_t* psize, CUdeviceptr dptr);
	cudaError_t (*cudaHostRegister)(void *ptr, size_t size,
					unsigned int flags);
	cudaError_t (*cudaHostUnregister)(void *ptr);
	cudaError_t (*cudaGetDeviceCount)(int *count);
	cudaError_t (*cudaGetDevice)(int *device);
	cudaError_t (*cudaSetDevice)(int device);
	cudaError_t (*cudaIpcOpenMemHandle)(void **devptr,
					    cudaIpcMemHandle_t handle,
					    unsigned int flags);
	cudaError_t (*cudaIpcGetMemHandle)(cudaIpcMemHandle_t *handle,
					   void *devptr);
	cudaError_t (*cudaIpcCloseMemHandle)(void *devptr);
} cuda_ops
#if !ENABLE_CUDA_DLOPEN
#define CUDA_OPS_INIT(sym) .sym = sym,
= {
	CUDA_DRIVER_FUNCS_DEF(CUDA_OPS_INIT)
	CUDA_RUNTIME_FUNCS_DEF(CUDA_OPS_INIT)
}
#endif
;

static cudaError_t cuda_disabled_cudaMemcpy(void *dst, const void *src,
					    size_t size, enum cudaMemcpyKind kind);

cudaError_t ofi_cudaMemcpy(void *dst, const void *src, size_t size,
			   enum cudaMemcpyKind kind)
{
	cudaError_t cuda_ret;
	CUcontext data;
	cuda_ret = cuda_ops.cudaMemcpy(dst, src, size, kind);
	if (cuda_ret != cudaSuccess)
		return cuda_ret;

	/* If either dst or src buffer is not allocated,
	 * mapped by, or registered with a CUcontext, the
	 * cuPointerGetAttribute call will return
	 * CUDA_ERROR_INVALID_VALUE. In this case the
	 * cudaDeviceSynchronize() needs to be called
	 * to ensure data consistency.
	 */
	if (ofi_cuPointerGetAttribute(&data,
	    CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr) dst) == CUDA_SUCCESS
	    && ofi_cuPointerGetAttribute(&data,
	    CU_POINTER_ATTRIBUTE_CONTEXT, (CUdeviceptr) src) == CUDA_SUCCESS)
		return cudaSuccess;
	FI_WARN_ONCE(&core_prov, FI_LOG_CORE,
		"Either dst or src buffer of cudaMemcpy is not allocated or registered"
		" by cuda device. cudaDeviceSynchronize() will be performed to ensure"
		" data consistency. The performance may be impacted.\n");
	return cuda_ops.cudaDeviceSynchronize();
}

const char *ofi_cudaGetErrorName(cudaError_t error)
{
	return cuda_ops.cudaGetErrorName(error);
}

const char *ofi_cudaGetErrorString(cudaError_t error)
{
	return cuda_ops.cudaGetErrorString(error);
}

CUresult ofi_cuGetErrorName(CUresult error, const char** pStr)
{
	return cuda_ops.cuGetErrorName(error, pStr);
}

CUresult ofi_cuGetErrorString(CUresult error, const char** pStr)
{
	return cuda_ops.cuGetErrorString(error, pStr);
}

CUresult ofi_cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
				   CUdeviceptr ptr)
{
	return cuda_ops.cuPointerGetAttribute(data, attribute, ptr);
}

/**
 * @brief Set CU_POINTER_ATTRIBUTE_SYNC_MEMOPS for a cuda ptr
 * to ensure any synchronous copies are completed prior
 * to any other access of the memory region, which ensure
 * the data consistency for CUDA IPC.
 *
 * @param ptr the cuda ptr
 * @return int 0 on success, -FI_EINVAL on failure.
 */
int cuda_set_sync_memops(void *ptr)
{
	CUresult cu_result;
	const char *cu_error_name;
	const char *cu_error_str;
	int true_flag = 1;

	cu_result = cuda_ops.cuPointerSetAttribute(&true_flag,
					CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
					(CUdeviceptr) ptr);
	if (cu_result == CUDA_SUCCESS)
		return FI_SUCCESS;

	ofi_cuGetErrorName(cu_result, &cu_error_name);
	ofi_cuGetErrorString(cu_result, &cu_error_str);
	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cuPointerSetAttribute: %s:%s\n",
		cu_error_name, cu_error_str);
	return -FI_EINVAL;
}

CUresult ofi_cuMemGetAddressRange(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr)
{
	return cuda_ops.cuMemGetAddressRange(pbase, psize, dptr);
}

cudaError_t ofi_cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
	return cuda_ops.cudaHostRegister(ptr, size, flags);
}

cudaError_t ofi_cudaHostUnregister(void *ptr)
{
	return cuda_ops.cudaHostUnregister(ptr);
}

static cudaError_t ofi_cudaGetDeviceCount(int *count)
{
	return cuda_ops.cudaGetDeviceCount(count);
}

static bool ofi_cudaIsDeviceId(uint64_t device) {
	return device < cuda_attr.device_count;
}

cudaError_t ofi_cudaMalloc(void **ptr, size_t size)
{
	return cuda_ops.cudaMalloc(ptr, size);
}

cudaError_t ofi_cudaFree(void *ptr)
{
	return cuda_ops.cudaFree(ptr);
}

int cuda_copy_to_dev(uint64_t device, void *dst, const void *src, size_t size)
{
	if (cuda_attr.use_gdrcopy && !ofi_cudaIsDeviceId(device)) {
		cuda_gdrcopy_to_dev(device, dst, src, size);
		return FI_SUCCESS;
	}

	cudaError_t cuda_ret;

	cuda_ret = ofi_cudaMemcpy(dst, src, size, cudaMemcpyDefault);
	if (cuda_ret == cudaSuccess)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaMemcpy: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return -FI_EIO;
}

int cuda_copy_from_dev(uint64_t device, void *dst, const void *src, size_t size)
{
	if (cuda_attr.use_gdrcopy && !ofi_cudaIsDeviceId(device)) {
		cuda_gdrcopy_from_dev(device, dst, src, size);
		return FI_SUCCESS;
	}

	cudaError_t cuda_ret;

	cuda_ret = ofi_cudaMemcpy(dst, src, size, cudaMemcpyDefault);
	if (cuda_ret == cudaSuccess)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaMemcpy: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return -FI_EIO;
}

int cuda_dev_register(struct fi_mr_attr *mr_attr, uint64_t *handle)
{
	if (cuda_attr.use_gdrcopy)
		return cuda_gdrcopy_dev_register(mr_attr, handle);

	*handle = mr_attr->device.cuda;
	return FI_SUCCESS;
}

int cuda_dev_unregister(uint64_t handle)
{
	if (cuda_attr.use_gdrcopy)
		return cuda_gdrcopy_dev_unregister(handle);

	return FI_SUCCESS;
}

int cuda_get_handle(void *dev_buf, size_t size, void **handle)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaIpcGetMemHandle((cudaIpcMemHandle_t *)handle,
						dev_buf);
	if (cuda_ret != cudaSuccess) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform cudaIpcGetMemHandle: %s:%s\n",
			ofi_cudaGetErrorName(cuda_ret),
			ofi_cudaGetErrorString(cuda_ret));
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

int cuda_open_handle(void **handle, size_t size, uint64_t device,
		     void **ipc_ptr)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaIpcOpenMemHandle(ipc_ptr,
						 *(cudaIpcMemHandle_t *)handle,
						 cudaIpcMemLazyEnablePeerAccess);

	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaIpcOpenMemHandle: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return (cuda_ret == cudaErrorAlreadyMapped) ? -FI_EALREADY:-FI_EINVAL;
}

int cuda_close_handle(void *ipc_ptr)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaIpcCloseMemHandle(ipc_ptr);

	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaIpcCloseMemHandle: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return -FI_EINVAL;
}

int cuda_get_base_addr(const void *ptr, void **base, size_t *size)
{
	CUresult cu_result;
	const char *cu_error_name;
	const char *cu_error_str;

	cu_result = ofi_cuMemGetAddressRange((CUdeviceptr *)base,
					      size, (CUdeviceptr) ptr);
	if (cu_result == CUDA_SUCCESS)
		return FI_SUCCESS;

	ofi_cuGetErrorName(cu_result, &cu_error_name);
	ofi_cuGetErrorString(cu_result, &cu_error_str);
	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cuMemGetAddressRange: %s:%s\n",
		cu_error_name, cu_error_str);
	return -FI_EIO;
}

static cudaError_t cuda_disabled_cudaMemcpy(void *dst, const void *src,
					    size_t size, enum cudaMemcpyKind kind)
{
	FI_WARN(&core_prov, FI_LOG_CORE,
		"cudaMemcpy was called but FI_HMEM_CUDA_ENABLE_XFER = 0, "
		"no copy will occur to prevent deadlock.");

	return cudaErrorInvalidValue;
}

/*
 * Convenience macro to dynamically load symbols in cuda_ops struct
 * Trailing semicolon is necessary for use with higher-order macro
 */
#define CUDA_FUNCS_DLOPEN(type, sym)					\
	do {								\
		cuda_ops.sym = dlsym(cuda_attr.type##_handle, #sym);	\
		if (!cuda_ops.sym) {					\
			FI_WARN(&core_prov, FI_LOG_CORE,		\
				"Failed to find " #sym "\n");		\
			goto err_dlclose_cuda_driver;			\
		}							\
	} while (0);

#define CUDA_DRIVER_FUNCS_DLOPEN(sym)  CUDA_FUNCS_DLOPEN(driver,  sym)
#define CUDA_RUNTIME_FUNCS_DLOPEN(sym) CUDA_FUNCS_DLOPEN(runtime, sym)

static int cuda_hmem_dl_init(void)
{
#if ENABLE_CUDA_DLOPEN
	/* Assume failure to dlopen CUDA runtime is caused by the library not
	 * being found. Thus, CUDA is not supported.
	 */
	cuda_attr.runtime_handle = dlopen("libcudart.so", RTLD_NOW);
	if (!cuda_attr.runtime_handle) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libcudart.so\n");
		return -FI_ENOSYS;
	}

	cuda_attr.driver_handle = dlopen("libcuda.so", RTLD_NOW);
	if (!cuda_attr.driver_handle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libcuda.so\n");
		goto err_dlclose_cuda_runtime;
	}

	CUDA_DRIVER_FUNCS_DEF(CUDA_DRIVER_FUNCS_DLOPEN)
	CUDA_RUNTIME_FUNCS_DEF(CUDA_RUNTIME_FUNCS_DLOPEN)

	return FI_SUCCESS;

err_dlclose_cuda_driver:
	dlclose(cuda_attr.driver_handle);
err_dlclose_cuda_runtime:
	dlclose(cuda_attr.runtime_handle);

	return -FI_ENODATA;
#else
	return FI_SUCCESS;
#endif /* ENABLE_CUDA_DLOPEN */
}

static void cuda_hmem_dl_cleanup(void)
{
#if ENABLE_CUDA_DLOPEN
	dlclose(cuda_attr.driver_handle);
	dlclose(cuda_attr.runtime_handle);
#endif
}

static int cuda_hmem_verify_devices(void)
{
	cudaError_t cuda_ret;

	/* Verify CUDA compute-capable devices are present on the host. */
	cuda_ret = ofi_cudaGetDeviceCount(&cuda_attr.device_count);
	switch (cuda_ret) {
	case cudaSuccess:
		break;

	case cudaErrorNoDevice:
		return -FI_ENOSYS;

	default:
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform cudaGetDeviceCount: %s:%s\n",
			ofi_cudaGetErrorName(cuda_ret),
			ofi_cudaGetErrorString(cuda_ret));
		return -FI_EIO;
	}

	if (cuda_attr.device_count <= 0)
		return -FI_ENOSYS;

	return FI_SUCCESS;
}

int cuda_hmem_init(void)
{
	int ret, param_value;
	int gdrcopy_ret;

	fi_param_define(NULL, "hmem_cuda_use_gdrcopy", FI_PARAM_BOOL,
			"Use gdrcopy to copy data to/from CUDA GPU memory. "
			"If libfabric is not compiled with gdrcopy support, "
			"this variable is not checked. (default: true)");
	fi_param_define(NULL, "hmem_cuda_enable_xfer", FI_PARAM_BOOL,
			"Enable use of CUDA APIs for copying data to/from CUDA "
			"GPU memory. This should be disabled if CUDA "
			"operations on the default stream would result in a "
			"deadlock in the application. (default: each provider "
			"decide whether to use CUDA API).");

	ret = cuda_hmem_dl_init();
	if (ret != FI_SUCCESS)
		return ret;

	ret = cuda_hmem_verify_devices();
	if (ret != FI_SUCCESS)
		goto dl_cleanup;

	ret = 1;
	fi_param_get_bool(NULL, "hmem_cuda_use_gdrcopy",
			  &ret);
	cuda_attr.use_gdrcopy = (ret != 0);
	if (cuda_attr.use_gdrcopy) {
		gdrcopy_ret = cuda_gdrcopy_hmem_init();
		if (gdrcopy_ret != FI_SUCCESS) {
			cuda_attr.use_gdrcopy = false;
			if (gdrcopy_ret != -FI_ENOSYS)
				FI_WARN(&core_prov, FI_LOG_CORE,
					"gdrcopy initialization failed! "
					"gdrcopy will not be used.\n");
		}
	}

	param_value = 1;
	ret = fi_param_get_bool(NULL, "hmem_cuda_enable_xfer", &param_value);
	if (!ret)
		cuda_attr.xfer_setting = (param_value > 0)
			? CUDA_XFER_ENABLED
			: CUDA_XFER_DISABLED;

	FI_INFO(&core_prov, FI_LOG_CORE, "cuda_xfer_setting: %d\n", cuda_attr.xfer_setting);

	if (cuda_attr.xfer_setting == CUDA_XFER_DISABLED)
		cuda_ops.cudaMemcpy = cuda_disabled_cudaMemcpy;

	/*
	 * CUDA IPC is only enabled if cudaMemcpy can be used.
	 */
	cuda_attr.use_ipc = (cuda_attr.xfer_setting != CUDA_XFER_DISABLED);

	return FI_SUCCESS;

dl_cleanup:
	cuda_hmem_dl_cleanup();

	return ret;
}

int cuda_hmem_cleanup(void)
{
	cuda_hmem_dl_cleanup();
	if (cuda_attr.use_gdrcopy)
		cuda_gdrcopy_hmem_cleanup();
	return FI_SUCCESS;
}

bool cuda_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	CUresult cuda_ret;
	unsigned int data;

	cuda_ret = ofi_cuPointerGetAttribute(&data,
					     CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
					     (CUdeviceptr)addr);
	switch (cuda_ret) {
	case CUDA_SUCCESS:
		if (data == CU_MEMORYTYPE_DEVICE) {
			if (flags)
				*flags = FI_HMEM_DEVICE_ONLY;

			if (device) {
				*device = 0;
				cuda_ret = ofi_cuPointerGetAttribute(
						(int *) device,
						CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
						(CUdeviceptr) addr);
				if (cuda_ret)
					break;
			}
			return true;
		}
		break;

	/* Returned if the buffer is not associated with the CUcontext support
	 * unified virtual addressing. Since host buffers may fall into this
	 * category, this is not treated as an error.
	 */
	case CUDA_ERROR_INVALID_VALUE:
		break;

	/* Returned if cuInit() has not been called. This can happen if support
	 * for CUDA is enabled but the user has not made a CUDA call. This is
	 * not treated as an error.
	 */
	case CUDA_ERROR_NOT_INITIALIZED:
		break;

	/* Returned if the CUcontext does not support unified virtual
	 * addressing.
	 */
	case CUDA_ERROR_INVALID_CONTEXT:
		FI_WARN(&core_prov, FI_LOG_CORE,
			"CUcontext does not support unified virtual addressing\n");
		break;

	default:
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Unhandle cuPointerGetAttribute return code: ret=%d\n",
			cuda_ret);
		break;
	}

	return false;
}

int cuda_host_register(void *ptr, size_t size)
{
	cudaError_t cuda_ret;

	cuda_ret = ofi_cudaHostRegister(ptr, size, cudaHostRegisterDefault);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaHostRegister: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return -FI_EIO;
}

int cuda_host_unregister(void *ptr)
{
	cudaError_t cuda_ret;

	cuda_ret = ofi_cudaHostUnregister(ptr);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform cudaHostUnregister: %s:%s\n",
		ofi_cudaGetErrorName(cuda_ret),
		ofi_cudaGetErrorString(cuda_ret));

	return -FI_EIO;
}

enum cuda_xfer_setting cuda_get_xfer_setting(void)
{
	return cuda_attr.xfer_setting;
}

bool cuda_is_ipc_enabled(void)
{
	return cuda_attr.use_ipc;
}

int cuda_get_ipc_handle_size(size_t *size)
{
	*size = sizeof(cudaIpcMemHandle_t);
	return FI_SUCCESS;
}

bool cuda_is_gdrcopy_enabled(void)
{
	return cuda_attr.use_gdrcopy;
}

#else

int cuda_copy_to_dev(uint64_t device, void *dev, const void *host, size_t size)
{
	return -FI_ENOSYS;
}

int cuda_copy_from_dev(uint64_t device, void *host, const void *dev, size_t size)
{
	return -FI_ENOSYS;
}

int cuda_hmem_init(void)
{
	return -FI_ENOSYS;
}

int cuda_hmem_cleanup(void)
{
	return -FI_ENOSYS;
}

bool cuda_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	return false;
}

int cuda_host_register(void *ptr, size_t size)
{
	return -FI_ENOSYS;
}

int cuda_host_unregister(void *ptr)
{
	return -FI_ENOSYS;
}

int cuda_dev_register(struct fi_mr_attr *mr_attr, uint64_t *handle)
{
	return FI_SUCCESS;
}

int cuda_dev_unregister(uint64_t handle)
{
	return FI_SUCCESS;
}

int cuda_get_handle(void *dev_buf, size_t size, void **handle)
{
	return -FI_ENOSYS;
}

int cuda_open_handle(void **handle, size_t size, uint64_t device,
		     void **ipc_ptr)
{
	return -FI_ENOSYS;
}

int cuda_close_handle(void *ipc_ptr)
{
	return -FI_ENOSYS;
}

int cuda_get_base_addr(const void *ptr, void **base, size_t *size)
{
	return -FI_ENOSYS;
}

enum cuda_xfer_setting cuda_get_xfer_setting(void)
{
	return CUDA_XFER_DISABLED;
}

bool cuda_is_ipc_enabled(void)
{
	return false;
}

int cuda_get_ipc_handle_size(size_t *size)
{
	return -FI_ENOSYS;
}

bool cuda_is_gdrcopy_enabled(void)
{
	return false;
}

int cuda_set_sync_memops(void *ptr)
{
        return FI_SUCCESS;
}
#endif /* HAVE_CUDA */
