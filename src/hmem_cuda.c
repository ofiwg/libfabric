/*
 * (C) Copyright 2020 Hewlett Packard Enterprise Development LP
 * (C) Copyright 2021 Amazon.com, Inc. or its affiliates.
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

#if HAVE_LIBCUDA

#include <cuda.h>
#include <cuda_runtime.h>

struct cuda_ops {
	cudaError_t (*cudaMemcpy)(void *dst, const void *src, size_t size,
				  enum cudaMemcpyKind kind);
	cudaError_t (*cudaFree)(void* ptr);
	cudaError_t (*cudaMalloc)(void** ptr, size_t size);
	const char *(*cudaGetErrorName)(cudaError_t error);
	const char *(*cudaGetErrorString)(cudaError_t error);
	CUresult (*cuPointerGetAttribute)(void *data,
					  CUpointer_attribute attribute,
					  CUdeviceptr ptr);
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
};

static bool hmem_cuda_use_gdrcopy;
static bool cuda_ipc_enabled;

static cudaError_t cuda_disabled_cudaMemcpy(void *dst, const void *src,
					    size_t size, enum cudaMemcpyKind kind);

#ifdef ENABLE_CUDA_DLOPEN

#include <dlfcn.h>

static void *cudart_handle;
static void *cuda_handle;
static struct cuda_ops cuda_ops;

#else

static struct cuda_ops cuda_ops = {
	.cudaMemcpy = cudaMemcpy,
	.cudaFree = cudaFree,
	.cudaMalloc = cudaMalloc,
	.cudaGetErrorName = cudaGetErrorName,
	.cudaGetErrorString = cudaGetErrorString,
	.cuPointerGetAttribute = cuPointerGetAttribute,
	.cudaHostRegister = cudaHostRegister,
	.cudaHostUnregister = cudaHostUnregister,
	.cudaGetDeviceCount = cudaGetDeviceCount,
	.cudaGetDevice = cudaGetDevice,
	.cudaSetDevice = cudaSetDevice,
	.cudaIpcOpenMemHandle = cudaIpcOpenMemHandle,
	.cudaIpcGetMemHandle = cudaIpcGetMemHandle,
	.cudaIpcCloseMemHandle = cudaIpcCloseMemHandle
};

#endif /* ENABLE_CUDA_DLOPEN */

cudaError_t ofi_cudaMemcpy(void *dst, const void *src, size_t size,
			   enum cudaMemcpyKind kind)
{
	return cuda_ops.cudaMemcpy(dst, src, size, kind);
}

const char *ofi_cudaGetErrorName(cudaError_t error)
{
	return cuda_ops.cudaGetErrorName(error);
}

const char *ofi_cudaGetErrorString(cudaError_t error)
{
	return cuda_ops.cudaGetErrorString(error);
}

CUresult ofi_cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
				   CUdeviceptr ptr)
{
	return cuda_ops.cuPointerGetAttribute(data, attribute, ptr);
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

int cuda_copy_to_dev(uint64_t device, void *dst, const void *src, size_t size)
{
	if (hmem_cuda_use_gdrcopy) {
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
	if (hmem_cuda_use_gdrcopy) {
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
	if (hmem_cuda_use_gdrcopy)
		return cuda_gdrcopy_dev_register(mr_attr, handle);

	*handle = mr_attr->device.cuda;
	return FI_SUCCESS;
}

int cuda_dev_unregister(uint64_t handle)
{
	if (hmem_cuda_use_gdrcopy)
		return cuda_gdrcopy_dev_unregister(handle);

	return FI_SUCCESS;
}

int cuda_get_handle(void *dev_buf, void **handle)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaIpcGetMemHandle((cudaIpcMemHandle_t *)handle,
						dev_buf);

	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform cudaIpcGetMemHandle: %s:%s\n",
			ofi_cudaGetErrorName(cuda_ret),
			ofi_cudaGetErrorString(cuda_ret));

	return -FI_EINVAL;
}

int cuda_open_handle(void **handle, uint64_t device, void **ipc_ptr)
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

	return -FI_EINVAL;
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

static cudaError_t cuda_disabled_cudaMemcpy(void *dst, const void *src,
					    size_t size, enum cudaMemcpyKind kind)
{
	FI_WARN(&core_prov, FI_LOG_CORE,
		"cudaMemcpy was called but FI_HMEM_CUDA_ENABLE_XFER = 0, "
		"no copy will occur to prevent deadlock.");

	return cudaErrorInvalidValue;
}

static int cuda_hmem_dl_init(void)
{
#ifdef ENABLE_CUDA_DLOPEN
	/* Assume failure to dlopen CUDA runtime is caused by the library not
	 * being found. Thus, CUDA is not supported.
	 */
	cudart_handle = dlopen("libcudart.so", RTLD_NOW);
	if (!cudart_handle) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libcudart.so\n");
		return -FI_ENOSYS;
	}

	cuda_handle = dlopen("libcuda.so", RTLD_NOW);
	if (!cuda_handle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libcuda.so\n");
		goto err_dlclose_cudart;
	}

	cuda_ops.cudaMemcpy = dlsym(cudart_handle, "cudaMemcpy");
	if (!cuda_ops.cudaMemcpy) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find cudaMemcpy\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaFree = dlsym(cudart_handle, "cudaFree");
	if (!cuda_ops.cudaFree) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find cudaFree\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaMalloc = dlsym(cudart_handle, "cudaMalloc");
	if (!cuda_ops.cudaMalloc) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find cudaMalloc\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetErrorName = dlsym(cudart_handle, "cudaGetErrorName");
	if (!cuda_ops.cudaGetErrorName) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaGetErrorName\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetErrorString = dlsym(cudart_handle,
					    "cudaGetErrorString");
	if (!cuda_ops.cudaGetErrorString) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaGetErrorString\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuPointerGetAttribute = dlsym(cuda_handle,
					       "cuPointerGetAttribute");
	if (!cuda_ops.cuPointerGetAttribute) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cuPointerGetAttribute\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaHostRegister = dlsym(cudart_handle, "cudaHostRegister");
	if (!cuda_ops.cudaHostRegister) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaHostRegister\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaHostUnregister = dlsym(cudart_handle,
					    "cudaHostUnregister");
	if (!cuda_ops.cudaHostUnregister) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaHostUnregister\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetDeviceCount = dlsym(cudart_handle,
					    "cudaGetDeviceCount");
	if (!cuda_ops.cudaGetDeviceCount) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaGetDeviceCount\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetDevice = dlsym(cudart_handle,
					    "cudaGetDevice");
	if (!cuda_ops.cudaGetDevice) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaGetDevice\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaSetDevice = dlsym(cudart_handle,
					    "cudaSetDevice");
	if (!cuda_ops.cudaSetDevice) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaSetDevice\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaIpcOpenMemHandle = dlsym(cudart_handle,
					    "cudaIpcOpenMemHandle");
	if (!cuda_ops.cudaIpcOpenMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaIpcOpenMemHandle\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaIpcGetMemHandle = dlsym(cudart_handle,
					    "cudaIpcGetMemHandle");
	if (!cuda_ops.cudaIpcGetMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaIpcGetMemHandle\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaIpcCloseMemHandle = dlsym(cudart_handle,
					    "cudaIpcCloseMemHandle");
	if (!cuda_ops.cudaIpcCloseMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find cudaIpcCloseMemHandle\n");
		goto err_dlclose_cuda;
	}

	return FI_SUCCESS;

err_dlclose_cuda:
	dlclose(cuda_handle);
err_dlclose_cudart:
	dlclose(cudart_handle);

	return -FI_ENODATA;
#else
	return FI_SUCCESS;
#endif /* ENABLE_CUDA_DLOPEN */
}

static void cuda_hmem_dl_cleanup(void)
{
#ifdef ENABLE_CUDA_DLOPEN
	dlclose(cuda_handle);
	dlclose(cudart_handle);
#endif
}

static int cuda_hmem_verify_devices(void)
{
	int device_count;
	cudaError_t cuda_ret;

	/* Verify CUDA compute-capable devices are present on the host. */
	cuda_ret = ofi_cudaGetDeviceCount(&device_count);
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

	if (device_count == 0)
		return -FI_ENOSYS;

	return FI_SUCCESS;
}

int cuda_hmem_init(void)
{
	int ret;
	int gdrcopy_ret;
	bool cuda_enable_xfer;

	fi_param_define(NULL, "hmem_cuda_use_gdrcopy", FI_PARAM_BOOL,
			"Use gdrcopy to copy data to/from CUDA GPU memory. "
			"If libfabric is not compiled with gdrcopy support, "
			"this variable is not checked. (default: true)");
	fi_param_define(NULL, "hmem_cuda_enable_xfer", FI_PARAM_BOOL,
			"Enable use of CUDA APIs for copying data to/from CUDA "
			"GPU memory. This should be disabled if CUDA "
			"operations on the default stream would result in a "
			"deadlock in the application. (default: true)");

	ret = cuda_hmem_dl_init();
	if (ret != FI_SUCCESS)
		return ret;

	ret = cuda_hmem_verify_devices();
	if (ret != FI_SUCCESS)
		goto dl_cleanup;

	fi_param_get_bool(NULL, "hmem_cuda_use_gdrcopy",
			  &ret);
	hmem_cuda_use_gdrcopy = (ret != 0);
	if (hmem_cuda_use_gdrcopy) {
		gdrcopy_ret = cuda_gdrcopy_hmem_init();
		if (gdrcopy_ret != FI_SUCCESS) {
			hmem_cuda_use_gdrcopy = false;
			if (gdrcopy_ret != -FI_ENOSYS)
				FI_WARN(&core_prov, FI_LOG_CORE,
					"gdrcopy initialization failed! "
					"gdrcopy will not be used.\n");
		}
	}

	ret = 1;
	fi_param_get_bool(NULL, "hmem_cuda_enable_xfer", &ret);
	cuda_enable_xfer = (ret != 0);

	if (!cuda_enable_xfer)
		cuda_ops.cudaMemcpy = cuda_disabled_cudaMemcpy;

	/*
	 * CUDA IPC is only enabled if gdrcopy is not in use and
	 * cudaMemcpy can be used.
	 */
	cuda_ipc_enabled = !hmem_cuda_use_gdrcopy && cuda_enable_xfer;

	return FI_SUCCESS;

dl_cleanup:
	cuda_hmem_dl_cleanup();

	return ret;
}

int cuda_hmem_cleanup(void)
{
	cuda_hmem_dl_cleanup();
	if (hmem_cuda_use_gdrcopy)
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
			"CUcontext does not support unified virtual addressining\n");
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

bool cuda_is_ipc_enabled(void)
{
	return !ofi_hmem_p2p_disabled() && cuda_ipc_enabled;
}

bool cuda_is_gdrcopy_enabled(void)
{
	return hmem_cuda_use_gdrcopy;
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

int cuda_get_handle(void *dev_buf, void **handle)
{
	return -FI_ENOSYS;
}

int cuda_open_handle(void **handle, uint64_t device, void **ipc_ptr)
{
	return -FI_ENOSYS;
}

int cuda_close_handle(void *ipc_ptr)
{
	return -FI_ENOSYS;
}

bool cuda_is_ipc_enabled(void)
{
	return false;
}

bool cuda_is_gdrcopy_enabled(void)
{
	return false;
}

#endif /* HAVE_LIBCUDA */
