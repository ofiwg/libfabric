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

#include "hmem.h"
#include "shared.h"

#if HAVE_CUDA_RUNTIME_H

#include <dlfcn.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

struct cuda_ops {
	cudaError_t (*cudaMemcpy)(void *dst, const void *src, size_t count,
				  enum cudaMemcpyKind kind);
	cudaError_t (*cudaMalloc)(void **ptr, size_t size);
	cudaError_t (*cudaMallocHost)(void **ptr, size_t size);
	cudaError_t (*cudaFree)(void *ptr);
	cudaError_t (*cudaFreeHost)(void *ptr);
	cudaError_t (*cudaMemset)(void *ptr, int value, size_t count);
	const char *(*cudaGetErrorName)(cudaError_t error);
	const char *(*cudaGetErrorString)(cudaError_t error);
	cudaError_t (*cudaSetDevice)(int device);
	CUresult (*cuPointerSetAttribute)(void *data,
					  CUpointer_attribute attribute,
					  CUdeviceptr ptr);
	CUresult (*cuGetErrorName)(CUresult error, const char** pStr);
	CUresult (*cuGetErrorString)(CUresult error, const char** pStr);
#if HAVE_CUDA_DMABUF
	CUresult (*cuMemGetHandleForAddressRange)(void* handle,
						  CUdeviceptr dptr, size_t size,
						  CUmemRangeHandleType handleType,
						  unsigned long long flags);
#endif /* HAVE_CUDA_DMABUF */
	CUresult (*cuDeviceGetAttribute)(int* pi,
					 CUdevice_attribute attrib, CUdevice dev);
	CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
	CUresult (*cuDeviceGetName)(char* name, int len, CUdevice dev);
	CUresult (*cuDriverGetVersion)(int* driverVersion);
	CUresult (*cuMemGetAddressRange)( CUdeviceptr* pbase,
					  size_t* psize, CUdeviceptr dptr);
};

static struct cuda_ops cuda_ops;
static void *cudart_handle;
static void *cuda_handle;
static bool dmabuf_supported;
static bool gdr_supported;
static enum ft_cuda_memory_support cuda_memory_support = FT_CUDA_NOT_INITIALIZED;
static const char* get_cuda_memory_support_str(enum ft_cuda_memory_support support) {
    switch (support) {
        case FT_CUDA_NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case FT_CUDA_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case FT_CUDA_DMA_BUF_ONLY:
            return "DMA_BUF_ONLY";
        case FT_CUDA_GDR_ONLY:
            return "GDR_ONLY";
        case FT_CUDA_DMABUF_GDR_BOTH:
            return "DMABUF_GDR_BOTH";
        default:
            return "INVALID";
    }
}

/**
 * Since function names can get redefined in cuda.h/cuda_runtime.h files,
 * we need to do this stringifying to get the latest function name from
 * the header files.  For example, cuda.h may have something like this:
 * #define cuMemFree cuMemFree_v2
 * We want to make sure we find cuMemFree_v2, not cuMemFree.
 */
#define STRINGIFY2(x) #x
#define STRINGIFY(x)  STRINGIFY2(x)

#define CUDA_ERR(err, fmt, ...) \
	FT_ERR(fmt ": %s %s", ##__VA_ARGS__, cuda_ops.cudaGetErrorName(err), \
	       cuda_ops.cudaGetErrorString(err))

static void ft_cuda_driver_api_print_error(CUresult cu_result, char *cuda_api_name)
{
	const char *cu_error_name;
	const char *cu_error_str;
	cuda_ops.cuGetErrorName(cu_result, &cu_error_name);
	cuda_ops.cuGetErrorString(cu_result, &cu_error_str);
	FT_ERR("%s failed: %s:%s\n",
		   cuda_api_name, cu_error_name, cu_error_str);
}

static int ft_cuda_pointer_set_attribute(void *buf)
{
	int true_flag = 1;
	CUresult cu_result;
	cu_result = cuda_ops.cuPointerSetAttribute((void *) &true_flag,
						  CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
						  (CUdeviceptr) buf);
	if (cu_result != CUDA_SUCCESS) {
	    ft_cuda_driver_api_print_error(cu_result, "cuPointerSetAttribute");
		return -FI_EIO;
	}
	return FI_SUCCESS;
}

/**
 * @brief Detect CUDA memory transport support (dma-buf vs P2P)
 *
 * This routine queries device 0 for:
 *   - CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED
 *   - CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
 *   - CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
 *
 * Logic is derived from CUDA 13.0 release notes and Blackwell compatibility guide:
 *   - https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
 *   - https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
 *   - https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support/
 *
 * NVIDIA deprecated GPUDirect RDMA (nv-p2p APIs) starting with Blackwell
 * (compute capability >= 10). Applications must migrate to dma-buf.
 *
 * Truth table for effective support (device memory only):
 *
 *  DMA_BUF   GPU_DIRECT_RDMA   Result
 *  -------   ----------------  ------------------------
 *     0            0           NOT_SUPPORTED
 *     0            1           GDR_ONLY
 *     1            0           DMA_BUF_ONLY
 *     1            1           DMABUF_GDR_BOTH
 *
 * Note:
 *  - CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED is orthogonal and
 *    indicates whether cudaHostAlloc() memory can be exported as dma-buf.
 *  - On compute capability >= 10 (Blackwell), we force GPU_DIRECT_RDMA=0
 *    regardless of attribute value to align with the deprecation notice.
 *
 * @return FI_SUCCESS on success
 *        -FI_EIO on CUDA API error
 *        Sets global flags dmabuf_supported, gdr_supported, and cuda_memory_support.
 */
static int ft_cuda_detect_memory_support(void)
{
#if HAVE_CUDA_DMABUF
    CUresult cuda_ret;
    CUdevice dev;
    int cc_major = 0, cc_minor = 0;
    int dma_buf_attr = 0;
    int gdr_attr = 0;
    cuda_memory_support = FT_CUDA_NOT_INITIALIZED;
    

    FT_INFO("ft_cuda_detect_memory_support() called");

    cuda_ret = cuda_ops.cuDeviceGet(&dev, 0);
    if (cuda_ret != CUDA_SUCCESS) {
        ft_cuda_driver_api_print_error(cuda_ret, "cuDeviceGet");
        cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
        return -FI_EIO;
    }

    cuda_ret = cuda_ops.cuDeviceGetAttribute(&cc_major,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (cuda_ret != CUDA_SUCCESS) {
        ft_cuda_driver_api_print_error(cuda_ret, "cuDeviceGetAttribute(CC_MAJOR)");
        cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
        return -FI_EIO;
    }

    cuda_ret = cuda_ops.cuDeviceGetAttribute(&cc_minor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (cuda_ret != CUDA_SUCCESS) {
        ft_cuda_driver_api_print_error(cuda_ret, "cuDeviceGetAttribute(CC_MINOR)");
        cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
        return -FI_EIO;
    }

    cuda_ret = cuda_ops.cuDeviceGetAttribute(&dma_buf_attr,
        CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    if (cuda_ret != CUDA_SUCCESS) {
        ft_cuda_driver_api_print_error(cuda_ret, "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)");
        cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
        return -FI_EIO;
    }

    cuda_ret = cuda_ops.cuDeviceGetAttribute(&gdr_attr,
        CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, dev);
    if (cuda_ret != CUDA_SUCCESS) {
        ft_cuda_driver_api_print_error(cuda_ret, "cuDeviceGetAttribute(GPU_DIRECT_RDMA_SUPPORTED)");
        cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
        return -FI_EIO;
    }

    dmabuf_supported = (dma_buf_attr == 1);

    if (cc_major >= 10) {
        // Blackwell or newer: nv-p2p deprecated
        FT_INFO("Compute capability %d.%d: forcing gdr_supported=false due to Blackwell deprecation", cc_major, cc_minor);
        gdr_supported = false;
    } else {
        gdr_supported = (gdr_attr == 1);
    }

    FT_INFO("Compute capability %d.%d", cc_major, cc_minor);
    FT_INFO("dmabuf_supported=%s", dmabuf_supported ? "true" : "false");
    FT_INFO("GPU_DIRECT_RDMA_SUPPORTED raw=%d -> gdr_supported=%s",
            gdr_attr, gdr_supported ? "true" : "false");

    // Final truth table
	if (!gdr_supported && !dmabuf_supported)
		cuda_memory_support = FT_CUDA_NOT_SUPPORTED;
	else if (gdr_supported && dmabuf_supported)
		cuda_memory_support = FT_CUDA_DMABUF_GDR_BOTH;
	else if (dmabuf_supported)
		cuda_memory_support = FT_CUDA_DMA_BUF_ONLY;
	else
		cuda_memory_support = FT_CUDA_GDR_ONLY;

    FT_INFO("cuda_memory_support=%s", get_cuda_memory_support_str(cuda_memory_support));
    return FI_SUCCESS;

#else
    FT_INFO("HAVE_CUDA_DMABUF not enabled, returning FT_CUDA_NOT_INITIALIZED");
    cuda_memory_support = FT_CUDA_NOT_INITIALIZED;
    return FI_SUCCESS;
#endif
}


int ft_cuda_init(void)
{
	cudaError_t cuda_ret;
	int ret;
    cuda_memory_support = FT_CUDA_NOT_INITIALIZED;

	cudart_handle = dlopen("libcudart.so", RTLD_NOW);
	if (!cudart_handle) {
		FT_ERR("Failed to dlopen libcudart.so");
		goto err;
	}

	cuda_handle = dlopen("libcuda.so.1", RTLD_NOW);
	if (!cuda_handle) {
		FT_ERR("Failed to dlopen libcuda.so.1\n");
		goto err_dlclose_cudart;
	}

	cuda_ops.cudaMemcpy = dlsym(cudart_handle, STRINGIFY(cudaMemcpy));
	if (!cuda_ops.cudaMemcpy) {
		FT_ERR("Failed to find cudaMemcpy");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaMalloc = dlsym(cudart_handle, STRINGIFY(cudaMalloc));
	if (!cuda_ops.cudaMalloc) {
		FT_ERR("Failed to find cudaMalloc");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaMallocHost = dlsym(cudart_handle, STRINGIFY(cudaMallocHost));
	if (!cuda_ops.cudaMallocHost) {
		FT_ERR("Failed to find cudaMallocHost");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaFree = dlsym(cudart_handle, STRINGIFY(cudaFree));
	if (!cuda_ops.cudaFree) {
		FT_ERR("Failed to find cudaFree");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaFreeHost = dlsym(cudart_handle, STRINGIFY(cudaFreeHost));
	if (!cuda_ops.cudaFreeHost) {
		FT_ERR("Failed to find cudaFreeHost");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaMemset = dlsym(cudart_handle, STRINGIFY(cudaMemset));
	if (!cuda_ops.cudaMemset) {
		FT_ERR("Failed to find cudaMemset");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetErrorName = dlsym(cudart_handle, STRINGIFY(cudaGetErrorName));
	if (!cuda_ops.cudaGetErrorName) {
		FT_ERR("Failed to find cudaGetErrorName");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaGetErrorString = dlsym(cudart_handle,
					    STRINGIFY(cudaGetErrorString));
	if (!cuda_ops.cudaGetErrorString) {
		FT_ERR("Failed to find cudaGetErrorString");
		goto err_dlclose_cuda;
	}

	cuda_ops.cudaSetDevice = dlsym(cudart_handle, STRINGIFY(cudaSetDevice));
	if (!cuda_ops.cudaSetDevice) {
		FT_ERR("Failed to find cudaSetDevice");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuPointerSetAttribute = dlsym(cuda_handle,
					       STRINGIFY(cuPointerSetAttribute));
	if (!cuda_ops.cuPointerSetAttribute) {
		FT_ERR("Failed to find cuPointerSetAttribute\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuGetErrorName = dlsym(cuda_handle,
					       STRINGIFY(cuGetErrorName));
	if (!cuda_ops.cuGetErrorName) {
		FT_ERR("Failed to find cuGetErrorName\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuGetErrorString = dlsym(cuda_handle,
					       STRINGIFY(cuGetErrorString));
	if (!cuda_ops.cuGetErrorString) {
		FT_ERR("Failed to find cuGetErrorString\n");
		goto err_dlclose_cuda;
	}

#if HAVE_CUDA_DMABUF
	cuda_ops.cuMemGetHandleForAddressRange = dlsym(cuda_handle,
						       STRINGIFY(cuMemGetHandleForAddressRange));
	if (!cuda_ops.cuMemGetHandleForAddressRange) {
		FT_ERR("Failed to find cuMemGetHandleForAddressRange\n");
		goto err_dlclose_cuda;
	}
#endif
	cuda_ops.cuDeviceGetAttribute = dlsym(cuda_handle,
					      STRINGIFY(cuDeviceGetAttribute));
	if (!cuda_ops.cuDeviceGetAttribute) {
		FT_ERR("Failed to find cuPointerSetAttribute\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuDeviceGet = dlsym(cuda_handle,
				     STRINGIFY(cuDeviceGet));
	if (!cuda_ops.cuDeviceGet) {
		FT_ERR("Failed to find cuDeviceGet\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuDeviceGetName = dlsym(cuda_handle,
					 STRINGIFY(cuDeviceGetName));
	if (!cuda_ops.cuDeviceGetName) {
		FT_ERR("Failed to find cuDeviceGetName\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuDriverGetVersion = dlsym(cuda_handle,
					    STRINGIFY(cuDriverGetVersion));
	if (!cuda_ops.cuDriverGetVersion) {
		FT_ERR("Failed to find cuDriverGetVersion\n");
		goto err_dlclose_cuda;
	}

	cuda_ops.cuMemGetAddressRange = dlsym(cuda_handle,
				     STRINGIFY(cuMemGetAddressRange));
	if (!cuda_ops.cuMemGetAddressRange) {
		FT_ERR("Failed to find cuMemGetAddressRange\n");
		goto err_dlclose_cuda;
	}

	cuda_ret = cuda_ops.cudaSetDevice(opts.device);
	if (cuda_ret != cudaSuccess) {
		CUDA_ERR(cuda_ret, "cudaSetDevice failed");
		goto err_dlclose_cuda;
	}

    ret = ft_cuda_detect_memory_support();
    if (ret != FI_SUCCESS) {
		goto err_dlclose_cuda;
	}
    

	return FI_SUCCESS;

err_dlclose_cuda:
	dlclose(cuda_handle);
err_dlclose_cudart:
	dlclose(cudart_handle);
err:
	return -FI_ENODATA;
}

int ft_cuda_cleanup(void)
{
	dlclose(cudart_handle);
	return FI_SUCCESS;
}

int ft_cuda_alloc(uint64_t device, void **buf, size_t size)
{
	cudaError_t cuda_ret;
	int ret;

	cuda_ret = cuda_ops.cudaMalloc(buf, size);

	if (cuda_ret != cudaSuccess) {
	        CUDA_ERR(cuda_ret, "cudaMalloc failed");
		return -FI_ENOMEM;
	}

	ret = ft_cuda_pointer_set_attribute(*buf);

	if (ret != FI_SUCCESS) {
		ft_cuda_free(*buf);
		*buf = NULL;
		return -FI_EIO;
	}

	return FI_SUCCESS;
}

int ft_cuda_alloc_host(void **buf, size_t size)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaMallocHost(buf, size);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaMallocHost failed");

	return -FI_ENOMEM;
}

int ft_cuda_free(void *buf)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaFree(buf);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaFree failed");

	return -FI_EIO;
}

int ft_cuda_free_host(void *buf)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaFreeHost(buf);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaFreeHost failed");

	return -FI_EIO;
}

int ft_cuda_memset(uint64_t device, void *buf, int value, size_t size)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaMemset(buf, value, size);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaMemset failed");

	return -FI_EIO;
}

int ft_cuda_copy_to_hmem(uint64_t device, void *dst, const void *src,
			 size_t size)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaMemcpy failed");

	return -FI_EIO;
}

int ft_cuda_copy_from_hmem(uint64_t device, void *dst, const void *src,
			   size_t size)
{
	cudaError_t cuda_ret;

	cuda_ret = cuda_ops.cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
	if (cuda_ret == cudaSuccess)
		return FI_SUCCESS;

	CUDA_ERR(cuda_ret, "cudaMemcpy failed");

	return -FI_EIO;
}

/* TODO: Make get_base_addr a general hmem ops API */
static
int ft_cuda_get_base_addr(const void *ptr, size_t len, void **base, size_t *size)
{
	CUresult cu_result;

	cu_result = cuda_ops.cuMemGetAddressRange(
				(CUdeviceptr *)base,
				size, (CUdeviceptr) ptr);
	if (cu_result == CUDA_SUCCESS)
		return FI_SUCCESS;

	ft_cuda_driver_api_print_error(cu_result, "cuMemGetAddressRange");
	return -FI_EIO;
}

/**
 * @brief Get dmabuf fd and offset for a given cuda memory allocation
 *
 * @param device cuda device index
 * @param buf the starting address of the cuda memory allocation
 * @param len the length of the cuda memory allocation
 * @param dmabuf_fd the fd of the dmabuf region
 * @param dmabuf_offset the offset of the buf in the dmabuf region
 * @return  FI_SUCCESS if dmabuf fd and offset are retrieved successfully
 *         -FI_EIO upon CUDA API error, -FI_EOPNOTSUPP upon dmabuf not supported
 */
int ft_cuda_get_dmabuf_fd(void *buf, size_t len,
			  int *dmabuf_fd, uint64_t *dmabuf_offset)
{
#if HAVE_CUDA_DMABUF
	CUdeviceptr aligned_ptr;
	CUresult cuda_ret;
	int ret;

	size_t aligned_size;
	size_t host_page_size = sysconf(_SC_PAGESIZE);
	void *base_addr;
	size_t total_size;
	unsigned long long flags;

	if (!dmabuf_supported) {
		FT_LOG("warn", "dmabuf is not supported\n");
		return -FI_EOPNOTSUPP;
	}

	ret = ft_cuda_get_base_addr(buf, len, &base_addr, &total_size);
	if (ret)
		return ret;

	aligned_ptr = (uintptr_t) ft_get_page_start(base_addr, host_page_size);
	aligned_size = (uintptr_t) ft_get_page_end((void *) ((uintptr_t) base_addr + total_size - 1),
						    host_page_size) - (uintptr_t) aligned_ptr + 1;

	flags = 0;
#if HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE
	flags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
# endif /* HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE */
	cuda_ret = cuda_ops.cuMemGetHandleForAddressRange(
						(void *)dmabuf_fd,
						aligned_ptr, aligned_size,
						CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
						flags);

	if ((cuda_ret == CUDA_ERROR_INVALID_VALUE || cuda_ret == CUDA_ERROR_NOT_SUPPORTED) && flags != 0) {
		FT_WARN("cuMemGetHandleForAddressRange failed with flags: %llu, "
		       "invalid argument. Retrying with no flags.\n", flags);
		cuda_ret = cuda_ops.cuMemGetHandleForAddressRange(
						(void *) dmabuf_fd,
						aligned_ptr, aligned_size,
						CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
						0);
	}

	if (cuda_ret != CUDA_SUCCESS) {
		ft_cuda_driver_api_print_error(cuda_ret,
				"cuMemGetHandleForAddressRange");
		return -FI_EIO;
	}

	*dmabuf_offset = (uintptr_t) buf - (uintptr_t) aligned_ptr;

	return FI_SUCCESS;
#else
	return -FI_EOPNOTSUPP;
#endif /* HAVE_CUDA_DMABUF */
}

int ft_cuda_put_dmabuf_fd(int fd)
{
#if HAVE_CUDA_DMABUF
	(void)close(fd);
	return FI_SUCCESS;
#else
	return -FI_EOPNOTSUPP;
#endif /* HAVE_CUDA_DMABUF */
}

enum ft_cuda_memory_support ft_cuda_memory_support(void)
{
    if (cuda_memory_support == FT_CUDA_NOT_INITIALIZED) {
        FT_INFO("ft_cuda_memory_support() not called yet!");
    }
    return cuda_memory_support;
}

#else

int ft_cuda_init(void)
{
	return -FI_ENOSYS;
}

int ft_cuda_cleanup(void)
{
	return -FI_ENOSYS;
}

int ft_cuda_alloc(uint64_t device, void **buf, size_t size)
{
	return -FI_ENOSYS;
}

int ft_cuda_alloc_host(void **buf, size_t size)
{
	return -FI_ENOSYS;
}

int ft_cuda_free(void *buf)
{
	return -FI_ENOSYS;
}

int ft_cuda_free_host(void *buf)
{
	return -FI_ENOSYS;
}

int ft_cuda_memset(uint64_t device, void *buf, int value, size_t size)
{
	return -FI_ENOSYS;
}

int ft_cuda_copy_to_hmem(uint64_t device, void *dst, const void *src,
			 size_t size)
{
	return -FI_ENOSYS;
}

int ft_cuda_copy_from_hmem(uint64_t device, void *dst, const void *src,
			   size_t size)
{
	return -FI_ENOSYS;
}

int ft_cuda_get_dmabuf_fd(void *buf, size_t len,
			  int *dmabuf_fd, uint64_t *dmabuf_offset)
{
	return -FI_ENOSYS;
}

int ft_cuda_put_dmabuf_fd(int fd)
{
	return -FI_ENOSYS;
}

enum ft_cuda_memory_support ft_cuda_memory_support(void)
{
	return FT_CUDA_NOT_SUPPORTED;
}
#endif /* HAVE_CUDA_RUNTIME_H */
