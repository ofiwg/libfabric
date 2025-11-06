/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "libcxi_test_common.h"

#ifdef HAVE_CUDA_SUPPORT
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda.h>

static cudaError_t (*cuda_malloc)(void **devPtr, size_t size);
static cudaError_t (*cuda_host_alloc)(void **devPtr, size_t size);
static cudaError_t (*cuda_free)(void *devPtr);
static cudaError_t (*cuda_host_free)(void *p);
static cudaError_t (*cuda_memset)(void *devPtr, int value, size_t count);
static cudaError_t (*cuda_memcpy)(void *dst, const void *src, size_t count,
				  enum cudaMemcpyKind kind);
static cudaError_t (*cuda_device_count)(int *count);
static cudaError_t (*cuda_ptr_set_attr)(const void *value,
					CUpointer_attribute attribute,
					CUdeviceptr ptr);

void *libcudart_handle;
void *libcuda_handle;

static int c_malloc(struct mem_window *win)
{
	int attr_value = 1;
	cudaError_t rc = cuda_malloc((void **)&win->buffer, win->length);
	cr_assert_eq(rc, cudaSuccess, "malloc() failed %d", rc);

	rc = cuda_ptr_set_attr(&attr_value, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
			       (CUdeviceptr)win->buffer);
	cr_assert_eq(rc, cudaSuccess, "cuda_ptr_set_attr() failed %d", rc);

	win->loc = on_device;
	win->is_device = true;

	return 0;
}

/* Note: is_device is false so we will use CPU pages.
 * cudaMallocHost is probably no different than malloc for our purposes
 * but adding to be consistent with Intel and AMD GPUs.
 */
static int c_host_alloc(struct mem_window *win)
{
	cudaError_t rc = cuda_host_alloc((void **)&win->buffer, win->length);

	cr_assert_eq(rc, cudaSuccess, "cuda_host_alloc() failed %d", rc);
	win->loc = on_host;
	win->is_device = false;

	return 0;
}

static int c_free(void *devPtr)
{
	cudaError_t rc = cuda_free(devPtr);

	cr_assert_eq(rc, cudaSuccess, "free() failed %d", rc);

	return 0;
}

static int c_host_free(void *p)
{
	cudaError_t rc = cuda_host_free(p);

	cr_assert_eq(rc, cudaSuccess, "cuda_host_free() failed %d", rc);

	return 0;
}

static int c_memset(void *devPtr, int value, size_t count)
{
	cudaError_t rc = cuda_memset(devPtr, value, count);

	cr_assert_eq(rc, cudaSuccess, "memset() failed %d", rc);

	return 0;
}

static int c_memcpy(void *dst, const void *src, size_t count,
			     enum gpu_copy_dir dir)
{
	cudaError_t rc = cuda_memcpy(dst, src, count,
					(enum cudaMemcpyKind)dir);

	cr_assert_eq(rc, cudaSuccess, "memcpy() failed %d", rc);

	return 0;
}

static int c_gpu_props(struct mem_window *win, void **base, size_t *size)
{
	win->hints.dmabuf_valid = false;

	return 0;
}

static int c_gpu_close_fd(int fd)
{
	return 0;
}

int cuda_lib_init(void)
{
	cudaError_t ret;
	int count;

	if (libcudart_handle)
		return 0;

	libcudart_handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
	if (!libcudart_handle)
		return -1;

	libcuda_handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
	if (!libcuda_handle) {
		dlclose(libcudart_handle);
		return -1;
	}

	cuda_malloc = dlsym(libcudart_handle, "cudaMalloc");
	cuda_host_alloc = dlsym(libcudart_handle, "cudaMallocHost");
	cuda_free = dlsym(libcudart_handle, "cudaFree");
	cuda_host_free = dlsym(libcuda_handle, "cuMemFreeHost");
	cuda_memset = dlsym(libcudart_handle, "cudaMemset");
	cuda_memcpy = dlsym(libcudart_handle, "cudaMemcpy");
	cuda_device_count = dlsym(libcudart_handle, "cudaGetDeviceCount");
	cuda_ptr_set_attr = dlsym(libcuda_handle, "cuPointerSetAttribute");

	if (!cuda_malloc || !cuda_free || !cuda_memset || !cuda_memcpy ||
	    !cuda_device_count || !cuda_ptr_set_attr | !cuda_host_alloc ||
	    !cuda_host_free) {
		printf("dlerror:%s\n", dlerror());
		dlclose(libcuda_handle);
		dlclose(libcudart_handle);
		return -1;
	}

	ret = cuda_device_count(&count);
	if (ret != cudaSuccess) {
		dlclose(libcuda_handle);
		dlclose(libcudart_handle);
		return -1;
	}

	gpu_malloc = c_malloc;
	gpu_host_alloc = c_host_alloc;
	gpu_free = c_free;
	gpu_host_free = c_host_free;
	gpu_memset = c_memset;
	gpu_memcpy = c_memcpy;
	gpu_props = c_gpu_props;
	gpu_close_fd = c_gpu_close_fd;

	printf("Found NVIDIA GPU\n");

	return 0;
}

void cuda_lib_fini(void)
{
	if (!libcuda_handle)
		return;

	gpu_malloc = NULL;
	gpu_host_alloc = NULL;
	gpu_free = NULL;
	gpu_host_free = NULL;
	gpu_memset = NULL;
	gpu_memcpy = NULL;
	gpu_props = NULL;
	gpu_close_fd = NULL;
	dlclose(libcuda_handle);
	dlclose(libcudart_handle);

	libcudart_handle = NULL;
}
#endif /* HAVE_CUDA_SUPPORT */
