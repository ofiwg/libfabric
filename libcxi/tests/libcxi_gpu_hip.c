/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

#include "libcxi_test_common.h"

#ifdef HAVE_HIP_SUPPORT
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

void *hsa_handle;
void *libhip_handle;
static hipError_t (*hip_malloc)(void **devPtr, size_t size);
static hipError_t (*hip_host_alloc)(void **devPtr, size_t size, uint flags);
static hipError_t (*hip_free)(void *devPtr);
static hipError_t (*hip_host_free)(void *p);
static hipError_t (*hip_memset)(void *devPtr, int value, size_t count);
static hipError_t (*hip_memcpy)(void *dst, const void *src, size_t count,
				enum hipMemcpyKind kind);
static hipError_t (*hip_device_count)(int *count);
static hipError_t (*hip_device_sync)(void);
static hipError_t (*hsa_get_dmabuf)(const void *ptr, size_t size,
				    int *dmabuf_fd, uint64_t *offset);
static hsa_status_t (*hsa_put_dmabuf)(int dmabuf_fd);
static hsa_status_t (*hsa_ptr_info)(const void *ptr,
				    hsa_amd_pointer_info_t *info,
				    void *(*alloc)(size_t),
				    uint32_t *num_agents_accessible,
				    hsa_agent_t **accessible);

static int h_malloc(struct mem_window *win)
{
	hipError_t rc = hip_malloc((void **)&win->buffer, win->length);

	cr_assert_eq(rc, hipSuccess, "malloc() failed %d", rc);
	win->loc = on_device;
	win->is_device = true;

	return 0;
}

static int h_host_alloc(struct mem_window *win)
{
	hipError_t rc = hip_host_alloc((void **)&win->buffer, win->length,
				       hipHostMallocMapped);

	cr_assert_eq(rc, hipSuccess, "hip_host_alloc() failed %d", rc);
	win->loc = on_host;
	win->is_device = true;

	return 0;
}

static int h_free(void *devPtr)
{
	hipError_t rc = hip_free(devPtr);

	cr_assert_eq(rc, hipSuccess, "free() failed %d", rc);

	return 0;
}

static int h_host_free(void *p)
{
	hipError_t rc = hip_host_free(p);

	cr_assert_eq(rc, hipSuccess, "hip_host_free() failed %d", rc);

	return 0;
}

static int h_memset(void *devPtr, int value, size_t count)
{
	hipError_t rc = hip_memset(devPtr, value, count);

	cr_assert_eq(rc, hipSuccess, "memset() failed %d", rc);

	rc = hip_device_sync();
	cr_assert_eq(rc, hipSuccess, "hip_device_sync failed %d", rc);

	return 0;
}

static int h_memcpy(void *dst, const void *src, size_t count,
			  enum gpu_copy_dir dir)
{
	hipError_t rc = hip_memcpy(dst, src, count, (enum hipMemcpyKind)dir);

	cr_assert_eq(rc, hipSuccess, "memcpy() failed %d", rc);

	return 0;
}

int h_put_dmabuf_fd(int dmabuf_fd)
{
	hsa_status_t rc;

	rc = hsa_put_dmabuf(dmabuf_fd);
	cr_assert_eq(rc, HSA_STATUS_SUCCESS, "hsa_put_dmabuf() failed %d", rc);

	return 0;
}

static int h_mem_props(struct mem_window *win, void **base, size_t *size)
{
	hsa_status_t rc;
	hsa_amd_pointer_info_t info = {
		.size = sizeof(info),
	};

	if (!win->use_dmabuf) {
		win->hints.dmabuf_valid = false;
		return 0;
	}

	rc = hsa_ptr_info(win->buffer, &info, NULL, NULL, NULL);
	cr_assert_eq(rc, HSA_STATUS_SUCCESS, "hsa_amd_ptr_info() error %d", rc);
	cr_assert_eq(info.type, HSA_EXT_POINTER_TYPE_HSA,
		      "hsa_amd_ptr_info() not HSA");

	*size = info.sizeInBytes;
	*base = info.agentBaseAddress;

	rc = hsa_get_dmabuf(win->buffer, *size, &win->hints.dmabuf_fd,
			    &win->hints.dmabuf_offset);
	cr_assert_eq(rc, HSA_STATUS_SUCCESS, "hsa_get_dmabuf() failed %d", rc);

	win->hints.dmabuf_valid = true;

	return 0;
}

int hip_lib_init(void)
{
	hipError_t ret;
	int count;

	if (libhip_handle)
		return 0;

	libhip_handle = dlopen("libamdhip64.so",
			       RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
	if (!libhip_handle)
		return -1;

	hsa_handle = dlopen("libhsa-runtime64.so",
			    RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
	if (!hsa_handle) {
		dlclose(libhip_handle);
		return -1;
	}

	hip_malloc = dlsym(libhip_handle, "hipMalloc");
	hip_host_alloc = dlsym(libhip_handle, "hipHostAlloc");
	hip_free = dlsym(libhip_handle, "hipFree");
	hip_host_free = dlsym(libhip_handle, "hipHostFree");
	hip_memset = dlsym(libhip_handle, "hipMemset");
	hip_memcpy = dlsym(libhip_handle, "hipMemcpy");
	hip_device_count = dlsym(libhip_handle, "hipGetDeviceCount");
	hip_device_sync = dlsym(libhip_handle, "hipDeviceSynchronize");
	hsa_get_dmabuf = dlsym(hsa_handle, "hsa_amd_portable_export_dmabuf");
	hsa_put_dmabuf = dlsym(hsa_handle, "hsa_amd_portable_close_dmabuf");
	hsa_ptr_info = dlsym(hsa_handle, "hsa_amd_pointer_info");

	if (!hip_malloc || !hip_free || !hip_memset || !hip_memcpy ||
	    !hip_device_count | !hip_device_sync || !hip_host_alloc ||
	    !hip_host_free || !hsa_get_dmabuf || !hsa_put_dmabuf ||
	    !hsa_ptr_info) {
		printf("dlerror:%s\n", dlerror());
		dlclose(libhip_handle);
		return -1;
	}

	ret = hip_device_count(&count);
	if (ret != hipSuccess) {
		dlclose(hsa_handle);
		dlclose(libhip_handle);
		return -1;
	}

	gpu_malloc = h_malloc;
	gpu_host_alloc = h_host_alloc;
	gpu_free = h_free;
	gpu_host_free = h_host_free;
	gpu_memset = h_memset;
	gpu_memcpy = h_memcpy;
	gpu_props = h_mem_props;
	gpu_close_fd = h_put_dmabuf_fd;

	printf("Found AMD GPU\n");

	return 0;
}

void hip_lib_fini(void)
{
	if (!libhip_handle)
		return;

	gpu_free = NULL;
	gpu_malloc = NULL;
	gpu_host_alloc = NULL;
	gpu_memset = NULL;
	gpu_memcpy = NULL;
	gpu_props = NULL;
	gpu_close_fd = NULL;

	dlclose(hsa_handle);
	hsa_handle = NULL;
	dlclose(libhip_handle);
	libhip_handle = NULL;
}
#endif /* HAVE_HIP_SUPPORT */
