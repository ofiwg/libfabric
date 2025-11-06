/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2023 Hewlett Packard Enterprise Development LP
 */

/* CUDA GPU functions */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

#include "utils_common.h"

/* include CUDA files if supported */
#ifdef HAVE_CUDA_SUPPORT
#include <cuda_runtime_api.h>
#include <driver_types.h>

static void *cuda_lib_handle;

/* Initialize CUDA library functions */
int cuda_lib_init(void)
{
	cuda_lib_handle =
		dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
	if (!cuda_lib_handle) {
		fprintf(stderr, "Unable to open libcudart.so\n");
		return -1;
	}
	g_malloc = dlsym(cuda_lib_handle, "cudaMalloc");
	g_free = dlsym(cuda_lib_handle, "cudaFree");
	g_device_count = dlsym(cuda_lib_handle, "cudaGetDeviceCount");
	g_set_device = dlsym(cuda_lib_handle, "cudaSetDevice");
	g_get_device = dlsym(cuda_lib_handle, "cudaGetDevice");
	g_memset = dlsym(cuda_lib_handle, "cudaMemset");
	g_memcpy = dlsym(cuda_lib_handle, "cudaMemcpy");
	g_memcpy_kind_htod = cudaMemcpyHostToDevice;
	g_mem_properties = NULL;

	if (!g_malloc || !g_free || !g_device_count || !g_set_device ||
	    !g_get_device || !g_memset || !g_memcpy) {
		fprintf(stderr, "dlerror:%s\n", dlerror());
		dlclose(cuda_lib_handle);
		return -1;
	}
	return 0;
}

/* clean up CUDA library */
void cuda_lib_fini(void)
{
	if (!cuda_lib_handle)
		return;

	dlclose(cuda_lib_handle);
	cuda_lib_handle = NULL;

	g_malloc = NULL;
	g_free = NULL;
	g_device_count = NULL;
	g_set_device = NULL;
	g_get_device = NULL;
	g_memset = NULL;
	g_memcpy = NULL;
	g_mem_properties = NULL;

}

#endif /* HAVE_CUDA_SUPPORT */
