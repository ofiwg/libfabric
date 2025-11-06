/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2023 Hewlett Packard Enterprise Development LP
 */

/* HIP GPU functions */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

#include "utils_common.h"

/* include HIP files if supported */
#ifdef HAVE_HIP_SUPPORT
#include <hip/hip_runtime_api.h>
#include <hip/driver_types.h>

static void *hip_lib_handle;

/* Initialize HIP library functions */
int hip_lib_init(void)
{
	hip_lib_handle = dlopen("libamdhip64.so",
				RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
	if (!hip_lib_handle) {
		fprintf(stderr, "Unable to open libamdhip64.so\n");
		return -1;
	}
	g_malloc = dlsym(hip_lib_handle, "hipMalloc");
	g_free = dlsym(hip_lib_handle, "hipFree");
	g_device_count = dlsym(hip_lib_handle, "hipGetDeviceCount");
	g_set_device = dlsym(hip_lib_handle, "hipSetDevice");
	g_get_device = dlsym(hip_lib_handle, "hipGetDevice");
	g_memset = dlsym(hip_lib_handle, "hipMemset");
	g_memcpy = dlsym(hip_lib_handle, "hipMemcpy");
	g_memcpy_kind_htod = hipMemcpyHostToDevice;
	g_mem_properties = NULL;

	if (!g_malloc || !g_free || !g_device_count || !g_set_device ||
	    !g_get_device || !g_memset || !g_memcpy) {
		fprintf(stderr, "dlerror:%s\n", dlerror());
		dlclose(hip_lib_handle);
		return -1;
	}
	return 0;
}

/* clean up HIP library */
void hip_lib_fini(void)
{
	if (!hip_lib_handle)
		return;

	dlclose(hip_lib_handle);
	hip_lib_handle = NULL;

	g_malloc = NULL;
	g_free = NULL;
	g_device_count = NULL;
	g_set_device = NULL;
	g_get_device = NULL;
	g_memset = NULL;
	g_memcpy = NULL;
	g_mem_properties = NULL;
}

#endif /* HAVE_HIP_SUPPORT */
