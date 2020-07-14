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

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "ofi_hmem.h"
#include "ofi.h"

#ifdef HAVE_ROCR

#include <hsa/hsa_ext_amd.h>

struct rocr_ops {
	hsa_status_t (*hsa_memory_copy)(void *dst, const void *src,
					size_t size);
	hsa_status_t (*hsa_amd_pointer_info)(void *ptr,
					     hsa_amd_pointer_info_t *info,
					     void *(*alloc)(size_t),
					     uint32_t *num_agents_accessible,
					     hsa_agent_t **accessible);
	hsa_status_t (*hsa_init)(void);
	hsa_status_t (*hsa_shut_down)(void);
	hsa_status_t (*hsa_status_string)(hsa_status_t status,
					  const char **status_string);
};

#ifdef ENABLE_ROCR_DLOPEN

#include <dlfcn.h>

static void *rocr_handle;
static struct rocr_ops rocr_ops;

#else

static struct rocr_ops rocr_ops = {
	.hsa_memory_copy = hsa_memory_copy,
	.hsa_amd_pointer_info = hsa_amd_pointer_info,
	.hsa_init = hsa_init,
	.hsa_shut_down = hsa_shut_down,
	.hsa_status_string = hsa_status_string,
};

#endif /* ENABLE_ROCR_DLOPEN */

hsa_status_t ofi_hsa_memory_copy(void *dst, const void *src, size_t size)
{
	return rocr_ops.hsa_memory_copy(dst, src, size);
}

hsa_status_t ofi_hsa_amd_pointer_info(void *ptr, hsa_amd_pointer_info_t *info,
				      void *(*alloc)(size_t),
				      uint32_t *num_agents_accessible,
				      hsa_agent_t **accessible)
{
	return rocr_ops.hsa_amd_pointer_info(ptr, info, alloc,
					     num_agents_accessible, accessible);
}

hsa_status_t ofi_hsa_init(void)
{
	return rocr_ops.hsa_init();
}

hsa_status_t ofi_hsa_shut_down(void)
{
	return rocr_ops.hsa_shut_down();
}

hsa_status_t ofi_hsa_status_string(hsa_status_t status,
				   const char **status_string)
{
	return rocr_ops.hsa_status_string(status, status_string);
}

const char *ofi_hsa_status_to_string(hsa_status_t status)
{
	const char *str;
	hsa_status_t hsa_ret;

	hsa_ret = ofi_hsa_status_string(status, &str);
	if (hsa_ret != HSA_STATUS_SUCCESS)
		return "unknown error";

	return str;
}

int rocr_memcpy(void *dest, const void *src, size_t size)
{
	hsa_status_t hsa_ret;

	hsa_ret = ofi_hsa_memory_copy(dest, src, size);
	if (hsa_ret == HSA_STATUS_SUCCESS)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hsa_memory_copy: %s\n",
		ofi_hsa_status_to_string(hsa_ret));

	return -FI_EIO;
}

bool rocr_is_addr_valid(const void *addr)
{
	hsa_amd_pointer_info_t hsa_info = {
		.size = sizeof(hsa_info),
	};
	hsa_status_t hsa_ret;

	hsa_ret = ofi_hsa_amd_pointer_info((void *)addr, &hsa_info, NULL, NULL,
					   NULL);
	if (hsa_ret == HSA_STATUS_SUCCESS) {
		if (hsa_info.type == HSA_EXT_POINTER_TYPE_HSA)
			return true;
	} else {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hsa_amd_pointer_info: %s\n",
			ofi_hsa_status_to_string(hsa_ret));
	}

	return false;
}

int rocr_hmem_init(void)
{
	hsa_status_t hsa_ret;

#ifdef ENABLE_ROCR_DLOPEN
	/* Assume if dlopen fails, the ROCR library could not be found. Do not
	 * treat this as an error.
	 */
	rocr_handle = dlopen("libhsa-runtime64.so", RTLD_NOW);
	if (!rocr_handle) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Unable to dlopen libhsa-runtime64.so\n");
		return -FI_ENOSYS;
	}

	rocr_ops.hsa_memory_copy = dlsym(rocr_handle, "hsa_memory_copy");
	if (!rocr_ops.hsa_memory_copy) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hsa_memory_copy\n");
		goto err;
	}

	rocr_ops.hsa_amd_pointer_info = dlsym(rocr_handle,
					      "hsa_amd_pointer_info");
	if (!rocr_ops.hsa_amd_pointer_info) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hsa_amd_pointer_info\n");
		goto err;
	}

	rocr_ops.hsa_init = dlsym(rocr_handle, "hsa_init");
	if (!rocr_ops.hsa_init) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hsa_init\n");
		goto err;
	}

	rocr_ops.hsa_shut_down = dlsym(rocr_handle, "hsa_shut_down");
	if (!rocr_ops.hsa_shut_down) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hsa_shut_down\n");
		goto err;
	}

	rocr_ops.hsa_status_string = dlsym(rocr_handle, "hsa_status_string");
	if (!rocr_ops.hsa_status_string) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hsa_status_string\n");
		goto err;
	}
#endif /* ENABLE_ROCR_DLOPEN */

	hsa_ret = ofi_hsa_init();
	if (hsa_ret != HSA_STATUS_SUCCESS) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hsa_init: %s\n",
			ofi_hsa_status_to_string(hsa_ret));
		goto err;
	}

	return FI_SUCCESS;

err:
#ifdef ENABLE_ROCR_DLOPEN
	dlclose(rocr_handle);
#endif
	return -FI_ENODATA;
}

int rocr_hmem_cleanup(void)
{
	hsa_status_t hsa_ret;

	hsa_ret = ofi_hsa_shut_down();
	if (hsa_ret != HSA_STATUS_SUCCESS) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hsa_shut_down: %s\n",
			ofi_hsa_status_to_string(hsa_ret));
		return -FI_ENODATA;
	}

#ifdef ENABLE_ROCR_DLOPEN
	dlclose(rocr_handle);
#endif

	return FI_SUCCESS;
}

#else

int rocr_memcpy(void *dest, const void *src, size_t size)
{
	return -FI_ENOSYS;
}

int rocr_hmem_init(void)
{
	return -FI_ENOSYS;
}

int rocr_hmem_cleanup(void)
{
	return -FI_ENOSYS;
}

bool rocr_is_addr_valid(const void *addr)
{
	return false;
}

#endif /* HAVE_ROCR */
