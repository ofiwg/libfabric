/*
 * Copyright (c) 2020 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
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

#ifdef HAVE_GDRCOPY

#include <gdrapi.h>

struct gdrcopy_ops {

	gdr_t (*gdr_open)();

	int (*gdr_close)(gdr_t g);

	int (*gdr_pin_buffer)(gdr_t g, unsigned long addr, size_t size,
			      uint64_t p2p_token, uint32_t va_space,
			      gdr_mh_t *handle);

	int (*gdr_unpin_buffer)(gdr_t g, gdr_mh_t handle);

	int (*gdr_map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);

	int (*gdr_unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);

	int (*gdr_copy_to_mapping)(gdr_mh_t handle, void *map_d_ptr,
				   const void *h_ptr, size_t size);

	int (*gdr_copy_from_mapping)(gdr_mh_t handle, void *map_d_ptr,
				     const void *h_ptr, size_t size);
};

#ifdef ENABLE_GDRCOPY_DLOPEN

#include <dlfcn.h>

static void *gdrapi_handle;
static struct gdrcopy_ops gdrcopy_ops;

#else

static struct gdrcopy_ops gdrcopy_ops = {
	.gdr_open = gdr_open,
	.gdr_close = gdr_close,
	.gdr_pin_buffer = gdr_pin_buffer,
	.gdr_unpin_buffer = gdr_unpin_buffer,
	.gdr_map = gdr_map,
	.gdr_unmap = gdr_unmap,
	.gdr_copy_to_mapping = gdr_copy_to_mapping,
	.gdr_copy_from_mapping = gdr_copy_from_mapping
};

#endif /* ENABLE_CUDA_DLOPEN */

gdr_t ofi_gdrcopy_open()
{
	if (!gdrcopy_ops.gdr_open)
		return NULL;

	return gdrcopy_ops.gdr_open();
}

int ofi_gdrcopy_close(gdr_t gdr)
{
	if (!gdrcopy_ops.gdr_close)
		return -FI_ENOSYS;

	return gdrcopy_ops.gdr_close(gdr);
}

ssize_t ofi_gdrcopy_reg(void *addr, size_t len, gdr_t gdr, struct ofi_gdrcopy_handle *gdrcopy)
{
	ssize_t err;
	uintptr_t regbgn = (uintptr_t)addr;
	uintptr_t regend = (uintptr_t)addr + len;
	size_t reglen;

	assert(gdr);
	assert(gdrcopy_ops.gdr_pin_buffer);
	assert(gdrcopy_ops.gdr_map);

	regbgn = regbgn & GPU_PAGE_MASK;
	regend = (regend & GPU_PAGE_MASK) + GPU_PAGE_SIZE;
	reglen = regend - regbgn;

	assert(gdr);
	err = gdrcopy_ops.gdr_pin_buffer(gdr, regbgn,
					 reglen, 0, 0, &gdrcopy->mh);
	if (err) {
		FI_WARN(&core_prov, FI_LOG_CORE, "gdr_pin_buffer failed! error: %s",
			strerror(err));
		return err;
	}

	gdrcopy->cuda_ptr = (void *)regbgn;
	gdrcopy->length = reglen;

	err = gdrcopy_ops.gdr_map(gdr, gdrcopy->mh, &gdrcopy->user_ptr, gdrcopy->length);
	if (err) {
		FI_WARN(&core_prov, FI_LOG_CORE, "gdr_map failed! error: %s\n",
			strerror(err));
		gdrcopy_ops.gdr_unpin_buffer(gdr, gdrcopy->mh);
		return err;
	}

	return 0;
}

ssize_t ofi_gdrcopy_dereg(gdr_t gdr, struct ofi_gdrcopy_handle *gdrcopy)
{
	ssize_t err;

	assert(gdr);
	assert(gdrcopy);
	assert(gdrcopy_ops.gdr_unmap);
	assert(gdrcopy_ops.gdr_unpin_buffer);
	err = gdrcopy_ops.gdr_unmap(gdr, gdrcopy->mh, gdrcopy->user_ptr, gdrcopy->length);
	if (err) {
		FI_WARN(&core_prov, FI_LOG_CORE, "gdr_unmap failed! error: %s\n",
			strerror(err));
		return err;
	}

	err = gdrcopy_ops.gdr_unpin_buffer(gdr, gdrcopy->mh);
	if (err) {
		FI_WARN(&core_prov, FI_LOG_MR, "gdr_unmap failed! error: %s\n",
			strerror(err));
		return err;
	}

	return 0;
}

/*
 * The following 4 functions
 *
 *     cuda_gdrcopy_hmem_init()
 *     cuda_gdrcopy_hmem_cleanup()
 *     cuda_gdrcopy_to_dev()
 *     cuda_gdrcopy_from_dev()
 *
 * are called by the corresponding cuda_hmem functions.
 */

int cuda_gdrcopy_hmem_init(void)
{
#ifdef ENABLE_GDRCOPY_DLOPEN
	gdrapi_handle = dlopen("libgdrapi.so", RTLD_NOW);
	if (!gdrapi_handle) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libgdrapi.so\n");
		return -FI_ENOSYS;
	}

	gdrcopy_ops.gdr_open = dlsym(gdrapi_handle, "gdr_open");
	if (!gdrcopy_ops.gdr_open) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_open\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_close = dlsym(gdrapi_handle, "gdr_close");
	if (!gdrcopy_ops.gdr_close) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_close\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_pin_buffer = dlsym(gdrapi_handle, "gdr_pin_buffer");
	if (!gdrcopy_ops.gdr_pin_buffer) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_pin_buffer\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_unpin_buffer = dlsym(gdrapi_handle, "gdr_unpin_buffer");
	if (!gdrcopy_ops.gdr_unpin_buffer) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_unpin_buffer\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_map = dlsym(gdrapi_handle, "gdr_map");
	if (!gdrcopy_ops.gdr_map) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_map\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_unmap = dlsym(gdrapi_handle, "gdr_unmap");
	if (!gdrcopy_ops.gdr_unmap) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_unmap\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_copy_to_mapping = dlsym(gdrapi_handle, "gdr_copy_to_mapping");
	if (!gdrcopy_ops.gdr_copy_to_mapping) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_copy_to_mapping\n");
		goto err_dlclose_gdrapi;
	}

	gdrcopy_ops.gdr_copy_from_mapping = dlsym(gdrapi_handle, "gdr_copy_from_mapping");
	if (!gdrcopy_ops.gdr_copy_from_mapping) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find gdr_copy_from_mapping\n");
		goto err_dlclose_gdrapi;
	}

	return FI_SUCCESS;

err_dlclose_gdrapi:
	memset(&gdrcopy_ops, 0, sizeof(gdrcopy_ops));
	dlclose(gdrapi_handle);
	return -FI_ENODATA;
#else /* ENABLE_GDRCOPY_DLOPEN */
	return FI_SUCCESS;
#endif /* ENABLE_GDRCOPY_DLOPEN */
}

int cuda_gdrcopy_hmem_cleanup(void)
{
#ifdef ENABLE_GDRCOPY_DLOPEN
	dlclose(gdrapi_handle);
#endif
	return FI_SUCCESS;
}

int cuda_gdrcopy_to_dev(uint64_t devhandle, void *devptr, const void *hostptr, size_t len)
{
	ssize_t off;
	struct ofi_gdrcopy_handle *gdrcopy;
	void *gdrcopy_user_ptr;

	if (!gdrcopy_ops.gdr_copy_to_mapping)
		return -FI_ENOSYS;

	gdrcopy = (struct ofi_gdrcopy_handle *)devhandle;
	off = (char *)devptr - (char *)gdrcopy->cuda_ptr;
	assert(off >= 0 && off + len <= gdrcopy->length);
	gdrcopy_user_ptr = (char *)gdrcopy->user_ptr + off;
	gdrcopy_ops.gdr_copy_to_mapping(gdrcopy->mh, gdrcopy_user_ptr, hostptr, len);
	return 0;
}

int cuda_gdrcopy_from_dev(uint64_t devhandle, void *hostptr, const void *devptr, size_t len)
{
	ssize_t off;
	struct ofi_gdrcopy_handle *gdrcopy;
	void *gdrcopy_user_ptr;

	if (!gdrcopy_ops.gdr_copy_from_mapping)
		return -FI_ENOSYS;

	gdrcopy = (struct ofi_gdrcopy_handle *)devhandle;

	off = (char *)devptr - (char *)gdrcopy->cuda_ptr;
	assert(off >= 0 && off + len <= gdrcopy->length);
	gdrcopy_user_ptr = (char *)gdrcopy->user_ptr + off;
	gdrcopy_ops.gdr_copy_from_mapping(gdrcopy->mh, gdrcopy_user_ptr, hostptr, len);
	return 0;
}

#endif /* HAVE_GDRCOPY */
