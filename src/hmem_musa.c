/*
 * Copyright (c) 2026 MThreads.
 *
 * This software is available to you under the BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions are
 *     met:
 *
 *      - Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "ofi.h"
#include "ofi_hmem.h"

#if HAVE_MUSA

#include <dlfcn.h>
#include <stdio.h>
#include <musa.h>
#include <musa_runtime.h>

#ifndef MUSA_LIBDIR
#define MUSA_LIBDIR ""
#endif

struct musa_ops {
	musaError_t (*get_device_count)(int *count);
	musaError_t (*set_device)(int device);
	musaError_t (*pointer_get_attributes)(
		struct musaPointerAttributes *attr, const void *ptr);
	musaError_t (*memcpy)(void *dst, const void *src, size_t size,
			      enum musaMemcpyKind kind);
	musaError_t (*malloc)(void **ptr, size_t size);
	musaError_t (*free)(void *ptr);
	musaError_t (*malloc_host)(void **ptr, size_t size);
	musaError_t (*free_host)(void *ptr);
	musaError_t (*host_register)(void *ptr, size_t size, unsigned int flags);
	musaError_t (*host_unregister)(void *ptr);
	const char *(*get_error_string)(musaError_t error);
	MUresult (*mem_get_address_range)(MUdeviceptr *base, size_t *size,
					  MUdeviceptr ptr);
};

struct musa_state {
	bool initialized;
	int device_count;
#if ENABLE_MUSA_DLOPEN
	void *runtime_handle;
	void *driver_handle;
#endif
};

#if ENABLE_MUSA_DLOPEN
static struct musa_ops musa_ops;
#else
static struct musa_ops musa_ops = {
	.get_device_count = musaGetDeviceCount,
	.set_device = musaSetDevice,
	.pointer_get_attributes = musaPointerGetAttributes,
	.memcpy = musaMemcpy,
	.malloc = musaMalloc,
	.free = musaFree,
	.malloc_host = musaMallocHost,
	.free_host = musaFreeHost,
	.host_register = musaHostRegister,
	.host_unregister = musaHostUnregister,
	.get_error_string = musaGetErrorString,
	.mem_get_address_range = muMemGetAddressRange,
};
#endif

static struct musa_state musa_state;

static int musa_error_to_fi(musaError_t error)
{
	switch (error) {
	case musaSuccess:
		return FI_SUCCESS;
	case musaErrorMemoryAllocation:
		return -FI_ENOMEM;
	case musaErrorInvalidValue:
	case musaErrorInvalidDevice:
		return -FI_EINVAL;
	default:
		return -FI_EIO;
	}
}

static const char *musa_error_string(musaError_t error)
{
	if (musa_ops.get_error_string)
		return musa_ops.get_error_string(error);
	return "unknown MUSA error";
}

#if ENABLE_MUSA_DLOPEN
static int musa_load_symbol(void *handle, const char *name, void *symbol)
{
	void *value = dlsym(handle, name);

	if (!value) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find MUSA symbol %s\n", name);
		return -FI_ENOSYS;
	}

	*(void **) symbol = value;
	return FI_SUCCESS;
}

static void *musa_open_library(const char *name)
{
	char path[4096];
	int path_len;
	void *handle;

	if (!MUSA_LIBDIR[0])
		return dlopen(name, RTLD_NOW | RTLD_LOCAL);

	path_len = snprintf(path, sizeof(path), "%s/%s", MUSA_LIBDIR, name);
	if (path_len < 0 || path_len >= (int) sizeof(path))
		return NULL;

	handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (handle)
		return handle;

	/* The library may not live in the SDK directory. The driver library
	 * (libmusa.so.1) is typically installed in the system loader path
	 * while the SDK directory only ships runtime and stub libraries.
	 * Fall back to the default search path. */
	FI_INFO(&core_prov, FI_LOG_CORE,
		"Failed to dlopen %s, falling back to default search path\n",
		path);
	return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

static int musa_hmem_dl_init(void)
{
	int ret;

	musa_state.runtime_handle = musa_open_library("libmusart.so");
	if (!musa_state.runtime_handle)
		goto err;

	musa_state.driver_handle = musa_open_library("libmusa.so.1");
	if (!musa_state.driver_handle)
		goto err;

#define MUSA_RUNTIME_SYMBOL(field, name) \
	do { \
		ret = musa_load_symbol(musa_state.runtime_handle, #name, \
				       (void *) &musa_ops.field); \
		if (ret) \
			goto err; \
	} while (0)

	MUSA_RUNTIME_SYMBOL(get_device_count, musaGetDeviceCount);
	MUSA_RUNTIME_SYMBOL(set_device, musaSetDevice);
	MUSA_RUNTIME_SYMBOL(pointer_get_attributes, musaPointerGetAttributes);
	MUSA_RUNTIME_SYMBOL(memcpy, musaMemcpy);
	MUSA_RUNTIME_SYMBOL(malloc, musaMalloc);
	MUSA_RUNTIME_SYMBOL(free, musaFree);
	MUSA_RUNTIME_SYMBOL(malloc_host, musaMallocHost);
	MUSA_RUNTIME_SYMBOL(free_host, musaFreeHost);
	MUSA_RUNTIME_SYMBOL(host_register, musaHostRegister);
	MUSA_RUNTIME_SYMBOL(host_unregister, musaHostUnregister);
	MUSA_RUNTIME_SYMBOL(get_error_string, musaGetErrorString);

#undef MUSA_RUNTIME_SYMBOL

	ret = musa_load_symbol(musa_state.driver_handle,
			       "muMemGetAddressRange_v2",
			       (void *) &musa_ops.mem_get_address_range);
	if (ret)
		ret = musa_load_symbol(musa_state.driver_handle,
				       "muMemGetAddressRange",
				       (void *) &musa_ops.mem_get_address_range);
	if (ret)
		goto err;

	return FI_SUCCESS;

err:
	if (musa_state.driver_handle)
		dlclose(musa_state.driver_handle);
	if (musa_state.runtime_handle)
		dlclose(musa_state.runtime_handle);
	musa_state.driver_handle = NULL;
	musa_state.runtime_handle = NULL;
	return -FI_ENOSYS;
}

static void musa_hmem_dl_cleanup(void)
{
	if (musa_state.driver_handle)
		dlclose(musa_state.driver_handle);
	if (musa_state.runtime_handle)
		dlclose(musa_state.runtime_handle);
	musa_state.driver_handle = NULL;
	musa_state.runtime_handle = NULL;
	memset(&musa_ops, 0, sizeof(musa_ops));
}
#else
static int musa_hmem_dl_init(void)
{
	return FI_SUCCESS;
}

static void musa_hmem_dl_cleanup(void)
{
}
#endif

int musa_hmem_init(void)
{
	musaError_t error;
	int device_count;
	int ret;

	if (musa_state.initialized)
		return FI_SUCCESS;

	ret = musa_hmem_dl_init();
	if (ret)
		return ret;

	error = musa_ops.get_device_count(&device_count);
	if (error != musaSuccess) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"MUSA runtime is unavailable: %s\n",
			musa_error_string(error));
		musa_hmem_dl_cleanup();
		return musa_error_to_fi(error);
	}

	if (device_count <= 0) {
		musa_hmem_dl_cleanup();
		return -FI_ENODEV;
	}

	musa_state.device_count = device_count;
	musa_state.initialized = true;
	return FI_SUCCESS;
}

int musa_hmem_cleanup(void)
{
	musa_state.initialized = false;
	musa_state.device_count = 0;
	musa_hmem_dl_cleanup();
	return FI_SUCCESS;
}

bool musa_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	struct musaPointerAttributes attr;
	musaError_t error;

	if (!musa_state.initialized || !addr)
		return false;

	error = musa_ops.pointer_get_attributes(&attr, addr);
	if (error != musaSuccess)
		return false;

	if (attr.type != musaMemoryTypeDevice &&
	    attr.type != musaMemoryTypeManaged)
		return false;

	if (attr.device < 0 || attr.device >= musa_state.device_count)
		return false;

	if (device)
		*device = (uint64_t) attr.device;
	if (flags)
		*flags = attr.type == musaMemoryTypeDevice ?
			FI_HMEM_DEVICE_ONLY : 0;

	return true;
}

static int musa_validate_device_pointer(const void *addr, uint64_t *device)
{
	if (musa_is_addr_valid(addr, device, NULL))
		return FI_SUCCESS;
	return -FI_EFAULT;
}

static int musa_query_base_addr(const void *ptr, void **base, size_t *size)
{
	MUdeviceptr musa_base;
	MUresult result;

	if (!musa_state.initialized)
		return -FI_ENOSYS;
	if (!base || !size)
		return -FI_EINVAL;
	if (musa_validate_device_pointer(ptr, NULL))
		return -FI_EFAULT;

	result = musa_ops.mem_get_address_range(&musa_base, size,
						(MUdeviceptr) ptr);
	if (result == MUSA_SUCCESS) {
		*base = (void *) (uintptr_t) musa_base;
		return FI_SUCCESS;
	}

	if (result == MUSA_ERROR_INVALID_VALUE)
		return -FI_EFAULT;
	return -FI_EIO;
}

static int musa_validate_device_range(const void *addr, size_t len,
				      uint64_t *device)
{
	void *base;
	size_t allocation_size;
	uintptr_t addr_value;
	uintptr_t base_value;
	size_t offset;
	int ret;

	ret = musa_validate_device_pointer(addr, device);
	if (ret || !len)
		return ret;

	ret = musa_query_base_addr(addr, &base, &allocation_size);
	if (ret)
		return ret;

	addr_value = (uintptr_t) addr;
	base_value = (uintptr_t) base;
	if (addr_value < base_value)
		return -FI_EFAULT;
	offset = addr_value - base_value;
	if (offset > allocation_size || len > allocation_size - offset)
		return -FI_EFAULT;

	return FI_SUCCESS;
}

int musa_copy_to_dev(uint64_t device, void *dev, const void *host, size_t size)
{
	musaError_t error;
	uint64_t detected_device;
	int ret;

	ret = musa_validate_device_range(dev, size, &detected_device);
	if (ret)
		return ret;
	if (device != detected_device)
		return -FI_EINVAL;

	error = musa_ops.memcpy(dev, host, size, musaMemcpyHostToDevice);
	if (error == musaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform musaMemcpy H2D: %s\n",
		musa_error_string(error));
	return musa_error_to_fi(error);
}

int musa_copy_from_dev(uint64_t device, void *host, const void *dev, size_t size)
{
	musaError_t error;
	uint64_t detected_device;
	int ret;

	ret = musa_validate_device_range(dev, size, &detected_device);
	if (ret)
		return ret;
	if (device != detected_device)
		return -FI_EINVAL;

	error = musa_ops.memcpy(host, dev, size, musaMemcpyDeviceToHost);
	if (error == musaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform musaMemcpy D2H: %s\n",
		musa_error_string(error));
	return musa_error_to_fi(error);
}

int musa_host_register(void *ptr, size_t size)
{
	musaError_t error;

	if (!musa_state.initialized)
		return -FI_ENOSYS;

	error = musa_ops.host_register(ptr, size, 0);
	if (error == musaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform musaHostRegister: %s\n",
		musa_error_string(error));
	return musa_error_to_fi(error);
}

int musa_host_unregister(void *ptr)
{
	musaError_t error;

	if (!musa_state.initialized)
		return -FI_ENOSYS;

	error = musa_ops.host_unregister(ptr);
	if (error == musaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform musaHostUnregister: %s\n",
		musa_error_string(error));
	return musa_error_to_fi(error);
}

int musa_get_base_addr(const void *ptr, size_t len, void **base, size_t *size)
{
	void *allocation_base;
	size_t allocation_size;
	uintptr_t ptr_value;
	uintptr_t base_value;
	size_t offset;
	int ret;

	if (!base || !size)
		return -FI_EINVAL;

	ret = musa_query_base_addr(ptr, &allocation_base, &allocation_size);
	if (ret)
		return ret;

	ptr_value = (uintptr_t) ptr;
	base_value = (uintptr_t) allocation_base;
	if (ptr_value < base_value)
		return -FI_EFAULT;
	offset = ptr_value - base_value;
	if (offset > allocation_size || len > allocation_size - offset)
		return -FI_EFAULT;

	*base = allocation_base;
	*size = allocation_size;
	return FI_SUCCESS;
}

int musa_dev_alloc(uint64_t device, uint64_t size, uint64_t alignment,
		   uint64_t flags, void **addr, int *fd, uint64_t *offset)
{
	musaError_t error;

	(void) alignment;
	if (!addr)
		return -FI_EINVAL;
	if (!musa_state.initialized)
		return -FI_ENOSYS;
	if (device >= (uint64_t) musa_state.device_count)
		return -FI_EINVAL;
	if (flags & FI_XPU_ALLOC_DMABUF)
		return -FI_EOPNOTSUPP;

	if (fd)
		*fd = -1;
	if (offset)
		*offset = 0;

	error = musa_ops.set_device((int) device);
	if (error != musaSuccess)
		return musa_error_to_fi(error);

	error = musa_ops.malloc(addr, size);
	if (error == musaSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"musaMalloc failed: %s\n", musa_error_string(error));
	return musa_error_to_fi(error);
}

void musa_dev_free(uint64_t device, void *addr)
{
	(void) device;
	if (musa_state.initialized && addr)
		(void) musa_ops.free(addr);
}

#else /* HAVE_MUSA */

int musa_copy_to_dev(uint64_t device, void *dev, const void *host, size_t size)
{
	(void) device;
	(void) dev;
	(void) host;
	(void) size;
	return -FI_ENOSYS;
}

int musa_copy_from_dev(uint64_t device, void *host, const void *dev, size_t size)
{
	(void) device;
	(void) host;
	(void) dev;
	(void) size;
	return -FI_ENOSYS;
}

int musa_hmem_init(void)
{
	return -FI_ENOSYS;
}

int musa_hmem_cleanup(void)
{
	return FI_SUCCESS;
}

bool musa_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	(void) addr;
	(void) device;
	(void) flags;
	return false;
}

int musa_host_register(void *ptr, size_t size)
{
	(void) ptr;
	(void) size;
	return -FI_ENOSYS;
}

int musa_host_unregister(void *ptr)
{
	(void) ptr;
	return -FI_ENOSYS;
}

int musa_get_base_addr(const void *ptr, size_t len, void **base, size_t *size)
{
	(void) ptr;
	(void) len;
	(void) base;
	(void) size;
	return -FI_ENOSYS;
}

int musa_dev_alloc(uint64_t device, uint64_t size, uint64_t alignment,
		   uint64_t flags, void **addr, int *fd, uint64_t *offset)
{
	(void) device;
	(void) size;
	(void) alignment;
	(void) flags;
	(void) addr;
	(void) fd;
	(void) offset;
	return -FI_ENOSYS;
}

void musa_dev_free(uint64_t device, void *addr)
{
	(void) device;
	(void) addr;
}

#endif /* HAVE_MUSA */
