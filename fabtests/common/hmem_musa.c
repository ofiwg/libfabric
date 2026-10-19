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

#include "hmem.h"
#include "shared.h"

#if HAVE_MUSA

#include <dlfcn.h>
#include <limits.h>
#include <musa_runtime.h>
#include <string.h>

struct musa_ops {
	musaError_t (*get_device_count)(int *count);
	musaError_t (*set_device)(int device);
	musaError_t (*copy)(void *dst, const void *src, size_t size,
			    enum musaMemcpyKind kind);
	musaError_t (*malloc)(void **ptr, size_t size);
	musaError_t (*free)(void *ptr);
	musaError_t (*malloc_host)(void **ptr, size_t size);
	musaError_t (*free_host)(void *ptr);
};

static struct musa_ops musa_ops;
static void *musa_handle;

static int musa_load_symbol(const char *name, void *symbol)
{
	void *value = dlsym(musa_handle, name);

	if (!value) {
		FT_ERR("Failed to find %s", name);
		return -FI_ENOSYS;
	}

	*(void **) symbol = value;
	return FI_SUCCESS;
}

static int musa_load(void)
{
	int ret;

	if (musa_handle)
		return FI_SUCCESS;

	musa_handle = dlopen("libmusart.so", RTLD_NOW);
	if (!musa_handle) {
		FT_ERR("Failed to dlopen libmusart.so");
		return -FI_ENOSYS;
	}

#define MUSA_RUNTIME_SYMBOL(field, name)                                 \
	do {                                                             \
		ret = musa_load_symbol(#name, (void *) &musa_ops.field); \
		if (ret)                                                 \
			goto err;                                        \
	} while (0)

	MUSA_RUNTIME_SYMBOL(get_device_count, musaGetDeviceCount);
	MUSA_RUNTIME_SYMBOL(set_device, musaSetDevice);
	MUSA_RUNTIME_SYMBOL(copy, musaMemcpy);
	MUSA_RUNTIME_SYMBOL(malloc, musaMalloc);
	MUSA_RUNTIME_SYMBOL(free, musaFree);
	MUSA_RUNTIME_SYMBOL(malloc_host, musaMallocHost);
	MUSA_RUNTIME_SYMBOL(free_host, musaFreeHost);

#undef MUSA_RUNTIME_SYMBOL
	return FI_SUCCESS;

err:
	dlclose(musa_handle);
	musa_handle = NULL;
	return ret;
}

static void musa_unload(void)
{
	if (musa_handle) {
		dlclose(musa_handle);
		musa_handle = NULL;
	}
}

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

static int musa_set_device(uint64_t device)
{
	if (device > INT_MAX)
		return -FI_EINVAL;

	return musa_error_to_fi(musa_ops.set_device((int) device));
}

int ft_musa_init(void)
{
	int device_count;
	int ret;

	ret = musa_load();
	if (ret)
		return ret;

	ret = musa_error_to_fi(musa_ops.get_device_count(&device_count));
	if (ret)
		return ret;

	return device_count > 0 ? FI_SUCCESS : -FI_ENODEV;
}

int ft_musa_cleanup(void)
{
	musa_unload();
	return FI_SUCCESS;
}

int ft_musa_alloc(uint64_t device, void **buf, size_t size)
{
	int ret;

	ret = musa_set_device(device);
	if (ret)
		return ret;

	return musa_error_to_fi(musa_ops.malloc(buf, size));
}

int ft_musa_alloc_host(void **buf, size_t size)
{
	return musa_error_to_fi(musa_ops.malloc_host(buf, size));
}

int ft_musa_free(void *buf)
{
	return musa_error_to_fi(musa_ops.free(buf));
}

int ft_musa_free_host(void *buf)
{
	return musa_error_to_fi(musa_ops.free_host(buf));
}

int ft_musa_memset(uint64_t device, void *buf, int value, size_t size)
{
	void *host_buf;
	int ret;

	if (!size)
		return FI_SUCCESS;

	host_buf = malloc(size);
	if (!host_buf)
		return -FI_ENOMEM;
	memset(host_buf, value, size);
	ret = ft_musa_copy_to_hmem(device, buf, host_buf, size);
	free(host_buf);
	return ret;
}

int ft_musa_copy_to_hmem(uint64_t device, void *dst, const void *src,
			 size_t size)
{
	int ret;

	ret = musa_set_device(device);
	if (ret)
		return ret;

	return musa_error_to_fi(
		musa_ops.copy(dst, src, size, musaMemcpyHostToDevice));
}

int ft_musa_copy_from_hmem(uint64_t device, void *dst, const void *src,
			   size_t size)
{
	int ret;

	ret = musa_set_device(device);
	if (ret)
		return ret;

	return musa_error_to_fi(
		musa_ops.copy(dst, src, size, musaMemcpyDeviceToHost));
}

#else

int ft_musa_init(void)
{
	return -FI_ENOSYS;
}

int ft_musa_cleanup(void)
{
	return -FI_ENOSYS;
}

int ft_musa_alloc(uint64_t device, void **buf, size_t size)
{
	return -FI_ENOSYS;
}

int ft_musa_alloc_host(void **buf, size_t size)
{
	return -FI_ENOSYS;
}

int ft_musa_free(void *buf)
{
	return -FI_ENOSYS;
}

int ft_musa_free_host(void *buf)
{
	return -FI_ENOSYS;
}

int ft_musa_memset(uint64_t device, void *buf, int value, size_t size)
{
	return -FI_ENOSYS;
}

int ft_musa_copy_to_hmem(uint64_t device, void *dst, const void *src,
			 size_t size)
{
	return -FI_ENOSYS;
}

int ft_musa_copy_from_hmem(uint64_t device, void *dst, const void *src,
			   size_t size)
{
	return -FI_ENOSYS;
}

#endif

int ft_musa_get_dmabuf_fd(void *buf, size_t len, int *fd, uint64_t *offset)
{
	(void) buf;
	(void) len;
	(void) fd;
	(void) offset;
	return -FI_ENOSYS;
}

int ft_musa_put_dmabuf_fd(int fd)
{
	(void) fd;
	return -FI_ENOSYS;
}
