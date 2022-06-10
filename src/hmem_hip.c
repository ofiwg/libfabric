/*
 * (C) Copyright 2022 Oak Ridge National Lab
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

#if HAVE_ROCR

#define D2H_THRESHOLD 16384

hipStream_t global_hip_streams[HMEM_NUM_STREAMS];
ofi_atomic32_t global_hip_stream_iter;
ofi_atomic32_t global_hip_stream_init;

struct hip_ops {
	hipError_t (*hipMemcpyAsync)(void *dst, const void *src, size_t size,
				  hipMemcpyKind kind, hipStream_t stream);
	hipError_t (*hipStreamCreate)(hipStream_t *stream);
	hipError_t (*hipStreamCreateWithFlags)(hipStream_t *stream, unsigned int flags);
	hipError_t (*hipStreamDestroy)(hipStream_t stream);
	hipError_t (*hipStreamQuery)(hipStream_t stream);
	hipError_t (*hipStreamSynchronize)(hipStream_t stream);
	hipError_t (*hipMemcpy)(void *dst, const void *src, size_t size,
				  hipMemcpyKind kind);
	hipError_t (*hipFree)(void *ptr);
	hipError_t (*hipMalloc)(void **ptr, size_t size);
	const char *(*hipGetErrorName)(hipError_t hip_error);
	const char *(*hipGetErrorString)(hipError_t hip_error);
	hipError_t (*hipPointerGetAttributes)(hipPointerAttribute_t *attribute,
					  const void *ptr);
	hipError_t (*hipHostRegister)(void *host_ptr, size_t size,
					unsigned int flags);
	hipError_t (*hipHostUnregister)(void *host_ptr);
	hipError_t (*hipGetDeviceCount)(int *count);
	hipError_t (*hipGetDevice)(int *device);
	hipError_t (*hipSetDevice)(int device);
	hipError_t (*hipIpcOpenMemHandle)(void **devptr,
					    hipIpcMemHandle_t handle,
					    unsigned int flags);
	hipError_t (*hipIpcGetMemHandle)(hipIpcMemHandle_t *handle,
					   void *devptr);
	hipError_t (*hipIpcCloseMemHandle)(void *devptr);
};

static bool hip_ipc_enabled;

static hipError_t hip_disabled_hipMemcpy(void *dst, const void *src,
					    size_t size, hipMemcpyKind kind);

#if ENABLE_HIP_DLOPEN

#include <dlfcn.h>

static void *hip_handle;
static struct hip_ops hip_ops;

#else

static struct hip_ops hip_ops = {
	.hipMemcpyAsync = hipMemcpyAsync,
	.hipStreamCreateWithFlags = hipStreamCreateWithFlags,
	.hipStreamCreate = hipStreamCreate,
	.hipStreamDestroy = hipStreamDestroy,
	.hipStreamQuery = hipStreamQuery,
	.hipStreamSynchronize = hipStreamSynchronize,
	.hipMemcpy = hipMemcpy,
	.hipFree = hipFree,
	.hipMalloc = hipMalloc,
	.hipGetErrorName = hipGetErrorName,
	.hipGetErrorString = hipGetErrorString,
	.hipPointerGetAttributes = hipPointerGetAttributes,
	.hipHostRegister = hipHostRegister,
	.hipHostUnregister = hipHostUnregister,
	.hipGetDeviceCount = hipGetDeviceCount,
	.hipGetDevice = hipGetDevice,
	.hipSetDevice = hipSetDevice,
	.hipIpcOpenMemHandle = hipIpcOpenMemHandle,
	.hipIpcGetMemHandle = hipIpcGetMemHandle,
	.hipIpcCloseMemHandle = hipIpcCloseMemHandle
};

#endif /* ENABLE_HIP_DLOPEN */

hipError_t ofi_hipMemcpyAsync(void *dst, const void *src, size_t size,
						 hipMemcpyKind kind, hipStream_t stream)
{
	return hip_ops.hipMemcpyAsync(dst, src, size, kind, stream);
}

hipError_t ofi_hipMemcpy(void *dst, const void *src, size_t size,
						 hipMemcpyKind kind)
{
	return hip_ops.hipMemcpy(dst, src, size, kind);
}

hipError_t ofi_hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
	return hip_ops.hipStreamCreateWithFlags(stream, flags);
}

hipError_t ofi_hipStreamCreate(hipStream_t *stream)
{
	return hip_ops.hipStreamCreate(stream);
}

hipError_t ofi_hipStreamDestroy(hipStream_t stream)
{
	return hip_ops.hipStreamDestroy(stream);
}

hipError_t ofi_hipStreamQuery(hipStream_t stream)
{
	return hip_ops.hipStreamQuery(stream);
}

hipError_t ofi_hipStreamSynchronize(hipStream_t stream)
{
	return hip_ops.hipStreamSynchronize(stream);
}

const char *ofi_hipGetErrorName(hipError_t error)
{
	return hip_ops.hipGetErrorName(error);
}

const char *ofi_hipGetErrorString(hipError_t error)
{
	return hip_ops.hipGetErrorString(error);
}

hipError_t ofi_hipPointerGetAttributes(hipPointerAttribute_t *attribute,
				   const void *ptr)
{
	uint64_t start;
	hipError_t rc;

	start = ofi_gettime_ns();
	rc = hip_ops.hipPointerGetAttributes(attribute, ptr);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipPointerGetAttributes took %lu ns\n",
			ofi_gettime_ns() - start);
	return rc;
}

hipError_t ofi_hipHostRegister(void *ptr, size_t size, unsigned int flags)
{
	uint64_t start;
	hipError_t rc;

	start = ofi_gettime_ns();
	rc = hip_ops.hipHostRegister(ptr, size, flags);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"ofi_hipHostRegister took %lu ns\n",
			ofi_gettime_ns() - start);

	return rc;
}

hipError_t ofi_hipHostUnregister(void *ptr)
{
	uint64_t start;
	hipError_t rc = hipSuccess;

	start = ofi_gettime_ns();
	rc = hip_ops.hipHostUnregister(ptr);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"ofi_hipHostUnregister took %lu ns\n",
			ofi_gettime_ns() - start);

	return rc;
}

static hipError_t ofi_hipGetDeviceCount(int *count)
{
	return hip_ops.hipGetDeviceCount(count);
}

hipError_t ofi_hipMalloc(void **ptr, size_t size)
{
	return hip_ops.hipMalloc(ptr, size);
}

hipError_t ofi_hipFree(void *ptr)
{
	return hip_ops.hipFree(ptr);
}

static int
hip_dev_async_copy(void *dst, const void *src, size_t size,
				   void *istream, void **ostream)
{
	hipPointerAttribute_t attr;
	hipError_t hip_ret;
	hipStream_t stream;
	uint64_t start;

	/* if the destination is a host pointer and the size is less than 16K
	 * then use memcpy
	 */
	if (size < D2H_THRESHOLD) {
		memset(&attr, 0, sizeof(attr));

		hip_ret = ofi_hipPointerGetAttributes(&attr, dst);
		if (hip_ret == hipSuccess) {
			if (attr.memoryType == hipMemoryTypeHost) {
				memcpy(dst, src, size);
				return 0;
			}
		}
	}

	if ((ostream && istream) || (!ostream && !istream))
		return -FI_EINVAL;

	if (ostream) {
		int stream_id;
		/*
		start = ofi_gettime_ns();
		hip_ret = ofi_hipStreamCreateWithFlags(&stream,
											   hipStreamNonBlocking);
		FI_INFO(&core_prov, FI_LOG_CORE,
				"hipStreamCreateWithFlags took %lu ns\n", ofi_gettime_ns() - start);
		if (hip_ret != hipSuccess) {
			FI_WARN(&core_prov, FI_LOG_CORE,
				"Failed to perform hipStreamCreateWithFlags: %s:%s\n",
				ofi_hipGetErrorName(hip_ret),
				ofi_hipGetErrorString(hip_ret));
			goto fail;
		}
		*ostream = stream;
		*/
		stream_id = ofi_atomic_get32(&global_hip_stream_iter)
		  % HMEM_NUM_STREAMS;
		ofi_atomic_inc32(&global_hip_stream_iter);
		*ostream = (void*)stream_id;
		stream = global_hip_streams[stream_id];
	} else if (istream) {
		stream = global_hip_streams[(uint32_t)istream];
	}

	start = ofi_gettime_ns();
	hip_ret = ofi_hipMemcpyAsync(dst, src, size, hipMemcpyDefault,
								 stream);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipMemcpyAsync took %lu ns\n", ofi_gettime_ns() - start);
	if (hip_ret == hipSuccess)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipMemcpyAsync: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));
/*
	start = ofi_gettime_ns();
	hip_ret = ofi_hipStreamDestroy(stream);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipStreamDestroy took %lu ns\n", ofi_gettime_ns() - start);
	if (hip_ret != hipSuccess) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipStreamDestroy: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));
	}
*/
	return -FI_EIO;
}

int hip_async_copy_to_dev(uint64_t device, void *dst, const void *src,
						  size_t size, void *istream, void **ostream)
{
	return hip_dev_async_copy(dst, src, size, istream, ostream);
}

int hip_async_copy_from_dev(uint64_t device, void *dst, const void *src,
							size_t size, void *istream, void **ostream)
{
	return hip_dev_async_copy(dst, src, size, istream, ostream);
}

int hip_async_copy_query(void *stream)
{
	int ret = -FI_EINVAL;
	hipError_t hip_ret;
	hipStream_t hip_stream = stream;
	uint64_t start = ofi_gettime_ns();

	hip_ret = ofi_hipStreamQuery(hip_stream);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipStreamQuery took %lu ns\n", ofi_gettime_ns() - start);
	if (hip_ret == hipSuccess) {
		ret = 0;
	} else if (hip_ret == hipErrorNotReady) {
		return -FI_EWOULDBLOCK;
	}

	if (ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipStreamQuery: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));
	}

	start = ofi_gettime_ns();
	if ((hip_ret = ofi_hipStreamDestroy(hip_stream)) == hipErrorInvalidHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipStreamDestroy: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));
	}
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipStreamDestroy took %lu ns\n", ofi_gettime_ns() - start);

	return ret;
}

int hip_stream_synchronize(int stream_id)
{
	hipError_t hip_ret;
	uint64_t start;

	if (stream_id < 0 || stream_id > HMEM_NUM_STREAMS)
		return -FI_EINVAL;

	start = ofi_gettime_ns();
	hip_ret = ofi_hipStreamSynchronize(global_hip_streams[stream_id]);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipStreamSynchronize took %lu ns\n", ofi_gettime_ns() - start);

	if (hip_ret == hipSuccess)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipStreamSynchronize: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));

	return -FI_EINVAL;
}

static int
hip_dev_copy(void *dst, const void *src, size_t size)
{
	hipError_t hip_ret;
	uint64_t start = ofi_gettime_ns();

	hip_ret = ofi_hipMemcpy(dst, src, size, hipMemcpyDefault);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipMemcpy took %lu ns\n", ofi_gettime_ns() - start);
	if (hip_ret == hipSuccess)
		return 0;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipMemcpy: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));

	return -FI_EIO;
}

int hip_copy_to_dev(uint64_t device, void *dst, const void *src, size_t size)
{
	return hip_dev_copy(dst, src, size);
}

int hip_copy_from_dev(uint64_t device, void *dst, const void *src, size_t size)
{
	return hip_dev_copy(dst, src, size);
}

int hip_get_handle(void *dev_buf, size_t *len, void **handle, uint64_t *offset)
{
	hipError_t hip_ret;
	uint64_t start;

	start = ofi_gettime_ns();
	hip_ret = hip_ops.hipIpcGetMemHandle((hipIpcMemHandle_t *)handle,
						dev_buf);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipIpcGetMemHandle took %lu ns\n",
			ofi_gettime_ns() - start);

	if (hip_ret == hipSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipIpcGetMemHandle: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));

	return -FI_EINVAL;
}

int hip_open_handle(void **handle, size_t len, uint64_t device, void **ipc_ptr)
{
	hipError_t hip_ret;
	uint64_t start;

	start = ofi_gettime_ns();
	hip_ret = hip_ops.hipIpcOpenMemHandle(ipc_ptr,
						 *(hipIpcMemHandle_t *)handle,
						 hipIpcMemLazyEnablePeerAccess);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipIpcOpenMemHandle took %lu ns\n",
			ofi_gettime_ns() - start);

	if (hip_ret == hipSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipIpcOpenMemHandle: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));

	return -FI_EINVAL;
}

int hip_close_handle(void *ipc_ptr)
{
	hipError_t hip_ret = hipSuccess;
	uint64_t start;

	start = ofi_gettime_ns();
	hip_ret = hip_ops.hipIpcCloseMemHandle(ipc_ptr);
	FI_INFO(&core_prov, FI_LOG_CORE,
			"hipIpcCloseMemHandle took %lu ns\n",
			ofi_gettime_ns() - start);

	if (hip_ret == hipSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipIpcCloseMemHandle: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));

	return -FI_EINVAL;
}

/* TODO is this needed? */
static hipError_t hip_disabled_hipMemcpy(void *dst, const void *src,
					    size_t size, hipMemcpyKind kind)
{
	FI_WARN(&core_prov, FI_LOG_CORE,
		"hipMemcpy was called but FI_HMEM_HIP_ENABLE_XFER = 0, "
		"no copy will occur to prevent deadlock.");

	return hipErrorInvalidValue;
}

static int hip_hmem_dl_init(void)
{
#if ENABLE_hip_DLOPEN
	/* Assume failure to dlopen hip runtime is caused by the library not
	 * being found. Thus, hip is not supported.
	 */
	hip_handle = dlopen("libamdhip64.so", RTLD_NOW);
	if (!hip_handle) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"Failed to dlopen libamdhip64.so\n");
		return -FI_ENOSYS;
	}

	hip_ops.hipMemcpyAsync = dlsym(hip_handle, "hipMemcpyAsync");
	if (!hip_ops.hipMemcpyAsync) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipMemcpyAsync\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipMemcpy = dlsym(hip_handle, "hipMemcpy");
	if (!hip_ops.hipMemcpy) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipMemcpy\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipStreamCreate = dlsym(hip_handle, "hipStreamCreate");
	if (!hip_ops.hipStreamCreate) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipStreamCreate\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipStreamCreateWithFlags = dlsym(hip_handle,
											 "hipStreamCreateWithFlags");
	if (!hip_ops.hipStreamCreateWithFlags) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipStreamCreateWithFlags\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipStreamDestroy = dlsym(hip_handle, "hipStreamDestroy");
	if (!hip_ops.hipStreamDestroy) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipStreamDestroy\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipStreamQuery = dlsym(hip_handle, "hipStreamQuery");
	if (!hip_ops.hipStreamQuery) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipStreamQuery\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipStreamSynchronize = dlsym(hip_handle, "hipStreamSynchronize");
	if (!hip_ops.hipStreamSynchronize) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipStreamSynchronize\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipFree = dlsym(hip_handle, "hipFree");
	if (!hip_ops.hipFree) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipFree\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipMalloc = dlsym(hip_handle, "hipMalloc");
	if (!hip_ops.hipMalloc) {
		FI_WARN(&core_prov, FI_LOG_CORE, "Failed to find hipMalloc\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipGetErrorName = dlsym(hip_handle, "hipGetErrorName");
	if (!hip_ops.hipGetErrorName) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipGetErrorName\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipGetErrorString = dlsym(hip_handle,
					    "hipGetErrorString");
	if (!hip_ops.hipGetErrorString) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipGetErrorString\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipPointerGetAttributes = dlsym(hip_handle,
					       "hipPointerGetAttributes");
	if (!hip_ops.hipPointerGetAttributes) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipPointerGetAttributes\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipHostRegister = dlsym(hip_handle, "hipHostRegister");
	if (!hip_ops.hipHostRegister) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipHostRegister\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipHostUnregister = dlsym(hip_handle,
					    "hipHostUnregister");
	if (!hip_ops.hipHostUnregister) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipHostUnregister\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipGetDeviceCount = dlsym(hip_handle,
					    "hipGetDeviceCount");
	if (!hip_ops.hipGetDeviceCount) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipGetDeviceCount\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipGetDevice = dlsym(hip_handle,
					    "hipGetDevice");
	if (!hip_ops.hipGetDevice) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipGetDevice\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipSetDevice = dlsym(hip_handle,
					    "hipSetDevice");
	if (!hip_ops.hipSetDevice) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipSetDevice\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipIpcOpenMemHandle = dlsym(hip_handle,
					    "hipIpcOpenMemHandle");
	if (!hip_ops.hipIpcOpenMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipIpcOpenMemHandle\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipIpcGetMemHandle = dlsym(hip_handle,
					    "hipIpcGetMemHandle");
	if (!hip_ops.hipIpcGetMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipIpcGetMemHandle\n");
		goto err_dlclose_hip;
	}

	hip_ops.hipIpcCloseMemHandle = dlsym(hip_handle,
					    "hipIpcCloseMemHandle");
	if (!hip_ops.hipIpcCloseMemHandle) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to find hipIpcCloseMemHandle\n");
		goto err_dlclose_hip;
	}

	return FI_SUCCESS;

err_dlclose_hip:
	dlclose(hip_handle);

	return -FI_ENODATA;
#else
	return FI_SUCCESS;
#endif /* ENABLE_hip_DLOPEN */
}

static void hip_hmem_dl_cleanup(void)
{
#if ENABLE_HIP_DLOPEN
	dlclose(hip_handle);
#endif
}

static int hip_hmem_verify_devices(void)
{
	int device_count;
	hipError_t hip_ret;

	/* Verify hip compute-capable devices are present on the host. */
	hip_ret = ofi_hipGetDeviceCount(&device_count);
	switch (hip_ret) {
	case hipSuccess:
		break;

	case hipErrorNoDevice:
		return -FI_ENOSYS;

	default:
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hipGetDeviceCount: %s:%s\n",
			ofi_hipGetErrorName(hip_ret),
			ofi_hipGetErrorString(hip_ret));
		return -FI_EIO;
	}

	if (device_count == 0)
		return -FI_ENOSYS;

	return FI_SUCCESS;
}

int hip_hmem_init(void)
{
	int ret, i;
	bool hip_enable_xfer;

	fi_param_define(NULL, "hmem_hip_enable_xfer", FI_PARAM_BOOL,
			"Enable use of hip APIs for copying data to/from hip "
			"GPU memory. This should be disabled if hip "
			"operations on the default stream would result in a "
			"deadlock in the application. (default: true)");

	ret = hip_hmem_dl_init();
	if (ret != FI_SUCCESS)
		return ret;

	ret = hip_hmem_verify_devices();
	if (ret != FI_SUCCESS)
		goto dl_cleanup;

	/* TODO: Not sure if we need this for ROCM */
	ret = 1;
	fi_param_get_bool(NULL, "hmem_hip_enable_xfer", &ret);
	hip_enable_xfer = (ret != 0);

	if (!hip_enable_xfer)
		hip_ops.hipMemcpy = hip_disabled_hipMemcpy;

	/*
	 * hip IPC is only enabled if hipMemcpy can be used.
	 */
	hip_ipc_enabled = hip_enable_xfer;

	if (ofi_atomic_get32(&global_hip_stream_init) == 1)
		goto out;

	/* create HIP_NUM_STREAMS streams to be used for data transfer */
	for (i = 0; i < HMEM_NUM_STREAMS; i++) {
		hipError_t hip_ret = ofi_hipStreamCreateWithFlags(&global_hip_streams[i],
								hipStreamDefault); //hipStreamNonBlocking);
		if (hip_ret != hipSuccess) {
			ret = -FI_EINVAL;
			goto dl_cleanup;
		}
	}

	ofi_atomic_inc32(&global_hip_stream_init);

out:
	return FI_SUCCESS;

dl_cleanup:
	hip_hmem_dl_cleanup();

	return ret;
}

int hip_hmem_cleanup(void)
{
	int i;

	hip_hmem_dl_cleanup();
	for (i = 0; i < HMEM_NUM_STREAMS; i++) {
		hipError_t hip_ret = ofi_hipStreamDestroy(global_hip_streams[i]);
		if (hip_ret != hipSuccess) {
			FI_WARN(&core_prov, FI_LOG_CORE,
				"Failed to perform hipStreamDestroy: %s:%s\n",
				ofi_hipGetErrorName(hip_ret),
				ofi_hipGetErrorString(hip_ret));
		}
	}
	return FI_SUCCESS;
}

bool hip_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	hipPointerAttribute_t attr;
	hipError_t hip_ret;

	memset(&attr, 0, sizeof(attr));

	hip_ret = ofi_hipPointerGetAttributes(&attr, addr);
	if (hip_ret == hipSuccess) {
		if (attr.memoryType == hipMemoryTypeDevice) {
			if (flags)
				*flags = FI_HMEM_DEVICE_ONLY;

			if (device) {
				*device = attr.device;
			}
			return true;
		}
	}

	return false;
}

int hip_host_register(void *ptr, size_t size)
{
	hipError_t hip_ret;

	hip_ret = ofi_hipHostRegister(ptr, size, hipHostRegisterDefault);
	if (hip_ret == hipSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipHostRegister: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));

	return -FI_EIO;
}

int hip_host_unregister(void *ptr)
{
	hipError_t hip_ret;

	hip_ret = ofi_hipHostUnregister(ptr);
	if (hip_ret == hipSuccess)
		return FI_SUCCESS;

	FI_WARN(&core_prov, FI_LOG_CORE,
		"Failed to perform hipHostUnregister: %s:%s\n",
		ofi_hipGetErrorName(hip_ret),
		ofi_hipGetErrorString(hip_ret));

	return -FI_EIO;
}

bool hip_is_ipc_enabled(void)
{
	return !ofi_hmem_p2p_disabled() && hip_ipc_enabled;
}

#else

int hip_async_copy_from_dev(uint64_t device, void *dest, const void *src,
		       size_t size, void *istream, void **ostream)
{
	return -FI_ENOSYS;
}

int hip_async_copy_to_dev(uint64_t device, void *dest, const void *src,
		     size_t size, void *istream, void **ostream)
{
	return -FI_ENOSYS;
}

int hip_async_copy_query(void *stream)
{
	return -FI_ENOSYS;
}

int hip_copy_from_dev(uint64_t device, void *dest, const void *src,
		       size_t size)
{
	return -FI_ENOSYS;
}

int hip_copy_to_dev(uint64_t device, void *dest, const void *src,
		     size_t size)
{
	return -FI_ENOSYS;
}

int hip_hmem_init(void)
{
	return -FI_ENOSYS;
}

int hip_hmem_cleanup(void)
{
	return -FI_ENOSYS;
}

bool hip_is_addr_valid(const void *addr, uint64_t *device, uint64_t *flags)
{
	return false;
}

int hip_get_handle(void *dev_buf, size_t *len, void **handle, uint64_t *offset)
{
	return -FI_ENOSYS;
}

int hip_open_handle(void **handle, size_t len, uint64_t device, void **ipc_ptr)
{
	return -FI_ENOSYS;
}

int hip_close_handle(void *ipc_ptr)
{
	return -FI_ENOSYS;
}

bool hip_is_ipc_enabled(void)
{
	return false;
}

int hip_host_register(void *ptr, size_t size)
{
	return -FI_ENOSYS;
}

int hip_host_unregister(void *ptr)
{
	return -FI_ENOSYS;
}

int hip_stream_synchronize(int stream_id)
{
	return -FI_ENOSYS;
}

#endif /* HAVE_ROCR */
