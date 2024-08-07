/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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
#include "hmem_util.h"

void hmem_init(void)
{
#ifdef USE_CUDA
	hmem_cuda_init();
#endif
#ifdef USE_ROCM
	hmem_rocm_init();
#endif
}

void hmem_cleanup(void)
{
#ifdef USE_CUDA
	hmem_cuda_cleanup();
#endif
#ifdef USE_ROCM
	hmem_rocm_cleanup();
#endif
}

int hmem_memcpy_d2h(enum fi_hmem_iface iface, void* dst, void* src, size_t len)
{
	switch (iface) {
	case FI_HMEM_CUDA:
		return hmem_cuda_memcpy_d2h(dst, src, len);
		break;
	case FI_HMEM_ROCR:
		return hmem_rocm_memcpy_d2h(dst, src, len);
		break;
	default:
		error("Unsupported hmem iface requested: %d", iface);
		return -EOPNOTSUPP;
		break;
	}

	return 0;
}

int hmem_memcpy_h2d(enum fi_hmem_iface iface, void* dst, void* src, size_t len)
{
	switch (iface) {
	case FI_HMEM_CUDA:
		return hmem_cuda_memcpy_h2d(dst, src, len);
		break;
	case FI_HMEM_ROCR:
		return hmem_rocm_memcpy_h2d(dst, src, len);
		break;
	default:
		error("Unsupported hmem iface requested: %d", iface);
		return -EOPNOTSUPP;
		break;
	}

	return 0;
}

int hmem_alloc(enum fi_hmem_iface iface, void *uaddr, size_t len)
{
	switch (iface) {
	case  FI_HMEM_CUDA:
		return hmem_cuda_alloc(uaddr, len);
		break;
	case  FI_HMEM_ROCR:
		return hmem_rocm_alloc(uaddr, len);
		break;
	default:
		error("hmem iface %d not supported", iface);
		return -ENOSYS;
		break;
	}
}

void hmem_free(enum fi_hmem_iface iface, void *uaddr)
{
	switch (iface) {
	case FI_HMEM_CUDA:
		hmem_cuda_free(uaddr);
		break;
	case FI_HMEM_ROCR:
		hmem_rocm_free(uaddr);
		break;
	default:
		error("Unsupported hmem iface requested: %d", iface);
		break;
	}
}
