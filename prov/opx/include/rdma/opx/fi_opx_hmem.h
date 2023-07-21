/*
 * Copyright (C) 2023 by Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HMEM_H_
#define _FI_PROV_OPX_HMEM_H_

#include <assert.h>
#include "rdma/opx/fi_opx_compiler.h"
//#include "rdma/opx/fi_opx_domain.h"
#include "ofi_hmem.h"

struct fi_opx_hmem_info {
	uint64_t			device;
	enum fi_hmem_iface		iface;
	uint32_t			unused;
} __attribute__((__packed__)) __attribute__((aligned(8)));

OPX_COMPILE_TIME_ASSERT((sizeof(struct fi_opx_hmem_info) & 0x7) == 0,
			"sizeof(fi_opx_hmem_info) should be a multiple of 8");

__OPX_FORCE_INLINE__
enum fi_hmem_iface fi_opx_hmem_get_iface(const void *ptr,
					 const struct fi_opx_mr *desc,
					 uint64_t *device)
{
#ifdef OPX_HMEM
	if (desc) {
		switch (desc->attr.iface) {
			case FI_HMEM_CUDA:
				*device = desc->attr.device.cuda;
				break;
			case FI_HMEM_ZE:
				*device = desc->attr.device.ze;
				break;
			default:
				*device = 0;
		}
		return desc->attr.iface;
	}

	return ofi_get_hmem_iface(ptr, device, NULL);
#else
	*device = 0;
	return FI_HMEM_SYSTEM;
#endif
}

#endif
