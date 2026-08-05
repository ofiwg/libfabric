/*
 * Copyright (C) 2025-2026 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HFISVC_H_
#define _FI_PROV_OPX_HFISVC_H_

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "rdma/opx/fi_opx_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int opx_hfisvc_log_enabled;

#ifdef OPX_HFISVC_DEBUG
#define OPX_HFISVC_DEBUG_LOG(fmt, ...)                                                                  \
	do {                                                                                            \
		if (opx_hfisvc_log_enabled) {                                                           \
			fprintf(stderr, "(%d) %s:%s():%d " fmt, getpid(), __FILE__, __func__, __LINE__, \
				##__VA_ARGS__);                                                         \
		}                                                                                       \
	} while (0)
#else
#define OPX_HFISVC_DEBUG_LOG(fmt, ...)
#endif

/* reserved for a future phase; do not remove */
#define OPX_HFISVC_XFER_FLAG_FREE_ACCESS_KEY (1 << 0ul)

enum opx_hfisvc_xfer_type {
	OPX_HFISVC_XFER_TYPE_MR = 0,
	OPX_HFISVC_XFER_TYPE_RZV,
	OPX_HFISVC_XFER_TYPE_RMA_READ,
	OPX_HFISVC_XFER_TYPE_RMA_WRITE,
	OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH,
	OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH_COMPARE,
};

struct opx_hfisvc_xfer_completion {
	enum opx_hfisvc_xfer_type	  type;
	uint32_t			  access_key;
	size_t				  len;
	struct opx_context		 *context;
	struct fi_opx_completion_counter *cc;
	struct fi_opx_mr		 *opx_mr;
	struct fi_opx_ep		 *opx_ep;

	uint8_t	 flags;
	uint8_t	 unused_byte[7];
	uint64_t unused_qw[1];
} __attribute__((__aligned__(L2_CACHE_LINE_SIZE))) __attribute__((__packed__));

struct fi_opx_mr;

void opx_hfisvc_mr_report_completion_error(struct fi_opx_mr *opx_mr);

#ifdef __cplusplus
}
#endif

#endif
