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

/* enum fi_datatype / enum fi_op used by-value in struct opx_hfisvc_atomic_op */
#include <rdma/fi_domain.h>

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

/* single source of truth for the atomic operand staging cap: pool element size == recv_rts len guard.
 * Sized to 8 MiB so atomic-write/fetch engage hfisvc across the same 8 KiB..4 MiB range as RMA read/write;
 * compare-fetch packs [swap|compare] contiguously so an N-byte operand needs 2*N of staging (4 MiB -> 8 MiB). */
#define OPX_HFISVC_ATOMIC_STAGING_MAX (1ul << 23)

/* bounded-retry cap: past this a stuck iov is dropped-and-drained */
#define OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS (16384ul)

enum opx_hfisvc_xfer_type {
	OPX_HFISVC_XFER_TYPE_MR = 0,
	OPX_HFISVC_XFER_TYPE_RZV,
	OPX_HFISVC_XFER_TYPE_RMA_READ,
	OPX_HFISVC_XFER_TYPE_RMA_WRITE,
	OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH,
	OPX_HFISVC_XFER_TYPE_ATOMIC_WRITE,
	OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH_COMPARE,
};

struct opx_hfisvc_atomic_op;

struct opx_hfisvc_xfer_completion {
	enum opx_hfisvc_xfer_type	  type;
	uint32_t			  access_key;
	size_t				  len;
	struct opx_context		 *context;
	struct fi_opx_completion_counter *cc;
	struct fi_opx_mr		 *opx_mr;
	struct fi_opx_ep		 *opx_ep;

	uint8_t flags;
	uint8_t unused_byte[7];
	/* atomic fetch/compare-fetch carry the target-side apply state here across staged completions */
	struct opx_hfisvc_atomic_op *atomic_op;
} __attribute__((__aligned__(L2_CACHE_LINE_SIZE))) __attribute__((__packed__));

/* target-side atomic apply state: operand staged via rdma_read_va, applied on completion */
struct opx_hfisvc_atomic_op {
	uint8_t				 *staging;
	void				 *target_vaddr;
	uint64_t			  apply_len;
	struct fi_opx_mr		 *opx_mr;
	struct fi_opx_completion_counter *cc;
	struct opx_context		 *context;
	enum fi_datatype		  dt;
	enum fi_op			  op;
	/* fetch step-A (operand pull) parameters, stashed by recv_rts for the poller to issue */
	uint32_t op_client_key;
	uint32_t op_access_key;
	uint64_t op_origin_offset;
	uint32_t origin_lid;
	uint32_t op_len;
	/* compare-fetch second operand pull; staged after the swap operand at staging[op_len] */
	uint32_t op2_client_key;
	uint32_t op2_access_key;
	uint64_t op2_origin_offset;
	uint32_t op2_len;
	uint8_t	 pending;
};

struct fi_opx_mr;

void opx_hfisvc_mr_report_completion_error(struct fi_opx_mr *opx_mr);

union fi_opx_hfi1_deferred_work;
struct fi_opx_ep;

int  opx_hfisvc_atomic_stage_work(union fi_opx_hfi1_deferred_work *work);
void opx_hfisvc_atomic_poll_advance(struct fi_opx_ep *opx_ep, struct opx_hfisvc_xfer_completion *completion,
				    struct opx_context *context, int ctx_idx);

#ifdef __cplusplus
}
#endif

#endif
