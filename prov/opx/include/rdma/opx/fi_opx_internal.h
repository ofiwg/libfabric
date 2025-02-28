/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2024 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_INTERNAL_H_
#define _FI_PROV_OPX_INTERNAL_H_

#include <config.h>

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "rdma/opx/opx_tracer.h"

#define FI_OPX_CACHE_LINE_SIZE (64)

#define FI_OPX_CQ_CONTEXT_HMEM	    (0x8000000000000000ull)
#define FI_OPX_CQ_CONTEXT_MULTIRECV (0x4000000000000000ull)

#define OPX_HMEM_SIZE_QWS (3)

union fi_opx_mp_egr_id {
	uint64_t id;

	struct {
		uint32_t uid;
		uint32_t slid_origin_tx; // first 3 bytes is slid and last byte is tx
	};
} __attribute__((__packed__));

struct opx_context {
	/**** CACHELINE 0 ****/
	struct opx_context *next;  /* fi_cq_entry::op_context */
	uint64_t	    flags; /* fi_cq_msg_entry::flags */
	size_t		    len;   /* fi_cq_msg_entry::len */
	void		   *buf; /* fi_cq_data_entry::buf (unused for tagged cq's and non-multi-receive message cq's) */

	union {
		uint64_t  data;	    /* fi_cq_data_entry::data; only used _after_ a message is matched */
		fi_addr_t src_addr; /* only used _before_ a message is matched ('FI_DIRECTED_RECEIVE') */
	};

	uint64_t tag; /* fi_cq_tagged_entry::tag */
	union {
		uint64_t	       ignore;		   /* only for tagged receive */
		void		      *claim;		   /* only for peek/claim */
		void		      *multi_recv_context; /* only for individual FI_MULTI_RECV's */
		union fi_opx_mp_egr_id mp_egr_id;
	};

	volatile uint64_t byte_counter;

	/**** CACHELINE 1 & 2 ****/
	uint64_t hmem_info_qws[OPX_HMEM_SIZE_QWS];

	struct {
		size_t	      iov_count;
		struct iovec *iov;
	} msg;

	struct fi_cq_err_entry err_entry; // 88 bytes
} __attribute__((__packed__)) __attribute__((__aligned__(64)));
static_assert(offsetof(struct opx_context, hmem_info_qws) == FI_OPX_CACHE_LINE_SIZE,
	      "struct opx_context.hmem_info_qws offset should start at Cacheline 1!");
static_assert(sizeof(struct opx_context) == (FI_OPX_CACHE_LINE_SIZE * 3),
	      "sizeof(struct opx_context) should be equal to 3 cachelines!");

struct opx_sdma_queue {
	struct slist list;
	uint16_t     num_reqs;
	uint16_t     num_iovs;
	uint16_t     max_iovs;
	uint16_t     slots_avail;
};

#ifndef MAX
#define MAX(a, b) ((a) ^ (((a) ^ (b)) & -((a) < (b))))
#endif
#ifndef MIN
#define MIN(a, b) ((b) ^ (((a) ^ (b)) & -((a) < (b))))
#endif
#ifndef MIN3
#define MIN3(a, b, c) (MIN(MIN(a, b), c))
#endif
#ifndef MIN4
#define MIN4(a, b, c, d) (MIN(MIN(a, b), MIN(c, d)))
#endif

#define FI_OPX_LOCK_REQUIRED	 1
#define FI_OPX_LOCK_NOT_REQUIRED 0

static inline int fi_opx_threading_unknown(const enum fi_threading threading)
{
	return threading != FI_THREAD_DOMAIN && // Most likely
	       threading != FI_THREAD_SAFE && threading != FI_THREAD_COMPLETION &&
	       threading != FI_THREAD_UNSPEC; // Least likely
}

static inline int fi_opx_threading_lock_required(const enum fi_threading threading, enum fi_progress progress)
{
	return !(threading == FI_THREAD_DOMAIN || threading == FI_THREAD_COMPLETION) || progress == FI_PROGRESS_AUTO;
}

static inline void fi_opx_lock_if_required(ofi_spin_t *lock, const int required)
{
	if (required) {
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_BEGIN, "LOCK");
		ofi_spin_lock(lock);
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_END_SUCCESS, "LOCK");
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_BEGIN, "LOCK-HELD");
	}
}

static inline void fi_opx_lock(ofi_spin_t *lock)
{
	ofi_spin_lock(lock);
}

static inline void fi_opx_unlock_if_required(ofi_spin_t *lock, const int required)
{
	if (required) {
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_END_SUCCESS, "LOCK-HELD");
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_BEGIN, "UNLOCK");
		ofi_spin_unlock(lock);
		OPX_TRACER_TRACE_LOCK_IF_REQUIRED(OPX_TRACER_END_SUCCESS, "UNLOCK");
	}
}

static inline void fi_opx_unlock(ofi_spin_t *lock)
{
	ofi_spin_unlock(lock);
}

#endif /* _FI_PROV_OPX_INTERNAL_H_ */
