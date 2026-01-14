/*
 * Copyright (C) 2026-2026 Cornelis Networks.
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
#ifndef _OPX_TRACER_CATEGORIES_H_
#define _OPX_TRACER_CATEGORIES_H_

#include <stdint.h>
#include <assert.h>

enum opx_trace_category {
	OPX_TRACE_CAT_TX       = (1 << 0),
	OPX_TRACE_CAT_RX       = (1 << 1),
	OPX_TRACE_CAT_RELI     = (1 << 2),
	OPX_TRACE_CAT_SDMA     = (1 << 3),
	OPX_TRACE_CAT_PIO      = (1 << 4),
	OPX_TRACE_CAT_CQ       = (1 << 5),
	OPX_TRACE_CAT_MR       = (1 << 6),
	OPX_TRACE_CAT_TID      = (1 << 7),
	OPX_TRACE_CAT_PROGRESS = (1 << 8),
	OPX_TRACE_CAT_HMEM     = (1 << 9),
	OPX_TRACE_CAT_ATOMIC   = (1 << 10),
	OPX_TRACE_CAT_RMA      = (1 << 11),
	OPX_TRACE_CAT_LOCK     = (1 << 12),
	OPX_TRACE_CAT_INTERNAL = (1 << 13),
	OPX_TRACE_CAT_LAST /* Sentinel: must be last, used for static_assert validation */
};

#define OPX_TRACE_NUM_CATEGORIES 14

/*
 * Ensure OPX_TRACE_NUM_CATEGORIES stays in sync with
 * the enum. OPX_TRACE_CAT_LAST is one more than OPX_TRACE_CAT_INTERNAL (1 << 13),
 * so it equals (1 << 13) + 1. The highest category bit is (1 << (NUM_CATEGORIES - 1)).
 */
static_assert((1 << (OPX_TRACE_NUM_CATEGORIES - 1)) == (OPX_TRACE_CAT_LAST - 1),
	      "OPX_TRACE_NUM_CATEGORIES is out of sync with enum opx_trace_category");

#define OPX_TRACE_CAT_ALL 0x3FFF

/*
 * Ensure OPX_TRACE_CAT_ALL has exactly
 * OPX_TRACE_NUM_CATEGORIES bits set (i.e., it covers all categories).
 * For N categories, the bitmask should be (1 << N) - 1.
 */
static_assert(OPX_TRACE_CAT_ALL == ((1 << OPX_TRACE_NUM_CATEGORIES) - 1),
	      "OPX_TRACE_CAT_ALL does not match OPX_TRACE_NUM_CATEGORIES");

enum opx_trace_filter_level {
	OPX_TRACE_FILTER_NONE	  = 0,
	OPX_TRACE_FILTER_COMPLETE = 1,
	OPX_TRACE_FILTER_ALL	  = 2,
};

static const char *opx_trace_category_names[] __attribute__((unused)) = {
	"TX", "RX", "RELI", "SDMA", "PIO", "CQ", "MR", "TID", "PROGRESS", "HMEM", "ATOMIC", "RMA", "LOCK", "INTERNAL",
};

static_assert(sizeof(opx_trace_category_names) / sizeof(opx_trace_category_names[0]) == OPX_TRACE_NUM_CATEGORIES,
	      "opx_trace_category_names array size does not match OPX_TRACE_NUM_CATEGORIES");

#endif /* _OPX_TRACER_CATEGORIES_H_ */
