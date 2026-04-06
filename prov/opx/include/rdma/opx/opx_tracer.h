/*
 * Copyright (C) 2024-2026 Cornelis Networks.
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
#ifndef _OPX_TRACER_H_
#define _OPX_TRACER_H_

#ifdef OPX_TRACER_ENABLED

#include "rdma/opx/opx_tracer_internal.h"
#include <rdma/fi_errno.h>

/*
 * Helper macro to extract SLID from packet header for flow event correlation.
 * Returns opx_lid_t from either 9B or 16B LRH format based on hfi1_type.
 */
#define OPX_TRACE_GET_SLID(hdr, hfi1_type)                                  \
	(((hfi1_type) & (OPX_HFI1_WFR | OPX_HFI1_MIXED_9B)) ?               \
		 (opx_lid_t) __be16_to_cpu24((__be16) (hdr)->lrh_9B.slid) : \
		 (opx_lid_t) __le24_to_cpu(((hdr)->lrh_16B.slid20 << 20) | (hdr)->lrh_16B.slid))

/*
 * Helper macro to pack SLID and length into a single 64-bit value for tracing.
 * Format: (slid << 32) | (len & 0xFFFFFFFF)
 */
#define OPX_TRACE_PACK_SLID_LEN(slid, len) (((uint64_t) (slid) << 32) | ((len) & 0xFFFFFFFFULL))

#define OPX_TRACER_INIT() opx_trace_global_init()
#define OPX_TRACER_EXIT() opx_trace_global_fini()

/*
 * OPX_TRACE_ENABLED - Check if tracing is enabled for a category
 *
 * The (cat) != 0 check prevents undefined behavior from __builtin_ctz(0),
 * which is undefined on some architectures. Categories should always be
 * non-zero power-of-2 values, but this provides defensive protection.
 */
#define OPX_TRACE_ENABLED(cat)                                                                          \
	((cat) != 0 && opx_trace_global.initialized && (opx_trace_global.enabled_categories & (cat)) && \
	 opx_trace_global.runtime_filters[__builtin_ctz(cat)] != OPX_TRACE_FILTER_NONE)

#define OPX_TRACE_BEGIN(cat, event_id, arg0, arg1)                                                              \
	do {                                                                                                    \
		if (OPX_TRACE_ENABLED(cat)) {                                                                   \
			struct opx_trace_thread_buffer *_buf = opx_trace_get_buffer();                          \
			opx_trace_write_event(_buf, (cat), OPX_TRACE_STATUS_BEGIN, (event_id), (arg0), (arg1)); \
		}                                                                                               \
	} while (0)

#define OPX_TRACE_END(cat, event_id, status, arg0, arg1)                                          \
	do {                                                                                      \
		if (OPX_TRACE_ENABLED(cat)) {                                                     \
			struct opx_trace_thread_buffer *_buf = opx_trace_get_buffer();            \
			opx_trace_write_event(_buf, (cat), (status), (event_id), (arg0), (arg1)); \
		}                                                                                 \
	} while (0)

#define OPX_TRACE_INSTANT(cat, event_id, arg0, arg1)                                                              \
	do {                                                                                                      \
		if (OPX_TRACE_ENABLED(cat)) {                                                                     \
			struct opx_trace_thread_buffer *_buf = opx_trace_get_buffer();                            \
			opx_trace_write_event(_buf, (cat), OPX_TRACE_STATUS_INSTANT, (event_id), (arg0), (arg1)); \
		}                                                                                                 \
	} while (0)

#define OPX_TRACE_END_SUCCESS(cat, event_id, arg0, arg1) \
	OPX_TRACE_END(cat, event_id, OPX_TRACE_STATUS_END_SUCCESS, arg0, arg1)

#define OPX_TRACE_END_EAGAIN(cat, event_id, arg0, arg1) \
	OPX_TRACE_END(cat, event_id, OPX_TRACE_STATUS_END_EAGAIN, arg0, arg1)

#define OPX_TRACE_END_ERROR(cat, event_id, arg0, arg1) \
	OPX_TRACE_END(cat, event_id, OPX_TRACE_STATUS_END_ERROR, arg0, arg1)

#define OPX_TRACE_END_ENOBUFS(cat, event_id, arg0, arg1) \
	OPX_TRACE_END(cat, event_id, OPX_TRACE_STATUS_END_ENOBUFS, arg0, arg1)

#define OPX_TRACE_END_IGNORED(cat, event_id, arg0, arg1) \
	OPX_TRACE_END(cat, event_id, OPX_TRACE_STATUS_END_IGNORED, arg0, arg1)

#define OPX_TRACE_END_RC(cat, event_id, rc, arg0, arg1)                   \
	do {                                                              \
		if ((rc) == FI_SUCCESS || (rc) == 0)                      \
			OPX_TRACE_END_SUCCESS(cat, event_id, arg0, arg1); \
		else if ((rc) == -FI_EAGAIN)                              \
			OPX_TRACE_END_EAGAIN(cat, event_id, arg0, arg1);  \
		else if ((rc) == -FI_ENOBUFS)                             \
			OPX_TRACE_END_ENOBUFS(cat, event_id, arg0, arg1); \
		else                                                      \
			OPX_TRACE_END_ERROR(cat, event_id, arg0, arg1);   \
	} while (0)

#define OPX_TRACE_BEGIN_COND(cond, cat, event_id, arg0, arg1)       \
	do {                                                        \
		if (cond) {                                         \
			OPX_TRACE_BEGIN(cat, event_id, arg0, arg1); \
		}                                                   \
	} while (0)

#define OPX_TRACE_END_COND(cond, cat, event_id, status, arg0, arg1)       \
	do {                                                              \
		if (cond) {                                               \
			OPX_TRACE_END(cat, event_id, status, arg0, arg1); \
		}                                                         \
	} while (0)

#define OPX_TRACE_INSTANT_COND(cond, cat, event_id, arg0, arg1)       \
	do {                                                          \
		if (cond) {                                           \
			OPX_TRACE_INSTANT(cat, event_id, arg0, arg1); \
		}                                                     \
	} while (0)

/* TX subsystem macros */
#ifdef OPX_TRACER_TX
#define OPX_TRACE_TX_BEGIN(event_id, arg0, arg1)       OPX_TRACE_BEGIN(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_TX, event_id, status, arg0, arg1)
#define OPX_TRACE_TX_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_END_EAGAIN(event_id, arg0, arg1)  OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ERROR(event_id, arg0, arg1)   OPX_TRACE_END_ERROR(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ENOBUFS(event_id, arg0, arg1) OPX_TRACE_END_ENOBUFS(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_END_RC(event_id, rc, arg0, arg1)  OPX_TRACE_END_RC(OPX_TRACE_CAT_TX, event_id, rc, arg0, arg1)
#define OPX_TRACE_TX_INSTANT(event_id, arg0, arg1)     OPX_TRACE_INSTANT(OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#define OPX_TRACE_TX_INSTANT_COND(cond, event_id, arg0, arg1) \
	OPX_TRACE_INSTANT_COND(cond, OPX_TRACE_CAT_TX, event_id, arg0, arg1)
#else
#define OPX_TRACE_TX_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_TX_END(event_id, status, arg0, arg1)
#define OPX_TRACE_TX_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ENOBUFS(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_RC(event_id, rc, arg0, arg1)
#define OPX_TRACE_TX_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_TX_INSTANT_COND(cond, event_id, arg0, arg1)
#endif

/* RX subsystem macros */
#ifdef OPX_TRACER_RX
#define OPX_TRACE_RX_BEGIN(event_id, arg0, arg1)       OPX_TRACE_BEGIN(OPX_TRACE_CAT_RX, event_id, arg0, arg1)
#define OPX_TRACE_RX_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_RX, event_id, status, arg0, arg1)
#define OPX_TRACE_RX_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_RX, event_id, arg0, arg1)
#define OPX_TRACE_RX_END_EAGAIN(event_id, arg0, arg1)  OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_RX, event_id, arg0, arg1)
#define OPX_TRACE_RX_END_ERROR(event_id, arg0, arg1)   OPX_TRACE_END_ERROR(OPX_TRACE_CAT_RX, event_id, arg0, arg1)
#define OPX_TRACE_RX_INSTANT(event_id, arg0, arg1)     OPX_TRACE_INSTANT(OPX_TRACE_CAT_RX, event_id, arg0, arg1)
#else
#define OPX_TRACE_RX_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RX_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RX_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RX_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RX_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RX_INSTANT(event_id, arg0, arg1)
#endif

/* RELI subsystem macros */
#ifdef OPX_TRACER_RELI
#define OPX_TRACE_RELI_BEGIN(event_id, arg0, arg1)	 OPX_TRACE_BEGIN(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_RELI, event_id, status, arg0, arg1)
#define OPX_TRACE_RELI_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_EAGAIN(event_id, arg0, arg1)	 OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_ERROR(event_id, arg0, arg1)	 OPX_TRACE_END_ERROR(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT(event_id, arg0, arg1)	 OPX_TRACE_INSTANT(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_IGNORED(event_id, arg0, arg1) OPX_TRACE_END_IGNORED(OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT_COND(cond, event_id, arg0, arg1) \
	OPX_TRACE_INSTANT_COND(cond, OPX_TRACE_CAT_RELI, event_id, arg0, arg1)
#else
#define OPX_TRACE_RELI_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RELI_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_IGNORED(event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT_COND(cond, event_id, arg0, arg1)
#endif

/* SDMA subsystem macros */
#ifdef OPX_TRACER_SDMA
#define OPX_TRACE_SDMA_BEGIN(event_id, arg0, arg1)	 OPX_TRACE_BEGIN(OPX_TRACE_CAT_SDMA, event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_SDMA, event_id, status, arg0, arg1)
#define OPX_TRACE_SDMA_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_SDMA, event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_EAGAIN(event_id, arg0, arg1)	 OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_SDMA, event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_ERROR(event_id, arg0, arg1)	 OPX_TRACE_END_ERROR(OPX_TRACE_CAT_SDMA, event_id, arg0, arg1)
#define OPX_TRACE_SDMA_INSTANT(event_id, arg0, arg1)	 OPX_TRACE_INSTANT(OPX_TRACE_CAT_SDMA, event_id, arg0, arg1)
#else
#define OPX_TRACE_SDMA_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END(event_id, status, arg0, arg1)
#define OPX_TRACE_SDMA_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_INSTANT(event_id, arg0, arg1)
#endif

/* PIO subsystem macros */
#ifdef OPX_TRACER_PIO
#define OPX_TRACE_PIO_BEGIN(event_id, arg0, arg1)	OPX_TRACE_BEGIN(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#define OPX_TRACE_PIO_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_PIO, event_id, status, arg0, arg1)
#define OPX_TRACE_PIO_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_EAGAIN(event_id, arg0, arg1)	OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ERROR(event_id, arg0, arg1)	OPX_TRACE_END_ERROR(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ENOBUFS(event_id, arg0, arg1) OPX_TRACE_END_ENOBUFS(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#define OPX_TRACE_PIO_INSTANT(event_id, arg0, arg1)	OPX_TRACE_INSTANT(OPX_TRACE_CAT_PIO, event_id, arg0, arg1)
#else
#define OPX_TRACE_PIO_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END(event_id, status, arg0, arg1)
#define OPX_TRACE_PIO_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ENOBUFS(event_id, arg0, arg1)
#define OPX_TRACE_PIO_INSTANT(event_id, arg0, arg1)
#endif

/* CQ subsystem macros */
#ifdef OPX_TRACER_CQ
#define OPX_TRACE_CQ_BEGIN(event_id, arg0, arg1)       OPX_TRACE_BEGIN(OPX_TRACE_CAT_CQ, event_id, arg0, arg1)
#define OPX_TRACE_CQ_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_CQ, event_id, status, arg0, arg1)
#define OPX_TRACE_CQ_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_CQ, event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_EAGAIN(event_id, arg0, arg1)  OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_CQ, event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_ERROR(event_id, arg0, arg1)   OPX_TRACE_END_ERROR(OPX_TRACE_CAT_CQ, event_id, arg0, arg1)
#define OPX_TRACE_CQ_INSTANT(event_id, arg0, arg1)     OPX_TRACE_INSTANT(OPX_TRACE_CAT_CQ, event_id, arg0, arg1)
#else
#define OPX_TRACE_CQ_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END(event_id, status, arg0, arg1)
#define OPX_TRACE_CQ_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_CQ_INSTANT(event_id, arg0, arg1)
#endif

/* MR subsystem macros */
#ifdef OPX_TRACER_MR
#define OPX_TRACE_MR_BEGIN(event_id, arg0, arg1)       OPX_TRACE_BEGIN(OPX_TRACE_CAT_MR, event_id, arg0, arg1)
#define OPX_TRACE_MR_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_MR, event_id, status, arg0, arg1)
#define OPX_TRACE_MR_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_MR, event_id, arg0, arg1)
#define OPX_TRACE_MR_END_EAGAIN(event_id, arg0, arg1)  OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_MR, event_id, arg0, arg1)
#define OPX_TRACE_MR_END_ERROR(event_id, arg0, arg1)   OPX_TRACE_END_ERROR(OPX_TRACE_CAT_MR, event_id, arg0, arg1)
#define OPX_TRACE_MR_INSTANT(event_id, arg0, arg1)     OPX_TRACE_INSTANT(OPX_TRACE_CAT_MR, event_id, arg0, arg1)
#else
#define OPX_TRACE_MR_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_MR_END(event_id, status, arg0, arg1)
#define OPX_TRACE_MR_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_MR_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_MR_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_MR_INSTANT(event_id, arg0, arg1)
#endif

/* TID subsystem macros */
#ifdef OPX_TRACER_TID
#define OPX_TRACE_TID_BEGIN(event_id, arg0, arg1)	OPX_TRACE_BEGIN(OPX_TRACE_CAT_TID, event_id, arg0, arg1)
#define OPX_TRACE_TID_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_TID, event_id, status, arg0, arg1)
#define OPX_TRACE_TID_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_TID, event_id, arg0, arg1)
#define OPX_TRACE_TID_END_EAGAIN(event_id, arg0, arg1)	OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_TID, event_id, arg0, arg1)
#define OPX_TRACE_TID_END_ERROR(event_id, arg0, arg1)	OPX_TRACE_END_ERROR(OPX_TRACE_CAT_TID, event_id, arg0, arg1)
#define OPX_TRACE_TID_INSTANT(event_id, arg0, arg1)	OPX_TRACE_INSTANT(OPX_TRACE_CAT_TID, event_id, arg0, arg1)
#else
#define OPX_TRACE_TID_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_TID_END(event_id, status, arg0, arg1)
#define OPX_TRACE_TID_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_TID_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_TID_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_TID_INSTANT(event_id, arg0, arg1)
#endif

/* PROGRESS subsystem macros */
#ifdef OPX_TRACER_PROGRESS
#define OPX_TRACE_PROGRESS_BEGIN(event_id, arg0, arg1) OPX_TRACE_BEGIN(OPX_TRACE_CAT_PROGRESS, event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END(event_id, status, arg0, arg1) \
	OPX_TRACE_END(OPX_TRACE_CAT_PROGRESS, event_id, status, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_SUCCESS(event_id, arg0, arg1) \
	OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_PROGRESS, event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_EAGAIN(event_id, arg0, arg1) \
	OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_PROGRESS, event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_ERROR(event_id, arg0, arg1) \
	OPX_TRACE_END_ERROR(OPX_TRACE_CAT_PROGRESS, event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_INSTANT(event_id, arg0, arg1) OPX_TRACE_INSTANT(OPX_TRACE_CAT_PROGRESS, event_id, arg0, arg1)
#else
#define OPX_TRACE_PROGRESS_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END(event_id, status, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_INSTANT(event_id, arg0, arg1)
#endif

/* HMEM subsystem macros */
#ifdef OPX_TRACER_HMEM
#define OPX_TRACE_HMEM_BEGIN(event_id, arg0, arg1)	 OPX_TRACE_BEGIN(OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_HMEM, event_id, status, arg0, arg1)
#define OPX_TRACE_HMEM_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_EAGAIN(event_id, arg0, arg1)	 OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_ERROR(event_id, arg0, arg1)	 OPX_TRACE_END_ERROR(OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT(event_id, arg0, arg1)	 OPX_TRACE_INSTANT(OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT_COND(cond, event_id, arg0, arg1) \
	OPX_TRACE_INSTANT_COND(cond, OPX_TRACE_CAT_HMEM, event_id, arg0, arg1)
#else
#define OPX_TRACE_HMEM_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END(event_id, status, arg0, arg1)
#define OPX_TRACE_HMEM_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT_COND(cond, event_id, arg0, arg1)
#endif

/* ATOMIC subsystem macros */
#ifdef OPX_TRACER_ATOMIC
#define OPX_TRACE_ATOMIC_BEGIN(event_id, arg0, arg1) OPX_TRACE_BEGIN(OPX_TRACE_CAT_ATOMIC, event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END(event_id, status, arg0, arg1) \
	OPX_TRACE_END(OPX_TRACE_CAT_ATOMIC, event_id, status, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_SUCCESS(event_id, arg0, arg1) \
	OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_ATOMIC, event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_EAGAIN(event_id, arg0, arg1) \
	OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_ATOMIC, event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_ERROR(event_id, arg0, arg1) OPX_TRACE_END_ERROR(OPX_TRACE_CAT_ATOMIC, event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_INSTANT(event_id, arg0, arg1)	 OPX_TRACE_INSTANT(OPX_TRACE_CAT_ATOMIC, event_id, arg0, arg1)
#else
#define OPX_TRACE_ATOMIC_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END(event_id, status, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_INSTANT(event_id, arg0, arg1)
#endif

/* RMA subsystem macros */
#ifdef OPX_TRACER_RMA
#define OPX_TRACE_RMA_BEGIN(event_id, arg0, arg1)	OPX_TRACE_BEGIN(OPX_TRACE_CAT_RMA, event_id, arg0, arg1)
#define OPX_TRACE_RMA_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_RMA, event_id, status, arg0, arg1)
#define OPX_TRACE_RMA_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_RMA, event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_EAGAIN(event_id, arg0, arg1)	OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_RMA, event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_ERROR(event_id, arg0, arg1)	OPX_TRACE_END_ERROR(OPX_TRACE_CAT_RMA, event_id, arg0, arg1)
#define OPX_TRACE_RMA_INSTANT(event_id, arg0, arg1)	OPX_TRACE_INSTANT(OPX_TRACE_CAT_RMA, event_id, arg0, arg1)
#else
#define OPX_TRACE_RMA_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RMA_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RMA_INSTANT(event_id, arg0, arg1)
#endif

/* LOCK subsystem macros */
#ifdef OPX_TRACER_LOCK
#define OPX_TRACE_LOCK_BEGIN(event_id, arg0, arg1)	 OPX_TRACE_BEGIN(OPX_TRACE_CAT_LOCK, event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END(event_id, status, arg0, arg1) OPX_TRACE_END(OPX_TRACE_CAT_LOCK, event_id, status, arg0, arg1)
#define OPX_TRACE_LOCK_END_SUCCESS(event_id, arg0, arg1) OPX_TRACE_END_SUCCESS(OPX_TRACE_CAT_LOCK, event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_EAGAIN(event_id, arg0, arg1)	 OPX_TRACE_END_EAGAIN(OPX_TRACE_CAT_LOCK, event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_ERROR(event_id, arg0, arg1)	 OPX_TRACE_END_ERROR(OPX_TRACE_CAT_LOCK, event_id, arg0, arg1)
#define OPX_TRACE_LOCK_INSTANT(event_id, arg0, arg1)	 OPX_TRACE_INSTANT(OPX_TRACE_CAT_LOCK, event_id, arg0, arg1)
#else
#define OPX_TRACE_LOCK_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END(event_id, status, arg0, arg1)
#define OPX_TRACE_LOCK_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_INSTANT(event_id, arg0, arg1)
#endif

#else /* !OPX_TRACER_ENABLED */

#define OPX_TRACER_INIT()
#define OPX_TRACER_EXIT()

#define OPX_TRACE_ENABLED(cat) (0)
#define OPX_TRACE_BEGIN(cat, event_id, arg0, arg1)
#define OPX_TRACE_END(cat, event_id, status, arg0, arg1)
#define OPX_TRACE_INSTANT(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_SUCCESS(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_EAGAIN(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_ERROR(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_ENOBUFS(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_IGNORED(cat, event_id, arg0, arg1)
#define OPX_TRACE_END_RC(cat, event_id, rc, arg0, arg1)
#define OPX_TRACE_BEGIN_COND(cond, cat, event_id, arg0, arg1)
#define OPX_TRACE_END_COND(cond, cat, event_id, status, arg0, arg1)
#define OPX_TRACE_INSTANT_COND(cond, cat, event_id, arg0, arg1)

#define OPX_TRACE_TX_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_TX_END(event_id, status, arg0, arg1)
#define OPX_TRACE_TX_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_ENOBUFS(event_id, arg0, arg1)
#define OPX_TRACE_TX_END_RC(event_id, rc, arg0, arg1)
#define OPX_TRACE_TX_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_TX_INSTANT_COND(cond, event_id, arg0, arg1)

#define OPX_TRACE_RX_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RX_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RX_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RX_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RX_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RX_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_RELI_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RELI_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_RELI_END_IGNORED(event_id, arg0, arg1)
#define OPX_TRACE_RELI_INSTANT_COND(cond, event_id, arg0, arg1)

#define OPX_TRACE_SDMA_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END(event_id, status, arg0, arg1)
#define OPX_TRACE_SDMA_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_SDMA_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_PIO_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END(event_id, status, arg0, arg1)
#define OPX_TRACE_PIO_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_PIO_END_ENOBUFS(event_id, arg0, arg1)
#define OPX_TRACE_PIO_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_CQ_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END(event_id, status, arg0, arg1)
#define OPX_TRACE_CQ_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_CQ_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_CQ_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_MR_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_MR_END(event_id, status, arg0, arg1)
#define OPX_TRACE_MR_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_MR_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_MR_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_MR_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_TID_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_TID_END(event_id, status, arg0, arg1)
#define OPX_TRACE_TID_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_TID_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_TID_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_TID_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_PROGRESS_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END(event_id, status, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_PROGRESS_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_HMEM_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END(event_id, status, arg0, arg1)
#define OPX_TRACE_HMEM_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT(event_id, arg0, arg1)
#define OPX_TRACE_HMEM_INSTANT_COND(cond, event_id, arg0, arg1)

#define OPX_TRACE_ATOMIC_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END(event_id, status, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_ATOMIC_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_RMA_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END(event_id, status, arg0, arg1)
#define OPX_TRACE_RMA_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_RMA_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_RMA_INSTANT(event_id, arg0, arg1)

#define OPX_TRACE_LOCK_BEGIN(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END(event_id, status, arg0, arg1)
#define OPX_TRACE_LOCK_END_SUCCESS(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_EAGAIN(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_END_ERROR(event_id, arg0, arg1)
#define OPX_TRACE_LOCK_INSTANT(event_id, arg0, arg1)

#endif /* OPX_TRACER_ENABLED */

#endif /* _OPX_TRACER_H_ */
