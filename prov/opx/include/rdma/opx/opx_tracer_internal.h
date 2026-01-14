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
#ifndef _OPX_TRACER_INTERNAL_H_
#define _OPX_TRACER_INTERNAL_H_

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <time.h>

#include "rdma/opx/opx_tracer_format.h"
#include "rdma/opx/opx_tracer_categories.h"
#include "rdma/opx/opx_tracer_events.h"

#define OPX_TRACE_DEFAULT_BUFFER_SIZE (16 * 1024 * 1024) /* 16MB */
#define OPX_TRACE_MAX_PATH_LEN	      256

/*
 * Timer mode selection at build time:
 *
 * OPX_TRACER_TIMER_TSC     - Force TSC mode (RDTSC instruction)
 * OPX_TRACER_TIMER_GETTIME - Force clock_gettime mode
 * Neither defined          - Default to TSC mode on x86_64
 *
 * Configure option syntax:
 *   --enable-opx-tracer=CATEGORIES:MODE
 *   where MODE is 'tsc', 'gettime', or omitted for default (TSC)
 *
 * Examples:
 *   --enable-opx-tracer=TX,RX           # Default timer (TSC on x86_64)
 *   --enable-opx-tracer=TX,RX:tsc       # Force TSC
 *   --enable-opx-tracer=TX,RX:gettime   # Force clock_gettime
 *   --enable-opx-tracer=all:gettime     # All categories, clock_gettime
 *
 * TSC mode characteristics:
 *   - Uses RDTSC instruction for ~4-10ns overhead per event
 *   - Requires calibration at startup to determine frequency
 *   - Best for bare-metal systems with constant TSC
 *
 * Gettime mode characteristics:
 *   - Uses clock_gettime(CLOCK_MONOTONIC) for ~20-50ns overhead
 *   - No calibration needed - timestamps are already nanoseconds
 *   - Better for VMs or systems with CPU frequency scaling
 */

/*
 * OPX_TRACE_TIMER_TSC and OPX_TRACE_TIMER_GETTIME are defined as macros
 * in opx_tracer_format.h (included above) so they can be used in
 * preprocessor #if directives without -Wundef warnings.
 */

/* Determine effective timer mode */
#if defined(OPX_TRACER_TIMER_GETTIME)
/* User explicitly requested clock_gettime mode */
#define OPX_TRACE_TIMER_MODE OPX_TRACE_TIMER_GETTIME
#elif defined(OPX_TRACER_TIMER_TSC)
/* User explicitly requested TSC mode */
#define OPX_TRACE_TIMER_MODE OPX_TRACE_TIMER_TSC
#else
/* Default to TSC mode on x86_64 */
#define OPX_TRACE_TIMER_MODE OPX_TRACE_TIMER_TSC
#endif

struct opx_trace_thread_buffer {
	uint8_t *buffer;
	size_t	 buffer_size;
	size_t	 write_offset;
	int	 output_fd;
	uint32_t tid;
	uint32_t flush_count;
	uint64_t blocked_ns;
	bool	 header_written;
};

struct opx_trace_global {
	uint64_t tsc_frequency;
	uint64_t start_time_ns;	    /* CLOCK_MONOTONIC at init (relative timing) */
	uint64_t start_realtime_ns; /* CLOCK_REALTIME at init (cross-host alignment) */
	uint64_t start_tsc;
	char	 hostname[OPX_TRACE_HOSTNAME_LEN];
	char	 output_path[OPX_TRACE_MAX_PATH_LEN];
	size_t	 buffer_size;
	uint32_t enabled_categories;
	uint8_t	 runtime_filters[OPX_TRACE_NUM_CATEGORIES];
	bool	 initialized;
	uint32_t pid;
	uint32_t self_lid; /* Endpoint's LID for TX flow correlation */
};

extern struct opx_trace_global			opx_trace_global;
extern __thread struct opx_trace_thread_buffer *opx_trace_tls_buffer;

void				opx_trace_global_init(void);
void				opx_trace_global_fini(void);
struct opx_trace_thread_buffer *opx_trace_init_thread_buffer(void);
void				opx_trace_fini_thread_buffer(struct opx_trace_thread_buffer *buf);
void				opx_trace_flush_buffer(struct opx_trace_thread_buffer *buf);
void opx_trace_emit_event(uint16_t category, enum opx_trace_status status, enum opx_trace_event_id event_id,
			  uint64_t arg0, uint64_t arg1);
void opx_trace_set_self_lid(uint32_t lid);

/*
 * Get current timestamp value
 *
 * In TSC mode: Returns CPU cycle counter via RDTSC instruction
 * In gettime mode: Returns nanoseconds from CLOCK_MONOTONIC
 *
 * The timer mode is selected at build time via OPX_TRACE_TIMER_MODE
 */
static inline uint64_t opx_trace_rdtsc(void)
{
#if OPX_TRACE_TIMER_MODE == OPX_TRACE_TIMER_GETTIME
	/* clock_gettime mode - returns nanoseconds directly */
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
#else
	/* TSC mode - RDTSC instruction (x86_64) */
	uint32_t lo, hi;
	__asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
	return ((uint64_t) hi << 32) | lo;
#endif
}

static inline struct opx_trace_thread_buffer *opx_trace_get_buffer(void)
{
	if (__builtin_expect(!opx_trace_tls_buffer, 0)) {
		opx_trace_tls_buffer = opx_trace_init_thread_buffer();
	}
	return opx_trace_tls_buffer;
}

static inline void opx_trace_ensure_space(struct opx_trace_thread_buffer *buf, size_t needed)
{
	if (__builtin_expect(buf->write_offset + needed > buf->buffer_size, 0)) {
		opx_trace_flush_buffer(buf);
	}
}

static inline void opx_trace_write_event(struct opx_trace_thread_buffer *buf, uint16_t category,
					 enum opx_trace_status status, enum opx_trace_event_id event_id, uint64_t arg0,
					 uint64_t arg1)
{
	opx_trace_ensure_space(buf, OPX_TRACE_EVENT_SIZE);

	struct opx_trace_event *ev = (struct opx_trace_event *) (buf->buffer + buf->write_offset);

	ev->hdr	      = opx_trace_make_header(OPX_TRACE_RECORD_EVENT, status, category, event_id, 2);
	ev->timestamp = opx_trace_rdtsc();
	ev->args[0]   = arg0;
	ev->args[1]   = arg1;

	buf->write_offset += OPX_TRACE_EVENT_SIZE;
}

#endif /* _OPX_TRACER_INTERNAL_H_ */
