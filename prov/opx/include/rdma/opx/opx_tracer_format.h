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
#ifndef _OPX_TRACER_FORMAT_H_
#define _OPX_TRACER_FORMAT_H_

#include <stdint.h>

#define OPX_TRACE_MAGIC	       0x4F505854 /* "OPXT" */
#define OPX_TRACE_VERSION      1	  /* Development version - format not yet finalized */
#define OPX_TRACE_HOSTNAME_LEN 64

/*
 * Timer modes for timestamp collection
 *
 * OPX_TRACE_TIMER_TSC: Uses CPU cycle counter via RDTSC instruction
 *   - Lowest overhead (~4-10ns per event)
 *   - Requires calibration to convert cycles to nanoseconds
 *   - tsc_frequency field contains cycles per second
 *   - Best for bare-metal systems with constant TSC
 *
 * OPX_TRACE_TIMER_GETTIME: Uses clock_gettime(CLOCK_MONOTONIC)
 *   - Higher overhead (~20-50ns per event)
 *   - Timestamps are already in nanoseconds
 *   - tsc_frequency field is set to 1,000,000,000 (1 ns = 1 tick)
 *   - Better for VMs or systems with CPU frequency scaling
 *
 * Build-time selection via configure:
 *   --enable-opx-tracer=TX,RX              # Default timer (TSC)
 *   --enable-opx-tracer=TX,RX:tsc          # Force TSC mode
 *   --enable-opx-tracer=TX,RX:gettime      # Force clock_gettime mode
 *
 * Note: These are defined as macros (not just enum values) so they can be
 * used in preprocessor #if directives without -Wundef warnings.
 */
#define OPX_TRACE_TIMER_TSC	0 /* CPU cycle counter (RDTSC) */
#define OPX_TRACE_TIMER_GETTIME 1 /* clock_gettime(CLOCK_MONOTONIC) */

enum opx_trace_timer_mode {
	OPX_TRACE_TIMER_MODE_TSC     = OPX_TRACE_TIMER_TSC,
	OPX_TRACE_TIMER_MODE_GETTIME = OPX_TRACE_TIMER_GETTIME,
};

enum opx_trace_record_type {
	OPX_TRACE_RECORD_EVENT	    = 0,
	OPX_TRACE_RECORD_STRING_DEF = 1,
};

enum opx_trace_status {
	OPX_TRACE_STATUS_BEGIN	     = 0,
	OPX_TRACE_STATUS_END_SUCCESS = 1,
	OPX_TRACE_STATUS_END_EAGAIN  = 2,
	OPX_TRACE_STATUS_END_ERROR   = 3,
	OPX_TRACE_STATUS_INSTANT     = 4,
	OPX_TRACE_STATUS_END_ENOBUFS = 5,
	OPX_TRACE_STATUS_END_IGNORED = 6,
};

/*
 * Record header - 8 bytes packed into single 64-bit word
 * Layout:
 *   bits [0:3]   - record_type (enum opx_trace_record_type)
 *   bits [4:7]   - status (enum opx_trace_status)
 *   bits [8:23]  - category bitmask (16 bits for categories)
 *   bits [24:39] - event_id (compile-time event ID)
 *   bits [40:43] - num_args (0-15)
 *   bits [44:63] - reserved
 */
struct opx_trace_header {
	uint64_t word0;
};

#define OPX_TRACE_HDR_TYPE_SHIFT     0
#define OPX_TRACE_HDR_TYPE_MASK	     0xFULL
#define OPX_TRACE_HDR_STATUS_SHIFT   4
#define OPX_TRACE_HDR_STATUS_MASK    0xFULL
#define OPX_TRACE_HDR_CATEGORY_SHIFT 8
#define OPX_TRACE_HDR_CATEGORY_MASK  0xFFFFULL
#define OPX_TRACE_HDR_EVENT_SHIFT    24
#define OPX_TRACE_HDR_EVENT_MASK     0xFFFFULL
#define OPX_TRACE_HDR_NARGS_SHIFT    40
#define OPX_TRACE_HDR_NARGS_MASK     0xFULL

static inline uint64_t opx_trace_make_header(enum opx_trace_record_type type, enum opx_trace_status status,
					     uint16_t category, uint16_t event_id, uint8_t num_args)
{
	return ((uint64_t) type << OPX_TRACE_HDR_TYPE_SHIFT) | ((uint64_t) status << OPX_TRACE_HDR_STATUS_SHIFT) |
	       ((uint64_t) category << OPX_TRACE_HDR_CATEGORY_SHIFT) |
	       ((uint64_t) event_id << OPX_TRACE_HDR_EVENT_SHIFT) | ((uint64_t) num_args << OPX_TRACE_HDR_NARGS_SHIFT);
}

static inline enum opx_trace_record_type opx_trace_header_type(uint64_t hdr)
{
	return (enum opx_trace_record_type)((hdr >> OPX_TRACE_HDR_TYPE_SHIFT) & OPX_TRACE_HDR_TYPE_MASK);
}

static inline enum opx_trace_status opx_trace_header_status(uint64_t hdr)
{
	return (enum opx_trace_status)((hdr >> OPX_TRACE_HDR_STATUS_SHIFT) & OPX_TRACE_HDR_STATUS_MASK);
}

static inline uint16_t opx_trace_header_category(uint64_t hdr)
{
	return (uint16_t) ((hdr >> OPX_TRACE_HDR_CATEGORY_SHIFT) & OPX_TRACE_HDR_CATEGORY_MASK);
}

static inline uint16_t opx_trace_header_event_id(uint64_t hdr)
{
	return (uint16_t) ((hdr >> OPX_TRACE_HDR_EVENT_SHIFT) & OPX_TRACE_HDR_EVENT_MASK);
}

static inline uint8_t opx_trace_header_num_args(uint64_t hdr)
{
	return (uint8_t) ((hdr >> OPX_TRACE_HDR_NARGS_SHIFT) & OPX_TRACE_HDR_NARGS_MASK);
}

/*
 * Event record - 32 bytes (cache-line friendly)
 * Contains header, timestamp, and up to 2 inline arguments
 */
struct opx_trace_event {
	uint64_t hdr;	    /* 8 bytes - packed header */
	uint64_t timestamp; /* 8 bytes - rdtsc value */
	uint64_t args[2];   /* 16 bytes - inline arguments */
};

/*
 * String definition record - variable length
 * Written once per event ID at trace file start
 */
struct opx_trace_string_def {
	uint64_t hdr;	  /* 8 bytes - record_type=STRING_DEF */
	uint16_t id;	  /* Event ID */
	uint16_t len;	  /* String length (excluding null) */
	uint32_t padding; /* Alignment padding */
			  /* char str[] follows - null-terminated, padded to 8-byte boundary */
};

/*
 * File header - written at start of each trace file
 * Contains metadata for parsing and clock correlation
 *
 * Clock fields:
 *   timer_mode        - Timer mode used (OPX_TRACE_TIMER_TSC or OPX_TRACE_TIMER_GETTIME)
 *   tsc_frequency     - TSC ticks per second (or 1e9 for gettime mode)
 *   start_time_ns     - CLOCK_MONOTONIC at start (for precise relative timing)
 *   start_realtime_ns - CLOCK_REALTIME at start (for cross-host alignment via NTP)
 *   start_tsc         - TSC/timestamp counter at start (for converting event timestamps)
 *
 * In TSC mode:
 *   - tsc_frequency is calibrated cycles per second
 *   - start_tsc is the RDTSC value at trace start
 *   - Event timestamps are RDTSC values, converted via: ns = (tsc - start_tsc) * 1e9 / tsc_frequency
 *
 * In gettime mode:
 *   - tsc_frequency is 1,000,000,000 (nanoseconds per second)
 *   - start_tsc is the CLOCK_MONOTONIC nanosecond value at trace start
 *   - Event timestamps are already nanoseconds, converted via: ns = timestamp - start_tsc
 */
struct opx_trace_file_header {
	uint32_t magic;				   /* OPX_TRACE_MAGIC */
	uint32_t version;			   /* OPX_TRACE_VERSION */
	uint32_t pid;				   /* Process ID */
	uint32_t tid;				   /* Thread ID */
	uint64_t tsc_frequency;			   /* Ticks per second (1e9 for gettime mode) */
	uint64_t start_time_ns;			   /* CLOCK_MONOTONIC at start */
	uint64_t start_realtime_ns;		   /* CLOCK_REALTIME at start (NTP-synced) */
	uint64_t start_tsc;			   /* TSC/timestamp at start */
	char	 hostname[OPX_TRACE_HOSTNAME_LEN]; /* Hostname */
	uint32_t enabled_categories;		   /* Bitmask of compiled categories */
	uint32_t num_event_strings;		   /* Number of string defs following */
	uint32_t timer_mode;			   /* enum opx_trace_timer_mode */
	uint32_t self_lid;			   /* Endpoint's LID for TX flow correlation */
};

#define OPX_TRACE_EVENT_SIZE sizeof(struct opx_trace_event)

#endif /* _OPX_TRACER_FORMAT_H_ */
