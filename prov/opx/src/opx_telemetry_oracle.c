/*
 * Copyright (C) 2026 Cornelis Networks.
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

#include "rdma/opx/opx_telemetry_oracle.h"

#ifndef OPX_TELEMETRY_ORACLE

typedef int opx_telemetry_oracle_empty_translation_unit;

#else

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_hfi1_progress.h"

#define OPX_TELEMETRY_ORACLE_MAX_STATES		 8u
#define OPX_TELEMETRY_ORACLE_DEFAULT_INTERVAL	 64u
#define OPX_TELEMETRY_ORACLE_DEFAULT_MAX_SAMPLES (1u << 20)
#define OPX_TELEMETRY_ORACLE_RCVHDRTAIL_MASK	 ((1u << 19) - 1u)
#define OPX_TELEMETRY_ORACLE_FLAG_TAIL_STABLE	 (1u << 0)

#define OPX_TELEMETRY_ORACLE_DIR_ENV	     "_FI_OPX_TELEMETRY_ORACLE_DIR_"
#define OPX_TELEMETRY_ORACLE_INTERVAL_ENV    "_FI_OPX_TELEMETRY_ORACLE_INTERVAL_"
#define OPX_TELEMETRY_ORACLE_MAX_SAMPLES_ENV "_FI_OPX_TELEMETRY_ORACLE_MAX_SAMPLES_"

struct opx_telemetry_oracle_state {
	struct fi_opx_ep		   *opx_ep;
	struct fi_opx_hfi1_context	   *hfi;
	struct opx_telemetry_oracle_sample *samples;
	const char			   *output_dir;
	uint64_t			    interval;
	uint64_t			    decimation_remaining;
	uint64_t			    runtime_flags;
	uint64_t			    samples_dropped;
	uint32_t			    capacity;
	uint32_t			    samples_written;
	uint32_t			    elemcnt;
	uint32_t			    elemsz_dws;
	uint32_t			    elemlast_dws;
	uint32_t			    hfi1_type;
	uint32_t			    context_id;
};

/* Recorder ownership stays out of fi_opx_ep so oracle builds do not perturb any
 * endpoint or receive hot-structure layout. */
static _Atomic(struct opx_telemetry_oracle_state *) opx_telemetry_oracle_states[OPX_TELEMETRY_ORACLE_MAX_STATES];
static pthread_mutex_t				    opx_telemetry_oracle_table_lock  = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t				    opx_telemetry_oracle_atexit_once = PTHREAD_ONCE_INIT;
static _Atomic uint64_t				    opx_telemetry_oracle_table_overflow;

static void opx_telemetry_oracle_dump(void);

static void opx_telemetry_oracle_register_atexit(void)
{
	if (atexit(opx_telemetry_oracle_dump) != 0) {
		fprintf(stderr, "OPX telemetry oracle: unable to register process-exit dump\n");
	}
}

static uint64_t opx_telemetry_oracle_parse_u64(const char *name, const uint64_t default_value, const uint64_t maximum,
					       const bool allow_zero)
{
	const char	  *text = getenv(name);
	char		  *end;
	unsigned long long value;

	if (text == NULL || *text == '\0') {
		return default_value;
	}

	errno = 0;
	value = strtoull(text, &end, 0);
	if (errno != 0 || end == text || *end != '\0' || (!allow_zero && value == 0) || value > maximum) {
		return default_value;
	}

	return (uint64_t) value;
}

static struct opx_telemetry_oracle_state *opx_telemetry_oracle_find_state(struct fi_opx_ep *opx_ep)
{
	const size_t first = ((uintptr_t) opx_ep >> 6) & (OPX_TELEMETRY_ORACLE_MAX_STATES - 1u);

	for (size_t probe = 0; probe < OPX_TELEMETRY_ORACLE_MAX_STATES; ++probe) {
		const size_t			   index = (first + probe) & (OPX_TELEMETRY_ORACLE_MAX_STATES - 1u);
		struct opx_telemetry_oracle_state *state =
			atomic_load_explicit(&opx_telemetry_oracle_states[index], memory_order_acquire);

		if (state != NULL && state->opx_ep == opx_ep) {
			return state;
		}
	}

	return NULL;
}

static struct opx_telemetry_oracle_state *opx_telemetry_oracle_create_state(struct fi_opx_ep *opx_ep)
{
	const size_t			   first = ((uintptr_t) opx_ep >> 6) & (OPX_TELEMETRY_ORACLE_MAX_STATES - 1u);
	struct opx_telemetry_oracle_state *state;
	size_t				   empty_index = OPX_TELEMETRY_ORACLE_MAX_STATES;

	pthread_mutex_lock(&opx_telemetry_oracle_table_lock);

	state = opx_telemetry_oracle_find_state(opx_ep);
	if (state != NULL) {
		pthread_mutex_unlock(&opx_telemetry_oracle_table_lock);
		return state;
	}

	for (size_t probe = 0; probe < OPX_TELEMETRY_ORACLE_MAX_STATES; ++probe) {
		const size_t index = (first + probe) & (OPX_TELEMETRY_ORACLE_MAX_STATES - 1u);
		if (atomic_load_explicit(&opx_telemetry_oracle_states[index], memory_order_relaxed) == NULL) {
			empty_index = index;
			break;
		}
	}
	if (empty_index == OPX_TELEMETRY_ORACLE_MAX_STATES) {
		atomic_fetch_add_explicit(&opx_telemetry_oracle_table_overflow, 1u, memory_order_relaxed);
		pthread_mutex_unlock(&opx_telemetry_oracle_table_lock);
		return NULL;
	}

	state = calloc(1, sizeof(*state));
	if (state == NULL) {
		pthread_mutex_unlock(&opx_telemetry_oracle_table_lock);
		return NULL;
	}

	state->opx_ep		    = opx_ep;
	state->hfi		    = opx_ep->hfi;
	state->interval		    = opx_telemetry_oracle_parse_u64(OPX_TELEMETRY_ORACLE_INTERVAL_ENV,
								     OPX_TELEMETRY_ORACLE_DEFAULT_INTERVAL, UINT64_MAX, false);
	state->decimation_remaining = state->interval;
	const uint64_t max_capacity =
		SIZE_MAX / sizeof(*state->samples) < UINT32_MAX ? SIZE_MAX / sizeof(*state->samples) : UINT32_MAX;
	state->capacity = (uint32_t) opx_telemetry_oracle_parse_u64(
		OPX_TELEMETRY_ORACLE_MAX_SAMPLES_ENV, OPX_TELEMETRY_ORACLE_DEFAULT_MAX_SAMPLES, max_capacity, true);

	const char *output_dir = getenv(OPX_TELEMETRY_ORACLE_DIR_ENV);
	if (output_dir == NULL || *output_dir == '\0') {
		output_dir = ".";
	}
	state->output_dir = strdup(output_dir);
	if (state->output_dir == NULL) {
		state->output_dir = ".";
	}

	if (state->capacity != 0) {
		state->samples = malloc((size_t) state->capacity * sizeof(*state->samples));
		if (state->samples == NULL) {
			state->capacity = 0;
		}
	}

	state->elemcnt	     = opx_ep->hfi->info.rxe.hdrq.elemcnt;
	state->elemsz_dws    = opx_ep->hfi->info.rxe.hdrq.elemsz;
	state->elemlast_dws  = opx_ep->hfi->info.rxe.hdrq.elemlast;
	state->hfi1_type     = (uint32_t) opx_ep->hfi->hfi1_type;
	state->runtime_flags = opx_ep->hfi->runtime_flags;
	state->context_id    = opx_ep->hfi->info.rxe.id;

	pthread_once(&opx_telemetry_oracle_atexit_once, opx_telemetry_oracle_register_atexit);
	atomic_store_explicit(&opx_telemetry_oracle_states[empty_index], state, memory_order_release);
	pthread_mutex_unlock(&opx_telemetry_oracle_table_lock);
	return state;
}

static void opx_telemetry_oracle_drop_sample(struct opx_telemetry_oracle_state *state)
{
	++state->samples_dropped;
}

static uint32_t opx_telemetry_oracle_read_tail(const struct opx_telemetry_oracle_state *state)
{
	/* RcvHdrTail is a 19-bit DW offset even though its CSR mapping is 64-bit. */
	const uint64_t tail = *state->hfi->ctrl->__hfi_rcvhdrtail;
	return (uint32_t) tail & OPX_TELEMETRY_ORACLE_RCVHDRTAIL_MASK;
}

void opx_telemetry_oracle_sample(struct fi_opx_ep *opx_ep, const uint64_t hdrq_mask, const enum opx_hfi1_type hfi1_type)
{
	struct opx_telemetry_oracle_state *state = opx_telemetry_oracle_find_state(opx_ep);
	if (state == NULL) {
		state = opx_telemetry_oracle_create_state(opx_ep);
		if (state == NULL) {
			return;
		}
	}

	if (--state->decimation_remaining != 0) {
		return;
	}
	state->decimation_remaining = state->interval;

	if (state->samples_written == state->capacity) {
		opx_telemetry_oracle_drop_sample(state);
		return;
	}

	const uint32_t tail1   = opx_telemetry_oracle_read_tail(state);
	const uint32_t backlog = opx_hfi1_sample_rx_backlog(opx_ep, hdrq_mask, hfi1_type);
	const uint32_t tail2   = opx_telemetry_oracle_read_tail(state);
	const uint64_t tsc     = fi_opx_timer_get_cycles();

	struct opx_telemetry_oracle_sample *sample = &state->samples[state->samples_written];
	sample->tsc				   = tsc;
	sample->sw_head_dws			   = opx_ep->rx->state.hdrq.head;
	sample->rhf_seq				   = opx_ep->rx->state.hdrq.rhf_seq;
	sample->tail1_dws			   = tail1;
	sample->tail2_dws			   = tail2;
	sample->backlog_entries			   = backlog;
	/* A stable tail brackets the RHF scan with one coherent producer position. */
	sample->flags = tail1 == tail2 ? OPX_TELEMETRY_ORACLE_FLAG_TAIL_STABLE : 0u;
	++state->samples_written;
}

static void opx_telemetry_oracle_dump_state(const struct opx_telemetry_oracle_state *state, const char *hostname,
					    const pid_t pid)
{
	char	  filename[PATH_MAX];
	const int length = snprintf(filename, sizeof(filename), "%s/opx-oracle-%s-%ld-ctxt%u.bin", state->output_dir,
				    hostname, (long) pid, state->context_id);
	if (length < 0 || (size_t) length >= sizeof(filename)) {
		errno = ENAMETOOLONG;
		perror("OPX telemetry oracle output filename");
		return;
	}

	FILE *output = fopen(filename, "wb");
	if (output == NULL) {
		perror(filename);
		return;
	}

	const struct opx_telemetry_oracle_file_header header = {
		.magic		 = OPX_TELEMETRY_ORACLE_MAGIC,
		.version	 = OPX_TELEMETRY_ORACLE_VERSION,
		.record_bytes	 = (uint32_t) sizeof(struct opx_telemetry_oracle_sample),
		.elemcnt	 = state->elemcnt,
		.elemsz_dws	 = state->elemsz_dws,
		.elemlast_dws	 = state->elemlast_dws,
		.hfi1_type	 = state->hfi1_type,
		.runtime_flags	 = state->runtime_flags,
		.samples_written = state->samples_written,
		.samples_dropped = state->samples_dropped > UINT32_MAX ? UINT32_MAX : (uint32_t) state->samples_dropped,
		.reserved	 = {0, 0},
	};

	if (fwrite(&header, sizeof(header), 1, output) != 1) {
		if (ferror(output)) {
			perror(filename);
		} else {
			fprintf(stderr, "OPX telemetry oracle: short write to %s\n", filename);
		}
		fclose(output);
		return;
	}
	if (state->samples_written != 0 &&
	    fwrite(state->samples, sizeof(*state->samples), state->samples_written, output) != state->samples_written) {
		if (ferror(output)) {
			perror(filename);
		} else {
			fprintf(stderr, "OPX telemetry oracle: short write to %s\n", filename);
		}
		fclose(output);
		return;
	}
	if (fclose(output) != 0) {
		perror(filename);
	}
}

static void opx_telemetry_oracle_dump(void)
{
	char hostname[HOST_NAME_MAX + 1];
	if (gethostname(hostname, HOST_NAME_MAX) != 0) {
		memcpy(hostname, "unknown", sizeof("unknown"));
	} else {
		hostname[HOST_NAME_MAX] = '\0';
	}

	const pid_t pid = getpid();
	for (size_t index = 0; index < OPX_TELEMETRY_ORACLE_MAX_STATES; ++index) {
		const struct opx_telemetry_oracle_state *state =
			atomic_load_explicit(&opx_telemetry_oracle_states[index], memory_order_acquire);
		if (state != NULL) {
			opx_telemetry_oracle_dump_state(state, hostname, pid);
		}
	}
}

#endif /* OPX_TELEMETRY_ORACLE */
