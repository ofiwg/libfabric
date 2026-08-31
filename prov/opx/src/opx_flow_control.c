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

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "rdma/opx/fi_opx.h"

#include <rdma/fi_errno.h>
#include <rdma/providers/fi_log.h>
#include <rdma/providers/fi_prov.h>

#include "rdma/opx/opx_flow_control.h"
#include "rdma/opx/opx_flow_control_rccc_private.h"

enum opx_flow_control_process_state {
	OPX_FLOW_CONTROL_PROCESS_UNPUBLISHED = 0,
	OPX_FLOW_CONTROL_PROCESS_PUBLISHED,
	OPX_FLOW_CONTROL_PROCESS_TERMINAL
};

static pthread_mutex_t			       opx_flow_control_process_lock = PTHREAD_MUTEX_INITIALIZER;
static enum opx_flow_control_process_state     opx_flow_control_process_state;
static enum opx_flow_control_mode	       opx_flow_control_process_mode	   = OPX_FLOW_CONTROL_MODE_NONE;
static size_t				       opx_flow_control_process_state_size = OPX_FLOW_CONTROL_NONE_STATE_SIZE;
static int				       opx_flow_control_process_result;
static uint64_t				       opx_flow_control_process_references;
static struct opx_flow_control_rccc_tx_profile opx_flow_control_rccc_profile;
static struct opx_flow_control_rccc_scheduler_profile opx_flow_control_rccc_scheduler_profile;
static uint16_t					      opx_flow_control_rccc_control_headroom;
static uint16_t					      opx_flow_control_rccc_max_control_emits;
static bool					      opx_flow_control_rccc_initialized;

static int __attribute__((cold)) opx_flow_control_rccc_parse_u64(const char *name, uint64_t default_value,
								 uint64_t *value)
{
	const char	  *text = getenv(name);
	char		  *end;
	unsigned long long parsed;

	if (text == NULL) {
		*value = default_value;
		return FI_SUCCESS;
	}
	errno  = 0;
	parsed = strtoull(text, &end, 0);
	if (errno != 0 || end == text || *end != '\0') {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Invalid %s value '%s': not a valid unsigned integer\n",
			name, text);
		return -FI_EINVAL;
	}
	*value = (uint64_t) parsed;
	return FI_SUCCESS;
}

/* unused until the selection enablement change wires the caller */
static int __attribute__((cold, unused)) opx_flow_control_rccc_profile_configure(void)
{
	struct opx_flow_control_rccc_tx_profile profile = {
		.ffirst = 256u,
		/* Measured: fmax 4096 restores sustained unidirectional
		 * bandwidth at 256 KB-4 MB to no-flow-control parity; 1024
		 * (MAX_BATCH_REQUEST) was the binding window ceiling. */
		.fmax			     = 4096u,
		.idle_retire_delay_ns	     = UINT64_C(1000000),
		.request_retry_initial_us    = OPX_FLOW_CONTROL_RCCC_RETRY_DEFAULT_US,
		.retirement_retry_initial_us = OPX_FLOW_CONTROL_RCCC_RETRY_DEFAULT_US,
		.reset_retry_initial_us	     = OPX_FLOW_CONTROL_RCCC_RETRY_DEFAULT_US,
	};
	uint64_t value;

#define OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE(name, field, limit)                                                       \
	do {                                                                                                          \
		if (opx_flow_control_rccc_parse_u64(name, profile.field, &value) != FI_SUCCESS) {                     \
			return -FI_EINVAL;                                                                            \
		}                                                                                                     \
		if (value > (limit)) {                                                                                \
			FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "%s value %lu exceeds maximum %lu\n", name, value, \
				(uint64_t) (limit));                                                                  \
			return -FI_EINVAL;                                                                            \
		}                                                                                                     \
		profile.field = value;                                                                                \
	} while (0)
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_FFIRST_", ffirst, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_FMAX_", fmax, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_IDLE_RETIRE_NS_", idle_retire_delay_ns,
					    UINT64_MAX);
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_REQUEST_RETRY_INITIAL_US_",
					    request_retry_initial_us, UINT16_MAX);
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_RETIREMENT_RETRY_INITIAL_US_",
					    retirement_retry_initial_us, UINT16_MAX);
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_RESET_RETRY_INITIAL_US_", reset_retry_initial_us,
					    UINT16_MAX);
#undef OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE
	if (opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_CONTROL_HEADROOM_", 8u, &value) != FI_SUCCESS) {
		return -FI_EINVAL;
	}
	if (value > UINT16_MAX) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"_FI_OPX_FLOW_CONTROL_RCCC_CONTROL_HEADROOM_ value %lu exceeds maximum %u\n", value,
			UINT16_MAX);
		return -FI_EINVAL;
	}
	opx_flow_control_rccc_control_headroom = (uint16_t) value;
	if (opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_MAX_CONTROL_EMITS_", 8u, &value) != FI_SUCCESS) {
		return -FI_EINVAL;
	}
	if (value == 0u || value > UINT16_MAX) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"_FI_OPX_FLOW_CONTROL_RCCC_MAX_CONTROL_EMITS_ value %lu outside valid range [1, %u]\n", value,
			UINT16_MAX);
		return -FI_EINVAL;
	}
	opx_flow_control_rccc_max_control_emits = (uint16_t) value;
	if (!opx_flow_control_rccc_tx_profile_is_valid(&profile) || profile.ffirst == 0u ||
	    profile.fmax < OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"Invalid RCCC TX profile: ffirst=%u fmax=%u (ffirst must be nonzero and <= fmax; fmax must be >= %u)\n",
			profile.ffirst, profile.fmax, OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST);
		return -FI_EINVAL;
	}

	/* Scheduler geometry.  Every value is validated and invalid geometry
	 * fails RCCC initialization closed.  The occupancy input is synthetic
	 * full-rate (no hysteretic occupancy curve yet), so the accrual rate
	 * constant only needs to keep the grant tier from starving the
	 * per-poll bulk budget. */
	struct opx_flow_control_rccc_scheduler_profile scheduler_profile = {
		.max_outstanding_permissions = UINT64_C(65536),
		/* Measured: quantum 1024 halves the large bidirectional
		 * bandwidth gap at 2-4 MB; 256 quantized a hot flow's window
		 * fill across multiple grant-pass visits. */
		.drr_quantum			= UINT64_C(1024),
		.quarantine_deadline_ns		= UINT64_C(100000000),
		.minimum_packet_rate_q16_per_ns = UINT64_C(65536),
		.maximum_elapsed_ns		= UINT64_C(1000000),
		.tracked_capacity		= 64u,
		.decision_capacity		= 1024u,
		.reject_capacity		= 1024u,
	};
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE(name, field, limit)                                                 \
	do {                                                                                                      \
		if (opx_flow_control_rccc_parse_u64(name, scheduler_profile.field, &value) != FI_SUCCESS) {       \
			return -FI_EINVAL;                                                                        \
		}                                                                                                 \
		if (value == 0u || value > (limit)) {                                                             \
			FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "%s value %lu outside valid range [1, %lu]\n", \
				name, value, (uint64_t) (limit));                                                 \
			return -FI_EINVAL;                                                                        \
		}                                                                                                 \
		scheduler_profile.field = value;                                                                  \
	} while (0)
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_MAX_OUTSTANDING_", max_outstanding_permissions,
					      (uint64_t) OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE - 1u);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DRR_QUANTUM_", drr_quantum,
					      (uint64_t) OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE - 1u);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_QUARANTINE_NS_", quarantine_deadline_ns,
					      UINT64_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_TRACKED_", tracked_capacity, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DECISIONS_", decision_capacity, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_REJECTS_", reject_capacity, UINT32_MAX);
#undef OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE
	if (scheduler_profile.decision_capacity < profile.ffirst ||
	    scheduler_profile.reject_capacity < profile.ffirst ||
	    scheduler_profile.drr_quantum > scheduler_profile.max_outstanding_permissions) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"Invalid RCCC scheduler geometry: decisions=%u rejects=%u (each must be >= ffirst %u), quantum=%lu (must be <= max outstanding %lu)\n",
			scheduler_profile.decision_capacity, scheduler_profile.reject_capacity, profile.ffirst,
			scheduler_profile.drr_quantum, scheduler_profile.max_outstanding_permissions);
		return -FI_EINVAL;
	}
	opx_flow_control_rccc_scheduler_profile = scheduler_profile;
	opx_flow_control_rccc_profile		= profile;
	opx_flow_control_rccc_initialized	= true;
	return FI_SUCCESS;
}

const struct opx_flow_control_rccc_tx_profile *opx_flow_control_rccc_tx_profile_get(void)
{
	return opx_flow_control_rccc_initialized ? &opx_flow_control_rccc_profile : NULL;
}

const struct opx_flow_control_rccc_scheduler_profile *opx_flow_control_rccc_scheduler_profile_get(void)
{
	return opx_flow_control_rccc_initialized ? &opx_flow_control_rccc_scheduler_profile : NULL;
}

uint16_t opx_flow_control_rccc_control_headroom_get(void)
{
	return opx_flow_control_rccc_initialized ? opx_flow_control_rccc_control_headroom : 0u;
}

uint16_t opx_flow_control_rccc_max_control_emits_get(void)
{
	return opx_flow_control_rccc_initialized ? opx_flow_control_rccc_max_control_emits : 0u;
}

static int opx_flow_control_initialize_none(void)
{
	return FI_SUCCESS;
}

static int opx_flow_control_initialize_mode(enum opx_flow_control_mode mode, size_t *state_size)
{
	int ret;

	switch (mode) {
	case OPX_FLOW_CONTROL_MODE_NONE:
		ret = opx_flow_control_initialize_none();
		if (ret == FI_SUCCESS) {
			*state_size = OPX_FLOW_CONTROL_NONE_STATE_SIZE;
		}
		return ret;
	case OPX_FLOW_CONTROL_MODE_RCCC:
		/* The RCCC engine is compiled but not yet wired into the send
		 * and receive paths; selection fails closed until the final
		 * enablement change flips this to the profile configure. */
		return -FI_EOPNOTSUPP;
	case OPX_FLOW_CONTROL_MODE_CONFORMANCE:
		return -FI_EOPNOTSUPP;
	default:
		return -FI_EINVAL;
	}
}

static int opx_flow_control_select_mode(enum opx_flow_control_mode *mode, bool *retryable)
{
	char *value = NULL;
	int   ret;

	*retryable = false;
	ret	   = fi_param_get_str(fi_opx_global.prov, "flow_control", &value);
	if (ret == -FI_ENODATA) {
		*mode = OPX_FLOW_CONTROL_MODE_NONE;
		return FI_SUCCESS;
	}
	if (ret != FI_SUCCESS) {
		*retryable = true;
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Unable to read FI_OPX_FLOW_CONTROL: %s (%d)\n",
			fi_strerror(-ret), ret);
		return ret;
	}

	if (value != NULL) {
		if (strcmp(value, "none") == 0) {
			*mode = OPX_FLOW_CONTROL_MODE_NONE;
			return FI_SUCCESS;
		}
		if (strcmp(value, "rccc") == 0) {
			*mode = OPX_FLOW_CONTROL_MODE_RCCC;
			return FI_SUCCESS;
		}
		if (strcmp(value, "conformance") == 0) {
			*mode = OPX_FLOW_CONTROL_MODE_CONFORMANCE;
			return FI_SUCCESS;
		}
	}

	FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "Invalid FI_OPX_FLOW_CONTROL value '%s'\n",
		value != NULL ? value : "<null>");
	return -FI_EINVAL;
}

static int opx_flow_control_process_initialize(bool *retryable)
{
	enum opx_flow_control_mode mode = OPX_FLOW_CONTROL_MODE_NONE;
	size_t			   state_size;
	int			   ret;

	ret = opx_flow_control_select_mode(&mode, retryable);
	if (ret != FI_SUCCESS) {
		return ret;
	}

	ret = opx_flow_control_initialize_mode(mode, &state_size);
	if (ret != FI_SUCCESS) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC, "FI_OPX_FLOW_CONTROL mode '%s' is unavailable\n",
			opx_flow_control_mode_name(mode));
		return ret;
	}

	opx_flow_control_process_mode	    = mode;
	opx_flow_control_process_state_size = state_size;
	return FI_SUCCESS;
}

int opx_flow_control_process_acquire(void)
{
	bool retryable = false;
	int  ret       = -FI_EOPBADSTATE;

	pthread_mutex_lock(&opx_flow_control_process_lock);

	switch (opx_flow_control_process_state) {
	case OPX_FLOW_CONTROL_PROCESS_PUBLISHED:
		opx_flow_control_process_references++;
		ret = FI_SUCCESS;
		break;
	case OPX_FLOW_CONTROL_PROCESS_TERMINAL:
		ret = opx_flow_control_process_result;
		break;
	case OPX_FLOW_CONTROL_PROCESS_UNPUBLISHED:
		ret = opx_flow_control_process_initialize(&retryable);
		if (ret == FI_SUCCESS) {
			opx_flow_control_process_references = 1;
			opx_flow_control_process_result	    = FI_SUCCESS;
			opx_flow_control_process_state	    = OPX_FLOW_CONTROL_PROCESS_PUBLISHED;
		} else if (!retryable) {
			opx_flow_control_process_result = ret;
			opx_flow_control_process_state	= OPX_FLOW_CONTROL_PROCESS_TERMINAL;
		}
		break;
	}

	pthread_mutex_unlock(&opx_flow_control_process_lock);
	return ret;
}

void opx_flow_control_process_release(void)
{
	pthread_mutex_lock(&opx_flow_control_process_lock);
	if (opx_flow_control_process_state == OPX_FLOW_CONTROL_PROCESS_PUBLISHED &&
	    opx_flow_control_process_references != 0) {
		opx_flow_control_process_references--;
	}
	pthread_mutex_unlock(&opx_flow_control_process_lock);
}

int __attribute__((cold)) opx_flow_control_endpoint_init(struct fi_opx_ep *opx_ep)
{
	(void) opx_ep;

	switch (opx_flow_control_selected_mode()) {
	case OPX_FLOW_CONTROL_MODE_NONE:
	case OPX_FLOW_CONTROL_MODE_RCCC:
		return FI_SUCCESS;
	default:
		return -FI_EOPNOTSUPP;
	}
}

void __attribute__((cold)) opx_flow_control_endpoint_fini(struct fi_opx_ep *opx_ep)
{
	(void) opx_ep;

	switch (opx_flow_control_selected_mode()) {
	case OPX_FLOW_CONTROL_MODE_NONE:
	case OPX_FLOW_CONTROL_MODE_RCCC:
	default:
		return;
	}
}

enum opx_flow_control_mode opx_flow_control_selected_mode(void)
{
	enum opx_flow_control_mode mode = OPX_FLOW_CONTROL_MODE_NONE;

	pthread_mutex_lock(&opx_flow_control_process_lock);
	if (opx_flow_control_process_state == OPX_FLOW_CONTROL_PROCESS_PUBLISHED) {
		mode = opx_flow_control_process_mode;
	}
	pthread_mutex_unlock(&opx_flow_control_process_lock);

	return mode;
}

int opx_flow_control_selected_state_size(size_t *state_size)
{
	int ret = FI_SUCCESS;

	if (state_size == NULL) {
		return -FI_EINVAL;
	}

	pthread_mutex_lock(&opx_flow_control_process_lock);
	if (opx_flow_control_process_state == OPX_FLOW_CONTROL_PROCESS_PUBLISHED) {
		*state_size = opx_flow_control_process_state_size;
	} else {
		ret = -FI_EOPBADSTATE;
	}
	pthread_mutex_unlock(&opx_flow_control_process_lock);

	return ret;
}

const char *opx_flow_control_mode_name(enum opx_flow_control_mode mode)
{
	switch (mode) {
	case OPX_FLOW_CONTROL_MODE_NONE:
		return "none";
	case OPX_FLOW_CONTROL_MODE_RCCC:
		return "rccc";
	case OPX_FLOW_CONTROL_MODE_CONFORMANCE:
		return "conformance";
	default:
		return "unknown";
	}
}

static const char *opx_flow_control_fault_scope_name(enum opx_flow_control_fault_scope scope)
{
	switch (scope) {
	case OPX_FLOW_CONTROL_FAULT_NONE:
		return "none";
	case OPX_FLOW_CONTROL_FAULT_FLOW:
		return "flow";
	case OPX_FLOW_CONTROL_FAULT_CONTEXT:
		return "context";
	case OPX_FLOW_CONTROL_FAULT_PROCESS:
		return "process";
	default:
		return "unknown";
	}
}

/* Diagnostic-only; opx_ep may be NULL. No retry/env lookup/hot-path effect. */
void __attribute__((cold)) opx_flow_control_report_fault(struct fi_opx_ep		  *opx_ep,
							 enum opx_flow_control_fault_scope scope, uint32_t reason)
{
	FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "OPX flow control fault: ep=%p scope=%s(%u) reason=%u\n",
		(void *) opx_ep, opx_flow_control_fault_scope_name(scope), (unsigned int) scope, reason);
}
