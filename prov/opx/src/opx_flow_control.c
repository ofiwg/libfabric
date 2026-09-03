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

static int __attribute__((cold)) opx_flow_control_rccc_profile_configure(void)
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
		.retry_cap_us		     = OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US_DEFAULT,
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
	OPX_FLOW_CONTROL_RCCC_PROFILE_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US_", retry_cap_us, UINT16_MAX);
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
	if (opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_GRANT_EMITS_", 8u, &value) != FI_SUCCESS ||
	    value == 0u || value > UINT16_MAX) {
		return -FI_EINVAL;
	}
	if (!opx_flow_control_rccc_tx_profile_is_valid(&profile) || profile.ffirst == 0u) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"Invalid RCCC TX profile: ffirst=%u fmax=%u (ffirst must be nonzero and <= fmax)\n",
			profile.ffirst, profile.fmax);
		return -FI_EINVAL;
	}

	/* Capacity geometry is completed from the endpoint's receive-header
	 * queue at enable.  Non-sharing RCCC endpoints sample occupancy after
	 * each RHQ drain; context sharing remains sampler-inactive at full rate.
	 * Tracked/decision pools are append-only with sticky per-epoch rejects:
	 * the 65th distinct peer flow at the old tracked default of 64
	 * livelocked 256-rank jobs in a permanent CREDIT_REQUEST/CAPACITY_REJECT
	 * storm.  1024 tracked flows covers realistic per-receiver peer counts;
	 * decisions/rejects sized above it because rejects consume both pools. */
	struct opx_flow_control_rccc_scheduler_profile scheduler_profile = {
		/* Zero means derive per endpoint; an explicit zero override is invalid. */
		.max_outstanding_permissions	= 0u,
		.drr_quantum			= 0u,
		.quarantine_deadline_ns		= UINT64_C(100000000),
		.telemetry_health_deadline_ns	= UINT64_C(10000000),
		.minimum_packet_rate_q16_per_ns = UINT64_C(65536),
		.maximum_elapsed_ns		= UINT64_C(1000000),
		.tracked_capacity		= 1024u,
		.decision_capacity		= 4096u,
		.reject_capacity		= 4096u,
		.capacity_guard			= 16u,
		.grant_emits			= (uint16_t) value,
		/* Cold-ramp + hot-set window governor.  Grants are monotonic
		 * within an epoch, so the first grant IS the admission policy:
		 * every flow starts at window_cap_min and doubles per clean
		 * window epoch toward fmax while cold (a flow that never
		 * consumes its window keeps growing - collectives), while a
		 * flow consuming at least half its ceiling (and the absolute
		 * floor, to keep tiny windows from qualifying) for
		 * hot_entry_streak consecutive epochs is clamped to the usable
		 * ring divided by the hot count (sustained incast).  Division
		 * therefore never applies to tiny-demand flows - the
		 * micro-grant control storms that pure divisors caused at high
		 * peer counts cannot re-form.  The epoch is time-based and
		 * must be long enough for a saturating flow's per-epoch
		 * consumption to clear the absolute floor at the receiver's
		 * per-flow service rate (measured: ~100K pkts/s/flow at
		 * 15-to-1 loopback, so 1 ms sees ~100); shorter epochs let the
		 * cold ramp reach fmax before any flow can classify hot.
		 * Entry is fast and exit slow so phase boundaries do not flap
		 * membership; idle decay halves an unused ceiling so a
		 * workload flip cannot ride a stale fmax ceiling for long. */
		.window_cap_min	     = 64u,
		.window_epoch_ns     = UINT64_C(1000000),
		.hot_entry_streak    = 2u,
		.hot_exit_streak     = 4u,
		.hot_absolute_floor  = 32u,
		.hot_consumed_floor  = 8u,
		.hot_min_flows	     = 4u,
		.storm_request_floor = 256u,
		.idle_decay_streak   = 16u,
		.duty_min_flows	     = 4u,
		.duty_streak_k	     = 3u,
		.duty_drain_floor    = 0u,
		.duty_deadline_ns    = UINT64_C(1000000),
	};
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE(name, field, limit)                                   \
	do {                                                                                        \
		if (getenv(name) != NULL) {                                                         \
			if (opx_flow_control_rccc_parse_u64(name, 0u, &value) != FI_SUCCESS) {      \
				return -FI_EINVAL;                                                  \
			}                                                                           \
			if (value == 0u || value > (limit)) {                                       \
				FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,                          \
					"%s value %lu outside valid range [1, %lu]\n", name, value, \
					(uint64_t) (limit));                                        \
				return -FI_EINVAL;                                                  \
			}                                                                           \
			scheduler_profile.field = value;                                            \
		}                                                                                   \
	} while (0)
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_MAX_OUTSTANDING_", max_outstanding_permissions,
					      (uint64_t) OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE - 1u);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DRR_QUANTUM_", drr_quantum,
					      (uint64_t) OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE - 1u);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_QUARANTINE_NS_", quarantine_deadline_ns,
					      UINT64_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_TELEMETRY_DEADLINE_NS_",
					      telemetry_health_deadline_ns, UINT64_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_TRACKED_", tracked_capacity, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DECISIONS_", decision_capacity, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_REJECTS_", reject_capacity, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_WINDOW_CAP_MIN_", window_cap_min, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_WINDOW_EPOCH_NS_", window_epoch_ns,
					      UINT64_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_HOT_ENTRY_STREAK_", hot_entry_streak,
					      UINT8_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_HOT_EXIT_STREAK_", hot_exit_streak, UINT8_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_HOT_ABSOLUTE_FLOOR_", hot_absolute_floor,
					      UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_HOT_CONSUMED_FLOOR_", hot_consumed_floor,
					      UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_HOT_MIN_FLOWS_", hot_min_flows, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_STORM_REQUEST_FLOOR_", storm_request_floor,
					      UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_IDLE_DECAY_STREAK_", idle_decay_streak,
					      UINT8_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DUTY_STREAK_K_", duty_streak_k, UINT32_MAX);
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE("_FI_OPX_FLOW_CONTROL_RCCC_DUTY_DEADLINE_NS_", duty_deadline_ns,
					      UINT64_MAX);
#undef OPX_FLOW_CONTROL_RCCC_SCHEDULER_PARSE
	if (opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_DUTY_MIN_FLOWS_", 4u, &value) != FI_SUCCESS ||
	    value > UINT32_MAX) {
		return -FI_EINVAL;
	}
	scheduler_profile.duty_min_flows = (uint32_t) value;
	if (opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_DUTY_DRAIN_FLOOR_", 0u, &value) != FI_SUCCESS) {
		return -FI_EINVAL;
	}
	scheduler_profile.duty_drain_floor = value;
	if (scheduler_profile.window_cap_min < 8u ||
	    opx_flow_control_rccc_parse_u64("_FI_OPX_FLOW_CONTROL_RCCC_CAPACITY_GUARD_", 16u, &value) != FI_SUCCESS ||
	    value > UINT32_MAX) {
		return -FI_EINVAL;
	}
	scheduler_profile.capacity_guard = (uint32_t) value;
	if (scheduler_profile.decision_capacity < profile.ffirst ||
	    scheduler_profile.reject_capacity < profile.ffirst) {
		FI_WARN(fi_opx_global.prov, FI_LOG_FABRIC,
			"Invalid RCCC scheduler geometry: decisions=%u rejects=%u (each must be >= ffirst %u)\n",
			scheduler_profile.decision_capacity, scheduler_profile.reject_capacity, profile.ffirst);
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
		ret = opx_flow_control_rccc_profile_configure();
		if (ret == FI_SUCCESS) {
			*state_size = OPX_FLOW_CONTROL_STATE_SIZE;
		}
		return ret;
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
