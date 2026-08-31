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

#ifndef _FI_PROV_OPX_FLOW_CONTROL_RCCC_PRIVATE_H_
#define _FI_PROV_OPX_FLOW_CONTROL_RCCC_PRIVATE_H_

#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "rdma/opx/fi_opx_reliability.h"
#include "rdma/opx/opx_flow_control_wire.h"

struct ofi_bufpool;

#define OPX_FLOW_CONTROL_RCCC_RESERVATION_POOL_CAPACITY OPX_MAX_OUTSTANDING_REPLAYS

int  opx_flow_control_reservation_pool_create(struct ofi_bufpool **pool);
int  opx_flow_control_reservation_pool_capacity_verify(const struct ofi_bufpool		       *pool,
						       const struct fi_opx_reliability_service *service,
						       size_t				       *capacity_out);
void opx_flow_control_reservation_pool_destroy(struct ofi_bufpool **pool);
#define OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST \
	((OPX_MP_EGR_MAX_PAYLOAD_BYTES_MAX + FI_OPX_CACHE_LINE_SIZE - 1u) / FI_OPX_CACHE_LINE_SIZE)

#define OPX_FLOW_CONTROL_RCCC_SERIAL_MASK	UINT32_C(0x00FFFFFF)
#define OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE UINT32_C(0x00800000)

#define OPX_FLOW_CONTROL_RCCC_RETRY_INITIAL_MIN_US UINT16_C(1)
#define OPX_FLOW_CONTROL_RCCC_RETRY_DEFAULT_US	   UINT16_C(8)
#define OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US	   UINT16_C(500)

/* The desired window is stored as a 16-bit count, so approved profiles must
 * keep Fmax within it; replay-pool capacity bounds it far below this anyway. */
#define OPX_FLOW_CONTROL_RCCC_FMAX_LIMIT ((uint32_t) UINT16_MAX)

/* Wire kinds 1 and 2 were used by an earlier protocol revision.  They are
 * permanently retired and must not be reassigned; a receiver treats them
 * as invalid. */
/* Grant-window control kinds.  Source-to-receiver kinds (3-5) ride the
 * REQUEST opcode; receiver-to-source kinds (6-8) ride the FEEDBACK opcode.
 * The envelope generation slot carries the epoch for kinds 3-7 and the
 * request_generation serial for kind 8, whose value[0] echoes the epoch so a
 * late old-epoch reject cannot alias a rebased request_generation. */
#define OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST  UINT8_C(3)
#define OPX_FLOW_CONTROL_RCCC_KIND_RETIREMENT	   UINT8_C(4)
#define OPX_FLOW_CONTROL_RCCC_KIND_RESET	   UINT8_C(5)
#define OPX_FLOW_CONTROL_RCCC_KIND_FEEDBACK	   UINT8_C(6)
#define OPX_FLOW_CONTROL_RCCC_KIND_RESET_ACK	   UINT8_C(7)
#define OPX_FLOW_CONTROL_RCCC_KIND_CAPACITY_REJECT UINT8_C(8)

#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING	  UINT8_C(0x01)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING  UINT8_C(0x02)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING		  UINT8_C(0x04)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING	  UINT8_C(0x08)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED UINT8_C(0x10)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED	  UINT8_C(0x20)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT	  UINT8_C(0x40)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_ON_WORK_LIST	  UINT8_C(0x80)
#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED		  UINT8_C(0x01)
#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING	  UINT8_C(0x02)
#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID	  UINT8_C(0x04)
#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING	  UINT8_C(0x40)
#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING	  UINT8_C(0x80)

#define FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MAGIC		UINT64_C(0x484353434658504F)
#define FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_VERSION	UINT32_C(1)
#define FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ENDIAN_LITTLE UINT32_C(1)
#define FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT	64u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK		(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT - 1)
#define FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS	8u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_HEADER_BYTES	1024u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES	256u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_DECISION_BYTES	128u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_REJECT_BYTES	128u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES	512u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_ACCUMULATOR_BYTES 128u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_BYTES	1536u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBERS_OFFSET	1024u
#define FI_OPX_FLOW_CONTROL_SCHEDULER_DECISIONS_OFFSET  \
	(FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBERS_OFFSET + \
	 FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS * FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES)

static_assert((FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) == 0,
	      "FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT must be a power of 2!");

/* The exact half-range distance has no safe serial ordering. */
enum opx_flow_control_rccc_serial_classification {
	OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID = 0,
	OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE,
	OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER,
	OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS,
	OPX_FLOW_CONTROL_RCCC_SERIAL_STALE
};

enum opx_flow_control_rccc_block_reason {
	OPX_FLOW_CONTROL_RCCC_BLOCK_NONE = 0,
	OPX_FLOW_CONTROL_RCCC_BLOCK_NO_CREDIT,
	OPX_FLOW_CONTROL_RCCC_BLOCK_BYTE_THROTTLE,
	OPX_FLOW_CONTROL_RCCC_BLOCK_PIO,
	OPX_FLOW_CONTROL_RCCC_BLOCK_PSN,
	OPX_FLOW_CONTROL_RCCC_BLOCK_SCHEDULER,
	OPX_FLOW_CONTROL_RCCC_BLOCK_RESET,
	OPX_FLOW_CONTROL_RCCC_BLOCK_QUARANTINE,
	OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY,
	OPX_FLOW_CONTROL_RCCC_BLOCK_FAULT
};

enum opx_flow_control_rccc_scheduler_state {
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT = 0,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_PREPARING,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_QUARANTINED
};

enum opx_flow_control_rccc_scheduler_initialization_state {
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_UNINITIALIZED = 0,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_INITIALIZING,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_READY
};

enum opx_flow_control_rccc_scheduler_mapping_result {
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK = 0,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_INVALID,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OVERFLOW,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_ALIGNMENT,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_TRUNCATED,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_SCHEMA,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_GEOMETRY,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_RESERVED,
	OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_STATE
};

struct opx_flow_control_rccc_retry {
	uint16_t delay_us;
	uint8_t	 active;
	uint8_t	 reserved;
};

struct opx_flow_control_rccc_scheduler_reference {
	uint32_t slot;
	uint64_t generation;
};

struct opx_flow_control_rccc_tombstone {
	union fi_opx_reliability_service_flow_key flow_key;
	uint32_t				  last_epoch;
	uint64_t				  terminal_time_ns;
};

enum opx_flow_control_rccc_reservation_state {
	OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD = 0,
	OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED
};

struct opx_flow_control_rccc_reservation_record {
	struct fi_opx_ep		  *opx_ep;
	struct fi_opx_reliability_tx_flow *flow_identity;
	uint64_t			   generation;
	union {
		uint64_t					 identity_cookie;
		struct opx_flow_control_rccc_reservation_record *flow_next;
	};
	union fi_opx_reliability_service_flow_key flow_key;
	uint32_t				  epoch;
	uint32_t				  reserved_count;
	uint32_t				  committed_count;
	uint32_t				  completed_count;
	uint8_t					  state;
	uint8_t					  packet_class;
	uint16_t				  reserved_accepted;
} __attribute__((aligned(L2_CACHE_LINE_SIZE))) __attribute__((__packed__));

/* Grant-window TX overlay.  The flow tail begins 16 bytes into the enclosing
 * generic state, so keeping available/held/flags within the first 48 bytes
 * here places the whole release-mode admission and send working set in the
 * flow tail's first cache line; the u64 deadlines that only context progress
 * touches sit on the second line.  Serial fields are 24-bit values in u32
 * storage; credit counters are bounded by Fmax and replay-pool capacity, so
 * u32 is exact (overflow guards check UINT32_MAX).  committed_originals is
 * monotonic per epoch - packet completion never decrements it - because the
 * epoch conservation invariant needs the original-commit count, and a
 * decrement from an old-epoch completion after rebase would corrupt the new
 * epoch's term. */
struct opx_flow_control_rccc_tx_state {
	uint32_t				  epoch;
	uint32_t				  target;
	uint32_t				  grant;
	uint32_t				  available_credits;
	uint32_t				  held_credits;
	uint32_t				  committed_originals;
	uint16_t				  desired_window;
	uint8_t					  block_reason;
	uint8_t					  flags;
	uint32_t				  request_generation;
	struct opx_flow_control_rccc_retry	  request_retry;
	struct opx_flow_control_rccc_retry	  retirement_retry;
	struct opx_flow_control_rccc_retry	  reset_retry;
	uint32_t				  retired;
	uint32_t				  retired_acked;
	uint32_t				  reserved_accepted_serial;
	uint64_t				  activity_time_ns;
	uint64_t				  control_deadline_ns;
	uint64_t				  reset_deadline_ns;
	uint32_t				  fault_contained;
	uint32_t				  reserved_backlog;
	uint32_t				  reserved_committed_unreturned;
	uint32_t				  reserved_requested;
	union fi_opx_reliability_service_flow_key flow_key;
	uint32_t				  reserved0;
};

/* Scheduler-core RX overlay.  epoch_snapshot retains the last closed
 * epoch across DORMANT rebinds so a stale kind-3 cannot blindly rebind an
 * old epoch (classified against the snapshot, STALE dropped).
 * request_generation stores the requester's kind-3 value[2] serial and is
 * echoed as the kind-8 envelope generation.  decision_slot/reject_slot are
 * back-references into the endpoint scheduler's fixed pools.  pending_epoch
 * holds the proposed epoch while RESET_ACK withholding drains outstanding
 * grants. */
struct opx_flow_control_rccc_rx_state {
	uint32_t					 epoch;
	uint32_t					 target;
	uint32_t					 grant;
	uint32_t					 observed;
	uint32_t					 retired;
	uint32_t					 retired_accepted;
	uint8_t						 scheduler_state;
	uint8_t						 flags;
	uint8_t						 work_listed;
	uint8_t						 reserved0;
	struct opx_flow_control_rccc_scheduler_reference scheduler;
	uint64_t					 permission_contribution;
	int64_t						 drr_deficit;
	uint64_t					 quarantine_deadline_ns;
	uint32_t					 epoch_snapshot;
	uint32_t					 request_generation;
	uint32_t					 decision_slot;
	uint32_t					 reject_slot;
	uint8_t						 reserved_scheduler[16];
	uint32_t					 pending_epoch;
	uint32_t					 reserved2;
};

enum opx_flow_control_rccc_capacity_reason {
	OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_TRACKED = 1,
	OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_DECISION,
	OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_LEDGER
};

enum opx_flow_control_rccc_decision_state {
	OPX_FLOW_CONTROL_RCCC_DECISION_EMPTY = 0,
	OPX_FLOW_CONTROL_RCCC_DECISION_ACCEPT,
	OPX_FLOW_CONTROL_RCCC_DECISION_REJECT
};

/* Sticky decisions are keyed {flow key, epoch}: a replayed kind-3 for a
 * decided pairing gets the original outcome verbatim and is never
 * re-evaluated; request_generation is only the kind-8 reply serial. */
struct opx_flow_control_rccc_decision_key {
	union fi_opx_reliability_service_flow_key flow_key;
	uint32_t				  epoch;
};

struct opx_flow_control_rccc_decision_record {
	struct opx_flow_control_rccc_decision_key key;
	uint32_t				  request_generation;
	uint32_t				  reject_slot;
	uint8_t					  state;
	uint8_t					  occupied;
	uint8_t					  grant_seen;
	uint8_t					  reserved;
};

/* Reject detail (reason/limit/observed/FI_ENOSPC) is local-only evidence;
 * the kind-8 wire carries only {request_generation, epoch echo, TERMINAL}. */
struct opx_flow_control_rccc_reject_record {
	struct opx_flow_control_rccc_decision_key key;
	uint32_t				  request_generation;
	uint32_t				  limit;
	uint32_t				  observed;
	int32_t					  fi_errno;
	uint8_t					  reason;
	uint8_t					  occupied;
	uint8_t					  reserved[2];
};

struct opx_flow_control_rccc_scheduler_profile {
	uint64_t max_outstanding_permissions;
	uint64_t drr_quantum;
	uint64_t quarantine_deadline_ns;
	uint64_t minimum_packet_rate_q16_per_ns;
	uint64_t maximum_elapsed_ns;
	uint32_t tracked_capacity;
	uint32_t decision_capacity;
	uint32_t reject_capacity;
	uint32_t reserved;
};

/* Live interior of the header's ledger block (offset 384).
 * RX flows are member-owned under OPX context sharing (the hardware-RHQ
 * poller forwards every packet to its owner's software RHQ), so per-flow
 * state never races cross-process; the only cross-member state is this
 * aggregate ledger and the tracked-capacity counter, both lock-free
 * atomics.  attached_count is the mapping lifecycle refcount: the last
 * detacher unlinks the segment.  No robust/member-death handling exists -
 * a member dying while attached leaks the segment until job cleanup,
 * matching the exposure of the pre-existing context-sharing spinlocks. */
struct opx_flow_control_rccc_scheduler_shared_ledger {
	_Atomic uint64_t outstanding_permissions;
	_Atomic uint32_t tracked_used;
	_Atomic uint32_t attached_count;
	uint8_t		 reserved[112];
};

#define OPX_FLOW_CONTROL_RCCC_MEMBER_EMPTY  UINT32_C(0)
#define OPX_FLOW_CONTROL_RCCC_MEMBER_JOINED UINT32_C(1)
#define OPX_FLOW_CONTROL_RCCC_MEMBER_LEFT   UINT32_C(2)

/* Live interior of a member slot's identity block: indexed by
 * subcontext, single-writer (each member stamps only its own slot). */
struct opx_flow_control_rccc_scheduler_member_identity {
	uint64_t pid;
	uint64_t process_start_token;
	uint64_t member_generation;
	uint32_t subctxt;
	uint32_t state;
	uint8_t	 reserved[32];
};

/* Process-local scheduler core hung off fi_opx_ep for RCCC endpoints only
 * (reservation-pool precedent); under context sharing it re-targets shared
 * slots.  outstanding_permissions is the aggregate ledger, updated
 * incrementally with the masked 24-bit delta at each grant/observe/settle/
 * reclaim event and never re-derived from per-flow masked subtractions.
 * Under context sharing (shared_header != NULL) the authoritative ledger
 * and tracked counter live in the shared mapping; the local fields mirror
 * only this member's own contribution for the teardown and NDEBUG
 * reconcile invariants.  Decisions/rejects stay process-local even when
 * sharing: member-owned RX flows mean sticky decisions never need
 * cross-process coherence (the mapping's decision/reject regions are sized
 * and zeroed but inactive). */
struct opx_flow_control_rccc_scheduler {
	struct opx_flow_control_rccc_scheduler_profile	      profile;
	uint64_t					      outstanding_permissions;
	uint64_t					      accrual_remainder_q16;
	uint64_t					      accrual_last_ns;
	uint64_t					      allocation_generation;
	uint32_t					      tracked_used;
	uint32_t					      decision_used;
	uint32_t					      reject_used;
	uint32_t					      reserved;
	struct opx_flow_control_rccc_decision_record	     *decisions;
	struct opx_flow_control_rccc_reject_record	     *rejects;
	uint8_t						     *shared_base;
	struct opx_flow_control_rccc_scheduler_header	     *shared_header;
	struct opx_flow_control_rccc_scheduler_shared_ledger *shared_ledger;
	size_t						      shared_length;
	uint32_t					      shared_member_index;
	uint32_t					      shared_member_count;
	uint64_t					      shared_release_pending;
	char						      shared_name[128];
};

#define OPX_FLOW_CONTROL_RCCC_Q16_ONE UINT32_C(65536)

struct opx_flow_control_rccc_scheduler_accumulator {
	_Atomic uint64_t active_bank_epoch;
	uint64_t	 entry_generation;
	uint64_t	 bank_generation[2];
	_Atomic uint64_t producer_refs[2];
	uint8_t		 reserved0[16];
	_Atomic uint64_t bank_deltas[2];
	uint8_t		 reserved1[48];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

struct opx_flow_control_rccc_scheduler_member {
	struct opx_flow_control_rccc_scheduler_member_identity identity;
	uint8_t						       software_rhq[64];
	uint8_t						       software_eager[64];
	uint8_t						       lifecycle[64];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

struct opx_flow_control_rccc_scheduler_decision {
	uint8_t outer_identity[32];
	uint8_t flow_identity[16];
	uint8_t decision_state[16];
	uint8_t response[32];
	uint8_t lifetime[32];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

struct opx_flow_control_rccc_scheduler_reject {
	uint8_t wire_envelope[64];
	uint8_t decision_detail[32];
	uint8_t lifetime[32];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

/* Live interior of a tracked flow entry's core: a CAS-claimed owner
 * word ((member_index + 1) << 32 | low bits of the claim generation; zero
 * means free) plus the owner-stamped flow key for diagnostics.  Claim-scan
 * allocation keeps slot references unique across member join/leave, which
 * a bare shared counter cannot (indices would be reused while still held). */
struct opx_flow_control_rccc_scheduler_flow_owner {
	_Atomic uint64_t claim;
	uint64_t	 flow_key_prefix;
	uint32_t	 flow_key_suffix;
	uint32_t	 reserved;
};

struct opx_flow_control_rccc_scheduler_flow {
	uint8_t						   core[FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES];
	struct opx_flow_control_rccc_scheduler_accumulator accumulators[FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

struct opx_flow_control_rccc_scheduler_header {
	uint64_t magic;
	uint32_t abi_version;
	uint32_t endian;
	uint32_t alignment;
	uint32_t header_bytes;
	uint32_t member_bytes;
	uint32_t decision_bytes;
	uint32_t reject_bytes;
	uint32_t flow_core_bytes;
	uint32_t accumulator_bytes;
	uint32_t flow_bytes;
	uint64_t region_length;
	uint8_t	 reserved0[8];

	uint32_t member_count;
	uint32_t configured_inventory;
	uint32_t ffirst;
	uint32_t decision_capacity;
	uint32_t reject_capacity;
	uint32_t tracked_capacity;
	uint32_t eligible_capacity;
	uint32_t fmax;
	uint8_t	 reserved1[32];

	uint64_t members_offset;
	uint64_t decisions_offset;
	uint64_t rejects_offset;
	uint64_t flows_offset;
	uint64_t members_region_bytes;
	uint64_t decisions_region_bytes;
	uint64_t rejects_region_bytes;
	uint64_t flows_region_bytes;

	_Atomic uint32_t initialization_state;
	uint32_t	 reserved2;
	uint64_t	 initialization_generation;
	uint64_t	 membership_generation;
	uint8_t		 reserved3[40];

	uint8_t						     reserved_telemetry[128];
	struct opx_flow_control_rccc_scheduler_shared_ledger ledger;
	uint8_t						     reserved_scheduler[128];
	uint8_t						     reserved_mutex[128];
	uint8_t						     reserved_journal[256];
} __attribute__((aligned(FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT)));

struct opx_flow_control_rccc_scheduler_geometry {
	const uint8_t					    *mapping_base;
	const struct opx_flow_control_rccc_scheduler_header *header;
	size_t						     mapping_length;
	size_t						     members_offset;
	size_t						     decisions_offset;
	size_t						     rejects_offset;
	size_t						     flows_offset;
	size_t						     members_region_bytes;
	size_t						     decisions_region_bytes;
	size_t						     rejects_region_bytes;
	size_t						     flows_region_bytes;
	uint32_t					     member_count;
	uint32_t					     configured_inventory;
	uint32_t					     ffirst;
	uint32_t					     decision_capacity;
	uint32_t					     reject_capacity;
	uint32_t					     tracked_capacity;
	uint32_t					     eligible_capacity;
	uint32_t					     fmax;
};

static inline bool opx_flow_control_rccc_size_add(size_t left, size_t right, size_t *sum)
{
	if (sum == NULL || left > SIZE_MAX - right) {
		return false;
	}

	*sum = left + right;
	return true;
}

static inline bool opx_flow_control_rccc_size_multiply(size_t left, size_t right, size_t *product)
{
	if (product == NULL || (right != 0u && left > SIZE_MAX / right)) {
		return false;
	}

	*product = left * right;
	return true;
}

static inline bool opx_flow_control_rccc_size_align(size_t value, size_t alignment, size_t *aligned_value)
{
	size_t rounded;

	if (aligned_value == NULL || alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
	    !opx_flow_control_rccc_size_add(value, alignment - 1u, &rounded)) {
		return false;
	}

	*aligned_value = rounded & ~(alignment - 1u);
	return true;
}

static inline enum opx_flow_control_rccc_scheduler_mapping_result
opx_flow_control_rccc_scheduler_geometry_compute(uint32_t decision_capacity, uint32_t reject_capacity,
						 uint32_t					  tracked_capacity,
						 struct opx_flow_control_rccc_scheduler_geometry *geometry)
{
	struct opx_flow_control_rccc_scheduler_geometry candidate = {0};
	size_t						end;

	if (geometry == NULL) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_INVALID;
	}
	if (!opx_flow_control_rccc_size_multiply(FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS,
						 FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES,
						 &candidate.members_region_bytes) ||
	    !opx_flow_control_rccc_size_multiply(decision_capacity, FI_OPX_FLOW_CONTROL_SCHEDULER_DECISION_BYTES,
						 &candidate.decisions_region_bytes) ||
	    !opx_flow_control_rccc_size_multiply(reject_capacity, FI_OPX_FLOW_CONTROL_SCHEDULER_REJECT_BYTES,
						 &candidate.rejects_region_bytes) ||
	    !opx_flow_control_rccc_size_multiply(tracked_capacity, FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_BYTES,
						 &candidate.flows_region_bytes)) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OVERFLOW;
	}

	candidate.members_offset   = FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBERS_OFFSET;
	candidate.decisions_offset = FI_OPX_FLOW_CONTROL_SCHEDULER_DECISIONS_OFFSET;
	if (!opx_flow_control_rccc_size_add(candidate.decisions_offset, candidate.decisions_region_bytes,
					    &candidate.rejects_offset) ||
	    !opx_flow_control_rccc_size_add(candidate.rejects_offset, candidate.rejects_region_bytes,
					    &candidate.flows_offset) ||
	    !opx_flow_control_rccc_size_add(candidate.flows_offset, candidate.flows_region_bytes, &end) ||
	    !opx_flow_control_rccc_size_align(end, FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
					      &candidate.mapping_length)) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OVERFLOW;
	}

	candidate.decision_capacity = decision_capacity;
	candidate.reject_capacity   = reject_capacity;
	candidate.tracked_capacity  = tracked_capacity;
	*geometry		    = candidate;
	return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK;
}

static inline bool opx_flow_control_rccc_scheduler_bytes_are_zero(const uint8_t *bytes, size_t length)
{
	size_t index;

	for (index = 0u; index < length; ++index) {
		if (bytes[index] != 0u) {
			return false;
		}
	}
	return true;
}

static inline enum opx_flow_control_rccc_scheduler_mapping_result
opx_flow_control_rccc_scheduler_mapping_validate(const uint8_t *mapping_base, size_t mapping_length,
						 const struct opx_flow_control_rccc_scheduler_header *header,
						 struct opx_flow_control_rccc_scheduler_geometry     *geometry)
{
	struct opx_flow_control_rccc_scheduler_geometry	    candidate;
	enum opx_flow_control_rccc_scheduler_mapping_result result;

	assert(mapping_base != NULL);
	assert(header != NULL);
	assert(geometry != NULL);
	assert((const void *) mapping_base == (const void *) header);
	if (((uintptr_t) mapping_base & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) != 0u) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_ALIGNMENT;
	}
	if (mapping_length < FI_OPX_FLOW_CONTROL_SCHEDULER_HEADER_BYTES) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_TRUNCATED;
	}
	if (header->magic != FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MAGIC ||
	    header->abi_version != FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_VERSION ||
	    header->endian != FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ENDIAN_LITTLE ||
	    header->alignment != FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT ||
	    header->header_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_HEADER_BYTES ||
	    header->member_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES ||
	    header->decision_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_DECISION_BYTES ||
	    header->reject_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_REJECT_BYTES ||
	    header->flow_core_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES ||
	    header->accumulator_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_ACCUMULATOR_BYTES ||
	    header->flow_bytes != FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_BYTES) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_SCHEMA;
	}
	if (!opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved0, sizeof(header->reserved0)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved1, sizeof(header->reserved1)) ||
	    header->reserved2 != 0u ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved3, sizeof(header->reserved3)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved_telemetry,
							    sizeof(header->reserved_telemetry)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->ledger.reserved, sizeof(header->ledger.reserved)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved_scheduler,
							    sizeof(header->reserved_scheduler)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved_mutex, sizeof(header->reserved_mutex)) ||
	    !opx_flow_control_rccc_scheduler_bytes_are_zero(header->reserved_journal,
							    sizeof(header->reserved_journal))) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_RESERVED;
	}
	if (header->member_count < 2u || header->member_count > FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS ||
	    header->configured_inventory > header->ffirst || header->ffirst == 0u ||
	    header->ffirst > header->decision_capacity || header->reject_capacity < header->ffirst ||
	    header->tracked_capacity == 0u || header->eligible_capacity == 0u ||
	    header->eligible_capacity > header->tracked_capacity || header->fmax == 0u) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_GEOMETRY;
	}

	result = opx_flow_control_rccc_scheduler_geometry_compute(header->decision_capacity, header->reject_capacity,
								  header->tracked_capacity, &candidate);
	if (result != OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK) {
		return result;
	}
	if (header->region_length > SIZE_MAX || (size_t) header->region_length != candidate.mapping_length ||
	    mapping_length < candidate.mapping_length) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_TRUNCATED;
	}
	if (mapping_length != candidate.mapping_length) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_GEOMETRY;
	}
	if ((header->members_offset & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) != 0u ||
	    (header->decisions_offset & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) != 0u ||
	    (header->rejects_offset & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) != 0u ||
	    (header->flows_offset & FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MASK) != 0u) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_ALIGNMENT;
	}
	if (header->members_offset != candidate.members_offset ||
	    header->decisions_offset != candidate.decisions_offset ||
	    header->rejects_offset != candidate.rejects_offset || header->flows_offset != candidate.flows_offset ||
	    header->members_region_bytes != candidate.members_region_bytes ||
	    header->decisions_region_bytes != candidate.decisions_region_bytes ||
	    header->rejects_region_bytes != candidate.rejects_region_bytes ||
	    header->flows_region_bytes != candidate.flows_region_bytes) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_GEOMETRY;
	}
	if (atomic_load_explicit(&header->initialization_state, memory_order_acquire) !=
	    OPX_FLOW_CONTROL_RCCC_SCHEDULER_READY) {
		return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_STATE;
	}

	candidate.mapping_base	       = mapping_base;
	candidate.header	       = header;
	candidate.member_count	       = header->member_count;
	candidate.configured_inventory = header->configured_inventory;
	candidate.ffirst	       = header->ffirst;
	candidate.eligible_capacity    = header->eligible_capacity;
	candidate.fmax		       = header->fmax;
	*geometry		       = candidate;
	return OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK;
}

static inline bool opx_flow_control_rccc_serial_is_valid(uint32_t value)

{
	return (value & ~OPX_FLOW_CONTROL_RCCC_SERIAL_MASK) == 0u;
}

static inline enum opx_flow_control_rccc_serial_classification opx_flow_control_rccc_serial_classify(uint32_t current,
												     uint32_t candidate)
{
	uint32_t delta;

	if (!opx_flow_control_rccc_serial_is_valid(current) || !opx_flow_control_rccc_serial_is_valid(candidate)) {
		return OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID;
	}

	delta = (candidate - current) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
	if (delta == 0u) {
		return OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE;
	}
	if (delta < OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
		return OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER;
	}
	if (delta == OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
		return OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS;
	}
	return OPX_FLOW_CONTROL_RCCC_SERIAL_STALE;
}

static inline enum opx_flow_control_rccc_serial_classification
opx_flow_control_rccc_serial_monotonic_max(uint32_t current, uint32_t candidate, uint32_t *maximum)
{
	enum opx_flow_control_rccc_serial_classification classification =
		opx_flow_control_rccc_serial_classify(current, candidate);

	if (maximum == NULL) {
		return OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID;
	}

	switch (classification) {
	case OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE:
	case OPX_FLOW_CONTROL_RCCC_SERIAL_STALE:
		*maximum = current;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER:
		*maximum = candidate;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID:
	case OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS:
		break;
	}

	return classification;
}

static inline uint32_t opx_flow_control_rccc_serial_add(uint32_t serial, uint32_t delta)
{
	return (serial + delta) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
}

static inline bool opx_flow_control_rccc_retry_initial_is_valid(uint32_t initial_delay_us)
{
	return initial_delay_us >= OPX_FLOW_CONTROL_RCCC_RETRY_INITIAL_MIN_US &&
	       initial_delay_us <= OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US;
}

static inline bool opx_flow_control_rccc_retry_next(uint16_t delay_us, uint16_t *next_delay_us)
{
	uint32_t doubled;

	if (next_delay_us == NULL || !opx_flow_control_rccc_retry_initial_is_valid(delay_us)) {
		return false;
	}

	doubled = (uint32_t) delay_us * 2u;
	*next_delay_us =
		doubled < OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US ? (uint16_t) doubled : OPX_FLOW_CONTROL_RCCC_RETRY_CAP_US;
	return true;
}

/* Exhausted local generations fail instead of wrapping into an observable stale identity. */
static inline bool opx_flow_control_rccc_generation_advance(uint64_t generation, uint64_t *next_generation)
{
	if (next_generation == NULL || generation == UINT64_MAX) {
		return false;
	}

	*next_generation = generation + UINT64_C(1);
	return true;
}

enum opx_flow_control_result opx_flow_control_rccc_reservation_hold(
	struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow_identity,
	const union fi_opx_reliability_service_flow_key *flow_key, struct opx_flow_control_rccc_tx_state *tx_state,
	struct opx_flow_control_rccc_reservation_record *record, enum opx_flow_control_packet_class packet_class,
	uint64_t generation, uint32_t reserved_count);
void opx_flow_control_rccc_reservation_reduce(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_tx_state *tx_state,
					      struct opx_flow_control_rccc_reservation_record *record,
					      uint64_t generation, uint32_t new_count);
void opx_flow_control_rccc_reservation_register(struct fi_opx_ep				*opx_ep,
						struct opx_flow_control_rccc_tx_state		*tx_state,
						struct opx_flow_control_rccc_reservation_record *record,
						uint64_t					 generation);
void opx_flow_control_rccc_reservation_cancel(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_tx_state *tx_state,
					      struct opx_flow_control_rccc_reservation_record *record,
					      uint64_t					       generation);
bool opx_flow_control_rccc_reservation_complete(struct fi_opx_ep				*opx_ep,
						struct opx_flow_control_rccc_reservation_record *record,
						uint64_t					 generation);

/* All timer/window values are injectable inputs; no production defaults are
 * installed here. */
struct opx_flow_control_rccc_tx_profile {
	uint32_t ffirst;
	uint32_t fmax;
	uint64_t idle_retire_delay_ns;
	uint16_t request_retry_initial_us;
	uint16_t retirement_retry_initial_us;
	uint16_t reset_retry_initial_us;
	uint16_t reserved;
};

enum opx_flow_control_rccc_tx_emission {
	OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE = 0,
	OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST,
	OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT,
	OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET
};

struct opx_flow_control_rccc_tx_emission_intent {
	uint32_t epoch;
	uint32_t target;
	uint32_t retired;
	uint32_t request_generation;
	uint8_t	 kind;
};

bool opx_flow_control_rccc_tx_profile_is_valid(const struct opx_flow_control_rccc_tx_profile *profile);
int  opx_flow_control_rccc_tx_window_open(struct opx_flow_control_rccc_tx_state		*tx_state,
					  const struct opx_flow_control_rccc_tx_profile *profile);
enum opx_flow_control_result opx_flow_control_rccc_tx_demand(struct opx_flow_control_rccc_tx_state	   *tx_state,
							     const struct opx_flow_control_rccc_tx_profile *profile,
							     uint32_t requested_packets);
enum opx_flow_control_rccc_tx_emission
     opx_flow_control_rccc_tx_progress(struct opx_flow_control_rccc_tx_state	     *tx_state,
				       const struct opx_flow_control_rccc_tx_profile *profile, uint64_t now_ns,
				       struct opx_flow_control_rccc_tx_emission_intent *intent);
void opx_flow_control_rccc_tx_emitted(struct opx_flow_control_rccc_tx_state	    *tx_state,
				      const struct opx_flow_control_rccc_tx_profile *profile,
				      enum opx_flow_control_rccc_tx_emission kind, uint64_t now_ns);
enum opx_flow_control_rccc_serial_classification
opx_flow_control_rccc_tx_feedback_apply(struct opx_flow_control_rccc_tx_state	      *tx_state,
					const struct opx_flow_control_rccc_tx_profile *profile, uint32_t grant,
					uint32_t retired_acked_value, uint32_t generation,
					struct opx_flow_control_rccc_feedback_counters *counters);
enum opx_flow_control_result
opx_flow_control_rccc_tx_capacity_reject(struct opx_flow_control_rccc_tx_state	       *tx_state,
					 const struct opx_flow_control_rccc_tx_profile *profile, uint32_t generation,
					 struct opx_flow_control_rccc_feedback_counters *counters);
enum opx_flow_control_result
			     opx_flow_control_rccc_tx_reset_begin(struct opx_flow_control_rccc_tx_state		*tx_state,
								  const struct opx_flow_control_rccc_tx_profile *profile, uint64_t now_ns);
enum opx_flow_control_result opx_flow_control_rccc_tx_reset_ack(struct opx_flow_control_rccc_tx_state	      *tx_state,
								const struct opx_flow_control_rccc_tx_profile *profile,
								uint32_t generation);
int opx_flow_control_rccc_scheduler_shared_attach(struct opx_flow_control_rccc_scheduler *scheduler,
						  const char *job_key_str, uint32_t hfi_unit, uint32_t context_number,
						  uint32_t subctxt, uint32_t subctxt_cnt);
enum opx_flow_control_result opx_flow_control_rccc_rx_control(struct fi_opx_ep				  *opx_ep,
							      struct fi_opx_reliability_rx_flow		  *flow,
							      const struct opx_flow_control_wire_envelope *envelope);
enum opx_flow_control_result opx_flow_control_rccc_rx_emit_pending(struct fi_opx_ep		     *opx_ep,
								   struct fi_opx_reliability_rx_flow *flow,
								   uint8_t			      pending_flag);

_Static_assert(FI_OPX_FLOW_CONTROL_SCHEDULER_DECISIONS_OFFSET == 3072u, "RCCC scheduler fixed region size");
_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_accumulator) ==
		       FI_OPX_FLOW_CONTROL_SCHEDULER_ACCUMULATOR_BYTES,
	       "RCCC scheduler accumulator size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_accumulator) ==
		       FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler accumulator alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_accumulator, active_bank_epoch) == 0u,
	       "RCCC scheduler active-bank offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_accumulator, entry_generation) == 8u,
	       "RCCC scheduler entry-generation offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_accumulator, bank_generation) == 16u,
	       "RCCC scheduler bank-generation offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_accumulator, producer_refs) == 32u,
	       "RCCC scheduler producer-reference offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_accumulator, bank_deltas) == 64u,
	       "RCCC scheduler bank-delta offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_shared_ledger) == 128u,
	       "RCCC scheduler shared ledger must fill the header reserved_ledger block");
_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_member_identity) == 64u,
	       "RCCC scheduler member identity must fill the member identity block");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_member, identity) == 0u,
	       "RCCC scheduler member identity offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_member) == FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES,
	       "RCCC scheduler member size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_member) == FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler member alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_member, software_rhq) == 64u,
	       "RCCC scheduler member RHQ offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_member, software_eager) == 128u,
	       "RCCC scheduler member eager offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_member, lifecycle) == 192u,
	       "RCCC scheduler member lifecycle offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_decision) == FI_OPX_FLOW_CONTROL_SCHEDULER_DECISION_BYTES,
	       "RCCC scheduler decision size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_decision) == FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler decision alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_decision, flow_identity) == 32u,
	       "RCCC scheduler decision flow offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_decision, decision_state) == 48u,
	       "RCCC scheduler decision state offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_decision, response) == 64u,
	       "RCCC scheduler decision response offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_decision, lifetime) == 96u,
	       "RCCC scheduler decision lifetime offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_reject) == FI_OPX_FLOW_CONTROL_SCHEDULER_REJECT_BYTES,
	       "RCCC scheduler reject size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_reject) == FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler reject alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_reject, decision_detail) == 64u,
	       "RCCC scheduler reject detail offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_reject, lifetime) == 96u,
	       "RCCC scheduler reject lifetime offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_flow_owner) <=
		       FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES,
	       "RCCC scheduler flow owner must fit the flow core block");
_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_flow) == FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_BYTES,
	       "RCCC scheduler flow size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_flow) == FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler flow alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_flow, accumulators) ==
		       FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES,
	       "RCCC scheduler accumulator array offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_flow, accumulators[1]) == 640u,
	       "RCCC scheduler second accumulator offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_flow, accumulators[7]) == 1408u,
	       "RCCC scheduler final accumulator offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_header) == FI_OPX_FLOW_CONTROL_SCHEDULER_HEADER_BYTES,
	       "RCCC scheduler header size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_scheduler_header) == FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT,
	       "RCCC scheduler header alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, member_count) == 64u,
	       "RCCC scheduler count block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, members_offset) == 128u,
	       "RCCC scheduler offset block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, initialization_state) == 192u,
	       "RCCC scheduler initialization block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, reserved_telemetry) == 256u,
	       "RCCC scheduler telemetry block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, ledger) == 384u,
	       "RCCC scheduler ledger block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, reserved_scheduler) == 512u,
	       "RCCC scheduler state block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, reserved_mutex) == 640u,
	       "RCCC scheduler mutex block offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_header, reserved_journal) == 768u,
	       "RCCC scheduler journal block offset");

_Static_assert(sizeof(struct opx_flow_control_rccc_retry) == 4u, "RCCC retry representation size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_retry) == 2u, "RCCC retry representation alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_retry, delay_us) == 0u, "RCCC retry delay offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_retry *) 0)->delay_us) == sizeof(uint16_t),
	       "RCCC retry delay width");
_Static_assert(offsetof(struct opx_flow_control_rccc_retry, active) == 2u, "RCCC retry active offset");

_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_reference, slot) == 0u, "RCCC scheduler slot offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_scheduler_reference *) 0)->slot) == sizeof(uint32_t),
	       "RCCC scheduler slot width");
_Static_assert(offsetof(struct opx_flow_control_rccc_scheduler_reference, generation) == 8u,
	       "RCCC scheduler generation offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_scheduler_reference *) 0)->generation) == sizeof(uint64_t),
	       "RCCC scheduler generation width");
_Static_assert(sizeof(struct opx_flow_control_rccc_scheduler_reference) == 16u, "RCCC scheduler reference size");

_Static_assert(offsetof(struct opx_flow_control_rccc_tombstone, flow_key) == 0u, "RCCC tombstone key offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_tombstone *) 0)->flow_key) ==
		       sizeof(union fi_opx_reliability_service_flow_key),
	       "RCCC tombstone key width");
_Static_assert(offsetof(struct opx_flow_control_rccc_tombstone, last_epoch) == 12u, "RCCC tombstone epoch offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_tombstone *) 0)->last_epoch) == sizeof(uint32_t),
	       "RCCC tombstone epoch width");
_Static_assert(offsetof(struct opx_flow_control_rccc_tombstone, terminal_time_ns) == 16u,
	       "RCCC tombstone terminal-time offset");
_Static_assert(sizeof(struct opx_flow_control_rccc_tombstone) == 24u, "RCCC tombstone size");

_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, opx_ep) == 0u,
	       "RCCC reservation endpoint offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, flow_identity) == 8u,
	       "RCCC reservation flow-identity offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, generation) == 16u,
	       "RCCC reservation generation offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_reservation_record *) 0)->generation) == sizeof(uint64_t),
	       "RCCC reservation generation width");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, identity_cookie) == 24u,
	       "RCCC reservation identity-cookie offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, flow_next) == 24u,
	       "RCCC reservation committed-record link offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, flow_key) == 32u,
	       "RCCC reservation key offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_reservation_record *) 0)->flow_key) ==
		       sizeof(union fi_opx_reliability_service_flow_key),
	       "RCCC reservation key width");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, epoch) == 44u,
	       "RCCC reservation epoch offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_reservation_record *) 0)->epoch) == sizeof(uint32_t),
	       "RCCC reservation epoch width");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, reserved_count) == 48u,
	       "RCCC reservation reserved-count offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, committed_count) == 52u,
	       "RCCC reservation committed-count offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, completed_count) == 56u,
	       "RCCC reservation completed-count offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, state) == 60u,
	       "RCCC reservation state offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_reservation_record *) 0)->state) == sizeof(uint8_t),
	       "RCCC reservation state width");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, packet_class) == 61u,
	       "RCCC reservation packet-class offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_reservation_record, reserved_accepted) == 62u,
	       "RCCC reservation reserved-accepted offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_reservation_record *) 0)->reserved_accepted) == sizeof(uint16_t),
	       "RCCC reservation reserved-accepted width");
_Static_assert(sizeof(struct opx_flow_control_rccc_reservation_record) == L2_CACHE_LINE_SIZE,
	       "RCCC reservation record must occupy one L2 cache line");
_Static_assert(_Alignof(struct opx_flow_control_rccc_reservation_record) == L2_CACHE_LINE_SIZE,
	       "RCCC reservation record cache-line alignment");

_Static_assert(sizeof(struct opx_flow_control_rccc_tx_state) == OPX_FLOW_CONTROL_MODE_STATE_BYTES,
	       "RCCC TX state size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_tx_state) <= OPX_FLOW_CONTROL_STATE_ALIGNMENT,
	       "RCCC TX state alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, available_credits) == 12u,
	       "RCCC TX available-credit offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, flags) == 27u, "RCCC TX flags offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, request_retry) == 32u, "RCCC TX request-retry offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, activity_time_ns) == 56u,
	       "RCCC TX activity-time offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, fault_contained) == 80u,
	       "RCCC TX fault-contained offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, flow_key) == 96u, "RCCC TX flow-key offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_tx_state, reserved_requested) == 92u,
	       "RCCC TX reserved-requested offset");
_Static_assert(sizeof(((struct opx_flow_control_rccc_tx_state *) 0)->flow_key) ==
		       sizeof(union fi_opx_reliability_service_flow_key),
	       "RCCC TX flow-key width");

_Static_assert(sizeof(struct opx_flow_control_rccc_rx_state) == OPX_FLOW_CONTROL_MODE_STATE_BYTES,
	       "RCCC RX state size");
_Static_assert(_Alignof(struct opx_flow_control_rccc_rx_state) <= OPX_FLOW_CONTROL_STATE_ALIGNMENT,
	       "RCCC RX state alignment");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, scheduler) == 32u, "RCCC RX scheduler-reference offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, permission_contribution) == 48u,
	       "RCCC RX contribution offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, epoch_snapshot) == 72u, "RCCC RX epoch-snapshot offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, request_generation) == 76u,
	       "RCCC RX request-generation offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, decision_slot) == 80u, "RCCC RX decision-slot offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, reject_slot) == 84u, "RCCC RX reject-slot offset");
_Static_assert(offsetof(struct opx_flow_control_rccc_rx_state, pending_epoch) == 104u, "RCCC RX pending-epoch offset");

#endif
