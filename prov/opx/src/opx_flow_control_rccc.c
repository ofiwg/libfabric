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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <rdma/fi_errno.h>

#include <ofi_mem.h>

#include "rdma/opx/opx_flow_control_rccc_private.h"
#include "rdma/opx/fi_opx_endpoint.h"

int __attribute__((cold)) opx_flow_control_reservation_pool_create(struct ofi_bufpool **pool)
{
	struct ofi_bufpool_attr attr = {
		.size	   = sizeof(struct opx_flow_control_rccc_reservation_record),
		.alignment = L2_CACHE_LINE_SIZE,
		.max_cnt   = OPX_FLOW_CONTROL_RCCC_RESERVATION_POOL_CAPACITY,
		.chunk_cnt = OPX_FLOW_CONTROL_RCCC_RESERVATION_POOL_CAPACITY,
		.flags	   = OFI_BUFPOOL_NO_ZERO,
	};
	struct ofi_bufpool *created_pool = NULL;
	int		    ret;

	if (pool == NULL || *pool != NULL) {
		return -FI_EINVAL;
	}

	ret = ofi_bufpool_create_attr(&attr, &created_pool);
	if (ret != FI_SUCCESS) {
		return ret;
	}

	ret = ofi_bufpool_grow(created_pool);
	if (ret != FI_SUCCESS || created_pool->entry_cnt != OPX_FLOW_CONTROL_RCCC_RESERVATION_POOL_CAPACITY ||
	    created_pool->region_cnt != 1u || created_pool->attr.size != attr.size ||
	    created_pool->attr.alignment != attr.alignment || created_pool->attr.max_cnt != attr.max_cnt ||
	    created_pool->attr.chunk_cnt != attr.chunk_cnt || created_pool->attr.flags != attr.flags) {
		ofi_bufpool_destroy(created_pool);
		return ret != FI_SUCCESS ? ret : -FI_EIO;
	}

	*pool = created_pool;
	return FI_SUCCESS;
}

int __attribute__((cold)) opx_flow_control_reservation_pool_capacity_verify(
	const struct ofi_bufpool *pool, const struct fi_opx_reliability_service *service, size_t *capacity_out)
{
	size_t derived_capacity;

	if (pool == NULL || service == NULL || capacity_out == NULL || service->tx.replay_pool == NULL ||
	    service->tx.replay_iov_pool == NULL) {
		return -FI_EINVAL;
	}
	derived_capacity =
		(size_t) service->tx.replay_pool->attr.max_cnt + (size_t) service->tx.replay_iov_pool->attr.max_cnt;
	if (derived_capacity != OPX_FLOW_CONTROL_RCCC_RESERVATION_POOL_CAPACITY ||
	    pool->attr.max_cnt != derived_capacity || pool->entry_cnt != derived_capacity) {
		return -FI_EINVAL;
	}
	*capacity_out = derived_capacity;
	return FI_SUCCESS;
}

void __attribute__((cold)) opx_flow_control_reservation_pool_destroy(struct ofi_bufpool **pool)
{
	size_t		    free_count = 0u;
	struct slist_entry *entry;

	if (pool == NULL || *pool == NULL) {
		return;
	}

	/* The generic use counter is debug-only, so count returned records to
	 * enforce the endpoint teardown invariant in optimized builds too. */
	for (entry = (*pool)->free_list.entries.head; entry != NULL; entry = entry->next) {
		if (++free_count > (*pool)->entry_cnt) {
			abort();
		}
	}
	if (free_count != (*pool)->entry_cnt) {
		abort();
	}

	ofi_bufpool_destroy(*pool);
	*pool = NULL;
}

#define OPX_FLOW_CONTROL_RCCC_SHARED_READY_TIMEOUT_NS UINT64_C(5000000000)
#define OPX_FLOW_CONTROL_RCCC_SHARED_POLL_NS	      UINT64_C(1000000)

static void opx_flow_control_rccc_shared_release_fold(struct opx_flow_control_rccc_scheduler *scheduler);
static bool opx_flow_control_rccc_shared_tracked_reserve(struct opx_flow_control_rccc_scheduler *scheduler);

static struct opx_flow_control_rccc_scheduler_flow_owner *
opx_flow_control_rccc_shared_flow_owner(struct opx_flow_control_rccc_scheduler *scheduler, uint32_t slot)
{
	struct opx_flow_control_rccc_scheduler_flow *flows =
		(struct opx_flow_control_rccc_scheduler_flow *) (scheduler->shared_base +
								 scheduler->shared_header->flows_offset);

	return (struct opx_flow_control_rccc_scheduler_flow_owner *) flows[slot].core;
}

static bool opx_flow_control_rccc_shared_flow_claim(struct opx_flow_control_rccc_scheduler	    *scheduler,
						    const union fi_opx_reliability_service_flow_key *flow_key,
						    uint32_t *slot_out, uint64_t *claim_out)
{
	const uint64_t claim = ((uint64_t) (scheduler->shared_member_index + 1u) << 32) |
			       (scheduler->allocation_generation & UINT64_C(0xFFFFFFFF));
	uint32_t slot;

	for (slot = 0u; slot < scheduler->profile.tracked_capacity; ++slot) {
		struct opx_flow_control_rccc_scheduler_flow_owner *owner =
			opx_flow_control_rccc_shared_flow_owner(scheduler, slot);
		uint64_t expected = 0u;

		if (atomic_compare_exchange_strong_explicit(&owner->claim, &expected, claim, memory_order_acq_rel,
							    memory_order_relaxed)) {
			owner->flow_key_prefix = flow_key->qw_prefix;
			owner->flow_key_suffix = flow_key->dw_suffix;
			*slot_out	       = slot;
			*claim_out	       = claim;
			return true;
		}
	}
	return false;
}

static void opx_flow_control_rccc_shared_flow_claims_release(struct opx_flow_control_rccc_scheduler *scheduler)
{
	const uint64_t member_bits = (uint64_t) (scheduler->shared_member_index + 1u) << 32;
	uint32_t       slot;

	for (slot = 0u; slot < scheduler->profile.tracked_capacity; ++slot) {
		struct opx_flow_control_rccc_scheduler_flow_owner *owner =
			opx_flow_control_rccc_shared_flow_owner(scheduler, slot);

		if ((atomic_load_explicit(&owner->claim, memory_order_relaxed) & UINT64_C(0xFFFFFFFF00000000)) ==
		    member_bits) {
			owner->flow_key_prefix = 0u;
			owner->flow_key_suffix = 0u;
			atomic_store_explicit(&owner->claim, 0u, memory_order_release);
		}
	}
}

static uint64_t opx_flow_control_rccc_monotonic_ns(void)
{
	struct timespec now;

	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t) now.tv_sec * UINT64_C(1000000000) + (uint64_t) now.tv_nsec;
}

static void opx_flow_control_rccc_shared_header_fill(struct opx_flow_control_rccc_scheduler_header	   *header,
						     const struct opx_flow_control_rccc_scheduler_geometry *geometry,
						     const struct opx_flow_control_rccc_scheduler_profile  *profile,
						     const struct opx_flow_control_rccc_tx_profile	   *tx_profile,
						     uint32_t member_count)
{
	header->magic		  = FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_MAGIC;
	header->abi_version	  = FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_VERSION;
	header->endian		  = FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ENDIAN_LITTLE;
	header->alignment	  = FI_OPX_FLOW_CONTROL_SCHEDULER_ABI_ALIGNMENT;
	header->header_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_HEADER_BYTES;
	header->member_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_MEMBER_BYTES;
	header->decision_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_DECISION_BYTES;
	header->reject_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_REJECT_BYTES;
	header->flow_core_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_CORE_BYTES;
	header->accumulator_bytes = FI_OPX_FLOW_CONTROL_SCHEDULER_ACCUMULATOR_BYTES;
	header->flow_bytes	  = FI_OPX_FLOW_CONTROL_SCHEDULER_FLOW_BYTES;
	header->region_length	  = geometry->mapping_length;

	header->member_count	     = member_count;
	header->configured_inventory = 0u;
	header->ffirst		     = tx_profile->ffirst;
	header->decision_capacity    = profile->decision_capacity;
	header->reject_capacity	     = profile->reject_capacity;
	header->tracked_capacity     = profile->tracked_capacity;
	header->eligible_capacity    = profile->tracked_capacity;
	header->fmax		     = tx_profile->fmax;

	header->members_offset	       = geometry->members_offset;
	header->decisions_offset       = geometry->decisions_offset;
	header->rejects_offset	       = geometry->rejects_offset;
	header->flows_offset	       = geometry->flows_offset;
	header->members_region_bytes   = geometry->members_region_bytes;
	header->decisions_region_bytes = geometry->decisions_region_bytes;
	header->rejects_region_bytes   = geometry->rejects_region_bytes;
	header->flows_region_bytes     = geometry->flows_region_bytes;

	header->initialization_generation = 1u;
	header->membership_generation	  = 1u;
}

static struct opx_flow_control_rccc_scheduler_member *
opx_flow_control_rccc_shared_member(struct opx_flow_control_rccc_scheduler *scheduler, uint32_t index)
{
	return (struct opx_flow_control_rccc_scheduler_member *) (scheduler->shared_base +
								  scheduler->shared_header->members_offset) +
	       index;
}

/* Create-or-attach for the per-HFI-context shared scheduler mapping.  The
 * O_EXCL winner sizes and fills the region, then release-publishes READY;
 * losers map and bounded-wait for READY before validating.  This does not
 * rely on the driver's subctxt-0-first ordering, so any member of the
 * sharing group may arrive first.  Geometry and profile values must match
 * the local env profile exactly - a mismatch across members fails the
 * attach closed.  No member-death recovery exists: a slot found JOINED at
 * attach, or a creator that dies before READY, fails closed. */
int __attribute__((cold))
opx_flow_control_rccc_scheduler_shared_attach(struct opx_flow_control_rccc_scheduler *scheduler,
					      const char *job_key_str, uint32_t hfi_unit, uint32_t context_number,
					      uint32_t subctxt, uint32_t subctxt_cnt)
{
	const struct opx_flow_control_rccc_scheduler_profile *profile;
	const struct opx_flow_control_rccc_tx_profile	     *tx_profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_scheduler_geometry	      geometry;
	struct opx_flow_control_rccc_scheduler_member	     *member;
	struct opx_flow_control_rccc_scheduler_header	     *header;
	uint8_t						     *base;
	uint64_t					      deadline_ns;
	bool						      creator;
	int						      fd;
	int						      written;

	if (scheduler == NULL || tx_profile == NULL || job_key_str == NULL || scheduler->shared_header != NULL ||
	    subctxt_cnt < 2u || subctxt_cnt > FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS || subctxt >= subctxt_cnt) {
		return -FI_EINVAL;
	}
	profile = &scheduler->profile;
	/* The context capacity is statically partitioned: each member grants
	 * only against its 1/member_count share.  Members run independent
	 * DRR passes, so an unpartitioned shared pool splits between flows
	 * that each need a full admission batch atomically and can never
	 * consolidate - an absorbing livelock (proven at max_outstanding=64,
	 * 2 members, 64-packet TID batches: the pool wedges 62/2).  A share
	 * below fmax could still leave a single batch ungrantable, so the
	 * capacity-derived default intentionally fails shared attach closed. */
	const uint64_t member_share = profile->max_outstanding_permissions / subctxt_cnt;

	if (member_share < tx_profile->fmax) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC shared scheduler geometry invalid: max_outstanding %lu / %u members = %lu per-member share is below fmax %u\n",
			(unsigned long) profile->max_outstanding_permissions, subctxt_cnt, (unsigned long) member_share,
			tx_profile->fmax);
		return -FI_EINVAL;
	}
	if (opx_flow_control_rccc_scheduler_geometry_compute(profile->decision_capacity, profile->reject_capacity,
							     profile->tracked_capacity,
							     &geometry) != OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK) {
		return -FI_EINVAL;
	}
	written = snprintf(scheduler->shared_name, sizeof(scheduler->shared_name), "/opx-fc-rccc-%s-u%u-c%u",
			   job_key_str, hfi_unit, context_number);
	if (written <= 0 || (size_t) written >= sizeof(scheduler->shared_name)) {
		return -FI_EINVAL;
	}

	fd	= shm_open(scheduler->shared_name, O_RDWR | O_CREAT | O_EXCL, 0600);
	creator = fd >= 0;
	if (!creator) {
		if (errno != EEXIST) {
			return -FI_EIO;
		}
		fd = shm_open(scheduler->shared_name, O_RDWR, 0600);
		if (fd < 0) {
			return -FI_EIO;
		}
	} else if (ftruncate(fd, (off_t) geometry.mapping_length) != 0) {
		close(fd);
		shm_unlink(scheduler->shared_name);
		return -FI_EIO;
	}

	if (!creator) {
		struct stat segment_stat;

		deadline_ns = opx_flow_control_rccc_monotonic_ns() + OPX_FLOW_CONTROL_RCCC_SHARED_READY_TIMEOUT_NS;
		while (fstat(fd, &segment_stat) == 0 && (size_t) segment_stat.st_size < geometry.mapping_length) {
			if (opx_flow_control_rccc_monotonic_ns() >= deadline_ns) {
				close(fd);
				return -FI_ETIMEDOUT;
			}
			usleep(OPX_FLOW_CONTROL_RCCC_SHARED_POLL_NS / 1000u);
		}
	}

	base = mmap(NULL, geometry.mapping_length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	close(fd);
	if (base == MAP_FAILED) {
		if (creator) {
			shm_unlink(scheduler->shared_name);
		}
		return -FI_ENOMEM;
	}
	header = (struct opx_flow_control_rccc_scheduler_header *) base;

	if (creator) {
		opx_flow_control_rccc_shared_header_fill(header, &geometry, profile, tx_profile, subctxt_cnt);
		atomic_store_explicit(&header->initialization_state, OPX_FLOW_CONTROL_RCCC_SCHEDULER_READY,
				      memory_order_release);
	} else {
		deadline_ns = opx_flow_control_rccc_monotonic_ns() + OPX_FLOW_CONTROL_RCCC_SHARED_READY_TIMEOUT_NS;
		while (atomic_load_explicit(&header->initialization_state, memory_order_acquire) !=
		       OPX_FLOW_CONTROL_RCCC_SCHEDULER_READY) {
			if (opx_flow_control_rccc_monotonic_ns() >= deadline_ns) {
				munmap(base, geometry.mapping_length);
				return -FI_ETIMEDOUT;
			}
			usleep(OPX_FLOW_CONTROL_RCCC_SHARED_POLL_NS / 1000u);
		}
		if (opx_flow_control_rccc_scheduler_mapping_validate(base, geometry.mapping_length, header,
								     &geometry) !=
			    OPX_FLOW_CONTROL_RCCC_SCHEDULER_MAPPING_OK ||
		    header->member_count != subctxt_cnt || header->ffirst != tx_profile->ffirst ||
		    header->fmax != tx_profile->fmax || header->tracked_capacity != profile->tracked_capacity) {
			munmap(base, geometry.mapping_length);
			return -FI_EINVAL;
		}
	}

	scheduler->shared_base			       = base;
	scheduler->shared_header		       = header;
	scheduler->shared_ledger		       = &header->ledger;
	scheduler->shared_telemetry		       = &header->telemetry;
	scheduler->telemetry_sample_interval_ns	       = profile->telemetry_health_deadline_ns / 10u;
	scheduler->shared_length		       = geometry.mapping_length;
	scheduler->shared_member_index		       = subctxt;
	scheduler->shared_member_count		       = subctxt_cnt;
	scheduler->profile.max_outstanding_permissions = member_share;

	member = opx_flow_control_rccc_shared_member(scheduler, subctxt);
	if (member->identity.state == OPX_FLOW_CONTROL_RCCC_MEMBER_JOINED) {
		scheduler->shared_base	    = NULL;
		scheduler->shared_header    = NULL;
		scheduler->shared_ledger    = NULL;
		scheduler->shared_telemetry = NULL;
		munmap(base, geometry.mapping_length);
		return -FI_EBUSY;
	}
	member->identity.pid		     = (uint64_t) getpid();
	member->identity.process_start_token = opx_flow_control_rccc_monotonic_ns();
	member->identity.member_generation++;
	member->identity.subctxt = subctxt;
	member->identity.state	 = OPX_FLOW_CONTROL_RCCC_MEMBER_JOINED;
	atomic_fetch_add_explicit(&header->ledger.attached_count, 1u, memory_order_acq_rel);
	return FI_SUCCESS;
}

static void __attribute__((cold))
opx_flow_control_rccc_scheduler_shared_detach(struct opx_flow_control_rccc_scheduler *scheduler)
{
	struct opx_flow_control_rccc_scheduler_member *member;
	uint32_t				       remaining;

	if (scheduler->shared_header == NULL) {
		return;
	}
	member		       = opx_flow_control_rccc_shared_member(scheduler, scheduler->shared_member_index);
	member->identity.state = OPX_FLOW_CONTROL_RCCC_MEMBER_LEFT;
	opx_flow_control_rccc_shared_flow_claims_release(scheduler);
	if (scheduler->tracked_used != 0u) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->tracked_used, memory_order_relaxed) >=
		       scheduler->tracked_used);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->tracked_used, scheduler->tracked_used,
					  memory_order_acq_rel);
	}
	/* Graceful teardown bulk-releases this member's still-BOUND flows;
	 * ungraceful member death remains the shared mapping's pre-existing
	 * no-recovery exposure. */
	if (scheduler->local_bound_flows != 0u) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->active_bound_flows, memory_order_relaxed) >=
		       scheduler->local_bound_flows);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->active_bound_flows, scheduler->local_bound_flows,
					  memory_order_acq_rel);
	}
	if (scheduler->local_demand_active_flows != 0u) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->demand_active_flows, memory_order_relaxed) >=
		       scheduler->local_demand_active_flows);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->demand_active_flows,
					  scheduler->local_demand_active_flows, memory_order_acq_rel);
	}
	if (scheduler->local_hot_flows != 0u) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->hot_flows, memory_order_relaxed) >=
		       scheduler->local_hot_flows);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->hot_flows, scheduler->local_hot_flows,
					  memory_order_acq_rel);
	}
	remaining = atomic_fetch_sub_explicit(&scheduler->shared_ledger->attached_count, 1u, memory_order_acq_rel);
	munmap(scheduler->shared_base, scheduler->shared_length);
	if (remaining == 1u) {
		shm_unlink(scheduler->shared_name);
	}
	scheduler->shared_base	    = NULL;
	scheduler->shared_header    = NULL;
	scheduler->shared_ledger    = NULL;
	scheduler->shared_telemetry = NULL;
	scheduler->shared_length    = 0u;
}

int __attribute__((cold)) opx_flow_control_rccc_scheduler_create(struct opx_flow_control_rccc_scheduler **scheduler,
								 uint32_t				  rcvhdrq_cnt)
{
	const struct opx_flow_control_rccc_scheduler_profile *profile = opx_flow_control_rccc_scheduler_profile_get();
	const struct opx_flow_control_rccc_tx_profile	     *tx_profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_scheduler		     *created;
	uint64_t					      base_margin;
	uint64_t					      control_headroom;
	uint64_t					      drr_quantum;
	uint64_t					      high_water;
	uint64_t					      low_water;
	uint64_t					      max_outstanding;
	uint64_t					      physical_capacity;
	uint64_t					      usable_capacity;
	const uint32_t					      ffirst = tx_profile != NULL ? tx_profile->ffirst : 0u;

	if (scheduler == NULL || *scheduler != NULL) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: output must be nonnull and empty (output=%p current=%p)\n",
			(void *) scheduler, scheduler != NULL ? (void *) *scheduler : NULL);
		return -FI_EINVAL;
	}
	if (profile == NULL || tx_profile == NULL) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: process profiles must be initialized (scheduler_profile=%p tx_profile=%p)\n",
			(const void *) profile, (const void *) tx_profile);
		return -FI_EINVAL;
	}
	if (ffirst == 0u || profile->decision_capacity < ffirst || profile->reject_capacity < ffirst ||
	    profile->tracked_capacity == 0u || profile->quarantine_deadline_ns == 0u ||
	    profile->telemetry_health_deadline_ns == 0u || profile->minimum_packet_rate_q16_per_ns == 0u ||
	    profile->maximum_elapsed_ns == 0u || profile->grant_emits == 0u || profile->window_cap_min < 8u ||
	    profile->window_epoch_ns == 0u || profile->hot_entry_streak == 0u ||
	    profile->hot_entry_streak > UINT8_MAX || profile->hot_exit_streak == 0u ||
	    profile->hot_exit_streak > UINT8_MAX || profile->hot_absolute_floor == 0u ||
	    profile->hot_consumed_floor == 0u || profile->storm_request_floor == 0u ||
	    profile->idle_decay_streak == 0u || profile->idle_decay_streak > UINT8_MAX) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: ffirst=%u decision=%u reject=%u tracked=%u grant_emits=%u quarantine=%lu telemetry=%lu packet_rate=%lu elapsed=%lu window_cap_min=%u epoch_ns=%lu hot_entry=%u hot_exit=%u hot_floor=%u idle_decay=%u\n",
			ffirst, profile->decision_capacity, profile->reject_capacity, profile->tracked_capacity,
			profile->grant_emits, (unsigned long) profile->quarantine_deadline_ns,
			(unsigned long) profile->telemetry_health_deadline_ns,
			(unsigned long) profile->minimum_packet_rate_q16_per_ns,
			(unsigned long) profile->maximum_elapsed_ns, profile->window_cap_min,
			(unsigned long) profile->window_epoch_ns, profile->hot_entry_streak, profile->hot_exit_streak,
			profile->hot_absolute_floor, profile->idle_decay_streak);
		return -FI_EINVAL;
	}
	if (profile->window_cap_min > tx_profile->fmax) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: window_cap_min %u exceeds fmax %u\n",
			profile->window_cap_min, tx_profile->fmax);
		return -FI_EINVAL;
	}
	if ((uint64_t) profile->capacity_guard > (uint64_t) rcvhdrq_cnt / 2u) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: capacity_guard %u exceeds rcvhdrq_cnt/2 %u\n",
			profile->capacity_guard, rcvhdrq_cnt / 2u);
		return -FI_EINVAL;
	}

	/* T101 uses the WS01-qualified guard band because the physical ring
	 * stopped 7-12 entries before elemcnt under hard overflow. */
	physical_capacity = (uint64_t) rcvhdrq_cnt - profile->capacity_guard;
	base_margin	  = (physical_capacity + 9u) / 10u;
	if (base_margin < 8u) {
		base_margin = 8u;
	}
	control_headroom = opx_flow_control_rccc_control_headroom_get();
	if (base_margin > physical_capacity || control_headroom > physical_capacity - base_margin) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: base_margin %lu + control_headroom %lu exceeds physical_capacity %lu\n",
			(unsigned long) base_margin, (unsigned long) control_headroom,
			(unsigned long) physical_capacity);
		return -FI_EINVAL;
	}
	usable_capacity = physical_capacity - base_margin - control_headroom;
	if (usable_capacity < 64u) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: usable_capacity %lu is below 64 (physical=%lu margin=%lu headroom=%lu)\n",
			(unsigned long) usable_capacity, (unsigned long) physical_capacity, (unsigned long) base_margin,
			(unsigned long) control_headroom);
		return -FI_EINVAL;
	}
	if (profile->window_cap_min > usable_capacity) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: window_cap_min %u exceeds usable_capacity %lu\n",
			profile->window_cap_min, (unsigned long) usable_capacity);
		return -FI_EINVAL;
	}
	if (profile->duty_drain_floor > usable_capacity) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: duty_drain_floor %lu exceeds usable_capacity %lu\n",
			(unsigned long) profile->duty_drain_floor, (unsigned long) usable_capacity);
		return -FI_EINVAL;
	}
	low_water  = usable_capacity / 2u;
	high_water = (usable_capacity * 3u) / 4u;
	if (low_water >= high_water || high_water > usable_capacity) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: low %lu < high %lu <= usable %lu is false\n",
			(unsigned long) low_water, (unsigned long) high_water, (unsigned long) usable_capacity);
		return -FI_EINVAL;
	}
	if (profile->max_outstanding_permissions == 0u) {
		/* The permission ledger is a windowing resource, not the
		 * overflow bound: occupancy watermarks and the rate curve own
		 * overflow protection (proven zero-overflow with the ledger at
		 * both 659 and 65536), while a ring-derived ledger starves
		 * multi-flow windowing - at physical-high-headroom (~659 for a
		 * 2048 ring) np=4 Allreduce ran ~600x slower than at 65536
		 * because no peer flow could hold a standing window against
		 * fmax.  Size the ledger for flow concurrency instead: one
		 * full window per trackable flow, which also keeps the
		 * per-member share >= fmax after sharing-attach division at
		 * every member count up to the 8-member maximum.  The product
		 * can exceed the 24-bit serial half-range at large tracked
		 * sizings; a derived (not user-set) value clamps rather than
		 * failing enable, since the ledger only needs enough headroom
		 * for concurrent windows, not one per lifetime flow.  At high
		 * non-shared fan-in, the grant pass applies the receive-ring
		 * runtime backstop described there. */
		max_outstanding = (uint64_t) profile->tracked_capacity * tx_profile->fmax;
		if (max_outstanding >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
			max_outstanding = (uint64_t) OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE - 1u;
		}
	} else {
		max_outstanding = profile->max_outstanding_permissions;
	}
	if (max_outstanding == 0u || max_outstanding >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: 1 <= max_outstanding %lu < serial_half_range %u is false\n",
			(unsigned long) max_outstanding, OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE);
		return -FI_EINVAL;
	}
	drr_quantum = profile->drr_quantum;
	if (drr_quantum == 0u) {
		drr_quantum = max_outstanding / 4u;
		if (drr_quantum < 64u) {
			drr_quantum = 64u;
		}
	}
	if (drr_quantum > max_outstanding) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC scheduler constraint invalid: drr_quantum %lu exceeds max_outstanding %lu\n",
			(unsigned long) drr_quantum, (unsigned long) max_outstanding);
		return -FI_EINVAL;
	}

	/* The scheduler struct is cacheline-aligned for hot/cold line
	 * separation, which calloc does not guarantee. */
	if (posix_memalign((void **) &created, L2_CACHE_LINE_SIZE, sizeof(*created)) != 0) {
		return -FI_ENOMEM;
	}
	memset(created, 0, sizeof(*created));
	created->decisions = calloc(profile->decision_capacity, sizeof(*created->decisions));
	created->rejects   = calloc(profile->reject_capacity, sizeof(*created->rejects));
	if (created->decisions == NULL || created->rejects == NULL) {
		free(created->decisions);
		free(created->rejects);
		free(created);
		return -FI_ENOMEM;
	}
	created->profile			     = *profile;
	created->profile.max_outstanding_permissions = max_outstanding;
	created->profile.drr_quantum		     = drr_quantum;
	created->profile.duty_drain_floor =
		profile->duty_drain_floor != 0u ? created->profile.duty_drain_floor : usable_capacity / 4u;
	created->low_water	       = (uint32_t) low_water;
	created->high_water	       = (uint32_t) high_water;
	created->physical_capacity     = (uint32_t) physical_capacity;
	created->usable_capacity       = (uint32_t) usable_capacity;
	created->allocation_generation = 1u;
	created->decision_free_head    = OPX_FLOW_CONTROL_RCCC_DECISION_FREE_END;
	FI_INFO(fi_opx_global.prov, FI_LOG_EP_CTRL,
		"RCCC scheduler geometry: rcvhdrq=%u guard=%u physical=%u margin=%lu headroom=%lu usable=%u low=%u high=%u max_outstanding=%lu drr_quantum=%lu\n",
		rcvhdrq_cnt, profile->capacity_guard, created->physical_capacity, (unsigned long) base_margin,
		(unsigned long) control_headroom, created->usable_capacity, created->low_water, created->high_water,
		(unsigned long) created->profile.max_outstanding_permissions,
		(unsigned long) created->profile.drr_quantum);
	*scheduler = created;
	return FI_SUCCESS;
}

void opx_flow_control_rccc_scheduler_occupancy_set(struct opx_flow_control_rccc_scheduler *scheduler,
						   uint32_t occupancy, bool drain_maxed, uint64_t now_ns)
{
	scheduler->occupancy		     = occupancy;
	scheduler->occupancy_sample_ns	     = now_ns;
	scheduler->occupancy_sampling_active = true;
	if (drain_maxed) {
		if (scheduler->maxed_drain_streak != UINT8_MAX) {
			scheduler->maxed_drain_streak++;
		}
		scheduler->quiet_drain_streak = 0u;
	} else {
		if (scheduler->quiet_drain_streak != UINT8_MAX) {
			scheduler->quiet_drain_streak++;
		}
		scheduler->maxed_drain_streak = 0u;
	}
}

/* Caller must hold the RHQ hardware context lock: due-ness gates the
 * lock-held backlog scan so the whole sharing group performs at most one
 * scan per cadence interval, and the CLOCK_MONOTONIC read (comparable
 * across member processes, unlike the per-process cycle-timer ns base)
 * returned through mono_now_out is reused by the publish. */
bool opx_flow_control_rccc_shared_sample_due(struct opx_flow_control_rccc_scheduler *scheduler, uint64_t *mono_now_out)
{
	if (scheduler->shared_telemetry == NULL) {
		return false;
	}
	const uint64_t sample =
		atomic_load_explicit(&scheduler->shared_telemetry->occupancy_sample, memory_order_relaxed);
	const uint64_t mono_now = opx_flow_control_rccc_monotonic_ns();

	*mono_now_out = mono_now;
	if ((sample & OPX_FLOW_CONTROL_RCCC_TELEMETRY_VALID) &&
	    opx_flow_control_rccc_telemetry_age_ns(sample, mono_now) < scheduler->telemetry_sample_interval_ns) {
		return false;
	}
	return true;
}

void opx_flow_control_rccc_shared_sample_publish(struct opx_flow_control_rccc_scheduler *scheduler, uint32_t backlog,
						 uint64_t mono_now)
{
	atomic_store_explicit(&scheduler->shared_telemetry->occupancy_sample,
			      opx_flow_control_rccc_telemetry_pack(backlog, mono_now), memory_order_relaxed);
}

/* Translates the shared CLOCK_MONOTONIC-stamped sample into this process's
 * sweep time base (sample_ns = sweep_now - age) so the grant pass's
 * telemetry-health deadline check works unchanged; sampling_active latches
 * on the first consumed sample, after which staleness fails closed. */
void opx_flow_control_rccc_shared_occupancy_consume(struct opx_flow_control_rccc_scheduler *scheduler,
						    uint64_t				    sweep_now_ns)
{
	const uint64_t sample =
		atomic_load_explicit(&scheduler->shared_telemetry->occupancy_sample, memory_order_relaxed);

	if (!(sample & OPX_FLOW_CONTROL_RCCC_TELEMETRY_VALID)) {
		return;
	}
	if (sample == scheduler->telemetry_last_sample) {
		return;
	}
	scheduler->telemetry_last_sample = sample;

	const uint64_t age_ns = opx_flow_control_rccc_telemetry_age_ns(sample, opx_flow_control_rccc_monotonic_ns());

	opx_flow_control_rccc_scheduler_occupancy_set(scheduler, opx_flow_control_rccc_telemetry_backlog(sample), false,
						      (age_ns < sweep_now_ns) ? (sweep_now_ns - age_ns) : 0u);
}

void __attribute__((cold)) opx_flow_control_rccc_scheduler_destroy(struct opx_flow_control_rccc_scheduler **scheduler)
{
	if (scheduler == NULL || *scheduler == NULL) {
		return;
	}
	/* Endpoint teardown parallels the reservation-pool live-record
	 * invariant: opx_flow_control_rx_flows_reclaim has already
	 * force-released every flow's remaining contribution, so a nonzero
	 * ledger here means the per-flow contribution sum diverged from the
	 * aggregate (the ledger_reconcile invariant) - fail closed. */
	if ((*scheduler)->outstanding_permissions != 0u) {
		abort();
	}
	opx_flow_control_rccc_shared_release_fold(*scheduler);
	opx_flow_control_rccc_scheduler_shared_detach(*scheduler);
	free((*scheduler)->decisions);
	free((*scheduler)->rejects);
	free(*scheduler);
	*scheduler = NULL;
}

static inline struct opx_flow_control_state *
opx_flow_control_rccc_tx_flow_state(struct fi_opx_reliability_tx_flow *flow)
{
	assert(flow != NULL);
	return (struct opx_flow_control_state *) opx_flow_control_tx_flow_state(flow);
}

static inline struct opx_flow_control_rccc_tx_state *
opx_flow_control_rccc_tx_state(struct fi_opx_reliability_tx_flow *flow)
{
	return (struct opx_flow_control_rccc_tx_state *) opx_flow_control_rccc_tx_flow_state(flow)->mode_state;
}

static inline struct opx_flow_control_state *
opx_flow_control_rccc_rx_flow_state(struct fi_opx_reliability_rx_flow *flow)
{
	assert(flow != NULL);
	return (struct opx_flow_control_state *) opx_flow_control_rx_flow_state(flow);
}

static inline struct opx_flow_control_rccc_rx_state *
opx_flow_control_rccc_rx_state(struct fi_opx_reliability_rx_flow *flow)
{
	return (struct opx_flow_control_rccc_rx_state *) opx_flow_control_rccc_rx_flow_state(flow)->mode_state;
}

static inline uint64_t
opx_flow_control_rccc_reservation_identity_cookie(const struct fi_opx_ep			  *opx_ep,
						  const struct fi_opx_reliability_tx_flow	  *flow,
						  const union fi_opx_reliability_service_flow_key *flow_key)
{
	return (uint64_t) (uintptr_t) opx_ep ^ (uint64_t) (uintptr_t) flow ^ flow_key->qw_prefix ^
	       (uint64_t) flow_key->dw_suffix;
}

static inline void
opx_flow_control_rccc_reservation_validate(const struct opx_flow_control_rccc_reservation_record *record,
					   const struct fi_opx_ep *opx_ep, uint64_t generation)
{
	assert(record != NULL);
	assert(opx_ep != NULL);
	assert(record->opx_ep == opx_ep);
	assert(record->flow_identity != NULL);
	assert(generation != 0u);
	assert(record->generation == generation);
	assert(opx_flow_control_rccc_serial_is_valid(record->epoch));
	assert(record->packet_class < OPX_FLOW_CONTROL_PACKET_CLASS_COUNT);
	assert(record->reserved_count != 0u);
	assert(record->committed_count <= record->reserved_count);
	assert(record->completed_count <= record->committed_count);
	assert((record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD &&
		record->identity_cookie == opx_flow_control_rccc_reservation_identity_cookie(
						   record->opx_ep, record->flow_identity, &record->flow_key)) ||
	       record->state != OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD);
	assert(record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD ||
	       record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED);
}

static inline void opx_flow_control_rccc_committed_append(struct fi_opx_reliability_tx_flow		  *flow,
							  struct opx_flow_control_rccc_reservation_record *record)
{
	assert(flow != NULL);
	assert(record != NULL);
	assert(record->flow_identity == flow);
	assert(record->flow_next == NULL);
	assert((flow->flow_control.committed_head == NULL) == (flow->flow_control.committed_tail == NULL));
	if (flow->flow_control.committed_tail == NULL) {
		flow->flow_control.committed_head = record;
	} else {
		flow->flow_control.committed_tail->flow_next = record;
	}
	flow->flow_control.committed_tail = record;
}

enum opx_flow_control_result opx_flow_control_rccc_reservation_hold(
	struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow_identity,
	const union fi_opx_reliability_service_flow_key *flow_key, struct opx_flow_control_rccc_tx_state *tx_state,
	struct opx_flow_control_rccc_reservation_record *record, enum opx_flow_control_packet_class packet_class,
	uint64_t generation, uint32_t reserved_count)
{
	uint32_t held_credits;

	assert(opx_ep != NULL);
	assert(flow_identity != NULL);
	assert(flow_key != NULL);
	assert(tx_state != NULL);
	assert(record != NULL);
	assert(packet_class >= OPX_FLOW_CONTROL_PACKET_INJECT);
	assert(packet_class < OPX_FLOW_CONTROL_PACKET_CLASS_COUNT);
	assert(opx_flow_control_rccc_serial_is_valid(tx_state->epoch));
	assert(generation != 0u);
	assert(reserved_count != 0u);
	if (tx_state->available_credits < reserved_count) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	assert(tx_state->held_credits <= UINT32_MAX - reserved_count);

	held_credits		  = tx_state->held_credits + reserved_count;
	record->opx_ep		  = opx_ep;
	record->flow_identity	  = flow_identity;
	record->generation	  = generation;
	record->identity_cookie	  = opx_flow_control_rccc_reservation_identity_cookie(opx_ep, flow_identity, flow_key);
	record->flow_key	  = *flow_key;
	record->epoch		  = tx_state->epoch;
	record->reserved_count	  = reserved_count;
	record->committed_count	  = 0u;
	record->completed_count	  = 0u;
	record->state		  = OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD;
	record->packet_class	  = (uint8_t) packet_class;
	record->reserved_accepted = 0u;
	tx_state->available_credits -= reserved_count;
	tx_state->held_credits = held_credits;
	return OPX_FLOW_CONTROL_OK;
}

void opx_flow_control_rccc_reservation_reduce(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_tx_state *tx_state,
					      struct opx_flow_control_rccc_reservation_record *record,
					      uint64_t generation, uint32_t new_count)
{
	uint32_t released_count;

	opx_flow_control_rccc_reservation_validate(record, opx_ep, generation);
	assert(tx_state != NULL);
	assert(tx_state->epoch == record->epoch);
	assert(record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD);
	assert(record->committed_count == 0u);
	assert(record->completed_count == 0u);
	assert(new_count != 0u);
	assert(new_count < record->reserved_count);
	released_count = record->reserved_count - new_count;
	assert(tx_state->held_credits >= released_count);
	assert(tx_state->available_credits <= UINT32_MAX - released_count);

	tx_state->held_credits -= released_count;
	tx_state->available_credits += released_count;
	record->reserved_count = new_count;
}

void opx_flow_control_rccc_reservation_register(struct fi_opx_ep				*opx_ep,
						struct opx_flow_control_rccc_tx_state		*tx_state,
						struct opx_flow_control_rccc_reservation_record *record,
						uint64_t					 generation)
{
	opx_flow_control_rccc_reservation_validate(record, opx_ep, generation);
	assert(tx_state != NULL);
	assert(tx_state->epoch == record->epoch);
	assert(record->state != OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD || record->committed_count == 0u);
	assert(record->state != OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED || record->committed_count != 0u);
	assert(record->committed_count < record->reserved_count);
	assert(tx_state->held_credits != 0u);

	record->state = OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED;
	record->committed_count++;
	tx_state->held_credits--;
	/* Monotonic per epoch: completion never decrements this counter, so the
	 * epoch conservation invariant keeps its original-commit term intact
	 * and an old-epoch completion after rebase cannot corrupt it.  The
	 * 24-bit invariant arithmetic tolerates wrap. */
	tx_state->committed_originals++;
}

void opx_flow_control_rccc_reservation_cancel(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_tx_state *tx_state,
					      struct opx_flow_control_rccc_reservation_record *record,
					      uint64_t					       generation)
{
	opx_flow_control_rccc_reservation_validate(record, opx_ep, generation);
	assert(tx_state != NULL);
	assert(tx_state->epoch == record->epoch);
	assert(record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD);
	assert(record->committed_count == 0u);
	assert(record->completed_count == 0u);
	assert(tx_state->held_credits >= record->reserved_count);
	assert(tx_state->available_credits <= UINT32_MAX - record->reserved_count);

	tx_state->held_credits -= record->reserved_count;
	tx_state->available_credits += record->reserved_count;
}

bool opx_flow_control_rccc_reservation_complete(struct fi_opx_ep				*opx_ep,
						struct opx_flow_control_rccc_reservation_record *record,
						uint64_t					 generation)
{
#ifndef NDEBUG
	opx_flow_control_rccc_reservation_validate(record, opx_ep, generation);
#endif
	assert(record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED);
	assert(record->completed_count < record->committed_count);

	record->completed_count++;
	return record->committed_count == record->reserved_count && record->completed_count == record->committed_count;
}

/* Epoch conservation: every granted permission is in exactly one term.  The
 * check is mod-2^24 because grant is a wrapping serial while the counter
 * terms are plain monotonic counts. */
static inline void opx_flow_control_rccc_tx_invariant_check(const struct opx_flow_control_rccc_tx_state *tx_state)
{
	assert(((tx_state->available_credits + tx_state->held_credits + tx_state->committed_originals +
		 tx_state->retired + tx_state->fault_contained) &
		OPX_FLOW_CONTROL_RCCC_SERIAL_MASK) == tx_state->grant);
}

bool opx_flow_control_rccc_tx_profile_is_valid(const struct opx_flow_control_rccc_tx_profile *profile)
{
	return profile != NULL && profile->ffirst != 0u && profile->ffirst <= profile->fmax &&
	       profile->fmax <= OPX_FLOW_CONTROL_RCCC_FMAX_LIMIT && profile->idle_retire_delay_ns != 0u &&
	       profile->retry_cap_us >= OPX_FLOW_CONTROL_RCCC_RETRY_INITIAL_MIN_US &&
	       opx_flow_control_rccc_retry_initial_is_valid(profile->request_retry_initial_us, profile->retry_cap_us) &&
	       opx_flow_control_rccc_retry_initial_is_valid(profile->retirement_retry_initial_us,
							    profile->retry_cap_us) &&
	       opx_flow_control_rccc_retry_initial_is_valid(profile->reset_retry_initial_us, profile->retry_cap_us);
}

int opx_flow_control_rccc_tx_window_open(struct opx_flow_control_rccc_tx_state	       *tx_state,
					 const struct opx_flow_control_rccc_tx_profile *profile)
{
	assert(tx_state != NULL);
	assert(opx_flow_control_rccc_tx_profile_is_valid(profile));
	if (tx_state->epoch == 0u) {
		tx_state->epoch = 1u;
	}
	tx_state->desired_window	    = (uint16_t) profile->ffirst;
	tx_state->request_retry.delay_us    = profile->request_retry_initial_us;
	tx_state->retirement_retry.delay_us = profile->retirement_retry_initial_us;
	tx_state->reset_retry.delay_us	    = profile->reset_retry_initial_us;
	tx_state->block_reason		    = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
	opx_flow_control_rccc_tx_invariant_check(tx_state);
	return 0;
}

static void opx_flow_control_rccc_tx_window_grow(struct opx_flow_control_rccc_tx_state	       *tx_state,
						 const struct opx_flow_control_rccc_tx_profile *profile)
{
	uint32_t grown = (uint32_t) tx_state->desired_window + profile->ffirst;

	tx_state->desired_window = (uint16_t) (grown < profile->fmax ? grown : profile->fmax);
	tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED;
}

static bool opx_flow_control_rccc_tx_force(struct opx_flow_control_rccc_tx_state *tx_state)
{
	const bool progress_seen =
		(tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT) != 0u ||
		(tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_RETIREMENT_SEEN) != 0u;

	if (!progress_seen || (tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_REARMED) != 0u) {
		return false;
	}
	tx_state->control_deadline_ns = 0u;
	tx_state->request_retry.progress_flags |=
		OPX_FLOW_CONTROL_RCCC_RETRY_FORCE_EMIT | OPX_FLOW_CONTROL_RCCC_RETRY_REARMED;
	return true;
}

static void opx_flow_control_rccc_tx_target_rearm(struct opx_flow_control_rccc_tx_state *tx_state)
{
	const uint32_t pending_demand = (tx_state->target - tx_state->grant) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
	const uint32_t threshold      = MIN(MAX((uint32_t) tx_state->desired_window / 4u, 1u), MAX(pending_demand, 1u));

	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) == 0u ||
	    tx_state->request_retry.active == 0u ||
	    opx_flow_control_rccc_serial_classify(tx_state->target_advertised_checkpoint, tx_state->target) !=
		    OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER ||
	    ((tx_state->target - tx_state->target_advertised_checkpoint) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK) <
		    threshold) {
		return;
	}
	/* Force at most one immediate emission per grant epoch when a
	 * quarter-window, bounded by pending demand, is unadvertised.  Preserve
	 * active/delay so forced emissions advance the existing backoff ramp. */
	if (opx_flow_control_rccc_tx_force(tx_state)) {
		tx_state->target_advertised_checkpoint = tx_state->target;
	}
}

enum opx_flow_control_result opx_flow_control_rccc_tx_demand(struct opx_flow_control_rccc_tx_state	   *tx_state,
							     const struct opx_flow_control_rccc_tx_profile *profile,
							     uint32_t requested_packets)
{
	uint32_t candidate_target;
	bool	 request_was_pending;

	assert(tx_state != NULL);
	assert(profile != NULL);
	assert(requested_packets != 0u);
	assert(requested_packets <= profile->fmax);
	assert(tx_state->desired_window >= profile->ffirst && tx_state->desired_window <= profile->fmax);
	tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING;
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) != 0u) {
		tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_RESET;
		return OPX_FLOW_CONTROL_DEFER;
	}
	if (tx_state->available_credits >= requested_packets) {
		tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
		/* Proactive low-watermark request: when this admission will
		 * drop the remaining credits below half the desired window,
		 * raise the target and arm a CREDIT_REQUEST now so the
		 * grant round trip overlaps the remaining credits instead
		 * of stalling the flow at exhaustion (stop-and-wait was the
		 * measured small-message rate limiter).  Growth must apply
		 * here too: a flow whose refills always arrive ahead of
		 * exhaustion never reaches the shortfall branch below, so
		 * the armed once-per-generation growth would otherwise
		 * never be consumed and the window would pin at Ffirst. */
		if ((tx_state->available_credits - requested_packets) < (uint32_t) (tx_state->desired_window >> 1) &&
		    (tx_state->flags & (OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING |
					OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED)) == 0u) {
			if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED) != 0u) {
				opx_flow_control_rccc_tx_window_grow(tx_state, profile);
			}
			candidate_target = opx_flow_control_rccc_serial_add(tx_state->grant, tx_state->desired_window);
			if (opx_flow_control_rccc_serial_classify(tx_state->target, candidate_target) ==
			    OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
				tx_state->target = candidate_target;
			}
			tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
			tx_state->target_advertised_checkpoint = tx_state->target;
			tx_state->control_deadline_ns	       = 0u;
			tx_state->request_retry.active	       = 0u;
			tx_state->request_retry.progress_flags = 0u;
		}
		opx_flow_control_rccc_tx_target_rearm(tx_state);
		return OPX_FLOW_CONTROL_OK;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) != 0u) {
		/* Rejected-and-parked: the capacity retry timer owns the
		 * cadence.  Advancing target here would trip the rearm's
		 * force-emission against a receiver known to be full.  A
		 * park outliving the fail-open deadline admits ungoverned
		 * (baseline behavior, never starve) while the retry keeps
		 * probing for readmission. */
		tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY;
		if ((tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED) != 0u) {
			return OPX_FLOW_CONTROL_OK;
		}
		return OPX_FLOW_CONTROL_DEFER;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED) != 0u) {
		opx_flow_control_rccc_tx_window_grow(tx_state, profile);
	}
	candidate_target = opx_flow_control_rccc_serial_add(tx_state->grant, tx_state->desired_window);
	if (opx_flow_control_rccc_serial_classify(tx_state->target, candidate_target) ==
	    OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		tx_state->target = candidate_target;
	}
	request_was_pending = (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) != 0u;
	if (!request_was_pending) {
		tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
		tx_state->target_advertised_checkpoint = tx_state->target;
		tx_state->control_deadline_ns	       = 0u;
		tx_state->request_retry.active	       = 0u;
		tx_state->request_retry.progress_flags = 0u;
	}
	opx_flow_control_rccc_tx_target_rearm(tx_state);
	/* A blocked send with no new target information still owes one prompt
	 * refresh per grant epoch; otherwise it inherits the stale capped retry
	 * deadline across collective phases. */
	if (request_was_pending && tx_state->target == tx_state->target_advertised_checkpoint) {
		opx_flow_control_rccc_tx_force(tx_state);
	}
	tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_NO_CREDIT;
	return OPX_FLOW_CONTROL_DEFER;
}

static void opx_flow_control_rccc_tx_idle_retire(struct opx_flow_control_rccc_tx_state *tx_state)
{
	tx_state->retired	    = opx_flow_control_rccc_serial_add(tx_state->retired, tx_state->available_credits);
	tx_state->available_credits = 0u;
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) == 0u) {
		if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING) == 0u) {
			tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING;
			tx_state->control_deadline_ns	  = 0u;
			tx_state->retirement_retry.active = 0u;
		}
	} else {
		/* The advanced retirement is new wire information a pending
		 * request will carry, but the request's deadline can sit at the
		 * backoff cap from an earlier phase; a reawakening flow then
		 * waits out the whole capped period before the receiver learns
		 * anything.  Pull the deadline forward without touching the
		 * ramp or the force epoch, so the prompt emission still
		 * advances the existing backoff. */
		tx_state->control_deadline_ns = 0u;
	}
	opx_flow_control_rccc_tx_invariant_check(tx_state);
}

enum opx_flow_control_rccc_tx_emission
opx_flow_control_rccc_tx_progress(struct opx_flow_control_rccc_tx_state		*tx_state,
				  const struct opx_flow_control_rccc_tx_profile *profile, uint64_t now_ns,
				  struct opx_flow_control_rccc_tx_emission_intent *intent)
{
	assert(tx_state != NULL);
	assert(profile != NULL);
	assert(intent != NULL);
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING) != 0u) {
		tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING;
		tx_state->activity_time_ns = now_ns;
	} else if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) == 0u &&
		   tx_state->available_credits != 0u &&
		   now_ns - tx_state->activity_time_ns >= profile->idle_retire_delay_ns) {
		opx_flow_control_rccc_tx_idle_retire(tx_state);
	}

	intent->epoch		   = tx_state->epoch;
	intent->target		   = tx_state->target;
	intent->retired		   = tx_state->retired;
	intent->request_generation = opx_flow_control_rccc_serial_add(tx_state->request_generation, 1u);
	if (OFI_UNLIKELY((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) != 0u) &&
	    (tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED) == 0u &&
	    now_ns >= tx_state->reset_deadline_ns) {
		tx_state->request_retry.progress_flags |= OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) != 0u) {
		if (tx_state->reset_retry.active == 0u || now_ns >= tx_state->reset_deadline_ns) {
			intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET;
			return OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET;
		}
		intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
		return OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) != 0u &&
	    (tx_state->request_retry.active == 0u || now_ns >= tx_state->control_deadline_ns)) {
		/* A grant or settled retirement proves the receiver received the
		 * pending demand and is progressing it, so another request adds pure
		 * control load to the ring the window protects.  Unsettled retirement
		 * instead makes kind-3 productive and must bypass deferral.  Otherwise,
		 * defer once per progress epoch; a full quiet retry period still
		 * re-emits for request-loss or receiver-failure recovery. */
		if (tx_state->retired == tx_state->retired_acked && tx_state->request_retry.active != 0u &&
		    (tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_FORCE_EMIT) == 0u &&
		    ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT) != 0u ||
		     (tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_RETIREMENT_SEEN) != 0u) &&
		    (tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_PROGRESS_DEFERRED) == 0u) {
			/* Keep GRANT_SINCE_EMIT set so a delayed Capacity Reject for
			 * this request generation remains superseded by the grant. */
			tx_state->request_retry.progress_flags |= OPX_FLOW_CONTROL_RCCC_RETRY_PROGRESS_DEFERRED;
			tx_state->control_deadline_ns =
				now_ns + (uint64_t) tx_state->request_retry.delay_us * UINT64_C(1000);
			intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
			return OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
		}
		intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST;
		return OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING) != 0u &&
	    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) == 0u &&
	    (tx_state->retirement_retry.active == 0u || now_ns >= tx_state->control_deadline_ns)) {
		intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT;
		return OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT;
	}
	intent->kind = OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
	return OPX_FLOW_CONTROL_RCCC_TX_EMIT_NONE;
}

void opx_flow_control_rccc_tx_emitted(struct opx_flow_control_rccc_tx_state	    *tx_state,
				      const struct opx_flow_control_rccc_tx_profile *profile,
				      enum opx_flow_control_rccc_tx_emission kind, uint64_t now_ns)
{
	struct opx_flow_control_rccc_retry *retry;

	assert(tx_state != NULL);
	assert(profile != NULL);
	switch (kind) {
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST:
		tx_state->request_generation = opx_flow_control_rccc_serial_add(tx_state->request_generation, 1u);
		tx_state->target_advertised_checkpoint = tx_state->target;
		tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT;
		/* REARMED survives the forced emission; only a NEWER grant starts
		 * the next force epoch.  UNGOVERNED survives too: only a grant
		 * or RESET_ACK restores governance, not the retry itself. */
		tx_state->request_retry.progress_flags &=
			OPX_FLOW_CONTROL_RCCC_RETRY_REARMED | OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED;
		retry = &tx_state->request_retry;
		break;
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT:
		retry = &tx_state->retirement_retry;
		break;
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET:
		retry = &tx_state->reset_retry;
		break;
	default:
		return;
	}
	if (retry->active == 0u) {
		retry->active = 1u;
	} else {
		/* A fired retry is a new feedback/retry generation; plain send
		 * spinning never reaches this path (at most one window
		 * growth per generation). */
		if (kind == OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST &&
		    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) == 0u) {
			tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED;
		}
		opx_flow_control_rccc_retry_next(retry->delay_us, profile->retry_cap_us, &retry->delay_us);
	}
	if (kind == OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET) {
		tx_state->reset_deadline_ns = now_ns + (uint64_t) retry->delay_us * UINT64_C(1000);
	} else {
		tx_state->control_deadline_ns = now_ns + (uint64_t) retry->delay_us * UINT64_C(1000);
	}
}

static void opx_flow_control_rccc_feedback_count(struct opx_flow_control_rccc_feedback_counters	 *counters,
						 enum opx_flow_control_rccc_serial_classification classification)
{
	if (counters == NULL) {
		return;
	}
	switch (classification) {
	case OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER:
		counters->newer++;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE:
		counters->duplicate++;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_STALE:
		counters->stale++;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS:
		counters->ambiguous++;
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID:
		counters->invalid++;
		break;
	}
}

enum opx_flow_control_rccc_serial_classification
opx_flow_control_rccc_tx_feedback_apply(struct opx_flow_control_rccc_tx_state	      *tx_state,
					const struct opx_flow_control_rccc_tx_profile *profile, uint32_t grant,
					uint32_t retired_acked_value, uint32_t generation,
					struct opx_flow_control_rccc_feedback_counters *counters)
{
	enum opx_flow_control_rccc_serial_classification grant_classification;
	enum opx_flow_control_rccc_serial_classification acked_classification;
	uint32_t					 grant_delta;

	assert(tx_state != NULL);
	assert(profile != NULL);
	/* Grants dropped here while RESETTING are unrecoverable by design
	 * (grants are monotonic, no recall); the receiver's kind-5 handler
	 * force-releases the resulting granted-but-never-applied residue
	 * when it closes the epoch, so this drop is load-bearing for the
	 * RESET convergence contract. */
	if (generation != tx_state->epoch || (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) != 0u) {
		opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
		return OPX_FLOW_CONTROL_RCCC_SERIAL_STALE;
	}

	grant_classification = opx_flow_control_rccc_serial_classify(tx_state->grant, grant);
	opx_flow_control_rccc_feedback_count(counters, grant_classification);
	if (grant_classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		/* A valid-looking grant beyond target is a flow-local
		 * protocol fault and is never clamped or applied. */
		if (opx_flow_control_rccc_serial_classify(tx_state->target, grant) ==
		    OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
			tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_FAULT;
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
			return OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID;
		}
		grant_delta = (grant - tx_state->grant) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		assert(tx_state->available_credits <= UINT32_MAX - grant_delta);
		tx_state->available_credits += grant_delta;
		tx_state->grant = grant;
		tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT |
				   OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED;
		tx_state->request_retry.progress_flags &= (uint8_t) ~(
			OPX_FLOW_CONTROL_RCCC_RETRY_FORCE_EMIT | OPX_FLOW_CONTROL_RCCC_RETRY_REARMED |
			OPX_FLOW_CONTROL_RCCC_RETRY_PROGRESS_DEFERRED | OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED);
		tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED;
		if (tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY ||
		    tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_NO_CREDIT) {
			tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
		}
		if (tx_state->grant == tx_state->target &&
		    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) != 0u) {
			tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
			tx_state->request_retry.active	       = 0u;
			tx_state->request_retry.progress_flags = 0u;
			if (tx_state->retired != tx_state->retired_acked) {
				/* Retirement settlement outlives the request:
				 * re-arm the shared control channel from the
				 * retirement retry state or its retry dies. */
				tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING;
				tx_state->control_deadline_ns	  = 0u;
				tx_state->retirement_retry.active = 0u;
			}
		}
		opx_flow_control_rccc_tx_invariant_check(tx_state);
	} else if (grant_classification != OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE &&
		   grant_classification != OPX_FLOW_CONTROL_RCCC_SERIAL_STALE) {
		return grant_classification;
	}

	acked_classification = opx_flow_control_rccc_serial_classify(tx_state->retired_acked, retired_acked_value);
	if (acked_classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		const uint32_t acked_delta =
			(retired_acked_value - tx_state->retired_acked) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		const uint32_t outstanding =
			(tx_state->retired - tx_state->retired_acked) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;

		if (acked_delta > outstanding) {
			/* Claiming retirement beyond the sent value is a
			 * flow-local protocol fault. */
			tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_FAULT;
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
			return OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID;
		}
		tx_state->retired_acked = retired_acked_value;
		tx_state->request_retry.progress_flags |= OPX_FLOW_CONTROL_RCCC_RETRY_RETIREMENT_SEEN;
		tx_state->request_retry.progress_flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_RETRY_PROGRESS_DEFERRED;
		if (tx_state->retired_acked == tx_state->retired) {
			tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING;
			tx_state->retirement_retry.active = 0u;
		}
	}
	return grant_classification;
}

enum opx_flow_control_result
opx_flow_control_rccc_tx_capacity_reject(struct opx_flow_control_rccc_tx_state	       *tx_state,
					 const struct opx_flow_control_rccc_tx_profile *profile, uint32_t generation,
					 uint64_t now_ns, struct opx_flow_control_rccc_feedback_counters *counters)
{
	enum opx_flow_control_rccc_serial_classification classification;

	assert(tx_state != NULL);
	assert(profile != NULL);
	classification = opx_flow_control_rccc_serial_classify(tx_state->request_generation, generation);
	opx_flow_control_rccc_feedback_count(counters, classification);
	switch (classification) {
	case OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE:
		break;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_STALE:
	case OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER:
		/* Stale rejects answer superseded generations; a reject newer
		 * than any locally emitted generation is dropped. */
		return OPX_FLOW_CONTROL_DEFER;
	case OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS:
	case OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID:
		tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_FAULT;
		return OPX_FLOW_CONTROL_FAULT;
	}
	if ((tx_state->flags &
	     (OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT | OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED |
	      OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING)) != 0u) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	/* Capacity rejects are retryable: the request stays pending and the
	 * retry restarts at the backoff cap - a full receiver is a
	 * job-phase-scale condition, so sub-cap probes only add control
	 * load.  Zeroed progress_flags keep the once-per-progress-epoch
	 * deferral from swallowing the capacity retry.  Retirement rides the
	 * retried kind-3 (value[1]), so no dedicated retirement arming. */
	tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED;
	tx_state->block_reason		       = OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY;
	tx_state->request_retry.delay_us       = profile->retry_cap_us;
	tx_state->request_retry.active	       = 1u;
	tx_state->request_retry.progress_flags = 0u;
	tx_state->control_deadline_ns	       = now_ns + (uint64_t) profile->retry_cap_us * UINT64_C(1000);
	/* reset_deadline_ns is dead storage while parked (RESETTING is
	 * mutually exclusive with the park), so it carries the fail-open
	 * deadline. */
	tx_state->reset_deadline_ns = now_ns + OPX_FLOW_CONTROL_RCCC_UNGOVERNED_DELAY_NS;
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result
opx_flow_control_rccc_tx_reset_begin(struct opx_flow_control_rccc_tx_state	   *tx_state,
				     const struct opx_flow_control_rccc_tx_profile *profile, uint64_t now_ns)
{
	assert(tx_state != NULL);
	assert(opx_flow_control_rccc_tx_profile_is_valid(profile));
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) != 0u) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	assert(tx_state->available_credits == 0u);
	assert(tx_state->held_credits == 0u);
	tx_state->flags &= (uint8_t) ~(
		OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED |
		OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT | OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED);
	tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING;
	tx_state->desired_window	       = (uint16_t) profile->ffirst;
	tx_state->request_retry.active	       = 0u;
	tx_state->request_retry.progress_flags = 0u;
	tx_state->reset_retry.active	       = 0u;
	tx_state->reset_retry.delay_us	       = profile->reset_retry_initial_us;
	tx_state->reset_deadline_ns	       = now_ns;
	tx_state->block_reason		       = OPX_FLOW_CONTROL_RCCC_BLOCK_RESET;
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result opx_flow_control_rccc_tx_reset_ack(struct opx_flow_control_rccc_tx_state	      *tx_state,
								const struct opx_flow_control_rccc_tx_profile *profile,
								uint32_t generation)
{
	assert(tx_state != NULL);
	assert(opx_flow_control_rccc_tx_profile_is_valid(profile));
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING) == 0u || generation != tx_state->epoch) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	assert(tx_state->desired_window == profile->ffirst);
	assert(tx_state->held_credits == 0u);
	tx_state->epoch			       = opx_flow_control_rccc_serial_add(tx_state->epoch, 1u);
	tx_state->grant			       = 0u;
	tx_state->target		       = 0u;
	tx_state->retired		       = 0u;
	tx_state->retired_acked		       = 0u;
	tx_state->available_credits	       = 0u;
	tx_state->committed_originals	       = 0u;
	tx_state->fault_contained	       = 0u;
	tx_state->request_generation	       = 0u;
	tx_state->target_advertised_checkpoint = 0u;
	tx_state->flags &=
		(uint8_t) ~(OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING |
			    OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED);
	tx_state->request_retry.progress_flags = 0u;
	tx_state->reset_retry.active	       = 0u;
	tx_state->block_reason		       = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
	opx_flow_control_rccc_tx_invariant_check(tx_state);
	return OPX_FLOW_CONTROL_OK;
}

static inline bool opx_flow_control_rccc_tx_work_needed(const struct opx_flow_control_rccc_tx_state *tx_state)
{
	/* A capacity-rejected flow always keeps its request pending, so no
	 * separate CAPACITY_REJECTED work-list clause is needed.  A settled
	 * flow with an open granted epoch (nonzero grant, everything retired
	 * and acked) stays listed so the sweep can idle-close it, releasing
	 * the receiver's tracked/decision slots. */
	assert((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) == 0u ||
	       (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) != 0u);
	return (tx_state->flags &
		(OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING |
		 OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING)) != 0u ||
	       tx_state->available_credits != 0u || tx_state->grant != 0u;
}

void opx_flow_control_rccc_tx_work_enqueue(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow)
{
	struct opx_flow_control_rccc_tx_state *tx_state;

	assert(opx_ep != NULL);
	assert(flow != NULL);
	tx_state = opx_flow_control_rccc_tx_state(flow);
	if (!opx_flow_control_rccc_tx_work_needed(tx_state) ||
	    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_ON_WORK_LIST) != 0u) {
		return;
	}
	tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_ON_WORK_LIST;
	flow->flow_control.work_next = NULL;
	if (opx_ep->flow_control_tx_work_tail == NULL) {
		opx_ep->flow_control_tx_work_head = flow;
	} else {
		opx_ep->flow_control_tx_work_tail->flow_control.work_next = flow;
	}
	opx_ep->flow_control_tx_work_tail = flow;
}

enum opx_flow_control_result opx_flow_control_rccc_tx_admit(struct fi_opx_ep		      *opx_ep,
							    struct fi_opx_reliability_tx_flow *flow,
							    uint32_t			       requested_packets)
{
	const struct opx_flow_control_rccc_tx_profile *profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_tx_state	      *tx_state;
	enum opx_flow_control_result		       result;

	assert(opx_ep != NULL);
	assert(flow != NULL);
	assert(profile != NULL);
	/* Demand is advisory: clamp each request to both the transport batch
	 * ceiling and the configured grant window, then re-demand as the
	 * per-packet reservations exhaust that window. */
	const uint32_t batch_request = profile->fmax < OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST ?
					       profile->fmax :
					       OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST;

	if (requested_packets > batch_request) {
		requested_packets = batch_request;
	}
	tx_state = opx_flow_control_rccc_tx_state(flow);
	result	 = opx_flow_control_rccc_tx_demand(tx_state, profile, requested_packets);
	if (result != OPX_FLOW_CONTROL_FAULT && opx_flow_control_rccc_tx_work_needed(tx_state)) {
		opx_flow_control_rccc_tx_work_enqueue(opx_ep, flow);
	}
	return result;
}

static _Atomic uint64_t opx_flow_control_rccc_reservation_generation;

static inline bool
opx_flow_control_rccc_reservation_recyclable(const struct opx_flow_control_rccc_reservation_record *record)
{
	assert(record != NULL);
	assert(record->completed_count <= record->committed_count);
	return record->committed_count == record->reserved_count && record->committed_count == record->completed_count;
}

static int opx_flow_control_rccc_tx_flow_init(struct fi_opx_reliability_tx_flow		      *flow,
					      const union fi_opx_reliability_service_flow_key *key)
{
	const struct opx_flow_control_rccc_tx_profile *profile = opx_flow_control_rccc_tx_profile_get();

	assert(flow != NULL);
	assert(key != NULL);
	/* Reinitializing a flow that still retains committed records would
	 * zero the FIFO roots and orphan live pool records; fail closed. */
	if (flow->flow_control.committed_head != NULL || flow->flow_control.committed_tail != NULL) {
		return -FI_EBUSY;
	}
	struct opx_flow_control_state *flow_state = opx_flow_control_rccc_tx_flow_state(flow);
	flow_state->lifecycle			  = OPX_FLOW_CONTROL_ACTIVE;
	flow_state->fault_class			  = 0;
	flow_state->fault_scope			  = 0;
	flow_state->flags			  = 0;
	flow_state->reserved			  = 0;
	flow_state->transition_reason		  = 0;
	flow_state->transition_time_ns		  = 0;
	/* Zero the whole mode-state overlay before the field inits below:
	 * a flow buffer recycled through the create-error unwind carries
	 * stale (debug: poisoned) contents, and window_open's epoch/counter
	 * logic depends on zeroed fields.  Compound-literal assignment emits
	 * inline stores - no libc memset call. */
	struct opx_flow_control_rccc_tx_state *tx_state = opx_flow_control_rccc_tx_state(flow);
	*tx_state					= (struct opx_flow_control_rccc_tx_state) {0};
	tx_state->flow_key				= *key;
	assert(profile != NULL);
	(void) opx_flow_control_rccc_tx_window_open(tx_state, profile);
	return FI_SUCCESS;
}

int opx_flow_control_tx_flow_init(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow,
				  const union fi_opx_reliability_service_flow_key *key)
{
	assert(opx_ep != NULL);
	return opx_flow_control_rccc_tx_flow_init(flow, key);
}

enum opx_flow_control_result opx_flow_control_rccc_tx_eager_admit(struct fi_opx_ep *opx_ep, uint32_t dlid,
								  uint16_t dst_subctxt_rx, uint8_t tx_index,
								  uint32_t requested_packets)
{
	union fi_opx_reliability_service_flow_key key;
	struct fi_opx_reliability_tx_flow	 *flow;
	void					 *iterator;

	assert(opx_ep != NULL);
	assert(opx_ep->reli_service != NULL);
	if (OFI_UNLIKELY(tx_index >= opx_ep->num_tx_contexts)) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	key.slid	   = OPX_RELIABILITY_ENCODE_LID_TX(opx_ep->rx->self.planes[OPX_PRIMARY_PLANE].lid, tx_index);
	key.src_subctxt_rx = opx_ep->reli_service->subctxt_rx;
	key.dlid	   = OPX_RELIABILITY_ENCODE_LID_TX(dlid, tx_index);
	key.dst_subctxt_rx = dst_subctxt_rx;
	iterator	   = fi_opx_rbt_find(opx_ep->reli_service->tx.tx_flow_rbtree, (void *) &key);
	if (iterator == NULL) {
		flow = opx_reliability_create_tx_flow(opx_ep->reli_service, &key, opx_ep, OPX_FLOW_CONTROL_MODE_RCCC);
		if (OFI_UNLIKELY(flow == NULL)) {
			return OPX_FLOW_CONTROL_FAULT;
		}
	} else {
		flow = fi_opx_rbt_value(opx_ep->reli_service->tx.tx_flow_rbtree, iterator);
	}
	return opx_flow_control_rccc_tx_admit(opx_ep, flow, requested_packets);
}

/* Quiesced committed-FIFO finalizer: at endpoint teardown, accepted feedback
 * will never arrive, so fully ACKed records retained only for cumulative
 * accepted accounting must return to the reservation pool.  Records that are
 * not fully completed stay linked so the pool-destroy live-record invariant
 * still fails closed on a genuinely unquiesced endpoint. */
void opx_flow_control_tx_flow_fini(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow)
{
	struct opx_flow_control_rccc_reservation_record *record;
	struct opx_flow_control_rccc_reservation_record *previous = NULL;

	assert(opx_ep != NULL);
	assert(flow != NULL);
	assert((flow->flow_control.committed_head == NULL) == (flow->flow_control.committed_tail == NULL));
	record = flow->flow_control.committed_head;
	while (record != NULL) {
		struct opx_flow_control_rccc_reservation_record *next = record->flow_next;

		assert(record->flow_identity == flow);
		assert(record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_COMMITTED);
		if (opx_flow_control_rccc_reservation_recyclable(record)) {
			if (previous == NULL) {
				flow->flow_control.committed_head = next;
			} else {
				previous->flow_next = next;
			}
			if (flow->flow_control.committed_tail == record) {
				flow->flow_control.committed_tail = previous;
			}
			record->flow_next = NULL;
			OPX_BUF_FREE(record);
		} else {
			previous = record;
		}
		record = next;
	}
	assert((flow->flow_control.committed_head == NULL) == (flow->flow_control.committed_tail == NULL));
}

void opx_flow_control_tx_flows_quiesce(struct fi_opx_ep *opx_ep)
{
	struct fi_opx_reliability_service *service = opx_ep->reli_service;
	RbtIterator			   itr;

	if (service == NULL || service->tx.tx_flow_rbtree == NULL) {
		return;
	}
	for (itr = fi_opx_rbt_begin(service->tx.tx_flow_rbtree); itr != NULL;
	     itr = fi_opx_rbt_next(service->tx.tx_flow_rbtree, itr)) {
		struct fi_opx_reliability_tx_flow *flow =
			(struct fi_opx_reliability_tx_flow *) fi_opx_rbt_value(service->tx.tx_flow_rbtree, itr);

		opx_flow_control_tx_flow_fini(opx_ep, flow);
	}
}

int opx_flow_control_rx_flow_init(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow)
{
	assert(opx_ep != NULL);
	assert(flow != NULL);

	struct opx_flow_control_state *flow_state = opx_flow_control_rccc_rx_flow_state(flow);
	flow_state->lifecycle			  = OPX_FLOW_CONTROL_ACTIVE;
	flow_state->fault_class			  = 0;
	flow_state->fault_scope			  = 0;
	flow_state->flags			  = 0;
	flow_state->reserved			  = 0;
	flow_state->transition_reason		  = 0;
	flow_state->transition_time_ns		  = 0;

	struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);
	*rx_state					= (struct opx_flow_control_rccc_rx_state) {0};
	rx_state->scheduler_state			= OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT;

	return FI_SUCCESS;
}

void opx_flow_control_rx_flow_fini(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow)
{
	(void) opx_ep;
	(void) flow;
}

static inline void opx_flow_control_rccc_rx_work_enqueue(struct fi_opx_ep		   *opx_ep,
							 struct fi_opx_reliability_rx_flow *flow)
{
	struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

	if (rx_state->work_listed != 0u) {
		return;
	}
	rx_state->work_listed	     = 1u;
	flow->flow_control_work_next = NULL;
	if (opx_ep->flow_control_rx_work_tail == NULL) {
		opx_ep->flow_control_rx_work_head = flow;
	} else {
		opx_ep->flow_control_rx_work_tail->flow_control_work_next = flow;
	}
	opx_ep->flow_control_rx_work_tail = flow;
}

/* Under context sharing the local ledger fields mirror only this member's
 * contribution; releases batch into shared_release_pending and fold into
 * the shared aggregate once per sweep so the per-packet first-accept path
 * never touches a cross-process atomic.  Unfolded releases only make the
 * shared ledger conservative (grants delayed, never over-granted). */
static inline void opx_flow_control_rccc_ledger_release(struct opx_flow_control_rccc_scheduler *scheduler,
							struct opx_flow_control_rccc_rx_state *rx_state, uint32_t delta)
{
	assert(delta < OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE);
	assert(rx_state->permission_contribution >= delta);
	assert(scheduler->outstanding_permissions >= delta);
	scheduler->outstanding_permissions -= delta;
	rx_state->permission_contribution -= delta;
	if (OFI_UNLIKELY(scheduler->shared_ledger != NULL)) {
		scheduler->shared_release_pending += delta;
	}
}

/* Slot eviction at epoch close: the ACCEPT decision record recycles onto
 * the free list (decision_used stays a high-water mark) and the tracked
 * entry returns to the pool, converting tracked_used from a lifetime-peer
 * bound into a concurrent-active bound.  Every caller guarantees
 * permission_contribution == 0, so the shared owner-claim release-store
 * publishes strictly after this member's contribution is provably zero.
 * scheduler.generation zeroes so a rebind re-claims through the full
 * decide_accept gate; replay safety needs no decision record because
 * epoch_snapshot/SNAPSHOT_VALID survive in the rx_state overlay and
 * classify a late kind-3 for the closed epoch STALE before any lookup. */
static void opx_flow_control_rccc_rx_slot_release(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow)
{
	struct opx_flow_control_rccc_scheduler *scheduler = opx_ep->flow_control_scheduler;
	struct opx_flow_control_rccc_rx_state  *rx_state  = opx_flow_control_rccc_rx_state(flow);

	if (rx_state->scheduler.generation == 0u) {
		return;
	}
	assert(rx_state->permission_contribution == 0u);
	if (rx_state->decision_slot < scheduler->decision_used) {
		struct opx_flow_control_rccc_decision_record *record = &scheduler->decisions[rx_state->decision_slot];

		if (record->occupied != 0u && record->key.flow_key.qw_prefix == flow->key->qw_prefix &&
		    record->key.flow_key.dw_suffix == flow->key->dw_suffix) {
			record->occupied	      = 0u;
			record->state		      = OPX_FLOW_CONTROL_RCCC_DECISION_EMPTY;
			record->free_next	      = scheduler->decision_free_head;
			scheduler->decision_free_head = rx_state->decision_slot;
		}
	}
	/* The local tracked_used mirrors this member's share in both modes
	 * (shared_detach bulk-subtracts it from the aggregate at destroy), so
	 * it must decrement here too or detach would double-subtract. */
	assert(scheduler->tracked_used != 0u);
	scheduler->tracked_used--;
	if (scheduler->shared_ledger != NULL) {
		struct opx_flow_control_rccc_scheduler_flow_owner *owner =
			opx_flow_control_rccc_shared_flow_owner(scheduler, rx_state->scheduler.slot);

		if (atomic_load_explicit(&owner->claim, memory_order_relaxed) == rx_state->scheduler.generation) {
			owner->flow_key_prefix = 0u;
			owner->flow_key_suffix = 0u;
			atomic_store_explicit(&owner->claim, 0u, memory_order_release);
		}
		assert(atomic_load_explicit(&scheduler->shared_ledger->tracked_used, memory_order_relaxed) != 0u);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->tracked_used, 1u, memory_order_release);
	}
	rx_state->scheduler.generation = 0u;
	rx_state->scheduler.slot       = 0u;
	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.slot_releases);
}

/* Withholding convergence: the closing epoch's RESET_ACK is releasable only
 * when grant - observed - retired_accepted == 0 under 24-bit serial rules.
 * The rebase retains the closed epoch as the serial snapshot (the ACK echoes
 * it, and a stale kind-3 for it is classified and dropped) and returns to
 * DORMANT so a fresh kind-3 rebinds through the decision path. */
static void opx_flow_control_rccc_rx_reset_drain_check(struct fi_opx_ep			 *opx_ep,
						       struct fi_opx_reliability_rx_flow *flow)
{
	struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

	if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING ||
	    ((rx_state->grant - rx_state->observed - rx_state->retired_accepted) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK) !=
		    0u) {
		return;
	}
	assert(rx_state->permission_contribution == 0u);
	rx_state->epoch_snapshot	 = rx_state->epoch;
	rx_state->epoch			 = rx_state->pending_epoch;
	rx_state->pending_epoch		 = 0u;
	rx_state->target		 = 0u;
	rx_state->grant			 = 0u;
	rx_state->observed		 = 0u;
	rx_state->retired		 = 0u;
	rx_state->retired_accepted	 = 0u;
	rx_state->drr_deficit		 = 0;
	rx_state->quarantine_deadline_ns = 0u;
	rx_state->scheduler_state	 = OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT;
	opx_flow_control_rccc_rx_flags_replace(opx_ep->flow_control_scheduler, rx_state,
					       OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING |
						       OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID);
	opx_flow_control_rccc_rx_slot_release(opx_ep, flow);
	opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
}

void opx_flow_control_rx_first_accept_slow(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow,
					   uint32_t psn)
{
	struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

	(void) psn;
	assert(opx_ep != NULL);
	assert(opx_ep->reli_service != NULL);
	switch (rx_state->scheduler_state) {
	case OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND:
	case OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING:
	case OPX_FLOW_CONTROL_RCCC_SCHEDULER_QUARANTINED:
		if (OFI_UNLIKELY((rx_state->flags & OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED) != 0u)) {
			break;
		}
		uint32_t outstanding = (rx_state->grant - rx_state->observed - rx_state->retired_accepted) &
				       OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		if (OFI_UNLIKELY(outstanding == 0u || outstanding >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE)) {
			break;
		}
		rx_state->observed = opx_flow_control_rccc_serial_add(rx_state->observed, 1u);
		if (rx_state->consumed_epoch != UINT32_MAX) {
			rx_state->consumed_epoch++;
		}
		if (opx_ep->flow_control_scheduler->consumed_total_epoch != UINT32_MAX) {
			opx_ep->flow_control_scheduler->consumed_total_epoch++;
		}
		opx_flow_control_rccc_ledger_release(opx_ep->flow_control_scheduler, rx_state, 1u);
		if (OFI_UNLIKELY(rx_state->scheduler_state == OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING)) {
			opx_flow_control_rccc_rx_reset_drain_check(opx_ep, flow);
		}
		return;
	default:
		break;
	}
	opx_ep->reli_service->flow_control_feedback_counters.invalid++;
}

static inline bool opx_flow_control_rccc_decision_key_equal(const struct opx_flow_control_rccc_decision_key *left,
							    const struct opx_flow_control_rccc_decision_key *right)
{
	return left->flow_key.qw_prefix == right->flow_key.qw_prefix &&
	       left->flow_key.dw_suffix == right->flow_key.dw_suffix && left->epoch == right->epoch;
}

static inline struct opx_flow_control_rccc_decision_record *
opx_flow_control_rccc_decision_find(struct opx_flow_control_rccc_scheduler	    *scheduler,
				    const struct opx_flow_control_rccc_decision_key *key)
{
	uint32_t index;

	for (index = 0u; index < scheduler->decision_used; ++index) {
		if (scheduler->decisions[index].occupied != 0u &&
		    opx_flow_control_rccc_decision_key_equal(&scheduler->decisions[index].key, key)) {
			return &scheduler->decisions[index];
		}
	}
	return NULL;
}

static inline enum opx_flow_control_result
opx_flow_control_rccc_decide_accept(struct opx_flow_control_rccc_scheduler	    *scheduler,
				    const struct opx_flow_control_rccc_decision_key *key, uint32_t request_generation,
				    bool consume_tracked, uint32_t *decision_slot)
{
	struct opx_flow_control_rccc_decision_record *record = opx_flow_control_rccc_decision_find(scheduler, key);

	if (record != NULL) {
		if (OFI_UNLIKELY(record->state != OPX_FLOW_CONTROL_RCCC_DECISION_ACCEPT)) {
			return OPX_FLOW_CONTROL_FAULT;
		}
		*decision_slot = (uint32_t) (record - scheduler->decisions);
		return OPX_FLOW_CONTROL_OK;
	}
	/* Capacity gates precede every mutation so a rejected bind leaves
	 * the pools untouched (model decide_accept contract).  decision_used
	 * is a high-water mark: released records recycle through the free
	 * list before the append region grows.  Sharing: tracked capacity is
	 * context-wide, taken from the shared counter with a bounded CAS
	 * after the local gates pass. */
	if ((scheduler->decision_free_head == OPX_FLOW_CONTROL_RCCC_DECISION_FREE_END &&
	     scheduler->decision_used >= scheduler->profile.decision_capacity) ||
	    (consume_tracked && scheduler->tracked_used >= scheduler->profile.tracked_capacity)) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	if (consume_tracked && scheduler->shared_ledger != NULL &&
	    !opx_flow_control_rccc_shared_tracked_reserve(scheduler)) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	if (scheduler->decision_free_head != OPX_FLOW_CONTROL_RCCC_DECISION_FREE_END) {
		*decision_slot		      = scheduler->decision_free_head;
		record			      = &scheduler->decisions[scheduler->decision_free_head];
		scheduler->decision_free_head = record->free_next;
	} else {
		*decision_slot = scheduler->decision_used;
		record	       = &scheduler->decisions[scheduler->decision_used];
		scheduler->decision_used++;
	}
	/* Key-stamp before publish: occupied flips last so a reused record
	 * is never findable under its previous key. */
	record->key		   = *key;
	record->request_generation = request_generation;
	record->free_next	   = 0u;
	record->state		   = OPX_FLOW_CONTROL_RCCC_DECISION_ACCEPT;
	record->grant_seen	   = 0u;
	record->occupied	   = 1u;
	if (consume_tracked) {
		scheduler->tracked_used++;
	}
	return OPX_FLOW_CONTROL_OK;
}

static inline void opx_flow_control_rccc_demand_active_inc(struct opx_flow_control_rccc_scheduler *scheduler)
{
	assert(scheduler->local_demand_active_flows < UINT32_MAX);
	scheduler->local_demand_active_flows++;
	if (scheduler->shared_ledger != NULL) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->demand_active_flows, memory_order_relaxed) <
		       UINT32_MAX);
		atomic_fetch_add_explicit(&scheduler->shared_ledger->demand_active_flows, 1u, memory_order_relaxed);
	} else {
		assert(scheduler->demand_active_flows < UINT32_MAX);
		scheduler->demand_active_flows++;
	}
}

static inline void opx_flow_control_rccc_demand_active_dec(struct opx_flow_control_rccc_scheduler *scheduler)
{
	assert(scheduler->local_demand_active_flows != 0u);
	scheduler->local_demand_active_flows--;
	if (scheduler->shared_ledger != NULL) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->demand_active_flows, memory_order_relaxed) !=
		       0u);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->demand_active_flows, 1u, memory_order_relaxed);
	} else {
		assert(scheduler->demand_active_flows != 0u);
		scheduler->demand_active_flows--;
	}
}

static inline void opx_flow_control_rccc_hot_inc(struct opx_flow_control_rccc_scheduler *scheduler)
{
	assert(scheduler->local_hot_flows < UINT32_MAX);
	scheduler->local_hot_flows++;
	if (scheduler->shared_ledger != NULL) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->hot_flows, memory_order_relaxed) < UINT32_MAX);
		atomic_fetch_add_explicit(&scheduler->shared_ledger->hot_flows, 1u, memory_order_relaxed);
	} else {
		assert(scheduler->hot_flows < UINT32_MAX);
		scheduler->hot_flows++;
	}
}

static inline void opx_flow_control_rccc_hot_dec(struct opx_flow_control_rccc_scheduler *scheduler)
{
	assert(scheduler->local_hot_flows != 0u);
	scheduler->local_hot_flows--;
	if (scheduler->shared_ledger != NULL) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->hot_flows, memory_order_relaxed) != 0u);
		atomic_fetch_sub_explicit(&scheduler->shared_ledger->hot_flows, 1u, memory_order_relaxed);
	} else {
		assert(scheduler->hot_flows != 0u);
		scheduler->hot_flows--;
	}
}

static inline void opx_flow_control_rccc_rx_window_state_init(struct opx_flow_control_rccc_scheduler *scheduler,
							      struct opx_flow_control_rccc_rx_state  *rx_state)
{
	rx_state->window_ceiling     = scheduler->profile.window_cap_min;
	rx_state->demand_peak_epoch  = 0u;
	rx_state->consumed_epoch     = 0u;
	rx_state->window_epoch_stamp = scheduler->window_epoch;
	rx_state->hot		     = 0u;
	rx_state->hi_streak	     = 0u;
	rx_state->lo_streak	     = 0u;
	rx_state->idle_streak	     = 0u;
}

/* Cold-ramp + hot-set window classification, processed lazily per flow when
 * the scheduler's window epoch has advanced past the flow's stamp.  Hot
 * entry and retention require BOTH a backlog leg (peak pending
 * target - grant over the epoch clears hot_absolute_floor - proves
 * standing sender demand) and a throughput leg (accepted packets per
 * epoch clear hot_consumed_floor - proves the receiver is actually
 * servicing the flow).  Both thresholds are absolute constants
 * independent of the ceiling, so classification cannot be ramp-raced;
 * a flow with qualifying backlog also does not cold-ramp that epoch,
 * which parks incast flows at the small initial ceiling until the
 * consumed leg confirms them.  Entry is fast and exit slow so collective
 * phase boundaries do not flap membership; an idle ceiling decays by
 * halves so a workload flip cannot ride a stale fmax ceiling for long. */
static inline void opx_flow_control_rccc_rx_window_epoch_update(struct fi_opx_ep		       *opx_ep,
								struct opx_flow_control_rccc_scheduler *scheduler,
								struct opx_flow_control_rccc_rx_state  *rx_state,
								uint32_t elapsed_epochs, bool congested)
{
	const struct opx_flow_control_rccc_scheduler_profile *profile	 = &scheduler->profile;
	const struct opx_flow_control_rccc_tx_profile	     *tx_profile = opx_flow_control_rccc_tx_profile_get();
	const bool backlog_hi = rx_state->demand_peak_epoch >= profile->hot_absolute_floor;
	const bool serviced   = rx_state->consumed_epoch >= profile->hot_consumed_floor;

	(void) opx_ep;
	if (backlog_hi && serviced) {
		if (rx_state->hi_streak != UINT8_MAX) {
			rx_state->hi_streak++;
		}
		rx_state->idle_streak = 0u;
		if (rx_state->hot == 0u && rx_state->hi_streak >= profile->hot_entry_streak) {
			rx_state->hot = 1u;
			opx_flow_control_rccc_hot_inc(scheduler);
		}
	} else {
		rx_state->hi_streak = 0u;
		if (backlog_hi) {
			rx_state->idle_streak = 0u;
		}
	}
	/* Hot RETENTION requires both legs: backlog proves standing sender
	 * demand and the consumed leg proves the receiver is servicing it.
	 * Backlog alone is endogenous - withheld grants inflate pending, so
	 * a division-induced micro-window storm keeps every flow looking
	 * demand-heavy (measured at np=256: all 192 peers hot, ceilings
	 * floored, Alltoallv rows at 13.5 ms).  Withholding also collapses
	 * consumption, so storm-classified flows fail the consumed leg and
	 * exit within hot_exit_streak epochs, division releases, and the
	 * ramp resumes - the fixed point self-recovers.  True incast holds
	 * both legs (measured 100 pkts/ms/flow at 15-to-1 and ~24 at 63-to-1
	 * against the floor of 8) and stays stably hot.  Receiver-imposed
	 * stalls are exempt from the exit evidence: a DRAINING duty episode
	 * stops grants and a congested pass means delivery itself is
	 * backlogged - exiting hot flows during either re-opens fmax windows
	 * at exactly the wrong moment (measured as a 4-6x overflow regression
	 * at 15-to-1 without the exemptions). */
	if (rx_state->hot != 0u) {
		if ((backlog_hi && serviced) || congested ||
		    scheduler->duty_state == OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING) {
			rx_state->lo_streak = 0u;
		} else {
			if (rx_state->lo_streak != UINT8_MAX) {
				rx_state->lo_streak++;
			}
			if (rx_state->lo_streak >= profile->hot_exit_streak) {
				rx_state->hot	    = 0u;
				rx_state->lo_streak = 0u;
				opx_flow_control_rccc_hot_dec(scheduler);
			}
		}
	}
	if (!backlog_hi) {
		if (rx_state->demand_peak_epoch == 0u && rx_state->consumed_epoch == 0u) {
			const uint32_t bumped = (uint32_t) rx_state->idle_streak + elapsed_epochs;

			rx_state->idle_streak = bumped > UINT8_MAX ? UINT8_MAX : (uint8_t) bumped;
			if (rx_state->idle_streak >= profile->idle_decay_streak) {
				rx_state->idle_streak = 0u;
				if ((rx_state->window_ceiling >> 1) >= profile->window_cap_min) {
					rx_state->window_ceiling >>= 1;
				}
			}
		} else {
			rx_state->idle_streak = 0u;
		}
	}
	/* The ramp freezes on the backlog leg alone: a flow with standing
	 * demand must not grow its ceiling while the consumed leg is still
	 * accumulating classification evidence, or the ramp outruns the
	 * classifier (measured at 15-to-1: freezing only on the full AND
	 * roughly doubled overflow versus freezing on backlog alone).  The
	 * receiver-global storm mode bypasses the freeze: a control-retry
	 * storm inflates backlog on every flow while granting almost
	 * nothing, and the only exit is growing the windows. */
	if (rx_state->hot == 0u && (!backlog_hi || scheduler->storm_mode != 0u) && !congested &&
	    rx_state->window_ceiling < tx_profile->fmax) {
		const uint32_t doubled = rx_state->window_ceiling << 1;

		rx_state->window_ceiling = doubled < tx_profile->fmax ? doubled : tx_profile->fmax;
	}
	rx_state->demand_peak_epoch = 0u;
	rx_state->consumed_epoch    = 0u;
	FI_OPX_DEBUG_COUNTERS_INC(
		opx_ep->debug_counters.flow_control.window_ceiling_hist[rx_state->window_ceiling <= 64u	  ? 0 :
									rx_state->window_ceiling <= 256u  ? 1 :
									rx_state->window_ceiling <= 1024u ? 2 :
													    3]);
}

static inline void opx_flow_control_rccc_ledger_charge(struct opx_flow_control_rccc_scheduler *scheduler,
						       struct opx_flow_control_rccc_rx_state *rx_state, uint32_t delta)
{
	assert(delta < OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE);
	assert(scheduler->outstanding_permissions <= scheduler->profile.max_outstanding_permissions - delta);
	scheduler->outstanding_permissions += delta;
	rx_state->permission_contribution += delta;
}

static inline void opx_flow_control_rccc_shared_release_fold(struct opx_flow_control_rccc_scheduler *scheduler)
{
	if (scheduler == NULL || scheduler->shared_ledger == NULL || scheduler->shared_release_pending == 0u) {
		return;
	}
	assert(atomic_load_explicit(&scheduler->shared_ledger->outstanding_permissions, memory_order_relaxed) >=
	       scheduler->shared_release_pending);
	atomic_fetch_sub_explicit(&scheduler->shared_ledger->outstanding_permissions, scheduler->shared_release_pending,
				  memory_order_acq_rel);
	scheduler->shared_release_pending = 0u;
}

/* With the statically partitioned ledger each member's local
 * max_outstanding_permissions (the 1/member_count share installed at
 * attach) is the authoritative grant bound, so the shared counter is pure
 * aggregate accounting: charge unconditionally here, fold releases at the
 * sweep.  The sum is bounded by construction (member_count * share). */
static inline void opx_flow_control_rccc_shared_ledger_charge(struct opx_flow_control_rccc_scheduler *scheduler,
							      uint64_t				      delta)
{
	atomic_fetch_add_explicit(&scheduler->shared_ledger->outstanding_permissions, delta, memory_order_acq_rel);
}

static inline bool opx_flow_control_rccc_shared_tracked_reserve(struct opx_flow_control_rccc_scheduler *scheduler)
{
	_Atomic uint32_t *tracked = &scheduler->shared_ledger->tracked_used;
	uint32_t	  current = atomic_load_explicit(tracked, memory_order_relaxed);

	while (current < scheduler->profile.tracked_capacity) {
		if (atomic_compare_exchange_weak_explicit(tracked, &current, current + 1u, memory_order_acq_rel,
							  memory_order_relaxed)) {
			return true;
		}
	}
	return false;
}

#ifndef NDEBUG
static inline void opx_flow_control_rccc_ledger_reconcile(struct fi_opx_ep *opx_ep)
{
	struct opx_flow_control_rccc_scheduler *scheduler	 = opx_ep->flow_control_scheduler;
	uint64_t				contribution	 = 0u;
	uint32_t				demand_active	 = 0u;
	uint32_t				hot		 = 0u;
	uint32_t				slot_holders	 = 0u;
	uint32_t				occupied	 = 0u;
	uint32_t				feedback_pending = 0u;
	uint32_t				reject_pending	 = 0u;
	uint32_t				index;
	RbtIterator				itr;

	if (scheduler == NULL || opx_ep->reli_service == NULL || opx_ep->reli_service->rx.rx_flow_rbtree == NULL) {
		return;
	}
	for (itr = fi_opx_rbt_begin(opx_ep->reli_service->rx.rx_flow_rbtree); itr != NULL;
	     itr = fi_opx_rbt_next(opx_ep->reli_service->rx.rx_flow_rbtree, itr)) {
		struct fi_opx_reliability_rx_flow *flow =
			fi_opx_rbt_value(opx_ep->reli_service->rx.rx_flow_rbtree, itr);

		struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

		contribution += rx_state->permission_contribution;
		if (rx_state->scheduler.generation != 0u) {
			slot_holders++;
		}
		if (rx_state->scheduler_state == OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND &&
		    rx_state->target != rx_state->grant) {
			demand_active++;
		}
		if (rx_state->hot != 0u) {
			hot++;
		}
		if ((rx_state->flags & OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING) != 0u) {
			feedback_pending++;
		}
		if ((rx_state->flags & OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING) != 0u) {
			reject_pending++;
		}
	}
	for (index = 0u; index < scheduler->decision_used; ++index) {
		if (scheduler->decisions[index].occupied != 0u) {
			occupied++;
		}
	}
	assert(contribution == scheduler->outstanding_permissions);
	assert(occupied == slot_holders);
	assert(feedback_pending == scheduler->feedback_pending_count);
	assert(reject_pending == scheduler->reject_pending_count);
	if (scheduler->shared_ledger == NULL) {
		assert(slot_holders == scheduler->tracked_used);
		assert(demand_active == scheduler->demand_active_flows);
		assert(hot == scheduler->hot_flows);
	}
}
#endif

static inline enum opx_flow_control_result
opx_flow_control_rccc_rx_retirement_settle(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_rx_state *rx_state,
					   uint32_t retired, struct opx_flow_control_rccc_feedback_counters *counters)
{
	const enum opx_flow_control_rccc_serial_classification classification =
		opx_flow_control_rccc_serial_classify(rx_state->retired_accepted, retired);

	opx_flow_control_rccc_feedback_count(counters, classification);
	if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		const uint32_t delta	   = (retired - rx_state->retired_accepted) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		const uint32_t outstanding = (rx_state->grant - rx_state->observed - rx_state->retired_accepted) &
					     OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;

		if (OFI_UNLIKELY(outstanding >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE || delta > outstanding)) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
			return OPX_FLOW_CONTROL_FAULT;
		}
		rx_state->retired_accepted = retired;
		rx_state->retired	   = retired;
		rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_RETIREMENT_ACK_PENDING;
		opx_flow_control_rccc_ledger_release(opx_ep->flow_control_scheduler, rx_state, delta);
		return OPX_FLOW_CONTROL_OK;
	}

	return classification == OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE ||
			       classification == OPX_FLOW_CONTROL_RCCC_SERIAL_STALE ?
		       OPX_FLOW_CONTROL_OK :
		       OPX_FLOW_CONTROL_FAULT;
}

/* Fresh kind-3 on an unbound flow: STALE/DUPLICATE epochs against the
 * retained snapshot are dropped (the interim issuer's blind epoch adoption
 * defect), then the decision pool either binds the pairing (PREPARING is
 * instantaneous under process-local single-threaded authority) or the
 * capacity shortfall is answered STATELESSLY: no decision/reject slot is
 * burned, the kind-8 is built from rx_state fields alone, and a retried
 * kind-3 re-runs the same admission - recomputing the same reject while
 * full and upgrading to ACCEPT the moment capacity frees.  One-directional
 * stickiness is preserved by construction: an ACCEPT record replays
 * verbatim, and ACCEPT-after-reject is always safe because grants are
 * receiver-authoritative from zero. */
static enum opx_flow_control_result
opx_flow_control_rccc_rx_bind(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow,
			      const struct opx_flow_control_wire_envelope    *envelope,
			      struct opx_flow_control_rccc_feedback_counters *counters)
{
	struct opx_flow_control_rccc_scheduler		*scheduler = opx_ep->flow_control_scheduler;
	struct opx_flow_control_rccc_rx_state		*rx_state  = opx_flow_control_rccc_rx_state(flow);
	struct opx_flow_control_rccc_decision_key	 key = {.flow_key = *flow->key, .epoch = envelope->generation};
	enum opx_flow_control_rccc_serial_classification classification;
	enum opx_flow_control_result			 result;
	uint32_t					 slot;

	assert(scheduler != NULL);
	if ((rx_state->flags & OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID) != 0u) {
		classification = opx_flow_control_rccc_serial_classify(rx_state->epoch_snapshot, envelope->generation);
		if (classification != OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
			return OPX_FLOW_CONTROL_DEFER;
		}
	}
	result = opx_flow_control_rccc_decide_accept(scheduler, &key, envelope->value[2],
						     rx_state->scheduler.generation == 0u, &slot);
	if (OFI_UNLIKELY(result == OPX_FLOW_CONTROL_FAULT)) {
		opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
		return OPX_FLOW_CONTROL_FAULT;
	}
	if (result == OPX_FLOW_CONTROL_DEFER) {
		rx_state->rejected_epoch     = envelope->generation;
		rx_state->request_generation = envelope->value[2];
		opx_flow_control_rccc_rx_pending_set(scheduler, rx_state, OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING);
		opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
		return OPX_FLOW_CONTROL_OK;
	}
	scheduler->decisions[slot].request_generation = envelope->value[2];
	/* The scheduler reference is the tracked-entry handle
	 * (decision_slot/reject_slot stay independent pool
	 * indices).  Non-shared: the tracked index just claimed by
	 * decide_accept.  Sharing: a CAS-claimed slot in the mapping's
	 * tracked region, generation-stamped so references stay unique
	 * across member join/leave. */
	if (rx_state->scheduler.generation == 0u) {
		if (scheduler->shared_ledger != NULL) {
			uint32_t shared_slot;
			uint64_t shared_claim;

			if (OFI_UNLIKELY(!opx_flow_control_rccc_shared_flow_claim(scheduler, flow->key, &shared_slot,
										  &shared_claim))) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
				return OPX_FLOW_CONTROL_FAULT;
			}
			rx_state->scheduler.slot       = shared_slot;
			rx_state->scheduler.generation = shared_claim;
			scheduler->allocation_generation++;
		} else {
			rx_state->scheduler.slot       = scheduler->tracked_used - 1u;
			rx_state->scheduler.generation = scheduler->allocation_generation++;
		}
	}
	rx_state->epoch		     = envelope->generation;
	rx_state->target	     = envelope->value[0];
	rx_state->grant		     = 0u;
	rx_state->observed	     = 0u;
	rx_state->retired	     = 0u;
	rx_state->retired_accepted   = 0u;
	rx_state->decision_slot	     = slot;
	rx_state->request_generation = envelope->value[2];
	/* BOUND is entered only here and left only by RESET below; quarantine
	 * reclaim operates on RESETTING flows, so this transition pair owns the
	 * live-flow counters for the whole state-machine lifetime. */
	rx_state->scheduler_state = OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND;
	opx_flow_control_rccc_rx_window_state_init(scheduler, rx_state);
	opx_flow_control_rccc_demand_active_inc(scheduler);
	assert(scheduler->local_bound_flows < UINT32_MAX);
	scheduler->local_bound_flows++;
	if (scheduler->shared_ledger != NULL) {
		assert(atomic_load_explicit(&scheduler->shared_ledger->active_bound_flows, memory_order_relaxed) <
		       UINT32_MAX);
		atomic_fetch_add_explicit(&scheduler->shared_ledger->active_bound_flows, 1u, memory_order_relaxed);
	} else {
		assert(scheduler->active_bound_flows < UINT32_MAX);
		scheduler->active_bound_flows++;
	}
	opx_flow_control_rccc_rx_pending_set(scheduler, rx_state, OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
	opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
	return OPX_FLOW_CONTROL_OK;
}

static enum opx_flow_control_rccc_serial_classification
opx_flow_control_rccc_rx_target_apply(struct fi_opx_ep *opx_ep, struct opx_flow_control_rccc_rx_state *rx_state,
				      uint32_t target, struct opx_flow_control_rccc_feedback_counters *counters)
{
	const struct opx_flow_control_rccc_tx_profile	      *profile = opx_flow_control_rccc_tx_profile_get();
	const enum opx_flow_control_rccc_serial_classification classification =
		opx_flow_control_rccc_serial_classify(rx_state->target, target);

	opx_flow_control_rccc_feedback_count(counters, classification);
	if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		const uint32_t delta = (target - rx_state->target) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;

		if (OFI_UNLIKELY(delta > profile->fmax)) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
			return OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID;
		}
		if (rx_state->target == rx_state->grant) {
			opx_flow_control_rccc_demand_active_inc(opx_ep->flow_control_scheduler);
		}
		rx_state->target = target;
	}
	return classification;
}

/* Piggybacked-target ingestion (PING QW6).  Deliberately narrower than the
 * dedicated kind-3 path: no request_epoch increment (the storm detector's
 * standalone-request signature must stay undiluted), no DORMANT bind (the
 * decision path needs the request-generation echo and reject handling that
 * only a dedicated request carries), no retirement, no feedback-pending
 * response (stale/duplicate piggybacks are silent).  A newer target
 * enqueues the flow so the grant pass visits it. */
enum opx_flow_control_result opx_flow_control_rccc_rx_target_piggyback(struct fi_opx_ep			 *opx_ep,
								       struct fi_opx_reliability_rx_flow *flow,
								       uint32_t generation, uint32_t target)
{
	struct opx_flow_control_rccc_rx_state	       *rx_state = opx_flow_control_rccc_rx_state(flow);
	struct opx_flow_control_rccc_feedback_counters *counters =
		&opx_ep->reli_service->flow_control_feedback_counters;
	enum opx_flow_control_rccc_serial_classification classification;

	if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND || rx_state->epoch != generation) {
		opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
		return OPX_FLOW_CONTROL_OK;
	}
	classification = opx_flow_control_rccc_rx_target_apply(opx_ep, rx_state, target, counters);
	if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS ||
	    classification == OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.piggyback_ping_received);
		opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
	}
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result opx_flow_control_rccc_rx_control(struct fi_opx_ep				  *opx_ep,
							      struct fi_opx_reliability_rx_flow		  *flow,
							      const struct opx_flow_control_wire_envelope *envelope)
{
	struct opx_flow_control_rccc_feedback_counters	*counters;
	struct opx_flow_control_rccc_rx_state		*rx_state;
	enum opx_flow_control_rccc_serial_classification classification;
	enum opx_flow_control_result			 result;
	uint32_t					 outstanding;

	assert(opx_ep != NULL);
	assert(opx_ep->reli_service != NULL);
	assert(flow != NULL);
	assert(envelope != NULL);
	assert(opx_flow_control_rccc_tx_profile_get() != NULL);
	counters = &opx_ep->reli_service->flow_control_feedback_counters;
	rx_state = opx_flow_control_rccc_rx_state(flow);
	switch (envelope->kind) {
	case OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST:
		/* Standalone request arrivals feed the storm detector; keep
		 * this count distinct from any future piggybacked carrier so
		 * the retry-storm signature stays undiluted. */
		if (opx_ep->flow_control_scheduler != NULL &&
		    opx_ep->flow_control_scheduler->request_epoch != UINT32_MAX) {
			opx_ep->flow_control_scheduler->request_epoch++;
		}
		if (rx_state->scheduler_state == OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT) {
			if (OFI_UNLIKELY(envelope->generation == 0u || envelope->value[0] == 0u)) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
				return OPX_FLOW_CONTROL_FAULT;
			}
			return opx_flow_control_rccc_rx_bind(opx_ep, flow, envelope, counters);
		}
		if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND ||
		    rx_state->epoch != envelope->generation) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
			return OPX_FLOW_CONTROL_DEFER;
		}
		classification = opx_flow_control_rccc_rx_target_apply(opx_ep, rx_state, envelope->value[0], counters);
		if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_AMBIGUOUS ||
		    classification == OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID) {
			return OPX_FLOW_CONTROL_FAULT;
		}
		rx_state->request_generation = envelope->value[2];
		result = opx_flow_control_rccc_rx_retirement_settle(opx_ep, rx_state, envelope->value[1], counters);
		if (result != OPX_FLOW_CONTROL_OK) {
			return result;
		}
		opx_flow_control_rccc_rx_pending_set(opx_ep->flow_control_scheduler, rx_state,
						     OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
		break;
	case OPX_FLOW_CONTROL_RCCC_KIND_RETIREMENT:
		if ((rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND &&
		     rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING) ||
		    rx_state->epoch != envelope->generation) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
			return OPX_FLOW_CONTROL_DEFER;
		}
		result = opx_flow_control_rccc_rx_retirement_settle(opx_ep, rx_state, envelope->value[1], counters);
		if (result != OPX_FLOW_CONTROL_OK) {
			return result;
		}
		if (rx_state->scheduler_state == OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING) {
			opx_flow_control_rccc_rx_reset_drain_check(opx_ep, flow);
			return OPX_FLOW_CONTROL_OK;
		}
		opx_flow_control_rccc_rx_pending_set(opx_ep->flow_control_scheduler, rx_state,
						     OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
		break;
	case OPX_FLOW_CONTROL_RCCC_KIND_RESET:
		if (OFI_UNLIKELY(envelope->generation == 0u)) {
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
			return OPX_FLOW_CONTROL_FAULT;
		}
		switch (rx_state->scheduler_state) {
		case OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT:
			if ((rx_state->flags & OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID) != 0u) {
				classification = opx_flow_control_rccc_serial_classify(rx_state->epoch_snapshot,
										       envelope->generation);
				if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE) {
					rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING;
					break;
				}
				if (classification != OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
					opx_flow_control_rccc_feedback_count(counters,
									     OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
					return OPX_FLOW_CONTROL_DEFER;
				}
			}
			/* Nothing is outstanding on an unbound flow, so the
			 * epoch closes immediately and the flow stays
			 * DORMANT (RESET-on-DORMANT routes through the
			 * same snapshot/immediate-ACK discipline). */
			rx_state->epoch_snapshot = envelope->generation;
			rx_state->epoch		 = opx_flow_control_rccc_serial_add(envelope->generation, 1u);
			rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING |
					   OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID;
			break;
		case OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND:
			if (envelope->generation != rx_state->epoch) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
				return OPX_FLOW_CONTROL_DEFER;
			}
			result = opx_flow_control_rccc_rx_retirement_settle(opx_ep, rx_state, envelope->value[1],
									    counters);
			if (result != OPX_FLOW_CONTROL_OK) {
				return result;
			}
			outstanding = (rx_state->grant - rx_state->observed - rx_state->retired_accepted) &
				      OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
			if (outstanding != 0u) {
				/* RESET is the sender's terminal ledger statement:
				 * its emit predicate (committed FIFO drained,
				 * retirement acked) means every consumed credit was
				 * delivered and counted before this kind-5 was
				 * sent.  The residue is exactly the
				 * granted-but-never-applied window - kind-6 grants
				 * lost on the unreliable control plane or dropped
				 * STALE while the sender was already RESETTING -
				 * which the sender can never consume or retire, so
				 * withholding the ACK on it can never converge
				 * (this livelocked 15->1 incast: 411K RESETs
				 * unanswered).  Force-release it: the permissions
				 * return to the ledger (a release, not a leak) and
				 * the epoch closes convergently. */
				assert(outstanding < OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE);
				assert(rx_state->permission_contribution == outstanding);
				opx_flow_control_rccc_ledger_release(opx_ep->flow_control_scheduler, rx_state,
								     (uint32_t) outstanding);
				rx_state->retired_accepted =
					(rx_state->grant - rx_state->observed) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
				outstanding = 0u;
			}
			rx_state->pending_epoch = opx_flow_control_rccc_serial_add(rx_state->epoch, 1u);
			if (rx_state->target != rx_state->grant) {
				opx_flow_control_rccc_demand_active_dec(opx_ep->flow_control_scheduler);
			}
			if (rx_state->hot != 0u) {
				rx_state->hot = 0u;
				opx_flow_control_rccc_hot_dec(opx_ep->flow_control_scheduler);
			}
			rx_state->scheduler_state = OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING;
			assert(opx_ep->flow_control_scheduler->local_bound_flows != 0u);
			opx_ep->flow_control_scheduler->local_bound_flows--;
			if (opx_ep->flow_control_scheduler->shared_ledger != NULL) {
				assert(atomic_load_explicit(
					       &opx_ep->flow_control_scheduler->shared_ledger->active_bound_flows,
					       memory_order_relaxed) != 0u);
				atomic_fetch_sub_explicit(
					&opx_ep->flow_control_scheduler->shared_ledger->active_bound_flows, 1u,
					memory_order_relaxed);
			} else {
				assert(opx_ep->flow_control_scheduler->active_bound_flows != 0u);
				opx_ep->flow_control_scheduler->active_bound_flows--;
			}
			rx_state->quarantine_deadline_ns = 0u;
			opx_flow_control_rccc_rx_pending_clear(opx_ep->flow_control_scheduler, rx_state,
							       OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
			if (outstanding == 0u) {
				opx_flow_control_rccc_rx_reset_drain_check(opx_ep, flow);
				return OPX_FLOW_CONTROL_OK;
			}
			break;
		case OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING:
			if (envelope->generation != rx_state->epoch) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
				return OPX_FLOW_CONTROL_DEFER;
			}
			/* Duplicate RESET while draining: settle the retirement
			 * echo and force-release any residue exactly as the
			 * BOUND arm does, so a flow left withholding by an
			 * earlier code version (or a settle race) converges on
			 * the retry instead of deferring silently forever. */
			result = opx_flow_control_rccc_rx_retirement_settle(opx_ep, rx_state, envelope->value[1],
									    counters);
			if (result != OPX_FLOW_CONTROL_OK) {
				return result;
			}
			outstanding = (rx_state->grant - rx_state->observed - rx_state->retired_accepted) &
				      OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
			if (outstanding != 0u) {
				assert(outstanding < OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE);
				assert(rx_state->permission_contribution == outstanding);
				opx_flow_control_rccc_ledger_release(opx_ep->flow_control_scheduler, rx_state,
								     (uint32_t) outstanding);
				rx_state->retired_accepted =
					(rx_state->grant - rx_state->observed) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
			}
			opx_flow_control_rccc_rx_reset_drain_check(opx_ep, flow);
			return OPX_FLOW_CONTROL_OK;
		case OPX_FLOW_CONTROL_RCCC_SCHEDULER_QUARANTINED:
			/* Quarantine already force-released this flow's
			 * contribution and slot, so the RESET is answerable
			 * immediately.  DUPLICATE (the quarantined epoch) or
			 * NEWER (peer restart) rebases to DORMANT and ACKs -
			 * without this arm a quarantined peer's RESETs fall
			 * through unanswered forever and the peer can never
			 * rebind (rx_bind accepts kind-3 from DORMANT only). */
			classification = opx_flow_control_rccc_serial_classify(rx_state->epoch, envelope->generation);
			if (classification != OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE &&
			    classification != OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
				return OPX_FLOW_CONTROL_DEFER;
			}
			assert(rx_state->permission_contribution == 0u);
			rx_state->epoch_snapshot	 = envelope->generation;
			rx_state->epoch			 = opx_flow_control_rccc_serial_add(envelope->generation, 1u);
			rx_state->pending_epoch		 = 0u;
			rx_state->target		 = 0u;
			rx_state->grant			 = 0u;
			rx_state->observed		 = 0u;
			rx_state->retired		 = 0u;
			rx_state->retired_accepted	 = 0u;
			rx_state->quarantine_deadline_ns = 0u;
			rx_state->scheduler_state	 = OPX_FLOW_CONTROL_RCCC_SCHEDULER_DORMANT;
			opx_flow_control_rccc_rx_flags_replace(opx_ep->flow_control_scheduler, rx_state,
							       OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING |
								       OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID);
			break;
		default:
			opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_STALE);
			return OPX_FLOW_CONTROL_DEFER;
		}
		break;
	default:
		opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
		return OPX_FLOW_CONTROL_FAULT;
	}
	opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result opx_flow_control_rccc_rx_emit_pending(struct fi_opx_ep		     *opx_ep,
								   struct fi_opx_reliability_rx_flow *flow,
								   uint8_t			      pending_flag)
{
	struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);
	struct opx_flow_control_wire_envelope  envelope = {
		 .key	    = flow->key->qw_prefix,
		 .mode	    = OPX_FLOW_CONTROL_MODE_RCCC,
		 .direction = OPX_FLOW_CONTROL_WIRE_DIRECTION_FEEDBACK,
	 };
	enum opx_flow_control_result result;

	assert(pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING ||
	       pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING ||
	       pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING);
	if ((rx_state->flags & pending_flag) == 0u) {
		return OPX_FLOW_CONTROL_OK;
	}
	if (pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING) {
		envelope.generation =
			opx_flow_control_rccc_serial_add(rx_state->epoch, OPX_FLOW_CONTROL_RCCC_SERIAL_MASK);
		envelope.kind = OPX_FLOW_CONTROL_RCCC_KIND_RESET_ACK;
	} else if (pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING) {
		/* Kind-8 wire shape: generation carries the
		 * request_generation reply serial and value[0] echoes the
		 * rejected epoch - both from rx_state alone (stateless
		 * rejects burn no decision/reject slots). */
		envelope.generation = rx_state->request_generation;
		envelope.value[0]   = rx_state->rejected_epoch;
		envelope.flags	    = OPX_FLOW_CONTROL_WIRE_FLAG_TERMINAL;
		envelope.kind	    = OPX_FLOW_CONTROL_RCCC_KIND_CAPACITY_REJECT;
	} else {
		envelope.generation = rx_state->epoch;
		envelope.value[0]   = rx_state->grant;
		envelope.value[1]   = rx_state->retired_accepted;
		envelope.kind	    = OPX_FLOW_CONTROL_RCCC_KIND_FEEDBACK;
	}
	result = opx_flow_control_wire_send(opx_ep, false, flow->key, &envelope);
	if (result == OPX_FLOW_CONTROL_OK) {
		if (pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING ||
		    pending_flag == OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING) {
			opx_flow_control_rccc_rx_pending_clear(opx_ep->flow_control_scheduler, rx_state, pending_flag);
		} else {
			rx_state->flags &= (uint8_t) ~pending_flag;
		}
		if (pending_flag != OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING) {
			rx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_RX_FLAG_RETIREMENT_ACK_PENDING;
		}
	}
	return result;
}

static enum opx_flow_control_result
opx_flow_control_rccc_tx_emit(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow,
			      enum opx_flow_control_rccc_tx_emission		     kind,
			      const struct opx_flow_control_rccc_tx_emission_intent *intent, uint64_t now_ns)
{
	const struct opx_flow_control_rccc_tx_profile *profile	= opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_tx_state	      *tx_state = opx_flow_control_rccc_tx_state(flow);
	struct opx_flow_control_wire_envelope	       envelope = {
			 .key	    = tx_state->flow_key.qw_prefix,
			 .mode	    = OPX_FLOW_CONTROL_MODE_RCCC,
			 .flags	    = OPX_FLOW_CONTROL_WIRE_FLAG_RESPONSE_REQUESTED,
			 .direction = OPX_FLOW_CONTROL_WIRE_DIRECTION_REQUEST,
	 };
	enum opx_flow_control_result result;

	envelope.generation = intent->epoch;
	envelope.value[1]   = intent->retired;
	switch (kind) {
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST:
		envelope.kind	  = OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST;
		envelope.value[0] = intent->target;
		envelope.value[2] = intent->request_generation;
		break;
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT:
		envelope.kind = OPX_FLOW_CONTROL_RCCC_KIND_RETIREMENT;
		break;
	case OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET:
		envelope.kind = OPX_FLOW_CONTROL_RCCC_KIND_RESET;
		break;
	default:
		return OPX_FLOW_CONTROL_FAULT;
	}
	result = opx_flow_control_wire_send(opx_ep, true, &tx_state->flow_key, &envelope);
	if (result == OPX_FLOW_CONTROL_OK) {
		opx_flow_control_rccc_tx_emitted(tx_state, profile, kind, now_ns);
	}
	return result;
}

static void opx_flow_control_rccc_tx_sweep_pass(struct fi_opx_ep *opx_ep, uint64_t now_ns, bool reset_pass,
						uint16_t *budget)
{
	struct fi_opx_reliability_tx_flow *flow;
	struct fi_opx_reliability_tx_flow *previous = NULL;

	for (flow = opx_ep->flow_control_tx_work_head; flow != NULL;) {
		struct fi_opx_reliability_tx_flow	       *next	 = flow->flow_control.work_next;
		struct opx_flow_control_rccc_tx_state	       *tx_state = opx_flow_control_rccc_tx_state(flow);
		enum opx_flow_control_rccc_tx_emission		kind;
		struct opx_flow_control_rccc_tx_emission_intent intent;

		/* Idle-close: a fully settled open epoch (no pending
		 * control, no credits, no replays, retirement acked) that
		 * has sat idle for two idle-retire periods RESETs, so the
		 * receiver's RESET-drain releases the tracked/decision
		 * slots - converting tracked capacity into a
		 * concurrent-active bound.  The jittered reset backoff
		 * staggers churn-boundary RESET bursts. */
		if (reset_pass && tx_state->grant != 0u &&
		    (tx_state->flags &
		     (OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING |
		      OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING)) ==
			    0u &&
		    tx_state->available_credits == 0u && tx_state->held_credits == 0u &&
		    tx_state->retired == tx_state->retired_acked && flow->flow_control.committed_head == NULL &&
		    now_ns - tx_state->activity_time_ns >=
			    2u * opx_flow_control_rccc_tx_profile_get()->idle_retire_delay_ns) {
			(void) opx_flow_control_rccc_tx_reset_begin(tx_state, opx_flow_control_rccc_tx_profile_get(),
								    now_ns);
		}
		if (*budget == 0u) {
			return;
		}
		kind = opx_flow_control_rccc_tx_progress(tx_state, opx_flow_control_rccc_tx_profile_get(), now_ns,
							 &intent);
		if ((reset_pass && kind == OPX_FLOW_CONTROL_RCCC_TX_EMIT_RESET) ||
		    (!reset_pass && (kind == OPX_FLOW_CONTROL_RCCC_TX_EMIT_CREDIT_REQUEST ||
				     kind == OPX_FLOW_CONTROL_RCCC_TX_EMIT_RETIREMENT))) {
			if (opx_flow_control_rccc_tx_emit(opx_ep, flow, kind, &intent, now_ns) == OPX_FLOW_CONTROL_OK) {
				(*budget)--;
			}
		}
		if (!opx_flow_control_rccc_tx_work_needed(tx_state)) {
			if (previous == NULL) {
				opx_ep->flow_control_tx_work_head = next;
			} else {
				previous->flow_control.work_next = next;
			}
			if (opx_ep->flow_control_tx_work_tail == flow) {
				opx_ep->flow_control_tx_work_tail = previous;
			}
			flow->flow_control.work_next = NULL;
			tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_ON_WORK_LIST;
		} else {
			previous = flow;
		}
		flow = next;
	}
}

static void opx_flow_control_rccc_rx_sweep_pass(struct fi_opx_ep *opx_ep, uint8_t pending_flag, uint16_t *budget)
{
	struct fi_opx_reliability_rx_flow *flow;
	struct fi_opx_reliability_rx_flow *previous = NULL;

	for (flow = opx_ep->flow_control_rx_work_head; flow != NULL;) {
		struct fi_opx_reliability_rx_flow     *next	= flow->flow_control_work_next;
		struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

		if (*budget == 0u) {
			return;
		}
		if ((rx_state->flags & pending_flag) != 0u &&
		    opx_flow_control_rccc_rx_emit_pending(opx_ep, flow, pending_flag) == OPX_FLOW_CONTROL_OK) {
			(*budget)--;
		}
		/* Extended stay-listed predicate: pending emissions
		 * OR BOUND with unmet demand (DRR membership) OR RESETTING
		 * (drain/quarantine progress) OR undrained contribution. */
		if ((rx_state->flags &
		     (OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING | OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING |
		      OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING)) == 0u &&
		    !(rx_state->scheduler_state == OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND &&
		      rx_state->target != rx_state->grant) &&
		    rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING &&
		    rx_state->permission_contribution == 0u) {
			if (previous == NULL) {
				opx_ep->flow_control_rx_work_head = next;
			} else {
				previous->flow_control_work_next = next;
			}
			if (opx_ep->flow_control_rx_work_tail == flow) {
				opx_ep->flow_control_rx_work_tail = previous;
			}
			flow->flow_control_work_next = NULL;
			rx_state->work_listed	     = 0u;
		} else {
			previous = flow;
		}
		flow = next;
	}
}

/* Quarantine reclaim: a RESETTING flow whose withholding never
 * converges is quarantined at the deadline; reclaim zeroes its outstanding
 * ledger share and terminally closes the epoch with NO RESET_ACK.  A
 * post-reclaim governed arrival is a flow fault (first-accept RECLAIMED
 * gate). */
static void opx_flow_control_rccc_rx_quarantine_check(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow,
						      uint64_t now_ns)
{
	const struct opx_flow_control_rccc_scheduler_profile *profile  = &opx_ep->flow_control_scheduler->profile;
	struct opx_flow_control_rccc_rx_state		     *rx_state = opx_flow_control_rccc_rx_state(flow);

	if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING) {
		return;
	}
	if (rx_state->quarantine_deadline_ns == 0u) {
		rx_state->quarantine_deadline_ns = now_ns + profile->quarantine_deadline_ns;
		return;
	}
	if (now_ns < rx_state->quarantine_deadline_ns) {
		return;
	}
	opx_flow_control_rccc_ledger_release(opx_ep->flow_control_scheduler, rx_state,
					     (uint32_t) rx_state->permission_contribution);
	rx_state->scheduler_state = OPX_FLOW_CONTROL_RCCC_SCHEDULER_QUARANTINED;
	opx_flow_control_rccc_rx_flags_replace(opx_ep->flow_control_scheduler, rx_state,
					       OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED);
	opx_flow_control_rccc_rx_slot_release(opx_ep, flow);
	opx_ep->reli_service->flow_control_feedback_counters.invalid++;
}

/* Teardown reclaim: the RX mirror of opx_flow_control_tx_flows_quiesce.
 * Grants are monotonic and the wire has no receiver-initiated recall, so a
 * sender whose traffic simply stops (e.g. a time-budget-truncated benchmark
 * finalizing promptly) leaves its granted-but-unconsumed window outstanding
 * forever.  At endpoint teardown no further retirement can arrive; each
 * flow's remaining contribution is force-reclaimed exactly as quarantine
 * reclaim does (receiver-silent epoch close, no RESET_ACK - the quarantine
 * precedent).  This is expected protocol convergence for truncated peers,
 * not a peer fault, so it feeds dedicated census counters rather than
 * invalid.  Bound-flow/demand/hot counter bookkeeping is intentionally
 * skipped: scheduler_shared_detach bulk-releases this member's shared
 * counters at destroy, and no grant pass runs after this point. */
void opx_flow_control_rx_flows_reclaim(struct fi_opx_ep *opx_ep)
{
	struct opx_flow_control_rccc_scheduler *scheduler	      = opx_ep->flow_control_scheduler;
	uint64_t				reclaimed_flows	      = 0u;
	uint64_t				reclaimed_permissions = 0u;
	RbtIterator				itr;

	if (scheduler == NULL || opx_ep->reli_service == NULL || opx_ep->reli_service->rx.rx_flow_rbtree == NULL) {
		return;
	}
	for (itr = fi_opx_rbt_begin(opx_ep->reli_service->rx.rx_flow_rbtree); itr != NULL;
	     itr = fi_opx_rbt_next(opx_ep->reli_service->rx.rx_flow_rbtree, itr)) {
		struct fi_opx_reliability_rx_flow *flow =
			fi_opx_rbt_value(opx_ep->reli_service->rx.rx_flow_rbtree, itr);

		struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);

		if (rx_state->permission_contribution != 0u) {
			reclaimed_flows++;
			reclaimed_permissions += rx_state->permission_contribution;
			opx_flow_control_rccc_ledger_release(scheduler, rx_state,
							     (uint32_t) rx_state->permission_contribution);
			rx_state->scheduler_state = OPX_FLOW_CONTROL_RCCC_SCHEDULER_QUARANTINED;
		}
		opx_flow_control_rccc_rx_flags_replace(scheduler, rx_state, OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED);
		opx_flow_control_rccc_rx_slot_release(opx_ep, flow);
	}
	if (reclaimed_flows != 0u) {
		FI_INFO(fi_opx_global.prov, FI_LOG_EP_CTRL,
			"RCCC teardown reclaimed %lu outstanding permissions across %lu flows\n",
			(unsigned long) reclaimed_permissions, (unsigned long) reclaimed_flows);
		FI_OPX_DEBUG_COUNTERS_INC_N(reclaimed_flows,
					    opx_ep->debug_counters.flow_control.teardown_reclaimed_flows);
		FI_OPX_DEBUG_COUNTERS_INC_N(reclaimed_permissions,
					    opx_ep->debug_counters.flow_control.teardown_reclaimed_permissions);
	}
}

/* DRR grant tier: occupancy-scaled Q16.16 accrual sets a bulk budget
 * separate from the control-emission budget; each listed BOUND flow with
 * unmet demand earns one quantum per visit and receives min(demand, deficit,
 * ledger headroom, ring-share window headroom, budget).  Deficits carry
 * across polls; cap-pinned flows do not bank deficit while awaiting relief. */
static void opx_flow_control_rccc_rx_grant_pass(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	struct opx_flow_control_rccc_scheduler	      *scheduler  = opx_ep->flow_control_scheduler;
	const struct opx_flow_control_rccc_tx_profile *tx_profile = opx_flow_control_rccc_tx_profile_get();
	struct fi_opx_reliability_rx_flow	      *flow;
	__uint128_t				       product;
	uint64_t				       elapsed_ns;
	uint64_t				       accrued;
	uint64_t				       bulk_budget;
	uint64_t				       effective_ledger_bound;
	uint32_t				       bound_flows;
	uint32_t				       hot_flows;
	uint32_t				       rate_q16;
	uint32_t				       occupancy     = 0u;
	uint32_t				       bypass_budget = 8u;
	uint16_t				       grant_emit_budget;
	bool					       occupancy_sample_fresh = false;
	bool					       ledger_cap_active      = false;
	bool					       congested;

	if (scheduler == NULL) {
		return;
	}
	assert(tx_profile != NULL);
	grant_emit_budget = scheduler->profile.grant_emits;
	assert(grant_emit_budget != 0u);
	if (scheduler->accrual_last_ns == 0u) {
		scheduler->accrual_last_ns = now_ns;
		return;
	}
	elapsed_ns = now_ns - scheduler->accrual_last_ns;
	/* Advance on every visit so releasing a stopped latch cannot accrue a catch-up burst. */
	scheduler->accrual_last_ns = now_ns;
	if (elapsed_ns > scheduler->profile.maximum_elapsed_ns) {
		elapsed_ns = scheduler->profile.maximum_elapsed_ns;
	}
	bound_flows =
		scheduler->shared_ledger != NULL ?
			atomic_load_explicit(&scheduler->shared_ledger->active_bound_flows, memory_order_relaxed) :
			scheduler->active_bound_flows;
	/* The hot-set sum-of-windows defense degenerates once
	 * usable_capacity / hot_flows falls below window_cap_min: the floor then
	 * admits window_cap_min * bound_flows outstanding permissions, which can
	 * exceed the receive ring.  At the same high-fan-in gate used to exempt
	 * np2/single-stream shapes from hot-set division, cap the aggregate ledger
	 * at the usable ring so it restores Sigma(in-flight) <= usable_capacity.
	 * This clamp only tightens the configured bound, including an explicit
	 * user value; it never raises a smaller bound.  Context-sharing members do
	 * share one physical receive ring, but their grant authorities use static
	 * per-member shares while the shared atomic is accounting-only.  Applying
	 * one member's full-ring cap independently would still permit one ring per
	 * member, so a correct shared aggregate clamp is deferred.  The proven
	 * 48-flow anomaly used non-shared contexts. */
	effective_ledger_bound = scheduler->profile.max_outstanding_permissions;
	if (scheduler->shared_ledger == NULL && bound_flows >= scheduler->profile.hot_min_flows &&
	    effective_ledger_bound > scheduler->usable_capacity) {
		effective_ledger_bound = scheduler->usable_capacity;
		ledger_cap_active      = true;
	}
	if (!scheduler->occupancy_sampling_active) {
		/* No sample consumed yet (sharing groups before the first
		 * published sample arrives): full-rate accrual until the
		 * sampler latches active, then staleness fails closed. */
		rate_q16 = OPX_FLOW_CONTROL_RCCC_Q16_ONE;
	} else if (scheduler->occupancy_sample_ns == 0u ||
		   now_ns - scheduler->occupancy_sample_ns > scheduler->profile.telemetry_health_deadline_ns) {
		/* Polling is required to reach this pass, so sampler expiry cannot trigger
		 * the otherwise-required receive-context failure machinery here. */
		rate_q16 = 0u;
	} else {
		occupancy_sample_fresh = true;
		occupancy	       = scheduler->occupancy;
		/* A stopped rate remains latched until a sample is strictly below low water. */
		if (scheduler->rate_latched != 0u) {
			if (occupancy < scheduler->low_water) {
				scheduler->rate_latched = 0u;
				rate_q16		= OPX_FLOW_CONTROL_RCCC_Q16_ONE;
			} else {
				rate_q16 = 0u;
			}
		} else if (occupancy >= scheduler->high_water) {
			scheduler->rate_latched = 1u;
			rate_q16		= 0u;
		} else if (occupancy <= scheduler->low_water) {
			rate_q16 = OPX_FLOW_CONTROL_RCCC_Q16_ONE;
		} else {
			rate_q16 = (uint32_t) ((((uint64_t) (scheduler->high_water - occupancy)) << 16) /
					       (scheduler->high_water - scheduler->low_water));
		}
	}
	if (scheduler->profile.duty_min_flows != 0u) {
		if (scheduler->duty_state == OPX_FLOW_CONTROL_RCCC_DUTY_NORMAL) {
			/* Shared samplers do not publish drain effort yet, so sharing
			 * enters conservatively from the persistent crest latch alone. */
			const bool drain_pressure =
				scheduler->shared_ledger != NULL ?
					scheduler->rate_latched != 0u :
					scheduler->maxed_drain_streak >= scheduler->profile.duty_streak_k;
			const bool congested = scheduler->rate_latched != 0u ||
					       (occupancy_sample_fresh && occupancy >= scheduler->high_water);

			if (drain_pressure && congested && bound_flows >= scheduler->profile.duty_min_flows) {
				scheduler->duty_state	  = OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING;
				scheduler->drain_enter_ns = now_ns;
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.drain_entries);
			}
		} else {
			/* Sustained arrival pressure makes consecutive quiet polls
			 * unachievable because every visit maxes the drain bound.  The
			 * permission floor already bounds unarrived traffic below low water.
			 * The deadline is both the liveness escape and the duty-period bound:
			 * a service-bound episode holds grants for at most one period so its
			 * re-saturation burst stays small instead of accumulating over a
			 * coarse long stop. */
			const bool drained = scheduler->outstanding_permissions <= scheduler->profile.duty_drain_floor;
			const bool forced  = now_ns - scheduler->drain_enter_ns > scheduler->profile.duty_deadline_ns;

			if (drained) {
				scheduler->duty_state = OPX_FLOW_CONTROL_RCCC_DUTY_NORMAL;
			} else if (forced) {
				scheduler->duty_state = OPX_FLOW_CONTROL_RCCC_DUTY_NORMAL;
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.drain_forced_exits);
			}
		}
	}
	accrued = scheduler->accrual_remainder_q16;
	if (scheduler->duty_state != OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING && rate_q16 != 0u) {
		product = (__uint128_t) elapsed_ns * scheduler->profile.minimum_packet_rate_q16_per_ns * rate_q16;
		accrued += (uint64_t) (product >> 16);
	}
	bulk_budget = scheduler->duty_state == OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING ? 0u : accrued >> 16;
	scheduler->accrual_remainder_q16 = accrued & UINT64_C(0xFFFF);
	/* Cold-ramp + hot-set governor: grants are monotonic within an epoch,
	 * so the initial grant arithmetic is the only effective incast defense
	 * (both fixed big ceilings and reactive scaling were falsified on
	 * hardware - by the time any signal fires the ring-dwarfing window sum
	 * is already granted).  Every flow's private ceiling starts small and
	 * ramps toward fmax while cold; only flows measured consuming real
	 * windows join the hot set and split the usable ring, so tiny-demand
	 * collectives at high peer count never see division and the
	 * micro-grant control storms cannot re-form.  Congestion freezes cold
	 * ramps; the hot share bounds Sigma(hot windows) <= usable ring by
	 * construction, independent of the trough-biased occupancy sampler.
	 * Drain pressure stays out of this predicate: a sustained two-party
	 * stream maxes every poll's drain bound while the ring is healthy, and
	 * slamming on it pinned np2 Biband at the minimum window (-44%). */
	congested = scheduler->rate_latched != 0u || (occupancy_sample_fresh && occupancy >= scheduler->high_water);
	if (scheduler->window_epoch_start_ns == 0u) {
		scheduler->window_epoch_start_ns = now_ns;
	} else if (now_ns - scheduler->window_epoch_start_ns >= scheduler->profile.window_epoch_ns) {
		/* Control-storm detector: standalone credit-request arrivals
		 * outnumbering serviced data packets is the retry-storm
		 * signature - healthy senders emit one request per exhausted
		 * window, so requests are structurally bounded by
		 * consumption/window (15-to-1 measures ~24 requests against
		 * ~1500 consumed per epoch), while a micro-window storm
		 * measured 187M standalone requests against ~0.2 consumed
		 * per flow-epoch at np=256.  While storm mode holds, the
		 * per-flow backlog ramp freeze is bypassed so ceilings can
		 * grow to dissolve the storm; two-epoch hysteresis on both
		 * edges keeps a transient receiver stall from flapping it. */
		const bool storm_raw = scheduler->request_epoch >= scheduler->profile.storm_request_floor &&
				       scheduler->request_epoch > scheduler->consumed_total_epoch;

		if (storm_raw) {
			scheduler->storm_off_streak = 0u;
			if (scheduler->storm_on_streak != UINT8_MAX) {
				scheduler->storm_on_streak++;
			}
			if (scheduler->storm_on_streak >= 2u) {
				scheduler->storm_mode = 1u;
			}
		} else {
			scheduler->storm_on_streak = 0u;
			if (scheduler->storm_off_streak != UINT8_MAX) {
				scheduler->storm_off_streak++;
			}
			if (scheduler->storm_off_streak >= 2u) {
				scheduler->storm_mode = 0u;
			}
		}
		scheduler->request_epoch	 = 0u;
		scheduler->consumed_total_epoch	 = 0u;
		scheduler->window_epoch_start_ns = now_ns;
		scheduler->window_epoch++;
	}
	hot_flows = scheduler->shared_ledger != NULL ?
			    atomic_load_explicit(&scheduler->shared_ledger->hot_flows, memory_order_relaxed) :
			    scheduler->hot_flows;
	FI_OPX_DEBUG_COUNTERS_MAX_OF(opx_ep->debug_counters.flow_control.hot_flows_high_watermark,
				     (uint64_t) hot_flows);
	if (congested) {
		scheduler->hot_ceiling = scheduler->profile.window_cap_min;
	} else if (hot_flows < scheduler->profile.hot_min_flows) {
		/* At low fan-in the fair share drops below the loopback BDP
		 * and taxes sustained bidirectional traffic (np2 Biband
		 * measured -46% at share = usable/2) while the occupancy
		 * curve already governs so few senders; division starts at
		 * the same fan-in the duty-cycler requires. */
		scheduler->hot_ceiling = tx_profile->fmax;
	} else {
		scheduler->hot_ceiling = scheduler->usable_capacity / hot_flows;
		if (scheduler->hot_ceiling < scheduler->profile.window_cap_min) {
			scheduler->hot_ceiling = scheduler->profile.window_cap_min;
		}
		if (scheduler->hot_ceiling > tx_profile->fmax) {
			scheduler->hot_ceiling = tx_profile->fmax;
		}
	}
	for (flow = opx_ep->flow_control_rx_work_head; flow != NULL; flow = flow->flow_control_work_next) {
		struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);
		uint64_t ledger_space = effective_ledger_bound > scheduler->outstanding_permissions ?
						effective_ledger_bound - scheduler->outstanding_permissions :
						0u;
		uint64_t demand;
		uint64_t grant_count;
		uint64_t window_headroom;
		uint64_t flow_ceiling;

		if (grant_emit_budget == 0u) {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.grant_pass_budget_breaks);
			break;
		}
		opx_flow_control_rccc_rx_quarantine_check(opx_ep, flow, now_ns);
		if (ledger_space == 0u) {
			if (ledger_cap_active) {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.ledger_cap_clamps);
			}
			break;
		}
		if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND ||
		    rx_state->target == rx_state->grant) {
			continue;
		}
#ifndef NDEBUG
		assert(rx_state->permission_contribution ==
		       ((rx_state->grant - rx_state->observed - rx_state->retired_accepted) &
			OPX_FLOW_CONTROL_RCCC_SERIAL_MASK));
#endif
		demand = (rx_state->target - rx_state->grant) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		if (demand >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
			opx_ep->reli_service->flow_control_feedback_counters.invalid++;
			continue;
		}
		if ((uint32_t) demand > rx_state->demand_peak_epoch) {
			rx_state->demand_peak_epoch = (uint32_t) demand;
		}
		if (rx_state->window_epoch_stamp != scheduler->window_epoch) {
			opx_flow_control_rccc_rx_window_epoch_update(
				opx_ep, scheduler, rx_state, scheduler->window_epoch - rx_state->window_epoch_stamp,
				congested);
			rx_state->window_epoch_stamp = scheduler->window_epoch;
		}
		flow_ceiling	= rx_state->hot != 0u ? scheduler->hot_ceiling : rx_state->window_ceiling;
		window_headroom = flow_ceiling > rx_state->permission_contribution ?
					  flow_ceiling - rx_state->permission_contribution :
					  0u;
		if (window_headroom == 0u) {
			continue;
		}
		if (scheduler->duty_state == OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING) {
			if (bypass_budget == 0u || demand > 4u || rx_state->permission_contribution > 4u) {
				continue;
			}
			grant_count = MIN(demand, UINT64_C(4));
			bypass_budget--;
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.drain_bypass_grants);
		} else {
			if (bulk_budget == 0u) {
				break;
			}
			assert(rx_state->drr_deficit <= INT64_MAX - (int64_t) scheduler->profile.drr_quantum);
			rx_state->drr_deficit += (int64_t) scheduler->profile.drr_quantum;
			grant_count = demand;
		}
		if (scheduler->duty_state != OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING &&
		    grant_count > (uint64_t) rx_state->drr_deficit) {
			grant_count = (uint64_t) rx_state->drr_deficit;
		}
		if (grant_count > ledger_space) {
			grant_count = ledger_space;
		}
		if (grant_count > window_headroom) {
			grant_count = window_headroom;
		}
		if (scheduler->duty_state != OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING && grant_count > bulk_budget) {
			grant_count = bulk_budget;
		}
		if (grant_count == 0u) {
			continue;
		}
		if (OFI_UNLIKELY(scheduler->shared_ledger != NULL)) {
			opx_flow_control_rccc_shared_ledger_charge(scheduler, grant_count);
		}
		rx_state->grant = opx_flow_control_rccc_serial_add(rx_state->grant, (uint32_t) grant_count);
		if (rx_state->grant == rx_state->target) {
			opx_flow_control_rccc_demand_active_dec(scheduler);
		}
		if (scheduler->duty_state != OPX_FLOW_CONTROL_RCCC_DUTY_DRAINING) {
			rx_state->drr_deficit -= (int64_t) grant_count;
			bulk_budget -= grant_count;
		}
		opx_flow_control_rccc_ledger_charge(scheduler, rx_state, (uint32_t) grant_count);
		opx_flow_control_rccc_rx_pending_set(scheduler, rx_state,
						     OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
		if (opx_flow_control_rccc_rx_emit_pending(
			    opx_ep, flow, OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING) == OPX_FLOW_CONTROL_OK) {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.grant_emit_fused);
			grant_emit_budget--;
			if (grant_emit_budget == 0u) {
				/* Stop minting when the fused delivery budget is exhausted;
				 * residual work must reach the wire before more grants exist. */
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.grant_pass_budget_breaks);
				break;
			}
		} else {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.grant_emit_deferred);
			break;
		}
	}
}

void opx_flow_control_rccc_control_sweep(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	uint16_t budget = opx_flow_control_rccc_max_control_emits_get();

	assert(opx_ep != NULL);
	assert(opx_flow_control_rccc_tx_profile_get() != NULL);
	if (budget == 0u) {
		return;
	}
	opx_flow_control_rccc_shared_release_fold(opx_ep->flow_control_scheduler);
	opx_flow_control_rccc_tx_sweep_pass(opx_ep, now_ns, true, &budget);
	opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING, &budget);
	/* Leftovers from prior sweeps must deliver before new grants are minted. */
	if (opx_ep->flow_control_scheduler->feedback_pending_count != 0u) {
		opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING, &budget);
	}
	opx_flow_control_rccc_rx_grant_pass(opx_ep, now_ns);
	if (opx_ep->flow_control_scheduler->reject_pending_count != 0u) {
		opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING, &budget);
	}
	opx_flow_control_rccc_tx_sweep_pass(opx_ep, now_ns, false, &budget);
#ifndef NDEBUG
	opx_flow_control_rccc_ledger_reconcile(opx_ep);
#endif
}

/* Fused admit + record-free single-packet charge for get_replay: the
 * demand check and the credit charge share one tx_state access, and the
 * caller's redundant pre-admit rbtree find is eliminated (the flow is already
 * resolved).  The held interval is strictly intra-send: commit or cancel
 * follows within the same send invocation. */
enum opx_flow_control_result opx_flow_control_tx_admit_reserve_single(struct fi_opx_ep			*opx_ep,
								      struct fi_opx_reliability_tx_flow *flow,
								      void **reservation_record_out)
{
	const struct opx_flow_control_rccc_tx_profile *profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_tx_state	      *tx_state;
	enum opx_flow_control_result		       result;

	assert(opx_ep != NULL);
	assert(flow != NULL);
	assert(reservation_record_out != NULL);
	assert(profile != NULL);
	*reservation_record_out = NULL;
	tx_state		= opx_flow_control_rccc_tx_state(flow);
	result			= opx_flow_control_rccc_tx_demand(tx_state, profile, 1u);
	if (result != OPX_FLOW_CONTROL_FAULT && opx_flow_control_rccc_tx_work_needed(tx_state)) {
		opx_flow_control_rccc_tx_work_enqueue(opx_ep, flow);
	}
	if (result != OPX_FLOW_CONTROL_OK) {
		return result;
	}
	/* OK with zero credits is exactly the fail-open lane: the packet
	 * admits without touching the credit ledger (identical to the
	 * FLOW_CONTROL=none baseline) via the sentinel record every
	 * registration/settle/cancel consumer no-ops on. */
	if (OFI_UNLIKELY(tx_state->available_credits == 0u)) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.ungoverned_admissions);
		*reservation_record_out = OPX_FLOW_CONTROL_UNGOVERNED_RECORD;
		return OPX_FLOW_CONTROL_OK;
	}
	tx_state->available_credits--;
	tx_state->held_credits++;
	*reservation_record_out = OPX_FLOW_CONTROL_SINGLE_TAG(flow);
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result opx_flow_control_tx_reserve(struct fi_opx_ep		   *opx_ep,
							 struct fi_opx_reliability_tx_flow *flow,
							 enum opx_flow_control_packet_class packet_class,
							 uint32_t requested_packets, void **reservation_record_out)
{
	struct opx_flow_control_rccc_reservation_record *record;
	struct opx_flow_control_rccc_tx_state		*tx_state;
	uint64_t					 generation;
	enum opx_flow_control_result			 result;

	assert(opx_ep != NULL);
	assert(flow != NULL);
	assert(reservation_record_out != NULL);
	assert(requested_packets != 0u);
	*reservation_record_out = NULL;
	tx_state		= opx_flow_control_rccc_tx_state(flow);
	if (tx_state->available_credits < requested_packets) {
		if (OFI_UNLIKELY((tx_state->request_retry.progress_flags & OPX_FLOW_CONTROL_RCCC_RETRY_UNGOVERNED) !=
				 0u)) {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.flow_control.ungoverned_admissions);
			*reservation_record_out = OPX_FLOW_CONTROL_UNGOVERNED_RECORD;
			return OPX_FLOW_CONTROL_OK;
		}
		return OPX_FLOW_CONTROL_DEFER;
	}
	assert(opx_ep->flow_control_reservation_pool != NULL);
	if (ofi_bufpool_empty(opx_ep->flow_control_reservation_pool)) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	record = ofi_buf_alloc(opx_ep->flow_control_reservation_pool);
	assert(record != NULL);
	generation = atomic_fetch_add_explicit(&opx_flow_control_rccc_reservation_generation, UINT64_C(1),
					       memory_order_relaxed) +
		     UINT64_C(1);
	if (generation == 0u) {
		abort();
	}
	result = opx_flow_control_rccc_reservation_hold(opx_ep, flow, &tx_state->flow_key, tx_state, record,
							packet_class, generation, requested_packets);
	if (result != OPX_FLOW_CONTROL_OK) {
		assert(result == OPX_FLOW_CONTROL_DEFER);
		OPX_BUF_FREE(record);
		return result;
	}
	*reservation_record_out = record;
	return OPX_FLOW_CONTROL_OK;
}

void opx_flow_control_tx_reduce_reservation(struct fi_opx_ep *opx_ep, void *reservation_record, uint32_t new_count)
{
	struct opx_flow_control_rccc_reservation_record *record = reservation_record;
	struct opx_flow_control_rccc_tx_state		*tx_state;

	assert(record != NULL);
	if (OPX_FLOW_CONTROL_IS_UNGOVERNED(reservation_record)) {
		return;
	}
	opx_flow_control_rccc_reservation_validate(record, opx_ep, record->generation);
	tx_state = opx_flow_control_rccc_tx_state(record->flow_identity);
	opx_flow_control_rccc_reservation_reduce(opx_ep, tx_state, record, record->generation, new_count);
}

void opx_flow_control_tx_register_packet(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	struct opx_flow_control_rccc_reservation_record *record = reservation_record;
	struct opx_flow_control_rccc_tx_state		*tx_state;
	bool						 first_commit;

	assert(record != NULL);
	if (OPX_FLOW_CONTROL_IS_UNGOVERNED(reservation_record)) {
		return;
	}
	if (OPX_FLOW_CONTROL_IS_SINGLE(reservation_record)) {
		tx_state = opx_flow_control_rccc_tx_state(OPX_FLOW_CONTROL_SINGLE_FLOW(reservation_record));
		assert(tx_state->held_credits != 0u);
		tx_state->held_credits--;
		tx_state->committed_originals++;
		return;
	}
	opx_flow_control_rccc_reservation_validate(record, opx_ep, record->generation);
	tx_state = opx_flow_control_rccc_tx_state(record->flow_identity);
	/* HELD records still carry identity_cookie in the union slot shared
	 * with flow_next.  Register first so validation observes the intact
	 * cookie and the state transitions to COMMITTED, then repurpose the
	 * union slot as the committed-FIFO link on the first commit only. */
	first_commit = (record->state == OPX_FLOW_CONTROL_RCCC_RESERVATION_HELD);
	opx_flow_control_rccc_reservation_register(opx_ep, tx_state, record, record->generation);
	if (first_commit) {
		record->flow_next = NULL;
		opx_flow_control_rccc_committed_append(record->flow_identity, record);
	}
}

void opx_flow_control_tx_complete_packet(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	struct opx_flow_control_rccc_reservation_record *record = reservation_record;
	struct fi_opx_reliability_tx_flow		*flow;

	assert(record != NULL);
#ifndef NDEBUG
	opx_flow_control_rccc_reservation_validate(record, opx_ep, record->generation);
#endif
	flow = record->flow_identity;
	assert(opx_flow_control_rccc_tx_state(flow)->committed_originals != 0u);
	opx_flow_control_rccc_reservation_complete(opx_ep, record, record->generation);
	while (flow->flow_control.committed_head != NULL &&
	       opx_flow_control_rccc_reservation_recyclable(flow->flow_control.committed_head)) {
		record				  = flow->flow_control.committed_head;
		flow->flow_control.committed_head = record->flow_next;
		if (flow->flow_control.committed_head == NULL) {
			flow->flow_control.committed_tail = NULL;
		}
		OPX_BUF_FREE(record);
	}
	opx_flow_control_rccc_tx_work_enqueue(opx_ep, flow);
}

void opx_flow_control_tx_cancel(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	struct opx_flow_control_rccc_reservation_record *record = reservation_record;
	struct opx_flow_control_rccc_tx_state		*tx_state;

	assert(record != NULL);
	if (OPX_FLOW_CONTROL_IS_UNGOVERNED(reservation_record)) {
		return;
	}
	if (OPX_FLOW_CONTROL_IS_SINGLE(reservation_record)) {
		tx_state = opx_flow_control_rccc_tx_state(OPX_FLOW_CONTROL_SINGLE_FLOW(reservation_record));
		assert(tx_state->held_credits != 0u);
		tx_state->held_credits--;
		tx_state->available_credits++;
		return;
	}
	opx_flow_control_rccc_reservation_validate(record, opx_ep, record->generation);
	tx_state = opx_flow_control_rccc_tx_state(record->flow_identity);
	opx_flow_control_rccc_reservation_cancel(opx_ep, tx_state, record, record->generation);
	OPX_BUF_FREE(record);
}

uint64_t opx_flow_control_tx_record_generation(const void *reservation_record)
{
	const struct opx_flow_control_rccc_reservation_record *record = reservation_record;

	assert(record != NULL);
	assert(record->generation != 0u);
	return record->generation;
}

/* Cold failure-path classifier: distinguishes a terminal current-generation
 * Capacity Reject (FI_ENOSPC) from ordinary transient defers
 * (FI_EAGAIN) after an admission seam returned failure.  Pool exhaustion and
 * credit shortage without a reject stay FI_EAGAIN. */
bool __attribute__((cold)) opx_flow_control_tx_capacity_rejected(const struct fi_opx_reliability_tx_flow *flow)
{
	if (flow == NULL) {
		return false;
	}
	return opx_flow_control_rccc_tx_state((struct fi_opx_reliability_tx_flow *) flow)->block_reason ==
	       OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY;
}
