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
	const struct opx_flow_control_rccc_scheduler_profile *profile = opx_flow_control_rccc_scheduler_profile_get();
	const struct opx_flow_control_rccc_tx_profile	     *tx_profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_scheduler_geometry	      geometry;
	struct opx_flow_control_rccc_scheduler_member	     *member;
	struct opx_flow_control_rccc_scheduler_header	     *header;
	uint8_t						     *base;
	uint64_t					      deadline_ns;
	bool						      creator;
	int						      fd;
	int						      written;

	if (scheduler == NULL || profile == NULL || tx_profile == NULL || job_key_str == NULL ||
	    scheduler->shared_header != NULL || subctxt_cnt < 2u ||
	    subctxt_cnt > FI_OPX_FLOW_CONTROL_SCHEDULER_MAX_MEMBERS || subctxt >= subctxt_cnt) {
		return -FI_EINVAL;
	}
	/* The context capacity is statically partitioned: each member grants
	 * only against its 1/member_count share.  Members run independent
	 * DRR passes, so an unpartitioned shared pool splits between flows
	 * that each need a full admission batch atomically and can never
	 * consolidate - an absorbing livelock (proven at max_outstanding=64,
	 * 2 members, 64-packet TID batches: the pool wedges 62/2).  A share
	 * below fmax could still leave a single batch ungrantable, so such
	 * geometry fails the attach closed. */
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
	scheduler->shared_length		       = geometry.mapping_length;
	scheduler->shared_member_index		       = subctxt;
	scheduler->shared_member_count		       = subctxt_cnt;
	scheduler->profile.max_outstanding_permissions = member_share;

	member = opx_flow_control_rccc_shared_member(scheduler, subctxt);
	if (member->identity.state == OPX_FLOW_CONTROL_RCCC_MEMBER_JOINED) {
		scheduler->shared_base	 = NULL;
		scheduler->shared_header = NULL;
		scheduler->shared_ledger = NULL;
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
	remaining = atomic_fetch_sub_explicit(&scheduler->shared_ledger->attached_count, 1u, memory_order_acq_rel);
	munmap(scheduler->shared_base, scheduler->shared_length);
	if (remaining == 1u) {
		shm_unlink(scheduler->shared_name);
	}
	scheduler->shared_base	 = NULL;
	scheduler->shared_header = NULL;
	scheduler->shared_ledger = NULL;
	scheduler->shared_length = 0u;
}

int __attribute__((cold)) opx_flow_control_rccc_scheduler_create(struct opx_flow_control_rccc_scheduler **scheduler)
{
	const struct opx_flow_control_rccc_scheduler_profile *profile = opx_flow_control_rccc_scheduler_profile_get();
	const struct opx_flow_control_rccc_tx_profile	     *tx_profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_scheduler		     *created;
	const uint32_t					      ffirst = tx_profile != NULL ? tx_profile->ffirst : 0u;

	if (scheduler == NULL || *scheduler != NULL || profile == NULL || ffirst == 0u ||
	    profile->decision_capacity < ffirst || profile->reject_capacity < ffirst ||
	    profile->tracked_capacity == 0u || profile->max_outstanding_permissions == 0u ||
	    profile->max_outstanding_permissions >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE ||
	    profile->drr_quantum == 0u || profile->drr_quantum > profile->max_outstanding_permissions ||
	    profile->quarantine_deadline_ns == 0u || profile->minimum_packet_rate_q16_per_ns == 0u ||
	    profile->maximum_elapsed_ns == 0u) {
		return -FI_EINVAL;
	}
	created = calloc(1u, sizeof(*created));
	if (created == NULL) {
		return -FI_ENOMEM;
	}
	created->decisions = calloc(profile->decision_capacity, sizeof(*created->decisions));
	created->rejects   = calloc(profile->reject_capacity, sizeof(*created->rejects));
	if (created->decisions == NULL || created->rejects == NULL) {
		free(created->decisions);
		free(created->rejects);
		free(created);
		return -FI_ENOMEM;
	}
	created->profile	       = *profile;
	created->allocation_generation = 1u;
	*scheduler		       = created;
	return FI_SUCCESS;
}

void __attribute__((cold)) opx_flow_control_rccc_scheduler_destroy(struct opx_flow_control_rccc_scheduler **scheduler)
{
	if (scheduler == NULL || *scheduler == NULL) {
		return;
	}
	/* Endpoint teardown parallels the reservation-pool live-record
	 * invariant: an unquiesced ledger means permissions were granted
	 * that were never observed/settled/reclaimed - fail closed. */
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
	       opx_flow_control_rccc_retry_initial_is_valid(profile->request_retry_initial_us) &&
	       opx_flow_control_rccc_retry_initial_is_valid(profile->retirement_retry_initial_us) &&
	       opx_flow_control_rccc_retry_initial_is_valid(profile->reset_retry_initial_us);
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

enum opx_flow_control_result opx_flow_control_rccc_tx_demand(struct opx_flow_control_rccc_tx_state	   *tx_state,
							     const struct opx_flow_control_rccc_tx_profile *profile,
							     uint32_t requested_packets)
{
	uint32_t candidate_target;

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
			tx_state->control_deadline_ns  = 0u;
			tx_state->request_retry.active = 0u;
		}
		return OPX_FLOW_CONTROL_OK;
	}
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) != 0u) {
		/* The reject was terminal for this generation; the first
		 * subsequent governed send with no usable credit initiates
		 * RESET and still reports FI_ENOSPC upward. */
		tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY;
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
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) == 0u) {
		tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
		tx_state->control_deadline_ns  = 0u;
		tx_state->request_retry.active = 0u;
	}
	tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_NO_CREDIT;
	return OPX_FLOW_CONTROL_DEFER;
}

static void opx_flow_control_rccc_tx_idle_retire(struct opx_flow_control_rccc_tx_state *tx_state)
{
	tx_state->retired	    = opx_flow_control_rccc_serial_add(tx_state->retired, tx_state->available_credits);
	tx_state->available_credits = 0u;
	if ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) == 0u &&
	    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING) == 0u) {
		tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING;
		tx_state->control_deadline_ns	  = 0u;
		tx_state->retirement_retry.active = 0u;
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
		tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_GRANT_SINCE_EMIT;
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
		opx_flow_control_rccc_retry_next(retry->delay_us, &retry->delay_us);
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
		tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED;
		if (tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY ||
		    tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_NO_CREDIT) {
			tx_state->block_reason = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
		}
		if (tx_state->grant == tx_state->target &&
		    (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING) != 0u) {
			tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
			tx_state->request_retry.active = 0u;
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
					 struct opx_flow_control_rccc_feedback_counters *counters)
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
	tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED;
	tx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING;
	tx_state->request_retry.active = 0u;
	tx_state->block_reason	       = OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY;
	if (tx_state->retired != tx_state->retired_acked) {
		tx_state->flags |= OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING;
		tx_state->control_deadline_ns	  = 0u;
		tx_state->retirement_retry.active = 0u;
	}
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
	tx_state->desired_window       = (uint16_t) profile->ffirst;
	tx_state->request_retry.active = 0u;
	tx_state->reset_retry.active   = 0u;
	tx_state->reset_retry.delay_us = profile->reset_retry_initial_us;
	tx_state->reset_deadline_ns    = now_ns;
	tx_state->block_reason	       = OPX_FLOW_CONTROL_RCCC_BLOCK_RESET;
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
	tx_state->epoch		      = opx_flow_control_rccc_serial_add(tx_state->epoch, 1u);
	tx_state->grant		      = 0u;
	tx_state->target	      = 0u;
	tx_state->retired	      = 0u;
	tx_state->retired_acked	      = 0u;
	tx_state->available_credits   = 0u;
	tx_state->committed_originals = 0u;
	tx_state->fault_contained     = 0u;
	tx_state->request_generation  = 0u;
	tx_state->flags &=
		(uint8_t) ~(OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING |
			    OPX_FLOW_CONTROL_RCCC_TX_FLAG_WINDOW_GROWTH_ARMED);
	tx_state->reset_retry.active = 0u;
	tx_state->block_reason	     = OPX_FLOW_CONTROL_RCCC_BLOCK_NONE;
	opx_flow_control_rccc_tx_invariant_check(tx_state);
	return OPX_FLOW_CONTROL_OK;
}

static inline bool opx_flow_control_rccc_tx_work_needed(const struct opx_flow_control_rccc_tx_state *tx_state)
{
	return (tx_state->flags &
		(OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING | OPX_FLOW_CONTROL_RCCC_TX_FLAG_RETIREMENT_PENDING |
		 OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING)) != 0u ||
	       tx_state->available_credits != 0u ||
	       ((tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) != 0u &&
		tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY);
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
	/* Demand is advisory: a message needing more packets than one batch
	 * request may carry (fmax >= MAX_BATCH_REQUEST is validated at init)
	 * asks for the maximum and re-demands from the deferred-work requeue
	 * when the per-packet reservations exhaust the granted window. */
	if (requested_packets > OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST) {
		requested_packets = OPX_FLOW_CONTROL_RCCC_MAX_BATCH_REQUEST;
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
	if (profile == NULL || opx_flow_control_rccc_tx_window_open(tx_state, profile) != FI_SUCCESS) {
		return -FI_EINVAL;
	}
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
	rx_state->flags =
		OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING | OPX_FLOW_CONTROL_RCCC_RX_FLAG_SNAPSHOT_VALID;
	opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
}

/* Release-state table: the first accepted copy of a governed
 * packet releases exactly one permission while BOUND, RESETTING, or
 * pre-reclaim QUARANTINED; an arrival on an unbound (DORMANT/PREPARING)
 * entry, beyond the outstanding window, or after quarantine reclaim is a
 * flow-local protocol fault that releases zero.  Duplicates never reach
 * acceptance, so exactly-once needs no per-PSN state here. */
void opx_flow_control_rx_first_accept(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow, uint32_t psn)
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
	 * the pools untouched (model decide_accept contract).  Sharing:
	 * tracked capacity is context-wide, taken from the shared counter
	 * with a bounded CAS after the local gates pass. */
	if (scheduler->decision_used >= scheduler->profile.decision_capacity ||
	    (consume_tracked && scheduler->tracked_used >= scheduler->profile.tracked_capacity)) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	if (consume_tracked && scheduler->shared_ledger != NULL &&
	    !opx_flow_control_rccc_shared_tracked_reserve(scheduler)) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	record			   = &scheduler->decisions[scheduler->decision_used];
	record->key		   = *key;
	record->request_generation = request_generation;
	record->reject_slot	   = 0u;
	record->state		   = OPX_FLOW_CONTROL_RCCC_DECISION_ACCEPT;
	record->occupied	   = 1u;
	record->grant_seen	   = 0u;
	*decision_slot		   = scheduler->decision_used;
	scheduler->decision_used++;
	if (consume_tracked) {
		scheduler->tracked_used++;
	}
	return OPX_FLOW_CONTROL_OK;
}

static inline enum opx_flow_control_result opx_flow_control_rccc_decide_capacity_reject(
	struct opx_flow_control_rccc_scheduler *scheduler, const struct opx_flow_control_rccc_decision_key *key,
	uint32_t request_generation, enum opx_flow_control_rccc_capacity_reason reason, uint32_t limit,
	uint32_t observed, uint32_t *reject_slot)
{
	struct opx_flow_control_rccc_decision_record *record = opx_flow_control_rccc_decision_find(scheduler, key);
	struct opx_flow_control_rccc_reject_record   *reject;

	if (record != NULL) {
		if (OFI_UNLIKELY(record->state != OPX_FLOW_CONTROL_RCCC_DECISION_REJECT)) {
			return OPX_FLOW_CONTROL_FAULT;
		}
		*reject_slot = record->reject_slot;
		return OPX_FLOW_CONTROL_OK;
	}
	if (scheduler->decision_used >= scheduler->profile.decision_capacity ||
	    scheduler->reject_used >= scheduler->profile.reject_capacity) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	record			   = &scheduler->decisions[scheduler->decision_used];
	reject			   = &scheduler->rejects[scheduler->reject_used];
	record->key		   = *key;
	record->request_generation = request_generation;
	record->reject_slot	   = scheduler->reject_used;
	record->state		   = OPX_FLOW_CONTROL_RCCC_DECISION_REJECT;
	record->occupied	   = 1u;
	record->grant_seen	   = 0u;
	reject->key		   = *key;
	reject->request_generation = request_generation;
	reject->limit		   = limit;
	reject->observed	   = observed;
	reject->fi_errno	   = FI_ENOSPC;
	reject->reason		   = (uint8_t) reason;
	reject->occupied	   = 1u;
	*reject_slot		   = scheduler->reject_used;
	scheduler->decision_used++;
	scheduler->reject_used++;
	return OPX_FLOW_CONTROL_OK;
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
	struct opx_flow_control_rccc_scheduler *scheduler    = opx_ep->flow_control_scheduler;
	uint64_t				contribution = 0u;
	RbtIterator				itr;

	if (scheduler == NULL || opx_ep->reli_service == NULL || opx_ep->reli_service->rx.rx_flow_rbtree == NULL) {
		return;
	}
	for (itr = fi_opx_rbt_begin(opx_ep->reli_service->rx.rx_flow_rbtree); itr != NULL;
	     itr = fi_opx_rbt_next(opx_ep->reli_service->rx.rx_flow_rbtree, itr)) {
		struct fi_opx_reliability_rx_flow *flow =
			fi_opx_rbt_value(opx_ep->reli_service->rx.rx_flow_rbtree, itr);

		contribution += opx_flow_control_rccc_rx_state(flow)->permission_contribution;
	}
	assert(contribution == scheduler->outstanding_permissions);
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
 * defect), then the sticky decision pool either binds the pairing
 * (PREPARING is instantaneous under process-local single-threaded
 * authority) or records a Capacity Reject for kind-8 emission.  A replayed
 * kind-3 for a decided pairing refreshes only the echoed reply serial. */
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
	/* Sticky replay: a decided pairing always gets its original outcome.
	 * A retried kind-3 against a rejected pairing re-emits the kind-8
	 * echoing the retry's reply serial instead of re-evaluating. */
	{
		const struct opx_flow_control_rccc_decision_record *existing =
			opx_flow_control_rccc_decision_find(scheduler, &key);

		if (existing != NULL && existing->state == OPX_FLOW_CONTROL_RCCC_DECISION_REJECT) {
			scheduler->rejects[existing->reject_slot].request_generation = envelope->value[2];
			rx_state->reject_slot					     = existing->reject_slot;
			rx_state->request_generation				     = envelope->value[2];
			rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING;
			opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
			return OPX_FLOW_CONTROL_OK;
		}
	}
	result = opx_flow_control_rccc_decide_accept(scheduler, &key, envelope->value[2],
						     rx_state->scheduler.generation == 0u, &slot);
	if (OFI_UNLIKELY(result == OPX_FLOW_CONTROL_FAULT)) {
		opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
		return OPX_FLOW_CONTROL_FAULT;
	}
	if (result == OPX_FLOW_CONTROL_DEFER) {
		const enum opx_flow_control_rccc_capacity_reason reason =
			scheduler->decision_used >= scheduler->profile.decision_capacity ?
				OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_DECISION :
				OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_TRACKED;
		const uint32_t limit	= reason == OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_DECISION ?
						  scheduler->profile.decision_capacity :
						  scheduler->profile.tracked_capacity;
		const uint32_t observed = reason == OPX_FLOW_CONTROL_RCCC_CAPACITY_REASON_DECISION ?
						  scheduler->decision_used :
						  scheduler->tracked_used;

		result = opx_flow_control_rccc_decide_capacity_reject(scheduler, &key, envelope->value[2], reason,
								      limit, observed, &slot);
		if (result != OPX_FLOW_CONTROL_OK) {
			return result;
		}
		scheduler->rejects[slot].request_generation = envelope->value[2];
		rx_state->reject_slot			    = slot;
		rx_state->request_generation		    = envelope->value[2];
		rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING;
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
	rx_state->scheduler_state    = OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND;
	rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING;
	opx_flow_control_rccc_rx_work_enqueue(opx_ep, flow);
	return OPX_FLOW_CONTROL_OK;
}

enum opx_flow_control_result opx_flow_control_rccc_rx_control(struct fi_opx_ep				  *opx_ep,
							      struct fi_opx_reliability_rx_flow		  *flow,
							      const struct opx_flow_control_wire_envelope *envelope)
{
	const struct opx_flow_control_rccc_tx_profile	*profile = opx_flow_control_rccc_tx_profile_get();
	struct opx_flow_control_rccc_feedback_counters	*counters;
	struct opx_flow_control_rccc_rx_state		*rx_state;
	enum opx_flow_control_rccc_serial_classification classification;
	enum opx_flow_control_result			 result;
	uint32_t					 delta;
	uint32_t					 outstanding;

	assert(opx_ep != NULL);
	assert(opx_ep->reli_service != NULL);
	assert(flow != NULL);
	assert(envelope != NULL);
	assert(profile != NULL);
	counters = &opx_ep->reli_service->flow_control_feedback_counters;
	rx_state = opx_flow_control_rccc_rx_state(flow);
	switch (envelope->kind) {
	case OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST:
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
		classification = opx_flow_control_rccc_serial_classify(rx_state->target, envelope->value[0]);
		opx_flow_control_rccc_feedback_count(counters, classification);
		if (classification == OPX_FLOW_CONTROL_RCCC_SERIAL_NEWER) {
			delta = (envelope->value[0] - rx_state->target) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;

			if (OFI_UNLIKELY(delta > profile->fmax)) {
				opx_flow_control_rccc_feedback_count(counters, OPX_FLOW_CONTROL_RCCC_SERIAL_INVALID);
				return OPX_FLOW_CONTROL_FAULT;
			}
			rx_state->target = envelope->value[0];
		} else if (classification != OPX_FLOW_CONTROL_RCCC_SERIAL_DUPLICATE &&
			   classification != OPX_FLOW_CONTROL_RCCC_SERIAL_STALE) {
			return OPX_FLOW_CONTROL_FAULT;
		}
		rx_state->request_generation = envelope->value[2];
		result = opx_flow_control_rccc_rx_retirement_settle(opx_ep, rx_state, envelope->value[1], counters);
		if (result != OPX_FLOW_CONTROL_OK) {
			return result;
		}
		rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING;
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
		rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING;
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
			rx_state->pending_epoch		 = opx_flow_control_rccc_serial_add(rx_state->epoch, 1u);
			rx_state->scheduler_state	 = OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING;
			rx_state->quarantine_deadline_ns = 0u;
			rx_state->flags &= (uint8_t) ~OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING;
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
			/* Duplicate RESET while withholding: idempotent
			 * no-op - the ACK is not yet releasable. */
			return OPX_FLOW_CONTROL_OK;
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
		const struct opx_flow_control_rccc_scheduler *scheduler = opx_ep->flow_control_scheduler;

		/* Kind-8 wire shape: generation carries the
		 * request_generation reply serial, value[0] echoes the
		 * rejected epoch, and reason/limit/observed stay local. */
		assert(scheduler != NULL && rx_state->reject_slot < scheduler->profile.reject_capacity);
		envelope.generation = rx_state->request_generation;
		envelope.value[0]   = scheduler->rejects[rx_state->reject_slot].key.epoch;
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
		rx_state->flags &= (uint8_t) ~pending_flag;
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

		if (reset_pass && (tx_state->flags & OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED) != 0u &&
		    tx_state->block_reason == OPX_FLOW_CONTROL_RCCC_BLOCK_CAPACITY &&
		    tx_state->available_credits == 0u && tx_state->held_credits == 0u &&
		    flow->flow_control.committed_head == NULL) {
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
	const struct opx_flow_control_rccc_scheduler_profile *profile  = opx_flow_control_rccc_scheduler_profile_get();
	struct opx_flow_control_rccc_rx_state		     *rx_state = opx_flow_control_rccc_rx_state(flow);

	if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_RESETTING || profile == NULL) {
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
	rx_state->flags		  = OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED;
	opx_ep->reli_service->flow_control_feedback_counters.invalid++;
}

/* DRR grant tier: Q16.16 accrual (synthetic full rate) sets a
 * bulk budget separate from the control-emission budget; each listed BOUND
 * flow with unmet demand earns one quantum per visit and receives
 * min(demand, deficit, ledger headroom, budget).  Deficits carry across
 * polls; the ledger cap is the aggregate outstanding-permission bound. */
static void opx_flow_control_rccc_rx_grant_pass(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	struct opx_flow_control_rccc_scheduler *scheduler = opx_ep->flow_control_scheduler;
	struct fi_opx_reliability_rx_flow      *flow;
	__uint128_t				product;
	uint64_t				elapsed_ns;
	uint64_t				accrued;
	uint64_t				bulk_budget;

	if (scheduler == NULL) {
		return;
	}
	if (scheduler->accrual_last_ns == 0u) {
		scheduler->accrual_last_ns = now_ns;
		return;
	}
	elapsed_ns		   = now_ns - scheduler->accrual_last_ns;
	scheduler->accrual_last_ns = now_ns;
	if (elapsed_ns > scheduler->profile.maximum_elapsed_ns) {
		elapsed_ns = scheduler->profile.maximum_elapsed_ns;
	}
	product = (__uint128_t) elapsed_ns * scheduler->profile.minimum_packet_rate_q16_per_ns *
		  OPX_FLOW_CONTROL_RCCC_Q16_ONE;
	accrued				 = scheduler->accrual_remainder_q16 + (uint64_t) (product >> 16);
	bulk_budget			 = accrued >> 16;
	scheduler->accrual_remainder_q16 = accrued & UINT64_C(0xFFFF);
	for (flow = opx_ep->flow_control_rx_work_head; flow != NULL; flow = flow->flow_control_work_next) {
		struct opx_flow_control_rccc_rx_state *rx_state = opx_flow_control_rccc_rx_state(flow);
		uint64_t			       ledger_space =
			scheduler->profile.max_outstanding_permissions - scheduler->outstanding_permissions;
		uint64_t demand;
		uint64_t grant_count;

		opx_flow_control_rccc_rx_quarantine_check(opx_ep, flow, now_ns);
		if (bulk_budget == 0u || ledger_space == 0u) {
			break;
		}
		if (rx_state->scheduler_state != OPX_FLOW_CONTROL_RCCC_SCHEDULER_BOUND ||
		    rx_state->target == rx_state->grant) {
			continue;
		}
		demand = (rx_state->target - rx_state->grant) & OPX_FLOW_CONTROL_RCCC_SERIAL_MASK;
		if (demand >= OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE) {
			opx_ep->reli_service->flow_control_feedback_counters.invalid++;
			continue;
		}
		assert(rx_state->drr_deficit <= INT64_MAX - (int64_t) scheduler->profile.drr_quantum);
		rx_state->drr_deficit += (int64_t) scheduler->profile.drr_quantum;
		grant_count = demand;
		if (grant_count > (uint64_t) rx_state->drr_deficit) {
			grant_count = (uint64_t) rx_state->drr_deficit;
		}
		if (grant_count > ledger_space) {
			grant_count = ledger_space;
		}
		if (grant_count > bulk_budget) {
			grant_count = bulk_budget;
		}
		if (grant_count == 0u) {
			continue;
		}
		if (OFI_UNLIKELY(scheduler->shared_ledger != NULL)) {
			opx_flow_control_rccc_shared_ledger_charge(scheduler, grant_count);
		}
		rx_state->grant = opx_flow_control_rccc_serial_add(rx_state->grant, (uint32_t) grant_count);
		rx_state->drr_deficit -= (int64_t) grant_count;
		opx_flow_control_rccc_ledger_charge(scheduler, rx_state, (uint32_t) grant_count);
		bulk_budget -= grant_count;
		rx_state->flags |= OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING;
	}
}

void opx_flow_control_rccc_control_sweep(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	uint16_t budget = opx_flow_control_rccc_max_control_emits_get();

	if (opx_ep == NULL || opx_flow_control_rccc_tx_profile_get() == NULL || budget == 0u) {
		return;
	}
	opx_flow_control_rccc_shared_release_fold(opx_ep->flow_control_scheduler);
	opx_flow_control_rccc_tx_sweep_pass(opx_ep, now_ns, true, &budget);
	opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING, &budget);
	opx_flow_control_rccc_rx_grant_pass(opx_ep, now_ns);
	opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_REJECT_PENDING, &budget);
	opx_flow_control_rccc_rx_sweep_pass(opx_ep, OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING, &budget);
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
	*reservation_record_out = NULL;
	if (profile == NULL) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	tx_state = opx_flow_control_rccc_tx_state(flow);
	result	 = opx_flow_control_rccc_tx_demand(tx_state, profile, 1u);
	if (result != OPX_FLOW_CONTROL_FAULT && opx_flow_control_rccc_tx_work_needed(tx_state)) {
		opx_flow_control_rccc_tx_work_enqueue(opx_ep, flow);
	}
	if (result != OPX_FLOW_CONTROL_OK) {
		return result;
	}
	assert(tx_state->available_credits != 0u);
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
