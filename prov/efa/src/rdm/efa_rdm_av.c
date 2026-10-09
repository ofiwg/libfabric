/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <malloc.h>
#include <stdio.h>

#include <infiniband/efadv.h>
#include <ofi_enosys.h>

#include "efa.h"
#include "../efa_av.h"
#include "efa_rdm_av.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_fabric.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_pke_utils.h"

/*
 * The efa_rdm_ah_* helpers layer the RDM-only AH policy (split
 * explicit/implicit reference counts, the per-domain AH LRU list and
 * out-of-memory eviction of implicit-only AHs) on top of the base efa_ah_*
 * functions in efa_ah.c.
 */

/**
 * @brief move an AH to the tail of the domain's AH LRU list
 *
 * This is not called on the explicit AV insertion critical path so that we
 * don't add extra latency there. The LRU list is only used to pick AH entries
 * with only implicit AV entries for eviction, so that is OK.
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	address handle
 */
static void efa_rdm_ah_implicit_av_lru_ah_move(struct efa_domain *domain,
					struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_ah *rdm_ah = ((struct efa_rdm_ah *)(ah));
	struct efa_rdm_domain *rdm_domain;

	assert(domain->info_type == EFA_INFO_RDM);
	assert(ofi_genlock_held(&domain->util_domain.lock));

	rdm_domain = (struct efa_rdm_domain *) domain;
	assert(rdm_ah->implicit_refcnt > 0 || rdm_ah->explicit_refcnt > 0);
	assert(dlist_entry_in_list(&rdm_domain->ah_lru_list,
				   &rdm_ah->domain_lru_ah_list_entry));

	dlist_remove(&rdm_ah->domain_lru_ah_list_entry);
	dlist_insert_tail(&rdm_ah->domain_lru_ah_list_entry,
			  &rdm_domain->ah_lru_list);
}

/**
 * @brief destroy an RDM AH: unlink it from the domain's AH LRU list, then run
 * the base teardown. LRU-list membership is RDM-only state.
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	address handle
 */
static void efa_rdm_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_ah *rdm_ah = ((struct efa_rdm_ah *)(ah));

	dlist_remove(&rdm_ah->domain_lru_ah_list_entry);
	efa_ah_destroy_ah(domain, ah);
}

/**
 * @brief evict an AH that has only implicit AV entries to free device resources
 *
 * @param[in]	domain	efa domain
 * @param[in]	insert_implicit_av	true when inserting for an implicit AV entry
 */
static int efa_rdm_ah_implicit_av_evict_ah(struct efa_domain *domain,
					   bool insert_implicit_av)
	OFI_TSA_NO_ANALYSIS // clang cannot reason about conditional locking statically
{
	struct efa_rdm_av_entry *av_entry_to_release;
	struct efa_rdm_ah *rdm_ah_tmp, *rdm_ah_to_release = NULL;
	struct dlist_entry *tmp;
	struct efa_rdm_domain *rdm_domain;

	assert(domain->info_type == EFA_INFO_RDM);
	assert(ofi_genlock_held(&domain->util_domain.lock));
	rdm_domain = (struct efa_rdm_domain *) domain;

	dlist_foreach_container (&rdm_domain->ah_lru_list, struct efa_rdm_ah,
				 rdm_ah_tmp, domain_lru_ah_list_entry) {
		if (rdm_ah_tmp->explicit_refcnt == 0) {
			rdm_ah_to_release = rdm_ah_tmp;
			break;
		}
	}

	if (!rdm_ah_to_release) {
		EFA_WARN(FI_LOG_AV,
			 "AH creation for implicit AV entry failed with ENOMEM "
			 "but no AH entries available to evict\n");
		return -FI_ENOMEM;
	}

	assert(rdm_ah_to_release->implicit_refcnt > 0);

	dlist_foreach_container_safe(&rdm_ah_to_release->implicit_conn_list,
				      struct efa_rdm_av_entry, av_entry_to_release,
				      ah_implicit_conn_list_entry, tmp) {

		assert(efa_rdm_av_entry_implicit_fi_addr(av_entry_to_release) != FI_ADDR_NOTAVAIL &&
		       efa_rdm_av_entry_fi_addr(av_entry_to_release) == FI_ADDR_NOTAVAIL);

		/*
		 * The implicit insert path already holds util_av_implicit.lock.
		 * The explicit insert path does not, so acquire it here.
		 */
		if (!insert_implicit_av)
			EFA_GENLOCK_LOCK(&av_entry_to_release->av->util_av_implicit.lock, efa_implicit_av_lock_sym);
		else
			assert(EFA_GENLOCK_HELD(&av_entry_to_release->av->util_av_implicit.lock, efa_implicit_av_lock_sym));
		efa_rdm_av_entry_release_implicit_ah_unsafe(&av_entry_to_release->av->efa_av, av_entry_to_release);
		if (!insert_implicit_av)
			EFA_GENLOCK_UNLOCK(&av_entry_to_release->av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	}

	if (rdm_ah_to_release->implicit_refcnt == 0 &&
	    rdm_ah_to_release->explicit_refcnt == 0) {
		efa_rdm_ah_destroy_ah(domain, &rdm_ah_to_release->efa_ah);
	}

	return FI_SUCCESS;
}

/**
 * @brief find-or-create an RDM AH, managing split refcnts, the AH LRU list and
 * out-of-memory eviction retry
 *
 * Returns the existing AH (bumping its base reference count) when one already
 * exists for the GID; otherwise allocates a struct efa_rdm_ah, initializes its
 * base portion via efa_ah_base_construct and sets up the RDM-only policy: split
 * explicit/implicit reference counts, the per-domain AH LRU list and OOM
 * eviction of implicit-only AHs.
 *
 * @param[in]	domain	efa domain
 * @param[in]	gid	GID
 * @param[in]	insert_implicit_av	true when inserting for an implicit AV entry
 */
struct efa_ah *efa_rdm_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
				bool insert_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_domain *rdm_domain = (struct efa_rdm_domain *) domain;
	struct efa_ah *ah = NULL;
	struct efa_rdm_ah *rdm_ah;
	int err;

	assert(domain->info_type == EFA_INFO_RDM);

	HASH_FIND(hh, domain->ah_map, gid, EFA_GID_LEN, ah);
	if (ah) {
		rdm_ah = (struct efa_rdm_ah *)ah;
		ah->refcnt++;
		efa_rdm_ah_implicit_av_lru_ah_move(domain, ah);
		insert_implicit_av ? rdm_ah->implicit_refcnt++ :
				     rdm_ah->explicit_refcnt++;
		return ah;
	}

	rdm_ah = malloc(sizeof(struct efa_rdm_ah));
	if (!rdm_ah) {
		errno = FI_ENOMEM;
		EFA_WARN(FI_LOG_AV, "cannot allocate memory for efa_rdm_ah\n");
		return NULL;
	}

	err = efa_ah_base_construct(&rdm_ah->efa_ah, domain, gid);
	if (err && errno == FI_ENOMEM) {
		EFA_INFO(FI_LOG_AV,
			 "ibv_create_ah failed with ENOMEM for %s AV insertion. "
			 "Attempting to evict AH entry\n",
			 insert_implicit_av ? "implicit" : "explicit");
		if (efa_rdm_ah_implicit_av_evict_ah(domain, insert_implicit_av)) {
			free(rdm_ah);
			return NULL;
		}
		err = efa_ah_base_construct(&rdm_ah->efa_ah, domain, gid);
	}
	if (err) {
		free(rdm_ah);
		return NULL;
	}

	/* Newly created: initialize RDM-only state and add to LRU */
	rdm_ah->explicit_refcnt = 0;
	rdm_ah->implicit_refcnt = 0;
	dlist_init(&rdm_ah->implicit_conn_list);
	dlist_insert_tail(&rdm_ah->domain_lru_ah_list_entry,
			  &rdm_domain->ah_lru_list);

	insert_implicit_av ? rdm_ah->implicit_refcnt++ :
			     rdm_ah->explicit_refcnt++;
	return &rdm_ah->efa_ah;
}

/**
 * @brief release an RDM AH reference (split refcnt + LRU unlink + base release)
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	address handle
 * @param[in]	release_from_implicit_av	true when releasing an implicit-AV reference
 */
void efa_rdm_ah_release(struct efa_domain *domain, struct efa_ah *ah,
			bool release_from_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_ah *rdm_ah = ((struct efa_rdm_ah *)(ah));

	assert(ofi_genlock_held(&domain->util_domain.lock));
	assert((release_from_implicit_av && rdm_ah->implicit_refcnt > 0) ||
	       (!release_from_implicit_av && rdm_ah->explicit_refcnt > 0));

	release_from_implicit_av ? rdm_ah->implicit_refcnt-- :
				   rdm_ah->explicit_refcnt--;

	assert(ah->refcnt > 0);
	if (ah->refcnt == 1) {
		/* Last reference: unlink the LRU entry before the base release frees the AH. */
		assert(rdm_ah->implicit_refcnt == 0 &&
		       rdm_ah->explicit_refcnt == 0);
		dlist_remove(&rdm_ah->domain_lru_ah_list_entry);
	}

	efa_ah_release(domain, ah);
}


/*
 * Local/remote peer detection by comparing peer GID with stored local GIDs
 */
static bool efa_rdm_av_is_local_peer(struct efa_av *av, const void *addr)
{
	int i;
	uint8_t *raw_gid = ((struct efa_ep_addr *)addr)->raw;

#if ENABLE_DEBUG
	char raw_gid_str[INET6_ADDRSTRLEN] = { 0 };

	if (!inet_ntop(AF_INET6, raw_gid, raw_gid_str, INET6_ADDRSTRLEN)) {
		EFA_WARN(FI_LOG_AV, "Failed to get current EFA's GID, errno: %d\n", errno);
		return 0;
	}
	EFA_INFO(FI_LOG_AV, "The peer's GID is %s.\n", raw_gid_str);
#endif
	for (i = 0; i < g_efa_ibv_gid_cnt; ++i) {
		if (!memcmp(raw_gid, g_efa_ibv_gid_list[i].raw, EFA_GID_LEN)) {
			EFA_INFO(FI_LOG_AV, "The peer is local.\n");
			return 1;
		}
	}

	return 0;
}


/*
 * Conn lifetime.
 *
 * A conn (struct efa_rdm_av_entry) represents one remote endpoint for as long
 * as it is in the AV, implicit or explicit; see the struct comment in
 * efa_rdm_av.h and prov/efa/docs/efa_rdm_av_locking.md.
 */

static inline void efa_rdm_av_entry_set_fi_addr(struct efa_rdm_av_entry *av_entry,
						fi_addr_t fi_addr)
{
	__atomic_store_n(&av_entry->efa_av_entry.fi_addr, fi_addr, __ATOMIC_RELEASE);
}

static inline void
efa_rdm_av_entry_set_implicit_fi_addr(struct efa_rdm_av_entry *av_entry,
				      fi_addr_t fi_addr)
{
	__atomic_store_n(&av_entry->implicit_fi_addr, fi_addr, __ATOMIC_RELEASE);
}

/**
 * @brief allocate and initialize a conn that is in neither AV yet
 *
 * @param[in]	rdm_av		RDM address vector
 * @param[in]	raw_addr	raw address of the remote endpoint
 * @return	the conn, or NULL on allocation failure
 */
static struct efa_rdm_av_entry *efa_rdm_av_conn_alloc(struct efa_rdm_av *rdm_av,
						      struct efa_ep_addr *raw_addr)
	OFI_TSA_EXCLUDES(efa_av_conn_pool_lock_sym)
{
	struct efa_rdm_av_entry *av_entry;

	EFA_GENLOCK_LOCK(&rdm_av->conn_pool_lock, efa_av_conn_pool_lock_sym);
	av_entry = ofi_ibuf_alloc(rdm_av->conn_pool);
	EFA_GENLOCK_UNLOCK(&rdm_av->conn_pool_lock, efa_av_conn_pool_lock_sym);
	if (OFI_UNLIKELY(!av_entry)) {
		EFA_WARN(FI_LOG_AV, "Cannot allocate AV entry\n");
		return NULL;
	}

	/* Not reachable by any other thread until it is published. */
	memset(av_entry, 0, sizeof(*av_entry));
	memcpy(av_entry->efa_av_entry.ep_addr, raw_addr, EFA_EP_ADDR_LEN);
	av_entry->efa_av_entry.fi_addr = FI_ADDR_NOTAVAIL;
	av_entry->av = rdm_av;
	av_entry->implicit_fi_addr = FI_ADDR_NOTAVAIL;
	av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	av_entry->peer_idx = (uint32_t) ofi_buf_index(av_entry);
	av_entry->released = false;
	return av_entry;
}

/**
 * @brief return a conn to the pool
 *
 * The conn must already be unpublished from every map and have no peers. Its
 * peer_idx can be handed out again from here on, which is safe because every
 * endpoint's slot for it was cleared by efa_rdm_av_entry_destroy_peers.
 */
static void efa_rdm_av_conn_free(struct efa_rdm_av *rdm_av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_EXCLUDES(efa_av_conn_pool_lock_sym)
{
	memset(av_entry->efa_av_entry.ep_addr, 0, EFA_EP_ADDR_LEN);
	EFA_GENLOCK_LOCK(&rdm_av->conn_pool_lock, efa_av_conn_pool_lock_sym);
	ofi_ibuf_free(av_entry);
	EFA_GENLOCK_UNLOCK(&rdm_av->conn_pool_lock, efa_av_conn_pool_lock_sym);
}

/**
 * @brief destroy every bound endpoint's peer for a conn that is leaving the AV
 *
 * The caller has already unpublished the conn from every map, so no new lookup
 * can find it. Marking the conn released before visiting the endpoints makes a
 * creator that read the conn before it was unpublished skip creating a peer once
 * it gets the endpoint's ctrl_lock (see efa_rdm_ep_create_peer).
 *
 * @param[in]	av		address vector
 * @param[in]	av_entry	conn
 */
static void efa_rdm_av_entry_destroy_peers(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_EXCLUDES(efa_av_ep_list_lock_sym, efa_ctrl_lock_sym)
{
	struct dlist_entry *entry;
	struct efa_rdm_ep *ep;

	__atomic_store_n(&av_entry->released, true, __ATOMIC_RELEASE);

	EFA_GENLOCK_LOCK(&av->util_av.ep_list_lock, efa_av_ep_list_lock_sym);
	dlist_foreach(&av->util_av.ep_list, entry) {
		ep = container_of(entry, struct efa_rdm_ep,
				  base_ep.util_ep.av_entry);
		efa_rdm_ep_destroy_peer(ep, av_entry);
	}
	EFA_GENLOCK_UNLOCK(&av->util_av.ep_list_lock, efa_av_ep_list_lock_sym);
}


/**
 * @brief Add the entry to the implicit AV LRU list; if the list is full, evict
 * the least recently used entry at the front and add the latest one.
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
static inline int efa_rdm_av_implicit_av_lru_insert(struct efa_av *av,
						    struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	size_t cur_size;
	struct efa_ep_addr_hashable *ep_addr_hashable;
	struct efa_rdm_av_entry *av_entry_to_release;

	/* Implicit AV size of 0 means we allow the implicit AV to grow without
	 * bound */
	if (rdm_av->implicit_av_size == 0)
		goto out;

	cur_size = HASH_CNT(hh, rdm_av->util_av_implicit.hash);
	if (cur_size <= rdm_av->implicit_av_size)
		goto out;

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(!dlist_empty(&rdm_av->implicit_av_lru_list));

	av_entry_to_release = container_of(rdm_av->implicit_av_lru_list.next,
					   struct efa_rdm_av_entry,
					   implicit_av_lru_entry);
	EFA_INFO(FI_LOG_AV,
		 "Evicting AV entry for peer implicit fi_addr %" PRIu64
		 " AHN %" PRIu16 " QPN %" PRIu16 " QKEY %" PRIu32 " from "
		 "implicit AV\n",
		 efa_rdm_av_entry_implicit_fi_addr(av_entry_to_release),
		 av_entry_to_release->efa_av_entry.ah->ahn,
		 efa_av_entry_ep_addr(&av_entry_to_release->efa_av_entry)->qpn,
		 efa_av_entry_ep_addr(&av_entry_to_release->efa_av_entry)->qkey);

	/* Remember the evicted peer, so that its packets are dropped. */
	ep_addr_hashable = malloc(sizeof(struct efa_ep_addr_hashable));
	if (!ep_addr_hashable) {
		EFA_WARN(FI_LOG_AV, "Could not allocate memory for LRU AV entry hashset entry\n");
		return -FI_ENOMEM;
	}
	memcpy(ep_addr_hashable,
	       efa_av_entry_ep_addr(&av_entry_to_release->efa_av_entry),
	       sizeof(struct efa_ep_addr));
	HASH_ADD(hh, rdm_av->evicted_peers_hashset, addr, sizeof(struct efa_ep_addr), ep_addr_hashable);

	efa_rdm_av_entry_release_implicit(av, av_entry_to_release);

	assert(HASH_CNT(hh, rdm_av->util_av_implicit.hash) == rdm_av->implicit_av_size);

out:
	dlist_insert_tail(&av_entry->implicit_av_lru_entry,
			  &rdm_av->implicit_av_lru_list);
	return FI_SUCCESS;
}


/**
 * @brief Insert the address into SHM provider's AV for RDM endpoints
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
static int efa_rdm_av_entry_insert_shm_av(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_ep_addr *ep_addr = efa_av_entry_ep_addr(&av_entry->efa_av_entry);
	int err, ret;
	char smr_name[EFA_SHM_NAME_MAX];
	size_t smr_name_len;

	assert(av->domain->info_type == EFA_INFO_RDM);

	if (efa_rdm_av_is_local_peer(av, ep_addr) && rdm_av->shm_rdm_av) {
		if (rdm_av->shm_used >= efa_env.shm_av_size) {
			EFA_WARN(FI_LOG_AV,
				 "Max number of shm AV entry (%d) has been reached.\n",
				 efa_env.shm_av_size);
			return -FI_ENOMEM;
		}

		smr_name_len = EFA_SHM_NAME_MAX;
		err = efa_shm_ep_name_construct(smr_name, &smr_name_len, ep_addr);
		if (err != FI_SUCCESS) {
			EFA_WARN(FI_LOG_AV,
				 "efa_rdm_ep_efa_addr_to_str() failed! err=%d\n", err);
			return err;
		}

		av_entry->shm_fi_addr = efa_rdm_av_entry_fi_addr(av_entry);
		ret = fi_av_insert(rdm_av->shm_rdm_av, smr_name, 1, &av_entry->shm_fi_addr, FI_AV_USER_ID, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			EFA_WARN(FI_LOG_AV,
				 "Failed to insert address to shm provider's av: %s\n",
				 fi_strerror(-ret));
			av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
			return ret;
		}

		EFA_INFO(FI_LOG_AV,
			"Successfully inserted %s to shm provider's av. efa_fiaddr: %ld shm_fiaddr = %ld\n",
			smr_name, efa_rdm_av_entry_fi_addr(av_entry), av_entry->shm_fi_addr);

		assert(av_entry->shm_fi_addr < efa_env.shm_av_size);
		rdm_av->shm_used++;
	}

	return 0;
}


/**
 * @brief remove a conn from the SHM AV, if it was inserted there
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
static void efa_rdm_av_entry_remove_shm_av(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	int err;

	if (av_entry->shm_fi_addr == FI_ADDR_NOTAVAIL || !rdm_av->shm_rdm_av)
		return;

	err = fi_av_remove(rdm_av->shm_rdm_av, &av_entry->shm_fi_addr, 1, 0);
	if (err) {
		EFA_WARN(FI_LOG_AV, "remove address from shm av failed! err=%d\n",
			 err);
	} else {
		rdm_av->shm_used--;
		assert(av_entry->shm_fi_addr < efa_env.shm_av_size);
	}
}


/*
 * @brief Reserve everything efa_rdm_av_reverse_av_add_reserved() will need
 *
 * Allocates the cur_reverse_av slot backing and, when that slot already holds a
 * previous entry for the same (ahn, qpn), the prv_reverse_av entry that previous
 * entry gets demoted into. The reverse AV is left untouched, so a caller that
 * fails here has nothing to unwind.
 *
 * Every writer of a reverse AV holds the lock that guards it, so the slot
 * cannot become occupied (or be vacated) between the reservation and the
 * matching add as long as the caller holds that lock throughout.
 *
 * @param[in]		cur_reverse_av	Reverse AV keyed by efa_av_reverse_av_key()
 * @param[in]		entry		efa_av_entry object to be added later
 * @param[out]		prv_entry	prv_reverse_av entry to hand to the add,
 *					NULL when the slot is free
 * @return		On success, return 0.
 *			Otherwise, return a negative libfabric error code
 */
static int efa_rdm_av_reverse_av_reserve(struct efa_av_array *cur_reverse_av,
					 struct efa_av_entry *entry,
					 struct efa_prv_reverse_av **prv_entry)
{
	uint64_t key = efa_av_entry_reverse_av_key(entry);
	int err;

	*prv_entry = NULL;

	err = efa_av_array_reserve(cur_reverse_av, key);
	if (err) {
		EFA_WARN(FI_LOG_AV,
			 "Cannot reserve cur_reverse_av slot for key %" PRIu64
			 ": %s\n", key, fi_strerror(-err));
		return err;
	}

	if (!efa_av_array_at(cur_reverse_av, key))
		return 0;

	*prv_entry = malloc(sizeof(**prv_entry));
	if (!*prv_entry) {
		EFA_WARN(FI_LOG_AV, "Cannot allocate memory for prv_reverse_av entry\n");
		return -FI_ENOMEM;
	}

	return 0;
}

/*
 * @brief Demote the entry currently holding an (ahn, qpn) slot into prv_reverse_av
 *
 * A (ahn, qpn) collision means a QP number was reused. Only the RDM protocol
 * disambiguates reused QPNs via the connid-keyed prv_reverse_av, so the previous
 * connection is preserved there before the base cur slot is overwritten. The
 * efa-direct reverse lookup reads cur_reverse_av only and uses the base variant.
 *
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		prv_entry	caller-allocated prv_reverse_av entry
 * @param[in]		entry		efa_av_entry object taking over the slot
 * @param[in]		cur_entry	efa_av_entry object holding the slot today
 */
static void efa_rdm_av_prv_reverse_av_add(struct efa_prv_reverse_av **prv_reverse_av,
					  struct efa_prv_reverse_av *prv_entry,
					  struct efa_av_entry *entry,
					  struct efa_av_entry *cur_entry)
{
	memset(&prv_entry->key, 0, sizeof(prv_entry->key));
	prv_entry->key.ahn = entry->ah->ahn;
	prv_entry->key.qpn = efa_av_entry_ep_addr(entry)->qpn;
	prv_entry->key.connid = efa_av_entry_ep_addr(cur_entry)->qkey;
	prv_entry->entry = cur_entry;
	HASH_ADD(hh, *prv_reverse_av, key, sizeof(prv_entry->key), prv_entry);
}

/*
 * @brief RDM reverse-AV add using a reservation, which cannot fail
 *
 * For callers that have already taken an irreversible step and so cannot report
 * a failure here. Callers that can unwind may use efa_rdm_av_reverse_av_add.
 *
 * @param[in,out]	cur_reverse_av	Reverse AV keyed by efa_av_reverse_av_key()
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		entry		efa_av_entry object
 * @param[in]		prv_entry	reservation from
 *					efa_rdm_av_reverse_av_reserve()
 */
static void efa_rdm_av_reverse_av_add_reserved(struct efa_av_array *cur_reverse_av,
					       struct efa_prv_reverse_av **prv_reverse_av,
					       struct efa_av_entry *entry,
					       struct efa_prv_reverse_av *prv_entry)
{
	struct efa_av_entry *cur_entry;
	int err;

	cur_entry = efa_av_array_at(cur_reverse_av,
				    efa_av_entry_reverse_av_key(entry));
	/* The reservation was taken for this slot under the same lock */
	assert(!cur_entry == !prv_entry);

	if (cur_entry)
		efa_rdm_av_prv_reverse_av_add(prv_reverse_av, prv_entry, entry,
					      cur_entry);

	/* The slot was reserved, so this cannot fail */
	err = efa_av_reverse_av_add(cur_reverse_av, entry);
	assert(!err);
	(void) err;
}

/*
 * @brief RDM reverse-AV add: base cur add/replace plus connid-keyed prv preserve
 *
 * Allocates as it goes and reports a failure to the caller, which must be able
 * to unwind whatever it has already done.
 *
 * @param[in,out]	cur_reverse_av	Reverse AV keyed by efa_av_reverse_av_key()
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		entry		efa_av_entry object
 * @return		On success, return 0.
 * 			Otherwise, return a negative libfabric error code
 */
int efa_rdm_av_reverse_av_add(struct efa_av_array *cur_reverse_av,
			      struct efa_prv_reverse_av **prv_reverse_av,
			      struct efa_av_entry *entry)
{
	struct efa_prv_reverse_av *prv_entry;
	int err;

	err = efa_rdm_av_reverse_av_reserve(cur_reverse_av, entry, &prv_entry);
	if (err)
		return err;

	efa_rdm_av_reverse_av_add_reserved(cur_reverse_av, prv_reverse_av,
					   entry, prv_entry);
	return 0;
}


/*
 * @brief RDM reverse-AV remove: base cur remove, else drop from prv_reverse_av
 *
 * If the entry is no longer the current one for its (ahn, qpn) it was demoted
 * into the connid-keyed prv_reverse_av; remove it from there. efa-direct never
 * populates prv_reverse_av and uses the base variant.
 *
 * @param[in,out]	cur_reverse_av	Reverse AV keyed by efa_av_reverse_av_key()
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		entry		efa_av_entry object
 */
void efa_rdm_av_reverse_av_remove(struct efa_av_array *cur_reverse_av,
				  struct efa_prv_reverse_av **prv_reverse_av,
				  struct efa_av_entry *entry)
{
	struct efa_prv_reverse_av *prv_reverse_av_entry;
	struct efa_prv_reverse_av_key prv_key;

	if (efa_av_reverse_av_remove(cur_reverse_av, entry))
		return;

	memset(&prv_key, 0, sizeof(prv_key));
	prv_key.ahn = entry->ah->ahn;
	prv_key.qpn = efa_av_entry_ep_addr(entry)->qpn;
	prv_key.connid = efa_av_entry_ep_addr(entry)->qkey;
	HASH_FIND(hh, *prv_reverse_av, &prv_key, sizeof(prv_key),
		  prv_reverse_av_entry);
	assert(prv_reverse_av_entry &&
	       prv_reverse_av_entry->entry == entry);
	if (!prv_reverse_av_entry)
		return;
	HASH_DEL(*prv_reverse_av, prv_reverse_av_entry);
	free(prv_reverse_av_entry);
}


/**
 * @brief allocate a conn for a new address and insert it into the explicit AV
 *
 * Everything that can fail -- the util AV slot, the conn, the AH, the map slot
 * reservations and the SHM insertion -- happens before the conn is published,
 * so a failure has nothing visible to unwind. The two publications at the end
 * may happen in either order: the conn is new, no endpoint has a peer for its
 * peer_idx, and every reader creates peers only through efa_rdm_ep_get_peer.
 *
 * @param[in]	av	efa address vector
 * @param[in]	raw_addr	raw endpoint address being inserted
 * @param[in]	flags	flags passed to fi_av_insert
 * @param[in]	context	user context associated with the address
 */
struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_explicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_prv_reverse_av *prv_entry = NULL;
	struct efa_rdm_av_entry *av_entry;
	fi_addr_t fi_addr;
	int err;

	assert(EFA_GENLOCK_HELD(&av->util_av.lock, efa_util_av_lock_sym));
	assert(av->type == FI_AV_TABLE);

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	err = ofi_av_insert_addr(&av->util_av, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	av_entry = efa_rdm_av_conn_alloc(rdm_av, raw_addr);
	if (!av_entry)
		goto err_remove_addr;

	av_entry->efa_av_entry.ah = efa_rdm_ah_alloc(av->domain, raw_addr->raw, false);
	if (!av_entry->efa_av_entry.ah)
		goto err_free_conn;

	/* Set before the conn is reachable; SHM insertion reads it too. */
	efa_rdm_av_entry_set_fi_addr(av_entry, fi_addr);

	err = efa_av_array_reserve(av->addr_to_entry_map, fi_addr);
	if (err)
		goto err_release_ah;

	err = efa_rdm_av_reverse_av_reserve(av->cur_reverse_av,
					    &av_entry->efa_av_entry, &prv_entry);
	if (err)
		goto err_release_ah;

	/*
	 * The explicit AV insertion is triggered by the application calling the
	 * fi_av_insert API. Attempt shm av insertion; efa_rdm_av_entry_insert_shm_av is
	 * a no-op for peers that are not local.
	 */
	err = efa_rdm_av_entry_insert_shm_av(av, av_entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert fi_addr %" PRIu64
			" into shm provider's AV: %s\n", fi_addr, fi_strerror(-err));
		goto err_free_prv;
	}

	/* Publish. Nothing below can fail. */
	efa_rdm_av_reverse_av_add_reserved(av->cur_reverse_av,
					   &rdm_av->prv_reverse_av,
					   &av_entry->efa_av_entry, prv_entry);
	err = efa_av_array_insert(av->addr_to_entry_map, fi_addr,
				  &av_entry->efa_av_entry);
	assert(!err);

	return av_entry;

err_free_prv:
	free(prv_entry);
err_release_ah:
	efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, false);
err_free_conn:
	efa_rdm_av_conn_free(rdm_av, av_entry);
err_remove_addr:
	err = ofi_av_remove_addr(&av->util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));
	return NULL;
}


/**
 * @brief allocate a conn for a new address and insert it into the implicit AV
 *
 * The caller must hold util_av.lock as well, and must already have established
 * under it that the address is in neither AV (efa_rdm_av_insert_one_implicit
 * enforces this contract).
 *
 * @param[in]	av	efa address vector
 * @param[in]	raw_addr	raw endpoint address being inserted
 * @param[in]	flags	flags passed to fi_av_insert
 * @param[in]	context	user context associated with the address
 */
struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_implicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct util_av *util_av_implicit = &rdm_av->util_av_implicit;
	struct efa_prv_reverse_av *prv_entry = NULL;
	struct efa_rdm_av_entry *av_entry;
	struct efa_rdm_ah *rdm_ah;
	fi_addr_t fi_addr;
	int err;

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(av->domain->info_type == EFA_INFO_RDM);
	assert(av->type == FI_AV_TABLE);

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	err = ofi_av_insert_addr(util_av_implicit, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	av_entry = efa_rdm_av_conn_alloc(rdm_av, raw_addr);
	if (!av_entry)
		goto err_remove_addr;

	efa_rdm_av_entry_set_implicit_fi_addr(av_entry, fi_addr);
	dlist_init(&av_entry->implicit_av_lru_entry);

	av_entry->efa_av_entry.ah = efa_rdm_ah_alloc(av->domain, raw_addr->raw, true);
	if (!av_entry->efa_av_entry.ah)
		goto err_free_conn;

	rdm_ah = (struct efa_rdm_ah *) av_entry->efa_av_entry.ah;
	dlist_insert_tail(&av_entry->ah_implicit_conn_list_entry,
			  &rdm_ah->implicit_conn_list);

	/*
	 * The LRU insertion can evict another implicit conn, which can vacate
	 * the (ahn, qpn) slot this conn is about to take. Do it before reserving
	 * that slot so the reservation sees the final state.
	 */
	err = efa_rdm_av_implicit_av_lru_insert(av, av_entry);
	if (err)
		goto err_release_ah;

	err = efa_av_array_reserve(rdm_av->addr_to_entry_map_implicit, fi_addr);
	if (err)
		goto err_lru_remove;

	err = efa_rdm_av_reverse_av_reserve(rdm_av->cur_reverse_av_implicit,
					    &av_entry->efa_av_entry, &prv_entry);
	if (err)
		goto err_lru_remove;

	/* Publish. Nothing below can fail. */
	efa_rdm_av_reverse_av_add_reserved(rdm_av->cur_reverse_av_implicit,
					   &rdm_av->prv_reverse_av_implicit,
					   &av_entry->efa_av_entry, prv_entry);
	err = efa_av_array_insert(rdm_av->addr_to_entry_map_implicit, fi_addr,
				  &av_entry->efa_av_entry);
	assert(!err);

	return av_entry;

err_lru_remove:
	dlist_remove(&av_entry->implicit_av_lru_entry);
err_release_ah:
	dlist_remove(&av_entry->ah_implicit_conn_list_entry);
	efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
err_free_conn:
	efa_rdm_av_conn_free(rdm_av, av_entry);
err_remove_addr:
	err = ofi_av_remove_addr(util_av_implicit, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for implicit fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));

	return NULL;
}


/**
 * @brief remove a conn from the explicit AV and free it
 *
 * Unpublish first, so no new lookup can reach the conn, then destroy every
 * endpoint's peer for it, then release its resources.
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
void efa_rdm_av_entry_release_explicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	fi_addr_t fi_addr = efa_rdm_av_entry_fi_addr(av_entry);
	int err;

	assert(EFA_GENLOCK_HELD(&av->util_av.lock, efa_util_av_lock_sym));
	assert(fi_addr != FI_ADDR_NOTAVAIL);
	assert(efa_rdm_av_entry_implicit_fi_addr(av_entry) == FI_ADDR_NOTAVAIL);

	/* The slot is occupied, so clearing it allocates nothing. */
	err = efa_av_array_insert(av->addr_to_entry_map, fi_addr, NULL);
	assert(!err);
	efa_rdm_av_reverse_av_remove(av->cur_reverse_av, &rdm_av->prv_reverse_av,
				     &av_entry->efa_av_entry);

	efa_rdm_av_entry_destroy_peers(av, av_entry);
	efa_rdm_av_entry_remove_shm_av(av, av_entry);
	efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, false);

	err = ofi_av_remove_addr(&av->util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "ofi_av_remove_addr failed for fi_addr %" PRIu64
			 ": %s\n", fi_addr, fi_strerror(-err));

	EFA_INFO(FI_LOG_AV, "Released explicit AV entry fi_addr %" PRIu64 "\n", fi_addr);
	efa_rdm_av_conn_free(rdm_av, av_entry);
}


/**
 * @brief remove a conn from the implicit AV and free it
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 * @param[in]	release_ah	release the AH reference; false when the caller
 *				(AH eviction) drops the reference itself
 */
static void efa_rdm_av_entry_release_implicit_common(struct efa_av *av,
						     struct efa_rdm_av_entry *av_entry,
						     bool release_ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	fi_addr_t fi_addr = efa_rdm_av_entry_implicit_fi_addr(av_entry);
	struct efa_rdm_ah *rdm_ah;
	int err;

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(fi_addr != FI_ADDR_NOTAVAIL);
	assert(efa_rdm_av_entry_fi_addr(av_entry) == FI_ADDR_NOTAVAIL);

	err = efa_av_array_insert(rdm_av->addr_to_entry_map_implicit, fi_addr, NULL);
	assert(!err);
	efa_rdm_av_reverse_av_remove(rdm_av->cur_reverse_av_implicit,
				     &rdm_av->prv_reverse_av_implicit,
				     &av_entry->efa_av_entry);
	dlist_remove(&av_entry->implicit_av_lru_entry);

	efa_rdm_av_entry_destroy_peers(av, av_entry);

	dlist_remove(&av_entry->ah_implicit_conn_list_entry);
	if (release_ah) {
		efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
	} else {
		rdm_ah = (struct efa_rdm_ah *) av_entry->efa_av_entry.ah;
		rdm_ah->implicit_refcnt--;
		/* Mirror the base reference drop that efa_rdm_ah_release would
		 * do; the caller (eviction) destroys the AH once its refcnt
		 * reaches zero. */
		av_entry->efa_av_entry.ah->refcnt--;
	}

	err = ofi_av_remove_addr(&rdm_av->util_av_implicit, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "ofi_av_remove_addr failed for implicit fi_addr %" PRIu64
			 ": %s\n", fi_addr, fi_strerror(-err));

	efa_rdm_av_entry_set_implicit_fi_addr(av_entry, FI_ADDR_NOTAVAIL);
	efa_rdm_av_conn_free(rdm_av, av_entry);
}

/**
 * @brief release an efa_rdm_av_entry from the implicit AV
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
void efa_rdm_av_entry_release_implicit(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	efa_rdm_av_entry_release_implicit_common(av, av_entry, true);
}


/**
 * @brief release an implicit efa_rdm_av_entry during AH eviction
 *
 * Like efa_rdm_av_entry_release_implicit but leaves the AH to the caller.
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
void efa_rdm_av_entry_release_implicit_ah_unsafe(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	efa_rdm_av_entry_release_implicit_common(av, av_entry, false);
}


/**
 * @brief find the efa_rdm_av_entry using fi_addr in the implicit AV
 */
struct efa_rdm_av_entry *efa_rdm_av_addr_to_entry_implicit(struct efa_av *av,
							   fi_addr_t fi_addr)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;

	entry = efa_av_addr_to_entry_impl(rdm_av->addr_to_entry_map_implicit, fi_addr);
	return entry ? container_of(entry, struct efa_rdm_av_entry, efa_av_entry) : NULL;
}


/**
 * @brief lock-free reverse lookup in a current reverse AV
 *
 * Reads the entry currently registered for (ahn, qpn) and validates it against
 * the packet's connection ID. cur_reverse_av is an efa_av_array, so this needs
 * no lock.
 *
 * @param[in]	cur_reverse_av	reverse AV indexed by efa_av_reverse_av_key()
 * @param[in]	ahn		address handle number
 * @param[in]	qpn		QP number
 * @param[in]	pkt_entry	NULL or rdm packet entry, used to extract connid
 * @param[out]	check_prv	set when the caller must look in prv_reverse_av
 * @param[out]	prv_connid	the connid to look up there
 * @return	the matching efa_av_entry, or NULL. NULL with *check_prv false
 *		means no peer is known for (ahn, qpn) at all.
 */
static inline struct efa_av_entry *
efa_rdm_av_reverse_lookup_cur(struct efa_av_array *cur_reverse_av, uint16_t ahn,
			      uint16_t qpn, struct efa_rdm_pke *pkt_entry,
			      bool *check_prv, uint32_t *prv_connid)
{
	uint32_t *connid;
	struct efa_av_entry *cur_entry;

	*check_prv = false;

	cur_entry = efa_av_array_at(cur_reverse_av,
				    efa_av_reverse_av_key(ahn, qpn));
	if (OFI_UNLIKELY(!cur_entry))
		return NULL;

	if (!pkt_entry) {
		/**
		 * There is no packet entry to extract connid from when we get
		 * an IBV_WC_RECV_RDMA_WITH_IMM completion from rdma-core. Or
		 * the pkt_entry is allocated from a buffer user posted that
		 * doesn't expect any pkt hdr.
		 */
		return cur_entry;
	}

	connid = efa_rdm_pke_connid_ptr(pkt_entry);
	if (!connid) {
		EFA_WARN_ONCE(FI_LOG_EP_CTRL,
			      "An incoming packet does NOT have connection ID "
			      "in its header.\n"
			      "This means the peer is using an older version "
			      "of libfabric.\n"
			      "The communication can continue but it is "
			      "encouraged to use\n"
			      "a newer version of libfabric\n");
		return cur_entry;
	}

	if (OFI_LIKELY(*connid == efa_av_entry_ep_addr(cur_entry)->qkey))
		return cur_entry;

	*check_prv = true;
	*prv_connid = *connid;
	return NULL;
}


/**
 * @brief reverse lookup of a previous connection on a reused (ahn, qpn)
 *
 * Slow path taken only when a QP number was reused: prv_reverse_av is a hash map
 * and the caller must hold the lock that guards it.
 */
static struct efa_av_entry *
efa_rdm_av_reverse_lookup_prv(struct efa_prv_reverse_av **prv_reverse_av,
			      uint16_t ahn, uint16_t qpn, uint32_t connid)
{
	struct efa_prv_reverse_av *prv_entry;
	struct efa_prv_reverse_av_key prv_key;

	memset(&prv_key, 0, sizeof(prv_key));
	prv_key.ahn = ahn;
	prv_key.qpn = qpn;
	prv_key.connid = connid;
	HASH_FIND(hh, *prv_reverse_av, &prv_key, sizeof(prv_key), prv_entry);

	return OFI_LIKELY(!!prv_entry) ? prv_entry->entry : NULL;
}

static inline struct efa_rdm_av_entry *efa_rdm_av_entry_of(struct efa_av_entry *entry)
{
	return entry ? container_of(entry, struct efa_rdm_av_entry, efa_av_entry) : NULL;
}


/**
 * @brief lock-free connid-aware reverse lookup in the explicit AV
 *
 * Only the current connection on (ahn, qpn) is served here; a packet from a
 * previous connection on a reused QP number returns NULL and must be resolved
 * with efa_rdm_av_reverse_lookup_entry_unsafe under util_av.lock.
 *
 * @return	the conn, or NULL
 */
struct efa_rdm_av_entry *efa_rdm_av_reverse_lookup_entry(struct efa_av *av,
							 uint16_t ahn, uint16_t qpn,
							 struct efa_rdm_pke *pkt_entry)
{
	uint32_t prv_connid = 0;
	bool check_prv;

	return efa_rdm_av_entry_of(efa_rdm_av_reverse_lookup_cur(
		av->cur_reverse_av, ahn, qpn, pkt_entry, &check_prv, &prv_connid));
}


/**
 * @brief connid-aware reverse lookup in the explicit AV, current and previous
 * connections
 *
 * @return	the conn, or NULL
 */
struct efa_rdm_av_entry *efa_rdm_av_reverse_lookup_entry_unsafe(struct efa_av *av,
								uint16_t ahn, uint16_t qpn,
								struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;
	uint32_t prv_connid = 0;
	bool check_prv;

	entry = efa_rdm_av_reverse_lookup_cur(av->cur_reverse_av, ahn, qpn,
					      pkt_entry, &check_prv,
					      &prv_connid);
	if (!entry && check_prv)
		entry = efa_rdm_av_reverse_lookup_prv(&rdm_av->prv_reverse_av,
						      ahn, qpn, prv_connid);
	return efa_rdm_av_entry_of(entry);
}


/**
 * @brief connid-aware reverse lookup in the implicit AV, current and previous
 * connections
 *
 * @return	the conn, or NULL
 */
struct efa_rdm_av_entry *
efa_rdm_av_reverse_lookup_entry_implicit_unsafe(struct efa_av *av,
						uint16_t ahn, uint16_t qpn,
						struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;
	uint32_t prv_connid = 0;
	bool check_prv;

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));

	entry = efa_rdm_av_reverse_lookup_cur(rdm_av->cur_reverse_av_implicit,
					      ahn, qpn, pkt_entry, &check_prv,
					      &prv_connid);
	if (!entry && check_prv)
		entry = efa_rdm_av_reverse_lookup_prv(
			&rdm_av->prv_reverse_av_implicit, ahn, qpn, prv_connid);
	return efa_rdm_av_entry_of(entry);
}


/**
 * @brief raw address -> conn in the explicit AV
 */
struct efa_rdm_av_entry *efa_rdm_av_addr_lookup_entry_unsafe(struct efa_av *av,
							     struct efa_ep_addr *addr)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
{
	fi_addr_t fi_addr;

	fi_addr = ofi_av_lookup_fi_addr_unsafe(&av->util_av, addr);
	return fi_addr == FI_ADDR_NOTAVAIL ? NULL : efa_rdm_av_addr_to_entry(av, fi_addr);
}


/**
 * @brief raw address -> conn in the implicit AV
 */
struct efa_rdm_av_entry *
efa_rdm_av_addr_lookup_entry_implicit_unsafe(struct efa_av *av,
					     struct efa_ep_addr *addr)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	fi_addr_t fi_addr;

	fi_addr = ofi_av_lookup_fi_addr_unsafe(&rdm_av->util_av_implicit, addr);
	return fi_addr == FI_ADDR_NOTAVAIL ? NULL : efa_rdm_av_addr_to_entry_implicit(av, fi_addr);
}


/**
 * @brief find fi_addr for rdm endpoint in the explicit AV (connid aware)
 *
 * The common case -- the packet comes from the current connection on its
 * (ahn, qpn) -- is served lock free out of cur_reverse_av. Only a reused QP
 * number falls through to the lock-protected prv_reverse_av hash map.
 *
 * @return	On success, return fi_addr to the peer who sent the packet.
 * 		If no such peer exists, return FI_ADDR_NOTAVAIL
 */
fi_addr_t efa_rdm_av_reverse_lookup(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;
	uint32_t prv_connid = 0;
	bool check_prv;
	fi_addr_t fi_addr;

	entry = efa_rdm_av_reverse_lookup_cur(av->cur_reverse_av, ahn, qpn,
					      pkt_entry, &check_prv,
					      &prv_connid);
	if (OFI_LIKELY(!!entry))
		return efa_rdm_av_entry_fi_addr(efa_rdm_av_entry_of(entry));

	if (!check_prv)
		return FI_ADDR_NOTAVAIL;

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	entry = efa_rdm_av_reverse_lookup_prv(&rdm_av->prv_reverse_av, ahn, qpn,
					      prv_connid);
	fi_addr = (OFI_LIKELY(!!entry)) ?
		  efa_rdm_av_entry_fi_addr(efa_rdm_av_entry_of(entry)) :
		  FI_ADDR_NOTAVAIL;
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	return fi_addr;
}

/**
 * @brief Same as efa_rdm_av_reverse_lookup but does not take the util_av
 * lock. The caller is expected to hold the util_av lock.
 */
fi_addr_t efa_rdm_av_reverse_lookup_unsafe(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
{
	struct efa_rdm_av_entry *av_entry;

	av_entry = efa_rdm_av_reverse_lookup_entry_unsafe(av, ahn, qpn, pkt_entry);
	return av_entry ? efa_rdm_av_entry_fi_addr(av_entry) : FI_ADDR_NOTAVAIL;
}

/**
 * @brief find the implicit fi_addr for rdm endpoint in the implicit AV
 * (connid aware). The caller holds util_av_implicit.lock.
 */
fi_addr_t efa_rdm_av_reverse_lookup_implicit_unsafe(struct efa_av *av,
						    uint16_t ahn, uint16_t qpn,
						    struct efa_rdm_pke *pkt_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
{
	struct efa_rdm_av_entry *av_entry;

	av_entry = efa_rdm_av_reverse_lookup_entry_implicit_unsafe(av, ahn, qpn,
								   pkt_entry);
	return av_entry ? efa_rdm_av_entry_implicit_fi_addr(av_entry) : FI_ADDR_NOTAVAIL;
}


/**
 * @brief Move the entry to the end of the implicit AV LRU list and bump its AH
 *
 * Moving the entry to the tail marks it as the most recently used implicit AV
 * entry.
 *
 * @param[in]	av	efa address vector
 * @param[in]	av_entry	efa_rdm_av_entry
 */
void efa_rdm_av_implicit_av_lru_move(struct efa_av *av,
				     struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(rdm_av->implicit_av_size == 0 ||
	       HASH_CNT(hh, rdm_av->util_av_implicit.hash) <= rdm_av->implicit_av_size);
	assert(dlist_entry_in_list(&rdm_av->implicit_av_lru_list,
				   &av_entry->implicit_av_lru_entry));

	dlist_remove(&av_entry->implicit_av_lru_entry);
	dlist_insert_tail(&av_entry->implicit_av_lru_entry,
			  &rdm_av->implicit_av_lru_list);

	assert(ofi_genlock_held(&av->domain->util_domain.lock));
	efa_rdm_ah_implicit_av_lru_ah_move(av->domain, av_entry->efa_av_entry.ah);
}


static fi_addr_t
efa_rdm_av_get_addr_from_peer_rx_entry(struct fi_peer_rx_entry *rx_entry)
{
	struct efa_rdm_pke *pke;

	pke = (struct efa_rdm_pke *) rx_entry->peer_context;

	return efa_rdm_av_entry_fi_addr(pke->peer->av_entry);
}


/**
 * @brief promote a conn from the implicit to the explicit AV
 *
 * The conn object, its peer_idx and therefore every endpoint's peer for it stay
 * exactly where they are; only the maps that point at the conn change. No peer
 * map is read or written here, so there is no ordering between this function
 * and a concurrent lock-free reader to get right: any reader that finds the
 * conn, through either AV, lands on the same peer_map slot.
 *
 * Everything that can fail is done (reserved) before the first visible change,
 * so the function either leaves the AV untouched or completes.
 *
 * @param[in]	av			address vector
 * @param[in]	raw_addr		address being inserted
 * @param[in]	implicit_fi_addr	the conn's implicit fi_addr
 * @param[out]	fi_addr			the conn's new explicit fi_addr
 * @return	0 on success, or a negative libfabric error code
 */
static int efa_rdm_av_entry_implicit_to_explicit(struct efa_av *av,
					   struct efa_ep_addr *raw_addr,
					   fi_addr_t implicit_fi_addr,
					   fi_addr_t *fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym,
			 efa_implicit_av_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_prv_reverse_av *prv_entry = NULL;
	struct efa_rdm_av_entry *av_entry;
	struct efa_rdm_ah *rdm_ah;
	struct fid_peer_srx *peer_srx;
	struct dlist_entry *entry;
	struct efa_rdm_ep *ep;
	fi_addr_t new_fi_addr;
	int err, cleanup_err;

	EFA_INFO(FI_LOG_AV,
		 "Moving peer with implicit fi_addr %" PRIu64
		 " to explicit AV\n",
		 implicit_fi_addr);

	assert(EFA_GENLOCK_HELD(&av->util_av.lock, efa_util_av_lock_sym));
	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(av->type == FI_AV_TABLE);

	av_entry = efa_rdm_av_addr_to_entry_implicit(av, implicit_fi_addr);
	assert(av_entry);
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(&av_entry->efa_av_entry)));
	assert(efa_rdm_av_entry_fi_addr(av_entry) == FI_ADDR_NOTAVAIL &&
	       efa_rdm_av_entry_implicit_fi_addr(av_entry) == implicit_fi_addr);

	/* Reserve. */
	err = ofi_av_insert_addr(&av->util_av, raw_addr, &new_fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV,
			 "Failed to insert implicit fi_addr %" PRIu64 " into explicit util AV: %s\n",
			 implicit_fi_addr, fi_strerror(-err));
		return err;
	}

	err = efa_av_array_reserve(av->addr_to_entry_map, new_fi_addr);
	if (err)
		goto err_remove_addr;

	err = efa_rdm_av_reverse_av_reserve(av->cur_reverse_av,
					    &av_entry->efa_av_entry, &prv_entry);
	if (err)
		goto err_remove_addr;

	/*
	 * Commit. Nothing below can fail.
	 *
	 * The explicit fi_addr is set before the conn becomes reachable through
	 * the explicit maps, so a reader that finds it there also sees it as
	 * explicit (efa_av_array publishes with release/acquire).
	 */
	efa_rdm_av_entry_set_fi_addr(av_entry, new_fi_addr);
	efa_rdm_av_reverse_av_add_reserved(av->cur_reverse_av,
					   &rdm_av->prv_reverse_av,
					   &av_entry->efa_av_entry, prv_entry);
	err = efa_av_array_insert(av->addr_to_entry_map, new_fi_addr,
				  &av_entry->efa_av_entry);
	assert(!err);

	/* Leave the implicit AV. */
	err = efa_av_array_insert(rdm_av->addr_to_entry_map_implicit,
				  implicit_fi_addr, NULL);
	assert(!err);
	efa_rdm_av_reverse_av_remove(rdm_av->cur_reverse_av_implicit,
				     &rdm_av->prv_reverse_av_implicit,
				     &av_entry->efa_av_entry);
	cleanup_err = ofi_av_remove_addr(&rdm_av->util_av_implicit, implicit_fi_addr);
	if (cleanup_err)
		EFA_WARN(FI_LOG_AV, "Failed to remove implicit fi_addr %" PRIu64 " from implicit util AV: %s\n",
			 implicit_fi_addr, fi_strerror(-cleanup_err));
	efa_rdm_av_entry_set_implicit_fi_addr(av_entry, FI_ADDR_NOTAVAIL);
	dlist_remove(&av_entry->implicit_av_lru_entry);

	/* Move the AH reference from implicit to explicit. */
	rdm_ah = (struct efa_rdm_ah *) av_entry->efa_av_entry.ah;
	assert(!dlist_empty(&rdm_ah->implicit_conn_list));
	dlist_remove(&av_entry->ah_implicit_conn_list_entry);
	efa_rdm_ah_implicit_av_lru_ah_move(av->domain, &rdm_ah->efa_ah);
	rdm_ah->implicit_refcnt--;
	rdm_ah->explicit_refcnt++;

	*fi_addr = new_fi_addr;

	EFA_INFO(FI_LOG_AV,
		 "Peer with implicit fi_addr %" PRIu64
		 " moved to explicit AV. Explicit fi_addr: %" PRIu64 "\n",
		 implicit_fi_addr, new_fi_addr);

	/* Call foreach_unspec_addr to move unexpected messages
	 * from the unspecified queue to the specified queues.
	 *
	 * util_ep is bound to the explicit util_av, so the explicit util_av's
	 * ep_list contains all of the endpoints bound to this AV */
	EFA_GENLOCK_LOCK(&av->util_av.ep_list_lock, efa_av_ep_list_lock_sym);
	dlist_foreach(&av->util_av.ep_list, entry) {
		ep = container_of(entry, struct efa_rdm_ep, base_ep.util_ep.av_entry);
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		peer_srx->owner_ops->foreach_unspec_addr(peer_srx, &efa_rdm_av_get_addr_from_peer_rx_entry);
	}
	EFA_GENLOCK_UNLOCK(&av->util_av.ep_list_lock, efa_av_ep_list_lock_sym);

	return FI_SUCCESS;

err_remove_addr:
	cleanup_err = ofi_av_remove_addr(&av->util_av, new_fi_addr);
	if (cleanup_err)
		EFA_WARN(FI_LOG_AV, "Failed to remove fi_addr %" PRIu64 " from explicit util AV during cleanup: %s\n",
			 new_fi_addr, fi_strerror(-cleanup_err));
	return err;
}


/**
 * @brief insert one address into the explicit AV (RDM), migrating from the
 * implicit AV if the address is already present there
 *
 * If the address already exists in the explicit AV, return the existing
 * fi_addr. If it exists in the implicit AV, move it from implicit to
 * explicit. Otherwise allocate a new connection entry in the explicit AV.
 */
static int efa_rdm_av_insert_one_explicit(struct efa_av *av, struct efa_ep_addr *addr,
					  fi_addr_t *fi_addr, uint64_t flags,
					  void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	char raw_gid_str[INET6_ADDRSTRLEN];
	struct efa_rdm_av_entry *av_entry;
	fi_addr_t efa_fiaddr;
	fi_addr_t implicit_fi_addr;
	int ret;

	ret = efa_av_insert_one_validate(addr, fi_addr, raw_gid_str);
	if (ret)
		return ret;

	EFA_INFO(FI_LOG_AV,
		 "Inserting address GID[%s] QP[%u] QKEY[%u] to explicit AV\n",
		 raw_gid_str, addr->qpn, addr->qkey);

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);

	/* Check if this address already exists in the explicit AV */
	efa_fiaddr = ofi_av_lookup_fi_addr_unsafe(&av->util_av, addr);
	if (efa_fiaddr != FI_ADDR_NOTAVAIL) {
		EFA_INFO(FI_LOG_AV,
			 "Found existing AV entry pointing to this "
			 "address! fi_addr: %" PRId64 "\n",
			 efa_fiaddr);
		*fi_addr = efa_fiaddr;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return 0;
	}

	/* Check if this address exists in the implicit AV */
	EFA_GENLOCK_LOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	implicit_fi_addr = ofi_av_lookup_fi_addr_unsafe(&rdm_av->util_av_implicit, addr);
	if (implicit_fi_addr != FI_ADDR_NOTAVAIL) {
		EFA_INFO(FI_LOG_AV,
			 "Found implicit AV entry id %" PRId64
			 " for the same address\n",
			 implicit_fi_addr);

		ret = efa_rdm_av_entry_implicit_to_explicit(av, addr, implicit_fi_addr,
							  fi_addr);
		if (ret)
			*fi_addr = FI_ADDR_NOTAVAIL;

		EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return ret;
	}
	EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);

	/* Address not found in either AV, allocate a new explicit entry */
	av_entry = efa_rdm_av_entry_alloc_explicit(av, addr, flags, context);
	if (!av_entry) {
		*fi_addr = FI_ADDR_NOTAVAIL;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		return -FI_EADDRNOTAVAIL;
	}

	*fi_addr = efa_rdm_av_entry_fi_addr(av_entry);
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	EFA_INFO(FI_LOG_AV,
		 "Successfully inserted address GID[%s] QP[%u] "
		 "QKEY[%u] to explicit AV. fi_addr: %" PRId64 "\n",
		 raw_gid_str, addr->qpn, addr->qkey, *fi_addr);

	return 0;
}


/**
 * @brief insert one address into the implicit address vector (RDM only)
 *
 * Unconditionally allocates a new connection entry in the implicit AV. The
 * caller must have already established, while holding the locks below, that
 * the address is in neither the explicit nor the implicit AV. Otherwise a
 * duplicate entry for the same address is created. util_av.lock is required
 * for that reason even though this function does not touch the explicit AV.
 *
 * The caller owns the locks for the whole call; this function neither
 * acquires nor releases them.
 */
int efa_rdm_av_insert_one_implicit(struct efa_av *av, struct efa_ep_addr *addr,
				   fi_addr_t *fi_addr, uint64_t flags,
				   void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym,
			 efa_implicit_av_lock_sym)
{
	char raw_gid_str[INET6_ADDRSTRLEN];
	struct efa_rdm_av_entry *av_entry;
	int ret;

	ret = efa_av_insert_one_validate(addr, fi_addr, raw_gid_str);
	if (ret)
		return ret;

	EFA_INFO(FI_LOG_AV,
		 "Inserting address GID[%s] QP[%u] QKEY[%u] to implicit AV\n",
		 raw_gid_str, addr->qpn, addr->qkey);

	assert(ofi_av_lookup_fi_addr_unsafe(&av->util_av, addr) == FI_ADDR_NOTAVAIL);

	av_entry = efa_rdm_av_entry_alloc_implicit(av, addr, flags, context);
	if (!av_entry) {
		*fi_addr = FI_ADDR_NOTAVAIL;
		return -FI_EADDRNOTAVAIL;
	}

	*fi_addr = efa_rdm_av_entry_implicit_fi_addr(av_entry);

	EFA_INFO(FI_LOG_AV,
		 "Successfully inserted address GID[%s] QP[%u] "
		 "QKEY[%u] to implicit AV. fi_addr: %" PRId64 "\n",
		 raw_gid_str, addr->qpn, addr->qkey, *fi_addr);

	return 0;
}


static int efa_rdm_av_insert(struct fid_av *av_fid, const void *addr,
			     size_t count, fi_addr_t *fi_addr,
			     uint64_t flags, void *context)
{
	struct efa_av *av = container_of(av_fid, struct efa_av, util_av.av_fid);
	int ret = 0, success_cnt = 0;
	size_t i = 0;
	struct efa_ep_addr *addr_i;
	fi_addr_t fi_addr_res;

	if (av->util_av.flags & FI_EVENT)
		return -FI_ENOEQ;

	if ((flags & FI_SYNC_ERR) && (!context || (flags & FI_EVENT)))
		return -FI_EINVAL;

	/*
	 * Providers are allowed to ignore FI_MORE.
	 */
	flags &= ~FI_MORE;
	if (flags)
		return -FI_ENOSYS;

	/*
	 * Acquire domain lock because AH is a domain-level resource whose fields
	 * are modified during av insert.
	 * The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock in the AV insertion and removal
	 * paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	for (i = 0; i < count; i++) {
		addr_i = (struct efa_ep_addr *) ((uint8_t *)addr + i * EFA_EP_ADDR_LEN);

		ret = efa_rdm_av_insert_one_explicit(av, addr_i, &fi_addr_res, flags, context);
		if (ret) {
			EFA_WARN(FI_LOG_AV, "insert raw_addr to av failed! ret=%d\n",
				 ret);
			break;
		}

		if (fi_addr)
			fi_addr[i] = fi_addr_res;
		success_cnt++;
	}

	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	/* cancel remaining request and log to event queue */
	for (; i < count ; i++) {
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}

	return success_cnt;
}


static int efa_rdm_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
			     size_t count, uint64_t flags)
{
	int err = 0;
	size_t i;
	struct efa_av *av;
	struct efa_rdm_av_entry *av_entry;

	if (!fi_addr)
		return -FI_EINVAL;

	av = container_of(av_fid, struct efa_av, util_av.av_fid);
	if (av->type != FI_AV_TABLE)
		return -FI_EINVAL;

	/* The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock in the AV insertion and removal
	 * paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	for (i = 0; i < count; i++) {
		av_entry = efa_rdm_av_addr_to_entry(av, fi_addr[i]);
		if (!av_entry) {
			err = -FI_EINVAL;
			break;
		}

		efa_rdm_av_entry_release_explicit(av, av_entry);
	}

	if (i < count) {
		/* something went wrong, so err cannot be zero */
		assert(err);
	}

	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	return err;
}


static struct fi_ops_av efa_rdm_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = efa_rdm_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = efa_rdm_av_remove,
	.lookup = efa_av_lookup,
	.straddr = efa_av_straddr,
	.lookup2 = ofi_av_lookup2,
};


/*
 * Release an explicit conn reached through the explicit forward map. Called
 * only from the close path, where clearing the slot from under the iteration is
 * safe because efa_av_array_iter has already loaded the pointer. Every live
 * conn is in exactly one forward map, including conns that QPN reuse moved into
 * a prv_reverse_av.
 */
static int efa_rdm_av_close_release_explicit(struct efa_av_array *arr,
					     void *entry, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_util_av_lock_sym)
{
	struct efa_av *av = context;

	efa_rdm_av_entry_release_explicit(
		av, container_of((struct efa_av_entry *) entry,
				 struct efa_rdm_av_entry, efa_av_entry));
	return 0;
}

/* Implicit AV counterpart of efa_rdm_av_close_release_explicit. */
static int efa_rdm_av_close_release_implicit(struct efa_av_array *arr,
					     void *entry, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym, efa_implicit_av_lock_sym)
{
	struct efa_av *av = context;

	efa_rdm_av_entry_release_implicit(
		av, container_of((struct efa_av_entry *) entry,
				 struct efa_rdm_av_entry, efa_av_entry));
	return 0;
}

static int efa_rdm_av_close(struct fid *fid)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_av *av;
	struct efa_rdm_av *rdm_av;
	struct efa_ep_addr_hashable *ep_addr_hashable, *tmp;
	int err = 0;

	av = container_of(fid, struct efa_av, util_av.av_fid.fid);
	rdm_av = ((struct efa_rdm_av *)(av));

	/* The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock -> util_av_implicit.lock
	 * in the AV insertion, removal and CQ read paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);
	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	efa_av_array_iter(av->addr_to_entry_map, av,
			  efa_rdm_av_close_release_explicit);
	assert(!rdm_av->prv_reverse_av);

	EFA_GENLOCK_LOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	efa_av_array_iter(rdm_av->addr_to_entry_map_implicit, av,
			  efa_rdm_av_close_release_implicit);
	assert(!rdm_av->prv_reverse_av_implicit);
	assert(dlist_empty(&rdm_av->implicit_av_lru_list));
	EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);

	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
	EFA_GENLOCK_UNLOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	err = ofi_av_close(&av->util_av);
	if (OFI_UNLIKELY(err))
		EFA_WARN(FI_LOG_AV, "Failed to close util av: %s\n", fi_strerror(-err));

	err = ofi_av_close(&rdm_av->util_av_implicit);
	if (OFI_UNLIKELY(err))
		EFA_WARN(FI_LOG_AV, "Failed to close implicit util av: %s\n", fi_strerror(-err));

	if (rdm_av->shm_rdm_av) {
		err = fi_close(&rdm_av->shm_rdm_av->fid);
		if (OFI_UNLIKELY(err))
			EFA_WARN(FI_LOG_AV, "Failed to close shm av: %s\n", fi_strerror(-err));
	}
	HASH_ITER(hh, rdm_av->evicted_peers_hashset, ep_addr_hashable, tmp) {
		HASH_DEL(rdm_av->evicted_peers_hashset, ep_addr_hashable);
		free(ep_addr_hashable);
	}

	efa_av_array_destroy(av->addr_to_entry_map);
	efa_av_array_destroy(rdm_av->addr_to_entry_map_implicit);
	efa_av_array_destroy(av->cur_reverse_av);
	efa_av_array_destroy(rdm_av->cur_reverse_av_implicit);
	ofi_bufpool_destroy(rdm_av->conn_pool);
	ofi_genlock_destroy(&rdm_av->conn_pool_lock);

	free(rdm_av);
	return err;
}


static struct fi_ops efa_rdm_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_rdm_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


int efa_rdm_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		    struct fid_av **av_fid, void *context)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_domain *efa_domain;
	struct efa_rdm_av *rdm_av;
	struct efa_av *av;
	struct fi_av_attr av_attr = { 0 };
	int ret, retv;

	ret = efa_av_open_prepare_attr(domain_fid, attr, &efa_domain);
	if (ret)
		return ret;

	rdm_av = calloc(1, sizeof(*rdm_av));
	if (!rdm_av)
		return -FI_ENOMEM;
	av = &rdm_av->efa_av;

	/* Conns live in conn_pool; the util AV entries carry no context. */
	ret = efa_av_init_base(av, efa_domain, attr, context, 0);
	if (ret)
		goto err_free;

	ret = efa_av_array_init(&rdm_av->addr_to_entry_map_implicit);
	if (ret)
		goto err_destruct_base;

	ret = efa_av_reverse_av_init(&rdm_av->cur_reverse_av_implicit);
	if (ret)
		goto err_destroy_implicit_map;

	ret = efa_av_init_util_av(efa_domain, attr, &rdm_av->util_av_implicit, context, 0);
	if (ret)
		goto err_destroy_implicit_reverse_av;

	ret = ofi_genlock_init(&rdm_av->conn_pool_lock,
			       efa_domain->util_domain.threading == FI_THREAD_DOMAIN &&
			       efa_domain->util_domain.control_progress ==
				       FI_PROGRESS_CONTROL_UNIFIED ?
			       OFI_LOCK_NOOP : OFI_LOCK_MUTEX);
	if (ret)
		goto err_close_util_av_implicit;

	ret = ofi_bufpool_create(&rdm_av->conn_pool,
				 sizeof(struct efa_rdm_av_entry), 16,
				 0, /* no limit to max_cnt */
				 1024, OFI_BUFPOOL_INDEXED);
	if (ret)
		goto err_destroy_conn_pool_lock;

	if (efa_domain->fabric &&
	    ((struct efa_rdm_fabric *) efa_domain->fabric)->shm_fabric) {
		struct efa_rdm_domain *rdm_domain =
			(struct efa_rdm_domain *) efa_domain;
		/*
		 * shm av supports maximum 256 entries
		 * Reset the count to 128 to reduce memory footprint and satisfy
		 * the need of the instances with more CPUs.
		 */
		av_attr = *attr;
		if (efa_env.shm_av_size > EFA_SHM_MAX_AV_COUNT) {
			ret = -FI_ENOSYS;
			EFA_WARN(FI_LOG_AV,
				 "The requested av size is beyond"
				 " shm supported maximum av size: %s\n",
				 fi_strerror(-ret));
			goto err_destroy_conn_pool;
		}
		av_attr.count = efa_env.shm_av_size;
		assert(av_attr.type == FI_AV_TABLE);
		ret = fi_av_open(rdm_domain->shm_domain, &av_attr,
				 &rdm_av->shm_rdm_av, context);
		if (ret)
			goto err_destroy_conn_pool;
	}

	EFA_INFO(FI_LOG_AV, "fi_av_attr:%" PRId64 "\n",
			attr->flags);

	rdm_av->implicit_av_size = efa_env.implicit_av_size;
	rdm_av->shm_used = 0;

	*av_fid = &av->util_av.av_fid;
	(*av_fid)->fid.fclass = FI_CLASS_AV;
	(*av_fid)->fid.context = context;
	(*av_fid)->fid.ops = &efa_rdm_av_fi_ops;
	(*av_fid)->ops = &efa_rdm_av_ops;

	dlist_init(&rdm_av->implicit_av_lru_list);

	return 0;

err_destroy_conn_pool:
	ofi_bufpool_destroy(rdm_av->conn_pool);

err_destroy_conn_pool_lock:
	ofi_genlock_destroy(&rdm_av->conn_pool_lock);

err_close_util_av_implicit:
	retv = ofi_av_close(&rdm_av->util_av_implicit);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av_implicit: %s\n", fi_strerror(-retv));

err_destroy_implicit_reverse_av:
	efa_av_array_destroy(rdm_av->cur_reverse_av_implicit);

err_destroy_implicit_map:
	efa_av_array_destroy(rdm_av->addr_to_entry_map_implicit);

err_destruct_base:
	retv = ofi_av_close(&av->util_av);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av: %s\n", fi_strerror(-retv));
	efa_av_array_destroy(av->addr_to_entry_map);
	efa_av_array_destroy(av->cur_reverse_av);

err_free:

	free(rdm_av);
	return ret;
}
