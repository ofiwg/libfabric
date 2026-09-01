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
 * The efa_rdm_ah_* helpers below layer the RDM-only AH policy (split
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
 */
void efa_rdm_ah_implicit_av_lru_ah_move(struct efa_domain *domain,
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

		assert(av_entry_to_release->implicit_fi_addr != FI_ADDR_NOTAVAIL &&
		       av_entry_to_release->efa_av_entry.fi_addr == FI_ADDR_NOTAVAIL);

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
 * Layers the RDM-only policy (split explicit/implicit reference counts, the
 * per-domain AH LRU list and OOM eviction of implicit-only AHs) on top of the
 * base efa_ah_alloc, which does the shared find-or-create and manages the
 * single base reference count.
 */
struct efa_ah *efa_rdm_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
				bool insert_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_domain *rdm_domain = (struct efa_rdm_domain *) domain;
	struct efa_ah *ah;
	struct efa_rdm_ah *rdm_ah;
	int err;

	assert(domain->info_type == EFA_INFO_RDM);

	ah = efa_ah_alloc(domain, gid, sizeof(struct efa_rdm_ah));
	if (!ah && errno == FI_ENOMEM) {
		EFA_INFO(FI_LOG_AV,
			 "ibv_create_ah failed with ENOMEM for %s AV insertion. "
			 "Attempting to evict AH entry\n",
			 insert_implicit_av ? "implicit" : "explicit");
		err = efa_rdm_ah_implicit_av_evict_ah(domain, insert_implicit_av);
		if (err)
			return NULL;
		ah = efa_ah_alloc(domain, gid, sizeof(struct efa_rdm_ah));
	}
	if (!ah)
		return NULL;

	rdm_ah = ((struct efa_rdm_ah *)(ah));
	if (ah->refcnt == 1) {
		/* Newly created: initialize RDM-only state and add to LRU */
		rdm_ah->explicit_refcnt = 0;
		rdm_ah->implicit_refcnt = 0;
		dlist_init(&rdm_ah->implicit_conn_list);
		dlist_insert_tail(&rdm_ah->domain_lru_ah_list_entry,
				  &rdm_domain->ah_lru_list);
	} else {
		efa_rdm_ah_implicit_av_lru_ah_move(domain, ah);
	}

	insert_implicit_av ? rdm_ah->implicit_refcnt++ : rdm_ah->explicit_refcnt++;
	return ah;
}

/**
 * @brief release an RDM AH reference (split refcnt + LRU unlink + base release)
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


/**
 * @brief Add the entry to the implicit AV LRU list; if the list is full, evict
 * the least recently used entry at the front and add the latest one.
 */
static inline int efa_rdm_av_implicit_av_lru_insert(struct efa_av *av,
						    struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
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

	dlist_pop_front(&rdm_av->implicit_av_lru_list, struct efa_rdm_av_entry,
			av_entry_to_release, implicit_av_lru_entry);
	EFA_INFO(FI_LOG_AV,
		 "Evicting AV entry for peer implicit fi_addr %" PRIu64
		 " AHN %" PRIu16 " QPN %" PRIu16 " QKEY %" PRIu32 " from "
		 "implicit AV\n",
		 av_entry_to_release->implicit_fi_addr,
		 av_entry_to_release->efa_av_entry.ah->ahn,
		 efa_av_entry_ep_addr(&av_entry_to_release->efa_av_entry)->qpn,
		 efa_av_entry_ep_addr(&av_entry_to_release->efa_av_entry)->qkey);

	/* Add to hashset with list of evicted peers */
	ep_addr_hashable = malloc(sizeof(struct efa_ep_addr_hashable));
	if (!ep_addr_hashable) {
		EFA_WARN(FI_LOG_AV, "Could not allocate memory for LRU AV entry hashset entry\n");
		return FI_ENOMEM;
	}
	memcpy(ep_addr_hashable, efa_av_entry_ep_addr(&av_entry->efa_av_entry), sizeof(struct efa_ep_addr));
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
 */
static int efa_rdm_av_entry_insert_shm_av(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
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

		av_entry->shm_fi_addr = av_entry->efa_av_entry.fi_addr;
		ret = fi_av_insert(rdm_av->shm_rdm_av, smr_name, 1, &av_entry->shm_fi_addr, FI_AV_USER_ID, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			EFA_WARN(FI_LOG_AV,
				 "Failed to insert address to shm provider's av: %s\n",
				 fi_strerror(-ret));
			return ret;
		}

		EFA_INFO(FI_LOG_AV,
			"Successfully inserted %s to shm provider's av. efa_fiaddr: %ld shm_fiaddr = %ld\n",
			smr_name, av_entry->efa_av_entry.fi_addr, av_entry->shm_fi_addr);

		assert(av_entry->shm_fi_addr < efa_env.shm_av_size);
		rdm_av->shm_used++;
	}

	return 0;
}


/**
 * @brief release the rdm related resources of an efa_rdm_av_entry (shm + peers)
 */
static void efa_rdm_av_entry_deinit(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	int err;
	struct dlist_entry *entry, *tmp;
	struct efa_rdm_ep *ep;
	struct efa_av_array *peer_map;
	struct efa_rdm_peer *peer;
	fi_addr_t fi_addr;

	assert(av->domain->info_type == EFA_INFO_RDM);

	assert((av_entry->efa_av_entry.fi_addr != FI_ADDR_NOTAVAIL &&
		av_entry->implicit_fi_addr == FI_ADDR_NOTAVAIL) ||
	       (av_entry->implicit_fi_addr != FI_ADDR_NOTAVAIL &&
		av_entry->efa_av_entry.fi_addr == FI_ADDR_NOTAVAIL));

	if (av_entry->shm_fi_addr != FI_ADDR_NOTAVAIL && rdm_av->shm_rdm_av) {
		err = fi_av_remove(rdm_av->shm_rdm_av, &av_entry->shm_fi_addr, 1, 0);
		if (err) {
			EFA_WARN(FI_LOG_AV,
				 "remove address from shm av failed! err=%d\n",
				 err);
		} else {
			rdm_av->shm_used--;
			assert(av_entry->shm_fi_addr < efa_env.shm_av_size);
		}
	}

	/* since an av entry is all connections to a specific remote ep, we must
	 * walk all local ep peer maps and remove the connection to the remote ep */
	ofi_genlock_lock(&av->util_av.ep_list_lock);
	dlist_foreach_safe(&av->util_av.ep_list, entry, tmp) {
		ep = container_of(entry, struct efa_rdm_ep,
				  base_ep.util_ep.av_entry);
		if (av_entry->efa_av_entry.fi_addr != FI_ADDR_NOTAVAIL) {
			peer_map = ep->fi_addr_to_peer_map;
			fi_addr = av_entry->efa_av_entry.fi_addr;
		} else {
			peer_map = ep->fi_addr_to_peer_map_implicit;
			fi_addr = av_entry->implicit_fi_addr;
		}
		EFA_GENLOCK_LOCK(&ep->ctrl_lock, efa_ctrl_lock_sym);
		peer = efa_rdm_ep_peer_map_remove(peer_map, fi_addr);
		if (peer) {
			efa_rdm_peer_destruct(peer, ep);
			ofi_buf_free(peer);
		}
		EFA_GENLOCK_UNLOCK(&ep->ctrl_lock, efa_ctrl_lock_sym);
	}
	ofi_genlock_unlock(&av->util_av.ep_list_lock);
}


/**
 * @brief allocate an explicit efa_rdm_av_entry (base entry + rdm state + shm).
 * caller of this function must hold av->util_av.lock
 */
struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_explicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct util_av *util_av = &av->util_av;
	struct util_av_entry *util_av_entry;
	struct efa_av_entry *entry;
	struct efa_rdm_av_entry *av_entry;
	fi_addr_t fi_addr;
	int err;

	assert(ofi_genlock_held(&av->util_av.lock));

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	err = ofi_av_insert_addr(util_av, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	util_av_entry = ofi_bufpool_get_ibuf(util_av->av_entry_pool, fi_addr);
	entry = (struct efa_av_entry *)util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(entry)));
	assert(av->type == FI_AV_TABLE);
	entry->fi_addr = fi_addr;

	entry->ah = efa_rdm_ah_alloc(av->domain, raw_addr->raw, false);
	if (!entry->ah)
		goto err_remove_addr;

	err = efa_av_array_insert(av->addr_to_entry_map, fi_addr, entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert entry for fi_addr %" PRIu64
			" into array: %s\n", fi_addr, fi_strerror(-err));
		goto err_release_ah;
	}

	if (efa_rdm_av_reverse_av_add(&av->cur_reverse_av, &rdm_av->prv_reverse_av,
				      entry)) {
		EFA_WARN(FI_LOG_AV, "Failed to insert entry for fi_addr %" PRIu64
			" into reverse AV\n", fi_addr);
		efa_rdm_ah_release(av->domain, entry->ah, false);
		efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
						 entry, fi_addr);
		return NULL;
	}

	av_entry = container_of(entry, struct efa_rdm_av_entry, efa_av_entry);
	av_entry->av = rdm_av;
	av_entry->implicit_fi_addr = FI_ADDR_NOTAVAIL;
	av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	dlist_init(&av_entry->implicit_av_lru_entry);
	dlist_init(&av_entry->ah_implicit_conn_list_entry);

	/*
	 * The explicit AV insertion is triggered by the application calling the
	 * fi_av_insert API. Attempt shm av insertion; efa_rdm_av_entry_insert_shm_av is
	 * a no-op for peers that are not local.
	 */
	err = efa_rdm_av_entry_insert_shm_av(av, av_entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert fi_addr %" PRIu64
			" into shm provider's AV: %s\n", fi_addr, fi_strerror(-err));
		efa_rdm_av_reverse_av_remove(&av->cur_reverse_av,
					     &rdm_av->prv_reverse_av, entry);
		efa_rdm_ah_release(av->domain, entry->ah, false);
		efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
						 entry, fi_addr);
		return NULL;
	}

	return av_entry;

err_release_ah:
	efa_rdm_ah_release(av->domain, entry->ah, false);
err_remove_addr:
	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));
	return NULL;
}


/**
 * @brief allocate an efa_rdm_av_entry in the implicit AV (RDM only).
 * caller of this function must hold av->util_av_implicit.lock
 */
struct efa_rdm_av_entry *efa_rdm_av_entry_alloc_implicit(struct efa_av *av,
						   struct efa_ep_addr *raw_addr,
						   uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct util_av *util_av_implicit = &rdm_av->util_av_implicit;
	struct util_av_entry *util_av_entry;
	struct efa_av_entry *efa_av_entry;
	struct efa_rdm_av_entry *av_entry;
	fi_addr_t fi_addr;
	int err;

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	assert(av->domain->info_type == EFA_INFO_RDM);

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	err = ofi_av_insert_addr(util_av_implicit, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	util_av_entry = ofi_bufpool_get_ibuf(util_av_implicit->av_entry_pool, fi_addr);
	efa_av_entry = (struct efa_av_entry *)util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(efa_av_entry)));
	assert(av->type == FI_AV_TABLE);

	av_entry = container_of(efa_av_entry, struct efa_rdm_av_entry, efa_av_entry);
	av_entry->av = rdm_av;
	av_entry->efa_av_entry.fi_addr = FI_ADDR_NOTAVAIL;
	av_entry->implicit_fi_addr = fi_addr;
	av_entry->shm_fi_addr = FI_ADDR_NOTAVAIL;
	dlist_init(&av_entry->implicit_av_lru_entry);
	dlist_init(&av_entry->ah_implicit_conn_list_entry);

	err = efa_rdm_av_implicit_av_lru_insert(av, av_entry);
	if (err)
		return NULL;

	av_entry->efa_av_entry.ah = efa_rdm_ah_alloc(av->domain, raw_addr->raw, true);
	if (!av_entry->efa_av_entry.ah)
		goto err_release;

	dlist_insert_tail(&av_entry->ah_implicit_conn_list_entry,
			  &((struct efa_rdm_ah *)(av_entry->efa_av_entry.ah))->implicit_conn_list);

	err = efa_rdm_av_reverse_av_add(&rdm_av->cur_reverse_av_implicit,
					&rdm_av->prv_reverse_av_implicit, efa_av_entry);
	if (err) {
		efa_rdm_av_entry_deinit(av, av_entry);
		goto err_release;
	}

	err = efa_av_array_insert(rdm_av->addr_to_entry_map_implicit, fi_addr, efa_av_entry);
	if (err) {
		efa_rdm_av_reverse_av_remove(&rdm_av->cur_reverse_av_implicit,
					     &rdm_av->prv_reverse_av_implicit, efa_av_entry);
		efa_rdm_av_entry_deinit(av, av_entry);
		goto err_release;
	}
	return av_entry;

err_release:
	dlist_remove(&av_entry->implicit_av_lru_entry);
	if (av_entry->efa_av_entry.ah) {
		dlist_remove(&av_entry->ah_implicit_conn_list_entry);
		efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
	}

	memset(av_entry->efa_av_entry.ep_addr, 0, EFA_EP_ADDR_LEN);
	err = ofi_av_remove_addr(util_av_implicit, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for implicit fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));

	return NULL;
}


/**
 * @brief release an explicit efa_rdm_av_entry (rdm teardown + base teardown).
 * Caller must hold util_domain + util_av.
 */
void efa_rdm_av_entry_release_explicit(struct efa_av *av,
				 struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));

	assert(ofi_genlock_held(&av->util_av.lock));

	efa_rdm_av_reverse_av_remove(&av->cur_reverse_av, &rdm_av->prv_reverse_av,
				     &av_entry->efa_av_entry);
	efa_rdm_av_entry_deinit(av, av_entry);
	efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, false);
	efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
					 &av_entry->efa_av_entry,
					 av_entry->efa_av_entry.fi_addr);
}


/**
 * @brief release an efa_rdm_av_entry from the implicit AV
 */
void efa_rdm_av_entry_release_implicit(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	efa_rdm_av_reverse_av_remove(&rdm_av->cur_reverse_av_implicit,
				     &rdm_av->prv_reverse_av_implicit,
				     &av_entry->efa_av_entry);

	efa_rdm_av_entry_deinit(av, av_entry);

	dlist_remove(&av_entry->ah_implicit_conn_list_entry);
	efa_rdm_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
	efa_av_entry_remove_from_util_av(rdm_av->addr_to_entry_map_implicit,
					 &rdm_av->util_av_implicit,
					 &av_entry->efa_av_entry,
					 av_entry->implicit_fi_addr);
}


/**
 * @brief release an implicit efa_rdm_av_entry during AH eviction
 */
void efa_rdm_av_entry_release_implicit_ah_unsafe(struct efa_av *av,
					   struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));

	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));
	efa_rdm_av_reverse_av_remove(&rdm_av->cur_reverse_av_implicit,
				     &rdm_av->prv_reverse_av_implicit,
				     &av_entry->efa_av_entry);

	efa_rdm_av_entry_deinit(av, av_entry);

	assert(ofi_genlock_held(&av->domain->util_domain.lock));
	dlist_remove(&av_entry->ah_implicit_conn_list_entry);

	efa_av_entry_remove_from_util_av(rdm_av->addr_to_entry_map_implicit,
					 &rdm_av->util_av_implicit,
					 &av_entry->efa_av_entry,
					 av_entry->implicit_fi_addr);
	((struct efa_rdm_ah *)(av_entry->efa_av_entry.ah))->implicit_refcnt--;
	av_entry->efa_av_entry.ah->refcnt--;
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


static inline struct efa_av_entry *
efa_rdm_av_reverse_lookup_entry(struct efa_cur_reverse_av **cur_reverse_av,
				struct efa_prv_reverse_av **prv_reverse_av,
				uint16_t ahn, uint16_t qpn,
				struct efa_rdm_pke *pkt_entry)
{
	uint32_t *connid;
	struct efa_cur_reverse_av *cur_entry;
	struct efa_prv_reverse_av *prv_entry;
	struct efa_cur_reverse_av_key cur_key;
	struct efa_prv_reverse_av_key prv_key;

	cur_key.ahn = ahn;
	cur_key.qpn = qpn;

	/* coverity[overflow_const : FALSE] - intentional unsigned wraparound in uthash Jenkins hash */
	HASH_FIND(hh, *cur_reverse_av, &cur_key, sizeof(cur_key), cur_entry);

	if (OFI_UNLIKELY(!cur_entry))
		return NULL;

	if (!pkt_entry) {
		/**
		 * There is no packet entry to extract connid from when we get
		 * an IBV_WC_RECV_RDMA_WITH_IMM completion from rdma-core. Or
		 * the pkt_entry is allocated from a buffer user posted that
		 * doesn't expect any pkt hdr.
		 */
		return cur_entry->entry;
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
		return cur_entry->entry;
	}

	if (OFI_LIKELY(*connid == efa_av_entry_ep_addr(cur_entry->entry)->qkey))
		return cur_entry->entry;

	/* the packet is from a previous peer, look for its address from the
	 * prv_reverse_av */
	prv_key.ahn = ahn;
	prv_key.qpn = qpn;
	prv_key.connid = *connid;
	HASH_FIND(hh, *prv_reverse_av, &prv_key, sizeof(prv_key), prv_entry);

	return OFI_LIKELY(!!prv_entry) ? prv_entry->entry : NULL;
};


/**
 * @brief find fi_addr for rdm endpoint in the explicit AV (connid aware)
 *
 * @param[in]	av	address vector
 * @param[in]	ahn	address handle number
 * @param[in]	qpn	QP number
 * @param[in]   pkt_entry	NULL or rdm packet entry, used to extract connid
 * @return	On success, return fi_addr to the peer who sent the packet.
 * 		If no such peer exists, return FI_ADDR_NOTAVAIL
 */
fi_addr_t efa_rdm_av_reverse_lookup(struct efa_av *av, uint16_t ahn,
				    uint16_t qpn, struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;
	fi_addr_t fi_addr;

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	entry = efa_rdm_av_reverse_lookup_entry(
		&av->cur_reverse_av, &rdm_av->prv_reverse_av, ahn, qpn, pkt_entry);
	fi_addr = (OFI_LIKELY(!!entry)) ? entry->fi_addr : FI_ADDR_NOTAVAIL;
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	return fi_addr;
}


/**
 * @brief find fi_addr for rdm endpoint in the implicit AV (connid aware)
 *
 * @param[in]	av	address vector
 * @param[in]	ahn	address handle number
 * @param[in]	qpn	QP number
 * @param[in]   pkt_entry	NULL or rdm packet entry, used to extract connid
 * @return	On success, return fi_addr to the peer who sent the packet.
 * 		If no such peer exists, return FI_ADDR_NOTAVAIL
 */
fi_addr_t efa_rdm_av_reverse_lookup_implicit(struct efa_av *av, uint16_t ahn,
					     uint16_t qpn,
					     struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	struct efa_av_entry *entry;
	fi_addr_t implicit_fi_addr = FI_ADDR_NOTAVAIL;

	ofi_genlock_lock(&av->domain->util_domain.lock);
	EFA_GENLOCK_LOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	entry = efa_rdm_av_reverse_lookup_entry(&rdm_av->cur_reverse_av_implicit,
						&rdm_av->prv_reverse_av_implicit, ahn,
						qpn, pkt_entry);

	if (OFI_LIKELY(!!entry)) {
		struct efa_rdm_av_entry *av_entry =
			container_of(entry, struct efa_rdm_av_entry, efa_av_entry);
		efa_rdm_av_implicit_av_lru_move(av, av_entry);
		implicit_fi_addr = av_entry->implicit_fi_addr;
	}
	EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	ofi_genlock_unlock(&av->domain->util_domain.lock);

	return implicit_fi_addr;
}


/**
 * @brief Move the entry to the end of the implicit AV LRU list and bump its AH
 *
 * Moving the entry to the tail marks it as the most recently used implicit AV
 * entry.
 */
void efa_rdm_av_implicit_av_lru_move(struct efa_av *av,
				     struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
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


/*
 * @brief RDM reverse-AV add: base cur add/replace plus connid-keyed prv preserve
 *
 * A (ahn, qpn) collision means a QP number was reused. Only the RDM protocol
 * disambiguates reused QPNs via the connid-keyed prv_reverse_av, so the previous
 * connection is preserved there before the base cur slot is overwritten. The
 * efa-direct reverse lookup reads cur_reverse_av only and uses the base variant.
 *
 * @param[in,out]	cur_reverse_av	Reverse AV with AHN and QPN as key
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		entry		efa_av_entry object
 * @return		On success, return 0.
 * 			Otherwise, return a negative libfabric error code
 */
int efa_rdm_av_reverse_av_add(struct efa_cur_reverse_av **cur_reverse_av,
				     struct efa_prv_reverse_av **prv_reverse_av,
				     struct efa_av_entry *entry)
{
	struct efa_cur_reverse_av *cur_entry;
	struct efa_prv_reverse_av *prv_entry;
	struct efa_cur_reverse_av_key cur_key;

	memset(&cur_key, 0, sizeof(cur_key));
	cur_key.ahn = entry->ah->ahn;
	cur_key.qpn = efa_av_entry_ep_addr(entry)->qpn;
	cur_entry = NULL;

	/* coverity[overflow_const : FALSE] - intentional unsigned wraparound in uthash Jenkins hash */
	HASH_FIND(hh, *cur_reverse_av, &cur_key, sizeof(cur_key), cur_entry);
	if (cur_entry) {
		prv_entry = malloc(sizeof(*prv_entry));
		if (!prv_entry) {
			EFA_WARN(FI_LOG_AV, "Cannot allocate memory for prv_reverse_av entry\n");
			return -FI_ENOMEM;
		}

		prv_entry->key.ahn = cur_key.ahn;
		prv_entry->key.qpn = cur_key.qpn;
		prv_entry->key.connid = efa_av_entry_ep_addr(cur_entry->entry)->qkey;
		prv_entry->entry = cur_entry->entry;
		HASH_ADD(hh, *prv_reverse_av, key, sizeof(prv_entry->key), prv_entry);
	}

	return efa_av_reverse_av_add(cur_reverse_av, entry);
}


/*
 * @brief RDM reverse-AV remove: base cur remove, else drop from prv_reverse_av
 *
 * If the entry is no longer the current one for its (ahn, qpn) it was demoted
 * into the connid-keyed prv_reverse_av; remove it from there. efa-direct never
 * populates prv_reverse_av and uses the base variant.
 *
 * @param[in,out]	cur_reverse_av	Reverse AV with AHN and QPN as key
 * @param[in,out]	prv_reverse_av	Reverse AV with AHN, QPN and QKEY as key
 * @param[in]		entry		efa_av_entry object
 */
void efa_rdm_av_reverse_av_remove(struct efa_cur_reverse_av **cur_reverse_av,
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
	HASH_DEL(*prv_reverse_av, prv_reverse_av_entry);
	free(prv_reverse_av_entry);
}



static fi_addr_t
efa_rdm_av_get_addr_from_peer_rx_entry(struct fi_peer_rx_entry *rx_entry)
{
	struct efa_rdm_pke *pke;

	pke = (struct efa_rdm_pke *) rx_entry->peer_context;

	return pke->peer->av_entry->efa_av_entry.fi_addr;
}


static int efa_rdm_av_entry_implicit_to_explicit(struct efa_av *av,
					   struct efa_ep_addr *raw_addr,
					   fi_addr_t implicit_fi_addr,
					   fi_addr_t *fi_addr)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	int cleanup_err, err;
	struct efa_ah *ah;
	struct efa_rdm_av_entry *implicit_av_entry, *explicit_av_entry;
	struct efa_rdm_ep *ep;
	struct dlist_entry *entry;
	struct util_av_entry *explicit_util_av_entry;
	struct efa_rdm_peer *peer;
	struct efa_av_entry *explicit_base_entry;
	struct fid_peer_srx *peer_srx;

	EFA_INFO(FI_LOG_AV,
		 "Moving peer with implicit fi_addr %" PRIu64
		 " to explicit AV\n",
		 implicit_fi_addr);

	assert(EFA_GENLOCK_HELD(&av->util_av.lock, efa_util_av_lock_sym));
	assert(EFA_GENLOCK_HELD(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym));

	implicit_av_entry = efa_rdm_av_addr_to_entry_implicit(av, implicit_fi_addr);
	assert(implicit_av_entry);
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(&implicit_av_entry->efa_av_entry)));
	assert(implicit_av_entry->efa_av_entry.fi_addr == FI_ADDR_NOTAVAIL &&
	       implicit_av_entry->implicit_fi_addr == implicit_fi_addr);

	ah = implicit_av_entry->efa_av_entry.ah;

	/* Create explicit util AV entry */
	err = ofi_av_insert_addr(&av->util_av, raw_addr, fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV,
			 "Failed to insert implicit fi_addr %" PRIu64 " into explicit util AV: %s\n",
			 implicit_fi_addr, fi_strerror(-err));
		return err;
	}

	explicit_util_av_entry =
		ofi_bufpool_get_ibuf(av->util_av.av_entry_pool, *fi_addr);
	explicit_base_entry = (struct efa_av_entry *) explicit_util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(explicit_base_entry)));
	assert(av->type == FI_AV_TABLE);

	/* Copy information from the implicit entry to the explicit entry */
	explicit_av_entry = container_of(explicit_base_entry, struct efa_rdm_av_entry, efa_av_entry);
	explicit_base_entry->ah = implicit_av_entry->efa_av_entry.ah;
	explicit_base_entry->fi_addr = *fi_addr;
	explicit_av_entry->av = rdm_av;
	explicit_av_entry->shm_fi_addr = implicit_av_entry->shm_fi_addr;
	explicit_av_entry->implicit_fi_addr = FI_ADDR_NOTAVAIL;
	dlist_init(&explicit_av_entry->implicit_av_lru_entry);
	dlist_init(&explicit_av_entry->ah_implicit_conn_list_entry);

	err = efa_av_array_insert(av->addr_to_entry_map, *fi_addr, explicit_base_entry);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_AV, "Failed to insert explicit entry for fi_addr %" PRIu64 " into addr_to_entry_map: %s\n",
			 *fi_addr, fi_strerror(-err));
		cleanup_err = ofi_av_remove_addr(&av->util_av, *fi_addr);
		if (cleanup_err)
			EFA_WARN(FI_LOG_AV, "Failed to remove fi_addr %" PRIu64 " from explicit util AV during cleanup: %s\n",
				 *fi_addr, fi_strerror(-cleanup_err));
		return err;
	}

	err = efa_rdm_av_reverse_av_add(&av->cur_reverse_av, &rdm_av->prv_reverse_av,
					explicit_base_entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert explicit entry for fi_addr %" PRIu64 " into reverse AV: %s\n",
			 *fi_addr, fi_strerror(-err));
		cleanup_err = efa_av_array_insert(av->addr_to_entry_map, *fi_addr, NULL);
		assert(!cleanup_err);
		cleanup_err = ofi_av_remove_addr(&av->util_av, *fi_addr);
		if (cleanup_err)
			EFA_WARN(FI_LOG_AV, "Failed to remove fi_addr %" PRIu64 " from explicit util AV during cleanup: %s\n",
				 *fi_addr, fi_strerror(-cleanup_err));
		return err;
	}

	/* Handle reverse AV and AV ref counts */
	efa_rdm_av_reverse_av_remove(&rdm_av->cur_reverse_av_implicit,
				     &rdm_av->prv_reverse_av_implicit,
				     &implicit_av_entry->efa_av_entry);

	dlist_remove(&implicit_av_entry->implicit_av_lru_entry);
	err = efa_av_array_insert(rdm_av->addr_to_entry_map_implicit, implicit_fi_addr, NULL);
	assert(!err);

	err = ofi_av_remove_addr(&rdm_av->util_av_implicit, implicit_fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to remove implicit fi_addr %" PRIu64 " from implicit util AV: %s\n",
			 implicit_fi_addr, fi_strerror(-err));
		return err;
	}

	/* Handle AH LRU list and refcnt */
	assert(ofi_genlock_held(&av->domain->util_domain.lock));
	assert(!dlist_empty(&((struct efa_rdm_ah *)(ah))->implicit_conn_list));
	dlist_remove(&implicit_av_entry->ah_implicit_conn_list_entry);
	efa_rdm_ah_implicit_av_lru_ah_move(av->domain, ah);
	((struct efa_rdm_ah *)(ah))->implicit_refcnt--;
	((struct efa_rdm_ah *)(ah))->explicit_refcnt++;

	EFA_INFO(FI_LOG_AV,
		 "Peer with implicit fi_addr %" PRIu64
		 " moved to explicit AV. Explicit fi_addr: %" PRIu64 "\n",
		 implicit_fi_addr, *fi_addr);

	/* Call foreach_unspec_addr to move unexpected messages
	 * from the unspecified queue to the specified queues
	 *
	 * util_ep is bound to the explicit util_av, so the explicit util_av's
	 * ep_list contains all of the endpoints bound to this AV */
	ofi_genlock_lock(&av->util_av.ep_list_lock);
	dlist_foreach(&av->util_av.ep_list, entry) {
		ep = container_of(entry, struct efa_rdm_ep, base_ep.util_ep.av_entry);
		/* move from implicit to explicit peer map, using new fi_addr */
		EFA_GENLOCK_LOCK(&ep->ctrl_lock, efa_ctrl_lock_sym);
		peer = efa_rdm_ep_peer_map_remove(ep->fi_addr_to_peer_map_implicit,
					       implicit_fi_addr);
		if (peer) {
			peer->av_entry = explicit_av_entry;
			if (efa_rdm_ep_peer_map_insert(ep->fi_addr_to_peer_map,
						       *fi_addr, peer))
				EFA_WARN(FI_LOG_AV,
					 "Failed to insert peer into explicit map for addr %lu\n",
					 *fi_addr);
		}
		EFA_GENLOCK_UNLOCK(&ep->ctrl_lock, efa_ctrl_lock_sym);
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		peer_srx->owner_ops->foreach_unspec_addr(peer_srx, &efa_rdm_av_get_addr_from_peer_rx_entry);
	}
	ofi_genlock_unlock(&av->util_av.ep_list_lock);

	return FI_SUCCESS;
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

	*fi_addr = av_entry->efa_av_entry.fi_addr;
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
 * If the address already exists in the explicit AV, return the existing
 * explicit fi_addr (no implicit insertion needed). If it already exists in
 * the implicit AV, update its LRU position. Otherwise allocate a new
 * connection entry in the implicit AV.
 */
int efa_rdm_av_insert_one_implicit(struct efa_av *av, struct efa_ep_addr *addr,
				   fi_addr_t *fi_addr, uint64_t flags,
				   void *context)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct efa_rdm_av *rdm_av = ((struct efa_rdm_av *)(av));
	char raw_gid_str[INET6_ADDRSTRLEN];
	struct efa_rdm_av_entry *av_entry;
	fi_addr_t implicit_fi_addr;
	fi_addr_t efa_fiaddr;
	int ret;

	ret = efa_av_insert_one_validate(addr, fi_addr, raw_gid_str);
	if (ret)
		return ret;

	EFA_INFO(FI_LOG_AV,
		 "Inserting address GID[%s] QP[%u] QKEY[%u] to implicit AV\n",
		 raw_gid_str, addr->qpn, addr->qkey);

	/* Check if this address already exists in the explicit AV */
	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	efa_fiaddr = ofi_av_lookup_fi_addr_unsafe(&av->util_av, addr);
	if (efa_fiaddr != FI_ADDR_NOTAVAIL) {
		*fi_addr = efa_fiaddr;
		EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);
		EFA_INFO(FI_LOG_AV,
			 "Found existing AV entry pointing to this "
			 "address! fi_addr: %" PRId64 "\n",
			 efa_fiaddr);
		return 0;
	}
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	/* Check if address already exists in the implicit AV */
	EFA_GENLOCK_LOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	implicit_fi_addr =
		ofi_av_lookup_fi_addr_unsafe(&rdm_av->util_av_implicit, addr);
	if (implicit_fi_addr != FI_ADDR_NOTAVAIL) {
		EFA_INFO(FI_LOG_AV,
			 "Found implicit AV entry id %" PRId64
			 " for the same address\n",
			 implicit_fi_addr);

		/* Move to the end of the LRU list */
		av_entry = efa_rdm_av_addr_to_entry_implicit(av, implicit_fi_addr);
		efa_rdm_av_implicit_av_lru_move(av, av_entry);

		*fi_addr = implicit_fi_addr;
		EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
		return 0;
	}

	/* Address not found in either AV, allocate a new implicit entry */
	av_entry = efa_rdm_av_entry_alloc_implicit(av, addr, flags, context);
	if (!av_entry) {
		*fi_addr = FI_ADDR_NOTAVAIL;
		EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
		return -FI_EADDRNOTAVAIL;
	}

	*fi_addr = av_entry->implicit_fi_addr;
	EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);

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
	struct efa_av_entry *entry;

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
		entry = efa_av_addr_to_entry(av, fi_addr[i]);
		if (!entry) {
			err = -FI_EINVAL;
			break;
		}

		efa_rdm_av_entry_release_explicit(av, container_of(entry, struct efa_rdm_av_entry, efa_av_entry));
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


static int efa_rdm_av_close(struct fid *fid)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_av *av;
	struct efa_rdm_av *rdm_av;
	struct efa_cur_reverse_av *cur_entry, *curtmp;
	struct efa_prv_reverse_av *prv_entry, *prvtmp;
	struct efa_ep_addr_hashable *ep_addr_hashable, *tmp;
	int err = 0;

	av = container_of(fid, struct efa_av, util_av.av_fid.fid);
	rdm_av = ((struct efa_rdm_av *)(av));

	/* The order in which the util domain and av locks are acquired must be
	 * util_domain.lock -> util_av.lock -> util_av_implicit.lock
	 * in the AV insertion, removal and CQ read paths to prevent deadlocks */
	EFA_GENLOCK_LOCK(&av->domain->util_domain.lock, efa_util_domain_lock_sym);

	EFA_GENLOCK_LOCK(&av->util_av.lock, efa_util_av_lock_sym);
	HASH_ITER(hh, av->cur_reverse_av, cur_entry, curtmp) {
		efa_rdm_av_entry_release_explicit(av, container_of(cur_entry->entry, struct efa_rdm_av_entry, efa_av_entry));
	}
	HASH_ITER(hh, rdm_av->prv_reverse_av, prv_entry, prvtmp) {
		efa_rdm_av_entry_release_explicit(av, container_of(prv_entry->entry, struct efa_rdm_av_entry, efa_av_entry));
	}
	EFA_GENLOCK_UNLOCK(&av->util_av.lock, efa_util_av_lock_sym);

	EFA_GENLOCK_LOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);
	HASH_ITER(hh, rdm_av->cur_reverse_av_implicit, cur_entry, curtmp) {
		efa_rdm_av_entry_release_implicit(av, container_of(cur_entry->entry, struct efa_rdm_av_entry, efa_av_entry));
	}
	HASH_ITER(hh, rdm_av->prv_reverse_av_implicit, prv_entry, prvtmp) {
		efa_rdm_av_entry_release_implicit(av, container_of(prv_entry->entry, struct efa_rdm_av_entry, efa_av_entry));
	}
	EFA_GENLOCK_UNLOCK(&rdm_av->util_av_implicit.lock, efa_implicit_av_lock_sym);

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

	ret = efa_av_init_base(av, efa_domain, attr, context,
			       sizeof(struct efa_rdm_av_entry) - EFA_EP_ADDR_LEN);
	if (ret)
		goto err_free;

	ret = efa_av_array_init(&rdm_av->addr_to_entry_map_implicit);
	if (ret)
		goto err_destruct_base;

	ret = efa_av_init_util_av(efa_domain, attr, &rdm_av->util_av_implicit, context,
				  sizeof(struct efa_rdm_av_entry) - EFA_EP_ADDR_LEN);
	if (ret)
		goto err_destroy_implicit_map;

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
			goto err_close_util_av_implicit;
		}
		av_attr.count = efa_env.shm_av_size;
		assert(av_attr.type == FI_AV_TABLE);
		ret = fi_av_open(rdm_domain->shm_domain, &av_attr,
				 &rdm_av->shm_rdm_av, context);
		if (ret)
			goto err_close_util_av_implicit;
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

err_close_util_av_implicit:
	retv = ofi_av_close(&rdm_av->util_av_implicit);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av_implicit: %s\n", fi_strerror(-retv));

err_destroy_implicit_map:
	efa_av_array_destroy(rdm_av->addr_to_entry_map_implicit);

err_destruct_base:
	retv = ofi_av_close(&av->util_av);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av: %s\n", fi_strerror(-retv));
	efa_av_array_destroy(av->addr_to_entry_map);

err_free:
	free(rdm_av);
	return ret;
}
