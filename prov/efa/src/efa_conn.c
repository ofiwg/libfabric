/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <infiniband/efadv.h>

#include "efa.h"
#include "efa_conn.h"
#include "rdm/efa_rdm_ep.h"

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
 * @brief remove an efa_av_entry from an addr map + util AV and clear its address
 *
 * Shared by the base explicit release path and the RDM implicit release paths
 * (each supplies its own addr map and util AV).
 */
void efa_av_entry_remove_from_util_av(struct efa_av_array *entry_map,
				      struct util_av *util_av,
				      struct efa_av_entry *entry,
				      fi_addr_t fi_addr)
{
	struct efa_ep_addr *ep_addr = efa_av_entry_ep_addr(entry);
	char gidstr[INET6_ADDRSTRLEN];
	int err;

	err = efa_av_array_insert(entry_map, fi_addr, NULL);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to remove entry for fi_addr %" PRIu64
			 " from array: %s\n", fi_addr, fi_strerror(-err));
	}

	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_remove_addr failed for fi_addr %" PRIu64
			 ": %s\n", fi_addr, fi_strerror(-err));
	}

	inet_ntop(AF_INET6, ep_addr->raw, gidstr, INET6_ADDRSTRLEN);
	EFA_INFO(FI_LOG_AV, "efa_av_entry released! entry[%p] GID[%s] QP[%u]\n",
		 entry, gidstr, ep_addr->qpn);

	memset(entry->ep_addr, 0, EFA_EP_ADDR_LEN);
}

/**
 * @brief create a base explicit AV entry: insert into the explicit util AV,
 * set the base fields (fi_addr, base AH) and register in the addr map.
 * Reverse-AV indexing and RDM-only state are layered on by the caller.
 * Caller must hold util_domain.lock and util_av.lock.
 */
struct efa_av_entry *efa_av_entry_alloc_explicit(struct efa_av *av,
						 struct efa_ep_addr *raw_addr,
						 fi_addr_t *fi_addr_out)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct util_av *util_av = &av->util_av;
	struct util_av_entry *util_av_entry;
	struct efa_av_entry *efa_av_entry;
	fi_addr_t fi_addr;
	int err;

	err = ofi_av_insert_addr(util_av, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n", fi_strerror(-err));
		return NULL;
	}

	util_av_entry = ofi_bufpool_get_ibuf(util_av->av_entry_pool, fi_addr);
	efa_av_entry = (struct efa_av_entry *)util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, efa_av_entry_ep_addr(efa_av_entry)));
	assert(av->type == FI_AV_TABLE);
	efa_av_entry->fi_addr = fi_addr;

	efa_av_entry->ah = efa_ah_alloc(av->domain, raw_addr->raw, false, sizeof(struct efa_ah));
	if (!efa_av_entry->ah)
		goto err_remove_addr;

	err = efa_av_array_insert(av->addr_to_entry_map, fi_addr, efa_av_entry);
	if (err) {
		EFA_WARN(FI_LOG_AV, "Failed to insert entry for fi_addr %" PRIu64
			" into array: %s\n", fi_addr, fi_strerror(-err));
		goto err_release_ah;
	}

	*fi_addr_out = fi_addr;
	return efa_av_entry;

err_release_ah:
	efa_ah_release(av->domain, efa_av_entry->ah, false);
err_remove_addr:
	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed for fi_addr %" PRIu64
			": %s\n", fi_addr, fi_strerror(-err));
	return NULL;
}

/**
 * @brief release the base resources of an explicit AV entry
 *
 * Release the entry's base AH and remove it from the explicit addr map and
 * util AV (clearing its raw address). Reverse-AV removal and RDM-only teardown
 * are handled by the caller before calling this. Caller must hold
 * util_domain.lock and util_av.lock.
 */
void efa_av_entry_release_explicit(struct efa_av *av, struct efa_av_entry *entry,
				   fi_addr_t fi_addr)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	efa_av_reverse_av_remove(&av->cur_reverse_av, entry);
	efa_ah_release(av->domain, entry->ah, false);
	efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
					 entry, fi_addr);
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

	entry->ah = efa_ah_alloc(av->domain, raw_addr->raw, false, sizeof(struct efa_ah));
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
		efa_ah_release(av->domain, entry->ah, false);
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
		efa_ah_release(av->domain, entry->ah, false);
		efa_av_entry_remove_from_util_av(av->addr_to_entry_map, &av->util_av,
						 entry, fi_addr);
		return NULL;
	}

	return av_entry;

err_release_ah:
	efa_ah_release(av->domain, entry->ah, false);
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

	av_entry->efa_av_entry.ah = efa_ah_alloc(av->domain, raw_addr->raw, true, sizeof(struct efa_ah));
	if (!av_entry->efa_av_entry.ah)
		goto err_release;

	dlist_insert_tail(&av_entry->ah_implicit_conn_list_entry,
			  &av_entry->efa_av_entry.ah->implicit_conn_list);

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
		efa_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
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
	efa_ah_release(av->domain, av_entry->efa_av_entry.ah, false);
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
	efa_ah_release(av->domain, av_entry->efa_av_entry.ah, true);
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
	av_entry->efa_av_entry.ah->implicit_refcnt--;
}
