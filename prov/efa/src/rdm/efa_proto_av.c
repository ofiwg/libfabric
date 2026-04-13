/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <infiniband/efadv.h>
#include <ofi_enosys.h>

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_proto_av.h"
#include "rdm/efa_rdm_pke_utils.h"

/**
 * @brief Local/remote peer detection by comparing peer GID with stored local GIDs
 *
 * @param[in]	av	efa AV
 * @param[in]	addr	peer address to check
 * @return	true if local, false otherwise
 */
static bool efa_is_local_peer(struct efa_av *av, const void *addr)
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

/* ---- Address lookup ---- */

/**
 * @brief find proto AV entry using fi_addr in the given util_av
 *
 * @param[in]	util_av	util AV to search
 * @param[in]	fi_addr	fabric address to look up
 * @return	pointer to entry if valid, NULL otherwise
 */
static inline struct efa_proto_av_entry *
efa_proto_av_addr_to_entry_impl(struct util_av *util_av, fi_addr_t fi_addr)
{
	struct util_av_entry *util_av_entry;
	struct efa_proto_av_entry *entry;

	if (OFI_UNLIKELY(fi_addr == FI_ADDR_UNSPEC || fi_addr == FI_ADDR_NOTAVAIL))
		return NULL;

	if (OFI_LIKELY(ofi_bufpool_ibuf_is_valid(util_av->av_entry_pool, fi_addr)))
		util_av_entry = ofi_bufpool_get_ibuf(util_av->av_entry_pool, fi_addr);
	else
		return NULL;

	entry = (struct efa_proto_av_entry *)util_av_entry->data;
	return entry->ah ? entry : NULL;
}

/**
 * @brief find proto AV entry using fi_addr in the explicit AV
 *
 * @param[in]	av	protocol AV
 * @param[in]	fi_addr	fabric address
 * @return	pointer to entry if valid, NULL otherwise
 */
struct efa_proto_av_entry *efa_proto_av_addr_to_entry(struct efa_proto_av *av,
						      fi_addr_t fi_addr)
{
	return efa_proto_av_addr_to_entry_impl(&av->efa_av.util_av, fi_addr);
}

/**
 * @brief find proto AV entry using fi_addr in the implicit AV
 *
 * @param[in]	av	protocol AV
 * @param[in]	fi_addr	fabric address
 * @return	pointer to entry if valid, NULL otherwise
 */
struct efa_proto_av_entry *efa_proto_av_addr_to_entry_implicit(
	struct efa_proto_av *av, fi_addr_t fi_addr)
{
	return efa_proto_av_addr_to_entry_impl(&av->util_av_implicit, fi_addr);
}

/* ---- Peer map operations ---- */

/**
 * @brief insert an entry into the peer map for a given AV entry
 *
 * @param[in]	entry		proto AV entry
 * @param[in]	map_entry	peer map entry to insert
 */
void efa_proto_av_entry_ep_peer_map_insert(
	struct efa_proto_av_entry *entry,
	struct efa_proto_av_entry_ep_peer_map_entry *map_entry)
{
	HASH_ADD_PTR(entry->ep_peer_map, ep_ptr, map_entry);
}

/**
 * @brief look up a peer in the peer map for a given AV entry and endpoint
 *
 * @param[in]	entry	proto AV entry
 * @param[in]	ep	RDM endpoint
 * @return	pointer to peer if found, NULL otherwise
 */
struct efa_rdm_peer *efa_proto_av_entry_ep_peer_map_lookup(
	struct efa_proto_av_entry *entry, struct efa_rdm_ep *ep)
{
	struct efa_proto_av_entry_ep_peer_map_entry *map_entry;

	HASH_FIND_PTR(entry->ep_peer_map, &ep, map_entry);
	return map_entry ? &map_entry->peer : NULL;
}

/**
 * @brief remove an endpoint's peer from the peer map for a given AV entry
 *
 * @param[in]	entry	proto AV entry
 * @param[in]	ep	RDM endpoint whose peer to remove
 */
void efa_proto_av_entry_ep_peer_map_remove(
	struct efa_proto_av_entry *entry, struct efa_rdm_ep *ep)
{
	struct efa_proto_av_entry_ep_peer_map_entry *map_entry;

	HASH_FIND_PTR(entry->ep_peer_map, &ep, map_entry);
	assert(map_entry);
	HASH_DELETE(hh, entry->ep_peer_map, map_entry);
	ofi_buf_free(map_entry);
}

/* ---- SHM AV operations ---- */

/**
 * @brief Insert the address into SHM provider's AV
 *
 * If shm transfer is enabled and the addr comes from local peer,
 *  1. convert addr to format 'gid_qpn', which will be set as shm's ep name later.
 *  2. insert gid_qpn into shm's av
 *  3. store returned fi_addr from shm into the hash table
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry
 * @return	On success return 0, otherwise return a negative error code
 */
int efa_proto_av_entry_insert_shm_av(struct efa_proto_av *av,
					  struct efa_proto_av_entry *entry)
{
	int err, ret;
	char smr_name[EFA_SHM_NAME_MAX];
	size_t smr_name_len;
	struct efa_ep_addr *ep_addr = efa_proto_av_entry_ep_addr(entry);

	assert(ep_addr);

	if (efa_is_local_peer(&av->efa_av, ep_addr) && av->shm_rdm_av) {
		if (av->shm_used >= efa_env.shm_av_size) {
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

		/*
		 * The shm provider supports FI_AV_USER_ID flag. This flag
		 * associates a user-assigned identifier with each av entry that
		 * is returned with any completion entry in place of the AV's
		 * address. Below, &entry->shm_fi_addr is both input and output.
		 * It is passed in with value entry->fi_addr (the efa provider's
		 * fi_addr). shm records this as user id for cq write, then
		 * overwrites shm_fi_addr with the actual fi_addr in shm's av.
		 * The efa provider uses shm_fi_addr for transmissions through
		 * the shm ep.
		 */
		entry->shm_fi_addr = entry->fi_addr;
		ret = fi_av_insert(av->shm_rdm_av, smr_name, 1, &entry->shm_fi_addr, FI_AV_USER_ID, NULL);
		if (OFI_UNLIKELY(ret != 1)) {
			EFA_WARN(FI_LOG_AV,
				 "Failed to insert address to shm provider's av: %s\n",
				 fi_strerror(-ret));
			return ret;
		}

		EFA_INFO(FI_LOG_AV,
			"Successfully inserted %s to shm provider's av. efa_fiaddr: %ld shm_fiaddr = %ld\n",
			smr_name, entry->fi_addr, entry->shm_fi_addr);

		assert(entry->shm_fi_addr < efa_env.shm_av_size);
		av->shm_used++;
	}

	return 0;
}

/**
 * @brief Release the protocol-specific resources of an AV entry.
 *
 * Releases the shm av entry and destroys rdm peers. Caller must hold
 * the SRX lock because this function modifies the peer map and destroys
 * peers which are accessed and modified in the CQ read path.
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry
 */
void efa_proto_av_entry_deinit(struct efa_proto_av *av,
				   struct efa_proto_av_entry *entry)
{
	int err;
	struct efa_proto_av_entry_ep_peer_map_entry *peer_map_entry, *tmp;

	assert((entry->fi_addr != FI_ADDR_NOTAVAIL &&
		entry->implicit_fi_addr == FI_ADDR_NOTAVAIL) ||
	       (entry->implicit_fi_addr != FI_ADDR_NOTAVAIL &&
		entry->fi_addr == FI_ADDR_NOTAVAIL));

	if (entry->shm_fi_addr != FI_ADDR_NOTAVAIL && av->shm_rdm_av) {
		err = fi_av_remove(av->shm_rdm_av, &entry->shm_fi_addr, 1, 0);
		if (err) {
			EFA_WARN(FI_LOG_AV,
				 "remove address from shm av failed! err=%d\n",
				 err);
		} else {
			av->shm_used--;
			assert(entry->shm_fi_addr < efa_env.shm_av_size);
		}
	}

	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));
	HASH_ITER(hh, entry->ep_peer_map, peer_map_entry, tmp) {
		dlist_remove(&peer_map_entry->peer.ep_peer_list_entry);
		efa_rdm_peer_destruct(&peer_map_entry->peer, peer_map_entry->ep_ptr);
		HASH_DEL(entry->ep_peer_map, peer_map_entry);
		ofi_buf_free(peer_map_entry);
	}
	assert(HASH_CNT(hh, entry->ep_peer_map) == 0);
}

/* ---- Implicit AV LRU ---- */

/**
 * @brief Add entry to the LRU list. If the list is full, evict the least
 * recently used entry at the front of the LRU list and add the latest one.
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry to be added to the LRU list
 */
static inline int efa_proto_av_implicit_av_lru_insert(struct efa_proto_av *av,
						      struct efa_proto_av_entry *entry)
{
	size_t cur_size;
	struct efa_ep_addr_hashable *ep_addr_hashable;
	struct efa_proto_av_entry *entry_to_release;

	if (av->implicit_av_size == 0)
		goto out;

	cur_size = HASH_CNT(hh, av->util_av_implicit.hash);
	if (cur_size <= av->implicit_av_size)
		goto out;

	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));

	dlist_pop_front(&av->implicit_av_lru_list, struct efa_proto_av_entry,
			entry_to_release, implicit_av_lru_entry);
	EFA_INFO(FI_LOG_AV,
		 "Evicting AV entry for peer implicit fi_addr %" PRIu64
		 " AHN %" PRIu16 " QPN %" PRIu16 " QKEY %" PRIu32 " from "
		 "implicit AV\n",
		 entry_to_release->implicit_fi_addr, entry_to_release->ah->ahn,
		 efa_proto_av_entry_ep_addr(entry_to_release)->qpn,
		 efa_proto_av_entry_ep_addr(entry_to_release)->qkey);

	ep_addr_hashable = malloc(sizeof(struct efa_ep_addr_hashable));
	if (!ep_addr_hashable) {
		EFA_WARN(FI_LOG_AV, "Could not allocate memory for LRU AV entry hashset entry\n");
		return FI_ENOMEM;
	}
	memcpy(ep_addr_hashable, entry->ep_addr, sizeof(struct efa_ep_addr));
	HASH_ADD(hh, av->evicted_peers_hashset, addr, sizeof(struct efa_ep_addr), ep_addr_hashable);

	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));
	efa_proto_av_entry_release(av, entry_to_release, true);

	assert(HASH_CNT(hh, av->util_av_implicit.hash) == av->implicit_av_size);

out:
	dlist_insert_tail(&entry->implicit_av_lru_entry,
			  &av->implicit_av_lru_list);
	return FI_SUCCESS;
}

/**
 * @brief Move entry to the end of the LRU list (most recently used)
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry to move
 */
void efa_proto_av_implicit_av_lru_entry_move(struct efa_proto_av *av,
					     struct efa_proto_av_entry *entry)
{
	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));
	assert(av->implicit_av_size == 0 ||
	       HASH_CNT(hh, av->util_av_implicit.hash) <= av->implicit_av_size);
	assert(dlist_entry_in_list(&av->implicit_av_lru_list,
				   &entry->implicit_av_lru_entry));

	dlist_remove(&entry->implicit_av_lru_entry);
	dlist_insert_tail(&entry->implicit_av_lru_entry,
			  &av->implicit_av_lru_list);

	efa_ah_implicit_av_lru_ah_move(av->efa_av.domain, entry->ah);
}

/* ---- Reverse lookup (protocol, connid-aware) ---- */

/**
 * @brief reverse lookup a proto AV entry by AHN, QPN, and optional connid
 *
 * @param[in]	cur_reverse_av	current reverse AV hash table
 * @param[in]	prv_reverse_av	previous reverse AV hash table
 * @param[in]	ahn		address handle number
 * @param[in]	qpn		QP number
 * @param[in]	pkt_entry	NULL or packet entry to extract connid from
 * @return	pointer to entry if found, NULL otherwise
 */
static inline struct efa_proto_av_entry *
efa_proto_av_reverse_lookup_entry(struct efa_cur_reverse_av **cur_reverse_av,
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

	HASH_FIND(hh, *cur_reverse_av, &cur_key, sizeof(cur_key), cur_entry);

	if (OFI_UNLIKELY(!cur_entry))
		return NULL;

	/*
	 * Cast is safe: in protocol path, av_entry points to the ep_addr field
	 * of a efa_proto_av_entry which has the same layout prefix.
	 */
	if (!pkt_entry ||
	    (pkt_entry->alloc_type == EFA_RDM_PKE_FROM_USER_RX_POOL)) {
		return (struct efa_proto_av_entry *)cur_entry->av_entry;
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
		return (struct efa_proto_av_entry *)cur_entry->av_entry;
	}

	if (OFI_LIKELY(*connid == efa_av_entry_ep_addr(cur_entry->av_entry)->qkey))
		return (struct efa_proto_av_entry *)cur_entry->av_entry;

	prv_key.ahn = ahn;
	prv_key.qpn = qpn;
	prv_key.connid = *connid;
	HASH_FIND(hh, *prv_reverse_av, &prv_key, sizeof(prv_key), prv_entry);

	return OFI_LIKELY(!!prv_entry) ? (struct efa_proto_av_entry *)prv_entry->av_entry : NULL;
}

/**
 * @brief find fi_addr for RDM endpoint in the explicit AV (connid-aware)
 *
 * @param[in]	av		protocol AV
 * @param[in]	ahn		address handle number
 * @param[in]	qpn		QP number
 * @param[in]	pkt_entry	NULL or RDM packet entry, used to extract connid
 * @return	fi_addr on success, FI_ADDR_NOTAVAIL if not found
 */
fi_addr_t efa_proto_av_reverse_lookup(struct efa_proto_av *av,
					  uint16_t ahn, uint16_t qpn,
					  struct efa_rdm_pke *pkt_entry)
{
	struct efa_proto_av_entry *entry;

	entry = efa_proto_av_reverse_lookup_entry(
		&av->efa_av.cur_reverse_av, &av->efa_av.prv_reverse_av,
		ahn, qpn, pkt_entry);

	if (OFI_LIKELY(!!entry))
		return entry->fi_addr;

	return FI_ADDR_NOTAVAIL;
}

/**
 * @brief find fi_addr for RDM endpoint in the implicit AV (connid-aware)
 *
 * Caller must hold srx_lock. Updates LRU list on hit.
 *
 * @param[in]	av		protocol AV
 * @param[in]	ahn		address handle number
 * @param[in]	qpn		QP number
 * @param[in]	pkt_entry	NULL or RDM packet entry, used to extract connid
 * @return	implicit fi_addr on success, FI_ADDR_NOTAVAIL if not found
 */
fi_addr_t efa_proto_av_reverse_lookup_implicit(struct efa_proto_av *av,
						   uint16_t ahn, uint16_t qpn,
						   struct efa_rdm_pke *pkt_entry)
{
	struct efa_proto_av_entry *entry;

	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));

	entry = efa_proto_av_reverse_lookup_entry(
		&av->cur_reverse_av_implicit, &av->prv_reverse_av_implicit,
		ahn, qpn, pkt_entry);

	if (OFI_LIKELY(!!entry)) {
		efa_proto_av_implicit_av_lru_entry_move(av, entry);
		return entry->implicit_fi_addr;
	}

	return FI_ADDR_NOTAVAIL;
}

/* ---- Entry release helpers ---- */

/**
 * @brief remove entry from the appropriate reverse AV hash tables
 *
 * @param[in]	av	protocol AV
 * @param[in]	entry	entry to remove
 * @param[in]	release_from_implicit_av	whether entry is in implicit AV
 */
static void efa_proto_av_entry_release_reverse_av(struct efa_proto_av *av,
						  struct efa_proto_av_entry *entry,
						  bool release_from_implicit_av)
{
	if (release_from_implicit_av) {
		assert(ofi_genlock_held(&av->util_av_implicit.lock));
		efa_av_reverse_av_remove(&av->cur_reverse_av_implicit,
					 &av->prv_reverse_av_implicit,
					 (struct efa_av_entry *)entry);
	} else {
		assert(ofi_genlock_held(&av->efa_av.util_av.lock));
		efa_av_reverse_av_remove(&av->efa_av.cur_reverse_av,
					 &av->efa_av.prv_reverse_av,
					 (struct efa_av_entry *)entry);
	}
}

/**
 * @brief remove entry from the appropriate util_av and clear its fields
 *
 * @param[in]	av	protocol AV
 * @param[in]	entry	entry to remove
 * @param[in]	release_from_implicit_av	whether entry is in implicit AV
 */
static void efa_proto_av_entry_release_util_av(struct efa_proto_av *av,
					       struct efa_proto_av_entry *entry,
					       bool release_from_implicit_av)
{
	struct util_av *util_av;
	char gidstr[INET6_ADDRSTRLEN];
	fi_addr_t fi_addr;
	int err;

	if (release_from_implicit_av) {
		assert(ofi_genlock_held(&av->util_av_implicit.lock));
		util_av = &av->util_av_implicit;
		fi_addr = entry->implicit_fi_addr;
	} else {
		assert(ofi_genlock_held(&av->efa_av.util_av.lock));
		util_av = &av->efa_av.util_av;
		fi_addr = entry->fi_addr;
	}

	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "ofi_av_remove_addr failed! err=%d\n", err);

	inet_ntop(AF_INET6, efa_proto_av_entry_ep_addr(entry)->raw, gidstr, INET6_ADDRSTRLEN);
	EFA_INFO(FI_LOG_AV, "efa_proto_av_entry released! entry[%p] GID[%s] QP[%u]\n",
		 entry, gidstr, efa_proto_av_entry_ep_addr(entry)->qpn);

	entry->ah = NULL;
	memset(entry->ep_addr, 0, EFA_EP_ADDR_LEN);
}

/**
 * @brief Release a proto AV entry.
 *
 * Caller must hold srx_lock. Acquires util_domain.lock internally
 * via efa_ah_release. Called from the AV removal path.
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry to release
 * @param[in]	release_from_implicit_av	whether entry is in implicit AV
 */
void efa_proto_av_entry_release(struct efa_proto_av *av,
				struct efa_proto_av_entry *entry,
				bool release_from_implicit_av)
{
	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));

	efa_proto_av_entry_release_reverse_av(av, entry, release_from_implicit_av);
	efa_proto_av_entry_deinit(av, entry);

	if (release_from_implicit_av)
		dlist_remove(&entry->ah_implicit_conn_list_entry);

	efa_ah_release(av->efa_av.domain, entry->ah, release_from_implicit_av);
	efa_proto_av_entry_release_util_av(av, entry, release_from_implicit_av);

	release_from_implicit_av ? av->used_implicit-- : av->efa_av.used--;
}

/**
 * @brief Release a proto AV entry without acquiring util_domain.lock.
 *
 * Caller must hold srx_lock AND util_domain.lock. Called from the AH
 * eviction path in the CQ read path which already holds both locks.
 *
 * @param[in]	av	protocol address vector
 * @param[in]	entry	proto av entry to release
 * @param[in]	release_from_implicit_av	whether entry is in implicit AV
 */
void efa_proto_av_entry_release_ah_unsafe(struct efa_proto_av *av,
					  struct efa_proto_av_entry *entry,
					  bool release_from_implicit_av)
{
	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));
	assert(ofi_genlock_held(&av->efa_av.domain->util_domain.lock));

	efa_proto_av_entry_release_reverse_av(av, entry, release_from_implicit_av);
	efa_proto_av_entry_deinit(av, entry);

	if (release_from_implicit_av)
		dlist_remove(&entry->ah_implicit_conn_list_entry);

	/* Decrement refcnts before release_util_av which NULLs entry->ah */
	release_from_implicit_av ? entry->ah->implicit_refcnt-- :
				   entry->ah->explicit_refcnt--;

	efa_proto_av_entry_release_util_av(av, entry, release_from_implicit_av);

	release_from_implicit_av ? av->used_implicit-- : av->efa_av.used--;
}

/* ---- AH alloc with eviction ---- */

/**
 * @brief Evict the least recently used AH that has no explicit AV entries.
 *
 * Finds the LRU AH with only implicit references, releases all its
 * implicit AV entries, and destroys the AH. Called when ibv_create_ah
 * fails with ENOMEM.
 *
 * Caller must hold srx_lock. This function acquires util_domain.lock.
 *
 * @param[in]	domain	efa domain
 * @return	0 on success, -FI_ENOMEM if no AH is available to evict
 */
static int efa_proto_ah_evict(struct efa_domain *domain)
{
	struct efa_proto_av_entry *entry_to_release;
	struct efa_ah *ah_tmp, *ah_to_release = NULL;
	struct dlist_entry *tmp;

	assert(ofi_genlock_held(&domain->srx_lock));

	ofi_genlock_lock(&domain->util_domain.lock);

	dlist_foreach_container(&domain->ah_lru_list, struct efa_ah, ah_tmp,
				domain_lru_ah_list_entry) {
		if (ah_tmp->explicit_refcnt == 0) {
			ah_to_release = ah_tmp;
			break;
		}
	}

	if (!ah_to_release) {
		ofi_genlock_unlock(&domain->util_domain.lock);
		EFA_WARN(FI_LOG_AV,
			 "AH creation for implicit AV entry failed with ENOMEM "
			 "but no AH entries available to evict\n");
		return -FI_ENOMEM;
	}

	assert(ah_to_release->implicit_refcnt > 0);

	dlist_foreach_container_safe(&ah_to_release->implicit_conn_list,
				     struct efa_proto_av_entry, entry_to_release,
				     ah_implicit_conn_list_entry, tmp) {

		assert(entry_to_release->implicit_fi_addr != FI_ADDR_NOTAVAIL &&
		       entry_to_release->fi_addr == FI_ADDR_NOTAVAIL);

		efa_proto_av_entry_release_ah_unsafe(entry_to_release->av,
						     entry_to_release, true);
	}

	if (ah_to_release->implicit_refcnt == 0 &&
	    ah_to_release->explicit_refcnt == 0) {
		efa_ah_destroy_ah(domain, ah_to_release);
	}

	ofi_genlock_unlock(&domain->util_domain.lock);

	return FI_SUCCESS;
}

/**
 * @brief Allocate an AH with eviction retry for protocol AV.
 *
 * Wraps efa_ah_alloc with ENOMEM handling: if ibv_create_ah fails due
 * to too many AH entries, evicts an AH with only implicit references
 * and retries.
 *
 * @param[in]	domain		efa domain
 * @param[in]	gid		GID
 * @param[in]	insert_implicit_av	whether this is for an implicit AV entry
 * @return	pointer to efa_ah on success, NULL on failure
 */
static struct efa_ah *efa_proto_ah_alloc(struct efa_domain *domain,
					 const uint8_t *gid,
					 bool insert_implicit_av)
{
	struct efa_ah *ah;
	int err;

	ah = efa_ah_alloc(domain, gid, insert_implicit_av);
	if (ah)
		return ah;

	if (errno != FI_ENOMEM)
		return NULL;

	EFA_INFO(FI_LOG_AV,
		 "ibv_create_ah failed with ENOMEM. "
		 "Attempting to evict AH entry\n");

	err = efa_proto_ah_evict(domain);
	if (err)
		return NULL;

	return efa_ah_alloc(domain, gid, insert_implicit_av);
}

/* ---- Entry alloc ---- */

/**
 * @brief Allocate and initialize a proto AV entry.
 *
 * Caller must hold util_av.lock (explicit) or util_av_implicit.lock (implicit).
 *
 * @param[in]	av		protocol address vector
 * @param[in]	raw_addr	raw efa address
 * @param[in]	flags		flags application passed to fi_av_insert
 * @param[in]	context		context application passed to fi_av_insert
 * @param[in]	insert_shm_av	whether to insert address into shm av
 * @param[in]	insert_implicit_av	whether to insert into implicit AV
 * @return	on success, return a pointer to the entry; otherwise NULL
 */
struct efa_proto_av_entry *efa_proto_av_entry_alloc(
	struct efa_proto_av *av, struct efa_ep_addr *raw_addr,
	uint64_t flags, void *context, bool insert_shm_av,
	bool insert_implicit_av)
{
	struct util_av *util_av;
	struct efa_cur_reverse_av **cur_reverse_av;
	struct efa_prv_reverse_av **prv_reverse_av;
	struct util_av_entry *util_av_entry = NULL;
	struct efa_proto_av_entry *entry;
	fi_addr_t fi_addr;
	int err;

	if (flags & FI_SYNC_ERR)
		memset(context, 0, sizeof(int));

	if (insert_implicit_av) {
		assert(ofi_genlock_held(&av->util_av_implicit.lock));
		util_av = &av->util_av_implicit;
		cur_reverse_av = &av->cur_reverse_av_implicit;
		prv_reverse_av = &av->prv_reverse_av_implicit;
	} else {
		assert(ofi_genlock_held(&av->efa_av.util_av.lock));
		util_av = &av->efa_av.util_av;
		cur_reverse_av = &av->efa_av.cur_reverse_av;
		prv_reverse_av = &av->efa_av.prv_reverse_av;
	}

	err = ofi_av_insert_addr(util_av, raw_addr, &fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV, "ofi_av_insert_addr failed! Error message: %s\n",
			 fi_strerror(err));
		return NULL;
	}

	util_av_entry = ofi_bufpool_get_ibuf(util_av->av_entry_pool, fi_addr);
	entry = (struct efa_proto_av_entry *)util_av_entry->data;
	assert(efa_is_same_addr(raw_addr, (struct efa_ep_addr *)entry->ep_addr));

	memset((char *)entry + EFA_EP_ADDR_LEN, 0,
	       sizeof(*entry) - EFA_EP_ADDR_LEN);
	assert(av->efa_av.type == FI_AV_TABLE);

	entry->av = av;

	if (insert_implicit_av) {
		entry->fi_addr = FI_ADDR_NOTAVAIL;
		entry->implicit_fi_addr = fi_addr;
		err = efa_proto_av_implicit_av_lru_insert(av, entry);
		if (err)
			return NULL;
	} else {
		entry->fi_addr = fi_addr;
		entry->implicit_fi_addr = FI_ADDR_NOTAVAIL;
	}

	entry->ah = efa_proto_ah_alloc(av->efa_av.domain, raw_addr->raw, insert_implicit_av);
	if (!entry->ah)
		goto err_release;

	if (insert_implicit_av)
		dlist_insert_tail(&entry->ah_implicit_conn_list_entry,
				  &entry->ah->implicit_conn_list);

	entry->shm_fi_addr = FI_ADDR_NOTAVAIL;

	/*
	 * This function is called in two situations:
	 * 1. application calls fi_av_insert API
	 * 2. efa progress engine gets a message from unknown peer through
	 *    efa device, meaning peer is not local or shm is disabled.
	 * For situation 1, shm av insertion should happen when peer is local
	 * (insert_shm_av=1). For situation 2, it shouldn't (insert_shm_av=0).
	 */
	if (insert_shm_av) {
		err = efa_proto_av_entry_insert_shm_av(av, entry);
		if (err) {
			errno = -err;
			goto err_release;
		}
	}

	err = efa_av_reverse_av_add(&av->efa_av, cur_reverse_av, prv_reverse_av,
				    (struct efa_av_entry *)entry);
	if (err) {
		if (insert_implicit_av)
			ofi_genlock_lock(&av->efa_av.domain->srx_lock);
		efa_proto_av_entry_deinit(av, entry);
		if (insert_implicit_av)
			ofi_genlock_unlock(&av->efa_av.domain->srx_lock);
		goto err_release;
	}

	insert_implicit_av ? av->used_implicit++ : av->efa_av.used++;

	return entry;

err_release:
	if (entry->ah)
		efa_ah_release(av->efa_av.domain, entry->ah, insert_implicit_av);

	entry->ah = NULL;
	memset(entry->ep_addr, 0, EFA_EP_ADDR_LEN);
	err = ofi_av_remove_addr(util_av, fi_addr);
	if (err)
		EFA_WARN(FI_LOG_AV, "While processing previous failure, ofi_av_remove_addr failed! err=%d\n",
			 err);

	return NULL;
}

/* ---- Implicit to explicit migration ---- */

/**
 * @brief get the fi_addr from a peer rx entry's packet context
 *
 * Used as a callback for foreach_unspec_addr during implicit-to-explicit
 * migration.
 *
 * @param[in]	rx_entry	peer rx entry
 * @return	fi_addr of the peer
 */
static fi_addr_t
efa_proto_av_get_addr_from_peer_rx_entry(struct fi_peer_rx_entry *rx_entry)
{
	struct efa_rdm_pke *pke;

	pke = (struct efa_rdm_pke *) rx_entry->peer_context;

	return pke->peer->av_entry->fi_addr;
}

/**
 * @brief migrate an implicit AV entry to the explicit AV
 *
 * Moves the entry, its peer map, AH, and SHM fi_addr from the implicit
 * AV to the explicit AV. Updates reverse AVs and notifies the SRX to
 * move unexpected messages from the unspecified queue.
 *
 * Caller must hold util_av.lock and util_av_implicit.lock.
 *
 * @param[in]	av		protocol AV
 * @param[in]	raw_addr	raw efa address
 * @param[in]	implicit_fi_addr	fi_addr in the implicit AV
 * @param[out]	fi_addr		fi_addr assigned in the explicit AV
 * @return	0 on success, negative error code on failure
 */
int efa_proto_av_entry_implicit_to_explicit(struct efa_proto_av *av,
					    struct efa_ep_addr *raw_addr,
					    fi_addr_t implicit_fi_addr,
					    fi_addr_t *fi_addr)
{
	int err;
	struct efa_ah *ah;
	struct efa_proto_av_entry *implicit_entry, *explicit_entry;
	struct efa_rdm_ep *ep;
	struct dlist_entry *list_entry;
	struct util_av_entry *implicit_util_av_entry, *explicit_util_av_entry;
	struct efa_proto_av_entry_ep_peer_map_entry *map_entry, *tmp;
	struct fid_peer_srx *peer_srx;

	EFA_INFO(FI_LOG_AV,
		 "Moving peer with implicit fi_addr %" PRIu64
		 " to explicit AV\n",
		 implicit_fi_addr);

	assert(ofi_genlock_held(&av->efa_av.util_av.lock));
	assert(ofi_genlock_held(&av->util_av_implicit.lock));

	implicit_util_av_entry =
		ofi_bufpool_get_ibuf(av->util_av_implicit.av_entry_pool, implicit_fi_addr);
	implicit_entry = (struct efa_proto_av_entry *) implicit_util_av_entry->data;

	assert(implicit_entry);
	assert(efa_is_same_addr(
		raw_addr, (struct efa_ep_addr *) implicit_entry->ep_addr));
	assert(implicit_entry->fi_addr == FI_ADDR_NOTAVAIL &&
	       implicit_entry->implicit_fi_addr == implicit_fi_addr);

	ah = implicit_entry->ah;

	/* Create explicit util AV entry */
	err = ofi_av_insert_addr(&av->efa_av.util_av, raw_addr, fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV,
			 "ofi_av_insert_addr into explicit AV failed! Error "
			 "message: %s\n",
			 fi_strerror(err));
		return err;
	}

	explicit_util_av_entry =
		ofi_bufpool_get_ibuf(av->efa_av.util_av.av_entry_pool, *fi_addr);
	explicit_entry = (struct efa_proto_av_entry *) explicit_util_av_entry->data;
	assert(efa_is_same_addr(
		raw_addr, (struct efa_ep_addr *) explicit_entry->ep_addr));

	/* Copy information from implicit to explicit */
	memset((char *)explicit_entry + EFA_EP_ADDR_LEN, 0,
	       sizeof(*explicit_entry) - EFA_EP_ADDR_LEN);
	assert(av->efa_av.type == FI_AV_TABLE);
	explicit_entry->av = av;
	explicit_entry->ah = implicit_entry->ah;
	explicit_entry->fi_addr = *fi_addr;
	explicit_entry->shm_fi_addr = implicit_entry->shm_fi_addr;
	explicit_entry->implicit_fi_addr = FI_ADDR_NOTAVAIL;
	HASH_ITER(hh, implicit_entry->ep_peer_map, map_entry, tmp) {
		HASH_DELETE(hh, implicit_entry->ep_peer_map, map_entry);
		HASH_ADD_PTR(explicit_entry->ep_peer_map, ep_ptr, map_entry);
		map_entry->peer.av_entry = explicit_entry;
	}
	assert(HASH_CNT(hh, implicit_entry->ep_peer_map) == 0);

	/* Handle reverse AV and AV ref counts */
	efa_av_reverse_av_remove(&av->cur_reverse_av_implicit,
				 &av->prv_reverse_av_implicit,
				 (struct efa_av_entry *)implicit_entry);

	dlist_remove(&implicit_entry->implicit_av_lru_entry);

	err = ofi_av_remove_addr(&av->util_av_implicit, implicit_fi_addr);
	if (err) {
		EFA_WARN(FI_LOG_AV,
			 "ofi_av_remove_addr from implicit AV failed! Error "
			 "message: %s\n",
			 fi_strerror(err));
		return err;
	}

	av->used_implicit--;

	err = efa_av_reverse_av_add(&av->efa_av, &av->efa_av.cur_reverse_av,
				    &av->efa_av.prv_reverse_av,
				    (struct efa_av_entry *)explicit_entry);
	if (err)
		return err;

	av->efa_av.used++;

	/* Handle AH LRU list and refcnt */
	assert(!dlist_empty(&ah->implicit_conn_list));
	dlist_remove(&implicit_entry->ah_implicit_conn_list_entry);
	efa_ah_implicit_av_lru_ah_move(av->efa_av.domain, ah);
	ah->implicit_refcnt--;
	ah->explicit_refcnt++;

	EFA_INFO(FI_LOG_AV,
		 "Peer with implicit fi_addr %" PRIu64
		 " moved to explicit AV. Explicit fi_addr: %" PRIu64 "\n",
		 implicit_fi_addr, *fi_addr);

	ofi_genlock_lock(&av->efa_av.util_av.ep_list_lock);
	dlist_foreach(&av->efa_av.util_av.ep_list, list_entry) {
		ep = container_of(list_entry, struct efa_rdm_ep, base_ep.util_ep.av_entry);
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		peer_srx->owner_ops->foreach_unspec_addr(peer_srx, &efa_proto_av_get_addr_from_peer_rx_entry);
	}
	ofi_genlock_unlock(&av->efa_av.util_av.ep_list_lock);

	return FI_SUCCESS;
}

/* ---- Protocol AV insert_one ---- */

/**
 * @brief insert one address into the protocol AV
 *
 * Checks explicit and implicit AVs for duplicates. Handles
 * implicit-to-explicit migration when an implicit entry exists.
 *
 * Caller must hold srx_lock.
 *
 * @param[in]	av		protocol AV
 * @param[in]	addr		raw address (gid:qpn:qkey)
 * @param[out]	fi_addr		output fi_addr
 * @param[in]	flags		flags from fi_av_insert
 * @param[in]	context		context from fi_av_insert
 * @param[in]	insert_shm_av	whether to insert into SHM AV
 * @param[in]	insert_implicit_av	whether to insert into implicit AV
 * @return	0 on success, negative error code on failure
 */
int efa_proto_av_insert_one(struct efa_proto_av *av, struct efa_ep_addr *addr,
			    fi_addr_t *fi_addr, uint64_t flags, void *context,
			    bool insert_shm_av, bool insert_implicit_av)
{
	struct efa_proto_av_entry *entry;
	char raw_gid_str[INET6_ADDRSTRLEN];
	fi_addr_t efa_fiaddr;
	fi_addr_t implicit_fi_addr;
	int ret = 0;

	if (!efa_av_is_valid_address(addr)) {
		EFA_WARN(FI_LOG_AV, "Failed to insert bad addr\n");
		*fi_addr = FI_ADDR_NOTAVAIL;
		return -FI_EADDRNOTAVAIL;
	}

	assert(ofi_genlock_held(&av->efa_av.domain->srx_lock));
	ofi_genlock_lock(&av->util_av_implicit.lock);
	ofi_genlock_lock(&av->efa_av.util_av.lock);

	memset(raw_gid_str, 0, sizeof(raw_gid_str));
	if (!inet_ntop(AF_INET6, addr->raw, raw_gid_str, INET6_ADDRSTRLEN)) {
		EFA_WARN(FI_LOG_AV, "cannot convert address to string. errno: %d\n", errno);
		ret = -FI_EINVAL;
		*fi_addr = FI_ADDR_NOTAVAIL;
		goto out;
	}

	EFA_INFO(FI_LOG_AV,
		 "Inserting address GID[%s] QP[%u] QKEY[%u] to %s AV ....\n",
		 raw_gid_str, addr->qpn, addr->qkey,
		 insert_implicit_av ? "implicit" : "explicit");

	/* Check explicit AV */
	efa_fiaddr = ofi_av_lookup_fi_addr_unsafe(&av->efa_av.util_av, addr);
	if (efa_fiaddr != FI_ADDR_NOTAVAIL) {
		assert(!insert_implicit_av);
		EFA_INFO(FI_LOG_AV, "Found existing AV entry pointing to this address! fi_addr: %ld\n", efa_fiaddr);
		*fi_addr = efa_fiaddr;
		ret = 0;
		goto out;
	}

	/* Check implicit AV */
	implicit_fi_addr =
		ofi_av_lookup_fi_addr_unsafe(&av->util_av_implicit, addr);
	if (implicit_fi_addr != FI_ADDR_NOTAVAIL) {
		EFA_INFO(FI_LOG_AV,
			 "Found implicit AV entry id %ld for the same address\n",
			 implicit_fi_addr);

		if (insert_implicit_av) {
			entry = efa_proto_av_addr_to_entry_implicit(av, implicit_fi_addr);
			efa_proto_av_implicit_av_lru_entry_move(av, entry);
			*fi_addr = implicit_fi_addr;
			goto out;
		}

		ret = efa_proto_av_entry_implicit_to_explicit(av, addr, implicit_fi_addr, fi_addr);
		if (ret)
			*fi_addr = FI_ADDR_NOTAVAIL;
		goto out;
	}

	entry = efa_proto_av_entry_alloc(av, addr, flags, context, insert_shm_av, insert_implicit_av);
	if (!entry) {
		*fi_addr = FI_ADDR_NOTAVAIL;
		ret = -FI_EADDRNOTAVAIL;
		goto out;
	}

	if (insert_implicit_av) {
		*fi_addr = entry->implicit_fi_addr;
		EFA_INFO(FI_LOG_AV,
			 "Successfully inserted address GID[%s] QP[%u] QKEY[%u] to implicit AV. fi_addr: %ld\n",
			 raw_gid_str, addr->qpn, addr->qkey, *fi_addr);
	} else {
		*fi_addr = entry->fi_addr;
		EFA_INFO(FI_LOG_AV,
			 "Successfully inserted address GID[%s] QP[%u] QKEY[%u] to explicit AV. fi_addr: %ld\n",
			 raw_gid_str, addr->qpn, addr->qkey, *fi_addr);
	}
	ret = 0;

out:
	ofi_genlock_unlock(&av->efa_av.util_av.lock);
	ofi_genlock_unlock(&av->util_av_implicit.lock);
	return ret;
}

/* ---- Protocol AV fi_ops ---- */

/**
 * @brief insert addresses into protocol AV (fi_av_insert implementation)
 *
 * @param[in]	av_fid	fid of AV
 * @param[in]	addr	buffer containing addresses to insert
 * @param[in]	count	number of addresses
 * @param[out]	fi_addr	array for returned fabric addresses
 * @param[in]	flags	operation flags
 * @param[in]	context	user context
 * @return	number of addresses successfully inserted
 */
static int efa_proto_av_insert(struct fid_av *av_fid, const void *addr,
			       size_t count, fi_addr_t *fi_addr,
			       uint64_t flags, void *context)
{
	struct efa_av *base_av = container_of(av_fid, struct efa_av, util_av.av_fid);
	struct efa_proto_av *av = container_of(base_av, struct efa_proto_av, efa_av);
	int ret = 0, success_cnt = 0;
	size_t i = 0;
	struct efa_ep_addr *addr_i;
	fi_addr_t fi_addr_res;

	if (av->efa_av.util_av.flags & FI_EVENT)
		return -FI_ENOEQ;

	if ((flags & FI_SYNC_ERR) && (!context || (flags & FI_EVENT)))
		return -FI_EINVAL;

	flags &= ~FI_MORE;
	if (flags)
		return -FI_ENOSYS;

	ofi_genlock_lock(&av->efa_av.domain->srx_lock);

	for (i = 0; i < count; i++) {
		addr_i = (struct efa_ep_addr *) ((uint8_t *)addr + i * EFA_EP_ADDR_LEN);

		ret = efa_proto_av_insert_one(av, addr_i, &fi_addr_res, flags, context, true, false);
		if (ret) {
			EFA_WARN(FI_LOG_AV, "insert raw_addr to av failed! ret=%d\n", ret);
			break;
		}

		if (fi_addr)
			fi_addr[i] = fi_addr_res;
		success_cnt++;
	}

	ofi_genlock_unlock(&av->efa_av.domain->srx_lock);

	for (; i < count ; i++) {
		if (fi_addr)
			fi_addr[i] = FI_ADDR_NOTAVAIL;
	}

	return success_cnt;
}

/**
 * @brief retrieve an address from the protocol AV (fi_av_lookup implementation)
 *
 * @param[in]		av_fid	fid of AV
 * @param[in]		fi_addr	fabric address to look up
 * @param[out]		addr	buffer to store the returned address
 * @param[in,out]	addrlen	on input, size of addr buffer; on output, bytes written
 * @return	0 on success, negative error code on failure
 */
static int efa_proto_av_lookup(struct fid_av *av_fid, fi_addr_t fi_addr,
			       void *addr, size_t *addrlen)
{
	struct efa_av *base_av = container_of(av_fid, struct efa_av, util_av.av_fid);
	struct efa_proto_av *av = container_of(base_av, struct efa_proto_av, efa_av);
	struct efa_proto_av_entry *entry = NULL;

	if (av->efa_av.type != FI_AV_TABLE)
		return -FI_EINVAL;

	if (fi_addr == FI_ADDR_NOTAVAIL)
		return -FI_EINVAL;

	ofi_genlock_lock(&av->efa_av.util_av.lock);
	entry = efa_proto_av_addr_to_entry(av, fi_addr);
	if (!entry) {
		ofi_genlock_unlock(&av->efa_av.util_av.lock);
		return -FI_EINVAL;
	}

	memcpy(addr, (void *)entry->ep_addr, MIN(EFA_EP_ADDR_LEN, *addrlen));
	ofi_genlock_unlock(&av->efa_av.util_av.lock);
	if (*addrlen > EFA_EP_ADDR_LEN)
		*addrlen = EFA_EP_ADDR_LEN;
	return 0;
}

/**
 * @brief remove addresses from the protocol AV (fi_av_remove implementation)
 *
 * @param[in]	av_fid	fid of AV
 * @param[in]	fi_addr	array of fabric addresses to remove
 * @param[in]	count	number of addresses
 * @param[in]	flags	operation flags
 * @return	0 on success, negative error code on failure
 */
static int efa_proto_av_remove(struct fid_av *av_fid, fi_addr_t *fi_addr,
			       size_t count, uint64_t flags)
{
	int err = 0;
	size_t i;
	struct efa_av *base_av;
	struct efa_proto_av *av;
	struct efa_proto_av_entry *entry;

	if (!fi_addr)
		return -FI_EINVAL;

	base_av = container_of(av_fid, struct efa_av, util_av.av_fid);
	av = container_of(base_av, struct efa_proto_av, efa_av);
	if (av->efa_av.type != FI_AV_TABLE)
		return -FI_EINVAL;

	ofi_genlock_lock(&av->efa_av.domain->srx_lock);
	ofi_genlock_lock(&av->efa_av.util_av.lock);
	for (i = 0; i < count; i++) {
		entry = efa_proto_av_addr_to_entry(av, fi_addr[i]);
		if (!entry) {
			err = -FI_EINVAL;
			break;
		}

		efa_proto_av_entry_release(av, entry, false);
	}

	if (i < count)
		assert(err);

	ofi_genlock_unlock(&av->efa_av.util_av.lock);
	ofi_genlock_unlock(&av->efa_av.domain->srx_lock);
	return err;
}

/**
 * @brief convert an address to a printable string (fi_av_straddr implementation)
 *
 * @param[in]		av_fid	fid of AV
 * @param[in]		addr	address to convert
 * @param[out]		buf	buffer to store the string
 * @param[in,out]	len	on input, size of buf; on output, bytes written
 * @return	pointer to buf
 */
static const char *efa_proto_av_straddr(struct fid_av *av_fid, const void *addr,
					char *buf, size_t *len)
{
	return ofi_straddr(buf, len, FI_ADDR_EFA, addr);
}

static struct fi_ops_av efa_proto_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = efa_proto_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = efa_proto_av_remove,
	.lookup = efa_proto_av_lookup,
	.straddr = efa_proto_av_straddr
};

/**
 * @brief release all entries in the explicit and implicit reverse AVs
 *
 * @param[in]	av	protocol AV
 */
static void efa_proto_av_close_reverse_av(struct efa_proto_av *av)
{
	struct efa_cur_reverse_av *cur_entry, *curtmp;
	struct efa_prv_reverse_av *prv_entry, *prvtmp;

	ofi_genlock_lock(&av->efa_av.domain->srx_lock);

	ofi_genlock_lock(&av->efa_av.util_av.lock);

	HASH_ITER(hh, av->efa_av.cur_reverse_av, cur_entry, curtmp) {
		efa_proto_av_entry_release(av, (struct efa_proto_av_entry *)cur_entry->av_entry, false);
	}

	HASH_ITER(hh, av->efa_av.prv_reverse_av, prv_entry, prvtmp) {
		efa_proto_av_entry_release(av, (struct efa_proto_av_entry *)prv_entry->av_entry, false);
	}

	ofi_genlock_unlock(&av->efa_av.util_av.lock);

	ofi_genlock_lock(&av->util_av_implicit.lock);

	HASH_ITER(hh, av->cur_reverse_av_implicit, cur_entry, curtmp) {
		efa_proto_av_entry_release(av, (struct efa_proto_av_entry *)cur_entry->av_entry, true);
	}

	HASH_ITER(hh, av->prv_reverse_av_implicit, prv_entry, prvtmp) {
		efa_proto_av_entry_release(av, (struct efa_proto_av_entry *)prv_entry->av_entry, true);
	}

	ofi_genlock_unlock(&av->util_av_implicit.lock);

	ofi_genlock_unlock(&av->efa_av.domain->srx_lock);
}

/**
 * @brief close the protocol AV and release all resources (fi_close implementation)
 *
 * @param[in]	fid	fid of AV
 * @return	0 on success, negative error code on failure
 */
static int efa_proto_av_close(struct fid *fid)
{
	struct efa_av *base_av;
	struct efa_proto_av *av;
	int err = 0;
	struct efa_ep_addr_hashable *ep_addr_hashable, *tmp;

	base_av = container_of(fid, struct efa_av, util_av.av_fid.fid);
	av = container_of(base_av, struct efa_proto_av, efa_av);

	efa_proto_av_close_reverse_av(av);

	err = ofi_av_close(&av->efa_av.util_av);
	if (OFI_UNLIKELY(err))
		EFA_WARN(FI_LOG_AV, "Failed to close util av: %s\n",
			fi_strerror(err));

	err = ofi_av_close(&av->util_av_implicit);
	if (OFI_UNLIKELY(err))
		EFA_WARN(FI_LOG_AV, "Failed to close implicit util av: %s\n",
			fi_strerror(err));

	if (av->shm_rdm_av) {
		err = fi_close(&av->shm_rdm_av->fid);
		if (OFI_UNLIKELY(err))
			EFA_WARN(FI_LOG_AV,
				 "Failed to close shm av: %s\n",
				 fi_strerror(err));
	}

	HASH_ITER(hh, av->evicted_peers_hashset, ep_addr_hashable, tmp) {
		HASH_DEL(av->evicted_peers_hashset, ep_addr_hashable);
		free(ep_addr_hashable);
	}

	free(av);
	return err;
}

static struct fi_ops efa_proto_av_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_proto_av_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/**
 * @brief open a protocol AV (fi_av_open implementation for RDM)
 *
 * @param[in]	domain_fid	fid of domain
 * @param[in]	attr		AV attributes
 * @param[out]	av_fid		pointer to store the opened AV fid
 * @param[in]	context		user context
 * @return	0 on success, negative error code on failure
 */
int efa_proto_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		      struct fid_av **av_fid, void *context)
{
	struct efa_domain *efa_domain;
	struct efa_proto_av *av;
	struct fi_av_attr av_attr = { 0 };
	size_t context_len;
	size_t universe_size;
	int ret, retv;

	if (!attr)
		return -FI_EINVAL;

	if (attr->name)
		return -FI_ENOSYS;

	if (attr->flags)
		return -FI_ENOSYS;

	if (!attr->count)
		attr->count = EFA_MIN_AV_SIZE;
	else
		attr->count = MAX(attr->count, EFA_MIN_AV_SIZE);

	av = calloc(1, sizeof(*av));
	if (!av)
		return -FI_ENOMEM;

	if (attr->type == FI_AV_MAP) {
		EFA_INFO(FI_LOG_AV, "FI_AV_MAP is deprecated in Libfabric 2.x. Please use FI_AV_TABLE. "
					"EFA provider will now switch to using FI_AV_TABLE.\n");
	}
	attr->type = FI_AV_TABLE;

	efa_domain = container_of(domain_fid, struct efa_domain, util_domain.domain_fid);

	if (fi_param_get_size_t(NULL, "universe_size",
				&universe_size) == FI_SUCCESS)
		attr->count = MAX(attr->count, universe_size);

	context_len = sizeof(struct efa_proto_av_entry) - EFA_EP_ADDR_LEN;

	ret = efa_av_init_util_av(efa_domain, attr, &av->util_av_implicit, context,
				  context_len);
	if (ret)
		goto err;

	ret = efa_av_init_util_av(efa_domain, attr, &av->efa_av.util_av, context,
				  context_len);
	if (ret)
		goto err_close_util_av_implicit;

	if (efa_domain->fabric && efa_domain->fabric->shm_fabric) {
		av_attr = *attr;
		if (efa_env.shm_av_size > EFA_SHM_MAX_AV_COUNT) {
			ret = -FI_ENOSYS;
			EFA_WARN(FI_LOG_AV,
				 "The requested av size is beyond"
				 " shm supported maximum av size: %s\n",
				 fi_strerror(-ret));
			goto err_close_util_av;
		}
		av_attr.count = efa_env.shm_av_size;
		assert(av_attr.type == FI_AV_TABLE);
		ret = fi_av_open(efa_domain->shm_domain, &av_attr,
				 &av->shm_rdm_av, context);
		if (ret)
			goto err_close_util_av;
	}

	EFA_INFO(FI_LOG_AV, "fi_av_attr:%" PRId64 "\n", attr->flags);

	av->efa_av.domain = efa_domain;
	av->efa_av.type = attr->type;
	av->efa_av.used = 0;
	av->implicit_av_size = efa_env.implicit_av_size;
	av->used_implicit = 0;
	av->shm_used = 0;

	*av_fid = &av->efa_av.util_av.av_fid;
	(*av_fid)->fid.fclass = FI_CLASS_AV;
	(*av_fid)->fid.context = context;
	(*av_fid)->fid.ops = &efa_proto_av_fi_ops;
	(*av_fid)->ops = &efa_proto_av_ops;

	dlist_init(&av->implicit_av_lru_list);

	return 0;

err_close_util_av:
	retv = ofi_av_close(&av->efa_av.util_av);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av: %s\n", fi_strerror(-retv));

err_close_util_av_implicit:
	retv = ofi_av_close(&av->util_av_implicit);
	if (retv)
		EFA_WARN(FI_LOG_AV,
			 "Unable to close util_av_implicit: %s\n", fi_strerror(-retv));

err:
	free(av);
	return ret;
}
