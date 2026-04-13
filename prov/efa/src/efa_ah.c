/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_ah.h"
#include <infiniband/efadv.h>

/**
 * @brief Move the AH to the end of the LRU list to indicate that it is the
 * most recently used entry
 *
 * This function is not called in the efa_rdm_ep_get_peer so that we don't add
 * extra latency to the critical path with explicit AV insertion. We use the LRU
 * list to remove AH entries with only implicit AV entries, so it is OK to do
 * that.
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	efa AH to move
 */
void efa_ah_implicit_av_lru_ah_move(struct efa_domain *domain,
					struct efa_ah *ah)
{
	assert(ah->implicit_refcnt > 0 || ah->explicit_refcnt > 0);
	assert(dlist_entry_in_list(&domain->ah_lru_list,
				   &ah->domain_lru_ah_list_entry));

	dlist_remove(&ah->domain_lru_ah_list_entry);
	dlist_insert_tail(&ah->domain_lru_ah_list_entry,
			  &domain->ah_lru_list);
}

/**
 * @brief allocate an ibv_ah object from GID.
 *
 * Uses a hash map to store GID to ibv_ah mapping and reuses ibv_ah for
 * the same GID. If ibv_create_ah fails, returns NULL with errno set.
 * The caller is responsible for handling ENOMEM (e.g. by evicting AH
 * entries and retrying).
 *
 * @param[in]	domain		efa_domain
 * @param[in]	gid		GID
 * @param[in]	insert_implicit_av	whether this is for an implicit AV entry
 * @return	pointer to efa_ah on success, NULL on failure (errno set)
 */
struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    bool insert_implicit_av)
{
	struct ibv_pd *ibv_pd = domain->ibv_pd;
	struct efa_ah *efa_ah;
	struct ibv_ah_attr ibv_ah_attr = { 0 };
	struct efadv_ah_attr efa_ah_attr = { 0 };
	int err;

	efa_ah = NULL;

	ofi_genlock_lock(&domain->util_domain.lock);
	HASH_FIND(hh, domain->ah_map, gid, EFA_GID_LEN, efa_ah);
	if (efa_ah) {
		insert_implicit_av ? efa_ah->implicit_refcnt++ : efa_ah->explicit_refcnt++;
		efa_ah_implicit_av_lru_ah_move(domain, efa_ah);
		ofi_genlock_unlock(&domain->util_domain.lock);
		return efa_ah;
	}

	efa_ah = malloc(sizeof(struct efa_ah));
	if (!efa_ah) {
		errno = FI_ENOMEM;
		EFA_WARN(FI_LOG_AV, "cannot allocate memory for efa_ah\n");
		ofi_genlock_unlock(&domain->util_domain.lock);
		return NULL;
	}

	ibv_ah_attr.port_num = 1;
	ibv_ah_attr.is_global = 1;
	memcpy(ibv_ah_attr.grh.dgid.raw, gid, EFA_GID_LEN);
	efa_ah->ibv_ah = ibv_create_ah(ibv_pd, &ibv_ah_attr);
	if (!efa_ah->ibv_ah) {
		EFA_WARN(FI_LOG_AV,
			 "ibv_create_ah failed! errno: %d\n", errno);
		goto err_free_efa_ah;
	}

	err = efadv_query_ah(efa_ah->ibv_ah, &efa_ah_attr, sizeof(efa_ah_attr));
	if (err) {
		errno = err;
		EFA_WARN(FI_LOG_AV, "efadv_query_ah failed! err: %d\n", err);
		goto err_destroy_ibv_ah;
	}

	dlist_init(&efa_ah->implicit_conn_list);
	dlist_insert_tail(&efa_ah->domain_lru_ah_list_entry, &domain->ah_lru_list);
	efa_ah->implicit_refcnt = 0;
	efa_ah->explicit_refcnt = 0;
	insert_implicit_av ? efa_ah->implicit_refcnt++ : efa_ah->explicit_refcnt++;
	efa_ah->ahn = efa_ah_attr.ahn;
	memcpy(efa_ah->gid, gid, EFA_GID_LEN);
	HASH_ADD(hh, domain->ah_map, gid, EFA_GID_LEN, efa_ah);
	ofi_genlock_unlock(&domain->util_domain.lock);
	return efa_ah;

err_destroy_ibv_ah:
	ibv_destroy_ah(efa_ah->ibv_ah);
err_free_efa_ah:
	free(efa_ah);
	ofi_genlock_unlock(&domain->util_domain.lock);
	return NULL;
}

/**
 * @brief destroy an efa_ah object
 *
 * Removes AH from hash map and LRU list, destroys ibv_ah, frees memory.
 * Caller must hold util_domain.lock.
 *
 * @param[in]	domain	efa_domain
 * @param[in]	ah	efa_ah object to destroy
 */
void efa_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
{
	int err;

	assert(ah->implicit_refcnt == 0 && ah->explicit_refcnt == 0);
	assert(dlist_empty(&ah->implicit_conn_list));

	EFA_INFO(FI_LOG_AV, "Destroying AH for ahn %d\n", ah->ahn);
	dlist_remove(&ah->domain_lru_ah_list_entry);
	HASH_DEL(domain->ah_map, ah);

	err = ibv_destroy_ah(ah->ibv_ah);
	if (err)
		EFA_WARN(FI_LOG_AV, "ibv_destroy_ah failed! err=%d\n", err);
	free(ah);
}

/**
 * @brief release an efa_ah object after acquiring the util domain lock
 *
 * Decrements the appropriate refcount. If both refcounts reach zero,
 * destroys the AH.
 *
 * @param[in]	domain			efa_domain
 * @param[in]	ah			efa_ah object pointer
 * @param[in]	release_from_implicit_av	whether releasing from implicit AV
 */
void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah,
		    bool release_from_implicit_av)
{
	ofi_genlock_lock(&domain->util_domain.lock);
#if ENABLE_DEBUG
	struct efa_ah *tmp;

	HASH_FIND(hh, domain->ah_map, ah->gid, EFA_GID_LEN, tmp);
	assert(tmp == ah);
#endif
	assert((release_from_implicit_av && ah->implicit_refcnt > 0) ||
	       (!release_from_implicit_av && ah->explicit_refcnt > 0));

	release_from_implicit_av ? ah->implicit_refcnt-- :
				   ah->explicit_refcnt--;

	if (ah->implicit_refcnt == 0 && ah->explicit_refcnt == 0) {
		efa_ah_destroy_ah(domain, ah);
	}
	ofi_genlock_unlock(&domain->util_domain.lock);
}
