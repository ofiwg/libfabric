/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_ah.h"
#include "rdm/efa_rdm_av.h"
#include "rdm/efa_rdm_domain.h"
#include <infiniband/efadv.h>

/*
 * The efa_rdm_ah_* helpers below layer the RDM-only AH policy (split
 * explicit/implicit reference counts, the per-domain AH LRU list and
 * out-of-memory eviction of implicit-only AHs) on top of the base efa_ah_*
 * functions above. They are defined here for now; a follow-up patch moves them
 * to rdm/efa_rdm_av.c alongside the rest of the RDM AV code.
 */

/*
 * Forward declaration: efa_rdm_ah_implicit_av_evict_ah below destroys an
 * evicted AH through the RDM destroy variant, which is defined next to the
 * base efa_ah_destroy_ah further down.
 */
static void efa_rdm_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

/**
 * @brief move an AH to the tail of the domain's AH LRU list
 *
 * This is not called on the explicit AV insertion critical path so that we
 * don't add extra latency there. The LRU list is only used to pick AH entries
 * with only implicit AV entries for eviction, so that is OK.
 *
 * NOTE: this helper still has callers in rdm/efa_rdm_av.c while it lives in
 * efa_ah.c, so it is exported (declared in efa_ah.h) for this transitional
 * patch. It becomes static again once it moves next to those callers.
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

static void efa_ah_warn_create_einval(struct efa_domain *domain, const uint8_t *gid)
{
	char remote_gid_str[INET6_ADDRSTRLEN] = {0};
	char local_gid_str[INET6_ADDRSTRLEN] = {0};

	if (!inet_ntop(AF_INET6, gid, remote_gid_str, INET6_ADDRSTRLEN))
		snprintf(remote_gid_str, sizeof(remote_gid_str), "(unable to convert GID to string)");
	if (!inet_ntop(AF_INET6, domain->device->ibv_gid.raw, local_gid_str, INET6_ADDRSTRLEN))
		snprintf(local_gid_str, sizeof(local_gid_str), "(unable to convert GID to string)");

	EFA_WARN(FI_LOG_AV,
		 "ibv_create_ah failed with EINVAL. "
		 "Local GID: %s, remote GID: %s. "
		 "Possible causes: "
		 "1) Remote GID is in a different availability zone (cross-AZ communication is not enabled). "
		 "2) Remote GID is invalid. "
		 "3) Protection domain %p is invalid.\n",
		 local_gid_str, remote_gid_str, domain->ibv_pd);
}

/**
 * @brief find-or-create a base efa_ah object for a GID
 *
 * Uses a per-domain hash map to reuse an ibv_ah for the same GID. On a hit the
 * shared reference count is incremented. The allocation size is supplied by the
 * caller so the RDM layer can allocate a larger struct efa_rdm_ah that embeds
 * this base AH.
 *
 * On ENOMEM from ibv_create_ah this base helper simply fails; the RDM layer
 * (efa_rdm_ah_alloc) is responsible for evicting an implicit-only AH and
 * retrying, since eviction is an RDM-only policy.
 *
 * @param[in]	domain		efa_domain
 * @param[in]	gid		GID
 * @param[in]	alloc_size	size of the (base or wrapping) AH struct to allocate
 */
struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    size_t alloc_size)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	struct ibv_pd *ibv_pd = domain->ibv_pd;
	struct efa_ah *efa_ah;
	struct ibv_ah_attr ibv_ah_attr = { 0 };
	struct efadv_ah_attr efa_ah_attr = { 0 };
	int err;

	assert(alloc_size >= sizeof(struct efa_ah));

	efa_ah = NULL;

	HASH_FIND(hh, domain->ah_map, gid, EFA_GID_LEN, efa_ah);
	if (efa_ah) {
		efa_ah->refcnt++;
		return efa_ah;
	}

	efa_ah = malloc(alloc_size);
	if (!efa_ah) {
		errno = FI_ENOMEM;
		EFA_WARN(FI_LOG_AV, "cannot allocate memory for efa_ah\n");
		return NULL;
	}

	ibv_ah_attr.port_num = 1;
	ibv_ah_attr.is_global = 1;
	memcpy(ibv_ah_attr.grh.dgid.raw, gid, EFA_GID_LEN);
	efa_ah->ibv_ah = ibv_create_ah(ibv_pd, &ibv_ah_attr);
	if (!efa_ah->ibv_ah) {
		if (errno == EINVAL)
			efa_ah_warn_create_einval(domain, gid);
		else if (errno != FI_ENOMEM)
			EFA_WARN(FI_LOG_AV,
				 "ibv_create_ah failed! errno: %s\n", strerror(errno));
		goto err_free_efa_ah;
	}

	err = efadv_query_ah(efa_ah->ibv_ah, &efa_ah_attr, sizeof(efa_ah_attr));
	if (err) {
		errno = err;
		EFA_WARN(FI_LOG_AV, "efadv_query_ah failed! err: %d\n", err);
		goto err_destroy_ibv_ah;
	}

	efa_ah->refcnt = 1;
	efa_ah->ahn = efa_ah_attr.ahn;
	memcpy(efa_ah->gid, gid, EFA_GID_LEN);
	HASH_ADD(hh, domain->ah_map, gid, EFA_GID_LEN, efa_ah);
	return efa_ah;

err_destroy_ibv_ah:
	ibv_destroy_ah(efa_ah->ibv_ah);
err_free_efa_ah:
	free(efa_ah);
	return NULL;
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

void efa_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
	int err;

	assert(ah->refcnt == 0);

	EFA_INFO(FI_LOG_AV, "Destroying AH for ahn %d\n", ah->ahn);
	HASH_DEL(domain->ah_map, ah);

	err = ibv_destroy_ah(ah->ibv_ah);
	if (err)
		EFA_WARN(FI_LOG_AV, "ibv_destroy_ah failed! err=%d\n", err);
	free(ah);
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
 * @brief release a base efa_ah reference; destroy at zero
 *
 * @param[in]	domain	efa_domain
 * @param[in]	ah	efa_ah object pointer
 */
void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym)
{
#if ENABLE_DEBUG
	struct efa_ah *tmp;

	HASH_FIND(hh, domain->ah_map, ah->gid, EFA_GID_LEN, tmp);
	assert(tmp == ah);
#endif
	assert(ofi_genlock_held(&domain->util_domain.lock));
	assert(ah->refcnt > 0);

	ah->refcnt--;

	if (ah->refcnt == 0)
		efa_ah_destroy_ah(domain, ah);
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
