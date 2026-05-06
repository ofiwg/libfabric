/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_ah.h"
#include <infiniband/efadv.h>

/**
 * @brief Emit a detailed warning for ibv_create_ah EINVAL.
 *
 * The most common reasons for EINVAL are cross-AZ addressing, invalid
 * remote GID, and invalid PD. Log both local and remote GIDs plus the
 * PD pointer to help operators diagnose failures from logs alone.
 *
 * @param[in]	domain	efa domain (for local GID and PD)
 * @param[in]	gid	remote GID that failed
 */
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
 * @brief allocate an ibv_ah from GID, reusing existing AH if possible
 *
 * Uses a hash map to store GID to ibv_ah mapping and reuses ibv_ah for
 * the same GID. If ibv_create_ah fails, returns NULL with errno set.
 * The caller is responsible for handling ENOMEM (e.g. by evicting AH
 * entries and retrying).
 *
 * @param[in]	domain		efa domain
 * @param[in]	gid		GID
 * @param[in]	alloc_size	size to allocate (sizeof(efa_ah) or larger for protocol wrapper)
 * @return	pointer to efa_ah on success, NULL on failure (errno set)
 */
struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    size_t alloc_size)
{
	struct ibv_pd *ibv_pd = domain->ibv_pd;
	struct efa_ah *efa_ah;
	struct ibv_ah_attr ibv_ah_attr = { 0 };
	struct efadv_ah_attr efa_ah_attr = { 0 };
	int err;

	assert(alloc_size >= sizeof(struct efa_ah));

	efa_ah = NULL;

	ofi_genlock_lock(&domain->util_domain.lock);
	HASH_FIND(hh, domain->ah_map, gid, EFA_GID_LEN, efa_ah);
	if (efa_ah) {
		efa_ah->refcnt++;
		ofi_genlock_unlock(&domain->util_domain.lock);
		return efa_ah;
	}

	efa_ah = calloc(1, alloc_size);
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
		if (errno == EINVAL) {
			efa_ah_warn_create_einval(domain, gid);
		} else {
			EFA_WARN(FI_LOG_AV,
				 "ibv_create_ah failed! errno: %d\n", errno);
		}
		goto err_free;
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
	ofi_genlock_unlock(&domain->util_domain.lock);
	return efa_ah;

err_destroy_ibv_ah:
	ibv_destroy_ah(efa_ah->ibv_ah);
err_free:
	free(efa_ah);
	ofi_genlock_unlock(&domain->util_domain.lock);
	return NULL;
}

/**
 * @brief destroy an efa_ah (remove from hash, destroy ibv_ah, free)
 *
 * Caller must hold util_domain.lock.
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	efa_ah to destroy
 */
void efa_ah_destroy(struct efa_domain *domain, struct efa_ah *ah)
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
 * @brief release an efa_ah, destroying it when refcount reaches zero
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	efa_ah to release
 */
void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah)
{
	ofi_genlock_lock(&domain->util_domain.lock);

	assert(ah->refcnt > 0);
	ah->refcnt--;

	if (ah->refcnt == 0)
		efa_ah_destroy(domain, ah);

	ofi_genlock_unlock(&domain->util_domain.lock);
}
