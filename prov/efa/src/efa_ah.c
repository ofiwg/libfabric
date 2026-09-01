/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2016, Cisco Systems, Inc. All rights reserved. */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_ah.h"
#include <infiniband/efadv.h>

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
