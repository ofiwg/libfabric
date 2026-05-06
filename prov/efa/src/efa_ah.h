/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AH_H
#define EFA_AH_H

#include "efa_domain.h"
#include "ofi_util.h"

#define EFA_GID_LEN	16

/**
 * @brief Base address handle — shared by efa-direct and protocol paths
 *
 * Contains only the ibv_ah, GID, AHN, refcount, and hash handle.
 * Protocol-specific fields (implicit_refcnt, implicit_conn_list,
 * LRU list entry) are in efa_proto_ah.
 *
 * pahole: size: 88, cachelines: 2 (2-byte hole after ahn)
 *
 * TX hot path: ibv_ah (off=16) is passed to ibv post_send/read/write
 *   on every send. Both ibv_ah and ahn are in cacheline 0.
 * All other fields are control path only (AH alloc/release/hash lookup).
 */
struct efa_ah {
	uint8_t		gid[EFA_GID_LEN];              /*     0    16 */
	struct ibv_ah	*ibv_ah;                       /*    16     8 */
	uint16_t	ahn;                           /*    24     2 */
	/* 2-byte hole */
	int		refcnt;                        /*    28     4 */
	UT_hash_handle	hh;                            /*    32    56 */
};

/**
 * @brief allocate an ibv_ah from GID, reusing existing AH if possible
 *
 * @param[in]	domain		efa domain
 * @param[in]	gid		GID
 * @param[in]	alloc_size	size to allocate (sizeof(efa_ah) or sizeof(efa_proto_ah))
 * @return	pointer to efa_ah on success, NULL on failure (errno set)
 */
struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    size_t alloc_size);

/**
 * @brief release an efa_ah, destroying it when refcount reaches zero
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	efa_ah to release
 */
void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah);

/**
 * @brief destroy an efa_ah (remove from hash, destroy ibv_ah, free)
 *
 * Caller must hold util_domain.lock.
 *
 * @param[in]	domain	efa domain
 * @param[in]	ah	efa_ah to destroy
 */
void efa_ah_destroy(struct efa_domain *domain, struct efa_ah *ah);

#endif
