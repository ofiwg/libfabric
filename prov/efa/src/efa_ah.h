/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AH_H
#define EFA_AH_H

#include "efa_domain.h"
#include "ofi_util.h"

#define EFA_GID_LEN	16

struct efa_ah {
	uint8_t		gid[EFA_GID_LEN]; /* efa device GID */
	struct ibv_ah	*ibv_ah; /* created by ibv_create_ah() using GID */
	uint16_t	ahn; /* adress handle number */
	/* Number of AV entries (across all paths) associated with this AH */
	int refcnt OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	UT_hash_handle	hh; /* hash map handle, link all efa_ah with efa_ep->ah_map */
};

/**
 * @brief RDM address handle
 *
 * Embeds the base efa_ah as its first member and adds the RDM-only split
 * reference counts, the list of implicit AV entries using this AH, and the
 * position in the domain's AH LRU list used for out-of-memory eviction.
 *
 * NOTE: struct efa_rdm_ah and the efa_rdm_ah_* helpers are defined here for now.
 * A follow-up patch moves them to rdm/efa_rdm_av.{h,c} alongside the rest of the
 * RDM AV code.
 */
struct efa_rdm_ah {
	struct efa_ah	efa_ah;
	/* Number of explicit AV entries associated with this AH */
	int explicit_refcnt OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* Number of implicit AV entries associated with this AH */
	int implicit_refcnt OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* dlist of all implicit AV entries associated with this AH entry */
	struct dlist_entry implicit_conn_list OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
	/* dlist entry in domain's LRU AH list */
	struct dlist_entry domain_lru_ah_list_entry OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
};

_Static_assert(offsetof(struct efa_rdm_ah, efa_ah) == 0,
	       "efa_ah must be the first member of efa_rdm_ah");

void efa_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    size_t alloc_size)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_ah_implicit_av_lru_ah_move(struct efa_domain *domain,
					struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_ah *efa_rdm_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
				bool insert_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_rdm_ah_release(struct efa_domain *domain, struct efa_ah *ah,
			bool release_from_implicit_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
