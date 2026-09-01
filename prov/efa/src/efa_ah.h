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

void efa_ah_destroy_ah(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_ah *efa_ah_alloc(struct efa_domain *domain, const uint8_t *gid,
			    size_t alloc_size)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_ah_release(struct efa_domain *domain, struct efa_ah *ah)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
