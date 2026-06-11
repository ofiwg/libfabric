/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_MR_H
#define EFA_RDM_MR_H

#include "efa_mr.h"
#include <stddef.h>

 /* UINT32_MAX is reserved as a sentinel indicating "not yet set." A valid gen is any other value. */
#define EFA_RDM_MR_INVALID_GEN_VALUE UINT32_MAX

struct efa_rdm_mr {
	struct efa_mr		efa_mr;
	bool			inserted_to_mr_map;
	bool			needs_sync;
	/* RDM-specific HMEM data handle */
	void			*hmem_data;
	/* RDM-specific flags */
	uint64_t		flags;
	uint64_t		device;
	/* Used only in MR cache */
	struct ofi_mr_entry	*entry;
	/* Used only in rdm */
	struct fid_mr		*shm_mr;
	/*
	 * Monotonic generation counter bumped on every close.
	 * Preserved across bufpool slot reuse so that in-flight ops
	 * that captured a stale desc can detect the invalidation and
	 * be canceled early.
	 */
	uint32_t		gen;
};

/* Compile-time assertion to ensure safe casting between efa_mr and efa_rdm_mr */
_Static_assert(offsetof(struct efa_rdm_mr, efa_mr) == 0,
               "efa_mr must be the first member of efa_rdm_mr for safe casting");

			   /* RDM specific extern declarations */
extern struct fi_ops_mr efa_domain_mr_cache_ops;
extern int efa_mr_cache_enable;
extern size_t efa_mr_max_cached_count;
extern size_t efa_mr_max_cached_size;
extern struct fi_ops_mr efa_rdm_domain_mr_ops;

/*
 * Multiplier to give some room in the device memory registration limits
 * to allow processes added to a running job to bootstrap.
 */
#define EFA_MR_CACHE_LIMIT_MULT (.9)

/* RDM MR cache functions */
int efa_rdm_mr_cache_open(struct ofi_mr_cache **cache, struct efa_domain *domain);
int efa_rdm_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			       struct ofi_mr_entry *entry);
void efa_rdm_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
				  struct ofi_mr_entry *entry);
int efa_rdm_mr_cache_regv(struct fid_domain *domain_fid, const struct iovec *iov,
			  size_t count, uint64_t access, uint64_t offset,
			  uint64_t requested_key, uint64_t flags,
			  struct fid_mr **mr, void *context);

/**
 * @brief Advance the MR generation counter, skipping EFA_RDM_MR_INVALID_GEN_VALUE.
 *
 * EFA_RDM_MR_INVALID_GEN_VALUE is reserved as a sentinel in ope->desc_gen[]
 * to indicate "not yet captured." A live MR must never hold that value.
 */
static inline void efa_rdm_mr_gen_bump(struct efa_rdm_mr *mr)
{
	mr->gen++;
	if (OFI_UNLIKELY(mr->gen == EFA_RDM_MR_INVALID_GEN_VALUE))
		mr->gen = 0;
}

/**
 * @brief Check whether a gen value is valid.
 */
static inline bool efa_rdm_mr_gen_value_is_valid(uint32_t gen)
{
	return (gen != EFA_RDM_MR_INVALID_GEN_VALUE);
}

#endif /* EFA_RDM_MR_H */
