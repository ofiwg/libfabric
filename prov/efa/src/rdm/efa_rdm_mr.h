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

/**
 * @brief Initialize desc_gen[] to the invalid sentinel value.
 *
 * Must be called when an ope is constructed, before the capture
 * function runs. This ensures that recycled bufpool slots do not
 * carry stale gen values from a previous ope, and that the
 * capture function can distinguish "not yet captured" from a
 * real gen snapshot.
 */
static inline void efa_rdm_mr_gen_init_ope_desc(struct efa_rdm_ope *ope)
{
	unsigned int i;

	for (i = 0; i < ope->iov_count; i++)
		ope->desc_gen[i] = EFA_RDM_MR_INVALID_GEN_VALUE;
}

/**
 * @brief Capture the gen of each efa_rdm_mr in ope->desc[].
 *
 * Must be called after ope->desc[] and ope->iov_count are populated.
 */
static inline void efa_rdm_mr_gen_capture_in_ope_desc(struct efa_rdm_ope *ope)
{
	struct efa_rdm_mr *efa_rdm_mr;
	unsigned int i;

	/*
	 * Skip if descs are not populated yet (app passed NULL, provider
	 * will fill them later via try_fill_desc), or if the gen was
	 * already captured on a previous call (prevents re-capture on
	 * the repost path from overwriting the dispatch-time snapshot).
	 */
	if (!ope->desc[0] || efa_rdm_mr_gen_value_is_valid(ope->desc_gen[0]))
		return;

	for (i = 0; i < ope->iov_count; i++) {
		/* We statically assert that efa_mr is first member of efa_rdm_mr */
		efa_rdm_mr = (struct efa_rdm_mr *)ope->desc[i];

		if (!efa_rdm_mr)
			break;

		ope->desc_gen[i] = efa_rdm_mr->gen;
	}
}

/**
 * @brief Check whether any MR in ope->desc[] has been closed since dispatch.
 *
 * @return true if all MRs are still valid, false on any gen mismatches.
 */
static inline bool efa_rdm_mr_gen_check_ope(struct efa_rdm_ope *ope)
{
	struct efa_rdm_mr *efa_rdm_mr;
	unsigned int i;

	for (i = 0; i < ope->iov_count; i++) {
		/* We statically assert that efa_mr is first member of efa_rdm_mr */
		efa_rdm_mr = ope->desc[i];
		if (!efa_rdm_mr)
			break;

		/* Gen check is only invoked in the repost path, after descs gen were already captured */
		assert(efa_rdm_mr_gen_value_is_valid(ope->desc_gen[i]));

		if (efa_rdm_mr->gen != ope->desc_gen[i])
			return false;
	}

	return true;
}

#endif /* EFA_RDM_MR_H */
