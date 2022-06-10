/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include "pgtable.h"
#include "bitops.h"

#define pgt_entry_clear(_pte) \
	{ (_pte)->value = 0; }

#define pgt_entry_value(_pte) \
	((void*)((_pte)->value & PGT_ENTRY_PTR_MASK))

#define pgt_entry_test(_pte, _flag) \
	((_pte)->value & (_flag))

#define pgt_entry_present(_pte) \
	pgt_entry_test(_pte, PGT_ENTRY_FLAG_REGION | PGT_ENTRY_FLAG_DIR)

#define pgt_is_addr_aligned(_addr) \
	(!((_addr) & (PGT_ADDR_ALIGN - 1)))

#define pgt_check_ptr(_ptr) \
	do { \
		assert(!((uintptr_t)(_ptr) & (PGT_ENTRY_MIN_ALIGN - 1))); \
	} while (0)

#define pgt_entry_set_region(_pte, _region) \
	do { \
		pgt_region_t *tmp = (_region); \
		pgt_check_ptr(tmp); \
		(_pte)->value = ((uintptr_t)tmp) | PGT_ENTRY_FLAG_REGION; \
	} while (0)

#define pgt_entry_set_dir(_pte, _dir) \
	do { \
		pgt_dir_t *tmp = (_dir); \
		pgt_check_ptr(tmp); \
		(_pte)->value = ((uintptr_t)tmp) | PGT_ENTRY_FLAG_DIR; \
	} while (0)

#define pgt_entry_get_region(_pte) \
	({ \
		assert(pgt_entry_test(_pte, PGT_ENTRY_FLAG_REGION)); \
		(pgt_region_t*)pgt_entry_value(_pte); \
	})

#define pgt_entry_get_dir(_pte) \
	({ \
		assert(pgt_entry_test(_pte, PGT_ENTRY_FLAG_DIR)); \
		(pgt_dir_t*)pgt_entry_value(_pte); \
	})


static inline pgt_dir_t* pgt_dir_alloc(pgtable_t *pgtable)
{
	pgt_dir_t *pgd;

	pgd = pgtable->pgd_alloc_cb(pgtable);
	if (pgd == NULL) {
		fprintf(stderr, "Failed to allocate page table directory");
	}

	pgt_check_ptr(pgd);
	memset(pgd, 0, sizeof(*pgd));
	return pgd;
}

static inline void pgt_dir_release(pgtable_t *pgtable, pgt_dir_t* pgd)
{
	pgtable->pgd_release_cb(pgtable, pgd);
}

static inline void pgt_address_advance(pgt_addr_t *address_p,
										   unsigned order)
{
	assert(order < 64);
	/* coverity[large_shift] */
	*address_p += 1ul << order;
}

static void pgtable_reset(pgtable_t *pgtable)
{
	pgtable->base  = 0;
	pgtable->mask  = ((pgt_addr_t)-1) << PGT_ADDR_SHIFT;
	pgtable->shift = PGT_ADDR_SHIFT;
}

/**
 * Make the page table map a wider range of addresses - expands by PGT_ENTRY_SHIFT.
 */
static void pgtable_expand(pgtable_t *pgtable)
{
	pgt_dir_t *pgd;

	assert(pgtable->shift <= (PGT_ADDR_ORDER - PGT_ENTRY_SHIFT));

	if (pgt_entry_present(&pgtable->root)) {
		pgd = pgt_dir_alloc(pgtable);
		pgd->entries[(pgtable->base >> pgtable->shift) & PGT_ENTRY_MASK] =
						pgtable->root;
		pgd->count = 1;
		pgt_entry_set_dir(&pgtable->root, pgd);
	}

	pgtable->shift += PGT_ENTRY_SHIFT;
	pgtable->mask <<= PGT_ENTRY_SHIFT;
	pgtable->base  &= pgtable->mask;
}

/**
 * Shrink the page table address span if possible
 *
 * @return Whether it was shrinked.
 */
static int pgtable_shrink(pgtable_t *pgtable)
{
	pgt_entry_t *pte;
	pgt_dir_t *pgd;
	unsigned pte_idx;

	if (!pgt_entry_present(&pgtable->root)) {
		pgtable_reset(pgtable);
		return 0;
	} else if (!pgt_entry_test(&pgtable->root, PGT_ENTRY_FLAG_DIR)) {
		return 0;
	}

	pgd = pgt_entry_get_dir(&pgtable->root);
	assert(pgd->count > 0); /* should be empty */

	/* If there is just one PTE, we can reduce the page table to map
	 * this PTE only.
	 */
	if (pgd->count != 1) {
		return 0;
	}

	/* Search for the single PTE in dir */
	for (pte_idx = 0, pte = pgd->entries; !pgt_entry_present(pte); ++pte_idx, ++pte) {
		assert(pte_idx < PGT_ENTRIES_PER_DIR);
	}

	/* Remove one level */
	pgtable->shift -= PGT_ENTRY_SHIFT;
	pgtable->base  |= (pgt_addr_t)pte_idx << pgtable->shift;
	pgtable->mask  |= PGT_ENTRY_MASK << pgtable->shift;
	pgtable->root   = *pte;
	pgt_dir_release(pgtable, pgd);
	return 1;
}

static void pgtable_check_page(pgt_addr_t address, unsigned order)
{
	assert( (address & ((1ul << order) - 1)) == 0 );
	assert( ((order - PGT_ADDR_SHIFT) % PGT_ENTRY_SHIFT) == 0);
}

/**
 * @return Order of the next whole page starting in "start" and ending before "end"
 *         If both are 0, return the full word size.
 */
static unsigned pgtable_get_next_page_order(pgt_addr_t start, pgt_addr_t end)
{
	unsigned log2_len;

	assert(pgt_is_addr_aligned(start));
	assert(pgt_is_addr_aligned(end));

	if ((end == 0) && (start == 0)) {
		log2_len = PGT_ADDR_ORDER; /* entire range */
	} else if (end == start) {
		log2_len = PGT_ADDR_SHIFT;
	} else {
		log2_len = ilog2(end - start);
		if (start) {
			log2_len = MIN(ffs64(start), log2_len);
		}
	}

	assert((log2_len >= PGT_ADDR_SHIFT) &&
		   (log2_len <= PGT_ADDR_ORDER));

	/* Order should be: [ADDR_SHIFT + k * ENTRY_SHIFT] */
	return (((log2_len - PGT_ADDR_SHIFT) / PGT_ENTRY_SHIFT) * PGT_ENTRY_SHIFT)
			+ PGT_ADDR_SHIFT;
}

/**
 * Insert a variable-size page to the page table.
 *
 * @param address  address to insert
 * @param order    page size to insert - should be k*PTE_SHIFT for a certain k
 * @param region   region to insert
 */
static int
pgtable_insert_page(pgtable_t *pgtable, pgt_addr_t address,
						unsigned order, pgt_region_t *region)
{
	pgt_dir_t dummy_pgd = {};
	pgt_entry_t *pte;
	pgt_dir_t *pgd;
	unsigned shift;

	pgtable_check_page(address, order);

	/* Make root map addresses which include our interval */
	while (pgtable->shift < order) {
		pgtable_expand(pgtable);
	}

	if (pgt_entry_present(&pgtable->root)) {
		while ((address & pgtable->mask) != pgtable->base) {
			pgtable_expand(pgtable);
		}
	} else {
		pgtable->base = address & pgtable->mask;
	}

	/* Insert the page in the PTE */
	pgd   = &dummy_pgd;
	shift = pgtable->shift;
	pte   = &pgtable->root;
	while (1) {
		if (order == shift) {
			if (pgt_entry_present(pte)) {
				goto err;
			}
			pgt_entry_set_region(pte, region);
			++pgd->count;
			break;
		} else {
			if (pgt_entry_test(pte, PGT_ENTRY_FLAG_REGION)) {
				goto err;
			}

			assert(shift >= PGT_ENTRY_SHIFT + order);  /* sub PTE should be able to hold it */

			if (!pgt_entry_present(pte)) {
				++pgd->count;
				pgt_entry_set_dir(pte, pgt_dir_alloc(pgtable));
			}

			pgd    = pgt_entry_get_dir(pte);
			shift -= PGT_ENTRY_SHIFT;
			pte    = &pgd->entries[(address >> shift) & PGT_ENTRY_MASK];
		}
	}

	return 0;

err:
	while (pgtable_shrink(pgtable));
	return -EALREADY;
}

/*
 * `region' is only used to compare pointers
 */
static int
pgtable_remove_page_recurs(pgtable_t *pgtable, pgt_addr_t address,
							   unsigned order, pgt_dir_t *pgd,
							   pgt_entry_t *pte, unsigned shift,
							   pgt_region_t *region)
{
	pgt_dir_t *next_dir;
	pgt_entry_t *next_pte;
	int status;
	unsigned next_shift;

	if (pgt_entry_test(pte, PGT_ENTRY_FLAG_REGION)) {
		assert(shift == order);
		if (pgt_entry_get_region(pte) != region) {
			goto no_elem;
		}

		--pgd->count;
		pgt_entry_clear(pte);
		return 0;
	} else if (pgt_entry_test(pte, PGT_ENTRY_FLAG_DIR)) {
		next_dir   = pgt_entry_get_dir(pte);
		next_shift = shift - PGT_ENTRY_SHIFT;
		next_pte   = &next_dir->entries[(address >> next_shift) & PGT_ENTRY_MASK];

		status = pgtable_remove_page_recurs(pgtable, address, order, next_dir,
												next_pte, next_shift, region);
		if (status != 0) {
			goto no_elem;
		}

		if (next_dir->count == 0) {
			pgt_entry_clear(pte);
			--pgd->count;
			pgt_dir_release(pgtable, next_dir);
		}
		return 0;
	}

no_elem:
	return -ENOENT;
}

static int
pgtable_remove_page(pgtable_t *pgtable, pgt_addr_t address,
						unsigned order, pgt_region_t *region)
{
	pgt_dir_t dummy_pgd = {};
	int status;

	pgtable_check_page(address, order);

	if ((address & pgtable->mask) != pgtable->base) {
		return -ENOENT;
	}

	status = pgtable_remove_page_recurs(pgtable, address, order, &dummy_pgd,
											&pgtable->root, pgtable->shift,
											region);
	if (status != 0) {
		return status;
	}

	while (pgtable_shrink(pgtable));
	return 0;
}

int pgtable_insert(pgtable_t *pgtable, pgt_region_t *region)
{
	pgt_addr_t address = region->start;
	pgt_addr_t end     = region->end;
	int status;
	unsigned order;

	if ((address >= end) || !pgt_is_addr_aligned(address) ||
		!pgt_is_addr_aligned(end))
	{
		return -EINVAL;
	}

	assert(address != end);
	while (address < end) {
		order = pgtable_get_next_page_order(address, end);
		status = pgtable_insert_page(pgtable, address, order, region);
		if (status != 0) {
			goto err;
		}

		pgt_address_advance(&address, order);
	}
	++pgtable->num_regions;

	return 0;

err:
	/* Revert all pages we've inserted by now */
	end     = address;
	address = region->start;
	while (address < end) {
		order = pgtable_get_next_page_order(address, end);
		pgtable_remove_page(pgtable, address, order, region);
		pgt_address_advance(&address, order);
	}
	return status;
}

int pgtable_remove(pgtable_t *pgtable, pgt_region_t *region)
{
	pgt_addr_t address = region->start;
	pgt_addr_t end     = region->end;
	int status;
	unsigned order;

	if ((address >= end) || !pgt_is_addr_aligned(address) ||
		!pgt_is_addr_aligned(end))
	{
		return -ENOENT;
	}

	while (address < end) {
		order = pgtable_get_next_page_order(address, end);
		status = pgtable_remove_page(pgtable, address, order, region);
		if (status != 0) {
			assert(address == region->start); /* Cannot be partially removed */
			return status;
		}

		pgt_address_advance(&address, order);
	}

	assert(pgtable->num_regions > 0);
	--pgtable->num_regions;

	return 0;
}

pgt_region_t *pgtable_lookup(const pgtable_t *pgtable,
									 pgt_addr_t address)
{
	const pgt_entry_t *pte;
	pgt_region_t *region;
	pgt_dir_t *dir;
	unsigned shift;

	/* Check if the address is mapped by the page table */
	if ((address & pgtable->mask) != pgtable->base) {
		return NULL;
	}

	/* Descend into the page table */
	pte   = &pgtable->root;
	shift = pgtable->shift;
	for (;;) {
		if (pgt_entry_test(pte, PGT_ENTRY_FLAG_REGION)) {
			region = pgt_entry_get_region(pte);
			assert((address >= region->start) && (address < region->end));
			return region;
		} else if (pgt_entry_test(pte, PGT_ENTRY_FLAG_DIR)) {
			dir = pgt_entry_get_dir(pte);
			shift -= PGT_ENTRY_SHIFT;
			pte = &dir->entries[(address >> shift) & PGT_ENTRY_MASK];
		} else {
			return NULL;
		}
	}
}

static void pgtable_search_recurs(const pgtable_t *pgtable,
									  pgt_addr_t address, unsigned order,
									  const pgt_entry_t *pte, unsigned shift,
									  pgt_search_callback_t cb, void *arg,
									  pgt_region_t **last_p)
{
	pgt_entry_t *next_pte;
	pgt_region_t *region;
	pgt_dir_t *dir;
	unsigned next_shift;
	unsigned i;

	if (pgt_entry_test(pte, PGT_ENTRY_FLAG_REGION)) {
		region = pgt_entry_value(pte);

		/* Check that we are not continuing with the previous region */
		if (*last_p == region) {
			return;
		} else if (*last_p != NULL) {
			assert(region->start >= (*last_p)->end);
		}
		*last_p = region;

		/* Assert that the region actually overlaps the address */
		assert(MAX(region->start,   address) <=
			   MIN(region->end - 1, address + MASK_SAFE(order)));

		/* Call the callback */
		cb(pgtable, region, arg);

	} else if (pgt_entry_test(pte, PGT_ENTRY_FLAG_DIR)) {
		dir = pgt_entry_get_dir(pte);
		assert(shift >= PGT_ENTRY_SHIFT);
		next_shift = shift - PGT_ENTRY_SHIFT;

		if (order < shift) {
			/* One of the sub-ptes maps the region */
			assert(order <= next_shift);
			next_pte = &dir->entries[(address >> next_shift) & PGT_ENTRY_MASK];
			pgtable_search_recurs(pgtable, address, order, next_pte,
									  next_shift, cb, arg, last_p);
		} else {
			/* All sub-ptes contained in the region */
			for (i = 0; i < PGT_ENTRIES_PER_DIR; ++i) {
				next_pte = &dir->entries[i];
				pgtable_search_recurs(pgtable, address, order, next_pte,
										  next_shift, cb, arg, last_p);
			}
		}
	}
}

void pgtable_search_range(const pgtable_t *pgtable,
							  pgt_addr_t from, pgt_addr_t to,
							  pgt_search_callback_t cb, void *arg)
{
	pgt_addr_t address = align_down_pow2(from, PGT_ADDR_ALIGN);
	pgt_addr_t end     = align_up_pow2(to, PGT_ADDR_ALIGN);
	pgt_region_t *last;
	unsigned order;

	/* if the page table is covering only part of the address space, intersect
	 * the range with page table address span */
	if (pgtable->shift < (sizeof(uint64_t) * 8)) {
		address = MAX(address, pgtable->base);
		end     = MIN(end,     pgtable->base + BIT(pgtable->shift));
	} else {
		assert(pgtable->base == 0);
	}

	last = NULL;
	while (address <= to) {
		order = pgtable_get_next_page_order(address, end);
		if ((address & pgtable->mask) == pgtable->base) {
			pgtable_search_recurs(pgtable, address, order, &pgtable->root,
									  pgtable->shift, cb, arg, &last);
		}

		if (order == PGT_ADDR_ORDER) {
			break;
		}

		pgt_address_advance(&address, order);
	}
}

static void pgtable_purge_callback(const pgtable_t *pgtable,
									   pgt_region_t *region,
									   void *arg)
{
	pgt_region_t ***region_pp = arg;
	**region_pp = region;
	++(*region_pp);
}

void pgtable_purge(pgtable_t *pgtable, pgt_search_callback_t cb,
					   void *arg)
{
	unsigned num_regions = pgtable->num_regions;
	pgt_region_t **all_regions, **next_region, *region;
	pgt_addr_t from, to;
	int status;
	unsigned i;

	if (num_regions == 0) {
		goto out;
	}

	all_regions = calloc(num_regions, sizeof(*all_regions));
	if (all_regions == NULL) {
		fprintf(stderr, "failed to allocate array to collect all regions, will leak");
		return;
	}

	next_region = all_regions;
	from = pgtable->base;
	to   = pgtable->base + ((1ul << pgtable->shift) & pgtable->mask) - 1;
	pgtable_search_range(pgtable, from, to, pgtable_purge_callback,
							 &next_region);
	assert(next_region == all_regions + num_regions);

	for (i = 0; i < num_regions; ++i) {
		region = all_regions[i];
		status = pgtable_remove(pgtable, region);
		if (status != 0) {
			fprintf(stderr, "failed to remove pgtable region" PGT_REGION_FMT,
					PGT_REGION_ARG(region));
		}
		cb(pgtable, region, arg);
	}

	free(all_regions);

out:
	/* Page table should be totally empty */
	assert(!pgt_entry_present(&pgtable->root));
	assert(pgtable->shift == PGT_ADDR_SHIFT);
	assert(pgtable->base == 0);
	assert(pgtable->num_regions == 0);
}

int pgtable_init(pgtable_t *pgtable,
				pgt_dir_alloc_callback_t alloc_cb,
				pgt_dir_release_callback_t release_cb)
{
	STATIC_ASSERT(is_pow2(PGT_ENTRY_MIN_ALIGN));

	/* ADDR_MAX+1 must be power of 2, or wrap around to 0. */
	STATIC_ASSERT(is_pow2_or_zero(PGT_ADDR_MAX + 1));

	/* We must cover all bits of the address up to ADDR_MAX */
	STATIC_ASSERT(((ilog2(PGT_ADDR_MAX) + 1 - PGT_ADDR_SHIFT) %
					  PGT_ENTRY_SHIFT) == 0);

	pgt_entry_clear(&pgtable->root);
	pgtable_reset(pgtable);
	pgtable->num_regions    = 0;
	pgtable->pgd_alloc_cb   = alloc_cb;
	pgtable->pgd_release_cb = release_cb;
	return 0;
}

void pgtable_cleanup(pgtable_t *pgtable)
{
	if (pgtable->num_regions != 0) {
		fprintf(stderr, "page table not empty during cleanup");
	}
}
