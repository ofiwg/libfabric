/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef PGTABLE_H
#define PGTABLE_H

#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <ofi.h>

/*
 * Assertions which are checked in compile-time
 *
 * Usage: UCS_STATIC_ASSERT(condition)
 */
#define STATIC_ASSERT(_cond) \
     switch(0) {case 0:case (_cond):;}

/*
 * The Page Table data structure organizes non-overlapping regions of memory in
 * an efficient radix tree, optimized for large and/or aligned regions.
 *
 * A page table entry can point to either a region (indicated by setting the
 * PGT_PTE_FLAG_REGION bit), or another entry (indicated by PGT_PTE_FLAG_DIR),
 * or be empty - if none of these bits is set.
 *
 */

/* Address alignment requirements */
#define PGT_ADDR_SHIFT         4
#define PGT_ADDR_ALIGN         (1ul << PGT_ADDR_SHIFT)
#define PGT_ADDR_ORDER          (sizeof(pgt_addr_t) * 8)
#define PGT_ADDR_MAX           ((pgt_addr_t)-1)

/* Page table entry/directory constants */
#define PGT_ENTRY_SHIFT        4
#define PGT_ENTRIES_PER_DIR    (1ul << (PGT_ENTRY_SHIFT))
#define PGT_ENTRY_MASK         (PGT_ENTRIES_PER_DIR - 1)

/* Page table pointers constants and flags */
#define PGT_ENTRY_FLAG_REGION  BIT(0)
#define PGT_ENTRY_FLAG_DIR     BIT(1)
#define PGT_ENTRY_FLAGS_MASK   (PGT_ENTRY_FLAG_REGION|PGT_ENTRY_FLAG_DIR)
#define PGT_ENTRY_PTR_MASK     (~PGT_ENTRY_FLAGS_MASK)
#define PGT_ENTRY_MIN_ALIGN    (PGT_ENTRY_FLAGS_MASK + 1)

/* Declare a variable as aligned so it could be placed in page table entry */
#define PGT_ENTRY_V_ALIGNED    ALIGNED(PGT_ENTRY_MIN_ALIGN > sizeof(long) ? \
										PGT_ENTRY_MIN_ALIGN : sizeof(long))

#define PGT_REGION_FMT            "%p [0x%lx..0x%lx]"
#define PGT_REGION_ARG(_region)   (_region), (_region)->start, (_region)->end


/* Define the address type */
typedef unsigned long              pgt_addr_t;

/* Forward declarations */
typedef struct pgtable         pgtable_t;
typedef struct pgt_region      pgt_region_t;
typedef struct pgt_entry       pgt_entry_t;
typedef struct pgt_dir         pgt_dir_t;


/**
 * Callback for allocating a page table directory.
 *
 * @param [in]  pgtable  Pointer to the page table to allocate the directory for.
 *
 * @return Pointer to newly allocated pgdir, or NULL if failed. The pointer must
 *         be aligned to PGT_ENTRY_ALIGN boundary.
 * */
typedef pgt_dir_t* (*pgt_dir_alloc_callback_t)(const pgtable_t *pgtable);


/**
 * Callback for releasing a page table directory.
 *
 * @param [in]  pgtable  Pointer to the page table to in which the directory was
 *                       allocated.
 * @param [in]  pgdir    Page table directory to release.
 */
typedef void (*pgt_dir_release_callback_t)(const pgtable_t *pgtable,
											   pgt_dir_t *pgdir);


/**
 * Callback for searching for regions in the page table.
 *
 * @param [in]  pgtable  The page table.
 * @param [in]  region   Found region.
 * @param [in]  arg      User-defined argument.
 */
typedef void (*pgt_search_callback_t)(const pgtable_t *pgtable,
										  pgt_region_t *region, void *arg);


/**
 * Memory region in the page table.
 * The structure itself, and the pointers in it, must be aligned to 2^PTR_SHIFT.
 */
struct pgt_region {
	pgt_addr_t                 start; /**< Region start address */
	pgt_addr_t                 end;   /**< Region end address */
} PGT_ENTRY_V_ALIGNED;


/**
 * Page table entry:
 *
 * +--------------------+---+---+
 * |    pointer (MSB)   | d | r |
 * +--------------------+---+---+
 * |                    |   |   |
 * 64                   2   1   0
 *
 */
struct pgt_entry {
	pgt_addr_t                 value;  /**< Pointer + type bits. Can point
												to either a @ref pgt_dir_t or
												a @ref pgt_region_t. */
};


/**
 * Page table directory.
 */
struct pgt_dir {
	pgt_entry_t                entries[PGT_ENTRIES_PER_DIR];
	unsigned                       count;       /**< Number of valid entries */
};


/* Page table structure */
struct pgtable {

	/* Maps addresses whose (63-shift) high bits equal to value
	 * This means: value * (2**shift) .. value * (2**(shift+1)) - 1
	 */
	pgt_entry_t                root;        /**< root entry */
	pgt_addr_t                 base;        /**< base address */
	pgt_addr_t                 mask;        /**< mask for page table address range */
	unsigned                       shift;       /**< page table address span is 2**shift */
	unsigned                       num_regions; /**< total number of regions */
	pgt_dir_alloc_callback_t   pgd_alloc_cb;
	pgt_dir_release_callback_t pgd_release_cb;
};


/**
 * Initialize a page table.
 *
 * @param [in]  pgtable     Page table to initialize.
 * @param [in]  alloc_cb    Callback that will be used to allocate page directory,
 *                           which is the basic building block of the page table
 *                           data structure. This may allow the page table functions
 *                           to be safe to use from memory allocation context.
 * @param [in]  release_cb  Callback to release memory which was allocated by alloc_cb.
 */
int pgtable_init(pgtable_t *pgtable,
					 pgt_dir_alloc_callback_t alloc_cb,
					 pgt_dir_release_callback_t release_cb);

/**
 * Cleanup the page table and release all associated memory.
 *
 * @param [in]  pgtable     Page table to initialize.
 */
void pgtable_cleanup(pgtable_t *pgtable);


/**
 * Add a memory region to the page table.
 *
 * @param [in]  pgtable     Page table to insert the region to.
 * @param [in]  region      Memory region to insert. The region must remain valid
 *                           and unchanged s long as it's in the page table.
 *
 * @return 0 - region was added.
 *         -EINVAL - memory region address in invalid (misaligned or empty)
 *         -EALREADY - the region overlaps with existing region.
 *
 */
int pgtable_insert(pgtable_t *pgtable, pgt_region_t *region);


/**
 * Remove a memory region from the page table.
 *
 * @param [in]  pgtable     Page table to remove the region from.
 * @param [in]  region      Memory region to remove. This must be the same pointer
 *                           passed to @ref pgtable_insert.
 *
 * @return 0 - region was added.
 *         -EINVAL - memory region address in invalid (misaligned or empty)
 *         -EALREADY - the region overlaps with existing region.
 *
 */
int pgtable_remove(pgtable_t *pgtable, pgt_region_t *region);


/*
 * Find a region which contains the given address.
 *
 * @param [in]  pgtable     Page table to search the address in.
 * @param [in]  address     Address to search.
 *
 * @return Region which contains 'address', or NULL if not found.
 */
pgt_region_t *pgtable_lookup(const pgtable_t *pgtable,
									 pgt_addr_t address);


/**
 * Search for all regions overlapping with a given address range.
 *
 * @param [in]  pgtable     Page table to search the range in.
 * @param [in]  from        Lower bound of the range.
 * @param [in]  to          Upper bound of the range (inclusive).
 * @param [in]  cb          Callback to be called for every region found.
 *                           The callback must not modify the page table.
 * @param [in]  arg         User-defined argument to the callback.
 */
void pgtable_search_range(const pgtable_t *pgtable,
							  pgt_addr_t from, pgt_addr_t to,
							  pgt_search_callback_t cb, void *arg);


/**
 * Remove all regions from the page table and call the provided callback for each.
 *
 * @param [in]  pgtable     Page table to clean up.
 * @param [in]  cb          Callback to be called for every region, after it (and
 *                           all others) are removed.
 *                           The callback must not modify the page table.
 * @param [in]  arg         User-defined argument to the callback.
 */
void pgtable_purge(pgtable_t *pgtable, pgt_search_callback_t cb,
					   void *arg);


/**
 * @return >Number of regions currently present in the page table.
 */
static inline unsigned pgtable_num_regions(const pgtable_t *pgtable)
{
	return pgtable->num_regions;
}


#endif
