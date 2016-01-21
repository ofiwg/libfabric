/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 *
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>

#include "gnix.h"
#include "gnix_nic.h"
#include "gnix_util.h"
#include "gnix_mr.h"
#include "gnix_priv.h"

typedef enum cache_entry_flags {
	GNIX_CE_RETIRED = 1 << 0,
} cache_entry_flags_e;

typedef enum cache_entry_state {
	GNIX_CES_DEAD = 0,
	GNIX_CES_INUSE,
	GNIX_CES_STALE,
} cache_entry_state_e;

/**
 * @brief gnix memory registration cache entry
 *
 * @var   state      state of the memory registration cache entry
 * @var   mr         gnix memory registration descriptor
 * @var   mem_hndl   gni memory handle for the memory registration
 * @var   key        gnix memory registration cache key
 * @var   domain     gnix domain associated with the memory registration
 * @var   nic        gnix nic associated with the memory registration
 * @var   ref_cnt    reference counting for the cache
 * @var   lru_entry  lru list entry
 * @var   siblings   list of sibling entries
 * @var   children   list of subsumed child entries
 * @var   flags      cache entry flags @see cache_entry_flags_e
 */
typedef struct gnix_mr_cache_entry {
	cache_entry_state_e state;
	struct gnix_fid_mem_desc mr;
	gni_mem_handle_t mem_hndl;
	gnix_mr_cache_key_t key;
	struct gnix_fid_domain *domain;
	struct gnix_nic *nic;
	atomic_t ref_cnt;
	struct dlist_entry lru_entry;
	struct dlist_entry siblings;
	struct dlist_entry children;
	cache_entry_flags_e flags;
} gnix_mr_cache_entry_t;

/* forward declarations */
static int __gnix_mr_cache_init(
		gnix_mr_cache_t      **cache,
		gnix_mr_cache_attr_t *attr);

static int __mr_cache_register(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		struct gnix_fid_mem_desc **mr);

static int __mr_cache_deregister(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_mem_desc *mr);

static int fi_gnix_mr_close(fid_t fid);

static inline int __mr_cache_entry_put(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t *entry);

static inline int __mr_cache_entry_get(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t *entry);

static int __mr_cache_create_registration(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		gnix_mr_cache_entry_t    **entry,
		gnix_mr_cache_key_t      *key);


/* global declarations */
/* memory registration operations */
static struct fi_ops fi_gnix_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_gnix_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

/* default attributes for new caches */
gnix_mr_cache_attr_t __default_mr_cache_attr = {
		.soft_reg_limit      = 4096,
		.hard_reg_limit      = -1,
		.hard_stale_limit    = 128,
		.lazy_deregistration = 1
};

/**
 * Sign extends the value passed into up to length parameter
 *
 * @param[in]  val  value to be sign extended
 * @param[in]  len  length to sign extend the value
 * @return          sign extended value to length, len
 */
static inline int64_t __sign_extend(
		uint64_t val,
		int len)
{
	int64_t m = 1UL << (len - 1);
	int64_t r = (val ^ m) - m;

	return r;
}

/**
 * Key comparison function for finding overlapping gnix memory
 * registration cache entries
 *
 * @param[in] x key to be inserted or found
 * @param[in] y key to be compared against
 *
 * @return    -1 if it should be positioned at the left, 0 if the same,
 *             1 otherwise
 */
static int __find_overlapping_addr(
		void *x,
		void *y)
{
	gnix_mr_cache_key_t *to_find  = (gnix_mr_cache_key_t *) x;
	gnix_mr_cache_key_t *to_compare = (gnix_mr_cache_key_t *) y;
	uint64_t to_find_end = to_find->address + to_find->length;
	uint64_t to_compare_end = to_compare->address + to_compare->length;

	/* format: (x_addr,  x_len) - (y_addr,  y_len) truth_value
	 *
	 * case 1: (0x1000, 0x1000) - (0x1400, 0x0800) true
	 * case 2: (0x1000, 0x1000) - (0x0C00, 0x0800) true
	 * case 3: (0x1000, 0x1000) - (0x1C00, 0x0800) true
	 * case 4: (0x1000, 0x1000) - (0x0C00, 0x2000) true
	 * case 5: (0x1000, 0x1000) - (0x0400, 0x0400) false
	 * case 6: (0x1000, 0x1000) - (0x2400, 0x0400) false
	 */
	if (!(to_find_end < to_compare->address ||
			to_compare_end < to_find->address))
		return 0;

	/* left */
	if (to_find->address < to_compare->address)
		return -1;

	return 1;
}

/**
 * Key comparison function for gnix memory registration caches
 *
 * @param[in] x key to be inserted or found
 * @param[in] y key to be compared against
 *
 * @return    -1 if it should be positioned at the left, 0 if the same,
 *             1 otherwise
 */
static inline int __mr_cache_key_comp(
		void *x,
		void *y)
{
	gnix_mr_cache_key_t *to_insert  = (gnix_mr_cache_key_t *) x;
	gnix_mr_cache_key_t *to_compare = (gnix_mr_cache_key_t *) y;

	if (to_compare->address == to_insert->address)
		return 0;

	/* to the left */
	if (to_insert->address < to_compare->address)
		return -1;

	/* to the right */
	return 1;
}

/**
 * Helper function for matching the exact key entry
 *
 * @param entry     memory registration cache key
 * @param to_match  memory registration cache key
 * @return 1 if match, otherwise 0
 */
static inline int __match_exact_key(
		gnix_mr_cache_key_t *entry,
		gnix_mr_cache_key_t *to_match)
{
	return entry->address == to_match->address &&
			entry->length == to_match->length;
}

/**
 * dlist search function for matching the exact memory registration key
 *
 * @param entry memory registration cache entry
 * @param match memory registration cache key
 * @return 1 if match, otherwise 0
 */
static inline int __mr_exact_key(struct dlist_entry *entry,
		const void *match)
{
	gnix_mr_cache_entry_t *x = container_of(entry,
							gnix_mr_cache_entry_t,
							siblings);

	gnix_mr_cache_key_t *y = (gnix_mr_cache_key_t *) match;

	return __match_exact_key(&x->key, y);
}


/**
 * Helper function to determine if one key subsumes another
 *
 * @param x  gnix_mr_cache_key
 * @param y  gnix_mr_cache_key
 * @return 1 if x subsumes y, 0 otherwise
 */
static inline int __can_subsume(
		gnix_mr_cache_key_t *x,
		gnix_mr_cache_key_t *y)
{
	return (x->address <= y->address) &&
			((x->address + x->length) >=
					(y->address + y->length));
}

static inline void __attach_retired_entries_to_registration(
		gnix_mr_cache_t *cache,
		struct dlist_entry *retired_entries,
		gnix_mr_cache_entry_t *parent)
{
	gnix_mr_cache_entry_t *entry, *tmp;

	dlist_for_each_safe(retired_entries, entry, tmp, siblings) {
		dlist_remove(&entry->siblings);
		dlist_insert_tail(&entry->siblings,
				&parent->children);
		if (!dlist_empty(&entry->children)) {
			/* move the entry's children to the sibling tree
			 * and decrement the reference count */
			dlist_splice_tail(&parent->children,
					&entry->children);
			__mr_cache_entry_put(cache, entry);
		}
	}
	assert(dlist_empty(retired_entries));
	__mr_cache_entry_get(cache, parent);
}

static inline void __remove_sibling_entries_from_tree(
		gnix_mr_cache_t *cache,
		struct dlist_entry *list,
		RbtHandle tree)
{
	RbtStatus rc;
	RbtIterator iter;
	gnix_mr_cache_entry_t *entry;

	dlist_for_each(list, entry, siblings)
	{
		GNIX_INFO(FI_LOG_MR,
				"removing key from tree, key=%llx:%llx\n",
				entry->key.address,
				entry->key.length);
		iter = rbtFind(tree, &entry->key);
		assert(iter);

		rc = rbtErase(tree, iter);
		if (unlikely(rc != RBT_STATUS_OK))
			GNIX_INFO(FI_LOG_MR,
					"could not remove entry from tree\n");
		assert(rc == RBT_STATUS_OK);
	}
}

/**
 * Pushes an entry into the LRU cache. No limits are maintained here as
 *   the hard_stale_limit attr value will directly limit the lru size
 *
 * @param[in] cache  a memory registration cache object
 * @param[in] entry  a memory registration cache entry
 *
 * @return           FI_SUCCESS, always
 */
static inline int __mr_cache_lru_enqueue(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t *entry)
{
	dlist_insert_tail(&entry->lru_entry, &cache->lru_head);

	return FI_SUCCESS;
}

/**
 * Pops an registration cache entry from the lru cache.
 *
 * @param[in] cache  a memory registration cache
 * @param[in] entry  a memory registration cache entry
 *
 * @return           FI_SUCCESS, on success
 * @return           -FI_ENOENT, on empty LRU
 */
static inline int __mr_cache_lru_dequeue(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t **entry)
{
	gnix_mr_cache_entry_t *ret;

	ret = dlist_first_entry(&cache->lru_head,
			gnix_mr_cache_entry_t, lru_entry);
	if (unlikely(!ret)) { /* we check list_empty before calling */
		*entry = NULL;
		return -FI_ENOENT;
	}

	/* remove entry from the list */
	*entry = ret;
	dlist_remove(&ret->lru_entry);

	return FI_SUCCESS;
}

/**
 * Destroys the memory registration cache entry and deregisters the memory
 *   region with uGNI
 *
 * @param[in] entry  a memory registration cache entry
 *
 * @return           grc from GNI_MemDeregister
 */
static inline int __mr_cache_entry_destroy(
		gnix_mr_cache_entry_t *entry)
{
	gni_return_t ret;

	fastlock_acquire(&entry->nic->lock);
	ret = GNI_MemDeregister(entry->nic->gni_nic_hndl, &entry->mem_hndl);
	fastlock_release(&entry->nic->lock);
	if (ret == GNI_RC_SUCCESS) {
		/* release reference to domain */
		_gnix_ref_put(entry->domain);

		/* release reference to nic */
		_gnix_ref_put(entry->nic);
		entry->state = GNIX_CES_DEAD;

		free(entry);
	} else {
		GNIX_WARN(FI_LOG_MR, "failed to deregister memory"
				" region, cache_entry=%p ret=%i\n", entry, ret);
	}

	return ret;
}

static inline int __insert_entry_into_stale(
		gnix_mr_cache_t *cache,
		gnix_mr_cache_entry_t *entry)
{
	RbtStatus rc;
	int grc = GNI_RC_SUCCESS;

	rc = rbtInsert(cache->stale.rb_tree,
			&entry->key,
			entry);
	if (rc != RBT_STATUS_OK) {
		GNIX_ERR(FI_LOG_MR,
				"could not insert into stale rb tree,"
				" rc=%d key.address=%llx key.length=%llx entry=%p",
				rc,
				entry->key.address,
				entry->key.length,
				entry);

		grc = __mr_cache_entry_destroy(entry);
	} else {
		GNIX_INFO(FI_LOG_MR,
				"inserted key=%llx:%llx into stale\n",
				entry->key.address,
				entry->key.length);

		__mr_cache_lru_enqueue(cache, entry);
		atomic_inc(&cache->stale.elements);
		entry->state = GNIX_CES_STALE;
	}

	return grc;
}

static inline void __resolve_stale_entry_collision(
		gnix_mr_cache_t *cache,
		RbtIterator found,
		gnix_mr_cache_entry_t *entry)
{
	RbtStatus rc;
	gnix_mr_cache_entry_t *c_entry, *tmp;
	gnix_mr_cache_key_t *c_key;
	int grc;
	DLIST_HEAD(to_destroy);
	RbtIterator iter = found;
	int add_new_entry = 1, cmp;

	GNIX_WARN(FI_LOG_MR, "resolving collisions\n");

	while (iter) {
		rbtKeyValue(cache->stale.rb_tree, iter, (void **) &c_key,
				(void **) &c_entry);

		cmp = __find_overlapping_addr(&entry->key, c_key);
		if (cmp != 0)
			break;

		if (__can_subsume(&entry->key, c_key) ||
				(entry->key.length > c_key->length)) {
			GNIX_INFO(FI_LOG_MR,
					"adding stale entry to destroy list, key=%llx:%llx",
					c_key->address, c_key->length);
			dlist_insert_tail(&c_entry->siblings, &to_destroy);
		} else {
			add_new_entry = 0;
		}

		iter = rbtNext(cache->stale.rb_tree, iter);
	}

	/* TODO I can probably do this in a single sweep, avoiding a second
	 * pass and incurring n lg n removal time
	 */
	dlist_for_each_safe(&to_destroy, c_entry, tmp, siblings)
	{
		GNIX_INFO(FI_LOG_MR, "removing key from tree, key=%llx:%ll\n",
				c_entry->key.address, c_entry->key.length);
		iter = rbtFind(cache->stale.rb_tree, &c_entry->key);
		assert(iter);

		rc = rbtErase(cache->stale.rb_tree,
						iter);
		assert(rc == RBT_STATUS_OK);

		dlist_remove(&c_entry->lru_entry);
		dlist_remove(&c_entry->siblings);
		atomic_dec(&cache->stale.elements);
		__mr_cache_entry_destroy(c_entry);
	}
	assert(dlist_empty(&to_destroy));

	if (add_new_entry) {
		grc = __insert_entry_into_stale(cache, entry);
		assert(grc == GNI_RC_SUCCESS);
	} else {
		/* stale entry is larger than this one
		 * so lets just toss this entry out
		 */
		GNIX_INFO(FI_LOG_MR,
				"larger entry already exists, "
				"to_destroy=%llx:%llx\n",
				entry->key.address,
				entry->key.length);

		grc = __mr_cache_entry_destroy(entry);
		if (grc != GNI_RC_SUCCESS) {
			GNIX_ERR(FI_LOG_MR,
					"failed to destroy a "
					"registration, entry=%p grc=%d\n",
					c_entry, grc);
		}
	}
}

/**
 * Increments the reference count on a memory registration cache entry
 *
 * @param[in] cache  gnix memory registration cache
 * @param[in] entry  a memory registration cache entry
 *
 * @return           reference count for the registration
 */
static inline int __mr_cache_entry_get(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t *entry)
{
	return atomic_inc(&entry->ref_cnt);
}

/**
 * Decrements the reference count on a memory registration cache entry
 *
 * @param[in] cache  gnix memory registration cache
 * @param[in] entry  a memory registration cache entry
 *
 * @return           grc from GNI_MemDeregister
 */
static inline int __mr_cache_entry_put(
		gnix_mr_cache_t       *cache,
		gnix_mr_cache_entry_t *entry)
{
	RbtIterator iter;
	gni_return_t grc = GNI_RC_SUCCESS;
	RbtIterator found;
	gnix_mr_cache_entry_t *parent = NULL;
	struct dlist_entry *next;

	if (atomic_dec(&entry->ref_cnt) == 0) {
		next = entry->siblings.next;
		dlist_remove(&entry->children);
		dlist_remove(&entry->siblings);

		/* if this is the last child to deallocate,
		 * release the reference to the parent
		 */
		if (next != &entry->siblings && dlist_empty(next)) {
			parent = container_of(next, gnix_mr_cache_entry_t,
					children);

			grc = __mr_cache_entry_put(cache, parent);
			if (unlikely(grc != GNI_RC_SUCCESS)) {
				GNIX_ERR(FI_LOG_MR,
						"failed to release reference to parent, "
						"parent=%p refs=%d\n",
						parent,
						atomic_get(&parent->ref_cnt));
			}
		}

		atomic_dec(&cache->inuse.elements);

		if (!(entry->flags & GNIX_CE_RETIRED)) {
			iter = rbtFind(cache->inuse.rb_tree, &entry->key);
			if (unlikely(!iter)) {
				GNIX_ERR(FI_LOG_MR,
						"failed to find entry in the inuse cache\n");
			} else {
				rbtErase(cache->inuse.rb_tree, iter);
			}
		}

		/* if we are doing lazy dereg and the entry
		 * isn't retired, put it in the stale cache
		 */
		if (cache->attr.lazy_deregistration &&
				!(entry->flags & GNIX_CE_RETIRED)) {
			GNIX_INFO(FI_LOG_MR, "moving key %llx:%llx to stale\n",
					entry->key.address, entry->key.length);

			found = rbtFindLeftmost(cache->stale.rb_tree,
					&entry->key, __find_overlapping_addr);
			if (found) {
				/* one or more stale entries would overlap with this
				 * new entry. We need to resolve these collisions by dropping
				 * registrations
				 */
				__resolve_stale_entry_collision(cache, found, entry);
			} else {
				/* if not found, ... */
				grc = __insert_entry_into_stale(cache, entry);
			}
		} else {
			/* if retired or not using lazy registration */
			GNIX_INFO(FI_LOG_MR,
					"destroying entry, key=%llx:%llx\n",
					entry->key.address,
					entry->key.length);

			grc = __mr_cache_entry_destroy(entry);
		}

		if (unlikely(grc != GNI_RC_SUCCESS)) {
			GNIX_WARN(FI_LOG_MR,
					"GNI_MemDeregister returned '%s'\n",
					gni_err_str[grc]);
		}
	}


	return grc;
}

/**
 * Converts a key to a gni memory handle without calculating crc
 *
 * @param key   gnix memory registration key
 * @param mhdl  gni memory handle
 */
void _gnix_convert_key_to_mhdl_no_crc(
		gnix_mr_key_t *key,
		gni_mem_handle_t *mhdl)
{
	uint64_t va = key->pfn;
	uint8_t flags = 0;

	va = (uint64_t) __sign_extend(va << GNIX_MR_PAGE_SHIFT,
				      GNIX_MR_VA_BITS);

	if (key->flags & GNIX_MR_FLAG_READONLY)
		flags |= GNI_MEMHNDL_ATTR_READONLY;

	GNI_MEMHNDL_INIT((*mhdl));
	GNI_MEMHNDL_SET_VA((*mhdl), va);
	GNI_MEMHNDL_SET_MDH((*mhdl), key->mdd);
	GNI_MEMHNDL_SET_NPAGES((*mhdl), GNI_MEMHNDL_NPGS_MASK);
	GNI_MEMHNDL_SET_FLAGS((*mhdl), flags);
	GNI_MEMHNDL_SET_PAGESIZE((*mhdl), GNIX_MR_PAGE_SHIFT);
}

/**
 * Converts a key to a gni memory handle
 *
 * @param key   gnix memory registration key
 * @param mhdl  gni memory handle
 */
void _gnix_convert_key_to_mhdl(
		gnix_mr_key_t *key,
		gni_mem_handle_t *mhdl)
{
	_gnix_convert_key_to_mhdl_no_crc(key, mhdl);
	compiler_barrier();
	GNI_MEMHNDL_SET_CRC((*mhdl));
}

/**
 * Converts a gni memory handle to gnix memory registration key
 *
 * @param mhdl  gni memory handle
 * @return uint64_t representation of a gnix memory registration key
 */
uint64_t _gnix_convert_mhdl_to_key(gni_mem_handle_t *mhdl)
{
	gnix_mr_key_t key = {{{{0}}}};
	key.pfn = GNI_MEMHNDL_GET_VA((*mhdl)) >> GNIX_MR_PAGE_SHIFT;
	key.mdd = GNI_MEMHNDL_GET_MDH((*mhdl));
	//key->format = GNI_MEMHNDL_NEW_FRMT((*mhdl));
	key.flags = 0;

	if (GNI_MEMHNDL_GET_FLAGS((*mhdl)) & GNI_MEMHNDL_FLAG_READONLY)
		key.flags |= GNIX_MR_FLAG_READONLY;

	return key.value;
}

/**
 * Helper function to calculate the length of a potential registration
 * based on some rules of the registration cache.
 *
 * Registrations should be page aligned and contain all of page(s)
 *
 * @param address   base address of the registration
 * @param length    length of the registration
 * @param pagesize  assumed page size of the registration
 * @return length for the new registration
 */
static inline uint64_t __calculate_length(
		uint64_t address,
		uint64_t length,
		uint64_t pagesize)
{
	uint64_t baseaddr = address & ~(pagesize - 1);
	uint64_t reg_len = (address + length) - baseaddr;
	uint64_t pages = reg_len / pagesize;

	if (reg_len % pagesize != 0)
		pages += 1;

	return pages * pagesize;
}

int gnix_mr_reg(struct fid *fid, const void *buf, size_t len,
		uint64_t access, uint64_t offset, uint64_t requested_key,
		uint64_t flags, struct fid_mr **mr_o, void *context)
{
	struct gnix_fid_mem_desc *mr = NULL;
	int fi_gnix_access = 0;
	struct gnix_fid_domain *domain;
	struct gnix_nic *nic;
	int rc;
	uint64_t reg_addr, reg_len;

	GNIX_TRACE(FI_LOG_MR, "\n");

	/* Flags are reserved for future use and must be 0. */
	if (unlikely(flags))
		return -FI_EBADFLAGS;

	/* The offset parameter is reserved for future use and must be 0.
	 *   Additionally, check for invalid pointers, bad access flags and the
	 *   correct fclass on associated fid
	 */
	if (offset || !buf || !mr_o || !access ||
			(access & ~(FI_READ | FI_WRITE | FI_RECV | FI_SEND |
						FI_REMOTE_READ |
						FI_REMOTE_WRITE)) ||
			(fid->fclass != FI_CLASS_DOMAIN))

		return -FI_EINVAL;

	domain = container_of(fid, struct gnix_fid_domain, domain_fid.fid);

	/* If network would be able to write to this buffer, use read-write */
	if (access & (FI_RECV | FI_READ | FI_REMOTE_WRITE))
		fi_gnix_access |= GNI_MEM_READWRITE;
	else
		fi_gnix_access |= GNI_MEM_READ_ONLY;

	/* If the nic list is empty, create a nic */
	if (unlikely(dlist_empty(&domain->nic_list))) {
		rc = gnix_nic_alloc(domain, NULL, &nic);
		if (rc) {
			GNIX_WARN(FI_LOG_MR, "could not allocate nic to do mr_reg,"
					" ret=%i\n", rc);
			goto err;
		}
	}

	reg_addr = ((uint64_t) buf) & ~((1 << GNIX_MR_PAGE_SHIFT) - 1);
	reg_len = __calculate_length((uint64_t) buf, len,
			1 << GNIX_MR_PAGE_SHIFT);

	/* call cache register op to retrieve the right entry */
	fastlock_acquire(&domain->mr_cache_lock);
	if (unlikely(!domain->mr_cache)) {
		rc = __gnix_mr_cache_init(&domain->mr_cache,
				&domain->mr_cache_attr);
		if (rc != FI_SUCCESS) {
			fastlock_release(&domain->mr_cache_lock);
			goto err;
		}
	}

	rc = __mr_cache_register(domain->mr_cache, domain,
			(uint64_t) reg_addr, reg_len, NULL,
			fi_gnix_access, -1, &mr);
	fastlock_release(&domain->mr_cache_lock);

	/* check retcode */
	if (unlikely(rc != FI_SUCCESS))
		goto err;

	/* md.domain */
	mr->domain = domain;
	_gnix_ref_get(mr->domain); /* take reference on domain */

	/* md.mr_fid */
	mr->mr_fid.mem_desc = mr;
	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = context;
	mr->mr_fid.fid.ops = &fi_gnix_mr_ops;

	/* nic */
	_gnix_ref_get(mr->nic); /* take reference on nic */

	/* setup internal key structure */
	mr->mr_fid.key = _gnix_convert_mhdl_to_key(&mr->mem_hndl);

	/* set up mr_o out pointer */
	*mr_o = &mr->mr_fid;
	return FI_SUCCESS;

err:
	return rc;
}

/**
 * Closes and deallocates a libfabric memory registration
 *
 * @param[in]  fid  libfabric memory registration fid
 *
 * @return     FI_SUCCESS on success
 *             -FI_EINVAL on invalid fid
 *             -FI_NOENT when there isn't a matching registration for the
 *               provided fid
 *             Otherwise, GNI_RC_* ret codes converted to FI_* err codes
 */
static int fi_gnix_mr_close(fid_t fid)
{
	struct gnix_fid_mem_desc *mr;
	gni_return_t ret;
	struct gnix_fid_domain *domain;
	struct gnix_nic *nic;

	GNIX_TRACE(FI_LOG_MR, "\n");

	if (unlikely(fid->fclass != FI_CLASS_MR))
		return -FI_EINVAL;

	mr = container_of(fid, struct gnix_fid_mem_desc, mr_fid.fid);

	domain = mr->domain;
	nic = mr->nic;

	/* call cache deregister op */
	fastlock_acquire(&domain->mr_cache_lock);
	ret = __mr_cache_deregister(mr->domain->mr_cache, mr);
	fastlock_release(&domain->mr_cache_lock);

	/* check retcode */
	if (likely(ret == FI_SUCCESS)) {
		/* release references to the domain and nic */
		_gnix_ref_put(domain);
		_gnix_ref_put(nic);
	} else {
		GNIX_WARN(FI_LOG_MR, "failed to deregister memory, "
				"ret=%i\n", ret);
	}

	return ret;
}

/**
 * Checks the sanity of cache attributes
 *
 * @param[in]   attr  attributes structure to be checked
 * @return      FI_SUCCESS if the attributes are valid
 *              -FI_EINVAL if the attributes are invalid
 */
static inline int __check_mr_cache_attr_sanity(gnix_mr_cache_attr_t *attr)
{
	/* 0 < attr->hard_reg_limit < attr->soft_reg_limit */
	if (attr->hard_reg_limit > 0 &&
			attr->hard_reg_limit < attr->soft_reg_limit)
		return -FI_EINVAL;

	/* valid otherwise */
	return FI_SUCCESS;
}

static int __gnix_mr_cache_init(
		gnix_mr_cache_t      **cache,
		gnix_mr_cache_attr_t *attr)
{
	gnix_mr_cache_attr_t *cache_attr = &__default_mr_cache_attr;
	gnix_mr_cache_t *cache_p;
	int rc;

	GNIX_TRACE(FI_LOG_MR, "\n");

	if (!cache)
		return -FI_EINVAL;

	/* if the provider asks us to use their attributes, are they sane? */
	if (attr) {
		if (__check_mr_cache_attr_sanity(attr) != FI_SUCCESS)
			return -FI_EINVAL;

		cache_attr = attr;
	}

	cache_p = (gnix_mr_cache_t *) calloc(1, sizeof(*cache_p));
	if (!cache_p)
		return -FI_ENOMEM;

	/* save the attribute values */
	memcpy(&cache_p->attr, cache_attr, sizeof(*cache_attr));

	/* list is used because entries can be removed from the stale list if
	 *   a user might call register on a stale entry's memory region
	 */
	dlist_init(&cache_p->lru_head);

	/* set up inuse tree */
	cache_p->inuse.rb_tree = rbtNew(__mr_cache_key_comp);
	if (!cache_p->inuse.rb_tree) {
		rc = -FI_ENOMEM;
		goto err_inuse;
	}

	/* if using lazy deregistration, set up stale tree */
	if (cache_p->attr.lazy_deregistration) {
		cache_p->stale.rb_tree = rbtNew(__mr_cache_key_comp);
		if (!cache_p->stale.rb_tree) {
			rc = -FI_ENOMEM;
			goto err_stale;
		}
	}

	/* initialize the element counts. If we are reinitializing a dead cache,
	 *   destroy will have already set the element counts
	 */
	if (cache_p->state == GNIX_MRC_STATE_UNINITIALIZED) {
		atomic_initialize(&cache_p->inuse.elements, 0);
		atomic_initialize(&cache_p->stale.elements, 0);
	}

	cache_p->hits = 0;
	cache_p->misses = 0;

	cache_p->state = GNIX_MRC_STATE_READY;
	*cache = cache_p;

	return FI_SUCCESS;

err_stale:
	rbtDelete(cache_p->inuse.rb_tree);
	cache_p->inuse.rb_tree = NULL;
err_inuse:
	free(cache_p);

	return rc;
}

int _gnix_mr_cache_destroy(gnix_mr_cache_t *cache)
{
	if (cache->state != GNIX_MRC_STATE_READY)
		return -FI_EINVAL;

	GNIX_TRACE(FI_LOG_MR, "\n");

	/*
	 * Remove all of the stale entries from the cache
	 */
	_gnix_mr_cache_flush(cache);

	/*
	 * if there are still elements in the cache after the flush,
	 *   then someone forgot to deregister memory.
	 *   We probably shouldn't destroy the cache at this point.
	 */
	if (atomic_get(&cache->inuse.elements) != 0) {
		return -FI_EAGAIN;
	}

	/* destroy the tree */
	rbtDelete(cache->inuse.rb_tree);
	cache->inuse.rb_tree = NULL;

	/* stale will been flushed already, so just destroy the tree */
	if (cache->attr.lazy_deregistration) {
		rbtDelete(cache->stale.rb_tree);
		cache->stale.rb_tree = NULL;
	}

	cache->state = GNIX_MRC_STATE_DEAD;
	free(cache);

	return FI_SUCCESS;
}

int __mr_cache_flush(gnix_mr_cache_t *cache, int flush_count) {
	int rc;
	RbtIterator iter;
	gnix_mr_cache_key_t *e_key;
	gnix_mr_cache_entry_t *entry, *e_entry;
	int destroyed = 0;

	GNIX_TRACE(FI_LOG_MR, "\n");

	GNIX_INFO(FI_LOG_MR, "starting flush on memory registration cache\n");

	/* flushes are unnecessary for caches without lazy deregistration */
	if (!cache->attr.lazy_deregistration)
		return FI_SUCCESS;

	while (!dlist_empty(&cache->lru_head)) {

		if (flush_count >= 0 && flush_count == destroyed)
			break;

		rc = __mr_cache_lru_dequeue(cache, &entry);
		if (unlikely(rc != FI_SUCCESS)) {
			GNIX_ERR(FI_LOG_MR,
					"list may be corrupt, no entries from lru pop\n");
			break;
		}

		GNIX_INFO(FI_LOG_MR, "attempting to flush key %llx:%llx\n",
				entry->key.address, entry->key.length);
		iter = rbtFind(cache->stale.rb_tree, &entry->key);
		if (unlikely(!iter)) {
			GNIX_ERR(FI_LOG_MR,
					"lru entries MUST be present in the cache,"
					" could not find key in stale tree\n");
			break;
		}

		rbtKeyValue(cache->stale.rb_tree, iter, (void **) &e_key,
			    (void **) &e_entry);
		if (e_entry != entry) {
			/* If not an exact match, remove the found entry,
			 and then put the original entry back on the LRU list */
			GNIX_INFO(FI_LOG_MR,
				  "Flushing non-lru entry %llx:%llx\n",
				  e_entry->key.address, e_entry->key.length);
			dlist_remove(&e_entry->lru_entry);
			dlist_insert_tail(&entry->lru_entry, &cache->lru_head);
			/* Destroy the actual entry below */
			entry = e_entry;
		}

		rc = rbtErase(cache->stale.rb_tree, iter);
		if (unlikely(rc != RBT_STATUS_OK)) {
			GNIX_ERR(FI_LOG_MR,
					"failed to erase lru entry from stale tree\n");
			break;
		}

		__mr_cache_entry_destroy(entry);
		entry = NULL;
		++destroyed;
	}

	GNIX_INFO(FI_LOG_MR, "flushed %i of %i entries from memory "
				"registration cache\n", destroyed,
				atomic_get(&cache->stale.elements));

	if (destroyed > 0) {
		atomic_sub(&cache->stale.elements, destroyed);
	}

	return FI_SUCCESS;
}

int _gnix_mr_cache_flush(gnix_mr_cache_t *cache)
{

	if (unlikely(cache->state != GNIX_MRC_STATE_READY))
		return -FI_EINVAL;

	__mr_cache_flush(cache, cache->attr.hard_reg_limit);

	return FI_SUCCESS;
}

static int __mr_cache_search_inuse(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		gnix_mr_cache_entry_t    **entry,
		gnix_mr_cache_key_t      *key)
{
	int ret = FI_SUCCESS, cmp;
	RbtIterator iter;
	gnix_mr_cache_key_t *found_key, new_key;
	gnix_mr_cache_entry_t *found_entry;
	uint64_t new_end, found_end;
	DLIST_HEAD(retired_entries);

	/* first we need to find an entry that overlaps with this one.
	 * care should be taken to find the left most entry that overlaps
	 * with this entry since the entry we are searching for might overlap
	 * many entries and the tree may be left or right balanced
	 * at the head
	 */
	iter = rbtFindLeftmost(cache->inuse.rb_tree, (void *) key,
			__find_overlapping_addr);
	if (!iter) {
		GNIX_INFO(FI_LOG_MR,
				"could not find key in inuse, key=%llx:%llx\n",
				key->address, key->length);
		return -FI_ENOENT;
	}

	rbtKeyValue(cache->inuse.rb_tree, iter, (void **) &found_key,
			(void **) &found_entry);

	GNIX_INFO(FI_LOG_MR,
			"found a key that matches the search criteria, "
			"found=%llx:%llx key=%llx:%llx\n",
			found_key->address, found_key->length,
			key->address, key->length);

	/* if the entry that we've found completely subsumes
	 * the requested entry, just return a reference to
	 * that existing registration
	 */
	if (__can_subsume(found_key, key)) {
		GNIX_INFO(FI_LOG_MR,
				"found an entry that subsumes the request, "
				"existing=%llx:%llx key=%llx:%llx\n",
				found_key->address, found_key->length,
				key->address, key->length);
		*entry = found_entry;
		__mr_cache_entry_get(cache, found_entry);

		cache->hits++;
		return FI_SUCCESS;
	}

	/* otherwise, iterate from the existing entry until we can no longer
	 * find an entry that overlaps with the new registration and remove
	 * and retire each of those entries.
	 */
	new_key.address = MIN(found_key->address, key->address);
	new_end = key->address + key->length;
	while (iter) {
		rbtKeyValue(cache->inuse.rb_tree, iter, (void **) &found_key,
				(void **) &found_entry);


		cmp = __find_overlapping_addr(found_key, key);
		GNIX_INFO(FI_LOG_MR,
				"candidate: key=%llx:%llx result=%d\n",
				found_key->address,
				found_key->length, cmp);
		if (cmp != 0)
			break;

		/* compute new ending address */
		found_end = found_key->address + found_key->length;

		/* mark the entry as retired */
		GNIX_INFO(FI_LOG_MR, "retiring entry, key=%llx:%llx\n",
				found_key->address, found_key->length);
		found_entry->flags |= GNIX_CE_RETIRED;
		dlist_insert_tail(&found_entry->siblings, &retired_entries);

		iter = rbtNext(cache->inuse.rb_tree, iter);
	}
	/* Since our new key might fully overlap every other entry in the tree,
	 * we need to take the maximum of the last entry and the new entry
	 */
	new_key.length = MAX(found_end, new_end) - new_key.address;


	/* remove retired entries from tree */
	GNIX_INFO(FI_LOG_MR, "removing retired entries from inuse tree\n");
	__remove_sibling_entries_from_tree(cache,
			&retired_entries, cache->inuse.rb_tree);

	/* create new registration */
	GNIX_INFO(FI_LOG_MR,
			"creating a new merged registration, key=%llx:%llx\n",
			new_key.address, new_key.length);
	ret = __mr_cache_create_registration(cache, domain,
			new_key.address, new_key.length, dst_cq_hndl, flags,
			vmdh_index, entry, &new_key);
	if (ret) {
		GNIX_ERR(FI_LOG_MR, "failure in registration, can't really "
				"recover at this point since we've already retired "
				"the existing entries\n");
		return ret;
	}

	/* move retired entries to the head of the new entry's child list */
	if (!dlist_empty(&retired_entries)) {
		__attach_retired_entries_to_registration(cache,
				&retired_entries, *entry);
	}

	cache->misses++;

	return ret;
}

static int __mr_cache_search_stale(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		gnix_mr_cache_entry_t    **entry,
		gnix_mr_cache_key_t      *key)
{
	int ret;
	RbtStatus rc;
	RbtIterator iter;
	gnix_mr_cache_key_t *mr_key;
	gnix_mr_cache_entry_t *mr_entry, *tmp;

	GNIX_INFO(FI_LOG_MR, "searching for stale entry, key=%llx:%llx\n",
			key->address, key->length);

	iter = rbtFindLeftmost(cache->stale.rb_tree, (void *) key,
			__find_overlapping_addr);
	if (!iter)
		return -FI_ENOENT;

	rbtKeyValue(cache->stale.rb_tree, iter, (void **) &mr_key,
			(void **) &mr_entry);

	GNIX_INFO(FI_LOG_MR,
			"found a matching entry, found=%llx:%llx key=%llx:%llx\n",
			mr_key->address, mr_key->length,
			key->address, key->length);


	/* if the entry that we've found completely subsumes
	 * the requested entry, just return a reference to
	 * that existing registration
	 */
	if (__can_subsume(mr_key, key)) {
		ret = __mr_cache_search_inuse(cache, domain, address,
				length, dst_cq_hndl, flags, vmdh_index,
				&tmp, mr_key);
		if (ret == FI_SUCCESS) {
			/* if we found an entry in the inuse tree
			 * in this manner, it means that there was
			 * an entry either overlapping or contiguous
			 * with the stale entry in the inuse tree, and
			 * a new entry has been made and saved to tmp.
			 * The old entry (mr_entry) should be destroyed
			 * now as it is no longer needed.
			 */
			GNIX_INFO(FI_LOG_MR,
					"removing entry from stale key=%llx:%llx\n",
					mr_key->address, mr_key->length);

			rc = rbtErase(cache->stale.rb_tree, iter);
			if (unlikely(rc != RBT_STATUS_OK)) {
				GNIX_ERR(FI_LOG_MR,
						"failed to erase entry from stale tree\n");
			} else {
				dlist_remove(&mr_entry->lru_entry);

				atomic_dec(&cache->stale.elements);

				__mr_cache_entry_destroy(mr_entry);
			}

			*entry = tmp;
		} else {
			GNIX_INFO(FI_LOG_MR,
					"removing entry from stale and migrating to inuse, "
					"key=%llx:%llx\n",
					mr_key->address, mr_key->length);
			rc = rbtErase(cache->stale.rb_tree, iter);
			if (unlikely(rc != RBT_STATUS_OK))
				GNIX_WARN(FI_LOG_MR,
						"failed to erase entry from stale tree\n");
			assert(rc == RBT_STATUS_OK);

			dlist_remove(&mr_entry->lru_entry);

			atomic_dec(&cache->stale.elements);

			/* if we made it to this point, there weren't
			 * any entries in the inuse tree that would
			 * have overlapped with this entry
			 */
			rc = rbtInsert(cache->inuse.rb_tree,
					&mr_entry->key, mr_entry);
			if (unlikely(rc != RBT_STATUS_OK))
				GNIX_WARN(FI_LOG_MR,
						"failed to insert entry into inuse tree\n");
			assert(rc == RBT_STATUS_OK);

			atomic_set(&mr_entry->ref_cnt, 1);
			atomic_inc(&cache->inuse.elements);

			*entry = mr_entry;
		}

		return FI_SUCCESS;
	}

	GNIX_INFO(FI_LOG_MR,
			"could not use matching entry, "
			"found=%llx:%llx\n",
			mr_key->address, mr_key->length);

	return -FI_ENOENT;
}

static int __mr_cache_create_registration(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		gnix_mr_cache_entry_t    **entry,
		gnix_mr_cache_key_t      *key)
{
	int rc;
	struct gnix_nic *nic;
	gni_return_t grc = GNI_RC_SUCCESS;
	gnix_mr_cache_entry_t *current_entry;

	/* if we made it here, we didn't find the entry at all */
	current_entry = calloc(1, sizeof(*current_entry));
	if (!current_entry)
		return -FI_ENOMEM;

	dlist_init(&current_entry->lru_entry);
	dlist_init(&current_entry->children);
	dlist_init(&current_entry->siblings);

	dlist_for_each(&domain->nic_list, nic, dom_nic_list)
	{
		fastlock_acquire(&nic->lock);
		grc = GNI_MemRegister(nic->gni_nic_hndl, address, length,
					dst_cq_hndl, flags,
					vmdh_index, &current_entry->mem_hndl);
		fastlock_release(&nic->lock);
		if (grc == GNI_RC_SUCCESS)
			break;
	}

	if (unlikely(grc != GNI_RC_SUCCESS)) {
		free(current_entry);
		GNIX_WARN(FI_LOG_MR, "failed to register memory with uGNI, "
				"ret=%s\n", gni_err_str[grc]);
		return -gnixu_to_fi_errno(grc);
	}

	/* set up the entry's key */
	current_entry->key.address = address;
	current_entry->key.length = length;
	current_entry->flags = 0;

	rc = rbtInsert(cache->inuse.rb_tree, &current_entry->key,
			current_entry);
	if (unlikely(rc != RBT_STATUS_OK)) {
		GNIX_ERR(FI_LOG_MR, "failed to insert registration "
				"into cache, ret=%i\n", rc);

		fastlock_acquire(&nic->lock);
		grc = GNI_MemDeregister(nic->gni_nic_hndl,
				&current_entry->mem_hndl);
		fastlock_release(&nic->lock);
		if (unlikely(grc != GNI_RC_SUCCESS)) {
			GNIX_WARN(FI_LOG_MR,
					"failed to deregister memory with "
					"uGNI, ret=%s\n",
					gni_err_str[grc]);
		}

		free(current_entry);

		return -FI_ENOMEM;
	}

	GNIX_INFO(FI_LOG_MR, "inserted key %llx:%llx into inuse\n",
			current_entry->key.address, current_entry->key.length);


	atomic_inc(&cache->inuse.elements);
	atomic_initialize(&current_entry->ref_cnt, 1);
	current_entry->domain = domain;
	current_entry->nic = nic;

	/* take references on domain and nic */
	_gnix_ref_get(current_entry->domain);
	_gnix_ref_get(current_entry->nic);

	*entry = current_entry;

	return FI_SUCCESS;
}

/**
 * Function to register memory with the cache
 *
 * @param[in] cache        gnix memory registration cache pointer
 * @param[in] mr           gnix memory region descriptor pointer
 * @param[in] domain       gnix domain pointer
 * @param[in] address      base address of the memory region to be registered
 * @param[in] length       length of the memory region to be registered
 * @param[in] dst_cq_hndl  destination gni cq handle for cq event delivery
 * @param[in] flags        gni memory registration flags
 * @param[in] vmdh_index   desired index for the new vmdh
 * @param[in,out] mem_hndl gni memory handle pointer to written to and returned
 */
static int __mr_cache_register(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_domain   *domain,
		uint64_t                 address,
		uint64_t                 length,
		gni_cq_handle_t          dst_cq_hndl,
		uint32_t                 flags,
		uint32_t                 vmdh_index,
		struct gnix_fid_mem_desc **mr)
{
	int ret;
	gnix_mr_cache_key_t key;
	gnix_mr_cache_entry_t *entry;

	GNIX_TRACE(FI_LOG_MR, "\n");

	/* build key for searching */
	key.address = address;
	key.length = length;

	/* fastpath inuse */
	ret = __mr_cache_search_inuse(cache, domain,
			address, length, dst_cq_hndl, flags,
			vmdh_index, &entry, &key);
	if (ret == FI_SUCCESS)
		goto success;

	/* if we shouldn't introduce any new elements, return -FI_ENOSPC */
	if (unlikely(cache->attr.hard_reg_limit > 0 &&
			(atomic_get(&cache->inuse.elements) >=
					cache->attr.hard_reg_limit)))
		return -FI_ENOSPC;

	if (cache->attr.lazy_deregistration) {
		/* if lazy deregistration is in use, we can check the
		 *   stale tree
		 */
		ret = __mr_cache_search_stale(cache, domain,
				address, length, dst_cq_hndl, flags,
				vmdh_index, &entry, &key);
		if (ret == FI_SUCCESS) {
			cache->hits++;
			goto success;
		}
	}

	/* If the cache is full, then flush one of the stale entries to make
	 *   room for the new entry. This works because we check above to see if
	 *   the number of inuse entries exceeds the hard reg limit
	 */
	if ((atomic_get(&cache->inuse.elements) +
			atomic_get(&cache->stale.elements)) == cache->attr.hard_reg_limit)
		__mr_cache_flush(cache, 1);

	ret = __mr_cache_create_registration(cache, domain,
			address, length, dst_cq_hndl, flags,
			vmdh_index, &entry, &key);
	if (ret)
		return ret;

	cache->misses++;

success:
	entry->state = GNIX_CES_INUSE;
	*mr = &entry->mr;

	(*mr)->nic = entry->nic;
	(*mr)->key.address = entry->key.address;
	(*mr)->key.length = entry->key.length;
	(*mr)->mem_hndl = entry->mem_hndl;

	return FI_SUCCESS;
}

/**
 * Function to deregister memory in the cache
 *
 * @param[in]  cache  gnix memory registration cache pointer
 * @param[in]  mr     gnix memory registration descriptor pointer
 *
 * @return     FI_SUCCESS on success
 *             -FI_ENOENT if there isn't an active memory registration
 *               associated with the mr
 *             GNI_RC_* return codes for potential calls to GNI_MemDeregister
 */
static int __mr_cache_deregister(
		gnix_mr_cache_t          *cache,
		struct gnix_fid_mem_desc *mr)
{
	gnix_mr_cache_entry_t *entry;
	gni_return_t grc;

	GNIX_TRACE(FI_LOG_MR, "\n");

	/* check to see if we can find the entry so that we can drop the
	 *   held reference
	 */

	entry = container_of(mr, gnix_mr_cache_entry_t, mr);
	if (entry->state != GNIX_CES_INUSE)
		return -FI_EINVAL;

	GNIX_INFO(FI_LOG_MR, "entry found, entry=%p refs=%d\n",
			entry, atomic_get(&entry->ref_cnt));

	grc = __mr_cache_entry_put(cache, entry);

	/* Since we check this on each deregistration, the amount of elements
	 * over the limit should always be 1
	 */
	if (atomic_get(&cache->stale.elements) > cache->attr.hard_stale_limit)
		__mr_cache_flush(cache, 1);

	return gnixu_to_fi_errno(grc);
}
