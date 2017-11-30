/*
 * Copyright (c) 2016-2017 Cray Inc. All rights reserved.
 * Copyright (c) 2017 Intel Inc. All rights reserved.
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

#include <fi_util.h>
#include <inttypes.h>

/* This macro allows to retrive a provider from the ofi_mr_cache structure */
#define UTIL_MR_CACHE_GET_PROV(cache)	((cache)->domain->prov)

#define UTIL_MR_CACHE_IS_LAZY_DEREG(cache)	((cache)->stale_size > 0)

#if ENABLE_DEBUG
#define UTIL_MR_CACHE_INC_HITS(cache)	((cache)->hits++)
#define UTIL_MR_CACHE_INC_MISSES(cache)	((cache)->misses++)
#else
#define UTIL_MR_CACHE_INC_HITS(cache)	do {} while (0)
#define UTIL_MR_CACHE_INC_MISSES(cache)	do {} while (0)
#endif

#define UTIL_MR_CACHE_DEFINE_ENTRY_FLAG_FUNCS(flag)				\
static inline									\
int util_mr_cache_entry_is_ ## flag (struct ofi_mr_cache_entry *entry)		\
{										\
	return entry->flags.is_ ## flag;					\
}										\
static inline									\
void util_mr_cache_entry_set_ ## flag (struct ofi_mr_cache_entry *entry)	\
{										\
	entry->flags.is_ ## flag = 1;						\
}

UTIL_MR_CACHE_DEFINE_ENTRY_FLAG_FUNCS(retired);
UTIL_MR_CACHE_DEFINE_ENTRY_FLAG_FUNCS(merged);
UTIL_MR_CACHE_DEFINE_ENTRY_FLAG_FUNCS(unmapped);

static inline int util_mr_cache_entry_put(struct ofi_mr_cache *cache,
					  struct ofi_mr_cache_entry *entry);
static inline int util_mr_cache_entry_get(struct ofi_mr_cache *cache,
					  struct ofi_mr_cache_entry *entry);

static inline int util_mr_cache_entry_destroy(struct ofi_mr_cache *cache,
					      struct ofi_mr_cache_entry *entry);

static int util_mr_cache_create_registration(struct ofi_mr_cache *cache,
					     struct ofi_mr_region *region,
					     struct ofi_mr_reg_attr *reg_attr,
					     struct ofi_mr_cache_entry **entry);

static int util_mr_cache_check_overlapping_region(void *x, void *y)
{
	struct ofi_mr_region *to_find = (struct ofi_mr_region *)x;
	struct ofi_mr_region *to_compare = (struct ofi_mr_region *)y;
	uint64_t to_find_end = to_find->address + to_find->length - 1;
	uint64_t to_compare_end = to_compare->address + to_compare->length - 1;

	if (!((to_find_end < to_compare->address) ||
	      (to_compare_end < to_find->address)))
		return 0;

	/* left */
	if (to_find->address < to_compare->address)
		return -1;

	return 1;
}

static inline int util_mr_cache_comp_regions(void *x, void *y)
{
	struct ofi_mr_region *to_insert  = (struct ofi_mr_region *)x;
	struct ofi_mr_region *to_compare = (struct ofi_mr_region *)y;

	if (to_compare->address == to_insert->address)
		return 0;

	/* to the left */
	if (to_insert->address < to_compare->address)
		return -1;

	/* to the right */
	return 1;
}

static inline int util_mr_cache_can_subsume(struct ofi_mr_region *x,
					    struct ofi_mr_region *y)
{
	return ((x->address <= y->address) &&
		((x->address + x->length) >= (y->address + y->length)));
}

static inline void
util_mr_cache_attach_retired_entries_to_registration(struct ofi_mr_cache *cache,
						     struct dlist_entry *retired_entries,
						     struct ofi_mr_cache_entry *parent)
{
	struct ofi_mr_cache_entry *entry;
	struct dlist_entry *tmp;

	dlist_foreach_container_safe(retired_entries,
				     struct ofi_mr_cache_entry,
				     entry, siblings, tmp) {
		dlist_remove(&entry->siblings);
		dlist_insert_tail(&entry->siblings,
				  &parent->children);
		if (!dlist_empty(&entry->children)) {
			/* move the entry's children to the sibling tree
			 * and decrement the reference count */
			dlist_splice_tail(&parent->children,
					  &entry->children);
			util_mr_cache_entry_put(cache, entry);
		}
	}

	if (!dlist_empty(retired_entries))
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"retired_entries not empty\n");

	util_mr_cache_entry_get(cache, parent);
}

static inline void
util_mr_cache_remove_sibling_entries_from_tree(struct ofi_mr_cache *cache,
					       struct dlist_entry *list,
					       RbtHandle tree)
{
	RbtStatus rc;
	RbtIterator iter;
	struct ofi_mr_cache_entry *entry;

	dlist_foreach_container(list, struct ofi_mr_cache_entry,
				entry, siblings) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "removing region from tree, region=%"PRIu64":%"PRIu64"\n",
		       entry->region.address, entry->region.length);
		iter = rbtFind(tree, &entry->region);
		if (OFI_UNLIKELY(!iter))
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"region not found\n");

		rc = rbtErase(tree, iter);
		if (OFI_UNLIKELY(rc != RBT_STATUS_OK))
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"could not remove entry from tree\n");
	}
}

static inline void util_mr_cache_lru_enqueue(struct ofi_mr_cache *cache,
					     struct ofi_mr_cache_entry *entry)
{
	dlist_insert_tail(&entry->lru_entry, &cache->lru_list);
}

static inline int util_mr_cache_lru_dequeue(struct ofi_mr_cache *cache,
					    struct ofi_mr_cache_entry **entry)
{
	if (OFI_UNLIKELY(dlist_empty(&cache->lru_list))) {
		*entry = NULL;
		return -FI_ENOENT;
	}
	/* Takes the first entry from the LRU */
	dlist_pop_front(&cache->lru_list, struct ofi_mr_cache_entry,
			*entry, lru_entry);

	return FI_SUCCESS;
}

static inline void util_mr_cache_lru_remove(struct ofi_mr_cache *cache,
					    struct ofi_mr_cache_entry *entry)
{
	dlist_remove(&entry->lru_entry);
}

static int util_mr_cache_notifier_warned = 0;
static void
util_mr_cache_clear_notifier_events(struct ofi_mr_cache *cache)
{
	int ret = -FI_EAGAIN;
	struct ofi_mr_cache_entry *entry;
	struct ofi_subscription *subscription;
	RbtIterator iter;

	if (!UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		return;

	while ((subscription = ofi_monitor_get_event(&cache->mem_nq))) {
		entry = container_of(subscription, struct ofi_mr_cache_entry,
				     subscription);
		if (ofi_atomic_get32(&entry->ref_cnt) != 0) {
			/* First, warn that this might be a problem.*/
			if (!util_mr_cache_notifier_warned &&
			    !util_mr_cache_entry_is_merged(entry)) {
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"Registered memory region includes unmapped pages."
					" Have you freed memory w/o closing the memory region?\n");
				util_mr_cache_notifier_warned = 1;
			}

			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "Marking unmapped entry (%p) as retired %"PRIu64":%"PRIu64"\n",
			       entry, entry->region.address, entry->region.length);

			util_mr_cache_entry_set_unmapped(entry);

			if (util_mr_cache_entry_is_retired(entry))
				/* Nothing to do */
				break;

			/* Retire this entry (remove from inuse tree) */
			util_mr_cache_entry_set_retired(entry);
			iter = rbtFind(cache->inuse_tree, &entry->region);
			if (OFI_LIKELY(iter != NULL)) {
				ret = rbtErase(cache->inuse_tree, iter);
				if (OFI_UNLIKELY(ret != RBT_STATUS_OK)) {
					FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
						"Unmapped entry could not be removed from in usetree.\n");
					abort();
				}
			} else {
				/*  The only way we should get here is if we're in the
				 *  middle of retiring this entry.  Not sure if this
				 *  is worth a separate warning from the one above.
				 */
			}

			break;
		} else {
			util_mr_cache_entry_set_unmapped(entry);
			iter = rbtFind(cache->stale_tree, &entry->region);
			if (!iter)
				break;

			ret = rbtErase(cache->stale_tree, iter);
			if (OFI_UNLIKELY(ret != RBT_STATUS_OK)) {
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"Unmapped entry could not be removed from stale tree.\n");
				abort();
			}

			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "Removed unmapped entry (%p) from stale tree %"PRIu64":%"PRIu64"\n",
			       entry, entry->region.address, entry->region.length);

			util_mr_cache_lru_remove(cache, entry);
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "Removed unmapped entry (%p) from lru list %"PRIu64":%"PRIu64"\n",
			       entry, entry->region.address, entry->region.length);
			cache->stale_elem--;

			util_mr_cache_entry_destroy(cache, entry);

			break;
		}
	}
	if (OFI_UNLIKELY(ret != -FI_EAGAIN))
		/* Should we do something else here? */
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"`get event` returned error: %s\n",
			fi_strerror(-ret));

	return;
}

static int
util_mr_cache_notifier_monitor(struct ofi_mr_cache *cache,
			       struct ofi_mr_cache_entry *entry)
{

	if (!UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		return FI_SUCCESS;

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "monitoring entry=%p %"PRIu64":%"PRIu64"\n",
	       entry, entry->region.address, entry->region.length);

	return ofi_monitor_subscribe(&cache->mem_nq,
				     (void *)entry->region.address,
				     entry->region.length,
				     &entry->subscription);
}

static void
util_mr_cache_notifier_unmonitor(struct ofi_mr_cache *cache,
				 struct ofi_mr_cache_entry *entry)
{

	if (!UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		return;

	util_mr_cache_clear_notifier_events(cache);

	if (!util_mr_cache_entry_is_unmapped(entry)) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "unmonitoring entry=%p (ref_cnt=%d)\n",
		       entry, ofi_atomic_get32(&entry->ref_cnt));
		ofi_monitor_unsubscribe((void *)entry->region.address,
					entry->region.length,
					&entry->subscription);
	}
}

static inline int
util_mr_cache_entry_destroy(struct ofi_mr_cache *cache,
			    struct ofi_mr_cache_entry *entry)
{
	int ret;

	ret = cache->dereg_callback(cache, entry);
	if (!ret) {
		if (!util_mr_cache_entry_is_unmapped(entry))
			util_mr_cache_notifier_unmonitor(cache, entry);

		ofi_freealign(entry);
	} else {
		FI_INFO(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"failed to deregister memory region with callback, cache_entry=%p ret=%i\n",
			entry, ret);
	}

	return ret;
}

static inline
int util_mr_cache_insert_entry_into_stale(struct ofi_mr_cache *cache,
					  struct ofi_mr_cache_entry *entry)
{
	RbtStatus rc;
	int ret = FI_SUCCESS;

	if (util_mr_cache_entry_is_unmapped(entry)) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "entry (%p) unmapped, not inserting into stale %"PRIu64":%"PRIu64"\n",
		       entry, entry->region.address, entry->region.length);
		/* Should we return some other value? */
		return ret;
	}

	rc = rbtInsert(cache->stale_tree, &entry->region, entry);
	if (rc != RBT_STATUS_OK) {
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"could not insert into stale rb tree,"
			" rc=%d region.address=%"PRIu64" region.length=%"PRIu64" entry=%p\n",
			rc, entry->region.address, entry->region.length, entry);

		ret = util_mr_cache_entry_destroy(cache, entry);
	} else {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "inserted region=%"PRIu64":%"PRIu64" into stale\n",
		       entry->region.address, entry->region.length);

		assert(ofi_atomic_get32(&entry->ref_cnt) == 0);

		util_mr_cache_lru_enqueue(cache, entry);
		cache->stale_elem++;
	}

	return ret;
}

static inline
void util_mr_cache_resolve_stale_entry_collision(struct ofi_mr_cache *cache,
					         RbtIterator found,
						 struct ofi_mr_cache_entry *entry)
{
	RbtStatus rc;
	struct ofi_mr_cache_entry *c_entry;
	struct dlist_entry *tmp;
	struct ofi_mr_region *c_region;
	int ret;
	DEFINE_LIST(to_destroy);
	RbtIterator iter = found;
	int add_new_entry = 1, cmp;

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "resolving collisions with entry (%p) %"PRIu64":%"PRIu64"\n",
	       entry, entry->region.address, entry->region.length);

	while (iter) {
		rbtKeyValue(cache->stale_tree, iter, (void **)&c_region,
			    (void **)&c_entry);

		cmp = util_mr_cache_check_overlapping_region(&entry->region, c_region);
		if (cmp != 0)
			break;

		if (util_mr_cache_can_subsume(&entry->region, c_region) ||
		    (entry->region.length > c_region->length)) {
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "adding stale entry (%p) to destroy list,"
			       " region=%"PRIu64":%"PRIu64"\n", c_entry,
			       c_region->address, c_region->length);
			dlist_insert_tail(&c_entry->siblings, &to_destroy);
		} else {
			add_new_entry = 0;
		}

		iter = rbtNext(cache->stale_tree, iter);
	}

	/* TODO I can probably do this in a single sweep, avoiding a second
	 * pass and incurring n lg n removal time
	 */
	dlist_foreach_container_safe(&to_destroy, struct ofi_mr_cache_entry,
				     c_entry, siblings, tmp) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "removing region from tree, entry %p region=%"PRIu64":%"PRIu64"\n",
		       c_entry, c_entry->region.address, c_entry->region.length);
		iter = rbtFind(cache->stale_tree, &c_entry->region);
		if (OFI_UNLIKELY(!iter))
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR, "region not found\n");

		rc = rbtErase(cache->stale_tree,
			      iter);
		if (OFI_UNLIKELY(rc != RBT_STATUS_OK))
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"could not remove entry from tree\n");

		util_mr_cache_lru_remove(cache, c_entry);
		cache->stale_elem--;
		dlist_remove(&c_entry->siblings);
		util_mr_cache_entry_destroy(cache, c_entry);
	}
	if (OFI_UNLIKELY(!dlist_empty(&to_destroy)))
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"to_destroy not empty\n");

	if (add_new_entry) {
		ret = util_mr_cache_insert_entry_into_stale(cache, entry);
		if (ret)
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"Failed to insert subsumed MR entry (%p) into stale list\n",
				entry);
	} else {
		/* stale entry is larger than this one
		 * so lets just toss this entry out
		 */
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "larger entry already exists, to_destroy=%"PRIu64":%"PRIu64"\n",
		       entry->region.address, entry->region.length);

		ret = util_mr_cache_entry_destroy(cache, entry);
		if (ret)
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"failed to destroy a registration, entry=%p ret=%d\n",
				c_entry, ret);
	}
}

static inline int
util_mr_cache_entry_get(struct ofi_mr_cache *cache,
			struct ofi_mr_cache_entry *entry)
{
	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "Up ref cnt on entry %p\n", entry);
	return ofi_atomic_inc32(&entry->ref_cnt);
}

static inline
int util_mr_cache_entry_put(struct ofi_mr_cache *cache,
			    struct ofi_mr_cache_entry *entry)
{
	RbtIterator iter;
	int rc;
	int ret = FI_SUCCESS;
	RbtIterator found;
	struct ofi_mr_cache_entry *parent = NULL;
	struct dlist_entry *next;

	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		util_mr_cache_clear_notifier_events(cache);

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "Decrease ref cnt on entry %p\n", entry);

	if (ofi_atomic_dec32(&entry->ref_cnt) == 0) {
		next = entry->siblings.next;
		dlist_remove(&entry->children);
		dlist_remove(&entry->siblings);

		/* if this is the last child to deallocate,
		 * release the reference to the parent
		 */
		if (next != &entry->siblings && dlist_empty(next)) {
			parent = container_of(next, struct ofi_mr_cache_entry,
					      children);
			ret = util_mr_cache_entry_put(cache, parent);
			if (OFI_UNLIKELY(ret))
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"failed to release reference to parent, "
					"parent=%p refs=%d\n",
					parent, ofi_atomic_get32(&parent->ref_cnt));
		}

		cache->inuse_elem--;

		if (!util_mr_cache_entry_is_retired(entry)) {
			iter = rbtFind(cache->inuse_tree, &entry->region);
			if (OFI_UNLIKELY(!iter)) {
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"failed to find entry in the inuse cache\n");
			} else {
				rc = rbtErase(cache->inuse_tree, iter);
				if (OFI_UNLIKELY(rc != RBT_STATUS_OK))
					FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
						"failed to erase lru entry from stale tree\n");
			}
		}

		/* if we are doing lazy deregistration (stale size > 0) and the entry
		 * isn't retired, put it in the stale cache
		 */
		if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache) &&
		    !(util_mr_cache_entry_is_retired(entry))) {
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "moving region %"PRIu64":%"PRIu64" to stale (entry %p)\n",
			       entry->region.address, entry->region.length, entry);

			found = rbtFindLeftmost(cache->stale_tree,
						&entry->region,
						util_mr_cache_check_overlapping_region);
			if (found) {
				/* one or more stale entries would overlap with this
				 * new entry. We need to resolve these collisions by dropping
				 * registrations
				 */
				util_mr_cache_resolve_stale_entry_collision(cache, found, entry);
			} else {
				/* if not found, ... */
				ret = util_mr_cache_insert_entry_into_stale(cache, entry);
			}
		} else {
			/* if retired or not using lazy registration (stale size == 0) */
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "destroying entry, region=%"PRIu64":%"PRIu64" (entry %p)\n",
			       entry->region.address, entry->region.length, entry);

			ret = util_mr_cache_entry_destroy(cache, entry);
		}

		if (OFI_UNLIKELY(ret))
			FI_INFO(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"dereg callback returned %d\n", ret);
	}


	return ret;
}

int ofi_mr_cache_init(struct ofi_mr_cache *cache,
		      struct util_domain *domain,
		      struct ofi_mem_monitor *monitor)
{
	int ret;

	assert(cache && (cache->reg_size > 0) && (cache->stale_size >= 0));
	assert(cache->reg_callback && cache->dereg_callback);

	cache->domain = domain;

	/* list is used because entries can be removed from the stale list if
	 * a user might call register on a stale entry's memory region
	 */
	dlist_init(&cache->lru_list);

	/* set up inuse tree */
	cache->inuse_tree = rbtNew(util_mr_cache_comp_regions);
	if (!cache->inuse_tree)
		return -FI_ENOMEM;

	/* if using lazy deregistration (stale size > 0), set up stale tree */
	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache)) {
		assert(monitor);
		ofi_monitor_add_queue(monitor, &cache->mem_nq);
		cache->stale_tree = rbtNew(util_mr_cache_comp_regions);
		if (!cache->stale_tree) {
			ret = -FI_ENOMEM;
			goto err;
		}
	}

	return FI_SUCCESS;

err:
	rbtDelete(cache->inuse_tree);
	return ret;
}

void ofi_mr_cache_cleanup(struct ofi_mr_cache *cache)
{
	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR, "\n");

	/*
	 * Remove all of the stale entries from the cache
	 */
	ofi_mr_cache_flush(cache);

	/*
	 * if there are still elements in the cache after the flush,
	 * then someone forgot to deregister memory.
	 * We probably shouldn't destroy the cache at this point.
	 */
	assert(!cache->inuse_elem);
	if (cache->inuse_elem) {
		FI_INFO(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"There are some element in use. "
			"Cleanups haven't been done\n");
	}

	/* destroy the tree */
	rbtDelete(cache->inuse_tree);
	cache->inuse_tree = NULL;

	/* stale will been flushed already, so just destroy the tree */
	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache)) {
		ofi_monitor_del_queue(&cache->mem_nq);
		rbtDelete(cache->stale_tree);
		cache->stale_tree = NULL;
	}

	FI_INFO(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		"All cleanups have been done\n");
}

static void util_mr_cache_flush(struct ofi_mr_cache *cache, int flush_count)
{
	int ret;
	RbtIterator iter;
	struct ofi_mr_cache_entry *entry;
	int destroyed = 0;

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "starting flush on memory registration cache (flush count = %d)\n",
	       flush_count);

	/* Flushes are unnecessary for caches w/o lazy deregistration
	 * (stale size == 0). The LRU list is empty in this case */
	while (!dlist_empty(&cache->lru_list) && (destroyed < flush_count)) {
		ret = util_mr_cache_lru_dequeue(cache, &entry);
		if (OFI_UNLIKELY(ret)) {
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"list may be corrupt, no entries from lru pop\n");
			assert(0);
			break;
		}

		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "attempting to flush region %"PRIu64":%"PRIu64"\n",
		       entry->region.address, entry->region.length);
		iter = rbtFind(cache->stale_tree, &entry->region);
		if (OFI_UNLIKELY(!iter)) {
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"lru entries MUST be present in the cache,"
				 " could not find entry (%p) in stale tree"
				 " %"PRIu64":%"PRIu64"\n",
				 entry, entry->region.address, entry->region.length);
			assert(0);
			break;
		}

		ret = rbtErase(cache->stale_tree, iter);
		if (OFI_UNLIKELY(ret != RBT_STATUS_OK)) {
			FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
				"failed to erase lru entry from stale tree\n");
			assert(0);
			break;
		}

		util_mr_cache_entry_destroy(cache, entry);
		destroyed++;
	}

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "flushed %i of %lu entries from memory registration cache\n",
	       destroyed, cache->stale_elem);

	cache->stale_elem -= destroyed;
}

void ofi_mr_cache_flush(struct ofi_mr_cache *cache)
{
	util_mr_cache_flush(cache, cache->reg_size);
}

static int util_mr_cache_add_overlap_to_inuse(struct ofi_mr_cache *cache,
					      struct ofi_mr_region *region,
					      struct ofi_mr_reg_attr *reg_attr,
					      RbtIterator iter,
					      struct ofi_mr_cache_entry **entry)
{
	struct ofi_mr_region *found_region, new_region;
	struct ofi_mr_cache_entry *found_entry;
	int ret, cmp;
	uint64_t new_end, found_end = 0;
	DEFINE_LIST(retired_entries);

	rbtKeyValue(cache->inuse_tree, iter, (void **)&found_region,
		    (void **)&found_entry);
	/* otherwise, iterate from the existing entry until we can no longer
	 * find an entry that overlaps with the new registration and remove
	 * and retire each of those entries.
	 */
	new_region.address = MIN(found_region->address, region->address);
	new_end = region->address + region->length;
	while (iter) {
		rbtKeyValue(cache->inuse_tree, iter, (void **)&found_region,
			    (void **)&found_entry);
		cmp = util_mr_cache_check_overlapping_region(found_region, region);
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "candidate: region=%"PRIu64":%"PRIu64" result=%d\n",
		       found_region->address, found_region->length, cmp);
		if (cmp != 0)
			break;

		/* compute new ending address */
		found_end = found_region->address + found_region->length;

		/* mark the entry as retired */
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "retiring entry, region=%"PRIu64":%"PRIu64" entry %p\n",
		       found_region->address, found_region->length, found_entry);
		util_mr_cache_entry_set_retired(found_entry);
		dlist_insert_tail(&found_entry->siblings, &retired_entries);

		iter = rbtNext(cache->inuse_tree, iter);
	}
	/* Since our new region might fully overlap every other entry in the tree,
	 * we need to take the maximum of the last entry and the new entry
	 */
	new_region.length = MAX(found_end, new_end) - new_region.address;

	/* remove retired entries from tree */
	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "removing retired entries from inuse tree\n");
	util_mr_cache_remove_sibling_entries_from_tree(cache, &retired_entries,
						       cache->inuse_tree);

	/* create new registration */
	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "creating a new merged registration, region=%"PRIu64":%"PRIu64"\n",
	       new_region.address, new_region.length);
	ret = util_mr_cache_create_registration(cache, &new_region, reg_attr, entry);
	if (ret) {
		/* If we get here, one of two things have happened.
		 * Either some part of the new merged registration was
		 * unmapped (i.e., freed by user) or the merged
		 * registration failed for some other reason. 
		 * The first case is a user error
		 * (which they should have been warned about by
		 * the notifier), and the second case is always
		 * possible.  Neither case is a problem.  The entries
		 * above have been retired, and here we return the
		 * error */
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"failed to create merged registration,"
			" region=%"PRIu64":%"PRIu64"\n",
			new_region.address, new_region.length);
		return ret;
	}

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "created a new merged registration, region=%"PRIu64":%"PRIu64" entry %p\n",
	       new_region.address, new_region.length, *entry);

	util_mr_cache_entry_set_merged(*entry);

	/* move retired entries to the head of the new entry's child list */
	if (!dlist_empty(&retired_entries))
		util_mr_cache_attach_retired_entries_to_registration(cache,
								     &retired_entries,
								     *entry);

	UTIL_MR_CACHE_INC_MISSES(cache);

	return ret;
}

static int util_mr_cache_search_inuse(struct ofi_mr_cache *cache,
				      struct ofi_mr_region *region,
				      struct ofi_mr_cache_entry **entry,
				      RbtIterator *out_overlap_iter)
{
	RbtIterator iter;
	struct ofi_mr_region *found_region;
	struct ofi_mr_cache_entry *found_entry;

	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		util_mr_cache_clear_notifier_events(cache);

	/* first we need to find an entry that overlaps with this one.
	 * care should be taken to find the left most entry that overlaps
	 * with this entry since the entry we are searching for might overlap
	 * many entries and the tree may be left or right balanced
	 * at the head
	 */
	iter = rbtFindLeftmost(cache->inuse_tree, (void *)region,
			       util_mr_cache_check_overlapping_region);
	if (!iter) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "could not find region in inuse, region=%"PRIu64":%"PRIu64"\n",
		       region->address, region->length);
		return -FI_ENOENT;
	}

	rbtKeyValue(cache->inuse_tree, iter, (void **)&found_region,
		    (void **)&found_entry);

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "found a region that matches the search criteria, "
	       "found=%"PRIu64":%"PRIu64" region=%"PRIu64":%"PRIu64"\n",
	       found_region->address, found_region->length,
	       region->address, region->length);

	/* if the entry that we've found completely subsumes
	 * the requested entry, just return a reference to
	 * that existing registration
	 */
	if (util_mr_cache_can_subsume(found_region, region)) {
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "found an entry that subsumes the request, "
		       "existing=%"PRIu64":%"PRIu64" region=%"PRIu64":%"PRIu64
		       "entry %p\n", found_region->address, found_region->length,
		       region->address, region->length, found_entry);
		*entry = found_entry;
		util_mr_cache_entry_get(cache, found_entry);

		UTIL_MR_CACHE_INC_HITS(cache);
		return FI_SUCCESS;
	}

	*out_overlap_iter = iter;

	return -FI_ENOENT;
}

static int util_mr_cache_process_inuse(struct ofi_mr_cache *cache,
				       struct ofi_mr_region *region,
				       struct ofi_mr_reg_attr *reg_attr,
				       struct ofi_mr_cache_entry **entry)
{
	RbtIterator *iter = NULL;
	int ret = util_mr_cache_search_inuse(cache, region, entry, iter);
	if (ret == -FI_ENOENT && (iter != NULL))
	    ret = util_mr_cache_add_overlap_to_inuse(cache, region, reg_attr,
						     *iter, entry);

	return ret;
}

static int util_mr_cache_search_stale(struct ofi_mr_cache *cache,
				      struct ofi_mr_region *region,
				      struct ofi_mr_reg_attr *reg_attr,
				      struct ofi_mr_cache_entry **entry)
{
	int ret;
	RbtStatus rc;
	RbtIterator iter;
	struct ofi_mr_region *mr_region;
	struct ofi_mr_cache_entry *mr_entry, *tmp;

	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache))
		util_mr_cache_clear_notifier_events(cache);

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "searching for stale entry, region=%"PRIu64":%"PRIu64"\n",
	       region->address, region->length);

	iter = rbtFindLeftmost(cache->stale_tree, (void *)region,
			       util_mr_cache_check_overlapping_region);
	if (!iter)
		return -FI_ENOENT;

	rbtKeyValue(cache->stale_tree, iter, (void **)&mr_region,
		    (void **)&mr_entry);

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "found a matching entry, found=%"PRIu64":%"PRIu64" region=%"PRIu64":%"PRIu64"\n",
	       mr_region->address, mr_region->length,
	       region->address, region->length);

	/* if the entry that we've found completely subsumes
	 * the requested entry, just return a reference to
	 * that existing registration
	 */
	if (util_mr_cache_can_subsume(mr_region, region)) {
		ret = util_mr_cache_process_inuse(cache, mr_region, reg_attr, &tmp);
		if (!ret) {
			/* if we found an entry in the inuse tree
			 * in this manner, it means that there was
			 * an entry either overlapping or contiguous
			 * with the stale entry in the inuse tree, and
			 * a new entry has been made and saved to tmp.
			 * The old entry (mr_entry) should be destroyed
			 * now as it is no longer needed.
			 */
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "removing entry from stale region=%"PRIu64":%"PRIu64"\n",
			       mr_region->address, mr_region->length);

			rc = rbtErase(cache->stale_tree, iter);
			if (OFI_UNLIKELY(rc != RBT_STATUS_OK)) {
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"failed to erase entry from stale tree\n");
			} else {
				util_mr_cache_lru_remove(cache, mr_entry);
				cache->stale_elem--;
				util_mr_cache_entry_destroy(cache, mr_entry);
			}

			*entry = tmp;
		} else {
			FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			       "removing entry (%p) from stale and migrating to inuse,"
			       " region=%"PRIu64":%"PRIu64"\n",
			       mr_entry, mr_region->address, mr_region->length);
			rc = rbtErase(cache->stale_tree, iter);
			if (OFI_UNLIKELY(rc != RBT_STATUS_OK))
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"failed to erase entry (%p) from stale tree\n",
					mr_entry);

			util_mr_cache_lru_remove(cache, mr_entry);
			cache->stale_elem--;

			/* if we made it to this point, there weren't
			 * any entries in the inuse tree that would
			 * have overlapped with this entry
			 */
			rc = rbtInsert(cache->inuse_tree, &mr_entry->region, mr_entry);
			if (OFI_UNLIKELY(rc != RBT_STATUS_OK))
				FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
					"failed to insert entry into inuse tree\n");

			ofi_atomic_set32(&mr_entry->ref_cnt, 1);
			cache->inuse_elem++;

			*entry = mr_entry;
		}

		return FI_SUCCESS;
	}

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "could not use matching entry, found=%"PRIu64":%"PRIu64"\n",
	       mr_region->address, mr_region->length);

	return -FI_ENOENT;
}

static int util_mr_cache_create_registration(struct ofi_mr_cache *cache,
					     struct ofi_mr_region *region,
					     struct ofi_mr_reg_attr *reg_attr,
					     struct ofi_mr_cache_entry **entry)
{
	int ret;
	struct ofi_mr_cache_entry *current_entry;

	/* if we made it here, we didn't find the entry at all */
	ret = ofi_memalign((void **)&current_entry, 16,
			   sizeof(*current_entry) + cache->elem_size);
	if (ret)
		return -FI_ENOMEM;

	dlist_init(&current_entry->lru_entry);
	dlist_init(&current_entry->children);
	dlist_init(&current_entry->siblings);

	current_entry->reg_attr = *reg_attr;

	ret = cache->reg_callback(cache, current_entry,
				  (void *)region->address, region->length);
	if (OFI_UNLIKELY(ret)) {
		FI_INFO(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"failed to register memory with callback\n");
		goto err;
	}

	/* set up the entry's region */
	current_entry->region.address = region->address;
	current_entry->region.length = region->length;

	ret = util_mr_cache_notifier_monitor(cache, current_entry);
	if (OFI_UNLIKELY(ret))
		goto err_dereg;

	ret = rbtInsert(cache->inuse_tree, &current_entry->region,
			current_entry);
	if (OFI_UNLIKELY(ret != RBT_STATUS_OK)) {
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"failed to insert registration(%p) into cache, ret=%i\n",
			current_entry, ret);
		goto err_dereg;
	}

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "inserted region %"PRIu64":%"PRIu64" into inuse %p\n",
	       current_entry->region.address, current_entry->region.length,
	       current_entry);

	cache->inuse_elem++;
	ofi_atomic_initialize32(&current_entry->ref_cnt, 1);

	*entry = current_entry;

	return FI_SUCCESS;

err_dereg:
	ret = cache->dereg_callback(cache, current_entry);
	if (OFI_UNLIKELY(ret))
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"failed to deregister memory with callback, ret=%d\n",
			ret);
	else
		FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
		       "Successfully deregister memory with callback\n");
err:
	ofi_freealign(current_entry);
	return -FI_EAVAIL;
}

int ofi_mr_cache_register(struct ofi_mr_cache *cache,
			  uint64_t address, uint64_t length,
			  struct ofi_mr_reg_attr *reg_attr,
			  struct ofi_mr_cache_entry **entry)
{
	int ret;
	struct ofi_mr_region region = {
		.address = address,
		.length = length,
	};
	struct ofi_mr_cache_entry *cache_entry;

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR, "\n");

	/* fastpath inuse */
	ret = util_mr_cache_process_inuse(cache, &region, reg_attr,
					  &cache_entry);
	if (ret == FI_SUCCESS)
		goto success;

	/* if we shouldn't introduce any new elements, return -FI_ENOSPC */
	if (OFI_UNLIKELY(cache->reg_size > 0 &&
			 (cache->inuse_elem >= cache->reg_size))) {
		ret = -FI_ENOSPC;
		goto err;
	}

	if (UTIL_MR_CACHE_IS_LAZY_DEREG(cache)) {
		util_mr_cache_clear_notifier_events(cache);

		/* if lazy deregistration (stale size > 0) is in use, we can check the
		 * stale tree
		 */
		ret = util_mr_cache_search_stale(cache, &region,
						 reg_attr, &cache_entry);
		if (!ret) {
			UTIL_MR_CACHE_INC_HITS(cache);
			goto success;
		}
	}

	/* If the cache is full, then flush one of the stale entries to make
	 * room for the new entry. This works because we check above to see if
	 * the number of inuse entries exceeds the reg size
	 */
	if ((cache->inuse_elem + cache->stale_elem) == cache->reg_size)
		util_mr_cache_flush(cache, 1);

	ret = util_mr_cache_create_registration(cache, &region, reg_attr,
						&cache_entry);
	if (ret)
		goto err;

	UTIL_MR_CACHE_INC_MISSES(cache);

success:
	*entry = cache_entry;

	return FI_SUCCESS;

err:
	return ret;
}

int ofi_mr_cache_deregister(struct ofi_mr_cache *cache,
			    struct ofi_mr_cache_entry *entry)
{
	int ret;

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR, "\n");

	if (ofi_atomic_get32(&entry->ref_cnt) == 0) {
		FI_WARN(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
			"There are no references to the entry (%p)\n",
			entry);
		return -FI_EINVAL;
	}

	FI_DBG(UTIL_MR_CACHE_GET_PROV(cache), FI_LOG_MR,
	       "entry found, entry=%p refs=%d\n",
	       entry, ofi_atomic_get32(&entry->ref_cnt));

	ret = util_mr_cache_entry_put(cache, entry);

	/* Since we check this on each deregistration, the amount of elements
	 * over the limit should always be 1
	 */
	if (cache->stale_elem > cache->stale_size)
		util_mr_cache_flush(cache, 1);

	return ret;
}
