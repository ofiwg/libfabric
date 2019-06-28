/*
 * Copyright (c) 2016-2017 Cray Inc. All rights reserved.
 * Copyright (c) 2017-2019 Intel Corporation, Inc.  All rights reserved.
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

#include <config.h>
#include <stdlib.h>
#include <ofi_util.h>
#include <ofi_iov.h>
#include <ofi_mr.h>
#include <ofi_list.h>
#include <ofi_tree.h>


struct ofi_mr_cache_params cache_params = {
	.max_cnt = 1024,
};

static int util_mr_find_within(struct ofi_rbmap *map, void *key, void *data)
{
	struct ofi_mr_entry *entry = data;
	struct iovec *iov = key;

	if (ofi_iov_shifted_left(iov, &entry->iov))
		return -1;
	else if (ofi_iov_shifted_right(iov, &entry->iov))
		return 1;
	else
		return 0;
}

static int util_mr_find_overlap(struct ofi_rbmap *map, void *key, void *data)
{
	struct ofi_mr_entry *entry = data;
	struct iovec *iov = key;

	if (ofi_iov_left(iov, &entry->iov))
		return -1;
	else if (ofi_iov_right(iov, &entry->iov))
		return 1;
	else
		return 0;
}

static void util_mr_free_entry(struct ofi_mr_cache *cache,
			       struct ofi_mr_entry *entry)
{
	FI_DBG(cache->domain->prov, FI_LOG_MR, "free %p (len: %" PRIu64 ")\n",
	       entry->iov.iov_base, entry->iov.iov_len);

	assert(!entry->cached);
	/* If regions are not being merged, then we can't safely
	 * unsubscribe this region from the monitor.  Otherwise, we
	 * might unsubscribe an address range in use by another region.
	 * As a result, we remain subscribed.  This may result in extra
	 * notification events, but is harmless to correct operation.
	 */
	if (entry->subscribed && cache_params.merge_regions) {
		ofi_monitor_unsubscribe(cache->monitor, entry->iov.iov_base,
					entry->iov.iov_len);
		entry->subscribed = 0;
	}
	cache->delete_region(cache, entry);
	ofi_buf_free(entry);
}

static void util_mr_uncache_entry(struct ofi_mr_cache *cache,
				  struct ofi_mr_entry *entry)
{
	assert(entry->cached);
	cache->storage.erase(&cache->storage, entry);
	entry->cached = 0;
	cache->cached_cnt--;
	cache->cached_size -= entry->iov.iov_len;
}

/* Caller must hold ofi_mem_monitor lock */
void ofi_mr_cache_notify(struct ofi_mr_cache *cache, const void *addr, size_t len)
{
	struct ofi_mr_entry *entry;
	struct iovec iov;

	cache->notify_cnt++;
	iov.iov_base = (void *) addr;
	iov.iov_len = len;

	for (entry = cache->storage.overlap(&cache->storage, &iov); entry;
	     entry = cache->storage.overlap(&cache->storage, &iov)) {
		util_mr_uncache_entry(cache, entry);

		if (entry->use_cnt == 0) {
			dlist_remove_init(&entry->lru_entry);
			util_mr_free_entry(cache, entry);
		}
	}

	/* See comment in util_mr_free_entry.  If we're not merging address
	 * ranges, we can only safely unsubscribe for the reported range.
	 */
	if (!cache_params.merge_regions)
		ofi_monitor_unsubscribe(cache->monitor, addr, len);
}

static bool mr_cache_flush(struct ofi_mr_cache *cache)
{
	struct ofi_mr_entry *entry;

	if (dlist_empty(&cache->lru_list))
		return false;

	dlist_pop_front(&cache->lru_list, struct ofi_mr_entry,
			entry, lru_entry);
	dlist_init(&entry->lru_entry);
	FI_DBG(cache->domain->prov, FI_LOG_MR, "flush %p (len: %" PRIu64 ")\n",
	       entry->iov.iov_base, entry->iov.iov_len);

	util_mr_uncache_entry(cache, entry);
	util_mr_free_entry(cache, entry);
	return true;
}

bool ofi_mr_cache_flush(struct ofi_mr_cache *cache)
{
	bool empty;

	fastlock_acquire(&cache->monitor->lock);
	empty = mr_cache_flush(cache);
	fastlock_release(&cache->monitor->lock);
	return empty;
}

void ofi_mr_cache_delete(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry)
{
	FI_DBG(cache->domain->prov, FI_LOG_MR, "delete %p (len: %" PRIu64 ")\n",
	       entry->iov.iov_base, entry->iov.iov_len);

	fastlock_acquire(&cache->monitor->lock);
	cache->delete_cnt++;

	if (--entry->use_cnt == 0) {
		if (entry->cached) {
			dlist_insert_tail(&entry->lru_entry, &cache->lru_list);
		} else {
			cache->uncached_cnt--;
			cache->uncached_size -= entry->iov.iov_len;
			util_mr_free_entry(cache, entry);
		}
	}
	fastlock_release(&cache->monitor->lock);
}

static int
util_mr_cache_create(struct ofi_mr_cache *cache, const struct iovec *iov,
		     uint64_t access, struct ofi_mr_entry **entry)
{
	int ret;

	FI_DBG(cache->domain->prov, FI_LOG_MR, "create %p (len: %" PRIu64 ")\n",
	       iov->iov_base, iov->iov_len);

	*entry = ofi_buf_alloc(cache->entry_pool);
	if (OFI_UNLIKELY(!*entry))
		return -FI_ENOMEM;

	(*entry)->iov = *iov;
	(*entry)->use_cnt = 1;

	ret = cache->add_region(cache, *entry);
	if (ret) {
		while (ret && mr_cache_flush(cache)) {
			ret = cache->add_region(cache, *entry);
		}
		if (ret) {
			assert(!mr_cache_flush(cache));
			ofi_buf_free(*entry);
			return ret;
		}
	}

	if ((cache->cached_cnt > cache_params.max_cnt) ||
	    (cache->cached_size > cache_params.max_size)) {
		(*entry)->cached = 0;
		cache->uncached_cnt++;
		cache->uncached_size += iov->iov_len;
	} else {
		if (cache->storage.insert(&cache->storage,
					  &(*entry)->iov, *entry)) {
			ret = -FI_ENOMEM;
			goto err;
		}
		(*entry)->cached = 1;
		cache->cached_cnt++;
		cache->cached_size += iov->iov_len;

		ret = ofi_monitor_subscribe(cache->monitor, iov->iov_base,
					    iov->iov_len);
		if (ret)
			util_mr_uncache_entry(cache, *entry);
		else
			(*entry)->subscribed = 1;
	}

	return 0;

err:
	util_mr_free_entry(cache, *entry);
	return ret;
}

static int
util_mr_cache_merge(struct ofi_mr_cache *cache, const struct fi_mr_attr *attr,
		    struct ofi_mr_entry *old_entry, struct ofi_mr_entry **entry)
{
	struct iovec iov, *old_iov;

	iov = *attr->mr_iov;
	do {
		FI_DBG(cache->domain->prov, FI_LOG_MR,
		       "merging %p (len: %" PRIu64 ") with %p (len: %" PRIu64 ")\n",
		       iov.iov_base, iov.iov_len,
		       old_entry->iov.iov_base, old_entry->iov.iov_len);
		old_iov = &old_entry->iov;

		iov.iov_len = ((uintptr_t)
			MAX(ofi_iov_end(&iov), ofi_iov_end(old_iov))) + 1 -
			((uintptr_t) MIN(iov.iov_base, old_iov->iov_base));
		iov.iov_base = MIN(iov.iov_base, old_iov->iov_base);
		FI_DBG(cache->domain->prov, FI_LOG_MR, "merged %p (len: %" PRIu64 ")\n",
		       iov.iov_base, iov.iov_len);

		/* New entry will expand range of subscription */
		old_entry->subscribed = 0;

		util_mr_uncache_entry(cache, old_entry);

		if (old_entry->use_cnt == 0) {
			dlist_remove_init(&old_entry->lru_entry);
			util_mr_free_entry(cache, old_entry); 
		}

	} while ((old_entry = cache->storage.find(&cache->storage, &iov)));

	return util_mr_cache_create(cache, &iov, attr->access, entry);
}

int ofi_mr_cache_search(struct ofi_mr_cache *cache, const struct fi_mr_attr *attr,
			struct ofi_mr_entry **entry)
{
	int ret = 0;

	assert(attr->iov_count == 1);
	FI_DBG(cache->domain->prov, FI_LOG_MR, "search %p (len: %" PRIu64 ")\n",
	       attr->mr_iov->iov_base, attr->mr_iov->iov_len);

	fastlock_acquire(&cache->monitor->lock);
	cache->search_cnt++;

	while (((cache->cached_cnt >= cache_params.max_cnt) ||
		(cache->cached_size >= cache_params.max_size)) &&
	       mr_cache_flush(cache))
		;

	*entry = cache->storage.find(&cache->storage, attr->mr_iov);
	if (!*entry) {
		ret = util_mr_cache_create(cache, attr->mr_iov,
					   attr->access, entry);
		goto unlock;
	}

	/* This branch is always false if the merging entries wasn't requested */
	if (!ofi_iov_within(attr->mr_iov, &(*entry)->iov)) {
		ret = util_mr_cache_merge(cache, attr, *entry, entry);
		goto unlock;
	}

	cache->hit_cnt++;
	if ((*entry)->use_cnt++ == 0)
		dlist_remove_init(&(*entry)->lru_entry);

unlock:
	fastlock_release(&cache->monitor->lock);
	return ret;
}

void ofi_mr_cache_cleanup(struct ofi_mr_cache *cache)
{
	struct ofi_mr_entry *entry;
	struct dlist_entry *tmp;

	/* If we don't have a domain, initialization failed */
	if (!cache->domain)
		return;

	FI_INFO(cache->domain->prov, FI_LOG_MR, "MR cache stats: "
		"searches %zu, deletes %zu, hits %zu notify %zu\n",
		cache->search_cnt, cache->delete_cnt, cache->hit_cnt,
		cache->notify_cnt);

	fastlock_acquire(&cache->monitor->lock);
	dlist_foreach_container_safe(&cache->lru_list, struct ofi_mr_entry,
				     entry, lru_entry, tmp) {
		assert(entry->use_cnt == 0);
		util_mr_uncache_entry(cache, entry);
		dlist_remove_init(&entry->lru_entry);
		util_mr_free_entry(cache, entry);
	}
	fastlock_release(&cache->monitor->lock);

	ofi_monitor_del_cache(cache);
	cache->storage.destroy(&cache->storage);
	ofi_atomic_dec32(&cache->domain->ref);
	ofi_bufpool_destroy(cache->entry_pool);
	assert(cache->cached_cnt == 0);
	assert(cache->cached_size == 0);
	assert(cache->uncached_cnt == 0);
	assert(cache->uncached_size == 0);
}

static void ofi_mr_rbt_destroy(struct ofi_mr_storage *storage)
{
	ofi_rbmap_destroy(storage->storage);
}

static struct ofi_mr_entry *ofi_mr_rbt_find(struct ofi_mr_storage *storage,
					    const struct iovec *key)
{
	struct ofi_rbnode *node;

	node = ofi_rbmap_find(storage->storage, (void *) key);
	if (!node)
		return NULL;

	return node->data;
}

static struct ofi_mr_entry *ofi_mr_rbt_overlap(struct ofi_mr_storage *storage,
					    const struct iovec *key)
{
	struct ofi_rbnode *node;

	node = ofi_rbmap_search(storage->storage, (void *) key,
				util_mr_find_overlap);
	if (!node)
		return NULL;

	return node->data;
}

static int ofi_mr_rbt_insert(struct ofi_mr_storage *storage,
			     struct iovec *key,
			     struct ofi_mr_entry *entry)
{
	return ofi_rbmap_insert(storage->storage, (void *) &entry->iov,
			        (void *) entry);
}

static int ofi_mr_rbt_erase(struct ofi_mr_storage *storage,
			    struct ofi_mr_entry *entry)
{
	struct ofi_rbnode *node;

	node = ofi_rbmap_find(storage->storage, &entry->iov);
	assert(node);
	ofi_rbmap_delete(storage->storage, node);
	return 0;
}

static int ofi_mr_cache_init_rbt(struct ofi_mr_cache *cache)
{
	cache->storage.storage = ofi_rbmap_create(cache_params.merge_regions ?
						  util_mr_find_overlap :
						  util_mr_find_within);
	if (!cache->storage.storage)
		return -FI_ENOMEM;

	cache->storage.overlap = ofi_mr_rbt_overlap;
	cache->storage.destroy = ofi_mr_rbt_destroy;
	cache->storage.find = ofi_mr_rbt_find;
	cache->storage.insert = ofi_mr_rbt_insert;
	cache->storage.erase = ofi_mr_rbt_erase;
	return 0;
}

static int ofi_mr_cache_init_storage(struct ofi_mr_cache *cache)
{
	int ret;

	switch (cache->storage.type) {
	case OFI_MR_STORAGE_DEFAULT:
	case OFI_MR_STORAGE_RBT:
		ret = ofi_mr_cache_init_rbt(cache);
		break;
	case OFI_MR_STORAGE_USER:
		ret = (cache->storage.storage && cache->storage.overlap &&
		      cache->storage.destroy && cache->storage.find &&
		      cache->storage.insert && cache->storage.erase) ?
			0 : -FI_EINVAL;
		break;
	default:
		ret = -FI_EINVAL;
		break;
	}

	return ret;
}

int ofi_mr_cache_init(struct util_domain *domain,
		      struct ofi_mem_monitor *monitor,
		      struct ofi_mr_cache *cache)
{
	int ret;

	assert(cache->add_region && cache->delete_region);
	if (!cache_params.max_cnt)
		return -FI_ENOSPC;

	dlist_init(&cache->lru_list);
	cache->cached_cnt = 0;
	cache->cached_size = 0;
	cache->uncached_cnt = 0;
	cache->uncached_size = 0;
	cache->search_cnt = 0;
	cache->delete_cnt = 0;
	cache->hit_cnt = 0;
	cache->notify_cnt = 0;
	cache->domain = domain;
	ofi_atomic_inc32(&domain->ref);

	ret = ofi_mr_cache_init_storage(cache);
	if (ret)
		goto dec;

	ret = ofi_monitor_add_cache(monitor, cache);
	if (ret)
		goto destroy;

	ret = ofi_bufpool_create(&cache->entry_pool,
				 sizeof(struct ofi_mr_entry) +
				 cache->entry_data_size,
				 16, 0, 0, 0);
	if (ret)
		goto del;

	return 0;
del:
	ofi_monitor_del_cache(cache);
destroy:
	cache->storage.destroy(&cache->storage);
dec:
	ofi_atomic_dec32(&cache->domain->ref);
	cache->domain = NULL;
	return ret;
}
