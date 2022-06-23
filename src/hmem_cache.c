/*
 * Copyright (c) 2022 UT-Battelle, LLC. All rights reserved.
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <ctype.h>

#include <hmem_cache.h>
#include <ofi_list.h>
#include <bitops.h>

static pgt_dir_t *ipc_cache_pgt_dir_alloc(const pgtable_t *pgtable)
{
	void *ptr;
	int ret;

	ret =  posix_memalign(&ptr,
				MAX(sizeof(void *), PGT_ENTRY_MIN_ALIGN),
				sizeof(pgt_dir_t));
	return (ret == 0) ? ptr : NULL;
}

static void ipc_cache_pgt_dir_release(const pgtable_t *pgtable,
									  pgt_dir_t *dir)
{
	free(dir);
}

static void
ipc_cache_region_collect_callback(const pgtable_t *pgtable,
								  pgt_region_t *pgt_region,
								  void *arg)
{
	struct dlist_entry *list = arg;
	struct ipc_cache_region *region;

	region = container_of(pgt_region, struct ipc_cache_region, super);
	dlist_insert_after(&region->list, list);
}

static void ipc_cache_purge(struct hmem_cache *cache)
{
	struct ipc_cache_region *region;
	struct dlist_entry *tmp;
	struct dlist_entry region_list;
	int ret;

	dlist_init(&region_list);
	pgtable_purge(&cache->pgtable, ipc_cache_region_collect_callback,
				  &region_list);

	dlist_foreach_container_safe(&region_list, struct ipc_cache_region,
								region, list, tmp) {
		ret = cache->unmap_cb(region->key.iface, region->mapped_addr);
		if (ret) {
			FI_WARN(&core_prov, FI_LOG_CORE,
					 "failed to unmap addr:%p\n", region->mapped_addr);
		}

		free(region);
	}

	FI_INFO(&core_prov, FI_LOG_CORE,
			"%s: ipc cache purged\n", cache->name);
}

static void ipc_cache_invalidate_regions(struct hmem_cache *cache,
										 void *from, void *to)
{
	struct dlist_entry region_list;
	struct ipc_cache_region *region;
	struct dlist_entry *tmp;
	int ret;

	dlist_init(&region_list);
	pgtable_search_range(&cache->pgtable, (pgt_addr_t)from,
						(pgt_addr_t)to - 1,
						ipc_cache_region_collect_callback,
						&region_list);
	dlist_foreach_container_safe(&region_list, struct ipc_cache_region,
								region, list, tmp) {
		ret = pgtable_remove(&cache->pgtable, &region->super);
		if (ret) {
			FI_WARN(&core_prov, FI_LOG_CORE,
					 "failed to remove address:%p from cache (%d)\n",
					 (void *)region->key.address, ret);
		}

		ret = cache->unmap_cb(region->key.iface, region->mapped_addr);
		if (ret) {
			FI_WARN(&core_prov, FI_LOG_CORE,
					 "failed to unmap addr:%p\n", region->mapped_addr);
		}
		free(region);
	}
	FI_INFO(&core_prov, FI_LOG_CORE,
			"%s: closed memhandles in the range [%p..%p]\n",
			cache->name, from, to);
}

void ipc_cache_invalidate(struct hmem_cache *cache, void *address)
{
	pgt_region_t *pgt_region;
	struct ipc_cache_region *region;

	pthread_rwlock_rdlock(&cache->lock);
	pgt_region = pgtable_lookup(&cache->pgtable, (uintptr_t) address);
	if (OFI_LIKELY(pgt_region != NULL)) {
		region = container_of(pgt_region, struct ipc_cache_region, super);
		ipc_cache_invalidate_regions(cache,
								(void *)region->super.start,
								(void *)region->super.end);
	}
	pthread_rwlock_unlock(&cache->lock);
}

int ipc_cache_map_memhandle(struct hmem_cache *cache, struct ipc_info *key,
							void **mapped_addr)
{
	pgt_region_t *pgt_region;
	struct ipc_cache_region *region;
	int ret;

	pthread_rwlock_rdlock(&cache->lock);
	pgt_region = pgtable_lookup(&cache->pgtable, key->address);
	if (OFI_LIKELY(pgt_region != NULL)) {
		region = container_of(pgt_region, struct ipc_cache_region, super);
		if ((memcmp(&key->ipc_handle, &region->key.ipc_handle,
				   IPC_HANDLE_SIZE) == 0) &&
			(region->super.end >= key->address+key->length)) {
			/*cache hit */
			FI_INFO(&core_prov, FI_LOG_CORE,
					"%s: ipc cache hit addr:%p size:%lu region:"
					PGT_REGION_FMT"\n", cache->name, (void *)key->address,
					key->base_length, PGT_REGION_ARG(&region->super));

			*mapped_addr = region->mapped_addr;
			pthread_rwlock_unlock(&cache->lock);
			return 0;
		} else {
			FI_INFO(&core_prov, FI_LOG_CORE,
					"%s: ipc cache remove stale region:"
					PGT_REGION_FMT  " new_addr:%p new_size:%lu\n",
					cache->name, PGT_REGION_ARG(&region->super),
					(void *)key->address, key->base_length);

			ret = pgtable_remove(&cache->pgtable, &region->super);
			if (ret) {
				FI_WARN(&core_prov, FI_LOG_CORE,
						 "%s: failed to remove address:%p from cache\n",
						 cache->name, (void *)key->address);
				goto err;
			}

			ret = cache->unmap_cb(key->iface, region->mapped_addr);
			if (ret) {
				FI_WARN(&core_prov, FI_LOG_CORE,
						 "failed to unmap addr:%p\n", region->mapped_addr);
			}

			free(region);
		}
	}

	ret = cache->map_cb(key->iface, (void**) &key->ipc_handle,
						key->base_length, key->dev_num, mapped_addr);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				 "%s: failed to open ipc mem handle. addr:%p len:%lu\n",
				 cache->name, (void *)key->address, key->base_length);
	}

	/*create new cache entry */
	ret = posix_memalign((void **)&region,
						MAX(sizeof(void *), PGT_ENTRY_MIN_ALIGN),
						sizeof(struct ipc_cache_region));
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				"failed to allocate ipc_cache region");
		ret = -FI_ENOMEM;
		goto err;
	}

	region->super.start = align_down_pow2(key->address, PGT_ADDR_ALIGN);
	region->super.end = align_up_pow2(key->address + key->base_length, PGT_ADDR_ALIGN);
	region->key = *key;
	region->mapped_addr = *mapped_addr;

	ret = pgtable_insert(&cache->pgtable, &region->super);
	if (ret == -FI_EALREADY) {
		/* overlapped region means memory freed at source. remove and try insert */
		ipc_cache_invalidate_regions(cache,
									(void *)region->super.start,
									(void *)region->super.end);
		ret = pgtable_insert(&cache->pgtable, &region->super);
	}
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				 "%s: failed to insert region:"PGT_REGION_FMT" size:%lu:%d\n",
				 cache->name, PGT_REGION_ARG(&region->super), key->base_length, ret);
		free(region);
		goto err;
	}

	FI_INFO(&core_prov, FI_LOG_CORE,
			"%s: ipc cache new region:"PGT_REGION_FMT" size:%lu\n",
			cache->name, PGT_REGION_ARG(&region->super), key->base_length);

	pthread_rwlock_unlock(&cache->lock);
	return 0;
err:
	pthread_rwlock_unlock(&cache->lock);
	return ret;
}

int ipc_create_hmem_cache(struct hmem_cache **cache,
						  const char *name, map_cb_t map_cb,
						  unmap_cb_t unmap_cb)
{
	struct hmem_cache *cache_desc;
	int ret;

	cache_desc = calloc(1, sizeof(*cache_desc));
	if (!cache_desc) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				 "failed to allocate memory for ipc cache\n");
		return -FI_ENOMEM;
	}

	ret = pthread_rwlock_init(&cache_desc->lock, NULL);
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_CORE,
				 "pthread_rwlock_init() failed\n");
		ret = -FI_EINVAL;
		goto err;
	}

	ret = pgtable_init(&cache_desc->pgtable,
						ipc_cache_pgt_dir_alloc,
						ipc_cache_pgt_dir_release);
	if (ret) {
		goto err_destroy_rwlock;
	}

	cache_desc->name = strdup(name);
	if (cache_desc->name == NULL) {
		ret = -FI_ENOMEM;
		goto err_destroy_rwlock;
	}

	cache_desc->map_cb = map_cb;
	cache_desc->unmap_cb = unmap_cb;

	*cache = cache_desc;
	return 0;

err_destroy_rwlock:
	pthread_rwlock_destroy(&cache_desc->lock);
err:
	free(cache_desc);
	return ret;
}

void ipc_destroy_hmem_cache(struct hmem_cache *cache)
{
	ipc_cache_purge(cache);
	pgtable_cleanup(&cache->pgtable);
	pthread_rwlock_destroy(&cache->lock);
	free(cache->name);
	free(cache);
}
