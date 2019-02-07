/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 * Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include <unistd.h>
#include <ofi_enosys.h>
#include <ofi_mem.h>
#include <ofi.h>
#include <ofi_osd.h>


enum {
	OFI_BUFPOOL_REGION_CHUNK_CNT = 16
};


int ofi_bufpool_grow(struct ofi_bufpool *pool)
{
	void *buf;
	int ret;
	size_t i;
	struct ofi_bufpool_region *buf_region;
	ssize_t hp_size;
	struct ofi_bufpool_ftr *buf_ftr;

	if (pool->attr.max_cnt && pool->num_allocated >= pool->attr.max_cnt) {
		return -1;
	}

	buf_region = calloc(1, sizeof(*buf_region));
	if (!buf_region)
		return -1;

	buf_region->pool = pool;
	dlist_init(&buf_region->buf_list);

	if (pool->attr.is_mmap_region) {
		hp_size = ofi_get_hugepage_size();
		if (hp_size < 0)
			goto err1;

		buf_region->size = fi_get_aligned_sz(pool->attr.chunk_cnt *
						     pool->entry_sz, hp_size);

		ret = ofi_alloc_hugepage_buf((void **)&buf_region->mem_region,
					     buf_region->size);
		if (ret) {
			FI_DBG(&core_prov, FI_LOG_CORE,
			       "Huge page allocation failed: %s\n",
			       fi_strerror(-ret));

			if (pool->num_allocated > 0)
				goto err1;

			pool->attr.is_mmap_region = 0;
		}
	}

	if (!pool->attr.is_mmap_region) {
		buf_region->size = pool->attr.chunk_cnt * pool->entry_sz;

		ret = ofi_memalign((void **)&buf_region->mem_region,
				   pool->attr.alignment, buf_region->size);
		if (ret)
			goto err1;
	}

	memset(buf_region->mem_region, 0, buf_region->size);
	if (pool->attr.alloc_fn) {
		ret = pool->attr.alloc_fn(pool->attr.ctx,
					     buf_region->mem_region,
					     buf_region->size,
					     &buf_region->context);
		if (ret)
			goto err2;
	}

	if (!(pool->regions_cnt % OFI_BUFPOOL_REGION_CHUNK_CNT)) {
		struct ofi_bufpool_region **new_table =
			realloc(pool->regions_table,
				(pool->regions_cnt +
				 OFI_BUFPOOL_REGION_CHUNK_CNT) *
				sizeof(*pool->regions_table));
		if (!new_table)
			goto err3;
		pool->regions_table = new_table;
	}
	pool->regions_table[pool->regions_cnt] = buf_region;
	pool->regions_cnt++;

	for (i = 0; i < pool->attr.chunk_cnt; i++) {
		buf = (buf_region->mem_region + i * pool->entry_sz);
		buf_ftr = ofi_buf_ftr(pool, buf);

		if (pool->attr.init_fn) {
#if ENABLE_DEBUG
			if (!pool->attr.indexing.ordered) {
				buf_ftr->entry.slist.next = (void *) OFI_MAGIC_64;

				pool->attr.init_fn(pool->attr.ctx, buf);

				assert(buf_ftr->entry.slist.next == (void *) OFI_MAGIC_64);
			} else {
				buf_ftr->entry.dlist.next = (void *) OFI_MAGIC_64;
				buf_ftr->entry.dlist.prev = (void *) OFI_MAGIC_64;

				pool->attr.init_fn(pool->attr.ctx, buf);

				assert((buf_ftr->entry.dlist.next == (void *) OFI_MAGIC_64) &&
				       (buf_ftr->entry.dlist.prev == (void *) OFI_MAGIC_64));
			}
#else
			pool->attr.init_fn(pool->attr.ctx, buf);
#endif
		}

		buf_ftr->region = buf_region;
		buf_ftr->index = pool->num_allocated + i;
		if (!pool->attr.indexing.ordered) {
			slist_insert_tail(&buf_ftr->entry.slist,
					  &pool->list.buffers);
		} else {
			dlist_insert_tail(&buf_ftr->entry.dlist,
					  &buf_region->buf_list);
		}
	}

	if (pool->attr.indexing.ordered) {
		dlist_insert_tail(&buf_region->entry,
				  &pool->list.regions);
	}

	pool->num_allocated += pool->attr.chunk_cnt;
	return 0;
err3:
	if (pool->attr.free_fn)
	    pool->attr.free_fn(pool->attr.ctx, buf_region->context);
err2:
	ofi_freealign(buf_region->mem_region);
err1:
	free(buf_region);
	return -1;
}

int ofi_bufpool_create_attr(struct ofi_bufpool_attr *attr,
			      struct ofi_bufpool **buf_pool)
{
	size_t entry_sz;
	ssize_t hp_size;

	(*buf_pool) = calloc(1, sizeof(**buf_pool));
	if (!*buf_pool)
		return -FI_ENOMEM;

	(*buf_pool)->attr = *attr;

	entry_sz = (attr->size + sizeof(struct ofi_bufpool_ftr));
	(*buf_pool)->entry_sz = fi_get_aligned_sz(entry_sz, attr->alignment);

	hp_size = ofi_get_hugepage_size();

	if ((*buf_pool)->attr.chunk_cnt * (*buf_pool)->entry_sz < hp_size)
		(*buf_pool)->attr.is_mmap_region = 0;
	else
		(*buf_pool)->attr.is_mmap_region = 1;

	if (!(*buf_pool)->attr.indexing.ordered)
		slist_init(&(*buf_pool)->list.buffers);
	else
		dlist_init(&(*buf_pool)->list.regions);

	return FI_SUCCESS;
}

int ofi_bufpool_create_ex(struct ofi_bufpool **buf_pool,
			    size_t size, size_t alignment,
			    size_t max_cnt, size_t chunk_cnt,
			    ofi_bufpool_alloc_fn alloc_fn,
			    ofi_bufpool_free_fn free_fn,
			    void *pool_ctx)
{
	struct ofi_bufpool_attr attr = {
		.size		= size,
		.alignment 	= alignment,
		.max_cnt	= max_cnt,
		.chunk_cnt	= chunk_cnt,
		.alloc_fn	= alloc_fn,
		.free_fn	= free_fn,
		.ctx		= pool_ctx,
		.track_used	= 1,
		.indexing	= {
			.used		= 1,
			.ordered	= 0,
		},
	};
	return ofi_bufpool_create_attr(&attr, buf_pool);
}

void ofi_bufpool_destroy(struct ofi_bufpool *pool)
{
	struct ofi_bufpool_region *buf_region;
	int ret;
	size_t i;

	for (i = 0; i < pool->regions_cnt; i++) {
		buf_region = pool->regions_table[i];
#if ENABLE_DEBUG
		if (pool->attr.track_used)
			assert(buf_region->num_used == 0);
#endif
		if (pool->attr.free_fn)
			pool->attr.free_fn(pool->attr.ctx, buf_region->context);
		if (pool->attr.is_mmap_region) {
			ret = ofi_free_hugepage_buf(buf_region->mem_region,
						    buf_region->size);
			if (ret) {
				FI_DBG(&core_prov, FI_LOG_CORE,
				       "Huge page free failed: %s\n",
				       fi_strerror(-ret));
				assert(0);
			}
		} else {
			ofi_freealign(buf_region->mem_region);
		}

		free(buf_region);
	}
	free(pool->regions_table);
	free(pool);
}

int ofi_ibuf_is_lower(struct dlist_entry *item, const void *arg)
{
	struct ofi_bufpool_ftr *buf_ftr1 =
		container_of((struct dlist_entry *)arg,
			     struct ofi_bufpool_ftr, entry.dlist);
	struct ofi_bufpool_ftr *buf_ftr2 =
		container_of(item, struct ofi_bufpool_ftr, entry.dlist);
	return (buf_ftr1->index < buf_ftr2->index);
}

int ofi_ibufpool_region_is_lower(struct dlist_entry *item, const void *arg)
{
	struct ofi_bufpool_region *buf_region1 =
		container_of((struct dlist_entry *)arg,
			     struct ofi_bufpool_region, entry);
	struct ofi_bufpool_region *buf_region2 =
		container_of(item, struct ofi_bufpool_region, entry);
	struct ofi_bufpool_ftr *buf_region1_head =
		container_of(buf_region1->buf_list.next,
			     struct ofi_bufpool_ftr, entry.dlist);
	struct ofi_bufpool_ftr *buf_region2_head =
		container_of(buf_region2->buf_list.next,
			     struct ofi_bufpool_ftr, entry.dlist);
	size_t buf_region1_index =
		(size_t)(buf_region1_head->index / buf_region1->pool->attr.chunk_cnt);
	size_t buf_region2_index =
		(size_t)(buf_region2_head->index / buf_region2->pool->attr.chunk_cnt);

	return (buf_region1_index < buf_region2_index);
}
