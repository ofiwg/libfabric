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


int util_buf_grow(struct util_buf_pool *pool)
{
	void *buf;
	int ret;
	size_t i;
	struct util_buf_region *buf_region;
	ssize_t hp_size;
	struct util_buf_footer *buf_ftr;

	if (pool->attr.max_cnt && pool->num_allocated >= pool->attr.max_cnt) {
		return -1;
	}

	buf_region = calloc(1, sizeof(*buf_region));
	if (!buf_region)
		return -1;

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

	if (pool->attr.alloc_hndlr) {
		ret = pool->attr.alloc_hndlr(pool->attr.ctx,
					     buf_region->mem_region,
					     buf_region->size,
					     &buf_region->context);
		if (ret)
			goto err2;
	}

	if (!(pool->regions_cnt % UTIL_BUF_POOL_REGION_CHUNK_CNT)) {
		struct util_buf_region **new_table =
			realloc(pool->regions_table,
				(pool->regions_cnt +
				 UTIL_BUF_POOL_REGION_CHUNK_CNT) *
				sizeof(*pool->regions_table));
		if (!new_table)
			goto err3;
		pool->regions_table = new_table;
	}
	pool->regions_table[pool->regions_cnt] = buf_region;
	pool->regions_cnt++;

	for (i = 0; i < pool->attr.chunk_cnt; i++) {
		buf = (buf_region->mem_region + i * pool->entry_sz);
		buf_ftr = util_buf_get_ftr(pool, buf);

		if (pool->attr.init) {
#if ENABLE_DEBUG
			buf_ftr->entry.next = (void *) OFI_MAGIC_64;
#endif
			pool->attr.init(pool->attr.ctx, buf);
			assert(buf_ftr->entry.next == (void *) OFI_MAGIC_64);
		}

		buf_ftr->region = buf_region;
		buf_ftr->index = pool->num_allocated + i;
		slist_insert_tail(&buf_ftr->entry, &pool->buf_list);
	}

	slist_insert_tail(&buf_region->entry, &pool->region_list);
	pool->num_allocated += pool->attr.chunk_cnt;
	return 0;
err3:
	if (pool->attr.free_hndlr)
	    pool->attr.free_hndlr(pool->attr.ctx, buf_region->context);
err2:
	ofi_freealign(buf_region->mem_region);
err1:
	free(buf_region);
	return -1;
}

int util_buf_pool_create_attr(struct util_buf_attr *attr,
			      struct util_buf_pool **buf_pool)
{
	size_t entry_sz;
	ssize_t hp_size;

	(*buf_pool) = calloc(1, sizeof(**buf_pool));
	if (!*buf_pool)
		return -FI_ENOMEM;

	(*buf_pool)->attr = *attr;

	entry_sz = (attr->size + sizeof(struct util_buf_footer));
	(*buf_pool)->entry_sz = fi_get_aligned_sz(entry_sz, attr->alignment);

	hp_size = ofi_get_hugepage_size();

	if ((*buf_pool)->attr.chunk_cnt * (*buf_pool)->entry_sz < hp_size)
		(*buf_pool)->attr.is_mmap_region = 0;
	else
		(*buf_pool)->attr.is_mmap_region = 1;

	slist_init(&(*buf_pool)->buf_list);
	slist_init(&(*buf_pool)->region_list);

	return FI_SUCCESS;
}

int util_buf_pool_create_ex(struct util_buf_pool **buf_pool,
			    size_t size, size_t alignment,
			    size_t max_cnt, size_t chunk_cnt,
			    util_buf_region_alloc_hndlr alloc_hndlr,
			    util_buf_region_free_hndlr free_hndlr,
			    void *pool_ctx)
{
	struct util_buf_attr attr = {
		.size		= size,
		.alignment 	= alignment,
		.max_cnt	= max_cnt,
		.chunk_cnt	= chunk_cnt,
		.alloc_hndlr	= alloc_hndlr,
		.free_hndlr	= free_hndlr,
		.ctx		= pool_ctx,
		.track_used	= 1,
	};
	return util_buf_pool_create_attr(&attr, buf_pool);
}

void util_buf_pool_destroy(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	struct util_buf_region *buf_region;
	int ret;

	while (!slist_empty(&pool->region_list)) {
		entry = slist_remove_head(&pool->region_list);
		buf_region = container_of(entry, struct util_buf_region, entry);
#if ENABLE_DEBUG
		if (pool->attr.track_used)
			assert(buf_region->num_used == 0);
#endif
		if (pool->attr.free_hndlr)
			pool->attr.free_hndlr(pool->attr.ctx, buf_region->context);
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
