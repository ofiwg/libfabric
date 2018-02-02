/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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

static inline void util_buf_set_region(union util_buf *buf,
				       struct util_buf_region *region,
				       struct util_buf_pool *pool)
{
	struct util_buf_footer *buf_ftr;
	if (util_buf_use_ftr(pool)) {
		buf_ftr = (struct util_buf_footer *) ((char *) buf + pool->data_sz);
		buf_ftr->region = region;
	}
}

int util_buf_grow(struct util_buf_pool *pool)
{
	int ret;
	size_t i;
	union util_buf *util_buf;
	struct util_buf_region *buf_region;

	if (pool->max_cnt && pool->num_allocated >= pool->max_cnt) {
		return -1;
	}

	buf_region = calloc(1, sizeof(*buf_region));
	if (!buf_region)
		return -1;

	ret = ofi_memalign((void **)&buf_region->mem_region, pool->alignment,
			     pool->chunk_cnt * pool->entry_sz);
	if (ret)
		goto err;

	if (pool->alloc_hndlr) {
		ret = pool->alloc_hndlr(pool->ctx, buf_region->mem_region,
					pool->chunk_cnt * pool->entry_sz,
					&buf_region->context);
		if (ret)
			goto err;
	}

	for (i = 0; i < pool->chunk_cnt; i++) {
		util_buf = (union util_buf *)
			(buf_region->mem_region + i * pool->entry_sz);
		util_buf_set_region(util_buf, buf_region, pool);
		slist_insert_tail(&util_buf->entry, &pool->buf_list);
	}

	slist_insert_tail(&buf_region->entry, &pool->region_list);
	pool->num_allocated += pool->chunk_cnt;
	return 0;
err:
	free(buf_region);
	return -1;
}

int util_buf_pool_create_ex(struct util_buf_pool **buf_pool,
			    size_t size, size_t alignment,
			    size_t max_cnt, size_t chunk_cnt,
			    util_buf_region_alloc_hndlr alloc_hndlr,
			    util_buf_region_free_hndlr free_hndlr,
			    void *pool_ctx)
{
	size_t entry_sz;

	(*buf_pool) = calloc(1, sizeof(**buf_pool));
	if (!*buf_pool)
		return -FI_ENOMEM;

	(*buf_pool)->alloc_hndlr = alloc_hndlr;
	(*buf_pool)->free_hndlr = free_hndlr;
	(*buf_pool)->data_sz = size;
	(*buf_pool)->alignment = alignment;
	(*buf_pool)->max_cnt = max_cnt;
	(*buf_pool)->chunk_cnt = chunk_cnt;
	(*buf_pool)->ctx = pool_ctx;

	entry_sz = util_buf_use_ftr(*buf_pool) ?
		(size + sizeof(struct util_buf_footer)) : size;
	(*buf_pool)->entry_sz = fi_get_aligned_sz(entry_sz, alignment);

	slist_init(&(*buf_pool)->buf_list);
	slist_init(&(*buf_pool)->region_list);

	if (util_buf_grow(*buf_pool)) {
		free(*buf_pool);
		return -FI_ENOMEM;
	}
	return FI_SUCCESS;
}

#if ENABLE_DEBUG
void *util_buf_get(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	struct util_buf_footer *buf_ftr;

	entry = slist_remove_head(&pool->buf_list);
	buf_ftr = (struct util_buf_footer *) ((char *) entry + pool->data_sz);
	buf_ftr->region->num_used++;
	return entry;
}

void util_buf_release(struct util_buf_pool *pool, void *buf)
{
	union util_buf *util_buf = buf;
	struct util_buf_footer *buf_ftr;

	buf_ftr = (struct util_buf_footer *) ((char *) buf + pool->data_sz);
	buf_ftr->region->num_used--;
	slist_insert_head(&util_buf->entry, &pool->buf_list);
}
#endif

void util_buf_pool_destroy(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	struct util_buf_region *buf_region;

	while (!slist_empty(&pool->region_list)) {
		entry = slist_remove_head(&pool->region_list);
		buf_region = container_of(entry, struct util_buf_region, entry);
#if ENABLE_DEBUG
		assert(buf_region->num_used == 0);
#endif
		if (pool->free_hndlr)
			pool->free_hndlr(pool->ctx, buf_region->context);
		ofi_freealign(buf_region->mem_region);
		free(buf_region);
	}
	free(pool);
}
