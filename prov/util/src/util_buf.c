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
#include <fi_enosys.h>
#include <fi_mem.h>
#include <fi.h>


static int util_buf_region_add(struct util_buf_pool *pool)
{
	int ret;
	size_t i;
	union util_buf *util_buf;
	struct util_buf_region *buf_region;

	buf_region = calloc(1, sizeof(*buf_region));
	if (!buf_region)
		return -1;

	ret = posix_memalign((void **)&buf_region->mem_region, pool->alignment,
			     pool->chunk_cnt * pool->entry_sz);
	if (ret) {
		free(buf_region);
		return -1;
	}

	for (i = 0; i < pool->chunk_cnt; i++) {
		util_buf = (union util_buf *)
			(buf_region->mem_region + i * pool->entry_sz);
		slist_insert_tail(&util_buf->entry, &pool->buf_list);
	}
	slist_insert_tail(&buf_region->entry, &pool->region_list);
	pool->num_allocated += pool->chunk_cnt;
	return 0;
}

struct util_buf_pool *util_buf_pool_create(size_t size, size_t alignment,
					   size_t max_cnt, size_t chunk_cnt)
{
	struct util_buf_pool *buf_pool;
	buf_pool = calloc(1, sizeof(*buf_pool));
	if (!buf_pool)
		return NULL;

	buf_pool->entry_sz = MAX(sizeof(union util_buf),
				 fi_get_aligned_sz(size, alignment));
	buf_pool->alignment = alignment;
	buf_pool->max_cnt = max_cnt;
	buf_pool->chunk_cnt = chunk_cnt;

	slist_init(&buf_pool->buf_list);
	slist_init(&buf_pool->region_list);

	if (util_buf_region_add(buf_pool)) {
		free(buf_pool);
		return NULL;
	}
	return buf_pool;
}

void *util_buf_get(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	union util_buf *buf;

	if (slist_empty(&pool->buf_list)) {
		if (pool->max_cnt == 0 || pool->max_cnt < pool->num_allocated) {
			if (util_buf_region_add(pool))
				return NULL;
		} else {
			return NULL;
		}
	}

	entry = slist_remove_head(&pool->buf_list);
	buf = container_of(entry, union util_buf, entry);

#if ENABLE_DEBUG
	pool->num_used++;
#endif
	return buf;
}

void util_buf_release(struct util_buf_pool *pool, void *buf)
{
	union util_buf *util_buf = buf;
#if ENABLE_DEBUG
	pool->num_used--;
#endif
	slist_insert_head(&util_buf->entry, &pool->buf_list);
}

void util_buf_pool_destroy(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	struct util_buf_region *buf_region;

#if ENABLE_DEBUG
	assert(pool->num_used == 0);
#endif
	while (!slist_empty(&pool->region_list)) {
		entry = slist_remove_head(&pool->region_list);
		buf_region = container_of(entry, struct util_buf_region, entry);
		free(buf_region->mem_region);
		free(buf_region);
	}
	free(pool);
}
