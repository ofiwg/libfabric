/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
 * Copyright (c) 2018-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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


static void ofi_bufpool_set_region_size(struct ofi_bufpool *pool)
{
	ssize_t hp_size;

	if (pool->entry_cnt) {
		assert(pool->alloc_size && pool->region_size);
		return;
	}

	hp_size = ofi_get_hugepage_size();
	if (pool->attr.flags & OFI_BUFPOOL_MMAPPED)
		pool->alloc_size = ofi_get_aligned_size((pool->attr.chunk_cnt + 1) *
							pool->entry_size, hp_size);
	else
		pool->alloc_size = (pool->attr.chunk_cnt + 1) * pool->entry_size;
	pool->region_size = pool->alloc_size - pool->entry_size;
}

int ofi_bufpool_grow(struct ofi_bufpool *pool)
{
	struct ofi_bufpool_region *buf_region;
	struct ofi_bufpool_hdr *buf_hdr;
	void *buf;
	int ret;
	size_t i;

	if (pool->attr.max_cnt && pool->entry_cnt >= pool->attr.max_cnt)
		return -FI_EINVAL;

	ofi_bufpool_set_region_size(pool);
	buf_region = calloc(1, sizeof(*buf_region));
	if (!buf_region)
		return -FI_ENOSPC;

	buf_region->pool = pool;
	dlist_init(&buf_region->free_list);

	if (pool->attr.flags & OFI_BUFPOOL_MMAPPED) {
		ret = ofi_alloc_hugepage_buf((void **) &buf_region->alloc_region,
					     pool->alloc_size);
		/* If we can't allocate huge pages, fall back to normal
		 * allocations if this is the first allocation attempt.
		 */
		if (ret && !pool->entry_cnt) {
			pool->attr.flags &= ~OFI_BUFPOOL_MMAPPED;
			pool->alloc_size = (pool->attr.chunk_cnt + 1) *
					   pool->entry_size;
			pool->region_size = pool->alloc_size - pool->entry_size;
			goto retry;
		}
	} else {
retry:
		ret = ofi_memalign((void **) &buf_region->alloc_region,
				   pool->attr.alignment, pool->alloc_size);
	}
	if (ret) {
		FI_DBG(&core_prov, FI_LOG_CORE, "Allocation failed: %s\n",
		       fi_strerror(-ret));
		goto err1;
	}

	memset(buf_region->alloc_region, 0, pool->alloc_size);
	buf_region->mem_region = buf_region->alloc_region + pool->entry_size;
	if (pool->attr.alloc_fn) {
		ret = pool->attr.alloc_fn(buf_region);
		if (ret)
			goto err2;
	}

	if (!(pool->region_cnt % OFI_BUFPOOL_REGION_CHUNK_CNT)) {
		struct ofi_bufpool_region **new_table;

		new_table = realloc(pool->region_table,
				(pool->region_cnt + OFI_BUFPOOL_REGION_CHUNK_CNT) *
				sizeof(*pool->region_table));
		if (!new_table) {
			ret = -FI_ENOMEM;
			goto err3;
		}
		pool->region_table = new_table;
	}
	pool->region_table[pool->region_cnt] = buf_region;
	buf_region->index = pool->region_cnt++;

	for (i = 0; i < pool->attr.chunk_cnt; i++) {
		buf = (buf_region->mem_region + i * pool->entry_size);
		buf_hdr = ofi_buf_hdr(buf);

		if (pool->attr.init_fn) {
#if ENABLE_DEBUG
			if (pool->attr.flags & OFI_BUFPOOL_INDEXED) {
				buf_hdr->entry.dlist.next = (void *) OFI_MAGIC_64;
				buf_hdr->entry.dlist.prev = (void *) OFI_MAGIC_64;

				pool->attr.init_fn(buf_region, buf);

				assert((buf_hdr->entry.dlist.next == (void *) OFI_MAGIC_64) &&
				       (buf_hdr->entry.dlist.prev == (void *) OFI_MAGIC_64));
			} else {
				buf_hdr->entry.slist.next = (void *) OFI_MAGIC_64;

				pool->attr.init_fn(buf_region, buf);

				assert(buf_hdr->entry.slist.next == (void *) OFI_MAGIC_64);
			}
#else
			pool->attr.init_fn(buf_region, buf);
#endif
		}

		buf_hdr->region = buf_region;
		buf_hdr->index = pool->entry_cnt + i;
		if (pool->attr.flags & OFI_BUFPOOL_INDEXED) {
			dlist_insert_tail(&buf_hdr->entry.dlist,
					  &buf_region->free_list);
		} else {
			slist_insert_tail(&buf_hdr->entry.slist,
					  &pool->free_list.entries);
		}
	}

	if (pool->attr.flags & OFI_BUFPOOL_INDEXED)
		dlist_insert_tail(&buf_region->entry, &pool->free_list.regions);

	pool->entry_cnt += pool->attr.chunk_cnt;
	return 0;

err3:
	if (pool->attr.free_fn)
	    pool->attr.free_fn(buf_region);
err2:
	if (pool->attr.flags & OFI_BUFPOOL_MMAPPED)
		ofi_free_hugepage_buf(buf_region->alloc_region, pool->alloc_size);
	else
		ofi_freealign(buf_region->alloc_region);
err1:
	free(buf_region);
	return ret;
}

int ofi_bufpool_create_attr(struct ofi_bufpool_attr *attr,
			      struct ofi_bufpool **buf_pool)
{
	size_t entry_sz;

	(*buf_pool) = calloc(1, sizeof(**buf_pool));
	if (!*buf_pool)
		return -FI_ENOMEM;

	(*buf_pool)->attr = *attr;

	entry_sz = (attr->size + sizeof(struct ofi_bufpool_hdr));
	(*buf_pool)->entry_size = ofi_get_aligned_size(entry_sz, attr->alignment);

	if (!attr->chunk_cnt) {
		(*buf_pool)->attr.chunk_cnt =
			((*buf_pool)->entry_size < page_sizes[OFI_PAGE_SIZE]) ?
			64 : 16;
	}

	if ((*buf_pool)->attr.flags & OFI_BUFPOOL_INDEXED)
		dlist_init(&(*buf_pool)->free_list.regions);
	else
		slist_init(&(*buf_pool)->free_list.entries);

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
		.context	= pool_ctx,
	};
	return ofi_bufpool_create_attr(&attr, buf_pool);
}

void ofi_bufpool_destroy(struct ofi_bufpool *pool)
{
	struct ofi_bufpool_region *buf_region;
	int ret;
	size_t i;

	for (i = 0; i < pool->region_cnt; i++) {
		buf_region = pool->region_table[i];

		assert((pool->attr.flags & OFI_BUFPOOL_NO_TRACK) ||
			(buf_region->use_cnt == 0));
		if (pool->attr.free_fn)
			pool->attr.free_fn(buf_region);

		if (pool->attr.flags & OFI_BUFPOOL_MMAPPED) {
			ret = ofi_free_hugepage_buf(buf_region->alloc_region,
						    pool->alloc_size);
			if (ret) {
				FI_DBG(&core_prov, FI_LOG_CORE,
				       "Huge page free failed: %s\n",
				       fi_strerror(-ret));
				assert(0);
			}
		} else {
			ofi_freealign(buf_region->alloc_region);
		}

		free(buf_region);
	}
	free(pool->region_table);
	free(pool);
}

int ofi_ibuf_is_lower(struct dlist_entry *item, const void *arg)
{
	struct ofi_bufpool_hdr *hdr1, *hdr2;

	hdr1 = container_of(arg, struct ofi_bufpool_hdr, entry.dlist);
	hdr2 = container_of(item, struct ofi_bufpool_hdr, entry.dlist);

	return hdr1->index < hdr2->index;
}

int ofi_ibufpool_region_is_lower(struct dlist_entry *item, const void *arg)
{
	struct ofi_bufpool_region *reg1, *reg2;

	reg1 = container_of(arg, struct ofi_bufpool_region, entry);
	reg2 = container_of(item, struct ofi_bufpool_region, entry);

	return reg1->index < reg2->index;
}
