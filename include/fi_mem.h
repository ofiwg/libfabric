/*
 * Copyright (c) 2015-2016 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _FI_MEM_H_
#define _FI_MEM_H_

#include <config.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <fi_list.h>
#include <fi_osd.h>


#ifdef INCLUDE_VALGRIND
#   include <valgrind/memcheck.h>
#   ifndef VALGRIND_MAKE_MEM_DEFINED
#      warning "Valgrind requested, but VALGRIND_MAKE_MEM_DEFINED undefined"
#   endif
#endif

#ifndef VALGRIND_MAKE_MEM_DEFINED
#   define VALGRIND_MAKE_MEM_DEFINED(addr, len)
#endif

/* We implement memdup to avoid external library dependency */
static inline void *mem_dup(const void *src, size_t size)
{
	void *dest;

	if ((dest = malloc(size)))
		memcpy(dest, src, size);
	return dest;
}


/*
 * Buffer pool (free stack) template
 */
#define FREESTACK_EMPTY	NULL

#define freestack_isempty(fs)	((fs)->next == FREESTACK_EMPTY)
#define freestack_push(fs, p)					\
{								\
	*(void **) p = (fs)->next;				\
	(fs)->next = p;						\
}
#define freestack_pop(fs)					\
	({							\
		void *p = (fs)->next;				\
		assert(!freestack_isempty(fs));			\
		(fs)->next = *((void **) p);			\
		p;						\
	})

#define DECLARE_FREESTACK(entrytype, name)			\
struct name {							\
	size_t		size;					\
	void		*next;					\
	entrytype	buf[];					\
};								\
								\
static inline void name ## _init(struct name *fs, size_t size)	\
{								\
	int i;							\
	assert(size == roundup_power_of_two(size));		\
	assert(sizeof(fs->buf[0]) >= sizeof(void *));		\
	fs->size = size;					\
	fs->next = FREESTACK_EMPTY;				\
	for (i = size - 1; i >= 0; i--)				\
		freestack_push(fs, &fs->buf[i]);		\
}								\
								\
static inline struct name * name ## _create(size_t size)	\
{								\
	struct name *fs;					\
	fs = calloc(1, sizeof(*fs) + sizeof(entrytype) *	\
		    (roundup_power_of_two(size)));		\
	if (fs)							\
		name ##_init(fs, roundup_power_of_two(size));	\
	return fs;						\
}								\
								\
static inline int name ## _index(struct name *fs,		\
		entrytype *entry)				\
{								\
	return entry - fs->buf;					\
}								\
								\
static inline void name ## _free(struct name *fs)		\
{								\
	free(fs);						\
}


/*
 * Buffer Pool
 */
struct util_buf_pool;
typedef int (*util_buf_region_alloc_hndlr) (void *pool_ctx, void *addr, size_t len,
					    void **context);
typedef void (*util_buf_region_free_hndlr) (void *pool_ctx, void *context);

struct util_buf_pool {
	size_t data_sz;
	size_t entry_sz;
	size_t max_cnt;
	size_t chunk_cnt;
	size_t alignment;
	size_t num_allocated;
	struct slist buf_list;
	struct slist region_list;
	util_buf_region_alloc_hndlr alloc_hndlr;
	util_buf_region_free_hndlr free_hndlr;
	void *ctx;
};

struct util_buf_region {
	struct slist_entry entry;
	char *mem_region;
	void *context;
#if ENABLE_DEBUG
	size_t num_used;
#endif
};

struct util_buf_footer {
	struct util_buf_region *region;
};

union util_buf {
	struct slist_entry entry;
	uint8_t data[0];
};

/* create buffer pool with alloc/free handlers */
struct util_buf_pool *util_buf_pool_create_ex(size_t size, size_t alignment,
					      size_t max_cnt, size_t chunk_cnt,
					      util_buf_region_alloc_hndlr alloc_hndlr,
					      util_buf_region_free_hndlr free_hndlr,
					      void *pool_ctx);

/* create buffer pool */
static inline struct util_buf_pool *util_buf_pool_create(size_t size,
							 size_t alignment,
							 size_t max_cnt,
							 size_t chunk_cnt)
{
	return util_buf_pool_create_ex(size, alignment, max_cnt, chunk_cnt,
				       NULL, NULL, NULL);
}

static inline int util_buf_avail(struct util_buf_pool *pool)
{
	return !slist_empty(&pool->buf_list);
}

int util_buf_grow(struct util_buf_pool *pool);

#if ENABLE_DEBUG

void *util_buf_get(struct util_buf_pool *pool);
void util_buf_release(struct util_buf_pool *pool, void *buf);

#else

static inline void *util_buf_get(struct util_buf_pool *pool)
{
	struct slist_entry *entry;
	entry = slist_remove_head(&pool->buf_list);
	return entry;
}

static inline void util_buf_release(struct util_buf_pool *pool, void *buf)
{
	union util_buf *util_buf = buf;
	slist_insert_head(&util_buf->entry, &pool->buf_list);
}
#endif

static inline void *util_buf_get_ex(struct util_buf_pool *pool, void **context)
{
	union util_buf *buf;
	struct util_buf_footer *buf_ftr;

	buf = util_buf_get(pool);
	buf_ftr = (struct util_buf_footer *) ((char *) buf + pool->data_sz);
	assert(context);
	*context = buf_ftr->region->context;
	return buf;
}

static inline void *util_buf_alloc(struct util_buf_pool *pool)
{
	if (!util_buf_avail(pool)) {
		if (util_buf_grow(pool))
			return NULL;
	}
	return util_buf_get(pool);
}

static inline void *util_buf_alloc_ex(struct util_buf_pool *pool, void **context)
{
	union util_buf *buf;
	struct util_buf_footer *buf_ftr;

	buf = util_buf_alloc(pool);
	buf_ftr = (struct util_buf_footer *) ((char *) buf + pool->data_sz);
	assert(context);
	*context = buf_ftr->region->context;
	return buf;
}

void util_buf_pool_destroy(struct util_buf_pool *pool);

#endif /* _FI_MEM_H_ */
