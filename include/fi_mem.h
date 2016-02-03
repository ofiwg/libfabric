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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <fi.h>
#include <fi_list.h>

#ifndef _FI_MEM_H_
#define _FI_MEM_H_

/*
 * Buffer Pool
 */

struct util_buf_pool {
	size_t entry_sz;
	size_t max_cnt;
	size_t chunk_cnt;
	size_t alignment;
	size_t num_allocated;
#if ENABLE_DEBUG
	size_t num_used;
#endif
	struct slist buf_list;
	struct slist region_list;
};

struct util_buf_region {
	struct slist_entry entry;
	char *mem_region;
};

union util_buf {
	struct slist_entry entry;
	uint8_t data[0];
};

struct util_buf_pool *util_buf_pool_create(size_t size, size_t alignment,
					   size_t max_cnt, size_t chunk_cnt);
void util_buf_pool_destroy(struct util_buf_pool *pool);

void *util_buf_get(struct util_buf_pool *pool);
void util_buf_release(struct util_buf_pool *pool, void *buf);

#endif
