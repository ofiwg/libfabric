/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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
#include <fi_enosys.h>
#include <fi_util.h>
#include <assert.h>
#include <rbtree.h>

/* key is `addr + len` value */
static int util_rtc_compare_keys(void *key1, void *key2)
{
	uint64_t k1 = *((uint64_t *) key1);
	uint64_t k2 = *((uint64_t *) key2);
	return (k1 < k2) ? -1 : (k1 > k2);
}

static int util_rtc_tree_init(struct util_rtc *rtc)
{
	rtc->rb_tree = rbtNew(util_rtc_compare_keys);
	return rtc->rb_tree ? FI_SUCCESS : -FI_ENOMEM;
}

static void util_rtc_tree_close(struct util_rtc *rtc)
{
	rbtDelete(rtc->rb_tree);
}

int ofi_util_rtc_init(struct fid_domain *domain_fid, struct util_rtc_attr *attr,
		      struct util_rtc **rtc)
{
	int ret;
	
	*rtc = calloc(1, sizeof(*rtc));
	if (!*rtc) {
		ret = -FI_ENOMEM;
		goto fn1;
	}

	(*rtc)->domain_fid = domain_fid;
	(*rtc)->total_num_entries_thr = attr->total_num_entries_thr;
	(*rtc)->total_size_thr = attr->total_size_thr;

	(*rtc)->buf_pool = util_buf_pool_create(sizeof(struct util_rtc_entry), 16,
						attr->total_num_entries_thr,
						attr->total_num_entries_thr);
	if (!(*rtc)->buf_pool) {
		ret = -FI_ENOMEM;
		goto fn2;
	}
	ret = util_rtc_tree_init(*rtc);
	if (ret)
		goto fn3;

	return FI_SUCCESS;
fn3:
	util_buf_pool_destroy((*rtc)->buf_pool);
fn2:
	free(*rtc);
	*rtc = NULL;
fn1:
	return ret;
}

int ofi_util_rtc_close(struct util_rtc *rtc)
{
	util_rtc_tree_close(rtc);
	util_buf_pool_destroy(rtc->buf_pool);
	free(rtc);
	return FI_SUCCESS;
}

static inline int util_rtc_insert(struct util_rtc *rtc, void *buf, size_t len)
{
	uint64_t key = (uint64_t)((uint64_t)(uintptr_t)buf + len);
	struct util_rtc_entry *item = util_buf_alloc(rtc->buf_pool);
	if (OFI_UNLIKELY(!item))
		return -FI_ENOMEM;
	rbtInsert(rtc->rb_tree, &key, item);
	item->key = key;

	return FI_SUCCESS;
}
