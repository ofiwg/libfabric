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

int ofi_util_rtc_init(struct util_rtc_attr *attr, struct util_rtc **rtc)
{
	int ret;
	
	*rtc = calloc(1, sizeof(**rtc));
	if (!*rtc) {
		ret = -FI_ENOMEM;
		goto fn1;
	}
	(*rtc)->total_num_entries_thr = attr->total_num_entries_thr;
	(*rtc)->total_size_thr = attr->total_size_thr;
	fastlock_init(&(*rtc)->lock);
	dlist_init(&(*rtc)->queue_list);
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

	(*rtc)->prov_reg_mr = attr->prov_reg_mr;
	if (attr->reg_arg && attr->reg_arg_len) {
		(*rtc)->reg_arg = mem_dup(attr->reg_arg,
					  attr->reg_arg_len);
		if (!(*rtc)->reg_arg) {
			ret = -FI_ENOMEM;
			goto fn4;
		}
	}

	(*rtc)->prov_dereg_mr = attr->prov_dereg_mr;
	if (attr->dereg_arg && attr->dereg_arg_len) {
		(*rtc)->dereg_arg = mem_dup(attr->dereg_arg,
					    attr->dereg_arg_len);
		if (!(*rtc)->dereg_arg) {
			ret = -FI_ENOMEM;
			goto fn5;
		}
	}

	return FI_SUCCESS;
fn5:
	free((*rtc)->reg_arg);
fn4:
	util_rtc_tree_close(*rtc);
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
	fastlock_destroy(&rtc->lock);
	free(rtc->reg_arg);
	free(rtc->dereg_arg);
	free(rtc);

	return FI_SUCCESS;
}

static inline int util_rtc_insert(struct util_rtc *rtc, uint64_t *key,
				  void *buf, size_t len,
				  struct util_rtc_entry **item)
{
	struct util_rtc_entry *rtc_item = *item = util_buf_alloc(rtc->buf_pool);
	if (OFI_UNLIKELY(!rtc_item))
		return -FI_ENOMEM;

	rtc_item->buf = buf;
	rtc_item->len = len;
	rtc_item->key = *key;
	rtc_item->in_use = 1;

	fastlock_acquire(&rtc->lock);
	rbtInsert(rtc->rb_tree, &rtc_item->key, rtc_item);
	dlist_insert_tail(&rtc_item->list_entry, &rtc->queue_list);
	fastlock_release(&rtc->lock);

	return FI_SUCCESS;
}

static inline void util_rtc_reinsert(struct util_rtc *rtc,
				     struct util_rtc_entry *item)
{
	fastlock_acquire(&rtc->lock);
	dlist_remove(&item->list_entry);
	item->in_use = 1;
	dlist_insert_tail(&rtc->queue_list, &item->list_entry);
	fastlock_release(&rtc->lock);
}

/* This function assumes that the item has already been dequeued */
static inline int util_rtc_remove(struct util_rtc *rtc,
				  struct util_rtc_entry *item)
{
	int ret;
	void *key_ptr, *itr = rbtFind(rtc->rb_tree, &item->key);
	if (!itr)
		return -FI_ENOKEY;
	rbtKeyValue(rtc->rb_tree, itr, &key_ptr, (void **)item);

	ret = rtc->prov_dereg_mr(item->mr, item->buf, item->len,
				 rtc->dereg_arg);
	util_buf_release(rtc->buf_pool, item);
	rbtErase(rtc->rb_tree, itr);

	return ret;
}

static inline struct util_rtc_entry *
util_rtc_lookup(struct util_rtc *rtc, uint64_t *key)
{
	struct util_rtc_entry *item;
	void *key_ptr, *itr = rbtFind(rtc->rb_tree, key);
	if (!itr)
		return NULL;
	rbtKeyValue(rtc->rb_tree, itr, &key_ptr, (void **)&item);

	return item;
}

static inline int util_rtc_is_rtc_entry_in_use(struct dlist_entry *entry,
					       const void *arg)
{
	OFI_UNUSED(arg);
	struct util_rtc_entry *item = container_of(entry, struct util_rtc_entry,
						   list_entry);
	return item->in_use;
}

static inline void util_rtc_make_avail_space(struct util_rtc *rtc)
{
	if (!util_buf_avail(rtc->buf_pool)) {
		struct util_rtc_entry *item;
		struct dlist_entry *entry =
			dlist_remove_first_match(
				&rtc->queue_list,
				util_rtc_is_rtc_entry_in_use,
				NULL);
		if (!entry)
			return;
		item = container_of(entry, struct util_rtc_entry, list_entry);
		(void) util_rtc_remove(rtc, item);
	}
}

static int util_rtc_full_reg_buffer(struct util_rtc *rtc, void **prov_mr,
				    uint64_t *key, void *buf, size_t len)
{
	int ret;
	struct util_rtc_entry *item;

	util_rtc_make_avail_space(rtc);
	ret = util_rtc_insert(rtc, key, buf, len, &item);
	if (ret) {
		*prov_mr = NULL;
		return ret;
	}

	ret = rtc->prov_reg_mr(&item->mr, buf, len, rtc->reg_arg);
	if (ret) {
		*prov_mr = NULL;
		return ret;
	}

	*prov_mr = item->mr;

	return ret;
}

static int util_rtc_light_reg_buffer(struct util_rtc *rtc, struct util_rtc_entry *item,
				     void **prov_mr, void *buf, size_t len)
{
	util_rtc_reinsert(rtc, item);
	*prov_mr = item->mr;

	return FI_SUCCESS;
}

int ofi_rtc_reg_buffer(struct util_rtc *rtc, void *buf, size_t len,
		       void **prov_mr)
{
	uint64_t key = (uint64_t)((uint64_t)(uintptr_t)buf + len);
	struct util_rtc_entry *item = util_rtc_lookup(rtc, &key);
	if (!item)
		return util_rtc_full_reg_buffer(rtc, prov_mr, &key,
						buf, len);
	else
		return util_rtc_light_reg_buffer(rtc, item, prov_mr,
						 buf, len);
}

int ofi_rtc_dereg_buffer(struct util_rtc *rtc, void *buf, size_t len)
{
	uint64_t key = (uint64_t)((uint64_t)(uintptr_t)buf + len);
	struct util_rtc_entry *item = util_rtc_lookup(rtc, &key);
	if (item) {
		item->in_use = 0;
		return FI_SUCCESS;
	} else {
		return -FI_ENOENT;
	}
}
