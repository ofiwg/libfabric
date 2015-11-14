/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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
#include <stdio.h>
#include <rdma/fi_errno.h>

#include "gnix_hashtable.h"
#include "fasthash/fasthash.h"

#include "gnix_util.h"

#define __GNIX_HT_INITIAL_SIZE 128
#define __GNIX_HT_MAXIMUM_SIZE 1024
#define __GNIX_HT_INCREASE_STEP 2

#define __GNIX_HT_COLLISION_THRESH 400 /* average of 4 elements per bucket */

/*
 * __gnix_ht_lf* prefixes denote lock free version of functions intended for
 *   use with hashtables that had attr->ht_internal_locking set to zero
 *   during initialization
 *
 * __gnix_ht_lk* prefixes denote locking versions of functions intended for
 *   use with hash tables that had attr->ht_internal_locking set to a non-zero
 *   value during initialization
 */

/*
 * default_attr is global for a criterion test.
 */
gnix_hashtable_attr_t default_attr = {
		.ht_initial_size     = __GNIX_HT_INITIAL_SIZE,
		.ht_maximum_size     = __GNIX_HT_MAXIMUM_SIZE,
		.ht_increase_step    = __GNIX_HT_INCREASE_STEP,
		.ht_increase_type    = GNIX_HT_INCREASE_MULT,
		.ht_collision_thresh = __GNIX_HT_COLLISION_THRESH,
		.ht_hash_seed        = 0,
		.ht_internal_locking = 0,
		.destructor          = NULL
};

static gnix_hashtable_ops_t __gnix_lockless_ht_ops;
static gnix_hashtable_ops_t __gnix_locked_ht_ops;

static int __gnix_ht_check_attr_sanity(gnix_hashtable_attr_t *attr)
{
	if (attr->ht_initial_size == 0 ||
			attr->ht_initial_size > attr->ht_maximum_size)
		return -FI_EINVAL;

	if (attr->ht_maximum_size == 0)
		return -FI_EINVAL;

	if (attr->ht_increase_step == 0)
		return -FI_EINVAL;

	if (!(attr->ht_increase_type == GNIX_HT_INCREASE_ADD ||
			attr->ht_increase_type == GNIX_HT_INCREASE_MULT))
		return -FI_EINVAL;

	if (attr->ht_increase_step == 1 &&
			attr->ht_increase_type == GNIX_HT_INCREASE_MULT)
		return -FI_EINVAL;

	if (attr->ht_collision_thresh == 0)
		return -FI_EINVAL;

	return 0;
}

static inline void __gnix_ht_delete_entry(gnix_ht_entry_t *ht_entry)
{
	dlist_remove(&ht_entry->entry);

	ht_entry->value = NULL;
	ht_entry->key = 0;
	free(ht_entry);
}

static inline void __gnix_ht_init_lk_list_head(gnix_ht_lk_lh_t *lh)
{
	dlist_init(&lh->head);
	rwlock_init(&lh->lh_lock);
}

static inline void __gnix_ht_init_lf_list_head(gnix_ht_lf_lh_t *lh)
{
	dlist_init(&lh->head);
}

static inline gnix_ht_key_t __gnix_hash_func(
		gnix_hashtable_t *ht,
		gnix_ht_key_t key)
{
	return fasthash64(&key, sizeof(gnix_ht_key_t),
			ht->ht_attr.ht_hash_seed) % ht->ht_size;
}

static inline gnix_ht_entry_t *__gnix_ht_lookup_entry(
		struct dlist_entry *head,
		gnix_ht_key_t key,
		uint64_t *collision_count)
{
	gnix_ht_entry_t *ht_entry;

	if (dlist_empty(head))
		return NULL;

	dlist_for_each(head, ht_entry, entry) {
		if (ht_entry->key == key)
			return ht_entry;

		if (collision_count)
			*collision_count += 1;
	}

	return NULL;
}

static inline void *__gnix_ht_lookup_key(
		struct dlist_entry *head,
		gnix_ht_key_t key)
{
	gnix_ht_entry_t *ht_entry = __gnix_ht_lookup_entry(head, key, NULL);

	if (!ht_entry)
		return NULL;

	return ht_entry->value;
}

static inline int __gnix_ht_destroy_list(
		gnix_hashtable_t *ht,
		struct dlist_entry *head)
{
	gnix_ht_entry_t *ht_entry, *iter;
	void *value;
	int entries_freed = 0;

	dlist_for_each_safe(head, ht_entry, iter, entry) {
		value = ht_entry->value;
		__gnix_ht_delete_entry(ht_entry);
		if (ht->ht_attr.destructor != NULL) {
			ht->ht_attr.destructor(value);
		}
		++entries_freed;
	}

	return entries_freed;
}

static inline int __gnix_ht_insert_list(
		struct dlist_entry *head,
		gnix_ht_entry_t *ht_entry,
		uint64_t *collisions)
{
	gnix_ht_entry_t *found;

	found = __gnix_ht_lookup_entry(head, ht_entry->key, collisions);
	if (!found) {
		dlist_insert_tail(&ht_entry->entry, head);
	} else {
		return -FI_ENOSPC;
	}

	return 0;
}

static inline int __gnix_ht_remove_list(
		struct dlist_entry *head,
		gnix_ht_key_t key)
{
	gnix_ht_entry_t *ht_entry;

	ht_entry = __gnix_ht_lookup_entry(head, key, NULL);
	if (!ht_entry) {
		return -FI_ENOENT;
	}
	__gnix_ht_delete_entry(ht_entry);

	return 0;
}

static inline void __gnix_ht_rehash_list(
		gnix_hashtable_t *ht,
		struct dlist_entry *head)
{
	gnix_ht_entry_t *ht_entry, *tmp;
	gnix_ht_key_t bucket;
	struct dlist_entry *ht_lh;

	if (dlist_empty(head))
		return;

	dlist_for_each_safe(head, ht_entry, tmp, entry) {
		bucket = __gnix_hash_func(ht, ht_entry->key);
		ht_lh = ht->ht_ops->retrieve_list(ht, bucket);

		dlist_remove(&ht_entry->entry);

		__gnix_ht_insert_list(ht_lh, ht_entry, NULL);
	}
}

static inline void __gnix_ht_resize_hashtable_inc(gnix_hashtable_t *ht)
{
	int old_size = ht->ht_size;
	int new_size;

	/* set up the new bucket list size */
	if (ht->ht_attr.ht_increase_type == GNIX_HT_INCREASE_ADD)
		new_size = old_size + ht->ht_attr.ht_increase_step;
	else
		new_size = old_size * ht->ht_attr.ht_increase_step;

	new_size = MIN(new_size, ht->ht_attr.ht_maximum_size);

	/* ignore ret code for now. In the future, we might provide an info
	 *   if the hash table wont resize. It is generally a performance
	 *   issue if we cannot, and not really a bug.
	 */

	ht->ht_ops->resize(ht, new_size, old_size);
}

static inline void __gnix_ht_resize_hashtable_dec(gnix_hashtable_t *ht)
{
	int old_size = ht->ht_size;
	int new_size;

	/* set up the new bucket list size */
	if (ht->ht_attr.ht_increase_type == GNIX_HT_INCREASE_ADD)
		new_size = old_size - ht->ht_attr.ht_increase_step;
	else
		new_size = old_size / ht->ht_attr.ht_increase_step;

	new_size = MAX(new_size, ht->ht_attr.ht_initial_size);

	/* ignore ret code for now. In the future, we might provide an info
	 *   if the hash table wont resize. It is generally a performance
	 *   issue if we cannot, and not really a bug.
	 */

	ht->ht_ops->resize(ht, new_size, old_size);
}

static inline void __gnix_ht_common_init(gnix_hashtable_t *ht)
{
	if (ht->ht_state == GNIX_HT_STATE_UNINITIALIZED) {
		atomic_initialize(&ht->ht_elements, 0);
		atomic_initialize(&ht->ht_collisions, 0);
		atomic_initialize(&ht->ht_insertions, 0);
	} else {
		atomic_set(&ht->ht_elements, 0);
		atomic_set(&ht->ht_collisions, 0);
		atomic_set(&ht->ht_insertions, 0);
	}

	ht->ht_state = GNIX_HT_STATE_READY;
}

static inline void __gnix_ht_common_destroy(gnix_hashtable_t *ht)
{
	ht->ht_size = 0;
	atomic_set(&ht->ht_collisions, 0);
	atomic_set(&ht->ht_insertions, 0);
	atomic_set(&ht->ht_elements, 0);
	ht->ht_state = GNIX_HT_STATE_DEAD;
}

static gnix_ht_lf_lh_t *__gnix_ht_lf_init_new_table(int nelem)
{
	int i;
	gnix_ht_lf_lh_t *tbl = calloc(nelem, sizeof(gnix_ht_lf_lh_t));

	if (!tbl)
		return NULL;

	for (i = 0; i < nelem; ++i)
		__gnix_ht_init_lf_list_head(&tbl[i]);

	return tbl;
}

static int __gnix_ht_lf_init(gnix_hashtable_t *ht)
{
	ht->ht_lf_tbl = __gnix_ht_lf_init_new_table(ht->ht_size);
	if (!ht->ht_lf_tbl)
		return -FI_ENOMEM;

	__gnix_ht_common_init(ht);

	return 0;
}

static int __gnix_ht_lf_destroy(gnix_hashtable_t *ht)
{
	int i, freed_entries;
	gnix_ht_lf_lh_t *lh;

	for (i = 0; i < ht->ht_size; ++i) {
		lh = &ht->ht_lf_tbl[i];

		freed_entries = __gnix_ht_destroy_list(ht, &lh->head);

		if (freed_entries)
			atomic_sub(&ht->ht_elements, freed_entries);
	}

	free(ht->ht_lf_tbl);
	ht->ht_lf_tbl = NULL;

	__gnix_ht_common_destroy(ht);

	return 0;
}

static int __gnix_ht_lf_insert(
		gnix_hashtable_t *ht,
		gnix_ht_entry_t *entry,
		uint64_t *collisions)
{
	int ret;
	int bucket;
	gnix_ht_lf_lh_t *lh;

	bucket = __gnix_hash_func(ht, entry->key);
	lh = &ht->ht_lf_tbl[bucket];

	ret = __gnix_ht_insert_list(&lh->head, entry, collisions);

	return ret;
}

static int __gnix_ht_lf_remove(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	int ret;
	int bucket;

	bucket = __gnix_hash_func(ht, key);
	ret = __gnix_ht_remove_list(&ht->ht_lf_tbl[bucket].head, key);

	return ret;
}

static void *__gnix_ht_lf_lookup(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	int bucket = __gnix_hash_func(ht, key);
	gnix_ht_lf_lh_t *lh = &ht->ht_lf_tbl[bucket];

	return __gnix_ht_lookup_key(&lh->head, key);
}

static struct dlist_entry *__gnix_ht_lf_retrieve_list(
		gnix_hashtable_t *ht,
		int bucket)
{
	if (bucket < 0 || bucket >= ht->ht_size)
			return NULL;

	return &ht->ht_lf_tbl[bucket].head;
}

static int __gnix_ht_lf_resize(
		gnix_hashtable_t *ht,
		int new_size,
		int old_size)
{
	gnix_ht_lf_lh_t *old_tbl, *new_tbl;
	int i;

	if (ht->ht_size != old_size)
		return -FI_EBUSY;

	new_tbl = __gnix_ht_lf_init_new_table(new_size);
	if (!new_tbl)
		return -FI_ENOMEM;

	old_tbl = ht->ht_lf_tbl;
	ht->ht_lf_tbl = new_tbl;
	ht->ht_size = new_size;

	for (i = 0; i < old_size; ++i)
		__gnix_ht_rehash_list(ht, &old_tbl[i].head);

	free(old_tbl);

	return 0;
}

static gnix_ht_lk_lh_t *__gnix_ht_lk_init_new_table(int nelem)
{
	int i;
	gnix_ht_lk_lh_t *tbl = calloc(nelem, sizeof(gnix_ht_lk_lh_t));

	if (!tbl)
		return NULL;

	for (i = 0; i < nelem; ++i)
		__gnix_ht_init_lk_list_head(&tbl[i]);

	return tbl;
}

static int __gnix_ht_lk_init(gnix_hashtable_t *ht)
{
	if (ht->ht_state != GNIX_HT_STATE_DEAD)
		rwlock_init(&ht->ht_lock);

	rwlock_wrlock(&ht->ht_lock);

	ht->ht_lk_tbl = __gnix_ht_lk_init_new_table(ht->ht_size);
	if (!ht->ht_lk_tbl) {
		rwlock_unlock(&ht->ht_lock);
		return -FI_ENOMEM;
	}

	__gnix_ht_common_init(ht);

	rwlock_unlock(&ht->ht_lock);

	return 0;
}

static int __gnix_ht_lk_destroy(gnix_hashtable_t *ht)
{
	int i, freed_entries;
	gnix_ht_lk_lh_t *lh;

	if (ht->ht_state != GNIX_HT_STATE_READY)
		return -FI_EINVAL;

	rwlock_wrlock(&ht->ht_lock);

	for (i = 0; i < ht->ht_size; ++i) {
		lh = &ht->ht_lk_tbl[i];

		freed_entries = __gnix_ht_destroy_list(ht, &lh->head);

		if (freed_entries)
			atomic_sub(&ht->ht_elements, freed_entries);
	}

	free(ht->ht_lk_tbl);
	ht->ht_lk_tbl = NULL;

	__gnix_ht_common_destroy(ht);

	rwlock_unlock(&ht->ht_lock);

	return 0;
}

static int __gnix_ht_lk_insert(
		gnix_hashtable_t *ht,
		gnix_ht_entry_t *entry,
		uint64_t *collisions)
{
	int ret, bucket;
	gnix_ht_lk_lh_t *lh;

	rwlock_rdlock(&ht->ht_lock);

	bucket = __gnix_hash_func(ht, entry->key);
	lh = &ht->ht_lk_tbl[bucket];

	rwlock_wrlock(&lh->lh_lock);
	ret = __gnix_ht_insert_list(&lh->head, entry, collisions);
	rwlock_unlock(&lh->lh_lock);

	rwlock_unlock(&ht->ht_lock);

	return ret;
}

static int __gnix_ht_lk_remove(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	int ret;
	int bucket;
	gnix_ht_lk_lh_t *lh;

	rwlock_rdlock(&ht->ht_lock);

	bucket = __gnix_hash_func(ht, key);
	lh = &ht->ht_lk_tbl[bucket];

	rwlock_wrlock(&lh->lh_lock);
	ret = __gnix_ht_remove_list(&lh->head, key);
	rwlock_unlock(&lh->lh_lock);

	rwlock_unlock(&ht->ht_lock);

	return ret;
}

static void *__gnix_ht_lk_lookup(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	void *ret;
	int bucket;
	gnix_ht_lk_lh_t *lh;

	rwlock_rdlock(&ht->ht_lock);

	bucket = __gnix_hash_func(ht, key);
	lh = &ht->ht_lk_tbl[bucket];

	rwlock_rdlock(&lh->lh_lock);
	ret = __gnix_ht_lookup_key(&lh->head, key);
	rwlock_unlock(&lh->lh_lock);

	rwlock_unlock(&ht->ht_lock);

	return ret;
}

static struct dlist_entry *__gnix_ht_lk_retrieve_list(
		gnix_hashtable_t *ht,
		int bucket)
{
	if (bucket < 0 || bucket >= ht->ht_size)
		return NULL;

	return &ht->ht_lk_tbl[bucket].head;
}

static int __gnix_ht_lk_resize(
		gnix_hashtable_t *ht,
		int new_size,
		int old_size)
{
	int i;
	gnix_ht_lk_lh_t *old_tbl, *new_tbl;

	/* race to resize... let one of them resize the hash table and the rest
	 * can just release after the first is done.
	 */
	rwlock_wrlock(&ht->ht_lock);
	if (ht->ht_size != old_size) {
		rwlock_unlock(&ht->ht_lock);
		return -FI_EBUSY;
	}

	new_tbl = __gnix_ht_lk_init_new_table(new_size);
	if (!new_tbl) {
		rwlock_unlock(&ht->ht_lock);
		return -FI_ENOMEM;
	}

	old_tbl = ht->ht_lk_tbl;
	ht->ht_lk_tbl = new_tbl;
	ht->ht_size = new_size;

	for (i = 0; i < old_size; ++i)
		__gnix_ht_rehash_list(ht, &old_tbl[i].head);

	free(old_tbl);

	rwlock_unlock(&ht->ht_lock);

	return 0;
}

static inline int __gnix_ht_should_decrease_size(gnix_hashtable_t *ht)
{
	int decrease;
	int desired_thresh = (ht->ht_attr.ht_collision_thresh >> 2) * 3;

	if (ht->ht_attr.ht_increase_type == GNIX_HT_INCREASE_ADD)
		decrease = ht->ht_attr.ht_increase_step;
	else
		decrease = ht->ht_size / ht->ht_attr.ht_increase_step;

	/* This is just an approximation of the collision rate since we
	 *     don't track collisions on removal
	 */
	return ((atomic_get(&ht->ht_elements) * 100) /
			(ht->ht_size - decrease)) <= desired_thresh;
}

int _gnix_ht_init(gnix_hashtable_t *ht, gnix_hashtable_attr_t *attr)
{
	int ret;
	gnix_hashtable_attr_t *tbl_attr = &default_attr;

	if (attr) {
		ret = __gnix_ht_check_attr_sanity(attr);
		if (ret < 0)
			return ret;

		tbl_attr = attr;
	}

	if (ht->ht_state == GNIX_HT_STATE_READY)
		return -FI_EINVAL;

	memcpy(&ht->ht_attr, tbl_attr, sizeof(gnix_hashtable_attr_t));
	ht->ht_size = ht->ht_attr.ht_initial_size;

	if (ht->ht_attr.ht_internal_locking)
		ht->ht_ops = &__gnix_locked_ht_ops;
	else
		ht->ht_ops = &__gnix_lockless_ht_ops;

	return ht->ht_ops->init(ht);
}

int _gnix_ht_destroy(gnix_hashtable_t *ht)
{
	if (ht->ht_state != GNIX_HT_STATE_READY)
		return -FI_EINVAL;

	return ht->ht_ops->destroy(ht);
}

int _gnix_ht_insert(gnix_hashtable_t *ht, gnix_ht_key_t key, void *value)
{
	int ret;
	int collisions, insertions;
	uint64_t hits = 0;
	gnix_ht_entry_t *list_entry;

	if (ht->ht_state != GNIX_HT_STATE_READY)
		return -FI_EINVAL;

	list_entry = calloc(1, sizeof(gnix_ht_entry_t));
	if (!list_entry)
		return -FI_ENOMEM;

	list_entry->value = value;
	list_entry->key = key;

	ret = ht->ht_ops->insert(ht, list_entry, &hits);
	if (ret != 0) {
		free(list_entry);
		return ret;
	}

	if (ht->ht_size < ht->ht_attr.ht_maximum_size) {
		collisions = atomic_add(&ht->ht_collisions, hits);
		insertions = atomic_inc(&ht->ht_insertions);
		if (insertions > 10 &&
				((collisions * 100) / insertions)
				> ht->ht_attr.ht_collision_thresh) {

			atomic_set(&ht->ht_collisions, 0);
			atomic_set(&ht->ht_insertions, 0);

			__gnix_ht_resize_hashtable_inc(ht);
		}
	}

	atomic_inc(&ht->ht_elements);

	return ret;
}

int _gnix_ht_remove(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	int ret;

	if (ht->ht_state != GNIX_HT_STATE_READY)
		return -FI_EINVAL;

	ret = ht->ht_ops->remove(ht, key);

	/* on success, we may have to resize */
	if (ret == 0) {
		atomic_dec(&ht->ht_elements);

		if (ht->ht_size > ht->ht_attr.ht_initial_size &&
				__gnix_ht_should_decrease_size(ht)) {

			/* since we are resizing the table,
			 * reset the collision info
			 */
			atomic_set(&ht->ht_collisions, 0);
			atomic_set(&ht->ht_insertions, 0);

			__gnix_ht_resize_hashtable_dec(ht);
		}
	}

	return ret;
}

void *_gnix_ht_lookup(gnix_hashtable_t *ht, gnix_ht_key_t key)
{
	if (ht->ht_state != GNIX_HT_STATE_READY)
		return NULL;

	return ht->ht_ops->lookup(ht, key);
}

int _gnix_ht_empty(gnix_hashtable_t *ht)
{
	return atomic_get(&ht->ht_elements) == 0;
}

static gnix_hashtable_ops_t __gnix_lockless_ht_ops = {
		.init          = __gnix_ht_lf_init,
		.destroy       = __gnix_ht_lf_destroy,
		.insert        = __gnix_ht_lf_insert,
		.remove        = __gnix_ht_lf_remove,
		.lookup        = __gnix_ht_lf_lookup,
		.resize        = __gnix_ht_lf_resize,
		.retrieve_list = __gnix_ht_lf_retrieve_list
};

static gnix_hashtable_ops_t __gnix_locked_ht_ops = {
		.init          = __gnix_ht_lk_init,
		.destroy       = __gnix_ht_lk_destroy,
		.insert        = __gnix_ht_lk_insert,
		.remove        = __gnix_ht_lk_remove,
		.lookup        = __gnix_ht_lk_lookup,
		.resize        = __gnix_ht_lk_resize,
		.retrieve_list = __gnix_ht_lk_retrieve_list
};
