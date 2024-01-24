/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef _OFI_RCU_H_
#define _OFI_RCU_H_

#include <stddef.h>
#include <stdlib.h>

#include <ofi_list.h>


struct ofi_rcu_list {
	void **items;
	size_t num_items;
};

/**
 * @brief  RCU uses a non-atomic swap for performance reasons.
 *         There will only ever be one writer at a time b/c of
 *         mutex lock and a pointer is an atomic sized word
 *
 * @param old Pointer to return
 * @param new Pointer to swap into old pointer
 * @return struct ofi_rcu_list*
 */
static inline struct ofi_rcu_list *
ofi_rcu_list_swap(struct ofi_rcu_list **old, struct ofi_rcu_list *new)
{
	struct ofi_rcu_list *tmp = *old;
	*old = new;
	return tmp;
}

/**
 * @param num_items Size of RCU list to create
 * @return struct ofi_rcu_list* Newly created list
 */
static inline struct ofi_rcu_list*
ofi_rcu_list_create(size_t num_items)
{
	struct ofi_rcu_list* list = calloc(1, sizeof(struct ofi_rcu_list));
	if (!list)
		return list;

	list->items = calloc(num_items, sizeof(void *));
	list->num_items = num_items;
	return list;
}

/**
 * @param dst  list to copy into
 * @param src list to copy from
 * @param size size of src list
 */
static inline void
ofi_rcu_list_copy(struct ofi_rcu_list* dst, struct ofi_rcu_list* src, size_t size)
{
	assert(dst->num_items >= src->num_items);

	for (int i = 0; i < size; i++) {
		dst->items[i] = src->items[i];
	}
}

/**
 * @param list to destroy
 */
static inline void
ofi_rcu_list_destroy(struct ofi_rcu_list** list)
{
	(*list)->num_items = 0;
	free((*list)->items);
	free(*list);
	*list = NULL;
}

/**
 * @brief Copies a list with all but one item
 *
 * @param old_list The original list
 * @param item The item to not include in new list
 * @return struct ofi_rcu_list* The new list
 */
static inline struct ofi_rcu_list*
ofi_rcu_list_clone_list_without_item(struct ofi_rcu_list* old_list, void *item)
{
	struct ofi_rcu_list *new_rcu_list = NULL;
	int i, j;

	new_rcu_list = ofi_rcu_list_create(old_list->num_items - 1);
	if (old_list->num_items > 1) {
		for (i = 0, j = 0; i < old_list->num_items; i++) {
			if (old_list->items[i] != item) {
				new_rcu_list->items[j] = old_list->items[i];
				j++;
			}
		}
	}

	return new_rcu_list;
}

/**
 * @brief Checks to see if item is in list
 *
 * @param list The list to iterate through
 * @param item The item to check if it is in list
 * @return true Item is in the list
 * @return false Item is not in the list
 */
static inline bool
ofi_rcu_is_item_in_list(struct ofi_rcu_list* list, void *item)
{
	int i;

	for (i = 0; i < list->num_items; i++) {
		if (list->items[i] == item)
			return true;
	}

	return false;
}

#endif /* _OFI_RCU_H_ */
