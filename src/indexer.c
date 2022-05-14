/*
 * Copyright (c) 2011 Intel Corporation.  All rights reserved.
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
 *
 */

#include "config.h"

#include <errno.h>
#include <sys/types.h>
#include <stdlib.h>
#include <assert.h>
#include <ofi_indexer.h>

/*
 * Indexer - to find a structure given an index
 *
 * We store pointers using a double lookup and return an index to the
 * user which is then used to retrieve the pointer.  The upper bits of
 * the index are itself an index into an array of memory allocations.
 * The lower bits specify the offset into the allocated memory where
 * the pointer is stored.
 *
 * This allows us to adjust the number of pointers stored by the index
 * list without taking a lock during data lookups.
 */

static int ofi_idx_grow(struct indexer *idx)
{
	struct ofi_idx_entry *chunk;
	int i, start_index;

	if (idx->size >= OFI_IDX_MAX_CHUNKS)
		goto nomem;

	idx->chunk[idx->size] = calloc(OFI_IDX_CHUNK_SIZE, sizeof(struct ofi_idx_entry));
	if (!idx->chunk[idx->size])
		goto nomem;

	chunk = idx->chunk[idx->size];
	start_index = idx->size << OFI_IDX_ENTRY_BITS;
	chunk[OFI_IDX_CHUNK_SIZE - 1].next = idx->free_list;

	for (i = OFI_IDX_CHUNK_SIZE - 2; i >= 0; i--)
		chunk[i].next = start_index + i + 1;

	/* Index 0 is reserved */
	if (start_index == 0)
		start_index++;
	idx->free_list = start_index;
	idx->size++;
	return start_index;

nomem:
	errno = ENOMEM;
	return -1;
}

int ofi_idx_insert(struct indexer *idx, void *item)
{
	struct ofi_idx_entry *chunk;
	int index;

	if ((index = idx->free_list) == 0) {
		if ((index = ofi_idx_grow(idx)) <= 0)
			return index;
	}

	chunk = idx->chunk[ofi_idx_array_index(index)];
	idx->free_list = chunk[ofi_idx_entry_index(index)].next;
	chunk[ofi_idx_entry_index(index)].item = item;
	return index;
}

void *ofi_idx_remove(struct indexer *idx, int index)
{
	struct ofi_idx_entry *chunk;
	void *item;
	int entry_index = ofi_idx_entry_index(index);

	chunk = idx->chunk[ofi_idx_array_index(index)];
	item = chunk[entry_index].item;
	chunk[entry_index].item = NULL;
	chunk[entry_index].next = idx->free_list;
	idx->free_list = index;
	return item;
}

void *ofi_idx_remove_ordered(struct indexer *idx, int index)
{
	struct ofi_idx_entry *chunk;
	void *item;
	int temp_index;
	int entry_index = ofi_idx_entry_index(index);

	chunk = idx->chunk[ofi_idx_array_index(index)];
	item = chunk[entry_index].item;
	chunk[entry_index].item = NULL;
	if (ofi_idx_free_list_empty(idx) || index < idx->free_list) {
		chunk[entry_index].next = idx->free_list;
		idx->free_list = index;
		return item;
	}
	temp_index = idx->free_list;
	while (chunk[ofi_idx_entry_index(temp_index)].next < index) {
		temp_index = chunk[ofi_idx_entry_index(temp_index)].next;
	}
	chunk[entry_index].next = chunk[ofi_idx_entry_index(temp_index)].next;
	chunk[ofi_idx_entry_index(temp_index)].next = index;

	return item;
}

void ofi_idx_replace(struct indexer *idx, int index, void *item)
{
	struct ofi_idx_entry *chunk;

	chunk = idx->chunk[ofi_idx_array_index(index)];
	chunk[ofi_idx_entry_index(index)].item = item;
}

void ofi_idx_reset(struct indexer *idx)
{
	while (idx->size) {
		free(idx->chunk[idx->size - 1]);
		idx->chunk[idx->size - 1] = NULL;
		idx->size--;
	}
	idx->free_list = 0;
}

static int ofi_idm_grow(struct index_map *idm, int index)
{
	idm->chunk[ofi_idx_array_index(index)] = calloc(OFI_IDX_CHUNK_SIZE, sizeof(void *));
	if (!idm->chunk[ofi_idx_array_index(index)])
		goto nomem;

	return index;

nomem:
	errno = ENOMEM;
	return -1;
}

int ofi_idm_set(struct index_map *idm, int index, void *item)
{
	void **chunk;

	if (index > OFI_IDX_MAX_INDEX) {
		errno = ENOMEM;
		return -1;
	}

	if (!idm->chunk[ofi_idx_array_index(index)]) {
		if (ofi_idm_grow(idm, index) < 0)
			return -1;
	}

	chunk = idm->chunk[ofi_idx_array_index(index)];
	chunk[ofi_idx_entry_index(index)] = item;
	idm->count[ofi_idx_array_index(index)]++;
	return index;
}

void *ofi_idm_clear(struct index_map *idm, int index)
{
	void **chunk;
	void *item;

	chunk = idm->chunk[ofi_idx_array_index(index)];
	item = chunk[ofi_idx_entry_index(index)];
	chunk[ofi_idx_entry_index(index)] = NULL;
	if (--idm->count[ofi_idx_array_index(index)] == 0) {
		free(idm->chunk[ofi_idx_array_index(index)]);
		idm->chunk[ofi_idx_array_index(index)] = NULL;
	}
	return item;
}

void ofi_idm_reset(struct index_map *idm, void (*callback)(void *item))
{
	void **chunk;
	void *item;
	int a, i;

	for (a = 0; a < OFI_IDX_MAX_CHUNKS; a++) {
		if (!idm->chunk[a]) {
			assert(idm->count[a] == 0);
			continue;
		}

		for (i = 0; idm->count[a] && i < OFI_IDX_CHUNK_SIZE; i++) {
			chunk = idm->chunk[a];
			item = chunk[i];
			if (item) {
				if (callback)
					callback(item);
				idm->count[a]--;
			}
		}
		free(idm->chunk[a]);
		idm->chunk[a] = NULL;
	}
}

