/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <stdlib.h>
#include <ofi_mb.h>
#include "efa.h"
#include "efa_av_array.h"

int efa_av_array_init_attr(struct efa_av_array **arr_out,
			   const struct efa_av_array_attr *attr)
{
	struct efa_av_array *arr;
	unsigned max_idx = attr ? attr->max_idx : 0;
	unsigned inline_size = (attr && attr->inline_size) ?
			       attr->inline_size : EFA_AV_ARRAY_INLINE_SIZE;
	unsigned chunk_size = (attr && attr->chunk_size) ?
			      attr->chunk_size : EFA_AV_ARRAY_CHUNK_SIZE;

	*arr_out = NULL;

	if (max_idx == 0) {
		size_t def = (size_t) inline_size * chunk_size - 1;

		max_idx = def > EFA_AV_ARRAY_MAX_IDX_CEILING ?
			  EFA_AV_ARRAY_MAX_IDX_CEILING : (unsigned) def;
	}
	if (max_idx > EFA_AV_ARRAY_MAX_IDX_CEILING) {
		EFA_WARN(FI_LOG_AV, "efa_av_array max_idx %u out of range\n",
			 max_idx);
		return -FI_EINVAL;
	}

	arr = calloc(1, sizeof(*arr) +
		     (size_t) inline_size * sizeof(*arr->inline_entries));
	if (!arr)
		return -FI_ENOMEM;

	arr->max_idx = max_idx;
	arr->inline_size = inline_size;
	arr->chunk_size = chunk_size;
	if (max_idx >= inline_size) {
		size_t span = (size_t) max_idx - inline_size + 1;

		arr->chunk_table_len = (span + chunk_size - 1) / chunk_size;
	}

	*arr_out = arr;
	return FI_SUCCESS;
}

int efa_av_array_init(struct efa_av_array **arr)
{
	return efa_av_array_init_attr(arr, NULL);
}

void efa_av_array_destroy(struct efa_av_array *arr)
{
	size_t i;

	if (!arr)
		return;

	if (arr->chunk_table) {
		for (i = 0; i < arr->chunk_table_len; i++)
			free(arr->chunk_table[i]);
		free(arr->chunk_table);
	}
	free(arr);
}

/*
 * IMPORTANT: must be called with a lock to serialize writes.
 *
 * Publishing model (single writer, any number of lock-free readers):
 *
 * All backing memory -- the entry payload and calloc'd chunk/cunk table --
 * is fully initialized before the single ofi_wmb() so that every pointer
 * that is published is a valid pointer.
 *
 * A reader (efa_av_array_at) issues one ofi_rmb() before walking
 * chunk_table -> chunk -> entry. Any pointer that is NULL in this chain
 * just indicates "no entry," which is a valid, safe return for a racing
 * insert and read call.
 *
 * see comments in efa_av_array.h for more context and information.
 */
int efa_av_array_insert(struct efa_av_array *arr, uint64_t index, void *entry)
{
	void **table;
	void **chunk;
	size_t rel, chunk_idx, off;
	bool new_table = false;
	bool new_chunk = false;

	if (index > arr->max_idx)
		return -FI_EINVAL;

	if (index < arr->inline_size) {
		/* Make sure the caller's entry is written before publishing */
		ofi_wmb();
		arr->inline_entries[index] = entry;
		return FI_SUCCESS;
	}

	rel = (size_t) index - arr->inline_size;
	chunk_idx = rel / arr->chunk_size;
	off = rel % arr->chunk_size;

	table = arr->chunk_table;
	if (OFI_UNLIKELY(!table)) {
		table = calloc(arr->chunk_table_len, sizeof(*table));
		if (!table)
			return -FI_ENOMEM;
		new_table = true;
	}

	chunk = table[chunk_idx];
	if (OFI_UNLIKELY(!chunk)) {
		chunk = calloc(arr->chunk_size, sizeof(*chunk));
		if (!chunk) {
			if (new_table)
				free(table);
			return -FI_ENOMEM;
		}
		new_chunk = true;
	}

	/*
	 * We just need to guarantee that memory is initialized before
	 * publishing the pointer to it. See function comment for more info
	 */
	ofi_wmb();
	chunk[off] = entry;

	/* avoid re-publishing pointers to avoid cache invalidations */
	if (OFI_UNLIKELY(new_chunk))
		table[chunk_idx] = chunk;
	if (OFI_UNLIKELY(new_table))
		arr->chunk_table = table;

	return FI_SUCCESS;
}

void efa_av_array_iter(struct efa_av_array *arr, void *context,
		       int (*fn)(struct efa_av_array *arr, void *entry,
				 void *context))
{
	size_t i, j;

	for (i = 0; i < arr->inline_size; i++) {
		if (arr->inline_entries[i] &&
		    fn(arr, arr->inline_entries[i], context))
			return;
	}

	if (!arr->chunk_table)
		return;

	for (i = 0; i < arr->chunk_table_len; i++) {
		void **chunk = (void **) arr->chunk_table[i];

		if (!chunk)
			continue;
		for (j = 0; j < arr->chunk_size; j++) {
			if (chunk[j] && fn(arr, chunk[j], context))
				return;
		}
	}
}
