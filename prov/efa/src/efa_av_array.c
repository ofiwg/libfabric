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

	*arr_out = NULL;

	if (max_idx == 0)
		max_idx = EFA_AV_ARRAY_DEFAULT_MAX_IDX;
	if (max_idx > EFA_AV_ARRAY_MAX_IDX_CEILING) {
		EFA_WARN(FI_LOG_AV, "efa_av_array max_idx %u out of range\n",
			 max_idx);
		return -FI_EINVAL;
	}

	arr = calloc(1, sizeof(*arr));
	if (!arr)
		return -FI_ENOMEM;

	arr->max_idx = max_idx;
	if (max_idx >= EFA_AV_ARRAY_FAST_CUTOFF) {
		size_t span = (size_t) max_idx - EFA_AV_ARRAY_FAST_CUTOFF + 1;

		arr->chunk_table_len = (span + EFA_AV_ARRAY_CHUNK_SIZE - 1) /
				       EFA_AV_ARRAY_CHUNK_SIZE;
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

/* This function must be protected by a lock. The lock will guarantee that
 * only 1 writer can insert at a time and will make sure there is a memory
 * barrier before entering and after exiting.
 * This is sufficient because the man pages state that a program must wait
 * for fi_av_insert to return before doing work that requires that AV.
 */
int efa_av_array_insert(struct efa_av_array *arr, unsigned index, void *entry)
{
	void **table = NULL;
	void **chunk = NULL;
	size_t rel, chunk_idx;

	if (index > arr->max_idx)
		return -FI_EINVAL;

	if (index < EFA_AV_ARRAY_FAST_CUTOFF) {
		arr->fast[index] = entry;
		return FI_SUCCESS;
	}

	if (OFI_UNLIKELY(!arr->chunk_table)) {
		table = calloc(arr->chunk_table_len, sizeof(*table));
		if (!table)
			return -FI_ENOMEM;

		arr->chunk_table = table;
	}

	table = arr->chunk_table;

	rel = (size_t) index - EFA_AV_ARRAY_FAST_CUTOFF;
	chunk_idx = rel / EFA_AV_ARRAY_CHUNK_SIZE;

	chunk = table[chunk_idx];
	if (OFI_UNLIKELY(!chunk)) {
		chunk = calloc(EFA_AV_ARRAY_CHUNK_SIZE, sizeof(*chunk));
		if (!chunk)
			return -FI_ENOMEM;

		table[chunk_idx] = chunk;
	}

	chunk[rel % EFA_AV_ARRAY_CHUNK_SIZE] = entry;
	return FI_SUCCESS;
}

void efa_av_array_iter(struct efa_av_array *arr, void *context,
		       int (*fn)(struct efa_av_array *arr, void *entry,
				 void *context))
{
	size_t i, j;

	for (i = 0; i < EFA_AV_ARRAY_FAST_CUTOFF; i++) {
		if (arr->fast[i] && fn(arr, arr->fast[i], context))
			return;
	}

	if (!arr->chunk_table)
		return;

	for (i = 0; i < arr->chunk_table_len; i++) {
		void **chunk = (void **) arr->chunk_table[i];

		if (!chunk)
			continue;
		for (j = 0; j < EFA_AV_ARRAY_CHUNK_SIZE; j++) {
			if (chunk[j] && fn(arr, chunk[j], context))
				return;
		}
	}
}
