/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AV_ARRAY_H
#define EFA_AV_ARRAY_H

#include <stddef.h>
#include <ofi_mb.h>

/* Indexes below this are served from the embedded fast array. */
#define EFA_AV_ARRAY_FAST_CUTOFF 8192
/* Indexes per chunk for the region at or above the fast array. */
#define EFA_AV_ARRAY_CHUNK_SIZE 8192
/* Largest max_idx an array may be created with. */
#define EFA_AV_ARRAY_MAX_IDX_CEILING 1000000000
/* max_idx applied when the caller passes 0. */
#define EFA_AV_ARRAY_DEFAULT_MAX_IDX \
	(EFA_AV_ARRAY_FAST_CUTOFF * EFA_AV_ARRAY_CHUNK_SIZE - 1)

/*
 * A pointer array indexed by an unsigned int in [0, max_idx] for endpoint peer
 * lookup by fi_addr. Each slot holds a caller-owned pointer or NULL.
 *
 * Indexes below EFA_AV_ARRAY_FAST_CUTOFF live in the embedded fast array and are
 * reached by direct indexing. Higher indexes live in chunks reached through a
 * chunk table that is sized once at creation and never reallocated; chunks are
 * allocated on first use and are neither freed nor moved before destroy.
 *
 * Note that while this struct is thread-safe, it is not safe for
 * for concurrent writes and reads, only concurrent reads.
 */
struct efa_av_array {
	void **chunk_table;
	size_t chunk_table_len;
	unsigned max_idx;
	void *fast[EFA_AV_ARRAY_FAST_CUTOFF];
};

struct efa_av_array_attr {
	/* Largest valid index; 0 selects EFA_AV_ARRAY_DEFAULT_MAX_IDX. */
	unsigned max_idx;
};

/*
 * Allocate and initialize an array. On success returns FI_SUCCESS and sets
 * *arr; on failure returns a negative libfabric error and sets *arr to NULL.
 * index arguments to at and insert must be in [0, max_idx].
 */
int efa_av_array_init(struct efa_av_array **arr);
int efa_av_array_init_attr(struct efa_av_array **arr,
			   const struct efa_av_array_attr *attr);
void efa_av_array_destroy(struct efa_av_array *arr);

int efa_av_array_insert(struct efa_av_array *arr, unsigned index, void *entry);

/* Calls fn for each non-NULL entry, stopping early if fn returns non-zero. */
void efa_av_array_iter(struct efa_av_array *arr, void *context,
		       int (*fn)(struct efa_av_array *arr, void *entry,
				 void *context));

/*
 * Returns the pointer stored at index, or NULL if the index is out of range or
 * holds no entry.
 */
static inline void *efa_av_array_at(const struct efa_av_array *arr, unsigned index)
{
	void **table;
	void *chunk;
	void *entry;
	size_t rel;

	/* Single read memory barrier. Since this code is thread safe, but
	 * non-concurrent, we just need to make sure that we don't have
	 * stale, cached reads from before the last insert() call was made.
	 * This can be accomplished with a single rmb here.
	 */
	ofi_rmb();

	if (index > arr->max_idx)
		return NULL;

	if (index < EFA_AV_ARRAY_FAST_CUTOFF) {
		entry = arr->fast[index];
	} else {
		table = arr->chunk_table;
		if (!table)
			return NULL;

		rel = (size_t) index - EFA_AV_ARRAY_FAST_CUTOFF;
		chunk = table[rel / EFA_AV_ARRAY_CHUNK_SIZE];
		if (!chunk)
			return NULL;

		entry = ((void **) chunk)[rel % EFA_AV_ARRAY_CHUNK_SIZE];
	}

	return entry;
}

#endif /* EFA_AV_ARRAY_H */
