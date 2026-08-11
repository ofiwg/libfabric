/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_AV_ARRAY_H
#define EFA_AV_ARRAY_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <ofi_mb.h>

#define EFA_AV_ARRAY_INLINE_SIZE 8192
#define EFA_AV_ARRAY_CHUNK_SIZE 8192
/* Largest max_idx an array may be created with. */
#define EFA_AV_ARRAY_MAX_IDX_CEILING 1000000000
/* max_idx applied when the caller passes 0. */
#define EFA_AV_ARRAY_DEFAULT_MAX_IDX \
	(EFA_AV_ARRAY_INLINE_SIZE * EFA_AV_ARRAY_CHUNK_SIZE - 1)

/*
 * A pointer array indexed by a uint64_t in [0, max_idx] for endpoint peer
 * lookup by fi_addr. Each slot holds a caller-owned pointer or NULL.
 *
 * Indexes less than inline_size live in the inline array and are reached by direct
 * indexing. Higher indexes live in chunks reached through a chunk table that is
 * sized once at creation and never reallocated; chunks are allocated on first
 * use and are neither freed nor moved before destroy.
 *
 * insert() publishes each newly reachable pointer after a write barrier and
 * at() consumes them behind a read barrier, so a single writer may run
 * concurrently with any number of lock-free readers. This works for both of our
 * use cases:
 *   1) the av array itself, which is only written by fi_av_insert() and cannot
 *      be read until that insert finishes (stated in the man pages)
 *   2) peer maps, which are added to lazily, so a reader may run concurrently
 *      with a writer
 */
struct efa_av_array {
	void **chunk_table;
	size_t chunk_table_len;
	unsigned max_idx;
	unsigned inline_size;
	unsigned chunk_size;
	/* Flexible array member allocated with the struct; must stay last. */
	void *inline_entries[];
};

_Static_assert(offsetof(struct efa_av_array, inline_entries) % 8 == 0,
	       "efa_av_array inline_entries must be 8-byte aligned");

struct efa_av_array_attr {
	/* Largest valid index; 0 selects EFA_AV_ARRAY_DEFAULT_MAX_IDX. */
	unsigned max_idx;
	unsigned inline_size;
	unsigned chunk_size;
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

int efa_av_array_insert(struct efa_av_array *arr, uint64_t index, void *entry);

/* Calls fn for each non-NULL entry, stopping early if fn returns non-zero. */
void efa_av_array_iter(struct efa_av_array *arr, void *context,
		       int (*fn)(struct efa_av_array *arr, void *entry,
				 void *context));

/*
 * Returns the pointer stored at index, or NULL if the index is out of range or
 * holds no entry.
 *
 * Concurrency: one writer, any number of lock-free readers.
 * This relies on a structural invariance of the fi_addr_t usage-- once a
 * a pointer is published, the pointer is good for the life of the array,
 * meaning that we only need a single rmb() at the beginning of the read
 * function, as long as we are careful about publishing pointers in the
 * correct order and after their data is written.
 *
 * An observant reader might note that we DO have a remove function, but
 * The man pages state that a remove CANNOT be called with in-flight
 * packets though (think: why would a user want this anyways??) so a remove
 * will not race with a lookup, meaning that the single rmb() guarantees
 * safety.
 */
static inline void *efa_av_array_at(const struct efa_av_array *arr, uint64_t index)
{
	void **table;
	void *chunk;
	size_t rel;

	/* This rmb() is defensive, but possibly not necessary. More
	 * research would be needed
	 */
	ofi_rmb();

	if (index > arr->max_idx)
		return NULL;

	if (index < arr->inline_size)
		return arr->inline_entries[index];

	table = arr->chunk_table;
	if (!table)
		return NULL;

	rel = (size_t) index - arr->inline_size;
	chunk = table[rel / arr->chunk_size];
	if (!chunk)
		return NULL;

	return ((void **) chunk)[rel % arr->chunk_size];
}

#endif /* EFA_AV_ARRAY_H */
