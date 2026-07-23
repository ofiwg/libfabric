/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/* C-linkage wrappers for ofi_dyn_arr. ofi_indexer.h pulls in ofi_osd.h, which
 * does not compile under C++, so tests reach the array through these. */

#ifndef EFA_GTEST_DYN_ARR_UTILS_H
#define EFA_GTEST_DYN_ARR_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ofi_dyn_arr;

/* item_size is the size of each element (tests store an int). */
struct ofi_dyn_arr *dyn_arr_create(size_t item_size);
/* Same, but every newly grown slot is pre-filled with dyn_arr_sentinel(). */
struct ofi_dyn_arr *dyn_arr_create_with_sentinel(size_t item_size);
void dyn_arr_destroy(struct ofi_dyn_arr *arr);

void *dyn_arr_at(struct ofi_dyn_arr *arr, int index);
void *dyn_arr_at_max(struct ofi_dyn_arr *arr, int index, int max_size);

/* Count slots in allocated chunks whose leading int equals value. */
int dyn_arr_count_value(struct ofi_dyn_arr *arr, int value);

/* Iterate, stopping at the first slot equal to value (callback returns
 * non-zero). Returns the number of matches visited before stopping. */
int dyn_arr_iter_first_hit(struct ofi_dyn_arr *arr, int value);

int dyn_arr_default_max_index(void);
int dyn_arr_default_chunk_size(void);
int dyn_arr_sentinel(void);

/* Create an array with a custom maximum capacity (max_cnt items). */
struct ofi_dyn_arr *dyn_arr_create_max(size_t item_size, size_t max_cnt);
/* The array's highest valid index. */
long dyn_arr_get_max_index(struct ofi_dyn_arr *arr);

/* Create an array with a custom chunk size (items per chunk). */
struct ofi_dyn_arr *dyn_arr_create_chunk_cnt(size_t item_size, size_t chunk_cnt);
/* The array's actual (possibly rounded-up) chunk size. */
int dyn_arr_chunk_size(struct ofi_dyn_arr *arr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_DYN_ARR_UTILS_H */
