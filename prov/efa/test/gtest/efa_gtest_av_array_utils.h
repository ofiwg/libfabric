/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/* C-linkage wrappers exposing efa_av_array operations to C++ callers. */

#ifndef EFA_GTEST_AV_ARRAY_UTILS_H
#define EFA_GTEST_AV_ARRAY_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

struct efa_av_array;

/* Create with the default max_idx; NULL on failure. */
struct efa_av_array *efa_test_av_array_create(void);
/* Create with a custom max_idx; NULL if the value is rejected. */
struct efa_av_array *efa_test_av_array_create_max(unsigned max_idx);
void efa_test_av_array_destroy(struct efa_av_array *arr);

int efa_test_av_array_insert(struct efa_av_array *arr, unsigned index,
			     void *entry);
void *efa_test_av_array_at(struct efa_av_array *arr, unsigned index);

/* Non-zero once the array has allocated its chunk table. */
int efa_test_av_array_has_chunk_table(struct efa_av_array *arr);
/* Number of non-NULL entries, via iteration. */
int efa_test_av_array_count(struct efa_av_array *arr);
/* Iterate, stopping at the first entry equal to target; 1 if found, else 0. */
int efa_test_av_array_iter_first_hit(struct efa_av_array *arr, void *target);

extern const int efa_test_av_array_fast_cutoff;
extern const int efa_test_av_array_chunk_size;
extern const int efa_test_av_array_max_idx_ceiling;
extern const int efa_test_av_array_default_max_idx;

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_AV_ARRAY_UTILS_H */
