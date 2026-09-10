/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

/* C-linkage wrappers exposing the current reverse AV to C++ callers. */

#ifndef EFA_GTEST_REVERSE_AV_UTILS_H
#define EFA_GTEST_REVERSE_AV_UTILS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct efa_av_array;

/* The bit-interleaved index the current reverse AV uses for (ahn, qpn). */
uint64_t efa_test_reverse_av_key(uint16_t ahn, uint16_t qpn);

/* Create a reverse AV covering the whole key space; NULL on failure. */
struct efa_av_array *efa_test_reverse_av_create(void);
void efa_test_reverse_av_destroy(struct efa_av_array *arr);

/*
 * Allocate a fake AV entry carrying only what the reverse AV reads: the AHN of
 * its AH, the raw address QPN, and the fi_addr a lookup returns. Returns a
 * struct efa_av_entry * as an opaque pointer; NULL on failure.
 */
void *efa_test_reverse_av_entry_alloc(uint16_t ahn, uint16_t qpn,
				      uint64_t fi_addr);
void efa_test_reverse_av_entry_free(void *entry);

int efa_test_reverse_av_add(struct efa_av_array *arr, void *entry);
/* Non-zero if the entry was still the current one for its key and was removed. */
int efa_test_reverse_av_remove(struct efa_av_array *arr, void *entry);
/* efa_av_reverse_lookup() against arr; FI_ADDR_NOTAVAIL when absent. */
uint64_t efa_test_reverse_av_lookup(struct efa_av_array *arr, uint16_t ahn,
				    uint16_t qpn);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_REVERSE_AV_UTILS_H */
