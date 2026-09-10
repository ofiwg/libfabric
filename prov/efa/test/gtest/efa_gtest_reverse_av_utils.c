/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <stdlib.h>
#include "efa_gtest_reverse_av_utils.h"
#include "efa.h"
#include "efa_av.h"

/*
 * An efa_av_entry together with the efa_ah it points at, so one allocation
 * yields a complete entry. efa_av_entry stays first so a test entry pointer can
 * be cast back and forth.
 */
struct efa_test_reverse_av_entry {
	struct efa_av_entry av_entry;
	struct efa_ah ah;
};

uint64_t efa_test_reverse_av_key(uint16_t ahn, uint16_t qpn)
{
	return efa_av_reverse_av_key(ahn, qpn);
}

struct efa_av_array *efa_test_reverse_av_create(void)
{
	struct efa_av_array *arr;

	if (efa_av_reverse_av_init(&arr))
		return NULL;
	return arr;
}

void efa_test_reverse_av_destroy(struct efa_av_array *arr)
{
	efa_av_array_destroy(arr);
}

void *efa_test_reverse_av_entry_alloc(uint16_t ahn, uint16_t qpn,
				      uint64_t fi_addr)
{
	struct efa_test_reverse_av_entry *entry = calloc(1, sizeof(*entry));

	if (!entry)
		return NULL;

	entry->ah.ahn = ahn;
	entry->av_entry.ah = &entry->ah;
	entry->av_entry.fi_addr = fi_addr;
	efa_av_entry_ep_addr(&entry->av_entry)->qpn = qpn;

	return &entry->av_entry;
}

void efa_test_reverse_av_entry_free(void *entry)
{
	free(entry);
}

int efa_test_reverse_av_add(struct efa_av_array *arr, void *entry)
{
	return efa_av_reverse_av_add(arr, entry);
}

int efa_test_reverse_av_remove(struct efa_av_array *arr, void *entry)
{
	return efa_av_reverse_av_remove(arr, entry) ? 1 : 0;
}

uint64_t efa_test_reverse_av_lookup(struct efa_av_array *arr, uint16_t ahn,
				    uint16_t qpn)
{
	struct efa_av av = {0};

	av.cur_reverse_av = arr;
	return efa_av_reverse_lookup(&av, ahn, qpn);
}
