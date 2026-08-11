/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_av_array_utils.h"
#include "efa_av_array.h"

const int efa_test_av_array_inline_size = EFA_AV_ARRAY_INLINE_SIZE;
const int efa_test_av_array_chunk_size = EFA_AV_ARRAY_CHUNK_SIZE;
const int efa_test_av_array_max_idx_ceiling = EFA_AV_ARRAY_MAX_IDX_CEILING;
const int efa_test_av_array_default_max_idx = EFA_AV_ARRAY_DEFAULT_MAX_IDX;

struct efa_av_array *efa_test_av_array_create(void)
{
	struct efa_av_array *arr;

	if (efa_av_array_init(&arr))
		return NULL;
	return arr;
}

struct efa_av_array *efa_test_av_array_create_max(unsigned max_idx)
{
	struct efa_av_array *arr;
	struct efa_av_array_attr attr = {0};

	attr.max_idx = max_idx;
	if (efa_av_array_init_attr(&arr, &attr))
		return NULL;
	return arr;
}

struct efa_av_array *efa_test_av_array_create_attr(unsigned max_idx,
						   unsigned inline_size,
						   unsigned chunk_size)
{
	struct efa_av_array *arr;
	struct efa_av_array_attr attr = {0};

	attr.max_idx = max_idx;
	attr.inline_size = inline_size;
	attr.chunk_size = chunk_size;
	if (efa_av_array_init_attr(&arr, &attr))
		return NULL;
	return arr;
}

void efa_test_av_array_destroy(struct efa_av_array *arr)
{
	efa_av_array_destroy(arr);
}

int efa_test_av_array_insert(struct efa_av_array *arr, uint64_t index,
			     void *entry)
{
	return efa_av_array_insert(arr, index, entry);
}

void *efa_test_av_array_at(struct efa_av_array *arr, uint64_t index)
{
	return efa_av_array_at(arr, index);
}

int efa_test_av_array_has_chunk_table(struct efa_av_array *arr)
{
	return arr->chunk_table != NULL;
}

struct efa_test_av_array_count_ctx {
	int count;
};

static int efa_test_av_array_count_cb(struct efa_av_array *arr, void *entry,
				      void *context)
{
	struct efa_test_av_array_count_ctx *ctx = context;

	(void) arr;
	(void) entry;
	ctx->count++;
	return 0;
}

int efa_test_av_array_count(struct efa_av_array *arr)
{
	struct efa_test_av_array_count_ctx ctx = { 0 };

	efa_av_array_iter(arr, &ctx, efa_test_av_array_count_cb);
	return ctx.count;
}

struct efa_test_av_array_hit_ctx {
	void *target;
	int hits;
};

static int efa_test_av_array_hit_cb(struct efa_av_array *arr, void *entry,
				    void *context)
{
	struct efa_test_av_array_hit_ctx *ctx = context;

	(void) arr;
	if (entry == ctx->target) {
		ctx->hits++;
		return 1;
	}
	return 0;
}

int efa_test_av_array_iter_first_hit(struct efa_av_array *arr, void *target)
{
	struct efa_test_av_array_hit_ctx ctx = { target, 0 };

	efa_av_array_iter(arr, &ctx, efa_test_av_array_hit_cb);
	return ctx.hits;
}
