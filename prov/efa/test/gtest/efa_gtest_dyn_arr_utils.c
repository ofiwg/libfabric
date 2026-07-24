/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_dyn_arr_utils.h"
#include <stdlib.h>
#include "ofi_indexer.h"

#define DYN_ARR_TEST_SENTINEL 0x5a5a5a5a

static void dyn_arr_sentinel_init(struct ofi_dyn_arr *arr, void *item)
{
	(void) arr;
	*(int *) item = DYN_ARR_TEST_SENTINEL;
}

struct ofi_dyn_arr *dyn_arr_create(size_t item_size)
{
	struct ofi_dyn_arr *arr = malloc(sizeof(*arr));

	if (arr)
		ofi_array_init(arr, item_size, NULL);
	return arr;
}

struct ofi_dyn_arr *dyn_arr_create_with_sentinel(size_t item_size)
{
	struct ofi_dyn_arr *arr = malloc(sizeof(*arr));

	if (arr)
		ofi_array_init(arr, item_size, dyn_arr_sentinel_init);
	return arr;
}

void dyn_arr_destroy(struct ofi_dyn_arr *arr)
{
	ofi_array_destroy(arr);
	free(arr);
}

void *dyn_arr_at(struct ofi_dyn_arr *arr, int index)
{
	return ofi_array_at(arr, index);
}

void *dyn_arr_at_max(struct ofi_dyn_arr *arr, int index, int max_size)
{
	return ofi_array_at_max(arr, index, max_size);
}

struct dyn_arr_count_ctx {
	int value;
	int count;
};

static int dyn_arr_count_cb(struct ofi_dyn_arr *arr, void *item, void *context)
{
	struct dyn_arr_count_ctx *ctx = context;

	(void) arr;
	if (*(int *) item == ctx->value)
		ctx->count++;
	return 0;
}

int dyn_arr_count_value(struct ofi_dyn_arr *arr, int value)
{
	struct dyn_arr_count_ctx ctx = { value, 0 };

	ofi_array_iter(arr, &ctx, dyn_arr_count_cb);
	return ctx.count;
}

struct dyn_arr_hit_ctx {
	int value;
	int hits;
};

static int dyn_arr_hit_cb(struct ofi_dyn_arr *arr, void *item, void *context)
{
	struct dyn_arr_hit_ctx *ctx = context;

	(void) arr;
	if (*(int *) item == ctx->value) {
		ctx->hits++;
		return 1; /* non-zero return stops iteration */
	}
	return 0;
}

int dyn_arr_iter_first_hit(struct ofi_dyn_arr *arr, int value)
{
	struct dyn_arr_hit_ctx ctx = { value, 0 };

	ofi_array_iter(arr, &ctx, dyn_arr_hit_cb);
	return ctx.hits;
}

int dyn_arr_default_max_index(void)
{
	return OFI_IDX_MAX_INDEX;
}

int dyn_arr_default_chunk_size(void)
{
	return OFI_IDX_CHUNK_SIZE;
}

int dyn_arr_sentinel(void)
{
	return DYN_ARR_TEST_SENTINEL;
}

struct ofi_dyn_arr *dyn_arr_create_max(size_t item_size, size_t max_cnt)
{
	struct ofi_dyn_arr *arr = malloc(sizeof(*arr));
	struct ofi_dyn_arr_attr attr = {0};

	if (!arr)
		return NULL;

	attr.item_size = item_size;
	attr.max_cnt = max_cnt;
	if (ofi_array_init_attr(arr, &attr)) {
		free(arr);
		return NULL;
	}
	return arr;
}

long dyn_arr_get_max_index(struct ofi_dyn_arr *arr)
{
	return (long) arr->max_index;
}

struct ofi_dyn_arr *dyn_arr_create_chunk_cnt(size_t item_size, size_t chunk_cnt)
{
	struct ofi_dyn_arr *arr = malloc(sizeof(*arr));
	struct ofi_dyn_arr_attr attr = {0};

	if (!arr)
		return NULL;

	attr.item_size = item_size;
	attr.chunk_cnt = chunk_cnt;
	if (ofi_array_init_attr(arr, &attr)) {
		free(arr);
		return NULL;
	}
	return arr;
}

int dyn_arr_chunk_size(struct ofi_dyn_arr *arr)
{
	return (int) arr->chunk_size;
}
