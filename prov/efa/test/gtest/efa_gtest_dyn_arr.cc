/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_dyn_arr_utils.h"
#include <climits>
#include <cstdint>
#include <gtest/gtest.h>

using testing::Test;

class OfiDynArrTest : public Test
{
	protected:
	struct ofi_dyn_arr *arr = nullptr;

	void SetUp() override
	{
		arr = dyn_arr_create(sizeof(int));
		ASSERT_NE(arr, nullptr);
	}

	void TearDown() override
	{
		if (arr)
			dyn_arr_destroy(arr);
	}

	int *at(int index)
	{
		return (int *) dyn_arr_at(arr, index);
	}

	int *at_max(int index, int max)
	{
		return (int *) dyn_arr_at_max(arr, index, max);
	}
};

/* A fresh array has no chunks, so at_max never returns a slot. */
TEST_F(OfiDynArrTest, empty_lookup_returns_null)
{
	EXPECT_EQ(at_max(0, dyn_arr_default_max_index() + 1), nullptr);
	EXPECT_EQ(at_max(1000, dyn_arr_default_max_index() + 1), nullptr);
}

/* at() allocates the backing chunk and zero-initializes the slot. */
TEST_F(OfiDynArrTest, at_allocates_zeroed_slot)
{
	int *slot = at(7);
	ASSERT_NE(slot, nullptr);
	EXPECT_EQ(*slot, 0);
}

TEST_F(OfiDynArrTest, store_and_read_back)
{
	*at(7) = 42;
	EXPECT_EQ(*at(7), 42);
	EXPECT_EQ(*at_max(7, dyn_arr_default_max_index() + 1), 42);
}

/* Indices in the same chunk are reachable once any of them is touched. */
TEST_F(OfiDynArrTest, same_chunk_indices_share_backing)
{
	*at(1) = 11;
	int *neighbor = at_max(0, dyn_arr_default_max_index() + 1);
	ASSERT_NE(neighbor, nullptr);
	EXPECT_EQ(*neighbor, 0);
	EXPECT_EQ(*at_max(1, dyn_arr_default_max_index() + 1), 11);
}

/* Indices in different chunks are independent. */
TEST_F(OfiDynArrTest, distinct_chunks_are_independent)
{
	int other = dyn_arr_default_chunk_size();
	*at(0) = 100;
	*at(other) = 200;
	EXPECT_EQ(*at(0), 100);
	EXPECT_EQ(*at(other), 200);
}

/* The last valid index works. */
TEST_F(OfiDynArrTest, max_index_is_valid)
{
	int *slot = at(dyn_arr_default_max_index());
	ASSERT_NE(slot, nullptr);
	*slot = 5;
	EXPECT_EQ(*at(dyn_arr_default_max_index()), 5);
}

/* Anything past the capacity (or negative) is rejected, not stored. */
TEST_F(OfiDynArrTest, out_of_range_is_rejected)
{
	EXPECT_EQ(at(dyn_arr_default_max_index() + 1), nullptr);
	EXPECT_EQ(at(-1), nullptr);
	EXPECT_EQ(at_max(dyn_arr_default_max_index() + 1, dyn_arr_default_max_index() + 100),
		  nullptr);
}

/* at_max rejects indices at or beyond the caller-supplied max_size, even when
 * the index is an otherwise-valid, allocated slot. */
TEST_F(OfiDynArrTest, at_max_enforces_max_size)
{
	*at(5) = 55;
	EXPECT_EQ(*at_max(5, 6), 55);     /* index <  max_size: returned */
	EXPECT_EQ(at_max(5, 5), nullptr); /* index == max_size: rejected */
	EXPECT_EQ(at_max(6, 5), nullptr); /* index >  max_size: rejected */
}

/* The init callback pre-fills each newly grown slot. */
TEST_F(OfiDynArrTest, init_callback_prefills_slots)
{
	struct ofi_dyn_arr *sarr = dyn_arr_create_with_sentinel(sizeof(int));
	ASSERT_NE(sarr, nullptr);
	int *slot = (int *) dyn_arr_at(sarr, 3);
	ASSERT_NE(slot, nullptr);
	EXPECT_EQ(*slot, dyn_arr_sentinel());
	dyn_arr_destroy(sarr);
}

/* iter visits every allocated slot; only written values are counted. */
TEST_F(OfiDynArrTest, iter_visits_written_values)
{
	*at(2) = 777;
	*at(5) = 777;
	*at(9) = 123;
	EXPECT_EQ(dyn_arr_count_value(arr, 777), 2);
	EXPECT_EQ(dyn_arr_count_value(arr, 123), 1);
}

/* A larger max_cnt raises the ceiling above the default ~1M limit. */
TEST_F(OfiDynArrTest, custom_max_cnt_indexes_past_default_limit)
{
	int past_default = dyn_arr_default_max_index() + 1;
	struct ofi_dyn_arr *big =
		dyn_arr_create_max(sizeof(int), (size_t) past_default * 2);
	ASSERT_NE(big, nullptr);
	EXPECT_GT(dyn_arr_get_max_index(big), (long) dyn_arr_default_max_index());

	int *slot = (int *) dyn_arr_at(big, past_default);
	ASSERT_NE(slot, nullptr);
	*slot = 314;
	EXPECT_EQ(*(int *) dyn_arr_at(big, past_default), 314);

	dyn_arr_destroy(big);
}

/* The custom array still rejects indices past its own (larger) ceiling. */
TEST_F(OfiDynArrTest, custom_max_cnt_rejects_past_new_limit)
{
	int past_default = dyn_arr_default_max_index() + 1;
	struct ofi_dyn_arr *big =
		dyn_arr_create_max(sizeof(int), (size_t) past_default * 2);
	ASSERT_NE(big, nullptr);

	long new_max = dyn_arr_get_max_index(big);
	EXPECT_EQ(dyn_arr_at(big, (int) (new_max + 1)), nullptr);

	dyn_arr_destroy(big);
}

/* iter walks every allocated chunk, skipping unallocated ones. */
TEST_F(OfiDynArrTest, iter_visits_multiple_chunks)
{
	int cs = dyn_arr_default_chunk_size();

	*at(0) = 555;          /* chunk 0 */
	*at(2 * cs) = 555;     /* chunk 2 (chunk 1 left unallocated) */
	*at(2 * cs + 1) = 555; /* chunk 2 */
	EXPECT_EQ(dyn_arr_count_value(arr, 555), 3);
}

/* A non-zero callback return stops iteration early. */
TEST_F(OfiDynArrTest, iter_stops_on_nonzero_callback)
{
	*at(2) = 888;
	*at(6) = 888;
	/* Two matches exist, but iteration must stop after the first. */
	EXPECT_EQ(dyn_arr_iter_first_hit(arr, 888), 1);
}

/* A custom chunk_cnt is honored and cross-chunk addressing still works. */
TEST_F(OfiDynArrTest, custom_chunk_cnt_addressing)
{
	struct ofi_dyn_arr *a = dyn_arr_create_chunk_cnt(sizeof(int), 4);
	ASSERT_NE(a, nullptr);
	EXPECT_EQ(dyn_arr_chunk_size(a), 4);

	*(int *) dyn_arr_at(a, 0) = 10; /* chunk 0 */
	*(int *) dyn_arr_at(a, 4) = 20; /* chunk 1 */
	*(int *) dyn_arr_at(a, 5) = 30; /* chunk 1, offset 1 */
	EXPECT_EQ(*(int *) dyn_arr_at(a, 0), 10);
	EXPECT_EQ(*(int *) dyn_arr_at(a, 4), 20);
	EXPECT_EQ(*(int *) dyn_arr_at(a, 5), 30);
	dyn_arr_destroy(a);
}

/* A non-power-of-two chunk_cnt is rounded up to a power of two. */
TEST_F(OfiDynArrTest, chunk_cnt_rounds_up_to_pow2)
{
	struct ofi_dyn_arr *a = dyn_arr_create_chunk_cnt(sizeof(int), 1000);
	ASSERT_NE(a, nullptr);
	EXPECT_EQ(dyn_arr_chunk_size(a), 1024);
	dyn_arr_destroy(a);
}

/* A max_cnt beyond the int-index ceiling is rejected, not clamped. */
TEST_F(OfiDynArrTest, huge_max_cnt_is_rejected)
{
	EXPECT_EQ(dyn_arr_create_max(sizeof(int), SIZE_MAX), nullptr);
}
