/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_av_array_utils.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <rdma/fi_errno.h>

using testing::Test;

/* Distinct non-NULL pointer values used only as opaque entries (never read). */
static void *entry_ptr(int k)
{
	return reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + k * 0x40));
}

class EfaAvArrayTest : public Test
{
	protected:
	struct efa_av_array *arr = nullptr;

	void SetUp() override
	{
		arr = efa_test_av_array_create();
		ASSERT_NE(arr, nullptr);
	}

	void TearDown() override
	{
		if (arr)
			efa_test_av_array_destroy(arr);
	}

	int cutoff() { return efa_test_av_array_fast_cutoff; }
	int chunk() { return efa_test_av_array_chunk_size; }
};

/* A fresh array is empty and has not allocated its chunk table. */
TEST_F(EfaAvArrayTest, fresh_array_is_empty_and_has_no_chunk_table)
{
	EXPECT_EQ(efa_test_av_array_at(arr, 0), nullptr);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
}

/* Insert and read back an entry in the fast region. */
TEST_F(EfaAvArrayTest, insert_and_read_fast)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(5)), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, 5), entry_ptr(5));
	EXPECT_EQ(efa_test_av_array_at(arr, 6), nullptr);
}

/* An insert within the fast region does not allocate the chunk table. */
TEST_F(EfaAvArrayTest, fast_insert_allocates_no_chunk_table)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() - 1, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() - 1), entry_ptr(1));
}

/* The first index at the cutoff allocates the chunk table and reads back. */
TEST_F(EfaAvArrayTest, chunk_insert_allocates_chunk_table)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff(), entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 1);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff()), entry_ptr(2));
}

/* Fast-region and chunk-region entries do not affect one another. */
TEST_F(EfaAvArrayTest, fast_and_chunk_regions_isolated)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff()), nullptr);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff(), entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, 5), entry_ptr(1));
}

/* cutoff-1 stays in the fast region; cutoff is the first chunk-region index. */
TEST_F(EfaAvArrayTest, fast_chunk_boundary)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() - 1, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff(), entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 1);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() - 1), entry_ptr(1));
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff()), entry_ptr(2));
}

/* A chunk-region lookup before the chunk table exists returns NULL and does not allocate. */
TEST_F(EfaAvArrayTest, chunk_lookup_before_chunk_table_is_null)
{
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() + 100), nullptr);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
}

/* With the chunk table allocated, an index in an unallocated chunk still returns NULL. */
TEST_F(EfaAvArrayTest, unallocated_chunk_returns_null)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff(), entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 1);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() + 3 * chunk()), nullptr);
	/* The lookup did not create an entry. */
	EXPECT_EQ(efa_test_av_array_count(arr), 1);
}

/* Entries in different chunks are independent; an untouched chunk is NULL. */
TEST_F(EfaAvArrayTest, distinct_chunks_independent)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff(), entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() + chunk(), entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff()), entry_ptr(1));
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() + chunk()), entry_ptr(2));
	EXPECT_EQ(efa_test_av_array_at(arr, cutoff() + 2 * chunk()), nullptr);
}

/* A second insert at an index overwrites the stored pointer. */
TEST_F(EfaAvArrayTest, overwrite_slot)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, 5), entry_ptr(2));
}

/* Inserting NULL clears a slot. */
TEST_F(EfaAvArrayTest, insert_null_clears_slot)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, nullptr), 0);
	EXPECT_EQ(efa_test_av_array_at(arr, 5), nullptr);
}

/* Out-of-range indices are rejected by both at and insert. */
TEST_F(EfaAvArrayTest, out_of_range_is_rejected)
{
	struct efa_av_array *a = efa_test_av_array_create_max(100);
	ASSERT_NE(a, nullptr);

	EXPECT_EQ(efa_test_av_array_at(a, 101), nullptr);
	EXPECT_EQ(efa_test_av_array_insert(a, 101, entry_ptr(1)), -FI_EINVAL);

	/* The largest valid index is reachable. */
	EXPECT_EQ(efa_test_av_array_insert(a, 100, entry_ptr(9)), 0);
	EXPECT_EQ(efa_test_av_array_at(a, 100), entry_ptr(9));

	efa_test_av_array_destroy(a);
}

/* A max_idx below the fast cutoff never allocates a chunk table. */
TEST_F(EfaAvArrayTest, small_max_idx_allocates_no_chunk_table)
{
	struct efa_av_array *a = efa_test_av_array_create_max(100);
	ASSERT_NE(a, nullptr);

	EXPECT_EQ(efa_test_av_array_insert(a, 50, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_at(a, 50), entry_ptr(1));
	EXPECT_EQ(efa_test_av_array_has_chunk_table(a), 0);

	efa_test_av_array_destroy(a);
}

/* A max_idx above the cap is rejected; the cap itself is accepted. */
TEST_F(EfaAvArrayTest, bad_max_idx_is_rejected)
{
	EXPECT_EQ(efa_test_av_array_create_max(efa_test_av_array_max_idx_ceiling + 1), nullptr);

	struct efa_av_array *a = efa_test_av_array_create_max(efa_test_av_array_max_idx_ceiling);
	ASSERT_NE(a, nullptr);
	efa_test_av_array_destroy(a);
}

/* max_idx of zero selects the default: the default max index is reachable and
 * one past it is rejected. */
TEST_F(EfaAvArrayTest, zero_max_idx_uses_default)
{
	struct efa_av_array *a = efa_test_av_array_create_max(0);
	ASSERT_NE(a, nullptr);

	int m = efa_test_av_array_default_max_idx;
	EXPECT_EQ(efa_test_av_array_insert(a, m, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_at(a, m), entry_ptr(1));
	EXPECT_EQ(efa_test_av_array_at(a, m + 1), nullptr);
	EXPECT_EQ(efa_test_av_array_insert(a, m + 1, entry_ptr(1)), -FI_EINVAL);

	efa_test_av_array_destroy(a);
}

/* The largest index at the cap is reachable and reads back. */
TEST_F(EfaAvArrayTest, max_idx_at_cap_reachable)
{
	struct efa_av_array *a = efa_test_av_array_create_max(efa_test_av_array_max_idx_ceiling);
	ASSERT_NE(a, nullptr);

	int m = efa_test_av_array_max_idx_ceiling;
	EXPECT_EQ(efa_test_av_array_insert(a, m, entry_ptr(7)), 0);
	EXPECT_EQ(efa_test_av_array_at(a, m), entry_ptr(7));
	EXPECT_EQ(efa_test_av_array_has_chunk_table(a), 1);

	efa_test_av_array_destroy(a);
}

/* iter visits every non-NULL entry across the fast and chunk regions. */
TEST_F(EfaAvArrayTest, iter_counts_non_null_entries)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 1, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() + 5, entry_ptr(2)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() + chunk() + 9, entry_ptr(3)), 0);
	EXPECT_EQ(efa_test_av_array_count(arr), 3);
}

/* iter skips slots that were cleared back to NULL. */
TEST_F(EfaAvArrayTest, iter_skips_cleared_slots)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, entry_ptr(1)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, 5, nullptr), 0);
	EXPECT_EQ(efa_test_av_array_count(arr), 0);
}

/* iter stops early when the callback returns non-zero, across regions. */
TEST_F(EfaAvArrayTest, iter_stops_early)
{
	EXPECT_EQ(efa_test_av_array_insert(arr, 1, entry_ptr(8)), 0);
	EXPECT_EQ(efa_test_av_array_insert(arr, cutoff() + 5, entry_ptr(8)), 0);
	EXPECT_EQ(efa_test_av_array_iter_first_hit(arr, entry_ptr(8)), 1);
}

/* Destroying a NULL array is a no-op. */
TEST_F(EfaAvArrayTest, destroy_null_is_safe)
{
	efa_test_av_array_destroy(nullptr);
	SUCCEED();
}
