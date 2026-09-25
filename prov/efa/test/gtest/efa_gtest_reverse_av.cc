/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_av_array_utils.h"
#include "efa_gtest_reverse_av_utils.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <set>
#include <vector>
#include <rdma/fabric.h>

using testing::Test;

TEST(EfaReverseAvKeyTest, zero_pair_is_index_zero)
{
	EXPECT_EQ(efa_test_reverse_av_key(0, 0), 0u);
}

TEST(EfaReverseAvKeyTest, halves_interleave_into_alternating_bits)
{
	EXPECT_EQ(efa_test_reverse_av_key(0, 1), 0x00000001u);
	EXPECT_EQ(efa_test_reverse_av_key(1, 0), 0x00000002u);
	EXPECT_EQ(efa_test_reverse_av_key(1, 1), 0x00000003u);
	EXPECT_EQ(efa_test_reverse_av_key(0xffff, 0), 0xaaaaaaaau);
	EXPECT_EQ(efa_test_reverse_av_key(0, 0xffff), 0x55555555u);
	EXPECT_EQ(efa_test_reverse_av_key(0xffff, 0xffff), 0xffffffffu);
}

TEST(EfaReverseAvKeyTest, distinct_pairs_get_distinct_keys)
{
	std::set<uint64_t> keys;

	for (uint16_t ahn = 0; ahn < 64; ahn++) {
		for (uint16_t qpn = 0; qpn < 64; qpn++)
			EXPECT_TRUE(keys.insert(efa_test_reverse_av_key(ahn, qpn))
					    .second)
				<< "collision at ahn " << ahn << " qpn " << qpn;
	}
	EXPECT_EQ(keys.size(), 64u * 64u);
}

TEST(EfaReverseAvKeyTest, k_bit_halves_produce_a_2k_bit_key)
{
	for (uint16_t ahn = 0; ahn < 64; ahn++) {
		for (uint16_t qpn = 0; qpn < 64; qpn++)
			EXPECT_LT(efa_test_reverse_av_key(ahn, qpn), 64u * 64u);
	}
}

class EfaReverseAvTest : public Test
{
	protected:
	struct efa_av_array *arr = nullptr;
	std::vector<void *> entries;

	void SetUp() override
	{
		arr = efa_test_reverse_av_create();
		ASSERT_NE(arr, nullptr);
	}

	void TearDown() override
	{
		for (void *entry : entries)
			efa_test_reverse_av_entry_free(entry);
		if (arr)
			efa_test_reverse_av_destroy(arr);
	}

	void *new_entry(uint16_t ahn, uint16_t qpn, uint64_t fi_addr)
	{
		void *entry = efa_test_reverse_av_entry_alloc(ahn, qpn, fi_addr);

		EXPECT_NE(entry, nullptr);
		if (entry)
			entries.push_back(entry);
		return entry;
	}
};

TEST_F(EfaReverseAvTest, fresh_reverse_av_is_empty)
{
	EXPECT_EQ(efa_test_av_array_count(arr), 0);
	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 7),
		  (uint64_t) FI_ADDR_NOTAVAIL);
}

TEST_F(EfaReverseAvTest, lookup_finds_added_entry)
{
	ASSERT_EQ(efa_test_reverse_av_add(arr, new_entry(3, 7, 42)), 0);

	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 7), 42u);
	EXPECT_EQ(efa_test_av_array_count(arr), 1);
	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 7, 3),
		  (uint64_t) FI_ADDR_NOTAVAIL);
	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 8),
		  (uint64_t) FI_ADDR_NOTAVAIL);
}

TEST_F(EfaReverseAvTest, distinct_pairs_do_not_collide)
{
	uint64_t fi_addr = 0;

	for (uint16_t ahn = 0; ahn < 16; ahn++) {
		for (uint16_t qpn = 0; qpn < 16; qpn++)
			ASSERT_EQ(efa_test_reverse_av_add(
					  arr, new_entry(ahn, qpn, fi_addr++)),
				  0);
	}

	EXPECT_EQ(efa_test_av_array_count(arr), 16 * 16);
	fi_addr = 0;
	for (uint16_t ahn = 0; ahn < 16; ahn++) {
		for (uint16_t qpn = 0; qpn < 16; qpn++)
			EXPECT_EQ(efa_test_reverse_av_lookup(arr, ahn, qpn),
				  fi_addr++);
	}
}

TEST_F(EfaReverseAvTest, add_replaces_the_current_entry_for_a_reused_qpn)
{
	ASSERT_EQ(efa_test_reverse_av_add(arr, new_entry(3, 7, 42)), 0);
	ASSERT_EQ(efa_test_reverse_av_add(arr, new_entry(3, 7, 43)), 0);

	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 7), 43u);
	EXPECT_EQ(efa_test_av_array_count(arr), 1);
}

TEST_F(EfaReverseAvTest, remove_clears_the_current_entry)
{
	void *entry = new_entry(3, 7, 42);

	ASSERT_EQ(efa_test_reverse_av_add(arr, entry), 0);

	EXPECT_EQ(efa_test_reverse_av_remove(arr, entry), 1);
	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 7),
		  (uint64_t) FI_ADDR_NOTAVAIL);
	EXPECT_EQ(efa_test_av_array_count(arr), 0);
}

TEST_F(EfaReverseAvTest, remove_leaves_a_slot_owned_by_a_newer_entry)
{
	void *displaced = new_entry(3, 7, 42);

	ASSERT_EQ(efa_test_reverse_av_add(arr, displaced), 0);
	ASSERT_EQ(efa_test_reverse_av_add(arr, new_entry(3, 7, 43)), 0);

	EXPECT_EQ(efa_test_reverse_av_remove(arr, displaced), 0);
	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 3, 7), 43u);
	EXPECT_EQ(efa_test_av_array_count(arr), 1);
}

TEST_F(EfaReverseAvTest, remove_of_an_absent_entry_is_a_noop)
{
	void *absent = new_entry(3, 7, 42);

	EXPECT_EQ(efa_test_reverse_av_remove(arr, absent), 0);
	EXPECT_EQ(efa_test_av_array_count(arr), 0);
}

TEST_F(EfaReverseAvTest, small_pairs_stay_in_the_inline_region)
{
	for (uint16_t ahn = 0; ahn < 32; ahn++) {
		for (uint16_t qpn = 0; qpn < 32; qpn++)
			ASSERT_EQ(efa_test_reverse_av_add(
					  arr, new_entry(ahn, qpn, 1)),
				  0);
	}

	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 0);
}

TEST_F(EfaReverseAvTest, largest_pair_is_addressable)
{
	ASSERT_EQ(efa_test_reverse_av_add(arr, new_entry(0xffff, 0xffff, 42)), 0);

	EXPECT_EQ(efa_test_reverse_av_lookup(arr, 0xffff, 0xffff), 42u);
	EXPECT_EQ(efa_test_av_array_has_chunk_table(arr), 1);
}
