/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <gtest/gtest.h>

using testing::Return;
using testing::StrictMock;
using testing::Invoke;

class EfaRdmPeerTest : public testing::Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_UNSPEC;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}

	void expect_order(enum efa_test_order_scenario scenario,
			  uint32_t first_msg_id, uint32_t exp_msg_id_before,
			  size_t count)
	{
		struct efa_test_order_result res = {};

		ASSERT_EQ(efa_test_ordered_receive(resource.ep, peer_addr,
						   scenario, &res),
			  0);
		EXPECT_EQ(res.pre_log_len, 0u);
		EXPECT_EQ(res.exp_msg_id_before, exp_msg_id_before);
		ASSERT_EQ(res.log_len, count);
		for (size_t i = 0; i < count; i++) {
			EXPECT_EQ(res.log[i], first_msg_id + i);
			EXPECT_EQ(res.callback_count[i], 1u);
		}
		EXPECT_EQ(res.exp_msg_id, first_msg_id + count);
		EXPECT_TRUE(res.drained);
	}
};

TEST_F(EfaRdmPeerTest, reordered_packet_retains_callback)
{
	struct efa_test_reorder_callback_result res = {};

	ASSERT_EQ(efa_test_reordered_packet_retains_callback(
			  resource.ep, peer_addr, &res),
		  0);
	EXPECT_TRUE(res.callback_set);
	EXPECT_TRUE(res.protocol_correct);
}

TEST_F(EfaRdmPeerTest, reordered_rta_retains_callback)
{
	struct efa_test_reorder_callback_result res = {};

	ASSERT_EQ(efa_test_reordered_rta_retains_callback(
			  resource.ep, peer_addr, &res),
		  0);
	EXPECT_TRUE(res.callback_set);
	EXPECT_TRUE(res.protocol_correct);
}

TEST_F(EfaRdmPeerTest, reorder_drain_invokes_callback)
{
	expect_order(EFA_TEST_ORDER_SINGLE_DRAIN, 1, 0, 1);
}

TEST_F(EfaRdmPeerTest, mixed_rtm_rta_drains_in_msg_id_order)
{
	expect_order(EFA_TEST_ORDER_MIXED_RTM_RTA, 50, 50, 3);
}

TEST_F(EfaRdmPeerTest, reorder_drain_continues_after_callback_error)
{
	expect_order(EFA_TEST_ORDER_DRAIN_ERROR, 1, 1, 2);
}

TEST_F(EfaRdmPeerTest, in_order_callback_error_still_drains_ready_successor)
{
	expect_order(EFA_TEST_ORDER_IN_ORDER_ERROR, 10, 10, 2);
}

/**
 * @brief Asserts that a efa_rdm_peer_reorder_msg failure on the ooo_clone path releases the rx_pkt
 */
TEST_F(EfaRdmPeerTest, failed_reorder_msg_releases_rx_pkt)
{
	size_t to_post_before = 0, to_post_after = 0;

	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_clone).WillOnce(Return(nullptr));

	ASSERT_EQ(efa_test_failed_reorder_msg_releases_rx_pkt(
			  resource.ep, peer_addr, &to_post_before,
			  &to_post_after),
		  0);

	EXPECT_EQ(to_post_after, to_post_before + 1);
}

/**
 * @brief Asserts that a efa_rdm_peer_reorder_msg failure on the overflow path
 * releases the rx_pkt and frees the overflow_pke_list_entry
 */
TEST_F(EfaRdmPeerTest, failed_reorder_msg_overflow_releases_rx_pkt_and_entry)
{
	size_t to_post_before = 0, to_post_after = 0;
	size_t overflow_free_before = 0, overflow_free_after = 0;

	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_clone).WillOnce(Return(nullptr));

	ASSERT_EQ(efa_test_failed_reorder_msg_overflow_releases_rx_pkt_and_entry(
			  resource.ep, peer_addr, &to_post_before,
			  &to_post_after, &overflow_free_before,
			  &overflow_free_after),
		  0);

	EXPECT_EQ(to_post_after, to_post_before + 1);
	EXPECT_EQ(overflow_free_after, overflow_free_before);
}
