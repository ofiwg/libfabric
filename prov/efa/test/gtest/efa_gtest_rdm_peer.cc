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
};

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
