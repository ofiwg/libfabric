/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::Test;
using testing::_;
using testing::Return;
using testing::Invoke;
using testing::StrictMock;

class EfaConnTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
		MockEfa::set(nullptr);
	}
};

/**
 * @brief Test that efa_conn_alloc cleans up RDM resources when
 * efa_av_reverse_av_add fails (efa_conn.c:304-316).
 * Verifies the cleanup path calls efa_conn_rdm_deinit without crash.
 */
TEST_F(EfaConnTest, alloc_reverse_av_add_failure_rdm_cleanup)
{
	fi_addr_t addr;
	static struct ibv_ah dummy_ibv_ah;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	MockEfa::set(&mock_efa);
	EXPECT_CALL(mock_efa, ibv_create_ah(_, _))
		.WillOnce(Return(&dummy_ibv_ah));
	EXPECT_CALL(mock_efa, ibv_destroy_ah(_)).WillRepeatedly(Return(0));
	EXPECT_CALL(mock_efa, efadv_query_ah(_, _, _))
		.WillRepeatedly(Return(0));
	EXPECT_CALL(mock_efa, efa_av_reverse_av_add(_, _, _, _))
		.WillOnce(Return(-FI_ENOMEM));

	addr = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

/**
 * @brief Test that efa_conn_alloc cleans up RDM resources when
 * efa_av_reverse_av_add fails via the explicit fi_av_insert path
 * (insert_implicit_av=false). The explicit path acquires the SRX lock
 * itself before calling efa_av_insert_one.
 */
TEST_F(EfaConnTest, alloc_reverse_av_add_failure_explicit_insert)
{
	fi_addr_t addr;
	int num_addr;
	static struct ibv_ah dummy_ibv_ah;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	MockEfa::set(&mock_efa);
	EXPECT_CALL(mock_efa, ibv_create_ah(_, _))
		.WillOnce(Return(&dummy_ibv_ah));
	EXPECT_CALL(mock_efa, ibv_destroy_ah(_)).WillRepeatedly(Return(0));
	EXPECT_CALL(mock_efa, efadv_query_ah(_, _, _))
		.WillRepeatedly(Return(0));
	EXPECT_CALL(mock_efa, efa_av_reverse_av_add(_, _, _, _))
		.WillOnce(Return(-FI_ENOMEM));

	num_addr = efa_test_explicit_av_insert(resource.ep, resource.av, &addr);
	EXPECT_EQ(num_addr, 0);
}
