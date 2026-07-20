/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::Test;
using testing::_;
using testing::Return;
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
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief efa_conn_alloc unwinds the conn via efa_conn_rdm_deinit when
 * efa_av_reverse_av_add fails, so the insert fails cleanly.
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
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Return(&dummy_ibv_ah));
	/* The unwind releases only the peer's dummy AH; self_ah is destroyed
	 * in teardown after the mock is uninstalled (real destroy). */
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ibv_ah)
		.WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efadv_query_ah)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
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
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah, _, _)
		.WillOnce(Return(&dummy_ibv_ah));
	/* The unwind releases only the peer's dummy AH; self_ah is destroyed
	 * in teardown after the mock is uninstalled (real destroy). */
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ibv_ah)
		.WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efadv_query_ah)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillOnce(Return(-FI_ENOMEM));

	num_addr = efa_test_explicit_av_insert(resource.ep, resource.av, &addr);
	EXPECT_EQ(num_addr, 0);
}
