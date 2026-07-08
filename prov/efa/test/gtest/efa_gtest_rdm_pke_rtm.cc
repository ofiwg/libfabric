/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <gtest/gtest.h>

using testing::Bool;
using testing::StrictMock;
using testing::TestWithParam;

class EfaRtmTest : public TestWithParam<bool>
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

		peer_addr =
			efa_test_insert_self_gid_peer(resource.ep, resource.av);
		ASSERT_NE(peer_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Asserts that a READ_NACK RTM whose msg_id misses the peer's rxe_map is
 * handled (not a NULL deref) in both efa_rdm_pke_proc_{msg/tag}rtm
 */
TEST_P(EfaRtmTest, read_nack_missing_rxe_no_null_deref)
{
	ssize_t ret = 0;

	ASSERT_EQ(efa_test_rtm_read_nack_missing_rxe(resource.ep, peer_addr,
						     GetParam(), &ret),
		  1);
	/* Reached only if the handler did not crash; the missing entry is an
	 * error, not silent success. */
	EXPECT_NE(ret, 0);
}

INSTANTIATE_TEST_SUITE_P(, EfaRtmTest, Bool(),
			 [](const testing::TestParamInfo<bool> &info) {
				 return info.param ? "tagrtm" : "msgrtm";
			 });
