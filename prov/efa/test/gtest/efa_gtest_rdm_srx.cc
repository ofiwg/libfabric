/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_srx_utils.h"
#include <gtest/gtest.h>
#include <string>

struct EfaRdmSrxDirectedCase {
	const char *name;
	int tagged;
};

class EfaRdmSrxDirectedTest
	: public testing::TestWithParam<EfaRdmSrxDirectedCase>
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_MSG | FI_TAGGED | FI_DIRECTED_RECV;

		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmSrxDirectedTest, matches_earlier_unspec_receive_first)
{
	const EfaRdmSrxDirectedCase &c = GetParam();
	struct efa_test_srx_directed_match_result res = {};

	ASSERT_EQ(efa_test_srx_directed_match_honors_post_order(
			  resource.ep, resource.av, c.tagged, &res),
		  0);
	ASSERT_TRUE(res.dir_recv);
	EXPECT_EQ(res.matched_context[0], res.posted_context[0]);
	EXPECT_EQ(res.matched_context[1], res.posted_context[1]);
	EXPECT_TRUE(res.queues_empty);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmSrxDirectedTest,
	testing::Values(EfaRdmSrxDirectedCase{"msg", 0},
			EfaRdmSrxDirectedCase{"tagged", 1}),
	[](const testing::TestParamInfo<EfaRdmSrxDirectedCase> &info) {
		return std::string(info.param.name);
	});
