/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_srx_utils.h"
#include <gtest/gtest.h>
#include <string>

struct EfaRdmSrxCase {
	const char *name;
	int unexpected;
};

class EfaRdmSrxTest : public testing::TestWithParam<EfaRdmSrxCase>
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmSrxTest, dispatches_receive_callback)
{
	const EfaRdmSrxCase &c = GetParam();
	struct efa_test_srx_dispatch_result res = {};

	ASSERT_EQ(efa_test_srx_dispatches_receive_callback(
			  resource.ep, resource.av, c.unexpected, &res),
		  0);
	if (c.unexpected) {
		EXPECT_TRUE(res.was_unexpected);
		EXPECT_TRUE(res.callback_set);
	}
	EXPECT_EQ(res.callback_invocations, 1);
	EXPECT_TRUE(res.matched);
	EXPECT_TRUE(res.unexpected_packet_cleared);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmSrxTest,
	testing::Values(EfaRdmSrxCase{"expected", 0},
			EfaRdmSrxCase{"unexpected", 1}),
	[](const testing::TestParamInfo<EfaRdmSrxCase> &info) {
		return std::string(info.param.name);
	});
