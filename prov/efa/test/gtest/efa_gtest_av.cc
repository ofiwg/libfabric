/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>
#include <vector>

using testing::TestWithParam;
using testing::Values;

class EfaAvCloseTest : public TestWithParam<const char *>
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(FI_EP_RDM,
							       GetParam())));
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaAvCloseTest, bound_endpoint_returns_busy_without_mutating_av)
{
	fi_addr_t addr;
	std::vector<unsigned char> raw_addr(resource.info->src_addrlen);
	size_t raw_addr_len;

	ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av, &addr), 1);
	ASSERT_FALSE(raw_addr.empty());
	raw_addr_len = raw_addr.size();

	EXPECT_EQ(fi_close(&resource.av->fid), -FI_EBUSY);
	EXPECT_EQ(fi_av_lookup(resource.av, addr, raw_addr.data(), &raw_addr_len),
		  0);

	ASSERT_EQ(fi_close(&resource.ep->fid), 0);
	resource.ep = nullptr;
	EXPECT_EQ(fi_close(&resource.av->fid), 0);
	resource.av = nullptr;
}

INSTANTIATE_TEST_SUITE_P(
	Fabrics, EfaAvCloseTest,
	Values(EFA_FABRIC_NAME, EFA_DIRECT_FABRIC_NAME),
	[](const testing::TestParamInfo<const char *> &info) {
		return info.index ? "efaDirect" : "efa";
	});
