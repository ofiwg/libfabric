/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <gtest/gtest.h>

using testing::Test;
using testing::WithParamInterface;
using testing::Range;

class EfaRdmPkeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/* pke_release_cloned must release arbitrary length packet list correctly */
class EfaRdmPkeChainTest : public EfaRdmPkeTest,
			   public WithParamInterface<size_t>
{
};

TEST_P(EfaRdmPkeChainTest, release_cloned_frees_whole_chain)
{
	size_t n = GetParam();
	struct efa_rdm_pke *head;

	head = efa_test_pke_build_unexp_chain(resource.ep, n);
	ASSERT_EQ(head == nullptr, n == 0);
	ASSERT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), n);

	efa_test_pke_release_cloned(head);

	EXPECT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), 0u);
}

INSTANTIATE_TEST_SUITE_P(ChainLengths, EfaRdmPkeChainTest, Range<size_t>(0, 6));
