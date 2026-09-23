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

class EfaRdmPkeHmemTest : public Test
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
		ASSERT_EQ(efa_test_pke_open_hmem_ep(&resource.ep,
						    resource.domain,
						    resource.info),
			  0);
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

class EfaRdmPkePoolTest : public EfaRdmPkeHmemTest,
			  public WithParamInterface<enum efa_test_pke_pool>
{
};

TEST_P(EfaRdmPkePoolTest, metadata_and_bounce_split)
{
	enum efa_test_pke_pool pool = GetParam();
	struct efa_test_pke_pool_facts facts;

	efa_test_pke_pool_check(resource.ep, pool, &facts);

	ASSERT_TRUE(facts.both_pools_exist);
	EXPECT_EQ(facts.metadata_pool_size, efa_test_pke_metadata_struct_size());
	EXPECT_GT(facts.bounce_pool_size, facts.metadata_pool_size);
	EXPECT_EQ(facts.bounce_pool_alignment,
		  efa_test_pke_pool_expected_alignment(resource.ep, pool));
	EXPECT_TRUE(facts.metadata_from_metadata_pool);
	EXPECT_TRUE(facts.wiredata_from_bounce_pool);
	EXPECT_TRUE(facts.wiredata_separate_from_metadata);
	EXPECT_EQ(facts.pkt_size, facts.bounce_pool_size);
}

TEST_P(EfaRdmPkePoolTest, bounce_buffer_registration_matches_need_mr)
{
	enum efa_test_pke_pool pool = GetParam();
	struct efa_test_pke_pool_facts facts;

	efa_test_pke_pool_check(resource.ep, pool, &facts);

	ASSERT_TRUE(facts.both_pools_exist);
	EXPECT_EQ(facts.mr_present, efa_test_pke_pool_needs_mr(pool));
}

static std::string efa_test_pke_pool_name(
	const testing::TestParamInfo<enum efa_test_pke_pool> &info)
{
	switch (info.param) {
	case EFA_TEST_PKE_POOL_TX: return "tx";
	case EFA_TEST_PKE_POOL_RX: return "rx";
	case EFA_TEST_PKE_POOL_UNEXP: return "unexp";
	case EFA_TEST_PKE_POOL_OOO: return "ooo";
	case EFA_TEST_PKE_POOL_READCOPY: return "readcopy";
	default: return "unknown";
	}
}

INSTANTIATE_TEST_SUITE_P(AllPools, EfaRdmPkePoolTest,
			 testing::Values(EFA_TEST_PKE_POOL_TX,
					 EFA_TEST_PKE_POOL_RX,
					 EFA_TEST_PKE_POOL_UNEXP,
					 EFA_TEST_PKE_POOL_OOO,
					 EFA_TEST_PKE_POOL_READCOPY),
			 efa_test_pke_pool_name);

TEST_F(EfaRdmPkeTest, release_frees_metadata_and_bounce)
{
	int metadata_reused = 0, wiredata_reused = 0;

	efa_test_pke_release_frees_both(resource.ep, &metadata_reused,
					&wiredata_reused);

	EXPECT_TRUE(metadata_reused);
	EXPECT_TRUE(wiredata_reused);
}

TEST_F(EfaRdmPkeHmemTest, grow_rx_pools_grows_metadata_and_bounce_pools)
{
	struct efa_test_rx_pool_growth g;

	efa_test_grow_rx_pools(resource.ep, &g);

	ASSERT_EQ(g.err, 0);

	EXPECT_GT(g.efa_rx_meta_after, g.efa_rx_meta_before);
	EXPECT_GT(g.efa_rx_bounce_after, g.efa_rx_bounce_before);

	if (g.unexp_meta_before >= 0) {
		EXPECT_GT(g.unexp_meta_after, g.unexp_meta_before);
		EXPECT_GT(g.unexp_bounce_after, g.unexp_bounce_before);
	}
	if (g.ooo_meta_before >= 0) {
		EXPECT_GT(g.ooo_meta_after, g.ooo_meta_before);
		EXPECT_GT(g.ooo_bounce_after, g.ooo_bounce_before);
	}
	if (g.readcopy_meta_before >= 0) {
		EXPECT_GT(g.readcopy_meta_after, g.readcopy_meta_before);
		EXPECT_GT(g.readcopy_bounce_after, g.readcopy_bounce_before);
	}
}
