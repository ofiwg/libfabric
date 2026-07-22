/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <cerrno>
#include <gtest/gtest.h>

using testing::Test;
using testing::Return;
using testing::Invoke;
using testing::StrictMock;

class EfaAhTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
		MockEfa::set(nullptr);
	}
};

/**
 * @brief Exerise efa_ah_alloc's ENOMEM eviction branch where the retry succeeds
 */
TEST_F(EfaAhTest, alloc_enomem_evict_and_retry_succeeds)
{
	fi_addr_t addr_a, addr_b;
	static struct ibv_ah dummy_ah_a, dummy_ah_b;

	/*
	 * Three ibv_create_ah calls in order:
	 *   1. first GID  -> success (populates AH map + LRU)
	 *   2. second GID -> ENOMEM (drives the errno == FI_ENOMEM branch)
	 *   3. second GID retry after eviction of dummy_ah_a -> success
	 */
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Return(&dummy_ah_a))
		.WillOnce(Invoke([](struct ibv_pd *,
				    struct ibv_ah_attr *) -> struct ibv_ah * {
			errno = ENOMEM;
			return nullptr;
		}))
		.WillOnce(Return(&dummy_ah_b));

	EFA_EXPECT_CALL(mock_efa, efadv_query_ah)
		.Times(2)
		.WillRepeatedly(Return(0));

	/*
	 * ibv_destroy_ah is called for three AH's: we need to fake
	 * dummy_ah_a/b's destroy calls, but actually call destroy on self_ah.
	 */
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah)
		.Times(3)
		.WillRepeatedly(Invoke([](struct ibv_ah *ah) -> int {
			if (ah == &dummy_ah_a || ah == &dummy_ah_b)
				return 0;
			return __real_ibv_destroy_ah(ah);
		}));


	addr_a = efa_test_av_insert_new_ah(resource.ep, resource.av);
	EXPECT_NE(addr_a, (fi_addr_t) FI_ADDR_NOTAVAIL);

	addr_b = efa_test_av_insert_new_ah(resource.ep, resource.av);
	EXPECT_NE(addr_b, (fi_addr_t) FI_ADDR_NOTAVAIL);

	/* This proves A specifically was the evicted entry */
	EXPECT_EQ(efa_test_implicit_addr_to_ibv_ah(resource.av, addr_a),
		  nullptr);
	/*
	 * The conn for addr_b must be backed by the AH from the retried
	 * ibv_create_ah (dummy_ah_b), proving the insert kept the AH created
	 * after eviction rather than the one that failed with ENOMEM.
	 */
	EXPECT_EQ(efa_test_implicit_addr_to_ibv_ah(resource.av, addr_b),
		  &dummy_ah_b);
}

/**
 * @brief Exercise efa_ah_alloc's ENOMEM eviction branch where there is nothing
 * to evict, so the retry never happens and the insert fails
 */
TEST_F(EfaAhTest, alloc_enomem_no_evictable_ah_fails)
{
	fi_addr_t addr;

	/* one failed AH creation: empty LRU = no retry */
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Invoke([](struct ibv_pd *,
				    struct ibv_ah_attr *) -> struct ibv_ah * {
			errno = ENOMEM;
			return nullptr;
		}));

	addr = efa_test_av_insert_new_ah(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}
