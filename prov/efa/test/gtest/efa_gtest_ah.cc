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

/**
 * @brief Exercises the ENOMEM recovery branch of efa_ah_alloc.
 *
 * When ibv_create_ah fails with FI_ENOMEM during an implicit AV insertion,
 * efa_ah_alloc attempts to evict an AH entry that has no explicit AV references
 * and retries ibv_create_ah. These tests drive both outcomes of that branch:
 * the retry succeeding after a successful eviction, and the branch bailing out
 * because there is no AH available to evict.
 */
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
		/*
		 * Closing the (QP-enabled) endpoint in TearDown drains the CQ
		 * via efa_rdm_ep_wait_send ->
		 * efa_rdm_cq_poll_ibv_cq_closing_ep, which calls
		 * efa_ibv_cq_start_poll while the mock is still installed.
		 * Route it to the real stub (returns ENOENT, i.e. empty CQ) so
		 * the drain loop never enters and never reaches qp_num lookup.
		 */
		EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll)
			.WillRepeatedly(Invoke(__real_efa_ibv_cq_start_poll));

		efa_test_resource_destruct(&resource);
		// It's necessary to uninstall the mock after destruct
		// because the tests have dummy AHs that cannot be given
		// to the real ibv_destroy_ah
		MockEfa::set(nullptr);
	}
};

/**
 * @brief efa_ah_alloc's ENOMEM eviction branch where the retry succeeds: a
 * second insert hits ENOMEM, efa_ah_implicit_av_evict_ah releases the first
 * (implicit-only) AH, and the retried ibv_create_ah succeeds.
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
	EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Return(&dummy_ah_a))
		.WillOnce(Invoke([](struct ibv_pd *,
				    struct ibv_ah_attr *) -> struct ibv_ah * {
			errno = ENOMEM;
			return nullptr;
		}))
		.WillOnce(Return(&dummy_ah_b));

	EXPECT_CALL(mock_efa, efadv_query_ah)
		.Times(2)
		.WillRepeatedly(Return(0));

	/*
	 * ibv_destroy_ah is called for three AH's: we need to fake
	 * dummy_ah_a/b's destroy calls, but actually call destroy on self_ah.
	 */
	EXPECT_CALL(mock_efa, ibv_destroy_ah)
		.Times(3)
		.WillRepeatedly(Invoke([](struct ibv_ah *ah) -> int {
			if (ah == &dummy_ah_a || ah == &dummy_ah_b)
				return 0;
			return __real_ibv_destroy_ah(ah);
		}));

	// run the real reverse_av_add to populate the reverse av hash
	EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillRepeatedly(Invoke(__real_efa_av_reverse_av_add));

	addr_a = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_NE(addr_a, (fi_addr_t) FI_ADDR_NOTAVAIL);

	// we have configured this insertion to fail-retry-succeed
	addr_b = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_NE(addr_b, (fi_addr_t) FI_ADDR_NOTAVAIL);

	/*
	 * A's conn was released during B's insert, so addr_a no longer
	 * resolves. This proves A specifically was the evicted entry.
	 */
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
 * @brief efa_ah_alloc's ENOMEM eviction branch where the retry never happens:
 * with an empty AH map efa_ah_implicit_av_evict_ah returns -FI_ENOMEM, so
 * efa_ah_alloc bails via err_free_efa_ah and the insert fails.
 */
TEST_F(EfaAhTest, alloc_enomem_no_evictable_ah_fails)
{
	fi_addr_t addr;

	/* The only AH creation attempt fails with ENOMEM; with an empty LRU
	 * list the eviction helper cannot free anything, so no retry occurs. */
	EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Invoke([](struct ibv_pd *,
				    struct ibv_ah_attr *) -> struct ibv_ah * {
			errno = ENOMEM;
			return nullptr;
		}));

	/* The failed insert creates no AH, so ibv_destroy_ah is only expected
	 * to be called once on the real AH. */
	EXPECT_CALL(mock_efa, ibv_destroy_ah)
		.Times(1)
		.WillOnce(Invoke(__real_ibv_destroy_ah));

	addr = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}
