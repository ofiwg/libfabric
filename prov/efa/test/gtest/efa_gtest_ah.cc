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
 * @brief ibv_create_ah fails with ENOMEM, an existing implicit AH is evicted,
 * and the retried ibv_create_ah succeeds (the ENOMEM eviction branch in
 * efa_ah_alloc, via efa_ah_implicit_av_evict_ah).
 *
 * Two distinct GIDs are inserted into the implicit AV. The first insert
 * succeeds and populates the AH map / LRU list. The second insert's initial
 * ibv_create_ah returns ENOMEM, which triggers efa_ah_implicit_av_evict_ah to
 * release the first (implicit-only) AH; the subsequent retry then succeeds, so
 * the second insert returns a valid fi_addr.
 */
TEST_F(EfaAhTest, alloc_enomem_evict_and_retry_succeeds)
{
	fi_addr_t addr_a, addr_b;
	static struct ibv_ah dummy_ah_a, dummy_ah_b;

	/*
	 * Three ibv_create_ah calls in order:
	 *   1. first GID  -> success (populates AH map + LRU)
	 *   2. second GID -> ENOMEM (drives the errno == FI_ENOMEM branch)
	 *   3. second GID retry after eviction -> success
	 */
	EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Return(&dummy_ah_a))
		.WillOnce(Invoke([](struct ibv_pd *,
				    struct ibv_ah_attr *) -> struct ibv_ah * {
			errno = ENOMEM;
			return nullptr;
		}))
		.WillOnce(Return(&dummy_ah_b));

	/* efadv_query_ah runs once per successful ibv_create_ah (GIDs A and B).
	 */
	EXPECT_CALL(mock_efa, efadv_query_ah)
		.Times(2)
		.WillRepeatedly(Return(0));

	/*
	 * ibv_destroy_ah fires three times: when AH A is evicted during the
	 * second insert, for AH B during teardown, and for the endpoint's
	 * self_ah during teardown (created by fi_enable before the mock was
	 * installed, but destroyed while the mock is still active).
	 */
	EXPECT_CALL(mock_efa, ibv_destroy_ah)
		.Times(3)
		.WillRepeatedly(Return(0));

	/*
	 * Route reverse-AV bookkeeping to the real implementation so the
	 * cur/prv reverse-AV hashes stay consistent across the eviction (which
	 * removes A's entry) and the later teardown removal.
	 */
	EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillRepeatedly(Invoke(__real_efa_av_reverse_av_add));

	/*
	 * Insert through efa_av_insert_one, the only production entry to
	 * efa_ah_alloc. The eviction path releases a fully-wired implicit
	 * efa_conn (conn->av, implicit_fi_addr, implicit_conn_list linkage)
	 * that efa_conn_alloc builds under the AV locks insert_one holds; a
	 * bare efa_ah_alloc would leave an empty implicit_conn_list with no
	 * AH to evict.
	 */
	addr_a = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_NE(addr_a, (fi_addr_t) FI_ADDR_NOTAVAIL);

	/*
	 * Inserting B is what drives the ENOMEM eviction: its first
	 * ibv_create_ah fails, A (the only entry with no explicit refs) is
	 * evicted, and the retry succeeds.
	 */
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
 * @brief ibv_create_ah fails with ENOMEM but no AH can be evicted, so the
 * insert fails (the ENOMEM eviction branch in efa_ah_alloc, where
 * efa_ah_implicit_av_evict_ah returns -FI_ENOMEM).
 *
 * A single implicit AV insert is attempted with an empty AH map / LRU list.
 * ibv_create_ah returns ENOMEM, the eviction helper finds no releasable AH and
 * returns -FI_ENOMEM, so efa_ah_alloc bails out via err_free_efa_ah without
 * ever reaching efadv_query_ah or creating a reverse-AV entry. The insert
 * therefore fails with FI_ADDR_NOTAVAIL.
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

	/* The failed insert creates no AH, so the only ibv_destroy_ah is for
	 * the endpoint's self_ah (created by fi_enable before the mock was
	 * installed) during teardown while the mock is still active. */
	EXPECT_CALL(mock_efa, ibv_destroy_ah).Times(1).WillOnce(Return(0));

	addr = efa_test_insert_peer_new_gid(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}
