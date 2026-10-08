/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_av_utils.h"
#include <gtest/gtest.h>

using testing::Test;
using testing::_;
using testing::Invoke;
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
 * @brief An implicit insert whose map reservation fails unwinds the conn, the
 * AH, the LRU entry and the util AV slot, so the insert fails cleanly.
 *
 * Every map slot is reserved before the conn is published, so the publication
 * itself (efa_av_reverse_av_add on a reserved slot) cannot fail; the
 * reservation is where an allocation failure can happen.
 */
TEST_F(EfaConnTest, alloc_reserve_failure_rdm_cleanup)
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
	EFA_EXPECT_CALL(mock_efa, efa_av_array_reserve)
		.WillOnce(Return(-FI_ENOMEM));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add).Times(0);

	addr = efa_test_av_insert_new_ah(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

/**
 * @brief Same as alloc_reserve_failure_rdm_cleanup, through the explicit
 * fi_av_insert path.
 */
TEST_F(EfaConnTest, alloc_reserve_failure_explicit_insert)
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
	EFA_EXPECT_CALL(mock_efa, efa_av_array_reserve)
		.WillOnce(Return(-FI_ENOMEM));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add).Times(0);

	num_addr = efa_test_av_insert_fake_gid(resource.ep, resource.av, &addr);
	EXPECT_EQ(num_addr, 0);
}

/**
 * @brief The implicit -> explicit promotion leaves the existing peer in place,
 * so it is found the instant the entry becomes visible to the CQ read path's
 * lock-free reverse AV lookup.
 *
 * The CQ read path resolves (GID, QPN) -> conn without a lock and only then
 * looks up the conn's peer, so a conn must never be visible in the reverse AV
 * while the peer lookup would miss for a peer that already exists. Promotion
 * does not touch peers at all, which this test checks at the most exposed
 * point: it runs both lookups from inside the wrapped efa_av_reverse_av_add,
 * the instant a racing CQ read could first see the entry, so it is checked
 * deterministically rather than by winning a race.
 */
TEST_F(EfaConnTest, promotion_keeps_peer_when_publishing_reverse_av)
{
	fi_addr_t explicit_fi_addr;
	int num_addr;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	ASSERT_EQ(efa_test_av_publish_ordering_setup(resource.ep, resource.av), 0);

	/* efa_av_reverse_av_add is the last step of publishing the entry to the
	 * reverse AV, so running the probe just before __real_ puts it exactly
	 * where a racing CQ read would first observe the new fi_addr. The
	 * implicit insert in the setup above also reaches here, but it ran
	 * before the mock was installed. */
	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillOnce(Invoke([](struct efa_av_array *cur_reverse_av,
				    struct efa_av_entry *entry) {
			efa_test_av_publish_ordering_probe(entry);
			return __real_efa_av_reverse_av_add(cur_reverse_av,
							    entry);
		}));

	num_addr = efa_test_av_publish_ordering_promote(resource.av,
						       &explicit_fi_addr);
	MockEfa::set(nullptr);
	ASSERT_EQ(num_addr, 1);
	ASSERT_NE(explicit_fi_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);

	const struct efa_test_av_publish_observation *obs =
		efa_test_av_publish_ordering_observation();

	/* Guards the test itself: the probe has to have run before publication,
	 * or the checks below prove nothing. */
	ASSERT_TRUE(obs->ran);
	EXPECT_EQ(obs->explicit_fi_addr, explicit_fi_addr);
	EXPECT_FALSE(obs->reverse_av_resolves);

	/* The ordering under test */
	EXPECT_TRUE(obs->peer_map_resolves);
	EXPECT_TRUE(obs->peer_is_migrated_peer);
}

/**
 * @brief The promotion leaves exactly one peer for the promoted address.
 *
 * Pin the post-condition: the implicit fi_addr no longer resolves, and the
 * explicit fi_addr resolves to the same peer the implicit one had.
 */
TEST_F(EfaConnTest, promotion_keeps_single_peer)
{
	fi_addr_t explicit_fi_addr;
	int num_addr;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	ASSERT_EQ(efa_test_av_publish_ordering_setup(resource.ep, resource.av), 0);

	num_addr = efa_test_av_publish_ordering_promote(resource.av,
						       &explicit_fi_addr);
	ASSERT_EQ(num_addr, 1);

	EXPECT_TRUE(efa_test_av_publish_ordering_single_peer(explicit_fi_addr));
}

/**
 * @brief Promotion into a recycled explicit fi_addr keeps the peer too.
 *
 * Freeing an explicit slot first and then promoting into it proves the checks
 * are not passing only because they happened to run against a fresh fi_addr
 * and a fresh conn slot.
 */
TEST_F(EfaConnTest, promotion_keeps_peer_on_recycled_fi_addr)
{
	fi_addr_t recycled_fi_addr, explicit_fi_addr;
	int num_addr;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	ASSERT_EQ(efa_test_av_publish_ordering_recycle_slot(
			  resource.ep, resource.av, &recycled_fi_addr),
		  0);
	ASSERT_EQ(efa_test_av_publish_ordering_setup(resource.ep, resource.av), 0);

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillOnce(Invoke([](struct efa_av_array *cur_reverse_av,
				    struct efa_av_entry *entry) {
			efa_test_av_publish_ordering_probe(entry);
			return __real_efa_av_reverse_av_add(cur_reverse_av,
							    entry);
		}));

	num_addr = efa_test_av_publish_ordering_promote(resource.av,
						       &explicit_fi_addr);
	MockEfa::set(nullptr);
	ASSERT_EQ(num_addr, 1);

	/* Guards the test itself: without fi_addr reuse this adds no coverage
	 * over promotion_keeps_peer_when_publishing_reverse_av. */
	ASSERT_EQ(explicit_fi_addr, recycled_fi_addr);

	const struct efa_test_av_publish_observation *obs =
		efa_test_av_publish_ordering_observation();

	ASSERT_TRUE(obs->ran);
	EXPECT_FALSE(obs->reverse_av_resolves);
	EXPECT_TRUE(obs->peer_map_resolves);
	EXPECT_TRUE(obs->peer_is_migrated_peer);
}
