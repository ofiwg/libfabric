/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_cq_utils.h"
#include <gtest/gtest.h>

using testing::DoAll;
using testing::Return;
using testing::SetArgPointee;
using testing::StrictMock;
using testing::Test;

/**
 * @brief The CQ read path looks the packet's source up in the explicit AV lock
 * free, then repeats the lookup under the AV locks. These tests drive one
 * fabricated RECV completion and let a concurrent fi_av_insert land in between,
 * from inside the mocked lock free lookup, so the insert is observable only to
 * the lookups under the locks.
 */
class EfaRdmCqRaceTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct efa_ibv_cq *ibv_cq = nullptr;
	uint32_t qp_num = 0;
	struct efa_test_rdm_cq_race_ctx ctx = {};

	void construct(int with_connid)
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_NE(resource.cq, nullptr);
		ibv_cq = efa_test_get_ibv_cq(resource.cq);
		ASSERT_NE(ibv_cq, nullptr);
		qp_num = efa_test_get_qp_num(resource.ep);

		ASSERT_EQ(efa_test_rdm_cq_race_setup(resource.ep, resource.av,
						     resource.cq, with_connid,
						     &ctx),
			  0);
		efa_test_set_ibv_cq_ex(ibv_cq, IBV_WC_SUCCESS, ctx.wr_id);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}

	/* The fabricated CQE: a successful RECV on this endpoint's QP, whose
	 * source is reported as (reported_ahn, ctx.qpn). */
	void expect_recv_cqe(uint16_t reported_ahn)
	{
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
			.WillRepeatedly(Return(IBV_WC_RECV));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
			.WillRepeatedly(Return(qp_num));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
			.WillRepeatedly(Return(ctx.pkt_size));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid)
			.WillRepeatedly(Return(reported_ahn));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp)
			.WillRepeatedly(Return(ctx.qpn));
	}

	/*
	 * The packet must be attributed to the peer of the racing insert, and
	 * that peer must be the explicit one: the handshake fields are stamped
	 * by efa_rdm_pke_handle_handshake_recv on pkt_entry->peer only, and a
	 * peer resolved through the implicit AV instead would carry an implicit
	 * fi_addr and leave an entry behind in the implicit AV.
	 */
	void expect_peer_resolved_from_explicit_av()
	{
		struct efa_test_rdm_cq_peer_state peer = {};

		ASSERT_NE(ctx.racing_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
		efa_test_rdm_cq_peer_state(resource.ep, ctx.racing_addr, &peer);

		ASSERT_TRUE(peer.peer_exists);
		EXPECT_TRUE(peer.handshake_received);
		EXPECT_EQ(peer.extra_info0,
			  (uint64_t) EFA_TEST_RDM_CQ_EXTRA_INFO_SENTINEL);
		EXPECT_EQ(peer.nextra_p3, ctx.nextra_p3);
		EXPECT_EQ(peer.device_version,
			  (uint32_t) EFA_TEST_RDM_CQ_DEVICE_VERSION);

		EXPECT_EQ(peer.explicit_fi_addr, ctx.racing_addr);
		EXPECT_EQ(peer.implicit_fi_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
		EXPECT_EQ(efa_test_rdm_cq_implicit_av_count(resource.av), 0u);
	}
};

/**
 * @brief The insert lands after the lock free reverse (AHN, QPN) lookup missed.
 * The packet carries no raw address, so the reverse lookup repeated under the
 * AV locks is the only thing that can find the new entry. Without that repeat
 * the packet is dropped as coming from an unknown peer.
 */
TEST_F(EfaRdmCqRaceTest, reverse_lookup_under_locks_finds_racing_insert)
{
	ASSERT_NO_FATAL_FAILURE(construct(/* with_connid */ 0));

	/* The device reports the AHN the racing insert's address is keyed by. */
	expect_recv_cqe(ctx.ahn);

	/* The lookup runs for real and really misses -- the insert has not
	 * happened yet -- and its actual result is what is returned. */
	EFA_EXPECT_CALL(mock_efa, efa_rdm_av_reverse_lookup)
		.WillOnce([&](struct efa_av *av, uint16_t ahn, uint16_t qpn,
			      struct efa_rdm_pke *pkt_entry) {
			fi_addr_t missed = __real_efa_rdm_av_reverse_lookup(
				av, ahn, qpn, pkt_entry);

			EXPECT_EQ(missed, (fi_addr_t) FI_ADDR_NOTAVAIL);
			EXPECT_NE(efa_test_rdm_cq_race_insert(&ctx),
				  (fi_addr_t) FI_ADDR_NOTAVAIL);
			return missed;
		});
	/* Without a connid the packet carries no raw address at all, so no raw
	 * address lookup is possible. */
	EFA_EXPECT_CALL(mock_efa, ofi_av_lookup_fi_addr).Times(0);

	EXPECT_EQ(efa_test_rdm_cq_race_poll(&ctx), 0);

	ASSERT_NO_FATAL_FAILURE(expect_peer_resolved_from_explicit_av());
}

/**
 * @brief The insert lands after the lock free raw address lookup missed, with
 * the device reporting an AHN this endpoint has no entry for. The raw address
 * lookup repeated under the AV locks is then the only thing that can find the
 * new entry; without it the peer is inserted into the implicit AV instead.
 */
TEST_F(EfaRdmCqRaceTest, raw_addr_lookup_under_locks_finds_racing_insert)
{
	ASSERT_NO_FATAL_FAILURE(construct(/* with_connid */ 1));

	if (!efa_test_rdm_cq_reports_sgid(ibv_cq))
		GTEST_SKIP() << "CQ cannot report a source GID";

	/* No reverse AV entry is ever keyed by this AHN, so every reverse
	 * lookup misses no matter when the insert lands. */
	expect_recv_cqe(ctx.ahn + 1);

	/* The raw address comes from the device, since the packet is not a REQ
	 * packet and so carries no raw address header. */
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_sgid)
		.WillOnce(DoAll(SetArgPointee<1>(ctx.src_gid), Return(0)));

	EFA_EXPECT_CALL(mock_efa, ofi_av_lookup_fi_addr)
		.WillOnce([&](struct util_av *av, const void *addr) {
			fi_addr_t missed =
				__real_ofi_av_lookup_fi_addr(av, addr);

			EXPECT_EQ(missed, (fi_addr_t) FI_ADDR_NOTAVAIL);
			EXPECT_NE(efa_test_rdm_cq_race_insert(&ctx),
				  (fi_addr_t) FI_ADDR_NOTAVAIL);
			return missed;
		});
	/* Pins that miss, so the raw address lookup is the only lookup under the
	 * locks that can resolve the peer. */
	EFA_EXPECT_CALL(mock_efa, efa_rdm_av_reverse_lookup_unsafe)
		.WillOnce(Return(FI_ADDR_NOTAVAIL));

	EXPECT_EQ(efa_test_rdm_cq_race_poll(&ctx), 0);

	ASSERT_NO_FATAL_FAILURE(expect_peer_resolved_from_explicit_av());
}
