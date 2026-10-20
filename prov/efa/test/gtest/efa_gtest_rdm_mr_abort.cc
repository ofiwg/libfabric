/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_mr_abort_utils.h"
#include "efa_gtest_rdm_ope_helpers.h"
#include <gtest/gtest.h>
#include <rdma/fi_errno.h>
#include <string>

using testing::DoAll;
using testing::Return;
using testing::SaveArg;
using testing::StrictMock;
using testing::Test;

namespace
{

class EfaRdmMrAbortTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	int op_context_sentinel = 0;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));
		ASSERT_NE(resource.ep, nullptr);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		efa_test_mr_abort_cleanup();
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}

	void setup_txe(int req_pkt, bool handshake_received)
	{
		ASSERT_EQ(efa_test_mr_abort_setup(resource.ep, resource.av,
						  req_pkt, handshake_received,
						  &op_context_sentinel),
			  0);
	}

	void expect_cq_empty()
	{
		struct fi_cq_err_entry err_entry = {};

		EXPECT_EQ(fi_cq_readerr(resource.cq, &err_entry, 0), -FI_EAGAIN);
	}

	void expect_peer_aborted_completion()
	{
		struct fi_cq_err_entry err_entry = {};

		ASSERT_EQ(fi_cq_readerr(resource.cq, &err_entry, 0), 1);
		EXPECT_EQ(err_entry.err, FI_ECANCELED);
		EXPECT_EQ(err_entry.prov_errno, efa_test_peer_abort_prov_errno());
		EXPECT_EQ(err_entry.op_context, &op_context_sentinel);
	}
};

TEST_F(EfaRdmMrAbortTest, rnr_on_errored_txe_takes_mr_abort_path)
{
	struct efa_test_mr_abort_state before = {}, st = {};
	uintptr_t wr_id = 0;

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM, true));
	efa_test_mr_abort_snapshot(&before);
	void *pke1 = efa_test_mr_abort_forge_tx_pke(
		EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM);
	void *pke2 = efa_test_mr_abort_forge_tx_pke(
		EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM);
	ASSERT_NE(pke1, nullptr);
	ASSERT_NE(pke2, nullptr);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	efa_test_mr_abort_handle_tx_error(pke1,
					  EFA_TEST_MR_ABORT_ERR_INVALID_LKEY);
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.state_err);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_FALSE(st.peer_error_emitted);
	EXPECT_EQ(st.txe_outstanding, 1u);

	efa_test_mr_abort_handle_tx_error(pke2, EFA_TEST_MR_ABORT_ERR_RNR);
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_FALSE(st.queued_rnr);
	EXPECT_FALSE(st.on_ope_queued_list);
	EXPECT_EQ(st.queued_pkts, 0u);
	EXPECT_EQ(st.ep_rnr_queued_pkt_cnt, 0u);
	EXPECT_TRUE(st.peer_error_emitted);
	EXPECT_EQ(st.txe_outstanding, 1u);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding + 1);
	ASSERT_NE(wr_id, 0u);
	EXPECT_EQ(efa_test_wire_pkt_type_from_wr_id(wr_id),
		  efa_test_mr_abort_wire_type(EFA_TEST_MR_ABORT_PKT_PEER_ERROR));
	expect_cq_empty();

	efa_test_mr_abort_complete_send(wr_id);
	expect_peer_aborted_completion();
	efa_test_mr_abort_snapshot(&st);
	EXPECT_FALSE(st.alive);
}

TEST_F(EfaRdmMrAbortTest, rnr_after_mr_close_takes_mr_abort_path)
{
	struct efa_test_mr_abort_state before = {}, st = {};
	uintptr_t wr_id = 0;

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM, true));
	efa_test_mr_abort_snapshot(&before);
	void *pke = efa_test_mr_abort_forge_tx_pke(
		EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM);
	ASSERT_NE(pke, nullptr);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	efa_test_mr_abort_handle_tx_error(pke, EFA_TEST_MR_ABORT_ERR_RNR);
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.state_err);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_TRUE(st.peer_error_emitted);
	EXPECT_FALSE(st.queued_rnr);
	EXPECT_FALSE(st.on_ope_queued_list);
	EXPECT_EQ(st.ep_rnr_queued_pkt_cnt, 0u);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding + 1);
	ASSERT_NE(wr_id, 0u);
	EXPECT_EQ(efa_test_wire_pkt_type_from_wr_id(wr_id),
		  efa_test_mr_abort_wire_type(EFA_TEST_MR_ABORT_PKT_PEER_ERROR));
	expect_cq_empty();

	efa_test_mr_abort_complete_send(wr_id);
	expect_peer_aborted_completion();
}

TEST_F(EfaRdmMrAbortTest, rnr_on_peer_error_pkt_requeues)
{
	struct efa_test_mr_abort_state st = {};

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_MEDIUM_TAGRTM, true));
	efa_test_mr_abort_set_peer_error_emitted();
	void *pke = efa_test_mr_abort_forge_tx_pke(
		EFA_TEST_MR_ABORT_PKT_PEER_ERROR);
	ASSERT_NE(pke, nullptr);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);

	efa_test_mr_abort_handle_tx_error(pke, EFA_TEST_MR_ABORT_ERR_RNR);
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.queued_rnr);
	EXPECT_TRUE(st.on_ope_queued_list);
	EXPECT_EQ(st.queued_pkts, 1u);
	EXPECT_EQ(st.ep_rnr_queued_pkt_cnt, 1u);
	expect_cq_empty();
}

TEST_F(EfaRdmMrAbortTest, pre_handshake_abort_sends_handshake)
{
	struct efa_test_mr_abort_state before = {}, st = {};
	uintptr_t wr_id = 0;

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_LONGCTS_TAGRTM, false));
	efa_test_mr_abort_set_peer_flags(true, false, false);
	efa_test_mr_abort_snapshot(&before);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	efa_test_mr_abort_txe_handle_error();
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_FALSE(st.peer_error_emitted);
	EXPECT_TRUE(st.peer_handshake_sent);
	EXPECT_FALSE(st.peer_handshake_queued);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding + 1);
	ASSERT_NE(wr_id, 0u);
	EXPECT_EQ(efa_test_wire_pkt_type_from_wr_id(wr_id),
		  efa_test_mr_abort_wire_type(EFA_TEST_MR_ABORT_PKT_HANDSHAKE));
	expect_cq_empty();

	efa_test_mr_abort_complete_send(wr_id);
	efa_test_mr_abort_snapshot(&st);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding);
	EXPECT_TRUE(st.alive);
}

struct PendingHandshakeCase {
	const char *name;
	bool handshake_sent;
	bool handshake_queued;
};

class EfaRdmMrAbortPendingHandshakeTest
	: public EfaRdmMrAbortTest,
	  public testing::WithParamInterface<PendingHandshakeCase>
{
};

TEST_P(EfaRdmMrAbortPendingHandshakeTest, pre_handshake_abort_skips_handshake)
{
	const PendingHandshakeCase &c = GetParam();
	struct efa_test_mr_abort_state before = {}, st = {};

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_LONGCTS_TAGRTM, false));
	efa_test_mr_abort_set_peer_flags(true, c.handshake_sent,
					 c.handshake_queued);
	efa_test_mr_abort_snapshot(&before);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);

	efa_test_mr_abort_txe_handle_error();
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_FALSE(st.peer_error_emitted);
	EXPECT_EQ(st.peer_handshake_sent, c.handshake_sent);
	EXPECT_EQ(st.peer_handshake_queued, c.handshake_queued);
	EXPECT_EQ(st.handshake_queued_peers, c.handshake_queued ? 1u : 0u);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding);
	expect_cq_empty();
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmMrAbortPendingHandshakeTest,
	testing::Values(PendingHandshakeCase{"sent", true, false},
			PendingHandshakeCase{"queued", false, true}),
	[](const testing::TestParamInfo<PendingHandshakeCase> &info) {
		return std::string(info.param.name);
	});

TEST_F(EfaRdmMrAbortTest, ctsdata_completion_on_aborting_txe_writes_nothing)
{
	struct efa_test_mr_abort_state st = {};
	struct fi_cq_data_entry entry = {};

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_LONGCTS_TAGRTM, true));
	efa_test_mr_abort_set_aborting_with_wr_in_flight();

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);

	ASSERT_EQ(efa_test_mr_abort_complete_ctsdata(), 0);
	efa_test_mr_abort_snapshot(&st);
	EXPECT_TRUE(st.alive);
	EXPECT_FALSE(st.peer_error_emitted);
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry),
		  -FI_EAGAIN);
	expect_cq_empty();
}

TEST_F(EfaRdmMrAbortTest, longread_abort_after_post_releases_read_slot)
{
	struct efa_test_mr_abort_state before = {}, st = {};
	uintptr_t wr_id = 0;

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_LONGREAD_TAGRTM, true));
	efa_test_mr_abort_count_read_msg();
	efa_test_mr_abort_snapshot(&before);
	ASSERT_TRUE(before.read_msg_counted);
	ASSERT_EQ(before.read_msg_in_flight, 1);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	efa_test_mr_abort_txe_handle_error();
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding + 1);
	EXPECT_EQ(st.read_msg_in_flight, 0);
	EXPECT_FALSE(st.read_msg_counted);
	EXPECT_TRUE(st.peer_error_prov_errno_is_pkt_post);
	ASSERT_NE(wr_id, 0u);
	EXPECT_EQ(efa_test_wire_pkt_type_from_wr_id(wr_id),
		  efa_test_mr_abort_wire_type(EFA_TEST_MR_ABORT_PKT_PEER_ERROR));

	efa_test_mr_abort_complete_send(wr_id);
	expect_peer_aborted_completion();
}

TEST_F(EfaRdmMrAbortTest, longread_abort_before_post_keeps_read_count)
{
	struct efa_test_mr_abort_state before = {}, st = {};
	uintptr_t wr_id = 0;

	ASSERT_NO_FATAL_FAILURE(
		setup_txe(EFA_TEST_MR_ABORT_PKT_LONGREAD_TAGRTM, true));
	efa_test_mr_abort_set_read_msg_in_flight(1);
	efa_test_mr_abort_snapshot(&before);
	ASSERT_FALSE(before.read_msg_counted);
	ASSERT_EQ(before.read_msg_in_flight, 1);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	efa_test_mr_abort_txe_handle_error();
	efa_test_mr_abort_snapshot(&st);
	ASSERT_TRUE(st.alive);
	EXPECT_TRUE(st.peer_abort_pending);
	EXPECT_EQ(st.ep_outstanding, before.ep_outstanding + 1);
	EXPECT_EQ(st.read_msg_in_flight, 1);
	ASSERT_NE(wr_id, 0u);
	EXPECT_EQ(efa_test_wire_pkt_type_from_wr_id(wr_id),
		  efa_test_mr_abort_wire_type(EFA_TEST_MR_ABORT_PKT_PEER_ERROR));

	efa_test_mr_abort_complete_send(wr_id);
	expect_peer_aborted_completion();
}

} // namespace
