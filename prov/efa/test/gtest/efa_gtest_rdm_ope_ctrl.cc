/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_ope_ctrl_utils.h"
#include <gtest/gtest.h>
#include <rdma/fi_errno.h>

using testing::_;
using testing::DoAll;
using testing::Return;
using testing::SaveArg;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::Values;

namespace
{

struct CtrlCase {
	const char *name;
	int which;
};

class EfaRdmOpeCtrlBase : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_MSG | FI_RMA | FI_ATOMIC;

		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));
		ASSERT_NE(resource.ep, nullptr);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		efa_test_ctrl_cleanup();
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

class EfaRdmOpeCtrlPostTest : public EfaRdmOpeCtrlBase,
			     public testing::WithParamInterface<CtrlCase>
{
};

TEST_P(EfaRdmOpeCtrlPostTest, builds_expected_wire_packet)
{
	int which = GetParam().which;
	struct efa_test_ctrl_result res = {};
	struct efa_test_ctrl_wire wire = {};
	uint64_t seen_flags = ~0ull;

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<7>(&seen_flags), Return(0)));
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);

	ASSERT_EQ(efa_test_ctrl_post(resource.ep, resource.av, which,
				     EFA_TEST_CTRL_PERTURB_NONE, &res),
		  0);
	ASSERT_EQ(res.ret, 0);

	efa_test_ctrl_decode_posted(resource.ep, &wire);
	ASSERT_TRUE(wire.decoded);

	EXPECT_EQ(wire.pkt_type, efa_test_ctrl_expected_pkt_type(which));
	EXPECT_EQ(wire.version, efa_test_ctrl_protocol_version());
	EXPECT_TRUE(wire.flags & efa_test_ctrl_connid_hdr_flag());
	EXPECT_EQ(wire.connid, res.ids.connid);
	EXPECT_FALSE(seen_flags & FI_MORE);

	switch (which) {
	case EFA_TEST_CTRL_CTS_RXE:
		EXPECT_EQ(wire.send_id, res.ids.tx_id);
		EXPECT_EQ(wire.recv_id, res.ids.rx_id);
		EXPECT_EQ(wire.recv_length,
			  efa_test_ctrl_expected_cts_recv_length(resource.ep,
								 64));
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which));
		EXPECT_EQ(res.ope_window, wire.recv_length);
		EXPECT_FALSE(wire.flags & efa_test_ctrl_cts_read_req_flag());
		break;
	case EFA_TEST_CTRL_CTS_TXE:
		/* A CTS from a txe swaps the two ids and marks the read. */
		EXPECT_EQ(wire.send_id, res.ids.rx_id);
		EXPECT_EQ(wire.recv_id, res.ids.tx_id);
		EXPECT_TRUE(wire.flags & efa_test_ctrl_cts_read_req_flag());
		EXPECT_EQ(res.ope_window, wire.recv_length);
		break;
	case EFA_TEST_CTRL_READRSP_FITS:
		EXPECT_EQ(wire.send_id, res.ids.rx_id);
		EXPECT_EQ(wire.recv_id, res.ids.tx_id);
		EXPECT_EQ(wire.seg_length, res.source_len);
		EXPECT_EQ(wire.payload_size, res.source_len);
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which) +
				  res.source_len);
		EXPECT_EQ(memcmp(wire.payload_bytes, res.source_bytes,
				 res.source_len),
			  0);
		EXPECT_EQ(res.ope_bytes_sent, res.source_len);
		EXPECT_FALSE(res.on_longcts_send_list);
		break;
	case EFA_TEST_CTRL_READRSP_REMAINDER:
		/* Clamped to one packet, so the rest follows over long CTS. */
		EXPECT_LT(wire.seg_length, res.source_len);
		EXPECT_EQ(wire.seg_length,
			  efa_test_ctrl_readrsp_max_payload(resource.ep));
		EXPECT_TRUE(res.on_longcts_send_list);
		break;
	case EFA_TEST_CTRL_EOR:
	case EFA_TEST_CTRL_READ_NACK:
		EXPECT_EQ(wire.send_id, res.ids.tx_id);
		EXPECT_EQ(wire.recv_id, res.ids.rx_id);
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which));
		break;
	case EFA_TEST_CTRL_RECEIPT:
		EXPECT_EQ(wire.tx_id, res.ids.tx_id);
		EXPECT_EQ(wire.msg_id, res.ids.msg_id);
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which));
		EXPECT_TRUE(res.on_posted_ack_list);
		break;
	case EFA_TEST_CTRL_ATOMRSP:
		EXPECT_EQ(wire.recv_id, res.ids.tx_id);
		EXPECT_EQ(wire.seg_length, res.source_len);
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which) +
				  res.source_len);
		EXPECT_EQ(memcmp(wire.payload_bytes, res.source_bytes,
				 res.source_len),
			  0);
		break;
	case EFA_TEST_CTRL_PEER_ERROR_RXE:
		EXPECT_EQ(wire.op_id, res.ids.tx_id);
		EXPECT_EQ(wire.msg_id, res.ids.msg_id);
		EXPECT_EQ((int) wire.emitter_ope_type,
			  efa_test_ctrl_ope_type_rxe());
		EXPECT_EQ((int) wire.prov_errno, res.ids.prov_errno);
		EXPECT_EQ(wire.pkt_size,
			  efa_test_ctrl_expected_hdr_size(which));
		break;
	case EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO:
		/* The migrated protocol on the txe must not have been used. */
		EXPECT_EQ(wire.op_id, efa_test_ctrl_ope_id_invalid());
		EXPECT_EQ(wire.msg_id, res.ids.msg_id);
		EXPECT_EQ((int) wire.emitter_ope_type,
			  efa_test_ctrl_ope_type_txe());
		EXPECT_EQ((int) wire.prov_errno, res.ids.prov_errno);
		break;
	default:
		FAIL() << "unhandled case " << which;
	}
}

INSTANTIATE_TEST_SUITE_P(
	NonReqCtrl, EfaRdmOpeCtrlPostTest,
	Values(CtrlCase{"cts_rxe", EFA_TEST_CTRL_CTS_RXE},
	       CtrlCase{"cts_txe", EFA_TEST_CTRL_CTS_TXE},
	       CtrlCase{"readrsp_fits", EFA_TEST_CTRL_READRSP_FITS},
	       CtrlCase{"readrsp_remainder", EFA_TEST_CTRL_READRSP_REMAINDER},
	       CtrlCase{"eor", EFA_TEST_CTRL_EOR},
	       CtrlCase{"receipt", EFA_TEST_CTRL_RECEIPT},
	       CtrlCase{"read_nack", EFA_TEST_CTRL_READ_NACK},
	       CtrlCase{"atomrsp", EFA_TEST_CTRL_ATOMRSP},
	       CtrlCase{"peer_error_rxe", EFA_TEST_CTRL_PEER_ERROR_RXE},
	       CtrlCase{"peer_error_txe_proto",
			EFA_TEST_CTRL_PEER_ERROR_TXE_PROTO}),
	[](const testing::TestParamInfo<CtrlCase> &info) {
		return info.param.name;
	});

class EfaRdmOpeCtrlQueueTest : public EfaRdmOpeCtrlBase
{
};

TEST_F(EfaRdmOpeCtrlQueueTest, tx_full_queues_without_allocating)
{
	struct efa_test_ctrl_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);

	ASSERT_EQ(efa_test_ctrl_post(resource.ep, resource.av,
				     EFA_TEST_CTRL_CTS_RXE,
				     EFA_TEST_CTRL_PERTURB_TX_FULL, &res),
		  0);

	EXPECT_EQ(res.ret, 0);
	EXPECT_TRUE(res.queued_ctrl_flag_set);
	EXPECT_EQ(res.queued_ctrl_type,
		  efa_test_ctrl_expected_pkt_type(EFA_TEST_CTRL_CTS_RXE));
	EXPECT_FALSE(res.queued_list_empty);
}

TEST_F(EfaRdmOpeCtrlQueueTest, device_queue_full_queues_and_releases_packet)
{
	struct efa_test_ctrl_result res = {};

	/* ENOMEM is the device's queue-full, which sendv maps to -FI_EAGAIN. */
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillOnce(Return(ENOMEM));
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);

	ASSERT_EQ(efa_test_ctrl_post(resource.ep, resource.av,
				     EFA_TEST_CTRL_CTS_RXE,
				     EFA_TEST_CTRL_PERTURB_NONE, &res),
		  0);

	EXPECT_EQ(res.ret, 0);
	EXPECT_TRUE(res.queued_ctrl_flag_set);
	EXPECT_FALSE(res.queued_list_empty);
	EXPECT_EQ(res.outstanding_tx_ops, 0u);
}

TEST_F(EfaRdmOpeCtrlQueueTest, post_error_is_reported_not_queued)
{
	struct efa_test_ctrl_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillOnce(Return(EINVAL));
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);

	ASSERT_EQ(efa_test_ctrl_post(resource.ep, resource.av,
				     EFA_TEST_CTRL_CTS_RXE,
				     EFA_TEST_CTRL_PERTURB_NONE, &res),
		  0);

	EXPECT_EQ(res.ret, -FI_EINVAL);
	EXPECT_FALSE(res.queued_ctrl_flag_set);
	EXPECT_TRUE(res.queued_list_empty);
	EXPECT_EQ(res.outstanding_tx_ops, 0u);
}

TEST_F(EfaRdmOpeCtrlQueueTest, fi_more_is_not_propagated_to_the_qp)
{
	struct efa_test_ctrl_result res = {};
	uint64_t seen_flags = ~0ull;

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<7>(&seen_flags), Return(0)));
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);

	ASSERT_EQ(efa_test_ctrl_post(resource.ep, resource.av,
				     EFA_TEST_CTRL_CTS_RXE,
				     EFA_TEST_CTRL_PERTURB_FI_MORE, &res),
		  0);

	ASSERT_EQ(res.ret, 0);
	EXPECT_FALSE(seen_flags & FI_MORE);
}

class EfaRdmOpeContinuationTest : public EfaRdmOpeCtrlBase
{
};

TEST_F(EfaRdmOpeContinuationTest, eagain_leaves_ope_queued)
{
	struct efa_test_cont_result res = {};
	uint64_t bytes_sent_before;

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);

	ASSERT_EQ(efa_test_ctrl_setup_longcts_continuation(
			  resource.ep, resource.av, &res),
		  0);
	bytes_sent_before = res.ope_bytes_sent;

	efa_test_ctrl_drive_continuation(resource.ep, 1 /* tx_full */, &res);

	EXPECT_TRUE(res.on_longcts_send_list);
	EXPECT_EQ(res.ope_bytes_sent, bytes_sent_before);
}

TEST_F(EfaRdmOpeContinuationTest, success_posts_continuation_and_advances)
{
	struct efa_test_cont_result res = {};
	uint64_t bytes_sent_before;

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillOnce(Return(0));

	ASSERT_EQ(efa_test_ctrl_setup_longcts_continuation(
			  resource.ep, resource.av, &res),
		  0);
	bytes_sent_before = res.ope_bytes_sent;

	efa_test_ctrl_drive_continuation(resource.ep, 0 /* tx_full */, &res);

	EXPECT_GT(res.ope_bytes_sent, bytes_sent_before);
}

} /* namespace */
