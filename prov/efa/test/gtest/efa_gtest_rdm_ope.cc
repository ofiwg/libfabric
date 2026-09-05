/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_ope_helpers.h"
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

class EfaRdmOpeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Assert that the context for an unexpected
 * packet is not copied into the error entry
 */
TEST_F(EfaRdmOpeTest, rxe_unexp_error_suppresses_op_context)
{
	int prov_errno = 0;
	void *sentinel = (void *) 0xdeadbeef;
	struct fi_cq_err_entry err_entry = {};

	ASSERT_EQ(efa_test_drive_rxe_unexp_handle_error(
			  resource.ep, sentinel, FI_ENOTCONN, &prov_errno),
		  0);

	ASSERT_EQ(fi_cq_readerr(resource.cq, &err_entry, 0), 1);
	EXPECT_EQ(err_entry.err, FI_ENOTCONN);
	EXPECT_EQ(err_entry.prov_errno, prov_errno);
	/* The sentinel should not be here */
	EXPECT_EQ(err_entry.op_context, nullptr);
}

/**
 * @brief Covers efa_rdm_ope_process_queued_ope's FI_MORE stripping.
 *
 * An op posted with FI_MORE to a peer with no handshake is queued in
 * software; by the time it is replayed the application's flushing
 * (non-FI_MORE) op has already passed, so the repost must not carry the
 * stale FI_MORE into the QP post or the doorbell is never rung.
 */
class EfaRdmOpeQueuedFiMoreTest : public TestWithParam<int>
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
		hints->caps |= FI_MSG | FI_RMA;

		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);

		/* Checked after construct: the device list is only populated
		 * by fi_getinfo. */
		if (!efa_test_device_supports_rma())
			GTEST_SKIP()
				<< "device does not support RDMA read+write";

		MockEfa::set(&mock_efa);
		efa_test_arm_inert_data_path(mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmOpeQueuedFiMoreTest, repost_does_not_carry_fi_more_to_qp)
{
	int op_kind = GetParam();
	struct efa_test_queued_op qop = {};
	uintptr_t wr_id = 0;
	uint64_t seen_flags = 0;

	ASSERT_EQ(efa_test_queue_op_with_fi_more(resource.ep, resource.av,
						 resource.domain, op_kind,
						 &qop),
		  0);
	/* Queued with the caller's FI_MORE preserved on the ope */
	ASSERT_TRUE(qop.fi_more_was_set);

	switch (op_kind) {
	case EFA_TEST_QUEUED_OP_SEND:
		EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
			.WillOnce(DoAll(SaveArg<5>(&wr_id),
					SaveArg<7>(&seen_flags), Return(0)));
		break;
	case EFA_TEST_QUEUED_OP_READ:
		EFA_EXPECT_CALL(mock_efa, efa_qp_post_read)
			.WillOnce(DoAll(SaveArg<5>(&wr_id),
					SaveArg<6>(&seen_flags), Return(0)));
		break;
	case EFA_TEST_QUEUED_OP_WRITE:
		EFA_EXPECT_CALL(mock_efa, efa_qp_post_write)
			.WillOnce(DoAll(SaveArg<7>(&wr_id),
					SaveArg<9>(&seen_flags), Return(0)));
		break;
	default:
		FAIL() << "unknown op kind " << op_kind;
	}

	EXPECT_EQ(efa_test_process_queued_ope_after_handshake(&qop), 0);
	EXPECT_FALSE(seen_flags & FI_MORE);

	efa_test_queued_op_cleanup(&qop, wr_id);
}

INSTANTIATE_TEST_SUITE_P(QueuedOps, EfaRdmOpeQueuedFiMoreTest,
			 Values(EFA_TEST_QUEUED_OP_SEND,
				EFA_TEST_QUEUED_OP_READ,
				EFA_TEST_QUEUED_OP_WRITE),
			 [](const testing::TestParamInfo<int> &info) {
				 switch (info.param) {
				 case EFA_TEST_QUEUED_OP_SEND:
					 return "send";
				 case EFA_TEST_QUEUED_OP_READ:
					 return "read";
				 default:
					 return "write";
				 }
			 });

/**
 * @brief Covers what efa_rdm_ope_process_queued_ope() does with a queued ope
 * either side of the -FI_EAGAIN early return: leave it untouched, or clear its
 * flag, dequeue it, and give back its before-handshake slot.
 *
 * Both tests queue a send with FI_MORE to a peer that has not handshaked.
 */
class EfaRdmOpeProcessQueuedTest : public Test
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
		hints->caps |= FI_MSG;

		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));
		ASSERT_NE(resource.ep, nullptr);

		MockEfa::set(&mock_efa);
		efa_test_arm_inert_data_path(mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaRdmOpeProcessQueuedTest, derives_before_handshake_flag)
{
	struct efa_test_queued_op qop = {};
	struct efa_test_process_queued_result res = {};

	ASSERT_EQ(efa_test_queue_op_with_fi_more(resource.ep, resource.av,
						 resource.domain,
						 EFA_TEST_QUEUED_OP_SEND, &qop),
		  0);

	ASSERT_EQ(efa_test_process_queued_ope_derives_before_handshake_flag(
			  &qop, &res),
		  0);

	/* Dispatched to the before-handshake repost, which short-circuits */
	EXPECT_EQ(res.ret, -FI_EAGAIN);
	/* Reaching the dispatch means the FI_MORE strip ran */
	EXPECT_FALSE(res.fi_more_still_set);
	/* On EAGAIN the ope stays flagged, queued, and counted */
	EXPECT_TRUE(res.before_handshake_flag_set);
	EXPECT_FALSE(res.queued_list_empty);
	EXPECT_EQ(res.before_handshake_cnt, 1u);

	efa_test_queued_op_cleanup(&qop, 0);
}

/*
 * On a successful dispatch the derived flag also drives the bookkeeping: the
 * bit is cleared, the ope leaves ope_queued_list, and the before-handshake
 * counter gives back the slot the queueing path took.
 */
TEST_F(EfaRdmOpeProcessQueuedTest, success_clears_flag_dequeues_and_releases_slot)
{
	struct efa_test_queued_op qop = {};
	struct efa_test_process_queued_result res = {};
	uintptr_t wr_id = 0;

	ASSERT_EQ(efa_test_queue_op_with_fi_more(resource.ep, resource.av,
						 resource.domain,
						 EFA_TEST_QUEUED_OP_SEND, &qop),
		  0);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillOnce(DoAll(SaveArg<5>(&wr_id), Return(0)));

	ASSERT_EQ(efa_test_process_queued_ope_after_handshake_result(&qop, &res),
		  0);

	EXPECT_EQ(res.ret, 0);
	EXPECT_FALSE(res.any_queued_flag_set);
	EXPECT_TRUE(res.queued_list_empty);
	EXPECT_EQ(res.before_handshake_cnt, 0u);

	efa_test_queued_op_cleanup(&qop, wr_id);
}

/*
 * Third outcome: the source MR was closed while the op sat on the queue. The
 * gen check fails before any arm is dispatched, so the op is canceled rather
 * than reposted, and it must be reported as a peer/MR abort -- FI_ECANCELED
 * with the dedicated abort reason code, not the packet-post failure code the
 * other error paths use.
 */
TEST_F(EfaRdmOpeProcessQueuedTest, mr_abort_cancels_without_dispatch)
{
	struct efa_test_queued_op qop = {};
	struct efa_test_process_queued_result res = {};
	struct fi_cq_err_entry err_entry = {};

	ASSERT_EQ(efa_test_queue_ope_with_flag(resource.ep, resource.av,
					       EFA_TEST_QUEUED_FLAG_CTRL, &qop),
		  0);
	efa_test_simulate_source_mr_canceled(&qop);

	/* A canceled op must reach neither a dispatch arm nor the wire */
	EFA_EXPECT_CALL(mock_efa, efa_rdm_ep_post_queued_pkts).Times(0);
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);
	EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_read).Times(0);
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).Times(0);

	ASSERT_EQ(efa_test_process_queued_flag_op(&qop, &res), 0);

	EXPECT_EQ(res.ret, -FI_ECANCELED);
	/* The error handler took the ope off the queue */
	EXPECT_FALSE(res.any_queued_flag_set);
	EXPECT_TRUE(res.queued_list_empty);

	ASSERT_EQ(fi_cq_readerr(resource.cq, &err_entry, 0), 1);
	EXPECT_EQ(err_entry.err, FI_ECANCELED);
	EXPECT_EQ(err_entry.prov_errno, efa_test_peer_abort_prov_errno());

	efa_test_queued_op_cleanup(&qop, 0);
}

/*
 * The flag is no longer supplied by the caller, so the bit pattern alone
 * selects the dispatch arm. Each flag must reach its own post routine and no
 * other, and only BEFORE_HANDSHAKE may touch the before-handshake counter.
 */
class EfaRdmOpeQueuedFlagDispatchTest : public TestWithParam<int>
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
		hints->caps |= FI_MSG;

		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));
		ASSERT_NE(resource.ep, nullptr);

		MockEfa::set(&mock_efa);
		efa_test_arm_inert_data_path(mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmOpeQueuedFlagDispatchTest, derived_flag_selects_post_routine)
{
	int flag_kind = GetParam();
	struct efa_test_queued_op qop = {};
	struct efa_test_process_queued_result res = {};

	ASSERT_EQ(efa_test_queue_ope_with_flag(resource.ep, resource.av,
					       flag_kind, &qop),
		  0);

	/*
	 * Each arm is intercepted one level below the switch, at a seam only
	 * that arm reaches. Times(0) on the other two seams is the assertion
	 * that the derived bit selected no other arm.
	 *
	 * RNR returns success, so the post-dispatch bookkeeping runs under a
	 * flag that is not BEFORE_HANDSHAKE and must leave the before-handshake
	 * counter alone. CTRL and READ return -FI_EAGAIN, which posts nothing
	 * and leaves the ope intact.
	 */
	int expected_ret = -FI_EAGAIN;

	switch (flag_kind) {
	case EFA_TEST_QUEUED_FLAG_RNR:
		expected_ret = 0;
		EFA_EXPECT_CALL(mock_efa, efa_rdm_ep_post_queued_pkts)
			.WillOnce(Return(0));
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_read).Times(0);
		break;
	case EFA_TEST_QUEUED_FLAG_CTRL:
		/* The CTRL arm must forward the recorded queued_ctrl_type */
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data, _,
				qop.queued_ctrl_type, qop.txe, _, _)
			.WillOnce(Return(-FI_EAGAIN));
		EFA_EXPECT_CALL(mock_efa, efa_rdm_ep_post_queued_pkts).Times(0);
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_read).Times(0);
		break;
	case EFA_TEST_QUEUED_FLAG_READ:
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_read)
			.WillOnce(Return(-FI_EAGAIN));
		EFA_EXPECT_CALL(mock_efa, efa_rdm_ep_post_queued_pkts).Times(0);
		EFA_EXPECT_CALL(mock_efa, efa_rdm_pke_fill_data).Times(0);
		break;
	default:
		FAIL() << "unknown flag kind " << flag_kind;
	}

	ASSERT_EQ(efa_test_process_queued_flag_op(&qop, &res), 0);

	EXPECT_EQ(res.ret, expected_ret);
	/* Success dequeues and clears the flag; EAGAIN leaves both in place */
	EXPECT_EQ(!!res.any_queued_flag_set, expected_ret != 0);
	EXPECT_EQ(!!res.queued_list_empty, expected_ret == 0);
	/* Only the BEFORE_HANDSHAKE flag may move this counter */
	EXPECT_EQ(res.before_handshake_cnt, 0u);

	efa_test_queued_op_cleanup(&qop, 0);
}

INSTANTIATE_TEST_SUITE_P(QueuedFlags, EfaRdmOpeQueuedFlagDispatchTest,
			 Values(EFA_TEST_QUEUED_FLAG_RNR,
				EFA_TEST_QUEUED_FLAG_CTRL,
				EFA_TEST_QUEUED_FLAG_READ),
			 [](const testing::TestParamInfo<int> &info) {
				 switch (info.param) {
				 case EFA_TEST_QUEUED_FLAG_RNR:
					 return "rnr";
				 case EFA_TEST_QUEUED_FLAG_CTRL:
					 return "ctrl";
				 default:
					 return "read";
				 }
			 });
