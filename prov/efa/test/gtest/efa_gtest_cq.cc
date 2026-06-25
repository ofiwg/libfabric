/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <errno.h>
#include <gtest/gtest.h>

using testing::Test;
using testing::_;
using testing::Return;
using testing::StrictMock;
using testing::Values;
using testing::WithParamInterface;

/**
 * @brief Tests for efa_cq_poll_ibv_cq — the inner loop that polls the device CQ
 * and dispatches completions by opcode.
 */
class EfaCQPollTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct efa_ibv_cq *ibv_cq = nullptr;
	uint32_t qp_num = 0;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.cq, nullptr);
		ibv_cq = efa_test_get_ibv_cq(resource.cq);
		ASSERT_NE(ibv_cq, nullptr);
		efa_test_alloc_err_buf(ibv_cq);
		qp_num = efa_test_get_qp_num(resource.ep);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		if (ibv_cq)
			efa_test_free_err_buf(ibv_cq);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Parameterized fixture for error-path tests over TX opcodes.
 * All three TX opcodes (SEND, RDMA_READ, RDMA_WRITE) share the same
 * case in fill_err_entry for extracting addr from wr_id.
 */
class EfaCQPollErrTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Error path (status != 0) with wr_id=0: read_entry_common takes the
 * else branch, setting op_context=NULL and flags from the opcode.
 */
TEST_P(EfaCQPollErrTest, poll_error_status)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// qp_num/opcode/vendor_err are each read twice: once in the poll loop,
	// once again in fill_err_entry.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.Times(2)
		.WillRepeatedly(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status != 0 triggers error path; wr_id=0 means no context
	efa_test_set_ibv_cq_ex(ibv_cq, 1, 0);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);
	EXPECT_EQ(err_entry.op_context, nullptr);
	// flags come purely from the opcode; each TX opcode maps to a distinct
	// set, proving the opcode (not a stale ctx) drove the mapping.
	uint64_t expected_flags = (opcode == IBV_WC_SEND) ? (FI_SEND | FI_MSG) :
				  (opcode == IBV_WC_RDMA_READ) ?
							    (FI_RMA | FI_READ) :
							    (FI_RMA | FI_WRITE);
	EXPECT_EQ(err_entry.flags, expected_flags);
	// prov_errno is the raw vendor_err; err is its to_fi_errno mapping.
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}

/**
 * @brief Error path with a valid wr_id: fill_err_entry dereferences wr_id as an
 * efa_context for op_context and flags (TX opcodes take the wr_id addr branch).
 */
TEST_P(EfaCQPollErrTest, poll_error_status_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context ctx = {};
	ctx.completion_flags = FI_SEND | FI_MSG;
	ctx.addr = 42;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// qp_num/opcode/vendor_err read once in the poll loop, once in
	// fill_err_entry.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.Times(2)
		.WillRepeatedly(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillOnce(Return(128));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status != 0 triggers error path; wr_id points to our efa_context
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) &ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	// flags come from ctx.completion_flags, not the opcode: the same
	// sentinel for all three TX opcodes proves the ctx branch ran.
	EXPECT_EQ(err_entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 128);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollErrTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief Parameterized fixture for RX error-path tests.
 * Both RX opcodes (RECV, RECV_RDMA_WITH_IMM) share the same case in
 * fill_err_entry — addr comes from efa_av_reverse_lookup(slid, src_qp)
 * rather than from wr_id.
 */
class EfaCQPollRxErrTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Error path with a valid wr_id, RX opcodes: fill_err_entry takes the
 * slid/src_qp reverse-lookup branch for addr, while read_entry_common still
 * reads op_context/flags from wr_id.
 */
TEST_P(EfaCQPollRxErrTest, poll_error_status_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context ctx = {};
	ctx.completion_flags = FI_RECV | FI_MSG;
	ctx.addr = 99;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// qp_num/opcode/vendor_err read once in the poll loop, once in
	// fill_err_entry.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.Times(2)
		.WillRepeatedly(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillOnce(Return(256));
	// RX branch resolves addr via efa_av_reverse_lookup(slid, src_qp).
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(2));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status != 0 triggers error path; wr_id points to our efa_context
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) &ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	// RECV_RDMA_WITH_IMM overrides flags from the opcode; plain RECV keeps
	// ctx.completion_flags.
	if (opcode == IBV_WC_RECV_RDMA_WITH_IMM)
		EXPECT_EQ(err_entry.flags,
			  (uint64_t) (FI_REMOTE_CQ_DATA | FI_RMA |
				      FI_REMOTE_WRITE));
	else
		EXPECT_EQ(err_entry.flags, (uint64_t) (FI_RECV | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 256);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}

INSTANTIATE_TEST_SUITE_P(RxOpcodes, EfaCQPollRxErrTest,
			 Values(IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM));

/**
 * @brief Parameterized fixture for TX opcodes that all route to
 * efa_cq_handle_tx_completion (SEND, RDMA_READ, RDMA_WRITE).
 */
class EfaCQPollTxTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Success dispatch of TX opcodes to efa_cq_handle_tx_completion with
 * wr_id=0, where the handler early-returns without writing a completion.
 */
TEST_P(EfaCQPollTxTest, poll_tx_completion)
{
	enum ibv_wc_opcode opcode = GetParam();

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	// ENOENT drains the CQ and becomes the poll return value.
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status=0 (success), wr_id=0 (TX handler early-returns)
	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// Early-return means nothing was staged on the util_cq.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry),
		  -FI_EAGAIN);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollTxTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief Success dispatch of IBV_WC_RECV to efa_cq_handle_rx_completion with
 * wr_id=0: the handler early-returns before the FI_SOURCE reverse-lookup, so no
 * slid/src_qp read and no completion staged.
 */
TEST_F(EfaCQPollTest, poll_rx_recv_completion)
{
	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_RECV));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// Early-return means nothing was staged on the util_cq.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry),
		  -FI_EAGAIN);
}

/**
 * @brief Success dispatch of IBV_WC_RECV_RDMA_WITH_IMM to the IMM handler. It
 * has no wr_id guard, so it always reaches the FI_SOURCE reverse-lookup (set by
 * default for RDM endpoints) and writes a completion carrying imm_data.
 */
TEST_F(EfaCQPollTest, poll_rx_rdma_with_imm)
{
	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_RECV_RDMA_WITH_IMM));
	// IBV_WC_WITH_IMM in turn triggers the imm_data read below.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillOnce(Return(IBV_WC_WITH_IMM));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_imm_data)
		.WillOnce(Return(0x1234));
	// FI_SOURCE reverse lookup.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// Read the staged completion back to prove imm_data reached the entry.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry), 1);
	EXPECT_EQ(entry.data, (uint64_t) 0x1234);
	EXPECT_EQ(entry.flags,
		  (uint64_t) (FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE));
	EXPECT_EQ(entry.len, (size_t) 64);
}

/**
 * @brief Test that the loop breaks after processing cqe_to_process entries.
 *
 * Flow: start_poll(ok) -> process CQE#1 -> num_cqe=1 -> next_poll(ok)
 *       -> process CQE#2 -> num_cqe=2 == cqe_to_process -> break
 */
TEST_F(EfaCQPollTest, poll_cqe_limit_breaks_loop)
{
	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// Per-CQE reads fire twice (two CQEs processed before the break).
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(IBV_WC_SEND));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.Times(2)
		.WillRepeatedly(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.Times(2)
		.WillRepeatedly(Return(64));
	// Fires once between the two CQEs; the loop breaks before a 2nd call.
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(2, ibv_cq);
	// 0 (not ENOENT): the loop broke at the limit, not on a drained CQ.
	EXPECT_EQ(ret, 0);
}

/**
 * @brief Fixture for the track_mr-enabled CQ poll branches, where wr_id is an
 * efa_direct_ope* and efa_cq reads op_context/flags/addr through
 * direct_ope->context and releases the op. Enables track_mr (saved/restored)
 * and provides a direct_ope wrapping a sentinel efa_context.
 */
class EfaCQPollTrackMrTest : public EfaCQPollTest
{
	protected:
	int saved_track_mr = 0;
	struct efa_context ctx = {};
	void *direct_ope = nullptr;

	void SetUp() override
	{
		EfaCQPollTest::SetUp();
		saved_track_mr = efa_test_get_track_mr();
		efa_test_set_track_mr(1);

		// Distinct sentinels so assertions prove the production code
		// dereferenced direct_ope->context rather than treating wr_id
		// as an efa_context directly.
		ctx.completion_flags = FI_SEND | FI_MSG;
		ctx.addr = 0xABCD;
		direct_ope = efa_test_alloc_direct_ope(&ctx);
		ASSERT_NE(direct_ope, nullptr);
	}

	void TearDown() override
	{
		efa_test_free_direct_ope(direct_ope);
		efa_test_set_track_mr(saved_track_mr);
		EfaCQPollTest::TearDown();
	}
};

/**
 * @brief Parameterized over TX opcodes (SEND/RDMA_READ/RDMA_WRITE), which all
 * route to efa_cq_handle_tx_completion.
 */
class EfaCQPollTrackMrTxTest :
	public EfaCQPollTrackMrTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief TX completion with track_mr on and a non-NULL wr_id, driving the full
 * efa_cq_handle_tx_completion (read_entry_common's track_mr branch plus the
 * release) rather than the wr_id=0 early-return.
 */
TEST_P(EfaCQPollTrackMrTxTest, poll_tx_completion_track_mr)
{
	enum ibv_wc_opcode opcode = GetParam();

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	// Release receives the raw wr_id direct_ope pointer.
	EXPECT_CALL(mock_efa, efa_direct_ope_release(
				      _, (struct efa_direct_ope *) direct_ope))
		.Times(1);
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status=0 (success), wr_id points to our direct_ope
	efa_test_set_ibv_cq_ex(ibv_cq, 0, (uint64_t) (uintptr_t) direct_ope);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// op_context is direct_ope->context (&ctx), not the raw wr_id, proving
	// read_entry_common followed the track_mr indirection; flags came from
	// ctx.completion_flags.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry), 1);
	EXPECT_EQ(entry.op_context, (void *) &ctx);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(entry.len, (size_t) 64);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollTrackMrTxTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief RX completion with track_mr on and a non-NULL wr_id, driving
 * efa_cq_handle_rx_completion's release and FI_SOURCE reverse lookup.
 */
TEST_F(EfaCQPollTrackMrTest, poll_rx_recv_completion_track_mr)
{
	// efa_cntr_report_rx_completion asserts the flags are an RX flag, so
	// override the fixture's TX sentinel.
	ctx.completion_flags = FI_RECV | FI_MSG;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_RECV));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_direct_ope_release(
				      _, (struct efa_direct_ope *) direct_ope))
		.Times(1);
	// FI_SOURCE reverse lookup.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, (uint64_t) (uintptr_t) direct_ope);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// op_context is direct_ope->context (&ctx), not the raw wr_id, proving
	// the track_mr indirection; flags came from ctx.completion_flags.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry), 1);
	EXPECT_EQ(entry.op_context, (void *) &ctx);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_RECV | FI_MSG));
	EXPECT_EQ(entry.len, (size_t) 64);
}

/**
 * @brief Parameterized over TX opcodes for the track_mr error path.
 */
class EfaCQPollTrackMrErrTest :
	public EfaCQPollTrackMrTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Error path with track_mr on and a non-NULL wr_id (TX opcodes):
 * fill_err_entry's track_mr addr branch, read_entry_common's track_mr branch,
 * and efa_cq_handle_error's release.
 */
TEST_P(EfaCQPollTrackMrErrTest, poll_error_status_track_mr)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// qp_num/opcode/vendor_err read once in the poll loop, once in
	// fill_err_entry.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(opcode));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.Times(2)
		.WillRepeatedly(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillOnce(Return(128));
	EXPECT_CALL(mock_efa, efa_direct_ope_release(
				      _, (struct efa_direct_ope *) direct_ope))
		.Times(1);
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status != 0 triggers error path; wr_id points to our direct_ope
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) direct_ope);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	// op_context is direct_ope->context (&ctx), not the raw wr_id, proving
	// the track_mr indirection was followed.
	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	EXPECT_EQ(err_entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 128);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollTrackMrErrTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief Fixture for the util-cq-bypass data path entry point efa_cq_readfrom
 * (the sole caller of efa_cq_get_src_addr), exercised through fi_cq_readfrom /
 * fi_cq_readerr. track_mr is left at 0, so wr_id is a efa_context*.
 */
class EfaCQReadfromTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct efa_ibv_cq *ibv_cq = nullptr;
	uint32_t qp_num = 0;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		// Must be efa-direct: we want to exercise efa_cq_readfrom 
		// not efa_rdm_cq_readfrom
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_DIRECT_FABRIC_NAME));
		ASSERT_NE(resource.cq, nullptr);
		ibv_cq = efa_test_get_ibv_cq(resource.cq);
		ASSERT_NE(ibv_cq, nullptr);
		qp_num = efa_test_get_qp_num(resource.ep);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief RX completion through fi_cq_readfrom takes efa_cq_get_src_addr's RECV
 * branch: efa_av_reverse_lookup of slid/src_qp under the default FI_SOURCE caps.
 */
TEST_F(EfaCQReadfromTest, readfrom_rx_src_addr_lookup)
{
	struct efa_context ctx = {};
	struct fi_cq_data_entry entry = {};
	fi_addr_t src_addr = FI_ADDR_NOTAVAIL;

	ctx.completion_flags = FI_RECV | FI_MSG;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_RECV));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	// RECV branch reverse lookup: get_base_ep reads qp_num, then slid/src_qp.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(2));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status=0 (success), wr_id points to our efa_context (track_mr off)
	efa_test_set_ibv_cq_ex(ibv_cq, 0, (uint64_t) (uintptr_t) &ctx);

	ssize_t ret = fi_cq_readfrom(resource.cq, &entry, 1, &src_addr);
	EXPECT_EQ(ret, 1);
	EXPECT_EQ(entry.op_context, (void *) &ctx);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_RECV | FI_MSG));
	EXPECT_EQ(entry.len, (size_t) 64);
	// slid/src_qp do not resolve to a real AV entry: the lookup ran but
	// yields FI_ADDR_NOTAVAIL.
	EXPECT_EQ(src_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

/**
 * @brief TX completion through fi_cq_readfrom takes efa_cq_get_src_addr's
 * default branch, returning FI_ADDR_NOTAVAIL without touching the device —
 * proven by the absence of slid/src_qp under StrictMock.
 */
TEST_F(EfaCQReadfromTest, readfrom_tx_src_addr_notavail)
{
	struct efa_context ctx = {};
	struct fi_cq_data_entry entry = {};
	fi_addr_t src_addr = 0;

	ctx.completion_flags = FI_SEND | FI_MSG;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_SEND));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	// No qp_num/slid/src_qp: the TX default branch never touches the device.
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status=0 (success), wr_id points to our efa_context (track_mr off)
	efa_test_set_ibv_cq_ex(ibv_cq, 0, (uint64_t) (uintptr_t) &ctx);

	ssize_t ret = fi_cq_readfrom(resource.cq, &entry, 1, &src_addr);
	EXPECT_EQ(ret, 1);
	EXPECT_EQ(entry.op_context, (void *) &ctx);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(entry.len, (size_t) 64);
	EXPECT_EQ(src_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

/**
 * @brief Like EfaCQReadfromTest but with track_mr enabled, covering the
 * efa_cq_direct_ope_release path on the readfrom/readerr data path.
 */
class EfaCQReadfromTrackMrTest : public EfaCQReadfromTest
{
	protected:
	int saved_track_mr = 0;
	struct efa_context ctx = {};
	void *direct_ope = nullptr;

	void SetUp() override
	{
		EfaCQReadfromTest::SetUp();
		saved_track_mr = efa_test_get_track_mr();
		efa_test_set_track_mr(1);

		// Distinct sentinels so assertions prove the production code
		// dereferenced direct_ope->context rather than treating wr_id
		// as an efa_context directly.
		ctx.completion_flags = FI_SEND | FI_MSG;
		ctx.addr = 0xABCD;
		direct_ope = efa_test_alloc_direct_ope(&ctx);
		ASSERT_NE(direct_ope, nullptr);
	}

	void TearDown() override
	{
		efa_test_free_direct_ope(direct_ope);
		efa_test_set_track_mr(saved_track_mr);
		EfaCQReadfromTest::TearDown();
	}
};

/**
 * @brief Success path: fi_cq_readfrom reads one TX completion and runs
 * efa_cq_direct_ope_release's guarded body. efa_cq_get_src_addr returns
 * FI_ADDR_NOTAVAIL for TX opcodes without touching the mock.
 */
TEST_F(EfaCQReadfromTrackMrTest, readfrom_tx_completion_release)
{
	struct fi_cq_data_entry entry = {};
	fi_addr_t src_addr = FI_ADDR_NOTAVAIL;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_SEND));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	// Release guarded body: get_base_ep reads qp_num, then releases. The
	// pointer is the raw wr_id direct_ope (NOT &ctx) — the release does not
	// follow the track_mr->context indirection that op_context does.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_direct_ope_release(
				      _, (struct efa_direct_ope *) direct_ope))
		.Times(1);
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status=0 (success), wr_id points to our direct_ope
	efa_test_set_ibv_cq_ex(ibv_cq, 0, (uint64_t) (uintptr_t) direct_ope);

	ssize_t ret = fi_cq_readfrom(resource.cq, &entry, 1, &src_addr);
	EXPECT_EQ(ret, 1);
	// op_context/flags came via the track_mr indirection (direct_ope->context
	// == &ctx and its completion_flags), not the raw wr_id or the opcode.
	EXPECT_EQ(entry.op_context, (void *) &ctx);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(entry.len, (size_t) 64);
}

/**
 * @brief Error path: a failing CQE leaves the device CQ poll-active after
 * fi_cq_readfrom returns -FI_EAVAIL; fi_cq_readerr then consumes it and
 * releases the tracked op through efa_cq_direct_ope_release's guarded body
 * (reached from efa_cq_readerr).
 */
TEST_F(EfaCQReadfromTrackMrTest, readerr_release)
{
	struct fi_cq_data_entry entry = {};
	struct fi_cq_err_entry err_entry = {};
	fi_addr_t src_addr = FI_ADDR_NOTAVAIL;

	// readfrom hits status!=0 and bails before end_poll, leaving the poll
	// active for readerr; it reads no opcode/qp_num on that first call.
	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// qp_num read twice in readerr: fill_err_entry + direct_ope_release.
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_SEND));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillOnce(Return(128));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_direct_ope_release(
				      _, (struct efa_direct_ope *) direct_ope))
		.Times(1);
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// status!=0 triggers the error path; wr_id points to our direct_ope
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) direct_ope);

	// First read surfaces EAVAIL and leaves the poll active.
	ssize_t ret = fi_cq_readfrom(resource.cq, &entry, 1, &src_addr);
	EXPECT_EQ(ret, -FI_EAVAIL);

	// readerr consumes the error CQE and releases the tracked op.
	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);
	// op_context/flags came via the track_mr indirection (direct_ope->context
	// == &ctx); prov_errno/err from vendor_err.
	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	EXPECT_EQ(err_entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 128);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}
