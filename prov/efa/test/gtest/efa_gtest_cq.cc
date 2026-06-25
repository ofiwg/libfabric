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
