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
 */
class EfaCQPollTxErrTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Exercises efa_cq_handle_error with TX opcodes 
 * (IBV_WC_SEND, IBV_WC_RDMA_READ, IBV_WC_RDMA_WRITE)
 * Tests that it fills out error entry correctly with wr_id not set.
 */
TEST_P(EfaCQPollTxErrTest, poll_error_status)
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

	efa_test_set_ibv_cq_ex(ibv_cq, 1, 0);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);
	EXPECT_EQ(err_entry.op_context, nullptr);
	// flags come from the opcode, since wr_id isn't set
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
 * @brief Same as above, but with wr_id != 0
 */
TEST_P(EfaCQPollTxErrTest, poll_error_status_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context ctx = {};
	ctx.completion_flags = FI_SEND | FI_MSG;
	ctx.addr = 42;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
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

	// wr_id points to our efa_context
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) &ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	// flags come from ctx.completion_flags not the opcode, since wr_id is set
	EXPECT_EQ(err_entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 128);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollTxErrTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

class EfaCQPollRxErrTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Same rrror path with RX opcodes 
 * (IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM) with wr_id set
 */
TEST_P(EfaCQPollRxErrTest, poll_error_status_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context ctx = {};
	ctx.completion_flags = FI_RECV | FI_MSG;
	ctx.addr = 99;

	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
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
	// arbitrary return values, since the real functions would query the device and fail
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(2));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) &ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) &ctx);
	/* 	RECV_RDMA_WITH_IMM overrides flags from the opcode 
	but plain RECV keeps ctx.completion_flags. */	
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
 * @brief Exercises efa_cq_handle_tx_completion with wr_id=0 and
 * TX opcodes (IBV_WC_SEND, IBV_WC_RDMA_READ, IBV_WC_RDMA_WRITE)
 * Check that the handler early-returns without writing a completion.
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
	// an error code terminates the polling loop and becomes the poll return value.
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

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollTxTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief Exercises efa_cq_handle_tx_completion with wr_id=0 and IBV_WC_RECV
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
 * @brief Similar to above, but with IBV_WC_RECV_RDMA_WITH_IMM to the IMM handler. 
 * Asserts that it writes a completion carrying imm_data even without wr_id set.
 */
TEST_F(EfaCQPollTest, poll_rx_rdma_with_imm)
{
	EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillOnce(Return(qp_num));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillOnce(Return(IBV_WC_RECV_RDMA_WITH_IMM));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillOnce(Return(IBV_WC_WITH_IMM));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len).WillOnce(Return(64));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_imm_data)
		.WillOnce(Return(0x1234));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid).WillOnce(Return(1));
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	struct fi_cq_data_entry entry = {};
	/* Exactly one entry is staged */
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry), 1);
	EXPECT_EQ(entry.data, (uint64_t) 0x1234);
	EXPECT_EQ(entry.flags,
		  (uint64_t) (FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE));
	EXPECT_EQ(entry.len, (size_t) 64);
}

/**
 * @brief Test that the loop breaks after processing cqe_to_process entries.
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
	EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(0));
	EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(2, ibv_cq);
	EXPECT_EQ(ret, 0);
}
