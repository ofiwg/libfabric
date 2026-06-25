/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <cstdlib>
#include <errno.h>
#include <gtest/gtest.h>

using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::Values;
using testing::WithParamInterface;

class EfaCqTest : public Test
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

	static constexpr uint32_t kProvErrno = 1;

	/* Set up expectations to get an error cqe */
	void set_expectations_for_cq_err()
	{
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
			.WillOnce(Return(qp_num));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
			.WillOnce(Return(IBV_WC_SEND));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
			.WillOnce(Return(kProvErrno));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
			.WillOnce(Return(64));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
			.WillOnce(Return(0));
		EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);
	}

	void drive_error_cqe(struct efa_context *ctx,
			     struct fi_cq_err_entry *err_entry)
	{
		struct fi_cq_data_entry data_entry = {};

		efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) ctx);

		ssize_t ret = fi_cq_readfrom(resource.cq, &data_entry, 1,
					     nullptr);
		ASSERT_EQ(ret, -FI_EAVAIL);

		ssize_t readerr_ret = fi_cq_readerr(resource.cq, err_entry, 0);
		ASSERT_EQ(readerr_ret, 1);

		EXPECT_EQ(err_entry->prov_errno, kProvErrno);
		EXPECT_NE(err_entry->err, 0);
	}
};

/**
 * @brief fi_cq_readerr must not write past the caller's err_data buffer.
 */
TEST_F(EfaCqTest, readerr_respects_small_err_data_buffer)
{
	struct fi_cq_err_entry err_entry = {};

	struct efa_context *ctx =
		efa_test_alloc_context(FI_SEND | FI_MSG, FI_ADDR_NOTAVAIL);
	ASSERT_NE(ctx, nullptr);

	// only allow the provider to write 16 bytes,
	// and use backing buffer to catch overwrites
	constexpr size_t kDeclared = 16;
	constexpr size_t kBacking = 1024;
	constexpr unsigned char kCanary = 0xA5;
	unsigned char buf[kBacking];
	memset(buf, kCanary, sizeof(buf));

	err_entry.err_data = buf;
	err_entry.err_data_size = kDeclared;

	set_expectations_for_cq_err();
	drive_error_cqe(ctx, &err_entry);

	/* no byte at or beyond the declared size may be written. */
	for (size_t i = kDeclared; i < kBacking; i++)
		ASSERT_EQ(buf[i], kCanary)
			<< "err_data written out of bounds at offset " << i;

	/* reported size must not exceed the caller's buffer. */
	EXPECT_LE(err_entry.err_data_size, kDeclared);

	free(ctx);
}

/**
 * @brief fi_cq_readerr will use provider-owned buffer if error_data_size == 0.
 */
TEST_F(EfaCqTest, readerr_falls_back_to_internal_err_buf)
{
	struct fi_cq_err_entry err_entry = {};

	struct efa_context *ctx =
		efa_test_alloc_context(FI_SEND | FI_MSG, FI_ADDR_NOTAVAIL);
	ASSERT_NE(ctx, nullptr);

	err_entry.err_data = nullptr;
	err_entry.err_data_size = 0;

	set_expectations_for_cq_err();
	drive_error_cqe(ctx, &err_entry);

	/* err_data is set to the efa-owned buffer */
	ASSERT_EQ(err_entry.err_data, efa_test_get_cq_err_buf(resource.cq));
	ASSERT_GT(err_entry.err_data_size, 0u);
	ASSERT_LE(err_entry.err_data_size, efa_test_cq_err_buf_len);
	EXPECT_EQ(((const char *) err_entry.err_data)[err_entry.err_data_size - 1],
		  '\0');

	free(ctx);
}

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
 * @brief Parameterized fixture for error-path tests over TX opcodes.
 */
class EfaCQPollErrTxTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Exercises efa_cq_handle_error with TX opcodes
 * (IBV_WC_SEND, IBV_WC_RDMA_READ, IBV_WC_RDMA_WRITE)
 * Tests that it fills out error entry correctly with wr_id not set.
 */
TEST_P(EfaCQPollErrTxTest, err_entry_without_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};

	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(opcode));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 1, 0);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);
	EXPECT_EQ(err_entry.op_context, nullptr);
	// flags come from the opcode, since wr_id isn't set
	uint64_t expected_flags;
	switch (opcode) {
	case IBV_WC_SEND:
		expected_flags = FI_SEND | FI_MSG;
		break;
	case IBV_WC_RDMA_READ:
		expected_flags = FI_RMA | FI_READ;
		break;
	case IBV_WC_RDMA_WRITE:
		expected_flags = FI_RMA | FI_WRITE;
		break;
	default:
		FAIL() << "unexpected opcode " << opcode;
	}
	EXPECT_EQ(err_entry.flags, expected_flags);
	// prov_errno is the raw vendor_err; err is its to_fi_errno mapping.
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);
}

/**
 * @brief Same as above, but with wr_id != 0
 */
TEST_P(EfaCQPollErrTxTest, err_entry_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context *ctx =
		efa_test_alloc_context(FI_SEND | FI_MSG, 42);
	ASSERT_NE(ctx, nullptr);

	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(opcode));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(128));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	// wr_id points to our efa_context
	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) ctx);
	// flags come from ctx.completion_flags not the opcode, since wr_id is set
	EXPECT_EQ(err_entry.flags, (uint64_t) (FI_SEND | FI_MSG));
	EXPECT_EQ(err_entry.len, (size_t) 128);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);

	free(ctx);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollErrTxTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

class EfaCQPollErrRxTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Exercises efa_cq_handle_error with RX opcodes
 * (IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM) with wr_id set
 */
TEST_P(EfaCQPollErrRxTest, err_entry_with_context)
{
	enum ibv_wc_opcode opcode = GetParam();
	struct fi_cq_err_entry err_entry = {};
	struct efa_context *ctx =
		efa_test_alloc_context(FI_RECV | FI_MSG, 99);
	ASSERT_NE(ctx, nullptr);

	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(opcode));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(256));
	// slid/src_qp only feed the err-message addr lookup; value is arbitrary.
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp)
		.WillRepeatedly(Return(2));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 1, (uint64_t) (uintptr_t) ctx);

	int ret = efa_cq_poll_ibv_cq(1, ibv_cq);
	EXPECT_EQ(ret, 0);

	ssize_t readerr_ret = fi_cq_readerr(resource.cq, &err_entry, 0);
	EXPECT_EQ(readerr_ret, 1);

	EXPECT_EQ(err_entry.op_context, (void *) ctx);
	/* 	RECV_RDMA_WITH_IMM overrides flags from the opcode
	but plain RECV keeps ctx.completion_flags. */
	uint64_t expected_flags;
	switch (opcode) {
	case IBV_WC_RECV:
		expected_flags = FI_RECV | FI_MSG;
		break;
	case IBV_WC_RECV_RDMA_WITH_IMM:
		expected_flags = FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE;
		break;
	default:
		FAIL() << "unexpected opcode " << opcode;
	}
	EXPECT_EQ(err_entry.flags, expected_flags);
	EXPECT_EQ(err_entry.len, (size_t) 256);
	EXPECT_EQ(err_entry.prov_errno, 1);
	EXPECT_NE(err_entry.err, 0);

	free(ctx);
}

INSTANTIATE_TEST_SUITE_P(RxOpcodes, EfaCQPollErrRxTest,
			 Values(IBV_WC_RECV, IBV_WC_RECV_RDMA_WITH_IMM));

/**
 * @brief Parameterized fixture for TX opcodes that all route to
 * efa_cq_handle_tx_completion (SEND, RDMA_READ, RDMA_WRITE).
 */
class EfaCQPollCompletionTxTest :
	public EfaCQPollTest,
	public WithParamInterface<enum ibv_wc_opcode>
{
};

/**
 * @brief Exercises efa_cq_handle_tx_completion with wr_id=0 and
 * TX opcodes (IBV_WC_SEND, IBV_WC_RDMA_READ, IBV_WC_RDMA_WRITE)
 * Check that the handler early-returns without writing a completion.
 */
TEST_P(EfaCQPollCompletionTxTest, no_context_writes_no_completion)
{
	enum ibv_wc_opcode opcode = GetParam();

	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(opcode));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(64));
	// an error code terminates the polling loop and becomes the poll return value.
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(10, ibv_cq);
	EXPECT_EQ(ret, ENOENT);

	// Early-return means nothing was staged on the util_cq.
	struct fi_cq_data_entry entry = {};
	EXPECT_EQ(efa_test_cq_read_staged_data_entry(resource.cq, &entry),
		  -FI_EAGAIN);
}

INSTANTIATE_TEST_SUITE_P(TxOpcodes, EfaCQPollCompletionTxTest,
			 Values(IBV_WC_SEND, IBV_WC_RDMA_READ,
				IBV_WC_RDMA_WRITE));

/**
 * @brief Exercises efa_cq_handle_rx_completion with wr_id=0 and IBV_WC_RECV
 */
TEST_F(EfaCQPollTest, rx_recv_no_context_writes_no_completion)
{
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(IBV_WC_RECV));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(64));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

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
TEST_F(EfaCQPollTest, rx_rdma_with_imm_writes_completion_without_context)
{
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.WillRepeatedly(Return(IBV_WC_RECV_RDMA_WITH_IMM));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.WillRepeatedly(Return(IBV_WC_WITH_IMM));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.WillRepeatedly(Return(64));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_imm_data)
		.WillRepeatedly(Return(0x1234));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_src_qp)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_slid)
		.WillRepeatedly(Return(1));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(ENOENT));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

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
TEST_F(EfaCQPollTest, stops_at_cqe_to_process)
{
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
	// Per-CQE reads fire twice (two CQEs processed before the break).
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
		.Times(2)
		.WillRepeatedly(Return(qp_num));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
		.Times(2)
		.WillRepeatedly(Return(IBV_WC_SEND));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
		.Times(2)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
		.Times(2)
		.WillRepeatedly(Return(64));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_next_poll).WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);

	efa_test_set_ibv_cq_ex(ibv_cq, 0, 0);

	int ret = efa_cq_poll_ibv_cq(2, ibv_cq);
	EXPECT_EQ(ret, 0);
}
