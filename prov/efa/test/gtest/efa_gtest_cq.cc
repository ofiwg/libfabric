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
		EXPECT_CALL(mock_efa, efa_ibv_cq_start_poll).WillOnce(Return(0));
		EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_qp_num)
			.WillOnce(Return(qp_num));
		EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_opcode)
			.WillOnce(Return(IBV_WC_SEND));
		EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_vendor_err)
			.WillOnce(Return(kProvErrno));
		EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_byte_len)
			.WillOnce(Return(64));
		EXPECT_CALL(mock_efa, efa_ibv_cq_wc_read_wc_flags)
			.WillOnce(Return(0));
		EXPECT_CALL(mock_efa, efa_ibv_cq_end_poll).Times(1);
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
