/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::_;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::Truly;

class EfaMsgTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	uint8_t *local_buf = nullptr;
	struct fid_mr *local_mr = nullptr;
	void *local_desc = nullptr;
	int prev_track_mr = 0;

	void construct(size_t buf_size)
	{
		int ret;
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);

		local_buf = (uint8_t *) calloc(buf_size, 1);
		ASSERT_NE(local_buf, nullptr);
		ret = fi_mr_reg(resource.domain, local_buf, buf_size,
				FI_SEND | FI_RECV, 0, 0, 0, &local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);

		peer_addr =
			efa_test_insert_self_gid_peer(resource.ep, resource.av);
		ASSERT_NE(peer_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);

		MockEfa::set(&mock_efa);
	}

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);

		if (local_mr) {
			EXPECT_EQ(fi_close(&local_mr->fid), 0);
			local_mr = nullptr;
		}
		free(local_buf);
		local_buf = nullptr;

		efa_test_resource_destruct(&resource);
		efa_test_set_track_mr(prev_track_mr);
	}
};

/**
 * @brief Asserts that efa_post_send with track_mr posts a direct_ope which
 * persists
 */
TEST_F(EfaMsgTest, send_success_keeps_direct_ope_alive)
{
	struct fi_context2 ctx = {};

	prev_track_mr = efa_test_set_track_mr(1);

	ASSERT_NO_FATAL_FAILURE(construct(4096));

	auto wr_id_is_not_ctx = Truly([&ctx](uintptr_t wr_id) {
		return wr_id != 0 && wr_id != (uintptr_t) &ctx;
	});
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send, _, _, _, 1, _,
			wr_id_is_not_ctx, _, _, _, _, _)
		.WillOnce(Return(0));

	int ret = fi_send(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  &ctx);
	EXPECT_EQ(ret, 0);

	EXPECT_EQ(efa_test_ope_list_count(resource.ep), 1u);
}

/**
 * @brief Asserts that efa_post_recv with track_mr posts a direct_ope which
 * persists
 */
TEST_F(EfaMsgTest, recv_success_keeps_direct_ope_alive)
{
	struct fi_context2 ctx = {};

	prev_track_mr = efa_test_set_track_mr(1);

	ASSERT_NO_FATAL_FAILURE(construct(4096));

	auto wr_id_is_not_ctx = Truly([&ctx](const struct ibv_recv_wr *wr) {
		return wr->wr_id != 0 && wr->wr_id != (uintptr_t) &ctx;
	});
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_recv, _, wr_id_is_not_ctx, _)
		.WillOnce(Return(0));

	int ret = fi_recv(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  &ctx);
	EXPECT_EQ(ret, 0);

	EXPECT_EQ(efa_test_ope_list_count(resource.ep), 1u);
}
