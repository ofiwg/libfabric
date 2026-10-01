/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_wr_utils.h"
#include <cerrno>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

using testing::_;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::Truly;
using testing::Values;

#define EFA_TEST_WR_SEG_LEN 4096

class EfaWrFlushTestBase
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	uint8_t *local_buf = nullptr;
	struct fid_mr *local_mr = nullptr;
	void *local_desc = nullptr;

	void construct()
	{
		int ret;
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);

		local_buf = (uint8_t *) calloc(2 * EFA_TEST_WR_SEG_LEN, 1);
		ASSERT_NE(local_buf, nullptr);
		ret = fi_mr_reg(resource.domain, local_buf,
				2 * EFA_TEST_WR_SEG_LEN, FI_SEND | FI_RECV, 0,
				0, 0, &local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		MockEfa::set(&mock_efa);
	}

	/* Stage one receive with FI_MORE, i.e. leave it on the receive queue
	 * without handing it to the device. @p seq selects both the buffer
	 * segment and the context, so each staged receive is distinguishable. */
	void stage_recv(void *context, size_t seq)
	{
		struct iovec iov;
		struct fi_msg msg;

		iov.iov_base = local_buf + seq * (EFA_TEST_WR_SEG_LEN / 2);
		iov.iov_len = EFA_TEST_WR_SEG_LEN / 2;

		memset(&msg, 0, sizeof(msg));
		msg.msg_iov = &iov;
		msg.desc = &local_desc;
		msg.iov_count = 1;
		msg.addr = peer_addr;
		msg.context = context;

		ASSERT_EQ(fi_recvmsg(resource.ep, &msg, FI_MORE), 0);
	}

	void setup() { memset(&resource, 0, sizeof(resource)); }

	void teardown()
	{
		MockEfa::set(nullptr);

		if (local_mr) {
			EXPECT_EQ(fi_close(&local_mr->fid), 0);
			local_mr = nullptr;
		}
		free(local_buf);
		local_buf = nullptr;

		efa_test_resource_destruct(&resource);
	}
};

class EfaWrFlushTest : public EfaWrFlushTestBase, public Test
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

/**
 * @brief fi_recv_flush posts the whole chain staged by FI_MORE receives in a
 * single ibv_post_recv, and leaves nothing staged behind.
 */
TEST_F(EfaWrFlushTest, recv_flush_posts_staged_more_chain)
{
	struct fi_context2 ctx[2] = {};

	ASSERT_NO_FATAL_FAILURE(construct());

	auto chain_of_both_segments = Truly([this](const struct ibv_recv_wr *wr) {
		return wr->num_sge == 1 &&
		       wr->sg_list[0].addr == (uintptr_t) local_buf &&
		       wr->next && wr->next->num_sge == 1 &&
		       wr->next->sg_list[0].addr ==
			       (uintptr_t) (local_buf + EFA_TEST_WR_SEG_LEN / 2) &&
		       wr->next->next == nullptr;
	});
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_recv, _, chain_of_both_segments,
			_)
		.WillOnce(Return(0));

	ASSERT_NO_FATAL_FAILURE(stage_recv(&ctx[0], 0));
	ASSERT_NO_FATAL_FAILURE(stage_recv(&ctx[1], 1));
	ASSERT_EQ(efa_test_ep_recv_wr_index(resource.ep), 2u);

	EXPECT_EQ(fi_recv_flush(resource.ep, 0), 0);
	EXPECT_EQ(efa_test_ep_recv_wr_index(resource.ep), 0u);

	/* The chain was consumed: flushing again posts nothing, which the
	 * single WillOnce above enforces. */
	EXPECT_EQ(fi_recv_flush(resource.ep, 0), 0);
}

/**
 * @brief With no receive staged, fi_recv_flush has nothing to initiate.
 */
TEST_F(EfaWrFlushTest, recv_flush_without_staged_work_posts_nothing)
{
	ASSERT_NO_FATAL_FAILURE(construct());

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_recv, _, _, _).Times(0);

	EXPECT_EQ(fi_recv_flush(resource.ep, 0), 0);
	EXPECT_EQ(efa_test_ep_recv_wr_index(resource.ep), 0u);
}

struct efa_test_wr_post_err {
	/* positive errno ibv_post_recv reports */
	int post_errno;
	/* negative libfabric error fi_recv_flush must return for it */
	ssize_t expected_ret;
	const char *name;
};

class EfaWrRecvFlushErrTest : public EfaWrFlushTestBase,
			      public TestWithParam<efa_test_wr_post_err>
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

/**
 * @brief A failed post of the staged chain is reported as a libfabric error --
 * a full receive queue as -FI_EAGAIN, anything else as the negated errno -- and
 * the chain is dropped either way, so a later flush does not re-post it.
 */
TEST_P(EfaWrRecvFlushErrTest, recv_flush_maps_post_error)
{
	struct fi_context2 ctx = {};

	ASSERT_NO_FATAL_FAILURE(construct());

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_recv, _, _, _)
		.WillOnce(Return(GetParam().post_errno));

	ASSERT_NO_FATAL_FAILURE(stage_recv(&ctx, 0));
	ASSERT_EQ(efa_test_ep_recv_wr_index(resource.ep), 1u);

	EXPECT_EQ(fi_recv_flush(resource.ep, 0), GetParam().expected_ret);
	EXPECT_EQ(efa_test_ep_recv_wr_index(resource.ep), 0u);

	EXPECT_EQ(fi_recv_flush(resource.ep, 0), 0);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrRecvFlushErrTest,
	Values(efa_test_wr_post_err{ENOMEM, -FI_EAGAIN, "enomem_to_eagain"},
	       efa_test_wr_post_err{EINVAL, -EINVAL, "einval_negated"}),
	[](const testing::TestParamInfo<efa_test_wr_post_err> &info) {
		return std::string(info.param.name);
	});

/**
 * @brief With nothing deferred, fi_tx_flush succeeds without starting a
 * transmit batch.
 */
TEST_F(EfaWrFlushTest, tx_flush_without_pending_work_is_noop)
{
	ASSERT_NO_FATAL_FAILURE(construct());

	ASSERT_FALSE(efa_test_ep_tx_wr_pending(resource.ep));

	EXPECT_EQ(fi_tx_flush(resource.ep, 0), 0);
	EXPECT_FALSE(efa_test_ep_tx_wr_pending(resource.ep));
}
