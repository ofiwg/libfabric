/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>
#include <rdma/fi_rma.h>

using testing::_;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::Truly;

static constexpr uint64_t kRemoteAddr = 0x87654321;
static constexpr uint32_t kRemoteKey = 123456;

/**
 * @brief Covers efa_rma_post_read / efa_rma_post_write, reached through the
 * public fi_read / fi_write ops on an efa-direct EP that requested FI_RMA.
 * Requires a device with RDMA read+write caps; skips otherwise, since fi_enable
 * creates a real QP with RDMA send-ops flags that a non-RDMA device rejects.
 */
class EfaRmaTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	uint8_t *local_buf = nullptr;
	struct fid_mr *local_mr = nullptr;
	void *local_desc = nullptr;
	int prev_track_mr = 0;

	void construct(struct fi_info *hints)
	{
		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);

		/* Insert the peer before installing the mock so fi_av_insert's
		 * wrapped ibv_create_ah/efadv_query_ah run for real and
		 * populate conn->ah; we only want to intercept the QP post. */
		peer_addr = efa_test_rma_insert_peer(resource.ep, resource.av);
		ASSERT_NE(peer_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);

		MockEfa::set(&mock_efa);
	}

	/* FI_RX_CQ_DATA is required for FI_RMA when the device lacks
	 * unsolicited-write-recv support, so we always set it. */
	struct fi_info *make_hints()
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

		if (hints) {
			hints->caps |= FI_RMA;
			hints->mode |= FI_RX_CQ_DATA;
		}
		return hints;
	}

	void reg_buffer(size_t size)
	{
		int ret;

		local_buf = (uint8_t *) calloc(size, 1);
		ASSERT_NE(local_buf, nullptr);
		ret = fi_mr_reg(resource.domain, local_buf, size,
				FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0,
				&local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);
	}

	void SetUp() override
	{
		if (!efa_test_device_supports_rma())
			GTEST_SKIP()
				<< "device does not support RDMA read+write";

		memset(&resource, 0, sizeof(resource));
		/* Normalize track_mr to 0 and remember the real default so a
		 * test that toggles it stays order-independent. */
		prev_track_mr = efa_test_set_track_mr(0);
	}

	void TearDown() override
	{
		/* Clear the mock first so teardown's wrapped calls (e.g.
		 * ibv_destroy_ah) run for real and aren't checked. */
		MockEfa::set(nullptr);

		if (local_mr) {
			EXPECT_EQ(fi_close(&local_mr->fid), 0);
			local_mr = nullptr;
		}
		free(local_buf);
		local_buf = nullptr;

		/* Reset track_mr only after destruct: efa_ep_close frees the
		 * direct-ope pool only while track_mr is still set. */
		efa_test_resource_destruct(&resource);
		efa_test_set_track_mr(prev_track_mr);
	}
};

/**
 * @brief A 0-byte read takes the zero-byte bounce-buffer branch instead of the
 * normal SGE-loop path.
 */
TEST_F(EfaRmaTest, read_zero_byte_uses_bounce_buffer)
{
	ASSERT_NO_FATAL_FAILURE(construct(make_hints()));

	uint64_t bounce_addr;
	uint32_t bounce_lkey;
	efa_test_get_zero_byte_bounce_buf(resource.ep, &bounce_addr,
					  &bounce_lkey);

	/* The single SGE must point at the bounce buffer (addr + lkey) with
	 * length 0, not at caller memory. */
	auto sge_is_bounce_buf = Truly([=](const struct ibv_sge *sge) {
		return sge[0].addr == bounce_addr && sge[0].length == 0 &&
		       sge[0].lkey == bounce_lkey;
	});

	/* sge_count forced to 1 even though the caller passed NULL/0-length. */
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_read, _, sge_is_bounce_buf, 1,
			kRemoteKey, kRemoteAddr, 0, _, _, _, _)
		.WillOnce(Return(0));

	int ret = fi_read(resource.ep, NULL, 0, NULL, peer_addr, kRemoteAddr,
			  kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief A post returning ENOMEM is remapped to -FI_EAGAIN.
 */
TEST_F(EfaRmaTest, read_post_enomem_maps_to_eagain)
{
	ASSERT_NO_FATAL_FAILURE(construct(make_hints()));
	reg_buffer(4096);

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_read, _, _, 1, kRemoteKey,
			kRemoteAddr, 0, _, _, _, _)
		.WillOnce(Return(ENOMEM));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -FI_EAGAIN);
}

/**
 * @brief A post returning a non-ENOMEM errno is returned negated.
 */
TEST_F(EfaRmaTest, read_post_generic_errno_negated)
{
	ASSERT_NO_FATAL_FAILURE(construct(make_hints()));
	reg_buffer(4096);

	/* EFAULT (not ENOMEM) so we exercise the generic "return -err" arm. */
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_read, _, _, 1, kRemoteKey,
			kRemoteAddr, 0, _, _, _, _)
		.WillOnce(Return(EFAULT));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -EFAULT);
}

/**
 * @brief With track_mr set and a non-NULL context, the read path takes the
 * direct-ope branch: it posts the direct-ope address (not the efa_context) as
 * wr_id, and the direct-ope persists on the ope_list past the post.
 */
TEST_F(EfaRmaTest, read_track_mr_keeps_direct_ope_alive)
{
	struct fi_context2 ctx = {};

	/* Enable track_mr before construct() so the EP allocates the
	 * direct-ope pool. */
	efa_test_set_track_mr(1);

	ASSERT_NO_FATAL_FAILURE(construct(make_hints()));
	reg_buffer(4096);

	/* wr_id is the direct-ope, not &ctx: proves the track_mr branch ran
	 * rather than the plain efa_context path (&ctx != direct_ope). */
	auto wr_id_is_direct_ope = Truly([&ctx](uintptr_t wr_id) {
		return wr_id != 0 && wr_id != (uintptr_t) &ctx;
	});
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_read, _, _, 1, kRemoteKey,
			kRemoteAddr, wr_id_is_direct_ope, _, _, _, _)
		.WillOnce(Return(0));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, &ctx);
	EXPECT_EQ(ret, 0);

	EXPECT_EQ(efa_test_ope_list_count(resource.ep), 1u);
}

/**
 * @brief The efa_rma_post_write track_mr branch: posts the direct-ope as wr_id
 * and leaves it on the ope_list past the post.
 */
TEST_F(EfaRmaTest, write_track_mr_keeps_direct_ope_alive)
{
	struct fi_context2 ctx = {};

	efa_test_set_track_mr(1);

	ASSERT_NO_FATAL_FAILURE(construct(make_hints()));
	reg_buffer(4096);

	auto wr_id_is_direct_ope = Truly([&ctx](uintptr_t wr_id) {
		return wr_id != 0 && wr_id != (uintptr_t) &ctx;
	});
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_write, _, _, 1, _, _, kRemoteKey,
			kRemoteAddr, wr_id_is_direct_ope, _, _, _, _, _)
		.WillOnce(Return(0));

	int ret = fi_write(resource.ep, local_buf, 4096, local_desc, peer_addr,
			   kRemoteAddr, kRemoteKey, &ctx);
	EXPECT_EQ(ret, 0);

	EXPECT_EQ(efa_test_ope_list_count(resource.ep), 1u);
}
