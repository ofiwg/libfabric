/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

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
 * @brief Base fixture for efa_rma_post_read / efa_rma_post_write, reached
 * through the public fi_read / fi_write ops on an efa-direct EP that
 * requested FI_RMA. Requires a device with RDMA read+write caps, otherwise skips test.
 */
class EfaRmaTestBase : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	/* A registered local buffer + its descriptor, set by reg_buffer(). */
	uint8_t *local_buf = nullptr;
	struct fid_mr *local_mr = nullptr;
	void *local_desc = nullptr;

	/* When non-zero, the write inline branch will be taken. */
	size_t inject_rma_size = 0;

	void construct(struct fi_info *hints)
	{
		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);

		if (inject_rma_size)
			efa_test_set_inject_rma_size(resource.ep,
						     inject_rma_size);

		/* Insert the peer before installing the mock so fi_av_insert's
		 * wrapped ibv_create_ah/efadv_query_ah run for real and populate
		 * conn->ah; we only want to intercept the QP post. */
		peer_addr = efa_test_rma_insert_peer(resource.ep, resource.av);
		ASSERT_NE(peer_addr, (fi_addr_t) FI_ADDR_NOTAVAIL);

		MockEfa::set(&mock_efa);
	}

	/* efa-direct hints that request FI_RMA. FI_RX_CQ_DATA is required for
	 * FI_RMA when the device lacks unsolicited-write-recv support, so we
	 * always set it. */
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
				FI_SEND | FI_RECV, 0, 0, 0, &local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);
	}

	void SetUp() override
	{
		if (!efa_test_device_supports_rma())
			GTEST_SKIP()
				<< "device does not support RDMA read+write";

		memset(&resource, 0, sizeof(resource));
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
		efa_test_set_track_mr(0);
	}
};

/* ---------------------------------------------------------------------------
 * efa_rma_post_read
 * ------------------------------------------------------------------------- */

class EfaRmaReadTest : public EfaRmaTestBase
{
};

/**
 * @brief 0-byte read takes the bounce-buffer branch: iov_count is forced to 1
 * and the single SGE is wired to the domain's zero-byte bounce buffer
 * (addr + lkey) with length 0, rather than to any caller memory (no desc
 * required). Asserting the SGE contents is what distinguishes this branch from
 * the normal SGE-loop path.
 */
TEST_F(EfaRmaReadTest, read_zero_byte_uses_bounce_buffer)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));

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
	EXPECT_CALL(mock_efa,
		    efa_qp_post_read(_, sge_is_bounce_buf, 1, kRemoteKey,
				     kRemoteAddr, 0, _, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_read(resource.ep, NULL, 0, NULL, peer_addr, kRemoteAddr,
			  kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief When the underlying post returns ENOMEM, efa_rma_post_read remaps it
 * to -FI_EAGAIN; any other positive errno is returned negated.
 */
TEST_F(EfaRmaReadTest, read_post_enomem_maps_to_eagain)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	EXPECT_CALL(mock_efa, efa_qp_post_read(_, _, 1, kRemoteKey, kRemoteAddr,
					       0, _, _, _, _))
		.WillOnce(Return(ENOMEM));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -FI_EAGAIN);
}

TEST_F(EfaRmaReadTest, read_post_generic_errno_negated)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	/* EFAULT (not ENOMEM) so we exercise the generic "return -err" arm. */
	EXPECT_CALL(mock_efa, efa_qp_post_read(_, _, 1, kRemoteKey, kRemoteAddr,
					       0, _, _, _, _))
		.WillOnce(Return(EFAULT));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -EFAULT);
}

/**
 * @brief With track_mr set and a non-NULL context, the read path allocates a
 * direct-ope and posts its address as wr_id instead of the efa_context. The
 * non-track path would post wr_id == &ctx, so wr_id non-zero and != &ctx
 * confirms the direct-ope branch.
 */
TEST_F(EfaRmaReadTest, read_track_mr_allocates_direct_ope)
{
	struct fi_info *hints = make_hints();
	struct fi_context2 ctx = {};

	/* Enable track_mr before construct() so the EP allocates the
	 * direct-ope pool. */
	efa_test_set_track_mr(1);

	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	/* The branch needs efa_ctx non-NULL: both a non-NULL context (&ctx) and
	 * FI_COMPLETION in the tx flags. Without it wr_id would be 0 and the
	 * matcher would fail. */
	auto wr_id_is_direct_ope = Truly([&ctx](uintptr_t wr_id) {
		return wr_id != 0 && wr_id != (uintptr_t) &ctx;
	});
	EXPECT_CALL(mock_efa, efa_qp_post_read(_, _, 1, kRemoteKey, kRemoteAddr,
					       wr_id_is_direct_ope, _, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_read(resource.ep, local_buf, 4096, local_desc, peer_addr,
			  kRemoteAddr, kRemoteKey, &ctx);
	EXPECT_EQ(ret, 0);
}

/* ---------------------------------------------------------------------------
 * efa_rma_post_write
 * ------------------------------------------------------------------------- */

class EfaRmaWriteTest : public EfaRmaTestBase
{
};

/**
 * @brief fi_writedata forwards FI_REMOTE_CQ_DATA and the 64-bit immediate to
 * efa_qp_post_write.
 */
TEST_F(EfaRmaWriteTest, writedata_forwards_remote_cq_data)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	const uint64_t imm = 0xdeadbeef;
	/* data (9th arg) is the immediate; flags (10th arg) must carry
	 * FI_REMOTE_CQ_DATA. */
	auto has_remote_cq_data = Truly([](uint64_t flags) {
		return (flags & FI_REMOTE_CQ_DATA) != 0;
	});
	EXPECT_CALL(mock_efa, efa_qp_post_write(_, _, 1, _, false, kRemoteKey,
						kRemoteAddr, 0, imm,
						has_remote_cq_data, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_writedata(resource.ep, local_buf, 4096, local_desc, imm,
			       peer_addr, kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief When the underlying post returns ENOMEM, efa_rma_post_write remaps it
 * to -FI_EAGAIN; any other positive errno is returned negated.
 */
TEST_F(EfaRmaWriteTest, write_post_enomem_maps_to_eagain)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	EXPECT_CALL(mock_efa, efa_qp_post_write(_, _, 1, _, false, kRemoteKey,
						kRemoteAddr, 0, 0, _, _, _, _))
		.WillOnce(Return(ENOMEM));

	int ret = fi_write(resource.ep, local_buf, 4096, local_desc, peer_addr,
			   kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -FI_EAGAIN);
}

TEST_F(EfaRmaWriteTest, write_post_generic_errno_negated)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	/* EFAULT (not ENOMEM) so we exercise the generic "return -err" arm. */
	EXPECT_CALL(mock_efa, efa_qp_post_write(_, _, 1, _, false, kRemoteKey,
						kRemoteAddr, 0, 0, _, _, _, _))
		.WillOnce(Return(EFAULT));

	int ret = fi_write(resource.ep, local_buf, 4096, local_desc, peer_addr,
			   kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, -EFAULT);
}

/**
 * @brief 0-byte write takes the bounce-buffer branch: use_inline is forced
 * false, iov_count becomes 1, and the single SGE is wired to the domain's
 * zero-byte bounce buffer (addr + lkey, length 0) rather than caller memory.
 * Asserting the SGE contents (and use_inline false) distinguishes this branch
 * from both the inline and normal SGE-loop paths.
 */
TEST_F(EfaRmaWriteTest, write_zero_byte_uses_bounce_buffer)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));

	uint64_t bounce_addr;
	uint32_t bounce_lkey;
	efa_test_get_zero_byte_bounce_buf(resource.ep, &bounce_addr,
					  &bounce_lkey);

	auto sge_is_bounce_buf = Truly([=](const struct ibv_sge *sge) {
		return sge[0].addr == bounce_addr && sge[0].length == 0 &&
		       sge[0].lkey == bounce_lkey;
	});

	/* use_inline (5th arg) forced false even though total_len fits inline.
	 */
	EXPECT_CALL(mock_efa, efa_qp_post_write(_, sge_is_bounce_buf, 1, _,
						false, kRemoteKey, kRemoteAddr,
						0, 0, _, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_write(resource.ep, NULL, 0, NULL, peer_addr, kRemoteAddr,
			   kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief FI_INJECT with a message exceeding inject_rma_size fails with
 * -FI_EINVAL (the !len_fits_inline arm) before any post. This fixture sets no
 * inject_size hint, so inject_rma_size is 0 and the 4096-byte message always
 * exceeds it.
 */
TEST_F(EfaRmaWriteTest, write_inject_too_large_returns_einval)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(4096);

	struct iovec iov = {local_buf, 4096};
	struct fi_rma_iov rma_iov = {kRemoteAddr, 4096, kRemoteKey};
	struct fi_msg_rma msg = {};
	msg.msg_iov = &iov;
	msg.desc = &local_desc;
	msg.iov_count = 1;
	msg.addr = peer_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	EXPECT_CALL(mock_efa, efa_qp_post_write).Times(0);

	int ret = fi_writemsg(resource.ep, &msg, FI_INJECT);
	EXPECT_EQ(ret, -FI_EINVAL);
}

/**
 * @brief Write fixture with a non-zero inject_rma_size so the inline-write
 * branch is reachable.
 */
class EfaRmaWriteInlineTest : public EfaRmaTestBase
{
	protected:
	void SetUp() override
	{
		EfaRmaTestBase::SetUp();
		inject_rma_size = 64;
	}
};

/**
 * @brief Happy-path inline write: a small (<= inject_rma_size) non-HMEM write
 * takes the inline path, so efa_qp_post_write is called with use_inline true
 * and the inline_data_list (not the SGE list) is built from the caller's iov.
 */
TEST_F(EfaRmaWriteInlineTest, write_small_uses_inline)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(32);

	/* The inline path builds inline_data_list[0] from the caller's iov
	 * (addr + length), not from a desc; assert it points at local_buf. */
	auto inline_buf_is_local = Truly([=](const struct ibv_data_buf *dbuf) {
		return dbuf[0].addr == (void *) local_buf &&
		       dbuf[0].length == 32;
	});

	/* use_inline (5th arg) true; inline_data_list (4th arg) carries the
	 * iov. */
	EXPECT_CALL(mock_efa, efa_qp_post_write(_, _, 1, inline_buf_is_local,
						true, kRemoteKey, kRemoteAddr,
						0, 0, _, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_write(resource.ep, local_buf, 32, local_desc, peer_addr,
			   kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief A small write whose desc reports HMEM does NOT use inline: is_hmem
 * forces use_inline false even though the 32-byte message fits inject_rma_size
 * (64). The post then fires with use_inline false and the SGE list (not the
 * inline_data_list) built from the caller's iov.
 */
TEST_F(EfaRmaWriteInlineTest, write_hmem_small_not_inline)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(32);
	efa_test_mr_set_iface_cuda(local_mr);

	/* Because use_inline is false, the SGE/desc path runs and builds
	 * sge_list[0] from the iov; assert it points at local_buf. */
	auto sge_is_local = Truly([=](const struct ibv_sge *sge) {
		return sge[0].addr == (uint64_t) local_buf &&
		       sge[0].length == 32;
	});

	EXPECT_CALL(mock_efa,
		    efa_qp_post_write(_, sge_is_local, 1, _, false, kRemoteKey,
				      kRemoteAddr, 0, 0, _, _, _, _))
		.WillOnce(Return(0));

	int ret = fi_write(resource.ep, local_buf, 32, local_desc, peer_addr,
			   kRemoteAddr, kRemoteKey, NULL);
	EXPECT_EQ(ret, 0);
}

/**
 * @brief FI_INJECT of an HMEM buffer that fits inject_rma_size fails with
 * -FI_ENOSYS: the len_fits_inline-but-is_hmem else arm of the FI_INJECT block,
 * distinct from the too-large arm in write_inject_too_large_returns_einval. No
 * post fires.
 */
TEST_F(EfaRmaWriteInlineTest, write_inject_hmem_returns_enosys)
{
	struct fi_info *hints = make_hints();
	ASSERT_NO_FATAL_FAILURE(construct(hints));
	reg_buffer(32);
	efa_test_mr_set_iface_cuda(local_mr);

	struct iovec iov = {local_buf, 32};
	struct fi_rma_iov rma_iov = {kRemoteAddr, 32, kRemoteKey};
	struct fi_msg_rma msg = {};
	msg.msg_iov = &iov;
	msg.desc = &local_desc;
	msg.iov_count = 1;
	msg.addr = peer_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	EXPECT_CALL(mock_efa, efa_qp_post_write).Times(0);

	int ret = fi_writemsg(resource.ep, &msg, FI_INJECT);
	EXPECT_EQ(ret, -FI_ENOSYS);
}
