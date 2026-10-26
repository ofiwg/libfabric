/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_domain_utils.h"
#include "efa_gtest_wr_utils.h"
#include "fi_ext_efa.h"
#include "rdma/fi_ext.h"
#include <rdma/fi_rma.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_wr.h>
#include <cerrno>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using testing::_;
using testing::Bool;
using testing::Invoke;
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

	void construct(bool request_wr = false)
	{
		int ret;
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		if (request_wr) {
			hints->caps |= FI_WR;
			hints->mode &= ~FI_CONTEXT2;
		}

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

#if HAVE_EFA_DATA_PATH_DIRECT

class EfaWrTestBase : public EfaWrFlushTestBase
{
	protected:
	void setup()
	{
		EfaWrFlushTestBase::setup();
		if (efa_test_device_probe() || !efa_test_device_supports_fi_wr())
			GTEST_SKIP() << "device does not support FI_WR "
					"(no direct data path)";
	}

	uint8_t *wr() { return wr_buf.data(); }

	void fill_iovs(uint8_t *buf, void *desc, size_t count, size_t seg_len)
	{
		for (size_t i = 0; i < count; i++) {
			iovs[i].iov_base = buf + i * seg_len;
			iovs[i].iov_len = seg_len;
			descs[i] = desc;
		}
	}

	void prepare_send(const struct fi_wr_attr **attr_out, size_t *wr_len,
			  uint64_t flags = 0, size_t count = 1,
			  size_t seg_len = EFA_TEST_WR_SEG_LEN,
			  uint64_t data = 0)
	{
		fill_iovs(local_buf, local_desc, count, seg_len);

		memset(&op_msg, 0, sizeof(op_msg));
		op_msg.ep = resource.ep;
		op_msg.msg.msg_iov = iovs;
		op_msg.msg.desc = descs;
		op_msg.msg.iov_count = count;
		op_msg.msg.addr = peer_addr;
		op_msg.msg.data = data;
		op_msg.flags = flags;

		memset(&attr, 0, sizeof(attr));
		attr.op_type = FI_OP_SEND;
		attr.op.msg = &op_msg;

		*wr_len = efa_test_wr_tx_size();
		*attr_out = &attr;
		wr_buf.assign(*wr_len, 0);
	}

	void prepare_rma(enum fi_op_type op_type,
			 const struct fi_wr_attr **attr_out, size_t *wr_len)
	{
		fill_iovs(local_buf, local_desc, 1, EFA_TEST_WR_SEG_LEN);

		rma_iov.addr = (uint64_t) (local_buf + EFA_TEST_WR_SEG_LEN);
		rma_iov.len = EFA_TEST_WR_SEG_LEN;
		rma_iov.key = fi_mr_key(local_mr);

		memset(&op_rma, 0, sizeof(op_rma));
		op_rma.ep = resource.ep;
		op_rma.msg.msg_iov = iovs;
		op_rma.msg.desc = descs;
		op_rma.msg.iov_count = 1;
		op_rma.msg.addr = peer_addr;
		op_rma.msg.rma_iov = &rma_iov;
		op_rma.msg.rma_iov_count = 1;

		memset(&attr, 0, sizeof(attr));
		attr.op_type = op_type;
		attr.op.rma = &op_rma;

		*wr_len = efa_test_wr_tx_size();
		*attr_out = &attr;
		wr_buf.assign(*wr_len, 0);
	}

	void prepare_recv(const struct fi_wr_attr **attr_out, size_t *wr_len,
			  bool provide_desc = true, size_t count = 1)
	{
		fill_iovs(local_buf, local_desc, count, EFA_TEST_WR_SEG_LEN);

		memset(&op_msg, 0, sizeof(op_msg));
		op_msg.ep = resource.ep;
		op_msg.msg.msg_iov = iovs;
		op_msg.msg.desc = provide_desc ? descs : nullptr;
		op_msg.msg.iov_count = count;
		op_msg.msg.addr = peer_addr;

		memset(&attr, 0, sizeof(attr));
		attr.op_type = FI_OP_RECV;
		attr.op.msg = &op_msg;

		*wr_len = efa_test_wr_rx_size(count);
		*attr_out = &attr;
		wr_buf.assign(*wr_len, 0);
	}

	void alloc_alt_mr()
	{
		int ret;

		alt_buf = (uint8_t *) malloc(EFA_TEST_WR_SEG_LEN);
		ASSERT_NE(alt_buf, nullptr);
		memset(alt_buf, 0xa5, EFA_TEST_WR_SEG_LEN);

		ret = fi_mr_reg(resource.domain, alt_buf, EFA_TEST_WR_SEG_LEN,
				FI_SEND | FI_RECV, 0, 0, 0, &alt_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		alt_desc = fi_mr_desc(alt_mr);
	}

	void teardown()
	{
		if (alt_mr) {
			EXPECT_EQ(fi_close(&alt_mr->fid), 0);
			alt_mr = nullptr;
		}
		free(alt_buf);
		alt_buf = nullptr;

		EfaWrFlushTestBase::teardown();
	}

	std::vector<uint8_t> wr_buf;
	struct fi_wr_attr attr = {};
	struct fi_op_msg op_msg = {};
	struct fi_op_rma op_rma = {};
	struct iovec iovs[2] = {};
	void *descs[2] = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t *alt_buf = nullptr;
	struct fid_mr *alt_mr = nullptr;
	void *alt_desc = nullptr;
};

class EfaWrTest : public EfaWrTestBase, public Test
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

/**
 * @brief fi_wr_prepare formats a send into the caller's buffer, reports the
 * provider's work request size in wr_len, and stamps the SEND op type into the
 * work request.
 */
TEST_F(EfaWrTest, prepare_send_formats_send_wqe)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
	ASSERT_NE(efa_test_wr_tx_size(), 0u);
	ASSERT_LE(efa_test_wr_tx_size(), sizeof(wr));

	prepare_send(&attr, &wr_len);

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), 0);
	EXPECT_EQ(wr_len, efa_test_wr_tx_size());
	EXPECT_EQ(efa_test_wr_tx_op_type(wr), efa_test_wr_op_type_send());
}

/**
 * @brief A work request buffer smaller than the provider's work request size
 * is rejected with -FI_ETOOSMALL and wr_len is left reporting the required
 * size.
 */
TEST_F(EfaWrTest, prepare_rejects_too_small_buffer)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
	ASSERT_NE(efa_test_wr_tx_size(), 0u);

	prepare_send(&attr, &wr_len);
	wr_len = efa_test_wr_tx_size() - 1;

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len),
		  -FI_ETOOSMALL);
}

/**
 * @brief An op type the transmit path does not format (here an atomic) is
 * rejected with -FI_ENOSYS.
 */
TEST_F(EfaWrTest, prepare_rejects_unsupported_op_type)
{
	const struct fi_wr_attr *attr;
	struct fi_wr_attr bad;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	prepare_send(&attr, &wr_len);

	bad = *attr;
	bad.op_type = FI_OP_ATOMIC;

	EXPECT_EQ(fi_wr_prepare(resource.ep, &bad, wr, &wr_len), -FI_ENOSYS);
}

struct efa_test_wr_rma_op {
	enum fi_op_type op_type;
	const char *name;
};

class EfaWrRmaTest : public EfaWrTestBase,
		       public TestWithParam<efa_test_wr_rma_op>
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

/**
 * @brief fi_wr_prepare formats an RMA read/write and stamps the matching RDMA
 * op type into the work request. Needs a device that supports RDMA, since
 * efa-direct advertises FI_RMA only there.
 */
TEST_P(EfaWrRmaTest, prepare_rma_formats_rdma_op_type)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;
	int expected_op_type;

	if (!efa_test_device_supports_rma())
		GTEST_SKIP() << "device does not support RDMA read/write";

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	prepare_rma(GetParam().op_type, &attr, &wr_len);

	expected_op_type = GetParam().op_type == FI_OP_WRITE
				   ? efa_test_wr_op_type_rdma_write()
				   : efa_test_wr_op_type_rdma_read();

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), 0);
	EXPECT_EQ(wr_len, efa_test_wr_tx_size());
	EXPECT_EQ(efa_test_wr_tx_op_type(wr), expected_op_type);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrRmaTest,
	Values(efa_test_wr_rma_op{FI_OP_WRITE, "write"},
	       efa_test_wr_rma_op{FI_OP_READ, "read"}),
	[](const testing::TestParamInfo<efa_test_wr_rma_op> &info) {
		return std::string(info.param.name);
	});

/**
 * @brief fi_wr_queue_tx rejects a NULL work request with -FI_EINVAL before
 * touching the send queue, so no transmit work is left pending.
 */
TEST_F(EfaWrTest, queue_tx_rejects_null_wr)
{
	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	EXPECT_EQ(fi_wr_queue_tx(resource.ep, nullptr, nullptr), -FI_EINVAL);
	EXPECT_FALSE(efa_test_ep_tx_wr_pending(resource.ep));
}

/**
 * @brief Queueing a prepared send batches one entry on the send queue without
 * initiating it, so transmit work is left pending for a later flush. Needs the
 * direct data path, whose live send queue the queue call writes into.
 */
TEST_F(EfaWrTest, queue_tx_batches_prepared_send)
{
	const struct fi_wr_attr *attr;
	struct fi_context2 ctx = {};
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	if (!efa_test_ep_data_path_direct_enabled(resource.ep))
		GTEST_SKIP() << "endpoint does not use the direct data path";

	ASSERT_LE(efa_test_wr_tx_size(), sizeof(wr));
	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), 0);

	ASSERT_EQ(efa_test_ep_tx_num_wqe_pending(resource.ep), 0u);

	EXPECT_EQ(fi_wr_queue_tx(resource.ep, wr, &ctx), 0);
	EXPECT_EQ(efa_test_ep_tx_num_wqe_pending(resource.ep), 1u);
	EXPECT_TRUE(efa_test_ep_tx_wr_pending(resource.ep));
}

/**
 * @brief fi_wr_prepare formats a receive into the caller's buffer, reports the
 * receive work request size in wr_len, and marks the final descriptor with the
 * LAST bit that tells the device where the work request ends.
 */
TEST_F(EfaWrTest, prepare_recv_formats_rx_descs)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
	ASSERT_NE(efa_test_wr_rx_size(1), 0u);
	ASSERT_LE(efa_test_wr_rx_size(1), sizeof(wr));

	prepare_recv(&attr, &wr_len);

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), 0);
	EXPECT_EQ(wr_len, efa_test_wr_rx_size(1));
	EXPECT_EQ(efa_test_wr_rx_desc_is_last(wr, 0), 1);
}

/**
 * @brief A receive work request buffer smaller than the required size is
 * rejected with -FI_ETOOSMALL.
 */
TEST_F(EfaWrTest, prepare_recv_rejects_too_small_buffer)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
	ASSERT_NE(efa_test_wr_rx_size(1), 0u);

	prepare_recv(&attr, &wr_len);
	wr_len = efa_test_wr_rx_size(1) - 1;

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len),
		  -FI_ETOOSMALL);
}

/**
 * @brief efa-direct requires FI_MR_LOCAL, so a receive prepared without a
 * memory descriptor is rejected with -FI_EINVAL.
 */
TEST_F(EfaWrTest, prepare_recv_rejects_missing_desc)
{
	const struct fi_wr_attr *attr;
	uint8_t wr[256];
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	prepare_recv(&attr, &wr_len, false);

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), -FI_EINVAL);
}

/**
 * @brief fi_wr_queue_recv rejects a NULL work request with -FI_EINVAL before
 * touching the receive queue, so no receive is posted.
 */
TEST_F(EfaWrTest, queue_recv_rejects_null_wr)
{
	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	EXPECT_EQ(fi_wr_queue_recv(resource.ep, nullptr, nullptr), -FI_EINVAL);
}

/**
 * @brief Queueing a prepared receive posts one work request to the receive
 * queue, advancing its posted count. Needs the direct data path, whose live
 * receive queue the queue call writes into.
 */
TEST_F(EfaWrTest, queue_recv_posts_prepared_recv)
{
	const struct fi_wr_attr *attr;
	struct fi_context2 ctx = {};
	uint8_t wr[256];
	size_t wr_len;
	unsigned int posted_before;

	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	if (!efa_test_ep_data_path_direct_enabled(resource.ep))
		GTEST_SKIP() << "endpoint does not use the direct data path";

	ASSERT_LE(efa_test_wr_rx_size(1), sizeof(wr));
	prepare_recv(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr, &wr_len), 0);

	posted_before = efa_test_ep_rx_wqe_posted(resource.ep);

	EXPECT_EQ(fi_wr_queue_recv(resource.ep, wr, &ctx), 0);
	EXPECT_EQ(efa_test_ep_rx_wqe_posted(resource.ep), posted_before + 1);
}

/**
 * @brief efa-direct has no tagged receive queue, so fi_wr_queue_trecv is not
 * supported and returns -FI_ENOSYS.
 */
TEST_F(EfaWrTest, queue_trecv_not_supported)
{
	ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));

	EXPECT_EQ(fi_wr_queue_trecv(resource.ep, nullptr, nullptr),
		  -FI_ENOSYS);
}

class EfaWrModifyTest : public EfaWrTestBase, public Test
{
	protected:
	void SetUp() override
	{
		setup();
		if (IsSkipped())
			return;
		ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
		ASSERT_NO_FATAL_FAILURE(alloc_alt_mr());
	}
	void TearDown() override { teardown(); }
};

TEST_F(EfaWrModifyTest, modify_addr_rewrites_tx_ud_addr)
{
	const struct fi_wr_attr *attr;
	fi_addr_t new_addr = FI_ADDR_NOTAVAIL;
	uint16_t ahn, qpn, want_ahn, want_qpn, self_ahn;
	uint32_t qkey, want_qkey;
	size_t wr_len;
	static struct ibv_ah dummy_ibv_ah;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(efa_test_av_ud_addr(resource.ep, peer_addr, &self_ahn,
				      nullptr, nullptr),
		  0);
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah, _, _)
		.WillOnce(Return(&dummy_ibv_ah));
	EFA_EXPECT_CALL(mock_efa, efadv_query_ah, &dummy_ibv_ah, _, _)
		.WillOnce(Invoke([self_ahn](struct ibv_ah *, struct efadv_ah_attr *attr,
					    uint32_t) {
			attr->ahn = (uint16_t) (self_ahn + 1);
			return 0;
		}));
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ibv_ah)
		.WillOnce(Return(0));

	ASSERT_EQ(efa_test_av_insert_fake_gid(resource.ep, resource.av,
					      &new_addr),
		  1);
	ASSERT_EQ(efa_test_av_ud_addr(resource.ep, new_addr, &want_ahn,
				      &want_qpn, &want_qkey),
		  0);

	efa_test_wr_tx_ud_addr(wr(), &ahn, &qpn, &qkey);
	ASSERT_NE(ahn, want_ahn);
	ASSERT_NE(qpn, want_qpn);
	ASSERT_NE(qkey, want_qkey);

	EXPECT_EQ(fi_wr_modify_addr(resource.ep, wr(), FI_OP_SEND, new_addr),
		  0);

	efa_test_wr_tx_ud_addr(wr(), &ahn, &qpn, &qkey);
	EXPECT_EQ(ahn, want_ahn);
	EXPECT_EQ(qpn, want_qpn);
	EXPECT_EQ(qkey, want_qkey);
	EXPECT_EQ(efa_test_wr_tx_op_type(wr()), efa_test_wr_op_type_send());
	EXPECT_EQ(fi_av_remove(resource.av, &new_addr, 1, 0), 0);
}

TEST_F(EfaWrModifyTest, modify_addr_rejects_unspec_on_tx)
{
	const struct fi_wr_attr *attr;
	uint16_t ahn, qpn, want_ahn, want_qpn;
	uint32_t qkey, want_qkey;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(efa_test_av_ud_addr(resource.ep, peer_addr, &want_ahn,
				      &want_qpn, &want_qkey),
		  0);

	EXPECT_EQ(fi_wr_modify_addr(resource.ep, wr(), FI_OP_SEND,
				    FI_ADDR_UNSPEC),
		  -FI_EINVAL);

	efa_test_wr_tx_ud_addr(wr(), &ahn, &qpn, &qkey);
	EXPECT_EQ(ahn, want_ahn);
	EXPECT_EQ(qpn, want_qpn);
	EXPECT_EQ(qkey, want_qkey);
}

TEST_F(EfaWrModifyTest, modify_addr_rejects_addr_not_in_av)
{
	const struct fi_wr_attr *attr;
	fi_addr_t addr_not_in_av = 4096;
	uint16_t ahn, qpn, want_ahn, want_qpn;
	uint32_t qkey, want_qkey;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(efa_test_av_ud_addr(resource.ep, peer_addr, &want_ahn,
				      &want_qpn, &want_qkey),
		  0);
	ASSERT_EQ(efa_test_av_ud_addr(resource.ep, addr_not_in_av, nullptr,
				      nullptr, nullptr),
		  -1);

	EXPECT_EQ(fi_wr_modify_addr(resource.ep, wr(), FI_OP_SEND,
				    addr_not_in_av),
		  -FI_EINVAL);

	efa_test_wr_tx_ud_addr(wr(), &ahn, &qpn, &qkey);
	EXPECT_EQ(ahn, want_ahn);
	EXPECT_EQ(qpn, want_qpn);
	EXPECT_EQ(qkey, want_qkey);
}

TEST_F(EfaWrModifyTest, modify_iov_replaces_send_sgl)
{
	const struct fi_wr_attr *attr;
	struct iovec new_iov;
	void *new_desc = alt_desc;
	uint64_t addr;
	uint32_t len, lkey;
	size_t wr_len;

	ASSERT_GT((size_t) (EFA_TEST_WR_SEG_LEN / 2),
		  efa_test_device_inline_buf_size());

	prepare_send(&attr, &wr_len, 0, /*count=*/2,
		     EFA_TEST_WR_SEG_LEN / 2);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_FALSE(efa_test_wr_tx_is_inline(wr()));
	ASSERT_EQ(efa_test_wr_tx_length(wr()), 2);

	new_iov.iov_base = alt_buf;
	new_iov.iov_len = EFA_TEST_WR_SEG_LEN;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_SEND, &new_iov,
				   &new_desc, 1),
		  0);

	EXPECT_FALSE(efa_test_wr_tx_is_inline(wr()));
	EXPECT_EQ(efa_test_wr_tx_length(wr()), 1);

	efa_test_wr_tx_sge(wr(), FI_OP_SEND, 0, &addr, &len, &lkey);
	EXPECT_EQ(addr, (uint64_t) alt_buf);
	EXPECT_EQ(len, (uint32_t) EFA_TEST_WR_SEG_LEN);
	EXPECT_EQ(lkey, efa_test_mr_desc_lkey(alt_desc));
	EXPECT_NE(lkey, efa_test_mr_desc_lkey(local_desc));
}

TEST_F(EfaWrModifyTest, modify_iov_replaces_send_inline_data)
{
	const struct fi_wr_attr *attr;
	struct iovec new_iov;
	void *new_desc = alt_desc;
	size_t prepared_len = 8, new_len = 16;
	size_t wr_len;

	if (efa_test_device_inline_buf_size() < new_len)
		GTEST_SKIP() << "device inline buffer is too small";

	prepare_send(&attr, &wr_len, 0, /*count=*/1, prepared_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_TRUE(efa_test_wr_tx_is_inline(wr()));
	ASSERT_EQ(efa_test_wr_tx_length(wr()), prepared_len);

	new_iov.iov_base = alt_buf;
	new_iov.iov_len = new_len;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_SEND, &new_iov,
				   &new_desc, 1),
		  0);

	EXPECT_TRUE(efa_test_wr_tx_is_inline(wr()));
	EXPECT_EQ(efa_test_wr_tx_length(wr()), (uint16_t) new_len);
	EXPECT_EQ(memcmp(efa_test_wr_tx_inline_data(wr(), FI_OP_SEND), alt_buf,
			 new_len),
		  0);
}

/**
 * @brief An iov too long for the device to take inline re-formats the work
 * request to carry buffer descriptors rather than refusing the modification,
 * since inline is a choice the provider made at prepare and not a property the
 * application asked of this work request.
 */
TEST_F(EfaWrModifyTest, modify_iov_converts_inline_to_sgl)
{
	const struct fi_wr_attr *attr;
	struct iovec new_iov;
	void *new_desc = alt_desc;
	size_t prepared_len = 8;
	uint64_t addr;
	uint32_t len, lkey;
	size_t wr_len;

	ASSERT_GT((size_t) EFA_TEST_WR_SEG_LEN,
		  efa_test_device_inline_buf_size());

	prepare_send(&attr, &wr_len, 0, /*count=*/1, prepared_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_TRUE(efa_test_wr_tx_is_inline(wr()));

	new_iov.iov_base = alt_buf;
	new_iov.iov_len = EFA_TEST_WR_SEG_LEN;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_SEND, &new_iov,
				   &new_desc, 1),
		  0);

	EXPECT_FALSE(efa_test_wr_tx_is_inline(wr()));
	/* an sgl work request carries its sge count, not a byte count */
	EXPECT_EQ(efa_test_wr_tx_length(wr()), 1);
	efa_test_wr_tx_sge(wr(), FI_OP_SEND, 0, &addr, &len, &lkey);
	EXPECT_EQ(addr, (uint64_t) alt_buf);
	EXPECT_EQ(len, (uint32_t) EFA_TEST_WR_SEG_LEN);
	EXPECT_EQ(lkey, efa_test_mr_desc_lkey(alt_desc));
}

TEST_F(EfaWrModifyTest, modify_iov_rebuilds_recv_descs)
{
	const struct fi_wr_attr *attr;
	struct iovec new_iovs[2];
	void *new_descs[2] = {alt_desc, alt_desc};
	uint64_t addr;
	uint16_t len;
	uint32_t lkey;
	size_t wr_len;

	if (resource.info->rx_attr->iov_limit < 2u)
		GTEST_SKIP() << "device takes a single receive segment";

	prepare_recv(&attr, &wr_len, /*provide_desc=*/true, /*count=*/2);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	new_iovs[0].iov_base = alt_buf;
	new_iovs[0].iov_len = 64;
	new_iovs[1].iov_base = alt_buf + 64;
	new_iovs[1].iov_len = 128;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_RECV, new_iovs,
				   new_descs, 2),
		  0);

	efa_test_wr_rx_desc(wr(), 0, &addr, &len, &lkey);
	EXPECT_EQ(addr, (uint64_t) alt_buf);
	EXPECT_EQ(len, 64);
	EXPECT_EQ(lkey, efa_test_mr_desc_lkey(alt_desc));
	EXPECT_EQ(efa_test_wr_rx_desc_is_last(wr(), 0), 0);

	efa_test_wr_rx_desc(wr(), 1, &addr, &len, &lkey);
	EXPECT_EQ(addr, (uint64_t) (alt_buf + 64));
	EXPECT_EQ(len, 128);
	EXPECT_EQ(lkey, efa_test_mr_desc_lkey(alt_desc));
	EXPECT_EQ(efa_test_wr_rx_desc_is_last(wr(), 1), 1);
}

TEST_F(EfaWrModifyTest, modify_iov_recv_rejects_missing_desc)
{
	const struct fi_wr_attr *attr;
	struct iovec new_iov;
	void *new_desc = nullptr;
	uint64_t addr;
	size_t wr_len;

	prepare_recv(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	new_iov.iov_base = alt_buf;
	new_iov.iov_len = 64;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_RECV, &new_iov,
				   &new_desc, 1),
		  -FI_EINVAL);

	efa_test_wr_rx_desc(wr(), 0, &addr, nullptr, nullptr);
	EXPECT_EQ(addr, (uint64_t) local_buf);
}

TEST_F(EfaWrModifyTest, modify_rma_iov_rejects_non_rma_op)
{
	const struct fi_wr_attr *attr;
	struct fi_rma_iov new_rma_iov;
	std::vector<uint8_t> prepared;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	new_rma_iov.addr = (uint64_t) alt_buf;
	new_rma_iov.len = EFA_TEST_WR_SEG_LEN;
	new_rma_iov.key = fi_mr_key(alt_mr);

	EXPECT_EQ(fi_wr_modify_rma_iov(resource.ep, wr(), FI_OP_SEND,
				       &new_rma_iov, 1),
		  -FI_EINVAL);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

class EfaWrModifyRmaTest : public EfaWrTestBase,
			   public TestWithParam<efa_test_wr_rma_op>
{
	protected:
	void SetUp() override
	{
		setup();
		if (IsSkipped())
			return;
		if (!efa_test_device_supports_rma())
			GTEST_SKIP() << "device does not support RDMA "
					"read/write";
		ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
		ASSERT_NO_FATAL_FAILURE(alloc_alt_mr());
	}
	void TearDown() override { teardown(); }

	int expected_op_type()
	{
		return GetParam().op_type == FI_OP_WRITE ?
			       efa_test_wr_op_type_rdma_write() :
			       efa_test_wr_op_type_rdma_read();
	}
};

TEST_P(EfaWrModifyRmaTest, modify_iov_moves_rdma_length)
{
	const struct fi_wr_attr *attr;
	enum fi_op_type op_type = GetParam().op_type;
	struct iovec new_iov;
	void *new_desc = alt_desc;
	uint64_t addr;
	uint32_t len, lkey, rdma_len;
	size_t wr_len;

	prepare_rma(op_type, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_FALSE(efa_test_wr_tx_is_inline(wr()));
	efa_test_wr_tx_rdma_addr(wr(), nullptr, nullptr, &rdma_len);
	ASSERT_EQ(rdma_len, (uint32_t) EFA_TEST_WR_SEG_LEN);

	new_iov.iov_base = alt_buf;
	new_iov.iov_len = EFA_TEST_WR_SEG_LEN / 2;

	EXPECT_EQ(fi_wr_modify_iov(resource.ep, wr(), op_type, &new_iov,
				   &new_desc, 1),
		  0);

	efa_test_wr_tx_sge(wr(), op_type, 0, &addr, &len, &lkey);
	EXPECT_EQ(addr, (uint64_t) alt_buf);
	EXPECT_EQ(len, (uint32_t) (EFA_TEST_WR_SEG_LEN / 2));
	EXPECT_EQ(lkey, efa_test_mr_desc_lkey(alt_desc));
	EXPECT_EQ(efa_test_wr_tx_length(wr()), 1);

	efa_test_wr_tx_rdma_addr(wr(), nullptr, nullptr, &rdma_len);
	EXPECT_EQ(rdma_len, (uint32_t) (EFA_TEST_WR_SEG_LEN / 2));
	EXPECT_EQ(efa_test_wr_tx_op_type(wr()), expected_op_type());
}

TEST_P(EfaWrModifyRmaTest, modify_rma_iov_replaces_remote_buf)
{
	const struct fi_wr_attr *attr;
	enum fi_op_type op_type = GetParam().op_type;
	struct fi_rma_iov new_rma_iov;
	uint64_t addr;
	uint32_t rkey, len;
	size_t wr_len;

	prepare_rma(op_type, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	new_rma_iov.addr = (uint64_t) alt_buf;
	new_rma_iov.len = EFA_TEST_WR_SEG_LEN;
	new_rma_iov.key = fi_mr_key(alt_mr);
	ASSERT_NE(new_rma_iov.key, fi_mr_key(local_mr));

	EXPECT_EQ(fi_wr_modify_rma_iov(resource.ep, wr(), op_type, &new_rma_iov,
				       1),
		  0);

	efa_test_wr_tx_rdma_addr(wr(), &addr, &rkey, &len);
	EXPECT_EQ(addr, (uint64_t) alt_buf);
	EXPECT_EQ(rkey, (uint32_t) fi_mr_key(alt_mr));
	EXPECT_EQ(len, (uint32_t) EFA_TEST_WR_SEG_LEN);
	EXPECT_EQ(efa_test_wr_tx_op_type(wr()), expected_op_type());
}

TEST_P(EfaWrModifyRmaTest, modify_rma_iov_rejects_length_mismatch)
{
	const struct fi_wr_attr *attr;
	enum fi_op_type op_type = GetParam().op_type;
	struct fi_rma_iov new_rma_iov;
	uint64_t addr;
	uint32_t rkey, len;
	size_t wr_len;

	prepare_rma(op_type, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	new_rma_iov.addr = (uint64_t) alt_buf;
	new_rma_iov.len = EFA_TEST_WR_SEG_LEN / 2;
	new_rma_iov.key = fi_mr_key(alt_mr);

	EXPECT_EQ(fi_wr_modify_rma_iov(resource.ep, wr(), op_type, &new_rma_iov,
				       1),
		  -FI_EINVAL);

	efa_test_wr_tx_rdma_addr(wr(), &addr, &rkey, &len);
	EXPECT_EQ(addr, (uint64_t) (local_buf + EFA_TEST_WR_SEG_LEN));
	EXPECT_EQ(rkey, (uint32_t) fi_mr_key(local_mr));
	EXPECT_EQ(len, (uint32_t) EFA_TEST_WR_SEG_LEN);
	EXPECT_EQ(efa_test_wr_tx_op_type(wr()), expected_op_type());
}

TEST_P(EfaWrModifyRmaTest, modify_rma_iov_rejects_count_other_than_one)
{
	const struct fi_wr_attr *attr;
	enum fi_op_type op_type = GetParam().op_type;
	struct fi_rma_iov new_rma_iov[2];
	uint64_t addr;
	uint32_t rkey;
	size_t wr_len;

	prepare_rma(op_type, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	for (size_t i = 0; i < 2; i++) {
		new_rma_iov[i].addr = (uint64_t) alt_buf;
		new_rma_iov[i].len = EFA_TEST_WR_SEG_LEN;
		new_rma_iov[i].key = fi_mr_key(alt_mr);
	}

	for (size_t count : {(size_t) 0, (size_t) 2}) {
		EXPECT_EQ(fi_wr_modify_rma_iov(resource.ep, wr(), op_type,
					       new_rma_iov, count),
			  -FI_EINVAL)
			<< "count " << count;

		efa_test_wr_tx_rdma_addr(wr(), &addr, &rkey, nullptr);
		EXPECT_EQ(addr, (uint64_t) (local_buf + EFA_TEST_WR_SEG_LEN));
		EXPECT_EQ(rkey, (uint32_t) fi_mr_key(local_mr));
		EXPECT_EQ(efa_test_wr_tx_op_type(wr()), expected_op_type());
	}
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrModifyRmaTest,
	Values(efa_test_wr_rma_op{FI_OP_WRITE, "write"},
	       efa_test_wr_rma_op{FI_OP_READ, "read"}),
	[](const testing::TestParamInfo<efa_test_wr_rma_op> &info) {
		return std::string(info.param.name);
	});

TEST_F(EfaWrModifyTest, modify_tag_not_supported)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	EXPECT_EQ(fi_wr_modify_tag(resource.ep, wr(), FI_OP_SEND, 0x1, 0),
		  -FI_ENOSYS);
}

TEST_F(EfaWrModifyTest, modify_data_replaces_imm_data)
{
	const struct fi_wr_attr *attr;
	uint32_t imm_data = 0x1234abcd, new_imm_data = 0xfeed5678;
	size_t wr_len;

	prepare_send(&attr, &wr_len, FI_REMOTE_CQ_DATA, /*count=*/1,
		     EFA_TEST_WR_SEG_LEN, imm_data);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_TRUE(efa_test_wr_tx_has_imm(wr()));
	ASSERT_EQ(efa_test_wr_tx_imm_data(wr()), imm_data);

	EXPECT_EQ(fi_wr_modify_data(resource.ep, wr(), FI_OP_SEND,
				    new_imm_data),
		  0);

	EXPECT_EQ(efa_test_wr_tx_imm_data(wr()), new_imm_data);
	EXPECT_TRUE(efa_test_wr_tx_has_imm(wr()));
}

TEST_F(EfaWrModifyTest, modify_data_rejects_wr_without_remote_cq_data)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_FALSE(efa_test_wr_tx_has_imm(wr()));

	EXPECT_EQ(fi_wr_modify_data(resource.ep, wr(), FI_OP_SEND, 0x1234abcd),
		  -FI_EINVAL);

	EXPECT_FALSE(efa_test_wr_tx_has_imm(wr()));
	EXPECT_EQ(efa_test_wr_tx_imm_data(wr()), 0u);
}

TEST_F(EfaWrModifyTest, modify_data_rejects_recv_op)
{
	const struct fi_wr_attr *attr;
	std::vector<uint8_t> prepared;
	size_t wr_len;

	prepare_recv(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	EXPECT_EQ(fi_wr_modify_data(resource.ep, wr(), FI_OP_RECV, 0x1234abcd),
		  -FI_EINVAL);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

class EfaWrModifyFlagsImmTest : public EfaWrTestBase,
				public TestWithParam<bool>
{
	protected:
	void SetUp() override
	{
		setup();
		if (IsSkipped())
			return;
		ASSERT_NO_FATAL_FAILURE(construct(/*request_wr=*/true));
	}
	void TearDown() override { teardown(); }
};

TEST_P(EfaWrModifyFlagsImmTest, modify_flags_cannot_change_remote_cq_data)
{
	const struct fi_wr_attr *attr;
	bool with_imm = GetParam();
	uint64_t flags = with_imm ? FI_REMOTE_CQ_DATA : 0;
	size_t wr_len;

	prepare_send(&attr, &wr_len, flags, /*count=*/1, EFA_TEST_WR_SEG_LEN,
		     0x1234abcd);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(!!efa_test_wr_tx_has_imm(wr()), with_imm);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND,
				     with_imm ? 0 : FI_REMOTE_CQ_DATA),
		  -FI_EINVAL);
	EXPECT_EQ(!!efa_test_wr_tx_has_imm(wr()), with_imm);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND,
				     with_imm ? FI_REMOTE_CQ_DATA : 0),
		  0);
	EXPECT_EQ(!!efa_test_wr_tx_has_imm(wr()), with_imm);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrModifyFlagsImmTest, Bool(),
	[](const testing::TestParamInfo<bool> &info) {
		return std::string(info.param ? "with_remote_cq_data" :
						"without_remote_cq_data");
	});

TEST_F(EfaWrModifyTest, modify_flags_rejects_high_pps_on_send)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND,
				     FI_EFA_WR_HIGH_PPS),
		  -FI_EINVAL);
	EXPECT_EQ(efa_test_wr_tx_processing_hints(wr()), 0);
}

TEST_F(EfaWrModifyTest, modify_flags_on_recv_is_noop)
{
	const struct fi_wr_attr *attr;
	std::vector<uint8_t> prepared;
	size_t wr_len;

	prepare_recv(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_RECV,
				     FI_REMOTE_CQ_DATA),
		  0);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

TEST_F(EfaWrModifyTest, modify_flags_sets_and_clears_high_pps)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	if (efa_test_device_probe() ||
	    !efa_test_device_supports_rma())
		GTEST_SKIP() << "device does not support RDMA read/write";

	prepare_rma(FI_OP_WRITE, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(efa_test_wr_tx_processing_hints(wr()), 0);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_WRITE,
				     FI_EFA_WR_HIGH_PPS),
		  0);
	EXPECT_EQ(efa_test_wr_tx_processing_hints(wr()),
		  efa_test_wr_processing_hint_high_pps());

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_WRITE, 0), 0);
	EXPECT_EQ(efa_test_wr_tx_processing_hints(wr()), 0);
}

/**
 * @brief A flag the endpoint did not advertise is rejected rather than dropped,
 * so an application cannot transmit believing it got a semantic the device was
 * never told about.
 */
TEST_F(EfaWrModifyTest, modify_flags_rejects_unadvertised_flag)
{
	const struct fi_wr_attr *attr;
	std::vector<uint8_t> prepared;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND, FI_FENCE),
		  -FI_EBADFLAGS);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

/**
 * @brief The unsupported flag check precedes the receive path, which otherwise
 * accepts any flags because a receive descriptor carries none of them.
 */
TEST_F(EfaWrModifyTest, modify_flags_rejects_unsupported_flag_on_recv)
{
	const struct fi_wr_attr *attr;
	std::vector<uint8_t> prepared;
	size_t wr_len;

	prepare_recv(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_RECV,
				     FI_MULTI_RECV),
		  -FI_EBADFLAGS);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

/**
 * @brief The completion semantics the endpoint advertises, and FI_COMPLETION,
 * are accepted and change nothing: the device has one completion behavior and
 * the work request always requests a completion. An application that passes
 * its full flag set on each call is therefore not forced to strip them.
 */
TEST_F(EfaWrModifyTest, modify_flags_accepts_advertised_completion_semantics)
{
	const struct fi_wr_attr *attr;
	std::vector<uint8_t> prepared;
	size_t wr_len;
	/* the layout flags are covered on their own, below */
	uint64_t flags = FI_COMPLETION |
			 (resource.info->tx_attr->op_flags &
			  ~(FI_INJECT | FI_REMOTE_CQ_DATA));

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	prepared = wr_buf;

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND, flags), 0);
	EXPECT_EQ(memcmp(wr(), prepared.data(), wr_len), 0);
}

/**
 * @brief FI_INJECT cannot be added to a work request whose buffers were
 * formatted as a scatter-gather list, since the data would have to move into
 * the descriptor itself.
 */
TEST_F(EfaWrModifyTest, modify_flags_cannot_add_inject_to_sgl_wr)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_FALSE(efa_test_wr_tx_is_inline(wr()));

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND, FI_INJECT),
		  -FI_EINVAL);
	EXPECT_FALSE(efa_test_wr_tx_is_inline(wr()));
}

class EfaWrInlineWriteTest : public EfaWrTestBase, public Test
{
	protected:
	void SetUp() override
	{
		setup();
		if (testing::Test::IsSkipped())
			return;
		if (!efa_test_device_supports_rma())
			GTEST_SKIP() << "device does not support RDMA read/write";
		if (!efa_test_device_supports_wide_wqe())
			GTEST_SKIP() << "device does not support wide send queue "
					"entries";
	}

	void TearDown() override { teardown(); }

	/* An FI_WR endpoint whose QP is created for inline RDMA writes. */
	void construct_inline_write()
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		int ret;

		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_WR;
		hints->mode &= ~FI_CONTEXT2;

		efa_test_resource_getinfo(&resource, hints);
		ASSERT_NE(resource.info, nullptr);
		resource.info->tx_attr->inject_size =
			efa_test_device_inline_buf_size() + 1;
		efa_test_resource_open(&resource);
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_EQ(fi_enable(resource.ep), 0);

		local_buf = (uint8_t *) calloc(2 * EFA_TEST_WR_SEG_LEN, 1);
		ASSERT_NE(local_buf, nullptr);
		ret = fi_mr_reg(resource.domain, local_buf,
				2 * EFA_TEST_WR_SEG_LEN,
				FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0,
				&local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		MockEfa::set(&mock_efa);
	}
};

TEST_F(EfaWrInlineWriteTest, modify_iov_inline_to_sgl_keeps_remote_addr)
{
	const struct fi_wr_attr *attr;
	struct iovec big_iov;
	size_t wr_len;
	uint64_t addr;
	uint32_t rkey, len;

	ASSERT_NO_FATAL_FAILURE(construct_inline_write());
	prepare_rma(FI_OP_WRITE, &attr, &wr_len);
	iovs[0].iov_len = 8;
	rma_iov.len = 8;
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_TRUE(efa_test_wr_tx_is_inline(wr()));

	big_iov.iov_base = local_buf;
	big_iov.iov_len = EFA_TEST_WR_SEG_LEN;
	ASSERT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_WRITE, &big_iov,
				   descs, 1),
		  0);
	ASSERT_FALSE(efa_test_wr_tx_is_inline(wr()));

	efa_test_wr_tx_rdma_addr(wr(), &addr, &rkey, &len);
	EXPECT_EQ(addr, rma_iov.addr);
	EXPECT_EQ(rkey, (uint32_t) rma_iov.key);
	EXPECT_EQ(len, (uint32_t) EFA_TEST_WR_SEG_LEN);
}


class EfaWrCompActionTestBase : public EfaWrTestBase
{
	protected:
	int prev_enable_comp_action = 0;
	struct fi_efa_op_rma eop = {};

	void setup()
	{
		EfaWrTestBase::setup();
		if (testing::Test::IsSkipped())
			return;
		prev_enable_comp_action = efa_test_set_enable_comp_action(1);
		if (!efa_test_device_supports_rma())
			GTEST_SKIP() << "device does not support RDMA read/write";
		if (!efa_test_device_supports_comp_action())
			GTEST_SKIP() << "device without completion action support";
	}

	void teardown()
	{
		EfaWrTestBase::teardown();
		efa_test_set_enable_comp_action(prev_enable_comp_action);
	}

	void construct_actions(bool enable_actions, size_t inject_size = 0)
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		bool on = true;
		int ret;

		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_WR;
		hints->mode &= ~FI_CONTEXT2;

		efa_test_resource_getinfo(&resource, hints);
		ASSERT_NE(resource.info, nullptr);
		if (inject_size)
			resource.info->tx_attr->inject_size = inject_size;
		efa_test_resource_open(&resource);
		ASSERT_NE(resource.ep, nullptr);
		if (enable_actions)
			ASSERT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
					    FI_OPT_EFA_COMP_ACTION, &on,
					    sizeof(on)),
				  0);
		ASSERT_EQ(fi_enable(resource.ep), 0);
		ASSERT_EQ(efa_test_get_comp_action_enabled(resource.ep),
			  enable_actions);

		local_buf = (uint8_t *) calloc(2 * EFA_TEST_WR_SEG_LEN, 1);
		ASSERT_NE(local_buf, nullptr);
		ret = fi_mr_reg(resource.domain, local_buf,
				2 * EFA_TEST_WR_SEG_LEN,
				FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0, 0,
				&local_mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		local_desc = fi_mr_desc(local_mr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		MockEfa::set(&mock_efa);
	}

	void prepare_write_ext(const struct fi_wr_attr **attr_out,
			       size_t *wr_len, uint64_t feature_bits)
	{
		prepare_rma(FI_OP_WRITE, attr_out, wr_len);

		memset(&eop, 0, sizeof(eop));
		eop.op = op_rma;
		eop.op.flags = FI_EFA_EXTENDED_MSG;
		eop.feature_bits = feature_bits;
		eop.local_id = 7;
		eop.local_value = 0x1122334455667788ULL;
		eop.remote_id = 9;
		eop.remote_value = 0xa5;
		attr.op.rma = &eop.op;
	}
};

class EfaWrCompActionTest : public EfaWrCompActionTestBase, public Test
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

TEST_F(EfaWrCompActionTest, prepare_write_attaches_actions)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;
	int local_req, remote_req;
	uint32_t local_id, remote_id;
	uint64_t local_data, remote_data;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	prepare_write_ext(&attr, &wr_len,
			  FI_EFA_LOCAL_ACTION_ID | FI_EFA_LOCAL_ACTION_VALUE |
				  FI_EFA_REMOTE_ACTION_ID |
				  FI_EFA_REMOTE_ACTION_VALUE);

	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	EXPECT_EQ(efa_test_wr_tx_op_type(wr()),
		  efa_test_wr_op_type_rdma_write());

	efa_test_wr_tx_comp_actions(resource.ep, wr(), &local_req, &local_id,
				    &local_data, &remote_req, &remote_id,
				    &remote_data);
	EXPECT_TRUE(local_req);
	EXPECT_EQ(local_id, 7u);
	EXPECT_EQ(local_data, 0x1122334455667788ULL);
	EXPECT_TRUE(remote_req);
	EXPECT_EQ(remote_id, 9u);
	EXPECT_EQ(remote_data, 0xa5u);
}

TEST_F(EfaWrCompActionTest, prepare_write_attaches_remote_action_without_value)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;
	int local_req, remote_req;
	uint32_t local_id, remote_id;
	uint64_t local_data, remote_data;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	prepare_write_ext(&attr, &wr_len, FI_EFA_REMOTE_ACTION_ID);

	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	efa_test_wr_tx_comp_actions(resource.ep, wr(), &local_req, &local_id,
				    &local_data, &remote_req, &remote_id,
				    &remote_data);
	EXPECT_FALSE(local_req);
	EXPECT_TRUE(remote_req);
	EXPECT_EQ(remote_id, 9u);
	EXPECT_EQ(remote_data, 0u);
}

TEST_F(EfaWrCompActionTest, prepare_write_without_extended_msg_attaches_none)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;
	int local_req, remote_req;
	uint32_t local_id, remote_id;
	uint64_t local_data, remote_data;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	prepare_rma(FI_OP_WRITE, &attr, &wr_len);

	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	efa_test_wr_tx_comp_actions(resource.ep, wr(), &local_req, &local_id,
				    &local_data, &remote_req, &remote_id,
				    &remote_data);
	EXPECT_FALSE(local_req);
	EXPECT_FALSE(remote_req);
}

TEST_F(EfaWrCompActionTest, queue_tx_posts_prepared_actions)
{
	const struct fi_wr_attr *attr;
	struct fi_context2 ctx = {};
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	prepare_write_ext(&attr, &wr_len,
			  FI_EFA_LOCAL_ACTION_ID | FI_EFA_REMOTE_ACTION_ID);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	ASSERT_EQ(efa_test_ep_tx_num_wqe_pending(resource.ep), 0u);

	EXPECT_EQ(fi_wr_queue_tx(resource.ep, wr(), &ctx), 0);
	EXPECT_EQ(efa_test_ep_tx_num_wqe_pending(resource.ep), 1u);
}

TEST_F(EfaWrCompActionTest, modify_iov_inline_to_sgl_keeps_actions)
{
	const struct fi_wr_attr *attr;
	struct iovec big_iov;
	size_t wr_len;
	int local_req, remote_req;
	uint32_t local_id, remote_id;
	uint64_t local_data, remote_data;

	if (!efa_test_device_supports_wide_wqe())
		GTEST_SKIP() << "device does not support wide send queue entries";

	ASSERT_NO_FATAL_FAILURE(
		construct_actions(true, efa_test_device_inline_buf_size() + 1));
	prepare_write_ext(&attr, &wr_len,
			  FI_EFA_LOCAL_ACTION_ID | FI_EFA_LOCAL_ACTION_VALUE |
				  FI_EFA_REMOTE_ACTION_ID |
				  FI_EFA_REMOTE_ACTION_VALUE);
	iovs[0].iov_len = 8;
	rma_iov.len = 8;
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);
	if (!efa_test_wr_tx_is_inline(wr()))
		GTEST_SKIP() << "endpoint has no inline RDMA write room beside "
				"the action block";

	big_iov.iov_base = local_buf;
	big_iov.iov_len = EFA_TEST_WR_SEG_LEN;
	ASSERT_EQ(fi_wr_modify_iov(resource.ep, wr(), FI_OP_WRITE, &big_iov,
				   descs, 1),
		  0);
	ASSERT_FALSE(efa_test_wr_tx_is_inline(wr()));

	efa_test_wr_tx_comp_actions(resource.ep, wr(), &local_req, &local_id,
				    &local_data, &remote_req, &remote_id,
				    &remote_data);
	EXPECT_TRUE(local_req);
	EXPECT_EQ(local_id, 7u);
	EXPECT_EQ(local_data, 0x1122334455667788ULL);
	EXPECT_TRUE(remote_req);
	EXPECT_EQ(remote_id, 9u);
	EXPECT_EQ(remote_data, 0xa5u);
}

struct WrCompActionRejectCase {
	const char *name;
	bool enable_actions;
	uint64_t feature_bits;
	int expected;
};

class EfaWrCompActionRejectTest
	: public EfaWrCompActionTestBase,
	  public TestWithParam<WrCompActionRejectCase>
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

TEST_P(EfaWrCompActionRejectTest, prepare_write_rejects_bad_descriptor)
{
	const WrCompActionRejectCase &param = GetParam();
	const struct fi_wr_attr *attr;
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct_actions(param.enable_actions));
	prepare_write_ext(&attr, &wr_len, param.feature_bits);

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len),
		  param.expected);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrCompActionRejectTest,
	Values(WrCompActionRejectCase{"actions_not_enabled", false,
				      FI_EFA_LOCAL_ACTION_ID, -FI_EINVAL},
	       WrCompActionRejectCase{"unknown_feature_bits", true,
				      FI_EFA_LOCAL_ACTION_ID | (1ULL << 40),
				      -FI_EOPNOTSUPP},
	       WrCompActionRejectCase{"value_without_id", true,
				      FI_EFA_LOCAL_ACTION_VALUE, -FI_EINVAL}),
	[](const testing::TestParamInfo<WrCompActionRejectCase> &info) {
		return std::string(info.param.name);
	});

class EfaWrCompActionOtherOpTest
	: public EfaWrCompActionTestBase,
	  public TestWithParam<enum fi_op_type>
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

TEST_P(EfaWrCompActionOtherOpTest, prepare_rejects_extended_msg)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));

	switch (GetParam()) {
	case FI_OP_SEND:
		prepare_send(&attr, &wr_len, FI_EFA_EXTENDED_MSG);
		break;
	case FI_OP_RECV:
		prepare_recv(&attr, &wr_len);
		op_msg.flags = FI_EFA_EXTENDED_MSG;
		break;
	default:
		prepare_rma(GetParam(), &attr, &wr_len);
		op_rma.flags = FI_EFA_EXTENDED_MSG;
		break;
	}

	EXPECT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len),
		  -FI_EBADFLAGS);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrCompActionOtherOpTest, Values(FI_OP_SEND, FI_OP_RECV, FI_OP_READ),
	[](const testing::TestParamInfo<enum fi_op_type> &info) {
		switch (info.param) {
		case FI_OP_SEND:
			return std::string("send");
		case FI_OP_RECV:
			return std::string("recv");
		default:
			return std::string("read");
		}
	});

struct WrCompActionModifyCase {
	const char *name;
	bool prepared_with_actions;
	uint64_t flags;
	int expected;
};

class EfaWrCompActionModifyTest
	: public EfaWrCompActionTestBase,
	  public TestWithParam<WrCompActionModifyCase>
{
	protected:
	void SetUp() override { setup(); }
	void TearDown() override { teardown(); }
};

TEST_P(EfaWrCompActionModifyTest, modify_flags_keeps_extended_msg_layout)
{
	const WrCompActionModifyCase &param = GetParam();
	const struct fi_wr_attr *attr;
	size_t wr_len;
	int local_req, remote_req;
	uint32_t local_id, remote_id;
	uint64_t local_data, remote_data;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	if (param.prepared_with_actions)
		prepare_write_ext(&attr, &wr_len, FI_EFA_LOCAL_ACTION_ID);
	else
		prepare_rma(FI_OP_WRITE, &attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_WRITE,
				     param.flags),
		  param.expected);

	efa_test_wr_tx_comp_actions(resource.ep, wr(), &local_req, &local_id,
				    &local_data, &remote_req, &remote_id,
				    &remote_data);
	EXPECT_EQ(!!local_req, param.prepared_with_actions);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaWrCompActionModifyTest,
	Values(WrCompActionModifyCase{"keep_with_actions", true,
				      FI_EFA_EXTENDED_MSG | FI_COMPLETION, 0},
	       WrCompActionModifyCase{"drop_with_actions", true, FI_COMPLETION,
				      -FI_EINVAL},
	       WrCompActionModifyCase{"add_without_actions", false,
				      FI_EFA_EXTENDED_MSG, -FI_EINVAL},
	       WrCompActionModifyCase{"absent_without_actions", false,
				      FI_COMPLETION, 0}),
	[](const testing::TestParamInfo<WrCompActionModifyCase> &info) {
		return std::string(info.param.name);
	});

TEST_F(EfaWrCompActionTest, modify_flags_rejects_extended_msg_on_send)
{
	const struct fi_wr_attr *attr;
	size_t wr_len;

	ASSERT_NO_FATAL_FAILURE(construct_actions(true));
	prepare_send(&attr, &wr_len);
	ASSERT_EQ(fi_wr_prepare(resource.ep, attr, wr(), &wr_len), 0);

	EXPECT_EQ(fi_wr_modify_flags(resource.ep, wr(), FI_OP_SEND,
				     FI_EFA_EXTENDED_MSG),
		  -FI_EINVAL);
}
#endif /* HAVE_EFA_DATA_PATH_DIRECT */
