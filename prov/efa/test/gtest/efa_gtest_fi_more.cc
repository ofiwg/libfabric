/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/*
 * FI_MORE on the transmit path: a work request must be staged without notifying
 * the device, and a later post without FI_MORE must submit the whole batch.
 *
 * Both arms of the efa_qp_post_* dispatch implement that same contract, so the
 * tests are written once and parameterized over the backend. Only where the
 * submission is observed differs: the direct data path rings the send queue
 * doorbell, the rdma-core path calls ibv_wr_complete, and efa_test_dp_probe
 * hides which of the two it is watching.
 *
 * These are the straight-through cases. EfaRdmOpeQueuedFiMoreTest in
 * efa_gtest_rdm_ope.cc covers the different claim that an op queued in software
 * must not carry a stale FI_MORE when it is later reposted.
 */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_fi_more_helpers.h"
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

using testing::Combine;
using testing::TestWithParam;
using testing::Values;

static constexpr size_t kBufSize = 4096;

/**
 * @brief efa_test_device_supports_rma reads the selected device list, which the
 * provider only populates on the first fi_getinfo. Probe with a throwaway one
 * so the answer does not depend on whether an earlier test happened to run.
 */
static bool device_supports_rma()
{
	if (efa_test_device_supports_rma())
		return true;

	struct fi_info *hints = efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	struct fi_info *info = nullptr;

	if (hints) {
		fi_getinfo(FI_VERSION(2, 0), nullptr, nullptr, 0, hints, &info);
		fi_freeinfo(info);
		fi_freeinfo(hints);
	}
	return efa_test_device_supports_rma();
}

using EfaFiMoreParam = std::tuple<int, int>;

static std::string efa_test_param_name(
	const testing::TestParamInfo<EfaFiMoreParam> &info)
{
	std::string name;

	switch (std::get<0>(info.param)) {
	case EFA_TEST_POST_SEND:
		name = "send";
		break;
	case EFA_TEST_POST_READ:
		name = "read";
		break;
	default:
		name = "write";
		break;
	}
	return name + (std::get<1>(info.param) == EFA_TEST_DP_DIRECT ?
			       "_direct" : "_rdma_core");
}

/**
 * @brief An efa-direct endpoint whose selected backend has had its device sink
 * redirected, so a post can be observed without reaching the hardware.
 *
 * efa-direct passes the caller's flags through to the QP post unchanged, which
 * is what makes it the right fabric for testing the submission decision itself.
 */
class EfaFiMoreTest : public TestWithParam<EfaFiMoreParam>
{
	protected:
	struct efa_resource resource = {};
	struct efa_test_dp_probe probe = {};
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	uint8_t *buf = nullptr;
	struct fid_mr *mr = nullptr;
	void *desc = nullptr;
	int prev_track_mr = 0;

	int op() const { return std::get<0>(GetParam()); }
	int backend() const { return std::get<1>(GetParam()); }

	void SetUp() override
	{
		if (op() != EFA_TEST_POST_SEND && !device_supports_rma())
			GTEST_SKIP() << "device does not support RDMA read+write";

		memset(&resource, 0, sizeof(resource));
		/* Without MR tracking a post allocates no direct ope, so nothing
		 * is left outstanding for a completion nothing here reaps. */
		prev_track_mr = efa_test_set_track_mr(0);

		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		if (op() != EFA_TEST_POST_SEND) {
			/* FI_RX_CQ_DATA is required alongside FI_RMA when the
			 * device lacks unsolicited-write-recv support. */
			hints->caps |= FI_RMA;
			hints->mode |= FI_RX_CQ_DATA;
		}

		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));
		ASSERT_NE(resource.ep, nullptr);

		ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
						  &peer_addr),
			  1);

		buf = (uint8_t *) calloc(kBufSize, 1);
		ASSERT_NE(buf, nullptr);
		int ret = fi_mr_reg(resource.domain, buf, kBufSize,
				    FI_SEND | FI_RECV | FI_READ | FI_WRITE, 0, 0,
				    0, &mr, NULL);
		ASSERT_EQ(ret, 0) << "fi_mr_reg failed: " << fi_strerror(-ret);
		desc = fi_mr_desc(mr);

		ret = efa_test_dp_probe_install(resource.ep, backend(), &probe);
		if (ret == -FI_EOPNOTSUPP)
			GTEST_SKIP() << "backend not available on this device";
		ASSERT_EQ(ret, 0);
	}

	void TearDown() override
	{
		efa_test_dp_probe_restore(&probe);
		if (mr) {
			EXPECT_EQ(fi_close(&mr->fid), 0);
			mr = nullptr;
		}
		free(buf);
		buf = nullptr;
		efa_test_resource_destruct(&resource);
		efa_test_set_track_mr(prev_track_mr);
	}

	bool submitted() const { return efa_test_dp_probe_submitted(&probe); }
	bool pending() const { return efa_test_dp_probe_pending(&probe); }

	/* fi_sendmsg / fi_readmsg / fi_writemsg: the only variants that can
	 * carry FI_MORE, since it is a per-call flag. */
	ssize_t post_msg(uint64_t flags)
	{
		struct iovec iov = {.iov_base = buf, .iov_len = kBufSize};

		if (op() == EFA_TEST_POST_SEND) {
			struct fi_msg msg = {};

			msg.msg_iov = &iov;
			msg.desc = &desc;
			msg.iov_count = 1;
			msg.addr = peer_addr;
			return fi_sendmsg(resource.ep, &msg, flags);
		}

		struct fi_rma_iov rma_iov = {};
		struct fi_msg_rma msg = {};

		rma_iov.addr = EFA_TEST_RADDR;
		rma_iov.len = kBufSize;
		rma_iov.key = EFA_TEST_RKEY;
		msg.msg_iov = &iov;
		msg.desc = &desc;
		msg.iov_count = 1;
		msg.addr = peer_addr;
		msg.rma_iov = &rma_iov;
		msg.rma_iov_count = 1;

		if (op() == EFA_TEST_POST_READ)
			return fi_readmsg(resource.ep, &msg, flags);
		return fi_writemsg(resource.ep, &msg, flags);
	}

	/* The variants with no flags argument. FI_MORE is not among EFA's
	 * advertised tx op_flags, so these can never defer. */
	ssize_t post_plain()
	{
		switch (op()) {
		case EFA_TEST_POST_SEND:
			return fi_send(resource.ep, buf, kBufSize, desc,
				       peer_addr, NULL);
		case EFA_TEST_POST_READ:
			return fi_read(resource.ep, buf, kBufSize, desc,
				       peer_addr, EFA_TEST_RADDR, EFA_TEST_RKEY,
				       NULL);
		default:
			return fi_write(resource.ep, buf, kBufSize, desc,
					peer_addr, EFA_TEST_RADDR,
					EFA_TEST_RKEY, NULL);
		}
	}
};

TEST_P(EfaFiMoreTest, fi_more_defers_submission)
{
	EXPECT_EQ(post_msg(FI_MORE), 0);

	EXPECT_TRUE(pending());
	EXPECT_FALSE(submitted());
}

TEST_P(EfaFiMoreTest, no_fi_more_submits)
{
	EXPECT_EQ(post_msg(0), 0);

	EXPECT_FALSE(pending());
	EXPECT_TRUE(submitted());
}

TEST_P(EfaFiMoreTest, fi_more_batch_submits_once)
{
	/* Nothing may be submitted while the batch is still being built. A
	 * submission at any point here would show up as either a rung doorbell
	 * or a completed work request session. */
	for (int i = 0; i < 3; i++) {
		ASSERT_EQ(post_msg(FI_MORE), 0) << "post " << i;
		ASSERT_TRUE(pending()) << "post " << i;
		ASSERT_FALSE(submitted()) << "post " << i;
	}

	EXPECT_EQ(post_msg(0), 0);

	EXPECT_FALSE(pending());
	EXPECT_TRUE(submitted());
}

TEST_P(EfaFiMoreTest, plain_call_submits)
{
	EXPECT_EQ(post_plain(), 0);

	EXPECT_FALSE(pending());
	EXPECT_TRUE(submitted());
}

/**
 * @brief A failed submission still ends the batch, so the work request session
 * must not be left open. Only the rdma-core path can fail this way; the direct
 * path writes a doorbell register, which reports nothing.
 */
TEST_P(EfaFiMoreTest, submit_failure_clears_wr_session)
{
	if (backend() != EFA_TEST_DP_RDMA_CORE)
		GTEST_SKIP() << "submission cannot fail on the direct path";

	efa_test_dp_probe_set_submit_error(&probe, EINVAL);
	EXPECT_NE(post_msg(0), 0);
	EXPECT_FALSE(pending());

	/* So the next post opens a new session rather than reusing a dead one. */
	efa_test_dp_probe_set_submit_error(&probe, 0);
	EXPECT_EQ(post_msg(FI_MORE), 0);
	EXPECT_TRUE(pending());
}

INSTANTIATE_TEST_SUITE_P(Ops, EfaFiMoreTest,
			 Combine(Values(EFA_TEST_POST_SEND, EFA_TEST_POST_READ,
					EFA_TEST_POST_WRITE),
				 Values(EFA_TEST_DP_DIRECT,
					EFA_TEST_DP_RDMA_CORE)),
			 efa_test_param_name);
