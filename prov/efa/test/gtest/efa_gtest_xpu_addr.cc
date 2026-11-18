/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_xpu_addr_utils.h"
#include <gtest/gtest.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_xpu_device_efa.h>
#include <cstring>
#include <vector>

using testing::TestWithParam;
using testing::Values;

/*
 * The layouts are an ABI between libfabric and kernels compiled against this
 * header, and their sizes are what fi_xpu_ctx_query() reports, so they are
 * pinned: both are 8 bytes.
 */
static_assert(sizeof(struct efa_xpu_peer) == 8, "efa_xpu_peer is 8 bytes");
static_assert(sizeof(struct efa_xpu_desc) == 8, "efa_xpu_desc is 8 bytes");

/*
 * fi_av_lookup2() and fi_mr_get_xpu_desc() hand an XPU kernel the peer and the
 * key it builds work requests from (fi_av(3), fi_mr(3)). FI_XPU is an
 * efa-direct capability; the efa fabric refuses it (below). Neither call needs
 * FI_XPU to have been negotiated. The XPU context has to be one of the AV's or
 * the MR's domain.
 */
class EfaXpuAddrTest : public TestWithParam<const char *>
{
	protected:
	struct efa_resource resource = {};
	/* The domain's context on efa-direct; a stand-in on the efa fabric. */
	struct fid_xpu_ctx *xpu_ctx = nullptr;
	struct fid_xpu_ctx stand_in = {};
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	struct fid_mr *mr = nullptr;
	uint8_t buf[4096] = {};
	/* Arbitrary, and distinct from anything the provider could default to. */
	static constexpr uint16_t peer_qpn = 0x4321;
	static constexpr uint32_t peer_qkey = 0x00abcdef;

	void SetUp() override
	{
		ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(FI_EP_RDM,
							       GetParam())));
		if (strcmp(GetParam(), EFA_DIRECT_FABRIC_NAME) == 0) {
			struct fi_xpu_attr attr = {};

			attr.iface = FI_HMEM_SYSTEM;
			ASSERT_EQ(fi_xpu_ctx(resource.domain, &attr, &xpu_ctx,
					     NULL),
				  0);
		} else {
			stand_in.fid.fclass = FI_CLASS_XPU_CTX;
			xpu_ctx = &stand_in;
		}

		ASSERT_EQ(efa_test_xpu_av_insert_peer(resource.ep, resource.av,
						      peer_qpn, peer_qkey,
						      &peer_addr),
			  1);
		ASSERT_EQ(fi_mr_reg(resource.domain, buf, sizeof(buf),
				    FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			  0);
	}

	void TearDown() override
	{
		if (mr)
			EXPECT_EQ(fi_close(&mr->fid), 0);
		if (xpu_ctx && xpu_ctx != &stand_in)
			EXPECT_EQ(fi_close(&xpu_ctx->fid), 0);
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaXpuAddrTest, lookup2_reports_the_peer_for_the_xpu)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer);

	ASSERT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				xpu_ctx),
		  0);

	EXPECT_EQ(len, sizeof(peer));
	EXPECT_EQ((int) peer.ahn, efa_test_xpu_av_ahn(resource.av, peer_addr));
	EXPECT_EQ(peer.remote_qpn, peer_qpn);
	EXPECT_EQ(peer.remote_qkey, peer_qkey);
}

/* Without FI_XPU, fi_av_lookup2() is fi_av_lookup(). */
TEST_P(EfaXpuAddrTest, lookup2_without_xpu_is_lookup)
{
	std::vector<uint8_t> want(resource.info->src_addrlen);
	std::vector<uint8_t> got(resource.info->src_addrlen);
	size_t want_len = want.size(), got_len = got.size();

	ASSERT_EQ(fi_av_lookup(resource.av, peer_addr, want.data(), &want_len), 0);
	ASSERT_EQ(fi_av_lookup2(resource.av, peer_addr, got.data(), &got_len, 0,
				NULL),
		  0);

	EXPECT_EQ(got_len, want_len);
	EXPECT_EQ(got, want);
}

TEST_P(EfaXpuAddrTest, lookup2_reports_the_size_a_small_buffer_needs)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer) - 1;

	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				xpu_ctx),
		  -FI_ETOOSMALL);
	EXPECT_EQ(len, sizeof(peer));
	EXPECT_EQ(peer.remote_qpn, 0);
}

TEST_P(EfaXpuAddrTest, lookup2_refuses_bad_arguments)
{
	struct efa_xpu_peer peer = {};
	struct fid_xpu_ctx not_ctx = {};
	size_t len = sizeof(peer);

	not_ctx.fid.fclass = FI_CLASS_DOMAIN;
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				NULL),
		  -FI_EINVAL);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				&not_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, nullptr, &len, FI_XPU,
				xpu_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr + 4096, &peer, &len,
				FI_XPU, xpu_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len,
				FI_XPU | FI_MORE, xpu_ctx),
		  -FI_EBADFLAGS);
}

/* A context of another domain is not one these objects can be used with. */
TEST_P(EfaXpuAddrTest, a_context_of_another_domain_is_refused)
{
	struct efa_resource other = {};
	struct fid_xpu_ctx *other_ctx = nullptr;
	struct fi_xpu_attr attr = {};
	struct efa_xpu_peer peer = {};
	struct efa_xpu_desc desc = {};
	size_t len;

	ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct(
		&other, efa_test_alloc_default_hints(FI_EP_RDM, GetParam())));
	attr.iface = FI_HMEM_SYSTEM;
	ASSERT_EQ(fi_xpu_ctx(other.domain, &attr, &other_ctx, NULL), 0);

	len = sizeof(peer);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				other_ctx),
		  -FI_EINVAL);
	len = sizeof(desc);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, other_ctx),
		  -FI_EINVAL);

	EXPECT_EQ(fi_close(&other_ctx->fid), 0);
	efa_test_resource_destruct(&other);
}

TEST_P(EfaXpuAddrTest, xpu_desc_reports_the_registration_lkey)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc);

	ASSERT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, xpu_ctx), 0);

	EXPECT_EQ(len, sizeof(desc));
	EXPECT_EQ(desc.lkey, efa_test_xpu_mr_lkey(mr));
}

TEST_P(EfaXpuAddrTest, xpu_desc_reserved_word_is_zero)
{
	struct efa_xpu_desc desc;
	size_t len = sizeof(desc);

	memset(&desc, 0xa5, sizeof(desc));
	ASSERT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, xpu_ctx), 0);
	EXPECT_EQ(desc.reserved, 0u);
}

TEST_P(EfaXpuAddrTest, xpu_desc_reports_the_size_a_small_buffer_needs)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc) - 1;

	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, xpu_ctx),
		  -FI_ETOOSMALL);
	EXPECT_EQ(len, sizeof(desc));
	EXPECT_EQ(desc.lkey, 0u);
}

TEST_P(EfaXpuAddrTest, xpu_desc_refuses_bad_arguments)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc);

	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, NULL), -FI_EINVAL);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, nullptr, &len, FI_XPU, xpu_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, nullptr, FI_XPU, xpu_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU | FI_MORE,
				     xpu_ctx),
		  -FI_EBADFLAGS);
	EXPECT_EQ(fi_control(&mr->fid, -1, NULL), -FI_ENOSYS);
}

INSTANTIATE_TEST_SUITE_P(Fabrics, EfaXpuAddrTest,
			 Values(EFA_DIRECT_FABRIC_NAME),
			 [](const testing::TestParamInfo<const char *> &info) {
				 return "efaDirect";
			 });

/*
 * The efa fabric has no FI_XPU: its AV and MR refuse the XPU form, and
 * fi_av_lookup2() without FI_XPU is still fi_av_lookup().
 */
class EfaRdmXpuAddrTest : public EfaXpuAddrTest
{
};

TEST_P(EfaRdmXpuAddrTest, xpu_forms_are_refused)
{
	struct efa_xpu_peer peer = {};
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(peer);

	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, FI_XPU,
				xpu_ctx),
		  -FI_ENOSYS);
	len = sizeof(desc);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, FI_XPU, xpu_ctx),
		  -FI_ENOSYS);
}

TEST_P(EfaRdmXpuAddrTest, lookup2_without_xpu_is_lookup)
{
	std::vector<uint8_t> want(resource.info->src_addrlen);
	std::vector<uint8_t> got(resource.info->src_addrlen);
	size_t want_len = want.size(), got_len = got.size();

	ASSERT_EQ(fi_av_lookup(resource.av, peer_addr, want.data(), &want_len), 0);
	ASSERT_EQ(fi_av_lookup2(resource.av, peer_addr, got.data(), &got_len, 0,
				NULL),
		  0);
	EXPECT_EQ(got, want);
}

INSTANTIATE_TEST_SUITE_P(Fabrics, EfaRdmXpuAddrTest, Values(EFA_FABRIC_NAME),
			 [](const testing::TestParamInfo<const char *> &info) {
				 return "efa";
			 });
