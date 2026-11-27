/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <config.h>
#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_xpu_device_efa.h>

using testing::Test;

/* Callbacks that are never called: a context only records them. */
static int xpu_test_alloc(uint64_t, uint64_t, uint64_t, uint64_t, void **,
			  int *, uint64_t *)
{
	return -FI_ENOSYS;
}

static void xpu_test_free(uint64_t, void *)
{
}

/* fi_xpu_ctx(3) on an EFA domain: one context per efa-direct domain. */
class EfaXpuCtxTest : public Test
{
	protected:
	struct efa_resource resource = {};
	struct fi_xpu_attr attr = {};
	struct fid_xpu_ctx *ctx = nullptr;

	void construct(const char *fabric)
	{
		ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, fabric)));
		attr.iface = FI_HMEM_SYSTEM;
	}

	void TearDown() override
	{
		if (ctx)
			EXPECT_EQ(fi_close(&ctx->fid), 0);
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaXpuCtxTest, opens_queries_and_closes_on_efa_direct)
{
	struct fi_xpu_ctx_attr ctx_attr = {};

	ASSERT_NO_FATAL_FAILURE(construct(EFA_DIRECT_FABRIC_NAME));
	ASSERT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), 0);
	EXPECT_EQ(ctx->fid.fclass, (size_t) FI_CLASS_XPU_CTX);

	ASSERT_EQ(fi_xpu_ctx_query(ctx, &ctx_attr), 0);
	/* The objects a context exports, where the build has the direct data path. */
#if HAVE_EFA_DATA_PATH_DIRECT
	EXPECT_EQ(ctx_attr.caps, (uint64_t) (FI_XPU_CAP_EP | FI_XPU_CAP_CQ |
					     FI_XPU_CAP_CNTR));
#else
	EXPECT_EQ(ctx_attr.caps, 0u);
#endif
	EXPECT_EQ(ctx_attr.av_addr_size, sizeof(struct efa_xpu_peer));
	EXPECT_EQ(ctx_attr.mr_desc_size, sizeof(struct efa_xpu_desc));

	EXPECT_EQ(fi_close(&ctx->fid), 0);
	ctx = nullptr;
}

TEST_F(EfaXpuCtxTest, efa_fabric_refuses_a_context)
{
	ASSERT_NO_FATAL_FAILURE(construct(EFA_FABRIC_NAME));
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(ctx, nullptr);
}

TEST_F(EfaXpuCtxTest, a_domain_has_one_context)
{
	struct fid_xpu_ctx *second = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(EFA_DIRECT_FABRIC_NAME));
	ASSERT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), 0);
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &attr, &second, NULL),
		  -FI_EBUSY);

	/* Once it is closed, the domain can open another. */
	ASSERT_EQ(fi_close(&ctx->fid), 0);
	ctx = nullptr;
	ASSERT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), 0);
}

TEST_F(EfaXpuCtxTest, the_domain_cannot_close_under_its_context)
{
	ASSERT_NO_FATAL_FAILURE(construct(EFA_DIRECT_FABRIC_NAME));
	ASSERT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), 0);

	/* Close everything but the domain, then the domain. */
	ASSERT_EQ(fi_close(&resource.ep->fid), 0);
	resource.ep = nullptr;
	ASSERT_EQ(fi_close(&resource.av->fid), 0);
	resource.av = nullptr;
	ASSERT_EQ(fi_close(&resource.cq->fid), 0);
	resource.cq = nullptr;
	EXPECT_EQ(fi_close(&resource.domain->fid), -FI_EBUSY);
}

TEST_F(EfaXpuCtxTest, incomplete_memory_callbacks_are_refused)
{
	struct fi_xpu_ops ops = {};

	ASSERT_NO_FATAL_FAILURE(construct(EFA_DIRECT_FABRIC_NAME));
	attr.ops = &ops;

	ops.size = sizeof(ops) - 1;
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), -FI_EINVAL);

	ops.size = sizeof(ops);
	ops.alloc = xpu_test_alloc;
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), -FI_EINVAL);

	ops.free = xpu_test_free;
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &attr, &ctx, NULL), 0);
}
