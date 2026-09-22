/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include <config.h>
#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_xpu_utils.h"
#include <gtest/gtest.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_xpu_device_efa.h>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using testing::Invoke;
using testing::StrictMock;
using testing::Test;
using testing::Values;

/* Hardware geometry the mocked efadv queries report. */
#define XPU_TEST_SQ_ENTRIES	32u
#define XPU_TEST_SQ_ENTRY_SIZE	64u
#define XPU_TEST_SQ_MAX_BATCH	8u
#define XPU_TEST_SQ_SHIFT	5 /* log2(XPU_TEST_SQ_ENTRIES) */
#define XPU_TEST_RQ_ENTRIES	64u
#define XPU_TEST_RQ_ENTRY_SIZE	16u
#define XPU_TEST_RQ_MAX_BATCH	16u
#define XPU_TEST_RQ_SHIFT	6 /* log2(XPU_TEST_RQ_ENTRIES) */
#define XPU_TEST_CQ_ENTRIES	128u
#define XPU_TEST_CQ_ENTRY_SIZE	32u
#define XPU_TEST_CQ_SHIFT	7 /* log2(XPU_TEST_CQ_ENTRIES) */

namespace
{

/*
 * fi_xpu_ops backed by host memory.
 *
 * The provider only stores and copies what these callbacks hand back, so a
 * host-backed allocator exercises the host side of the XPU API without an XPU
 * runtime, and lets a test read back the device-side structs the provider
 * published. It also counts calls and tracks live allocations, so an error path
 * can be checked for a leaked device struct.
 */
struct XpuHostOps {
	std::set<void *> live;
	int alloc_error = 0;
	int import_error = 0;
	unsigned alloc_calls = 0;
	unsigned import_calls = 0;
	unsigned unimport_calls = 0;
	unsigned free_calls = 0;

	void free_all()
	{
		for (void *addr : live)
			free(addr);
		live.clear();
	}

	void reset()
	{
		free_all();
		alloc_error = 0;
		import_error = 0;
		alloc_calls = 0;
		import_calls = 0;
		unimport_calls = 0;
		free_calls = 0;
	}
};

XpuHostOps g_xpu;

void *xpu_test_device_alloc(size_t size)
{
	void *addr;

	if (posix_memalign(&addr, 64, size))
		return nullptr;

	memset(addr, 0, size);
	g_xpu.live.insert(addr);
	return addr;
}

int xpu_test_alloc(uint64_t device, uint64_t size, uint64_t alignment,
		   uint64_t flags, void **addr, int *fd, uint64_t *offset)
{
	void *buf;

	g_xpu.alloc_calls++;
	if (g_xpu.alloc_error)
		return g_xpu.alloc_error;

	buf = xpu_test_device_alloc(size);
	if (!buf)
		return -FI_ENOMEM;

	*addr = buf;
	if (fd)
		*fd = -1;
	if (offset)
		*offset = 0;
	return 0;
}

int xpu_test_import(uint64_t device, void *host_addr, uint64_t size,
		    uint64_t flags, void **dev_addr)
{
	g_xpu.import_calls++;
	if (g_xpu.import_error)
		return g_xpu.import_error;

	/* Host memory is already reachable from the "device". */
	*dev_addr = host_addr;
	return 0;
}

int xpu_test_unimport(uint64_t device, void *host_addr)
{
	g_xpu.unimport_calls++;
	/* Host memory was already reachable, so there is nothing to undo. */
	return 0;
}

void xpu_test_free(uint64_t device, void *addr)
{
	g_xpu.free_calls++;
	if (addr && g_xpu.live.erase(addr))
		free(addr);
}

struct fi_xpu_ops g_xpu_ops = { sizeof(struct fi_xpu_ops), xpu_test_alloc,
				xpu_test_import, xpu_test_unimport,
				xpu_test_free };

/*
 * Whether the provider advertises FI_XPU, i.e. whether the device can expose
 * its queues. That is only readable once the provider has initialized, which
 * the first fi_getinfo() does.
 */
bool xpu_available()
{
	struct fi_info *hints, *info = nullptr;
	int ret;

	hints = efa_test_alloc_default_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	if (!hints)
		return false;

	ret = fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints, &info);
	fi_freeinfo(hints);
	if (ret)
		return false;
	fi_freeinfo(info);

	return efa_test_xpu_device_support();
}

} /* namespace */

/*
 * fi_getinfo negotiation: FI_XPU is an opt-in capability of the efa-direct
 * fabric that the provider only returns when the application asked for it and
 * the device can support it.
 */
class EfaXpuInfoTest : public Test
{
	protected:
	struct fi_info *hints = nullptr;
	struct fi_info *info = nullptr;

	void TearDown() override
	{
		if (info)
			fi_freeinfo(info);
		if (hints)
			fi_freeinfo(hints);
	}

	/**
	 * @brief Run fi_getinfo for the given fabric with the given caps hints.
	 *
	 * @param fabric_name	fabric to request, or NULL for any
	 * @param caps		hints->caps
	 * @param domain_caps	hints->domain_attr->caps
	 * @return the fi_getinfo return code
	 */
	int getinfo(const char *fabric_name, uint64_t caps, uint64_t domain_caps)
	{
		hints = efa_test_alloc_default_hints(FI_EP_RDM, fabric_name);
		if (!hints)
			return -FI_ENOMEM;
		hints->caps = caps;
		hints->domain_attr->caps = domain_caps;
		/* The provider only offers FI_XPU to an application that
		 * accepts FI_MR_XPU_DESC, because the device path reads the
		 * memory key out of a descriptor. */
		if (caps & FI_XPU)
			hints->domain_attr->mr_mode |= FI_MR_XPU_DESC;

		return fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints,
				  &info);
	}
};

/*
 * A device that cannot expose its queues must not be offered FI_XPU, so assert
 * the correspondence in both directions rather than only the supported path.
 */
TEST_F(EfaXpuInfoTest, advertisement_follows_device_support)
{
	bool expect_xpu;

	/* Reads device state, so the provider must be initialized first. */
	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, 0, 0), 0);
	expect_xpu = efa_test_xpu_device_support();

	fi_freeinfo(info);
	info = nullptr;
	fi_freeinfo(hints);
	hints = nullptr;

	if (expect_xpu)
		EXPECT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, FI_XPU, 0), 0);
	else
		EXPECT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, FI_XPU, 0),
			  -FI_ENODATA);
}

TEST_F(EfaXpuInfoTest, direct_without_hint_reports_no_xpu)
{
	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, 0, 0), 0);

	for (struct fi_info *cur = info; cur; cur = cur->next) {
		EXPECT_FALSE(cur->caps & FI_XPU);
		EXPECT_FALSE(cur->domain_attr->caps & FI_XPU);
		EXPECT_EQ(cur->domain_attr->max_xpu_ctx_cnt, 0u);
	}
}

/* The XPU data path bypasses the rdm protocol layer, so efa cannot offer it. */
TEST_F(EfaXpuInfoTest, rdm_fabric_rejects_xpu_hint)
{
	EXPECT_EQ(getinfo(EFA_FABRIC_NAME, FI_XPU, 0), -FI_ENODATA);
	EXPECT_EQ(info, nullptr);
}

TEST_F(EfaXpuInfoTest, requested_xpu_is_reported_on_direct)
{
	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, FI_XPU, 0), 0);

	for (struct fi_info *cur = info; cur; cur = cur->next) {
		EXPECT_TRUE(cur->caps & FI_XPU);
		/* One XPU context per domain, i.e. per NIC. */
		EXPECT_EQ(cur->domain_attr->max_xpu_ctx_cnt, 1u);
	}
}

/*
 * A kernel takes the memory key from a descriptor, so an XPU-capable info has
 * to say so in mr_mode - and an info without FI_XPU must not, or every
 * application would be told it needs XPU descriptors.
 */
TEST_F(EfaXpuInfoTest, xpu_info_reports_mr_xpu_desc)
{
	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, FI_XPU, 0), 0);
	for (struct fi_info *cur = info; cur; cur = cur->next)
		EXPECT_TRUE(cur->domain_attr->mr_mode & FI_MR_XPU_DESC);

	fi_freeinfo(info);
	info = nullptr;
	fi_freeinfo(hints);
	hints = nullptr;

	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, 0, 0), 0);
	for (struct fi_info *cur = info; cur; cur = cur->next)
		EXPECT_FALSE(cur->domain_attr->mr_mode & FI_MR_XPU_DESC);
}

/*
 * An application that does not accept FI_MR_XPU_DESC cannot pass a descriptor
 * to a kernel, so it must not be handed a device data path at all.
 */
TEST_F(EfaXpuInfoTest, xpu_without_mr_xpu_desc_is_refused)
{
	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	hints = efa_test_alloc_default_hints(FI_EP_RDM,
					     EFA_DIRECT_FABRIC_NAME);
	ASSERT_NE(hints, nullptr);
	hints->caps = FI_XPU;
	hints->domain_attr->mr_mode &= ~FI_MR_XPU_DESC;

	EXPECT_EQ(fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints, &info),
		  -FI_ENODATA);
	EXPECT_EQ(info, nullptr);
}

/* FI_XPU in hints must select the fabric that implements it. */
TEST_F(EfaXpuInfoTest, xpu_hint_selects_direct_fabric_only)
{
	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	ASSERT_EQ(getinfo(NULL, FI_XPU, 0), 0);

	for (struct fi_info *cur = info; cur; cur = cur->next) {
		EXPECT_STREQ(cur->fabric_attr->name, EFA_DIRECT_FABRIC_NAME);
		EXPECT_TRUE(cur->caps & FI_XPU);
	}
}

/*
 * An application that asks for FI_XPU at domain level must get it back there,
 * because that is where it binds the XPU context.
 */
TEST_F(EfaXpuInfoTest, domain_caps_hint_is_reported)
{
	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	ASSERT_EQ(getinfo(EFA_DIRECT_FABRIC_NAME, FI_XPU, FI_XPU), 0);

	for (struct fi_info *cur = info; cur; cur = cur->next) {
		EXPECT_TRUE(cur->domain_attr->caps & FI_XPU);
		EXPECT_EQ(cur->domain_attr->max_xpu_ctx_cnt, 1u);
	}
}

/* Fixture for the XPU objects an application creates on a domain. */
class EfaXpuCtxTest : public Test
{
	protected:
	struct efa_resource resource = {};
	struct fi_xpu_attr xpu_attr = {};
	struct fid_xpu_ctx *xpu_ctx = nullptr;
	int ctx_arg = 0;

	void SetUp() override
	{
		struct fi_info *hints;

		memset(&resource, 0, sizeof(resource));

		if (!xpu_available())
			GTEST_SKIP() << "provider does not advertise FI_XPU";

		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_XPU;
		hints->domain_attr->mr_mode |= FI_MR_XPU_DESC;
		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));

		xpu_attr.iface = FI_HMEM_SYSTEM;
		xpu_attr.device = 0;
		xpu_attr.ops = &g_xpu_ops;
		ASSERT_EQ(fi_xpu_ctx(resource.domain, &xpu_attr, &xpu_ctx,
				     &ctx_arg),
			  0);
		ASSERT_NE(xpu_ctx, nullptr);
	}

	void TearDown() override
	{
		if (xpu_ctx) {
			EXPECT_EQ(fi_close(&xpu_ctx->fid), 0);
			xpu_ctx = nullptr;
		}
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaXpuCtxTest, ctx_open_sets_class_and_context)
{
	EXPECT_EQ(xpu_ctx->fid.fclass, (uint32_t) FI_CLASS_XPU_CTX);
	EXPECT_EQ(xpu_ctx->fid.context, &ctx_arg);
}

TEST_F(EfaXpuCtxTest, ctx_open_rejects_null_arguments)
{
	struct fid_xpu_ctx *ctx = nullptr;

	EXPECT_EQ(fi_xpu_ctx(resource.domain, nullptr, &ctx, NULL), -FI_EINVAL);
	EXPECT_EQ(ctx, nullptr);
	EXPECT_EQ(fi_xpu_ctx(resource.domain, &xpu_attr, nullptr, NULL),
		  -FI_EINVAL);
}

/*
 * Only efa-direct can export its queues, so every other endpoint type has to
 * refuse: their send queues are driven by the provider's own protocol state,
 * which a kernel posting into the queue would corrupt. The dgram domain shares
 * its ops with efa-direct and is refused by the domain check in
 * efa_xpu_ctx_open(); the rdm domain has its own ops and does not offer the
 * call at all.
 */
TEST(EfaXpuCtx, ctx_open_refuses_a_domain_that_is_not_efa_direct)
{
	const struct {
		enum fi_ep_type ep_type;
		int expected;
	} cases[] = {
		{ FI_EP_DGRAM, -FI_EOPNOTSUPP },
		{ FI_EP_RDM, -FI_ENOSYS },
	};

	if (!xpu_available())
		GTEST_SKIP() << "provider does not advertise FI_XPU";

	for (auto &c : cases) {
		struct efa_resource resource = {};
		struct fi_xpu_attr xpu_attr = {};
		struct fid_xpu_ctx *ctx = nullptr;
		struct fi_info *hints;

		hints = efa_test_alloc_default_hints(c.ep_type,
						     EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));

		xpu_attr.iface = FI_HMEM_SYSTEM;
		xpu_attr.device = 0;
		xpu_attr.ops = &g_xpu_ops;

		EXPECT_EQ(fi_xpu_ctx(resource.domain, &xpu_attr, &ctx, NULL),
			  c.expected)
			<< "ep_type " << c.ep_type;
		EXPECT_EQ(ctx, nullptr);

		efa_test_resource_destruct(&resource);
	}
}

/*
 * The sizes reported here are what an application uses to size the buffers it
 * passes to fi_av_lookup2 and fi_mr_get_xpu_desc, so they must match what those
 * calls require (asserted by EfaXpuAvTest / EfaXpuMrTest).
 */
TEST_F(EfaXpuCtxTest, ctx_query_reports_objects_and_address_sizes)
{
	struct fi_xpu_ctx_attr attr = {};

	ASSERT_EQ(fi_xpu_ctx_query(xpu_ctx, &attr), 0);

	EXPECT_EQ(attr.caps, (uint64_t) (FI_XPU_CAP_EP | FI_XPU_CAP_CQ |
					 FI_XPU_CAP_CNTR));
	EXPECT_EQ(attr.av_addr_size, sizeof(struct efa_xpu_peer));
	EXPECT_EQ(attr.mr_desc_size, sizeof(struct efa_xpu_desc));
}

/*
 * fi_av_lookup2 translates an fi_addr into the peer descriptor an XPU kernel
 * addresses its work requests with. Neither it nor fi_mr_get_xpu_desc depends
 * on FI_XPU being advertised, so both run unconditionally.
 */
class EfaXpuAvTest : public Test
{
	protected:
	struct efa_resource resource = {};
	struct fi_xpu_attr xpu_attr = {};
	struct fid_xpu_ctx *xpu_ctx = nullptr;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;
	/* Arbitrary, and distinct from anything the provider could default to. */
	static constexpr uint16_t peer_qpn = 0x4321;
	static constexpr uint32_t peer_qkey = 0x00abcdef;

	void SetUp() override
	{
		struct fi_info *hints;

		memset(&resource, 0, sizeof(resource));

		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));

		xpu_attr.iface = FI_HMEM_SYSTEM;
		xpu_attr.ops = &g_xpu_ops;
		ASSERT_EQ(fi_xpu_ctx(resource.domain, &xpu_attr, &xpu_ctx, NULL),
			  0);

		ASSERT_EQ(efa_test_xpu_av_insert_peer(resource.ep, resource.av,
						      peer_qpn, peer_qkey,
						      &peer_addr),
			  1);
	}

	void TearDown() override
	{
		if (xpu_ctx) {
			EXPECT_EQ(fi_close(&xpu_ctx->fid), 0);
			xpu_ctx = nullptr;
		}
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaXpuAvTest, lookup2_reports_peer_address_handle_and_qp)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer);

	ASSERT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, 0, xpu_ctx),
		  0);

	EXPECT_EQ(len, sizeof(peer));
	EXPECT_EQ((int) peer.ahn, efa_test_xpu_av_ahn(resource.av, peer_addr));
	EXPECT_EQ(peer.remote_qpn, peer_qpn);
	EXPECT_EQ(peer.remote_qkey, peer_qkey);
}

TEST_F(EfaXpuAvTest, lookup2_rejects_small_buffer)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer) - 1;

	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, &len, 0, xpu_ctx),
		  -FI_ETOOSMALL);
	/* Nothing was written, so the caller's length must be untouched. */
	EXPECT_EQ(len, sizeof(peer) - 1);
	EXPECT_EQ(peer.ahn, 0);
	EXPECT_EQ(peer.remote_qpn, 0);
}

TEST_F(EfaXpuAvTest, lookup2_rejects_null_arguments)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer);

	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, nullptr, &len, 0,
				xpu_ctx),
		  -FI_EINVAL);
	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr, &peer, nullptr, 0,
				xpu_ctx),
		  -FI_EINVAL);
}

TEST_F(EfaXpuAvTest, lookup2_rejects_unknown_address)
{
	struct efa_xpu_peer peer = {};
	size_t len = sizeof(peer);

	EXPECT_EQ(fi_av_lookup2(resource.av, peer_addr + 4096, &peer, &len, 0,
				xpu_ctx),
		  -FI_EINVAL);
}

/* fi_mr_get_xpu_desc hands an XPU kernel the key it stamps into its WQEs. */
class EfaXpuMrTest : public Test
{
	protected:
	struct efa_resource resource = {};
	struct fi_xpu_attr xpu_attr = {};
	struct fid_xpu_ctx *xpu_ctx = nullptr;
	struct fid_mr *mr = nullptr;
	uint8_t buf[4096] = {};

	void SetUp() override
	{
		struct fi_info *hints;

		memset(&resource, 0, sizeof(resource));

		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		ASSERT_NO_FATAL_FAILURE(
			efa_test_resource_construct(&resource, hints));

		xpu_attr.iface = FI_HMEM_SYSTEM;
		xpu_attr.ops = &g_xpu_ops;
		ASSERT_EQ(fi_xpu_ctx(resource.domain, &xpu_attr, &xpu_ctx, NULL),
			  0);

		ASSERT_EQ(fi_mr_reg(resource.domain, buf, sizeof(buf),
				    FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			  0);
	}

	void TearDown() override
	{
		if (mr) {
			EXPECT_EQ(fi_close(&mr->fid), 0);
			mr = nullptr;
		}
		if (xpu_ctx) {
			EXPECT_EQ(fi_close(&xpu_ctx->fid), 0);
			xpu_ctx = nullptr;
		}
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaXpuMrTest, xpu_desc_reports_registration_lkey)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc);

	ASSERT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, 0, xpu_ctx), 0);

	EXPECT_EQ(len, sizeof(desc));
	EXPECT_EQ(desc.lkey, efa_test_xpu_mr_lkey(mr));
	EXPECT_NE(desc.lkey, 0u);
}

TEST_F(EfaXpuMrTest, xpu_desc_rejects_small_buffer)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc) - 1;

	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, &len, 0, xpu_ctx),
		  -FI_ETOOSMALL);
	EXPECT_EQ(len, sizeof(desc) - 1);
	EXPECT_EQ(desc.lkey, 0u);
}

TEST_F(EfaXpuMrTest, xpu_desc_rejects_null_arguments)
{
	struct efa_xpu_desc desc = {};
	size_t len = sizeof(desc);

	EXPECT_EQ(fi_mr_get_xpu_desc(mr, nullptr, &len, 0, xpu_ctx), -FI_EINVAL);
	EXPECT_EQ(fi_mr_get_xpu_desc(mr, &desc, nullptr, 0, xpu_ctx),
		  -FI_EINVAL);
}

TEST_F(EfaXpuMrTest, mr_control_rejects_unknown_command)
{
	EXPECT_EQ(fi_control(&mr->fid, -1, NULL), -FI_ENOSYS);
}

/*
 * Export tests.
 *
 * The exported handle is opaque to the application, but it is the contract
 * between the host and the device code: the kernel casts prov_ctx to the
 * device-side struct and reads the geometry out of it. The XPU memory ops are
 * host-backed (g_xpu_ops), so the "device" struct is readable here and is
 * asserted field by field.
 *
 * Exporting needs the efadv queue geometry queries, which is what HAVE_EFA_XPU
 * tests, and so do the mocks that stand in for them.
 */
#if HAVE_EFA_XPU
class EfaXpuExportTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	struct fi_xpu_attr xpu_attr = {};
	struct fid_xpu_ctx *xpu_ctx = nullptr;
	struct fid_cntr *cntr = nullptr;
	/* Stand-ins for the QP's rings and doorbells, which are BAR MMIO. */
	std::vector<uint8_t> sq_ring;
	std::vector<uint8_t> rq_ring;
	uint32_t sq_db = 0;
	uint32_t rq_db = 0;
	struct efa_test_xpu_wq_geometry sq_geo = {};
	struct efa_test_xpu_wq_geometry rq_geo = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		if (!xpu_available())
			GTEST_SKIP() << "provider does not advertise FI_XPU";

		g_xpu.reset();
		ASSERT_NO_FATAL_FAILURE(construct());
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);

		/* A counter is still bound to the endpoint, so the endpoint has
		 * to go first. */
		if (resource.ep) {
			EXPECT_EQ(fi_close(&resource.ep->fid), 0);
			resource.ep = nullptr;
		}
		if (cntr) {
			EXPECT_EQ(fi_close(&cntr->fid), 0);
			cntr = nullptr;
		}
		/* Closing the CQ and counter releases the state installed on
		 * them through the context's memory ops, so the context must
		 * outlive them. */
		efa_test_resource_destruct(&resource);
		if (xpu_ctx) {
			EXPECT_EQ(fi_close(&xpu_ctx->fid), 0);
			xpu_ctx = nullptr;
		}

		/* The provider has no unexport call, so the device structs a
		 * successful export allocates are still live here. */
		g_xpu.free_all();
	}

	/*
	 * Build the resources by hand: the endpoint must be opened with FI_XPU
	 * and from info that already carries the XPU context, which is how an
	 * application asks for one, and fi_enable is left to the test.
	 */
	void construct()
	{
		struct fi_info *hints;
		struct fi_av_attr av_attr = {};
		struct fi_cq_attr cq_attr = {};
		int ret;

		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->caps |= FI_XPU;
		hints->domain_attr->mr_mode |= FI_MR_XPU_DESC;
		resource.hints = hints;

		ret = fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints,
				 &resource.info);
		ASSERT_EQ(ret, 0) << "fi_getinfo failed: " << fi_strerror(-ret);
		ASSERT_TRUE(resource.info->caps & FI_XPU);

		ret = fi_fabric(resource.info->fabric_attr, &resource.fabric,
				NULL);
		ASSERT_EQ(ret, 0) << "fi_fabric failed: " << fi_strerror(-ret);

		ret = fi_domain(resource.fabric, resource.info,
				&resource.domain, NULL);
		ASSERT_EQ(ret, 0) << "fi_domain failed: " << fi_strerror(-ret);

		xpu_attr.iface = FI_HMEM_SYSTEM;
		xpu_attr.device = 0;
		xpu_attr.ops = &g_xpu_ops;
		ret = fi_xpu_ctx(resource.domain, &xpu_attr, &xpu_ctx, NULL);
		ASSERT_EQ(ret, 0) << "fi_xpu_ctx failed: " << fi_strerror(-ret);

		resource.info->ep_attr->xpu_ctx = xpu_ctx;

		ret = fi_endpoint2(resource.domain, resource.info, &resource.ep,
				   FI_XPU, NULL);
		ASSERT_EQ(ret, 0) << "fi_endpoint2 failed: "
				  << fi_strerror(-ret);

		ret = fi_av_open(resource.domain, &av_attr, &resource.av, NULL);
		ASSERT_EQ(ret, 0) << "fi_av_open failed: " << fi_strerror(-ret);
		ASSERT_EQ(fi_ep_bind(resource.ep, &resource.av->fid, 0), 0);

		cq_attr.format = FI_CQ_FORMAT_DATA;
		ret = fi_cq_open(resource.domain, &cq_attr, &resource.cq, NULL);
		ASSERT_EQ(ret, 0) << "fi_cq_open failed: " << fi_strerror(-ret);
		ASSERT_EQ(fi_ep_bind(resource.ep, &resource.cq->fid,
				     FI_SEND | FI_RECV),
			  0);
	}

	void enable_ep()
	{
		int ret = fi_enable(resource.ep);

		ASSERT_EQ(ret, 0) << "fi_enable failed: " << fi_strerror(-ret);
	}

	/**
	 * @brief Arm the queue geometry query with a fabricated SQ and, when
	 * asked for, RQ. A QP with no receive queue reports num_entries 0 and a
	 * NULL buffer.
	 */
	void expect_qp_wqs_query(bool with_rq)
	{
		sq_ring.assign(XPU_TEST_SQ_ENTRIES * XPU_TEST_SQ_ENTRY_SIZE, 0);
		rq_ring.assign(XPU_TEST_RQ_ENTRIES * XPU_TEST_RQ_ENTRY_SIZE, 0);

		sq_geo = { XPU_TEST_SQ_ENTRIES, XPU_TEST_SQ_ENTRY_SIZE,
			   XPU_TEST_SQ_MAX_BATCH, sq_ring.data(), &sq_db };
		if (with_rq)
			rq_geo = { XPU_TEST_RQ_ENTRIES, XPU_TEST_RQ_ENTRY_SIZE,
				   XPU_TEST_RQ_MAX_BATCH, rq_ring.data(),
				   &rq_db };
		else
			rq_geo = {};

		MockEfa::set(&mock_efa);
		EFA_EXPECT_CALL(mock_efa, efadv_query_qp_wqs)
			.WillOnce(Invoke([this](struct ibv_qp *qp,
						struct efadv_wq_attr *sq_attr,
						struct efadv_wq_attr *rq_attr,
						uint32_t inlen) {
				return efa_test_xpu_fill_wq_attrs(
					sq_attr, rq_attr, &sq_geo, &rq_geo);
			}));
	}

	static struct efa_xpu_ep *device_ep(const struct fid_xpu_ep &handle)
	{
		return (struct efa_xpu_ep *) (uintptr_t) handle.fid.prov_ctx;
	}
};

TEST_F(EfaXpuExportTest, ep_export_publishes_queue_geometry)
{
	struct fid_xpu_ep xpu_ep = {};
	struct efa_xpu_ep *dev_ep;

	ASSERT_NO_FATAL_FAILURE(enable_ep());
	expect_qp_wqs_query(true);

	ASSERT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), 0);

	EXPECT_EQ(xpu_ep.fid.fclass, (uint32_t) FI_CLASS_EP);
	EXPECT_EQ(xpu_ep.fid.prov_id, (uint32_t) FI_XPU_PROV_EFA);
	ASSERT_NE(xpu_ep.fid.prov_ctx, 0u);

	/* One device struct allocated; both rings and both doorbells mapped. */
	EXPECT_EQ(g_xpu.alloc_calls, 1u);
	EXPECT_EQ(g_xpu.import_calls, 4u);

	dev_ep = device_ep(xpu_ep);
	/* The device struct describes itself, so a kernel can find it again. */
	EXPECT_EQ(dev_ep->xpu_ep.fid.prov_ctx, xpu_ep.fid.prov_ctx);
	EXPECT_EQ(dev_ep->xpu_ep.fid.fclass, (uint32_t) FI_CLASS_EP);

	/* Stamped with the exporting library's version, and accepted by the
	 * device-side compatibility check compiled from the same header. */
	EXPECT_EQ(dev_ep->version, fi_version());
	EXPECT_TRUE(efa_xpu_ep_compat(dev_ep));

	EXPECT_EQ(dev_ep->sq.queue_mask, XPU_TEST_SQ_ENTRIES - 1);
	EXPECT_EQ(dev_ep->sq.queue_size_shift, (uint32_t) XPU_TEST_SQ_SHIFT);
	EXPECT_EQ(dev_ep->sq.max_batch, XPU_TEST_SQ_MAX_BATCH);
	EXPECT_EQ(dev_ep->sq.entry_size, XPU_TEST_SQ_ENTRY_SIZE);
	EXPECT_EQ(dev_ep->sq.buf, sq_ring.data());
	EXPECT_EQ(dev_ep->sq.db, &sq_db);
	/*
	 * Nothing reserved, nothing handed on, no doorbell rung, and the first
	 * pass over the ring carries phase 0 - the three cursors a kernel needs
	 * to produce into the queue without a lock.
	 */
	EXPECT_EQ(dev_ep->sq.pc, 0u);
	EXPECT_EQ(dev_ep->sq.released, 0u);
	EXPECT_EQ(dev_ep->sq.db_rung, 0u);
	EXPECT_EQ(dev_ep->sq.init_phase, 0);
	EXPECT_EQ(dev_ep->sq_size, XPU_TEST_SQ_ENTRIES);
	EXPECT_EQ(dev_ep->submitted_count, 0u);

	EXPECT_EQ(dev_ep->rq.queue_mask, XPU_TEST_RQ_ENTRIES - 1);
	EXPECT_EQ(dev_ep->rq.queue_size_shift, (uint32_t) XPU_TEST_RQ_SHIFT);
	EXPECT_EQ(dev_ep->rq.max_batch, XPU_TEST_RQ_MAX_BATCH);
	EXPECT_EQ(dev_ep->rq.entry_size, XPU_TEST_RQ_ENTRY_SIZE);
	EXPECT_EQ(dev_ep->rq.buf, rq_ring.data());
	EXPECT_EQ(dev_ep->rq.db, &rq_db);
	EXPECT_EQ(dev_ep->rq.pc, 0u);
	EXPECT_EQ(dev_ep->rq.released, 0u);
	EXPECT_EQ(dev_ep->rq.db_rung, 0u);
	EXPECT_EQ(dev_ep->rq.init_phase, 0);

	/* No counter was bound, so there is no backpressure counter to read. */
	EXPECT_EQ(dev_ep->local_cntr, nullptr);
}

TEST_F(EfaXpuExportTest, ep_export_without_rq_leaves_rq_empty)
{
	struct fid_xpu_ep xpu_ep = {};
	struct efa_xpu_ep *dev_ep;

	ASSERT_NO_FATAL_FAILURE(enable_ep());
	expect_qp_wqs_query(false);

	ASSERT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), 0);

	/* Only the send ring and its doorbell are mapped. */
	EXPECT_EQ(g_xpu.import_calls, 2u);

	dev_ep = device_ep(xpu_ep);
	EXPECT_EQ(dev_ep->sq.buf, sq_ring.data());
	EXPECT_EQ(dev_ep->rq.buf, nullptr);
	EXPECT_EQ(dev_ep->rq.db, nullptr);
	EXPECT_EQ(dev_ep->rq.queue_mask, 0u);
	EXPECT_EQ(dev_ep->rq.entry_size, 0u);
}

/*
 * The write counter is how an XPU kernel learns that submitted work completed,
 * which is what bounds the send queue, so a bound counter must reach the device
 * struct.
 */
TEST_F(EfaXpuExportTest, ep_export_links_bound_write_counter)
{
	struct fi_cntr_attr cntr_attr = {};
	struct fid_xpu_ep xpu_ep = {};
	void *value_dev, *err_dev;

	value_dev = xpu_test_device_alloc(sizeof(uint64_t));
	err_dev = xpu_test_device_alloc(sizeof(uint64_t));
	ASSERT_NE(value_dev, nullptr);
	ASSERT_NE(err_dev, nullptr);

	ASSERT_EQ(fi_cntr_open(resource.domain, &cntr_attr, &cntr, NULL), 0);
	ASSERT_EQ(efa_test_xpu_install_cntr_state(cntr, xpu_ctx, value_dev,
						  err_dev),
		  0);
	ASSERT_EQ(fi_ep_bind(resource.ep, &cntr->fid, FI_WRITE), 0);

	ASSERT_NO_FATAL_FAILURE(enable_ep());
	expect_qp_wqs_query(true);

	ASSERT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), 0);

	EXPECT_EQ((void *) device_ep(xpu_ep)->local_cntr, value_dev);
}

TEST_F(EfaXpuExportTest, ep_export_without_context_returns_einval)
{
	struct fid_xpu_ep xpu_ep = {};
	struct fid_ep *ep = nullptr;

	/* A plain endpoint drives its queues from the host, so it has nothing
	 * to export to a device. */
	resource.info->ep_attr->xpu_ctx = nullptr;
	ASSERT_EQ(fi_endpoint(resource.domain, resource.info, &ep, NULL), 0);
	resource.info->ep_attr->xpu_ctx = xpu_ctx;

	EXPECT_EQ(fi_ep_export_xpu(ep, 0, &xpu_ep), -FI_EINVAL);
	EXPECT_EQ(xpu_ep.fid.prov_ctx, 0u);
	EXPECT_TRUE(g_xpu.live.empty());

	EXPECT_EQ(fi_close(&ep->fid), 0);
}

/*
 * FI_XPU and ep_attr->xpu_ctx are only meaningful together, so an endpoint
 * opened with one of them is refused rather than opened as something else.
 */
TEST_F(EfaXpuExportTest, endpoint2_rejects_xpu_flag_without_context)
{
	struct fid_ep *ep = nullptr;

	resource.info->ep_attr->xpu_ctx = nullptr;
	EXPECT_EQ(fi_endpoint2(resource.domain, resource.info, &ep, FI_XPU,
			       NULL),
		  -FI_EINVAL);
	resource.info->ep_attr->xpu_ctx = xpu_ctx;

	EXPECT_EQ(ep, nullptr);
}

TEST_F(EfaXpuExportTest, endpoint_rejects_context_without_xpu_flag)
{
	struct fid_ep *ep = nullptr;

	EXPECT_EQ(fi_endpoint(resource.domain, resource.info, &ep, NULL),
		  -FI_EINVAL);
	EXPECT_EQ(fi_endpoint2(resource.domain, resource.info, &ep, 0, NULL),
		  -FI_EINVAL);
	EXPECT_EQ(ep, nullptr);
}

TEST_F(EfaXpuExportTest, endpoint2_rejects_unsupported_flags)
{
	struct fid_ep *ep = nullptr;

	EXPECT_EQ(fi_endpoint2(resource.domain, resource.info, &ep,
			       FI_XPU | FI_PEER, NULL),
		  -FI_EBADFLAGS);
	EXPECT_EQ(fi_endpoint2(resource.domain, resource.info, &ep, FI_PEER,
			       NULL),
		  -FI_EBADFLAGS);
	EXPECT_EQ(ep, nullptr);
}

TEST_F(EfaXpuExportTest, export_rejects_null_handle)
{
	struct fi_cntr_attr cntr_attr = {};

	EXPECT_EQ(fi_ep_export_xpu(resource.ep, 0, nullptr), -FI_EINVAL);
	EXPECT_EQ(fi_cq_export_xpu(resource.cq, 0, nullptr), -FI_EINVAL);

	ASSERT_EQ(fi_cntr_open(resource.domain, &cntr_attr, &cntr, NULL), 0);
	EXPECT_EQ(fi_cntr_export_xpu(cntr, 0, nullptr), -FI_EINVAL);
}

/* A CQ or counter opened without FI_XPU has no device-side ring to export. */
TEST_F(EfaXpuExportTest, export_without_xpu_state_returns_enodata)
{
	struct fi_cntr_attr cntr_attr = {};
	struct fid_xpu_cq xpu_cq = {};
	struct fid_xpu_cntr xpu_cntr = {};

	EXPECT_EQ(fi_cq_export_xpu(resource.cq, 0, &xpu_cq), -FI_ENODATA);
	EXPECT_EQ(xpu_cq.fid.prov_ctx, 0u);

	ASSERT_EQ(fi_cntr_open(resource.domain, &cntr_attr, &cntr, NULL), 0);
	EXPECT_EQ(fi_cntr_export_xpu(cntr, 0, &xpu_cntr), -FI_ENODATA);
	EXPECT_EQ(xpu_cntr.fid.prov_ctx, 0u);

	EXPECT_TRUE(g_xpu.live.empty());
}

TEST_F(EfaXpuExportTest, cq_export_publishes_ring_geometry)
{
	struct fid_xpu_cq xpu_cq = {};
	struct efa_xpu_cq *dev_cq;
	void *cq_buf;

	cq_buf = xpu_test_device_alloc(XPU_TEST_CQ_ENTRIES *
				       XPU_TEST_CQ_ENTRY_SIZE);
	ASSERT_NE(cq_buf, nullptr);
	ASSERT_EQ(efa_test_xpu_install_cq_state(resource.cq, xpu_ctx, cq_buf),
		  0);

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efadv_query_cq)
		.WillOnce(Invoke([](struct ibv_cq *cq,
				    struct efadv_cq_attr *attr, uint32_t inlen) {
			return efa_test_xpu_fill_cq_attr(attr,
							 XPU_TEST_CQ_ENTRIES,
							 XPU_TEST_CQ_ENTRY_SIZE,
							 nullptr);
		}));

	ASSERT_EQ(fi_cq_export_xpu(resource.cq, 0, &xpu_cq), 0);

	EXPECT_EQ(xpu_cq.fid.fclass, (uint32_t) FI_CLASS_CQ);
	EXPECT_EQ(xpu_cq.fid.prov_id, (uint32_t) FI_XPU_PROV_EFA);
	ASSERT_NE(xpu_cq.fid.prov_ctx, 0u);

	dev_cq = (struct efa_xpu_cq *) (uintptr_t) xpu_cq.fid.prov_ctx;
	EXPECT_EQ(dev_cq->xpu_cq.fid.prov_ctx, xpu_cq.fid.prov_ctx);
	EXPECT_EQ(dev_cq->version, fi_version());
	EXPECT_TRUE(efa_xpu_cq_compat(dev_cq));

	/* The kernel polls the ring the CQ was opened with, not the one the
	 * query reports, which is the host mapping. */
	EXPECT_EQ(dev_cq->buf, cq_buf);
	EXPECT_EQ(dev_cq->entry_size, XPU_TEST_CQ_ENTRY_SIZE);
	EXPECT_EQ(dev_cq->queue_mask, XPU_TEST_CQ_ENTRIES - 1);
	EXPECT_EQ(dev_cq->queue_size_shift, (uint32_t) XPU_TEST_CQ_SHIFT);
	EXPECT_EQ(dev_cq->cc, 0u);
	/* Completions start in phase 1, so an untouched ring reads as empty. */
	EXPECT_EQ(dev_cq->init_phase, 1);
}

TEST_F(EfaXpuExportTest, cntr_export_publishes_counter_pointers)
{
	struct fi_cntr_attr cntr_attr = {};
	struct fid_xpu_cntr xpu_cntr = {};
	struct efa_xpu_cntr *dev_cntr;
	void *value_dev, *err_dev;

	value_dev = xpu_test_device_alloc(sizeof(uint64_t));
	err_dev = xpu_test_device_alloc(sizeof(uint64_t));
	ASSERT_NE(value_dev, nullptr);
	ASSERT_NE(err_dev, nullptr);

	ASSERT_EQ(fi_cntr_open(resource.domain, &cntr_attr, &cntr, NULL), 0);
	ASSERT_EQ(efa_test_xpu_install_cntr_state(cntr, xpu_ctx, value_dev,
						  err_dev),
		  0);

	ASSERT_EQ(fi_cntr_export_xpu(cntr, 0, &xpu_cntr), 0);

	EXPECT_EQ(xpu_cntr.fid.fclass, (uint32_t) FI_CLASS_CNTR);
	EXPECT_EQ(xpu_cntr.fid.prov_id, (uint32_t) FI_XPU_PROV_EFA);
	ASSERT_NE(xpu_cntr.fid.prov_ctx, 0u);

	dev_cntr = (struct efa_xpu_cntr *) (uintptr_t) xpu_cntr.fid.prov_ctx;
	EXPECT_EQ(dev_cntr->xpu_cntr.fid.prov_ctx, xpu_cntr.fid.prov_ctx);
	EXPECT_EQ(dev_cntr->version, fi_version());
	EXPECT_TRUE(efa_xpu_cntr_compat(dev_cntr));
	EXPECT_EQ((void *) dev_cntr->value, value_dev);
	EXPECT_EQ((void *) dev_cntr->err_value, err_dev);
}

/*
 * The device struct is allocated before the rings are mapped, so a failure to
 * map must not leak it.
 */
TEST_F(EfaXpuExportTest, ep_export_releases_device_struct_when_mapping_fails)
{
	struct fid_xpu_ep xpu_ep = {};

	ASSERT_NO_FATAL_FAILURE(enable_ep());
	expect_qp_wqs_query(true);
	g_xpu.import_error = -FI_EIO;

	EXPECT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), -FI_EIO);

	EXPECT_EQ(xpu_ep.fid.prov_ctx, 0u);
	EXPECT_EQ(g_xpu.alloc_calls, 1u);
	EXPECT_TRUE(g_xpu.live.empty());
}

TEST_F(EfaXpuExportTest, ep_export_propagates_alloc_failure)
{
	struct fid_xpu_ep xpu_ep = {};

	ASSERT_NO_FATAL_FAILURE(enable_ep());
	expect_qp_wqs_query(true);
	g_xpu.alloc_error = -FI_ENOMEM;

	EXPECT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), -FI_ENOMEM);

	EXPECT_EQ(xpu_ep.fid.prov_ctx, 0u);
	EXPECT_EQ(g_xpu.import_calls, 0u);
	EXPECT_TRUE(g_xpu.live.empty());
}

struct EfaXpuQueryFailure {
	/* errno the efadv query reports */
	int efadv_error;
	int expected;
	const char *name;
};

class EfaXpuExportQueryFailureTest
	: public EfaXpuExportTest,
	  public testing::WithParamInterface<EfaXpuQueryFailure>
{
};

/*
 * A device that cannot describe its queues must be reported as unsupported
 * rather than as a generic error, so an application can fall back.
 */
TEST_P(EfaXpuExportQueryFailureTest, ep_export_maps_query_failure)
{
	const EfaXpuQueryFailure &param = GetParam();
	struct fid_xpu_ep xpu_ep = {};

	ASSERT_NO_FATAL_FAILURE(enable_ep());

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efadv_query_qp_wqs)
		.WillOnce(testing::Return(param.efadv_error));

	EXPECT_EQ(fi_ep_export_xpu(resource.ep, 0, &xpu_ep), param.expected);

	EXPECT_EQ(xpu_ep.fid.prov_ctx, 0u);
	/* Nothing was allocated before the query, so nothing can leak. */
	EXPECT_EQ(g_xpu.alloc_calls, 0u);
	EXPECT_TRUE(g_xpu.live.empty());
}

INSTANTIATE_TEST_SUITE_P(
	, EfaXpuExportQueryFailureTest,
	Values(EfaXpuQueryFailure{ EOPNOTSUPP, -FI_EOPNOTSUPP, "eopnotsupp" },
	       EfaXpuQueryFailure{ EINVAL, -FI_EINVAL, "einval" },
	       EfaXpuQueryFailure{ EIO, -FI_EINVAL, "other_errno" }),
	[](const testing::TestParamInfo<EfaXpuQueryFailure> &info) {
		return std::string(info.param.name);
	});

#endif /* HAVE_EFA_XPU */

/*
 * Device-side queue protocol.
 *
 * The device header is compiled here as host code, so the lock-free posting
 * protocol, the phase bookkeeping and the cooperative scopes can be driven
 * directly against stand-in rings rather than from a kernel. A host build of the
 * header is a single-threaded group of one, which is enough to pin what a
 * poster does with the cursors, which slot it writes and when it rings the
 * doorbell.
 */
class EfaXpuDeviceTest : public Test
{
	protected:
	struct fid_xpu_ep ep_handle = {};
	struct fid_xpu_cq cq_handle = {};
	struct efa_xpu_ep dev_ep = {};
	struct efa_xpu_cq dev_cq = {};
	std::vector<uint8_t> sq_ring;
	std::vector<uint8_t> rq_ring;
	std::vector<uint8_t> cq_ring;
	uint32_t sq_db = 0;
	uint32_t rq_db = 0;
	uint64_t nic_consumed = 0;
	uint8_t data_buf[64] = {};
	struct efa_xpu_desc desc = {};
	struct efa_xpu_peer peer = {};

	void SetUp() override
	{
		sq_ring.assign(XPU_TEST_SQ_ENTRIES * XPU_TEST_SQ_ENTRY_SIZE, 0);
		rq_ring.assign(XPU_TEST_RQ_ENTRIES * XPU_TEST_RQ_ENTRY_SIZE, 0);
		cq_ring.assign(XPU_TEST_CQ_ENTRIES * XPU_TEST_CQ_ENTRY_SIZE, 0);

		dev_ep.version = fi_version();
		dev_ep.sq.queue_mask = XPU_TEST_SQ_ENTRIES - 1;
		dev_ep.sq.queue_size_shift = XPU_TEST_SQ_SHIFT;
		dev_ep.sq.max_batch = XPU_TEST_SQ_MAX_BATCH;
		dev_ep.sq.entry_size = XPU_TEST_SQ_ENTRY_SIZE;
		dev_ep.sq.buf = sq_ring.data();
		dev_ep.sq.db = &sq_db;
		dev_ep.rq.queue_mask = XPU_TEST_RQ_ENTRIES - 1;
		dev_ep.rq.queue_size_shift = XPU_TEST_RQ_SHIFT;
		dev_ep.rq.max_batch = XPU_TEST_RQ_MAX_BATCH;
		dev_ep.rq.entry_size = XPU_TEST_RQ_ENTRY_SIZE;
		dev_ep.rq.buf = rq_ring.data();
		dev_ep.rq.db = &rq_db;
		dev_ep.sq_size = XPU_TEST_SQ_ENTRIES;
		dev_ep.local_cntr = &nic_consumed;
		ep_handle.fid.prov_ctx = (uint64_t) (uintptr_t) &dev_ep;

		dev_cq.version = fi_version();
		dev_cq.queue_mask = XPU_TEST_CQ_ENTRIES - 1;
		dev_cq.queue_size_shift = XPU_TEST_CQ_SHIFT;
		dev_cq.entry_size = XPU_TEST_CQ_ENTRY_SIZE;
		dev_cq.init_phase = 1;
		dev_cq.buf = cq_ring.data();
		/*
		 * The widest format, so one fixture can check every field a
		 * completion carries; the narrower formats are set per test.
		 */
		dev_cq.format = FI_CQ_FORMAT_DATA;
		dev_cq.user_entry_size = sizeof(struct fi_cq_data_entry);
		cq_handle.fid.prov_ctx = (uint64_t) (uintptr_t) &dev_cq;
	}

	struct efa_io_tx_wqe wqe_at(uint32_t slot) const
	{
		struct efa_io_tx_wqe wqe;

		memcpy(&wqe,
		       sq_ring.data() +
			       (slot & dev_ep.sq.queue_mask) * sizeof(struct efa_io_tx_wqe),
		       sizeof(wqe));
		return wqe;
	}

	struct efa_io_rx_desc rqe_at(uint32_t slot) const
	{
		struct efa_io_rx_desc rqe;

		memcpy(&rqe,
		       rq_ring.data() + (slot & dev_ep.rq.queue_mask) *
						sizeof(struct efa_io_rx_desc),
		       sizeof(rqe));
		return rqe;
	}

	void write_cqe(uint32_t slot, int phase)
	{
		cq_ring[(slot & dev_cq.queue_mask) * dev_cq.entry_size + 3] =
			phase ? EFA_IO_CDESC_COMMON_PHASE_MASK : 0;
	}

	uint8_t *cqe_at(uint32_t slot)
	{
		return cq_ring.data() +
		       (slot & dev_cq.queue_mask) * dev_cq.entry_size;
	}

	static uint8_t cqe_flags(int phase, enum efa_io_queue_type q_type,
				 enum efa_io_send_op_type op_type)
	{
		return (uint8_t) ((phase ? EFA_IO_CDESC_COMMON_PHASE_MASK : 0) |
				  (q_type << 1) | (op_type << 4));
	}

	/* A send-queue completion carrying a 64-bit request ID. */
	void write_tx_cqe(uint32_t slot, int phase,
			  enum efa_io_send_op_type op_type, uint64_t req_id,
			  uint8_t status = 0)
	{
		struct efa_io_tx_cdesc tx = {};

		tx.common.req_id = (uint16_t) req_id;
		tx.common.status = status;
		tx.common.flags = cqe_flags(phase, EFA_IO_SEND_QUEUE, op_type);
		tx.req_id_ex.w[0] = (uint16_t) (req_id >> 16);
		tx.req_id_ex.w[1] = (uint16_t) (req_id >> 32);
		tx.req_id_ex.w[2] = (uint16_t) (req_id >> 48);
		memcpy(cqe_at(slot), &tx, sizeof(tx));
	}

	void write_rx_cqe(uint32_t slot, int phase,
			  enum efa_io_send_op_type op_type, uint16_t req_id,
			  uint32_t length, uint32_t imm = 0,
			  bool unsolicited = false)
	{
		struct efa_io_rx_cdesc_ex rx = {};

		rx.base.common.req_id = req_id;
		rx.base.common.flags =
			cqe_flags(phase, EFA_IO_RECV_QUEUE, op_type);
		if (imm)
			rx.base.common.flags |= EFA_IO_CDESC_COMMON_HAS_IMM_MASK;
		if (unsolicited)
			rx.base.common.flags |=
				EFA_IO_CDESC_COMMON_UNSOLICITED_MASK;
		rx.base.length = (uint16_t) length;
		rx.base.imm = imm;
		rx.u.rdma_write.length_hi = (uint16_t) (length >> 16);
		memcpy(cqe_at(slot), &rx, sizeof(rx));
	}

	/* Put the SQ where it stands after a full pass the NIC has consumed. */
	void fill_sq(uint64_t consumed)
	{
		dev_ep.sq.pc = XPU_TEST_SQ_ENTRIES;
		dev_ep.sq.released = XPU_TEST_SQ_ENTRIES;
		dev_ep.sq.db_rung = XPU_TEST_SQ_ENTRIES;
		nic_consumed = consumed;
	}

	/*
	 * A group of one thread, which is what FI_XPU_WORK_ITEM is and what a
	 * host build collapses every scope to.
	 */
	struct efa_xpu_group solo_group()
	{
		struct efa_xpu_group g = {};

		efa_xpu_group_enter(&g, FI_XPU_WORK_ITEM);
		return g;
	}

	/*
	 * One rank of a group of @size, to be driven a rank at a time.
	 *
	 * There are no threads here to form a real group from - every device
	 * intrinsic collapses to a group of one in a host build - so a group of
	 * @size is emulated by making the same call once per rank. The scope is
	 * FI_XPU_WORK_GROUP because that is the one whose broadcast a host build
	 * leaves alone, so @base - the run of slots the leader already claimed -
	 * reaches the other ranks the way a real broadcast would.
	 *
	 * The barriers are no-ops here, so what this covers is the slot
	 * arithmetic and the doorbell bookkeeping: which slot each rank writes,
	 * and that one claim and one doorbell serve all of them. It does not
	 * cover the write-before-ring ordering the barriers exist to enforce.
	 */
	struct efa_xpu_group group_rank(int rank, int size, uint32_t base = 0)
	{
		struct efa_xpu_group g = {};

		g.scope = FI_XPU_WORK_GROUP;
		g.rank = rank;
		g.size = size;
		g.leader = (rank == 0);
		g.bcast = base;
		return g;
	}

	int post_one(uint64_t flags)
	{
		struct efa_io_tx_wqe wqe = {};
		struct efa_xpu_group g = solo_group();

		return efa_xpu_post_wqe(&dev_ep, &g, &wqe, flags);
	}

	/* Post one send descriptor per rank of a group of @size. */
	uint32_t post_group(int size, uint64_t flags)
	{
		struct efa_xpu_group lead = group_rank(0, size);
		struct efa_io_tx_wqe wqe = {};
		uint32_t base;

		wqe.meta.length = 1;
		EXPECT_EQ(efa_xpu_post_wqe(&dev_ep, &lead, &wqe, flags), 0);
		base = lead.bcast;

		for (int r = 1; r < size; r++) {
			struct efa_xpu_group g = group_rank(r, size, base);
			struct efa_io_tx_wqe w = {};

			w.meta.length = 1;
			EXPECT_EQ(efa_xpu_post_wqe(&dev_ep, &g, &w, flags), 0);
		}

		return base;
	}

	/* Post one receive descriptor per rank of a group of @size. */
	uint32_t recv_group(int size, uint64_t flags)
	{
		struct efa_xpu_group lead = group_rank(0, size);
		uint32_t base;

		EXPECT_EQ(efa_xpu_post_rqe(&dev_ep, &lead, data_buf,
					   sizeof(data_buf), desc.lkey, flags),
			  0);
		base = lead.bcast;

		for (int r = 1; r < size; r++) {
			struct efa_xpu_group g = group_rank(r, size, base);

			EXPECT_EQ(efa_xpu_post_rqe(&dev_ep, &g, data_buf,
						   sizeof(data_buf), desc.lkey,
						   flags),
				  0);
		}

		return base;
	}
};

TEST_F(EfaXpuDeviceTest, posting_hands_out_successive_slots)
{
	for (uint32_t i = 0; i < 3; i++)
		ASSERT_EQ(post_one(FI_MORE), 0);

	EXPECT_EQ(dev_ep.sq.pc, 3u);
	EXPECT_EQ(wqe_at(0).meta.req_id, 0);
	EXPECT_EQ(wqe_at(1).meta.req_id, 1);
	EXPECT_EQ(wqe_at(2).meta.req_id, 2);
}

TEST_F(EfaXpuDeviceTest, wq_slot_phase_flips_on_every_pass)
{
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, 0), 0);
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, XPU_TEST_SQ_ENTRIES - 1), 0);
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, XPU_TEST_SQ_ENTRIES), 1);
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, 2 * XPU_TEST_SQ_ENTRIES), 0);

	dev_ep.sq.init_phase = 1;
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, 0), 1);
	EXPECT_EQ(efa_xpu_wq_slot_phase(&dev_ep.sq, XPU_TEST_SQ_ENTRIES), 0);
}

TEST_F(EfaXpuDeviceTest, post_wqe_writes_the_claimed_slot_and_rings_the_doorbell)
{
	struct efa_io_tx_wqe wqe = {};
	struct efa_xpu_group g = solo_group();

	wqe.meta.length = 1;

	EXPECT_EQ(efa_xpu_post_wqe(&dev_ep, &g, &wqe, 0), 0);

	/* No 64-bit request ID on this endpoint, so it carries the slot. */
	EXPECT_EQ(wqe_at(0).meta.req_id, 0);
	EXPECT_EQ(wqe_at(0).meta.length, 1);
	EXPECT_EQ(wqe_at(0).meta.ctrl2 & EFA_IO_TX_META_DESC_PHASE_MASK, 0);
	EXPECT_EQ(sq_db, 1u);
	EXPECT_EQ(dev_ep.sq.db_rung, 1u);
	EXPECT_EQ(dev_ep.sq.released, 1u);
	EXPECT_EQ(dev_ep.submitted_count, 1u);
}

TEST_F(EfaXpuDeviceTest, post_wqe_holds_the_doorbell_back_for_fi_more)
{
	for (uint32_t i = 0; i < 3; i++)
		ASSERT_EQ(post_one(FI_MORE), 0);

	EXPECT_EQ(sq_db, 0u);
	EXPECT_EQ(dev_ep.sq.released, 3u);
	EXPECT_EQ(dev_ep.submitted_count, 0u);

	ASSERT_EQ(post_one(0), 0);

	EXPECT_EQ(sq_db, 4u);
	EXPECT_EQ(dev_ep.sq.db_rung, 4u);
	EXPECT_EQ(dev_ep.submitted_count, 4u);
}

TEST_F(EfaXpuDeviceTest, post_wqe_rings_the_doorbell_once_a_batch_has_gathered)
{
	for (uint32_t i = 0; i < XPU_TEST_SQ_MAX_BATCH; i++)
		ASSERT_EQ(post_one(FI_MORE), 0);

	EXPECT_EQ(sq_db, XPU_TEST_SQ_MAX_BATCH);
	EXPECT_EQ(dev_ep.submitted_count, XPU_TEST_SQ_MAX_BATCH);
}

TEST_F(EfaXpuDeviceTest, post_wqe_stamps_the_phase_of_the_second_pass)
{
	fill_sq(XPU_TEST_SQ_ENTRIES);

	ASSERT_EQ(post_one(0), 0);

	EXPECT_EQ(dev_ep.sq.pc, XPU_TEST_SQ_ENTRIES + 1);
	EXPECT_TRUE(wqe_at(0).meta.ctrl2 & EFA_IO_TX_META_DESC_PHASE_MASK);
	EXPECT_EQ(sq_db, XPU_TEST_SQ_ENTRIES + 1);
}

/*
 * A slot one pass ahead of the NIC is the last one that still fits in the ring,
 * so it must be accepted rather than waited on.
 */
TEST_F(EfaXpuDeviceTest, post_wqe_takes_the_last_slot_the_ring_has_room_for)
{
	fill_sq(1);

	ASSERT_EQ(post_one(0), 0);

	EXPECT_EQ(sq_db, XPU_TEST_SQ_ENTRIES + 1);
	EXPECT_EQ(dev_ep.sq.released, XPU_TEST_SQ_ENTRIES + 1);
}

/*
 * What a group scope is for: a group of N threads issues N operations, one per
 * thread, the same as N separate work-item calls would. The group only makes
 * them cheaper - one claim and one doorbell between them.
 */
TEST_F(EfaXpuDeviceTest, a_group_posts_one_descriptor_per_thread)
{
	const int size = 4;
	uint32_t base = post_group(size, 0);

	EXPECT_EQ(base, 0u);
	EXPECT_EQ(dev_ep.sq.pc, (uint32_t) size);

	for (int r = 0; r < size; r++) {
		SCOPED_TRACE(r);
		EXPECT_EQ(wqe_at(base + r).meta.length, 1);
		EXPECT_EQ(wqe_at(base + r).meta.req_id, base + r);
	}

	EXPECT_EQ(sq_db, (uint32_t) size);
	EXPECT_EQ(dev_ep.sq.released, (uint32_t) size);
	EXPECT_EQ(dev_ep.submitted_count, (uint64_t) size);
}

/* One atomic add for the group, so its slots are one contiguous run. */
TEST_F(EfaXpuDeviceTest, a_group_claims_its_slots_in_one_contiguous_run)
{
	EXPECT_EQ(post_group(3, 0), 0u);
	EXPECT_EQ(dev_ep.sq.pc, 3u);

	EXPECT_EQ(post_group(5, 0), 3u);
	EXPECT_EQ(dev_ep.sq.pc, 8u);
}

/*
 * A group can be larger than the number of descriptors the hardware stages
 * between doorbells, so it is posted in chunks of at most max_batch - and no
 * thread's operation may be dropped on the way.
 */
TEST_F(EfaXpuDeviceTest, a_group_larger_than_max_batch_is_posted_in_chunks)
{
	const int size = 2 * (int) XPU_TEST_SQ_MAX_BATCH;

	ASSERT_EQ(post_group(size, 0), 0u);

	for (int r = 0; r < size; r++) {
		SCOPED_TRACE(r);
		EXPECT_EQ(wqe_at(r).meta.req_id, r);
		EXPECT_EQ(wqe_at(r).meta.length, 1);
	}

	EXPECT_EQ(sq_db, (uint32_t) size);
	EXPECT_EQ(dev_ep.sq.released, (uint32_t) size);
	EXPECT_EQ(dev_ep.submitted_count, (uint64_t) size);
}

/*
 * A group whose run straddles the end of the ring wraps, and the phase bit each
 * rank stamps follows its own slot rather than the group's first.
 */
TEST_F(EfaXpuDeviceTest, a_group_that_wraps_stamps_the_phase_per_rank)
{
	const uint32_t first = XPU_TEST_SQ_ENTRIES - 2;

	dev_ep.sq.pc = first;
	dev_ep.sq.released = first;
	dev_ep.sq.db_rung = first;
	nic_consumed = first;

	ASSERT_EQ(post_group(4, 0), first);

	EXPECT_EQ(wqe_at(XPU_TEST_SQ_ENTRIES - 2).meta.ctrl2 &
			  EFA_IO_TX_META_DESC_PHASE_MASK,
		  0);
	EXPECT_EQ(wqe_at(XPU_TEST_SQ_ENTRIES - 1).meta.ctrl2 &
			  EFA_IO_TX_META_DESC_PHASE_MASK,
		  0);
	EXPECT_TRUE(wqe_at(0).meta.ctrl2 & EFA_IO_TX_META_DESC_PHASE_MASK);
	EXPECT_TRUE(wqe_at(1).meta.ctrl2 & EFA_IO_TX_META_DESC_PHASE_MASK);
}

/* A group defers as one: nothing is rung until the batch has gathered. */
TEST_F(EfaXpuDeviceTest, a_group_posting_with_fi_more_holds_the_doorbell_back)
{
	ASSERT_EQ(post_group(3, FI_MORE), 0u);

	EXPECT_EQ(sq_db, 0u);
	EXPECT_EQ(dev_ep.sq.released, 3u);
	EXPECT_EQ(dev_ep.submitted_count, 0u);

	ASSERT_EQ(post_one(0), 0);

	EXPECT_EQ(sq_db, 4u);
	EXPECT_EQ(dev_ep.submitted_count, 4u);
}

/* The receive queue posts one descriptor per thread the same way. */
TEST_F(EfaXpuDeviceTest, a_group_posts_one_receive_descriptor_per_thread)
{
	const int size = 4;
	uint32_t base;

	desc.lkey = 0x21;
	base = recv_group(size, 0);

	EXPECT_EQ(base, 0u);
	EXPECT_EQ(dev_ep.rq.pc, (uint32_t) size);

	for (int r = 0; r < size; r++) {
		SCOPED_TRACE(r);
		EXPECT_EQ(rqe_at(base + r).req_id, base + r);
		EXPECT_EQ(rqe_at(base + r).length, sizeof(data_buf));
		EXPECT_EQ(rqe_at(base + r).lkey_ctrl &
				  EFA_IO_RX_DESC_LKEY_MASK,
			  0x21u);
	}

	EXPECT_EQ(rq_db, (uint32_t) size);
	EXPECT_EQ(dev_ep.rq.released, (uint32_t) size);
}

TEST_F(EfaXpuDeviceTest, send_posts_one_descriptor_to_the_send_queue)
{
	peer.ahn = 3;
	peer.remote_qpn = 5;
	peer.remote_qkey = 7;
	desc.lkey = 0xabc;

	EXPECT_EQ(efa_xpu_send(&ep_handle, data_buf, sizeof(data_buf), &desc, 0,
			       &peer, NULL, 0, FI_XPU_WORK_ITEM),
		  0);

	EXPECT_EQ(dev_ep.sq.pc, 1u);
	EXPECT_EQ(sq_db, 1u);
	EXPECT_EQ(wqe_at(0).meta.ctrl1 & EFA_IO_TX_META_DESC_OP_TYPE_MASK,
		  EFA_IO_SEND);
	EXPECT_EQ(wqe_at(0).meta.ctrl1 & EFA_IO_TX_META_DESC_HAS_IMM_MASK, 0);
	EXPECT_EQ(wqe_at(0).meta.ah, 3);
	EXPECT_EQ(wqe_at(0).meta.dest_qp_num, 5);
	EXPECT_EQ(wqe_at(0).meta.qkey, 7u);
	EXPECT_EQ(wqe_at(0).meta.req_id, 0);
	EXPECT_EQ(wqe_at(0).data.sgl[0].lkey, 0xabcu);
	EXPECT_EQ(wqe_at(0).data.sgl[0].length, 64u);
	EXPECT_EQ(wqe_at(0).data.sgl[0].buf_addr_lo,
		  (uint32_t) ((uintptr_t) data_buf & 0xFFFFFFFF));
}

TEST_F(EfaXpuDeviceTest, send_with_remote_cq_data_carries_the_immediate)
{
	EXPECT_EQ(efa_xpu_send(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       0x12345678, &peer, NULL, FI_REMOTE_CQ_DATA,
			       FI_XPU_WORK_ITEM),
		  0);

	EXPECT_TRUE(wqe_at(0).meta.ctrl1 & EFA_IO_TX_META_DESC_HAS_IMM_MASK);
	EXPECT_EQ(wqe_at(0).meta.immediate_data, 0x12345678u);
}

TEST_F(EfaXpuDeviceTest, write_describes_both_ends_of_the_transfer)
{
	desc.lkey = 0xabc;

	EXPECT_EQ(efa_xpu_write(&ep_handle, data_buf, sizeof(data_buf), &desc, 0,
				&peer, 0x1000, 0xdef, NULL, 0,
				FI_XPU_WORK_ITEM),
		  0);

	EXPECT_EQ(wqe_at(0).meta.ctrl1 & EFA_IO_TX_META_DESC_OP_TYPE_MASK,
		  EFA_IO_RDMA_WRITE);
	EXPECT_EQ(wqe_at(0).data.rdma_req.remote_mem.rkey, 0xdefu);
	EXPECT_EQ(wqe_at(0).data.rdma_req.remote_mem.buf_addr_lo, 0x1000u);
	EXPECT_EQ(wqe_at(0).data.rdma_req.local_mem[0].lkey, 0xabcu);
	EXPECT_EQ(wqe_at(0).data.rdma_req.local_mem[0].length, 64u);
}

TEST_F(EfaXpuDeviceTest, read_posts_an_rdma_read_descriptor)
{
	EXPECT_EQ(efa_xpu_read(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       &peer, 0x1000, 0xdef, NULL, 0,
			       FI_XPU_WORK_ITEM),
		  0);

	EXPECT_EQ(wqe_at(0).meta.ctrl1 & EFA_IO_TX_META_DESC_OP_TYPE_MASK,
		  EFA_IO_RDMA_READ);
	EXPECT_EQ(wqe_at(0).data.rdma_req.remote_mem.buf_addr_lo, 0x1000u);
}

TEST_F(EfaXpuDeviceTest, recv_posts_to_the_receive_queue_and_rings_its_doorbell)
{
	desc.lkey = 0xabc;

	EXPECT_EQ(efa_xpu_recv(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       NULL, NULL, 0, FI_XPU_WORK_ITEM),
		  0);

	EXPECT_EQ(dev_ep.rq.pc, 1u);
	EXPECT_EQ(dev_ep.rq.released, 1u);
	EXPECT_EQ(rq_db, 1u);
	EXPECT_EQ(sq_db, 0u);
	EXPECT_EQ(rqe_at(0).req_id, 0);
	EXPECT_EQ(rqe_at(0).length, 64u);
	EXPECT_EQ(rqe_at(0).lkey_ctrl,
		  0xabcu | EFA_IO_RX_DESC_FIRST_MASK | EFA_IO_RX_DESC_LAST_MASK);
	/* The send queue's doorbell must stay untouched by a receive. */
	EXPECT_EQ(dev_ep.submitted_count, 0u);
}

TEST_F(EfaXpuDeviceTest, recv_holds_the_doorbell_back_for_fi_more)
{
	EXPECT_EQ(efa_xpu_recv(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       NULL, NULL, FI_MORE, FI_XPU_WORK_ITEM),
		  0);
	EXPECT_EQ(rq_db, 0u);
	EXPECT_EQ(dev_ep.rq.released, 1u);

	EXPECT_EQ(efa_xpu_recv(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       NULL, NULL, 0, FI_XPU_WORK_ITEM),
		  0);
	EXPECT_EQ(rq_db, 2u);
}

TEST_F(EfaXpuDeviceTest, cq_read_reports_eagain_on_an_untouched_ring)
{
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);
	EXPECT_EQ(dev_cq.cc, 0u);
}

TEST_F(EfaXpuDeviceTest, cq_read_takes_one_completion_and_advances_the_cursor)
{
	write_cqe(0, 1);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ(dev_cq.cc, 1u);
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);
	EXPECT_EQ(dev_cq.cc, 1u);
}

/*
 * The expected phase follows from the cursor, so a slot left behind by the
 * previous pass must not be reported again.
 */
TEST_F(EfaXpuDeviceTest, cq_read_expects_the_opposite_phase_after_a_wrap)
{
	dev_cq.cc = XPU_TEST_CQ_ENTRIES;
	write_cqe(0, 1);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);

	write_cqe(0, 0);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ(dev_cq.cc, XPU_TEST_CQ_ENTRIES + 1);
}

/*
 * A poll fills the caller's buffer in the format the CQ was opened with, so a
 * kernel reads the same entry out of fi_xpu_cq_read() that a host thread reads
 * out of fi_cq_read() on the same CQ.
 */
TEST_F(EfaXpuDeviceTest, cq_read_fills_a_send_completion_from_its_request_id)
{
	struct fi_cq_data_entry entry = {};

	write_tx_cqe(0, 1, EFA_IO_SEND, 0xfeedfacecafebeefULL);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) entry.op_context,
		  0xfeedfacecafebeefULL);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_MSG | FI_SEND));
	EXPECT_EQ(entry.len, 0u);
}

TEST_F(EfaXpuDeviceTest, cq_read_reports_the_rma_direction_of_a_completion)
{
	struct fi_cq_data_entry entry[2] = {};

	write_tx_cqe(0, 1, EFA_IO_RDMA_WRITE, 0x1111);
	write_tx_cqe(1, 1, EFA_IO_RDMA_READ, 0x2222);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, entry, 2, FI_XPU_WORK_ITEM), 2);
	EXPECT_EQ(entry[0].flags, (uint64_t) (FI_RMA | FI_WRITE));
	EXPECT_EQ(entry[1].flags, (uint64_t) (FI_RMA | FI_READ));
}

/*
 * A receive completion reports what arrived. Its length has 16 more bits in the
 * extended completion, which is the only way a write of more than 64KiB reports
 * its true length.
 */
TEST_F(EfaXpuDeviceTest, cq_read_fills_a_receive_completion_with_its_length)
{
	struct fi_cq_data_entry entry = {};

	write_rx_cqe(0, 1, EFA_IO_SEND, 9, 4096);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) entry.op_context, 9u);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_MSG | FI_RECV));
	EXPECT_EQ(entry.len, 4096u);
}

TEST_F(EfaXpuDeviceTest, cq_read_reports_a_remote_write_with_its_immediate)
{
	struct fi_cq_data_entry entry = {};

	write_rx_cqe(0, 1, EFA_IO_RDMA_WRITE, 0, 0x30000, 0xabcd, true);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ(entry.flags, (uint64_t) (FI_RMA | FI_REMOTE_WRITE |
					   FI_REMOTE_CQ_DATA));
	EXPECT_EQ(entry.len, 0x30000u);
	EXPECT_EQ(entry.data, 0xabcdu);
	/* Nothing was posted for it, so there is no context to report. */
	EXPECT_EQ(entry.op_context, nullptr);
}

TEST_F(EfaXpuDeviceTest, cq_read_stops_at_the_count_it_was_given)
{
	struct fi_cq_data_entry entry[4] = {};

	for (uint32_t i = 0; i < 4; i++)
		write_tx_cqe(i, 1, EFA_IO_SEND, 0x100 + i);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, entry, 3, FI_XPU_WORK_ITEM), 3);
	EXPECT_EQ(dev_cq.cc, 3u);
	EXPECT_EQ((uint64_t) (uintptr_t) entry[0].op_context, 0x100u);
	EXPECT_EQ((uint64_t) (uintptr_t) entry[2].op_context, 0x102u);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, entry, 3, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) entry[0].op_context, 0x103u);
	EXPECT_EQ(dev_cq.cc, 4u);
}

TEST_F(EfaXpuDeviceTest, cq_read_reports_what_is_ready_rather_than_the_count)
{
	struct fi_cq_data_entry entry[4] = {};

	write_tx_cqe(0, 1, EFA_IO_SEND, 0x1);
	write_tx_cqe(1, 1, EFA_IO_SEND, 0x2);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, entry, 4, FI_XPU_WORK_ITEM), 2);
	EXPECT_EQ(dev_cq.cc, 2u);
}

/*
 * A narrower format is a prefix of a wider one, so a poll of a CONTEXT or MSG
 * format CQ must write only that prefix - anything beyond the entry belongs to
 * the caller.
 */
TEST_F(EfaXpuDeviceTest, cq_read_writes_only_as_much_as_the_format_asks_for)
{
	struct {
		struct fi_cq_entry entry;
		uint64_t guard;
	} context_buf = { {}, 0x5a5a5a5a5a5a5a5aULL };
	struct {
		struct fi_cq_msg_entry entry;
		uint64_t guard;
	} msg_buf = { {}, 0x5a5a5a5a5a5a5a5aULL };

	dev_cq.format = FI_CQ_FORMAT_CONTEXT;
	dev_cq.user_entry_size = sizeof(struct fi_cq_entry);
	write_rx_cqe(0, 1, EFA_IO_SEND, 3, 64);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, &context_buf, 1,
				  FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) context_buf.entry.op_context, 3u);
	EXPECT_EQ(context_buf.guard, 0x5a5a5a5a5a5a5a5aULL);

	dev_cq.format = FI_CQ_FORMAT_MSG;
	dev_cq.user_entry_size = sizeof(struct fi_cq_msg_entry);
	write_rx_cqe(1, 1, EFA_IO_SEND, 4, 128);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, &msg_buf, 1, FI_XPU_WORK_ITEM),
		  1);
	EXPECT_EQ((uint64_t) (uintptr_t) msg_buf.entry.op_context, 4u);
	EXPECT_EQ(msg_buf.entry.len, 128u);
	EXPECT_EQ(msg_buf.guard, 0x5a5a5a5a5a5a5a5aULL);
}

/*
 * A failing completion is never reported as a successful entry. It is staged,
 * and the poll that found it says so with -FI_EAVAIL, exactly as a host
 * fi_cq_read() does.
 */
TEST_F(EfaXpuDeviceTest, cq_read_stages_a_failing_completion_for_readerr)
{
	struct fi_cq_data_entry entry = {};
	struct fi_cq_err_entry err = {};

	write_tx_cqe(0, 1, EFA_IO_SEND, 0x77, 8 /* some error status */);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM),
		  -FI_EAVAIL);
	EXPECT_EQ(entry.op_context, nullptr);
	EXPECT_EQ(dev_cq.cc, 1u);

	ASSERT_EQ(efa_xpu_cq_readerr(&cq_handle, &err, 0, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) err.op_context, 0x77u);
	EXPECT_EQ(err.flags, (uint64_t) (FI_MSG | FI_SEND));
	EXPECT_EQ(err.err, FI_EIO);
	EXPECT_EQ(err.prov_errno, 8);

	/* The staged error is gone, and polling resumes. */
	EXPECT_EQ(efa_xpu_cq_readerr(&cq_handle, &err, 0, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);
}

/* Completions that did succeed are reported before the error behind them. */
TEST_F(EfaXpuDeviceTest, cq_read_reports_successes_ahead_of_an_error)
{
	struct fi_cq_data_entry entry[4] = {};

	write_tx_cqe(0, 1, EFA_IO_SEND, 0x1);
	write_tx_cqe(1, 1, EFA_IO_SEND, 0x2, 8);

	ASSERT_EQ(efa_xpu_cq_read(&cq_handle, entry, 4, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ((uint64_t) (uintptr_t) entry[0].op_context, 0x1u);
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, entry, 4, FI_XPU_WORK_ITEM),
		  -FI_EAVAIL);
}

/*
 * There is room for one staged error, so a poll reports the staged one rather
 * than looking at the queue again and overwriting an error the application has
 * not read yet.
 */
TEST_F(EfaXpuDeviceTest, a_staged_error_is_reported_before_the_queue_again)
{
	struct fi_cq_data_entry entry = {};
	struct fi_cq_err_entry err = {};

	write_tx_cqe(0, 1, EFA_IO_SEND, 0x1, 8);
	write_tx_cqe(1, 1, EFA_IO_SEND, 0x2, 9);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM),
		  -FI_EAVAIL);
	EXPECT_EQ(dev_cq.err_dropped, 0u);

	/* The staged error is reported before the queue is looked at again. */
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM),
		  -FI_EAVAIL);
	EXPECT_EQ(dev_cq.cc, 1u);

	ASSERT_EQ(efa_xpu_cq_readerr(&cq_handle, &err, 0, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ(err.prov_errno, 8);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, &entry, 1, FI_XPU_WORK_ITEM),
		  -FI_EAVAIL);
	ASSERT_EQ(efa_xpu_cq_readerr(&cq_handle, &err, 0, FI_XPU_WORK_ITEM), 1);
	EXPECT_EQ(err.prov_errno, 9);
}

/* A poll with no buffer consumes the completions without reporting them. */
TEST_F(EfaXpuDeviceTest, cq_read_without_a_buffer_only_advances_the_cursor)
{
	write_tx_cqe(0, 1, EFA_IO_SEND, 0x1);
	write_tx_cqe(1, 1, EFA_IO_SEND, 0x2);

	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 8, FI_XPU_WORK_ITEM), 2);
	EXPECT_EQ(dev_cq.cc, 2u);
}

TEST_F(EfaXpuDeviceTest, cq_readerr_reports_eagain_when_nothing_has_failed)
{
	struct fi_cq_err_entry err = {};

	EXPECT_EQ(efa_xpu_cq_readerr(&cq_handle, &err, 0, FI_XPU_WORK_ITEM),
		  -FI_EAGAIN);
}

TEST_F(EfaXpuDeviceTest, group_enter_accepts_the_scopes_efa_can_make_collective)
{
	for (int scope : { FI_XPU_WORK_ITEM, FI_XPU_SUBGROUP,
			   FI_XPU_WORK_GROUP }) {
		struct efa_xpu_group g = {};

		EXPECT_EQ(efa_xpu_group_enter(&g, scope), 0) << "scope "
							     << scope;
		EXPECT_EQ(g.scope, scope);
		EXPECT_TRUE(g.leader);
		EXPECT_EQ(efa_xpu_group_leave(&g, 7), 7);
	}
}

TEST_F(EfaXpuDeviceTest, group_enter_refuses_a_scope_it_cannot_make_collective)
{
	struct efa_xpu_group g = {};

	EXPECT_EQ(efa_xpu_group_enter(&g, FI_XPU_DEVICE), -FI_EOPNOTSUPP);
	EXPECT_FALSE(g.leader);

	EXPECT_EQ(efa_xpu_group_enter(&g, 0x7fff), -FI_EOPNOTSUPP);
	EXPECT_FALSE(g.leader);
}

/*
 * An unsupported scope has to be refused before anything is claimed, or a
 * kernel would leave a hole in the queue behind.
 */
TEST_F(EfaXpuDeviceTest, an_unsupported_scope_is_refused_without_posting)
{
	EXPECT_EQ(efa_xpu_send(&ep_handle, data_buf, sizeof(data_buf), &desc, 0,
			       &peer, NULL, 0, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_recv(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       NULL, NULL, 0, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_write(&ep_handle, data_buf, sizeof(data_buf), &desc, 0,
				&peer, 0x1000, 0xdef, NULL, 0, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_read(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       &peer, 0x1000, 0xdef, NULL, 0, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_cq_readerr(&cq_handle, NULL, 0, FI_XPU_DEVICE),
		  -FI_EOPNOTSUPP);

	EXPECT_EQ(dev_ep.sq.pc, 0u);
	EXPECT_EQ(dev_ep.rq.pc, 0u);
	EXPECT_EQ(sq_db, 0u);
	EXPECT_EQ(rq_db, 0u);
	EXPECT_EQ(dev_cq.cc, 0u);
}

/*
 * A handle exported by a library older than the layouts this header describes
 * cannot be interpreted, so it is refused rather than misread.
 */
TEST_F(EfaXpuDeviceTest, a_handle_from_an_older_library_is_refused)
{
	dev_ep.version = FI_VERSION(2, 6);
	dev_cq.version = FI_VERSION(2, 6);

	EXPECT_EQ(efa_xpu_send(&ep_handle, data_buf, sizeof(data_buf), &desc, 0,
			       &peer, NULL, 0, FI_XPU_WORK_ITEM),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_recv(&ep_handle, data_buf, sizeof(data_buf), &desc,
			       NULL, NULL, 0, FI_XPU_WORK_ITEM),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_cq_read(&cq_handle, NULL, 1, FI_XPU_WORK_ITEM),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(efa_xpu_cq_readerr(&cq_handle, NULL, 0, FI_XPU_WORK_ITEM),
		  -FI_EOPNOTSUPP);
	EXPECT_EQ(dev_ep.sq.pc, 0u);
	EXPECT_EQ(dev_cq.cc, 0u);
}
