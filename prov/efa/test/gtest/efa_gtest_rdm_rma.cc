/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>

using testing::TestWithParam;
using testing::ValuesIn;

/*
 * The addr and len of an incoming RTW/RTR/RTA packet are supplied by the remote
 * peer, so efa_rdm_rma_verified_copy_iov() must reject every target region that
 * is not fully contained in the registered MR - including the ones that make
 * the bounds arithmetic wrap around.
 */
struct RmaVerifyCase {
	const char *name;
	/* offset of the peer-supplied addr relative to the start of the MR */
	ptrdiff_t offset;
	/* peer-supplied length */
	size_t len;
	int expected_ret;
};

static constexpr size_t kMrLen = 4096;

static const RmaVerifyCase kRmaVerifyCases[] = {
	{"whole_region", 0, kMrLen, 0},
	{"interior_region", 64, 64, 0},
	{"last_byte", kMrLen - 1, 1, 0},
	/* (addr + len) wraps to (addr - 1), which is below the MR end. */
	{"len_size_max", 0, SIZE_MAX, -FI_EACCES},
	/* (MR end - addr) wraps, so the space left in the MR looks huge. */
	{"addr_past_mr_end", kMrLen + 64, 64, -FI_EACCES},
	{"len_past_mr_end", 0, kMrLen + 1, -FI_EACCES},
	{"addr_before_mr_start", -64, 64, -FI_EACCES},
};

class EfaRdmRmaVerifyTest : public TestWithParam<RmaVerifyCase>
{
	protected:
	struct efa_resource resource = {};
	uint8_t *buf = nullptr;
	struct fid_mr *mr = nullptr;

	void SetUp() override
	{
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		buf = new uint8_t[kMrLen]();
		ASSERT_EQ(fi_mr_reg(resource.domain, buf, kMrLen,
				    FI_REMOTE_READ | FI_REMOTE_WRITE, 0, 0, 0,
				    &mr, nullptr),
			  0);
	}

	void TearDown() override
	{
		if (mr)
			fi_close(&mr->fid);
		delete[] buf;
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmRmaVerifyTest, enforces_mr_bounds)
{
	const RmaVerifyCase &c = GetParam();
	uint64_t addr = (uint64_t) (uintptr_t) buf + c.offset;
	struct iovec iov = {};
	void *desc = nullptr;
	int ret;

	ret = efa_test_rdm_rma_verified_copy_iov(resource.ep, addr, c.len,
						 fi_mr_key(mr), FI_REMOTE_WRITE,
						 &iov, &desc);
	EXPECT_EQ(ret, c.expected_ret);

	if (c.expected_ret) {
		/* A rejected region must not reach the caller's iov/desc. */
		EXPECT_EQ(iov.iov_base, nullptr);
		EXPECT_EQ(iov.iov_len, 0u);
		EXPECT_EQ(desc, nullptr);
	} else {
		EXPECT_EQ(iov.iov_base, (void *) (uintptr_t) addr);
		EXPECT_EQ(iov.iov_len, c.len);
		EXPECT_EQ(desc, fi_mr_desc(mr));
	}
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmRmaVerifyTest, ValuesIn(kRmaVerifyCases),
	[](const testing::TestParamInfo<RmaVerifyCase> &info) {
		return std::string(info.param.name);
	});
