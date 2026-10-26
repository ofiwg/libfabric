/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_pke_utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <string>

using testing::_;
using testing::Range;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::Values;
using testing::ValuesIn;
using testing::WithParamInterface;

class EfaRdmPkeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/* pke_release_cloned must release arbitrary length packet list correctly */
class EfaRdmPkeChainTest : public EfaRdmPkeTest,
			   public WithParamInterface<size_t>
{
};

TEST_P(EfaRdmPkeChainTest, release_cloned_frees_whole_chain)
{
	size_t n = GetParam();
	struct efa_rdm_pke *head;

	head = efa_test_pke_build_unexp_chain(resource.ep, n);
	ASSERT_EQ(head == nullptr, n == 0);
	ASSERT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), n);

	efa_test_pke_release_cloned(head);

	EXPECT_EQ(efa_test_ep_unexp_pool_outstanding(resource.ep), 0u);
}

INSTANTIATE_TEST_SUITE_P(ChainLengths, EfaRdmPkeChainTest, Range<size_t>(0, 6));

struct EfaCopyFromHmemCase {
	const char *name;
	size_t total_len;
	enum efa_test_copy_src src;
	bool cuda_api_permitted;
	bool expect_gdrcopy;
};

static constexpr size_t kTestMaxGdrcopyFromHmemSize = 4096;

class EfaRdmPkeCopyFromHmemTest : public TestWithParam<EfaCopyFromHmemCase>
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	size_t saved_max_gdrcopy_from_hmem_size = 0;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		saved_max_gdrcopy_from_hmem_size = efa_test_get_max_gdrcopy_from_hmem_size();
		efa_test_set_max_gdrcopy_from_hmem_size(kTestMaxGdrcopyFromHmemSize);
		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_set_max_gdrcopy_from_hmem_size(saved_max_gdrcopy_from_hmem_size);
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmPkeCopyFromHmemTest, selects_copy_method)
{
	const EfaCopyFromHmemCase &c = GetParam();
	struct efa_test_copy_from_hmem_result res;

	if (c.expect_gdrcopy) {
		EFA_EXPECT_CALL(mock_efa, ofi_dev_reg_copy_from_hmem_iov, _,
				EFA_TEST_COPY_HMEM_DATA_SIZE, FI_HMEM_CUDA,
				EFA_TEST_COPY_HMEM_HANDLE, _, 1u, 0u)
			.WillOnce(Return(EFA_TEST_COPY_HMEM_DATA_SIZE));
		EFA_EXPECT_CALL(mock_efa, ofi_copy_from_hmem_iov).Times(0);
	} else {
		bool host = c.src == EFA_TEST_COPY_SRC_HOST;

		EFA_EXPECT_CALL(mock_efa, ofi_copy_from_hmem_iov, _,
				EFA_TEST_COPY_HMEM_DATA_SIZE,
				host ? FI_HMEM_SYSTEM : FI_HMEM_CUDA,
				host ? 0u : EFA_TEST_COPY_HMEM_DEVICE, _, 1u, 0u)
			.WillOnce(Return(EFA_TEST_COPY_HMEM_DATA_SIZE));
		EFA_EXPECT_CALL(mock_efa, ofi_dev_reg_copy_from_hmem_iov)
			.Times(0);
	}

	efa_test_selects_copy_method(resource.ep, resource.av, c.total_len,
				     c.src, c.cuda_api_permitted,
				     &res);

	EXPECT_EQ(res.ret, 0);
	EXPECT_EQ(res.payload_size, EFA_TEST_COPY_HMEM_DATA_SIZE);
	EXPECT_TRUE(res.payload_in_wiredata);
}

static const EfaCopyFromHmemCase kCopyFromHmemCases[] = {
	{"gdrcopy_at_max_size", kTestMaxGdrcopyFromHmemSize,
	 EFA_TEST_COPY_SRC_CUDA_GDRCOPY, true, true},
	{"cudamemcpy_above_max_size", kTestMaxGdrcopyFromHmemSize + 1,
	 EFA_TEST_COPY_SRC_CUDA_GDRCOPY, true, false},
	{"gdrcopy_above_max_size_without_cuda_api",
	 kTestMaxGdrcopyFromHmemSize + 1, EFA_TEST_COPY_SRC_CUDA_GDRCOPY, false,
	 true},
	{"cudamemcpy_without_gdrcopy_handle", kTestMaxGdrcopyFromHmemSize,
	 EFA_TEST_COPY_SRC_CUDA, true, false},
	{"host_memory_uses_memcpy", kTestMaxGdrcopyFromHmemSize,
	 EFA_TEST_COPY_SRC_HOST, true, false},
};

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmPkeCopyFromHmemTest, ValuesIn(kCopyFromHmemCases),
	[](const testing::TestParamInfo<EfaCopyFromHmemCase> &info) {
		return std::string(info.param.name);
	});

class EfaRdmPkeQueuedCopyTest : public TestWithParam<size_t>
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;
	size_t saved_max_queued_copy = 0;

	void SetUp() override
	{
		struct fi_info *hints;

		saved_max_queued_copy = efa_test_get_max_queued_copy();
		efa_test_set_max_queued_copy(GetParam());

		memset(&resource, 0, sizeof(resource));
		hints = efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		hints->rx_attr->size = 4;
		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);
		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
		efa_test_set_max_queued_copy(saved_max_queued_copy);
	}
};

TEST_P(EfaRdmPkeQueuedCopyTest, raises_rx_size_to_minimum)
{
	size_t max_queued_copy = GetParam();
	size_t rx_size = 0;
	size_t len = sizeof(rx_size);

	ASSERT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT, FI_OPT_RX_SIZE,
			    &rx_size, &len),
		  0);
	EXPECT_EQ(rx_size,
		  std::min(efa_test_device_max_rx_size(),
			   efa_test_get_internal_rx_refill_threshold() +
				   max_queued_copy));
}

TEST_P(EfaRdmPkeQueuedCopyTest, flushes_when_queue_full)
{
	size_t max_queued_copy = GetParam();
	struct efa_test_queued_copy_step steps[EFA_TEST_QUEUED_COPY_MAX_PKTS];
	size_t i;

	EFA_EXPECT_CALL(mock_efa, ofi_copy_to_hmem_iov, FI_HMEM_ROCR,
			EFA_TEST_COPY_HMEM_DEVICE, _, 1u, _, _,
			EFA_TEST_COPY_HMEM_DATA_SIZE)
		.Times((int) max_queued_copy)
		.WillRepeatedly(Return(EFA_TEST_COPY_HMEM_DATA_SIZE));

	ASSERT_EQ(efa_test_flushes_when_queue_full(resource.ep, max_queued_copy,
						   steps),
		  0);

	for (i = 0; i + 1 < max_queued_copy; i++) {
		EXPECT_EQ(steps[i].ret, 0);
		EXPECT_EQ(steps[i].queued_copy_num, (int) (i + 1));
		EXPECT_EQ(steps[i].bytes_copied, 0u);
		EXPECT_EQ(steps[i].bytes_queued_blocking_copy,
			  (i + 1) * EFA_TEST_COPY_HMEM_DATA_SIZE);
		EXPECT_EQ(steps[i].rx_pkts_released, 0u);
	}

	EXPECT_EQ(steps[i].ret, 0);
	EXPECT_EQ(steps[i].queued_copy_num, 0);
	EXPECT_EQ(steps[i].bytes_copied,
		  max_queued_copy * EFA_TEST_COPY_HMEM_DATA_SIZE);
	EXPECT_EQ(steps[i].bytes_queued_blocking_copy, 0u);
	EXPECT_EQ(steps[i].rx_pkts_released, max_queued_copy);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmPkeQueuedCopyTest, Values((size_t) 1, (size_t) 4),
	[](const testing::TestParamInfo<size_t> &info) {
		return "max_queued_copy_" + std::to_string(info.param);
	});
