/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include <algorithm>
#include <gtest/gtest.h>

using testing::Test;

/**
 * @brief Covers the FI_OPT_TX_SIZE / FI_OPT_RX_SIZE branches of
 * efa_ep_getopt, i.e. the queue depths an efa-direct endpoint reports.
 */
class EfaEpQueueDepthTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		ASSERT_EQ(efa_test_device_probe(), 0);
	}

	void construct(struct fi_info *hints)
	{
		efa_test_resource_construct(&resource, hints);
		ASSERT_NE(resource.ep, nullptr);
	}

	/**
	 * @brief fi_getinfo only, so a test can overwrite an attribute of
	 * resource.info before construct_from_info() creates the endpoint.
	 */
	void getinfo(struct fi_info *hints)
	{
		efa_test_resource_getinfo(&resource, hints);
		ASSERT_NE(resource.info, nullptr);
	}

	void construct_from_info()
	{
		efa_test_resource_open(&resource);
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_EQ(fi_enable(resource.ep), 0);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}

	size_t getopt_size(int optname)
	{
		size_t val = 0;
		size_t len = sizeof(val);

		EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT, optname,
				    &val, &len),
			  0);
		EXPECT_EQ(len, sizeof(val));
		return val;
	}

	/**
	 * @brief The depths reported must be the ones the QP was created with,
	 * both coming from efa_base_ep_get_{tx,rx}_pool_size().
	 */
	void expect_getopt_matches_qp_cap()
	{
		size_t max_send_wr = 0, max_recv_wr = 0;

		efa_test_ep_qp_cap(resource.ep, &max_send_wr, &max_recv_wr,
				   NULL);
		EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), max_send_wr);
		EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), max_recv_wr);
	}
};

TEST_F(EfaEpQueueDepthTest, getopt_reports_configured_depth)
{
	ASSERT_NO_FATAL_FAILURE(construct(efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME)));

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), resource.info->tx_attr->size);
	EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), resource.info->rx_attr->size);
	expect_getopt_matches_qp_cap();
}

TEST_F(EfaEpQueueDepthTest, getopt_honors_requested_depth)
{
	struct fi_info *hints = efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ASSERT_NE(hints, nullptr);
	hints->tx_attr->size = 8;
	hints->rx_attr->size = 8;
	ASSERT_NO_FATAL_FAILURE(construct(hints));

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), 8);
	EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), 8);
	expect_getopt_matches_qp_cap();
}

TEST_F(EfaEpQueueDepthTest, getopt_reports_configured_depth_rdm)
{
	ASSERT_NO_FATAL_FAILURE(construct(
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME)));

	EXPECT_GT(getopt_size(FI_OPT_TX_SIZE), 0);
	EXPECT_LE(getopt_size(FI_OPT_TX_SIZE), resource.info->tx_attr->size);
	EXPECT_GT(getopt_size(FI_OPT_RX_SIZE), 0);
	EXPECT_LE(getopt_size(FI_OPT_RX_SIZE), resource.info->rx_attr->size);
	expect_getopt_matches_qp_cap();
}

TEST_F(EfaEpQueueDepthTest, getopt_honors_requested_depth_rdm)
{
	struct fi_info *hints =
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME);

	ASSERT_NE(hints, nullptr);
	hints->tx_attr->size = 8;
	hints->rx_attr->size = 8;
	ASSERT_NO_FATAL_FAILURE(construct(hints));

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), 8);
	EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), 8);
}

TEST_F(EfaEpQueueDepthTest, getopt_tx_size_reflects_wide_wqe)
{
	struct fi_info *hints = efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ASSERT_NE(hints, nullptr);
	hints->tx_attr->inject_size = efa_test_device_inline_buf_size() + 1;

	if (!efa_test_device_supports_wide_wqe()) {
		struct fi_info *info = nullptr;

		EXPECT_EQ(fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints,
				     &info),
			  -FI_ENODATA);
		EXPECT_EQ(info, nullptr);
		fi_freeinfo(hints);
		return;
	}

	ASSERT_NO_FATAL_FAILURE(construct(hints));

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), resource.info->tx_attr->size);
	EXPECT_LT(getopt_size(FI_OPT_TX_SIZE), efa_test_device_max_tx_size());
	EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), resource.info->rx_attr->size);
	expect_getopt_matches_qp_cap();
}

/**
 * @brief An application can also ask for wide send queue entries by overwriting
 * tx_attr->inject_size in the fi_info returned by fi_getinfo, which therefore
 * never caps tx_attr->size for it. The endpoint must still be created with, and
 * report, the lower depth such an entry allows.
 */
TEST_F(EfaEpQueueDepthTest, getopt_tx_size_reflects_wide_wqe_set_after_getinfo)
{
	size_t wide_inject = efa_test_device_inline_buf_size() + 1;
	ssize_t wide_depth;
	size_t max_inline_data = 0;

	if (!efa_test_device_supports_wide_wqe())
		GTEST_SKIP() << "device does not support wide send queue entries";

	wide_depth = efa_test_device_max_wide_wqe_sq_depth(wide_inject);
	ASSERT_GT(wide_depth, 0);
	ASSERT_LT((size_t) wide_depth, efa_test_device_max_tx_size());

	ASSERT_NO_FATAL_FAILURE(getinfo(efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME)));
	/* fi_getinfo left tx_attr->size at the regular entry's depth */
	ASSERT_EQ(resource.info->tx_attr->size, efa_test_device_max_tx_size());
	resource.info->tx_attr->inject_size = wide_inject;
	ASSERT_NO_FATAL_FAILURE(construct_from_info());

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), (size_t) wide_depth);
	EXPECT_EQ(getopt_size(FI_OPT_RX_SIZE), resource.info->rx_attr->size);
	expect_getopt_matches_qp_cap();

	efa_test_ep_qp_cap(resource.ep, NULL, NULL, &max_inline_data);
	EXPECT_EQ(max_inline_data, wide_inject);
}

/**
 * @brief A depth the application asked for that is below the one a wide send
 * queue entry allows is still the depth reported.
 */
TEST_F(EfaEpQueueDepthTest, getopt_honors_requested_depth_with_wide_wqe)
{
	struct fi_info *hints = efa_test_alloc_default_hints(
		FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ASSERT_NE(hints, nullptr);
	if (!efa_test_device_supports_wide_wqe()) {
		fi_freeinfo(hints);
		GTEST_SKIP() << "device does not support wide send queue entries";
	}

	hints->tx_attr->size = 8;
	ASSERT_NO_FATAL_FAILURE(getinfo(hints));
	resource.info->tx_attr->inject_size =
		efa_test_device_inline_buf_size() + 1;
	ASSERT_NO_FATAL_FAILURE(construct_from_info());

	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE), 8);
	expect_getopt_matches_qp_cap();
}

/**
 * @brief The efa fabric injects through its own protocol, so its inject size,
 * large as it is, leaves the send queue entry regular and the depth untouched.
 */
TEST_F(EfaEpQueueDepthTest, getopt_tx_size_rdm_unaffected_by_inject_size)
{
	ASSERT_NO_FATAL_FAILURE(construct(
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME)));

	ASSERT_GT(resource.info->tx_attr->inject_size,
		  efa_test_device_inline_buf_size());
	EXPECT_EQ(getopt_size(FI_OPT_TX_SIZE),
		  std::min(efa_test_device_max_tx_size(),
			   resource.info->tx_attr->size));
	expect_getopt_matches_qp_cap();
}

TEST_F(EfaEpQueueDepthTest, getopt_reports_configured_depth_dgram)
{
	ASSERT_NO_FATAL_FAILURE(construct(
		efa_test_alloc_default_hints(FI_EP_DGRAM, EFA_FABRIC_NAME)));

	EXPECT_GT(getopt_size(FI_OPT_TX_SIZE), 0);
	EXPECT_LE(getopt_size(FI_OPT_TX_SIZE), resource.info->tx_attr->size);
	EXPECT_GT(getopt_size(FI_OPT_RX_SIZE), 0);
	EXPECT_LE(getopt_size(FI_OPT_RX_SIZE), resource.info->rx_attr->size);
	expect_getopt_matches_qp_cap();
}
