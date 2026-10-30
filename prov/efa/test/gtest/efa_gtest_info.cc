/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_info_utils.h"
#include <gtest/gtest.h>

using testing::Test;

/**
 * @brief Covers the send queue depth fi_getinfo reports for an inject size that
 * makes every WQE wide, i.e. the efa_query_max_sq_depth call efa_user_info
 * makes, on devices that support wide WQEs and on devices that do not.
 */
class EfaInfoQueueDepthTest : public Test
{
	protected:
	struct fi_info *hints = nullptr;
	struct fi_info *info = nullptr;
	bool wide_wqe = false;
	size_t wide_inject = 0;
	ssize_t max_sq_depth = -1;
	size_t prev_tx_size = 0;
	size_t prev_rx_size = 0;
	bool env_set = false;

	void SetUp() override
	{
		ASSERT_EQ(efa_test_device_probe(), 0);

		wide_wqe = efa_test_device_supports_wide_wqe();
		wide_inject = efa_test_device_inline_buf_size() + 1;
		if (wide_wqe) {
			max_sq_depth = efa_test_device_max_wide_wqe_sq_depth(
				wide_inject);
			ASSERT_GT(max_sq_depth, 0);
			ASSERT_LE((size_t) max_sq_depth,
				  efa_test_device_max_tx_size());
		}

		hints = efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
	}

	void TearDown() override
	{
		if (info)
			fi_freeinfo(info);
		if (hints)
			fi_freeinfo(hints);
		if (env_set)
			EXPECT_EQ(efa_test_set_queue_size_env(prev_tx_size,
							      prev_rx_size,
							      NULL, NULL),
				  0);
	}

	void set_queue_size_env(size_t tx, size_t rx)
	{
		int ret = efa_test_set_queue_size_env(tx, rx, &prev_tx_size,
						      &prev_rx_size);

		env_set = true;
		ASSERT_EQ(ret, 0);
	}

	int getinfo()
	{
		return fi_getinfo(FI_VERSION(2, 0), NULL, NULL, 0ULL, hints,
				  &info);
	}
};

TEST_F(EfaInfoQueueDepthTest, getinfo_tx_size_unchanged_without_wide_wqe)
{
	hints->tx_attr->inject_size = efa_test_device_inline_buf_size();

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->inject_size,
		  efa_test_device_inline_buf_size());
	EXPECT_EQ(info->tx_attr->size, efa_test_device_max_tx_size());
}

TEST_F(EfaInfoQueueDepthTest, getinfo_tx_size_capped_by_wide_wqe)
{
	hints->tx_attr->inject_size = wide_inject;

	if (!wide_wqe) {
		EXPECT_EQ(getinfo(), -FI_ENODATA);
		EXPECT_EQ(info, nullptr);
		return;
	}

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->inject_size, wide_inject);
	EXPECT_EQ(info->tx_attr->size, (size_t) max_sq_depth);
	EXPECT_LT(info->tx_attr->size, efa_test_device_max_tx_size());
	EXPECT_EQ(info->rx_attr->size, efa_test_device_max_rx_size());
}

TEST_F(EfaInfoQueueDepthTest, getinfo_accepts_tx_size_at_wide_wqe_depth)
{
	hints->tx_attr->inject_size = wide_inject;
	hints->tx_attr->size = wide_wqe ? (size_t) max_sq_depth : 1;

	if (!wide_wqe) {
		EXPECT_EQ(getinfo(), -FI_ENODATA);
		return;
	}

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->size, (size_t) max_sq_depth);
}

TEST_F(EfaInfoQueueDepthTest, getinfo_rejects_tx_size_above_wide_wqe_depth)
{
	hints->tx_attr->inject_size = wide_inject;
	hints->tx_attr->size =
		wide_wqe ? (size_t) max_sq_depth + 1 : efa_test_device_max_tx_size();

	EXPECT_EQ(getinfo(), -FI_ENODATA);
	EXPECT_EQ(info, nullptr);
}

TEST_F(EfaInfoQueueDepthTest, getinfo_size_honors_env_below_device)
{
	size_t tx_size = efa_test_device_max_tx_size() / 2;
	size_t rx_size = efa_test_device_max_rx_size() / 2;

	ASSERT_GT(tx_size, 0);
	ASSERT_GT(rx_size, 0);
	ASSERT_NO_FATAL_FAILURE(set_queue_size_env(tx_size, rx_size));

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->size, tx_size);
	EXPECT_EQ(info->rx_attr->size, rx_size);
}

TEST_F(EfaInfoQueueDepthTest, getinfo_size_capped_by_device_when_env_above)
{
	ASSERT_NO_FATAL_FAILURE(
		set_queue_size_env(efa_test_device_max_tx_size() * 2,
				   efa_test_device_max_rx_size() * 2));

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->size, efa_test_device_max_tx_size());
	EXPECT_EQ(info->rx_attr->size, efa_test_device_max_rx_size());
}

TEST_F(EfaInfoQueueDepthTest, getinfo_wide_wqe_tx_size_honors_env_below_depth)
{
	if (!wide_wqe)
		GTEST_SKIP() << "device without wide WQE support";

	size_t tx_size = (size_t) max_sq_depth / 2;

	ASSERT_GT(tx_size, 0);
	ASSERT_NO_FATAL_FAILURE(set_queue_size_env(tx_size, 0));
	hints->tx_attr->inject_size = wide_inject;

	ASSERT_EQ(getinfo(), 0);
	ASSERT_NE(info, nullptr);

	EXPECT_EQ(info->tx_attr->inject_size, wide_inject);
	EXPECT_EQ(info->tx_attr->size, tx_size);
}
