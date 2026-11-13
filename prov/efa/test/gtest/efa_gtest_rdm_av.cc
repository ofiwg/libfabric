/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_av_utils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <unistd.h>

using testing::Test;
using testing::_;
using testing::Return;
using testing::StrictMock;

class EfaConnTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief efa_conn_alloc unwinds the conn via efa_rdm_av_entry_deinit when
 * efa_av_reverse_av_add fails, so the insert fails cleanly.
 */
TEST_F(EfaConnTest, alloc_reverse_av_add_failure_rdm_cleanup)
{
	fi_addr_t addr;
	static struct ibv_ah dummy_ibv_ah;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah)
		.WillOnce(Return(&dummy_ibv_ah));
	/* The unwind releases only the peer's dummy AH; self_ah is destroyed
	 * in teardown after the mock is uninstalled (real destroy). */
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ibv_ah)
		.WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efadv_query_ah)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillOnce(Return(-FI_ENOMEM));

	addr = efa_test_av_insert_new_ah(resource.ep, resource.av);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

/**
 * @brief Test that efa_conn_alloc cleans up RDM resources when
 * efa_av_reverse_av_add fails via the explicit fi_av_insert path
 * (insert_implicit_av=false). The explicit path acquires the SRX lock
 * itself before calling efa_av_insert_one_explicit.
 */
TEST_F(EfaConnTest, alloc_reverse_av_add_failure_explicit_insert)
{
	fi_addr_t addr;
	int num_addr;
	static struct ibv_ah dummy_ibv_ah;

	efa_test_resource_construct(
		&resource,
		efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
	ASSERT_NE(resource.ep, nullptr);

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, ibv_create_ah, _, _)
		.WillOnce(Return(&dummy_ibv_ah));
	/* The unwind releases only the peer's dummy AH; self_ah is destroyed
	 * in teardown after the mock is uninstalled (real destroy). */
	EFA_EXPECT_CALL(mock_efa, ibv_destroy_ah, &dummy_ibv_ah)
		.WillOnce(Return(0));
	EFA_EXPECT_CALL(mock_efa, efadv_query_ah)
		.WillRepeatedly(Return(0));
	EFA_EXPECT_CALL(mock_efa, efa_av_reverse_av_add)
		.WillOnce(Return(-FI_ENOMEM));

	num_addr = efa_test_av_insert_fake_gid(resource.ep, resource.av, &addr);
	EXPECT_EQ(num_addr, 0);
}

struct EfaShmAvSizeParam {
	const char *name;
	int env_value;
};

class EfaShmAvSizeTest : public testing::TestWithParam<EfaShmAvSizeParam>
{
	protected:
	struct efa_resource resource = {};
	int prev_shm_av_size;
	int limit;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		prev_shm_av_size = efa_test_get_shm_av_size();
		if (GetParam().env_value)
			efa_test_set_shm_av_size(GetParam().env_value);
		limit = efa_test_get_shm_av_size();
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_NE(efa_test_get_shm_domain(resource.domain), nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
		efa_test_set_shm_av_size(prev_shm_av_size);
	}
};

TEST_P(EfaShmAvSizeTest, local_peers_up_to_shm_av_size_use_shm)
{
	fi_addr_t addr;
	int i;

	for (i = 0; i < limit; i++) {
		ASSERT_EQ(efa_test_av_insert_local_peer(resource.ep, resource.av,
							i + 1, &addr),
			  1)
			<< "local peer " << i;
		ASSERT_TRUE(efa_test_av_peer_uses_shm(resource.ep, addr))
			<< "local peer " << i;
	}

	EXPECT_EQ(efa_test_av_insert_local_peer(resource.ep, resource.av,
						limit + 1, &addr),
		  0);
	EXPECT_EQ(addr, (fi_addr_t) FI_ADDR_NOTAVAIL);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaShmAvSizeTest,
	testing::Values(EfaShmAvSizeParam{"default", 0},
			EfaShmAvSizeParam{"lowered", 64}),
	[](const testing::TestParamInfo<EfaShmAvSizeParam> &info) {
		return std::string(info.param.name);
	});

class EfaShmAvSizeLimitTest : public Test
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
		ASSERT_NE(efa_test_get_shm_domain(resource.domain), nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}

	int av_open_with_shm_av_size(int shm_av_size)
	{
		struct fi_av_attr av_attr = {};
		struct fid_av *av = nullptr;
		int prev, ret;

		av_attr.type = FI_AV_TABLE;
		prev = efa_test_set_shm_av_size(shm_av_size);
		ret = fi_av_open(resource.domain, &av_attr, &av, NULL);
		efa_test_set_shm_av_size(prev);
		if (!ret)
			fi_close(&av->fid);
		return ret;
	}
};

TEST_F(EfaShmAvSizeLimitTest, default_follows_online_cpus)
{
	const char *env = getenv("FI_EFA_SHM_AV_SIZE");
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	long expected;

	if (env)
		expected = atol(env);
	else
		expected = std::max(256L, cpus);

	EXPECT_EQ(efa_test_get_shm_av_size(), expected);
}

TEST_F(EfaShmAvSizeLimitTest, av_open_fails_above_shm_capacity)
{
	long cpus = sysconf(_SC_NPROCESSORS_ONLN);
	int above_capacity = (int) std::max(256L, cpus) + 1;

	EXPECT_EQ(av_open_with_shm_av_size(above_capacity), -FI_ENOSYS);
}
