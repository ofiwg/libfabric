/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::Return;
using testing::StrictMock;
using testing::Test;

class EfaRdmMrTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.domain, nullptr);

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief A failed domain mr_map insert error must propagate out of registration.
 */
TEST_F(EfaRdmMrTest, reg_map_insert_failure_propagates_error)
{
	char buf[64];
	struct iovec iov = {};
	struct fi_mr_attr attr = {};
	struct fid_mr *mr = nullptr;
	struct fid_domain *saved_shm_domain;
	int ret;

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	attr.mr_iov = &iov;
	attr.iov_count = 1;
	attr.access = FI_SEND | FI_RECV;
	attr.iface = FI_HMEM_SYSTEM;

	EXPECT_CALL(mock_efa, ofi_mr_map_insert)
		.Times(1)
		.WillOnce(Return(-FI_ENOMEM));

	/* Detach shm to skip the irrelevant shm block in mr_regattr. */
	saved_shm_domain = efa_test_get_shm_domain(resource.domain);
	efa_test_set_shm_domain(resource.domain, nullptr);
	ret = fi_mr_regattr(resource.domain, &attr, 0, &mr);
	efa_test_set_shm_domain(resource.domain, saved_shm_domain);
	EXPECT_EQ(ret, -FI_ENOMEM);

	if (mr)
		fi_close(&mr->fid);
}
