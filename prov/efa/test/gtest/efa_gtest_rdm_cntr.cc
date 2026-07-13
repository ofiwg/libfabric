/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::StrictMock;
using testing::Test;

class EfaRdmCntrTest : public Test
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

static int fake_shm_cntr_open(struct fid_domain *domain,
			      struct fi_cntr_attr *attr,
			      struct fid_cntr **cntr, void *context)
{
	return -FI_ENOMEM;
}

/**
 * @brief Asserts that rdm_cntr_open cleans up properly after shm_cntr open fails
 */
TEST_F(EfaRdmCntrTest, shm_cntr_open_failure_does_not_leak_domain_ref)
{
	struct fi_ops_domain fake_ops = {};
	struct fid_domain fake_shm_domain = {};
	struct fi_cntr_attr attr = {};
	struct fid_cntr *cntr_fid = nullptr;
	struct fid_domain *saved_shm_domain;
	int ref_before, ref_after, ret;

	fake_ops.size = sizeof(fake_ops);
	fake_ops.cntr_open = fake_shm_cntr_open;
	fake_shm_domain.ops = &fake_ops;

	attr.wait_obj = FI_WAIT_NONE;

	saved_shm_domain = efa_test_get_shm_domain(resource.domain);
	efa_test_set_shm_domain(resource.domain, &fake_shm_domain);

	ref_before = efa_test_get_util_domain_ref(resource.domain);
	ret = fi_cntr_open(resource.domain, &attr, &cntr_fid, NULL);
	ref_after = efa_test_get_util_domain_ref(resource.domain);

	efa_test_set_shm_domain(resource.domain, saved_shm_domain);

	EXPECT_EQ(ret, -FI_ENOMEM);
	/* util_domain's refcount should not increment if cntr_open failed */
	EXPECT_EQ(ref_after, ref_before);

	if (cntr_fid)
		fi_close(&cntr_fid->fid);
}
