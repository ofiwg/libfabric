/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_ope_helpers.h"
#include <gtest/gtest.h>
#include <rdma/fi_errno.h>

using testing::Test;

class EfaRdmOpeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));

		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief Assert that the context for an unexpected
 * packet is not copied into the error entry
 */
TEST_F(EfaRdmOpeTest, rxe_unexp_error_suppresses_op_context)
{
	int prov_errno = 0;
	void *sentinel = (void *) 0xdeadbeef;
	struct fi_cq_err_entry err_entry = {};

	ASSERT_EQ(efa_test_drive_rxe_unexp_handle_error(
			  resource.ep, sentinel, FI_ENOTCONN, &prov_errno),
		  0);

	ASSERT_EQ(fi_cq_readerr(resource.cq, &err_entry, 0), 1);
	EXPECT_EQ(err_entry.err, FI_ENOTCONN);
	EXPECT_EQ(err_entry.prov_errno, prov_errno);
	/* The sentinel should not be here */
	EXPECT_EQ(err_entry.op_context, nullptr);
}
