/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include <gtest/gtest.h>

using testing::Test;

class EfaOpeTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM,
						     EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
		ASSERT_NE(resource.av, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/*
 * efa_rdm_ope_process_queued_ope now derives the queued flag from the ope's
 * internal_flags rather than taking it as a parameter. With no
 * EFA_RDM_OPE_QUEUED_* bit set, it must be a no-op: return 0, dispatch
 * nothing, and leave the ope_queued_list untouched.
 */
TEST_F(EfaOpeTest, process_queued_ope_no_flag_is_noop)
{
	EXPECT_EQ(efa_test_ope_process_queued_no_flag(resource.ep, resource.av),
		  0);
}

/*
 * With exactly the BEFORE_HANDSHAKE bit set, the function must derive that
 * flag from internal_flags and dispatch to the before-handshake repost.
 * Because the peer has not received a handshake, the repost returns
 * -FI_EAGAIN, and the function must leave the flag set, keep the ope queued,
 * and not decrement the before-handshake counter.
 */
TEST_F(EfaOpeTest, process_queued_ope_derives_before_handshake_flag)
{
	EXPECT_EQ(efa_test_ope_process_queued_before_handshake_eagain(
			  resource.ep, resource.av),
		  0);
}
