/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_shm_utils.h"
#include <gtest/gtest.h>

using testing::Test;

class EfaRdmShmPermittedTest : public Test
{
	protected:
	struct efa_resource resource = {};
	struct fid_ep *ep2 = nullptr;
	struct fid_cntr *cntr = nullptr;
	struct fid_cq *ep2_cq = nullptr;

	void SetUp() override
	{
		ASSERT_NO_FATAL_FAILURE(efa_test_resource_construct_no_enable(
			&resource,
			efa_test_alloc_default_hints(FI_EP_RDM, EFA_FABRIC_NAME)));
		ASSERT_NE(efa_test_get_shm_domain(resource.domain), nullptr);
		ASSERT_TRUE(efa_test_rdm_ep_has_shm_ep(resource.ep));
	}

	void TearDown() override
	{
		if (ep2)
			EXPECT_EQ(fi_close(&ep2->fid), 0);
		if (cntr)
			EXPECT_EQ(fi_close(&cntr->fid), 0);
		if (ep2_cq)
			EXPECT_EQ(fi_close(&ep2_cq->fid), 0);
		efa_test_resource_destruct(&resource);
	}

	void open_ep2(struct fid_cq *cq = nullptr)
	{
		if (!cq)
			cq = resource.cq;

		ASSERT_EQ(fi_endpoint(resource.domain, resource.info, &ep2,
				      NULL),
			  0);
		ASSERT_EQ(fi_ep_bind(ep2, &resource.eq->fid, 0), 0);
		ASSERT_EQ(fi_ep_bind(ep2, &resource.av->fid, 0), 0);
		ASSERT_EQ(fi_ep_bind(ep2, &cq->fid, FI_SEND | FI_RECV), 0);
		ASSERT_TRUE(efa_test_rdm_ep_has_shm_ep(ep2));
	}

	void disable_shm(struct fid_ep *ep)
	{
		bool shm_permitted = false;

		ASSERT_EQ(fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
				    FI_OPT_SHARED_MEMORY_PERMITTED,
				    &shm_permitted, sizeof(shm_permitted)),
			  0);
	}
};

TEST_F(EfaRdmShmPermittedTest, disable_closes_only_ep_shm_resources)
{
	ASSERT_NO_FATAL_FAILURE(open_ep2());
	ASSERT_NO_FATAL_FAILURE(disable_shm(resource.ep));
	ASSERT_NO_FATAL_FAILURE(disable_shm(ep2));

	ASSERT_EQ(fi_enable(resource.ep), 0);
	ASSERT_EQ(fi_enable(ep2), 0);

	EXPECT_FALSE(efa_test_rdm_ep_has_shm_ep(resource.ep));
	EXPECT_FALSE(efa_test_rdm_ep_has_shm_ep(ep2));
	EXPECT_NE(efa_test_get_shm_domain(resource.domain), nullptr);
	EXPECT_TRUE(efa_test_rdm_av_has_shm_av(resource.av));
	EXPECT_EQ(efa_test_rdm_cq_shm_cq_ref(resource.cq), 0);
}

TEST_F(EfaRdmShmPermittedTest, shm_cq_ref_tracks_bound_shm_eps)
{
	ASSERT_NO_FATAL_FAILURE(open_ep2());
	ASSERT_NO_FATAL_FAILURE(disable_shm(resource.ep));

	ASSERT_EQ(fi_enable(resource.ep), 0);
	ASSERT_EQ(fi_enable(ep2), 0);

	EXPECT_FALSE(efa_test_rdm_ep_has_shm_ep(resource.ep));
	EXPECT_TRUE(efa_test_rdm_ep_has_shm_ep(ep2));
	EXPECT_GT(efa_test_rdm_cq_shm_cq_ref(resource.cq), 0);

	ASSERT_EQ(fi_close(&ep2->fid), 0);
	ep2 = nullptr;
	EXPECT_EQ(efa_test_rdm_cq_shm_cq_ref(resource.cq), 0);
}

TEST_F(EfaRdmShmPermittedTest, shm_cntr_ref_tracks_bound_shm_eps)
{
	struct fi_cntr_attr cntr_attr = {};
	struct fi_cq_attr cq_attr = {};

	ASSERT_NO_FATAL_FAILURE(disable_shm(resource.ep));
	ASSERT_EQ(fi_enable(resource.ep), 0);
	ASSERT_FALSE(efa_test_rdm_ep_has_shm_ep(resource.ep));

	ASSERT_EQ(fi_cntr_open(resource.domain, &cntr_attr, &cntr, NULL), 0);
	EXPECT_EQ(efa_test_rdm_cntr_shm_cntr_ref(cntr), 0);

	/*
	 * TODO: share resource.cq once the cntr cq poll list entry is removed
	 * on ep close even when another ep not bound to the cntr uses the cq.
	 */
	cq_attr.format = FI_CQ_FORMAT_DATA;
	ASSERT_EQ(fi_cq_open(resource.domain, &cq_attr, &ep2_cq, NULL), 0);
	ASSERT_NO_FATAL_FAILURE(open_ep2(ep2_cq));
	ASSERT_EQ(fi_ep_bind(ep2, &cntr->fid, FI_TRANSMIT), 0);
	ASSERT_EQ(fi_enable(ep2), 0);

	EXPECT_TRUE(efa_test_rdm_ep_has_shm_ep(ep2));
	EXPECT_GT(efa_test_rdm_cntr_shm_cntr_ref(cntr), 0);
}
