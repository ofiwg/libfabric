/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with the same raw address,
 * and verifies that returned fi_addr is the same and
 * ibv_create_ah only gets called once.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_raw_addr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);
	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_equal(addr1, addr2);
}

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with two difference raw address with same GID,
 * and verifies that returned fi_addr is different and ibv_create_ah only gets called once.
 * this is because libfabric EFA provider has a cache for address handle (AH).
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_gid(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);
	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_ep *ep2, *ep3;
	int ret;

	/* Resource construct function creates and binds 1 EP to the AV */
	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	/* Create and bind two new endpoints to the same AV */
	fi_endpoint(resource->domain, resource->info, &ep2, NULL);
	ret = fi_ep_bind(ep2, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	fi_endpoint(resource->domain, resource->info, &ep3, NULL);
	ret = fi_ep_bind(ep3, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	/* Bind the two new endpoints to the same CQ and enable them */
	fi_ep_bind(ep2, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep2);
	assert_int_equal(ret, 0);

	fi_ep_bind(ep3, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep3);
	assert_int_equal(ret, 0);

	fi_close(&ep2->fid);
	fi_close(&ep3->fid);
}
