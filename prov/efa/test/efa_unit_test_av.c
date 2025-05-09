/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "efa_av.h"

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with the same raw address,
 * and verifies that returned fi_addr is the same.
 * Since the addresses to be inserted have the same GID with the ep's self ah,
 * there should be only 1 ibv_create_ah call in the whole test.
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

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_equal(addr1, addr2);
}

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with two difference raw address with same GID,
 * and verifies that returned fi_addr is different.
 * Since the addresses to be inserted have the same GID with the ep's self ah,
 * there should be only 1 ibv_create_ah call in the whole test.
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

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);
}

void test_efa_ah_cnt_one_av(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* So far we should only have 1 ah from ep self ah, and its refcnt is 1 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);
	assert_int_equal(efa_ah->refcnt, 1);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);

	/* So far we should still have 1 ah, and its refcnt is 3 (plus the 2 av entries) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->refcnt, 3);

	/* ah refcnt should be decremented to 1 after av entry removals */
	assert_int_equal(fi_av_remove(resource->av, &addr1, 1, 0), 0);
	assert_int_equal(fi_av_remove(resource->av, &addr2, 1, 0), 0);

	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->refcnt, 1);

	/* ah map should be empty now after closing ep which destroys the self ah */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

void test_efa_ah_cnt_multi_av(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;
	struct fi_av_attr av_attr = {0};
	struct fid_av *av1, *av2;
	struct fid_ep *ep1, *ep2;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* So far we should only have 1 ah from ep self ah, and its refcnt is 1 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);
	assert_int_equal(efa_ah->refcnt, 1);


	/* We open 2 avs with the same domain (PD) so they should share same AH given the same GID */
	assert_int_equal(fi_av_open(resource->domain, &av_attr, &av1, NULL), 0);
	assert_int_equal(fi_av_open(resource->domain, &av_attr, &av2, NULL), 0);

	/* Due to the current restriction in efa provider, we have to bind av to ep before inserting av entry */
	/* These eps will not create self ah as they are not enabled */
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep1, NULL), 0);
	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep2, NULL), 0);

	assert_int_equal(fi_ep_bind(ep1, &av1->fid, 0), 0);
	assert_int_equal(fi_ep_bind(ep2, &av2->fid, 0), 0);

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(av1, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(av2, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	/* They should be as equal as 0 they are in different avs */
	assert_int_equal(addr1, addr2);

	/* So far we should still have 1 ah, and its refcnt is 3 (plus the 2 av entries) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->refcnt, 3);

	/* ah refcnt should be decremented to 1 after av close */
	assert_int_equal(fi_close(&ep1->fid), 0);
	assert_int_equal(fi_close(&ep2->fid), 0);
	assert_int_equal(fi_close(&av1->fid), 0);
	assert_int_equal(fi_close(&av2->fid), 0);

	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->refcnt, 1);

	/* ah map should be empty now after closing ep which destroys the self ah */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}