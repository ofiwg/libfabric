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
void test_av_insert_duplicate_raw_addr(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	expect_function_call(efa_mock_ibv_create_ah_check_mock);

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
void test_av_insert_duplicate_gid(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	expect_function_call(efa_mock_ibv_create_ah_check_mock);

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

/**
 * @brief Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with the same raw address while
 * strict mode is on, and verifies that the second (duplicate) insert fails
 * with -FI_EINVAL instead of returning the existing fi_addr.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_raw_addr_strict_mode(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	int strict_mode_orig = efa_env.strict_mode;

	efa_env.strict_mode = 1;

	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;
	expect_function_call(efa_mock_ibv_create_ah_check_mock);

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	addr2 = FI_ADDR_NOTAVAIL;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 0);
	assert_int_equal(addr2, FI_ADDR_NOTAVAIL);

	efa_env.strict_mode = strict_mode_orig;
}

/**
 * @brief Test base AV (efa-direct) insert of a duplicate address while
 * strict mode is on fails with -FI_EINVAL instead of returning the existing
 * fi_addr.
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_insert_duplicate_raw_addr_efa_direct_strict_mode(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	int strict_mode_orig = efa_env.strict_mode;

	efa_env.strict_mode = 1;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	addr2 = FI_ADDR_NOTAVAIL;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 0);
	assert_int_equal(addr2, FI_ADDR_NOTAVAIL);

	efa_env.strict_mode = strict_mode_orig;
}

static void efa_ah_cnt_av_efa_impl(void **state, bool multi_av)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;
	struct fi_av_attr av_attr = {0};
	struct fid_av *av1 = NULL, *av2 = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* So far we should only have 1 ah from ep self ah, and its refcnt is 1 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->explicit_refcnt, 1);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->implicit_refcnt, 0);

	if (multi_av) {
		/* We open 2 avs with the same domain (PD) so they should share same AH given the same GID */
		assert_int_equal(fi_av_open(resource->domain, &av_attr, &av1, NULL), 0);
		assert_int_equal(fi_av_open(resource->domain, &av_attr, &av2, NULL), 0);
	}

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(multi_av ? av1 : resource->av, &raw_addr, 1, &addr1, 0, NULL);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(multi_av ? av2 : resource->av, &raw_addr, 1, &addr2, 0, NULL);
	assert_int_equal(num_addr, 1);

	if (multi_av) {
		/* They should be equal as 0 since they are in different avs */
		assert_int_equal(addr1, addr2);
	} else {
		assert_int_not_equal(addr1, addr2);
	}

	/* So far we should still have 1 ah, and its refcnt is 3 (including self AH) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->explicit_refcnt, 3);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->implicit_refcnt, 0);

	if (multi_av) {
		/* ah refcnt should be decremented to 1 after av close */
		assert_int_equal(fi_close(&av1->fid), 0);
		assert_int_equal(fi_close(&av2->fid), 0);
	} else {
		/* ah refcnt should be decremented to 1 after av entry removals */
		assert_int_equal(fi_av_remove(resource->av, &addr1, 1, 0), 0);
		assert_int_equal(fi_av_remove(resource->av, &addr2, 1, 0), 0);
	}

	/* efa_ah is still alive because self-AH holds a reference */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->explicit_refcnt, 1);
	assert_int_equal(((struct efa_rdm_ah *)(efa_ah))->implicit_refcnt, 0);

	/* ah map should be empty now after closing ep which destroys the self ah */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

static void efa_ah_cnt_av_efa_direct_impl(void **state, bool multi_av)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;
	struct efa_domain *efa_domain;
	struct efa_ah *efa_ah = NULL;
	struct fi_av_attr av_attr = {0};
	struct fid_av *av1 = NULL, *av2 = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	/* efa-direct does not create a self AH, so ah_map should be empty */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_null(efa_ah);

	if (multi_av) {
		/* We open 2 avs with the same domain (PD) so they should share same AH given the same GID */
		assert_int_equal(fi_av_open(resource->domain, &av_attr, &av1, NULL), 0);
		assert_int_equal(fi_av_open(resource->domain, &av_attr, &av2, NULL), 0);
	}

	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	num_addr = fi_av_insert(multi_av ? av1 : resource->av, &raw_addr, 1, &addr1, 0, NULL);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(multi_av ? av2 : resource->av, &raw_addr, 1, &addr2, 0, NULL);
	assert_int_equal(num_addr, 1);

	if (multi_av) {
		/* They should be equal as 0 since they are in different avs */
		assert_int_equal(addr1, addr2);
	} else {
		assert_int_not_equal(addr1, addr2);
	}

	HASH_FIND(hh, efa_domain->ah_map, raw_addr.raw, EFA_GID_LEN, efa_ah);
	assert_non_null(efa_ah);

	/* So far we should still have 1 ah, and its refcnt is 2 */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 1);
	assert_int_equal(efa_ah->refcnt, 2);

	if (multi_av) {
		/* ah refcnt should be decremented to 0 after av close */
		assert_int_equal(fi_close(&av1->fid), 0);
		assert_int_equal(fi_close(&av2->fid), 0);
	} else {
		/* ah refcnt should be decremented to 0 after av entry removals */
		assert_int_equal(fi_av_remove(resource->av, &addr1, 1, 0), 0);
		assert_int_equal(fi_av_remove(resource->av, &addr2, 1, 0), 0);
	}

	/* efa_ah has been freed (no self AH holding a reference on efa-direct) */
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);

	/* ah map should still be empty after closing ep */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	assert_int_equal(HASH_CNT(hh, efa_domain->ah_map), 0);
	/* Reset to NULL to avoid test reaper closing again */
	resource->ep = NULL;
}

void test_efa_ah_cnt_one_av_efa(void **state)
{
	efa_ah_cnt_av_efa_impl(state, false);
}

void test_efa_ah_cnt_one_av_efa_direct(void **state)
{
	efa_ah_cnt_av_efa_direct_impl(state, false);
}

void test_efa_ah_cnt_multi_av_efa(void **state)
{
	efa_ah_cnt_av_efa_impl(state, true);
}

void test_efa_ah_cnt_multi_av_efa_direct(void **state)
{
	efa_ah_cnt_av_efa_direct_impl(state, true);
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_impl(void **state, char *fabric_name)
{
	struct efa_resource *resource = *state;
	struct fid_ep *ep1, *ep2;
	int ret;

	/* Resource construct function creates and binds 1 EP to the AV */
	efa_unit_test_resource_construct(resource, FI_EP_RDM, fabric_name);

	/* Create and bind two new endpoints to the same AV */
	fi_endpoint(resource->domain, resource->info, &ep1, NULL);
	ret = fi_ep_bind(ep1, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	fi_endpoint(resource->domain, resource->info, &ep2, NULL);
	ret = fi_ep_bind(ep2, &resource->av->fid, 0);
	assert_int_equal(ret, 0);

	/* Bind the two new endpoints to the same CQ and enable them */
	fi_ep_bind(ep1, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep1);
	assert_int_equal(ret, 0);

	fi_ep_bind(ep2, &resource->cq->fid, FI_SEND | FI_RECV);
	ret = fi_enable(ep2);
	assert_int_equal(ret, 0);

	fi_close(&ep1->fid);
	fi_close(&ep2->fid);
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 * for the efa fabric
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_efa(void **state)
{
	return test_av_multiple_ep_impl(state, EFA_FABRIC_NAME);
}

/**
 * @brief This test verifies that multiple endpoints can bind to the same AV
 * for the efa-direct fabric
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_multiple_ep_efa_direct(void **state)
{
	return test_av_multiple_ep_impl(state, EFA_DIRECT_FABRIC_NAME);
}

/**
 * @brief Test base AV (efa-direct) insert, lookup, remove cycle
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_insert_remove_lookup_efa_direct(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0}, raw_addr_out = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t fi_addr;
	struct efa_av *av;
	struct efa_av_entry *entry;
	int err, num_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	av = container_of(resource->av, struct efa_av, util_av.av_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 7;
	raw_addr.qkey = 0xABCD;

	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &fi_addr, 0, NULL);
	assert_int_equal(num_addr, 1);
	assert_int_equal(fi_addr, 0);
	assert_int_equal(HASH_CNT(hh, av->util_av.hash), 1);

	/* Verify entry is accessible and fields are correct */
	entry = efa_av_addr_to_entry(av, fi_addr);
	assert_non_null(entry);
	assert_non_null(entry->ah);
	assert_int_equal(entry->fi_addr, fi_addr);
	assert_int_equal(efa_av_entry_ep_addr(entry)->qpn, 7);
	assert_int_equal(efa_av_entry_ep_addr(entry)->qkey, 0xABCD);

	/* Lookup should return the same address */
	raw_addr_len = sizeof(raw_addr_out);
	err = fi_av_lookup(resource->av, fi_addr, &raw_addr_out, &raw_addr_len);
	assert_int_equal(err, 0);
	assert_int_equal(raw_addr_out.qpn, 7);
	assert_int_equal(raw_addr_out.qkey, 0xABCD);
	assert_int_equal(efa_is_same_addr(&raw_addr, &raw_addr_out), 1);

	/* Remove and verify */
	err = fi_av_remove(resource->av, &fi_addr, 1, 0);
	assert_int_equal(err, 0);
	assert_int_equal(HASH_CNT(hh, av->util_av.hash), 0);

	/* Entry should be NULL after remove */
	entry = efa_av_addr_to_entry(av, fi_addr);
	assert_null(entry);

	/* Lookup should fail after remove */
	err = fi_av_lookup(resource->av, fi_addr, &raw_addr_out, &raw_addr_len);
	assert_int_not_equal(err, 0);
}

/**
 * @brief Test base AV (efa-direct) addr_to_entry returns NULL for invalid fi_addr
 *
 * @param[in]	state	struct efa_resource that is managed by the framework
 */
void test_av_base_addr_to_entry_invalid(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_av *av;
	struct efa_av_entry *entry;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	av = container_of(resource->av, struct efa_av, util_av.av_fid);

	entry = efa_av_addr_to_entry(av, FI_ADDR_NOTAVAIL);
	assert_null(entry);

	entry = efa_av_addr_to_entry(av, FI_ADDR_UNSPEC);
	assert_null(entry);
}

/**
 * @brief Insert, look up, reverse-look-up and remove an address on a FI_AV_MAP AV
 *
 * With FI_AV_MAP the fi_addr is the address of the provider's AV entry, while
 * the entry keeps its util_av index.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 * @param[in]	fabric_name	efa or efa-direct
 */
static void test_av_map_insert_remove_lookup_impl(void **state, char *fabric_name)
{
	struct efa_resource *resource = *state;
	struct efa_ep_addr raw_addr = {0}, raw_addr_out = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	int strict_mode_orig;
	struct fi_av_attr av_attr = {.type = FI_AV_MAP};
	struct fid_av *map_av;
	struct efa_av *av;
	struct efa_av_entry *entry;
	fi_addr_t fi_addr, fi_addr2, src;
	int err, num_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, fabric_name);

	err = fi_av_open(resource->domain, &av_attr, &map_av, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(av_attr.type, FI_AV_MAP);
	av = container_of(map_av, struct efa_av, util_av.av_fid);
	assert_true(efa_av_is_map(av));

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 7;
	raw_addr.qkey = 0xABCD;

	num_addr = fi_av_insert(map_av, &raw_addr, 1, &fi_addr, 0, NULL);
	assert_int_equal(num_addr, 1);

	/* the fi_addr is the entry, which keeps the util_av index */
	entry = efa_av_addr_to_entry(av, fi_addr);
	assert_non_null(entry);
	assert_ptr_equal((void *) (uintptr_t) fi_addr, entry);
	assert_int_equal(entry->fi_addr, 0);
	assert_int_equal(efa_av_fi_addr_to_index(av, fi_addr), 0);
	assert_int_equal(efa_av_index_to_fi_addr(av, 0), fi_addr);
	assert_int_equal(efa_av_entry_ep_addr(entry)->qpn, 7);

	/* inserting the same address again returns the same fi_addr */
	num_addr = fi_av_insert(map_av, &raw_addr, 1, &fi_addr2, 0, NULL);
	assert_int_equal(num_addr, 1);
	assert_int_equal(fi_addr2, fi_addr);

	/* a completion's source is reported as the application fi_addr */
	if (!strcmp(fabric_name, EFA_DIRECT_FABRIC_NAME))
		src = efa_av_reverse_lookup(av, entry->ah->ahn, 7);
	else
		src = efa_rdm_av_reverse_lookup(av, entry->ah->ahn, 7, NULL);
	assert_int_equal(src, fi_addr);

	raw_addr_len = sizeof(raw_addr_out);
	err = fi_av_lookup(map_av, fi_addr, &raw_addr_out, &raw_addr_len);
	assert_int_equal(err, 0);
	assert_int_equal(efa_is_same_addr(&raw_addr, &raw_addr_out), 1);

	err = fi_av_remove(map_av, &fi_addr, 1, 0);
	assert_int_equal(err, 0);
	assert_int_equal(HASH_CNT(hh, av->util_av.hash), 0);

	/* a removed FI_AV_MAP fi_addr is rejected, not dereferenced */
	assert_int_equal(fi_av_lookup(map_av, fi_addr, &raw_addr_out, &raw_addr_len), -FI_EINVAL);
	assert_int_equal(fi_av_remove(map_av, &fi_addr, 1, 0), -FI_EINVAL);
	/* data transfers only check it in strict mode */
	strict_mode_orig = efa_env.strict_mode;
	efa_env.strict_mode = 1;
	assert_null(efa_av_addr_to_entry(av, fi_addr));
	assert_int_equal(efa_av_fi_addr_to_index(av, fi_addr), FI_ADDR_NOTAVAIL);
	efa_env.strict_mode = strict_mode_orig;

	err = fi_close(&map_av->fid);
	assert_int_equal(err, 0);
}

void test_av_map_insert_remove_lookup_efa(void **state)
{
	test_av_map_insert_remove_lookup_impl(state, EFA_FABRIC_NAME);
}

void test_av_map_insert_remove_lookup_efa_direct(void **state)
{
	test_av_map_insert_remove_lookup_impl(state, EFA_DIRECT_FABRIC_NAME);
}

/**
 * @brief With FI_EFA_SUPPORT_AV_MAP=0 a FI_AV_MAP request opens a table
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_map_disabled(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_av_attr av_attr = {.type = FI_AV_MAP};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct fid_av *map_av;
	fi_addr_t fi_addr;
	int err, saved = efa_env.support_av_map;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_env.support_av_map = 0;
	err = fi_av_open(resource->domain, &av_attr, &map_av, NULL);
	efa_env.support_av_map = saved;
	assert_int_equal(err, 0);
	assert_int_equal(av_attr.type, FI_AV_TABLE);
	assert_false(efa_av_is_map(container_of(map_av, struct efa_av, util_av.av_fid)));

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 7;
	assert_int_equal(fi_av_insert(map_av, &raw_addr, 1, &fi_addr, 0, NULL), 1);
	assert_int_equal(fi_addr, 0);

	err = fi_close(&map_av->fid);
	assert_int_equal(err, 0);
}

/**
 * @brief On the efa (RDM) fabric with FI_AV_MAP, the AV entry caches the peer
 * of the first endpoint that resolves it, and the peer keeps the util_av index
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_av_map_rdm_peer_cache(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_av_attr av_attr = {.type = FI_AV_MAP};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_rdm_av_entry *av_entry;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *ep;
	struct fid_av *map_av;
	struct fid_ep *ep_fid;
	fi_addr_t fi_addr;
	int err;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	err = fi_av_open(resource->domain, &av_attr, &map_av, NULL);
	assert_int_equal(err, 0);

	err = fi_endpoint(resource->domain, resource->info, &ep_fid, NULL);
	assert_int_equal(err, 0);
	assert_int_equal(fi_ep_bind(ep_fid, &map_av->fid, 0), 0);
	assert_int_equal(fi_ep_bind(ep_fid, &resource->cq->fid, FI_SEND | FI_RECV), 0);
	assert_int_equal(fi_enable(ep_fid), 0);
	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 9;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(map_av, &raw_addr, 1, &fi_addr, 0, NULL), 1);
	av_entry = (struct efa_rdm_av_entry *) (uintptr_t) fi_addr;
	assert_null(av_entry->cache_ep);

	/* first resolution creates the peer and caches it in the entry */
	peer = efa_rdm_ep_get_peer_explicit(ep, fi_addr);
	assert_non_null(peer);
	assert_ptr_equal(av_entry->cache_ep, ep);
	assert_ptr_equal(av_entry->cache_peer, peer);
	assert_int_equal(efa_rdm_peer_av_index(peer), av_entry->efa_av_entry.fi_addr);
	assert_int_equal(efa_rdm_peer_fi_addr(peer), fi_addr);

	/* later resolutions return the cached peer */
	assert_ptr_equal(efa_rdm_ep_get_peer_explicit(ep, fi_addr), peer);

	assert_int_equal(fi_close(&ep_fid->fid), 0);
	/* closing the endpoint cleared the cache before freeing the peer */
	assert_null(av_entry->cache_ep);
	assert_null(av_entry->cache_peer);

	assert_int_equal(fi_av_remove(map_av, &fi_addr, 1, 0), 0);
	assert_int_equal(fi_close(&map_av->fid), 0);
}

static struct fid_ep *test_av_map_open_rdm_ep(struct efa_resource *resource,
					       struct fid_av *av)
{
	struct fid_ep *ep_fid;

	assert_int_equal(fi_endpoint(resource->domain, resource->info, &ep_fid, NULL), 0);
	assert_int_equal(fi_ep_bind(ep_fid, &av->fid, 0), 0);
	assert_int_equal(fi_ep_bind(ep_fid, &resource->cq->fid, FI_SEND | FI_RECV), 0);
	assert_int_equal(fi_enable(ep_fid), 0);
	return ep_fid;
}

/**
 * @brief Two endpoints on one FI_AV_MAP AV: the first to resolve an entry owns
 * its peer cache, the second never overwrites it and uses its peer map, and
 * closing the owner releases the cache without affecting the second.
 */
void test_av_map_rdm_peer_cache_two_eps(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_av_attr av_attr = {.type = FI_AV_MAP};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	struct efa_rdm_av_entry *av_entry;
	struct efa_rdm_peer *peer1, *peer2;
	struct efa_rdm_ep *ep1, *ep2;
	struct fid_ep *ep1_fid, *ep2_fid;
	struct fid_av *map_av;
	fi_addr_t fi_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	assert_int_equal(fi_av_open(resource->domain, &av_attr, &map_av, NULL), 0);
	ep1_fid = test_av_map_open_rdm_ep(resource, map_av);
	ep2_fid = test_av_map_open_rdm_ep(resource, map_av);
	ep1 = container_of(ep1_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	ep2 = container_of(ep2_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	assert_int_equal(fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len), 0);
	raw_addr.qpn = 9;
	raw_addr.qkey = 0x1234;
	assert_int_equal(fi_av_insert(map_av, &raw_addr, 1, &fi_addr, 0, NULL), 1);
	av_entry = (struct efa_rdm_av_entry *) (uintptr_t) fi_addr;

	/* ep1 resolves first and owns the cache */
	peer1 = efa_rdm_ep_get_peer_explicit(ep1, fi_addr);
	assert_non_null(peer1);
	assert_ptr_equal(av_entry->cache_ep, ep1);
	assert_ptr_equal(av_entry->cache_peer, peer1);

	/* ep2 gets its own peer from its peer map; ep1's cache is untouched */
	peer2 = efa_rdm_ep_get_peer_explicit(ep2, fi_addr);
	assert_non_null(peer2);
	assert_ptr_not_equal(peer2, peer1);
	assert_ptr_equal(av_entry->cache_ep, ep1);
	assert_ptr_equal(av_entry->cache_peer, peer1);
	assert_ptr_equal(efa_rdm_ep_get_peer_explicit(ep2, fi_addr), peer2);
	assert_ptr_equal(efa_rdm_ep_get_peer_explicit(ep1, fi_addr), peer1);

	/* closing ep1 releases the cache; ep2 keeps resolving through its map */
	assert_int_equal(fi_close(&ep1_fid->fid), 0);
	assert_null(av_entry->cache_ep);
	assert_null(av_entry->cache_peer);
	assert_ptr_equal(efa_rdm_ep_get_peer_explicit(ep2, fi_addr), peer2);

	assert_int_equal(fi_close(&ep2_fid->fid), 0);
	assert_null(av_entry->cache_ep);
	assert_int_equal(fi_av_remove(map_av, &fi_addr, 1, 0), 0);
	assert_int_equal(fi_close(&map_av->fid), 0);
}
