/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

/**
 * @brief Verify the info type in struct efa_domain for efa RDM path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_info_type_efa_rdm(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->info_type == EFA_INFO_RDM);
}
/**
 * @brief Verify bounce buffer is NOT allocated for efa RDM domain
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_rdm_no_bounce_buffer(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_null(efa_domain->zero_byte_bounce_buf);
	assert_null(efa_domain->zero_byte_bounce_buf_mr);
}
/**
 * @brief Verify FI_MR_ALLOCATED is set for efa rdm path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_rdm_attr_mr_allocated(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->rdm_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}
/**
 * @brief Verify that the domain level peer lists get cleared when an endpoint is closed
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_peer_list_cleared(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct efa_rdm_domain *rdm_domain;
	struct fid_ep *ep1, *ep2;
	struct efa_rdm_ep *efa_rdm_ep1, *efa_rdm_ep2;
	struct efa_rdm_peer *peer1, *peer2, *peer3, *peer4;
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2, addr3, addr4;
	int err, num_addr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	rdm_domain = efa_rdm_domain_from_efa_domain(efa_domain);

	// Create two endpoints
	err = fi_endpoint(resource->domain, resource->info, &ep1, NULL);
	assert_int_equal(err, 0);
	err = fi_endpoint(resource->domain, resource->info, &ep2, NULL);
	assert_int_equal(err, 0);

	// Bind endpoints to AV and enable them
	err = fi_ep_bind(ep1, &resource->av->fid, 0);
	assert_int_equal(err, 0);
	err = fi_ep_bind(ep2, &resource->av->fid, 0);
	assert_int_equal(err, 0);
	err = fi_ep_bind(ep1, &resource->cq->fid, FI_SEND | FI_RECV);
	assert_int_equal(err, 0);
	err = fi_ep_bind(ep2, &resource->cq->fid, FI_SEND | FI_RECV);
	assert_int_equal(err, 0);
	err = fi_enable(ep1);
	assert_int_equal(err, 0);
	err = fi_enable(ep2);
	assert_int_equal(err, 0);

	efa_rdm_ep1 = container_of(ep1, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep2 = container_of(ep2, struct efa_rdm_ep, base_ep.util_ep.ep_fid);

	// Get base address and create different addresses
	err = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);

	// Insert addresses to create peers
	raw_addr.qpn = 1; raw_addr.qkey = 0x1234;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr1, 0, NULL);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2; raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr2, 0, NULL);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 3; raw_addr.qkey = 0x9abc;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr3, 0, NULL);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 4; raw_addr.qkey = 0xdef0;
	num_addr = fi_av_insert(resource->av, &raw_addr, 1, &addr4, 0, NULL);
	assert_int_equal(num_addr, 1);

	// Create peers through normal code path
	peer1 = efa_rdm_ep_get_peer(efa_rdm_ep1, addr1);
	assert_non_null(peer1);
	peer2 = efa_rdm_ep_get_peer(efa_rdm_ep1, addr2);
	assert_non_null(peer2);
	peer3 = efa_rdm_ep_get_peer(efa_rdm_ep2, addr3);
	assert_non_null(peer3);
	peer4 = efa_rdm_ep_get_peer(efa_rdm_ep2, addr4);
	assert_non_null(peer4);

	// Manually add peers to domain lists to simulate the conditions
	dlist_insert_tail(&peer1->handshake_queued_entry, &rdm_domain->handshake_queued_peer_list);
	peer1->flags |= EFA_RDM_PEER_HANDSHAKE_QUEUED;
	dlist_insert_tail(&peer2->rnr_backoff_entry, &rdm_domain->peer_backoff_list);
	peer2->flags |= EFA_RDM_PEER_IN_BACKOFF;
	dlist_insert_tail(&peer3->handshake_queued_entry, &rdm_domain->handshake_queued_peer_list);
	peer3->flags |= EFA_RDM_PEER_HANDSHAKE_QUEUED;
	dlist_insert_tail(&peer4->rnr_backoff_entry, &rdm_domain->peer_backoff_list);
	peer4->flags |= EFA_RDM_PEER_IN_BACKOFF;

	// Close endpoints - this should clear the domain lists
	fi_close(&ep1->fid);
	fi_close(&ep2->fid);

	// Verify domain lists are cleared
	assert_true(dlist_empty(&rdm_domain->peer_backoff_list));
	assert_true(dlist_empty(&rdm_domain->handshake_queued_peer_list));
}
/**
 * @brief Verify that EFA RDM domains use the correct MR operations
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_rdm_mr_ops(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);

	/* RDM domains should use efa_rdm_domain_mr_ops */
	assert_ptr_equal(efa_domain->util_domain.domain_fid.mr, &efa_rdm_domain_mr_ops);
	assert_int_equal(efa_domain->info_type, EFA_INFO_RDM);
}
/**
 * @brief Common helper function to validate MR cache configuration for RDM domains
 *
 * @param efa_domain EFA RDM domain to test
 * @param cache_expected Whether MR cache should be enabled
 */
static void test_efa_rdm_domain_mr_cache_common(struct efa_rdm_domain *rdm_domain, bool cache_expected)
{
	struct ofi_mr_cache *cache = rdm_domain->cache;

	/* This helper is only for RDM domains */
	assert_int_equal(rdm_domain->efa_domain.info_type, EFA_INFO_RDM);

	if (cache_expected) {
		/* Test Case: MR cache should be available */
		assert_non_null(cache);
		assert_true(efa_is_cache_available(rdm_domain));

		/* Validate entry_data_size is correct for efa_rdm_mr */
		assert_int_equal(cache->entry_data_size, sizeof(struct efa_rdm_mr));

		/* Validate add_region function pointer */
		assert_ptr_equal(cache->add_region, efa_rdm_mr_cache_entry_reg);

		/* Validate delete_region function pointer */
		assert_ptr_equal(cache->delete_region, efa_rdm_mr_cache_entry_dereg);
	} else {
		/* Test Case: MR cache should be disabled for RDM */
		assert_null(cache);
		assert_false(efa_is_cache_available(rdm_domain));
	}
}
/**
 * @brief Test MR cache happy path: no FI_MR_LOCAL, cache enabled
 *
 * This test validates that when the application doesn't request FI_MR_LOCAL
 * and efa_mr_cache_enable is true, the MR cache is properly initialized
 * and configured with correct function pointers and data structures.
 */
void test_efa_domain_mr_cache_enabled(struct efa_resource **state)
{
#ifdef ENABLE_ASAN
	skip();
#else
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct efa_rdm_domain *rdm_domain;
	struct fi_info *hints;

	/* Create hints without FI_MR_LOCAL to enable cache */
	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	rdm_domain = efa_rdm_domain_from_efa_domain(efa_domain);

	/* Validate cache is enabled and properly configured */
	test_efa_rdm_domain_mr_cache_common(rdm_domain, true);
	fi_freeinfo(hints);
#endif
}
/**
 * @brief Test MR cache disabled path: FI_MR_LOCAL requested
 *
 * This test validates that when the application requests FI_MR_LOCAL,
 * the MR cache is disabled and the domain uses direct MR registration.
 */
void test_efa_domain_mr_cache_disabled_with_mr_local(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct efa_rdm_domain *rdm_domain;
	struct fi_info *hints;

	/* Create hints with FI_MR_LOCAL to disable cache */
	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode |= FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	rdm_domain = efa_rdm_domain_from_efa_domain(efa_domain);

	/* Validate cache is disabled */
	test_efa_rdm_domain_mr_cache_common(rdm_domain, false);
	fi_freeinfo(hints);
}
