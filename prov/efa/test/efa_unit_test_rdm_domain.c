/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"
#include "rdm/efa_rdm_cq.h"
#include "rdm/efa_rdm_cntr.h"
#include "rdm/efa_rdm_atomic.h"

/**
 * @brief Verify the info type in struct efa_domain for efa RDM path
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_info_type_efa_rdm(void **state)
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
void test_efa_domain_rdm_no_bounce_buffer(void **state)
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
void test_efa_domain_rdm_attr_mr_allocated(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	efa_domain = container_of(resource->domain, struct efa_domain, util_domain.domain_fid);
	assert_true(efa_domain->device->rdm_info->domain_attr->mr_mode & FI_MR_ALLOCATED);
}
/**
 * @brief Verify that EFA RDM domains use the correct MR operations
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_domain_rdm_mr_ops(void **state)
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
void test_efa_domain_mr_cache_enabled(void **state)
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
	rdm_domain = (struct efa_rdm_domain *) efa_domain;

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
void test_efa_domain_mr_cache_disabled_with_mr_local(void **state)
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
	rdm_domain = (struct efa_rdm_domain *) efa_domain;

	/* Validate cache is disabled */
	test_efa_rdm_domain_mr_cache_common(rdm_domain, false);
	fi_freeinfo(hints);
}

/**
 * @brief Verify EFA RDM domains install RDM-specific domain ops
 *
 * After the domain split, struct efa_rdm_domain installs efa_domain_ops_rdm
 * which routes av/cq/endpoint/cntr operations to RDM-specific functions.
 *
 * @param[in]	state	struct efa_resource managed by the framework
 */
void test_efa_rdm_domain_open_installs_rdm_domain_ops(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_ops_domain *ops;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);
	ops = resource->domain->ops;

	assert_ptr_equal(ops->av_open, efa_rdm_av_open);
	assert_ptr_equal(ops->cq_open, efa_rdm_cq_open);
	assert_ptr_equal(ops->endpoint, efa_rdm_ep_open);
	assert_ptr_equal(ops->cntr_open, efa_rdm_cntr_open);
	assert_ptr_equal(ops->query_atomic, efa_rdm_atomic_query);
}
/**
 * @brief Verify FI_EFA_GDA_OPS is rejected on the RDM path
 *
 * GDA ops are direct-only. RDM domains should fail FI_EOPNOTSUPP
 * when the application requests them.
 *
 * @param[in]	state	struct efa_resource managed by the framework
 */
void test_efa_domain_gda_ops_rejected_for_rdm(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_ops_gda *efa_gda_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	ret = fi_open_ops(&resource->domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **)&efa_gda_ops, NULL);
	assert_int_equal(ret, -FI_EOPNOTSUPP);
}
