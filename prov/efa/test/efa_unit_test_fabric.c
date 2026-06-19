/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

/**
 * @brief Verify query("mixed_hmem_iov") returns true on efa-direct,
 * which is the fabric where this feature is advertised.
 */
void test_efa_fabric_open_ops_feature_known(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_feature_ops *feat_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_open_ops(&resource->fabric->fid, FI_EFA_FEATURE_OPS, 0,
			  (void **) &feat_ops, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(feat_ops->query);
	assert_true(feat_ops->query("mixed_hmem_iov"));
}

/**
 * @brief Verify query("mixed_hmem_iov") returns false on efa (RDM),
 * which does not currently advertise the feature.
 */
void test_efa_fabric_open_ops_feature_not_on_proto(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_feature_ops *feat_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	ret = fi_open_ops(&resource->fabric->fid, FI_EFA_FEATURE_OPS, 0,
			  (void **) &feat_ops, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(feat_ops->query);
	assert_false(feat_ops->query("mixed_hmem_iov"));
}

/**
 * @brief Verify query() returns false for an unknown feature and for NULL.
 */
void test_efa_fabric_open_ops_feature_unknown(void **state)
{
	struct efa_resource *resource = *state;
	struct fi_efa_feature_ops *feat_ops;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);

	ret = fi_open_ops(&resource->fabric->fid, FI_EFA_FEATURE_OPS, 0,
			  (void **) &feat_ops, NULL);
	assert_int_equal(ret, 0);
	assert_false(feat_ops->query("no_such_feature"));
	assert_false(feat_ops->query(NULL));
}
