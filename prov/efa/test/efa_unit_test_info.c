#include "efa_unit_tests.h"

/**
 * @brief test that when a wrong fi_info was used to open resource, the error is handled
 * gracefully
 */
void test_info_open_ep_with_wrong_info()
{
	struct fi_info *hints, *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	struct fid_ep *ep = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_DGRAM);

	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	assert_int_equal(err, 0);

	/* dgram endpoint require FI_MSG_PREFIX */
	assert_int_equal(info->mode, FI_MSG_PREFIX);

	/* make the info wrong by setting the mode to 0 */
	info->mode = 0;

	err = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(err, 0);

	err = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(err, 0);

	/* because of the error in the info object, fi_endpoint() should fail with -FI_ENODATA */
	err = fi_endpoint(domain, info, &ep, NULL);
	assert_int_equal(err, -FI_ENODATA);
	assert_null(ep);

	err = fi_close(&domain->fid);
	assert_int_equal(err, 0);

	err = fi_close(&fabric->fid);
	assert_int_equal(err, 0);
}

/**
 * @brief test that we support older version of libfabric API version 1.1
 */
void test_info_open_ep_with_api_1_1_info()
{
	struct fi_info *hints, *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	struct fid_ep *ep = NULL;
	int err;

	hints = calloc(sizeof(struct fi_info), 1);
	assert_non_null(hints);

	hints->domain_attr = calloc(sizeof(struct fi_domain_attr), 1);
	assert_non_null(hints->domain_attr);

	hints->fabric_attr = calloc(sizeof(struct fi_fabric_attr), 1);
	assert_non_null(hints->fabric_attr);

	hints->ep_attr = calloc(sizeof(struct fi_ep_attr), 1);
	assert_non_null(hints->ep_attr);

	hints->fabric_attr->prov_name = "efa";
	hints->ep_attr->type = FI_EP_RDM;

	/* in libfabric API < 1.5, domain_attr->mr_mode is an enum with
	 * two options: FI_MR_BASIC or FI_MR_SCALABLE, (EFA does not support FI_MR_SCALABLE).
	 *
	 * Additional information about memory registration is specified as bits in
	 * "mode". For example, the requirement of local memory registration
	 * is specified as FI_LOCAL_MR.
	 */
	hints->mode = FI_LOCAL_MR;
	hints->domain_attr->mr_mode = FI_MR_BASIC;

	err = fi_getinfo(FI_VERSION(1, 1), NULL, NULL, 0ULL, hints, &info);
	assert_int_equal(err, 0);

	err = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(err, 0);

	err = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(err, 0);

	err = fi_endpoint(domain, info, &ep, NULL);
	assert_int_equal(err, 0);

	err = fi_close(&ep->fid);
	assert_int_equal(err, 0);

	err = fi_close(&domain->fid);
	assert_int_equal(err, 0);

	err = fi_close(&fabric->fid);
	assert_int_equal(err, 0);
}
