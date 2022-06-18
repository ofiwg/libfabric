#include "efa_unit_tests.h"

int efa_unit_test_resource_construct(struct efa_resource *resource)
{
	int ret;
	struct fi_av_attr av_attr = {0};
	struct fi_cq_attr cq_attr = {0};

	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
	if (ret)
		goto err;

	ret = fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL);
	if (ret)
		goto err;

	ret = fi_domain(resource->fabric, resource->info, &resource->domain, NULL);
	if (ret)
		goto err;

	ret = fi_endpoint(resource->domain, resource->info, &resource->ep, NULL);
	if (ret)
		goto err;

	if (resource->eq_attr) {
		ret = fi_eq_open(resource->fabric, resource->eq_attr, &resource->eq, NULL);
		if (ret)
			goto err;
	}

	ret = fi_av_open(resource->domain, &av_attr, &resource->av, NULL);
	if (ret)
		goto err;

	ret = fi_cq_open(resource->domain, &cq_attr, &resource->cq, NULL);
	if (ret)
		goto err;

	return 0;
err:
	efa_unit_test_resource_destroy(resource);
	return ret;
}

/**
 * @brief Clean up test resources.
 * Note: Resources should be destroyed in order.
 * @param[in] resource	struct efa_resource to clean up.
 */
void efa_unit_test_resource_destroy(struct efa_resource *resource)
{
	if (resource->ep) {
		assert_int_equal(fi_close(&resource->ep->fid), 0);
	}

	if (resource->eq) {
		assert_int_equal(fi_close(&resource->eq->fid), 0);
	}

	if (resource->cq) {
		assert_int_equal(fi_close(&resource->cq->fid), 0);
	}

	if (resource->av) {
		assert_int_equal(fi_close(&resource->av->fid), 0);
	}

	if (resource->domain) {
		assert_int_equal(fi_close(&resource->domain->fid), 0);
	}

	if (resource->fabric) {
		assert_int_equal(fi_close(&resource->fabric->fid), 0);
	}

	if (resource->info) {
		fi_freeinfo(resource->info);
	}
}
