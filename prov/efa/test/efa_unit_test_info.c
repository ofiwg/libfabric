#include <stdlib.h>
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

static void test_info_check_shm_info_from_hints(struct fi_info *hints)
{
	struct fi_info *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	int err;
	struct efa_domain *efa_domain;

	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	/* Do nothing if the current setup does not support FI_HMEM */
	if (err && hints->caps & FI_HMEM) {
		return;
	}
	assert_int_equal(err, 0);

	err = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(err, 0);

	err = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(err, 0);

	efa_domain = container_of(domain, struct efa_domain, util_domain.domain_fid);
	if (efa_domain->shm_info) {
		if (hints->caps & FI_HMEM)
			assert_true(efa_domain->shm_info->caps & FI_HMEM);
		else
			assert_false(efa_domain->shm_info->caps & FI_HMEM);
	}

	fi_close(&domain->fid);

	fi_close(&fabric->fid);

	fi_freeinfo(info);
}
/**
 * @brief Check shm info created by efa_domain() has correct caps.
 *
 */
void test_info_check_shm_info()
{
	struct fi_info *hints;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM);

	hints->caps |= FI_HMEM;
	test_info_check_shm_info_from_hints(hints);

	hints->caps &= ~FI_HMEM;
	test_info_check_shm_info_from_hints(hints);

}

/**
 * @brief Check that case when a user requested FI_HMEM support
 *        using libfabric API < 1.18,
 */
void test_info_check_hmem_cuda_support_on_api_lt_1_18()
{
	struct fi_info *hints, *info = NULL;
	int err;

	if (!hmem_ops[FI_HMEM_CUDA].initialized)
		skip();

	hints = efa_unit_test_alloc_hints(FI_EP_RDM);

	hints->caps |= FI_HMEM;
	hints->domain_attr->mr_mode |= FI_MR_HMEM;

	/* For libfabric API < 1.18,
	 * on a system that support GPUDirect RDMA read,
	 * HMEM cuda is available when GPUDirect RDMA is available,
	 * and environment variable FI_EFA_USE_DEVICE_RDMA set to 1/on/true
	 * otherwise it is not available.
	 */
	setenv("FI_EFA_USE_DEVICE_RDMA", "1", true /* overwrite */);
	err = fi_getinfo(FI_VERSION(1, 6), NULL, NULL, 0, hints, &info);
	if (efa_device_support_rdma_read()) {
		assert_int_equal(err, 0);
		fi_freeinfo(info);
	} else {
		assert_int_equal(err, -FI_ENODATA);
	}

	setenv("FI_EFA_USE_DEVICE_RDMA", "0", true /* overwrite */);
	err = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, -FI_ENODATA);

	unsetenv("FI_EFA_USE_DEVICE_RDMA");
}

/**
 * @brief Check that case when a user requested FI_HMEM support
 *        using libfabric API >= 1.18,
 */
void test_info_check_hmem_cuda_support_on_api_ge_1_18()
{
	struct fi_info *hints, *info = NULL;
	int err;

	if (!hmem_ops[FI_HMEM_CUDA].initialized)
		skip();

	hints = efa_unit_test_alloc_hints(FI_EP_RDM);

	hints->caps |= FI_HMEM;
	hints->domain_attr->mr_mode |= FI_MR_HMEM;

	/* Piror to version 1.18, libfabric EFA provider support CUDA
	 * memory only when GPUDirect RDMA is available.
	 * In version 1.18, libfabric EFA provider implemented
	 * universal CUDA support through CUDA library.
	 * However, this feature (universal CUDA support) can cause some
	 * middleware to deadlock, thus it is only available
	 * when a user is using 1.18 API.
	 */
	err = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, 0);
	fi_freeinfo(info);
}

/**
 * @brief Check that EFA does not claim support of FI_HMEM when
 *        it is not requested
 */
void test_info_check_no_hmem_support_when_not_requested()
{
	struct fi_info *hints, *info = NULL;
	int err;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM);

	err = fi_getinfo(FI_VERSION(1,6), NULL, NULL, 0, hints, &info);
	assert_int_equal(err, 0);
	assert_non_null(info);
	assert_false(info->caps & FI_HMEM);
	fi_freeinfo(info);
}