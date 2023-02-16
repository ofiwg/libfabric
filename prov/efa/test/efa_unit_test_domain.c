#include "efa_unit_tests.h"

static enum test_domain_number{
	rdma_test,
	hmem_cuda_test
};

void set_resources_and_check_value(const int val, const enum test_domain_number test_number)
{
	int ret = 0;
	struct fi_info *hints, *info;
	struct fid_fabric *fabric = NULL;
	struct fid_domain *domain = NULL;
	struct efa_domain *efa_domain;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM);

	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &info);
	assert_int_equal(ret, 0);

	ret = fi_fabric(info->fabric_attr, &fabric, NULL);
	assert_int_equal(ret, 0);

	ret = fi_domain(fabric, info, &domain, NULL);
	assert_int_equal(ret, 0);

	efa_domain = container_of(domain, struct efa_domain,
			util_domain.domain_fid);

	if (test_number)
		assert_int_equal(efa_domain->cuda_xfer_setting, val);
	else
		assert_int_equal(efa_domain->use_device_rdma, val);

	assert_int_equal(fi_close(&domain->fid), 0);
	assert_int_equal(fi_close(&fabric->fid), 0);
	fi_freeinfo(info);

	return;
}

/* Test the case of using EFA device's RDMA */
void test_efa_use_device_rdma()
{
	int ret = 0;
	ret = setenv("FI_EFA_USE_DEVICE_RDMA", "1", 1);
	if (ret) {
		fprintf(stdout, "Failed to set the environment variable FI_EFA_USE_DEVICE_RDMA \n");
		return;
	}
	set_resources_and_check_value(true, rdma_test);
	return;
}


/* Test the case of not using EFA device's RDMA */
void test_efa_dont_use_device_rdma()
{
	int ret = 0;
	ret = setenv("FI_EFA_USE_DEVICE_RDMA", "0", 1);
	if (ret) {
		fprintf(stdout, "Failed to set the environment variable FI_EFA_USE_DEVICE_RDMA \n");
		return;
	}
	set_resources_and_check_value(false, rdma_test);
	return;
}

/* Test the case of using HMEM CUDA transfer */
void test_efa_use_hmem_cuda_xfer()
{
	int ret = 0;
	ret = setenv("FI_HMEM_CUDA_ENABLE_XFER", "1", 1);
	if (ret) {
		fprintf(stdout, "Failed to set the environment variable FI_HMEM_CUDA_ENABLE_XFER \n");
		return;
	}
	set_resources_and_check_value(CUDA_XFER_ENABLED, hmem_cuda_test);
	return;
}

/* Test the case of not using HMEM CUDA transfer */
void test_efa_dont_use_hmem_cuda_xfer()
{
	int ret = 0;
	ret = setenv("FI_HMEM_CUDA_ENABLE_XFER", "0", 1);
	if (ret) {
		fprintf(stdout, "Failed to set the environment variable FI_HMEM_CUDA_ENABLE_XFER \n");
		return;
	}
	set_resources_and_check_value(CUDA_XFER_DISABLED, hmem_cuda_test);
	return;
}
