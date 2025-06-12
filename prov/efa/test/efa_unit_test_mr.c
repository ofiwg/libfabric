/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

static void test_efa_mr_impl(struct efa_domain *efa_domain, struct fid_mr *mr,
			int mr_reg_count, int mr_reg_size, bool gdrcopy_flag)
{
	struct efa_mr *efa_mr;

	assert_int_equal(efa_domain->ibv_mr_reg_ct, mr_reg_count);
	assert_int_equal(efa_domain->ibv_mr_reg_sz, mr_reg_size);

	if (mr) {
		efa_mr = container_of(mr, struct efa_mr, mr_fid);
		if (cuda_is_gdrcopy_enabled()) {
			if (gdrcopy_flag)
				assert_true(efa_mr->peer.flags &
					    OFI_HMEM_DATA_DEV_REG_HANDLE);
			else
				assert_false(efa_mr->peer.flags &
					     OFI_HMEM_DATA_DEV_REG_HANDLE);
		}
	}
}

void test_efa_rdm_mr_reg_host_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	test_efa_mr_impl(efa_domain, mr, 0, 0, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr, 1, mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, 0, 0, false);
	free(buf);
}

void test_efa_rdm_mr_reg_host_memory_no_mr_local(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_info *hints;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						 FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	test_efa_mr_impl(efa_domain, mr, 0, 0, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr, 1, mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, 0, 0, false);
	free(buf);
}

void test_efa_rdm_mr_reg_host_memory_overlapping_buffers(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 1024;
	void *buf;
	struct fid_mr *mr_1 = NULL, *mr_2 = NULL;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	test_efa_mr_impl(efa_domain, mr_1, 0, 0, false);
	test_efa_mr_impl(efa_domain, mr_2, 0, 0, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr_1, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr_1, 1, mr_size, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr_2, NULL),
			 0);

	test_efa_mr_impl(efa_domain, mr_2, 2, mr_size * 2, false);

	assert_int_equal(fi_close(&mr_1->fid), 0);
	test_efa_mr_impl(efa_domain, mr_2, 1, mr_size, false);

	assert_int_equal(fi_close(&mr_2->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, 0, 0, false);

	free(buf);
}

#if HAVE_CUDA
void test_efa_rdm_mr_reg_cuda_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = { 0 };
	struct iovec iovec;
	int err;

	if (hmem_ops[FI_HMEM_CUDA].initialized &&
	    g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device) {
		resource->hints = efa_unit_test_alloc_hints_hmem(
			FI_EP_RDM, EFA_FABRIC_NAME);
		efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
							    FI_VERSION(2, 0),
							    resource->hints,
							    true, true);

		efa_domain = container_of(resource->domain, struct efa_domain,
					  util_domain.domain_fid);
		test_efa_mr_impl(efa_domain, mr, 0, 0, false);

		err = ofi_cudaMalloc(&buf, mr_size);
		assert_int_equal(err, 0);
		assert_non_null(buf);

		mr_reg_attr.access = FI_SEND | FI_RECV;
		mr_reg_attr.iface = FI_HMEM_CUDA;
		iovec.iov_base = buf;
		iovec.iov_len = mr_size;
		mr_reg_attr.mr_iov = &iovec;
		mr_reg_attr.iov_count = 1;

		err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
		assert_int_equal(err, 0);

		/* FI_MR_DMABUF flag was not set, so GDRCopy should be registered if available */
		test_efa_mr_impl(efa_domain, mr, 1, mr_size, true);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_mr_impl(efa_domain, NULL, 0, 0, false);

		err = ofi_cudaFree(buf);
		assert_int_equal(err, 0);
	}
}
#else
void test_efa_rdm_mr_reg_cuda_memory(struct efa_resource **state)
{
	skip();
}
#endif

#if HAVE_CUDA
void test_efa_direct_mr_reg_no_gdrcopy(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = { 0 };
	struct iovec iovec;
	int err;

	if (g_efa_hmem_info[FI_HMEM_CUDA].initialized &&
	    g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device) {
		resource->hints = efa_unit_test_alloc_hints_hmem(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
							    FI_VERSION(2, 0),
							    resource->hints,
							    true, true);

		efa_domain = container_of(resource->domain, struct efa_domain,
					  util_domain.domain_fid);
		test_efa_mr_impl(efa_domain, mr, 0, 0, false);

		err = ofi_cudaMalloc(&buf, mr_size);
		assert_int_equal(err, 0);
		assert_non_null(buf);

		mr_reg_attr.access = FI_SEND | FI_RECV;
		mr_reg_attr.iface = FI_HMEM_CUDA;
		iovec.iov_base = buf;
		iovec.iov_len = mr_size;
		mr_reg_attr.mr_iov = &iovec;
		mr_reg_attr.iov_count = 1;

		err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
		assert_int_equal(err, 0);

		/* no GDRCopy in the efa-direct path */
		test_efa_mr_impl(efa_domain, mr, 1, mr_size, false);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_mr_impl(efa_domain, NULL, 0, 0, false);

		err = ofi_cudaFree(buf);
		assert_int_equal(err, 0);
	}
}
#else
void test_efa_direct_mr_reg_no_gdrcopy(struct efa_resource **state)
{
	skip();
}
#endif
