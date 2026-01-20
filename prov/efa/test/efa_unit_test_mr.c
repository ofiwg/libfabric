/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

static void test_efa_mr_impl(struct efa_domain *efa_domain, struct fid_mr *mr,
			int mr_reg_count, int mr_reg_size, bool gdrcopy_flag)
{
	struct efa_mr *efa_mr;

	assert_int_equal(ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct), (int64_t)mr_reg_count);
	assert_int_equal(ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz), (int64_t)mr_reg_size);

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
	int baseline_ct, baseline_sz;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* fi_endpoint calls ofi_bufpool_grow, which registers mr */
	baseline_ct = ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct);
	baseline_sz = ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz);

	buf = malloc(mr_size);
	assert_non_null(buf);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);
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
	int baseline_ct, baseline_sz;

	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						 FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* fi_endpoint calls ofi_bufpool_grow, which registers mr */
	baseline_ct = ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct);
	baseline_sz = ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz);

	buf = malloc(mr_size);
	assert_non_null(buf);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);
	free(buf);
}

void test_efa_rdm_mr_reg_host_memory_overlapping_buffers(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 1024;
	void *buf;
	struct fid_mr *mr_1 = NULL, *mr_2 = NULL;
	int baseline_ct, baseline_sz;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* fi_endpoint calls ofi_bufpool_grow, which registers mr */
	baseline_ct = ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct);
	baseline_sz = ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz);

	buf = malloc(mr_size);
	assert_non_null(buf);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr_1, NULL),
			 0);

	/* No GDRCopy registration for host memory */
	test_efa_mr_impl(efa_domain, mr_1, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr_2, NULL),
			 0);

	test_efa_mr_impl(efa_domain, mr_2, baseline_ct + 2, baseline_sz + mr_size * 2, false);

	assert_int_equal(fi_close(&mr_1->fid), 0);
	test_efa_mr_impl(efa_domain, mr_2, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr_2->fid), 0);
	test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);

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
	int err, baseline_ct, baseline_sz;

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
		/* fi_endpoint calls ofi_bufpool_grow, which registers mr */
		baseline_ct = ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct);
		baseline_sz = ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz);

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
		test_efa_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, true);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);

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
	int err, baseline_ct, baseline_sz;

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
		/* fi_endpoint calls ofi_bufpool_grow, which registers mr */
		baseline_ct = ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct);
		baseline_sz = ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz);

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
		test_efa_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, false);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);

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

void test_efa_direct_mr_reg_rdma_read_not_supported(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = {0};
	struct iovec iovec;
	int err;
	uint32_t efa_device_caps_orig;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0),
						    resource->hints,
						    true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Mock device to not support RDMA read */
	efa_device_caps_orig = efa_domain->device->device_caps;
	efa_domain->device->device_caps &= ~EFADV_DEVICE_ATTR_CAPS_RDMA_READ;

	buf = malloc(mr_size);
	assert_non_null(buf);

	mr_reg_attr.access = FI_READ;
	mr_reg_attr.iface = FI_HMEM_SYSTEM;
	iovec.iov_base = buf;
	iovec.iov_len = mr_size;
	mr_reg_attr.mr_iov = &iovec;
	mr_reg_attr.iov_count = 1;

	err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
	assert_int_equal(err, -FI_EOPNOTSUPP);

	free(buf);
	efa_domain->device->device_caps = efa_device_caps_orig;
}

void test_efa_direct_mr_reg_rdma_write_not_supported(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = {0};
	struct iovec iovec;
	int err;
	uint32_t efa_device_caps_orig;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0),
						    resource->hints,
						    true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	efa_device_caps_orig = efa_domain->device->device_caps;
	/* Mock device to not support RDMA write */
#if HAVE_CAPS_RDMA_WRITE
	efa_domain->device->device_caps &= ~EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE;
#endif

	buf = malloc(mr_size);
	assert_non_null(buf);

	mr_reg_attr.access = FI_WRITE;
	mr_reg_attr.iface = FI_HMEM_SYSTEM;
	iovec.iov_base = buf;
	iovec.iov_len = mr_size;
	mr_reg_attr.mr_iov = &iovec;
	mr_reg_attr.iov_count = 1;

	err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
	assert_int_equal(err, -FI_EOPNOTSUPP);

	free(buf);
	efa_domain->device->device_caps = efa_device_caps_orig;
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access with no access flags
 * 
 * When no access flags are provided, the function should default to 
 * FI_SEND | FI_RECV and return IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ.
 */
void test_efa_mr_ofi_to_ibv_access_no_access(struct efa_resource **state)
{
	int ibv_access;
	
	ibv_access = efa_mr_ofi_to_ibv_access(0, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access with one flag when rdma read and write are available
 * 
 */
void test_efa_mr_ofi_to_ibv_access_one_flag(struct efa_resource **state)
{
	int ibv_access;
	
	ibv_access = efa_mr_ofi_to_ibv_access(FI_SEND, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_REMOTE_READ);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_RECV, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_READ, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_WRITE, true, true);
	assert_int_equal(ibv_access, 0);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_REMOTE_READ, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_REMOTE_READ);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_REMOTE_WRITE, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access when RDMA read not supported
 */
void test_efa_mr_ofi_to_ibv_access_read_not_supported(struct efa_resource **state)
{
	int ibv_access;
	
	ibv_access = efa_mr_ofi_to_ibv_access(FI_READ, false, false);
	assert_int_equal(ibv_access, 0);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_REMOTE_READ, false, false);
	assert_int_equal(ibv_access, 0);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access when RDMA write not supported
 * 
 * When device doesn't support RDMA write, emulate with RDMA read
 */
void test_efa_mr_ofi_to_ibv_access_write_not_supported(struct efa_resource **state)
{
	int ibv_access;

	ibv_access = efa_mr_ofi_to_ibv_access(FI_WRITE, true, false);
	assert_int_equal(ibv_access, IBV_ACCESS_REMOTE_READ);

	ibv_access = efa_mr_ofi_to_ibv_access(FI_REMOTE_WRITE, true, false);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access with FI_REMOTE_READ | FI_REMOTE_WRITE,
 * only read supported
 *
 * When only remote read is supported, FI_REMOTE_READ gets IBV_ACCESS_REMOTE_READ
 * and FI_REMOTE_WRITE gets IBV_ACCESS_LOCAL_WRITE.
 */
void test_efa_mr_ofi_to_ibv_access_remote_read_write_read_only_supported(struct efa_resource **state)
{
	int ibv_access;
	
	ibv_access = efa_mr_ofi_to_ibv_access(FI_REMOTE_READ | FI_REMOTE_WRITE, true, false);
	assert_int_equal(ibv_access, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access with all access flags combined
 * 
 * Test all OFI access flags together with full device support.
 */
void test_efa_mr_ofi_to_ibv_access_all_flags_supported(struct efa_resource **state)
{
	int ibv_access;
	uint64_t all_flags = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
	
	ibv_access = efa_mr_ofi_to_ibv_access(all_flags, true, true);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
}

/**
 * @brief Test efa_mr_ofi_to_ibv_access with all access flags, no device support
 */
void test_efa_mr_ofi_to_ibv_access_all_flags_not_supported(struct efa_resource **state)
{
	int ibv_access;
	uint64_t all_flags = FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
	
	ibv_access = efa_mr_ofi_to_ibv_access(all_flags, false, false);
	assert_int_equal(ibv_access, IBV_ACCESS_LOCAL_WRITE);
}
/**
 * @brief Test efa_mr_internal_regv does not create shm MR
 *
 * This test verifies that efa_mr_internal_regv only creates EFA MR
 * and does not create a corresponding SHM MR, even when SHM domain exists.
 */
void test_efa_mr_internal_regv_no_shm_mr(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct efa_mr *efa_mr;
	struct iovec iov;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	assert_int_equal(efa_mr_internal_regv(resource->domain, &iov, 1,
					      FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);
	assert_non_null(mr);

	efa_mr = container_of(mr, struct efa_mr, mr_fid);
	/* Verify that shm_mr is NULL even if shm_domain exists */
	assert_null(efa_mr->shm_mr);

	assert_int_equal(fi_close(&mr->fid), 0);
	free(buf);
}