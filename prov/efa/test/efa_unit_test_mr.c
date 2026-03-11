/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

static void test_efa_mr_impl(struct efa_domain *efa_domain, struct fid_mr *mr,
			int mr_reg_count, int mr_reg_size)
{
	assert_int_equal(ofi_atomic_get64(&efa_domain->ibv_mr_reg_ct), (int64_t)mr_reg_count);
	assert_int_equal(ofi_atomic_get64(&efa_domain->ibv_mr_reg_sz), (int64_t)mr_reg_size);
}

static void test_efa_rdm_mr_impl(struct efa_domain *efa_domain, struct fid_mr *mr,
				int mr_reg_count, int mr_reg_size, bool gdrcopy_flag)
{
	struct efa_rdm_mr *efa_rdm_mr;

	test_efa_mr_impl(efa_domain, mr, mr_reg_count, mr_reg_size);

	if (mr) {
		efa_rdm_mr = container_of(mr, struct efa_rdm_mr, efa_mr.mr_fid);

		/* Test RDM-specific MR map insertion status */
		assert_true(efa_rdm_mr->inserted_to_mr_map);

		/* Test SHM MR creation based on domain availability */
		if (efa_domain->shm_domain) {
			assert_non_null(efa_rdm_mr->shm_mr);
		} else {
			assert_null(efa_rdm_mr->shm_mr);
		}

		/* Test needs_sync flag for CUDA memory */
		if (efa_rdm_mr->efa_mr.iface == FI_HMEM_CUDA) {
			assert_true(efa_rdm_mr->needs_sync);
		} else {
			assert_false(efa_rdm_mr->needs_sync);
		}

		if (cuda_is_gdrcopy_enabled()) {
			if (gdrcopy_flag) {
				assert_true(efa_rdm_mr->flags &
					    OFI_HMEM_DATA_DEV_REG_HANDLE);
				assert_non_null(efa_rdm_mr->hmem_data);
			} else {
				assert_false(efa_rdm_mr->flags &
					     OFI_HMEM_DATA_DEV_REG_HANDLE);
				assert_null(efa_rdm_mr->hmem_data);
			}
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
	test_efa_rdm_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_rdm_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);
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
	test_efa_rdm_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr->fid), 0);
	test_efa_rdm_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);
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
	test_efa_rdm_mr_impl(efa_domain, mr_1, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr_2, NULL),
			 0);

	test_efa_rdm_mr_impl(efa_domain, mr_2, baseline_ct + 2, baseline_sz + mr_size * 2, false);

	assert_int_equal(fi_close(&mr_1->fid), 0);
	test_efa_rdm_mr_impl(efa_domain, mr_2, baseline_ct + 1, baseline_sz + mr_size, false);

	assert_int_equal(fi_close(&mr_2->fid), 0);
	test_efa_rdm_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);

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
		test_efa_rdm_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size, true);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_rdm_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz, false);

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
void test_efa_direct_mr_reg_cuda_memory(struct efa_resource **state)
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

		test_efa_mr_impl(efa_domain, mr, baseline_ct + 1, baseline_sz + mr_size);

		assert_int_equal(fi_close(&mr->fid), 0);
		test_efa_mr_impl(efa_domain, NULL, baseline_ct, baseline_sz);

		err = ofi_cudaFree(buf);
		assert_int_equal(err, 0);
	}
}
#else
void test_efa_direct_mr_reg_cuda_memory(struct efa_resource **state)
{
	skip();
}
#endif

void test_efa_direct_mr_reg_fi_read_support_status(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = {0};
	struct iovec iovec;
	int err;
	fi_addr_t fi_addr;
	bool fi_rma_supported;

	fi_rma_supported = test_efa_rma_prep(resource, &fi_addr);

	buf = malloc(mr_size);
	assert_non_null(buf);

	mr_reg_attr.access = FI_READ;
	mr_reg_attr.iface = FI_HMEM_SYSTEM;
	iovec.iov_base = buf;
	iovec.iov_len = mr_size;
	mr_reg_attr.mr_iov = &iovec;
	mr_reg_attr.iov_count = 1;

	err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
	assert_int_equal(err, fi_rma_supported ? FI_SUCCESS : -FI_EOPNOTSUPP);

	free(buf);
}

void test_efa_direct_mr_reg_fi_write_support_status(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct fi_mr_attr mr_reg_attr = {0};
	struct iovec iovec;
	int err;
	fi_addr_t fi_addr;
	bool fi_rma_supported;

	fi_rma_supported = test_efa_rma_prep(resource, &fi_addr);

	buf = malloc(mr_size);
	assert_non_null(buf);

	mr_reg_attr.access = FI_WRITE;
	mr_reg_attr.iface = FI_HMEM_SYSTEM;
	iovec.iov_base = buf;
	iovec.iov_len = mr_size;
	mr_reg_attr.mr_iov = &iovec;
	mr_reg_attr.iov_count = 1;

	err = fi_mr_regattr(resource->domain, &mr_reg_attr, 0, &mr);
	assert_int_equal(err, fi_rma_supported ? FI_SUCCESS : -FI_EOPNOTSUPP);

	free(buf);
}

void test_efa_mr_validate_regattr_invalid_iov_count(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_mr_attr mr_attr = { 0 };
	struct iovec iov[2];
	void *buf1, *buf2;
	size_t mr_size = 64;
	struct fid_mr *mr = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	buf1 = malloc(mr_size);
	buf2 = malloc(mr_size);
	assert_non_null(buf1);
	assert_non_null(buf2);

	iov[0].iov_base = buf1;
	iov[0].iov_len = mr_size;
	iov[1].iov_base = buf2;
	iov[1].iov_len = mr_size;
	mr_attr.mr_iov = iov;
	mr_attr.iov_count = 2; /* Exceeds EFA_MR_IOV_LIMIT */
	mr_attr.access = FI_SEND | FI_RECV;
	mr_attr.iface = FI_HMEM_SYSTEM;

	/* Test through the actual API */
	ret = fi_mr_regattr(resource->domain, &mr_attr, 0, &mr);
	assert_int_equal(ret, -FI_EINVAL);
	assert_null(mr);

	free(buf1);
	free(buf2);
}

void test_efa_mr_validate_regattr_uninitialized_iface(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_mr_attr mr_attr = { 0 };
	struct iovec iov;
	void *buf;
	size_t mr_size = 64;
	struct fid_mr *mr = NULL;
	int ret;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;
	mr_attr.mr_iov = &iov;
	mr_attr.iov_count = 1;
	mr_attr.access = FI_SEND | FI_RECV;
	mr_attr.iface = FI_HMEM_CUDA;

	/* Mock CUDA as uninitialized by temporarily setting it to false */
	g_efa_hmem_info[FI_HMEM_CUDA].initialized = false;

	/* Test through the actual API */
	ret = fi_mr_regattr(resource->domain, &mr_attr, 0, &mr);
	assert_int_equal(ret, -FI_ENOSYS);
	assert_null(mr);

	free(buf);
}

/**
 * @brief Test RDM MR structure casting safety
 */
void test_efa_rdm_mr_structure_casting(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct efa_mr *efa_mr;
	struct efa_rdm_mr *efa_rdm_mr;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

				  buf = malloc(mr_size);
	assert_non_null(buf);

	assert_int_equal(fi_mr_reg(resource->domain, buf, mr_size,
				   FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);

	/* Test safe casting from fid_mr to efa_mr */
	efa_mr = container_of(mr, struct efa_mr, mr_fid);
	assert_non_null(efa_mr);
	assert_ptr_equal(efa_mr->domain, efa_domain);
	assert_int_equal(efa_mr->iface, FI_HMEM_SYSTEM);

	/* Test safe casting from efa_mr to efa_rdm_mr */
	efa_rdm_mr = (struct efa_rdm_mr *)efa_mr;
	assert_non_null(efa_rdm_mr);
	/* Verify that efa_mr is the first member */
	assert_ptr_equal(&efa_rdm_mr->efa_mr, efa_mr);
	assert_ptr_equal(efa_rdm_mr->efa_mr.domain, efa_domain);
	assert_int_equal(efa_rdm_mr->efa_mr.iface, FI_HMEM_SYSTEM);

	assert_int_equal(fi_close(&mr->fid), 0);
	free(buf);
}

/**
 * @brief Test EFA_MR_ATTR_INIT_SYSTEM macro
 */
void test_efa_mr_attr_init_system_macro(struct efa_resource **state)
{
	struct iovec iov;
	void *buf;
	size_t mr_size = 64;
	uint64_t access = FI_SEND | FI_RECV;
	uint64_t offset = 0;
	uint64_t requested_key = 123;
	void *context = (void *)0xdeadbeef;

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* Test the macro by initializing a new struct */
	struct fi_mr_attr mr_attr = EFA_MR_ATTR_INIT_SYSTEM(&iov, 1, access, offset, requested_key, context);

	assert_ptr_equal(mr_attr.mr_iov, &iov);
	assert_int_equal(mr_attr.iov_count, 1);
	assert_int_equal(mr_attr.access, access);
	assert_int_equal(mr_attr.offset, offset);
	assert_int_equal(mr_attr.requested_key, requested_key);
	assert_ptr_equal(mr_attr.context, context);
	assert_int_equal(mr_attr.iface, FI_HMEM_SYSTEM);

	free(buf);
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
 * @brief Test efa_rdm_mr_cache_regv with cache disabled
 *
 * This test verifies that efa_rdm_mr_cache_regv properly handles the no-cache
 * scenario by falling back to internal registration and creates the expected
 * MR structure without cache entries.
 */
void test_efa_rdm_mr_cache_regv_no_cache(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fi_info *hints;
	struct efa_domain *efa_domain;
	size_t mr_size = 64;
	void *buf;
	struct fid_mr *mr = NULL;
	struct efa_rdm_mr *efa_rdm_mr;
	struct iovec iov;

	/* Create domain with cache disabled (FI_MR_LOCAL) */
	hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode |= FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Verify cache is not available */
	assert_null(efa_domain->cache);
	assert_false(efa_is_cache_available(efa_domain));

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	assert_int_equal(efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
					      FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL),
			 0);
	assert_non_null(mr);

	efa_rdm_mr = container_of(mr, struct efa_rdm_mr, efa_mr.mr_fid);
	/* Verify that shm_mr is NULL even if shm_domain exists */
	assert_null(efa_rdm_mr->shm_mr);

	/* Verify no cache entry since cache is disabled */
	assert_null(efa_rdm_mr->entry);
	/* Verify MR is inserted to domain map */
	assert_true(efa_rdm_mr->inserted_to_mr_map);

	assert_int_equal(fi_close(&mr->fid), 0);
	free(buf);
}

/**
 * @brief Test efa_rdm_mr_cache_regv with cache enabled
 *
 * This test validates that efa_rdm_mr_cache_regv properly uses the MR cache
 * when it's available and creates the expected MR structure.
 */
void test_efa_rdm_mr_cache_regv_with_cache(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_mr *mr = NULL;
	struct efa_rdm_mr *efa_rdm_mr;
	struct iovec iov;
	size_t mr_size = 64;
	void *buf;
	int ret;

	/* Create domain with cache enabled (no FI_MR_LOCAL) */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Verify cache is available */
	assert_non_null(efa_domain->cache);
	assert_true(efa_is_cache_available(efa_domain));

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* Test efa_rdm_mr_cache_regv with cache available */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr);

	/* Verify it's an efa_rdm_mr from cache */
	efa_rdm_mr = container_of(mr, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr->entry); /* Should have cache entry */
	assert_true(efa_rdm_mr->inserted_to_mr_map);
	assert_ptr_equal(efa_rdm_mr->efa_mr.domain, efa_domain);

	assert_int_equal(fi_close(&mr->fid), 0);
	free(buf);
}

/**
 * @brief Test efa_rdm_mr_cache_regv cache hit scenario
 *
 * This test validates that multiple registrations of the same buffer
 * result in cache hits when cache is enabled.
 */
void test_efa_rdm_mr_cache_regv_cache_hit(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_mr *mr1 = NULL, *mr2 = NULL;
	struct efa_rdm_mr *efa_rdm_mr1, *efa_rdm_mr2;
	struct iovec iov;
	size_t mr_size = 64;
	void *buf;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Verify cache is available */
	assert_non_null(efa_domain->cache);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* First registration - should create cache entry */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr1, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr1);

	efa_rdm_mr1 = container_of(mr1, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr1->entry);

	/* Second registration of same buffer - should hit cache */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr2, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr2);

	efa_rdm_mr2 = container_of(mr2, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr2->entry);

	/* Both should reference the same cache entry */
	assert_ptr_equal(efa_rdm_mr1->entry, efa_rdm_mr2->entry);

	assert_int_equal(fi_close(&mr1->fid), 0);
	assert_int_equal(fi_close(&mr2->fid), 0);
	free(buf);
}

/**
 * @brief Test MR cache encapsulation behavior - smaller region within larger cached region
 *
 * New registration is smaller and fully encapsulated within a previous region.
 * The reference count should be incremented and the same cache entry returned.
 */
void test_efa_rdm_mr_cache_encapsulation_smaller(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_mr *mr_large = NULL, *mr_small = NULL;
	struct efa_rdm_mr *efa_rdm_mr_large, *efa_rdm_mr_small;
	struct iovec iov_large, iov_small;
	size_t large_size = 1024, small_size = 256;
	void *buf;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	buf = malloc(large_size);
	assert_non_null(buf);

	/* Register large region first */
	iov_large.iov_base = buf;
	iov_large.iov_len = large_size;

	ret = efa_rdm_mr_cache_regv(resource->domain, &iov_large, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr_large, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr_large);

	efa_rdm_mr_large = container_of(mr_large, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr_large->entry);

	/* Register smaller region within the large region */
	iov_small.iov_base = (char *)buf + 100; /* Offset within large region */
	iov_small.iov_len = small_size;

	ret = efa_rdm_mr_cache_regv(resource->domain, &iov_small, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr_small, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr_small);

	efa_rdm_mr_small = container_of(mr_small, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr_small->entry);

	/* Both should reference the same cache entry (encapsulation) */
	assert_ptr_equal(efa_rdm_mr_large->entry, efa_rdm_mr_small->entry);

	assert_int_equal(fi_close(&mr_large->fid), 0);
	assert_int_equal(fi_close(&mr_small->fid), 0);
	free(buf);
}

/**
 * @brief Test MR cache non-overlapping regions behavior
 *
 * New registration and previous registrations have no full encapsulations.
 * A new registration should be created with reference count of 1.
 */
void test_efa_rdm_mr_cache_non_overlapping(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct fid_mr *mr1 = NULL, *mr2 = NULL;
	struct efa_rdm_mr *efa_rdm_mr1, *efa_rdm_mr2;
	struct iovec iov1, iov2;
	size_t mr_size = 256;
	void *buf1, *buf2;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	/* Allocate two separate, non-overlapping buffers */
	buf1 = malloc(mr_size);
	buf2 = malloc(mr_size);
	assert_non_null(buf1);
	assert_non_null(buf2);

	/* Ensure buffers are non-overlapping by checking addresses */
	assert_true((char *)buf1 + mr_size <= (char *)buf2 || 
		    (char *)buf2 + mr_size <= (char *)buf1);

	/* Register first region */
	iov1.iov_base = buf1;
	iov1.iov_len = mr_size;

	ret = efa_rdm_mr_cache_regv(resource->domain, &iov1, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr1, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr1);

	efa_rdm_mr1 = container_of(mr1, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr1->entry);

	/* Register second, non-overlapping region */
	iov2.iov_base = buf2;
	iov2.iov_len = mr_size;

	ret = efa_rdm_mr_cache_regv(resource->domain, &iov2, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr2, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr2);

	efa_rdm_mr2 = container_of(mr2, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr2->entry);

	/* Should have different cache entries (non-overlapping) */
	assert_ptr_not_equal(efa_rdm_mr1->entry, efa_rdm_mr2->entry);

	assert_int_equal(fi_close(&mr1->fid), 0);
	assert_int_equal(fi_close(&mr2->fid), 0);
	free(buf1);
	free(buf2);
}

/**
 * @brief Test MR cache LRU behavior
 *
 * Tests that MRs with reference count of zero are moved to LRU list
 * and can be reused when the same region is registered again.
 */
void test_efa_rdm_mr_cache_lru_behavior(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_mr *mr1 = NULL, *mr2 = NULL;
	struct efa_rdm_mr *efa_rdm_mr1, *efa_rdm_mr2;
	struct iovec iov;
	size_t mr_size = 256;
	void *buf;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* Initially LRU list should be empty */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* First registration */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr1, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr1);

	efa_rdm_mr1 = container_of(mr1, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr1->entry);
	/* LRU should be empty while entry is in use */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Close first MR - should move to LRU list */
	assert_int_equal(fi_close(&mr1->fid), 0);
	/* LRU should now contain the entry */
	assert_false(dlist_empty(&efa_domain->cache->lru_list));

	/* Register same region again - should reuse from LRU */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr2, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr2);

	efa_rdm_mr2 = container_of(mr2, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr2->entry);

	/* Should reuse the same cache entry from LRU */
	assert_ptr_equal(efa_rdm_mr1->entry, efa_rdm_mr2->entry);
	/* LRU should be empty again since entry is back in use */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	assert_int_equal(fi_close(&mr2->fid), 0);
	/* LRU should contain the entry again */
	assert_false(dlist_empty(&efa_domain->cache->lru_list));

	free(buf);
}

/**
 * @brief Test MR cache flush behavior
 *
 * Tests that cache flush operations properly clean up entries
 * and that subsequent registrations create new entries.
 */
void test_efa_rdm_mr_cache_flush_behavior(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_mr *mr1 = NULL, *mr2 = NULL;
	struct efa_rdm_mr *efa_rdm_mr1, *efa_rdm_mr2;
	struct iovec iov;
	size_t mr_size = 256;
	void *buf;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* Initially LRU list should be empty */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* First registration */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr1, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr1);

	efa_rdm_mr1 = container_of(mr1, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr1->entry);
	/* LRU should be empty while entry is in use */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Close MR - moves to LRU */
	assert_int_equal(fi_close(&mr1->fid), 0);
	/* LRU should now contain the entry */
	assert_false(dlist_empty(&efa_domain->cache->lru_list));

	/* Force cache flush - should clean up LRU entries */
	ofi_mr_cache_flush(efa_domain->cache, true /* flush_lru */);
	/* LRU should be empty after flush */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Register same region again - should create new entry after flush */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr2, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr2);

	efa_rdm_mr2 = container_of(mr2, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr2->entry);
	/* Should have a valid new cache entry */
	assert_int_equal(efa_rdm_mr2->entry->use_cnt, 1);

	assert_int_equal(fi_close(&mr2->fid), 0);
	free(buf);
}

/**
 * @brief Test MR cache reference counting
 *
 * Tests that multiple references to the same cached region properly
 * increment/decrement reference counts and manage LRU list placement.
 */
void test_efa_rdm_mr_cache_reference_counting(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct fid_mr *mr1 = NULL, *mr2 = NULL, *mr3 = NULL;
	struct efa_rdm_mr *efa_rdm_mr1, *efa_rdm_mr2, *efa_rdm_mr3;
	struct iovec iov;
	size_t mr_size = 256;
	void *buf;
	int ret;

	/* Create domain with cache enabled */
	struct fi_info *hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
	hints->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM,
						    FI_VERSION(2, 0), hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	buf = malloc(mr_size);
	assert_non_null(buf);

	iov.iov_base = buf;
	iov.iov_len = mr_size;

	/* Initially LRU list should be empty */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* First registration - creates cache entry with use_cnt = 1 */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr1, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr1);

	efa_rdm_mr1 = container_of(mr1, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_non_null(efa_rdm_mr1->entry);
	assert_int_equal(efa_rdm_mr1->entry->use_cnt, 1);
	/* LRU should still be empty since use_cnt > 0 */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Second registration - should increment reference count to 2 */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr2, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr2);

	efa_rdm_mr2 = container_of(mr2, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_ptr_equal(efa_rdm_mr1->entry, efa_rdm_mr2->entry);
	assert_int_equal(efa_rdm_mr1->entry->use_cnt, 2);
	/* LRU should still be empty since use_cnt > 0 */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Third registration - should increment reference count to 3 */
	ret = efa_rdm_mr_cache_regv(resource->domain, &iov, 1,
				    FI_SEND | FI_RECV, 0, 0, 0, &mr3, NULL);
	assert_int_equal(ret, 0);
	assert_non_null(mr3);

	efa_rdm_mr3 = container_of(mr3, struct efa_rdm_mr, efa_mr.mr_fid);
	assert_ptr_equal(efa_rdm_mr1->entry, efa_rdm_mr3->entry);
	assert_int_equal(efa_rdm_mr1->entry->use_cnt, 3);
	/* LRU should still be empty since use_cnt > 0 */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Close first MR - should decrement reference count to 2, not move to LRU */
	assert_int_equal(fi_close(&mr1->fid), 0);
	assert_int_equal(efa_rdm_mr2->entry->use_cnt, 2);
	/* LRU should still be empty since use_cnt > 0 */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Close second MR - should decrement reference count to 1, not move to LRU */
	assert_int_equal(fi_close(&mr2->fid), 0);
	assert_int_equal(efa_rdm_mr3->entry->use_cnt, 1);
	/* LRU should still be empty since use_cnt > 0 */
	assert_true(dlist_empty(&efa_domain->cache->lru_list));

	/* Close third MR - should decrement to zero and move to LRU */
	assert_int_equal(fi_close(&mr3->fid), 0);
	/* Now LRU should contain the entry since use_cnt reached 0 */
	assert_false(dlist_empty(&efa_domain->cache->lru_list));

	free(buf);
}
