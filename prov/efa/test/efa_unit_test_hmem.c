/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

void test_efa_dmabuf_support(struct efa_resource **state,
                             enum fi_hmem_iface iface,
                             bool require_dmabuf,
                             bool dmabuf_fd_supported,
                             bool dmabuf_mr_supported,
                             bool ibv_reg_mr_supported,
                             bool expect_ibv_reg_mr,
                             bool expect_ibv_reg_dmabuf_mr,
                             bool expect_p2p_support,
                             bool should_succeed)
{
	bool iface_initialized;
	const size_t bufsize = 4096;
	const uint64_t buf = g_efa_unit_test_mocks.dummy_address;
	int ret;
	struct efa_domain *efa_domain;
	struct efa_resource *resource = *state;
	struct fi_mr_attr mr_attr = {0};
	struct fi_mr_dmabuf mr_dmabuf = {0};
	struct fid_mr *mr = NULL;
	struct ibv_mr ibv_mr = {0};
	struct iovec iov = {0};
	uint64_t flags = 0;
	get_dmabuf_fd_fn_t get_dmabuf_fd = NULL;

	mr_attr.iface = iface;
	mr_attr.access = FI_SEND | FI_RECV;

	if (require_dmabuf) {
		mr_dmabuf.base_addr = (void *) buf;
		mr_dmabuf.len = bufsize;
		mr_attr.dmabuf = &mr_dmabuf;
		flags |= FI_MR_DMABUF;
	} else {
		iov.iov_base = (void *) buf;
		iov.iov_len = bufsize;
		mr_attr.iov_count = 1;
		mr_attr.mr_iov = &iov;
	}

	/* Override global attributes */
	get_dmabuf_fd = hmem_ops[iface].get_dmabuf_fd;
	hmem_ops[iface].get_dmabuf_fd = efa_mock_get_dmabuf_fd_set_errno_return_mock;
	iface_initialized = hmem_ops[iface].initialized;
	hmem_ops[iface].initialized = true;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM);
	efa_unit_test_resource_construct_with_hints(resource, FI_EP_RDM, FI_VERSION(1, 20),
	                                            resource->hints, true, true);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid.fid);

	assert_true(efa_domain->hmem_info[iface].initialized);
	assert_true(efa_domain->hmem_info[iface].p2p_supported_by_device);
	/* Force domain to support FI_HMEM so we can test on any platform */
	efa_domain->util_domain.info_domain_caps |= FI_HMEM;

	/* Required during cleanup */
	g_efa_unit_test_mocks.ibv_dereg_mr = efa_mock_ibv_dereg_mr_return_mock_for_dummy_mr;
	will_return_maybe(efa_mock_ibv_dereg_mr_return_mock_for_dummy_mr, 0);

	if (expect_ibv_reg_dmabuf_mr && !require_dmabuf) {
		will_return(efa_mock_get_dmabuf_fd_set_errno_return_mock,
			    dmabuf_fd_supported ? FI_SUCCESS : -FI_EOPNOTSUPP);
	}

#if HAVE_EFA_DMABUF_MR
	g_efa_unit_test_mocks.ibv_reg_dmabuf_mr =
		efa_mock_ibv_reg_dmabuf_mr_set_eopnotsupp_and_return_mock;
	if ((expect_ibv_reg_dmabuf_mr && dmabuf_fd_supported) || require_dmabuf) {
		will_return(
			efa_mock_ibv_reg_dmabuf_mr_set_eopnotsupp_and_return_mock,
			dmabuf_mr_supported ? &ibv_mr : NULL);
	}
#else
	assert_false(dmabuf_mr_supported);
#endif /* HAVE_EFA_DMABUF_MR */

	if (expect_ibv_reg_mr) {
		/* Fallback to ibv_reg_mr */
		if (ibv_reg_mr_supported) {
			g_efa_unit_test_mocks.ibv_reg_mr_iova2 =
					efa_mock_ibv_reg_mr_iova2_success_return_mock_for_dummy_addr;
			will_return_maybe(
				efa_mock_ibv_reg_mr_iova2_success_return_mock_for_dummy_addr,
				&ibv_mr);
			g_efa_unit_test_mocks.ibv_reg_mr_fn =
				efa_mock_ibv_reg_mr_success_return_mock_for_dummy_addr;
			will_return_maybe(
				efa_mock_ibv_reg_mr_return_mock_for_dummy_addr,
				&ibv_mr);
		} else {
			g_efa_unit_test_mocks.ibv_reg_mr_iova2 =
					efa_mock_ibv_reg_mr_iova2_set_errno_return_mock_for_dummy_addr;
			will_return_maybe(
				efa_mock_ibv_reg_mr_iova2_set_errno_return_mock_for_dummy_addr,
				NULL);
			g_efa_unit_test_mocks.ibv_reg_mr_fn =
				efa_mock_ibv_reg_mr_set_errno_return_mock_for_dummy_addr;
			will_return_maybe(
				efa_mock_ibv_reg_mr_set_errno_return_mock_for_dummy_addr,
				NULL);
		}
	}

	assert_int_equal(0, g_efa_unit_test_mocks.ibv_reg_mr_calls);

	ret = fi_mr_regattr(resource->domain, &mr_attr, flags, &mr);

	if (expect_ibv_reg_mr) {
		assert_int_equal(1, g_efa_unit_test_mocks.ibv_reg_mr_calls);
	}

	assert_int_equal(expect_p2p_support, efa_domain->hmem_info[iface].p2p_supported_by_device);

	/* Reset global attributes */
	hmem_ops[iface].get_dmabuf_fd = get_dmabuf_fd;
	hmem_ops[iface].initialized = iface_initialized;

	if (should_succeed) {
		assert_int_equal(ret, FI_SUCCESS);
	} else {
		assert_int_not_equal(ret, FI_SUCCESS);
	}

	if (mr) {
		fi_close((fid_t) &mr->fid);
	}
}

/**
 * System memory does not support dmabuf so it should always use ibv_reg_mr
 */
void test_efa_system_always_ibv_reg_mr(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_SYSTEM, false, false, false, true, true, false, true, true);
}

/**
 * Unless FI_MR_DMABUF is required CUDA should always use ibv_reg_mr
 */
void test_efa_cuda_dmabuf_support_always_ibv_reg_mr(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, false, false, false, true, true, false, true, true);
}

/**
 * P2P is not required for FI_VERSION >= 1.18, and CUDA ibv_reg_mr
 * is allowed to fail and fallback to keygen.
 * If MR fails for reasons other than ENOMEM, P2P support will be disabled.
 */
void test_efa_cuda_dmabuf_support_ibv_reg_mr_fail_disable_p2p_fallback_keygen(struct efa_resource **state)
{
	g_efa_unit_test_mocks.ibv_reg_mr_errno = EINVAL;
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, false, false, false, false, true, false, false, true);
}

/**
 * P2P is not required for FI_VERSION >= 1.18, and CUDA ibv_reg_mr
 * is allowed to fail and fallback to keygen.
 * If MR fails due to ENOMEM, P2P support will NOT be disabled.
 */
void test_efa_cuda_dmabuf_support_ibv_reg_mr_fail_retain_p2p_fallback_keygen(struct efa_resource **state)
{
	g_efa_unit_test_mocks.ibv_reg_mr_errno = ENOMEM;
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, false, false, false, false, true, false, true, true);
}

/**
 * If dmabuf is NOT supported, but the application still requires FI_MR_DMABUF,
 * we should respect application's request and use ibv_reg_dmabuf_mr.
 * This is a theoretical corner case to allow application override.
 */
void test_efa_cuda_dmabuf_support_require_dmabuf_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, true, false, false, true, false, true, false, false);
}

/**
 * Verify dmabuf is supported lazily for the HMEM iface
 */
void test_efa_neuron_dmabuf_support_dmabuf_success(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, true, true, true, false, true, true, true);
}

/**
 * If dmabuf fd cannot be retrieved, verify dmabuf is not supported lazily
 * for the HMEM iface
 */
void test_efa_neuron_dmabuf_support_get_fd_fail_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, false, false, true, true, true, true, true);
}

/**
 * If ibv_reg_dmabuf_mr fails, verify dmabuf is not supported lazily
 * for the HMEM iface
 */
void test_efa_neuron_dmabuf_support_mr_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, true, false, true, false, true, true, false);
}

/**
 * If dmabuf is NOT supported, but the application still requires FI_MR_DMABUF,
 * we should respect application's request and use ibv_reg_dmabuf_mr.
 * This is a theoretical corner case to allow application override.
 */
void test_efa_neuron_dmabuf_support_require_dmabuf_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, true, false, false, true, false, true, true, false);
}

/**
 * If dmabuf is supported we should always use ibv_reg_dmabuf_mr,
 * even if it fails unexpctedly
 */
void test_efa_synapseai_dmabuf_support_fd_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_SYNAPSEAI, false, false, false, true, false, true, true, false);
}
