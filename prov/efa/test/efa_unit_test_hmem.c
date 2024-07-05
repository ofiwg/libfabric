/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

void test_efa_hmem_p2p_support(struct efa_resource **state, bool file_exists,
			       char *prov_name, enum fi_hmem_iface iface,
			       bool expected_p2p_support)
{
	int fd = -1, ret;
	bool iface_initialized;
	uint32_t device_caps;
	ssize_t written_len;
	char p2p_file[] = "XXXXXXXXXX";
	char ibdev_path[IBV_SYSFS_PATH_MAX] = {0};
	char *p2p_file_suffix = NULL;
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	if (file_exists) {
		fd = mkstemp(p2p_file);
		if (fd < 0) {
			fail();
		}

		written_len = write(fd, prov_name, strlen(prov_name));
		if (written_len != strlen(prov_name)) {
			close(fd);
			fail();
		}
	}

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM);
	assert_non_null(resource->hints);

	ret = fi_getinfo(FI_VERSION(1, 20), NULL, NULL, 0ULL, resource->hints,
			 &resource->info);
	assert_int_equal(ret, FI_SUCCESS);

	ret = fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL);
	assert_int_equal(ret, FI_SUCCESS);

	/* Override global attributes */
	strcpy(ibdev_path, g_device_list[0].ibv_ctx->device->ibdev_path);
	strcpy(g_device_list[0].ibv_ctx->device->ibdev_path, p2p_file);
	iface_initialized = hmem_ops[iface].initialized;
	hmem_ops[iface].initialized = true;
	device_caps = g_device_list[0].device_caps;
	p2p_file_suffix = efa_env.p2p_file_suffix;
	efa_env.p2p_file_suffix = "";

	ret = fi_domain(resource->fabric, resource->info, &resource->domain,
			NULL);

	/* Reset global attributes */
	strcpy(g_device_list[0].ibv_ctx->device->ibdev_path, ibdev_path);
	hmem_ops[iface].initialized = iface_initialized;
	g_device_list[0].device_caps = device_caps;
	efa_env.p2p_file_suffix = p2p_file_suffix;

	/* Remove the temporary file */
	if (file_exists) {
		unlink(p2p_file);
		close(fd);
	}

	assert_int_equal(ret, FI_SUCCESS);
	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid.fid);

	assert_true(efa_domain->hmem_info[iface].initialized);
	assert_int_equal(efa_domain->hmem_info[iface].p2p_supported_by_device,
			 expected_p2p_support);
}

void test_efa_dmabuf_support(struct efa_resource **state,
                             enum fi_hmem_iface iface,
                             bool require_dmabuf,
                             bool dmabuf_fd_supported,
                             bool dmabuf_mr_supported,
			     bool expect_ibv_reg_mr,
			     bool expect_ibv_reg_dmabuf_mr)
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

	efa_domain->hmem_info[iface].initialized = true;
	efa_domain->hmem_info[iface].p2p_supported_by_device = true;
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
		g_efa_unit_test_mocks.ibv_reg_mr_iova2 =
			efa_mock_ibv_reg_mr_iova2_return_mock_for_dummy_addr;
		will_return_maybe(efa_mock_ibv_reg_mr_iova2_return_mock_for_dummy_addr, &ibv_mr);
		g_efa_unit_test_mocks.ibv_reg_mr_fn =
			efa_mock_ibv_reg_mr_return_mock_for_dummy_addr;
		will_return_maybe(efa_mock_ibv_reg_mr_return_mock_for_dummy_addr, &ibv_mr);
	}

	assert_int_equal(0, g_efa_unit_test_mocks.ibv_reg_mr_calls);

	ret = fi_mr_regattr(resource->domain, &mr_attr, flags, &mr);

	if (expect_ibv_reg_mr) {
		assert_int_equal(1, g_efa_unit_test_mocks.ibv_reg_mr_calls);
	}

	/* Reset global attributes */
	hmem_ops[iface].get_dmabuf_fd = get_dmabuf_fd;
	hmem_ops[iface].initialized = iface_initialized;

	if (!expect_ibv_reg_mr && (!dmabuf_fd_supported || !dmabuf_mr_supported) ) {
		assert_int_not_equal(ret, FI_SUCCESS);
	} else {
		/* Either ibv_reg_dmabuf_mr or ibv_reg_mr should work */
		assert_int_equal(ret, FI_SUCCESS);
	}

	if (mr) {
		fi_close((fid_t) &mr->fid);
	}
}

void test_efa_system_p2p_support_true_system_p2p_file_does_not_exist(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, false, NULL, FI_HMEM_SYSTEM, true);
}

void test_efa_system_p2p_support_true_system_p2p_file_empty(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "", FI_HMEM_SYSTEM, true);
}

void test_efa_cuda_p2p_support_true_nvidia_prov(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "NVIDIA", FI_HMEM_CUDA, true);
}

void test_efa_cuda_p2p_support_true_nvidia_peermem_prov(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "NVIDIA peermem", FI_HMEM_CUDA, true);
}

void test_efa_cuda_p2p_support_false_p2p_file_does_not_exist(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, false, NULL, FI_HMEM_CUDA, false);
}

void test_efa_cuda_p2p_support_false_p2p_file_empty(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "", FI_HMEM_CUDA, false);
}

void test_efa_neuron_p2p_support_true_neuron_prov(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "NEURON", FI_HMEM_NEURON, true);
}

void test_efa_neuron_p2p_support_false_p2p_file_does_not_exist(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, false, NULL, FI_HMEM_NEURON, false);
}

void test_efa_neuron_p2p_support_false_p2p_file_empty(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "", FI_HMEM_NEURON, false);
}

void test_efa_synapseai_p2p_support_true_p2p_file_does_not_exist(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, false, NULL, FI_HMEM_SYNAPSEAI, true);
}

void test_efa_synapseai_p2p_support_true_p2p_file_empty(struct efa_resource **state)
{
	test_efa_hmem_p2p_support(state, true, "", FI_HMEM_SYNAPSEAI, true);
}

void test_efa_system_always_ibv_reg_mr(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_SYSTEM, false, false, false, true, false);
}

void test_efa_cuda_dmabuf_support_always_ibv_reg_mr(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, false, false, false, true, false);
}

void test_efa_cuda_dmabuf_support_require_dmabuf_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_CUDA, true, false, false, false, true);
}

void test_efa_neuron_dmabuf_support_dmabuf_success(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, true, true, false, true);
}

void test_efa_neuron_dmabuf_support_get_fd_fail_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, false, false, true, true);
}

void test_efa_neuron_dmabuf_support_mr_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, false, true, false, false, true);
}

void test_efa_neuron_dmabuf_support_require_dmabuf_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_NEURON, true, false, false, false, true);
}

void test_efa_synapseai_dmabuf_support_fd_fail_no_fallback(struct efa_resource **state)
{
	test_efa_dmabuf_support(state, FI_HMEM_SYNAPSEAI, false, false, false, false, true);
}
