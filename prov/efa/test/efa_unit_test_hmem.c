/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_unit_tests.h"

#if HAVE_NEURON
/**
 * @brief Verify that Neuron p2p and dmabuf support are assumed without
 * explicit checking (to avoid early buffer allocation) if HAVE_EFA_DMABUF_MR is
 * enabled.
 *
 * @param[in]	state		struct efa_resource that is managed by the
 * framework
 */
void test_efa_hmem_info_p2p_dmabuf_assumed_neuron(void **state)
{
	int ret;

	hmem_ops[FI_HMEM_NEURON].initialized = true;

	ret = efa_hmem_info_initialize();

	assert_int_equal(ret, 0);
	assert_true(g_efa_hmem_info[FI_HMEM_NEURON].initialized);
	assert_true(g_efa_hmem_info[FI_HMEM_NEURON].p2p_supported_by_device);
#if HAVE_EFA_DMABUF_MR
	assert_int_equal(
		g_efa_hmem_info[FI_HMEM_NEURON].dmabuf_supported_by_device,
		EFA_DMABUF_ASSUMED);
#else /* !HAVE_EFA_DMABUF_MR */
	assert_int_equal(
		g_efa_hmem_info[FI_HMEM_NEURON].dmabuf_supported_by_device,
		EFA_DMABUF_NOT_SUPPORTED);
#endif
}
#else
void test_efa_hmem_info_p2p_dmabuf_assumed_neuron(void **state)
{
	skip();
}
#endif /* HAVE_NEURON */

#if HAVE_NEURON
/**
 * @brief Verify that Neuron is not initialized when p2p is disabled.
 *
 * @param[in]	state		struct efa_resource that is managed by the
 * framework
 */
void test_efa_hmem_info_p2p_disabled_neuron(void **state)
{
	int ret;

	ofi_hmem_disable_p2p = 1;

	hmem_ops[FI_HMEM_NEURON].initialized = true;

	ret = efa_hmem_info_initialize();

	assert_int_equal(ret, 0);
	assert_false(g_efa_hmem_info[FI_HMEM_NEURON].initialized);
}
#else
void test_efa_hmem_info_p2p_disabled_neuron(void **state)
{
	skip();
}
#endif /* HAVE_NEURON */

#if HAVE_SYNAPSEAI
/**
 * @brief Verify that SynapseAI is not initialized when p2p is disabled.
 *
 * @param[in]	state		struct efa_resource that is managed by the
 * framework
 */
void test_efa_hmem_info_p2p_disabled_synapse(void **state)
{
	int ret;

	ofi_hmem_disable_p2p = 1;

	hmem_ops[FI_HMEM_SYNAPSEAI].initialized = true;

	ret = efa_hmem_info_initialize();

	assert_int_equal(ret, 0);
	assert_false(g_efa_hmem_info[FI_HMEM_SYNAPSEAI].initialized);
}
#else
void test_efa_hmem_info_p2p_disabled_synapse(void **state)
{
	skip();
}
#endif /* HAVE_SYNAPSEAI */

#if HAVE_CUDA
/**
 * @brief Verify when p2p is disabled, we don't check p2p support with
 * ofi_cudaMalloc. Just leave p2p_supported_by_device to false for cuda.
 *
 * @param[in]	state		struct efa_resource that is managed by the
 * framework
 */
void test_efa_hmem_info_disable_p2p_cuda(void **state)
{
	int ret;

	ofi_hmem_disable_p2p = 1;

	hmem_ops[FI_HMEM_CUDA].initialized = true;
	/* ofi_cudaMalloc should not be called when p2p is disabled.
	 * efa_mock_ofi_cudaMalloc_return_mock will fail the test when it is
	 * called. */
	g_efa_unit_test_mocks.ofi_cudaMalloc =
		efa_mock_ofi_cudaMalloc_return_mock;

	ret = efa_hmem_info_initialize();

	assert_int_equal(ret, 0);
	assert_true(g_efa_hmem_info[FI_HMEM_CUDA].initialized);
	assert_false(g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device);
}
#else
void test_efa_hmem_info_disable_p2p_cuda(void **state)
{
	skip();
}
#endif /* HAVE_CUDA */

/**
 * @brief Verify instance types are matched exactly against the blocklist
 */
void test_efa_instance_type_gdr_unsupported(void **state)
{
	/* Listed */
	assert_true(efa_instance_type_gdr_unsupported("g5.48xlarge"));
	assert_true(efa_instance_type_gdr_unsupported("p5.4xlarge"));
	assert_true(efa_instance_type_gdr_unsupported("g7e.8xlarge"));

	/* Unlisted sizes of families with listed sizes */
	assert_false(efa_instance_type_gdr_unsupported("p5.48xlarge"));
	assert_false(efa_instance_type_gdr_unsupported("g7e.12xlarge"));
	assert_false(efa_instance_type_gdr_unsupported("p4d.24xlarge"));

	/* Similar names */
	assert_false(efa_instance_type_gdr_unsupported("p4de.24xlarge"));
	assert_false(efa_instance_type_gdr_unsupported("g5g.16xlarge"));
	assert_false(efa_instance_type_gdr_unsupported("g5"));
	assert_false(efa_instance_type_gdr_unsupported("g5.48xlarge2"));

	/* Unknown */
	assert_false(efa_instance_type_gdr_unsupported(""));
	assert_false(efa_instance_type_gdr_unsupported("HVM domU"));
}

#if HAVE_CUDA
/**
 * @brief Initialize CUDA hmem info with a given instance type file content
 *
 * ofi_cudaMalloc, ofi_cuCtxCreate_v2 and ofi_cuMemAlloc are mocked to fail the
 * test if called, since determining p2p support must not allocate CUDA memory
 * or create a CUDA context.
 *
 * @param[in]	instance_type	Content of the instance type file, or NULL
 *				to use a non-existent file
 */
static void test_efa_hmem_info_cuda_instance_type_impl(const char *instance_type)
{
	char instance_type_file[] = "/tmp/efa_instance_type_XXXXXX";
	int fd = -1;
	int ret;

	if (instance_type) {
		fd = mkstemp(instance_type_file);
		assert_true(fd >= 0);
		assert_int_equal(write(fd, instance_type, strlen(instance_type)),
				 strlen(instance_type));
	}
	efa_env.instance_type_file = instance_type ? instance_type_file :
						     "/nonexistent/product_name";

	hmem_ops[FI_HMEM_CUDA].initialized = true;
	/*
	 * The instance-type path must not allocate device memory or create a
	 * CUDA context. These mocks have no queued return, so any call aborts
	 * the test.
	 */
	g_efa_unit_test_mocks.ofi_cudaMalloc =
		efa_mock_ofi_cudaMalloc_return_mock;
	g_efa_unit_test_mocks.ofi_cuCtxCreate_v2 =
		efa_mock_ofi_cuCtxCreate_v2_return_mock;
	g_efa_unit_test_mocks.ofi_cuMemAlloc =
		efa_mock_ofi_cuMemAlloc_return_mock;

	ret = efa_hmem_info_initialize();

	if (fd >= 0) {
		close(fd);
		unlink(instance_type_file);
	}

	assert_int_equal(ret, 0);
	assert_true(g_efa_hmem_info[FI_HMEM_CUDA].initialized);
}

/**
 * @brief Verify CUDA p2p is not supported on a blocklisted instance type
 */
void test_efa_hmem_info_cuda_p2p_instance_type_unsupported(void **state)
{
	test_efa_hmem_info_cuda_instance_type_impl("g5.48xlarge\n");
	assert_false(g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device);
}

/**
 * @brief Verify CUDA p2p is supported on an instance type not in the blocklist
 */
void test_efa_hmem_info_cuda_p2p_instance_type_supported(void **state)
{
	test_efa_hmem_info_cuda_instance_type_impl("p5.48xlarge\n");
	assert_true(g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device);
}

/**
 * @brief Verify CUDA p2p is assumed supported if the instance type is unknown
 */
void test_efa_hmem_info_cuda_p2p_instance_type_missing(void **state)
{
	test_efa_hmem_info_cuda_instance_type_impl(NULL);
	assert_true(g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device);
}
#else
void test_efa_hmem_info_cuda_p2p_instance_type_unsupported(void **state)
{
	skip();
}

void test_efa_hmem_info_cuda_p2p_instance_type_supported(void **state)
{
	skip();
}

void test_efa_hmem_info_cuda_p2p_instance_type_missing(void **state)
{
	skip();
}
#endif /* HAVE_CUDA */

#if HAVE_CUDA_CTX_SYNC_MEMOPS
struct efa_hmem_sync_worker_args {
	void *ptr;
	uint64_t device;
	CUcontext initial_ctx;
	CUresult get_current_result;
	int sync_result;
};

static int efa_unit_test_cuda_set_sync_memops_not_supported(void *ptr)
{
	(void) ptr;
	return -FI_EOPNOTSUPP;
}

static void *efa_unit_test_sync_memops_worker(void *arg)
{
	struct efa_hmem_sync_worker_args *args = arg;

	args->get_current_result = ofi_cuCtxGetCurrent(&args->initial_ctx);
	if (args->get_current_result != CUDA_SUCCESS || args->initial_ctx) {
		args->sync_result = -FI_EINVAL;
		return NULL;
	}

	args->sync_result = efa_hmem_set_sync_memops(args->ptr, args->device);
	return NULL;
}

void test_efa_hmem_set_sync_memops_active_primary_context_worker_thread(
	void **state)
{
	struct efa_hmem_sync_worker_args args = {0};
	const char *failed_api = NULL;
	CUdevice cu_device;
	CUresult cu_result = CUDA_SUCCESS;
	CUresult restore_result = CUDA_SUCCESS;
	unsigned int original_flags = 0;
	unsigned int cleared_flags = 0;
	unsigned int final_flags = 0;
	uint64_t device;
	pthread_t worker;
	cudaError_t cuda_result;
	bool restore_flags = false;
	int active = 0;
	int create_result = 0;
	int join_result = 0;

	if (!hmem_ops[FI_HMEM_CUDA].initialized) {
		skip();
		return;
	}

	cuda_result = ofi_cudaMalloc(&args.ptr, 1);
	if (cuda_result != cudaSuccess) {
		skip();
		return;
	}

	if (!cuda_is_addr_valid(args.ptr, &device, NULL)) {
		failed_api = "cuda_is_addr_valid";
		goto out;
	}

	cu_device = (CUdevice) device;

	cu_result = ofi_cuDevicePrimaryCtxGetState(
		cu_device, &original_flags, &active);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuDevicePrimaryCtxGetState";
		goto out;
	}

	if (!active) {
		failed_api = "primary context is not active";
		goto out;
	}

	cleared_flags = original_flags & ~CU_CTX_SYNC_MEMOPS;
	cu_result = ofi_cuDevicePrimaryCtxSetFlags(cu_device, cleared_flags);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuDevicePrimaryCtxSetFlags(clear)";
		goto out;
	}
	restore_flags = true;

	cu_result = ofi_cuDevicePrimaryCtxGetState(cu_device, &final_flags, &active);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuDevicePrimaryCtxGetState(before worker)";
		goto out;
	}

	if (final_flags & CU_CTX_SYNC_MEMOPS) {
		failed_api = "CU_CTX_SYNC_MEMOPS was not cleared";
		goto out;
	}

	g_efa_unit_test_mocks.cuda_set_sync_memops =
		efa_unit_test_cuda_set_sync_memops_not_supported;
	args.device = device;
	args.sync_result = -FI_EINVAL;

	create_result = pthread_create(
		&worker, NULL, efa_unit_test_sync_memops_worker, &args);
	if (create_result)
		goto out;

	join_result = pthread_join(worker, NULL);
	if (join_result)
		goto out;

	cu_result = ofi_cuDevicePrimaryCtxGetState(cu_device, &final_flags, &active);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuDevicePrimaryCtxGetState(after worker)";
		goto out;
	}

out:
	if (restore_flags)
		restore_result = ofi_cuDevicePrimaryCtxSetFlags(
			cu_device, original_flags);
	cuda_result = ofi_cudaFree(args.ptr);

	assert_null(failed_api);
	assert_int_equal(cu_result, CUDA_SUCCESS);
	assert_int_equal(create_result, 0);
	assert_int_equal(join_result, 0);
	assert_int_equal(args.get_current_result, CUDA_SUCCESS);
	assert_null(args.initial_ctx);
	assert_int_equal(args.sync_result, FI_SUCCESS);
	assert_true(final_flags & CU_CTX_SYNC_MEMOPS);
	assert_int_equal(restore_result, CUDA_SUCCESS);
	assert_int_equal(cuda_result, cudaSuccess);
}
#else
void test_efa_hmem_set_sync_memops_active_primary_context_worker_thread(
	void **state)
{
	skip();
}
#endif
