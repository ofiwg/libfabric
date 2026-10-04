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

#if HAVE_CUDA
/**
 * @brief Verify that when the p2p probe has to create its own CUDA context
 * (because none is already current), it destroys that context even when the
 * buffer allocation fails. Guards against leaking a context we explicitly
 * created.
 *
 * Mocking ofi_cuCtxGetCurrent to report no current context forces the
 * create-and-own path (own_cuda_ctx = true). Mocking the alloc failure is also
 * nice, since it lets us not have to deal with ibv registration.
 *
 * @param[in]	state		struct efa_resource that is managed by the
 * framework
 */
void test_efa_hmem_info_check_p2p_cuda_ctx_create_destroy_on_memalloc_fail(
	void **state)
{
	int ret;

	hmem_ops[FI_HMEM_CUDA].initialized = true;

	/* Mock ofi_cuCtxGetCurrent to report NO current context, forcing the
	 * probe to create (and own) its own temporary context. */
	g_efa_unit_test_mocks.ofi_cuCtxGetCurrent =
		efa_mock_ofi_cuCtxGetCurrent_return_mock;
	will_return(efa_mock_ofi_cuCtxGetCurrent_return_mock, NULL); /* pctx */
	will_return(efa_mock_ofi_cuCtxGetCurrent_return_mock, CUDA_SUCCESS);

	/* Mock ofi_cuDeviceGet to succeed */
	g_efa_unit_test_mocks.ofi_cuDeviceGet =
		efa_mock_ofi_cuDeviceGet_return_mock;
	will_return(efa_mock_ofi_cuDeviceGet_return_mock, 0); /* device */
	will_return(efa_mock_ofi_cuDeviceGet_return_mock, CUDA_SUCCESS);

	/* Mock ofi_cuCtxCreate_v2 to succeed */
	g_efa_unit_test_mocks.ofi_cuCtxCreate_v2 =
		efa_mock_ofi_cuCtxCreate_v2_return_mock;
	will_return(efa_mock_ofi_cuCtxCreate_v2_return_mock, NULL); /* pctx */
	will_return(efa_mock_ofi_cuCtxCreate_v2_return_mock, CUDA_SUCCESS);
	expect_function_call(efa_mock_ofi_cuCtxCreate_v2_return_mock);

	/* Mock ofi_cudaMalloc to FAIL */
	g_efa_unit_test_mocks.ofi_cudaMalloc =
		efa_mock_ofi_cudaMalloc_return_mock;
	will_return(efa_mock_ofi_cudaMalloc_return_mock,
		    cudaErrorMemoryAllocation);

	/* Expect ofi_cuCtxDestroy to be called (cleanup of the context we
	 * created, on the alloc-failure path) */
	g_efa_unit_test_mocks.ofi_cuCtxDestroy =
		efa_mock_ofi_cuCtxDestroy_return_mock;
	will_return(efa_mock_ofi_cuCtxDestroy_return_mock, CUDA_SUCCESS);
	expect_function_call(efa_mock_ofi_cuCtxDestroy_return_mock);

	ret = efa_hmem_info_initialize();

	assert_int_equal(ret, 0);
	assert_false(g_efa_hmem_info[FI_HMEM_CUDA].initialized);
}
#else
void test_efa_hmem_info_check_p2p_cuda_ctx_create_destroy_on_memalloc_fail(void **state)
{
	skip();
}
#endif /* HAVE_CUDA */

#if OFI_HAVE_CUDA_CTX_SYNC_MEMOPS
static void efa_unit_test_expect_pointer_sync_memops(void *ptr,
						      int result)
{
	g_efa_unit_test_mocks.cuda_set_sync_memops =
		efa_mock_cuda_set_sync_memops_return_mock;
	expect_value(efa_mock_cuda_set_sync_memops_return_mock, ptr, ptr);
	will_return(efa_mock_cuda_set_sync_memops_return_mock, result);
}

static void efa_unit_test_expect_current_ctx(CUcontext current_ctx,
					     CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuCtxGetCurrent =
		efa_mock_ofi_cuCtxGetCurrent_return_mock;
	will_return(efa_mock_ofi_cuCtxGetCurrent_return_mock, current_ctx);
	will_return(efa_mock_ofi_cuCtxGetCurrent_return_mock, result);
}

static void efa_unit_test_expect_current_ctx_device(CUdevice device,
						    CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuCtxGetDevice =
		efa_mock_ofi_cuCtxGetDevice_return_mock;
	will_return(efa_mock_ofi_cuCtxGetDevice_return_mock, device);
	will_return(efa_mock_ofi_cuCtxGetDevice_return_mock, result);
}

static void efa_unit_test_expect_current_ctx_flags(unsigned int flags,
						   CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuCtxGetFlags =
		efa_mock_ofi_cuCtxGetFlags_return_mock;
	will_return(efa_mock_ofi_cuCtxGetFlags_return_mock, flags);
	will_return(efa_mock_ofi_cuCtxGetFlags_return_mock, result);
}

static void efa_unit_test_expect_current_ctx_set_flags(unsigned int flags,
						       CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuCtxSetFlags =
		efa_mock_ofi_cuCtxSetFlags_return_mock;
	expect_value(efa_mock_ofi_cuCtxSetFlags_return_mock, flags, flags);
	will_return(efa_mock_ofi_cuCtxSetFlags_return_mock, result);
}

static void efa_unit_test_expect_primary_ctx_state(CUdevice device,
						   unsigned int flags,
						   int active,
						   CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuDevicePrimaryCtxGetState =
		efa_mock_ofi_cuDevicePrimaryCtxGetState_return_mock;
	expect_value(efa_mock_ofi_cuDevicePrimaryCtxGetState_return_mock,
		     dev, device);
	will_return(efa_mock_ofi_cuDevicePrimaryCtxGetState_return_mock,
		    flags);
	will_return(efa_mock_ofi_cuDevicePrimaryCtxGetState_return_mock,
		    active);
	will_return(efa_mock_ofi_cuDevicePrimaryCtxGetState_return_mock,
		    result);
}

static void efa_unit_test_expect_primary_ctx_set_flags(CUdevice device,
						       unsigned int flags,
						       CUresult result)
{
	g_efa_unit_test_mocks.ofi_cuDevicePrimaryCtxSetFlags =
		efa_mock_ofi_cuDevicePrimaryCtxSetFlags_return_mock;
	expect_value(efa_mock_ofi_cuDevicePrimaryCtxSetFlags_return_mock,
		     dev, device);
	expect_value(efa_mock_ofi_cuDevicePrimaryCtxSetFlags_return_mock,
		     flags, flags);
	will_return(efa_mock_ofi_cuDevicePrimaryCtxSetFlags_return_mock,
		    result);
}

void test_efa_hmem_set_sync_memops_pointer_success(void **state)
{
	void *ptr = (void *) 0x1234;
	efa_unit_test_expect_pointer_sync_memops(ptr, FI_SUCCESS);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 1), FI_SUCCESS);
}

void test_efa_hmem_set_sync_memops_pointer_failure(void **state)
{
	void *ptr = (void *) 0x1234;
	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EINVAL);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 1), -FI_EINVAL);
}

void test_efa_hmem_set_sync_memops_context_fallback_success(void **state)
{
	void *ptr = (void *) 0x1234;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(NULL, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_state(
		1, CU_CTX_SCHED_YIELD, 1, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_set_flags(
		1, CU_CTX_SCHED_YIELD | CU_CTX_SYNC_MEMOPS, CUDA_SUCCESS);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 1), FI_SUCCESS);
}

void test_efa_hmem_set_sync_memops_context_already_enabled(void **state)
{
	void *ptr = (void *) 0x1234;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(NULL, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_state(
		2, CU_CTX_SYNC_MEMOPS, 1, CUDA_SUCCESS);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 2), FI_SUCCESS);
}

void test_efa_hmem_set_sync_memops_context_get_failure(void **state)
{
	void *ptr = (void *) 0x1234;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(NULL, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_state(
		3, 0, 0, CUDA_ERROR_INVALID_DEVICE);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 3), -FI_EINVAL);
}

void test_efa_hmem_set_sync_memops_context_fallback_failure(void **state)
{
	void *ptr = (void *) 0x1234;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(NULL, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_state(4, 0, 1, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_set_flags(
		4, CU_CTX_SYNC_MEMOPS, CUDA_ERROR_NOT_SUPPORTED);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 4), -FI_EINVAL);
}

void test_efa_hmem_set_sync_memops_current_context_get_failure(void **state)
{
	void *ptr = (void *) 0x1234;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(NULL, CUDA_ERROR_INVALID_CONTEXT);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 5), -FI_EINVAL);
}

void test_efa_hmem_set_sync_memops_current_context_device_mismatch(
	void **state)
{
	void *ptr = (void *) 0x1234;
	CUcontext current_ctx = (CUcontext) (uintptr_t) 0x6000;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(current_ctx, CUDA_SUCCESS);
	efa_unit_test_expect_current_ctx_device(99, CUDA_SUCCESS);
	efa_unit_test_expect_primary_ctx_state(
		6, CU_CTX_SYNC_MEMOPS, 1, CUDA_SUCCESS);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 6), FI_SUCCESS);
}

void test_efa_hmem_set_sync_memops_current_context_success(void **state)
{
	void *ptr = (void *) 0x1234;
	CUcontext current_ctx = (CUcontext) (uintptr_t) 0x7000;

	efa_unit_test_expect_pointer_sync_memops(ptr, -FI_EOPNOTSUPP);
	efa_unit_test_expect_current_ctx(current_ctx, CUDA_SUCCESS);
	efa_unit_test_expect_current_ctx_device(7, CUDA_SUCCESS);
	efa_unit_test_expect_current_ctx_flags(
		CU_CTX_SCHED_YIELD, CUDA_SUCCESS);
	efa_unit_test_expect_current_ctx_set_flags(
		CU_CTX_SCHED_YIELD | CU_CTX_SYNC_MEMOPS, CUDA_SUCCESS);
	assert_int_equal(efa_hmem_set_sync_memops(ptr, 7), FI_SUCCESS);
}

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

	cu_result = ofi_cuCtxGetFlags(&final_flags);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuCtxGetFlags(before worker)";
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

	cu_result = ofi_cuCtxGetFlags(&final_flags);
	if (cu_result != CUDA_SUCCESS) {
		failed_api = "cuCtxGetFlags(after worker)";
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
void test_efa_hmem_set_sync_memops_pointer_success(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_pointer_failure(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_context_fallback_success(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_context_already_enabled(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_context_get_failure(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_context_fallback_failure(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_current_context_get_failure(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_current_context_device_mismatch(
	void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_current_context_success(void **state)
{
	skip();
}

void test_efa_hmem_set_sync_memops_active_primary_context_worker_thread(
	void **state)
{
	skip();
}
#endif
