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
