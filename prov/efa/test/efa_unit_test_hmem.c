/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"


#if HAVE_NEURON
/**
 * @brief Verify that Neuron p2p and dmabuf support are assumed without
 * explicit checking (to avoid early buffer allocation) if HAVE_EFA_DMABUF_MR is enabled.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_hmem_info_p2p_dmabuf_assumed_neuron(struct efa_resource **state)
{
        int ret;
        struct efa_resource *resource = *state;
        bool neuron_initialized_orig;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
        assert_non_null(resource->hints);

        ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
        assert_int_equal(ret, 0);

        neuron_initialized_orig = hmem_ops[FI_HMEM_NEURON].initialized;
        hmem_ops[FI_HMEM_NEURON].initialized = true;

        ret = efa_hmem_info_initialize();

        /* recover the modified global variables before doing check */
        hmem_ops[FI_HMEM_NEURON].initialized = neuron_initialized_orig;

        assert_int_equal(ret, 0);
        assert_true(g_efa_hmem_info[FI_HMEM_NEURON].initialized);
        assert_true(g_efa_hmem_info[FI_HMEM_NEURON].p2p_supported_by_device);
#if HAVE_EFA_DMABUF_MR
        assert_int_equal(g_efa_hmem_info[FI_HMEM_NEURON].dmabuf_supported_by_device, EFA_DMABUF_ASSUMED);
#else /* !HAVE_EFA_DMABUF_MR */
        assert_int_equal(g_efa_hmem_info[FI_HMEM_NEURON].dmabuf_supported_by_device, EFA_DMABUF_NOT_SUPPORTED);
#endif
}
#else
void test_efa_hmem_info_p2p_dmabuf_assumed_neuron()
{
        skip();
}
#endif /* HAVE_NEURON */

#if HAVE_CUDA
/**
 * @brief Verify when p2p is disabled, we don't check p2p support with ofi_cudaMalloc. 
 * Just leave p2p_supported_by_device to false for cuda.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_hmem_info_disable_p2p_cuda(struct efa_resource **state)
{
        int ret;
        struct efa_resource *resource = *state;
        bool cuda_initialized_orig;

        ofi_hmem_disable_p2p = 1;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
        assert_non_null(resource->hints);

        ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
        assert_int_equal(ret, 0);

        cuda_initialized_orig = hmem_ops[FI_HMEM_CUDA].initialized;
        hmem_ops[FI_HMEM_CUDA].initialized = true;
        /* ofi_cudaMalloc should not be called when p2p is disabled. efa_mock_ofi_cudaMalloc_return_mock will fail the test when it is called. */
        g_efa_unit_test_mocks.ofi_cudaMalloc = efa_mock_ofi_cudaMalloc_return_mock;

        ret = efa_hmem_info_initialize();

        /* recover the modified global variables before doing check */
        ofi_hmem_disable_p2p = 0;
        hmem_ops[FI_HMEM_CUDA].initialized = cuda_initialized_orig;

        assert_int_equal(ret, 0);
        assert_true(g_efa_hmem_info[FI_HMEM_CUDA].initialized);
        assert_false(g_efa_hmem_info[FI_HMEM_CUDA].p2p_supported_by_device);
}
#else
void test_efa_hmem_info_disable_p2p_cuda()
{
        skip();
}
#endif /* HAVE_CUDA */
