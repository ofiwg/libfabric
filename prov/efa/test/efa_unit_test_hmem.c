/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"


#if HAVE_NEURON
/**
 * @brief Verify when neuron_alloc failed (return null),
 * efa_hmem_info_initialize will still return 0 but leave
 * efa_hmem_info[FI_HMEM_NEURON].initialized and
 * efa_hmem_info[FI_HMEM_NEURON].p2p_supported_by_device as false.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_hmem_info_update_neuron(struct efa_resource **state)
{
        int ret;
        struct efa_resource *resource = *state;
        uint32_t efa_device_caps_orig;
        bool neuron_initialized_orig;

        resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
        assert_non_null(resource->hints);

        ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
        assert_int_equal(ret, 0);

        neuron_initialized_orig = hmem_ops[FI_HMEM_NEURON].initialized;
        hmem_ops[FI_HMEM_NEURON].initialized = true;
        efa_device_caps_orig = g_device_list[0].device_caps;
        g_device_list[0].device_caps |= EFADV_DEVICE_ATTR_CAPS_RDMA_READ;
        g_efa_unit_test_mocks.neuron_alloc = &efa_mock_neuron_alloc_return_null;

        ret = efa_hmem_info_initialize();

        /* recover the modified global variables before doing check */
        hmem_ops[FI_HMEM_NEURON].initialized = neuron_initialized_orig;
        g_device_list[0].device_caps = efa_device_caps_orig;

        assert_int_equal(ret, 0);
        assert_false(g_efa_hmem_info[FI_HMEM_NEURON].initialized);
        assert_false(g_efa_hmem_info[FI_HMEM_NEURON].p2p_supported_by_device);
}

/**
 * @brief Verify when p2p is disabled, we don't check p2p support with neuron_alloc.
 * Just leave p2p_supported_by_device to false.
 *
 * @param[in]	state		struct efa_resource that is managed by the framework
 */
void test_efa_hmem_info_disable_p2p_neuron(struct efa_resource **state)
{
        int ret;
        struct efa_resource *resource = *state;
        uint32_t efa_device_caps_orig;
        bool neuron_initialized_orig;

        ofi_hmem_disable_p2p = 1;

	resource->hints = efa_unit_test_alloc_hints(FI_EP_RDM, EFA_FABRIC_NAME);
        assert_non_null(resource->hints);

        ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, resource->hints, &resource->info);
        assert_int_equal(ret, 0);

        neuron_initialized_orig = hmem_ops[FI_HMEM_NEURON].initialized;
        hmem_ops[FI_HMEM_NEURON].initialized = true;
        efa_device_caps_orig = g_device_list[0].device_caps;
        g_device_list[0].device_caps |= EFADV_DEVICE_ATTR_CAPS_RDMA_READ;
        /* neuron_alloc should not be called when p2p is disabled. efa_mock_neuron_alloc_return_mock will fail the test when it is called. */
        g_efa_unit_test_mocks.neuron_alloc = efa_mock_neuron_alloc_return_mock;

        ret = efa_hmem_info_initialize();

        /* recover the modified global variables before doing check */
        ofi_hmem_disable_p2p = 0;
        g_device_list[0].device_caps = efa_device_caps_orig;
        hmem_ops[FI_HMEM_NEURON].initialized = neuron_initialized_orig;

        assert_int_equal(ret, 0);
        assert_true(g_efa_hmem_info[FI_HMEM_NEURON].initialized);
        assert_false(g_efa_hmem_info[FI_HMEM_NEURON].p2p_supported_by_device);
}
#else
void test_efa_hmem_info_update_neuron()
{
        skip();
}

void test_efa_hmem_info_disable_p2p_neuron()
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
