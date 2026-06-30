/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_unit_tests.h"

/*
 * test the error handling path of efa_device_construct()
 */
void test_efa_device_construct_error_handling(void **state)
{
	int ibv_err = 4242;
	struct ibv_device **ibv_device_list;
	struct efa_device efa_device = {0};

	ibv_device_list = ibv_get_device_list(&g_efa_selected_device_cnt);
	if (ibv_device_list == NULL) {
		skip();
		return;
	}

	g_efa_unit_test_mocks.efadv_query_device = &efa_mock_efadv_query_device_return_mock;
	will_return(efa_mock_efadv_query_device_return_mock, ibv_err);

	efa_device_construct_gid(&efa_device, ibv_device_list[0]);

	/* when error happend, resources in efa_device should be NULL */
	assert_null(efa_device.ibv_ctx);
	assert_null(efa_device.rdm_info);
	assert_null(efa_device.dgram_info);

	ibv_free_device_list(ibv_device_list);
}

/**
 * @brief Verify that qp_table_initialized is set after successful device
 * construction and that it is accessible from the domain.
 */
void test_efa_device_qp_table_initialized_after_construct(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	assert_int_equal(
		ofi_atomic_load_explicit32(&efa_domain->device->qp_table_initialized,
					   memory_order_acquire), 1);
}

/**
 * @brief Verify that fi_close(ep) does not crash when
 * qp_table_initialized is 0 (simulates fi_fini teardown race).
 */
void test_efa_device_ep_close_after_table_invalidated(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);

	/* Simulate the destructor clearing the flag */
	ofi_atomic_store_explicit32(&efa_domain->device->qp_table_initialized, 0,
				    memory_order_release);

	/* fi_close(ep) should not crash — destruct_qp_unsafe skips the qp_table write */
	assert_int_equal(fi_close(&resource->ep->fid), 0);
	resource->ep = NULL;

	/* Restore for clean teardown of remaining resources */
	ofi_atomic_store_explicit32(&efa_domain->device->qp_table_initialized, 1,
				    memory_order_release);
}

