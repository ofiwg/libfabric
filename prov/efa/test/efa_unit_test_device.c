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
 * @brief Verify that efa_device_list_finalize skips device destruction
 * when ref_cnt > 0
 */
void test_efa_device_finalize_skips_active_device(void **state)
{
	struct efa_resource *resource = *state;
	struct efa_domain *efa_domain;
	struct efa_device *device;

	efa_unit_test_resource_construct(resource, FI_EP_RDM, EFA_FABRIC_NAME);

	efa_domain = container_of(resource->domain, struct efa_domain,
				  util_domain.domain_fid);
	device = efa_domain->device;

	assert_true(ofi_atomic_get32(&device->ref_cnt) > 0);

	efa_device_list_finalize();

	assert_non_null(device->qp_table);
	assert_non_null(g_efa_selected_device_list);
}

