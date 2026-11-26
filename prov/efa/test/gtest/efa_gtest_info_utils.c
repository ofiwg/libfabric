/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_prov_info.h"
#include "efa_gtest_info_utils.h"

void efa_test_set_queue_size_env(size_t tx, size_t rx, size_t *prev_tx,
				 size_t *prev_rx)
{
	if (prev_tx)
		*prev_tx = efa_env.tx_size;
	if (prev_rx)
		*prev_rx = efa_env.rx_size;

	efa_env.tx_size = tx;
	efa_env.rx_size = rx;
}

int efa_test_prov_info_link_speed(uint64_t port_speed, size_t *link_speed)
{
	struct efa_device device;
	struct fi_info *info = NULL;
	int ret;

	memcpy(&device, &g_efa_selected_device_list[0], sizeof(device));
	device.port_speed = port_speed;

	ret = efa_prov_info_alloc(&info, &device, FI_EP_RDM);
	if (ret)
		return ret;

	*link_speed = info->nic->link_attr->speed;
	fi_freeinfo(info);
	return 0;
}

size_t efa_test_device_vrb_link_speed(void)
{
	struct ibv_port_attr *attr = &g_efa_selected_device_list[0].ibv_port_attr;

	return ofi_vrb_speed(attr->active_speed, attr->active_width);
}

uint64_t efa_test_device_port_speed(void)
{
	return g_efa_selected_device_list[0].port_speed;
}

size_t efa_test_device_rdm_info_link_speed(void)
{
	return g_efa_selected_device_list[0].rdm_info->nic->link_attr->speed;
}
