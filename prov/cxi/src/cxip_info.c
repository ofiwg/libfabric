/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

/* CXI fabric discovery implementation. */

#include "ofi_prov.h"
#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_FABRIC, __VA_ARGS__)

char cxip_prov_name[] = "cxi";

struct fi_fabric_attr cxip_fabric_attr = {
	.prov_version = CXIP_PROV_VERSION,
	.name = cxip_prov_name,
};

struct fi_domain_attr cxip_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_ENDPOINT | FI_MR_SCALABLE,
	.mr_key_size = sizeof(uint64_t),
	.cq_data_size = 8,
	.cq_cnt = 32,
	.ep_cnt = 128,
	.tx_ctx_cnt = 16,
	.rx_ctx_cnt = 16,
	.max_ep_tx_ctx = 16,
	.max_ep_rx_ctx = 16,
	.max_ep_stx_ctx = 16,
	.max_ep_srx_ctx = 16,
	.cntr_cnt = 0,
	.mr_iov_limit = 1,
	.mr_cnt = 100,
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
};

struct fi_ep_attr cxip_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_CXI,
	.protocol_version = CXIP_WIRE_PROTO_VERSION,
	.max_msg_size = CXIP_EP_MAX_MSG_SZ,
	.max_order_raw_size = 0,
	.max_order_war_size = 0,
	.max_order_waw_size = 0,
	.mem_tag_format = FI_TAG_GENERIC,
};

struct fi_tx_attr cxip_tx_attr = {
	.caps = CXIP_EP_CAPS,
	.op_flags = CXIP_TX_COMP_MODES | FI_COMPLETION,
	.msg_order = CXIP_EP_MSG_ORDER,
	.inject_size = C_MAX_IDC_PAYLOAD_UNR,
	.size = 256,  /* 64k / 256b */
	.iov_limit = 1,
	.rma_iov_limit = 1,
};

struct fi_rx_attr cxip_rx_attr = {
	.caps = CXIP_EP_CAPS,
	.op_flags = FI_COMPLETION,
	.msg_order = CXIP_EP_MSG_ORDER,
	.comp_order = FI_ORDER_NONE,
	.total_buffered_recv = CXIP_UX_BUFFER_SIZE,
	.size = 1024, /* 64k / 64b */
	.iov_limit = 1,
};

struct fi_info cxip_info = {
	.caps = CXIP_EP_CAPS,
	.addr_format = FI_ADDR_CXI,
	.tx_attr = &cxip_tx_attr,
	.rx_attr = &cxip_rx_attr,
	.ep_attr = &cxip_ep_attr,
	.domain_attr = &cxip_domain_attr,
	.fabric_attr = &cxip_fabric_attr,
};

struct fi_provider cxip_prov;

struct util_prov cxip_util_prov = {
	.prov = &cxip_prov,
	.info = NULL,
	.flags = 0,
};

/*
 * cxip_info_alloc() - Create a fabric info structure for the CXI interface.
 */
static int cxip_info_alloc(struct cxip_if *nic_if, struct fi_info **info)
{
	int ret;
	struct fi_info *fi;
	struct cxip_addr addr;

	fi = fi_dupinfo(&cxip_info);
	if (!fi)
		return -FI_ENOMEM;

	ret = asprintf(&fi->domain_attr->name, cxip_dom_fmt, nic_if->if_idx);
	assert(ret > 0);

	addr.nic = nic_if->if_nic;
	addr.pid = C_PID_ANY;
	fi->src_addr = mem_dup(&addr, sizeof(addr));
	if (!fi->src_addr) {
		ret = -ENOMEM;
		goto err;
	}
	fi->src_addrlen = sizeof(addr);

	fi->nic = ofi_nic_dup(NULL);
	if (!fi->nic) {
		ret = -FI_ENOMEM;
		goto err;
	}

	fi->nic->device_attr->name = strdup(nic_if->if_info.device_name);

	ret = asprintf(&fi->nic->device_attr->device_id, "0x%x",
		       nic_if->if_info.device_id);
	if (ret < 0)
		goto err;

	ret = asprintf(&fi->nic->device_attr->device_version, "%u",
		       nic_if->if_info.device_rev);
	if (ret < 0)
		goto err;

	ret = asprintf(&fi->nic->device_attr->vendor_id, "0x%x",
		       nic_if->if_info.vendor_id);
	if (ret < 0)
		goto err;

	fi->nic->device_attr->driver = strdup(nic_if->if_info.driver_name);

	fi->nic->bus_attr->bus_type = FI_BUS_PCI;
	fi->nic->bus_attr->attr.pci.domain_id = nic_if->if_info.pci_domain;
	fi->nic->bus_attr->attr.pci.bus_id = nic_if->if_info.pci_bus;
	fi->nic->bus_attr->attr.pci.device_id = nic_if->if_info.pci_device;
	fi->nic->bus_attr->attr.pci.function_id =
			nic_if->if_info.pci_function;

	ret = asprintf(&fi->nic->link_attr->address, "0x%x",
		       nic_if->if_info.nic_addr);
	if (ret < 0)
		goto err;

	fi->nic->link_attr->mtu = nic_if->if_info.link_mtu;
	fi->nic->link_attr->speed = nic_if->if_info.link_speed * 1000 * 1000;
	fi->nic->link_attr->state = nic_if->if_info.link_state ?
			FI_LINK_UP : FI_LINK_DOWN;
	fi->nic->link_attr->network_type = strdup("HPC Ethernet");

	*info = fi;
	return FI_SUCCESS;

err:
	fi_freeinfo((void *)fi);
	return ret;
}

/*
 * cxip_info_init() - Initialize fabric info for each CXI interface.
 */
static int cxip_info_init(void)
{
	struct slist_entry *entry, *prev __attribute__ ((unused));
	struct cxip_if *nic_if;
	struct fi_info **fi_list = (void *)&cxip_util_prov.info;
	struct fi_info *fi;
	int ret;

	slist_foreach(&cxip_if_list, entry, prev) {
		nic_if = container_of(entry, struct cxip_if, if_entry);
		ret = cxip_info_alloc(nic_if, &fi);
		if (ret != FI_SUCCESS) {
			fi_freeinfo((void *)cxip_util_prov.info);
			break;
		}
		CXIP_LOG_DBG("NIC: 0x%x info created\n", nic_if->if_nic);

		*fi_list = fi;
		fi_list = &(fi->next);
	}

	return ret;
}

/*
 * CXI_INI - Provider constructor.
 */
CXI_INI
{
	cxip_if_init();

	cxip_info_init();

	cxip_fault_inject_init();

	return &cxip_prov;
}

/*
 * cxip_fini() - Provider destructor.
 */
static void cxip_fini(void)
{
	cxip_fault_inject_fini();

	fi_freeinfo((void *)cxip_util_prov.info);

	cxip_if_fini();
}

/*
 * cxip_getinfo() - Provider fi_getinfo() implementation.
 */
static int
cxip_getinfo(uint32_t version, const char *node, const char *service,
	     uint64_t flags, const struct fi_info *hints,
	     struct fi_info **info)
{
	int ret;

	/* Find all matching domains, ignoring addresses. */
	ret = util_getinfo(&cxip_util_prov, version, NULL, NULL,
			   flags & ~FI_SOURCE, hints, info);

	/* TODO refine info list with node and service. */

	return ret;
}

struct fi_provider cxip_prov = {
	.name = cxip_prov_name,
	.version = CXIP_PROV_VERSION,
	.fi_version = CXIP_FI_VERSION,
	.getinfo = cxip_getinfo,
	.fabric = cxip_fabric,
	.cleanup = cxip_fini,
};
