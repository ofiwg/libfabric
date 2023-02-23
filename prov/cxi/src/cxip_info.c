/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019, 2022 Cray Inc. All rights reserved.
 */

/* CXI fabric discovery implementation. */

#include "ofi_prov.h"
#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_FABRIC, __VA_ARGS__)

char cxip_prov_name[] = "cxi";

struct fi_fabric_attr cxip_fabric_attr = {
	.prov_version = CXIP_PROV_VERSION,
	.name = cxip_prov_name,
};

/* No ODP, provider specified MR keys */
struct fi_domain_attr cxip_prov_key_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_PROV_KEY | FI_MR_ALLOCATED | FI_MR_ENDPOINT,
	.mr_key_size = CXIP_MR_PROV_KEY_SIZE,
	.cq_data_size = CXIP_REMOTE_CQ_DATA_SZ,
	.cq_cnt = 32,
	.ep_cnt = 128,
	.tx_ctx_cnt = CXIP_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXIP_EP_MAX_RX_CNT,
	.max_ep_tx_ctx = CXIP_EP_MAX_TX_CNT,
	.max_ep_rx_ctx = CXIP_EP_MAX_RX_CNT,
	.max_ep_stx_ctx = 0,
	.max_ep_srx_ctx = 0,
	.cntr_cnt = 16,
	.mr_iov_limit = 1,
	.mr_cnt = 100,
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
	.auth_key_size = sizeof(struct cxi_auth_key),
};

/* ODP, provider specified MR keys */
struct fi_domain_attr cxip_odp_prov_key_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_PROV_KEY | FI_MR_ENDPOINT,
	.mr_key_size = CXIP_MR_PROV_KEY_SIZE,
	.cq_data_size = CXIP_REMOTE_CQ_DATA_SZ,
	.cq_cnt = 32,
	.ep_cnt = 128,
	.tx_ctx_cnt = CXIP_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXIP_EP_MAX_RX_CNT,
	.max_ep_tx_ctx = CXIP_EP_MAX_TX_CNT,
	.max_ep_rx_ctx = CXIP_EP_MAX_RX_CNT,
	.max_ep_stx_ctx = 0,
	.max_ep_srx_ctx = 0,
	.cntr_cnt = 16,
	.mr_iov_limit = 1,
	.mr_cnt = 100,
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
	.auth_key_size = sizeof(struct cxi_auth_key),
};

/* No ODP, client specified MR keys */
struct fi_domain_attr cxip_client_key_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_ENDPOINT | FI_MR_ALLOCATED,
	.mr_key_size = CXIP_MR_KEY_SIZE,
	.cq_data_size = CXIP_REMOTE_CQ_DATA_SZ,
	.cq_cnt = 32,
	.ep_cnt = 128,
	.tx_ctx_cnt = CXIP_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXIP_EP_MAX_RX_CNT,
	.max_ep_tx_ctx = CXIP_EP_MAX_TX_CNT,
	.max_ep_rx_ctx = CXIP_EP_MAX_RX_CNT,
	.max_ep_stx_ctx = 0,
	.max_ep_srx_ctx = 0,
	.cntr_cnt = 16,
	.mr_iov_limit = 1,
	.mr_cnt = 100,
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
	.auth_key_size = sizeof(struct cxi_auth_key),
};

/* ODP, client specified MR keys */
struct fi_domain_attr cxip_odp_client_key_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_MANUAL,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_ENDPOINT,
	.mr_key_size = CXIP_MR_KEY_SIZE,
	.cq_data_size = CXIP_REMOTE_CQ_DATA_SZ,
	.cq_cnt = 32,
	.ep_cnt = 128,
	.tx_ctx_cnt = CXIP_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXIP_EP_MAX_RX_CNT,
	.max_ep_tx_ctx = CXIP_EP_MAX_TX_CNT,
	.max_ep_rx_ctx = CXIP_EP_MAX_RX_CNT,
	.max_ep_stx_ctx = 0,
	.max_ep_srx_ctx = 0,
	.cntr_cnt = 16,
	.mr_iov_limit = 1,
	.mr_cnt = 100,
	.caps = FI_LOCAL_COMM | FI_REMOTE_COMM,
	.auth_key_size = sizeof(struct cxi_auth_key),
};

struct fi_ep_attr cxip_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_CXI,
	.protocol_version = CXIP_WIRE_PROTO_VERSION,
	.max_msg_size = CXIP_EP_MAX_MSG_SZ,
	.max_order_raw_size = -1,
	.max_order_war_size = -1,
	.max_order_waw_size = -1,
	.mem_tag_format = FI_TAG_GENERIC >> (64 - CXIP_TAG_WIDTH),
	.auth_key_size = sizeof(struct cxi_auth_key),
};

struct fi_tx_attr cxip_tx_attr = {
	.caps = CXIP_EP_CAPS & ~OFI_IGNORED_TX_CAPS,
	.op_flags = CXIP_TX_OP_FLAGS,
	.msg_order = CXIP_MSG_ORDER,
	.inject_size = CXIP_INJECT_SIZE,
	.size = CXIP_MAX_TX_SIZE,
	.iov_limit = 1,
	.rma_iov_limit = 1,
};

struct fi_rx_attr cxip_rx_attr = {
	.caps = CXIP_EP_CAPS & ~OFI_IGNORED_RX_CAPS,
	.op_flags = CXIP_RX_OP_FLAGS,
	.msg_order = CXIP_MSG_ORDER,
	.comp_order = FI_ORDER_NONE,
	.total_buffered_recv = CXIP_UX_BUFFER_SIZE,
	.size = 1024, /* 64k / 64b */
	.iov_limit = 1,
};

/* The CXI provider supports multiple operating modes by exporting
 * several fi_info structures. The application can filter the fi_info
 * with hints, or choose the fi_info based on desired application
 * behavior. Matched fi_info are returned in the order of highest
 * to lowest provider performance.:
 *
 * 1. Pinned memory with provider MR Keys
 * 2. Pinned memory with application provided MR Keys
 * 3. On-Demand paging with provider MR Keys
 * 4. On-Demand paging with application provided MR Keys
 */
struct fi_info cxip_infos[] = {
	{
		.caps = CXIP_EP_CAPS,
		.addr_format = FI_ADDR_CXI,
		.tx_attr = &cxip_tx_attr,
		.rx_attr = &cxip_rx_attr,
		.ep_attr = &cxip_ep_attr,
		.domain_attr = &cxip_prov_key_domain_attr,
		.fabric_attr = &cxip_fabric_attr,
	},
	{
		.caps = CXIP_EP_CAPS,
		.addr_format = FI_ADDR_CXI,
		.tx_attr = &cxip_tx_attr,
		.rx_attr = &cxip_rx_attr,
		.ep_attr = &cxip_ep_attr,
		.domain_attr = &cxip_client_key_domain_attr,
		.fabric_attr = &cxip_fabric_attr,
	},
	{
		.caps = CXIP_EP_CAPS,
		.addr_format = FI_ADDR_CXI,
		.tx_attr = &cxip_tx_attr,
		.rx_attr = &cxip_rx_attr,
		.ep_attr = &cxip_ep_attr,
		.domain_attr = &cxip_odp_prov_key_domain_attr,
		.fabric_attr = &cxip_fabric_attr,
	},
	{
		.caps = CXIP_EP_CAPS,
		.addr_format = FI_ADDR_CXI,
		.tx_attr = &cxip_tx_attr,
		.rx_attr = &cxip_rx_attr,
		.ep_attr = &cxip_ep_attr,
		.domain_attr = &cxip_odp_client_key_domain_attr,
		.fabric_attr = &cxip_fabric_attr,
	},
};

struct fi_provider cxip_prov;

struct util_prov cxip_util_prov = {
	.prov = &cxip_prov,
	.info = NULL,
	.flags = 0,
};

/*
 * Test platform tracing function, if NULL, disabled.
 */
cxip_trace_t cxip_trace_attr cxip_trace_fn;

/*
 * cxip_info_alloc() - Create a fabric info structure for the CXI interface.
 */
static int cxip_info_alloc(struct cxip_if *nic_if, int info_index,
			   struct fi_info **info)
{
	int ret;
	struct fi_info *fi;
	struct cxip_addr addr = {};

	/* For now only expose ODP fi_info if ODP selection is enabled.
	 * TODO: When ODP is always available remove this filter.
	 */
	if (!(cxip_infos[info_index].domain_attr->mr_mode & FI_MR_ALLOCATED) &&
	    !cxip_env.odp)
		return -FI_ENODATA;

	fi = fi_dupinfo(&cxip_infos[info_index]);
	if (!fi)
		return -FI_ENOMEM;

	fi->domain_attr->name = strdup(nic_if->info->device_name);
	if (!fi->domain_attr->name)
		return -ENOMEM;

	addr.nic = nic_if->info->nic_addr;
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

	fi->nic->device_attr->name = strdup(nic_if->info->device_name);
	if (!fi->nic->device_attr->name) {
		ret = -ENOMEM;
		goto err;
	}

	ret = asprintf(&fi->nic->device_attr->device_id, "0x%x",
		       nic_if->info->device_id);
	if (ret < 0)
		goto err;

	ret = asprintf(&fi->nic->device_attr->device_version, "%u",
		       nic_if->info->device_rev);
	if (ret < 0)
		goto err;

	ret = asprintf(&fi->nic->device_attr->vendor_id, "0x%x",
		       nic_if->info->vendor_id);
	if (ret < 0)
		goto err;

	fi->nic->device_attr->driver = strdup(nic_if->info->driver_name);

	fi->nic->bus_attr->bus_type = FI_BUS_PCI;
	fi->nic->bus_attr->attr.pci.domain_id = nic_if->info->pci_domain;
	fi->nic->bus_attr->attr.pci.bus_id = nic_if->info->pci_bus;
	fi->nic->bus_attr->attr.pci.device_id = nic_if->info->pci_device;
	fi->nic->bus_attr->attr.pci.function_id = nic_if->info->pci_function;

	ret = asprintf(&fi->nic->link_attr->address, "0x%x",
		       nic_if->info->nic_addr);
	if (ret < 0)
		goto err;

	fi->nic->link_attr->mtu = nic_if->info->link_mtu;
	/* Convert Mb/s to libfabric reported b/s */
	fi->nic->link_attr->speed = (size_t)nic_if->speed * 1000000;
	fi->nic->link_attr->state = nic_if->link ?  FI_LINK_UP : FI_LINK_DOWN;
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
	int ndx;
	int ret;

	slist_foreach(&cxip_if_list, entry, prev) {
		nic_if = container_of(entry, struct cxip_if, if_entry);

		/* TODO: Remove check before pushing upstream. This is a
		 * CXI provider helper to not export the new constants if
		 * they break an application. Upstream we will always add
		 * the new constants.
		 */
		if (ofi_cxi_compat == 2)
			goto add_compat;

		for (ndx = 0; ndx < ARRAY_SIZE(cxip_infos); ndx++) {
			ret = cxip_info_alloc(nic_if, ndx, &fi);
			if (ret == -FI_ENODATA)
				continue;
			if (ret != FI_SUCCESS)
				goto free_info;

			CXIP_DBG("%s info created\n",
				 nic_if->info->device_name);
			*fi_list = fi;
			fi_list = &(fi->next);
		}

		/* TODO: The compat infos MUST be removed before pushing
		 * upstream. If requested add the CXI provider helper fi_info
		 * to handle client usage of old values for FI_ADDR_CXI and
		 * FI_PROTO_CXI.
		 */
		if (!ofi_cxi_compat)
			continue;

add_compat:
		for (ndx = 0; ndx < ARRAY_SIZE(cxip_infos); ndx++) {
			ret = cxip_info_alloc(nic_if, ndx, &fi);
			if (ret == -FI_ENODATA)
				continue;
			if (ret != FI_SUCCESS)
				goto free_info;

			fi->addr_format = FI_ADDR_OPX;
			fi->ep_attr->protocol = FI_PROTO_OPX;

			CXIP_DBG("%s compat info created\n",
				 nic_if->info->device_name);
			*fi_list = fi;
			fi_list = &(fi->next);
		}
	}

	return FI_SUCCESS;

free_info:
	fi_freeinfo((void *)cxip_util_prov.info);
	return ret;
}

static bool cxip_env_validate_device_token(const char *device_token)
{
	unsigned int device_index;
	unsigned int device_strlen;

	/* Only allow for device tokens of cxi0 - cxi99. */
	device_strlen = strlen(device_token);
	if (device_strlen != 4 && device_strlen != 5)
		return false;

	/* Ensure device token is of cxi## format. */
	if (sscanf(device_token, "cxi%u", &device_index) != 1)
		return false;

	/* Ensure that a device string length of 5 chars is only true if the
	 * device index is greater than 9.
	 */
	if (device_strlen == 5 && device_index < 10)
		return false;

	return true;
}

static int cxip_env_validate_device_name(const char *device_name)
{
	const char *device_token;
	char *device_name_copy;
	int ret = FI_SUCCESS;

	device_name_copy = malloc(strlen(device_name) + 1);
	if (!device_name_copy)
		return -FI_ENOMEM;

	strcpy(device_name_copy, device_name);

	device_token = strtok(device_name_copy, ",");
	while (device_token != NULL) {
		if (!cxip_env_validate_device_token(device_token)) {
			ret = -FI_EINVAL;
			break;
		}

		device_token = strtok(NULL, ",");
	}

	free(device_name_copy);

	return ret;
}

static int cxip_env_validate_url(const char *url)
{
	/* Trying to validate further is likely to generate false failures */
	if (url && strlen(url) > 7 && !strncasecmp(url, "http://", 7))
		return FI_SUCCESS;
	if (url && strlen(url) > 8 && !strncasecmp(url, "https://", 8))
		return FI_SUCCESS;
	return -FI_EINVAL;
}

/* Provider environment variables are FI_CXI_{NAME} in all-caps */
struct cxip_environment cxip_env = {
	.odp = false,
	.ats = false,
	.iotlb = true,
	.ats_mlock_mode = CXIP_ATS_MLOCK_ALL,
	.rx_match_mode = CXIP_PTLTE_DEFAULT_MODE,
	.rdzv_threshold = CXIP_RDZV_THRESHOLD,
	.rdzv_get_min = 2049, /* Avoid single packet Gets */
	.rdzv_eager_size = CXIP_RDZV_THRESHOLD,
	.rdzv_aligned_sw_rget = 1,
	.disable_non_inject_msg_idc = 0,
	.disable_host_register = 0,
	.oflow_buf_size = CXIP_OFLOW_BUF_SIZE,
	.oflow_buf_min_posted = CXIP_OFLOW_BUF_MIN_POSTED,
	.oflow_buf_max_cached = CXIP_OFLOW_BUF_MAX_CACHED,
	.safe_devmem_copy_threshold =  CXIP_SAFE_DEVMEM_COPY_THRESH,
	.optimized_mrs = true,
	.llring_mode = CXIP_LLRING_IDLE,
	.cq_policy = CXI_CQ_UPDATE_LOW_FREQ_EMPTY,
	.default_vni = 10,
	.eq_ack_batch_size = 32,
	.req_buf_size = CXIP_REQ_BUF_SIZE,
	.req_buf_min_posted = CXIP_REQ_BUF_MIN_POSTED,
	.req_buf_max_cached = CXIP_REQ_BUF_MAX_CACHED,
	.msg_offload = 1,
	.msg_lossless = 0,
	.hybrid_preemptive = 0,
	.hybrid_recv_preemptive = 0,
	.fc_retry_usec_delay = 1000,
	.ctrl_rx_eq_max_size = 67108864,
	.default_cq_size = CXIP_CQ_DEF_SZ,
	.default_tx_size = CXIP_DEFAULT_TX_SIZE,
	.disable_eq_hugetlb = false,
	.zbcoll_radix = 2,
	.cq_fill_percent = 50,
	.enable_unrestricted_end_ro = true,
	.rget_tc = FI_TC_UNSPEC,
	.cacheline_size = CXIP_DEFAULT_CACHE_LINE_SIZE,
	.coll_job_id = NULL,
	.coll_step_id = NULL,
	.coll_timeout_usec = 0,
	.coll_fabric_mgr_url = NULL,
	.coll_fabric_mgr_token = NULL,
	.coll_use_dma_put = false,
	.coll_use_repsum = false,
	.telemetry_rgid = -1,
	.disable_hmem_dev_register = 0,
};

static void cxip_env_init(void)
{
	char *param_str = NULL;
	size_t min_free;
	int ret;

	gethostname(cxip_env.hostname, sizeof(cxip_env.hostname));

	fi_param_define(&cxip_prov, "rget_tc", FI_PARAM_STRING,
			"Traffic class used for software initiated rendezvous gets.");
	fi_param_get_str(&cxip_prov, "rget_tc", &param_str);

	if (param_str) {
		if (!strcmp(param_str, "BEST_EFFORT"))
			cxip_env.rget_tc = FI_TC_BEST_EFFORT;
		else if (!strcmp(param_str, "LOW_LATENCY"))
			cxip_env.rget_tc = FI_TC_LOW_LATENCY;
		else if (!strcmp(param_str, "DEDICATED_ACCESS"))
			cxip_env.rget_tc = FI_TC_DEDICATED_ACCESS;
		else if (!strcmp(param_str, "BULK_DATA"))
			cxip_env.rget_tc = FI_TC_BULK_DATA;
		else
			CXIP_WARN("Unrecognized rget_tc: %s\n", param_str);
		param_str = NULL;
	}

	cxip_env.cacheline_size = cxip_cacheline_size();
	CXIP_DBG("Provider using cacheline size of %d\n",
		 cxip_env.cacheline_size);

	fi_param_define(&cxip_prov, "rdzv_aligned_sw_rget", FI_PARAM_BOOL,
			"Enables SW RGet address alignment (default: %d).",
			cxip_env.rdzv_aligned_sw_rget);
	fi_param_get_bool(&cxip_prov, "rdzv_aligned_sw_rget",
			  &cxip_env.rdzv_aligned_sw_rget);

	fi_param_define(&cxip_prov, "disable_non_inject_msg_idc", FI_PARAM_BOOL,
			"Disables IDC for non-inject messages (default: %d).",
			cxip_env.disable_non_inject_msg_idc);
	fi_param_get_bool(&cxip_prov, "disable_non_inject_msg_idc",
			  &cxip_env.disable_non_inject_msg_idc);

	fi_param_define(&cxip_prov, "disable_host_register", FI_PARAM_BOOL,
			"Disables host buffer GPU registration (default: %d).",
			cxip_env.disable_host_register);
	fi_param_get_bool(&cxip_prov, "disable_host_register",
			  &cxip_env.disable_host_register);

	fi_param_define(&cxip_prov, "enable_unrestricted_end_ro", FI_PARAM_BOOL,
			"Default: %d", cxip_env.enable_unrestricted_end_ro);
	fi_param_get_bool(&cxip_prov, "enable_unrestricted_end_ro",
			  &cxip_env.enable_unrestricted_end_ro);

	fi_param_define(&cxip_prov, "odp", FI_PARAM_BOOL,
			"Enables on-demand paging.");
	fi_param_get_bool(&cxip_prov, "odp", &cxip_env.odp);

	fi_param_define(&cxip_prov, "ats", FI_PARAM_BOOL,
			"Enables PCIe ATS.");
	fi_param_get_bool(&cxip_prov, "ats", &cxip_env.ats);

	fi_param_define(&cxip_prov, "iotlb", FI_PARAM_BOOL,
			"Enables the NIC IOTLB (default %d).", cxip_env.iotlb);
	fi_param_get_bool(&cxip_prov, "iotlb", &cxip_env.iotlb);

	fi_param_define(&cxip_prov, "ats_mlock_mode", FI_PARAM_STRING,
			"Sets ATS mlock mode (off | all).");
	fi_param_get_str(&cxip_prov, "ats_mlock_mode", &param_str);

	if (param_str) {
		if (!strcmp(param_str, "off"))
			cxip_env.ats_mlock_mode = CXIP_ATS_MLOCK_OFF;
		else if (!strcmp(param_str, "all"))
			cxip_env.ats_mlock_mode = CXIP_ATS_MLOCK_ALL;
		else
			CXIP_WARN("Unrecognized ats_mlock_mode: %s\n",
				  param_str);
		param_str = NULL;
	}

	fi_param_define(&cxip_prov, "device_name", FI_PARAM_STRING,
			"Restrict CXI provider to specific CXI devices. Format is a comma separated list of CXI devices (e.g. cxi0,cxi1).");
	fi_param_get_str(&cxip_prov, "device_name", &cxip_env.device_name);

	if (cxip_env.device_name) {
		ret = cxip_env_validate_device_name(cxip_env.device_name);
		if (ret) {
			CXIP_WARN("Failed to validate device name: name=%s rc=%d. Ignoring device name.\n",
				  cxip_env.device_name, ret);
			cxip_env.device_name = NULL;
		}
	}

	/* Counters env string is validate when the cxip_env.telemetry string
	 * is used.
	 */
	fi_param_define(&cxip_prov, "telemetry", FI_PARAM_STRING,
			"Perform a telemetry delta captured between fi_domain open and close. "
			"Format is a comma separated list of telemetry files as defined in /sys/class/cxi/cxi*/device/telemetry/. "
			"Default is counter delta captured disabled.");
	fi_param_get_str(&cxip_prov, "telemetry", &cxip_env.telemetry);

	fi_param_define(&cxip_prov, "telemetry_rgid", FI_PARAM_INT,
			"Resource group ID (RGID) to restrict the telemetry collection to. "
			"Value less than 0 is no restrictions. "
			"Default is no restrictions.");
	fi_param_get_int(&cxip_prov, "telemetry_rgid",
			 &cxip_env.telemetry_rgid);

	fi_param_define(&cxip_prov, "rx_match_mode", FI_PARAM_STRING,
			"Sets RX message match mode (hardware | software | hybrid).");
	fi_param_get_str(&cxip_prov, "rx_match_mode", &param_str);

	if (param_str) {
		if (!strcasecmp(param_str, "hardware")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_HARDWARE_MODE;
			cxip_env.msg_offload = true;
		} else if (!strcmp(param_str, "software")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_SOFTWARE_MODE;
			cxip_env.msg_offload = false;
		} else if (!strcmp(param_str, "hybrid")) {
			cxip_env.rx_match_mode = CXIP_PTLTE_HYBRID_MODE;
			cxip_env.msg_offload = true;
		} else {
			CXIP_WARN("Unrecognized rx_match_mode: %s\n",
				  param_str);
			cxip_env.rx_match_mode = CXIP_PTLTE_HARDWARE_MODE;
			cxip_env.msg_offload = true;
		}

		param_str = NULL;
	}

	fi_param_define(&cxip_prov, "rdzv_threshold", FI_PARAM_SIZE_T,
			"Message size threshold for rendezvous protocol.");
	fi_param_get_size_t(&cxip_prov, "rdzv_threshold",
			    &cxip_env.rdzv_threshold);

	/* Rendezvous protocol does support FI_INJECT, make sure
	 * eager send message is selected for FI_INJECT.
	 */
	if (cxip_env.rdzv_threshold < CXIP_INJECT_SIZE) {
		cxip_env.rdzv_threshold = CXIP_INJECT_SIZE;
		CXIP_WARN("Increased rdzv_threshold size to: %lu\n",
			  cxip_env.rdzv_threshold);
	}

	/* If aligned SW Rget is enabled, rendezvous eager data must
	 * be greater than cache-line size.
	 */
	if (cxip_env.rdzv_aligned_sw_rget &&
	    cxip_env.rdzv_threshold < cxip_env.cacheline_size) {
		cxip_env.rdzv_threshold = cxip_env.cacheline_size;
		CXIP_WARN("Increased rdzv_threshold size to: %lu\n",
			  cxip_env.rdzv_threshold);
	}

	fi_param_define(&cxip_prov, "rdzv_get_min", FI_PARAM_SIZE_T,
			"Minimum rendezvous Get payload size.");
	fi_param_get_size_t(&cxip_prov, "rdzv_get_min",
			    &cxip_env.rdzv_get_min);

	fi_param_define(&cxip_prov, "rdzv_eager_size", FI_PARAM_SIZE_T,
			"Eager data size for rendezvous protocol.");
	fi_param_get_size_t(&cxip_prov, "rdzv_eager_size",
			    &cxip_env.rdzv_eager_size);

	if (cxip_env.rdzv_eager_size > cxip_env.rdzv_threshold) {
		cxip_env.rdzv_eager_size = cxip_env.rdzv_threshold;
		CXIP_WARN("Invalid rdzv_eager_size, new size: %lu\n",
			  cxip_env.rdzv_eager_size);
	}

	fi_param_define(&cxip_prov, "oflow_buf_size", FI_PARAM_SIZE_T,
			"Overflow buffer size.");
	fi_param_get_size_t(&cxip_prov, "oflow_buf_size",
			    &cxip_env.oflow_buf_size);

	if (cxip_env.rdzv_threshold > cxip_env.oflow_buf_size) {
		CXIP_WARN("Invalid rdzv_threshold: %lu\n",
			  cxip_env.rdzv_threshold);
		cxip_env.rdzv_threshold = CXIP_RDZV_THRESHOLD;
	}

	if (cxip_env.rdzv_get_min >
	    (cxip_env.oflow_buf_size - cxip_env.rdzv_threshold)) {
		CXIP_WARN("Invalid rdzv_get_min: %lu\n",
			  cxip_env.rdzv_get_min);
		cxip_env.rdzv_get_min = 0;
	}

	/* Allow either FI_CXI_OFLOW_BUF_COUNT or FI_CXI_FLOW_BUF_MIN_POSTED */
	fi_param_define(&cxip_prov, "oflow_buf_count", FI_PARAM_SIZE_T,
			"Overflow buffer count/min posted.");
	fi_param_get_size_t(&cxip_prov, "oflow_buf_count",
			    &cxip_env.oflow_buf_min_posted);
	fi_param_define(&cxip_prov, "oflow_buf_min_posted", FI_PARAM_SIZE_T,
			"Overflow buffer count/min posted.");
	fi_param_get_size_t(&cxip_prov, "oflow_buf_min_posted",
			    &cxip_env.oflow_buf_min_posted);
	cxip_env.oflow_buf_max_cached = cxip_env.oflow_buf_min_posted * 3;

	fi_param_define(&cxip_prov, "oflow_buf_max_cached", FI_PARAM_SIZE_T,
			"Maximum number of overflow buffers cached.");
	fi_param_get_size_t(&cxip_prov, "oflow_buf_max_cached",
			    &cxip_env.oflow_buf_max_cached);
	if (cxip_env.oflow_buf_max_cached && cxip_env.oflow_buf_max_cached <
	    cxip_env.oflow_buf_min_posted) {
		cxip_env.oflow_buf_max_cached = cxip_env.oflow_buf_min_posted;
		CXIP_WARN("Adjusted oflow buffer max cached to %lu\n",
			  cxip_env.oflow_buf_max_cached);
	}

	fi_param_define(&cxip_prov, "safe_devmem_copy_threshold",
			FI_PARAM_SIZE_T,
			"Max memcpy for load/store HMEM access (default %lu).",
			cxip_env.safe_devmem_copy_threshold);
	fi_param_get_size_t(&cxip_prov, "safe_devmem_copy_threshold",
			    &cxip_env.safe_devmem_copy_threshold);

	fi_param_define(&cxip_prov, "optimized_mrs", FI_PARAM_BOOL,
			"Enables optimized memory regions.");
	fi_param_get_bool(&cxip_prov, "optimized_mrs",
			  &cxip_env.optimized_mrs);

	fi_param_define(&cxip_prov, "llring_mode", FI_PARAM_STRING,
			"Set low-latency command queue ring mode.");
	fi_param_get_str(&cxip_prov, "llring_mode", &param_str);

	if (param_str) {
		if (!strcmp(param_str, "always"))
			cxip_env.llring_mode = CXIP_LLRING_ALWAYS;
		else if (!strcmp(param_str, "idle"))
			cxip_env.llring_mode = CXIP_LLRING_IDLE;
		else if (!strcmp(param_str, "never"))
			cxip_env.llring_mode = CXIP_LLRING_NEVER;
		else
			CXIP_WARN("Unrecognized llring_mode: %s\n",
				  param_str);

		param_str = NULL;
	}

	fi_param_define(&cxip_prov, "zbcoll_radix", FI_PARAM_INT,
			"Set radix of the zero-byte barrier tree.");
	fi_param_get_int(&cxip_prov, "zbcoll_radix", &cxip_env.zbcoll_radix);
	if (cxip_env.zbcoll_radix < 2) {
		CXIP_WARN("Invalid zbcoll_radix=%d, reset to 2\n",
			  cxip_env.zbcoll_radix);
		cxip_env.zbcoll_radix = 2;
	}

	fi_param_define(&cxip_prov, "cq_policy", FI_PARAM_STRING,
			"Set Command Queue write-back policy.");
	fi_param_get_str(&cxip_prov, "cq_policy", &param_str);

	if (param_str) {
		if (!strcmp(param_str, "always"))
			cxip_env.cq_policy = CXI_CQ_UPDATE_ALWAYS;
		else if (!strcmp(param_str, "high_empty"))
			cxip_env.cq_policy = CXI_CQ_UPDATE_HIGH_FREQ_EMPTY;
		else if (!strcmp(param_str, "low_empty"))
			cxip_env.cq_policy = CXI_CQ_UPDATE_LOW_FREQ_EMPTY;
		else if (!strcmp(param_str, "low"))
			cxip_env.cq_policy = CXI_CQ_UPDATE_LOW_FREQ;
		else
			CXIP_WARN("Unrecognized cq_policy: %s\n",
				  param_str);

		param_str = NULL;
	}

	fi_param_define(&cxip_prov, "default_vni", FI_PARAM_SIZE_T,
			"Default VNI value used only for service IDs where the VNI is not restricted.");
	fi_param_get_size_t(&cxip_prov, "default_vni", &cxip_env.default_vni);

	fi_param_define(&cxip_prov, "eq_ack_batch_size", FI_PARAM_SIZE_T,
			"Number of EQ events to process before acknowledgement");
	fi_param_get_size_t(&cxip_prov, "eq_ack_batch_size",
			    &cxip_env.eq_ack_batch_size);

	if (!cxip_env.eq_ack_batch_size)
		cxip_env.eq_ack_batch_size = 1;

	fi_param_define(&cxip_prov, "msg_lossless", FI_PARAM_BOOL,
			"Enable/Disable lossless message matching.");
	fi_param_get_bool(&cxip_prov, "msg_lossless", &cxip_env.msg_lossless);

	fi_param_define(&cxip_prov, "req_buf_size", FI_PARAM_SIZE_T,
			"Size of request buffer.");
	fi_param_get_size_t(&cxip_prov, "req_buf_size", &cxip_env.req_buf_size);

	fi_param_define(&cxip_prov, "req_buf_min_posted", FI_PARAM_SIZE_T,
			"Minimum number of request buffer posted.");
	fi_param_get_size_t(&cxip_prov, "req_buf_min_posted",
			    &cxip_env.req_buf_min_posted);

	/* Allow either FI_CXI_REQ_BUF_MAX_CACHED or FI_CXI_REQ_BUF_MAX_COUNT */
	fi_param_define(&cxip_prov, "req_buf_max_count", FI_PARAM_SIZE_T,
			"Maximum number of request buffer cached.");
	fi_param_get_size_t(&cxip_prov, "req_buf_max_count",
			    &cxip_env.req_buf_max_cached);
	fi_param_define(&cxip_prov, "req_buf_max_cached", FI_PARAM_SIZE_T,
			"Maximum number of request buffer cached.");
	fi_param_get_size_t(&cxip_prov, "req_buf_max_cached",
			    &cxip_env.req_buf_max_cached);

	/* Parameters to tailor hybrid hardware to software transitions
	 * that are initiated by software.
	 */
	fi_param_define(&cxip_prov, "hybrid_preemptive", FI_PARAM_BOOL,
			"Enable/Disable low LE preemptive UX transitions.");
	fi_param_get_bool(&cxip_prov, "hybrid_preemptive",
			  &cxip_env.hybrid_preemptive);
	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_preemptive) {
		cxip_env.hybrid_preemptive = false;
		CXIP_WARN("Not in hybrid mode, ignoring preemptive\n");
	}

	fi_param_define(&cxip_prov, "hybrid_recv_preemptive", FI_PARAM_BOOL,
			"Enable/Disable low LE preemptive recv transitions.");
	fi_param_get_bool(&cxip_prov, "hybrid_recv_preemptive",
			  &cxip_env.hybrid_recv_preemptive);

	if (cxip_env.rx_match_mode != CXIP_PTLTE_HYBRID_MODE &&
	    cxip_env.hybrid_recv_preemptive) {
		CXIP_WARN("Not in hybrid mode, ignore LE  recv preemptive\n");
		cxip_env.hybrid_recv_preemptive = 0;
	}

	if (cxip_software_pte_allowed()) {
		min_free = CXIP_REQ_BUF_HEADER_MAX_SIZE +
			cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;

		if (cxip_env.req_buf_size < min_free) {
			cxip_env.req_buf_size = min_free;
			CXIP_WARN("Requested request buffer size to small. Setting to %lu bytes\n",
				  cxip_env.req_buf_size);
		}

		if (cxip_env.req_buf_min_posted < 2) {
			cxip_env.req_buf_min_posted = 2;
			CXIP_WARN("Adjusted request buffer min posted to %lu\n",
				  cxip_env.req_buf_min_posted);
		}

		/* Zero max count is unlimited */
		if (cxip_env.req_buf_max_cached &&
		    cxip_env.req_buf_max_cached < cxip_env.req_buf_min_posted) {
			cxip_env.req_buf_max_cached =
					cxip_env.req_buf_min_posted;
			CXIP_WARN("Adjusted request buffer max cached to %lu\n",
				  cxip_env.req_buf_max_cached);
		}
	}

	fi_param_define(&cxip_prov, "fc_retry_usec_delay", FI_PARAM_INT,
			"Micro-second delay before retrying failed flow-control messages. Default: %d usecs",
			cxip_env.fc_retry_usec_delay);
	fi_param_get_int(&cxip_prov, "fc_retry_usec_delay",
			 &cxip_env.fc_retry_usec_delay);
	if (cxip_env.fc_retry_usec_delay < 0) {
		cxip_env.fc_retry_usec_delay = 0;
		CXIP_WARN("FC retry delay invalid. Setting to %d usecs\n",
			  cxip_env.fc_retry_usec_delay);
	}

	fi_param_define(&cxip_prov, "ctrl_rx_eq_max_size", FI_PARAM_SIZE_T,
			"Control receive event queue max size. Values are aligned up to 4KiB. Default: %lu bytes",
			cxip_env.ctrl_rx_eq_max_size);
	fi_param_get_size_t(&cxip_prov, "ctrl_rx_eq_max_size",
			    &cxip_env.ctrl_rx_eq_max_size);

	fi_param_define(&cxip_prov, "default_cq_size", FI_PARAM_SIZE_T,
			"Default provider CQ size (default: %lu).",
			cxip_env.default_cq_size);
	fi_param_get_size_t(&cxip_prov, "default_cq_size",
			    &cxip_env.default_cq_size);
	if (cxip_env.default_cq_size == 0) {
		cxip_env.default_cq_size = CXIP_CQ_DEF_SZ;
		CXIP_WARN("Default CQ size invalid. Setting to %lu\n",
			  cxip_env.default_cq_size);
	}

	/* FI_CXI_DISABLE_EQ_HUGETLB will deprecate use of
	 * FI_CXI_DISABLE_CQ_HUGETLB, both are allowed for now.
	 */
	fi_param_define(&cxip_prov, "disable_cq_hugetlb", FI_PARAM_BOOL,
			"Disable 2MiB hugetlb allocates for HW event queues (default: %u).",
			cxip_env.disable_eq_hugetlb);
	fi_param_get_bool(&cxip_prov, "disable_cq_hugetlb",
			  &cxip_env.disable_eq_hugetlb);
	fi_param_define(&cxip_prov, "disable_eq_hugetlb", FI_PARAM_BOOL,
			"Disable 2MiB hugetlb allocates for HW event queues (default: %u).",
			cxip_env.disable_eq_hugetlb);
	fi_param_get_bool(&cxip_prov, "disable_eq_hugetlb",
			  &cxip_env.disable_eq_hugetlb);

	fi_param_define(&cxip_prov, "cq_fill_percent", FI_PARAM_SIZE_T,
			"Fill percent of underlying hardware event queue used to determine when completion queue is saturated (default: %lu).",
			cxip_env.cq_fill_percent);
	fi_param_get_size_t(&cxip_prov, "cq_fill_percent",
			    &cxip_env.cq_fill_percent);

	if (cxip_env.cq_fill_percent < 1 ||
	    cxip_env.cq_fill_percent > 100) {
		cxip_env.cq_fill_percent = 50;
		CXIP_WARN("CQ fill percent invalid. Setting to %lu.\n",
			  cxip_env.cq_fill_percent);
	}

	fi_param_define(&cxip_prov, "coll_job_id", FI_PARAM_STRING,
		"Collective job identifier (default %s).",
		cxip_env.coll_job_id);
	fi_param_get_str(&cxip_prov, "coll_job_id",
			  &cxip_env.coll_job_id);

	fi_param_define(&cxip_prov, "coll_step_id", FI_PARAM_STRING,
		"Collective job-step identifier (default %s).",
		cxip_env.coll_step_id);
	fi_param_get_str(&cxip_prov, "coll_step_id",
			  &cxip_env.coll_step_id);

	fi_param_define(&cxip_prov, "coll_fabric_mgr_url", FI_PARAM_STRING,
		"Fabric multicast REST API URL (default %s).",
		cxip_env.coll_fabric_mgr_url);
	fi_param_get_str(&cxip_prov, "coll_fabric_mgr_url",
			  &cxip_env.coll_fabric_mgr_url);
	if (cxip_env.coll_fabric_mgr_url) {
		ret = cxip_env_validate_url(cxip_env.coll_fabric_mgr_url);
		if (ret) {
			CXIP_WARN("Failed to validate fabric multicast URL: name=%s rc=%d. Ignoring URL.\n",
				  cxip_env.coll_fabric_mgr_url, ret);
			cxip_env.coll_fabric_mgr_url = NULL;
		}
	}

	fi_param_define(&cxip_prov, "coll_fabric_mgr_token", FI_PARAM_STRING,
		"Fabric multicast REST API TOKEN (default none).",
		cxip_env.coll_fabric_mgr_token);
	fi_param_get_str(&cxip_prov, "coll_fabric_mgr_token",
			  &cxip_env.coll_fabric_mgr_token);

	fi_param_define(&cxip_prov, "coll_use_dma_put", FI_PARAM_BOOL,
		"Use DMA Put for collectives (default: %d).",
		cxip_env.coll_use_dma_put);
	fi_param_get_bool(&cxip_prov, "coll_use_dma_put",
			  &cxip_env.coll_use_dma_put);

	fi_param_define(&cxip_prov, "coll_timeout_usec", FI_PARAM_SIZE_T,
		"Nominal estimated compute cycle (usec) (default %d).",
		cxip_env.coll_timeout_usec);
	fi_param_get_size_t(&cxip_prov, "coll_timeout_usec",
			    &cxip_env.coll_timeout_usec);

	fi_param_define(&cxip_prov, "coll_use_repsum", FI_PARAM_BOOL,
		"Use reproducible sum for collective double sum (default %d)",
		cxip_env.coll_use_repsum);

	fi_param_define(&cxip_prov, "default_tx_size", FI_PARAM_SIZE_T,
			"Default provider tx_attr.size (default: %lu).",
			cxip_env.default_tx_size);
	fi_param_get_size_t(&cxip_prov, "default_tx_size",
			    &cxip_env.default_tx_size);
	if (cxip_env.default_tx_size < 16 ||
	    cxip_env.default_tx_size > CXIP_MAX_TX_SIZE) {
		cxip_env.default_tx_size = CXIP_DEFAULT_TX_SIZE;
		CXIP_WARN("Default TX size invalid. Setting to %lu\n",
			  cxip_env.default_tx_size);
	}

	fi_param_define(&cxip_prov, "disable_hmem_dev_register", FI_PARAM_BOOL,
			"Disable registering HMEM device buffer for load/store access (default: %u).",
			cxip_env.disable_hmem_dev_register);
	fi_param_get_bool(&cxip_prov, "disable_hmem_dev_register",
			  &cxip_env.disable_hmem_dev_register);
}

/*
 * CXI_INI - Provider constructor.
 */
CXI_INI
{
	cxip_env_init();

	cxip_curl_init();

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

	cxip_curl_fini();
}

static void cxip_alter_caps(struct fi_info *info, const struct fi_info *hints)
{
	/* If FI_COLLECTIVE explicitly requested then must enable
	 * FI_MSG for send and receive if not already enabled.
	 */
	if (hints && hints->caps && (hints->caps & FI_COLLECTIVE)) {
		if (!(info->caps & (FI_MSG | FI_TAGGED))) {
			info->caps |= FI_MSG | FI_SEND | FI_RECV;
			info->tx_attr->caps |= FI_MSG | FI_SEND;
			info->rx_attr->caps |= FI_MSG | FI_RECV;
		}
	}
}

static void cxip_alter_tx_attr(struct fi_tx_attr *attr,
			       const struct fi_tx_attr *hints,
			       uint64_t info_caps)
{
	if (!hints || hints->size == 0)
		attr->size = cxip_env.default_tx_size;
}

static void cxip_alter_info(struct fi_info *info, const struct fi_info *hints,
			    uint32_t api_version)
{
	for (; info; info = info->next) {
		cxip_alter_caps(info, hints);
		cxip_alter_tx_attr(info->tx_attr, hints ? hints->tx_attr : NULL,
				   info->caps);

		/* Remove secondary capabilities that impact performance if
		 * hints are not specified. They must be explicitly requested.
		 */
		if (!hints) {
			info->caps &= ~(FI_SOURCE | FI_SOURCE_ERR);
			info->rx_attr->caps &= ~(FI_SOURCE | FI_SOURCE_ERR);
		}
	}
}

static int cxip_alter_auth_key_align_domain_ep(struct fi_info **info)
{
	struct fi_info *fi_ptr;

	/* CXI provider requires the endpoint to have the same service ID as the
	 * domain. Account for edge case where users only set endpoint auth_key
	 * and leave domain auth_key as NULL by duplicating the endpoint
	 * auth_key to the domain.
	 */
	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		if (!fi_ptr->domain_attr->auth_key &&
		    fi_ptr->ep_attr->auth_key) {
			fi_ptr->domain_attr->auth_key =
				mem_dup(fi_ptr->ep_attr->auth_key,
					fi_ptr->ep_attr->auth_key_size);
			if (!fi_ptr->domain_attr->auth_key)
				return -FI_ENOMEM;

			fi_ptr->domain_attr->auth_key_size =
				fi_ptr->ep_attr->auth_key_size;
		}
	}

	return FI_SUCCESS;
}

static void cxip_alter_auth_key_scrub_auth_key_size(struct fi_info **info)
{
	struct fi_info *fi_ptr;

	/* Zero the auth_key_size for any NULL auth_key. */
	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		if (!fi_ptr->domain_attr->auth_key)
			fi_ptr->domain_attr->auth_key_size = 0;

		if (!fi_ptr->ep_attr->auth_key)
			fi_ptr->ep_attr->auth_key_size = 0;
	}
}

static int cxip_alter_auth_key_validate(struct fi_info **info)
{
	struct fi_info *fi_ptr;
	struct fi_info *fi_ptr_tmp;
	struct fi_info *fi_prev_ptr;
	int ret;

	/* Core auth_key checks only verify auth_key_size. This check verifies
	 * that the user provided auth_key is valid.
	 */
	fi_ptr = *info;
	*info = NULL;
	fi_prev_ptr = NULL;

	while (fi_ptr) {
		ret = cxip_check_auth_key_info(fi_ptr);
		if (ret) {
			/* discard entry */
			if (fi_prev_ptr)
				fi_prev_ptr->next = fi_ptr->next;

			fi_ptr_tmp = fi_ptr;
			fi_ptr = fi_ptr->next;
			fi_ptr_tmp->next = NULL;
			fi_freeinfo(fi_ptr_tmp);
			continue;
		}

		if (*info == NULL)
				*info = fi_ptr;

		fi_prev_ptr = fi_ptr;
		fi_ptr = fi_ptr->next;
	}

	return FI_SUCCESS;
}

static int cxip_gen_auth_key_ss_env_get_vni(void)
{
	char *vni_str;
	char *vni_str_dup;
	char *token;
	int vni = -FI_EINVAL;

	vni_str = getenv("SLINGSHOT_VNIS");
	if (!vni_str) {
		CXIP_INFO("SLINGSHOT_VNIS not found\n");
		return -FI_ENOSYS;
	}

	vni_str_dup = strdup(vni_str);
	if (!vni_str_dup)
		return -FI_ENOMEM;

	/* Index/token zero is the per job-step VNI. Only use this value. Index
	 * one is the inter-job-step VNI. Ignore this one.
	 */
	token = strtok(vni_str_dup, ",");
	if (token)
		vni = (uint16_t)atoi(token);
	else
		CXIP_WARN("VNI not found in SLINGSHOT_VNIS: %s\n", vni_str);

	free(vni_str_dup);

	return vni;
}

static int cxip_gen_auth_key_ss_env_get_svc_id(struct fi_info *info)
{
	char *svc_id_str;
	char *dev_str;
	char *svc_id_str_dup;
	char *dev_str_dup;
	int device_index;
	char *token;
	bool found;
	int svc_id;

	if (!info->domain_attr->name) {
		CXIP_WARN("Domain attr name is NULL\n");
		return -FI_EINVAL;
	}

	svc_id_str = getenv("SLINGSHOT_SVC_IDS");
	if (!svc_id_str) {
		CXIP_INFO("SLINGSHOT_SVC_IDS not found\n");
		return -FI_ENOSYS;
	}

	dev_str = getenv("SLINGSHOT_DEVICES");
	if (!dev_str) {
		CXIP_INFO("SLINGSHOT_DEVICES not found\n");
		return -FI_ENOSYS;
	}

	dev_str_dup = strdup(dev_str);
	if (!dev_str_dup)
		return -FI_ENOMEM;

	found = false;
	device_index = 0;
	token = strtok(dev_str_dup, ",");
	while (token != NULL) {
		if (strcmp(token, info->domain_attr->name) == 0) {
			found = true;
			break;
		}

		device_index++;
		token = strtok(NULL, ",");
	}

	free(dev_str_dup);

	if (!found) {
		CXIP_WARN("Failed to find %s in SLINGSHOT_DEVICES: %s\n",
			  info->domain_attr->name, dev_str);
		return -FI_ENOSYS;
	}

	svc_id_str_dup = strdup(svc_id_str);
	if (!svc_id_str_dup)
		return -FI_ENOMEM;

	found = false;
	token = strtok(svc_id_str_dup, ",");
	while (token != NULL) {
		if (device_index == 0) {
			svc_id = atoi(token);
			found = true;
			break;
		}

		device_index--;
		token = strtok(NULL, ",");
	}

	free(svc_id_str_dup);

	if (!found) {
		CXIP_WARN("Failed to find service ID in SLINGSHOT_SVC_IDS: %s\n",
			  svc_id_str);
		return -FI_EINVAL;
	}

	return svc_id;
}

static int cxip_gen_auth_key_ss_env(struct fi_info *info,
				    struct cxi_auth_key *key)
{
	int ret;
	uint16_t vni;

	ret = cxip_gen_auth_key_ss_env_get_vni();
	if (ret < 0)
		return ret;

	vni = ret;

	ret = cxip_gen_auth_key_ss_env_get_svc_id(info);
	if (ret < 0)
		return ret;

	key->vni = vni;
	key->svc_id = ret;

	CXIP_INFO("Generated auth key (%u:%u) for %s\n", ret, vni,
		  info->domain_attr->name);

	return FI_SUCCESS;
}

static int cxip_gen_auth_key_best_svc_id(struct fi_info *info,
					 struct cxi_auth_key *key)
{
	int ret;
	struct cxi_auth_key auth_key;
	struct cxip_addr *src_addr;
	struct cxip_if *iface;
	struct cxil_svc_list *svc_list;
	uid_t uid;
	gid_t gid;
	int i;
	int j;
	struct cxi_svc_desc *desc;
	int found_uid;
	int found_gid;
	int found_unrestricted;

	uid = geteuid();
	gid = getegid();

	src_addr = (struct cxip_addr *)info->src_addr;
	if (!src_addr) {
		CXIP_WARN("NULL src_addr in fi_info\n");
		return -FI_EINVAL;
	}

	ret = cxip_get_if(src_addr->nic, &iface);
	if (ret) {
		CXIP_WARN("cxip_get_if with NIC %#x failed: %d:%s\n",
			  src_addr->nic, ret, fi_strerror(-ret));
		return ret;
	}

	ret = cxil_get_svc_list(iface->dev, &svc_list);
	if (ret) {
		CXIP_WARN("cxil_get_svc_list failed: %d:%s\n", ret,
			  strerror(-ret));
		cxip_put_if(iface);
		return ret;
	}

	/* Find the service indexes which can be used by this process. These are
	 * services which are unrestricted, have a matching UID, or have a
	 * matching GID. If there are multiple service IDs which could match
	 * unrestricted, UID, and GID, only the first one found is selected.
	 */
	found_uid = -1;
	found_gid = -1;
	found_unrestricted = -1;

	for (i = 0; i < svc_list->count; i++) {
		desc = svc_list->descs + i;

		if (!desc->enable || desc->is_system_svc)
			continue;

		if (!desc->restricted_members) {
			if (found_unrestricted == -1)
				found_unrestricted = i;
			continue;
		}

		for (j = 0; j < CXI_SVC_MAX_MEMBERS; j++) {
			if (desc->members[j].type == CXI_SVC_MEMBER_UID &&
			    desc->members[j].svc_member.uid == uid &&
			    found_uid == -1)
				found_uid = i;
			else if (desc->members[j].type == CXI_SVC_MEMBER_GID &&
				 desc->members[j].svc_member.gid == gid &&
				 found_gid == -1)
				found_gid = i;
		}
	}

	/* Prioritized list for matching service ID. */
	if (found_uid != -1)
		i = found_uid;
	else if (found_gid != -1) {
		i = found_gid;
	} else if (found_unrestricted != -1) {
		i = found_unrestricted;
	} else {
		cxil_free_svc_list(svc_list);
		cxip_put_if(iface);
		return -FI_ENOSYS;
	}

	/* Generate auth_key using matched service ID. */
	desc = svc_list->descs + i;

	if (desc->restricted_vnis) {
		if (desc->num_vld_vnis == 0) {
			CXIP_WARN("No valid VNIs for %s service ID %u\n",
				  info->domain_attr->name, i);

			cxil_free_svc_list(svc_list);
			cxip_put_if(iface);

			return -FI_EINVAL;
		}

		auth_key.vni = (uint16_t)desc->vnis[0];
	} else {
		auth_key.vni = (uint16_t)cxip_env.default_vni;
	}

	auth_key.svc_id = desc->svc_id;

	cxil_free_svc_list(svc_list);
	cxip_put_if(iface);

	*key = auth_key;

	CXIP_INFO("Generated auth key (%u:%u) for %s\n", auth_key.svc_id,
		  auth_key.vni, info->domain_attr->name);

	return FI_SUCCESS;
}

int cxip_gen_auth_key(struct fi_info *info, struct cxi_auth_key *key)
{
	int ret;

	memset(key, 0, sizeof(*key));

	if (info->domain_attr->auth_key) {
		CXIP_WARN("Domain auth_key not NULL\n");
		return -FI_EINVAL;
	}

	ret = cxip_gen_auth_key_ss_env(info, key);
	if (ret == FI_SUCCESS)
		return FI_SUCCESS;

	return cxip_gen_auth_key_best_svc_id(info, key);
}

static int cxip_alter_auth_key(struct fi_info **info)
{
	int ret;
	struct cxi_auth_key auth_key;
	struct fi_info *fi_ptr;

	ret = cxip_alter_auth_key_align_domain_ep(info);
	if (ret)
		return ret;

	cxip_alter_auth_key_scrub_auth_key_size(info);

	/* For any domain with a NULL auth_key, attempt to generate an auth_key.
	 */
	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		if (fi_ptr->domain_attr->auth_key)
			continue;

		ret = cxip_gen_auth_key(fi_ptr, &auth_key);
		if (ret == -FI_ENOSYS)
			continue;
		else if (ret)
			return ret;

		fi_ptr->domain_attr->auth_key =
			mem_dup(&auth_key, sizeof(auth_key));
		if (!fi_ptr->domain_attr->auth_key)
			return -FI_ENOMEM;

		fi_ptr->domain_attr->auth_key_size = sizeof(auth_key);

		CXIP_INFO("Assigned auth key (%u:%u) to %s\n", auth_key.svc_id,
			  auth_key.vni, fi_ptr->domain_attr->name);
	}

	return cxip_alter_auth_key_validate(info);
}

static int cxip_validate_iface_auth_key(struct cxip_if *iface,
					struct cxi_auth_key *auth_key)
{
	struct cxi_svc_desc svc_desc;
	int ret;
	int i;
	bool vni_found = false;

	if (!auth_key)
		return FI_SUCCESS;

	ret = cxil_get_svc(iface->dev, auth_key->svc_id, &svc_desc);
	if (ret) {
		CXIP_WARN("cxil_get_svc with %s and svc_id %d failed: %d:%s\n",
			  iface->dev->info.device_name, auth_key->svc_id, ret,
			  strerror(-ret));
		return -FI_EINVAL;
	}

	if (svc_desc.restricted_vnis) {
		for (i = 0; i < svc_desc.num_vld_vnis; i++) {
			if (auth_key->vni == svc_desc.vnis[i]) {
				vni_found = true;
				break;
			}
		}

		if (!vni_found) {
			CXIP_WARN("Invalidate VNI %d for %s and svc_id %d\n",
				  auth_key->vni, iface->dev->info.device_name,
				  auth_key->svc_id);
			return -FI_EINVAL;
		}
	}

	return FI_SUCCESS;
}

int cxip_check_auth_key_info(struct fi_info *info)
{
	struct cxip_addr *src_addr;
	struct cxip_if *iface;
	int ret;

	src_addr = (struct cxip_addr *)info->src_addr;
	if (!src_addr) {
		CXIP_WARN("NULL src_addr in fi_info\n");
		return -FI_EINVAL;
	}

	ret = cxip_get_if(src_addr->nic, &iface);
	if (ret) {
		CXIP_WARN("cxip_get_if with NIC %#x failed: %d:%s\n",
				src_addr->nic, ret, fi_strerror(-ret));
		return ret;
	}

	if (info->domain_attr) {
		ret = cxip_validate_iface_auth_key(iface,
						   (struct cxi_auth_key *)info->domain_attr->auth_key);
		if (ret) {
			CXIP_WARN("Invalid domain auth_key\n");
			goto err_put_if;
		}
	}

	if (info->ep_attr) {
		ret = cxip_validate_iface_auth_key(iface,
						   (struct cxi_auth_key *)info->ep_attr->auth_key);
		if (ret) {
			CXIP_WARN("Invalid endpoint auth_key\n");
			goto err_put_if;
		}
	}

	cxip_put_if(iface);

	return FI_SUCCESS;

err_put_if:
	cxip_put_if(iface);

	return ret;
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
	struct fi_info *fi_ptr;
	struct fi_info *fi_ptr_tmp;
	struct fi_info *fi_prev_ptr;
	struct ether_addr *mac;
	uint32_t scan_nic = 0;
	uint32_t scan_pid = 0;
	struct cxip_addr *addr;
	struct cxip_if *iface;
	bool copy_dest = NULL;
	struct fi_info *temp_hints = NULL;

	if (flags & FI_SOURCE) {
		if (!node && !service) {
			CXIP_WARN("FI_SOURCE set, but no node or service\n");
			return -FI_EINVAL;
		}
	}

	if (node) {
		iface = cxip_if_lookup_name(node);
		if (iface) {
			scan_nic = iface->info->nic_addr;
		} else if ((mac = ether_aton(node))) {
			scan_nic = cxip_mac_to_nic(mac);
		} else if (sscanf(node, "%i", &scan_nic) != 1) {
			CXIP_WARN("Invalid node: %s\n", node);
			return -FI_EINVAL;
		}

		CXIP_DBG("Node NIC: %#x\n", scan_nic);
	}

	if (service) {
		if (sscanf(service, "%i", &scan_pid) != 1) {
			CXIP_WARN("Invalid service: %s\n", service);
			return -FI_EINVAL;
		}

		if (scan_pid >= C_PID_ANY) {
			CXIP_WARN("Service out of range [0-%d): %u\n",
				  C_PID_ANY, scan_pid);
			return -FI_EINVAL;
		}

		CXIP_DBG("Service PID: %u\n", scan_pid);
	}

	/* Previously when remote access ODP was not enabled, the provider
	 * did not indicate it required FI_MR_ALLOCATED. To correct this
	 * while not breaking applications, when ODP is NOT enabled add
	 * FI_MR_ALLOCATED to the hints. Note that if the client sets
	 * FI_MR_UNSPEC in hints the correct provider required mode
	 * bits will be returned that the applicaiton must support.
	 *
	 * TODO: When ODP is enabled by default, this should be removed
	 * and applications should use hints to pick the desired mode.
	 */
	if (!cxip_env.odp && hints && hints->domain_attr &&
	    hints->domain_attr->mr_mode  == FI_MR_ENDPOINT) {
		temp_hints = fi_dupinfo(hints);
		if (!temp_hints)
			return -FI_ENOMEM;

		temp_hints->domain_attr->mr_mode |= FI_MR_ALLOCATED;

		CXIP_INFO("FI_MR_ALLOCATED added to hints MR mode\n");
	}

	/* Find all matching domains, ignoring addresses. */
	ret = util_getinfo(&cxip_util_prov, version, NULL, NULL, 0,
			   temp_hints ? temp_hints : hints,
			   info);
	if (temp_hints)
		fi_freeinfo(temp_hints);

	if (ret)
		return ret;

	/* Remove any info that did match based on mr_mode requirements.
	 * Note that mr_mode FI_MR_ENDPOINT is only required if target
	 * RMA/ATOMIC access is required.
	 */
	if (hints) {
		fi_ptr = *info;
		*info = NULL;
		fi_prev_ptr = NULL;

		while (fi_ptr) {
			if (fi_ptr->caps & (FI_ATOMIC | FI_RMA) &&
			    !fi_ptr->domain_attr->mr_mode) {
				/* discard entry */
				if (fi_prev_ptr)
					fi_prev_ptr->next = fi_ptr->next;

				fi_ptr_tmp = fi_ptr;
				fi_ptr = fi_ptr->next;
				fi_ptr_tmp->next = NULL;
				fi_freeinfo(fi_ptr_tmp);
				continue;
			}

			/* Keep the matching info */
			if (*info == NULL)
				*info = fi_ptr;

			fi_prev_ptr = fi_ptr;
			fi_ptr = fi_ptr->next;
		}
	}

	/* Search for a specific OFI Domain by node string. */
	if (flags & FI_SOURCE && node) {
		iface = cxip_if_lookup_addr(scan_nic);
		if (!iface) {
			/* This shouldn't fail. */
			ret = -FI_EINVAL;
			goto freeinfo;
		}

		fi_ptr = *info;
		*info = NULL;
		fi_prev_ptr = NULL;

		while (fi_ptr) {
			if (strcmp(fi_ptr->domain_attr->name,
				   iface->info->device_name)) {
				/* discard entry */
				if (fi_prev_ptr)
					fi_prev_ptr->next = fi_ptr->next;

				fi_ptr_tmp = fi_ptr;
				fi_ptr = fi_ptr->next;
				fi_ptr_tmp->next = NULL;
				fi_freeinfo(fi_ptr_tmp);
				continue;
			}

			/* Keep the matching info */
			if (*info == NULL)
				*info = fi_ptr;

			fi_prev_ptr = fi_ptr;
			fi_ptr = fi_ptr->next;
		}
	}

	/* Search for a specific OFI Domain by name. The CXI Domain name
	 * matches the NIC device file name (cxi[0-9]).
	 */
	if (hints && hints->domain_attr && hints->domain_attr->name) {
		fi_ptr = *info;
		*info = NULL;
		fi_prev_ptr = NULL;

		while (fi_ptr) {
			if (strcmp(fi_ptr->domain_attr->name,
				   hints->domain_attr->name)) {
				/* discard entry */
				if (fi_prev_ptr)
					fi_prev_ptr->next = fi_ptr->next;

				fi_ptr_tmp = fi_ptr;
				fi_ptr = fi_ptr->next;
				fi_ptr_tmp->next = NULL;
				fi_freeinfo(fi_ptr_tmp);
				continue;
			}

			/* Keep the matching info */
			if (*info == NULL)
				*info = fi_ptr;

			fi_prev_ptr = fi_ptr;
			fi_ptr = fi_ptr->next;
		}
	}

	cxip_alter_info(*info, hints, version);

	/* Check if any infos remain. */
	if (!*info)
		return FI_SUCCESS;

	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		if (flags & FI_SOURCE) {
			/* Set client-assigned PID value in source address. */
			if (service) {
				addr = (struct cxip_addr *)fi_ptr->src_addr;
				addr->pid = scan_pid;
			}

			copy_dest = (hints && hints->dest_addr);
		} else {
			if (node) {
				struct cxip_addr addr = {};

				addr.nic = scan_nic;
				addr.pid = scan_pid;

				fi_ptr->dest_addr = mem_dup(&addr,
							    sizeof(addr));
				if (!fi_ptr->dest_addr) {
					ret = -FI_ENOMEM;
					goto freeinfo;
				}
				fi_ptr->dest_addrlen = sizeof(addr);
			} else {
				copy_dest = (hints && hints->dest_addr);
			}

			if (hints && hints->src_addr) {
				fi_ptr->src_addr = mem_dup(hints->src_addr,
							   hints->src_addrlen);
				if (!fi_ptr->src_addr) {
					ret = -FI_ENOMEM;
					goto freeinfo;
				}
				fi_ptr->src_addrlen = hints->src_addrlen;
				fi_ptr->addr_format = hints->addr_format;
			}
		}

		if (copy_dest) {
			fi_ptr->dest_addr = mem_dup(hints->dest_addr,
						    hints->dest_addrlen);
			if (!fi_ptr->dest_addr) {
				ret = -FI_ENOMEM;
				goto freeinfo;
			}
			fi_ptr->dest_addrlen = hints->dest_addrlen;
			fi_ptr->addr_format = hints->addr_format;
		}
	}

	ret = cxip_alter_auth_key(info);
	if (ret)
		goto freeinfo;

	/* Nothing left to do if hints weren't provided. */
	if (!hints)
		return FI_SUCCESS;

	/* util_getinfo() returns a list of fi_info that match the MR mode
	 * for each nic. They are listed in provider preference order.
	 * Since hints were provided, keep only the most preferred fi_info for
	 * any given domain/interface using the same address format. We
	 * always keep the first one.
	 */
	fi_ptr = *info;
	fi_prev_ptr = NULL;

	while (fi_ptr) {
		if (fi_prev_ptr &&
		    !strcmp(fi_ptr->domain_attr->name,
			    fi_prev_ptr->domain_attr->name) &&
		    fi_ptr->addr_format == fi_prev_ptr->addr_format) {
			/* discard entry */
			fi_prev_ptr->next = fi_ptr->next;
			fi_ptr_tmp = fi_ptr;
			fi_ptr = fi_ptr->next;

			fi_ptr_tmp->next = NULL;
			fi_freeinfo(fi_ptr_tmp);
			continue;
		}

		/* Keep the preferred info for this domain */
		fi_prev_ptr = fi_ptr;
		fi_ptr = fi_ptr->next;
	}

	/* util_getinfo() returns a list of fi_info for each matching OFI
	 * Domain (physical CXI interface).
	 *
	 * Perform fixups:
	 * -Use input ordering requirements.
	 * -Remove unrequested secondary caps that impact performance.
	 */
	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		/* Ordering requirements prevent the use of restricted packets.
		 * If hints exist, copy msg_order settings directly.
		 */
		fi_ptr->tx_attr->msg_order = hints->tx_attr->msg_order;

		/* Requesting FI_RMA_EVENT prevents the use of restricted
		 * packets. Do not set FI_RMA_EVENT unless explicitly
		 * requested.
		 */
		if (hints->caps && !(hints->caps & FI_RMA_EVENT)) {
			fi_ptr->caps &= ~FI_RMA_EVENT;
			fi_ptr->rx_attr->caps &= ~FI_RMA_EVENT;
		}

		/* FI_SOURCE_ERR requires that FI_SOURCE be set, it is
		 * an error if requested but can not be honored.
		 */
		if (hints->caps & FI_SOURCE_ERR && !(hints->caps & FI_SOURCE)) {
			ret = -FI_ENODATA;
			goto freeinfo;
		}

		/* Requesting FI_SOURCE adds overhead to a receive operation.
		 * Do not set FI_SOURCE unless explicitly requested.
		 */
		if (!(hints->caps & FI_SOURCE)) {
			fi_ptr->caps &= ~FI_SOURCE;
			fi_ptr->rx_attr->caps &= ~FI_SOURCE;
		}

		/* Requesting FI_SOURCE_ERR adds additional overhead to receive
		 * operations beyond FI_SOURCE, do not set if not explicitly
		 * asked.
		 */
		if (!(hints->caps & FI_SOURCE_ERR)) {
			fi_ptr->caps &= ~FI_SOURCE_ERR;
			fi_ptr->rx_attr->caps &= ~FI_SOURCE_ERR;
		}

		/* Requesting FI_FENCE prevents the use PCIe RO for RMA. Do not
		 * set FI_FENCE unless explicitly requested.
		 */
		if (hints->caps && !(hints->caps & FI_FENCE)) {
			fi_ptr->caps &= ~FI_FENCE;
			fi_ptr->tx_attr->caps &= ~FI_FENCE;
		}

		/* Requesting FI_HMEM requires use of device memory safe
		 * copy routines. Do not set FI_HMEM unless requested or
		 * all supported provider capabilities are requested.
		 */
		if (hints->caps && !(hints->caps & FI_HMEM)) {
			fi_ptr->caps &= ~FI_HMEM;
			fi_ptr->tx_attr->caps &= ~FI_HMEM;
			fi_ptr->rx_attr->caps &= ~FI_HMEM;
		}
	}

	return FI_SUCCESS;

freeinfo:
	fi_freeinfo(*info);

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
