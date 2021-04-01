/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

/* CXI fabric discovery implementation. */

#include "ofi_prov.h"
#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_FABRIC, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_FABRIC, __VA_ARGS__)

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
	.mr_mode = FI_MR_ENDPOINT,
	.mr_key_size = sizeof(uint16_t),
	.cq_data_size = 8,
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
	.size = 256,  /* 64k / 256b */
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
	struct cxip_addr addr = {};

	fi = fi_dupinfo(&cxip_info);
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
	fi->nic->link_attr->speed = nic_if->speed * 1000;
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
	int ret;

	slist_foreach(&cxip_if_list, entry, prev) {
		nic_if = container_of(entry, struct cxip_if, if_entry);
		ret = cxip_info_alloc(nic_if, &fi);
		if (ret != FI_SUCCESS) {
			fi_freeinfo((void *)cxip_util_prov.info);
			break;
		}
		CXIP_DBG("%s info created\n", nic_if->info->device_name);

		*fi_list = fi;
		fi_list = &(fi->next);
	}

	return ret;
}

struct cxip_environment cxip_env = {
	.odp = false,
	.ats = true,
	.ats_mlock_mode = CXIP_ATS_MLOCK_ALL,
	.rdzv_offload = true,
	.fc_recovery = true,
	.rdzv_threshold = CXIP_RDZV_THRESHOLD,
	.rdzv_get_min = 2049, /* Avoid single packet Gets */
	.rdzv_eager_size = CXIP_RDZV_THRESHOLD,
	.oflow_buf_size = CXIP_OFLOW_BUF_SIZE,
	.oflow_buf_count = CXIP_OFLOW_BUF_COUNT,
	.optimized_mrs = true,
	.llring_mode = CXIP_LLRING_IDLE,
	.cq_policy = CXI_CQ_UPDATE_LOW_FREQ_EMPTY,
	.default_vni = 10,
	.eq_ack_batch_size = 32,
	.req_buf_size = CXIP_REQ_BUF_SIZE,
	.req_buf_count = CXIP_REQ_BUF_COUNT,
	.msg_offload = 1,
	.fc_retry_usec_delay = 1000,
	.ctrl_rx_eq_max_size = 67108864,
	.default_cq_size = CXIP_CQ_DEF_SZ,
};

static void cxip_env_init(void)
{
	char *param_str = NULL;
	size_t min_free;

	fi_param_define(&cxip_prov, "odp", FI_PARAM_BOOL,
			"Enables on-demand paging.");
	fi_param_get_bool(&cxip_prov, "odp", &cxip_env.odp);

	fi_param_define(&cxip_prov, "ats", FI_PARAM_BOOL,
			"Enables PCIe ATS.");
	fi_param_get_bool(&cxip_prov, "ats", &cxip_env.ats);

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
	}

	fi_param_define(&cxip_prov, "rdzv_offload", FI_PARAM_BOOL,
			"Enables offloaded rendezvous messaging protocol.");
	fi_param_get_bool(&cxip_prov, "rdzv_offload", &cxip_env.rdzv_offload);

	fi_param_define(&cxip_prov, "fc_recovery", FI_PARAM_BOOL,
			"Enables message flow control recovery.");
	fi_param_get_bool(&cxip_prov, "fc_recovery", &cxip_env.fc_recovery);

	fi_param_define(&cxip_prov, "rdzv_threshold", FI_PARAM_SIZE_T,
			"Message size threshold for rendezvous protocol.");
	fi_param_get_size_t(&cxip_prov, "rdzv_threshold",
			    &cxip_env.rdzv_threshold);

	fi_param_define(&cxip_prov, "rdzv_get_min", FI_PARAM_SIZE_T,
			"Minimum rendezvous Get payload size.");
	fi_param_get_size_t(&cxip_prov, "rdzv_get_min",
			    &cxip_env.rdzv_get_min);

	fi_param_define(&cxip_prov, "rdzv_eager_size", FI_PARAM_SIZE_T,
			"Eager data size for rendezvous protocol.");
	fi_param_get_size_t(&cxip_prov, "rdzv_eager_size",
			    &cxip_env.rdzv_eager_size);

	if (cxip_env.rdzv_eager_size > cxip_env.rdzv_threshold) {
		CXIP_WARN("Invalid rdzv_eager_size: %lu\n",
			  cxip_env.rdzv_eager_size);
		cxip_env.rdzv_eager_size = cxip_env.rdzv_threshold;
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

	fi_param_define(&cxip_prov, "oflow_buf_count", FI_PARAM_SIZE_T,
			"Overflow buffer count.");
	fi_param_get_size_t(&cxip_prov, "oflow_buf_count",
			    &cxip_env.oflow_buf_count);

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
	}

	fi_param_define(&cxip_prov, "default_vni", FI_PARAM_SIZE_T,
			"Default VNI value.");
	fi_param_get_size_t(&cxip_prov, "default_vni",
			    &cxip_env.default_vni);

	fi_param_define(&cxip_prov, "eq_ack_batch_size", FI_PARAM_SIZE_T,
			"Number of EQ events to process before acknowledgement");
	fi_param_get_size_t(&cxip_prov, "eq_ack_batch_size",
			    &cxip_env.eq_ack_batch_size);

	if (!cxip_env.eq_ack_batch_size)
		cxip_env.eq_ack_batch_size = 1;

	fi_param_define(&cxip_prov, "msg_offload", FI_PARAM_BOOL,
			"Enable or disable hardware message matching.");
	fi_param_get_bool(&cxip_prov, "msg_offload", &cxip_env.msg_offload);

	fi_param_define(&cxip_prov, "req_buf_size", FI_PARAM_SIZE_T,
			"Size of request buffer.");
	fi_param_get_size_t(&cxip_prov, "req_buf_size", &cxip_env.req_buf_size);

	fi_param_define(&cxip_prov, "req_buf_count", FI_PARAM_SIZE_T,
			"Number of request buffer.");
	fi_param_get_size_t(&cxip_prov, "req_buf_count",
			    &cxip_env.req_buf_count);

	if (cxip_env.msg_offload) {
		min_free = sizeof(struct c_port_fab_hdr) +
			sizeof(struct c_port_unrestricted_hdr) +
			cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;

		if (cxip_env.req_buf_size < min_free) {
			cxip_env.req_buf_size = min_free;
			CXIP_WARN("Requested request buffer size to small. Setting to %lu bytes\n",
				  cxip_env.req_buf_size);
		}

		if (cxip_env.req_buf_count < 2) {
			cxip_env.req_buf_count = 2;
			CXIP_WARN("Requested request buffer count to small. Setting to %lu\n",
				  cxip_env.req_buf_count);
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
			"Default provider completion queue size (default: %lu).",
			cxip_env.default_cq_size);
	fi_param_get_size_t(&cxip_prov, "default_cq_size",
			    &cxip_env.default_cq_size);
	if (cxip_env.default_cq_size == 0) {
		cxip_env.default_cq_size = CXIP_CQ_DEF_SZ;
		CXIP_WARN("Default CQ size invalid. Setting to %lu\n",
			  cxip_env.default_cq_size);
	}
}

/*
 * CXI_INI - Provider constructor.
 */
CXI_INI
{
	cxip_if_init();

	cxip_env_init();

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
	struct fi_info *fi_ptr;
	struct fi_info *fi_ptr_tmp;
	struct ether_addr *mac;
	uint32_t scan_nic = 0;
	uint32_t scan_pid = 0;
	struct cxip_addr *addr;
	struct cxip_if *iface;
	bool copy_dest = NULL;

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

	/* Find all matching domains, ignoring addresses. */
	ret = util_getinfo(&cxip_util_prov, version, NULL, NULL, 0, hints,
			   info);
	if (ret)
		return ret;

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
		while (fi_ptr) {
			if (strcmp(fi_ptr->domain_attr->name,
				   iface->info->device_name)) {
				/* discard entry */
				fi_ptr_tmp = fi_ptr;
				fi_ptr = fi_ptr->next;

				fi_ptr_tmp->next = NULL;
				fi_freeinfo(fi_ptr_tmp);
				continue;
			}

			/* Keep the matching info */
			*info = fi_ptr;

			/* free the rest */
			fi_freeinfo((*info)->next);
			(*info)->next = NULL;
			break;
		}
	}

	/* Search for a specific OFI Domain by name. The CXI Domain name
	 * matches the NIC device file name (cxi[0-9]).
	 */
	if (hints && hints->domain_attr && hints->domain_attr->name) {
		fi_ptr = *info;
		*info = NULL;
		while (fi_ptr) {
			if (strcmp(fi_ptr->domain_attr->name,
				   hints->domain_attr->name)) {
				/* discard entry */
				fi_ptr_tmp = fi_ptr;
				fi_ptr = fi_ptr->next;

				fi_ptr_tmp->next = NULL;
				fi_freeinfo(fi_ptr_tmp);
				continue;
			}

			/* Keep the matching info */
			*info = fi_ptr;

			/* free the rest */
			fi_freeinfo((*info)->next);
			(*info)->next = NULL;
			break;
		}
	}


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

	/* TODO: auth_key can't be set in hints yet. Common code needs to be
	 * updated to support that. Set auth_key in info before creating Domain
	 * and/or EPs.
	 */
	for (fi_ptr = *info; fi_ptr; fi_ptr = fi_ptr->next) {
		fi_ptr->domain_attr->auth_key_size = 0;
		fi_ptr->ep_attr->auth_key_size = 0;
	}

	/* Nothing left to do if hints weren't provided. */
	if (!hints)
		return FI_SUCCESS;

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

		/* Requesting FI_SOURCE adds overhead to a receive operation.
		 * Do not set FI_SOURCE unless explicitly requested.
		 */
		if (hints->caps && !(hints->caps & FI_SOURCE)) {
			fi_ptr->caps &= ~FI_SOURCE;
			fi_ptr->rx_attr->caps &= ~FI_SOURCE;
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
