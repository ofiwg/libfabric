/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "ofi_prov.h"
#include <ofi_util.h>

#include "efa.h"
#if HAVE_EFA_DL
#include <ofi_shm.h>
#endif

#define EFA_FABRIC_PREFIX "EFA-"
#define EFA_FABRIC_NAME "efa"
#define EFA_DOMAIN_CAPS (FI_LOCAL_COMM | FI_REMOTE_COMM)

#define EFA_RDM_TX_CAPS (OFI_TX_MSG_CAPS)
#define EFA_RDM_RX_CAPS (OFI_RX_MSG_CAPS | FI_SOURCE)
#define EFA_DGRM_TX_CAPS (OFI_TX_MSG_CAPS)
#define EFA_DGRM_RX_CAPS (OFI_RX_MSG_CAPS | FI_SOURCE)
#define EFA_RDM_CAPS (EFA_RDM_TX_CAPS | EFA_RDM_RX_CAPS | EFA_DOMAIN_CAPS)
#define EFA_DGRM_CAPS (EFA_DGRM_TX_CAPS | EFA_DGRM_RX_CAPS | EFA_DOMAIN_CAPS)

#define EFA_TX_OP_FLAGS (FI_TRANSMIT_COMPLETE)

#define EFA_RX_MODE (0)

#define EFA_RX_RDM_OP_FLAGS (0)
#define EFA_RX_DGRM_OP_FLAGS (0)

#define EFA_MSG_ORDER (FI_ORDER_NONE)

#define EFA_NO_DEFAULT -1

const struct fi_fabric_attr efa_fabric_attr = {
	.fabric		= NULL,
	.name		= NULL,
	.prov_name	= NULL,
	.prov_version	= OFI_VERSION_DEF_PROV,
};

const struct fi_domain_attr efa_domain_attr = {
	.caps			= EFA_DOMAIN_CAPS,
	.threading		= FI_THREAD_DOMAIN,
	.control_progress	= FI_PROGRESS_AUTO,
	.data_progress		= FI_PROGRESS_AUTO,
	.resource_mgmt		= FI_RM_DISABLED,
	.mr_mode		= OFI_MR_BASIC_MAP | FI_MR_LOCAL | FI_MR_BASIC,
	.mr_key_size		= sizeof_field(struct ibv_sge, lkey),
	.cq_data_size		= 0,
	.tx_ctx_cnt		= 1024,
	.rx_ctx_cnt		= 1024,
	.max_ep_tx_ctx		= 1,
	.max_ep_rx_ctx		= 1,
	.mr_iov_limit		= EFA_MR_IOV_LIMIT,
};

const struct fi_ep_attr efa_ep_attr = {
	.protocol		= FI_PROTO_EFA,
	.protocol_version	= 1,
	.msg_prefix_size	= 0,
	.max_order_war_size	= 0,
	.mem_tag_format		= 0,
	.tx_ctx_cnt		= 1,
	.rx_ctx_cnt		= 1,
};

const struct fi_rx_attr efa_dgrm_rx_attr = {
	.caps			= EFA_DGRM_RX_CAPS,
	.mode			= FI_MSG_PREFIX | EFA_RX_MODE,
	.op_flags		= EFA_RX_DGRM_OP_FLAGS,
	.msg_order		= EFA_MSG_ORDER,
	.comp_order		= FI_ORDER_NONE,
	.total_buffered_recv	= 0,
	.iov_limit		= 1
};

const struct fi_rx_attr efa_rdm_rx_attr = {
	.caps			= EFA_RDM_RX_CAPS,
	.mode			= EFA_RX_MODE,
	.op_flags		= EFA_RX_RDM_OP_FLAGS,
	.msg_order		= EFA_MSG_ORDER,
	.comp_order		= FI_ORDER_NONE,
	.total_buffered_recv	= 0,
	.iov_limit		= 1
};

const struct fi_tx_attr efa_dgrm_tx_attr = {
	.caps			= EFA_DGRM_TX_CAPS,
	.mode			= FI_MSG_PREFIX,
	.op_flags		= EFA_TX_OP_FLAGS,
	.msg_order		= EFA_MSG_ORDER,
	.comp_order		= FI_ORDER_NONE,
	.inject_size		= 0,
	.rma_iov_limit		= 0,
};

const struct fi_tx_attr efa_rdm_tx_attr = {
	.caps			= EFA_RDM_TX_CAPS,
	.mode			= 0,
	.op_flags		= EFA_TX_OP_FLAGS,
	.msg_order		= EFA_MSG_ORDER,
	.comp_order		= FI_ORDER_NONE,
	.inject_size		= 0,
	.rma_iov_limit		= 1,
};

const struct efa_ep_domain efa_rdm_domain = {
	.suffix			= "-rdm",
	.type			= FI_EP_RDM,
	.caps			= EFA_RDM_CAPS,
};

const struct efa_ep_domain efa_dgrm_domain = {
	.suffix			= "-dgrm",
	.type			= FI_EP_DGRAM,
	.caps			= EFA_DGRM_CAPS,
};

static void efa_addr_to_str(const uint8_t *raw_addr, char *str)
{
	size_t name_len = strlen(EFA_FABRIC_PREFIX) + INET6_ADDRSTRLEN;
	char straddr[INET6_ADDRSTRLEN] = { 0 };

	if (!inet_ntop(AF_INET6, raw_addr, straddr, INET6_ADDRSTRLEN))
		return;
	snprintf(str, name_len, EFA_FABRIC_PREFIX "%s", straddr);
}

static int efa_alloc_fid_nic(struct fi_info *fi, struct efa_device *device)
{
	struct fi_device_attr *device_attr;
	struct fi_link_attr *link_attr;
	struct fi_bus_attr *bus_attr;
	struct fi_pci_attr *pci_attr;
	void *src_addr;
	int name_len;
	int ret;

	/* Sets nic ops and allocates basic structure */
	fi->nic = ofi_nic_dup(NULL);
	if (!fi->nic)
		return -FI_ENOMEM;

	device_attr = fi->nic->device_attr;
	bus_attr = fi->nic->bus_attr;
	pci_attr = &bus_attr->attr.pci;
	link_attr = fi->nic->link_attr;

	/* fi_device_attr */
	device_attr->name = strdup(device->ibv_ctx->device->name);
	if (!device_attr->name) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	ret = asprintf(&device_attr->device_id, "0x%x",
		       device->ibv_attr.vendor_part_id);
	/* ofi_nic_close will free all attributes of the fi_nic struct */
	if (ret < 0) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	ret = efa_device_get_version(device, &device_attr->device_version);
	if (ret != 0){
		goto err_free_nic;
	}

	ret = asprintf(&device_attr->vendor_id, "0x%x",
		       device->ibv_attr.vendor_id);
	if (ret < 0) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	ret = efa_device_get_driver(device, &device_attr->driver);
	if (ret != 0) {
		goto err_free_nic;
	}

	device_attr->firmware = strdup(device->ibv_attr.fw_ver);
	if (!device_attr->firmware) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	/* fi_bus_attr */
	bus_attr->bus_type = FI_BUS_PCI;

	/* fi_pci_attr */
	ret = efa_device_get_pci_attr(device, pci_attr);
	if (ret != 0) {
		goto err_free_nic;
	}
	/* fi_link_attr */
	src_addr = calloc(1, EFA_EP_ADDR_LEN);
	if (!src_addr) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	memcpy(src_addr, &device->ibv_gid, sizeof(device->ibv_gid));

	name_len = strlen(EFA_FABRIC_PREFIX) + INET6_ADDRSTRLEN;
	link_attr->address = calloc(1, name_len + 1);
	if (!link_attr->address) {
		ret = -FI_ENOMEM;
		goto err_free_src_addr;
	}

	efa_addr_to_str(src_addr, link_attr->address);

	link_attr->mtu = device->ibv_port_attr.max_msg_sz - rxr_pkt_max_header_size();
	link_attr->speed = ofi_vrb_speed(device->ibv_port_attr.active_speed,
	                                 device->ibv_port_attr.active_width);

	switch (device->ibv_port_attr.state) {
	case IBV_PORT_DOWN:
		link_attr->state = FI_LINK_DOWN;
		break;
	case IBV_PORT_ACTIVE:
		link_attr->state = FI_LINK_UP;
		break;
	default:
		link_attr->state = FI_LINK_UNKNOWN;
		break;
	}

	link_attr->network_type = strdup("Ethernet");
	if (!link_attr->network_type) {
		ret = -FI_ENOMEM;
		goto err_free_src_addr;
	}

	free(src_addr);
	return FI_SUCCESS;

err_free_src_addr:
	free(src_addr);
err_free_nic:
	fi_close(&fi->nic->fid);
	fi->nic = NULL;
	return ret;
}

static int efa_get_device_attrs(struct efa_device *device, struct fi_info *info)
{
	int ret;

	info->domain_attr->cq_cnt		= device->ibv_attr.max_cq;
	info->domain_attr->ep_cnt		= device->ibv_attr.max_qp;
	info->domain_attr->tx_ctx_cnt		= MIN(info->domain_attr->tx_ctx_cnt, device->ibv_attr.max_qp);
	info->domain_attr->rx_ctx_cnt		= MIN(info->domain_attr->rx_ctx_cnt, device->ibv_attr.max_qp);
	info->domain_attr->max_ep_tx_ctx	= 1;
	info->domain_attr->max_ep_rx_ctx	= 1;
	info->domain_attr->resource_mgmt	= FI_RM_DISABLED;
	info->domain_attr->mr_cnt		= device->ibv_attr.max_mr;

#if HAVE_CUDA || HAVE_NEURON
	if (info->ep_attr->type == FI_EP_RDM &&
	    (ofi_hmem_is_initialized(FI_HMEM_CUDA) ||
	     ofi_hmem_is_initialized(FI_HMEM_NEURON))) {
		info->caps			|= FI_HMEM;
		info->tx_attr->caps		|= FI_HMEM;
		info->rx_attr->caps		|= FI_HMEM;
		info->domain_attr->mr_mode	|= FI_MR_HMEM;
	}
#endif

	EFA_DBG(FI_LOG_DOMAIN, "Domain attribute :\n"
				"\t info->domain_attr->cq_cnt		= %zu\n"
				"\t info->domain_attr->ep_cnt		= %zu\n"
				"\t info->domain_attr->rx_ctx_cnt	= %zu\n"
				"\t info->domain_attr->tx_ctx_cnt	= %zu\n"
				"\t info->domain_attr->max_ep_tx_ctx	= %zu\n"
				"\t info->domain_attr->max_ep_rx_ctx	= %zu\n",
				info->domain_attr->cq_cnt,
				info->domain_attr->ep_cnt,
				info->domain_attr->tx_ctx_cnt,
				info->domain_attr->rx_ctx_cnt,
				info->domain_attr->max_ep_tx_ctx,
				info->domain_attr->max_ep_rx_ctx);

	info->tx_attr->iov_limit = device->efa_attr.max_sq_sge;
	info->tx_attr->size = rounddown_power_of_two(device->efa_attr.max_sq_wr);
	if (info->ep_attr->type == FI_EP_RDM) {
		info->tx_attr->inject_size = device->efa_attr.inline_buf_size;
	} else if (info->ep_attr->type == FI_EP_DGRAM) {
                /*
                 * Currently, there is no mechanism for EFA layer (lower layer)
                 * to discard completions internally and FI_INJECT is not optional,
                 * it can only be disabled by setting inject_size to 0. RXR
                 * layer does not have this issue as completions can be read from
                 * the EFA layer and discarded in the RXR layer. For dgram
                 * endpoint, inject size needs to be set to 0
                 */
		info->tx_attr->inject_size = 0;
	}
	info->rx_attr->iov_limit = device->efa_attr.max_rq_sge;
	info->rx_attr->size = rounddown_power_of_two(device->efa_attr.max_rq_wr / info->rx_attr->iov_limit);

	EFA_DBG(FI_LOG_DOMAIN, "Tx/Rx attribute :\n"
				"\t info->tx_attr->iov_limit		= %zu\n"
				"\t info->tx_attr->size			= %zu\n"
				"\t info->tx_attr->inject_size		= %zu\n"
				"\t info->rx_attr->iov_limit		= %zu\n"
				"\t info->rx_attr->size			= %zu\n",
				info->tx_attr->iov_limit,
				info->tx_attr->size,
				info->tx_attr->inject_size,
				info->rx_attr->iov_limit,
				info->rx_attr->size);


	info->ep_attr->max_msg_size		= device->ibv_port_attr.max_msg_sz;
	info->ep_attr->max_order_raw_size	= device->ibv_port_attr.max_msg_sz;
	info->ep_attr->max_order_waw_size	= device->ibv_port_attr.max_msg_sz;

	/* Set fid nic attributes. */
	ret = efa_alloc_fid_nic(info, device);
	if (ret) {
		EFA_WARN(FI_LOG_FABRIC,
			 "Unable to allocate fid_nic: %s\n", fi_strerror(-ret));
		return ret;
	}

	return 0;
}

/**
 * @brief allocate a prov_info object.
 *
 * A prov_info is a fi_info object used by libfabric utility code to verify fi_info passed
 * by user (user_info). This function allocate such an object.
 *
 * @param	info[out]	info object to be allocated
 * @param	device[in]	efa_device that contains device's information
 * @param	ep_dom[in]	either ep_rdm_dom or ep_dgrm_dom
 * @return	0 on success
 * 		negative libfabric error code on failure
 */
int efa_prov_info_alloc(struct fi_info **info,
			struct efa_device *device,
			const struct efa_ep_domain *ep_dom)
{
	struct fi_info *fi;
	size_t name_len;
	int ret;

	fi = fi_allocinfo();
	if (!fi)
		return -FI_ENOMEM;

	fi->caps		= ep_dom->caps;
	fi->handle		= NULL;
	*fi->ep_attr		= efa_ep_attr;
	if (ep_dom->type == FI_EP_RDM) {
		*fi->tx_attr	= efa_rdm_tx_attr;
		*fi->rx_attr	= efa_rdm_rx_attr;
	} else if (ep_dom->type == FI_EP_DGRAM) {
		fi->mode |= FI_MSG_PREFIX;
		fi->ep_attr->msg_prefix_size = 40;
		*fi->tx_attr	= efa_dgrm_tx_attr;
		*fi->rx_attr	= efa_dgrm_rx_attr;
	}

	*fi->domain_attr	= efa_domain_attr;
	*fi->fabric_attr	= efa_fabric_attr;

	fi->ep_attr->protocol	= FI_PROTO_EFA;
	fi->ep_attr->type	= ep_dom->type;

	ret = efa_get_device_attrs(device, fi);
	if (ret)
		goto err_free_info;

	name_len = strlen(EFA_FABRIC_NAME);

	fi->fabric_attr->name = calloc(1, name_len + 1);
	if (!fi->fabric_attr->name) {
		ret = -FI_ENOMEM;
		goto err_free_info;
	}

	strcpy(fi->fabric_attr->name, EFA_FABRIC_NAME);

	name_len = strlen(device->ibv_ctx->device->name) + strlen(ep_dom->suffix);
	fi->domain_attr->name = malloc(name_len + 1);
	if (!fi->domain_attr->name) {
		ret = -FI_ENOMEM;
		goto err_free_info;
	}

	snprintf(fi->domain_attr->name, name_len + 1, "%s%s",
		 device->ibv_ctx->device->name, ep_dom->suffix);
	fi->domain_attr->name[name_len] = '\0';

	fi->addr_format = FI_ADDR_EFA;
	fi->src_addr = calloc(1, EFA_EP_ADDR_LEN);
	if (!fi->src_addr) {
		ret = -FI_ENOMEM;
		goto err_free_info;
	}
	fi->src_addrlen = EFA_EP_ADDR_LEN;
	memcpy(fi->src_addr, &device->ibv_gid, sizeof(device->ibv_gid));

	fi->domain_attr->av_type = FI_AV_TABLE;

	*info = fi;
	return 0;

err_free_info:
	fi_freeinfo(fi);
	return ret;
}

