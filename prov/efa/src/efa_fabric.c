/*
 * Copyright (c) 2014-2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017-2020 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"

#include <netdb.h>
#include <inttypes.h>

#include <infiniband/efadv.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include "ofi_prov.h"
#include <ofi_util.h>

#include "efa.h"
#if HAVE_EFA_DL
#include <ofi_shm.h>
#endif

#define EFA_FABRIC_PREFIX "EFA-"

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


#ifdef EFA_PERF_ENABLED
const char *efa_perf_counters_str[] = {
	EFA_PERF_FOREACH(OFI_STR)
};
#endif
static void efa_addr_to_str(const uint8_t *raw_addr, char *str);

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

static int efa_check_hints(uint32_t version, const struct fi_info *hints,
			   const struct fi_info *info)
{
	uint64_t prov_mode;
	size_t size;
	int ret;

	if (hints->caps & ~(info->caps)) {
		EFA_INFO(FI_LOG_CORE, "Unsupported capabilities\n");
		OFI_INFO_CHECK(&efa_prov, info, hints, caps, FI_TYPE_CAPS);
		return -FI_ENODATA;
	}

	prov_mode = ofi_mr_get_prov_mode(version, hints, info);

	if ((hints->mode & prov_mode) != prov_mode) {
		EFA_INFO(FI_LOG_CORE, "Required hints mode bits not set\n");
		OFI_INFO_MODE(&efa_prov, prov_mode, hints->mode);
		return -FI_ENODATA;
	}

	if (hints->fabric_attr) {
		ret = ofi_check_fabric_attr(&efa_prov, info->fabric_attr,
					    hints->fabric_attr);

		if (ret)
			return ret;
	}

	switch (hints->addr_format) {
	case FI_FORMAT_UNSPEC:
	case FI_ADDR_EFA:
		size = EFA_EP_ADDR_LEN;
		break;
	default:
		EFA_INFO(FI_LOG_CORE,
			 "Address format not supported: hints[%u], supported[%u,%u]\n",
			 hints->addr_format, FI_FORMAT_UNSPEC, FI_ADDR_EFA);
		return -FI_ENODATA;
	}

	if (hints->src_addr && hints->src_addrlen < size)
		return -FI_ENODATA;

	if (hints->dest_addr && hints->dest_addrlen < size)
		return -FI_ENODATA;

	if (hints->domain_attr) {
		ret = ofi_check_domain_attr(&efa_prov, version, info->domain_attr, hints);
		if (ret)
			return ret;
	}

	if (hints->ep_attr) {
		ret = ofi_check_ep_attr(&efa_util_prov, info->fabric_attr->api_version, info, hints);
		if (ret)
			return ret;
	}

	if (hints->rx_attr) {
		ret = ofi_check_rx_attr(&efa_prov, info, hints->rx_attr, hints->mode);
		if (ret)
			return ret;
	}

	if (hints->tx_attr) {
		ret = ofi_check_tx_attr(&efa_prov, info->tx_attr, hints->tx_attr, hints->mode);
		if (ret)
			return ret;
	}

	return 0;
}

static char *get_sysfs_path(void)
{
	char *env = NULL;
	char *sysfs_path = NULL;
	int len;

	/*
	 * Only follow use path passed in through the calling user's
	 * environment if we're not running SUID.
	 */
	if (getuid() == geteuid())
		env = getenv("SYSFS_PATH");

	if (env) {
		sysfs_path = strndup(env, IBV_SYSFS_PATH_MAX);
		len = strlen(sysfs_path);
		while (len > 0 && sysfs_path[len - 1] == '/') {
			--len;
			sysfs_path[len] = '\0';
		}
	} else {
		sysfs_path = strdup("/sys");
	}

	return sysfs_path;
}

#ifndef _WIN32

static int efa_get_driver(struct efa_device *ctx,
			     char **efa_driver)
{
	int ret;
	char *driver_sym_path;
	char driver_real_path[PATH_MAX];
	char *driver;
	ret = asprintf(&driver_sym_path, "%s%s",
		       ctx->ibv_ctx->device->ibdev_path, "/device/driver");
	if (ret < 0) {
		return -FI_ENOMEM;
	}

	if (!realpath(driver_sym_path, driver_real_path)) {
		ret = -errno;
		goto err_free_driver_sym;
	}

	driver = strrchr(driver_real_path, '/');
	if (!driver) {
		ret = -FI_EINVAL;
		goto err_free_driver_sym;
	}
	driver++;
	*efa_driver = strdup(driver);
	if (!*efa_driver) {
		ret = -FI_ENOMEM;
		goto err_free_driver_sym;
	}

	free(driver_sym_path);
	return 0;

err_free_driver_sym:
	free(driver_sym_path);
	return ret;
}

#else // _WIN32

static int efa_get_driver(struct efa_device *ctx,
			     char **efa_driver)
{
	int ret;
	/*
	 * On windows efa device is not exposed as infiniband device.
	 * The driver for efa device can be queried using Windows Setup API.
	 * The code required to do that is more complex than necessary in this context.
	 * We will return a hardcoded string as driver.
	 */
	ret = asprintf(efa_driver, "%s", "efa.sys");
	if (ret < 0) {
		return -FI_ENOMEM;
	}
	return 0;
}

#endif // _WIN32

#ifndef _WIN32

static int efa_get_device_version(struct efa_device *efa_device,
				  char **device_version)
{
	char *sysfs_path;
	int ret;

	*device_version = calloc(1, EFA_ABI_VER_MAX_LEN + 1);
	if (!*device_version) {
		return -FI_ENOMEM;
	}

	sysfs_path = get_sysfs_path();
	if (!sysfs_path) {
		return -FI_ENOMEM;
	}

	ret = fi_read_file(sysfs_path, "class/infiniband_verbs/abi_version",
			   *device_version,
			   EFA_ABI_VER_MAX_LEN);
	if (ret < 0) {
		goto free_sysfs_path;
	}

	free(sysfs_path);
	return 0;

free_sysfs_path:
	free(sysfs_path);
	return ret;
}

#else // _WIN32

static int efa_get_device_version(struct efa_device *efa_device,
				  char **device_version)
{
	int ret;
	/*
	 * On Windows, there is no sysfs. We use hw_ver field of ibv_attr to obtain it
	 */
	ret = asprintf(device_version, "%u", efa_device->ibv_attr.hw_ver);
	if (ret < 0) {
		return -FI_ENOMEM;
	}
	return 0;
}

#endif // _WIN32

#ifndef _WIN32

static int efa_get_pci_attr(struct efa_device *device,
			     struct fi_pci_attr *pci_attr)
{
	char *dbdf_sym_path;
	char *dbdf;
	char dbdf_real_path[PATH_MAX];
	int ret;
	ret = asprintf(&dbdf_sym_path, "%s%s",
	       device->ibv_ctx->device->ibdev_path, "/device");
	if (ret < 0) {
		return -FI_ENOMEM;
	}

	if (!realpath(dbdf_sym_path, dbdf_real_path)) {
		ret = -errno;
		goto err_free_dbdf_sym;
	}

	dbdf = strrchr(dbdf_real_path, '/');
	if (!dbdf) {
		ret = -FI_EINVAL;
		goto err_free_dbdf_sym;
	}
	dbdf++;

	ret = sscanf(dbdf, "%hx:%hhx:%hhx.%hhx", &pci_attr->domain_id,
		     &pci_attr->bus_id, &pci_attr->device_id,
		     &pci_attr->function_id);
	if (ret != 4) {
		ret = -FI_EINVAL;
		goto err_free_dbdf_sym;
	}

	free(dbdf_sym_path);
	return 0;

err_free_dbdf_sym:
	free(dbdf_sym_path);
	return ret;
}

#else // _WIN32

static int efa_get_pci_attr(struct efa_device *device,
			     struct fi_pci_attr *pci_attr)
{
	/*
	 * pci_attr is currently not supported on Windows. We return success
	 * to let applications continue without failures.
	 */
	return 0;
}

#endif // _WIN32

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

	ret = efa_get_device_version(device, &device_attr->device_version);
	if (ret != 0){
		goto err_free_nic;
	}

	ret = asprintf(&device_attr->vendor_id, "0x%x",
		       device->ibv_attr.vendor_id);
	if (ret < 0) {
		ret = -FI_ENOMEM;
		goto err_free_nic;
	}

	ret = efa_get_driver(device, &device_attr->driver);
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
	ret = efa_get_pci_attr(device, pci_attr);
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

static void efa_addr_to_str(const uint8_t *raw_addr, char *str)
{
	size_t name_len = strlen(EFA_FABRIC_PREFIX) + INET6_ADDRSTRLEN;
	char straddr[INET6_ADDRSTRLEN] = { 0 };

	if (!inet_ntop(AF_INET6, raw_addr, straddr, INET6_ADDRSTRLEN))
		return;
	snprintf(str, name_len, EFA_FABRIC_PREFIX "%s", straddr);
}

static int efa_str_to_ep_addr(const char *node, const char *service, struct efa_ep_addr *addr)
{
	int ret;

	if (!node)
		return -FI_EINVAL;

	memset(addr, 0, sizeof(*addr));

	ret = inet_pton(AF_INET6, node, addr->raw);
	if (ret != 1)
		return -FI_EINVAL;
	if (service)
		addr->qpn = atoi(service);

	return 0;
}

static int efa_alloc_info(struct efa_device *device, struct fi_info **info,
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

	name_len = strlen(EFA_FABRIC_PREFIX) + INET6_ADDRSTRLEN;

	fi->fabric_attr->name = calloc(1, name_len + 1);
	if (!fi->fabric_attr->name) {
		ret = -FI_ENOMEM;
		goto err_free_info;
	}
	efa_addr_to_str(device->ibv_gid.raw, fi->fabric_attr->name);

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

const struct fi_info *efa_get_efa_info(const char *domain_name)
{
	const struct fi_info *fi;

	for (fi = efa_util_prov.info; fi; fi = fi->next) {
		if (!strcmp(fi->domain_attr->name, domain_name))
			return fi;
	}

	return NULL;
}

static int efa_node_matches_addr(struct efa_ep_addr *addr, const char *node)
{
	struct efa_ep_addr eaddr;

	efa_str_to_ep_addr(node, NULL, &eaddr);
	return memcmp(&eaddr.raw, &addr->raw, sizeof(addr->raw));
}

static int efa_get_matching_info(uint32_t version, const char *node, uint64_t flags,
				 const struct fi_info *hints, struct fi_info **info)
{
	const struct fi_info *check_info;
	struct fi_info *fi, *tail;
	int ret;

	*info = tail = NULL;

	for (check_info = efa_util_prov.info; check_info; check_info = check_info->next) {
		ret = 0;
		if (flags & FI_SOURCE) {
			if (node)
				ret = efa_node_matches_addr(check_info->src_addr, node);
		} else if (hints && hints->src_addr) {
			ret = memcmp(check_info->src_addr, hints->src_addr, EFA_EP_ADDR_LEN);
		}

		if (ret)
			continue;
		EFA_INFO(FI_LOG_FABRIC, "found match for interface %s %s\n", node, check_info->fabric_attr->name);
		if (hints) {
			ret = efa_check_hints(version, hints, check_info);
			if (ret)
				continue;
		}

		fi = fi_dupinfo(check_info);
		if (!fi) {
			ret = -FI_ENOMEM;
			goto err_free_info;
		}

		fi->fabric_attr->api_version = version;

		if (!*info)
			*info = fi;
		else
			tail->next = fi;
		tail = fi;
	}

	if (!*info)
		return -FI_ENODATA;

	return 0;

err_free_info:
	fi_freeinfo(*info);
	*info = NULL;
	return ret;
}

static int efa_set_fi_address(const char *node, const char *service, uint64_t flags,
			      const struct fi_info *hints, struct fi_info *fi)
{
	struct efa_ep_addr tmp_addr;
	void *dest_addr = NULL;
	int ret = FI_SUCCESS;
	struct fi_info *cur;

	if (flags & FI_SOURCE) {
		if (hints && hints->dest_addr)
			dest_addr = hints->dest_addr;
	} else {
		if (node || service) {
			ret = efa_str_to_ep_addr(node, service, &tmp_addr);
			if (ret)
				return ret;
			dest_addr = &tmp_addr;
		} else if (hints && hints->dest_addr) {
			dest_addr = hints->dest_addr;
		}
	}

	if (dest_addr) {
		for (cur = fi; cur; cur = cur->next) {
			cur->dest_addr = malloc(EFA_EP_ADDR_LEN);
			if (!cur->dest_addr) {
				for (; fi->dest_addr; fi = fi->next)
					free(fi->dest_addr);
				return -FI_ENOMEM;
			}
			memcpy(cur->dest_addr, dest_addr, EFA_EP_ADDR_LEN);
			cur->dest_addrlen = EFA_EP_ADDR_LEN;
		}
	}
	return ret;
}

int efa_getinfo(uint32_t version, const char *node, const char *service,
		uint64_t flags, const struct fi_info *hints, struct fi_info **info)
{
	int ret;

	if (!(flags & FI_SOURCE) && hints && hints->src_addr &&
	    hints->src_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	if (((!node && !service) || (flags & FI_SOURCE)) &&
	    hints && hints->dest_addr &&
	    hints->dest_addrlen != EFA_EP_ADDR_LEN)
		return -FI_ENODATA;

	ret = efa_get_matching_info(version, node, flags, hints, info);
	if (ret)
		goto out;

	ret = efa_set_fi_address(node, service, flags, hints, *info);
	if (ret)
		goto out;

	ofi_alter_info(*info, hints, version);

out:
	if (!ret || ret == -FI_ENOMEM || ret == -FI_ENODEV) {
		return ret;
	} else {
		fi_freeinfo(*info);
		*info = NULL;
		return -FI_ENODATA;
	}
}

static int efa_fabric_close(fid_t fid)
{
	struct efa_fabric *efa_fabric;
	int ret;

	efa_fabric = container_of(fid, struct efa_fabric, util_fabric.fabric_fid.fid);
	ret = ofi_fabric_close(&efa_fabric->util_fabric);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_FABRIC,
			"Unable to close fabric: %s\n",
			fi_strerror(-ret));
		return ret;
	}

	if (efa_fabric->shm_fabric) {
		ret = fi_close(&efa_fabric->shm_fabric->fid);
		if (ret) {
			FI_WARN(&rxr_prov, FI_LOG_FABRIC,
				"Unable to close fabric: %s\n",
				fi_strerror(-ret));
			return ret;
		}
	}

#ifdef EFA_PERF_ENABLED
	ofi_perfset_log(&efa_fabric->perf_set, efa_perf_counters_str);
	ofi_perfset_close(&efa_fabric->perf_set);
#endif
	free(efa_fabric);

	return 0;
}

static struct fi_ops efa_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric efa_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	/*
	 * The reason we use rxr_domain_open() here is because it actually handles
	 * both RDM and DGRAM.
	 */
	.domain = rxr_domain_open,
	.passive_ep = fi_no_passive_ep,
	.eq_open = ofi_eq_create,
	.wait_open = ofi_wait_fd_open,
	.trywait = ofi_trywait
};

int efa_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric_fid,
	       void *context)
{
	const struct fi_info *info;
	struct efa_fabric *efa_fabric;
	int ret = 0, retv;

	efa_fabric = calloc(1, sizeof(*efa_fabric));
	if (!efa_fabric)
		return -FI_ENOMEM;

	for (info = efa_util_prov.info; info; info = info->next) {
		ret = ofi_fabric_init(&efa_prov, info->fabric_attr, attr,
				      &efa_fabric->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}

	if (ret)
		goto err_free_fabric;

	/* Open shm provider's fabric domain */
	if (shm_info) {
		assert(!strcmp(shm_info->fabric_attr->name, "shm"));
		ret = fi_fabric(shm_info->fabric_attr,
				&efa_fabric->shm_fabric, context);
		if (ret)
			goto err_close_util_fabric;
	} else {
		efa_fabric->shm_fabric = NULL;
	}

#ifdef EFA_PERF_ENABLED
	ret = ofi_perfset_create(&rxr_prov, &efa_fabric->perf_set,
				 efa_perf_size, perf_domain, perf_cntr,
				 perf_flags);

	if (ret)
		FI_WARN(&rxr_prov, FI_LOG_FABRIC,
			"Error initializing EFA perfset: %s\n",
			fi_strerror(-ret));
#endif


	*fabric_fid = &efa_fabric->util_fabric.fabric_fid;
	(*fabric_fid)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric_fid)->fid.ops = &efa_fi_ops;
	(*fabric_fid)->ops = &efa_ops_fabric;
	(*fabric_fid)->api_version = attr->api_version;

	return 0;

err_close_util_fabric:
	retv = ofi_fabric_close(&efa_fabric->util_fabric);
	if (retv)
		FI_WARN(&rxr_prov, FI_LOG_FABRIC,
			"Unable to close fabric: %s\n",
			fi_strerror(-retv));
err_free_fabric:
	free(efa_fabric);

	return ret;
}

#ifndef _WIN32


void efa_win_lib_finalize(void)
{
	// Nothing to do when we are not compiling for Windows
}

int efa_win_lib_initialize(void)
{
	return 0;
}

#else // _WIN32

#include "efawin.h"

/**
 * @brief open efawin.dll and load the symbols on windows platform
 *
 * This function is a no-op on windows
 */
int efa_win_lib_initialize(void)
{
	/* On Windows we need to load efawin dll to interact with
 	* efa device as there is no built-in verbs integration in the OS.
	* efawin dll provides all the ibv_* functions on Windows.
	* efa_load_efawin_lib function will replace stub ibv_* functions with
	* functions from efawin dll
	*/
	return efa_load_efawin_lib();
}

/**
 * @brief close efawin.dll on windows
 *
 * This function is a no-op on windows
 */
void efa_win_lib_finalize(void) {
	efa_free_efawin_lib();
}

#endif // _WIN32

struct fi_info *g_device_info_list;

/**
 * @brief initialize global variable: util_prov and g_device_info_list
 *
 * g_device_info_list is a linked list of fi_info
 * objects, with each fi_info object containing
 * one EFA device's attribute. Each device has
 * two fi_info objects, one for dgram, the other
 * for rdm.
 *
 * util_prov is the util_provider with its
 * info pointing to the head of g_device_info_list
 */
static int efa_util_prov_initialize()
{
	int i, err;
	struct fi_info *rdm_info = NULL;
	struct fi_info *dgrm_info = NULL;
	struct fi_info *tail_info = NULL;

	g_device_info_list = NULL;
	for (i = 0; i < g_device_cnt; i++) {
		rdm_info = NULL;
		dgrm_info = NULL;
		err = efa_alloc_info(&g_device_list[i], &rdm_info, &efa_rdm_domain);
		if (err)
			goto err_free;

		err = efa_alloc_info(&g_device_list[i], &dgrm_info, &efa_dgrm_domain);
		if (err)
			goto err_free;

		if (i==0) {
			g_device_info_list = rdm_info;
		} else {
			tail_info->next = rdm_info;
		}

		rdm_info->next = dgrm_info;
		tail_info = dgrm_info;
	}

	efa_util_prov.info = g_device_info_list;
	return 0;

err_free:
	fi_freeinfo(g_device_info_list);
	fi_freeinfo(rdm_info);
	fi_freeinfo(dgrm_info);
	return err;
}

/**
 * @brief release resources of g_device_info_list and reset util_prov
 */
static void efa_util_prov_finalize()
{
	fi_freeinfo(g_device_info_list);
	efa_util_prov.info = NULL;
}

/**
 * @brief initialize global variables use by EFA provider.
 *
 * This function call various functions to initialize
 * device_list, pd_list, win_lib and util_prov. All
 * of them are global variables.
 */
int efa_prov_initialize(void)
{
	int ret = 0, err;

	err = efa_device_list_initialize();
	if (err)
		return err;

	if (g_device_cnt <= 0)
		return -FI_ENODEV;

	err = efa_win_lib_initialize();
	if (err) {
		ret = err;
		goto err_free;
	}

	err = efa_util_prov_initialize();
	if (err) {
		ret = err;
		goto err_free;
	}

	return 0;

err_free:
	efa_win_lib_finalize();
	efa_device_list_finalize();
	return ret;
}

/**
 * @brief release the resources of global variables of provider
 *
 * This function calls various functions to release
 * util_prov, device_list, pd_list, win_lib
 */
void efa_prov_finalize(void)
{
	efa_util_prov_finalize();

	efa_device_list_finalize();

	efa_win_lib_finalize();

#if HAVE_EFA_DL
	smr_cleanup();
#endif
}

struct fi_provider efa_prov = {
	.name = EFA_PROV_NAME,
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = efa_getinfo,
	.fabric = efa_fabric,
	.cleanup = efa_prov_finalize
};

struct util_prov efa_util_prov = {
	.prov = &efa_prov,
	.info = NULL,
	.flags = 0,
};

