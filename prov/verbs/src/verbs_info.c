/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <ofi_util.h>

#include <ifaddrs.h>
#include <net/if.h>

#include "fi_verbs.h"
#include "ep_rdm/verbs_rdm.h"


#define VERBS_IB_PREFIX "IB-0x"
#define VERBS_IWARP_FABRIC "Ethernet-iWARP"

#define VERBS_DOMAIN_CAPS (FI_LOCAL_COMM | FI_REMOTE_COMM)

#define VERBS_MSG_CAPS (FI_MSG | FI_RMA | FI_ATOMICS | FI_READ | FI_WRITE |	\
			FI_SEND | FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE |	\
			VERBS_DOMAIN_CAPS)
#define VERBS_RDM_CAPS (FI_MSG | FI_RMA | FI_TAGGED | FI_READ | FI_WRITE |	\
			FI_RECV | FI_MULTI_RECV | FI_SEND | FI_REMOTE_READ |	\
			FI_REMOTE_WRITE | VERBS_DOMAIN_CAPS)
#define VERBS_DGRAM_CAPS (FI_MSG | FI_RECV | FI_SEND | VERBS_DOMAIN_CAPS)

#define VERBS_RDM_MODE (FI_CONTEXT)

#define VERBS_TX_OP_FLAGS (FI_INJECT | FI_COMPLETION | FI_TRANSMIT_COMPLETE)
#define VERBS_TX_OP_FLAGS_IWARP (FI_INJECT | FI_COMPLETION)
#define VERBS_TX_OP_FLAGS_IWARP_RDM (VERBS_TX_OP_FLAGS)

#define VERBS_TX_RDM_MODE VERBS_RDM_MODE

#define VERBS_RX_MODE (FI_RX_CQ_DATA)

#define VERBS_RX_RDM_OP_FLAGS (FI_COMPLETION)

#define VERBS_MSG_ORDER (FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS | \
		FI_ORDER_WAW | FI_ORDER_WAS | FI_ORDER_SAW | FI_ORDER_SAS )

#define VERBS_INFO_NODE_2_UD_ADDR(sybsys, node, svc, ib_ud_addr)			\
	VERBS_INFO(sybsys, "'%s:%u' resolved to <gid <interface_id=%"PRIu64		\
			   ", subnet_prefix=%"PRIu64">, lid=%d, service = %u>\n",	\
		   node, svc, be64toh((ib_ud_addr)->gid.global.interface_id),		\
		   be64toh((ib_ud_addr)->gid.global.subnet_prefix),			\
		   (ib_ud_addr)->lid, (ib_ud_addr)->service)

const struct fi_fabric_attr verbs_fabric_attr = {
	.prov_version		= VERBS_PROV_VERS,
};

const struct fi_domain_attr verbs_domain_attr = {
	.caps			= VERBS_DOMAIN_CAPS,
	.threading		= FI_THREAD_SAFE,
	.control_progress	= FI_PROGRESS_AUTO,
	.data_progress		= FI_PROGRESS_AUTO,
	.resource_mgmt		= FI_RM_ENABLED,
	.mr_mode		= OFI_MR_BASIC_MAP | FI_MR_LOCAL | FI_MR_BASIC,
	.mr_key_size		= sizeof_field(struct ibv_sge, lkey),
	.cq_data_size		= sizeof_field(struct ibv_send_wr, imm_data),
	.tx_ctx_cnt		= 1024,
	.rx_ctx_cnt		= 1024,
	.max_ep_tx_ctx		= 1,
	.max_ep_rx_ctx		= 1,
	.mr_iov_limit		= VERBS_MR_IOV_LIMIT,
	/* max_err_data is size of ibv_wc::vendor_err for CQ, 0 - for EQ */
	.max_err_data		= sizeof_field(struct ibv_wc, vendor_err),
};

const struct fi_ep_attr verbs_ep_attr = {
	.protocol_version	= 1,
	.msg_prefix_size	= 0,
	.max_order_war_size	= 0,
	.mem_tag_format		= 0,
	.tx_ctx_cnt		= 1,
	.rx_ctx_cnt		= 1,
};

const struct fi_rx_attr verbs_rx_attr = {
	.mode			= VERBS_RX_MODE,
	.msg_order		= VERBS_MSG_ORDER,
	.comp_order		= FI_ORDER_STRICT | FI_ORDER_DATA,
	.total_buffered_recv	= 0,
};

const struct fi_rx_attr verbs_rdm_rx_attr = {
	.mode			= VERBS_RDM_MODE | VERBS_RX_MODE,
	.op_flags		= VERBS_RX_RDM_OP_FLAGS,
	.msg_order		= VERBS_MSG_ORDER,
	.total_buffered_recv	= 0,
	.iov_limit		= 1
};

const struct fi_tx_attr verbs_tx_attr = {
	.mode			= 0,
	.op_flags		= VERBS_TX_OP_FLAGS,
	.msg_order		= VERBS_MSG_ORDER,
	.comp_order		= FI_ORDER_STRICT,
	.inject_size		= 0,
	.rma_iov_limit		= 1,
};

const struct fi_tx_attr verbs_rdm_tx_attr = {
	.mode			= VERBS_TX_RDM_MODE,
	.op_flags		= VERBS_TX_OP_FLAGS,
	.msg_order		= VERBS_MSG_ORDER,
	.inject_size		= FI_IBV_RDM_DFLT_BUFFERED_SIZE,
	.rma_iov_limit		= 1,
};

const struct verbs_ep_domain verbs_msg_domain = {
	.suffix			= "",
	.type			= FI_EP_MSG,
	.caps			= VERBS_MSG_CAPS,
};

const struct verbs_ep_domain verbs_rdm_domain = {
	.suffix			= "-rdm",
	.type			= FI_EP_RDM,
	.caps			= VERBS_RDM_CAPS,
};

const struct verbs_ep_domain verbs_dgram_domain = {
	.suffix			= "-dgram",
	.type			= FI_EP_DGRAM,
	.caps			= VERBS_DGRAM_CAPS,
};

struct fi_ibv_rdm_sysaddr
{
	struct sockaddr_in addr;
	int is_found;
};

int fi_ibv_check_ep_attr(const struct fi_info *hints,
			 const struct fi_info *info)
{
	struct fi_info *user_hints;
	struct util_prov tmp_util_prov = {
		.prov = &fi_ibv_prov,
		.info = NULL,
		.flags = (info->domain_attr->max_ep_srx_ctx &&
			  info->ep_attr->type == FI_EP_MSG) ?
			 UTIL_RX_SHARED_CTX : 0,
	};
	int ret;

	switch (hints->ep_attr->protocol) {
	case FI_PROTO_UNSPEC:
	case FI_PROTO_RDMA_CM_IB_RC:
	case FI_PROTO_IWARP:
	case FI_PROTO_IB_UD:
	case FI_PROTO_IB_RDM:
	case FI_PROTO_IWARP_RDM:
		break;
	default:
		VERBS_INFO(FI_LOG_CORE,
			   "Unsupported protocol\n");
		return -FI_ENODATA;
	}

	user_hints = fi_dupinfo(hints);
	if (!user_hints)
		return -FI_ENOMEM;

	/*
	 * verbs provider requires more complex verification of the
	 * protocol in compare to verification that is presented in
	 * the utility function. Change the protocol to FI_PROTO_UNSPEC
	 * to avoid verification of protocol in the ofi_check_ep_attr
	 */
	user_hints->ep_attr->protocol = FI_PROTO_UNSPEC;

	ret = ofi_check_ep_attr(&tmp_util_prov, info->fabric_attr->api_version,
				info, user_hints);
	fi_freeinfo(user_hints);
	return ret;
}

int fi_ibv_check_rx_attr(const struct fi_rx_attr *attr,
			 const struct fi_info *hints,
			 const struct fi_info *info)
{
	/* WARNING: This is not thread safe */
	uint64_t saved_prov_mode = info->rx_attr->mode;
	int ret;

	info->rx_attr->mode = (hints->domain_attr &&
			       hints->domain_attr->cq_data_size) ?
			info->rx_attr->mode :
			(info->rx_attr->mode & ~FI_RX_CQ_DATA);

	ret = ofi_check_rx_attr(&fi_ibv_prov, info, attr, hints->mode);

	info->rx_attr->mode = saved_prov_mode;

	return ret;
}

static int fi_ibv_check_hints(uint32_t version, const struct fi_info *hints,
			      const struct fi_info *info)
{
	int ret;
	uint64_t prov_mode;

	if (hints->caps & ~(info->caps)) {
		VERBS_INFO(FI_LOG_CORE, "Unsupported capabilities\n");
		FI_INFO_CHECK(&fi_ibv_prov, info, hints, caps, FI_TYPE_CAPS);
		return -FI_ENODATA;
	}

	prov_mode = ofi_mr_get_prov_mode(version, hints, info);

	if ((hints->mode & prov_mode) != prov_mode) {
		VERBS_INFO(FI_LOG_CORE, "needed mode not set\n");
		FI_INFO_MODE(&fi_ibv_prov, prov_mode, hints->mode);
		return -FI_ENODATA;
	}

	if (hints->fabric_attr) {
		ret = ofi_check_fabric_attr(&fi_ibv_prov, info->fabric_attr,
					    hints->fabric_attr);
		if (ret)
			return ret;
	}

	if (hints->domain_attr) {
		ret = ofi_check_domain_attr(&fi_ibv_prov, version,
					    info->domain_attr,
					    hints);
		if (ret)
			return ret;
	}

	if (hints->ep_attr) {
		ret = fi_ibv_check_ep_attr(hints, info);
		if (ret)
			return ret;
	}

	if (hints->rx_attr) {
		ret = fi_ibv_check_rx_attr(hints->rx_attr, hints, info);
		if (ret)
			return ret;
	}

	if (hints->tx_attr) {
		ret = ofi_check_tx_attr(&fi_ibv_prov, info->tx_attr,
					hints->tx_attr, hints->mode);
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

int fi_ibv_fi_to_rai(const struct fi_info *fi, uint64_t flags,
		     struct rdma_addrinfo *rai)
{
	memset(rai, 0, sizeof *rai);
	if (flags & FI_SOURCE)
		rai->ai_flags = RAI_PASSIVE;
	if (flags & FI_NUMERICHOST)
		rai->ai_flags |= RAI_NUMERICHOST;

	rai->ai_qp_type = IBV_QPT_RC;
	rai->ai_port_space = RDMA_PS_TCP;

	if (!fi)
		return 0;

	switch(fi->addr_format) {
	case FI_SOCKADDR_IN:
	case FI_FORMAT_UNSPEC:
		rai->ai_family = AF_INET;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR_IN6:
		rai->ai_family = AF_INET6;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR_IB:
		rai->ai_family = AF_IB;
		rai->ai_flags |= RAI_FAMILY;
		break;
	case FI_SOCKADDR:
		if (fi->src_addrlen) {
			rai->ai_family = ((struct sockaddr *)fi->src_addr)->sa_family;
			rai->ai_flags |= RAI_FAMILY;
		} else if (fi->dest_addrlen) {
			rai->ai_family = ((struct sockaddr *)fi->dest_addr)->sa_family;
			rai->ai_flags |= RAI_FAMILY;
		}
		break;
	default:
		VERBS_INFO(FI_LOG_FABRIC, "Unknown fi->addr_format\n");
	}

	if (fi->src_addrlen) {
		if (!(rai->ai_src_addr = malloc(fi->src_addrlen)))
			return -FI_ENOMEM;
		memcpy(rai->ai_src_addr, fi->src_addr, fi->src_addrlen);
		rai->ai_src_len = fi->src_addrlen;
	}
	if (fi->dest_addrlen) {
		if (!(rai->ai_dst_addr = malloc(fi->dest_addrlen)))
			return -FI_ENOMEM;
		memcpy(rai->ai_dst_addr, fi->dest_addr, fi->dest_addrlen);
		rai->ai_dst_len = fi->dest_addrlen;
	}

	return 0;
}

static inline
void *fi_ibv_dgram_ep_name_to_string(const struct ofi_ib_ud_ep_name *name,
				     size_t *len)
{
	char *str;
	if (!name)
		return NULL;

	*len = sizeof(struct ofi_ib_ud_ep_name);

	str = calloc(*len, 1);
	if (!str)
		return NULL;

	if (!ofi_straddr((void *)str, len, FI_ADDR_IB_UD, name)) {
		free(str);
		return NULL;
	}

	return str;
}

static int fi_ibv_fill_addr_by_ep_name(struct ofi_ib_ud_ep_name *ep_name,
				       uint32_t fmt, void **addr, size_t *addrlen)
{
	if (fmt == FI_ADDR_STR) {
		*addr = fi_ibv_dgram_ep_name_to_string(ep_name, addrlen);
		if (!*addr)
			return -FI_ENOMEM;
	} else {
		*addr = calloc(1, sizeof(*ep_name));
		if (!*addr)
			return -FI_ENOMEM;
		memcpy(*addr, ep_name, sizeof(*ep_name));
		*addrlen = sizeof(*ep_name);
	}

	return FI_SUCCESS;
}

static int fi_ibv_rai_to_fi(struct rdma_addrinfo *rai, struct fi_info *fi)
{
	if (!rai)
		return FI_SUCCESS;

	fi->addr_format = ofi_translate_addr_format(rai->ai_family);
	if (fi->addr_format == FI_FORMAT_UNSPEC) {
		VERBS_WARN(FI_LOG_FABRIC, "Unknown address format\n");
		return -FI_EINVAL;
	}

	if (rai->ai_src_len) {
		free(fi->src_addr);
 		if (!(fi->src_addr = malloc(rai->ai_src_len)))
 			return -FI_ENOMEM;
 		memcpy(fi->src_addr, rai->ai_src_addr, rai->ai_src_len);
 		fi->src_addrlen = rai->ai_src_len;
 	}
 	if (rai->ai_dst_len) {
		free(fi->dest_addr);
		if (!(fi->dest_addr = malloc(rai->ai_dst_len)))
			return -FI_ENOMEM;
 		memcpy(fi->dest_addr, rai->ai_dst_addr, rai->ai_dst_len);
 		fi->dest_addrlen = rai->ai_dst_len;
 	}

 	return FI_SUCCESS;
}

static inline int fi_ibv_get_qp_cap(struct ibv_context *ctx,
				    struct fi_info *info)
{
	struct ibv_pd *pd;
	struct ibv_cq *cq;
	struct ibv_qp *qp;
	struct ibv_qp_init_attr init_attr;
	enum ibv_qp_type qp_type;
	int ret = 0;

	pd = ibv_alloc_pd(ctx);
	if (!pd) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_alloc_pd", errno);
		return -errno;
	}

	cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
	if (!cq) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_cq", errno);
		ret = -errno;
		goto err1;
	}

	qp_type = (info->ep_attr->type != FI_EP_DGRAM) ?
			    IBV_QPT_RC : IBV_QPT_UD;

	memset(&init_attr, 0, sizeof init_attr);
	init_attr.send_cq = cq;
	init_attr.recv_cq = cq;
	init_attr.cap.max_send_wr = fi_ibv_gl_data.def_tx_size;
	init_attr.cap.max_recv_wr = fi_ibv_gl_data.def_rx_size;
	init_attr.cap.max_send_sge = fi_ibv_gl_data.def_tx_iov_limit;
	init_attr.cap.max_recv_sge = fi_ibv_gl_data.def_rx_iov_limit;
	init_attr.cap.max_inline_data = fi_ibv_find_max_inline(pd, ctx, qp_type);
	init_attr.qp_type = qp_type;

	qp = ibv_create_qp(pd, &init_attr);
	if (!qp) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "ibv_create_qp", errno);
		ret = -errno;
		goto err2;
	}

	info->tx_attr->inject_size = init_attr.cap.max_inline_data;

	ibv_destroy_qp(qp);
err2:
	ibv_destroy_cq(cq);
err1:
	ibv_dealloc_pd(pd);

	return ret;
}

static int fi_ibv_mtu_type_to_len(enum ibv_mtu mtu_type)
{
	switch (mtu_type) {
	case IBV_MTU_256:
		return 256;
	case IBV_MTU_512:
		return 512;
	case IBV_MTU_1024:
		return 1024;
	case IBV_MTU_2048:
		return 2048;
	case IBV_MTU_4096:
		return 4096;
	default:
		return -FI_EINVAL;
	}
}

static int fi_ibv_get_device_attrs(struct ibv_context *ctx,
				   struct fi_info *info)
{
	struct ibv_device_attr device_attr;
	struct ibv_port_attr port_attr;
	size_t max_sup_size;
	int ret = 0;
	uint8_t port_num;

	ret = ibv_query_device(ctx, &device_attr);
	if (ret) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC,
				 "ibv_query_device", errno);
		return -errno;
	}

	info->domain_attr->cq_cnt 		= device_attr.max_cq;
	info->domain_attr->ep_cnt 		= device_attr.max_qp;
	info->domain_attr->tx_ctx_cnt 		= MIN(info->domain_attr->tx_ctx_cnt,
						      device_attr.max_qp);
	info->domain_attr->rx_ctx_cnt 		= MIN(info->domain_attr->rx_ctx_cnt,
						      device_attr.max_qp);
	info->domain_attr->max_ep_tx_ctx 	= MIN(info->domain_attr->tx_ctx_cnt,
						      device_attr.max_qp);
	info->domain_attr->max_ep_rx_ctx 	= MIN(info->domain_attr->rx_ctx_cnt,
						      device_attr.max_qp);
	info->domain_attr->max_ep_srx_ctx	= device_attr.max_srq;
	info->domain_attr->mr_cnt		= device_attr.max_mr;
	if (info->ep_attr->type == FI_EP_RDM)
		info->domain_attr->cntr_cnt	= device_attr.max_qp * 4;

	info->tx_attr->size 			= device_attr.max_qp_wr;
	info->tx_attr->iov_limit 		= device_attr.max_sge;

	info->rx_attr->size 			= device_attr.max_srq_wr ?
						  MIN(device_attr.max_qp_wr,
						      device_attr.max_srq_wr) :
						  device_attr.max_qp_wr;
	info->rx_attr->iov_limit 		= device_attr.max_srq_sge ?
						  MIN(device_attr.max_sge,
						      device_attr.max_srq_sge) :
						  device_attr.max_sge;

	ret = fi_ibv_get_qp_cap(ctx, info);
	if (ret)
		return ret;

	for (port_num = 1; port_num < device_attr.phys_port_cnt + 1; port_num++) {
		ret = ibv_query_port(ctx, port_num, &port_attr);
		if (ret) {
			VERBS_INFO_ERRNO(FI_LOG_FABRIC,
					 "ibv_query_port", errno);
			return -errno;
		}
		if (port_attr.state == IBV_PORT_ACTIVE)
			break;
	}

	if (port_num == device_attr.phys_port_cnt + 1) {
		VERBS_WARN(FI_LOG_FABRIC, "There are no active ports\n");
		return -FI_ENODATA;
	} else {
		VERBS_INFO(FI_LOG_FABRIC,
			   "The first found active port is %"PRIu8"\n",
			   port_num);
	}

	if (info->ep_attr->type == FI_EP_DGRAM) {
		ret = fi_ibv_mtu_type_to_len(port_attr.active_mtu);
		if (ret < 0) {
			VERBS_WARN(FI_LOG_FABRIC, "Device %s (port: %d) reports"
				   " an unrecognized MTU (%d) \n",
				   ibv_get_device_name(ctx->device), port_num,
				   port_attr.active_mtu);
			return ret;
		}
		max_sup_size = MIN(ret, port_attr.max_msg_sz);
	} else {
		max_sup_size = port_attr.max_msg_sz;
	}

	info->ep_attr->max_msg_size 		= max_sup_size;
	info->ep_attr->max_order_raw_size 	= max_sup_size;
	info->ep_attr->max_order_waw_size	= max_sup_size;

	return 0;
}

/*
 * USNIC plugs into the verbs framework, but is not a usable device.
 * Manually check for devices and fail gracefully if none are present.
 * This avoids the lower libraries (libibverbs and librdmacm) from
 * reporting error messages to stderr.
 */
static int fi_ibv_have_device(void)
{
	struct ibv_device **devs;
	struct ibv_context *verbs;
	int i, ret = 0;

	devs = ibv_get_device_list(NULL);
	if (!devs)
		return 0;

	for (i = 0; devs[i]; i++) {
		verbs = ibv_open_device(devs[i]);
		if (verbs) {
			ibv_close_device(verbs);
			ret = 1;
			break;
		}
	}

	ibv_free_device_list(devs);
	return ret;
}

static int fi_ibv_alloc_info(struct ibv_context *ctx, struct fi_info **info,
			     const struct verbs_ep_domain *ep_dom)
{
	struct fi_info *fi;
	union ibv_gid gid;
	size_t name_len;
	int ret;

	if ((ctx->device->transport_type != IBV_TRANSPORT_IB) &&
	    (ep_dom->type == FI_EP_DGRAM))
		return -FI_EINVAL;

	if (!(fi = fi_allocinfo()))
		return -FI_ENOMEM;

	fi->caps		= ep_dom->caps;
	fi->handle		= NULL;
	if (ep_dom->type == FI_EP_RDM) {
		fi->mode	= VERBS_RDM_MODE;
		*(fi->tx_attr)	= verbs_rdm_tx_attr;
		*(fi->rx_attr)	= verbs_rdm_rx_attr;
	} else {
		*(fi->tx_attr)	= verbs_tx_attr;
		*(fi->rx_attr)	= verbs_rx_attr;
	}

	*(fi->ep_attr)		= verbs_ep_attr;
	*(fi->domain_attr)	= verbs_domain_attr;

	if (ep_dom->type == FI_EP_RDM)
		fi->domain_attr->mr_mode &= ~FI_MR_LOCAL;

	*(fi->fabric_attr)	= verbs_fabric_attr;

	fi->ep_attr->type	= ep_dom->type;
	fi->tx_attr->caps	= ep_dom->caps;
	fi->rx_attr->caps	= ep_dom->caps;

	ret = fi_ibv_get_device_attrs(ctx, fi);
	if (ret)
		goto err;

	if (ep_dom->type == FI_EP_RDM) {
		fi->tx_attr->iov_limit = 1;
		fi->tx_attr->rma_iov_limit = 1;
		fi->tx_attr->inject_size = fi_ibv_gl_data.rdm.buffer_size;

		fi->rx_attr->iov_limit = 1;
	}

	switch (ctx->device->transport_type) {
	case IBV_TRANSPORT_IB:
		if(ibv_query_gid(ctx, 1, 0, &gid)) {
			VERBS_INFO_ERRNO(FI_LOG_FABRIC,
					 "ibv_query_gid", errno);
			ret = -errno;
			goto err;
		}

		name_len = strlen(VERBS_IB_PREFIX) + INET6_ADDRSTRLEN;
		if (!(fi->fabric_attr->name = calloc(1, name_len + 1))) {
			ret = -FI_ENOMEM;
			goto err;
		}

		snprintf(fi->fabric_attr->name, name_len, VERBS_IB_PREFIX "%" PRIu64,
			 be64toh(gid.global.subnet_prefix));

		switch (ep_dom->type) {
		case FI_EP_MSG:
			fi->ep_attr->protocol = FI_PROTO_RDMA_CM_IB_RC;
			break;
		case FI_EP_RDM:
			fi->ep_attr->protocol = FI_PROTO_IB_RDM;
			break;
		case FI_EP_DGRAM:
			fi->ep_attr->protocol = FI_PROTO_IB_UD;
			break;
		default:
			assert(0); /* Shouldn't go here */
			ret = -FI_EINVAL;
			goto err;
		}
		break;
	case IBV_TRANSPORT_IWARP:
		fi->fabric_attr->name = strdup(VERBS_IWARP_FABRIC);
		if (!fi->fabric_attr->name) {
			ret = -FI_ENOMEM;
			goto err;
		}

		if (ep_dom->type == FI_EP_MSG) {
			fi->ep_attr->protocol = FI_PROTO_IWARP;
			fi->tx_attr->op_flags = VERBS_TX_OP_FLAGS_IWARP;
		} else {
			fi->ep_attr->protocol = FI_PROTO_IWARP_RDM;
			fi->tx_attr->op_flags = VERBS_TX_OP_FLAGS_IWARP_RDM;
		}

		/* TODO Some iWarp HW may support immediate data as per RFC 7306
		 * (RDMA Protocol Extensions). Update this to figure out if the
		 * hw supports immediate data dynamically */
		fi->domain_attr->cq_data_size = 0;
		break;
	default:
		VERBS_INFO(FI_LOG_CORE, "Unknown transport type\n");
		ret = -FI_ENODATA;
		goto err;
	}

	name_len = strlen(ctx->device->name) + strlen(ep_dom->suffix);
	fi->domain_attr->name = malloc(name_len + 1);
	if (!fi->domain_attr->name) {
		ret = -FI_ENOMEM;
		goto err;
	}

	snprintf(fi->domain_attr->name, name_len + 1, "%s%s",
		 ctx->device->name, ep_dom->suffix);
	fi->domain_attr->name[name_len] = '\0';

	*info = fi;
	return 0;
err:
	fi_freeinfo(fi);
	return ret;
}

static void fi_ibv_verbs_devs_free(struct dlist_entry *verbs_devs)
{
	struct verbs_dev_info *dev;
	struct verbs_addr *addr;

	while (!dlist_empty(verbs_devs)) {
		dlist_pop_front(verbs_devs, struct verbs_dev_info, dev, entry);
		while (!dlist_empty(&dev->addrs)) {
			dlist_pop_front(&dev->addrs, struct verbs_addr, addr, entry);
			rdma_freeaddrinfo(addr->rai);
			free(addr);
		}
		free(dev->name);
		free(dev);
	}
}

static int fi_ibv_add_rai(struct dlist_entry *verbs_devs, struct rdma_cm_id *id,
		struct rdma_addrinfo *rai)
{
	struct verbs_dev_info *dev;
	struct verbs_addr *addr;
	const char *dev_name;

	if (!(addr = malloc(sizeof(*addr))))
		return -FI_ENOMEM;

	addr->rai = rai;

	dev_name = ibv_get_device_name(id->verbs->device);
	dlist_foreach_container(verbs_devs, struct verbs_dev_info, dev, entry)
		if (!strcmp(dev_name, dev->name))
			goto add_rai;

	if (!(dev = malloc(sizeof(*dev))))
		goto err1;

	if (!(dev->name = strdup(dev_name)))
		goto err2;

	dlist_init(&dev->addrs);
	dlist_insert_tail(&dev->entry, verbs_devs);
add_rai:
	dlist_insert_tail(&addr->entry, &dev->addrs);
	return 0;
err2:
	free(dev);
err1:
	free(addr);
	return -FI_ENOMEM;
}

/* Builds a list of interfaces that correspond to active verbs devices */
static int fi_ibv_getifaddrs(struct dlist_entry *verbs_devs)
{
	struct ifaddrs *ifaddr, *ifa;
	char name[INET6_ADDRSTRLEN];
	struct rdma_addrinfo *rai;
	struct rdma_cm_id *id;
	const char *ret_ptr;
	char *iface = fi_ibv_gl_data.iface;
	int ret, num_verbs_ifs = 0;
	size_t iface_len = 0;
	int exact_match = 0;

	ret = ofi_getifaddrs(&ifaddr);
	if (ret) {
		VERBS_WARN(FI_LOG_FABRIC,
			   "Unable to get interface addresses\n");
		return ret;
	}

	/* select best iface name based on user's input */
	if (iface) {
		iface_len = strlen(iface);
		if (iface_len > IFNAMSIZ) {
			VERBS_INFO(FI_LOG_EP_CTRL,
				   "Too long iface name: %s, max: %d\n",
				   iface, IFNAMSIZ);

		}
		for (ifa = ifaddr; ifa && !exact_match; ifa = ifa->ifa_next)
			exact_match = !strcmp(ifa->ifa_name, iface);
	}

	for (ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
		if (!ifa->ifa_addr || !(ifa->ifa_flags & IFF_UP) ||
				!strcmp(ifa->ifa_name, "lo"))
			continue;

		if (iface) {
			if (exact_match) {
				if (strcmp(ifa->ifa_name, iface))
					continue;
			} else {
				if (strncmp(ifa->ifa_name, iface, iface_len))
					continue;
			}
		}

		switch (ifa->ifa_addr->sa_family) {
		case AF_INET:
			ret_ptr = inet_ntop(AF_INET, &ofi_sin_addr(ifa->ifa_addr),
				name, INET6_ADDRSTRLEN);
			break;
		case AF_INET6:
			ret_ptr = inet_ntop(AF_INET6, &ofi_sin6_addr(ifa->ifa_addr),
				name, INET6_ADDRSTRLEN);
			break;
		default:
			continue;
		}
		if (!ret_ptr) {
			VERBS_WARN(FI_LOG_FABRIC,
				   "inet_ntop failed: %s(%d)\n",
				   strerror(errno), errno);
			ret = -errno;
			goto err1;
		}

		ret = fi_ibv_create_ep(name, NULL, FI_NUMERICHOST | FI_SOURCE,
				NULL, &rai, &id);
		if (ret)
			continue;

		ret = fi_ibv_add_rai(verbs_devs, id, rai);
		if (ret)
			goto err2;

		VERBS_DBG(FI_LOG_FABRIC, "Found active interface for verbs device: "
			  "%s with address: %s\n",
			  ibv_get_device_name(id->verbs->device), name);

		rdma_destroy_ep(id);

		num_verbs_ifs++;
	}
	freeifaddrs(ifaddr);
	return num_verbs_ifs ? 0 : -FI_ENODATA;
err2:
	rdma_destroy_ep(id);
err1:
	fi_ibv_verbs_devs_free(verbs_devs);
	freeifaddrs(ifaddr);
	return ret;
}

static int fi_ibv_get_srcaddr_devs(struct fi_info **info)
{
	struct fi_info *fi, *add_info;
	struct fi_info *fi_unconf = NULL, *fi_prev = NULL;
	struct verbs_dev_info *dev;
	struct verbs_addr *addr;
	int ret = 0;

	DEFINE_LIST(verbs_devs);

	ret = fi_ibv_getifaddrs(&verbs_devs);
	if (ret)
		return ret;

	if (dlist_empty(&verbs_devs)) {
		VERBS_WARN(FI_LOG_CORE, "No interface address found\n");
		return 0;
	}

	for (fi = *info; fi; fi = fi->next) {
		if (fi->ep_attr->type == FI_EP_DGRAM)
			continue;
		dlist_foreach_container(&verbs_devs, struct verbs_dev_info, dev, entry)
			if (!strncmp(fi->domain_attr->name, dev->name, strlen(dev->name))) {
				dlist_foreach_container(&dev->addrs, struct verbs_addr, addr, entry) {
					/* When a device has multiple interfaces/addresses configured
					 * duplicate fi_info and add the address info. fi->src_addr
					 * would have been set in the previous iteration */
					if (fi->src_addr) {
						if (!(add_info = fi_dupinfo(fi))) {
							ret = -FI_ENOMEM;
							goto out;
						}

						add_info->next = fi->next;
						fi->next = add_info;
						fi = add_info;
					}

					ret = fi_ibv_rai_to_fi(addr->rai, fi);
					if (ret)
						goto out;
				}
				break;
			}
	}

        /* re-order info: move info without src_addr to tail */
	for (fi = *info; fi;) {
		if (!fi->src_addr) {
			/* re-link list - exclude current element */
			if (fi == *info) {
				*info = fi->next;
				fi->next = fi_unconf;
				fi_unconf = fi;
				fi = *info;
			} else {
				assert(fi_prev);
				fi_prev->next = fi->next;
				fi->next = fi_unconf;
				fi_unconf = fi;
				fi = fi_prev->next;
			}
		} else {
			fi_prev = fi;
			fi = fi->next;
		}
	}

	/* append excluded elements to tail of list */
	if (fi_unconf) {
		if (fi_prev) {
			assert(!fi_prev->next);
			fi_prev->next = fi_unconf;
		} else if (*info) {
			assert(!(*info)->next);
			(*info)->next = fi_unconf;
		} else /* !(*info) */ {
			(*info) = fi_unconf;
		}
	}

out:
	fi_ibv_verbs_devs_free(&verbs_devs);
	return ret;
}

static void fi_ibv_sockaddr_set_port(struct sockaddr *sa, uint16_t port)
{
	switch(sa->sa_family) {
	case AF_INET:
		((struct sockaddr_in *)sa)->sin_port = port;
		break;
	case AF_INET6:
		((struct sockaddr_in6 *)sa)->sin6_port = port;
		break;
	}
}

/* the `rai` parameter is used for the MSG/RDM EP types */
/* the `fmt`, `[src | dest]_addr` parameters are used for the DGRAM EP type */
/* if the `fmt` parameter isn't used, pass FI_FORMAT_UNSPEC */
static int fi_ibv_set_info_addrs(struct fi_info *info,
				 struct rdma_addrinfo *rai,
				 uint32_t fmt,
				 struct ofi_ib_ud_ep_name *src_addr,
				 struct ofi_ib_ud_ep_name *dest_addr)
{
	struct fi_info *iter_info = info;
	int ret;

	for (; iter_info; iter_info = iter_info->next) {
		if (iter_info->ep_attr->type != FI_EP_DGRAM) {
			ret = fi_ibv_rai_to_fi(rai, iter_info);
			if (ret)
				return ret;
		} else {
			iter_info->addr_format = fmt;
			if (src_addr) {
				ret = fi_ibv_fill_addr_by_ep_name(src_addr, fmt,
								  &iter_info->src_addr,
								  &iter_info->src_addrlen);
				if (ret)
					return ret;
			}
			if (dest_addr) {
				ret = fi_ibv_fill_addr_by_ep_name(dest_addr, fmt,
								  &iter_info->dest_addr,
								  &iter_info->dest_addrlen);
				if (ret)
					return ret;
			}
		}
	}

	return FI_SUCCESS;
}

static int fi_ibv_fill_addr(struct rdma_addrinfo *rai, struct fi_info **info,
			    struct rdma_cm_id *id)
{
	struct sockaddr *local_addr;

	/*
	 * TODO MPICH CH3 doesn't work with verbs provider without skipping the
	 * loopback address. An alternative approach if there is one is needed
	 * to allow both.
	 */
	if (rai->ai_src_addr && !ofi_is_loopback_addr(rai->ai_src_addr))
		goto rai_to_fi;

	if (!id->verbs)
		return fi_ibv_get_srcaddr_devs(info);

	/* Handle the case when rdma_cm doesn't fill src address even
	 * though it fills the destination address (presence of id->verbs
	 * corresponds to a valid dest addr) */
	local_addr = rdma_get_local_addr(id);
	if (!local_addr) {
		VERBS_WARN(FI_LOG_CORE,
			   "Unable to get local address\n");
		return -FI_ENODATA;
	}

	rai->ai_src_len = fi_ibv_sockaddr_len(local_addr);
	if (!(rai->ai_src_addr = malloc(rai->ai_src_len)))
		return -FI_ENOMEM;

	memcpy(rai->ai_src_addr, local_addr, rai->ai_src_len);
	/* User didn't specify a port. Zero out the random port
	 * assigned by rdmamcm so that this rai/fi_info can be
	 * used multiple times to create rdma endpoints.*/
	fi_ibv_sockaddr_set_port(rai->ai_src_addr, 0);

rai_to_fi:
	return fi_ibv_set_info_addrs(*info, rai, FI_FORMAT_UNSPEC,
				     NULL, NULL);
}

int fi_ibv_init_info(const struct fi_info **all_infos)
{
	struct ibv_context **ctx_list;
	struct fi_info *fi = NULL, *tail = NULL;
	int ret = 0, i, num_devices;

	*all_infos = NULL;

	if (!fi_ibv_gl_data.fork_unsafe) {
		VERBS_INFO(FI_LOG_CORE, "Enabling IB fork support\n");
		ret = ibv_fork_init();
		if (ret) {
			VERBS_WARN(FI_LOG_CORE,
				   "Enabling IB fork support failed: %s (%d)\n",
				   strerror(ret), ret);
			goto done;
		}
	} else {
		VERBS_INFO(FI_LOG_CORE, "Not enabling IB fork support\n");
	}

	if (!fi_ibv_have_device()) {
		VERBS_INFO(FI_LOG_FABRIC, "No RDMA devices found\n");
		ret = -FI_ENODATA;
		goto done;
	}

	ctx_list = rdma_get_devices(&num_devices);
	if (!num_devices) {
		VERBS_INFO_ERRNO(FI_LOG_FABRIC, "rdma_get_devices", errno);
		ret = -errno;
		goto done;
	}

	for (i = 0; i < num_devices; i++) {
		ret = fi_ibv_alloc_info(ctx_list[i], &fi, &verbs_msg_domain);
		if (!ret) {
			if (!*all_infos)
				*all_infos = fi;
			else
				tail->next = fi;
			tail = fi;

			ret = fi_ibv_alloc_info(ctx_list[i], &fi,
						&verbs_rdm_domain);
			if (!ret) {
				tail->next = fi;
				tail = fi;
			}

			ret = fi_ibv_alloc_info(ctx_list[i], &fi,
						&verbs_dgram_domain);
			if (!ret) {
				tail->next = fi;
				tail = fi;
			}
		}
	}

	/* note we are possibly discarding ENOMEM */
	ret = *all_infos ? 0 : ret;

	rdma_free_devices(ctx_list);
done:
	return ret;
}

static int fi_ibv_set_default_attr(struct fi_info *info, size_t *attr,
				   size_t default_attr, char *attr_str)
{
	if (default_attr > *attr) {
		VERBS_WARN(FI_LOG_FABRIC, "Ignoring provider default value "
			   "for %s as it is greater than the value supported "
			   "by domain: %s\n", attr_str, info->domain_attr->name);
	} else {
		*attr = default_attr;
	}
	return 0;
}

/* Set default values for attributes. ofi_alter_info would change them if the
 * user has asked for a different value in hints */
static int fi_ibv_set_default_info(struct fi_info *info)
{
	int ret;

	ret = fi_ibv_set_default_attr(info, &info->tx_attr->size,
				      fi_ibv_gl_data.def_tx_size,
				      "tx context size");
	if (ret)
		return ret;

	ret = fi_ibv_set_default_attr(info, &info->rx_attr->size,
				      fi_ibv_gl_data.def_rx_size,
				      "rx context size");
	if (ret)
		return ret;

	/* Don't set defaults for verb/RDM as
	 * it supports an iov limit of just 1 */
	if (info->ep_attr->type != FI_EP_RDM) {
		ret = fi_ibv_set_default_attr(
			info, &info->tx_attr->iov_limit,
			fi_ibv_gl_data.def_tx_iov_limit,
			"tx iov_limit");
		if (ret)
			return ret;

		ret = fi_ibv_set_default_attr(
			info, &info->rx_attr->iov_limit,
			fi_ibv_gl_data.def_rx_iov_limit,
			"rx iov_limit");
		if (ret)
			return ret;

		if (info->ep_attr->type != FI_EP_DGRAM) {
			/* For verbs iov limit is same for
			 * both regular messages and RMA */
			ret = fi_ibv_set_default_attr(
				info, &info->tx_attr->rma_iov_limit,
				fi_ibv_gl_data.def_tx_iov_limit,
				"tx rma_iov_limit");
			if (ret)
				return ret;
		}
	}
	return 0;
}

static struct fi_info *fi_ibv_get_passive_info(const struct fi_info *prov_info,
					       const struct fi_info *hints)
{
	struct fi_info *info;

	if (!(info = fi_dupinfo(hints)))
		return NULL;

	info->mode = prov_info->mode;
	info->tx_attr->mode = prov_info->tx_attr->mode;
	info->rx_attr->mode = prov_info->rx_attr->mode;
	info->ep_attr->type = prov_info->ep_attr->type;

	info->domain_attr->domain 	= prov_info->domain_attr->domain;
	if (!info->domain_attr->name)
		info->domain_attr->name = strdup(VERBS_ANY_DOMAIN);
	info->domain_attr->mr_mode 	= prov_info->domain_attr->mr_mode;
	info->domain_attr->mode 	= prov_info->domain_attr->mode;

	info->fabric_attr->fabric = prov_info->fabric_attr->fabric;
	if (!info->fabric_attr->name)
		info->fabric_attr->name = strdup(VERBS_ANY_FABRIC);

	/* prov_name is set by libfabric core */
	free(info->fabric_attr->prov_name);
	info->fabric_attr->prov_name = NULL;
	return info;
}

static int fi_ibv_get_matching_info(uint32_t version,
				    const struct fi_info *hints,
				    struct fi_info **info,
				    const struct fi_info *verbs_info,
				    uint8_t passive)
{
	const struct fi_info *check_info = verbs_info;
	struct fi_info *fi, *tail;
	int ret;
	uint8_t got_passive_info = 0;

	*info = tail = NULL;

	for ( ; check_info; check_info = check_info->next) {
		if (hints) {
			VERBS_DBG(FI_LOG_FABRIC, "Checking domain: %s\n",
				  check_info->domain_attr->name);
			ret = fi_ibv_check_hints(version, hints,
						 check_info);
			if (ret)
				continue;
		}

		if ((check_info->ep_attr->type == FI_EP_MSG) && passive) {
			if (got_passive_info)
				continue;

			if (!(fi = fi_ibv_get_passive_info(check_info, hints))) {
				ret = -FI_ENOMEM;
				goto err;
			}
			got_passive_info = 1;
		} else {
			if (!(fi = fi_dupinfo(check_info))) {
				ret = -FI_ENOMEM;
				goto err;
			}
			ret = fi_ibv_set_default_info(fi);
			if (ret) {
				fi_freeinfo(fi);
				continue;
			}
		}

		if (!*info)
			*info = fi;
		else
			tail->next = fi;
		tail = fi;
	}

	if (!*info)
		return -FI_ENODATA;

	return FI_SUCCESS;
err:
	fi_freeinfo(*info);
	return ret;
}

static int fi_ibv_del_info_not_belong_to_dev(const char *dev_name, struct fi_info **info)
{
	struct fi_info *check_info = *info;
	struct fi_info *cur, *prev = NULL;

	*info = NULL;

	while (check_info) {
		/* Use strncmp since verbs RDM domain name
		 * would have "-rdm" suffix */
		if (dev_name && strncmp(dev_name, check_info->domain_attr->name,
					strlen(dev_name))) {
			/* This branch removing `check_info` entry from the list */
			cur = check_info;
			if (prev)
				prev->next = check_info->next;
			check_info = check_info->next;

			cur->next = NULL;
			fi_freeinfo(cur);
		} else {
			prev = check_info;
			if (!*info)
				/* if find the first matched `fi_info` entry,
				 * then save this to original list */
				*info = check_info;
			check_info = check_info->next;
		}
	}

	if (!*info)
		return -FI_ENODATA;

	return FI_SUCCESS;
}

static int fi_ibv_resolve_ib_ud_dest_addr(const char *node, const char *service,
					  struct ofi_ib_ud_ep_name **dest_addr)
{
	int svc = VERBS_IB_UD_NS_ANY_SERVICE;
	struct util_ns ns = {
		.port = fi_ibv_gl_data.dgram.name_server_port,
		.name_len = sizeof(**dest_addr),
		.service_len = sizeof(svc),
		.service_cmp = fi_ibv_dgram_ns_service_cmp,
		.is_service_wildcard = fi_ibv_dgram_ns_is_service_wildcard,
	};

	ofi_ns_init(&ns);

	if (service)
		svc = atoi(service);
	*dest_addr = (struct ofi_ib_ud_ep_name *)
		ofi_ns_resolve_name(&ns, node, &svc);
	if (*dest_addr) {
		VERBS_INFO_NODE_2_UD_ADDR(FI_LOG_CORE, node, svc, *dest_addr);
	} else {
		VERBS_INFO(FI_LOG_CORE,
			   "failed to resolve '%s:%u'.\n", node, svc);
		return -FI_ENODATA;
	}

	return 0;
}

static int fi_ibv_handle_ib_ud_addr(const char *node, const char *service,
				    uint64_t flags, struct fi_info **info)
{
	struct ofi_ib_ud_ep_name *dest_addr = NULL;
	struct ofi_ib_ud_ep_name *src_addr = NULL;
	void *addr = NULL;
	size_t len = 0;
	uint32_t fmt = FI_FORMAT_UNSPEC;
	int svc = VERBS_IB_UD_NS_ANY_SERVICE, ret = FI_SUCCESS;

	if (node && !ofi_str_toaddr(node, &fmt, &addr, &len) &&
	    fmt == FI_ADDR_IB_UD) {
		if (flags & FI_SOURCE) {
			src_addr = addr;
			VERBS_INFO_NODE_2_UD_ADDR(FI_LOG_CORE, node,
						  svc, src_addr);
		} else {
			dest_addr = addr;
			VERBS_INFO_NODE_2_UD_ADDR(FI_LOG_CORE, node,
						  svc, dest_addr);
		}
		node = NULL;
	}

	if (!src_addr) {
		src_addr = calloc(1, sizeof(*src_addr));
		if (!src_addr) {
			VERBS_INFO(FI_LOG_CORE,
			           "failed to allocate src addr.\n");
			return -FI_ENODATA;
		}

		if (flags & FI_SOURCE) {
			if (service)
				sscanf(service, "%" SCNu16, &src_addr->service);
			VERBS_INFO(FI_LOG_CORE, "node '%s' service '%s' "
				                "converted to <service=%d>\n",
				   node, service, src_addr->service);
		}
	}

	if (!dest_addr && node && !(flags & FI_SOURCE)) {
		ret = fi_ibv_resolve_ib_ud_dest_addr(node, service, &dest_addr);
		if (ret)
			goto fn2; /* Here possible that `src_addr` isn't a NULL */
	}

	ret = fi_ibv_set_info_addrs(*info, NULL, fmt, src_addr, dest_addr);
	if  (ret)
		goto fn2;

	/* `fi_info::src_addr` and `fi_info::dest_addr` is freed
	 * in the `fi_freeinfo` function in case of failure */
fn2:
	if (src_addr)
		free(src_addr);
	if (dest_addr)
		free(dest_addr);
	return ret;
}

static void fi_ibv_remove_nosrc_info(struct fi_info **info)
{
	struct fi_info **fi = info, *next;
	while (*fi && ((*fi)->ep_attr->type == FI_EP_MSG)) {
		if (!(*fi)->src_addr) {
			VERBS_INFO(FI_LOG_FABRIC, "Not reporting fi_info "
				   "corresponding to domain: %s as it has no IP"
				   "address configured\n",
				   (*fi)->domain_attr->name);
			next = (*fi)->next;
			(*fi)->next = NULL;
			fi_freeinfo(*fi);
			*fi = next;
		} else {
			fi = &(*fi)->next;
		}
	}
}

static int fi_ibv_handle_sock_addr(const char *node, const char *service,
				   uint64_t flags, const struct fi_info *hints,
				   struct fi_info **info)
{
	struct rdma_cm_id *id = NULL;
	struct rdma_addrinfo *rai;
	const char *dev_name = NULL;
	int ret;

	ret = fi_ibv_create_ep(node, service, flags, hints, &rai, &id);
	if (ret)
		return ret;
	if (id->verbs) {
		dev_name = ibv_get_device_name(id->verbs->device);
		ret = fi_ibv_del_info_not_belong_to_dev(dev_name, info);
		if (ret)
			goto fn;
	}

	ret = fi_ibv_fill_addr(rai, info, id);
	fi_ibv_remove_nosrc_info(info);
fn:
	fi_ibv_destroy_ep(rai, &id);
	return ret;
}


static int fi_ibv_get_match_infos(uint32_t version, const char *node,
				  const char *service, uint64_t flags,
				  const struct fi_info *hints,
				  const struct fi_info **raw_info,
				  struct fi_info **info)
{
	int ret, ret_sock_addr, ret_ib_ud_addr;

	// TODO check for AF_IB addr
	ret = fi_ibv_get_matching_info(version, hints, info, *raw_info,
				       ofi_is_wildcard_listen_addr(node, service,
								   flags, hints));
	if (ret)
		return ret;

	if (hints && (hints->addr_format == FI_ADDR_IB_UD)) {
		/* This is case when only IB UD addresses are passed */
		ret = fi_ibv_handle_ib_ud_addr(node, service, flags, info);
		if (ret)
			fi_freeinfo(*info);
		return ret;
	}

	ret_sock_addr = fi_ibv_handle_sock_addr(node, service, flags, hints, info);
	if (ret_sock_addr)
		VERBS_INFO(FI_LOG_CORE, "Handling of the socket address fails - %d\n",
			   ret_sock_addr);

	if (!*info)
		return -FI_ENODATA;

	ret_ib_ud_addr = fi_ibv_handle_ib_ud_addr(node, service, flags, info);
	if (ret_ib_ud_addr)
		VERBS_INFO(FI_LOG_CORE, "Handling of the IB ID address fails - %d\n",
			   ret_ib_ud_addr);

	if (ret_sock_addr && ret_ib_ud_addr) {
		/* neither the sockaddr nor the ib_ud address wasn't
		 * handled to satisfy the selection procedure */
		VERBS_INFO(FI_LOG_CORE, "Handling of the addresses fails, "
			   "the getting infos is unsuccessful\n");
		fi_freeinfo(*info);
		return -FI_ENODATA;
	}

	return FI_SUCCESS;
}

int fi_ibv_getinfo(uint32_t version, const char *node, const char *service,
		   uint64_t flags, const struct fi_info *hints,
		   struct fi_info **info)
{
	int ret;
	const struct fi_info *cur;

	ret = fi_ibv_get_match_infos(version, node, service,
				     flags, hints,
				     &fi_ibv_util_prov.info, info);
	if (ret)
		goto out;

	ofi_alter_info(*info, hints, version);

	if (!ofi_check_rx_mode(hints, FI_RX_CQ_DATA)) {
		for (cur = *info; cur; cur = cur->next)
			cur->domain_attr->cq_data_size = 0;
	}

	if (!hints || !hints->tx_attr || !hints->tx_attr->inject_size) {
		for (cur = *info; cur; cur = cur->next) {
			if (cur->ep_attr->type != FI_EP_MSG)
				continue;
			/* The default inline size is usually smaller.
			 * This is to avoid drop in throughput */
			cur->tx_attr->inject_size = MIN(cur->tx_attr->inject_size,
							fi_ibv_gl_data.def_inline_size);
		}
	}
out:
	if (!ret || ret == -FI_ENOMEM || ret == -FI_ENODEV)
		return ret;
	else
		return -FI_ENODATA;
}
