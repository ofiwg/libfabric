/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <limits.h>

#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

const struct fi_ep_attr cxi_rdm_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_CXI,
	.protocol_version = CXI_WIRE_PROTO_VERSION,
	.max_msg_size = CXI_EP_MAX_MSG_SZ,
	.msg_prefix_size = CXI_EP_MSG_PREFIX_SZ,
	.max_order_raw_size = CXI_EP_MAX_ORDER_RAW_SZ,
	.max_order_war_size = CXI_EP_MAX_ORDER_WAR_SZ,
	.max_order_waw_size = CXI_EP_MAX_ORDER_WAW_SZ,
	.mem_tag_format = CXI_EP_MEM_TAG_FMT,
	.tx_ctx_cnt = CXI_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXI_EP_MAX_RX_CNT,
};

const struct fi_tx_attr cxi_rdm_tx_attr = {
	.caps = CXI_EP_RDM_CAP_BASE,
	.mode = CXI_MODE,
	.op_flags = CXI_EP_DEFAULT_OP_FLAGS,
	.msg_order = CXI_EP_MSG_ORDER,
	.inject_size = CXI_EP_MAX_INJECT_SZ,
	.size = CXI_EP_TX_SZ,
	.iov_limit = CXI_EP_MAX_IOV_LIMIT,
	.rma_iov_limit = CXI_EP_MAX_IOV_LIMIT,
};

const struct fi_rx_attr cxi_rdm_rx_attr = {
	.caps = CXI_EP_RDM_CAP_BASE,
	.mode = CXI_MODE,
	.op_flags = 0,
	.msg_order = CXI_EP_MSG_ORDER,
	.comp_order = CXI_EP_COMP_ORDER,
	.total_buffered_recv = CXI_EP_MAX_BUFF_RECV,
	.size = CXI_EP_RX_SZ,
	.iov_limit = CXI_EP_MAX_IOV_LIMIT,
};

static int cxi_rdm_verify_rx_attr(const struct fi_rx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | CXI_EP_RDM_CAP) != CXI_EP_RDM_CAP) {
		CXI_LOG_DBG("Unsupported RDM rx caps\n");
		return -FI_ENODATA;
	}

	if ((attr->msg_order | CXI_EP_MSG_ORDER) != CXI_EP_MSG_ORDER) {
		CXI_LOG_DBG("Unsuported rx message order\n");
		return -FI_ENODATA;
	}

	if ((attr->comp_order | CXI_EP_COMP_ORDER) != CXI_EP_COMP_ORDER) {
		CXI_LOG_DBG("Unsuported rx completion order\n");
		return -FI_ENODATA;
	}

	if (attr->total_buffered_recv > cxi_rdm_rx_attr.total_buffered_recv) {
		CXI_LOG_DBG("Buffered receive size too large\n");
		return -FI_ENODATA;
	}

	if (attr->size > cxi_rdm_rx_attr.size) {
		CXI_LOG_DBG("Rx size too large\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > cxi_rdm_rx_attr.iov_limit) {
		CXI_LOG_DBG("Rx iov limit too large\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int cxi_rdm_verify_tx_attr(const struct fi_tx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | CXI_EP_RDM_CAP) != CXI_EP_RDM_CAP) {
		CXI_LOG_DBG("Unsupported RDM tx caps\n");
		return -FI_ENODATA;
	}

	if ((attr->msg_order | CXI_EP_MSG_ORDER) != CXI_EP_MSG_ORDER) {
		CXI_LOG_DBG("Unsupported tx message order\n");
		return -FI_ENODATA;
	}

	if (attr->inject_size > cxi_rdm_tx_attr.inject_size) {
		CXI_LOG_DBG("Inject size too large\n");
		return -FI_ENODATA;
	}

	if (attr->size > cxi_rdm_tx_attr.size) {
		CXI_LOG_DBG("Tx size too large\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > cxi_rdm_tx_attr.iov_limit) {
		CXI_LOG_DBG("Tx iov limit too large\n");
		return -FI_ENODATA;
	}

	if (attr->rma_iov_limit > cxi_rdm_tx_attr.rma_iov_limit) {
		CXI_LOG_DBG("RMA iov limit too large\n");
		return -FI_ENODATA;
	}

	return 0;
}

int cxi_rdm_verify_ep_attr(const struct fi_ep_attr *ep_attr,
			   const struct fi_tx_attr *tx_attr,
			   const struct fi_rx_attr *rx_attr)
{
	int ret;

	if (ep_attr) {
		switch (ep_attr->protocol) {
		case FI_PROTO_UNSPEC:
		case FI_PROTO_CXI:
			break;
		default:
			CXI_LOG_DBG("Unsupported protocol\n");
			return -FI_ENODATA;
		}

		if (ep_attr->protocol_version &&
		    (ep_attr->protocol_version !=
		     cxi_rdm_ep_attr.protocol_version)) {
			CXI_LOG_DBG("Invalid protocol version\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_msg_size > cxi_rdm_ep_attr.max_msg_size) {
			CXI_LOG_DBG("Message size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->msg_prefix_size >
		    cxi_rdm_ep_attr.msg_prefix_size) {
			CXI_LOG_DBG("Msg prefix size not supported\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_raw_size >
		    cxi_rdm_ep_attr.max_order_raw_size) {
			CXI_LOG_DBG("RAW order size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_war_size >
		    cxi_rdm_ep_attr.max_order_war_size) {
			CXI_LOG_DBG("WAR order size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_waw_size >
		    cxi_rdm_ep_attr.max_order_waw_size) {
			CXI_LOG_DBG("WAW order size too large\n");
			return -FI_ENODATA;
		}

		if ((ep_attr->tx_ctx_cnt > CXI_EP_MAX_TX_CNT) &&
		    ep_attr->tx_ctx_cnt != FI_SHARED_CONTEXT)
			return -FI_ENODATA;

		if ((ep_attr->rx_ctx_cnt > CXI_EP_MAX_RX_CNT) &&
		    ep_attr->rx_ctx_cnt != FI_SHARED_CONTEXT)
			return -FI_ENODATA;
	}

	ret = cxi_rdm_verify_tx_attr(tx_attr);
	if (ret)
		return ret;

	ret = cxi_rdm_verify_rx_attr(rx_attr);
	if (ret)
		return ret;

	return 0;
}

int cxi_rdm_fi_info(uint32_t version, void *src_addr, void *dest_addr,
		    const struct fi_info *hints, struct fi_info **info)
{
	*info = cxi_fi_info(version, FI_EP_RDM, hints, src_addr, dest_addr);
	if (!*info)
		return -FI_ENOMEM;

	*(*info)->tx_attr = cxi_rdm_tx_attr;
	(*info)->tx_attr->size = cxi_rdm_tx_attr.size;
	*(*info)->rx_attr = cxi_rdm_rx_attr;
	(*info)->rx_attr->size = cxi_rdm_rx_attr.size;
	*(*info)->ep_attr = cxi_rdm_ep_attr;

	if (hints && hints->ep_attr) {
		if (hints->ep_attr->rx_ctx_cnt)
			(*info)->ep_attr->rx_ctx_cnt =
					hints->ep_attr->rx_ctx_cnt;
		if (hints->ep_attr->tx_ctx_cnt)
			(*info)->ep_attr->tx_ctx_cnt =
					hints->ep_attr->tx_ctx_cnt;
	}

	if (hints && hints->rx_attr) {
		(*info)->rx_attr->op_flags |= hints->rx_attr->op_flags;
		if (hints->rx_attr->caps)
			(*info)->rx_attr->caps = CXI_EP_RDM_SEC_CAP |
							hints->rx_attr->caps;
	}

	if (hints && hints->tx_attr) {
		(*info)->tx_attr->op_flags |= hints->tx_attr->op_flags;
		if (hints->tx_attr->caps)
			(*info)->tx_attr->caps = CXI_EP_RDM_SEC_CAP |
							hints->tx_attr->caps;
	}

	(*info)->caps = CXI_EP_RDM_CAP |
			(*info)->rx_attr->caps | (*info)->tx_attr->caps;
	if (hints && hints->caps) {
		(*info)->caps = CXI_EP_RDM_SEC_CAP | hints->caps;
		(*info)->rx_attr->caps = CXI_EP_RDM_SEC_CAP |
			((*info)->rx_attr->caps & (*info)->caps);
		(*info)->tx_attr->caps = CXI_EP_RDM_SEC_CAP |
			((*info)->tx_attr->caps & (*info)->caps);
	}
	return 0;
}

static int cxi_rdm_endpoint(struct fid_domain *domain, struct fi_info *info,
			    struct cxi_ep **ep, void *context, size_t fclass)
{
	int ret;

	if (info) {
		if (info->ep_attr) {
			ret = cxi_rdm_verify_ep_attr(info->ep_attr,
						     info->tx_attr,
						     info->rx_attr);
			if (ret)
				return -FI_EINVAL;
		}

		if (info->tx_attr) {
			ret = cxi_rdm_verify_tx_attr(info->tx_attr);
			if (ret)
				return -FI_EINVAL;
		}

		if (info->rx_attr) {
			ret = cxi_rdm_verify_rx_attr(info->rx_attr);
			if (ret)
				return -FI_EINVAL;
		}
	}

	ret = cxi_alloc_endpoint(domain, info, ep, context, fclass);
	if (ret)
		return ret;

	if (!info || !info->ep_attr)
		(*ep)->attr->ep_attr = cxi_rdm_ep_attr;

	if (!info || !info->tx_attr)
		(*ep)->tx_attr = cxi_rdm_tx_attr;

	if (!info || !info->rx_attr)
		(*ep)->rx_attr = cxi_rdm_rx_attr;

	return 0;
}

int cxi_rdm_ep(struct fid_domain *domain, struct fi_info *info,
	       struct fid_ep **ep, void *context)
{
	int ret;
	struct cxi_ep *endpoint;

	ret = cxi_rdm_endpoint(domain, info, &endpoint, context, FI_CLASS_EP);
	if (ret)
		return ret;

	*ep = &endpoint->ep;
	return 0;
}

int cxi_rdm_sep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **sep, void *context)
{
	int ret;
	struct cxi_ep *endpoint;

	ret = cxi_rdm_endpoint(domain, info, &endpoint, context, FI_CLASS_SEP);
	if (ret)
		return ret;

	*sep = &endpoint->ep;

	return 0;
}

