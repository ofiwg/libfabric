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

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

const struct fi_ep_attr cxip_rdm_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_CXI,
	.protocol_version = CXIP_WIRE_PROTO_VERSION,
	.max_msg_size = CXIP_EP_MAX_MSG_SZ,
	.msg_prefix_size = CXIP_EP_MSG_PREFIX_SZ,
	.max_order_raw_size = CXIP_EP_MAX_ORDER_RAW_SZ,
	.max_order_war_size = CXIP_EP_MAX_ORDER_WAR_SZ,
	.max_order_waw_size = CXIP_EP_MAX_ORDER_WAW_SZ,
	.mem_tag_format = CXIP_EP_MEM_TAG_FMT,
	.tx_ctx_cnt = CXIP_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXIP_EP_MAX_RX_CNT,
};

const struct fi_tx_attr cxip_rdm_tx_attr = {
	.caps = CXIP_EP_RDM_CAP_BASE,
	.mode = CXIP_MODE,
	.op_flags = CXIP_EP_DEFAULT_OP_FLAGS,
	.msg_order = CXIP_EP_MSG_ORDER,
	.inject_size = CXIP_EP_MAX_INJECT_SZ,
	.size = CXIP_EP_TX_SZ,
	.iov_limit = CXIP_EP_MAX_IOV_LIMIT,
	.rma_iov_limit = CXIP_EP_MAX_IOV_LIMIT,
};

const struct fi_rx_attr cxip_rdm_rx_attr = {
	.caps = CXIP_EP_RDM_CAP_BASE,
	.mode = CXIP_MODE,
	.op_flags = 0,
	.msg_order = CXIP_EP_MSG_ORDER,
	.comp_order = CXIP_EP_COMP_ORDER,
	.total_buffered_recv = CXIP_EP_MAX_BUFF_RECV,
	.size = CXIP_EP_RX_SZ,
	.iov_limit = CXIP_EP_MAX_IOV_LIMIT,
};

static int cxip_rdm_verify_rx_attr(const struct fi_rx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | CXIP_EP_RDM_CAP) != CXIP_EP_RDM_CAP) {
		CXIP_LOG_DBG("Unsupported RDM rx caps\n");
		return -FI_ENODATA;
	}

	if ((attr->msg_order | CXIP_EP_MSG_ORDER) != CXIP_EP_MSG_ORDER) {
		CXIP_LOG_DBG("Unsupported rx message order\n");
		return -FI_ENODATA;
	}

	if ((attr->comp_order | CXIP_EP_COMP_ORDER) != CXIP_EP_COMP_ORDER) {
		CXIP_LOG_DBG("Unsupported rx completion order\n");
		return -FI_ENODATA;
	}

	if (attr->total_buffered_recv > cxip_rdm_rx_attr.total_buffered_recv) {
		CXIP_LOG_DBG("Buffered receive size too large\n");
		return -FI_ENODATA;
	}

	if (attr->size > cxip_rdm_rx_attr.size) {
		CXIP_LOG_DBG("Rx size too large\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > cxip_rdm_rx_attr.iov_limit) {
		CXIP_LOG_DBG("Rx iov limit too large\n");
		return -FI_ENODATA;
	}

	return 0;
}

static int cxip_rdm_verify_tx_attr(const struct fi_tx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | CXIP_EP_RDM_CAP) != CXIP_EP_RDM_CAP) {
		CXIP_LOG_DBG("Unsupported RDM tx caps\n");
		return -FI_ENODATA;
	}

	if ((attr->msg_order | CXIP_EP_MSG_ORDER) != CXIP_EP_MSG_ORDER) {
		CXIP_LOG_DBG("Unsupported tx message order\n");
		return -FI_ENODATA;
	}

	if (attr->inject_size > cxip_rdm_tx_attr.inject_size) {
		CXIP_LOG_DBG("Inject size too large\n");
		return -FI_ENODATA;
	}

	if (attr->size > cxip_rdm_tx_attr.size) {
		CXIP_LOG_DBG("Tx size too large\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > cxip_rdm_tx_attr.iov_limit) {
		CXIP_LOG_DBG("Tx iov limit too large\n");
		return -FI_ENODATA;
	}

	if (attr->rma_iov_limit > cxip_rdm_tx_attr.rma_iov_limit) {
		CXIP_LOG_DBG("RMA iov limit too large\n");
		return -FI_ENODATA;
	}

	return 0;
}

int cxip_rdm_verify_ep_attr(const struct fi_ep_attr *ep_attr,
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
			CXIP_LOG_DBG("Unsupported protocol\n");
			return -FI_ENODATA;
		}

		if (ep_attr->protocol_version &&
		    (ep_attr->protocol_version !=
		     cxip_rdm_ep_attr.protocol_version)) {
			CXIP_LOG_DBG("Invalid protocol version\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_msg_size > cxip_rdm_ep_attr.max_msg_size) {
			CXIP_LOG_DBG("Message size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->msg_prefix_size >
		    cxip_rdm_ep_attr.msg_prefix_size) {
			CXIP_LOG_DBG("Msg prefix size not supported\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_raw_size >
		    cxip_rdm_ep_attr.max_order_raw_size) {
			CXIP_LOG_DBG("RAW order size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_war_size >
		    cxip_rdm_ep_attr.max_order_war_size) {
			CXIP_LOG_DBG("WAR order size too large\n");
			return -FI_ENODATA;
		}

		if (ep_attr->max_order_waw_size >
		    cxip_rdm_ep_attr.max_order_waw_size) {
			CXIP_LOG_DBG("WAW order size too large\n");
			return -FI_ENODATA;
		}

		if ((ep_attr->tx_ctx_cnt > CXIP_EP_MAX_TX_CNT) &&
		    ep_attr->tx_ctx_cnt != FI_SHARED_CONTEXT) {
			CXIP_LOG_DBG("TX CTX count too large\n");
			return -FI_ENODATA;
		}

		if ((ep_attr->rx_ctx_cnt > CXIP_EP_MAX_RX_CNT) &&
		    ep_attr->rx_ctx_cnt != FI_SHARED_CONTEXT) {
			CXIP_LOG_DBG("RX CTX count too large\n");
			return -FI_ENODATA;
		}
	}

	ret = cxip_rdm_verify_tx_attr(tx_attr);
	if (ret)
		return ret;

	ret = cxip_rdm_verify_rx_attr(rx_attr);
	if (ret)
		return ret;

	return 0;
}

int cxip_rdm_fi_info(uint32_t version, void *src_addr, void *dest_addr,
		     const struct fi_info *hints, struct fi_info **info)
{
	/* This creates info structure.
	 * - sets mode to CXIP_MODE
	 * - sets addr_format to FI_ADDR_CXI
	 * - sets handle according to hints
	 * - sets src_addr to argument, or default
	 * - sets dest_addr to argument, or NULL
	 * - sets ep_attr->type to FI_EP_RDM
	 * - initializes domain with hints
	 * - initializes fabric with hints
	 *
	 * Everything else gets overridden by the code that follows.
	 */
	*info = cxip_fi_info(version, FI_EP_RDM, hints, src_addr, dest_addr);
	if (!*info)
		return -FI_ENOMEM;

	/* overrides with fixed constants */
	*(*info)->tx_attr = cxip_rdm_tx_attr;
	(*info)->tx_attr->size = cxip_rdm_tx_attr.size;	// TODO: redundant
	*(*info)->rx_attr = cxip_rdm_rx_attr;
	(*info)->rx_attr->size = cxip_rdm_rx_attr.size;	// TODO: redundant
	*(*info)->ep_attr = cxip_rdm_ep_attr;

	/* only certain hints are used */
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
			(*info)->rx_attr->caps =
				CXIP_EP_RDM_SEC_CAP | hints->rx_attr->caps;
	}

	if (hints && hints->tx_attr) {
		(*info)->tx_attr->op_flags |= hints->tx_attr->op_flags;
		if (hints->tx_attr->caps)
			(*info)->tx_attr->caps =
				CXIP_EP_RDM_SEC_CAP | hints->tx_attr->caps;
	}

	/* apply TX/RX caps to all caps */
	(*info)->caps = CXIP_EP_RDM_CAP | (*info)->rx_attr->caps |
			(*info)->tx_attr->caps;

	/* apply caps hints to TX/RX and all caps */
	if (hints && hints->caps) {
		(*info)->caps = CXIP_EP_RDM_SEC_CAP | hints->caps;
		(*info)->rx_attr->caps =
			CXIP_EP_RDM_SEC_CAP |
			((*info)->rx_attr->caps & (*info)->caps);
		(*info)->tx_attr->caps =
			CXIP_EP_RDM_SEC_CAP |
			((*info)->tx_attr->caps & (*info)->caps);
	}
	return 0;
}

static int cxip_rdm_endpoint(struct fid_domain *domain, struct fi_info *info,
			     struct cxip_ep **ep, void *context, size_t fclass)
{
	int ret;

	if (info && info->ep_attr) {
		if (cxip_rdm_verify_ep_attr(info->ep_attr, info->tx_attr,
					    info->rx_attr))
			return -FI_EINVAL;
	}

	ret = cxip_alloc_endpoint(domain, info, ep, context, fclass);
	if (ret)
		return ret;

	// TODO: WTF?
	/* The info argument is required for EPs, but not SEPs. SEPs add TX/RX
	 * contexts after SEP creation, and they can have different attributes.
	 * So we specify the attributes for the EP to be the maximum attributes
	 * we support, so that TX/RX context attributes will always be a subset
	 * of the SEP attributes.
	 */
	if (!info || !info->ep_attr)
		(*ep)->ep_obj->ep_attr = cxip_rdm_ep_attr;

	if (!info || !info->tx_attr)
		(*ep)->tx_attr = cxip_rdm_tx_attr;

	if (!info || !info->rx_attr)
		(*ep)->rx_attr = cxip_rdm_rx_attr;

	return 0;
}

int cxip_rdm_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context)
{
	int ret;
	struct cxip_ep *endpoint;

	ret = cxip_rdm_endpoint(domain, info, &endpoint, context, FI_CLASS_EP);
	if (ret)
		return ret;

	*ep = &endpoint->ep;
	return 0;
}

int cxip_rdm_sep(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **sep, void *context)
{
	int ret;
	struct cxip_ep *endpoint;

	ret = cxip_rdm_endpoint(domain, info, &endpoint, context, FI_CLASS_SEP);
	if (ret)
		return ret;

	*sep = &endpoint->ep;

	return 0;
}
