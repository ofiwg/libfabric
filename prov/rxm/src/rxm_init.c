/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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

#include <netdb.h>

#include <rdma/fi_errno.h>
#include <ofi_net.h>

#include <ofi_prov.h>
#include "rxm.h"

int rxm_defer_requests = 0;
size_t rxm_msg_tx_size		= 128;
size_t rxm_msg_rx_size		= 128;
size_t rxm_def_univ_size	= 256;

char *rxm_proto_state_str[] = {
	RXM_PROTO_STATES(OFI_STR)
};

/*
 * - Support FI_MR_LOCAL/FI_LOCAL_MR as ofi_rxm can handle it.
 * - The RxM FI_RMA implementation is pass-through but the provider can handle
 *   FI_MR_PROV_KEY and FI_MR_VIRT_ADDR in its large message transfer rendezvous
 *   protocol.
 * - fi_alter_domain_attr should correctly set the mr_mode in return fi_info
 *   based on hints.
 */
void rxm_info_to_core_mr_modes(uint32_t version, const struct fi_info *hints,
			       struct fi_info *core_info)
{
	/* We handle FI_MR_BASIC and FI_MR_SCALABLE irrespective of version */
	if (hints && hints->domain_attr &&
	    (hints->domain_attr->mr_mode & (FI_MR_SCALABLE | FI_MR_BASIC))) {
		core_info->mode = FI_LOCAL_MR;
		core_info->domain_attr->mr_mode = hints->domain_attr->mr_mode;
	} else if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		core_info->mode |= FI_LOCAL_MR;
		/* Specify FI_MR_UNSPEC (instead of FI_MR_BASIC) so that
		 * providers that support only FI_MR_SCALABLE aren't dropped */
		core_info->domain_attr->mr_mode = FI_MR_UNSPEC;
	} else {
		core_info->domain_attr->mr_mode |= FI_MR_LOCAL;
		if (!hints || !ofi_rma_target_allowed(hints->caps))
			core_info->domain_attr->mr_mode |= OFI_MR_BASIC_MAP;
		else if (hints->domain_attr)
			core_info->domain_attr->mr_mode |=
				hints->domain_attr->mr_mode & OFI_MR_BASIC_MAP;
	}
}

int rxm_info_to_core(uint32_t version, const struct fi_info *hints,
		     struct fi_info *core_info)
{
	rxm_info_to_core_mr_modes(version, hints, core_info);

	core_info->mode |= FI_RX_CQ_DATA | FI_CONTEXT;

	if (hints) {
		if (hints->caps & FI_TAGGED)
			core_info->caps |= FI_MSG;

		/* FI_RMA cap is needed for large message transfer protocol */
		if (hints->caps & (FI_MSG | FI_TAGGED))
			core_info->caps |= FI_RMA;

		if (hints->domain_attr) {
			core_info->domain_attr->caps |= hints->domain_attr->caps;
		}
		if (hints->tx_attr) {
			core_info->tx_attr->msg_order = hints->tx_attr->msg_order;
			core_info->tx_attr->comp_order = hints->tx_attr->comp_order;
		}
		if (hints->rx_attr) {
			core_info->rx_attr->msg_order = hints->rx_attr->msg_order;
			core_info->rx_attr->comp_order = hints->rx_attr->comp_order;
		}
	}
	core_info->ep_attr->type = FI_EP_MSG;

	core_info->tx_attr->size = rxm_msg_tx_size;
	core_info->rx_attr->size = rxm_msg_rx_size;

	return 0;
}

int rxm_info_to_rxm(uint32_t version, const struct fi_info *core_info,
		    struct fi_info *info)
{
	info->caps = rxm_info.caps;
	info->mode = core_info->mode | rxm_info.mode;

	info->tx_attr->caps		= rxm_info.tx_attr->caps;
	info->tx_attr->mode		= info->mode;
	info->tx_attr->msg_order 	= core_info->tx_attr->msg_order;
	info->tx_attr->comp_order 	= rxm_info.tx_attr->comp_order;
	info->tx_attr->inject_size	= rxm_info.tx_attr->inject_size;
	info->tx_attr->size 		= rxm_info.tx_attr->size;
	info->tx_attr->iov_limit 	= MIN(rxm_info.tx_attr->iov_limit,
					      core_info->tx_attr->iov_limit);
	info->tx_attr->rma_iov_limit	= MIN(rxm_info.tx_attr->rma_iov_limit,
					      core_info->tx_attr->rma_iov_limit);

	info->rx_attr->caps		= rxm_info.rx_attr->caps;
	info->rx_attr->mode		= info->mode;
	info->rx_attr->msg_order 	= core_info->rx_attr->msg_order;
	info->rx_attr->comp_order 	= rxm_info.rx_attr->comp_order;
	info->rx_attr->size 		= rxm_info.rx_attr->size;
	info->rx_attr->iov_limit 	= MIN(rxm_info.rx_attr->iov_limit,
					      core_info->rx_attr->iov_limit);

	*info->ep_attr = *rxm_info.ep_attr;
	info->ep_attr->max_msg_size = core_info->ep_attr->max_msg_size;
	info->ep_attr->max_order_raw_size = core_info->ep_attr->max_order_raw_size;
	info->ep_attr->max_order_war_size = core_info->ep_attr->max_order_war_size;
	info->ep_attr->max_order_waw_size = core_info->ep_attr->max_order_waw_size;

	*info->domain_attr = *rxm_info.domain_attr;
	info->domain_attr->mr_mode |= core_info->domain_attr->mr_mode;
	info->domain_attr->cq_data_size = MIN(core_info->domain_attr->cq_data_size,
					      rxm_info.domain_attr->cq_data_size);
	info->domain_attr->mr_key_size = core_info->domain_attr->mr_key_size;

	return 0;
}

static int rxm_init_info(void)
{
	int param;

	if (!fi_param_get_int(&rxm_prov, "buffer_size", &param)) {
		if (param > sizeof(struct rxm_pkt)) {
			rxm_info.tx_attr->inject_size = param;
		} else {
			FI_WARN(&rxm_prov, FI_LOG_CORE,
				"Requested buffer size too small\n");
			return -FI_EINVAL;
		}
	} else {
		rxm_info.tx_attr->inject_size = RXM_BUF_SIZE;
	}
	rxm_info.tx_attr->inject_size -= sizeof(struct rxm_pkt);
	rxm_util_prov.info = &rxm_info;
	return 0;
}

static void rxm_alter_info(const struct fi_info *hints, struct fi_info *info)
{
	struct fi_info *cur;

	for (cur = info; cur; cur = cur->next) {
		/* Remove the following caps if they are not requested as they
		 * may affect performance in fast-path */
		if (!hints) {
			cur->caps &= ~(FI_DIRECTED_RECV | FI_SOURCE);
		} else {
			if (!(hints->caps & FI_DIRECTED_RECV))
				cur->caps &= ~FI_DIRECTED_RECV;
			if (!(hints->caps & FI_SOURCE))
				cur->caps &= ~FI_SOURCE;
			if (hints->ep_attr && hints->ep_attr->mem_tag_format &&
			    (info->caps & FI_TAGGED)) {
				FI_INFO(&rxm_prov, FI_LOG_CORE,
					"mem_tag_format requested: 0x%" PRIx64
					" (note: provider doesn't optimize "
					"based on mem_tag_format)\n",
					hints->ep_attr->mem_tag_format);
				info->ep_attr->mem_tag_format =
					hints->ep_attr->mem_tag_format;
			}
		}
	}
}

static int rxm_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	struct fi_info *cur, *dup;
	struct addrinfo *ai;
	uint16_t port_save = 0;
	int ret;

	/* Avoid getting wild card address from MSG provider */
	if (ofi_is_wildcard_listen_addr(node, service, flags, hints)) {
		if (service) {
			ret = getaddrinfo(NULL, service, NULL, &ai);
			if (ret) {
				FI_WARN(&rxm_prov, FI_LOG_CORE,
					"Unable to getaddrinfo\n");
				return ret;
			}
			port_save = ofi_addr_get_port(ai->ai_addr);
			freeaddrinfo(ai);
			service = NULL;
		} else {
			port_save = ofi_addr_get_port(hints->src_addr);
			ofi_addr_set_port(hints->src_addr, 0);
		}
	}

	ret = ofix_getinfo(version, node, service, flags, &rxm_util_prov, hints,
			   rxm_info_to_core, rxm_info_to_rxm, info);
	if (ret)
		return ret;

	if (port_save) {
		for (cur = *info; cur; cur = cur->next) {
			assert(cur->src_addr);
			ofi_addr_set_port(cur->src_addr, port_save);
		}
	}

	rxm_alter_info(hints, *info);

	/* If app supports FI_MR_LOCAL, prioritize requiring it for
	 * better performance. */
	if (hints && hints->domain_attr &&
	    (OFI_CHECK_MR_LOCAL(hints))) {
		for (cur = *info; cur; cur = cur->next) {
			if (!OFI_CHECK_MR_LOCAL(cur))
				continue;
			dup = fi_dupinfo(cur);
			if (!dup) {
				fi_freeinfo(*info);
				return -FI_ENOMEM;
			} 

			dup->mode &= ~FI_LOCAL_MR;
			dup->domain_attr->mr_mode &= ~FI_MR_LOCAL;

			dup->next = cur->next;
			cur->next = dup;
			cur = dup;
		}
	} else {
		for (cur = *info; cur; cur = cur->next) {
			cur->mode &= ~FI_LOCAL_MR;
			cur->domain_attr->mr_mode &= ~FI_MR_LOCAL;
		}
	}
	return 0;
}


static void rxm_fini(void)
{
	/* yawn */
}

struct fi_provider rxm_prov = {
	.name = OFI_UTIL_PREFIX "rxm",
	.version = FI_VERSION(RXM_MAJOR_VERSION, RXM_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = rxm_getinfo,
	.fabric = rxm_fabric,
	.cleanup = rxm_fini
};

RXM_INI
{
	fi_param_define(&rxm_prov, "buffer_size", FI_PARAM_INT,
			"Defines the transmit buffer size / inject size. Messages"
			" of size less than this would be transmitted via an "
			"eager protocol and those above would be transmitted "
			"via a rendezvous protocol. Transmit data would be copied"
			" up to this size (default: ~16k).");

	fi_param_define(&rxm_prov, "comp_per_progress", FI_PARAM_INT,
			"Defines the maximum number of MSG provider CQ entries "
			"(default: 1) that would be read per progress "
			"(RxM CQ read).");

	fi_param_define(&rxm_prov, "defer_requests", FI_PARAM_BOOL,
			"Defer requests when connection is not established "
			"(default: false)\n");

	fi_param_get_bool(&rxm_prov, "defer_requests", &rxm_defer_requests);

	fi_param_define(&rxm_prov, "tx_size", FI_PARAM_SIZE_T,
			"Defines default tx context size (default: 1024).");

	fi_param_define(&rxm_prov, "rx_size", FI_PARAM_SIZE_T,
			"Defines default rx context size (default: 1024).");

	fi_param_define(&rxm_prov, "msg_tx_size", FI_PARAM_SIZE_T,
			"Defines FI_EP_MSG tx size that would be requested "
			"(default: 128). Setting this to 0 would get default "
			"value defined by the MSG provider.");

	fi_param_define(&rxm_prov, "msg_rx_size", FI_PARAM_SIZE_T,
			"Defines FI_EP_MSG rx size that would be requested "
			"(default: 128). Setting this to 0 would get default "
			"value defined by the MSG provider.");

	fi_param_get_size_t(&rxm_prov, "tx_size", &rxm_info.tx_attr->size);
	fi_param_get_size_t(&rxm_prov, "rx_size", &rxm_info.rx_attr->size);
	fi_param_get_size_t(&rxm_prov, "msg_tx_size", &rxm_msg_tx_size);
	fi_param_get_size_t(&rxm_prov, "msg_rx_size", &rxm_msg_rx_size);
	fi_param_get_size_t(NULL, "universe_size", &rxm_def_univ_size);

	if (rxm_init_info()) {
		FI_WARN(&rxm_prov, FI_LOG_CORE, "Unable to initialize rxm_info\n");
		return NULL;
	}

	return &rxm_prov;
}
