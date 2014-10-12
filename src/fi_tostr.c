/*
 * Copyright (c) 2014 Intel Corp., Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *	copyright notice, this list of conditions and the following
 *	disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *	copyright notice, this list of conditions and the following
 *	disclaimer in the documentation and/or other materials
 *	provided with the distribution.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include "fi.h"

#define CASEENUMSTR(SYM) \
	case SYM: { strcat(buf, #SYM "\n"); break; }
#define IFFLAGSTR(flags, SYM) \
	do { if (flags & SYM) strcat(buf, #SYM " "); } while(0)


static void strcatf(char *dest, const char *fmt, ...)
{
	size_t len = strlen(dest);
	va_list arglist;

	va_start (arglist, fmt);
	vsprintf(&dest[len], fmt, arglist);
	va_end (arglist);
}

static void fi_tostr_flags(char *buf, uint64_t flags)
{
	IFFLAGSTR(flags, FI_INJECT);
	IFFLAGSTR(flags, FI_MULTI_RECV);
	IFFLAGSTR(flags, FI_SOURCE);
	IFFLAGSTR(flags, FI_SYMMETRIC);

	IFFLAGSTR(flags, FI_READ);
	IFFLAGSTR(flags, FI_WRITE);
	IFFLAGSTR(flags, FI_RECV);
	IFFLAGSTR(flags, FI_SEND);
	IFFLAGSTR(flags, FI_REMOTE_READ);
	IFFLAGSTR(flags, FI_REMOTE_WRITE);

	IFFLAGSTR(flags, FI_REMOTE_CQ_DATA);
	IFFLAGSTR(flags, FI_EVENT);
	IFFLAGSTR(flags, FI_REMOTE_SIGNAL);
	IFFLAGSTR(flags, FI_REMOTE_COMPLETE);
	IFFLAGSTR(flags, FI_CANCEL);
	IFFLAGSTR(flags, FI_MORE);
	IFFLAGSTR(flags, FI_PEEK);
	IFFLAGSTR(flags, FI_TRIGGER);
}

static void fi_tostr_addr_format(char *buf, uint32_t addr_format)
{
	switch (addr_format) {
	CASEENUMSTR(FI_ADDR_PROTO);
	CASEENUMSTR(FI_SOCKADDR);
	CASEENUMSTR(FI_SOCKADDR_IN);
	CASEENUMSTR(FI_SOCKADDR_IN6);
	CASEENUMSTR(FI_SOCKADDR_IB);
	default:
		if (addr_format & FI_PROV_SPECIFIC)
			strcat(buf, "Provider specific\n");
		else
			strcat(buf, "Unknown\n");
		break;
	}
}

static void fi_tostr_progress(char *buf, enum fi_progress progress)
{
	switch (progress) {
	CASEENUMSTR(FI_PROGRESS_UNSPEC);
	CASEENUMSTR(FI_PROGRESS_AUTO);
	CASEENUMSTR(FI_PROGRESS_MANUAL);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void fi_tostr_threading(char *buf, enum fi_threading threading)
{
	switch (threading) {
	CASEENUMSTR(FI_THREAD_UNSPEC);
	CASEENUMSTR(FI_THREAD_SAFE);
	CASEENUMSTR(FI_THREAD_PROGRESS);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}


static void fi_tostr_order(char *buf, uint64_t flags)
{
	IFFLAGSTR(flags, FI_ORDER_RAR);
	IFFLAGSTR(flags, FI_ORDER_RAW);
	IFFLAGSTR(flags, FI_ORDER_RAS);
	IFFLAGSTR(flags, FI_ORDER_WAR);
	IFFLAGSTR(flags, FI_ORDER_WAW);
	IFFLAGSTR(flags, FI_ORDER_WAS);
	IFFLAGSTR(flags, FI_ORDER_SAR);
	IFFLAGSTR(flags, FI_ORDER_SAW);
	IFFLAGSTR(flags, FI_ORDER_SAS);
}

static void fi_tostr_caps(char *buf, uint64_t caps)
{
	IFFLAGSTR(caps, FI_PASSIVE);
	IFFLAGSTR(caps, FI_MSG);
	IFFLAGSTR(caps, FI_RMA);
	IFFLAGSTR(caps, FI_TAGGED);
	IFFLAGSTR(caps, FI_ATOMICS);
	IFFLAGSTR(caps, FI_MULTICAST);
	IFFLAGSTR(caps, FI_USER_MR_KEY);
	IFFLAGSTR(caps, FI_DYNAMIC_MR);
	IFFLAGSTR(caps, FI_BUFFERED_RECV);
	fi_tostr_flags(buf, caps);
}

static void fi_tostr_ep_type(char *buf, enum fi_ep_type ep_type)
{
	switch (ep_type) {
	CASEENUMSTR(FI_EP_UNSPEC);
	CASEENUMSTR(FI_EP_MSG);
	CASEENUMSTR(FI_EP_DGRAM);
	CASEENUMSTR(FI_EP_RDM);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void fi_tostr_protocol(char *buf, uint32_t protocol)
{
	switch (protocol) {
	CASEENUMSTR(FI_PROTO_UNSPEC);
	CASEENUMSTR(FI_PROTO_RDMA_CM_IB_RC);
	CASEENUMSTR(FI_PROTO_IWARP);
	CASEENUMSTR(FI_PROTO_IB_UD);
	CASEENUMSTR(FI_PROTO_PSMX);
	default:
		if (protocol & FI_PROV_SPECIFIC)
			strcat(buf, "Provider specific\n");
		else
			strcat(buf, "Unknown\n");
		break;
	}
}

static void fi_tostr_mode(char *buf, uint64_t mode)
{
	IFFLAGSTR(mode, FI_WRITE_NONCOHERENT);
	IFFLAGSTR(mode, FI_CONTEXT);
	IFFLAGSTR(mode, FI_LOCAL_MR);
}

static void fi_tostr_tx_attr(char *buf, const struct fi_tx_ctx_attr *attr,
			     const char *prefix)
{
	strcatf(buf, "%sfi_tx_attr: [\n", prefix);
	strcatf(buf, "%s\tcaps: [ ", prefix);
	fi_tostr_caps(buf, attr->caps);
	strcat(buf, "]\n");

	strcatf(buf, "%s\top_flags: [ ", prefix);
	fi_tostr_flags(buf, attr->op_flags);
	strcat(buf, "]\n");

	strcatf(buf, "%s\tmsg_order: [ ", prefix);
	fi_tostr_order(buf, attr->msg_order);
	strcat(buf, "]\n");

	strcatf(buf, "%s\tinject_size:\t%zd\n", prefix, attr->inject_size);
	strcatf(buf, "%s\tsize:\t%zd\n", prefix, attr->size);
	strcatf(buf, "%s\tiov_limit:\t%zd\n", prefix, attr->iov_limit);
	strcatf(buf, "%s\top_alignment:\t%zd\n", prefix, attr->op_alignment);
	strcatf(buf, "%s]\n", prefix);
}

static void fi_tostr_rx_attr(char *buf, const struct fi_rx_ctx_attr *attr,
			     const char *prefix)
{
	strcatf(buf, "%sfi_rx_attr: [\n", prefix);
	strcatf(buf, "%s\tcaps: [ ", prefix);
	fi_tostr_caps(buf, attr->caps);
	strcat(buf, "]\n");

	strcatf(buf, "%s\top_flags: [ ", prefix);
	fi_tostr_flags(buf, attr->op_flags);
	strcat(buf, "]\n");

	strcatf(buf, "%s\tmsg_order: [ ", prefix);
	fi_tostr_order(buf, attr->msg_order);
	strcat(buf, "]\n");

	strcatf(buf, "%s\ttotal_buffered_recv:\t%zd\n", prefix, attr->total_buffered_recv);
	strcatf(buf, "%s\tsize:\t%zd\n", prefix, attr->size);
	strcatf(buf, "%s\tiov_limit:\t%zd\n", prefix, attr->iov_limit);
	strcatf(buf, "%s\top_alignment:\t%zd\n", prefix, attr->op_alignment);
	strcatf(buf, "%s]\n", prefix);
}

static void fi_tostr_ep_attr(char *buf, const struct fi_ep_attr *attr, const char *prefix)
{
	strcatf(buf, "%sfi_ep_attr: [\n", prefix);
	strcatf(buf, "%s\tprotocol:\t", prefix);
	fi_tostr_protocol(buf, attr->protocol);
	strcatf(buf, "%s\tmax_msg_size:\t%zd\n", prefix, attr->max_msg_size);
	strcatf(buf, "%s\tinject_size:\t%zd\n", prefix, attr->inject_size);
	strcatf(buf, "%s\ttotal_buffered_recv:\t%zd\n", prefix, attr->total_buffered_recv);
	strcatf(buf, "%s\tmax_order_raw_size:\t%zd\n", prefix, attr->max_order_raw_size);
	strcatf(buf, "%s\tmax_order_war_size:\t%zd\n", prefix, attr->max_order_war_size);
	strcatf(buf, "%s\tmax_order_waw_size:\t%zd\n", prefix, attr->max_order_waw_size);
	strcatf(buf, "%s\tmem_tag_format:\t0x%016llx\n", prefix, attr->mem_tag_format);

	strcatf(buf, "%s\tmsg_order: [ ", prefix);
	fi_tostr_order(buf, attr->msg_order);
	strcat(buf, "]\n");

	strcatf(buf, "%s\ttx_ctx_cnt:\t%zd\n", prefix, attr->tx_ctx_cnt);
	strcatf(buf, "%s\trx_ctx_cnt:\t%zd\n", prefix, attr->rx_ctx_cnt);
	strcatf(buf, "%s]\n", prefix);
}

static void fi_tostr_domain_attr(char *buf, const struct fi_domain_attr *attr,
				 const char *prefix)
{
	strcatf(buf, "%sfi_domain_attr: [\n", prefix);
	strcatf(buf, "%s\tname:\t%s\n", prefix, attr->name);
	strcatf(buf, "%s\tthreading:\t", prefix);
	fi_tostr_threading(buf, attr->threading);

	strcatf(buf, "%s\tcontrol_progress:\t", prefix);
	fi_tostr_progress(buf, attr->control_progress);
	strcatf(buf, "%s\tdata_progress:\t", prefix);
	fi_tostr_progress(buf, attr->data_progress);

	strcatf(buf, "%s\tmr_key_size:\t%zd\n", prefix, attr->mr_key_size);
	strcatf(buf, "%s\tcq_data_size:\t%zd\n", prefix, attr->cq_data_size);
	strcatf(buf, "%s\tep_cnt:\t%zd\n", prefix, attr->ep_cnt);
	strcatf(buf, "%s\ttx_ctx_cnt:\t%zd\n", prefix, attr->tx_ctx_cnt);
	strcatf(buf, "%s\trx_ctx_cnt:\t%zd\n", prefix, attr->rx_ctx_cnt);
	strcatf(buf, "%s\tmax_ep_tx_ctx:\t%zd\n", prefix, attr->max_ep_tx_ctx);
	strcatf(buf, "%s\tmax_ep_rx_ctx:\t%zd\n", prefix, attr->max_ep_rx_ctx);
	strcatf(buf, "%s\top_size:\t%zd\n", prefix, attr->op_size);
	strcatf(buf, "%s\tiov_size:\t%zd\n", prefix, attr->iov_size);
	strcatf(buf, "%s]\n", prefix);
}

static void fi_tostr_fabric_attr(char *buf, const struct fi_fabric_attr *attr,
				 const char *prefix)
{
	strcatf(buf, "%sfi_fabric: [\n", prefix);
	strcatf(buf, "%s\tname:\t%s\n", prefix, attr->name);
	strcatf(buf, "%s\tprov_name:\t%s\n", prefix, attr->prov_name);
	strcatf(buf, "%s\tprov_version:\t%d.%d\n", prefix,
		FI_MAJOR(attr->prov_version), FI_MINOR(attr->prov_version));
	strcatf(buf, "%s]\n", prefix);
}

static void fi_tostr_info(char *buf, const struct fi_info *info)
{
	strcat(buf, "fi_info: [\n");
	strcat(buf, "\tcaps: [ ");
	fi_tostr_caps(buf, info->caps);
	strcat(buf, "]\n");

	strcat(buf, "\tmode: [ ");
	fi_tostr_mode(buf, info->mode);
	strcat(buf, "]\n");

	strcat(buf, "\tep_type:\t");
	fi_tostr_ep_type(buf, info->ep_type);
	strcat(buf, "\tfi_addr_format:\t");
	fi_tostr_addr_format(buf, info->addr_format);

	strcatf(buf, "\tsrc_addrlen:\t%zd\n", info->src_addrlen);
	strcatf(buf, "\tdest_addrlen:\t%zd\n", info->dest_addrlen);
	strcatf(buf, "\tsrc_addr:\t%p\n", info->src_addr);
	strcatf(buf, "\tdest_addr:\t%p\n", info->dest_addr);
	strcatf(buf, "\tconnreq:\t%p\n", info->connreq);

	fi_tostr_tx_attr(buf, info->tx_attr, "\t");
	fi_tostr_rx_attr(buf, info->rx_attr, "\t");
	fi_tostr_ep_attr(buf, info->ep_attr, "\t");
	fi_tostr_domain_attr(buf, info->domain_attr, "\t");
	fi_tostr_fabric_attr(buf, info->fabric_attr, "\t");
	strcat(buf, "]\n");
}

char *fi_tostr(const void *data, enum fi_type datatype)
{
	static __thread char *buf;
	uint64_t val64 = *(const uint64_t *) data;
	uint32_t val32 = *(const uint32_t *) data;
	int enumval = *(const int *) data;

	if (!data)
		return NULL;

	if (!buf) {
		buf = calloc(4096, sizeof (*buf));
		if (!buf)
			return NULL;
	} else {
		buf[0] = 0;
	}

	switch (datatype) {
	case FI_TYPE_INFO:
		fi_tostr_info(buf, data);
		break;
	case FI_TYPE_EP_TYPE:
		fi_tostr_ep_type(buf, enumval);
		break;
	case FI_TYPE_CAPS:
		fi_tostr_caps(buf, val64);
		break;
	case FI_TYPE_OP_FLAGS:
		fi_tostr_flags(buf, val64);
		break;
	case FI_TYPE_ADDR_FORMAT:
		fi_tostr_addr_format(buf, val32);
		break;
	case FI_TYPE_TX_ATTR:
		fi_tostr_tx_attr(buf, data, "");
		break;
	case FI_TYPE_RX_ATTR:
		fi_tostr_rx_attr(buf, data, "");
		break;
	case FI_TYPE_EP_ATTR:
		fi_tostr_ep_attr(buf, data, "");
		break;
	case FI_TYPE_DOMAIN_ATTR:
		fi_tostr_domain_attr(buf, data, "");
		break;
	case FI_TYPE_FABRIC_ATTR:
		fi_tostr_fabric_attr(buf, data, "");
		break;
	case FI_TYPE_THREADING:
		fi_tostr_threading(buf, enumval);
		break;
	case FI_TYPE_PROGRESS:
		fi_tostr_progress(buf, enumval);
		break;
	case FI_TYPE_PROTOCOL:
		fi_tostr_protocol(buf, val64);
		break;
	case FI_TYPE_MSG_ORDER:
		fi_tostr_order(buf, val64);
		break;
	case FI_TYPE_MODE:
		fi_tostr_mode(buf, val64);
		break;
	default:
		strcat(buf, "Unknown type");
		break;
	}
	return buf;
}

#undef CASEENUMSTR
#undef IFFLAGSTR
