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
	do { if (flags & SYM) strcat(buf, #SYM ", "); } while(0)


static void strcatf(char *dest, const char *fmt, ...)
{
	size_t len = strlen(dest);
	va_list arglist;

	va_start (arglist, fmt);
	vsprintf(&dest[len], fmt, arglist);
	va_end (arglist);
}

static void _pp_ep_type(char *buf, uint64_t ep_type)
{
	strcat(buf, "type:\t");
	switch (ep_type) {
	CASEENUMSTR(FI_EP_UNSPEC);
	CASEENUMSTR(FI_EP_MSG);
	CASEENUMSTR(FI_EP_DGRAM);
	CASEENUMSTR(FI_EP_RDM);
	CASEENUMSTR(FI_EP_MAX);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void _pp_ep_cap(char *buf, uint64_t ep_cap)
{
	strcat(buf, "ep_cap: [");
	IFFLAGSTR(ep_cap, FI_PASSIVE);
	IFFLAGSTR(ep_cap, FI_MSG);
	IFFLAGSTR(ep_cap, FI_RMA);
	IFFLAGSTR(ep_cap, FI_TAGGED);
	IFFLAGSTR(ep_cap, FI_ATOMICS);
	IFFLAGSTR(ep_cap, FI_MULTICAST);
	IFFLAGSTR(ep_cap, FI_BUFFERED_RECV);
	strcat(buf, "]\n");
}

static void _pp_op_flags(char *buf, uint64_t flags)
{
	strcat(buf, "op_flags: [");
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
	strcat(buf, "]\n");
}

static void _pp_addr_format(char *buf, enum fi_addr_format addr_format)
{
	strcat(buf, "fi_addr_format:\t");
	switch(addr_format) {
	CASEENUMSTR(FI_ADDR_PROTO);
	CASEENUMSTR(FI_SOCKADDR);
	CASEENUMSTR(FI_SOCKADDR_IN);
	CASEENUMSTR(FI_SOCKADDR_IN6);
	CASEENUMSTR(FI_SOCKADDR_IB);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void _pp_protocol(char *buf, enum fi_proto protocol)
{
	strcat(buf, "protocol:\t");
	switch (protocol) {
	CASEENUMSTR(FI_PROTO_UNSPEC);
	CASEENUMSTR(FI_PROTO_RDMA_CM_IB_RC);
	CASEENUMSTR(FI_PROTO_IWARP);
	CASEENUMSTR(FI_PROTO_IB_UD);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void _pp_msg_order(char *buf, uint64_t flags)
{
	strcat(buf, "msg_order: [");
	IFFLAGSTR(flags, FI_ORDER_RAR);
	IFFLAGSTR(flags, FI_ORDER_RAW);
	IFFLAGSTR(flags, FI_ORDER_RAS);
	IFFLAGSTR(flags, FI_ORDER_WAR);
	IFFLAGSTR(flags, FI_ORDER_WAW);
	IFFLAGSTR(flags, FI_ORDER_WAS);
	IFFLAGSTR(flags, FI_ORDER_SAR);
	IFFLAGSTR(flags, FI_ORDER_SAW);
	IFFLAGSTR(flags, FI_ORDER_SAS);
	strcat(buf, "]\n");
}

static void _pp_domain_cap(char *buf, uint64_t domain_cap)
{
	strcat(buf, "domain_cap: [");
	IFFLAGSTR(domain_cap, FI_WRITE_COHERENT);
	IFFLAGSTR(domain_cap, FI_CONTEXT);
	IFFLAGSTR(domain_cap, FI_LOCAL_MR);
	strcat(buf, "]\n");
}

static void _pp_threading(char *buf, enum fi_threading threading)
{
	strcat(buf, "threading:\t");
	switch (threading) {
	CASEENUMSTR(FI_THREAD_UNSPEC);
	CASEENUMSTR(FI_THREAD_SAFE);
	CASEENUMSTR(FI_THREAD_PROGRESS);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void _pp_progress(char *buf, enum fi_progress progress, const char* label)
{
	strcatf(buf, "%s:\t", label);
	switch (progress) {
	CASEENUMSTR(FI_PROGRESS_UNSPEC);
	CASEENUMSTR(FI_PROGRESS_AUTO);
	CASEENUMSTR(FI_PROGRESS_MANUAL);
	default:
		strcat(buf, "Unknown\n");
		break;
	}
}

static void _pp_tx_attr(char *buf, const struct fi_tx_ctx_attr *attr,
			const char *indent)
{
	strcatf(buf, "fi_tx_ctx_attr:\n%s", indent);
	strcat(buf, indent);
	_pp_ep_cap(buf, attr->ep_cap);
	strcat(buf, indent);
	_pp_op_flags(buf, attr->op_flags);
	strcat(buf, indent);
	_pp_msg_order(buf, attr->msg_order);
	strcatf(buf, "%sinject_size:\t%d\n", indent, attr->inject_size);
	strcatf(buf, "%ssize:\t%d\n", indent, attr->size);
	strcatf(buf, "%siov_limit:\t%d\n", indent, attr->iov_limit);
	strcatf(buf, "%sop_alignment:\t%d\n", indent, attr->op_alignment);
}

static void _pp_rx_attr(char *buf, const struct fi_rx_ctx_attr *attr,
			const char *indent)
{
	strcatf(buf, "fi_rx_ctx_attr:\n%s", indent);
	strcat(buf, indent);
	_pp_ep_cap(buf, attr->ep_cap);
	strcat(buf, indent);
	_pp_op_flags(buf, attr->op_flags);
	strcat(buf, indent);
	_pp_msg_order(buf, attr->msg_order);
	strcatf(buf, "%stotal_buffered_recv:\t%d\n", indent, attr->total_buffered_recv);
	strcatf(buf, "%ssize:\t%d\n", indent, attr->size);
	strcatf(buf, "%siov_limit:\t%d\n", indent, attr->iov_limit);
	strcatf(buf, "%sop_alignment:\t%d\n", indent, attr->op_alignment);
}

static void _pp_ep_attr(char *buf, const struct fi_ep_attr *attr, const char *indent)
{
	strcatf(buf, "fi_ep_attr:\n%s", indent);
	_pp_protocol(buf, attr->protocol);
	strcatf(buf, "%smax_msg_size:\t%d\n", indent, attr->max_msg_size);
	strcatf(buf, "%sinject_size:\t%d\n", indent, attr->inject_size);
	strcatf(buf, "%stotal_buffered_recv:\t%d\n", indent, attr->total_buffered_recv);
	strcatf(buf, "%smax_order_raw_size:\t%d\n", indent, attr->max_order_raw_size);
	strcatf(buf, "%smax_order_war_size:\t%d\n", indent, attr->max_order_war_size);
	strcatf(buf, "%smax_order_waw_size:\t%d\n", indent, attr->max_order_waw_size);
	strcatf(buf, "%smem_tag_format:\t%016llx\n", indent, attr->mem_tag_format);
	strcat(buf, indent);
	_pp_msg_order(buf, attr->msg_order);
	strcatf(buf, "%stx_ctx_cnt:\t%d\n", indent, attr->tx_ctx_cnt);
	strcatf(buf, "%srx_ctx_cnt:\t%d\n", indent, attr->rx_ctx_cnt);
}

static void _pp_domain_attr(char *buf, const struct fi_domain_attr *attr,
			    const char *indent)
{
	strcat(buf, "fi_domain_attr:\n");
	strcatf(buf, "%sname:\t%s\n", indent, attr->name);
	strcat(buf, indent);
	_pp_threading(buf, attr->threading);

	strcat(buf, indent);
	_pp_progress(buf, attr->control_progress, "control_progress");
	strcat(buf, indent);
	_pp_progress(buf, attr->data_progress, "data_progress");

	strcatf(buf, "%smr_key_size:\t%d\n", indent, attr->mr_key_size);
	strcatf(buf, "%scq_data_size:\t%d\n", indent, attr->cq_data_size);
	strcatf(buf, "%sep_cnt:\t%d\n", indent, attr->ep_cnt);
	strcatf(buf, "%stx_ctx_cnt:\t%d\n", indent, attr->tx_ctx_cnt);
	strcatf(buf, "%srx_ctx_cnt:\t%d\n", indent, attr->rx_ctx_cnt);
	strcatf(buf, "%smax_ep_tx_ctx:\t%d\n", indent, attr->max_ep_tx_ctx);
	strcatf(buf, "%smax_ep_rx_ctx:\t%d\n", indent, attr->max_ep_rx_ctx);
	strcatf(buf, "%sop_size:\t%d\n", indent, attr->op_size);
	strcatf(buf, "%siov_size:\t%d\n", indent, attr->iov_size);
}

static void _pp_fabric_attr(char *buf, const struct fi_fabric_attr *attr,
			    const char *indent)
{
	strcat(buf, "fi_fabric_attr:\n");
	strcatf(buf, "%sname:\t\t%s\n", indent, attr->name);
	strcatf(buf, "%sprov_name:\t%s\n", indent, attr->prov_name);
	strcatf(buf, "%sprov_version:\t%d.%d\n", indent,
		FI_MAJOR(attr->prov_version), FI_MINOR(attr->prov_version));
}

static void _pp_info(char *buf, const struct fi_info *info, const char *indent)
{
	char *rindent = "    ";

	strcat(buf, "fi_info:\n");
	strcat(buf, indent);
	_pp_ep_type(buf, info->type);
	strcat(buf, indent);
	_pp_ep_cap(buf, info->ep_cap);
	strcat(buf, indent);
	_pp_domain_cap(buf, info->domain_cap);
	strcat(buf, indent);
	_pp_addr_format(buf, info->addr_format);

	strcatf(buf, "%ssource_addr:\t%p\n", indent, info->src_addr);
	strcatf(buf, "%sdest_addr:\t%p\n", indent, info->dest_addr);
	strcatf(buf, "%sconnreq:\t%p\n", indent, info->connreq);

	strcat(buf, indent);
	_pp_tx_attr(buf, info->tx_attr, rindent);
	strcat(buf, indent);
	_pp_rx_attr(buf, info->rx_attr, rindent);
	strcat(buf, indent);
	_pp_ep_attr(buf, info->ep_attr, rindent);
	strcat(buf, indent);
	_pp_domain_attr(buf, info->domain_attr, rindent);
	strcat(buf, indent);
	_pp_fabric_attr(buf, info->fabric_attr, rindent);
}

char *fi_tostr(const void *data, enum fi_pp_type datatype)
{
	char *indent = "  ";
	static __thread char *buf;
	uint64_t val64 = *(const uint64_t *) data;
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
	case FI_PP_INFO:
		_pp_info(buf, data, indent);
		break;
	case FI_PP_EP_TYPE:
		_pp_ep_type(buf, val64);
		break;
	case FI_PP_EP_CAP:
		_pp_ep_cap(buf, val64);
		break;
	case FI_PP_OP_FLAGS:
		_pp_op_flags(buf, val64);
		break;
	case FI_PP_ADDR_FORMAT:
		_pp_addr_format(buf, enumval);
		break;
	case FI_PP_TX_ATTR:
		_pp_tx_attr(buf, data, indent);
		break;
	case FI_PP_RX_ATTR:
		_pp_rx_attr(buf, data, indent);
		break;
	case FI_PP_EP_ATTR:
		_pp_ep_attr(buf, data, indent);
		break;
	case FI_PP_DOMAIN_ATTR:
		_pp_domain_attr(buf, data, indent);
		break;
	case FI_PP_FABRIC_ATTR:
		_pp_fabric_attr(buf, data, indent);
		break;
	case FI_PP_DOMAIN_CAP:
		_pp_domain_cap(buf, val64);
		break;
	case FI_PP_THREADING:
		_pp_threading(buf, enumval);
		break;
	case FI_PP_PROGRESS:
		_pp_progress(buf, enumval, "progress");
		break;
	case FI_PP_PROTOCOL:
		_pp_protocol(buf, val64);
		break;
	case FI_PP_MSG_ORDER:
		_pp_msg_order(buf, val64);
		break;
	default:
		strcat(buf, "Unknown type");
		break;
	}
	return buf;
}

#undef CASEENUMSTR
#undef IFFLAGSTR
