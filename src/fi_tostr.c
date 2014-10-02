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

/* generate enum cases inside switch */
#define EN(SYM) \
	case SYM: \
		return strcat(buf, #SYM "\n");

static char *strcatf(char *dest, const char *fmt, ...)
{
	size_t len = strlen(dest);
	va_list arglist;

	va_start (arglist, fmt);
	vsprintf(&dest[len], fmt, arglist);
	va_end (arglist);

	return dest;
}

static char *_pp_fi_ep_type(char *buf, const enum fi_ep_type *t)
{
	strcat(buf, "type:\t");

	switch (*t) {
	EN(FI_EP_UNSPEC)
	EN(FI_EP_MSG)
	EN(FI_EP_DGRAM)
	EN(FI_EP_RDM)
	EN(FI_EP_MAX)
	default:
		return strcat(buf, "Unknown\n");
	}
}

static char *_pp_ep_cap(char *buf, const uint64_t *flags)
{
	strcat(buf, "ep_cap: [");

#define EP_CAPS \
	EC(FI_PASSIVE) EC(FI_MSG) EC(FI_RMA) \
	EC(FI_TAGGED) EC(FI_ATOMICS) \
	EC(FI_MULTICAST) EC(FI_BUFFERED_RECV)

#define EC(SYM) \
	if (*flags & SYM) \
		strcat(buf, #SYM ", ");

	EP_CAPS

#undef EC

	return strcat(buf, "]\n");
}

static char *_pp_op_flags(char *buf, const uint64_t *flags)
{
	strcat(buf, "op_flags: [");

#define OP_FLAGS \
	OF(FI_INJECT) OF(FI_MULTI_RECV) OF(FI_SOURCE) OF(FI_SYMMETRIC) \
	OF(FI_READ) OF(FI_WRITE) OF(FI_RECV) OF(FI_SEND) \
	OF(FI_REMOTE_READ) OF(FI_REMOTE_WRITE) \
	OF(FI_REMOTE_READ) OF(FI_REMOTE_WRITE) \
	OF(FI_REMOTE_EQ_DATA) OF(FI_EVENT) OF(FI_REMOTE_SIGNAL) \
	OF(FI_REMOTE_COMPLETE) OF(FI_CANCEL) OF(FI_MORE) OF(FI_PEEK) OF(FI_TRIGGER)

#define OF(SYM) \
	if (*flags & SYM) \
		strcat(buf, #SYM ", ");

	OP_FLAGS

#undef OF

	return strcat(buf, "]\n");
}

static char *_pp_fi_addr_format(char *buf, const enum fi_addr_format *addrf)
{
	strcat(buf, "fi_addr_format:\t");

	switch(*addrf) {
	EN(FI_ADDR_PROTO)
	EN(FI_SOCKADDR)
	EN(FI_SOCKADDR_IN)
	EN(FI_SOCKADDR_IN6)
	EN(FI_SOCKADDR_IB)
	default:
		return strcat(buf, "Unknown\n");
	}
}

static char *_pp_fi_proto(char *buf, const enum fi_proto *proto)
{
	strcat(buf, "protocol:\t");

	switch (*proto) {
	EN(FI_PROTO_UNSPEC)
	EN(FI_PROTO_RDMA_CM_IB_RC)
	EN(FI_PROTO_IWARP)
	EN(FI_PROTO_IB_UD)
	default:
		return strcat(buf, "Unknown\n");
	}
}

static char *_pp_msg_order(char *buf, const uint64_t *flags)
{
	strcat(buf, "msg_order: [");

#define MSG_ORDER \
	MO(FI_ORDER_RAR) MO(FI_ORDER_RAW) MO(FI_ORDER_RAS) \
	MO(FI_ORDER_WAR) MO(FI_ORDER_WAW) MO(FI_ORDER_WAS) \
	MO(FI_ORDER_SAR) MO(FI_ORDER_SAW) MO(FI_ORDER_SAS)

#define MO(SYM) \
	if (*flags & SYM) \
		strcat(buf, #SYM ", ");

	MSG_ORDER

#undef MO

	return strcat(buf, "]\n");
}

static char *_pp_caps(char *buf, const uint64_t *flags)
{
	strcat(buf, "caps: [");

#define DOM_CAPS \
	DC(FI_WRITE_COHERENT) DC(FI_CONTEXT) DC(FI_LOCAL_MR)

#define DC(SYM) \
	if (*flags & SYM) \
		strcat(buf, #SYM ", ");

	DOM_CAPS

#undef DC

	return strcat(buf, "]\n");
}

static char *_pp_fi_threading(char *buf, const enum fi_threading *threadv)
{
	strcat(buf, "threading:\t");


	switch (*threadv) {
	EN(FI_THREAD_UNSPEC)
	EN(FI_THREAD_SAFE)
	EN(FI_THREAD_PROGRESS)
	default:
		return strcat(buf, "Unknown\n");
	}
}

static char *_pp_fi_progress(char *buf, const enum fi_progress *prog, const char* label)
{
	strcatf(buf, "%s:\t", label);

	switch (*prog) {
	EN(FI_PROGRESS_UNSPEC)
	EN(FI_PROGRESS_AUTO)
	EN(FI_PROGRESS_MANUAL)
	default:
		return strcat(buf, "Unknown\n");
	}
}

static char *_pp_fi_ep_attr(char *buf, const struct fi_ep_attr *ptr, const char *indent)
{
	strcat(buf, "fi_ep_attr:\n");

	strcat(buf, indent);
	_pp_fi_proto(buf, (const enum fi_proto*) &ptr->protocol);

	strcatf(buf, "%smax_msg_size:\t%d\n", indent, ptr->max_msg_size);
	strcatf(buf, "%sinject_size:\t%d\n", indent, ptr->inject_size);
	strcatf(buf, "%stotal_buffered_recv:\t%d\n", indent, ptr->total_buffered_recv);
	strcatf(buf, "%smax_order_raw_size:\t%d\n", indent, ptr->max_order_raw_size);
	strcatf(buf, "%smax_order_war_size:\t%d\n", indent, ptr->max_order_war_size);
	strcatf(buf, "%smax_order_waw_size:\t%d\n", indent, ptr->max_order_waw_size);
	strcatf(buf, "%smem_tag_format:\t%016llx\n", indent, ptr->mem_tag_format);

	strcat(buf, indent);
	_pp_msg_order(buf, &ptr->msg_order);

	strcatf(buf, "%stx_ctx_cnt:\t%d\n", indent, ptr->tx_ctx_cnt);
	strcatf(buf, "%srx_ctx_cnt:\t%d\n", indent, ptr->rx_ctx_cnt);

	return buf;
}

static char *_pp_fi_domain_attr(char *buf, const struct fi_domain_attr *ptr, const char *indent)
{
	strcat(buf, "fi_domain_attr:\n");

	strcatf(buf, "%sname:\t%s\n", indent, ptr->name);

	strcat(buf, indent);
	_pp_caps(buf, &ptr->caps);

	strcat(buf, indent);
	_pp_fi_threading(buf, &ptr->threading);

	strcat(buf, indent);
	_pp_fi_progress(buf, &ptr->control_progress, "control_progress");

	strcat(buf, indent);
	_pp_fi_progress(buf, &ptr->data_progress, "data_progress");

	strcatf(buf, "%smr_key_size:\t%d\n", indent, ptr->mr_key_size);
	strcatf(buf, "%seq_data_size:\t%d\n", indent, ptr->eq_data_size);
	strcatf(buf, "%sep_cnt:\t%d\n", indent, ptr->ep_cnt);
	strcatf(buf, "%stx_ctx_cnt:\t%d\n", indent, ptr->tx_ctx_cnt);
	strcatf(buf, "%srx_ctx_cnt:\t%d\n", indent, ptr->rx_ctx_cnt);
	strcatf(buf, "%smax_ep_tx_ctx:\t%d\n", indent, ptr->max_ep_tx_ctx);
	strcatf(buf, "%smax_ep_rx_ctx:\t%d\n", indent, ptr->max_ep_rx_ctx);

	return buf;
}

static char *_pp_fi_fabric_attr(char *buf, const struct fi_fabric_attr *ptr, const char *indent)
{
	strcat(buf, "fi_fabric_attr:\n");

	strcatf(buf, "%sname:\t\t%s\n", indent, ptr->name);
	strcatf(buf, "%sprov_name:\t%s\n", indent, ptr->prov_name);
	strcatf(buf, "%sprov_version:\t%d\n", indent, ptr->prov_version);

	return buf;
}

static char *_pp_fi_info(char *buf, const struct fi_info *v, const char *indent)
{
	char *rindent = "    ";

	strcat(buf, "fi_info:\n");

	strcat(buf, indent);
	_pp_fi_ep_type(buf, (const enum fi_ep_type*) &v->type);

	strcat(buf, indent);
	_pp_ep_cap(buf, &v->ep_cap);

	strcat(buf, indent);
	_pp_op_flags(buf, &v->op_flags);

	strcat(buf, indent);
	_pp_fi_addr_format(buf, &v->addr_format);

	strcatf(buf, "%ssource_addr:\t%s\n", indent, v->src_addr);
	strcatf(buf, "%sdest_addr:\t%s\n", indent, v->dest_addr);
	strcatf(buf, "%sconnreq:\t%d\n", indent, v->connreq);

	strcat(buf, indent);
	_pp_fi_ep_attr(buf, v->ep_attr, rindent);

	strcat(buf, indent);
	_pp_fi_domain_attr(buf, v->domain_attr, rindent);

	strcat(buf, indent);
	_pp_fi_fabric_attr(buf, v->fabric_attr, rindent);

	return buf;
}

char *fi_tostr(void *ptr, enum fi_pp_type tp)
{
	char *indent = "  ";
	char *buf;

	if (!ptr)
		return NULL;

	buf = calloc(4096, sizeof (*buf));

	if (!buf)
		return NULL;

	switch (tp) {
	case FI_PP_INFO:
		return _pp_fi_info(buf, (struct fi_info*) ptr, indent);
	case FI_PP_EP_TYPE:
		return _pp_fi_ep_type(buf, (const enum fi_ep_type*) ptr);
	case FI_PP_EP_CAP:
		return _pp_ep_cap(buf, (const uint64_t*) ptr);
	case FI_PP_OP_FLAGS:
		return _pp_op_flags(buf, (uint64_t*) ptr);
	case FI_PP_ADDR_FORMAT:
		return _pp_fi_addr_format(buf, (const enum fi_addr_format*) ptr);
	case FI_PP_EP_ATTR:
		return _pp_fi_ep_attr(buf, (const struct fi_ep_attr*) ptr, indent);
	case FI_PP_DOMAIN_ATTR:
		return _pp_fi_domain_attr(buf, (const struct fi_domain_attr*) ptr, indent);
	case FI_PP_FABRIC_ATTR:
		return _pp_fi_fabric_attr(buf, (const struct fi_fabric_attr*) ptr, indent);
	case FI_PP_CAPS:
		return _pp_caps(buf, (const uint64_t*) ptr);
	case FI_PP_THREADING:
		return _pp_fi_threading(buf, (const enum fi_threading*) ptr);
	case FI_PP_PROGRESS:
		return _pp_fi_progress(buf, (const enum fi_progress*) ptr, "progress");
	case FI_PP_PROTO:
		return _pp_fi_proto(buf, (const enum fi_proto*) ptr);
	case FI_PP_MSG_ORDER:
		return _pp_msg_order(buf, (const uint64_t*) ptr);
	default:
		return strcat(buf, "Unknown type");
	}
}

#undef EN
