/*
 * Copyright (c) 2014-2017 Intel Corp., Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "ofi.h"
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_collective.h>


/* Print fi_info and related structs, enums, OR_able flags, addresses.
 *
 * Each printable type should be well formatted YAML.
 *
 * A struct is a dictionary containing one key named after the struct tag
 * which contains a dictionary of member-value mappings. The struct member
 * keys are the field names (not the types).
 *
 * Enum values are currently just bare strings.
 * OR-able flags are a list of the values, ie: [ VAL1, VAL2 ]
 *
 * YAML does not contain tabs.
 * Indentation delineates lists and dictionaries (or they can be inline).
 *
 * Printing functions are generally named after this pattern:
 *
 * struct fi_info : ofi_tostr_info(..., struct fi_info, ...)
 * fi_info->caps  : ofi_tostr_caps(..., typeof(caps), ...)
 */

#define OFI_BUFSIZ 8192

static void ofi_tostr_fid(const char *label, char *buf, size_t buflen, const struct fid *fid)
{
	if (!fid || !FI_CHECK_OP(fid->ops, struct fi_ops, tostr))
		ofi_strncatf(buf, buflen, "%s%p\n", label, fid);
	else
		fid->ops->tostr(fid, buf, buflen - strnlen(buf, buflen));
}

static void ofi_tostr_opflags(char *buf, size_t buflen, uint64_t flags)
{
	IFFLAGSTR(flags, FI_MULTICAST);

	IFFLAGSTR(flags, FI_MULTI_RECV);
	IFFLAGSTR(flags, FI_REMOTE_CQ_DATA);
	IFFLAGSTR(flags, FI_MORE);
	IFFLAGSTR(flags, FI_PEEK);
	IFFLAGSTR(flags, FI_TRIGGER);
	IFFLAGSTR(flags, FI_FENCE);

	IFFLAGSTR(flags, FI_COMPLETION);
	IFFLAGSTR(flags, FI_INJECT);
	IFFLAGSTR(flags, FI_INJECT_COMPLETE);
	IFFLAGSTR(flags, FI_TRANSMIT_COMPLETE);
	IFFLAGSTR(flags, FI_DELIVERY_COMPLETE);
	IFFLAGSTR(flags, FI_AFFINITY);

	IFFLAGSTR(flags, FI_CLAIM);
	IFFLAGSTR(flags, FI_DISCARD);

	ofi_remove_comma(buf);
}

static void ofi_tostr_addr_format(char *buf, size_t buflen, uint32_t addr_format)
{
	switch (addr_format) {
	CASEENUMSTR(FI_FORMAT_UNSPEC);
	CASEENUMSTR(FI_SOCKADDR);
	CASEENUMSTR(FI_SOCKADDR_IN);
	CASEENUMSTR(FI_SOCKADDR_IN6);
	CASEENUMSTR(FI_SOCKADDR_IB);
	CASEENUMSTR(FI_ADDR_PSMX);
	CASEENUMSTR(FI_ADDR_PSMX2);
	CASEENUMSTR(FI_ADDR_GNI);
	CASEENUMSTR(FI_ADDR_BGQ);
	CASEENUMSTR(FI_ADDR_MLX);
	CASEENUMSTR(FI_ADDR_STR);
	CASEENUMSTR(FI_ADDR_IB_UD);
	CASEENUMSTR(FI_ADDR_EFA);
	default:
		if (addr_format & FI_PROV_SPECIFIC)
			ofi_strncatf(buf, buflen, "Provider specific");
		else
			ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_progress(char *buf, size_t buflen, enum fi_progress progress)
{
	switch (progress) {
	CASEENUMSTR(FI_PROGRESS_UNSPEC);
	CASEENUMSTR(FI_PROGRESS_AUTO);
	CASEENUMSTR(FI_PROGRESS_MANUAL);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_threading(char *buf, size_t buflen, enum fi_threading threading)
{
	switch (threading) {
	CASEENUMSTR(FI_THREAD_UNSPEC);
	CASEENUMSTR(FI_THREAD_SAFE);
	CASEENUMSTR(FI_THREAD_FID);
	CASEENUMSTR(FI_THREAD_DOMAIN);
	CASEENUMSTR(FI_THREAD_COMPLETION);
	CASEENUMSTR(FI_THREAD_ENDPOINT);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_msgorder(char *buf, size_t buflen, int64_t flags)
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
	IFFLAGSTR(flags, FI_ORDER_RMA_RAR);
	IFFLAGSTR(flags, FI_ORDER_RMA_RAW);
	IFFLAGSTR(flags, FI_ORDER_RMA_WAR);
	IFFLAGSTR(flags, FI_ORDER_RMA_WAW);
	IFFLAGSTR(flags, FI_ORDER_ATOMIC_RAR);
	IFFLAGSTR(flags, FI_ORDER_ATOMIC_RAW);
	IFFLAGSTR(flags, FI_ORDER_ATOMIC_WAR);
	IFFLAGSTR(flags, FI_ORDER_ATOMIC_WAW);

	ofi_remove_comma(buf);
}

static void ofi_tostr_comporder(char *buf, size_t buflen, uint64_t flags)
{
	if ((flags & FI_ORDER_STRICT) == FI_ORDER_NONE) {
		ofi_strncatf(buf, buflen, "FI_ORDER_NONE, ");
	} else if ((flags & FI_ORDER_STRICT) == FI_ORDER_STRICT) {
		ofi_strncatf(buf, buflen, "FI_ORDER_STRICT, ");
	}

	IFFLAGSTR(flags, FI_ORDER_DATA);

	ofi_remove_comma(buf);
}

static void ofi_tostr_caps(char *buf, size_t buflen, uint64_t caps)
{
	IFFLAGSTR(caps, FI_MSG);
	IFFLAGSTR(caps, FI_RMA);
	IFFLAGSTR(caps, FI_TAGGED);
	IFFLAGSTR(caps, FI_ATOMIC);
	IFFLAGSTR(caps, FI_MULTICAST);
	IFFLAGSTR(caps, FI_COLLECTIVE);

	IFFLAGSTR(caps, FI_READ);
	IFFLAGSTR(caps, FI_WRITE);
	IFFLAGSTR(caps, FI_RECV);
	IFFLAGSTR(caps, FI_SEND);
	IFFLAGSTR(caps, FI_REMOTE_READ);
	IFFLAGSTR(caps, FI_REMOTE_WRITE);

	IFFLAGSTR(caps, FI_MULTI_RECV);
	IFFLAGSTR(caps, FI_REMOTE_CQ_DATA);
	IFFLAGSTR(caps, FI_TRIGGER);
	IFFLAGSTR(caps, FI_FENCE);

	IFFLAGSTR(caps, FI_VARIABLE_MSG);
	IFFLAGSTR(caps, FI_RMA_PMEM);
	IFFLAGSTR(caps, FI_SOURCE_ERR);
	IFFLAGSTR(caps, FI_LOCAL_COMM);
	IFFLAGSTR(caps, FI_REMOTE_COMM);
	IFFLAGSTR(caps, FI_SHARED_AV);
	IFFLAGSTR(caps, FI_RMA_EVENT);
	IFFLAGSTR(caps, FI_SOURCE);
	IFFLAGSTR(caps, FI_NAMED_RX_CTX);
	IFFLAGSTR(caps, FI_DIRECTED_RECV);
	IFFLAGSTR(caps, FI_HMEM);

	ofi_remove_comma(buf);
}

static void ofi_tostr_ep_type(char *buf, size_t buflen, enum fi_ep_type ep_type)
{
	switch (ep_type) {
	CASEENUMSTR(FI_EP_UNSPEC);
	CASEENUMSTR(FI_EP_MSG);
	CASEENUMSTR(FI_EP_DGRAM);
	CASEENUMSTR(FI_EP_RDM);
	CASEENUMSTR(FI_EP_SOCK_STREAM);
	CASEENUMSTR(FI_EP_SOCK_DGRAM);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_protocol(char *buf, size_t buflen, uint32_t protocol)
{
	switch (protocol) {
	CASEENUMSTR(FI_PROTO_UNSPEC);
	CASEENUMSTR(FI_PROTO_RDMA_CM_IB_RC);
	CASEENUMSTR(FI_PROTO_IWARP);
	CASEENUMSTR(FI_PROTO_IB_UD);
	CASEENUMSTR(FI_PROTO_PSMX);
	CASEENUMSTR(FI_PROTO_PSMX2);
	CASEENUMSTR(FI_PROTO_UDP);
	CASEENUMSTR(FI_PROTO_SOCK_TCP);
	CASEENUMSTR(FI_PROTO_IB_RDM);
	CASEENUMSTR(FI_PROTO_IWARP_RDM);
	CASEENUMSTR(FI_PROTO_GNI);
	CASEENUMSTR(FI_PROTO_RXM);
	CASEENUMSTR(FI_PROTO_RXD);
	CASEENUMSTR(FI_PROTO_MLX);
	CASEENUMSTR(FI_PROTO_NETWORKDIRECT);
	CASEENUMSTR(FI_PROTO_SHM);
	CASEENUMSTR(FI_PROTO_RSTREAM);
	CASEENUMSTR(FI_PROTO_RDMA_CM_IB_XRC);
	CASEENUMSTR(FI_PROTO_EFA);
	default:
		if (protocol & FI_PROV_SPECIFIC)
			ofi_strncatf(buf, buflen, "Provider specific");
		else
			ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_mode(char *buf, size_t buflen, uint64_t mode)
{
	IFFLAGSTR(mode, FI_CONTEXT);
	IFFLAGSTR(mode, FI_MSG_PREFIX);
	IFFLAGSTR(mode, FI_ASYNC_IOV);
	IFFLAGSTR(mode, FI_RX_CQ_DATA);
	IFFLAGSTR(mode, FI_LOCAL_MR);
	IFFLAGSTR(mode, FI_NOTIFY_FLAGS_ONLY);
	IFFLAGSTR(mode, FI_RESTRICTED_COMP);
	IFFLAGSTR(mode, FI_CONTEXT2);
	IFFLAGSTR(mode, FI_BUFFERED_RECV);

	ofi_remove_comma(buf);
}

static void ofi_tostr_addr(char *buf, size_t buflen, uint32_t addr_format, void *addr)
{
	char *p;
	size_t len;

	p = buf + strlen(buf);

	if (addr == NULL) {
		ofi_strncatf(p, buflen, "(null)");
		return;
	}

	len = 64;
	ofi_straddr(p, &len, addr_format, addr);
}

static void ofi_tostr_tx_attr(char *buf, size_t buflen, const struct fi_tx_attr *attr,
				const char *prefix)
{
	if (!attr) {
		ofi_strncatf(buf, buflen, "%sfi_tx_attr: (null)\n", prefix);
		return;
	}

	ofi_strncatf(buf, buflen, "%sfi_tx_attr:\n", prefix);
	ofi_strncatf(buf, buflen, "%s%scaps: [ ", prefix, TAB);
	ofi_tostr_caps(buf, buflen, attr->caps);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%smode: [ ", prefix, TAB);
	ofi_tostr_mode(buf, buflen, attr->mode);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%sop_flags: [ ", prefix, TAB);
	ofi_tostr_opflags(buf, buflen, attr->op_flags);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%smsg_order: [ ", prefix, TAB);
	ofi_tostr_msgorder(buf, buflen, attr->msg_order);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%scomp_order: [ ", prefix, TAB);
	ofi_tostr_comporder(buf, buflen, attr->comp_order);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%sinject_size: %zu\n", prefix, TAB, attr->inject_size);
	ofi_strncatf(buf, buflen, "%s%ssize: %zu\n", prefix, TAB, attr->size);
	ofi_strncatf(buf, buflen, "%s%siov_limit: %zu\n", prefix, TAB, attr->iov_limit);
	ofi_strncatf(buf, buflen, "%s%srma_iov_limit: %zu\n", prefix, TAB, attr->rma_iov_limit);
}

static void ofi_tostr_rx_attr(char *buf, size_t buflen, const struct fi_rx_attr *attr,
				const char *prefix)
{
	if (!attr) {
		ofi_strncatf(buf, buflen, "%sfi_rx_attr: (null)\n", prefix);
		return;
	}

	ofi_strncatf(buf, buflen, "%sfi_rx_attr:\n", prefix);
	ofi_strncatf(buf, buflen ,"%s%scaps: [ ", prefix, TAB);
	ofi_tostr_caps(buf, buflen, attr->caps);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%smode: [ ", prefix, TAB);
	ofi_tostr_mode(buf, buflen, attr->mode);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%sop_flags: [ ", prefix, TAB);
	ofi_tostr_opflags(buf, buflen, attr->op_flags);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%smsg_order: [ ", prefix, TAB);
	ofi_tostr_msgorder(buf, buflen, attr->msg_order);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%scomp_order: [ ", prefix, TAB);
	ofi_tostr_comporder(buf, buflen, attr->comp_order);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%stotal_buffered_recv: %zu\n", prefix, TAB, attr->total_buffered_recv);
	ofi_strncatf(buf, buflen, "%s%ssize: %zu\n", prefix, TAB, attr->size);
	ofi_strncatf(buf, buflen, "%s%siov_limit: %zu\n", prefix, TAB, attr->iov_limit);
}

static void ofi_tostr_ep_attr(char *buf, size_t buflen, const struct fi_ep_attr *attr, const char *prefix)
{
	if (!attr) {
		ofi_strncatf(buf, buflen, "%sfi_ep_attr: (null)\n", prefix);
		return;
	}

	ofi_strncatf(buf, buflen, "%sfi_ep_attr:\n", prefix);
	ofi_strncatf(buf, buflen, "%s%stype: ", prefix, TAB);
	ofi_tostr_ep_type(buf, buflen, attr->type);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%sprotocol: ", prefix, TAB);
	ofi_tostr_protocol(buf, buflen, attr->protocol);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%sprotocol_version: %d\n", prefix, TAB, attr->protocol_version);
	ofi_strncatf(buf, buflen, "%s%smax_msg_size: %zu\n", prefix, TAB, attr->max_msg_size);
	ofi_strncatf(buf, buflen, "%s%smsg_prefix_size: %zu\n", prefix, TAB, attr->msg_prefix_size);
	ofi_strncatf(buf, buflen, "%s%smax_order_raw_size: %zu\n", prefix, TAB, attr->max_order_raw_size);
	ofi_strncatf(buf, buflen, "%s%smax_order_war_size: %zu\n", prefix, TAB, attr->max_order_war_size);
	ofi_strncatf(buf, buflen, "%s%smax_order_waw_size: %zu\n", prefix, TAB, attr->max_order_waw_size);
	ofi_strncatf(buf, buflen, "%s%smem_tag_format: 0x%016llx\n", prefix, TAB, attr->mem_tag_format);

	ofi_strncatf(buf, buflen, "%s%stx_ctx_cnt: %zu\n", prefix, TAB, attr->tx_ctx_cnt);
	ofi_strncatf(buf, buflen, "%s%srx_ctx_cnt: %zu\n", prefix, TAB, attr->rx_ctx_cnt);

	ofi_strncatf(buf, buflen, "%s%sauth_key_size: %zu\n", prefix, TAB, attr->auth_key_size);
}

static void ofi_tostr_resource_mgmt(char *buf, size_t buflen, enum fi_resource_mgmt rm)
{
	switch (rm) {
	CASEENUMSTR(FI_RM_UNSPEC);
	CASEENUMSTR(FI_RM_DISABLED);
	CASEENUMSTR(FI_RM_ENABLED);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_av_type(char *buf, size_t buflen, enum fi_av_type type)
{
	switch (type) {
	CASEENUMSTR(FI_AV_UNSPEC);
	CASEENUMSTR(FI_AV_MAP);
	CASEENUMSTR(FI_AV_TABLE);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_mr_mode(char *buf, size_t buflen, int mr_mode)
{
	IFFLAGSTR(mr_mode, FI_MR_BASIC);
	IFFLAGSTR(mr_mode, FI_MR_SCALABLE);
	IFFLAGSTR(mr_mode, FI_MR_LOCAL);
	IFFLAGSTR(mr_mode, FI_MR_RAW);
	IFFLAGSTR(mr_mode, FI_MR_VIRT_ADDR);
	IFFLAGSTR(mr_mode, FI_MR_ALLOCATED);
	IFFLAGSTR(mr_mode, FI_MR_PROV_KEY);
	IFFLAGSTR(mr_mode, FI_MR_MMU_NOTIFY);
	IFFLAGSTR(mr_mode, FI_MR_RMA_EVENT);
	IFFLAGSTR(mr_mode, FI_MR_ENDPOINT);
	IFFLAGSTR(mr_mode, FI_MR_HMEM);

	ofi_remove_comma(buf);
}

static void ofi_tostr_op_type(char *buf, size_t buflen, int op_type)
{
	switch (op_type) {
	CASEENUMSTR(FI_OP_RECV);
	CASEENUMSTR(FI_OP_SEND);
	CASEENUMSTR(FI_OP_TRECV);
	CASEENUMSTR(FI_OP_TSEND);
	CASEENUMSTR(FI_OP_READ);
	CASEENUMSTR(FI_OP_WRITE);
	CASEENUMSTR(FI_OP_ATOMIC);
	CASEENUMSTR(FI_OP_FETCH_ATOMIC);
	CASEENUMSTR(FI_OP_COMPARE_ATOMIC);
	CASEENUMSTR(FI_OP_CNTR_SET);
	CASEENUMSTR(FI_OP_CNTR_ADD);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_domain_attr(char *buf, size_t buflen,
				 const struct fi_domain_attr *attr, const char *prefix)
{
	if (!attr) {
		ofi_strncatf(buf, buflen, "%sfi_domain_attr: (null)\n", prefix);
		return;
	}

	ofi_strncatf(buf, buflen, "%sfi_domain_attr:\n", prefix);

	ofi_strncatf(buf, buflen, "%s%sdomain: 0x%x\n", prefix, TAB, attr->domain);

	ofi_strncatf(buf, buflen, "%s%sname: %s\n", prefix, TAB, attr->name);
	ofi_strncatf(buf, buflen, "%s%sthreading: ", prefix, TAB);
	ofi_tostr_threading(buf, buflen, attr->threading);
	ofi_strncatf(buf, buflen, "\n");

	ofi_strncatf(buf, buflen, "%s%scontrol_progress: ", prefix,TAB);
	ofi_tostr_progress(buf, buflen, attr->control_progress);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%sdata_progress: ", prefix, TAB);
	ofi_tostr_progress(buf, buflen, attr->data_progress);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%sresource_mgmt: ", prefix, TAB);
	ofi_tostr_resource_mgmt(buf, buflen, attr->resource_mgmt);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%sav_type: ", prefix, TAB);
	ofi_tostr_av_type(buf, buflen, attr->av_type);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%s%smr_mode: [ ", prefix, TAB);
	ofi_tostr_mr_mode(buf, buflen, attr->mr_mode);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%smr_key_size: %zu\n", prefix, TAB, attr->mr_key_size);
	ofi_strncatf(buf, buflen, "%s%scq_data_size: %zu\n", prefix, TAB, attr->cq_data_size);
	ofi_strncatf(buf, buflen, "%s%scq_cnt: %zu\n", prefix, TAB, attr->cq_cnt);
	ofi_strncatf(buf, buflen, "%s%sep_cnt: %zu\n", prefix, TAB, attr->ep_cnt);
	ofi_strncatf(buf, buflen, "%s%stx_ctx_cnt: %zu\n", prefix, TAB, attr->tx_ctx_cnt);
	ofi_strncatf(buf, buflen, "%s%srx_ctx_cnt: %zu\n", prefix, TAB, attr->rx_ctx_cnt);
	ofi_strncatf(buf, buflen, "%s%smax_ep_tx_ctx: %zu\n", prefix, TAB, attr->max_ep_tx_ctx);
	ofi_strncatf(buf, buflen, "%s%smax_ep_rx_ctx: %zu\n", prefix, TAB, attr->max_ep_rx_ctx);
	ofi_strncatf(buf, buflen, "%s%smax_ep_stx_ctx: %zu\n", prefix, TAB, attr->max_ep_stx_ctx);
	ofi_strncatf(buf, buflen, "%s%smax_ep_srx_ctx: %zu\n", prefix, TAB, attr->max_ep_srx_ctx);
	ofi_strncatf(buf, buflen, "%s%scntr_cnt: %zu\n", prefix, TAB, attr->cntr_cnt);
	ofi_strncatf(buf, buflen, "%s%smr_iov_limit: %zu\n", prefix, TAB, attr->mr_iov_limit);

	ofi_strncatf(buf, buflen, "%scaps: [ ", TAB);
	ofi_tostr_caps(buf, buflen, attr->caps);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%smode: [ ", TAB);
	ofi_tostr_mode(buf, buflen, attr->mode);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%s%sauth_key_size: %zu\n", prefix, TAB, attr->auth_key_size);
	ofi_strncatf(buf, buflen, "%s%smax_err_data: %zu\n", prefix, TAB, attr->max_err_data);
	ofi_strncatf(buf, buflen, "%s%smr_cnt: %zu\n", prefix, TAB, attr->mr_cnt);
}

static void ofi_tostr_fabric_attr(char *buf, size_t buflen,
				 const struct fi_fabric_attr *attr, const char *prefix)
{
	if (!attr) {
		ofi_strncatf(buf, buflen, "%sfi_fabric_attr: (null)\n", prefix);
		return;
	}

	ofi_strncatf(buf, buflen, "%sfi_fabric_attr:\n", prefix);
	ofi_strncatf(buf, buflen, "%s%sname: %s\n", prefix, TAB, attr->name);
	ofi_strncatf(buf, buflen, "%s%sprov_name: %s\n", prefix, TAB, attr->prov_name);
	ofi_strncatf(buf, buflen, "%s%sprov_version: %d.%d\n", prefix, TAB,
		FI_MAJOR(attr->prov_version), FI_MINOR(attr->prov_version));
	ofi_strncatf(buf, buflen, "%s%sapi_version: %d.%d\n", prefix, TAB,
		FI_MAJOR(attr->api_version), FI_MINOR(attr->api_version));
}

static void ofi_tostr_info(char *buf, size_t buflen, const struct fi_info *info)
{
	ofi_strncatf(buf, buflen, "fi_info:\n");
	ofi_strncatf(buf, buflen, "%scaps: [ ", TAB);
	ofi_tostr_caps(buf, buflen, info->caps);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%smode: [ ", TAB);
	ofi_tostr_mode(buf, buflen, info->mode);
	ofi_strncatf(buf, buflen, " ]\n");

	ofi_strncatf(buf, buflen, "%saddr_format: ", TAB);
	ofi_tostr_addr_format(buf, buflen, info->addr_format);
	ofi_strncatf(buf, buflen, "\n");

	ofi_strncatf(buf, buflen, "%ssrc_addrlen: %zu\n", TAB, info->src_addrlen);
	ofi_strncatf(buf, buflen, "%sdest_addrlen: %zu\n", TAB, info->dest_addrlen);
	ofi_strncatf(buf, buflen, "%ssrc_addr: ", TAB);
	ofi_tostr_addr(buf, buflen, info->addr_format, info->src_addr);
	ofi_strncatf(buf, buflen, "\n");
	ofi_strncatf(buf, buflen, "%sdest_addr: ", TAB);
	ofi_tostr_addr(buf, buflen, info->addr_format, info->dest_addr);
	ofi_strncatf(buf, buflen, "\n");
	ofi_tostr_fid(TAB "handle: ", buf, buflen, info->handle);

	ofi_tostr_tx_attr(buf, buflen, info->tx_attr, TAB);
	ofi_tostr_rx_attr(buf, buflen, info->rx_attr, TAB);
	ofi_tostr_ep_attr(buf, buflen, info->ep_attr, TAB);
	ofi_tostr_domain_attr(buf, buflen, info->domain_attr, TAB);
	ofi_tostr_fabric_attr(buf, buflen, info->fabric_attr, TAB);
	ofi_tostr_fid(TAB "nic_fid: ", buf, buflen, &info->nic->fid);
}

static void ofi_tostr_atomic_type(char *buf, size_t buflen,
				 enum fi_datatype type)
{
	switch (type) {
	CASEENUMSTR(FI_INT8);
	CASEENUMSTR(FI_UINT8);
	CASEENUMSTR(FI_INT16);
	CASEENUMSTR(FI_UINT16);
	CASEENUMSTR(FI_INT32);
	CASEENUMSTR(FI_UINT32);
	CASEENUMSTR(FI_INT64);
	CASEENUMSTR(FI_UINT64);
	CASEENUMSTR(FI_FLOAT);
	CASEENUMSTR(FI_DOUBLE);
	CASEENUMSTR(FI_FLOAT_COMPLEX);
	CASEENUMSTR(FI_DOUBLE_COMPLEX);
	CASEENUMSTR(FI_LONG_DOUBLE);
	CASEENUMSTR(FI_LONG_DOUBLE_COMPLEX);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_atomic_op(char *buf, size_t buflen, enum fi_op op)
{
	switch (op) {
	CASEENUMSTR(FI_MIN);
	CASEENUMSTR(FI_MAX);
	CASEENUMSTR(FI_SUM);
	CASEENUMSTR(FI_PROD);
	CASEENUMSTR(FI_LOR);
	CASEENUMSTR(FI_LAND);
	CASEENUMSTR(FI_BOR);
	CASEENUMSTR(FI_BAND);
	CASEENUMSTR(FI_LXOR);
	CASEENUMSTR(FI_BXOR);
	CASEENUMSTR(FI_ATOMIC_READ);
	CASEENUMSTR(FI_ATOMIC_WRITE);
	CASEENUMSTR(FI_CSWAP);
	CASEENUMSTR(FI_CSWAP_NE);
	CASEENUMSTR(FI_CSWAP_LE);
	CASEENUMSTR(FI_CSWAP_LT);
	CASEENUMSTR(FI_CSWAP_GE);
	CASEENUMSTR(FI_CSWAP_GT);
	CASEENUMSTR(FI_MSWAP);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_collective_op(char *buf, size_t buflen,
				 enum fi_collective_op op)
{
	switch (op) {
	CASEENUMSTR(FI_BARRIER);
	CASEENUMSTR(FI_BROADCAST);
	CASEENUMSTR(FI_ALLTOALL);
	CASEENUMSTR(FI_ALLREDUCE);
	CASEENUMSTR(FI_ALLGATHER);
	CASEENUMSTR(FI_REDUCE_SCATTER);
	CASEENUMSTR(FI_REDUCE);
	CASEENUMSTR(FI_SCATTER);
	CASEENUMSTR(FI_GATHER);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_version(char *buf, size_t buflen)
{
	ofi_strncatf(buf, buflen, VERSION);
	ofi_strncatf(buf, buflen, BUILD_ID);
}

static void ofi_tostr_eq_event(char *buf, size_t buflen, int type)
{
	switch (type) {
	CASEENUMSTR(FI_NOTIFY);
	CASEENUMSTR(FI_CONNREQ);
	CASEENUMSTR(FI_CONNECTED);
	CASEENUMSTR(FI_SHUTDOWN);
	CASEENUMSTR(FI_MR_COMPLETE);
	CASEENUMSTR(FI_AV_COMPLETE);
	CASEENUMSTR(FI_JOIN_COMPLETE);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

static void ofi_tostr_cq_event_flags(char *buf, size_t buflen, uint64_t flags)
{
	IFFLAGSTR(flags, FI_SEND);
	IFFLAGSTR(flags, FI_RECV);
	IFFLAGSTR(flags, FI_RMA);
	IFFLAGSTR(flags, FI_ATOMIC);
	IFFLAGSTR(flags, FI_MSG);
	IFFLAGSTR(flags, FI_TAGGED);
	IFFLAGSTR(flags, FI_READ);
	IFFLAGSTR(flags, FI_WRITE);
	IFFLAGSTR(flags, FI_REMOTE_READ);
	IFFLAGSTR(flags, FI_REMOTE_WRITE);
	IFFLAGSTR(flags, FI_REMOTE_CQ_DATA);
	IFFLAGSTR(flags, FI_MULTI_RECV);
	IFFLAGSTR(flags, FI_MORE);
	IFFLAGSTR(flags, FI_CLAIM);
	ofi_remove_comma(buf);
}

static void ofi_tostr_hmem_iface(char *buf, size_t buflen,
				 enum fi_hmem_iface iface)
{
	switch (iface) {
	CASEENUMSTR(FI_HMEM_SYSTEM);
	CASEENUMSTR(FI_HMEM_CUDA);
	default:
		ofi_strncatf(buf, buflen, "Unknown");
		break;
	}
}

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
char *DEFAULT_SYMVER_PRE(fi_tostr)(const void *data, enum fi_type datatype)
{
	static char *buf = NULL;
	const uint64_t *val64;
	const uint32_t *val32;
	const int *enumval;

	if (!data)
		return NULL;

	val64 = (const uint64_t *) data;
	val32 = (const uint32_t *) data;
	enumval = (const int *) data;

	if (!buf) {
		buf = calloc(OFI_BUFSIZ, 1);
		if (!buf)
			return NULL;
	}
	buf[0] = '\0';

	switch (datatype) {
	case FI_TYPE_INFO:
		ofi_tostr_info(buf, buflen, data);
		break;
	case FI_TYPE_EP_TYPE:
		ofi_tostr_ep_type(buf, buflen, *enumval);
		break;
	case FI_TYPE_CAPS:
		ofi_tostr_caps(buf, buflen, *val64);
		break;
	case FI_TYPE_OP_FLAGS:
		ofi_tostr_opflags(buf, buflen, *val64);
		break;
	case FI_TYPE_ADDR_FORMAT:
		ofi_tostr_addr_format(buf, buflen, *val32);
		break;
	case FI_TYPE_TX_ATTR:
		ofi_tostr_tx_attr(buf, buflen, data, "");
		break;
	case FI_TYPE_RX_ATTR:
		ofi_tostr_rx_attr(buf, buflen, data, "");
		break;
	case FI_TYPE_EP_ATTR:
		ofi_tostr_ep_attr(buf, buflen, data, "");
		break;
	case FI_TYPE_DOMAIN_ATTR:
		ofi_tostr_domain_attr(buf, buflen, data, "");
		break;
	case FI_TYPE_FABRIC_ATTR:
		ofi_tostr_fabric_attr(buf, buflen, data, "");
		break;
	case FI_TYPE_THREADING:
		ofi_tostr_threading(buf, buflen, *enumval);
		break;
	case FI_TYPE_PROGRESS:
		ofi_tostr_progress(buf, buflen, *enumval);
		break;
	case FI_TYPE_PROTOCOL:
		ofi_tostr_protocol(buf, buflen, *val32);
		break;
	case FI_TYPE_MSG_ORDER:
		ofi_tostr_msgorder(buf, buflen, *val64);
		break;
	case FI_TYPE_MODE:
		ofi_tostr_mode(buf, buflen, *val64);
		break;
	case FI_TYPE_AV_TYPE:
		ofi_tostr_av_type(buf, buflen, *enumval);
		break;
	case FI_TYPE_ATOMIC_TYPE:
		ofi_tostr_atomic_type(buf, buflen, *enumval);
		break;
	case FI_TYPE_ATOMIC_OP:
		ofi_tostr_atomic_op(buf, buflen, *enumval);
		break;
	case FI_TYPE_VERSION:
		ofi_tostr_version(buf, buflen);
		break;
	case FI_TYPE_EQ_EVENT:
		ofi_tostr_eq_event(buf, buflen, *enumval);
		break;
	case FI_TYPE_CQ_EVENT_FLAGS:
		ofi_tostr_cq_event_flags(buf, buflen, *val64);
		break;
	case FI_TYPE_MR_MODE:
		/* mr_mode was an enum converted to int flags */
		ofi_tostr_mr_mode(buf, buflen, *enumval);
		break;
	case FI_TYPE_OP_TYPE:
		ofi_tostr_op_type(buf, buflen, *enumval);
		break;
	case FI_TYPE_FID:
		ofi_tostr_fid("fid: ", buf, buflen, data);
		break;
	case FI_TYPE_COLLECTIVE_OP:
		ofi_tostr_collective_op(buf, buflen, *enumval);
		break;
	case FI_TYPE_HMEM_IFACE:
		ofi_tostr_hmem_iface(buf, buflen, *enumval);
		break;
	default:
		ofi_strncatf(buf, buflen, "Unknown type");
		break;
	}
	return buf;
}
DEFAULT_SYMVER(fi_tostr_, fi_tostr, FABRIC_1.0);

#undef CASEENUMSTR
#undef IFFLAGSTR
