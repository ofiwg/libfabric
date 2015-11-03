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

#include <string.h>
#include <prov/verbs/src/verbs_checks.h>
#include <prov/verbs/src/fi_verbs.h>

#include <rdma/fabric.h>
#include <rdma/fi_log.h>
#include <rdma/fi_errno.h>

extern struct fi_provider fi_ibv_prov;

int fi_ibv_check_fabric_attr(const struct fi_fabric_attr *attr,
		const struct fi_info *info)
{
	if (attr->name && strcmp(attr->name, info->fabric_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown fabric name\n");
		return -FI_ENODATA;
	}

	if (attr->prov_version > info->fabric_attr->prov_version) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported provider version\n");
		return -FI_ENODATA;
	}

	return 0;
}

int fi_ibv_check_ep_attr(const struct fi_ep_attr *attr,
		const struct fi_info *info)
{
	switch (attr->type) {
	case FI_EP_UNSPEC:
	case FI_EP_MSG:
    case FI_EP_RDM:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported endpoint type\n");
		return -FI_ENODATA;
	}

	switch (attr->protocol) {
	case FI_PROTO_UNSPEC:
	case FI_PROTO_RDMA_CM_IB_RC:
	case FI_PROTO_IWARP:
	case FI_PROTO_IB_UD:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported protocol\n");
		return -FI_ENODATA;
	}

	if (attr->protocol_version > 1) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Unsupported protocol version\n");
		return -FI_ENODATA;
	}

	if (attr->max_msg_size > info->ep_attr->max_msg_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Max message size too large\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_raw_size > info->ep_attr->max_order_raw_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_raw_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_war_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_war_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_order_waw_size > info->ep_attr->max_order_waw_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"max_order_waw_size exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->tx_ctx_cnt > info->domain_attr->max_ep_tx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"tx_ctx_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->rx_ctx_cnt > info->domain_attr->max_ep_rx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"rx_ctx_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	return 0;
}

int fi_ibv_check_rx_attr(const struct fi_rx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info)
{
	uint64_t compare_mode, check_mode;

	if (attr->caps & ~(info->rx_attr->caps)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->caps not supported\n");
		return -FI_ENODATA;
	}

	compare_mode = attr->mode ? attr->mode : hints->mode;

    if (info->ep_attr->type == FI_EP_MSG) {
        check_mode = (hints->caps & FI_RMA) ? info->rx_attr->mode : VERBS_MODE;
    } else {
        check_mode = VERBS_EP_RDM_MODE;
    }

	if ((compare_mode & check_mode) != check_mode) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->op_flags & ~(info->rx_attr->op_flags)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->op_flags not supported\n");
		return -FI_ENODATA;
	}

	if (attr->msg_order & ~(info->rx_attr->msg_order)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->msg_order not supported\n");
		return -FI_ENODATA;
	}

	if (attr->size > info->rx_attr->size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->size is greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->total_buffered_recv > info->rx_attr->total_buffered_recv) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->total_buffered_recv exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > info->rx_attr->iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given rx_attr->iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	return 0;
}

int fi_ibv_check_tx_attr(const struct fi_tx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info)
{
	if (attr->caps & ~(info->tx_attr->caps)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->caps not supported\n");
		return -FI_ENODATA;
	}

	if (((attr->mode ? attr->mode : hints->mode) &
				info->tx_attr->mode) != info->tx_attr->mode) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->op_flags & ~(info->tx_attr->op_flags)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->op_flags not supported\n");
		return -FI_ENODATA;
	}

	if (attr->msg_order & ~(info->tx_attr->msg_order)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->msg_order not supported\n");
		return -FI_ENODATA;
	}

	if (attr->size > info->tx_attr->size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->size is greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->iov_limit > info->tx_attr->iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	if (attr->rma_iov_limit > info->tx_attr->rma_iov_limit) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given tx_attr->rma_iov_limit greater than supported\n");
		return -FI_ENODATA;
	}

	return 0;
}

int fi_ibv_check_domain_attr(const struct fi_domain_attr *attr,
		const struct fi_info *info)
{
	if (attr->name && strcmp(attr->name, info->domain_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE, "Unknown domain name\n");
		return -FI_ENODATA;
	}

	switch (attr->threading) {
	case FI_THREAD_UNSPEC:
	case FI_THREAD_SAFE:
	case FI_THREAD_FID:
	case FI_THREAD_DOMAIN:
	case FI_THREAD_COMPLETION:
	case FI_THREAD_ENDPOINT:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Invalid threading model\n");
		return -FI_ENODATA;
	}

	switch (attr->control_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given control progress mode not supported\n");
		return -FI_ENODATA;
	}

	switch (attr->data_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"Given data progress mode not supported!\n");
		return -FI_ENODATA;
	}

	switch (attr->mr_mode) {
	case FI_MR_UNSPEC:
	case FI_MR_BASIC:
		break;
	default:
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"MR mode not supported\n");
		return -FI_ENODATA;
	}

	if (attr->mr_key_size > info->domain_attr->mr_key_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"MR key size too large\n");
		return -FI_ENODATA;
	}

	if (attr->cq_data_size > info->domain_attr->cq_data_size) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"CQ data size too large\n");
		return -FI_ENODATA;
	}

	if (attr->cq_cnt > info->domain_attr->cq_cnt) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"cq_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->ep_cnt > info->domain_attr->ep_cnt) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"ep_cnt exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_ep_tx_ctx > info->domain_attr->max_ep_tx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"domain_attr: max_ep_tx_ctx exceeds supported size\n");
		return -FI_ENODATA;
	}

	if (attr->max_ep_rx_ctx > info->domain_attr->max_ep_rx_ctx) {
		FI_INFO(&fi_ibv_prov, FI_LOG_CORE,
			"domain_attr: max_ep_rx_ctx exceeds supported size\n");
		return -FI_ENODATA;
	}

	return 0;
}

