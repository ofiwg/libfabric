/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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

#include "tcpx.h"


static struct fi_tx_attr tcpx_tx_attr = {
	.caps = FI_MSG | FI_SEND,
	.comp_order = FI_ORDER_STRICT,
	.inject_size = 64,
	.size = 1024,
	.iov_limit = TCPX_IOV_LIMIT
};

static struct fi_rx_attr tcpx_rx_attr = {
	.caps = FI_MSG | FI_RECV,
	.comp_order = FI_ORDER_STRICT,
	.total_buffered_recv = 0,
	.size = 1024,
	.iov_limit = TCPX_IOV_LIMIT
};

static struct fi_ep_attr tcpx_ep_attr = {
	.type = FI_EP_MSG,
	.protocol = FI_PROTO_SOCK_TCP,
	.protocol_version = 0,
	.max_msg_size = SIZE_MAX,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1
};

static struct fi_domain_attr tcpx_domain_attr = {
	.name = "tcp",
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_AUTO,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = 0,
	.cq_cnt = 256,
	.ep_cnt = 8192,
	.tx_ctx_cnt = 8192,
	.rx_ctx_cnt = 8192,
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1
};

static struct fi_fabric_attr tcpx_fabric_attr = {
	.name = "TCP-IP",
	.prov_version = FI_VERSION(TCPX_MAJOR_VERSION, TCPX_MINOR_VERSION),
};

struct fi_info tcpx_info = {
	.caps = FI_MSG | FI_SEND | FI_RECV,
	.addr_format = FI_SOCKADDR,
	.tx_attr = &tcpx_tx_attr,
	.rx_attr = &tcpx_rx_attr,
	.ep_attr = &tcpx_ep_attr,
	.domain_attr = &tcpx_domain_attr,
	.fabric_attr = &tcpx_fabric_attr
};

struct util_prov tcpx_util_prov = {
	.prov = &tcpx_prov,
	.info = &tcpx_info,
	.flags = 0,
};
