/*
 * Copyright (c) 2015-2018 Intel Corporation. All rights reserved.
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

#include "smr.h"

struct fi_tx_attr smr_tx_attr = {
	.caps = FI_MSG | FI_SEND | FI_READ | FI_WRITE,
	.comp_order = FI_ORDER_STRICT,
	.inject_size = SMR_INJECT_SIZE,
	.size = 1024,
	.iov_limit = SMR_IOV_LIMIT,
	.rma_iov_limit = SMR_IOV_LIMIT
};

struct fi_rx_attr smr_rx_attr = {
	.caps = FI_MSG | FI_RECV | FI_SOURCE,
	.comp_order = FI_ORDER_STRICT,
	.size = 1024,
	.iov_limit = SMR_IOV_LIMIT
};

struct fi_ep_attr smr_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_SHM,
	.protocol_version = 1,
	.max_msg_size = SIZE_MAX,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1
};

struct fi_domain_attr smr_domain_attr = {
	.name = "shm",
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_ENABLED,
	.av_type = FI_AV_UNSPEC,
	.mr_mode = FI_MR_VIRT_ADDR,
	.cq_data_size = sizeof_field(struct smr_msg_hdr, data),
	.cq_cnt = (1 << 10),
	.ep_cnt = (1 << 10),
	.tx_ctx_cnt = (1 << 10),
	.rx_ctx_cnt = (1 << 10),
	.max_ep_tx_ctx = 1,
	.max_ep_rx_ctx = 1
};

struct fi_fabric_attr smr_fabric_attr = {
	.name = "shm",
	.prov_version = FI_VERSION(SMR_MAJOR_VERSION, SMR_MINOR_VERSION)
};

struct fi_info smr_info = {
	.caps = FI_MSG | FI_SEND | FI_RECV | FI_SOURCE | FI_TAGGED | FI_RMA |
		FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
	.addr_format = FI_ADDR_STR,
	.tx_attr = &smr_tx_attr,
	.rx_attr = &smr_rx_attr,
	.ep_attr = &smr_ep_attr,
	.domain_attr = &smr_domain_attr,
	.fabric_attr = &smr_fabric_attr
};
