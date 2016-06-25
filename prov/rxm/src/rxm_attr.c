/*
 * Copyright (c) 2015-2016 Intel Corporation. All rights reserved.
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

#include "rxm.h"

struct fi_tx_attr rxm_tx_attr = {
	.caps = FI_MSG | FI_SEND,
};

struct fi_rx_attr rxm_rx_attr = {
	.caps = FI_MSG | FI_RECV,
};

struct fi_ep_attr rxm_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_RXM,
	.protocol_version = 0,
};

struct fi_domain_attr rxm_domain_attr = {
	.name = "rxm",
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_AUTO,
	.av_type = FI_AV_UNSPEC,
};

struct fi_fabric_attr rxm_fabric_attr = {
	.name = "",
	.prov_version = FI_VERSION(RXM_MAJOR_VERSION, RXM_MINOR_VERSION),
	.prov_name = "rxm",
};

struct fi_info rxm_info = {
	.caps = FI_MSG | FI_SEND | FI_RECV | FI_SOURCE,
	.addr_format = FI_SOCKADDR,
	.tx_attr = &rxm_tx_attr,
	.rx_attr = &rxm_rx_attr,
	.ep_attr = &rxm_ep_attr,
	.domain_attr = &rxm_domain_attr,
	.fabric_attr = &rxm_fabric_attr
};

struct util_prov rxm_util_prov = {
	.prov = &rxm_prov,
	.info = &rxm_info,
	.flags = 0,
};
