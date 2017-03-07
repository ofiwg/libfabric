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

#include <rdma/fi_errno.h>

#include <prov.h>
#include "rxm.h"

int rxm_info_to_core(struct fi_info *rxm_info, struct fi_info *core_info)
{
	/* TODO choose core_info attr based on rxm_info attr */
	core_info->caps = FI_MSG;
	core_info->mode = FI_LOCAL_MR;
	core_info->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;
	core_info->ep_attr->type = FI_EP_MSG;

	return 0;
}

int rxm_info_to_rxm(struct fi_info *core_info, struct fi_info *info)
{
	// TODO choose caps based on core_info caps
	info->caps = rxm_info.caps;
	info->mode = rxm_info.mode;

	*info->tx_attr = *rxm_info.tx_attr;
	info->tx_attr->iov_limit = MIN(MIN(info->tx_attr->iov_limit,
			core_info->tx_attr->iov_limit),
			core_info->tx_attr->rma_iov_limit);

	*info->rx_attr = *rxm_info.rx_attr;
	info->rx_attr->iov_limit = MIN(info->rx_attr->iov_limit,
			core_info->rx_attr->iov_limit);

	*info->ep_attr = *rxm_info.ep_attr;
	info->ep_attr->max_msg_size = core_info->ep_attr->max_msg_size;
	*info->domain_attr = *rxm_info.domain_attr;

	return 0;
}

static int rxm_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	return ofix_getinfo(version, node, service, flags, &rxm_util_prov,
			    hints, rxm_info_to_core, rxm_info_to_rxm, info);
}

static void rxm_fini(void)
{
	/* yawn */
}

struct fi_provider rxm_prov = {
	.name = "ofi-rxm",
	.version = FI_VERSION(RXM_MAJOR_VERSION, RXM_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 3),
	.getinfo = rxm_getinfo,
	.fabric = rxm_fabric,
	.cleanup = rxm_fini
};

RXM_INI
{
	return &rxm_prov;
}
