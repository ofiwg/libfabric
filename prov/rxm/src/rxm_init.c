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

int rxm_info_to_core(uint32_t version, struct fi_info *hints,
		     struct fi_info *core_info)
{
	core_info->caps = FI_MSG;
	if (hints) {
		core_info->mode = hints->mode;

		if (FI_VERSION_LT(version, FI_VERSION(1, 5)))
			core_info->mode |= FI_LOCAL_MR;

		if (hints->domain_attr) {
			core_info->domain_attr->mr_mode =
				hints->domain_attr->mr_mode;

			if (FI_VERSION_GE(version, FI_VERSION(1, 5)))
				core_info->domain_attr->mr_mode |=
					(FI_MR_LOCAL | OFI_MR_BASIC_MAP);

			if (hints->domain_attr->caps)
				core_info->domain_attr->caps =
					hints->domain_attr->caps | FI_MSG;
			if (hints->domain_attr->mode)
				core_info->domain_attr->mode = hints->domain_attr->mode;
		}
	}
	core_info->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;
	core_info->ep_attr->type = FI_EP_MSG;

	return 0;
}

int rxm_info_to_rxm(uint32_t version, struct fi_info *core_info,
		    struct fi_info *info)
{
	info->caps = rxm_info.caps;
	info->mode = core_info->mode | rxm_info.mode;

	*info->tx_attr = *rxm_info.tx_attr;

	/* Export TX queue size same as that of MSG provider as we post TX
	 * operations directly */
	info->tx_attr->size = core_info->tx_attr->size;

	info->tx_attr->iov_limit = MIN(MIN(info->tx_attr->iov_limit,
			core_info->tx_attr->iov_limit),
			core_info->tx_attr->rma_iov_limit);

	*info->rx_attr = *rxm_info.rx_attr;
	info->rx_attr->iov_limit = MIN(info->rx_attr->iov_limit,
			core_info->rx_attr->iov_limit);

	*info->ep_attr = *rxm_info.ep_attr;
	info->ep_attr->max_msg_size = core_info->ep_attr->max_msg_size;

	*info->domain_attr = *rxm_info.domain_attr;
	info->domain_attr->mr_mode = core_info->domain_attr->mr_mode;
	info->domain_attr->cq_data_size = MIN(core_info->domain_attr->cq_data_size,
					      rxm_info.domain_attr->cq_data_size);

	return 0;
}

static int rxm_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *cur, *dup;
	int ret;

	ret = ofix_getinfo(version, node, service, flags, &rxm_util_prov, hints,
			   rxm_info_to_core, rxm_info_to_rxm, info);
	if (ret)
		return ret;

	/* If app supports FI_MR_LOCAL, prioritize requiring it for
	 * better performance. */
	if (hints && hints->domain_attr &&
	    (RXM_MR_LOCAL(hints))) {
		for (cur = *info; cur; cur = cur->next) {
			if (!RXM_MR_LOCAL(cur))
				continue;
			if (!(dup = fi_dupinfo(cur))) {
				fi_freeinfo(*info);
				return -FI_ENOMEM;
			}
			if (FI_VERSION_LT(version, FI_VERSION(1, 5)))
				dup->mode &= ~FI_LOCAL_MR;
			else
				dup->domain_attr->mr_mode &= ~FI_MR_LOCAL;
			dup->next = cur->next;
			cur->next = dup;
			cur = dup;
		}
	} else {
		for (cur = *info; cur; cur = cur->next) {
			if (FI_VERSION_LT(version, FI_VERSION(1, 5)))
				cur->mode &= ~FI_LOCAL_MR;
			else
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
	.name = "ofi-rxm",
	.version = FI_VERSION(RXM_MAJOR_VERSION, RXM_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 5),
	.getinfo = rxm_getinfo,
	.fabric = rxm_fabric,
	.cleanup = rxm_fini
};

RXM_INI
{
	return &rxm_prov;
}
