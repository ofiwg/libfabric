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

int rxm_dup_addr(struct fi_info *dup, struct fi_info *info)
{
	dup->addr_format = info->addr_format;
	if (info->src_addr) {
		dup->src_addrlen = info->src_addrlen;
		dup->src_addr = mem_dup(info->src_addr, info->src_addrlen);
		if (dup->src_addr == NULL)
			return -FI_ENOMEM;
	}
	if (info->dest_addr) {
		dup->dest_addrlen = info->dest_addrlen;
		dup->dest_addr = mem_dup(info->dest_addr, info->dest_addrlen);
		if (dup->dest_addr == NULL) {
			free(dup->src_addr);
			dup->src_addr = NULL;
			return -FI_ENOMEM;
		}
	}
	return 0;
}

int rxm_alter_layer_info(struct fi_info *layer_info, struct fi_info **base_info)
{
	struct fi_info *info;

	if (!(info = fi_allocinfo()))
		return -FI_ENOMEM;

	/* TODO choose base_info attr based on layer_info attr */
	info->caps = FI_MSG;
	info->mode = FI_LOCAL_MR;
	info->ep_attr->type = FI_EP_MSG;

	if (!layer_info)
		goto out;

	if (rxm_dup_addr(layer_info, info))
		goto err;

	if (layer_info->domain_attr && layer_info->domain_attr->name &&
			!(info->domain_attr->name =
			ofi_strdup_less_prefix(layer_info->domain_attr->name,
			rxm_info.domain_attr->name))) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unable to alter layer_info domain name\n");
		goto err;
	}
out:
	*base_info = info;
	return 0;
err:
	fi_freeinfo(info);
	return -FI_ENOMEM;
}

int rxm_alter_base_info(struct fi_info *base_info, struct fi_info **layer_info)
{
	struct fi_info *info;

	if (!(info = fi_allocinfo()))
		return -FI_ENOMEM;

	// TODO choose caps based on base_info caps
	info->caps = rxm_info.caps;
	info->mode = rxm_info.mode;

	if (rxm_dup_addr(base_info, info))
		goto err;

	*info->tx_attr = *rxm_info.tx_attr;
	*info->rx_attr = *rxm_info.rx_attr;
	*info->ep_attr = *rxm_info.ep_attr;
	*info->domain_attr = *rxm_info.domain_attr;

	if (!(info->domain_attr->name =
				ofi_strdup_add_prefix(base_info->domain_attr->name,
					rxm_info.domain_attr->name))) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC,
				"Unable to alter base prov domain name\n");
		goto err;
	}
	info->fabric_attr->prov_version = rxm_info.fabric_attr->prov_version;
	if (!(info->fabric_attr->name = strdup(base_info->fabric_attr->name)))
		goto err;

	*layer_info = info;
	return 0;
err:
	fi_freeinfo(info);
	return -FI_ENOMEM;
}

static int rxm_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, struct fi_info *hints, struct fi_info **info)
{
	return ofi_layered_prov_getinfo(version, node, service, flags, &rxm_prov, &rxm_info,
			hints, rxm_alter_layer_info, rxm_alter_base_info, 0, info);
}

static void rxm_fini(void)
{
	/* yawn */
}

struct fi_provider rxm_prov = {
	.name = "rxm",
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
