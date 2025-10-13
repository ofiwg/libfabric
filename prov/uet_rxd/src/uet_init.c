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

#include <rdma/fi_errno.h>

#include <ofi.h>
#include <ofi_prov.h>

#include "uet_proto.h"
#include "uet.h"

struct uet_env uet_env = {
	.spin_count	= 1000,
	.retry		= 1,
	.max_peers	= 1024,
	.max_unacked	= 128,
	.rescan		= -1,
};

char *uet_pkt_type_str[] = {
	UET_FOREACH_TYPE(OFI_STR)
};

static void uet_init_env(void)
{
	fi_param_get_int(&uet_prov, "spin_count", &uet_env.spin_count);
	fi_param_get_bool(&uet_prov, "retry", &uet_env.retry);
	fi_param_get_int(&uet_prov, "max_peers", &uet_env.max_peers);
	fi_param_get_int(&uet_prov, "max_unacked", &uet_env.max_unacked);
	fi_param_get_bool(&uet_prov, "rescan", &uet_env.rescan);
}

void uet_info_to_core_mr_modes(uint32_t version, const struct fi_info *hints,
			       struct fi_info *core_info)
{
	/* We handle FI_MR_BASIC and FI_MR_SCALABLE irrespective of version */
	if (hints && hints->domain_attr &&
	    (hints->domain_attr->mr_mode & (OFI_MR_SCALABLE | OFI_MR_BASIC))) {
		core_info->mode = OFI_LOCAL_MR;
		core_info->domain_attr->mr_mode = hints->domain_attr->mr_mode;
	} else if (FI_VERSION_LT(version, FI_VERSION(1, 5))) {
		core_info->mode |= OFI_LOCAL_MR;
		/* Specify FI_MR_UNSPEC (instead of FI_MR_BASIC) so that
		 * providers that support only FI_MR_SCALABLE aren't dropped */
		core_info->domain_attr->mr_mode = OFI_MR_UNSPEC;
	} else {
		core_info->domain_attr->mr_mode |= FI_MR_LOCAL;
		core_info->domain_attr->mr_mode |= OFI_MR_BASIC_MAP;
	}
}

int uet_info_to_core(uint32_t version, const struct fi_info *uet_info_in,
		     const struct fi_info *base_info, struct fi_info *core_info)
{
	uet_info_to_core_mr_modes(version, uet_info_in, core_info);
	core_info->caps = FI_MSG;
	core_info->mode = OFI_LOCAL_MR | FI_CONTEXT | FI_MSG_PREFIX;
	core_info->ep_attr->type = FI_EP_DGRAM;

	return 0;
}

int uet_info_to_rxd(uint32_t version, const struct fi_info *core_info,
		    const struct fi_info *base_info, struct fi_info *info)
{
	info->caps = ofi_pick_core_flags(uet_info.caps, core_info->caps,
					 FI_LOCAL_COMM | FI_REMOTE_COMM);
	info->mode = uet_info.mode;

	*info->tx_attr = *uet_info.tx_attr;
	info->tx_attr->inject_size = MIN(core_info->ep_attr->max_msg_size,
			UET_MAX_MTU_SIZE) - (sizeof(struct uet_base_hdr) +
			core_info->ep_attr->msg_prefix_size +
			sizeof(struct uet_rma_hdr) + (UET_IOV_LIMIT *
			sizeof(struct ofi_rma_iov)) + sizeof(struct uet_atom_hdr));

	*info->rx_attr = *uet_info.rx_attr;
	*info->ep_attr = *uet_info.ep_attr;
	*info->domain_attr = *uet_info.domain_attr;
	info->domain_attr->caps = ofi_pick_core_flags(uet_info.domain_attr->caps,
						core_info->domain_attr->caps,
						FI_LOCAL_COMM | FI_REMOTE_COMM);
	if (core_info->nic) {
		info->nic = ofi_nic_dup(core_info->nic);
		if (!info->nic)
			return -FI_ENOMEM;
	}
	return 0;
}

static int uet_getinfo(uint32_t version, const char *node, const char *service,
			uint64_t flags, const struct fi_info *hints,
			struct fi_info **info)
{
	if (uet_env.rescan > 0) /* Explicitly enabled */
		flags |= FI_RESCAN;
	else if (!uet_env.rescan) /* Explicitly disabled */
		flags &= ~FI_RESCAN;
	return ofix_getinfo(version, node, service, flags, &uet_util_prov,
			    hints, uet_info_to_core, uet_info_to_rxd, info);
}

static void uet_fini(void)
{
	/* yawn */
}

struct fi_provider uet_prov = {
	.name = OFI_UTIL_PREFIX "uet",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = uet_getinfo,
	.fabric = uet_fabric,
	.cleanup = uet_fini
};

UET_RXD_INI
{
	fi_param_define(&uet_prov, "spin_count", FI_PARAM_INT,
			"Number of iterations to receive packets (0 - infinite)");
	fi_param_define(&uet_prov, "retry", FI_PARAM_BOOL,
			"Toggle packet retrying (default: yes)");
	fi_param_define(&uet_prov, "max_peers", FI_PARAM_INT,
			"Maximum number of peers to track (default: 1024)");
	fi_param_define(&uet_prov, "max_unacked", FI_PARAM_INT,
			"Maximum number of packets to send at once (default: 128)");
	fi_param_define(&uet_prov, "rescan", FI_PARAM_BOOL,
			"Force or disable rescanning for network interface changes. "
			"Setting this to true will force rescanning on each fi_getinfo() invocation; "
			"setting it to false will disable rescanning. (default: unset)");

	uet_init_env();

	return &uet_prov;
}
