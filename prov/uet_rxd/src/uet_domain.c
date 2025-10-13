/*
 * Copyright (c) 2016-2017 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "uet.h"


static struct fi_ops_domain uet_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = uet_av_create,
	.cq_open = uet_cq_open,
	.endpoint = uet_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = uet_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = uet_query_atomic,
	.query_collective = fi_no_query_collective,
};

static int uet_domain_close(fid_t fid)
{
	int ret;
	struct uet_domain *uet_domain;

	uet_domain = container_of(fid, struct uet_domain, util_domain.domain_fid.fid);

	ret = fi_close(&uet_domain->dg_domain->fid);
	if (ret)
		return ret;

	ret = ofi_domain_close(&uet_domain->util_domain);
	if (ret)
		return ret;

	ofi_mr_map_close(&uet_domain->mr_map);
	free(uet_domain);
	return 0;
}

static struct fi_ops uet_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = uet_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr uet_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = ofi_mr_reg,
	.regv = ofi_mr_regv,
	.regattr = ofi_mr_regattr,
};

int uet_mr_verify(struct uet_domain *uet_domain, ssize_t len,
		  uintptr_t *io_addr, uint64_t key, uint64_t access)
{
	int ret;

	ofi_genlock_lock(&uet_domain->util_domain.lock);
	ret = ofi_mr_map_verify(&uet_domain->mr_map, io_addr, len,
				key, access, NULL);
	ofi_genlock_unlock(&uet_domain->util_domain.lock);
	return ret;
}

int uet_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct fi_info *dg_info;
	struct uet_domain *uet_domain;
	struct uet_fabric *uet_fabric;

	uet_fabric = container_of(fabric, struct uet_fabric,
				  util_fabric.fabric_fid);

	uet_domain = calloc(1, sizeof(*uet_domain));
	if (!uet_domain)
		return -FI_ENOMEM;

	ret = ofi_get_core_info(fabric->api_version, NULL, NULL,
				0, &uet_util_prov, info, NULL,
				uet_info_to_core, &dg_info);
	if (ret)
		goto err1;


	ret = fi_domain(uet_fabric->dg_fabric, dg_info,
			&uet_domain->dg_domain, context);
	if (ret)
		goto err2;

	uet_domain->max_mtu_sz = MIN(dg_info->ep_attr->max_msg_size, UET_MAX_MTU_SIZE);
	uet_domain->max_inline_msg = uet_domain->max_mtu_sz -
					sizeof(struct uet_base_hdr) -
					dg_info->ep_attr->msg_prefix_size;
	uet_domain->max_inline_rma = uet_domain->max_inline_msg -
					(sizeof(struct uet_rma_hdr) +
					(UET_IOV_LIMIT * sizeof(struct ofi_rma_iov)));
	uet_domain->max_inline_atom = uet_domain->max_inline_rma -
					sizeof(struct uet_atom_hdr);
	uet_domain->max_seg_sz = uet_domain->max_mtu_sz - sizeof(struct uet_data_pkt) -
				 dg_info->ep_attr->msg_prefix_size;

	ret = ofi_domain_init(fabric, info, &uet_domain->util_domain, context,
			      OFI_LOCK_MUTEX);
	if (ret) {
		goto err3;
	}

	ret = ofi_mr_map_init(&uet_prov, info->domain_attr->mr_mode,
			      &uet_domain->mr_map);
	if (ret)
		goto err4;

	*domain = &uet_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &uet_domain_fi_ops;
	(*domain)->ops = &uet_domain_ops;
	(*domain)->mr = &uet_mr_ops;
	fi_freeinfo(dg_info);
	return 0;
err4:
	if (ofi_domain_close(&uet_domain->util_domain))
		FI_WARN(&uet_prov, FI_LOG_DOMAIN,
			"ofi_domain_close failed");
err3:
	fi_close(&uet_domain->dg_domain->fid);
err2:
	fi_freeinfo(dg_info);
err1:
	free(uet_domain);
	return ret;
}
