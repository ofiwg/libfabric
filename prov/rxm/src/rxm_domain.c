/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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
#include <unistd.h>

#include <fi_util.h>
#include "rxm.h"

static struct fi_ops_domain rxm_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = ip_av_create,
	.cq_open = rxm_cq_open,
	.endpoint = rxm_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
};

static int rxm_domain_close(fid_t fid)
{
	struct rxm_domain *rxm_domain;
	int ret;

	rxm_domain = container_of(fid, struct rxm_domain, util_domain.domain_fid.fid);

	ret = fi_close(&rxm_domain->msg_domain->fid);
	if (ret)
		return ret;

	ret = ofi_domain_close(&rxm_domain->util_domain);
	if (ret)
		return ret;

	free(rxm_domain);
	return 0;
}

static struct fi_ops rxm_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int rxm_mr_close(fid_t fid)
{
	struct rxm_mr *rxm_mr;
	int ret;

	rxm_mr = container_of(fid, struct rxm_mr, mr_fid.fid);
	ret = fi_close(&rxm_mr->msg_mr->fid);
	if (ret)
		FI_WARN(&rxm_prov, FI_LOG_DOMAIN, "Unable to close MSG MR\n");
	free(rxm_mr);
	return ret;
}

static struct fi_ops rxm_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int rxm_mr_reg(struct fid *domain_fid, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct rxm_domain *rxm_domain;
	struct rxm_mr *rxm_mr;
	int ret;

	rxm_domain = container_of(domain_fid, struct rxm_domain,
			util_domain.domain_fid.fid);

	if (!(rxm_mr = calloc(1, sizeof(*rxm_mr))))
		return -FI_ENOMEM;

	ret = fi_mr_reg(rxm_domain->msg_domain, buf, len, access, offset, requested_key,
			flags, &rxm_mr->msg_mr, context);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_DOMAIN, "Unable to register MSG MR\n");
		goto err;
	}

	rxm_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	rxm_mr->mr_fid.fid.context = context;
	rxm_mr->mr_fid.fid.ops = &rxm_mr_ops;
	rxm_mr->mr_fid.mem_desc = fi_mr_desc(rxm_mr->msg_mr);
	rxm_mr->mr_fid.key = fi_mr_key(rxm_mr->msg_mr);
	*mr = &rxm_mr->mr_fid;

	return 0;
err:
	free(rxm_mr);
	return ret;
}

static struct fi_ops_mr rxm_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = rxm_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

int rxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct rxm_domain *rxm_domain;
	struct rxm_fabric *rxm_fabric;
	struct fi_info *msg_info;

	rxm_domain = calloc(1, sizeof(*rxm_domain));
	if (!rxm_domain)
		return -FI_ENOMEM;

	rxm_fabric = container_of(fabric, struct rxm_fabric, util_fabric.fabric_fid);

	ret = ofix_getinfo(rxm_prov.version, NULL, NULL, 0, &rxm_util_prov,
			info, rxm_alter_layer_info, rxm_alter_base_info,
			1, &msg_info);
	if (ret)
		goto err1;

	ret = fi_domain(rxm_fabric->msg_fabric, msg_info,
			&rxm_domain->msg_domain, context);
	if (ret)
		goto err2;

	ret = ofi_domain_init(fabric, info, &rxm_domain->util_domain, context);
	if (ret) {
		goto err3;
	}

	*domain = &rxm_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &rxm_domain_fi_ops;
	/* Replace MR ops set by ofi_domain_init() */
	(*domain)->mr = &rxm_domain_mr_ops;
	(*domain)->ops = &rxm_domain_ops;

	return 0;
err3:
	fi_close(&rxm_domain->msg_domain->fid);
err2:
	fi_freeinfo(msg_info);
err1:
	free(rxm_domain);
	return ret;
}
