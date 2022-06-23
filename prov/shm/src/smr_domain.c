/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#include "smr.h"

static struct fi_ops_domain smr_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = smr_av_open,
	.cq_open = smr_cq_open,
	.endpoint = smr_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = smr_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = smr_query_atomic,
	.query_collective = fi_no_query_collective,
};

static int smr_domain_close(fid_t fid)
{
	int ret;
	struct smr_domain *domain;

	domain = container_of(fid, struct smr_domain, util_domain.domain_fid.fid);
	ret = ofi_domain_close(&domain->util_domain);
	if (ret)
		return ret;

	free(domain);
	return 0;
}

struct smr_mr {
	struct ofi_mr *mr;
	enum fi_hmem_iface	iface;
	struct fi_rma_iov *iov;
	int iov_count;
};

static int smr_mr_close(struct fid *fid)
{
	int i;
	struct smr_mr *smr_mr;
	struct ofi_mr *mr;

	mr = container_of(fid, struct ofi_mr, mr_fid.fid);
	smr_mr = container_of(mr, struct smr_mr, mr);

	for (i = 0; i < smr_mr->iov_count; i++)
		ofi_hmem_host_unregister_iface(smr_mr->iface,
					(void *) smr_mr->iov[i].addr);

	free(smr_mr->iov);
	free(smr_mr);

	return ofi_mr_close(fid);
}

static int smr_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
	uint64_t flags, struct fid_mr **mr_fid)
{
	int ret = 0, i, j;
	uint64_t key;
	struct smr_mr *smr_mr;

	if (attr->iface < 0)
		goto out;

	smr_mr = calloc(sizeof(*smr_mr), 1);
	if (!smr_mr)
		return -FI_ENOMEM;

	smr_mr->iov = calloc(sizeof(*smr_mr->iov), attr->iov_count);
	if (!smr_mr->iov) {
		free(smr_mr);
		return -FI_ENOMEM;
	}

	ret = ofi_mr_regattr(fid, attr, flags, mr_fid);
	if (ret)
		goto free_smr;

	smr_mr->mr = container_of(fid, struct ofi_mr, mr_fid.fid);
	smr_mr->iface = attr->iface;
	smr_mr->iov_count = attr->iov_count;

	for (i = 0; i < attr->iov_count; i++) {
		ret = ofi_hmem_host_register_iface(attr->iface,
						attr->mr_iov[i].iov_base,
						attr->mr_iov[i].iov_len, &key);
		if (ret)
			goto fail;

		smr_mr->iov[i].addr = (uint64_t) attr->mr_iov[i].iov_base;
		smr_mr->iov[i].len = attr->mr_iov[i].iov_len;
		smr_mr->iov[i].key = key;
	}

	/* TODO: assume that all the keys are the same? 
	 * AFAIU, mr_fid.key will be returned as part of fi_mr_key() call.
	 * This is called by the application and the key returned is passed to
	 * the rma operations */
	smr_mr->mr->mr_fid.key = smr_mr->iov[0].key;
	smr_mr->mr->mr_fid.fid.ops->close = smr_mr_close;

	goto out;

fail:
	for (j = i; j >= 0; j--)
		ofi_hmem_host_unregister_iface(attr->iface,
					attr->mr_iov[i].iov_base);
free_smr:
	free(smr_mr->iov);
	free(smr_mr);
out:
	return ret;
}

static struct fi_ops smr_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = smr_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_mr smr_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = ofi_mr_reg,
	.regv = ofi_mr_regv,
	.regattr = smr_mr_regattr,
};

int smr_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct smr_domain *smr_domain;
	struct smr_fabric *smr_fabric;

	ret = ofi_prov_check_info(&smr_util_prov, fabric->api_version, info);
	if (ret)
		return ret;

	smr_domain = calloc(1, sizeof(*smr_domain));
	if (!smr_domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(fabric, info, &smr_domain->util_domain, context,
			      OFI_DOMAIN_SPINLOCK);
	if (ret) {
		free(smr_domain);
		return ret;
	}

	smr_domain->util_domain.threading = FI_THREAD_SAFE;
	smr_fabric = container_of(fabric, struct smr_fabric, util_fabric.fabric_fid);
	ofi_mutex_lock(&smr_fabric->util_fabric.lock);
	smr_domain->fast_rma = smr_fast_rma_enabled(info->domain_attr->mr_mode,
						    info->tx_attr->msg_order);
	ofi_mutex_unlock(&smr_fabric->util_fabric.lock);

	*domain = &smr_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &smr_domain_fi_ops;
	(*domain)->ops = &smr_domain_ops;
	(*domain)->mr = &smr_mr_ops;

	return 0;
}
