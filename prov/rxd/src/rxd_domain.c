/*
 * Copyright (c) 2016-2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2026 ETH Zurich. All rights reserved.
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

#include "rxd.h"


static struct fi_ops_domain rxd_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = rxd_av_create,
	.cq_open = rxd_cq_open,
	.endpoint = rxd_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = rxd_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = rxd_query_atomic,
	.query_collective = fi_no_query_collective,
};

static int rxd_domain_close(fid_t fid)
{
	int ret;
	struct rxd_domain *rxd_domain;

	rxd_domain = container_of(fid, struct rxd_domain, util_domain.domain_fid.fid);

	ret = fi_close(&rxd_domain->dg_domain->fid);
	if (ret)
		return ret;

	ret = ofi_domain_close(&rxd_domain->util_domain);
	if (ret)
		return ret;

	free(rxd_domain);
	return 0;
}

static struct fi_ops rxd_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void rxd_mr_remove_map_entry(struct rxd_mr *mr)
{
	ofi_genlock_lock(&mr->domain->util_domain.lock);
	(void) ofi_mr_map_remove(&mr->domain->util_domain.mr_map,
				 mr->mr_fid.key);
	ofi_genlock_unlock(&mr->domain->util_domain.lock);
}

static int rxd_mr_add_map_entry(struct util_domain *domain,
				struct fi_mr_attr *dg_attr,
				struct rxd_mr *rxd_mr,
				uint64_t flags)
{
	uint64_t temp_key;
	int ret;

	dg_attr->requested_key = rxd_mr->mr_fid.key;

	ofi_genlock_lock(&domain->lock);
	ret = ofi_mr_map_insert(&domain->mr_map, dg_attr, &temp_key, rxd_mr, flags);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&rxd_prov, FI_LOG_DOMAIN,
			"MR map insert for atomic verification failed %d\n",
			ret);
	} else {
		assert(rxd_mr->mr_fid.key == temp_key);
	}
	ofi_genlock_unlock(&domain->lock);

	return ret;
}

static int rxd_mr_close(fid_t fid)
{
	struct rxd_mr *rxd_mr;
	int ret;

	rxd_mr = container_of(fid, struct rxd_mr, mr_fid.fid);

	rxd_mr_remove_map_entry(rxd_mr);

	ret = fi_close(&rxd_mr->dg_mr->fid);
	if (ret)
		FI_WARN(&rxd_prov, FI_LOG_DOMAIN, "Unable to close MSG MR\n");

	ofi_atomic_dec32(&rxd_mr->domain->util_domain.ref);
	free(rxd_mr);
	return ret;
}

static struct fi_ops rxd_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void rxd_mr_init(struct rxd_mr *rxd_mr, struct rxd_domain *domain,
			void *context)
{
	rxd_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	rxd_mr->mr_fid.fid.context = context;
	rxd_mr->mr_fid.fid.ops = &rxd_mr_fi_ops;
	rxd_mr->mr_fid.mem_desc = rxd_mr;
	rxd_mr->mr_fid.key = fi_mr_key(rxd_mr->dg_mr);
	rxd_mr->domain = domain;
	ofi_atomic_inc32(&domain->util_domain.ref);
}

static int rxd_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			  uint64_t flags, struct fid_mr **mr)
{
	struct rxd_domain *rxd_domain;
	struct fi_mr_attr dg_attr = *attr;
	struct rxd_mr *rxd_mr;
	int ret;

	rxd_domain = container_of(fid, struct rxd_domain,
				  util_domain.domain_fid.fid);

	rxd_mr = calloc(1, sizeof(*rxd_mr));
	if (!rxd_mr)
		return -FI_ENOMEM;

	ofi_mr_update_attr(
		rxd_domain->util_domain.fabric->fabric_fid.api_version,
		rxd_domain->util_domain.info_domain_caps, attr, &dg_attr,
		flags);

	ret = fi_mr_regattr(rxd_domain->dg_domain, &dg_attr, flags,
			    &rxd_mr->dg_mr);
	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_DOMAIN,
			"Unable to register MSG MR\n");
		goto err;
	}
	rxd_mr_init(rxd_mr, rxd_domain, attr->context);
	*mr = &rxd_mr->mr_fid;

	ret = rxd_mr_add_map_entry(&rxd_domain->util_domain, &dg_attr, rxd_mr,
				   flags);
	if (ret)
		goto map_err;

	FI_INFO(&rxd_prov, FI_LOG_DOMAIN, "mr_regattr\n");
	return 0;

map_err:
	fi_close(&rxd_mr->mr_fid.fid);
	return ret;
err:
	free(rxd_mr);
	return ret;
}

static int rxd_mr_regv(struct fid *fid, const struct iovec *iov, size_t count,
		       uint64_t access, uint64_t offset, uint64_t requested_key,
		       uint64_t flags, struct fid_mr **mr, void *context)
{
	struct rxd_domain *rxd_domain;
	struct rxd_mr *rxd_mr;
	int ret;
	struct fi_mr_attr dg_attr = {
		.mr_iov = iov,
		.iov_count = count,
		.access = access,
		.offset = offset,
		.requested_key = requested_key,
		.context = context,
	};

	rxd_domain = container_of(fid, struct rxd_domain,
				  util_domain.domain_fid.fid);

	rxd_mr = calloc(1, sizeof(*rxd_mr));
	if (!rxd_mr)
		return -FI_ENOMEM;

	ret = fi_mr_regv(rxd_domain->dg_domain, iov, count, access, offset,
			 requested_key, flags, &rxd_mr->dg_mr, context);
	if (ret) {
		FI_WARN(&rxd_prov, FI_LOG_DOMAIN,
			"Unable to register MSG MR\n");
		goto err;
	}
	rxd_mr_init(rxd_mr, rxd_domain, context);
	ofi_atomic_inc32(&rxd_domain->util_domain.ref);
	*mr = &rxd_mr->mr_fid;

	ret = rxd_mr_add_map_entry(&rxd_domain->util_domain, &dg_attr, rxd_mr,
				   flags);
	if (ret)
		goto map_err;

	return 0;
map_err:
	fi_close(&rxd_mr->mr_fid.fid);
	return ret;
err:
	free(rxd_mr);
	return ret;
}

static int rxd_mr_reg(struct fid *fid, const void *buf, size_t len,
		      uint64_t access, uint64_t offset, uint64_t requested_key,
		      uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	return rxd_mr_regv(fid, &iov, 1, access, offset, requested_key, flags,
			   mr, context);
}

static struct fi_ops_mr rxd_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = rxd_mr_reg,
	.regv = rxd_mr_regv,
	.regattr = rxd_mr_regattr,
};

int rxd_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct fi_info *dg_info;
	struct rxd_domain *rxd_domain;
	struct rxd_fabric *rxd_fabric;

	rxd_fabric = container_of(fabric, struct rxd_fabric,
				  util_fabric.fabric_fid);

	rxd_domain = calloc(1, sizeof(*rxd_domain));
	if (!rxd_domain)
		return -FI_ENOMEM;

	ret = ofi_get_core_info(fabric->api_version, NULL, NULL,
				0, &rxd_util_prov, info, NULL,
				rxd_info_to_core, &dg_info);
	if (ret)
		goto err1;


	ret = fi_domain(rxd_fabric->dg_fabric, dg_info,
			&rxd_domain->dg_domain, context);
	if (ret)
		goto err2;

	rxd_domain->max_mtu_sz = MIN(dg_info->ep_attr->max_msg_size, RXD_MAX_MTU_SIZE);
	rxd_domain->max_inline_msg = rxd_domain->max_mtu_sz -
					sizeof(struct rxd_base_hdr) -
					dg_info->ep_attr->msg_prefix_size;
	rxd_domain->max_inline_rma = rxd_domain->max_inline_msg -
					(sizeof(struct rxd_rma_hdr) +
					(RXD_IOV_LIMIT * sizeof(struct ofi_rma_iov)));
	rxd_domain->max_inline_atom = rxd_domain->max_inline_rma -
					sizeof(struct rxd_atom_hdr);
	rxd_domain->max_seg_sz = rxd_domain->max_mtu_sz - sizeof(struct rxd_data_pkt) -
				 dg_info->ep_attr->msg_prefix_size;

	ret = ofi_domain_init(fabric, info, &rxd_domain->util_domain, context,
			      OFI_LOCK_MUTEX);
	if (ret) {
		goto err3;
	}

	*domain = &rxd_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &rxd_domain_fi_ops;
	(*domain)->ops = &rxd_domain_ops;
	(*domain)->mr = &rxd_mr_ops;
	fi_freeinfo(dg_info);
	return 0;
err3:
	fi_close(&rxd_domain->dg_domain->fid);
err2:
	fi_freeinfo(dg_info);
err1:
	free(rxd_domain);
	return ret;
}
