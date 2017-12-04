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

	ofi_mr_map_close(&rxd_domain->mr_map);
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

struct rxd_mr_entry {
	struct fid_mr mr_fid;
	struct rxd_domain *domain;
	uint64_t key;
	uint64_t flags;
};

static int rxd_mr_close(struct fid *fid)
{
	struct rxd_domain *dom;
	struct rxd_mr_entry *mr;
	int err = 0;

	mr = container_of(fid, struct rxd_mr_entry, mr_fid.fid);
	dom = mr->domain;

	fastlock_acquire(&dom->util_domain.lock);
	err = ofi_mr_map_remove(&dom->mr_map, mr->key);
	fastlock_release(&dom->util_domain.lock);
	if (err)
		return err;

	ofi_atomic_dec32(&dom->util_domain.ref);
	free(mr);
	return 0;
}

static struct fi_ops rxd_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_mr_close,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


static int rxd_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		uint64_t flags, struct fid_mr **mr)
{
	struct rxd_domain *dom;
	struct rxd_mr_entry *_mr;
	uint64_t key;
	int ret = 0;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0) {
		return -FI_EINVAL;
	}

	dom = container_of(fid, struct rxd_domain, util_domain.domain_fid.fid);
	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	fastlock_acquire(&dom->util_domain.lock);

	_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	_mr->mr_fid.fid.context = attr->context;
	_mr->mr_fid.fid.ops = &rxd_mr_fi_ops;

	_mr->domain = dom;
	_mr->flags = flags;

	ret = ofi_mr_map_insert(&dom->mr_map, attr, &key, _mr);
	if (ret != 0) {
		goto err;
	}

	_mr->mr_fid.key = _mr->key = key;
	_mr->mr_fid.mem_desc = (void *) (uintptr_t) key;
	fastlock_release(&dom->util_domain.lock);

	*mr = &_mr->mr_fid;
	ofi_atomic_inc32(&dom->util_domain.ref);

	return 0;
err:
	fastlock_release(&dom->util_domain.lock);
	free(_mr);
	return ret;
}

static int rxd_mr_regv(struct fid *fid, const struct iovec *iov,
			size_t count, uint64_t access,
			uint64_t offset, uint64_t requested_key,
			uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;
	return rxd_mr_regattr(fid, &attr, flags, mr);
}

static int rxd_mr_reg(struct fid *fid, const void *buf, size_t len,
		       uint64_t access, uint64_t offset, uint64_t requested_key,
		       uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	return rxd_mr_regv(fid, &iov, 1, access,  offset, requested_key,
			    flags, mr, context);
}

static struct fi_ops_mr rxd_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = rxd_mr_reg,
	.regv = rxd_mr_regv,
	.regattr = rxd_mr_regattr,
};

int rxd_mr_verify(struct rxd_domain *rxd_domain, ssize_t len,
		  uintptr_t *io_addr, uint64_t key, uint64_t access)
{
	int ret;

	fastlock_acquire(&rxd_domain->util_domain.lock);
	ret = ofi_mr_map_verify(&rxd_domain->mr_map, io_addr, len,
				key, access, NULL);
	fastlock_release(&rxd_domain->util_domain.lock);
	return ret;
}

int rxd_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct fi_info *dg_info;
	struct rxd_domain *rxd_domain;
	struct rxd_fabric *rxd_fabric;

	rxd_fabric = container_of(fabric, struct rxd_fabric,
				  util_fabric.fabric_fid);
	ret = ofi_prov_check_info(&rxd_util_prov, fabric->api_version, info);
	if (ret)
		return ret;

	rxd_domain = calloc(1, sizeof(*rxd_domain));
	if (!rxd_domain)
		return -FI_ENOMEM;

	ret = ofi_get_core_info(fabric->api_version, NULL, NULL,
				0, &rxd_util_prov, info,
				rxd_info_to_core, &dg_info);
	if (ret)
		goto err1;


	ret = fi_domain(rxd_fabric->dg_fabric, dg_info,
			&rxd_domain->dg_domain, context);
	if (ret)
		goto err2;

	rxd_domain->max_mtu_sz = dg_info->ep_attr->max_msg_size;
	rxd_domain->mr_mode = dg_info->domain_attr->mr_mode;

	ret = ofi_domain_init(fabric, info, &rxd_domain->util_domain, context);
	if (ret) {
		goto err3;
	}

	ret = ofi_mr_map_init(&rxd_prov, info->domain_attr->mr_mode,
			      &rxd_domain->mr_map);
	if (ret)
		goto err4;

	*domain = &rxd_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &rxd_domain_fi_ops;
	(*domain)->ops = &rxd_domain_ops;
	(*domain)->mr = &rxd_mr_ops;
	fi_freeinfo(dg_info);
	return 0;
err4:
	if (ofi_domain_close(&rxd_domain->util_domain))
		FI_WARN(&rxd_prov, FI_LOG_DOMAIN,
			"ofi_domain_close failed");
err3:
	fi_close(&rxd_domain->dg_domain->fid);
err2:
	fi_freeinfo(dg_info);
err1:
	free(rxd_domain);
	return ret;
}
