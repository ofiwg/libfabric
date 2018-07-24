/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <ofi_util.h>

#include "cxi_prov.h"

#define CXIX_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIX_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_MR, __VA_ARGS__)

static int cxix_mr_close(struct fid *fid)
{
	struct cxi_domain *dom;
	struct cxix_mr *mr;

	mr = container_of(fid, struct cxix_mr, mr_fid.fid);
	dom = mr->domain;

	fastlock_acquire(&dom->lock);
	/* TODO Destroy window */

	fastlock_release(&dom->lock);
	ofi_atomic_dec32(&dom->ref);
	free(mr);

	return 0;
}

static int cxix_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_cntr *cntr;
	struct cxi_cq *cq;
	struct cxix_mr *mr;

	mr = container_of(fid, struct cxix_mr, mr_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct cxi_cq, cq_fid.fid);
		if (mr->domain != cq->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cq = cq;
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxi_cntr, cntr_fid.fid);
		if (mr->domain != cntr->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cntr = cntr;
		break;

	default:
		return -FI_EINVAL;
	}
	return 0;
}

static struct fi_ops cxix_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxix_mr_close,
	.bind = cxix_mr_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int cxix_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	//struct fi_eq_entry eq_entry;
	struct cxi_domain *dom;
	struct cxix_mr *_mr;
	int ret = 0;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	dom = container_of(fid, struct cxi_domain, dom_fid.fid);

	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	fastlock_acquire(&dom->lock);

	_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	_mr->mr_fid.fid.context = attr->context;
	_mr->mr_fid.fid.ops = &cxix_mr_fi_ops;

	_mr->domain = dom;
	_mr->flags = flags;

	/* TODO Create window */

	_mr->mr_fid.key = _mr->key = attr->requested_key;
	_mr->mr_fid.mem_desc = (void *)_mr;

	ofi_atomic_inc32(&dom->ref);
	fastlock_release(&dom->lock);

/* TODO EQs */
#if 0
	if (dom->mr_eq) {
		eq_entry.fid = &dom->dom_fid;
		eq_entry.context = attr->context;

		return cxi_eq_report_event(dom->mr_eq, FI_MR_COMPLETE,
					   &eq_entry, sizeof(eq_entry), 0);
	}
#endif

	*mr = &_mr->mr_fid;

	return 0;

free_mr:
	fastlock_release(&dom->lock);
	free(_mr);

	return ret;
}

static int cxix_regv(struct fid *fid, const struct iovec *iov,
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

	return cxix_regattr(fid, &attr, flags, mr);
}

static int cxix_reg(struct fid *fid, const void *buf, size_t len,
		    uint64_t access, uint64_t offset, uint64_t requested_key,
		    uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	return cxix_regv(fid, &iov, 1, access, offset, requested_key,
			 flags, mr, context);
}

struct fi_ops_mr cxix_dom_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = cxix_reg,
	.regv = cxix_regv,
	.regattr = cxix_regattr,
};

