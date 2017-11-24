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

#include <fi_util.h>
#include "fi_verbs.h"

static int fi_ibv_mr_close(fid_t fid)
{
	struct fi_ibv_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct fi_ibv_mem_desc, mr_fid.fid);
	ret = -ibv_dereg_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops fi_ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int
fi_ibv_mr_reg(struct fid *fid, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_ibv_mem_desc *md;
	int fi_ibv_access = 0;
	struct fid_domain *domain;

	if (flags)
		return -FI_EBADFLAGS;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		return -FI_EINVAL;
	}
	domain = container_of(fid, struct fid_domain, fid);

	md = calloc(1, sizeof *md);
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain, struct fi_ibv_domain,
				  util_domain.domain_fid);
	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &fi_ibv_mr_ops;

	/* Enable local write access by default for FI_EP_RDM which hides local
	 * registration requirements. This allows to avoid buffering or double
	 * registration */
	if (!(md->domain->info->caps & FI_LOCAL_MR) &&
	    !(md->domain->info->domain_attr->mr_mode & FI_MR_LOCAL))
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	/* Local read access to an MR is enabled by default in verbs */
	if (access & FI_RECV)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	/* iWARP spec requires Remote Write access for an MR that is used
	 * as a data sink for a Remote Read */
	if (access & FI_READ) {
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;
		if (md->domain->verbs->device->transport_type == IBV_TRANSPORT_IWARP)
			fi_ibv_access |= IBV_ACCESS_REMOTE_WRITE;
	}

	if (access & FI_WRITE)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	if (access & FI_REMOTE_READ)
		fi_ibv_access |= IBV_ACCESS_REMOTE_READ;

	/* Verbs requires Local Write access too for Remote Write access */
	if (access & FI_REMOTE_WRITE)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE |
			IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

	md->mr = ibv_reg_mr(md->domain->pd, (void *) buf, len, fi_ibv_access);
	if (!md->mr)
		goto err;

	md->mr_fid.mem_desc = (void *) (uintptr_t) md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr = &md->mr_fid;
	if (md->domain->eq_flags & FI_REG_MR) {
		struct fi_eq_entry entry = {
			.fid = &md->mr_fid.fid,
			.context = context
		};
		if (md->domain->eq)
			fi_ibv_eq_write_event(md->domain->eq, FI_MR_COMPLETE,
				 	      &entry, sizeof(entry));
		else if (md->domain->util_domain.eq)
			 /* This branch is taken for the verbs/DGRAM */
			fi_eq_write(&md->domain->util_domain.eq->eq_fid,
				    FI_MR_COMPLETE, &entry, sizeof(entry), 0);
	}
	return 0;

err:
	free(md);
	return -errno;
}

static int fi_ibv_mr_regv(struct fid *fid, const struct iovec * iov,
		size_t count, uint64_t access, uint64_t offset, uint64_t requested_key,
		uint64_t flags, struct fid_mr **mr, void *context)
{
	if (count > VERBS_MR_IOV_LIMIT) {
		VERBS_WARN(FI_LOG_FABRIC,
			   "iov count > %d not supported\n",
			   VERBS_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}
	return fi_ibv_mr_reg(fid, iov->iov_base, iov->iov_len, access, offset,
			requested_key, flags, mr, context);
}

static int fi_ibv_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		uint64_t flags, struct fid_mr **mr)
{
	return fi_ibv_mr_regv(fid, attr->mr_iov, attr->iov_count, attr->access,
			0, attr->requested_key, flags, mr, attr->context);
}

struct fi_ops_mr fi_ibv_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_ibv_mr_reg,
	.regv = fi_ibv_mr_regv,
	.regattr = fi_ibv_mr_regattr,
};
