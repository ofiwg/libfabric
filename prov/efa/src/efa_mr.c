/*
 * Copyright (c) 2017-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"
#include <ofi_util.h>
#include "efa.h"
#include "efa_verbs.h"

static int efa_mr_cache_close(fid_t fid)
{
	struct efa_mem_desc *mr = container_of(fid, struct efa_mem_desc,
					       mr_fid.fid);

	ofi_mr_cache_delete(&mr->domain->cache, mr->entry);

	return 0;
}

static struct fi_ops efa_mr_cache_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_mr_cache_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int efa_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			   struct ofi_mr_entry *entry)
{
	int fi_ibv_access = IBV_ACCESS_LOCAL_WRITE;

	struct efa_mem_desc *md = (struct efa_mem_desc *)entry->data;

	md->domain = container_of(cache->domain, struct efa_domain,
				  util_domain);
	md->mr_fid.fid.ops = &efa_mr_cache_ops;

	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = NULL;

	md->mr = efa_cmd_reg_mr(md->domain->pd, entry->iov.iov_base,
				entry->iov.iov_len, fi_ibv_access);
	if (!md->mr) {
		EFA_WARN_ERRNO(FI_LOG_MR, "efa_cmd_reg_mr", errno);
		return -errno;
	}

	md->mr_fid.mem_desc = (void *)(uintptr_t)md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;

	return 0;
}

void efa_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
			      struct ofi_mr_entry *entry)
{
	struct efa_mem_desc *md = (struct efa_mem_desc *)entry->data;
	int ret = -efa_cmd_dereg_mr(md->mr);
	if (ret)
		EFA_WARN(FI_LOG_MR, "Unable to dereg mr: %d\n", ret);
}

static int efa_mr_cache_reg(struct fid *fid, const void *buf, size_t len,
			    uint64_t access, uint64_t offset,
			    uint64_t requested_key, uint64_t flags,
			    struct fid_mr **mr_fid, void *context)
{
	struct efa_domain *domain;
	struct efa_mem_desc *md;
	struct ofi_mr_entry *entry;
	int ret;

	struct iovec iov = {
		.iov_base	= (void *)buf,
		.iov_len	= len,
	};

	struct fi_mr_attr attr = {
		.mr_iov		= &iov,
		.iov_count	= 1,
		.access		= access,
		.offset		= offset,
		.requested_key	= requested_key,
		.context	= context,
	};

	if (access & ~EFA_MR_SUPPORTED_PERMISSIONS) {
		EFA_WARN(FI_LOG_MR,
			 "Unsupported access permissions. requested[0x%" PRIx64 "] supported[0x%" PRIx64 "]\n",
			 access, (uint64_t)EFA_MR_SUPPORTED_PERMISSIONS);
		return -FI_EINVAL;
	}

	domain = container_of(fid, struct efa_domain,
			      util_domain.domain_fid.fid);

	ret = ofi_mr_cache_search(&domain->cache, &attr, &entry);
	if (OFI_UNLIKELY(ret))
		return ret;

	md = (struct efa_mem_desc *)entry->data;
	md->entry = entry;

	*mr_fid = &md->mr_fid;
	return 0;
}

static int efa_mr_cache_regv(struct fid *fid, const struct iovec *iov,
			     size_t count, uint64_t access, uint64_t offset,
			     uint64_t requested_key, uint64_t flags,
			     struct fid_mr **mr_fid, void *context)
{
	if (count > EFA_MR_IOV_LIMIT) {
		EFA_WARN(FI_LOG_MR, "iov count > %d not supported\n",
			 EFA_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}
	return efa_mr_cache_reg(fid, iov->iov_base, iov->iov_len, access,
				offset, requested_key, flags, mr_fid, context);
}

static int efa_mr_cache_regattr(struct fid *fid, const struct fi_mr_attr *attr,
				uint64_t flags, struct fid_mr **mr_fid)
{
	return efa_mr_cache_regv(fid, attr->mr_iov, attr->iov_count,
				 attr->access, attr->offset,
				 attr->requested_key, flags, mr_fid,
				 attr->context);
}

struct fi_ops_mr efa_domain_mr_cache_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = efa_mr_cache_reg,
	.regv = efa_mr_cache_regv,
	.regattr = efa_mr_cache_regattr,
};

static int efa_mr_close(fid_t fid)
{
	struct efa_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct efa_mem_desc, mr_fid.fid);
	ret = -efa_cmd_dereg_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops efa_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int efa_mr_reg(struct fid *fid, const void *buf, size_t len,
		      uint64_t access, uint64_t offset, uint64_t requested_key,
		      uint64_t flags, struct fid_mr **mr_fid, void *context)
{
	struct fid_domain *domain_fid;
	struct efa_mem_desc *md;
	int fi_ibv_access = 0;

	if (flags)
		return -FI_EBADFLAGS;

	if (fid->fclass != FI_CLASS_DOMAIN)
		return -FI_EINVAL;

	if (access & ~EFA_MR_SUPPORTED_PERMISSIONS) {
		EFA_WARN(FI_LOG_MR,
			 "Unsupported access permissions. requested[0x%" PRIx64 "] supported[0x%" PRIx64 "]\n",
			 access, (uint64_t)EFA_MR_SUPPORTED_PERMISSIONS);
		return -FI_EINVAL;
	}

	domain_fid = container_of(fid, struct fid_domain, fid);

	md = calloc(1, sizeof(*md));
	if (!md)
		return -FI_ENOMEM;

	md->domain = container_of(domain_fid, struct efa_domain,
				  util_domain.domain_fid);
	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = context;
	md->mr_fid.fid.ops = &efa_mr_ops;

	/* Local read access to an MR is enabled by default in verbs */
	if (access & FI_RECV)
		fi_ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	md->mr = efa_cmd_reg_mr(md->domain->pd, (void *)buf, len, fi_ibv_access);
	if (!md->mr) {
		EFA_WARN_ERRNO(FI_LOG_MR, "efa_cmd_reg_mr", errno);
		goto err;
	}

	md->mr_fid.mem_desc = (void *)(uintptr_t)md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;
	*mr_fid = &md->mr_fid;

	return 0;

err:
	free(md);
	return -errno;
}

static int efa_mr_regv(struct fid *fid, const struct iovec *iov,
		       size_t count, uint64_t access, uint64_t offset, uint64_t requested_key,
		       uint64_t flags, struct fid_mr **mr_fid, void *context)
{
	if (count > EFA_MR_IOV_LIMIT) {
		EFA_WARN(FI_LOG_MR, "iov count > %d not supported\n",
			 EFA_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}
	return efa_mr_reg(fid, iov->iov_base, iov->iov_len, access, offset,
			  requested_key, flags, mr_fid, context);
}

static int efa_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			  uint64_t flags, struct fid_mr **mr_fid)
{
	return efa_mr_regv(fid, attr->mr_iov, attr->iov_count, attr->access,
			   attr->offset, attr->requested_key, flags, mr_fid,
			   attr->context);
}

struct fi_ops_mr efa_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = efa_mr_reg,
	.regv = efa_mr_regv,
	.regattr = efa_mr_regattr,
};
