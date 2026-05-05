/*
 * Copyright (c) 2025 ORNL. All rights reserved.
 * Copyright (c) Intel Corporation. All rights reserved.
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

#include "lnx.h"

/*
 * 1. On MR registration we store the information passed in the fi_mr_attr
 * 2. If we get a bind, we flag it and store the flags and fid pointer
 * 3. If we get a control command we flag it and store the command
 * 4. When a data operation we're passed the descriptor, using that we can
 * find out the information we stored, and since we already know the
 * target core provider we can do memory registration at that point
 */

int lnx_mr_regattr_core(struct lnx_core_domain *cd, void **desc, size_t count,
			void **core_desc)
{
	int rc, i;
	struct lnx_mr **lm;

	lm = (struct lnx_mr **)desc;
	if (!lm)
		return FI_SUCCESS;

	for (i = 0; i < count; i++) {
		if (lm[i] && !lm[i]->lm_core_mr[cd->idx]) {
			rc = fi_mr_regattr(cd->cd_domain, &lm[i]->lm_attr,
					   lm[i]->lm_mr.flags,
					   &lm[i]->lm_core_mr[cd->idx]);
			if (rc)
				return rc;
			core_desc[i] = fi_mr_desc(lm[i]->lm_core_mr[cd->idx]);
		} else {
			core_desc[i] = NULL;
		}
	}

	return FI_SUCCESS;
}

static int lnx_mr_close(struct fid *fid)
{
	int rc, frc = FI_SUCCESS;
	struct lnx_domain *domain;
	struct lnx_mr *lm;
	int i;

	lm = container_of(fid, struct lnx_mr, lm_mr.mr_fid.fid);
	domain = container_of(lm->lm_mr.domain, struct lnx_domain, ld_domain);

	for (i = 0; i < domain->ld_num_doms; i++) {
		if (lm->lm_core_mr[i]) {
			rc = fi_close(&lm->lm_core_mr[i]->fid);
			if (rc)
				frc = rc;
		}
	}
	free(lm->lm_core_mr);
	ofi_atomic_dec32(&domain->ld_domain.ref);
	ofi_buf_free(lm);

	return frc;
}

static int lnx_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return -FI_ENOSYS;
}

static int lnx_mr_control(struct fid *fid, int command, void *arg)
{
	struct lnx_mr *lm;
	struct fi_mr_raw_attr *raw_attr = arg;

	if (command != FI_GET_RAW_MR)
		return -FI_ENOSYS;

	lm = container_of(fid, struct lnx_mr, lm_mr.mr_fid.fid);
	if (*raw_attr->key_size < lm->key_size) {
		FI_WARN(&lnx_prov, FI_LOG_MR,
			"Raw key buffer is too small: input %lu, needed %lu\n",
			*raw_attr->key_size, lm->key_size);
		*raw_attr->key_size = lm->key_size;
		return -FI_ETOOSMALL;
	}

	memcpy(raw_attr->raw_key, lm->raw_key, lm->key_size);
	*raw_attr->key_size = lm->key_size;

	return FI_SUCCESS;
}

static struct fi_ops lnx_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = lnx_mr_close,
	.bind = lnx_mr_bind,
	.control = lnx_mr_control,
	.ops_open = fi_no_ops_open
};

int lnx_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		   uint64_t flags, struct fid_mr **mr_fid)
{
	struct lnx_core_domain *cd;
	struct lnx_domain *domain;
	struct ofi_mr *mr;
	struct lnx_mr *lm = NULL;
	uint64_t key;
	int rc, i;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr ||
	    attr->iov_count <= 0 || attr->iov_count > LNX_IOV_LIMIT)
		return -FI_EINVAL;

	domain = container_of(fid, struct lnx_domain, ld_domain.domain_fid.fid);

	lm = ofi_buf_alloc(domain->ld_mem_reg_bp);
	if (!lm)
		return -FI_ENOMEM;

	memset(lm, 0, sizeof(*lm));

	lm->lm_attr = *attr;
	memcpy(lm->lm_iov, attr->mr_iov,
	       sizeof(*attr->mr_iov) * attr->iov_count);
	lm->lm_attr.mr_iov = lm->lm_iov;
	lm->lm_core_mr = calloc(sizeof(struct fid_mr *), domain->ld_num_doms);
	if (!lm->lm_core_mr) {
		free(lm);
		return -FI_ENOMEM;
	}
	mr = &lm->lm_mr;
	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.ops = &lnx_mr_fi_ops;
	mr->mr_fid.mem_desc = lm;
	mr->domain = &domain->ld_domain;
	mr->flags = flags;

	if (attr->access & (FI_REMOTE_WRITE | FI_REMOTE_READ)) {
		lm->key_size = sizeof(uint64_t) * domain->ld_num_doms;
		lm->raw_key = malloc(lm->key_size);

		for (i = 0; i < domain->ld_num_doms; i++) {
			cd = &domain->ld_core_domains[i];

			rc = fi_mr_regattr(cd->cd_domain, &lm->lm_attr,
					   flags, &lm->lm_core_mr[i]);
			if (rc)
				return rc;
			key = fi_mr_key(lm->lm_core_mr[i]);
			memcpy((char *)lm->raw_key + sizeof(key) * i, &key,
				sizeof(key));
		}
	} else {
		lm->key_size = sizeof(uint64_t);
	}

	*mr_fid = &mr->mr_fid;
	ofi_atomic_inc32(&domain->ld_domain.ref);

	return FI_SUCCESS;
}

int lnx_mr_regv(struct fid *fid, const struct iovec *iov, size_t count,
		uint64_t access, uint64_t offset, uint64_t requested_key,
		uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;
	attr.iface = FI_HMEM_SYSTEM;
	attr.device.reserved = 0;
	attr.hmem_data = NULL;

	return lnx_mr_regattr(fid, &attr, flags, mr);
}

int lnx_mr_reg(struct fid *fid, const void *buf, size_t len, uint64_t access,
	       uint64_t offset, uint64_t requested_key, uint64_t flags,
	       struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	return lnx_mr_regv(fid, &iov, 1, access, offset, requested_key, flags,
			   mr, context);
}
