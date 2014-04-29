/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "psmx.h"

static int psmx_mr_close(fid_t fid)
{
	struct psmx_fid_mr *fid_mr;

	fid_mr = container_of(fid, struct psmx_fid_mr, mr.fid);
	fid_mr->signature = 0;

	free(fid_mr);

	return 0;
}

static int psmx_mr_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	return -ENOSYS;
}

static int psmx_mr_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_mr_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_mr_close,
	.bind = psmx_mr_bind,
	.sync = psmx_mr_sync,
	.control = psmx_mr_control,
};

static void psmx_mr_normalize_iov(struct iovec *iov, size_t *count)
{
	struct iovec tmp_iov;
	int i, j, n, new_len;

	n = *count;

	if (!n)
		return;

	/* sort segments by base address */
	for (i = 0; i < n - 1; i++) {
		for (j = i + 1; j < n; j++) {
			if (iov[i].iov_base > iov[j].iov_base) {
				tmp_iov = iov[i];
				iov[i] = iov[j];
				iov[j] = tmp_iov;
			}
		}
	}

	/* merge overlapping segments */
	for (i = 0; i < n - 1; i++) {
		if (iov[i].iov_len == 0)
			continue;

		for (j = i + 1; j < n; j++) {
			if (iov[j].iov_len == 0)
				continue;

			if (iov[i].iov_base + iov[i].iov_len >= iov[j].iov_base) {
				new_len = iov[j].iov_base + iov[j].iov_len - iov[i].iov_base;
				if (new_len > iov[i].iov_len)
					iov[i].iov_len = new_len;
				iov[j].iov_len = 0;
			}
			else {
				break;
			}
		}
	}

	/* remove empty segments */
	for (i = 0, j = 1; i < n; i++, j++) {
		if (iov[i].iov_len)
			continue;

		while (j < n && iov[j].iov_len == 0)
			j++;

		if (j >= n)
			break;

		iov[i] = iov[j];
		iov[j].iov_len = 0;
	}

	*count = i;
}

static int psmx_mr_reg(fid_t fid, const void *buf, size_t len,
			uint64_t access, uint64_t requested_key,
			uint64_t flags, fid_t *mr, void *context)
{
	struct psmx_fid_mr *fid_mr;

	fid_mr = (struct psmx_fid_mr *) calloc(1, sizeof(*fid_mr) + sizeof(struct iovec));
	if (!fid_mr)
		return -ENOMEM;

	fid_mr->mr.fid.size = sizeof(struct fid_mr);
	fid_mr->mr.fid.fclass = FID_CLASS_MR;
	fid_mr->mr.fid.context = context;
	fid_mr->mr.fid.ops = &psmx_fi_ops;
	fid_mr->mr.mem_desc = (uint64_t)(uintptr_t)fid_mr;
	fid_mr->mr.key = (uint64_t)(uintptr_t)fid_mr; /* requested_key is ignored */
	fid_mr->signature = PSMX_MR_SIGNATURE;
	fid_mr->access = access;
	fid_mr->flags = flags;
	fid_mr->iov_count = 1;
	fid_mr->iov[0].iov_base = (void *)buf;
	fid_mr->iov[0].iov_len = len;

	*mr = &fid_mr->mr.fid;

	return 0;
}

static int psmx_mr_regv(fid_t fid, const struct iovec *iov, size_t count,
			uint64_t access, uint64_t requested_key,
			uint64_t flags, fid_t *mr, void *context)
{
	struct psmx_fid_mr *fid_mr;
	int i;

	if (count == 0 || iov == NULL)
		return -EINVAL;

	fid_mr = (struct psmx_fid_mr *)
			calloc(1, sizeof(*fid_mr) +
				  sizeof(struct iovec) * count);
	if (!fid_mr)
		return -ENOMEM;

	fid_mr->mr.fid.size = sizeof(struct fid_mr);
	fid_mr->mr.fid.fclass = FID_CLASS_MR;
	fid_mr->mr.fid.context = context;
	fid_mr->mr.fid.ops = &psmx_fi_ops;
	fid_mr->mr.mem_desc = (uint64_t)(uintptr_t)fid_mr;
	fid_mr->mr.key = (uint64_t)(uintptr_t)fid_mr; /* requested_key is ignored */
	fid_mr->signature = PSMX_MR_SIGNATURE;
	fid_mr->access = access;
	fid_mr->flags = flags;
	fid_mr->iov_count = count;
	for (i=0; i<count; i++)
		fid_mr->iov[i] = iov[i];

	psmx_mr_normalize_iov(fid_mr->iov, &fid_mr->iov_count);

	*mr = &fid_mr->mr.fid;

	return 0;
}

static int psmx_mr_regattr(fid_t fid, const struct fi_mr_attr *attr,
			uint64_t flags, fid_t *mr)
{
	struct psmx_fid_mr *fid_mr;
	int i;

	if (!attr)
		return -EINVAL;

	if (!(attr->mask & FI_MR_ATTR_IOV))
		return -EINVAL;

	if (attr->iov_count == 0 || attr->mr_iov == NULL)
		return -EINVAL;

	fid_mr = (struct psmx_fid_mr *)
			calloc(1, sizeof(*fid_mr) +
				  sizeof(struct iovec) * attr->iov_count);
	if (!fid_mr)
		return -ENOMEM;

	fid_mr->mr.fid.size = sizeof(struct fid_mr);
	fid_mr->mr.fid.fclass = FID_CLASS_MR;
	fid_mr->mr.fid.ops = &psmx_fi_ops;
	fid_mr->mr.mem_desc = (uint64_t)(uintptr_t)fid_mr;
	fid_mr->mr.key = (uint64_t)(uintptr_t)fid_mr;
	fid_mr->signature = PSMX_MR_SIGNATURE;
	fid_mr->access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
	fid_mr->flags = flags;
	fid_mr->iov_count = attr->iov_count;
	for (i=0; i<attr->iov_count; i++)
		fid_mr->iov[i] = attr->mr_iov[i];

	if (attr->mask & FI_MR_ATTR_CONTEXT)
		fid_mr->mr.fid.context = attr->context;

	if (attr->mask & FI_MR_ATTR_ACCESS)
		fid_mr->access = attr->access;

	if (attr->mask & FI_MR_ATTR_KEY)
		; /* requested_key is ignored */

	psmx_mr_normalize_iov(fid_mr->iov, &fid_mr->iov_count);

	*mr = &fid_mr->mr.fid;

	return 0;
}

struct fi_ops_mr psmx_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.mr_reg = psmx_mr_reg,
	.mr_regv = psmx_mr_regv,
	.mr_regattr = psmx_mr_regattr,
};

