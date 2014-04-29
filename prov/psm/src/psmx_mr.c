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

static uint64_t mr_key;

int psmx_mr_reg(fid_t fid, const void *buf, size_t len,
		       struct fi_mr_attr *attr, fid_t *mr, void *context)
{
	struct psmx_fid_mr *fid_mr;

	fid_mr = (struct psmx_fid_mr *) calloc(1, sizeof *fid_mr);
	if (!fid_mr)
		return -ENOMEM;

	fid_mr->mr.fid.size = sizeof(struct fid_mr);
	fid_mr->mr.fid.fclass = FID_CLASS_MR;
	fid_mr->mr.fid.context = context;
	fid_mr->mr.fid.ops = &psmx_fi_ops;
	fid_mr->mr.mem_desc = 0x5F109530; /* dummy value "sfi psm" */
	fid_mr->mr.key = ++mr_key;

	*mr = &fid_mr->mr.fid;

	return 0;
}

int psmx_mr_regv(fid_t fid, const struct iovec *iov, size_t count,
			struct fi_mr_attr *attr, fid_t *mr, void *context)
{
	struct psmx_fid_mr *fid_mr;

	fid_mr = (struct psmx_fid_mr *) calloc(1, sizeof *fid_mr);
	if (!fid_mr)
		return -ENOMEM;

	fid_mr->mr.fid.size = sizeof(struct fid_mr);
	fid_mr->mr.fid.fclass = FID_CLASS_MR;
	fid_mr->mr.fid.context = context;
	fid_mr->mr.fid.ops = &psmx_fi_ops;
	fid_mr->mr.mem_desc = 0x5F109530; /* dummy value "sfi psm" */
	fid_mr->mr.key = ++mr_key;

	*mr = &fid_mr->mr.fid;
	return 0;
}

