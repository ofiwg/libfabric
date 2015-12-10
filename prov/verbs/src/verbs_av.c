/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <fi_enosys.h>
#include "fi_verbs.h"


static int fi_ibv_av_close(fid_t fid)
{
	struct fi_ibv_av *fid_av = container_of(fid, struct fi_ibv_av, av.fid);
	free(fid_av);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_av_close,
	.bind = fi_no_bind,
};

/* TODO: match rest of verbs code for variable naming */
int fi_ibv_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		   struct fid_av **av, void *context)
{
	struct fi_ibv_domain *fid_domain;
	struct fi_ibv_av *fid_av;
	int type = FI_AV_MAP;
	size_t count = 64;

	fid_domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	if (!attr)
		return -FI_EINVAL;

	switch (attr->type) {
	case FI_AV_MAP:
	case FI_AV_TABLE:
		type = attr->type;
		break;
	default:
		return -EINVAL;
	}

	if (attr->count)
		count = attr->count;

	fid_av = calloc(1, sizeof *fid_av);
	if (!fid_av)
		return -ENOMEM;

	fid_av->domain = fid_domain;
	fid_av->type = type;
	fid_av->count = count;

	fid_av->av.fid.fclass = FI_CLASS_AV;
	fid_av->av.fid.context = context;
	fid_av->av.fid.ops = &fi_ibv_fi_ops;

	assert(fid_domain->rdm);
	fid_av->av.ops = fi_ibv_rdm_set_av_ops();

	*av = &fid_av->av;
	return 0;
}
