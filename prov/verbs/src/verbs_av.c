/*
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
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

#include <pthread.h>
#include <stdio.h>

#include <fi_enosys.h>
#include "fi_verbs.h"


static int fi_ibv_av_close(fid_t fid)
{
	struct fi_ibv_av *av = container_of(fid, struct fi_ibv_av, av_fid.fid);
	free(av);
	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_av_close,
	.bind = fi_no_bind,
};

/* TODO: match rest of verbs code for variable naming */
int fi_ibv_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		   struct fid_av **av_fid, void *context)
{
	struct fi_ibv_domain *fid_domain;
	struct fi_ibv_av *av;
	size_t count = 64;

	fid_domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	if (!attr)
		return -FI_EINVAL;

	switch (attr->type) {
	case FI_AV_UNSPEC:
		attr->type = FI_AV_MAP;
	case FI_AV_MAP:
	case FI_AV_TABLE:
		break;
	default:
		return -EINVAL;
	}

	if (attr->count)
		count = attr->count;

	av = calloc(1, sizeof *av);
	if (!av)
		return -ENOMEM;

	assert(fid_domain->rdm);
	av->domain = fid_domain;
	av->type = attr->type;
	av->count = count;

	av->av_fid.fid.fclass = FI_CLASS_AV;
	av->av_fid.fid.context = context;
	av->av_fid.fid.ops = &fi_ibv_fi_ops;

	av->av_fid.ops = fi_ibv_rdm_set_av_ops();

	*av_fid = &av->av_fid;
	return 0;
}
