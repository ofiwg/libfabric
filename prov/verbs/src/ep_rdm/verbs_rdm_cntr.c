/*
 * Copyright (c) 2013-2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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

#include "ofi_enosys.h"

#include "verbs_rdm.h"


static uint64_t fi_ibv_rdm_cntr_read(struct fid_cntr *cntr_fid)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	return cntr->value;
}

static uint64_t fi_ibv_rdm_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	return cntr->err_count;
}

static int fi_ibv_rdm_cntr_add(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	cntr->value += value;
	return 0;
}

static int fi_ibv_rdm_cntr_set(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	cntr->value = value;
	return 0;
}

static int fi_ibv_rdm_cntr_adderr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	cntr->err_count += value;
	return 0;
}

static int fi_ibv_rdm_cntr_seterr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(cntr_fid, struct fi_ibv_rdm_cntr, fid);
	cntr->err_count = value;
	return 0;
}

static struct fi_ops_cntr fi_ibv_rdm_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = fi_ibv_rdm_cntr_read,
	.readerr = fi_ibv_rdm_cntr_readerr,
	.add = fi_ibv_rdm_cntr_add,
	.set = fi_ibv_rdm_cntr_set,
	.wait = fi_no_cntr_wait,
	.adderr = fi_ibv_rdm_cntr_adderr,
	.seterr = fi_ibv_rdm_cntr_seterr,
};

static int fi_ibv_rdm_cntr_close(struct fid *fid)
{
	struct fi_ibv_rdm_cntr *cntr =
		container_of(fid, struct fi_ibv_rdm_cntr, fid);

	if (ofi_atomic_get32(&cntr->ep_ref) > 0) {
		return -FI_EBUSY;
	}

	free(cntr);
	return 0;
}

static struct fi_ops fi_ibv_rdm_cntr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_rdm_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int fi_rbv_rdm_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr_fid, void *context)
{
	struct fi_ibv_rdm_cntr *cntr;

	struct fi_ibv_domain *dom =
		container_of(domain, struct fi_ibv_domain, util_domain.domain_fid);

	if (attr) {
		switch (attr->events) {
		case FI_CNTR_EVENTS_COMP:
			break;
		default:
			return -FI_ENOSYS;
		}

		switch (attr->wait_obj) {
		case FI_WAIT_NONE:
		case FI_WAIT_UNSPEC:
			break;
		case FI_WAIT_MUTEX_COND:
		case FI_WAIT_SET:
		case FI_WAIT_FD:
		default:
			return -FI_ENOSYS;
		}

		if (attr->flags) {
			return -FI_EINVAL;
		}
	}

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	if (attr) {
		assert(sizeof(cntr->attr) == sizeof(*attr));
		memcpy(&cntr->attr, attr, sizeof(*attr));
	}

	cntr->fid.fid.fclass = FI_CLASS_CNTR;
	cntr->fid.fid.context = context;
	cntr->fid.fid.ops = &fi_ibv_rdm_cntr_fi_ops;
	cntr->fid.ops = &fi_ibv_rdm_cntr_ops;
	cntr->domain = dom;
	ofi_atomic_initialize32(&cntr->ep_ref, 0);

	*cntr_fid = &cntr->fid;

	return FI_SUCCESS;
}
