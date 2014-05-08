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

static uint64_t psmx_cntr_read(struct fid_cntr *cntr)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(cntr, struct psmx_fid_cntr, cntr);

	return fid_cntr->counter;
}

static int psmx_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(cntr, struct psmx_fid_cntr, cntr);
	fid_cntr->counter += value;

	if (fid_cntr->wait_obj == FI_CNTR_WAIT_MUT_COND)
		pthread_cond_signal(&fid_cntr->cond);

	return 0;
}

static int psmx_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(cntr, struct psmx_fid_cntr, cntr);
	fid_cntr->counter = value;

	if (fid_cntr->wait_obj == FI_CNTR_WAIT_MUT_COND)
		pthread_cond_signal(&fid_cntr->cond);

	return 0;
}

static int psmx_cntr_wait(struct fid_cntr *cntr, uint64_t threshold)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(cntr, struct psmx_fid_cntr, cntr);

	switch (fid_cntr->wait_obj) {
	case FI_CNTR_WAIT_NONE:
		while (fid_cntr->counter < threshold)
			sched_yield();
		break;

	case FI_CNTR_WAIT_MUT_COND:
		pthread_mutex_lock(&fid_cntr->mutex);
		while (fid_cntr->counter < threshold)
			pthread_cond_wait(&fid_cntr->cond, &fid_cntr->mutex);
		pthread_mutex_unlock(&fid_cntr->mutex);
		break;

	default:
		return -EBADF;
	}

	return 0;
}

static int psmx_cntr_close(fid_t fid)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(fid, struct psmx_fid_cntr, cntr.fid);
	free(fid_cntr);

	return 0;
}

static int psmx_cntr_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct fi_resource ress;
	int err;
	int i;

	for (i=0; i<nfids; i++) {
		if (!fids[i].fid)
			return -EINVAL;
		switch (fids[i].fid->fclass) {
		case FID_CLASS_EP:
		case FID_CLASS_MR:
			if (!fids[i].fid->ops || !fids[i].fid->ops->bind)
				return -EINVAL;
			ress.fid = fid;
			ress.flags = fids[i].flags;
			err = fids[i].fid->ops->bind(fids[i].fid, &ress, 1);
			if (err)
				return err;
			break;

		default:
			return -ENOSYS;
		}
	}
	return 0;
}

static int psmx_cntr_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_cntr_control(fid_t fid, int command, void *arg)
{
	struct psmx_fid_cntr *fid_cntr;

	fid_cntr = container_of(fid, struct psmx_fid_cntr, cntr.fid);

	switch (command) {
	case FI_SETOPSFLAG:
		fid_cntr->flags = *(uint64_t *)arg;
		break;

	case FI_GETOPSFLAG:
		if (!arg)
			return -EINVAL;
		*(uint64_t *)arg = fid_cntr->flags;
		break;

	case FI_GETWAIT:
		if (!arg)
			return -EINVAL;
		((void **)arg)[0] = &fid_cntr->mutex;
		((void **)arg)[1] = &fid_cntr->cond;
		break;

	default:
		return -ENOSYS;
	}

	return 0;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_cntr_close,
	.bind = psmx_cntr_bind,
	.sync = psmx_cntr_sync,
	.control = psmx_cntr_control,
};

static struct fi_ops_cntr psmx_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = psmx_cntr_read,
	.add = psmx_cntr_add,
	.set = psmx_cntr_set,
	.wait = psmx_cntr_wait,
};

int psmx_cntr_alloc(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psmx_fid_cntr *fid_cntr;
	int events , wait_obj;
	uint64_t flags;

	events = FI_CNTR_EVENTS_COMP;
	wait_obj = FI_CNTR_WAIT_NONE;
	flags = 0;

	if (attr->mask & FI_CNTR_ATTR_EVENTS) {
		switch (attr->events) {
		case FI_CNTR_EVENTS_COMP:
			events = attr->events;
			break;

		default:
			psmx_debug("%s: attr->events=%d, supported=%d\n", __func__,
					attr->events, FI_CNTR_EVENTS_COMP);
			return -EINVAL;
		}
	}

	if (attr->mask & FI_CNTR_ATTR_WAIT_OBJ) {
		switch (attr->wait_obj) {
		case FI_CNTR_WAIT_NONE:
		case FI_CNTR_WAIT_MUT_COND:
			wait_obj = attr->wait_obj;
			break;

		default:
			psmx_debug("%s: attr->wait_obj=%d, supported=%d,%d\n", __func__,
					attr->wait_obj, FI_CNTR_WAIT_NONE, FI_CNTR_WAIT_MUT_COND);
			return -EINVAL;
		}
	}

	fid_domain = container_of(domain, struct psmx_fid_domain, domain);
	fid_cntr = (struct psmx_fid_cntr *) calloc(1, sizeof *fid_cntr);
	if (!fid_cntr)
		return -ENOMEM;

	fid_cntr->domain = fid_domain;
	fid_cntr->events = events;
	fid_cntr->wait_obj = wait_obj;
	fid_cntr->flags = flags;
	fid_cntr->cntr.fid.size = sizeof(struct fid_cntr);
	fid_cntr->cntr.fid.fclass = FID_CLASS_CNTR;
	fid_cntr->cntr.fid.context = context;
	fid_cntr->cntr.fid.ops = &psmx_fi_ops;
	fid_cntr->cntr.ops = &psmx_cntr_ops;

	if (wait_obj == FI_CNTR_WAIT_MUT_COND) {
		pthread_mutex_init(&fid_cntr->mutex, NULL);
		pthread_cond_init(&fid_cntr->cond, NULL);
	}

	*cntr = &fid_cntr->cntr;
	return 0;
}

