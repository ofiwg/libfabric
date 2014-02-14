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

int psmx_epid_to_epaddr(psm_ep_t ep, psm_epid_t epid, psm_epaddr_t *epaddr)
{
        int err;
        psm_error_t errors;
	psm_epconn_t epconn;

	err = psm_ep_epid_lookup(epid, &epconn);
	if (err == PSM_OK) {
		*epaddr = epconn.addr;
		return 0;
	}

        err = psm_ep_connect(ep, 1, &epid, NULL, &errors, epaddr, 30*1e9);
        if (err != PSM_OK)
                return psmx_errno(err);

	psm_epaddr_setctxt(*epaddr, (void *)epid);
        return 0;
}

static int psmx_av_insert(fid_t fid, const void *addr, size_t count,
			  void **fi_addr, uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	psm_error_t *errors;
	int *mask;
	int err;
	int i;

	fid_av = container_of(fid, struct psmx_fid_av, av.fid);

	errors = (psm_error_t *) calloc(count, sizeof *errors);
	if (!errors)
		return -ENOMEM;

	mask = (int *) calloc(count, sizeof *mask);
	if (!mask) {
		free(errors);
		return -ENOMEM;
	}

	/* prevent connecting to the same ep twice, which is fatal in PSM */
	for (i=0; i<count; i++) {
		psm_epconn_t epconn;
		if (psm_ep_epid_lookup(((psm_epid_t *) addr)[i], &epconn) == PSM_OK)
			((psm_epaddr_t *) fi_addr)[i] = epconn.addr;
		else
			mask[i] = 1;
	}

	err = psm_ep_connect(fid_av->domain->psm_ep, count, 
			(psm_epid_t *) addr, mask, errors,
			(psm_epaddr_t *) fi_addr, 30*1e9);

	for (i=0; i<count; i++){
		if (mask[i] && errors[i] == PSM_OK) {
			psm_epaddr_setctxt(
				((psm_epaddr_t *) fi_addr)[i],
				(void *)((psm_epid_t *) addr)[i]);
		}
	}

	free(mask);
	free(errors);

	return psmx_errno(err);
}

static int psmx_av_remove(fid_t fid, void *fi_addr, size_t count,
			  uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	int err = PSM_OK;
	fid_av = container_of(fid, struct psmx_fid_av, av.fid);
	return psmx_errno(err);
}

static int psmx_av_close(fid_t fid)
{
	struct psmx_fid_av *fid_av;
	fid_av = container_of(fid, struct psmx_fid_av, av.fid);
	free(fid_av);
	return 0;
}

static int psmx_av_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct fi_resource ress;
	int i;

	for (i=0; i<nfids; i++) {
		if (!fids[i].fid)
			return -EINVAL;
		switch (fids[i].fid->fclass) {
		case FID_CLASS_EP:
			if (!fids[i].fid->ops || !fids[i].fid->ops->bind)
				return -EINVAL;
			ress.fid = fid;
			ress.flags = fids[i].flags;
			return fids[i].fid->ops->bind(fids[i].fid, &ress, 1);
		default:
			return -ENOSYS;
		}
	}
	return 0;
}

static int psmx_av_sync(fid_t fid, uint64_t flags, void *context)
{
	/* no-op since insert/remove is synchronous */
	return 0;
}

static int psmx_av_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_av_close,
	.bind = psmx_av_bind,
	.sync = psmx_av_sync,
	.control = psmx_av_control,
};

static struct fi_ops_av psmx_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = psmx_av_insert,
	.remove = psmx_av_remove,
};

int psmx_av_open(fid_t fid, struct fi_av_attr *attr, fid_t *av, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psmx_fid_av *fid_av;

	fid_domain = container_of(fid, struct psmx_fid_domain, domain.fid);

	if (attr) {
		if ((attr->mask & FI_AV_ATTR_TYPE) &&
			attr->type != FI_AV_MAP)
			return -ENOSYS;

		if ((attr->mask & FI_AV_ATTR_ADDR_FORMAT) &&
			attr->addr_format != FI_ADDR)
			return -ENOSYS;

		if ((attr->mask & FI_AV_ATTR_ADDRLEN) &&
			attr->addrlen != sizeof(psm_epaddr_t))
			return -ENOSYS;
	}

	fid_av = (struct psmx_fid_av *) calloc(1, sizeof *fid_av);
	if (!fid_av)
		return -ENOMEM;

	fid_av->domain = fid_domain;
	fid_av->type = FI_AV_MAP;
	fid_av->format = FI_ADDR;
	fid_av->addrlen = sizeof(psm_epaddr_t);

	fid_av->av.fid.size = sizeof(struct fid_av);
	fid_av->av.fid.fclass = FID_CLASS_AV;
	fid_av->av.fid.context = context;
	fid_av->av.fid.ops = &psmx_fi_ops;
	fid_av->av.ops = &psmx_av_ops;

	*av = &fid_av->av.fid;
	return 0;
}

