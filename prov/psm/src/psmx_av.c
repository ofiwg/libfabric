/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

static void psmx_set_epaddr_context(struct psmx_fid_domain *domain,
				    psm_epid_t epid, psm_epaddr_t epaddr)
{
	struct psmx_epaddr_context *context;

	context = (void *)psm_epaddr_getctxt(epaddr);
	if (context) {
		if (context->domain != domain || context->epid != epid) {
			fprintf(stderr, "%s: domain or epid doesn't match\n", __func__);
			context = NULL;
		}
	}

	if (context)
		return;

	context = malloc(sizeof *context);
	if (!context) {
		fprintf(stderr, "%s: cannot allocate context\n", __func__);
		return;
	}

	context->domain = domain;
	context->epid = epid;
	psm_epaddr_setctxt(epaddr, context);
}

int psmx_epid_to_epaddr(struct psmx_fid_domain *domain,
			psm_epid_t epid, psm_epaddr_t *epaddr)
{
        int err;
        psm_error_t errors;
	psm_epconn_t epconn;

	err = psm_ep_epid_lookup(epid, &epconn);
	if (err == PSM_OK) {
		*epaddr = epconn.addr;
		psmx_set_epaddr_context(domain,epid,*epaddr);
		return 0;
	}

        err = psm_ep_connect(domain->psm_ep, 1, &epid, NULL, &errors, epaddr, 30*1e9);
        if (err != PSM_OK)
                return psmx_errno(err);

	psmx_set_epaddr_context(domain,epid,*epaddr);

        return 0;
}

static int psmx_av_check_table_size(struct psmx_fid_av *fid_av, size_t count)
{
	size_t new_count;
	psm_epid_t *new_psm_epids;
	psm_epaddr_t *new_psm_epaddrs;

	new_count = fid_av->count;
	while (new_count < fid_av->last + count)
		new_count = new_count * 2 + 1;

	if ((new_count <= fid_av->count) && fid_av->psm_epids)
		return 0;

	new_psm_epids = realloc(fid_av->psm_epids, new_count * sizeof(*new_psm_epids));
	if (!new_psm_epids)
		return -ENOMEM;

	fid_av->psm_epids = new_psm_epids;

	new_psm_epaddrs = realloc(fid_av->psm_epaddrs, new_count * sizeof(*new_psm_epaddrs));
	if (!new_psm_epaddrs)
		return -ENOMEM;

	fid_av->psm_epaddrs = new_psm_epaddrs;
	fid_av->count = new_count;
	return 0;
}

static int psmx_av_insert(struct fid_av *av, const void *addr, size_t count,
			  void **fi_addr, uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	psm_error_t *errors;
	int *mask;
	int err;
	int i;
	void **result = NULL;

	fid_av = container_of(av, struct psmx_fid_av, av);

	errors = (psm_error_t *) calloc(count, sizeof *errors);
	if (!errors)
		return -ENOMEM;

	mask = (int *) calloc(count, sizeof *mask);
	if (!mask) {
		free(errors);
		return -ENOMEM;
	}

	if (fid_av->type == FI_AV_TABLE) {
		if (psmx_av_check_table_size(fid_av, count)) {
			free(mask);
			free(errors);
			return -ENOMEM;
		}

		for (i=0; i<count; i++)
			fid_av->psm_epids[fid_av->last + i] = ((psm_epid_t *)addr)[i];

		result = fi_addr;
		addr = (const void *)(fid_av->psm_epids + fid_av->last);
		fi_addr = (void **)(fid_av->psm_epaddrs + fid_av->last);
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
			psmx_set_epaddr_context(fid_av->domain,
						((psm_epid_t *) addr)[i],
						((psm_epaddr_t *) fi_addr)[i]);
		}
	}

	free(mask);
	free(errors);

	if (fid_av->type == FI_AV_TABLE) {
		if (result) {
			for (i=0; i<count; i++)
				((uint64_t *)result)[i] = fid_av->last + i;
		}
		fid_av->last += count;
	}

	return psmx_errno(err);
}

static int psmx_av_remove(struct fid_av *av, void *fi_addr, size_t count,
			  uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	int err = PSM_OK;

	fid_av = container_of(av, struct psmx_fid_av, av);

	return psmx_errno(err);
}

static int psmx_av_lookup(struct fid_av *av, const void *fi_addr, void *addr,
			  size_t *addrlen)
{
	struct psmx_fid_av *fid_av;
	struct psmx_epaddr_context *context;
	psm_epid_t epid;
	int idx;

	if (!addr || !addrlen)
		return -EINVAL;

	fid_av = container_of(av, struct psmx_fid_av, av);

	if (fid_av->type == FI_AV_TABLE) {
		idx = (int)(int64_t)fi_addr;
		if (idx >= fid_av->last)
			return -EINVAL;

		epid = fid_av->psm_epids[idx];
	}
	else {
		context = psm_epaddr_getctxt((void *)fi_addr);
		epid = context->epid;
	}

	if (*addrlen >= sizeof(epid))
		*(psm_epid_t *)addr = epid;
	else
		memcpy(addr, &epid, *addrlen);
	*addrlen = sizeof(epid);

	return 0;
}

static const char *psmx_av_straddr(struct fid_av *av, const void *addr,
				   char *buf, size_t *len)
{
	int n;

	if (!buf || !len)
		return NULL;

	n = snprintf(buf, *len, "%lx", (uint64_t)(uintptr_t)addr);
	if (n < 0)
		return NULL;

	*len = n + 1;
	return buf;
}

static int psmx_av_close(fid_t fid)
{
	struct psmx_fid_av *fid_av;
	fid_av = container_of(fid, struct psmx_fid_av, av.fid);
	if (fid_av->psm_epids)
		free(fid_av->psm_epids);
	if (fid_av->psm_epaddrs)
		free(fid_av->psm_epaddrs);
	free(fid_av);
	return 0;
}

/* Currently only support synchronous insertions */
static int psmx_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return -FI_ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.close = psmx_av_close,
	.bind = psmx_av_bind,
};

static struct fi_ops_av psmx_av_ops = {
	.insert = psmx_av_insert,
	.remove = psmx_av_remove,
	.lookup = psmx_av_lookup,
	.straddr = psmx_av_straddr,
};

int psmx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psmx_fid_av *fid_av;
	int type = FI_AV_MAP;
	size_t count = 64;

	fid_domain = container_of(domain, struct psmx_fid_domain, domain);

	if (attr) {
		if (attr->mask & FI_AV_ATTR_TYPE) {
			switch (attr->type) {
			case FI_AV_MAP:
			case FI_AV_TABLE:
				type = attr->type;
				break;
			default:
				psmx_debug("%s: attr->type=%d, supported=%d %d\n",
					__func__, attr->type, FI_AV_MAP, FI_AV_TABLE);
				return -EINVAL;
			}
		}

		if (attr->mask & FI_AV_ATTR_COUNT) {
			count = attr->count;
		}
	}

	fid_av = (struct psmx_fid_av *) calloc(1, sizeof *fid_av);
	if (!fid_av)
		return -ENOMEM;

	fid_av->domain = fid_domain;
	fid_av->type = type;
	fid_av->format = FI_ADDR;
	fid_av->addrlen = sizeof(psm_epaddr_t);
	fid_av->count = count;

	fid_av->av.fid.size = sizeof(struct fid_av);
	fid_av->av.fid.fclass = FID_CLASS_AV;
	fid_av->av.fid.context = context;
	fid_av->av.fid.ops = &psmx_fi_ops;
	fid_av->av.ops = &psmx_av_ops;

	*av = &fid_av->av;
	return 0;
}

