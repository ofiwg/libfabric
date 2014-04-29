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
#include "uthash.h"

static struct psmx_hash {
        psm_epid_t      epid;
        psm_epaddr_t    epaddr;
        UT_hash_handle  hh;
} *psmx_addr_hash = NULL;

#define PSMX_HASH_ADD_ADDR(_epid, _epaddr) do { \
                struct psmx_hash *s; \
                s = (struct psmx_hash *)malloc(sizeof(*s)); \
                if (!s) return -1; \
                s->epid = (_epid); \
                s->epaddr = (_epaddr); \
                HASH_ADD(hh, psmx_addr_hash, epid, sizeof(psm_epid_t), s); \
        } while (0)

#define PSMX_HASH_GET_ADDR(_epid, _epaddr) do { \
                struct psmx_hash *s; \
                HASH_FIND(hh, psmx_addr_hash, &(_epid), sizeof(psm_epid_t), s); \
                if (s) (_epaddr) = s->epaddr; \
        } while (0)

#define PSMX_HASH_DEL_ADDR(_epid) do { \
                struct psmx_hash *s; \
                HASH_FIND(hh, psmx_addr_hash, &(_epid), sizeof(psm_epid_t), s); \
                if (s) {HASH_DELETE(hh, psmx_addr_hash, s); } \
        } while (0)

/* This funciton is a special case of psmx_av_insert_with_hash, in that it
 * handles only one address, and doesn't need the AV fid as input. When making
 * changes to either of them, make sure the two functions are compatible.
 */
int psmx_epid_to_epaddr(psm_ep_t ep, psm_epid_t epid, psm_epaddr_t *epaddr)
{
        int err;
        psm_error_t errors;

        *epaddr = 0;
        PSMX_HASH_GET_ADDR(epid, *epaddr);
        if (*epaddr)
                return 0;

        err = psm_ep_connect(ep, 1, &epid, NULL, &errors, epaddr, 30*1e9);
        if (err != PSM_OK)
                return psmx_errno(err);

	psm_epaddr_setctxt(*epaddr, (void *)epid);
        PSMX_HASH_ADD_ADDR(epid, *epaddr);
        return 0;
}

static int psmx_av_insert(fid_t fid, const void *addr, size_t count,
			  void **fi_addr, uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	psm_error_t *errors;
	int err;
	int i;

	fid_av = container_of(fid, struct psmx_fid_av, av.fid);

	errors = (psm_error_t *) calloc(count, sizeof *errors);
	if (!errors)
		return -ENOMEM;

	err = psm_ep_connect(fid_av->domain->psm_ep, count, 
			(psm_epid_t *) addr, NULL, errors,
			(psm_epaddr_t *) fi_addr, 30*1e9);

	for (i=0; i<count; i++){
		if (errors[i] == PSM_OK) {
			psm_epaddr_setctxt(
				((psm_epaddr_t *) fi_addr)[i],
				(void *)((psm_epid_t *) addr)[i]);
		}
	}

	free(errors);

	return psmx_errno(err);
}

static int psmx_av_insert_with_hash(fid_t fid, const void *addr, size_t count,
			  void **fi_addr, uint64_t flags)
{
	struct psmx_fid_av *fid_av;
	int *mask;
	psm_error_t *errors;
	int err;
	int i;
	psm_epid_t *epid = (psm_epid_t *)addr;
	psm_epaddr_t *epaddr = (psm_epaddr_t *)fi_addr;

	fid_av = container_of(fid, struct psmx_fid_av, av.fid);

	mask = (int *) calloc(count, sizeof *mask);
	if (!mask)
		return -ENOMEM;

	errors = (psm_error_t *) calloc(count, sizeof *errors);
	if (!errors) {
		free(mask);
		return -ENOMEM;
	}

	for (i=0; i<count; i++) {
		epaddr[i] = 0;
		PSMX_HASH_GET_ADDR(epid[i], epaddr[i]);
		if (epaddr[i] == 0)
			mask[i] = 1;
	}

	err = psm_ep_connect(fid_av->domain->psm_ep, count, 
			epid, mask, errors, epaddr, 30*1e9);

	for (i=0; i<count; i++) {
		if (mask[i] && errors[i] == PSM_OK) {
			psm_epaddr_setctxt(epaddr[i], (void *)epid[i]);
			PSMX_HASH_ADD_ADDR(epid[i], epaddr[i]);
		}
	}

	free(errors);
	free(mask);

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

static int psmx_av_remove_with_hash(fid_t fid, void *fi_addr, size_t count,
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
	/* no need to bind an EQ since insert/remove is synchronous */
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

static struct fi_ops_av psmx_av_ops_with_hash = {
	.size = sizeof(struct fi_ops_av),
	.insert = psmx_av_insert_with_hash,
	.remove = psmx_av_remove_with_hash,
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

	if (fid_domain->reserved_tag_bits & PSMX_NONMATCH_BIT)
		fid_av->av.ops = &psmx_av_ops_with_hash;
	else
		fid_av->av.ops = &psmx_av_ops;

	*av = &fid_av->av.fid;
	return 0;
}

