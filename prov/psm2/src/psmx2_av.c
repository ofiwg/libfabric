/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

static void psmx2_set_epaddr_context(struct psmx2_fid_domain *domain,
				     psm2_epid_t epid, psm2_epaddr_t epaddr)
{
	struct psmx2_epaddr_context *context;

	context = (void *)psm2_epaddr_getctxt(epaddr);
	if (context) {
		if (context->domain != domain || context->epid != epid) {
			FI_WARN(&psmx2_prov, FI_LOG_AV,
				"domain or epid doesn't match\n");
			context = NULL;
		}
	}

	if (context)
		return;

	context = malloc(sizeof *context);
	if (!context) {
		FI_WARN(&psmx2_prov, FI_LOG_AV,
			"cannot allocate context\n");
		return;
	}

	context->domain = domain;
	context->epid = epid;
	psm2_epaddr_setctxt(epaddr, context);
}

int psmx2_epid_to_epaddr(struct psmx2_fid_domain *domain,
			 psm2_epid_t epid, psm2_epaddr_t *epaddr)
{
        int err;
        psm2_error_t errors;
	psm2_epconn_t epconn;
	struct psmx2_epaddr_context *context;

	err = psm2_ep_epid_lookup(epid, &epconn);
	if (err == PSM2_OK) {
		context = psm2_epaddr_getctxt(epconn.addr);
		if (context && context->epid  == epid) {
			*epaddr = epconn.addr;
			return 0;
		}
	}

        err = psm2_ep_connect(domain->psm2_ep, 1, &epid, NULL, &errors, epaddr, 30*1e9);
        if (err != PSM2_OK)
                return psmx2_errno(err);

	psmx2_set_epaddr_context(domain,epid,*epaddr);

        return 0;
}

static int psmx2_av_check_table_size(struct psmx2_fid_av *av, size_t count)
{
	size_t new_count;
	psm2_epid_t *new_psm2_epids;
	psm2_epaddr_t *new_psm2_epaddrs;

	new_count = av->count;
	while (new_count < av->last + count)
		new_count = new_count * 2 + 1;

	if ((new_count <= av->count) && av->psm2_epids)
		return 0;

	new_psm2_epids = realloc(av->psm2_epids,
				 new_count * sizeof(*new_psm2_epids));
	if (!new_psm2_epids)
		return -FI_ENOMEM;

	av->psm2_epids = new_psm2_epids;

	new_psm2_epaddrs = realloc(av->psm2_epaddrs,
				   new_count * sizeof(*new_psm2_epaddrs));
	if (!new_psm2_epaddrs)
		return -FI_ENOMEM;

	av->psm2_epaddrs = new_psm2_epaddrs;
	av->count = new_count;
	return 0;
}

static int psmx2_av_insert(struct fid_av *av, const void *addr,
			   size_t count, fi_addr_t *fi_addr,
			   uint64_t flags, void *context)
{
	struct psmx2_fid_av *av_priv;
	psm2_error_t *errors;
	int error_count = 0;
	int *mask;
	int i, j;
	fi_addr_t *result = NULL;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_eq_event *event;

	av_priv = container_of(av, struct psmx2_fid_av, av);

	if ((av_priv->flags & FI_EVENT) && !av_priv->eq)
		return -FI_ENOEQ;

	errors = (psm2_error_t *) calloc(count, sizeof *errors);
	if (!errors)
		return -FI_ENOMEM;

	mask = (int *) calloc(count, sizeof *mask);
	if (!mask) {
		free(errors);
		return -FI_ENOMEM;
	}

	if (av_priv->type == FI_AV_TABLE) {
		if (psmx2_av_check_table_size(av_priv, count)) {
			free(mask);
			free(errors);
			return -FI_ENOMEM;
		}

		for (i=0; i<count; i++)
			av_priv->psm2_epids[av_priv->last + i] =
				((psm2_epid_t *)addr)[i];

		result = fi_addr;
		addr = (const void *)(av_priv->psm2_epids + av_priv->last);
		fi_addr = (fi_addr_t *)(av_priv->psm2_epaddrs + av_priv->last);
	}

	/* prevent connecting to the same ep twice, which is fatal in PSM */
	for (i=0; i<count; i++) {
		psm2_epconn_t epconn;
		if (psm2_ep_epid_lookup(((psm2_epid_t *) addr)[i], &epconn) == PSM2_OK) {
			epaddr_context = psm2_epaddr_getctxt(epconn.addr);
			if (epaddr_context && epaddr_context->epid  == ((psm2_epid_t *) addr)[i])
				((psm2_epaddr_t *) fi_addr)[i] = epconn.addr;
			else
				mask[i] = 1;
		}
		else {
			mask[i] = 1;
		}
	}

	psm2_ep_connect(av_priv->domain->psm2_ep, count, 
			(psm2_epid_t *) addr, mask, errors,
			(psm2_epaddr_t *) fi_addr, 30*1e9);

	for (i=0; i<count; i++){
		if (!mask[i])
			continue;

		if (errors[i] == PSM2_OK || errors[i] == PSM2_EPID_ALREADY_CONNECTED) {
			psmx2_set_epaddr_context(av_priv->domain,
						 ((psm2_epid_t *) addr)[i],
						 ((psm2_epaddr_t *) fi_addr)[i]);
		}
		else {
			psm2_epconn_t epconn;

			/* If duplicated addresses are passed to psm2_ep_connect(), all but one will fail
			 * with error "Endpoint could not be reached". They should be treated as already
			 * connected.
			 */
			if (psm2_ep_epid_lookup(((psm2_epid_t *) addr)[i], &epconn) == PSM2_OK) {
				epaddr_context = psm2_epaddr_getctxt(epconn.addr);
				if (epaddr_context && epaddr_context->epid  == ((psm2_epid_t *) addr)[i]) {
					((psm2_epaddr_t *) fi_addr)[i] = epconn.addr;
					continue;
				}
			}

			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"%d: psm2_ep_connect returned %s. remote epid=%lx.\n",
				i, psm2_error_get_string(errors[i]),
				((psm2_epid_t *)addr)[i]);
			if (((psm2_epid_t *)addr)[i] == 0)
				FI_INFO(&psmx2_prov, FI_LOG_AV,
					"does the application depend on the provider"
					"to resolve IP address into endpoint id? if so"
					"check if the name server has started correctly"
					"at the other side.\n");
			fi_addr[i] = FI_ADDR_NOTAVAIL;
			error_count++;

			if (av_priv->flags & FI_EVENT) {
				event = psmx2_eq_create_event(av_priv->eq,
							      FI_AV_COMPLETE,		/* event */
							      context,			/* context */
							      i,			/* data: failed index */
							      psmx2_errno(errors[i]),	/* err */
							      errors[i],		/* prov_errno */
							      NULL,			/* err_data */
							      0);			/* err_data_size */
				if (!event) {
					free(mask);
					free(errors);
					return -FI_ENOMEM;
				}

				psmx2_eq_enqueue_event(av_priv->eq, event);
			}
		}
	}

	free(mask);
	free(errors);

	if (av_priv->type == FI_AV_TABLE) {
		/* NOTE: unresolved addresses are left in the AV table */
		if (result) {
			for (i=0; i<count; i++) {
				j = av_priv->last + i;
				if ((fi_addr_t)av_priv->psm2_epaddrs[j] == FI_ADDR_NOTAVAIL)
					result[i] = FI_ADDR_NOTAVAIL;
				else
					result[i] = j;
			}
		}
		av_priv->last += count;
	}

	if (!(av_priv->flags & FI_EVENT))
		return count - error_count;

	event = psmx2_eq_create_event(av_priv->eq,
				      FI_AV_COMPLETE,		/* event */
				      context,			/* context */
				      count - error_count,	/* data: succ count */
				      0,			/* err */
				      0,			/* prov_errno */
				      NULL,			/* err_data */
				      0);			/* err_data_size */
	if (!event)
		return -FI_ENOMEM;

	psmx2_eq_enqueue_event(av_priv->eq, event);
	return 0;
}

static int psmx2_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count,
			   uint64_t flags)
{
	int err = PSM2_OK;

	return psmx2_errno(err);
}

static int psmx2_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr,
			   size_t *addrlen)
{
	struct psmx2_fid_av *av_priv;
	struct psmx2_epaddr_context *context;
	psm2_epid_t epid;
	int idx;

	if (!addr || !addrlen)
		return -FI_EINVAL;

	av_priv = container_of(av, struct psmx2_fid_av, av);

	if (av_priv->type == FI_AV_TABLE) {
		idx = (int)(int64_t)fi_addr;
		if (idx >= av_priv->last)
			return -FI_EINVAL;

		epid = av_priv->psm2_epids[idx];
	}
	else {
		context = psm2_epaddr_getctxt((void *)fi_addr);
		epid = context->epid;
	}

	if (*addrlen >= sizeof(epid))
		*(psm2_epid_t *)addr = epid;
	else
		memcpy(addr, &epid, *addrlen);
	*addrlen = sizeof(epid);

	return 0;
}

static const char *psmx2_av_straddr(struct fid_av *av, const void *addr,
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

static int psmx2_av_close(fid_t fid)
{
	struct psmx2_fid_av *av;

	av = container_of(fid, struct psmx2_fid_av, av.fid);
	psmx2_domain_release(av->domain);
	if (av->psm2_epids)
		free(av->psm2_epids);
	if (av->psm2_epaddrs)
		free(av->psm2_epaddrs);
	free(av);
	return 0;
}

static int psmx2_av_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct psmx2_fid_av *av;
	struct psmx2_fid_eq *eq;

	av = container_of(fid, struct psmx2_fid_av, av.fid);

	if (!bfid)
		return -FI_EINVAL;

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct psmx2_fid_eq, eq.fid);
		av->eq = eq;
		break;

	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static struct fi_ops psmx2_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_av_close,
	.bind = psmx2_av_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_av psmx2_av_ops = {
	.size = sizeof(struct fi_ops_av),
	.insert = psmx2_av_insert,
	.insertsvc = fi_no_av_insertsvc,
	.insertsym = fi_no_av_insertsym,
	.remove = psmx2_av_remove,
	.lookup = psmx2_av_lookup,
	.straddr = psmx2_av_straddr,
};

int psmx2_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		  struct fid_av **av, void *context)
{
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_fid_av *av_priv;
	int type = FI_AV_MAP;
	size_t count = 64;
	uint64_t flags = 0;

	domain_priv = container_of(domain, struct psmx2_fid_domain, domain);

	if (attr) {
		switch (attr->type) {
		case FI_AV_MAP:
		case FI_AV_TABLE:
			type = attr->type;
			break;
		default:
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->type=%d, supported=%d %d\n",
				attr->type, FI_AV_MAP, FI_AV_TABLE);
			return -FI_EINVAL;
		}

		count = attr->count;
		flags = attr->flags;

		if (flags & (FI_READ | FI_SYMMETRIC)) {
			FI_INFO(&psmx2_prov, FI_LOG_AV,
				"attr->flags=%x, supported=%x\n",
				attr->flags, FI_EVENT);
			return -FI_EINVAL;
		}
	}

	av_priv = (struct psmx2_fid_av *) calloc(1, sizeof *av_priv);
	if (!av_priv)
		return -FI_ENOMEM;

	psmx2_domain_acquire(domain_priv);

	av_priv->domain = domain_priv;
	av_priv->type = type;
	av_priv->addrlen = sizeof(psm2_epaddr_t);
	av_priv->count = count;
	av_priv->flags = flags;

	av_priv->av.fid.fclass = FI_CLASS_AV;
	av_priv->av.fid.context = context;
	av_priv->av.fid.ops = &psmx2_fi_ops;
	av_priv->av.ops = &psmx2_av_ops;

	*av = &av_priv->av;
	return 0;
}

