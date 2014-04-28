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

static ssize_t psmx_ep_cancel(fid_t fid, struct fi_context *context)
{
	struct psmx_fid_ep *fid_ep;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	if (!fid_ep->domain)
		return -EBADF;

	if (!context)
		return -EINVAL;

	if (context->internal[0] == NULL)
		return 0;

	err = psm_mq_cancel((psm_mq_req_t *)&context->internal[0]);
	return psmx_errno(err);
}

static int psmx_ep_getopt(fid_t fid, int level, int optname,
			void *optval, size_t *optlen)
{
	struct psmx_fid_ep *fid_ep;
	uint32_t size;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	if (level != FI_OPT_ENDPOINT)
		return -ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MAX_BUFFERED_SEND:
		if (!optval)
			return 0;

		if (!optlen || *optlen < sizeof(size_t))
			return -EINVAL;

		if (!fid_ep->domain)
			return -EBADF;

		/* FIXME:
		 * PSM has different thresholds for fabric & shm data path. Just use
		 * the value of the fabric path at this point.
		 */
		err = psm_mq_getopt(fid_ep->domain->psm_mq, PSM_MQ_RNDV_IPATH_SZ, &size);
		if (err)
			return psmx_errno(err);

		*(size_t *)optval = size;
		*optlen = sizeof(size_t);
		break;

	default:
		return -ENOPROTOOPT;
	}

	return 0;
}

static int psmx_ep_setopt(fid_t fid, int level, int optname,
			const void *optval, size_t optlen)
{
	struct psmx_fid_ep *fid_ep;
	uint32_t size;
	int err;

	if (level != FI_OPT_ENDPOINT)
		return -ENOPROTOOPT;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	switch (optname) {
	case FI_OPT_MAX_BUFFERED_SEND:
		if (!optval)
			return -EFAULT;

		if (optlen != sizeof(size_t))
			return -EINVAL;

		if (!fid_ep->domain)
			return -EBADF;

		/* FIXME:
		 * PSM has different thresholds for fabric & shm data path. Only set
		 * the value of the fabric path at this point.
		 */
		size = *(size_t *)optval;
		err = psm_mq_setopt(fid_ep->domain->psm_mq, PSM_MQ_RNDV_IPATH_SZ, &size);
		if (err)
			return psmx_errno(err);
		break;

	default:
		return -ENOPROTOOPT;
	}

	return 0;
}

static int psmx_ep_close(fid_t fid)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	free(fid_ep);

	return 0;
}

static int psmx_ep_bind(fid_t fid, struct fi_resource *ress, int nress)
{
	int i;
	struct psmx_fid_ep *fid_ep;
	struct psmx_fid_domain *domain;
	struct psmx_fid_av *av;
	struct psmx_fid_ec *ec;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	for (i=0; i<nress; i++) {
		if (!ress[i].fid)
			return -EINVAL;
		switch (ress[i].fid->fclass) {
		case FID_CLASS_DOMAIN:
			domain = container_of(ress[i].fid,
					struct psmx_fid_domain, domain.fid);
			if (fid_ep->domain && fid_ep->domain != domain)
				return -EEXIST;
			fid_ep->domain = domain;
			break;

		case FID_CLASS_EC:
			/* TODO: check ress flags for send/recv EQs */
			ec = container_of(ress[i].fid,
					struct psmx_fid_ec, ec.fid);
			if (fid_ep->ec && fid_ep->ec != ec)
				return -EEXIST;
			if (fid_ep->domain && fid_ep->domain != ec->domain)
				return -EINVAL;
			fid_ep->ec = ec;
			fid_ep->domain = ec->domain;
			break;

		case FID_CLASS_AV:
			av = container_of(ress[i].fid,
					struct psmx_fid_av, av.fid);
			if (fid_ep->av && fid_ep->av != av)
				return -EEXIST;
			if (fid_ep->domain && fid_ep->domain != av->domain)
				return -EINVAL;
			fid_ep->av = av;
			fid_ep->domain = av->domain;
			break;

		default:
			return -ENOSYS;
		}
	}

	return 0;
}

static int psmx_ep_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_ep_control(fid_t fid, int command, void *arg)
{
	struct psmx_fid_ep *fid_ep, *new_fid_ep;
	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	switch (command) {
	case FI_ALIAS:
		new_fid_ep = (struct psmx_fid_ep *) calloc(1, sizeof *fid_ep);
		if (!new_fid_ep)
			return -ENOMEM;
		*new_fid_ep = *fid_ep;
		*(fid_t *)arg = &new_fid_ep->ep.fid;
		break;

	case FI_SETFIDFLAG:
		fid_ep->flags = *(uint64_t *)arg;
		break;

	case FI_GETFIDFLAG:
		if (!arg)
			return -EINVAL;
		*(uint64_t *)arg = fid_ep->flags;
		break;

	default:
		return -ENOSYS;
	}

	return 0;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_ep_close,
	.bind = psmx_ep_bind,
	.sync = psmx_ep_sync,
	.control = psmx_ep_control,
};

static struct fi_ops_ep psmx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = psmx_ep_cancel,
	.getopt = psmx_ep_getopt,
	.setopt = psmx_ep_setopt,
};

int psmx_ep_open(fid_t domain, struct fi_info *info, fid_t *fid, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = (struct psmx_fid_ep *) calloc(1, sizeof *fid_ep);
	if (!fid_ep)
		return -ENOMEM;

	fid_ep->ep.fid.size = sizeof(struct fid_ep);
	fid_ep->ep.fid.fclass = FID_CLASS_EP;
	fid_ep->ep.fid.context = context;
	fid_ep->ep.fid.ops = &psmx_fi_ops;
	fid_ep->ep.ops = &psmx_ep_ops;
	fid_ep->ep.cm = &psmx_cm_ops;
	fid_ep->ep.tagged = &psmx_tagged_ops;

	if (info) {
		fid_ep->flags = info->flags;
		if (info->protocol_cap & FI_PROTO_CAP_MSG)
			fid_ep->ep.msg = &psmx_msg_ops;
		if (info->protocol_cap & FI_PROTO_CAP_RMA)
			fid_ep->ep.rma = &psmx_rma_ops;
	}

	*fid = &fid_ep->ep.fid;

	return 0;
}

