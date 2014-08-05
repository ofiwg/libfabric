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

static ssize_t psmx_ep_cancel(fid_t fid, void *context)
{
	struct psmx_fid_ep *fid_ep;
	psm_mq_status_t status;
	struct fi_context *fi_context = context;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	if (!fid_ep->domain)
		return -EBADF;

	if (!fi_context)
		return -EINVAL;

	err = psm_mq_cancel((psm_mq_req_t *)&PSMX_CTXT_REQ(fi_context));
	if (err == PSM_OK)
		err = psm_mq_test((psm_mq_req_t *)&PSMX_CTXT_REQ(fi_context), &status);

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
	case FI_OPT_MAX_INJECTED_SEND:
		if (!optval)
			return 0;

		if (!optlen || *optlen < sizeof(size_t))
			return -EINVAL;

		if (!fid_ep->domain)
			return -EBADF;

		*(size_t *)optval = PSMX_INJECT_SIZE;
		*optlen = sizeof(size_t);
		break;

	case FI_OPT_MAX_MSG_SIZE:
		if (!optval)
			return 0;

		if (!optlen || *optlen < sizeof(size_t))
			return -EINVAL;

		if (!fid_ep->domain)
			return -EBADF;

		*(size_t *)optval = PSMX_MAX_MSG_SIZE;
		*optlen = sizeof(size_t);
		break;

	case FI_OPT_TOTAL_BUFFERED_RECV:
		if (!optval)
			return 0;

		if (!optlen || *optlen < sizeof(size_t))
			return -EINVAL;

		if (!fid_ep->domain)
			return -EBADF;

		err = psm_mq_getopt(fid_ep->domain->psm_mq, PSM_MQ_MAX_SYSBUF_MBYTES, &size);
		if (err)
			return psmx_errno(err);

		*(size_t *)optval = size * 1048576;
		*optlen = sizeof(size_t);
		break;

		break;

	default:
		return -ENOPROTOOPT;
	}

	return 0;
}

static int psmx_ep_setopt(fid_t fid, int level, int optname,
			const void *optval, size_t optlen)
{
	return -ENOPROTOOPT;
}

static int psmx_ep_enable(struct fid_ep *ep)
{
	return 0;
}

static int psmx_ep_close(fid_t fid)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	free(fid_ep);

	return 0;
}

static int psmx_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct psmx_fid_ep *fid_ep;
	struct psmx_fid_domain *domain;
	struct psmx_fid_av *av;
	struct psmx_fid_eq *eq;
	struct psmx_fid_cntr *cntr;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	if (!bfid)
		return -EINVAL;
	switch (bfid->fclass) {
	case FID_CLASS_DOMAIN:
		domain = container_of(bfid, struct psmx_fid_domain, domain.fid);
		if (fid_ep->domain && fid_ep->domain != domain)
			return -EEXIST;
		fid_ep->domain = domain;
		break;

	case FID_CLASS_EQ:
		eq = container_of(bfid, struct psmx_fid_eq, eq.fid);
		if (flags & (FI_SEND | FI_READ | FI_WRITE)) {
			if (fid_ep->send_eq && fid_ep->send_eq != eq)
				return -EEXIST;
		}
		if (flags & FI_RECV) {
			if (fid_ep->recv_eq && fid_ep->recv_eq != eq)
				return -EEXIST;
		}
		if (fid_ep->domain && fid_ep->domain != eq->domain)
			return -EINVAL;
		if (flags & (FI_SEND | FI_READ | FI_WRITE)) {
			fid_ep->send_eq = eq;
			if (flags & FI_EVENT)
				fid_ep->send_eq_event_flag = 1;
		}
		if (flags & FI_RECV) {
			fid_ep->recv_eq = eq;
			if (flags & FI_EVENT)
				fid_ep->recv_eq_event_flag = 1;
		}
		fid_ep->domain = eq->domain;
		break;

	case FID_CLASS_CNTR:
		cntr = container_of(bfid, struct psmx_fid_cntr, cntr.fid);
		if (flags & (FI_SEND)) {
			if (fid_ep->send_cntr && fid_ep->send_cntr != cntr)
				return -EEXIST;
		}
		if (flags & FI_RECV) {
			if (fid_ep->recv_cntr && fid_ep->recv_cntr != cntr)
				return -EEXIST;
		}
		if (flags & FI_WRITE) {
			if (fid_ep->write_cntr && fid_ep->write_cntr != cntr)
				return -EEXIST;
		}
		if (flags & FI_READ) {
			if (fid_ep->read_cntr && fid_ep->read_cntr != cntr)
				return -EEXIST;
		}
		if (fid_ep->domain && fid_ep->domain != cntr->domain)
			return -EINVAL;
		if (flags & FI_SEND) {
			fid_ep->send_cntr = cntr;
			if (flags & FI_EVENT)
				fid_ep->send_cntr_event_flag = 1;
		}
		if (flags & FI_RECV){
			fid_ep->recv_cntr = cntr;
			if (flags & FI_EVENT)
				fid_ep->recv_cntr_event_flag = 1;
		}
		if (flags & FI_WRITE) {
			fid_ep->write_cntr = cntr;
			if (flags & FI_EVENT)
				fid_ep->write_cntr_event_flag = 1;
		}
		if (flags & FI_READ){
			fid_ep->read_cntr = cntr;
			if (flags & FI_EVENT)
				fid_ep->read_cntr_event_flag = 1;
		}
		fid_ep->domain = cntr->domain;
		break;

	case FID_CLASS_AV:
		av = container_of(bfid,
				struct psmx_fid_av, av.fid);
		if (fid_ep->av && fid_ep->av != av)
			return -EEXIST;
		if (fid_ep->domain && fid_ep->domain != av->domain)
			return -EINVAL;
		fid_ep->av = av;
		fid_ep->domain = av->domain;
		break;

	case FID_CLASS_MR:
		if (!bfid->ops || !bfid->ops->bind)
			return -EINVAL;
		err = bfid->ops->bind(bfid, fid, flags);
		if (err)
			return err;
		break;

	default:
		return -ENOSYS;
	}

	return 0;
}

static inline int psmx_ep_progress(struct psmx_fid_ep *fid_ep)
{
	return psmx_eq_poll_mq(NULL, fid_ep->domain);
}

static int psmx_ep_sync(fid_t fid, uint64_t flags, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	if (!flags || (flags & FI_SEND)) {
		while (fid_ep->pending_sends)
			psmx_ep_progress(fid_ep);
	}

	if (!flags || (flags & FI_WRITE)) {
		while (fid_ep->pending_writes)
			psmx_ep_progress(fid_ep);
	}

	if (!flags || (flags & FI_READ)) {
		while (fid_ep->pending_reads)
			psmx_ep_progress(fid_ep);
	}

	if (!flags) {
		while (fid_ep->pending_atomics)
			psmx_ep_progress(fid_ep);
	}

	return 0;
}

static int psmx_ep_control(fid_t fid, int command, void *arg)
{
	struct fi_alias *alias;
	struct psmx_fid_ep *fid_ep, *new_fid_ep;
	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);

	switch (command) {
	case FI_ALIAS:
		new_fid_ep = (struct psmx_fid_ep *) calloc(1, sizeof *fid_ep);
		if (!new_fid_ep)
			return -ENOMEM;
		alias = arg;
		*new_fid_ep = *fid_ep;
		new_fid_ep->flags = alias->flags;
		*alias->fid = &new_fid_ep->ep.fid;
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
	.close = psmx_ep_close,
	.bind = psmx_ep_bind,
	.sync = psmx_ep_sync,
	.control = psmx_ep_control,
};

static struct fi_ops_ep psmx_ep_ops = {
	.cancel = psmx_ep_cancel,
	.getopt = psmx_ep_getopt,
	.setopt = psmx_ep_setopt,
	.enable = psmx_ep_enable,
};

int psmx_ep_open(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = (struct psmx_fid_ep *) calloc(1, sizeof *fid_ep);
	if (!fid_ep)
		return -ENOMEM;

	fid_ep->ep.fid.fclass = FID_CLASS_EP;
	fid_ep->ep.fid.context = context;
	fid_ep->ep.fid.ops = &psmx_fi_ops;
	fid_ep->ep.ops = &psmx_ep_ops;
	fid_ep->ep.cm = &psmx_cm_ops;
	fid_ep->ep.tagged = &psmx_tagged_ops;
	PSMX_CTXT_TYPE(&fid_ep->nocomp_send_context) = PSMX_NOCOMP_SEND_CONTEXT;
	PSMX_CTXT_EP(&fid_ep->nocomp_send_context) = fid_ep;
	PSMX_CTXT_TYPE(&fid_ep->nocomp_recv_context) = PSMX_NOCOMP_RECV_CONTEXT;
	PSMX_CTXT_EP(&fid_ep->nocomp_recv_context) = fid_ep;
	PSMX_CTXT_TYPE(&fid_ep->sendimm_context) = PSMX_INJECT_CONTEXT;
	PSMX_CTXT_EP(&fid_ep->sendimm_context) = fid_ep;
	PSMX_CTXT_TYPE(&fid_ep->writeimm_context) = PSMX_INJECT_WRITE_CONTEXT;
	PSMX_CTXT_EP(&fid_ep->writeimm_context) = fid_ep;

	if (info) {
		fid_ep->flags = info->op_flags;
		if (info->ep_cap & FI_MSG) {
			fid_ep->ep.msg = &psmx_msg_ops;
		}
	}

	*ep = &fid_ep->ep;

	return 0;
}

