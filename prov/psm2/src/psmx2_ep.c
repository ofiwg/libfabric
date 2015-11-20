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

#define BIT(i)		(1ULL << i)
#define BITMAP_SIZE	(PSMX2_MAX_VL + 1)

static void inline bitmap_set(uint64_t *map, unsigned id)
{
	int i, j;

	i = id / sizeof(uint64_t);
	j = id % sizeof(uint64_t);
	
	map[i] |= BIT(j);
}

static void inline bitmap_clear(uint64_t *map, unsigned id)
{
	int i, j;

	i = id / sizeof(uint64_t);
	j = id % sizeof(uint64_t);
	
	map[i] &= ~BIT(j);
}

static int inline bitmap_test(uint64_t *map, unsigned id)
{
	int i, j;

	i = id / sizeof(uint64_t);
	j = id % sizeof(uint64_t);
	
	return !!(map[i] & BIT(j));
}

static void psmx2_free_vlane(struct psmx2_fid_domain *domain, uint8_t vl)
{
	fastlock_acquire(&domain->vl_lock);
	bitmap_clear(domain->vl_map, vl);
	fastlock_release(&domain->vl_lock);
}

static int psmx2_alloc_vlane(struct psmx2_fid_domain *domain, uint8_t *vl)
{
	int i;
	int id;

	fastlock_acquire(&domain->vl_lock);
	for (i=0; i<BITMAP_SIZE; i++) {
		id = (domain->vl_alloc + i) % BITMAP_SIZE;
		if (bitmap_test(domain->vl_map, id) == 0) {
			bitmap_set(domain->vl_map, id);
			domain->vl_alloc = id + 1;
			break;
		}
	}
	fastlock_release(&domain->vl_lock);

	if (i >= BITMAP_SIZE)
		return -FI_ENOSPC;

	*vl = (uint8_t)id;
	return 0;
}

static void psmx2_ep_optimize_ops(struct psmx2_fid_ep *ep)
{
	if (ep->ep.tagged) {
		if (ep->flags) {
			ep->ep.tagged = &psmx2_tagged_ops;
			FI_INFO(&psmx2_prov, FI_LOG_EP_DATA,
				"generic tagged ops.\n");
		}
		else if (ep->send_selective_completion && ep->recv_selective_completion) {
			if (ep->av && ep->av->type == FI_AV_TABLE)
				ep->ep.tagged = &psmx2_tagged_ops_no_event_av_table;
			else
				ep->ep.tagged = &psmx2_tagged_ops_no_event_av_map;
			FI_INFO(&psmx2_prov, FI_LOG_EP_DATA,
				"tagged ops optimized for op_flags=0 and event suppression\n");
		}
		else if (ep->send_selective_completion) {
			if (ep->av && ep->av->type == FI_AV_TABLE)
				ep->ep.tagged = &psmx2_tagged_ops_no_send_event_av_table;
			else
				ep->ep.tagged = &psmx2_tagged_ops_no_send_event_av_map;
			FI_INFO(&psmx2_prov, FI_LOG_EP_DATA,
				"tagged ops optimized for op_flags=0 and send event suppression\n");
		}
		else if (ep->recv_selective_completion) {
			if (ep->av && ep->av->type == FI_AV_TABLE)
				ep->ep.tagged = &psmx2_tagged_ops_no_recv_event_av_table;
			else
				ep->ep.tagged = &psmx2_tagged_ops_no_recv_event_av_map;
			FI_INFO(&psmx2_prov, FI_LOG_EP_DATA,
				"tagged ops optimized for op_flags=0 and recv event suppression\n");
		}
		else {
			if (ep->av && ep->av->type == FI_AV_TABLE)
				ep->ep.tagged = &psmx2_tagged_ops_no_flag_av_table;
			else
				ep->ep.tagged = &psmx2_tagged_ops_no_flag_av_map;
			FI_INFO(&psmx2_prov, FI_LOG_EP_DATA,
				"tagged ops optimized for op_flags=0\n");
		}
	}
}

static ssize_t psmx2_ep_cancel(fid_t fid, void *context)
{
	struct psmx2_fid_ep *ep;
	psm2_mq_status2_t status;
	struct fi_context *fi_context = context;
	uint64_t flags;
	struct psmx2_cq_event *event;
	int err;

	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);
	if (!ep->domain)
		return -FI_EBADF;

	if (!fi_context)
		return -FI_EINVAL;

	switch (PSMX2_CTXT_TYPE(fi_context)) {
	case PSMX2_TRECV_CONTEXT:
		flags = FI_RECV | FI_TAGGED;
		break;
	case PSMX2_RECV_CONTEXT:
	case PSMX2_MULTI_RECV_CONTEXT:
		flags = FI_RECV | FI_MSG;
		break;
	default:
		return  -FI_EOPNOTSUPP;
	}

	err = psm2_mq_cancel((psm2_mq_req_t *)&PSMX2_CTXT_REQ(fi_context));
	if (err == PSM2_OK) {
		err = psm2_mq_test2((psm2_mq_req_t *)&PSMX2_CTXT_REQ(fi_context), &status);
		if (err == PSM2_OK && ep->recv_cq) {
			event = psmx2_cq_create_event(
					ep->recv_cq,
					status.context,
					NULL,	/* buf */
					flags,
					0,	/* len */
					0,	/* data */
					0,	/* tag */
					0	/* olen */,
					-FI_ECANCELED);
			if (event)
				psmx2_cq_enqueue_event(ep->recv_cq, event);
			else
				return -FI_ENOMEM;
		}
	}

	return psmx2_errno(err);
}

static int psmx2_ep_getopt(fid_t fid, int level, int optname,
			   void *optval, size_t *optlen)
{
	struct psmx2_fid_ep *ep;

	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		*(size_t *)optval = ep->min_multi_recv;
		*optlen = sizeof(size_t);
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static int psmx2_ep_setopt(fid_t fid, int level, int optname,
			   const void *optval, size_t optlen)
{
	struct psmx2_fid_ep *ep;

	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		ep->min_multi_recv = *(size_t *)optval;
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static int psmx2_ep_close(fid_t fid)
{
	struct psmx2_fid_ep *ep;

	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);

	psmx2_domain_disable_ep(ep->domain, ep);
	ep->domain->eps[ep->vlane] = NULL;
	psmx2_free_vlane(ep->domain, ep->vlane);
	psmx2_domain_release(ep->domain);
	free(ep);

	return 0;
}

static int psmx2_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct psmx2_fid_ep *ep;
	struct psmx2_fid_av *av;
	struct psmx2_fid_cq *cq;
	struct psmx2_fid_cntr *cntr;
	struct psmx2_fid_stx *stx;
	int err;

	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);

	if (!bfid)
		return -FI_EINVAL;
	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		return -FI_ENOSYS;

	case FI_CLASS_CQ:
		cq = container_of(bfid, struct psmx2_fid_cq, cq.fid);
		if (ep->domain != cq->domain)
			return -FI_EINVAL;
		if (flags & FI_SEND) {
			ep->send_cq = cq;
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->send_selective_completion = 1;
		}
		if (flags & FI_RECV) {
			ep->recv_cq = cq;
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->recv_selective_completion = 1;
		}
		psmx2_ep_optimize_ops(ep);
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct psmx2_fid_cntr, cntr.fid);
		if (ep->domain != cntr->domain)
			return -FI_EINVAL;
		if (flags & FI_SEND)
			ep->send_cntr = cntr;
		if (flags & FI_RECV)
			ep->recv_cntr = cntr;
		if (flags & FI_WRITE)
			ep->write_cntr = cntr;
		if (flags & FI_READ)
			ep->read_cntr = cntr;
		if (flags & FI_REMOTE_WRITE)
			ep->remote_write_cntr = cntr;
		if (flags & FI_REMOTE_READ)
			ep->remote_read_cntr = cntr;
		break;

	case FI_CLASS_AV:
		av = container_of(bfid,
				struct psmx2_fid_av, av.fid);
		if (ep->domain != av->domain)
			return -FI_EINVAL;
		ep->av = av;
		psmx2_ep_optimize_ops(ep);
		break;

	case FI_CLASS_MR:
		if (!bfid->ops || !bfid->ops->bind)
			return -FI_EINVAL;
		err = bfid->ops->bind(bfid, fid, flags);
		if (err)
			return err;
		break;

	case FI_CLASS_STX_CTX:
		stx = container_of(bfid,
				   struct psmx2_fid_stx, stx.fid);
		if (ep->domain != stx->domain)
			return -FI_EINVAL;
		break;

	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static int psmx2_ep_control(fid_t fid, int command, void *arg)
{
	struct fi_alias *alias;
	struct psmx2_fid_ep *ep, *new_ep;
	ep = container_of(fid, struct psmx2_fid_ep, ep.fid);

	switch (command) {
	case FI_ALIAS:
		new_ep = (struct psmx2_fid_ep *) calloc(1, sizeof *ep);
		if (!new_ep)
			return -FI_ENOMEM;
		alias = arg;
		*new_ep = *ep;
		new_ep->flags = alias->flags;
		psmx2_ep_optimize_ops(new_ep);
		*alias->fid = &new_ep->ep.fid;
		break;

	case FI_SETFIDFLAG:
		ep->flags = *(uint64_t *)arg;
		psmx2_ep_optimize_ops(ep);
		break;

	case FI_GETFIDFLAG:
		if (!arg)
			return -FI_EINVAL;
		*(uint64_t *)arg = ep->flags;
		break;

	case FI_ENABLE:
		return 0;

	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static ssize_t psmx2_rx_size_left(struct fid_ep *ep)
{
	return 0x7fffffff; /* a random choice */
}

static ssize_t psmx2_tx_size_left(struct fid_ep *ep)
{
	return 0x7fffffff; /* a random choice */
}

static struct fi_ops psmx2_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_ep_close,
	.bind = psmx2_ep_bind,
	.control = psmx2_ep_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_ep psmx2_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = psmx2_ep_cancel,
	.getopt = psmx2_ep_getopt,
	.setopt = psmx2_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = psmx2_rx_size_left,
	.tx_size_left = psmx2_tx_size_left,
};

int psmx2_ep_open(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context)
{
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_fid_ep *ep_priv;
	uint8_t vlane;
	uint64_t ep_cap;
	int err = -FI_EINVAL;

	if (info)
		ep_cap = info->caps;
	else
		ep_cap = FI_TAGGED;

	domain_priv = container_of(domain, struct psmx2_fid_domain, domain.fid);
	if (!domain_priv)
		goto errout;

	err = psmx2_domain_check_features(domain_priv, ep_cap);
	if (err)
		goto errout;

	err = psmx2_alloc_vlane(domain_priv, &vlane);
	if (err)
		goto errout;

	ep_priv = (struct psmx2_fid_ep *) calloc(1, sizeof *ep_priv);
	if (!ep_priv) {
		err = -FI_ENOMEM;
		goto errout_free_vlane;
	}

	ep_priv->ep.fid.fclass = FI_CLASS_EP;
	ep_priv->ep.fid.context = context;
	ep_priv->ep.fid.ops = &psmx2_fi_ops;
	ep_priv->ep.ops = &psmx2_ep_ops;
	ep_priv->ep.cm = &psmx2_cm_ops;
	ep_priv->domain = domain_priv;
	ep_priv->vlane = vlane;

	PSMX2_CTXT_TYPE(&ep_priv->nocomp_send_context) = PSMX2_NOCOMP_SEND_CONTEXT;
	PSMX2_CTXT_EP(&ep_priv->nocomp_send_context) = ep_priv;
	PSMX2_CTXT_TYPE(&ep_priv->nocomp_recv_context) = PSMX2_NOCOMP_RECV_CONTEXT;
	PSMX2_CTXT_EP(&ep_priv->nocomp_recv_context) = ep_priv;

	if (ep_cap & FI_TAGGED)
		ep_priv->ep.tagged = &psmx2_tagged_ops;
	if (ep_cap & FI_MSG)
		ep_priv->ep.msg = &psmx2_msg_ops;
	if ((ep_cap & FI_MSG) && psmx2_env.am_msg)
		ep_priv->ep.msg = &psmx2_msg2_ops;
	if (ep_cap & FI_RMA)
		ep_priv->ep.rma = &psmx2_rma_ops;
	if (ep_cap & FI_ATOMICS)
		ep_priv->ep.atomic = &psmx2_atomic_ops;

	ep_priv->caps = ep_cap;

	err = psmx2_domain_enable_ep(domain_priv, ep_priv);
	if (err)
		goto errout_free_ep;

	psmx2_domain_acquire(domain_priv);
	domain_priv->eps[ep_priv->vlane] = ep_priv;

	if (info) {
		if (info->tx_attr)
			ep_priv->flags = info->tx_attr->op_flags;
		if (info->rx_attr)
			ep_priv->flags |= info->rx_attr->op_flags;
	}

	psmx2_ep_optimize_ops(ep_priv);

	*ep = &ep_priv->ep;

	return 0;

errout_free_ep:
	free(ep_priv);

errout_free_vlane:
	psmx2_free_vlane(domain_priv, vlane);

errout:
	return err;
}

/* STX support is essentially no-op since PSM supports only one send/recv
 * context and thus always works in shared context mode.
 */

static int psmx2_stx_close(fid_t fid)
{
	struct psmx2_fid_stx *stx;

	stx = container_of(fid, struct psmx2_fid_stx, stx.fid);
	psmx2_domain_release(stx->domain);
	free(stx);

	return 0;
}

static struct fi_ops psmx2_fi_ops_stx = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_stx_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
};

int psmx2_stx_ctx(struct fid_domain *domain, struct fi_tx_attr *attr,
		 struct fid_stx **stx, void *context)
{
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_fid_stx *stx_priv;

	FI_INFO(&psmx2_prov, FI_LOG_EP_DATA, "\n");

	domain_priv = container_of(domain, struct psmx2_fid_domain, domain.fid);

	stx_priv = (struct psmx2_fid_stx *) calloc(1, sizeof *stx_priv);
	if (!stx_priv)
		return -FI_ENOMEM;

	psmx2_domain_acquire(domain_priv);

	stx_priv->stx.fid.fclass = FI_CLASS_STX_CTX;
	stx_priv->stx.fid.context = context;
	stx_priv->stx.fid.ops = &psmx2_fi_ops_stx;
	stx_priv->domain = domain_priv;

	*stx = &stx_priv->stx;
	return 0;
}

