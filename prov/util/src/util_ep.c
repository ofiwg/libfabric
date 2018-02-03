/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>

#include <ofi_enosys.h>
#include <ofi_util.h>

int ofi_ep_bind_cq(struct util_ep *ep, struct util_cq *cq, uint64_t flags)
{
	int ret;

	ret = ofi_check_bind_cq_flags(ep, cq, flags);
	if (ret)
		return ret;

	if (flags & FI_TRANSMIT) {
		ep->tx_cq = cq;
		if (!(flags & FI_SELECTIVE_COMPLETION))
			ep->tx_op_flags |= FI_COMPLETION;
		ofi_atomic_inc32(&cq->ref);
	}

	if (flags & FI_RECV) {
		ep->rx_cq = cq;
		if (!(flags & FI_SELECTIVE_COMPLETION))
			ep->rx_op_flags |= FI_COMPLETION;
		ofi_atomic_inc32(&cq->ref);
	}

	if (flags & (FI_TRANSMIT | FI_RECV)) {
		return fid_list_insert(&cq->ep_list,
				       &cq->ep_list_lock,
				       &ep->ep_fid.fid);
	}

	return FI_SUCCESS;
}

int ofi_ep_bind_eq(struct util_ep *ep, struct util_eq *eq)
{
	if (ep->eq)
		ofi_atomic_dec32(&ep->eq->ref);

	ep->eq = eq;
	ofi_atomic_inc32(&eq->ref);
	return 0;
}

int ofi_ep_bind_av(struct util_ep *util_ep, struct util_av *av)
{
	if (util_ep->av) {
		FI_WARN(util_ep->av->prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
		return -FI_EINVAL;
	}
	util_ep->av = av;
	ofi_atomic_inc32(&av->ref);

	fastlock_acquire(&av->lock);
	dlist_insert_tail(&util_ep->av_entry, &av->ep_list);
	fastlock_release(&av->lock);

	return 0;
}

int ofi_ep_bind_cntr(struct util_ep *ep, struct util_cntr *cntr, uint64_t flags)
{
	if (flags & ~(FI_TRANSMIT | FI_RECV | FI_READ  | FI_WRITE |
		      FI_REMOTE_READ | FI_REMOTE_WRITE)) {
		FI_WARN(ep->domain->fabric->prov, FI_LOG_EP_CTRL,
			"Unsupported bind flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && ep->tx_cntr) ||
	    ((flags & FI_RECV) && ep->rx_cntr) ||
	    ((flags & FI_READ) && ep->rd_cntr) ||
	    ((flags & FI_WRITE) && ep->wr_cntr) ||
	    ((flags & FI_REMOTE_READ) && ep->rem_rd_cntr) ||
	    ((flags & FI_REMOTE_WRITE) && ep->rem_wr_cntr)) {
		FI_WARN(ep->domain->fabric->prov, FI_LOG_EP_CTRL,
			"Duplicate counter binding\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		ep->tx_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	if (flags & FI_RECV) {
		ep->rx_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	if (flags & FI_READ) {
		ep->rd_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	if (flags & FI_WRITE) {
		ep->wr_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	if (flags & FI_REMOTE_READ) {
		ep->rem_rd_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	if (flags & FI_REMOTE_WRITE) {
		ep->rem_wr_cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
	}

	ep->flags |= OFI_CNTR_ENABLED;

	return fid_list_insert(&cntr->ep_list, &cntr->ep_list_lock,
			       &ep->ep_fid.fid);
}

int ofi_ep_bind(struct util_ep *util_ep, struct fid *fid, uint64_t flags)
{
	int ret;
	struct util_av   *av;
	struct util_cq   *cq;
	struct util_eq   *eq;
	struct util_cntr *cntr;

	ret = ofi_ep_bind_valid(util_ep->domain->prov, fid, flags);
	if (ret)
		return ret;

	switch (fid->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(fid, struct util_cq, cq_fid.fid);
			return ofi_ep_bind_cq(util_ep, cq, flags);
		case FI_CLASS_EQ:
			eq = container_of(fid, struct util_eq, eq_fid.fid);
			return ofi_ep_bind_eq(util_ep, eq);
		case FI_CLASS_AV:
			av = container_of(fid, struct util_av, av_fid.fid);
			return ofi_ep_bind_av(util_ep, av);
		case FI_CLASS_CNTR:
			cntr = container_of(fid, struct util_cntr, cntr_fid.fid);
			return ofi_ep_bind_cntr(util_ep, cntr, flags);
	}

	return -FI_EINVAL;
}

int ofi_endpoint_init(struct fid_domain *domain, const struct util_prov *util_prov,
		      struct fi_info *info, struct util_ep *ep, void *context,
		      ofi_ep_progress_func progress)
{
	struct util_domain *util_domain;
	int ret;

	util_domain = container_of(domain, struct util_domain, domain_fid);

	if (!info || !info->ep_attr || !info->rx_attr || !info->tx_attr)
		return -FI_EINVAL;

	ret = ofi_prov_check_info(util_prov,
				  util_domain->fabric->fabric_fid.api_version,
				  info);
	if (ret)
		return ret;

	ep->ep_fid.fid.fclass = FI_CLASS_EP;
	ep->ep_fid.fid.context = context;
	ep->domain = util_domain;
	ep->caps = info->caps;
	ep->flags = 0;
	ep->progress = progress;
	ep->tx_op_flags = info->tx_attr->op_flags;
	ep->rx_op_flags = info->rx_attr->op_flags;
	ofi_atomic_inc32(&util_domain->ref);
	if (util_domain->eq)
		ofi_ep_bind_eq(ep, util_domain->eq);
	fastlock_init(&ep->lock);
	return 0;
}

int ofi_endpoint_close(struct util_ep *util_ep)
{
	fastlock_destroy(&util_ep->lock);

	if (util_ep->tx_cq) {
		fid_list_remove(&util_ep->tx_cq->ep_list,
				&util_ep->tx_cq->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->tx_cq->ref);
	}

	if (util_ep->rx_cq) {
		fid_list_remove(&util_ep->rx_cq->ep_list,
				&util_ep->rx_cq->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->rx_cq->ref);
	}

	if (util_ep->rx_cntr) {
		fid_list_remove(&util_ep->rx_cntr->ep_list,
				&util_ep->rx_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->rx_cntr->ref);
	}

	if (util_ep->tx_cntr) {
		fid_list_remove(&util_ep->tx_cntr->ep_list,
				&util_ep->tx_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->tx_cntr->ref);
	}

	if (util_ep->rd_cntr) {
		fid_list_remove(&util_ep->rd_cntr->ep_list,
				&util_ep->rd_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->rd_cntr->ref);
	}

	if (util_ep->wr_cntr) {
		fid_list_remove(&util_ep->wr_cntr->ep_list,
				&util_ep->wr_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->wr_cntr->ref);
	}

	if (util_ep->rem_rd_cntr) {
		fid_list_remove(&util_ep->rem_rd_cntr->ep_list,
				&util_ep->rem_rd_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->rem_rd_cntr->ref);
	}

	if (util_ep->rem_wr_cntr) {
		fid_list_remove(&util_ep->rem_wr_cntr->ep_list,
				&util_ep->rem_wr_cntr->ep_list_lock,
				&util_ep->ep_fid.fid);
		ofi_atomic_dec32(&util_ep->rem_wr_cntr->ref);
	}

	if (util_ep->av) {
		fastlock_acquire(&util_ep->av->lock);
		dlist_remove(&util_ep->av_entry);
		fastlock_release(&util_ep->av->lock);

		ofi_atomic_dec32(&util_ep->av->ref);
	}

	if (util_ep->eq)
		ofi_atomic_dec32(&util_ep->eq->ref);
	ofi_atomic_dec32(&util_ep->domain->ref);
	return 0;
}
