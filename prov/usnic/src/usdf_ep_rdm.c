/*
 * Copyright (c) 2014-2018, Cisco Systems, Inc. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "config.h"

#include <asm/types.h>
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/eventfd.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>
#include "ofi.h"
#include "ofi_enosys.h"
#include "ofi_util.h"

#include "usd.h"
#include "usdf.h"
#include "usnic_direct.h"
#include "usdf_endpoint.h"
#include "fi_ext_usnic.h"
#include "usdf_rudp.h"
#include "usdf_cq.h"
#include "usdf_cm.h"
#include "usdf_av.h"
#include "usdf_timer.h"
#include "usdf_rdm.h"


/*******************************************************************************
 * Default values for rdm attributes
 ******************************************************************************/
static const struct fi_tx_attr rdm_dflt_tx_attr = {
	.caps = USDF_RDM_CAPS,
	.mode = USDF_RDM_SUPP_MODE,
	.size = USDF_RDM_DFLT_CTX_SIZE,
	.op_flags = 0,
	.msg_order = USDF_RDM_MSG_ORDER,
	.comp_order = USDF_RDM_COMP_ORDER,
	.inject_size = USDF_RDM_MAX_INJECT_SIZE,
	.iov_limit = USDF_RDM_IOV_LIMIT,
	.rma_iov_limit = USDF_RDM_RMA_IOV_LIMIT
};

static const struct fi_rx_attr rdm_dflt_rx_attr = {
	.caps = USDF_RDM_CAPS,
	.mode = USDF_RDM_SUPP_MODE,
	.size = USDF_RDM_DFLT_CTX_SIZE,
	.op_flags = 0,
	.msg_order = USDF_RDM_MSG_ORDER,
	.comp_order = USDF_RDM_COMP_ORDER,
	.total_buffered_recv = 0,
	.iov_limit = USDF_RDM_DFLT_SGE
};

/* The protocol for RDM is still under development. Version 0 does not provide
 * any interoperability.
 */
static const struct fi_ep_attr rdm_dflt_ep_attr = {
	.type = FI_EP_RDM,
	.protocol = FI_PROTO_RUDP,
	.protocol_version = 0,
	.max_msg_size = USDF_RDM_MAX_MSG,
	.msg_prefix_size = 0,
	.max_order_raw_size = 0,
	.max_order_war_size = 0,
	.max_order_waw_size = 0,
	.mem_tag_format = 0,
	.tx_ctx_cnt = 1,
	.rx_ctx_cnt = 1
};

static const struct fi_domain_attr rdm_dflt_domain_attr = {
	.caps = USDF_DOM_CAPS,
	.threading = FI_THREAD_ENDPOINT,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_MANUAL,
	.resource_mgmt = FI_RM_DISABLED,
	.mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL | FI_MR_BASIC,
	.cntr_cnt = USDF_RDM_CNTR_CNT,
	.mr_iov_limit = USDF_RDM_MR_IOV_LIMIT,
	.mr_cnt = USDF_RDM_MR_CNT,
};

static struct fi_ops_atomic usdf_rdm_atomic_ops = {
	.size = sizeof(struct fi_ops_atomic),
	.write = fi_no_atomic_write,
	.writev = fi_no_atomic_writev,
	.writemsg = fi_no_atomic_writemsg,
	.inject = fi_no_atomic_inject,
	.readwrite = fi_no_atomic_readwrite,
	.readwritev = fi_no_atomic_readwritev,
	.readwritemsg = fi_no_atomic_readwritemsg,
	.compwrite = fi_no_atomic_compwrite,
	.compwritev = fi_no_atomic_compwritev,
	.compwritemsg = fi_no_atomic_compwritemsg,
	.writevalid = fi_no_atomic_writevalid,
	.readwritevalid = fi_no_atomic_readwritevalid,
	.compwritevalid = fi_no_atomic_compwritevalid,
};

/*******************************************************************************
 * Fill functions for attributes
 ******************************************************************************/
int usdf_rdm_fill_ep_attr(const struct fi_info *hints, struct fi_info *fi,
		struct usd_device_attrs *dap)
{
	struct fi_ep_attr defaults;

	defaults = rdm_dflt_ep_attr;

	if (!hints || !hints->ep_attr)
		goto out;

	if (hints->ep_attr->max_msg_size > defaults.max_msg_size)
		return -FI_ENODATA;

	switch (hints->ep_attr->protocol) {
	case FI_PROTO_UNSPEC:
	case FI_PROTO_RUDP:
		break;
	default:
		return -FI_ENODATA;
	}

	if (hints->ep_attr->tx_ctx_cnt > defaults.tx_ctx_cnt)
		return -FI_ENODATA;

	if (hints->ep_attr->rx_ctx_cnt > defaults.rx_ctx_cnt)
		return -FI_ENODATA;

	if (hints->ep_attr->max_order_raw_size > defaults.max_order_raw_size)
		return -FI_ENODATA;

	if (hints->ep_attr->max_order_war_size > defaults.max_order_war_size)
		return -FI_ENODATA;

	if (hints->ep_attr->max_order_waw_size > defaults.max_order_waw_size)
		return -FI_ENODATA;

out:
	*fi->ep_attr = defaults;

	return FI_SUCCESS;

}

int usdf_rdm_fill_dom_attr(uint32_t version, const struct fi_info *hints,
			   struct fi_info *fi, struct usd_device_attrs *dap)
{
	int ret;
	struct fi_domain_attr defaults;

	defaults = rdm_dflt_domain_attr;
	ret = usdf_domain_getname(version, dap, &defaults.name);
	if (ret < 0)
		return -FI_ENODATA;

	if (!hints || !hints->domain_attr)
		goto catch;

	/* how to handle fi_thread_fid, fi_thread_completion, etc?
	 */
	switch (hints->domain_attr->threading) {
	case FI_THREAD_UNSPEC:
	case FI_THREAD_ENDPOINT:
		break;
	default:
		return -FI_ENODATA;
	}

	/* how to handle fi_progress_manual?
	 */
	switch (hints->domain_attr->control_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
		break;
	default:
		return -FI_ENODATA;
	}

	switch (hints->domain_attr->data_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_MANUAL:
		break;
	default:
		return -FI_ENODATA;
	}

	switch (hints->domain_attr->resource_mgmt) {
	case FI_RM_UNSPEC:
	case FI_RM_DISABLED:
		break;
	default:
		return -FI_ENODATA;
	}

	switch (hints->domain_attr->caps) {
	case 0:
	case FI_REMOTE_COMM:
		break;
	default:
		USDF_WARN_SYS(DOMAIN,
			"invalid domain capabilities\n");
		return -FI_ENODATA;
	}

	if (ofi_check_mr_mode(&usdf_ops, version, defaults.mr_mode, hints))
		return -FI_ENODATA;

	if (hints->domain_attr->mr_cnt <= USDF_RDM_MR_CNT) {
		defaults.mr_cnt = hints->domain_attr->mr_cnt;
	} else {
		USDF_DBG_SYS(DOMAIN, "mr_count exceeded provider limit\n");
		return -FI_ENODATA;
	}

catch:
	/* catch the version changes here. */
	ret = usdf_catch_dom_attr(version, hints, &defaults);
	if (ret)
		return ret;

	*fi->domain_attr = defaults;

	return FI_SUCCESS;
}

int usdf_rdm_fill_tx_attr(uint32_t version, const struct fi_info *hints,
			  struct fi_info *fi)
{
	int ret;
	struct fi_tx_attr defaults;

	defaults = rdm_dflt_tx_attr;

	if (!hints || !hints->tx_attr)
		goto catch;

	/* make sure we can support the caps that are requested*/
	if (hints->tx_attr->caps & ~USDF_RDM_CAPS)
		return -FI_ENODATA;

	/* clear the mode bits the app doesn't support */
	if (hints->mode || hints->tx_attr->mode)
		defaults.mode &= (hints->mode | hints->tx_attr->mode);

	defaults.op_flags |= hints->tx_attr->op_flags;

	if ((hints->tx_attr->msg_order | USDF_RDM_MSG_ORDER) !=
			USDF_RDM_MSG_ORDER)
		return -FI_ENODATA;

	if ((hints->tx_attr->comp_order | USDF_RDM_COMP_ORDER) !=
			USDF_RDM_COMP_ORDER)
		return -FI_ENODATA;

	if (hints->tx_attr->inject_size > defaults.inject_size)
		return -FI_ENODATA;

	if (hints->tx_attr->iov_limit > defaults.iov_limit)
		return -FI_ENODATA;

	if (hints->tx_attr->rma_iov_limit > defaults.rma_iov_limit)
		return -FI_ENODATA;

	if (hints->tx_attr->size > defaults.size)
		return -FI_ENODATA;

catch:
	/* catch version changes here. */
	ret = usdf_catch_tx_attr(version, &defaults);
	if (ret)
		return ret;

	*fi->tx_attr = defaults;

	return FI_SUCCESS;
}

int usdf_rdm_fill_rx_attr(uint32_t version, const struct fi_info *hints,
			  struct fi_info *fi)
{
	int ret;
	struct fi_rx_attr defaults;

	defaults = rdm_dflt_rx_attr;

	if (!hints || !hints->rx_attr)
		goto catch;

	/* make sure we can support the capabilities that are requested */
	if (hints->rx_attr->caps & ~USDF_RDM_CAPS)
		return -FI_ENODATA;

	/* clear the mode bits the app doesn't support */
	if (hints->mode || hints->rx_attr->mode)
		defaults.mode &= (hints->mode | hints->rx_attr->mode);

	defaults.op_flags |= hints->rx_attr->op_flags;

	if ((hints->rx_attr->msg_order | USDF_RDM_MSG_ORDER) !=
			USDF_RDM_MSG_ORDER)
		return -FI_ENODATA;
	if ((hints->rx_attr->comp_order | USDF_RDM_COMP_ORDER) !=
			USDF_RDM_COMP_ORDER)
		return -FI_ENODATA;

	if (hints->rx_attr->total_buffered_recv >
			defaults.total_buffered_recv)
		return -FI_ENODATA;

	if (hints->rx_attr->iov_limit > defaults.iov_limit)
		return -FI_ENODATA;

	if (hints->rx_attr->size > defaults.size)
		return -FI_ENODATA;

catch:
	/* catch version changes here. */
	ret = usdf_catch_rx_attr(version, &defaults);
	if (ret)
		return ret;

	*fi->rx_attr = defaults;

	return FI_SUCCESS;
}

static int
usdf_tx_rdm_enable(struct usdf_tx *tx)
{
	struct usdf_rdm_qe *wqe;
	struct usdf_domain *udp;
	struct usdf_cq_hard *hcq;
	struct usd_filter filt;
	int ret;
	size_t i;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	udp = tx->tx_domain;

	hcq = tx->t.rdm.tx_hcq;
	if (hcq == NULL) {
		return -FI_ENOCQ;
	}

	/* XXX temp until we can allocate WQ and RQ independently */
	filt.uf_type = USD_FTY_UDP;
	filt.uf_filter.uf_udp.u_port = 0;
	ret = usd_create_qp(udp->dom_dev,
			USD_QTR_UDP,
			USD_QTY_UD,
			hcq->cqh_ucq,
			hcq->cqh_ucq,
			udp->dom_fabric->fab_dev_attrs->uda_max_send_credits,
			udp->dom_fabric->fab_dev_attrs->uda_max_recv_credits,
			&filt,
			&tx->tx_qp);
	if (ret != 0) {
		goto fail;
	}
	tx->tx_qp->uq_context = tx;

	/* rdm send queue */
	tx->t.rdm.tx_wqe_buf = malloc(tx->tx_attr.size *
			sizeof(struct usdf_rdm_qe));
	if (tx->t.rdm.tx_wqe_buf == NULL) {
		ret = -errno;
		goto fail;
	}

	ret = usd_alloc_mr(tx->tx_domain->dom_dev,
			tx->tx_attr.size * USDF_RDM_MAX_INJECT_SIZE,
			(void **)&tx->t.rdm.tx_inject_bufs);
	if (ret) {
		USDF_INFO("usd_alloc_mr failed (%s)\n", strerror(-ret));
		goto fail;
	}

	/* populate free list */
	TAILQ_INIT(&tx->t.rdm.tx_free_wqe);
	wqe = tx->t.rdm.tx_wqe_buf;
	for (i = 0; i < tx->tx_attr.size; ++i) {
		wqe->rd_inject_buf =
			&tx->t.rdm.tx_inject_bufs[USDF_RDM_MAX_INJECT_SIZE * i];
		TAILQ_INSERT_TAIL(&tx->t.rdm.tx_free_wqe, wqe, rd_link);
		++wqe;
	}
	tx->t.rdm.tx_num_free_wqe = tx->tx_attr.size;

	return 0;

fail:
	if (tx->t.rdm.tx_wqe_buf != NULL) {
		free(tx->t.rdm.tx_wqe_buf);
		tx->t.rdm.tx_wqe_buf = NULL;
		TAILQ_INIT(&tx->t.rdm.tx_free_wqe);
		tx->t.rdm.tx_num_free_wqe = 0;
	}

	if (tx->t.rdm.tx_inject_bufs != NULL) {
		usd_free_mr(tx->t.rdm.tx_inject_bufs);
		tx->t.rdm.tx_inject_bufs = NULL;
	}

	if (tx->tx_qp != NULL) {
		usd_destroy_qp(tx->tx_qp);
	}
	return ret;
}

static int
usdf_rx_rdm_enable(struct usdf_rx *rx)
{
	struct usdf_domain *udp;
	struct usdf_cq_hard *hcq;
	struct usdf_rdm_qe *rqe;
	struct usd_filter filt;
	struct usd_qp_impl *qp;
	uint8_t *ptr;
	size_t mtu;
	int ret;
	size_t i;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	udp = rx->rx_domain;

	hcq = rx->r.rdm.rx_hcq;
	if (hcq == NULL) {
		return -FI_ENOCQ;
	}

	/* XXX temp until we can allocate WQ and RQ independently */
	filt.uf_type = USD_FTY_UDP_SOCK;
	filt.uf_filter.uf_udp_sock.u_sock = rx->r.rdm.rx_sock;
	ret = usd_create_qp(udp->dom_dev,
			USD_QTR_UDP,
			USD_QTY_UD,
			hcq->cqh_ucq,
			hcq->cqh_ucq,
			udp->dom_fabric->fab_dev_attrs->uda_max_send_credits,
			udp->dom_fabric->fab_dev_attrs->uda_max_recv_credits,
			&filt,
			&rx->rx_qp);
	if (ret != 0) {
		goto fail;
	}
	rx->rx_qp->uq_context = rx;
	qp = to_qpi(rx->rx_qp);

	/* receive buffers */
	mtu = rx->rx_domain->dom_fabric->fab_dev_attrs->uda_mtu;
	ret = usd_alloc_mr(rx->rx_domain->dom_dev,
			qp->uq_rq.urq_num_entries * mtu,
			(void **)&rx->r.rdm.rx_bufs);
	if (ret != 0) {
		goto fail;
	}

	/* post all the buffers */
	ptr = rx->r.rdm.rx_bufs;
	for (i = 0; i < qp->uq_rq.urq_num_entries - 1; ++i) {
		usdf_rdm_post_recv(rx, ptr, mtu);
		ptr += mtu;
	}

	/* rdm recv queue */
	rx->r.rdm.rx_rqe_buf = malloc(rx->rx_attr.size *
			sizeof(struct usdf_rdm_qe));
	if (rx->r.rdm.rx_rqe_buf == NULL) {
		ret = -errno;
		goto fail;
	}

	/* populate free list */
	TAILQ_INIT(&rx->r.rdm.rx_free_rqe);
	rqe = rx->r.rdm.rx_rqe_buf;
	for (i = 0; i < rx->rx_attr.size; ++i) {
		TAILQ_INSERT_TAIL(&rx->r.rdm.rx_free_rqe, rqe, rd_link);
		++rqe;
	}
	rx->r.rdm.rx_num_free_rqe = rx->rx_attr.size;

	return 0;

fail:
	if (rx->r.rdm.rx_rqe_buf != NULL) {
		free(rx->r.rdm.rx_rqe_buf);
		rx->r.rdm.rx_rqe_buf = NULL;
		TAILQ_INIT(&rx->r.rdm.rx_free_rqe);
		rx->r.rdm.rx_num_free_rqe = 0;
	}
	if (rx->r.rdm.rx_bufs != NULL) {
		usd_free_mr(rx->r.rdm.rx_bufs);
		rx->r.rdm.rx_bufs = NULL;
	}
	if (rx->rx_qp != NULL) {
		usd_destroy_qp(rx->rx_qp);
	}
	return ret;
}

/*
 * Allocate any missing queue resources for this endpoint
 */
static int
usdf_ep_rdm_get_queues(struct usdf_ep *ep)
{
	struct usdf_tx *tx;
	struct usdf_rx *rx;
	int ret;

	/* Must have TX context at this point */
	tx = ep->ep_tx;
	if (tx == NULL) {
		ret = -FI_EINVAL;
		goto fail;
	}
	if (tx->tx_qp == NULL) {
		ret = usdf_tx_rdm_enable(tx);
		if (ret != 0) {
			goto fail;
		}
	}

	/* Must have RX context at this point */
	rx = ep->ep_rx;
	if (rx == NULL) {
		ret = -FI_EINVAL;
		goto fail;
	}
	if (rx->rx_qp == NULL) {
		ret = usdf_rx_rdm_enable(rx);
		if (ret != 0) {
			goto fail;
		}
	}

	return 0;
fail:
	return ret;
}

static int
usdf_ep_rdm_enable(struct fid_ep *fep)
{
	struct usdf_ep *ep;
	int ret;

	ep = ep_ftou(fep);

	ret = usdf_ep_rdm_get_queues(ep);
	if (ret == FI_SUCCESS)
		ep->flags |= USDF_EP_ENABLED;

	return ret;
}

static ssize_t
usdf_ep_rdm_cancel(fid_t fid, void *context)
{
	USDF_TRACE_SYS(EP_CTRL, "\n");
	/* XXX should this have a non-empty implementation? */
	return 0;
}

/*
 * Find a hard CQ within this soft CQ that services message EPs
 */
static struct usdf_cq_hard *
usdf_ep_rdm_find_cqh(struct usdf_cq *cq)
{
	struct usdf_cq_hard *hcq;

	TAILQ_FOREACH(hcq, &cq->c.soft.cq_list, cqh_link) {
		if (hcq->cqh_progress == usdf_rdm_hcq_progress) {
			return hcq;
		}
	}
	return NULL;
}

static int
usdf_ep_rdm_bind_cq(struct usdf_ep *ep, struct usdf_cq *cq, uint64_t flags)
{
	struct usdf_cq_hard **hcqp;
	struct usdf_cq_hard *hcq;
	int ret;

	/*
	 * The CQ is actually bound the RX or TX ctx, not the EP directly
	 */
	if (flags & FI_SEND) {
		/* if TX is shared, but bind directly */
		if (ep->ep_tx->tx_fid.fid.fclass == FI_CLASS_STX_CTX) {
			return -FI_EINVAL;
		}
		hcqp = &ep->ep_tx->t.rdm.tx_hcq;
	} else {
		/* if RX is shared, but bind directly */
		if (ep->ep_rx->rx_fid.fid.fclass == FI_CLASS_SRX_CTX) {
			return -FI_EINVAL;
		}
		hcqp = &ep->ep_rx->r.rdm.rx_hcq;
	}
	if (*hcqp != NULL) {
		return -FI_EINVAL;
	}

	/* Make sure this CQ is "soft" */
	ret = usdf_cq_make_soft(cq);
	if (ret != 0) {
		return ret;
	}

	if ((cq->cq_attr.wait_obj == FI_WAIT_FD) ||
			(cq->cq_attr.wait_obj == FI_WAIT_SET)) {
		cq->object.fd = eventfd(0, EFD_NONBLOCK);
		if (cq->object.fd == -1) {
			USDF_DBG_SYS(CQ, "creating eventfd failed: %s\n",
					strerror(errno));
			return -errno;
		}

		USDF_DBG_SYS(CQ, "successfully created eventfd: %d\n",
				cq->object.fd);
	}

	/* Use existing rdm CQ if present */
	hcq = usdf_ep_rdm_find_cqh(cq);
	if (hcq == NULL) {
		hcq = malloc(sizeof(*hcq));
		if (hcq == NULL) {
			return -errno;
		}

		ret = usdf_cq_create_cq(cq, &hcq->cqh_ucq, false);
		if (ret)
			goto fail;

		hcq->cqh_cq = cq;
		ofi_atomic_initialize32(&hcq->cqh_refcnt, 0);
		hcq->cqh_progress = usdf_rdm_hcq_progress;
		hcq->cqh_post = usdf_cq_post_soft;
		TAILQ_INSERT_TAIL(&cq->c.soft.cq_list, hcq, cqh_link);

		/* add to domain progression list */
		TAILQ_INSERT_TAIL(&ep->ep_domain->dom_hcq_list,
				hcq, cqh_dom_link);
	}
	ofi_atomic_inc32(&hcq->cqh_refcnt);
	ofi_atomic_inc32(&cq->cq_refcnt);
	*hcqp = hcq;
	return 0;

fail:
	if (hcq != NULL) {
		free(hcq);
	}
	return ret;
}

static int
usdf_ep_rdm_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int ret;
	struct usdf_ep *ep;
	struct usdf_cq *cq;
	struct usdf_av *av;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	/* Check if the binding flags are valid. */
	ret = ofi_ep_bind_valid(&usdf_ops, bfid, flags);
	if (ret)
		return ret;

	ep = ep_fidtou(fid);

	switch (bfid->fclass) {

	case FI_CLASS_AV:
		if (ep->e.rdm.ep_av != NULL) {
			return -FI_EINVAL;
		}

		av = av_fidtou(bfid);
		ep->e.rdm.ep_av = av;
		ofi_atomic_inc32(&av->av_refcnt);
		break;

	case FI_CLASS_CQ:
		if (flags & FI_SEND) {
			cq = cq_fidtou(bfid);
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->ep_tx_dflt_signal_comp = 0;
			else
				ep->ep_tx_dflt_signal_comp = 1;
			usdf_ep_rdm_bind_cq(ep, cq, FI_SEND);
		}

		if (flags & FI_RECV) {
			cq = cq_fidtou(bfid);
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->ep_rx_dflt_signal_comp = 0;
			else
				ep->ep_rx_dflt_signal_comp = 1;
			usdf_ep_rdm_bind_cq(ep, cq, FI_RECV);
		}
		break;

	case FI_CLASS_EQ:
		if (ep->ep_eq != NULL) {
			return -FI_EINVAL;
		}
		ep->ep_eq = eq_fidtou(bfid);
		ofi_atomic_inc32(&ep->ep_eq->eq_refcnt);
		break;
	default:
		return -FI_EINVAL;
	}

	return 0;
}

/*
 * XXX clean up pending transmits
 */
static int
usdf_rdm_rx_ctx_close(fid_t fid)
{
	struct usdf_rx *rx;
	struct usdf_cq_hard *hcq;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	rx = rx_fidtou(fid);

	if (ofi_atomic_get32(&rx->rx_refcnt) > 0) {
		return -FI_EBUSY;
	}

	hcq = rx->r.rdm.rx_hcq;
	if (hcq != NULL) {
		ofi_atomic_dec32(&hcq->cqh_refcnt);
		ofi_atomic_dec32(&hcq->cqh_cq->cq_refcnt);
	}
	if (rx->r.rdm.rx_sock != -1) {
		close(rx->r.rdm.rx_sock);
	}

	if (rx->rx_qp != NULL) {
		usd_free_mr(rx->r.rdm.rx_bufs);
		free(rx->r.rdm.rx_rqe_buf);
		usd_destroy_qp(rx->rx_qp);
	}
	ofi_atomic_dec32(&rx->rx_domain->dom_refcnt);

	free(rx);

	return 0;
}

/*
 * XXX clean up pending receives
 */
static int
usdf_rdm_tx_ctx_close(fid_t fid)
{
	struct usdf_tx *tx;
	struct usdf_cq_hard *hcq;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	tx = tx_fidtou(fid);

	if (ofi_atomic_get32(&tx->tx_refcnt) > 0) {
		return -FI_EBUSY;
	}

	hcq = tx->t.rdm.tx_hcq;
	if (hcq != NULL) {
		ofi_atomic_dec32(&hcq->cqh_refcnt);
		ofi_atomic_dec32(&hcq->cqh_cq->cq_refcnt);
	}

	if (tx->tx_qp != NULL) {
		usd_free_mr(tx->t.rdm.tx_inject_bufs);
		free(tx->t.rdm.tx_wqe_buf);
		usd_destroy_qp(tx->tx_qp);
	}
	ofi_atomic_dec32(&tx->tx_domain->dom_refcnt);

	free(tx);

	return 0;
}

static int
usdf_rx_rdm_port_bind(struct usdf_rx *rx, struct fi_info *info)
{
	struct sockaddr_in *sin;
	struct sockaddr_in src;
	socklen_t addrlen;
	int ret;

	if (info->src_addr != NULL) {
		switch (info->addr_format) {
		case FI_SOCKADDR:
		case FI_SOCKADDR_IN:
		case FI_ADDR_STR:
			sin = usdf_format_to_sin(info, info->src_addr);
			if (NULL == sin) {
				return -FI_ENOMEM;
			}
			break;
		default:
			return -FI_EINVAL;
		}
	} else {
		memset(&src, 0, sizeof(src));
		sin = &src;
		sin->sin_family = AF_INET;
		sin->sin_addr.s_addr =
			rx->rx_domain->dom_fabric->fab_dev_attrs->uda_ipaddr_be;
	}

	rx->r.rdm.rx_sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (rx->r.rdm.rx_sock == -1) {
		return -errno;
	}
	ret = bind(rx->r.rdm.rx_sock, (struct sockaddr *)sin, sizeof(*sin));
	if (ret == -1) {
		return -errno;
	}

	addrlen = sizeof(*sin);
	ret = getsockname(rx->r.rdm.rx_sock, (struct sockaddr *)sin, &addrlen);
	if (ret == -1) {
		 return -errno;
	}

	/* This has to be here because usdf_sin_to_format will allocate
	 * new piece of memory if the string conversion happens.
	 */
	if (info->addr_format == FI_ADDR_STR)
		free(info->src_addr);

	info->src_addr = usdf_sin_to_format(info, sin, &info->src_addrlen);

	return 0;
}

static int
usdf_ep_rdm_close(fid_t fid)
{
	struct usdf_ep *ep;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ep = ep_fidtou(fid);

	if (ofi_atomic_get32(&ep->ep_refcnt) > 0) {
		return -FI_EBUSY;
	}

	if (ep->ep_rx != NULL) {
		ofi_atomic_dec32(&ep->ep_rx->rx_refcnt);
		if (rx_utofid(ep->ep_rx)->fclass  == FI_CLASS_RX_CTX) {
			(void) usdf_rdm_rx_ctx_close(rx_utofid(ep->ep_rx));
		}
	}

	if (ep->ep_tx != NULL) {
		ofi_atomic_dec32(&ep->ep_tx->tx_refcnt);
		if (tx_utofid(ep->ep_tx)->fclass  == FI_CLASS_TX_CTX) {
			(void) usdf_rdm_tx_ctx_close(tx_utofid(ep->ep_tx));
		}
	}

	ofi_atomic_dec32(&ep->ep_domain->dom_refcnt);
	if (ep->ep_eq != NULL) {
		ofi_atomic_dec32(&ep->ep_eq->eq_refcnt);
	}

	if (ep->e.rdm.ep_av)
		ofi_atomic_dec32(&ep->e.rdm.ep_av->av_refcnt);

	free(ep);
	return 0;
}

static struct fi_ops_ep usdf_base_rdm_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = usdf_ep_rdm_cancel,
	.getopt = usdf_ep_getopt_unconnected,
	.setopt = usdf_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = usdf_rdm_rx_size_left,
	.tx_size_left = usdf_rdm_tx_size_left,
};

static struct fi_ops_cm usdf_cm_rdm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = usdf_cm_rdm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static struct fi_ops_msg usdf_rdm_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = usdf_rdm_recv,
	.recvv = usdf_rdm_recvv,
	.recvmsg = usdf_rdm_recvmsg,
	.send = usdf_rdm_send,
	.sendv = usdf_rdm_sendv,
	.sendmsg = usdf_rdm_sendmsg,
	.inject = usdf_rdm_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static int usdf_ep_rdm_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep;
	int ret;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ep = container_of(fid, struct fid_ep, fid);
		switch (command) {
		case FI_ENABLE:
			ret = usdf_ep_rdm_enable(ep);
			break;
		default:
			ret = -FI_ENOSYS;
		}
		break;
	default:
		ret = -FI_ENOSYS;
	}

	return ret;
}

static struct fi_ops usdf_ep_rdm_ops = {
	.size = sizeof(struct fi_ops),
	.close = usdf_ep_rdm_close,
	.bind = usdf_ep_rdm_bind,
	.control = usdf_ep_rdm_control,
	.ops_open = fi_no_ops_open
};

int
usdf_ep_rdm_open(struct fid_domain *domain, struct fi_info *info,
	    struct fid_ep **ep_o, void *context)
{
	struct usdf_domain *udp;
	struct usdf_tx *tx;
	struct usdf_rx *rx;
	struct usdf_ep *ep;
	int ret;
	uint32_t api_version;

	USDF_TRACE_SYS(EP_CTRL, "\n");

	ep = NULL;
	rx = NULL;
	tx = NULL;
	if ((info->caps & ~USDF_RDM_CAPS) != 0) {
		return -FI_EBADFLAGS;
	}

	udp = dom_ftou(domain);
	api_version = udp->dom_fabric->fab_attr.fabric->api_version;

	/* allocate peer table if not done */
	if (udp->dom_peer_tab == NULL) {
		udp->dom_peer_tab = calloc(USDF_MAX_PEERS, sizeof(ep));
	}
	if (udp->dom_peer_tab == NULL) {
		ret = -errno;
		goto fail;
	}

	ep = calloc(1, sizeof(*ep));
	if (ep == NULL) {
		ret = -errno;
		goto fail;
	}

	ep->ep_fid.fid.fclass = FI_CLASS_EP;
	ep->ep_fid.fid.context = context;
	ep->ep_fid.fid.ops = &usdf_ep_rdm_ops;
	ep->ep_fid.ops = &usdf_base_rdm_ops;
	ep->ep_fid.cm = &usdf_cm_rdm_ops;
	ep->ep_fid.msg = &usdf_rdm_ops;
	ep->ep_fid.atomic = &usdf_rdm_atomic_ops;
	ep->ep_domain = udp;
	ep->ep_caps = info->caps;
	ep->ep_mode = info->mode;
	ep->ep_tx_dflt_signal_comp = 1;
	ep->ep_rx_dflt_signal_comp = 1;

	/* implicitly create TX context if not to be shared */
	if (info->ep_attr == NULL ||
	    info->ep_attr->tx_ctx_cnt != FI_SHARED_CONTEXT) {
		tx = calloc(1, sizeof(*tx));
		if (tx == NULL) {
			ret = -errno;
			goto fail;
		}
		tx->tx_fid.fid.fclass = FI_CLASS_TX_CTX;
		ofi_atomic_initialize32(&tx->tx_refcnt, 0);
		tx->tx_domain = udp;
		tx->tx_progress = usdf_rdm_tx_progress;
		ofi_atomic_initialize32(&tx->t.rdm.tx_next_msg_id, 1);
		ofi_atomic_inc32(&udp->dom_refcnt);

		/* info is both hints and output */
		ret = usdf_rdm_fill_tx_attr(api_version, info, info);
		if (ret)
			goto fail;
		tx->tx_attr = *info->tx_attr;

		TAILQ_INIT(&tx->t.rdm.tx_free_wqe);
		TAILQ_INIT(&tx->t.rdm.tx_rdc_ready);
		TAILQ_INIT(&tx->t.rdm.tx_rdc_have_acks);

		ep->ep_tx = tx;
		ofi_atomic_inc32(&tx->tx_refcnt);
	}

	/* implicitly create RX context if not to be shared */
	if (info->ep_attr == NULL ||
	    info->ep_attr->rx_ctx_cnt != FI_SHARED_CONTEXT) {
		rx = calloc(1, sizeof(*rx));
		if (rx == NULL) {
			ret = -errno;
			goto fail;
		}

		rx->rx_fid.fid.fclass = FI_CLASS_RX_CTX;
		ofi_atomic_initialize32(&rx->rx_refcnt, 0);
		rx->rx_domain = udp;
		rx->r.rdm.rx_tx = tx;
		rx->r.rdm.rx_sock = -1;
		ofi_atomic_inc32(&udp->dom_refcnt);

		ret = usdf_rx_rdm_port_bind(rx, info);
		if (ret) {
			goto fail;
		}

		/* info is both hints and output */
		ret = usdf_rdm_fill_rx_attr(api_version, info, info);
		if (ret) {
			goto fail;
		}
		rx->rx_attr = *info->rx_attr;

		TAILQ_INIT(&rx->r.rdm.rx_free_rqe);
		TAILQ_INIT(&rx->r.rdm.rx_posted_rqe);

		ep->ep_rx = rx;
		ofi_atomic_inc32(&rx->rx_refcnt);
	}

	ofi_atomic_initialize32(&ep->ep_refcnt, 0);
	ofi_atomic_inc32(&udp->dom_refcnt);

	*ep_o = ep_utof(ep);
	return 0;
fail:
	if (rx != NULL) {
		if (rx->r.rdm.rx_sock != -1) {
			close(rx->r.rdm.rx_sock);
		}
		free(rx);
		ofi_atomic_dec32(&udp->dom_refcnt);
	}
	if (tx != NULL) {
		free(tx);
		ofi_atomic_dec32(&udp->dom_refcnt);
	}
	if (ep != NULL) {
		free(ep);
	}
	return ret;
}
