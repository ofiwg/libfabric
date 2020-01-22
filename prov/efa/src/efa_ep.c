/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2020 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "config.h"

#include "efa.h"

#include <infiniband/efadv.h>

static int efa_ep_destroy_qp(struct efa_qp *qp)
{
	struct efa_domain *domain;
	int err;

	if (!qp)
		return 0;

	domain = qp->ep->domain;
	domain->qp_table[qp->qp_num & domain->qp_table_sz_m1] = NULL;
	err = -ibv_destroy_qp(qp->ibv_qp);
	if (err)
		EFA_INFO(FI_LOG_CORE, "destroy qp[%u] failed!\n", qp->qp_num);

	free(qp);
	return err;
}

static int efa_ep_modify_qp_state(struct efa_qp *qp, enum ibv_qp_state qp_state,
				  int attr_mask)
{
	struct ibv_qp_attr attr = {};

	attr.qp_state = qp_state;

	if (attr_mask & IBV_QP_PORT)
		attr.port_num = 1;

	if (attr_mask & IBV_QP_QKEY)
		attr.qkey = EFA_QKEY;

	return -ibv_modify_qp(qp->ibv_qp, &attr, attr_mask);

}

static int efa_ep_modify_qp_rst2rts(struct efa_qp *qp)
{
	int err;

	err = efa_ep_modify_qp_state(qp, IBV_QPS_INIT,
				     IBV_QP_STATE | IBV_QP_PKEY_INDEX |
				     IBV_QP_PORT | IBV_QP_QKEY);
	if (err)
		return err;

	err = efa_ep_modify_qp_state(qp, IBV_QPS_RTR, IBV_QP_STATE);
	if (err)
		return err;

	return efa_ep_modify_qp_state(qp, IBV_QPS_RTS,
				      IBV_QP_STATE | IBV_QP_SQ_PSN);
}

static int efa_ep_create_qp(struct efa_ep *ep,
			    struct ibv_pd *ibv_pd,
			    struct ibv_qp_init_attr *init_attr)
{
	struct efa_domain *domain = ep->domain;
	struct efa_qp *qp;
	int err;

	qp = calloc(1, sizeof(*qp));
	if (!qp)
		return -FI_ENOMEM;

	if (init_attr->qp_type == IBV_QPT_UD)
		qp->ibv_qp = ibv_create_qp(ibv_pd, init_attr);
	else
		qp->ibv_qp = efadv_create_driver_qp(ibv_pd, init_attr,
						    EFADV_QP_DRIVER_TYPE_SRD);
	if (!qp->ibv_qp) {
		EFA_WARN(FI_LOG_EP_CTRL, "ibv_create_qp failed\n");
		err = -EINVAL;
		goto err_free_qp;
	}

	err = efa_ep_modify_qp_rst2rts(qp);
	if (err)
		goto err_destroy_qp;

	qp->qp_num = qp->ibv_qp->qp_num;
	ep->qp = qp;
	qp->ep = ep;
	domain->qp_table[ep->qp->qp_num & domain->qp_table_sz_m1] = ep->qp;
	EFA_INFO(FI_LOG_EP_CTRL, "%s(): create QP %d\n", __func__, qp->qp_num);

	return 0;

err_destroy_qp:
	ibv_destroy_qp(qp->ibv_qp);
err_free_qp:
	free(qp);

	return err;
}

static int efa_ep_getopt(fid_t fid, int level, int optname,
			 void *optval, size_t *optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int efa_ep_setopt(fid_t fid, int level, int optname, const void *optval, size_t optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static struct fi_ops_ep efa_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = efa_ep_getopt,
	.setopt = efa_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct efa_ep *efa_ep_alloc(struct fi_info *info)
{
	struct efa_ep *ep;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return NULL;

	ep->info = fi_dupinfo(info);
	if (!ep->info)
		goto err;

	return ep;

err:
	free(ep);
	return NULL;
}

static void efa_ep_destroy(struct efa_ep *ep)
{
	efa_ep_destroy_qp(ep->qp);
	fi_freeinfo(ep->info);
	free(ep->src_addr);
	free(ep);
}

static int efa_ep_close(fid_t fid)
{
	struct efa_ep *ep;

	ep = container_of(fid, struct efa_ep, ep_fid.fid);

	ofi_bufpool_destroy(ep->recv_wr_pool);
	ofi_bufpool_destroy(ep->send_wr_pool);
	efa_ep_destroy(ep);

	return 0;
}

static int efa_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct efa_ep *ep;
	struct efa_cq *cq;
	struct efa_av *av;
	int ret;

	ep = container_of(fid, struct efa_ep, ep_fid.fid);
	ret = ofi_ep_bind_valid(&efa_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		if (flags & FI_SELECTIVE_COMPLETION) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "Endpoint cannot be bound with selective completion.\n");
			return -FI_EBADFLAGS;
		}

		/* Must bind a CQ to either RECV or SEND completions */
		if (!(flags & (FI_RECV | FI_TRANSMIT)))
			return -FI_EBADFLAGS;

		cq = container_of(bfid, struct efa_cq, cq_fid);
		if (ep->domain != cq->domain)
			return -FI_EINVAL;

		if (flags & FI_RECV) {
			if (ep->rcq)
				return -EINVAL;
			ep->rcq = cq;
		}
		if (flags & FI_TRANSMIT) {
			if (ep->scq)
				return -EINVAL;
			ep->scq = cq;
		}
		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct efa_av, util_av.av_fid.fid);
		if (ep->domain != av->domain) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "Address vector doesn't belong to same domain as EP.\n");
			return -FI_EINVAL;
		}
		if (ep->av) {
			EFA_WARN(FI_LOG_EP_CTRL,
				 "Address vector already bound to EP.\n");
			return -FI_EINVAL;
		}
		ep->av = av;

		ep->av->ep = ep;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static int efa_ep_getflags(struct fid_ep *ep_fid, uint64_t *flags)
{
	struct efa_ep *ep = container_of(ep_fid, struct efa_ep, ep_fid);
	struct fi_tx_attr *tx_attr = ep->info->tx_attr;
	struct fi_rx_attr *rx_attr = ep->info->rx_attr;

	if ((*flags & FI_TRANSMIT) && (*flags & FI_RECV)) {
		EFA_WARN(FI_LOG_EP_CTRL, "Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (*flags & FI_TRANSMIT)) {
		*flags = tx_attr->op_flags;
	} else if (rx_attr && (*flags & FI_RECV)) {
		*flags = rx_attr->op_flags;
	} else {
		EFA_WARN(FI_LOG_EP_CTRL, "Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}
	return 0;
}

static int efa_ep_setflags(struct fid_ep *ep_fid, uint64_t flags)
{
	struct efa_ep *ep = container_of(ep_fid, struct efa_ep, ep_fid);
	struct fi_tx_attr *tx_attr = ep->info->tx_attr;
	struct fi_rx_attr *rx_attr = ep->info->rx_attr;

	if ((flags & FI_TRANSMIT) && (flags & FI_RECV)) {
		EFA_WARN(FI_LOG_EP_CTRL, "Both Tx/Rx flags cannot be specified.\n");
		return -FI_EINVAL;
	} else if (tx_attr && (flags & FI_TRANSMIT)) {
		tx_attr->op_flags = flags;
		tx_attr->op_flags &= ~FI_TRANSMIT;
	} else if (rx_attr && (flags & FI_RECV)) {
		rx_attr->op_flags = flags;
		rx_attr->op_flags &= ~FI_RECV;
	} else {
		EFA_WARN(FI_LOG_EP_CTRL, "Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int efa_ep_enable(struct fid_ep *ep_fid)
{
	struct ibv_qp_init_attr attr = { 0 };
	const struct fi_info *efa_info;
	struct ibv_pd *ibv_pd;
	struct efa_ep *ep;

	ep = container_of(ep_fid, struct efa_ep, ep_fid);

	if (!ep->scq && !ep->rcq) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Endpoint is not bound to a send or receive completion queue\n");
		return -FI_ENOCQ;
	}

	if (!ep->scq && ofi_send_allowed(ep->info->caps)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Endpoint is not bound to a send completion queue when it has transmit capabilities enabled (FI_SEND).\n");
		return -FI_ENOCQ;
	}

	if (!ep->rcq && ofi_recv_allowed(ep->info->caps)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Endpoint is not bound to a receive completion queue when it has receive capabilities enabled. (FI_RECV)\n");
		return -FI_ENOCQ;
	}

	efa_info = efa_get_efa_info(ep->info->domain_attr->name);
	if (!efa_info) {
		EFA_INFO(FI_LOG_EP_CTRL, "Unable to find matching efa_info\n");
		return -FI_EINVAL;
	}

	if (ep->scq) {
		attr.cap.max_send_wr = ep->info->tx_attr->size;
		attr.cap.max_send_sge = ep->info->tx_attr->iov_limit;
		attr.send_cq = ep->scq->ibv_cq;
		ibv_pd = ep->scq->domain->ibv_pd;
	} else {
		attr.send_cq = ep->rcq->ibv_cq;
		ibv_pd = ep->rcq->domain->ibv_pd;
	}

	if (ep->rcq) {
		attr.cap.max_recv_wr = ep->info->rx_attr->size;
		attr.cap.max_recv_sge = ep->info->rx_attr->iov_limit;
		attr.recv_cq = ep->rcq->ibv_cq;
	} else {
		attr.recv_cq = ep->scq->ibv_cq;
	}

	attr.cap.max_inline_data = ep->domain->ctx->inline_buf_size;
	attr.qp_type = ep->domain->rdm ? IBV_QPT_DRIVER : IBV_QPT_UD;
	attr.qp_context = ep;
	attr.sq_sig_all = 1;

	return efa_ep_create_qp(ep, ibv_pd, &attr);
}

static int efa_ep_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep_fid;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ep_fid = container_of(fid, struct fid_ep, fid);
		switch (command) {
		case FI_GETOPSFLAG:
			return efa_ep_getflags(ep_fid, (uint64_t *)arg);
		case FI_SETOPSFLAG:
			return efa_ep_setflags(ep_fid, *(uint64_t *)arg);
		case FI_ENABLE:
			return efa_ep_enable(ep_fid);
		default:
			return -FI_ENOSYS;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
}

static struct fi_ops efa_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = efa_ep_close,
	.bind = efa_ep_bind,
	.control = efa_ep_control,
	.ops_open = fi_no_ops_open,
};

int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		struct fid_ep **ep_fid, void *context)
{
	struct efa_domain *domain;
	const struct fi_info *fi;
	struct efa_ep *ep;
	int ret;

	domain = container_of(domain_fid, struct efa_domain,
			      util_domain.domain_fid);

	if (!info || !info->ep_attr || !info->domain_attr ||
	    strncmp(domain->ctx->ibv_ctx->device->name, info->domain_attr->name,
		    strlen(domain->ctx->ibv_ctx->device->name))) {
		EFA_INFO(FI_LOG_DOMAIN, "Invalid info->domain_attr->name\n");
		return -FI_EINVAL;
	}

	fi = efa_get_efa_info(info->domain_attr->name);
	if (!fi) {
		EFA_INFO(FI_LOG_DOMAIN, "Unable to find matching efa_info\n");
		return -FI_EINVAL;
	}

	if (info->ep_attr) {
		ret = ofi_check_ep_attr(&efa_util_prov, info->fabric_attr->api_version, fi, info);
		if (ret)
			return ret;
	}

	if (info->tx_attr) {
		ret = ofi_check_tx_attr(&efa_prov, fi->tx_attr,
					info->tx_attr, info->mode);
		if (ret)
			return ret;
	}

	if (info->rx_attr) {
		ret = ofi_check_rx_attr(&efa_prov, fi, info->rx_attr, info->mode);
		if (ret)
			return ret;
	}

	ep = efa_ep_alloc(info);
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_bufpool_create(&ep->send_wr_pool,
		sizeof(struct efa_send_wr) +
		info->tx_attr->iov_limit * sizeof(struct ibv_sge),
		16, 0, 1024, 0);
	if (ret)
		goto err_ep_destroy;

	ret = ofi_bufpool_create(&ep->recv_wr_pool,
		sizeof(struct efa_recv_wr) +
		info->rx_attr->iov_limit * sizeof(struct ibv_sge),
		16, 0, 1024, 0);
	if (ret)
		goto err_send_wr_destroy;

	ep->domain = domain;
	ep->xmit_more_wr_tail = &ep->xmit_more_wr_head;
	ep->recv_more_wr_tail = &ep->recv_more_wr_head;
	ep->ep_fid.fid.fclass = FI_CLASS_EP;
	ep->ep_fid.fid.context = context;
	ep->ep_fid.fid.ops = &efa_ep_ops;
	ep->ep_fid.ops = &efa_ep_base_ops;
	ep->ep_fid.msg = &efa_ep_msg_ops;
	ep->ep_fid.cm = &efa_ep_cm_ops;
	ep->ep_fid.rma = NULL;
	ep->ep_fid.atomic = NULL;

	if (info->src_addr) {
		ep->src_addr = (void *)calloc(1, EFA_EP_ADDR_LEN);
		if (!ep->src_addr) {
			ret = -FI_ENOMEM;
			goto err_recv_wr_destroy;
		}
		memcpy(ep->src_addr, info->src_addr, info->src_addrlen);
	}

	*ep_fid = &ep->ep_fid;

	return 0;

err_recv_wr_destroy:
	ofi_bufpool_destroy(ep->recv_wr_pool);
err_send_wr_destroy:
	ofi_bufpool_destroy(ep->send_wr_pool);
err_ep_destroy:
	efa_ep_destroy(ep);
	return ret;
}
