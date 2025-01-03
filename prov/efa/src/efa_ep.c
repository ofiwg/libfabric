/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "config.h"
#include "efa.h"
#include "efa_av.h"
#include "efa_cq.h"

#include <infiniband/efadv.h>

extern struct fi_ops_msg efa_msg_ops;
extern struct fi_ops_rma efa_rma_ops;

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

static void efa_ep_destroy(struct efa_base_ep *ep)
{
	int ret;

	ret = efa_base_ep_destruct(ep);
	if (ret) {
		EFA_WARN(FI_LOG_EP_CTRL, "Unable to close base endpoint\n");
	}

	free(ep);
}

static int efa_ep_close(fid_t fid)
{
	struct efa_base_ep *ep;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);

	efa_ep_destroy(ep);

	return 0;
}

static int efa_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct efa_base_ep *ep;
	struct efa_cq *cq;
	struct efa_av *av;
	struct efa_domain *efa_domain;
	struct util_eq *eq;
	struct util_cntr *cntr;
	int ret;

	ep = container_of(fid, struct efa_base_ep, util_ep.ep_fid.fid);
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

		cq = container_of(bfid, struct efa_cq, util_cq.cq_fid);
		efa_domain = container_of(cq->util_cq.domain, struct efa_domain, util_domain);
		if (ep->domain != efa_domain)
			return -FI_EINVAL;

		ret = ofi_ep_bind_cq(&ep->util_ep, &cq->util_cq, flags);
		if (ret)
			return ret;

		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct efa_av, util_av.av_fid.fid);
		ret = efa_base_ep_bind_av(ep, av);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&ep->util_ep, cntr, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct util_eq, eq_fid.fid);

		ret = ofi_ep_bind_eq(&ep->util_ep, eq);
		if (ret)
			return ret;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static int efa_ep_getflags(struct fid_ep *ep_fid, uint64_t *flags)
{
	struct efa_base_ep *ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
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
	struct efa_base_ep *ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
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
	struct ibv_qp_init_attr_ex attr_ex = { 0 };
	struct efa_base_ep *ep;
	struct efa_cq *scq, *rcq;
	int err;

	ep = container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);

	scq = ep->util_ep.tx_cq ? container_of(ep->util_ep.tx_cq, struct efa_cq, util_cq) : NULL;
	rcq = ep->util_ep.rx_cq ? container_of(ep->util_ep.rx_cq, struct efa_cq, util_cq) : NULL;

	if (!scq && !rcq) {
		EFA_WARN(FI_LOG_EP_CTRL,
			"Endpoint is not bound to a send or receive completion queue\n");
		return -FI_ENOCQ;
	}

	if (!scq && ofi_needs_tx(ep->info->caps)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			"Endpoint is not bound to a send completion queue when it has transmit capabilities enabled (FI_SEND).\n");
		return -FI_ENOCQ;
	}

	if (!rcq && ofi_needs_rx(ep->info->caps)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			"Endpoint is not bound to a receive completion queue when it has receive capabilities enabled. (FI_RECV)\n");
		return -FI_ENOCQ;
	}

	if (scq) {
		attr_ex.cap.max_send_wr = ep->info->tx_attr->size;
		attr_ex.cap.max_send_sge = ep->info->tx_attr->iov_limit;
		attr_ex.send_cq = ibv_cq_ex_to_cq(scq->ibv_cq.ibv_cq_ex);
	} else {
		attr_ex.send_cq = ibv_cq_ex_to_cq(rcq->ibv_cq.ibv_cq_ex);
	}

	if (rcq) {
		attr_ex.cap.max_recv_wr = ep->info->rx_attr->size;
		attr_ex.cap.max_recv_sge = ep->info->rx_attr->iov_limit;
		attr_ex.recv_cq = ibv_cq_ex_to_cq(rcq->ibv_cq.ibv_cq_ex);
	} else {
		attr_ex.recv_cq = ibv_cq_ex_to_cq(scq->ibv_cq.ibv_cq_ex);
	}

	attr_ex.cap.max_inline_data =
		ep->domain->device->efa_attr.inline_buf_size;

	assert(EFA_EP_TYPE_IS_DGRAM(ep->domain->info));
	attr_ex.qp_type = IBV_QPT_UD;
	attr_ex.comp_mask = IBV_QP_INIT_ATTR_PD;
	attr_ex.pd = container_of(ep->util_ep.domain, struct efa_domain, util_domain)->ibv_pd;

	attr_ex.qp_context = ep;
	attr_ex.sq_sig_all = 1;

	err = efa_base_ep_create_qp(ep, &attr_ex);
	if (err)
		return err;

	return efa_base_ep_enable(ep);
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

/**
 * @brief progress engine for the EFA dgram endpoint
 *
 * This function now a no-op.
 *
 * @param[in] util_ep The endpoint FID to progress
 */
static
void efa_ep_progress_no_op(struct util_ep *util_ep)
{
	return;
}

static struct fi_ops_atomic efa_atomic_ops = {
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

struct fi_ops_cm efa_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = efa_base_ep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

int efa_ep_open(struct fid_domain *domain_fid, struct fi_info *user_info,
		struct fid_ep **ep_fid, void *context)
{
	struct efa_domain *domain;
	const struct fi_info *prov_info;
	struct efa_base_ep *ep;
	int ret;

	domain = container_of(domain_fid, struct efa_domain,
			      util_domain.domain_fid);

	if (!user_info || !user_info->ep_attr || !user_info->domain_attr ||
	    strncmp(domain->device->ibv_ctx->device->name, user_info->domain_attr->name,
		    strlen(domain->device->ibv_ctx->device->name))) {
		EFA_INFO(FI_LOG_DOMAIN, "Invalid info->domain_attr->name\n");
		return -FI_EINVAL;
	}

	prov_info = efa_domain_get_prov_info(domain, user_info->ep_attr->type);
	assert(prov_info);

	assert(user_info->ep_attr);
	ret = ofi_check_ep_attr(&efa_util_prov, user_info->fabric_attr->api_version, prov_info, user_info);
	if (ret)
		return ret;

	if (user_info->tx_attr) {
		ret = ofi_check_tx_attr(&efa_prov, prov_info->tx_attr,
					user_info->tx_attr, user_info->mode);
		if (ret)
			return ret;
	}

	if (user_info->rx_attr) {
		ret = ofi_check_rx_attr(&efa_prov, prov_info, user_info->rx_attr, user_info->mode);
		if (ret)
			return ret;
	}

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = efa_base_ep_construct(ep, domain_fid, user_info, efa_ep_progress_no_op, context);
	if (ret)
		goto err_ep_destroy;

	/* struct efa_send_wr and efa_recv_wr allocates memory for 2 IOV
	 * So check with an assert statement that iov_limit is 2 or less
	 */
	assert(user_info->tx_attr->iov_limit <= 2);

	ep->domain = domain;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.fclass = FI_CLASS_EP;
	(*ep_fid)->fid.context = context;
	(*ep_fid)->fid.ops = &efa_ep_ops;
	(*ep_fid)->ops = &efa_ep_base_ops;
	(*ep_fid)->msg = &efa_msg_ops;
	(*ep_fid)->cm = &efa_ep_cm_ops;
	(*ep_fid)->rma = &efa_rma_ops;
	(*ep_fid)->atomic = &efa_atomic_ops;

	return 0;

err_ep_destroy:
	efa_ep_destroy(ep);
	return ret;
}
