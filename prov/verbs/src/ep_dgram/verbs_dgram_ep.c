/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#include "verbs_dgram.h"

static int fi_ibv_dgram_ep_enable(struct fid_ep *ep_fid)
{
	struct fi_ibv_dgram_ep *ep;
	struct fi_ibv_dgram_cq *tx_cq = NULL, *rx_cq = NULL;
	struct fi_ibv_fabric *fab;
	int ret = FI_SUCCESS;
	union ibv_gid gid;
	uint16_t p_key;

	assert(ep_fid->fid.fclass == FI_CLASS_EP);
	if (ep_fid->fid.fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid);
	if (!ep)
		return -FI_EINVAL;

	if (!ep->util_ep.rx_cq && !ep->util_ep.tx_cq) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Endpoint is not bound to "
			   "a send or receive completion queue\n");
		return -FI_ENOCQ;
	}

	if (!ep->util_ep.tx_cq && ofi_send_allowed(ep->util_ep.caps)) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Endpoint is not bound to "
			   "a send completion queue when it has transmit "
			   "capabilities enabled (FI_SEND or FI_TRANSMIT).\n");
		return -FI_ENOCQ;
	} else if (ep->util_ep.tx_cq) {
		tx_cq = container_of(&ep->util_ep.tx_cq->cq_fid,
				     struct fi_ibv_dgram_cq,
				     util_cq.cq_fid);
	}

	if (!ep->util_ep.rx_cq && ofi_recv_allowed(ep->util_ep.caps)) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Endpoint is not bound to "
			   "a receive completion queue when it has receive "
			   "capabilities enabled. (FI_RECV)\n");
		return -FI_ENOCQ;
	} else {
		rx_cq = container_of(&ep->util_ep.rx_cq->cq_fid,
				     struct fi_ibv_dgram_cq,
				     util_cq.cq_fid);
	}

	const struct fi_info *info = ep->info;
	struct ibv_qp_init_attr init_attr = {
		.send_cq 	= tx_cq ? tx_cq->ibv_cq : NULL,
		.recv_cq 	= rx_cq ? rx_cq->ibv_cq : NULL,
		.cap 		= {
			.max_send_wr		= info->tx_attr->size,
		       	.max_recv_wr		= info->rx_attr->size,
		       	.max_send_sge		= info->tx_attr->iov_limit,
		       	.max_recv_sge		= info->rx_attr->iov_limit,
			.max_inline_data	= info->tx_attr->inject_size,
	       	},
	       	.qp_type 	= IBV_QPT_UD,
       	};

	ep->ibv_qp = ibv_create_qp(ep->domain->pd, &init_attr);
	if (!ep->ibv_qp) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to create IBV "
			   "Queue Pair\n");
		return -errno;
	}

	struct ibv_qp_attr attr = {
		.qp_state = IBV_QPS_INIT,
		.pkey_index = 0,
		.port_num = 1,
		.qkey = 0x11111111,
	};

	ret = ibv_modify_qp(ep->ibv_qp, &attr,
			    IBV_QP_STATE |
			    IBV_QP_PKEY_INDEX |
			    IBV_QP_PORT |
			    IBV_QP_QKEY);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to modify QP state "
			   "to INIT\n");
		return -errno;
	}

	memset(&attr, 0, sizeof(attr));
	attr.qp_state = IBV_QPS_RTR;
	ret = ibv_modify_qp(ep->ibv_qp, &attr,
			    IBV_QP_STATE);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to modify QP state "
			   "to RTR\n");
		return -errno;
	}

	if (tx_cq) {
		memset(&attr, 0, sizeof(attr));
		attr.qp_state = IBV_QPS_RTS;
		attr.sq_psn = 0xffffff;
		ret = ibv_modify_qp(ep->ibv_qp, &attr,
				    IBV_QP_STATE |
				    IBV_QP_SQ_PSN);
		if (ret) {
			VERBS_WARN(FI_LOG_EP_CTRL, "Unable to modify QP state "
				   "to RTS\n");
			return -errno;
		}
	}

	if (ibv_query_gid(ep->domain->verbs, 1, 0, &gid)) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Unable to query GID, errno = %d",
			   errno);
		return -errno;
	}

	if (ibv_query_pkey(ep->domain->verbs, 1, 0, &p_key)) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Unable to query P_Key, errno = %d",
			   errno);
		return -errno;
	}

	struct ibv_port_attr port_attr;
	if (ibv_query_port(ep->domain->verbs, 1, &port_attr)) {
		VERBS_WARN(FI_LOG_EP_CTRL,
			   "Unable to query port attributes, errno = %d",
			   errno);
		return -errno;
	}

	ep->ep_name.lid = port_attr.lid;
	ep->ep_name.sl = port_attr.sm_sl;
	ep->ep_name.gid = gid;
	ep->ep_name.qpn = ep->ibv_qp->qp_num;
	ep->ep_name.pkey = p_key;

	fab = container_of(&ep->util_ep.domain->fabric->fabric_fid,
			   struct fi_ibv_fabric, util_fabric.fabric_fid.fid);

	ofi_ns_add_local_name(&fab->name_server,
			      &ep->service, &ep->ep_name);

	return ret;
}

static int fi_ibv_dgram_ep_control(fid_t ep_fid, int command, void *arg)
{
	struct fi_ibv_dgram_ep *ep;

	assert(ep_fid->fclass == FI_CLASS_EP);
	if (ep_fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid.fid);
	if (!ep)
		return -FI_EINVAL;

	switch (command) {
	case FI_ENABLE:
		return fi_ibv_dgram_ep_enable(&ep->util_ep.ep_fid);
	default:
		return -FI_ENOSYS;
	}
}

static int fi_ibv_dgram_ep_close(fid_t ep_fid)
{
	struct fi_ibv_dgram_ep *ep;
	struct fi_ibv_fabric *fab;
	int ret = FI_SUCCESS;

	assert(ep_fid->fclass == FI_CLASS_EP);
	if (ep_fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid.fid);
	if (!ep)
		return -FI_EINVAL;

	fab = container_of(&ep->util_ep.domain->fabric->fabric_fid,
			   struct fi_ibv_fabric, util_fabric.fabric_fid.fid);

	ofi_ns_del_local_name(&fab->name_server,
			      &ep->service, &ep->ep_name);

	fi_ibv_dgram_pool_destroy(&ep->grh_pool);

	ret = ofi_endpoint_close(&ep->util_ep);
	if (ret)
		return ret;

	ret = ibv_destroy_qp(ep->ibv_qp);
	if (ret) {
		VERBS_WARN(FI_LOG_EP_CTRL, "Unable to destroy QP "
			   "(errno = %d)\n", errno);
 		ret = -errno;
	}

	free(ep);
	return ret;
}

static int fi_ibv_dgram_ep_bind(fid_t ep_fid, struct fid *bfid, uint64_t flags)
{
	int ret = FI_SUCCESS;
	struct fi_ibv_dgram_ep *ep;
	struct fi_ibv_dgram_cq *cq;
	struct fi_ibv_dgram_av *av;
	struct fi_ibv_dgram_eq *eq;
	struct fi_ibv_dgram_cntr *cntr;

	assert(ep_fid->fclass == FI_CLASS_EP);
	if (ep_fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid.fid);
	if (!ep)
		return -FI_EINVAL;

	ret = ofi_ep_bind_valid(&fi_ibv_prov, bfid, flags);
	if (ret)
		return ret;

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct fi_ibv_dgram_cq,
				  util_cq.cq_fid.fid);
		if (!cq)
			return -FI_EINVAL;
		if (flags & (FI_RECV | FI_TRANSMIT))
			ep->util_ep.progress =
				fi_ibv_dgram_send_recv_cq_progress;
		else if (flags & FI_RECV)
			ep->util_ep.progress = (ep->util_ep.tx_cq) ?
				fi_ibv_dgram_send_recv_cq_progress :
				fi_ibv_dgram_recv_cq_progress;
		else if (flags & FI_TRANSMIT)
			ep->util_ep.progress = (ep->util_ep.rx_cq) ?
				fi_ibv_dgram_send_recv_cq_progress :
				fi_ibv_dgram_send_cq_progress;
		return ofi_ep_bind_cq(&ep->util_ep, &cq->util_cq, flags);
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct fi_ibv_dgram_eq,
				  util_eq.eq_fid.fid);
		if (!eq)
			return -FI_EINVAL;
		return ofi_ep_bind_eq(&ep->util_ep, &eq->util_eq);
		break;
	case FI_CLASS_AV:
		av = container_of(bfid, struct fi_ibv_dgram_av,
				  util_av.av_fid.fid);
		if (!av)
			return -FI_EINVAL;
		return ofi_ep_bind_av(&ep->util_ep, &av->util_av);
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct fi_ibv_dgram_cntr,
				    util_cntr.cntr_fid.fid);
		if (!cntr)
			return -FI_EINVAL;
		return ofi_ep_bind_cntr(&ep->util_ep, &cntr->util_cntr, flags);
	default:
		return -FI_EINVAL;
	}

	return ret;
}

static int fi_ibv_dgram_ep_setname(fid_t ep_fid, void *addr, size_t addrlen)
{
	struct fi_ibv_dgram_ep *ep;
	void *save_addr;
	int ret = FI_SUCCESS;

	if (ep_fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid.fid);
	if (!ep)
		return -FI_EINVAL;

	if (addrlen < ep->info->src_addrlen) {
		VERBS_INFO(FI_LOG_EP_CTRL,
			   "addrlen expected: %"PRIu64", got: %"PRIu64"\n",
			   ep->info->src_addrlen, addrlen);
		return -FI_ETOOSMALL;
	}
	/*
	 * save previous address to be able make
	 * a roll back on the previous one
	 */
	save_addr = ep->info->src_addr;

	ep->info->src_addr = calloc(1, ep->info->src_addrlen);
	if (!ep->info->src_addr) {
		ep->info->src_addr = save_addr;
		ret = -FI_ENOMEM;
		goto err;
	}

	memcpy(ep->info->src_addr, addr, ep->info->src_addrlen);
	memcpy(&ep->ep_name, addr, ep->info->src_addrlen);

err:
	ep->info->src_addr = save_addr;
	return ret;
	
}

static int fi_ibv_dgram_ep_getname(fid_t ep_fid, void *addr, size_t *addrlen)
{
	struct fi_ibv_dgram_ep *ep;

	if (ep_fid->fclass != FI_CLASS_EP)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct fi_ibv_dgram_ep, util_ep.ep_fid.fid);
	if (!ep)
		return -FI_EINVAL;

	if (*addrlen < sizeof(ep->ep_name)) {
		*addrlen = sizeof(ep->ep_name);
		VERBS_INFO(FI_LOG_EP_CTRL,
			   "addrlen expected: %"PRIu64", got: %"PRIu64"\n",
			   sizeof(ep->ep_name), *addrlen);
		return -FI_ETOOSMALL;
	}

	memset(addr, 0, *addrlen);
	memcpy(addr, &ep->ep_name, sizeof(ep->ep_name));
	*addrlen = sizeof(ep->ep_name);

	return FI_SUCCESS;
}

static struct fi_ops fi_ibv_dgram_fi_ops = {
	.size = sizeof(fi_ibv_dgram_fi_ops),
	.close = fi_ibv_dgram_ep_close,
	.bind = fi_ibv_dgram_ep_bind,
	.control = fi_ibv_dgram_ep_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cm fi_ibv_dgram_cm_ops = {
	.size = sizeof(fi_ibv_dgram_cm_ops),
	.setname = fi_ibv_dgram_ep_setname,
	.getname = fi_ibv_dgram_ep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static struct fi_ops_ep fi_ibv_dgram_ep_ops = {
	.size = sizeof(fi_ibv_dgram_ep_ops),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

int fi_ibv_dgram_endpoint_open(struct fid_domain *domain_fid,
			       struct fi_info *info,
			       struct fid_ep **ep_fid,
			       void *context)
{
	struct fi_ibv_dgram_ep *ep;
	int ret = FI_SUCCESS;

	assert(info && info->ep_attr && info->rx_attr && info->tx_attr);
	assert(domain_fid);
	assert(domain_fid->fid.fclass == FI_CLASS_DOMAIN);

	if (!info || !info->ep_attr ||
	    domain_fid->fid.fclass != FI_CLASS_DOMAIN)
		return -FI_EINVAL;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	struct fi_ibv_dgram_pool_attr pool_attr = {
		.count		= MIN(info->rx_attr->size, info->tx_attr->size),
		.size		= VERBS_DGRAM_WR_ENTRY_SIZE,
		.pool_ctx	= domain_fid,
		.cancel_hndlr	= fi_ibv_dgram_pool_wr_entry_cancel,
		.alloc_hndlr	= fi_ibv_dgram_mr_buf_reg,
		.free_hndlr	= fi_ibv_dgram_mr_buf_close,
	};

	ret = fi_ibv_dgram_pool_create(&pool_attr, &ep->grh_pool);
	if (ret)
		goto err1;

	/* Temporary solution */
	struct fi_ibv_domain *domain;
	domain = container_of(domain_fid, struct fi_ibv_domain,
			      util_domain.domain_fid);
	if (!domain) {
		ret = -FI_EINVAL;
		goto err2;
	}

	ret = ofi_endpoint_init(domain_fid, &fi_ibv_util_prov,
				info, &ep->util_ep, context, NULL);
	if (ret)
		goto err2;

	ep->info = fi_dupinfo(info);
	if (!ep->info) {
		ret = -FI_ENOMEM;
		goto err3;
	}
	ep->domain = domain;
	ep->service = (info->src_addr) ?
			(((struct ofi_ib_ud_ep_name *)info->src_addr)->service) :
			(((getpid() & 0x7FFF) << 16) + ((uintptr_t)ep & 0xFFFF));
	
	ofi_atomic_initialize32(&ep->unsignaled_send_cnt, 0);
	ep->max_unsignaled_send_cnt = ep->info->tx_attr->size / 2;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->cm = &fi_ibv_dgram_cm_ops;
	(*ep_fid)->msg = &fi_ibv_dgram_msg_ops;
	(*ep_fid)->fid.ops = &fi_ibv_dgram_fi_ops;
	(*ep_fid)->ops = &fi_ibv_dgram_ep_ops;

	return ret;
err3:
	ofi_endpoint_close(&ep->util_ep);
err2:
	fi_ibv_dgram_pool_destroy(&ep->grh_pool);
err1:
	free(ep);
	return ret;
}

