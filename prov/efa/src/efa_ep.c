/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017-2018 Amazon.com, Inc. or its affiliates. All rights reserved.
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
#include "efa_verbs.h"
#include "efa_ib.h"
#include "efa_io_defs.h"

static void efa_ep_init_qp_indices(struct efa_qp *qp)
{
	qp->sq.wq.wqe_posted = 0;
	qp->sq.wq.wqe_completed = 0;
	qp->sq.wq.desc_idx = 0;
	qp->sq.wq.wrid_idx_pool_next = 0;

	qp->rq.wq.wqe_posted = 0;
	qp->rq.wq.wqe_completed = 0;
	qp->rq.wq.desc_idx = 0;
	qp->rq.wq.wrid_idx_pool_next = 0;
}

static void efa_ep_setup_qp(struct efa_qp *qp,
			    struct ibv_qp_cap *cap,
			    size_t page_size)
{
	uint16_t rq_desc_cnt;

	efa_ep_init_qp_indices(qp);

	qp->sq.wq.wqe_cnt = align_up_queue_size(cap->max_send_wr);
	qp->sq.wq.max_sge = cap->max_send_sge;
	qp->sq.wq.desc_mask = qp->sq.wq.wqe_cnt - 1;

	qp->rq.wq.max_sge = cap->max_recv_sge;
	rq_desc_cnt = align_up_queue_size(cap->max_recv_sge * cap->max_recv_wr);
	qp->rq.wq.desc_mask = rq_desc_cnt - 1;
	qp->rq.wq.wqe_cnt = rq_desc_cnt / qp->rq.wq.max_sge;

	qp->page_size = page_size;
}

static void efa_ep_wq_terminate(struct efa_wq *wq)
{
	free(wq->wrid_idx_pool);
	free(wq->wrid);
}

static int efa_ep_wq_initialize(struct efa_wq *wq)
{
	int i, err;

	wq->wrid = malloc(wq->wqe_cnt * sizeof(*wq->wrid));
	if (!wq->wrid)
		return -ENOMEM;

	wq->wrid_idx_pool = malloc(wq->wqe_cnt * sizeof(__u32));
	if (!wq->wrid_idx_pool) {
		err = -ENOMEM;
		goto err_free_wrid;
	}

	/* Initialize the wrid free indexes pool. */
	for (i = 0; i < wq->wqe_cnt; i++)
		wq->wrid_idx_pool[i] = i;

	return 0;

err_free_wrid:
	free(wq->wrid);

	return err;
}

static int efa_ep_sq_initialize(struct efa_qp *qp, struct efa_create_qp_resp *resp, int fd)
{
	size_t desc_ring_size;
	uint8_t *db_base;
	int err;

	if (!qp->sq.wq.wqe_cnt)
		return 0;

	err = efa_ep_wq_initialize(&qp->sq.wq);
	if (err)
		return err;

	qp->sq.immediate_data_width = 8;
	qp->sq.desc_offset = resp->efa_resp.llq_desc_offset;
	desc_ring_size = qp->sq.wq.wqe_cnt * sizeof(struct efa_io_tx_wqe);
	qp->sq.desc_ring_mmap_size = align(desc_ring_size + qp->sq.desc_offset, qp->page_size);
	qp->sq.max_inline_data = resp->ibv_resp.max_inline_data;

	qp->sq.desc = mmap(NULL, qp->sq.desc_ring_mmap_size, PROT_WRITE,
			   MAP_SHARED, fd, resp->efa_resp.llq_desc_mmap_key);
	if (qp->sq.desc == MAP_FAILED)
		goto err_terminate_wq;
	qp->sq.desc += qp->sq.desc_offset;

	db_base = mmap(NULL, qp->page_size, PROT_WRITE, MAP_SHARED, fd, resp->efa_resp.sq_db_mmap_key);
	if (db_base == MAP_FAILED)
		goto err_unmap_desc_ring;
	qp->sq.db = (uint32_t *)(db_base + resp->efa_resp.sq_db_offset);
	qp->sq.sub_cq_idx = resp->efa_resp.send_sub_cq_idx;

	return 0;

err_unmap_desc_ring:
	if (munmap(qp->sq.desc - qp->sq.desc_offset, qp->sq.desc_ring_mmap_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: desc unmap failed!\n", qp->qp_num);
err_terminate_wq:
	efa_ep_wq_terminate(&qp->sq.wq);
	return -EINVAL;
}

static void efa_ep_sq_terminate(struct efa_qp *qp)
{
	void *db_aligned;

	if (!qp->sq.wq.wrid)
		return;

	db_aligned = (void *)((__u64)qp->sq.db & ~(qp->page_size - 1));
	if (munmap(db_aligned, qp->page_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: sq db unmap failed!\n", qp->qp_num);
	if (munmap(qp->sq.desc - qp->sq.desc_offset, qp->sq.desc_ring_mmap_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: desc data unmap failed!\n", qp->qp_num);

	efa_ep_wq_terminate(&qp->sq.wq);
}

static void efa_ep_rq_terminate(struct efa_qp *qp)
{
	void *db_aligned;

	if (!qp->rq.wq.wrid)
		return;

	db_aligned = (void *)((__u64)qp->rq.db & ~(qp->page_size - 1));
	if (munmap(db_aligned, qp->page_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: rq db unmap failed!\n", qp->qp_num);
	if (munmap(qp->rq.buf, qp->rq.buf_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: rq buffer unmap failed!\n", qp->qp_num);

	efa_ep_wq_terminate(&qp->rq.wq);
}

static int efa_ep_rq_initialize(struct efa_qp *qp, struct efa_create_qp_resp *resp, int fd)
{
	uint8_t *db_base;
	int err;

	if (!qp->rq.wq.wqe_cnt)
		return 0;

	err = efa_ep_wq_initialize(&qp->rq.wq);
	if (err)
		return err;

	qp->rq.buf_size = resp->efa_resp.rq_mmap_size;
	qp->rq.buf = mmap(NULL, qp->rq.buf_size, PROT_WRITE, MAP_SHARED, fd, resp->efa_resp.rq_mmap_key);
	if (qp->rq.buf == MAP_FAILED)
		goto err_terminate_wq;

	db_base = mmap(NULL, qp->page_size, PROT_WRITE, MAP_SHARED, fd, resp->efa_resp.rq_db_mmap_key);
	if (db_base == MAP_FAILED)
		goto err_unmap_rq_buf;
	qp->rq.db = (uint32_t *)(db_base + resp->efa_resp.rq_db_offset);
	qp->rq.sub_cq_idx = resp->efa_resp.recv_sub_cq_idx;

	return 0;

err_unmap_rq_buf:
	if (munmap(qp->rq.buf, qp->rq.buf_size))
		EFA_WARN(FI_LOG_EP_CTRL, "qp[%u]: rq buf unmap failed!\n", qp->qp_num);
err_terminate_wq:
	efa_ep_wq_terminate(&qp->rq.wq);
	return -EINVAL;
}

static void efa_ep_lock_cqs(struct ibv_qp *ibqp)
{
	struct efa_cq *send_cq = to_efa_cq(ibqp->send_cq);
	struct efa_cq *recv_cq = to_efa_cq(ibqp->recv_cq);

	if (recv_cq == send_cq && recv_cq) {
		fastlock_acquire(&recv_cq->inner_lock);
	} else {
		if (recv_cq)
			fastlock_acquire(&recv_cq->inner_lock);
		if (send_cq)
			fastlock_acquire(&send_cq->inner_lock);
	}
}

static void efa_ep_unlock_cqs(struct ibv_qp *ibqp)
{
	struct efa_cq *send_cq = to_efa_cq(ibqp->send_cq);
	struct efa_cq *recv_cq = to_efa_cq(ibqp->recv_cq);

	if (recv_cq == send_cq && recv_cq) {
		fastlock_release(&recv_cq->inner_lock);
	} else {
		if (recv_cq)
			fastlock_release(&recv_cq->inner_lock);
		if (send_cq)
			fastlock_release(&send_cq->inner_lock);
	}
}

static int efa_ep_destroy_qp(struct efa_qp *qp)
{
	struct efa_context *ctx;
	struct efa_cq *send_cq;
	struct efa_cq *recv_cq;
	struct ibv_qp *ibqp;
	int err;

	if (!qp)
		return 0;

	ibqp = &qp->ibv_qp;
	ctx = to_efa_ctx(ibqp->context);

	pthread_mutex_lock(&ctx->qp_table_mutex);
	efa_ep_lock_cqs(ibqp);

	if (ibqp->send_cq) {
		send_cq = to_efa_cq(ibqp->send_cq);
		efa_cq_dec_ref_cnt(send_cq, qp->sq.sub_cq_idx);
	}
	if (ibqp->recv_cq) {
		recv_cq = to_efa_cq(ibqp->recv_cq);
		efa_cq_dec_ref_cnt(recv_cq, qp->rq.sub_cq_idx);
	}
	ctx->qp_table[ibqp->qp_num] = NULL;

	efa_ep_unlock_cqs(ibqp);
	pthread_mutex_unlock(&ctx->qp_table_mutex);

	err = efa_cmd_destroy_qp(qp);
	if (err)
		EFA_INFO(FI_LOG_CORE, "destroy qp[%u] failed!\n", qp->qp_num);
	efa_ep_sq_terminate(qp);
	efa_ep_rq_terminate(qp);

	free(qp);
	return err;
}

static int efa_ep_create_qp(struct efa_ep *ep,
			    struct efa_pd *pd,
			    struct ibv_qp_init_attr *init_attr)
{
	struct ibv_pd *ibpd = &pd->ibv_pd;
	struct efa_device *dev = to_efa_dev(ibpd->context->device);
	struct efa_create_qp_resp resp;
	struct efa_cq *send_cq;
	struct efa_cq *recv_cq;
	struct efa_qp *qp;
	int err;

	qp = calloc(1, sizeof(*qp));
	if (!qp)
		return -FI_ENOMEM;

	efa_ep_setup_qp(qp, &init_attr->cap, dev->page_size);

	err = efa_cmd_create_qp(qp, pd, init_attr, ep->domain->rdm, &resp);
	if (err) {
		EFA_WARN(FI_LOG_EP_CTRL, "efa_cmd_create_qp failed [%u]!\n", err);
		goto err_free_qp;
	}

	qp->qp_num = qp->ibv_qp.qp_num;
	err = efa_ep_rq_initialize(qp, &resp, ibpd->context->cmd_fd);
	if (err)
		goto err_destroy_qp;

	err = efa_ep_sq_initialize(qp, &resp, ibpd->context->cmd_fd);
	if (err)
		goto err_terminate_rq;

	pthread_mutex_lock(&pd->context->qp_table_mutex);
	pd->context->qp_table[qp->qp_num] = qp;
	pthread_mutex_unlock(&pd->context->qp_table_mutex);

	if (init_attr->send_cq) {
		send_cq = to_efa_cq(init_attr->send_cq);
		fastlock_acquire(&send_cq->inner_lock);
		efa_cq_inc_ref_cnt(send_cq, resp.efa_resp.send_sub_cq_idx);
		fastlock_release(&send_cq->inner_lock);
	}
	if (init_attr->recv_cq) {
		recv_cq = to_efa_cq(init_attr->recv_cq);
		fastlock_acquire(&recv_cq->inner_lock);
		efa_cq_inc_ref_cnt(recv_cq, resp.efa_resp.recv_sub_cq_idx);
		fastlock_release(&recv_cq->inner_lock);
	}

	ep->qp = qp;
	qp->ep = ep;
	EFA_INFO(FI_LOG_EP_CTRL, "%s(): create QP %d\n", __func__, qp->qp_num);

	return 0;

err_terminate_rq:
	efa_ep_rq_terminate(qp);
err_destroy_qp:
	efa_cmd_destroy_qp(qp);
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
		av = container_of(bfid, struct efa_av, av_fid.fid);
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
	struct efa_ep *ep;
	struct efa_pd *pd;

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
		attr.send_cq = &ep->scq->ibv_cq;
		pd = ep->scq->domain->pd;
	} else {
		attr.send_cq = &ep->rcq->ibv_cq;
		pd = ep->rcq->domain->pd;
	}

	if (ep->rcq) {
		attr.cap.max_recv_wr = ep->info->rx_attr->size;
		attr.cap.max_recv_sge = ep->info->rx_attr->iov_limit;
		attr.recv_cq = &ep->rcq->ibv_cq;
	} else {
		attr.recv_cq = &ep->scq->ibv_cq;
	}

	attr.cap.max_inline_data = pd->context->inject_size;
	attr.qp_type = ep->domain->rdm ? IBV_QPT_DRIVER : IBV_QPT_UD;
	attr.sq_sig_all = 0;
	attr.qp_context = ep;

	return efa_ep_create_qp(ep, pd, &attr);
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
	    strncmp(domain->ctx->ibv_ctx.device->name, info->domain_attr->name,
		    strlen(domain->ctx->ibv_ctx.device->name))) {
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

	ep->domain = domain;
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
			goto err;
		}
		memcpy(ep->src_addr, info->src_addr, info->src_addrlen);
	}

	*ep_fid = &ep->ep_fid;

	return 0;

err:
	efa_ep_destroy(ep);
	return ret;
}
