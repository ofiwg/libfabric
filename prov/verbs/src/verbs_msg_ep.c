/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include "fi_verbs.h"


#define VERBS_CM_DATA_SIZE 56
#define VERBS_RESOLVE_TIMEOUT 2000	// ms


static int
fi_ibv_msg_ep_getopt(fid_t fid, int level, int optname,
		  void *optval, size_t *optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		switch (optname) {
		case FI_OPT_CM_DATA_SIZE:
			if (*optlen < sizeof(size_t))
				return -FI_ETOOSMALL;
			*((size_t *) optval) = VERBS_CM_DATA_SIZE;
			*optlen = sizeof(size_t);
			return 0;
		default:
			return -FI_ENOPROTOOPT;
		}
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static int
fi_ibv_msg_ep_setopt(fid_t fid, int level, int optname,
		  const void *optval, size_t optlen)
{
	switch (level) {
	case FI_OPT_ENDPOINT:
		return -FI_ENOPROTOOPT;
	default:
		return -FI_ENOPROTOOPT;
	}
	return 0;
}

static struct fi_ops_ep fi_ibv_msg_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_ibv_msg_ep_getopt,
	.setopt = fi_ibv_msg_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ibv_msg_ep *fi_ibv_alloc_msg_ep(struct fi_info *info)
{
	struct fi_ibv_msg_ep *ep;

	ep = calloc(1, sizeof *ep);
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

static void fi_ibv_free_msg_ep(struct fi_ibv_msg_ep *ep)
{
	if (ep->id)
		rdma_destroy_ep(ep->id);
	fi_freeinfo(ep->info);
	free(ep);
}

static int fi_ibv_msg_ep_close(fid_t fid)
{
	struct fi_ibv_msg_ep *ep;

	ep = container_of(fid, struct fi_ibv_msg_ep, ep_fid.fid);
	fi_ibv_free_msg_ep(ep);

	return 0;
}

static int fi_ibv_msg_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_msg_ep *ep;
	int ret;

	ep = container_of(fid, struct fi_ibv_msg_ep, ep_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		/* Must bind a CQ to either RECV or SEND completions, and
		 * the FI_SELECTIVE_COMPLETION flag is only valid when binding the
		 * FI_SEND CQ. */
		if (!(flags & (FI_RECV|FI_SEND))
				|| (flags & (FI_SEND|FI_SELECTIVE_COMPLETION))
							== FI_SELECTIVE_COMPLETION) {
			return -EINVAL;
		}
		if (flags & FI_RECV) {
			if (ep->rcq)
				return -EINVAL;
			ep->rcq = container_of(bfid, struct fi_ibv_cq, cq_fid.fid);
		}
		if (flags & FI_SEND) {
			if (ep->scq)
				return -EINVAL;
			ep->scq = container_of(bfid, struct fi_ibv_cq, cq_fid.fid);
			if (flags & FI_SELECTIVE_COMPLETION)
				ep->ep_flags |= FI_SELECTIVE_COMPLETION;
			else
				ep->info->tx_attr->op_flags |= FI_COMPLETION;
		}
		break;
	case FI_CLASS_EQ:
		ep->eq = container_of(bfid, struct fi_ibv_eq, eq_fid.fid);
		ret = rdma_migrate_id(ep->id, ep->eq->channel);
		if (ret)
			return -errno;
		break;
	default:
		return -EINVAL;
	}

	return 0;
}

static int fi_ibv_msg_ep_enable(struct fid_ep *ep)
{
	struct ibv_qp_init_attr attr;
	struct fi_ibv_msg_ep *_ep;

	_ep = container_of(ep, struct fi_ibv_msg_ep, ep_fid);
	if (!_ep->eq)
		return -FI_ENOEQ;
	if (!_ep->scq || !_ep->rcq)
		return -FI_ENOCQ;

	memset(&attr, 0, sizeof attr);
	attr.cap.max_send_wr = _ep->info->tx_attr->size;
	attr.cap.max_recv_wr = _ep->info->rx_attr->size;
	attr.cap.max_send_sge = _ep->info->tx_attr->iov_limit;
	attr.cap.max_recv_sge = _ep->info->rx_attr->iov_limit;
	attr.cap.max_inline_data = _ep->info->tx_attr->inject_size;

	attr.srq = NULL;
	attr.qp_type = IBV_QPT_RC;
	attr.sq_sig_all = 0;
	attr.qp_context = _ep;
	attr.send_cq = _ep->scq->cq;
	attr.recv_cq = _ep->rcq->cq;

	return rdma_create_qp(_ep->id, _ep->rcq->domain->pd, &attr) ?
		-errno : 0;
}

static int fi_ibv_msg_ep_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		ep = container_of(fid, struct fid_ep, fid);
		switch (command) {
		case FI_ENABLE:
			return fi_ibv_msg_ep_enable(ep);
			break;
		default:
			return -FI_ENOSYS;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
}

static struct fi_ops fi_ibv_msg_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_msg_ep_close,
	.bind = fi_ibv_msg_ep_bind,
	.control = fi_ibv_msg_ep_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_open_ep(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep, void *context)
{
	struct fi_ibv_domain *dom;
	struct fi_ibv_msg_ep *_ep;
	struct fi_ibv_connreq *connreq;
	struct fi_ibv_pep *pep;
	struct fi_info *fi;
	int ret;

	dom = container_of(domain, struct fi_ibv_domain, domain_fid);
	if (strcmp(dom->verbs->device->name, info->domain_attr->name)) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Invalid info->domain_attr->name\n");
		return -FI_EINVAL;
	}

	fi = fi_ibv_get_verbs_info(info->domain_attr->name);
	if (!fi) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to find matching verbs_info\n");
		return -FI_EINVAL;
	}

	if (info->ep_attr) {
		ret = fi_ibv_check_ep_attr(info->ep_attr, fi);
		if (ret)
			return ret;
	}

	if (info->tx_attr) {
		ret = fi_ibv_check_tx_attr(info->tx_attr, info, fi);
		if (ret)
			return ret;
	}

	if (info->rx_attr) {
		ret = fi_ibv_check_rx_attr(info->rx_attr, info, fi);
		if (ret)
			return ret;
	}

	_ep = fi_ibv_alloc_msg_ep(info);
	if (!_ep)
		return -FI_ENOMEM;

	if (!info->handle) {
		ret = fi_ibv_create_ep(NULL, NULL, 0, info, NULL, &_ep->id);
		if (ret)
			goto err;
	} else if (info->handle->fclass == FI_CLASS_CONNREQ) {
		connreq = container_of(info->handle, struct fi_ibv_connreq, handle);
		_ep->id = connreq->id;
        } else if (info->handle->fclass == FI_CLASS_PEP) {
		pep = container_of(info->handle, struct fi_ibv_pep, pep_fid.fid);
		_ep->id = pep->id;
		pep->id = NULL;

		if (rdma_resolve_addr(_ep->id, info->src_addr, info->dest_addr,
				      VERBS_RESOLVE_TIMEOUT)) {
			ret = -errno;
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to rdma_resolve_addr\n");
			goto err;
		}

		if (rdma_resolve_route(_ep->id, VERBS_RESOLVE_TIMEOUT)) {
			ret = -errno;
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to rdma_resolve_route\n");
			goto err;
		}
	} else {
		ret = -FI_ENOSYS;
		goto err;
	}

	_ep->id->context = &_ep->ep_fid.fid;

	_ep->ep_fid.fid.fclass = FI_CLASS_EP;
	_ep->ep_fid.fid.context = context;
	_ep->ep_fid.fid.ops = &fi_ibv_msg_ep_ops;
	_ep->ep_fid.ops = &fi_ibv_msg_ep_base_ops;
	_ep->ep_fid.msg = fi_ibv_msg_ep_ops_msg(_ep);
	_ep->ep_fid.cm = fi_ibv_msg_ep_ops_cm(_ep);
	_ep->ep_fid.rma = fi_ibv_msg_ep_ops_rma(_ep);
	_ep->ep_fid.atomic = fi_ibv_msg_ep_ops_atomic(_ep);

	*ep = &_ep->ep_fid;

	return 0;
err:
	fi_ibv_free_msg_ep(_ep);
	return ret;
}

static int fi_ibv_pep_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_pep *pep;
	int ret;

	pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;

	pep->eq = container_of(bfid, struct fi_ibv_eq, eq_fid.fid);
	ret = rdma_migrate_id(pep->id, pep->eq->channel);
	if (ret)
		return -errno;

	return 0;
}

static int fi_ibv_pep_control(struct fid *fid, int command, void *arg)
{
	struct fi_ibv_pep *pep;
	int ret = 0;

	switch (fid->fclass) {
	case FI_CLASS_PEP:
		pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
		switch (command) {
		case FI_BACKLOG:
			if (!arg)
				return -FI_EINVAL;
			pep->backlog = *(int *) arg;
			break;
		default:
			ret = -FI_ENOSYS;
			break;
		}
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_pep_close(fid_t fid)
{
	struct fi_ibv_pep *pep;

	pep = container_of(fid, struct fi_ibv_pep, pep_fid.fid);
	if (pep->id)
		rdma_destroy_ep(pep->id);

	free(pep);
	return 0;
}

static struct fi_ops fi_ibv_pep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_pep_close,
	.bind = fi_ibv_pep_bind,
	.control = fi_ibv_pep_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		      struct fid_pep **pep, void *context)
{
	struct fi_ibv_pep *_pep;
	int ret;

	_pep = calloc(1, sizeof *_pep);
	if (!_pep)
		return -FI_ENOMEM;

	ret = rdma_create_id(NULL, &_pep->id, NULL, RDMA_PS_TCP);
	if (ret) {
		FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to create rdma_cm_id\n");
		goto err1;
	}

	if (info->src_addr) {
		ret = rdma_bind_addr(_pep->id, (struct sockaddr *)info->src_addr);
		if (ret) {
			FI_INFO(&fi_ibv_prov, FI_LOG_DOMAIN, "Unable to bind addres to rdma_cm_id\n");
			goto err2;
		}
		_pep->bound = 1;
	}

	_pep->id->context = &_pep->pep_fid.fid;

	_pep->pep_fid.fid.fclass = FI_CLASS_PEP;
	_pep->pep_fid.fid.context = context;
	_pep->pep_fid.fid.ops = &fi_ibv_pep_ops;
	_pep->pep_fid.cm = fi_ibv_pep_ops_cm(_pep);

	_pep->src_addrlen = info->src_addrlen;

	*pep = &_pep->pep_fid;
	return 0;

err2:
	rdma_destroy_id(_pep->id);
err1:
	free(_pep);
	return ret;
}

