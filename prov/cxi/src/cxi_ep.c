/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "ofi_util.h"
#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

extern struct fi_ops_rma cxi_ep_rma;
extern struct fi_ops_msg cxi_ep_msg_ops;
extern struct fi_ops_tagged cxi_ep_tagged;
extern struct fi_ops_atomic cxi_ep_atomic;

struct fi_ops_rma cxi_ep_rma = {0};
struct fi_ops_msg cxi_ep_msg_ops = {0};
struct fi_ops_tagged cxi_ep_tagged = {0};
struct fi_ops_atomic cxi_ep_atomic = {0};

extern struct fi_ops_ep cxi_ep_ops;
extern struct fi_ops cxi_ep_fi_ops;
extern struct fi_ops_ep cxi_ctx_ep_ops;

extern const struct fi_domain_attr cxi_domain_attr;
extern const struct fi_fabric_attr cxi_fabric_attr;

const struct fi_tx_attr cxi_stx_attr = {
	.caps = CXI_EP_RDM_CAP_BASE,
	.mode = CXI_MODE,
	.op_flags = FI_TRANSMIT_COMPLETE,
	.msg_order = CXI_EP_MSG_ORDER,
	.inject_size = CXI_EP_MAX_INJECT_SZ,
	.size = CXI_EP_TX_SZ,
	.iov_limit = CXI_EP_MAX_IOV_LIMIT,
	.rma_iov_limit = CXI_EP_MAX_IOV_LIMIT,
};

const struct fi_rx_attr cxi_srx_attr = {
	.caps = CXI_EP_RDM_CAP_BASE,
	.mode = CXI_MODE,
	.op_flags = 0,
	.msg_order = CXI_EP_MSG_ORDER,
	.comp_order = CXI_EP_COMP_ORDER,
	.total_buffered_recv = 0,
	.size = CXI_EP_MAX_MSG_SZ,
	.iov_limit = CXI_EP_MAX_IOV_LIMIT,
};

static void cxi_tx_ctx_close(struct cxi_tx_ctx *tx_ctx)
{
	if (tx_ctx->comp.send_cq)
		cxi_cq_remove_tx_ctx(tx_ctx->comp.send_cq, tx_ctx);

	if (tx_ctx->comp.send_cntr)
		cxi_cntr_remove_tx_ctx(tx_ctx->comp.send_cntr, tx_ctx);

	if (tx_ctx->comp.read_cntr)
		cxi_cntr_remove_tx_ctx(tx_ctx->comp.read_cntr, tx_ctx);

	if (tx_ctx->comp.write_cntr)
		cxi_cntr_remove_tx_ctx(tx_ctx->comp.write_cntr, tx_ctx);
}

static void cxi_rx_ctx_close(struct cxi_rx_ctx *rx_ctx)
{
	if (rx_ctx->comp.recv_cq)
		cxi_cq_remove_rx_ctx(rx_ctx->comp.recv_cq, rx_ctx);

	if (rx_ctx->comp.recv_cntr)
		cxi_cntr_remove_rx_ctx(rx_ctx->comp.recv_cntr, rx_ctx);

	if (rx_ctx->comp.rem_read_cntr)
		cxi_cntr_remove_rx_ctx(rx_ctx->comp.rem_read_cntr, rx_ctx);

	if (rx_ctx->comp.rem_write_cntr)
		cxi_cntr_remove_rx_ctx(rx_ctx->comp.rem_write_cntr, rx_ctx);
}

static int cxi_ctx_close(struct fid *fid)
{
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	switch (fid->fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx = container_of(fid, struct cxi_tx_ctx, fid.ctx.fid);
		ofi_atomic_dec32(&tx_ctx->ep_attr->num_tx_ctx);
		ofi_atomic_dec32(&tx_ctx->domain->ref);
		cxi_tx_ctx_close(tx_ctx);
		cxi_tx_ctx_free(tx_ctx);
		break;

	case FI_CLASS_RX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		ofi_atomic_dec32(&rx_ctx->ep_attr->num_rx_ctx);
		ofi_atomic_dec32(&rx_ctx->domain->ref);
		cxi_rx_ctx_close(rx_ctx);
		cxi_rx_ctx_free(rx_ctx);
		break;

	case FI_CLASS_STX_CTX:
		tx_ctx = container_of(fid, struct cxi_tx_ctx, fid.stx.fid);
		ofi_atomic_dec32(&tx_ctx->domain->ref);
		cxi_tx_ctx_free(tx_ctx);
		break;

	case FI_CLASS_SRX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		ofi_atomic_dec32(&rx_ctx->domain->ref);
		cxi_rx_ctx_free(rx_ctx);
		break;

	default:
		CXI_LOG_ERROR("Invalid fid\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int cxi_ctx_bind_cq(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_cq *cxi_cq;
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	if ((flags | CXI_EP_CQ_FLAGS) != CXI_EP_CQ_FLAGS) {
		CXI_LOG_ERROR("Invalid cq flag\n");
		return -FI_EINVAL;
	}
	cxi_cq = container_of(bfid, struct cxi_cq, cq_fid.fid);
	switch (fid->fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx = container_of(fid, struct cxi_tx_ctx, fid.ctx);
		if (flags & FI_SEND) {
			tx_ctx->comp.send_cq = cxi_cq;
			if (flags & FI_SELECTIVE_COMPLETION)
				tx_ctx->comp.send_cq_event = 1;
		}

		cxi_cq_add_tx_ctx(cxi_cq, tx_ctx);
		break;

	case FI_CLASS_RX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		if (flags & FI_RECV) {
			rx_ctx->comp.recv_cq = cxi_cq;
			if (flags & FI_SELECTIVE_COMPLETION)
				rx_ctx->comp.recv_cq_event = 1;
		}

		cxi_cq_add_rx_ctx(cxi_cq, rx_ctx);
		break;

	default:
		CXI_LOG_ERROR("Invalid fid\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int cxi_ctx_bind_cntr(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_cntr *cntr;
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	if ((flags | CXI_EP_CNTR_FLAGS) != CXI_EP_CNTR_FLAGS) {
		CXI_LOG_ERROR("Invalid cntr flag\n");
		return -FI_EINVAL;
	}

	cntr = container_of(bfid, struct cxi_cntr, cntr_fid.fid);
	switch (fid->fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx = container_of(fid, struct cxi_tx_ctx, fid.ctx.fid);
		if (flags & FI_SEND) {
			tx_ctx->comp.send_cntr = cntr;
			cxi_cntr_add_tx_ctx(cntr, tx_ctx);
		}

		if (flags & FI_READ) {
			tx_ctx->comp.read_cntr = cntr;
			cxi_cntr_add_tx_ctx(cntr, tx_ctx);
		}

		if (flags & FI_WRITE) {
			tx_ctx->comp.write_cntr = cntr;
			cxi_cntr_add_tx_ctx(cntr, tx_ctx);
		}
		break;

	case FI_CLASS_RX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		if (flags & FI_RECV) {
			rx_ctx->comp.recv_cntr = cntr;
			cxi_cntr_add_rx_ctx(cntr, rx_ctx);
		}

		if (flags & FI_REMOTE_READ) {
			rx_ctx->comp.rem_read_cntr = cntr;
			cxi_cntr_add_rx_ctx(cntr, rx_ctx);
		}

		if (flags & FI_REMOTE_WRITE) {
			rx_ctx->comp.rem_write_cntr = cntr;
			cxi_cntr_add_rx_ctx(cntr, rx_ctx);
		}
		break;

	default:
		CXI_LOG_ERROR("Invalid fid\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int cxi_ctx_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		return cxi_ctx_bind_cq(fid, bfid, flags);

	case FI_CLASS_CNTR:
		return cxi_ctx_bind_cntr(fid, bfid, flags);

	case FI_CLASS_MR:
		return 0;

	default:
		CXI_LOG_ERROR("Invalid bind()\n");
		return -FI_EINVAL;
	}
}

static int cxi_ctx_enable(struct fid_ep *ep)
{
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	switch (ep->fid.fclass) {
	case FI_CLASS_RX_CTX:
		rx_ctx = container_of(ep, struct cxi_rx_ctx, ctx.fid);
		rx_ctx->enabled = 1;
		return 0;

	case FI_CLASS_TX_CTX:
		tx_ctx = container_of(ep, struct cxi_tx_ctx, fid.ctx.fid);
		tx_ctx->enabled = 1;
		return 0;

	default:
		CXI_LOG_ERROR("Invalid CTX\n");
		break;
	}

	return -FI_EINVAL;
}

int cxi_getopflags(struct fi_tx_attr *tx_attr, struct fi_rx_attr *rx_attr,
			uint64_t *flags)
{
	if ((*flags & FI_TRANSMIT) && (*flags & FI_RECV)) {
		CXI_LOG_ERROR("Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (*flags & FI_TRANSMIT)) {
		*flags = tx_attr->op_flags;
	} else if (rx_attr && (*flags & FI_RECV)) {
		*flags = rx_attr->op_flags;
	} else {
		CXI_LOG_ERROR("Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

int cxi_setopflags(struct fi_tx_attr *tx_attr, struct fi_rx_attr *rx_attr,
			uint64_t flags)
{
	if ((flags & FI_TRANSMIT) && (flags & FI_RECV)) {
		CXI_LOG_ERROR("Both Tx/Rx flags cannot be specified\n");
		return -FI_EINVAL;
	} else if (tx_attr && (flags & FI_TRANSMIT)) {
		tx_attr->op_flags = flags;
		tx_attr->op_flags &= ~FI_TRANSMIT;
		if (!(flags & (FI_INJECT_COMPLETE | FI_TRANSMIT_COMPLETE |
			       FI_DELIVERY_COMPLETE)))
			tx_attr->op_flags |= FI_TRANSMIT_COMPLETE;
	} else if (rx_attr && (flags & FI_RECV)) {
		rx_attr->op_flags = flags;
		rx_attr->op_flags &= ~FI_RECV;
	} else {
		CXI_LOG_ERROR("Tx/Rx flags not specified\n");
		return -FI_EINVAL;
	}

	return 0;
}

static int cxi_ctx_control(struct fid *fid, int command, void *arg)
{
	struct fid_ep *ep;
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;
	int ret;

	switch (fid->fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx = container_of(fid, struct cxi_tx_ctx, fid.ctx.fid);
		switch (command) {
		case FI_GETOPSFLAG:
			ret = cxi_getopflags(&tx_ctx->attr, NULL,
					     (uint64_t *) arg);
			if (ret)
				return -FI_EINVAL;
			break;
		case FI_SETOPSFLAG:
			ret = cxi_setopflags(&tx_ctx->attr, NULL,
					     *(uint64_t *) arg);
			if (ret)
				return -FI_EINVAL;
			break;
		case FI_ENABLE:
			ep = container_of(fid, struct fid_ep, fid);
			return cxi_ctx_enable(ep);
			break;
		default:
			return -FI_ENOSYS;
		}
		break;

	case FI_CLASS_RX_CTX:
	case FI_CLASS_SRX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		switch (command) {
		case FI_GETOPSFLAG:
			ret = cxi_getopflags(NULL, &rx_ctx->attr,
					     (uint64_t *) arg);
			if (ret)
				return -FI_EINVAL;
			break;
		case FI_SETOPSFLAG:
			ret = cxi_setopflags(NULL, &rx_ctx->attr,
					     *(uint64_t *) arg);
			if (ret)
				return -FI_EINVAL;
			break;
		case FI_ENABLE:
			ep = container_of(fid, struct fid_ep, fid);
			return cxi_ctx_enable(ep);
			break;
		default:
			return -FI_ENOSYS;
		}
		break;

	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static struct fi_ops cxi_ctx_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_ctx_close,
	.bind = cxi_ctx_bind,
	.control = cxi_ctx_control,
	.ops_open = fi_no_ops_open,
};

static int cxi_ctx_getopt(fid_t fid, int level, int optname,
			  void *optval, size_t *optlen)
{
	struct cxi_rx_ctx *rx_ctx;

	rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		if (*optlen < sizeof(size_t))
			return -FI_ETOOSMALL;
		*(size_t *)optval = rx_ctx->min_multi_recv;
		*optlen = sizeof(size_t);
		break;
	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static int cxi_ctx_setopt(fid_t fid, int level, int optname,
			  const void *optval, size_t optlen)
{
	struct cxi_rx_ctx *rx_ctx;

	rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:
		rx_ctx->min_multi_recv = *(size_t *)optval;
		break;
	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static ssize_t cxi_rx_ctx_cancel(struct cxi_rx_ctx *rx_ctx, void *context)
{
	return FI_SUCCESS;
}

static ssize_t cxi_ep_cancel(fid_t fid, void *context)
{
	struct cxi_rx_ctx *rx_ctx = NULL;
	struct cxi_ep *cxi_ep;

	switch (fid->fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(fid, struct cxi_ep, ep.fid);
		rx_ctx = cxi_ep->attr->rx_ctx;
		break;

	case FI_CLASS_RX_CTX:
	case FI_CLASS_SRX_CTX:
		rx_ctx = container_of(fid, struct cxi_rx_ctx, ctx.fid);
		break;

	case FI_CLASS_TX_CTX:
	case FI_CLASS_STX_CTX:
		return -FI_ENOENT;

	default:
		CXI_LOG_ERROR("Invalid ep type\n");
		return -FI_EINVAL;
	}

	return cxi_rx_ctx_cancel(rx_ctx, context);
}

struct fi_ops_ep cxi_ctx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = cxi_ep_cancel,
	.getopt = cxi_ctx_getopt,
	.setopt = cxi_ctx_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int cxi_ep_close(struct fid *fid)
{
	struct cxi_ep *cxi_ep;

	switch (fid->fclass) {
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		cxi_ep = container_of(fid, struct cxi_ep, ep.fid);
		break;

	default:
		return -FI_EINVAL;
	}

	if (cxi_ep->is_alias) {
		ofi_atomic_dec32(&cxi_ep->attr->ref);
		return 0;
	}
	if (ofi_atomic_get32(&cxi_ep->attr->ref) ||
	    ofi_atomic_get32(&cxi_ep->attr->num_rx_ctx) ||
	    ofi_atomic_get32(&cxi_ep->attr->num_tx_ctx))
		return -FI_EBUSY;

	if (cxi_ep->attr->av) {
		ofi_atomic_dec32(&cxi_ep->attr->av->ref);

		fastlock_acquire(&cxi_ep->attr->av->list_lock);
		fid_list_remove(&cxi_ep->attr->av->ep_list,
				&cxi_ep->attr->lock, &cxi_ep->ep.fid);
		fastlock_release(&cxi_ep->attr->av->list_lock);
	}

	if (cxi_ep->attr->tx_shared) {
		fastlock_acquire(&cxi_ep->attr->tx_ctx->lock);
		dlist_remove(&cxi_ep->attr->tx_ctx_entry);
		fastlock_release(&cxi_ep->attr->tx_ctx->lock);
	}

	if (cxi_ep->attr->rx_shared) {
		fastlock_acquire(&cxi_ep->attr->rx_ctx->lock);
		dlist_remove(&cxi_ep->attr->rx_ctx_entry);
		fastlock_release(&cxi_ep->attr->rx_ctx->lock);
	}

	if (cxi_ep->attr->fclass != FI_CLASS_SEP) {
		cxi_tx_ctx_close(cxi_ep->attr->tx_array[0]);
		cxi_tx_ctx_free(cxi_ep->attr->tx_array[0]);
	}

	if (cxi_ep->attr->fclass != FI_CLASS_SEP) {
		cxi_rx_ctx_close(cxi_ep->attr->rx_array[0]);
		cxi_rx_ctx_free(cxi_ep->attr->rx_array[0]);
	}

	free(cxi_ep->attr->tx_array);
	free(cxi_ep->attr->rx_array);

	if (cxi_ep->attr->src_addr)
		free(cxi_ep->attr->src_addr);

	ofi_atomic_dec32(&cxi_ep->attr->domain->ref);
	fastlock_destroy(&cxi_ep->attr->lock);
	free(cxi_ep->attr);
	free(cxi_ep);

	return 0;
}

static int cxi_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	int ret;
	size_t i;
	struct cxi_ep *ep;
	struct cxi_eq *eq;
	struct cxi_cq *cq;
	struct cxi_av *av;
	struct cxi_cntr *cntr;
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	ret = ofi_ep_bind_valid(&cxi_prov, bfid, flags);
	if (ret)
		return ret;

	switch (fid->fclass) {
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		ep = container_of(fid, struct cxi_ep, ep.fid);
		break;

	default:
		return -FI_EINVAL;
	}

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct cxi_eq, eq.fid);
		ep->attr->eq = eq;
		break;

	case FI_CLASS_CQ:
		cq = container_of(bfid, struct cxi_cq, cq_fid.fid);
		if (ep->attr->domain != cq->domain)
			return -FI_EINVAL;

		if (flags & FI_SEND) {
			for (i = 0; i < ep->attr->ep_attr.tx_ctx_cnt; i++) {
				tx_ctx = ep->attr->tx_array[i];

				if (!tx_ctx)
					continue;

				ret = cxi_ctx_bind_cq(&tx_ctx->fid.ctx.fid,
						      bfid, flags);
				if (ret)
					return ret;
			}
		}

		if (flags & FI_RECV) {
			for (i = 0; i < ep->attr->ep_attr.rx_ctx_cnt; i++) {
				rx_ctx = ep->attr->rx_array[i];

				if (!rx_ctx)
					continue;

				ret = cxi_ctx_bind_cq(&rx_ctx->ctx.fid, bfid,
						      flags);
				if (ret)
					return ret;
			}
		}
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxi_cntr, cntr_fid.fid);
		if (ep->attr->domain != cntr->domain)
			return -FI_EINVAL;

		if (flags & FI_SEND || flags & FI_WRITE || flags & FI_READ) {
			for (i = 0; i < ep->attr->ep_attr.tx_ctx_cnt; i++) {
				tx_ctx = ep->attr->tx_array[i];

				if (!tx_ctx)
					continue;

				ret = cxi_ctx_bind_cntr(&tx_ctx->fid.ctx.fid,
							bfid, flags);
				if (ret)
					return ret;
			}
		}

		if (flags & FI_RECV || flags & FI_REMOTE_READ ||
		    flags & FI_REMOTE_WRITE) {
			for (i = 0; i < ep->attr->ep_attr.rx_ctx_cnt; i++) {
				rx_ctx = ep->attr->rx_array[i];

				if (!rx_ctx)
					continue;

				ret = cxi_ctx_bind_cntr(&rx_ctx->ctx.fid, bfid,
							flags);
				if (ret)
					return ret;
			}
		}
		break;

	case FI_CLASS_AV:
		av = container_of(bfid, struct cxi_av, av_fid.fid);
		if (ep->attr->domain != av->domain)
			return -FI_EINVAL;

		ep->attr->av = av;
		ofi_atomic_inc32(&av->ref);

		if (ep->attr->tx_ctx &&
		    ep->attr->tx_ctx->fid.ctx.fid.fclass == FI_CLASS_TX_CTX) {
			ep->attr->tx_ctx->av = av;
		}

		if (ep->attr->rx_ctx &&
		    ep->attr->rx_ctx->ctx.fid.fclass == FI_CLASS_RX_CTX)
			ep->attr->rx_ctx->av = av;

		for (i = 0; i < ep->attr->ep_attr.tx_ctx_cnt; i++) {
			if (ep->attr->tx_array[i])
				ep->attr->tx_array[i]->av = av;
		}

		for (i = 0; i < ep->attr->ep_attr.rx_ctx_cnt; i++) {
			if (ep->attr->rx_array[i])
				ep->attr->rx_array[i]->av = av;
		}
		fastlock_acquire(&av->list_lock);
		ret = fid_list_insert(&av->ep_list, &ep->attr->lock,
				      &ep->ep.fid);
		fastlock_release(&av->list_lock);
		if (ret) {
			CXI_LOG_ERROR("Error in adding fid in the EP list\n");
			return ret;
		}
		break;

	case FI_CLASS_STX_CTX:
		tx_ctx = container_of(bfid, struct cxi_tx_ctx, fid.stx.fid);
		fastlock_acquire(&tx_ctx->lock);
		dlist_insert_tail(&ep->attr->tx_ctx_entry, &tx_ctx->ep_list);
		fastlock_release(&tx_ctx->lock);

		ep->attr->tx_ctx->use_shared = 1;
		ep->attr->tx_ctx->stx_ctx = tx_ctx;
		break;

	case FI_CLASS_SRX_CTX:
		rx_ctx = container_of(bfid, struct cxi_rx_ctx, ctx);
		fastlock_acquire(&rx_ctx->lock);
		dlist_insert_tail(&ep->attr->rx_ctx_entry, &rx_ctx->ep_list);
		fastlock_release(&rx_ctx->lock);

		ep->attr->rx_ctx->use_shared = 1;
		ep->attr->rx_ctx->srx_ctx = rx_ctx;
		break;

	default:
		return -FI_EINVAL;
	}

	return 0;
}

static int cxi_ep_control(struct fid *fid, int command, void *arg)
{
	int ret;
	struct fid_ep *ep_fid;
	struct fi_alias *alias;
	struct cxi_ep *cxi_ep, *new_ep;

	switch (fid->fclass) {
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		cxi_ep = container_of(fid, struct cxi_ep, ep.fid);
		break;

	default:
		return -FI_EINVAL;
	}

	switch (command) {
	case FI_ALIAS:
		if (!arg)
			return -FI_EINVAL;
		alias = (struct fi_alias *)arg;
		if (!alias->fid)
			return -FI_EINVAL;
		new_ep = calloc(1, sizeof(*new_ep));
		if (!new_ep)
			return -FI_ENOMEM;

		memcpy(&new_ep->tx_attr, &cxi_ep->tx_attr,
		       sizeof(struct fi_tx_attr));
		memcpy(&new_ep->rx_attr, &cxi_ep->rx_attr,
		       sizeof(struct fi_rx_attr));
		ret = cxi_setopflags(&new_ep->tx_attr, &new_ep->rx_attr,
				     alias->flags);
		if (ret) {
			free(new_ep);
			return -FI_EINVAL;
		}
		new_ep->attr = cxi_ep->attr;
		new_ep->is_alias = 1;
		memcpy(&new_ep->ep, &cxi_ep->ep, sizeof(struct fid_ep));
		*alias->fid = &new_ep->ep.fid;
		ofi_atomic_inc32(&new_ep->attr->ref);
		break;
	case FI_GETOPSFLAG:
		if (!arg)
			return -FI_EINVAL;
		ret = cxi_getopflags(&cxi_ep->tx_attr, &cxi_ep->rx_attr,
				     (uint64_t *) arg);
		if (ret)
			return -FI_EINVAL;
		break;
	case FI_SETOPSFLAG:
		if (!arg)
			return -FI_EINVAL;
		ret = cxi_setopflags(&cxi_ep->tx_attr, &cxi_ep->rx_attr,
				     *(uint64_t *) arg);
		if (ret)
			return -FI_EINVAL;
		break;
	case FI_ENABLE:
		ep_fid = container_of(fid, struct fid_ep, fid);
		return cxi_ep_enable(ep_fid);

	default:
		return -FI_EINVAL;
	}
	return 0;
}


struct fi_ops cxi_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_ep_close,
	.bind = cxi_ep_bind,
	.control = cxi_ep_control,
	.ops_open = fi_no_ops_open,
};

int cxi_ep_enable(struct fid_ep *ep)
{
	size_t i;
	int ret;
	struct cxi_ep *cxi_ep;
	struct cxi_tx_ctx *tx_ctx;
	struct cxi_rx_ctx *rx_ctx;

	cxi_ep = container_of(ep, struct cxi_ep, ep);

	ret = cxix_domain_enable(cxi_ep->attr->domain);
	if (ret != FI_SUCCESS) {
		CXI_LOG_DBG("cxix_domain_enable returned: %d\n", ret);
		return ret;
	}

	for (i = 0; i < cxi_ep->attr->ep_attr.tx_ctx_cnt; i++) {
		tx_ctx = cxi_ep->attr->tx_array[i];
		if (tx_ctx) {
			ret = cxix_tx_ctx_enable(tx_ctx);
			if (ret != FI_SUCCESS) {
				CXI_LOG_DBG("cxix_tx_ctx_enable returned: %d\n",
					    ret);
				return ret;
			}

			if (tx_ctx->use_shared) {
				if (tx_ctx->stx_ctx) {
					tx_ctx->stx_ctx->enabled = 1;
				}
			}
		}
	}

	for (i = 0; i < cxi_ep->attr->ep_attr.rx_ctx_cnt; i++) {
		rx_ctx = cxi_ep->attr->rx_array[i];
		if (rx_ctx) {
			rx_ctx->enabled = 1;
			if (rx_ctx->use_shared) {
				if (rx_ctx->srx_ctx) {
					rx_ctx->srx_ctx->enabled = 1;
				}
			}
		}
	}

	cxi_ep->attr->is_enabled = 1;

	return 0;
}

int cxi_ep_disable(struct fid_ep *ep)
{
	size_t i;
	struct cxi_ep *cxi_ep;

	cxi_ep = container_of(ep, struct cxi_ep, ep);

	if (cxi_ep->attr->tx_ctx &&
	    cxi_ep->attr->tx_ctx->fid.ctx.fid.fclass == FI_CLASS_TX_CTX) {
		cxi_ep->attr->tx_ctx->enabled = 0;
	}

	if (cxi_ep->attr->rx_ctx &&
	    cxi_ep->attr->rx_ctx->ctx.fid.fclass == FI_CLASS_RX_CTX) {
		cxi_ep->attr->rx_ctx->enabled = 0;
	}

	for (i = 0; i < cxi_ep->attr->ep_attr.tx_ctx_cnt; i++) {
		if (cxi_ep->attr->tx_array[i])
			cxi_ep->attr->tx_array[i]->enabled = 0;
	}

	for (i = 0; i < cxi_ep->attr->ep_attr.rx_ctx_cnt; i++) {
		if (cxi_ep->attr->rx_array[i])
			cxi_ep->attr->rx_array[i]->enabled = 0;
	}
	cxi_ep->attr->is_enabled = 0;

	return 0;
}

static int cxi_ep_getopt(fid_t fid, int level, int optname,
			 void *optval, size_t *optlen)
{
	struct cxi_ep *cxi_ep;

	cxi_ep = container_of(fid, struct cxi_ep, ep.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:

		if (!optval || !optlen)
			return -FI_EINVAL;

		*(size_t *)optval = cxi_ep->attr->min_multi_recv;
		*optlen = sizeof(size_t);
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static int cxi_ep_setopt(fid_t fid, int level, int optname,
			 const void *optval, size_t optlen)
{
	size_t i;
	struct cxi_ep *cxi_ep;

	cxi_ep = container_of(fid, struct cxi_ep, ep.fid);

	if (level != FI_OPT_ENDPOINT)
		return -FI_ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_MIN_MULTI_RECV:

		if (!optval)
			return -FI_EINVAL;

		cxi_ep->attr->min_multi_recv = *(size_t *)optval;
		for (i = 0; i < cxi_ep->attr->ep_attr.rx_ctx_cnt; i++) {
			if (cxi_ep->attr->rx_array[i] != NULL) {
				cxi_ep->attr->rx_array[i]->min_multi_recv =
					cxi_ep->attr->min_multi_recv;
			}
		}
		break;

	default:
		return -FI_ENOPROTOOPT;
	}

	return 0;
}

static int cxi_ep_tx_ctx(struct fid_ep *ep, int index, struct fi_tx_attr *attr,
			 struct fid_ep **tx_ep, void *context)
{
	struct cxi_ep *cxi_ep;
	struct cxi_tx_ctx *tx_ctx;

	cxi_ep = container_of(ep, struct cxi_ep, ep);
	if (cxi_ep->attr->fclass != FI_CLASS_SEP ||
	    index >= (int)cxi_ep->attr->ep_attr.tx_ctx_cnt ||
	    !tx_ep)
		return -FI_EINVAL;

	if (attr) {
		if (ofi_check_tx_attr(&cxi_prov, cxi_ep->attr->info.tx_attr,
				      attr, 0) ||
			ofi_check_attr_subset(&cxi_prov,
				cxi_ep->attr->info.tx_attr->caps, attr->caps))
			return -FI_ENODATA;
		tx_ctx = cxi_tx_ctx_alloc(attr, context, 0);
	} else {
		tx_ctx = cxi_tx_ctx_alloc(&cxi_ep->tx_attr, context, 0);
	}
	if (!tx_ctx)
		return -FI_ENOMEM;

	tx_ctx->tx_id = index;
	tx_ctx->ep_attr = cxi_ep->attr;
	tx_ctx->domain = cxi_ep->attr->domain;
	tx_ctx->av = cxi_ep->attr->av;
	dlist_insert_tail(&cxi_ep->attr->tx_ctx_entry, &tx_ctx->ep_list);

	tx_ctx->fid.ctx.fid.ops = &cxi_ctx_ops;
	tx_ctx->fid.ctx.ops = &cxi_ctx_ep_ops;
	tx_ctx->fid.ctx.msg = &cxi_ep_msg_ops;
	tx_ctx->fid.ctx.tagged = &cxi_ep_tagged;
	tx_ctx->fid.ctx.rma = &cxi_ep_rma;
	tx_ctx->fid.ctx.atomic = &cxi_ep_atomic;

	*tx_ep = &tx_ctx->fid.ctx;
	cxi_ep->attr->tx_array[index] = tx_ctx;
	ofi_atomic_inc32(&cxi_ep->attr->num_tx_ctx);
	ofi_atomic_inc32(&cxi_ep->attr->domain->ref);

	return 0;
}

static int cxi_ep_rx_ctx(struct fid_ep *ep, int index, struct fi_rx_attr *attr,
			 struct fid_ep **rx_ep, void *context)
{
	struct cxi_ep *cxi_ep;
	struct cxi_rx_ctx *rx_ctx;

	cxi_ep = container_of(ep, struct cxi_ep, ep);
	if (cxi_ep->attr->fclass != FI_CLASS_SEP ||
	    index >= (int)cxi_ep->attr->ep_attr.rx_ctx_cnt ||
	    !rx_ep)
		return -FI_EINVAL;

	if (attr) {
		if (ofi_check_rx_attr(&cxi_prov, &cxi_ep->attr->info, attr, 0)
		    || ofi_check_attr_subset(&cxi_prov,
					     cxi_ep->attr->info.rx_attr->caps,
					     attr->caps))
			return -FI_ENODATA;
		rx_ctx = cxi_rx_ctx_alloc(attr, context, 0);
	} else {
		rx_ctx = cxi_rx_ctx_alloc(&cxi_ep->rx_attr, context, 0);
	}
	if (!rx_ctx)
		return -FI_ENOMEM;

	rx_ctx->rx_id = index;
	rx_ctx->ep_attr = cxi_ep->attr;
	rx_ctx->domain = cxi_ep->attr->domain;
	rx_ctx->av = cxi_ep->attr->av;
	dlist_insert_tail(&cxi_ep->attr->rx_ctx_entry, &rx_ctx->ep_list);

	rx_ctx->ctx.fid.ops = &cxi_ctx_ops;
	rx_ctx->ctx.ops = &cxi_ctx_ep_ops;
	rx_ctx->ctx.msg = &cxi_ep_msg_ops;
	rx_ctx->ctx.tagged = &cxi_ep_tagged;

	rx_ctx->min_multi_recv = cxi_ep->attr->min_multi_recv;
	*rx_ep = &rx_ctx->ctx;
	cxi_ep->attr->rx_array[index] = rx_ctx;
	ofi_atomic_inc32(&cxi_ep->attr->num_rx_ctx);
	ofi_atomic_inc32(&cxi_ep->attr->domain->ref);

	return 0;
}

struct fi_ops_ep cxi_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = cxi_ep_cancel,
	.getopt = cxi_ep_getopt,
	.setopt = cxi_ep_setopt,
	.tx_ctx = cxi_ep_tx_ctx,
	.rx_ctx = cxi_ep_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int cxi_verify_tx_attr(const struct fi_tx_attr *attr)
{
	if (!attr)
		return 0;

	if (attr->inject_size > CXI_EP_MAX_INJECT_SZ)
		return -FI_ENODATA;

	if (attr->size > CXI_EP_TX_SZ)
		return -FI_ENODATA;

	if (attr->iov_limit > CXI_EP_MAX_IOV_LIMIT)
		return -FI_ENODATA;

	if (attr->rma_iov_limit > CXI_EP_MAX_IOV_LIMIT)
		return -FI_ENODATA;

	return 0;
}

int cxi_stx_ctx(struct fid_domain *domain, struct fi_tx_attr *attr,
		struct fid_stx **stx, void *context)
{
	struct cxi_domain *dom;
	struct cxi_tx_ctx *tx_ctx;

	if ((attr && cxi_verify_tx_attr(attr)) ||
	    !stx)
		return -FI_EINVAL;

	dom = container_of(domain, struct cxi_domain, dom_fid);

	tx_ctx = cxi_stx_ctx_alloc(attr ? attr : &cxi_stx_attr, context);
	if (!tx_ctx)
		return -FI_ENOMEM;

	tx_ctx->domain = dom;

	tx_ctx->fid.stx.fid.ops = &cxi_ctx_ops;
	tx_ctx->fid.stx.ops = &cxi_ep_ops;
	ofi_atomic_inc32(&dom->ref);

	*stx = &tx_ctx->fid.stx;

	return 0;
}

static int cxi_verify_rx_attr(const struct fi_rx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->msg_order | CXI_EP_MSG_ORDER) != CXI_EP_MSG_ORDER)
		return -FI_ENODATA;

	if ((attr->comp_order | CXI_EP_COMP_ORDER) != CXI_EP_COMP_ORDER)
		return -FI_ENODATA;

	if (attr->total_buffered_recv > CXI_EP_MAX_BUFF_RECV)
		return -FI_ENODATA;

	if (attr->size > CXI_EP_TX_SZ)
		return -FI_ENODATA;

	if (attr->iov_limit > CXI_EP_MAX_IOV_LIMIT)
		return -FI_ENODATA;

	return 0;
}

int cxi_srx_ctx(struct fid_domain *domain, struct fi_rx_attr *attr,
		struct fid_ep **srx, void *context)
{
	struct cxi_domain *dom;
	struct cxi_rx_ctx *rx_ctx;

	if ((attr && cxi_verify_rx_attr(attr)) ||
	    !srx)
		return -FI_EINVAL;

	dom = container_of(domain, struct cxi_domain, dom_fid);
	rx_ctx = cxi_rx_ctx_alloc(attr ? attr : &cxi_srx_attr, context, 0);
	if (!rx_ctx)
		return -FI_ENOMEM;

	rx_ctx->domain = dom;
	rx_ctx->ctx.fid.fclass = FI_CLASS_SRX_CTX;

	rx_ctx->ctx.fid.ops = &cxi_ctx_ops;
	rx_ctx->ctx.ops = &cxi_ctx_ep_ops;
	rx_ctx->ctx.msg = &cxi_ep_msg_ops;
	rx_ctx->ctx.tagged = &cxi_ep_tagged;
	rx_ctx->enabled = 1;

	/* default config */
	rx_ctx->min_multi_recv = CXI_EP_MIN_MULTI_RECV;
	*srx = &rx_ctx->ctx;
	ofi_atomic_inc32(&dom->ref);

	return 0;
}

static void cxi_set_fabric_attr(void *src_addr,
				const struct fi_fabric_attr *hint_attr,
				struct fi_fabric_attr *attr)
{
	struct cxi_fabric *fabric;

	*attr = cxi_fabric_attr;
	if (hint_attr && hint_attr->fabric) {
		attr->fabric = hint_attr->fabric;
	} else {
		fabric = cxi_fab_list_head();
		attr->fabric = fabric ? &fabric->fab_fid : NULL;
	}

	attr->name = cxi_get_fabric_name(src_addr);
	attr->prov_name = NULL;
}

static void cxi_set_domain_attr(uint32_t api_version, void *src_addr,
				const struct fi_domain_attr *hint_attr,
				struct fi_domain_attr *attr)
{
	struct cxi_domain *domain;

	domain = cxi_dom_list_head();
	attr->domain = domain ? &domain->dom_fid : NULL;
	if (!hint_attr) {
		*attr = cxi_domain_attr;

		if (FI_VERSION_LT(api_version, FI_VERSION(1, 5)))
			attr->mr_mode = FI_MR_SCALABLE;
		goto out;
	}

	if (hint_attr->domain) {
		domain = container_of(hint_attr->domain,
				      struct cxi_domain, dom_fid);
		*attr = domain->attr;
		attr->domain = hint_attr->domain;
		goto out;
	}

	*attr = *hint_attr;
	if (attr->threading == FI_THREAD_UNSPEC)
		attr->threading = cxi_domain_attr.threading;
	if (attr->control_progress == FI_PROGRESS_UNSPEC)
		attr->control_progress = cxi_domain_attr.control_progress;
	if (attr->data_progress == FI_PROGRESS_UNSPEC)
		attr->data_progress = cxi_domain_attr.data_progress;
	if (FI_VERSION_LT(api_version, FI_VERSION(1, 5))) {
		if (attr->mr_mode == FI_MR_UNSPEC)
			attr->mr_mode = FI_MR_SCALABLE;
	} else {
		if ((attr->mr_mode != FI_MR_BASIC) &&
		    (attr->mr_mode != FI_MR_SCALABLE))
			attr->mr_mode = 0;
	}

	if (attr->cq_cnt == 0)
		attr->cq_cnt = cxi_domain_attr.cq_cnt;
	if (attr->ep_cnt == 0)
		attr->ep_cnt = cxi_domain_attr.ep_cnt;
	if (attr->tx_ctx_cnt == 0)
		attr->tx_ctx_cnt = cxi_domain_attr.tx_ctx_cnt;
	if (attr->rx_ctx_cnt == 0)
		attr->rx_ctx_cnt = cxi_domain_attr.rx_ctx_cnt;
	if (attr->max_ep_tx_ctx == 0)
		attr->max_ep_tx_ctx = cxi_domain_attr.max_ep_tx_ctx;
	if (attr->max_ep_rx_ctx == 0)
		attr->max_ep_rx_ctx = cxi_domain_attr.max_ep_rx_ctx;
	if (attr->max_ep_stx_ctx == 0)
		attr->max_ep_stx_ctx = cxi_domain_attr.max_ep_stx_ctx;
	if (attr->max_ep_srx_ctx == 0)
		attr->max_ep_srx_ctx = cxi_domain_attr.max_ep_srx_ctx;
	if (attr->cntr_cnt == 0)
		attr->cntr_cnt = cxi_domain_attr.cntr_cnt;
	if (attr->mr_iov_limit == 0)
		attr->mr_iov_limit = cxi_domain_attr.mr_iov_limit;

	attr->mr_key_size = cxi_domain_attr.mr_key_size;
	attr->cq_data_size = cxi_domain_attr.cq_data_size;
	attr->resource_mgmt = cxi_domain_attr.resource_mgmt;
out:
	/* reverse lookup interface from node and assign it as domain name */
	attr->name = cxi_get_domain_name(src_addr);
}


struct fi_info *cxi_fi_info(uint32_t version, enum fi_ep_type ep_type,
			    const struct fi_info *hints, void *src_addr,
			    void *dest_addr)
{
	struct fi_info *info;

	info = fi_allocinfo();
	if (!info)
		return NULL;

	info->src_addr = calloc(1, sizeof(struct cxi_addr));
	if (!info->src_addr)
		goto err;

	info->mode = CXI_MODE;
	info->addr_format = FI_ADDR_CXI;

	if (src_addr)
		memcpy(info->src_addr, src_addr, sizeof(struct cxi_addr));
	else
		cxi_get_src_addr(NULL, info->src_addr);
	info->src_addrlen = sizeof(struct cxi_addr);

	if (dest_addr) {
		info->dest_addr = calloc(1, sizeof(struct cxi_addr));
		if (!info->dest_addr)
			goto err;
		info->dest_addrlen = sizeof(struct cxi_addr);
		memcpy(info->dest_addr, dest_addr, sizeof(struct cxi_addr));
	}

	if (hints) {
		if (hints->caps)
			info->caps = hints->caps;

		if (hints->ep_attr)
			*(info->ep_attr) = *(hints->ep_attr);

		if (hints->tx_attr)
			*(info->tx_attr) = *(hints->tx_attr);

		if (hints->rx_attr)
			*(info->rx_attr) = *(hints->rx_attr);

		if (hints->handle)
			info->handle = hints->handle;

		cxi_set_domain_attr(version, info->src_addr, hints->domain_attr,
				    info->domain_attr);
		cxi_set_fabric_attr(info->src_addr, hints->fabric_attr,
				    info->fabric_attr);
	} else {
		cxi_set_domain_attr(version, info->src_addr, NULL,
				    info->domain_attr);
		cxi_set_fabric_attr(info->src_addr, NULL, info->fabric_attr);
	}

	info->ep_attr->type = ep_type;
	return info;
err:
	fi_freeinfo(info);
	return NULL;
}

static int cxi_ep_assign_src_addr(struct cxi_ep *cxi_ep, struct fi_info *info)
{
	cxi_ep->attr->src_addr = calloc(1, sizeof(struct cxi_addr));
	if (!cxi_ep->attr->src_addr)
		return -FI_ENOMEM;

	if (info)
		return cxi_get_src_addr(info->dest_addr,
					cxi_ep->attr->src_addr);

	return -FI_EINVAL;
}

int cxi_alloc_endpoint(struct fid_domain *domain, struct fi_info *info,
		       struct cxi_ep **ep, void *context, size_t fclass)
{
	int ret;
	struct cxi_ep *cxi_ep;
	struct cxi_tx_ctx *tx_ctx = NULL;
	struct cxi_rx_ctx *rx_ctx = NULL;
	struct cxi_domain *cxi_dom;

	cxi_dom = container_of(domain, struct cxi_domain, dom_fid);
	if (info) {
		ret = cxi_verify_info(cxi_dom->fab->fab_fid.api_version, info);
		if (ret) {
			CXI_LOG_DBG("Cannot support requested options!\n");
			return -FI_EINVAL;
		}
	}

	cxi_ep = calloc(1, sizeof(*cxi_ep));
	if (!cxi_ep)
		return -FI_ENOMEM;

	switch (fclass) {
	case FI_CLASS_EP:
		cxi_ep->ep.fid.fclass = FI_CLASS_EP;
		cxi_ep->ep.fid.context = context;
		cxi_ep->ep.fid.ops = &cxi_ep_fi_ops;

		cxi_ep->ep.ops = &cxi_ep_ops;
		cxi_ep->ep.msg = &cxi_ep_msg_ops;
		cxi_ep->ep.rma = &cxi_ep_rma;
		cxi_ep->ep.tagged = &cxi_ep_tagged;
		cxi_ep->ep.atomic = &cxi_ep_atomic;
		break;

	case FI_CLASS_SEP:
		cxi_ep->ep.fid.fclass = FI_CLASS_SEP;
		cxi_ep->ep.fid.context = context;
		cxi_ep->ep.fid.ops = &cxi_ep_fi_ops;

		cxi_ep->ep.ops = &cxi_ep_ops;
		break;

	default:
		ret = -FI_EINVAL;
		goto err;
	}

	cxi_ep->attr = calloc(1, sizeof(struct cxi_ep_attr));
	if (!cxi_ep->attr) {
		ret = -FI_ENOMEM;
		goto err;
	}
	cxi_ep->attr->fclass = fclass;
	*ep = cxi_ep;

	if (info) {
		cxi_ep->attr->info.caps = info->caps;
		cxi_ep->attr->info.addr_format = FI_ADDR_CXI;

		if (info->ep_attr) {
			cxi_ep->attr->ep_type = info->ep_attr->type;
			cxi_ep->attr->ep_attr.tx_ctx_cnt =
					info->ep_attr->tx_ctx_cnt;
			cxi_ep->attr->ep_attr.rx_ctx_cnt =
					info->ep_attr->rx_ctx_cnt;
		}

		if (info->src_addr) {
			cxi_ep->attr->src_addr =
					calloc(1, sizeof(struct cxi_addr));
			if (!cxi_ep->attr->src_addr) {
				ret = -FI_ENOMEM;
				goto err;
			}
			memcpy(cxi_ep->attr->src_addr, info->src_addr,
			       sizeof(struct cxi_addr));
		}

		if (info->tx_attr) {
			cxi_ep->tx_attr = *info->tx_attr;
			if (!(cxi_ep->tx_attr.op_flags & (FI_INJECT_COMPLETE |
			     FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)))
				cxi_ep->tx_attr.op_flags |=
						FI_TRANSMIT_COMPLETE;
			cxi_ep->tx_attr.size = cxi_ep->tx_attr.size ?
				cxi_ep->tx_attr.size : CXI_EP_TX_SZ;
		}

		if (info->rx_attr)
			cxi_ep->rx_attr = *info->rx_attr;
		cxi_ep->attr->info.handle = info->handle;
	}

	if (!cxi_ep->attr->src_addr && cxi_ep_assign_src_addr(cxi_ep, info)) {
		CXI_LOG_ERROR("failed to get src_address\n");
		ret = -FI_EINVAL;
		goto err;
	}

	ofi_atomic_initialize32(&cxi_ep->attr->ref, 0);
	ofi_atomic_initialize32(&cxi_ep->attr->num_tx_ctx, 0);
	ofi_atomic_initialize32(&cxi_ep->attr->num_rx_ctx, 0);
	fastlock_init(&cxi_ep->attr->lock);

	if (cxi_ep->attr->ep_attr.tx_ctx_cnt == FI_SHARED_CONTEXT)
		cxi_ep->attr->tx_shared = 1;
	if (cxi_ep->attr->ep_attr.rx_ctx_cnt == FI_SHARED_CONTEXT)
		cxi_ep->attr->rx_shared = 1;

	if (cxi_ep->attr->fclass != FI_CLASS_SEP) {
		cxi_ep->attr->ep_attr.tx_ctx_cnt = 1;
		cxi_ep->attr->ep_attr.rx_ctx_cnt = 1;
	}

	cxi_ep->attr->tx_array = calloc(cxi_ep->attr->ep_attr.tx_ctx_cnt,
					sizeof(struct cxi_tx_ctx *));
	if (!cxi_ep->attr->tx_array) {
		ret = -FI_ENOMEM;
		goto err;
	}

	cxi_ep->attr->rx_array = calloc(cxi_ep->attr->ep_attr.rx_ctx_cnt,
					sizeof(struct cxi_rx_ctx *));
	if (!cxi_ep->attr->rx_array) {
		ret = -FI_ENOMEM;
		goto err;
	}

	if (cxi_ep->attr->fclass != FI_CLASS_SEP) {
		/* default tx ctx */
		tx_ctx = cxi_tx_ctx_alloc(&cxi_ep->tx_attr, context,
					  cxi_ep->attr->tx_shared);
		if (!tx_ctx) {
			ret = -FI_ENOMEM;
			goto err;
		}
		tx_ctx->ep_attr = cxi_ep->attr;
		tx_ctx->domain = cxi_dom;
		tx_ctx->tx_id = 0;
		dlist_insert_tail(&cxi_ep->attr->tx_ctx_entry,
				  &tx_ctx->ep_list);
		cxi_ep->attr->tx_array[0] = tx_ctx;
		cxi_ep->attr->tx_ctx = tx_ctx;

		/* default rx_ctx */
		rx_ctx = cxi_rx_ctx_alloc(&cxi_ep->rx_attr, context,
					   cxi_ep->attr->rx_shared);
		if (!rx_ctx) {
			ret = -FI_ENOMEM;
			goto err;
		}
		rx_ctx->ep_attr = cxi_ep->attr;
		rx_ctx->domain = cxi_dom;
		rx_ctx->rx_id = 0;
		dlist_insert_tail(&cxi_ep->attr->rx_ctx_entry,
				  &rx_ctx->ep_list);
		cxi_ep->attr->rx_array[0] = rx_ctx;
		cxi_ep->attr->rx_ctx = rx_ctx;
	}

	/* default config */
	cxi_ep->attr->min_multi_recv = CXI_EP_MIN_MULTI_RECV;

	if (info)
		memcpy(&cxi_ep->attr->info, info, sizeof(struct fi_info));

	cxi_ep->attr->domain = cxi_dom;

	ofi_atomic_inc32(&cxi_dom->ref);
	return 0;

err:
	if (rx_ctx)
		cxi_rx_ctx_free(rx_ctx);
	if (tx_ctx)
		cxi_tx_ctx_free(tx_ctx);
	if (cxi_ep->attr) {
		if (cxi_ep->attr->rx_array)
			free(cxi_ep->attr->rx_array);
		if (cxi_ep->attr->tx_array)
			free(cxi_ep->attr->tx_array);
		if (cxi_ep->attr->src_addr)
			free(cxi_ep->attr->src_addr);
		free(cxi_ep->attr);
	}
	if (cxi_ep)
		free(cxi_ep);
	return ret;
}

