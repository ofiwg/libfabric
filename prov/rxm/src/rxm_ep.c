/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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

#include "rxm.h"


int rxm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_setname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

int rxm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_getname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

static struct fi_ops_cm rxm_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxm_setname,
	.getname = rxm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

int rxm_getopt(fid_t fid, int level, int optname,
		void *optval, size_t *optlen)
{
	return -FI_ENOPROTOOPT;
}

int rxm_setopt(fid_t fid, int level, int optname,
		const void *optval, size_t optlen)
{
	return -FI_ENOPROTOOPT;
}

static struct fi_ops_ep rxm_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = rxm_getopt,
	.setopt = rxm_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

ssize_t rxm_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_recvmsg(rxm_ep->srx_ctx, msg, flags);
}

ssize_t rxm_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_recvv(rxm_ep->srx_ctx, iov, desc, count, 0, context);
}

ssize_t rxm_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep;

	/* TODO Handle recv from particular src */
	if (src_addr) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"Unable to post recv for a particular source\n");
		return -FI_EINVAL;
	}

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_recv(rxm_ep->srx_ctx, buf, len, desc, 0, context);
}

ssize_t rxm_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct rxm_ep *rxm_ep;
	struct fid_ep *msg_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&rxm_ep->cmap->lock);
	ret = rxm_get_msg_ep(rxm_ep, dest_addr, &msg_ep);
	if (ret)
		goto unlock;

	// TODO handle the case when send fails due to connection shutdown
	ret = fi_send(msg_ep, buf, len, desc, 0, context);
unlock:
	fastlock_release(&rxm_ep->cmap->lock);
	return ret;
}

ssize_t rxm_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct rxm_ep *rxm_ep;
	struct fid_ep *msg_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&rxm_ep->cmap->lock);
	ret = rxm_get_msg_ep(rxm_ep, msg->addr, &msg_ep);
	if (ret)
		goto unlock;

	ret = fi_sendmsg(msg_ep, msg, flags);
unlock:
	fastlock_release(&rxm_ep->cmap->lock);
	return ret;
}

ssize_t rxm_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	struct rxm_ep *rxm_ep;
	struct fid_ep *msg_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&rxm_ep->cmap->lock);
	ret = rxm_get_msg_ep(rxm_ep, dest_addr, &msg_ep);
	if (ret)
		goto unlock;

	ret = fi_sendv(msg_ep, iov, desc, count, 0, context);
unlock:
	fastlock_release(&rxm_ep->cmap->lock);
	return ret;
}

ssize_t rxm_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	struct rxm_ep *rxm_ep;
	struct fid_ep *msg_ep;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&rxm_ep->cmap->lock);
	ret = rxm_get_msg_ep(rxm_ep, dest_addr, &msg_ep);
	if (ret)
		goto unlock;

	ret = fi_inject(msg_ep, buf, len, 0);
unlock:
	fastlock_release(&rxm_ep->cmap->lock);
	return ret;
}

static struct fi_ops_msg rxm_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxm_recv,
	.recvv = rxm_recvv,
	.recvmsg = rxm_recvmsg,
	.send = rxm_send,
	.sendv = rxm_sendv,
	.sendmsg = rxm_sendmsg,
	.inject = rxm_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static int rxm_ep_close(struct fid *fid)
{
	struct rxm_ep *rxm_ep;
	int ret, retv = 0;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);

	if (rxm_ep->util_ep.av) {
		ofi_cmap_free(rxm_ep->cmap);
		atomic_dec(&rxm_ep->util_ep.av->ref);
	}

	ret = fi_close(&rxm_ep->srx_ctx->fid);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg shared ctx\n");
		retv = ret;
	}

	ret = fi_close(&rxm_ep->msg_pep->fid);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg passive EP\n");
		retv = ret;
	}

	fi_freeinfo(rxm_ep->msg_info);

	if (rxm_ep->util_ep.rx_cq)
		atomic_dec(&rxm_ep->util_ep.rx_cq->ref);

	if (rxm_ep->util_ep.tx_cq)
		atomic_dec(&rxm_ep->util_ep.tx_cq->ref);

	atomic_dec(&rxm_ep->util_ep.domain->ref);
	free(rxm_ep);
	return retv;
}

static int rxm_ep_bind_cq(struct rxm_ep *rxm_ep, struct util_cq *util_cq, uint64_t flags)
{
	struct rxm_cq *rxm_cq;

	rxm_cq = container_of(util_cq, struct rxm_cq, util_cq);

	if (flags & ~(FI_TRANSMIT | FI_RECV)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "unsupported flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && rxm_ep->tx_cq) ||
	    ((flags & FI_RECV) && rxm_ep->rx_cq)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "duplicate CQ binding\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		rxm_ep->util_ep.tx_cq = &rxm_cq->util_cq;
		atomic_inc(&rxm_cq->util_cq.ref);
	}

	if (flags & FI_RECV) {
		rxm_ep->util_ep.rx_cq = &rxm_cq->util_cq;
		atomic_inc(&rxm_cq->util_cq.ref);
	}
	return 0;
}

int rxm_ep_bind_av(struct rxm_ep *rxm_ep, struct util_av *av)
{
	if (rxm_ep->util_ep.av) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"duplicate AV binding\n");
		return -FI_EINVAL;
	}
	rxm_ep->cmap = ofi_cmap_alloc(av, rxm_conn_close);
	if (!rxm_ep->cmap)
		return -FI_ENOMEM;

	atomic_inc(&av->ref);
	rxm_ep->util_ep.av = av;
	return 0;
}

static int rxm_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxm_ep *rxm_ep;
	int ret = 0;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		ret = rxm_ep_bind_av(rxm_ep, container_of(bfid, struct util_av,
					av_fid.fid));
		break;
	case FI_CLASS_CQ:
		ret = rxm_ep_bind_cq(rxm_ep, container_of(bfid, struct util_cq,
					cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int rxm_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct rxm_ep *rxm_ep;
	struct rxm_fabric *rxm_fabric;
	int ret;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	rxm_fabric = container_of(rxm_ep->util_ep.domain->fabric,
			struct rxm_fabric, util_fabric);
	switch (command) {
	case FI_ENABLE:
		if (!rxm_ep->util_ep.rx_cq || !rxm_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!rxm_ep->util_ep.av)
			return -FI_EOPBADSTATE; /* TODO: Add FI_ENOAV */
		ret = fi_pep_bind(rxm_ep->msg_pep, &rxm_fabric->msg_eq->fid, 0);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to bind msg PEP to msg EQ\n");
			return ret;
		}
		ret = fi_listen(rxm_ep->msg_pep);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to set msg PEP to listen state\n");
			return ret;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static struct fi_ops rxm_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_ep_close,
	.bind = rxm_ep_bind,
	.control = rxm_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int rxm_ep_msg_res_open(struct fi_info *rxm_info,
		struct util_domain *util_domain, struct rxm_ep *rxm_ep)
{
	struct rxm_fabric *rxm_fabric;
	struct rxm_domain *rxm_domain;
	int ret;

	ret = ofix_getinfo(rxm_prov.version, NULL, NULL, 0, &rxm_util_prov,
			rxm_info, rxm_alter_layer_info, rxm_alter_base_info,
			1, &rxm_ep->msg_info);
	if (ret)
		return ret;

	rxm_domain = container_of(util_domain, struct rxm_domain, util_domain);
	rxm_fabric = container_of(util_domain->fabric, struct rxm_fabric, util_fabric);

	ret = fi_passive_ep(rxm_fabric->msg_fabric, rxm_ep->msg_info, &rxm_ep->msg_pep, rxm_ep);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to open msg PEP\n");
		goto err1;
	}

	ret = fi_srx_context(rxm_domain->msg_domain, rxm_ep->msg_info->rx_attr,
			&rxm_ep->srx_ctx, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_FABRIC, "Unable to open shared receive context\n");
		goto err2;
	}

	/* We don't care what's in the dest_addr at this point. We go by AV. */
	if (rxm_ep->msg_info->dest_addr) {
		free(rxm_ep->msg_info->dest_addr);
		rxm_ep->msg_info->dest_addr = NULL;
		rxm_ep->msg_info->dest_addrlen = 0;
	}

	/* Zero out the port as we would be creating multiple MSG EPs for a single
	 * RXM EP and we don't want address conflicts. */
	// TODO handle other address types?
	if (rxm_ep->msg_info->src_addr)
		((struct sockaddr_in *)(rxm_ep->msg_info->src_addr))->sin_port = 0;

	return 0;
err2:
	fi_close(&rxm_ep->msg_pep->fid);
err1:
	fi_freeinfo(rxm_ep->msg_info);
	return ret;
}

int rxm_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct util_domain *util_domain;
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = calloc(1, sizeof(*rxm_ep));
	if (!rxm_ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &rxm_util_prov, info, &rxm_ep->util_ep,
			context, FI_MATCH_PREFIX);
	if (ret)
		goto err;

	util_domain = container_of(domain, struct util_domain, domain_fid);

	ret = rxm_ep_msg_res_open(info, util_domain, rxm_ep);
	if (ret)
		goto err;

	*ep_fid = &rxm_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &rxm_ep_fi_ops;
	(*ep_fid)->ops = &rxm_ep_ops;
	(*ep_fid)->cm = &rxm_cm_ops;
	(*ep_fid)->msg = &rxm_msg_ops;

	return 0;
err:
	free(rxm_ep);
	return ret;
}
