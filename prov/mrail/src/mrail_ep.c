/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
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

#include "mrail.h"

#define MRAIL_DEFINE_GET_RAIL(txrx_rail)					\
static inline size_t mrail_get_ ## txrx_rail(struct mrail_ep *mrail_ep)		\
{										\
	return (ofi_atomic_inc32(&mrail_ep->txrx_rail) - 1) % mrail_ep->num_eps;\
}

MRAIL_DEFINE_GET_RAIL(tx_rail)
MRAIL_DEFINE_GET_RAIL(rx_rail)

static ssize_t mrail_recv(struct fid_ep *ep_fid, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr, void *context)
{
	struct mrail_ep *mrail_ep = container_of(ep_fid, struct mrail_ep,
					     util_ep.ep_fid.fid);
	uint32_t rail = mrail_get_rx_rail(mrail_ep);
	ssize_t ret;

	assert(!src_addr);

	ret = fi_recv(mrail_ep->eps[rail], buf, len, desc, 0, context);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_EP_DATA,
			"Unable to post recv on rail: %" PRIu32 "\n", rail);
		return ret;
	}
	return 0;
}

static ssize_t mrail_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
				uint64_t flags)
{
	struct mrail_ep *mrail_ep = container_of(ep_fid, struct mrail_ep,
					     util_ep.ep_fid.fid);
	struct fi_msg rail_msg = *msg;
	uint32_t rail = mrail_get_tx_rail(mrail_ep);
	ssize_t ret;

	rail_msg.addr = *(fi_addr_t *)ofi_av_get_addr(mrail_ep->util_ep.av,
						      (int)msg->addr);

	ret = fi_sendmsg(mrail_ep->eps[rail], &rail_msg, flags);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_EP_DATA,
			"Unable to post sendmsg on rail: %" PRIu32 "\n", rail);
		return ret;
	}
	return 0;
}

static ssize_t mrail_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, void *context)
{
	struct mrail_ep *mrail_ep = container_of(ep_fid, struct mrail_ep,
					     util_ep.ep_fid.fid);
	fi_addr_t *rail_fi_addr = ofi_av_get_addr(mrail_ep->util_ep.av,
						  (int)dest_addr);
	uint32_t rail = mrail_get_tx_rail(mrail_ep);
	ssize_t ret;

	assert(rail_fi_addr);

	ret = fi_send(mrail_ep->eps[rail], buf, len, desc, rail_fi_addr[rail],
		      context);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_EP_DATA,
			"Unable to post send on rail: %" PRIu32 "\n", rail);
		return ret;
	}
	return 0;
}

static ssize_t mrail_ep_inject(struct fid_ep *ep_fid, const void *buf,
			       size_t len, fi_addr_t dest_addr)
{
	struct mrail_ep *mrail_ep = container_of(ep_fid, struct mrail_ep,
					     util_ep.ep_fid.fid);
	fi_addr_t *rail_fi_addr = ofi_av_get_addr(mrail_ep->util_ep.av,
						  (int)dest_addr);
	uint32_t rail = mrail_get_tx_rail(mrail_ep);
	ssize_t ret;

	assert(rail_fi_addr);

	ret = fi_inject(mrail_ep->eps[rail], buf, len, rail_fi_addr[rail]);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_EP_DATA,
			"Unable to post send on rail: %" PRIu32 "\n", rail);
		return ret;
	}
	return 0;
}

static int mrail_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct mrail_ep *mrail_ep =
		container_of(fid, struct mrail_ep, util_ep.ep_fid.fid);
	struct mrail_domain *mrail_domain =
		container_of(mrail_ep->util_ep.domain, struct mrail_domain,
			     util_domain);
	size_t i, offset = 0, rail_addrlen;
	int ret;

	if (*addrlen < mrail_domain->addrlen)
		return -FI_ETOOSMALL;

	for (i = 0; i < mrail_ep->num_eps; i++) {
		rail_addrlen = *addrlen - offset;
		ret = fi_getname(&mrail_ep->eps[i]->fid, (char *)addr + offset,
				 &rail_addrlen);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_EP_CTRL,
				"Unable to get name for rail: %zd\n", i);
			return ret;
		}
		offset += rail_addrlen;
	}
	return 0;
}

static int mrail_ep_close(fid_t fid)
{
	struct mrail_ep *mrail_ep =
		container_of(fid, struct mrail_ep, util_ep.ep_fid.fid);
	int ret, retv = 0;

	ret = mrail_close_fids((struct fid **)mrail_ep->eps,
			       mrail_ep->num_eps);
	if (ret)
		retv = ret;
	free(mrail_ep->eps);

	ret = ofi_endpoint_close(&mrail_ep->util_ep);
	if (ret)
		retv = ret;
	free(mrail_ep);
	return retv;
}

static int mrail_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct mrail_ep *mrail_ep =
		container_of(ep_fid, struct mrail_ep, util_ep.ep_fid.fid);
	struct mrail_cq *mrail_cq;
	struct mrail_av *mrail_av;
	struct util_cntr *cntr;
	int ret = 0;
	size_t i;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		mrail_av = container_of(bfid, struct mrail_av,
					util_av.av_fid.fid);
		ret = ofi_ep_bind_av(&mrail_ep->util_ep, &mrail_av->util_av);
		if (ret)
			return ret;
		for (i = 0; i < mrail_ep->num_eps; i++) {
			ret = fi_ep_bind(mrail_ep->eps[i],
					 &mrail_av->avs[i]->fid, flags);
			if (ret)
				return ret;
		}
		break;
	case FI_CLASS_CQ:
		mrail_cq = container_of(bfid, struct mrail_cq,
					util_cq.cq_fid.fid);

		ret = ofi_ep_bind_cq(&mrail_ep->util_ep, &mrail_cq->util_cq,
				     flags);
		if (ret)
			return ret;
		for (i = 0; i < mrail_ep->num_eps; i++) {
			ret = fi_ep_bind(mrail_ep->eps[i],
					 &mrail_cq->cqs[i]->fid, flags);
			if (ret)
				return ret;
		}
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&mrail_ep->util_ep, cntr, flags);
		if (ret)
			return ret;
	case FI_CLASS_EQ:
		ret = -FI_ENOSYS;
		break;
	default:
		FI_WARN(&mrail_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int mrail_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct mrail_ep *mrail_ep;
	size_t i;
	int ret;

	mrail_ep = container_of(fid, struct mrail_ep, util_ep.ep_fid.fid);

	switch (command) {
	case FI_ENABLE:
		if (!mrail_ep->util_ep.rx_cq || !mrail_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!mrail_ep->util_ep.av)
			return -FI_ENOAV;
		for (i = 0; i < mrail_ep->num_eps; i++) {
			ret = fi_enable(mrail_ep->eps[i]);
			if (ret)
				return ret;
		}
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static struct fi_ops mrail_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mrail_ep_close,
	.bind = mrail_ep_bind,
	.control = mrail_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_ep mrail_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ops_cm mrail_ops_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = mrail_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static struct fi_ops_msg mrail_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = mrail_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = mrail_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = mrail_ep_sendmsg,
	.inject = mrail_ep_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

struct fi_ops_tagged mrail_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = fi_no_tagged_send,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

struct fi_ops_rma mrail_ops_rma = {
	.size = sizeof (struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = fi_no_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

int mrail_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct mrail_domain *mrail_domain =
		container_of(domain_fid, struct mrail_domain,
			     util_domain.domain_fid);
	struct mrail_ep *mrail_ep;
	struct fi_info *fi;
	size_t i;
	int ret;

	if (strcmp(mrail_domain->info->domain_attr->name,
		    info->domain_attr->name)) {
		FI_WARN(&mrail_prov, FI_LOG_EP_CTRL, "info domain name: %s "
			"doesn't match fid_domain name: %s!\n",
			info->domain_attr->name,
			mrail_domain->info->domain_attr->name);
		return -FI_EINVAL;
	}

	mrail_ep = calloc(1, sizeof(*mrail_ep));
	if (!mrail_ep)
		return -FI_ENOMEM;

	// TODO detect changes b/w mrail_domain->info and info arg
	// this may be difficult and we may not support such changes
	mrail_ep->info = mrail_domain->info;
	mrail_ep->num_eps = mrail_domain->num_domains;

	ret = ofi_endpoint_init(domain_fid, &mrail_util_prov, info, &mrail_ep->util_ep,
				context, NULL);
	if (ret) {
		free(mrail_ep);
		return ret;
	}

	if (!(mrail_ep->eps = calloc(mrail_ep->num_eps,
				     sizeof(*mrail_ep->eps)))) {
		ret = -FI_ENOMEM;
		goto err;
	}

	for (i = 0, fi = mrail_ep->info->next; fi; fi = fi->next, i++) {
		ret = fi_endpoint(mrail_domain->domains[i], fi,
				  &mrail_ep->eps[i], context);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_EP_CTRL,
				"Unable to open EP\n");
			goto err;
		}
	}

	ofi_atomic_initialize32(&mrail_ep->tx_rail, 0);
	ofi_atomic_initialize32(&mrail_ep->rx_rail, 0);

	*ep_fid = &mrail_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &mrail_ep_fi_ops;
	(*ep_fid)->ops = &mrail_ops_ep;
	(*ep_fid)->cm = &mrail_ops_cm;
	(*ep_fid)->msg = &mrail_ops_msg;
	(*ep_fid)->tagged = &mrail_ops_tagged;
	(*ep_fid)->rma = &mrail_ops_rma;

	return 0;
err:
	mrail_ep_close(&mrail_ep->util_ep.ep_fid.fid);
	return ret;
}
