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

static int mrail_ep_close(fid_t fid)
{
	struct mrail_ep *mrail_ep =
		container_of(fid, struct mrail_ep, util_ep.ep_fid.fid);

	ofi_endpoint_close(&mrail_ep->util_ep);
	free(mrail_ep);
	//return 0;
	return -FI_ENOSYS;
}

static int mrail_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct mrail_ep *mrail_ep =
		container_of(ep_fid, struct mrail_ep, util_ep.ep_fid.fid);
	struct util_cq *cq;
	struct util_av *av;
	struct util_cntr *cntr;
	int ret = 0;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&mrail_ep->util_ep, av);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&mrail_ep->util_ep, cq, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&mrail_ep->util_ep, cntr, flags);
		if (ret)
			return ret;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&mrail_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	//return ret;
	return -FI_ENOSYS;
}

static int mrail_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct mrail_ep *mrail_ep;

	mrail_ep = container_of(fid, struct mrail_ep, util_ep.ep_fid.fid);

	switch (command) {
	case FI_ENABLE:
		if (!mrail_ep->util_ep.rx_cq || !mrail_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!mrail_ep->util_ep.av)
			return -FI_EOPBADSTATE;
		break;
	default:
		return -FI_ENOSYS;
	}
	//return 0;
	return -FI_ENOSYS;
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
	.getname = fi_no_getname,
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
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
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

void mrail_ep_progress(struct util_ep *util_ep)
{
	//struct mrail_ep *mrail_ep =
	//	container_of(util_ep, struct mrail_ep, util_ep);
}

int mrail_ep_open(struct fid_domain *domain, struct fi_info *info,
		   struct fid_ep **ep_fid, void *context)
{
	struct mrail_ep *mrail_ep;
	int ret;

	mrail_ep = calloc(1, sizeof(*mrail_ep));
	if (!mrail_ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &mrail_util_prov, info, &mrail_ep->util_ep,
				context, &mrail_ep_progress);
	if (ret)
		goto err1;

	*ep_fid = &mrail_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &mrail_ep_fi_ops;
	(*ep_fid)->ops = &mrail_ops_ep;
	(*ep_fid)->cm = &mrail_ops_cm;
	(*ep_fid)->msg = &mrail_ops_msg;
	(*ep_fid)->tagged = &mrail_ops_tagged;
	(*ep_fid)->rma = &mrail_ops_rma;

	//return 0;
	return -FI_ENOSYS;
err1:
	free(mrail_ep);
	return ret;
}
