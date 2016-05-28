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

static struct fi_ops_ep fi_ibv_srq_ep_base_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static struct fi_ops_cm fi_ibv_srq_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

static struct fi_ops_rma fi_ibv_srq_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
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

static struct fi_ops_atomic fi_ibv_srq_atomic_ops = {
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

static ssize_t
fi_ibv_srq_ep_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct fi_ibv_srq_ep *_ep;
	struct ibv_recv_wr wr, *bad;
	struct ibv_sge *sge = NULL;
	ssize_t ret;
	size_t i;

	_ep = container_of(ep, struct fi_ibv_srq_ep, ep_fid);
	assert(_ep->srq);

	wr.wr_id = (uintptr_t) msg->context;
	wr.next = NULL;
	if (msg->iov_count) {
		sge = alloca(sizeof(*sge) * msg->iov_count);
		for (i = 0; i < msg->iov_count; i++) {
			sge[i].addr = (uintptr_t) msg->msg_iov[i].iov_base;
			sge[i].length = (uint32_t) msg->msg_iov[i].iov_len;
			sge[i].lkey = (uint32_t) (uintptr_t) (msg->desc[i]);
		}

	}
	wr.sg_list = sge;
	wr.num_sge = msg->iov_count;

	ret = ibv_post_srq_recv(_ep->srq, &wr, &bad);

	switch (ret) {
	case ENOMEM:
		return -FI_EAGAIN;
	case -1:
		/* Deal with non-compliant libibverbs drivers which set errno
		 * instead of directly returning the error value */
		return (errno == ENOMEM) ? -FI_EAGAIN : -errno;
	default:
		return -ret;
	}
}

static ssize_t
fi_ibv_srq_ep_recv(struct fid_ep *ep, void *buf, size_t len,
		void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct fi_msg msg;

	iov.iov_base = buf;
	iov.iov_len = len;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_srq_ep_recvmsg(ep, &msg, 0);
}

static ssize_t
fi_ibv_srq_ep_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
                 size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;

	return fi_ibv_srq_ep_recvmsg(ep, &msg, 0);
}

static struct fi_ops_msg fi_ibv_srq_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_ibv_srq_ep_recv,
	.recvv = fi_ibv_srq_ep_recvv,
	.recvmsg = fi_ibv_srq_ep_recvmsg,
	.send = fi_no_msg_send,
	.sendv = fi_no_msg_sendv,
	.sendmsg = fi_no_msg_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

int fi_ibv_srq_close(fid_t fid)
{
	struct fi_ibv_srq_ep *ep;
	int rc;

	ep = container_of(fid, struct fi_ibv_srq_ep, ep_fid.fid);

	rc = ibv_destroy_srq(ep->srq);
	if (rc)
		FI_WARN(&fi_ibv_prov, FI_LOG_EP_CTRL, "Cannot destroy SRQ rc=%d\n", rc);

	free(ep);

	return 0;
}

static struct fi_ops fi_ibv_srq_ep_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_srq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


int fi_ibv_srq_context(struct fid_domain *domain, struct fi_rx_attr *attr,
		struct fid_ep **rx_ep, void *context)
{
	struct ibv_srq_init_attr srq_init_attr = {};
	struct fi_ibv_domain *dom;
	struct fi_ibv_srq_ep *_rx_ep;

	if (!domain)
		return -FI_EINVAL;

	_rx_ep = calloc(1, sizeof *_rx_ep);
	if (!_rx_ep)
		return -FI_ENOMEM;

	dom = container_of(domain, struct fi_ibv_domain, domain_fid);

	_rx_ep->ep_fid.fid.fclass = FI_CLASS_SRX_CTX;
	_rx_ep->ep_fid.fid.context = context;
	_rx_ep->ep_fid.fid.ops = &fi_ibv_srq_ep_ops;
	_rx_ep->ep_fid.ops = &fi_ibv_srq_ep_base_ops;
	_rx_ep->ep_fid.msg = &fi_ibv_srq_msg_ops;
	_rx_ep->ep_fid.cm = &fi_ibv_srq_cm_ops;
	_rx_ep->ep_fid.rma = &fi_ibv_srq_rma_ops;
	_rx_ep->ep_fid.atomic = &fi_ibv_srq_atomic_ops;

	srq_init_attr.attr.max_wr = attr->size;
	srq_init_attr.attr.max_sge = attr->iov_limit;

	_rx_ep->srq = ibv_create_srq(dom->pd, &srq_init_attr);
	if (!_rx_ep->srq) {
		free(_rx_ep);
		return -errno;
	}

	*rx_ep = &_rx_ep->ep_fid;

	return 0;
}

