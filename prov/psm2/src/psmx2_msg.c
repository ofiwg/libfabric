/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

ssize_t _psmx2_recv(struct fid_ep *ep, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr, void *context,
		    uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag, psm2_tagsel;
	struct fi_context *fi_context;
	int err;
	int recv_flag = 0;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_RECV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->recv.ep = ep;
		trigger->recv.buf = buf;
		trigger->recv.len = len;
		trigger->recv.desc = desc;
		trigger->recv.src_addr = src_addr;
		trigger->recv.context = context;
		trigger->recv.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if ((ep_priv->caps & FI_DIRECTED_RECV) && src_addr != FI_ADDR_UNSPEC) {
		av = ep_priv->av;
		if (av && av->type == FI_AV_TABLE) {
			idx = (size_t)src_addr;
			if (idx >= av->last)
				return -FI_EINVAL;

			src_addr = (fi_addr_t)av->psm2_epaddrs[idx];
		}
	}
	else {
		src_addr = 0;
	}

	if (ep_priv->recv_selective_completion && !(flags & FI_COMPLETION)) {
		fi_context = &ep_priv->nocomp_recv_context;
	}
	else {
		if (!context)
			return -FI_EINVAL;

		fi_context = context;
		if (flags & FI_MULTI_RECV) {
			struct psmx2_multi_recv *req;

			req = calloc(1, sizeof(*req));
			if (!req)
				return -FI_ENOMEM;

			req->flag = recv_flag;
			req->buf = buf;
			req->len = len;
			req->offset = 0;
			req->min_buf_size = ep_priv->min_multi_recv;
			req->context = fi_context; 
			PSMX2_CTXT_TYPE(fi_context) = PSMX2_MULTI_RECV_CONTEXT;
			PSMX2_CTXT_USER(fi_context) = req;
		}
		else {
			PSMX2_CTXT_TYPE(fi_context) = PSMX2_RECV_CONTEXT;
			PSMX2_CTXT_USER(fi_context) = buf;
		}
		PSMX2_CTXT_EP(fi_context) = ep_priv;
	}

	PSMX2_SET_TAG(psm2_tag, 0ULL, PSMX2_MSG_BIT);
	PSMX2_SET_TAG(psm2_tagsel, 0ULL, -1);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			    (psm2_epaddr_t)src_addr,
			    &psm2_tag, &psm2_tagsel, recv_flag,
			    buf, len, (void *)fi_context, &psm2_req);
	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (fi_context == context)
		PSMX2_CTXT_REQ(fi_context) = psm2_req;

	return 0;
}

static ssize_t psmx2_recv(struct fid_ep *ep, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_recv(ep, buf, len, desc, src_addr, context,
			   ep_priv->flags);
}

static ssize_t psmx2_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t flags)
{
	void *buf;
	size_t len;

	if (!msg || (msg->iov_count && !msg->msg_iov))
		return -FI_EINVAL;

	if (msg->iov_count > 1) {
		return -FI_EINVAL;
	}
	else if (msg->iov_count) {
		buf = msg->msg_iov[0].iov_base;
		len = msg->msg_iov[0].iov_len;
	}
	else {
		buf = NULL;
		len = 0;
	}

	return _psmx2_recv(ep, buf, len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->context, flags);
}

static ssize_t psmx2_recvv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t src_addr,
			   void *context)
{
	void *buf;
	size_t len;

	if (count && !iov)
		return -FI_EINVAL;

	if (count > 1) {
		return -FI_EINVAL;
	}
	else if (count) {
		buf = iov[0].iov_base;
		len = iov[0].iov_len;
	}
	else {
		buf = NULL;
		len = 0;
	}

	return psmx2_recv(ep, buf, len, desc ? desc[0] : NULL,
			  src_addr, context);
}

ssize_t _psmx2_send(struct fid_ep *ep, const void *buf, size_t len,
		    void *desc, fi_addr_t dest_addr, void *context,
		    uint64_t flags, uint64_t data)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	int send_flag = 0;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag;
	struct fi_context * fi_context;
	int err;
	size_t idx;
	int no_completion = 0;
	struct psmx2_cq_event *event;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_SEND;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->send.ep = ep;
		trigger->send.buf = buf;
		trigger->send.len = len;
		trigger->send.desc = desc;
		trigger->send.dest_addr = dest_addr;
		trigger->send.context = context;
		trigger->send.flags = flags & ~FI_TRIGGER;
		trigger->send.data = data;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = (size_t)dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->psm2_epaddrs[idx];
	}
	else  {
		psm2_epaddr = (psm2_epaddr_t) dest_addr;
	}

	PSMX2_SET_TAG(psm2_tag, data, PSMX2_MSG_BIT);

	if ((flags & PSMX2_NO_COMPLETION) ||
	    (ep_priv->send_selective_completion && !(flags & FI_COMPLETION)))
		no_completion = 1;

	if (flags & FI_INJECT) {
		if (len > PSMX2_INJECT_SIZE)
			return -FI_EMSGSIZE;

		err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr, send_flag,
				    &psm2_tag, buf, len);

		if (err != PSM2_OK)
			return psmx2_errno(err);

		if (ep_priv->send_cntr)
			psmx2_cntr_inc(ep_priv->send_cntr);

		if (ep_priv->send_cq && !no_completion) {
			event = psmx2_cq_create_event(
					ep_priv->send_cq,
					context, (void *)buf, flags, len,
					(uint64_t) data,
					0 /* tag */,
					0 /* olen */,
					0 /* err */);

			if (event)
				psmx2_cq_enqueue_event(ep_priv->send_cq, event);
			else
				return -FI_ENOMEM;
		}

		return 0;
	}

	if (no_completion && !context) {
		fi_context = &ep_priv->nocomp_send_context;
	}
	else {
		if (!context)
			return -FI_EINVAL;

		fi_context = context;
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_SEND_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = (void *)buf;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
	}

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, send_flag,
			     &psm2_tag, buf, len, (void *)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (fi_context == context)
		PSMX2_CTXT_REQ(fi_context) = psm2_req;

	return 0;
}

static ssize_t psmx2_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_send(ep, buf, len, desc, dest_addr, context,
			   ep_priv->flags, 0);
}

static ssize_t psmx2_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t flags)
{
	void *buf;
	size_t len;

	if (!msg || (msg->iov_count && !msg->msg_iov))
		return -FI_EINVAL;

	if (msg->iov_count > 1) {
		return -FI_EINVAL;
	}
	else if (msg->iov_count) {
		buf = msg->msg_iov[0].iov_base;
		len = msg->msg_iov[0].iov_len;
	}
	else {
		buf = NULL;
		len = 0;
	}

	return _psmx2_send(ep, buf, len,
			   msg->desc ? msg->desc[0] : NULL, msg->addr,
			   msg->context, flags, msg->data);
}

static ssize_t psmx2_sendv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t dest_addr,
			   void *context)
{
	void *buf;
	size_t len;

	if (count && !iov)
		return -FI_EINVAL;

	if (count > 1) {
		return -FI_EINVAL;
	}
	else if (count) {
		buf = iov[0].iov_base;
		len = iov[0].iov_len;
	}
	else {
		buf = NULL;
		len = 0;
	}

	return psmx2_send(ep, buf, len, desc ? desc[0] : NULL,
			  dest_addr, context);
}

static ssize_t psmx2_inject(struct fid_ep *ep, const void *buf, size_t len,
			    fi_addr_t dest_addr)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_send(ep, buf, len, NULL, dest_addr, NULL,
			   ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION, 0);
}

static ssize_t psmx2_senddata(struct fid_ep *ep, const void *buf, size_t len,
			      void *desc, uint64_t data, fi_addr_t dest_addr,
			      void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_send(ep, buf, len, desc, dest_addr, context,
			   ep_priv->flags, data);
}

static ssize_t psmx2_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			        uint64_t data, fi_addr_t dest_addr)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_send(ep, buf, len, NULL, dest_addr, NULL,
			   ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION,
			   data);
}

struct fi_ops_msg psmx2_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = psmx2_recv,
	.recvv = psmx2_recvv,
	.recvmsg = psmx2_recvmsg,
	.send = psmx2_send,
	.sendv = psmx2_sendv,
	.sendmsg = psmx2_sendmsg,
	.inject = psmx2_inject,
	.senddata = psmx2_senddata,
	.injectdata = psmx2_injectdata,
};

