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

ssize_t _psmx2_tagged_peek(struct fid_ep *ep, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr,
			   uint64_t tag, uint64_t ignore,
			   void *context, uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_status2_t psm2_status2;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	psm2_mq_req_t req;
	struct psmx2_fid_av *av;
	size_t idx;
	psm2_epaddr_t psm2_src_addr;
	uint64_t psm2_tag, psm2_tagsel;
	struct psmx2_cq_event *event;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (tag & ep_priv->domain->reserved_tag_bits) {
		FI_WARN(&psmx2_prov, FI_LOG_EP_DATA,
			"using reserved tag bits."
			"tag=%lx. reserved_bits=%lx.\n", tag,
			ep_priv->domain->reserved_tag_bits);
	}

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	if (src_addr != FI_ADDR_UNSPEC) {
		av = ep_priv->av;
		if (av && av->type == FI_AV_TABLE) {
			idx = (size_t)src_addr;
			if (idx >= av->last)
				return -FI_EINVAL;

			psm2_src_addr = av->psm2_epaddrs[idx];
		}
		else {
			psm2_src_addr = (psm2_epaddr_t)src_addr;
		}
	}
	else {
		psm2_src_addr = NULL;
	}

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	if (flags & (FI_CLAIM | FI_DISCARD))
		err = psm2_mq_improbe2(ep_priv->domain->psm2_mq,
				       psm2_src_addr,
				       &psm2_tag2, &psm2_tagsel2,
				       &req, &psm2_status2);
	else
		err = psm2_mq_iprobe2(ep_priv->domain->psm2_mq,
				      psm2_src_addr,
				      &psm2_tag2, &psm2_tagsel2,
				      &psm2_status2);
	switch (err) {
	case PSM2_OK:
		if (ep_priv->recv_cq) {
			if ((flags & FI_CLAIM) && context)
				PSMX2_CTXT_REQ((struct fi_context *)context) = req;

			tag = psm2_status2.msg_tag.tag0 | (((uint64_t)psm2_status2.msg_tag.tag1) << 32);
			len = psm2_status2.msg_length;
			src_addr = (fi_addr_t)psm2_status2.msg_peer;
			event = psmx2_cq_create_event(
					ep_priv->recv_cq,
					context,		/* op_context */
					NULL,			/* buf */
					flags|FI_RECV|FI_TAGGED,/* flags */
					len,			/* len */
					0,			/* data */
					tag,			/* tag */
					len,			/* olen */
					0);			/* err */

			if (!event)
				return -FI_ENOMEM;

			event->source = src_addr;
			psmx2_cq_enqueue_event(ep_priv->recv_cq, event);
		}
		return 0;

	case PSM2_MQ_NO_COMPLETIONS:
		if (ep_priv->recv_cq) {
			event = psmx2_cq_create_event(
					ep_priv->recv_cq,
					context,		/* op_context */
					NULL,			/* buf */
					flags|FI_RECV|FI_TAGGED,/* flags */
					len,			/* len */
					0,			/* data */
					tag,			/* tag */
					len,			/* olen */
					-FI_ENOMSG);		/* err */

			if (!event)
				return -FI_ENOMEM;

			event->source = 0;
			psmx2_cq_enqueue_event(ep_priv->recv_cq, event);
		}
		return 0;

	default:
		return psmx2_errno(err);
	}
}

ssize_t _psmx2_tagged_recv(struct fid_ep *ep, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr,
			   uint64_t tag, uint64_t ignore,
			   void *context, uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag, psm2_tagsel;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	struct psmx2_fid_av *av;
	size_t idx;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_PEEK)
		return _psmx2_tagged_peek(ep, buf, len, desc, src_addr,
					  tag, ignore, context, flags);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_TRECV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->trecv.ep = ep;
		trigger->trecv.buf = buf;
		trigger->trecv.len = len;
		trigger->trecv.desc = desc;
		trigger->trecv.src_addr = src_addr;
		trigger->trecv.tag = tag;
		trigger->trecv.ignore = ignore;
		trigger->trecv.context = context;
		trigger->trecv.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (flags & FI_CLAIM) {
		if (!context)
			return -FI_EINVAL;

		/* TODO: handle FI_DISCARD */

		fi_context = context;
		psm2_req = PSMX2_CTXT_REQ(fi_context);
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_TRECV_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = buf;
		PSMX2_CTXT_EP(fi_context) = ep_priv;

		err = psm2_mq_imrecv(ep_priv->domain->psm2_mq, 0, /*flags*/
				     buf, len, context, &psm2_req);
		if (err != PSM2_OK)
			return psmx2_errno(err);

		PSMX2_CTXT_REQ(fi_context) = psm2_req;
		return 0;
	}

	if (tag & ep_priv->domain->reserved_tag_bits) {
		FI_WARN(&psmx2_prov, FI_LOG_EP_DATA,
			"using reserved tag bits."
			"tag=%lx. reserved_bits=%lx.\n", tag,
			ep_priv->domain->reserved_tag_bits);
	}

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	if (ep_priv->recv_selective_completion && !(flags & FI_COMPLETION)) {
		fi_context = &ep_priv->nocomp_recv_context;
	}
	else {
		if (!context)
			return -FI_EINVAL;

		fi_context = context;
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_TRECV_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = buf;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
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

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			     (psm2_epaddr_t)src_addr,
			      &psm2_tag2, &psm2_tagsel2, 0, /* flags */
			      buf, len, (void *)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (fi_context == context)
		PSMX2_CTXT_REQ(fi_context) = psm2_req;

	return 0;
}

ssize_t psmx2_tagged_recv_no_flag_av_map(struct fid_ep *ep, void *buf,
					 size_t len, void *desc,
					 fi_addr_t src_addr,
					 uint64_t tag, uint64_t ignore,
					 void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag, psm2_tagsel;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	fi_context = context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_TRECV_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = buf;
	PSMX2_CTXT_EP(fi_context) = ep_priv;

	if (! ((ep_priv->caps & FI_DIRECTED_RECV) && src_addr != FI_ADDR_UNSPEC))
		src_addr = 0;

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			     (psm2_epaddr_t)src_addr,
			     &psm2_tag2, &psm2_tagsel2, 0, /* flags */
			     buf, len, (void *)fi_context, &psm2_req);
	if (err != PSM2_OK)
		return psmx2_errno(err);

	PSMX2_CTXT_REQ(fi_context) = psm2_req;
	return 0;
}

ssize_t psmx2_tagged_recv_no_flag_av_table(struct fid_ep *ep, void *buf,
					   size_t len, void *desc,
					   fi_addr_t src_addr,
					   uint64_t tag, uint64_t ignore,
					   void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag, psm2_tagsel;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	size_t idx;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	fi_context = context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_TRECV_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = buf;
	PSMX2_CTXT_EP(fi_context) = ep_priv;

	if ((ep_priv->caps & FI_DIRECTED_RECV) && src_addr != FI_ADDR_UNSPEC) {
		av = ep_priv->av;
		idx = (size_t)src_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->psm2_epaddrs[idx];
	}
	else {
		psm2_epaddr = NULL;
	}

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			     psm2_epaddr,
			     &psm2_tag2, &psm2_tagsel2, 0, /* flags */
			     buf, len, (void *)fi_context, &psm2_req);
	if (err != PSM2_OK)
		return psmx2_errno(err);

	PSMX2_CTXT_REQ(fi_context) = psm2_req;
	return 0;
}

ssize_t psmx2_tagged_recv_no_event_av_map(struct fid_ep *ep, void *buf,
					  size_t len, void *desc,
					  fi_addr_t src_addr,
					  uint64_t tag, uint64_t ignore,
					  void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag, psm2_tagsel;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	fi_context = &ep_priv->nocomp_recv_context;

	if (! ((ep_priv->caps & FI_DIRECTED_RECV) && src_addr != FI_ADDR_UNSPEC))
		src_addr = 0;

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			     (psm2_epaddr_t)src_addr,
			     &psm2_tag2, &psm2_tagsel2, 0, /* flags */
			     buf, len, (void *)fi_context, &psm2_req);

	return psmx2_errno(err);
}

ssize_t psmx2_tagged_recv_no_event_av_table(struct fid_ep *ep, void *buf,
					    size_t len, void *desc,
					    fi_addr_t src_addr,
					    uint64_t tag, uint64_t ignore,
					    void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag, psm2_tagsel;
	psm2_mq_tag_t psm2_tag2, psm2_tagsel2;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	size_t idx;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	psm2_tagsel = (~ignore) | ep_priv->domain->reserved_tag_bits;

	fi_context = &ep_priv->nocomp_recv_context;

	if ((ep_priv->caps & FI_DIRECTED_RECV) && src_addr != FI_ADDR_UNSPEC) {
		av = ep_priv->av;
		idx = (size_t)src_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		psm2_epaddr = av->psm2_epaddrs[idx];
	}
	else {
		psm2_epaddr = NULL;
	}

	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);
	PSMX2_SET_TAG(psm2_tagsel2, psm2_tagsel, 0);

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq,
			     psm2_epaddr,
			     &psm2_tag2, &psm2_tagsel2, 0, /* flags */
			     buf, len, (void *)fi_context, &psm2_req);

	return psmx2_errno(err);
}

static ssize_t psmx2_tagged_recv(struct fid_ep *ep, void *buf,
				 size_t len, void *desc,
				 fi_addr_t src_addr, uint64_t tag,
				 uint64_t ignore, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_tagged_recv(ep, buf, len, desc, src_addr, tag, ignore,
				  context, ep_priv->flags);
}

static ssize_t psmx2_tagged_recvmsg(struct fid_ep *ep,
				    const struct fi_msg_tagged *msg,
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

	return _psmx2_tagged_recv(ep, buf, len,
				  msg->desc ? msg->desc[0] : NULL,
				  msg->addr, msg->tag, msg->ignore,
				  msg->context, flags);
}

static ssize_t psmx2_tagged_recv_no_flag(struct fid_ep *ep, void *buf,
					 size_t len, void *desc,
					 fi_addr_t src_addr,
					 uint64_t tag, uint64_t ignore,
					 void *context)
{
	return psmx2_tagged_recv_no_flag_av_map(
					ep, buf, len, desc, src_addr,
					tag, ignore, context);
}

static ssize_t psmx2_tagged_recv_no_event(struct fid_ep *ep, void *buf,
					  size_t len, void *desc,
					  fi_addr_t src_addr,
					  uint64_t tag, uint64_t ignore,
					  void *context)
{
	return psmx2_tagged_recv_no_event_av_map(
					ep, buf, len, desc, src_addr,
					tag, ignore, context);
}

static ssize_t psmx2_tagged_recvv(struct fid_ep *ep,
				  const struct iovec *iov, void **desc,
				  size_t count, fi_addr_t src_addr,
				  uint64_t tag, uint64_t ignore,
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

	return psmx2_tagged_recv(ep, buf, len, desc ? desc[0] : NULL,
				 src_addr, tag, ignore, context);
}

static ssize_t psmx2_tagged_recvv_no_flag(struct fid_ep *ep,
					  const struct iovec *iov,
					  void **desc, size_t count,
					  fi_addr_t src_addr,
					  uint64_t tag, uint64_t ignore,
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

	return psmx2_tagged_recv_no_flag(ep, buf, len,
					 desc ? desc[0] : NULL, src_addr,
					 tag, ignore, context);
}

static ssize_t psmx2_tagged_recvv_no_event(struct fid_ep *ep,
					   const struct iovec *iov,
					   void **desc, size_t count,
					   fi_addr_t src_addr,
					   uint64_t tag, uint64_t ignore,
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

	return psmx2_tagged_recv_no_event(ep, buf, len,
					  desc ? desc[0] : NULL, src_addr,
					  tag, ignore, context);
}

ssize_t _psmx2_tagged_send(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, uint64_t tag,
			   void *context, uint64_t flags, uint32_t data)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	struct fi_context *fi_context;
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

		trigger->op = PSMX2_TRIGGERED_TSEND;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->tsend.ep = ep;
		trigger->tsend.buf = buf;
		trigger->tsend.len = len;
		trigger->tsend.desc = desc;
		trigger->tsend.dest_addr = dest_addr;
		trigger->tsend.tag = tag;
		trigger->tsend.context = context;
		trigger->tsend.flags = flags & ~FI_TRIGGER;
		trigger->tsend.data = data;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (tag & ep_priv->domain->reserved_tag_bits) {
		FI_WARN(&psmx2_prov, FI_LOG_EP_DATA,
			"using reserved tag bits."
			"tag=%lx. reserved_bits=%lx.\n", tag,
			ep_priv->domain->reserved_tag_bits);
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

	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, data);

	if ((flags & PSMX2_NO_COMPLETION) ||
	    (ep_priv->send_selective_completion && !(flags & FI_COMPLETION)))
		no_completion = 1;

	if (flags & FI_INJECT) {
		if (len > PSMX2_INJECT_SIZE)
			return -FI_EMSGSIZE;

		err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
				    &psm2_tag2, buf, len);

		if (err != PSM2_OK)
			return psmx2_errno(err);

		if (ep_priv->send_cntr)
			psmx2_cntr_inc(ep_priv->send_cntr);

		if (ep_priv->send_cq && !no_completion) {
			event = psmx2_cq_create_event(
					ep_priv->send_cq,
					context, (void *)buf, flags, len,
					(uint64_t) data, psm2_tag,
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
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_TSEND_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = (void *)buf;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
	}

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			     &psm2_tag2, buf, len, (void*)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (fi_context == context)
		PSMX2_CTXT_REQ(fi_context) = psm2_req;

	return 0;
}

ssize_t psmx2_tagged_send_no_flag_av_map(struct fid_ep *ep, const void *buf,
					 size_t len, void *desc,
					 fi_addr_t dest_addr, uint64_t tag,
				         void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_epaddr = (psm2_epaddr_t) dest_addr;
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	fi_context = context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_TSEND_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = (void *)buf;
	PSMX2_CTXT_EP(fi_context) = ep_priv;

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			     &psm2_tag2, buf, len, (void*)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	PSMX2_CTXT_REQ(fi_context) = psm2_req;
	return 0;
}

ssize_t psmx2_tagged_send_no_flag_av_table(struct fid_ep *ep, const void *buf,
					   size_t len, void *desc,
					   fi_addr_t dest_addr, uint64_t tag,
					   void *context)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	struct fi_context *fi_context;
	int err;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	av = ep_priv->av;
	idx = (size_t)dest_addr;
	if (idx >= av->last)
		return -FI_EINVAL;

	psm2_epaddr = av->psm2_epaddrs[idx];
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	fi_context = context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_TSEND_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = (void *)buf;
	PSMX2_CTXT_EP(fi_context) = ep_priv;

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			     &psm2_tag2, buf, len, (void*)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	PSMX2_CTXT_REQ(fi_context) = psm2_req;
	return 0;
}

ssize_t psmx2_tagged_send_no_event_av_map(struct fid_ep *ep, const void *buf,
					  size_t len, void *desc,
					  fi_addr_t dest_addr, uint64_t tag,
				          void *context)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	struct fi_context *fi_context;
	int err;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_epaddr = (psm2_epaddr_t) dest_addr;
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	fi_context = &ep_priv->nocomp_send_context;

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			     &psm2_tag2, buf, len, (void*)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	return 0;
}

ssize_t psmx2_tagged_send_no_event_av_table(struct fid_ep *ep, const void *buf,
					    size_t len, void *desc,
					    fi_addr_t dest_addr, uint64_t tag,
					    void *context)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	struct fi_context *fi_context;
	int err;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	av = ep_priv->av;
	idx = (size_t)dest_addr;
	if (idx >= av->last)
		return -FI_EINVAL;

	psm2_epaddr = av->psm2_epaddrs[idx];
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	fi_context = &ep_priv->nocomp_send_context;

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			     &psm2_tag2, buf, len, (void*)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	return 0;
}

ssize_t psmx2_tagged_inject_no_flag_av_map(struct fid_ep *ep,
					   const void *buf, size_t len,
					   fi_addr_t dest_addr, uint64_t tag)
{
	struct psmx2_fid_ep *ep_priv;
	psm2_epaddr_t psm2_epaddr;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	int err;

	if (len > PSMX2_INJECT_SIZE)
		return -FI_EMSGSIZE;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	psm2_epaddr = (psm2_epaddr_t) dest_addr;
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			    &psm2_tag2, buf, len);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (ep_priv->send_cntr)
		psmx2_cntr_inc(ep_priv->send_cntr);

	return 0;
}

ssize_t psmx2_tagged_inject_no_flag_av_table(struct fid_ep *ep,
					     const void *buf, size_t len,
					     fi_addr_t dest_addr, uint64_t tag)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	uint64_t psm2_tag;
	psm2_mq_tag_t psm2_tag2;
	int err;
	size_t idx;

	if (len > PSMX2_INJECT_SIZE)
		return -FI_EMSGSIZE;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	av = ep_priv->av;
	idx = (size_t)dest_addr;
	if (idx >= av->last)
		return -FI_EINVAL;

	psm2_epaddr = av->psm2_epaddrs[idx];
	psm2_tag = tag & (~ep_priv->domain->reserved_tag_bits);
	PSMX2_SET_TAG(psm2_tag2, psm2_tag, 0);

	err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr, 0,
			    &psm2_tag2, buf, len);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (ep_priv->send_cntr)
		psmx2_cntr_inc(ep_priv->send_cntr);

	return 0;
}

static ssize_t psmx2_tagged_send(struct fid_ep *ep,
				 const void *buf, size_t len,
				 void *desc, fi_addr_t dest_addr,
				 uint64_t tag, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_tagged_send(ep, buf, len, desc, dest_addr, tag, context,
				 ep_priv->flags, 0);
}

static ssize_t psmx2_tagged_sendmsg(struct fid_ep *ep,
				    const struct fi_msg_tagged *msg,
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

	return _psmx2_tagged_send(ep, buf, len,
				  msg->desc ? msg->desc[0] : NULL, msg->addr,
				  msg->tag, msg->context, flags,
				  (uint32_t) msg->data);
}

static ssize_t psmx2_tagged_sendv(struct fid_ep *ep,
				  const struct iovec *iov, void **desc,
				  size_t count, fi_addr_t dest_addr,
				  uint64_t tag, void *context)
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

	return psmx2_tagged_send(ep, buf, len,
				 desc ? desc[0] : NULL, dest_addr, tag, context);
}

static ssize_t psmx2_tagged_sendv_no_flag_av_map(struct fid_ep *ep,
						 const struct iovec *iov,
						 void **desc, size_t count,
						 fi_addr_t dest_addr,
						 uint64_t tag, void *context)
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

	return psmx2_tagged_send_no_flag_av_map(ep, buf, len,
					        desc ? desc[0] : NULL, dest_addr,
					        tag, context);
}

static ssize_t psmx2_tagged_sendv_no_flag_av_table(struct fid_ep *ep,
						   const struct iovec *iov,
						   void **desc, size_t count,
						   fi_addr_t dest_addr,
						   uint64_t tag, void *context)
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

	return psmx2_tagged_send_no_flag_av_table(ep, buf, len,
					          desc ? desc[0] : NULL, dest_addr,
					          tag, context);
}

static ssize_t psmx2_tagged_sendv_no_event_av_map(struct fid_ep *ep,
						  const struct iovec *iov,
						  void **desc, size_t count,
						  fi_addr_t dest_addr,
						  uint64_t tag, void *context)
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

	return psmx2_tagged_send_no_event_av_map(ep, buf, len,
					         desc ? desc[0] : NULL, dest_addr,
					         tag, context);
}

static ssize_t psmx2_tagged_sendv_no_event_av_table(struct fid_ep *ep,
						    const struct iovec *iov,
						    void **desc, size_t count,
						    fi_addr_t dest_addr,
						    uint64_t tag, void *context)
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

	return psmx2_tagged_send_no_event_av_table(ep, buf, len,
					           desc ? desc[0] : NULL,
					           dest_addr, tag, context);
}

static ssize_t psmx2_tagged_inject(struct fid_ep *ep,
				   const void *buf, size_t len,
				   fi_addr_t dest_addr, uint64_t tag)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_tagged_send(ep, buf, len, NULL, dest_addr, tag, NULL,
				  ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION, 0);
}

static ssize_t psmx2_tagged_senddata(struct fid_ep *ep, const void *buf,
				     size_t len, void *desc, uint64_t data,
				     fi_addr_t dest_addr,
                                     uint64_t tag, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_tagged_send(ep, buf, len, desc, dest_addr, tag, context,
				  ep_priv->flags,  (uint32_t)data);
}

static ssize_t psmx2_tagged_injectdata(struct fid_ep *ep, const void *buf,
				       size_t len, uint64_t data,
				       fi_addr_t dest_addr, uint64_t tag)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_tagged_send(ep, buf, len, NULL, dest_addr, tag, NULL,
				  ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION,
				  (uint32_t)data);
}

/* general case */
struct fi_ops_tagged psmx2_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv,
	.recvv = psmx2_tagged_recvv,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send,
	.sendv = psmx2_tagged_sendv,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, no event suppression, FI_AV_MAP */
struct fi_ops_tagged psmx2_tagged_ops_no_flag_av_map = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_flag,
	.recvv = psmx2_tagged_recvv_no_flag,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_flag_av_map,
	.sendv = psmx2_tagged_sendv_no_flag_av_map,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_map,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, no event suppression, FI_AV_TABLE */
struct fi_ops_tagged psmx2_tagged_ops_no_flag_av_table = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_flag,
	.recvv = psmx2_tagged_recvv_no_flag,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_flag_av_table,
	.sendv = psmx2_tagged_sendv_no_flag_av_table,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_table,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, event suppression, FI_AV_MAP */
struct fi_ops_tagged psmx2_tagged_ops_no_event_av_map = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_event,
	.recvv = psmx2_tagged_recvv_no_event,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_event_av_map,
	.sendv = psmx2_tagged_sendv_no_event_av_map,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_map,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, event suppression, FI_AV_TABLE */
struct fi_ops_tagged psmx2_tagged_ops_no_event_av_table = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_event,
	.recvv = psmx2_tagged_recvv_no_event,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_event_av_table,
	.sendv = psmx2_tagged_sendv_no_event_av_table,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_table,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, send event suppression, FI_AV_MAP */
struct fi_ops_tagged psmx2_tagged_ops_no_send_event_av_map = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_flag,
	.recvv = psmx2_tagged_recvv_no_flag,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_event_av_map,
	.sendv = psmx2_tagged_sendv_no_event_av_map,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_map,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, send event suppression, FI_AV_TABLE */
struct fi_ops_tagged psmx2_tagged_ops_no_send_event_av_table = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_flag,
	.recvv = psmx2_tagged_recvv_no_flag,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_event_av_table,
	.sendv = psmx2_tagged_sendv_no_event_av_table,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_table,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, recv event suppression, FI_AV_MAP */
struct fi_ops_tagged psmx2_tagged_ops_no_recv_event_av_map = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_event,
	.recvv = psmx2_tagged_recvv_no_event,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_flag_av_map,
	.sendv = psmx2_tagged_sendv_no_flag_av_map,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_map,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};

/* op_flags=0, recv event suppression, FI_AV_TABLE */
struct fi_ops_tagged psmx2_tagged_ops_no_recv_event_av_table = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx2_tagged_recv_no_event,
	.recvv = psmx2_tagged_recvv_no_event,
	.recvmsg = psmx2_tagged_recvmsg,
	.send = psmx2_tagged_send_no_flag_av_table,
	.sendv = psmx2_tagged_sendv_no_flag_av_table,
	.sendmsg = psmx2_tagged_sendmsg,
	.inject = psmx2_tagged_inject_no_flag_av_table,
	.senddata = psmx2_tagged_senddata,
	.injectdata = psmx2_tagged_injectdata,
};
