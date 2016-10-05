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

ssize_t psmx2_recv_generic(struct fid_ep *ep, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, void *context,
			   uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag, psm2_tagsel;
	uint32_t tag32, tagsel32;
	struct fi_context *fi_context;
	int recv_flag = 0;
	size_t idx;
	int err;

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

			psm2_epaddr = av->epaddrs[idx];
			vlane = av->vlanes[idx];
		} else {
			psm2_epaddr = PSMX2_ADDR_TO_EP(src_addr);
			vlane = PSMX2_ADDR_TO_VL(src_addr);
		}
		tag32 = PSMX2_TAG32(PSMX2_MSG_BIT, vlane, ep_priv->vlane);
		tagsel32 = ~PSMX2_IOV_BIT;
	} else {
		psm2_epaddr = 0;
		tag32 = PSMX2_TAG32(PSMX2_MSG_BIT, 0, ep_priv->vlane);
		tagsel32 = ~(PSMX2_IOV_BIT | PSMX2_SRC_BITS);
	}

	PSMX2_SET_TAG(psm2_tag, 0ULL, tag32);
	PSMX2_SET_TAG(psm2_tagsel, 0ULL, tagsel32);

	if (ep_priv->recv_selective_completion && !(flags & FI_COMPLETION)) {
		fi_context = psmx2_ep_get_op_context(ep_priv);
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_NOCOMP_RECV_CONTEXT_ALLOC;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
		PSMX2_CTXT_USER(fi_context) = buf;
		PSMX2_CTXT_SIZE(fi_context) = len;
	} else {
		if (!context)
			return -FI_EINVAL;

		fi_context = context;
		if (flags & FI_MULTI_RECV) {
			struct psmx2_multi_recv *req;

			req = calloc(1, sizeof(*req));
			if (!req)
				return -FI_ENOMEM;

			req->src_addr = psm2_epaddr;
			req->tag = psm2_tag;
			req->tagsel = psm2_tagsel;
			req->flag = recv_flag;
			req->buf = buf;
			req->len = len;
			req->offset = 0;
			req->min_buf_size = ep_priv->min_multi_recv;
			req->context = fi_context; 
			PSMX2_CTXT_TYPE(fi_context) = PSMX2_MULTI_RECV_CONTEXT;
			PSMX2_CTXT_USER(fi_context) = req;
			if (len > PSMX2_MAX_MSG_SIZE)
				len = PSMX2_MAX_MSG_SIZE;
		} else {
			PSMX2_CTXT_TYPE(fi_context) = PSMX2_RECV_CONTEXT;
			PSMX2_CTXT_USER(fi_context) = buf;
		}
		PSMX2_CTXT_EP(fi_context) = ep_priv;
		PSMX2_CTXT_SIZE(fi_context) = len;
	}

	err = psm2_mq_irecv2(ep_priv->domain->psm2_mq, psm2_epaddr,
			     &psm2_tag, &psm2_tagsel, recv_flag, buf, len,
			     (void *)fi_context, &psm2_req);
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

	return psmx2_recv_generic(ep, buf, len, desc, src_addr, context,
				  ep_priv->rx_flags);
}

static ssize_t psmx2_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t flags)
{
	void *buf;
	size_t len;

	if (!msg || (msg->iov_count && !msg->msg_iov))
		return -FI_EINVAL;

	if (msg->iov_count > 1) {
		return -FI_ENOSYS;
	} else if (msg->iov_count) {
		buf = msg->msg_iov[0].iov_base;
		len = msg->msg_iov[0].iov_len;
	} else {
		buf = NULL;
		len = 0;
	}

	return psmx2_recv_generic(ep, buf, len,
				  msg->desc ? msg->desc[0] : NULL,
				  msg->addr, msg->context, flags);
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
		return -FI_ENOSYS;
	} else if (count) {
		buf = iov[0].iov_base;
		len = iov[0].iov_len;
	} else {
		buf = NULL;
		len = 0;
	}

	return psmx2_recv(ep, buf, len, desc ? desc[0] : NULL,
			  src_addr, context);
}

ssize_t psmx2_send_generic(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context,
			   uint64_t flags, uint64_t data)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag;
	uint32_t tag32;
	struct fi_context * fi_context;
	int send_flag = 0;
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

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else  {
		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	tag32 = PSMX2_TAG32(PSMX2_MSG_BIT, ep_priv->vlane, vlane);
	PSMX2_SET_TAG(psm2_tag, data, tag32);

	if ((flags & PSMX2_NO_COMPLETION) ||
	    (ep_priv->send_selective_completion && !(flags & FI_COMPLETION)))
		no_completion = 1;

	if (flags & FI_INJECT) {
		if (len > PSMX2_INJECT_SIZE)
			return -FI_EMSGSIZE;

		err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr,
				    send_flag, &psm2_tag, buf, len);

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
	} else {
		if (!context)
			return -FI_EINVAL;

		fi_context = context;
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_SEND_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = (void *)buf;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
	}

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr,
			     send_flag, &psm2_tag, buf, len,
			     (void *)fi_context, &psm2_req);

	if (err != PSM2_OK)
		return psmx2_errno(err);

	if (fi_context == context)
		PSMX2_CTXT_REQ(fi_context) = psm2_req;

	return 0;
}

ssize_t psmx2_sendv_generic(struct fid_ep *ep, const struct iovec *iov,
			    void *desc, size_t count, fi_addr_t dest_addr,
			    void *context, uint64_t flags, uint64_t data)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	psm2_epaddr_t psm2_epaddr;
	uint8_t vlane;
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag;
	uint32_t tag32, tag32_base;
	struct fi_context * fi_context;
	int send_flag = 0;
	int err;
	size_t idx;
	int no_completion = 0;
	struct psmx2_cq_event *event;
	size_t real_count;
	size_t len, total_len;
	char *p;
	uint32_t *q;
	int i;
	struct psmx2_sendv_request *req;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_SENDV;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->sendv.ep = ep;
		trigger->sendv.iov = iov;
		trigger->sendv.desc = desc;
		trigger->sendv.count = count;
		trigger->sendv.dest_addr = dest_addr;
		trigger->sendv.context = context;
		trigger->sendv.flags = flags & ~FI_TRIGGER;
		trigger->sendv.data = data;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	total_len = 0;
	real_count = 0;
	for (i=0; i<count; i++) {
		if (iov[i].iov_len) {
			total_len += iov[i].iov_len;
			real_count++;
		}
	}

	req = malloc(sizeof(*req));
	if (!req)
		return -FI_ENOMEM;

	if (total_len <= PSMX2_IOV_BUF_SIZE) {
		req->iov_protocol = PSMX2_IOV_PROTO_PACK;
		p = req->buf;
		for (i=0; i<count; i++) {
			if (iov[i].iov_len) {
				memcpy(p, iov[i].iov_base, iov[i].iov_len);
				p += iov[i].iov_len;
			}
		}

		tag32_base = PSMX2_MSG_BIT;
		len = total_len;
	} else {
		req->iov_protocol = PSMX2_IOV_PROTO_MULTI;
		req->iov_done = 0;
		req->iov_info.seq_num = (++ep_priv->iov_seq_num) %
					PSMX2_IOV_MAX_SEQ_NUM + 1;
		req->iov_info.count = (uint32_t)real_count;
		req->iov_info.total_len = (uint32_t)total_len;

		q = req->iov_info.len;
		for (i=0; i<count; i++) {
			if (iov[i].iov_len)
				*q++ = (uint32_t)iov[i].iov_len;
		}

		tag32_base = PSMX2_MSG_BIT | PSMX2_IOV_BIT;
		len = (3 + real_count) * sizeof(uint32_t);
	}

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = (size_t)dest_addr;
		if (idx >= av->last) {
			free(req);
			return -FI_EINVAL;
		}

		psm2_epaddr = av->epaddrs[idx];
		vlane = av->vlanes[idx];
	} else  {
		psm2_epaddr = PSMX2_ADDR_TO_EP(dest_addr);
		vlane = PSMX2_ADDR_TO_VL(dest_addr);
	}

	tag32 = PSMX2_TAG32(tag32_base, ep_priv->vlane, vlane);
	PSMX2_SET_TAG(psm2_tag, data, tag32);

	if ((flags & PSMX2_NO_COMPLETION) ||
	    (ep_priv->send_selective_completion && !(flags & FI_COMPLETION)))
		no_completion = 1;

	if (flags & FI_INJECT) {
		if (len > PSMX2_INJECT_SIZE) {
			free(req);
			return -FI_EMSGSIZE;
		}

		err = psm2_mq_send2(ep_priv->domain->psm2_mq, psm2_epaddr,
				    send_flag, &psm2_tag, req->buf, len);

		free(req);

		if (err != PSM2_OK)
			return psmx2_errno(err);

		if (ep_priv->send_cntr)
			psmx2_cntr_inc(ep_priv->send_cntr);

		if (ep_priv->send_cq && !no_completion) {
			event = psmx2_cq_create_event(
					ep_priv->send_cq,
					context, NULL, flags, len,
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

	req->no_completion = no_completion;
	req->user_context = context;
	req->comp_flag = FI_MSG;

	fi_context = &req->fi_context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_SENDV_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = req;
	PSMX2_CTXT_EP(fi_context) = ep_priv;

	err = psm2_mq_isend2(ep_priv->domain->psm2_mq, psm2_epaddr,
			     send_flag, &psm2_tag, req->buf, len,
			     (void *)fi_context, &psm2_req);

	if (err != PSM2_OK) {
		free(req);
		return psmx2_errno(err);
	}

	PSMX2_CTXT_REQ(fi_context) = psm2_req;

	if (req->iov_protocol == PSMX2_IOV_PROTO_MULTI) {
		fi_context = &req->fi_context_iov;
		PSMX2_CTXT_TYPE(fi_context) = PSMX2_IOV_SEND_CONTEXT;
		PSMX2_CTXT_USER(fi_context) = req;
		PSMX2_CTXT_EP(fi_context) = ep_priv;
		tag32 = PSMX2_TAG32(PSMX2_MSG_BIT, ep_priv->vlane, vlane);
		PSMX2_TAG32_SET_SEQ(tag32, req->iov_info.seq_num);
		PSMX2_SET_TAG(psm2_tag, data, tag32);
		for (i=0; i<count; i++) {
			if (iov[i].iov_len) {
				err = psm2_mq_isend2(ep_priv->domain->psm2_mq,
						     psm2_epaddr, send_flag, &psm2_tag,
						     iov[i].iov_base, iov[i].iov_len,
						     (void *)fi_context, &psm2_req);
				if (err != PSM2_OK)
					return psmx2_errno(err);
			}
		}
	}

	return 0;
}

int psmx2_handle_sendv_req(struct psmx2_fid_ep *ep,
			   psm2_mq_status2_t *psm2_status,
			   int multi_recv)
{
	psm2_mq_req_t psm2_req;
	psm2_mq_tag_t psm2_tag, psm2_tagsel;
	struct psmx2_sendv_reply *rep;
	struct psmx2_multi_recv *recv_req;
	struct fi_context *fi_context;
	struct fi_context *recv_context;
	int i, err;
	uint8_t *recv_buf;
	size_t recv_len, len;

	if (psm2_status->error_code != PSM2_OK)
		return psmx2_errno(psm2_status->error_code);

	rep = malloc(sizeof(*rep));
	if (!rep) {
		psm2_status->error_code = PSM2_NO_MEMORY;
		return -FI_ENOMEM;
	}

	recv_context = psm2_status->context;
	if (multi_recv) {
		recv_req = PSMX2_CTXT_USER(recv_context);
		recv_buf = recv_req->buf + recv_req->offset;
		recv_len = recv_req->len - recv_req->offset;
		rep->multi_recv = 1;
	} else {
		recv_buf = PSMX2_CTXT_USER(recv_context);
		recv_len = PSMX2_CTXT_SIZE(recv_context);
		rep->multi_recv = 0;
	}

	/* assert(psm2_status->nbytes <= PSMX2_IOV_BUF_SIZE */

	memcpy(&rep->iov_info, recv_buf, psm2_status->nbytes);

	rep->user_context = psm2_status->context;
	rep->buf = recv_buf;
	rep->no_completion = 0;
	rep->iov_done = 0;
	rep->bytes_received = 0;
	rep->msg_length = 0;
	rep->error_code = PSM2_OK;

	fi_context = &rep->fi_context;
	PSMX2_CTXT_TYPE(fi_context) = PSMX2_IOV_RECV_CONTEXT;
	PSMX2_CTXT_USER(fi_context) = rep;
	PSMX2_CTXT_EP(fi_context) = ep;

	/* use the same tag, with IOV bit cleared, and seq_num added */
	psm2_tag = psm2_status->msg_tag;
	psm2_tag.tag2 &= ~PSMX2_IOV_BIT;
	PSMX2_TAG32_SET_SEQ(psm2_tag.tag2, rep->iov_info.seq_num);

	rep->comp_flag = (psm2_tag.tag2 & PSMX2_MSG_BIT) ? FI_MSG : FI_TAGGED;

	/* match every bit of the tag */
	PSMX2_SET_TAG(psm2_tagsel, -1UL, -1);

	for (i=0; i<rep->iov_info.count; i++) {
		if (recv_len) {
			len = MIN(recv_len, rep->iov_info.len[i]);
			err = psm2_mq_irecv2(ep->domain->psm2_mq,
					     psm2_status->msg_peer,
					     &psm2_tag, &psm2_tagsel,
					     0/*flag*/, recv_buf, len,
					     (void *)fi_context, &psm2_req);
			if (err) {
				psm2_status->error_code = err;
				return psmx2_errno(psm2_status->error_code);
			}
			recv_buf += len;
			recv_len -= len;
		} else {
			/* recv buffer full, pust empty recvs */
			err = psm2_mq_irecv2(ep->domain->psm2_mq,
					     psm2_status->msg_peer,
					     &psm2_tag, &psm2_tagsel,
					     0/*flag*/, NULL, 0,
					     (void *)fi_context, &psm2_req);
			if (err) {
				psm2_status->error_code = err;
				return psmx2_errno(psm2_status->error_code);
			}
		}
	}

	return 0;
}

static ssize_t psmx2_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return psmx2_send_generic(ep, buf, len, desc, dest_addr, context,
				  ep_priv->tx_flags, 0);
}

static ssize_t psmx2_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t flags)
{
	void *buf;
	size_t len;

	if (!msg || (msg->iov_count && !msg->msg_iov))
		return -FI_EINVAL;

	if (msg->iov_count > 1) {
		return psmx2_sendv_generic(ep, msg->msg_iov, msg->desc,
					   msg->iov_count, msg->addr,
					   msg->context, flags,
					   msg->data);
	} else if (msg->iov_count) {
		buf = msg->msg_iov[0].iov_base;
		len = msg->msg_iov[0].iov_len;
	} else {
		buf = NULL;
		len = 0;
	}

	return psmx2_send_generic(ep, buf, len,
				  msg->desc ? msg->desc[0] : NULL,
				  msg->addr, msg->context, flags,
				  msg->data);
}

static ssize_t psmx2_sendv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t dest_addr,
			   void *context)
{
	void *buf;
	size_t len;

	if (count && !iov)
		return -FI_EINVAL;

	if (count > PSMX2_IOV_MAX_COUNT) {
		return -FI_EINVAL;
	} else if (count > 1) {
		struct psmx2_fid_ep *ep_priv;
		ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

		return psmx2_sendv_generic(ep, iov, desc, count, dest_addr,
					   context, ep_priv->tx_flags, 0);
	} else if (count) {
		buf = iov[0].iov_base;
		len = iov[0].iov_len;
	} else {
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

	return psmx2_send_generic(ep, buf, len, NULL, dest_addr, NULL,
				  ep_priv->tx_flags | FI_INJECT | PSMX2_NO_COMPLETION, 
				  0);
}

static ssize_t psmx2_senddata(struct fid_ep *ep, const void *buf, size_t len,
			      void *desc, uint64_t data, fi_addr_t dest_addr,
			      void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return psmx2_send_generic(ep, buf, len, desc, dest_addr, context,
				  ep_priv->tx_flags, data);
}

static ssize_t psmx2_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			        uint64_t data, fi_addr_t dest_addr)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return psmx2_send_generic(ep, buf, len, NULL, dest_addr, NULL,
				  ep_priv->tx_flags | FI_INJECT | PSMX2_NO_COMPLETION,
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

