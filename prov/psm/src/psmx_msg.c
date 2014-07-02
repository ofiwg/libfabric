/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "psmx.h"

static inline ssize_t _psmx_recvfrom(struct fid_ep *ep, void *buf, size_t len,
			void *desc, const void *src_addr, void *context,
			uint64_t flags, uint64_t data)
{
	struct psmx_fid_ep *fid_ep;
	struct psmx_epaddr_context *epaddr_context;
	psm_mq_req_t psm_req;
	uint64_t psm_tag, psm_tagsel;
	struct fi_context *fi_context;
	int user_fi_context = 0;
	int err;
	int recv_flag = 0;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	if (src_addr) {
		epaddr_context = psm_epaddr_getctxt((void *)src_addr);
		psm_tag = epaddr_context->epid | PSMX_MSG_BIT;
		psm_tagsel = -1ULL;
	}
	else {
		psm_tag = PSMX_MSG_BIT;
		psm_tagsel = PSMX_MSG_BIT;
	}

	if (fid_ep->recv_eq_event_flag && !(flags & FI_EVENT) && !context) {
		fi_context = &fid_ep->nocomp_recv_context;
	}
	else {
		if (!context)
			return -EINVAL;

		fi_context = context;
		user_fi_context = 1;
		if (flags & FI_MULTI_RECV) {
			struct psmx_multi_recv *req;

			req = calloc(1, sizeof(*req));
			if (!req)
				return -ENOMEM;

			req->tag = psm_tag;
			req->tagsel = psm_tagsel;
			req->flag = recv_flag;
			req->buf = buf;
			req->len = len;
			req->offset = 0;
			req->min_buf_size = data;
			req->context = fi_context; 
			PSMX_CTXT_TYPE(fi_context) = PSMX_MULTI_RECV_CONTEXT;
			PSMX_CTXT_USER(fi_context) = req;
		}
		else {
			PSMX_CTXT_TYPE(fi_context) = PSMX_RECV_CONTEXT;
			PSMX_CTXT_USER(fi_context) = buf;
		}
		PSMX_CTXT_EP(fi_context) = fid_ep;
	}

	err = psm_mq_irecv(fid_ep->domain->psm_mq,
			   psm_tag, psm_tagsel, recv_flag,
			   buf, len, (void *)fi_context, &psm_req);
	if (err != PSM_OK)
		return psmx_errno(err);

	if (user_fi_context)
		PSMX_CTXT_REQ(fi_context) = psm_req;

	return 0;
}

static ssize_t psmx_recvfrom(struct fid_ep *ep, void *buf, size_t len, void *desc,
			     const void *src_addr, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	return _psmx_recvfrom(ep, buf, len, desc, src_addr, context, fid_ep->flags, 0);
}

static ssize_t psmx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!msg || msg->iov_count != 1)
		return -EINVAL;

	return _psmx_recvfrom(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			      msg->desc ? msg->desc[0] : NULL, msg->addr,
			      msg->context, flags, msg->data);
}

static ssize_t psmx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	if (fid_ep->connected)
		return psmx_recvfrom(ep, buf, len, desc,
				     fid_ep->peer_psm_epaddr, context);
	else
		return psmx_recvfrom(ep, buf, len, desc, NULL, context);
}

static ssize_t psmx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, void *context)
{
	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!iov || count != 1)
		return -EINVAL;

	return psmx_recv(ep, iov->iov_base, iov->iov_len, desc ? desc[0] : NULL, context);
}

static inline ssize_t _psmx_sendto(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, const void *dest_addr, void *context,
			uint64_t flags)
{
	struct psmx_fid_ep *fid_ep;
	struct psmx_fid_av *fid_av;
	int send_flag = 0;
	psm_epaddr_t psm_epaddr;
	psm_mq_req_t psm_req;
	uint64_t psm_tag;
	struct fi_context * fi_context;
	int user_fi_context = 0;
	int err;
	size_t idx;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);
	assert(fid_ep->domain);

	fid_av = fid_ep->av;
	if (fid_av && fid_av->type == FI_AV_TABLE) {
		idx = (size_t)dest_addr;
		if (idx >= fid_av->last)
			return -EINVAL;

		psm_epaddr = fid_av->psm_epaddrs[idx];
	}
	else  {
		psm_epaddr = (psm_epaddr_t) dest_addr;
	}

	psm_tag = fid_ep->domain->psm_epid | PSMX_MSG_BIT;

	if (flags & FI_BLOCK) {
		err = psm_mq_send(fid_ep->domain->psm_mq, psm_epaddr,
				  send_flag, psm_tag, buf, len);
		if (err == PSM_OK)
			return len;
		else
			return psmx_errno(err);
	}

	if (flags & FI_INJECT) {
		fi_context = malloc(sizeof(*fi_context) + len);
		if (!fi_context)
			return -ENOMEM;

		memcpy((void *)fi_context + sizeof(*fi_context), buf, len);
		buf = (void *)fi_context + sizeof(*fi_context);

		PSMX_CTXT_TYPE(fi_context) = PSMX_INJECT_CONTEXT;
		PSMX_CTXT_EP(fi_context) = fid_ep;
	}
	else if (fid_ep->send_eq_event_flag && !(flags & FI_EVENT) && !context) {
		fi_context = &fid_ep->nocomp_send_context;
	}
	else {
		if (!context)
			return -EINVAL;

		fi_context = context;
		if (fi_context != &fid_ep->sendimm_context) {
			user_fi_context = 1;
			PSMX_CTXT_TYPE(fi_context) = PSMX_SEND_CONTEXT;
			PSMX_CTXT_USER(fi_context) = (void *)buf;
			PSMX_CTXT_EP(fi_context) = fid_ep;
		}
	}

	err = psm_mq_isend(fid_ep->domain->psm_mq, psm_epaddr, send_flag,
				psm_tag, buf, len, (void *)fi_context, &psm_req);

	if (err != PSM_OK)
		return psmx_errno(err);

	fid_ep->pending_sends++;

	if (user_fi_context)
		PSMX_CTXT_REQ(fi_context) = psm_req;

	return 0;
}

static ssize_t psmx_sendto(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, const void *dest_addr, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	return _psmx_sendto(ep, buf, len, desc, dest_addr, context, fid_ep->flags);
}

static ssize_t psmx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!msg || msg->iov_count != 1)
		return -EINVAL;

	return _psmx_sendto(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			    msg->desc ? msg->desc[0] : NULL, msg->addr,
			    msg->context, flags);
}

static ssize_t psmx_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			 void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	if (!fid_ep->connected)
		return -ENOTCONN;

	return psmx_sendto(ep, buf, len, desc, fid_ep->peer_psm_epaddr, context);
}

static ssize_t psmx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, void *context)
{
	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!iov || count != 1)
		return -EINVAL;

	return psmx_send(ep, iov->iov_base, iov->iov_len, desc ? desc[0] : NULL,
			 context);
}

static ssize_t psmx_injectto(struct fid_ep *ep, const void *buf, size_t len,
				const void *dest_addr)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	return _psmx_sendto(ep, buf, len, NULL, dest_addr, NULL,
			    fid_ep->flags | FI_INJECT);
}

static ssize_t psmx_inject(struct fid_ep *ep, const void *buf, size_t len)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(ep, struct psmx_fid_ep, ep);

	if (!fid_ep->connected)
		return -ENOTCONN;

	return psmx_injectto(ep, buf, len, fid_ep->peer_psm_epaddr);
}

struct fi_ops_msg psmx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = psmx_recv,
	.recvv = psmx_recvv,
	.recvfrom = psmx_recvfrom,
	.recvmsg = psmx_recvmsg,
	.send = psmx_send,
	.sendv = psmx_sendv,
	.sendto = psmx_sendto,
	.sendmsg = psmx_sendmsg,
	.inject = psmx_inject,
	.injectto = psmx_injectto,
};

