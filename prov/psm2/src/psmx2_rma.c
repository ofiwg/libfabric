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

static inline void psmx2_am_enqueue_rma(struct psmx2_fid_domain *domain,
				        struct psmx2_am_request *req)
{
	fastlock_acquire(&domain->rma_queue.lock);
	slist_insert_tail(&req->list_entry, &domain->rma_queue.list);
	fastlock_release(&domain->rma_queue.lock);
}

/* RMA protocol:
 *
 * Write REQ:
 *	args[0].u32w0	cmd, flag
 *	args[0].u32w1	len
 *	args[1].u64	req
 *	args[2].u64	addr
 *	args[3].u64	key
 *	args[4].u64	data (optional) / tag(long protocol)
 *	payload		<unused> / data (optional, long protocol)
 *
 * Write REP:
 *	args[0].u32w0	cmd, flag
 *	args[0].u32w1	error
 *	args[1].u64	req
 *
 * Read REQ:
 *	args[0].u32w0	cmd, flag
 *	args[0].u32w1	len
 *	args[1].u64	req
 *	args[2].u64	addr
 *	args[3].u64	key
 *	args[4].u64	offset / tag(long protocol)
 *
 * Read REP:
 *	args[0].u32w0	cmd, flag
 *	args[0].u32w1	error
 *	args[1].u64	req
 *	args[2].u64	offset
 */

int psmx2_am_rma_handler(psm2_am_token_t token, psm2_amarg_t *args,
			 int nargs, void *src, uint32_t len)
{
	psm2_amarg_t rep_args[8];
	void *rma_addr;
	ssize_t rma_len;
	uint64_t key;
	int err = 0;
	int op_error = 0;
	int cmd, eom, has_data;
	struct psmx2_am_request *req;
	struct psmx2_cq_event *event;
	uint64_t offset;
	struct psmx2_fid_mr *mr;
	psm2_epaddr_t epaddr;

	psm2_am_get_source(token, &epaddr);

	cmd = args[0].u32w0 & PSMX2_AM_OP_MASK;
	eom = args[0].u32w0 & PSMX2_AM_EOM;
	has_data = args[0].u32w0 & PSMX2_AM_DATA;

	switch (cmd) {
	case PSMX2_AM_REQ_WRITE:
		rma_len = args[0].u32w1;
		rma_addr = (void *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)rma_addr, len, FI_REMOTE_WRITE) :
			-FI_EINVAL;
		if (!op_error) {
			rma_addr += mr->offset;
			memcpy(rma_addr, src, len);
			if (eom) {
				if (mr->domain->rma_ep->recv_cq && has_data) {
					/* TODO: report the addr/len of the whole write */
					event = psmx2_cq_create_event(
							mr->domain->rma_ep->recv_cq,
							0, /* context */
							rma_addr,
							FI_REMOTE_WRITE | FI_RMA | FI_REMOTE_CQ_DATA,
							rma_len,
							args[4].u64,
							0, /* tag */
							0, /* olen */
							0);

					if (event)
						psmx2_cq_enqueue_event(mr->domain->rma_ep->recv_cq, event);
					else
						err = -FI_ENOMEM;
				}

				if (mr->domain->rma_ep->remote_write_cntr)
					psmx2_cntr_inc(mr->domain->rma_ep->remote_write_cntr);

				if (mr->cntr && mr->cntr != mr->domain->rma_ep->remote_write_cntr)
					psmx2_cntr_inc(mr->cntr);
			}
		}
		if (eom || op_error) {
			rep_args[0].u32w0 = PSMX2_AM_REP_WRITE | eom;
			rep_args[0].u32w1 = op_error;
			rep_args[1].u64 = args[1].u64;
			err = psm2_am_reply_short(token, PSMX2_AM_RMA_HANDLER,
						  rep_args, 2, NULL, 0, 0,
						  NULL, NULL );
		}
		break;

	case PSMX2_AM_REQ_WRITE_LONG:
		rma_len = args[0].u32w1;
		rma_addr = (void *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)rma_addr, rma_len, FI_REMOTE_WRITE) :
			-FI_EINVAL;
		if (op_error) {
			rep_args[0].u32w0 = PSMX2_AM_REP_WRITE | eom;
			rep_args[0].u32w1 = op_error;
			rep_args[1].u64 = args[1].u64;
			err = psm2_am_reply_short(token, PSMX2_AM_RMA_HANDLER,
						  rep_args, 2, NULL, 0, 0,
						  NULL, NULL );
			break;
		}

		rma_addr += mr->offset;

		req = calloc(1, sizeof(*req));
		if (!req) {
			err = -FI_ENOMEM;
		}
		else {
			req->op = args[0].u32w0;
			req->write.addr = (uint64_t)rma_addr;
			req->write.len = rma_len;
			req->write.key = key;
			req->write.context = (void *)args[4].u64;
			req->write.peer_context = (void *)args[1].u64;
			req->write.peer_addr = (void *)epaddr;
			req->write.data = has_data ? *(uint64_t *)src: 0;
			req->cq_flags = FI_REMOTE_WRITE | FI_RMA |
					(has_data ? FI_REMOTE_CQ_DATA : 0),
			PSMX2_CTXT_TYPE(&req->fi_context) = PSMX2_REMOTE_WRITE_CONTEXT;
			PSMX2_CTXT_USER(&req->fi_context) = mr;
			psmx2_am_enqueue_rma(mr->domain, req);
		}
		break;

	case PSMX2_AM_REQ_READ:
		rma_len = args[0].u32w1;
		rma_addr = (void *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		offset = args[4].u64;
		mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)rma_addr, rma_len, FI_REMOTE_READ) :
			-FI_EINVAL;
		if (!op_error) {
			rma_addr += mr->offset;
		}
		else {
			rma_addr = NULL;
			rma_len = 0;
		}

		rep_args[0].u32w0 = PSMX2_AM_REP_READ | eom;
		rep_args[0].u32w1 = op_error;
		rep_args[1].u64 = args[1].u64;
		rep_args[2].u64 = offset;
		err = psm2_am_reply_short(token, PSMX2_AM_RMA_HANDLER,
				rep_args, 3, rma_addr, rma_len, 0,
				NULL, NULL );

		if (eom && !op_error) {
			if (mr->domain->rma_ep->remote_read_cntr)
				psmx2_cntr_inc(mr->domain->rma_ep->remote_read_cntr);
		}
		break;

	case PSMX2_AM_REQ_READ_LONG:
		rma_len = args[0].u32w1;
		rma_addr = (void *)(uintptr_t)args[2].u64;
		key = args[3].u64;
		mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
		op_error = mr ?
			psmx2_mr_validate(mr, (uint64_t)rma_addr, rma_len, FI_REMOTE_READ) :
			-FI_EINVAL;
		if (op_error) {
			rep_args[0].u32w0 = PSMX2_AM_REP_READ | eom;
			rep_args[0].u32w1 = op_error;
			rep_args[1].u64 = args[1].u64;
			rep_args[2].u64 = 0;
			err = psm2_am_reply_short(token, PSMX2_AM_RMA_HANDLER,
					rep_args, 3, NULL, 0, 0,
					NULL, NULL );
			break;
		}

		rma_addr += mr->offset;

		req = calloc(1, sizeof(*req));
		if (!req) {
			err = -FI_ENOMEM;
		}
		else {
			req->op = args[0].u32w0;
			req->read.addr = (uint64_t)rma_addr;
			req->read.len = rma_len;
			req->read.key = key;
			req->read.context = (void *)args[4].u64;
			req->read.peer_addr = (void *)epaddr;
			PSMX2_CTXT_TYPE(&req->fi_context) = PSMX2_REMOTE_READ_CONTEXT;
			PSMX2_CTXT_USER(&req->fi_context) = mr;
			psmx2_am_enqueue_rma(mr->domain, req);
		}
		break;

	case PSMX2_AM_REP_WRITE:
		req = (struct psmx2_am_request *)(uintptr_t)args[1].u64;
		assert(req->op == PSMX2_AM_REQ_WRITE);
		op_error = (int)args[0].u32w1;
		if (!req->error)
			req->error = op_error;
		if (eom) {
			if (req->ep->send_cq && !req->no_event) {
				event = psmx2_cq_create_event(
						req->ep->send_cq,
						req->write.context,
						req->write.buf,
						req->cq_flags,
						req->write.len,
						0, /* data */
						0, /* tag */
						0, /* olen */
						req->error);
				if (event)
					psmx2_cq_enqueue_event(req->ep->send_cq, event);
				else
					err = -FI_ENOMEM;
			}

			if (req->ep->write_cntr)
				psmx2_cntr_inc(req->ep->write_cntr);

			free(req);
		}
		break;

	case PSMX2_AM_REP_READ:
		req = (struct psmx2_am_request *)(uintptr_t)args[1].u64;
		assert(req->op == PSMX2_AM_REQ_READ);
		op_error = (int)args[0].u32w1;
		offset = args[2].u64;
		if (!req->error)
			req->error = op_error;
		if (!op_error) {
			memcpy(req->read.buf + offset, src, len);
			req->read.len_read += len;
		}
		if (eom) {
			if (req->ep->send_cq && !req->no_event) {
				event = psmx2_cq_create_event(
						req->ep->send_cq,
						req->read.context,
						req->read.buf,
						req->cq_flags,
						req->read.len_read,
						0, /* data */
						0, /* tag */
						req->read.len - req->read.len_read,
						req->error);
				if (event)
					psmx2_cq_enqueue_event(req->ep->send_cq, event);
				else
					err = -FI_ENOMEM;
			}

			if (req->ep->read_cntr)
				psmx2_cntr_inc(req->ep->read_cntr);

			free(req);
		}
		break;

	default:
		err = -FI_EINVAL;
	}
	return err;
}

static ssize_t psmx2_rma_self(int am_cmd,
			      struct psmx2_fid_ep *ep,
			      void *buf, size_t len, void *desc,
			      uint64_t addr, uint64_t key,
			      void *context, uint64_t flags, uint64_t data)
{
	struct psmx2_fid_mr *mr;
	struct psmx2_cq_event *event;
	struct psmx2_fid_cntr *cntr;
	struct psmx2_fid_cntr *mr_cntr = NULL;
	struct psmx2_fid_cq *cq = NULL;
	int no_event;
	int err = 0;
	int op_error = 0;
	int access;
	void *dst, *src;
	uint64_t cq_flags;

	switch (am_cmd) {
	case PSMX2_AM_REQ_WRITE:
		access = FI_REMOTE_WRITE;
		cq_flags = FI_WRITE | FI_RMA;
		break;
	case PSMX2_AM_REQ_READ:
		access = FI_REMOTE_READ;
		cq_flags = FI_READ | FI_RMA;
		break;
	default:
		return -FI_EINVAL;
	}

	mr = psmx2_mr_get(psmx2_active_fabric->active_domain, key);
	op_error = mr ? psmx2_mr_validate(mr, addr, len, access) : -FI_EINVAL;

	if (!op_error) {
		addr += mr->offset;
		if (am_cmd == PSMX2_AM_REQ_WRITE) {
			dst = (void *)addr;
			src = buf;
			cntr = mr->domain->rma_ep->remote_write_cntr;
			if (flags & FI_REMOTE_CQ_DATA)
				cq = mr->domain->rma_ep->recv_cq;
			if (mr->cntr != cntr)
				mr_cntr = mr->cntr;
		}
		else {
			dst = buf;
			src = (void *)addr;
			cntr = mr->domain->rma_ep->remote_read_cntr;
		}

		memcpy(dst, src, len);

		if (cq) {
			event = psmx2_cq_create_event(
					cq,
					0, /* context */
					(void *)addr,
					FI_REMOTE_WRITE | FI_RMA | FI_REMOTE_CQ_DATA,
					len,
					data,
					0, /* tag */
					0, /* olen */
					0 /* err */);

			if (event)
				psmx2_cq_enqueue_event(cq, event);
			else
				err = -FI_ENOMEM;
		}

		if (cntr)
			psmx2_cntr_inc(cntr);

		if (mr_cntr)
			psmx2_cntr_inc(mr_cntr);
	}

	no_event = (flags & PSMX2_NO_COMPLETION) ||
		   (ep->send_selective_completion && !(flags & FI_COMPLETION));

	if (ep->send_cq && !no_event) {
		event = psmx2_cq_create_event(
				ep->send_cq,
				context,
				(void *)buf,
				cq_flags,
				len,
				0, /* data */
				0, /* tag */
				0, /* olen */
				op_error);
		if (event)
			psmx2_cq_enqueue_event(ep->send_cq, event);
		else
			err = -FI_ENOMEM;
	}

	switch (am_cmd) {
	case PSMX2_AM_REQ_WRITE:
		if (ep->write_cntr)
			psmx2_cntr_inc(ep->write_cntr);
		break;

	case PSMX2_AM_REQ_READ:
		if (ep->read_cntr)
			psmx2_cntr_inc(ep->read_cntr);
		break;
	}

	return err;
}

void psmx2_am_ack_rma(struct psmx2_am_request *req)
{
	psm2_amarg_t args[8];

	if ((req->op & PSMX2_AM_OP_MASK) != PSMX2_AM_REQ_WRITE_LONG)
		return;

	args[0].u32w0 = PSMX2_AM_REP_WRITE | PSMX2_AM_EOM;
	args[0].u32w1 = req->error;
	args[1].u64 = (uint64_t)(uintptr_t)req->write.peer_context;

	psm2_am_request_short(req->write.peer_addr,
			      PSMX2_AM_RMA_HANDLER, args, 2, NULL, 0,
			      PSM2_AM_FLAG_NOREPLY, NULL, NULL);
}

int psmx2_am_process_rma(struct psmx2_fid_domain *domain,
			 struct psmx2_am_request *req)
{
	int err;
	psm2_mq_req_t psm2_req;

	if ((req->op & PSMX2_AM_OP_MASK) == PSMX2_AM_REQ_WRITE_LONG) {
		err = psm2_mq_irecv(domain->psm2_mq,
				    (uint64_t)req->write.context, -1ULL, 0,
				    (void *)req->write.addr, req->write.len,
				    (void *)&req->fi_context, &psm2_req);
	}
	else {
		err = psm2_mq_isend(domain->psm2_mq,
				    (psm2_epaddr_t)req->read.peer_addr,
				    0, (uint64_t)req->read.context,
				    (void *)req->read.addr, req->read.len,
				    (void *)&req->fi_context, &psm2_req);
	}

	return psmx2_errno(err);
}

ssize_t _psmx2_read(struct fid_ep *ep, void *buf, size_t len,
		    void *desc, fi_addr_t src_addr,
		    uint64_t addr, uint64_t key, void *context,
		    uint64_t flags)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	int chunk_size;
	size_t offset = 0;
	uint64_t psm2_tag;
	psm2_mq_req_t psm2_req;
	size_t idx;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_READ;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->read.ep = ep;
		trigger->read.buf = buf;
		trigger->read.len = len;
		trigger->read.desc = desc;
		trigger->read.src_addr = src_addr;
		trigger->read.addr = addr;
		trigger->read.key = key;
		trigger->read.context = context;
		trigger->read.flags = flags & ~FI_TRIGGER;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!buf)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = src_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		src_addr = (fi_addr_t) av->psm2_epaddrs[idx];
	}
	else if (!src_addr) {
		return -FI_EINVAL;
	}

	epaddr_context = psm2_epaddr_getctxt((void *)src_addr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid)
		return psmx2_rma_self(PSMX2_AM_REQ_READ,
				      ep_priv, buf, len, desc,
				      addr, key, context, flags, 0);

	req = calloc(1, sizeof(*req));
	if (!req)
		return -FI_ENOMEM;

	req->op = PSMX2_AM_REQ_READ;
	req->read.buf = buf;
	req->read.len = len;
	req->read.addr = addr;	/* needed? */
	req->read.key = key; 	/* needed? */
	req->read.context = context;
	req->ep = ep_priv;
	req->cq_flags = FI_READ | FI_RMA;
	PSMX2_CTXT_TYPE(&req->fi_context) = PSMX2_READ_CONTEXT;
	PSMX2_CTXT_USER(&req->fi_context) = context;
	PSMX2_CTXT_EP(&req->fi_context) = ep_priv;

	if (ep_priv->send_selective_completion && !(flags & FI_COMPLETION)) {
		PSMX2_CTXT_TYPE(&req->fi_context) = PSMX2_NOCOMP_READ_CONTEXT;
		req->no_event = 1;
	}

	chunk_size = MIN(PSMX2_AM_CHUNK_SIZE, psmx2_am_param.max_reply_short);

	if (psmx2_env.tagged_rma && len > chunk_size) {
		psm2_tag = PSMX2_RMA_BIT | ep_priv->domain->psm2_epid;
		psm2_mq_irecv(ep_priv->domain->psm2_mq, psm2_tag, -1ULL,
			      0, buf, len, (void *)&req->fi_context, &psm2_req);

		args[0].u32w0 = PSMX2_AM_REQ_READ_LONG;
		args[0].u32w1 = len;
		args[1].u64 = (uint64_t)req;
		args[2].u64 = addr;
		args[3].u64 = key;
		args[4].u64 = psm2_tag;
		psm2_am_request_short((psm2_epaddr_t) src_addr,
				      PSMX2_AM_RMA_HANDLER, args, 5, NULL, 0,
				      0, NULL, NULL);

		return 0;
	}

	args[0].u32w0 = PSMX2_AM_REQ_READ;
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[3].u64 = key;
	while (len > chunk_size) {
		args[0].u32w1 = chunk_size;
		args[2].u64 = addr;
		args[4].u64 = offset;
		psm2_am_request_short((psm2_epaddr_t) src_addr,
				      PSMX2_AM_RMA_HANDLER, args, 5, NULL, 0,
				      0, NULL, NULL);
		addr += chunk_size;
		len -= chunk_size;
		offset += chunk_size;
	}

	args[0].u32w0 = PSMX2_AM_REQ_READ | PSMX2_AM_EOM;
	args[0].u32w1 = len;
	args[2].u64 = addr;
	args[4].u64 = offset;
	psm2_am_request_short((psm2_epaddr_t) src_addr,
			      PSMX2_AM_RMA_HANDLER, args, 5, NULL, 0,
			      0, NULL, NULL);

	return 0;
}

static ssize_t psmx2_read(struct fid_ep *ep, void *buf, size_t len,
			  void *desc, fi_addr_t src_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_read(ep, buf, len, desc, src_addr, addr,
			   key, context, ep_priv->flags);
}

static ssize_t psmx2_readmsg(struct fid_ep *ep,
			     const struct fi_msg_rma *msg,
			     uint64_t flags)
{
	if (!msg || msg->iov_count != 1 || !msg->msg_iov || !msg->rma_iov)
		return -FI_EINVAL;

	return _psmx2_read(ep, msg->msg_iov[0].iov_base,
			   msg->msg_iov[0].iov_len,
			   msg->desc ? msg->desc[0] : NULL, msg->addr,
			   msg->rma_iov[0].addr, msg->rma_iov[0].key,
			   msg->context, flags);
}

static ssize_t psmx2_readv(struct fid_ep *ep, const struct iovec *iov,
		           void **desc, size_t count, fi_addr_t src_addr,
		           uint64_t addr, uint64_t key, void *context)
{
	if (!iov || count != 1)
		return -FI_EINVAL;

	return psmx2_read(ep, iov->iov_base, iov->iov_len,
			  desc ? desc[0] : NULL, src_addr, addr, key, context);
}

ssize_t _psmx2_write(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr,
		     uint64_t addr, uint64_t key, void *context,
		     uint64_t flags, uint64_t data)
{
	struct psmx2_fid_ep *ep_priv;
	struct psmx2_fid_av *av;
	struct psmx2_epaddr_context *epaddr_context;
	struct psmx2_am_request *req;
	psm2_amarg_t args[8];
	int nargs;
	int am_flags = PSM2_AM_FLAG_ASYNC;
	int chunk_size;
	psm2_mq_req_t psm2_req;
	uint64_t psm2_tag;
	size_t idx;
	void *psm2_context;
	int no_event;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	if (flags & FI_TRIGGER) {
		struct psmx2_trigger *trigger;
		struct fi_triggered_context *ctxt = context;

		trigger = calloc(1, sizeof(*trigger));
		if (!trigger)
			return -FI_ENOMEM;

		trigger->op = PSMX2_TRIGGERED_WRITE;
		trigger->cntr = container_of(ctxt->trigger.threshold.cntr,
					     struct psmx2_fid_cntr, cntr);
		trigger->threshold = ctxt->trigger.threshold.threshold;
		trigger->write.ep = ep;
		trigger->write.buf = buf;
		trigger->write.len = len;
		trigger->write.desc = desc;
		trigger->write.dest_addr = dest_addr;
		trigger->write.addr = addr;
		trigger->write.key = key;
		trigger->write.context = context;
		trigger->write.flags = flags & ~FI_TRIGGER;
		trigger->write.data = data;

		psmx2_cntr_add_trigger(trigger->cntr, trigger);
		return 0;
	}

	if (!buf)
		return -FI_EINVAL;

	av = ep_priv->av;
	if (av && av->type == FI_AV_TABLE) {
		idx = dest_addr;
		if (idx >= av->last)
			return -FI_EINVAL;

		dest_addr = (fi_addr_t) av->psm2_epaddrs[idx];
	}
	else if (!dest_addr) {
		return -FI_EINVAL;
	}

	epaddr_context = psm2_epaddr_getctxt((void *)dest_addr);
	if (epaddr_context->epid == ep_priv->domain->psm2_epid)
		return psmx2_rma_self(PSMX2_AM_REQ_WRITE,
				      ep_priv, (void *)buf, len, desc,
				      addr, key, context, flags, data);

	no_event = (flags & PSMX2_NO_COMPLETION) ||
		   (ep_priv->send_selective_completion && !(flags & FI_COMPLETION));

	if (flags & FI_INJECT) {
		if (len > PSMX2_INJECT_SIZE)
			return -FI_EMSGSIZE;

		req = malloc(sizeof(*req) + len);
		if (!req)
			return -FI_ENOMEM;

		memset((void *)req, 0, sizeof(*req));
		memcpy((void *)req + sizeof(*req), (void *)buf, len);
		buf = (void *)req + sizeof(*req);
	}
	else {
		req = calloc(1, sizeof(*req));
		if (!req)
			return -FI_ENOMEM;

		PSMX2_CTXT_TYPE(&req->fi_context) = no_event ?
						    PSMX2_NOCOMP_WRITE_CONTEXT :
						    PSMX2_WRITE_CONTEXT;
	}

	req->no_event = no_event;
	req->op = PSMX2_AM_REQ_WRITE;
	req->write.buf = (void *)buf;
	req->write.len = len;
	req->write.addr = addr;	/* needed? */
	req->write.key = key; 	/* needed? */
	req->write.context = context;
	req->ep = ep_priv;
	req->cq_flags = FI_WRITE | FI_RMA;
	PSMX2_CTXT_USER(&req->fi_context) = context;
	PSMX2_CTXT_EP(&req->fi_context) = ep_priv;

	chunk_size = MIN(PSMX2_AM_CHUNK_SIZE, psmx2_am_param.max_request_short);

	if (psmx2_env.tagged_rma && len > chunk_size) {
		void *payload = NULL;
		int payload_len = 0;

		psm2_tag = PSMX2_RMA_BIT | ep_priv->domain->psm2_epid;
		args[0].u32w0 = PSMX2_AM_REQ_WRITE_LONG;
		args[0].u32w1 = len;
		args[1].u64 = (uint64_t)req;
		args[2].u64 = addr;
		args[3].u64 = key;
		args[4].u64 = psm2_tag;
		nargs = 5;
		if (flags & FI_REMOTE_CQ_DATA) {
			args[0].u32w0 |= PSMX2_AM_DATA;
			payload = &data;
			payload_len = sizeof(data);
			am_flags = 0;
		}

		if (flags & FI_DELIVERY_COMPLETE) {
			args[0].u32w0 |= PSMX2_AM_FORCE_ACK;
			psm2_context = NULL;
		}
		else {
			psm2_context = (void *)&req->fi_context;
		}

		/* NOTE: if nargs is greater than 5, the following psm2_mq_isend
		 * would hang if the destination is on the same node (i.e. going
		 * through the shared memory path). As the result, the immediate
		 * data is sent as payload instead of args[5].
		 */
		psm2_am_request_short((psm2_epaddr_t) dest_addr,
				      PSMX2_AM_RMA_HANDLER, args, nargs,
				      payload, payload_len, am_flags,
				      NULL, NULL);

		psm2_mq_isend(ep_priv->domain->psm2_mq, (psm2_epaddr_t) dest_addr,
			      0, psm2_tag, buf, len, psm2_context, &psm2_req);

		return 0;
	}

	nargs = 4;
	while (len > chunk_size) {
		args[0].u32w0 = PSMX2_AM_REQ_WRITE;
		args[0].u32w1 = chunk_size;
		args[1].u64 = (uint64_t)(uintptr_t)req;
		args[2].u64 = addr;
		args[3].u64 = key;
		psm2_am_request_short((psm2_epaddr_t) dest_addr,
				      PSMX2_AM_RMA_HANDLER, args, nargs,
				      (void *)buf, chunk_size,
				      am_flags, NULL, NULL);
		buf += chunk_size;
		addr += chunk_size;
		len -= chunk_size;
	}

	args[0].u32w0 = PSMX2_AM_REQ_WRITE | PSMX2_AM_EOM;
	args[0].u32w1 = len;
	args[1].u64 = (uint64_t)(uintptr_t)req;
	args[2].u64 = addr;
	args[3].u64 = key;
	if (flags & FI_REMOTE_CQ_DATA) {
		args[4].u64 = data;
		args[0].u32w0 |= PSMX2_AM_DATA;
		nargs++;
	}
	psm2_am_request_short((psm2_epaddr_t) dest_addr,
			      PSMX2_AM_RMA_HANDLER, args, nargs,
			      (void *)buf, len, am_flags, NULL, NULL);

	return 0;
}

static ssize_t psmx2_write(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, uint64_t addr,
			   uint64_t key, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_write(ep, buf, len, desc, dest_addr, addr, key, context,
			    ep_priv->flags, 0);
}

static ssize_t psmx2_writemsg(struct fid_ep *ep,
			      const struct fi_msg_rma *msg,
			      uint64_t flags)
{
	if (!msg || msg->iov_count != 1 || !msg->msg_iov || !msg->rma_iov)
		return -FI_EINVAL;

	return _psmx2_write(ep, msg->msg_iov[0].iov_base,
			    msg->msg_iov[0].iov_len,
			    msg->desc ? msg->desc[0] : NULL, msg->addr,
			    msg->rma_iov[0].addr, msg->rma_iov[0].key,
			    msg->context, flags, msg->data);
}

static ssize_t psmx2_writev(struct fid_ep *ep, const struct iovec *iov,
		            void **desc, size_t count, fi_addr_t dest_addr,
		            uint64_t addr, uint64_t key, void *context)
{
	if (!iov || count != 1)
		return -FI_EINVAL;

	return psmx2_write(ep, iov->iov_base, iov->iov_len,
			   desc ? desc[0] : NULL, dest_addr, addr, key, context);
}

static ssize_t psmx2_inject(struct fid_ep *ep, const void *buf, size_t len,
			    fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_write(ep, buf, len, NULL, dest_addr, addr, key,
			    NULL, ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION, 0);
}

static ssize_t psmx2_writedata(struct fid_ep *ep, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key, void *context)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_write(ep, buf, len, desc, dest_addr, addr, key, context,
			    ep_priv->flags | FI_REMOTE_CQ_DATA, data);
}

static ssize_t psmx2_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			        uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			        uint64_t key)
{
	struct psmx2_fid_ep *ep_priv;

	ep_priv = container_of(ep, struct psmx2_fid_ep, ep);

	return _psmx2_write(ep, buf, len, NULL, dest_addr, addr, key, NULL,
			    ep_priv->flags | FI_INJECT | PSMX2_NO_COMPLETION,
			    data);
}

struct fi_ops_rma psmx2_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = psmx2_read,
	.readv = psmx2_readv,
	.readmsg = psmx2_readmsg,
	.write = psmx2_write,
	.writev = psmx2_writev,
	.writemsg = psmx2_writemsg,
	.inject = psmx2_inject,
	.writedata = psmx2_writedata,
	.injectdata = psmx2_injectdata,
};

