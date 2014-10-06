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

#include "psmx.h"

void psmx_cq_enqueue_event(struct psmx_cq_event_queue *ceq,
			   struct psmx_cq_event *event)
{
	if (ceq->tail) {
		ceq->tail->next = event;
		ceq->tail = event;
	}
	else {
		ceq->head = ceq->tail = event;
	}
}

static struct psmx_cq_event *psmx_cq_dequeue_event(struct psmx_cq_event_queue *ceq)
{
	struct psmx_cq_event *event;

	if (!ceq->head)
		return NULL;

	event = ceq->head;
	ceq->head = event->next;
	if (!ceq->head)
		ceq->tail = NULL;

	event->next = NULL;
	return event;
}

struct psmx_cq_event *psmx_cq_create_event(enum fi_cq_format format,
					   void *op_context, void *buf,
					   uint64_t flags, size_t len,
					   uint64_t data, uint64_t tag,
					   size_t olen, int err)
{
	struct psmx_cq_event *event;

	event = calloc(1, sizeof(*event));
	if (!event) {
		fprintf(stderr, "%s: out of memory\n", __func__);
		return NULL;
	}

	if ((event->error = !!err)) {
		event->cqe.err.op_context = op_context;
		event->cqe.err.err = -err;
		event->cqe.err.data = data;
		event->cqe.err.tag = tag;
		event->cqe.err.olen = olen;
		event->cqe.err.prov_errno = 0;
		goto out;
	}

	switch (format) {
	case FI_CQ_FORMAT_CONTEXT:
		event->cqe.context.op_context = op_context;
		break;

	case FI_CQ_FORMAT_MSG:
		event->cqe.msg.op_context = op_context;
		event->cqe.msg.flags = flags;
		event->cqe.msg.len = len;
		break;

	case FI_CQ_FORMAT_DATA:
		event->cqe.data.op_context = op_context;
		event->cqe.data.buf = buf;
		event->cqe.data.flags = flags;
		event->cqe.data.len = len;
		event->cqe.data.data = data;
		break;

	case FI_CQ_FORMAT_TAGGED:
		event->cqe.tagged.op_context = op_context;
		event->cqe.tagged.buf = buf;
		event->cqe.tagged.flags = flags;
		event->cqe.tagged.len = len;
		event->cqe.tagged.data = data;
		event->cqe.tagged.tag = tag;
		break;

	default:
		fprintf(stderr, "%s: unsupported CC format %d\n", __func__, format);
		return NULL;
	}

out:
	return event;
}

static struct psmx_cq_event *psmx_cq_create_event_from_status(
				enum fi_cq_format format,
				psm_mq_status_t *psm_status)
{
	struct psmx_cq_event *event;
	struct psmx_multi_recv *req;
	struct fi_context *fi_context = psm_status->context;
	void *op_context, *buf;
	int is_recv = 0;

	event = calloc(1, sizeof(*event));
	if (!event) {
		fprintf(stderr, "%s: out of memory\n", __func__);
		return NULL;
	}

	switch(PSMX_CTXT_TYPE(fi_context)) {
	case PSMX_SEND_CONTEXT:
		op_context = fi_context;
		buf = PSMX_CTXT_USER(fi_context);
		break;
	case PSMX_RECV_CONTEXT:
		op_context = fi_context;
		buf = PSMX_CTXT_USER(fi_context);
		is_recv = 1;
		break;
	case PSMX_MULTI_RECV_CONTEXT:
		op_context = fi_context;
		req = PSMX_CTXT_USER(fi_context);
		buf = req->buf + req->offset;
		is_recv = 1;
		break;
	default:
		op_context = PSMX_CTXT_USER(fi_context);
		buf = NULL;
		break;
	}

	if ((event->error = !!psm_status->error_code)) {
		event->cqe.err.op_context = op_context;
		event->cqe.err.err = -psmx_errno(psm_status->error_code);
		event->cqe.err.prov_errno = psm_status->error_code;
		event->cqe.err.tag = psm_status->msg_tag;
		event->cqe.err.olen = psm_status->msg_length - psm_status->nbytes;
		goto out;
	}

	switch (format) {
	case FI_CQ_FORMAT_CONTEXT:
		event->cqe.context.op_context = op_context;
		break;

	case FI_CQ_FORMAT_MSG:
		event->cqe.msg.op_context = op_context;
		event->cqe.msg.len = psm_status->nbytes;
		break;

	case FI_CQ_FORMAT_DATA:
		event->cqe.data.op_context = op_context;
		event->cqe.data.buf = buf;
		event->cqe.data.len = psm_status->nbytes;
		break;

	case FI_CQ_FORMAT_TAGGED:
		event->cqe.tagged.op_context = op_context;
		event->cqe.tagged.buf = buf;
		event->cqe.tagged.len = psm_status->nbytes;
		event->cqe.tagged.tag = psm_status->msg_tag;
		break;

	default:
		fprintf(stderr, "%s: unsupported EQ format %d\n", __func__, format);
		return NULL;
	}

out:
	if (is_recv)
		event->source = psm_status->msg_tag;

	return event;
}

static int psmx_cq_get_event_src_addr(struct psmx_fid_cq *cq,
				      struct psmx_cq_event *event,
				      fi_addr_t *src_addr)
{
	int err;

	if (!src_addr)
		return 0;

	if ((cq->domain->reserved_tag_bits & PSMX_MSG_BIT) &&
		(event->source & PSMX_MSG_BIT)) {
		err = psmx_epid_to_epaddr(cq->domain,
					  event->source & ~PSMX_MSG_BIT,
					  (psm_epaddr_t *) src_addr);
		if (err)
			return err;

		return 0;
	}

	return -ENODATA;
}

int psmx_cq_poll_mq(struct psmx_fid_cq *cq, struct psmx_fid_domain *domain)
{
	psm_mq_req_t psm_req;
	psm_mq_status_t psm_status;
	struct fi_context *fi_context;
	struct psmx_fid_ep *tmp_ep;
	struct psmx_fid_cq *tmp_cq;
	struct psmx_fid_cntr *tmp_cntr;
	struct psmx_cq_event *event;
	int multi_recv;
	int err;

	while (1) {
		err = psm_mq_ipeek(domain->psm_mq, &psm_req, NULL);

		if (err == PSM_OK) {
			err = psm_mq_test(&psm_req, &psm_status);

			fi_context = psm_status.context;

			tmp_ep = PSMX_CTXT_EP(fi_context);
			tmp_cq = NULL;
			tmp_cntr = NULL;
			multi_recv = 0;

			switch (PSMX_CTXT_TYPE(fi_context)) {
			case PSMX_NOCOMP_SEND_CONTEXT:
				tmp_ep->pending_sends--;
				if (!tmp_ep->send_cntr_event_flag)
					tmp_cntr = tmp_ep->send_cntr;
				break;

			case PSMX_NOCOMP_RECV_CONTEXT:
				if (!tmp_ep->recv_cntr_event_flag)
					tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX_NOCOMP_WRITE_CONTEXT:
				tmp_ep->pending_writes--;
				if (!tmp_ep->write_cntr_event_flag)
					tmp_cntr = tmp_ep->write_cntr;
				break;

			case PSMX_NOCOMP_READ_CONTEXT:
				tmp_ep->pending_reads--;
				if (!tmp_ep->read_cntr_event_flag)
					tmp_cntr = tmp_ep->read_cntr;
				break;

			case PSMX_INJECT_CONTEXT:
				tmp_ep->pending_sends--;
				if (!tmp_ep->send_cntr_event_flag)
					tmp_cntr = tmp_ep->send_cntr;
				free(fi_context);
				break;

			case PSMX_INJECT_WRITE_CONTEXT:
				tmp_ep->pending_writes--;
				if (!tmp_ep->write_cntr_event_flag)
					tmp_cntr = tmp_ep->write_cntr;
				free(fi_context);
				break;

			case PSMX_SEND_CONTEXT:
				tmp_ep->pending_sends--;
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->send_cntr;
				break;

			case PSMX_RECV_CONTEXT:
				tmp_cq = tmp_ep->recv_cq;
				tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX_MULTI_RECV_CONTEXT:
				multi_recv = 1;
				tmp_cq = tmp_ep->recv_cq;
				tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX_READ_CONTEXT:
				tmp_ep->pending_reads--;
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->read_cntr;
				break;

			case PSMX_WRITE_CONTEXT:
				tmp_ep->pending_writes--;
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->write_cntr;
				break;

			case PSMX_REMOTE_WRITE_CONTEXT:
			case PSMX_REMOTE_READ_CONTEXT:
				{
				  struct fi_context *fi_context = psm_status.context;
				  struct psmx_fid_mr *mr;
				  mr = PSMX_CTXT_USER(fi_context);
				  if (mr->cq) {
					event = psmx_cq_create_event_from_status(
							mr->cq->format, &psm_status);
					if (!event)
						return -ENOMEM;
					psmx_cq_enqueue_event(&mr->cq->event_queue, event);
				  }
				  if (mr->cntr)
					mr->cntr->cntr.ops->add(&tmp_cntr->cntr, 1);
				  if (!cq || mr->cq == cq)
					return 1;
				  continue;
				}
			}

			if (tmp_cq) {
				event = psmx_cq_create_event_from_status(tmp_cq->format, &psm_status);
				if (!event)
					return -ENOMEM;

				psmx_cq_enqueue_event(&tmp_cq->event_queue, event);
			}

			if (tmp_cntr)
				tmp_cntr->cntr.ops->add(&tmp_cntr->cntr, 1);

			if (multi_recv) {
				struct psmx_multi_recv *req;
				psm_mq_req_t psm_req;

				req = PSMX_CTXT_USER(fi_context);
				req->offset += psm_status.nbytes;
				if (req->offset + req->min_buf_size <= req->len) {
					err = psm_mq_irecv(tmp_ep->domain->psm_mq,
							   req->tag, req->tagsel, req->flag,
							   req->buf + req->offset, 
							   req->len - req->offset,
							   (void *)fi_context, &psm_req);
					if (err != PSM_OK)
						return psmx_errno(err);

					PSMX_CTXT_REQ(fi_context) = psm_req;
				}
				else {
					if (tmp_cq) {
						event = psmx_cq_create_event(
								tmp_cq->format,
								req->context,
								req->buf,
								FI_MULTI_RECV,
								req->len,
								req->len - req->offset, /* data */
								0,	/* tag */
								0,	/* olen */
								0);	/* err */
						if (!event)
							return -ENOMEM;

						psmx_cq_enqueue_event(&tmp_cq->event_queue, event);
					}

					free(req);
				}
			}

			if (!cq || tmp_cq == cq)
				return 1;
		}
		else if (err == PSM_MQ_NO_COMPLETIONS) {
			return 0;
		}
		else {
			return psmx_errno(err);
		}
	}
}

static ssize_t psmx_cq_readfrom(struct fid_cq *cq, void *buf, size_t len,
				fi_addr_t *src_addr)
{
	struct psmx_fid_cq *cq_priv;
	struct psmx_cq_event *event;

	cq_priv = container_of(cq, struct psmx_fid_cq, cq);
	assert(cq_priv->domain);

	cq_priv->poll_am_before_mq = !cq_priv->poll_am_before_mq;
	if (cq_priv->poll_am_before_mq)
		psmx_am_progress(cq_priv->domain);

	psmx_cq_poll_mq(cq_priv, cq_priv->domain);

	if (!cq_priv->poll_am_before_mq)
		psmx_am_progress(cq_priv->domain);

	if (cq_priv->pending_error)
		return -FI_EAVAIL;

	if (len < cq_priv->entry_size)
		return -FI_ETOOSMALL;

	if (!buf)
		return -FI_EINVAL;

	event = psmx_cq_dequeue_event(&cq_priv->event_queue);
	if (event) {
		if (!event->error) {
			memcpy(buf, (void *)&event->cqe, cq_priv->entry_size);
			if (psmx_cq_get_event_src_addr(cq_priv, event, src_addr))
				*src_addr = FI_ADDR_UNSPEC;
			free(event);
			return cq_priv->entry_size;
		}
		else {
			cq_priv->pending_error = event;
			return -FI_EAVAIL;
		}
	}

	return 0;
}

static ssize_t psmx_cq_read(struct fid_cq *cq, void *buf, size_t len)
{
	return psmx_cq_readfrom(cq, buf, len, NULL);
}

static ssize_t psmx_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			       size_t len, uint64_t flags)
{
	struct psmx_fid_cq *cq_priv;

	cq_priv = container_of(cq, struct psmx_fid_cq, cq);

	if (len < sizeof *buf)
		return -FI_ETOOSMALL;

	if (cq_priv->pending_error) {
		memcpy(buf, &cq_priv->pending_error->cqe, sizeof *buf);
		free(cq_priv->pending_error);
		cq_priv->pending_error = NULL;
		return sizeof *buf;
	}

	return 0;
}

static ssize_t psmx_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t len,
				 fi_addr_t *src_addr, const void *cond,
				 int timeout)
{
	return -FI_ENOSYS;
}

static ssize_t psmx_cq_sread(struct fid_cq *cq, void *buf, size_t len,
			     const void *cond, int timeout)
{
	return psmx_cq_sreadfrom(cq, buf, len, NULL, cond, timeout);
}

static const char *psmx_cq_strerror(struct fid_cq *cq, int prov_errno, const void *prov_data,
				    void *buf, size_t len)
{
	return psm_error_get_string(prov_errno);
}

static int psmx_cq_close(fid_t fid)
{
	struct psmx_fid_cq *cq;

	cq = container_of(fid, struct psmx_fid_cq, cq.fid);
	free(cq);

	return 0;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_cq_close,
	.bind = fi_no_bind,
	.sync = fi_no_sync,
	.control = fi_no_control,
};

static struct fi_ops_cq psmx_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = psmx_cq_read,
	.readfrom = psmx_cq_readfrom,
	.readerr = psmx_cq_readerr,
	.write = fi_no_cq_write,
	.sread = psmx_cq_sread,
	.sreadfrom = psmx_cq_sreadfrom,
	.strerror = psmx_cq_strerror,
};

int psmx_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct psmx_fid_domain *domain_priv;
	struct psmx_fid_cq *cq_priv;
	int entry_size;

	if (attr->flags & FI_WRITE)
		return -FI_ENOSYS;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
		attr->format = FI_CQ_FORMAT_TAGGED;
		entry_size = sizeof(struct fi_cq_tagged_entry);
		break;

	case FI_CQ_FORMAT_CONTEXT:
		entry_size = sizeof(struct fi_cq_entry);
		break;

	case FI_CQ_FORMAT_MSG:
		entry_size = sizeof(struct fi_cq_msg_entry);
		break;

	case FI_CQ_FORMAT_DATA:
		entry_size = sizeof(struct fi_cq_data_entry);
		break;

	case FI_CQ_FORMAT_TAGGED:
		entry_size = sizeof(struct fi_cq_tagged_entry);
		break;

	default:
		psmx_debug("%s: attr->format=%d, supported=%d...%d\n", __func__, attr->format,
				FI_CQ_FORMAT_UNSPEC, FI_CQ_FORMAT_TAGGED);
		return -FI_EINVAL;
	}

	domain_priv = container_of(domain, struct psmx_fid_domain, domain);
	cq_priv = (struct psmx_fid_cq *) calloc(1, sizeof *cq_priv);
	if (!cq_priv)
		return -FI_ENOMEM;

	cq_priv->domain = domain_priv;
	cq_priv->format = attr->format;
	cq_priv->entry_size = entry_size;
	cq_priv->cq.fid.fclass = FI_CLASS_CQ;
	cq_priv->cq.fid.context = context;
	cq_priv->cq.fid.ops = &psmx_fi_ops;
	cq_priv->cq.ops = &psmx_cq_ops;

	*cq = &cq_priv->cq;
	return 0;
}

