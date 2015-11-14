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

void psmx2_cq_enqueue_event(struct psmx2_fid_cq *cq,
			    struct psmx2_cq_event *event)
{
	fastlock_acquire(&cq->lock);
	slist_insert_tail(&event->list_entry, &cq->event_queue);
	cq->event_count++;
	fastlock_release(&cq->lock);

	if (cq->wait)
		psmx2_wait_signal((struct fid_wait *)cq->wait);
}

static struct psmx2_cq_event *psmx2_cq_dequeue_event(struct psmx2_fid_cq *cq)
{
	struct slist_entry *entry;

	if (slist_empty(&cq->event_queue))
		return NULL;

	fastlock_acquire(&cq->lock);
	entry = slist_remove_head(&cq->event_queue);
	cq->event_count--;
	fastlock_release(&cq->lock);

	return container_of(entry, struct psmx2_cq_event, list_entry);
}

static struct psmx2_cq_event *psmx2_cq_alloc_event(struct psmx2_fid_cq *cq)
{
	struct psmx2_cq_event *event;

	fastlock_acquire(&cq->lock);
	if (!slist_empty(&cq->free_list)) {
		event = container_of(slist_remove_head(&cq->free_list),
				     struct psmx2_cq_event, list_entry);
		fastlock_release(&cq->lock);
		return event;
	}

	fastlock_release(&cq->lock);
	event = calloc(1, sizeof(*event));
	if (!event)
		FI_WARN(&psmx2_prov, FI_LOG_CQ, "out of memory.\n");

	return event;
}

static void psmx2_cq_free_event(struct psmx2_fid_cq *cq,
				struct psmx2_cq_event *event)
{
	memset(event, 0, sizeof(*event));

	fastlock_acquire(&cq->lock);
	slist_insert_tail(&event->list_entry, &cq->free_list);
	fastlock_release(&cq->lock);
}

struct psmx2_cq_event *psmx2_cq_create_event(struct psmx2_fid_cq *cq,
					     void *op_context, void *buf,
					     uint64_t flags, size_t len,
					     uint64_t data, uint64_t tag,
					     size_t olen, int err)
{
	struct psmx2_cq_event *event;

	event = psmx2_cq_alloc_event(cq);
	if (!event)
		return NULL;

	if ((event->error = !!err)) {
		event->cqe.err.op_context = op_context;
		event->cqe.err.err = -err;
		event->cqe.err.data = data;
		event->cqe.err.tag = tag;
		event->cqe.err.olen = olen;
		event->cqe.err.prov_errno = PSM2_INTERNAL_ERR;
		goto out;
	}

	switch (cq->format) {
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
		FI_WARN(&psmx2_prov, FI_LOG_CQ,
			"unsupported CQ format %d\n", cq->format);
		psmx2_cq_free_event(cq, event);
		return NULL;
	}

out:
	return event;
}

static struct psmx2_cq_event *
psmx2_cq_create_event_from_status(struct psmx2_fid_cq *cq,
				  psm2_mq_status2_t *psm2_status,
				  uint64_t data,
				  struct psmx2_cq_event *event_in,
				  int count,
				  fi_addr_t *src_addr)
{
	struct psmx2_cq_event *event;
	struct psmx2_multi_recv *req;
	struct fi_context *fi_context = psm2_status->context;
	void *op_context, *buf;
	int is_recv = 0;
	uint64_t flags;

	switch((int)PSMX2_CTXT_TYPE(fi_context)) {
	case PSMX2_SEND_CONTEXT:
		op_context = fi_context;
		buf = PSMX2_CTXT_USER(fi_context);
		flags = FI_SEND | FI_MSG;
		break;
	case PSMX2_RECV_CONTEXT:
		op_context = fi_context;
		buf = PSMX2_CTXT_USER(fi_context);
		flags = FI_RECV | FI_MSG;
		is_recv = 1;
		break;
	case PSMX2_MULTI_RECV_CONTEXT:
		op_context = fi_context;
		req = PSMX2_CTXT_USER(fi_context);
		buf = req->buf + req->offset;
		flags = FI_RECV | FI_MSG;
		if (req->offset + psm2_status->nbytes + req->min_buf_size > req->len)
			flags |= FI_MULTI_RECV;	/* buffer used up */
		is_recv = 1;
		break;
	case PSMX2_TSEND_CONTEXT:
		op_context = fi_context;
		buf = PSMX2_CTXT_USER(fi_context);
		flags = FI_SEND | FI_TAGGED;
		break;
	case PSMX2_TRECV_CONTEXT:
		op_context = fi_context;
		buf = PSMX2_CTXT_USER(fi_context);
		flags = FI_RECV | FI_TAGGED;
		is_recv = 1;
		break;
	case PSMX2_READ_CONTEXT:
		op_context = PSMX2_CTXT_USER(fi_context);
		buf = NULL;
		flags = FI_READ | FI_RMA;
		break;
	case PSMX2_WRITE_CONTEXT:
		op_context = PSMX2_CTXT_USER(fi_context);
		buf = NULL;
		flags = FI_WRITE | FI_RMA;
		break;
	case PSMX2_REMOTE_READ_CONTEXT:
		op_context = NULL;
		buf = NULL;
		flags = FI_REMOTE_READ | FI_RMA;
		break;
	case PSMX2_REMOTE_WRITE_CONTEXT:
		op_context = NULL;
		buf = NULL;
		flags = FI_REMOTE_WRITE | FI_RMA | FI_REMOTE_CQ_DATA;
		break;
	default:
		op_context = PSMX2_CTXT_USER(fi_context);
		buf = NULL;
		flags = 0;
		break;
	}

	/* NOTE: "event_in" only has space for the CQE of the current CQ format.
	 * Fields like "error_code" and "source" should not be filled in.
	 */
	if (event_in && count && !psm2_status->error_code) {
		event = event_in;
	}
	else {
		event = psmx2_cq_alloc_event(cq);
		if (!event)
			return NULL;

		event->error = !!psm2_status->error_code;
	}

	if (psm2_status->error_code) {
		event->cqe.err.op_context = op_context;
		event->cqe.err.flags = flags;
		event->cqe.err.err = -psmx2_errno(psm2_status->error_code);
		event->cqe.err.prov_errno = psm2_status->error_code;
		event->cqe.err.tag = psm2_status->msg_tag.tag0 |
				     (((uint64_t)psm2_status->msg_tag.tag1) << 32);
		event->cqe.err.olen = psm2_status->msg_length - psm2_status->nbytes;
		if (data)
			event->cqe.err.data = data;
		goto out;
	}

	switch (cq->format) {
	case FI_CQ_FORMAT_CONTEXT:
		event->cqe.context.op_context = op_context;
		break;

	case FI_CQ_FORMAT_MSG:
		event->cqe.msg.op_context = op_context;
		event->cqe.msg.flags = flags;
		event->cqe.msg.len = psm2_status->nbytes;
		break;

	case FI_CQ_FORMAT_DATA:
		event->cqe.data.op_context = op_context;
		event->cqe.data.buf = buf;
		event->cqe.data.flags = flags;
		event->cqe.data.len = psm2_status->nbytes;
		event->cqe.data.data = psm2_status->msg_tag.tag2;
		if (data)
			event->cqe.data.data = data;
		break;

	case FI_CQ_FORMAT_TAGGED:
		event->cqe.tagged.op_context = op_context;
		event->cqe.tagged.buf = buf;
		event->cqe.tagged.flags = flags;
		event->cqe.tagged.len = psm2_status->nbytes;
		event->cqe.data.data = psm2_status->msg_tag.tag2;
		event->cqe.tagged.tag = psm2_status->msg_tag.tag0 |
					(((uint64_t)psm2_status->msg_tag.tag1) << 32);
		if (data)
			event->cqe.tagged.data = data;
		break;

	default:
		FI_WARN(&psmx2_prov, FI_LOG_CQ,
			"unsupported CQ format %d\n", cq->format);
		if (event != event_in)
			psmx2_cq_free_event(cq, event);
		return NULL;
	}

out:
	if (is_recv) {
		if (event == event_in) {
			if (src_addr)
				*src_addr = (fi_addr_t) psm2_status->msg_peer;
		}
		else {
			event->source = (uint64_t) psm2_status->msg_peer;
		}
	}

	return event;
}

int psmx2_cq_poll_mq(struct psmx2_fid_cq *cq,
		     struct psmx2_fid_domain *domain,
		     struct psmx2_cq_event *event_in,
		     int count, fi_addr_t *src_addr)
{
	psm2_mq_req_t psm2_req;
	psm2_mq_status2_t psm2_status;
	struct fi_context *fi_context;
	struct psmx2_fid_ep *tmp_ep;
	struct psmx2_fid_cq *tmp_cq;
	struct psmx2_fid_cntr *tmp_cntr;
	struct psmx2_cq_event *event;
	int multi_recv;
	int err;
	int read_more = 1;
	int read_count = 0;
	void *event_buffer = count ? event_in : NULL;

	while (1) {
		/* psm2_mq_ipeek and psm2_mq_test is suposed to be called
		 * in sequence. If the same sequence from different threads
		 * are interleaved the behavior is errorous: the second
		 * psm2_mq_test could derefernce a request that has been
		 * freed because the two psm2_mq_ipeek calls may return the
		 * same request. Use a lock to ensure that won't happen.
		 */
		if (fastlock_tryacquire(&domain->poll_lock))
			return read_count;

		err = psm2_mq_ipeek(domain->psm2_mq, &psm2_req, NULL);

		if (err == PSM2_OK) {
			err = psm2_mq_test2(&psm2_req, &psm2_status);
			fastlock_release(&domain->poll_lock);

			fi_context = psm2_status.context;

			if (!fi_context)
				continue;

			tmp_ep = PSMX2_CTXT_EP(fi_context);
			tmp_cq = NULL;
			tmp_cntr = NULL;
			multi_recv = 0;

			switch ((int)PSMX2_CTXT_TYPE(fi_context)) {
			case PSMX2_NOCOMP_SEND_CONTEXT:
				tmp_cntr = tmp_ep->send_cntr;
				break;

			case PSMX2_NOCOMP_RECV_CONTEXT:
				tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX2_NOCOMP_WRITE_CONTEXT:
				tmp_cntr = tmp_ep->write_cntr;
				break;

			case PSMX2_NOCOMP_READ_CONTEXT:
				tmp_cntr = tmp_ep->read_cntr;
				break;

			case PSMX2_SEND_CONTEXT:
			case PSMX2_TSEND_CONTEXT:
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->send_cntr;
				break;

			case PSMX2_RECV_CONTEXT:
			case PSMX2_TRECV_CONTEXT:
				tmp_cq = tmp_ep->recv_cq;
				tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX2_MULTI_RECV_CONTEXT:
				multi_recv = 1;
				tmp_cq = tmp_ep->recv_cq;
				tmp_cntr = tmp_ep->recv_cntr;
				break;

			case PSMX2_READ_CONTEXT:
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->read_cntr;
				break;

			case PSMX2_WRITE_CONTEXT:
				tmp_cq = tmp_ep->send_cq;
				tmp_cntr = tmp_ep->write_cntr;
				break;

			case PSMX2_REMOTE_WRITE_CONTEXT:
				{
				  struct fi_context *fi_context = psm2_status.context;
				  struct psmx2_fid_mr *mr;
				  struct psmx2_am_request *req;

				  req = container_of(fi_context, struct psmx2_am_request, fi_context);
				  if (req->op & PSMX2_AM_FORCE_ACK) {
					req->error = psmx2_errno(psm2_status.error_code);
					psmx2_am_ack_rma(req);
				  }

				  mr = PSMX2_CTXT_USER(fi_context);
				  if (mr->domain->rma_ep->recv_cq && (req->cq_flags & FI_REMOTE_CQ_DATA)) {
					event = psmx2_cq_create_event_from_status(
							mr->domain->rma_ep->recv_cq,
							&psm2_status, req->write.data,
							(mr->domain->rma_ep->recv_cq == cq) ?
								event_buffer : NULL,
							count, src_addr);
					if (!event)
						return -FI_ENOMEM;

					if (event == event_buffer) {
						read_count++;
						read_more = --count;
						event_buffer = count ? event_buffer + cq->entry_size : NULL;
						if (src_addr)
							src_addr = count ? src_addr + 1 : NULL;
					}
					else {
						psmx2_cq_enqueue_event(mr->domain->rma_ep->recv_cq, event);
						if (mr->domain->rma_ep->recv_cq == cq)
							read_more = 0;
					}
				  }

				  if (mr->domain->rma_ep->remote_write_cntr)
					psmx2_cntr_inc(mr->domain->rma_ep->remote_write_cntr);

				  if (mr->cntr && mr->cntr != mr->domain->rma_ep->remote_write_cntr)
					psmx2_cntr_inc(mr->cntr);

				  if (read_more)
					continue;

				  return read_count;
				}

			case PSMX2_REMOTE_READ_CONTEXT:
				{
				  struct fi_context *fi_context = psm2_status.context;
				  struct psmx2_fid_mr *mr;
				  mr = PSMX2_CTXT_USER(fi_context);
				  if (mr->domain->rma_ep->remote_read_cntr)
					psmx2_cntr_inc(mr->domain->rma_ep->remote_read_cntr);

				  continue;
				}
			}

			if (tmp_cq) {
				event = psmx2_cq_create_event_from_status(
						tmp_cq, &psm2_status, 0,
						(tmp_cq == cq) ? event_buffer : NULL, count,
						src_addr);
				if (!event)
					return -FI_ENOMEM;

				if (event == event_buffer) {
					read_count++;
					read_more = --count;
					event_buffer = count ? event_buffer + cq->entry_size : NULL;
					if (src_addr)
						src_addr = count ? src_addr + 1 : NULL;
				}
				else {
					psmx2_cq_enqueue_event(tmp_cq, event);
					if (tmp_cq == cq)
						read_more = 0;
				}
			}

			if (tmp_cntr)
				psmx2_cntr_inc(tmp_cntr);

			if (multi_recv) {
				struct psmx2_multi_recv *req;
				psm2_mq_req_t psm2_req;

				req = PSMX2_CTXT_USER(fi_context);
				req->offset += psm2_status.nbytes;
				if (req->offset + req->min_buf_size <= req->len) {
					err = psm2_mq_irecv(tmp_ep->domain->psm2_mq,
							    req->tag, req->tagsel, req->flag,
							    req->buf + req->offset, 
							    req->len - req->offset,
							    (void *)fi_context, &psm2_req);
					if (err != PSM2_OK)
						return psmx2_errno(err);

					PSMX2_CTXT_REQ(fi_context) = psm2_req;
				}
				else {
					free(req);
				}
			}

			if (read_more)
				continue;

			return read_count;
		}
		else if (err == PSM2_MQ_NO_COMPLETIONS) {
			fastlock_release(&domain->poll_lock);
			return read_count;
		}
		else {
			fastlock_release(&domain->poll_lock);
			return psmx2_errno(err);
		}
	}
}

static ssize_t psmx2_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
				 fi_addr_t *src_addr)
{
	struct psmx2_fid_cq *cq_priv;
	struct psmx2_cq_event *event;
	int ret;
	ssize_t read_count;

	cq_priv = container_of(cq, struct psmx2_fid_cq, cq);

	if (slist_empty(&cq_priv->event_queue) || !buf) {
		ret = psmx2_cq_poll_mq(cq_priv, cq_priv->domain,
				       (struct psmx2_cq_event *)buf, count, src_addr);
		if (ret > 0)
			return ret;

		if (cq_priv->domain->am_initialized)
			psmx2_am_progress(cq_priv->domain);
	}

	if (cq_priv->pending_error)
		return -FI_EAVAIL;

	if (!buf && count)
		return -FI_EINVAL;

	read_count = 0;
	while (count--) {
		event = psmx2_cq_dequeue_event(cq_priv);
		if (event) {
			if (!event->error) {
				memcpy(buf, (void *)&event->cqe, cq_priv->entry_size);
				if (src_addr)
					*src_addr = (fi_addr_t) event->source;

				psmx2_cq_free_event(cq_priv, event);

				read_count++;
				buf += cq_priv->entry_size;
				if (src_addr)
					src_addr++;
				continue;
			}
			else {
				cq_priv->pending_error = event;
				if (!read_count)
					read_count = -FI_EAVAIL;
				break;
			}
		}
		else {
			break;
		}
	}

	return read_count ? read_count : -FI_EAGAIN;
}

static ssize_t psmx2_cq_read(struct fid_cq *cq, void *buf, size_t count)
{
	return psmx2_cq_readfrom(cq, buf, count, NULL);
}

static ssize_t psmx2_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			        uint64_t flags)
{
	struct psmx2_fid_cq *cq_priv;

	cq_priv = container_of(cq, struct psmx2_fid_cq, cq);

	if (cq_priv->pending_error) {
		memcpy(buf, &cq_priv->pending_error->cqe, sizeof *buf);
		free(cq_priv->pending_error);
		cq_priv->pending_error = NULL;
		return 1;
	}

	return -FI_EAGAIN;
}

static ssize_t psmx2_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
				  fi_addr_t *src_addr, const void *cond,
				  int timeout)
{
	struct psmx2_fid_cq *cq_priv;
	struct timespec ts0, ts;
	size_t threshold, event_count;
	int msec_passed = 0;

	cq_priv = container_of(cq, struct psmx2_fid_cq, cq);
	if (cq_priv->wait_cond == FI_CQ_COND_THRESHOLD)
		threshold = (size_t) cond;
	else
		threshold = 1;

	/* NOTE: "cond" is only a hint, not a mandatory condition. */
	event_count = cq_priv->event_count;
	if (event_count < threshold) {
		if (cq_priv->wait) {
			psmx2_wait_wait((struct fid_wait *)cq_priv->wait, timeout);
		}
		else {
			clock_gettime(CLOCK_REALTIME, &ts0);
			while (1) {
				if (psmx2_cq_poll_mq(cq_priv, cq_priv->domain, NULL, 0, NULL))
					break;

				/* CQ may be updated asynchronously by the AM handlers */
				if (cq_priv->event_count > event_count)
					break;

				if (timeout < 0)
					continue;

				clock_gettime(CLOCK_REALTIME, &ts);
				msec_passed = (ts.tv_sec - ts0.tv_sec) * 1000 +
					       (ts.tv_nsec - ts0.tv_nsec) / 1000000;

				if (msec_passed >= timeout)
					break;
			}
		}
	}

	return psmx2_cq_readfrom(cq, buf, count, src_addr);
}

static ssize_t psmx2_cq_sread(struct fid_cq *cq, void *buf, size_t count,
			      const void *cond, int timeout)
{
	return psmx2_cq_sreadfrom(cq, buf, count, NULL, cond, timeout);
}

static int psmx2_cq_signal(struct fid_cq *cq)
{
	struct psmx2_fid_cq *cq_priv;
	cq_priv = container_of(cq, struct psmx2_fid_cq, cq);

	if (cq_priv->wait)
		psmx2_wait_signal((struct fid_wait *)cq_priv->wait);

	return 0;
}

static const char *psmx2_cq_strerror(struct fid_cq *cq, int prov_errno, const void *prov_data,
				     char *buf, size_t len)
{
	return psm2_error_get_string(prov_errno);
}

static int psmx2_cq_close(fid_t fid)
{
	struct psmx2_fid_cq *cq;
	struct slist_entry *entry;
	struct psmx2_cq_event *item;

	cq = container_of(fid, struct psmx2_fid_cq, cq.fid);

	psmx2_domain_release(cq->domain);

	while (!slist_empty(&cq->free_list)) {
		entry = slist_remove_head(&cq->free_list);
		item = container_of(entry, struct psmx2_cq_event, list_entry);
		free(item);
	}

	fastlock_destroy(&cq->lock);

	if (cq->wait && cq->wait_is_local)
		fi_close((fid_t)cq->wait);

	free(cq);

	return 0;
}

static int psmx2_cq_control(struct fid *fid, int command, void *arg)
{
	struct psmx2_fid_cq *cq;
	int ret = 0;

	cq = container_of(fid, struct psmx2_fid_cq, cq.fid);

	switch (command) {
	case FI_GETWAIT:
		ret = psmx2_wait_get_obj(cq->wait, arg);
		break;

	default:
		return -FI_ENOSYS;
	}

	return ret;
}

static struct fi_ops psmx2_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx2_cq_close,
	.bind = fi_no_bind,
	.control = psmx2_cq_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq psmx2_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = psmx2_cq_read,
	.readfrom = psmx2_cq_readfrom,
	.readerr = psmx2_cq_readerr,
	.sread = psmx2_cq_sread,
	.sreadfrom = psmx2_cq_sreadfrom,
	.signal = psmx2_cq_signal,
	.strerror = psmx2_cq_strerror,
};

int psmx2_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct psmx2_fid_domain *domain_priv;
	struct psmx2_fid_cq *cq_priv;
	struct psmx2_fid_wait *wait = NULL;
	struct psmx2_cq_event *event;
	struct fi_wait_attr wait_attr;
	int wait_is_local = 0;
	int entry_size;
	int err;
	int i;

	domain_priv = container_of(domain, struct psmx2_fid_domain, domain);
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
		FI_INFO(&psmx2_prov, FI_LOG_CQ,
			"attr->format=%d, supported=%d...%d\n", attr->format,
			FI_CQ_FORMAT_UNSPEC, FI_CQ_FORMAT_TAGGED);
		return -FI_EINVAL;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
		break;

	case FI_WAIT_SET:
		if (!attr->wait_set) {
			FI_INFO(&psmx2_prov, FI_LOG_CQ,
				"FI_WAIT_SET is specified but attr->wait_set is NULL\n");
			return -FI_EINVAL;
		}
		wait = (struct psmx2_fid_wait *)attr->wait_set;
		break;

	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		wait_attr.wait_obj = attr->wait_obj;
		wait_attr.flags = 0;
		err = psmx2_wait_open(&domain_priv->fabric->fabric,
				      &wait_attr, (struct fid_wait **)&wait);
		if (err)
			return err;
		wait_is_local = 1;
		break;

	default:
		FI_INFO(&psmx2_prov, FI_LOG_CQ,
			"attr->wait_obj=%d, supported=%d...%d\n", attr->wait_obj,
			FI_WAIT_NONE, FI_WAIT_MUTEX_COND);
		return -FI_EINVAL;
	}

	if (wait) {
		switch (attr->wait_cond) {
		case FI_CQ_COND_NONE:
		case FI_CQ_COND_THRESHOLD:
			break;

		default:
			FI_INFO(&psmx2_prov, FI_LOG_CQ,
				"attr->wait_cond=%d, supported=%d...%d\n",
				attr->wait_cond, FI_CQ_COND_NONE, FI_CQ_COND_THRESHOLD);
			return -FI_EINVAL;
		}
	}

	cq_priv = (struct psmx2_fid_cq *) calloc(1, sizeof *cq_priv);
	if (!cq_priv) {
		if (wait)
			free(wait);
		return -FI_ENOMEM;
	}

	psmx2_domain_acquire(domain_priv);

	cq_priv->domain = domain_priv;
	cq_priv->format = attr->format;
	cq_priv->entry_size = entry_size;
	cq_priv->wait = wait;
	if (wait)
		cq_priv->wait_cond = attr->wait_cond;
	cq_priv->wait_is_local = wait_is_local;

	cq_priv->cq.fid.fclass = FI_CLASS_CQ;
	cq_priv->cq.fid.context = context;
	cq_priv->cq.fid.ops = &psmx2_fi_ops;
	cq_priv->cq.ops = &psmx2_cq_ops;

	slist_init(&cq_priv->event_queue);
	slist_init(&cq_priv->free_list);
	fastlock_init(&cq_priv->lock);

#define PSMX2_FREE_LIST_SIZE	64
	for (i=0; i<PSMX2_FREE_LIST_SIZE; i++) {
		event = calloc(1, sizeof(*event));
		if (!event) {
			FI_WARN(&psmx2_prov, FI_LOG_CQ, "out of memory.\n");
			exit(-1);
		}
		slist_insert_tail(&event->list_entry, &cq_priv->free_list);
	}

	*cq = &cq_priv->cq;
	return 0;
}

