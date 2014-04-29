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

void psmx_ec_enqueue_event(struct psmx_fid_ec *ec,
				struct psmx_event *event)
{
	if (ec->event_queue.tail)
		ec->event_queue.tail->next = event;
	else
		ec->event_queue.head = ec->event_queue.tail = event;
}

static struct psmx_event *psmx_ec_dequeue_event(struct psmx_fid_ec *ec)
{
	struct psmx_event *event;

	if (!ec->event_queue.head)
		return NULL;

	event = ec->event_queue.head;
	ec->event_queue.head = event->next;
	if (!ec->event_queue.head)
		ec->event_queue.tail = NULL;

	event->next = NULL;
	return event;
}

struct psmx_event *psmx_ec_create_event(struct psmx_fid_ec *ec,
					void *op_context, void *buf,
					uint64_t flags, size_t len,
					uint64_t data, uint64_t tag,
					size_t olen, int err)
{
	struct psmx_event *event;

	event = calloc(1, sizeof(*event));
	if (!event) {
		fprintf(stderr, "%s: out of memory\n", __func__);
		return NULL;
	}

	if (err) {
		event->format = ec->err_format;
		ec->num_errors++;
	}
	else {
		event->format = ec->format;
		ec->num_events++;
	}

	switch (event->format) {
	case FI_EC_FORMAT_CONTEXT:
		event->ece.context.op_context = op_context;
		break;

	case FI_EC_FORMAT_COMP:
		event->ece.comp.op_context = op_context;
		event->ece.comp.flags = flags;
		event->ece.comp.len = len;
		break;

	case FI_EC_FORMAT_DATA:
		event->ece.data.op_context = op_context;
		event->ece.data.buf = buf;
		event->ece.data.flags = flags;
		event->ece.data.len = len;
		event->ece.data.data = data;
		break;

	case FI_EC_FORMAT_TAGGED:
		event->ece.tagged.op_context = op_context;
		event->ece.tagged.buf = buf;
		event->ece.tagged.flags = flags;
		event->ece.tagged.len = len;
		event->ece.tagged.data = data;
		event->ece.tagged.tag = tag;
		event->ece.tagged.olen = olen;
		break;

	case FI_EC_FORMAT_ERR:
		event->ece.err.op_context = op_context;
		event->ece.err.err = err;
		event->ece.err.prov_errno = 0;
		event->ece.err.prov_data = NULL;
		break;

	case FI_EC_FORMAT_COMP_ERR:
		event->ece.err.op_context = op_context;
		event->ece.err.flags = flags;
		event->ece.err.len = len;
		event->ece.err.err = err;
		event->ece.err.prov_errno = 0;
		event->ece.err.prov_data = NULL;
		break;

	case FI_EC_FORMAT_DATA_ERR:
		event->ece.err.op_context = op_context;
		event->ece.err.buf = buf;
		event->ece.err.flags = flags;
		event->ece.err.len = len;
		event->ece.err.data = data;
		event->ece.err.err = err;
		event->ece.err.prov_errno = 0;
		event->ece.err.prov_data = NULL;
		break;

	case FI_EC_FORMAT_TAGGED_ERR:
		if (err) {
			event->ece.tagged_err.status = err;
			event->ece.tagged_err.err.op_context = op_context;
			event->ece.tagged_err.err.fid_context = ec->ec.fid.context;
			event->ece.tagged_err.err.flags = flags;
			event->ece.tagged_err.err.len = len;
			event->ece.tagged_err.err.data = data;
			event->ece.tagged_err.err.err = err;
			event->ece.tagged_err.err.prov_errno = 0;
			event->ece.tagged_err.err.prov_data = NULL;
		}
		else {
			event->ece.tagged_err.status = 0;
			event->ece.tagged_err.tagged.op_context = op_context;
			event->ece.tagged_err.tagged.buf = buf;
			event->ece.tagged_err.tagged.flags = flags;
			event->ece.tagged_err.tagged.len = len;
			event->ece.tagged_err.tagged.data = data;
			event->ece.tagged_err.tagged.tag = tag;
			event->ece.tagged_err.tagged.olen = olen;
		}
		break;

	case FI_EC_FORMAT_COUNTER:
		event->ece.counter.events = ec->num_events;
		break;

	case FI_EC_FORMAT_COUNTER_ERR:
		event->ece.counter_err.events = ec->num_events;
		event->ece.counter_err.errors = ec->num_errors;
		break;

	case FI_EC_FORMAT_CM:
	default:
		fprintf(stderr, "%s: unsupported EC format %d\n", __func__, event->format);
		return NULL;
	}

	return event;
}

static struct psmx_event *psmx_ec_create_event_from_status(
				struct psmx_fid_ec *ec,
				psm_mq_status_t *psm_status)
{
	struct psmx_event *event;
	struct fi_context *fi_context = psm_status->context;
	int err;

	event = calloc(1, sizeof(*event));
	if (!event) {
		fprintf(stderr, "%s: out of memory\n", __func__);
		return NULL;
	}

	if (psm_status->error_code) {
		event->format = ec->err_format;
		ec->num_errors++;
	}
	else {
		event->format = ec->format;
		ec->num_events++;
	}

	switch (event->format) {
	case FI_EC_FORMAT_CONTEXT:
		event->ece.context.op_context = PSMX_CTXT_USER(fi_context);
		break;

	case FI_EC_FORMAT_COMP:
		event->ece.comp.op_context = PSMX_CTXT_USER(fi_context);
		//event->ece.comp.flags = 0; /* FIXME */
		event->ece.comp.len = psm_status->nbytes;
		break;

	case FI_EC_FORMAT_DATA:
		event->ece.data.op_context = PSMX_CTXT_USER(fi_context);
		//event->ece.data.buf = NULL; /* FIXME */
		//event->ece.data.flags = 0; /* FIXME */
		event->ece.data.len = psm_status->nbytes;
		//event->ece.data.data = 0; /* FIXME */
		break;

	case FI_EC_FORMAT_TAGGED:
		event->ece.tagged.op_context = PSMX_CTXT_USER(fi_context);
		//event->ece.tagged.buf = NULL; /* FIXME */
		//event->ece.tagged.flags = 0; /* FIXME */
		event->ece.tagged.len = psm_status->nbytes;
		//event->ece.tagged.data = 0; /* FIXME */
		event->ece.tagged.tag = psm_status->msg_tag;
		event->ece.tagged.olen = psm_status->msg_length - psm_status->nbytes;
		break;

	case FI_EC_FORMAT_ERR:
		event->ece.err.op_context = PSMX_CTXT_USER(fi_context);
		event->ece.err.err = psmx_errno(psm_status->error_code);
		event->ece.err.prov_errno = psm_status->error_code;
		//event->ece.err.prov_data = NULL; /* FIXME */
		break;

	case FI_EC_FORMAT_COMP_ERR:
		event->ece.err.op_context = PSMX_CTXT_USER(fi_context);
		//event->ece.err.flags = 0; /* FIXME */
		event->ece.err.len = psm_status->nbytes;
		event->ece.err.err = psmx_errno(psm_status->error_code);
		event->ece.err.prov_errno = psm_status->error_code;
		//event->ece.err.prov_data = NULL; /* FIXME */
		break;

	case FI_EC_FORMAT_DATA_ERR:
		event->ece.err.op_context = PSMX_CTXT_USER(fi_context);
		//event->ece.err.buf = NULL; /* FIXME */
		//event->ece.err.flags = 0; /* FIXME */
		event->ece.err.len = psm_status->nbytes;
		//event->ece.err.data = 0; /* FIXME */
		event->ece.err.err = psmx_errno(psm_status->error_code);
		event->ece.err.prov_errno = psm_status->error_code;
		//event->ece.err.prov_data = NULL; /* FIXME */
		break;

	case FI_EC_FORMAT_TAGGED_ERR:
		err = psmx_errno(psm_status->error_code);
		if (err) {
			event->ece.tagged_err.status = err;
			event->ece.tagged_err.err.op_context = PSMX_CTXT_USER(fi_context);
			event->ece.tagged_err.err.fid_context = ec->ec.fid.context;
			//event->ece.tagged_err.err.flags = 0; /* FIXME */
			event->ece.tagged_err.err.len = psm_status->nbytes;
			//event->ece.tagged_err.err.data = 0; /* FIXME */
			event->ece.tagged_err.err.err = err;
			event->ece.tagged_err.err.prov_errno = psm_status->error_code;
			//event->ece.tagged_err.err.prov_data = NULL; /* FIXME */
		}
		else {
			event->ece.tagged_err.status = 0;
			event->ece.tagged_err.tagged.op_context = PSMX_CTXT_USER(fi_context);
			//event->ece.tagged_err.tagged.buf = NULL; /* FIXME */
			//event->ece.tagged_err.tagged.flags = 0; /* FIXME */
			event->ece.tagged_err.tagged.len = psm_status->nbytes;
			//event->ece.tagged_err.tagged.data = 0; /* FIXME */
			event->ece.tagged_err.tagged.tag = psm_status->msg_tag;
			event->ece.tagged_err.tagged.olen = psm_status->msg_length -
								psm_status->nbytes;
		}
		break;

	case FI_EC_FORMAT_COUNTER:
		event->ece.counter.events = ec->num_events;
		break;

	case FI_EC_FORMAT_COUNTER_ERR:
		event->ece.counter_err.events = ec->num_events;
		event->ece.counter_err.errors = ec->num_errors;
		break;

	case FI_EC_FORMAT_CM:
	default:
		fprintf(stderr, "%s: unsupported EC format %d %d %d\n", __func__, event->format, ec->format, ec->err_format);
		return NULL;
	}

	event->source = psm_status->msg_tag;

	return event;
}

static int psmx_ec_get_event_src_addr(struct psmx_fid_ec *fid_ec,
					struct psmx_event *event,
					void *src_addr, size_t *addrlen)
{
	int err;

	if (!src_addr)
		return 0;

	if ((fid_ec->domain->reserved_tag_bits & PSMX_MSG_BIT) &&
		(event->source & PSMX_MSG_BIT)) {
		err = psmx_epid_to_epaddr(
			fid_ec->domain->psm_ep,
			event->source & ~PSMX_MSG_BIT,
			src_addr);
		*addrlen = sizeof(psm_epaddr_t);
	}

	return 0;
}

static int psmx_ec_poll_mq(struct psmx_fid_ec *ec)
{
	psm_mq_req_t psm_req;
	psm_mq_status_t psm_status;
	struct fi_context *fi_context;
	struct psmx_fid_ec *tmp_ec;
	struct psmx_event *event;
	int err;

	while (1) {
		err = psm_mq_ipeek(ec->domain->psm_mq, &psm_req, NULL);

		if (err == PSM_OK) {
			err = psm_mq_test(&psm_req, &psm_status);

			fi_context = psm_status.context;

			if (!fi_context) /* only possible with FI_SYNC set */
				continue;

			if (PSMX_CTXT_TYPE(fi_context) == PSMX_NOCOMP_CONTEXT)
				continue;

			tmp_ec = PSMX_CTXT_EC(fi_context);
			event = psmx_ec_create_event_from_status(tmp_ec, &psm_status);
			if (!event)
				return -ENOMEM;

			psmx_ec_enqueue_event(tmp_ec, event);

			if (tmp_ec == ec)
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

static ssize_t psmx_ec_readfrom(fid_t fid, void *buf, size_t len,
				void *src_addr, size_t *addrlen)
{
	struct psmx_fid_ec *fid_ec;
	struct psmx_event *event;

	fid_ec = container_of(fid, struct psmx_fid_ec, ec.fid);
	assert(fid_ec->domain);

	if (len < fid_ec->entry_size)
		return -FI_ETOOSMALL;

	psmx_ec_poll_mq(fid_ec);

	if (fid_ec->pending_error)
		return -FI_EAVAIL;

	event = psmx_ec_dequeue_event(fid_ec);
	if (event) {
		if (event->format == fid_ec->format) {
			memcpy(buf, (void *)&event->ece, fid_ec->entry_size);
			psmx_ec_get_event_src_addr(fid_ec, event, src_addr, addrlen);
			free(event);
			return fid_ec->entry_size;
		}
		else {
			fid_ec->pending_error = event;
			return -FI_EAVAIL;
		}
	}

	return 0;
}

static ssize_t psmx_ec_read(fid_t fid, void *buf, size_t len)
{
	return psmx_ec_readfrom(fid, buf, len, NULL, NULL);
}

static ssize_t psmx_ec_readerr(fid_t fid, void *buf, size_t len, uint64_t flags)
{
	struct psmx_fid_ec *fid_ec;

	fid_ec = container_of(fid, struct psmx_fid_ec, ec.fid);

	if (len < fid_ec->err_entry_size)
		return -FI_ETOOSMALL;

	if (fid_ec->pending_error) {
		memcpy(buf, &fid_ec->pending_error->ece, fid_ec->err_entry_size);
		free(fid_ec->pending_error);
		fid_ec->pending_error = NULL;
		return fid_ec->err_entry_size;
	}

	return 0;
}

static ssize_t psmx_ec_write(fid_t fid, const void *buf, size_t len)
{
	return -ENOSYS;
}

static int psmx_ec_reset(fid_t fid, const void *cond)
{
	return -ENOSYS;
}

static ssize_t psmx_ec_condread(fid_t fid, void *buf, size_t len, const void *cond)
{
	return -ENOSYS;
}

static ssize_t psmx_ec_condreadfrom(fid_t fid, void *buf, size_t len,
				    void *src_addr, size_t *addrlen, const void *cond)
{
	return -ENOSYS;
}

static const char *psmx_ec_strerror(fid_t fid, int prov_errno, const void *prov_data,
				    void *buf, size_t len)
{
	return psm_error_get_string(prov_errno);
}

static int psmx_ec_close(fid_t fid)
{
	struct psmx_fid_ec *fid_ec;

	fid_ec = container_of(fid, struct psmx_fid_ec, ec.fid);
	free(fid_ec);

	return 0;
}

static int psmx_ec_bind(fid_t fid, struct fi_resource *fids, int nfids)
{
	struct fi_resource ress;
	int err;
	int i;

	for (i=0; i<nfids; i++) {
		if (!fids[i].fid)
			return -EINVAL;
		switch (fids[i].fid->fclass) {
		case FID_CLASS_EP:
		case FID_CLASS_MR:
			if (!fids[i].fid->ops || !fids[i].fid->ops->bind)
				return -EINVAL;
			ress.fid = fid;
			ress.flags = fids[i].flags;
			err = fids[i].fid->ops->bind(fids[i].fid, &ress, 1);
			if (err)
				return err;
			break;

		default:
			return -ENOSYS;
		}
	}
	return 0;
}

static int psmx_ec_sync(fid_t fid, uint64_t flags, void *context)
{
	return -ENOSYS;
}

static int psmx_ec_control(fid_t fid, int command, void *arg)
{
	return -ENOSYS;
}

static struct fi_ops psmx_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = psmx_ec_close,
	.bind = psmx_ec_bind,
	.sync = psmx_ec_sync,
	.control = psmx_ec_control,
};

static struct fi_ops_ec psmx_ec_ops = {
	.size = sizeof(struct fi_ops_ec),
	.read = psmx_ec_read,
	.readfrom = psmx_ec_readfrom,
	.readerr = psmx_ec_readerr,
	.write = psmx_ec_write,
	.reset = psmx_ec_reset,
	.condread = psmx_ec_condread,
	.condreadfrom = psmx_ec_condreadfrom,
	.strerror = psmx_ec_strerror,
};

int psmx_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context)
{
	struct psmx_fid_domain *fid_domain;
	struct psmx_fid_ec *fid_ec;
	int format, err_format;
	int entry_size, err_entry_size;

	switch (attr->domain) {
	case FI_EC_DOMAIN_GENERAL:
	case FI_EC_DOMAIN_COMP:
		break;

	default:
		psmx_debug("%s: attr->domain=%d, supported=%d,%d\n", __func__, attr->domain,
				FI_EC_DOMAIN_GENERAL, FI_EC_DOMAIN_COMP);
		return -ENOSYS;
	}

	switch (attr->type) {
	case FI_EC_QUEUE:
	case FI_EC_COUNTER:
		break;

	default:
		psmx_debug("%s: attr->type=%d, supported=%d,%d\n", __func__, attr->type,
				FI_EC_QUEUE, FI_EC_COUNTER);
		return -EINVAL;
	}

	switch (attr->format) {
	case FI_EC_FORMAT_UNSPEC:
		format = FI_EC_FORMAT_TAGGED;
		err_format = FI_EC_FORMAT_ERR;
		entry_size = sizeof(struct fi_ec_tagged_entry);
		err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_CONTEXT:
		format = attr->format;
		err_format = FI_EC_FORMAT_ERR;
		entry_size = sizeof(struct fi_ec_entry);
		err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_COMP:
		format = attr->format;
		err_format = FI_EC_FORMAT_ERR;
		entry_size = sizeof(struct fi_ec_comp_entry);
		err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_DATA:
		format = attr->format;
		err_format = FI_EC_FORMAT_ERR;
		entry_size = sizeof(struct fi_ec_data_entry);
		err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_TAGGED:
		format = attr->format;
		err_format = FI_EC_FORMAT_ERR;
		entry_size = sizeof(struct fi_ec_tagged_entry);
		err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_ERR:
	case FI_EC_FORMAT_COMP_ERR:
	case FI_EC_FORMAT_DATA_ERR:
		format = err_format = attr->format;
		entry_size = err_entry_size = sizeof(struct fi_ec_err_entry);
		break;

	case FI_EC_FORMAT_TAGGED_ERR:
		format = err_format = attr->format;
		entry_size = err_entry_size = sizeof(struct fi_ec_tagged_err_entry);
		break;

	case FI_EC_FORMAT_CM:
		format = err_format = attr->format;
		entry_size = err_entry_size = sizeof(struct fi_ec_cm_entry);
		break;

	case FI_EC_FORMAT_COUNTER:
		format = attr->format;
		err_format = FI_EC_FORMAT_COUNTER_ERR;
		entry_size = sizeof(struct fi_ec_counter_entry);
		err_entry_size = sizeof(struct fi_ec_counter_err_entry);
		break;

	case FI_EC_FORMAT_COUNTER_ERR:
		format = err_format = attr->format;
		entry_size = err_entry_size = sizeof(struct fi_ec_counter_err_entry);
		break;

	default:
		psmx_debug("%s: attr->format=%d, supported=%d...%d\n", __func__, attr->format,
				FI_EC_FORMAT_UNSPEC, FI_EC_FORMAT_COUNTER_ERR);
		return -EINVAL;
	}

	fid_domain = container_of(fid, struct psmx_fid_domain, domain.fid);
	fid_ec = (struct psmx_fid_ec *) calloc(1, sizeof *fid_ec);
	if (!fid_ec)
		return -ENOMEM;

	fid_ec->domain = fid_domain;
	fid_ec->type = attr->type;
	fid_ec->format = format;
	fid_ec->err_format = err_format;
	fid_ec->entry_size = entry_size;
	fid_ec->err_entry_size = err_entry_size;
	fid_ec->ec.fid.size = sizeof(struct fid_ec);
	fid_ec->ec.fid.fclass = FID_CLASS_EC;
	fid_ec->ec.fid.context = context;
	fid_ec->ec.fid.ops = &psmx_fi_ops;
	fid_ec->ec.ops = &psmx_ec_ops;

	*ec = &fid_ec->ec.fid;
	return 0;
}

