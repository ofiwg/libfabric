
/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_CQ, __VA_ARGS__)

/*
 * cxip_cq_req_cancel() - Cancel one request.
 *
 * Cancel one Receive request. If match is true, cancel the request with
 * matching op_ctx. Only Receive requests should be in the request list.
 */
int cxip_cq_req_cancel(struct cxip_cq *cq, void *req_ctx, void *op_ctx,
		       bool match)
{
	int ret = -FI_ENOENT;
	struct cxip_req *req;

	/* Serialize with event processing that could update request state. */
	fastlock_acquire(&cq->lock);

	dlist_foreach_container(&cq->req_list, struct cxip_req, req,
				cq_entry) {
		if (req->req_ctx == req_ctx &&
		    !req->recv.canceled &&
		    (!match || (void *)req->context == op_ctx)) {
			ret = cxip_msg_recv_cancel(req);
			break;
		}
	}

	fastlock_release(&cq->lock);

	return ret;
}

/*
 * cxip_cq_req_discard() - Discard all matching requests.
 *
 * Mark all requests on the Completion Queue to be discarded. When a marked
 * request completes, it's completion event will be dropped. This is the
 * behavior defined for requests belonging to a closed Endpoint.
 */
void cxip_cq_req_discard(struct cxip_cq *cq, void *req_ctx)
{
	struct cxip_req *req;
	int discards = 0;

	/* Serialize with event processing that could update request state. */
	fastlock_acquire(&cq->lock);

	dlist_foreach_container(&cq->req_list, struct cxip_req, req,
				cq_entry) {
		if (req->req_ctx == req_ctx) {
			req->discard = true;
			discards++;
		}
	}

	if (discards)
		CXIP_LOG_DBG("Marked %d requests\n", discards);

	fastlock_release(&cq->lock);
}

/*
 * cxip_cq_req_complete() - Generate a completion event for the request.
 */
int cxip_cq_req_complete(struct cxip_req *req)
{
	if (req->discard) {
		CXIP_LOG_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	return ofi_cq_write(&req->cq->util_cq, (void *)req->context,
			    req->flags, req->data_len, (void *)req->buf,
			    req->data, req->tag);
}

/*
 * cxip_cq_req_complete() - Generate a completion event with source address for
 * the request.
 */
int cxip_cq_req_complete_addr(struct cxip_req *req, fi_addr_t src)
{
	if (req->discard) {
		CXIP_LOG_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	return ofi_cq_write_src(&req->cq->util_cq, (void *)req->context,
				req->flags, req->data_len, (void *)req->buf,
				req->data, req->tag, src);
}

/*
 * cxip_cq_req_complete() - Generate an error event for the request.
 */
int cxip_cq_req_error(struct cxip_req *req, size_t olen,
		      int err, int prov_errno, void *err_data,
		      size_t err_data_size)
{
	struct fi_cq_err_entry err_entry;

	if (req->discard) {
		CXIP_LOG_DBG("Event discarded: %p\n", req);
		return FI_SUCCESS;
	}

	err_entry.err = err;
	err_entry.olen = olen;
	err_entry.err_data = err_data;
	err_entry.err_data_size = err_data_size;
	err_entry.len = req->data_len;
	err_entry.prov_errno = prov_errno;
	err_entry.flags = req->flags;
	err_entry.data = req->data;
	err_entry.tag = req->tag;
	err_entry.op_context = (void *)(uintptr_t)req->context;
	err_entry.buf = (void *)(uintptr_t)req->buf;

	return ofi_cq_write_error(&req->cq->util_cq, &err_entry);
}

/*
 * cxip_cq_req_find() - Look up a request by ID (from an event).
 */
static struct cxip_req *cxip_cq_req_find(struct cxip_cq *cq, int id)
{
	return ofi_idx_at(&cq->req_table, id);
}

/*
 * cxip_cq_req_alloc() - Allocate a request.
 *
 * If remap is set, allocate a 16-bit request ID and map it to the new
 * request.
 */
struct cxip_req *cxip_cq_req_alloc(struct cxip_cq *cq, int remap,
				   void *req_ctx)
{
	struct cxip_req *req;

	fastlock_acquire(&cq->req_lock);

	req = (struct cxip_req *)ofi_buf_alloc(cq->req_pool);
	if (!req) {
		CXIP_LOG_ERROR("Failed to allocate request\n");
		goto out;
	}

	if (remap) {
		req->req_id = ofi_idx_insert(&cq->req_table, req);

		/* Target command buffer IDs are 16 bits wide. */
		if (req->req_id < 0 || req->req_id >= (1 << 16)) {
			CXIP_LOG_ERROR("Failed to map request: %d\n",
				       req->req_id);
			ofi_buf_free(req);
			req = NULL;
			goto out;
		}
	} else {
		req->req_id = -1;
	}

	CXIP_LOG_DBG("Allocated req: %p (ID: %d)\n", req, req->req_id);
	req->cq = cq;
	req->req_ctx = req_ctx;
	req->discard = false;
	dlist_insert_tail(&req->cq_entry, &cq->req_list);

out:
	fastlock_release(&cq->req_lock);

	return req;
}

/*
 * cxip_cq_req_free() - Free a request.
 */
void cxip_cq_req_free(struct cxip_req *req)
{
	struct cxip_req *table_req;
	struct cxip_cq *cq = req->cq;

	fastlock_acquire(&cq->req_lock);

	dlist_remove(&req->cq_entry);

	if (req->req_id >= 0) {
		table_req = (struct cxip_req *)ofi_idx_remove(
			&req->cq->req_table, req->req_id);
		if (table_req != req)
			CXIP_LOG_ERROR("Failed to free request\n");
	}

	ofi_buf_free(req);

	fastlock_release(&cq->req_lock);
}

/*
 * cxip_cq_event_req() - Locate a request corresponding to the Cassini event.
 */
static struct cxip_req *cxip_cq_event_req(struct cxip_cq *cq,
					  const union c_event *event)
{
	struct cxip_req *req;
	uint32_t pte_num;
	enum c_ptlte_state pte_state;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		req = (struct cxip_req *)event->init_short.user_ptr;
		break;
	case C_EVENT_LINK:
	case C_EVENT_UNLINK:
	case C_EVENT_GET:
	case C_EVENT_PUT:
	case C_EVENT_PUT_OVERFLOW:
	case C_EVENT_RENDEZVOUS:
		req = cxip_cq_req_find(cq, event->tgt_long.buffer_id);
		if (!req)
			CXIP_LOG_ERROR("Invalid buffer_id: %d (%s)\n",
				       event->tgt_long.buffer_id,
				       cxi_event_to_str(event));
		break;
	case C_EVENT_REPLY:
	case C_EVENT_SEND:
		if (!event->init_short.rendezvous) {
			req = (struct cxip_req *)event->init_short.user_ptr;
		} else {
			struct cxi_rdzv_user_ptr *up =
					(struct cxi_rdzv_user_ptr *)
					 &event->init_short.user_ptr;
			req = cxip_cq_req_find(cq, up->buffer_id);
			if (!req)
				CXIP_LOG_ERROR("Invalid buffer_id: %d (%s)\n",
					       event->tgt_long.buffer_id,
					       cxi_event_to_str(event));
		}
		break;
	case C_EVENT_STATE_CHANGE:
		pte_num = event->tgt_long.ptlte_index;
		pte_state = event->tgt_long.initiator.state_change.ptlte_state;

		cxip_pte_state_change(cq->domain->dev_if, pte_num, pte_state);

		req = NULL;
		break;
	default:
		CXIP_LOG_ERROR("Invalid event type: %d\n",
				event->hdr.event_type);
		req = NULL;
	}

	CXIP_LOG_DBG("got event: %s rc: %s (req: %p)\n",
		     cxi_event_to_str(event),
		     cxi_rc_to_str(cxi_event_rc(event)),
		     req);

	return req;
}

/*
 * cxip_cq_progress() - Progress the CXI Completion Queue.
 *
 * Process events on the underlying Cassini event queue.
 */
void cxip_cq_progress(struct cxip_cq *cq)
{
	const union c_event *event;
	struct cxip_req *req;
	int events = 0;
	int ret;

	fastlock_acquire(&cq->lock);

	if (!cq->enabled)
		goto out;

	/* TODO Limit the maximum number of events processed */
	while ((event = cxi_eq_peek_event(cq->evtq))) {
		req = cxip_cq_event_req(cq, event);
		if (req) {
			ret = req->cb(req, event);
			if (ret != FI_SUCCESS)
				break;
		}

		/* Consume event. */
		cxi_eq_next_event(cq->evtq);

		events++;
	}

	if (events)
		cxi_eq_ack_events(cq->evtq);

	if (cxi_eq_get_drops(cq->evtq)) {
		CXIP_LOG_ERROR("EQ drops detected\n");
		abort();
	}

out:
	fastlock_release(&cq->lock);
}

/*
 * cxip_util_cq_enable() - Progress function wrapper for utility CQ.
 */
void cxip_util_cq_progress(struct util_cq *util_cq)
{
	struct cxip_cq *cq = container_of(util_cq, struct cxip_cq, util_cq);

	cxip_cq_progress(cq);
}

/*
 * cxip_cq_enable() - Assign hardware resources to the CQ.
 */
int cxip_cq_enable(struct cxip_cq *cxi_cq)
{
	struct cxi_eq_attr eq_attr = {};
	struct ofi_bufpool_attr bp_attrs = {};
	int ret = FI_SUCCESS;

	fastlock_acquire(&cxi_cq->lock);

	if (cxi_cq->enabled)
		goto unlock;

	/* TODO set EVTQ size with CQ attrs */
	cxi_cq->evtq_buf_len = C_PAGE_SIZE * 64;
	cxi_cq->evtq_buf = aligned_alloc(C_PAGE_SIZE,
					 cxi_cq->evtq_buf_len);
	if (!cxi_cq->evtq_buf) {
		CXIP_LOG_DBG("Unable to allocate MR EVTQ buffer\n");
		goto unlock;
	}

	ret = cxil_map(cxi_cq->domain->dev_if->if_lni,
		       cxi_cq->evtq_buf, cxi_cq->evtq_buf_len,
		       CXI_MAP_NTA | CXI_MAP_PIN | CXI_MAP_WRITE,
		       NULL, &cxi_cq->evtq_buf_md);
	if (ret) {
		CXIP_LOG_DBG("Unable to MAP MR EVTQ buffer, ret: %d\n",
			     ret);
		goto free_evtq_buf;
	}

	eq_attr.queue = cxi_cq->evtq_buf,
	eq_attr.queue_len = cxi_cq->evtq_buf_len,
	eq_attr.queue_md = cxi_cq->evtq_buf_md,
	eq_attr.flags = CXI_EQ_TGT_LONG;

	ret = cxil_alloc_evtq(cxi_cq->domain->dev_if->if_lni, &eq_attr,
			      NULL, NULL, &cxi_cq->evtq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate EVTQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unmap_evtq_buf;
	}

	bp_attrs.size = sizeof(struct cxip_req);
	bp_attrs.alignment = 8;
	bp_attrs.max_cnt = UINT16_MAX;
	bp_attrs.chunk_cnt = 64;
	bp_attrs.flags = OFI_BUFPOOL_NO_TRACK;
	ret = ofi_bufpool_create_attr(&bp_attrs, &cxi_cq->req_pool);
	if (ret) {
		ret = -FI_ENOMEM;
		goto free_evtq;
	}

	memset(&cxi_cq->req_table, 0, sizeof(cxi_cq->req_table));

	cxi_cq->enabled = true;
	fastlock_release(&cxi_cq->lock);
	dlist_init(&cxi_cq->req_list);

	CXIP_LOG_DBG("CQ enabled: %p (EQ: %d)\n", cxi_cq, cxi_cq->evtq->eqn);
	return FI_SUCCESS;

free_evtq:
	cxil_destroy_evtq(cxi_cq->evtq);
unmap_evtq_buf:
	ret = cxil_unmap(cxi_cq->evtq_buf_md);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap evtq MD, ret: %d\n", ret);
free_evtq_buf:
	free(cxi_cq->evtq_buf);
unlock:
	fastlock_release(&cxi_cq->lock);

	return ret;
}

/*
 * cxip_cq_disable() - Release hardware resources from the CQ.
 */
static void cxip_cq_disable(struct cxip_cq *cxi_cq)
{
	int ret;

	fastlock_acquire(&cxi_cq->lock);

	if (!cxi_cq->enabled)
		goto unlock;

	ofi_idx_reset(&cxi_cq->req_table);

	ofi_bufpool_destroy(cxi_cq->req_pool);

	ret = cxil_destroy_evtq(cxi_cq->evtq);
	if (ret)
		CXIP_LOG_ERROR("Failed to free evtq, ret: %d\n", ret);

	ret = cxil_unmap(cxi_cq->evtq_buf_md);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap evtq MD, ret: %d\n", ret);

	free(cxi_cq->evtq_buf);

	cxi_cq->enabled = false;

	CXIP_LOG_DBG("CQ disabled: %p\n", cxi_cq);
unlock:
	fastlock_release(&cxi_cq->lock);
}

/*
 * cxip_cq_close() - Destroy the Completion Queue object.
 */
static int cxip_cq_close(struct fid *fid)
{
	struct cxip_cq *cq;

	cq = container_of(fid, struct cxip_cq, util_cq.cq_fid.fid);
	if (ofi_atomic_get32(&cq->ref))
		return -FI_EBUSY;

	cxip_cq_disable(cq);

	ofi_cq_cleanup(&cq->util_cq);

	fastlock_destroy(&cq->lock);
	fastlock_destroy(&cq->req_lock);

	ofi_atomic_dec32(&cq->domain->ref);

	free(cq);

	return 0;
}

static struct fi_ops cxip_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_cq_attr cxip_cq_def_attr = {
	.size = CXIP_CQ_DEF_SZ,
	.flags = 0,
	.format = FI_CQ_FORMAT_CONTEXT,
	.wait_obj = FI_WAIT_NONE,
	.signaling_vector = 0,
	.wait_cond = FI_CQ_COND_NONE,
	.wait_set = NULL,
};

/*
 * cxip_cq_verify_attr() - Verify input Completion Queue attributes.
 */
static int cxip_cq_verify_attr(struct fi_cq_attr *attr)
{
	if (!attr)
		return 0;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_TAGGED:
		break;
	case FI_CQ_FORMAT_UNSPEC:
		attr->format = cxip_cq_def_attr.format;
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		break;
	case FI_WAIT_UNSPEC:
		attr->wait_obj = cxip_cq_def_attr.wait_obj;
		break;
	case FI_WAIT_MUTEX_COND:
	case FI_WAIT_SET:
	case FI_WAIT_FD:
	default:
		return -FI_ENOSYS;
	}

	if (!attr->size)
		attr->size = cxip_cq_def_attr.size;

	return 0;
}

/*
 * cxip_cq_open() - Allocate a new Completion Queue object.
 */
int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context)
{
	struct cxip_domain *cxi_dom;
	struct cxip_cq *cxi_cq;
	int ret;

	if (!domain || !cq)
		return -FI_EINVAL;

	cxi_dom = container_of(domain, struct cxip_domain,
			       util_domain.domain_fid);

	ret = cxip_cq_verify_attr(attr);
	if (ret)
		return ret;

	cxi_cq = calloc(1, sizeof(*cxi_cq));
	if (!cxi_cq)
		return -FI_ENOMEM;

	if (!attr)
		cxi_cq->attr = cxip_cq_def_attr;
	else
		cxi_cq->attr = *attr;

	ret = ofi_cq_init(&cxip_prov, domain, &cxi_cq->attr, &cxi_cq->util_cq,
			  cxip_util_cq_progress, context);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("ofi_cq_init() failed: %d\n", ret);
		goto err_util_cq;
	}

	cxi_cq->util_cq.cq_fid.fid.ops = &cxip_cq_fi_ops;

	cxi_cq->domain = cxi_dom;
	ofi_atomic_initialize32(&cxi_cq->ref, 0);
	fastlock_init(&cxi_cq->lock);
	fastlock_init(&cxi_cq->req_lock);

	ofi_atomic_inc32(&cxi_dom->ref);

	*cq = &cxi_cq->util_cq.cq_fid;

	return FI_SUCCESS;

err_util_cq:
	free(cxi_cq);

	return ret;
}
