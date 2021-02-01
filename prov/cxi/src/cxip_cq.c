
/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018-2020 Cray Inc. All rights reserved.
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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_CQ, __VA_ARGS__)

struct cxip_md *cxip_cq_ibuf_md(void *ibuf)
{
	return ofi_buf_hdr(ibuf)->region->context;
}

/*
 * cxip_ibuf_alloc() - Allocate an inject buffer.
 */
void *cxip_cq_ibuf_alloc(struct cxip_cq *cq)
{
	void *ibuf;

	fastlock_acquire(&cq->ibuf_lock);
	ibuf = (struct cxip_req *)ofi_buf_alloc(cq->ibuf_pool);
	fastlock_release(&cq->ibuf_lock);

	if (ibuf)
		CXIP_DBG("Allocated inject buffer: %p\n", ibuf);
	else
		CXIP_WARN("Failed to allocate inject buffer\n");

	return ibuf;
}

/*
 * cxip_ibuf_free() - Free an inject buffer.
 */
void cxip_cq_ibuf_free(struct cxip_cq *cq, void *ibuf)
{
	fastlock_acquire(&cq->ibuf_lock);
	ofi_buf_free(ibuf);
	fastlock_release(&cq->ibuf_lock);

	CXIP_DBG("Freed inject buffer: %p\n", ibuf);
}

int cxip_ibuf_chunk_init(struct ofi_bufpool_region *region)
{
	struct cxip_cq *cq = region->pool->attr.context;
	struct cxip_md *md;
	int ret;

	ret = cxip_map(cq->domain, region->mem_region,
		       region->pool->region_size, &md);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map inject buffer chunk\n");
		return ret;
	}

	region->context = md;

	return FI_SUCCESS;
}

void cxip_ibuf_chunk_fini(struct ofi_bufpool_region *region)
{
	cxip_unmap(region->context);
}

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
	struct dlist_entry *tmp;

	/* Serialize with event processing that could update request state. */
	fastlock_acquire(&cq->lock);

	dlist_foreach_container_safe(&cq->req_list, struct cxip_req, req,
				     cq_entry, tmp) {
		if (req->req_ctx == req_ctx &&
		    !req->recv.canceled &&
		    !req->recv.parent &&
		    (!match || (void *)req->context == op_ctx)) {
			ret = cxip_recv_cancel(req);
			break;
		}
	}

	fastlock_release(&cq->lock);

	return ret;
}

static void cxip_cq_req_free_no_lock(struct cxip_req *req)
{
	struct cxip_req *table_req;

	CXIP_DBG("Freeing req: %p (ID: %d)\n", req, req->req_id);

	dlist_remove(&req->cq_entry);

	if (req->req_id >= 0) {
		table_req = (struct cxip_req *)ofi_idx_remove(
			&req->cq->req_table, req->req_id);
		if (table_req != req)
			CXIP_WARN("Failed to unmap request: %p\n", req);
	}

	ofi_buf_free(req);
}

/*
 * cxip_cq_flush_trig_reqs() - Flush all triggered requests on the CQ.
 *
 * This function will free all triggered requests associated with a CQ. This
 * should only be called after cancelling triggered operations against all
 * counters in use and verifying the cancellations have completed successfully.
 */
void cxip_cq_flush_trig_reqs(struct cxip_cq *cq)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;
	struct cxip_txc *txc;

	fastlock_acquire(&cq->lock);

	dlist_foreach_container_safe(&cq->req_list, struct cxip_req, req,
				     cq_entry, tmp) {

		if (cxip_is_trig_req(req)) {
			/* If a request is triggered, the context will only be
			 * a TX context (never a RX context).
			 */
			txc = req->req_ctx;

			/* Since an event will not arrive to progress the
			 * request, MDs must be cleaned up now.
			 */
			switch (req->type) {
			case CXIP_REQ_RMA:
				if (req->rma.local_md)
					cxip_unmap(req->rma.local_md);
				if (req->rma.ibuf)
					cxip_cq_ibuf_free(req->cq,
							  req->rma.ibuf);
				break;

			case CXIP_REQ_AMO:
				if (req->amo.oper1_md)
					cxip_unmap(req->amo.oper1_md);
				if (req->amo.result_md)
					cxip_unmap(req->amo.result_md);
				if (req->amo.ibuf)
					cxip_cq_ibuf_free(req->cq,
							  req->amo.ibuf);
				break;

			case CXIP_REQ_SEND:
				if (req->send.send_md)
					cxip_unmap(req->send.send_md);
				if (req->send.ibuf)
					cxip_cq_ibuf_free(req->cq,
							  req->send.ibuf);
				break;

			default:
				CXIP_WARN("Invalid trig req type: %d\n",
					  req->type);
			}

			ofi_atomic_dec32(&txc->otx_reqs);
			cxip_cq_req_free_no_lock(req);
		}

	}

	fastlock_release(&cq->lock);
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
		CXIP_DBG("Marked %d requests\n", discards);

	fastlock_release(&cq->lock);
}

/*
 * cxip_cq_req_complete() - Generate a completion event for the request.
 */
int cxip_cq_req_complete(struct cxip_req *req)
{
	if (req->discard) {
		CXIP_DBG("Event discarded: %p\n", req);
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
		CXIP_DBG("Event discarded: %p\n", req);
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
		CXIP_DBG("Event discarded: %p\n", req);
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
		CXIP_DBG("Failed to allocate request\n");
		goto out;
	}
	memset(req, 0, sizeof(*req));

	if (remap) {
		req->req_id = ofi_idx_insert(&cq->req_table, req);

		/* Target command buffer IDs are 16 bits wide. */
		if (req->req_id < 0 || req->req_id >= CXIP_BUFFER_ID_MAX) {
			CXIP_WARN("Failed to map request: %d\n",
				  req->req_id);
			if (req->req_id > 0)
				ofi_idx_remove(&cq->req_table, req->req_id);
			ofi_buf_free(req);
			req = NULL;
			goto out;
		}
	} else {
		req->req_id = -1;
	}

	CXIP_DBG("Allocated req: %p (ID: %d)\n", req, req->req_id);
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
	fastlock_acquire(&req->cq->req_lock);
	cxip_cq_req_free_no_lock(req);
	fastlock_release(&req->cq->req_lock);
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
	int return_code;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		req = (struct cxip_req *)event->init_short.user_ptr;
		break;
	case C_EVENT_UNLINK:
		switch (cxi_tgt_event_rc(event)) {
		/* User issued unlink events can race with put events. Assume
		 * C_RC_ENTRY_NOT_FOUND is this case.
		 */
		case C_RC_ENTRY_NOT_FOUND:
			return NULL;
		case C_RC_OK:
			break;
		default:
			CXIP_FATAL("Unhandled unlink return code: %d\n",
				   cxi_tgt_event_rc(event));
		}

		/* Fall through. */
	case C_EVENT_LINK:
	case C_EVENT_GET:
	case C_EVENT_PUT:
	case C_EVENT_PUT_OVERFLOW:
	case C_EVENT_RENDEZVOUS:
	case C_EVENT_SEARCH:
		req = cxip_cq_req_find(cq, event->tgt_long.buffer_id);
		if (req)
			break;
		/* HW error can return zero buffer_id */
		CXIP_WARN("Invalid buffer_id: %d (%s)\n",
			  event->tgt_long.buffer_id, cxi_event_to_str(event));
		return_code = cxi_tgt_event_rc(event);
		if (return_code != C_RC_OK)
			CXIP_WARN("Hardware return code: %s (%s)\n",
				  cxi_rc_to_str(return_code),
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
			if (req)
				break;
			/* HW error can return zero buffer_id */
			CXIP_WARN("Invalid buffer_id: %d (%s)\n",
				  event->tgt_long.buffer_id,
				  cxi_event_to_str(event));
			return_code = cxi_tgt_event_rc(event);
			if (return_code != C_RC_OK)
				CXIP_WARN("Hardware return code: %s (%s)\n",
					  cxi_rc_to_str(return_code),
					  cxi_event_to_str(event));
		}
		break;
	case C_EVENT_STATE_CHANGE:
		pte_num = event->tgt_long.ptlte_index;
		pte_state = event->tgt_long.initiator.state_change.ptlte_state;

		cxip_pte_state_change(cq->domain->iface, pte_num, pte_state);

		req = NULL;
		break;
	case C_EVENT_COMMAND_FAILURE:
		CXIP_FATAL("Command failure: cq=%u target=%u fail_loc=%u cmd_type=%u cmd_size=%u opcode=%u\n",
			   event->cmd_fail.cq_id, event->cmd_fail.is_target,
			   event->cmd_fail.fail_loc,
			   event->cmd_fail.fail_command.cmd_type,
			   event->cmd_fail.fail_command.cmd_size,
			   event->cmd_fail.fail_command.opcode);
	default:
		CXIP_FATAL("Invalid event type: %d\n", event->hdr.event_type);
	}

	CXIP_DBG("got event: %s rc: %s (req: %p)\n",
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
	int ret;

	fastlock_acquire(&cq->lock);

	if (!cq->enabled)
		goto out;

	while ((event = cxi_eq_peek_event(cq->evtq))) {
		req = cxip_cq_event_req(cq, event);
		if (req) {
			ret = req->cb(req, event);
			if (ret != FI_SUCCESS)
				break;
		}

		/* Consume event. */
		cxi_eq_next_event(cq->evtq);

		cq->unacked_events++;
		if (cq->unacked_events == cq->ack_batch_size) {
			cxi_eq_ack_events(cq->evtq);
			cq->unacked_events = 0;
		}
	}

	if (cxi_eq_get_drops(cq->evtq))
		CXIP_FATAL("Cassini Event Queue overflow detected.\n");

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

	/* TODO support multiple EPs/CQ */
	if (cq->ep_obj)
		cxip_ep_ctrl_progress(cq->ep_obj);
}

/*
 * cxip_cq_strerror() - Converts provider specific error information into a
 * printable string.
 */
static const char *cxip_cq_strerror(struct fid_cq *cq, int prov_errno,
				    const void *err_data, char *buf,
				    size_t len)
{
	return cxi_rc_to_str(prov_errno);
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

	/* TODO set EQ size based on usage. */
	cxi_cq->evtq_buf_len = 2*1024*1024;

	cxi_cq->evtq_buf = mmap(NULL, cxi_cq->evtq_buf_len,
				PROT_READ | PROT_WRITE,
				MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
				-1, 0);
	if (cxi_cq->evtq_buf == MAP_FAILED) {
		CXIP_DBG("Unable to map hugepage for EQ\n");

		cxi_cq->evtq_buf = aligned_alloc(C_PAGE_SIZE,
						 cxi_cq->evtq_buf_len);
		if (!cxi_cq->evtq_buf) {
			CXIP_WARN("Unable to allocate EQ buffer\n");
			goto unlock;
		}

		ret = cxil_map(cxi_cq->domain->lni->lni,
			       cxi_cq->evtq_buf, cxi_cq->evtq_buf_len,
			       CXI_MAP_PIN | CXI_MAP_WRITE,
			       NULL, &cxi_cq->evtq_buf_md);
		if (ret) {
			CXIP_WARN("Unable to MAP MR EVTQ buffer, ret: %d\n",
				  ret);
			goto free_evtq_buf;
		}
	} else {
		cxi_cq->mmap_buf = true;
		eq_attr.flags |= CXI_EQ_PASSTHROUGH;
	}

	eq_attr.queue = cxi_cq->evtq_buf,
	eq_attr.queue_len = cxi_cq->evtq_buf_len,
	eq_attr.flags |= CXI_EQ_TGT_LONG | CXI_EQ_EC_DISABLE;

	ret = cxil_alloc_evtq(cxi_cq->domain->lni->lni, cxi_cq->evtq_buf_md,
			      &eq_attr, NULL, NULL, &cxi_cq->evtq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate EQ, ret: %d\n", ret);
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

	memset(&bp_attrs, 0, sizeof(bp_attrs));
	bp_attrs.size = CXIP_INJECT_SIZE;
	bp_attrs.alignment = 8;
	bp_attrs.max_cnt = UINT16_MAX;
	bp_attrs.chunk_cnt = 64;
	bp_attrs.alloc_fn = cxip_ibuf_chunk_init;
	bp_attrs.free_fn = cxip_ibuf_chunk_fini;
	bp_attrs.context = cxi_cq;

	ret = ofi_bufpool_create_attr(&bp_attrs, &cxi_cq->ibuf_pool);
	if (ret) {
		ret = -FI_ENOMEM;
		goto free_req_pool;
	}

	cxi_cq->enabled = true;
	fastlock_release(&cxi_cq->lock);
	dlist_init(&cxi_cq->req_list);

	CXIP_DBG("CQ enabled: %p (EQ: %d)\n", cxi_cq, cxi_cq->evtq->eqn);
	return FI_SUCCESS;

free_req_pool:
	ofi_bufpool_destroy(cxi_cq->req_pool);
free_evtq:
	cxil_destroy_evtq(cxi_cq->evtq);
unmap_evtq_buf:
	if (!cxi_cq->mmap_buf) {
		ret = cxil_unmap(cxi_cq->evtq_buf_md);
		if (ret)
			CXIP_WARN("Failed to unmap evtq MD, ret: %d\n", ret);
	}
free_evtq_buf:
	if (cxi_cq->mmap_buf)
		munmap(cxi_cq->evtq_buf, cxi_cq->evtq_buf_len);
	else
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

	ofi_bufpool_destroy(cxi_cq->ibuf_pool);

	ofi_bufpool_destroy(cxi_cq->req_pool);

	ret = cxil_destroy_evtq(cxi_cq->evtq);
	if (ret)
		CXIP_WARN("Failed to free evtq, ret: %d\n", ret);

	if (cxi_cq->mmap_buf) {
		munmap(cxi_cq->evtq_buf, cxi_cq->evtq_buf_len);
	} else {
		ret = cxil_unmap(cxi_cq->evtq_buf_md);
		if (ret)
			CXIP_WARN("Failed to unmap evtq MD, ret: %d\n",
				  ret);
		free(cxi_cq->evtq_buf);
	}

	cxi_cq->enabled = false;

	CXIP_DBG("CQ disabled: %p\n", cxi_cq);
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
	fastlock_destroy(&cq->ibuf_lock);
	fastlock_destroy(&cq->req_lock);

	cxip_domain_remove_cq(cq->domain, cq);

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
		return FI_SUCCESS;

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
		CXIP_WARN("Unsupported CQ attribute format: %d\n",
			  attr->format);
		return -FI_ENOSYS;
	}

	if (attr->wait_obj != FI_WAIT_NONE) {
		CXIP_WARN("CQ wait objects not supported\n");
		return -FI_ENOSYS;
	}

	if (!attr->size)
		attr->size = cxip_cq_def_attr.size;

	return FI_SUCCESS;
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
	if (ret != FI_SUCCESS)
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
		CXIP_WARN("ofi_cq_init() failed: %d\n", ret);
		goto err_util_cq;
	}

	cxi_cq->util_cq.cq_fid.ops->strerror = &cxip_cq_strerror;
	cxi_cq->util_cq.cq_fid.fid.ops = &cxip_cq_fi_ops;

	cxi_cq->domain = cxi_dom;
	cxi_cq->ack_batch_size = cxip_env.eq_ack_batch_size;
	ofi_atomic_initialize32(&cxi_cq->ref, 0);
	fastlock_init(&cxi_cq->lock);
	fastlock_init(&cxi_cq->req_lock);
	fastlock_init(&cxi_cq->ibuf_lock);

	cxip_domain_add_cq(cxi_dom, cxi_cq);

	*cq = &cxi_cq->util_cq.cq_fid;

	return FI_SUCCESS;

err_util_cq:
	free(cxi_cq);

	return ret;
}
