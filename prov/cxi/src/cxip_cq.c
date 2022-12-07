
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

bool cxip_cq_saturated(struct cxip_cq *cq)
{
	if (cq->eq.eq_saturated)
		return true;

	/* Hardware will automatically update the EQ status writeback area,
	 * which includes a timestamp, once the EQ reaches a certain fill
	 * percentage. The EQ status timestamp is compare against cached
	 * versions of the previous EQ status timestamp to determine if new
	 * writebacks have occurred. Each time a new writeback occurs, the EQ
	 * is treated as saturated.
	 *
	 * Note that the previous EQ status is always updated when the
	 * corresponding OFI completion queue is progressed.
	 */
	if (cq->eq.eq->status->timestamp_sec > cq->eq.prev_eq_status.timestamp_sec ||
	    cq->eq.eq->status->timestamp_ns > cq->eq.prev_eq_status.timestamp_ns) {
		cq->eq.eq_saturated = true;
		return true;
	}

	return false;
}

int cxip_cq_adjust_reserved_fc_event_slots(struct cxip_cq *cq, int value)
{
	int ret;

	ofi_spin_lock(&cq->lock);

	if (!cq->enabled) {
		ret = -FI_EINVAL;
		goto unlock_out;
	}

	ret = cxil_evtq_adjust_reserved_fc(cq->eq.eq, value);
	if (ret >= 0)
		ret = 0;

unlock_out:
	ofi_spin_unlock(&cq->lock);

	return ret;
}

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

	ofi_spin_lock(&cq->ibuf_lock);
	ibuf = (struct cxip_req *)ofi_buf_alloc(cq->ibuf_pool);
	ofi_spin_unlock(&cq->ibuf_lock);

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
	ofi_spin_lock(&cq->ibuf_lock);
	ofi_buf_free(ibuf);
	ofi_spin_unlock(&cq->ibuf_lock);

	CXIP_DBG("Freed inject buffer: %p\n", ibuf);
}

int cxip_ibuf_chunk_init(struct ofi_bufpool_region *region)
{
	struct ofi_bufpool *pool = region->pool;
	struct cxip_cq *cq = pool->attr.context;
	struct cxip_md *md;
	int ret;
	uintptr_t page_mask = ofi_get_page_size() - 1;
	uintptr_t addr = (uintptr_t)region->alloc_region;

	if ((addr & ~page_mask) != addr ||
	    (pool->alloc_size & ~page_mask) != pool->alloc_size) {
		CXIP_WARN("Buf pool region va=%p len=%lx not page aligned\n",
			  region->alloc_region, region->pool->alloc_size);
		return -FI_EFAULT;
	}

	ret = cxip_map(cq->domain, region->alloc_region, pool->alloc_size,
		       OFI_MR_NOCACHE, &md);
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
	ofi_spin_lock(&cq->lock);

	dlist_foreach_container_safe(&cq->req_list, struct cxip_req, req,
				     cq_entry, tmp) {
		if (req->req_ctx == req_ctx &&
		    req->type == CXIP_REQ_RECV &&
		    !req->recv.canceled &&
		    !req->recv.parent &&
		    (!match || (void *)req->context == op_ctx)) {
			ret = cxip_recv_cancel(req);
			break;
		}
	}

	ofi_spin_unlock(&cq->lock);

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

	ofi_spin_lock(&cq->lock);

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

	ofi_spin_unlock(&cq->lock);
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
	ofi_spin_lock(&cq->lock);

	dlist_foreach_container(&cq->req_list, struct cxip_req, req,
				cq_entry) {
		if (req->req_ctx == req_ctx) {
			req->discard = true;
			discards++;
		}
	}

	if (discards)
		CXIP_DBG("Marked %d requests\n", discards);

	ofi_spin_unlock(&cq->lock);
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

	ofi_spin_lock(&cq->req_lock);

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
	ofi_spin_unlock(&cq->req_lock);

	return req;
}

/*
 * cxip_cq_req_free() - Free a request.
 */
void cxip_cq_req_free(struct cxip_req *req)
{
	ofi_spin_lock(&req->cq->req_lock);
	cxip_cq_req_free_no_lock(req);
	ofi_spin_unlock(&req->cq->req_lock);
}

/*
 * cxip_cq_event_req() - Locate a request corresponding to the Cassini event.
 */
static struct cxip_req *cxip_cq_event_req(struct cxip_cq *cq,
					  const union c_event *event)
{
	struct cxip_req *req;
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
		cxip_pte_state_change(cq->domain->iface, event);

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

static void cxip_cq_eq_progress(struct cxip_cq *cq, struct cxip_cq_eq *eq)
{
	const union c_event *event;
	struct cxip_req *req;
	int ret = FI_SUCCESS;

	/* The EQ status needs to be cached on each poll to be able to properly
	 * determine if the OFI completion queue is saturated.
	 */
	eq->prev_eq_status = *eq->eq->status;

	while ((event = cxi_eq_peek_event(eq->eq))) {
		req = cxip_cq_event_req(cq, event);
		if (req) {
			ret = req->cb(req, event);
			if (ret != FI_SUCCESS)
				break;
		}

		cxi_eq_next_event(eq->eq);

		eq->unacked_events++;
		if (eq->unacked_events == cq->ack_batch_size) {
			cxi_eq_ack_events(eq->eq);
			eq->unacked_events = 0;
		}
	}

	if (cxi_eq_get_drops(eq->eq)) {
		CXIP_WARN("EQ dropped event, rsvd slots %u, free slots %u\n",
			  eq->eq->status->event_slots_rsrvd,
			  eq->eq->status->event_slots_free);
		CXIP_FATAL("Cassini Event Queue overflow detected.\n");
	}

	if (ret == FI_SUCCESS)
		eq->eq_saturated = false;
}

/*
 * cxip_cq_progress() - Progress the CXI Completion Queue.
 *
 * Process events on the underlying Cassini event queue.
 */
void cxip_cq_progress(struct cxip_cq *cq)
{
	ofi_spin_lock(&cq->lock);

	if (!cq->enabled)
		goto out;

	cxip_cq_eq_progress(cq, &cq->eq);

out:
	ofi_spin_unlock(&cq->lock);
}

/*
 * cxip_util_cq_progress() - Progress function wrapper for utility CQ.
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

static void cxip_cq_eq_fini(struct cxip_cq *cq, struct cxip_cq_eq *eq)
{
	cxil_destroy_evtq(eq->eq);

	if (eq->md)
		cxil_unmap(eq->md);
	else
		madvise(eq->buf, eq->len, MADV_DOFORK);

	if (eq->mmap)
		munmap(eq->buf, eq->len);
	else
		free(eq->buf);
}

#define MAP_HUGE_2MB    (21 << MAP_HUGE_SHIFT)
static int cxip_cq_eq_init(struct cxip_cq *cq, struct cxip_cq_eq *eq,
			   size_t len, unsigned int reserved_slots)
{
	struct cxi_eq_attr eq_attr = {
		.reserved_slots = reserved_slots,
	};
	size_t eq_len;
	bool eq_passthrough = false;
	int ret;

	assert(cq->domain->enabled);

	/* Attempt to use 2 MiB hugepages. */
	if (!cxip_env.disable_cq_hugetlb) {
		eq_len = roundup(len, 1U << 21);
		eq->buf = mmap(NULL, eq_len, PROT_READ | PROT_WRITE,
			       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
			       MAP_HUGE_2MB, -1, 0);
		if (eq->buf != MAP_FAILED) {
			eq->mmap = true;

			/* If a single hugepage is used, CXI_EQ_PASSTHROUGH can
			 * be used.
			 */
			if (eq_len <= (1U << 21))
				eq_passthrough = true;
			goto mmap_success;
		}

		CXIP_DBG("Unable to map hugepage for EQ\n");
	}

	eq->mmap = false;
	eq_len = roundup(len, C_PAGE_SIZE);
	eq->buf = aligned_alloc(C_PAGE_SIZE, eq_len);
	if (!eq->buf) {
		CXIP_WARN("Unable to allocate EQ buffer\n");
		return -FI_ENOMEM;
	}

mmap_success:
	/* Buffer has been allocated. Only map if needed. */
	eq->len = eq_len;
	if (eq_passthrough) {
		eq->md = NULL;
		eq_attr.flags |= CXI_EQ_PASSTHROUGH;

		ret = madvise(eq->buf, eq->len, MADV_DONTFORK);
		if (ret) {
			ret = -errno;
			CXIP_WARN("madvise failed: %d\n", ret);
			goto err_free_eq_buf;
		}
	} else {
		ret = cxil_map(cq->domain->lni->lni, eq->buf, eq->len,
			       CXIP_EQ_MAP_FLAGS, NULL, &eq->md);
		if (ret) {
			CXIP_WARN("Unable to map EQ buffer: %d\n", ret);
			goto err_free_eq_buf;
		}
	}

	/* Once the EQ is at CQ fill percentaged full, a status event is
	 * generated. When a status event occurs, the CXIP CQ is considered
	 * saturated until the CXI EQ is drained.
	 */
	eq_attr.status_thresh_base = cxip_env.cq_fill_percent;
	eq_attr.status_thresh_delta = 0;
	eq_attr.status_thresh_count = 1;

	eq_attr.queue = eq->buf;
	eq_attr.queue_len = eq->len;
	eq_attr.flags |= CXI_EQ_TGT_LONG | CXI_EQ_EC_DISABLE;

	/* CPU number will be ignored if invalid */
	if (cq->attr.flags & FI_AFFINITY && cq->attr.signaling_vector > 0)
		eq_attr.cpu_affinity = cq->attr.signaling_vector;

	/* cq->priv_wait is NULL if not backed by wait object */
	ret = cxil_alloc_evtq(cq->domain->lni->lni, eq->md, &eq_attr,
			      cq->priv_wait, NULL, &eq->eq);
	if (ret) {
		CXIP_WARN("Failed to allocated EQ: %d\n", ret);
		goto err_unmap_eq_buf;
	}

	return FI_SUCCESS;

err_unmap_eq_buf:
	if (eq->md)
		cxil_unmap(eq->md);
	else
		madvise(eq->buf, eq->len, MADV_DOFORK);
err_free_eq_buf:
	if (eq->mmap)
		munmap(eq->buf, eq->len);
	else
		free(eq->buf);

	return ret;
}

/*
 * cxip_cq_trywait - Return success if able to block waiting for CQ events.
 */
static int cxip_cq_trywait(void *arg)
{
	struct cxip_cq *cq = (struct cxip_cq *)arg;

	assert(cq->util_cq.wait);

	if (!cq->priv_wait) {
		CXIP_WARN("No CXI wait object\n");
		return -FI_EINVAL;
	}

	if (cxi_eq_peek_event(cq->eq.eq))
		return -FI_EAGAIN;

	/* Clear wait, and check for any events */
	ofi_spin_lock(&cq->lock);
	cxil_clear_wait_obj(cq->priv_wait);

	if (cxi_eq_peek_event(cq->eq.eq)) {
		ofi_spin_unlock(&cq->lock);
		return -FI_EAGAIN;
	}
	ofi_spin_unlock(&cq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_cq_enable() - Assign hardware resources to the CQ.
 */
int cxip_cq_enable(struct cxip_cq *cxi_cq, struct cxip_ep_obj *ep_obj)
{
	struct ofi_bufpool_attr bp_attrs = {};
	int ret = FI_SUCCESS;
	size_t min_eq_size;

	ofi_spin_lock(&cxi_cq->lock);

	if (cxi_cq->enabled)
		goto unlock;

	/* If the CQ is backed by a wait object, add the control
	 * event queue FD to the CQ wait object.
	 */
	if (cxi_cq->util_cq.wait && ep_obj->ctrl_wait) {
		ret = ofi_wait_add_fd(cxi_cq->util_cq.wait,
				      cxil_get_wait_obj_fd(ep_obj->ctrl_wait),
				      POLLIN, cxip_ep_ctrl_trywait, cxi_cq,
				      &cxi_cq->util_cq.cq_fid.fid);
		if (ret) {
			CXIP_WARN("Failed to add wait FD: %d\n", ret);
			goto unlock;
		}
	}

	min_eq_size = (cxi_cq->attr.size + cxi_cq->ack_batch_size) *
		C_EE_CFG_ECB_SIZE;
	ret = cxip_cq_eq_init(cxi_cq, &cxi_cq->eq, min_eq_size, 0);
	if (ret) {
		CXIP_WARN("Failed to initialize TX EQ: %d\n", ret);
		goto del_fd;
	}

	bp_attrs.size = sizeof(struct cxip_req);
	bp_attrs.alignment = 8;
	bp_attrs.chunk_cnt = 64;
	bp_attrs.flags = OFI_BUFPOOL_NO_TRACK;
	ret = ofi_bufpool_create_attr(&bp_attrs, &cxi_cq->req_pool);
	if (ret) {
		ret = -FI_ENOMEM;
		goto err_eq_fini;
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
	bp_attrs.flags = OFI_BUFPOOL_PAGE_ALIGNED;

	ret = ofi_bufpool_create_attr(&bp_attrs, &cxi_cq->ibuf_pool);
	if (ret) {
		ret = -FI_ENOMEM;
		goto err_free_req_pool;
	}

	cxi_cq->enabled = true;
	dlist_init(&cxi_cq->req_list);
	ofi_spin_unlock(&cxi_cq->lock);

	CXIP_DBG("CQ enabled: %p (EQ:%d)\n", cxi_cq, cxi_cq->eq.eq->eqn);
	return FI_SUCCESS;

err_free_req_pool:
	ofi_bufpool_destroy(cxi_cq->req_pool);
err_eq_fini:
	cxip_cq_eq_fini(cxi_cq, &cxi_cq->eq);
del_fd:
	if (cxi_cq->util_cq.wait && ep_obj->ctrl_wait)
		ofi_wait_del_fd(cxi_cq->util_cq.wait,
				cxil_get_wait_obj_fd(ep_obj->ctrl_wait));
unlock:
	ofi_spin_unlock(&cxi_cq->lock);

	return ret;
}

/*
 * cxip_cq_disable() - Release hardware resources from the CQ.
 */
static void cxip_cq_disable(struct cxip_cq *cxi_cq)
{
	ofi_spin_lock(&cxi_cq->lock);

	if (!cxi_cq->enabled)
		goto unlock;

	ofi_idx_reset(&cxi_cq->req_table);

	ofi_bufpool_destroy(cxi_cq->ibuf_pool);

	ofi_bufpool_destroy(cxi_cq->req_pool);

	cxip_cq_eq_fini(cxi_cq, &cxi_cq->eq);

	cxi_cq->enabled = false;

	CXIP_DBG("CQ disabled: %p\n", cxi_cq);
unlock:
	ofi_spin_unlock(&cxi_cq->lock);
}

/*
 * cxip_cq_close() - Destroy the Completion Queue object.
 */
static int cxip_cq_close(struct fid *fid)
{
	struct cxip_cq *cq;
	int ret;

	cq = container_of(fid, struct cxip_cq, util_cq.cq_fid.fid);
	if (ofi_atomic_get32(&cq->ref))
		return -FI_EBUSY;

	cxip_cq_disable(cq);

	if (cq->priv_wait) {
		ret = ofi_wait_del_fd(cq->util_cq.wait,
				      cxil_get_wait_obj_fd(cq->priv_wait));
		if (ret)
			CXIP_WARN("Wait FD delete error: %d\n", ret);

		ret = cxil_destroy_wait_obj(cq->priv_wait);
		if (ret)
			CXIP_WARN("Release CXI wait object failed: %d\n", ret);
	}

	ofi_cq_cleanup(&cq->util_cq);

	ofi_spin_destroy(&cq->lock);
	ofi_spin_destroy(&cq->ibuf_lock);
	ofi_spin_destroy(&cq->req_lock);

	cxip_domain_remove_cq(cq->domain, cq);

	free(cq);

	return 0;
}

static struct fi_ops cxip_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_cq_close,
	.bind = fi_no_bind,
	.control = ofi_cq_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_cq_attr cxip_cq_def_attr = {
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

	/* Applications should set wait_obj == FI_WAIT_NONE for best
	 * performance. However, if a wait_obj is required and not
	 * specified, default to FI_WAIT_FD.
	 */
	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
		break;
	case FI_WAIT_NONE:
	case FI_WAIT_FD:
	case FI_WAIT_POLLFD:
		break;
	default:
		CXIP_WARN("Unsupported CQ wait object: %d\n",
			  attr->wait_obj);
		return -FI_ENOSYS;
	}

	/* Use environment variable to allow for dynamic setting of default CQ
	 * size.
	 */
	if (!attr->size)
		attr->size = cxip_env.default_cq_size;

	return FI_SUCCESS;
}

/*
 * cxip_cq_alloc_priv_wait - Allocate an internal wait channel for the CQ.
 */
static int cxip_cq_alloc_priv_wait(struct cxip_cq *cq)
{
	int ret;
	int wait_fd;

	assert(cq->domain);

	/* Not required or already created */
	if (!cq->util_cq.wait || cq->priv_wait)
		return FI_SUCCESS;

	ret = cxil_alloc_wait_obj(cq->domain->lni->lni, &cq->priv_wait);
	if (ret) {
		CXIP_WARN("Allocation of internal wait object failed %d\n",
			  ret);
		return ret;
	}

	wait_fd = cxil_get_wait_obj_fd(cq->priv_wait);
	ret = fi_fd_nonblock(wait_fd);
	if (ret) {
		CXIP_WARN("Unable to set CQ wait non-blocking mode: %d\n", ret);
		goto destroy_wait;
	}

	ret = ofi_wait_add_fd(cq->util_cq.wait, wait_fd, POLLIN,
			      cxip_cq_trywait, cq, &cq->util_cq.cq_fid.fid);
	if (ret) {
		CXIP_WARN("Add FD of internal wait object failed: %d\n", ret);
		goto destroy_wait;
	}

	CXIP_DBG("Add CQ private wait object, CQ intr FD: %d\n", wait_fd);

	return FI_SUCCESS;

destroy_wait:
	cxil_destroy_wait_obj(cq->priv_wait);
	cq->priv_wait = NULL;

	return ret;
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

	if (!attr) {
		cxi_cq->attr = cxip_cq_def_attr;
		cxi_cq->attr.size = cxip_env.default_cq_size;
	} else {
		cxi_cq->attr = *attr;
	}

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
	ofi_spin_init(&cxi_cq->lock);
	ofi_spin_init(&cxi_cq->req_lock);
	ofi_spin_init(&cxi_cq->ibuf_lock);

	if (cxi_cq->util_cq.wait) {
		ret = cxip_cq_alloc_priv_wait(cxi_cq);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Unable to allocate CXI wait obj: %d\n",
				  ret);
			goto err_wait_alloc;
		}
	}

	cxip_domain_add_cq(cxi_dom, cxi_cq);
	*cq = &cxi_cq->util_cq.cq_fid;

	return FI_SUCCESS;

err_wait_alloc:
	ofi_cq_cleanup(&cxi_cq->util_cq);
err_util_cq:
	free(cxi_cq);

	return ret;
}
