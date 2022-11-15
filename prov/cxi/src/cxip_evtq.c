/*
 * Copyright (c) 2018-2020 Cray Inc. All rights reserved.
 * (C) Copyright 2021-2022 Hewlett Packard Enterprise Development LP
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

/* This file implements the provider interaction with hardware
 * event queues which are used for returning events related to
 * hardware command status and completions. These event queues
 * are unique from the libfabric definition of an EQ or CQ, both
 * of which are software only constructs in the CXI provider.
 */
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
	if (cq->eq.eq->status->timestamp_sec >
	    cq->eq.prev_eq_status.timestamp_sec ||
	    cq->eq.eq->status->timestamp_ns >
	    cq->eq.prev_eq_status.timestamp_ns) {
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

void cxip_cq_eq_progress(struct cxip_cq *cq, struct cxip_cq_eq *eq)
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
	int page_size;

	assert(cq->domain->enabled);

	/* Attempt to use 2 MiB hugepages. */
	if (!cxip_env.disable_cq_hugetlb) {
		eq_len = ofi_get_aligned_size(len, 1U << 21);
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

	page_size = ofi_get_page_size();
	if (page_size < 0)
		return -ofi_syserr();

	eq->mmap = false;
	eq_len = ofi_get_aligned_size(len, page_size);
	eq->buf = aligned_alloc(page_size, eq_len);
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
void cxip_cq_disable(struct cxip_cq *cxi_cq)
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
