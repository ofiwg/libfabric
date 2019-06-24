/*
 * SPDX-License-Identifier: GPL-2.0
 *
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

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

/*
 * issue_append_le() - Append a buffer to a PtlTE.
 *
 * TODO: Make common.
 */
static int issue_append_le(struct cxil_pte *pte, const void *buf, size_t len,
			   struct cxip_md *md, enum c_ptl_list list,
			   uint32_t buffer_id, uint64_t match_bits,
			   uint64_t ignore_bits, uint32_t match_id,
			   uint64_t min_free, bool event_success_disable,
			   bool use_once, bool manage_local, bool no_truncate,
			   bool op_put, bool op_get, struct cxip_cmdq *cmdq)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode      = C_CMD_TGT_APPEND;
	cmd.target.ptl_list     = list;
	cmd.target.ptlte_index  = pte->ptn;
	cmd.target.op_put       = op_put ? 1 : 0;
	cmd.target.op_get       = op_get ? 1 : 0;
	cmd.target.manage_local = manage_local ? 1 : 0;
	cmd.target.no_truncate  = no_truncate ? 1 : 0;
	cmd.target.unexpected_hdr_disable = 0;
	cmd.target.buffer_id    = buffer_id;
	cmd.target.lac          = md->md->lac;
	cmd.target.start        = CXI_VA_TO_IOVA(md->md, buf);
	cmd.target.length       = len;
	cmd.target.event_success_disable = event_success_disable ? 1 : 0;
	cmd.target.use_once     = use_once ? 1 : 0;
	cmd.target.match_bits   = match_bits;
	cmd.target.ignore_bits  = ignore_bits;
	cmd.target.match_id     = match_id;
	cmd.target.min_free     = min_free;

	fastlock_acquire(&cmdq->lock);

	rc = cxi_cq_emit_target(cmdq->dev_cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		fastlock_release(&cmdq->lock);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq->dev_cmdq);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;
}

/*
 * issue_unlink_le() - Unlink a buffer from a PtlTE.
 *
 * TODO: Make common.
 */
static int issue_unlink_le(struct cxil_pte *pte, enum c_ptl_list list,
			   int buffer_id, struct cxip_cmdq *cmdq)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = list;
	cmd.target.ptlte_index  = pte->ptn;
	cmd.target.buffer_id = buffer_id;

	fastlock_acquire(&cmdq->lock);

	rc = cxi_cq_emit_target(cmdq->dev_cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		fastlock_release(&cmdq->lock);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq->dev_cmdq);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;
}

/*
 * match_ux_send() - Search for an unexpected Put that matches a Put Overflow
 * event.
 *
 * Caller must hold rxc->rx_lock.
 */
static struct cxip_ux_send *
match_ux_send(struct cxip_rxc *rxc, const union c_event *event)
{
	struct cxip_ux_send *ux_send;

	if (event->tgt_long.rendezvous) {
		uint32_t process = event->tgt_long.initiator.initiator.process;
		union cxip_match_bits mb = {
			.raw = event->tgt_long.match_bits,
		};
		uint32_t ev_rdzv_id = RDZV_ID(event->tgt_long.rendezvous_id,
					      mb.rdzv_id_lo);

		/* Rendezvous events are correlated using rendezvous_id and
		 * initiator.
		 */
		dlist_foreach_container(&rxc->ux_rdzv_sends,
					struct cxip_ux_send, ux_send,
					ux_entry) {
			if ((ux_send->rdzv_id == ev_rdzv_id) &&
			    (ux_send->initiator == process))
				return ux_send;
		}
	} else {
		/* All other events are correlated using start address.
		 *
		 * TODO this assumes all overflow buffers use the same AC so
		 * all start pointers are unique.
		 */
		dlist_foreach_container(&rxc->ux_sends, struct cxip_ux_send,
					ux_send, ux_entry) {
			if (ux_send->start == event->tgt_long.start)
				return ux_send;
		}
	}

	return NULL;
}

/*
 * match_ux_recv() - Search for a previously matched request that matches an
 * unexpected Put event.
 *
 * Caller must hold rxc->rx_lock.
 */
static struct cxip_req *
match_ux_recv(struct cxip_rxc *rxc, const union c_event *event)
{
	struct cxip_req *req;

	if (event->tgt_long.rendezvous) {
		uint32_t process = event->tgt_long.initiator.initiator.process;
		union cxip_match_bits mb = {
			.raw = event->tgt_long.match_bits,
		};
		uint32_t ev_rdzv_id = RDZV_ID(event->tgt_long.rendezvous_id,
					      mb.rdzv_id_lo);

		/* Rendezvous events are correlated using rendezvous_id and
		 * initiator.
		 */
		dlist_foreach_container(&rxc->ux_rdzv_recvs, struct cxip_req,
					req, recv.ux_entry) {
			if ((req->recv.rdzv_id == ev_rdzv_id) &&
			    (req->recv.initiator == process))
				return req;
		}
	} else {
		/* All other events are correlated using start address.
		 *
		 * TODO this assumes all overflow buffers use the same AC so
		 * all start pointers are unique.
		 */
		dlist_foreach_container(&rxc->ux_recvs, struct cxip_req, req,
					recv.ux_entry) {
			if (req->recv.start == event->tgt_long.start)
				return req;
		}
	}

	return NULL;
}

/*
 * _rxc_event_src_addr() - Translate event process ID to FI address.
 */
static fi_addr_t _rxc_event_src_addr(struct cxip_rxc *rxc,
				     uint32_t process)
{
	/* If the FI_SOURCE capability is enabled, convert the initiator's
	 * address to an FI address to be reported in a CQ event. If
	 * application AVs are symmetric, the match_id in the EQ event is
	 * logical and translation is not needed.  Otherwise, translate the
	 * physical address in the EQ event to logical FI address.
	 */
	if (rxc->attr.caps & FI_SOURCE) {
		uint32_t pid_bits = rxc->domain->dev_if->if_dev->info.pid_bits;
		uint32_t nic;
		uint32_t pid;

		if (!rxc->ep_obj->rdzv_offload &&
		    rxc->ep_obj->av->attr.flags & FI_SYMMETRIC)
			return CXI_MATCH_ID_EP(pid_bits, process);

		nic = CXI_MATCH_ID_EP(pid_bits, process);
		pid = CXI_MATCH_ID_PID(pid_bits, process);

		return _cxip_av_reverse_lookup(rxc->ep_obj->av, nic, pid);
	}

	return FI_ADDR_NOTAVAIL;
}

/*
 * report_recv_completion() - Report the completion of a receive operation.
 */
static void report_recv_completion(struct cxip_req *req)
{
	int ret;
	int truncated;
	int err;
	fi_addr_t src_addr;

	req->data_len = req->recv.mlength;

	truncated = req->recv.rlength - req->recv.mlength;
	if (req->recv.rc == C_RC_OK && !truncated) {
		CXIP_LOG_DBG("Request success: %p\n", req);

		src_addr = _rxc_event_src_addr(req->recv.rxc,
					       req->recv.initiator);
		ret = cxip_cq_req_complete_addr(req, src_addr);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report completion: %d\n",
				       ret);

		if (req->recv.rxc->recv_cntr) {
			ret = cxip_cntr_mod(req->recv.rxc->recv_cntr, 1, false,
					    false);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	} else {
		if (req->recv.rc == C_RC_CANCELED)
			err = FI_ECANCELED;
		else if (truncated)
			err = FI_EMSGSIZE;
		else
			err = FI_EIO;

		CXIP_LOG_DBG("Request error: %p (err: %d, %d)\n", req, err,
			     req->recv.rc);

		ret = cxip_cq_req_error(req, truncated, err, req->recv.rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);

		if (req->recv.rxc->recv_cntr) {
			ret = cxip_cntr_mod(req->recv.rxc->recv_cntr, 1, false,
					    true);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	}
}

/*
 * oflow_buf_free() - Free an Overflow buffer.
 *
 * Caller must hold rxc->rx_lock.
 */
static void oflow_buf_free(struct cxip_oflow_buf *oflow_buf)
{
	dlist_remove(&oflow_buf->list);
	ofi_atomic_dec32(&oflow_buf->rxc->oflow_bufs_in_use);

	cxip_unmap(oflow_buf->md);
	free(oflow_buf->buf);
	free(oflow_buf);
}

/*
 * oflow_req_put_bytes() - Consume bytes in the Overflow buffer.
 *
 * An Overflow buffer is freed when all bytes are consumed by the NIC.
 *
 * Caller must hold rxc->rx_lock.
 */
static void oflow_req_put_bytes(struct cxip_req *req, size_t bytes)
{
	req->oflow.oflow_buf->min_bytes -= bytes;
	if (req->oflow.oflow_buf->min_bytes < 0) {
		oflow_buf_free(req->oflow.oflow_buf);
		cxip_cq_req_free(req);
	}
}

/*
 * issue_rdzv_get() - Perform a Get to pull source data from the Initiator of a
 * Send operation.
 */
static int issue_rdzv_get(struct cxip_req *req, const union c_event *event)
{
	const struct c_event_target_long *tev = &event->tgt_long;
	union c_cmdu cmd = {};
	struct cxip_rxc *rxc = req->recv.rxc;
	uint32_t pid_granule = rxc->domain->dev_if->if_dev->info.pid_granule;
	uint32_t pid_idx = rxc->domain->dev_if->if_dev->info.rdzv_get_idx;
	uint32_t pid_bits = rxc->domain->dev_if->if_dev->info.pid_bits;
	uint8_t idx_ext;
	uint32_t initiator = tev->initiator.initiator.process;
	int ret;

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_GET;
	cmd.full_dma.lac = req->recv.recv_md->md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.eq = rxc->recv_cq->evtq->eqn;
	cmd.full_dma.initiator = CXI_MATCH_ID(pid_bits,
					      rxc->ep_obj->src_addr.pid,
					      rxc->ep_obj->src_addr.nic);
	cmd.full_dma.match_bits = tev->match_bits;
	cmd.full_dma.user_ptr = (uint64_t)req;

	if (tev->rendezvous) {
		/* TODO fix address format translation issues */
		cmd.full_dma.dfa.unicast.nid = initiator >> 12;
		cmd.full_dma.dfa.unicast.endpoint_defined = initiator & 0xFFF;
		cmd.full_dma.index_ext =
				PTL_INDEX_EXT(0, pid_granule, pid_idx);
		cmd.full_dma.request_len = tev->rlength - tev->mlength;
		cmd.full_dma.local_addr = tev->start;
		cmd.full_dma.remote_offset = tev->remote_offset;
	} else {
		/* TODO translate initiator if logical */
		uint32_t nic = CXI_MATCH_ID_EP(pid_bits, initiator);
		uint32_t pid = CXI_MATCH_ID_PID(pid_bits, initiator);
		union c_fab_addr dfa;

		cxi_build_dfa(nic, pid, pid_granule, pid_idx, &dfa, &idx_ext);
		cmd.full_dma.dfa = dfa;
		cmd.full_dma.index_ext = idx_ext;
		cmd.full_dma.request_len = req->recv.mlength;
		cmd.full_dma.local_addr = CXI_VA_TO_IOVA(req->recv.recv_md->md,
							 req->recv.recv_buf);
		cmd.full_dma.remote_offset = 0;
	}

	fastlock_acquire(&rxc->tx_cmdq->lock);

	/* Issue Rendezvous Get command */
	ret = cxi_cq_emit_dma_f(rxc->tx_cmdq->dev_cmdq, &cmd.full_dma);
	if (ret) {
		CXIP_LOG_ERROR("Failed to queue GET command: %d\n", ret);

		ret = -FI_EAGAIN;
		goto unlock;
	}

	cxi_cq_ring(rxc->tx_cmdq->dev_cmdq);

	ret = FI_SUCCESS;
unlock:
	fastlock_release(&rxc->tx_cmdq->lock);
	return ret;
}

/*
 * recv_req_complete() - Complete receive request.
 */
static void recv_req_complete(struct cxip_req *req)
{
	cxip_unmap(req->recv.recv_md);
	report_recv_completion(req);
	ofi_atomic_dec32(&req->recv.rxc->orx_reqs);
	cxip_cq_req_free(req);
}

/*
 * recv_req_put_event() - Update receive request using the received Put event.
 *
 * Use rlength, match_bits, initiator, and return_code in Put event to update
 * receive request state. This information is available in all Put and Put
 * Overflow events.
 */
static void
recv_req_put_event(struct cxip_req *req, const union c_event *event)
{
	union cxip_match_bits mb;

	/* Only update request fields once. A Put event may be processed
	 * multiple times if an error is seen.
	 */
	if (req->recv.put_event)
		return;
	req->recv.put_event = true;

	if (event->tgt_long.rlength > req->recv.rlength) {
		/* Send truncated. */
		req->recv.mlength = req->recv.rlength;
	} else {
		req->recv.mlength = event->tgt_long.rlength;
	}
	req->recv.rlength = event->tgt_long.rlength;

	req->recv.rc = cxi_tgt_event_rc(event);

	mb.raw = event->tgt_long.match_bits;
	req->tag = mb.tag;
	req->recv.initiator = event->tgt_long.initiator.initiator.process;
}

/*
 * cxip_oflow_sink_cb() - Process an Overflow buffer event the sink buffer.
 *
 * The sink buffer matches all unexpected long eager sends. The sink buffer
 * truncates all send data and is never exhausted. See cxip_oflow_cb() for more
 * details about Overflow buffer event handling.
 */
static int
cxip_oflow_sink_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	struct cxip_rxc *rxc = req->oflow.rxc;
	struct cxip_req *ux_recv;
	struct cxip_ux_send *ux_send;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		ofi_atomic_inc32(&rxc->ux_sink_linked);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		/* Long sink buffer was manually unlinked. */
		ofi_atomic_dec32(&rxc->ux_sink_linked);

		/* Clean up overflow buffers */
		cxip_unmap(rxc->ux_sink_buf.md);
		free(rxc->ux_sink_buf.buf);
		cxip_cq_req_free(req);
		return FI_SUCCESS;
	case C_EVENT_PUT:
		/* Put event handling is complicated. Handle below. */
		break;
	default:
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
		return FI_SUCCESS;
	}

	/* Handle Put events */
	fastlock_acquire(&rxc->rx_lock);

	/* Check for a previously received Put Overflow event */
	ux_recv = match_ux_recv(rxc, event);
	if (!ux_recv) {
		/* A Put Overflow event is pending. Store a record of this
		 * unexpected Put event for lookup when the event arrives.
		 */

		/* TODO make fast allocator for ux_sends */
		ux_send = malloc_f(sizeof(struct cxip_ux_send));
		if (!ux_send) {
			CXIP_LOG_ERROR("Failed to malloc ux_send\n");
			goto err_put;
		}

		/* Use start pointer for matching. */
		ux_send->req = req;
		ux_send->start = event->tgt_long.start;

		dlist_insert_tail(&ux_send->ux_entry, &rxc->ux_sends);

		fastlock_release(&rxc->rx_lock);

		CXIP_LOG_DBG("Queued ux_send: %p\n", ux_send);

		return FI_SUCCESS;
	}

	CXIP_LOG_DBG("Matched ux_recv, data: 0x%lx\n",
		     ux_recv->recv.start);

	/* For long eager messages, issue a Get to retrieve data
	 * from the initiator.
	 */

	ret = issue_rdzv_get(ux_recv, event);
	if (ret != FI_SUCCESS)
		goto err_put;

	dlist_remove(&ux_recv->recv.ux_entry);

	fastlock_release(&rxc->rx_lock);

	return FI_SUCCESS;

err_put:
	fastlock_release(&rxc->rx_lock);

	return -FI_EAGAIN;
}

/*
 * cxip_oflow_rdzv_cb() - Progress an Overflow buffer rendezvous event.
 *
 * All target events which are related to a offloaded rendezvous Put operation
 * have the rendezvous bit set. Handle all rendezvous events from an Overflow
 * buffer. See cxip_oflow_cb() for more details about Overflow buffer event
 * handling.
 */
static int
cxip_oflow_rdzv_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->oflow.rxc;
	struct cxip_oflow_buf *oflow_buf = req->oflow.oflow_buf;
	struct cxip_req *ux_recv;
	struct cxip_ux_send *ux_send;
	void *oflow_va;

	if (event->hdr.event_type != C_EVENT_PUT) {
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
		return FI_SUCCESS;
	}

	/* Handle Put events */
	fastlock_acquire(&rxc->rx_lock);

	/* Check for a previously received Put Overflow event */
	ux_recv = match_ux_recv(rxc, event);
	if (!ux_recv) {
		/* A Put Overflow event is pending. Store a record of this
		 * unexpected Put event for lookup when the event arrives.
		 */
		union cxip_match_bits mb;

		/* TODO make fast allocator for ux_sends */
		ux_send = malloc_f(sizeof(struct cxip_ux_send));
		if (!ux_send) {
			CXIP_LOG_ERROR("Failed to malloc ux_send\n");
			goto err_put;
		}

		/* Use initiator and rdzv_id for matching. Store start pointer
		 * since this is the only place that it's available for
		 * offloaded rendezvous operations.
		 */
		mb.raw = event->tgt_long.match_bits;
		ux_send->req = req;
		ux_send->start = event->tgt_long.start;
		ux_send->initiator =
				event->tgt_long.initiator.initiator.process;
		ux_send->rdzv_id = RDZV_ID(event->tgt_long.rendezvous_id,
					   mb.rdzv_id_lo);
		ux_send->eager_bytes = event->tgt_long.mlength;

		dlist_insert_tail(&ux_send->ux_entry, &rxc->ux_rdzv_sends);

		fastlock_release(&rxc->rx_lock);

		CXIP_LOG_DBG("Queued ux_send: %p\n", ux_send);

		return FI_SUCCESS;
	}

	CXIP_LOG_DBG("Matched ux_recv, data: 0x%lx\n",
		     ux_recv->recv.start);

	/* A matching Put Overflow event arrived earlier. Data is
	 * waiting in the overflow buffer.
	 */

	oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md->md,
					  event->tgt_long.start);
	memcpy(ux_recv->recv.recv_buf, oflow_va, event->tgt_long.mlength);
	oflow_req_put_bytes(req, event->tgt_long.mlength);

	dlist_remove(&ux_recv->recv.ux_entry);

	fastlock_release(&rxc->rx_lock);

	return FI_SUCCESS;

err_put:
	fastlock_release(&rxc->rx_lock);

	return -FI_EAGAIN;
}

int cxip_rxc_eager_replenish(struct cxip_rxc *rxc);

/*
 * cxip_oflow_cb() - Process an Overflow buffer event.
 *
 * Overflow buffers are used to land unexpected Send data. Link, Unlink and Put
 * events are expected from Overflow buffers.
 *
 * A Link event indicates that a new buffer has been appended to the Overflow
 * list.
 *
 * An Unlink event indicates that buffer space was exhausted. Overflow buffers
 * are configured to use locally managed LEs. When enough Puts match in an
 * Overflow buffer, consuming its space, the NIC automatically unlinks the LE.
 * An automatic Unlink event is generated before the final Put which caused
 * buffer space to become exhausted.
 *
 * An Unlink may also be generated by an Unlink command. Overflow buffers are
 * manually unlinked in this way during teardown. When an LE is manually
 * unlinked the auto_unlinked field in the corresponding event is zero. In this
 * case, the request is freed immediately.
 *
 * A Put event is generated for each Put that matches the Overflow buffer LE.
 * This event indicates that data is available in the Overflow buffer. This
 * event must be correlated to a Put Overflow event from a user receive buffer
 * LE. The Put Overflow event may arrive before or after the Put event.
 *
 * When each Put event arrives, check for the existence of a previously posted
 * receive buffer which generated a matching Put Overflow event. If such a
 * buffer exists, copy data from the Overflow buffer to the user receive
 * buffer. Otherwise, store a record of the Put event for matching once a user
 * posts a new buffer that matches the unexpected Put.
 *
 * If data will remain in the Overflow buffer, take a reference to it to
 * prevent it from being freed. If a sequence of Unlink-Put events is detected,
 * drop a reference to the Overflow buffer so it is automatically freed once
 * all user data is copied out.
 */
static int cxip_oflow_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->oflow.rxc;
	struct cxip_oflow_buf *oflow_buf = req->oflow.oflow_buf;
	struct cxip_req *ux_recv;
	struct cxip_ux_send *ux_send;
	void *oflow_va;

	if (event->tgt_long.rendezvous)
		return cxip_oflow_rdzv_cb(req, event);

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		CXIP_LOG_DBG("Eager buffer linked: %p\n", req);
		ofi_atomic_inc32(&rxc->oflow_bufs_linked);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		CXIP_LOG_DBG("Eager buffer unlinked (%s): %p\n",
			     event->tgt_long.auto_unlinked ? "auto" : "manual",
			     req);

		fastlock_acquire(&rxc->rx_lock);

		ofi_atomic_dec32(&rxc->oflow_bufs_submitted);
		ofi_atomic_dec32(&rxc->oflow_bufs_linked);

		if (!event->tgt_long.auto_unlinked) {
			uint64_t bytes = rxc->oflow_buf_size -
					(event->tgt_long.start -
					 CXI_VA_TO_IOVA(oflow_buf->md->md,
							oflow_buf->buf));
			oflow_req_put_bytes(req, bytes);
		} else {
			/* Replace the eager overflow buffer */
			cxip_rxc_eager_replenish(rxc);
		}

		fastlock_release(&rxc->rx_lock);

		return FI_SUCCESS;
	case C_EVENT_PUT:
		/* Put event handling is complicated. Handle below. */
		break;
	default:
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
		return FI_SUCCESS;
	}

	/* Handle Put events */
	fastlock_acquire(&rxc->rx_lock);

	/* Check for a previously received Put Overflow event */
	ux_recv = match_ux_recv(rxc, event);
	if (!ux_recv) {
		/* A Put Overflow event is pending. Store a record of this
		 * unexpected Put event for lookup when the event arrives.
		 */

		/* TODO make fast allocator for ux_sends */
		ux_send = malloc_f(sizeof(struct cxip_ux_send));
		if (!ux_send) {
			CXIP_LOG_ERROR("Failed to malloc ux_send\n");
			goto err_put;
		}

		/* Use start pointer for matching. */
		ux_send->req = req;
		ux_send->start = event->tgt_long.start;
		ux_send->eager_bytes = event->tgt_long.mlength;

		dlist_insert_tail(&ux_send->ux_entry, &rxc->ux_sends);

		fastlock_release(&rxc->rx_lock);

		CXIP_LOG_DBG("Queued ux_send: %p\n", ux_send);

		return FI_SUCCESS;
	}

	/* A matching Put Overflow event arrived earlier. Data is
	 * waiting in the overflow buffer.
	 */

	CXIP_LOG_DBG("Matched ux_recv, data: 0x%lx\n",
		     ux_recv->recv.start);

	/* Copy data out of overflow buffer. */
	oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md->md,
					  event->tgt_long.start);
	memcpy(ux_recv->recv.recv_buf, oflow_va, ux_recv->recv.mlength);
	oflow_req_put_bytes(req, ux_recv->recv.mlength);

	dlist_remove(&ux_recv->recv.ux_entry);

	fastlock_release(&rxc->rx_lock);

	recv_req_complete(ux_recv);

	return FI_SUCCESS;

err_put:
	fastlock_release(&rxc->rx_lock);

	return -FI_EAGAIN;
}

/*
 * eager_buf_add() - Append a Locally Managed LE to the Overflow list to match
 * eager Sends.
 *
 * Caller must hold rxc->rx_lock.
 */
static int eager_buf_add(struct cxip_rxc *rxc)
{
	struct cxip_domain *dom;
	int ret;
	struct cxip_oflow_buf *oflow_buf;
	struct cxip_req *req;
	uint64_t min_free;

	/* Match all eager, long sends */
	union cxip_match_bits mb = { .sink = 0 };
	union cxip_match_bits ib = {
		.tagged = 1,
		.rdzv_id_lo = ~0,
		.tag = ~0
	};

	dom = rxc->domain;

	/* Create an overflow buffer structure */
	oflow_buf = calloc(1, sizeof(*oflow_buf));
	if (!oflow_buf) {
		CXIP_LOG_ERROR("Unable to allocate oflow buffer structure\n");
		return -FI_ENOMEM;
	}

	/* Allocate overflow data buffer */
	oflow_buf->buf = calloc(1, rxc->oflow_buf_size);
	if (!oflow_buf->buf) {
		CXIP_LOG_ERROR("Unable to allocate oflow buffer\n");
		ret = -FI_ENOMEM;
		goto free_oflow;
	}

	/* Map overflow data buffer */
	ret = cxip_map(dom, (void *)oflow_buf->buf, rxc->oflow_buf_size,
		       &oflow_buf->md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map oflow buffer: %d\n", ret);
		goto free_buf;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, NULL);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto oflow_unmap;
	}

	req->cb = cxip_oflow_cb;
	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = oflow_buf;

	min_free = (rxc->eager_threshold >>
			dom->dev_if->if_dev->info.min_free_shift);

	/* Issue Append command */
	ret = issue_append_le(rxc->rx_pte->pte, oflow_buf->buf,
			      rxc->oflow_buf_size, oflow_buf->md,
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, min_free, false, false, true,
			      true, true, false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto oflow_req_free;
	}

	/* Initialize oflow_buf structure */
	dlist_insert_tail(&oflow_buf->list, &rxc->oflow_bufs);
	oflow_buf->rxc = rxc;
	oflow_buf->min_bytes = rxc->oflow_buf_size - min_free;
	oflow_buf->buffer_id = req->req_id;
	oflow_buf->type = OFLOW_BUF_EAGER;

	ofi_atomic_inc32(&rxc->oflow_bufs_submitted);
	ofi_atomic_inc32(&rxc->oflow_bufs_in_use);
	CXIP_LOG_DBG("Eager buffer created: %p\n", req);

	return FI_SUCCESS;

oflow_req_free:
	cxip_cq_req_free(req);
oflow_unmap:
	cxip_unmap(oflow_buf->md);
free_buf:
	free(oflow_buf->buf);
free_oflow:
	free(oflow_buf);

	return ret;
}

/*
 * cxip_rxc_eager_replenish() - Replenish RXC eager overflow buffers.
 *
 * Caller must hold rxc->rx_lock.
 */
int cxip_rxc_eager_replenish(struct cxip_rxc *rxc)
{
	int ret = FI_SUCCESS;

	while (ofi_atomic_get32(&rxc->oflow_bufs_submitted) <
	       rxc->oflow_bufs_max) {
		ret = eager_buf_add(rxc);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_ERROR("Failed to append oflow buffer: %d\n",
				       ret);
			break;
		}
	}

	return ret;
}

/*
 * cxip_rxc_eager_fini() - Free RXC eager overflow buffers.
 */
static int cxip_rxc_eager_fini(struct cxip_rxc *rxc)
{
	int ret = FI_SUCCESS;
	struct cxip_oflow_buf *oflow_buf;

	/* Manually unlink each overflow buffer */
	dlist_foreach_container(&rxc->oflow_bufs, struct cxip_oflow_buf,
				oflow_buf, list) {
		ret = issue_unlink_le(rxc->rx_pte->pte, C_PTL_LIST_OVERFLOW,
				      oflow_buf->buffer_id, rxc->rx_cmdq);
		if (ret != FI_SUCCESS) {
			/* TODO handle error */
			CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n",
				       ret);
			break;
		}
	}

	return ret;
}

/*
 * cxip_rxc_sink_init() - Initialize RXC sink buffer.
 *
 * The sink buffer is used for matching long eager sends in the Overflow list.
 * The sink buffer matches all long eager sends that do not match in the
 * priority list and truncates all data. The sink buffer is not used with the
 * off-loaded rendezvous protocol.
 */
static int cxip_rxc_sink_init(struct cxip_rxc *rxc)
{
	struct cxip_domain *dom;
	int ret;
	struct cxip_req *req;
	void *ux_buf;
	struct cxip_md *md;

	/* Match all eager, long sends */
	union cxip_match_bits mb = { .sink = 1 };
	union cxip_match_bits ib = {
		.tagged = 1,
		.rdzv_id_lo = ~0,
		.tag = ~0
	};

	dom = rxc->domain;

	/* Allocate a small data buffer */
	ux_buf = calloc(1, 1);
	if (!ux_buf) {
		CXIP_LOG_ERROR("Unable to allocate ux buffer\n");
		return -FI_ENOMEM;
	}

	/* Map overflow data buffer */
	ret = cxip_map(dom, ux_buf, 1, &md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map ux buffer: %d\n", ret);
		goto free_ux_buf;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, NULL);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate ux request\n");
		ret = -FI_ENOMEM;
		goto unmap_ux;
	}

	ret = issue_append_le(rxc->rx_pte->pte, ux_buf, 1, md, // TODO len 1?
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, false, false, true, false,
			      true, false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write UX Append command: %d\n", ret);
		goto req_free;
	}

	/* Initialize oflow_buf structure */
	rxc->ux_sink_buf.type = OFLOW_BUF_SINK;
	rxc->ux_sink_buf.buf = ux_buf;
	rxc->ux_sink_buf.md = md;
	rxc->ux_sink_buf.rxc = rxc;
	rxc->ux_sink_buf.buffer_id = req->req_id;

	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = &rxc->ux_sink_buf;
	req->cb = cxip_oflow_sink_cb;

	return FI_SUCCESS;

req_free:
	cxip_cq_req_free(req);
unmap_ux:
	cxip_unmap(md);
free_ux_buf:
	free(ux_buf);

	return ret;
}

/*
 * cxip_rxc_sink_fini() - Tear down RXC sink buffer.
 */
static int cxip_rxc_sink_fini(struct cxip_rxc *rxc)
{
	int ret;

	ret = issue_unlink_le(rxc->rx_pte->pte, C_PTL_LIST_OVERFLOW,
			      rxc->ux_sink_buf.buffer_id, rxc->rx_cmdq);
	if (ret) {
		/* TODO handle error */
		CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n", ret);
	}

	return ret;
}

/*
 * cxip_msg_oflow_init() - Initialize overflow buffers used for messaging.
 *
 * Must be called with the RX PtlTE disabled.
 */
int cxip_msg_oflow_init(struct cxip_rxc *rxc)
{
	int ret;

	ret = cxip_rxc_eager_replenish(rxc);
	if (ret) {
		CXIP_LOG_ERROR("cxip_rxc_eager_replenish failed: %d\n", ret);
		return ret;
	}

	ret = cxip_rxc_sink_init(rxc);
	if (ret) {
		CXIP_LOG_ERROR("cxip_rxc_sink_init failed: %d\n", ret);
		cxip_rxc_eager_fini(rxc);
		return ret;
	}

	/* Wait for Overflow buffers to be linked. */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&rxc->oflow_bufs_linked) <
		 rxc->oflow_bufs_max ||
		 !ofi_atomic_get32(&rxc->ux_sink_linked));

	return FI_SUCCESS;
}

/*
 * cxip_msg_oflow_fini() - Finalize overflow buffers used for messaging.
 *
 * Must be called with the RX PtlTE disabled.
 */
void cxip_msg_oflow_fini(struct cxip_rxc *rxc)
{
	int ret;
	struct cxip_ux_send *ux_send;
	struct dlist_entry *tmp;
	int ux_sends = 0;

	/* Clean up unexpected Put records. The PtlTE is disabled, so no more
	 * events can be expected.
	 */
	dlist_foreach_container_safe(&rxc->ux_sends, struct cxip_ux_send,
				     ux_send, ux_entry, tmp) {
		/* Dropping the last reference will cause the oflow_buf to be
		 * removed from the RXC list and freed.
		 */
		if (ux_send->req->oflow.oflow_buf->type == OFLOW_BUF_EAGER)
			oflow_req_put_bytes(ux_send->req,
					    ux_send->eager_bytes);

		dlist_remove(&ux_send->ux_entry);
		free(ux_send);
		ux_sends++;
	}

	if (ux_sends)
		CXIP_LOG_DBG("Freed %d UX Send(s)\n", ux_sends);

	ret = cxip_rxc_sink_fini(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("cxip_rxc_sink_fini() returned: %d\n", ret);
		return;
	}

	ret = cxip_rxc_eager_fini(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_ERROR("cxip_rxc_eager_fini() returned: %d\n", ret);
		return;
	}

	/* Wait for all overflow buffers to be unlinked */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&rxc->oflow_bufs_linked) ||
		 ofi_atomic_get32(&rxc->ux_sink_linked));

	if (ofi_atomic_get32(&rxc->oflow_bufs_in_use))
		CXIP_LOG_ERROR("Leaked %d overflow buffers\n",
			       ofi_atomic_get32(&rxc->oflow_bufs_in_use));
}

/*
 * rdzv_recv_req_event() - Count a rendezvous event.
 *
 * Call for each target rendezvous event generated on a user receive buffer.
 * After three events, a rendezvous receive is complete. The three events could
 * be either:
 *   -Put, Rendezvous, Reply -- or
 *   -Put Overflow, Rendezvous, Reply
 *
 * In either case, the events could be generated in any order. As soon as three
 * events are processed, the request is complete.
 */
static void rdzv_recv_req_event(struct cxip_req *req)
{
	if (++req->recv.rdzv_events == 3)
		recv_req_complete(req);
}

/*
 * cxip_recv_rdzv_cb() - Progress rendezvous receive events.
 *
 * Handle rendezvous target events. All target events which are related to an
 * offloaded rendezvous Put operation have the rendezvous field set.
 *
 * Note that Reply events that were generated from a SW-issued Get will not
 * have the rendezvous bit set.
 */
static int cxip_recv_rdzv_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	struct cxip_ux_send *ux_send;
	struct cxip_oflow_buf *oflow_buf;
	void *oflow_va;
	union cxip_match_bits mb;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		/* TODO Handle unlink errors. */
		assert(cxi_event_rc(event) == C_RC_OK);
		return FI_SUCCESS;
	case C_EVENT_SEND:
		/* TODO Handle Send event errors. */
		assert(cxi_event_rc(event) == C_RC_OK);
		return FI_SUCCESS;
	case C_EVENT_PUT_OVERFLOW:
		/* We matched an unexpected header */

		fastlock_acquire(&rxc->rx_lock);

		/* Check for a previously received unexpected Put event */
		ux_send = match_ux_send(rxc, event);
		if (!ux_send) {
			/* An unexpected Put event is pending.  Link this
			 * request to the pending list for lookup when the
			 * event arrives.
			 *
			 * Store initiator and rendezvous ID to match against
			 * future events.
			 */
			mb.raw = event->tgt_long.match_bits;
			req->recv.initiator =
				event->tgt_long.initiator.initiator.process;
			req->recv.rdzv_id =
				RDZV_ID(event->tgt_long.rendezvous_id,
					mb.rdzv_id_lo);

			dlist_insert_tail(&req->recv.ux_entry,
					  &rxc->ux_rdzv_recvs);

			CXIP_LOG_DBG("Queued recv req, data: 0x%lx\n",
				     req->recv.start);

			fastlock_release(&rxc->rx_lock);

			/* Update request fields. */
			recv_req_put_event(req, event);

			/* Count the rendezvous event. */
			rdzv_recv_req_event(req);

			return FI_SUCCESS;
		}

		CXIP_LOG_DBG("Matched ux_send: %p\n", ux_send);

		oflow_buf = ux_send->req->oflow.oflow_buf;

		/* Update request fields. */
		recv_req_put_event(req, event);

		/* Copy data out of overflow buffer. */
		oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md->md,
						  ux_send->start);
		memcpy(req->recv.recv_buf, oflow_va, event->tgt_long.mlength);
		oflow_req_put_bytes(ux_send->req, event->tgt_long.mlength);

		dlist_remove(&ux_send->ux_entry);
		free(ux_send);

		fastlock_release(&rxc->rx_lock);

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);

		return FI_SUCCESS;
	case C_EVENT_PUT:
		/* Eager data was delivered directly to the user buffer. */
		recv_req_put_event(req, event);

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	case C_EVENT_RENDEZVOUS:
		if (!event->tgt_long.get_issued) {
			int ret = issue_rdzv_get(req, event);
			if (ret != FI_SUCCESS)
				return -FI_EAGAIN;
			CXIP_LOG_DBG("Software issued Get, req: %p\n", req);
		}

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	case C_EVENT_REPLY:
		/* Rendezvous Get completed. Complete the request. */
		req->recv.rc = cxi_init_event_rc(event);

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	default:
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
	}

	return FI_SUCCESS;
}

/*
 * cxip_recv_cb() - Process a user receive buffer event.
 *
 * A user receive buffer is described by an LE linked to the Priority list.
 * Link, Unlink, Put, Put Overflow, and Reply events are expected from a user
 * receive buffer.
 *
 * A Link event indicates that a new user buffer has been linked to the
 * priority list. Successful Link events may be suppressed.
 *
 * An Unlink event indicates that a user buffer has been unlinked. Normally, a
 * receive is used once and unlinked when it is matched with a Send. In this
 * case, a successful Unlink event may be suppressed.
 *
 * For expected, eager Sends, a Put will be matched to a user receive buffer by
 * the NIC. Send data is copied directly to the user buffer. A Put event is
 * generated describing the match.
 *
 * For unexpected, eager Sends, a Put will first match a buffer in the Overflow
 * list. See cxip_oflow_cb() for details on Overflow event handling. Once a
 * matching user receive buffer is appended to the Priority list, a Put
 * Overflow event is generated. Put and Put Overflow events for an unexpected,
 * eager Send must be correlated. These events may arrive in any order. Once
 * both events are accounted, data is copied from the Overflow buffer to the
 * user receive buffer.
 *
 * Unexpected, eager Sends that are longer than the eager threshold have their
 * data truncated to zero. This is to avoid long messages consuming too much
 * Overflow buffer space at the target. Once a match is made with a user
 * receive buffer, data is re-read from the initiator using a Get.
 *
 * Rendezvous receive events are handled by cxip_recv_rdzv_cb().
 */
static int cxip_recv_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	struct cxip_rxc *rxc = req->recv.rxc;
	struct cxip_ux_send *ux_send;
	struct cxip_oflow_buf *oflow_buf;
	void *oflow_va;

	/* All events related to an offloaded rendezvous receive will be
	 * handled by cxip_recv_rdzv_cb(). Those events are identified by the
	 * event rendezvous field. One exception is a Reply event generated
	 * from a SW-issued Get. When such an event is generated, the request
	 * will have already processed a Rendezvous event. If the rendezvous
	 * field is not set, but the rdzv_events count is elevated, this must
	 * be a SW-issued Reply event.
	 */
	if (event->tgt_long.rendezvous || req->recv.rdzv_events)
		return cxip_recv_rdzv_cb(req, event);

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		if (!event->tgt_long.auto_unlinked) {
			req->recv.rc = C_RC_CANCELED;
			recv_req_complete(req);
		} else {
			assert(cxi_event_rc(event) == C_RC_OK);
		}
		return FI_SUCCESS;
	case C_EVENT_SEND:
		/* TODO Handle Send event errors. */
		assert(cxi_event_rc(event) == C_RC_OK);
		return FI_SUCCESS;
	case C_EVENT_PUT_OVERFLOW:
		/* We matched an unexpected header */

		fastlock_acquire(&rxc->rx_lock);

		/* Check for a previously received unexpected Put event */
		ux_send = match_ux_send(rxc, event);
		if (!ux_send) {
			/* An unexpected Put event is pending. Link this
			 * request to the pending list for lookup when the
			 * event arrives. Store start address for matching.
			 */
			req->recv.start = event->tgt_long.start;

			dlist_insert_tail(&req->recv.ux_entry, &rxc->ux_recvs);

			CXIP_LOG_DBG("Queued recv req, data: 0x%lx\n",
				     req->recv.start);

			fastlock_release(&rxc->rx_lock);

			/* Update request fields. */
			recv_req_put_event(req, event);

			return FI_SUCCESS;
		}

		/* A matching unexpected-Put event arrived earlier. */

		CXIP_LOG_DBG("Matched ux_send: %p\n", ux_send);

		/* Update request fields. */
		recv_req_put_event(req, event);

		oflow_buf = ux_send->req->oflow.oflow_buf;

		if (oflow_buf->type == OFLOW_BUF_SINK) {
			/* For unexpected, long, eager messages, issue a Get to
			 * retrieve data from the initiator.
			 */
			ret = issue_rdzv_get(req, event);
			if (ret == FI_SUCCESS) {
				dlist_remove(&ux_send->ux_entry);
				free(ux_send);

				CXIP_LOG_DBG("Issued Get, req: %p\n", req);
			}

			fastlock_release(&rxc->rx_lock);

			return ret;
		}

		/* Copy data out of overflow buffer. */
		oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md->md,
				event->tgt_long.start);
		memcpy(req->recv.recv_buf, oflow_va,
		       req->recv.mlength);
		oflow_req_put_bytes(ux_send->req, req->recv.mlength);

		dlist_remove(&ux_send->ux_entry);
		free(ux_send);

		fastlock_release(&rxc->rx_lock);

		/* Complete receive request. */
		recv_req_complete(req);
		return FI_SUCCESS;
	case C_EVENT_PUT:
		/* Data was delivered directly to the user buffer. Complete the
		 * request.
		 */
		recv_req_put_event(req, event);

		/* Complete receive request. */
		recv_req_complete(req);
		return FI_SUCCESS;
	case C_EVENT_REPLY:
		/* Long-send Get completed. Complete the request. */
		req->recv.rc = cxi_init_event_rc(event);

		/* Complete receive request. */
		recv_req_complete(req);
		return FI_SUCCESS;
	default:
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
	}

	return FI_SUCCESS;
}

int cxip_msg_recv_cancel(struct cxip_req *req)
{
	int ret;
	struct cxip_rxc *rxc = req->recv.rxc;

	ret = issue_unlink_le(rxc->rx_pte->pte, C_PTL_LIST_PRIORITY,
			      req->req_id, rxc->rx_cmdq);
	if (ret == FI_SUCCESS)
		req->recv.canceled = true;

	return ret;
}

/*
 * _cxip_recv() - Common message receive function. Used for tagged and untagged
 * sends of all sizes.
 */
static ssize_t _cxip_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			  fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			  void *context, uint64_t flags, bool tagged)
{
	struct cxip_ep *cxi_ep;
	struct cxip_rxc *rxc;
	struct cxip_domain *dom;
	int ret;
	struct cxip_md *recv_md;
	struct cxip_req *req;
	struct cxip_addr caddr;
	uint32_t match_id;
	uint32_t pid_bits;
	union cxip_match_bits mb = {};
	union cxip_match_bits ib = { .sink = ~0, .rdzv_id_lo = ~0 };

	if (!ep || !buf)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a RX
	 * context), or a RX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		rxc = cxi_ep->ep_obj->rxcs[0];
		break;

	case FI_CLASS_RX_CTX:
		rxc = container_of(ep, struct cxip_rxc, ctx);
		break;

	default:
		CXIP_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	if (!rxc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_recv_allowed(rxc->attr.caps))
		return -FI_ENOPROTOOPT;

	dom = rxc->domain;

	/* If FI_DIRECTED_RECV and a src_addr is specified, encode the address
	 * in the LE for matching. If application AVs are symmetric, use
	 * logical FI address for matching. Otherwise, use physical address.
	 */
	pid_bits = dom->dev_if->if_dev->info.pid_bits;
	if (rxc->attr.caps & FI_DIRECTED_RECV &&
	    src_addr != FI_ADDR_UNSPEC) {
		if (!rxc->ep_obj->rdzv_offload &&
		    rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
			match_id = CXI_MATCH_ID(pid_bits, C_PID_ANY, src_addr);
		} else {
			ret = _cxip_av_lookup(rxc->ep_obj->av, src_addr,
					      &caddr);
			if (ret != FI_SUCCESS) {
				CXIP_LOG_DBG("Failed to look up FI addr: %d\n",
					     ret);
				return -FI_EINVAL;
			}

			match_id = CXI_MATCH_ID(pid_bits, caddr.pid, caddr.nic);
		}
	} else {
		match_id = CXI_MATCH_ID(pid_bits, C_PID_ANY, C_NID_ANY);
	}

	/* Map local buffer */
	ret = cxip_map(dom, (void *)buf, len, &recv_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map recv buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, rxc);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto recv_unmap;
	}

	/* req->data_len, req->tag must be set later.  req->buf and req->data
	 * may be overwritten later.
	 */
	req->context = (uint64_t)context;

	req->flags = FI_RECV;
	if (tagged) {
		req->flags |= FI_TAGGED;
		mb.tagged = 1;
		mb.tag = tag;
		ib.tag = ignore;
	} else {
		req->flags |= FI_MSG;
	}

	req->buf = 0;
	req->data = 0;
	req->cb = cxip_recv_cb;

	req->recv.rxc = rxc;
	req->recv.recv_buf = buf;
	req->recv.recv_md = recv_md;
	req->recv.rlength = len;

	/* Count Put, Rendezvous, and Reply events during offloaded RPut. */
	req->recv.rdzv_events = 0;
	req->recv.put_event = false;

	/* Issue Append command */
	ret = issue_append_le(rxc->rx_pte->pte, buf, len, recv_md,
			      C_PTL_LIST_PRIORITY, req->req_id, mb.raw, ib.raw,
			      match_id, 0, false, true, false, false, true,
			      false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto req_free;
	}

	ofi_atomic_inc32(&rxc->orx_reqs);

	CXIP_LOG_DBG("req: %p buf: %p len: %lu src_addr: %ld tag(%c): 0x%lx ignore: 0x%lx context: %p\n",
		     req, buf, len, src_addr, tagged ? '*' : '-', tag, ignore,
		     context);

	return FI_SUCCESS;

req_free:
	cxip_cq_req_free(req);
recv_unmap:
	cxip_unmap(recv_md);

	return ret;
}

/*
 * cxip_txc_fi_addr() - Return the FI address of the TXC.
 */
static fi_addr_t _txc_fi_addr(struct cxip_txc *txc)
{
	if (txc->ep_obj->fi_addr == FI_ADDR_NOTAVAIL) {
		txc->ep_obj->fi_addr =
				_cxip_av_reverse_lookup(
						txc->ep_obj->av,
						txc->ep_obj->src_addr.nic,
						txc->ep_obj->src_addr.pid);
		CXIP_LOG_DBG("Found EP FI Addr: %lu\n", txc->ep_obj->fi_addr);
	}

	return txc->ep_obj->fi_addr;
}

/*
 * cxip_msg_match_id() - Return the TXC's initiator address used to transmit a
 * message.
 *
 * By default, the physical address of the TXC is returned. This address is
 * sent along with message data and is used for source address matching at the
 * target. When the target receives a message, the physical ID is translated to
 * a logical FI address. Translation adds overhead to the receive path.
 *
 * As an optimization, if rendezvous offload is not being used and the process
 * is part of a job with symmetric AVs, a logical FI address is returned. This
 * way, there is no source address translation overhead involved in the
 * receive.
 */
static uint32_t cxip_msg_match_id(struct cxip_txc *txc)
{
	int pid_bits = txc->domain->dev_if->if_dev->info.pid_bits;

	if (!txc->ep_obj->rdzv_offload &&
	    txc->ep_obj->av->attr.flags & FI_SYMMETRIC)
		return CXI_MATCH_ID(pid_bits, txc->ep_obj->src_addr.pid,
				    _txc_fi_addr(txc));

	return CXI_MATCH_ID(pid_bits, txc->ep_obj->src_addr.pid,
			    txc->ep_obj->src_addr.nic);
}

/*
 * cxip_inject_cb() - Message inject event callback.
 */
static int cxip_inject_cb(struct cxip_req *req, const union c_event *event)
{
	return cxip_cq_req_error(req, 0, FI_EIO, cxi_event_rc(event), NULL, 0);
}

/*
 * cxip_inject_req() - Return request state associated with all inject
 * transactions on the transmit context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_inject_req(struct cxip_txc *txc)
{
	if (!txc->inject_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_inject_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_MSG | FI_SEND;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->addr = FI_ADDR_UNSPEC;

		txc->inject_req = req;
	}

	return txc->inject_req;
}

/*
 * cxip_tinject_req() - Return request state associated with all tagged inject
 * transactions on the transmit context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_tinject_req(struct cxip_txc *txc)
{
	if (!txc->tinject_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_inject_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_TAGGED | FI_SEND;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->addr = FI_ADDR_UNSPEC;

		txc->tinject_req = req;
	}

	return txc->tinject_req;
}

/*
 * report_send_completion() - Report the completion of a send operation.
 */
static void report_send_completion(struct cxip_req *req, bool sw_cntr)
{
	int ret;
	int success_event = (req->flags & FI_COMPLETION);

	req->flags &= (FI_MSG | FI_TAGGED | FI_SEND);

	if (req->send.rc == C_RC_OK) {
		CXIP_LOG_DBG("Request success: %p\n", req);

		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report completion: %d\n",
					       ret);
		}

		if (sw_cntr && req->send.txc->send_cntr) {
			ret = cxip_cntr_mod(req->send.txc->send_cntr, 1, false,
					    false);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	} else {
		CXIP_LOG_DBG("Request error: %p (err: %d, %d)\n", req, FI_EIO,
			     req->send.rc);

		ret = cxip_cq_req_error(req, 0, FI_EIO, req->send.rc, NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);

		if (sw_cntr && req->send.txc->send_cntr) {
			ret = cxip_cntr_mod(req->send.txc->send_cntr, 1, false,
					    true);
			if (ret)
				CXIP_LOG_ERROR("cxip_cntr_mod returned: %d\n",
					       ret);
		}
	}
}

/*
 * rdzv_send_req_free() - Clean up a long send request.
 */
static void rdzv_send_req_free(struct cxip_req *req)
{
	cxip_rdzv_id_free(req->send.txc->ep_obj, req->send.rdzv_id);

	cxip_unmap(req->send.send_md);

	ofi_atomic_dec32(&req->send.txc->otx_reqs);

	cxip_cq_req_free(req);
}

/*
 * long_send_req_event() - Count a long send event.
 *
 * Call for each initiator send event generated. After Ack, Unlink, and Get
 * events are generated, the send is complete. The events could be generated in
 * any order. As soon as three events are processed, the request is complete.
 */
static void long_send_req_event(struct cxip_req *req)
{
	if (++req->send.long_send_events == 3) {
		report_send_completion(req, true);
		rdzv_send_req_free(req);
	}
}

/*
 * cxip_send_long_cb() - Long send callback.
 *
 * Progress a long send operation to completion.
 */
static int cxip_send_long_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	struct cxip_txc *txc = req->send.txc;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Link error: %p, rc: %d\n",
				       req, event_rc);

			req->send.rc = event_rc;
			report_send_completion(req, true);
			rdzv_send_req_free(req);
			return FI_SUCCESS;
		}

		/* The send buffer LE is linked. Perform Put of the payload. */
		fastlock_acquire(&txc->tx_cmdq->lock);

		ret = cxi_cq_emit_dma_f(txc->tx_cmdq->dev_cmdq,
					&req->send.cmd);
		if (ret) {
			CXIP_LOG_ERROR("Failed to enqueue Put: %d\n", ret);
			fastlock_release(&txc->tx_cmdq->lock);
			return -FI_EAGAIN;
		}

		cxi_cq_ring(txc->tx_cmdq->dev_cmdq);

		fastlock_release(&txc->tx_cmdq->lock);

		CXIP_LOG_DBG("Enqueued Put: %p\n", req);
		return FI_SUCCESS;
	case C_EVENT_ACK:
		/* The source Put completed. */
		event_rc = cxi_init_event_rc(event);
		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Ack error: %p rc: %d\n",
				       req, event_rc);

			ret = issue_unlink_le_f(txc->rdzv_pte->pte,
						C_PTL_LIST_PRIORITY,
						req->req_id,
						txc->rx_cmdq);
			if (ret) {
				CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n",
					       ret);
				return -FI_EAGAIN;
			}

			/* Save RC for when the Unlink is complete. */
			req->send.rc = event_rc;
			return FI_SUCCESS;
		}

		CXIP_LOG_DBG("Put Acked (%s): %p\n",
			     cxi_ptl_list_to_str(event->init_short.ptl_list),
			     req);

		if (!txc->ep_obj->rdzv_offload &&
		    event->init_short.ptl_list == C_PTL_LIST_PRIORITY) {
			/* No Get is expected when a long eager Send matches in
			 * the Priority list. Unlink the source LE manually.
			 */
			ret = issue_unlink_le_f(txc->rdzv_pte->pte,
						C_PTL_LIST_PRIORITY,
						req->req_id,
						txc->rx_cmdq);
			if (ret) {
				CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n",
					       ret);
				return -FI_EAGAIN;
			}
			req->send.rc = event_rc;
			return FI_SUCCESS;
		}

		/* Wait for the source buffer LE to be unlinked. */
		long_send_req_event(req);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK) {
			/* The LE was unlinked unexpectedly. */
			CXIP_LOG_ERROR("Unlink error: %p rc: %d\n",
				       req, event_rc);

			req->send.rc = event_rc;
			report_send_completion(req, true);
			rdzv_send_req_free(req);
		} else if (!event->tgt_long.auto_unlinked) {
			/* Either the Put failed or a long Send matched in the
			 * Priority list. In either case, no Get event is
			 * expected, complete request. Use RC from the Ack.
			 */
			CXIP_LOG_DBG("Manually unlinked:  %p\n", req);

			report_send_completion(req, true);
			rdzv_send_req_free(req);
		} else {
			/* The source buffer was unlinked by a Get. */
			CXIP_LOG_DBG("Auto-unlinked: %p\n", req);
			long_send_req_event(req);
		}

		return FI_SUCCESS;
	case C_EVENT_GET:
		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK)
			CXIP_LOG_ERROR("Get error: %p rc: %d\n",
				       req, event_rc);
		else
			CXIP_LOG_DBG("Get received: %p rc: %d\n",
				     req, event_rc);

		req->send.rc = event_rc;
		long_send_req_event(req);

		return FI_SUCCESS;
	default:
		CXIP_LOG_ERROR("Unexpected event received: %s\n",
			       cxi_event_to_str(event));
		return FI_SUCCESS;
	}
}

/*
 * _cxip_send_long() - Initiate a long send operation.
 *
 * There are two long send protocols implemented: an eager (long) protocol and
 * an offloaded rendezvous protocol.
 *
 * The eager (long) protocol works as follows:
 *
 * 1. The Initiator prepares an LE describing the source buffer.
 * 2. The Initiator performs a Put of the entire source buffer.
 * 3. An Ack event is generated indicating the Put completed. The Ack indicates
 *    whether it matched in the Priority or Overflow list at the target.
 * 4a. If the Put matched in the Priority list, the entire payload was copied
 *     directly to a receive buffer at the target. The operation is complete.
 *     The source buffer LE was unused.
 * 4b. If the Put matched in the Overflow list, the payload was truncated to
 *     zero. The Target receives events describing the Put attempt.
 * 5b. The Target performs a Get of the entire source buffer using the source
 *     buffer LE.
 *
 * The rendezvous protocol works as follows:
 *
 * 1. The Initiator prepares an LE describing the source buffer.
 * 2. The Initiator performs a Rendezvous Put command which includes a portion
 *    of the source buffer data.
 * 3. Once the Put is matched to a user receive buffer (in the Priority list),
 *    a Get of the remaining source data is performed.
 */
static ssize_t _cxip_send_long(struct cxip_txc *txc, const void *buf,
			       size_t len, void *desc, fi_addr_t dest_addr,
			       uint64_t tag, void *context, uint64_t flags,
			       bool tagged)
{
	struct cxip_domain *dom;
	struct cxip_md *send_md;
	struct cxip_req *req;
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;
	uint64_t rx_id;
	struct c_full_dma_cmd cmd = {};
	union cxip_match_bits put_mb = {};
	union cxip_match_bits le_mb = {};
	union cxip_match_bits le_ib = { .rdzv_id_lo = ~0 }; /* inverted */
	int rdzv_id;
	int ret;

	dom = txc->domain;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer */
	ret = cxip_map(dom, (void *)buf, len, &send_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map send buffer: %d\n", ret);
		return ret;
	}

	/* Allocate and populate request */
	req = cxip_cq_req_alloc(txc->send_cq, true, txc);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto err_unmap;
	}

	req->context = (uint64_t)context;
	req->flags = FI_SEND | (flags & FI_COMPLETION);

	if (tagged) {
		req->flags |= FI_TAGGED;
		put_mb.tagged = 1;
		put_mb.tag = tag;
	} else {
		req->flags |= FI_MSG;
	}

	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;

	req->send.txc = txc;
	req->send.buf = (void *)buf;
	req->send.send_md = send_md;
	req->send.length = len;
	req->cb = cxip_send_long_cb;

	req->send.long_send_events = 0;

	/* Calculate DFA */
	rx_id = CXIP_AV_ADDR_RXC(txc->ep_obj->av, dest_addr);
	pid_granule = dom->dev_if->if_dev->info.pid_granule;
	pid_idx = CXIP_RXC_TO_IDX(rx_id);
	cxi_build_dfa(caddr.nic, caddr.pid, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	/* Allocate rendezvous ID */
	rdzv_id = cxip_rdzv_id_alloc(txc->ep_obj);
	if (rdzv_id < 0)
		goto err_req_free;

	req->send.rdzv_id = rdzv_id;
	put_mb.rdzv_id_lo = RDZV_ID_LO(rdzv_id);
	le_mb.rdzv_id_lo = RDZV_ID_LO(rdzv_id);

	/* Build Put command descriptor */
	cmd.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.index_ext = idx_ext;
	cmd.lac = send_md->md->lac;
	cmd.event_send_disable = 1;
	cmd.restricted = 0;
	cmd.dfa = dfa;
	cmd.remote_offset = 0;
	cmd.local_addr = CXI_VA_TO_IOVA(send_md->md, buf);
	cmd.request_len = len;
	cmd.eq = txc->send_cq->evtq->eqn;
	cmd.user_ptr = (uint64_t)req;
	cmd.initiator = cxip_msg_match_id(txc);

	if (txc->ep_obj->rdzv_offload) {
		cmd.command.opcode = C_CMD_RENDEZVOUS_PUT;
		cmd.eager_length = txc->eager_threshold;

		/* Use rendezvous ID extension */
		cmd.rendezvous_id = RDZV_ID_HI(rdzv_id);
		le_mb.rdzv_id_hi = RDZV_ID_HI(rdzv_id);
		le_ib.rdzv_id_hi = ~0;
	} else {
		cmd.command.opcode = C_CMD_PUT;

		/* Ensure the full rdzv_id fits in match bits */
		assert(rdzv_id < (1 << RDZV_ID_LO_WIDTH));

		/* Match sink buffer */
		put_mb.sink = 1;
	}

	cmd.match_bits = put_mb.raw;

	/* Store DMA command for use once the source data becomes visible */
	req->send.cmd = cmd;

	ret = issue_append_le(txc->rdzv_pte->pte, req->send.buf,
			      req->send.length, req->send.send_md,
			      C_PTL_LIST_PRIORITY, req->req_id,
			      le_mb.raw, ~le_ib.raw, CXI_MATCH_ID_ANY, 0,
			      false, true, false, true, false, true,
			      txc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed append source buffer: %d\n", ret);
		goto err_id_free;
	}

	ofi_atomic_inc32(&txc->otx_reqs);

	CXIP_LOG_DBG("req: %p buf: %p len: %lu dest_addr: %ld tag(%c): 0x%lx context %p\n",
		     req, buf, len, dest_addr, tagged ? '*' : '-', tag,
		     context);

	return FI_SUCCESS;

err_id_free:
	cxip_rdzv_id_free(txc->ep_obj, rdzv_id);
err_req_free:
	cxip_cq_req_free(req);
err_unmap:
	cxip_unmap(send_md);

	return FI_EAGAIN;
}

/*
 * cxip_send_eager_cb() - Eager send callback. Used for both tagged and
 * untagged messages.
 */
static int cxip_send_eager_cb(struct cxip_req *req,
			      const union c_event *event)
{
	/* IDCs don't have an MD */
	if (req->send.send_md)
		cxip_unmap(req->send.send_md);

	req->send.rc = cxi_init_event_rc(event);
	report_send_completion(req, false);
	ofi_atomic_dec32(&req->send.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
}

/*
 * _cxip_send_eager() - Enqueue eager send command.
 */
static ssize_t _cxip_send_eager(struct cxip_txc *txc, const void *buf,
				size_t len, void *desc, fi_addr_t dest_addr,
				uint64_t tag, void *context, uint64_t flags,
				bool tagged)
{
	struct cxip_domain *dom;
	struct cxip_md *send_md = NULL;
	struct cxip_req *req = NULL;
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;
	uint64_t rx_id;
	union cxip_match_bits mb = {};
	int idc;
	int ret;

	/* Always use IDCs when the payload fits */
	idc = (len <= C_MAX_IDC_PAYLOAD_UNR);

	dom = txc->domain;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer */
	if (!idc) {
		ret = cxip_map(dom, (void *)buf, len, &send_md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map send buffer: %d\n", ret);
			return ret;
		}
	}

	/* DMA commands must always be tracked. IDCs must be tracked if the
	 * user requested a completion event.
	 */
	if (!idc || (flags & FI_COMPLETION)) {
		req = cxip_cq_req_alloc(txc->send_cq, false, txc);
		if (!req) {
			CXIP_LOG_DBG("Failed to allocate request\n");
			ret = -FI_ENOMEM;
			goto err_unmap;
		}

		req->context = (uint64_t)context;
		req->flags = FI_SEND | (flags & FI_COMPLETION);

		if (tagged)
			req->flags |= FI_TAGGED;
		else
			req->flags |= FI_MSG;

		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;

		req->send.txc = txc;
		req->send.buf = (void *)buf;
		req->send.send_md = send_md;
		req->send.length = len;
		req->cb = cxip_send_eager_cb;

		req->send.long_send_events = 0;
	}

	/* Build Put command descriptor */
	rx_id = CXIP_AV_ADDR_RXC(txc->ep_obj->av, dest_addr);
	pid_granule = dom->dev_if->if_dev->info.pid_granule;
	pid_idx = CXIP_RXC_TO_IDX(rx_id);
	cxi_build_dfa(caddr.nic, caddr.pid, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	/* Build match bits */
	if (tagged) {
		mb.tagged = 1;
		mb.tag = tag;
	}

	fastlock_acquire(&txc->tx_cmdq->lock);

	if (idc) {
		union c_cmdu cmd = {};

		cmd.c_state.event_send_disable = 1;
		cmd.c_state.index_ext = idx_ext;
		cmd.c_state.eq = txc->send_cq->evtq->eqn;
		cmd.c_state.initiator = cxip_msg_match_id(txc);

		if (!req)
			cmd.c_state.event_success_disable = 1;

		if (txc->send_cntr) {
			cmd.c_state.event_ct_ack = 1;
			cmd.c_state.ct = txc->send_cntr->ct->ctn;
		}

		if (memcmp(&txc->tx_cmdq->c_state, &cmd.c_state,
			   sizeof(cmd.c_state))) {
			/* Update TXQ C_STATE */
			txc->tx_cmdq->c_state = cmd.c_state;

			ret = cxi_cq_emit_c_state(txc->tx_cmdq->dev_cmdq,
						  &cmd.c_state);
			if (ret) {
				CXIP_LOG_DBG("Failed to issue C_STATE command: %d\n",
					     ret);

				/* Return error according to Domain Resource
				 * Management
				 */
				ret = -FI_EAGAIN;
				goto err_unlock;
			}

			CXIP_LOG_DBG("Updated C_STATE: %p\n", req);
		}

		memset(&cmd.idc_msg, 0, sizeof(cmd.idc_msg));
		cmd.idc_msg.dfa = dfa;
		cmd.idc_msg.match_bits = mb.raw;

		if (req) {
			cmd.idc_msg.user_ptr = (uint64_t)req;
		} else {
			void *inject_req;

			if (tagged)
				inject_req = cxip_tinject_req(txc);
			else
				inject_req = cxip_inject_req(txc);

			if (!inject_req) {
				ret = -FI_ENOMEM;
				goto err_unlock;
			}

			cmd.idc_msg.user_ptr = (uint64_t)inject_req;
		}

		ret = cxi_cq_emit_idc_msg(txc->tx_cmdq->dev_cmdq, &cmd.idc_msg,
					  buf, len);
		if (ret) {
			CXIP_LOG_DBG("Failed to write IDC: %d\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	} else {
		struct c_full_dma_cmd cmd = {};

		cmd.command.cmd_type = C_CMD_TYPE_DMA;
		cmd.command.opcode = C_CMD_PUT;
		cmd.index_ext = idx_ext;
		cmd.lac = send_md->md->lac;
		cmd.event_send_disable = 1;
		cmd.restricted = 0;
		cmd.dfa = dfa;
		cmd.remote_offset = 0;
		cmd.local_addr = CXI_VA_TO_IOVA(send_md->md, buf);
		cmd.request_len = len;
		cmd.eq = txc->send_cq->evtq->eqn;
		cmd.user_ptr = (uint64_t)req;
		cmd.initiator = cxip_msg_match_id(txc);
		cmd.match_bits = mb.raw;

		if (txc->send_cntr) {
			cmd.event_ct_ack = 1;
			cmd.ct = txc->send_cntr->ct->ctn;
		}

		/* Issue Eager Put command */
		ret = cxi_cq_emit_dma(txc->tx_cmdq->dev_cmdq, &cmd);
		if (ret) {
			CXIP_LOG_DBG("Failed to write DMA command: %d\n", ret);

			/* Return error according to Domain Resource Mgmt */
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	}

	cxi_cq_ring(txc->tx_cmdq->dev_cmdq);

	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	fastlock_release(&txc->tx_cmdq->lock);

	CXIP_LOG_DBG("req: %p buf: %p len: %lu dest_addr: %ld tag(%c): 0x%lx context %p\n",
		     req, buf, len, dest_addr, tagged ? '*' : '-', tag,
		     context);

	return FI_SUCCESS;

err_unlock:
	fastlock_release(&txc->tx_cmdq->lock);
	if (req)
		cxip_cq_req_free(req);
err_unmap:
	if (!idc)
		cxip_unmap(send_md);

	return ret;
}

/*
 * _cxip_send() - Common message send function. Used for tagged and untagged
 * sends of all sizes.
 */
static ssize_t _cxip_send(struct cxip_txc *txc, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context, uint64_t flags, bool tagged)
{
	int ret;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_send_allowed(txc->attr.caps))
		return -FI_ENOPROTOOPT;

	if (!buf)
		return -FI_EINVAL;

	if (len > txc->eager_threshold)
		ret = _cxip_send_long(txc, buf, len, desc, dest_addr, tag,
				      context, flags, tagged);
	else
		ret = _cxip_send_eager(txc, buf, len, desc, dest_addr, tag,
				       context, flags, tagged);

	return ret;
}

/*
 * Libfabric APIs
 */

static ssize_t cxip_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			  fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			  void *context)
{
	return _cxip_recv(ep, buf, len, desc, src_addr, tag, ignore, context,
			  0, true);
}

static ssize_t cxip_trecvv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t src_addr,
			   uint64_t tag, uint64_t ignore, void *context)
{
	if (!iov || count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  src_addr, tag, ignore, context, 0, true);
}

static ssize_t cxip_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			     uint64_t flags)
{
	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->tag, msg->ignore, msg->context, 0, true);
}

static ssize_t cxip_tsend(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_send(txc, buf, len, desc, dest_addr, tag, context, flags,
			  true);
}

static ssize_t cxip_tsendv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t dest_addr,
			   uint64_t tag, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_send(txc, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  dest_addr, tag, context, flags, true);
}

#define CXIP_TSENDMSG_ALLOWED_FLAGS (FI_INJECT | FI_COMPLETION)

static ssize_t cxip_tsendmsg(struct fid_ep *ep,
			     const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_TSENDMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return _cxip_send(txc, msg->msg_iov[0].iov_base,
			  msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->tag, msg->context, flags, true);
}

static ssize_t cxip_tinject(struct fid_ep *ep, const void *buf, size_t len,
			    fi_addr_t dest_addr, uint64_t tag)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return _cxip_send(txc, buf, len, NULL, dest_addr, tag, NULL, FI_INJECT,
			  true);
}

struct fi_ops_tagged cxip_ep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = cxip_trecv,
	.recvv = cxip_trecvv,
	.recvmsg = cxip_trecvmsg,
	.send = cxip_tsend,
	.sendv = cxip_tsendv,
	.sendmsg = cxip_tsendmsg,
	.inject = cxip_tinject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

static ssize_t cxip_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, void *context)
{
	return _cxip_recv(ep, buf, len, desc, src_addr, 0, 0, context, 0,
			  false);
}

static ssize_t cxip_recvv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  void *context)
{
	if (!iov || count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  src_addr, 0, 0, context, 0, false);
}

static ssize_t cxip_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr, 0, 0,
			  msg->context, flags, false);
}

static ssize_t cxip_send(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_send(txc, buf, len, desc, dest_addr, 0, context, flags,
			  false);
}

static ssize_t cxip_sendv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_send(txc, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL, dest_addr, 0, context, flags,
			  false);
}

#define CXIP_SENDMSG_ALLOWED_FLAGS (FI_INJECT | FI_COMPLETION)

static ssize_t cxip_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_SENDMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return _cxip_send(txc, msg->msg_iov[0].iov_base,
			  msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr, 0,
			  msg->context, flags, false);
}

static ssize_t cxip_inject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return _cxip_send(txc, buf, len, NULL, dest_addr, 0, NULL, FI_INJECT,
			  false);
}

struct fi_ops_msg cxip_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = cxip_recv,
	.recvv = cxip_recvv,
	.recvmsg = cxip_recvmsg,
	.send = cxip_send,
	.sendv = cxip_sendv,
	.sendmsg = cxip_sendmsg,
	.inject = cxip_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

