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

/* Append a buffer to a PtlTE. */
/* Caller must hold cmdq lock */
static int issue_append_le(struct cxil_pte *pte, const void *buf, size_t len,
			   struct cxi_md *md, enum c_ptl_list list,
			   uint32_t buffer_id, uint64_t match_bits,
			   uint64_t ignore_bits, uint32_t match_id,
			   uint64_t min_free, bool event_success_disable,
			   bool use_once, bool manage_local, bool no_truncate,
			   bool op_put, bool op_get, struct cxi_cmdq *cmdq)
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
	cmd.target.lac          = md->lac;
	cmd.target.start        = CXI_VA_TO_IOVA(md, buf);
	cmd.target.length       = len;
	cmd.target.event_success_disable = event_success_disable ? 1 : 0;
	cmd.target.use_once     = use_once ? 1 : 0;
	cmd.target.match_bits   = match_bits;
	cmd.target.ignore_bits  = ignore_bits;
	cmd.target.match_id     = match_id;
	cmd.target.min_free     = min_free;

	rc = cxi_cq_emit_target(cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq);

	return FI_SUCCESS;
}

/* Unlink a buffer to a PtlTE. */
/* Caller must hold cmdq lock */
static int issue_unlink_le(struct cxil_pte *pte, enum c_ptl_list list,
			   int buffer_id, struct cxi_cmdq *cmdq)
{
	union c_cmdu cmd = {};
	int rc;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = list;
	cmd.target.ptlte_index  = pte->ptn;
	cmd.target.buffer_id = buffer_id;

	rc = cxi_cq_emit_target(cmdq, &cmd);
	if (rc) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", rc);

		/* Return error according to Domain Resource Management */
		return -FI_EAGAIN;
	}

	cxi_cq_ring(cmdq);

	return FI_SUCCESS;
}

/* Caller must hold rxc->lock */
static struct cxip_ux_send *
match_ux_send(struct cxip_rx_ctx *rxc, const union c_event *event)
{
	struct cxip_ux_send *ux_send;

	/* Look for a previously received unexpected Put event with matching
	 * start pointer.
	 *
	 * TODO this assumes all overflow buffers use the same AC so all start
	 * pointers are unique.
	 */
	dlist_foreach_container(&rxc->ux_sends, struct cxip_ux_send, ux_send,
				list) {
		if (ux_send->start == event->tgt_long.start) {
			dlist_remove(&ux_send->list);
			return ux_send;
		}
	}

	return NULL;
}

/* Caller must hold rxc->lock */
static struct cxip_req *
match_ux_recv(struct cxip_rx_ctx *rxc, const union c_event *event)
{
	struct cxip_req *req;

	/* Look for a previously completed request which was matched to an
	 * overflow buffer that has a matching start pointer.
	 *
	 * TODO this assumes all overflow buffers use the same AC so all start
	 * pointers are unique.
	 */
	dlist_foreach_container(&rxc->ux_recvs, struct cxip_req, req, list) {
		if (req->recv.start == event->tgt_long.start) {
			dlist_remove(&req->list);
			return req;
		}
	}

	return NULL;
}

static void report_recv_completion(struct cxip_req *req)
{
	int ret;
	int truncated;
	int err;

	req->data_len = req->recv.mlength;

	truncated = req->recv.rlength - req->recv.mlength;
	if (req->recv.rc == C_RC_OK && !truncated) {
		CXIP_LOG_DBG("Request success: %p\n", req);

		ret = req->cq->report_completion(req->cq, req->recv.src_addr,
						 req);
		if (ret != req->cq->cq_entry_size)
			CXIP_LOG_ERROR("Failed to report completion: %d\n",
				       ret);
	} else {
		err = truncated ? FI_EMSGSIZE : FI_EIO;
		CXIP_LOG_DBG("Request error: %p (err: %d)\n", req, err);

		ret = cxip_cq_report_error(req->cq, req, truncated, err,
					   req->recv.rc, NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}
}

static void oflow_buf_free(struct cxip_oflow_buf *oflow_buf)
{
	cxil_unmap(oflow_buf->md);
	free(oflow_buf->buf);
	free(oflow_buf);
}

static void oflow_buf_get(struct cxip_oflow_buf *oflow_buf)
{
	ofi_atomic_inc32(&oflow_buf->ref);
}

static void oflow_buf_put(struct cxip_oflow_buf *oflow_buf)
{
	if (!ofi_atomic_dec32(&oflow_buf->ref)) {
		oflow_buf_free(oflow_buf);
	}
}

static int rdvs_issue_get(struct cxip_req *req, struct cxip_ux_send *ux_send)
{

	union c_fab_addr dfa;
	uint32_t pid_granule, pid_idx;
	uint8_t idx_ext;
	union c_cmdu cmd = {};
	int ret;
	struct cxip_rx_ctx *rxc = req->recv.rxc;
	uint32_t pid_bits = rxc->domain->dev_if->if_dev->info.pid_bits;
	uint32_t nic;
	uint32_t pid;

	nic = CXI_MATCH_ID_EP(pid_bits, ux_send->initiator);
	pid = CXI_MATCH_ID_PID(pid_bits, ux_send->initiator);

	if (req->recv.rlength < ux_send->length)
		req->recv.mlength = req->recv.rlength;
	else
		req->recv.mlength = ux_send->length;
	req->recv.rlength = ux_send->length;

	pid_granule = rxc->domain->dev_if->if_pid_granule;
	pid_idx = CXIP_RDVS_IDX(pid_granule);
	cxi_build_dfa(nic, pid, pid_granule, pid_idx, &dfa, &idx_ext);

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_GET;
	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = req->recv.recv_md->lac;
	cmd.full_dma.remote_offset = 0;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(req->recv.recv_md,
						 req->recv.recv_buf);
	cmd.full_dma.eq = rxc->comp.recv_cq->evtq->eqn;
	cmd.full_dma.request_len = req->recv.mlength;
	cmd.full_dma.match_bits = ux_send->match_bits.rdvs_id;
	cmd.full_dma.initiator = CXI_MATCH_ID(pid_bits,
					      rxc->ep_obj->src_addr.pid,
					      rxc->ep_obj->src_addr.nic);
	cmd.full_dma.user_ptr = (uint64_t)req;

	fastlock_acquire(&rxc->lock);

	/* Issue Rendezvous Get command */
	ret = cxi_cq_emit_dma(rxc->tx_cmdq, &cmd.full_dma);
	if (ret) {
		CXIP_LOG_ERROR("Failed to queue GET command: %d\n", ret);

		ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO, ret, NULL,
					   0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
		goto unlock;
	}

	cxi_cq_ring(rxc->tx_cmdq);

	ret = FI_SUCCESS;
unlock:
	fastlock_release(&rxc->lock);
	return ret;
}

int cxip_rxc_oflow_replenish(struct cxip_rx_ctx *rxc);

/* Process an overflow buffer event.
 *
 * We can expect Link, Unlink and Put events from the overflow buffer.
 *
 * A Link event arrives when the append has been completed.  If successful Link
 * events are suppressed, it can be assumed that the event contains an error
 * and the append failed.
 *
 * An Unlink event is expected when the overflow buffer space is exhausted.
 * Overflow buffers are configured to use locally managed LEs.  When enough
 * Puts have matched in an overflow buffer, consuming its space, the NIC will
 * automatically unlink the LE.  An automatic Unlink event will be generated
 * before the final Put which caused space to be exhausted.
 *
 * An Unlink may also be generated by an Unlink command.  In this case, the
 * auto_unlinked field in the event will be zero.  In this case, free the
 * request immediately.
 *
 * A Put event will be generated for each Put that matches the overflow buffer
 * LE.  The Put event indicates that data is in the overflow buffer to be
 * copied into a user buffer.  This event must be correlated to a Put_Overflow
 * event from a user buffer LE.  The Put_Overflow event may arrive before or
 * after the Put event.
 *
 * When each Put event arrives, check for the existence of a previously posted
 * receive buffer which generated a matching Put_Overflow event.  If such a
 * buffer exists, copy data from the overflow buffer to the user buffer.
 * Otherwise, store a record of the Put event for matching once a user posts a
 * new buffer that matches.
 *
 * If data will remain in the overflow buffer, take a reference to it to
 * prevent it from being freed.  If a sequence of Unlink-Put events is
 * detected, drop a reference to the overflow buffer so it is automatically
 * freed once all user data is copied out.
 */
static void cxip_oflow_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rx_ctx *rxc = req->oflow.rxc;
	struct cxip_req *ux_recv;
	struct cxip_ux_send *ux_send;
	struct cxip_oflow_buf *oflow_buf;
	void *oflow_va;

	oflow_buf = req->oflow.oflow_buf;

	/* Netsim is currently giving events in the order: LINK-UNLINK-PUT.
	 * Assume this order is guaranteed for now.
	 */

	if (event->hdr.event_type == C_EVENT_LINK) {
		/* TODO Handle append errors. */
		return;
	}

	if (event->hdr.event_type == C_EVENT_UNLINK) {
		if (oflow_buf->type == EAGER_OFLOW_BUF)
			ofi_atomic_dec32(&rxc->oflow_buf_cnt);
		else if (oflow_buf->type == LONG_RDVS_OFLOW_BUF)
			ofi_atomic_dec32(&rxc->ux_rdvs_buf.ref);

		/* Check if this LE was Unlinked explicitly or
		 * automatically unlinked due to buffer exhaustion.
		 */
		if (!event->tgt_long.auto_unlinked) {
			cxip_cq_req_free(req);
			return;
		}

		/* Mark the overflow buffer exhausted.  One more Put event is
		 * expected.  When the event for the Put which exhausted
		 * resources arrives, drop a reference to the overflow buffer.
		 */
		oflow_buf->exhausted = 1;

		/* Refill overflow buffers */
		if (oflow_buf->type == EAGER_OFLOW_BUF)
			cxip_rxc_oflow_replenish(rxc);

		return;
	}

	if (event->hdr.event_type != C_EVENT_PUT) {
		CXIP_LOG_ERROR("Unexpected event type: %d\n",
			       event->hdr.event_type);
		return;
	}

	fastlock_acquire(&rxc->lock);

	/* Check for a previously received PUT_OVERFLOW event */
	ux_recv = match_ux_recv(rxc, event);
	if (!ux_recv) {
		/* A PUT_OVERFLOW event is pending.  Store a record of this
		 * unexpected Put event for lookup when the event arrives.
		 */

		/* TODO make fast allocator for ux_sends */
		ux_send = malloc(sizeof(struct cxip_ux_send));
		if (!ux_send) {
			CXIP_LOG_ERROR("Failed to malloc ux_send\n");
			abort();
		}

		if (oflow_buf->type == EAGER_OFLOW_BUF)
			ux_send->length = event->tgt_long.mlength;
		else
			ux_send->length = event->tgt_long.rlength;

		ux_send->oflow_buf = oflow_buf;
		ux_send->start = event->tgt_long.start;
		ux_send->match_bits.raw = event->tgt_long.match_bits;
		ux_send->initiator =
			event->tgt_long.initiator.initiator.process;

		/* Prevent the overflow buffer from being freed until the user
		 * has copied out data.
		 */
		if (oflow_buf->type == EAGER_OFLOW_BUF)
			oflow_buf_get(oflow_buf);

		dlist_insert_tail(&ux_send->list,
				  &rxc->ux_sends);

		fastlock_release(&rxc->lock);
	} else {
		/* A matching PUT_OVERFLOW event arrived earlier.  Data is
		 * waiting in the overflow buffer.
		 */

		fastlock_release(&rxc->lock);

		CXIP_LOG_DBG("Matched ux_recv, data: 0x%lx\n",
			     ux_recv->recv.start);

		if (oflow_buf->type == EAGER_OFLOW_BUF) {
			/* For eager messages, copy data from the overflow
			 * buffer
			 */
			oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md,
							  ux_recv->recv.start);

			if (event->tgt_long.mlength > req->recv.rlength) {
				req->recv.mlength = req->recv.rlength;
				req->recv.rlength = event->tgt_long.mlength;
			} else {
				req->recv.rlength = req->recv.mlength =
						event->tgt_long.mlength;
			}

			memcpy(ux_recv->recv.recv_buf, oflow_va,
			       ux_recv->recv.mlength);

			report_recv_completion(ux_recv);

			/* Free the matched user buffer request */
			cxip_cq_req_free(ux_recv);
		} else {
			/* For SW Rendezvous messages, issue a GET to retrieve
			 * the data from the initiator
			 */
			int ret;
			struct cxip_ux_send ux_send;

			ux_send.oflow_buf = oflow_buf;
			ux_send.start = event->tgt_long.start;
			ux_send.length = event->tgt_long.rlength;
			ux_send.match_bits.raw = event->tgt_long.match_bits;
			ux_send.initiator =
				event->tgt_long.initiator.initiator.process;

			ret = rdvs_issue_get(ux_recv, &ux_send);
			if (ret != FI_SUCCESS)
				cxip_cq_req_free(ux_recv);
		}
	}

	if (oflow_buf->type == EAGER_OFLOW_BUF && oflow_buf->exhausted) {
		CXIP_LOG_DBG("Oflow buf exhausted, buf: %p\n", oflow_buf->buf);

		/* This is the last event to the overflow buffer, drop a
		 * reference to it.  The buffer will be freed once all user
		 * data is copied out.
		 */
		oflow_buf_put(oflow_buf);

		/* No further events are expected, free the overflow buffer
		 * request immediately.
		 */
		cxip_cq_req_free(req);
	}
}

/* Append a Locally Managed LE to the Overflow list to match Eager Sends. */
static int oflow_buf_add(struct cxip_rx_ctx *rxc)
{
	struct cxip_domain *dom;
	int ret;
	struct cxip_oflow_buf *oflow_buf;
	struct cxip_req *req;
	uint64_t min_free;

	/* Match all tagged, eager sends */
	union cxip_match_bits mb = { .tagged = 1, .rdvs = 0 };
	union cxip_match_bits ib = { .rdvs_id = ~0, .tag = ~0 };

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
	ret = cxil_map(dom->dev_if->if_lni, (void *)oflow_buf->buf,
		       rxc->oflow_buf_size,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_WRITE,
		       NULL, &oflow_buf->md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map oflow buffer: %d\n", ret);
		goto free_buf;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->comp.recv_cq, 1);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto oflow_unmap;
	}

	req->cb = cxip_oflow_cb;
	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = oflow_buf;

	fastlock_acquire(&rxc->lock);

	/* Multiple callers can race to allocate overflow buffers */
	if (ofi_atomic_get32(&rxc->oflow_buf_cnt) >= rxc->oflow_bufs_max) {
		ret = FI_SUCCESS;
		goto oflow_unlock;
	}

	/* Issue Append command */
	min_free = (rxc->eager_threshold >>
			dom->dev_if->if_dev->info.min_free_shift);
	ret = issue_append_le(rxc->rx_pte->pte, oflow_buf->buf,
			      rxc->oflow_buf_size, oflow_buf->md,
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, min_free, false, false, true,
			      true, true, false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto oflow_unlock;
	}

	/* Initialize oflow_buf structure */
	dlist_insert_tail(&oflow_buf->list, &rxc->oflow_bufs);
	oflow_buf->rxc = rxc;
	ofi_atomic_initialize32(&oflow_buf->ref, 1);
	oflow_buf->exhausted = 0;
	oflow_buf->buffer_id = req->req_id;
	oflow_buf->type = EAGER_OFLOW_BUF;

	ofi_atomic_inc32(&rxc->oflow_buf_cnt);

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&rxc->lock);

	return FI_SUCCESS;

oflow_unlock:
	fastlock_release(&rxc->lock);
	cxip_cq_req_free(req);
oflow_unmap:
	cxil_unmap(oflow_buf->md);
free_buf:
	free(oflow_buf->buf);
free_oflow:
	free(oflow_buf);

	return ret;
}

/* Replenish RXC Overflow buffers for Eager Sends. */
int cxip_rxc_oflow_replenish(struct cxip_rx_ctx *rxc)
{
	int ret;

	while (ofi_atomic_get32(&rxc->oflow_buf_cnt) < rxc->oflow_bufs_max) {
		ret = oflow_buf_add(rxc);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_ERROR("Failed to append oflow buffer: %d\n",
				       ret);
			return ret;
		}
	}

	return FI_SUCCESS;
}

/* Free RX Context overflow buffers.
 *
 * The RXC must be disabled with no outstanding posted receives. Adding new
 * posted receives that could match the overflow buffers while cleanup is in
 * progress will cause issues. Also, with the RXC lock held, processing
 * messages on the context may cause a deadlock.
 *
 * Caller must hold rxc->lock.
 */
static void cxip_rxc_oflow_fini(struct cxip_rx_ctx *rxc)
{
	int ret;
	union c_cmdu cmd = {};
	struct cxip_oflow_buf *oflow_buf;
	struct cxip_ux_send *ux_send;
	struct dlist_entry *itmp;
	struct dlist_entry *otmp;

	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = C_PTL_LIST_OVERFLOW;
	cmd.target.ptlte_index  = rxc->rx_pte->pte->ptn;

	/* Manually unlink each overflow buffer */
	dlist_foreach_container(&rxc->oflow_bufs, struct cxip_oflow_buf,
				oflow_buf, list) {
		cmd.target.buffer_id = oflow_buf->buffer_id;

		ret = cxi_cq_emit_target(rxc->rx_cmdq, &cmd);
		if (ret) {
			/* TODO handle insufficient CMDQ space */
			CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		}
	}

	cxi_cq_ring(rxc->rx_cmdq);

	/* Wait for all overflow buffers to be unlinked */
	do {
		sched_yield();
		cxip_cq_progress(rxc->comp.recv_cq);
	} while (ofi_atomic_get32(&rxc->oflow_buf_cnt));

	/* Clean up overflow buffers */
	dlist_foreach_container_safe(&rxc->oflow_bufs, struct cxip_oflow_buf,
				     oflow_buf, list, otmp) {

		dlist_foreach_container_safe(&rxc->ux_sends,
					     struct cxip_ux_send, ux_send,
					     list, itmp) {
			if (ux_send->oflow_buf == oflow_buf) {
				dlist_remove(&ux_send->list);
				free(ux_send);
			}
		}

		dlist_remove(&oflow_buf->list);
		oflow_buf_free(oflow_buf);
	}
}

/* Append a persistent LE to the Overflow list to match RDVS Sends. */
static int cxip_rxc_rdvs_init(struct cxip_rx_ctx *rxc)
{
	struct cxip_domain *dom;
	int ret;
	struct cxip_req *req;
	void *ux_buf;
	struct cxi_md *md;

	/* Match all tagged, rendezvous sends */
	union cxip_match_bits mb = { .tagged = 1, .rdvs = 1 };
	union cxip_match_bits ib = { .rdvs_id = ~0, .tag = ~0 };

	dom = rxc->domain;

	/* Allocate a small data buffer */
	ux_buf = calloc(1, 1);
	if (!ux_buf) {
		CXIP_LOG_ERROR("Unable to allocate ux buffer\n");
		return -FI_ENOMEM;
	}

	/* Map overflow data buffer */
	ret = cxil_map(dom->dev_if->if_lni, ux_buf, 1,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_WRITE, NULL, &md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map ux buffer: %d\n", ret);
		goto free_ux_buf;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->comp.recv_cq, 1);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate ux request\n");
		ret = -FI_ENOMEM;
		goto unmap_ux;
	}

	/* Build Append command descriptor */
	fastlock_acquire(&rxc->lock);

	/* Multiple callers can race to allocate ux buffer */
	if (ofi_atomic_get32(&rxc->ux_rdvs_buf.ref) != 0) {
		ret = FI_SUCCESS;
		goto unlock_ux;
	}

	ret = issue_append_le(rxc->rx_pte->pte, ux_buf, 1, md, // TODO len 1?
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, false, false, true, false,
			      true, false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write UX Append command: %d\n", ret);
		goto unlock_ux;
	}

	/* Initialize oflow_buf structure */
	ofi_atomic_inc32(&rxc->ux_rdvs_buf.ref);
	rxc->ux_rdvs_buf.type = LONG_RDVS_OFLOW_BUF;
	rxc->ux_rdvs_buf.buf = ux_buf;
	rxc->ux_rdvs_buf.md = md;
	rxc->ux_rdvs_buf.rxc = rxc;
	rxc->ux_rdvs_buf.buffer_id = req->req_id;

	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = &rxc->ux_rdvs_buf;
	req->cb = cxip_oflow_cb;

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&rxc->lock);

	return FI_SUCCESS;

unlock_ux:
	fastlock_release(&rxc->lock);
	cxip_cq_req_free(req);
unmap_ux:
	cxil_unmap(md);
free_ux_buf:
	free(ux_buf);

	return ret;
}

/* Free RX Context Software Rendezvous buffers.
 *
 * The RXC must be disabled with no outstanding posted receives. Adding new
 * posted receives that could match the overflow buffers while cleanup is in
 * progress will cause issues. Also, with the RXC lock held, processing
 * messages on the context may cause a deadlock.
 *
 * Caller must hold rxc->lock.
 */
static void cxip_rxc_rdvs_fini(struct cxip_rx_ctx *rxc)
{
	int ret;

	ret = issue_unlink_le(rxc->rx_pte->pte, C_PTL_LIST_OVERFLOW,
			      rxc->ux_rdvs_buf.buffer_id, rxc->rx_cmdq);
	if (ret) {
		/* TODO handle error */
		CXIP_LOG_ERROR("Failed to enqueue unlink command: %d\n", ret);
	} else {
		do {
			sched_yield();
			cxip_cq_progress(rxc->comp.recv_cq);
		} while (ofi_atomic_get32(&rxc->ux_rdvs_buf.ref));

		/* Clean up overflow buffers */
		ret = cxil_unmap(rxc->ux_rdvs_buf.md);
		if (ret) {
			/* TODO handle error */
			CXIP_LOG_ERROR("Failed to unmap ux buffer: %d\n", ret);
		}
		free(rxc->ux_rdvs_buf.buf);
	}
}

/* Initialize an RXC for Tagged messaging */
int cxip_rxc_tagged_init(struct cxip_rx_ctx *rxc)
{
	int ret;

	ret = cxip_rxc_oflow_replenish(rxc);
	if (ret) {
		CXIP_LOG_ERROR("cxip_rxc_oflow_replenish failed: %d\n", ret);
		return ret;
	}

	ret = cxip_rxc_rdvs_init(rxc);
	if (ret) {
		CXIP_LOG_ERROR("cxip_rxc_rdvs_init failed: %d\n", ret);
		cxip_rxc_oflow_fini(rxc);
		return ret;
	}

	return FI_SUCCESS;
}

/* Finalize an RXC for Tagged messaging */
void cxip_rxc_tagged_fini(struct cxip_rx_ctx *rxc)
{
	cxip_rxc_rdvs_fini(rxc);

	cxip_rxc_oflow_fini(rxc);
}

/* Return the FI address of the initiator of the send operation. */
static fi_addr_t _rxc_event_src_addr(struct cxip_rx_ctx *rxc,
				     const union c_event *event)
{
	/* If the FI_SOURCE capability is enabled, convert the initiator's
	 * address to an FI address to be reported in a CQ event. If
	 * application AVs are symmetric, the match_id in the EQ event is
	 * logical and translation is not needed.  Otherwise, translate the
	 * physical address in the EQ event to logical FI address.
	 */
	if (rxc->attr.caps & FI_SOURCE) {
		uint32_t pid_bits = rxc->domain->dev_if->if_dev->info.pid_bits;
		uint32_t process = event->tgt_long.initiator.initiator.process;
		uint32_t nic;
		uint32_t pid;

		if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC)
			return CXI_MATCH_ID_EP(pid_bits, process);

		nic = CXI_MATCH_ID_EP(pid_bits, process);
		pid = CXI_MATCH_ID_PID(pid_bits, process);

		return _cxip_av_reverse_lookup(rxc->ep_obj->av, nic, pid);
	}

	return FI_ADDR_NOTAVAIL;
}

/* Process a posted receive buffer event.
 *
 * We can expect Link, Unlink, Put and Put_Overflow and Reply events from a
 * posted receive buffer.
 *
 * For eager-sized or expected sends a pre-posted receive matches an incoming
 * send, netsim generates events in the order: Link-Unlink-Put.  This order is
 * guaranteed. In this case, the Put event indicates that the LE was matched in
 * hardware and the operation is complete.
 *
 * For software rendezvous sends, the receive buffer will cause a match of the
 * truncated send that has occurred. The event order is: Put_Overflow-Reply The
 * Put_Overflow event indicates that a truncated send was attempted by an
 * initiator. A GET operation must be initiated by the receiver to collect the
 * data from the initiator's PUT operation. Once the GET operation completes and
 * the data is in the local buffer, the Reply event indicates the completion of
 * the GET operation.
 *
 * TODO Cassini allows successful Link and Unlink events to be suppressed.
 * Configure LEs to suppress these events.  In that case, a Link or Unlink
 * event would indicate a transaction failure.  Handle those errors.
 *
 * When a receive matches an unexpected header during the append, netsim
 * generates a Put_Overflow event.  There are no Link events associated with
 * the user buffer in this case.  The Put_Overflow event must be correlated
 * with a Put event generated from an overflow buffer.  The Put event may be
 * generated before or after the Put_Overflow event.
 *
 * When the Put_Overflow event arrives, check for the existence of a previously
 * received, matching Put event from an overflow buffer.  If such an event
 * exists, data may be copied from the overflow buffer to the user buffer and
 * the operation is completed.  If a matching event is not found, store a
 * record of the posted user buffer to be matched when the forthcoming Put
 * event arrives.
 */
static void cxip_trecv_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	struct cxip_rx_ctx *rxc = req->recv.rxc;
	struct cxip_ux_send *ux_send;
	struct cxip_oflow_buf *oflow_buf;
	void *oflow_va;
	union cxip_match_bits mb;
	int event_rc;

	if (event->hdr.event_type == C_EVENT_LINK) {
		/* TODO Handle append errors. */
		event_rc = cxi_tgt_event_rc(event);
		CXIP_LOG_DBG("Link completed with: %d\n", event_rc);
		return;
	} else if (event->hdr.event_type == C_EVENT_UNLINK) {
		/* TODO Handle unlink errors. */
		event_rc = cxi_tgt_event_rc(event);
		CXIP_LOG_DBG("Unlink completed with: %d\n", event_rc);
		return;
	} else if (event->hdr.event_type == C_EVENT_PUT_OVERFLOW) {
		/* We matched an unexpected header */

		fastlock_acquire(&rxc->lock);

		/* Check for a previously received unexpected Put event */
		ux_send = match_ux_send(rxc, event);
		if (!ux_send) {
			/* An unexpected Put event is pending.  Link this
			 * request to the pending list for lookup when the
			 * event arrives.
			 */

			/* Store start address to use for matching against
			 * future events.
			 */
			req->recv.start = event->tgt_long.start;
			mb.raw = event->tgt_long.match_bits;
			req->tag = mb.tag;
			req->recv.src_addr = _rxc_event_src_addr(rxc, event);

			dlist_insert_tail(&req->list, &rxc->ux_recvs);

			CXIP_LOG_DBG("Queued recv req, data: 0x%lx\n",
				     req->recv.start);

			fastlock_release(&rxc->lock);

			return;
		}

		fastlock_release(&rxc->lock);

		oflow_buf = ux_send->oflow_buf;

		req->recv.rc = cxi_tgt_event_rc(event);
		mb.raw = event->tgt_long.match_bits;
		req->tag = mb.tag;
		req->recv.src_addr = _rxc_event_src_addr(rxc, event);

		/* A matching, unexpected Put event arrived earlier.  Data is
		 * waiting in the overflow buffer.
		 */

		if (oflow_buf->type == EAGER_OFLOW_BUF) {
			/* This is an eager sized operation
			 * Copy data from the overflow buffer
			 */
			oflow_va =
				(void *)CXI_IOVA_TO_VA(oflow_buf->md,
						       event->tgt_long.start);

			if (ux_send->length > req->recv.rlength) {
				req->recv.mlength = req->recv.rlength;
				req->recv.rlength = ux_send->length;
			} else {
				req->recv.rlength = req->recv.mlength =
					ux_send->length;
			}

			memcpy(req->recv.recv_buf, oflow_va, req->recv.mlength);

			/* Drop reference to the overflow buffer. It will be
			 * freed once all user data is copied out.
			 */
			oflow_buf_put(oflow_buf);

			/* Release the unexpected Put event record */
			free(ux_send);

			ret = cxil_unmap(req->recv.recv_md);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

			report_recv_completion(req);

			/* Free the user buffer request */
			cxip_cq_req_free(req);
		} else {
			/* SW Rendezvous is in progress. Must issue a GET
			 * operation to receive the data that was sent by the
			 * Put operation's initiator.
			 */
			ret = rdvs_issue_get(req, ux_send);
			if (ret)
				cxip_cq_req_free(req);
		}

		return;
	} else if (event->hdr.event_type == C_EVENT_PUT) {
		/* Data was delivered directly to the user buffer. Complete the
		 * request.
		 */

		req->recv.rc = cxi_tgt_event_rc(event);
		req->recv.rlength = event->tgt_long.rlength;
		req->recv.mlength = event->tgt_long.mlength;
		mb.raw = event->tgt_long.match_bits;
		req->tag = mb.tag;
		req->recv.src_addr = _rxc_event_src_addr(rxc, event);

		ret = cxil_unmap(req->recv.recv_md);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

		report_recv_completion(req);

		/* Free the user buffer request */
		cxip_cq_req_free(req);
		return;
	} else if (event->hdr.event_type == C_EVENT_REPLY) {
		/* The GET operation that was issued as part of the SW
		 * rendezvous operation has completed, transferring the data
		 * into the data buffer.
		 * Complete the operation
		 */
		req->recv.rc = cxi_init_event_rc(event);

		ret = cxil_unmap(req->recv.recv_md);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

		report_recv_completion(req);

		/* Free the user buffer request */
		cxip_cq_req_free(req);
		return;
	}

	CXIP_LOG_ERROR("Unexpected event type: %d\n",
		event->hdr.event_type);

	req->recv.rc = C_RC_INVALID_EVENT;

	report_recv_completion(req);

	/* Free the user buffer request */
	cxip_cq_req_free(req);
}

static ssize_t _cxip_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			  fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			  void *context, uint64_t flags, bool tagged)
{
	struct cxip_ep *cxi_ep;
	struct cxip_rx_ctx *rxc;
	struct cxip_domain *dom;
	int ret;
	struct cxi_md *recv_md;
	struct cxip_req *req;
	struct cxip_addr caddr;
	uint32_t match_id;
	uint32_t pid_bits;
	union cxip_match_bits mb = {};
	union cxip_match_bits ib = { .rdvs = ~0, .rdvs_id = ~0 };

	if (!ep || !buf)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a RX
	 * context), or a RX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		rxc = cxi_ep->ep_obj->rx_ctx;
		break;

	case FI_CLASS_RX_CTX:
		rxc = container_of(ep, struct cxip_rx_ctx, ctx);
		break;

	default:
		CXIP_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	dom = rxc->domain;

	/* If FI_DIRECTED_RECV and a src_addr is specified, encode the address
	 * in the LE for matching. If application AVs are symmetric, use
	 * logical FI address for matching. Otherwise, use physical address.
	 */
	pid_bits = dom->dev_if->if_dev->info.pid_bits;
	if (rxc->attr.caps & FI_DIRECTED_RECV &&
	    src_addr != FI_ADDR_UNSPEC) {
		if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
			/* TODO switch to C_PID_ANY */
			match_id = CXI_MATCH_ID(pid_bits, 0x1ff, src_addr);
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
		/* TODO switch to C_PID_ANY */
		match_id = CXI_MATCH_ID(pid_bits, 0x1ff, C_NID_ANY);
	}

	/* Map local buffer */
	ret = cxil_map(dom->dev_if->if_lni, (void *)buf, len,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_WRITE, NULL,
		       &recv_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map recv buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->comp.recv_cq, 1);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto trecv_unmap;
	}

	/* req->data_len, req->tag must be set later.  req->buf and req->data
	 * may be overwritten later.
	 */
	req->context = (uint64_t)context;

	req->flags = FI_RECV;
	if (tagged)
		req->flags |= FI_TAGGED;
	else
		req->flags |= FI_MSG;

	req->buf = 0;
	req->data = 0;
	req->cb = cxip_trecv_cb;

	req->recv.rxc = rxc;
	req->recv.recv_buf = buf;
	req->recv.recv_md = recv_md;
	req->recv.rlength = len;

	if (tagged) {
		mb.tagged = 1;
		mb.tag = tag;
		ib.tag = ignore;
	}

	fastlock_acquire(&rxc->lock);

	/* Issue Append command */
	ret = issue_append_le(rxc->rx_pte->pte, buf, len, recv_md,
			      C_PTL_LIST_PRIORITY, req->req_id, mb.raw, ib.raw,
			      match_id, 0, false, true, false, false, true,
			      false, rxc->rx_cmdq);
	if (ret) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto trecv_unlock;
	}

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&rxc->lock);

	return FI_SUCCESS;

trecv_unlock:
	fastlock_release(&rxc->lock);
	cxip_cq_req_free(req);
trecv_unmap:
	cxil_unmap(recv_md);

	return ret;
}

static void rdvs_send_req_free(struct cxip_req *req)
{
	int ret;

	fastlock_acquire(&req->send.txc->lock);

	cxip_tx_ctx_free_rdvs_id(req->send.txc, req->send.rdvs_id);
	CXIP_LOG_DBG("Freed RDVS ID: %d\n", req->send.rdvs_id);

	fastlock_release(&req->send.txc->lock);

	ret = cxil_unmap(req->send.send_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

	cxip_cq_req_free(req);
}

/*
 * Rendezvous Send callback.
 *
 * Progress a Rendezvous Send operation to completion. A Rendezvous Send
 * operation is performed as follows:
 *
 * 1. An LE describing the Send buffer is appended to the TX PtlTE Priority list
 * 2. An Unlink event is generated indicating the LE was appended
 * 3. The Initiator performs a Put of the entire Send payload
 * 4. An Ack event is generated indicating the Put completed
 * 5. An Unlink event is generated indicating the source buffer LE has been
 *    matched to a transaction.
 * 6. A Get event is generated indicating the source data was read.
 */
static void cxip_rdvs_send_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	struct cxip_tx_ctx *txc = req->send.txc;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Link error: %d id: %d\n",
				       event_rc,
				       event->tgt_long.buffer_id);

			ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO,
						   event_rc, NULL, 0);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report error: %d\n",
					       ret);

			rdvs_send_req_free(req);
			return;
		}

		/* The send buffer LE is linked. Perform Put of the payload. */
		fastlock_acquire(&txc->lock);

		ret = cxi_cq_emit_dma(txc->tx_cmdq, &req->send.cmd.full_dma);
		if (ret) {
			CXIP_LOG_ERROR("Failed to enqueue PUT: %d\n", ret);

			/* Save the error and clean up operation. */
			req->send.event_failure = ret;
			goto rdvs_unlink_le;
		}

		cxi_cq_ring(txc->tx_cmdq);

		fastlock_release(&txc->lock);

		CXIP_LOG_DBG("Enqueued Put: %p\n", req);
		return;
	case C_EVENT_ACK:
		/* The payload was delivered to the target. */
		event_rc = cxi_init_event_rc(event);
		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Ack error: %d\n", event_rc);

			/* Save the error and clean up operation. */
			req->send.event_failure = event_rc;

			fastlock_acquire(&txc->lock);
rdvs_unlink_le:
			issue_unlink_le(txc->rdvs_pte->pte,
					C_PTL_LIST_PRIORITY,
					req->req_id,
					txc->rx_cmdq);

			fastlock_release(&txc->lock);
		} else if (event->init_short.ptl_list == C_PTL_LIST_PRIORITY) {
			CXIP_LOG_DBG("Put Acked (Priority): %p\n", req);

			/* The Put matched in the Priority list, the source
			 * buffer LE is unneeded.
			 */
			fastlock_acquire(&txc->lock);
			issue_unlink_le(txc->rdvs_pte->pte,
					C_PTL_LIST_PRIORITY,
					req->req_id,
					txc->rx_cmdq);
			fastlock_release(&txc->lock);

			/* Wait for the Unlink event */
			req->send.complete_on_unlink = 1;
		} else {
			CXIP_LOG_DBG("Put Acked (Overflow): %p\n", req);
		}

		/* Wait for the source buffer LE to be unlinked. */
		return;
	case C_EVENT_UNLINK:
		/* Check if the unlink was caused by a previous failure. */
		if (req->send.event_failure != C_RC_NO_EVENT)
			event_rc = req->send.event_failure;
		else
			event_rc = cxi_tgt_event_rc(event);

		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Unlink error: %d\n", event_rc);

			ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO,
						   event_rc, NULL, 0);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report error: %d\n",
					       ret);

			rdvs_send_req_free(req);
		} else if (req->send.complete_on_unlink) {
			/* The Ack event for the Put operation was received and
			 * all data has already transferred because it was an
			 * expected long Put operation. The data buffer has been
			 * successfully unlinked. Report the completion of the
			 * operation.
			 */
			CXIP_LOG_DBG("Request success (Priority): %p\n", req);
			ret = req->cq->report_completion(req->cq,
							 FI_ADDR_UNSPEC, req);
			if (ret != req->cq->cq_entry_size)
				CXIP_LOG_ERROR("Failed to report completion: %d\n",
					       ret);

			rdvs_send_req_free(req);
		} else {
			CXIP_LOG_DBG("Source LE unlinked:  %p\n", req);
		}

		/* A final Get event is expected indicating the source data has
		 * been read.
		 */
		return;
	case C_EVENT_GET:
		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK) {
			CXIP_LOG_ERROR("Get error: %d\n", event_rc);

			ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO,
						   event_rc, NULL, 0);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report error: %d\n",
					       ret);
		} else {
			CXIP_LOG_DBG("Request success (Overflow): %p\n", req);
			ret = req->cq->report_completion(req->cq,
							 FI_ADDR_UNSPEC, req);
			if (ret != req->cq->cq_entry_size)
				CXIP_LOG_ERROR("Failed to report completion: %d\n",
					       ret);
		}

		rdvs_send_req_free(req);

		return;
	default:
		CXIP_LOG_ERROR("Unexpected event received: %s\n",
			       cxi_event_to_str(event));

		ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO, -FI_EOTHER,
					   NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n",
				       ret);

		rdvs_send_req_free(req);

		return;
	}
}

/* Basic send callback, used for both tagged and untagged messages. */
static void cxip_send_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;

	ret = cxil_unmap(req->send.send_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		CXIP_LOG_DBG("Request success: %p\n", req);

		ret = req->cq->report_completion(req->cq, FI_ADDR_UNSPEC, req);
		if (ret != req->cq->cq_entry_size)
			CXIP_LOG_ERROR("Failed to report completion: %d\n",
				       ret);
	} else {
		CXIP_LOG_DBG("Request error: %p\n", req);

		ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO, event_rc,
					   NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}

	cxip_cq_req_free(req);
}

/* Return the FI address of the TXC. */
static fi_addr_t _txc_fi_addr(struct cxip_tx_ctx *txc)
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

static ssize_t _cxip_send(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context, uint64_t flags, bool tagged)
{
	struct cxip_ep *cxi_ep;
	struct cxip_tx_ctx *txc;
	struct cxip_domain *dom;
	int ret;
	struct cxi_md *send_md;
	struct cxip_req *req;
	union c_cmdu cmd = {};
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;
	uint64_t rx_id;
	bool rdvs;
	uint32_t match_id;
	uint32_t pid_bits;
	union cxip_match_bits mb = {};
	int rdvs_id = -1;

	if (!ep || !buf)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a TX
	 * context), or a TX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		txc = cxi_ep->ep_obj->tx_ctx;
		break;

	case FI_CLASS_TX_CTX:
		txc = container_of(ep, struct cxip_tx_ctx, fid.ctx);
		break;

	default:
		CXIP_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	if (tagged && len > txc->eager_threshold)
		rdvs = true;
	else
		rdvs = false;

	dom = txc->domain;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer */
	ret = cxil_map(dom->dev_if->if_lni, (void *)buf, len,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_READ, NULL,
		       &send_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map send buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(txc->comp.send_cq, rdvs);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto send_unmap;
	}

	req->context = (uint64_t)context;
	req->flags = FI_SEND;

	if (tagged)
		req->flags |= FI_TAGGED;
	else
		req->flags |= FI_MSG;

	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;

	if (rdvs)
		req->cb = cxip_rdvs_send_cb;
	else
		req->cb = cxip_send_cb;

	req->send.txc = txc;
	req->send.send_md = send_md;
	req->send.length = len;
	req->send.event_failure = C_RC_NO_EVENT;

	/* Build Put command descriptor */
	rx_id = CXIP_AV_ADDR_RXC(txc->ep_obj->av, dest_addr);
	pid_granule = dom->dev_if->if_pid_granule;
	pid_idx = CXIP_RXC_TO_IDX(rx_id);
	cxi_build_dfa(caddr.nic, caddr.pid, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	/* Encode the local EP address in the command for matching. If
	 * application AVs are symmetric, use my logical FI address for
	 * matching. Otherwise, use physical address.
	 */
	pid_bits = dom->dev_if->if_dev->info.pid_bits;
	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		match_id = CXI_MATCH_ID(pid_bits, txc->ep_obj->src_addr.pid,
					_txc_fi_addr(txc));
	} else {
		match_id = CXI_MATCH_ID(pid_bits, txc->ep_obj->src_addr.pid,
					txc->ep_obj->src_addr.nic);
	}

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;
	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = send_md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.restricted = 0;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = 0;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(send_md, buf);
	cmd.full_dma.request_len = len;
	cmd.full_dma.eq = txc->comp.send_cq->evtq->eqn;
	cmd.full_dma.user_ptr = (uint64_t)req;
	cmd.full_dma.initiator = match_id;

	fastlock_acquire(&txc->lock);

	if (rdvs) {
		/* Get Rendezvous ID */
		rdvs_id = cxip_tx_ctx_alloc_rdvs_id(txc);
		if (rdvs_id < 0) {
			CXIP_LOG_DBG("Failed alloc RDVS ID\n");
			goto send_unlock;
		}
		CXIP_LOG_DBG("Alloced RDVS ID: %d\n", rdvs_id);

		/* Generate hardware match bits */
		mb.tagged = 1;
		mb.rdvs = 1;
		mb.rdvs_id = rdvs_id;
		mb.tag = tag;
		cmd.full_dma.match_bits = mb.raw;

		/* Update request structure */
		req->send.rdvs_id = rdvs_id;
		req->send.cmd = cmd;

		ret = issue_append_le(txc->rdvs_pte->pte, buf, len, send_md,
				      C_PTL_LIST_PRIORITY, req->req_id,
				      rdvs_id, 0, CXI_MATCH_ID_ANY, 0, false,
				      true, false, true, false, true,
				      txc->rx_cmdq);
		if (ret) {
			CXIP_LOG_DBG("Failed append source buffer: %d\n", ret);
			cxip_tx_ctx_free_rdvs_id(txc, rdvs_id);
			goto send_unlock;
		}
	} else {
		/* Generate hardware match bits */
		if (tagged) {
			mb.tagged = 1;
			mb.tag = tag;
		}
		cmd.full_dma.match_bits = mb.raw;

		/* Issue Eager Put command */
		ret = cxi_cq_emit_dma(txc->tx_cmdq, &cmd.full_dma);
		if (ret) {
			CXIP_LOG_DBG("Failed to write DMA command: %d\n", ret);

			/* Return error according to Domain Resource Mgmt */
			ret = -FI_EAGAIN;
			goto send_unlock;
		}

		cxi_cq_ring(txc->tx_cmdq);
	}

	/* TODO take reference on EP or context for the outstanding
	 * request
	 */
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

send_unlock:
	fastlock_release(&txc->lock);
	cxip_cq_req_free(req);
send_unmap:
	cxil_unmap(send_md);

	return ret;
}

/*
 * APIs
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
	if (count != 1)
		return -FI_EINVAL;

	if (!iov)
		return -FI_EINVAL;

	return _cxip_recv(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  src_addr, tag, ignore, context, 0, true);
}

static ssize_t cxip_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			     uint64_t flags)
{
	if (!msg || !msg->msg_iov)
		return -FI_EINVAL;

	if (msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->tag, msg->ignore, msg->context, 0, true);
}

static ssize_t cxip_tsend(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context)
{
	return _cxip_send(ep, buf, len, desc, dest_addr, tag, context, 0, true);
}

static ssize_t cxip_tsendv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t dest_addr,
			   uint64_t tag, void *context)
{
	if (count != 1)
		return -FI_EINVAL;

	if (!iov)
		return -FI_EINVAL;

	return _cxip_send(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  dest_addr, tag, context, 0, true);
}

static ssize_t cxip_tsendmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			    uint64_t flags)
{
	if (!msg || !msg->msg_iov)
		return -FI_EINVAL;

	if (msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_send(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr,
			  msg->tag, msg->context, flags, true);
}

struct fi_ops_tagged cxip_ep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = cxip_trecv,
	.recvv = cxip_trecvv,
	.recvmsg = cxip_trecvmsg,
	.send = cxip_tsend,
	.sendv = cxip_tsendv,
	.sendmsg = cxip_tsendmsg,
	.inject = fi_no_tagged_inject,
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
	if (count != 1)
		return -FI_EINVAL;

	if (!iov)
		return -FI_EINVAL;

	return _cxip_recv(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL,
			  src_addr, 0, 0, context, 0, false);
}

static ssize_t cxip_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	if (!msg || !msg->msg_iov)
		return -FI_EINVAL;

	if (msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_recv(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr, 0, 0,
			  msg->context, flags, false);
}

static ssize_t cxip_send(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, void *context)
{
	return _cxip_send(ep, buf, len, desc, dest_addr, 0, context, 0, false);
}

static ssize_t cxip_sendv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  void *context)
{
	if (count != 1)
		return -FI_EINVAL;

	if (!iov)
		return -FI_EINVAL;

	return _cxip_send(ep, iov[0].iov_base, iov[0].iov_len,
			  desc ? desc[0] : NULL, dest_addr, 0, context, 0,
			  false);
}

static ssize_t cxip_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	if (!msg || !msg->msg_iov)
		return -FI_EINVAL;

	if (msg->iov_count != 1)
		return -FI_EINVAL;

	return _cxip_send(ep, msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			  msg->desc ? msg->desc[0] : NULL, msg->addr, 0,
			  msg->context, flags, false);
}

struct fi_ops_msg cxip_ep_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = cxip_recv,
	.recvv = cxip_recvv,
	.recvmsg = cxip_recvmsg,
	.send = cxip_send,
	.sendv = cxip_sendv,
	.sendmsg = cxip_sendmsg,
	.inject = fi_no_msg_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

