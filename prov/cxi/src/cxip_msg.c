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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_DATA, __VA_ARGS__)

#define FC_MSG "Message flow-control triggered.\n\n" \
"Flow-control recovery is disabled. To avoid this condition, increase\n" \
"Overflow buffer space using environment variables FI_CXI_OFLOW_*. To\n" \
"enable flow-control recovery (experimental), set environment variable\n" \
"FI_CXI_FC_RECOVERY=1.\n\n"

#define FC_SW_EP_MSG "Flow control triggered due to failure to append LE. "\
"Software endpoint mode required.\n"

static void cxip_ux_onload_complete(struct cxip_req *req);
static int cxip_ux_onload(struct cxip_rxc *rxc);
static int cxip_recv_req_queue(struct cxip_req *req, bool check_rxc_state,
			       bool restart_seq);
static int cxip_recv_req_dropped(struct cxip_req *req);
static void cxip_recv_req_dequeue_nolock(struct cxip_req *req);
static void cxip_recv_req_dequeue(struct cxip_req *req);
static ssize_t _cxip_recv_req(struct cxip_req *req, bool restart_seq);

static int cxip_send_req_dropped(struct cxip_txc *txc, struct cxip_req *req);
static void cxip_send_req_dequeue(struct cxip_txc *txc, struct cxip_req *req);

/*
 * match_put_event() - Find a matching event.
 *
 * For every Put Overflow event there is a matching Put event. These events can
 * be generated in any order. Both events must be received before progress can
 * be made.
 */
static struct cxip_deferred_event *
match_put_event(struct cxip_rxc *rxc, const union c_event *event)
{
	uint32_t process = event->tgt_long.initiator.initiator.process;
	uint32_t ev_rdzv_id = event->tgt_long.rendezvous_id;
	struct cxip_deferred_event *def_ev;
	uint8_t type = event->hdr.event_type;

	dlist_foreach_container(&rxc->deferred_events,
				struct cxip_deferred_event, def_ev,
				rxc_entry) {
		/* Match Put to Put Overflow */
		if (type == def_ev->ev.hdr.event_type)
			continue;

		if (event->tgt_long.rendezvous) {
			/* Rendezvous events are correlated using
			 * rendezvous_id and initiator.
			 */
			if ((def_ev->ev.tgt_long.rendezvous_id == ev_rdzv_id)
					&&
			    (def_ev->ev.tgt_long.initiator.initiator.process
					== process))
				goto found;
		} else {
			/* All other events are correlated using start
			 * address.
			 */
			if (def_ev->ev.tgt_long.start == event->tgt_long.start)
				goto found;
		}
	}

	return NULL;

found:
	assert(def_ev->ev.tgt_long.match_bits == event->tgt_long.match_bits);
	assert(def_ev->ev.tgt_long.initiator.initiator.process ==
	       event->tgt_long.initiator.initiator.process);

	return def_ev;
}

/*
 * defer_put_event() - Store a record of the event for later matching.
 *
 * A Deferred event will be matched to a new event in match_put_event().
 */
static struct cxip_deferred_event *
defer_put_event(struct cxip_rxc *rxc, struct cxip_req *req,
	    const union c_event *event)
{
	struct cxip_deferred_event *ev;

	ev = calloc(1, sizeof(*ev));
	if (!ev)
		return NULL;

	ev->req = req;
	ev->ev = *event;
	dlist_insert_tail(&ev->rxc_entry, &rxc->deferred_events);

	return ev;
}

/*
 * recv_req_src_addr() - Translate request source address to FI address.
 */
static fi_addr_t recv_req_src_addr(struct cxip_req *req)
{
	struct cxip_rxc *rxc = req->recv.rxc;

	/* If the FI_SOURCE capability is enabled, convert the initiator's
	 * address to an FI address to be reported in a CQ event. If
	 * application AVs are symmetric, the match_id in the EQ event is
	 * logical and translation is not needed. Otherwise, translate the
	 * physical address in the EQ event to logical FI address.
	 */
	if (rxc->attr.caps & FI_SOURCE) {
		uint32_t nic;
		uint32_t pid;

		if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC)
			return CXI_MATCH_ID(rxc->pid_bits, 0,
					    req->recv.initiator);

		nic = CXI_MATCH_ID_EP(rxc->pid_bits, req->recv.initiator);
		pid = CXI_MATCH_ID_PID(rxc->pid_bits, req->recv.initiator);

		return _cxip_av_reverse_lookup(rxc->ep_obj->av, nic, pid);
	}

	return FI_ADDR_NOTAVAIL;
}

/*
 * recv_req_complete() - Complete receive request.
 */
static void recv_req_complete(struct cxip_req *req)
{
	assert(dlist_empty(&req->recv.children));

	if (req->recv.recv_md)
		cxip_unmap(req->recv.recv_md);
	ofi_atomic_dec32(&req->recv.rxc->orx_reqs);
	cxip_cq_req_free(req);
}

/*
 * recv_req_report() - Report the completion of a receive operation.
 */
static void recv_req_report(struct cxip_req *req)
{
	int ret;
	int truncated;
	int err;
	fi_addr_t src_addr;
	int success_event = (req->flags & FI_COMPLETION);

	req->flags &= (FI_MSG | FI_TAGGED | FI_RECV);

	if (req->recv.parent) {
		struct cxip_req *parent = req->recv.parent;

		parent->recv.mrecv_bytes -= req->data_len;
		CXIP_DBG("Putting %lu mrecv bytes (req: %p left: %lu addr: %#lx)\n",
			 req->data_len, parent, parent->recv.mrecv_bytes,
			 req->buf);
		if (parent->recv.mrecv_bytes < req->recv.rxc->min_multi_recv) {
			CXIP_DBG("Freeing parent: %p\n", req->recv.parent);
			recv_req_complete(req->recv.parent);

			req->flags |= FI_MULTI_RECV;
		}
	}

	truncated = req->recv.rlen - req->data_len;
	if (req->recv.rc == C_RC_OK && !truncated) {
		CXIP_DBG("Request success: %p\n", req);

		if (success_event) {
			if (req->recv.rxc->attr.caps & FI_SOURCE) {
				src_addr = recv_req_src_addr(req);
				ret = cxip_cq_req_complete_addr(req, src_addr);
			} else {
				ret = cxip_cq_req_complete(req);
			}
			if (ret != FI_SUCCESS)
				CXIP_WARN("Failed to report completion: %d\n",
					  ret);
		}

		if (req->recv.cntr) {
			ret = cxip_cntr_mod(req->recv.cntr, 1, false, false);
			if (ret)
				CXIP_WARN("cxip_cntr_mod returned: %d\n", ret);
		}
	} else {
		if (req->recv.unlinked) {
			err = FI_ECANCELED;
			if (req->recv.multi_recv)
				req->flags |= FI_MULTI_RECV;
			CXIP_DBG("Request canceled: %p (err: %d)\n",
				     req, err);
		} else if (truncated) {
			err = FI_EMSGSIZE;
			CXIP_DBG("Request truncated: %p (err: %d)\n",
				     req, err);
		} else {
			err = FI_EIO;
			CXIP_WARN("Request error: %p (err: %d, %s)\n",
				  req, err, cxi_rc_to_str(req->recv.rc));
		}

		ret = cxip_cq_req_error(req, truncated, err, req->recv.rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_WARN("Failed to report error: %d\n", ret);

		if (req->recv.cntr) {
			ret = cxip_cntr_mod(req->recv.cntr, 1, false, true);
			if (ret)
				CXIP_WARN("cxip_cntr_mod returned: %d\n",
					  ret);
		}
	}
}

/*
 * recv_req_tgt_event() - Update common receive request fields
 *
 * Populate a receive request with information found in all receive event
 * types.
 */
static void
recv_req_tgt_event(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	union cxip_match_bits mb = {
		.raw = event->tgt_long.match_bits
	};

	assert(event->hdr.event_type == C_EVENT_PUT ||
	       event->hdr.event_type == C_EVENT_PUT_OVERFLOW ||
	       event->hdr.event_type == C_EVENT_RENDEZVOUS);

	/* Rendezvous events contain the wrong match bits. */
	if (event->hdr.event_type != C_EVENT_RENDEZVOUS)
		req->tag = mb.tag;

	/* remote_offset is not provided in Overflow events. */
	if (event->hdr.event_type != C_EVENT_PUT_OVERFLOW)
		req->recv.src_offset = event->tgt_long.remote_offset;

	/* Only need one event to set remaining fields. */
	if (req->recv.tgt_event)
		return;
	req->recv.tgt_event = true;

	/* rlen is used to detect truncation. */
	req->recv.rlen = event->tgt_long.rlength;

	/* RC is used when generating completion events. */
	req->recv.rc = cxi_tgt_event_rc(event);

	/* Header data is provided in all completion events. */
	req->data = event->tgt_long.header_data;

	/* rdzv_id is used to correlate Put and Put Overflow events when using
	 * offloaded RPut. Otherwise, Overflow buffer start address is used to
	 * correlate events.
	 */
	if (event->tgt_long.rendezvous) {
		req->recv.rdzv_id = event->tgt_long.rendezvous_id;
	} else {
		req->recv.oflow_start = event->tgt_long.start;
		req->recv.rdzv_id = mb.rdzv_id_hi;
	}
	req->recv.rdzv_lac = mb.rdzv_lac;
	req->recv.rdzv_mlen = event->tgt_long.mlength;

	/* Initiator is provided in completion events. */
	if (event->hdr.event_type == C_EVENT_RENDEZVOUS) {
		uint32_t dfa = event->tgt_long.initiator.initiator.process;

		req->recv.initiator = cxi_dfa_to_init(dfa, rxc->pid_bits);
	} else {
		uint32_t init = event->tgt_long.initiator.initiator.process;

		if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
			/* Take PID out of logical address. */
			req->recv.initiator = CXI_MATCH_ID_EP(rxc->pid_bits,
							      init);
		} else {
			req->recv.initiator = init;
		}
	}

	/* data_len must be set uniquely for each protocol! */
}

/*
 * rdzv_mrecv_req_lookup() - Search for a matching rendezvous, multi-receive
 * child request.
 */
static struct cxip_req *rdzv_mrecv_req_lookup(struct cxip_req *req,
					      const union c_event *event,
					      uint32_t *initiator,
					      uint32_t *rdzv_id)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	struct cxip_req *child_req;
	uint32_t ev_init;
	uint32_t ev_rdzv_id;

	if (event->hdr.event_type == C_EVENT_REPLY) {
		struct cxi_rdzv_user_ptr *user_ptr;

		/* Events for software-issued operations will return a
		 * reference to the correct request.
		 */
		if (!event->init_short.rendezvous)
			return req;

		user_ptr = (struct cxi_rdzv_user_ptr *)
				&event->init_short.user_ptr;

		ev_init = CXI_MATCH_ID(rxc->pid_bits, user_ptr->src_pid,
					user_ptr->src_nid);
		ev_rdzv_id = user_ptr->rendezvous_id;
	} else if (event->hdr.event_type == C_EVENT_RENDEZVOUS) {
		struct cxip_rxc *rxc = req->recv.rxc;
		uint32_t dfa = event->tgt_long.initiator.initiator.process;

		ev_init = cxi_dfa_to_init(dfa, rxc->pid_bits);
		ev_rdzv_id = event->tgt_long.rendezvous_id;
	} else {
		ev_init = event->tgt_long.initiator.initiator.process;
		ev_rdzv_id = event->tgt_long.rendezvous_id;
	}

	*initiator = ev_init;
	*rdzv_id = ev_rdzv_id;

	/* Events for hardware-issued operations will return a rendezvous_id
	 * and initiator data. Use these fields to find a matching child
	 * request.
	 */
	dlist_foreach_container(&req->recv.children,
				struct cxip_req, child_req,
				recv.children) {
		if (child_req->recv.rdzv_id == ev_rdzv_id &&
		    child_req->recv.initiator == ev_init) {
			return child_req;
		}
	}

	return NULL;
}

/*
 * mrecv_req_dup() - Create a new request using an event targeting a
 * multi-recv buffer.
 *
 * @mrecv_req: A previously posted multi-recv buffer request.
 */
static struct cxip_req *mrecv_req_dup(struct cxip_req *mrecv_req)
{
	struct cxip_rxc *rxc = mrecv_req->recv.rxc;
	struct cxip_req *req;

	req = cxip_cq_req_alloc(rxc->recv_cq, 0, rxc);
	if (!req)
		return NULL;

	/* Duplicate the parent request. */
	req->cb = mrecv_req->cb;
	req->context = mrecv_req->context;
	req->flags = mrecv_req->flags;
	req->type = mrecv_req->type;
	req->recv = mrecv_req->recv;

	/* Update fields specific to this Send */
	req->recv.parent = mrecv_req;

	/* Start pointer and data_len must be set elsewhere! */

	return req;
}

/*
 * rdzv_mrecv_req_event() - Look up a multi-recieve child request using an
 * event and multi-recv request.
 *
 * Each rendezvous Put transaction targeting a multi-receive buffer is tracked
 * using a separate child request. A child request is uniquely identified by
 * rendezvous ID and source address. Return a reference to a child request
 * which matches the event. Allocate a new child request, if necessary.
 */
static struct cxip_req *
rdzv_mrecv_req_event(struct cxip_req *mrecv_req, const union c_event *event)
{
	uint32_t ev_init;
	uint32_t ev_rdzv_id;
	struct cxip_req *req;

	assert(event->hdr.event_type == C_EVENT_REPLY ||
	       event->hdr.event_type == C_EVENT_PUT ||
	       event->hdr.event_type == C_EVENT_PUT_OVERFLOW ||
	       event->hdr.event_type == C_EVENT_RENDEZVOUS);

	req = rdzv_mrecv_req_lookup(mrecv_req, event, &ev_init, &ev_rdzv_id);
	if (!req) {
		req = mrecv_req_dup(mrecv_req);
		if (!req)
			return NULL;

		/* Store event initiator and rdzv_id for matching. */
		if (event->hdr.event_type == C_EVENT_REPLY) {
			req->recv.rdzv_id = ev_rdzv_id;
			req->recv.initiator = ev_init;
		}
		dlist_insert_tail(&req->recv.children,
				  &mrecv_req->recv.children);

		CXIP_DBG("New child: %p parent: %p event: %s\n",
			 req, mrecv_req, cxi_event_to_str(event));
	} else {
		CXIP_DBG("Found child: %p parent: %p event: %s\n",
			 req, mrecv_req, cxi_event_to_str(event));
	}

	return req;
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
	if (++req->recv.rdzv_events == 3) {
		if (req->recv.multi_recv) {
			dlist_remove(&req->recv.children);
			recv_req_report(req);
			cxip_cq_req_free(req);
		} else {
			recv_req_report(req);
			recv_req_complete(req);
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
	CXIP_DBG("Freeing: %p\n", oflow_buf);

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
	struct cxip_oflow_buf *oflow_buf = req->oflow.oflow_buf;

	oflow_buf->sw_consumed += bytes;
	CXIP_DBG("Putting %lu bytes (%lu/%lu): %p\n",
		 bytes, oflow_buf->sw_consumed, oflow_buf->hw_consumed, req);

	if (oflow_buf->sw_consumed == oflow_buf->hw_consumed) {
		oflow_buf_free(oflow_buf);
		cxip_cq_req_free(req);
	}
}

/*
 * issue_rdzv_get() - Perform a Get to pull source data from the Initiator of a
 * Send operation.
 */
static int issue_rdzv_get(struct cxip_req *req)
{
	union c_cmdu cmd = {};
	struct cxip_rxc *rxc = req->recv.rxc;
	uint32_t pid_idx = rxc->domain->iface->dev->info.rdzv_get_idx;
	uint8_t idx_ext;
	union cxip_match_bits mb = {};
	int ret;
	uint32_t nic;
	uint32_t pid;
	union c_fab_addr dfa;

	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		struct cxip_addr caddr;

		CXIP_DBG("Translating inititiator: %x, req: %p\n",
			 req->recv.initiator, req);

		ret = _cxip_av_lookup(rxc->ep_obj->av, req->recv.initiator,
				      &caddr);
		if (ret != FI_SUCCESS) {
			CXIP_FATAL("Failed to look up FI addr: %d\n", ret);
		}
		nic = caddr.nic;
		pid = caddr.pid;
	} else {
		nic = CXI_MATCH_ID_EP(rxc->pid_bits, req->recv.initiator);
		pid = CXI_MATCH_ID_PID(rxc->pid_bits, req->recv.initiator);
	}

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_GET;
	cmd.full_dma.lac = req->recv.recv_md->md->lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.eq = rxc->recv_cq->evtq->eqn;

	mb.rdzv_lac = req->recv.rdzv_lac;
	mb.rdzv_id_lo = req->recv.rdzv_id;
	cmd.full_dma.match_bits = mb.raw;

	cmd.full_dma.user_ptr = (uint64_t)req;
	cmd.full_dma.remote_offset = req->recv.src_offset;

	cxi_build_dfa(nic, pid, rxc->pid_bits, pid_idx, &dfa, &idx_ext);
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.index_ext = idx_ext;

	if (req->data_len < req->recv.rdzv_mlen)
		cmd.full_dma.request_len = 0;
	else
		cmd.full_dma.request_len = req->data_len - req->recv.rdzv_mlen;

	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(req->recv.recv_md->md,
						 req->recv.recv_buf);
	cmd.full_dma.local_addr += req->recv.rdzv_mlen;

	fastlock_acquire(&rxc->tx_cmdq->lock);

	/* Issue Rendezvous Get command */
	ret = cxi_cq_emit_dma_f(rxc->tx_cmdq->dev_cmdq, &cmd.full_dma);
	if (ret) {
		CXIP_DBG("Failed to queue GET command: %d\n", ret);

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
 * cxip_notify_match_cb() - Callback function for match complete notifiction
 * Ack events.
 */
static int
cxip_notify_match_cb(struct cxip_req *req, const union c_event *event)
{
	CXIP_DBG("Match complete: %p\n", req);

	recv_req_report(req);

	if (req->recv.multi_recv)
		cxip_cq_req_free(req);
	else
		recv_req_complete(req);

	return FI_SUCCESS;
}

/*
 * cxip_notify_match() - Notify the initiator of a Send that the match is
 * complete at the target.
 *
 * A transaction ID corresponding to the matched Send request is sent back to
 * the initiator in the match_bits field of a zero-byte Put.
 */
static int cxip_notify_match(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	uint32_t pid_idx = rxc->domain->iface->dev->info.rdzv_get_idx;
	uint32_t init = event->tgt_long.initiator.initiator.process;
	uint32_t nic = CXI_MATCH_ID_EP(rxc->pid_bits, init);
	uint32_t pid = CXI_MATCH_ID_PID(rxc->pid_bits, init);
	union c_fab_addr dfa;
	uint8_t idx_ext;
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_ZBP,
	};
	union cxip_match_bits event_mb;
	union c_cmdu cmd = {};
	int ret;

	event_mb.raw = event->tgt_long.match_bits;
	mb.tx_id = event_mb.tx_id;

	cxi_build_dfa(nic, pid, rxc->pid_bits, pid_idx, &dfa, &idx_ext);

	cmd.c_state.event_send_disable = 1;
	cmd.c_state.index_ext = idx_ext;
	cmd.c_state.eq = rxc->recv_cq->evtq->eqn;

	fastlock_acquire(&rxc->tx_cmdq->lock);

	ret = cxip_cmdq_emit_c_state(rxc->tx_cmdq, &cmd.c_state);
	if (ret) {
		CXIP_DBG("Failed to issue C_STATE command: %d\n", ret);
		goto err_unlock;
	}

	memset(&cmd.idc_msg, 0, sizeof(cmd.idc_msg));
	cmd.idc_msg.dfa = dfa;
	cmd.idc_msg.match_bits = mb.raw;

	cmd.idc_msg.user_ptr = (uint64_t)req;

	ret = cxi_cq_emit_idc_msg(rxc->tx_cmdq->dev_cmdq, &cmd.idc_msg,
				  NULL, 0);
	if (ret) {
		CXIP_DBG("Failed to write IDC: %d\n", ret);

		/* Return error according to Domain Resource Management
		 */
		ret = -FI_EAGAIN;
		goto err_unlock;
	}

	cxi_cq_ring(rxc->tx_cmdq->dev_cmdq);

	fastlock_release(&rxc->tx_cmdq->lock);

	req->cb = cxip_notify_match_cb;

	CXIP_DBG("Queued match completion message: %p\n", req);

	return FI_SUCCESS;

err_unlock:
	fastlock_release(&rxc->tx_cmdq->lock);

	return ret;
}

/*
 * mrecv_req_oflow_event() - Set start and length uniquely for an unexpected
 * mrecv request.
 *
 * Overflow buffer events contain a start address representing the offset into
 * the Overflow buffer where data was written. When a unexpected header is
 * later matched to a multi-receive buffer in the priority list, The Put
 * Overflow event does not contain the offset into the Priority list buffer
 * where data should be copied. Software must track the the Priority list
 * buffer offset using ordered Put Overflow events.
 */
static int mrecv_req_put_bytes(struct cxip_req *req, uint32_t rlen)
{
	uintptr_t send_tail;
	uintptr_t mrecv_tail;

	send_tail = (uintptr_t)req->recv.recv_buf +
			req->recv.start_offset +
			rlen;
	mrecv_tail = (uintptr_t)req->recv.recv_buf + req->recv.ulen;

	if (send_tail > mrecv_tail)
		rlen -= send_tail - mrecv_tail;

	req->recv.start_offset += rlen;

	return rlen;
}

/*
 * cxip_ux_send() - Progress an unexpected Send after receiving matching Put
 * and Put and Put Overflow events.
 */
static int cxip_ux_send(struct cxip_req *match_req,
			struct cxip_req *oflow_req,
			const union c_event *put_event,
			uint64_t mrecv_start,
			uint32_t mrecv_len)
{
	struct cxip_oflow_buf *oflow_buf = oflow_req->oflow.oflow_buf;
	void *oflow_va;
	size_t oflow_bytes;
	union cxip_match_bits mb;
	enum fi_hmem_iface iface = match_req->recv.recv_md->info.iface;
	uint64_t device = match_req->recv.recv_md->info.device;
	struct iovec hmem_iov;
	ssize_t ret;

	assert(match_req->type == CXIP_REQ_RECV);

	if (match_req->recv.multi_recv) {
		if (put_event->tgt_long.rendezvous)
			match_req = rdzv_mrecv_req_event(match_req, put_event);
		else
			match_req = mrecv_req_dup(match_req);
		if (!match_req)
			return -FI_EAGAIN;

		/* Set start and length uniquely for an unexpected
		 * mrecv request.
		 */
		match_req->recv.recv_buf = (uint8_t *)
				match_req->recv.parent->recv.recv_buf +
				mrecv_start;
		match_req->buf = (uint64_t)match_req->recv.recv_buf;
		match_req->data_len = mrecv_len;
	} else {
		match_req->data_len = put_event->tgt_long.rlength;
		if (match_req->data_len > match_req->recv.ulen)
			match_req->data_len = match_req->recv.ulen;
	}

	recv_req_tgt_event(match_req, put_event);

	if (oflow_buf->type == CXIP_LE_TYPE_SINK) {
		/* For unexpected, long, eager messages, issue a Get to
		 * retrieve data from the initiator.
		 */
		ret = issue_rdzv_get(match_req);
		if (ret != FI_SUCCESS) {
			if (match_req->recv.multi_recv)
				cxip_cq_req_free(match_req);
			return -FI_EAGAIN;
		}

		CXIP_DBG("Issued Get, req: %p\n", match_req);
		return FI_SUCCESS;
	}

	/* Copy data out of overflow buffer. */
	oflow_va = (void *)CXI_IOVA_TO_VA(oflow_buf->md->md,
			put_event->tgt_long.start);
	oflow_bytes = MIN(put_event->tgt_long.mlength, match_req->data_len);

	hmem_iov.iov_base = match_req->recv.recv_buf;
	hmem_iov.iov_len = match_req->data_len;

	ret = ofi_copy_to_hmem_iov(iface, device, &hmem_iov, 1, 0, oflow_va,
				   oflow_bytes);
	assert(ret == oflow_bytes);

	oflow_req_put_bytes(oflow_req, put_event->tgt_long.mlength);

	/* Remaining unexpected rendezvous processing is deferred until RGet
	 * completes.
	 */
	if (put_event->tgt_long.rendezvous) {
		rdzv_recv_req_event(match_req);
		return FI_SUCCESS;
	}

	mb.raw = put_event->tgt_long.match_bits;

	/* Check if the initiator requires match completion guarantees.
	 * If so, notify the initiator that the match is now complete.
	 * Delay the Receive event until the notification is complete.
	 */
	if (mb.match_comp) {
		ret = cxip_notify_match(match_req, put_event);
		if (ret != FI_SUCCESS) {
			if (match_req->recv.multi_recv)
				cxip_cq_req_free(match_req);

			return -FI_EAGAIN;
		}

		return FI_SUCCESS;
	}

	recv_req_report(match_req);

	if (match_req->recv.multi_recv)
		cxip_cq_req_free(match_req);
	else
		recv_req_complete(match_req);

	return FI_SUCCESS;
}

/*
 * cxip_ux_send_zb() - Progress an unexpected zero-byte Send after receiving
 * a Put Overflow event.
 *
 * Zero-byte Put events for unexpected Sends are discarded. Progress the Send
 * using only the Overflow event. There is no Send data to be copied out.
 */
static int cxip_ux_send_zb(struct cxip_req *match_req,
			   const union c_event *oflow_event,
			   uint64_t mrecv_start)
{
	union cxip_match_bits mb;
	int ret;

	assert(oflow_event->hdr.event_type == C_EVENT_PUT_OVERFLOW);
	assert(!oflow_event->tgt_long.rlength);

	if (match_req->recv.multi_recv) {
		match_req = mrecv_req_dup(match_req);
		if (!match_req)
			return -FI_EAGAIN;

		match_req->buf = (uint64_t)
				match_req->recv.parent->recv.recv_buf +
				mrecv_start;
	}

	recv_req_tgt_event(match_req, oflow_event);

	match_req->data_len = 0;

	mb.raw = oflow_event->tgt_long.match_bits;

	/* Check if the initiator requires match completion guarantees.
	 * If so, notify the initiator that the match is now complete.
	 * Delay the Receive event until the notification is complete.
	 */
	if (mb.match_comp) {
		ret = cxip_notify_match(match_req, oflow_event);
		if (ret != FI_SUCCESS) {
			if (match_req->recv.multi_recv)
				cxip_cq_req_free(match_req);

			return -FI_EAGAIN;
		}

		return FI_SUCCESS;
	}

	recv_req_report(match_req);

	if (match_req->recv.multi_recv)
		cxip_cq_req_free(match_req);
	else
		recv_req_complete(match_req);

	return FI_SUCCESS;
}

static bool cxip_ux_is_onload_complete(struct cxip_req *req)
{
	return !req->search.puts_pending && req->search.complete;
}

/* Must hold rxc->rx_lock. */
static int cxip_oflow_process_put_event(struct cxip_rxc *rxc,
					struct cxip_req *req,
					const union c_event *event)
{
	int ret;
	struct cxip_deferred_event *def_ev;

	def_ev = match_put_event(rxc, event);
	if (!def_ev) {
		/* Put Overflow event pending. Defer this event until it
		 * arrives.
		 */
		def_ev = defer_put_event(rxc, req, event);
		if (!def_ev)
			return -FI_EAGAIN;

		return FI_SUCCESS;
	}

	CXIP_DBG("Overflow beat Put event: %p\n", def_ev->req);

	if (def_ev->ux_send) {
		/* Send was onloaded */
		def_ev->ux_send->oflow_req = req;
		def_ev->ux_send->put_ev = *event;
		def_ev->req->search.puts_pending--;
		CXIP_DBG("put complete: %p\n", def_ev->req);

		if (cxip_ux_is_onload_complete(def_ev->req))
			cxip_ux_onload_complete(def_ev->req);
	} else {
		ret = cxip_ux_send(def_ev->req, req, event, def_ev->mrecv_start,
				   def_ev->mrecv_len);
		if (ret != FI_SUCCESS)
			return -FI_EAGAIN;

		cxip_recv_req_dequeue_nolock(def_ev->req);
	}

	dlist_remove(&def_ev->rxc_entry);
	free(def_ev);

	return FI_SUCCESS;
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

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		ofi_atomic_inc32(&rxc->sink_le_linked);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		/* Long sink buffer was manually unlinked. */
		ofi_atomic_dec32(&rxc->sink_le_linked);

		/* Clean up overflow buffers */
		cxip_cq_req_free(req);
		return FI_SUCCESS;
	case C_EVENT_PUT:
		fastlock_acquire(&rxc->rx_lock);
		ret = cxip_oflow_process_put_event(rxc, req, event);
		fastlock_release(&rxc->rx_lock);

		return ret;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
			  event->hdr.event_type);
		return FI_SUCCESS;
	}
}

int cxip_rxc_eager_replenish(struct cxip_rxc *rxc);

static int cxip_recv_onload_flow_control(struct cxip_rxc *rxc)
{
	int ret;

	fastlock_acquire(&rxc->lock);

	assert(rxc->state == RXC_ENABLED ||
	       rxc->state == RXC_ONLOAD_FLOW_CONTROL ||
	       rxc->state == RXC_FLOW_CONTROL);

	/* Having flow control triggered while in flow control is a sign of LE
	 * exhaustion. Software endpoint mode is required to scale past hardware
	 * LE limit.
	 */
	if (rxc->state == RXC_FLOW_CONTROL) {
		CXIP_FATAL(FC_SW_EP_MSG);
	} else if (rxc->state == RXC_ONLOAD_FLOW_CONTROL) {
		fastlock_release(&rxc->lock);
		return FI_SUCCESS;
	}

	ret = cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq, C_PTLTE_DISABLED,
				 0);
	if (ret == FI_SUCCESS)
		rxc->state = RXC_ONLOAD_FLOW_CONTROL;

	fastlock_release(&rxc->lock);

	return ret;
}

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
	int ret = FI_SUCCESS;

	fastlock_acquire(&rxc->rx_lock);

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		if (cxi_event_rc(event) == C_RC_NO_SPACE) {
			CXIP_WARN("Failed to append oflow buffer due to LE exhaustion\n");

			ret = cxip_recv_onload_flow_control(rxc);
			if (ret == FI_SUCCESS) {
				/* Clean up dropped buffer */
				ofi_atomic_dec32(&rxc->oflow_bufs_submitted);
				oflow_req_put_bytes(req, rxc->oflow_buf_size);
			}
		} else {
			assert(cxi_event_rc(event) == C_RC_OK);

			CXIP_DBG("Eager buffer linked: %p\n", req);

			ofi_atomic_inc32(&rxc->oflow_bufs_linked);
		}

		fastlock_release(&rxc->rx_lock);

		return ret;
	case C_EVENT_UNLINK:
		assert(cxi_event_rc(event) == C_RC_OK);

		CXIP_DBG("Eager buffer unlinked (%s): %p\n",
			 event->tgt_long.auto_unlinked ? "auto" : "manual",
			 req);

		ofi_atomic_dec32(&rxc->oflow_bufs_submitted);
		ofi_atomic_dec32(&rxc->oflow_bufs_linked);

		/* Set hardware consumed if the overflow buffer has been
		 * manually unlinked or if the overflow buffer hardware consumed
		 * is zero. For the latter, if the buffer is automatically
		 * unlinked, the corresponding put event will always update the
		 * hardware consumed again.
		 */
		if (!event->tgt_long.auto_unlinked || !oflow_buf->hw_consumed)
			oflow_buf->hw_consumed = event->tgt_long.start -
				CXI_VA_TO_IOVA(oflow_buf->md->md,
					       oflow_buf->buf);

		/* Check if SW has consumed a matching count of bytes. */
		oflow_req_put_bytes(req, 0);

		if (event->tgt_long.auto_unlinked) {
			/* Replace the eager overflow buffer */
			cxip_rxc_eager_replenish(rxc);
		}

		fastlock_release(&rxc->rx_lock);

		return FI_SUCCESS;
	case C_EVENT_PUT:
		/* Put event handling is complicated. Handle below. */
		break;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
			  event->hdr.event_type);
		abort();
	}

	if (event->tgt_long.auto_unlinked)
		oflow_buf->hw_consumed = event->tgt_long.start -
			CXI_VA_TO_IOVA(oflow_buf->md->md, oflow_buf->buf)
			+ event->tgt_long.mlength;

	/* Drop all unexpected 0-byte Put events. */
	if (!event->tgt_long.rlength) {
		fastlock_release(&rxc->rx_lock);
		return FI_SUCCESS;
	}

	/* Handle Put events */
	ret = cxip_oflow_process_put_event(rxc, req, event);
	fastlock_release(&rxc->rx_lock);

	return ret;
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
	uint32_t le_flags;

	/* Match all eager, long sends */
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_RX
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.tagged = 1,
		.match_comp = 1,
	};

	dom = rxc->domain;

	/* Create an overflow buffer structure */
	oflow_buf = calloc(1, sizeof(*oflow_buf));
	if (!oflow_buf) {
		CXIP_WARN("Unable to allocate oflow buffer structure\n");
		return -FI_ENOMEM;
	}

	/* Allocate overflow data buffer */
	oflow_buf->buf = calloc(1, rxc->oflow_buf_size);
	if (!oflow_buf->buf) {
		CXIP_WARN("Unable to allocate oflow buffer\n");
		ret = -FI_ENOMEM;
		goto free_oflow;
	}

	/* Map overflow data buffer */
	ret = cxip_map(dom, (void *)oflow_buf->buf, rxc->oflow_buf_size,
		       &oflow_buf->md);
	if (ret) {
		CXIP_WARN("Failed to map oflow buffer: %d\n", ret);
		goto free_buf;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, NULL);
	if (!req) {
		CXIP_WARN("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto oflow_unmap;
	}

	req->cb = cxip_oflow_cb;
	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = oflow_buf;
	req->type = CXIP_REQ_OFLOW;

	le_flags = C_LE_MANAGE_LOCAL | C_LE_NO_TRUNCATE |
		   C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_PUT;

	/* Issue Append command */
	ret = cxip_pte_append(rxc->rx_pte,
			      CXI_VA_TO_IOVA(oflow_buf->md->md,
					     oflow_buf->buf),
			      rxc->oflow_buf_size, oflow_buf->md->md->lac,
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY,
			      rxc->rdzv_threshold + rxc->rdzv_get_min,
			      le_flags, NULL, rxc->rx_cmdq, true);
	if (ret) {
		CXIP_WARN("Failed to write Append command: %d\n", ret);
		goto oflow_req_free;
	}

	/* Initialize oflow_buf structure */
	dlist_insert_tail(&oflow_buf->list, &rxc->oflow_bufs);
	oflow_buf->rxc = rxc;
	oflow_buf->buffer_id = req->req_id;
	oflow_buf->type = CXIP_LE_TYPE_RX;

	ofi_atomic_inc32(&rxc->oflow_bufs_submitted);
	ofi_atomic_inc32(&rxc->oflow_bufs_in_use);
	CXIP_DBG("Eager buffer created: %p\n", req);

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
			CXIP_WARN("Failed to append oflow buffer: %d\n",
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
		ret = cxip_pte_unlink(rxc->rx_pte, C_PTL_LIST_OVERFLOW,
				      oflow_buf->buffer_id, rxc->rx_cmdq);
		if (ret != FI_SUCCESS) {
			/* TODO handle error */
			CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
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
	int ret;
	struct cxip_req *req;
	uint32_t le_flags;

	/* Match all eager, long sends */
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_SINK,
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.tagged = 1,
		.match_comp = 1,
	};

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, NULL);
	if (!req) {
		CXIP_WARN("Failed to allocate UX request\n");
		return -FI_ENOMEM;
	}

	le_flags = C_LE_MANAGE_LOCAL | C_LE_UNRESTRICTED_BODY_RO |
		   C_LE_UNRESTRICTED_END_RO | C_LE_OP_PUT;

	ret = cxip_pte_append(rxc->rx_pte, 0, 0, 0,
			      C_PTL_LIST_OVERFLOW, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0,  le_flags, NULL,
			      rxc->rx_cmdq, true);
	if (ret) {
		CXIP_WARN("Failed to write UX Append command: %d\n", ret);
		goto req_free;
	}

	/* Initialize oflow_buf structure */
	rxc->sink_le.type = CXIP_LE_TYPE_SINK;
	rxc->sink_le.rxc = rxc;
	rxc->sink_le.buffer_id = req->req_id;

	req->type = CXIP_REQ_OFLOW;
	req->oflow.rxc = rxc;
	req->oflow.oflow_buf = &rxc->sink_le;
	req->cb = cxip_oflow_sink_cb;

	return FI_SUCCESS;

req_free:
	cxip_cq_req_free(req);

	return ret;
}

/*
 * cxip_rxc_sink_fini() - Tear down RXC sink buffer.
 */
static int cxip_rxc_sink_fini(struct cxip_rxc *rxc)
{
	int ret;

	ret = cxip_pte_unlink(rxc->rx_pte, C_PTL_LIST_OVERFLOW,
			      rxc->sink_le.buffer_id, rxc->rx_cmdq);
	if (ret) {
		/* TODO handle error */
		CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
	}

	return ret;
}

static void report_send_completion(struct cxip_req *req, bool sw_cntr);

/*
 * cxip_zbp_cb() - Process zero-byte Put events.
 *
 * Zero-byte Puts (ZBP) are used to transfer small messages without consuming
 * buffers outside of the EQ. ZBPs are currently only used for match complete
 * messages.
 */
static int
cxip_zbp_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_txc *txc = req->oflow.txc;
	struct cxip_req *put_req;
	union cxip_match_bits mb;
	int event_rc;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		ofi_atomic_inc32(&txc->zbp_le_linked);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		/* TODO Handle append errors. */
		assert(cxi_event_rc(event) == C_RC_OK);

		/* Zero-byte Put LE was manually unlinked. */
		ofi_atomic_dec32(&txc->zbp_le_linked);

		/* Clean up overflow buffers */
		cxip_cq_req_free(req);
		return FI_SUCCESS;
	case C_EVENT_PUT:
		mb.raw = event->tgt_long.match_bits;
		put_req = cxip_tx_id_lookup(txc->ep_obj, mb.tx_id);
		if (!put_req) {
			CXIP_WARN("Failed to find TX ID: %d\n", mb.tx_id);
			return FI_SUCCESS;
		}

		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK)
			CXIP_WARN("ZBP error: %p rc: %s\n",
				  put_req, cxi_rc_to_str(event_rc));
		else
			CXIP_DBG("ZBP received: %p rc: %s\n",
				 put_req, cxi_rc_to_str(event_rc));

		cxip_tx_id_free(txc->ep_obj, mb.tx_id);

		/* The unexpected message has been matched. Generate a
		 * completion event. The ZBP event is guaranteed to arrive
		 * after the eager Send Ack, so the transfer is always done at
		 * this point.
		 *
		 * If MATCH_COMPLETE was requested, software must manage
		 * counters.
		 */
		report_send_completion(put_req, true);

		cxip_send_req_dequeue(put_req->send.txc, put_req);

		ofi_atomic_dec32(&put_req->send.txc->otx_reqs);
		cxip_cq_req_free(put_req);

		return FI_SUCCESS;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
			  event->hdr.event_type);
		return FI_SUCCESS;
	}
}

/*
 * cxip_txc_zbp_init() - Initialize zero-byte Put LE.
 */
int cxip_txc_zbp_init(struct cxip_txc *txc)
{
	int ret;
	struct cxip_req *req;
	uint32_t le_flags;
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_ZBP,
	};
	union cxip_match_bits ib = {
		.tag = ~0,
		.tx_id = ~0,
		.tagged = 1,
		.match_comp = 1,
	};

	/* Populate request */
	req = cxip_cq_req_alloc(txc->send_cq, 1, NULL);
	if (!req) {
		CXIP_WARN("Failed to allocate request\n");
		return -FI_ENOMEM;
	}

	le_flags = C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_PUT;

	ret = cxip_pte_append(txc->rdzv_pte, 0, 0, 0,
			      C_PTL_LIST_PRIORITY, req->req_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, le_flags, NULL,
			      txc->rx_cmdq, true);
	if (ret) {
		CXIP_WARN("Failed to write Append command: %d\n", ret);
		goto req_free;
	}

	/* Initialize oflow_buf structure */
	txc->zbp_le.type = CXIP_LE_TYPE_ZBP;
	txc->zbp_le.txc = txc;
	txc->zbp_le.buffer_id = req->req_id;

	req->type = CXIP_REQ_OFLOW;
	req->oflow.txc = txc;
	req->oflow.oflow_buf = &txc->zbp_le;
	req->cb = cxip_zbp_cb;

	/* Wait for link */
	do {
		sched_yield();
		cxip_cq_progress(txc->send_cq);
	} while (!ofi_atomic_get32(&txc->zbp_le_linked));

	CXIP_DBG("ZBP LE linked: %p\n", txc);

	return FI_SUCCESS;

req_free:
	cxip_cq_req_free(req);

	return ret;
}

/*
 * cxip_txc_zbp_fini() - Tear down zero-byte Put LE.
 */
int cxip_txc_zbp_fini(struct cxip_txc *txc)
{
	int ret;

	ret = cxip_pte_unlink(txc->rdzv_pte, C_PTL_LIST_PRIORITY,
			      txc->zbp_le.buffer_id, txc->rx_cmdq);
	if (ret) {
		/* TODO handle error */
		CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
	}

	/* Wait for unlink */
	do {
		sched_yield();
		cxip_cq_progress(txc->send_cq);
	} while (ofi_atomic_get32(&txc->zbp_le_linked));

	CXIP_DBG("ZBP LE unlinked: %p\n", txc);

	return ret;
}

/*
 * cxip_rxc_oflow_init() - Initialize overflow buffers used for messaging.
 *
 * Must be called with the RX PtlTE disabled.
 */
int cxip_rxc_oflow_init(struct cxip_rxc *rxc)
{
	int ret;

	ret = cxip_rxc_eager_replenish(rxc);
	if (ret) {
		CXIP_WARN("cxip_rxc_eager_replenish failed: %d\n", ret);
		return ret;
	}

	ret = cxip_rxc_sink_init(rxc);
	if (ret) {
		CXIP_WARN("cxip_rxc_sink_init failed: %d\n", ret);
		cxip_rxc_eager_fini(rxc);
		return ret;
	}

	/* Wait for Overflow buffers to be linked. */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&rxc->oflow_bufs_linked) <
		 rxc->oflow_bufs_max ||
		 !ofi_atomic_get32(&rxc->sink_le_linked));

	return FI_SUCCESS;
}

/*
 * cxip_rxc_oflow_fini() - Finalize overflow buffers used for messaging.
 *
 * Must be called with the RX PtlTE disabled.
 */
void cxip_rxc_oflow_fini(struct cxip_rxc *rxc)
{
	int ret;
	struct cxip_deferred_event *def_ev;
	struct dlist_entry *tmp;
	int def_events = 0;

	/* Clean up unexpected Put records. The PtlTE is disabled, so no more
	 * events can be expected.
	 */
	dlist_foreach_container_safe(&rxc->deferred_events,
				     struct cxip_deferred_event,
				     def_ev, rxc_entry, tmp) {
		/* Dropping the last reference will cause the oflow_buf to be
		 * removed from the RXC list and freed.
		 */
		if (def_ev->req->oflow.oflow_buf->type == CXIP_LE_TYPE_RX)
			oflow_req_put_bytes(def_ev->req,
					    def_ev->ev.tgt_long.mlength);

		dlist_remove(&def_ev->rxc_entry);
		free(def_ev);
		def_events++;
	}

	if (def_events)
		CXIP_DBG("Freed %d deferred event(s)\n", def_events);

	ret = cxip_rxc_sink_fini(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_rxc_sink_fini() returned: %d\n", ret);
		return;
	}

	ret = cxip_rxc_eager_fini(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_rxc_eager_fini() returned: %d\n", ret);
		return;
	}

	/* Wait for all overflow buffers to be unlinked */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&rxc->oflow_bufs_linked) ||
		 ofi_atomic_get32(&rxc->sink_le_linked));

	if (ofi_atomic_get32(&rxc->oflow_bufs_in_use))
		CXIP_WARN("Leaked %d overflow buffers\n",
			  ofi_atomic_get32(&rxc->oflow_bufs_in_use));
}

/*
 * cxip_recv_rdzv_cb() - Progress rendezvous receive events.
 *
 * Handle rendezvous target events. All target events which are related to an
 * offloaded rendezvous Put operation have the rendezvous field set.
 *
 * Note that Reply events that were generated from a SW-issued Get will not
 * have the rendezvous bit set.
 *
 * There is some complexity in how the receive buffer start pointer (for
 * multi-receives) and receive length are set when using the rendezvous
 * protocol. The method for calculating these for each scenario is below.
 *
 * Expected Receives:
 *	Calculate receive length using Rendezvous event. It needs to be
 *	available for SW issued Gets.
 *
 * Unexpected Receives:
 *	Calculate receive length using Put Overflow event. It needs to be
 *	available for copying eager data into the user buffer. Note that
 *	receive length is set twice for a UX receive using both Rendezvous and
 *	Put Overflow events.
 *
 * Expected Multi-Receives:
 *	Use start, mlength and rlength in the Rendezvous event.
 *
 * Unexpected Multi-Receives:
 *	Track user buffer offset in software using the order of Put Overflow
 *	events.
 */
static int cxip_recv_rdzv_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	struct cxip_deferred_event *def_ev;
	int ret;

	switch (event->hdr.event_type) {
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
		def_ev = match_put_event(rxc, event);
		if (!def_ev) {
			/* Put event pending. Defer this event until it
			 * arrives.
			 */
			def_ev = defer_put_event(rxc, req, event);
			if (def_ev) {
				/* Calculate start, length */
				def_ev->mrecv_start = req->recv.start_offset;
				def_ev->mrecv_len = mrecv_req_put_bytes(req,
						event->tgt_long.rlength);
			}

			fastlock_release(&rxc->rx_lock);

			return def_ev ? FI_SUCCESS : -FI_EAGAIN;
		}

		CXIP_DBG("Matched deferred event: %p\n", def_ev);

		/* Calculate start, length */
		def_ev->mrecv_start = req->recv.start_offset;
		def_ev->mrecv_len = mrecv_req_put_bytes(req,
				event->tgt_long.rlength);

		ret = cxip_ux_send(req, def_ev->req, &def_ev->ev,
				   def_ev->mrecv_start, def_ev->mrecv_len);
		if (ret == FI_SUCCESS) {
			cxip_recv_req_dequeue_nolock(req);

			dlist_remove(&def_ev->rxc_entry);
			free(def_ev);
		} else {
			/* undo mrecv_req_put_bytes() */
			req->recv.start_offset -= def_ev->mrecv_len;
		}

		fastlock_release(&rxc->rx_lock);

		return ret;
	case C_EVENT_PUT:
		/* Eager data was delivered directly to the user buffer. */
		cxip_recv_req_dequeue(req);

		if (req->recv.multi_recv) {
			req = rdzv_mrecv_req_event(req, event);
			if (!req)
				return -FI_EAGAIN;

			/* Set start pointer and data_len using Rendezvous or
			 * Put Overflow event (depending on if message was
			 * unexpected).
			 */
		}

		recv_req_tgt_event(req, event);

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	case C_EVENT_RENDEZVOUS:
		if (req->recv.multi_recv) {
			req = rdzv_mrecv_req_event(req, event);
			if (!req)
				return -FI_EAGAIN;

			/* Use Rendezvous event to set start pointer and
			 * data_len for expected Sends.
			 */
			struct cxip_req *parent = req->recv.parent;
			uintptr_t rtail;
			uintptr_t mrecv_tail;

			req->buf = CXI_IOVA_TO_VA(
					parent->recv.recv_md->md,
					event->tgt_long.start) -
					event->tgt_long.mlength;
			req->recv.recv_buf = (void *)req->buf;
			rtail = req->buf + event->tgt_long.rlength;
			mrecv_tail = (uint64_t)parent->recv.recv_buf +
				parent->recv.ulen;

			req->data_len = event->tgt_long.rlength;
			if (rtail > mrecv_tail)
				req->data_len -= rtail - mrecv_tail;
		} else {
			req->data_len = event->tgt_long.rlength;
			if (req->data_len > req->recv.ulen)
				req->data_len = req->recv.ulen;
		}

		recv_req_tgt_event(req, event);

		if (!event->tgt_long.get_issued) {
			int ret = issue_rdzv_get(req);
			if (ret != FI_SUCCESS) {
				/* Undo multi-recv event processing. */
				if (req->recv.multi_recv &&
				    !req->recv.rdzv_events) {
					dlist_remove(&req->recv.children);
					cxip_cq_req_free(req);
				}
				return -FI_EAGAIN;
			}

			CXIP_DBG("Software issued Get, req: %p\n", req);
		}

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	case C_EVENT_REPLY:
		/* If mrecv, look up the correct child request. */
		if (req->recv.multi_recv) {
			req = rdzv_mrecv_req_event(req, event);
			if (!req)
				return -FI_EAGAIN;
		}

		/* Rendezvous Get completed. Complete the request. */
		req->recv.rc = cxi_init_event_rc(event);

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
		return FI_SUCCESS;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
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
	struct cxip_deferred_event *def_ev;
	bool rdzv = false;

	/* Transition into onload and flow control if an append fails. */
	if (event->hdr.event_type == C_EVENT_LINK) {
		if (cxi_tgt_event_rc(event) != C_RC_NO_SPACE)
			CXIP_FATAL("Unexpected link event rc: %d\n",
				   cxi_tgt_event_rc(event));

		CXIP_WARN("Failed to append user buffer due to LE exhaustion\n");

		ret = cxip_recv_onload_flow_control(rxc);
		if (ret == FI_SUCCESS)
			cxip_recv_req_dropped(req);

		return ret;
	}

	/* All events related to an offloaded rendezvous receive will be
	 * handled by cxip_recv_rdzv_cb(). Those events are identified by the
	 * event rendezvous field. One exception is a Reply event generated
	 * from a SW-issued Get. When such an event is generated, the request
	 * will have already processed a Rendezvous event. If the rendezvous
	 * field is not set, but the rdzv_events count is elevated, this must
	 * be a SW-issued Reply event.
	 */
	if (event->hdr.event_type == C_EVENT_REPLY)
		rdzv = (event->init_short.rendezvous || req->recv.rdzv_events);
	else
		rdzv = event->tgt_long.rendezvous;

	if (rdzv)
		return cxip_recv_rdzv_cb(req, event);

	switch (event->hdr.event_type) {
	case C_EVENT_UNLINK:
		if (!event->tgt_long.auto_unlinked) {
			req->recv.unlinked = true;
            cxip_recv_req_dequeue(req);
			recv_req_report(req);
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

		/* Unexpected 0-byte Put events are dropped. Skip matching. */
		if (!event->tgt_long.rlength) {
			ret = cxip_ux_send_zb(req, event,
					      req->recv.start_offset);
			if (ret == FI_SUCCESS)
				cxip_recv_req_dequeue_nolock(req);

			fastlock_release(&rxc->rx_lock);
			return ret;
		}

		/* Check for a previously received unexpected Put event */
		def_ev = match_put_event(rxc, event);
		if (!def_ev) {
			/* Put event pending. Defer this event until it
			 * arrives.
			 */
			def_ev = defer_put_event(rxc, req, event);
			if (def_ev) {
				/* Calculate start, length */
				def_ev->mrecv_start = req->recv.start_offset;
				def_ev->mrecv_len = mrecv_req_put_bytes(req,
						event->tgt_long.rlength);
			}

			fastlock_release(&rxc->rx_lock);

			return def_ev ? FI_SUCCESS : -FI_EAGAIN;
		}

		/* Calculate start, length */
		def_ev->mrecv_start = req->recv.start_offset;
		def_ev->mrecv_len = mrecv_req_put_bytes(req,
				event->tgt_long.rlength);

		ret = cxip_ux_send(req, def_ev->req, &def_ev->ev,
				   def_ev->mrecv_start, def_ev->mrecv_len);
		if (ret == FI_SUCCESS) {
			dlist_remove(&def_ev->rxc_entry);
			free(def_ev);

			cxip_recv_req_dequeue_nolock(req);
		} else {
			/* undo mrecv_req_put_bytes() */
			req->recv.start_offset -= def_ev->mrecv_len;
		}

		fastlock_release(&rxc->rx_lock);

		return ret;
	case C_EVENT_PUT:
		/* Data was delivered directly to the user buffer. Complete the
		 * request.
		 */
		cxip_recv_req_dequeue(req);

		if (req->recv.multi_recv) {
			req = mrecv_req_dup(req);
			if (!req)
				return -FI_EAGAIN;
			recv_req_tgt_event(req, event);

			req->buf = (uint64_t)(CXI_IOVA_TO_VA(
					req->recv.recv_md->md,
					event->tgt_long.start));
			req->data_len = event->tgt_long.mlength;

			recv_req_report(req);
			cxip_cq_req_free(req);
		} else {
			req->data_len = event->tgt_long.mlength;
			recv_req_tgt_event(req, event);
			recv_req_report(req);
			recv_req_complete(req);
		}
		return FI_SUCCESS;
	case C_EVENT_REPLY:
		/* Long-send Get completed. Complete the request. */
		req->recv.rc = cxi_init_event_rc(event);

		if (req->recv.multi_recv) {
			recv_req_report(req);
			cxip_cq_req_free(req);
		} else {
			/* Complete receive request. */
			recv_req_report(req);
			recv_req_complete(req);
		}
		return FI_SUCCESS;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
			  event->hdr.event_type);
	}

	return FI_SUCCESS;
}

/*
 * cxip_recv_cancel() - Cancel outstanding receive request.
 */
int cxip_recv_cancel(struct cxip_req *req)
{
	int ret;
	struct cxip_rxc *rxc = req->recv.rxc;

	ret = cxip_pte_unlink(rxc->rx_pte, C_PTL_LIST_PRIORITY,
			      req->req_id, rxc->rx_cmdq);
	if (ret == FI_SUCCESS)
		req->recv.canceled = true;

	return ret;
}

/*
 * cxip_recv_reenable() - Attempt to re-enable the RX queue.
 *
 * Called by disabled EP ready to re-enable.
 *
 * Determine if the RX queue can be re-enabled and perform a state change
 * command if necessary. The Endpoint must receive dropped Send notifications
 * from all peers who experienced drops before re-enabling the RX queue.
 *
 * Caller must hold rxc->lock.
 */
int cxip_recv_reenable(struct cxip_rxc *rxc)
{
	int total_drops = -1;
	struct cxi_pte_status pte_status = {};
	struct cxip_fc_drops *fc_drops;
	int ret __attribute__((unused));

	/* Check if we're ready to re-enable the RX queue */
	dlist_foreach_container(&rxc->fc_drops, struct cxip_fc_drops,
				fc_drops, rxc_entry) {
		total_drops += fc_drops->drops;
	}

	ret = cxil_pte_status(rxc->rx_pte->pte, &pte_status);
	assert(!ret);

	CXIP_DBG("Processed %d/%d drops\n",
		 total_drops+1, pte_status.drop_count+1);

	if (total_drops != pte_status.drop_count)
		return -FI_EAGAIN;

	CXIP_DBG("Re-enabling PTE\n");

	ret = cxip_rxc_msg_enable(rxc, total_drops);
	assert(ret == FI_SUCCESS);

	return FI_SUCCESS;
}

/*
 * cxip_fc_resume_cb() - Process FC resume completion events.
 */
int cxip_fc_resume_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_fc_drops *fc_drops = container_of(req,
			struct cxip_fc_drops, req);
	struct cxip_rxc *rxc = fc_drops->rxc;
	int ret = FI_SUCCESS;

	fastlock_acquire(&rxc->lock);

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		switch (cxi_event_rc(event)) {
		case C_RC_OK:
			free(fc_drops);
			break;

		/* This error occurs when the target's control event queue has
		 * run out of space. Since the target should be processing the
		 * event queue, it is safe to replay messages until C_RC_OK is
		 * returned.
		 */
		case C_RC_ENTRY_NOT_FOUND:
			CXIP_WARN("Target dropped flow control message... replaying message\n");
			ret = cxip_ctrl_msg_send(req);
			break;
		default:
			CXIP_FATAL("Unexpected event rc: %d\n",
				   cxi_event_rc(event));
		}
		break;
	default:
		CXIP_FATAL("Unexpected event type: %d\n",
			   event->hdr.event_type);
	}

	fastlock_release(&rxc->lock);

	return ret;
}

/*
 * cxip_fc_process_drops() - Process a dropped Send notification from a peer.
 *
 * Called by disabled EP waiting to re-enable.
 *
 * When a peer detects dropped Sends it follows up by sending a message to the
 * disabled Endpoint indicating the number of drops experienced. The disabled
 * Endpoint peer must count all drops before re-enabling its RX queue.
 */
int cxip_fc_process_drops(struct cxip_ep_obj *ep_obj, uint8_t rxc_id,
			  uint32_t nic_addr, uint32_t pid, uint8_t txc_id,
			  uint16_t drops)
{
	struct cxip_rxc *rxc = ep_obj->rxcs[rxc_id];
	struct cxip_fc_drops *fc_drops;
	int ret __attribute__((unused));

	fc_drops = calloc(1, sizeof(*fc_drops));
	if (!fc_drops) {
		CXIP_DBG("Failed to allocate drops\n");
		return -FI_ENOMEM;
	}

	fc_drops->rxc = rxc;
	fc_drops->nic_addr = nic_addr;
	fc_drops->pid = pid;
	fc_drops->txc_id = txc_id;
	fc_drops->rxc_id = rxc_id;
	fc_drops->drops = drops;

	fc_drops->req.send.nic_addr = nic_addr;
	fc_drops->req.send.pid = pid;
	fc_drops->req.send.mb.txc_id = txc_id;
	fc_drops->req.send.mb.rxc_id = rxc_id;
	fc_drops->req.send.mb.drops = drops;

	fc_drops->req.send.mb.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG;
	fc_drops->req.send.mb.ctrl_msg_type = CXIP_CTRL_MSG_FC_RESUME;
	fc_drops->req.cb = cxip_fc_resume_cb;
	fc_drops->req.ep_obj = rxc->ep_obj;

	fastlock_acquire(&rxc->lock);

	dlist_insert_tail(&fc_drops->rxc_entry, &rxc->fc_drops);

	CXIP_DBG("Processed drops: %d NIC: %#x TXC: %d RXC: %p\n",
		 drops, nic_addr, txc_id, rxc);

	/* Wait until search and delete completes before attempting to
	 * re-enable.
	 */
	if (rxc->state == RXC_FLOW_CONTROL) {
		ret = cxip_recv_reenable(rxc);
		assert(ret == FI_SUCCESS || ret == -FI_EAGAIN);
	}

	fastlock_release(&rxc->lock);

	return FI_SUCCESS;
}

/*
 * cxip_recv_replay() - Replay dropped Receive requests.
 *
 * When no LE is available while processing an Append command, the command is
 * dropped and future appends are disabled. After all outstanding commands are
 * dropped and resources are recovered, replayed all Receive requests in order.
 *
 * Caller must hold rxc->lock.
 */
static int cxip_recv_replay(struct cxip_rxc *rxc)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;
	bool restart_seq = true;
	int ret;

	ret = cxip_rxc_eager_replenish(rxc);
	if (ret != FI_SUCCESS)
		CXIP_WARN("cxip_rxc_eager_replenish failed: %d\n", ret);

	dlist_foreach_container_safe(&rxc->replay_queue,
				     struct cxip_req, req,
				     recv.rxc_entry, tmp) {
		dlist_remove(&req->recv.rxc_entry);

		CXIP_DBG("Replaying: %p\n", req);

		/* Since the RXC and PtlTE are in a controlled state and no new
		 * user receives are being posted, it is safe to ignore the RXC
		 * state when replaying failed user posted receives.
		 */
		ret = cxip_recv_req_queue(req, false, restart_seq);

		/* Match made in software? */
		if (ret == -FI_EALREADY)
			continue;

		/* TODO: Low memory or full CQ during SW matching would cause
		 * -FI_EAGAIN to be seen here.
		 */
		assert(ret == FI_SUCCESS);

		restart_seq = false;
	}

	return FI_SUCCESS;
}

/*
 * cxip_recv_resume() - Send a resume message to all peers who reported dropped
 * Sends.
 *
 * Called by disabled EP after re-enable.
 *
 * After counting all dropped sends targeting a disabled RX queue and
 * re-enabling the queue, notify all peers who experienced dropped Sends so
 * they can be replayed.
 *
 * Caller must hold rxc->lock.
 */
int cxip_recv_resume(struct cxip_rxc *rxc)
{
	struct cxip_fc_drops *fc_drops;
	struct dlist_entry *tmp;
	int ret;

	dlist_foreach_container_safe(&rxc->fc_drops,
				     struct cxip_fc_drops, fc_drops,
				     rxc_entry, tmp) {
		ret = cxip_ctrl_msg_send(&fc_drops->req);
		if (ret)
			return ret;

		dlist_remove(&fc_drops->rxc_entry);
	}

	return FI_SUCCESS;
}

/*
 * cxip_ux_onload_complete() - Complete a UX on-load operation.
 *
 * All unexpected message headers have been dequeued from HW, replay failed
 * Append commands and re-enable the PTE.
 */
static void cxip_ux_onload_complete(struct cxip_req *req)
{
	struct cxip_rxc *rxc = req->search.rxc;
	int ret __attribute__((unused));

	rxc->state = RXC_FLOW_CONTROL;

	CXIP_DBG("UX onload complete (sw_ux_list_len: %d): %p\n",
		 rxc->sw_ux_list_len, rxc);

	ofi_atomic_dec32(&rxc->orx_reqs);
	cxip_cq_req_free(req);

	ret = cxip_recv_replay(rxc);
	assert(ret == FI_SUCCESS || ret == -FI_EAGAIN);

	/* Check if the RX queue can be re-enabled now */
	ret = cxip_recv_reenable(rxc);
	assert(ret == FI_SUCCESS || ret == -FI_EAGAIN);
}

/*
 * cxip_ux_onload_cb() - Process SEARCH_AND_DELETE command events.
 */
static int cxip_ux_onload_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc *rxc = req->search.rxc;
	struct cxip_deferred_event *def_ev;
	struct cxip_ux_send *ux_send;

	fastlock_acquire(&rxc->lock);

	assert(rxc->state == RXC_ONLOAD_FLOW_CONTROL);

	switch (event->hdr.event_type) {
	case C_EVENT_PUT_OVERFLOW:
		assert(cxi_event_rc(event) == C_RC_OK);

		ux_send = calloc(1, sizeof(*ux_send));
		if (!ux_send)
			return -FI_EAGAIN;

		/* Zero-byte unexpected onloads require special handling since
		 * no deferred structure would be allocated.
		 */
		if (event->tgt_long.rlength) {
			def_ev = match_put_event(rxc, event);
			if (!def_ev) {
				def_ev = defer_put_event(rxc, req, event);
				if (!def_ev) {
					fastlock_release(&rxc->lock);
					free(ux_send);
					return -FI_EAGAIN;
				}

				/* Gather Put events later */
				def_ev->ux_send = ux_send;
				req->search.puts_pending++;
			} else {
				ux_send->oflow_req = def_ev->req;
				ux_send->put_ev = def_ev->ev;

				dlist_remove(&def_ev->rxc_entry);
				free(def_ev);
			}
		} else {
			ux_send->put_ev = *event;
		}

		dlist_insert_tail(&ux_send->rxc_entry, &rxc->sw_ux_list);
		rxc->sw_ux_list_len++;

		CXIP_DBG("Onloaded Send: %p\n", ux_send);

		break;
	case C_EVENT_SEARCH:
		req->search.complete = true;

		if (cxip_ux_is_onload_complete(req))
			cxip_ux_onload_complete(req);

		break;
	default:
		CXIP_WARN("Unexpected event type: %d\n",
			  event->hdr.event_type);
	}

	fastlock_release(&rxc->lock);

	return FI_SUCCESS;
}

/*
 * cxip_ux_onload() - Issue SEARCH_AND_DELETE command to on-load unexpected
 * Send headers queued on the RXC message queue.
 *
 * Caller must hold rxc->lock.
 */
static int cxip_ux_onload(struct cxip_rxc *rxc)
{
	struct cxip_req *req;
	union c_cmdu cmd = {};
	int ret;

	assert(rxc->state == RXC_ONLOAD_FLOW_CONTROL);

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, NULL);
	if (!req) {
		CXIP_WARN("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		return ret;
	}
	ofi_atomic_inc32(&rxc->orx_reqs);

	req->cb = cxip_ux_onload_cb;
	req->type = CXIP_REQ_SEARCH;
	req->search.rxc = rxc;

	cmd.command.opcode = C_CMD_TGT_SEARCH_AND_DELETE;
	cmd.target.ptl_list = C_PTL_LIST_UNEXPECTED;
	cmd.target.ptlte_index = rxc->rx_pte->pte->ptn;
	cmd.target.buffer_id = req->req_id;
	cmd.target.length = -1U;
	cmd.target.ignore_bits = -1UL;
	cmd.target.match_id = CXI_MATCH_ID_ANY;

	fastlock_acquire(&rxc->rx_cmdq->lock);

	ret = cxi_cq_emit_target(rxc->rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		CXIP_DBG("Failed to write Search command: %d\n", ret);

		ofi_atomic_dec32(&rxc->orx_reqs);
		cxip_cq_req_free(req);

		fastlock_release(&rxc->rx_cmdq->lock);

		return -FI_EAGAIN;
	}

	cxi_cq_ring(rxc->rx_cmdq->dev_cmdq);

	fastlock_release(&rxc->rx_cmdq->lock);

	return FI_SUCCESS;
}

/*
 * cxip_recv_pte_cb() - Process receive PTE state change events.
 */
void cxip_recv_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state)
{
	struct cxip_rxc *rxc = (struct cxip_rxc *)pte->ctx;
	int ret __attribute__((unused));

	fastlock_acquire(&rxc->lock);

	switch (state) {
	case C_PTLTE_ENABLED:
		assert(rxc->state == RXC_FLOW_CONTROL ||
		       rxc->state == RXC_DISABLED);

		CXIP_DBG("Enabled Receive PTE: %p\n", rxc);

		/* Progress the control EP until all control messages can be
		 * successfully queued.
		 */
		if (rxc->state == RXC_FLOW_CONTROL) {
			/* Progress the control TX queues until all resume
			 * control messages can be successfully queued.
			 */
			while ((ret = cxip_recv_resume(rxc)) == -FI_EAGAIN) {
				fastlock_release(&rxc->lock);
				cxip_ep_tx_ctrl_progress(rxc->ep_obj);
				fastlock_acquire(&rxc->lock);
			}

			assert(ret == FI_SUCCESS);
		}

		rxc->state = RXC_ENABLED;
		break;

	case C_PTLTE_DISABLED:
		if (rxc->state == RXC_DISABLED)
			break;

		/* Incorrect drop count was used. Another attempt will be made
		 * when a peer sends a sideband drop message.
		 */
		if (rxc->state == RXC_FLOW_CONTROL) {
			CXIP_WARN("Failed to reenable PtlTE while in flow control\n");
			break;
		}

		assert(rxc->state == RXC_ENABLED ||
		       rxc->state == RXC_ONLOAD_FLOW_CONTROL);

		rxc->state = RXC_ONLOAD_FLOW_CONTROL;

		CXIP_DBG("Flow control detected: %p\n", rxc);

		ret = cxip_ux_onload(rxc);
		assert(ret == FI_SUCCESS);

		CXIP_DBG("Started onload\n");

		break;
	default:
		CXIP_FATAL("Unexpected state received: %u\n", state);
	}

	fastlock_release(&rxc->lock);
}

#if 1
/*
 * tag_match() - Compare UX Send tag and Receive tags in SW.
 */
static bool tag_match(uint64_t init_mb, uint64_t mb, uint64_t ib)
{
	return !((init_mb ^ mb) & ~ib);
}

/*
 * tag_match() - Compare UX Send initiator and Receive initiator in SW.
 */
static bool init_match(uint32_t init, uint32_t match_id)
{
	if (match_id == CXI_MATCH_ID_ANY)
		return true;

	return init == match_id;
}

/*
 * cxip_recv_sw_match() - Progress the SW Receive match.
 *
 * Progress the operation which matched in SW.
 */
static int cxip_recv_sw_match(struct cxip_req *req,
			      struct cxip_ux_send *ux_send)
{
	int ret;
	uint64_t mrecv_start;
	uint32_t mrecv_len;
	bool req_done = true;

	mrecv_start = req->recv.start_offset;
	mrecv_len = mrecv_req_put_bytes(req, ux_send->put_ev.tgt_long.rlength);

	if (req->recv.multi_recv && mrecv_len < req->recv.mrecv_bytes)
		req_done = false;

	if (ux_send->put_ev.tgt_long.rendezvous) {
		/* TODO read ULE list before onloaded messages to find RPut
		 * remote offset.
		 */
		CXIP_FATAL("RPut onload not supported\n");

		ret = cxip_ux_send(req, ux_send->oflow_req, &ux_send->put_ev,
				   mrecv_start, mrecv_len);
		if (ret != FI_SUCCESS) {
			req->recv.start_offset += mrecv_len;
			return ret;
		}

		/* No Rendezvous event expected. Start RGet now. */
		ret = issue_rdzv_get(req);
		if (ret != FI_SUCCESS) {
			req->recv.start_offset += mrecv_len;
			return ret;
		}

		/* Count the rendezvous event. */
		rdzv_recv_req_event(req);
	} else {
		if (ux_send->put_ev.tgt_long.rlength)
			ret = cxip_ux_send(req, ux_send->oflow_req,
					   &ux_send->put_ev, mrecv_start,
					   mrecv_len);
		else
			ret = cxip_ux_send_zb(req, &ux_send->put_ev,
					      mrecv_start);

		if (ret != FI_SUCCESS) {
			/* undo mrecv_req_put_bytes() */
			req->recv.start_offset -= mrecv_len;
			return ret;
		}
	}

	/* If this is a multi-receive request and there is still space, return
	 * a special code to indicate SW should keep matching messages to it.
	 */
	if (ret == FI_SUCCESS && !req_done)
		return -FI_EINPROGRESS;

	return ret;
}

/*
 * cxip_recv_sw_matcher() - Attempt to match the Receive in SW.
 *
 * Loop through all onloaded UX Sends looking for a match for the Receive
 * request. If a match is found, progress the operation.
 *
 * Caller must hold req->recv.rxc->lock.
 */
static int cxip_recv_sw_matcher(struct cxip_req *req)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	struct cxip_ux_send *ux_send;
	struct dlist_entry *tmp;
	int ret = -FI_ENOMSG;
	union cxip_match_bits ux_mb;
	uint32_t ux_init;

	if (dlist_empty(&rxc->sw_ux_list))
		return -FI_ENOMSG;

	dlist_foreach_container_safe(&rxc->sw_ux_list, struct cxip_ux_send,
				     ux_send, rxc_entry, tmp) {
		ux_mb.raw = ux_send->put_ev.tgt_long.match_bits;
		ux_init = ux_send->put_ev.tgt_long.initiator.initiator.process;

		if (req->recv.tagged) {
			if (!ux_mb.tagged)
				continue;

			if (!tag_match(ux_mb.tag, req->recv.tag,
				       req->recv.ignore))
				continue;

			if (!init_match(ux_init, req->recv.match_id))
				continue;
		} else {
			if (ux_mb.tagged)
				continue;

			if (!init_match(ux_init, req->recv.match_id))
				continue;
		}

		ret = cxip_recv_sw_match(req, ux_send);
		if (ret == -FI_EAGAIN) {
			/* Couldn't process match, try again */
			break;
		}

		dlist_remove(&ux_send->rxc_entry);
		free(ux_send);
		rxc->sw_ux_list_len--;

		CXIP_DBG("Software match, req: %p ux_send: %p (sw_ux_list_len: %u)\n",
			 req, ux_send, req->recv.rxc->sw_ux_list_len);

		if (ret == -FI_EINPROGRESS) {
			/* Multi-recv, keep matching */
			ret = -FI_ENOMSG;
			continue;
		}

		ret = FI_SUCCESS;
		break;
	}

	return ret;
}

/*
 * cxip_recv_req_dropped() - Mark the Received request dropped.
 *
 * If HW does not have sufficient LEs to perform an append, the command is
 * dropped. Queue the request for replay. When all outstanding append commands
 * complete, replay all Receives.
 */
static int cxip_recv_req_dropped(struct cxip_req *req)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	int ret __attribute__((unused));

	if (!cxip_env.fc_recovery)
		CXIP_FATAL(FC_MSG);

	fastlock_acquire(&rxc->lock);

	dlist_remove(&req->recv.rxc_entry);
	dlist_insert_tail(&req->recv.rxc_entry, &rxc->replay_queue);

	CXIP_DBG("Receive dropped: %p\n", req);

	fastlock_release(&rxc->lock);

	return FI_SUCCESS;
}

/*
 * cxip_recv_req_queue() - Queue Receive request on RXC.
 *
 * Before appending a new Receive request to a HW list, attempt to match the
 * Receive to any onloaded UX Sends. A receive can only be appended to hardware
 * in RXC_ENABLED state unless check_rxc_state is false.
 *
 * Caller must hold the RXC lock.
 */
static int cxip_recv_req_queue(struct cxip_req *req, bool check_rxc_state,
			       bool restart_seq)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	int ret;

	/* Try to match against onloaded Sends first. */
	ret = cxip_recv_sw_matcher(req);
	if (ret == FI_SUCCESS)
		return -FI_EALREADY;
	else if (ret == -FI_EAGAIN)
		return -FI_EAGAIN;
	else if (ret != -FI_ENOMSG)
		CXIP_FATAL("SW matching failed: %d\n", ret);

	if (check_rxc_state && rxc->state != RXC_ENABLED)
		return -FI_EAGAIN;

	ret = _cxip_recv_req(req, restart_seq);
	if (ret)
		return -FI_EAGAIN;

	dlist_insert_tail(&req->recv.rxc_entry, &rxc->msg_queue);

	return FI_SUCCESS;
}

/*
 * cxip_recv_req_dequeue_nolock() - Dequeue Receive request from RXC.
 *
 * Caller must hold req->recv.rxc->lock.
 */
static void cxip_recv_req_dequeue_nolock(struct cxip_req *req)
{
	dlist_remove(&req->recv.rxc_entry);
}

/*
 * cxip_recv_req_dequeue() - Dequeue Receive request from RXC.
 *
 * A Receive request may be dequeued from the RXC as soon as there is evidence
 * that the append command has been accepted.
 */
static void cxip_recv_req_dequeue(struct cxip_req *req)
{
	struct cxip_rxc *rxc = req->recv.rxc;

	fastlock_acquire(&rxc->lock);

	cxip_recv_req_dequeue_nolock(req);

	fastlock_release(&rxc->lock);
}
#else
static int cxip_recv_req_dropped(struct cxip_req *req) {return FI_SUCCESS;}
static int cxip_recv_req_queue(struct cxip_req *req)  {return FI_SUCCESS;}
static void cxip_recv_req_dequeue_nolock(struct cxip_req *req) {}
static void cxip_recv_req_dequeue(struct cxip_req *req) {}
#endif

/*
 * _cxip_recv_req() - Submit Receive request to hardware.
 */
static ssize_t _cxip_recv_req(struct cxip_req *req, bool restart_seq)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	uint32_t le_flags;
	union cxip_match_bits mb = {};
	union cxip_match_bits ib = {
		.tx_id = ~0,
		.match_comp = 1,
		.le_type = ~0,
	};
	int ret;
	struct cxip_md *recv_md = req->recv.recv_md;
	uint64_t recv_iova = 0;

	if (req->recv.tagged) {
		mb.tagged = 1;
		mb.tag = req->recv.tag;
		ib.tag = req->recv.ignore;
	}

	/* Always set manage_local in Receive LEs. This makes Cassini ignore
	 * initiator remote_offset in all Puts. With this, remote_offset in Put
	 * events can be used by the initiator for protocol data. The behavior
	 * of use_once is not impacted by manage_local.
	 */
	le_flags = C_LE_EVENT_LINK_DISABLE | C_LE_EVENT_UNLINK_DISABLE |
		   C_LE_MANAGE_LOCAL |
		   C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_PUT;
	if (!req->recv.multi_recv)
		le_flags |= C_LE_USE_ONCE;
	if (restart_seq)
		le_flags |= C_LE_RESTART_SEQ;

	if (recv_md)
		recv_iova = CXI_VA_TO_IOVA(recv_md->md,
					   (uint64_t)req->recv.recv_buf +
					   req->recv.start_offset);

	/* Issue Append command */
	ret = cxip_pte_append(rxc->rx_pte, recv_iova,
			      req->recv.ulen - req->recv.start_offset,
			      recv_md ? recv_md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, req->req_id,
			      mb.raw, ib.raw, req->recv.match_id,
			      rxc->min_multi_recv,
			      le_flags, NULL, rxc->rx_cmdq,
			      !(req->recv.flags & FI_MORE));
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Failed to write Append command: %d\n", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * cxip_recv_common() - Common message receive function. Used for tagged and
 * untagged sends of all sizes.
 */
ssize_t cxip_recv_common(struct cxip_rxc *rxc, void *buf, size_t len,
			 void *desc, fi_addr_t src_addr, uint64_t tag,
			 uint64_t ignore, void *context, uint64_t flags,
			 bool tagged, struct cxip_cntr *comp_cntr)
{
	struct cxip_domain *dom = rxc->domain;
	int ret;
	struct cxip_md *recv_md = NULL;
	struct cxip_req *req;
	struct cxip_addr caddr;
	uint32_t match_id;

	if (len && !buf)
		return -FI_EINVAL;

	if (rxc->state == RXC_DISABLED)
		return -FI_EOPBADSTATE;

	if (!ofi_recv_allowed(rxc->attr.caps))
		return -FI_ENOPROTOOPT;

	if (tagged && (tag & ~CXIP_TAG_MASK || ignore & ~CXIP_TAG_MASK)) {
		CXIP_DBG("Invalid tag: %#018lx ignore: %#018lx (%#018lx)\n",
			 tag, ignore, CXIP_TAG_MASK);
		return -FI_EINVAL;
	}

	/* If FI_DIRECTED_RECV and a src_addr is specified, encode the address
	 * in the LE for matching. If application AVs are symmetric, use
	 * logical FI address for matching. Otherwise, use physical address.
	 */
	if (rxc->attr.caps & FI_DIRECTED_RECV &&
	    src_addr != FI_ADDR_UNSPEC) {
		if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
			/* PID is not used for matching */
			match_id = CXI_MATCH_ID(rxc->pid_bits, C_PID_ANY,
						src_addr);
		} else {
			ret = _cxip_av_lookup(rxc->ep_obj->av, src_addr,
					      &caddr);
			if (ret != FI_SUCCESS) {
				CXIP_DBG("Failed to look up FI addr: %d\n",
					 ret);
				return -FI_EINVAL;
			}

			match_id = CXI_MATCH_ID(rxc->pid_bits, caddr.pid,
						caddr.nic);
		}
	} else {
		match_id = CXI_MATCH_ID_ANY;
	}

	/* Map local buffer */
	if (len) {
		ret = cxip_map(dom, (void *)buf, len, &recv_md);
		if (ret) {
			CXIP_DBG("Failed to map recv buffer: %d\n", ret);
			return ret;
		}
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->recv_cq, 1, rxc);
	if (!req) {
		CXIP_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto recv_unmap;
	}
	ofi_atomic_inc32(&rxc->orx_reqs);

	/* req->data_len, req->tag, req->data must be set later. req->buf may
	 * be overwritten later.
	 */
	req->context = (uint64_t)context;

	req->flags = FI_RECV | (flags & FI_COMPLETION);
	if (tagged)
		req->flags |= FI_TAGGED;
	else
		req->flags |= FI_MSG;

	req->buf = 0;
	req->cb = cxip_recv_cb;

	req->type = CXIP_REQ_RECV;
	req->recv.rxc = rxc;
	req->recv.cntr = comp_cntr ? comp_cntr : rxc->recv_cntr;
	req->recv.recv_buf = buf;
	req->recv.recv_md = recv_md;
	req->recv.ulen = len;
	req->recv.match_id = match_id;
	req->recv.tag = tag;
	req->recv.ignore = ignore;
	req->recv.flags = flags;
	req->recv.tagged = tagged;
	req->recv.start_offset = 0;
	req->recv.multi_recv = (flags & FI_MULTI_RECV ? true : false);
	req->recv.mrecv_bytes = len;
	req->recv.parent = NULL;
	dlist_init(&req->recv.children);

	/* Count Put, Rendezvous, and Reply events during offloaded RPut. */
	req->recv.rdzv_events = 0;

	fastlock_acquire(&rxc->lock);
	ret = cxip_recv_req_queue(req, true, false);
	fastlock_release(&rxc->lock);

	/* Match made in software? */
	if (ret == -FI_EALREADY)
		return FI_SUCCESS;

	/* RXC busy (onloading Sends or full CQ)? */
	if (ret != FI_SUCCESS)
		goto req_free;

	CXIP_DBG("req: %p buf: %p len: %lu src_addr: %ld tag(%c): 0x%lx ignore: 0x%lx context: %p\n",
		 req, buf, len, src_addr, tagged ? '*' : '-', tag, ignore,
		 context);

	return FI_SUCCESS;

req_free:
	ofi_atomic_dec32(&rxc->orx_reqs);
	cxip_cq_req_free(req);
recv_unmap:
	if (recv_md)
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
		CXIP_DBG("Found EP FI Addr: %lu\n", txc->ep_obj->fi_addr);
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
	/* PID is not used for logical matching, but is used for rendezvous. */
	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC)
		return CXI_MATCH_ID(txc->pid_bits,
				    txc->ep_obj->src_addr.pid + txc->tx_id,
				    _txc_fi_addr(txc));

	return CXI_MATCH_ID(txc->pid_bits, txc->ep_obj->src_addr.pid,
			    txc->ep_obj->src_addr.nic);
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
		CXIP_DBG("Request success: %p\n", req);

		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_WARN("Failed to report completion: %d\n",
					 ret);
		}

		if (sw_cntr && req->send.cntr) {
			ret = cxip_cntr_mod(req->send.cntr, 1, false, false);
			if (ret)
				CXIP_WARN("cxip_cntr_mod returned: %d\n", ret);
		}
	} else {
		CXIP_WARN("Request error: %p (err: %d, %s)\n",
			  req, FI_EIO, cxi_rc_to_str(req->send.rc));

		ret = cxip_cq_req_error(req, 0, FI_EIO, req->send.rc, NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_WARN("Failed to report error: %d\n", ret);

		if (sw_cntr && req->send.cntr) {
			ret = cxip_cntr_mod(req->send.cntr, 1, false, true);
			if (ret)
				CXIP_WARN("cxip_cntr_mod returned: %d\n", ret);
		}
	}
}

/*
 * rdzv_send_req_complete() - Complete long send request.
 */
static void rdzv_send_req_complete(struct cxip_req *req)
{
	cxip_rdzv_id_free(req->send.txc->ep_obj, req->send.rdzv_id);

	cxip_unmap(req->send.send_md);

	report_send_completion(req, true);

	ofi_atomic_dec32(&req->send.txc->otx_reqs);
	cxip_cq_req_free(req);
}

/*
 * long_send_req_event() - Count a long send event.
 *
 * Call for each initiator event. The events could be generated in any order.
 * Once all expected events are received, complete the request.
 *
 * A successful long Send generates two events: Ack and Get. That applies to
 * both the offloaded rendezvous protocol and long eager protocol.
 *
 * Note: a Get is not used for a long eager Send if the Put matches in the
 * Priority list.
 */
static void long_send_req_event(struct cxip_req *req)
{
	if (++req->send.long_send_events == 2)
		rdzv_send_req_complete(req);
}

/*
 * cxip_send_long_cb() - Long send callback.
 *
 * Progress a long send operation to completion.
 */
static int cxip_send_long_cb(struct cxip_req *req, const union c_event *event)
{
	int event_rc;
	int ret;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		/* The source Put completed. */
		event_rc = cxi_init_event_rc(event);

		CXIP_DBG("Acked: %p (rc: %s list: %s)\n", req,
			 cxi_rc_to_str(event_rc),
			 cxi_ptl_list_to_str(event->init_short.ptl_list));

		/* If the message was dropped, mark the peer as disabled. Do
		 * not generate a completion. Free associated resources. Do not
		 * free the request (it will be used to replay the Send).
		 */
		if (event_rc == C_RC_PT_DISABLED) {
			ret = cxip_send_req_dropped(req->send.txc, req);
			if (ret == FI_SUCCESS) {
				cxip_rdzv_id_free(req->send.txc->ep_obj,
						  req->send.rdzv_id);
				cxip_unmap(req->send.send_md);
			} else {
				ret = -FI_EAGAIN;
			}

			return ret;
		}

		/* Message was accepted by the peer. Match order is preserved.
		 * The request can be dequeued from the SW message queue. This
		 * allows flow-control recovery to be performed before
		 * outstanding long Send operations have completed.
		 */
		cxip_send_req_dequeue(req->send.txc, req);

		/* The transaction is complete if:
		 * 1. The Put failed
		 * 2. Using the eager long protocol and data landed in the
		 *    Priority list
		 */
		if (event_rc != C_RC_OK ||
		    (!req->send.txc->ep_obj->rdzv_offload &&
		     event->init_short.ptl_list == C_PTL_LIST_PRIORITY)) {
			req->send.rc = event_rc;
			rdzv_send_req_complete(req);
		} else {
			/* Count the event, another may be expected. */
			long_send_req_event(req);
		}
		return FI_SUCCESS;
	default:
		CXIP_WARN("Unexpected event received: %s\n",
			  cxi_event_to_str(event));
		return FI_SUCCESS;
	}
}

/*
 * rdzv_src_cb() - Process rendezvous source buffer events.
 *
 * A Get event is generated for each rendezvous Send indicating Send completion.
 */
static int rdzv_src_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_txc *txc = req->rdzv_src.txc;
	struct cxip_req *get_req;
	union cxip_match_bits mb;
	int event_rc = cxi_tgt_event_rc(event);

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		if (event_rc != C_RC_OK)
			CXIP_WARN("%s error: %p rc: %s\n",
				  cxi_event_to_str(event), req,
				  cxi_rc_to_str(event_rc));
		else
			CXIP_DBG("%s received: %p rc: %s\n",
				 cxi_event_to_str(event), req,
				 cxi_rc_to_str(event_rc));

		req->rdzv_src.rc = cxi_tgt_event_rc(event);
		return FI_SUCCESS;
	case C_EVENT_UNLINK:
		dlist_remove(&req->rdzv_src.list);
		cxip_cq_req_free(req);
		ofi_atomic_sub32(&txc->rdzv_src_lacs, 1 << req->rdzv_src.lac);

		CXIP_DBG("RDZV source window unlinked (LAC: %u)\n",
			 req->rdzv_src.lac);
		return FI_SUCCESS;
	case C_EVENT_GET:
		mb.raw = event->tgt_long.match_bits;
		get_req = cxip_rdzv_id_lookup(txc->ep_obj, mb.rdzv_id_lo);
		if (!get_req) {
			CXIP_WARN("Failed to find RDZV ID: %d\n",
				  mb.rdzv_id_lo);
			return FI_SUCCESS;
		}

		event_rc = cxi_tgt_event_rc(event);
		if (event_rc != C_RC_OK)
			CXIP_WARN("Get error: %p rc: %s\n",
				  get_req, cxi_rc_to_str(event_rc));
		else
			CXIP_DBG("Get received: %p rc: %s\n",
				 get_req, cxi_rc_to_str(event_rc));

		get_req->send.rc = event_rc;

		/* Count the event, another may be expected. */
		long_send_req_event(get_req);

		return FI_SUCCESS;
	default:
		CXIP_WARN("Unexpected event received: %s\n",
			  cxi_event_to_str(event));
		return FI_SUCCESS;
	}
}

/*
 * cxip_txc_prep_rdzv_src() - Prepare an LAC for use with the rendezvous
 * protocol.
 *
 * Synchronously append an LE describing every address in the specified LAC.
 * Each rendezvous Send uses this LE to access source buffer data.
 */
static int cxip_txc_prep_rdzv_src(struct cxip_txc *txc, unsigned int lac)
{
	int ret;
	struct cxip_req *req;
	uint32_t le_flags;
	union cxip_match_bits mb = {};
	union cxip_match_bits ib = { .raw = ~0 };
	uint32_t lac_mask = 1 << lac;

	if (ofi_atomic_get32(&txc->rdzv_src_lacs) & lac_mask)
		return FI_SUCCESS;

	fastlock_acquire(&txc->rdzv_src_lock);

	if (ofi_atomic_get32(&txc->rdzv_src_lacs) & lac_mask) {
		ret = FI_SUCCESS;
		goto unlock;
	}

	req = cxip_cq_req_alloc(txc->send_cq, 1, NULL);
	if (!req) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	req->cb = rdzv_src_cb;
	req->type = CXIP_REQ_RDZV_SRC;
	req->rdzv_src.txc = txc;
	req->rdzv_src.lac = lac;
	req->rdzv_src.rc = 0;

	mb.rdzv_lac = lac;
	ib.rdzv_lac = 0;

	le_flags = C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_GET;

	ret = cxip_pte_append(txc->rdzv_pte, 0, -1ULL, lac,
			      C_PTL_LIST_PRIORITY, req->req_id,
			      mb.raw, ib.raw, CXI_MATCH_ID_ANY, 0,
			      le_flags, NULL, txc->rx_cmdq, true);
	if (ret != FI_SUCCESS) {
		ret = -FI_EAGAIN;
		goto req_free;
	}

	do {
		sched_yield();
		cxip_cq_progress(txc->send_cq);
	} while (!req->rdzv_src.rc);

	if (req->rdzv_src.rc == C_RC_OK) {
		ofi_atomic_add32(&txc->rdzv_src_lacs, lac_mask);
		dlist_insert_tail(&req->rdzv_src.list, &txc->rdzv_src_reqs);

		CXIP_DBG("RDZV source window linked (LAC: %u)\n", lac);
	} else {
		ret = -FI_EAGAIN;
		goto req_free;
	}

	fastlock_release(&txc->rdzv_src_lock);

	return FI_SUCCESS;

req_free:
	cxip_cq_req_free(req);
unlock:
	fastlock_release(&txc->rdzv_src_lock);

	return ret;
}

/*
 * cxip_txc_rdzv_src_fini() - Unlink rendezvous source LEs.
 */
int cxip_txc_rdzv_src_fini(struct cxip_txc *txc)
{
	struct cxip_req *req;
	int ret;

	dlist_foreach_container(&txc->rdzv_src_reqs, struct cxip_req, req,
				     rdzv_src.list) {
		ret = cxip_pte_unlink(txc->rdzv_pte, C_PTL_LIST_PRIORITY,
				      req->req_id, txc->rx_cmdq);
		if (ret) {
			CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
			return ret;
		}
	}

	/* Wait for unlink events */
	do {
		sched_yield();
		cxip_cq_progress(txc->send_cq);
	} while (ofi_atomic_get32(&txc->rdzv_src_lacs));

	return FI_SUCCESS;
}

/*
 * _cxip_send_long() - Initiate a long send operation.
 *
 * There are two long send protocols implemented: an eager (long) protocol and
 * an offloaded rendezvous protocol.
 *
 * The eager (long) protocol works as follows:
 *
 * 1. The Initiator performs a Put of the entire source buffer.
 * 2. An Ack event is generated indicating the Put completed. The Ack indicates
 *    whether it matched in the Priority or Overflow list at the target.
 * 3a. If the Put matched in the Priority list, the entire payload was copied
 *     directly to a receive buffer at the target. The operation is complete.
 * 3b. If the Put matched in the Overflow list, the payload was truncated to
 *     zero. The Target receives events describing the Put attempt.
 * 4b. The Target performs a Get of the entire source buffer using the source
 *     buffer LE.
 *
 * The rendezvous protocol works as follows:
 *
 * 1. The Initiator performs a Rendezvous Put command which includes a portion
 *    of the source buffer data.
 * 2. Once the Put is matched to a user receive buffer (in the Priority list),
 *    a Get of the remaining source data is performed.
 */
static ssize_t _cxip_send_long(struct cxip_req *req)
{
	struct cxip_txc *txc = req->send.txc;
	struct cxip_md *send_md;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	struct c_full_dma_cmd cmd = {};
	union cxip_match_bits put_mb = {};
	int rdzv_id;
	int ret;
	struct cxip_cmdq *cmdq =
		req->triggered ? txc->domain->trig_cmdq : txc->tx_cmdq;
	bool trig = req->triggered;

	/* Calculate DFA */
	cxi_build_dfa(req->send.caddr.nic, req->send.caddr.pid, txc->pid_bits,
		      CXIP_PTL_IDX_RXQ, &dfa, &idx_ext);

	/* Map local buffer */
	ret = cxip_map(txc->domain, req->send.buf, req->send.len, &send_md);
	if (ret) {
		CXIP_DBG("Failed to map send buffer: %d\n", ret);
		return ret;
	}
	req->send.send_md = send_md;

	/* Prepare rendezvous source buffer */
	ret = cxip_txc_prep_rdzv_src(txc, send_md->md->lac);
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Failed to prepare source window: %d\n",
			     ret);
		goto err_unmap;
	}

	/* Allocate rendezvous ID */
	rdzv_id = cxip_rdzv_id_alloc(txc->ep_obj, req);
	if (rdzv_id < 0)
		goto err_unmap;

	/* Build match bits */
	if (req->send.tagged) {
		put_mb.tagged = 1;
		put_mb.tag = req->send.tag;
	}

	req->send.rdzv_id = rdzv_id;
	req->cb = cxip_send_long_cb;
	req->send.long_send_events = 0;

	/* Build Put command descriptor */
	cmd.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.index_ext = idx_ext;
	cmd.lac = send_md->md->lac;
	cmd.event_send_disable = 1;
	cmd.restricted = 0;
	cmd.dfa = dfa;
	cmd.local_addr = CXI_VA_TO_IOVA(send_md->md, req->send.buf);
	cmd.request_len = req->send.len;
	cmd.eq = txc->send_cq->evtq->eqn;
	cmd.user_ptr = (uint64_t)req;
	cmd.initiator = cxip_msg_match_id(txc);
	cmd.header_data = req->send.data;
	cmd.remote_offset = CXI_VA_TO_IOVA(send_md->md, req->send.buf);

	fastlock_acquire(&cmdq->lock);

	ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
			      cxip_ofi_to_cxi_tc(txc->tclass),
			      CXI_TC_TYPE_DEFAULT);
	if (ret != FI_SUCCESS)
		goto err_unlock;

	if (req->send.flags & FI_FENCE) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_DBG("Failed to issue CQ_FENCE command: %d\n",
				 ret);
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	}

	if (txc->ep_obj->rdzv_offload) {
		cmd.command.opcode = C_CMD_RENDEZVOUS_PUT;
		cmd.eager_length = txc->rdzv_eager_size;
		cmd.use_offset_for_get = 1;

		put_mb.rdzv_lac = send_md->md->lac;
		put_mb.le_type = CXIP_LE_TYPE_RX;
		cmd.match_bits = put_mb.raw;

		/* RPut rdzv ID goes in command */
		cmd.rendezvous_id = rdzv_id;
	} else {
		cmd.command.opcode = C_CMD_PUT;

		/* Match sink buffer */
		put_mb.le_type = CXIP_LE_TYPE_SINK;
		/* Use match bits for rdzv_id */
		put_mb.rdzv_id_hi = rdzv_id;
		cmd.match_bits = put_mb.raw;
	}

	if (trig) {
		const struct c_ct_cmd ct_cmd = {
			.trig_ct = req->trig_cntr->ct->ctn,
			.threshold = req->trig_thresh,
		};

		/* Clear the triggered flag to prevent retrying of operation,
		 * due to flow control, from using the triggered path.
		 */
		req->triggered = false;

		ret = cxi_cq_emit_trig_full_dma(cmdq->dev_cmdq, &ct_cmd,
						&cmd);
	} else {
		ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &cmd);
	}

	if (ret) {
		CXIP_WARN("Failed to enqueue Put: %d\n", ret);
		goto err_unlock;
	}

	cxip_txq_ring(cmdq, req->send.flags & FI_MORE, trig,
		      ofi_atomic_get32(&req->send.txc->otx_reqs) - 1);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;

err_unlock:
	fastlock_release(&cmdq->lock);
	cxip_rdzv_id_free(txc->ep_obj, rdzv_id);
err_unmap:
	cxip_unmap(send_md);

	return -FI_EAGAIN;
}

/*
 * cxip_send_eager_cb() - Eager send callback. Used for both tagged and
 * untagged messages.
 */
static int cxip_send_eager_cb(struct cxip_req *req,
			      const union c_event *event)
{
	int match_complete = req->flags & FI_MATCH_COMPLETE;
	int ret;

	if (req->send.send_md) {
		cxip_unmap(req->send.send_md);
		req->send.send_md = NULL;
	}

	if (req->send.ibuf) {
		cxip_cq_ibuf_free(req->cq, req->send.ibuf);
		req->send.ibuf = NULL;
	}

	req->send.rc = cxi_init_event_rc(event);

	/* If MATCH_COMPLETE was requested and the the Put did not match a user
	 * buffer, do not generate a completion event until the target notifies
	 * the initiator that the match is complete.
	 */
	if (match_complete) {
		if (req->send.rc == C_RC_OK &&
		    event->init_short.ptl_list != C_PTL_LIST_PRIORITY) {
			CXIP_DBG("Waiting for match complete: %p\n", req);
			return FI_SUCCESS;
		}

		CXIP_DBG("Match complete with Ack: %p\n", req);
		cxip_tx_id_free(req->send.txc->ep_obj, req->send.tx_id);
	}

	/* If the message was dropped, mark the peer as disabled. Do not
	 * generate a completion. Free associated resources. Do not free the
	 * request (it will be used to replay the Send).
	 */
	if (req->send.rc == C_RC_PT_DISABLED) {
		ret = cxip_send_req_dropped(req->send.txc, req);
		if (ret != FI_SUCCESS)
			ret = -FI_EAGAIN;

		return ret;
	}

	/* If MATCH_COMPLETE was requested, software must manage counters. */
	report_send_completion(req, match_complete);

	cxip_send_req_dequeue(req->send.txc, req);

	ofi_atomic_dec32(&req->send.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
}

/*
 * _cxip_send_eager() - Enqueue eager send command.
 */
static ssize_t _cxip_send_eager(struct cxip_req *req)
{
	struct cxip_txc *txc = req->send.txc;
	struct cxip_md *send_md = NULL;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	union cxip_match_bits mb = {
		.le_type = CXIP_LE_TYPE_RX
	};
	int idc;
	ssize_t ret __attribute__((unused));
	int match_complete = req->send.flags & FI_MATCH_COMPLETE;
	int tx_id;
	struct cxip_cmdq *cmdq =
		req->triggered ? txc->domain->trig_cmdq : txc->tx_cmdq;
	const void *buf = NULL;
	bool trig = req->triggered;
	enum fi_hmem_iface iface;
	struct iovec hmem_iov;

	/* Always use IDCs when the payload fits */
	idc = (req->send.len <= CXIP_INJECT_SIZE) && !trig;

	/* Calculate DFA */
	cxi_build_dfa(req->send.caddr.nic, req->send.caddr.pid, txc->pid_bits,
		      CXIP_PTL_IDX_RXQ, &dfa, &idx_ext);

	if (req->send.len) {
		if (req->send.flags & FI_INJECT || (idc && txc->hmem)) {
			/* Allocate an internal buffer to hold source data for
			 * SW retry and/or a FI_HMEM bounce buffer.
			 */
			req->send.ibuf = cxip_cq_ibuf_alloc(txc->send_cq);
			if (!req->send.ibuf)
				return -FI_ENOSPC;

			if (txc->hmem)
				iface = ofi_get_hmem_iface(req->send.buf);
			else
				iface = FI_HMEM_SYSTEM;

			hmem_iov.iov_base = (void *)req->send.buf;
			hmem_iov.iov_len = req->send.len;

			ret = ofi_copy_from_hmem_iov(req->send.ibuf,
						     req->send.len, iface, 0,
						     &hmem_iov, 1, 0);
			assert(ret == req->send.len);

			buf = req->send.ibuf;
		} else {
			/* IDCs do not require memory mapping. */
			if (!idc) {
				ret = cxip_map(txc->domain, req->send.buf,
					       req->send.len, &send_md);
				if (ret != FI_SUCCESS) {
					CXIP_DBG("Failed to map send buffer: %ld\n",
						 ret);
					return ret;
				}
			}

			buf = req->send.buf;
		}
	}
	req->send.send_md = send_md;

	/* Build match bits */
	if (req->send.tagged) {
		mb.tagged = 1;
		mb.tag = req->send.tag;
	}

	/* Allocate a TX ID if match completion guarantees are required */
	if (match_complete) {
		tx_id = cxip_tx_id_alloc(txc->ep_obj, req);
		if (tx_id < 0) {
			CXIP_DBG("Failed to allocate TX ID: %ld\n", ret);
			goto err_unmap;
		}

		req->send.tx_id = tx_id;

		mb.match_comp = 1;
		mb.tx_id = tx_id;
	}

	req->cb = cxip_send_eager_cb;

	/* Submit command */
	fastlock_acquire(&cmdq->lock);

	ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
			      cxip_ofi_to_cxi_tc(txc->tclass),
			      CXI_TC_TYPE_DEFAULT);
	if (ret != FI_SUCCESS)
		goto err_unlock;

	if (req->send.flags & FI_FENCE) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_DBG("Failed to issue CQ_FENCE command: %ld\n",
				 ret);
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	}

	if (idc) {
		union c_cmdu cmd = {};

		cmd.c_state.event_send_disable = 1;
		cmd.c_state.index_ext = idx_ext;
		cmd.c_state.eq = txc->send_cq->evtq->eqn;
		cmd.c_state.initiator = cxip_msg_match_id(txc);

		/* If MATCH_COMPLETE was requested, software must manage
		 * counters.
		 */
		if (req->send.cntr && !match_complete) {
			cmd.c_state.event_ct_ack = 1;
			cmd.c_state.ct = req->send.cntr->ct->ctn;
		}

		ret = cxip_cmdq_emit_c_state(cmdq, &cmd.c_state);
		if (ret) {
			CXIP_DBG("Failed to issue C_STATE command: %ld\n", ret);
			goto err_unlock;
		}

		memset(&cmd.idc_msg, 0, sizeof(cmd.idc_msg));
		cmd.idc_msg.dfa = dfa;
		cmd.idc_msg.match_bits = mb.raw;
		cmd.idc_msg.header_data = req->send.data;
		cmd.idc_msg.user_ptr = (uint64_t)req;

		ret = cxi_cq_emit_idc_msg(cmdq->dev_cmdq, &cmd.idc_msg,
					  buf, req->send.len);
		if (ret) {
			CXIP_DBG("Failed to write IDC: %ld\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	} else {
		struct c_full_dma_cmd cmd = {};

		/* Inject should only use IDCs. */
		assert(!req->send.ibuf);

		cmd.command.cmd_type = C_CMD_TYPE_DMA;
		cmd.command.opcode = C_CMD_PUT;
		cmd.index_ext = idx_ext;
		cmd.lac = send_md->md->lac;
		cmd.event_send_disable = 1;
		cmd.restricted = 0;
		cmd.dfa = dfa;
		cmd.remote_offset = 0;
		cmd.local_addr = CXI_VA_TO_IOVA(send_md->md, req->send.buf);
		cmd.request_len = req->send.len;
		cmd.eq = txc->send_cq->evtq->eqn;
		cmd.user_ptr = (uint64_t)req;
		cmd.initiator = cxip_msg_match_id(txc);
		cmd.match_bits = mb.raw;
		cmd.header_data = req->send.data;

		/* If MATCH_COMPLETE was requested, software must manage
		 * counters.
		 */
		if (req->send.cntr && !match_complete) {
			cmd.event_ct_ack = 1;
			cmd.ct = req->send.cntr->ct->ctn;
		}

		/* Issue Eager Put command */
		if (trig) {
			const struct c_ct_cmd ct_cmd = {
				.trig_ct = req->trig_cntr->ct->ctn,
				.threshold = req->trig_thresh,
			};

			/* Clear the triggered flag to prevent retrying of
			 * operation, due to flow control, from using the
			 * triggered path.
			 */
			req->triggered = false;

			ret = cxi_cq_emit_trig_full_dma(cmdq->dev_cmdq, &ct_cmd,
							&cmd);
		} else {
			ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &cmd);
		}

		if (ret) {
			CXIP_DBG("Failed to write DMA command: %ld\n", ret);
			ret = -FI_EAGAIN;
			goto err_unlock;
		}
	}

	cxip_txq_ring(cmdq, req->send.flags & FI_MORE, trig,
		      ofi_atomic_get32(&req->send.txc->otx_reqs) - 1);

	fastlock_release(&cmdq->lock);

	return FI_SUCCESS;

err_unlock:
	fastlock_release(&cmdq->lock);
	if (match_complete)
		cxip_tx_id_free(txc->ep_obj, req->send.tx_id);
err_unmap:
	if (req->send.ibuf)
		cxip_cq_ibuf_free(req->cq, req->send.ibuf);
	if (send_md)
		cxip_unmap(send_md);

	return ret;
}

static ssize_t _cxip_send_req(struct cxip_req *req)
{
	if (req->send.len >
	    (req->send.txc->rdzv_threshold +
	     req->send.txc->rdzv_get_min))
		return _cxip_send_long(req);
	else
		return _cxip_send_eager(req);
}

/*
 * cxip_fc_peer_lookup() - Check if a peer is disabled.
 *
 * Look up disabled peer state and return it, if available.
 *
 * Caller must hold txc->lock.
 */
static struct cxip_fc_peer *cxip_fc_peer_lookup(struct cxip_txc *txc,
						struct cxip_addr caddr,
						uint8_t rxc_id)
{
	struct cxip_fc_peer *peer;

	dlist_foreach_container(&txc->fc_peers, struct cxip_fc_peer,
				peer, txc_entry) {
		if (CXIP_ADDR_EQUAL(peer->caddr, caddr) &&
		    peer->rxc_id == rxc_id) {
			return peer;
		}
	}

	return NULL;
}

/*
 * cxip_fc_peer_put() - Account for completion of an outstanding Send targeting
 * a disabled peer.
 *
 * Drop a reference to a disabled peer. When the last reference is dropped,
 * attempt flow-control recovery.
 *
 * Caller must hold txc->lock.
 */
static int cxip_fc_peer_put(struct cxip_fc_peer *peer)
{
	int ret;

	/* Account for the completed Send */
	if (!--peer->pending) {
		peer->req.send.mb.drops = peer->dropped;

		ret = cxip_ctrl_msg_send(&peer->req);
		if (ret)
			return ret;

		peer->pending_acks++;

		CXIP_DBG("Notified disabled peer, TXC: %p NIC: %#x PID: %u dropped: %u\n",
			 peer->txc, peer->caddr.nic, peer->caddr.pid,
			 peer->dropped);
	}

	return FI_SUCCESS;
}

/*
 * cxip_fc_peer_fini() - Remove disabled peer state.
 *
 * Caller must hold txc->lock.
 */
static void cxip_fc_peer_fini(struct cxip_fc_peer *peer)
{
	assert(dlist_empty(&peer->msg_queue));
	dlist_remove(&peer->txc_entry);
	free(peer);
}

/*
 * cxip_fc_notify_cb() - Process FC notify completion events.
 */
int cxip_fc_notify_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_fc_peer *peer = container_of(req, struct cxip_fc_peer, req);
	struct cxip_txc *txc = peer->txc;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		switch (cxi_event_rc(event)) {
		case C_RC_OK:
			fastlock_acquire(&txc->lock);

			/* Peer flow control structure can only be freed if
			 * replay is complete and all acks accounted for.
			 */
			peer->pending_acks--;
			if (!peer->pending_acks && peer->replayed)
				cxip_fc_peer_fini(peer);
			fastlock_release(&txc->lock);
			return FI_SUCCESS;

		/* This error occurs when the target's control event queue has
		 * run out of space. Since the target should be processing the
		 * event queue, it is safe to replay messages until C_RC_OK is
		 * returned.
		 */
		case C_RC_ENTRY_NOT_FOUND:
			CXIP_WARN("Target dropped flow control message... replaying message\n");
			return cxip_ctrl_msg_send(req);
		default:
			CXIP_FATAL("Unexpected event rc: %d\n",
				   cxi_event_rc(event));
		}
	default:
		CXIP_FATAL("Unexpected event type: %d\n",
			   event->hdr.event_type);
	}
}

/*
 * cxip_fc_peer_init() - Mark a peer as disabled.
 *
 * Called by sending EP after experiencing first dropped Send to a peer.
 *
 * Allocate state to track the disabled peer. Locate all outstanding Sends
 * targeting the peer.
 *
 * Caller must hold txc->lock.
 */
static int cxip_fc_peer_init(struct cxip_txc *txc, struct cxip_addr caddr,
			     uint8_t rxc_id, struct cxip_fc_peer **peer)
{
	struct cxip_fc_peer *p;
	struct cxip_req *req;
	struct dlist_entry *tmp;

	p = calloc(1, sizeof(*p));
	if (!p) {
		CXIP_WARN("Failed to allocate FC Peer\n");
		return -FI_ENOMEM;
	}

	p->caddr = caddr;
	p->rxc_id = rxc_id;
	p->txc = txc;
	dlist_init(&p->msg_queue);
	dlist_insert_tail(&p->txc_entry, &txc->fc_peers);

	p->req.send.nic_addr = caddr.nic;
	p->req.send.pid = caddr.pid;
	p->req.send.mb.txc_id = txc->tx_id;
	p->req.send.mb.rxc_id = rxc_id;

	p->req.send.mb.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG;
	p->req.send.mb.ctrl_msg_type = CXIP_CTRL_MSG_FC_NOTIFY;
	p->req.cb = cxip_fc_notify_cb;
	p->req.ep_obj = txc->ep_obj;

	/* Queue all Sends to the FC'ed peer */
	dlist_foreach_container_safe(&txc->msg_queue, struct cxip_req,
				     req, send.txc_entry, tmp) {
		if (CXIP_ADDR_EQUAL(req->send.caddr, caddr) &&
		     req->send.rxc_id == rxc_id) {
			dlist_remove(&req->send.txc_entry);
			dlist_insert_tail(&req->send.txc_entry, &p->msg_queue);
			p->pending++;
			req->send.fc_peer = p;
		}
	}

	*peer = p;

	return FI_SUCCESS;
}

/*
 * cxip_fc_resume() - Replay dropped Sends.
 *
 * Called by sending EP after being notified disabled peer was re-enabled.
 *
 * Replay all dropped Sends in order.
 */
int cxip_fc_resume(struct cxip_ep_obj *ep_obj, uint8_t txc_id,
		   uint32_t nic_addr, uint32_t pid, uint8_t rxc_id)
{
	struct cxip_txc *txc = ep_obj->txcs[txc_id];
	struct cxip_fc_peer *peer;
	struct cxip_addr caddr = {
		.nic = nic_addr,
		.pid = pid,
	};
	struct cxip_req *req;
	struct dlist_entry *tmp;
	int ret __attribute__((unused));

	fastlock_acquire(&txc->lock);

	peer = cxip_fc_peer_lookup(txc, caddr, rxc_id);
	if (!peer)
		CXIP_FATAL("FC peer not found: TXC: %u NIC: %#x PID: %d\n",
			   txc_id, nic_addr, pid);

	CXIP_DBG("Replaying dropped sends, TXC: %u NIC: %#x PID: %d\n",
		 txc_id, nic_addr, pid);

	dlist_foreach_container_safe(&peer->msg_queue, struct cxip_req,
				     req, send.txc_entry, tmp) {
		ret = _cxip_send_req(req);
		assert(ret == FI_SUCCESS);

		/* Move request back to the message queue. */
		dlist_remove(&req->send.txc_entry);
		req->send.fc_peer = NULL;
		dlist_insert_tail(&req->send.txc_entry, &txc->msg_queue);

		CXIP_DBG("Replayed %p\n", req);
	}

	/* Peer flow control structure can only be freed if replay is complete
	 * and all acks accounted for.
	 */
	if (!peer->pending_acks)
		cxip_fc_peer_fini(peer);
	else
		peer->replayed = true;

	fastlock_release(&txc->lock);

	return FI_SUCCESS;
}

/*
 * cxip_send_req_dropped() - Mark the Send request dropped.
 *
 * Mark the Send request dropped. Mark the target peer as disabled. Track all
 * outstanding Sends targeting the disabled peer. When all outstanding Sends
 * are completed, recovery will be performed.
 */
static int cxip_send_req_dropped(struct cxip_txc *txc, struct cxip_req *req)
{
	struct cxip_fc_peer *peer;
	int ret;

	if (!cxip_env.fc_recovery)
		CXIP_FATAL(FC_MSG);

	fastlock_acquire(&txc->lock);

	/* Check if peer is already disabled */
	peer = cxip_fc_peer_lookup(txc, req->send.caddr, req->send.rxc_id);
	if (!peer) {
		ret = cxip_fc_peer_init(txc, req->send.caddr, req->send.rxc_id,
					&peer);
		if (ret != FI_SUCCESS) {
			fastlock_release(&txc->lock);
			return ret;
		}

		CXIP_DBG("Disabled peer detected, TXC: %p NIC: %#x PID: %u pending: %u\n",
			 txc, peer->caddr.nic, peer->caddr.pid, peer->pending);
	}

	/* Account for the dropped message. */
	peer->dropped++;
	ret = cxip_fc_peer_put(peer);
	if (ret)
		peer->dropped--;
	else
		CXIP_DBG("Send dropped, req: %p NIC: %#x PID: %u pending: %u dropped: %u\n",
			 req, peer->caddr.nic, peer->caddr.pid, peer->pending,
			 peer->dropped);

	fastlock_release(&txc->lock);

	return ret;
}

/*
 * cxip_send_req_queue() - Queue Send request on TXC.
 *
 * Place the Send request in an ordered SW queue. Return error if the target
 * peer is disabled.
 */
static int cxip_send_req_queue(struct cxip_txc *txc, struct cxip_req *req)
{
	struct cxip_fc_peer *peer;

	fastlock_acquire(&txc->lock);

	if (!dlist_empty(&txc->fc_peers)) {
		peer = cxip_fc_peer_lookup(txc, req->send.caddr,
					   req->send.rxc_id);
		if (peer) {
			/* Peer is disabled */
			fastlock_release(&txc->lock);
			return -FI_EAGAIN;
		}
	}

	dlist_insert_tail(&req->send.txc_entry, &txc->msg_queue);

	fastlock_release(&txc->lock);

	return FI_SUCCESS;
}

/*
 * cxip_send_req_dequeue() - Dequeue Send request from TXC.
 *
 * Remove the Send requst from the ordered message queue. Update peer
 * flow-control state, if necessary.
 */
static void cxip_send_req_dequeue(struct cxip_txc *txc, struct cxip_req *req)
{
	fastlock_acquire(&txc->lock);

	if (req->send.fc_peer) {
		/* The peer was disabled after this message arrived. */
		CXIP_DBG("Send not dropped, req: %p NIC: %#x PID: %u pending: %u dropped: %u\n",
			 req, req->send.fc_peer->caddr.nic,
			 req->send.fc_peer->caddr.pid,
			 req->send.fc_peer->pending,
			 req->send.fc_peer->dropped);

		cxip_fc_peer_put(req->send.fc_peer);
	}

	dlist_remove(&req->send.txc_entry);

	fastlock_release(&txc->lock);
}

/*
 * cxip_send_common() - Common message send function. Used for tagged and
 * untagged sends of all sizes. This includes triggered operations.
 */
ssize_t cxip_send_common(struct cxip_txc *txc, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 uint64_t tag, void *context, uint64_t flags,
			 bool tagged, bool triggered, uint64_t trig_thresh,
			 struct cxip_cntr *trig_cntr,
			 struct cxip_cntr *comp_cntr)
{
	struct cxip_req *req;
	struct cxip_addr caddr;
	int ret;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_send_allowed(txc->attr.caps))
		return -FI_ENOPROTOOPT;

	if (len && !buf)
		return -FI_EINVAL;

	if (len > CXIP_EP_MAX_MSG_SZ)
		return -FI_EMSGSIZE;

	if (tagged && tag & ~CXIP_TAG_MASK) {
		CXIP_DBG("Invalid tag: %#018lx (%#018lx)\n",
			 tag, CXIP_TAG_MASK);
		return -FI_EINVAL;
	}

	if (flags & FI_INJECT && len > CXIP_INJECT_SIZE) {
		CXIP_DBG("Invalid inject length: %lu\n", len);
		return -FI_EMSGSIZE;
	}

	req = cxip_cq_req_alloc(txc->send_cq, false, txc);
	if (!req) {
		CXIP_DBG("Failed to allocate request\n");
		return -FI_EAGAIN;
	}
	ofi_atomic_inc32(&txc->otx_reqs);

	req->triggered = triggered;
	req->trig_thresh = trig_thresh;
	req->trig_cntr = trig_cntr;

	/* Save Send parameters to replay */
	req->type = CXIP_REQ_SEND;
	req->send.txc = txc;
	req->send.cntr = triggered ? comp_cntr : txc->send_cntr;
	req->send.buf = buf;
	req->send.len = len;
	req->send.tagged = tagged;
	req->send.tag = tag;
	req->send.data = data;
	req->send.flags = flags;

	/* Set completion parameters */
	req->context = (uint64_t)context;
	req->flags = FI_SEND | (flags & (FI_COMPLETION | FI_MATCH_COMPLETE));
	if (tagged)
		req->flags |= FI_TAGGED;
	else
		req->flags |= FI_MSG;

	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Failed to look up FI addr: %d\n", ret);
		goto req_free;
	}

	/* Check for RX context ID */
	req->send.rxc_id = CXIP_AV_ADDR_RXC(txc->ep_obj->av, dest_addr);
	caddr.pid += req->send.rxc_id;

	req->send.caddr = caddr;

	/* Check if target peer is disabled */
	ret = cxip_send_req_queue(req->send.txc, req);
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Target peer disabled\n");
		goto req_free;
	}

	/* Try Send */
	ret = _cxip_send_req(req);
	if (ret != FI_SUCCESS)
		goto req_dequeue;

	CXIP_DBG("req: %p buf: %p len: %lu dest_addr: %ld tag(%c): 0x%lx context %#lx\n",
		 req, req->send.buf, req->send.len, dest_addr,
		 req->send.tagged ? '*' : '-', req->send.tag,
		 req->context);

	/* Do progress inline if there are a lot of outstanding operations. */
	if (ofi_atomic_get32(&txc->otx_reqs) > CXIP_OTX_REQS_POLL_THRESH)
		cxip_cq_progress(txc->send_cq);

	return FI_SUCCESS;

req_dequeue:
	cxip_send_req_dequeue(req->send.txc, req);
req_free:
	ofi_atomic_dec32(&txc->otx_reqs);
	cxip_cq_req_free(req);

	return ret;
}

/*
 * Libfabric APIs
 */

static ssize_t cxip_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			  fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			  void *context)
{
	struct cxip_rxc *rxc;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_recv_common(rxc, buf, len, desc, src_addr, tag, ignore,
				context, rxc->attr.op_flags, true, NULL);
}

static ssize_t cxip_trecvv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t src_addr,
			   uint64_t tag, uint64_t ignore, void *context)
{
	struct cxip_rxc *rxc;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_recv_common(rxc, iov[0].iov_base, iov[0].iov_len,
				desc ? desc[0] : NULL, src_addr, tag, ignore,
				context, rxc->attr.op_flags, true, NULL);
}

static ssize_t cxip_trecvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg,
			     uint64_t flags)
{
	struct cxip_rxc *rxc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_RX_OP_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!rxc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_recv_common(rxc, msg->msg_iov[0].iov_base,
				msg->msg_iov[0].iov_len,
				msg->desc ? msg->desc[0] : NULL, msg->addr,
				msg->tag, msg->ignore, msg->context, flags,
				true, NULL);
}

static ssize_t cxip_tsend(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, desc, 0, dest_addr, tag, context,
				txc->attr.op_flags, true, false, 0, NULL, NULL);
}

static ssize_t cxip_tsendv(struct fid_ep *ep, const struct iovec *iov,
			   void **desc, size_t count, fi_addr_t dest_addr,
			   uint64_t tag, void *context)
{
	struct cxip_txc *txc;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, iov[0].iov_base, iov[0].iov_len,
				desc ? desc[0] : NULL, 0, dest_addr, tag,
				context, txc->attr.op_flags, true, false, 0,
				NULL, NULL);
}

static ssize_t cxip_tsendmsg(struct fid_ep *ep,
			     const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_TX_OP_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_send_common(txc, msg->msg_iov[0].iov_base,
				msg->msg_iov[0].iov_len,
				msg->desc ? msg->desc[0] : NULL, msg->data,
				msg->addr, msg->tag, msg->context, flags, true,
				false, 0, NULL, NULL);
}

static ssize_t cxip_tinject(struct fid_ep *ep, const void *buf, size_t len,
			    fi_addr_t dest_addr, uint64_t tag)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, NULL, 0, dest_addr, tag, NULL,
				FI_INJECT, true, false, 0, NULL, NULL);
}

static ssize_t cxip_tsenddata(struct fid_ep *ep, const void *buf, size_t len,
			      void *desc, uint64_t data, fi_addr_t dest_addr,
			      uint64_t tag, void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, desc, data, dest_addr, tag,
				context, txc->attr.op_flags, true, false, 0,
				NULL, NULL);
}

static ssize_t cxip_tinjectdata(struct fid_ep *ep, const void *buf, size_t len,
				uint64_t data, fi_addr_t dest_addr,
				uint64_t tag)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, NULL, data, dest_addr, tag, NULL,
				FI_INJECT, true, false, 0, NULL, NULL);
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
	.senddata = cxip_tsenddata,
	.injectdata = cxip_tinjectdata,
};

static ssize_t cxip_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, void *context)
{
	struct cxip_rxc *rxc;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_recv_common(rxc, buf, len, desc, src_addr, 0, 0, context,
				rxc->attr.op_flags, false, NULL);
}

static ssize_t cxip_recvv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  void *context)
{
	struct cxip_rxc *rxc;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_recv_common(rxc, iov[0].iov_base, iov[0].iov_len,
				desc ? desc[0] : NULL, src_addr, 0, 0, context,
				rxc->attr.op_flags, false, NULL);
}

static ssize_t cxip_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct cxip_rxc *rxc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_RX_OP_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_rxc(ep, &rxc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!rxc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_recv_common(rxc, msg->msg_iov[0].iov_base,
				msg->msg_iov[0].iov_len,
				msg->desc ? msg->desc[0] : NULL, msg->addr, 0,
				0, msg->context, flags, false, NULL);
}

static ssize_t cxip_send(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, desc, 0, dest_addr, 0, context,
				txc->attr.op_flags, false, false, 0, NULL,
				NULL);
}

static ssize_t cxip_sendv(struct fid_ep *ep, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  void *context)
{
	struct cxip_txc *txc;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, iov[0].iov_base, iov[0].iov_len,
				desc ? desc[0] : NULL, 0, dest_addr, 0, context,
				txc->attr.op_flags, false, false, 0, NULL,
				NULL);
}

static ssize_t cxip_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || msg->iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_TX_OP_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_send_common(txc, msg->msg_iov[0].iov_base,
				msg->msg_iov[0].iov_len,
				msg->desc ? msg->desc[0] : NULL, msg->data,
				msg->addr, 0, msg->context, flags, false, false,
				0, NULL, NULL);
}

static ssize_t cxip_inject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, NULL, 0, dest_addr, 0, NULL,
				FI_INJECT, false, false, 0, NULL, NULL);
}

static ssize_t cxip_senddata(struct fid_ep *ep, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, desc, data, dest_addr, 0,
				context, txc->attr.op_flags, false, false, 0,
				NULL, NULL);
}

static ssize_t cxip_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_send_common(txc, buf, len, NULL, data, dest_addr, 0, NULL,
				FI_INJECT, false, false, 0, NULL, NULL);
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
	.senddata = cxip_senddata,
	.injectdata = cxip_injectdata,
};

