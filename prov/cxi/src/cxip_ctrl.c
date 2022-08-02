/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018-2020 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <ofi_util.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * cxip_ctrl_msg_cb() - Process control message target events.
 */
int cxip_ctrl_msg_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	uint32_t pid_bits = req->ep_obj->domain->iface->dev->info.pid_bits;
	uint32_t nic_addr;
	uint32_t pid;
	union cxip_match_bits mb = {
		.raw = event->tgt_long.match_bits,
	};
	uint32_t init = event->tgt_long.initiator.initiator.process;
	int ret __attribute__((unused));

	switch (event->hdr.event_type) {
	case C_EVENT_PUT:
		assert(cxi_event_rc(event) == C_RC_OK);

		nic_addr = CXI_MATCH_ID_EP(pid_bits, init);
		pid = CXI_MATCH_ID_PID(pid_bits, init);

		switch (mb.ctrl_msg_type) {
		case CXIP_CTRL_MSG_FC_NOTIFY:
			ret = cxip_fc_process_drops(req->ep_obj, mb.rxc_id,
						    nic_addr, pid, mb.txc_id,
						    mb.drops);
			assert(ret == FI_SUCCESS);

			break;
		case CXIP_CTRL_MSG_FC_RESUME:
			ret = cxip_fc_resume(req->ep_obj, mb.txc_id, nic_addr,
					     pid, mb.rxc_id);
			assert(ret == FI_SUCCESS);

			break;
		case CXIP_CTRL_MSG_ZB_DATA:
			ret = cxip_zbcoll_recv_cb(req->ep_obj, nic_addr, pid,
						  mb.raw);
			assert(ret == FI_SUCCESS);
			break;
		default:
			CXIP_FATAL("Unexpected msg type: %d\n",
				   mb.ctrl_msg_type);
		}

		break;
	default:
		CXIP_FATAL("Unexpected event type: %d\n",
			   event->hdr.event_type);
	}

	CXIP_DBG("got event: %s rc: %s (req: %p)\n",
		 cxi_event_to_str(event),
		 cxi_rc_to_str(cxi_event_rc(event)),
		 req);

	return FI_SUCCESS;
}

/*
 * cxip_ctrl_msg_send() - Send a control message.
 */
int cxip_ctrl_msg_send(struct cxip_ctrl_req *req)
{
	struct cxip_cmdq *txq;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_bits;
	union c_cmdu cmd = {};
	uint32_t match_id;
	int ret;

	txq = req->ep_obj->ctrl_txq;
	pid_bits = req->ep_obj->domain->iface->dev->info.pid_bits;
	cxi_build_dfa(req->send.nic_addr, req->send.pid, pid_bits,
		      CXIP_PTL_IDX_CTRL, &dfa, &idx_ext);
	match_id = CXI_MATCH_ID(pid_bits, req->ep_obj->src_addr.pid,
				req->ep_obj->src_addr.nic);

	cmd.c_state.event_send_disable = 1;
	cmd.c_state.index_ext = idx_ext;
	cmd.c_state.eq = req->ep_obj->ctrl_tx_evtq->eqn;
	cmd.c_state.initiator = match_id;

	/* Cannot use ep_obj->lock else a deadlock will occur. Thus serialize on
	 * TXQ lock instead.
	 */
	ofi_spin_lock(&txq->lock);

	if (!req->ep_obj->ctrl_tx_credits) {
		CXIP_WARN("Control TX credits exhausted\n");
		ret = -FI_EAGAIN;
		goto err_unlock;
	}

	req->ep_obj->ctrl_tx_credits--;

	ret = cxip_cmdq_emit_c_state(txq, &cmd.c_state);
	if (ret) {
		CXIP_DBG("Failed to issue C_STATE command: %d\n", ret);
		goto err_return_credit;
	}

	memset(&cmd.idc_msg, 0, sizeof(cmd.idc_msg));
	cmd.idc_msg.dfa = dfa;
	cmd.idc_msg.match_bits = req->send.mb.raw;
	cmd.idc_msg.user_ptr = (uint64_t)req;

	ret = cxi_cq_emit_idc_msg(txq->dev_cmdq, &cmd.idc_msg, NULL, 0);
	if (ret) {
		CXIP_DBG("Failed to write IDC: %d\n", ret);

		/* Return error according to Domain Resource Management
		 */
		ret = -FI_EAGAIN;
		goto err_return_credit;
	}

	cxi_cq_ring(txq->dev_cmdq);

	ofi_spin_unlock(&txq->lock);

	CXIP_DBG("Queued control message: %p\n", req);

	return FI_SUCCESS;

err_return_credit:
	req->ep_obj->ctrl_tx_credits++;
err_unlock:
	ofi_spin_unlock(&txq->lock);

	return ret;
}

/*
 * cxip_ctrl_msg_init() - Initialize control messaging resources.
 *
 * Caller must hold ep_obj->lock.
 */
int cxip_ctrl_msg_init(struct cxip_ep_obj *ep_obj)
{
	const union c_event *event;
	int ret;
	uint32_t le_flags;
	union cxip_match_bits mb = {
		.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG,
	};
	union cxip_match_bits ib = {
		.raw = ~0,
	};

	ret = cxip_domain_ctrl_id_alloc(ep_obj->domain, &ep_obj->ctrl_msg_req);
	if (ret) {
		CXIP_WARN("Failed to allocate MR buffer ID: %d\n", ret);
		return -FI_ENOSPC;
	}
	ep_obj->ctrl_msg_req.ep_obj = ep_obj;
	ep_obj->ctrl_msg_req.cb = cxip_ctrl_msg_cb;

	le_flags = C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_PUT;

	ib.ctrl_le_type = 0;

	ret = cxip_pte_append(ep_obj->ctrl_pte, 0, 0, 0,
			      C_PTL_LIST_PRIORITY, ep_obj->ctrl_msg_req.req_id,
			      mb.raw, ib.raw, CXI_MATCH_ID_ANY, 0, le_flags,
			      NULL, ep_obj->ctrl_tgq, true);
	if (ret) {
		CXIP_DBG("Failed to write Append command: %d\n", ret);
		goto err_free_id;
	}

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->ctrl_tgt_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_LINK ||
	    event->tgt_long.buffer_id != ep_obj->ctrl_msg_req.req_id) {
		/* This is a device malfunction */
		CXIP_WARN("Invalid Link EQE %u %u %u %u\n",
			  event->hdr.event_type,
			  event->tgt_long.return_code,
			  event->tgt_long.buffer_id,
			  ep_obj->ctrl_msg_req.req_id);
		ret = -FI_EIO;
		goto err_free_id;
	}

	if (cxi_event_rc(event) != C_RC_OK) {
		CXIP_WARN("Append failed: %s\n",
			  cxi_rc_to_str(cxi_event_rc(event)));
		ret = -FI_ENOSPC;
		goto err_free_id;
	}

	cxi_eq_ack_events(ep_obj->ctrl_tgt_evtq);

	CXIP_DBG("Control messaging initialized: %p\n", ep_obj);

	return FI_SUCCESS;

err_free_id:
	cxip_domain_ctrl_id_free(ep_obj->domain, &ep_obj->ctrl_msg_req);

	return ret;
}

/*
 * cxip_ctrl_msg_fini() - Finalize control messaging resources.
 *
 * Caller must hold ep_obj->lock.
 */
void cxip_ctrl_msg_fini(struct cxip_ep_obj *ep_obj)
{
	cxip_domain_ctrl_id_free(ep_obj->domain, &ep_obj->ctrl_msg_req);

	CXIP_DBG("Control messaging finalized: %p\n", ep_obj);
}

/*
 * cxip_ep_ctrl_event_req() - Look up a control request using Cassini event.
 */
static struct cxip_ctrl_req *cxip_ep_ctrl_event_req(struct cxip_ep_obj *ep_obj,
						    const union c_event *event)
{
	struct cxip_ctrl_req *req;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		req = (struct cxip_ctrl_req *)event->init_short.user_ptr;
		break;
	case C_EVENT_LINK:
	case C_EVENT_UNLINK:
	case C_EVENT_PUT:
		req = cxip_domain_ctrl_id_at(ep_obj->domain,
					     event->tgt_long.buffer_id);
		if (!req)
			CXIP_WARN("Invalid buffer_id: %d (%s)\n",
				  event->tgt_long.buffer_id,
				  cxi_event_to_str(event));
		break;
	case C_EVENT_STATE_CHANGE:
		cxip_pte_state_change(ep_obj->domain->iface, event);

		req = NULL;
		break;
	case C_EVENT_COMMAND_FAILURE:
		CXIP_FATAL("Command failure: cq=%u target=%u fail_loc=%u cmd_type=%u cmd_size=%u opcode=%u\n",
			   event->cmd_fail.cq_id, event->cmd_fail.is_target,
			   event->cmd_fail.fail_loc,
			   event->cmd_fail.fail_command.cmd_type,
			   event->cmd_fail.fail_command.cmd_size,
			   event->cmd_fail.fail_command.opcode);

	/* Since the control PtlTE is used for unoptimized MRs, it is possible
	 * to trigger a target error message if the user uses an invalid MR key.
	 * For such operations, it is safe to just drop the EQ event.
	 */
	case C_EVENT_ATOMIC:
	case C_EVENT_FETCH_ATOMIC:
		if (cxi_event_rc(event) != C_RC_ENTRY_NOT_FOUND)
			CXIP_FATAL("Invalid %d event rc: %d\n",
				   event->hdr.event_type, cxi_event_rc(event));
		req = NULL;
		break;

	/* Get events can be generated when an invalid standard MR key is used
	 */
	case C_EVENT_GET:
		CXIP_WARN("Unexpected %s event rc: %s\n",
			  cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));
		req = NULL;
		break;

	default:
		CXIP_FATAL("Invalid event type: %d\n", event->hdr.event_type);
	}

	CXIP_DBG("got control event: %s rc: %s (req: %p)\n",
		 cxi_event_to_str(event),
		 cxi_rc_to_str(cxi_event_rc(event)),
		 req);

	return req;
}

static void cxip_ep_return_ctrl_tx_credits(struct cxip_ep_obj *ep_obj,
					   unsigned int credits)
{
	/* Control TX credits are serialized on TXQ lock. */
	ofi_spin_lock(&ep_obj->ctrl_txq->lock);
	ep_obj->ctrl_tx_credits += credits;
	ofi_spin_unlock(&ep_obj->ctrl_txq->lock);
}

void cxip_ep_ctrl_eq_progress(struct cxip_ep_obj *ep_obj,
			      struct cxi_eq *ctrl_evtq, bool tx_evtq,
			      bool ep_obj_locked)
{
	const union c_event *event;
	struct cxip_ctrl_req *req;
	int ret;

	/* The Control EQ is shared by a SEP. Avoid locking. */
	if (!cxi_eq_peek_event(ctrl_evtq))
		return;

	if (!ep_obj_locked)
		ofi_mutex_lock(&ep_obj->lock);

	while ((event = cxi_eq_peek_event(ctrl_evtq))) {
		req = cxip_ep_ctrl_event_req(ep_obj, event);
		if (req) {
			ret = req->cb(req, event);
			if (ret != FI_SUCCESS)
				break;
		}

		/* Consume and ack event. */
		cxi_eq_next_event(ctrl_evtq);

		cxi_eq_ack_events(ctrl_evtq);

		if (tx_evtq)
			cxip_ep_return_ctrl_tx_credits(ep_obj, 1);

	}

	if (cxi_eq_get_drops(ctrl_evtq))
		CXIP_FATAL("Control EQ drops detected\n");

	if (!ep_obj_locked)
		ofi_mutex_unlock(&ep_obj->lock);
}

void cxip_ep_tx_ctrl_progress(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_eq_progress(ep_obj, ep_obj->ctrl_tx_evtq, true, false);
}

void cxip_ep_tx_ctrl_progress_locked(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_eq_progress(ep_obj, ep_obj->ctrl_tx_evtq, true, true);
}

/*
 * cxip_ep_ctrl_progress() - Progress operations using the control EQ.
 */
void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_eq_progress(ep_obj, ep_obj->ctrl_tgt_evtq, false, false);
	cxip_ep_tx_ctrl_progress(ep_obj);
}

/*
 * cxip_ep_ctrl_trywait() - Return 0 if no events need to be progressed.
 */
int cxip_ep_ctrl_trywait(void *arg)
{
	struct cxip_cq *cq = (struct cxip_cq *)arg;

	if (!cq->ep_obj->ctrl_wait) {
		CXIP_WARN("No CXI ep_obj wait object\n");
		return -FI_EINVAL;
	}

	if (cxi_eq_peek_event(cq->ep_obj->ctrl_tgt_evtq) ||
	    cxi_eq_peek_event(cq->ep_obj->ctrl_tx_evtq))
		return -FI_EAGAIN;

	ofi_mutex_lock(&cq->ep_obj->lock);
	cxil_clear_wait_obj(cq->ep_obj->ctrl_wait);

	if (cxi_eq_peek_event(cq->ep_obj->ctrl_tgt_evtq) ||
	    cxi_eq_peek_event(cq->ep_obj->ctrl_tx_evtq)) {
		ofi_mutex_unlock(&cq->ep_obj->lock);
		return -FI_EAGAIN;
	}
	ofi_mutex_unlock(&cq->ep_obj->lock);

	return FI_SUCCESS;
}

static void cxip_eq_ctrl_eq_free(void *eq_buf, struct cxi_md *eq_md,
				 struct cxi_eq *eq)
{
	int ret;

	ret = cxil_destroy_evtq(eq);
	if (ret)
		CXIP_WARN("Failed to free CXI EQ: ret=%d", ret);

	ret = cxil_unmap(eq_md);
	if (ret)
		CXIP_WARN("Failed to unmap EQ buffer: ret=%d", ret);

	free(eq_buf);
}

static int cxip_ep_ctrl_eq_alloc(struct cxip_ep_obj *ep_obj, size_t len,
				 void **eq_buf, struct cxi_md **eq_md,
				 struct cxi_eq **eq)
{
	struct cxi_eq_attr eq_attr = {
		.flags = CXI_EQ_TGT_LONG,
	};
	int ret;
	int unmap_ret __attribute__((unused));

	/* Align up length to C_PAGE_SIZE boundary. */
	len = CXIP_ALIGN(len, C_PAGE_SIZE);

	*eq_buf = aligned_alloc(C_PAGE_SIZE, len);
	if (!eq_buf) {
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = cxil_map(ep_obj->domain->lni->lni, *eq_buf, len,
		       CXIP_EQ_MAP_FLAGS, NULL, eq_md);
	if (ret)
		goto err_free_eq_buf;

	eq_attr.queue = *eq_buf;
	eq_attr.queue_len = len;

	/* ep_obj->ctrl_wait will be NULL if not required */
	ret = cxil_alloc_evtq(ep_obj->domain->lni->lni, *eq_md, &eq_attr,
			      ep_obj->ctrl_wait, NULL, eq);
	if (ret)
		goto err_free_eq_md;

	return FI_SUCCESS;

err_free_eq_md:
	unmap_ret = cxil_unmap(*eq_md);
	assert(unmap_ret == 0);

err_free_eq_buf:
	free(*eq_buf);
err:
	return ret;
}

/*
 * cxip_ctrl_wait_required() - return true if base EP wait object is required.
 */
static bool cxip_ctrl_wait_required(struct cxip_ep_obj *ep_obj)
{
	if (ep_obj->rxcs && ep_obj->rxcs[0] && ep_obj->rxcs[0]->recv_cq &&
	    ep_obj->rxcs[0]->recv_cq->priv_wait)
		return true;

	if (ep_obj->txcs && ep_obj->txcs[0] && ep_obj->txcs[0]->send_cq &&
	    ep_obj->txcs[0]->send_cq->priv_wait)
		return true;

	return false;
}

/*
 * cxip_ep_ctrl_init() - Initialize endpoint control resources.
 *
 * Caller must hold ep_obj->lock.
 */
int cxip_ep_ctrl_init(struct cxip_ep_obj *ep_obj)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
	};
	const union c_event *event;
	int ret;
	int wait_fd;
	size_t rx_eq_size = MIN(cxip_env.ctrl_rx_eq_max_size,
				ofi_universe_size * 64);

	/* If CQ(s) are using a wait object, then control event
	 * queues need to unblock poll as well. CQ will add the
	 * associated FD to the CQ FD list.
	 */
	if (cxip_ctrl_wait_required(ep_obj)) {
		ret = cxil_alloc_wait_obj(ep_obj->domain->lni->lni,
					  &ep_obj->ctrl_wait);
		if (ret) {
			CXIP_WARN("Ctrl internal wait object failed: %d\n",
				  ret);
			return ret;
		}
		wait_fd = cxil_get_wait_obj_fd(ep_obj->ctrl_wait);
		ret = fi_fd_nonblock(wait_fd);
		if (ret) {
			CXIP_WARN("Unable to set ctrl wait non-blocking: %d\n",
				  ret);
			goto err;
		}

		CXIP_DBG("Added control EQ private wait object, intr FD: %d\n",
			 wait_fd);
	}

	ret = cxip_ep_ctrl_eq_alloc(ep_obj, 4 * C_PAGE_SIZE,
				    &ep_obj->ctrl_tx_evtq_buf,
				    &ep_obj->ctrl_tx_evtq_buf_md,
				    &ep_obj->ctrl_tx_evtq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate TX EQ resources, ret: %d\n", ret);
		goto err;
	}

	ret = cxip_ep_ctrl_eq_alloc(ep_obj, rx_eq_size,
				    &ep_obj->ctrl_tgt_evtq_buf,
				    &ep_obj->ctrl_tgt_evtq_buf_md,
				    &ep_obj->ctrl_tgt_evtq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate TGT EQ resources, ret: %d\n",
			  ret);
		goto free_tx_evtq;
	}

	ret = cxip_ep_cmdq(ep_obj, 0, true, ep_obj->domain->tclass,
			   ep_obj->ctrl_tx_evtq, &ep_obj->ctrl_txq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control TXQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_tgt_evtq;
	}

	ret = cxip_ep_cmdq(ep_obj, 0, false, FI_TC_UNSPEC,
			   ep_obj->ctrl_tgt_evtq, &ep_obj->ctrl_tgq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control TGQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_txq;
	}

	ret = cxip_pte_alloc_nomap(ep_obj->if_dom[0], ep_obj->ctrl_tgt_evtq,
				   &pt_opts, NULL, NULL, &ep_obj->ctrl_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control PTE: %d\n", ret);
		goto free_tgq;
	}

	/* CXIP_PTL_IDX_WRITE_MR_STD is shared with CXIP_PTL_IDX_CTRL. */
	ret = cxip_pte_map(ep_obj->ctrl_pte, CXIP_PTL_IDX_WRITE_MR_STD, false);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map write PTE: %d\n", ret);
		goto free_pte;
	}

	ret = cxip_pte_map(ep_obj->ctrl_pte, CXIP_PTL_IDX_READ_MR_STD, false);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map read PTE: %d\n", ret);
		goto free_pte;
	}

	ret = cxip_pte_set_state(ep_obj->ctrl_pte, ep_obj->ctrl_tgq,
				 C_PTLTE_ENABLED, 0);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_WARN("Failed to enqueue command: %d\n", ret);
		goto free_pte;
	}

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(ep_obj->ctrl_tgt_evtq)))
		sched_yield();

	switch (event->hdr.event_type) {
	case C_EVENT_STATE_CHANGE:
		if (event->tgt_long.return_code != C_RC_OK ||
		    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
		    event->tgt_long.ptlte_index != ep_obj->ctrl_pte->pte->ptn)
			CXIP_FATAL("Invalid PtlTE enable event\n");
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

	cxi_eq_ack_events(ep_obj->ctrl_tgt_evtq);

	ret = cxip_ctrl_msg_init(ep_obj);
	if (ret != FI_SUCCESS)
		goto free_pte;

	/* Reserve 4 event queue slots to prevent EQ overrun.
	 * 1. One slot for EQ status writeback
	 * 2. One slot for default reserved_fc value
	 * 3. One slot for EQ overrun detection.
	 * 4. TODO: Determine why an additional slot needs to be reserved.
	 */
	ep_obj->ctrl_tx_credits =
		ep_obj->ctrl_tx_evtq->byte_size / C_EE_CFG_ECB_SIZE - 4;

	CXIP_DBG("EP control initialized: %p\n", ep_obj);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(ep_obj->ctrl_pte);
free_tgq:
	cxip_ep_cmdq_put(ep_obj, 0, false);
free_txq:
	cxip_ep_cmdq_put(ep_obj, 0, true);
free_tgt_evtq:
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tgt_evtq_buf,
			     ep_obj->ctrl_tgt_evtq_buf_md,
			     ep_obj->ctrl_tgt_evtq);
free_tx_evtq:
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tx_evtq_buf,
			     ep_obj->ctrl_tx_evtq_buf_md, ep_obj->ctrl_tx_evtq);
err:
	cxil_destroy_wait_obj(ep_obj->ctrl_wait);
	ep_obj->ctrl_wait = NULL;

	return ret;
}

/*
 * cxip_ep_ctrl_fini() - Finalize endpoint control resources.
 *
 * Caller must hold ep_obj->lock.
 */
void cxip_ep_ctrl_fini(struct cxip_ep_obj *ep_obj)
{
	cxip_ctrl_mr_cache_flush(ep_obj);
	cxip_ctrl_msg_fini(ep_obj);
	cxip_pte_free(ep_obj->ctrl_pte);
	cxip_ep_cmdq_put(ep_obj, 0, false);
	cxip_ep_cmdq_put(ep_obj, 0, true);

	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tgt_evtq_buf,
			     ep_obj->ctrl_tgt_evtq_buf_md,
			     ep_obj->ctrl_tgt_evtq);
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tx_evtq_buf,
			     ep_obj->ctrl_tx_evtq_buf_md, ep_obj->ctrl_tx_evtq);

	if (ep_obj->ctrl_wait) {
		cxil_destroy_wait_obj(ep_obj->ctrl_wait);
		ep_obj->ctrl_wait = NULL;
	}
	CXIP_DBG("EP control finalized: %p\n", ep_obj);
}
