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
	struct cxip_cmdq *txq = req->ep_obj->ctrl_txq;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_bits;
	union c_cmdu cmd = {};
	uint32_t match_id;
	int ret;

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
	fastlock_acquire(&txq->lock);

	if (!req->ep_obj->ctrl_tx_credits) {
		CXIP_WARN("Control TX credits exhausted\n");
		ret = -FI_EAGAIN;
		goto err_unlock;
	}

	req->ep_obj->ctrl_tx_credits--;

	if (memcmp(&txq->c_state, &cmd.c_state, sizeof(cmd.c_state))) {
		ret = cxi_cq_emit_c_state(txq->dev_cmdq, &cmd.c_state);
		if (ret) {
			CXIP_DBG("Failed to issue C_STATE command: %d\n", ret);

			/* Return error according to Domain Resource
			 * Management
			 */
			ret = -FI_EAGAIN;
			goto err_return_credit;
		}

		/* Update TXQ C_STATE */
		txq->c_state = cmd.c_state;

		CXIP_DBG("Updated C_STATE: %p\n", req);
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

	fastlock_release(&txq->lock);

	CXIP_DBG("Queued control message: %p\n", req);

	return FI_SUCCESS;

err_return_credit:
	req->ep_obj->ctrl_tx_credits++;
err_unlock:
	fastlock_release(&txq->lock);

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
	int buffer_id;
	int ret;
	uint32_t le_flags;
	union cxip_match_bits mb = {
		.ctrl_le_type = CXIP_CTRL_LE_TYPE_CTRL_MSG,
	};
	union cxip_match_bits ib = {
		.raw = ~0,
	};

	buffer_id = ofi_idx_insert(&ep_obj->req_ids, &ep_obj->ctrl_msg_req);
	if (buffer_id < 0 || buffer_id >= CXIP_BUFFER_ID_MAX) {
		CXIP_WARN("Failed to allocate MR buffer ID: %d\n", buffer_id);
		return -FI_ENOSPC;
	}
	ep_obj->ctrl_msg_req.ep_obj = ep_obj;
	ep_obj->ctrl_msg_req.req_id = buffer_id;
	ep_obj->ctrl_msg_req.cb = cxip_ctrl_msg_cb;

	le_flags = C_LE_UNRESTRICTED_BODY_RO | C_LE_UNRESTRICTED_END_RO |
		   C_LE_OP_PUT;

	ib.ctrl_le_type = 0;

	ret = cxip_pte_append(ep_obj->ctrl_pte, 0, 0, 0,
			      C_PTL_LIST_PRIORITY, buffer_id, mb.raw, ib.raw,
			      CXI_MATCH_ID_ANY, 0, le_flags, NULL,
			      ep_obj->ctrl_tgq, true);
	if (ret) {
		CXIP_DBG("Failed to write Append command: %d\n", ret);
		goto err_free_id;
	}

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->ctrl_tgt_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_LINK ||
	    event->tgt_long.buffer_id != buffer_id) {
		/* This is a device malfunction */
		CXIP_WARN("Invalid Link EQE %u %u %u %u\n",
			  event->hdr.event_type,
			  event->tgt_long.return_code,
			  event->tgt_long.buffer_id, buffer_id);
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
	ofi_idx_remove(&ep_obj->req_ids, buffer_id);

	return ret;
}

/*
 * cxip_ctrl_msg_fini() - Finalize control messaging resources.
 *
 * Caller must hold ep_obj->lock.
 */
void cxip_ctrl_msg_fini(struct cxip_ep_obj *ep_obj)
{
	ofi_idx_remove(&ep_obj->req_ids, ep_obj->ctrl_msg_req.req_id);

	CXIP_DBG("Control messaging finalized: %p\n", ep_obj);
}

/*
 * cxip_ep_ctrl_event_req() - Look up a control request using Cassini event.
 */
static struct cxip_ctrl_req *cxip_ep_ctrl_event_req(struct cxip_ep_obj *ep_obj,
						    const union c_event *event)
{
	struct cxip_ctrl_req *req;
	uint32_t pte_num;
	enum c_ptlte_state pte_state;

	switch (event->hdr.event_type) {
	case C_EVENT_ACK:
		req = (struct cxip_ctrl_req *)event->init_short.user_ptr;
		break;
	case C_EVENT_LINK:
	case C_EVENT_UNLINK:
	case C_EVENT_PUT:
		req = ofi_idx_at(&ep_obj->req_ids, event->tgt_long.buffer_id);
		if (!req)
			CXIP_WARN("Invalid buffer_id: %d (%s)\n",
				  event->tgt_long.buffer_id,
				  cxi_event_to_str(event));
		break;
	case C_EVENT_STATE_CHANGE:
		pte_num = event->tgt_long.ptlte_index;
		pte_state = event->tgt_long.initiator.state_change.ptlte_state;

		cxip_pte_state_change(ep_obj->domain->iface, pte_num,
				      pte_state);

		req = NULL;
		break;
	default:
		CXIP_WARN("Invalid event type: %d\n", event->hdr.event_type);
		req = NULL;
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
	fastlock_acquire(&ep_obj->ctrl_txq->lock);
	ep_obj->ctrl_tx_credits += credits;
	fastlock_release(&ep_obj->ctrl_txq->lock);
}

void cxip_ep_ctrl_eq_progress(struct cxip_ep_obj *ep_obj,
			      struct cxi_eq *ctrl_evtq, bool tx_evtq)
{
	const union c_event *event;
	struct cxip_ctrl_req *req;
	int ret;

	/* The Control EQ is shared by a SEP. Avoid locking. */
	if (!cxi_eq_peek_event(ctrl_evtq))
		return;

	fastlock_acquire(&ep_obj->lock);

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

	fastlock_release(&ep_obj->lock);
}

void cxip_ep_tx_ctrl_progress(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_eq_progress(ep_obj, ep_obj->ctrl_tx_evtq, true);
}

/*
 * cxip_ep_ctrl_progress() - Progress operations using the control EQ.
 */
void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj)
{
	cxip_ep_ctrl_eq_progress(ep_obj, ep_obj->ctrl_tgt_evtq, false);
	cxip_ep_tx_ctrl_progress(ep_obj);
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

	*eq_buf = aligned_alloc(C_PAGE_SIZE, len);
	if (!eq_buf) {
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = cxil_map(ep_obj->domain->lni->lni, *eq_buf, len,
		       CXI_MAP_PIN | CXI_MAP_WRITE, NULL, eq_md);
	if (ret)
		goto err_free_eq_buf;

	eq_attr.queue = *eq_buf;
	eq_attr.queue_len = len;
	ret = cxil_alloc_evtq(ep_obj->domain->lni->lni, *eq_md, &eq_attr, NULL,
			      NULL, eq);
	if (ret)
		goto err_free_eq_md;

	return FI_SUCCESS;

err_free_eq_md:
	ret = cxil_unmap(*eq_md);
	assert(ret == 0);

err_free_eq_buf:
	free(*eq_buf);
err:
	return ret;
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

	ret = cxip_ep_cmdq(ep_obj, 0, true, ep_obj->domain->tclass,
			   &ep_obj->ctrl_txq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control TXQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	ret = cxip_ep_cmdq(ep_obj, 0, false, FI_TC_UNSPEC, &ep_obj->ctrl_tgq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control TGQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_txq;
	}

	ret = cxip_ep_ctrl_eq_alloc(ep_obj, 4 * C_PAGE_SIZE,
				    &ep_obj->ctrl_tx_evtq_buf,
				    &ep_obj->ctrl_tx_evtq_buf_md,
				    &ep_obj->ctrl_tx_evtq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate TX EQ resources, ret: %d\n", ret);
		goto free_tgq;
	}

	ret = cxip_ep_ctrl_eq_alloc(ep_obj, 4 * C_PAGE_SIZE,
				    &ep_obj->ctrl_tgt_evtq_buf,
				    &ep_obj->ctrl_tgt_evtq_buf_md,
				    &ep_obj->ctrl_tgt_evtq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate TGT EQ resources, ret: %d\n",
			  ret);
		goto free_tx_evtq;
	}

	ret = cxip_pte_alloc_nomap(ep_obj->if_dom[0], ep_obj->ctrl_tgt_evtq,
				   &pt_opts, NULL, NULL, &ep_obj->ctrl_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate control PTE: %d\n", ret);
		goto free_tgt_evtq;
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

	if (event->hdr.event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != ep_obj->ctrl_pte->pte->ptn) {
		/* This is a device malfunction */
		CXIP_WARN("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto free_pte;
	}

	cxi_eq_ack_events(ep_obj->ctrl_tgt_evtq);

	memset(&ep_obj->req_ids, 0, sizeof(ep_obj->req_ids));

	ret = cxip_ctrl_msg_init(ep_obj);
	if (ret != FI_SUCCESS)
		goto free_pte;

	/* Number of TX credits is equal to the number of 64-byte event slots
	 * in the EQ minus 1 for detecting full EQ.
	 */
	ep_obj->ctrl_tx_credits = ((ep_obj->ctrl_tx_evtq->byte_size -
				    C_EE_CFG_ECB_SIZE) / 64) - 1;

	CXIP_DBG("EP control initialized: %p\n", ep_obj);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(ep_obj->ctrl_pte);
free_tgt_evtq:
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tgt_evtq_buf,
			     ep_obj->ctrl_tgt_evtq_buf_md,
			     ep_obj->ctrl_tgt_evtq);
free_tx_evtq:
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tx_evtq_buf,
			     ep_obj->ctrl_tx_evtq_buf_md, ep_obj->ctrl_tx_evtq);
free_tgq:
	cxip_ep_cmdq_put(ep_obj, 0, false);
free_txq:
	cxip_ep_cmdq_put(ep_obj, 0, true);

	return ret;
}

/*
 * cxip_ep_ctrl_fini() - Finalize endpoint control resources.
 *
 * Caller must hold ep_obj->lock.
 */
void cxip_ep_ctrl_fini(struct cxip_ep_obj *ep_obj)
{
	cxip_ctrl_msg_fini(ep_obj);

	cxip_pte_free(ep_obj->ctrl_pte);

	ofi_idx_reset(&ep_obj->req_ids);

	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tgt_evtq_buf,
			     ep_obj->ctrl_tgt_evtq_buf_md,
			     ep_obj->ctrl_tgt_evtq);
	cxip_eq_ctrl_eq_free(ep_obj->ctrl_tx_evtq_buf,
			     ep_obj->ctrl_tx_evtq_buf_md, ep_obj->ctrl_tx_evtq);

	cxip_ep_cmdq_put(ep_obj, 0, false);
	cxip_ep_cmdq_put(ep_obj, 0, true);

	CXIP_DBG("EP control finalized: %p\n", ep_obj);
}
