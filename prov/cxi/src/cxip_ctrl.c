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

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

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
			CXIP_LOG_ERROR("Invalid buffer_id: %d (%s)\n",
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
		CXIP_LOG_ERROR("Invalid event type: %d\n",
				event->hdr.event_type);
		req = NULL;
	}

	CXIP_LOG_DBG("got control event: %s rc: %s (req: %p)\n",
		     cxi_event_to_str(event),
		     cxi_rc_to_str(cxi_event_rc(event)),
		     req);

	return req;
}

/*
 * cxip_ep_ctrl_progress() - Progress operations using the control EQ.
 */
void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj)
{
	const union c_event *event;
	struct cxip_ctrl_req *req;
	int events = 0;
	int ret;

	fastlock_acquire(&ep_obj->lock);

	while ((event = cxi_eq_peek_event(ep_obj->ctrl_evtq))) {
		req = cxip_ep_ctrl_event_req(ep_obj, event);
		if (req) {
			ret = req->cb(req, event);
			if (ret != FI_SUCCESS)
				break;
		}

		/* Consume event. */
		cxi_eq_next_event(ep_obj->ctrl_evtq);

		events++;
	}

	if (events)
		cxi_eq_ack_events(ep_obj->ctrl_evtq);

	if (cxi_eq_get_drops(ep_obj->ctrl_evtq)) {
		CXIP_LOG_ERROR("Control EQ drops detected\n");
		abort();
	}

	fastlock_release(&ep_obj->lock);
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
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY
	};
	struct cxi_eq_attr eq_attr = {};
	union c_cmdu cmd = {};
	const union c_event *event;
	int ret;
	int tmp;

	ret = cxip_ep_cmdq(ep_obj, 0, true, &ep_obj->ctrl_txq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate control TXQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	ret = cxip_ep_cmdq(ep_obj, 0, false, &ep_obj->ctrl_tgq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate control TGQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_txq;
	}

	ep_obj->ctrl_evtq_buf_len = C_PAGE_SIZE;
	ep_obj->ctrl_evtq_buf = aligned_alloc(C_PAGE_SIZE,
					      ep_obj->ctrl_evtq_buf_len);
	if (!ep_obj->ctrl_evtq_buf) {
		CXIP_LOG_DBG("Unable to allocate control EVTQ buffer\n");
		goto free_tgq;
	}

	ret = cxil_map(ep_obj->domain->lni->lni, ep_obj->ctrl_evtq_buf,
		       ep_obj->ctrl_evtq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_WRITE,
		       NULL, &ep_obj->ctrl_evtq_buf_md);
	if (ret) {
		CXIP_LOG_DBG("Unable to map control EVTQ buffer, ret: %d\n",
			     ret);
		goto free_evtq_buf;
	}

	eq_attr.queue = ep_obj->ctrl_evtq_buf;
	eq_attr.queue_len = ep_obj->ctrl_evtq_buf_len;
	eq_attr.queue_md = ep_obj->ctrl_evtq_buf_md;
	eq_attr.flags = CXI_EQ_TGT_LONG;

	ret = cxil_alloc_evtq(ep_obj->domain->lni->lni, &eq_attr,
			      NULL, NULL, &ep_obj->ctrl_evtq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate control EVTQ, ret: %d\n",
			     ret);
		ret = -FI_ENODEV;
		goto free_evtq_md;
	}

	ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->ctrl_evtq,
			     CXIP_PTL_IDX_CTRL, &pt_opts, NULL, NULL,
			     &ep_obj->ctrl_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate control PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_evtq;
	}

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = ep_obj->ctrl_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(ep_obj->ctrl_tgq->dev_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_pte;
	}

	cxi_cq_ring(ep_obj->ctrl_tgq->dev_cmdq);

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(ep_obj->ctrl_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != ep_obj->ctrl_pte->pte->ptn) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto free_pte;
	}

	cxi_eq_ack_events(ep_obj->ctrl_evtq);

	memset(&ep_obj->req_ids, 0, sizeof(ep_obj->req_ids));

	CXIP_LOG_DBG("EP control initialized: %p\n", ep_obj);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(ep_obj->ctrl_pte);
free_evtq:
	ret = cxil_destroy_evtq(ep_obj->ctrl_evtq);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy EVTQ: %d\n", ret);
free_evtq_md:
	tmp = cxil_unmap(ep_obj->ctrl_evtq_buf_md);
	if (tmp)
		CXIP_LOG_ERROR("Failed to unmap EVTQ buffer: %d\n", ret);
free_evtq_buf:
	free(ep_obj->ctrl_evtq_buf);
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
	int ret;

	cxip_pte_free(ep_obj->ctrl_pte);

	ofi_idx_reset(&ep_obj->req_ids);

	ret = cxil_destroy_evtq(ep_obj->ctrl_evtq);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy EVTQ: %d\n", ret);

	ret = cxil_unmap(ep_obj->ctrl_evtq_buf_md);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap EVTQ buffer: %d\n",
			       ret);

	free(ep_obj->ctrl_evtq_buf);

	cxip_ep_cmdq_put(ep_obj, 0, false);
	cxip_ep_cmdq_put(ep_obj, 0, true);

	CXIP_LOG_DBG("EP control finalized: %p\n", ep_obj);
}
