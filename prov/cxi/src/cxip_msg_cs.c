/*
 * SPDX-License-Identifier: BSD-2 Clause or GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
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

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

static void cxip_rnr_recv_pte_cb(struct cxip_pte *pte,
				 const union c_event *event)
{
	struct cxip_rxc *rxc = (struct cxip_rxc *)pte->ctx;
	uint32_t state;

	assert(rxc->protocol == FI_PROTO_CXI_CS);

	switch (event->hdr.event_type) {
	case C_EVENT_STATE_CHANGE:
		if (cxi_event_rc(event) != C_RC_OK ||
		    event->tgt_long.ptlte_index != rxc->rx_pte->pte->ptn)
			CXIP_FATAL("Failed receive PtlTE state change, %s\n",
				   cxi_rc_to_str(cxi_event_rc(event)));

		state = event->tgt_long.initiator.state_change.ptlte_state;

		switch (state) {
		case C_PTLTE_ENABLED:
			assert(rxc->state == RXC_DISABLED);
			rxc->state = RXC_ENABLED;
			RXC_DBG(rxc, "Receive PtlTE enabled\n");
			break;
		case C_PTLTE_DISABLED:
			/* Set to disabled before issuing command */
			assert(rxc->state == RXC_DISABLED);
			rxc->state = RXC_DISABLED;
			RXC_DBG(rxc, "Receive PtlTE disabled\n");
			break;
		default:
			CXIP_FATAL("Unexpected receive PtlTE state %d\n",
				   state);
		}
		break;

	case C_EVENT_COMMAND_FAILURE:
		CXIP_FATAL("Command failure: cq=%u target=%u fail_loc=%u "
			   "cmd_type=%u cmd_size=%u opcode=%u\n",
			   event->cmd_fail.cq_id, event->cmd_fail.is_target,
			   event->cmd_fail.fail_loc,
			   event->cmd_fail.fail_command.cmd_type,
			   event->cmd_fail.fail_command.cmd_size,
			   event->cmd_fail.fail_command.opcode);
		break;
	default:
		CXIP_FATAL("Invalid event type: %s\n", cxi_event_to_str(event));
	}
}

static void cxip_rxc_cs_progress(struct cxip_rxc *rxc)
{
	cxip_evtq_progress(&rxc->rx_evtq);
}

static void cxip_rxc_cs_recv_req_tgt_event(struct cxip_req *req,
					   const union c_event *event)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	union cxip_match_bits mb = {
		.raw = event->tgt_long.match_bits
	};
	uint32_t init = event->tgt_long.initiator.initiator.process;

	assert(rxc->protocol == FI_PROTO_CXI_CS);
	assert(event->hdr.event_type == C_EVENT_PUT);

	req->tag = mb.cs_tag;
	req->recv.initiator = init;

	if (mb.cs_cq_data)
		req->flags |= FI_REMOTE_CQ_DATA;

	req->recv.src_offset = event->tgt_long.remote_offset;

	/* Only need one event to set remaining fields. */
	if (req->recv.tgt_event)
		return;

	req->recv.tgt_event = true;

	/* VNI is needed to support FI_AV_AUTH_KEY. */
	req->recv.vni = event->tgt_long.vni;

	/* rlen is used to detect truncation. */
	req->recv.rlen = event->tgt_long.rlength;

	/* RC is used when generating completion events. */
	req->recv.rc = cxi_tgt_event_rc(event);

	/* Header data is provided in all completion events. */
	req->data = event->tgt_long.header_data;

	/* data_len must be set uniquely for each protocol! */
}

static int cxip_rxc_cs_cancel_msg_recv(struct cxip_req *req)
{
	/* Perform default */
	return cxip_recv_cancel(req);
}

/* Handle any control messaging callbacks specific to protocol */
static int cxip_rxc_cs_ctrl_msg_cb(struct cxip_ctrl_req *req,
				    const union c_event *event)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

static void cxip_rxc_cs_init_struct(struct cxip_rxc *rxc_base,
				    struct cxip_ep_obj *ep_obj)
{
	struct cxip_rxc_cs *rxc = container_of(rxc_base, struct cxip_rxc_cs,
					       base);

	assert(rxc->base.protocol == FI_PROTO_CXI_CS);

	/* Overrides */
	rxc->base.recv_ptl_idx = CXIP_PTL_IDX_RNR_RXQ;
}

static void cxip_rxc_cs_fini_struct(struct cxip_rxc *rxc)
{
	/* Placeholder */
}

static int cxip_rxc_cs_msg_init(struct cxip_rxc *rxc)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.lossless = cxip_env.msg_lossless,
	};
	int ret;

	assert(rxc->protocol == FI_PROTO_CXI_CS);

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->symmetric) {
		CXIP_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->ptable,
			     rxc->rx_evtq.eq, rxc->recv_ptl_idx, false,
			     &pt_opts, cxip_rnr_recv_pte_cb, rxc, &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate RX PTE: %d\n", ret);
		return ret;
	}

	/* Start accepting Puts. */
	ret = cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq, C_PTLTE_ENABLED, 0);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_pte_set_state returned: %d\n", ret);
		goto free_pte;
	}

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_evtq_progress(&rxc->rx_evtq);
	} while (rxc->rx_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(rxc->rx_pte);

	return ret;
}

static int cxip_rxc_cs_msg_fini(struct cxip_rxc *rxc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static void cxip_rxc_cs_cleanup(struct cxip_rxc *rxc_base)
{
	/* Placeholder */

	/* Cancel Receives */
	cxip_rxc_recv_req_cleanup(rxc_base);
}

/*
 * cxip_recv_common() - Common message receive function. Used for tagged and
 * untagged sends of all sizes.
 */
static ssize_t
cxip_recv_common(struct cxip_rxc *rxc, void *buf, size_t len, void *desc,
		 fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		 void *context, uint64_t flags, bool tagged,
		 struct cxip_cntr *comp_cntr)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

static void cxip_txc_cs_progress(struct cxip_txc *txc)
{
	/* Placeholder - must process RNR */
}

static int cxip_txc_cs_cancel_msg_send(struct cxip_req *req)
{
	/* Placeholder CS can cancel transmits */
	return -FI_ENOENT;
}

static void cxip_txc_cs_init_struct(struct cxip_txc *txc_base,
				    struct cxip_ep_obj *ep_obj)
{
	struct cxip_txc_cs *txc = container_of(txc_base, struct cxip_txc_cs,
					       base);
	int i;

	assert(txc->base.protocol == FI_PROTO_CXI_CS);

	txc->base.recv_ptl_idx = CXIP_PTL_IDX_RNR_RXQ;
	ofi_atomic_initialize32(&txc->time_wait_reqs, 0);
	txc->max_retry_wait_us = cxip_env.rnr_max_timeout_us;
	txc->next_retry_wait_us = UINT64_MAX;

	for (i = 0; i < CXIP_NUM_RNR_WAIT_QUEUE; i++)
		dlist_init(&txc->time_wait_queue[i]);
}

static void cxip_txc_cs_fini_struct(struct cxip_txc *txc)
{
	/* Placeholder */
}

static int cxip_txc_cs_msg_init(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static int cxip_txc_cs_msg_fini(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static void cxip_txc_cs_cleanup(struct cxip_txc *txc_base)
{
	/* Placeholder */
}

/*
 * cxip_send_common() - Common message send function. Used for tagged and
 * untagged sends of all sizes. This includes triggered operations.
 */
static ssize_t
cxip_send_common(struct cxip_txc *txc, uint32_t tclass, const void *buf,
		 size_t len, void *desc, uint64_t data, fi_addr_t dest_addr,
		 uint64_t tag, void *context, uint64_t flags, bool tagged,
		 bool triggered, uint64_t trig_thresh,
		 struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

struct cxip_rxc_ops cs_rxc_ops = {
	.recv_common = cxip_recv_common,
	.progress = cxip_rxc_cs_progress,
	.recv_req_tgt_event = cxip_rxc_cs_recv_req_tgt_event,
	.cancel_msg_recv = cxip_rxc_cs_cancel_msg_recv,
	.ctrl_msg_cb = cxip_rxc_cs_ctrl_msg_cb,
	.init_struct = cxip_rxc_cs_init_struct,
	.fini_struct = cxip_rxc_cs_fini_struct,
	.cleanup = cxip_rxc_cs_cleanup,
	.msg_init = cxip_rxc_cs_msg_init,
	.msg_fini = cxip_rxc_cs_msg_fini,
};

struct cxip_txc_ops cs_txc_ops = {
	.send_common = cxip_send_common,
	.progress = cxip_txc_cs_progress,
	.cancel_msg_send = cxip_txc_cs_cancel_msg_send,
	.init_struct = cxip_txc_cs_init_struct,
	.fini_struct = cxip_txc_cs_fini_struct,
	.cleanup = cxip_txc_cs_cleanup,
	.msg_init = cxip_txc_cs_msg_init,
	.msg_fini = cxip_txc_cs_msg_fini,
};
