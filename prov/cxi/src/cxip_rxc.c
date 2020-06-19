/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

/* CXI RX Context Management */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * rxc_msg_enable() - Enable RXC messaging.
 *
 * Change the RXC RX PtlTE to enabled state. Once in enabled state, messages
 * will be accepted by hardware. Prepare all messaging resources before
 * enabling the RX PtlTE.
 *
 * Caller must hold rxc->lock.
 */
int cxip_rxc_msg_enable(struct cxip_rxc *rxc, uint32_t drop_count)
{
	int ret;
	union c_cmdu cmd = {};

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = rxc->rx_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;
	cmd.set_state.drop_count = drop_count;

	fastlock_acquire(&rxc->rx_cmdq->lock);

	ret = cxi_cq_emit_target(rxc->rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);

		fastlock_release(&rxc->rx_cmdq->lock);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(rxc->rx_cmdq->dev_cmdq);

	fastlock_release(&rxc->rx_cmdq->lock);

	rxc->enable_pending = true;

	return FI_SUCCESS;
}

/*
 * rxc_msg_disable() - Disable RXC messaging.
 *
 * Change the RXC RX PtlTE to disabled state. Once in disabled state, the PtlTE
 * will receive no additional events.
 *
 * Caller must hold rxc->lock.
 */
static int rxc_msg_disable(struct cxip_rxc *rxc)
{
	int ret;
	union c_cmdu cmd = {};

	/* Don't treat the state change as flow control */
	rxc->disabling = true;

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = rxc->rx_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_DISABLED;

	fastlock_acquire(&rxc->rx_cmdq->lock);

	ret = cxi_cq_emit_target(rxc->rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);

		fastlock_release(&rxc->rx_cmdq->lock);
		return -FI_EAGAIN;
	}

	cxi_cq_ring(rxc->rx_cmdq->dev_cmdq);

	fastlock_release(&rxc->rx_cmdq->lock);

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (rxc->pte_state != C_PTLTE_DISABLED);

	CXIP_LOG_DBG("RXC PtlTE disabled: %p\n", rxc);

	return FI_SUCCESS;
}

/*
 * rxc_msg_init() - Initialize an RX context for messaging.
 *
 * Allocates and initializes hardware resources used for receiving expected and
 * unexpected message data.
 *
 * Caller must hold rxc->lock.
 */
static int rxc_msg_init(struct cxip_rxc *rxc)
{
	int ret;
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY,
		.en_flowctrl = 1,
	};
	uint64_t pid_idx;

	ret = cxip_ep_cmdq(rxc->ep_obj, rxc->rx_id, false, FI_TC_UNSPEC,
			   &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate RX CMDQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	ret = cxip_ep_cmdq(rxc->ep_obj, rxc->rx_id, true, FI_TC_UNSPEC,
			   &rxc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto put_rx_cmdq;
	}

	/* Select the LEP where the queue will be mapped */
	pid_idx = CXIP_PTL_IDX_RXC(rxc->rx_id);

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->if_dom, rxc->recv_cq->evtq,
			     pid_idx, &pt_opts, cxip_recv_pte_cb, rxc,
			     &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RX PTE: %d\n", ret);
		goto put_tx_cmdq;
	}

	return FI_SUCCESS;

put_tx_cmdq:
	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, true);
put_rx_cmdq:
	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, false);

	return ret;
}

/*
 * rxc_msg_fini() - Finalize RX context messaging.
 *
 * Free hardware resources allocated when the RX context was initialized for
 * messaging.
 *
 * Caller must hold rxc->lock.
 */
static int rxc_msg_fini(struct cxip_rxc *rxc)
{
	cxip_pte_free(rxc->rx_pte);

	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, false);

	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, true);

	return FI_SUCCESS;
}

/*
 * cxip_rxc_enable() - Enable an RX context for use.
 *
 * Called via fi_enable(). The context could be used in a standard endpoint or
 * a scalable endpoint.
 */
int cxip_rxc_enable(struct cxip_rxc *rxc)
{
	int ret = FI_SUCCESS;

	fastlock_acquire(&rxc->lock);

	if (rxc->enabled)
		goto unlock;

	if (!ofi_recv_allowed(rxc->attr.caps)) {
		rxc->enabled = true;
		goto unlock;
	}

	if (!rxc->recv_cq) {
		CXIP_LOG_DBG("Undefined recv CQ\n");
		ret = -FI_ENOCQ;
		goto unlock;
	}

	if (rxc->recv_cntr) {
		ret = cxip_cntr_enable(rxc->recv_cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable(FI_RECV) returned: %d\n",
				     ret);
			goto unlock;
		}
	}

	ret = cxip_cq_enable(rxc->recv_cq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_cq_enable returned: %d\n", ret);
		goto unlock;
	}

	fastlock_release(&rxc->lock);

	ret = rxc_msg_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("rxc_msg_init returned: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	ret = cxip_rxc_oflow_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_rxc_oflow_init returned: %d\n",
			     ret);
		goto msg_fini;
	}

	/* Start accepting Puts. */
	ret = cxip_rxc_msg_enable(rxc, 0);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_rxc_msg_enable returned: %d\n",
			     ret);
		goto oflow_fini;
	}

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (rxc->pte_state != C_PTLTE_ENABLED);

	CXIP_LOG_DBG("RXC messaging enabled: %p\n", rxc);

	rxc->pid_bits = rxc->domain->iface->dev->info.pid_bits;
	rxc->enabled = true;

	return FI_SUCCESS;

oflow_fini:
	cxip_rxc_oflow_fini(rxc);
msg_fini:
	ret = rxc_msg_fini(rxc);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("rxc_msg_fini returned: %d\n", ret);
unlock:
	fastlock_release(&rxc->lock);

	return ret;
}

/*
 * rxc_cleanup() - Attempt to free outstanding requests.
 *
 * Outstanding commands may be dropped when the RX Command Queue is freed.
 * This leads to missing events. Attempt to gather all events before freeing
 * the RX CQ. If events go missing, resources will be leaked until the
 * Completion Queue is freed.
 */
static void rxc_cleanup(struct cxip_rxc *rxc)
{
	int ret;
	uint64_t start;
	int canceled = 0;
	struct cxip_fc_drops *fc_drops;
	struct dlist_entry *tmp;

	if (!ofi_atomic_get32(&rxc->orx_reqs))
		return;

	cxip_cq_req_discard(rxc->recv_cq, rxc);

	do {
		ret = cxip_cq_req_cancel(rxc->recv_cq, rxc, 0, false);
		if (ret == FI_SUCCESS)
			canceled++;
	} while (ret == FI_SUCCESS);

	if (canceled)
		CXIP_LOG_DBG("Canceled %d Receives: %p\n", canceled, rxc);

	start = ofi_gettime_ms();
	while (ofi_atomic_get32(&rxc->orx_reqs)) {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);

		if (ofi_gettime_ms() - start > CXIP_REQ_CLEANUP_TO) {
			CXIP_LOG_ERROR("Timeout waiting for outstanding requests.\n");
			break;
		}
	}

	dlist_foreach_container_safe(&rxc->fc_drops, struct cxip_fc_drops,
				     fc_drops, rxc_entry, tmp) {
		dlist_remove(&fc_drops->rxc_entry);
		free(fc_drops);
	}
}

/*
 * cxip_rxc_disable() - Disable an RX context.
 *
 * Free hardware resources allocated when the context was enabled. Called via
 * fi_close(). The context could be used in a standard endpoint or a scalable
 * endpoint.
 */
static void rxc_disable(struct cxip_rxc *rxc)
{
	int ret;

	fastlock_acquire(&rxc->lock);

	if (!rxc->enabled) {
		fastlock_release(&rxc->lock);
		return;
	}

	rxc->enabled = false;

	fastlock_release(&rxc->lock);

	if (ofi_recv_allowed(rxc->attr.caps)) {
		/* Stop accepting Puts. */
		ret = rxc_msg_disable(rxc);
		if (ret != FI_SUCCESS)
			CXIP_LOG_DBG("rxc_msg_disable returned: %d\n", ret);

		rxc_cleanup(rxc);

		/* Clean up overflow buffers. */
		cxip_rxc_oflow_fini(rxc);

		/* Free hardware resources. */
		ret = rxc_msg_fini(rxc);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("rxc_msg_fini returned: %d\n", ret);
	}
}

/*
 * cxip_rxc_alloc() - Allocate an RX context.
 *
 * Used to support creating an RX context for fi_endpoint() or fi_rx_context().
 */
struct cxip_rxc *cxip_rxc_alloc(const struct fi_rx_attr *attr, void *context,
				int use_shared)
{
	struct cxip_rxc *rxc;

	rxc = calloc(1, sizeof(*rxc));
	if (!rxc)
		return NULL;

	dlist_init(&rxc->ep_list);
	fastlock_init(&rxc->lock);
	ofi_atomic_initialize32(&rxc->orx_reqs, 0);

	rxc->ctx.fid.fclass = FI_CLASS_RX_CTX;
	rxc->ctx.fid.context = context;
	rxc->attr = *attr;
	rxc->use_shared = use_shared;

	fastlock_init(&rxc->rx_lock);
	ofi_atomic_initialize32(&rxc->oflow_bufs_submitted, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_linked, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_in_use, 0);
	dlist_init(&rxc->oflow_bufs);
	dlist_init(&rxc->deferred_events);
	ofi_atomic_initialize32(&rxc->sink_le_linked, 0);
	dlist_init(&rxc->fc_drops);
	dlist_init(&rxc->msg_queue);
	dlist_init(&rxc->replay_queue);
	dlist_init(&rxc->sw_ux_list);
	rxc->pte_state = C_PTLTE_DISABLED;
	rxc->disabling = false;

	rxc->rdzv_threshold = cxip_env.rdzv_threshold;
	rxc->oflow_buf_size = cxip_env.oflow_buf_size;
	rxc->oflow_bufs_max = cxip_env.oflow_buf_count;

	/* TODO make configurable */
	rxc->min_multi_recv = CXIP_EP_MIN_MULTI_RECV;

	return rxc;
}

/*
 * cxip_rxc_free() - Free an RX context allocated using cxip_rxc_alloc()
 */
void cxip_rxc_free(struct cxip_rxc *rxc)
{
	rxc_disable(rxc);
	fastlock_destroy(&rxc->lock);
	free(rxc);
}
