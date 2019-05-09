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
static int rxc_msg_enable(struct cxip_rxc *rxc)
{
	int ret;
	union c_cmdu cmd = {};

	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = rxc->rx_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

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
		cxip_cq_progress(rxc->comp.recv_cq);
	} while (rxc->rx_pte->state != C_PTLTE_ENABLED);

	CXIP_LOG_DBG("RXC PtlTE enabled: %p\n", rxc);

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
		cxip_cq_progress(rxc->comp.recv_cq);
	} while (rxc->rx_pte->state != C_PTLTE_DISABLED);

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
	struct cxi_cq_alloc_opts cq_opts = {};
	struct cxi_pt_alloc_opts pt_opts = {
		.is_matching = 1,
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY
	};
	uint64_t pid_idx;

	/* TODO set CMDQ size with RX attrs */
	cq_opts.count = 64;
	cq_opts.is_transmit = 0;
	ret = cxip_cmdq_alloc(rxc->domain->dev_if, NULL, &cq_opts,
			      &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate RX CMDQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	/* TODO set CMDQ size with RX attrs */
	cq_opts.count = 64;
	cq_opts.is_transmit = 1;
	cq_opts.lcid = rxc->domain->dev_if->cps[0]->lcid;
	ret = cxip_cmdq_alloc(rxc->domain->dev_if, NULL, &cq_opts,
			      &rxc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_rx_cmdq;
	}

	/* Select the LEP where the queue will be mapped */
	pid_idx = CXIP_RXC_TO_IDX(rxc->rx_id);

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->if_dom, rxc->comp.recv_cq->evtq,
			     pid_idx, &pt_opts, &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RX PTE: %d\n", ret);
		goto free_tx_cmdq;
	}

	return FI_SUCCESS;

free_tx_cmdq:
	cxip_cmdq_free(rxc->tx_cmdq);
free_rx_cmdq:
	cxip_cmdq_free(rxc->rx_cmdq);

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

	cxip_cmdq_free(rxc->rx_cmdq);

	cxip_cmdq_free(rxc->tx_cmdq);

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

	if (!rxc->comp.recv_cq) {
		CXIP_LOG_DBG("Undefined recv CQ\n");
		ret = -FI_ENOCQ;
		goto unlock;
	}

	ret = cxip_cq_enable(rxc->comp.recv_cq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_cq_enable returned: %d\n", ret);
		goto unlock;
	}

	ret = rxc_msg_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("rxc_msg_init returned: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	ret = cxip_msg_oflow_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_msg_oflow_init returned: %d\n", ret);
		goto msg_fini;
	}

	/* Start accepting Puts. */
	ret = rxc_msg_enable(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("rxc_msg_disable returned: %d\n", ret);
		goto oflow_fini;
	}

	rxc->enabled = 1;

	fastlock_release(&rxc->lock);

	return FI_SUCCESS;

oflow_fini:
	cxip_msg_oflow_fini(rxc);
msg_fini:
	ret = rxc_msg_fini(rxc);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("rxc_msg_fini returned: %d\n", ret);
unlock:
	fastlock_release(&rxc->lock);

	return ret;
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

	if (!rxc->enabled)
		goto unlock;

	/* Stop accepting Puts. */
	ret = rxc_msg_disable(rxc);
	if (ret != FI_SUCCESS)
		CXIP_LOG_DBG("rxc_msg_disable returned: %d\n", ret);

	/* Clean up overflow buffers. */
	cxip_msg_oflow_fini(rxc);

	/* Free hardware resources. */
	ret = rxc_msg_fini(rxc);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("rxc_msg_fini returned: %d\n", ret);

	rxc->enabled = 0;
unlock:
	fastlock_release(&rxc->lock);
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

	rxc->ctx.fid.fclass = FI_CLASS_RX_CTX;
	rxc->ctx.fid.context = context;
	rxc->num_left = attr->size;
	rxc->attr = *attr;
	rxc->use_shared = use_shared;

	fastlock_init(&rxc->rx_lock);
	ofi_atomic_initialize32(&rxc->oflow_bufs_submitted, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_linked, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_in_use, 0);
	dlist_init(&rxc->oflow_bufs);
	dlist_init(&rxc->ux_sends);
	dlist_init(&rxc->ux_recvs);
	dlist_init(&rxc->ux_rdzv_sends);
	dlist_init(&rxc->ux_rdzv_recvs);
	ofi_atomic_initialize32(&rxc->ux_sink_linked, 0);

	/* TODO make configurable */
	rxc->eager_threshold = CXIP_EAGER_THRESHOLD;
	rxc->oflow_bufs_max = CXIP_MAX_OFLOW_BUFS;
	rxc->oflow_msgs_max = CXIP_MAX_OFLOW_MSGS;
	rxc->oflow_buf_size = CXIP_MAX_OFLOW_MSGS * CXIP_EAGER_THRESHOLD;

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
