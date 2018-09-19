/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

/* Caller must hold rxc->lock */
static int cxip_rx_ctx_recv_init(struct cxip_rx_ctx *rxc)
{
	int ret;
	union c_cmdu cmd = {};
	struct cxi_pt_alloc_opts opts = {};
	uint64_t pid_off;

	/* Select the LEP where the queue will be mapped */
	pid_off = CXIP_ADDR_RX_IDX(rxc->domain->dev_if->if_pid_granule, 0);

	ret = cxip_pte_alloc(rxc->ep_attr->if_dom, rxc->comp.recv_cq->evtq,
			     pid_off, &opts, &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RX PTE: %d\n", ret);
		return ret;
	}

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = rxc->rx_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(rxc->rx_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_pte;
	}

	cxi_cq_ring(rxc->rx_cmdq);

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_cq_progress(rxc->comp.recv_cq);
	} while (rxc->rx_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(rxc->rx_pte);

	return ret;
}

/* Caller must hold rxc->lock */
static int cxip_rx_ctx_recv_fini(struct cxip_rx_ctx *rxc)
{
	cxip_pte_free(rxc->rx_pte);

	return FI_SUCCESS;
}

int cxip_rx_ctx_enable(struct cxip_rx_ctx *rxc)
{
	int ret = FI_SUCCESS;

	fastlock_acquire(&rxc->lock);

	if (rxc->enabled)
		goto unlock;

	if (rxc->comp.recv_cq) {
		ret = cxip_cq_enable(rxc->comp.recv_cq);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cq_enable returned: %d\n", ret);
			goto unlock;
		}
	}

	/* TODO set CMDQ size with RX attrs */
	ret = cxil_alloc_cmdq(rxc->domain->dev_if->if_lni, 64, 0,
			      &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate RX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	ret = cxip_rx_ctx_recv_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_rx_ctx_recv_init returned: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_cmdq;
	}

	rxc->enabled = 1;

	fastlock_release(&rxc->lock);

	/* Allocate pool of overflow buffers */
	cxip_rxc_oflow_replenish(rxc);

	return FI_SUCCESS;

free_cmdq:
	ret = cxil_destroy_cmdq(rxc->rx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy RX CMDQ, ret: %d\n", ret);
unlock:
	fastlock_release(&rxc->lock);

	return ret;
}

static void cxip_rx_ctx_disable(struct cxip_rx_ctx *rxc)
{
	int ret;

	fastlock_acquire(&rxc->lock);

	if (!rxc->enabled)
		goto unlock;

	/* Free pool of overflow buffers */
	cxip_rxc_oflow_cleanup(rxc);

	ret = cxip_rx_ctx_recv_fini(rxc);
	if (ret)
		CXIP_LOG_ERROR("cxip_rx_ctx_recv_fini returned: %d\n", ret);

	ret = cxil_destroy_cmdq(rxc->rx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy RX CMDQ, ret: %d\n", ret);

	rxc->enabled = 0;
unlock:
	fastlock_release(&rxc->lock);
}

struct cxip_rx_ctx *cxip_rx_ctx_alloc(const struct fi_rx_attr *attr,
				      void *context, int use_shared)
{
	struct cxip_rx_ctx *rx_ctx;

	rx_ctx = calloc(1, sizeof(*rx_ctx));
	if (!rx_ctx)
		return NULL;

	dlist_init(&rx_ctx->cq_entry);
	dlist_init(&rx_ctx->ep_list);
	fastlock_init(&rx_ctx->lock);

	rx_ctx->ctx.fid.fclass = FI_CLASS_RX_CTX;
	rx_ctx->ctx.fid.context = context;
	rx_ctx->num_left = attr->size;
	rx_ctx->attr = *attr;
	rx_ctx->use_shared = use_shared;

	ofi_atomic_initialize32(&rx_ctx->oflow_buf_cnt, 0);
	dlist_init(&rx_ctx->oflow_bufs);
	dlist_init(&rx_ctx->ux_sends);
	dlist_init(&rx_ctx->ux_recvs);

	/* TODO make configurable */
	rx_ctx->eager_threshold = 1024;
	rx_ctx->oflow_bufs_max = 3;
	rx_ctx->oflow_msgs_max = 2 * 1024;
	rx_ctx->oflow_buf_size = rx_ctx->oflow_msgs_max *
			rx_ctx->eager_threshold;

	return rx_ctx;
}

void cxip_rx_ctx_free(struct cxip_rx_ctx *rx_ctx)
{
	cxip_rx_ctx_disable(rx_ctx);
	fastlock_destroy(&rx_ctx->lock);
	free(rx_ctx);
}

int cxip_tx_ctx_enable(struct cxip_tx_ctx *txc)
{
	int ret = FI_SUCCESS;

	fastlock_acquire(&txc->lock);

	if (txc->enabled)
		goto unlock;

	if (txc->comp.send_cq) {
		ret = cxip_cq_enable(txc->comp.send_cq);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cq_enable returned: %d\n", ret);
			goto unlock;
		}
	}

	/* TODO set CMDQ size with TX attrs */
	ret = cxil_alloc_cmdq(txc->domain->dev_if->if_lni, 64, 1,
			      &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	txc->enabled = 1;
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&txc->lock);

	return ret;
}

static void cxip_tx_ctx_disable(struct cxip_tx_ctx *txc)
{
	int ret;

	fastlock_acquire(&txc->lock);

	if (!txc->enabled)
		goto unlock;

	ret = cxil_destroy_cmdq(txc->tx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy TX CMDQ, ret: %d\n", ret);

	txc->enabled = 0;
unlock:
	fastlock_release(&txc->lock);
}

static struct cxip_tx_ctx *cxip_tx_context_alloc(const struct fi_tx_attr *attr,
						 void *context, int use_shared,
						 size_t fclass)
{
	struct cxip_tx_ctx *tx_ctx;

	tx_ctx = calloc(sizeof(*tx_ctx), 1);
	if (!tx_ctx)
		return NULL;

	dlist_init(&tx_ctx->cq_entry);
	dlist_init(&tx_ctx->ep_list);
	fastlock_init(&tx_ctx->lock);

	switch (fclass) {
	case FI_CLASS_TX_CTX:
		tx_ctx->fid.ctx.fid.fclass = FI_CLASS_TX_CTX;
		tx_ctx->fid.ctx.fid.context = context;
		tx_ctx->fclass = FI_CLASS_TX_CTX;
		tx_ctx->use_shared = use_shared;
		break;
	case FI_CLASS_STX_CTX:
		tx_ctx->fid.stx.fid.fclass = FI_CLASS_STX_CTX;
		tx_ctx->fid.stx.fid.context = context;
		tx_ctx->fclass = FI_CLASS_STX_CTX;
		break;
	default:
		goto err;
	}
	tx_ctx->attr = *attr;
	tx_ctx->attr.op_flags |= FI_TRANSMIT_COMPLETE;

	return tx_ctx;

err:
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
	return NULL;
}

struct cxip_tx_ctx *cxip_tx_ctx_alloc(const struct fi_tx_attr *attr,
				      void *context, int use_shared)
{
	return cxip_tx_context_alloc(attr, context, use_shared,
				     FI_CLASS_TX_CTX);
}

struct cxip_tx_ctx *cxip_stx_ctx_alloc(const struct fi_tx_attr *attr,
				       void *context)
{
	return cxip_tx_context_alloc(attr, context, 0, FI_CLASS_STX_CTX);
}

void cxip_tx_ctx_free(struct cxip_tx_ctx *tx_ctx)
{
	cxip_tx_ctx_disable(tx_ctx);
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
}
