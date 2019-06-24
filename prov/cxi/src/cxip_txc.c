/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

/* CXI TX Context Management */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * txc_msg_init() - Initialize an RX context for messaging.
 *
 * Allocates and initializes hardware resources used for transmitting messages.
 *
 * Caller must hold txc->lock.
 */
static int txc_msg_init(struct cxip_txc *txc)
{
	int ret;
	union c_cmdu cmd = {};
	struct cxi_cq_alloc_opts cq_opts = {};
	struct cxi_pt_alloc_opts pt_opts = {
		.is_matching = 1,
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY
	};
	uint64_t pid_idx;

	/* Allocate TGQ for posting source data */
	cq_opts.count = txc->attr.size;
	cq_opts.is_transmit = 0;
	ret = cxip_cmdq_alloc(txc->domain->dev_if, NULL, &cq_opts,
			      &txc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TGQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	/* Reserve the Rendezvous Send PTE */
	pid_idx = txc->domain->dev_if->if_dev->info.rdzv_get_idx;
	ret = cxip_pte_alloc(txc->ep_obj->if_dom, txc->send_cq->evtq,
			     pid_idx, &pt_opts, &txc->rdzv_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RDZV PTE: %d\n", ret);
		goto free_rx_cmdq;
	}

	/* Enable the Rendezvous PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = txc->rdzv_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(txc->rx_cmdq->dev_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_rdzv_pte;
	}

	cxi_cq_ring(txc->rx_cmdq->dev_cmdq);

	/* Wait for Rendezvous PTE state changes */
	do {
		sched_yield();
		cxip_cq_progress(txc->send_cq);
	} while (txc->rdzv_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_rdzv_pte:
	cxip_pte_free(txc->rdzv_pte);
free_rx_cmdq:
	cxip_cmdq_free(txc->rx_cmdq);

	return ret;
}

/*
 * txc_msg_fini() - Finalize TX context messaging.
 *
 * Free hardware resources allocated when the TX context was initialized for
 * messaging.
 *
 * Caller must hold txc->lock.
 */
static int txc_msg_fini(struct cxip_txc *txc)
{
	cxip_pte_free(txc->rdzv_pte);
	cxip_cmdq_free(txc->rx_cmdq);

	return FI_SUCCESS;
}

/*
 * cxip_txc_enable() - Enable a TX context for use.
 *
 * Called via fi_enable(). The context could be used in a standard endpoint or
 * a scalable endpoint.
 */
int cxip_txc_enable(struct cxip_txc *txc)
{
	int ret = FI_SUCCESS;
	struct cxi_cq_alloc_opts cq_opts = {};

	fastlock_acquire(&txc->lock);

	if (txc->enabled)
		goto unlock;

	if (!txc->send_cq) {
		CXIP_LOG_DBG("Undefined send CQ\n");
		ret = -FI_ENOCQ;
		goto unlock;
	}

	ret = cxip_cq_enable(txc->send_cq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_cq_enable returned: %d\n", ret);
		goto unlock;
	}

	if (txc->send_cntr) {
		ret = cxip_cntr_enable(txc->send_cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable(FI_SEND) returned: %d\n",
				     ret);
			goto unlock;
		}
	}

	if (txc->write_cntr) {
		ret = cxip_cntr_enable(txc->write_cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable(FI_WRITE) returned: %d\n",
				     ret);
			goto unlock;
		}
	}

	if (txc->read_cntr) {
		ret = cxip_cntr_enable(txc->read_cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable(FI_READ) returned: %d\n",
				     ret);
			goto unlock;
		}
	}

	/* An IDC command can use up to 4 64 byte slots. */
	cq_opts.count = txc->attr.size * 4;
	cq_opts.is_transmit = 1;
	cq_opts.lcid = txc->domain->dev_if->cps[0]->lcid;
	ret = cxip_cmdq_alloc(txc->domain->dev_if, NULL, &cq_opts,
			      &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_init(txc);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("Unable to init TX CTX, ret: %d\n", ret);
			goto free_tx_cmdq;
		}
	}

	txc->enabled = 1;
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

free_tx_cmdq:
	cxip_cmdq_free(txc->tx_cmdq);
unlock:
	fastlock_release(&txc->lock);

	return ret;
}

/*
 * txc_cleanup() - Attempt to free outstanding requests.
 *
 * Outstanding commands may be dropped when the TX Command Queue is freed.
 * This leads to missing events. Attempt to gather all events before freeing
 * the TX CQ. If events go missing, resources will be leaked until the
 * Completion Queue is freed.
 */
static void txc_cleanup(struct cxip_txc *txc)
{
	uint64_t start;

	if (!ofi_atomic_get32(&txc->otx_reqs))
		return;

	cxip_cq_req_discard(txc->send_cq, txc);

	start = fi_gettime_ms();
	while (ofi_atomic_get32(&txc->otx_reqs)) {
		sched_yield();
		cxip_cq_progress(txc->send_cq);

		if (fi_gettime_ms() - start > CXIP_REQ_CLEANUP_TO) {
			CXIP_LOG_ERROR(
				"Timeout waiting for outstanding requests.");
			break;
		}
	}
}

/*
 * cxip_txc_disable() - Disable a TX context.
 *
 * Free hardware resources allocated when the context was enabled. Called via
 * fi_close(). The context could be used in a standard endpoint or a scalable
 * endpoint.
 */
static void txc_disable(struct cxip_txc *txc)
{
	int ret;

	fastlock_acquire(&txc->lock);

	if (!txc->enabled)
		goto unlock;

	txc->enabled = 0;

	txc_cleanup(txc);

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_fini(txc);
		if (ret)
			CXIP_LOG_ERROR("Unable to destroy TX CTX, ret: %d\n",
				       ret);
	}

	cxip_cmdq_free(txc->tx_cmdq);
unlock:
	fastlock_release(&txc->lock);
}

/*
 * txc_alloc() - Allocate a TX context.
 *
 * Used to support creating a TX context for fi_endpoint() or fi_tx_context().
 */
static struct cxip_txc *txc_alloc(const struct fi_tx_attr *attr, void *context,
				  int use_shared, size_t fclass)
{
	struct cxip_txc *txc;

	txc = calloc(sizeof(*txc), 1);
	if (!txc)
		return NULL;

	dlist_init(&txc->ep_list);
	fastlock_init(&txc->lock);
	ofi_atomic_initialize32(&txc->otx_reqs, 0);

	switch (fclass) {
	case FI_CLASS_TX_CTX:
		txc->fid.ctx.fid.fclass = FI_CLASS_TX_CTX;
		txc->fid.ctx.fid.context = context;
		txc->fclass = FI_CLASS_TX_CTX;
		txc->use_shared = use_shared;
		break;
	case FI_CLASS_STX_CTX:
		txc->fid.stx.fid.fclass = FI_CLASS_STX_CTX;
		txc->fid.stx.fid.context = context;
		txc->fclass = FI_CLASS_STX_CTX;
		break;
	default:
		goto err;
	}

	txc->attr = *attr;
	txc->eager_threshold = CXIP_EAGER_THRESHOLD;

	return txc;

err:
	fastlock_destroy(&txc->lock);
	free(txc);
	return NULL;
}

/*
 * cxip_stx_alloc() - Allocate a regular (not shared) TX context.
 */
struct cxip_txc *cxip_txc_alloc(const struct fi_tx_attr *attr, void *context,
				int use_shared)
{
	return txc_alloc(attr, context, use_shared, FI_CLASS_TX_CTX);
}

/*
 * cxip_stx_alloc() - Allocate a shared TX context.
 */
struct cxip_txc *cxip_stx_alloc(const struct fi_tx_attr *attr, void *context)
{
	return txc_alloc(attr, context, 0, FI_CLASS_STX_CTX);
}

/*
 * cxip_txc_free() - Free a TX context allocated using cxip_txc_alloc()
 */
void cxip_txc_free(struct cxip_txc *txc)
{
	txc_disable(txc);
	fastlock_destroy(&txc->lock);
	free(txc);
}
