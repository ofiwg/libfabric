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
 * cxip_rdzv_pte_cb() - Process rendezvous source PTE state change events.
 */
void cxip_rdzv_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state)
{
	struct cxip_txc *txc = (struct cxip_txc *)pte->ctx;

	switch (state) {
	case C_PTLTE_ENABLED:
		assert(txc->pte_state == C_PTLTE_DISABLED);
		break;
	default:
		CXIP_LOG_ERROR("Unexpected state received: %u\n", state);
	}

	txc->pte_state = state;
}

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
	struct cxi_pt_alloc_opts pt_opts = {
		.is_matching = 1,
	};
	uint64_t pid_idx;

	/* Allocate TGQ for posting source data */
	ret = cxip_ep_cmdq(txc->ep_obj, txc->tx_id, false, FI_TC_UNSPEC,
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
	pid_idx = txc->domain->iface->dev->info.rdzv_get_idx;
	ret = cxip_pte_alloc(txc->ep_obj->if_dom, txc->send_cq->evtq,
			     pid_idx, &pt_opts, cxip_rdzv_pte_cb, txc,
			     &txc->rdzv_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RDZV PTE: %d\n", ret);
		goto put_rx_cmdq;
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
	} while (txc->pte_state != C_PTLTE_ENABLED);

	ret = cxip_txc_zbp_init(txc);
	if (ret) {
		CXIP_LOG_ERROR("Failed to initialize ZBP: %d\n", ret);
		goto free_rdzv_pte;
	}

	CXIP_LOG_DBG("TXC RDZV PtlTE enabled: %p\n", txc);

	return FI_SUCCESS;

free_rdzv_pte:
	cxip_pte_free(txc->rdzv_pte);
put_rx_cmdq:
	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, false);

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
	cxip_txc_zbp_fini(txc);
	cxip_txc_rdzv_src_fini(txc);
	cxip_pte_free(txc->rdzv_pte);
	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, false);

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

	ret = cxip_ep_cmdq(txc->ep_obj, txc->tx_id, true, txc->tclass,
			   &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	/* Make sure the TX CQ TC matches the TXC TC. */
	ret = cxip_cp_get(txc->domain->lni, txc->ep_obj->auth_key.vni,
			  cxip_ofi_to_cxi_tc(txc->tclass), &txc->cp);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to get CP: %d\n", ret);
		goto put_tx_cmdq;
	}

	fastlock_acquire(&txc->tx_cmdq->lock);
	ret = cxi_cq_emit_cq_lcid(txc->tx_cmdq->dev_cmdq, txc->cp->lcid);
	if (ret)
		CXIP_LOG_ERROR("Failed to update CMDQ(%p) CP: %d\n",
			       txc->tx_cmdq, ret);
	else
		cxi_cq_ring(txc->tx_cmdq->dev_cmdq);
	fastlock_release(&txc->tx_cmdq->lock);

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_init(txc);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("Unable to init TX CTX, ret: %d\n", ret);
			goto put_tx_cmdq;
		}
	}

	txc->pid_bits = txc->domain->iface->dev->info.pid_bits;
	txc->enabled = true;

	fastlock_release(&txc->lock);

	return FI_SUCCESS;

put_tx_cmdq:
	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, true);
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
	struct cxip_fc_peer *fc_peer;
	struct dlist_entry *tmp;

	if (!ofi_atomic_get32(&txc->otx_reqs))
		return;

	cxip_cq_req_discard(txc->send_cq, txc);

	start = ofi_gettime_ms();
	while (ofi_atomic_get32(&txc->otx_reqs)) {
		sched_yield();
		cxip_cq_progress(txc->send_cq);

		if (ofi_gettime_ms() - start > CXIP_REQ_CLEANUP_TO) {
			CXIP_LOG_ERROR("Timeout waiting for outstanding requests.\n");
			break;
		}
	}

	dlist_foreach_container_safe(&txc->fc_peers, struct cxip_fc_peer,
				     fc_peer, txc_entry, tmp) {
		dlist_remove(&fc_peer->txc_entry);
		free(fc_peer);
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

	if (!txc->enabled) {
		fastlock_release(&txc->lock);
		return;
	}

	txc->enabled = false;

	fastlock_release(&txc->lock);

	txc_cleanup(txc);

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_fini(txc);
		if (ret)
			CXIP_LOG_ERROR("Unable to destroy TX CTX, ret: %d\n",
				       ret);
	}

	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, true);
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
	fastlock_init(&txc->rdzv_src_lock);
	ofi_atomic_initialize32(&txc->otx_reqs, 0);
	ofi_atomic_initialize32(&txc->zbp_le_linked, 0);
	ofi_atomic_initialize32(&txc->rdzv_src_lacs, 0);
	dlist_init(&txc->rdzv_src_reqs);
	dlist_init(&txc->msg_queue);
	dlist_init(&txc->fc_peers);
	txc->pte_state = C_PTLTE_DISABLED;

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
	txc->rdzv_threshold = cxip_env.rdzv_threshold;

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

void cxip_txc_flush_msg_trig_reqs(struct cxip_txc *txc)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;

	fastlock_acquire(&txc->lock);

	/* Drain the message queue. */
	dlist_foreach_container_safe(&txc->msg_queue, struct cxip_req, req,
				     send.txc_entry, tmp) {
		if (cxip_is_trig_req(req)) {
			ofi_atomic_dec32(&txc->otx_reqs);
			cxip_unmap(req->send.send_md);
			cxip_cq_req_free(req);
		}
	}

	fastlock_release(&txc->lock);
}
