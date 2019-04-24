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
 * cxip_txc_alloc_rdzv_id() - Allocate a rendezvous ID.
 *
 * Caller must hold txc->lock.
 */
int cxip_txc_alloc_rdzv_id(struct cxip_txc *txc)
{
	int rc;

	/* Find a bitmap with cleared bits */
	for (int idx = 0; idx < CXIP_RDZV_BM_LEN; idx++) {
		if (txc->rdzv_ids.bitmap[idx] != 0xFFFF)

			/* Find the lowest cleared bit */
			for (int bit = 0; bit < 16; bit++)
				if (!(txc->rdzv_ids.bitmap[idx] & (1 << bit))) {

					/* Set the bit and save the context */
					txc->rdzv_ids.bitmap[idx] |= (1 << bit);
					rc = (idx << 4) | bit;

					return rc;
				}
	}

	/* No bitmap has a cleared bit */
	return -FI_ENOSPC;
}

/*
 * cxip_txc_free_rdzv_id() - Free a rendezvous ID.
 *
 * Caller must hold txc->lock.
 */
int cxip_txc_free_rdzv_id(struct cxip_txc *txc, int tag)
{
	int idx = tag >> 4;
	int bit = (tag & 0xF);
	uint16_t clear_bitmask = 1 << bit;

	if (idx >= CXIP_RDZV_BM_LEN || idx < 0)
		return -FI_EINVAL;

	txc->rdzv_ids.bitmap[idx] &= ~clear_bitmask;

	return FI_SUCCESS;
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
	struct cxi_pt_alloc_opts opts = { .is_matching = 1 };
	uint64_t pid_idx;

	/* initialize the rendezvous ID structure */
	memset(&txc->rdzv_ids, 0, sizeof(txc->rdzv_ids));

	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		opts.use_logical = 1;
	}

	/* Reserve the Rendezvous Send PTE */
	pid_idx = txc->domain->dev_if->if_dev->info.rdzv_get_idx;
	ret = cxip_pte_alloc(txc->ep_obj->if_dom, txc->comp.send_cq->evtq,
			     pid_idx, &opts, &txc->rdzv_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RDZV PTE: %d\n", ret);
		return ret;
	}

	/* Enable the Rendezvous PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = txc->rdzv_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(txc->rx_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_rdzv_pte;
	}

	cxi_cq_ring(txc->rx_cmdq);

	/* Wait for Rendezvous PTE state changes */
	do {
		sched_yield();
		cxip_cq_progress(txc->comp.send_cq);
	} while (txc->rdzv_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_rdzv_pte:
	cxip_pte_free(txc->rdzv_pte);

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
	struct cxi_cq_alloc_opts opts;

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
	memset(&opts, 0, sizeof(opts));
	opts.count = 64;
	opts.is_transmit = 1;
	opts.lcid = txc->domain->dev_if->cps[0]->lcid;
	ret = cxil_alloc_cmdq(txc->domain->dev_if->if_lni, NULL, &opts,
			      &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	/* Allocate a target-side cmdq for Rendezvous buffers */
	opts.count = 64;
	opts.is_transmit = 0;
	ret = cxil_alloc_cmdq(txc->domain->dev_if->if_lni, NULL, &opts,
			      &txc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate tgt_sd CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	if ((txc->attr.caps & (FI_TAGGED | FI_SEND)) == (FI_TAGGED | FI_SEND)) {
		ret = txc_msg_init(txc);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("Unable to init TX CTX, ret: %d\n", ret);
			goto unlock;
		}
	}

	txc->enabled = 1;
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&txc->lock);

	return ret;
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

	if ((txc->attr.caps & (FI_TAGGED | FI_SEND)) == (FI_TAGGED | FI_SEND)) {
		ret = txc_msg_fini(txc);
		if (ret)
			CXIP_LOG_ERROR("Unable to destroy TX CTX, ret: %d\n",
				       ret);
	}

	ret = cxil_destroy_cmdq(txc->rx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy RX CMDQ, ret: %d\n", ret);

	ret = cxil_destroy_cmdq(txc->tx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy TX CMDQ, ret: %d\n", ret);

	txc->enabled = 0;
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

	dlist_init(&txc->cq_entry);
	dlist_init(&txc->ep_list);
	fastlock_init(&txc->lock);

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
	txc->attr.op_flags |= FI_TRANSMIT_COMPLETE;
	txc->eager_threshold = CXIP_EAGER_THRESHOLD;

	if (getenv("RDZV_OFFLOAD")) {
		txc->rdzv_offload = 1;
		fprintf(stderr, "Rendezvous offload enabled\n");
	}

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
