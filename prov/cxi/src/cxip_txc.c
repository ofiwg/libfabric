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

/* Caller must hold txc->lock */
int cxip_tx_ctx_alloc_rdzv_id(struct cxip_tx_ctx *txc)
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

/* Caller must hold txc->lock */
int cxip_tx_ctx_free_rdzv_id(struct cxip_tx_ctx *txc, int tag)
{
	int idx = tag >> 4;
	int bit = (tag & 0xF);
	uint16_t clear_bitmask = 1 << bit;

	if (idx >= CXIP_RDZV_BM_LEN || idx < 0)
		return -FI_EINVAL;

	txc->rdzv_ids.bitmap[idx] &= ~clear_bitmask;

	return FI_SUCCESS;
}

/* Caller must hold txc->lock */
static int tx_ctx_msg_init(struct cxip_tx_ctx *txc)
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

/* Caller must hold txc->lock */
static int tx_ctx_msg_fini(struct cxip_tx_ctx *txc)
{
	cxip_pte_free(txc->rdzv_pte);

	return FI_SUCCESS;
}

int cxip_tx_ctx_enable(struct cxip_tx_ctx *txc)
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
		ret = tx_ctx_msg_init(txc);
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

static void tx_ctx_disable(struct cxip_tx_ctx *txc)
{
	int ret;

	fastlock_acquire(&txc->lock);

	if (!txc->enabled)
		goto unlock;

	if ((txc->attr.caps & (FI_TAGGED | FI_SEND)) == (FI_TAGGED | FI_SEND)) {
		ret = tx_ctx_msg_fini(txc);
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

static struct cxip_tx_ctx *tx_context_alloc(const struct fi_tx_attr *attr,
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
	tx_ctx->eager_threshold = CXIP_EAGER_THRESHOLD;

	if (getenv("RDZV_OFFLOAD")) {
		tx_ctx->rdzv_offload = 1;
		fprintf(stderr, "Rendezvous offload enabled\n");
	}

	return tx_ctx;

err:
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
	return NULL;
}

struct cxip_tx_ctx *cxip_tx_ctx_alloc(const struct fi_tx_attr *attr,
				      void *context, int use_shared)
{
	return tx_context_alloc(attr, context, use_shared,
				     FI_CLASS_TX_CTX);
}

struct cxip_tx_ctx *cxip_stx_ctx_alloc(const struct fi_tx_attr *attr,
				       void *context)
{
	return tx_context_alloc(attr, context, 0, FI_CLASS_STX_CTX);
}

void cxip_tx_ctx_free(struct cxip_tx_ctx *tx_ctx)
{
	tx_ctx_disable(tx_ctx);
	fastlock_destroy(&tx_ctx->lock);
	free(tx_ctx);
}
