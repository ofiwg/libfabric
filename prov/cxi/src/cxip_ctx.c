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
static int rx_ctx_recv_init(struct cxip_rx_ctx *rxc)
{
	int ret;
	union c_cmdu cmd = {};
	struct cxi_pt_alloc_opts opts = { .is_matching = 1 };
	uint64_t pid_idx;

	/* Select the LEP where the queue will be mapped */
	pid_idx = CXIP_RXC_TO_IDX(rxc->rx_id);

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->if_dom, rxc->comp.recv_cq->evtq,
			     pid_idx, &opts, &rxc->rx_pte);
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
		goto free_rx_pte;
	}

	cxi_cq_ring(rxc->rx_cmdq);

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_cq_progress(rxc->comp.recv_cq);
	} while (rxc->rx_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_rx_pte:
	cxip_pte_free(rxc->rx_pte);

	return ret;
}

/* Caller must hold rxc->lock */
static int rx_ctx_recv_fini(struct cxip_rx_ctx *rxc)
{
	cxip_pte_free(rxc->rx_pte);

	return FI_SUCCESS;
}

int cxip_rx_ctx_enable(struct cxip_rx_ctx *rxc)
{
	int ret = FI_SUCCESS;
	int tmp;
	struct cxi_cq_alloc_opts opts;

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
	memset(&opts, 0, sizeof(opts));
	opts.count = 64;
	opts.is_transmit = 0;
	ret = cxil_alloc_cmdq(rxc->domain->dev_if->if_lni, NULL, &opts,
			      &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate RX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	/* Create transmit cmdq for SW Rendezvous GET operations */
	/* TODO set CMDQ size with RX attrs */
	opts.count = 64;
	opts.is_transmit = 1;
	opts.lcid = rxc->domain->dev_if->cps[0]->lcid;
	ret = cxil_alloc_cmdq(rxc->domain->dev_if->if_lni, NULL, &opts,
			      &rxc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_rx_cmdq;
	}

	ret = rx_ctx_recv_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("rx_ctx_recv_init returned: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_tx_cmdq;
	}

	rxc->enabled = 1;

	fastlock_release(&rxc->lock);

	/* Initialize tagged messaging */
	ret = cxip_rxc_tagged_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("cxip_rxc_tagged_init returned: %d\n", ret);
		goto free_tx_cmdq;
	}

	return FI_SUCCESS;

free_tx_cmdq:
	tmp = cxil_destroy_cmdq(rxc->tx_cmdq);
	if (tmp)
		CXIP_LOG_ERROR("Unable to destroy TX CMDQ, ret: %d\n", tmp);
free_rx_cmdq:
	tmp = cxil_destroy_cmdq(rxc->rx_cmdq);
	if (tmp)
		CXIP_LOG_ERROR("Unable to destroy RX CMDQ, ret: %d\n", tmp);
unlock:
	fastlock_release(&rxc->lock);

	return ret;
}

static void rx_ctx_disable(struct cxip_rx_ctx *rxc)
{
	int ret;

	fastlock_acquire(&rxc->lock);

	if (!rxc->enabled)
		goto unlock;

	cxip_rxc_tagged_fini(rxc);

	ret = rx_ctx_recv_fini(rxc);
	if (ret)
		CXIP_LOG_ERROR("rx_ctx_recv_fini returned: %d\n", ret);

	ret = cxil_destroy_cmdq(rxc->rx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy RX CMDQ, ret: %d\n", ret);

	ret = cxil_destroy_cmdq(rxc->tx_cmdq);
	if (ret)
		CXIP_LOG_ERROR("Unable to destroy TX CMDQ, ret: %d\n", ret);

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
	ofi_atomic_initialize32(&rx_ctx->ux_rdvs_buf.ref, 0);
	dlist_init(&rx_ctx->oflow_bufs);
	dlist_init(&rx_ctx->ux_sends);
	dlist_init(&rx_ctx->ux_recvs);

	/* TODO make configurable */
	rx_ctx->eager_threshold = CXIP_EAGER_THRESHOLD;
	rx_ctx->oflow_bufs_max = CXIP_MAX_OFLOW_BUFS;
	rx_ctx->oflow_msgs_max = CXIP_MAX_OFLOW_MSGS;
	rx_ctx->oflow_buf_size = CXIP_MAX_OFLOW_MSGS * CXIP_EAGER_THRESHOLD;

	return rx_ctx;
}

void cxip_rx_ctx_free(struct cxip_rx_ctx *rx_ctx)
{
	rx_ctx_disable(rx_ctx);
	fastlock_destroy(&rx_ctx->lock);
	free(rx_ctx);
}

/* Caller must hold txc->lock */
int cxip_tx_ctx_alloc_rdvs_id(struct cxip_tx_ctx *txc)
{
	int rc;

	/* Find a bitmap with cleared bits */
	for (int idx = 0; idx < CXIP_RDVS_BM_LEN; idx++) {
		if (txc->rdvs_ids.bitmap[idx] != 0xFFFF)

			/* Find the lowest cleared bit */
			for (int bit = 0; bit < 16; bit++)
				if (!(txc->rdvs_ids.bitmap[idx] & (1 << bit))) {

					/* Set the bit and save the context */
					txc->rdvs_ids.bitmap[idx] |= (1 << bit);
					rc = (idx << 4) | bit;

					return rc;
				}
	}

	/* No bitmap has a cleared bit */
	return -FI_ENOSPC;
}

/* Caller must hold txc->lock */
int cxip_tx_ctx_free_rdvs_id(struct cxip_tx_ctx *txc, int tag)
{
	int idx = tag >> 4;
	int bit = (tag & 0xF);
	uint16_t clear_bitmask = 1 << bit;

	if (idx >= CXIP_RDVS_BM_LEN || idx < 0)
		return -FI_EINVAL;

	txc->rdvs_ids.bitmap[idx] &= ~clear_bitmask;

	return FI_SUCCESS;
}

/* Caller must hold txc->lock */
static int tx_ctx_recv_init(struct cxip_tx_ctx *txc)
{
	int ret;
	union c_cmdu cmd = {};
	struct cxi_pt_alloc_opts opts = { .is_matching = 1 };
	uint64_t pid_idx;

	/* initialize the rendezvous ID structure */
	memset(&txc->rdvs_ids, 0, sizeof(txc->rdvs_ids));

	if (txc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_LOG_DBG("Using logical PTE matching\n");
		opts.use_logical = 1;
	}

	/* Reserve the Rendezvous Send PTE */
	pid_idx = CXIP_RDVS_IDX(txc->domain->dev_if->if_pid_granule);
	ret = cxip_pte_alloc(txc->ep_obj->if_dom, txc->comp.send_cq->evtq,
			     pid_idx, &opts, &txc->rdvs_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate RDVS PTE: %d\n", ret);
		return ret;
	}

	/* Enable the Rendezvous PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = txc->rdvs_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(txc->rx_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_rdvs_pte;
	}

	cxi_cq_ring(txc->rx_cmdq);

	/* Wait for Rendezvous PTE state changes */
	do {
		sched_yield();
		cxip_cq_progress(txc->comp.send_cq);
	} while (txc->rdvs_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_rdvs_pte:
	cxip_pte_free(txc->rdvs_pte);

	return ret;
}

/* Caller must hold txc->lock */
static int tx_ctx_recv_fini(struct cxip_tx_ctx *txc)
{
	cxip_pte_free(txc->rdvs_pte);

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
		ret = tx_ctx_recv_init(txc);
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
		ret = tx_ctx_recv_fini(txc);
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
