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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

struct cxip_md *cxip_txc_ibuf_md(void *ibuf)
{
	return ofi_buf_hdr(ibuf)->region->context;
}

/*
 * cxip_txc_ibuf_alloc() - Allocate an inject buffer.
 */
void *cxip_txc_ibuf_alloc(struct cxip_txc *txc)
{
	void *ibuf;

	ofi_spin_lock(&txc->ibuf_lock);
	ibuf = (struct cxip_req *)ofi_buf_alloc(txc->ibuf_pool);
	ofi_spin_unlock(&txc->ibuf_lock);

	if (ibuf)
		CXIP_DBG("Allocated inject buffer: %p\n", ibuf);
	else
		CXIP_WARN("Failed to allocate inject buffer\n");

	return ibuf;
}

/*
 * cxip_txc_ibuf_free() - Free an inject buffer.
 */
void cxip_txc_ibuf_free(struct cxip_txc *txc, void *ibuf)
{
	ofi_spin_lock(&txc->ibuf_lock);
	ofi_buf_free(ibuf);
	ofi_spin_unlock(&txc->ibuf_lock);

	CXIP_DBG("Freed inject buffer: %p\n", ibuf);
}

int cxip_ibuf_chunk_init(struct ofi_bufpool_region *region)
{
	struct cxip_txc *txc = region->pool->attr.context;
	struct cxip_md *md;
	int ret;

	ret = cxip_map(txc->domain, region->mem_region,
		       region->pool->region_size, OFI_MR_NOCACHE, &md);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map inject buffer chunk\n");
		return ret;
	}

	region->context = md;

	return FI_SUCCESS;
}

void cxip_ibuf_chunk_fini(struct ofi_bufpool_region *region)
{
	cxip_unmap(region->context);
}

int cxip_txc_ibuf_create(struct cxip_txc *txc)
{
	struct ofi_bufpool_attr bp_attrs = {};
	int ret;

	bp_attrs.size = CXIP_INJECT_SIZE;
	bp_attrs.alignment = 8;
	bp_attrs.max_cnt = UINT16_MAX;
	bp_attrs.chunk_cnt = 64;
	bp_attrs.alloc_fn = cxip_ibuf_chunk_init;
	bp_attrs.free_fn = cxip_ibuf_chunk_fini;
	bp_attrs.context = txc;

	ret = ofi_bufpool_create_attr(&bp_attrs, &txc->ibuf_pool);
	if (ret)
		ret = -FI_ENOMEM;

	return ret;
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

	/* Allocate TGQ for posting source data */
	ret = cxip_ep_cmdq(txc->ep_obj, txc->tx_id, false, FI_TC_UNSPEC,
			   txc->send_cq->eq.eq, &txc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate TGQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	ret = cxip_rdzv_pte_alloc(txc, &txc->rdzv_pte);
	if (ret) {
		CXIP_WARN("Failed to allocate rendezvous PtlTE: %d:%s\n", ret,
			  fi_strerror(-ret));
		goto err_put_rx_cmdq;
	}

	CXIP_DBG("TXC RDZV PtlTE enabled: %p\n", txc);

	return FI_SUCCESS;

err_put_rx_cmdq:
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
	cxip_rdzv_pte_free(txc->rdzv_pte);
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

	ofi_spin_lock(&txc->lock);

	if (txc->enabled)
		goto unlock;

	if (!txc->send_cq) {
		CXIP_WARN("Undefined send CQ\n");
		ret = -FI_ENOCQ;
		goto unlock;
	}

	ret = cxip_txc_ibuf_create(txc);
	if (ret) {
		CXIP_WARN("Failed to create inject bufpool %d\n", ret);
		goto unlock;
	}

	ret = cxip_cq_enable(txc->send_cq, txc->ep_obj);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_cq_enable returned: %d\n", ret);
		goto destroy_ibuf;
	}

	ret = cxip_ep_cmdq(txc->ep_obj, txc->tx_id, true, txc->tclass,
			   txc->send_cq->eq.eq, &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_init(txc);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Unable to init TX CTX, ret: %d\n", ret);
			goto put_tx_cmdq;
		}
	}

	txc->pid_bits = txc->domain->iface->dev->info.pid_bits;
	txc->enabled = true;

	ofi_spin_unlock(&txc->lock);

	return FI_SUCCESS;

put_tx_cmdq:
	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, true);
destroy_ibuf:
	ofi_bufpool_destroy(txc->ibuf_pool);
unlock:
	ofi_spin_unlock(&txc->lock);

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
			CXIP_WARN("Timeout waiting for outstanding requests.\n");
			break;
		}
	}

	assert(ofi_atomic_get32(&txc->otx_reqs) == 0);

	dlist_foreach_container_safe(&txc->fc_peers, struct cxip_fc_peer,
				     fc_peer, txc_entry, tmp) {
		dlist_remove(&fc_peer->txc_entry);
		free(fc_peer);
	}

	ofi_bufpool_destroy(txc->ibuf_pool);
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

	ofi_spin_lock(&txc->lock);

	if (!txc->enabled) {
		ofi_spin_unlock(&txc->lock);
		return;
	}

	txc->enabled = false;

	ofi_spin_unlock(&txc->lock);

	txc_cleanup(txc);

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_fini(txc);
		if (ret)
			CXIP_WARN("Unable to destroy TX CTX, ret: %d\n",
				       ret);
	}

	cxip_ep_cmdq_put(txc->ep_obj, txc->tx_id, true);
}

/*
 * txc_alloc() - Allocate a TX context.
 *
 * Used to support creating a TX context for fi_endpoint() or fi_tx_context().
 */
struct cxip_txc *cxip_txc_alloc(const struct fi_tx_attr *attr, void *context)
{
	struct cxip_txc *txc;

	txc = calloc(sizeof(*txc), 1);
	if (!txc)
		return NULL;

	dlist_init(&txc->ep_list);
	ofi_spin_init(&txc->lock);
	ofi_atomic_initialize32(&txc->otx_reqs, 0);
	dlist_init(&txc->msg_queue);
	dlist_init(&txc->fc_peers);

	txc->fid.ctx.fid.fclass = FI_CLASS_TX_CTX;
	txc->fid.ctx.fid.context = context;
	txc->fclass = FI_CLASS_TX_CTX;
	txc->attr = *attr;
	txc->max_eager_size = cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;
	txc->rdzv_eager_size = cxip_env.rdzv_eager_size;
	txc->hmem = !!(attr->caps & FI_HMEM);

	/* TODO: The below should be covered by txc->lock */
	ofi_spin_init(&txc->ibuf_lock);
	return txc;
}

/*
 * cxip_txc_free() - Free a TX context allocated using cxip_txc_alloc()
 */
void cxip_txc_free(struct cxip_txc *txc)
{
	txc_disable(txc);
	ofi_spin_destroy(&txc->lock);
	ofi_spin_destroy(&txc->ibuf_lock);
	free(txc);
}

void cxip_txc_flush_msg_trig_reqs(struct cxip_txc *txc)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;

	ofi_spin_lock(&txc->lock);

	/* Drain the message queue. */
	dlist_foreach_container_safe(&txc->msg_queue, struct cxip_req, req,
				     send.txc_entry, tmp) {
		if (cxip_is_trig_req(req)) {
			ofi_atomic_dec32(&txc->otx_reqs);
			cxip_unmap(req->send.send_md);
			cxip_cq_req_free(req);
		}
	}

	ofi_spin_unlock(&txc->lock);
}
