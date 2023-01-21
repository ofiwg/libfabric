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

/* 8 Rendezvous, 2 RMA and 2 Atomic + 4 extra */
#define CXIP_INTERNAL_TX_REQS	16

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
 * cxip_tx_id_alloc() - Allocate a TX ID.
 *
 * TX IDs are assigned to Put operations that need to be tracked by the target.
 * One example of this is a Send with completion that guarantees match
 * completion at the target. This only applies to eager, unexpected Sends.
 */
int cxip_tx_id_alloc(struct cxip_txc *txc, void *ctx)
{
	int id;

	ofi_spin_lock(&txc->tx_id_lock);

	id = ofi_idx_insert(&txc->tx_ids, ctx);

	if (id < 0 || id >= CXIP_TX_IDS) {
		CXIP_DBG("Failed to allocate TX ID: %d\n", id);
		if (id > 0)
			ofi_idx_remove(&txc->tx_ids, id);
		ofi_spin_unlock(&txc->tx_id_lock);
		return -FI_ENOSPC;
	}

	ofi_spin_unlock(&txc->tx_id_lock);

	CXIP_DBG("Allocated ID: %d\n", id);

	return id;
}

/*
 * cxip_tx_id_free() - Free a TX ID.
 */
int cxip_tx_id_free(struct cxip_txc *txc, int id)
{
	if (id < 0 || id >= CXIP_TX_IDS)
		return -FI_EINVAL;

	ofi_spin_lock(&txc->tx_id_lock);
	ofi_idx_remove(&txc->tx_ids, id);
	ofi_spin_unlock(&txc->tx_id_lock);

	CXIP_DBG("Freed ID: %d\n", id);

	return FI_SUCCESS;
}

void *cxip_tx_id_lookup(struct cxip_txc *txc, int id)
{
	void *entry;

	ofi_spin_lock(&txc->tx_id_lock);
	entry = ofi_idx_lookup(&txc->tx_ids, id);
	ofi_spin_unlock(&txc->tx_id_lock);

	return entry;
}

/*
 * cxip_rdzv_id_alloc() - Allocate a rendezvous ID.
 *
 * A Rendezvous ID are assigned to rendezvous Send operation. The ID is used by
 * the target to differentiate rendezvous Send operations initiated by a source.
 */
int cxip_rdzv_id_alloc(struct cxip_txc *txc, void *ctx)
{
	int id;

	ofi_spin_lock(&txc->rdzv_id_lock);

	id = ofi_idx_insert(&txc->rdzv_ids, ctx);

	if (id < 0 || id >= CXIP_RDZV_IDS) {
		CXIP_DBG("Failed to allocate rdzv ID: %d\n", id);
		if (id > 0)
			ofi_idx_remove(&txc->rdzv_ids, id);
		ofi_spin_unlock(&txc->rdzv_id_lock);
		return -FI_ENOSPC;
	}

	ofi_spin_unlock(&txc->rdzv_id_lock);

	CXIP_DBG("Allocated ID: %d\n", id);

	return id;
}

/*
 * cxip_rdzv_id_free() - Free a rendezvous ID.
 */
int cxip_rdzv_id_free(struct cxip_txc *txc, int id)
{
	if (id < 0 || id >= CXIP_RDZV_IDS)
		return -FI_EINVAL;

	ofi_spin_lock(&txc->rdzv_id_lock);
	ofi_idx_remove(&txc->rdzv_ids, id);
	ofi_spin_unlock(&txc->rdzv_id_lock);

	CXIP_DBG("Freed ID: %d\n", id);

	return FI_SUCCESS;
}

void *cxip_rdzv_id_lookup(struct cxip_txc *txc, int id)
{
	void *entry;

	ofi_spin_lock(&txc->rdzv_id_lock);
	entry = ofi_idx_lookup(&txc->rdzv_ids, id);
	ofi_spin_unlock(&txc->rdzv_id_lock);

	return entry;
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
	ret = cxip_ep_cmdq(txc->ep_obj, false, FI_TC_UNSPEC,
			   txc->tx_evtq.eq, &txc->rx_cmdq);
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
	cxip_ep_cmdq_put(txc->ep_obj, false);

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
	cxip_ep_cmdq_put(txc->ep_obj, false);

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
	size_t min_eq_size;

	if (txc->enabled)
		return FI_SUCCESS;

	if (!txc->send_cq) {
		CXIP_WARN("Undefined send CQ\n");
		return -FI_ENOCQ;
	}

	ret = cxip_txc_ibuf_create(txc);
	if (ret) {
		CXIP_WARN("Failed to create inject bufpool %d\n", ret);
		return ret;
	}

	memset(&txc->rdzv_ids, 0, sizeof(txc->rdzv_ids));
	ofi_spin_init(&txc->rdzv_id_lock);

	memset(&txc->tx_ids, 0, sizeof(txc->tx_ids));
	ofi_spin_init(&txc->tx_id_lock);

	/* The send EQ size is based on the contexts TX attribute size */
	min_eq_size = (txc->attr.size + txc->send_cq->ack_batch_size +
		       CXIP_INTERNAL_TX_REQS + 1) * C_EE_CFG_ECB_SIZE;
	ret = cxip_evtq_init(txc->send_cq, &txc->tx_evtq, min_eq_size,
			     0, txc->attr.size + CXIP_INTERNAL_TX_REQS);
	if (ret) {
		CXIP_WARN("Failed to initialize TX event queue: %d, %s\n",
			  ret, fi_strerror(-ret));
		goto destroy_ibuf;
	}

	ret = cxip_cq_enable(txc->send_cq, txc->ep_obj);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_cq_enable returned: %d\n", ret);
		goto destroy_evtq;
	}

	ret = cxip_ep_cmdq(txc->ep_obj, true, txc->tclass,
			   txc->tx_evtq.eq, &txc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		/* CQ disable will be done at CQ close */
		goto destroy_evtq;
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

	return FI_SUCCESS;

put_tx_cmdq:
	cxip_ep_cmdq_put(txc->ep_obj, true);
destroy_evtq:
	cxip_evtq_fini(&txc->tx_evtq);
destroy_ibuf:
	ofi_idx_reset(&txc->tx_ids);
	ofi_spin_destroy(&txc->tx_id_lock);
	ofi_idx_reset(&txc->rdzv_ids);
	ofi_spin_destroy(&txc->rdzv_id_lock);
	ofi_bufpool_destroy(txc->ibuf_pool);

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
		goto free_fc_peers;

	cxip_evtq_req_discard(&txc->tx_evtq, txc);

	start = ofi_gettime_ms();
	while (ofi_atomic_get32(&txc->otx_reqs)) {
		sched_yield();

		cxip_ep_tx_progress(txc->ep_obj);
		cxip_ep_ctrl_progress(txc->ep_obj);

		if (ofi_gettime_ms() - start > CXIP_REQ_CLEANUP_TO) {
			CXIP_WARN("Timeout waiting for outstanding requests.\n");
			break;
		}
	}

	assert(ofi_atomic_get32(&txc->otx_reqs) == 0);

free_fc_peers:
	dlist_foreach_container_safe(&txc->fc_peers, struct cxip_fc_peer,
				     fc_peer, txc_entry, tmp) {
		dlist_remove(&fc_peer->txc_entry);
		free(fc_peer);
	}
}

void cxip_txc_struct_fini(struct cxip_txc *txc)
{
	ofi_spin_destroy(&txc->lock);
	ofi_spin_destroy(&txc->ibuf_lock);
}

void cxip_txc_struct_init(struct cxip_txc *txc, const struct fi_tx_attr *attr,
			  void *context)
{
	dlist_init(&txc->ep_list);
	ofi_spin_init(&txc->lock);
	ofi_atomic_initialize32(&txc->otx_reqs, 0);
	dlist_init(&txc->msg_queue);
	dlist_init(&txc->fc_peers);

	txc->context = context;
	txc->attr = *attr;
	txc->max_eager_size = cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;
	txc->rdzv_eager_size = cxip_env.rdzv_eager_size;
	txc->hmem = !!(attr->caps & FI_HMEM);

	/* TODO: The below should be covered by txc->lock */
	ofi_spin_init(&txc->ibuf_lock);
}

/*
 * cxip_txc_disable() - Disable a TX context for a base endpoint object.
 *
 * Free hardware resources allocated when the context was enabled. Called via
 * fi_close().
 */
void cxip_txc_disable(struct cxip_txc *txc)
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

	ofi_idx_reset(&txc->tx_ids);
	ofi_spin_destroy(&txc->tx_id_lock);
	ofi_idx_reset(&txc->rdzv_ids);
	ofi_spin_destroy(&txc->rdzv_id_lock);
	ofi_bufpool_destroy(txc->ibuf_pool);

	if (ofi_send_allowed(txc->attr.caps)) {
		ret = txc_msg_fini(txc);
		if (ret)
			CXIP_WARN("Unable to destroy TX CTX, ret: %d\n",
				       ret);
	}

	cxip_ep_cmdq_put(txc->ep_obj, true);
	cxip_evtq_fini(&txc->tx_evtq);
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
			cxip_evtq_req_free(req);
		}
	}

	ofi_spin_unlock(&txc->lock);
}
