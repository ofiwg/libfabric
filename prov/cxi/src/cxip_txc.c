/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 * Copyright (c) 2020-2023 Hewlett Packard Enterprise Development LP
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
 *
 * Caller must hold txc->ep_obj.lock
 */
void *cxip_txc_ibuf_alloc(struct cxip_txc *txc)
{
	void *ibuf;

	ibuf = (struct cxip_req *)ofi_buf_alloc(txc->ibuf_pool);
	if (ibuf)
		CXIP_DBG("Allocated inject buffer: %p\n", ibuf);
	else
		CXIP_WARN("Failed to allocate inject buffer\n");

	return ibuf;
}

/*
 * cxip_txc_ibuf_free() - Free an inject buffer.
 *
 * Caller must hold txc->ep_obj.lock
 */
void cxip_txc_ibuf_free(struct cxip_txc *txc, void *ibuf)
{
	ofi_buf_free(ibuf);
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

	/* Avoid creating VA holes outside the buffer pool
	 * if CXI_FORK_SAFE/CXI_FORK_SAFE_HP is enabled.
	 */
	if (cxip_env.fork_safe_requested)
		bp_attrs.flags = OFI_BUFPOOL_NONSHARED;

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
 *
 * Caller must hold txc->ep_obj.lock
 */
int cxip_tx_id_alloc(struct cxip_txc *txc, void *ctx)
{
	int id;

	id = ofi_idx_insert(&txc->tx_ids, ctx);
	if (id < 0 || id >= CXIP_TX_IDS) {
		CXIP_DBG("Failed to allocate TX ID: %d\n", id);
		if (id > 0)
			ofi_idx_remove(&txc->tx_ids, id);

		return -FI_ENOSPC;
	}

	CXIP_DBG("Allocated ID: %d\n", id);

	return id;
}

/*
 * cxip_tx_id_free() - Free a TX ID.
 *
 * Caller must hold txc->ep_obj.lock
 */
int cxip_tx_id_free(struct cxip_txc *txc, int id)
{
	if (id < 0 || id >= CXIP_TX_IDS)
		return -FI_EINVAL;

	ofi_idx_remove(&txc->tx_ids, id);
	CXIP_DBG("Freed ID: %d\n", id);

	return FI_SUCCESS;
}

/* Caller must hold txc->ep_obj.lock */
void *cxip_tx_id_lookup(struct cxip_txc *txc, int id)
{
	return ofi_idx_lookup(&txc->tx_ids, id);
}

/*
 * cxip_rdzv_id_alloc() - Allocate a rendezvous ID.
 *
 * A Rendezvous ID are assigned to rendezvous Send operation. The ID is used by
 * the target to differentiate rendezvous Send operations initiated by a source.
 *
 * Caller must hold txc->ep_obj->lock.
 */
int cxip_rdzv_id_alloc(struct cxip_txc *txc, struct cxip_req *req)
{
	struct indexer *rdzv_ids;
	int max_rdzv_id;
	int id_offset;
	int id;

	/* FI_TAGGED sends by definition do not support FI_MULTI_RECV;
	 * they can utilize the pool of rendezvous ID [256 to 32K-1].
	 * FI_MSG which supports FI_MULTI_RECV is restricted to a rendezvous
	 * ID range of [0 to 255].
	 */
	if (req->send.tagged) {
		rdzv_ids = &txc->rdzv_ids;
		max_rdzv_id = CXIP_RDZV_IDS;
		id_offset = CXIP_RDZV_IDS_MULTI_RECV;
	} else {
		rdzv_ids = &txc->msg_rdzv_ids;
		max_rdzv_id = CXIP_RDZV_IDS_MULTI_RECV;
		id_offset = 0;
	}

	id = ofi_idx_insert(rdzv_ids, req);
	if (id < 0 || id + id_offset >= max_rdzv_id) {
		CXIP_DBG("Failed to allocate rdzv ID: %d\n", id);
		if (id > 0)
			ofi_idx_remove(rdzv_ids, id);

		return -FI_ENOSPC;
	}

	id += id_offset;
	CXIP_DBG("Allocated ID: %d\n", id);

	return id;
}

/*
 * cxip_rdzv_id_free() - Free a rendezvous ID.
 *
 * Caller must hold txc->ep_obj->lock.
 */
int cxip_rdzv_id_free(struct cxip_txc *txc, int id)
{
	if (id < 0 || id >= CXIP_RDZV_IDS)
		return -FI_EINVAL;

	CXIP_DBG("Freed RDZV ID: %d\n", id);

	/* ID value indicates which pool it comes from */
	if (id >= CXIP_RDZV_IDS_MULTI_RECV) {
		id -= CXIP_RDZV_IDS_MULTI_RECV;
		ofi_idx_remove(&txc->rdzv_ids, id);
	} else {
		ofi_idx_remove(&txc->msg_rdzv_ids, id);
	}

	return FI_SUCCESS;
}

/* Caller must hold txc->ep_obj->lock. */
void *cxip_rdzv_id_lookup(struct cxip_txc *txc, int id)
{

	if (id >= CXIP_RDZV_IDS_MULTI_RECV) {
		id -= CXIP_RDZV_IDS_MULTI_RECV;
		return ofi_idx_lookup(&txc->rdzv_ids, id);
	}
	return ofi_idx_lookup(&txc->msg_rdzv_ids, id);
}

/*
 * txc_msg_init() - Initialize an RX context for messaging.
 *
 * Allocates and initializes hardware resources used for transmitting messages.
 *
 * Caller must hold ep_obj->lock
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
 * Caller must hold txc->ep_obj->lock.
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

	/* Protected with ep_obj->lock */
	memset(&txc->rdzv_ids, 0, sizeof(txc->rdzv_ids));
	memset(&txc->msg_rdzv_ids, 0, sizeof(txc->msg_rdzv_ids));
	memset(&txc->tx_ids, 0, sizeof(txc->tx_ids));

	/* The send EQ size is based on the contexts TX attribute size,
	 * and the number of TX operations that can be initiated
	 * by software RX processing.
	 */
	min_eq_size = (txc->attr.size + txc->send_cq->ack_batch_size +
		       + cxip_env.sw_rx_tx_init_max +
		       CXIP_INTERNAL_TX_REQS + 1) * C_EE_CFG_ECB_SIZE;
	ret = cxip_evtq_init(txc->send_cq, &txc->tx_evtq, min_eq_size,
			     0, txc->attr.size + CXIP_INTERNAL_TX_REQS);
	if (ret) {
		CXIP_WARN("Failed to initialize TX event queue: %d, %s\n",
			  ret, fi_strerror(-ret));
		goto destroy_ibuf;
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
	ofi_idx_reset(&txc->rdzv_ids);
	ofi_idx_reset(&txc->msg_rdzv_ids);
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

		cxip_evtq_progress(&txc->tx_evtq);
		cxip_ep_ctrl_progress_locked(txc->ep_obj);

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

void cxip_txc_struct_init(struct cxip_txc *txc, const struct fi_tx_attr *attr,
			  void *context)
{
	dlist_init(&txc->ep_list);
	ofi_atomic_initialize32(&txc->otx_reqs, 0);
	dlist_init(&txc->msg_queue);
	dlist_init(&txc->fc_peers);

	txc->context = context;
	txc->attr = *attr;
	txc->max_eager_size = cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;
	txc->rdzv_eager_size = cxip_env.rdzv_eager_size;
	txc->hmem = !!(attr->caps & FI_HMEM);
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

	if (!txc->enabled)
		return;

	txc->enabled = false;
	txc_cleanup(txc);

	ofi_idx_reset(&txc->tx_ids);
	ofi_idx_reset(&txc->rdzv_ids);
	ofi_idx_reset(&txc->msg_rdzv_ids);
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

/* Caller must hold ep_obj->lock. */
void cxip_txc_flush_msg_trig_reqs(struct cxip_txc *txc)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;

	/* Drain the message queue. */
	dlist_foreach_container_safe(&txc->msg_queue, struct cxip_req, req,
				     send.txc_entry, tmp) {
		if (cxip_is_trig_req(req)) {
			ofi_atomic_dec32(&txc->otx_reqs);
			cxip_unmap(req->send.send_md);
			cxip_evtq_req_free(req);
		}
	}
}
