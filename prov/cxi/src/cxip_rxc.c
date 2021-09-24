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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)

/*
 * rxc_msg_enable() - Enable RXC messaging.
 *
 * Change the RXC RX PtlTE to enabled state. Once in enabled state, messages
 * will be accepted by hardware. Prepare all messaging resources before
 * enabling the RX PtlTE.
 *
 * Caller must hold rxc->lock.
 */
int cxip_rxc_msg_enable(struct cxip_rxc *rxc, uint32_t drop_count)
{
	int ret;

	ret = cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq, C_PTLTE_ENABLED,
				 drop_count);

	return ret;
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

	if (rxc->state != RXC_ENABLED)
		RXC_FATAL(rxc, "RXC in bad state to be disabled: state=%d\n",
			  rxc->state);

	rxc->state = RXC_DISABLED;

	ret = cxip_pte_set_state_wait(rxc->rx_pte, rxc->rx_cmdq, rxc->recv_cq,
				      C_PTLTE_DISABLED, 0);
	if (ret == FI_SUCCESS)
		CXIP_DBG("RXC PtlTE disabled: %p\n", rxc);

	return ret;
}

#define RXC_RESERVED_FC_SLOTS 1

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
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.en_flowctrl = 1,
	};

	ret = cxip_ep_cmdq(rxc->ep_obj, rxc->rx_id, false, FI_TC_UNSPEC,
			   rxc->recv_cq->rx_eq.eq, &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate RX CMDQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	ret = cxip_ep_cmdq(rxc->ep_obj, rxc->rx_id, true, FI_TC_UNSPEC,
			   rxc->recv_cq->rx_eq.eq, &rxc->tx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate TX CMDQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto put_rx_cmdq;
	}

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->if_dom[rxc->rx_id],
			     rxc->recv_cq->rx_eq.eq, CXIP_PTL_IDX_RXQ, false,
			     &pt_opts, cxip_recv_pte_cb, rxc, &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate RX PTE: %d\n", ret);
		goto put_tx_cmdq;
	}

	/* One slot must be reserved to support hardware generated state change
	 * events.
	 */
	ret = cxip_cq_adjust_reserved_fc_event_slots(rxc->recv_cq,
						     RXC_RESERVED_FC_SLOTS);
	if (ret) {
		CXIP_WARN("Unable to adjust RX reserved event slots: %d\n",
			  ret);
		goto free_pte;
	}

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(rxc->rx_pte);
put_tx_cmdq:
	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, true);
put_rx_cmdq:
	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, false);

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
	int ret __attribute__((unused));

	cxip_pte_free(rxc->rx_pte);

	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, false);

	cxip_ep_cmdq_put(rxc->ep_obj, rxc->rx_id, true);

	cxip_cq_adjust_reserved_fc_event_slots(rxc->recv_cq,
					       -1 * RXC_RESERVED_FC_SLOTS);

	return FI_SUCCESS;
}

static void cxip_rxc_free_ux_entries(struct cxip_rxc *rxc)
{
	struct cxip_ux_send *ux_send;
	struct dlist_entry *tmp;

	/* TODO: Manage freeing of UX entries better. This code is redundant
	 * with the freeing in cxip_recv_sw_matcher().
	 */
	dlist_foreach_container_safe(&rxc->sw_ux_list, struct cxip_ux_send,
				     ux_send, rxc_entry, tmp) {
		dlist_remove(&ux_send->rxc_entry);
		if (ux_send->req->type == CXIP_REQ_RBUF) {
			cxip_req_buf_ux_free(ux_send);
		} else {
			free(ux_send);
			rxc->sw_ux_list_len--;
		}
	}
	assert(rxc->sw_ux_list_len == 0);
}

static void cxip_rxc_req_buf_dlist_free(struct dlist_entry *head)
{
	struct cxip_req_buf *buf;
	struct dlist_entry *tmp;

	dlist_foreach_container_safe(head, struct cxip_req_buf, buf,
				     req_buf_entry, tmp) {
		dlist_remove_init(&buf->req_buf_entry);
		cxip_req_buf_free(buf);
	}
}

static int cxip_rxc_req_buf_init(struct cxip_rxc *rxc)
{
	int i;
	struct cxip_req_buf *buf;
	struct dlist_entry tmp_req_buf_list;
	struct dlist_entry *tmp;
	int ret;

	dlist_init(&tmp_req_buf_list);

	for (i = 0; i < rxc->req_buf_count; i++) {
		buf = cxip_req_buf_alloc(rxc);
		if (!buf) {
			ret = -FI_ENOMEM;
			goto err_free_bufs;
		}

		dlist_insert_tail(&buf->req_buf_entry, &tmp_req_buf_list);
	}

	/* Since this is called during RXC initialization, RXQ CMDQ should be
	 * empty. Thus, linking a request buffer should not fail.
	 */
	dlist_foreach_container_safe(&tmp_req_buf_list, struct cxip_req_buf,
				     buf, req_buf_entry, tmp) {
		ret = cxip_req_buf_link(buf);
		if (ret != FI_SUCCESS)
			CXIP_FATAL("Failed to link request buffer: %d\n", ret);
	}

	return FI_SUCCESS;

err_free_bufs:
	cxip_rxc_req_buf_dlist_free(&tmp_req_buf_list);

	return ret;
}

static void cxip_rxc_req_buf_fini(struct cxip_rxc *rxc)
{
	struct cxip_req_buf *buf;
	int ret;

	assert(rxc->rx_pte->state == C_PTLTE_DISABLED);

	/* All request buffers are split between the active and consumed list.
	 * Only active buffers need to be unlinked.
	 */
	dlist_foreach_container(&rxc->active_req_bufs, struct cxip_req_buf, buf,
				req_buf_entry) {
		ret = cxip_req_buf_unlink(buf);
		if (ret != FI_SUCCESS)
			CXIP_FATAL("Failed to unlink request buffer: %d\n",
				   ret);
	}

	do {
		cxip_cq_progress(rxc->recv_cq);
	} while (ofi_atomic_get32(&rxc->req_bufs_linked));

	cxip_rxc_req_buf_dlist_free(&rxc->active_req_bufs);
	cxip_rxc_req_buf_dlist_free(&rxc->consumed_req_bufs);

	assert(ofi_atomic_get32(&rxc->req_bufs_allocated) == 0);
}

/*
 * cxip_rxc_enable() - Enable an RX context for use.
 *
 * Called via fi_enable(). The context could be used in a standard endpoint or
 * a scalable endpoint.
 */
int cxip_rxc_enable(struct cxip_rxc *rxc)
{
	int ret;
	int tmp;
	enum c_ptlte_state state;

	fastlock_acquire(&rxc->lock);

	if (rxc->state != RXC_DISABLED) {
		fastlock_release(&rxc->lock);
		return FI_SUCCESS;
	}

	if (!ofi_recv_allowed(rxc->attr.caps)) {
		rxc->state = RXC_ENABLED;
		fastlock_release(&rxc->lock);
		return FI_SUCCESS;
	}

	if (!rxc->recv_cq) {
		CXIP_WARN("Undefined recv CQ\n");
		fastlock_release(&rxc->lock);
		return -FI_ENOCQ;
	}

	if (rxc->recv_cntr) {
		ret = cxip_cntr_enable(rxc->recv_cntr);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("cxip_cntr_enable(FI_RECV) returned: %d\n",
				  ret);
			fastlock_release(&rxc->lock);
			return ret;
		}
	}

	ret = cxip_cq_enable(rxc->recv_cq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_cq_enable returned: %d\n", ret);
		fastlock_release(&rxc->lock);
		return ret;
	}

	fastlock_release(&rxc->lock);

	ret = rxc_msg_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("rxc_msg_init returned: %d\n", ret);
		return -FI_EDOMAIN;
	}

	if (!rxc->msg_offload) {
		state = C_PTLTE_SOFTWARE_MANAGED;
		ret = cxip_rxc_req_buf_init(rxc);
	} else {
		state = C_PTLTE_ENABLED;
		ret = cxip_rxc_oflow_init(rxc);
	}
	if (ret != FI_SUCCESS)
		goto err_msg_fini;

	/* Start accepting Puts. */
	ret = cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq, state, 0);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_pte_set_state returned: %d\n", ret);
		goto err_oflow_req_buf_fini;
	}

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);
	} while (rxc->rx_pte->state != state);

	CXIP_DBG("RXC messaging enabled: %p\n", rxc);

	rxc->pid_bits = rxc->domain->iface->dev->info.pid_bits;

	return FI_SUCCESS;

err_oflow_req_buf_fini:
	if (!rxc->msg_offload)
		cxip_rxc_req_buf_fini(rxc);
	else
		cxip_rxc_oflow_fini(rxc);
err_msg_fini:
	tmp = rxc_msg_fini(rxc);
	if (tmp != FI_SUCCESS)
		CXIP_WARN("rxc_msg_fini returned: %d\n", tmp);

	return ret;
}

/*
 * rxc_cleanup() - Attempt to free outstanding requests.
 *
 * Outstanding commands may be dropped when the RX Command Queue is freed.
 * This leads to missing events. Attempt to gather all events before freeing
 * the RX CQ. If events go missing, resources will be leaked until the
 * Completion Queue is freed.
 */
static void rxc_cleanup(struct cxip_rxc *rxc)
{
	int ret;
	uint64_t start;
	int canceled = 0;
	struct cxip_fc_drops *fc_drops;
	struct dlist_entry *tmp;

	if (!ofi_atomic_get32(&rxc->orx_reqs))
		return;

	cxip_cq_req_discard(rxc->recv_cq, rxc);

	do {
		ret = cxip_cq_req_cancel(rxc->recv_cq, rxc, 0, false);
		if (ret == FI_SUCCESS)
			canceled++;
	} while (ret == FI_SUCCESS);

	if (canceled)
		CXIP_DBG("Canceled %d Receives: %p\n", canceled, rxc);

	start = ofi_gettime_ms();
	while (ofi_atomic_get32(&rxc->orx_reqs)) {
		sched_yield();
		cxip_cq_progress(rxc->recv_cq);

		if (ofi_gettime_ms() - start > CXIP_REQ_CLEANUP_TO) {
			CXIP_WARN("Timeout waiting for outstanding requests.\n");
			break;
		}
	}

	dlist_foreach_container_safe(&rxc->fc_drops, struct cxip_fc_drops,
				     fc_drops, rxc_entry, tmp) {
		dlist_remove(&fc_drops->rxc_entry);
		free(fc_drops);
	}
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

	if (rxc->state == RXC_DISABLED) {
		fastlock_release(&rxc->lock);
		return;
	}

	fastlock_release(&rxc->lock);

	if (ofi_recv_allowed(rxc->attr.caps)) {
		/* Stop accepting Puts. */
		ret = rxc_msg_disable(rxc);
		if (ret != FI_SUCCESS)
			CXIP_WARN("rxc_msg_disable returned: %d\n", ret);

		cxip_rxc_free_ux_entries(rxc);

		rxc_cleanup(rxc);

		if (!rxc->msg_offload)
			cxip_rxc_req_buf_fini(rxc);
		else
			cxip_rxc_oflow_fini(rxc);

		/* Free hardware resources. */
		ret = rxc_msg_fini(rxc);
		if (ret != FI_SUCCESS)
			CXIP_WARN("rxc_msg_fini returned: %d\n", ret);
	}
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
	ofi_atomic_initialize32(&rxc->orx_reqs, 0);

	rxc->ctx.fid.fclass = FI_CLASS_RX_CTX;
	rxc->ctx.fid.context = context;
	rxc->attr = *attr;
	rxc->use_shared = use_shared;

	fastlock_init(&rxc->rx_lock);
	ofi_atomic_initialize32(&rxc->oflow_bufs_submitted, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_linked, 0);
	ofi_atomic_initialize32(&rxc->oflow_bufs_in_use, 0);
	dlist_init(&rxc->oflow_bufs);
	dlist_init(&rxc->deferred_events);
	ofi_atomic_initialize32(&rxc->sink_le_linked, 0);
	dlist_init(&rxc->fc_drops);
	dlist_init(&rxc->replay_queue);
	dlist_init(&rxc->sw_ux_list);
	dlist_init(&rxc->sw_recv_queue);

	dlist_init(&rxc->active_req_bufs);
	dlist_init(&rxc->consumed_req_bufs);
	ofi_atomic_initialize32(&rxc->req_bufs_linked, 0);
	ofi_atomic_initialize32(&rxc->req_bufs_allocated, 0);

	rxc->rdzv_threshold = cxip_env.rdzv_threshold;
	rxc->rdzv_get_min = cxip_env.rdzv_get_min;
	rxc->oflow_buf_size = cxip_env.oflow_buf_size;
	rxc->oflow_bufs_max = cxip_env.oflow_buf_count;
	rxc->req_buf_size = cxip_env.req_buf_size;
	rxc->req_buf_count = cxip_env.req_buf_count;
	rxc->drop_count = -1;

	/* TODO make configurable */
	rxc->min_multi_recv = CXIP_EP_MIN_MULTI_RECV;
	rxc->state = RXC_DISABLED;
	rxc->msg_offload = cxip_env.msg_offload;
	rxc->hmem = !!(attr->caps & FI_HMEM);

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
