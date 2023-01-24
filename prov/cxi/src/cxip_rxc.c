/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation. All rights reserved.
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 * Copyright (c) 2020-2023 Hewlett Packard Enterprise Development LP
 */

/* CXI RX Context Management */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

#define CXIP_SC_STATS "FC/SC stats - EQ full: %d append fail: %d no match: %d"\
		      " request full: %d unexpected: %d, NIC HW2SW unexp: %d"\
		      " NIC HW2SW append fail: %d\n"

/*
 * cxip_rxc_msg_enable() - Enable RXC messaging.
 *
 * Change the RXC RX PtlTE to enabled state. Once in enabled state, messages
 * will be accepted by hardware. Prepare all messaging resources before
 * enabling the RX PtlTE.
 *
 * Caller must hold ep_obj->lock.
 */
int cxip_rxc_msg_enable(struct cxip_rxc *rxc, uint32_t drop_count)
{
	int ret;

	/* If transitioning from disabled to the software managed state a
	 * synchronous call is used which handles drop count mismatches.
	 */
	if (rxc->new_state == RXC_ENABLED_SOFTWARE) {
		ret = cxil_pte_transition_sm(rxc->rx_pte->pte, drop_count);
		if (ret)
			RXC_WARN(rxc,
				 "Error transitioning to SW EP %d %s\n",
				  ret, fi_strerror(-ret));
		return ret;
	}

	return cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq,
				  C_PTLTE_ENABLED, drop_count);
}

/*
 * rxc_msg_disable() - Disable RXC messaging.
 *
 * Change the RXC RX PtlTE to disabled state. Once in disabled state, the PtlTE
 * will receive no additional events.
 *
 * Caller must hold rxc->ep_obj->lock.
 */
static int rxc_msg_disable(struct cxip_rxc *rxc)
{
	int ret;

	if (rxc->state != RXC_ENABLED &&
	    rxc->state != RXC_ENABLED_SOFTWARE)
		RXC_FATAL(rxc, "RXC in bad state to be disabled: state=%d\n",
			  rxc->state);

	rxc->state = RXC_DISABLED;

	ret = cxip_pte_set_state_wait(rxc->rx_pte, rxc->rx_cmdq, &rxc->rx_evtq,
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
 * Caller must hold ep_obj->lock.
 */
static int rxc_msg_init(struct cxip_rxc *rxc)
{
	int ret;
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.en_flowctrl = 1,
		.lossless = cxip_env.msg_lossless,
	};
	struct cxi_cq_alloc_opts cq_opts = {};

	ret = cxip_ep_cmdq(rxc->ep_obj, false, FI_TC_UNSPEC,
			   rxc->rx_evtq.eq, &rxc->rx_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to allocate RX CMDQ, ret: %d\n", ret);
		return -FI_EDOMAIN;
	}

	/* For FI_TC_UNSPEC, reuse the TX context command queue if possible. If
	 * a specific traffic class is requested, allocate a new command queue.
	 * This is done to prevent performance issues with reusing the TX
	 * context command queue and changing the communication profile.
	 */
	if (cxip_env.rget_tc == FI_TC_UNSPEC) {
		ret = cxip_ep_cmdq(rxc->ep_obj, true, FI_TC_UNSPEC,
				   rxc->rx_evtq.eq, &rxc->tx_cmdq);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Unable to allocate TX CMDQ, ret: %d\n", ret);
			ret = -FI_EDOMAIN;
			goto put_rx_cmdq;
		}
	} else {
		cq_opts.count = rxc->ep_obj->txq_size * 4;
		cq_opts.flags = CXI_CQ_IS_TX;
		cq_opts.policy = cxip_env.cq_policy;

		ret = cxip_cmdq_alloc(rxc->ep_obj->domain->lni,
				      rxc->rx_evtq.eq, &cq_opts,
				      rxc->ep_obj->auth_key.vni,
				      cxip_ofi_to_cxi_tc(cxip_env.rget_tc),
				      CXI_TC_TYPE_DEFAULT, &rxc->tx_cmdq);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("Unable to allocate CMDQ, ret: %d\n", ret);
			ret = -FI_ENOSPC;
			goto put_rx_cmdq;
		}
	}

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->ep_obj->av->attr.flags & FI_SYMMETRIC) {
		CXIP_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->ep_obj->if_dom,
			     rxc->rx_evtq.eq, CXIP_PTL_IDX_RXQ, false,
			     &pt_opts, cxip_recv_pte_cb, rxc, &rxc->rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate RX PTE: %d\n", ret);
		goto put_tx_cmdq;
	}

	/* One slot must be reserved to support hardware generated state change
	 * events.
	 */
	ret = cxip_evtq_adjust_reserved_fc_event_slots(&rxc->rx_evtq,
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
	if (cxip_env.rget_tc == FI_TC_UNSPEC)
		cxip_ep_cmdq_put(rxc->ep_obj, true);
	else
		cxip_cmdq_free(rxc->tx_cmdq);
put_rx_cmdq:
	cxip_ep_cmdq_put(rxc->ep_obj, false);

	return ret;
}

/*
 * rxc_msg_fini() - Finalize RX context messaging.
 *
 * Free hardware resources allocated when the RX context was initialized for
 * messaging.
 *
 * Caller must hold ep_obj->lock.
 */
static int rxc_msg_fini(struct cxip_rxc *rxc)
{
	int ret __attribute__((unused));

	cxip_pte_free(rxc->rx_pte);

	cxip_ep_cmdq_put(rxc->ep_obj, false);

	if (cxip_env.rget_tc == FI_TC_UNSPEC)
		cxip_ep_cmdq_put(rxc->ep_obj, true);
	else
		cxip_cmdq_free(rxc->tx_cmdq);

	cxip_evtq_adjust_reserved_fc_event_slots(&rxc->rx_evtq,
						 -1 * RXC_RESERVED_FC_SLOTS);

	cxip_evtq_fini(&rxc->rx_evtq);

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
		if (ux_send->req->type == CXIP_REQ_RBUF)
			cxip_req_buf_ux_free(ux_send);
		else
			free(ux_send);

		rxc->sw_ux_list_len--;
	}

	if (rxc->sw_ux_list_len != 0)
		CXIP_WARN("sw_ux_list_len %d != 0\n", rxc->sw_ux_list_len);
	assert(rxc->sw_ux_list_len == 0);

	/* Free any pending UX entries waiting from the request list */
	dlist_foreach_container_safe(&rxc->sw_pending_ux_list,
				     struct cxip_ux_send, ux_send,
				     rxc_entry, tmp) {
		dlist_remove(&ux_send->rxc_entry);
		if (ux_send->req->type == CXIP_REQ_RBUF)
			cxip_req_buf_ux_free(ux_send);
		else
			free(ux_send);

		rxc->sw_pending_ux_list_len--;
	}

	if (rxc->sw_pending_ux_list_len != 0)
		CXIP_WARN("sw_pending_ux_list_len %d != 0\n",
			  rxc->sw_pending_ux_list_len);
	assert(rxc->sw_pending_ux_list_len == 0);
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
	size_t min_eq_size;
	enum c_ptlte_state state;

	if (rxc->state != RXC_DISABLED)
		return FI_SUCCESS;

	if (!ofi_recv_allowed(rxc->attr.caps)) {
		rxc->state = RXC_ENABLED;
		return FI_SUCCESS;
	}

	if (!rxc->recv_cq) {
		CXIP_WARN("Undefined recv CQ\n");
		return -FI_ENOCQ;
	}

	min_eq_size = (rxc->recv_cq->attr.size + rxc->recv_cq->ack_batch_size) *
			C_EE_CFG_ECB_SIZE;
	ret = cxip_evtq_init(rxc->recv_cq, &rxc->rx_evtq, min_eq_size, 0, 0);
	if (ret) {
		CXIP_WARN("Failed to initialize RXC event queue: %d, %s\n",
			  ret, fi_strerror(-ret));
		return ret;
	}

	ret = rxc_msg_init(rxc);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("rxc_msg_init returned: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto evtq_fini;
	}

	/* If starting in or able to transition to software managed
	 * PtlTE, append request list entries first.
	 */
	if (cxip_software_pte_allowed()) {
		ret = cxip_req_bufpool_init(rxc);
		if (ret != FI_SUCCESS)
			goto err_msg_fini;
	}

	if (rxc->msg_offload) {
		state = C_PTLTE_ENABLED;
		ret = cxip_oflow_bufpool_init(rxc);
		if (ret != FI_SUCCESS)
			goto err_req_buf_fini;
	} else {
		state = C_PTLTE_SOFTWARE_MANAGED;
	}

	/* Start accepting Puts. */
	ret = cxip_pte_set_state(rxc->rx_pte, rxc->rx_cmdq, state, 0);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_pte_set_state returned: %d\n", ret);
		goto err_oflow_buf_fini;
	}

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_evtq_progress(&rxc->rx_evtq);
	} while (rxc->rx_pte->state != state);

	rxc->pid_bits = rxc->domain->iface->dev->info.pid_bits;
	CXIP_DBG("RXC messaging enabled: %p, pid_bits: %d\n",
		 rxc, rxc->pid_bits);

	return FI_SUCCESS;

err_oflow_buf_fini:
	if (rxc->msg_offload)
		cxip_oflow_bufpool_fini(rxc);

err_req_buf_fini:
	if (cxip_software_pte_allowed())
		cxip_req_bufpool_fini(rxc);

err_msg_fini:
	tmp = rxc_msg_fini(rxc);
	if (tmp != FI_SUCCESS)
		CXIP_WARN("rxc_msg_fini returned: %d\n", tmp);

evtq_fini:
	cxip_evtq_fini(&rxc->rx_evtq);

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

	cxip_evtq_req_discard(&rxc->rx_evtq, rxc);

	do {
		ret = cxip_evtq_req_cancel(&rxc->rx_evtq, rxc, 0, false);
		if (ret == FI_SUCCESS)
			canceled++;
	} while (ret == FI_SUCCESS);

	if (canceled)
		CXIP_DBG("Canceled %d Receives: %p\n", canceled, rxc);

	start = ofi_gettime_ms();
	while (ofi_atomic_get32(&rxc->orx_reqs)) {
		sched_yield();
		cxip_evtq_progress(&rxc->rx_evtq);

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

	if (rxc->num_fc_eq_full || rxc->num_fc_no_match ||
	    rxc->num_fc_req_full || rxc->num_fc_unexp ||
	    rxc->num_fc_append_fail || rxc->num_sc_nic_hw2sw_unexp ||
	    rxc->num_sc_nic_hw2sw_append_fail)
		CXIP_INFO(CXIP_SC_STATS, rxc->num_fc_eq_full,
			  rxc->num_fc_append_fail, rxc->num_fc_no_match,
			  rxc->num_fc_req_full, rxc->num_fc_unexp,
			  rxc->num_sc_nic_hw2sw_unexp,
			  rxc->num_sc_nic_hw2sw_append_fail);
}

static void cxip_rxc_dump_counters(struct cxip_rxc *rxc)
{
	int i;
	int j;
	int k;
	size_t msg_size;
	bool print_header;
	int count;

	for (i = 0; i < CXIP_LIST_COUNTS; i++) {
		for (j = 0; j < OFI_HMEM_MAX; j++) {

			print_header = true;

			for (k = 0; k < CXIP_COUNTER_BUCKETS; k++) {
				if (k == 0)
					msg_size = 0;
				else
					msg_size = (1ULL << (k - 1));

				count = ofi_atomic_get32(&rxc->cntrs.msg_count[i][j][k]);
				if (count) {
					if (print_header) {
						RXC_INFO(rxc, "Recv Message Size %s - %s Histogram\n",
							 c_ptl_list_strs[i],
							 fi_tostr(&j, FI_TYPE_HMEM_IFACE));
						RXC_INFO(rxc, "%-14s Count\n", "Size");
						print_header = false;
					}

					RXC_INFO(rxc, "%-14lu %u\n", msg_size,
						 count);
				}
			}
		}

	}
}

void cxip_rxc_struct_init(struct cxip_rxc *rxc, const struct fi_rx_attr *attr,
			  void *context)
{
	int i;

	dlist_init(&rxc->ep_list);
	ofi_atomic_initialize32(&rxc->orx_reqs, 0);

	rxc->context = context;
	rxc->attr = *attr;

	for (i = 0; i < CXIP_DEF_EVENT_HT_BUCKETS; i++)
		dlist_init(&rxc->deferred_events.bh[i]);

	dlist_init(&rxc->fc_drops);
	dlist_init(&rxc->replay_queue);
	dlist_init(&rxc->sw_ux_list);
	dlist_init(&rxc->sw_recv_queue);
	dlist_init(&rxc->sw_pending_ux_list);

	rxc->max_eager_size = cxip_env.rdzv_threshold + cxip_env.rdzv_get_min;
	rxc->drop_count = -1;

	/* TODO make configurable */
	rxc->min_multi_recv = CXIP_EP_MIN_MULTI_RECV;
	rxc->state = RXC_DISABLED;
	rxc->msg_offload = cxip_env.msg_offload;
	rxc->hmem = !!(attr->caps & FI_HMEM);
	rxc->sw_ep_only = cxip_env.rx_match_mode == CXIP_PTLTE_SOFTWARE_MODE;
	rxc->rget_align_mask = cxip_env.rdzv_aligned_sw_rget ?
					cxip_env.cacheline_size - 1 : 0;

	cxip_msg_counters_init(&rxc->cntrs);
}

/*
 * cxip_rxc_disable() - Disable the RX context of an base endpoint object.
 *
 * Free hardware resources allocated when the context was enabled. Called via
 * fi_close().
 */
void cxip_rxc_disable(struct cxip_rxc *rxc)
{
	int ret;

	cxip_rxc_dump_counters(rxc);

	if (rxc->state == RXC_DISABLED)
		return;

	if (ofi_recv_allowed(rxc->attr.caps)) {
		/* Stop accepting Puts. */
		ret = rxc_msg_disable(rxc);
		if (ret != FI_SUCCESS)
			CXIP_WARN("rxc_msg_disable returned: %d\n", ret);

		cxip_rxc_free_ux_entries(rxc);

		rxc_cleanup(rxc);

		if (cxip_software_pte_allowed())
			cxip_req_bufpool_fini(rxc);

		if (cxip_env.msg_offload)
			cxip_oflow_bufpool_fini(rxc);

		/* Free hardware resources. */
		ret = rxc_msg_fini(rxc);
		if (ret != FI_SUCCESS)
			CXIP_WARN("rxc_msg_fini returned: %d\n", ret);
	}
}
