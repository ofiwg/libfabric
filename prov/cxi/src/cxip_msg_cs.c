/*
 * SPDX-License-Identifier: BSD-2 Clause or GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_EP_CTRL, __VA_ARGS__)

#define APPEND_LE_FATAL "Recieve LE resources exhuasted. Requires use " \
	" of FI_PROTO_CXI endpoint protocol\n"

static int cxip_cs_send_cb(struct cxip_req *req, const union c_event *event);

static void cxip_rnr_recv_pte_cb(struct cxip_pte *pte,
				 const union c_event *event)
{
	struct cxip_rxc *rxc = (struct cxip_rxc *)pte->ctx;
	uint32_t state;

	assert(rxc->protocol == FI_PROTO_CXI_CS);

	switch (event->hdr.event_type) {
	case C_EVENT_STATE_CHANGE:
		if (cxi_event_rc(event) != C_RC_OK ||
		    event->tgt_long.ptlte_index != rxc->rx_pte->pte->ptn)
			CXIP_FATAL("Failed receive PtlTE state change, %s\n",
				   cxi_rc_to_str(cxi_event_rc(event)));

		state = event->tgt_long.initiator.state_change.ptlte_state;

		switch (state) {
		case C_PTLTE_ENABLED:
			assert(rxc->state == RXC_DISABLED);
			rxc->state = RXC_ENABLED;
			RXC_DBG(rxc, "Receive PtlTE enabled\n");
			break;
		case C_PTLTE_DISABLED:
			/* Set to disabled before issuing command */
			assert(rxc->state == RXC_DISABLED);
			rxc->state = RXC_DISABLED;
			RXC_DBG(rxc, "Receive PtlTE disabled\n");
			break;
		default:
			CXIP_FATAL("Unexpected receive PtlTE state %d\n",
				   state);
		}
		break;

	case C_EVENT_COMMAND_FAILURE:
		CXIP_FATAL("Command failure: cq=%u target=%u fail_loc=%u "
			   "cmd_type=%u cmd_size=%u opcode=%u\n",
			   event->cmd_fail.cq_id, event->cmd_fail.is_target,
			   event->cmd_fail.fail_loc,
			   event->cmd_fail.fail_command.cmd_type,
			   event->cmd_fail.fail_command.cmd_size,
			   event->cmd_fail.fail_command.opcode);
		break;
	default:
		CXIP_FATAL("Invalid event type: %s\n", cxi_event_to_str(event));
	}
}

/*
 * cxip_cs_recv_req() - Submit Receive request to hardware.
 */
static ssize_t cxip_cs_recv_req(struct cxip_req *req, bool restart_seq)
{
	struct cxip_rxc *rxc = req->recv.rxc;
	uint32_t le_flags = 0;
	union cxip_match_bits mb = {};
	union cxip_match_bits ib = {
		.cs_cq_data = 1,
	};
	int ret;
	struct cxip_md *recv_md = req->recv.recv_md;
	uint64_t recv_iova = 0;

	if (req->recv.tagged) {
		mb.tagged = 1;
		mb.cs_tag = req->recv.tag;
		ib.cs_tag = req->recv.ignore;
	}

	if (req->recv.match_id == CXI_MATCH_ID_ANY)
		ib.cs_vni = ~0;
	else
		mb.cs_vni = req->recv.vni;

	/* Always set manage_local in Receive LEs. This makes Cassini ignore
	 * initiator remote_offset in all Puts. With this, remote_offset in Put
	 * events can be used by the initiator for protocol data. The behavior
	 * of use_once is not impacted by manage_local.
	 */
	le_flags |= C_LE_EVENT_LINK_DISABLE | C_LE_EVENT_UNLINK_DISABLE |
		    C_LE_MANAGE_LOCAL | C_LE_UNRESTRICTED_BODY_RO |
		    C_LE_UNRESTRICTED_END_RO | C_LE_OP_PUT;

	if (!req->recv.multi_recv)
		le_flags |= C_LE_USE_ONCE;
	if (restart_seq)
		le_flags |= C_LE_RESTART_SEQ;

	if (recv_md)
		recv_iova = CXI_VA_TO_IOVA(recv_md->md,
					   (uint64_t)req->recv.recv_buf +
					   req->recv.start_offset);

	req->recv.hw_offloaded = true;

	/* Issue Append command */
	ret = cxip_pte_append(rxc->rx_pte, recv_iova,
			      req->recv.ulen - req->recv.start_offset,
			      recv_md ? recv_md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, req->req_id,
			      mb.raw, ib.raw, req->recv.match_id,
			      req->recv.multi_recv ?
			      rxc->min_multi_recv : 0,
			      le_flags, NULL, rxc->rx_cmdq,
			      !(req->recv.flags & FI_MORE));
	if (ret != FI_SUCCESS) {
		RXC_WARN(rxc, "Failed to write Append command: %d\n", ret);
		return ret;
	}

	return FI_SUCCESS;
}

/*
 * cxip_cs_recv_cb() - Process user receive buffer events.
 *
 * For the CS protocol a receive buffer is described by an LE linked to
 * the Priority List. Local unexpected message buffering and rendezvous
 * messaging are not enabled.
 */
static int cxip_cs_recv_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_rxc_cs *rxc = req->recv.rxc_cs;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		/* Success events are disabled */
		assert(cxi_tgt_event_rc(event) != C_RC_OK);

		/* Failure to link a receive buffer is a fatal operation and
		 * indicates that FI_PROTO_CXI and portals flow-control is
		 * required.
		 */
		RXC_FATAL(rxc, APPEND_LE_FATAL);
		break;

	case C_EVENT_UNLINK:
		assert(!event->tgt_long.auto_unlinked);

		/* If request is for FI_MULTI_RECV and success events are being
		 * taken (completions required) then cxip_recv_req_report()
		 * will handle making sure the unlink is not reported prior to
		 * all messages being reported.
		 */
		req->recv.unlinked = true;
		cxip_recv_req_report(req);
		cxip_recv_req_free(req);

		return FI_SUCCESS;

	case C_EVENT_PUT:
		cxip_rxc_record_req_stat(&rxc->base, C_PTL_LIST_PRIORITY,
					 event->tgt_long.rlength, req);
		return cxip_complete_put(req, event);
	default:
		RXC_FATAL(rxc, CXIP_UNEXPECTED_EVENT,
			  cxi_event_to_str(event),
			  cxi_rc_to_str(cxi_event_rc(event)));
	}
}

static void cxip_rxc_cs_progress(struct cxip_rxc *rxc)
{
	cxip_evtq_progress(&rxc->rx_evtq);
}

static void cxip_rxc_cs_recv_req_tgt_event(struct cxip_req *req,
					   const union c_event *event)
{
	union cxip_match_bits mb = {
		.raw = event->tgt_long.match_bits
	};
	uint32_t init = event->tgt_long.initiator.initiator.process;

	assert(req->recv.rxc->protocol == FI_PROTO_CXI_CS);
	assert(event->hdr.event_type == C_EVENT_PUT);

	req->tag = mb.cs_tag;
	req->recv.initiator = init;

	if (mb.cs_cq_data)
		req->flags |= FI_REMOTE_CQ_DATA;

	req->recv.src_offset = event->tgt_long.remote_offset;

	/* Only need one event to set remaining fields. */
	if (req->recv.tgt_event)
		return;

	req->recv.tgt_event = true;

	/* VNI is needed to support FI_AV_AUTH_KEY. */
	req->recv.vni = event->tgt_long.vni;

	/* rlen is used to detect truncation. */
	req->recv.rlen = event->tgt_long.rlength;

	/* RC is used when generating completion events. */
	req->recv.rc = cxi_tgt_event_rc(event);

	/* Header data is provided in all completion events. */
	req->data = event->tgt_long.header_data;

	/* data_len must be set uniquely for each protocol! */
}

static int cxip_rxc_cs_cancel_msg_recv(struct cxip_req *req)
{
	/* Perform default */
	return cxip_recv_cancel(req);
}

/* Handle any control messaging callbacks specific to protocol */
static int cxip_rxc_cs_ctrl_msg_cb(struct cxip_ctrl_req *req,
				    const union c_event *event)
{
	/* Placeholder */
	return -FI_ENOSYS;
}

static void cxip_rxc_cs_init_struct(struct cxip_rxc *rxc_base,
				    struct cxip_ep_obj *ep_obj)
{
	struct cxip_rxc_cs *rxc = container_of(rxc_base, struct cxip_rxc_cs,
					       base);

	assert(rxc->base.protocol == FI_PROTO_CXI_CS);

	/* Overrides */
	rxc->base.recv_ptl_idx = CXIP_PTL_IDX_RNR_RXQ;
}

static void cxip_rxc_cs_fini_struct(struct cxip_rxc *rxc)
{
	/* Placeholder */
}

static int cxip_rxc_cs_msg_init(struct cxip_rxc *rxc_base)
{
	struct cxip_rxc_cs *rxc = container_of(rxc_base, struct cxip_rxc_cs,
					       base);
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.lossless = cxip_env.msg_lossless,
	};
	int ret;

	assert(rxc->base.protocol == FI_PROTO_CXI_CS);

	/* If applications AVs are symmetric, use logical FI addresses for
	 * matching. Otherwise, physical addresses will be used.
	 */
	if (rxc->base.ep_obj->av->symmetric) {
		CXIP_DBG("Using logical PTE matching\n");
		pt_opts.use_logical = 1;
	}

	ret = cxip_pte_alloc(rxc->base.ep_obj->ptable, rxc->base.rx_evtq.eq,
			     rxc->base.recv_ptl_idx, false, &pt_opts,
			     cxip_rnr_recv_pte_cb, &rxc->base,
			     &rxc->base.rx_pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate RX PTE: %d\n", ret);
		return ret;
	}

	/* Start accepting Puts. */
	ret = cxip_pte_set_state(rxc->base.rx_pte, rxc->base.rx_cmdq,
				 C_PTLTE_ENABLED, 0);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("cxip_pte_set_state returned: %d\n", ret);
		goto free_pte;
	}

	/* Wait for PTE state change */
	do {
		sched_yield();
		cxip_evtq_progress(&rxc->base.rx_evtq);
	} while (rxc->base.rx_pte->state != C_PTLTE_ENABLED);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(rxc->base.rx_pte);

	return ret;
}

static int cxip_rxc_cs_msg_fini(struct cxip_rxc *rxc_base)
{
	assert(rxc_base->protocol == FI_PROTO_CXI_CS);

	return FI_SUCCESS;
}

static void cxip_rxc_cs_cleanup(struct cxip_rxc *rxc_base)
{
	/* Cancel Receives */
	cxip_rxc_recv_req_cleanup(rxc_base);
}

/*
 * cxip_recv_common() - Common message receive function. Used for tagged and
 * untagged sends of all sizes.
 */
static ssize_t
cxip_recv_common(struct cxip_rxc *rxc, void *buf, size_t len, void *desc,
		 fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		 void *context, uint64_t flags, bool tagged,
		 struct cxip_cntr *comp_cntr)
{
	struct cxip_req *req = NULL;
	struct cxip_mr *mr = rxc->domain->hybrid_mr_desc ? desc : NULL;
	int ret;
	uint32_t match_id;
	uint16_t vni;

	assert(rxc->protocol == FI_PROTO_CXI_CS);

#if ENABLE_DEBUG
	if (len && !buf) {
		RXC_WARN(rxc, "Length %ld but local buffer NULL\n", len);
		return -FI_EINVAL;
	}

	if (rxc->state == RXC_DISABLED)
		return -FI_EOPBADSTATE;

	if (tagged) {
		if (tag & ~CXIP_CS_TAG_MASK || ignore & ~CXIP_CS_TAG_MASK) {
			RXC_WARN(rxc,
				 "Invalid tag: %#018lx ignore: %#018lx (%#018lx)\n",
				 tag, ignore, CXIP_CS_TAG_MASK);
			return -FI_EINVAL;
		}
	}
#endif

	if (!rxc->selective_completion)
		flags |= FI_COMPLETION;

	ret = cxip_set_recv_match_id(rxc, src_addr, rxc->ep_obj->av_auth_key &&
				     (flags & FI_AUTH_KEY), &match_id, &vni);
	if (ret) {
		RXC_WARN(rxc, "Error setting match_id: %d %s\n",
			 ret, fi_strerror(-ret));
		return ret;
	}

	ofi_genlock_lock(&rxc->ep_obj->lock);
	ret = cxip_recv_req_alloc(rxc, buf, len, mr ? mr->md : NULL,
				  &req, cxip_cs_recv_cb);
	if (ret)
		goto err;

	req->flags = ((tagged ? FI_TAGGED : FI_MSG) | FI_RECV |
		       (flags & FI_COMPLETION));
	req->context = (uint64_t)context;
	req->recv.cntr = comp_cntr ? comp_cntr : rxc->recv_cntr;
	req->recv.match_id = match_id;
	req->recv.vni = vni;
	req->recv.tag = tag;
	req->recv.ignore = ignore;
	req->recv.flags = flags;
	req->recv.tagged = tagged;
	req->recv.multi_recv = (flags & FI_MULTI_RECV ? true : false);

	if (!(req->recv.flags & (FI_PEEK | FI_CLAIM))) {
		ret = cxip_cs_recv_req(req, false);
		if (ret) {
			RXC_WARN(rxc, "Receive append failed: %d %s\n",
				 ret, fi_strerror(-ret));
			goto free_req;
		}
		ofi_genlock_unlock(&rxc->ep_obj->lock);

		RXC_DBG(rxc,
			"req: %p buf: %p len: %lu src_addr: %ld tag(%c):"
			" 0x%lx ignore: 0x%lx context: %p\n",
			req, buf, len, src_addr, tagged ? '*' : '-', tag,
			ignore, context);

		return FI_SUCCESS;
	}

	/* No buffered unexpected messages, so FI_PEEK always fails */
	if (req->recv.flags & FI_PEEK) {
		req->recv.rc = C_RC_NO_MATCH;
		cxip_recv_req_peek_complete(req, NULL);
		ofi_genlock_unlock(&rxc->ep_obj->lock);

		return FI_SUCCESS;
	}

	/* FI_CLAIM specified by itself cannot be valid */
	RXC_WARN(rxc, "FI_CLAIM not valid\n");
	ret = -FI_EINVAL;

free_req:
	cxip_recv_req_free(req);
err:
	ofi_genlock_unlock(&rxc->ep_obj->lock);

	return ret;
}

static inline bool cxip_cs_req_uses_idc(struct cxip_req *req)
{
	/* TODO: Consider supporting HMEM and IDC by mapping memory */
	return  !req->send.txc->hmem && req->send.len &&
		req->send.len <= CXIP_INJECT_SIZE &&
		!req->triggered && !cxip_env.disable_non_inject_msg_idc;
}

static inline ssize_t cxip_cs_send_dma(struct cxip_req *req,
				       union cxip_match_bits *mb,
				       union c_fab_addr *dfa, uint8_t idx_ext)
{
	struct cxip_txc *txc = req->send.txc;
	struct c_full_dma_cmd cmd = {};

	cmd.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.command.opcode = C_CMD_PUT;
	cmd.index_ext = idx_ext;
	cmd.event_send_disable = 1;
	cmd.dfa = *dfa;
	cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);
	cmd.user_ptr = (uint64_t)req;
	cmd.initiator = cxip_msg_match_id(txc);
	cmd.match_bits = mb->raw;
	cmd.header_data = req->send.data;

	/* Triggered ops could result in 0 length DMA */
	if (req->send.send_md) {
		cmd.lac = req->send.send_md->md->lac;
		cmd.local_addr = CXI_VA_TO_IOVA(req->send.send_md->md,
						req->send.buf);
		cmd.request_len = req->send.len;
	}

	if (req->send.cntr) {
		cmd.event_ct_ack = 1;
		cmd.ct = req->send.cntr->ct->ctn;
	}

	return cxip_txc_emit_dma(txc, req->send.caddr.vni,
				 cxip_ofi_to_cxi_tc(req->send.tclass),
				 CXI_TC_TYPE_DEFAULT,
				 req->triggered ?  req->trig_cntr : NULL,
				 req->trig_thresh, &cmd, req->send.flags);
}

static inline ssize_t cxip_cs_send_idc(struct cxip_req *req,
				       union cxip_match_bits *mb,
				       union c_fab_addr *dfa, uint8_t idx_ext)
{
	struct cxip_txc *txc = req->send.txc;
	struct c_cstate_cmd cstate_cmd = {};
	struct c_idc_msg_hdr idc_cmd;

	assert(req->send.len > 0);
	assert(!txc->hmem);

	cstate_cmd.event_send_disable = 1;
	cstate_cmd.index_ext = idx_ext;
	cstate_cmd.eq = cxip_evtq_eqn(&txc->tx_evtq);
	cstate_cmd.initiator = cxip_msg_match_id(txc);

	if (req->send.cntr) {
		cstate_cmd.event_ct_ack = 1;
		cstate_cmd.ct = req->send.cntr->ct->ctn;
	}

	/* Note: IDC command completely filled in */
	idc_cmd.unused_0 = 0;
	idc_cmd.dfa = *dfa;
	idc_cmd.match_bits = mb->raw;
	idc_cmd.header_data = req->send.data;
	idc_cmd.user_ptr = (uint64_t)req;

	return cxip_txc_emit_idc_msg(txc, req->send.caddr.vni,
				     cxip_ofi_to_cxi_tc(req->send.tclass),
				     CXI_TC_TYPE_DEFAULT, &cstate_cmd, &idc_cmd,
				     req->send.buf, req->send.len,
				     req->send.flags);
}

/* Caller must hold ep_obj->lock */
static ssize_t cxip_cs_msg_send(struct cxip_req *req)
{
	struct cxip_txc *txc = req->send.txc;
	union cxip_match_bits mb = {
		.cs_vni = req->send.caddr.vni,
		.cs_tag = req->send.tag,
		.cs_cq_data = !!(req->send.flags & FI_REMOTE_CQ_DATA),
	};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	ssize_t ret;
	bool idc = !req->send.send_md || !req->send.len;

	/* Calculate DFA */
	cxi_build_dfa(req->send.caddr.nic, req->send.caddr.pid, txc->pid_bits,
		      txc->recv_ptl_idx, &dfa, &idx_ext);

	if (req->send.send_md || !req->send.len)
		ret = cxip_cs_send_dma(req, &mb, &dfa, idx_ext);
	else
		ret = cxip_cs_send_idc(req, &mb, &dfa, idx_ext);
	if (ret) {
		TXC_WARN(txc, "Failed to write %s command: %ld\n",
			 idc ? "IDC" : "DMA", ret);
		return ret;
	}

	TXC_DBG(txc, "Send %s command submitted for req %p\n",
		idc ? "IDC" : "DMA", req);

	return FI_SUCCESS;
}

/* Queue RNR retry. There are CXIP_NUM_RNR_WAIT_QUEUE, each
 * has a consistent time wait for that queue (smaller to larger).
 * Therefore, appends to tail will keep each queue in retry time
 * order.
 *
 * Caller must hold ep_obj->lock
 */
static int cxip_rnr_queue_retry(struct cxip_txc_cs *txc, struct cxip_req *req)
{
	uint64_t cur_time;
	uint64_t retry_time;
	int index;

	cur_time = ofi_gettime_us();

	index = req->send.retries < CXIP_NUM_RNR_WAIT_QUEUE ?
			req->send.retries : CXIP_NUM_RNR_WAIT_QUEUE - 1;

	/* 1us, 11us, 81us 271us, 641us (max) */
	retry_time = cur_time + 1 + (index * index * index) * 10;
#if 0
	TXC_WARN(txc, "retry_time %ld req->send.max_rnr_time %ld\n",
		 retry_time, req->send.max_rnr_time);
#endif
	if (retry_time > req->send.max_rnr_time)
		return -FI_ETIMEDOUT;

	/* Insert and update next timeout */
	req->send.retry_rnr_time = retry_time;

	dlist_insert_tail(&req->send.rnr_entry, &txc->time_wait_queue[index]);
	if (retry_time < txc->next_retry_wait_us)
		txc->next_retry_wait_us = retry_time;

	req->send.retries++;
	ofi_atomic_inc32(&txc->time_wait_reqs);
#if 0
	TXC_WARN(txc, "Entry added to txc->time_wait_queue[%d]\n", index);
	TXC_WARN(txc,
		 "txc->next_retry_wait_us %ld, req->send.retry_rnr_time %ld\n",
		 txc->next_retry_wait_us, req->send.retry_rnr_time);
#endif

	return FI_SUCCESS;
}

static int cxip_process_rnr_time_wait(struct cxip_txc_cs *txc)
{
	struct cxip_req *req;
	struct dlist_entry *tmp;
	uint64_t cur_time;
	uint64_t next_time;
	int index;
	int ret;

#if 0
	TXC_WARN(txc, "Process RNR timewait, wait_reqs %d "
		 "txc->next_retry_wait_us %ld\n",
		 ofi_atomic_get32(&txc->time_wait_reqs),
		 txc->next_retry_wait_us);
#endif
	if (!ofi_atomic_get32(&txc->time_wait_reqs))
		return FI_SUCCESS;

	cur_time = ofi_gettime_us();
	if (cur_time < txc->next_retry_wait_us)
		return FI_SUCCESS;

	ret = FI_SUCCESS;
	for (index = 0; index < CXIP_NUM_RNR_WAIT_QUEUE; index++) {
		dlist_foreach_container_safe(&txc->time_wait_queue[index],
					     struct cxip_req, req,
					     send.rnr_entry, tmp) {
#if 0
			TXC_WARN(txc, "req %p, req->send.retry_rnr_time "
				 "%ld cur_time %ld\n", req,
				 req->send.retry_rnr_time, cur_time);
#endif
			if (req->send.retry_rnr_time <= cur_time) {

				/* Do not retry if TX canceled */
				if (req->send.canceled) {
					dlist_remove_init(&req->send.rnr_entry);
					ofi_atomic_dec32(&txc->time_wait_reqs);
					cxip_send_buf_fini(req);
					cxip_report_send_completion(req, true);
					ofi_atomic_dec32(&txc->base.otx_reqs);
					cxip_evtq_req_free(req);

					continue;
				}

				/* Must TX return credit, will take it back if
				 * we could not send.
				 */
				ofi_atomic_dec32(&txc->base.otx_reqs);
				ret = cxip_cs_msg_send(req);
				if (ret != FI_SUCCESS) {
					ofi_atomic_inc32(&txc->base.otx_reqs);
					goto reset_min_time_wait;
				}

				dlist_remove_init(&req->send.rnr_entry);
				ofi_atomic_dec32(&txc->time_wait_reqs);
			} else {
				break;
			}
		}
	}

reset_min_time_wait:
	next_time = UINT64_MAX;

	for (index = 0; index < CXIP_NUM_RNR_WAIT_QUEUE; index++) {
		req = dlist_first_entry_or_null(&txc->time_wait_queue[index],
						struct cxip_req,
						send.rnr_entry);
		if (req && req->send.retry_rnr_time < next_time)
			next_time = req->send.retry_rnr_time;
	}
#if 0
	TXC_WARN(txc, "Set txc->next_retry_wait_us to %ld\n", next_time);
#endif
	txc->next_retry_wait_us = next_time;

	return ret;
}

static void cxip_txc_cs_progress(struct cxip_txc *txc_base)
{
	struct cxip_txc_cs *txc = container_of(txc_base, struct cxip_txc_cs,
					       base);

	assert(txc->base.protocol == FI_PROTO_CXI_CS);

	cxip_evtq_progress(&txc->base.tx_evtq);
	cxip_process_rnr_time_wait(txc);
}

static int cxip_txc_cs_cancel_msg_send(struct cxip_req *req)
{
	if (req->type != CXIP_REQ_SEND)
		return -FI_ENOENT;

	req->send.canceled = true;

	return FI_SUCCESS;
}

static void cxip_txc_cs_init_struct(struct cxip_txc *txc_base,
				    struct cxip_ep_obj *ep_obj)
{
	struct cxip_txc_cs *txc = container_of(txc_base, struct cxip_txc_cs,
					       base);
	int i;

	assert(txc->base.protocol == FI_PROTO_CXI_CS);

	txc->base.recv_ptl_idx = CXIP_PTL_IDX_RNR_RXQ;
	ofi_atomic_initialize32(&txc->time_wait_reqs, 0);
	txc->max_retry_wait_us = cxip_env.rnr_max_timeout_us;
	txc->next_retry_wait_us = UINT64_MAX;

	for (i = 0; i < CXIP_NUM_RNR_WAIT_QUEUE; i++)
		dlist_init(&txc->time_wait_queue[i]);
}

static void cxip_txc_cs_fini_struct(struct cxip_txc *txc)
{
	/* Placeholder */
}

static int cxip_txc_cs_msg_init(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static int cxip_txc_cs_msg_fini(struct cxip_txc *txc_base)
{
	/* Placeholder */
	return FI_SUCCESS;
}

static void cxip_txc_cs_cleanup(struct cxip_txc *txc_base)
{
	/* Placeholder */
}

static void cxip_cs_send_req_dequeue(struct cxip_req *req)
{
	/* TODO: Place holder for anything additional */

	dlist_remove(&req->send.txc_entry);
}

static int cxip_cs_send_cb(struct cxip_req *req, const union c_event *event)
{
	struct cxip_txc_cs *txc = req->send.txc_cs;
	int rc = cxi_event_rc(event);
	int ret;

#if 0
	TXC_WARN(txc, "Event %s RC %s received\n",
		 cxi_event_to_str(event),
		 cxi_rc_to_str(rc));
#endif

	/* Handle at TX FI_MSG/FI_TAGGED message events */
	if (event->hdr.event_type != C_EVENT_ACK) {
		TXC_WARN(req->send.txc, CXIP_UNEXPECTED_EVENT,
			 cxi_event_to_str(event),
			 cxi_rc_to_str(rc));
		return FI_SUCCESS;
	}

	req->send.rc = rc;

	/* Handle RNR acks */
	if (rc == C_RC_ENTRY_NOT_FOUND &&
	    txc->base.enabled && !req->send.canceled) {

		ret  = cxip_rnr_queue_retry(txc, req);

		if (ret == FI_SUCCESS)
			return ret;

		TXC_WARN(&txc->base, "req %p RNR max timeout buf: %p len: %lu, "
			 "dest_addr: 0x%lX nic: %#x pid: %d tag(%c) 0x%lx "
			 "retries %u TX outstanding %u\n", req, req->send.buf,
			 req->send.len, req->send.dest_addr,
			 req->send.caddr.nic, req->send.caddr.pid,
			 req->send.tagged ? '*' : '-',
			 req->send.tag, req->send.retries,
			 ofi_atomic_get32(&txc->base.otx_reqs));
	}

	cxip_cs_send_req_dequeue(req);
	cxip_send_buf_fini(req);

	/* If status is good, then the request completed before it could
	 * be canceled. If canceled, indicate software update of the
	 * error count is required.
	 */
	if (rc == C_RC_OK)
		req->send.canceled = false;

	cxip_report_send_completion(req, req->send.canceled);

	ofi_atomic_dec32(&txc->base.otx_reqs);
	cxip_evtq_req_free(req);

	return FI_SUCCESS;
}

/*
 * cxip_send_common() - Common message send function. Used for tagged and
 * untagged sends of all sizes. This includes triggered operations.
 */
static ssize_t
cxip_send_common(struct cxip_txc *txc, uint32_t tclass, const void *buf,
		 size_t len, void *desc, uint64_t data, fi_addr_t dest_addr,
		 uint64_t tag, void *context, uint64_t flags, bool tagged,
		 bool triggered, uint64_t trig_thresh,
		 struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr)
{
	struct cxip_txc_cs *txc_cs = container_of(txc, struct cxip_txc_cs,
						  base);
	struct cxip_mr *mr = txc->domain->hybrid_mr_desc ? desc : NULL;
	struct cxip_req *req;
	struct cxip_addr caddr;
	int ret;
	bool idc;

	assert(txc->protocol == FI_PROTO_CXI_CS);

#if ENABLE_DEBUG
	if (len && !buf) {
		TXC_WARN(txc, "Length %ld but source buffer NULL\n", len);
		return -FI_EINVAL;
	}
#endif
	/* TODO: This check should not be required in other than debug builds,
	 * to do that we would need to return -FI_EFAULT, so leaving here for
	 * now.
	 */
	if (len > CXIP_EP_MAX_MSG_SZ)
		return -FI_EMSGSIZE;

	/* TODO: Move to tagged sends */
	if (tagged && tag & ~CXIP_TAG_MASK) {
		TXC_WARN(txc, "Invalid tag: %#018lx (%#018lx)\n",
			 tag, CXIP_TAG_MASK);
		return -FI_EINVAL;
	}
#if 0 /* No inject support */
	/* TODO: move to inject/sendmsg */
	if (flags & FI_INJECT && len > CXIP_INJECT_SIZE) {
		TXC_WARN(txc, "Invalid inject length: %lu\n", len);
		return -FI_EMSGSIZE;
	}
#endif
	ofi_genlock_lock(&txc->ep_obj->lock);
	/* If RNR list is not empty, check if the first retry entry time
	 * wait has expired, and if so force progress to initiate any
	 * read retry/retries.
	 */
	if (txc_cs->next_retry_wait_us != UINT64_MAX &&
	    ofi_atomic_get32(&txc_cs->time_wait_reqs)) {
		if (ofi_gettime_us() >= txc_cs->next_retry_wait_us)
			cxip_txc_cs_progress(txc);
	}

	req = cxip_evtq_req_alloc(&txc->tx_evtq, false, txc);
	if (!req) {
		TXC_DBG(txc, "Failed to allocate request, return -FI_EAGAIN\n");
		ret = -FI_EAGAIN;
		goto unlock;
	}

	/* Restrict outstanding success event requests to queue size */
	if (ofi_atomic_get32(&txc->otx_reqs) > txc->attr.size) {
		ret = -FI_EAGAIN;
		goto free_req;
	}

	req->triggered = triggered;
	req->trig_thresh = trig_thresh;
	req->trig_cntr = trig_cntr;

	/* Save Send parameters to replay */
	req->type = CXIP_REQ_SEND;
	req->send.txc = txc;
	req->send.tclass = tclass;
	req->send.cntr = triggered ? comp_cntr : txc->send_cntr;
	req->send.buf = buf;
	req->send.len = len;
	req->send.data = data;
	req->send.flags = flags;
	/* Set completion parameters */
	req->context = (uint64_t)context;
	req->flags = FI_SEND | (flags & (FI_COMPLETION | FI_MATCH_COMPLETE));
	if (tagged) {
		req->send.tagged = tagged;
		req->send.tag = tag;
		req->flags |= FI_TAGGED;
	} else {
		req->flags |= FI_MSG;
	}

	req->cb = cxip_cs_send_cb;
	idc = cxip_cs_req_uses_idc(req);

	if (req->send.len && !idc) {
		if (!mr) {
			ret = cxip_map(txc->domain, req->send.buf,
				       req->send.len, 0, &req->send.send_md);
			if (ret) {
				TXC_WARN(txc,
					 "Local buffer map failed: %d %s\n",
					 ret, fi_strerror(-ret));
				goto free_req;
			}
		} else {
			req->send.send_md = mr->md;
			req->send.hybrid_md = true;
		}
	}

	/* Look up target CXI address */
	ret = cxip_av_lookup_addr(txc->ep_obj->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		TXC_WARN(txc, "Failed to look up FI addr: %d %s\n",
			 ret, fi_strerror(-ret));
		goto free_map;
	}

	if (!txc->ep_obj->av_auth_key)
		caddr.vni = txc->ep_obj->auth_key.vni;

	req->send.caddr = caddr;
	req->send.dest_addr = dest_addr;

	if (cxip_evtq_saturated(&txc->tx_evtq)) {
		TXC_DBG(txc, "TX HW EQ saturated\n");
		ret = -FI_EAGAIN;
		goto free_map;
	}

	/* Enqueue on the TXC. TODO: Consider if we should examine
	 * the RNR retry list and push back if a RNR is outstanding
	 * to a peer.
	 */
	dlist_insert_tail(&req->send.txc_entry, &txc->msg_queue);
	req->send.max_rnr_time = ofi_gettime_us() + txc_cs->max_retry_wait_us;

	/* Try Send */
	ret = cxip_cs_msg_send(req);
	if (ret != FI_SUCCESS)
		goto req_dequeue;

	ofi_genlock_unlock(&txc->ep_obj->lock);

	TXC_DBG(txc,
		"req: %p buf: %p len: %lu dest_addr: 0x%lX nic: %d "
		"pid: %d tag(%c): 0x%lx context %#lx\n",
		req, req->send.buf, req->send.len, dest_addr, caddr.nic,
		caddr.pid, req->send.tagged ? '*' : '-', req->send.tag,
		req->context);

	return FI_SUCCESS;

req_dequeue:
	cxip_cs_send_req_dequeue(req);
free_map:
	if (req->send.send_md && !req->send.hybrid_md)
		cxip_unmap(req->send.send_md);
free_req:
	cxip_evtq_req_free(req);
unlock:
	ofi_genlock_unlock(&txc->ep_obj->lock);

	return ret;
}

struct cxip_rxc_ops cs_rxc_ops = {
	.recv_common = cxip_recv_common,
	.progress = cxip_rxc_cs_progress,
	.recv_req_tgt_event = cxip_rxc_cs_recv_req_tgt_event,
	.cancel_msg_recv = cxip_rxc_cs_cancel_msg_recv,
	.ctrl_msg_cb = cxip_rxc_cs_ctrl_msg_cb,
	.init_struct = cxip_rxc_cs_init_struct,
	.fini_struct = cxip_rxc_cs_fini_struct,
	.cleanup = cxip_rxc_cs_cleanup,
	.msg_init = cxip_rxc_cs_msg_init,
	.msg_fini = cxip_rxc_cs_msg_fini,
};

struct cxip_txc_ops cs_txc_ops = {
	.send_common = cxip_send_common,
	.progress = cxip_txc_cs_progress,
	.cancel_msg_send = cxip_txc_cs_cancel_msg_send,
	.init_struct = cxip_txc_cs_init_struct,
	.fini_struct = cxip_txc_cs_fini_struct,
	.cleanup = cxip_txc_cs_cleanup,
	.msg_init = cxip_txc_cs_msg_init,
	.msg_fini = cxip_txc_cs_msg_fini,
};
