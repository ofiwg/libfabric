/*
 * Copyright (C) 2026 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_hfi1_transport.h"
#include "rdma/opx/fi_opx_rma.h"
#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/opx_hfisvc_poll.h"
#include "rdma/opx/opx_hfisvc_rma.h"

#if HAVE_HFISVC
/* Standalone CQ error for a dropped iov with no shared PUT_CQ context (plain PUT / GET). */
static inline void opx_hfisvc_rma_post_standalone_err(struct fi_opx_ep *opx_ep, uint64_t flags, uint64_t data, int err,
						      int prov_errno)
{
	struct opx_context *err_context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
	if (OFI_UNLIKELY(err_context == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "ctx_pool exhausted; HFISVC RMA error not posted to EQ\n");
		return;
	}
	err_context->next		     = NULL;
	err_context->flags		     = flags;
	err_context->len		     = 0;
	err_context->buf		     = NULL;
	err_context->data		     = data;
	err_context->tag		     = 0;
	err_context->byte_counter	     = 0;
	err_context->err_entry.flags	     = flags;
	err_context->err_entry.len	     = 0;
	err_context->err_entry.buf	     = NULL;
	err_context->err_entry.data	     = data;
	err_context->err_entry.tag	     = 0;
	err_context->err_entry.olen	     = 0;
	err_context->err_entry.err	     = err;
	err_context->err_entry.prov_errno    = prov_errno;
	err_context->err_entry.op_context    = NULL;
	err_context->err_entry.err_data	     = NULL;
	err_context->err_entry.err_data_size = 0;
	slist_insert_tail((struct slist_entry *) err_context, opx_ep->rx->cq_err_ptr);
}

/* PUT_CQ dropped iov: sticky error (first failure wins) with data from the canonical params->data;
 * flags/len keep PUT_CQ semantics (FI_REMOTE_CQ_DATA retained, len = bytes), unlike the standalone path. */
static inline void opx_hfisvc_rma_putcq_drop(struct fi_opx_ep *opx_ep, struct opx_context *context, uint64_t data,
					     uint64_t len, int err, int prov_errno)
{
	/* data is invariant across every iov of one PUT_CQ op (all share params->data == context->data, both
	 * sourced from hdr->match.ofi_data), so latch the whole err_entry together on the first failure. */
	assert((context->flags & FI_REMOTE_CQ_DATA) && context->data == data);
	if (context->err_entry.err == 0) {
		context->err_entry.flags	 = context->flags;
		context->err_entry.len		 = context->len;
		context->err_entry.buf		 = context->buf;
		context->err_entry.data		 = data;
		context->err_entry.tag		 = context->tag;
		context->err_entry.olen		 = 0;
		context->err_entry.err		 = err;
		context->err_entry.prov_errno	 = prov_errno;
		context->err_entry.err_data	 = NULL;
		context->err_entry.err_data_size = 0;
	}
	opx_hfisvc_rma_context_drain_and_route(opx_ep, context, len);
}

/* Route a dropped iov to the sticky PUT_CQ error or a standalone CQ error. */
static inline void opx_hfisvc_rma_drop(struct fi_opx_ep *opx_ep, struct opx_hfisvc_rma_rts_params *params,
				       uint64_t drop_flags, uint64_t len, bool has_cq_data, int err, int prov_errno)
{
	if (has_cq_data && params->context != NULL) {
		opx_hfisvc_rma_putcq_drop(opx_ep, params->context, params->data, len, err, prov_errno);
	} else {
		opx_hfisvc_rma_post_standalone_err(opx_ep, drop_flags, params->data, err, prov_errno);
	}
}

int opx_hfisvc_rma_send_rts(union fi_opx_hfi1_deferred_work *work)
{
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "HFISVC_RMA_SEND_RTS");
	struct opx_hfisvc_rma_rts_params *params    = &work->hfisvc_rma_rts;
	struct fi_opx_ep		 *opx_ep    = params->opx_ep;
	const enum opx_hfi1_type	  hfi1_type = OPX_SW_HFI1_TYPE(opx_ep->domain);
	const uint64_t			  niov	    = params->niov;
	uint64_t			  cur_iov   = params->cur_iov;
	const uint8_t			  plane_idx = params->plane_idx;

	OPX_HFISVC_DEBUG_LOG("HFISVC Attempting rma_send_rts for %s work item=%p cur_iov=%lu niov=%lu!\n",
			     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work, cur_iov, niov);

	const enum opx_hfisvc_xfer_type xfer_type = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_GET) ?
							    OPX_HFISVC_XFER_TYPE_RMA_READ :
							    OPX_HFISVC_XFER_TYPE_RMA_WRITE;

	/* iovs already registered on a prior pass; skip re-registering to avoid double-register */
	const uint64_t run_base	      = cur_iov;
	const uint64_t pre_registered = params->iovs_with_keys;
	cur_iov += pre_registered;

	while (cur_iov < niov) {
		uint32_t access_key = (uint32_t) -1;

		if (opx_hfisvc_keyset_alloc_key(&opx_ep->domain->hfisvc.ctxs[plane_idx].access_key_set, &access_key,
						FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep))) {
			break;
		}

		struct opx_hfisvc_xfer_completion *internal_completion =
			(struct opx_hfisvc_xfer_completion *) ofi_buf_alloc(opx_ep->hfisvc.completion_pool);
		if (OFI_UNLIKELY(internal_completion == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"HFISVC send completion_pool exhausted; retrying RTS\n");
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.rma_send_rts.enomem_completion);
			opx_hfisvc_keyset_free_key(opx_ep->domain->hfisvc.ctxs[plane_idx].access_key_set, access_key,
						   FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep));

			if (OFI_UNLIKELY(++params->send_eagain_attempts >= OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
				/* drop-drain only when no registered run is pending, else the RTS iov index desyncs */
				if (params->iovs_with_keys == 0) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC send completion_pool stuck exhausted after %u attempts; iov dropped\n",
						(unsigned) params->send_eagain_attempts);
					FI_OPX_DEBUG_COUNTERS_INC(
						opx_ep->debug_counters.hfisvc.rma_send_rts.enomem_completion_dropped);
					if (params->cc) {
						const uint64_t drop_len = params->local_buf_iovs[cur_iov].len;
						assert(drop_len <= params->cc->byte_counter);
						params->cc->byte_counter -= drop_len;
						if (params->cc->byte_counter == 0) {
							params->cc->hit_zero(params->cc);
						}
					}
					++cur_iov;
					continue;
				}
			}
			break;
		}

		internal_completion->type	= xfer_type;
		internal_completion->access_key = access_key;
		internal_completion->cc		= params->cc;
		internal_completion->context	= NULL;
		internal_completion->opx_mr	= NULL;
		internal_completion->opx_ep	= opx_ep;
		internal_completion->len	= params->local_buf_iovs[cur_iov].len;
		internal_completion->flags	= 0;

		struct hfisvc_client_completion completion = {
			.flags		= OPX_HFISVC_CMPL_CQ,
			.cq.handle	= opx_ep->hfisvc.emr_completion_queues[plane_idx],
			.cq.app_context = (uint64_t) internal_completion,
		};

		int rc = opx_ep->domain->hfisvc.cmd_dma_access_once_va(
			opx_ep->hfisvc.command_queues[plane_idx], completion, 0UL /* flags */, access_key,
			params->local_buf_iovs[cur_iov].len, (void *) params->local_buf_iovs[cur_iov].buf);

		if (OFI_UNLIKELY(rc != 0)) {
			OPX_BUF_FREE(internal_completion);
			opx_hfisvc_keyset_free_key(opx_ep->domain->hfisvc.ctxs[plane_idx].access_key_set, access_key,
						   FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep));

			if (OFI_LIKELY(rc == -FI_EAGAIN)) {
				OPX_HFISVC_DEBUG_LOG("EAGAIN (hfisvc_client queue returned %d)\n", rc);
				break;
			}

			/* hard error: drop-drain this iov (only safe with no registered run pending) */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "hfisvc cmd_dma_access failed rc=%d; iov dropped\n",
				rc);
			if (params->iovs_with_keys == 0) {
				if (params->cc) {
					const uint64_t drop_len = params->local_buf_iovs[cur_iov].len;
					assert(drop_len <= params->cc->byte_counter);
					params->cc->byte_counter -= drop_len;
					if (params->cc->byte_counter == 0) {
						params->cc->hit_zero(params->cc);
					}
				}
				++cur_iov;
				continue;
			}
			break;
		}

		params->hfisvc_iov[cur_iov].origin_hfisvc_client_key =
			opx_ep->domain->hfisvc.ctxs[plane_idx].client_key;
		params->hfisvc_iov[cur_iov].origin_hfisvc_access_key = access_key;
		++params->iovs_with_keys;
		++cur_iov;
	}

	/* ring the doorbell only if this pass registered a new buffer; roll back on failure */
	if (params->iovs_with_keys > pre_registered) {
		int rc = (*opx_ep->domain->hfisvc.doorbell)(opx_ep->domain->hfisvc.ctxs[plane_idx].ctx);
		if (OFI_UNLIKELY(rc != 0)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "HFISVC doorbell failed rc=%d\n", rc);
			for (uint64_t i = run_base + pre_registered; i < cur_iov; ++i) {
				opx_hfisvc_keyset_free_key(opx_ep->domain->hfisvc.ctxs[plane_idx].access_key_set,
							   params->hfisvc_iov[i].origin_hfisvc_access_key,
							   FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep));
			}
			params->iovs_with_keys = pre_registered;
			params->cur_iov	       = run_base;
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_SEND_RTS");
			return -FI_EAGAIN;
		}
	}

	// There are no iovs queued to be sent
	if (params->iovs_with_keys == 0) {
		if (cur_iov == niov) {
			OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_SEND_RTS");
			OPX_HFISVC_DEBUG_LOG(
				"HFISVC completed rma_send_rts (all iovs drop-drained) for %s work item=%p!\n",
				opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
			params->work_elem.complete = true;
			return FI_SUCCESS;
		}
		params->cur_iov = cur_iov;
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_SEND_RTS");
		OPX_HFISVC_DEBUG_LOG("HFISVC EAGAIN (no keys registered) for rma_send_rts for %s work item=%p!\n",
				     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
		return -FI_EAGAIN;
	}

	const uint64_t payload_qws_total = (params->iovs_with_keys * sizeof(union opx_hfisvc_rma_iov)) >> 3;

	OPX_SHD_CTX_PIO_LOCK(OPX_IS_CTX_SHARING_ENABLED, opx_ep->tx);

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	/* 1 PIO credit == 16 dwords; size for the larger 16B layout */
	const int credits_needed    = (((16 + 2 + 2) + 15) >> 4) + (((payload_qws_total << 1) + 15) >> 4);
	ssize_t	  credits_available = fi_opx_hfi1_tx_check_credits(opx_ep->tx, &pio_state, credits_needed);
	opx_ep->tx->pio_state->qw0  = pio_state.qw0;

	if (OFI_UNLIKELY(credits_available < credits_needed)) {
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_SEND_RTS");
		OPX_HFISVC_DEBUG_LOG("HFISVC EAGAIN (credits) for rma_send_rts for %s work item=%p!\n",
				     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
		/* preserve registered iovs and roll the cursor back so the retry emits their RTS */
		params->cur_iov = cur_iov - params->iovs_with_keys;
		OPX_SHD_CTX_PIO_UNLOCK(OPX_IS_CTX_SHARING_ENABLED, opx_ep->tx);
		return -FI_EAGAIN;
	}

	struct fi_opx_reliability_tx_replay *replay;
	struct fi_opx_reliability_tx_flow   *flow;
	void				    *reservation_record = NULL;
	int32_t				     psn;

	const struct fi_opx_addr	 addr		   = params->opx_target_addr;
	const enum opx_flow_control_mode flow_control_mode = opx_ep->flow_control_mode;

	/* The HFISVC RTS is the only governed packet on this path: the data
	 * itself moves by service-side rdma_read/write and never lands in the
	 * receiver's RX queue, but 0xC4 is RX-classified governed, so the RTS
	 * must reserve or the flow's withholding never converges. */

	psn = fi_opx_reliability_get_replay(
		&opx_ep->ep_fid, opx_ep->reli_service, opx_ep->rx->self.planes[OPX_PRIMARY_PLANE].lid,
		addr.planes[OPX_PRIMARY_PLANE].lid, addr.planes[OPX_PRIMARY_PLANE].hfi1_subctxt_rx, 0, &flow, &replay,
		params->reliability, hfi1_type, flow_control_mode, &reservation_record);

	if (OFI_UNLIKELY(psn == -1)) {
		fi_opx_reliability_cancel_record(flow_control_mode, opx_ep, &reservation_record);
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_SEND_RTS");
		OPX_HFISVC_DEBUG_LOG("HFISVC EAGAIN (PSN/replay) for rma_send_rts for %s work item=%p!\n",
				     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
		params->cur_iov = cur_iov - params->iovs_with_keys;
		OPX_SHD_CTX_PIO_UNLOCK(OPX_IS_CTX_SHARING_ENABLED, opx_ep->tx);
		return -FI_EAGAIN;
	}

	uint64_t pkt_niov    = (uint64_t) params->iovs_with_keys << 48;
	uint64_t op64	     = (uint64_t) params->fi_op_opcode << 40;
	uint64_t dt64	     = (uint64_t) params->fi_datatype_dt << 32;
	uint64_t dput_opcode = (uint64_t) params->dput_opcode;

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_MIXED_9B)) {
		/* 9B do_replay copies payload from replay->payload, so RTS iovs go there */
		const uint64_t payload_bytes = payload_qws_total << 3;
		/* pbc(2) + lrh(2) + bth(3) + kdeth(9) + payload dws for rma iovs */
		const uint64_t pbc_dws = 2 + 2 + 3 + 9 + ((payload_bytes + 3) >> 2);
		const uint16_t lrh_dws = htons(pbc_dws - 2 + 1);

		opx_cacheline_store_qw(
			replay->scb.qws,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.qw0 |
				OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid |
				OPX_PBC_LOOPBACK(opx_ep->domain, params->pbc_dlid, hfi1_type, 0),
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.hdr.qw_9B[0] |
				params->lrh_dlid | ((uint64_t) lrh_dws << 32),
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.hdr.qw_9B[1] |
				params->bth_subctxt_rx,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.hdr.qw_9B[2] | psn,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.hdr.qw_9B[3] | params->data,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_9B.hdr.qw_9B[4] | dt64 | op64 |
				pkt_niov | dput_opcode,
			params->client_key, 0ul /* params->rma_req */);

		union opx_hfisvc_rma_iov *replay_iovs = (union opx_hfisvc_rma_iov *) replay->payload;
		assert(cur_iov >= params->iovs_with_keys);
		int iov_index = cur_iov - params->iovs_with_keys;

		for (int i = 0; i < params->iovs_with_keys; i++) {
			replay_iovs[i] = params->hfisvc_iov[iov_index + i];
		}
	} else {
		/* 16B PBC is dws */
		const uint64_t pbc_dws = 16 +			       /* PIO SOP is 16 DWS/8 QWS*/
					 2 +			       /* RMA Key  (1 qw) */
					 (payload_qws_total << 1) + 2; /* ICRC/tail */

		/* 16B LRH is qws */
		const uint16_t lrh_qws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */

		opx_cacheline_store_qw(
			replay->scb.qws,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.qw0 |
				OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid |
				OPX_PBC_LOOPBACK(opx_ep->domain, params->pbc_dlid, hfi1_type, 0),
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[0] |
				((uint64_t) (params->lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B)
				 << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
				((uint64_t) lrh_qws << 20),
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[1] |
				((uint64_t) ((params->lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					     OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				(uint64_t) (params->bth_subctxt_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B),
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[2] |
				params->bth_subctxt_rx,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[3] | psn,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[4] |
				params->data,
			opx_ep->rx->tx.hfisvc_rma_rts[params->plane_idx].hfisvc_rma_rts_16B.hdr.qw_16B[5] | dt64 |
				op64 | pkt_niov | dput_opcode,
			/* | has_cq_data */
			params->client_key);

		replay->scb.qws[8] = 0ul; // params->rma_req

		union opx_hfisvc_rma_iov *replay_iovs = (union opx_hfisvc_rma_iov *) &replay->scb.qws[9];
		assert(cur_iov >= params->iovs_with_keys);
		int iov_index = cur_iov - params->iovs_with_keys;

		for (int i = 0; i < params->iovs_with_keys; i++) {
			replay_iovs[i] = params->hfisvc_iov[iov_index + i];
		}
	}

	fi_opx_reliability_service_do_replay(opx_ep, opx_ep->reli_service, replay);

	OPX_SHD_CTX_PIO_UNLOCK(OPX_IS_CTX_SHARING_ENABLED, opx_ep->tx);

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_MIXED_9B)) {
		fi_opx_reliability_service_replay_register_no_update(opx_ep->reli_service, flow, replay,
								     params->reliability, OPX_HFI1_WFR,
								     flow_control_mode, opx_ep, reservation_record);
	} else if (hfi1_type & OPX_HFI1_JKR) {
		fi_opx_reliability_service_replay_register_no_update(opx_ep->reli_service, flow, replay,
								     params->reliability, OPX_HFI1_JKR,
								     flow_control_mode, opx_ep, reservation_record);
	} else {
		fi_opx_reliability_service_replay_register_no_update(opx_ep->reli_service, flow, replay,
								     params->reliability, OPX_HFI1_CYR,
								     flow_control_mode, opx_ep, reservation_record);
	}

	if (cur_iov == niov) {
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_SEND_RTS");
		OPX_HFISVC_DEBUG_LOG("HFISVC completed rma_send_rts successfully for %s work item=%p!\n",
				     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
		params->work_elem.complete = true;
		return FI_SUCCESS;
	}

	params->iovs_with_keys = 0;
	params->cur_iov	       = cur_iov;

	OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_SEND_RTS");
	OPX_HFISVC_DEBUG_LOG("HFISVC EAGAIN (end) for rma_send_rts for %s work item=%p!\n",
			     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
	return -FI_EAGAIN;
}

int opx_hfisvc_rma_recv_rts(union fi_opx_hfi1_deferred_work *work)
{
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "HFISVC_RMA_RECV_RTS");
	struct opx_hfisvc_rma_rts_params *params    = &work->hfisvc_rma_rts;
	struct fi_opx_ep		 *opx_ep    = params->opx_ep;
	const uint64_t			  niov	    = params->niov;
	uint64_t			  cur_iov   = params->cur_iov;
	const uint8_t			  plane_idx = params->plane_idx;

	OPX_HFISVC_DEBUG_LOG("HFISVC Attempting rma_recv_rts for %s work item=%p cur_iov=%lu niov=%lu!\n",
			     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work, cur_iov, niov);

	// If the origin requested a GET, we will *write* the response back to them
	const bool write_to_origin = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_GET);
	const bool has_cq_data	   = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ);

	/* PUT/PUT_CQ carrying a real op (not the FI_NOOP-1 plain-PUT sentinel) => target-side atomic write */
	const bool is_atomic_write = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_PUT ||
				      params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ) &&
				     (params->fi_op_opcode != (FI_NOOP - 1));

	/* fetch: iov[0] = atomic target+operand, iov[1] = origin result buffer (pre-op value) */
	const bool is_fetch = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH);

	/* compare-fetch adds iov[2] = compare operand pulled into staging after the swap operand */
	const bool is_cmp_fetch = (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH);

	int rc = 0;

	/* FI_SUCCESS needs a committed doorbell; commands_pending tracks whether any command was queued */
	bool doorbell_committed = false;
	bool commands_pending	= false;

	/* doorbell-only retry: a prior pass queued every command but the committing doorbell failed */
	if (OFI_UNLIKELY(params->recv_needs_doorbell_only)) {
		int drc = (*opx_ep->domain->hfisvc.doorbell)(opx_ep->domain->hfisvc.ctxs[plane_idx].ctx);
		if (OFI_UNLIKELY(drc != 0)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "HFISVC doorbell retry failed rc=%d\n", drc);
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_RECV_RTS");
			return -FI_EAGAIN;
		}
		doorbell_committed		 = true;
		params->recv_needs_doorbell_only = 0;
		if (cur_iov == niov) {
			goto recv_rts_doorbell_committed;
		}
	}

	while (cur_iov < niov) {
		union opx_hfisvc_rma_iov *hfisvc_iov = &params->hfisvc_iov[cur_iov];
		// 1. Look up MR in MR cache
		uint64_t mr_key = hfisvc_iov->remote_auth_key;

		struct fi_opx_mr *opx_mr = NULL;
		HASH_FIND(hh, opx_ep->domain->mr_hashmap, &mr_key, sizeof(mr_key), opx_mr);

		if (opx_mr == NULL) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "lookup of key (%lu) failed; packet dropped\n",
				mr_key);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				     "===================================== RECV -- RMA HFISVC RTS - failed (end)\n");
			const uint64_t drop_flags = FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
			opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EIO, FI_EIO);

			++cur_iov;
			continue;
		}

		/* target-only MR starts OPEN_DEFERRED; enqueue its initial open so a GET/PUT cannot spin unopened */
		if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPEN_DEFERRED) {
			rc = opx_hfisvc_mr_lazy_open(opx_ep->domain, opx_mr);
			if (rc) {
				const uint64_t drop_flags =
					FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
				opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EIO,
						    FI_EIO);

				++cur_iov;
				continue;
			}
		}

		rc = opx_mr_hfisvc_check_state(opx_mr);

		if (OFI_UNLIKELY(rc)) {
			OPX_HFISVC_DEBUG_LOG("MR State check failed for opx_mr=%p, state=%d, rc=%d\n", opx_mr,
					     opx_mr->hfisvc.state, rc);
			/* a still-opening MR is transient: keep retrying so the open completion is never lost;
			 * only a closing MR (EFAULT) is a hard drop */
			if (rc == -FI_EFAULT) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC RMA MR key (%lu) unusable (closing); iov dropped\n", mr_key);

				const uint64_t drop_flags =
					FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
				opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EIO,
						    FI_EIO);

				++cur_iov;
				continue;
			}
			goto recv_rts_eagain;
		}

		assert(opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPENED);

		/* verify the MR's remote-access permission before issuing any RDMA op */
		const uint64_t required_access = write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE;
		if (OFI_UNLIKELY((opx_mr->attr.access & required_access) != required_access)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"HFISVC RMA %s denied: MR key (%lu) lacks %s (access=0x%lx); iov dropped\n",
				write_to_origin ? "GET" : "PUT", mr_key,
				write_to_origin ? "FI_REMOTE_READ" : "FI_REMOTE_WRITE", (uint64_t) opx_mr->attr.access);

			const uint64_t drop_flags = FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
			opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EACCES,
					    FI_EACCES);

			++cur_iov;
			continue;
		}

		/* Gate VA translation on FI_MR_VIRT_ADDR (not base_addr==0), and reject any MR whose base_addr
		 * disagrees so a mode mismatch can't skew va_bias. VIRT_ADDR: remote_offset is absolute VA; else
		 * MR-relative. */
		const uintptr_t mr_base_va	   = (uintptr_t) opx_mr->iov.iov_base;
		const uintptr_t mr_base_addr	   = (uintptr_t) opx_mr->base_addr;
		const bool	virt_addr	   = (opx_mr->domain->util_domain.mr_mode & FI_MR_VIRT_ADDR) != 0;
		const uintptr_t expected_base_addr = virt_addr ? 0 : mr_base_va;
		const bool	base_addr_valid	   = (mr_base_addr == expected_base_addr);
		const uintptr_t va_bias		   = virt_addr ? mr_base_va : 0;
		const uint64_t	mr_len		   = opx_mr->attr.mr_iov[0].iov_len;
		const uint64_t	mr_offset	   = (uint64_t) (hfisvc_iov->remote_offset - va_bias);
		if (OFI_UNLIKELY(!base_addr_valid || hfisvc_iov->remote_offset < va_bias || mr_offset > mr_len ||
				 hfisvc_iov->len > mr_len - mr_offset)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"HFISVC RMA %s out of MR bounds: mr_offset=%lu len=%lu mr_len=%lu (remote_offset=%lu); iov dropped\n",
				write_to_origin ? "GET" : "PUT", mr_offset, hfisvc_iov->len, mr_len,
				(uint64_t) hfisvc_iov->remote_offset);

			const uint64_t drop_flags = FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
			opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EACCES,
					    FI_EACCES);

			++cur_iov;
			continue;
		}

		if (is_atomic_write) {
			/* stage the operand into a local VA; the apply MR = op(MR, operand) runs on completion */
			if (OFI_UNLIKELY(hfisvc_iov->len > OPX_HFISVC_ATOMIC_STAGING_MAX)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic-write len %lu exceeds staging capacity; iov dropped\n",
					hfisvc_iov->len);
				opx_hfisvc_rma_drop(opx_ep, params, FI_RMA | FI_REMOTE_WRITE, hfisvc_iov->len,
						    has_cq_data, FI_EIO, FI_EIO);
				++cur_iov;
				continue;
			}

			uint8_t *staging = (uint8_t *) ofi_buf_alloc(opx_ep->hfisvc.atomic_staging_pool);
			struct opx_hfisvc_atomic_op *aop =
				(struct opx_hfisvc_atomic_op *) ofi_buf_alloc(opx_ep->hfisvc.atomic_op_pool);
			if (OFI_UNLIKELY(staging == NULL || aop == NULL)) {
				if (staging) {
					OPX_BUF_FREE(staging);
				}
				if (aop) {
					OPX_BUF_FREE(aop);
				}
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic staging/op pool exhausted; retrying RTS\n");
				if (OFI_UNLIKELY(++params->recv_eagain_attempts >=
						 OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC atomic pool stuck exhausted after %lu attempts; iov dropped\n",
						(uint64_t) params->recv_eagain_attempts);
					opx_hfisvc_rma_drop(opx_ep, params, FI_RMA | FI_REMOTE_WRITE, hfisvc_iov->len,
							    has_cq_data, FI_EIO, FI_EIO);
					++cur_iov;
					continue;
				}
				goto recv_rts_eagain;
			}

			aop->staging	  = staging;
			aop->target_vaddr = (uint8_t *) opx_mr->iov.iov_base + mr_offset;
			aop->apply_len	  = hfisvc_iov->len;
			aop->opx_mr	  = opx_mr;
			aop->cc		  = NULL;
			aop->context	  = params->context;
			aop->dt		  = (enum fi_datatype) params->fi_datatype_dt;
			aop->op		  = (enum fi_op) params->fi_op_opcode;
			aop->pending	  = 0;

			struct opx_hfisvc_xfer_completion *internal_completion =
				(struct opx_hfisvc_xfer_completion *) ofi_buf_alloc(opx_ep->hfisvc.completion_pool);
			if (OFI_UNLIKELY(internal_completion == NULL)) {
				OPX_BUF_FREE(staging);
				OPX_BUF_FREE(aop);
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC completion_pool exhausted; retrying RTS\n");
				if (OFI_UNLIKELY(++params->recv_eagain_attempts >=
						 OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC RMA completion_pool stuck exhausted after %lu attempts; iov dropped\n",
						(uint64_t) params->recv_eagain_attempts);
					opx_hfisvc_rma_drop(opx_ep, params, FI_RMA | FI_REMOTE_WRITE, hfisvc_iov->len,
							    has_cq_data, FI_EIO, FI_EIO);
					++cur_iov;
					continue;
				}
				goto recv_rts_eagain;
			}

			internal_completion->type	= OPX_HFISVC_XFER_TYPE_ATOMIC_WRITE;
			internal_completion->access_key = opx_mr->hfisvc.access_key;
			internal_completion->context	= params->context;
			internal_completion->cc		= NULL;
			internal_completion->opx_mr	= opx_mr;
			internal_completion->opx_ep	= opx_ep;
			internal_completion->len	= hfisvc_iov->len;
			internal_completion->flags	= 0;
			internal_completion->atomic_op	= aop;

			struct hfisvc_client_completion completion = {
				.flags		= OPX_HFISVC_CMPL_CQ,
				.cq.handle	= opx_ep->hfisvc.internal_completion_queues[plane_idx],
				.cq.app_context = (uint64_t) internal_completion,
			};

			rc = (*opx_ep->domain->hfisvc.cmd_rdma_read_va)(
				opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */,
				params->origin_lid, hfisvc_iov->origin_hfisvc_client_key, (uint32_t) hfisvc_iov->len,
				0ul /* immediate data */, hfisvc_iov->origin_hfisvc_access_key,
				hfisvc_iov->origin_offset, staging);

			if (OFI_UNLIKELY(rc)) {
				OPX_BUF_FREE(internal_completion);
				OPX_BUF_FREE(staging);
				OPX_BUF_FREE(aop);
				if (OFI_LIKELY(rc == -FI_EAGAIN)) {
					break;
				}
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"hfisvc cmd_rdma_read_va (atomic-write) failed rc=%d; iov dropped\n", rc);
				opx_hfisvc_rma_drop(opx_ep, params, FI_RMA | FI_REMOTE_WRITE, hfisvc_iov->len,
						    has_cq_data, FI_EIO, rc);
				++cur_iov;
				continue;
			}

			commands_pending = true;
			++cur_iov;
			continue;
		}

		if (is_fetch || is_cmp_fetch) {
			/* fetch/compare-fetch is a serial state machine keyed on iov[0]/iov[1] (+iov[2] compare);
			 * the RTS must carry every iov the state machine walks */
			const uint64_t fetch_niov = is_cmp_fetch ? 3ul : 2ul;
			if (OFI_UNLIKELY(niov != fetch_niov)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic-fetch malformed niov %lu (expected %lu); RTS dropped\n", niov,
					fetch_niov);
				opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ, params->data,
								   FI_EIO, FI_EIO);
				break;
			}
			union opx_hfisvc_rma_iov *op_iov  = &params->hfisvc_iov[0];
			union opx_hfisvc_rma_iov *res_iov = &params->hfisvc_iov[1];
			union opx_hfisvc_rma_iov *cmp_iov = is_cmp_fetch ? &params->hfisvc_iov[2] : NULL;

			/* compare-fetch stages [swap|compare] contiguously so the span is both operands; bound it to
			 * the staging buffer and require matching operand lens so the CSWAP/MSWAP apply re-indexes
			 * the compare half correctly */
			const uint64_t staging_span = is_cmp_fetch ? op_iov->len + cmp_iov->len : op_iov->len;
			if (OFI_UNLIKELY(staging_span > OPX_HFISVC_ATOMIC_STAGING_MAX ||
					 (is_cmp_fetch && cmp_iov->len != op_iov->len))) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic-fetch operand span %lu invalid (staging max %lu); iov dropped\n",
					staging_span, (uint64_t) OPX_HFISVC_ATOMIC_STAGING_MAX);
				opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ, params->data,
								   FI_EIO, FI_EIO);
				cur_iov += fetch_niov;
				continue;
			}

			/* step B reads res_iov->len bytes from the target MR at mr_offset; the per-iov gate only
			 * validated op_iov (iov[0]), so bound res_iov (iov[1]) against the same MR window */
			if (OFI_UNLIKELY(res_iov->len > mr_len - mr_offset)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic-fetch result len %lu out of MR bounds (mr_offset=%lu mr_len=%lu); iov dropped\n",
					res_iov->len, mr_offset, mr_len);
				opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ, params->data,
								   FI_EACCES, FI_EACCES);
				cur_iov += fetch_niov;
				continue;
			}

			uint8_t *staging = (uint8_t *) ofi_buf_alloc(opx_ep->hfisvc.atomic_staging_pool);
			struct opx_hfisvc_atomic_op *aop =
				(struct opx_hfisvc_atomic_op *) ofi_buf_alloc(opx_ep->hfisvc.atomic_op_pool);
			if (OFI_UNLIKELY(staging == NULL || aop == NULL)) {
				if (staging) {
					OPX_BUF_FREE(staging);
				}
				if (aop) {
					OPX_BUF_FREE(aop);
				}
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC atomic staging/op pool exhausted; retrying RTS\n");
				if (OFI_UNLIKELY(++params->recv_eagain_attempts >=
						 OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC atomic pool stuck exhausted after %lu attempts; iov dropped\n",
						(uint64_t) params->recv_eagain_attempts);
					opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ,
									   params->data, FI_EIO, FI_EIO);
					cur_iov += fetch_niov;
					continue;
				}
				goto recv_rts_eagain;
			}

			aop->staging	      = staging;
			aop->target_vaddr     = (uint8_t *) opx_mr->iov.iov_base + mr_offset;
			aop->apply_len	      = op_iov->len;
			aop->opx_mr	      = opx_mr;
			aop->cc		      = NULL;
			aop->context	      = params->context;
			aop->dt		      = (enum fi_datatype) params->fi_datatype_dt;
			aop->op		      = (enum fi_op) params->fi_op_opcode;
			aop->op_client_key    = op_iov->origin_hfisvc_client_key;
			aop->op_access_key    = op_iov->origin_hfisvc_access_key;
			aop->op_origin_offset = op_iov->origin_offset;
			aop->origin_lid	      = params->origin_lid;
			aop->op_len	      = (uint32_t) op_iov->len;
			aop->pending	      = 1;

			if (is_cmp_fetch) {
				aop->op2_client_key    = cmp_iov->origin_hfisvc_client_key;
				aop->op2_access_key    = cmp_iov->origin_hfisvc_access_key;
				aop->op2_origin_offset = cmp_iov->origin_offset;
				aop->op2_len	       = (uint32_t) cmp_iov->len;
			}

			struct opx_hfisvc_xfer_completion *internal_completion =
				(struct opx_hfisvc_xfer_completion *) ofi_buf_alloc(opx_ep->hfisvc.completion_pool);
			if (OFI_UNLIKELY(internal_completion == NULL)) {
				OPX_BUF_FREE(staging);
				OPX_BUF_FREE(aop);
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC completion_pool exhausted; retrying RTS\n");
				if (OFI_UNLIKELY(++params->recv_eagain_attempts >=
						 OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC RMA completion_pool stuck exhausted after %lu attempts; iov dropped\n",
						(uint64_t) params->recv_eagain_attempts);
					opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ,
									   params->data, FI_EIO, FI_EIO);
					cur_iov += fetch_niov;
					continue;
				}
				goto recv_rts_eagain;
			}

			internal_completion->type	= is_cmp_fetch ? OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH_COMPARE :
									 OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH;
			internal_completion->access_key = opx_mr->hfisvc.access_key;
			internal_completion->context	= params->context;
			internal_completion->cc		= NULL;
			internal_completion->opx_mr	= opx_mr;
			internal_completion->opx_ep	= opx_ep;
			internal_completion->len	= res_iov->len;
			internal_completion->flags	= 0;
			internal_completion->atomic_op	= aop;

			struct hfisvc_client_completion completion = {
				.flags		= OPX_HFISVC_CMPL_CQ,
				.cq.handle	= opx_ep->hfisvc.internal_completion_queues[plane_idx],
				.cq.app_context = (uint64_t) internal_completion,
			};

			/* step B first: push the pre-op MR value to the origin result buffer before applying the op */
			rc = opx_ep->domain->hfisvc.cmd_rdma_write(
				opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */,
				params->origin_lid, res_iov->origin_hfisvc_client_key, (uint32_t) res_iov->len,
				0ul /* immediate data */, res_iov->origin_hfisvc_access_key, res_iov->origin_offset,
				opx_mr->hfisvc.mr_handle, mr_offset);

			if (OFI_UNLIKELY(rc)) {
				OPX_BUF_FREE(internal_completion);
				OPX_BUF_FREE(staging);
				OPX_BUF_FREE(aop);
				if (OFI_LIKELY(rc == -FI_EAGAIN)) {
					break;
				}
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"hfisvc cmd_rdma_write (atomic-fetch result) failed rc=%d; iov dropped\n", rc);
				opx_hfisvc_rma_post_standalone_err(opx_ep, FI_RMA | FI_REMOTE_READ, params->data,
								   FI_EIO, rc);
				cur_iov += fetch_niov;
				continue;
			}

			commands_pending = true;
			cur_iov += fetch_niov;
			continue;
		}

		struct opx_hfisvc_xfer_completion *internal_completion;
		if (has_cq_data) {
			internal_completion =
				(struct opx_hfisvc_xfer_completion *) ofi_buf_alloc(opx_ep->hfisvc.completion_pool);
			if (OFI_UNLIKELY(internal_completion == NULL)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
					"HFISVC completion_pool exhausted; retrying RTS\n");
				if (OFI_UNLIKELY(++params->recv_eagain_attempts >=
						 OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"HFISVC RMA completion_pool stuck exhausted after %lu attempts; iov dropped\n",
						(uint64_t) params->recv_eagain_attempts);

					const uint64_t drop_flags =
						FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
					opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data,
							    FI_EIO, FI_EIO);

					++cur_iov;
					continue;
				}
				goto recv_rts_eagain;
			}
			/* target-side completion reuses the MR's key; poller must not free it */
			internal_completion->type =
				write_to_origin ? OPX_HFISVC_XFER_TYPE_RMA_WRITE : OPX_HFISVC_XFER_TYPE_RMA_READ;
			internal_completion->access_key = opx_mr->hfisvc.access_key;
			internal_completion->context	= params->context;
			internal_completion->cc		= NULL;
			internal_completion->opx_mr	= opx_mr;
			internal_completion->opx_ep	= opx_ep;
			internal_completion->len	= hfisvc_iov->len;
			internal_completion->flags	= 0;
		} else {
			internal_completion = NULL;
		}

		struct hfisvc_client_completion completion = {
			.flags		= OPX_HFISVC_CMPL_CQ,
			.cq.handle	= opx_ep->hfisvc.internal_completion_queues[plane_idx],
			.cq.app_context = (uint64_t) internal_completion,
		};

		if (write_to_origin) {
			rc = opx_ep->domain->hfisvc.cmd_rdma_write(
				opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */,
				params->origin_lid,
				hfisvc_iov->origin_hfisvc_client_key, // origin's client key
				hfisvc_iov->len, 0ul /* immediate data */, hfisvc_iov->origin_hfisvc_access_key,
				hfisvc_iov->origin_offset, opx_mr->hfisvc.mr_handle, mr_offset);
		} else {
			rc = opx_ep->domain->hfisvc.cmd_rdma_read(
				opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */,
				params->origin_lid,
				hfisvc_iov->origin_hfisvc_client_key, // origin's client key
				hfisvc_iov->len, 0ul /* immediate data */, hfisvc_iov->origin_hfisvc_access_key,
				hfisvc_iov->origin_offset, opx_mr->hfisvc.mr_handle, mr_offset);
		}

		if (OFI_UNLIKELY(rc)) {
			if (OFI_LIKELY(rc == -FI_EAGAIN)) {
				/* command queue full: free the pre-allocated completion and break to flush */
				if (internal_completion) {
					OPX_BUF_FREE(internal_completion);
				}
				break;
			}
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "hfisvc cmd_rdma_%s failed rc=%d; iov dropped\n",
				write_to_origin ? "write" : "read", rc);

			if (internal_completion) {
				OPX_BUF_FREE(internal_completion);
			}

			const uint64_t drop_flags = FI_RMA | (write_to_origin ? FI_REMOTE_READ : FI_REMOTE_WRITE);
			opx_hfisvc_rma_drop(opx_ep, params, drop_flags, hfisvc_iov->len, has_cq_data, FI_EIO, rc);

			++cur_iov;
			continue;
		}

		commands_pending = true;
		++cur_iov;
	}

	if (commands_pending) {
		// Ring the doorbell
		int rc = (*opx_ep->domain->hfisvc.doorbell)(opx_ep->domain->hfisvc.ctxs[plane_idx].ctx);
		if (OFI_UNLIKELY(rc != 0)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "HFISVC doorbell failed rc=%d\n", rc);
			/* commands already issued; set doorbell-only flag so the retry re-rings only */
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_RECV_RTS");
			params->cur_iov			 = cur_iov;
			params->recv_needs_doorbell_only = 1;
			return -FI_EAGAIN;
		}
		doorbell_committed = true;
	}

recv_rts_doorbell_committed:
	if ((doorbell_committed || !commands_pending) && cur_iov == niov) {
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_RECV_RTS");
		OPX_HFISVC_DEBUG_LOG("HFISVC completed rma_recv_rts successfully for %s work item=%p!\n",
				     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
		params->work_elem.complete = true;
		return FI_SUCCESS;
	}

recv_rts_eagain:
	params->cur_iov = cur_iov;

	OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "HFISVC_RMA_RECV_RTS");
	OPX_HFISVC_DEBUG_LOG("HFISVC EAGAIN (end) for rma_recv_rts for %s work item=%p!\n",
			     opx_hfi1_dput_opcode_to_string(params->dput_opcode), work);
	return -FI_EAGAIN;
}

void opx_hfisvc_rma_invoke_recv_rts(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
				    const union fi_opx_hfi1_packet_payload *const payload, const size_t payload_bytes,
				    const int lock_required, const enum ofi_reliability_kind reliability,
				    const enum opx_hfi1_type hfi1_type)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== RECV -- RMA HFISVC RTS (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "HFISVC_RMA_INVOKE_RCV_RTS");

	assert(hfi1_type == OPX_HFI1_JKR || hfi1_type == OPX_HFI1_CYR || hfi1_type == OPX_HFI1_WFR ||
	       hfi1_type == OPX_HFI1_MIXED_9B);
	assert(!lock_required);

	union fi_opx_hfi1_deferred_work *work =
		(union fi_opx_hfi1_deferred_work *) ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	if (OFI_UNLIKELY(work == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"work_pending_pool exhausted; HFISVC RMA RTS packet dropped\n");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== RECV -- RMA HFISVC RTS (end - alloc fail)\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}
	struct opx_hfisvc_rma_rts_params *params = &work->hfisvc_rma_rts;
	params->work_elem.slist_entry.next	 = NULL;
	params->work_elem.completion_action	 = NULL;
	params->work_elem.payload_copy		 = NULL;
	params->work_elem.complete		 = false;
	params->work_elem.work_type		 = OPX_WORK_TYPE_HFISVC;
	params->work_elem.work_fn		 = opx_hfisvc_rma_recv_rts;
	params->opx_ep				 = opx_ep;
	params->niov				 = hdr->rma_rts.niov;
	params->cur_iov				 = 0;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_MIXED_9B)) {
		params->origin_lid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
	} else {
		params->origin_lid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
	}
	params->data	    = hdr->match.ofi_data;
	params->dput_opcode = hdr->rma_rts.opcode;

	params->client_key  = (uint64_t) hdr->rma_rts.origin_hfisvc_client_key;
	params->reliability = reliability;
	params->hfi1_type   = hfi1_type;
	/* target replies on the primary plane; correctness comes from the origin's client_key + RTS SLID */
	params->plane_idx		 = OPX_PRIMARY_PLANE;
	params->recv_eagain_attempts	 = 0;
	params->recv_needs_doorbell_only = 0;

	const enum fi_datatype dt = (enum fi_datatype) hdr->rma_rts.dt;
	const enum fi_op       op = (enum fi_op) hdr->rma_rts.op;
	/* wire-controlled op/dt index the atomic dispatch table; plain RMA carries the FI_NOOP-1/FI_VOID-1 sentinel */
	if (OFI_UNLIKELY(!(op == FI_NOOP - 1 || op < OFI_ATOMIC_OP_LAST) ||
			 !(dt == FI_VOID - 1 || dt < OFI_DATATYPE_LAST))) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"HFISVC RMA RTS op/dt out of range (op=%u dt=%u); malformed packet dropped\n", (unsigned) op,
			(unsigned) dt);
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}
	params->fi_datatype_dt = dt;
	params->fi_op_opcode   = op;

	/* clamp wire-controlled niov so it cannot overflow the fixed-size hfisvc_iov[] */
	if (OFI_UNLIKELY(hdr->rma_rts.niov > OPX_MAX_RMA_HFISVC_IOVS)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"HFISVC RMA RTS niov (%u) exceeds max (%lu); malformed packet dropped\n",
			(unsigned) hdr->rma_rts.niov, (uint64_t) OPX_MAX_RMA_HFISVC_IOVS);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== RECV -- RMA HFISVC RTS (end - niov OOB)\n");
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}

	/* also bound niov by the delivered payload extent to close a payload over-read */
	const uint64_t delivered_iovs = (payload != NULL) ? (payload_bytes / sizeof(union opx_hfisvc_rma_iov)) : 0;
	if (OFI_UNLIKELY((uint64_t) hdr->rma_rts.niov > delivered_iovs)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"HFISVC RMA RTS niov (%u) exceeds delivered payload iov count (%lu, payload_bytes=%zu); malformed packet dropped\n",
			(unsigned) hdr->rma_rts.niov, delivered_iovs, payload_bytes);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== RECV -- RMA HFISVC RTS (end - niov > payload)\n");
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}

	const union opx_hfisvc_rma_iov *iov	    = payload->rma_rts.hfisvc_iovs;
	uint64_t			total_bytes = 0;
	for (int i = 0; i < params->niov; ++i) {
		params->hfisvc_iov[i] = iov[i];
		total_bytes += iov[i].len;
	}

	params->iovs_with_keys = 0;
	params->cc	       = NULL;
	params->lrh_dlid       = 0;
	params->pbc_dlid       = 0;

	if (params->dput_opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ) {
		params->context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
		if (OFI_UNLIKELY(params->context == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"ctx_pool exhausted; HFISVC RMA PUT_CQ RTS packet dropped\n");
			OPX_BUF_FREE(work);
			OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "HFISVC_RMA_INVOKE_RCV_RTS");
			return;
		}
		params->context->flags		      = FI_REMOTE_CQ_DATA | FI_RMA | FI_REMOTE_WRITE;
		params->context->data		      = hdr->match.ofi_data;
		params->context->buf		      = NULL;
		params->context->next		      = NULL;
		params->context->tag		      = 0;
		params->context->err_entry.err	      = 0;
		params->context->err_entry.op_context = NULL;
		params->context->len		      = total_bytes;
		params->context->byte_counter	      = total_bytes;

		slist_insert_tail((struct slist_entry *) params->context, opx_ep->rx->cq_pending_ptr);
	} else {
		params->context = NULL;
	}

	ssize_t rc = work->work_elem.work_fn(work);
	if (rc == FI_SUCCESS) {
		assert(work->work_elem.complete);
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}

	assert(rc == -FI_EAGAIN);
	if (work->work_elem.work_type == OPX_WORK_TYPE_LAST) {
		slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending_completion);
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_INVOKE_RCV_RTS");
		return;
	}

	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[work->work_elem.work_type]);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_INVOKE_RCV_RTS");

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== RECV -- RMA HFISVC RTS (end)\n");
}

#endif
