/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2025 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HFI1_PROGRESS_H_
#define _FI_PROV_OPX_HFI1_PROGRESS_H_

#ifndef FI_OPX_FABRIC_HFI1
#error "fabric selection #define error"
#endif

/*
 * Determines how frequently we update the HFI's Header Queue Register.
 * Writing to the register is expensive, so we don't want to do it too
 * often. However, the entries we've already processed will not be
 * available for reuse until we update the register.
 *
 * The offset is in terms of DWs. Each entry is 32 dws (128 bytes), and we
 * increment the hdrq offset by the entry size (0x20) after processing each
 * packet.
 *
 * Update masks are defined below to update the HQR every 32nd, 64th, 128th,
 * 512th, or 1024th entry. By default, the HQR will be updated every 64th entry.
 *
 * To use an update frequency other than the default, specify the desired mask at
 * compile time. For example, adding
 *
 * 	-DFI_OPX_HFI1_HDRQ_UPDATE_MASK=FI_OPX_HFI1_HDRQ_UPDATE_MASK_128
 *
 * to CPPFLAGS would cause the HQR to be updated every 128th entry.
 */
#include "rdma/opx/fi_opx_hfi1.h"
#include "uthash.h"
#include "fi_opx_reliability.h"
#include "rdma/opx/fi_opx_flight_recorder.h"
#include "rdma/opx/opx_tracer.h"

#define FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS (0x20ul)
#define FI_OPX_HFI1_HDRQ_INDEX_SHIFT	(5) /* index FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS entries */

#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_1024 (0x7FFFul)
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_512  (0x3FFFul)
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_256  (0x1FFFul)
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_128  (0xFFFul)
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_64	  (0x7FFul)
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK_32	  (0x3FFul)

#ifndef FI_OPX_HFI1_HDRQ_UPDATE_MASK
#define FI_OPX_HFI1_HDRQ_UPDATE_MASK FI_OPX_HFI1_HDRQ_UPDATE_MASK_32
#endif

OPX_COMPILE_TIME_ASSERT((FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_32) ||
				(FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_64) ||
				(FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_128) ||
				(FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_256) ||
				(FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_512) ||
				(FI_OPX_HFI1_HDRQ_UPDATE_MASK == FI_OPX_HFI1_HDRQ_UPDATE_MASK_1024),
			"FI_OPX_HFI1_HDRQ_UPDATE_MASK must be one of FI_OPX_HFI1_HDRQ_UPDATE_MASK_32, "
			"FI_OPX_HFI1_HDRQ_UPDATE_MASK_64, FI_OPX_HFI1_HDRQ_UPDATE_MASK_128, "
			"FI_OPX_HFI1_HDRQ_UPDATE_MASK_256, FI_OPX_HFI1_HDRQ_UPDATE_MASK_512, "
			"or FI_OPX_HFI1_HDRQ_UPDATE_MASK_1024");

unsigned fi_opx_hfi1_handle_poll_error(struct fi_opx_ep *opx_ep, volatile uint64_t *rhe_ptr, volatile uint32_t *rhf_ptr,
				       const uint32_t rhf_msb, const uint32_t rhf_lsb, const uint64_t rhf_seq,
				       const uint64_t hdrq_offset, const uint64_t rhf_rcvd,
				       const union opx_hfi1_packet_hdr *const hdr, const enum opx_hfi1_type hfi1_type);

__OPX_FORCE_INLINE__
void fi_opx_hfi1_update_hdrq_head_register(struct fi_opx_ep *opx_ep, const uint64_t hdrq_offset)
{
	if (OFI_UNLIKELY((hdrq_offset & FI_OPX_HFI1_HDRQ_UPDATE_MASK) == FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS)) {
		OPX_HFI1_BAR_STORE(opx_ep->rx->hdrq.head_register,
				   (const uint64_t)(hdrq_offset - FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS));
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "================== > Set HFI head register\n");
	}
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_handle_ud_eager_packet(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
					const uint64_t rhf, const opx_lid_t slid, const opx_lid_t dlid,
					const enum opx_hfi1_type hfi1_type)
{
	/* "eager" packet - has payload */
	const uint32_t				     egrbfr_index  = OPX_RHF_EGR_INDEX(rhf, hfi1_type);
	const uint32_t				     egrbfr_offset = OPX_RHF_EGR_OFFSET(rhf, hfi1_type);
	const uint8_t *const __attribute__((unused)) payload =
		(uint8_t *) ((uintptr_t) opx_ep->rx->egrq.base_addr +
			     (uintptr_t) egrbfr_index * (uintptr_t) opx_ep->rx->egrq.elemsz + egrbfr_offset * 64);

	assert(payload != NULL);

	/* currently no eager UD packets are defined */
	fprintf(stderr, "%s:%s():%d bad ud eager packet; abort.\n", __FILE__, __func__, __LINE__);
	abort();

	const uint32_t last_egrbfr_index = opx_ep->rx->egrq.last_egrbfr_index;
	if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
		OPX_HFI1_BAR_STORE(opx_ep->rx->egrq.head_register, (const uint64_t) last_egrbfr_index);
		opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
	}
}

static void fi_opx_hfi1_handle_ud_ping(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
				       const opx_lid_t slid)
{
	const uint64_t				    psn_start = hdr->service.psn_start;
	const uint64_t				    psn_count = hdr->service.psn_count;
	struct fi_opx_reliability_service	   *service   = opx_ep->reli_service;
	struct fi_opx_pending_rx_reliability_op_key lookup_key;
	lookup_key.flow_key.qw_prefix			 = hdr->service.key;
	lookup_key.flow_key.dw_suffix			 = hdr->service.key_dw_suffix;
	lookup_key.psn_start				 = psn_start;
	lookup_key.unused				 = 0;
	struct fi_opx_pending_rx_reliability_op *ping_op = NULL;

	HASH_FIND(hh, service->pending_rx_reliability_ops_hashmap, &lookup_key, sizeof(lookup_key), ping_op);

	/*
	 * Coalesce a ping op if we find it, otherwise allocate a ping op and add it as a new item to the hash of
	 * pending ops
	 */
	if (ping_op) {
		ping_op->psn_count_coalesce = MAX(ping_op->psn_count_coalesce, psn_count);
	} else {
		// Send the first ACK/NACK right away, it might be an RMA fence event
		fi_opx_hfi1_rx_reliability_ping(&opx_ep->ep_fid, service, &lookup_key.flow_key, psn_count,
						hdr->service.psn_start, slid,
						hdr->service.origin_reliability_subctxt_rx);

		ping_op = ofi_buf_alloc(service->rx.pending_rx_reliability_pool);
		if (ping_op) {
			ping_op->ud_opcode	    = hdr->ud.opcode & FI_OPX_HFI_UD_OPCODE_MASK;
			ping_op->slid		    = slid;
			ping_op->subctxt_rx	    = hdr->service.origin_reliability_subctxt_rx;
			ping_op->key.flow_key	    = lookup_key.flow_key;
			ping_op->psn_count	    = psn_count;
			ping_op->psn_count_coalesce = 0;
			ping_op->key.psn_start	    = hdr->service.psn_start;
			ping_op->key.unused	    = 0;

			HASH_ADD(hh, service->pending_rx_reliability_ops_hashmap, key, sizeof(ping_op->key), ping_op);
		}
	}
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_handle_ud_ack(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr)
{
	const union fi_opx_reliability_service_flow_key key	  = {.qw_prefix = hdr->service.key,
								     .dw_suffix = hdr->service.key_dw_suffix};
	const uint64_t					psn_count = hdr->service.psn_count;
	const uint64_t					psn_start = hdr->service.psn_start;

	fi_opx_hfi1_rx_reliability_ack(&opx_ep->ep_fid, opx_ep->reli_service, &key, psn_count, psn_start);
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_handle_ud_nack(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr)
{
	const union fi_opx_reliability_service_flow_key key	  = {.qw_prefix = hdr->service.key,
								     .dw_suffix = hdr->service.key_dw_suffix};
	const uint64_t					psn_count = hdr->service.psn_count;
	const uint64_t					psn_start = hdr->service.psn_start;

	fi_opx_hfi1_rx_reliability_nack(&opx_ep->ep_fid, opx_ep->reli_service, &key, psn_count, psn_start);
}

__OPX_FORCE_INLINE__
unsigned fi_opx_hfi1_handle_ud_packet(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
				      const uint64_t rhf_seq, const uint64_t hdrq_offset, const uint64_t rhf,
				      const opx_lid_t slid, const opx_lid_t dlid, const enum opx_hfi1_type hfi1_type)
{
	/* "header only" packet - no payload */
	if (OFI_LIKELY(!OPX_RHF_IS_USE_EGR_BUF(rhf, hfi1_type))) {
		const uint8_t ud_opcode = FI_OPX_HFI_UD_OPCODE_MASK & hdr->ud.opcode;
		switch (ud_opcode) {
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_PING:
			fi_opx_hfi1_handle_ud_ping(opx_ep, hdr, slid);
			break;
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_ACK:
			fi_opx_hfi1_handle_ud_ack(opx_ep, hdr);
			break;
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_NACK:
			fi_opx_hfi1_handle_ud_nack(opx_ep, hdr);
			break;
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_NOOP:
			break;
#ifdef OPX_DAOS
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_RESYNCH:
			fi_opx_hfi1_rx_reliability_resynch(&opx_ep->ep_fid, opx_ep->reliability->state.service,
							   hdr->service.origin_reliability_subctxt_rx, hdr,
							   OPX_IS_CTX_SHARING_ENABLED);
			break;
		case FI_OPX_HFI_UD_OPCODE_RELIABILITY_RESYNCH_ACK:
			fi_opx_hfi1_rx_reliability_ack_resynch(&opx_ep->ep_fid, opx_ep->reliability->state.service,
							       hdr);
			break;
#endif
		default:
			fprintf(stderr, "%s:%s():%d bad ud header packet; abort.\n", __FILE__, __func__, __LINE__);
			abort();
		}
	} else {
		fi_opx_hfi1_handle_ud_eager_packet(opx_ep, hdr, rhf, slid, dlid, hfi1_type);
	}

	opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INCREMENT(rhf_seq, hfi1_type);
	opx_ep->rx->state.hdrq.head    = hdrq_offset + FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS;

	fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset);

	return 1; /* one packet was processed */
}

__OPX_FORCE_INLINE__
unsigned fi_opx_hfi1_error_inject(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
				  const uint64_t rhf_seq, const uint64_t hdrq_offset, const uint64_t rhf)
{
#ifdef OPX_RELIABILITY_TEST
	/*
	 * Error injection .. purposefully drop packet
	 */
	if (OFI_UNLIKELY(FI_OPX_RELIABILITY_RX_DROP_PACKET(opx_ep->reli_service, hdr))) {
		opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INCREMENT(rhf_seq, OPX_HFI1_TYPE);
		opx_ep->rx->state.hdrq.head    = hdrq_offset + FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS;

		if (OPX_RHF_IS_USE_EGR_BUF(rhf, OPX_HFI1_TYPE)) { /* eager */
			const uint32_t egrbfr_index	 = OPX_RHF_EGR_INDEX(rhf, OPX_HFI1_TYPE);
			const uint32_t last_egrbfr_index = opx_ep->rx->egrq.last_egrbfr_index;
			if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
				OPX_HFI1_BAR_STORE(opx_ep->rx->egrq.head_register, (const uint64_t) last_egrbfr_index);
				opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
			}
		}

		fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset);

		return 0;
	}
#endif
	return -1;
}

__OPX_FORCE_INLINE__
unsigned fi_opx_hfi1_handle_reliability(struct fi_opx_ep *opx_ep, const uint8_t opcode,
					const union opx_hfi1_packet_hdr *const hdr, const uint64_t rhf_seq,
					const uint64_t hdrq_offset, const uint64_t rhf, const opx_lid_t slid,
					const enum opx_hfi1_type hfi1_type, const bool ctx_sharing)
{
	/*
	 * Check for 'reliability' exceptions
	 */
	const uint64_t pkt_origin_rx = FI_OPX_HFI1_PACKET_ORIGIN_RX(hdr);
	const uint64_t psn	     = FI_OPX_HFI1_PACKET_PSN(hdr);

	struct fi_opx_reliability_rx_flow *flow;
	enum opx_reliability_status	   rc =
		fi_opx_reliability_rx_check(opx_ep->reli_service, slid, pkt_origin_rx, psn, &flow);

	if (OFI_LIKELY(rc == OPX_RELIABILITY_SUCCESS)) {
		return -1;
	}

	if (OFI_UNLIKELY(rc == OPX_RELIABILITY_FLOW_ERROR)) {
		return 0;
	}

	assert(rc == OPX_RELIABILITY_EXCEPTION);
	assert(flow != NULL);

	if (!OPX_RHF_IS_USE_EGR_BUF(rhf, hfi1_type)) {
		/* no payload */
		fi_opx_reliability_rx_exception(opx_ep->reli_service, flow, slid, pkt_origin_rx, psn, &opx_ep->ep_fid,
						hdr, NULL, hfi1_type, opcode, ctx_sharing);

	} else {
		/* has payload */
		const uint32_t	     egrbfr_index  = OPX_RHF_EGR_INDEX(rhf, hfi1_type);
		const uint32_t	     egrbfr_offset = OPX_RHF_EGR_OFFSET(rhf, hfi1_type);
		const uint8_t *const payload =
			(uint8_t *) ((uintptr_t) opx_ep->rx->egrq.base_addr +
				     (uintptr_t) egrbfr_index * (uintptr_t) opx_ep->rx->egrq.elemsz +
				     egrbfr_offset * 64);

		assert(payload != NULL);

		fi_opx_reliability_rx_exception(opx_ep->reli_service, flow, slid, pkt_origin_rx, psn, &opx_ep->ep_fid,
						hdr, payload, hfi1_type, opcode, ctx_sharing);

		const uint32_t last_egrbfr_index = opx_ep->rx->egrq.last_egrbfr_index;
		if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
			OPX_HFI1_BAR_STORE(opx_ep->rx->egrq.head_register, (const uint64_t) last_egrbfr_index);
			opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
		}
	}

	opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INCREMENT(rhf_seq, hfi1_type);
	opx_ep->rx->state.hdrq.head    = hdrq_offset + FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS;

	fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset);

	return 1; /* one packet was processed - even though it was a "reliability event" packet */
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_handle_packet(struct fi_opx_ep *opx_ep, const uint8_t opcode,
			       const union opx_hfi1_packet_hdr *const hdr, const uint64_t rhf_seq,
			       const uint64_t hdrq_offset, const int lock_required,
			       const enum ofi_reliability_kind reliability, const uint64_t rhf,
			       const enum opx_hfi1_type hfi1_type, const opx_lid_t slid, const bool ctx_sharing)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "================ received a packet from the fabric\n");

	if (!OPX_RHF_IS_USE_EGR_BUF(rhf, hfi1_type)) {
		if (OFI_LIKELY(opcode == FI_OPX_HFI_BTH_OPCODE_RZV_DATA ||
			       FI_OPX_HFI_BTH_OPCODE_WITHOUT_CQ(opcode) == FI_OPX_HFI_BTH_OPCODE_TAG_INJECT)) {
			/* "header only" packet - no payload */
			fi_opx_ep_rx_process_header(&opx_ep->ep_fid, hdr, NULL, 0, FI_TAGGED, opcode,
						    OPX_INTRANODE_FALSE, lock_required, reliability, hfi1_type, slid);
		} else if (FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)) {
			/* all other "tag" packets */
			fi_opx_ep_rx_process_header_tag(&opx_ep->ep_fid, hdr, NULL, 0, opcode, OPX_INTRANODE_FALSE,
							lock_required, reliability, hfi1_type, slid);

		} else {
			fi_opx_ep_rx_process_header_msg(&opx_ep->ep_fid, hdr, NULL, 0, opcode, OPX_INTRANODE_FALSE,
							lock_required, reliability, hfi1_type, slid);
		}
	} else {
		/* "eager" packet - has payload */
		const uint32_t	     egrbfr_index  = OPX_RHF_EGR_INDEX(rhf, hfi1_type);
		const uint32_t	     egrbfr_offset = OPX_RHF_EGR_OFFSET(rhf, hfi1_type);
		const uint8_t *const payload =
			(uint8_t *) ((uintptr_t) opx_ep->rx->egrq.base_addr +
				     (uintptr_t) egrbfr_index * (uintptr_t) opx_ep->rx->egrq.elemsz +
				     egrbfr_offset * 64);

		assert(payload != NULL);

		size_t payload_bytes_to_copy = opx_hfi1_packet_hdr_payload_bytes(hdr, hfi1_type);

		if (OFI_LIKELY(FI_OPX_HFI_BTH_OPCODE_WITHOUT_CQ(opcode) == FI_OPX_HFI_BTH_OPCODE_TAG_EAGER)) {
			fi_opx_ep_rx_process_header(&opx_ep->ep_fid, hdr,
						    (const union fi_opx_hfi1_packet_payload *const) payload,
						    payload_bytes_to_copy, FI_TAGGED, opcode, OPX_INTRANODE_FALSE,
						    lock_required, reliability, hfi1_type, slid);
		} else if (FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)) { /* all other "tag" packets */
			fi_opx_ep_rx_process_header_tag(&opx_ep->ep_fid, hdr, payload, payload_bytes_to_copy, opcode,
							OPX_INTRANODE_FALSE, lock_required, reliability, hfi1_type,
							slid);

		} else {
			fi_opx_ep_rx_process_header_msg(&opx_ep->ep_fid, hdr, payload, payload_bytes_to_copy, opcode,
							OPX_INTRANODE_FALSE, lock_required, reliability, hfi1_type,
							slid);
		}
		const uint32_t last_egrbfr_index = opx_ep->rx->egrq.last_egrbfr_index;
		if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
			OPX_HFI1_BAR_STORE(opx_ep->rx->egrq.head_register, (const uint64_t) last_egrbfr_index);
			opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
		}

		FLIGHT_RECORDER_PACKET_HDR(opx_ep->fr, FR_EVENT_HFI1_POLL_ONCE, hdr);
	}

	opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INCREMENT(rhf_seq, hfi1_type);
	opx_ep->rx->state.hdrq.head    = hdrq_offset + FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS;

	fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset);

	/* Send preemptive ACKs at regular intervals to improve performance (when enabled).
	 *
	 * NOTE: While a PSN of 0 would technically be on a rate boundary, we do not do
	 *       preemptive ACKs for PSN 0, or the previous N packets leading up to MAX_PSN
	 *       before the PSN wrap. The main reason for this is performance. A PSN of 0
	 *       would require a separate if-block with two separate calls to send_pre_acks
	 *       in order to avoid a rollover range. The added cost of executing that check
	 *       2^24 times for the 1 time we'd see that edge case isn't worth the payoff.
	 */

	uint32_t psn = FI_OPX_HFI1_PACKET_PSN(hdr);
	if (!(psn & opx_ep->reli_service->preemptive_ack_rate_mask) && psn) {
		fi_opx_hfi1_rx_reliability_send_pre_acks(
			&opx_ep->ep_fid, opx_ep->reli_service->lid, opx_ep->reli_service->subctxt_rx,
			psn - opx_ep->reli_service->preemptive_ack_rate + 1, /* psn_start */
			opx_ep->reli_service->preemptive_ack_rate,	     /* psn_count */
			hdr, slid, hfi1_type, ctx_sharing);

	} else if (hdr->bth.opcode == FI_OPX_HFI_BTH_OPCODE_RZV_DATA && (ntohl(hdr->bth.psn) & 0x80000000)) {
		/* Send preemptive ACKs on Rendezvous Data packets when
		 * the high bit of the PSN - the Acknowledge Request bit - is set
		 */
		uint32_t psn_count = MAX(MIN(opx_ep->reli_service->preemptive_ack_rate, psn), 1);
		assert(psn >= psn_count - 1);

		fi_opx_hfi1_rx_reliability_send_pre_acks(&opx_ep->ep_fid, opx_ep->reli_service->lid,
							 opx_ep->reli_service->subctxt_rx,
							 psn - psn_count + 1, /* psn_start */
							 psn_count,	      /* psn_count */
							 hdr, slid, hfi1_type, ctx_sharing);
	}
}

/*
 * ============================================================================
 *                      THIS IS THE HFI POLL FUNCTION
 * ============================================================================
 */
__OPX_FORCE_INLINE__
unsigned fi_opx_hfi1_poll_once(struct fid_ep *ep, const int lock_required, const enum ofi_reliability_kind reliability,
			       const uint64_t hdrq_mask, const enum opx_hfi1_type hfi1_type, const bool ctx_sharing)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const uint64_t	  local_hdrq_mask =
		   (hdrq_mask == FI_OPX_HDRQ_MASK_RUNTIME) ? opx_ep->hfi->info.rxe.hdrq.rx_poll_mask : hdrq_mask;
	const uint64_t hdrq_offset = opx_ep->rx->state.hdrq.head & local_hdrq_mask;

	assert(local_hdrq_mask % FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS == 0);
	volatile uint32_t *rhf_ptr = opx_ep->rx->hdrq.rhf_base + hdrq_offset;

	const uint64_t rhf_rcvd = *((volatile uint64_t *) rhf_ptr);
	opx_lid_t      slid;
	opx_lid_t      dlid;

	const uint64_t rhf_seq = opx_ep->rx->state.hdrq.rhf_seq;
	/* The software must look at the RHF.RcvSeq.
	 * If it detects the next sequence number in the entry, the new header
	 * was written into memory.  Otherwise, do not process RHF - no packet.
	 */
	if (OPX_RHF_SEQ_MATCH(rhf_seq, rhf_rcvd, hfi1_type)) {
		const uint32_t rhf_msb = rhf_rcvd >> 32;

		const uint64_t hdrq_offset_dws = (rhf_msb >> 12) & 0x01FFu;

		uint32_t *pkt;
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			assert(hdrq_offset_dws); /* need padding before this header */
			pkt = (uint32_t *) rhf_ptr - FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS + 2 /* rhf field size in dw */
			      - 2 /* sizeof(uint64_t) in dw, offset back to align
				     for the 9B padding in the header union */
			      + hdrq_offset_dws;
		} else {
			assert(((union opx_jkr_rhf) rhf_rcvd).L2Type == 0x2);
			pkt = (uint32_t *) rhf_ptr - FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS + 2 /* rhf field size in dw */
			      + hdrq_offset_dws;
			/* Assert we got full expected kdeth split header.
			 * In the future, we may handle this so this is
			 *  not part of OPX_RHF_CHECK_HEADER */
			assert(!(((union opx_jkr_rhf) rhf_rcvd).KHdrLenErr));
		}

		const union opx_hfi1_packet_hdr *const hdr    = (union opx_hfi1_packet_hdr *) pkt;
		const uint8_t			       opcode = hdr->bth.opcode;

		/* If there's an RHF/RHE error or a bad header detected,
		   handle the error and return */
		if (OPX_RHF_CHECK_HEADER(rhf_rcvd, hdr, hfi1_type)) {
			const uint32_t	   rhf_lsb = rhf_rcvd & 0xFFFFFFFF;
			volatile uint64_t *rhe_ptr = opx_ep->rx->hdrq.rhe_base;
			return fi_opx_hfi1_handle_poll_error(opx_ep, rhe_ptr, rhf_ptr, rhf_msb, rhf_lsb, rhf_seq,
							     hdrq_offset, rhf_rcvd, hdr, hfi1_type);
		}

		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			slid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
			dlid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.dlid);
		} else {
			slid = (opx_lid_t) __le24_to_cpu((hdr->lrh_16B.slid20 << 20) | (hdr->lrh_16B.slid));
			dlid = (opx_lid_t) __le24_to_cpu(((hdr->lrh_16B.dlid20 << 20) | (hdr->lrh_16B.dlid)));
		}

		if (OFI_UNLIKELY(opcode == FI_OPX_HFI_BTH_OPCODE_UD)) {
			assert(reliability == OFI_RELIABILITY_KIND_ONLOAD);
			/*
			 * process "unreliable datagram" packets first - before all the
			 * software reliability protocol checks.
			 */
			return fi_opx_hfi1_handle_ud_packet(opx_ep, hdr, rhf_seq, hdrq_offset, rhf_rcvd, slid, dlid,
							    hfi1_type);
		}

		/*
		 * check for software reliability events
		 */
		/* This error inject call will compile out in optimized builds */
		unsigned rc = fi_opx_hfi1_error_inject(opx_ep, hdr, rhf_seq, hdrq_offset, rhf_rcvd);
		if (OFI_UNLIKELY(rc != -1)) {
			return rc;
		}

		rc = fi_opx_hfi1_handle_reliability(opx_ep, opcode, hdr, rhf_seq, hdrq_offset, rhf_rcvd, slid,
						    hfi1_type, ctx_sharing);
		if (OFI_UNLIKELY(rc != -1)) {
			return rc;
		}

		fi_opx_hfi1_handle_packet(opx_ep, opcode, hdr, rhf_seq, hdrq_offset, lock_required, reliability,
					  rhf_rcvd, hfi1_type, slid, ctx_sharing);
		return 1; /* one packet was processed */
	}
	return 0;
}

/*
 * ============================================================================
 *                      THIS IS THE SHM POLL FUNCTION
 * ============================================================================
 */
static inline void fi_opx_shm_poll_many(struct fid_ep *ep, const int lock_required, const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep	  *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	uint64_t		   pos;
	struct opx_shm_packet	  *packet = opx_shm_rx_next(&opx_ep->rx->shm, &pos);
	union opx_hfi1_packet_hdr *hdr	  = (packet) ? (union opx_hfi1_packet_hdr *) packet->data : NULL;
	opx_lid_t		   slid;

	while (hdr != NULL) {
		const uint8_t opcode = hdr->bth.opcode;

#ifndef NDEBUG
		opx_lid_t dlid;
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			dlid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.dlid);
		} else {
			dlid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.dlid20 << 20 | hdr->lrh_16B.dlid);
		}

		assert(dlid == opx_ep->rx->self.lid);
#endif

		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			slid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
		} else {
			slid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
		}

#ifndef OPX_DAOS
		assert((hdr->bth.subctxt_rx & OPX_BTH_SUBCTXT_RX_MASK) == opx_ep->rx->self.hfi1_subctxt_rx);
#else
		uint32_t origin_reliability_rx;

		/* DAOS HFI Rank Support: */
		if (opx_ep->daos_info.hfi_rank_enabled) {
			/* DAOS Persistent Address Support:
			 * No Context Resource Management Framework is supported by OPX to
			 * enable acquiring a context with attributes that exactly match the
			 * specified source address (fi_addr).  Therefore, all source addresses
			 * are treated as an ‘opaque’ ID, and only the essential data required
			 * to create a context that at least maps to the same HFI and HFI port is
			 * extracted from the specified source address.
			 *
			 * DAOS assigns a unique internal-proprietary rank value to each EP, that
			 * is used to drive all communication between EPs.  DAOS often stops/restarts
			 * EPs while reusing the same unique internal-proprietary rank value as part
			 * of the Persistent Address Support.  This causes the fi_addr associated
			 * with a rank to change.   The stl.bth.subctxt_rx & hdr->stl.lrh.dlid fields of the
			 * inbound packet header are set from fields in the fi_addr, which sometimes
			 * change due to support for Persistent Addressing.  The only reliable field
			 * in the fi_addr is the hfi1_unit.
			 */

			/* origin_reliability_rx is HFI rank instead of HFI rx */
			origin_reliability_rx = packet->origin_rank;

			/* Settings used for possible response patcket(s) */
			opx_ep->daos_info.rank	    = packet->origin_rank;
			opx_ep->daos_info.rank_inst = packet->origin_rank_inst;

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				     "================ SHM received a packet from %u Segment (%s)\n",
				     opx_ep->daos_info.rank, opx_ep->rx->shm.segment_key);
		} else {
			origin_reliability_rx = hdr->service.origin_reliability_subctxt_rx;
		}

		if (opcode == FI_OPX_HFI_BTH_OPCODE_UD) {
			const uint8_t ud_opcode = hdr->ud.opcode;

			if (ud_opcode == FI_OPX_HFI_UD_OPCODE_RELIABILITY_RESYNCH) {
				fi_opx_hfi1_rx_reliability_resynch(&opx_ep->ep_fid, opx_ep->reliability->state.service,
								   origin_reliability_rx, hdr,
								   OPX_IS_CTX_SHARING_ENABLED);

			} else if (ud_opcode == FI_OPX_HFI_UD_OPCODE_RELIABILITY_RESYNCH_ACK) {
				fi_opx_hfi1_rx_reliability_ack_resynch(&opx_ep->ep_fid,
								       opx_ep->reliability->state.service, hdr);

			} else {
				fprintf(stderr, "%s:%s():%d bad ud opcode (%u); abort.\n", __FILE__, __func__, __LINE__,
					ud_opcode);
				abort();
			}

		} else
#endif
		if (FI_OPX_HFI_BTH_OPCODE_WITHOUT_CQ(opcode) == FI_OPX_HFI_BTH_OPCODE_TAG_INJECT) {
			fi_opx_ep_rx_process_header(ep, hdr, NULL, 0, FI_TAGGED, opcode, OPX_INTRANODE_TRUE,
						    lock_required, OFI_RELIABILITY_KIND_NONE, hfi1_type, slid);

		} else {
			const uint8_t *const payload = (uint8_t *) (hdr + 1);

			/* reported in LRH as the number of 4-byte words in the packet; header + payload + icrc */
			uint16_t lrh_pktlen_le;
			size_t	 total_bytes_to_copy;
			size_t	 payload_bytes_to_copy;

			if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
				lrh_pktlen_le	    = ntohs(hdr->lrh_9B.pktlen);
				total_bytes_to_copy = (lrh_pktlen_le - 1) * 4; /* do not copy the trailing icrc */
				payload_bytes_to_copy =
					total_bytes_to_copy - sizeof(struct fi_opx_hfi1_stl_packet_hdr_9B);
			} else {
				lrh_pktlen_le = hdr->lrh_16B.pktlen;
				total_bytes_to_copy =
					(lrh_pktlen_le - 1) * 8; /* do not copy the trailing tail/icrc QW*/
				payload_bytes_to_copy =
					total_bytes_to_copy - sizeof(struct fi_opx_hfi1_stl_packet_hdr_16B);
			}

			if (FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)) {
				fi_opx_ep_rx_process_header_tag(ep, hdr, payload, payload_bytes_to_copy, opcode,
								OPX_INTRANODE_TRUE, lock_required,
								OFI_RELIABILITY_KIND_NONE, hfi1_type, slid);

			} else {
				fi_opx_ep_rx_process_header_msg(ep, hdr, payload, payload_bytes_to_copy, opcode,
								OPX_INTRANODE_TRUE, lock_required,
								OFI_RELIABILITY_KIND_NONE, hfi1_type, slid);
			}
		}

		opx_shm_rx_advance(&opx_ep->rx->shm, (void *) hdr, pos);
		packet = opx_shm_rx_next(&opx_ep->rx->shm, &pos);
		hdr    = (packet) ? (union opx_hfi1_packet_hdr *) packet->data : NULL;
	}
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_poll_sdma_completion(struct fi_opx_ep *opx_ep)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "===================================== SDMA POLL BEGIN\n");
	struct fi_opx_hfi1_context *hfi	       = opx_ep->hfi;
	uint16_t		    queue_size = hfi->info.sdma.queue_size;

	FI_OPX_DEBUG_COUNTERS_DECLARE_TMP(sdma_end_ns);
	OPX_COUNTERS_TIME_NS(sdma_end_ns, &opx_ep->debug_counters);

	while (hfi->info.sdma.available_counter < queue_size) {
		volatile struct hfi1_sdma_comp_entry *entry =
			&hfi->info.sdma.completion_queue[hfi->info.sdma.done_index];
		if (entry->status == QUEUED) {
			break;
		}

		// Update the status/errcode of the work entry who was using this index
		assert(hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]);
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]->status  = entry->status;
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]->errcode = entry->errcode;
		OPX_COUNTERS_STORE_VAL(hfi->info.sdma.queued_entries[hfi->info.sdma.done_index]->end_time_ns,
				       sdma_end_ns);
		hfi->info.sdma.queued_entries[hfi->info.sdma.done_index] = NULL;

		OPX_TRACER_TRACE_SDMA(OPX_TRACER_END_SUCCESS, "SDMA_COMPLETE_%hu", hfi->info.sdma.done_index);

		assert(entry->status == COMPLETE || entry->status == FREE ||
		       (entry->status == ERROR &&
			(entry->errcode != ECOMM || entry->errcode != -ENOENT))); // If it is a network error, retry
		++hfi->info.sdma.available_counter;
		hfi->info.sdma.done_index = (hfi->info.sdma.done_index + 1) % (queue_size);
		if (hfi->info.sdma.done_index == hfi->info.sdma.fill_index) {
			assert(hfi->info.sdma.available_counter == queue_size);
		}
	}
	assert(hfi->info.sdma.available_counter >= opx_ep->tx->sdma_request_queue.slots_avail);
	opx_ep->tx->sdma_request_queue.slots_avail = hfi->info.sdma.available_counter;
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "===================================== SDMA POLL COMPLETE\n");
}

__OPX_FORCE_INLINE__
int opx_is_rhf_empty(struct fi_opx_ep *opx_ep, const uint64_t hdrq_mask, const enum opx_hfi1_type hfi1_type)
{
	const uint64_t local_hdrq_mask =
		(hdrq_mask == FI_OPX_HDRQ_MASK_RUNTIME) ? opx_ep->hfi->info.rxe.hdrq.rx_poll_mask : hdrq_mask;
	const uint64_t	   hdrq_offset = opx_ep->rx->state.hdrq.head & local_hdrq_mask;
	volatile uint32_t *rhf_ptr     = opx_ep->rx->hdrq.rhf_base + hdrq_offset;
	const uint64_t	   rhf_rcvd    = *((volatile uint64_t *) rhf_ptr);
	const uint64_t	   rhf_seq     = opx_ep->rx->state.hdrq.rhf_seq;

	if (!OPX_RHF_SEQ_MATCH(rhf_seq, rhf_rcvd, hfi1_type)) {
		return 1;
	}
	return 0;
}

__OPX_FORCE_INLINE__
bool opx_handle_events(struct fi_opx_ep *opx_ep, const uint64_t hdrq_mask, const enum opx_hfi1_type hfi1_type)
{
	uint64_t events = *(uint64_t *) (opx_ep->hfi->ctrl->base_info.events_bufbase);
	/* In WFR, on a link down, driver/HW always enters a SPC freeze state. It always triggers a HFI1_EVENT_FROZEN.
		Hence, HFI1_EVENT_LINKDOWN can be ignored
	   In JKR, there is no freeze event because there are two ports. If one port is down, the other still functions.
	    Hence, handle HFI1_EVENT_LINKDOWN. */
	if (events & HFI1_EVENT_FROZEN || (events & HFI1_EVENT_LINKDOWN && !(hfi1_type & OPX_HFI1_WFR))) {
		/* reset context only if RHF queue is empty */
		if (opx_is_rhf_empty(opx_ep, hdrq_mask, hfi1_type)) {
			opx_reset_context(opx_ep, events, hfi1_type);
			int ret = opx_hfi1_wrapper_ack_events(opx_ep->hfi, events);
			if (ret) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "ack event failed: %s\n", strerror(errno));
			}
			return true;
		} else {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Context frozen: Not resetting because packets are present in receive queue\n");
		}
	}

	return false;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_poll_many(struct fid_ep *ep, const int lock_required, const uint64_t caps,
			   const enum ofi_reliability_kind reliability, const uint64_t hdrq_mask,
			   const enum opx_hfi1_type hfi1_type, const bool ctx_sharing)
{
	/* All callers to this function should have already obtained the necessary lock */
	assert(!lock_required);

	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	static const unsigned hfi1_poll_max   = 256;
	unsigned	      hfi1_poll_count = 0;
	unsigned	      packets	      = 0;

	if ((caps & FI_LOCAL_COMM) || (caps == 0)) {
		fi_opx_shm_poll_many(ep, 0, hfi1_type);
	}

	if ((caps & FI_REMOTE_COMM) || (caps == 0)) {
		do {
			packets = fi_opx_hfi1_poll_once(ep, FI_OPX_LOCK_NOT_REQUIRED, reliability, hdrq_mask, hfi1_type,
							ctx_sharing);
		} while ((packets > 0) && (hfi1_poll_count++ < hfi1_poll_max));

		struct fi_opx_reliability_service *service   = opx_ep->reli_service;
		union fi_opx_timer_state	  *timer     = &service->timer;
		union fi_opx_timer_stamp	  *timestamp = &service->timestamp;
		uint64_t			   compare   = fi_opx_timer_now(timestamp, timer);

		struct fi_opx_hfi1_context *context = opx_ep->hfi;

		if (OFI_UNLIKELY(compare > context->status_check_next_usec)) {
			int err = fi_opx_context_check_status(context, hfi1_type, opx_ep);

			if (context->status_lasterr != FI_SUCCESS && err == FI_SUCCESS) {
				if (opx_handle_events(opx_ep, hdrq_mask, hfi1_type)) {
					context->status_lasterr = FI_SUCCESS; /* clear error */
				}
			}
			compare = fi_opx_timer_now(timestamp, timer);
			context->status_check_next_usec =
				fi_opx_timer_next_event_usec(timer, timestamp, OPX_CONTEXT_STATUS_CHECK_INTERVAL_USEC);
		}

		// TODO: There needs to be feedback from the replay buffer pool into this following if as well
		//		If the pool is getting full, then send pings out more frequently

		if (OFI_UNLIKELY(compare > service->usec_next)) {
			// Drain all coalesced pings
			fi_opx_hfi_rx_reliablity_process_requests(ep, PENDING_RX_RELIABLITY_COUNT_MAX);
			fi_reliability_service_ping_remote(ep, service);
			// Fetch the timer again as it could have taken us a while to get through reliability
			compare		   = fi_opx_timer_now(timestamp, timer);
			service->usec_next = fi_opx_timer_next_event_usec(timer, timestamp, service->usec_max);
		} // End timer fired
	}

	fi_opx_compiler_msync_writes(); // Workaround for STL-62043

	return;
}

#endif /* _FI_PROV_OPX_HFI1_PROGRESS_H_ */
