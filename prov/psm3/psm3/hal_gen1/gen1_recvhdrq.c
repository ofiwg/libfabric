#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2021 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2021 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/* Copyright (c) 2003-2021 Intel Corporation. All rights reserved. */

#include "psm_user.h"
#include "psm2_hal.h"

#include "ips_epstate.h"
#include "ips_proto.h"
#include "ips_expected_proto.h"
#include "ips_proto_help.h"
#include "ips_proto_internal.h"
#include "gen1_hal.h"

/*
 * Receive header queue initialization.
 */
psm2_error_t
psm3_gen1_recvhdrq_init(const psmi_context_t *context,
		  const struct ips_epstate *epstate,
		  const struct ips_proto *proto,
		  const struct ips_recvhdrq_callbacks *callbacks,
		  uint32_t subcontext,
		  struct ips_recvhdrq *recvq
		 , struct ips_recvhdrq_state *recvq_state,
		  psm3_gen1_cl_q gen1_cl_hdrq
		)
{
	psm2_error_t err = PSM2_OK;

	memset(recvq, 0, sizeof(*recvq));
	recvq->proto = (struct ips_proto *)proto;
	recvq->context = context;
	recvq->subcontext = subcontext;
	recvq->state = recvq_state;
	recvq->gen1_cl_hdrq = gen1_cl_hdrq;
	pthread_spin_init(&recvq->hdrq_lock, PTHREAD_PROCESS_SHARED);
	recvq->hdrq_elemlast = ((psm3_gen1_get_rx_hdr_q_cnt(context->psm_hw_ctxt) - 1) *
				(psm3_gen1_get_rx_hdr_q_ent_size(context->psm_hw_ctxt) >> BYTE2DWORD_SHIFT));

	recvq->epstate = epstate;
	recvq->recvq_callbacks = *callbacks;	/* deep copy */
	SLIST_INIT(&recvq->pending_acks);

	recvq->state->hdrq_head = 0;
	recvq->state->rcv_egr_index_head = NO_EAGER_UPDATE;
	recvq->state->num_hdrq_done = 0;
	recvq->state->num_egrq_done = 0;
	recvq->state->hdr_countdown = 0;
	recvq->state->hdrq_cachedlastscan = 0;

	{
		union psmi_envvar_val env_hdr_update;
		psm3_getenv("PSM3_HEAD_UPDATE",
			    "header queue update interval (0 to update after all entries are processed). Default is 64",
			    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT_FLAGS,
			    (union psmi_envvar_val) 64, &env_hdr_update);

		/* Cap max header update interval to size of header/eager queue */
		recvq->state->head_update_interval =
			min(env_hdr_update.e_uint, psm3_gen1_get_rx_hdr_q_cnt(context->psm_hw_ctxt) - 1);
		recvq->state->egrq_update_interval = 1;
	}
	return err;
}


/* flush the eager buffers, by setting the eager index head to eager index tail
   if eager buffer queue is full.

   Called when we had eager buffer overflows (ERR_TID/HFI_RHF_H_TIDERR
   was set in RHF errors), and no good eager packets were received, so
   that eager head wasn't advanced.
*/
#if 0
static void psm3_gen1_flush_egrq_if_required(struct ips_recvhdrq *recvq)
{
	const uint32_t tail = ips_recvq_tail_get(&recvq->egrq);
	const uint32_t head = ips_recvq_head_get(&recvq->egrq);
	uint32_t egr_cnt = recvq->egrq.elemcnt;

	if ((head % egr_cnt) == ((tail + 1) % egr_cnt)) {
		_HFI_DBG("eager array full after overflow, flushing "
			 "(head %llx, tail %llx)\n",
			 (long long)head, (long long)tail);
		recvq->proto->stats.egr_overflow++;
	}
	return;
}
#endif

/*
 * Helpers for recvhdrq_progress.
 */

static __inline__ int
_get_proto_subcontext(const struct ips_message_header *p_hdr)
{
	return ((__be32_to_cpu(p_hdr->bth[1]) >>
		 HFI_BTH_SUBCTXT_SHIFT) & HFI_BTH_SUBCTXT_MASK);
}

static __inline__ void _dump_invalid_pkt(struct ips_recvhdrq_event *rcv_ev)
{
	uint8_t *payload = ips_recvhdrq_event_payload(rcv_ev);
	uint32_t paylen = ips_recvhdrq_event_paylen(rcv_ev) +
	    ((__be32_to_cpu(rcv_ev->p_hdr->bth[0]) >> 20) & 3);

#ifdef PSM_DEBUG
	psm3_ips_proto_show_header((struct ips_message_header *)
			      rcv_ev->p_hdr, "received invalid pkt");
#endif
	if (psm3_dbgmask & __HFI_PKTDBG) {
		psm3_ips_proto_dump_frame(rcv_ev->p_hdr, HFI_MESSAGE_HDR_SIZE,
				     "header");
		if (!payload) {
			_HFI_DBG("Cannot dump frame; payload is NULL\n");
		} else if (paylen) {
			psm3_ips_proto_dump_frame(payload, paylen, "data");
		}
	}

}

static __inline__ void
_update_error_stats(struct ips_proto *proto, uint32_t err)
{
	if (err & PSM3_GEN1_RHF_ERR_ICRC)
		proto->error_stats.num_icrc_err++;
	if (err & PSM3_GEN1_RHF_ERR_ECC)
		proto->error_stats.num_ecc_err++;
	if (err & PSM3_GEN1_RHF_ERR_LEN)
		proto->error_stats.num_len_err++;
	if (err & PSM3_GEN1_RHF_ERR_TID)
		proto->error_stats.num_tid_err++;
	if (err & PSM3_GEN1_RHF_ERR_DC)
		proto->error_stats.num_dc_err++;
	if (err & PSM3_GEN1_RHF_ERR_DCUN)
		proto->error_stats.num_dcunc_err++;
	if (err & PSM3_GEN1_RHF_ERR_KHDRLEN)
		proto->error_stats.num_khdrlen_err++;
}

#ifdef PSM_DEBUG

static int _check_headers(struct ips_recvhdrq_event *rcv_ev, psm3_gen1_cl_q cl_q)
{
	struct ips_recvhdrq *recvq = (struct ips_recvhdrq *)rcv_ev->recvq;
	struct ips_proto *proto = rcv_ev->proto;
	uint32_t *lrh = (uint32_t *) rcv_ev->p_hdr;
	uint32_t dest_context;
	const uint16_t pkt_dlid = __be16_to_cpu(rcv_ev->p_hdr->lrh[1]);
	const uint16_t base_dlid =
	    __be16_to_cpu(recvq->proto->epinfo.ep_base_lid);

	/* Check that the receive header queue entry has a sane sequence number */
	if (psm3_gen1_check_rhf_sequence_number(psm3_gen1_rhf_get_seq(rcv_ev->gen1_rhf))
	    != PSM_HAL_ERROR_OK) {
		unsigned int seqno=0;

		psm3_gen1_get_rhf_expected_sequence_number(&seqno, cl_q, recvq->context->psm_hw_ctxt);
		psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
				  "ErrPkt: Invalid header queue entry! RHF Sequence in Hdrq Seq: %d, Recvq State Seq: %d. LRH[0]: 0x%08x, LRH[1] (PktCount): 0x%08x\n",
				  psm3_gen1_rhf_get_seq(rcv_ev->gen1_rhf),
				  seqno, lrh[0], lrh[1]);
		return -1;
	}

	/* Verify that the packet was destined for our context */
	dest_context = ips_proto_dest_context_from_header(proto, rcv_ev->p_hdr);
	if_pf(dest_context != recvq->proto->epinfo.ep_context) {

		struct ips_recvhdrq_state *state = recvq->state;

		/* Packet not targeted at us. Drop packet and continue */
		psm3_gen1_ips_ptl_dump_err_stats(proto);
		_dump_invalid_pkt(rcv_ev);

		psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
				  "ErrPkt: Received packet for context %d on context %d. Receive Header Queue offset: 0x%x. Exiting.\n",
				  dest_context, recvq->proto->epinfo.ep_context,
				  state->hdrq_head);

		return -1;
	}

	/* Verify that rhf packet length matches the length in LRH */
	if_pf(psm3_gen1_rhf_get_packet_length(rcv_ev->gen1_rhf) !=
	      ips_proto_lrh2_be_to_bytes(proto, rcv_ev->p_hdr->lrh[2])) {
		_HFI_EPDBG
		    ("ErrPkt: RHF Packet Len (0x%x) does not match LRH (0x%x).\n",
		     psm3_gen1_rhf_get_packet_length(rcv_ev->gen1_rhf) >> 2,
		     __be16_to_cpu(rcv_ev->p_hdr->lrh[2]));

		psm3_gen1_ips_ptl_dump_err_stats(proto);
		_dump_invalid_pkt(rcv_ev);
		return -1;
	}

	/* Verify that the DLID matches our local LID. */
	if_pf(!((base_dlid <= pkt_dlid) &&
		(pkt_dlid <=
		 (base_dlid + (1 << recvq->proto->epinfo.ep_lmc))))) {
		_HFI_EPDBG
		    ("ErrPkt: DLID in LRH (0x%04x) does not match local LID (0x%04x) Skipping packet!\n",
		     rcv_ev->p_hdr->lrh[1], recvq->proto->epinfo.ep_base_lid);
		psm3_gen1_ips_ptl_dump_err_stats(proto);
		_dump_invalid_pkt(rcv_ev);
		return -1;
	}

	return 0;
}
#endif /* PSM_DEBUG */

static __inline__ int do_pkt_cksum(struct ips_recvhdrq_event *rcv_ev)
{
	uint8_t *payload = ips_recvhdrq_event_payload(rcv_ev);
	uint32_t paylen = ips_recvhdrq_event_paylen(rcv_ev) +
	    ((__be32_to_cpu(rcv_ev->p_hdr->bth[0]) >> 20) & 3);
	uint32_t *ckptr;
	uint32_t recv_cksum, cksum, dest_subcontext;
	/* With checksum every packet has a payload */
	psmi_assert_always(payload);

	ckptr = (uint32_t *) (payload + paylen);
	recv_cksum = ckptr[0];

	cksum = psm3_ips_cksum_calculate(rcv_ev->p_hdr, payload, paylen);

	if ((cksum != recv_cksum) || (ckptr[0] != ckptr[1])) {
		struct ips_epstate_entry *epstaddr;
		uint32_t lcontext;
		psm3_gen1_cl_idx hd, tl;

		epstaddr =
		    ips_epstate_lookup(rcv_ev->recvq->epstate,
				       rcv_ev->p_hdr->connidx);
		epstaddr = (epstaddr && epstaddr->ipsaddr) ? epstaddr : NULL;
		lcontext = epstaddr ? rcv_ev->proto->epinfo.ep_context : -1;

		hd = psm3_gen1_get_cl_q_head_index(PSM3_GEN1_CL_Q_RX_HDR_Q,
					rcv_ev->recvq->context->psm_hw_ctxt);
		tl = psm3_gen1_get_cl_q_tail_index(PSM3_GEN1_CL_Q_RX_HDR_Q,
					rcv_ev->recvq->context->psm_hw_ctxt);

		dest_subcontext = _get_proto_subcontext(rcv_ev->p_hdr);

		_HFI_ERROR
		    ("ErrPkt: SharedContext: %s. Local Context: %i, Checksum mismatch from LID %d! Received Checksum: 0x%08x, Expected: 0x%08x & 0x%08x. Opcode: 0x%08x, Error Flag: 0x%08x. hdrq hd 0x%x tl 0x%x rhf 0x%"
		     PRIx64 ", rhfseq 0x%x\n",
		     (dest_subcontext !=
		      rcv_ev->recvq->subcontext) ? "Yes" : "No", lcontext,
		     epstaddr ? __be16_to_cpu(epstaddr->ipsaddr->pathgrp->
					      pg_base_dlid) : -1, cksum,
		     ckptr[0], ckptr[1], _get_proto_hfi_opcode(rcv_ev->p_hdr),
		     psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf), hd, tl, rcv_ev->gen1_rhf.raw_rhf,
		     psm3_gen1_rhf_get_seq(rcv_ev->gen1_rhf));
		/* Dump packet */
		_dump_invalid_pkt(rcv_ev);
		return 0;	/* Packet checksum error */
	}

	return 1;
}

/* receive service routine for each packet opcode starting at
 * OPCODE_RESERVED (C0)
 */
ips_packet_service_fn_t
psm3_gen1_packet_service_routines[] = {
psm3_ips_proto_process_unknown_opcode,	/* 0xC0 */
psm3_ips_proto_mq_handle_tiny,		/* OPCODE_TINY */
psm3_ips_proto_mq_handle_short,		/* OPCODE_SHORT */
psm3_ips_proto_mq_handle_eager,		/* OPCODE_EAGER */
psm3_ips_proto_mq_handle_rts,		/* OPCODE_LONG_RTS */
psm3_ips_proto_mq_handle_cts,		/* OPCODE_LONG_CTS */
psm3_ips_proto_mq_handle_data,		/* OPCODE_LONG_DATA */
ips_protoexp_data,			/* OPCODE_EXPTID */
ips_protoexp_recv_tid_completion,	/* OPCODE_EXPTID_COMPLETION */

/* these are control packets */
psm3_ips_proto_process_ack,		/* OPCODE_ACK */
psm3_ips_proto_process_nak,		/* OPCODE_NAK */
psm3_gen1_ips_ptl_process_becn,			/* OPCODE_BECN */
psm3_ips_proto_process_err_chk,		/* OPCODE_ERR_CHK */
psm3_gen1_ips_ptl_process_err_chk_gen,		/* OPCODE_ERR_CHK_GEN */
psm3_ips_proto_connect_disconnect,	/* OPCODE_CONNECT_REQUEST */
psm3_ips_proto_connect_disconnect,	/* OPCODE_CONNECT_REPLY */
psm3_ips_proto_connect_disconnect,	/* OPCODE_DISCONNECT__REQUEST */
psm3_ips_proto_connect_disconnect,	/* OPCODE_DISCONNECT_REPLY */

/* rest are not control packets */
psm3_ips_proto_am,			/* OPCODE_AM_REQUEST_NOREPLY */
psm3_ips_proto_am,			/* OPCODE_AM_REQUEST */
psm3_ips_proto_am			/* OPCODE_AM_REPLY */

/* D5-DF (OPCODE_FUTURE_FROM to OPCODE_FUTURE_TO) reserved for expansion */
};

/*
 * Core receive progress function
 *
 * recvhdrq_progress is the core function that services the receive header
 * queue and optionally, the eager queue.  At the lowest level, it identifies
 * packets marked with errors by the chip and also detects and corrects when
 * eager overflow conditions occur.  At the highest level, it queries the
 * 'epstate' interface to classify packets from "known" and "unknown"
 * endpoints.  In order to support shared contexts, it can also handle packets
 * destined for other contexts (or "subcontexts").
 */
psm2_error_t psm3_gen1_recvhdrq_progress(struct ips_recvhdrq *recvq)
{
	GENERIC_PERF_BEGIN(PSM_RX_SPEEDPATH_CTR); /* perf stats */
	struct ips_recvhdrq_state *state = recvq->state;
	PSMI_CACHEALIGN struct ips_recvhdrq_event rcv_ev = {.proto =
		    recvq->proto,
		.recvq = recvq
	};
	struct ips_epstate_entry *epstaddr;
	uint32_t num_hdrq_done = 0;
	const uint32_t num_hdrq_todo = psm3_gen1_get_rx_hdr_q_cnt(recvq->context->psm_hw_ctxt);
	uint32_t dest_subcontext;
	const uint32_t hdrq_elemsz = psm3_gen1_get_rx_hdr_q_ent_size(recvq->context->psm_hw_ctxt) >> BYTE2DWORD_SHIFT;
	int ret = IPS_RECVHDRQ_CONTINUE;
	int done = 0, empty = 0;
	int do_hdr_update = 0;
	const psm3_gen1_cl_q gen1_hdr_q = recvq->gen1_cl_hdrq;
	const psm3_gen1_cl_q psm_hal_egr_q = gen1_hdr_q + 1;

	/* Returns whether the currently set 'rcv_hdr'/head is a readable entry */
#define next_hdrq_is_ready()  (! empty )

	if (psm3_gen1_cl_q_empty(state->hdrq_head, gen1_hdr_q, recvq->context->psm_hw_ctxt))
	    return PSM2_OK;

	PSM2_LOG_MSG("entering");

	done = !next_hdrq_is_ready();

	rcv_ev.gen1_hdr_q = gen1_hdr_q;

	while (!done) {
		psm3_gen1_get_receive_event(state->hdrq_head, recvq->context->psm_hw_ctxt, 1,
					   &rcv_ev);
		_HFI_VDBG
		    ("new packet: rcv_hdr %p, rhf %" PRIx64 "\n",
		     rcv_ev.p_hdr, rcv_ev.gen1_rhf.raw_rhf);

#ifdef PSM_DEBUG
		if_pf(_check_headers(&rcv_ev, gen1_hdr_q))
			goto skip_packet;
#endif
		dest_subcontext = _get_proto_subcontext(rcv_ev.p_hdr);

		/* If the destination is not our subcontext, process
		 * message as subcontext message (shared contexts) */
		if (dest_subcontext != recvq->subcontext) {
			rcv_ev.ipsaddr = NULL;

			ret = recvq->recvq_callbacks.callback_subcontext
						(&rcv_ev, dest_subcontext);
			if (ret == IPS_RECVHDRQ_REVISIT)
			{
				// try processing on next progress call
				PSM2_LOG_MSG("leaving");
				GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
				return PSM2_OK_NO_PROGRESS;
			}

			goto skip_packet;
		}

		if_pf(psm3_gen1_rhf_get_all_err_flags(rcv_ev.gen1_rhf)) {

			_update_error_stats(recvq->proto, psm3_gen1_rhf_get_all_err_flags(rcv_ev.gen1_rhf));

			recvq->recvq_callbacks.callback_error(&rcv_ev);

			if ((psm3_gen1_rhf_get_rx_type(rcv_ev.gen1_rhf) != PSM3_GEN1_RHF_RX_TYPE_EAGER) ||
			    (!(psm3_gen1_rhf_get_all_err_flags(rcv_ev.gen1_rhf) & PSM3_GEN1_RHF_ERR_TID)))
				goto skip_packet;

			/* no pending eager update, header
			 * is not currently under tracing. */
			if (state->hdr_countdown == 0 &&
			    state->rcv_egr_index_head == NO_EAGER_UPDATE) {
				uint32_t egr_cnt = psm3_gen1_get_rx_egr_tid_cnt(recvq->context->psm_hw_ctxt);
				psm3_gen1_cl_idx etail=0, ehead=0;

				ehead = psm3_gen1_get_cl_q_head_index(
					psm_hal_egr_q,
					rcv_ev.recvq->context->psm_hw_ctxt);
				etail = psm3_gen1_get_cl_q_tail_index(
					psm_hal_egr_q,
					rcv_ev.recvq->context->psm_hw_ctxt);
				if (ehead == ((etail + 1) % egr_cnt)) {
					/* eager is full,
					 * trace existing header entries */
					uint32_t hdr_size =
						recvq->hdrq_elemlast +
						hdrq_elemsz;
					psm3_gen1_cl_idx htail=0;

					htail = psm3_gen1_get_cl_q_tail_index(
					   gen1_hdr_q,
					   rcv_ev.recvq->context->psm_hw_ctxt);
					const uint32_t hhead = state->hdrq_head;

					state->hdr_countdown =
						(htail > hhead) ?
						(htail - hhead) :
						(htail + hdr_size - hhead);
				}
			}

			/* Eager packet and tiderr.
			 * Don't consider updating egr head, unless we're in
			 * the congested state.  If we're congested, we should
			 * try to keep the eager buffers free. */

			if (!rcv_ev.is_congested)
				goto skip_packet_no_egr_update;
			else
				goto skip_packet;
		}

		/* If checksum is enabled, verify that it is valid */
		if_pf(rcv_ev.has_cksum && !do_pkt_cksum(&rcv_ev))
			goto skip_packet;

		if (_HFI_VDBG_ON)
		{
			psm3_gen1_cl_idx egr_buff_q_head, egr_buff_q_tail;

			egr_buff_q_head = psm3_gen1_get_cl_q_head_index(
					    psm_hal_egr_q,
					    rcv_ev.recvq->context->psm_hw_ctxt);
			egr_buff_q_tail = psm3_gen1_get_cl_q_tail_index(
					    psm_hal_egr_q,
					    rcv_ev.recvq->context->psm_hw_ctxt);

			_HFI_VDBG_ALWAYS(
				"hdrq_head %d, p_hdr: %p, opcode %x, payload %p paylen %d; "
				"egrhead %x egrtail %x; "
				"useegrbit %x egrindex %x, egroffset %x, egrindexhead %x\n",
				state->hdrq_head,
				rcv_ev.p_hdr,
				_get_proto_hfi_opcode(rcv_ev.p_hdr),
				ips_recvhdrq_event_payload(&rcv_ev),
				ips_recvhdrq_event_paylen(&rcv_ev),
				egr_buff_q_head,egr_buff_q_tail,
				psm3_gen1_rhf_get_use_egr_buff(rcv_ev.gen1_rhf),
				psm3_gen1_rhf_get_egr_buff_index(rcv_ev.gen1_rhf),
				psm3_gen1_rhf_get_egr_buff_offset(rcv_ev.gen1_rhf),
				state->rcv_egr_index_head);
		}

                PSM2_LOG_PKT_STRM(PSM2_LOG_RX,rcv_ev.p_hdr,&rcv_ev.gen1_rhf.raw_rhf,
				  "PKT_STRM:");

		/* Classify packet from a known or unknown endpoint */
		epstaddr = ips_epstate_lookup(recvq->epstate,
					       rcv_ev.p_hdr->connidx);
		if_pf((epstaddr == NULL) || (epstaddr->ipsaddr == NULL)) {
			rcv_ev.ipsaddr = NULL;
			recvq->recvq_callbacks.
			    callback_packet_unknown(&rcv_ev);
		} else {
			rcv_ev.ipsaddr = epstaddr->ipsaddr;
			psmi_assert(PSMI_HOWMANY(psm3_gen1_packet_service_routines)
				== OPCODE_FUTURE_FROM - OPCODE_RESERVED);
			ret = ips_proto_process_packet(&rcv_ev,
				psm3_gen1_packet_service_routines);
			if (ret == IPS_RECVHDRQ_REVISIT)
			{
				// try processing on next progress call
				PSM2_LOG_MSG("leaving");
				GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
				return PSM2_OK_NO_PROGRESS;
			}
		}

skip_packet:
		/*
		 * if eager buffer is used, record the index.
		 */
		if (psm3_gen1_rhf_get_use_egr_buff(rcv_ev.gen1_rhf)) {
			/* set only when a new entry is used */
			if (psm3_gen1_rhf_get_egr_buff_offset(rcv_ev.gen1_rhf) == 0) {
				state->rcv_egr_index_head =
					psm3_gen1_rhf_get_egr_buff_index(rcv_ev.gen1_rhf);
				state->num_egrq_done++;
			}
			/* a header entry is using an eager entry, stop tracing. */
			state->hdr_countdown = 0;
		}

skip_packet_no_egr_update:
		/* Note that state->hdrq_head is sampled speculatively by the code
		 * in psm3_gen1_ips_ptl_shared_poll() when context sharing, so it is not safe
		 * for this shared variable to temporarily exceed the last element. */
		_HFI_VDBG
		    ("head %d, elemsz %d elemlast %d\n",
		     state->hdrq_head, hdrq_elemsz,
		     recvq->hdrq_elemlast);
		psm3_gen1_retire_hdr_q_entry(&state->hdrq_head, gen1_hdr_q,
					    recvq->context->psm_hw_ctxt,
					    hdrq_elemsz, recvq->hdrq_elemlast, &empty);
		state->num_hdrq_done++;
		num_hdrq_done++;
		done = (!next_hdrq_is_ready() || (ret == IPS_RECVHDRQ_BREAK)
			|| (num_hdrq_done == num_hdrq_todo));

		do_hdr_update = (state->head_update_interval ?
				 (state->num_hdrq_done ==
				  state->head_update_interval) : done);
		if (do_hdr_update) {

			psm3_gen1_set_cl_q_head_index(
					state->hdrq_head,
					gen1_hdr_q,
				 	rcv_ev.recvq->context->psm_hw_ctxt);
			/* Reset header queue entries processed */
			state->num_hdrq_done = 0;
		}
		if (state->num_egrq_done >= state->egrq_update_interval) {
			/* Lazy update of egrq */
			if (state->rcv_egr_index_head != NO_EAGER_UPDATE) {
				psm3_gen1_set_cl_q_head_index(
					state->rcv_egr_index_head,
				     	psm_hal_egr_q,
				        recvq->context->psm_hw_ctxt);
				state->rcv_egr_index_head = NO_EAGER_UPDATE;
				state->num_egrq_done = 0;
			}
		}
		if (state->hdr_countdown > 0) {
			/* a header entry is consumed. */
			state->hdr_countdown -= hdrq_elemsz;
			if (state->hdr_countdown == 0) {
				/* header entry count reaches zero. */
				psm3_gen1_cl_idx tail=0;

				tail = psm3_gen1_get_cl_q_tail_index(
					   psm_hal_egr_q,
					   recvq->context->psm_hw_ctxt);

				psm3_gen1_cl_idx head=0;

				head = psm3_gen1_get_cl_q_head_index(
					   psm_hal_egr_q,
					   recvq->context->psm_hw_ctxt);

				uint32_t egr_cnt = psm3_gen1_get_rx_egr_tid_cnt(recvq->context->psm_hw_ctxt);
				/* Checks eager-full again. This is a real false-egr-full */
				if (head == ((tail + 1) % egr_cnt)) {

					psm3_gen1_set_cl_q_tail_index(
						tail,
					        psm_hal_egr_q,
						recvq->context->psm_hw_ctxt);

					_HFI_DBG
					    ("eager array full after overflow, flushing "
					     "(head %llx, tail %llx)\n",
					     (long long)head, (long long)tail);
					recvq->proto->stats.egr_overflow++;
				} else
					_HFI_ERROR
					    ("PSM BUG: EgrOverflow: eager queue is not full\n");
			}
		}
	}
	/* while (hdrq_entries_to_read) */

	/* Process any pending acks before exiting */
	process_pending_acks(recvq);

	PSM2_LOG_MSG("leaving");
	GENERIC_PERF_END(PSM_RX_SPEEDPATH_CTR); /* perf stats */
	return num_hdrq_done ? PSM2_OK : PSM2_OK_NO_PROGRESS;
}

/*	This function is designed to implement RAPID CCA. It iterates
	through the recvq, checking each element for set FECN or BECN bits.
	In the case of finding one, the proper response is executed, and the bits
	are cleared.
*/
psm2_error_t psm3_gen1_recvhdrq_scan_cca (struct ips_recvhdrq *recvq)
{
// TBD - rcv_ev is never returned from this, is_congested and congested_pkts counts never used

/* Looks at hdr and determines if it is the last item in the queue */

#define is_last_hdr(idx)				\
	psm3_gen1_cl_q_empty(idx, gen1_hdr_q, recvq->context->psm_hw_ctxt)

	struct ips_recvhdrq_state *state = recvq->state;
	PSMI_CACHEALIGN struct ips_recvhdrq_event rcv_ev = {.proto = recvq->proto,
							    .recvq = recvq
	};

	uint32_t num_hdrq_done = state->hdrq_cachedlastscan /
		psm3_gen1_get_rx_hdr_q_ent_size(recvq->context->psm_hw_ctxt) >> BYTE2DWORD_SHIFT;
	const int num_hdrq_todo = psm3_gen1_get_rx_hdr_q_cnt(recvq->context->psm_hw_ctxt);
	const uint32_t hdrq_elemsz = psm3_gen1_get_rx_hdr_q_ent_size(recvq->context->psm_hw_ctxt) >> BYTE2DWORD_SHIFT;

	int done;
	uint32_t scan_head = state->hdrq_head + state->hdrq_cachedlastscan;
	const psm3_gen1_cl_q gen1_hdr_q = recvq->gen1_cl_hdrq;

	/* Skip the first element, since we're going to process it soon anyway */
	if ( state->hdrq_cachedlastscan == 0 )
	{
		scan_head += hdrq_elemsz;
		num_hdrq_done++;
	}

	PSM2_LOG_MSG("entering");
	done = !is_last_hdr(scan_head);
	rcv_ev.gen1_hdr_q = gen1_hdr_q;
	while (!done) {
		psm3_gen1_get_receive_event(scan_head, recvq->context->psm_hw_ctxt, 0,
					   &rcv_ev);
		_HFI_VDBG
			("scanning new packet for CCA: rcv_hdr %p, rhf %" PRIx64 "\n",
			 rcv_ev.p_hdr, rcv_ev.gen1_rhf.raw_rhf);

		if_pt ( _is_cca_fecn_set(rcv_ev.p_hdr) & IPS_RECV_EVENT_FECN ) {
			struct ips_epstate_entry *epstaddr = ips_epstate_lookup(recvq->epstate,
										rcv_ev.p_hdr->connidx);

			if (epstaddr != NULL && epstaddr->ipsaddr != NULL)
			{
				rcv_ev.ipsaddr = epstaddr->ipsaddr;

				/* Send BECN back */
				ips_epaddr_t *ipsaddr = rcv_ev.ipsaddr;
				struct ips_message_header *p_hdr = rcv_ev.p_hdr;
				ips_epaddr_flow_t flowid = ips_proto_flowid(p_hdr);
				struct ips_flow *flow;
				ips_scb_t ctrlscb;

				psmi_assert(flowid < EP_FLOW_LAST);
				flow = &ipsaddr->flows[flowid];
				ctrlscb.scb_flags = 0;
				ctrlscb.ips_lrh.data[0].u32w0 =
					flow->cca_ooo_pkts;

				rcv_ev.proto->epaddr_stats.congestion_pkts++;
				/* Clear FECN event */
				rcv_ev.is_congested &= ~IPS_RECV_EVENT_FECN;

				// no payload, pass cksum so non-NULL
				psm3_ips_proto_send_ctrl_message(flow,
							    OPCODE_BECN,
							    &flow->ipsaddr->
							    ctrl_msg_queued,
							    &ctrlscb, ctrlscb.cksum, 0);
			}
		}
		else if_pt (0 != (_is_cca_becn_set(rcv_ev.p_hdr) << (IPS_RECV_EVENT_BECN - 1))) {
			struct ips_epstate_entry *epstaddr = ips_epstate_lookup(recvq->epstate,
										rcv_ev.p_hdr->connidx);

			if (epstaddr != NULL && epstaddr->ipsaddr != NULL)
			{
				rcv_ev.ipsaddr = epstaddr->ipsaddr;

				/* Adjust flow */
				struct ips_proto *proto = rcv_ev.proto;
				struct ips_message_header *p_hdr = rcv_ev.p_hdr;
				ips_epaddr_t *ipsaddr = rcv_ev.ipsaddr;
				struct ips_flow *flow;
				ips_epaddr_flow_t flowid = ips_proto_flowid(p_hdr);

				psmi_assert(flowid < EP_FLOW_LAST);
				flow = &ipsaddr->flows[flowid];
				if ((flow->path->opa.pr_ccti +
				     proto->cace[flow->path->pr_sl].ccti_increase) <= proto->ccti_limit) {
					ips_cca_adjust_rate(flow->path,
							    proto->cace[flow->path->pr_sl].ccti_increase);
					/* Clear congestion event */
					rcv_ev.is_congested &= ~IPS_RECV_EVENT_BECN;
				}
			}
		}

		num_hdrq_done++;
		scan_head += hdrq_elemsz;
		state->hdrq_cachedlastscan += hdrq_elemsz;

		done = (num_hdrq_done == num_hdrq_todo && !is_last_hdr(scan_head) );

	}
	/* while (hdrq_entries_to_read) */


	PSM2_LOG_MSG("leaving");
	return PSM2_OK;
}
#endif /* PSM_OPA */
