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

/* This file implements the HAL specific code for PSM PTL for ips */
#include "psm_user.h"
#include "psm2_hal.h"
#include "ptl_ips.h"
#include "psm_mq_internal.h"
#include "gen1_hal.h"
#include "gen1_spio.c"	// TBD make this a normal .c file, just needed spio_init

/*
 * Sample implementation of shared contexts context.
 *
 * In shared mode, the hardware queue is serviced by more than one process.
 * Each process also mirrors the hardware queue in software (represented by an
 * ips_recvhdrq).  For packets we service in the hardware queue that are not
 * destined for us, we write them in other processes's receive queues
 * (represented by an gen1_ips_writehdrq).
 *
 */
struct gen1_ptl_shared {
	ptl_t *ptl;		/* backptr to main ptl */
	uint32_t context;
	uint32_t subcontext;
	uint32_t subcontext_cnt;

	pthread_spinlock_t *context_lock;
	struct gen1_ips_subcontext_ureg *subcontext_ureg[PSM_HAL_MAX_SHARED_CTXTS];
	struct gen1_ips_hwcontext_ctrl *hwcontext_ctrl;
	struct ips_recvhdrq recvq;	/* subcontext receive queue */
	struct ips_recvhdrq_state recvq_state;	/* subcontext receive queue state */
	struct gen1_ips_writehdrq writeq[PSM_HAL_MAX_SHARED_CTXTS];	/* peer subcontexts */
};

psm2_error_t psm3_gen1_ips_ptl_poll(ptl_t *ptl_gen, int _ignored);
int psm3_gen1_ips_ptl_recvq_isempty(const struct ptl *ptl);
psm2_error_t psm3_gen1_ips_ptl_shared_poll(ptl_t *ptl, int _ignored);

static inline int psm3_gen1_get_sc2vl_map(struct ips_proto *proto)
{
	hfp_gen1_pc_private *psm_hw_ctxt = proto->ep->context.psm_hw_ctxt;
	uint8_t i;

	/* Get SC2VL table for unit, port */
	for (i = 0; i < PSMI_N_SCS; i++) {
		int ret = psm3_gen1_get_port_sc2vl(proto->ep->unit_id,
									 proto->ep->portnum, i);
		if (ret < 0)
			/* Unable to get SC2VL. Set it to default */
			ret = PSMI_VL_DEFAULT;

		psm_hw_ctxt->sc2vl[i] = (uint16_t) ret;
	}
	return PSM_HAL_ERROR_OK;
}

/* (Re)load the SL2SC table */
void psm3_gen1_ips_ptl_init_sl2sc_table(struct ips_proto *proto)
{
	int ret, i;

	/* Get SL2SC table for unit, port */
	for (i = 0; i < PSMI_N_SCS; i++) {
		if ((ret =
		     psm3_gen1_get_port_sl2sc(proto->ep->unit_id,
					proto->ep->portnum, (uint8_t) i)) < 0) {
			/* Unable to get SL2SC. Set it to default */
			ret = PSMI_SC_DEFAULT;
		}

		proto->sl2sc[i] = (uint16_t) ret;
	}
	psm3_gen1_get_sc2vl_map(proto);
}

static inline int psm3_hfp_gen1_write_header_to_subcontext(struct ips_message_header *pimh,
					       psm3_gen1_cl_idx idx,
					       psm3_gen1_raw_rhf_t rhf,
					       psm3_gen1_cl_q cl_q,
					       psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];
	uint32_t *pu32 = pcl_q->hdr_qe.hdrq_base_addr + (idx + psm3_gen1_hdrget_hdrq_offset((uint32_t *)&rhf));
	struct ips_message_header *piph_dest = (struct ips_message_header *)pu32;

	*piph_dest = *pimh;
	return PSM_HAL_ERROR_OK;
}

static inline
int
psm3_gen1_write_eager_packet(struct gen1_ips_writehdrq *writeq,
		       struct ips_recvhdrq_event *rcv_ev,
		       psm3_gen1_cl_idx write_hdr_tail,
		       uint32_t subcontext,
		       psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	psm3_gen1_cl_idx write_egr_tail;
	write_egr_tail = psm3_gen1_get_cl_q_tail_index(
					 PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(subcontext),
					 ctxt);
	uint32_t next_write_egr_tail = write_egr_tail;
	/* checksum is trimmed from paylen, we need to add back */
	uint32_t rcv_paylen = ips_recvhdrq_event_paylen(rcv_ev) +
	    (rcv_ev->has_cksum ? PSM_CRC_SIZE_IN_BYTES : 0);
	psmi_assert(rcv_paylen > 0);
	uint32_t egr_elemcnt = ctrl->ctxt_info.egrtids;
	uint32_t egr_elemsz = ctrl->ctxt_info.rcvegr_size;

	/* Loop as long as the write eager queue is NOT full */
	while (1) {
		next_write_egr_tail++;
		if (next_write_egr_tail >= egr_elemcnt)
			next_write_egr_tail = 0;
		psm3_gen1_cl_idx egr_head;
		egr_head = psm3_gen1_get_cl_q_head_index(
				   PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(subcontext),
				   ctxt);
		if (next_write_egr_tail == egr_head) {
			break;
		}

		/* Move to next eager entry if leftover is not enough */
		if ((writeq->state->egrq_offset + rcv_paylen) >
		    egr_elemsz) {
			writeq->state->egrq_offset = 0;
			write_egr_tail = next_write_egr_tail;

			/* Update the eager buffer tail pointer */
			psm3_gen1_set_cl_q_tail_index(write_egr_tail,
						PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(subcontext),
						ctxt);
		} else {
			/* There is enough space in this entry! */
			/* Use pre-calculated address from look-up table */
			char *write_payload =
				psm_hw_ctxt->cl_qs[PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(subcontext)].egr_buffs[write_egr_tail]
				+ writeq->state->egrq_offset;
			const char *rcv_payload =
			    ips_recvhdrq_event_payload(rcv_ev);

			psmi_assert(write_payload != NULL);
			psmi_assert(rcv_payload != NULL);
			psm3_mq_mtucpy(write_payload, rcv_payload, rcv_paylen);

			/* Fix up the rhf with the subcontext's eager index/offset */
			psm3_gen1_hdrset_egrbfr_index((uint32_t*)(&rcv_ev->gen1_rhf.raw_rhf),write_egr_tail);
			psm3_gen1_hdrset_egrbfr_offset((uint32_t *)(&rcv_ev->gen1_rhf.raw_rhf), (writeq->state->
								egrq_offset >> 6));
			/* Copy the header to the subcontext's header queue */
			psm3_hfp_gen1_write_header_to_subcontext(rcv_ev->p_hdr,
							    write_hdr_tail,
							    rcv_ev->gen1_rhf.raw_rhf,
							    PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
							    ctxt);

			/* Update offset to next 64B boundary */
			writeq->state->egrq_offset =
			    (writeq->state->egrq_offset + rcv_paylen +
			     63) & (~63);
			return IPS_RECVHDRQ_CONTINUE;
		}
	}

	/* At this point, the eager queue is full -- drop the packet. */
	/* Copy the header to the subcontext's header queue */

	/* Mark header with ETIDERR (eager overflow) */
	psm3_gen1_hdrset_err_flags((uint32_t*) (&rcv_ev->gen1_rhf.raw_rhf), HFI_RHF_TIDERR);

	/* Clear UseEgrBfr bit because payload is dropped */
	psm3_gen1_hdrset_use_egrbfr((uint32_t *)(&rcv_ev->gen1_rhf.raw_rhf), 0);
	psm3_hfp_gen1_write_header_to_subcontext(rcv_ev->p_hdr,
					    write_hdr_tail,
					    rcv_ev->gen1_rhf.raw_rhf,
					    PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
					    ctxt);
	return IPS_RECVHDRQ_BREAK;
}

static inline
void
psm3_gen1_writehdrq_write_rhf_atomic(uint64_t *rhf_dest, uint64_t rhf_src)
{
	/*
	 * In 64-bit mode, we check in init that the rhf will always be 8-byte
	 * aligned
	 */
	*rhf_dest = rhf_src;
	return;
}

static inline int psm3_hfp_gen1_write_rhf_to_subcontext(psm3_gen1_raw_rhf_t rhf,
					    psm3_gen1_cl_idx idx,
					    uint32_t *phdrq_rhf_seq,
					    psm3_gen1_cl_q cl_q,
					    psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];

	if (!get_psm_gen1_hi()->hfp_private.dma_rtail)
	{
		uint32_t rhf_seq = *phdrq_rhf_seq;
		psm3_gen1_hdrset_seq((uint32_t *) &rhf, rhf_seq);
		rhf_seq++;
		if (rhf_seq > LAST_RHF_SEQNO)
			rhf_seq = 1;

		*phdrq_rhf_seq = rhf_seq;
	}

	/* Now write the new rhf */
	psm3_gen1_writehdrq_write_rhf_atomic((uint64_t*)(pcl_q->hdr_qe.hdrq_base_addr +
					       (idx + get_psm_gen1_hi()->hfp_private.hdrq_rhf_off)),
				    rhf);
	return PSM_HAL_ERROR_OK;
}

static
int
psm3_gen1_ips_subcontext_ignore(struct ips_recvhdrq_event *rcv_ev,
		      uint32_t subcontext)
{
	return IPS_RECVHDRQ_CONTINUE;
}

static inline
int
psm3_gen1_forward_packet_to_subcontext(struct gen1_ips_writehdrq *writeq,
				      struct ips_recvhdrq_event *rcv_ev,
				      uint32_t subcontext,
				      psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	psm3_gen1_cl_idx write_hdr_head;
	psm3_gen1_cl_idx write_hdr_tail;
	uint32_t hdrq_elemsz = ctrl->ctxt_info.rcvhdrq_entsize >> BYTE2DWORD_SHIFT;
	psm3_gen1_cl_idx next_write_hdr_tail;
	int result = IPS_RECVHDRQ_CONTINUE;

	/* Drop packet if write header queue is disabled */
	if_pf (!writeq->state->enabled) {
		return IPS_RECVHDRQ_BREAK;
	}

	write_hdr_head = psm3_gen1_get_cl_q_head_index(
				     PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
				     ctxt);
	write_hdr_tail = psm3_gen1_get_cl_q_tail_index(
					 PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
				     ctxt);
	/* Drop packet if write header queue is full */
	next_write_hdr_tail = write_hdr_tail + hdrq_elemsz;
	if (next_write_hdr_tail > writeq->hdrq_elemlast) {
		next_write_hdr_tail = 0;
	}
	if (next_write_hdr_tail == write_hdr_head) {
		return IPS_RECVHDRQ_BREAK;
	}
	// could test rcv_ev->payload instead of use_egr_buff
	if (psm3_gen1_rhf_get_use_egr_buff(rcv_ev->gen1_rhf))
	{
		result = psm3_gen1_write_eager_packet(writeq, rcv_ev,
						write_hdr_tail,
						subcontext,
						ctxt);
	} else {
		/* Copy the header to the subcontext's header queue */
		psm3_hfp_gen1_write_header_to_subcontext(rcv_ev->p_hdr,
						    write_hdr_tail,
						    rcv_ev->gen1_rhf.raw_rhf,
						    PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
						    ctxt);
	}

	/* Ensure previous writes are visible before writing rhf seq or tail */
	ips_wmb();

	/* The following func call may modify the hdrq_rhf_seq */
	psm3_hfp_gen1_write_rhf_to_subcontext(rcv_ev->gen1_rhf.raw_rhf, write_hdr_tail,
					 &writeq->state->hdrq_rhf_seq,
					 PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
					 ctxt);
	/* The tail must be updated regardless of PSM_HAL_CAP_DMA_RTAIL
	 * since this tail is also used to keep track of where
	 * to write to next. For subcontexts there is
	 * no separate shadow copy of the tail. */
	psm3_gen1_set_cl_q_tail_index(next_write_hdr_tail,
				PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(subcontext),
				ctxt);

	return result;
}

static
int
psm3_gen1_ips_subcontext_process(struct ips_recvhdrq_event *rcv_ev,
		       uint32_t subcontext)
{
	struct gen1_ptl_shared *recvshc = ((struct ptl_ips *)(rcv_ev->proto->ptl))->recvshc;
	if_pt(subcontext != recvshc->subcontext &&
	      subcontext < recvshc->subcontext_cnt) {
		return psm3_gen1_forward_packet_to_subcontext(&recvshc->writeq[subcontext],
							     rcv_ev, subcontext,
							     rcv_ev->recvq->context->psm_hw_ctxt);
	}
	else {
		_HFI_VDBG
			("Drop pkt for subcontext %d out of %d (I am %d) : errors 0x%x\n",
			 (int)subcontext, (int)recvshc->subcontext_cnt,
			 (int)recvshc->subcontext, psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf));
                return IPS_RECVHDRQ_BREAK;
	}
}

static psm2_error_t psm3_gen1_shrecvq_init(ptl_t *ptl, const psmi_context_t *context);
static psm2_error_t psm3_gen1_shrecvq_fini(ptl_t *ptl);

static inline int psm3_gen1_subcontext_ureg_get(ptl_t *ptl_gen,
					struct gen1_ips_subcontext_ureg **uregp,
					psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	int i;
	struct ptl_ips *ptl = (struct ptl_ips *) ptl_gen;

	ptl->recvshc->hwcontext_ctrl = psm_hw_ctxt->hwcontext_ctrl;
	for (i=0;i < psm_hw_ctxt->user_info.subctxt_cnt; i++)
		uregp[i] = psm_hw_ctxt->subcontext_ureg[i];
	return PSM_HAL_ERROR_OK;
}

// initialize HAL specific parts of ptl_ips
// This is called after most of the generic aspects have been initialized
// so we can use ptl->ep, ptl->ctl, etc as needed
// However it is called prior to ips_proto_init.  ips_proto_init requires some
// ips_ptl items such as ptl->spioc
psm2_error_t psm3_gen1_ips_ptl_init_pre_proto_init(struct ptl_ips *ptl)
{
	psm2_error_t err = PSM2_OK;
	const psmi_context_t *context = &ptl->ep->context;
	const int enable_shcontexts = (psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt) > 0);
	ptl->ctl->ep_poll = enable_shcontexts ? psm3_gen1_ips_ptl_shared_poll : psm3_gen1_ips_ptl_poll;
	/*
	 * Context sharing, setup subcontext ureg page.
	 */
	if (enable_shcontexts) {
		struct gen1_ptl_shared *recvshc;

		recvshc = (struct gen1_ptl_shared *)
		    psmi_calloc(ptl->ep, UNDEFINED, 1, sizeof(struct gen1_ptl_shared));
		if (recvshc == NULL) {
			err = PSM2_NO_MEMORY;
			goto fail;
		}

		ptl->recvshc = recvshc;
		recvshc->ptl = (ptl_t *)ptl;

		/* Initialize recvshc fields */
		recvshc->context = psm3_gen1_get_context(context->psm_hw_ctxt);
		recvshc->subcontext = psmi_hal_get_subctxt(context->psm_hw_ctxt);
		recvshc->subcontext_cnt = psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt);
		psmi_assert_always(recvshc->subcontext_cnt <=
				   PSM_HAL_MAX_SHARED_CTXTS);
		psmi_assert_always(recvshc->subcontext <
				   recvshc->subcontext_cnt);

		/*
		 * Using ep->context to avoid const modifier since this function
		 * will modify the content in ep->context.
		 */
		if ((err = psm3_gen1_subcontext_ureg_get((ptl_t *)ptl,
							recvshc->subcontext_ureg, context->psm_hw_ctxt)))
			goto fail;

		/* Note that the GEN1 HAL instance initializes struct gen1_ips_subcontext_ureg
		   during context open. */

		recvshc->context_lock = &recvshc->hwcontext_ctrl->context_lock;
		if (recvshc->subcontext == 0) {
			if (pthread_spin_init(recvshc->context_lock,
					      PTHREAD_PROCESS_SHARED) != 0) {
				err =
				    psm3_handle_error(ptl->ep,
						      PSM2_EP_DEVICE_FAILURE,
						      "Couldn't initialize process-shared spin lock");
				goto fail;
			}
		}
	}
	/*
	 * Hardware send pio used by eager and control messages.
	 */
	if ((err = psm3_gen1_spio_init(context, (ptl_t *)ptl, &ptl->spioc)))
		goto fail;
fail:
	return err;
}

// initialize HAL specific parts of ptl_ips
// This is called after after ips_proto_init and after most of the generic
// aspects of ips_ptl have been initialized
// so we can use ptl->ep and ptl->proto as needed
psm2_error_t psm3_gen1_ips_ptl_init_post_proto_init(struct ptl_ips *ptl)
{
	psm2_error_t err = PSM2_OK;
	const psmi_context_t *context = &ptl->ep->context;
	const int enable_shcontexts = (psmi_hal_get_subctxt_cnt(context->psm_hw_ctxt) > 0);
	/*
	 * Hardware receive hdr/egr queue, services incoming packets and issues
	 * callbacks for protocol handling in proto_recv.  It uses the epstate
	 * interface to determine if a packet is known or unknown.
	 */
	if (!enable_shcontexts) {
		struct ips_recvhdrq_callbacks recvq_callbacks;
		recvq_callbacks.callback_packet_unknown =
		    psm3_gen1_ips_ptl_process_unknown;
		recvq_callbacks.callback_subcontext = psm3_gen1_ips_subcontext_ignore;
		recvq_callbacks.callback_error = psm3_gen1_ips_ptl_process_packet_error;
		if ((err =
		     psm3_gen1_recvhdrq_init(context, &ptl->epstate, &ptl->proto,
				       &recvq_callbacks,
				       0,
				       &ptl->recvq
				       ,&ptl->recvq_state,
				       PSM3_GEN1_CL_Q_RX_HDR_Q)))
			goto fail;
	}
	/*
	 * Software receive hdr/egr queue, used in shared contexts.
	 */
	else if ((err = psm3_gen1_shrecvq_init((ptl_t*)ptl, context)))
		goto fail;
fail:
	return err;
}

// finalize HAL specific parts of ptl_ips
// This is called before the generic aspects have been finalized
// but after ips_proto has been finalized
// so we can use ptl->ep as needed
psm2_error_t psm3_gen1_ips_ptl_fini(struct ptl_ips *ptl)
{
	psm2_error_t err = PSM2_OK;
	const int enable_shcontexts = (psmi_hal_get_subctxt_cnt(ptl->ep->context.psm_hw_ctxt) > 0);
	if ((err = psm3_gen1_spio_fini(&ptl->spioc, ptl->ep->context.psm_hw_ctxt)))
		goto fail;
	if (enable_shcontexts && (err = psm3_gen1_shrecvq_fini((ptl_t*)ptl)))
		goto fail;
fail:
	return err;
}

psm2_error_t psm3_gen1_ips_ptl_poll(ptl_t *ptl_gen, int _ignored)
{
	struct ptl_ips *ptl = (struct ptl_ips *)ptl_gen;
	const uint64_t current_count = get_cycles();
	const int do_lock = PSMI_LOCK_DISABLED &&
		psmi_hal_has_sw_status(PSM_HAL_PSMI_RUNTIME_RX_THREAD_STARTED);
	psm2_error_t err = PSM2_OK_NO_PROGRESS;
	psm2_error_t err2;

	if (!psm3_gen1_recvhdrq_isempty(&ptl->recvq)) {
		if (do_lock && !ips_recvhdrq_trylock(&ptl->recvq))
			return err;
		if (ptl->recvq.proto->flags & IPS_PROTO_FLAG_CCA_PRESCAN) {
			psm3_gen1_recvhdrq_scan_cca(&ptl->recvq);
		}
		err = psm3_gen1_recvhdrq_progress(&ptl->recvq);
		if (do_lock)
			ips_recvhdrq_unlock(&ptl->recvq);
		if_pf(err > PSM2_OK_NO_PROGRESS)
		    return err;
		err2 =
		    psmi_timer_process_if_expired(&(ptl->timerq),
						  current_count);
		if (err2 != PSM2_OK_NO_PROGRESS)
			return err2;
		else
			return err;
	}

	/*
	 * Process timer expirations after servicing receive queues (some packets
	 * may have been acked, some requests-to-send may have been queued).
	 *
	 * It's safe to look at the timer without holding the lock because it's not
	 * incorrect to be wrong some of the time.
	 */
	if (psmi_timer_is_expired(&(ptl->timerq), current_count)) {
		if (do_lock)
			ips_recvhdrq_lock(&ptl->recvq);
		err = psm3_timer_process_expired(&(ptl->timerq), current_count);
		if (do_lock)
			ips_recvhdrq_unlock(&ptl->recvq);
	}

	return err;
}

PSMI_INLINE(int psm3_gen1_ips_try_lock_shared_context(struct gen1_ptl_shared *recvshc))
{
	return pthread_spin_trylock(recvshc->context_lock);
}
/* Unused
PSMI_INLINE(void psm3_gen1_ips_lock_shared_context(struct gen1_ptl_shared *recvshc))
{
	pthread_spin_lock(recvshc->context_lock);
}
*/
PSMI_INLINE(void psm3_gen1_ips_unlock_shared_context(struct gen1_ptl_shared *recvshc))
{
	pthread_spin_unlock(recvshc->context_lock);
}

psm2_error_t psm3_gen1_ips_ptl_shared_poll(ptl_t *ptl_gen, int _ignored)
{
	struct ptl_ips *ptl = (struct ptl_ips *)ptl_gen;
	const uint64_t current_count = get_cycles();
	psm2_error_t err = PSM2_OK_NO_PROGRESS;
	psm2_error_t err2;
	struct gen1_ptl_shared *recvshc = ptl->recvshc;
	psmi_assert(recvshc != NULL);

	/* The following header queue checks are speculative (but safe)
	 * until this process has acquired the lock. The idea is to
	 * minimize lock contention due to processes spinning on the
	 * shared context. */
	if (psm3_gen1_recvhdrq_isempty(&recvshc->recvq)) {
		if (!psm3_gen1_recvhdrq_isempty(&ptl->recvq) &&
		    psm3_gen1_ips_try_lock_shared_context(recvshc) == 0) {
			/* check that subcontext is empty while under lock to avoid
			 * re-ordering of incoming packets (since packets from
			 * hardware context will be processed immediately). */
			if_pt(psm3_gen1_recvhdrq_isempty(&recvshc->recvq)) {
				if (ptl->recvq.proto->flags & IPS_PROTO_FLAG_CCA_PRESCAN) {
					psm3_gen1_recvhdrq_scan_cca(&ptl->recvq);
				}
				err = psm3_gen1_recvhdrq_progress(&ptl->recvq);
			}
			psm3_gen1_ips_unlock_shared_context(recvshc);
		}
	}

	if_pf(err > PSM2_OK_NO_PROGRESS)
	    return err;

	if (!psm3_gen1_recvhdrq_isempty(&recvshc->recvq)) {
		if (recvshc->recvq.proto->flags & IPS_PROTO_FLAG_CCA_PRESCAN) {
			psm3_gen1_recvhdrq_scan_cca(&recvshc->recvq);
		}
		err2 = psm3_gen1_recvhdrq_progress(&recvshc->recvq);
		if (err2 != PSM2_OK_NO_PROGRESS) {
			err = err2;
		}
	}

	if_pf(err > PSM2_OK_NO_PROGRESS)
	    return err;

	/*
	 * Process timer expirations after servicing receive queues (some packets
	 * may have been acked, some requests-to-send may have been queued).
	 */
	err2 = psmi_timer_process_if_expired(&(ptl->timerq), current_count);
	if (err2 != PSM2_OK_NO_PROGRESS)
		err = err2;

	return err;
}

int psm3_gen1_ips_ptl_recvq_isempty(const ptl_t *ptl_gen)
{
	struct ptl_ips *ptl = (struct ptl_ips *)ptl_gen;
	struct gen1_ptl_shared *recvshc = ptl->recvshc;

	if (recvshc != NULL && !psm3_gen1_recvhdrq_isempty(&recvshc->recvq))
		return 0;
	return psm3_gen1_recvhdrq_isempty(&ptl->recvq);
}

static psm2_error_t
psm3_gen1_ips_ptl_writehdrq_init(const psmi_context_t *context,
		   struct gen1_ips_writehdrq *writeq,
		   struct gen1_ips_writehdrq_state *state,
		   uint32_t subcontext)
{
	uint32_t elemsz = psm3_gen1_get_rx_hdr_q_ent_size(context->psm_hw_ctxt),
		 elemcnt = psm3_gen1_get_rx_hdr_q_cnt(context->psm_hw_ctxt);

	memset(writeq, 0, sizeof(*writeq));
	writeq->context = context;
	writeq->state = state;
	writeq->hdrq_elemlast = (elemcnt - 1) * (elemsz >> BYTE2DWORD_SHIFT);

	writeq->state->enabled = 1;
	return PSM2_OK;
}

static psm2_error_t psm3_gen1_shrecvq_init(ptl_t *ptl_gen, const psmi_context_t *context)
{
	struct ptl_ips *ptl = (struct ptl_ips *)ptl_gen;
	struct gen1_ptl_shared *recvshc = ptl->recvshc;
	struct ips_recvhdrq_callbacks recvq_callbacks;
	psm2_error_t err = PSM2_OK;
	int i;

	/* Initialize (shared) hardware context recvq (ptl->recvq) */
	/* NOTE: uses recvq in ptl structure for shared h/w context */
	recvq_callbacks.callback_packet_unknown = psm3_gen1_ips_ptl_process_unknown;
	recvq_callbacks.callback_subcontext = psm3_gen1_ips_subcontext_process;
	recvq_callbacks.callback_error = psm3_gen1_ips_ptl_process_packet_error;
	if ((err = psm3_gen1_recvhdrq_init(context, &ptl->epstate, &ptl->proto,
				     &recvq_callbacks,
				     recvshc->subcontext,
				     &ptl->recvq,
				     &recvshc->hwcontext_ctrl->recvq_state,
				     PSM3_GEN1_CL_Q_RX_HDR_Q))) {
		goto fail;
	}

	/* Initialize software subcontext (recvshc->recvq). Subcontexts do */
	/* not require the rcvhdr copy feature. */
	recvq_callbacks.callback_subcontext = psm3_gen1_ips_subcontext_ignore;
	if ((err = psm3_gen1_recvhdrq_init(context, &ptl->epstate, &ptl->proto,
				     &recvq_callbacks,
				     recvshc->subcontext,
				     &recvshc->recvq, &recvshc->recvq_state,
				     PSM3_GEN1_GET_SC_CL_Q_RX_HDR_Q(recvshc->subcontext)))) {
		goto fail;
	}

	/* Initialize each recvshc->writeq for shared contexts */
	for (i = 0; i < recvshc->subcontext_cnt; i++) {
		if ((err = psm3_gen1_ips_ptl_writehdrq_init(context,
					      &recvshc->writeq[i],
					      &recvshc->subcontext_ureg[i]->
					      writeq_state,
					      i))) {
			goto fail;
		}
	}

	if (err == PSM2_OK)
		_HFI_DBG
		    ("Context sharing in use: %s, context %d, sub-context %d\n",
		     psm3_epid_fmt_addr(ptl->epid, 0), recvshc->context,
		     recvshc->subcontext);
fail:
	return err;
}

static psm2_error_t psm3_gen1_shrecvq_fini(ptl_t *ptl_gen)
{
	struct ptl_ips *ptl = (struct ptl_ips *)ptl_gen;
	psm2_error_t err = PSM2_OK;
	int i;

	/* disable my write header queue before deallocation */
	i = ptl->recvshc->subcontext;
	ptl->recvshc->subcontext_ureg[i]->writeq_state.enabled = 0;
	psmi_free(ptl->recvshc);
	return err;
}


#ifdef PSM2_MOCK_TESTING
void psm3_gen1_ips_ptl_non_dw_mul_sdma_init(void)
{
	uint16_t major_version = psm3_gen1_get_user_major_version();
	uint16_t minor_version = psm3_gen1_get_user_minor_version();
	int allow_non_dw_mul = 0;

	if ((major_version > HFI1_USER_SWMAJOR_NON_DW_MUL_MSG_SIZE_ALLOWED) ||
		((major_version == HFI1_USER_SWMAJOR_NON_DW_MUL_MSG_SIZE_ALLOWED) &&
		 (minor_version >= HFI1_USER_SWMINOR_NON_DW_MUL_MSG_SIZE_ALLOWED)))
	{
		allow_non_dw_mul = 1;
	}
	psm3_hal_current_hal_instance->params.cap_mask = 0;
	if (allow_non_dw_mul)
		psm3_hal_current_hal_instance->params.cap_mask |= PSM_HAL_CAP_NON_DW_MULTIPLE_MSG_SIZE;
}
#endif /* PSM2_MOCK_TESTING */

/* linux doesn't have strlcat; this is a stripped down implementation */
/* not super-efficient, but we use it rarely, and only for short strings */
/* not fully standards conforming! */
static size_t strlcat(char *d, const char *s, size_t l)
{
	int dlen = strlen(d), slen, max;
	if (l <= dlen)		/* bug */
		return l;
	slen = strlen(s);
	max = l - (dlen + 1);
	if (slen > max)
		slen = max;
	memcpy(d + dlen, s, slen);
	d[dlen + slen] = '\0';
	return dlen + slen + 1;	/* standard says to return full length, not actual */
}

void psm3_gen1_ips_ptl_dump_err_stats(struct ips_proto *proto)
{
	char err_stat_msg[2048];
	char tmp_buf[128];
	int len = sizeof(err_stat_msg);

	if (!(psm3_dbgmask & __HFI_PKTDBG))
		return;

	*err_stat_msg = '\0';

	if (proto->error_stats.num_icrc_err ||
	    proto->error_stats.num_ecc_err ||
	    proto->error_stats.num_len_err ||
	    proto->error_stats.num_tid_err ||
	    proto->error_stats.num_dc_err ||
	    proto->error_stats.num_dcunc_err ||
	    proto->error_stats.num_khdrlen_err) {

		snprintf(tmp_buf, sizeof(tmp_buf), "ERROR STATS: ");

		if (proto->error_stats.num_icrc_err) {
			snprintf(tmp_buf, sizeof(tmp_buf), "ICRC: %" PRIu64 " ",
				 proto->error_stats.num_icrc_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_ecc_err) {
			snprintf(tmp_buf, sizeof(tmp_buf), "ECC: %" PRIu64 " ",
				 proto->error_stats.num_ecc_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_len_err) {
			snprintf(tmp_buf, sizeof(tmp_buf), "LEN: %" PRIu64 " ",
				 proto->error_stats.num_len_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_tid_err) {
			snprintf(tmp_buf, sizeof(tmp_buf), "TID: %" PRIu64 " ",
				 proto->error_stats.num_tid_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_dc_err) {
			snprintf(tmp_buf, sizeof(tmp_buf), "DC: %" PRIu64 " ",
				 proto->error_stats.num_dc_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_dcunc_err) {
			snprintf(tmp_buf, sizeof(tmp_buf),
				 "DCUNC: %" PRIu64 " ",
				 proto->error_stats.num_dcunc_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}

		if (proto->error_stats.num_khdrlen_err) {
			snprintf(tmp_buf, sizeof(tmp_buf),
				 "KHDRLEN: %" PRIu64 " ",
				 proto->error_stats.num_khdrlen_err);
			strlcat(err_stat_msg, tmp_buf, len);
		}
		strlcat(err_stat_msg, "\n", len);
	} else
		strlcat(err_stat_msg, "No previous errors.\n", len);

	_HFI_ERROR("%s", err_stat_msg);
}

int
psm3_gen1_ips_ptl_process_err_chk_gen(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_recvhdrq *recvq = (struct ips_recvhdrq *)rcv_ev->recvq;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	struct ips_protoexp *protoexp = recvq->proto->protoexp;
	struct ips_tid_recv_desc *tidrecvc;
	psmi_seqnum_t err_seqnum, recvseq;
	ptl_arg_t desc_id = p_hdr->data[0];
	ptl_arg_t send_desc_id = p_hdr->data[1];
	int16_t seq_off;
	uint8_t ack_type;
	ips_scb_t ctrlscb;

	INC_TIME_SPEND(TIME_SPEND_USER4);
	PSM2_LOG_MSG("entering");
	recvq->proto->epaddr_stats.err_chk_recv++;

	/* Ignore FECN bit since this is the control path */
	rcv_ev->is_congested &= ~IPS_RECV_EVENT_FECN;

	/* Get the flowgenseq for err chk gen */
	err_seqnum.psn_val = __be32_to_cpu(p_hdr->bth[2]);

	/* Get receive descriptor */
	psmi_assert(desc_id._desc_idx < HFI_TF_NFLOWS);
	tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

	if (tidrecvc->rdescid._desc_genc != desc_id._desc_genc) {
		/* Receive descriptor mismatch in time and space.
		 * Stale err chk gen, drop packet
		 */
		_HFI_DBG
		    ("ERR_CHK_GEN: gen mismatch Pkt: 0x%x, Current: 0x%x\n",
		     desc_id._desc_genc, tidrecvc->rdescid._desc_genc);
		PSM2_LOG_MSG("leaving");
		return IPS_RECVHDRQ_CONTINUE;
	}
	psmi_assert(tidrecvc->state == TIDRECVC_STATE_BUSY);

	/*
	 * We change tidrecvc->tidflow_genseq here only when a new generation
	 * is allocated and programmed into hardware. Otherwise we use local
	 * variable recvseq to create the reply.
	 */
	recvseq = tidrecvc->tidflow_genseq;

	/* Get the latest seq from hardware tidflow table. But
	 * only do this when context sharing is not used, because
	 * context sharing might drop packet even though hardware
	 * has received it successfully.
	 */
	if (!tidrecvc->context->tf_ctrl)
	{
		uint64_t tf;
		uint32_t seqno=0;

		psmi_hal_tidflow_get(tidrecvc->rdescid._desc_idx, &tf,
				     tidrecvc->context->psm_hw_ctxt);
		psmi_hal_tidflow_get_seqnum(tf, &seqno);
		recvseq.psn_seq = seqno;
	}

	if (err_seqnum.psn_gen != recvseq.psn_gen) {
		ack_type = OPCODE_NAK;
		/* NAK without allocating a new generation */

		/* My current generation and last received seq */
		ctrlscb.ips_lrh.data[1].u32w0 = recvseq.psn_val;
	 } else {
		/* Either lost packets or lost ack, we need to deal
		 * with wrap around of the seq value from 2047 to 0
		 * because seq is only 11 bits */
		seq_off = (int16_t)(err_seqnum.psn_seq - recvseq.psn_seq);
		if (seq_off < 0)
			seq_off += 2048; /* seq is 11 bits */

		if (seq_off < 1024) {
			ack_type = OPCODE_NAK;
			/* NAK with allocating a new generation */

			/* set latest seq */
			tidrecvc->tidflow_genseq.psn_seq = recvseq.psn_seq;
			/* allocate and set a new generation */
			ips_protoexp_flow_newgen(tidrecvc);
			/* get the new generation */
			recvseq.psn_gen = tidrecvc->tidflow_genseq.psn_gen;

			/* My new generation and last received seq */
			ctrlscb.ips_lrh.data[1].u32w0 = recvseq.psn_val;
		} else
			/* ACK with last received seq,
			 * no need to set ips_lrh.data[1].u32w0 */
			ack_type = OPCODE_ACK;
	}

	ctrlscb.scb_flags = 0;
	ctrlscb.ips_lrh.data[0].u64 = send_desc_id.u64;
	/* Keep peer generation but use my last received sequence */
	err_seqnum.psn_seq = recvseq.psn_seq;
	ctrlscb.ips_lrh.ack_seq_num = err_seqnum.psn_val;

	/* May want to generate a BECN if a lot of swapped generations */
	if_pf((tidrecvc->tidflow_nswap_gen > 4) &&
	      (protoexp->proto->flags & IPS_PROTO_FLAG_CCA)) {
		_HFI_CCADBG
		    ("ERR_CHK_GEN: Generating BECN. Number of swapped generations: %d.\n",
		     tidrecvc->tidflow_nswap_gen);
		/* Mark flow to generate BECN in control packet */
		tidrecvc->tidflow.flags |= IPS_FLOW_FLAG_GEN_BECN;

		/* Update stats for congestion encountered */
		recvq->proto->epaddr_stats.congestion_pkts++;
	}

	// no payload, pass cksum so non-NULL
	psm3_ips_proto_send_ctrl_message(&tidrecvc->tidflow,
				    ack_type, &tidrecvc->ctrl_msg_queued,
				    &ctrlscb, ctrlscb.cksum, 0);

	/* Update stats for expected window */
	tidrecvc->stats.nErrChkReceived++;
	if (ack_type == OPCODE_NAK)
		tidrecvc->stats.nReXmit++;	/* Update stats for retransmit (Sent a NAK) */

	PSM2_LOG_MSG("leaving");
	return IPS_RECVHDRQ_CONTINUE;
}

int
psm3_gen1_ips_ptl_process_becn(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_proto *proto = rcv_ev->proto;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	ips_epaddr_t *ipsaddr = rcv_ev->ipsaddr;
	int flowid = ips_proto_flowid(p_hdr);
	struct ips_flow *flow;

	psmi_assert(flowid < EP_FLOW_LAST);
	flow = &ipsaddr->flows[flowid];
	if ((flow->path->opa.pr_ccti +
	proto->cace[flow->path->pr_sl].ccti_increase) <= proto->ccti_limit) {
		ips_cca_adjust_rate(flow->path,
			    proto->cace[flow->path->pr_sl].ccti_increase);
		/* Clear congestion event */
		rcv_ev->is_congested &= ~IPS_RECV_EVENT_BECN;
	}

	return IPS_RECVHDRQ_CONTINUE;
}

int psm3_gen1_ips_ptl_process_unknown(const struct ips_recvhdrq_event *rcv_ev)
{
	int opcode;
	struct ips_proto *proto = rcv_ev->proto;
	psm2_ep_t ep_err;
	char *pkt_type;

	if (0 == psm3_ips_proto_process_unknown(rcv_ev, &opcode))
		return IPS_RECVHDRQ_CONTINUE;

	// truely an unknown remote node, psm3_ips_proto_process_unknown already
	// did generic output and debug packet dumps
	// now output the final HAL specific error message
	psm3_gen1_ips_ptl_dump_err_stats(proto);

	/* Other messages are definitely crosstalk. */
	/* out-of-context expected messages are always fatal */
	if (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EXPECTED) {
		ep_err = PSMI_EP_NORETURN;
		pkt_type = "expected";
	} else if (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EAGER) {
		ep_err = PSMI_EP_LOGEVENT;
		pkt_type = "eager";
	} else {
		ep_err = PSMI_EP_NORETURN;
		pkt_type = "unknown";
	}

	/* At this point we are out of luck. */
	psm3_handle_error(ep_err, PSM2_EPID_NETWORK_ERROR,
			  "Received %s message(s) ptype=0x%x opcode=%x"
			  " from an unknown process", pkt_type, psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf), opcode);

	/* Always skip this packet unless the above call was a noreturn call */
	return IPS_RECVHDRQ_CONTINUE;
}

/* decode RHF errors; only used one place now, may want more later */
static void get_rhf_errstring(uint32_t err, char *msg, size_t len)
{
	*msg = '\0';		/* if no errors, and so don't need to check what's first */

	if (err & PSM3_GEN1_RHF_ERR_ICRC)
		strlcat(msg, "icrcerr ", len);
	if (err & PSM3_GEN1_RHF_ERR_ECC)
		strlcat(msg, "eccerr ", len);
	if (err & PSM3_GEN1_RHF_ERR_LEN)
		strlcat(msg, "lenerr ", len);
	if (err & PSM3_GEN1_RHF_ERR_TID)
		strlcat(msg, "tiderr ", len);
	if (err & PSM3_GEN1_RHF_ERR_DC)
		strlcat(msg, "dcerr ", len);
	if (err & PSM3_GEN1_RHF_ERR_DCUN)
		strlcat(msg, "dcuncerr ", len);
	if (err & PSM3_GEN1_RHF_ERR_KHDRLEN)
		strlcat(msg, "khdrlenerr ", len);
}

/* get the error string as a number and a string */
static void rhf_errnum_string(char *msg, size_t msglen, long err)
{
	int len;
	char *errmsg;

	len = snprintf(msg, msglen, "RHFerror %lx: ", err);
	if (len > 0 && len < msglen) {
		errmsg = msg + len;
		msglen -= len;
	} else
		errmsg = msg;
	*errmsg = 0;
	get_rhf_errstring(err, errmsg, msglen);
}

static void
psm3_gen1_ptl_ips_protoexp_handle_tiderr(const struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_tid_recv_desc *tidrecvc;
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;

	ptl_arg_t desc_id;
	int tidpair = (__le32_to_cpu(p_hdr->khdr.kdeth0) >>
		   HFI_KHDR_TID_SHIFT) & HFI_KHDR_TID_MASK;
	int tidctrl = (__le32_to_cpu(p_hdr->khdr.kdeth0) >>
		   HFI_KHDR_TIDCTRL_SHIFT) & HFI_KHDR_TIDCTRL_MASK;
	int tid0, tid1, tid;

	psmi_assert(_get_proto_hfi_opcode(p_hdr) == OPCODE_EXPTID);

	/* Expected sends not enabled */
	if (protoexp == NULL)
		return;

	/* Not doing extra tid debugging or not really a tiderr */
	if (!(protoexp->tid_flags & IPS_PROTOEXP_FLAG_TID_DEBUG) ||
	    !(psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) & PSM3_GEN1_RHF_ERR_TID))
		return;

	if (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) != PSM3_GEN1_RHF_RX_TYPE_EXPECTED) {
		_HFI_ERROR("receive type %d is not "
			   "expected in tid debugging\n", psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf));
		return;
	}

	desc_id._desc_idx = ips_proto_flowid(p_hdr);
	desc_id._desc_genc = p_hdr->exp_rdescid_genc;

	tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

	if (tidctrl != 3)
		tid0 = tid1 = tidpair * 2 + tidctrl - 1;
	else {
		tid0 = tidpair * 2;
		tid1 = tid0 + 1;
	}

	for (tid = tid0; tid <= tid1; tid++) {
		if (protoexp->tid_info[tid].state == TIDSTATE_USED)
			continue;

		char buf[128];
		char *s = "invalid (not even in table)";

		if (tidrecvc->rdescid._desc_genc ==
				    desc_id._desc_genc)
			s = "valid";
		else {
			snprintf(buf, sizeof(buf) - 1,
				 "wrong generation (gen=%d,received=%d)",
				 tidrecvc->rdescid._desc_genc,
				 desc_id._desc_genc);
			buf[sizeof(buf) - 1] = '\0';
			s = buf;
		}

		if (protoexp->tid_info[tid].tidrecvc != tidrecvc) {
			_HFI_ERROR
			    ("tid %d not a known member of tidsess %d\n",
			     tid, desc_id._desc_idx);
		}

		_HFI_ERROR("tid %d is marked unused (session=%d): %s\n", tid,
			   desc_id._desc_idx, s);
	}
	return;
}

static void
psm3_gen1_ptl_ips_protoexp_handle_data_err(const struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_tid_recv_desc *tidrecvc;
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	int hdr_err = psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) & PSM3_GEN1_RHF_ERR_KHDRLEN;
	uint8_t op_code = _get_proto_hfi_opcode(p_hdr);
	char pktmsg[128];
	char errmsg[256];

	psmi_assert(_get_proto_hfi_opcode(p_hdr) == OPCODE_EXPTID);

	/* Expected sends not enabled */
	if (protoexp == NULL)
		return;

	get_rhf_errstring(psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf), pktmsg,
				    sizeof(pktmsg));

	snprintf(errmsg, sizeof(errmsg),
		 "%s pkt type opcode 0x%x at hd=0x%x %s\n",
		 (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EAGER) ? "Eager" :
		 (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EXPECTED) ? "Expected" :
		 (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_NON_KD) ? "Non-kd" :
		 "<Error>", op_code, rcv_ev->recvq->state->hdrq_head,
		 pktmsg);

	if (!hdr_err) {
		ptl_arg_t desc_id;
		psmi_seqnum_t sequence_num;

		desc_id._desc_idx = ips_proto_flowid(p_hdr);
		desc_id._desc_genc = p_hdr->exp_rdescid_genc;

		tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

		if (tidrecvc->rdescid._desc_genc != desc_id._desc_genc) {
			/* Print this at very verbose level. Noisy links can have a few of
			 * these! */
			_HFI_VDBG
			    ("Data Error Pkt and Recv Generation Mismatch: %s",
			     errmsg);
			return;	/* skip */
		}

		if (tidrecvc->state == TIDRECVC_STATE_FREE) {
			_HFI_EPDBG
			    ("Data Error Pkt for a Completed Rendezvous: %s",
			     errmsg);
			return;	/* skip */
		}

		/* See if CRC error for a previous packet */
		sequence_num.psn_val = __be32_to_cpu(p_hdr->bth[2]);
		if (sequence_num.psn_gen == tidrecvc->tidflow_genseq.psn_gen) {
			/* Try to recover the flow by restarting from previous known good
			 * sequence (possible if the packet with CRC error is after the "known
			 * good PSN" else we can't restart the flow.
			 */
			return ips_protoexp_do_tf_seqerr(protoexp,
					tidrecvc, p_hdr);
		} else {
			/* Print this at very verbose level */
			_HFI_VDBG
			    ("Data Error Packet. GenMismatch: Yes. Tidrecvc: %p. "
			     "Pkt Gen.Seq: %d.%d, TF Gen.Seq: %d.%d. %s\n",
			     tidrecvc, sequence_num.psn_gen,
			     sequence_num.psn_seq,
			     tidrecvc->tidflow_genseq.psn_gen,
			     tidrecvc->tidflow_genseq.psn_seq, errmsg);
		}

	} else {
		_HFI_VDBG("HDR_ERROR: %s\n", errmsg);
	}

}

static void
psm3_gen1_ptl_ips_protoexp_handle_tf_seqerr(const struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	struct ips_tid_recv_desc *tidrecvc;
	ptl_arg_t desc_id;

	psmi_assert_always(protoexp != NULL);
	psmi_assert(_get_proto_hfi_opcode(p_hdr) == OPCODE_EXPTID);

	desc_id._desc_idx = ips_proto_flowid(p_hdr);
	desc_id._desc_genc = p_hdr->exp_rdescid_genc;

	tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

	if (tidrecvc->rdescid._desc_genc == desc_id._desc_genc
			&& tidrecvc->state == TIDRECVC_STATE_BUSY)
		ips_protoexp_do_tf_seqerr(protoexp, tidrecvc, p_hdr);

	return;
}

static void
psm3_gen1_ptl_ips_protoexp_handle_tf_generr(const struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	struct ips_tid_recv_desc *tidrecvc;
	ptl_arg_t desc_id;

	psmi_assert_always(protoexp != NULL);
	psmi_assert(_get_proto_hfi_opcode(p_hdr) == OPCODE_EXPTID);

	/* For a generation error our NAK crossed on the wire or this is a stale
	 * packet. Error recovery should sync things up again. Just drop this
	 * packet.
	 */
	desc_id._desc_idx = ips_proto_flowid(p_hdr);
	desc_id._desc_genc = p_hdr->exp_rdescid_genc;

	tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

	if (tidrecvc->rdescid._desc_genc == desc_id._desc_genc
			&& tidrecvc->state == TIDRECVC_STATE_BUSY)
		ips_protoexp_do_tf_generr(protoexp, tidrecvc, p_hdr);

	return;
}

/*
 * Error handling
 */
int psm3_gen1_ips_ptl_process_packet_error(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_proto *proto = rcv_ev->proto;
	int pkt_verbose_err = psm3_dbgmask & __HFI_PKTDBG;
	int tiderr    = psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) & PSM3_GEN1_RHF_ERR_TID;
	int tf_seqerr = psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) & PSM3_GEN1_RHF_ERR_TFSEQ;
	int tf_generr = psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) & PSM3_GEN1_RHF_ERR_TFGEN;
	int data_err  = psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf) &
	    (PSM3_GEN1_RHF_ERR_ICRC | PSM3_GEN1_RHF_ERR_ECC | PSM3_GEN1_RHF_ERR_LEN |
	     PSM3_GEN1_RHF_ERR_DC | PSM3_GEN1_RHF_ERR_DCUN | PSM3_GEN1_RHF_ERR_KHDRLEN);
	char pktmsg[128];

	*pktmsg = 0;
	/*
	 * Tid errors on eager pkts mean we get a headerq overflow, perfectly
	 * safe.  Tid errors on expected or other packets means trouble.
	 */
	if (tiderr && psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EAGER) {
		struct ips_message_header *p_hdr = rcv_ev->p_hdr;

		/* Payload dropped - Determine flow for this header and see if
		 * we need to generate a NAK.
		 *
		 * ALL PACKET DROPS IN THIS CATEGORY CAN BE FLAGGED AS DROPPED DUE TO
		 * CONGESTION AS THE EAGER BUFFER IS FULL.
		 *
		 * Possible eager packet type:
		 *
		 * Ctrl Message - ignore
		 * MQ message - Can get flow and see if we need to NAK.
		 * AM message - Can get flow and see if we need to NAK.
		 */

		proto->stats.hdr_overflow++;
		if (data_err)
			return 0;

		switch (_get_proto_hfi_opcode(p_hdr)) {
		case OPCODE_TINY:
		case OPCODE_SHORT:
		case OPCODE_EAGER:
		case OPCODE_LONG_RTS:
		case OPCODE_LONG_CTS:
		case OPCODE_LONG_DATA:
		case OPCODE_AM_REQUEST:
		case OPCODE_AM_REQUEST_NOREPLY:
		case OPCODE_AM_REPLY:
			{
				ips_epaddr_flow_t flowid =
				    ips_proto_flowid(p_hdr);
				struct ips_epstate_entry *epstaddr;
				struct ips_flow *flow;
				psmi_seqnum_t sequence_num;
				int16_t diff;

				/* Obtain ipsaddr for packet */
				epstaddr =
				    ips_epstate_lookup(rcv_ev->recvq->epstate,
						       rcv_ev->p_hdr->connidx);
				if_pf(epstaddr == NULL
				      || epstaddr->ipsaddr == NULL)
				    return 0;	/* Unknown packet - drop */

				rcv_ev->ipsaddr = epstaddr->ipsaddr;

				psmi_assert(flowid < EP_FLOW_LAST);
				flow = &rcv_ev->ipsaddr->flows[flowid];
				sequence_num.psn_val =
				    __be32_to_cpu(p_hdr->bth[2]);
				diff =
				    (int16_t) (sequence_num.psn_num -
					       flow->recv_seq_num.psn_num);

				if (diff >= 0
				    && !(flow->
					 flags & IPS_FLOW_FLAG_NAK_SEND)) {
					/* Mark flow as congested and attempt to generate NAK */
					flow->flags |= IPS_FLOW_FLAG_GEN_BECN;
					proto->epaddr_stats.congestion_pkts++;

					flow->flags |= IPS_FLOW_FLAG_NAK_SEND;
					flow->cca_ooo_pkts = 0;
					ips_proto_send_nak((struct ips_recvhdrq
							    *)rcv_ev->recvq,
							   flow);
				}

				/* Safe to process ACKs from header */
				psm3_ips_proto_process_ack(rcv_ev);
			}
			break;
		case OPCODE_EXPTID:
			/* If RSM is matching packets that are TID&FECN&SH,
			 * it is possible to have a EXPTID packet encounter
			 * the eager full condition and have the payload
			 * dropped (but the header delivered).
			 * Treat this condition as a data error (corruption,etc)
			 * and send a NAK.
			 */
			if (psmi_hal_has_cap(PSM_HAL_CAP_RSM_FECN_SUPP))
				psm3_gen1_ptl_ips_protoexp_handle_data_err(rcv_ev);
			break;
		default:
			break;
		}
	} else if (tf_generr) /* handle generr, ignore tiderr if any */
		psm3_gen1_ptl_ips_protoexp_handle_tf_generr(rcv_ev);
	else if (tf_seqerr)
		psm3_gen1_ptl_ips_protoexp_handle_tf_seqerr(rcv_ev);
	else if (tiderr) {	/* tid error, but not on an eager pkt */
		psm2_ep_t ep_err = PSMI_EP_LOGEVENT;
		uint16_t tid, offset;
		uint64_t t_now = get_cycles();

		proto->tiderr_cnt++;

		/* Whether and how we will be logging this event */
		if (proto->tiderr_max > 0
		    && proto->tiderr_cnt >= proto->tiderr_max)
			ep_err = PSMI_EP_NORETURN;
		else if (proto->tiderr_warn_interval != UINT64_MAX &&
			 proto->tiderr_tnext <= t_now)
			proto->tiderr_tnext =
			    get_cycles() + proto->tiderr_warn_interval;
		else
			ep_err = NULL;

		if (ep_err != NULL) {
			rhf_errnum_string(pktmsg, sizeof(pktmsg),
					  psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf));

			tid = (__le32_to_cpu(rcv_ev->p_hdr->khdr.kdeth0) >>
			       HFI_KHDR_TID_SHIFT) & HFI_KHDR_TID_MASK;
			offset = __le32_to_cpu(rcv_ev->p_hdr->khdr.kdeth0) &
			    HFI_KHDR_OFFSET_MASK;

			psm3_handle_error(ep_err, PSM2_EP_DEVICE_FAILURE,
					  "%s with tid=%d,offset=%d,count=%d: %s %s",
					  "TID Error",
					  tid, offset, proto->tiderr_cnt,
					  pktmsg, ep_err == PSMI_EP_NORETURN ?
					  "(Terminating...)" : "");
		}

		psm3_gen1_ptl_ips_protoexp_handle_tiderr(rcv_ev);
	} else if (data_err) {
#if _HFI_DEBUGGING
		if (_HFI_DBG_ON) {
			uint8_t op_code
				= _get_proto_hfi_opcode(rcv_ev->p_hdr);

			if (!pkt_verbose_err) {
				rhf_errnum_string(pktmsg, sizeof(pktmsg),
						  psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf));
				_HFI_DBG_ALWAYS
					("Error %s pkt type opcode 0x%x at hd=0x%x %s\n",
					 (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) ==
					  PSM3_GEN1_RHF_RX_TYPE_EAGER) ? "eager" : (
						  psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) ==
						  PSM3_GEN1_RHF_RX_TYPE_EXPECTED)
					 ? "expected" : (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) ==
							 PSM3_GEN1_RHF_RX_TYPE_NON_KD) ? "non-kd" :
					 "<error>", op_code,
					 rcv_ev->recvq->state->hdrq_head, pktmsg);
			}
		}
#endif

		if (psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf) == PSM3_GEN1_RHF_RX_TYPE_EXPECTED)
			psm3_gen1_ptl_ips_protoexp_handle_data_err(rcv_ev);
	} else {		/* not a tid or data error -- some other error */
#if _HFI_DEBUGGING
		if (_HFI_DBG_ON) {
			uint8_t op_code =
				__be32_to_cpu(rcv_ev->p_hdr->bth[0]) >> 24 & 0xFF;

			if (!pkt_verbose_err)
				rhf_errnum_string(pktmsg, sizeof(pktmsg),
						  psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf));

			/* else RHFerr decode printed below */
			_HFI_DBG_ALWAYS
				("Error pkt type 0x%x opcode 0x%x at hd=0x%x %s\n",
				 psm3_gen1_rhf_get_rx_type(rcv_ev->gen1_rhf), op_code,
				 rcv_ev->recvq->state->hdrq_head, pktmsg);
		}
#endif
	}
	if (pkt_verbose_err) {
		if (!*pktmsg)
			rhf_errnum_string(pktmsg, sizeof(pktmsg),
					  psm3_gen1_rhf_get_all_err_flags(rcv_ev->gen1_rhf));
		psm3_ips_proto_show_header(rcv_ev->p_hdr, pktmsg);
	}

	return 0;
}

static void psm3_gen1_gen_ipd_table(struct ips_proto *proto)
{
	uint8_t delay = 0, step = 1;
	/* Based on our current link rate setup the IPD table */
	memset(proto->ips_ipd_delay, 0xFF, sizeof(proto->ips_ipd_delay));

	/*
	 * Based on the starting rate of the link, we let the code to
	 * fall through to next rate without 'break' in the code. The
	 * decrement is doubled at each rate level...
	 */
	switch (proto->epinfo.ep_link_rate) {
	case PSM3_IBV_RATE_300_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_200_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_168_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_120_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
	case PSM3_IBV_RATE_112_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
	case PSM3_IBV_RATE_100_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_100_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_80_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_80_GBPS] = delay;
	case PSM3_IBV_RATE_60_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_60_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_40_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_40_GBPS] = delay;
	case PSM3_IBV_RATE_30_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_30_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_25_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_25_GBPS] = delay;
	case PSM3_IBV_RATE_20_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_20_GBPS] = delay;
		delay += step;
		step *= 2;
	case PSM3_IBV_RATE_10_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_10_GBPS] = delay;
	case PSM3_IBV_RATE_5_GBPS:
		proto->ips_ipd_delay[PSM3_IBV_RATE_5_GBPS] = delay;
	default:
		break;
	}
}

static psm2_error_t psm3_gen1_gen_cct_table(struct ips_proto *proto)
{
	psm2_error_t err = PSM2_OK;
	uint32_t cca_divisor, ipdidx, ipdval = 1;
	uint16_t *cct_table;

	/* The CCT table is static currently. If it's already created then return */
	if (proto->cct)
		goto fail;

	/* Allocate the CCT table */
	cct_table = psmi_calloc(proto->ep, UNDEFINED,
				proto->ccti_size, sizeof(uint16_t));
	if (!cct_table) {
		err = PSM2_NO_MEMORY;
		goto fail;
	}

	if (proto->ccti_size)
	{
		/* The first table entry is always 0 i.e. no IPD delay */
		cct_table[0] = 0;
	}

	/* Generate the remaining CCT table entries */
	for (ipdidx = 1; ipdidx < proto->ccti_size; ipdidx += 4, ipdval++)
		for (cca_divisor = 0; cca_divisor < 4; cca_divisor++) {
			if ((ipdidx + cca_divisor) == proto->ccti_size)
				break;
			cct_table[ipdidx + cca_divisor] =
			    (((cca_divisor ^ 0x3) << CCA_DIVISOR_SHIFT) |
			     (ipdval & 0x3FFF));
			_HFI_CCADBG("CCT[%d] = %x. Divisor: %x, IPD: %x\n",
				  ipdidx + cca_divisor,
				  cct_table[ipdidx + cca_divisor],
				  (cct_table[ipdidx + cca_divisor] >>
				   CCA_DIVISOR_SHIFT),
				  cct_table[ipdidx +
					    cca_divisor] & CCA_IPD_MASK);
		}

	/* On link up/down CCT is re-generated. If CCT table is previously created
	 * free it
	 */
	if (proto->cct) {
		psmi_free(proto->cct);
		proto->cct = NULL;
	}

	/* Update to the new CCT table */
	proto->cct = cct_table;

fail:
	return err;
}

// Fetch current link state to update linkinfo fields in ips_proto:
// 	ep_base_lid, ep_lmc, ep_link_rate, QoS tables, CCA tables
// These are all fields which can change during a link bounce.
// Note "active" state is not adjusted as on link down PSM will wait for
// the link to become usable again so it's always a viable/active device
// afer initial PSM startup has selected devices.
// Called during initialization of ips_proto during ibta_init as well
// as during a link bounce.
// TBD - may be able to call this from HAL ips_proto_init as well as
// directly within HAL event processing, in which case this could
// be completely internal to HAL and not exposed in HAL API
psm2_error_t psm3_gen1_ptl_ips_update_linkinfo(struct ips_proto *proto)
{
	psm2_error_t err = PSM2_OK;
	uint16_t lid;
	int ret;
	uint64_t link_speed;

	/* Get base lid, lmc and rate as these may have changed if the link bounced */
	// for Ethernet LID of 1 is returned
	lid = psm3_epid_lid(proto->ep->context.epid);
	proto->epinfo.ep_base_lid = __cpu_to_be16(lid);

	if ((ret = psm3_gen1_get_port_lmc(proto->ep->unit_id,
					 proto->ep->portnum)) < 0) {
		err = psm3_handle_error(proto->ep, PSM2_EP_DEVICE_FAILURE,
					"Could not obtain LMC for unit %u:%u. Error: %s",
					proto->ep->unit_id, proto->ep->portnum, strerror(errno));
		goto fail;
	}
	proto->epinfo.ep_lmc = min(ret, IPS_MAX_PATH_LMC);

	if (psm3_hfp_gen1_get_port_speed(proto->ep->unit_id,
					  proto->ep->portnum, &link_speed) <
	    0) {
		err =
		    psm3_handle_error(proto->ep, PSM2_EP_DEVICE_FAILURE,
				      "Could obtain link rate for unit %u:%u. Error: %s",
				      proto->ep->unit_id, proto->ep->portnum, strerror(errno));
		goto fail;
	}
	proto->epinfo.ep_link_rate = ips_link_speed_to_enum(link_speed);

	/* Load the SL2SC2VL table */
	psm3_gen1_ips_ptl_init_sl2sc_table(proto);

	/* Regenerate new IPD table for the updated link rate. */
	psm3_gen1_gen_ipd_table(proto);

	/* Generate the CCT table.  */
	err = psm3_gen1_gen_cct_table(proto);

fail:
	return err;
}

#endif // PSM_OPA
