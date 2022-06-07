#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2017 Intel Corporation.

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

  Copyright(c) 2017 Intel Corporation.

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

#ifndef _PSM_HAL_GEN1_HAL_H
#define _PSM_HAL_GEN1_HAL_H

#include "psm_user.h"
#include "ips_proto.h"
#include "ips_proto_internal.h"
#include "gen1_spio.h"
#include "gen1_sdma.h"
#include "psm_mq_internal.h"
#include "gen1_user.h"
#include "gen1_ptl_ips_subcontext.h"

COMPILE_TIME_ASSERT(MAX_SHARED_CTXTS_MUST_MATCH, PSM_HAL_MAX_SHARED_CTXTS == HFI1_MAX_SHARED_CTXTS);

/* Private struct on a per-context basis. */
typedef struct _hfp_gen1_pc_private
{
	struct _hfi_ctrl	    *ctrl; /* driver opaque hfi_proto */
	psm3_gen1_cl_q_t            cl_qs[PSM3_GEN1_GET_SC_CL_Q_RX_EGR_Q(7) + 1];
	struct gen1_ips_hwcontext_ctrl  *hwcontext_ctrl;
	struct gen1_ips_subcontext_ureg *subcontext_ureg[HFI1_MAX_SHARED_CTXTS];
	struct psm3_gen1_spio	    spio_ctrl;
	struct hfi1_user_info_dep   user_info;
	uint16_t                    sc2vl[PSMI_N_SCS];
} hfp_gen1_pc_private;

/* declare the hfp_gen1_private struct */
typedef struct _hfp_gen1_private
{
	/* GEN1 specific data that are common to all contexts: */
	int      sdmahdr_req_size;
	int      dma_rtail;
	uint32_t hdrq_rhf_off;
} hfp_gen1_private_t;

/* declare hfp_gen1_t struct, (combines public psmi_hal_instance_t
   together with a private struct) */
typedef struct _hfp_gen1
{
	psmi_hal_instance_t phi;
	hfp_gen1_private_t  hfp_private;
} hfp_gen1_t;

static inline struct _hfp_gen1 *get_psm_gen1_hi(void)
{
	return (struct _hfp_gen1*) psm3_hal_current_hal_instance;
}

const char* psm3_gen1_identify(void);

static inline
uint32_t
psm3_gen1_get_ht(volatile uint64_t *ht_register)
{
	uint64_t res = *ht_register;
	ips_rmb();
	return (uint32_t)res;
}

void psm3_gen1_ips_ptl_dump_err_stats(struct ips_proto *proto);

static inline
void
psm3_gen1_set_ht(volatile uint64_t *ht_register, uint64_t new_ht)
{
	*ht_register = new_ht;
	return;
}

/* Getter for cl q head indexes: */
static inline psm3_gen1_cl_idx psm3_gen1_get_cl_q_head_index(
						   psm3_gen1_cl_q cl_q,
						   psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	return psm3_gen1_get_ht(psm_hw_ctxt->cl_qs[cl_q].cl_q_head);
}

/* Getter for cl q tail indexes: */
static inline psm3_gen1_cl_idx psm3_gen1_get_cl_q_tail_index(
						psm3_gen1_cl_q cl_q,
						psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	return psm3_gen1_get_ht(psm_hw_ctxt->cl_qs[cl_q].cl_q_tail);
}

/* Setter for cl q head indexes: */
static inline void psm3_gen1_set_cl_q_head_index(
							psm3_gen1_cl_idx idx,
							psm3_gen1_cl_q cl_q,
							psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	psm3_gen1_set_ht(psm_hw_ctxt->cl_qs[cl_q].cl_q_head, idx);
	return;
}

/* Setter for cl q tail indexes: */
static inline void psm3_gen1_set_cl_q_tail_index(
							psm3_gen1_cl_idx idx,
							psm3_gen1_cl_q cl_q,
							psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	psm3_gen1_set_ht(psm_hw_ctxt->cl_qs[cl_q].cl_q_tail, idx);
	return;
}

/* Indicate whether the cl q is empty.
   When this returns > 0 the cl q is empty.
   When this returns == 0, the cl q is NOT empty (there are packets in the
   circular list that are available to receive).
   When this returns < 0, an error occurred.
   the parameter should correspond to the head index of the
   cl q circular list. */
static inline int psm3_gen1_cl_q_empty(psm3_gen1_cl_idx head_idx,
				      psm3_gen1_cl_q cl_q,
				      psmi_hal_hw_context ctxt)
{
	if (!get_psm_gen1_hi()->hfp_private.dma_rtail)
	{
		hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
		psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];
		int seq = psm3_gen1_hdrget_seq(pcl_q->hdr_qe.hdrq_base_addr +
		     (head_idx + get_psm_gen1_hi()->hfp_private.hdrq_rhf_off));

		return (*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq != seq);
	}

	return (head_idx == psm3_gen1_get_cl_q_tail_index(cl_q, ctxt));
}

/* Returns expected sequence number for RHF. */
static inline int psm3_gen1_get_rhf_expected_sequence_number(unsigned int *pseqnum,
						psm3_gen1_cl_q cl_q,
						psmi_hal_hw_context ctxt)

{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];

	*pseqnum = *pcl_q->hdr_qe.p_rx_hdrq_rhf_seq;
	return PSM_HAL_ERROR_OK;
}

/* Sets expected sequence number for RHF. */
static inline int psm3_gen1_set_rhf_expected_sequence_number(unsigned int seqnum,
									 psm3_gen1_cl_q cl_q,
									 psmi_hal_hw_context ctxt)

{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];

	*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq = seqnum;
	return PSM_HAL_ERROR_OK;
}

/* Checks sequence number from RHF. Returns PSM_HAL_ERROR_OK if the sequence number is good
 * returns something else if the sequence number is bad. */
static inline int psm3_gen1_check_rhf_sequence_number(unsigned int seqno)
{
	return (seqno <= LAST_RHF_SEQNO) ?
		PSM_HAL_ERROR_OK :
		PSM_HAL_ERROR_GENERAL_ERROR;
}

static inline int      psm3_gen1_get_rx_egr_tid_cnt(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->ctxt_info.egrtids;
}

static inline int      psm3_gen1_get_rx_hdr_q_cnt(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->ctxt_info.rcvhdrq_cnt;
}

static inline int      psm3_gen1_get_rx_hdr_q_ent_size(psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->ctxt_info.rcvhdrq_entsize;
}

/* Retire the given head idx of the header q, and change *head_idx to point to the next
      entry, lastly set *empty to indicate whether the headerq is empty at the new
      head_idx. */
static inline int psm3_gen1_retire_hdr_q_entry(psm3_gen1_cl_idx *idx,
				       psm3_gen1_cl_q cl_q,
				       psmi_hal_hw_context ctxt,
				       uint32_t elemsz, uint32_t elemlast,
				       int *emptyp)
{
	psm3_gen1_cl_idx tmp = *idx + elemsz;
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];

	if (!get_psm_gen1_hi()->hfp_private.dma_rtail)
	{
		(*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq)++;
		if (*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq > LAST_RHF_SEQNO)
			*pcl_q->hdr_qe.p_rx_hdrq_rhf_seq = 1;
	}
	if_pf(tmp > elemlast)
		tmp = 0;
	*emptyp = psm3_gen1_cl_q_empty(tmp, cl_q, ctxt);
	*idx = tmp;
	return PSM_HAL_ERROR_OK;
}

static inline void psm3_gen1_get_ips_message_hdr(psm3_gen1_cl_idx idx,
					psm3_gen1_raw_rhf_t rhf,
					struct ips_message_header **imhp,
					psm3_gen1_cl_q cl_q,
					psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];
	uint32_t *pu32 = pcl_q->hdr_qe.hdrq_base_addr + (idx + psm3_gen1_hdrget_hdrq_offset((uint32_t *)&rhf));
	*imhp = (struct ips_message_header*)pu32;
}

static inline void psm3_gen1_get_rhf(psm3_gen1_cl_idx idx,
			    psm3_gen1_raw_rhf_t *rhfp,
			    psm3_gen1_cl_q cl_q,
			    psmi_hal_hw_context ctxt)

{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];
	uint32_t *pu32 = (pcl_q->hdr_qe.hdrq_base_addr +
			  (idx + get_psm_gen1_hi()->hfp_private.hdrq_rhf_off));
	*rhfp = *((psm3_gen1_raw_rhf_t*)pu32);
}

/* Deliver an eager buffer given the index.
 * If the index does not refer to a current egr buffer, get_egr_buff()
 * returns NULL.
 */
static inline void *psm3_gen1_get_egr_buff(psm3_gen1_cl_idx idx,
				   psm3_gen1_cl_q cl_q,
				   psmi_hal_hw_context ctxt)
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	psm3_gen1_cl_q_t *pcl_q = &psm_hw_ctxt->cl_qs[cl_q];
	return pcl_q->egr_buffs[idx];
}

/* Receive the raw rhf, decompose it, and then receive the ips_message_hdr. */
/* caller has already initialized rcv_ev->proto, rcv_ev->recvq,
 * and rcv_ev->gen1_hdr_q
 */
static inline int psm3_gen1_get_receive_event(psm3_gen1_cl_idx head_idx, psmi_hal_hw_context ctxt, int get_payload,
				      struct ips_recvhdrq_event *rcv_ev)
{
	psm3_gen1_get_rhf(head_idx, &rcv_ev->gen1_rhf.raw_rhf, rcv_ev->gen1_hdr_q, ctxt);

	/* here, we turn off the TFSEQ err bit if set: */
	rcv_ev->gen1_rhf.decomposed_rhf = rcv_ev->gen1_rhf.raw_rhf & (~(PSM3_GEN1_RHF_ERR_MASK_64(TFSEQ)));

	/* Now, get the lrh: */
	psm3_gen1_get_ips_message_hdr(head_idx, rcv_ev->gen1_rhf.raw_rhf, &rcv_ev->p_hdr,
						rcv_ev->gen1_hdr_q, ctxt);

	// TBD - OPA computed this for CCA scan too, but not needed
	// could put this within if get_payload below, but placed it here
	// to faithfully duplicate the original OPA algorithm
	rcv_ev->has_cksum = ((rcv_ev->proto->flags & IPS_PROTO_FLAG_CKSUM) &&
			(rcv_ev->p_hdr->flags & IPS_SEND_FLAG_PKTCKSUM));

	// for FECN/BECN scan we don't need payload_size nor payload
	// we are inline and caller passes a const, so this if test will
	// optimize out.
	if (get_payload) {
		/* Compromise for better HAL API. For OPA, payload_size is not
		 * needed for TINY messages, getting payload_size and len here
		 * adds a few instructions to message rate critical path, but
		 * allows all the HALs to consistently set rcv_ev->payload_size 
		 * and rcv_ev->payload in recvhdrq_progress and eliminates
		 * need for OPA specific ips_recvhdrq_event_paylen and
		 * payload functions.
		 */
		uint32_t cksum_len = rcv_ev->has_cksum ? PSM_CRC_SIZE_IN_BYTES : 0;

		rcv_ev->payload_size = psm3_gen1_rhf_get_packet_length(rcv_ev->gen1_rhf) -
		    (sizeof(struct ips_message_header) +
		     HFI_CRC_SIZE_IN_BYTES + cksum_len);
		/* PSM does not use bth0].PadCnt, it figures out real datalen other way */

		if (psm3_gen1_rhf_get_use_egr_buff(rcv_ev->gen1_rhf))
			rcv_ev->payload = (uint8_t*)(psm3_gen1_get_egr_buff(
				psm3_gen1_rhf_get_egr_buff_index(rcv_ev->gen1_rhf),
				(psm3_gen1_cl_q)(rcv_ev->gen1_hdr_q + 1) /* The circular list q
						     (cl_q) for the egr buff for any rx
						     hdrq event is always one more than
						     the hdrq cl q */,
				rcv_ev->recvq->context->psm_hw_ctxt))+
				(psm3_gen1_rhf_get_egr_buff_offset(rcv_ev->gen1_rhf)*64);
		else
			rcv_ev->payload = NULL;
	}

	/* If the hdrq_head is before cachedlastscan, that means that we have
	 * already prescanned this for BECNs and FECNs, so we should not check
	 * again
	 */
	if_pt((rcv_ev->proto->flags & IPS_PROTO_FLAG_CCA) &&
	      (head_idx >= rcv_ev->recvq->state->hdrq_cachedlastscan)) {
		/* IBTA CCA handling:
		 * If FECN bit set handle IBTA CCA protocol. For the
		 * flow that suffered congestion we flag it to generate
		 * a control packet with the BECN bit set - This is
		 * currently an unsolicited ACK.
		 *
		 * For all MQ packets the FECN processing/BECN
		 * generation is done in the is_expected_or_nak
		 * function as each eager packet is inspected there.
		 *
		 * For TIDFLOW/Expected data transfers the FECN
		 * bit/BECN generation is done in protoexp_data. Since
		 * header suppression can result in even FECN packets
		 * being suppressed the expected protocol generated
		 * additional BECN packets if a "large" number of
		 * generations are swapped without progress being made
		 * for receive. "Large" is set empirically to 4.
		 *
		 * FECN packets are ignored for all control messages
		 * (except ACKs and NAKs) since they indicate
		 * congestion on the control path which is not rate
		 * controlled. The CCA specification allows FECN on
		 * ACKs to be disregarded as well.
		 */

		rcv_ev->is_congested =
			_is_cca_fecn_set(rcv_ev->
					 p_hdr) & IPS_RECV_EVENT_FECN;
		rcv_ev->is_congested |=
			(_is_cca_becn_set(rcv_ev->p_hdr) <<
			 (IPS_RECV_EVENT_BECN - 1));
	} else
		  rcv_ev->is_congested = 0;

	return PSM_HAL_ERROR_OK;
}

/* At the end of each scb struct, we have space reserved to accommodate
 * three structures (for GEN1)-
 * struct psm_hal_sdma_req_info, struct psm_hal_pbc and struct ips_message_header.
 * The HIC should get the size needed for the extended memory region
 * using a HAL call (psmi_hal_get_scb_extended_mem_size). For Gen1, this API
 * will return the size of the below struct psm_hal_gen1_scb_extended
 * aligned up to be able to fit struct psm_hal_pbc on a 64-byte boundary.
 */

#define PSMI_SHARED_CONTEXTS_ENABLED_BY_DEFAULT   1

struct psm_hal_gen1_scb_extended {
	union
	{
		struct sdma_req_info sri1;
		struct sdma_req_info_v6_3 sri2;
	};
	struct {
		struct psm_hal_pbc pbc;
		struct ips_message_header ips_lrh;
	} PSMI_CACHEALIGN;
};

static const struct
{
	uint32_t hfi1_event_bit, psmi_hal_hfi_event_bit;
} hfi1_events_map[] =
{
	{ HFI1_EVENT_FROZEN,		PSM_HAL_HFI_EVENT_FROZEN	},
	{ HFI1_EVENT_LINKDOWN,		PSM_HAL_HFI_EVENT_LINKDOWN	},
	{ HFI1_EVENT_LID_CHANGE,	PSM_HAL_HFI_EVENT_LID_CHANGE	},
	{ HFI1_EVENT_LMC_CHANGE,	PSM_HAL_HFI_EVENT_LMC_CHANGE	},
	{ HFI1_EVENT_SL2VL_CHANGE,	PSM_HAL_HFI_EVENT_SL2VL_CHANGE	},
	{ HFI1_EVENT_TID_MMU_NOTIFY,	PSM_HAL_HFI_EVENT_TID_MMU_NOTIFY},
};

psm2_error_t psm3_gen1_ips_ptl_init_pre_proto_init(struct ptl_ips *ptl);
psm2_error_t psm3_gen1_ips_ptl_init_post_proto_init(struct ptl_ips *ptl);
psm2_error_t psm3_gen1_ips_ptl_fini(struct ptl_ips *ptl);
void psm3_gen1_ips_ptl_init_sl2sc_table(struct ips_proto *proto);
psm2_error_t psm3_gen1_ptl_ips_update_linkinfo(struct ips_proto *proto);

psm2_error_t psm3_gen1_ips_ptl_pollintr(psm2_ep_t ep,
				struct ips_recvhdrq *recvq, int fd_pipe, int next_timeout,
				uint64_t *pollok, uint64_t *pollcyc);

int psm3_gen1_ips_ptl_process_err_chk_gen(struct ips_recvhdrq_event *rcv_ev);
int psm3_gen1_ips_ptl_process_becn(struct ips_recvhdrq_event *rcv_ev);
int psm3_gen1_ips_ptl_process_unknown(const struct ips_recvhdrq_event *rcv_ev);
int psm3_gen1_ips_ptl_process_packet_error(struct ips_recvhdrq_event *rcv_ev);
unsigned psm3_gen1_parse_tid(int reload);

psm2_error_t
psm3_gen1_recvhdrq_init(const psmi_context_t *context,
		  const struct ips_epstate *epstate,
		  const struct ips_proto *proto,
		  const struct ips_recvhdrq_callbacks *callbacks,
		  uint32_t subcontext,
		  struct ips_recvhdrq *recvq
		, struct ips_recvhdrq_state *recvq_state,
		  psm3_gen1_cl_q cl_q
		);

psm2_error_t psm3_gen1_recvhdrq_progress(struct ips_recvhdrq *recvq);

 /* This function is designed to implement RAPID CCA. It iterates
 * through the recvq, checking each element for set FECN or BECN bits.
 * In the case of finding one, the proper response is executed, and the bits
 * are cleared.
 */
psm2_error_t psm3_gen1_recvhdrq_scan_cca(struct ips_recvhdrq *recvq);

PSMI_INLINE(int psm3_gen1_recvhdrq_isempty(const struct ips_recvhdrq *recvq))
{
	return psm3_gen1_cl_q_empty(recvq->state->hdrq_head,
				   recvq->gen1_cl_hdrq,
		recvq->context->psm_hw_ctxt);
}

#ifdef PSM_CUDA
void psm3_hfp_gen1_gdr_open(void);
void psm3_gen1_gdr_close(void);
void* psm3_gen1_gdr_convert_gpu_to_host_addr(unsigned long buf,
                                size_t size, int flags, psm2_ep_t ep);
uint64_t psm3_gen1_gdr_cache_evict(void);
#endif /* PSM_CUDA */

/* Get pbc static rate value for flow for a given message length */
PSMI_ALWAYS_INLINE(
uint16_t
psm3_gen1_pbc_static_rate(struct ips_proto *proto, struct ips_flow *flow,
			  uint32_t msgLen))
{
	uint32_t rate = 0;

	/* The PBC rate is based on which HFI type as different media have different
	 * mechanism for static rate control.
	 */

	switch (proto->epinfo.ep_hfi_type) {
	case PSMI_HFI_TYPE_OPA1:
		{
		/*
		 * time_to_send is:
		 *
		 *  (packet_length) [bits] / (pkt_egress_rate) [bits/sec]
		 *  -----------------------------------------------------
		 *     fabric_clock_period == (1 / 805 * 10^6) [1/sec]
		 *
		 *   (where pkt_egress_rate is assumed to be 100 Gbit/s.)
		 */
		uint32_t time_to_send = (8 * msgLen * 805) / (100000);
		rate = (time_to_send >> flow->path->opa.pr_cca_divisor) *
				(flow->path->opa.pr_active_ipd);

		if (rate > 65535)
			rate = 65535;

		}
		break;

	default:
		rate = 0;
	}

	return (uint16_t) rate;
}

/* This is a helper function to convert Per Buffer Control to little-endian */
PSMI_ALWAYS_INLINE(
void psm3_gen1_pbc_to_le(struct psm_hal_pbc *pbc))
{
	pbc->pbc0 = __cpu_to_le32(pbc->pbc0);
	pbc->PbcStaticRateControlCnt = __cpu_to_le16(pbc->PbcStaticRateControlCnt);
	pbc->fill1 = __cpu_to_le16(pbc->fill1);
}

/* Set PBC struct that lies within the extended memory region of SCB */
/* This is used for PIO and SDMA cases; pbc is really a pointer to
 * struct ips_pbc_header * or the equivalent un-named structure
 * in ips_scb. Please note pcb will be in little-endian byte
 * order on return */
PSMI_ALWAYS_INLINE(
void
psm3_gen1_pbc_update(struct ips_proto *proto, struct ips_flow *flow,
		     uint32_t isCtrlMsg, struct psm_hal_pbc *pbc, uint32_t hdrlen,
		     uint32_t paylen))
{
	hfp_gen1_pc_private *psm_hw_ctxt = proto->ep->context.psm_hw_ctxt;
	int dw = (sizeof(struct psm_hal_pbc) + hdrlen + paylen) >> BYTE2DWORD_SHIFT;
	int sc = proto->sl2sc[flow->path->pr_sl];
	int vl = psm_hw_ctxt->sc2vl[sc];
	uint16_t static_rate = 0;

	if_pf(!isCtrlMsg && flow->path->opa.pr_active_ipd)
	    static_rate =
	    psm3_gen1_pbc_static_rate(proto, flow, hdrlen + paylen);

	pbc->pbc0 = __cpu_to_le32((dw & HFI_PBC_LENGTHDWS_MASK) |
	    ((vl & HFI_PBC_VL_MASK) << HFI_PBC_VL_SHIFT) |
	    (((sc >> HFI_PBC_SC4_SHIFT) &
	      HFI_PBC_SC4_MASK) << HFI_PBC_DCINFO_SHIFT));

	pbc->PbcStaticRateControlCnt = __cpu_to_le16(static_rate & HFI_PBC_STATICRCC_MASK);

	/* Per Buffer Control must be in little-endian */
	psm3_gen1_pbc_to_le(pbc);

	return;
}

PSMI_ALWAYS_INLINE(
int      psm3_gen1_get_sdma_ring_size(psmi_hal_hw_context ctxt))
{
        hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
        struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

        return ctrl->ctxt_info.sdma_ring_size;
}

PSMI_ALWAYS_INLINE(
int      psm3_gen1_get_fd(psmi_hal_hw_context ctxt))
{
	if (!ctxt)
		return -1;

	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;

	return psm_hw_ctxt->ctrl->fd;
}

PSMI_ALWAYS_INLINE(
int psm3_gen1_hfi_reset_context(psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return psm3_gen1_nic_reset_context(ctrl);
}

PSMI_ALWAYS_INLINE(int      psm3_gen1_get_context(psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	return ctrl->ctxt_info.ctxt;
}
#endif /* _PSM_HAL_GEN1_HAL_H */
#endif /* PSM_OPA */
