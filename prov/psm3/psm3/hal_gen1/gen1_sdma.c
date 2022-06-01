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

/* Copyright (c) 2003-2017 Intel Corporation. All rights reserved. */

/* included header files  */
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sched.h>

#include "psm_user.h"
#include "ips_proto_params.h"
#include "psm2_hal.h"
#include "ips_proto.h"
#include "gen1_user.h"
#include "psmi_wrappers.h"
#include "gen1_hal.h"

// could just replace with HFI_SDMA_HDR_SIZE in callers.
// should always be HFI_SDMA_HDR_SIZE
PSMI_ALWAYS_INLINE(int psm3_gen1_get_sdma_req_size(psmi_hal_hw_context ctxt))
{
	return get_psm_gen1_hi()->hfp_private.sdmahdr_req_size;
}

PSMI_ALWAYS_INLINE(int psm3_gen1_get_sdma_ring_slot_status(int slotIdx,
				      psmi_hal_sdma_ring_slot_status *status,
				      uint32_t *errorCode,
				      psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	if (slotIdx < 0 || slotIdx >= ctrl->ctxt_info.sdma_ring_size)
	{
		*status = PSM_HAL_SDMA_RING_ERROR;
		return -PSM_HAL_ERROR_GENERAL_ERROR;
	}

	struct hfi1_sdma_comp_entry *sdma_comp_queue = (struct hfi1_sdma_comp_entry *)
	  ctrl->base_info.sdma_comp_bufbase;

	switch (sdma_comp_queue[slotIdx].status)
	{
	case FREE:
		*status = PSM_HAL_SDMA_RING_AVAILABLE;
		break;
	case QUEUED:
		*status = PSM_HAL_SDMA_RING_QUEUED;
		break;
	case COMPLETE:
		*status = PSM_HAL_SDMA_RING_COMPLETE;
		break;
	case ERROR:
		*status = PSM_HAL_SDMA_RING_ERROR;
		break;
	default:
		*status = PSM_HAL_SDMA_RING_ERROR;
		return -PSM_HAL_ERROR_GENERAL_ERROR;
	}
	*errorCode = sdma_comp_queue[slotIdx].errcode;
	return PSM_HAL_ERROR_OK;
}

/* Returns > 0 if the specified slot is available.  0 if not available
   and a negative value if an error occurred. */
PSMI_ALWAYS_INLINE(int psm3_gen1_dma_slot_available(int slotidx, psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;

	if (slotidx < 0 || slotidx >= ctrl->ctxt_info.sdma_ring_size)
		return -1;

	struct hfi1_sdma_comp_entry *sdma_comp_queue = (struct hfi1_sdma_comp_entry *)
	  ctrl->base_info.sdma_comp_bufbase;

	return sdma_comp_queue[slotidx].status != QUEUED;
}

/* Initiate a DMA.  Intrinsically specifies a DMA slot to use. */
PSMI_ALWAYS_INLINE(int psm3_gen1_writev(const struct iovec *iov, int iovcnt, struct ips_epinfo *ignored, psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = (hfp_gen1_pc_private *)ctxt;

	return psm3_gen1_nic_cmd_writev(psm_hw_ctxt->ctrl->fd, iov, iovcnt);
}

/*
 * Driver defines the following sdma completion error code, returned
 * as negative value:
 * #define SDMA_TXREQ_S_OK        0
 * #define SDMA_TXREQ_S_SENDERROR 1
 * #define SDMA_TXREQ_S_ABORTED   2
 * #define SDMA_TXREQ_S_SHUTDOWN  3
 *
 * When hfi is in freeze mode, driver will complete all the pending
 * sdma request as aborted. Since PSM needs to recover from hfi
 * freeze mode, this routine ignore aborted error.
 */
psm2_error_t psm3_gen1_dma_completion_update(struct ips_proto *proto)
{
	ips_scb_t *scb;

	while (proto->sdma_done_index != proto->sdma_fill_index) {
		psmi_hal_sdma_ring_slot_status status;
		uint32_t errorCode;
		int rc = psm3_gen1_get_sdma_ring_slot_status(proto->sdma_done_index, &status, &errorCode,
							    proto->ep->context.psm_hw_ctxt);
		psmi_rmb();

		if (rc < 0)
			return PSM2_INTERNAL_ERR;

		if (status == PSM_HAL_SDMA_RING_QUEUED)
			return PSM2_OK;

		/* Mark sdma request is complete */
		scb = proto->sdma_scb_queue[proto->sdma_done_index];
		if (scb)
		{
			psmi_assert(status == PSM_HAL_SDMA_RING_COMPLETE);
			scb->sdma_outstanding--;
			proto->sdma_scb_queue[proto->sdma_done_index] = NULL;
		}

		if (status == PSM_HAL_SDMA_RING_ERROR && (int)errorCode != -2) {
			psm2_error_t err =
				psm3_handle_error(proto->ep, PSM2_EP_DEVICE_FAILURE,
						  "SDMA completion error: %d (fd=%d, index=%d)",
						  0 - ((int32_t)errorCode),
						  psm3_gen1_get_fd(proto->ep->context.
								  psm_hw_ctxt),
						  proto->sdma_done_index);
			return err;
		}

		proto->sdma_avail_counter++;
		proto->sdma_done_index++;
		if (proto->sdma_done_index == proto->sdma_queue_size)
			proto->sdma_done_index = 0;
	}

	return PSM2_OK;
}

#ifdef PSM_FI
/*
 * Fault injection in dma sends. Since DMA through writev() is all-or-nothing,
 * we don't inject faults on a packet-per-packet basis since the code gets
 * quite complex.  Instead, each call to flush_dma or transfer_frame is treated
 * as an "event" and faults are generated according to the IPS_FAULTINJ_DMASEND
 * setting.
 *
 * The effect is as if the event was successful but dropped on the wire
 * somewhere.
 */
PSMI_ALWAYS_INLINE(int dma_do_fault(psm2_ep_t ep))
{

	if_pf(PSM3_FAULTINJ_ENABLED()) {
		PSM3_FAULTINJ_STATIC_DECL(fi, "dmalost",
					 "discard SDMA packets before sending",
					 1, IPS_FAULTINJ_DMALOST);
		return PSM3_FAULTINJ_IS_FAULT(fi, ep, "");
	}
	else
	return 0;
}
#endif /* #ifdef PSM_FI */

/*

Handles ENOMEM on a DMA completion.

 */
static inline
psm2_error_t
handle_ENOMEM_on_DMA_completion(struct ips_proto *proto)
{
	psm2_error_t err;
	time_t now = time(NULL);

	if (proto->protoexp && proto->protoexp->tidc.tid_cachemap.payload.nidle) {
		uint64_t lengthEvicted =
			ips_tidcache_evict(&proto->protoexp->tidc, -1);

		if (!proto->writevFailTime)
			proto->writevFailTime = now;

		if (lengthEvicted)
			return PSM2_OK; /* signals a retry of the writev command. */
		else {
#ifdef PSM_CUDA
			if (PSMI_IS_GDR_COPY_ENABLED && psm3_gen1_gdr_cache_evict()) {
				return PSM2_OK;
			} else
#endif
				return PSM2_EP_NO_RESOURCES;  /* should signal a return of
							no progress, and retry later */
		}
	}
#ifdef PSM_CUDA
	else if (PSMI_IS_GDR_COPY_ENABLED) {
		uint64_t lengthEvicted = psm3_gen1_gdr_cache_evict();
		if (!proto->writevFailTime)
			proto->writevFailTime = now;

		if (lengthEvicted)
			return PSM2_OK;
		else
			return PSM2_EP_NO_RESOURCES;
	}
#endif
	else if (!proto->writevFailTime)
	{
		proto->writevFailTime = now;
		return PSM2_EP_NO_RESOURCES;  /* should signal a return of
						 no progress, and retry later */
	}
	else
	{
		static const double thirtySeconds = 30.0;

		if (difftime(now, proto->writevFailTime) >
		    thirtySeconds) {
			err = psm3_handle_error(
				proto->ep,
				PSM2_EP_DEVICE_FAILURE,
				"SDMA completion error: out of "
				"memory (fd=%d, index=%d)",
				psm3_gen1_get_fd(proto->ep->context.psm_hw_ctxt),
				proto->sdma_done_index);
			return err;
		}
		return PSM2_EP_NO_RESOURCES;  /* should signal a return of
						 no progress, and retry later */
	}
}

/*
 * Flush all packets currently marked as pending
 * Caller still expects num_sent to always be correctly set in case of an
 * error.
 *
 * Recoverable errors:
 * PSM2_OK: At least one packet was successfully queued up for DMA.
 * PSM2_EP_NO_RESOURCES: No scb's available to handle unaligned packets
 *                      or writev returned a recoverable error (no mem for
 *                      descriptors, dma interrupted or no space left in dma
 *                      queue).
 * PSM2_OK_NO_PROGRESS: Cable pulled.
 *
 * Unrecoverable errors:
 * PSM2_EP_DEVICE_FAILURE: Error calling hfi_sdma_inflight() or unexpected
 *                        error in calling writev(), or chip failure, rxe/txe
 *                        parity error.
 * PSM2_EP_NO_NETWORK: No network, no lid, ...
 */
psm2_error_t
psm3_gen1_dma_send_pending_scbs(struct ips_proto *proto, struct ips_flow *flow,
	     struct ips_scb_pendlist *slist, int *num_sent)
{
	psm2_error_t err = PSM2_OK;
	struct psm_hal_sdma_req_info *sdmahdr;
	struct ips_scb *scb;
	struct iovec *iovec;
	uint16_t iovcnt;

	unsigned int vec_idx = 0;
	unsigned int scb_idx = 0, scb_sent = 0;
	unsigned int num = 0, max_elem;
	uint32_t have_cksum;
	uint32_t fillidx;
	int16_t credits;
#ifdef PSM_BYTE_FLOW_CREDITS
	int16_t credit_bytes;
#endif
	ssize_t ret;

#ifdef PSM_FI
	/* See comments above for fault injection */
	if_pf(dma_do_fault(proto->ep)) goto fail;
#endif /* #ifdef PSM_FI */

	/* Check how many SCBs to send based on flow credits */
	credits = flow->credits;
#ifdef PSM_BYTE_FLOW_CREDITS
	credit_bytes = flow->credit_bytes;
#endif
	psmi_assert(SLIST_FIRST(slist) != NULL);
	SLIST_FOREACH(scb, slist, next) {
		num++;
		credits -= scb->nfrag;
#ifdef PSM_BYTE_FLOW_CREDITS
		credit_bytes -= scb->chunk_size;
		if (credits <= 0 || credit_bytes <= 0)
			break;
#else
		if (credits <= 0)
			break;
#endif
	}
	if (proto->sdma_avail_counter < num) {
		/* if there is not enough sdma slot,
		 * update and use what we have.
		 */
		err = psm3_gen1_dma_completion_update(proto);
		if (err)
			goto fail;
		if (proto->sdma_avail_counter == 0) {
			err = PSM2_EP_NO_RESOURCES;
			goto fail;
		}
		if (proto->sdma_avail_counter < num)
			num = proto->sdma_avail_counter;
	}

	/* header, payload, checksum, tidarray */
	max_elem = 4 * num;
	iovec = alloca(sizeof(struct iovec) * max_elem);

	fillidx = proto->sdma_fill_index;
	SLIST_FOREACH(scb, slist, next) {
		/* Can't exceed posix max writev count */
		if (vec_idx + (int)!!(scb->payload_size > 0) >= UIO_MAXIOV)
			break;

		psmi_assert(vec_idx < max_elem);
		psmi_assert_always(((scb->payload_size & 0x3) == 0) ||
				   psmi_hal_has_cap(PSM_HAL_CAP_NON_DW_MULTIPLE_MSG_SIZE));

		/* Checksum all eager packets */
		have_cksum = scb->ips_lrh.flags & IPS_SEND_FLAG_PKTCKSUM;

		/*
		 * Setup PBC.
		 */
		psm3_gen1_pbc_update(
		    proto,
		    flow,
		    PSMI_FALSE,
		    &scb->pbc,
		    HFI_MESSAGE_HDR_SIZE,
		    scb->payload_size +
			(have_cksum ? PSM_CRC_SIZE_IN_BYTES : 0));

		psmi_assert(psm3_gen1_dma_slot_available(fillidx, proto->ep->context.
								    psm_hw_ctxt));

		size_t extra_bytes;
		sdmahdr = psm3_get_sdma_req_info(scb, &extra_bytes);

		// for nfrag==1, *remaining and frag_size undefined
		sdmahdr->npkts =
			scb->nfrag > 1 ? scb->nfrag_remaining : scb->nfrag;
		sdmahdr->fragsize =
			scb->nfrag > 1 ? scb->frag_size : flow->frag_size;

		sdmahdr->comp_idx = fillidx;
		fillidx++;
		if (fillidx == proto->sdma_queue_size)
			fillidx = 0;

		/*
		 * Setup io vector.
		 */
		iovec[vec_idx].iov_base = sdmahdr;
		iovec[vec_idx].iov_len = psm3_gen1_get_sdma_req_size(proto->ep->context.
								    psm_hw_ctxt) + extra_bytes;
		vec_idx++;
		iovcnt = 1;
		_HFI_VDBG("hdr=%p,%d\n",
			  iovec[vec_idx - 1].iov_base,
			  (int)iovec[vec_idx - 1].iov_len);

		if (scb->payload_size > 0) {
			/*
			 * OPA1 supports byte-aligned payload. If it is
			 * single packet per scb, use payload_size, else
			 * multi-packets per scb, use remaining chunk_size.
			 * payload_size is the remaining chunk first packet
			 * length.
			 */
			iovec[vec_idx].iov_base = ips_scb_buffer(scb);
			iovec[vec_idx].iov_len = scb->nfrag > 1
						     ? scb->chunk_size_remaining
						     : scb->payload_size;
			vec_idx++;
			iovcnt++;
#ifdef PSM_CUDA
			if (PSMI_IS_GPU_ENABLED && IS_TRANSFER_BUF_GPU_MEM(scb)) {
				/* without this attr, CUDA memory accesses
				 * do not synchronize with gpudirect-rdma accesses.
				 * We set this field only if the currently loaded driver
				 * supports this field. If not, we have other problems
				 * where we have a non gpu-direct enabled driver loaded
				 * and PSM2 is trying to use GPU features.
				 */
				if (PSMI_IS_DRIVER_GPUDIRECT_ENABLED)
					sdmahdr->flags = PSM_HAL_BUF_GPU_MEM;
				else
					sdmahdr->flags = 0;
			} else if (PSMI_IS_DRIVER_GPUDIRECT_ENABLED)
				sdmahdr->flags = 0;
			_HFI_VDBG("seqno=%d hdr=%p,%d,flags 0x%x payload=%p,%d\n",
				  scb->seq_num.psn_num,
				  iovec[vec_idx - 2].iov_base,
				  (int)iovec[vec_idx - 2].iov_len,
				  sdmahdr->flags,
				  iovec[vec_idx - 1].iov_base,
				  (int)iovec[vec_idx - 1].iov_len);
#else
			_HFI_VDBG("seqno=%d hdr=%p,%d payload=%p,%d\n",
				  scb->seq_num.psn_num,
				  iovec[vec_idx - 2].iov_base,
				  (int)iovec[vec_idx - 2].iov_len,
				  iovec[vec_idx - 1].iov_base,
				  (int)iovec[vec_idx - 1].iov_len);
#endif
		}

		/* If checksum then update checksum  */
		if (have_cksum) {
			scb->cksum[1] = scb->cksum[0];
			iovec[vec_idx].iov_base = scb->cksum;
			iovec[vec_idx].iov_len = PSM_CRC_SIZE_IN_BYTES;
			vec_idx++;
			iovcnt++;

			_HFI_VDBG("chsum=%p,%d\n",
				  iovec[vec_idx - 1].iov_base,
				  (int)iovec[vec_idx - 1].iov_len);
		}

		/*
		 * If it is TID receive, attached tid info.
		 */
		if (scb->tidctrl) {
			iovec[vec_idx].iov_base = scb->tsess;
			iovec[vec_idx].iov_len = scb->tsess_length;
			vec_idx++;
			iovcnt++;

#ifdef PSM_CUDA
			/*
			 * The driver knows to check for "flags" field in
			 * sdma_req_info only if ctrl=2.
			 */
			if (PSMI_IS_DRIVER_GPUDIRECT_ENABLED) {
				sdmahdr->ctrl = 2 |
					(PSM_HAL_EXP << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
					(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
			} else
#endif
			{

				sdmahdr->ctrl = 1 |
					(PSM_HAL_EXP << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
					(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
			}
			_HFI_VDBG("tid-info=%p,%d\n",
				  iovec[vec_idx - 1].iov_base,
				  (int)iovec[vec_idx - 1].iov_len);
		} else {

#ifdef PSM_CUDA
			if (PSMI_IS_DRIVER_GPUDIRECT_ENABLED) {
				sdmahdr->ctrl = 2 |
					(PSM_HAL_EGR << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
					(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
			} else
#endif
			{
				sdmahdr->ctrl = 1 |
					(PSM_HAL_EGR << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
					(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
			}
		}

		/* Can bound the number to send by 'num' */
		if (++scb_idx == num)
			break;
	}
	psmi_assert(vec_idx > 0);
retry:
	ret = psm3_gen1_writev(iovec, vec_idx, &proto->epinfo, proto->ep->context.psm_hw_ctxt);

	if (ret > 0) {
		proto->writevFailTime = 0;
		/* No need for inflight system call, we can infer it's value
		 * from
		 * writev's return value */
		scb_sent += ret;
	} else {
		/*
		 * ret == 0: Driver did not queue packet. Try later.
		 * ENOMEM: No kernel memory to queue request, try later?
		 * ECOMM: Link may have gone down
		 * EINTR: Got interrupt while in writev
		 */
		if (errno == ENOMEM) {
			err = handle_ENOMEM_on_DMA_completion(proto);
			if (err == PSM2_OK)
				goto retry;
		} else if (ret == 0 || errno == ECOMM || errno == EINTR) {
			err = psm3_gen1_context_check_hw_status(proto->ep);
			/*
			 * During a link bounce the err returned from
			 * psm3_context_check_status is PSM2_EP_NO_NETWORK. In this case
			 * the error code which we need to return to the calling flush
			 * function(ips_proto_flow_flush_dma) is PSM2_EP_NO_RESOURCES to
			 * signal the caller to restart the timers to flush the packets.
			 * Not doing so would leave the packet on the unacked and
			 * pending q without the sdma descriptors ever being updated.
			 */
			if (err == PSM2_OK || err == PSM2_EP_NO_NETWORK)
				err = PSM2_EP_NO_RESOURCES;
		} else {
			err = psm3_handle_error(
			    proto->ep,
			    PSM2_EP_DEVICE_FAILURE,
			    "Unexpected error in writev(): %s (errno=%d) "
			    "(fd=%d,iovec=%p,len=%d)",
			    strerror(errno),
			    errno,
			    psm3_gen1_get_fd(proto->ep->context.psm_hw_ctxt),
			    iovec,
			    vec_idx);
			goto fail;
		}
	}

fail:
	*num_sent = scb_sent;
	psmi_assert(*num_sent <= num && *num_sent >= 0);
	return err;
}

/* dma_transfer_frame is used only for control messages, and is
 * not enabled by default, and not tested by QA; expected send
 * dma goes through dma_send_pending_scbs() */
psm2_error_t
psm3_gen1_dma_transfer_frame(struct ips_proto *proto, struct ips_flow *flow,
		       ips_scb_t *scb, void *payload, uint32_t paylen,
		       uint32_t have_cksum, uint32_t cksum)
{
	ssize_t ret;
	psm2_error_t err;
	struct psm_hal_sdma_req_info *sdmahdr;
	uint16_t iovcnt;
	struct iovec iovec[2];

#ifdef PSM_FI
	/* See comments above for fault injection */
	if_pf(dma_do_fault(proto->ep))
	    return PSM2_OK;
#endif /* #ifdef PSM_FI */
	/*
	 * Check if there is a sdma queue slot.
	 */
	if (proto->sdma_avail_counter == 0) {
		err = psm3_gen1_dma_completion_update(proto);
		if (err)
			return err;

		if (proto->sdma_avail_counter == 0) {
			return PSM2_EP_NO_RESOURCES;
		}
	}

	/*
	 * If we have checksum, put to the end of payload. We make sure
	 * there is enough space in payload for us to put 8 bytes checksum.
	 * for control message, payload is internal PSM buffer, not user buffer.
	 */
	if (have_cksum) {
		uint32_t *ckptr = (uint32_t *) ((char *)payload + paylen);
		*ckptr = cksum;
		ckptr++;
		*ckptr = cksum;
		paylen += PSM_CRC_SIZE_IN_BYTES;
	}

	/*
	 * Setup PBC.
	 */
	psm3_gen1_pbc_update(proto, flow, PSMI_TRUE,
			 &scb->pbc, HFI_MESSAGE_HDR_SIZE, paylen);

	/*
	 * Setup SDMA header and io vector.
	 */
	size_t extra_bytes;
	sdmahdr = psm3_get_sdma_req_info(scb, &extra_bytes);
	sdmahdr->npkts = 1;
	sdmahdr->fragsize = flow->frag_size;
	sdmahdr->comp_idx = proto->sdma_fill_index;
	psmi_assert(psm3_gen1_dma_slot_available(proto->sdma_fill_index, proto->ep->context.psm_hw_ctxt));

	iovcnt = 1;
	iovec[0].iov_base = sdmahdr;
	iovec[0].iov_len = psm3_gen1_get_sdma_req_size(proto->ep->context.psm_hw_ctxt) + extra_bytes;

	if (paylen > 0) {
		iovcnt++;
		iovec[1].iov_base = payload;
		iovec[1].iov_len = paylen;
	}

#ifdef PSM_CUDA
	if (PSMI_IS_DRIVER_GPUDIRECT_ENABLED) {
		sdmahdr->ctrl = 2 |
			(PSM_HAL_EGR << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
			(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
	} else
#endif
	{
		sdmahdr->ctrl = 1 |
			(PSM_HAL_EGR << PSM_HAL_SDMA_REQ_OPCODE_SHIFT) |
			(iovcnt << PSM_HAL_SDMA_REQ_IOVCNT_SHIFT);
	}

	/*
	 * Write into driver to do SDMA work.
	 */
retry:
	ret = psm3_gen1_writev(iovec, iovcnt, &proto->epinfo, proto->ep->context.psm_hw_ctxt);

	if (ret > 0) {
		proto->writevFailTime = 0;
		psmi_assert_always(ret == 1);

		proto->sdma_avail_counter--;
		proto->sdma_fill_index++;
		if (proto->sdma_fill_index == proto->sdma_queue_size)
			proto->sdma_fill_index = 0;

		/*
		 * Wait for completion of this control message if
		 * stack buffer payload is used. This should not be
		 * a performance issue because sdma control message
		 * is not a performance code path.
		 */
		if (iovcnt > 1) {
			/* Setup scb ready for completion. */
			psmi_assert(proto->sdma_scb_queue
					[sdmahdr->comp_idx] == NULL);
			proto->sdma_scb_queue[sdmahdr->comp_idx] = scb;
			scb->sdma_outstanding++;

			/* Wait for completion */
			proto->stats.sdma_compl_wait_ctrl++;
			err = ips_proto_dma_wait_until(proto, scb);
		} else
			err = PSM2_OK;
	} else {
		/*
		 * ret == 0: Driver did not queue packet. Try later.
		 * ENOMEM: No kernel memory to queue request, try later? *
		 * ECOMM: Link may have gone down
		 * EINTR: Got interrupt while in writev
		 */
		if (errno == ENOMEM) {
			err = handle_ENOMEM_on_DMA_completion(proto);
			if (err == PSM2_OK)
				goto retry;
		} else if (ret == 0 || errno == ECOMM || errno == EINTR) {
			err = psm3_gen1_context_check_hw_status(proto->ep);
			/*
			 * During a link bounce the err returned from
			 * psm3_context_check_status is PSM2_EP_NO_NETWORK. In this case
			 * the error code which we need to return to the calling flush
			 * function(ips_proto_flow_flush_dma) is PSM2_EP_NO_RESOURCES to
			 * signal it to restart the timers to flush the packets.
			 * Not doing so would leave the packet on the unacked and
			 * pending q without the sdma descriptors ever being updated.
			 */
			if (err == PSM2_OK || err == PSM2_EP_NO_NETWORK)
				err = PSM2_EP_NO_RESOURCES;
		}

		else
			err = psm3_handle_error(proto->ep,
						PSM2_EP_DEVICE_FAILURE,
						"Unhandled error in writev(): "
						"%s (fd=%d,iovec=%p,len=%d)",
						strerror(errno),
						psm3_gen1_get_fd(proto->ep->context.psm_hw_ctxt),
						&iovec,
						1);
	}

	return err;
}

PSMI_ALWAYS_INLINE(uint64_t psm3_gen1_get_hw_status(psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	struct hfi1_status *status =
	    (struct hfi1_status *) ctrl->base_info.status_bufbase;
	uint64_t hw_status = 0;
	int i;

	// TBD - known issue, when HAL is built as pure inline
	// can't declare static variables in an inline function
	// (and shouldn't delcare in a header file in general)
	static const struct
	{
		uint32_t hfi1_status_dev_bit, psmi_hal_status_bit;
	} status_dev_map[] =
	  {
		  { HFI1_STATUS_INITTED,	  PSM_HAL_HW_STATUS_INITTED },
		  { HFI1_STATUS_CHIP_PRESENT,	  PSM_HAL_HW_STATUS_CHIP_PRESENT },
		  { HFI1_STATUS_HWERROR,	  PSM_HAL_HW_STATUS_HWERROR },
	  };

	for (i=0; i < sizeof(status_dev_map)/sizeof(status_dev_map[0]); i++)
	{
		if (status->dev &status_dev_map[i].hfi1_status_dev_bit)
			hw_status |= status_dev_map[i].psmi_hal_status_bit;
	}

	static const struct
	{
		uint32_t hfi1_status_port_bit, psmi_hal_status_bit;
	} status_port_map[] =
	  {
		  { HFI1_STATUS_IB_READY,	  PSM_HAL_HW_STATUS_IB_READY },
		  { HFI1_STATUS_IB_CONF,	  PSM_HAL_HW_STATUS_IB_CONF },
	  };

	for (i=0; i < sizeof(status_port_map)/sizeof(status_port_map[0]); i++)
	{
		if (status->port &status_port_map[i].hfi1_status_port_bit)
			hw_status |= status_port_map[i].psmi_hal_status_bit;
	}

	return hw_status;
}

PSMI_ALWAYS_INLINE(int psm3_gen1_get_hw_status_freezemsg(volatile char** msg, psmi_hal_hw_context ctxt))
{
	hfp_gen1_pc_private *psm_hw_ctxt = ctxt;
	struct _hfi_ctrl *ctrl = psm_hw_ctxt->ctrl;
	struct hfi1_status *status =
	    (struct hfi1_status *) ctrl->base_info.status_bufbase;

	*msg = (volatile char *) status->freezemsg;

	return PSM2_OK;
}

/*
 * This function works whether a context is initialized or not in a psm2_ep.
 *
 * Returns one of
 *
 * PSM2_OK: Port status is ok (or context not initialized yet but still "ok")
 * PSM2_OK_NO_PROGRESS: Cable pulled
 * PSM2_EP_NO_NETWORK: No network, no lid, ...
 * PSM2_EP_DEVICE_FAILURE: Chip failures, rxe/txe parity, etc.
 * The message follows the per-port status
 * As of 7322-ready driver, need to check port-specific qword for IB
 * as well as older unit-only.  For now, we don't have the port interface
 * defined, so just check port 0 qword for spi_status
 */
psm2_error_t psm3_gen1_context_check_hw_status(psm2_ep_t ep)
{
	psm2_error_t err = PSM2_OK;
	psmi_context_t *context = &ep->context;
	char *errmsg = NULL;
	uint64_t status = psm3_gen1_get_hw_status(context->psm_hw_ctxt);

	/* Fatal chip-related errors */
	if (!(status & PSM_HAL_HW_STATUS_CHIP_PRESENT) ||
	    !(status & PSM_HAL_HW_STATUS_INITTED) ||
	    (status & PSM_HAL_HW_STATUS_HWERROR)) {

		err = PSM2_EP_DEVICE_FAILURE;
		if (err != context->status_lasterr) {	/* report once */
			volatile char *errmsg_sp="no err msg";

			psm3_gen1_get_hw_status_freezemsg(&errmsg_sp,
							 context->psm_hw_ctxt);

			if (*errmsg_sp)
				psm3_handle_error(ep, err,
						  "Hardware problem: %s",
						  errmsg_sp);
			else {
				if (status & PSM_HAL_HW_STATUS_HWERROR)
					errmsg = "Hardware error";
				else
					errmsg = "Hardware not found";

				psm3_handle_error(ep, err, "%s", errmsg);
			}
		}
	}
	/* Fatal network-related errors with timeout: */
	else if (!(status & PSM_HAL_HW_STATUS_IB_CONF) ||
		 !(status & PSM_HAL_HW_STATUS_IB_READY)) {
		err = PSM2_EP_NO_NETWORK;
		if (err != context->status_lasterr) {	/* report once */
			context->networkLostTime = time(NULL);
		}
		else
		{
			time_t now = time(NULL);
			static const double seventySeconds = 70.0;

			/* The linkup time duration for a system should allow the time needed
			   to complete 3 LNI passes which is:
			   50 seconds for a passive copper channel
			   65 seconds for optical channel.
			   (we add 5 seconds of margin.) */
			if (difftime(now,context->networkLostTime) > seventySeconds)
			{
				volatile char *errmsg_sp="no err msg";

				psm3_gen1_get_hw_status_freezemsg(&errmsg_sp,
								 context->psm_hw_ctxt);

				psm3_handle_error(ep, err, "%s",
						  *errmsg_sp ? errmsg_sp :
						  "Network down");
			}
		}
	}

	if (err == PSM2_OK && context->status_lasterr != PSM2_OK)
		context->status_lasterr = PSM2_OK;	/* clear error */
	else if (err != PSM2_OK)
		context->status_lasterr = err;	/* record error */

	return err;
}
#endif /* PSM_OPA */
