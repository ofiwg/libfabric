/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2016 Intel Corporation.

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

  Copyright(c) 2016 Intel Corporation.

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

/* Copyright (c) 2016 Intel Corporation. All rights reserved. */

// This file implements the TID protocol for STL100 and the RDMA
// protocol for UD mode.  The majority of functons in this file (perhaps all)
// are not used when TID/RDMA is disabled via PSM3_TID o PSM3_RDMA respectively
// RDMA is N/A for UDP, so it will behave as if PSM3_RDMA is disabled
// and not use functions in this file.

#include "psm_user.h"
#include "psm2_hal.h"

#include "ips_scb.h"
#include "ips_tid.h"
#include "ips_tidflow.h"
#include "ips_proto.h"
#include "ips_expected_proto.h"
#include "ips_proto_help.h"
#include "psm_mq_internal.h"

/*
 * Timer callbacks.  When we need work to be done out of the receive process
 * loop, we schedule work on timers to be done at a later time.
 */
static psm2_error_t
ips_tid_pendsend_timer_callback(struct psmi_timer *timer, uint64_t current);

static psm2_error_t
ips_tid_pendtids_timer_callback(struct psmi_timer *timer, uint64_t current);

// TBD explore when to use PSM_HAVE_REG_MR vs PSM_VERBS vs put in HAL
// any code which remains here and tests RNDV_MOD should test PSM_HAVE_RNDV_MOD
#if defined(PSM_VERBS)
#ifdef RNDV_MOD
static void ips_protoexp_send_err_chk_rdma_resp(struct ips_flow *flow);
static void ips_tid_reissue_rdma_write(struct ips_tid_send_desc *tidsendc);
#endif
#endif

static void ips_tid_scbavail_callback(struct ips_scbctrl *scbc, void *context);
#ifdef PSM_OPA
static void ips_tid_avail_callback(struct ips_tid *tidc, void *context);
#endif
static void ips_tidflow_avail_callback(struct ips_tf *tfc, void *context);

#ifdef PSM_OPA
/* Defined at the ptl-level (breaks abstractions but needed for shared vs
 * non-shared contexts */
extern int psm3_gen1_ips_ptl_recvq_isempty(const struct ptl *ptl);
#endif

#ifdef PSM_HAVE_RDMA
static psm2_error_t ips_tid_recv_free(struct ips_tid_recv_desc *tidrecvc);
#endif // PSM_HAVE_RDMA
static psm2_error_t ips_tid_send_exp(struct ips_tid_send_desc *tidsendc);

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
static
void psmi_cuda_run_prefetcher(struct ips_protoexp *protoexp,
			      struct ips_tid_send_desc *tidsendc);
static void psmi_attach_chb_to_tidsendc(struct ips_protoexp *protoexp,
					psm2_mq_req_t req,
					struct ips_tid_send_desc *tidsendc,
					struct ips_gpu_hostbuf *chb_prev,
					uint32_t tsess_srcoff,
					uint32_t tsess_length,
					uint32_t tsess_unaligned_start,
					psm2_chb_match_type_t type);
#endif

psm2_error_t
MOCKABLE(psm3_ips_protoexp_init)(const struct ips_proto *proto,
		  uint32_t protoexp_flags,
		  int num_of_send_bufs,
		  int num_of_send_desc, struct ips_protoexp **protoexp_o)
{
	struct ips_protoexp *protoexp = NULL;
	psm2_ep_t ep = proto->ep;
#ifdef PSM_OPA
	uint32_t tidmtu_max;
#endif
	psm2_error_t err = PSM2_OK;

#ifndef PSM_OPA
#ifdef PSM_HAVE_REG_MR
	if (!psmi_hal_has_cap(PSM_HAL_CAP_RDMA)) {
#else
	{
#endif
		psmi_assert_always(0);
		err = PSM2_INTERNAL_ERR;
		goto fail;
	}
#endif

	protoexp = (struct ips_protoexp *)
	    psmi_calloc(ep, UNDEFINED, 1, sizeof(struct ips_protoexp));
	if (protoexp == NULL) {
		err = PSM2_NO_MEMORY;
		goto fail;
	}
	*protoexp_o = protoexp;

	protoexp->ptl = (const struct ptl *)proto->ptl;
	protoexp->proto = (struct ips_proto *)proto;
	protoexp->timerq = proto->timerq;
#ifdef PSM_OPA
	srand48_r((long int) getpid(), &protoexp->tidflow_drand48_data);
#endif
	protoexp->tid_flags = protoexp_flags;

	if (ep->memmode == PSMI_MEMMODE_MINIMAL) {
		protoexp->tid_flags |= IPS_PROTOEXP_FLAG_CTS_SERIALIZED;
	}

#ifdef PSM_OPA
	// for RDMA Rendezvous we use a single MR for the message so
	// we only need 1 entry in the CTS.
	// For native mode, the CTS contains a list of TIDs and the window's
	// size must be constrained such that the list for all pages in a window
	// won't exceed an MTU (eg. CTS message must fit in an MTU)
	{
		/*
		 * Adjust the session window size so that tid-grant (CTS) message can
		 * fit into a single frag size packet for single transfer, PSM
		 * must send tid-grant message with a single packet.
		 */
		uint32_t fragsize, winsize;

#ifndef PSM_OPA
		fragsize = proto->epinfo.ep_mtu;
#else
		if (proto->flags & IPS_PROTO_FLAG_SDMA)
			fragsize = proto->epinfo.ep_mtu;
		else
			fragsize = proto->epinfo.ep_piosize;
#endif

		winsize = 2 * PSMI_PAGESIZE	/* bytes per tid-pair */
			/* space in packet */
			* min((fragsize - sizeof(ips_tid_session_list)),
			/* space in tidsendc/tidrecvc descriptor */
			PSM_TIDLIST_BUFSIZE)
			/ sizeof(uint32_t);	/* convert to tid-pair */

		if (proto->mq->hfi_base_window_rv > winsize)
			proto->mq->hfi_base_window_rv = winsize;
	}
#endif

	/* Must be initialized already */
	/* Comment out because of Klockwork scanning critical error. CQ 11/16/2012
	   psmi_assert_always(proto->ep != NULL && proto->ep->mq != NULL &&
	   proto->ep->mq->rreq_pool != NULL &&
	   proto->ep->mq->sreq_pool != NULL);
	 */
	psmi_assert_always(proto->timerq != NULL);

	/* These request pools are managed by the MQ component */
	protoexp->tid_sreq_pool = proto->ep->mq->sreq_pool;
	protoexp->tid_rreq_pool = proto->ep->mq->rreq_pool;

#ifdef PSM_OPA
	/* tid traffic xfer type */
	if (proto->flags & IPS_PROTO_FLAG_SPIO)
		protoexp->tid_xfer_type = PSM_TRANSFER_PIO;
	else
		protoexp->tid_xfer_type = PSM_TRANSFER_DMA;

	/* ctrl ack/nak xfer type */
	if (proto->flags & IPS_PROTO_FLAG_SDMA)
		protoexp->ctrl_xfer_type = PSM_TRANSFER_DMA;
	else
		protoexp->ctrl_xfer_type = PSM_TRANSFER_PIO;
#else
	protoexp->ctrl_xfer_type = PSM_TRANSFER_PIO;
#endif

	/* Initialize tid flow control. */
	err = psm3_ips_tf_init(protoexp, &protoexp->tfc,
			       ips_tidflow_avail_callback);
	if (err != PSM2_OK)
		goto fail;

#ifdef PSM_OPA
	if (proto->flags & IPS_PROTO_FLAG_SPIO)
		tidmtu_max = proto->epinfo.ep_piosize;
	else
		tidmtu_max = proto->epinfo.ep_mtu;

	protoexp->tid_send_fragsize = tidmtu_max;

	if ((err = ips_tid_init(&ep->context, protoexp,
				ips_tid_avail_callback, protoexp)))
		goto fail;
#endif

	if ((err = psm3_ips_scbctrl_init(ep, num_of_send_desc, 0,
				    0, 0, ips_tid_scbavail_callback,
				    protoexp, &protoexp->tid_scbc_rv)))
		goto fail;

#ifdef PSM_OPA
	{
		/* Determine interval to generate headers (relevant only when header
		 * suppression is enabled) else headers will always be generated.
		 *
		 * The PSM3_EXPECTED_HEADERS environment variable can specify the
		 * packet interval to generate headers at. Else a header packet is
		 * generated every
		 * min(PSM_DEFAULT_EXPECTED_HEADER, window_size/tid_send_fragsize).
		 * Note: A header is always generated for the last packet in the flow.
		 */

		union psmi_envvar_val env_exp_hdr;
		uint32_t defval = min(PSM_DEFAULT_EXPECTED_HEADER,
				      proto->mq->hfi_base_window_rv /
				      protoexp->tid_send_fragsize);

		psm3_getenv("PSM3_EXPECTED_HEADERS",
			    "Interval to generate expected protocol headers",
			    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT_FLAGS,
			    (union psmi_envvar_val)defval, &env_exp_hdr);

		protoexp->hdr_pkt_interval = env_exp_hdr.e_uint;
		/* Account for flow credits - Should try to have atleast 4 headers
		 * generated per window.
		 */
		protoexp->hdr_pkt_interval =
		    max(min
			(protoexp->hdr_pkt_interval, proto->flow_credits >> 2),
			1);

		if (protoexp->hdr_pkt_interval != env_exp_hdr.e_uint) {
			_HFI_VDBG
			    ("Overriding PSM3_EXPECTED_HEADERS=%u to be '%u'\n",
			     env_exp_hdr.e_uint, protoexp->hdr_pkt_interval);
		}

	}
#endif

	{
		union psmi_envvar_val env_rts_cts_interleave;

		psm3_getenv("PSM3_RTS_CTS_INTERLEAVE",
			    "Interleave the handling of RTS to provide a fair distribution between multiple senders",
			    PSMI_ENVVAR_LEVEL_USER, PSMI_ENVVAR_TYPE_UINT_FLAGS,
			    (union psmi_envvar_val)0, &env_rts_cts_interleave);
		if (env_rts_cts_interleave.e_uint)
			protoexp->tid_flags |= IPS_PROTOEXP_FLAG_RTS_CTS_INTERLEAVE;
	}

	/* Send descriptors.
	 *
	 * There can be up to 2^32 of these send descriptors.  We conservatively
	 * allocate 256 but large node configurations can allocate up to sdesc_num
	 * of these (they are about 2k each).
	 * We impose a theoretical limit of 2^30.
	 */
	{
		struct psmi_rlimit_mpool rlim = TID_SENDSESSIONS_LIMITS;
		uint32_t maxsz, chunksz;

		if ((err = psm3_parse_mpool_env(protoexp->proto->mq, 1,
						&rlim, &maxsz, &chunksz)))
			goto fail;

		protoexp->tid_desc_send_pool =
		    psm3_mpool_create(sizeof(struct ips_tid_send_desc), chunksz,
				      maxsz, 0, DESCRIPTORS, NULL, NULL);

		if (protoexp->tid_desc_send_pool == NULL) {
			err = psm3_handle_error(proto->ep, PSM2_NO_MEMORY,
						"Couldn't allocate tid descriptor memory pool");
			goto fail;
		}
	}

	/* Receive descriptors are an array in tidflow structure. */

	/* This pool can never be smaller than the max number of rreqs that can be
	 * allocated. */
	{
		uint32_t rreq_per_chunk, rreq_max;

		psmi_assert_always(protoexp->proto->mq->rreq_pool != NULL);

		psm3_mpool_get_obj_info(protoexp->proto->mq->rreq_pool,
					&rreq_per_chunk, &rreq_max);

		protoexp->tid_getreq_pool =
		    psm3_mpool_create(sizeof(struct ips_tid_get_request),
				      rreq_per_chunk, rreq_max, 0, DESCRIPTORS,
				      NULL, NULL);

		if (protoexp->tid_getreq_pool == NULL) {
			err = psm3_handle_error(proto->ep, PSM2_NO_MEMORY,
						"Couldn't allocate getreq descriptor memory pool");
			goto fail;
		}
	}

	/* Timers to handle requeueing of work out of the receive path */
	psmi_timer_entry_init(&protoexp->timer_send,
			      ips_tid_pendsend_timer_callback, protoexp);
	STAILQ_INIT(&protoexp->pend_sendq);
	psmi_timer_entry_init(&protoexp->timer_getreqs,
			      ips_tid_pendtids_timer_callback, protoexp);
	STAILQ_INIT(&protoexp->pend_getreqsq);
#if defined(PSM_VERBS)
#ifdef RNDV_MOD
	STAILQ_INIT(&protoexp->pend_err_resp);
#endif
#endif


#ifdef PSM_OPA
	protoexp->tid_page_offset_mask = PSMI_PAGESIZE - 1;
	protoexp->tid_page_mask = ~(PSMI_PAGESIZE - 1);

	/*
	 * After ips_tid_init(), we know if we use tidcache or not.
	 * if tid cache is used, we can't use tid debug.
	 */
#ifdef PSM_DEBUG
	if (protoexp->tidc.tid_array == NULL)
		protoexp->tid_flags |= IPS_PROTOEXP_FLAG_TID_DEBUG;
#endif

	if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_TID_DEBUG) {
		int i;
		protoexp->tid_info = (struct ips_tidinfo *)
		    psmi_calloc(ep, UNDEFINED, IPS_TID_MAX_TIDS,
				sizeof(struct ips_tidinfo));
		if (protoexp->tid_info == NULL) {
			err = PSM2_NO_MEMORY;
			goto fail;
		}
		for (i = 0; i < IPS_TID_MAX_TIDS; i++) {
			protoexp->tid_info[i].state = TIDSTATE_FREE;
			protoexp->tid_info[i].tidrecvc = NULL;
			protoexp->tid_info[i].tid = 0xFFFFFFFF;
		}
	} else
		protoexp->tid_info = NULL;
#endif // PSM_OPA

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	{
		if (PSMI_IS_GPU_ENABLED &&
			 !(proto->flags & IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV)) {
			struct psmi_rlimit_mpool rlim = GPU_HOSTBUFFER_LIMITS;
			uint32_t maxsz, chunksz, max_elements;

			if ((err = psm3_parse_mpool_env(protoexp->proto->mq, 1,
							&rlim, &maxsz, &chunksz)))
				goto fail;

			/* the maxsz is the amount in MB, not the number of entries,
			 * since the element size depends on the window size */
			max_elements = (maxsz*1024*1024) / proto->mq->hfi_base_window_rv;
			/* mpool requires max_elements to be power of 2. round down. */
			max_elements = 1 << (31 - __builtin_clz(max_elements));
			protoexp->cuda_hostbuf_recv_cfg.bufsz =
				proto->mq->hfi_base_window_rv;

			protoexp->cuda_hostbuf_pool_recv =
				psm3_mpool_create_for_gpu(sizeof(struct ips_gpu_hostbuf),
							  chunksz, max_elements, 0,
							  UNDEFINED, NULL, NULL,
							  psmi_gpu_hostbuf_alloc_func,
							  (void *)
							  &protoexp->cuda_hostbuf_recv_cfg);

			if (protoexp->cuda_hostbuf_pool_recv == NULL) {
				err = psm3_handle_error(proto->ep, PSM2_NO_MEMORY,
							"Couldn't allocate CUDA host receive buffer pool");
				goto fail;
			}

			protoexp->cuda_hostbuf_small_recv_cfg.bufsz =
				CUDA_SMALLHOSTBUF_SZ;
			protoexp->cuda_hostbuf_pool_small_recv =
				psm3_mpool_create_for_gpu(sizeof(struct ips_gpu_hostbuf),
							  chunksz, max_elements, 0,
							  UNDEFINED, NULL, NULL,
							  psmi_gpu_hostbuf_alloc_func,
							  (void *)
							  &protoexp->cuda_hostbuf_small_recv_cfg);

			if (protoexp->cuda_hostbuf_pool_small_recv == NULL) {
				err = psm3_handle_error(proto->ep, PSM2_NO_MEMORY,
							"Couldn't allocate CUDA host small receive buffer pool");
				goto fail;
			}
			PSM3_GPU_PREPARE_HTOD_MEMCPYS(protoexp);
			STAILQ_INIT(&protoexp->cudapend_getreqsq);
		} else {
			protoexp->cuda_hostbuf_pool_recv = NULL;
			protoexp->cuda_hostbuf_pool_small_recv = NULL;
		}
	}
#endif
	psmi_assert(err == PSM2_OK);
	return err;

fail:
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (protoexp != NULL && protoexp->cuda_hostbuf_pool_recv != NULL)
		psm3_mpool_destroy(protoexp->cuda_hostbuf_pool_recv);
	if (protoexp != NULL && protoexp->cuda_hostbuf_pool_small_recv != NULL)
		psm3_mpool_destroy(protoexp->cuda_hostbuf_pool_small_recv);
#endif
	if (protoexp != NULL && protoexp->tid_getreq_pool != NULL)
		psm3_mpool_destroy(protoexp->tid_getreq_pool);
	if (protoexp != NULL && protoexp->tid_desc_send_pool != NULL)
		psm3_mpool_destroy(protoexp->tid_desc_send_pool);
	if (protoexp != NULL)
		psm3_ips_scbctrl_fini(&protoexp->tid_scbc_rv);
	if (protoexp != NULL)
		psmi_free(protoexp);
	return err;
}
MOCK_DEF_EPILOGUE(psm3_ips_protoexp_init);

psm2_error_t psm3_ips_protoexp_fini(struct ips_protoexp *protoexp)
{
	psm2_error_t err = PSM2_OK;

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if(PSMI_IS_GPU_ENABLED &&
		 !(protoexp->proto->flags & IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV)) {
		psm3_mpool_destroy(protoexp->cuda_hostbuf_pool_small_recv);
		psm3_mpool_destroy(protoexp->cuda_hostbuf_pool_recv);
		PSM3_GPU_SHUTDOWN_HTOD_MEMCPYS(protoexp);
	}
#endif
	psm3_mpool_destroy(protoexp->tid_getreq_pool);
	psm3_mpool_destroy(protoexp->tid_desc_send_pool);

	if ((err = psm3_ips_scbctrl_fini(&protoexp->tid_scbc_rv)))
		goto fail;

#ifdef PSM_OPA
	if ((err = ips_tid_fini(&protoexp->tidc)))
		goto fail;
#endif

	/* finalize tid flow control. */
	if ((err = psm3_ips_tf_fini(&protoexp->tfc)))
		goto fail;

#ifdef PSM_OPA
	if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_TID_DEBUG)
		psmi_free(protoexp->tid_info);
#endif

	psmi_free(protoexp);

fail:
	return err;
}

/* New scbs now available.  If we have pending sends or pending get requests,
 * turn on the timer so it can be processed. */
/* for RDMA we can also use this routine when an MR is freed.  scbc is not used
 */
static
void ips_tid_scbavail_callback(struct ips_scbctrl *scbc, void *context)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *)context;

	if (!STAILQ_EMPTY(&protoexp->pend_sendq))
		psmi_timer_request(protoexp->timerq,
				   &protoexp->timer_send, PSMI_TIMER_PRIO_1);
	if (!STAILQ_EMPTY(&protoexp->pend_getreqsq)
#if defined(PSM_VERBS)
#ifdef RNDV_MOD
		|| !STAILQ_EMPTY(&protoexp->pend_err_resp)
#endif
#endif
		)
		psmi_timer_request(protoexp->timerq,
				   &protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
	return;
}

#ifdef PSM_HAVE_REG_MR
void ips_tid_mravail_callback(struct ips_proto *proto)
{
	// if we have Send DMA but not RDMA, no proto->protoexp
	if (proto->protoexp)
		ips_tid_scbavail_callback(NULL, proto->protoexp);
}
#endif

#ifdef PSM_OPA
/* New Tids are available. If there are pending get requests put the
 * get timer on the timerq so it can be processed. */
static
void ips_tid_avail_callback(struct ips_tid *tidc, void *context)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *)context;

	if (!STAILQ_EMPTY(&protoexp->pend_getreqsq))
		psmi_timer_request(protoexp->timerq,
				   &protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
	return;
}
#endif

// On STL100 ips_tf is a user space control for the HW tidflow which
// would fully process most valid inbound EXPTID packets within an RV Window.
// For UD we maintain the user space control to help manage each active
// RV window.
// There is one CTS per RV window (typically 128K).
// For UD with RV, RDMA is used instread of EXPTID, with 1 RDMA per RV window.
// Typically there are 32 (HFI_TF_NFLOWS) configured.
// The 32 is hard coded, could make it tunable.
// The tidflow provides a natural pacing mechanism and limits the total amount
// of inflight EXPTID or RDMA incoming to given receiver.
// In addition on STL100 there is an upper bound on TIDs which limited total
// inbound DMA for a receiver to avoid 4MB. For smaller messages tidflow
// count may be the limit, for larger messages TIDs would be the limit.

/* New Tid Flows are available. If there are pending get requests put the
 * get timer on the timerq so it can be processed. */
static
void ips_tidflow_avail_callback(struct ips_tf *tfc, void *context)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *)context;

	if (!STAILQ_EMPTY(&protoexp->pend_getreqsq))
	{
		psmi_timer_request(protoexp->timerq,
				   &protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
	}
	return;
}

// this is called from ips_proto_mq_rts_match_callback when a RTS is matched
// and we chose to use the TID receive mechanism
// this kicks off the receiver side protocol for preparing TIDs and issuing a
// CTS which requests use of TID
/*
 * The tid get request is always issued from within the receive progress loop,
 * which is why we always enqueue the request instead of issuing it directly.
 * Eventually, if we expose tid_get to users, we will want to differentiate
 * when the request comes from the receive progress loop from cases where the
 * tid_get is issued directly from user code.
 *
 */
psm2_error_t
psm3_ips_protoexp_tid_get_from_token(struct ips_protoexp *protoexp,
				void *buf,
				uint32_t length,
				psm2_epaddr_t epaddr,
				uint32_t remote_tok,
				uint32_t flags,
				ips_tid_completion_callback_t callback,
				psm2_mq_req_t req)
{
	struct ips_tid_get_request *getreq;
	int count;
#ifdef PSM_OPA
	int tids;
#endif
	int tidflows;
	uint64_t nbytes;

	PSM2_LOG_MSG("entering");
#ifdef PSM_OPA
	psmi_assert((((ips_epaddr_t *) epaddr)->opa.window_rv % PSMI_PAGESIZE) == 0);
#else
	psmi_assert((req->mq->hfi_base_window_rv % PSMI_PAGESIZE) == 0);
#endif
	getreq = (struct ips_tid_get_request *)
	    psm3_mpool_get(protoexp->tid_getreq_pool);

	/* We can't *really* run out of these here because we always allocate as
	 * much as available receive reqs */
	if_pf(getreq == NULL)
	{
		PSM2_LOG_MSG("leaving");
		psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			      "Ran out of 'getreq' descriptors");
	}

	getreq->tidgr_protoexp = protoexp;
	getreq->tidgr_epaddr = epaddr;
	getreq->tidgr_lbuf = buf;
	getreq->tidgr_length = length;
	getreq->tidgr_sendtoken = remote_tok;
	getreq->tidgr_req = req;
	getreq->tidgr_callback = callback;
	getreq->tidgr_offset = 0;
	getreq->tidgr_bytesdone = 0;
	getreq->tidgr_flags = flags;

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if ((req->is_buf_gpu_mem &&
	    !(protoexp->proto->flags & IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV)) ||
	    ((req->is_buf_gpu_mem &&
	     (protoexp->proto->flags & IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV) &&
	     (length > gpudirect_rdma_recv_limit
#ifndef PSM_OPA
		|| length & 0x03 || (uintptr_t)buf & 0x03
#endif
 		)))) {
		getreq->gpu_hostbuf_used = 1;
		getreq->tidgr_cuda_bytesdone = 0;
		STAILQ_INIT(&getreq->pend_cudabuf);
		protoexp->proto->strat_stats.rndv_rdma_hbuf_recv++;
		protoexp->proto->strat_stats.rndv_rdma_hbuf_recv_bytes += length;
	} else {
		getreq->gpu_hostbuf_used = 0;
		if (req->is_buf_gpu_mem) {
			protoexp->proto->strat_stats.rndv_rdma_gdr_recv++;
			protoexp->proto->strat_stats.rndv_rdma_gdr_recv_bytes += length;
		} else {
#endif
			protoexp->proto->strat_stats.rndv_rdma_cpu_recv++;
			protoexp->proto->strat_stats.rndv_rdma_cpu_recv_bytes += length;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		}
	}
#endif

	/* nbytes is the bytes each channel should transfer. */
	count = ((ips_epaddr_t *) epaddr)->msgctl->ipsaddr_count;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (req->is_buf_gpu_mem)
		nbytes = PSMI_ALIGNUP((length + count - 1) / count, PSMI_GPU_PAGESIZE);
	else
#endif
		nbytes = PSMI_ALIGNUP((length + count - 1) / count, PSMI_PAGESIZE);
	getreq->tidgr_rndv_winsz =
#ifndef PSM_OPA
	    min(nbytes, req->mq->hfi_base_window_rv);
#else
	    min(nbytes, ((ips_epaddr_t *) epaddr)->opa.window_rv);
	/* must be within the tid window size */
	if (getreq->tidgr_rndv_winsz > PSM_TID_WINSIZE)
		getreq->tidgr_rndv_winsz = PSM_TID_WINSIZE;
#endif
	_HFI_MMDBG("posting TID get request: nbytes=%"PRIu64" winsz=%u len=%u\n",
				 nbytes, getreq->tidgr_rndv_winsz, getreq->tidgr_length);
	// we have now computed the size of each TID sequence (tidgr_rndv_winsz)

	STAILQ_INSERT_TAIL(&protoexp->pend_getreqsq, getreq, tidgr_next);
#ifdef PSM_OPA
	tids = ips_tid_num_available(&protoexp->tidc);
#endif
	// by using tidflow we also constrain amount of concurrent RDMA to our NIC
	tidflows = ips_tf_available(&protoexp->tfc);
	_HFI_MMDBG("available tidflow %u\n", tidflows);

	if (
#ifdef PSM_OPA
		tids > 0 &&
#endif
		tidflows > 0)
		// get the actual TIDs and tidflows and send the CTS
		ips_tid_pendtids_timer_callback(&protoexp->timer_getreqs, 0);
	else if (
#ifdef PSM_OPA
		tids != -1 &&
#endif
		tidflows != -1)
		// out of TIDs, set a timer to try again later
		psmi_timer_request(protoexp->timerq, &protoexp->timer_getreqs,
				   PSMI_TIMER_PRIO_1);
	PSM2_LOG_MSG("leaving");
	return PSM2_OK;
}

/* List of perf events */
#define _ips_logeventid_tid_send_reqs	0	/* out of tid send descriptors */

#define ips_logevent_id(event)	 _ips_logeventid_ ## event
#define ips_logevent(proto, event, ptr) ips_logevent_inner(proto, ips_logevent_id(event), ptr)

static
void ips_logevent_inner(struct ips_proto *proto, int eventid, void *context)
{
	uint64_t t_now = get_cycles();

	switch (eventid) {
	case ips_logevent_id(tid_send_reqs):{
			psm2_epaddr_t epaddr = (psm2_epaddr_t) context;
			proto->psmi_logevent_tid_send_reqs.count++;

			if (t_now >=
			    proto->psmi_logevent_tid_send_reqs.next_warning) {
				psm3_handle_error(PSMI_EP_LOGEVENT, PSM2_OK,
#ifndef PSM_OPA
						  "Non-fatal temporary exhaustion of send rdma descriptors "
#else
						  "Non-fatal temporary exhaustion of send tid dma descriptors "
#endif
						  "(elapsed=%.3fs, source %s, count=%lld)",
						  (double)
						  cycles_to_nanosecs(t_now -
								     proto->
								     t_init) /
						  1.0e9,
						  psm3_epid_fmt_addr(epaddr->epid, 0),
						  (long long)proto->
						  psmi_logevent_tid_send_reqs.
						  count);
				proto->psmi_logevent_tid_send_reqs.
				    next_warning =
				    t_now +
				    sec_2_cycles(proto->
						 psmi_logevent_tid_send_reqs.
						 interval_secs);
			}
		}
		break;

	default:
		break;
	}

	return;
}

/*
 * Expected Protocol.
 *
 * We're granted tids (as part of a tid get request) and expected to fulfill
 * the request by associating the request's sendtoken to a tid send descriptor.
 *
 * It's possible to be out of tid send descriptors when somehow all allocated
 * descriptors can't complete all of their sends.  For example, the targets of
 * the sends may be busy in computation loops and not processing incoming
 * packets.
 */

// build and issue CTS
void
psm3_ips_protoexp_send_tid_grant(struct ips_tid_recv_desc *tidrecvc)
{
	ips_epaddr_t *ipsaddr = tidrecvc->ipsaddr;
	struct ips_proto *proto = tidrecvc->protoexp->proto;
	psmi_assert(proto->msgflowid < EP_FLOW_LAST);
	struct ips_flow *flow = &ipsaddr->flows[proto->msgflowid];
	ips_scb_t *scb;

	scb = tidrecvc->grantscb;
	ips_scb_opcode(scb) = OPCODE_LONG_CTS;
	scb->ips_lrh.khdr.kdeth0 = 0;
	scb->ips_lrh.mdata = tidrecvc->tidflow_genseq.psn_val;
	scb->ips_lrh.data[0] = tidrecvc->rdescid;
	scb->ips_lrh.data[1].u32w1 = tidrecvc->getreq->tidgr_length;
	scb->ips_lrh.data[1].u32w0 = tidrecvc->getreq->tidgr_sendtoken;

	ips_scb_buffer(scb) = (void *)&tidrecvc->tid_list;
#ifndef PSM_OPA
	scb->chunk_size = ips_scb_length(scb) = sizeof(tidrecvc->tid_list);
#else
	scb->chunk_size = ips_scb_length(scb) = tidrecvc->tsess_tidlist_length;
#endif
	_HFI_MMDBG("sending CTS\n");

	PSM2_LOG_EPM(OPCODE_LONG_CTS,PSM2_LOG_TX, proto->ep->epid,
		    flow->ipsaddr->epaddr.epid ,"tidrecvc->getreq->tidgr_sendtoken; %d",
		    tidrecvc->getreq->tidgr_sendtoken);
	proto->epaddr_stats.cts_rdma_send++;

	psm3_ips_proto_flow_enqueue(flow, scb);
	flow->flush(flow, NULL);
}

#ifdef PSM_OPA
// build and send EXPTID completion ACK. Indicates receiever has gotten
// all TIDs for a given CTS
// for RC QP RDMA, we can use the RC send completion on sender to know
// when all data has been successfully delivered
void
ips_protoexp_send_tid_completion(struct ips_tid_recv_desc *tidrecvc,
				ptl_arg_t sdescid)
{
	ips_epaddr_t *ipsaddr = tidrecvc->ipsaddr;
	struct ips_proto *proto = tidrecvc->protoexp->proto;
	psmi_assert(proto->msgflowid < EP_FLOW_LAST);
	struct ips_flow *flow = &ipsaddr->flows[proto->msgflowid];
	ips_scb_t *scb;

	PSM2_LOG_EPM(OPCODE_EXPTID_COMPLETION,PSM2_LOG_TX, proto->ep->epid,
		    flow->ipsaddr->epaddr.epid ,"sdescid._desc_idx: %d",
		    sdescid._desc_idx);
	scb = tidrecvc->completescb;

	ips_scb_opcode(scb) = OPCODE_EXPTID_COMPLETION;
	scb->ips_lrh.khdr.kdeth0 = 0;
	scb->ips_lrh.data[0] = sdescid;

	/* Attached tidflow gen/seq */
	scb->ips_lrh.mdata = tidrecvc->tidflow_genseq.psn_val;

	psm3_ips_proto_flow_enqueue(flow, scb);
	flow->flush(flow, NULL);

	if (tidrecvc->protoexp->tid_flags & IPS_PROTOEXP_FLAG_CTS_SERIALIZED) {
		flow->flags &= ~IPS_FLOW_FLAG_SKIP_CTS;                                  /* Let the next CTS be processed */
		ips_tid_pendtids_timer_callback(&tidrecvc->protoexp->timer_getreqs, 0);  /* and make explicit progress for it. */
	}
}
#endif // PSM_OPA

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
static
void psmi_deallocate_chb(struct ips_gpu_hostbuf* chb)
{
	PSM3_GPU_HOSTBUF_DESTROY(chb);
	psmi_free(chb);
	return;
}
#endif

#ifdef PSM_HAVE_RDMA
// indicate the given tidsendc has been completed and cleanup after it
static void
ips_protoexp_tidsendc_complete(struct ips_tid_send_desc *tidsendc)
{
#ifdef PSM_VERBS
	struct ips_protoexp *protoexp = tidsendc->protoexp;
#elif defined(PSM_OPA)
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	struct ips_protoexp *protoexp = tidsendc->protoexp;
#endif
#endif
	psm2_mq_req_t req = tidsendc->mqreq;

	_HFI_MMDBG("ips_protoexp_tidsendc_complete\n");
	PSM2_LOG_MSG("entering");

	req->send_msgoff += tidsendc->length;

#ifdef PSM_VERBS
	if (tidsendc->mr) {
		_HFI_MMDBG("send chunk complete, releasing MR: rkey: 0x%x\n", tidsendc->mr->rkey);
		psm3_verbs_release_mr(tidsendc->mr);
		tidsendc->mr = NULL;
	}
#endif

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (req->gpu_hostbuf_used) {
		if (tidsendc->cuda_num_buf == 1) {
			tidsendc->cuda_hostbuf[0]->bytes_read +=
				tidsendc->tid_list.tsess_length;
			if(tidsendc->cuda_hostbuf[0]->bytes_read ==
				tidsendc->cuda_hostbuf[0]->size){
				STAILQ_REMOVE(&req->sendreq_prefetch,
					      tidsendc->cuda_hostbuf[0],
					      ips_gpu_hostbuf, req_next);
				if (tidsendc->cuda_hostbuf[0]->is_tempbuf)
					psmi_deallocate_chb(tidsendc->cuda_hostbuf[0]);
				else {
					tidsendc->cuda_hostbuf[0]->req = NULL;
					tidsendc->cuda_hostbuf[0]->offset = 0;
					tidsendc->cuda_hostbuf[0]->bytes_read = 0;
					PSM3_GPU_HOSTBUF_RESET(tidsendc->cuda_hostbuf[0]);
					psm3_mpool_put(tidsendc->cuda_hostbuf[0]);
				}
				psmi_cuda_run_prefetcher(protoexp, tidsendc);
			}
		} else
			psmi_free(tidsendc->userbuf);
	}
#endif
	/* Check if we can complete the send request. */
	_HFI_MMDBG("ips_protoexp_tidsendc_complete off %u req len %u\n",
		req->send_msgoff, req->req_data.send_msglen);
	if (req->send_msgoff >= req->req_data.send_msglen) {
		psm3_mq_handle_rts_complete(req);
	}

	psm3_mpool_put(tidsendc);
#ifdef PSM_VERBS
	/* we freed an MR  If we have pending sends or pending get requests,
	 * turn on the timer so it can be processed. */
	ips_tid_mravail_callback(protoexp->proto);
#endif

	PSM2_LOG_MSG("leaving");
}
#endif // PSM_HAVE_RDMA

#ifdef PSM_HAVE_RDMA
#if defined(PSM_VERBS)
// our RDMA Write has completed on our send Q (RV or user space RC QP)
// This is called by the send CQE polling which might be within a send
// so it cannot issue any sends directly, otherwise we will have a recursive
// situation and potentially deeper recursion if more send CQEs found
// key notes in this regard:
//	OPA100 code which may send acks here is ifdef'ed out since N/A to RC QP RDMA
//	psm3_mq_handle_rts_complete - sets flags in req and queues it, no callbacks
//	psm3_mpool_put(tidsendc) - tid_desc_send_pool has no callback configured
//	ips_tid_mravail_callback - psmi_timer_request call queues timer for future
//							callback  (no immediate callback)
//	psm3_mpool_put(tidsendc->cuda_hostbuf[0]) - cuda_hostbuf_pool_send has a
//							callback of psmi_gpu_hostbuf_alloc_func which
//							manages cuda buffers but does not issue any sends

int
ips_protoexp_rdma_write_completion(uint64_t wr_id)
{
	struct ips_tid_send_desc *tidsendc = (struct ips_tid_send_desc *)(uintptr_t)wr_id;

	_HFI_MMDBG("ips_protoexp_rdma_write_completion\n");
	PSM2_LOG_MSG("entering");

	ips_protoexp_tidsendc_complete(tidsendc);

	PSM2_LOG_MSG("leaving");
	return IPS_RECVHDRQ_CONTINUE;
}
#elif defined(PSM_OPA)
// sender processing of EXPTID_COMPLETION message from receiver indicating
// receiver has completely received a given TID sequence
int
ips_protoexp_recv_tid_completion(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	struct ips_epaddr *ipsaddr = rcv_ev->ipsaddr;
	ptl_arg_t desc_id = p_hdr->data[0];
	struct ips_tid_send_desc *tidsendc;

	_HFI_MMDBG("ips_protoexp_recv_tid_completion\n");
	PSM2_LOG_MSG("entering");
	PSM2_LOG_EPM(OPCODE_EXPTID_COMPLETION,PSM2_LOG_RX,rcv_ev->ipsaddr->epaddr.epid,
		    rcv_ev->proto->ep->mq->ep->epid,"desc_id._desc_idx: %d",desc_id._desc_idx);

	/* normal packet reliabilty protocol handling */
	if (!ips_proto_is_expected_or_nak(rcv_ev))
	{
		PSM2_LOG_MSG("leaving");
		return IPS_RECVHDRQ_CONTINUE;
	}

	if (__be32_to_cpu(p_hdr->bth[2]) & IPS_SEND_FLAG_ACKREQ)
		ips_proto_send_ack((struct ips_recvhdrq *)rcv_ev->recvq,
				   &ipsaddr->flows[ips_proto_flowid(p_hdr)]);

	psm3_ips_proto_process_ack(rcv_ev);

	/* processing specific to tid_completion packet */
	/*
	 * Get the session send descriptor and complete.
	 */
	tidsendc = (struct ips_tid_send_desc *)
	    psm3_mpool_find_obj_by_index(protoexp->tid_desc_send_pool,
					 desc_id._desc_idx);
	_HFI_VDBG("desc_id=%d (%p)\n", desc_id._desc_idx, tidsendc);
	if (tidsendc == NULL) {
		_HFI_ERROR
		    ("exptid comp: Index %d is out of range\n",
		     desc_id._desc_idx);
		PSM2_LOG_MSG("leaving");
		return IPS_RECVHDRQ_CONTINUE;
	} else {
		ptl_arg_t desc_tidsendc;

		psm3_mpool_get_obj_index_gen_count(tidsendc,
						   &desc_tidsendc._desc_idx,
						   &desc_tidsendc._desc_genc);

		_HFI_VDBG("desc_req:id=%d,gen=%d desc_sendc:id=%d,gen=%d\n",
			  desc_id._desc_idx, desc_id._desc_genc,
			  desc_tidsendc._desc_idx, desc_tidsendc._desc_genc);

		/* See if the reference is still live and valid */
		if (desc_tidsendc.u64 != desc_id.u64) {
			_HFI_ERROR("exptid comp: Genc %d does not match\n",
				desc_id._desc_genc);
			PSM2_LOG_MSG("leaving");
			return IPS_RECVHDRQ_CONTINUE;
		}
	}

	if (!STAILQ_EMPTY(&tidsendc->tidflow.scb_unacked)) {
		struct ips_message_header hdr;

		/* Hack to handle the tidflow */
		hdr.data[0] = rcv_ev->p_hdr->data[0];
		hdr.ack_seq_num = rcv_ev->p_hdr->mdata;
		hdr.khdr.kdeth0 = __cpu_to_le32(3 << HFI_KHDR_TIDCTRL_SHIFT);
		rcv_ev->p_hdr = &hdr;

		/*
		 * This call should directly complete the tidflow
		 * and free all scb on the unacked queue.
		 */
		psm3_ips_proto_process_ack(rcv_ev);

		/* Keep KW happy. */
		rcv_ev->p_hdr = NULL;
		/* Prove that the scb will not leak in the unacked queue: */
		psmi_assert(STAILQ_EMPTY(&tidsendc->tidflow.scb_unacked));
	}

	ips_protoexp_tidsendc_complete(tidsendc);

	PSM2_LOG_MSG("leaving");
	return IPS_RECVHDRQ_CONTINUE;
}
#endif // defined(PSM_VERBS)
#endif // PSM_HAVE_RDMA

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
// our RV RDMA Write has completed with error on our send Q
// This is called by the send CQE polling which might be within a send
// so it cannot issue any sends directly, otherwise we will have a recursive
// situation and potentially deeper recursion if more send CQEs found
// key notes in this regard:
// if we don't return PSM2_OK, caller will consider it an unrecoverable error
int
ips_protoexp_rdma_write_completion_error(psm2_ep_t ep, uint64_t wr_id,
												enum ibv_wc_status wc_status)
{
	struct ips_tid_send_desc *tidsendc = (struct ips_tid_send_desc *)(uintptr_t)wr_id;
	struct ips_protoexp *protoexp;

	PSM2_LOG_MSG("entering");
	if (! tidsendc) {
		psm3_handle_error( PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			"rv RDMA Write with invalid tidsendc: status: '%s' (%d)\n",
			ibv_wc_status_str(wc_status),(int)wc_status);
		goto fail_ret;
	}
	protoexp = tidsendc->protoexp;
	_HFI_MMDBG("failed rv RDMA Write on %s to %s status: '%s' (%d)\n",
			ep->dev_name,
			psm3_epaddr_get_name(tidsendc->ipsaddr->epaddr.epid, 0),
			ibv_wc_status_str(wc_status),(int)wc_status);

	if (! protoexp->proto->ep->verbs_ep.rv_reconnect_timeout)
		goto fail; /* reconnect disabled, can't recover */

	// perhaps depending on wc_status
	// IBV_WC_REM_ACCESS_ERR and others unrecoverable
	// IBV_WC_RETRY_EXC_ERR may be recoverable
	// IBV_WC_RNR_RETRY_EXC_ERR may be recoverable
	// IBV_WC_RESP_TIMEOUT_ERR may be recoverable (is this applicable?)
	// any others?  IB_WC_GENERAL_ERR?

	tidsendc->rv_need_err_chk_rdma = 1;
	tidsendc->is_complete = 0;	// status of send of err_chk_rdma

	/* Add as a pending op and ring up the timer */
	/* ips_tid_pendsend_timer_callback timer will issue ERR_CHK_RDMA */
	STAILQ_INSERT_TAIL(&protoexp->pend_sendq, tidsendc, next);
	psmi_timer_request(protoexp->timerq, &protoexp->timer_send,
		   PSMI_TIMER_PRIO_1);

	return PSM2_OK;

fail:
	psm3_handle_error( PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			"failed rv RDMA Write on %s to %s status: '%s' (%d)\n",
			ep->dev_name,
			psm3_epaddr_get_name(tidsendc->ipsaddr->epaddr.epid, 0),
			ibv_wc_status_str(wc_status),(int)wc_status);
fail_ret:
	PSM2_LOG_MSG("leaving");
	return PSM2_INTERNAL_ERR;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
static psm2_error_t ips_protoexp_send_err_chk_rdma(struct ips_tid_send_desc *tidsendc)
{
	ips_scb_t *scb = NULL;
	struct ips_protoexp *protoexp = tidsendc->protoexp;
	struct ips_proto *proto = protoexp->proto;
	ips_epaddr_t *ipsaddr = tidsendc->ipsaddr;
	struct ips_flow *flow = &ipsaddr->flows[proto->msgflowid];
	psm2_error_t err = PSM2_OK;
	uint32_t conn_count;

	PSM2_LOG_MSG("entering");
	_HFI_MMDBG("ips_protoexp_send_err_chk_rdma\n");

	// we delay our sending of err chk rdma until after the connection is
	// restored as reflected by an increment of conn_count relative to when
	// tidsendc issued the rdma_write.  This also forms a barrier to
	// ensure our err chk rdma does not arrive at receiver prior to the
	// rdma completion (eg. in case we timeded out for RC QP ack but
	// receiver got the full rdma write).
	if (psm3_rv_get_conn_count(proto->ep->rv, ipsaddr->verbs.rv_conn,
			tidsendc->rv_sconn_index, &conn_count)) {
		psm3_handle_error( PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			"send_err_chk_rdma: Connect unrecoverable on %s to %s\n",
			proto->ep->dev_name,
			psm3_epaddr_get_name(ipsaddr->epaddr.epid, 0));
		err = PSM2_TIMEOUT; /* force a resend reschedule */
		goto done;
	}

	// conn_count only advances.  Only need to test for equality.
	// 32b reconnect_count sufficient for 13 years of constant reconnect
	// at 100ms intervals (eg. RV_DELAY) before wrapping
	if (conn_count == tidsendc->rv_conn_count) {
		err = PSM2_TIMEOUT; /* force a resend reschedule */
		goto done;
	}

	// limit to 1 outstanding per remote connection.
	// receiver can only queue 1 response if it's low on scb's
	if (ipsaddr->verbs.rv_err_chk_rdma_outstanding) {
		err = PSM2_TIMEOUT; /* force a resend reschedule */
		goto done;
	}

	scb = psm3_ips_scbctrl_alloc(&protoexp->tid_scbc_rv, 1, 0, 0);
	if (scb == NULL) {
		// ips_tid_scbavail_callback will trigger pend_sendq again
		// and call ips_tid_pendsend_timer_callback
		err = PSM2_EP_NO_RESOURCES;
		goto done;
	}

	_HFI_MMDBG("sending ERR_CHK_RDMA\n");
	PSM2_LOG_EPM(OPCODE_ERR_CHK_RDMA,PSM2_LOG_TX, proto->ep->epid,
			ipsaddr->epaddr.epid,
			"psm3_mpool_get_obj_index(tidsendc->mqreq): %d, tidsendc->rdescid. _desc_genc %d _desc_idx: %d, tidsendc->sdescid._desc_idx: %d",
			psm3_mpool_get_obj_index(tidsendc->mqreq),
			tidsendc->rdescid._desc_genc,tidsendc->rdescid._desc_idx,
			tidsendc->sdescid._desc_idx);

	ips_scb_opcode(scb) = OPCODE_ERR_CHK_RDMA;
	scb->ips_lrh.khdr.kdeth0 = 0;
		// providing our REQ index gives receiver an extra sanity check
	scb->ips_lrh.mdata = psm3_mpool_get_obj_index(tidsendc->mqreq);
	scb->ips_lrh.data[0] = tidsendc->rdescid;
	scb->ips_lrh.data[1] = tidsendc->sdescid;
	/* path is having issue, ask for ack */
	scb->scb_flags |= IPS_SEND_FLAG_ACKREQ;
	/* INTR makes sure remote end works on it immediately */
	if (proto->flags & IPS_PROTO_FLAG_RCVTHREAD)
		scb->scb_flags |= IPS_SEND_FLAG_INTR;

	ipsaddr->verbs.rv_err_chk_rdma_outstanding = 1;
	tidsendc->is_complete = 1;	// status of send of err_chk_rdma

	proto->epaddr_stats.err_chk_rdma_send++;

	psm3_ips_proto_flow_enqueue(flow, scb);
	flow->flush(flow, NULL);

	/* inbound ack will free scb */
done:
	PSM2_LOG_MSG("leaving");
	return err;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
// scan all alternate addresses for "expected" (multi-QP and multi-EP)
// to see if a match for "got" can be found
static
int ips_protoexp_ipsaddr_match(ips_epaddr_t *expected, ips_epaddr_t *got)
{
	ips_epaddr_t *p = expected;

	do {
		if (p == got)
			return 1;
		p = p->next;
	} while (p != expected);

	return 0;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
int ips_protoexp_process_err_chk_rdma(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_proto *proto = rcv_ev->proto;
	struct ips_protoexp *protoexp = proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	ips_epaddr_t *ipsaddr = rcv_ev->ipsaddr;
	__u32 sendtoken = p_hdr->mdata;
	ptl_arg_t rdesc_id = p_hdr->data[0];
	ptl_arg_t sdesc_id = p_hdr->data[1];
	struct ips_tid_recv_desc *tidrecvc;
	psmi_assert(proto->msgflowid < EP_FLOW_LAST);
	struct ips_flow *flow = &ipsaddr->flows[proto->msgflowid];

	PSM2_LOG_MSG("entering");
	_HFI_MMDBG("ips_protoexp_process_err_chk_rdma\n");

	/* normal packet reliabilty protocol handling */
	if (!ips_proto_is_expected_or_nak(rcv_ev))
		goto done;

	/* processing specific to err chk rdma packet */
	proto->epaddr_stats.err_chk_rdma_recv++;

	_HFI_MMDBG("received ERR_CHK_RDMA\n");
	PSM2_LOG_EPM(OPCODE_ERR_CHK_RDMA,PSM2_LOG_RX,ipsaddr->epaddr.epid,
			proto->ep->epid,
			"rdescid._desc_genc %d _desc_idx: %d, sdescid._desc_idx: %d",
			rdesc_id._desc_genc,rdesc_id._desc_idx, sdesc_id._desc_idx);

	if (ipsaddr->verbs.rv_need_send_err_chk_rdma_resp) {
		/* sender has >1 err chk rdma outstanding: protocol violation */
		psm3_handle_error( PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			"process_err_chk_rdma: Protocol Violation: > 1 outstanding from remote node %s on %s\n",
			psm3_epaddr_get_name(ipsaddr->epaddr.epid, 0),
			proto->ep->dev_name);
		goto do_acks;
	}

	/* Get receive descriptor */
	psmi_assert(rdesc_id._desc_idx < HFI_TF_NFLOWS);
	tidrecvc = &protoexp->tfc.tidrecvc[rdesc_id._desc_idx];

	tidrecvc->stats.nErrChkReceived++;

	// stash information to build resp in ipsaddr
	psmi_assert(! ipsaddr->verbs.rv_need_send_err_chk_rdma_resp);
	ipsaddr->verbs.rv_need_send_err_chk_rdma_resp = 1;
	ipsaddr->verbs.rv_err_chk_rdma_resp_rdesc_id = rdesc_id;
	ipsaddr->verbs.rv_err_chk_rdma_resp_sdesc_id = sdesc_id;

	// for the rare case that err_chk_rdma has a rdescid which we completed
	// a while ago, we need to sanity check not only rdescid, but also
	// the identity of the sender and the sendtoken for the senders RTS
	// this protects us in case rdescid generation has wrapped
	if (tidrecvc->rdescid._desc_genc != rdesc_id._desc_genc
		|| tidrecvc->state != TIDRECVC_STATE_BUSY
		|| ! ips_protoexp_ipsaddr_match(tidrecvc->ipsaddr, ipsaddr)
		|| tidrecvc->getreq->tidgr_sendtoken != sendtoken
		) {
		/* Receive descriptor mismatch in time and space.
		 * Must have completed recv for this RDMA
		 * (eg. sender timeout waiting for RC QP ack)
		 */
		ipsaddr->verbs.rv_err_chk_rdma_resp_need_resend = 0;
	} else if (psm3_rv_scan_cq(proto->ep->rv, RV_WC_RECV_RDMA_WITH_IMM,
				RDMA_IMMED_DESC_MASK,
				RDMA_PACK_IMMED(tidrecvc->rdescid._desc_genc,
								tidrecvc->rdescid._desc_idx, 0))) {
		// the CQ scan above solves a very rare race where the receiving QP is
		// very slow to issue CQEs and PSM happens to poll the UD QP and find
		// the err chk rdma before finding a succesful RDMA Write received.
		// Due to reconnection essentially being a barrier, we know the
		// CQE must be processed in RV drain prior to the new connection and
		// hence prior to the err chk rdma on UD QP.  So we scan the RV CQ
		// to close the race, if we find a matching completion we can
		// respond with resend_needed=0 and know we will process the CQE
		// soon to fully complete the RDMA receipt.
		// We ignore RV_IDX in this scan, it should always match us and better
		// to not ask for a resend and fail when we process the completion
		// than to ask for an a resend into a freed buffer
		ipsaddr->verbs.rv_err_chk_rdma_resp_need_resend = 0;
	} else {
		tidrecvc->stats.nReXmit++;
		ipsaddr->verbs.rv_err_chk_rdma_resp_need_resend = 1;
	}

	// try to send it now, will remain "queued" until we can send
	ips_protoexp_send_err_chk_rdma_resp(flow);
	if (ipsaddr->verbs.rv_need_send_err_chk_rdma_resp)
		// ips_tid_scbavail_callback will trigger pend_err_resp again
		// and call ips_tid_pendtids_timer_callback
		STAILQ_INSERT_TAIL(&protoexp->pend_err_resp, ipsaddr, verbs.pend_err_resp_next);

do_acks:
	if (__be32_to_cpu(p_hdr->bth[2]) & IPS_SEND_FLAG_ACKREQ)
		ips_proto_send_ack((struct ips_recvhdrq *)rcv_ev->recvq,
					&ipsaddr->flows[ips_proto_flowid(p_hdr)]);

	psm3_ips_proto_process_ack(rcv_ev);
done:
	PSM2_LOG_MSG("leaving");
	return IPS_RECVHDRQ_CONTINUE;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)


#if defined(PSM_VERBS)
#ifdef RNDV_MOD
static
void ips_protoexp_send_err_chk_rdma_resp(struct ips_flow *flow)
{
	ips_epaddr_t *ipsaddr = flow->ipsaddr;
	struct ips_proto *proto = ipsaddr->epaddr.proto;
	struct ips_protoexp *protoexp = proto->protoexp;
	ips_scb_t *scb;

	PSM2_LOG_MSG("entering");
	_HFI_MMDBG("ips_protoexp_send_err_chk_rdma_resp\n");
	psmi_assert(ipsaddr->verbs.rv_need_send_err_chk_rdma_resp);
	scb = psm3_ips_scbctrl_alloc(&protoexp->tid_scbc_rv, 1, 0, 0);
	if (scb == NULL) {
		/* ips_tid_scbavail_callback() will reschedule */
		return;
	}

	_HFI_MMDBG("sending ERR_CHK_RDMA_RESP\n");
	PSM2_LOG_EPM(OPCODE_ERR_CHK_RDMA,PSM2_LOG_TX, proto->ep->epid,
			ipsaddr->epaddr.epid,
			"need_resend %d rdescid. _desc_genc %d _desc_idx: %d, sdescid._desc_idx: %d",
			ipsaddr->verbs.rv_err_chk_rdma_resp_need_resend,
			ipsaddr->verbs.rv_err_chk_rdma_resp_rdesc_id._desc_genc,
			ipsaddr->verbs.rv_err_chk_rdma_resp_rdesc_id._desc_idx,
			ipsaddr->verbs.rv_err_chk_rdma_resp_sdesc_id._desc_idx);

	ips_scb_opcode(scb) = OPCODE_ERR_CHK_RDMA_RESP;
	scb->ips_lrh.khdr.kdeth0 = 0;
	scb->ips_lrh.mdata = ipsaddr->verbs.rv_err_chk_rdma_resp_need_resend;
	scb->ips_lrh.data[0] = ipsaddr->verbs.rv_err_chk_rdma_resp_rdesc_id;
	scb->ips_lrh.data[1] = ipsaddr->verbs.rv_err_chk_rdma_resp_sdesc_id;
	/* path is having issue, ask for ack */
	scb->scb_flags |= IPS_SEND_FLAG_ACKREQ;
	/* INTR makes sure remote end works on it immediately */
	if (proto->flags & IPS_PROTO_FLAG_RCVTHREAD)
		scb->scb_flags |= IPS_SEND_FLAG_INTR;

	// The scb will own reliable transmission of resp, we can clear flag
	ipsaddr->verbs.rv_need_send_err_chk_rdma_resp = 0;

	proto->epaddr_stats.err_chk_rdma_resp_send++;

	psm3_ips_proto_flow_enqueue(flow, scb);
	flow->flush(flow, NULL);

	PSM2_LOG_MSG("leaving");
	return;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
int ips_protoexp_process_err_chk_rdma_resp(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_protoexp *protoexp = rcv_ev->proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	ips_epaddr_t *ipsaddr = rcv_ev->ipsaddr;
	struct ips_tid_send_desc *tidsendc;
	uint32_t need_resend = p_hdr->mdata;
	//ptl_arg_t rdesc_id = p_hdr->data[0];
	ptl_arg_t sdesc_id = p_hdr->data[1];

	PSM2_LOG_MSG("entering");
	_HFI_MMDBG("ips_protoexp_process_err_chk_rdma_resp\n");

	/* normal packet reliabilty protocol handling */
	if (!ips_proto_is_expected_or_nak(rcv_ev))
		goto done;

	/* processing specific to err chk rdma resp packet */

	protoexp->proto->epaddr_stats.err_chk_rdma_resp_recv++;

	_HFI_MMDBG("received ERR_CHK_RDMA_RESP\n");
	PSM2_LOG_EPM(OPCODE_ERR_CHK_RDMA,PSM2_LOG_RX,ipsaddr->epaddr.epid,
			protoexp->proto->ep->epid,
			"rdescid. _desc_genc %d _desc_idx: %d, sdescid._desc_idx: %d",
			p_hdr->data[0]._desc_genc,p_hdr->data[0]._desc_idx,
			sdesc_id._desc_idx);
	/* Get the session send descriptor
	 * a subset of get_tidflow in ips_proto_recv.c since we don't
	 * have tidflow sequence numbers to check
	 */
	tidsendc = (struct ips_tid_send_desc *)
		psm3_mpool_find_obj_by_index(protoexp->tid_desc_send_pool,
					sdesc_id._desc_idx);
	_HFI_VDBG("desc_id=%d (%p)\n", sdesc_id._desc_idx, tidsendc);
	if (tidsendc == NULL) {
		_HFI_ERROR("err_chk_rdma_resp: Index %d is out of range\n",
					sdesc_id._desc_idx);
		goto do_acks;
	} else {
		ptl_arg_t desc_tidsendc;

		psm3_mpool_get_obj_index_gen_count(tidsendc,
						 &desc_tidsendc._desc_idx, &desc_tidsendc._desc_genc);

		_HFI_VDBG("sdesc_req:id=%d,gen=%d desc_sendc:id=%d,gen=%d\n",
				sdesc_id._desc_idx, sdesc_id._desc_genc,
				desc_tidsendc._desc_idx, desc_tidsendc._desc_genc);

		/* See if the reference is still live and valid */
		if (desc_tidsendc.u64 != sdesc_id.u64) {
			_HFI_ERROR("err_chk_rdma_resp: Genc %d does not match\n",
				sdesc_id._desc_genc);
			goto do_acks;
		}
	}

	ipsaddr->verbs.rv_err_chk_rdma_outstanding = 0;
	tidsendc->rv_need_err_chk_rdma = 0;
	if (need_resend)
		ips_tid_reissue_rdma_write(tidsendc);
	else
		ips_protoexp_tidsendc_complete(tidsendc);

do_acks:
	if (__be32_to_cpu(p_hdr->bth[2]) & IPS_SEND_FLAG_ACKREQ)
		ips_proto_send_ack((struct ips_recvhdrq *)rcv_ev->recvq,
					&ipsaddr->flows[ips_proto_flowid(p_hdr)]);

	psm3_ips_proto_process_ack(rcv_ev);
done:
	PSM2_LOG_MSG("leaving");
	return IPS_RECVHDRQ_CONTINUE;
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

#ifdef PSM_HAVE_RDMA
// Intermediate STL100 EXTID packets can be delivered to software when
// acks are requested.
// The final packet in a STL100 EXTID flow is also delivered to software
// to indicate the completion of the flow and can contain unaligned data.
// for RDMA Write we will simply use immediate data in the write
// to indicate the completed receive of the RDMA Write
// if we use RDMA Read, the local SQ Completion will indicate this
#if defined(PSM_VERBS)
// could build and pass a ips_recvhdrq_event or pass struct ips_recvhdrq
// but all we really need is proto and len
// conn indicates where we received RDMA Write, just for quick sanity check
// 	for RV module conn will be the psm3_rv_conn_t
// 	for user RC QPs conn will be will be the RC struct ibv_qp*
int ips_protoexp_handle_immed_data(struct ips_proto *proto, uint64_t conn_ref,
									int conn_type, uint32_t immed, uint32_t len)
{
	struct ips_tid_recv_desc *tidrecvc;
	struct ips_protoexp *protoexp = proto->protoexp;
	ptl_arg_t desc_id;
	_HFI_MMDBG("ips_protoexp_immed_data\n");
	PSM2_LOG_MSG("entering");
	desc_id._desc_genc = RDMA_UNPACK_IMMED_GENC(immed);
	desc_id._desc_idx = RDMA_UNPACK_IMMED_IDX(immed);
#elif defined(PSM_OPA)
int ips_protoexp_data(struct ips_recvhdrq_event *rcv_ev)
{
	struct ips_proto *proto = rcv_ev->proto;
	struct ips_protoexp *protoexp = proto->protoexp;
	struct ips_message_header *p_hdr = rcv_ev->p_hdr;
	struct ips_tid_recv_desc *tidrecvc;
	ptl_arg_t desc_id;
	psmi_seqnum_t sequence_num;

	psmi_assert(_get_proto_hfi_opcode(p_hdr) == OPCODE_EXPTID);
	_HFI_MMDBG("ips_protoexp_data\n");
	// final packet in a TID sequence, we do some processing here
	// for unaligned start and end bytes and send a OPCODE_EXPTID_COMPLETION

	PSM2_LOG_MSG("entering");

	desc_id._desc_idx = ips_proto_flowid(p_hdr);
	PSM2_LOG_EPM(OPCODE_EXPTID,PSM2_LOG_RX,rcv_ev->ipsaddr->epaddr.epid,
		    proto->ep->mq->ep->epid,"desc_id._desc_idx: %d", desc_id._desc_idx);

	desc_id._desc_genc = p_hdr->exp_rdescid_genc;
#endif

	tidrecvc = &protoexp->tfc.tidrecvc[desc_id._desc_idx];

	if ((tidrecvc->rdescid._desc_genc & IPS_HDR_RDESCID_GENC_MASK)
		!= desc_id._desc_genc) {
#ifndef PSM_OPA
		_HFI_ERROR("stale inbound rv RDMA generation: expected %u got %u\n",
				tidrecvc->rdescid._desc_genc, desc_id._desc_genc);
		tidrecvc->stats.nGenErr++;
#endif
		PSM2_LOG_MSG("leaving");
		return IPS_RECVHDRQ_CONTINUE;		/* skip */
	}

#ifndef PSM_OPA
	// maybe should use assert below so don't add test in production code
	if (tidrecvc->state != TIDRECVC_STATE_BUSY) {
		_HFI_ERROR("stale inbound rv RDMA (tidrecvc not busy)\n");
		PSM2_LOG_MSG("leaving");
		return IPS_RECVHDRQ_CONTINUE;		/* skip */
	}
	// some sanity checks
	// maybe this should be an assert so don't add test in production code
	if (len != tidrecvc->recv_msglen) {
		// RDMA Write does not match what we asked for in CTS
		_HFI_ERROR("incorrect RDMA Write Len: expected %u got %u\n",
				tidrecvc->recv_msglen, len);
		// TBD - what to do?
	}
	psmi_assert(IPS_PROTOEXP_FLAG_ENABLED & tidrecvc->protoexp->proto->ep->rdmamode);
#ifdef RNDV_MOD
	if (conn_type == RDMA_IMMED_RV
		&& RDMA_UNPACK_IMMED_RV_IDX(immed) != proto->ep->verbs_ep.rv_index) {
		// RV module should not have delivered this CQE to us
		_HFI_ERROR("incorrect RDMA RV Index: expected %u got %u\n",
				proto->ep->verbs_ep.rv_index, RDMA_UNPACK_IMMED_RV_IDX(immed));
		return IPS_RECVHDRQ_CONTINUE;		/* skip */
	}
#endif
#if defined(USE_RC)
	// For User RC conn_ref is context we set in rc_qp_create (*ipsaddr)
	// For Kernel RC, conn_ref is the conn handle (psm3_rv_conn_get_conn_handle)
	// maybe this should be an assert so don't add test in production code
	if ((conn_type == RDMA_IMMED_USER_RC)
			&& (uint64_t)tidrecvc->ipsaddr != conn_ref) {
		// RDWA Write is not on expected RC QP from remote node
		_HFI_ERROR("RDMA Write on Wrong User QP 0x%"PRIx64", expect 0x%"PRIx64"\n",
				 	conn_ref, (uint64_t)tidrecvc->ipsaddr);
		// TBD - what to do?
	}
#endif
#ifdef RNDV_MOD
	if (conn_type == RDMA_IMMED_RV
		&& psm3_rv_conn_get_conn_handle(tidrecvc->ipsaddr->verbs.rv_conn)
					 != conn_ref) {
		// RDWA Write is not on expected RV QP from remote node
		_HFI_ERROR("RDMA Write on Wrong RV QP 0x%"PRIx64", expect 0x%"PRIx64"\n",
				 	conn_ref,
		 			psm3_rv_conn_get_conn_handle(tidrecvc->ipsaddr->verbs.rv_conn));
		// TBD - what to do?
	}
#endif
	if (_HFI_PDBG_ON) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		if (tidrecvc->is_ptr_gpu_backed)
			_HFI_PDBG_DUMP_GPU_ALWAYS(tidrecvc->buffer, len);
		else
#endif
			_HFI_PDBG_DUMP_ALWAYS(tidrecvc->buffer, len);
	}
#else // PSM_OPA
	/* IBTA CCA handling for expected flow. */
	if (rcv_ev->is_congested & IPS_RECV_EVENT_FECN) {
		/* Mark flow to generate BECN in control packet */
		tidrecvc->tidflow.flags |= IPS_FLOW_FLAG_GEN_BECN;
		/* Update stats for congestion encountered */
		proto->epaddr_stats.congestion_pkts++;
		/* Clear FECN event */
		rcv_ev->is_congested &= ~IPS_RECV_EVENT_FECN;
	}

	sequence_num.psn_val = __be32_to_cpu(p_hdr->bth[2]);

	if_pf (PSM_HAL_ERROR_OK != psmi_hal_tidflow_check_update_pkt_seq(
		    protoexp,sequence_num,tidrecvc,p_hdr,
		    ips_protoexp_do_tf_generr,ips_protoexp_do_tf_seqerr))
			return IPS_RECVHDRQ_CONTINUE;
#endif // PSM_OPA

	/* Reset the swapped generation count as we received a valid packet */
	tidrecvc->tidflow_nswap_gen = 0;

	/* Do some sanity checking */
	psmi_assert_always(tidrecvc->state == TIDRECVC_STATE_BUSY);
#ifndef PSM_OPA
	// STL100 does this at the end of ips_protoexp_send_tid_completion
	// TBD - seems like this should be done after ips_tid_recv_free
	// so we have more likelihood of getting freshly freed resources?
	if (tidrecvc->protoexp->tid_flags & IPS_PROTOEXP_FLAG_CTS_SERIALIZED) {
		tidrecvc->ipsaddr->flows[protoexp->proto->msgflowid].flags &= ~IPS_FLOW_FLAG_SKIP_CTS;                                  /* Let the next CTS be processed */
		ips_tid_pendtids_timer_callback(&tidrecvc->protoexp->timer_getreqs, 0);  /* and make explicit progress for it. */
	}
#else
	int recv_completion = (tidrecvc->recv_tidbytes ==
			       (p_hdr->exp_offset + ips_recvhdrq_event_paylen(rcv_ev)));

	/* If sender requested an ACK with the packet and it is not the last
	 * packet, or if the incoming flow faced congestion, respond with an
	 * ACK packet. The ACK when congested will have the BECN bit set.
	 */
	if (((__be32_to_cpu(p_hdr->bth[2]) & IPS_SEND_FLAG_ACKREQ) &&
		!recv_completion) ||
	    (tidrecvc->tidflow.flags & IPS_FLOW_FLAG_GEN_BECN)) {
		ips_scb_t ctrlscb;

		/* Ack sender with descriptor index */
		ctrlscb.scb_flags = 0;
		ctrlscb.ips_lrh.data[0] = p_hdr->exp_sdescid;
		ctrlscb.ips_lrh.ack_seq_num = tidrecvc->tidflow_genseq.psn_val;

		// no payload, pass cksum so non-NULL
		psm3_ips_proto_send_ctrl_message(&tidrecvc->tidflow,
					    OPCODE_ACK,
					    &tidrecvc->ctrl_msg_queued,
					    &ctrlscb, ctrlscb.cksum, 0);
	}

	/* If RSM is a HW capability, and RSM has found a TID packet marked
	 * with FECN, the payload will be written to the eager buffer, and
	 * we will have a payload pointer here.  In that case, copy the payload
	 * into the user's buffer.  If RSM did not intercept this EXPTID
	 * packet, the HFI will handle the packet payload. Possibly should
	 * assert(0 < paylen < MTU).
	 */
	if (psmi_hal_has_cap(PSM_HAL_CAP_RSM_FECN_SUPP) &&
	    ips_recvhdrq_event_payload(rcv_ev) &&
	    ips_recvhdrq_event_paylen(rcv_ev))
		psm3_mq_mtucpy(tidrecvc->buffer + p_hdr->exp_offset,
			       ips_recvhdrq_event_payload(rcv_ev),
			       ips_recvhdrq_event_paylen(rcv_ev));

	/* If last packet then we are done. We send a tid transfer completion
	 * packet back to sender, free all tids and close the current tidflow
	 * as well as tidrecvc descriptor.
	 * Note: If we were out of tidflow, this will invoke the callback to
	 * schedule pending transfer.
	 */
	if (recv_completion) {
		/* copy unaligned data if any */
		uint8_t *dst, *src;

		if (tidrecvc->tid_list.tsess_unaligned_start) {
			dst = (uint8_t *)tidrecvc->buffer;
			src = (uint8_t *)p_hdr->exp_ustart;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (tidrecvc->is_ptr_gpu_backed) {
				PSM3_GPU_MEMCPY_HTOD(dst, src,
					tidrecvc->tid_list.tsess_unaligned_start);
			} else
#endif
				ips_protoexp_unaligned_copy(dst, src,
							    tidrecvc->tid_list.tsess_unaligned_start);
		}

		if (tidrecvc->tid_list.tsess_unaligned_end) {
			dst = (uint8_t *)tidrecvc->buffer +
				tidrecvc->recv_msglen -
				tidrecvc->tid_list.tsess_unaligned_end;
			src = (uint8_t *)p_hdr->exp_uend;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (tidrecvc->is_ptr_gpu_backed) {
				PSM3_GPU_MEMCPY_HTOD(dst, src,
					tidrecvc->tid_list.tsess_unaligned_end);
			} else
#endif
			  ips_protoexp_unaligned_copy(dst, src,
						      tidrecvc->tid_list.tsess_unaligned_end);
		}

		/* reply tid transfer completion packet to sender */
		ips_protoexp_send_tid_completion(tidrecvc, p_hdr->exp_sdescid);
#endif

		/* Mark receive as done */
		ips_tid_recv_free(tidrecvc);
		_HFI_MMDBG("tidrecv done\n");
#ifdef PSM_OPA
	}
#endif
	PSM2_LOG_MSG("leaving");

	return IPS_RECVHDRQ_CONTINUE;
}
#endif // PSM_HAVE_RDMA

#ifdef PSM_OPA
#ifndef PSM_DEBUG
#  define ips_dump_tids(tid_list, msg, ...)
#else
static
void ips_dump_tids(ips_tid_session_list *tid_list, const char *msg, ...)
{
	char buf[256];
	size_t off = 0;
	int i, num_tids = tid_list->tsess_tidcount;

	va_list argptr;
	va_start(argptr, msg);
	off += vsnprintf(buf, sizeof(buf) - off, msg, argptr);
	va_end(argptr);

	for (i = 0; i < num_tids && off < (sizeof(buf) - 1); i++)
		off += snprintf(buf + off, sizeof(buf) - off, "%d%s",
				IPS_TIDINFO_GET_TID(tid_list->tsess_list[i]),
				i < num_tids - 1 ? "," : "");

	_HFI_VDBG("%s\n", buf);
	return;
}
#endif
#endif // PSM_OPA

#ifdef PSM_OPA
static
void ips_expsend_tiderr(struct ips_tid_send_desc *tidsendc)
{
	char buf[256];
	size_t off = 0;
	int i;

	off += snprintf(buf + off, sizeof(buf) - off,
			"Remaining bytes: %d Member id %d is not in tid_session_id=%d :",
			tidsendc->remaining_tidbytes, tidsendc->tid_idx,
			tidsendc->rdescid._desc_idx);

	for (i = 0; i < tidsendc->tid_list.tsess_tidcount + 1; i++)
		off += snprintf(buf + off, sizeof(buf) - off, "%d,",
				IPS_TIDINFO_GET_TID(tidsendc->tid_list.
						    tsess_list[i]));
	psm3_handle_error(PSMI_EP_NORETURN, PSM2_INTERNAL_ERR,
			  "Trying to use tid idx %d and there are %d members: %s\n",
			  tidsendc->tid_idx, tidsendc->tid_list.tsess_tidcount,
			  buf);
	return;
}
#endif // PSM_OPA

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
static
psm2_error_t
psmi_cuda_reclaim_hostbufs(struct ips_tid_get_request *getreq)
{
	struct ips_protoexp *protoexp = getreq->tidgr_protoexp;
	struct ips_tid_getreq_cuda_hostbuf_pend *cmemcpyhead =
		&getreq->pend_cudabuf;
	struct ips_gpu_hostbuf *chb;

	/* Get the getreq's first memcpy op */
	while (!STAILQ_EMPTY(cmemcpyhead)) {
		chb = STAILQ_FIRST(cmemcpyhead);
		if (!PSM3_GPU_MEMCPY_DONE(chb)) {
			/* At least one of the copies is still
			 * in progress. Schedule the timer,
			 * then leave the CUDA progress phase
			 * and check for other pending TID work.
			 */
			psmi_timer_request(protoexp->timerq,
					   &protoexp->timer_getreqs,
					   PSMI_TIMER_PRIO_1);
			return PSM2_OK_NO_PROGRESS;
		}
		/* The getreq's oldest cudabuf is done. Reclaim it. */
		getreq->tidgr_cuda_bytesdone += chb->size;
		STAILQ_REMOVE_HEAD(cmemcpyhead, next);
		PSM3_GPU_HOSTBUF_RESET(chb);
		psm3_mpool_put(chb);
	}
	return PSM2_OK;
}
static
struct ips_gpu_hostbuf* psmi_allocate_chb(uint32_t window_len)
{
	struct ips_gpu_hostbuf* chb = (struct ips_gpu_hostbuf*)
						psmi_calloc(PSMI_EP_NONE,
							    UNDEFINED, 1,
							    sizeof(struct ips_gpu_hostbuf));
	if (chb == NULL) {
		psm3_handle_error(PSMI_EP_NORETURN, PSM2_NO_MEMORY,
						"Couldn't allocate cuda host buffers ");
	}
	PSM3_GPU_HOSTBUF_FORCE_INIT(chb, window_len);
	return chb;
}

static
void psmi_cuda_run_prefetcher(struct ips_protoexp *protoexp,
			      struct ips_tid_send_desc *tidsendc)
{
	struct ips_proto *proto = protoexp->proto;
	struct ips_gpu_hostbuf *chb = NULL;
	psm2_mq_req_t req = tidsendc->mqreq;
	uint32_t offset, window_len;

	/* try to push the prefetcher forward */
	if (req->prefetch_send_msgoff < req->req_data.send_msglen) {
		/* some data remains to be sent */
		offset = req->prefetch_send_msgoff;
		window_len =
			ips_cuda_next_window(
#ifdef PSM_OPA
					     tidsendc->ipsaddr->opa.window_rv,
#else
					     proto->mq->hfi_base_window_rv,
#endif
					     offset, req->req_data.buf_len);
		unsigned bufsz = 0;
		if (window_len <= CUDA_SMALLHOSTBUF_SZ) {
			chb = (struct ips_gpu_hostbuf *) psm3_mpool_get(
				proto->cuda_hostbuf_pool_small_send);
			bufsz = proto->cuda_hostbuf_small_send_cfg.bufsz;
		}
		if (chb == NULL) {
			chb = (struct ips_gpu_hostbuf *) psm3_mpool_get(
				proto->cuda_hostbuf_pool_send);
			bufsz = proto->cuda_hostbuf_send_cfg.bufsz;
		}
		/* were any buffers available for the prefetcher? */
		if (chb == NULL)
			return;
		req->prefetch_send_msgoff += window_len;
		chb->offset = offset;
		chb->size = window_len;
		chb->req = req;
		chb->gpu_buf = (uint8_t*)req->req_data.buf + offset;
		chb->bytes_read = 0;

		PSM3_GPU_MEMCPY_DTOH_START(proto, chb, window_len, bufsz);

		STAILQ_INSERT_TAIL(&req->sendreq_prefetch, chb, req_next);
		return;
	}
	return;
}

static
void psmi_attach_chb_to_tidsendc(struct ips_protoexp *protoexp,
				 psm2_mq_req_t req,
				 struct ips_tid_send_desc *tidsendc,
				 struct ips_gpu_hostbuf *chb_prev,
				 uint32_t tsess_srcoff,
				 uint32_t tsess_length,
				 uint32_t tsess_unaligned_start,
				 psm2_chb_match_type_t type)
{
	struct ips_proto *proto = protoexp->proto;
	struct ips_gpu_hostbuf *chb = NULL;
	uint32_t offset, window_len, attached=0;

	/* try to push the prefetcher forward */
	while (req->prefetch_send_msgoff < tsess_srcoff + tsess_length) {
		/* some data remains to be sent */
		offset = req->prefetch_send_msgoff;
		window_len =
			ips_cuda_next_window(
#ifdef PSM_OPA
					     tidsendc->ipsaddr->opa.window_rv,
#else
					     proto->mq->hfi_base_window_rv,
#endif
					     offset, req->req_data.buf_len);
		unsigned bufsz = 0;
		if (window_len <= CUDA_SMALLHOSTBUF_SZ) {
			chb = (struct ips_gpu_hostbuf *) psm3_mpool_get(
				proto->cuda_hostbuf_pool_small_send);
			bufsz = proto->cuda_hostbuf_small_send_cfg.bufsz;
		}
		if (chb == NULL) {
			chb = (struct ips_gpu_hostbuf *) psm3_mpool_get(
				proto->cuda_hostbuf_pool_send);
			bufsz = proto->cuda_hostbuf_send_cfg.bufsz;
		}

		/* were any buffers available? If not force allocate */
		if (chb == NULL) {
			chb = psmi_allocate_chb(window_len);
			psmi_assert(chb);
			chb->is_tempbuf = 1;
		}
		req->prefetch_send_msgoff += window_len;
		chb->offset = offset;
		chb->size = window_len;
		chb->req = req;
		chb->gpu_buf = (uint8_t*)req->req_data.buf + offset;
		chb->bytes_read = 0;

		PSM3_GPU_MEMCPY_DTOH_START(proto, chb, window_len, bufsz);

		STAILQ_INSERT_TAIL(&req->sendreq_prefetch, chb, req_next);
		if (type == PSMI_CUDA_PARTIAL_MATCH_FOUND) {
			if ((tsess_srcoff < chb->offset)
			     && ((tsess_srcoff + tsess_length) > chb->offset)) {
				tidsendc->cuda_hostbuf[0] = chb_prev;
				tidsendc->cuda_hostbuf[1] = chb;
				tidsendc->cuda_num_buf = 2;
				void *buffer = psmi_malloc(PSMI_EP_NONE, UNDEFINED,
						tsess_length);
				tidsendc->userbuf =
					(void *)((uintptr_t) buffer);
				tidsendc->buffer =
					(void *)((uintptr_t)tidsendc->userbuf +
						tsess_unaligned_start);
				return;
			}
		} else {
			if (attached) {
				tidsendc->cuda_hostbuf[0] = chb_prev;
				tidsendc->cuda_hostbuf[1] = chb;
				tidsendc->cuda_num_buf = 2;
				void *buffer = psmi_malloc(PSMI_EP_NONE, UNDEFINED,
						tsess_length);
				tidsendc->userbuf =
					(void *)((uintptr_t) buffer);
				tidsendc->buffer =
					(void *)((uintptr_t)tidsendc->userbuf +
						tsess_unaligned_start);
				attached = 0;
				return;
			}
			if ((tsess_srcoff > chb->offset)
			    && (tsess_srcoff < (chb->offset + chb->size))
			     && ((tsess_srcoff + tsess_length) > (chb->offset + chb->size))) {
				chb_prev = chb;
				attached = 1;
				chb = NULL;
				continue;
			} else if ((chb->offset <= tsess_srcoff) &&
				  ((tsess_srcoff + tsess_length) <=
				   (chb->offset+chb->size))) {
				tidsendc->cuda_hostbuf[0] = chb;
				tidsendc->cuda_hostbuf[1] = NULL;
				tidsendc->cuda_num_buf = 1;
				tidsendc->userbuf =
					(void *)((uintptr_t) chb->host_buf +
						tsess_srcoff - chb->offset);
				tidsendc->buffer =
					(void *)((uintptr_t)tidsendc->userbuf +
							tsess_unaligned_start );
				return;
			} else
				chb = NULL;
		}
	}
}

static
psm2_chb_match_type_t psmi_find_match_in_prefeteched_chb(struct ips_gpu_hostbuf* chb,
				       ips_tid_session_list *tid_list,
				       uint32_t prefetch_send_msgoff)
{
	/* To get a match:
	 * 1. Tid list offset + length is contained within a chb
	 * 2. Tid list offset + length is contained within
	 * the prefetched offset of this req.
	 * 3. Tid list offset + length is partially prefetched
	 * within one chb. (A partial match)
	 */
	if (chb->offset <= tid_list->tsess_srcoff) {
		if ((chb->offset + chb->size) >=
		    (tid_list->tsess_srcoff + tid_list->tsess_length)) {
			return PSMI_CUDA_FULL_MATCH_FOUND;
		} else {
			if((chb->offset + chb->size) > tid_list->tsess_srcoff){
				if(((chb->offset + (2 * chb->size)) >
				   (tid_list->tsess_srcoff + tid_list->tsess_length)) &&
						  ((prefetch_send_msgoff) >=
						   (tid_list->tsess_srcoff + tid_list->tsess_length))){
					return PSMI_CUDA_SPLIT_MATCH_FOUND;
				} else if((tid_list->tsess_srcoff + tid_list->tsess_length)
					> prefetch_send_msgoff) {
					return PSMI_CUDA_PARTIAL_MATCH_FOUND;
				}
			}
		}
	}
	return PSMI_CUDA_CONTINUE;
}
#endif

// sender handling of a CTS which indicates use of TID protocol
psm2_error_t
psm3_ips_tid_send_handle_tidreq(struct ips_protoexp *protoexp,
			   ips_epaddr_t *ipsaddr,
			   psm2_mq_req_t req,
			   ptl_arg_t rdescid,
			   uint32_t tidflow_genseq,
			   ips_tid_session_list *tid_list,
			   uint32_t tid_list_size)
{
	struct ips_tid_send_desc *tidsendc;
#ifdef PSM_OPA
	uint32_t i, j, *src, *dst;
#endif
	_HFI_MMDBG("psm3_ips_tid_send_handle_tidreq\n");

	PSM2_LOG_MSG("entering");
#ifdef PSM_OPA
	psmi_assert(tid_list_size > sizeof(ips_tid_session_list));
	psmi_assert(tid_list_size <= sizeof(tidsendc->filler));
	psmi_assert(tid_list->tsess_tidcount > 0);
	psmi_assert((rdescid._desc_genc>>16) == 0);
#else
	psmi_assert(tid_list_size == sizeof(ips_tid_session_list));
#endif

	tidsendc = (struct ips_tid_send_desc *)
	    psm3_mpool_get(protoexp->tid_desc_send_pool);
	if (tidsendc == NULL) {
		PSM2_LOG_MSG("leaving");
		ips_logevent(protoexp->proto, tid_send_reqs, ipsaddr);
		return PSM2_EP_NO_RESOURCES;
	}

	req->ptl_req_ptr = (void *)tidsendc;
	tidsendc->protoexp = protoexp;

	/* Uniquely identify this send descriptor in space and time */
	tidsendc->sdescid._desc_idx = psm3_mpool_get_obj_index(tidsendc);
	tidsendc->sdescid._desc_genc = psm3_mpool_get_obj_gen_count(tidsendc);
	tidsendc->rdescid = rdescid;
	tidsendc->ipsaddr = ipsaddr;
	tidsendc->mqreq = req;

	/* Copy received tidinfo to local tidsendc buffer. */
	psm3_mq_mtucpy_host_mem(&tidsendc->tid_list, tid_list,
				sizeof(ips_tid_session_list));
#if defined(PSM_SOCKETS) && PSMI_HAL_INST_CNT == 1
	psmi_assert_always(0);	// should not get here
#elif defined(PSM_VERBS)
	// for UD we do not need a ips_flow since we will use the RC QP and
	// then will use our main flow for the final RV completion control msg
	// The path record for use by RDMA will be selected when the connection
	// is established
	tidsendc->mr = NULL;	// be safe,but should be NULL since clear on release
	_HFI_VDBG("recv'd CTS: rkey 0x%x srcoff %u raddr 0x%"PRIx64" len %u\n",
		tid_list->tsess_rkey, tid_list->tsess_srcoff, tid_list->tsess_raddr,
		tid_list->tsess_length);
#elif defined(PSM_OPA)
	/*
	 * while doing the copy, we try to merge the tids based on
	 * following rules:
	 * 1. both tids are virtually contiguous(i and i+1 in the array);
	 * 2. both tids have the same tidpair value;
	 * 3. first tid (i) has tidctrl=1;
	 * 4. second tid (i+1) has tidctrl=2;
	 * 5. total length does not exceed 512 pages (2M);
	 * 6. The h/w supports merged tid_ctrl's.
	 *
	 * The restriction of 512 pages comes from the limited number
	 * of bits we have for KDETH.OFFSET:
	 *   - The entire mapping space provided through TIDs is to be
	 *     viewed as a zero-based address mapping.
	 *   - We have 15 bits in KDETH offset field through which we
	 *     can address upto a maximum of 2MB.
	 *     (with 64-byte offset mode or KDETH.OM = 1)
	 *   - Assuming a 4KB page size, 2MB/4KB = 512 pages.
	 */
	ips_dump_tids(tid_list, "Received %d tids: ",
				tid_list->tsess_tidcount);

	if (psmi_hal_has_cap(PSM_HAL_CAP_MERGED_TID_CTRLS))
	{
		src = tid_list->tsess_list;
		dst = tidsendc->tid_list.tsess_list;
		dst[0] = src[0];
		j = 0; i = 1;
		while (i < tid_list->tsess_tidcount) {
			if ((((dst[j]>>IPS_TIDINFO_TIDCTRL_SHIFT)+1) ==
			     (src[i]>>IPS_TIDINFO_TIDCTRL_SHIFT)) &&
			    (((dst[j]&IPS_TIDINFO_LENGTH_MASK)+
			      (src[i]&IPS_TIDINFO_LENGTH_MASK)) <=
			     		PSM_MAX_NUM_PAGES_IN_TIDPAIR)) {
				/* merge 'i' to 'j'
				 * (We need to specify "tidctrl" value as 3
				 *  if we merge the individual tid-pairs.
				 *  Doing that here) */
				dst[j] += (2 << IPS_TIDINFO_TIDCTRL_SHIFT) +
					(src[i] & IPS_TIDINFO_LENGTH_MASK);
				i++;
				if (i == tid_list->tsess_tidcount) break;
			}
			j++;
			/* copy 'i' to 'j' */
			dst[j] = src[i];
			i++;
		}
		tidsendc->tid_list.tsess_tidcount = j + 1;
		tid_list = &tidsendc->tid_list;
	}
	else
	{
		tidsendc->tid_list.tsess_tidcount = tid_list->tsess_tidcount;
		psm3_mq_mtucpy(&tidsendc->tid_list.tsess_list, tid_list->tsess_list,
			       tid_list->tsess_tidcount * sizeof(tid_list->tsess_list[0]));
		tid_list = &tidsendc->tid_list;
	}

	/* Initialize tidflow for window. Use path requested by remote endpoint */
	psm3_ips_flow_init(&tidsendc->tidflow, protoexp->proto, ipsaddr,
		      protoexp->tid_xfer_type, PSM_PROTOCOL_TIDFLOW,
		      IPS_PATH_LOW_PRIORITY, EP_FLOW_TIDFLOW);
	tidsendc->tidflow.xmit_seq_num.psn_val = tidflow_genseq;
	tidsendc->tidflow.xmit_ack_num.psn_val = tidflow_genseq;
	tidsendc->frag_size = min(protoexp->tid_send_fragsize,
		tidsendc->tidflow.frag_size);
#endif // defined(PSM_VERBS)

	tidsendc->userbuf =
	    (void *)((uintptr_t) req->req_data.buf + tid_list->tsess_srcoff);
	tidsendc->buffer = (void *)((uintptr_t)tidsendc->userbuf
#ifdef PSM_OPA
				+ tid_list->tsess_unaligned_start
#endif
				);
	tidsendc->length = tid_list->tsess_length;
	_HFI_MMDBG("tidsendc created userbuf %p buffer %p length %u\n",
			tidsendc->userbuf,  tidsendc->buffer, tidsendc->length);
#ifdef PSM_OPA
	tidsendc->ctrl_msg_queued = 0;
#endif

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	/* Matching on previous prefetches and initiating next prefetch */
	struct ips_gpu_hostbuf *chb = NULL, *chb_next = NULL;
	psm2_chb_match_type_t rc = PSMI_CUDA_CONTINUE;

	/* check if the prefetcher has a buffer ready to use */
	tidsendc->cuda_hostbuf[0] = NULL;
	tidsendc->cuda_hostbuf[1] = NULL;
	tidsendc->cuda_num_buf = 0;
	if (req->gpu_hostbuf_used) {
		/* To get a match:
		 * 1. Tid list offset + length is contained within a chb
		 * 2. Tid list offset + length is contained within
		 * the prefetched offset of this req.
		 * 3. Tid list offset + length is partially prefetched
		 * within one chb. (A partial match)
		 */
		STAILQ_FOREACH(chb, &req->sendreq_prefetch, req_next) {
			rc = psmi_find_match_in_prefeteched_chb(chb,
								tid_list,
								req->prefetch_send_msgoff);
			if (rc < PSMI_CUDA_CONTINUE)
				break;
		}
		if (rc == PSMI_CUDA_FULL_MATCH_FOUND) {
			tidsendc->userbuf =
				(void *)((uintptr_t) chb->host_buf+
					 tid_list->tsess_srcoff - chb->offset);
			tidsendc->buffer =
				(void *)((uintptr_t)tidsendc->userbuf
#ifdef PSM_OPA
					 + tid_list->tsess_unaligned_start
#endif
					);
			/* now associate the buffer with the tidsendc */
			tidsendc->cuda_hostbuf[0] = chb;
			tidsendc->cuda_hostbuf[1] = NULL;
			tidsendc->cuda_num_buf = 1;
		} else if (rc == PSMI_CUDA_SPLIT_MATCH_FOUND){
			void *buffer = psmi_malloc(PSMI_EP_NONE, UNDEFINED,
					tid_list->tsess_length);
			tidsendc->userbuf =
				(void *)((uintptr_t) buffer);
			tidsendc->buffer =
				(void *)((uintptr_t)tidsendc->userbuf
#ifdef PSM_OPA
				+ tid_list->tsess_unaligned_start
#endif
				);
			chb_next = STAILQ_NEXT(chb, req_next);
			tidsendc->cuda_hostbuf[0] = chb;
			tidsendc->cuda_hostbuf[1] = chb_next;
			tidsendc->cuda_num_buf = 2;
		} else if (rc == PSMI_CUDA_PARTIAL_MATCH_FOUND) {
			psmi_attach_chb_to_tidsendc(protoexp, req,
						    tidsendc,
						    chb,
						    tid_list->tsess_srcoff,
						    tid_list->tsess_length,
#ifdef PSM_OPA
						    tid_list->tsess_unaligned_start,
#else
							0,
#endif
						    rc);
		} else {
			psmi_attach_chb_to_tidsendc(protoexp, req,
						    tidsendc,
						    NULL,
						    tid_list->tsess_srcoff,
						    tid_list->tsess_length,
#ifdef PSM_OPA
						    tid_list->tsess_unaligned_start,
#else
							0,
#endif
						    PSMI_CUDA_CONTINUE);
		}
		protoexp->proto->strat_stats.rndv_rdma_hbuf_send++;
		protoexp->proto->strat_stats.rndv_rdma_hbuf_send_bytes += tid_list->tsess_length;
	} else if (req->is_buf_gpu_mem) {
		protoexp->proto->strat_stats.rndv_rdma_gdr_send++;
		protoexp->proto->strat_stats.rndv_rdma_gdr_send_bytes += tid_list->tsess_length;
	} else
#endif // PSM_CUDA || PSM_ONEAPI
	{
		protoexp->proto->strat_stats.rndv_rdma_cpu_send++;
		protoexp->proto->strat_stats.rndv_rdma_cpu_send_bytes += tid_list->tsess_length;
	}

#ifdef PSM_OPA
	/* frag size must be 64B multiples */
	tidsendc->frag_size &= (~63);
#endif
	tidsendc->is_complete = 0;
#ifdef PSM_OPA
	tidsendc->tid_idx = 0;
	tidsendc->frame_send = 0;
#else
	tidsendc->reserved = 0;
#ifdef PSM_HAVE_RNDV_MOD
	tidsendc->rv_need_err_chk_rdma = 0;
	tidsendc->rv_sconn_index = 0;
	tidsendc->rv_conn_count = 0;
#endif
#endif

#ifdef PSM_OPA
	tidsendc->tidbytes = 0;
	tidsendc->remaining_tidbytes = tid_list->tsess_length
	    - tid_list->tsess_unaligned_start - tid_list->tsess_unaligned_end;
	tidsendc->remaining_bytes_in_tid =
	    (IPS_TIDINFO_GET_LENGTH(tid_list->tsess_list[0]) << 12) -
	    tid_list->tsess_tidoffset;
	tidsendc->offset_in_tid = tid_list->tsess_tidoffset;
#endif

	_HFI_EXP
	    ("alloc tidsend=%4d tidrecv=%4d srcoff=%6d length=%6d"
#ifdef PSM_OPA
		",s=%d,e=%d"
#endif
		"\n",
	     tidsendc->sdescid._desc_idx, rdescid._desc_idx,
	     tid_list->tsess_srcoff, tid_list->tsess_length
#ifdef PSM_OPA
	     , tid_list->tsess_unaligned_start, tid_list->tsess_unaligned_end
#endif
		);

	// start sending TIDEXP packets
	ips_tid_send_exp(tidsendc);

	/* Add as a pending op and ring up the timer */
	if (tidsendc->is_complete == 0) {
		STAILQ_INSERT_TAIL(&protoexp->pend_sendq, tidsendc, next);
		psmi_timer_request(protoexp->timerq, &protoexp->timer_send,
			   PSMI_TIMER_PRIO_1);
	}

	PSM2_LOG_MSG("leaving");
	/* Consider breaking out of progress engine here */
	return PSM2_OK;
}

#ifdef PSM_OPA
// compose a sequence of EXPTID packets to be sent
// builds one scb with proper headers and tids.  When using PIO
// the scb is for a single packet. When using SDMA, header generation
// will let the scb describe a frag_size larger than a packet
static
ips_scb_t *
ips_scb_prepare_tid_sendctrl(struct ips_flow *flow,
			     struct ips_tid_send_desc *tidsendc)
{
	struct ips_protoexp *protoexp = tidsendc->protoexp;
	uint32_t *tsess_list = tidsendc->tid_list.tsess_list;
	uint32_t tid, omode, offset, chunk_size;
	uint32_t startidx, endidx;
	uint32_t frame_len, nfrag;
	uint8_t *bufptr = tidsendc->buffer;
	ips_scb_t *scb;

	uint8_t is_payload_per_frag_leq_8dw = 0;
	 /* If payload in the first and last nfrag is less then or equal
	  * to 8DW we disable header suppression so as to detect uncorrectable
	  * errors which will otherwise be non-detectable(since header is
	  * suppressed we lose RHF.EccErr)
	  */
	if ((scb = psm3_ips_scbctrl_alloc(&protoexp->tid_scbc_rv, 1, 0, 0)) == NULL)
		return NULL;

	/*
	 * Make sure the next offset is in 64B multiples with the tid.
	 */
	frame_len =
	    min(tidsendc->remaining_bytes_in_tid, tidsendc->remaining_tidbytes);
	if (frame_len > tidsendc->frag_size) {
		frame_len =
		    tidsendc->frag_size - (tidsendc->offset_in_tid & 63);
	}
	/*
	 * Frame length is the amount of payload to be included in a particular
	 * frag of the scb, so we check if frame len is less than or equal
	 * to 8DW. If length is less then then or equal to 8DW for the first
	 * frag then we avoid header suppression
	 */
	if (frame_len <= 32)
		is_payload_per_frag_leq_8dw = 1;

	/*
	 * Using large offset mode based on offset length.
	 */
	if (tidsendc->offset_in_tid < 131072) {	/* 2^15 * 4 */
		psmi_assert((tidsendc->offset_in_tid % 4) == 0);
		offset = tidsendc->offset_in_tid / 4;
		omode = 0;
	} else {
		psmi_assert((tidsendc->offset_in_tid % 64) == 0);
		offset = tidsendc->offset_in_tid / 64;
		omode = 1;
	}
	startidx = tidsendc->tid_idx;
	tid = IPS_TIDINFO_GET_TID(tsess_list[startidx]);
	scb->ips_lrh.khdr.kdeth0 = __cpu_to_le32((offset & HFI_KHDR_OFFSET_MASK)
	    | (omode << HFI_KHDR_OM_SHIFT) | (tid << HFI_KHDR_TID_SHIFT));

	scb->tidctrl = IPS_TIDINFO_GET_TIDCTRL(tsess_list[startidx]);
	scb->tsess = (uint32_t *) &tsess_list[startidx];

	/*
	 * Payload and buffer address for current packet. payload_size
	 * must be the first packet size because it is used to initialize
	 * the packet header.
	 */
	scb->payload_size = frame_len;
	ips_scb_buffer(scb) = (void *)bufptr;
	scb->frag_size = tidsendc->frag_size;

	/*
	 * Other packet fields.
	 */
	PSM2_LOG_EPM(OPCODE_EXPTID,PSM2_LOG_TX, protoexp->proto->ep->epid,
		    flow->ipsaddr->epaddr.epid,
		    "psm3_mpool_get_obj_index(tidsendc->mqreq): %d, tidsendc->rdescid._desc_idx: %d, tidsendc->sdescid._desc_idx: %d",
		    psm3_mpool_get_obj_index(tidsendc->mqreq),tidsendc->rdescid._desc_idx,tidsendc->sdescid._desc_idx);
	ips_scb_opcode(scb) = OPCODE_EXPTID;
	scb->ips_lrh.exp_sdescid = tidsendc->sdescid;
	scb->ips_lrh.exp_rdescid_genc = (uint16_t)tidsendc->rdescid._desc_genc;
	scb->ips_lrh.exp_offset = tidsendc->tidbytes;

	scb->tidsendc = tidsendc;
	SLIST_NEXT(scb, next) = NULL;

	/*
	 * Loop over the tid session list, count the frag number and payload size.
	 */
	nfrag = 1;
	chunk_size = frame_len;
	while (1) {
		/* Record last tididx used */
		endidx = tidsendc->tid_idx;
		/* Check if all tidbytes are done */
		tidsendc->remaining_tidbytes -= frame_len;
		if (!tidsendc->remaining_tidbytes) {
			/* We do another frame length check for the last frag */
			if (frame_len <= 32)
				is_payload_per_frag_leq_8dw = 1;
			break;
		}

		/* Update in current tid */
		tidsendc->remaining_bytes_in_tid -= frame_len;
		tidsendc->offset_in_tid += frame_len;
		psmi_assert((tidsendc->offset_in_tid >= 128*1024) ?
			    ((tidsendc->offset_in_tid % 64) == 0) :
			    ((tidsendc->offset_in_tid %  4) == 0));

		/* Done with this tid, move on to the next tid */
		if (!tidsendc->remaining_bytes_in_tid) {
			tidsendc->tid_idx++;
			psmi_assert_always(tidsendc->tid_idx <
				    tidsendc->tid_list.tsess_tidcount);
			tidsendc->remaining_bytes_in_tid =
			    IPS_TIDINFO_GET_LENGTH(tsess_list
						   [tidsendc->tid_idx]) << 12;
			tidsendc->offset_in_tid = 0;
		}

		/* For PIO, only single packet per scb allowed */
		if (flow->transfer == PSM_TRANSFER_PIO) {
			break;
		}

		frame_len =
		    min(tidsendc->remaining_bytes_in_tid,
			tidsendc->remaining_tidbytes);
		if (frame_len > tidsendc->frag_size)
			frame_len = tidsendc->frag_size;
		nfrag++;
		chunk_size += frame_len;
	}

	scb->nfrag = nfrag;
	scb->chunk_size = chunk_size;
	if (nfrag > 1) {
		scb->nfrag_remaining = scb->nfrag;
		scb->chunk_size_remaining = scb->chunk_size;
	}
	scb->tsess_length = (endidx - startidx + 1) * sizeof(uint32_t);

	/* Keep track of latest buffer location so we restart at the
	 * right location, if we don't complete the transfer */
	tidsendc->buffer = bufptr + chunk_size;
	tidsendc->tidbytes += chunk_size;

	if (flow->transfer == PSM_TRANSFER_DMA &&
	    psmi_hal_has_cap(PSM_HAL_CAP_DMA_HSUPP_FOR_32B_MSGS)) {
		is_payload_per_frag_leq_8dw = 0;
	}

	/* If last packet, we want a completion notification */
	if (!tidsendc->remaining_tidbytes) {
		/* last packet/chunk, attach unaligned data */
		uint8_t *dst, *src;

		if (tidsendc->tid_list.tsess_unaligned_start) {
			dst = (uint8_t *)scb->ips_lrh.exp_ustart;
			src = (uint8_t *)tidsendc->userbuf;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (IS_TRANSFER_BUF_GPU_MEM(scb) && !tidsendc->mqreq->gpu_hostbuf_used) {
				PSM3_GPU_MEMCPY_DTOH(dst, src,
						tidsendc->tid_list.tsess_unaligned_start);
			} else
#endif
				ips_protoexp_unaligned_copy(dst, src,
						tidsendc->tid_list.tsess_unaligned_start);
		}

		if (tidsendc->tid_list.tsess_unaligned_end) {
			dst = (uint8_t *)&scb->ips_lrh.exp_uend;
			src = (uint8_t *)tidsendc->userbuf +
				tidsendc->length -
				tidsendc->tid_list.tsess_unaligned_end;
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (IS_TRANSFER_BUF_GPU_MEM(scb) && !tidsendc->mqreq->gpu_hostbuf_used) {
				PSM3_GPU_MEMCPY_DTOH(dst, src,
						tidsendc->tid_list.tsess_unaligned_end);
			} else
#endif
				ips_protoexp_unaligned_copy(dst, src,
						tidsendc->tid_list.tsess_unaligned_end);
		}
		/*
		 * If the number of fragments is greater then one and
		 * "no header suppression" flag is unset then we go
		 * ahead and suppress the header */
		if ((scb->nfrag > 1) && (!is_payload_per_frag_leq_8dw))
			scb->scb_flags |= IPS_SEND_FLAG_HDRSUPP;
		else
			scb->scb_flags |= IPS_SEND_FLAG_ACKREQ;

		tidsendc->is_complete = 1;	// all scb's queued for send
	} else {
		/* Do not suppress header every hdr_pkt_interval */
		if ((++tidsendc->frame_send %
				protoexp->hdr_pkt_interval) == 0)
			/* Request an ACK */
			scb->scb_flags |= IPS_SEND_FLAG_ACKREQ;
		else {
			if (!is_payload_per_frag_leq_8dw) {
				/* Request hdr supp */
				scb->scb_flags |= IPS_SEND_FLAG_HDRSUPP;
			}
		}
		/* assert only single packet per scb */
		psmi_assert(scb->nfrag == 1);
	}

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (tidsendc->mqreq->is_buf_gpu_mem &&		/* request's buffer comes from GPU realm */
	   !tidsendc->mqreq->gpu_hostbuf_used) {	/* and it was NOT moved to HOST memory */
		scb->mq_req = tidsendc->mqreq;		/* so let's mark it per scb, not to check its locality again */
		ips_scb_flags(scb) |= IPS_SEND_FLAG_PAYLOAD_BUF_GPU;
	}
#endif

	return scb;
}
#endif // PSM_OPA

#if defined(PSM_VERBS)
/*
 * Returns:
 *
 * PSM2_OK: scb was allocated for at least one frame, the packet may be queued
 *         or actually sent.
 *
 * PSM2_OK_NO_PROGRESS: Reached a limit on the maximum number of sends we allow
 *		       to be enqueued before polling receive queue.
 *
 * PSM2_EP_NO_RESOURCES: No scbs, available, a callback will be issued when more
 *                      scbs become available.
 *
 * PSM2_TIMEOUT: PIO-busy or DMA-busy, stop trying to send for now.
 *
 */

// issue RDMA Write in response to a CTS
psm2_error_t ips_tid_issue_rdma_write(struct ips_tid_send_desc *tidsendc)
{
	struct ips_protoexp *protoexp = tidsendc->protoexp;
	struct ips_proto *proto = protoexp->proto;
	psm2_error_t err = PSM2_OK;

	// for STL100 native we would loop on ips_scb_prepare_tid_sendctrl and
	// ips_proto_flow_enqueue to prepare EXPTID scbs for the TIDFLOW protocol
	// and queue and issue them.  Once they were all posted the is_complete
	// flag would be set.  For larger messages, it might take multiple
	// attempts to get resources to queue everything in which case callbacks
	// and timers ensure progress
	// For verbs we are delegating the RC Write "flow" to the NIC's RC QP
	// it will manage segmentation, sequence numbers and acks for the flow
	// so our job is done here after one call.
	// we use immediate data with the rdescid to trigger a CQE on receiver
	// so it knows when RDMA is done
	// if too many RDMA in flight retry later when next RDMA completes
	if (protoexp->proto->ep->verbs_ep.send_rdma_outstanding
				 >= protoexp->proto->ep->hfi_num_send_rdma) {
		err = PSM2_EP_NO_RESOURCES; // try again on next RDMA completion
	} else if (tidsendc->mr) {
		// registered or referenced in previous failed ips_tid_send_exp attempt
		// no need to register again
		err = PSM2_OK;
	} else if (
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		! tidsendc->mqreq->gpu_hostbuf_used &&
#endif
			// separate MR cache's per EP, so this confirms we have the same EP
		tidsendc->mqreq->mr && tidsendc->mqreq->mr->cache == proto->mr_cache) {
		// we can use the same MR as the whole mqreq
		_HFI_MMDBG("CTS send chunk reference send: %p %u bytes via %p %"PRIu64"\n",
			tidsendc->buffer, tidsendc->length, tidsendc->mqreq->mr->addr, tidsendc->mqreq->mr->length);
		tidsendc->mr = psm3_verbs_ref_mr(tidsendc->mqreq->mr);
	} else {
		// we need an MR for this chunk
		_HFI_MMDBG("CTS send chunk register send: %p %u bytes\n", tidsendc->buffer , tidsendc->length);
		tidsendc->mr = psm3_verbs_reg_mr(proto->mr_cache, 1,
                         tidsendc->buffer, tidsendc->length, IBV_ACCESS_RDMA
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
						| ((tidsendc->mqreq->is_buf_gpu_mem
								 && !tidsendc->mqreq->gpu_hostbuf_used)
							?IBV_ACCESS_IS_GPU_ADDR:0)
#endif
						);
		if (! tidsendc->mr)
			err = PSM2_TIMEOUT;	/* force a resend reschedule */
	}

	// if post_send fails below, we'll try again later
	// completion handler decides how to handle any WQE/CQE errors
	_HFI_MMDBG("tidsendc prior to post userbuf %p buffer %p length %u err %d outstanding %u\n",
			tidsendc->userbuf,  tidsendc->buffer, tidsendc->length,
			err, protoexp->proto->ep->verbs_ep.send_rdma_outstanding);
#ifdef RNDV_MOD
	if (err == PSM2_OK) {
		psmi_assert(IPS_PROTOEXP_FLAG_ENABLED & protoexp->proto->ep->rdmamode);

		if (IPS_PROTOEXP_FLAG_KERNEL_QP(protoexp->proto->ep->rdmamode))
			err = psm3_verbs_post_rv_rdma_write_immed(
				protoexp->proto->ep,
				tidsendc->ipsaddr->verbs.rv_conn,
				tidsendc->buffer, tidsendc->mr,
				tidsendc->tid_list.tsess_raddr, tidsendc->tid_list.tsess_rkey,
				tidsendc->tid_list.tsess_length,
				RDMA_PACK_IMMED(tidsendc->rdescid._desc_genc,
							 tidsendc->rdescid._desc_idx,
							 tidsendc->ipsaddr->verbs.remote_rv_index),
				(uintptr_t)tidsendc,
				&tidsendc->rv_sconn_index, &tidsendc->rv_conn_count);
#if defined(USE_RC) /* AND */
		else if (IPS_PROTOEXP_FLAG_USER_RC_QP(protoexp->proto->ep->rdmamode))
			err = psm3_verbs_post_rdma_write_immed(
				protoexp->proto->ep,
				tidsendc->ipsaddr->verbs.rc_qp,
				tidsendc->buffer, tidsendc->mr,
				tidsendc->tid_list.tsess_raddr, tidsendc->tid_list.tsess_rkey,
				tidsendc->tid_list.tsess_length,
				RDMA_PACK_IMMED(tidsendc->rdescid._desc_genc,
							 tidsendc->rdescid._desc_idx, 0),
				(uintptr_t)tidsendc);
#endif // defined(USE_RC)
	}
	if (err == PSM2_OK) {
		if (_HFI_PDBG_ON) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (tidsendc->mqreq->is_buf_gpu_mem && !tidsendc->mqreq->gpu_hostbuf_used)
				_HFI_PDBG_DUMP_GPU_ALWAYS(tidsendc->buffer, tidsendc->tid_list.tsess_length);
			else
#endif
				_HFI_PDBG_DUMP_ALWAYS(tidsendc->buffer, tidsendc->tid_list.tsess_length);
		}
		tidsendc->is_complete = 1;	// send queued
	} else
		_HFI_MMDBG("after posted IBV Write: err %d\n", err);

#else // RNDV_MOD
#if defined(USE_RC)
	if (err == PSM2_OK) {
		psmi_assert(IPS_PROTOEXP_FLAG_ENABLED & protoexp->proto->ep->rdmamode);
		if (IPS_PROTOEXP_FLAG_USER_RC_QP(protoexp->proto->ep->rdmamode))
			err = psm3_verbs_post_rdma_write_immed(
				protoexp->proto->ep,
				tidsendc->ipsaddr->verbs.rc_qp,
				tidsendc->buffer, tidsendc->mr,
				tidsendc->tid_list.tsess_raddr, tidsendc->tid_list.tsess_rkey,
				tidsendc->tid_list.tsess_length,
				RDMA_PACK_IMMED(tidsendc->rdescid._desc_genc,
							 tidsendc->rdescid._desc_idx, 0),
				(uintptr_t)tidsendc);
	}
	if (err == PSM2_OK) {
		if (_HFI_PDBG_ON) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (tidsendc->mqreq->is_buf_gpu_mem && !tidsendc->mqreq->gpu_hostbuf_used)
				_HFI_PDBG_DUMP_GPU_ALWAYS(tidsendc->buffer, tidsendc->tid_list.tsess_length);
			else
#endif
				_HFI_PDBG_DUMP_ALWAYS(tidsendc->buffer, tidsendc->tid_list.tsess_length);
		}
		tidsendc->is_complete = 1;	// send queued
	} else
		_HFI_MMDBG("after posted IBV Write 2: err %d\n", err);
#endif /* defined(USE_RC) */
#endif // RNDV_MOD
	return err;
}
#endif // defined(PSM_VERBS)

/*
 * Returns:
 *
 * PSM2_OK: scb was allocated for at least one frame, the packet may be queued
 *         or actually sent.
 *
 * PSM2_OK_NO_PROGRESS: Reached a limit on the maximum number of sends we allow
 *		       to be enqueued before polling receive queue.
 *
 * PSM2_EP_NO_RESOURCES: No scbs, available, a callback will be issued when more
 *                      scbs become available.
 *
 * PSM2_TIMEOUT: PIO-busy or DMA-busy, stop trying to send for now.
 *
 */

// we got a CTS and processed it.  Now we can start sending EXPTID packets.
// For UD we will use RDMA instead of EXPTID
static
psm2_error_t ips_tid_send_exp(struct ips_tid_send_desc *tidsendc)
{
#ifdef PSM_OPA
	ips_scb_t *scb = NULL;
#endif
	psm2_error_t err = PSM2_OK;
#ifdef PSM_OPA
	psm2_error_t err_f;
	struct ips_protoexp *protoexp = tidsendc->protoexp;
	struct ips_proto *proto = protoexp->proto;
	struct ips_flow *flow = &tidsendc->tidflow;
#elif defined(PSM_CUDA) || defined(PSM_ONEAPI)
	struct ips_protoexp *protoexp = tidsendc->protoexp;
#endif

	_HFI_MMDBG("ips_tid_send_exp\n");
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	struct ips_gpu_hostbuf *chb, *chb_next;
	uint32_t offset_in_chb, i;
	for (i = 0; i < tidsendc->cuda_num_buf; i++) {
		chb = tidsendc->cuda_hostbuf[i];
		if (chb) {
			if (!PSM3_GPU_MEMCPY_DONE(chb)) {
				err = PSM2_OK_NO_PROGRESS;
				PSM2_LOG_MSG("leaving");
				return err;
			}
		}
	}

	if (tidsendc->cuda_num_buf == 2) {
		chb = tidsendc->cuda_hostbuf[0];
		chb_next = tidsendc->cuda_hostbuf[1];
		offset_in_chb = tidsendc->tid_list.tsess_srcoff - chb->offset;
		/* Copying data from multiple cuda
		 * host buffers into a bounce buffer.
		 */
		memcpy(tidsendc->buffer, (void *)((uintptr_t)chb->host_buf +
			offset_in_chb), chb->size-offset_in_chb);
		memcpy((void *)((uintptr_t)tidsendc->buffer + chb->size -
			offset_in_chb), chb_next->host_buf,
			tidsendc->tid_list.tsess_srcoff +
			tidsendc->tid_list.tsess_length - chb_next->offset);

		chb->bytes_read += chb->size - offset_in_chb;
		chb_next->bytes_read += tidsendc->tid_list.tsess_srcoff +
				  tidsendc->tid_list.tsess_length -
				  chb_next->offset;
		if(chb->bytes_read == chb->size) {
			STAILQ_REMOVE(&tidsendc->mqreq->sendreq_prefetch, chb,
				       ips_gpu_hostbuf, req_next);
			if (chb->is_tempbuf)
				psmi_deallocate_chb(chb);
			else {
				chb->req = NULL;
				chb->offset = 0;
				chb->bytes_read = 0;
				PSM3_GPU_HOSTBUF_RESET(chb);
				psm3_mpool_put(chb);
			}
			psmi_cuda_run_prefetcher(protoexp, tidsendc);
		 }
		if(chb_next->bytes_read == chb_next->size) {
			STAILQ_REMOVE(&tidsendc->mqreq->sendreq_prefetch, chb_next,
				       ips_gpu_hostbuf, req_next);
			if (chb_next->is_tempbuf)
				psmi_deallocate_chb(chb_next);
			else{
				chb_next->req = NULL;
				chb_next->offset = 0;
				chb_next->bytes_read = 0;
				PSM3_GPU_HOSTBUF_RESET(chb_next);
				psm3_mpool_put(chb_next);
			}
			psmi_cuda_run_prefetcher(protoexp, tidsendc);
		}
		/* Clean Up tidsendc ref's to split cuda hostbufs when no longer needed */
		tidsendc->cuda_num_buf = 0;
		tidsendc->cuda_hostbuf[0] = NULL;
		tidsendc->cuda_hostbuf[1] = NULL;
	}
#endif
#ifdef PSM_OPA
	/*
	 * We aggressively try to grab as many scbs as possible, enqueue them to a
	 * flow and flush them when either we're out of scbs or we've completely
	 * filled the send request.
	 */
	while (!tidsendc->is_complete) {
		if_pf(tidsendc->tid_list.tsess_tidcount &&
		      (tidsendc->tid_idx >= tidsendc->tid_list.tsess_tidcount ||
		       tidsendc->tid_idx < 0))
			ips_expsend_tiderr(tidsendc);

		if ((scb =
		     ips_scb_prepare_tid_sendctrl(flow, tidsendc)) == NULL) {
			proto->stats.scb_exp_unavail_cnt++;
			err = PSM2_EP_NO_RESOURCES;
			break;
		} else {
			// queue up the sends, likely to be SDMA
			psm3_ips_proto_flow_enqueue(flow, scb);
		}
	}

	if (!SLIST_EMPTY(&flow->scb_pend)) {	/* Something to flush */
		int num_sent;

		// this will kick off the sends, likely to be SDMA
		err_f = flow->flush(flow, &num_sent);

		// since we are using the tidflow, we ensure a future
		// timer callback will flush the remaining scbs or
		// process the rcvhdrq
		if (err != PSM2_EP_NO_RESOURCES) {
			/* PSM2_EP_NO_RESOURCES is reserved for out-of-scbs */
			if (err_f == PSM2_EP_NO_RESOURCES)
				err = PSM2_TIMEOUT;	/* force a resend reschedule */
			else if (err_f == PSM2_OK && num_sent > 0 &&
				 !psm3_gen1_ips_ptl_recvq_isempty(protoexp->ptl))
				err = PSM2_OK_NO_PROGRESS;	/* force a rcvhdrq service */
		}
	}
#elif defined(PSM_VERBS)
	err = ips_tid_issue_rdma_write(tidsendc);
#endif

	PSM2_LOG_MSG("leaving");
	return err;
}

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
// Used when err chk rdma resp indicates we must resend the rdma
static
void ips_tid_reissue_rdma_write(struct ips_tid_send_desc *tidsendc)
{
	struct ips_protoexp *protoexp = tidsendc->protoexp;

	_HFI_MMDBG("ips_tid_reissue_rdma_write\n");

	PSM2_LOG_MSG("entering");
	protoexp->proto->epaddr_stats.rdma_rexmit++;
	tidsendc->is_complete = 0;	// sends not yet queued

	ips_tid_issue_rdma_write(tidsendc);

	/* Add as a pending op and ring up the timer */
	if (tidsendc->is_complete == 0) {
		STAILQ_INSERT_TAIL(&protoexp->pend_sendq, tidsendc, next);
		psmi_timer_request(protoexp->timerq, &protoexp->timer_send,
			   PSMI_TIMER_PRIO_1);
	}

	PSM2_LOG_MSG("leaving");
}
#endif // RNDV_MOD
#endif // defined(PSM_VERBS)

static
psm2_error_t
ips_tid_pendsend_timer_callback(struct psmi_timer *timer, uint64_t current)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *)timer->context;
	struct ips_tid_send_pend *phead = &protoexp->pend_sendq;
	struct ips_tid_send_desc *tidsendc;
	psm2_error_t err = PSM2_OK;
	_HFI_MMDBG("ips_tid_pendsend_timer_callback\n");

	while (!STAILQ_EMPTY(phead)) {
		tidsendc = STAILQ_FIRST(phead);

		// we have some scb's and can use them to queue some more EXPTID packets
#if defined(PSM_VERBS)
#ifdef RNDV_MOD
		if (tidsendc->rv_need_err_chk_rdma)
			err = ips_protoexp_send_err_chk_rdma(tidsendc);
		else
#endif
#endif
			err = ips_tid_send_exp(tidsendc);

		if (tidsendc->is_complete)
			STAILQ_REMOVE_HEAD(phead, next);

		if (err == PSM2_OK) {
			/* Was able to complete the send, keep going */
		} else if (err == PSM2_EP_NO_RESOURCES) {
			/* No more sendbufs available, sendbuf callback will requeue this
			 * timer */
			break;
		} else if (err == PSM2_TIMEOUT
				  || err == PSM2_EPID_RV_CONNECT_RECOVERING
				  || err == PSM2_EPID_RV_CONNECT_ERROR) {
			/* Always a case of try later:
			 * On PIO flow, means no send pio bufs available
			 * On DMA flow, means kernel can't queue request or would have to block
			 * On RV or user RDMA QP is full or connection recovery/issues
			 */
			psmi_timer_request(protoexp->proto->timerq,
					   &protoexp->timer_send,
					   get_cycles() +
					   protoexp->proto->timeout_send);
			break;
		} else {
			/* Forced to reschedule later so we can check receive queue */
			psmi_assert(err == PSM2_OK_NO_PROGRESS);
			psmi_timer_request(protoexp->proto->timerq,
					   &protoexp->timer_send,
					   PSMI_TIMER_PRIO_1);
			break;
		}
	}

	return PSM2_OK;
}

/* Right now, in the kernel we are allowing for virtually non-contiguous pages,
   in a single call, and we are therefore locking one page at a time, but since
   the intended use of this routine is for a single group of
   virtually contiguous pages, that should change to improve
   performance.  That means possibly changing the calling MPI code.
   Doing so gets rid of some of the loop stuff here, and in the driver,
   and allows for a single call to the core VM code in the kernel,
   rather than one per page, definitely improving performance. */

#ifdef PSM_OPA
static
psm2_error_t
ips_tid_recv_alloc_frag(struct ips_protoexp *protoexp,
			struct ips_tid_recv_desc *tidrecvc,
			uint32_t nbytes_this)
{
	ips_tid_session_list *tid_list = &tidrecvc->tid_list;
	uintptr_t bufptr = (uintptr_t) tidrecvc->buffer;
	uint32_t size = nbytes_this;
	psm2_error_t err = PSM2_OK;
	uintptr_t pageaddr;
	uint32_t tidoff, pageoff, pagelen, reglen, num_tids;

	psmi_assert(size >= 4);

	/*
	 * The following calculation does not work when size < 4
	 * and bufptr is byte aligned, it can get negative value.
	 */
	tid_list->tsess_unaligned_start = (bufptr & 3) ? (4 - (bufptr & 3)) : 0;
	size -= tid_list->tsess_unaligned_start;
	bufptr += tid_list->tsess_unaligned_start;

	tid_list->tsess_unaligned_end = size & 3;
	size -= tid_list->tsess_unaligned_end;

	psmi_assert(size > 0);

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	/* Driver pins GPU pages when using GPU Direct RDMA for TID recieves,
	 * to accomadate this change the calculations of pageaddr, pagelen
	 * and pageoff have been modified to take GPU page size into
	 * consideration.
	 */
	if (tidrecvc->is_ptr_gpu_backed) {
		uint64_t page_mask = ~(PSMI_GPU_PAGESIZE -1);
		uint32_t page_offset_mask = (PSMI_GPU_PAGESIZE -1);
		pageaddr = bufptr & page_mask;
		pagelen = (uint32_t) (PSMI_GPU_PAGESIZE +
			  ((bufptr + size - 1) & page_mask) -
			  (bufptr & page_mask));
		tidoff = pageoff = (uint32_t) (bufptr & page_offset_mask);
	} else
#endif
	{
		pageaddr = bufptr & protoexp->tid_page_mask;
		pagelen = (uint32_t) (PSMI_PAGESIZE +
			  ((bufptr + size - 1) & protoexp->tid_page_mask) -
			  (bufptr & protoexp->tid_page_mask));
		tidoff = pageoff = (uint32_t) (bufptr & protoexp->tid_page_offset_mask);
	}

	reglen = pagelen;
	if (protoexp->tidc.tid_array) {
		if ((err = ips_tidcache_acquire(&protoexp->tidc,
			    (void *)pageaddr, &reglen,
			    (uint32_t *) tid_list->tsess_list, &num_tids,
			    &tidoff
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			    , tidrecvc->is_ptr_gpu_backed
#endif
			    )))
			goto fail;
	} else {
		if ((err = ips_tid_acquire(&protoexp->tidc,
			    (void *)pageaddr, &reglen,
			    (uint32_t *) tid_list->tsess_list, &num_tids
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			    , tidrecvc->is_ptr_gpu_backed
#endif
			)))
			goto fail;
	}

	/*
	 * PSM2 currently provides storage space enough to hold upto
	 * 1024 tids. (PSM_TIDLIST_BUFSIZE). So, make sure we
	 * don't get more than what we can hold from the tidcache here.
	 *
	 * The reason for 1024 tids comes from the PSM_TID_WINSIZE value
	 * (currently 4MB. So, if in future, there is a change to this macro,
	 * then you would need a change to PSM_TIDLIST_BUFSIZE as well).
	 *
	 * Assuming a 4KB page size, to be able to receive
	 * a message of 4MB size, we'd need an maximum of 4MB/4KB = 1024 tids.
	 */
	psmi_assert(num_tids > 0);
	psmi_assert(num_tids <= (PSM_TID_WINSIZE/PSM_TIDLIST_BUFSIZE));
	if (reglen > pagelen) {
		err = psm3_handle_error(protoexp->tidc.context->ep,
			    PSM2_EP_DEVICE_FAILURE,
			    "PSM tid registration: "
			    "register more pages than asked");
		goto fail;
	} else if (reglen < pagelen) {
		/*
		 * driver registered less pages, update PSM records.
		 */
		tid_list->tsess_unaligned_end = 0;
		tidrecvc->recv_tidbytes = reglen - pageoff;
		tidrecvc->recv_msglen = tid_list->tsess_unaligned_start +
		    tidrecvc->recv_tidbytes;
	} else {
		tidrecvc->recv_tidbytes = size;
		tidrecvc->recv_msglen = nbytes_this;
	}

	tid_list->tsess_tidcount = num_tids;
	tid_list->tsess_tidoffset = tidoff;

	ips_dump_tids(tid_list, "Registered %d tids: ", num_tids);

fail:
	return err;
}
#endif // PSM_OPA

static
psm2_error_t
ips_tid_recv_alloc(struct ips_protoexp *protoexp,
		   ips_epaddr_t *ipsaddr,
		   const struct ips_tid_get_request *getreq,
		   uint32_t nbytes_this, struct ips_tid_recv_desc **ptidrecvc)
{
	psm2_error_t err;
	ips_scb_t *grantscb;
#ifdef PSM_OPA
	ips_scb_t *completescb;
#endif
#ifdef PSM_VERBS
	psm2_mq_req_t req = getreq->tidgr_req;
#elif defined(PSM_CUDA) || defined(PSM_ONEAPI)
	psm2_mq_req_t req = getreq->tidgr_req;
#endif
#if defined(PSM_VERBS)
	struct ips_proto *proto = protoexp->proto;
#endif

	struct ips_tid_recv_desc *tidrecvc;

	PSM2_LOG_MSG("entering");
	/* Allocate all necessary resources. */
	_HFI_MMDBG("tid_recv_alloc\n");

	// allocate what we need to handle TID or RDMA on receive side
	// we do this before we issue CTS

	/* 1. allocate a tid grant (CTS) scb. */
	grantscb = psm3_ips_scbctrl_alloc(&protoexp->tid_scbc_rv, 1, 0, 0);
	if (grantscb == NULL) {
		_HFI_MMDBG("Wait: NO GRANT SCB\n");
		/* ips_tid_scbavail_callback() will reschedule */
		PSM2_LOG_MSG("leaving");
		return PSM2_EP_NO_RESOURCES;
	}

#ifdef PSM_OPA
	/* 2. allocate a tid complete (final ACK) scb. */
	completescb = psm3_ips_scbctrl_alloc(&protoexp->tid_scbc_rv, 1, 0, 0);
	if (completescb == NULL) {
		psm3_ips_scbctrl_free(grantscb);
		/* ips_tid_scbavail_callback() will reschedule */
		PSM2_LOG_MSG("leaving");
		return PSM2_EP_NO_RESOURCES;
	}
#endif

	/* 3. allocate a tid flow entry. */
	err = psm3_ips_tf_allocate(&protoexp->tfc, &tidrecvc);
	if (err != PSM2_OK) {
		_HFI_MMDBG("Wait: NO tid flow\n");
#ifdef PSM_OPA
		psm3_ips_scbctrl_free(completescb);
#endif
		psm3_ips_scbctrl_free(grantscb);
		/* Unable to get a tidflow for expected protocol. */
		psmi_timer_request(protoexp->timerq,
			&protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
		PSM2_LOG_MSG("leaving");
		return err;
	}
#if defined(PSM_VERBS)
	tidrecvc->mr = NULL;	// be safe,but should be NULL since clear on release
#endif

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
       if (req->is_buf_gpu_mem)
               tidrecvc->is_ptr_gpu_backed = !getreq->gpu_hostbuf_used;
       else
               tidrecvc->is_ptr_gpu_backed = req->is_buf_gpu_mem;

	/* 4. allocate a cuda bounce buffer, if required */
	struct ips_gpu_hostbuf *chb = NULL;
	if (getreq->gpu_hostbuf_used) {
		unsigned bufsz;
		if (nbytes_this <= CUDA_SMALLHOSTBUF_SZ) {
			chb = (struct ips_gpu_hostbuf *)
				psm3_mpool_get(
					protoexp->cuda_hostbuf_pool_small_recv);
			bufsz = protoexp->cuda_hostbuf_small_recv_cfg.bufsz;
		}
		if (chb == NULL) {
			chb = (struct ips_gpu_hostbuf *)
				psm3_mpool_get(
					protoexp->cuda_hostbuf_pool_recv);
			bufsz = protoexp->cuda_hostbuf_recv_cfg.bufsz;
		}
		if (chb == NULL) {
			/* Unable to get a cudahostbuf for TID.
			 * Release the resources we're holding and reschedule.*/
			psm3_ips_tf_deallocate(&protoexp->tfc,
					  tidrecvc->rdescid._desc_idx, 0);
#ifdef PSM_OPA
			psm3_ips_scbctrl_free(completescb);
#endif
			psm3_ips_scbctrl_free(grantscb);
			psmi_timer_request(protoexp->timerq,
					   &protoexp->timer_getreqs,
					   PSMI_TIMER_PRIO_1);
			PSM2_LOG_MSG("leaving");
			return PSM2_EP_NO_RESOURCES;
		}

		if (chb->host_buf == NULL) {
			PSM3_GPU_HOST_ALLOC(&chb->host_buf, bufsz);
		}
		tidrecvc->cuda_hostbuf = chb;
		tidrecvc->buffer = chb->host_buf;
		chb->size = 0;
		chb->gpu_buf = (uint8_t*)getreq->tidgr_lbuf +
					getreq->tidgr_offset;
	} else {
		chb = NULL;
		tidrecvc->buffer = (void *)((uintptr_t) getreq->tidgr_lbuf +
					    getreq->tidgr_offset);
		tidrecvc->cuda_hostbuf = NULL;
	}
#else // PSM_CUDA || PSM_ONEAPI
	tidrecvc->buffer =
	    (void *)((uintptr_t) getreq->tidgr_lbuf + getreq->tidgr_offset);
#endif // PSM_CUDA || PSM_ONEAPI

#if defined(PSM_SOCKETS) && PSMI_HAL_INST_CNT == 1
	psmi_assert_always(0);	// should not get here
#elif defined(PSM_VERBS)
	// separate MR cache's per EP, so this confirms we have the same EP
	if (
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		! getreq->gpu_hostbuf_used &&
#endif
		req->mr && req->mr->cache == proto->mr_cache) {
		_HFI_MMDBG("CTS chunk reference recv: %p %u bytes via %p %"PRIu64"\n", tidrecvc->buffer, nbytes_this, req->mr->addr, req->mr->length);
		tidrecvc->mr = psm3_verbs_ref_mr(req->mr);
	} else {
		_HFI_MMDBG("CTS chunk register recv: %p %u bytes\n", tidrecvc->buffer, nbytes_this);
		tidrecvc->mr = psm3_verbs_reg_mr(proto->mr_cache, 1,
                        tidrecvc->buffer, nbytes_this, IBV_ACCESS_RDMA|IBV_ACCESS_REMOTE_WRITE
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
               			| (tidrecvc->is_ptr_gpu_backed?IBV_ACCESS_IS_GPU_ADDR:0)
#endif
						);
		if (! tidrecvc->mr) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
			if (chb)
				psm3_mpool_put(chb);
#endif
			psm3_ips_tf_deallocate(&protoexp->tfc, tidrecvc->rdescid._desc_idx, 0);
			//psm3_ips_scbctrl_free(completescb);
			psm3_ips_scbctrl_free(grantscb);
			/* Unable to register MR */
			psmi_timer_request(protoexp->timerq,
				&protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
			PSM2_LOG_MSG("leaving");
			return PSM2_TIMEOUT;	// make sure we try again
		}
		_HFI_MMDBG("CTS chunk registered: addr %p len %d rkey 0x%x\n",  tidrecvc->buffer , nbytes_this, tidrecvc->mr->rkey);
	}

	tidrecvc->recv_msglen = nbytes_this;
#elif defined(PSM_OPA)
	/* 5. allocate some tids from driver. */
	err = ips_tid_recv_alloc_frag(protoexp, tidrecvc, nbytes_this);
	if (err != PSM2_OK) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		if (chb)
			psm3_mpool_put(chb);
#endif
		psm3_ips_tf_deallocate(&protoexp->tfc, tidrecvc->rdescid._desc_idx, 0);
		psm3_ips_scbctrl_free(completescb);
		psm3_ips_scbctrl_free(grantscb);
		/* Unable to register tids */
		psmi_timer_request(protoexp->timerq,
			&protoexp->timer_getreqs, PSMI_TIMER_PRIO_1);
		PSM2_LOG_MSG("leaving");
		return err;
	}

	if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_TID_DEBUG) {
		int num_tids = tidrecvc->tid_list.tsess_tidcount;
		int tid, i;
		for (i = 0; i < num_tids; i++) {
			tid =
			    IPS_TIDINFO_GET_TID(tidrecvc->tid_list.
					tsess_list[i]) * 2 +
			    IPS_TIDINFO_GET_TIDCTRL(tidrecvc->tid_list.
					tsess_list[i]) - 1;
			psmi_assert(protoexp->tid_info[tid].state ==
				    TIDSTATE_FREE);
			psmi_assert(protoexp->tid_info[tid].tidrecvc == NULL);
			psmi_assert(protoexp->tid_info[tid].tid == 0xFFFFFFFF);
			protoexp->tid_info[tid].state = TIDSTATE_USED;
			protoexp->tid_info[tid].tidrecvc = tidrecvc;
			protoexp->tid_info[tid].tid =
			    tidrecvc->tid_list.tsess_list[i];
		}
	}
#endif

	/* Initialize recv descriptor */
	tidrecvc->ipsaddr = ipsaddr;
	tidrecvc->getreq = (struct ips_tid_get_request *)getreq;

#ifdef PSM_OPA
	/* Initialize tidflow, instead calling generic routine:
	   psm3_ips_flow_init(&tidrecvc->tidflow, protoexp->proto, ipsaddr,
		      protoexp->ctrl_xfer_type, PSM_PROTOCOL_TIDFLOW,
		      IPS_PATH_LOW_PRIORITY, EP_FLOW_TIDFLOW);
	 * only reset following necessary field. */
	tidrecvc->tidflow.ipsaddr = ipsaddr;
	tidrecvc->tidflow.flags = 0;
#endif

	tidrecvc->tidflow_nswap_gen = 0;
	tidrecvc->tidflow_genseq.psn_gen = tidrecvc->tidflow_active_gen;
	tidrecvc->tidflow_genseq.psn_seq = 0;	/* Always start sequence number at 0 (zero),
	 	 	 	 	 	   in order to prevent wraparound sequence numbers */
#ifdef PSM_OPA
	psmi_hal_tidflow_set_entry(
			      tidrecvc->rdescid._desc_idx,
			      tidrecvc->tidflow_genseq.psn_gen,
			      tidrecvc->tidflow_genseq.psn_seq,
			      tidrecvc->context->psm_hw_ctxt);
#endif

	tidrecvc->tid_list.tsess_srcoff = getreq->tidgr_offset;
	tidrecvc->tid_list.tsess_length = tidrecvc->recv_msglen;
#if defined(PSM_VERBS)
	// when using kernel PD/MR for kernel rendezvous QP, we need to xlat
	// our buffer to the kernel PD/MR iova space.
	// for user space PD/MR iova == addr and xlat is a noop
	tidrecvc->tid_list.tsess_rkey = tidrecvc->mr->rkey;
	tidrecvc->tid_list.tsess_raddr = tidrecvc->mr->iova + ((uintptr_t)tidrecvc->buffer -  (uintptr_t)tidrecvc->mr->addr);
#endif

#ifdef PSM_OPA
	tidrecvc->ctrl_msg_queued = 0;
#endif
	tidrecvc->state = TIDRECVC_STATE_BUSY;

	tidrecvc->stats.nSeqErr = 0;
	tidrecvc->stats.nGenErr = 0;
	tidrecvc->stats.nReXmit = 0;
	tidrecvc->stats.nErrChkReceived = 0;

#ifdef PSM_OPA
	/* This gets sent out as a control message, so we need to force 4-byte IB
	 * alignment */
	tidrecvc->tsess_tidlist_length = (uint16_t)
	    PSMI_ALIGNUP((sizeof(ips_tid_session_list) +
			  (tidrecvc->tid_list.tsess_tidcount *
			   sizeof(uint32_t))), 4);

	_HFI_EXP("alloc tidrecv=%d, paylen=%d, ntid=%d\n",
		 tidrecvc->rdescid._desc_idx,
		 tidrecvc->tsess_tidlist_length,
		 tidrecvc->tid_list.tsess_tidcount);
#else
	_HFI_EXP("alloc tidrecv=%d\n",
		 tidrecvc->rdescid._desc_idx);
#endif

	tidrecvc->grantscb = grantscb;
#ifdef PSM_OPA
	tidrecvc->completescb = completescb;
#endif

	*ptidrecvc = tidrecvc; /* return to caller */
	PSM2_LOG_MSG("leaving");
	return PSM2_OK;
}

// process receiver side TID queue
// If we have TID resources, we will acquire them, setup TID HW,
// prepare a CTS and send it
// If we run out of resources with more to do, we reschedule ourselves on timer
// called directly or on timer
static
psm2_error_t
ips_tid_pendtids_timer_callback(struct psmi_timer *timer, uint64_t current)
{
	struct ips_tid_get_pend *phead =
	    &((struct ips_protoexp *)timer->context)->pend_getreqsq;
	struct ips_protoexp *protoexp;
	struct ips_tid_get_request *getreq;
	struct ips_tid_recv_desc *tidrecvc;
	ips_epaddr_t *ipsaddr;
	uint32_t nbytes_this, count;
#if defined(PSM_VERBS)
#ifdef RNDV_MOD
	struct ips_tid_err_resp_pend *phead_resp =
	    &((struct ips_protoexp *)timer->context)->pend_err_resp;
#endif
#endif
	int ret;

	PSM2_LOG_MSG("entering");
	_HFI_MMDBG("ips_tid_pendtids_timer_callback\n");

#if defined(PSM_VERBS)
#ifdef RNDV_MOD
	while (!STAILQ_EMPTY(phead_resp)) {
		ipsaddr = STAILQ_FIRST(phead_resp);
		protoexp = ipsaddr->epaddr.proto->protoexp;
		psmi_assert(ipsaddr->verbs.rv_need_send_err_chk_rdma_resp);
		ips_protoexp_send_err_chk_rdma_resp(&ipsaddr->flows[protoexp->proto->msgflowid]);
		if (! ipsaddr->verbs.rv_need_send_err_chk_rdma_resp)
			STAILQ_REMOVE_HEAD(phead_resp, verbs.pend_err_resp_next);
		else
			break; // ips_tid_scbavail_callback will trigger us again
	}
#endif
#endif

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (
#ifdef PSM_OPA
	    !(((struct ips_protoexp *)timer->context)->proto->flags
		& IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV) ||
		((((struct ips_protoexp *)timer->context)->proto->flags &
		   IPS_PROTO_FLAG_GPUDIRECT_RDMA_RECV) &&
		   gpudirect_rdma_recv_limit < UINT_MAX)
#else
	    1	/* due to unaligned recv using hostbuf, must always do this */
#endif
	) {
		/* Before processing pending TID requests, first try to free up
		 * any CUDA host buffers that are now idle. */
		struct ips_tid_get_cudapend *cphead =
			&((struct ips_protoexp *)timer->context)->cudapend_getreqsq;
		psm2_error_t err;

		/* See if any CUDA memcpys are in progress. Grab the first getreq... */
		while (!STAILQ_EMPTY(cphead)) {
			getreq = STAILQ_FIRST(cphead);

			err = psmi_cuda_reclaim_hostbufs(getreq);
			if (err == PSM2_OK_NO_PROGRESS)
				goto cudapend_exit;

			/* This pending cuda getreq has no more CUDA ops queued up.
			 * Either it's completely done, or the CUDA copies have caught
			 * up with the TID data xfer, but the TID xfer itself is not
			 * finished.
			 */
			if (getreq->tidgr_cuda_bytesdone == getreq->tidgr_length) {
				/* TID xfer is done.
				 * We should only get here if:
				 * this was involved a cuda copy, and
				 * the TIX xfer is done.
				 */
				psmi_assert(getreq->gpu_hostbuf_used);
				psmi_assert(getreq->tidgr_length ==
					    getreq->tidgr_offset);

				/* Remove from the cudapend list, and reclaim */
				getreq->tidgr_protoexp = NULL;
				getreq->tidgr_epaddr = NULL;
				STAILQ_REMOVE_HEAD(cphead, tidgr_next);

				/* mark the req as done */
				if (getreq->tidgr_callback)
					getreq->tidgr_callback(getreq->tidgr_req);
				psm3_mpool_put(getreq);
			} else
				break; /* CUDA xfers in progress. Leave. */
		}
	}
cudapend_exit:
#endif

	while (!STAILQ_EMPTY(phead)) {
		getreq = STAILQ_FIRST(phead);
		ipsaddr = (ips_epaddr_t *) (getreq->tidgr_epaddr);
		count = ipsaddr->msgctl->ipsaddr_count;

ipsaddr_next:
		// always stripe for rendezvous, ignore multirail_thresh_load_balance
		// TBD - for multi-rail does this eliminate any advantages of
		// registering the MR in ips_proto_mq_rts_match_callback
		ipsaddr = ipsaddr->msgctl->ipsaddr_next;
		ipsaddr->msgctl->ipsaddr_next = ipsaddr->next;
		protoexp = ((psm2_epaddr_t) ipsaddr)->proto->protoexp;

		if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_CTS_SERIALIZED) {
			psmi_assert(protoexp->proto->msgflowid < EP_FLOW_LAST);
			struct ips_flow *flow = &ipsaddr->flows[protoexp->proto->msgflowid];
			if (flow->flags & IPS_FLOW_FLAG_SKIP_CTS) {
				break;                                    /* skip sending next CTS */
			}
		}

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		if (getreq->gpu_hostbuf_used) {
			/* If this is a large transfer, we may be able to
			 * start reclaiming before all of the data is sent. */
			psmi_cuda_reclaim_hostbufs(getreq);
		}
#endif
		/*
		 * Calculate the next window size, avoid the last
		 * window too small.
		 */
		nbytes_this = getreq->tidgr_length - getreq->tidgr_offset;
		if (nbytes_this >= 2 * getreq->tidgr_rndv_winsz)
			nbytes_this = getreq->tidgr_rndv_winsz;
		else if (nbytes_this > getreq->tidgr_rndv_winsz)
			nbytes_this /= 2;
		_HFI_MMDBG("ips_tid_pendtids_timer_callback: getreq len %u offset %u nbytes_this %u\n", getreq->tidgr_length, getreq->tidgr_offset, nbytes_this);

		/*
		 * If there is a next window and the next window
		 * length is greater than PAGESIZE, make sure the window
		 * starts on a receive buffer page boundary.
		 * This assumes that page aligned receives will perform
		 * better so it strives to have only the 1st CTS be non-page
		 * aligned.  It does however incur a cost in that some large
		 * IOs may require an extra CTS.  For example with a GPU
		 * and 2MB window_sz, a 4M IO may end up usng 3 CTS instead of 2
		 * However by default such CTS are pipelined so most of the
		 * costs are hidden.  In the case of large GPU sends, the
		 * sender will be using a host bounce buffer and
		 * async cuda copies to fill it, so the extra CTS is minimal
		 * impact to the sender.
		 */
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		psm2_mq_req_t req = getreq->tidgr_req;
		if (req->is_buf_gpu_mem){
			if (((getreq->tidgr_offset + nbytes_this) <
					getreq->tidgr_length) &&
					nbytes_this > PSMI_GPU_PAGESIZE) {
				uint32_t pageoff =
					(((uintptr_t)getreq->tidgr_lbuf) &
						(PSMI_GPU_PAGESIZE - 1)) +
					getreq->tidgr_offset + nbytes_this;
				nbytes_this -= pageoff & (PSMI_GPU_PAGESIZE - 1);
			}
		} else
#endif
		{
			if ((getreq->tidgr_offset + nbytes_this) <
					getreq->tidgr_length &&
					nbytes_this > PSMI_PAGESIZE) {
				uint32_t pageoff =
					(((uintptr_t)getreq->tidgr_lbuf) &
						(PSMI_PAGESIZE - 1)) +
					getreq->tidgr_offset + nbytes_this;
				nbytes_this -= pageoff & (PSMI_PAGESIZE - 1);
			}
		}
		_HFI_MMDBG("ips_tid_pendtids_timer_callback: page align nbytes_this %u\n", nbytes_this);

		psmi_assert(nbytes_this >= 4);
#ifdef PSM_OPA
		psmi_assert(nbytes_this <= PSM_TID_WINSIZE);
#endif

		// for STL native the tids and tidflows available pace incoming TIDs
		// for UD we still use tidflows available to pace incoming RDMA
#ifdef PSM_OPA
		if ((ret = ips_tid_num_available(&protoexp->tidc)) <= 0) {
			/* We're out of tids. If this process used all the resource,
			 * the free callback will reschedule the operation, otherwise,
			 * we reschedule it here */
			if (ret == 0)
			{
				psmi_timer_request(protoexp->timerq,
						   &protoexp->timer_getreqs,
						   PSMI_TIMER_PRIO_1);
			}
		} else
#endif
			if ((ret = ips_tf_available(&protoexp->tfc)) <= 0) {
			/* We're out of tidflow. If this process used all the resource,
			 * the free callback will reschedule the operation, otherwise,
			 * we reschedule it here */
			if (ret == 0)
			{
				psmi_timer_request(protoexp->timerq,
						   &protoexp->timer_getreqs,
						   PSMI_TIMER_PRIO_1);
			}
		} else if (ips_tid_recv_alloc(protoexp, ipsaddr,
			      getreq, nbytes_this, &tidrecvc) == PSM2_OK) {
			// send the actual CTS
			psm3_ips_protoexp_send_tid_grant(tidrecvc);
			_HFI_VDBG("GRANT sent tididx=%d srcoff=%d nbytes=%d/%d\n",
				  tidrecvc->rdescid._desc_idx,
				  getreq->tidgr_offset, tidrecvc->recv_msglen,
				  getreq->tidgr_length);

			if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_CTS_SERIALIZED) {
				/*
				 * Once the CTS was sent, we mark it per 'flow' object
				 * not to proceed with next CTSes until that one is done.
				 */
				struct ips_proto *proto = tidrecvc->protoexp->proto;
				psmi_assert(proto->msgflowid < EP_FLOW_LAST);
				struct ips_flow *flow = &ipsaddr->flows[proto->msgflowid];
				flow->flags |= IPS_FLOW_FLAG_SKIP_CTS;
			}

			/*
			 * nbytes_this is the asked length for this session,
			 * ips_tid_recv_alloc() might register less pages, the
			 * real length is in tidrecvc->recv_msglen.
			 */
			getreq->tidgr_offset += tidrecvc->recv_msglen;
			psmi_assert(getreq->tidgr_offset <=
				    getreq->tidgr_length);

			if (getreq->tidgr_offset == getreq->tidgr_length) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
				if (getreq->gpu_hostbuf_used) {
					/* this completes the tid xfer setup.
					   move to the pending cuda ops queue,
					   set the timer to catch completion */
					STAILQ_REMOVE_HEAD(phead, tidgr_next);
					STAILQ_INSERT_TAIL(
						&getreq->tidgr_protoexp->cudapend_getreqsq,
						getreq, tidgr_next);
					psmi_timer_request(getreq->tidgr_protoexp->timerq,
							   &getreq->tidgr_protoexp->timer_getreqs,
							   PSMI_TIMER_PRIO_1);
					continue;
				}
#endif
				getreq->tidgr_protoexp = NULL;
				getreq->tidgr_epaddr = NULL;
				STAILQ_REMOVE_HEAD(phead, tidgr_next);
				continue;	/* try next grant request */
			}
			else if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_RTS_CTS_INTERLEAVE) {
				/* In case of multi rail, PSM sends one CTS per request
				 * per card after which the request is moved to the end
				 * of the queue.
				 */
				count--;
				if (count)
					goto ipsaddr_next;
				STAILQ_REMOVE_HEAD(phead, tidgr_next);
				STAILQ_INSERT_TAIL(phead, getreq ,tidgr_next);
				continue;
			}

			/* created a tidrecvc, reset count */
			count = ipsaddr->msgctl->ipsaddr_count;
			goto ipsaddr_next;	/* try next fragment on next ipsaddr */
		}

		/*
		 * We need to loop until we can't get a tidrecvc on all
		 * ipsaddrs, then the callbacks on the home protoexp where
		 * getreq is linked can resume this routine. Otherwise, we
		 * might make this getreq to be orphaned and cause deadlock.
		 */
		count--;
		if (count)
			goto ipsaddr_next;
		break;
	}
	PSM2_LOG_MSG("leaving");
	return PSM2_OK;		/* XXX err-broken */
}

#ifdef PSM_HAVE_RDMA
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
static
void psmi_cudamemcpy_tid_to_device(struct ips_tid_recv_desc *tidrecvc)
{
	struct ips_protoexp *protoexp = tidrecvc->protoexp;
	struct ips_gpu_hostbuf *chb;
	const uint32_t transfer_size =
#ifndef PSM_OPA
		tidrecvc->recv_msglen;
#else
		tidrecvc->recv_tidbytes
			+ tidrecvc->tid_list.tsess_unaligned_start
			+ tidrecvc->tid_list.tsess_unaligned_end;
#endif
	chb = tidrecvc->cuda_hostbuf;
	chb->size += transfer_size;

	PSM3_GPU_MEMCPY_HTOD_START(protoexp, chb, transfer_size);

	STAILQ_INSERT_TAIL(&tidrecvc->getreq->pend_cudabuf, chb, next);
	tidrecvc->cuda_hostbuf = NULL;
	ips_tid_pendtids_timer_callback(&tidrecvc->getreq->tidgr_protoexp->timer_getreqs,0);
}
#endif // PSM_CUDA || PSM_ONEAPI
#endif // PSM_HAVE_RDMA

#ifdef PSM_HAVE_RDMA
// we have completed receipt of the TIDs for a given CTS
// For RC QP, this is indicated by RDMA completion w/immediate
static
psm2_error_t ips_tid_recv_free(struct ips_tid_recv_desc *tidrecvc)
{
	struct ips_protoexp *protoexp = tidrecvc->protoexp;
	struct ips_tid_get_request *getreq = tidrecvc->getreq;
#ifdef PSM_OPA
	int tidcount = tidrecvc->tid_list.tsess_tidcount;
#endif
	psm2_error_t err = PSM2_OK;

	psmi_assert(getreq != NULL);
#ifdef PSM_OPA
	psmi_assert(tidcount > 0);
#endif
	psmi_assert(tidrecvc->state == TIDRECVC_STATE_BUSY);

#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
	if (tidrecvc->cuda_hostbuf)
		psmi_cudamemcpy_tid_to_device(tidrecvc);
#endif

#ifndef PSM_OPA
	if (tidrecvc->mr) {
		_HFI_MMDBG("CTS recv chunk complete, releasing MR: rkey: 0x%x\n", tidrecvc->mr->rkey);
        psm3_verbs_release_mr(tidrecvc->mr);
        tidrecvc->mr = NULL;
    }
#elif defined(PSM_OPA)
	if (protoexp->tid_flags & IPS_PROTOEXP_FLAG_TID_DEBUG) {
		int tid, i;

		for (i = 0; i < tidcount; i++) {
			tid =
			    IPS_TIDINFO_GET_TID(tidrecvc->tid_list.
					tsess_list[i]) * 2 +
			    IPS_TIDINFO_GET_TIDCTRL(tidrecvc->tid_list.
					tsess_list[i]) - 1;
			psmi_assert(protoexp->tid_info[tid].state ==
				    TIDSTATE_USED);
			psmi_assert(protoexp->tid_info[tid].tidrecvc ==
				    tidrecvc);
			psmi_assert(protoexp->tid_info[tid].tid ==
				    tidrecvc->tid_list.tsess_list[i]);
			protoexp->tid_info[tid].state = TIDSTATE_FREE;
			protoexp->tid_info[tid].tidrecvc = NULL;
			protoexp->tid_info[tid].tid = 0xFFFFFFFF;
		}
	}

	ips_dump_tids(&tidrecvc->tid_list, "Deregistered %d tids: ",
		      tidrecvc->tid_list.tsess_tidcount);

	if (protoexp->tidc.tid_array) {
		if ((err = ips_tidcache_release(&protoexp->tidc,
			tidrecvc->tid_list.tsess_list, tidcount)))
			goto fail;
	} else {
		if ((err = ips_tid_release(&protoexp->tidc,
			tidrecvc->tid_list.tsess_list, tidcount)))
			goto fail;
	}
#endif

	getreq->tidgr_bytesdone += tidrecvc->recv_msglen;

	_HFI_EXP("req=%p bytes=%d/%d\n",
		 getreq->tidgr_req,
		 getreq->tidgr_bytesdone, getreq->tidgr_length);

	tidrecvc->state = TIDRECVC_STATE_FREE;

	/* finally free the tidflow */
	psm3_ips_tf_deallocate(&protoexp->tfc, tidrecvc->rdescid._desc_idx, 1);

	if (getreq->tidgr_bytesdone == getreq->tidgr_length) {
#if defined(PSM_CUDA) || defined(PSM_ONEAPI)
		/* if cuda, we handle callbacks when the cuda xfer is done */
		if (!getreq->gpu_hostbuf_used) {
			if (getreq->tidgr_callback)
				getreq->tidgr_callback(getreq->tidgr_req);
			psm3_mpool_put(getreq);
		}
#else
		if (getreq->tidgr_callback)
			getreq->tidgr_callback(getreq->tidgr_req);
		psm3_mpool_put(getreq);
#endif
	} else {
		/* We just released some tids.
		 * If requests are waiting on tids to be
		 * freed, queue up the timer */
		if (getreq->tidgr_offset < getreq->tidgr_length) {
			ips_tid_pendtids_timer_callback(&getreq->
							tidgr_protoexp->
							timer_getreqs, 0);
		}
	}

#ifdef PSM_VERBS
	/* we freed some an MR  If we have pending sends or pending get requests,
	 * turn on the timer so it can be processed. */
	ips_tid_mravail_callback(protoexp->proto);
#elif defined(PSM_OPA)
	if (!STAILQ_EMPTY(&protoexp->pend_getreqsq)) {
		psmi_timer_request(protoexp->timerq,
				   &protoexp->timer_getreqs,
				   PSMI_TIMER_PRIO_1);
	}
#endif

#ifdef PSM_OPA
fail:
#endif
	return err;
}
#endif // PSM_HAVE_RDMA

#ifdef PSM_OPA
// This advancaes the generation for our tidflow
psm2_error_t
ips_protoexp_flow_newgen(struct ips_tid_recv_desc *tidrecvc)
{
	psmi_assert_always(tidrecvc->state == TIDRECVC_STATE_BUSY);
	ips_tfgen_allocate(&tidrecvc->protoexp->tfc,
				 tidrecvc->rdescid._desc_idx,
				 &tidrecvc->tidflow_active_gen);

	/* Update tidflow table with new generation number */
	tidrecvc->tidflow_genseq.psn_gen = tidrecvc->tidflow_active_gen;
	psmi_hal_tidflow_set_entry(
			      tidrecvc->rdescid._desc_idx,
			      tidrecvc->tidflow_genseq.psn_gen,
			      tidrecvc->tidflow_genseq.psn_seq,
			      tidrecvc->context->psm_hw_ctxt);
	/* Increment swapped generation count for tidflow */
	tidrecvc->tidflow_nswap_gen++;
	return PSM2_OK;
}
#endif // PSM_OPA

#ifdef PSM_OPA
void ips_protoexp_do_tf_seqerr(void *vpprotoexp
			       /* actually: struct ips_protoexp *protoexp */,
			       void *vptidrecvc
			       /* actually: struct ips_tid_recv_desc *tidrecvc */,
			       struct ips_message_header *p_hdr)
{
	struct ips_protoexp *protoexp = (struct ips_protoexp *) vpprotoexp;
	struct ips_tid_recv_desc *tidrecvc = (struct ips_tid_recv_desc *) vptidrecvc;
	psmi_seqnum_t sequence_num, tf_sequence_num;
	ips_scb_t ctrlscb;

	/* Update stats for sequence errors */
	tidrecvc->stats.nSeqErr++;

	sequence_num.psn_val = __be32_to_cpu(p_hdr->bth[2]);

	/* Only care about sequence error for currently active generation */
	if (tidrecvc->tidflow_active_gen != sequence_num.psn_gen)
		return;

	/* If a "large" number of swapped generation we are loosing packets
	 * for this flow. Request throttling of tidflow by generating a
	 * BECN. With header suppression we will miss some FECN packet
	 * on OPA hence keeping track of swapped generation is another
	 * mechanism to do congestion control for tidflows.
	 *
	 * For mismatched sender/receiver/link speeds we can get into a
	 * deadly embrace where minimal progress is made due to generation
	 * mismatch errors. This can occur if we wrap around the generation
	 * count without making progress. Hence in cases where the swapped
	 * generation count is > 254 stop sending BECN (and the NAK) so the
	 * send -> receiver pipeline is flushed with an error check and things
	 * can sync up. This should be an extremely rare event.
	 */

	if_pf(tidrecvc->tidflow_nswap_gen >= 254)
		return;	/* Do not send NAK. Let error check kick in. */

	if_pf((tidrecvc->tidflow_nswap_gen > 4) &&
	      (protoexp->proto->flags & IPS_PROTO_FLAG_CCA)) {
		_HFI_CCADBG("Generating BECN. Number of swapped gen: %d.\n",
				tidrecvc->tidflow_nswap_gen);
		/* Mark flow to generate BECN in control packet */
		tidrecvc->tidflow.flags |= IPS_FLOW_FLAG_GEN_BECN;

		/* Update stats for congestion encountered */
		protoexp->proto->epaddr_stats.congestion_pkts++;
	}

	/* Get the latest seq from hardware tidflow table, if that value is
	 * reliable. The value is not reliable if context sharing is used,
	 * because context sharing might drop packet even though hardware
	 * has received it successfully. The hardware table may also be
	 * incorrect if RSM is intercepting TID & FECN & SH packets.
	 * We can handle this condition by taking the most recent PSN whether
	 * it comes from the tidflow table or from PSM's own accounting.
	 */
	if (!tidrecvc->context->tf_ctrl) {
		uint64_t tf;
		uint32_t seqno=0;

		psmi_hal_tidflow_get(tidrecvc->rdescid._desc_idx, &tf,
				     tidrecvc->context->psm_hw_ctxt);
		psmi_hal_tidflow_get_seqnum(tf, &seqno);
		tf_sequence_num.psn_val = seqno;

		if (psmi_hal_has_cap(PSM_HAL_CAP_RSM_FECN_SUPP)) {
			if (tf_sequence_num.psn_val > tidrecvc->tidflow_genseq.psn_seq)
				tidrecvc->tidflow_genseq.psn_seq = tf_sequence_num.psn_seq;
		}
		else
			tidrecvc->tidflow_genseq.psn_seq = tf_sequence_num.psn_seq;
	}

	/* Swap generation for the flow. */
	ips_protoexp_flow_newgen(tidrecvc);

	ctrlscb.scb_flags = 0;
	ctrlscb.ips_lrh.data[0] = p_hdr->exp_sdescid;
	/* Keep peer generation but use my last received sequence */
	sequence_num.psn_seq = tidrecvc->tidflow_genseq.psn_seq;
	ctrlscb.ips_lrh.ack_seq_num = sequence_num.psn_val;

	/* My new generation and last received sequence */
	ctrlscb.ips_lrh.data[1].u32w0 = tidrecvc->tidflow_genseq.psn_val;

	// no payload, pass cksum so non-NULL
	psm3_ips_proto_send_ctrl_message(&tidrecvc->tidflow,
				    OPCODE_NAK,
				    &tidrecvc->ctrl_msg_queued,
				    &ctrlscb, ctrlscb.cksum, 0);

	/* Update stats for retransmit */
	tidrecvc->stats.nReXmit++;

	return;
}
#endif // PSM_OPA

#ifdef PSM_OPA
void ips_protoexp_do_tf_generr(void *vpprotoexp
			       /* actually: struct ips_protoexp *protoexp */,
			       void *vptidrecvc
			       /* actually: struct ips_tid_recv_desc *tidrecvc */,
			       struct ips_message_header *p_hdr)
{
	struct ips_tid_recv_desc *tidrecvc = (struct ips_tid_recv_desc *) vptidrecvc;
	/* Update stats for generation errors */
	tidrecvc->stats.nGenErr++;

	/* If packet faced congestion we may want to generate
	 * a CN packet to rate control sender.
	 */

	return;
}
#endif // PSM_OPA
