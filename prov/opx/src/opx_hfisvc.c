/*
 * Copyright (C) 2025-2026 Cornelis Networks.
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

#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/opx_tracer.h"

#if HAVE_HFISVC

#include <infiniband/hfi1dv.h>
#include <infiniband/verbs.h>
#include <infiniband/hfisvc_client.h>

#include "rdma/opx/opx_hfi1_rdma_core.h"

#endif

/**
 * @brief When OPX_HFISVC_DEBUG is defined in the build, we'll put out lots of
 * logs via the OPX_HFISVC_DEBUG_LOG macro. The logging can be turned off
 * at runtime by defining the env. variable OPX_HFISVC_LOG_DISABLE in the run
 * command. This is useful for not needing to rebuild entirely in order to
 * toggle logging on/off.
 */
int opx_hfisvc_log_enabled = 1;

int opx_hfisvc_deferred_recv_rts(union fi_opx_hfi1_deferred_work *work)
{
#if HAVE_HFISVC
	struct opx_hfisvc_recv_rts_params *params = &work->hfisvc_rts_params;

	FI_DBG_TRACE(
		fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- RENDEZVOUS RTS HFISVC (deferred) (begin) context %p\n",
		params->context);

	OPX_TRACE_RX_BEGIN(OPX_TRACE_EVENT_HFISVC_RZV_RTS, 0, 0);

	struct fi_opx_ep   *opx_ep   = params->opx_ep;
	struct opx_context *context  = params->context;
	const uint32_t	    niov     = params->niov;
	uint32_t	    cur_iov  = params->cur_iov;
	uint8_t		   *recv_buf = (uint8_t *) params->recv_buf;

	int rc	       = FI_SUCCESS;
	int read_count = 0;

	int plane_read_count[OPX_MAX_TX_CONTEXTS] = {0};
	for (int i = cur_iov; i < niov; ++i) {
		/* Extract plane index from sender LID and mask LID for API calls */
		const uint8_t	plane_idx	= OPX_LID_PLANE_GET_IDX(params->iovs[i].sender_lid);
		const opx_lid_t sbuf_lid_masked = OPX_LID_PLANE_GET_LID(params->iovs[i].sender_lid);

		const uint32_t sbuf_access_key = params->iovs[i].access_key;
		const uint32_t sbuf_client_key = params->iovs[i].client_key;
		const uint64_t sbuf_len	       = params->iovs[i].len;
		const uint64_t sbuf_offset     = params->iovs[i].offset;

		struct fi_opx_rzv_completion *recv_rzv_comp =
			(struct fi_opx_rzv_completion *) ofi_buf_alloc(opx_ep->rzv_completion_pool);
		if (OFI_UNLIKELY(recv_rzv_comp == NULL)) {
			OPX_HFISVC_DEBUG_LOG("ENOMEM (recv_rzv_comp in deferred)\n");
			params->cur_iov	 = i;
			params->recv_buf = (void *) recv_buf;
			rc		 = -FI_EAGAIN;
			break;
		}
		recv_rzv_comp->context	  = context;
		recv_rzv_comp->access_key = (uint32_t) -1;

		struct hfisvc_client_completion completion = {
			.flags		= HFISVC_CLIENT_COMPLETION_FLAG_CQ,
			.cq.app_context = (uint64_t) recv_rzv_comp,
			.cq.handle	= opx_ep->hfisvc.internal_completion_queues[plane_idx],
		};

		const uintptr_t sender_rzv_comp = params->iovs[i].rzv_comp_vaddr;

		if (context->flags & FI_OPX_CQ_CONTEXT_DMABUF_HMEM) {
			struct fi_opx_mr *opx_mr = ((struct fi_opx_hmem_info *) context->hmem_info_qws)->dmabuf.opx_mr;
			if (opx_mr->hfisvc.state != OPX_MR_HFISVC_STATE_OPENED) {
				OPX_BUF_FREE(recv_rzv_comp);
				rc = -FI_EAGAIN;
				break;
			}
			uint64_t local_offset = opx_mr_dmabuf_local_offset(opx_mr, recv_buf);
			rc		      = (*opx_ep->domain->hfisvc.cmd_rdma_read)(
				   opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */, sbuf_lid_masked,
				   sbuf_client_key, sbuf_len, sender_rzv_comp, sbuf_access_key, sbuf_offset,
				   opx_mr->hfisvc.mr_handle, local_offset);
			if (rc != FI_SUCCESS) {
				OPX_BUF_FREE(recv_rzv_comp);
				params->cur_iov	 = i;
				params->recv_buf = (void *) recv_buf;
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.rzv_recv_rts.eagain_hfisvc);
				OPX_HFISVC_DEBUG_LOG(
					"[%d/%d] rdma_read failed with rc=%d context=%p recv-mr_handle=%u recv-offset=%lu sbuf_key=%u, sbuf_access_key=%u sbuf_len=%lu\n",
					i + 1, niov, rc, context, (uint32_t) opx_mr->hfisvc.mr_handle, local_offset,
					sbuf_client_key, sbuf_access_key, sbuf_len);
				rc = -FI_EAGAIN;
				break;
			}
			++read_count;
			++plane_read_count[plane_idx];

			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.rzv_recv_rts.rdma_read);

			OPX_HFISVC_DEBUG_LOG(
				"[%d/%d] Successfully issued rdma_read sbuf_lid=%u context=%p recv-mr_handle=%u recv-offset=%lu sbuf_key=%u, sbuf_access_key=%u sbuf_len=%lu sender_rzv_comp=%lX\n",
				i + 1, niov, sbuf_lid_iov, context, (uint32_t) opx_mr->hfisvc.mr_handle, local_offset,
				sbuf_client_key, sbuf_access_key, sbuf_len, sender_rzv_comp);
		} else {
			rc = (*opx_ep->domain->hfisvc.cmd_rdma_read_va)(
				opx_ep->hfisvc.command_queues[plane_idx], completion, 0ul /* flags */, sbuf_lid_masked,
				sbuf_client_key, sbuf_len, sender_rzv_comp, sbuf_access_key, sbuf_offset, recv_buf);
			if (rc != FI_SUCCESS) {
				OPX_BUF_FREE(recv_rzv_comp);
				params->cur_iov	 = i;
				params->recv_buf = (void *) recv_buf;
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.rzv_recv_rts.eagain_hfisvc);
				OPX_HFISVC_DEBUG_LOG(
					"[%d/%d] rdma_read failed with rc=%d context=%p recv_buf=%p sbuf_key=%u, sbuf_access_key=%u sbuf_len=%lu\n",
					i + 1, niov, rc, context, recv_buf, sbuf_client_key, sbuf_access_key, sbuf_len);
				rc = -FI_EAGAIN;
				break;
			}
			++read_count;
			++plane_read_count[plane_idx];

			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.rzv_recv_rts.rdma_read);

			OPX_HFISVC_DEBUG_LOG(
				"STRIPE-RECV-DEFERRED: [IOV %d/%d] Issued RDMA read on plane %d: context=%p recv_buf=%p sbuf_lid=%u sbuf_client_key=%u sbuf_access_key=%u sbuf_len=%lu recv_rzv_comp=%p\n",
				i + 1, niov, i, context, recv_buf, sbuf_lid_iov, sbuf_client_key, sbuf_access_key,
				sbuf_len, recv_rzv_comp);
		}
		recv_buf += sbuf_len;
	}

	for (int i = 0; i < OPX_MAX_TX_CONTEXTS; ++i) {
		if (plane_read_count[i] > 0) {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hfisvc.doorbell_ring.deferred_work);
			__attribute__((unused)) int doorbell_rc =
				(*opx_ep->domain->hfisvc.doorbell)(opx_ep->domain->hfisvc.ctxs[i].ctx);
			assert(doorbell_rc == 0);
		}
	}

	FI_DBG_TRACE(
		fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- RENDEZVOUS RTS HFISVC (deferred) (end) context %p\n",
		params->context);

	OPX_TRACE_RX_END(OPX_TRACE_EVENT_HFISVC_RZV_RTS,
			 rc ? OPX_TRACE_STATUS_END_EAGAIN : OPX_TRACE_STATUS_END_SUCCESS, 0, 0);

	return rc;
#else
	return 0;
#endif
}

int opx_hfisvc_deferred_recv_rts_enqueue(struct fi_opx_ep *opx_ep, struct opx_context *context, const uint32_t niov,
					 const void *recv_buf, const union opx_hfisvc_iov *iovs)
{
	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	if (OFI_UNLIKELY(work == NULL)) {
		OPX_HFISVC_DEBUG_LOG("Error allocating deferred work for hfisvc recv rts\n");
		return -FI_ENOMEM;
	}
	struct opx_hfisvc_recv_rts_params *params = &work->hfisvc_rts_params;
	params->work_elem.slist_entry.next	  = NULL;
	params->work_elem.completion_action	  = NULL;
	params->work_elem.payload_copy		  = NULL;
	params->work_elem.work_fn		  = opx_hfisvc_deferred_recv_rts;
	params->work_elem.work_type		  = OPX_WORK_TYPE_HFISVC;
	params->work_elem.complete		  = false;
	params->opx_ep				  = opx_ep;
	params->context				  = context;
	params->niov				  = niov;
	params->cur_iov				  = 0;
	params->recv_buf			  = (void *) recv_buf;

	for (int i = 0; i < niov; ++i) {
		params->iovs[i] = iovs[i];
	}

	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[OPX_WORK_TYPE_HFISVC]);

	return FI_SUCCESS;
}
