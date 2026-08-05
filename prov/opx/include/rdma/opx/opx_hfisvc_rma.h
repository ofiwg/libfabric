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
#ifndef _FI_PROV_OPX_HFISVC_RMA_H_
#define _FI_PROV_OPX_HFISVC_RMA_H_

#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/fi_opx_rma.h"
#include "rdma/opx/fi_opx_hfi1_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_HFISVC

/* tracer API is a no-op on develop */
#ifndef OPX_TRACER_TRACE
#define OPX_TRACER_BEGIN       0
#define OPX_TRACER_END_SUCCESS 1
#define OPX_TRACER_END_EAGAIN  2
#define OPX_TRACER_END_ERROR   3
#define OPX_TRACER_TRACE(type, name) \
	do {                         \
		(void) (type);       \
	} while (0)
#endif

int opx_hfisvc_rma_send_rts(union fi_opx_hfi1_deferred_work *work);
int opx_hfisvc_rma_recv_rts(union fi_opx_hfi1_deferred_work *work);

/* bounded-retry cap: past this a stuck iov is dropped-and-drained */
#define OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS (16384ul)

/* RMA path uses hfisvc plane [0], so it is only safe on a single-plane endpoint */
static inline bool opx_hfisvc_single_plane(struct fi_opx_ep *opx_ep)
{
	return opx_ep->domain->hfisvc.num_ctxs == 1;
}

__OPX_FORCE_INLINE__
void opx_hfisvc_rma_invoke_send_rts(struct fi_opx_ep *opx_ep, const struct opx_rma_op_iov *iov, const size_t niov,
				    const struct fi_opx_addr opx_target_addr, const uint64_t ofi_data,
				    struct fi_opx_completion_counter *cc, const uint64_t dput_opcode,
				    const enum fi_datatype dt, const enum fi_op op, const int lock_required,
				    const enum ofi_reliability_kind reliability, const enum opx_hfi1_type hfi1_type)
{
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "HFISVC_RMA_INVOKE_SND_RTS");
	OPX_HFISVC_DEBUG_LOG("HFISVC invoke send RTS with %lu IOVs, dput_opcode=%s\n", niov,
			     opx_hfi1_dput_opcode_to_string((uint8_t) dput_opcode));
	assert(opx_hfisvc_single_plane(opx_ep));
	assert(op == FI_NOOP || op < OFI_ATOMIC_OP_LAST);
	assert(dt == FI_VOID || dt < OFI_DATATYPE_LAST);
	assert(hfi1_type == OPX_HFI1_JKR || hfi1_type == OPX_HFI1_CYR || hfi1_type == OPX_HFI1_WFR ||
	       hfi1_type == OPX_HFI1_MIXED_9B);
	assert(cc != NULL);
	assert(!lock_required);

	uint64_t lrh_dlid;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_MIXED_9B)) {
		lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(opx_target_addr.planes[OPX_PRIMARY_PLANE].lid);
	} else {
		lrh_dlid = opx_target_addr.planes[OPX_PRIMARY_PLANE].lid;
	}
	uint64_t pbc_dlid = OPX_PBC_DLID(opx_target_addr.planes[OPX_PRIMARY_PLANE].lid, hfi1_type);

	union fi_opx_hfi1_deferred_work *work =
		(union fi_opx_hfi1_deferred_work *) ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct opx_hfisvc_rma_rts_params *params = &work->hfisvc_rma_rts;
	params->work_elem.slist_entry.next	 = NULL;
	params->work_elem.completion_action	 = NULL;
	params->work_elem.payload_copy		 = NULL;
	params->work_elem.complete		 = false;
	params->work_elem.work_type		 = OPX_WORK_TYPE_HFISVC;
	params->work_elem.work_fn		 = opx_hfisvc_rma_send_rts;
	params->opx_ep				 = opx_ep;
	params->opx_target_addr			 = opx_target_addr;
	params->cc				 = cc;
	params->context				 = NULL;
	params->lrh_dlid			 = lrh_dlid;
	params->pbc_dlid			 = pbc_dlid;
	params->bth_subctxt_rx = ((uint64_t) opx_target_addr.planes[OPX_PRIMARY_PLANE].hfi1_subctxt_rx)
				 << OPX_BTH_SUBCTXT_RX_SHIFT;
	params->niov		     = niov;
	params->data		     = ofi_data;
	params->dput_opcode	     = dput_opcode;
	params->fi_datatype_dt	     = dt;
	params->fi_op_opcode	     = op;
	params->client_key	     = (uint64_t) opx_ep->domain->hfisvc.ctxs[0].client_key;
	params->cur_iov		     = 0;
	params->iovs_with_keys	     = 0;
	params->send_eagain_attempts = 0;
	params->reliability	     = reliability;
	params->hfi1_type	     = hfi1_type;
	params->plane_idx	     = 0;

	assert(niov <= OPX_MAX_RMA_HFISVC_IOVS);
	for (int i = 0; i < niov; ++i) {
		params->local_buf_iovs[i]	      = iov[i].hmem_iov;
		params->hfisvc_iov[i].len	      = iov[i].hmem_iov.len;
		params->hfisvc_iov[i].origin_offset   = 0UL;
		params->hfisvc_iov[i].remote_auth_key = iov[i].remote_auth_key;
		params->hfisvc_iov[i].remote_offset   = iov[i].remote_offset;
		params->hfisvc_iov[i].dt	      = dt;
		params->hfisvc_iov[i].op	      = op;
	}

	if (slist_empty(&opx_ep->tx->work_pending[work->work_elem.work_type])) {
		ssize_t rc = work->work_elem.work_fn(work);
		if (rc == FI_SUCCESS) {
			assert(work->work_elem.complete);
			OPX_BUF_FREE(work);
			OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_INVOKE_SND_RTS");
			return;
		}

		assert(rc == -FI_EAGAIN);
	}

	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[work->work_elem.work_type]);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "HFISVC_RMA_INVOKE_SND_RTS");
}

#ifdef __cplusplus
}
#endif
#endif // HAVE_HFISVC
#endif /* _FI_PROV_OPX_HFISVC_RMA_H_ */
