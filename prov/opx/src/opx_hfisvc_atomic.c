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
#include "rdma/opx/opx_hfisvc.h"
#include "rdma/opx/opx_hfisvc_rma.h"

#if HAVE_HFISVC
int opx_hfisvc_atomic_stage_work(union fi_opx_hfi1_deferred_work *work)
{
	struct opx_hfisvc_atomic_stage_params *p      = &work->hfisvc_atomic_stage;
	struct fi_opx_ep		      *opx_ep = p->opx_ep;

	if (!p->needs_doorbell_only) {
		struct hfisvc_client_completion c2 = {
			.flags		= OPX_HFISVC_CMPL_CQ,
			.cq.handle	= opx_ep->hfisvc.internal_completion_queues[p->ctx_idx],
			.cq.app_context = (uint64_t) p->ic,
		};
		int rc = (*opx_ep->domain->hfisvc.cmd_rdma_read_va)(opx_ep->hfisvc.command_queues[p->ctx_idx], c2, 0ul,
								    p->origin_lid, p->op_key, p->pull_len, 0ul,
								    p->acc_key, p->origin_off, p->dst);
		if (rc == -FI_EAGAIN) {
			if (OFI_UNLIKELY(++p->eagain_attempts >= OPX_HFISVC_RMA_MAX_EAGAIN_ATTEMPTS)) {
				goto hard_fail;
			}
			return -FI_EAGAIN;
		}
		if (OFI_UNLIKELY(rc)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"hfisvc cmd_rdma_read_va (atomic operand pull) failed rc=%d; op dropped\n", rc);
			goto hard_fail;
		}
		/* command is in flight; from here staging must not be freed on any error path */
		p->needs_doorbell_only = 1;
	}

	int drc = (*opx_ep->domain->hfisvc.doorbell)(opx_ep->domain->hfisvc.ctxs[p->ctx_idx].ctx);
	if (OFI_UNLIKELY(drc)) {
		/* the read command is already queued; keep re-ringing the doorbell on every poll (as
		 * opx_hfisvc_rma_recv_rts does) rather than abandoning an in-flight pull and leaking its staging */
		return -FI_EAGAIN;
	}

	work->work_elem.complete = true;
	return FI_SUCCESS;

hard_fail:
	OPX_BUF_FREE(p->ic);
	if (p->aop->context) {
		struct opx_context *context   = p->aop->context;
		context->err_entry.err	      = FI_EIO;
		context->err_entry.prov_errno = FI_EIO;
		context->byte_counter	      = 0;
		opx_hfisvc_rma_context_drain_and_route(opx_ep, context, 0);
	}
	OPX_BUF_FREE(p->aop->staging);
	OPX_BUF_FREE(p->aop);
	work->work_elem.complete = true;
	return FI_SUCCESS;
}

void opx_hfisvc_atomic_poll_advance(struct fi_opx_ep *opx_ep, struct opx_hfisvc_xfer_completion *completion,
				    struct opx_context *context, int ctx_idx)
{
	struct opx_hfisvc_atomic_op *aop      = completion->atomic_op;
	const uint8_t		     terminal = (completion->type == OPX_HFISVC_XFER_TYPE_ATOMIC_FETCH_COMPARE) ? 3 : 2;

	if (aop->pending < terminal) {
		const bool     pull_swap  = (terminal == 3) ? (aop->pending == 1) : true;
		uint8_t	      *dst	  = (terminal == 3 && !pull_swap) ? aop->staging + aop->op_len : aop->staging;
		const uint32_t op_key	  = pull_swap ? aop->op_client_key : aop->op2_client_key;
		const uint32_t acc_key	  = pull_swap ? aop->op_access_key : aop->op2_access_key;
		const uint64_t origin_off = pull_swap ? aop->op_origin_offset : aop->op2_origin_offset;
		const uint32_t pull_len	  = pull_swap ? aop->op_len : aop->op2_len;

		struct opx_hfisvc_xfer_completion *ic2 =
			(struct opx_hfisvc_xfer_completion *) ofi_buf_alloc(opx_ep->hfisvc.completion_pool);
		if (OFI_UNLIKELY(ic2 == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"HFISVC atomic operand-pull completion_pool exhausted; op dropped\n");
			if (context) {
				context->err_entry.err	      = FI_EIO;
				context->err_entry.prov_errno = FI_EIO;
				context->byte_counter	      = 0;
				opx_hfisvc_rma_context_drain_and_route(opx_ep, context, 0);
			}
			OPX_BUF_FREE(aop->staging);
			OPX_BUF_FREE(aop);
			return;
		}

		ic2->type	= completion->type;
		ic2->access_key = completion->access_key;
		ic2->context	= aop->context;
		ic2->cc		= aop->cc;
		ic2->opx_mr	= aop->opx_mr;
		ic2->opx_ep	= opx_ep;
		ic2->len	= aop->apply_len;
		ic2->flags	= 0;
		ic2->atomic_op	= aop;

		aop->pending = (terminal == 3) ? (pull_swap ? 2 : 3) : 2;

		/* handoff: clear atomic_op before enqueue so a late completion for THIS stage cannot free
		 * staging the pending pull will DMA into (completion_pool is NO_ZERO) */
		completion->atomic_op = NULL;

		union fi_opx_hfi1_deferred_work *work =
			(union fi_opx_hfi1_deferred_work *) ofi_buf_alloc(opx_ep->tx->work_pending_pool);
		if (OFI_UNLIKELY(work == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"HFISVC atomic operand-pull work_pending_pool exhausted; op dropped\n");
			OPX_BUF_FREE(ic2);
			if (context) {
				context->err_entry.err	      = FI_EIO;
				context->err_entry.prov_errno = FI_EIO;
				context->byte_counter	      = 0;
				opx_hfisvc_rma_context_drain_and_route(opx_ep, context, 0);
			}
			OPX_BUF_FREE(aop->staging);
			OPX_BUF_FREE(aop);
			return;
		}
		struct opx_hfisvc_atomic_stage_params *p = &work->hfisvc_atomic_stage;
		p->work_elem.slist_entry.next		 = NULL;
		p->work_elem.work_fn			 = opx_hfisvc_atomic_stage_work;
		p->work_elem.completion_action		 = NULL;
		p->work_elem.work_type			 = OPX_WORK_TYPE_HFISVC;
		p->work_elem.complete			 = false;
		p->work_elem.payload_copy		 = NULL;
		p->opx_ep				 = opx_ep;
		p->aop					 = aop;
		p->ic					 = ic2;
		p->dst					 = dst;
		p->origin_off				 = origin_off;
		p->op_key				 = op_key;
		p->acc_key				 = acc_key;
		p->pull_len				 = pull_len;
		p->origin_lid				 = aop->origin_lid;
		p->ctx_idx				 = ctx_idx;
		p->eagain_attempts			 = 0;
		p->needs_doorbell_only			 = 0;

		if (slist_empty(&opx_ep->tx->work_pending[OPX_WORK_TYPE_HFISVC])) {
			int rc = work->work_elem.work_fn(work);
			if (rc == FI_SUCCESS) {
				assert(work->work_elem.complete);
				OPX_BUF_FREE(work);
				return;
			}
			assert(rc == -FI_EAGAIN);
		}
		assert(work->work_elem.slist_entry.next == NULL);
		slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[OPX_WORK_TYPE_HFISVC]);
		return;
	}

	assert(aop->op_len == aop->apply_len);
	if (terminal == 3) {
		assert(aop->op2_len == aop->op_len);
	}

	uint64_t	   hmem_device;
	uint64_t	   hmem_handle;
	enum fi_hmem_iface hmem_iface = opx_hmem_get_mr_iface(aop->opx_mr, &hmem_device, &hmem_handle);
	(void) hmem_iface;
	assert(aop->apply_len <= OPX_HFISVC_ATOMIC_STAGING_MAX);
	OPX_HMEM_ATOMIC_DISPATCH_SIZED(aop->staging, aop->target_vaddr, aop->apply_len, aop->dt, aop->op, hmem_iface,
				       hmem_device, opx_ep->hfisvc.atomic_apply_scratch, OPX_HFISVC_ATOMIC_STAGING_MAX);

	const uint64_t len = completion->len;
	if (completion->cc) {
		struct fi_opx_completion_counter *cc = completion->cc;
		assert(len <= cc->byte_counter);
		cc->byte_counter -= len;
		if (cc->byte_counter == 0) {
			opx_hfisvc_rma_cc_hit_zero(cc);
		}
	} else if (context) {
		opx_hfisvc_rma_context_drain_and_route(opx_ep, context, len);
	}

	OPX_BUF_FREE(aop->staging);
	OPX_BUF_FREE(aop);
}
#endif
