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
#ifndef _FI_PROV_OPX_HFISVC_POLL_H_
#define _FI_PROV_OPX_HFISVC_POLL_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "rdma/opx/fi_opx_compiler.h"
#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/fi_opx_internal.h"
#include "rdma/opx/opx_hfisvc.h"

#if HAVE_HFISVC

__OPX_FORCE_INLINE__
void opx_domain_hfisvc_poll(struct fi_opx_domain *opx_domain)
{
	struct hfisvc_client_cq_entry hfisvc_out[64];

	size_t n = (*opx_domain->hfisvc.cq_read)(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out,
						 sizeof(struct hfisvc_client_cq_entry) * 64, 64);
	while (n > 0) {
		OPX_HFISVC_DEBUG_LOG("HFIService: Polled %lu completions from mr_completion_queue!\n", n);
		for (size_t i = 0; i < n; ++i) {
			if (hfisvc_out[i].status != HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS) {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr,
					"(%d) %s:%s():%d Completion error: status was %d type=%d app_context=%lX\n",
					getpid(), __FILE__, __func__, __LINE__, hfisvc_out[i].status,
					hfisvc_out[i].type, hfisvc_out[i].app_context);
				abort();
			}
			struct fi_opx_mr  *opx_mr    = (struct fi_opx_mr *) hfisvc_out[i].app_context;
			hfisvc_client_mr_t mr_handle = hfisvc_out[i].type_mr.mr;

			assert(hfisvc_out[i].status == HFISVC_CLIENT_CQ_ENTRY_STATUS_SUCCESS);

			if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_OPEN) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p hfisvc.mr_handle=%u state=PENDING_OPEN -> KEY_ALLOC\n",
					opx_mr, (uint32_t) mr_handle);
				opx_mr->hfisvc.mr_handle = mr_handle;
				opx_mr->hfisvc.state	 = OPX_MR_HFISVC_STATE_PENDING_KEY_ALLOC;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_KEY_ENABLE -> OPENED\n", opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_OPENED;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_OPENED) {
				assert(hfisvc_out[i].type_notify.access_key == opx_mr->hfisvc.access_key);
				OPX_HFISVC_DEBUG_LOG("Notify completion opx_mr=%p imm_data=%lX\n", opx_mr,
						     hfisvc_out[i].type_notify.imm_data);

				// TODO: Use generic struct for hvisvc mr completion
				struct opx_hfisvc_rzv_completion_tmp {
					struct opx_context *context;
					union {
						struct {
							uint64_t tid_length;
							uint64_t tid_vaddr;
						};
						struct {
							// uintptr_t app_context;
							uint64_t unused;
							uint32_t access_key;
							uint32_t unused_also;
						};
					};
					uint64_t byte_counter;
					uint64_t bytes_accumulated;
				} *rzv_comp =
					(struct opx_hfisvc_rzv_completion_tmp *) hfisvc_out[i].type_notify.imm_data;

				struct opx_context *context = rzv_comp->context;
				OPX_HFISVC_DEBUG_LOG(
					"STRIPE-MR-NOTIFY: MR notify completion for opx_mr=%p rzv_comp=%p context=%p byte_counter=%lu (no decrement/no free)\n",
					opx_mr, rzv_comp, context, context ? context->byte_counter : 0UL);
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_KEY_DISABLE) {
				opx_hfisvc_keyset_free_key(opx_domain->hfisvc.ctxs[0].access_key_set,
							   opx_mr->hfisvc.access_key, NULL);
				opx_mr->hfisvc.access_key = (uint32_t) -1;
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_KEY_DISABLE -> PENDING_DEREGISTER\n",
					opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_DEREGISTER;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_DEREGISTER) {
				OPX_HFISVC_DEBUG_LOG(
					"MR State transition opx_mr=%p state=PENDING_DEREGISTER -> PENDING_CLOSE\n",
					opx_mr);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_PENDING_CLOSE;
			} else if (opx_mr->hfisvc.state == OPX_MR_HFISVC_STATE_PENDING_CLOSE) {
				OPX_HFISVC_DEBUG_LOG("MR State transition opx_mr=%p state=PENDING_CLOSE -> CLOSED\n",
						     opx_mr);
				assert(opx_mr->hfisvc.mr_handle == mr_handle);
				opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_CLOSED;
			} else if (opx_mr->hfisvc.state & OPX_MR_HFISVC_STATE_CLOSE_ISSUED) {
				if (opx_mr->hfisvc.state ==
				    (OPX_MR_HFISVC_STATE_PENDING_OPEN | OPX_MR_HFISVC_STATE_CLOSE_ISSUED)) {
					OPX_HFISVC_DEBUG_LOG(
						"MR State transition opx_mr=%p hfisvc.mr_handle=%u state=PENDING_OPEN with CLOSE_ISSUED -> PENDING_DEREGISTER\n",
						opx_mr, (uint32_t) mr_handle);
					opx_mr->hfisvc.mr_handle = mr_handle;
					opx_mr->hfisvc.state	 = OPX_MR_HFISVC_STATE_PENDING_DEREGISTER;
				} else if (opx_mr->hfisvc.state == (OPX_MR_HFISVC_STATE_PENDING_KEY_ENABLE |
								    OPX_MR_HFISVC_STATE_CLOSE_ISSUED)) {
					OPX_HFISVC_DEBUG_LOG(
						"MR State transition opx_mr=%p state=PENDING_KEY_ENABLE with CLOSE_ISSUED -> OPENED\n",
						opx_mr);
					opx_mr->hfisvc.state = OPX_MR_HFISVC_STATE_OPENED;
				}
			} else {
				// TODO: FI_WARN, post some kind of error to the error queue
				fprintf(stderr,
					"(%d) %s:%s():%d Got unexpected completion for opx_mr=%p state=%d completion: type=%d status=%d\n",
					getpid(), __FILE__, __func__, __LINE__, opx_mr, opx_mr->hfisvc.state,
					hfisvc_out[i].type, hfisvc_out[i].status);
				assert(0);
			}
		}
		n = (*opx_domain->hfisvc.cq_read)(opx_domain->hfisvc.mr_completion_queue, 0ul /* flags */, hfisvc_out,
						  sizeof(struct hfisvc_client_cq_entry) * 64, 64);
	}

	opx_domain_deferred_work_do(opx_domain);
}

#else /* !HAVE_HFISVC */

__OPX_FORCE_INLINE__
void opx_domain_hfisvc_poll(struct fi_opx_domain *opx_domain)
{
	(void) opx_domain;
	abort();
}

#endif /* HAVE_HFISVC */

#endif
