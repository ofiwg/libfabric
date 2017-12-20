/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

int psmx2_am_progress(struct psmx2_trx_ctxt *trx_ctxt)
{
	struct slist_entry *item;
	struct psmx2_am_request *req;
	struct psmx2_trigger *trigger;

	if (psmx2_env.tagged_rma) {
		psmx2_lock(&trx_ctxt->rma_queue.lock, 2);
		while (!slist_empty(&trx_ctxt->rma_queue.list)) {
			item = slist_remove_head(&trx_ctxt->rma_queue.list);
			req = container_of(item, struct psmx2_am_request, list_entry);
			psmx2_unlock(&trx_ctxt->rma_queue.lock, 2);
			psmx2_am_process_rma(trx_ctxt, req);
			psmx2_lock(&trx_ctxt->rma_queue.lock, 2);
		}
		psmx2_unlock(&trx_ctxt->rma_queue.lock, 2);
	}

	psmx2_lock(&trx_ctxt->trigger_queue.lock, 2);
	while (!slist_empty(&trx_ctxt->trigger_queue.list)) {
		item = slist_remove_head(&trx_ctxt->trigger_queue.list);
		trigger = container_of(item, struct psmx2_trigger, list_entry);
		psmx2_unlock(&trx_ctxt->trigger_queue.lock, 2);
		psmx2_process_trigger(trx_ctxt, trigger);
		psmx2_lock(&trx_ctxt->trigger_queue.lock, 2);
	}
	psmx2_unlock(&trx_ctxt->trigger_queue.lock, 2);

	return 0;
}

int psmx2_am_init(struct psmx2_trx_ctxt *trx_ctxt)
{
	psm2_am_handler_2_fn_t psmx2_am_handlers[4];
	struct psmx2_trx_ctxt *hctx[4];
	int psmx2_am_handlers_idx[4];
	int num_handlers = 4;

	psm2_ep_t psm2_ep = trx_ctxt->psm2_ep;
	size_t size;
	int err = 0;
	uint32_t max_atomic_size;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	if (!trx_ctxt->am_initialized) {
		err = psm2_am_get_parameters(psm2_ep, &trx_ctxt->psm2_am_param,
					     sizeof(struct psm2_am_parameters),
					     &size);
		if (err)
			return psmx2_errno(err);

		max_atomic_size = trx_ctxt->psm2_am_param.max_request_short;
		if (trx_ctxt->domain->max_atomic_size > max_atomic_size)
			trx_ctxt->domain->max_atomic_size = max_atomic_size;

		psmx2_am_handlers[0] = psmx2_am_rma_handler;
		hctx[0] = trx_ctxt;
		psmx2_am_handlers[1] = psmx2_am_atomic_handler;
		hctx[1] = trx_ctxt;
		psmx2_am_handlers[2] = psmx2_am_sep_handler;
		hctx[2] = trx_ctxt;
		psmx2_am_handlers[3] = psmx2_am_trx_ctxt_handler;
		hctx[3] = trx_ctxt;

		err = psm2_am_register_handlers_2(psm2_ep, psmx2_am_handlers,
						num_handlers, (void **)hctx, psmx2_am_handlers_idx);
		if (err)
			return psmx2_errno(err);

		if ((psmx2_am_handlers_idx[0] != PSMX2_AM_RMA_HANDLER) ||
		    (psmx2_am_handlers_idx[1] != PSMX2_AM_ATOMIC_HANDLER) ||
		    (psmx2_am_handlers_idx[2] != PSMX2_AM_SEP_HANDLER) ||
		    (psmx2_am_handlers_idx[3] != PSMX2_AM_TRX_CTXT_HANDLER)) {
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"failed to register one or more AM handlers "
				"at indecies %d, %d, %d, %d\n", PSMX2_AM_RMA_HANDLER,
				PSMX2_AM_ATOMIC_HANDLER, PSMX2_AM_SEP_HANDLER,
				PSMX2_AM_TRX_CTXT_HANDLER);
			return -FI_EBUSY;
		}

		trx_ctxt->am_initialized = 1;
	}

	return err;
}

void psmx2_am_fini(struct psmx2_trx_ctxt *trx_ctxt)
{
	/* there is no way to unregister AM handlers */
}

