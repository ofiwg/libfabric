/*
 * Copyright (c) 2013-2014 Intel Corporation. All rights reserved.
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

struct psm2_am_parameters psmx2_am_param;

static psm2_am_handler_fn_t psmx2_am_handlers[2] = {
	psmx2_am_rma_handler,
	psmx2_am_atomic_handler,
};

static int psmx2_am_handlers_idx[2];
static int psmx2_am_handlers_initialized = 0;

int psmx2_am_progress(struct psmx2_fid_domain *domain)
{
	struct slist_entry *item;
	struct psmx2_am_request *req;
	struct psmx2_trigger *trigger;

	if (psmx2_env.tagged_rma) {
		fastlock_acquire(&domain->rma_queue.lock);
		while (!slist_empty(&domain->rma_queue.list)) {
			item = slist_remove_head(&domain->rma_queue.list);
			req = container_of(item, struct psmx2_am_request, list_entry);
			fastlock_release(&domain->rma_queue.lock);
			psmx2_am_process_rma(domain, req);
			fastlock_acquire(&domain->rma_queue.lock);
		}
		fastlock_release(&domain->rma_queue.lock);
	}

	fastlock_acquire(&domain->trigger_queue.lock);
	while (!slist_empty(&domain->trigger_queue.list)) {
		item = slist_remove_head(&domain->trigger_queue.list);
		trigger = container_of(item, struct psmx2_trigger, list_entry);
		fastlock_release(&domain->trigger_queue.lock);
		psmx2_process_trigger(domain, trigger);
		fastlock_acquire(&domain->trigger_queue.lock);
	}
	fastlock_release(&domain->trigger_queue.lock);

	return 0;
}

int psmx2_am_init(struct psmx2_fid_domain *domain)
{
	psm2_ep_t psm2_ep = domain->psm2_ep;
	size_t size;
	int err = 0;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	psmx2_atomic_init();

	if (!psmx2_am_handlers_initialized) {
		err = psm2_am_get_parameters(psm2_ep, &psmx2_am_param,
					     sizeof(psmx2_am_param), &size);
		if (err)
			return psmx2_errno(err);

		err = psm2_am_register_handlers(psm2_ep, psmx2_am_handlers, 2,
						psmx2_am_handlers_idx);
		if (err)
			return psmx2_errno(err);

		if ((psmx2_am_handlers_idx[0] != PSMX2_AM_RMA_HANDLER) ||
		    (psmx2_am_handlers_idx[1] != PSMX2_AM_ATOMIC_HANDLER)) {
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"failed to register one or more AM handlers "
				"at indecies %d, %d\n", PSMX2_AM_RMA_HANDLER,
				PSMX2_AM_ATOMIC_HANDLER);
			return -FI_EBUSY;
		}

		psmx2_am_handlers_initialized = 1;
	}

	slist_init(&domain->rma_queue.list);
	slist_init(&domain->trigger_queue.list);
	fastlock_init(&domain->rma_queue.lock);
	fastlock_init(&domain->trigger_queue.lock);

	return err;
}

int psmx2_am_fini(struct psmx2_fid_domain *domain)
{
	fastlock_destroy(&domain->rma_queue.lock);
	fastlock_destroy(&domain->trigger_queue.lock);

	psmx2_atomic_fini();

	return 0;
}

