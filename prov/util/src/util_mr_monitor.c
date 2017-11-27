/*
 * Copyright (c) 2017 Cray Inc. All rights reserved.
 * Copyright (c) 2017 Intel Inc. All rights reserved.
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

#include <ofi_mr.h>

int ofi_mr_monitor_init(struct ofi_mr_monitor *mr_monitor)
{
	/* TODO open Notifier, set subscribe/unsubscribe/get_event here */
	fastlock_init(&mr_monitor->lock);
	dlist_init(&mr_monitor->eq_list);
	return FI_SUCCESS;
}

void ofi_mr_monitor_finalize(struct ofi_mr_monitor *mr_monitor)
{
	/* TODO close Notifier here */
	assert(dlist_empty(&mr_monitor->eq_list));
}

void ofi_mr_monitor_eq_add(struct ofi_mr_monitor *mr_monitor,
			   struct ofi_mr_monitor_eq *eq,
			   int events)
{
	eq->mr_monitor = mr_monitor;
	eq->events = events;
	fastlock_init(&eq->lock);
	dlist_init(&eq->list);
	dlist_insert_tail(&eq->entry, &mr_monitor->eq_list);
}

void ofi_mr_monitor_eq_del(struct ofi_mr_monitor_eq *eq)
{
	assert(dlist_empty(&eq->list));

	fastlock_destroy(&eq->lock);
	dlist_remove(&eq->entry);
}

static void util_mr_monitor_read_events(struct ofi_mr_monitor_eq *eq)
{
	struct ofi_mr_subscription *subscript = NULL;

	do {
		subscript = eq->mr_monitor->get_event(eq);
		if (!subscript) {
			FI_DBG(&core_prov, FI_LOG_MR,
			       "no more events to be read\n");
			break;
		}

		FI_DBG(&core_prov, FI_LOG_MR,
		       "found event, context=%p eq=%p\n",
		       subscript, subscript->eq);

		fastlock_acquire(&subscript->eq->lock);
		dlist_insert_tail(&subscript->entry, &subscript->eq->list);
		fastlock_release(&subscript->eq->lock);
	} while (1);
}

struct ofi_mr_subscription *ofi_mr_monitor_get_event(struct ofi_mr_monitor_eq *eq)
{
	struct ofi_mr_subscription *subscript = NULL;

	assert(eq);

	util_mr_monitor_read_events(eq);

	fastlock_acquire(&eq->lock);
	if (!dlist_empty(&eq->list)) {
		dlist_pop_front(&eq->list, struct ofi_mr_subscription,
				subscript, entry);
		/* to avoid double insertions */
		dlist_init(&subscript->entry);
	}
	fastlock_release(&eq->lock);

	return subscript;
}
