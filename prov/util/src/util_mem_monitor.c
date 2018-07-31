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


void ofi_monitor_init(struct ofi_mem_monitor *monitor)
{
	ofi_atomic_initialize32(&monitor->refcnt, 0);
}

void ofi_monitor_cleanup(struct ofi_mem_monitor *monitor)
{
	assert(ofi_atomic_get32(&monitor->refcnt) == 0);
}

void ofi_monitor_add_queue(struct ofi_mem_monitor *monitor,
			   struct ofi_notification_queue *nq)
{
	fastlock_init(&nq->lock);
	dlist_init(&nq->list);
	fastlock_acquire(&nq->lock);
	nq->refcnt = 0;
	fastlock_release(&nq->lock);

	nq->monitor = monitor;
	ofi_atomic_inc32(&monitor->refcnt);
}

void ofi_monitor_del_queue(struct ofi_notification_queue *nq)
{
	assert(dlist_empty(&nq->list) && (nq->refcnt == 0));
	ofi_atomic_dec32(&nq->monitor->refcnt);
	fastlock_destroy(&nq->lock);
}

int ofi_monitor_subscribe(struct ofi_notification_queue *nq,
			  void *addr, size_t len,
			  struct ofi_subscription *subscription)
{
	int ret;

	FI_DBG(&core_prov, FI_LOG_MR,
	       "subscribing addr=%p len=%zu subscription=%p nq=%p\n",
	       addr, len, subscription, nq);

	/* Ensure the subscription is initialized before we can get events */
	dlist_init(&subscription->entry);

	subscription->nq = nq;
	subscription->addr = addr;
	subscription->len = len;
	fastlock_acquire(&nq->lock);
	nq->refcnt++;
	fastlock_release(&nq->lock);

	ret = nq->monitor->subscribe(nq->monitor, addr, len, subscription);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed (ret = %d) to monitor addr=%p len=%zu",
			ret, addr, len);
		fastlock_acquire(&nq->lock);
		nq->refcnt--;
		fastlock_release(&nq->lock);
	}
	return ret;
}

void ofi_monitor_unsubscribe(struct ofi_subscription *subscription)
{
	FI_DBG(&core_prov, FI_LOG_MR,
	       "unsubscribing addr=%p len=%zu subscription=%p\n",
	       subscription->addr, subscription->len, subscription);
	subscription->nq->monitor->unsubscribe(subscription->nq->monitor,
					       subscription->addr,
					       subscription->len,
					       subscription);
	fastlock_acquire(&subscription->nq->lock);
	dlist_init(&subscription->entry);
	subscription->nq->refcnt--;
	fastlock_release(&subscription->nq->lock);
}

static void util_monitor_read_events(struct ofi_mem_monitor *monitor)
{
	struct ofi_subscription *subscription;

	do {
		subscription = monitor->get_event(monitor);
		if (!subscription) {
			FI_DBG(&core_prov, FI_LOG_MR,
			       "no more events to be read\n");
			break;
		}

		FI_DBG(&core_prov, FI_LOG_MR,
		       "found event, context=%p, addr=%p, len=%zu nq=%p\n",
		       subscription, subscription->addr,
		       subscription->len, subscription->nq);

		fastlock_acquire(&subscription->nq->lock);
		if (dlist_empty(&subscription->entry))
			dlist_insert_tail(&subscription->entry,
					   &subscription->nq->list);
		fastlock_release(&subscription->nq->lock);
	} while (1);
}

struct ofi_subscription *ofi_monitor_get_event(struct ofi_notification_queue *nq)
{
	struct ofi_subscription *subscription;

	util_monitor_read_events(nq->monitor);

	fastlock_acquire(&nq->lock);
	if (!dlist_empty(&nq->list)) {
		dlist_pop_front(&nq->list, struct ofi_subscription,
				subscription, entry);
		/* needed to protect against double insertions */
		dlist_init(&subscription->entry);
	} else {
		subscription = NULL;
	}
	fastlock_release(&nq->lock);

	return subscription;
}
