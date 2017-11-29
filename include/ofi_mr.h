/*
 * Copyright (c) 2017 Intel Corporation, Inc. All rights reserved.
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

#ifndef _OFI_MR_H_
#define _OFI_MR_H_

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <inttypes.h>

#include <fi.h>
#include <fi_atom.h>
#include <fi_lock.h>
#include <fi_list.h>

/*
 * Memory notifier - Report memory mapping changes to address ranges
 */

struct ofi_subscription;
struct ofi_notification_queue;

struct ofi_mem_monitor {
	ofi_atomic32_t			refcnt;

	int (*subscribe)(struct ofi_mem_monitor *notifier, void *addr,
			 size_t len, struct ofi_subscription *subscription);
	void (*unsubscribe)(struct ofi_mem_monitor *notifier, void *addr,
			    size_t len, struct ofi_subscription *subscription);
	struct ofi_subscription *(*get_event)(struct ofi_mem_monitor *notifier);
};

struct ofi_notification_queue {
	struct ofi_mem_monitor		*monitor;
	fastlock_t			lock;
	struct dlist_entry		list;
	int				refcnt;
};

struct ofi_subscription {
	struct ofi_notification_queue	*nq;
	struct dlist_entry		entry;
};

void ofi_monitor_init(struct ofi_mem_monitor *monitor);
void ofi_monitor_cleanup(struct ofi_mem_monitor *monitor);
void ofi_monitor_add_queue(struct ofi_mem_monitor *monitor,
			   struct ofi_notification_queue *nq);
void ofi_monitor_del_queue(struct ofi_notification_queue *nq);

int ofi_monitor_subscribe(struct ofi_notification_queue *nq,
			  void *addr, size_t len,
			  struct ofi_subscription *subscription);
void ofi_monitor_unsubscribe(void *addr, size_t len,
			      struct ofi_subscription *subscription);
struct ofi_subscription *ofi_monitor_get_event(struct ofi_notification_queue *nq);

#endif /* _OFI_MR_H_ */
