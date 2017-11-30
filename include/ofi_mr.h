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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <inttypes.h>

#include "fi.h"
#include "fi_util.h"

/*
 * MR monitor
 */

struct ofi_mr_subscription;
struct ofi_mr_monitor_eq;

enum ofi_mr_monitor_event {
	OFI_MR_MONITOR_EVENT_MAPPED	= 1 << 1,
	OFI_MR_MONITOR_EVENT_UNMAPPED	= 1 << 2,
};

struct ofi_mr_monitor {
	void			*notifier;
	fastlock_t		lock;
	struct dlist_entry	eq_list;

	int (*subscribe)(struct ofi_mr_monitor *mr_monitor, void *addr,
			 uint64_t len, struct ofi_mr_subscription *subscribe);
	void (*unsubscribe)(struct ofi_mr_monitor *mr_monitor,
			    struct ofi_mr_subscription *subscribe);
	struct ofi_mr_subscription *(*get_event)(struct ofi_mr_monitor_eq *eq);
};

struct ofi_mr_monitor_eq {
	struct ofi_mr_monitor		*mr_monitor;
	struct dlist_entry		entry;
	int				events;
	fastlock_t			lock;
	struct dlist_entry		list;
};

struct ofi_mr_subscription {
	struct ofi_mr_monitor_eq	*eq;
	enum ofi_mr_monitor_event	event;
	/* this entry is used either for
	 * eq::subscirbe_list or for eq::read_list */
	struct dlist_entry		entry;
};

int ofi_mr_monitor_init(struct ofi_mr_monitor *mr_monitor);
void ofi_mr_monitor_finalize(struct ofi_mr_monitor *mr_monitor);
void ofi_mr_monitor_eq_add(struct ofi_mr_monitor *mr_monitor,
			   struct ofi_mr_monitor_eq *eq,
			   int events);
void ofi_mr_monitor_eq_del(struct ofi_mr_monitor_eq *eq);
struct ofi_mr_subscription *ofi_mr_monitor_get_event(struct ofi_mr_monitor_eq *eq);

static inline
int ofi_mr_monitor_subscribe(struct ofi_mr_monitor_eq *eq,
			     void *addr, uint64_t len,
			     struct ofi_mr_subscription *subscript)
{
	int ret;

	ret = eq->mr_monitor->subscribe(eq->mr_monitor, addr, len, subscript);
	if (OFI_LIKELY(!ret))
		FI_DBG(&core_prov, FI_LOG_MR,
		       "monitoring addr=%p len=%"PRIu64" subscript=%p eq=%p\n",
		       addr, len, subscript, eq);
	else
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed (ret = %d) to monitor addr=%p len=%"PRIu64" subscript=%p eq=%p\n",
			ret, addr, len, subscript, eq);
	return ret;
}

static inline
void ofi_mr_monitor_unsubscribe(struct ofi_mr_subscription *subscript)
{
	subscript->eq->mr_monitor->unsubscribe(subscript->eq->mr_monitor,
					       subscript);
	fastlock_acquire(&subscript->eq->lock);
	dlist_remove(&subscript->entry);
	dlist_init(&subscript->entry);
	fastlock_release(&subscript->eq->lock);
}
