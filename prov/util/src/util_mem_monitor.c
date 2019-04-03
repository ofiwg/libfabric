/*
 * Copyright (c) 2017 Cray Inc. All rights reserved.
 * Copyright (c) 2017-2019 Intel Inc. All rights reserved.
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


struct ofi_mem_monitor uffd_monitor;

void ofi_monitor_init(struct ofi_mem_monitor *monitor)
{
	fastlock_init(&monitor->lock);
}

void ofi_monitor_cleanup(struct ofi_mem_monitor *monitor)
{
	assert(dlist_empty(&monitor->list));
}

void ofi_monitor_add_cache(struct ofi_mem_monitor *monitor,
			   struct ofi_mr_cache *cache)
{
	cache->monitor = monitor;
	fastlock_acquire(&monitor->lock);
	dlist_insert_tail(&cache->notify_entry, &monitor->list);
	fastlock_release(&monitor->lock);
}

void ofi_monitor_del_cache(struct ofi_mr_cache *cache)
{
	fastlock_acquire(&cache->monitor->lock);
	dlist_remove(&cache->notify_entry);
	fastlock_release(&cache->monitor->lock);
}

int ofi_monitor_subscribe(struct ofi_mem_monitor *monitor,
			  const void *addr, size_t len)
{
	int ret;

	FI_DBG(&core_prov, FI_LOG_MR,
	       "subscribing addr=%p len=%zu\n", addr, len);

	ret = monitor->subscribe(monitor, addr, len);
	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed (ret = %d) to monitor addr=%p len=%zu",
			ret, addr, len);
	}
	return ret;
}

void ofi_monitor_unsubscribe(struct ofi_mem_monitor *monitor,
			     const void *addr, size_t len)
{
	FI_DBG(&core_prov, FI_LOG_MR,
	       "unsubscribing addr=%p len=%zu\n", addr, len);
	monitor->unsubscribe(monitor, addr, len);
}
