/*
 * Copyright (c) 2017 Cray Inc. All rights reserved.
 * Copyright (c) 2017-2019 Intel Inc. All rights reserved.
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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


static struct ofi_uffd uffd;
struct ofi_mem_monitor *uffd_monitor = &uffd.monitor;


/*
 * Initialize all available memory monitors
 */
void ofi_monitor_init(void)
{
	fastlock_init(&uffd_monitor->lock);
	dlist_init(&uffd_monitor->list);

	fi_param_define(NULL, "mr_cache_max_size", FI_PARAM_SIZE_T,
			"Defines the total number of bytes for all memory"
			" regions that may be tracked by the MR cache."
			" Setting this will reduce the amount of memory"
			" not actively in use that may be registered."
			" (default: 0 no limit is enforced)");
	fi_param_define(NULL, "mr_cache_max_count", FI_PARAM_SIZE_T,
			"Defines the total number of memory regions that"
			" may be store in the cache.  Setting this will"
			" reduce the number of registered regions, regardless"
			" of their size, stored in the cache.  Setting this"
			" to zero will disable MR caching.  (default: 1024)");
	fi_param_define(NULL, "mr_cache_merge_regions", FI_PARAM_BOOL,
			"If set to true, overlapping or adjacent memory"
			" regions will be combined into a single, larger"
			" region.  Merging regions can reduce the cache"
			" memory footprint, but can negatively impact"
			" performance in some situations.  (default: false)");

	fi_param_get_size_t(NULL, "mr_cache_max_size", &cache_params.max_size);
	fi_param_get_size_t(NULL, "mr_cache_max_count", &cache_params.max_cnt);
	fi_param_get_bool(NULL, "mr_cache_merge_regions",
			  &cache_params.merge_regions);

	if (!cache_params.max_size)
		cache_params.max_size = SIZE_MAX;
}

void ofi_monitor_cleanup(void)
{
	assert(dlist_empty(&uffd_monitor->list));
	fastlock_destroy(&uffd_monitor->lock);
}

int ofi_monitor_add_cache(struct ofi_mem_monitor *monitor,
			  struct ofi_mr_cache *cache)
{
	int ret = 0;

	fastlock_acquire(&monitor->lock);
	if (dlist_empty(&monitor->list)) {
		if (monitor == uffd_monitor)
			ret = ofi_uffd_init();
		else
			ret = -FI_ENOSYS;

		if (ret)
			goto out;
	}
	cache->monitor = monitor;
	dlist_insert_tail(&cache->notify_entry, &monitor->list);
out:
	fastlock_release(&monitor->lock);
	return ret;
}

void ofi_monitor_del_cache(struct ofi_mr_cache *cache)
{
	struct ofi_mem_monitor *monitor = cache->monitor;

	if (!monitor)
		return;

	fastlock_acquire(&monitor->lock);
	dlist_remove(&cache->notify_entry);

	if (dlist_empty(&monitor->list) && (monitor == uffd_monitor))
		ofi_uffd_cleanup();
	fastlock_release(&monitor->lock);
}

/* Must be called holding monitor lock */
void ofi_monitor_notify(struct ofi_mem_monitor *monitor,
			const void *addr, size_t len)
{
	struct ofi_mr_cache *cache;

	dlist_foreach_container(&monitor->list, struct ofi_mr_cache,
			cache, notify_entry) {
		ofi_mr_cache_notify(cache, addr, len);
	}
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
			"Failed (ret = %d) to monitor addr=%p len=%zu\n",
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

#if HAVE_UFFD_UNMAP

#include <poll.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/userfaultfd.h>


static void *ofi_uffd_handler(void *arg)
{
	struct uffd_msg msg;
	struct pollfd fds;
	int ret;

	fds.fd = uffd.fd;
	fds.events = POLLIN;
	for (;;) {
		ret = poll(&fds, 1, -1);
		if (ret != 1)
			break;

		fastlock_acquire(&uffd.monitor.lock);
		ret = read(uffd.fd, &msg, sizeof(msg));
		if (ret != sizeof(msg)) {
			fastlock_release(&uffd.monitor.lock);
			if (errno != EAGAIN)
				break;
			continue;
		}

		switch (msg.event) {
		case UFFD_EVENT_REMOVE:
		case UFFD_EVENT_UNMAP:
			ofi_monitor_notify(&uffd.monitor,
				(void *) (uintptr_t) msg.arg.remove.start,
				(size_t) (msg.arg.remove.end -
					  msg.arg.remove.start));
			break;
		case UFFD_EVENT_REMAP:
			ofi_monitor_notify(&uffd.monitor,
				(void *) (uintptr_t) msg.arg.remap.from,
				(size_t) msg.arg.remap.len);
			break;
		default:
			FI_WARN(&core_prov, FI_LOG_MR,
				"Unhandled uffd event %d\n", msg.event);
			break;
		}
		fastlock_release(&uffd.monitor.lock);
	}
	return NULL;
}

static int ofi_uffd_register(const void *addr, size_t len, size_t page_size)
{
	struct uffdio_register reg;
	int ret;

	reg.range.start = (uint64_t) (uintptr_t)
			  ofi_get_page_start(addr, page_size);
	reg.range.len = ofi_get_page_bytes(addr, len, page_size);
	reg.mode = UFFDIO_REGISTER_MODE_MISSING;
	ret = ioctl(uffd.fd, UFFDIO_REGISTER, &reg);
	if (ret < 0) {
		if (errno != EINVAL) {
			FI_WARN(&core_prov, FI_LOG_MR,
				"ioctl/uffd_unreg: %s\n", strerror(errno));
		}
		return -errno;
	}
	return 0;
}

static int ofi_uffd_subscribe(struct ofi_mem_monitor *monitor,
			      const void *addr, size_t len)
{
	int i;

	assert(monitor == &uffd.monitor);
	for (i = 0; i < num_page_sizes; i++) {
		if (!ofi_uffd_register(addr, len, page_sizes[i]))
			return 0;
	}
	return -FI_EFAULT;
}

static int ofi_uffd_unregister(const void *addr, size_t len, size_t page_size)
{
	struct uffdio_range range;
	int ret;

	range.start = (uint64_t) (uintptr_t)
		      ofi_get_page_start(addr, page_size);
	range.len = ofi_get_page_bytes(addr, len, page_size);
	ret = ioctl(uffd.fd, UFFDIO_UNREGISTER, &range);
	if (ret < 0) {
		if (errno != EINVAL) {
			FI_WARN(&core_prov, FI_LOG_MR,
				"ioctl/uffd_unreg: %s\n", strerror(errno));
		}
		return -errno;
	}
	return 0;
}

/* May be called from mr cache notifier callback */
static void ofi_uffd_unsubscribe(struct ofi_mem_monitor *monitor,
				 const void *addr, size_t len)
{
	int i;

	assert(monitor == &uffd.monitor);
	for (i = 0; i < num_page_sizes; i++) {
		if (!ofi_uffd_unregister(addr, len, page_sizes[i]))
			break;
	}
}

int ofi_uffd_init(void)
{
	struct uffdio_api api;
	int ret;

	uffd.monitor.subscribe = ofi_uffd_subscribe;
	uffd.monitor.unsubscribe = ofi_uffd_unsubscribe;

	if (!num_page_sizes)
		return -FI_ENODATA;

	uffd.fd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
	if (uffd.fd < 0) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"syscall/userfaultfd %s\n", strerror(errno));
		return -errno;
	}

	api.api = UFFD_API;
	api.features = UFFD_FEATURE_EVENT_UNMAP | UFFD_FEATURE_EVENT_REMOVE |
		       UFFD_FEATURE_EVENT_REMAP;
	ret = ioctl(uffd.fd, UFFDIO_API, &api);
	if (ret < 0) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"ioctl/uffdio: %s\n", strerror(errno));
		ret = -errno;
		goto closefd;
	}

	if (api.api != UFFD_API) {
		FI_WARN(&core_prov, FI_LOG_MR, "uffd features not supported\n");
		ret = -FI_ENOSYS;
		goto closefd;
	}

	ret = pthread_create(&uffd.thread, NULL, ofi_uffd_handler, &uffd);
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"failed to create handler thread %s\n", strerror(ret));
		ret = -ret;
		goto closefd;
	}
	return 0;

closefd:
	close(uffd.fd);
	return ret;
}

void ofi_uffd_cleanup(void)
{
	pthread_cancel(uffd.thread);
	pthread_join(uffd.thread, NULL);
	close(uffd.fd);
}

#else /* HAVE_UFFD_UNMAP */

int ofi_uffd_init(void)
{
	return -FI_ENOSYS;
}

void ofi_uffd_cleanup(void)
{
}

#endif /* HAVE_UFFD_UNMAP */
