/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * (C) Copyright 2022 Hewlett Packard Enterprise Development LP
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

#include "ofi_mr.h"

#if HAVE_KDREG2_MONITOR

#include "ofi_hmem.h"

#define EVICTOR_THREAD_ATTR NULL
#define INFINITE_TIMEOUT -1

static int kdreg2_monitor_subscribe(struct ofi_mem_monitor *monitor,
				    const void *addr,
				    size_t len,
				    union ofi_mr_hmem_info *hmem_info)
{
	struct ofi_kdreg2 *kdreg2 = container_of(monitor,
						 struct ofi_kdreg2,
						 monitor);

	struct kdreg2_ioctl_monitor  ioctl_monitor = {
		.addr = addr,
		.length = len,
		.cookie = (kdreg2_cookie_t)
			  ofi_atomic_inc64(&kdreg2->next_cookie),
	};

	int ret = ioctl(kdreg2->fd, KDREG2_IOCTL_MONITOR, &ioctl_monitor);

	if (ret)
		return ret;

	hmem_info->kdreg2.cookie            = ioctl_monitor.cookie;
	hmem_info->kdreg2.monitoring_params = ioctl_monitor.monitoring_params;

	return 0;
}

static void kdreg2_monitor_unsubscribe(struct ofi_mem_monitor *monitor,
				       const void *addr, size_t len,
				       union ofi_mr_hmem_info *hmem_info)
{
	struct ofi_kdreg2 *kdreg2 = container_of(monitor,
						 struct ofi_kdreg2,
						 monitor);

	struct kdreg2_ioctl_unmonitor  ioctl_unmonitor = {
		.cookie            = hmem_info->kdreg2.cookie,
		.monitoring_params = hmem_info->kdreg2.monitoring_params,
	};

	ioctl(kdreg2->fd, KDREG2_IOCTL_UNMONITOR, &ioctl_unmonitor);
}

static bool kdreg2_monitor_valid(struct ofi_mem_monitor *monitor,
				 const void *addr, size_t len,
				 union ofi_mr_hmem_info *hmem_info)
{
	struct ofi_kdreg2 *kdreg2 = container_of(monitor,
						 struct ofi_kdreg2,
						 monitor);

	return !kdreg2_mapping_changed(kdreg2->status_data,
				       &hmem_info->kdreg2.monitoring_params);
}

static int kdreg2_read_evictions(struct ofi_kdreg2 *kdreg2)
{
	struct kdreg2_event  event;

	while (kdreg2_read_counter(&kdreg2->status_data->pending_events) > 0) {

		ssize_t bytes = read(kdreg2->fd, &event, sizeof(event));

		if (bytes < 0) {
			int err = errno;

			/* EINTR means we caught a signal */
			if (err == EINTR)
				continue;

			/* Nothing left */
			if ((err == EAGAIN) ||
			    (err == EWOULDBLOCK))
				return 0;

			/* all other errors */
			return -err;
		}

		switch (event.type) {
		case KDREG2_EVENT_MAPPING_CHANGE:

			pthread_rwlock_rdlock(&mm_list_rwlock);
			pthread_mutex_lock(&mm_lock);

			ofi_monitor_notify(&kdreg2->monitor,
					   event.u.mapping_change.addr,
					   event.u.mapping_change.len);

			pthread_mutex_unlock(&mm_lock);
			pthread_rwlock_unlock(&mm_list_rwlock);

			break;

		default:

			return -ENOMSG;
		}
	}

	return 0;
}

static void kdreg2_evictor_cleanup(void *arg)
{
	/* It's possible that we kill the evictor while it holds
	 * one or both of the mm locks.
	 */

	pthread_mutex_unlock(&mm_lock);
	pthread_rwlock_unlock(&mm_list_rwlock);
}

static void *kdreg2_evictor(void *arg)
{
	struct ofi_kdreg2 *kdreg2 = (struct ofi_kdreg2 *) arg;
	int old_state;
	int ret = pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, &old_state);

	if (ret)
		goto error_ret;

	pthread_cleanup_push(kdreg2_evictor_cleanup, kdreg2);

	struct pollfd pollfd = {
		.fd = kdreg2->fd,
		.events = POLLIN,
	};

	while (1) {

		/* wait until there are events to read */

		int n = poll(&pollfd, 1, INFINITE_TIMEOUT);

		if (n == 0)           /* timeout(?) */
			continue;

		if (n < 0) {
			switch (errno) {
			case EINTR:   /* interrupted */
				continue;
			default:
				ret = errno;
				goto error_ret;
			}
		}

		ret = kdreg2_read_evictions(kdreg2);
		if (ret)
			goto error_ret;
	}

	/* Due to the way pthread_cleanup_push is implemented as a macro, we
	 * need to have a matching pop or it won't compile.  Even if it's
	 * unreachable.  Really.
	 */

	pthread_cleanup_pop(0);

error_ret:

	return (void *) (intptr_t) ret;
}


static int kdreg2_monitor_start(struct ofi_mem_monitor *monitor)
{
	struct ofi_kdreg2          *kdreg2 = container_of(monitor,
							  struct ofi_kdreg2,
							  monitor);
	int   ret = 0;
	struct kdreg2_config_data  config_data = {
		.max_regions = cache_params.max_cnt,
	};

	/* see if already started */
	if (kdreg2->fd >= 0)
		return 0;

	ofi_atomic_initialize64(&kdreg2->next_cookie, 0);

	kdreg2->fd = open(KDREG2_DEVICE_NAME, O_RDWR | O_NONBLOCK);

	if (kdreg2->fd <= 0) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed to open %s for monitor kdreg2: %d.\n",
			KDREG2_DEVICE_NAME, errno);
		return -errno;
	}

	ret = ioctl(kdreg2->fd, KDREG2_IOCTL_CONFIG_DATA, &config_data);
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed to set configuration data for kdreg2 monitor: %d.\n",
			errno);
		ret = -errno;
		goto exit_close;
	}

	kdreg2->status_data = config_data.status_data;

	ret = pthread_create(&kdreg2->thread,
			     EVICTOR_THREAD_ATTR,
			     kdreg2_evictor,
			     kdreg2);

	if (ret) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed to start thread for kdreg2 monitor: %d.\n",
			ret);
		goto exit_close;
	}

	return 0;

exit_close:

	close(kdreg2->fd);
	kdreg2->fd = -1;

	return ret;
}

static void kdreg2_monitor_stop(struct ofi_mem_monitor *monitor)
{
	struct ofi_kdreg2 *kdreg2 = container_of(monitor,
						 struct ofi_kdreg2,
						 monitor);

	if (kdreg2->fd < 0)
		return;

	pthread_cancel(kdreg2->thread);
	pthread_join(kdreg2->thread, NULL);

	close(kdreg2->fd);
	kdreg2->fd = -1;
	kdreg2->status_data = NULL;
}

#else /* !HAVE_KDREG2_MONITOR */

static int kdreg2_monitor_subscribe(struct ofi_mem_monitor *monitor,
				    const void *addr,
				    size_t len,
				    union ofi_mr_hmem_info *hmem_info)
{
	return -FI_ENOSYS;
}

static void kdreg2_monitor_unsubscribe(struct ofi_mem_monitor *monitor,
				       const void *addr, size_t len,
				       union ofi_mr_hmem_info *hmem_info)
{
}

static bool kdreg2_monitor_valid(struct ofi_mem_monitor *monitor,
				 const void *addr, size_t len,
				 union ofi_mr_hmem_info *hmem_info)
{
	return false;
}

static int kdreg2_monitor_start(struct ofi_mem_monitor *monitor)
{
	return -FI_ENOSYS;
}

void kdreg2_monitor_stop(struct ofi_mem_monitor *monitor)
{
	/* no-op */
}

#endif /* HAVE_KDREG2_MONITOR */

static struct ofi_kdreg2 kdreg2_mm = {
	.monitor.iface       = FI_HMEM_SYSTEM,
	.monitor.init        = ofi_monitor_init,
	.monitor.cleanup     = ofi_monitor_cleanup,
	.monitor.start       = kdreg2_monitor_start,
	.monitor.stop        = kdreg2_monitor_stop,
	.monitor.subscribe   = kdreg2_monitor_subscribe,
	.monitor.unsubscribe = kdreg2_monitor_unsubscribe,
	.monitor.valid       = kdreg2_monitor_valid,
	.fd                  = -1,
	.status_data         = NULL,
};

struct ofi_mem_monitor *kdreg2_monitor = &kdreg2_mm.monitor;
