/*
 * Copyright (c) 2020 Hewlett Packard Enterprise Development LP.
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

#ifdef HAVE_ROCR

#include "ofi_tree.h"
#include "ofi_iov.h"

#include <hsa/hsa_ext_amd.h>

struct ofi_rocr_mm_entry {
	struct iovec iov;
	struct dlist_entry entry;
	struct ofi_rbnode *node;
};

struct ofi_rocr_mm {
	struct ofi_rbmap entries;
	struct dlist_entry entry_list;
	struct ofi_mem_monitor mm;
};

static void ofi_rocr_mm_unsubscribe(struct ofi_mem_monitor *notifier,
				    const void *addr, size_t len);
static int ofi_rocr_mm_subscribe(struct ofi_mem_monitor *notifier,
				 const void *addr, size_t len);

static struct ofi_rocr_mm rocr_mm = {
	.mm = {
		.subscribe = ofi_rocr_mm_subscribe,
		.unsubscribe = ofi_rocr_mm_unsubscribe,
	},
};

struct ofi_mem_monitor *rocr_monitor = &rocr_mm.mm;

static int ofi_rocr_rbmap_compare(struct ofi_rbmap *map, void *key, void *data)
{
	struct ofi_rocr_mm_entry *entry = data;
	struct iovec *iov = key;

	if (ofi_iov_left(&entry->iov, iov))
		return -1;
	else if (ofi_iov_right(&entry->iov, iov))
		return 1;

	/* If this fails, the ROCr memory monitor failed to have a single ROCr
	 * memory monitor entry per user allocated ROCr buffer.
	 */
	assert(ofi_iov_within(iov, &entry->iov));

	return 0;
}

static struct ofi_rocr_mm_entry *ofi_rocr_find_mm_entry(const void *addr,
							size_t len)
{
	struct ofi_rbnode *node;
	struct iovec iov = {
		.iov_base = (void *)addr,
		.iov_len = len,
	};

	node = ofi_rbmap_find(&rocr_mm.entries, (void *)&iov);
	if (node)
		return node->data;

	return NULL;
}

static void ofi_rocr_mm_dealloc_cb(void *addr, void *data)
{
	struct ofi_rocr_mm_entry *entry = data;

	pthread_mutex_lock(&rocr_monitor->lock);
	ofi_rocr_mm_unsubscribe(rocr_monitor, entry->iov.iov_base,
				entry->iov.iov_len);
	pthread_mutex_unlock(&rocr_monitor->lock);
}

/* Must hold monitor lock. */
static void ofi_rocr_mm_entry_free(struct ofi_rocr_mm_entry *entry)
{
	const void *addr = entry->iov.iov_base;
	size_t len = entry->iov.iov_len;
	hsa_status_t hsa_ret;

	FI_DBG(&core_prov, FI_LOG_MR,
	       "ROCr buffer address %p length %lu monitor entry freed\n", addr,
	       len);

	/* Two return codes are expected. HSA_STATUS_SUCCESS is returned if the
	 * deallocation callback was not triggered and the entry is freed.
	 * HSA_STATUS_ERROR_INVALID_ARGUMENT is returned if the deallocation
	 * callback was triggered and the entry is freed. Any other return code
	 * puts the monitor in an unknown state.
	 */
	hsa_ret = hsa_amd_deregister_deallocation_callback(entry->iov.iov_base,
							   ofi_rocr_mm_dealloc_cb);
	assert(hsa_ret == HSA_STATUS_SUCCESS ||
	       hsa_ret == HSA_STATUS_ERROR_INVALID_ARGUMENT);

	ofi_rbmap_delete(&rocr_mm.entries, entry->node);
	dlist_remove(&entry->entry);
	free(entry);

	/* Have to notify all subscribed caches that the entire ROCr region is
	 * no longer being monitored. Caches with allocated MRs within this
	 * region cannot be reused.
	 */
	ofi_monitor_notify(rocr_monitor, addr, len);
}

/* Must hold monitor lock. */
static int ofi_rocr_mm_entry_alloc(const void *addr, size_t len,
				   struct ofi_rocr_mm_entry **entry)
{
	hsa_amd_pointer_info_t hsa_info;
	hsa_status_t hsa_ret;
	int ret;

	*entry = malloc(sizeof(**entry));
	if (!*entry) {
		ret = -FI_ENOMEM;
		goto err;
	}

	/* Get pointer information to determine full size of memory allocation.
	 */
	hsa_ret = hsa_amd_pointer_info((void *)addr, &hsa_info, NULL, NULL,
				       NULL);
	if (hsa_ret != HSA_STATUS_SUCCESS) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hsa_amd_pointer_info: ret=%d\n",
			hsa_ret);

		ret = -FI_EIO;
		goto err_free_entry;
	}

	/* Requested memory entry length must be within the allocated length. If
	 * not, user may be trying to monitor unallocated memory.
	 */
	if ((uintptr_t)addr + len >
	    (uintptr_t)hsa_info.agentBaseAddress + hsa_info.sizeInBytes) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Request ROCr buffer length %lu exceeded the actual "
			"length %lu. Cannot monitor this memory region.\n", len,
			hsa_info.sizeInBytes);
		ret = -FI_EFAULT;
		goto err_free_entry;
	}

	FI_DBG(&core_prov, FI_LOG_MR,
	       "ROCr buffer address %p length %lu expanded to %p length %lu\n",
	       addr, len, hsa_info.agentBaseAddress, hsa_info.sizeInBytes);

	(*entry)->iov.iov_base = hsa_info.agentBaseAddress;
	(*entry)->iov.iov_len = hsa_info.sizeInBytes;

	ret = ofi_rbmap_insert(&rocr_mm.entries, (void *)&(*entry)->iov,
			       (void *)*entry, &(*entry)->node);
	if (ret) {
		FI_WARN(&core_prov, FI_LOG_MR,
			"Failed to insert into RB tree: %s\n", strerror(ret));
		goto err_free_entry;
	}

	/* Register a deallocation callback for this allocation. */
	hsa_ret = hsa_amd_register_deallocation_callback(hsa_info.agentBaseAddress,
							 ofi_rocr_mm_dealloc_cb,
							 (void *)*entry);
	if (hsa_ret != HSA_STATUS_SUCCESS) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to perform hsa_amd_register_deallocation_callback: ret=%d\n",
			hsa_ret);

		ret = -FI_EIO;
		goto err_rbmap_delete_entry;
	}

	dlist_insert_tail(&(*entry)->entry, &rocr_mm.entry_list);

	FI_DBG(&core_prov, FI_LOG_MR,
	       "ROCr buffer address %p length %lu monitor entry allocated\n",
	       hsa_info.agentBaseAddress, hsa_info.sizeInBytes);

	return FI_SUCCESS;

err_rbmap_delete_entry:
	ofi_rbmap_delete(&rocr_mm.entries, (*entry)->node);
err_free_entry:
	free(*entry);
err:
	*entry = NULL;
	return ret;
}

/* Must hold monitor lock. */
static void ofi_rocr_mm_unsubscribe(struct ofi_mem_monitor *notifier,
				    const void *addr, size_t len)
{
	struct ofi_rocr_mm_entry *entry;

	/* ROCr monitor entries are freed only during unsubscribed. */
	entry = ofi_rocr_find_mm_entry(addr, len);
	if (entry) {
		ofi_rocr_mm_entry_free(entry);

		FI_DBG(&core_prov, FI_LOG_MR,
		       "ROCr buffer address %p length %lu unsubscribed\n",
		       addr, len);
	}
}

/* Must hold monitor lock. */
static int ofi_rocr_mm_subscribe(struct ofi_mem_monitor *notifier,
				 const void *addr, size_t len)
{
	struct ofi_rocr_mm_entry *entry;
	int ret;

	/* Prefer already allocated ROCr monitor entries over allocating a new
	 * entry. If entry is found, it is still valid.
	 */
	entry = ofi_rocr_find_mm_entry(addr, len);
	if (entry) {
		FI_DBG(&core_prov, FI_LOG_MR,
		       "Reusing monitored ROCr buffer address %p length %lu\n",
		       entry->iov.iov_base, entry->iov.iov_len);
		ret = FI_SUCCESS;
		goto out;
	}

	ret = ofi_rocr_mm_entry_alloc(addr, len, &entry);

out:
	if (ret == FI_SUCCESS)
		FI_DBG(&core_prov, FI_LOG_MR,
		       "ROCr buffer address %p length %lu subscribed\n",
		       addr, len);
	else
		FI_DBG(&core_prov, FI_LOG_MR,
		       "ROCr buffer address %p length %lu subscribe failed: %d\n",
		       addr, len, ret);


	return ret;
}

void ofi_rocr_monitor_init(void)
{
	int enabled = 1;

	fi_param_define(NULL, "mr_rocr_cache_monitor_enabled", FI_PARAM_BOOL,
			"Enable or disable the ROCr cache memory monitor. "
			"Monitor is enabled by default.");

	fi_param_get_bool(NULL, "mr_rocr_cache_monitor_enabled", &enabled);

	if (enabled) {
		pthread_mutex_init(&rocr_mm.mm.lock, NULL);
		dlist_init(&rocr_mm.mm.list);
		dlist_init(&rocr_mm.entry_list);
		ofi_rbmap_init(&rocr_mm.entries, ofi_rocr_rbmap_compare);
	} else {
		rocr_monitor = NULL;
	}
}

void ofi_rocr_monitor_cleanup(void)
{
	struct ofi_rocr_mm_entry *cur;
	struct dlist_entry *tmp;

	if (rocr_monitor) {
		dlist_foreach_container_safe(&rocr_mm.entry_list,
					     struct ofi_rocr_mm_entry,
					     cur, entry, tmp)
			ofi_rocr_mm_entry_free(cur);
	}
}

void ofi_rocr_monitor_teardown(void)
{
	if (rocr_monitor) {
		assert(dlist_empty(&rocr_mm.entry_list));
		assert(dlist_empty(&rocr_mm.mm.list));
		pthread_mutex_destroy(&rocr_mm.mm.lock);
	}
}

#else

struct ofi_mem_monitor *rocr_monitor = NULL;

void ofi_rocr_monitor_init(void)
{
}

void ofi_rocr_monitor_cleanup(void)
{
}

void ofi_rocr_monitor_teardown(void)
{
}

#endif /* HAVE_ROCR */
