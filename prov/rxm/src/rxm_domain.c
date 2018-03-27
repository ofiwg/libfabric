/*
 * Copyright (c) 2016 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ofi_util.h>
#include "rxm.h"

#define rxm_memory_hook_push(notifier)				\
{								\
	pthread_mutex_lock(&notifier->lock);			\
	ofi_set_mem_free_hook(notifier->prev_free_hook);	\
	ofi_set_mem_realloc_hook(notifier->prev_realloc_hook);	\

#define rxm_memory_hook_pop(notifier)					\
	ofi_set_mem_realloc_hook(rxm_mem_notifier_realloc_hook);	\
	ofi_set_mem_free_hook(rxm_mem_notifier_free_hook);		\
	pthread_mutex_unlock(&notifier->lock);				\
}

const uint64_t rxm_access_flags = FI_SEND | FI_RECV |
				  FI_READ | FI_WRITE |
				  FI_REMOTE_READ | FI_REMOTE_WRITE;

/* This is the memory notifier (singleton) for the RxM provider */
static struct rxm_mem_notifier *rxm_mem_notifier = NULL;

int rxm_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct util_cntr *cntr;

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	ret = ofi_cntr_init(&rxm_prov, domain, attr, cntr,
			    &ofi_cntr_progress, context);
	if (ret)
		goto free;

	*cntr_fid = &cntr->cntr_fid;
	return FI_SUCCESS;

free:
	free(cntr);
	return ret;
}

int rxm_av_create(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		  struct fid_av **av, void *context)
{
	return ip_av_create_flags(domain_fid, attr, av, context, OFI_AV_HASH);
}

static struct fi_ops_domain rxm_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = rxm_av_create,
	.cq_open = rxm_cq_open,
	.endpoint = rxm_endpoint,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = rxm_cntr_open,
	.poll_open = fi_poll_create,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};

void rxm_mem_notifier_free_hook(void *ptr, const void *caller)
{
	struct rxm_mem_ptr_entry *entry;
	OFI_UNUSED(caller);

	rxm_memory_hook_push(rxm_mem_notifier)

	free(ptr);

	if (!ptr)
		goto out;

	HASH_FIND(hh, rxm_mem_notifier->mem_ptrs_hash,
		  &ptr, sizeof(void *), entry);
	if (!entry)
		goto out;
	FI_DBG(&rxm_prov, FI_LOG_MR,
	       "Catch free hook for %p, entry - %p\n",
	       ptr, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);
	dlist_insert_tail(&entry->entry,
			  &rxm_mem_notifier->event_list);
out:
	rxm_memory_hook_pop(rxm_mem_notifier)
}

void *rxm_mem_notifier_realloc_hook(void *ptr, size_t size, const void *caller)
{
	struct rxm_mem_ptr_entry *entry;
	void *ret_ptr;
	OFI_UNUSED(caller);

	rxm_memory_hook_push(rxm_mem_notifier)
	
	ret_ptr = realloc(ptr, size);

	if (!ptr)
		goto out;

	HASH_FIND(hh, rxm_mem_notifier->mem_ptrs_hash,
		  &ptr, sizeof(void *), entry);
	if (!entry)
		goto out;
	FI_DBG(&rxm_prov, FI_LOG_MR,
	       "Catch realloc hook for %p, entry - %p\n",
	       ptr, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);
	dlist_insert_tail(&entry->entry,
			  &rxm_mem_notifier->event_list);
out:
	rxm_memory_hook_pop(rxm_mem_notifier)
	return ret_ptr;
}

static int rxm_mr_cache_entry_reg(struct ofi_mr_cache *cache,
				  struct ofi_mr_entry *entry)
{
	struct rxm_mr_cache_desc *mr_desc =
		(struct rxm_mr_cache_desc *)entry->data;
	struct rxm_domain *domain = mr_desc->domain =
		container_of(cache->domain, struct rxm_domain, util_domain);
	mr_desc->entry = entry;
	return fi_mr_reg(domain->msg_domain, entry->iov.iov_base,
			 entry->iov.iov_len, rxm_access_flags,
			 0, 0, 0, &mr_desc->mr, NULL);
}

static void rxm_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
				     struct ofi_mr_entry *entry)
{
	struct rxm_mr_cache_desc *mr_desc =
		(struct rxm_mr_cache_desc *)entry->data;
	if (fi_close(&mr_desc->mr->fid))
		FI_WARN(&rxm_prov, FI_LOG_MR,
			"Unable to close msg mr");
}

int rxm_monitor_subscribe(struct ofi_mem_monitor *notifier, void *addr,
			  size_t len, struct ofi_subscription *subscription)
{
	struct rxm_domain *domain =
		container_of(notifier, struct rxm_domain, monitor);
	struct rxm_mem_ptr_entry *entry;
	int ret = FI_SUCCESS;

	pthread_mutex_lock(&domain->notifier->lock);
	ofi_set_mem_free_hook(domain->notifier->prev_free_hook);
	ofi_set_mem_realloc_hook(domain->notifier->prev_realloc_hook);

	entry = util_buf_alloc(domain->notifier->mem_ptrs_ent_pool);
	if (OFI_UNLIKELY(!entry)) {
		ret = -FI_ENOMEM;
		goto fn;
	}

	entry->addr = addr;
	entry->subscription = subscription;
	dlist_init(&entry->entry);
	HASH_ADD(hh, domain->notifier->mem_ptrs_hash, addr, sizeof(void *), entry);

fn:
	ofi_set_mem_free_hook(rxm_mem_notifier_free_hook);
	ofi_set_mem_realloc_hook(rxm_mem_notifier_realloc_hook);
	pthread_mutex_unlock(&domain->notifier->lock);
	return ret;
}

void rxm_monitor_unsubscribe(struct ofi_mem_monitor *notifier, void *addr,
			     size_t len, struct ofi_subscription *subscription)
{
	struct rxm_domain *domain =
		container_of(notifier, struct rxm_domain, monitor);
	struct rxm_mem_ptr_entry *entry;

	pthread_mutex_lock(&domain->notifier->lock);
	ofi_set_mem_free_hook(domain->notifier->prev_free_hook);
	ofi_set_mem_realloc_hook(domain->notifier->prev_realloc_hook);

	HASH_FIND(hh, domain->notifier->mem_ptrs_hash, &addr, sizeof(void *), entry);
	assert(entry);

	HASH_DEL(domain->notifier->mem_ptrs_hash, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);

	util_buf_release(domain->notifier->mem_ptrs_ent_pool, entry);

	ofi_set_mem_realloc_hook(rxm_mem_notifier_realloc_hook);
	ofi_set_mem_free_hook(rxm_mem_notifier_free_hook);
	pthread_mutex_unlock(&domain->notifier->lock);
}

struct ofi_subscription *rxm_monitor_get_event(struct ofi_mem_monitor *notifier)
{
	struct rxm_domain *domain =
		container_of(notifier, struct rxm_domain, monitor);
	struct rxm_mem_ptr_entry *entry;

	pthread_mutex_lock(&domain->notifier->lock);
	if (!dlist_empty(&domain->notifier->event_list)) {
		dlist_pop_front(&domain->notifier->event_list,
				struct rxm_mem_ptr_entry,
				entry, entry);
		FI_DBG(&rxm_prov, FI_LOG_MR,
		       "Retrieve %p (entry %p) from event list\n",
		       entry->addr, entry);
		/* this is needed to protect against double insertions */
		dlist_init(&entry->entry);

		pthread_mutex_unlock(&domain->notifier->lock);
		return entry->subscription;
	} else {
		pthread_mutex_unlock(&domain->notifier->lock);
		return NULL;
	}
}

static void rxm_mem_notifier_finalize(struct rxm_mem_notifier *notifier)
{
	OFI_UNUSED(notifier);
#ifdef HAVE_GLIBC_MALLOC_HOOKS
	assert(rxm_mem_notifier && (notifier == rxm_mem_notifier));
	pthread_mutex_lock(&rxm_mem_notifier->lock);
	if (--rxm_mem_notifier->ref_cnt == 0) {
		ofi_set_mem_free_hook(rxm_mem_notifier->prev_free_hook);
		ofi_set_mem_realloc_hook(rxm_mem_notifier->prev_realloc_hook);
		util_buf_pool_destroy(rxm_mem_notifier->mem_ptrs_ent_pool);
		rxm_mem_notifier->prev_free_hook = NULL;
		rxm_mem_notifier->prev_realloc_hook = NULL;
		pthread_mutex_unlock(&rxm_mem_notifier->lock);
		pthread_mutex_destroy(&rxm_mem_notifier->lock);
		free(rxm_mem_notifier);
		rxm_mem_notifier = NULL;
		return;
	}
	pthread_mutex_unlock(&rxm_mem_notifier->lock);
#endif
}

static struct rxm_mem_notifier *rxm_mem_notifier_init(size_t mr_max_cached_cnt)
{
#ifdef HAVE_GLIBC_MALLOC_HOOKS
	int ret;
	pthread_mutexattr_t mutex_attr;
	if (rxm_mem_notifier) {
		/* already initialized */
		rxm_mem_notifier->ref_cnt++;
		goto fn;
	}
	rxm_mem_notifier = calloc(1, sizeof(*rxm_mem_notifier));
	if (!rxm_mem_notifier)
		goto fn;

	ret = util_buf_pool_create(&rxm_mem_notifier->mem_ptrs_ent_pool,
				   sizeof(struct rxm_mem_ptr_entry),
				   16, 0, mr_max_cached_cnt);
	if (ret)
		goto err1;

	pthread_mutexattr_init(&mutex_attr);
	pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
	if (pthread_mutex_init(&rxm_mem_notifier->lock, &mutex_attr))
		goto err2;
	pthread_mutexattr_destroy(&mutex_attr);

	dlist_init(&rxm_mem_notifier->event_list);

	pthread_mutex_lock(&rxm_mem_notifier->lock);
	rxm_mem_notifier->prev_free_hook = ofi_get_mem_free_hook();
	rxm_mem_notifier->prev_realloc_hook = ofi_get_mem_realloc_hook();
	ofi_set_mem_free_hook(rxm_mem_notifier_free_hook);
	ofi_set_mem_realloc_hook(rxm_mem_notifier_realloc_hook);
	rxm_mem_notifier->ref_cnt++;
	pthread_mutex_unlock(&rxm_mem_notifier->lock);
fn:
	return rxm_mem_notifier;

err2:
	util_buf_pool_destroy(rxm_mem_notifier->mem_ptrs_ent_pool);
err1:
	free(rxm_mem_notifier);
	rxm_mem_notifier = NULL;
#endif
	OFI_UNUSED(mr_max_cached_cnt);
	return NULL;
}

static int rxm_domain_close(fid_t fid)
{
	struct rxm_domain *rxm_domain;
	int ret;

	rxm_domain = container_of(fid, struct rxm_domain, util_domain.domain_fid.fid);
	
	if (rxm_domain->mr_cache_enable) {
		ofi_mr_cache_cleanup(&rxm_domain->cache);
		ofi_monitor_cleanup(&rxm_domain->monitor);
		rxm_mem_notifier_finalize(rxm_domain->notifier);
	}

	ret = fi_close(&rxm_domain->msg_domain->fid);
	if (ret)
		return ret;

	ret = ofi_domain_close(&rxm_domain->util_domain);
	if (ret)
		return ret;

	free(rxm_domain);
	return 0;
}

static struct fi_ops rxm_domain_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_domain_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int rxm_mr_close(fid_t fid)
{
	struct rxm_mr *rxm_mr;
	int ret;

	rxm_mr = container_of(fid, struct rxm_mr, mr_fid.fid);
	ret = fi_close(&rxm_mr->msg_mr->fid);
	if (ret)
		FI_WARN(&rxm_prov, FI_LOG_DOMAIN, "Unable to close MSG MR\n");
	free(rxm_mr);
	return ret;
}

static struct fi_ops rxm_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int rxm_mr_reg(struct fid *domain_fid, const void *buf, size_t len,
	   uint64_t access, uint64_t offset, uint64_t requested_key,
	   uint64_t flags, struct fid_mr **mr, void *context)
{
	struct rxm_domain *rxm_domain;
	struct rxm_mr *rxm_mr;
	int ret;

	rxm_domain = container_of(domain_fid, struct rxm_domain,
			util_domain.domain_fid.fid);

	rxm_mr = calloc(1, sizeof(*rxm_mr));
	if (!rxm_mr)
		return -FI_ENOMEM;

	/* Additional flags to use RMA read for large message transfers */
	access |= FI_READ | FI_REMOTE_READ;

	if (rxm_domain->mr_local)
		access |= FI_WRITE;

	ret = fi_mr_reg(rxm_domain->msg_domain, buf, len, access, offset, requested_key,
			flags, &rxm_mr->msg_mr, context);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_DOMAIN, "Unable to register MSG MR\n");
		goto err;
	}

	rxm_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	rxm_mr->mr_fid.fid.context = context;
	rxm_mr->mr_fid.fid.ops = &rxm_mr_ops;
	/* Store msg_mr as rxm_mr descriptor so that we can get its key when
	 * the app passes msg_mr as the descriptor in fi_send and friends.
	 * The key would be used in large message transfer protocol and RMA. */
	rxm_mr->mr_fid.mem_desc = rxm_mr->msg_mr;
	rxm_mr->mr_fid.key = fi_mr_key(rxm_mr->msg_mr);
	*mr = &rxm_mr->mr_fid;

	return 0;
err:
	free(rxm_mr);
	return ret;
}

static struct fi_ops_mr rxm_domain_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = rxm_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr,
};

int rxm_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **domain, void *context)
{
	int ret;
	struct rxm_domain *rxm_domain;
	struct rxm_fabric *rxm_fabric;
	struct fi_info *msg_info;

	rxm_domain = calloc(1, sizeof(*rxm_domain));
	if (!rxm_domain)
		return -FI_ENOMEM;

	rxm_fabric = container_of(fabric, struct rxm_fabric, util_fabric.fabric_fid);

	ret = ofi_get_core_info(fabric->api_version, NULL, NULL, 0, &rxm_util_prov,
				info, rxm_info_to_core, &msg_info);
	if (ret)
		goto err1;

	/* Force core provider to supply MR key */
	if (FI_VERSION_LT(fabric->api_version, FI_VERSION(1, 5)) ||
	    (msg_info->domain_attr->mr_mode & (FI_MR_BASIC | FI_MR_SCALABLE)))
		msg_info->domain_attr->mr_mode = FI_MR_BASIC;
	else
		msg_info->domain_attr->mr_mode |= FI_MR_PROV_KEY;

	ret = fi_domain(rxm_fabric->msg_fabric, msg_info,
			&rxm_domain->msg_domain, context);
	if (ret)
		goto err2;

	ret = ofi_domain_init(fabric, info, &rxm_domain->util_domain, context);
	if (ret) {
		goto err3;
	}

	*domain = &rxm_domain->util_domain.domain_fid;
	(*domain)->fid.ops = &rxm_domain_fi_ops;
	/* Replace MR ops set by ofi_domain_init() */
	(*domain)->mr = &rxm_domain_mr_ops;
	(*domain)->ops = &rxm_domain_ops;

	rxm_domain->mr_local = OFI_CHECK_MR_LOCAL(msg_info) &&
				!OFI_CHECK_MR_LOCAL(info);

	if (fi_param_get_int(&rxm_prov, "mr_max_cached_cnt",
			     &rxm_domain->mr_max_cached_cnt))
		rxm_domain->mr_max_cached_cnt = 4096;

	if (fi_param_get_size_t(&rxm_prov, "mr_max_cached_size",
				&rxm_domain->mr_max_cached_size))
		rxm_domain->mr_max_cached_size = ULONG_MAX;

	if (!fi_param_get_int(&rxm_prov, "mr_cache_enable",
			     (int *)&rxm_domain->mr_cache_enable) &&
	    rxm_domain->mr_cache_enable) {
		rxm_domain->notifier =
			rxm_mem_notifier_init(rxm_domain->mr_max_cached_cnt);
		rxm_domain->monitor.subscribe = rxm_monitor_subscribe;
		rxm_domain->monitor.unsubscribe = rxm_monitor_unsubscribe;
		rxm_domain->monitor.get_event = rxm_monitor_get_event;
		ofi_monitor_init(&rxm_domain->monitor);

		rxm_domain->cache.max_cached_cnt = rxm_domain->mr_max_cached_cnt;
		rxm_domain->cache.max_cached_size = rxm_domain->mr_max_cached_size;
		rxm_domain->cache.entry_data_size = sizeof(struct rxm_mr_cache_desc);
		rxm_domain->cache.add_region = rxm_mr_cache_entry_reg;
		rxm_domain->cache.delete_region = rxm_mr_cache_entry_dereg;
		ret = ofi_mr_cache_init(&rxm_domain->util_domain, &rxm_domain->monitor,
					&rxm_domain->cache);
		if (ret)
			goto err4;
	} else {
		rxm_domain->mr_cache_enable = 0;
	}

	fi_freeinfo(msg_info);
	return 0;
err4:
	(void) ofi_domain_close(&rxm_domain->util_domain);
err3:
	fi_close(&rxm_domain->msg_domain->fid);
err2:
	fi_freeinfo(msg_info);
err1:
	free(rxm_domain);
	return ret;
}
