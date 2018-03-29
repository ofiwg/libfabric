/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include "config.h"

#include "ep_rdm/verbs_rdm.h"
#include "ep_dgram/verbs_dgram.h"

#include "fi_verbs.h"
#include <malloc.h>

/* This is the memory notifier for the entire verbs provider */
static struct fi_ibv_mem_notifier *fi_ibv_mem_notifier = NULL;

static int fi_ibv_domain_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_eq *eq;
	struct fi_ibv_dgram_eq *dgram_eq;

	domain = container_of(fid, struct fi_ibv_domain,
			      util_domain.domain_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		switch (domain->ep_type) {
		case FI_EP_MSG:
		case FI_EP_RDM:
			eq = container_of(bfid, struct fi_ibv_eq, eq_fid);
			domain->eq = eq;
			domain->eq_flags = flags;
			break;
		case FI_EP_DGRAM:
			dgram_eq = container_of(bfid, struct fi_ibv_dgram_eq,
						util_eq.eq_fid);
			if (!dgram_eq)
				return -FI_EINVAL;
			return ofi_domain_bind_eq(&domain->util_domain,
						  &dgram_eq->util_eq);
		default:
			/* Shouldn't go here */
			assert(0);
			return -FI_EINVAL;
		}
	default:
		return -EINVAL;
	}

	return 0;
}

static void fi_ibv_rdm_cm_set_thread_affinity(void)
{
	if (fi_ibv_gl_data.rdm.cm_thread_affinity == NULL)
		return;

	if (ofi_set_thread_affinity(fi_ibv_gl_data.rdm.cm_thread_affinity) == -FI_ENOSYS)
		VERBS_WARN(FI_LOG_DOMAIN,
			   "FI_VERBS_RDM_CM_THREAD_AFFINITY is not supported on OS X\n");
}

static void *fi_ibv_rdm_cm_progress_thread(void *dom)
{
	struct fi_ibv_domain *domain =
		(struct fi_ibv_domain *)dom;
	struct slist_entry *item, *prev;

	fi_ibv_rdm_cm_set_thread_affinity();

	while (domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running) {
		struct fi_ibv_rdm_ep *ep = NULL;
		slist_foreach(&domain->ep_list, item, prev) {
			(void) prev;
			ep = container_of(item, struct fi_ibv_rdm_ep,
					  list_entry);
			if (fi_ibv_rdm_cm_progress(ep)) {
				VERBS_INFO(FI_LOG_EP_DATA,
					   "fi_ibv_rdm_cm_progress error\n");
				abort();
			}
		}
		usleep(domain->rdm_cm->cm_progress_timeout);
	}
	return NULL;
}

void fi_ibv_mem_notifier_free_hook(void *ptr, const void *caller)
{
	struct fi_ibv_mem_ptr_entry *entry;
	OFI_UNUSED(caller);

	FI_IBV_MEMORY_HOOK_BEGIN(fi_ibv_mem_notifier)

	free(ptr);

	if (!ptr)
		goto out;

	HASH_FIND(hh, fi_ibv_mem_notifier->mem_ptrs_hash, &ptr, sizeof(void *), entry);
	if (!entry)
		goto out;
	VERBS_DBG(FI_LOG_MR, "Catch free hook for %p, entry - %p\n", ptr, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);
	dlist_insert_tail(&entry->entry,
			  &fi_ibv_mem_notifier->event_list);
out:
	FI_IBV_MEMORY_HOOK_END(fi_ibv_mem_notifier)
}

void *fi_ibv_mem_notifier_realloc_hook(void *ptr, size_t size, const void *caller)
{
	struct fi_ibv_mem_ptr_entry *entry;
	void *ret_ptr;
	OFI_UNUSED(caller);

	FI_IBV_MEMORY_HOOK_BEGIN(fi_ibv_mem_notifier)
	
	ret_ptr = realloc(ptr, size);

	if (!ptr)
		goto out;

	HASH_FIND(hh, fi_ibv_mem_notifier->mem_ptrs_hash, &ptr, sizeof(void *), entry);
	if (!entry)
		goto out;
	VERBS_DBG(FI_LOG_MR, "Catch realloc hook for %p, entry - %p\n", ptr, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);
	dlist_insert_tail(&entry->entry,
			  &fi_ibv_mem_notifier->event_list);
out:
	FI_IBV_MEMORY_HOOK_END(fi_ibv_mem_notifier)
	return ret_ptr;
}

static void fi_ibv_mem_notifier_finalize(struct fi_ibv_mem_notifier *notifier)
{
#ifdef HAVE_GLIBC_MALLOC_HOOKS
	OFI_UNUSED(notifier);
	assert(fi_ibv_mem_notifier && (notifier == fi_ibv_mem_notifier));
	pthread_mutex_lock(&fi_ibv_mem_notifier->lock);
	if (--fi_ibv_mem_notifier->ref_cnt == 0) {
		fi_ibv_mem_notifier_set_free_hook(fi_ibv_mem_notifier->prev_free_hook);
		fi_ibv_mem_notifier_set_realloc_hook(fi_ibv_mem_notifier->prev_realloc_hook);
		util_buf_pool_destroy(fi_ibv_mem_notifier->mem_ptrs_ent_pool);
		fi_ibv_mem_notifier->prev_free_hook = NULL;
		fi_ibv_mem_notifier->prev_realloc_hook = NULL;
		pthread_mutex_unlock(&fi_ibv_mem_notifier->lock);
		pthread_mutex_destroy(&fi_ibv_mem_notifier->lock);
		free(fi_ibv_mem_notifier);
		fi_ibv_mem_notifier = NULL;
		return;
	}
	pthread_mutex_unlock(&fi_ibv_mem_notifier->lock);
#endif
}

static struct fi_ibv_mem_notifier *fi_ibv_mem_notifier_init(void)
{
#ifdef HAVE_GLIBC_MALLOC_HOOKS
	int ret;
	pthread_mutexattr_t mutex_attr;
	if (fi_ibv_mem_notifier) {
		/* already initialized */
		fi_ibv_mem_notifier->ref_cnt++;
		goto fn;
	}
	fi_ibv_mem_notifier = calloc(1, sizeof(*fi_ibv_mem_notifier));
	if (!fi_ibv_mem_notifier)
		goto fn;

	ret = util_buf_pool_create(&fi_ibv_mem_notifier->mem_ptrs_ent_pool,
				   sizeof(struct fi_ibv_mem_ptr_entry),
				   FI_IBV_MEM_ALIGNMENT, 0,
				   fi_ibv_gl_data.mr_max_cached_cnt);
	if (ret)
		goto err1;

	pthread_mutexattr_init(&mutex_attr);
	pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
	if (pthread_mutex_init(&fi_ibv_mem_notifier->lock, &mutex_attr))
		goto err2;
	pthread_mutexattr_destroy(&mutex_attr);

	dlist_init(&fi_ibv_mem_notifier->event_list);

	pthread_mutex_lock(&fi_ibv_mem_notifier->lock);
	fi_ibv_mem_notifier->prev_free_hook = fi_ibv_mem_notifier_get_free_hook();
	fi_ibv_mem_notifier->prev_realloc_hook = fi_ibv_mem_notifier_get_realloc_hook();
	fi_ibv_mem_notifier_set_free_hook(fi_ibv_mem_notifier_free_hook);
	fi_ibv_mem_notifier_set_realloc_hook(fi_ibv_mem_notifier_realloc_hook);
	fi_ibv_mem_notifier->ref_cnt++;
	pthread_mutex_unlock(&fi_ibv_mem_notifier->lock);
fn:
	return fi_ibv_mem_notifier;

err2:
	util_buf_pool_destroy(fi_ibv_mem_notifier->mem_ptrs_ent_pool);
err1:
	free(fi_ibv_mem_notifier);
	fi_ibv_mem_notifier = NULL;
	return NULL;
#else
	return NULL;
#endif
}

static int fi_ibv_domain_close(fid_t fid)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_fabric *fab;
	struct fi_ibv_rdm_av_entry *av_entry = NULL;
	struct slist_entry *item;
	void *status = NULL;
	int ret;

	domain = container_of(fid, struct fi_ibv_domain,
			      util_domain.domain_fid.fid);

	switch (domain->ep_type) {
	case FI_EP_RDM:
		domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running = 0;
		pthread_join(domain->rdm_cm->cm_progress_thread, &status);
		pthread_mutex_destroy(&domain->rdm_cm->cm_lock);

		for (item = slist_remove_head(
				&domain->rdm_cm->av_removed_entry_head);
	     	     item;
	     	     item = slist_remove_head(
				&domain->rdm_cm->av_removed_entry_head)) {
			av_entry = container_of(item,
						struct fi_ibv_rdm_av_entry,
						removed_next);
			fi_ibv_rdm_overall_conn_cleanup(av_entry);
			ofi_freealign(av_entry);
		}
		rdma_destroy_ep(domain->rdm_cm->listener);
		free(domain->rdm_cm);
		break;
	case FI_EP_DGRAM:
		fab = container_of(&domain->util_domain.fabric->fabric_fid,
				   struct fi_ibv_fabric,
				   util_fabric.fabric_fid.fid);
		/* Even if it's invoked not for the first time
		 * (e.g. multiple domains per fabric), it's safe
		 */
		if (fi_ibv_gl_data.dgram.use_name_server)
			ofi_ns_stop_server(&fab->name_server);
		break;
	case FI_EP_MSG:
		break;
	default:
		/* Never should go here */
		assert(0);
		return -FI_EINVAL;
	}

	if (fi_ibv_gl_data.mr_cache_enable) {
		ofi_mr_cache_cleanup(&domain->cache);
		ofi_monitor_cleanup(&domain->monitor);
		fi_ibv_mem_notifier_finalize(domain->notifier);
	}

	if (domain->pd) {
		ret = ibv_dealloc_pd(domain->pd);
		if (ret)
			return -ret;
		domain->pd = NULL;
	}

	ret = ofi_domain_close(&domain->util_domain);
	if (ret)
		return ret;

	fi_freeinfo(domain->info);
	free(domain);
	return 0;
}

static int fi_ibv_open_device_by_name(struct fi_ibv_domain *domain, const char *name)
{
	struct ibv_context **dev_list;
	int i, ret = -FI_ENODEV;

	if (!name)
		return -FI_EINVAL;

	dev_list = rdma_get_devices(NULL);
	if (!dev_list)
		return -errno;

	for (i = 0; dev_list[i] && ret; i++) {
		const char *rdma_name = ibv_get_device_name(dev_list[i]->device);
		switch (domain->ep_type) {
		case FI_EP_MSG:
			ret = strcmp(name, rdma_name);
			break;
		case FI_EP_RDM:
			ret = strncmp(name, rdma_name,
				      strlen(name) - strlen(verbs_rdm_domain.suffix));
			break;
		case FI_EP_DGRAM:
			ret = strncmp(name, rdma_name,
				      strlen(name) - strlen(verbs_dgram_domain.suffix));
			break;
		default:
			VERBS_WARN(FI_LOG_DOMAIN,
				   "Unsupported EP type - %d\n", domain->ep_type);
			/* Never should go here */
			assert(0);
			ret = -FI_EINVAL;
			break;
		}

		if (!ret)
			domain->verbs = dev_list[i];
	}
	rdma_free_devices(dev_list);
	return ret;
}

static struct fi_ops fi_ibv_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_domain_close,
	.bind = fi_ibv_domain_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain fi_ibv_msg_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_no_av_open,
	.cq_open = fi_ibv_cq_open,
	.endpoint = fi_ibv_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_ibv_srq_context,
	.query_atomic = fi_ibv_query_atomic,
};

static struct fi_ops_domain fi_ibv_rdm_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_rdm_av_open,
	.cq_open = fi_ibv_rdm_cq_open,
	.endpoint = fi_ibv_rdm_open_ep,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_rbv_rdm_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_ibv_query_atomic,
};

static struct fi_ops_domain fi_ibv_dgram_domain_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = fi_ibv_dgram_av_open,
	.cq_open = fi_ibv_dgram_cq_open,
	.endpoint = fi_ibv_dgram_endpoint_open,
	.scalable_ep = fi_no_scalable_ep,
	.cntr_open = fi_ibv_dgram_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};

static void fi_ibv_domain_process_exp(struct fi_ibv_domain *domain)
{
#ifdef HAVE_VERBS_EXP_H
	struct ibv_exp_device_attr exp_attr= {
		.comp_mask = IBV_EXP_DEVICE_ATTR_ODP |
			     IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS,
	};
	domain->use_odp = (!ibv_exp_query_device(domain->verbs, &exp_attr) &&
			   exp_attr.exp_device_cap_flags & IBV_EXP_DEVICE_ODP);
#else /* HAVE_VERBS_EXP_H */
	domain->use_odp = 0;
#endif /* HAVE_VERBS_EXP_H */
	if (!domain->use_odp && fi_ibv_gl_data.use_odp) {
		VERBS_WARN(FI_LOG_CORE,
			   "ODP is not supported on this configuration, ignore \n");
		return;
	}
	domain->use_odp = fi_ibv_gl_data.use_odp;
}

static int
fi_ibv_domain(struct fid_fabric *fabric, struct fi_info *info,
	      struct fid_domain **domain, void *context)
{
	struct fi_ibv_domain *_domain;
	struct fi_ibv_fabric *fab;
	const struct fi_info *fi;
	void *status = NULL;
	pthread_mutexattr_t mutex_attr;
	int ret;

	fab = container_of(fabric, struct fi_ibv_fabric,
			   util_fabric.fabric_fid);

	fi = fi_ibv_get_verbs_info(fi_ibv_util_prov.info,
				   info->domain_attr->name);
	if (!fi)
		return -FI_EINVAL;

	ret = ofi_check_domain_attr(&fi_ibv_prov, fabric->api_version,
				    fi->domain_attr, info);
	if (ret)
		return ret;

	_domain = calloc(1, sizeof *_domain);
	if (!_domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(fabric, info, &_domain->util_domain, context);
	if (ret)
		goto err1;

	_domain->info = fi_dupinfo(info);
	if (!_domain->info)
		goto err2;

	_domain->ep_type = FI_IBV_EP_TYPE(info);
	ret = fi_ibv_open_device_by_name(_domain, info->domain_attr->name);
	if (ret)
		goto err3;
	_domain->pd = ibv_alloc_pd(_domain->verbs);
	if (!_domain->pd) {
		ret = -errno;
		goto err3;
	}

	_domain->util_domain.domain_fid.fid.fclass = FI_CLASS_DOMAIN;
	_domain->util_domain.domain_fid.fid.context = context;
	_domain->util_domain.domain_fid.fid.ops = &fi_ibv_fid_ops;

	fi_ibv_domain_process_exp(_domain);

	if (fi_ibv_gl_data.mr_cache_enable) {
		_domain->notifier = fi_ibv_mem_notifier_init();
		_domain->monitor.subscribe = fi_ibv_monitor_subscribe;
		_domain->monitor.unsubscribe = fi_ibv_monitor_unsubscribe;
		_domain->monitor.get_event = fi_ibv_monitor_get_event;
		ofi_monitor_init(&_domain->monitor);

		_domain->cache.max_cached_cnt = fi_ibv_gl_data.mr_max_cached_cnt;
		_domain->cache.max_cached_size = fi_ibv_gl_data.mr_max_cached_size;
		_domain->cache.merge_regions = fi_ibv_gl_data.mr_cache_merge_regions;
		_domain->cache.entry_data_size = sizeof(struct fi_ibv_mem_desc);
		_domain->cache.add_region = fi_ibv_mr_cache_entry_reg;
		_domain->cache.delete_region = fi_ibv_mr_cache_entry_dereg;
		ret = ofi_mr_cache_init(&_domain->util_domain, &_domain->monitor,
					&_domain->cache);
		if (ret)
			goto err4;
		_domain->util_domain.domain_fid.mr = fi_ibv_mr_internal_cache_ops.fi_ops;
		_domain->internal_mr_reg = fi_ibv_mr_internal_cache_ops.internal_mr_reg;
		_domain->internal_mr_dereg = fi_ibv_mr_internal_cache_ops.internal_mr_dereg;
	} else {
		_domain->util_domain.domain_fid.mr = fi_ibv_mr_internal_ops.fi_ops;
		_domain->internal_mr_reg = fi_ibv_mr_internal_ops.internal_mr_reg;
		_domain->internal_mr_dereg = fi_ibv_mr_internal_ops.internal_mr_dereg;
	}

	switch (_domain->ep_type) {
	case FI_EP_RDM:
		_domain->rdm_cm = calloc(1, sizeof(*_domain->rdm_cm));
		if (!_domain->rdm_cm) {
			ret = -FI_ENOMEM;
			goto err5;
		}
		_domain->rdm_cm->cm_progress_timeout =
			fi_ibv_gl_data.rdm.thread_timeout;
		slist_init(&_domain->rdm_cm->av_removed_entry_head);

		pthread_mutexattr_init(&mutex_attr);
		pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
		pthread_mutex_init(&_domain->rdm_cm->cm_lock, &mutex_attr);
		pthread_mutexattr_destroy(&mutex_attr);
		_domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running = 1;
		ret = pthread_create(&_domain->rdm_cm->cm_progress_thread,
				     NULL, &fi_ibv_rdm_cm_progress_thread,
				     (void *)_domain);
		if (ret) {
			VERBS_INFO(FI_LOG_DOMAIN,
				   "Failed to launch CM progress thread, "
				   "err :%d\n", ret);
			ret = -FI_EOTHER;
			goto err6;
		}
		_domain->util_domain.domain_fid.ops = &fi_ibv_rdm_domain_ops;

		_domain->rdm_cm->ec = rdma_create_event_channel();
		if (!_domain->rdm_cm->ec) {
			VERBS_INFO(FI_LOG_DOMAIN,
				   "Failed to create listener event channel: %s\n",
				   strerror(errno));
			ret = -FI_EOTHER;
			goto err7;
		}

		if (fi_fd_nonblock(_domain->rdm_cm->ec->fd) != 0) {
			VERBS_INFO_ERRNO(FI_LOG_DOMAIN, "fcntl", errno);
			ret = -FI_EOTHER;
			goto err8;
		}

		if (rdma_create_id(_domain->rdm_cm->ec,
				   &_domain->rdm_cm->listener, NULL, RDMA_PS_TCP)) {
			VERBS_INFO(FI_LOG_DOMAIN, "Failed to create cm listener: %s\n",
				   strerror(errno));
			ret = -FI_EOTHER;
			goto err8;
		}
		_domain->rdm_cm->is_bound = 0;
		break;
	case FI_EP_DGRAM:
		if (fi_ibv_gl_data.dgram.use_name_server) {
			/* Even if it's invoked not for the first time
			 * (e.g. multiple domains per fabric), it's safe
			 */
			fab->name_server.port =
					fi_ibv_gl_data.dgram.name_server_port;
			fab->name_server.name_len = sizeof(struct ofi_ib_ud_ep_name);
			fab->name_server.service_len = sizeof(int);
			fab->name_server.service_cmp = fi_ibv_dgram_ns_service_cmp;
			fab->name_server.is_service_wildcard =
					fi_ibv_dgram_ns_is_service_wildcard;

			ofi_ns_init(&fab->name_server);
			ofi_ns_start_server(&fab->name_server);
		}
		_domain->util_domain.domain_fid.ops = &fi_ibv_dgram_domain_ops;
		break;
	case FI_EP_MSG:
		_domain->util_domain.domain_fid.ops = &fi_ibv_msg_domain_ops;
		break;
	default:
		VERBS_INFO(FI_LOG_DOMAIN, "Ivalid EP type is provided, "
			   "EP type :%d\n", _domain->ep_type);
		ret = -FI_EINVAL;
		goto err5;
	}

	*domain = &_domain->util_domain.domain_fid;
	return FI_SUCCESS;
/* Only verbs/RDM should be able to go through err[5-7] */
err8:
	rdma_destroy_event_channel(_domain->rdm_cm->ec);
err7:
	_domain->rdm_cm->fi_ibv_rdm_tagged_cm_progress_running = 0;
	pthread_join(_domain->rdm_cm->cm_progress_thread, &status);
err6:
	pthread_mutex_destroy(&_domain->rdm_cm->cm_lock);
	free(_domain->rdm_cm);
err5:
	if (fi_ibv_gl_data.mr_cache_enable)
		ofi_mr_cache_cleanup(&_domain->cache);
err4:
	if (fi_ibv_gl_data.mr_cache_enable)
		ofi_monitor_cleanup(&_domain->monitor);
	if (ibv_dealloc_pd(_domain->pd))
		VERBS_INFO_ERRNO(FI_LOG_DOMAIN,
				 "ibv_dealloc_pd", errno);
err3:
	fi_freeinfo(_domain->info);
err2:
	if (ofi_domain_close(&_domain->util_domain))
		VERBS_INFO(FI_LOG_DOMAIN,
			   "ofi_domain_close fails");
err1:
	free(_domain);
	return ret;
}

static int fi_ibv_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	struct fi_ibv_cq *cq;
	int ret, i;

	for (i = 0; i < count; i++) {
		switch (fids[i]->fclass) {
		case FI_CLASS_CQ:
			cq = container_of(fids[i], struct fi_ibv_cq, cq_fid.fid);
			ret = cq->trywait(fids[i]);
			if (ret)
				return ret;
			break;
		case FI_CLASS_EQ:
			/* We are always ready to wait on an EQ since
			 * rdmacm EQ is based on an fd */
			continue;
		case FI_CLASS_CNTR:
		case FI_CLASS_WAIT:
			return -FI_ENOSYS;
		default:
			return -FI_EINVAL;
		}

	}
	return FI_SUCCESS;
}

static int fi_ibv_fabric_close(fid_t fid)
{
	struct fi_ibv_fabric *fab;
	int ret;

	fab = container_of(fid, struct fi_ibv_fabric, util_fabric.fabric_fid.fid);
	ret = ofi_fabric_close(&fab->util_fabric);
	if (ret)
		return ret;
	free(fab);

	return 0;
}

static struct fi_ops fi_ibv_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_fabric_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_fabric fi_ibv_ops_fabric = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = fi_ibv_domain,
	.passive_ep = fi_ibv_passive_ep,
	.eq_open = fi_ibv_eq_open,
	.wait_open = fi_no_wait_open,
	.trywait = fi_ibv_trywait
};

int fi_ibv_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		  void *context)
{
	struct fi_ibv_fabric *fab;
	const struct fi_info *cur, *info = fi_ibv_util_prov.info;
	int ret = FI_SUCCESS;

	fab = calloc(1, sizeof(*fab));
	if (!fab)
		return -FI_ENOMEM;

	for (cur = info; cur; cur = info->next) {
		ret = ofi_fabric_init(&fi_ibv_prov, cur->fabric_attr, attr,
				      &fab->util_fabric, context);
		if (ret != -FI_ENODATA)
			break;
	}
	if (ret) {
		free(fab);
		return ret;
	}

	fab->info = cur;

	*fabric = &fab->util_fabric.fabric_fid;
	(*fabric)->fid.fclass = FI_CLASS_FABRIC;
	(*fabric)->fid.ops = &fi_ibv_fi_ops;
	(*fabric)->ops = &fi_ibv_ops_fabric;

	return 0;
}
