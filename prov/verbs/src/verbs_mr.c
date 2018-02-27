/*
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
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

#include <ofi_util.h>
#include "fi_verbs.h"
#include "verbs_rdm.h"

#define FI_IBV_DEFINE_MR_REG_OPS(type)							\
											\
static int										\
fi_ibv_mr ## type ## regv(struct fid *fid, const struct iovec *iov,			\
			  size_t count, uint64_t access, uint64_t offset,		\
			  uint64_t requested_key, uint64_t flags,			\
			  struct fid_mr **mr, void *context)				\
{											\
	const struct fi_mr_attr attr = {						\
		.mr_iov		= iov,							\
		.iov_count	= count,						\
		.access		= access,						\
		.offset		= offset,						\
		.requested_key	= requested_key,					\
		.context	= context,						\
	};										\
	return fi_ibv_mr ## type ## regattr(fid, &attr, flags, mr);			\
}											\
											\
static int										\
fi_ibv_mr ## type ## reg(struct fid *fid, const void *buf, size_t len,			\
			 uint64_t access, uint64_t offset, uint64_t requested_key,	\
			 uint64_t flags, struct fid_mr **mr, void *context)		\
{											\
	const struct iovec iov = {							\
		.iov_base	= (void *)buf,						\
		.iov_len	= len,							\
	};										\
	return fi_ibv_mr ## type ## regv(fid, &iov, 1, access, offset,			\
					 requested_key, flags, mr, context);		\
}											\
											\
static struct fi_ops_mr fi_ibv_domain_mr ##type## ops = {				\
	.size = sizeof(struct fi_ops_mr),						\
	.reg = fi_ibv_mr ## type ## reg,						\
	.regv = fi_ibv_mr ## type ## regv,						\
	.regattr = fi_ibv_mr ## type ## regattr,					\
};											\
											\
struct fi_ibv_mr_internal_ops fi_ibv_mr_internal ##type## ops = {			\
	.fi_ops = &fi_ibv_domain_mr ##type## ops,					\
	.internal_mr_reg = fi_ibv_mr_internal ##type## reg,				\
	.internal_mr_dereg = fi_ibv_mr_internal ##type## dereg,				\
};

static inline struct ibv_mr *
fi_ibv_mr_reg_ibv_mr(struct fi_ibv_domain *domain, void *buf,
		     size_t len, int fi_ibv_access)
{	
#if defined HAVE_VERBS_EXP_H
	struct ibv_exp_reg_mr_in in = {
		.pd		= domain->pd,
		.addr		= buf,
		.length		= len,
		.exp_access 	= fi_ibv_access,
		.comp_mask	= 0,
	};
	if (domain->use_odp)
		in.exp_access |= IBV_EXP_ACCESS_RELAXED |
				 IBV_EXP_ACCESS_ON_DEMAND;
	return ibv_exp_reg_mr(&in);
#else /* HAVE_VERBS_EXP_H */
	return ibv_reg_mr(domain->pd, buf, len, fi_ibv_access);
#endif /* HAVE_VERBS_EXP_H */
}

static inline
int fi_ibv_mr_dereg_ibv_mr(struct ibv_mr *mr)
{
	return -ibv_dereg_mr(mr);
}

static int fi_ibv_mr_close(fid_t fid)
{
	struct fi_ibv_mem_desc *mr;
	int ret;

	mr = container_of(fid, struct fi_ibv_mem_desc, mr_fid.fid);
	ret = fi_ibv_mr_dereg_ibv_mr(mr->mr);
	if (!ret)
		free(mr);
	return ret;
}

static struct fi_ops fi_ibv_mr_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_mr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static inline
int fi_ibv_mr_reg_common(struct fi_ibv_mem_desc *md, int fi_ibv_access,
			 const void *buf, size_t len, void *context)
{
	/* ops should be set in special functions */
	md->mr_fid.fid.fclass = FI_CLASS_MR;
	md->mr_fid.fid.context = context;

	md->mr = fi_ibv_mr_reg_ibv_mr(md->domain, (void *)buf, len, fi_ibv_access);
	if (!md->mr)
		return -errno;

	md->mr_fid.mem_desc = (void *)(uintptr_t)md->mr->lkey;
	md->mr_fid.key = md->mr->rkey;

	if (md->domain->eq_flags & FI_REG_MR) {
		struct fi_eq_entry entry = {
			.fid = &md->mr_fid.fid,
			.context = context,
		};
		if (md->domain->eq)
			fi_ibv_eq_write_event(md->domain->eq, FI_MR_COMPLETE,
				 	      &entry, sizeof(entry));
		else if (md->domain->util_domain.eq)
			 /* This branch is taken for the verbs/DGRAM */
			fi_eq_write(&md->domain->util_domain.eq->eq_fid,
				    FI_MR_COMPLETE, &entry, sizeof(entry), 0);
	}
	return FI_SUCCESS;
}

static inline
int fi_ibv_mr_regattr_check_args(struct fid *fid,
				 const struct fi_mr_attr *attr,
				 uint64_t flags)
{
	if (OFI_UNLIKELY(flags))
		return -FI_EBADFLAGS;
	if (OFI_UNLIKELY(fid->fclass != FI_CLASS_DOMAIN))
		return -FI_EINVAL;
	if (OFI_UNLIKELY(attr->iov_count > VERBS_MR_IOV_LIMIT)) {
		VERBS_WARN(FI_LOG_FABRIC,
			   "iov count > %d not supported\n",
			   VERBS_MR_IOV_LIMIT);
		return -FI_EINVAL;
	}
	return FI_SUCCESS;
}

static inline int
fi_ibv_mr_ofi2ibv_access(uint64_t ofi_access, struct fi_ibv_domain *domain)
{
	int ibv_access = 0;

	/* Enable local write access by default for FI_EP_RDM which hides local
	 * registration requirements. This allows to avoid buffering or double
	 * registration */
	if (!(domain->info->caps & FI_LOCAL_MR) &&
	    !(domain->info->domain_attr->mr_mode & FI_MR_LOCAL))
		ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	/* Local read access to an MR is enabled by default in verbs */
	if (ofi_access & FI_RECV)
		ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	/* iWARP spec requires Remote Write access for an MR that is used
	 * as a data sink for a Remote Read */
	if (ofi_access & FI_READ) {
		ibv_access |= IBV_ACCESS_LOCAL_WRITE;
		if (domain->verbs->device->transport_type == IBV_TRANSPORT_IWARP)
			ibv_access |= IBV_ACCESS_REMOTE_WRITE;
	}

	if (ofi_access & FI_WRITE)
		ibv_access |= IBV_ACCESS_LOCAL_WRITE;

	if (ofi_access & FI_REMOTE_READ)
		ibv_access |= IBV_ACCESS_REMOTE_READ;

	/* Verbs requires Local Write access too for Remote Write access */
	if (ofi_access & FI_REMOTE_WRITE)
		ibv_access |= IBV_ACCESS_LOCAL_WRITE |
			      IBV_ACCESS_REMOTE_WRITE |
			      IBV_ACCESS_REMOTE_ATOMIC;

	return ibv_access;
}

static inline struct fi_ibv_mem_desc *
fi_ibv_mr_common_cache_reg(struct fi_ibv_domain *domain,
			 struct fi_mr_attr *attr)
{
	struct fi_ibv_mem_desc *md;
	struct ofi_mr_entry *entry;
	int ret;

	ret = ofi_mr_cache_search(&domain->cache, attr, &entry);
	if (OFI_UNLIKELY(ret))
		return NULL;

	md = (struct fi_ibv_mem_desc *)entry->data;
	md->entry = entry;

	return md;
}

static inline
void fi_ibv_common_cache_dereg(struct fi_ibv_mem_desc *md)
{
	ofi_mr_cache_delete(&md->domain->cache, md->entry);
}

static inline
int fi_ibv_mr_internal_reg(struct fi_ibv_domain *domain, void *buf,
			   size_t len, uint64_t access,
			   struct fi_ibv_mem_desc *md)
{
	md->domain = domain;
	md->len = len;
	md->mr = fi_ibv_mr_reg_ibv_mr(domain, buf, len,
				      fi_ibv_mr_ofi2ibv_access(access,
							       domain));
	if (OFI_UNLIKELY(!md->mr))
		return -errno;
	return FI_SUCCESS;
}

static inline
int fi_ibv_mr_internal_dereg(struct fi_ibv_mem_desc *md)
{
	int ret = fi_ibv_mr_dereg_ibv_mr(md->mr);
	md->mr = NULL;
	return ret;
}

int fi_ibv_rdm_alloc_and_reg(struct fi_ibv_rdm_ep *ep,
			     void **buf, size_t size,
			     struct fi_ibv_mem_desc *md)
{
	if (!*buf) {
		if (ofi_memalign((void **)buf,
				 FI_IBV_BUF_ALIGNMENT, size))
			return -FI_ENOMEM;
	}

	memset(*buf, 0, size);
	return fi_ibv_mr_internal_reg(ep->domain, *buf, size,
				      FI_WRITE | FI_REMOTE_WRITE, md);
}

ssize_t fi_ibv_rdm_dereg_and_free(struct fi_ibv_mem_desc *md,
				  char **buff)
{
	ssize_t ret = FI_SUCCESS;
	ret = fi_ibv_mr_internal_dereg(md);
	if (ret)
		VERBS_WARN(FI_LOG_AV,
			   "Unable to deregister MR, ret = %"PRId64"\n", ret);

	if (buff && *buff) {
		ofi_freealign(*buff);
		*buff = NULL;
	}

	return ret;
}

static inline
int fi_ibv_mr_internal_cache_reg(struct fi_ibv_domain *domain, void *buf,
				 size_t len, uint64_t access,
				 struct fi_ibv_mem_desc *md)
{
	const struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};
	struct fi_mr_attr attr = {
		.mr_iov		= &iov,
		.iov_count	= 1,
		.access		= access,
	};
	struct fi_ibv_mem_desc *mdesc =
		fi_ibv_mr_common_cache_reg(domain, &attr);
	if (OFI_UNLIKELY(!mdesc))
		return -FI_EAVAIL;
	*md = *mdesc;
	md->len = len;
	return FI_SUCCESS;
}

static inline
int fi_ibv_mr_internal_cache_dereg(struct fi_ibv_mem_desc *md)
{
	fi_ibv_common_cache_dereg(md);
	md->mr = NULL;
	return FI_SUCCESS;
}

static int fi_ibv_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			     uint64_t flags, struct fid_mr **mr)
{
	struct fi_ibv_mem_desc *md;
	int ret;

	ret = fi_ibv_mr_regattr_check_args(fid, attr, flags);
	if (OFI_UNLIKELY(ret))
		return ret;

	md = calloc(1, sizeof(*md));
	if (OFI_UNLIKELY(!md))
		return -FI_ENOMEM;

	md->domain = container_of(fid, struct fi_ibv_domain,
				  util_domain.domain_fid.fid);
	md->mr_fid.fid.ops = &fi_ibv_mr_ops;

	ret = fi_ibv_mr_reg_common(md, fi_ibv_mr_ofi2ibv_access(attr->access,
								md->domain),
				   attr->mr_iov[0].iov_base,
				   attr->mr_iov[0].iov_len, attr->context);
	if (OFI_UNLIKELY(ret))
		goto err;

	*mr = &md->mr_fid;
	return FI_SUCCESS;
err:
	free(md);
	return ret;
}

FI_IBV_DEFINE_MR_REG_OPS(_)

static int fi_ibv_mr_cache_close(fid_t fid)
{
	struct fi_ibv_mem_desc *md =
		container_of(fid, struct fi_ibv_mem_desc, mr_fid.fid);
	
	fi_ibv_common_cache_dereg(md);

	return FI_SUCCESS;
}

static struct fi_ops fi_ibv_mr_cache_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_mr_cache_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_monitor_subscribe(struct ofi_mem_monitor *notifier, void *addr,
			     size_t len, struct ofi_subscription *subscription)
{
	struct fi_ibv_domain *domain =
		container_of(notifier, struct fi_ibv_domain, monitor);
	struct fi_ibv_mem_ptr_entry *entry;
	int ret = FI_SUCCESS;

	pthread_mutex_lock(&domain->notifier->lock);
	fi_ibv_mem_notifier_set_free_hook(domain->notifier->prev_free_hook);
	fi_ibv_mem_notifier_set_realloc_hook(domain->notifier->prev_realloc_hook);

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
	fi_ibv_mem_notifier_set_free_hook(fi_ibv_mem_notifier_free_hook);
	fi_ibv_mem_notifier_set_realloc_hook(fi_ibv_mem_notifier_realloc_hook);
	pthread_mutex_unlock(&domain->notifier->lock);
	return ret;
}

void fi_ibv_monitor_unsubscribe(struct ofi_mem_monitor *notifier, void *addr,
				size_t len, struct ofi_subscription *subscription)
{
	struct fi_ibv_domain *domain =
		container_of(notifier, struct fi_ibv_domain, monitor);
	struct fi_ibv_mem_ptr_entry *entry;

	pthread_mutex_lock(&domain->notifier->lock);
	fi_ibv_mem_notifier_set_free_hook(domain->notifier->prev_free_hook);
	fi_ibv_mem_notifier_set_realloc_hook(domain->notifier->prev_realloc_hook);

	HASH_FIND(hh, domain->notifier->mem_ptrs_hash, &addr, sizeof(void *), entry);
	assert(entry);

	HASH_DEL(domain->notifier->mem_ptrs_hash, entry);

	if (!dlist_empty(&entry->entry))
		dlist_remove_init(&entry->entry);

	util_buf_release(domain->notifier->mem_ptrs_ent_pool, entry);

	fi_ibv_mem_notifier_set_realloc_hook(fi_ibv_mem_notifier_realloc_hook);
	fi_ibv_mem_notifier_set_free_hook(fi_ibv_mem_notifier_free_hook);
	pthread_mutex_unlock(&domain->notifier->lock);
}

struct ofi_subscription *
fi_ibv_monitor_get_event(struct ofi_mem_monitor *notifier)
{
	struct fi_ibv_domain *domain =
		container_of(notifier, struct fi_ibv_domain, monitor);
	struct fi_ibv_mem_ptr_entry *entry;

	pthread_mutex_lock(&domain->notifier->lock);
	if (!dlist_empty(&domain->notifier->event_list)) {
		dlist_pop_front(&domain->notifier->event_list,
				struct fi_ibv_mem_ptr_entry,
				entry, entry);
		VERBS_DBG(FI_LOG_MR,
			  "Retrieve %p (entry %p) from event list\n",
			  entry->addr, entry);
		/* needed to protect against double insertions */
		dlist_init(&entry->entry);

		pthread_mutex_unlock(&domain->notifier->lock);
		return entry->subscription;
	} else {
		pthread_mutex_unlock(&domain->notifier->lock);
		return NULL;
	}
}

int fi_ibv_mr_cache_entry_reg(struct ofi_mr_cache *cache,
			      struct ofi_mr_entry *entry)
{
	int fi_ibv_access = IBV_ACCESS_LOCAL_WRITE |
			    IBV_ACCESS_REMOTE_WRITE |
			    IBV_ACCESS_REMOTE_ATOMIC |
			    IBV_ACCESS_REMOTE_READ;
	struct fi_ibv_mem_desc *md = (struct fi_ibv_mem_desc *)entry->data;
	md->domain = container_of(cache->domain, struct fi_ibv_domain, util_domain);
	md->mr_fid.fid.ops = &fi_ibv_mr_cache_ops;
	return fi_ibv_mr_reg_common(md, fi_ibv_access, entry->iov.iov_base,
				    entry->iov.iov_len, NULL);
}

void fi_ibv_mr_cache_entry_dereg(struct ofi_mr_cache *cache,
				 struct ofi_mr_entry *entry)
{
	struct fi_ibv_mem_desc *md = (struct fi_ibv_mem_desc *)entry->data;
	(void)fi_ibv_mr_dereg_ibv_mr(md->mr);
}

static int fi_ibv_mr_cache_regattr(struct fid *fid, const struct fi_mr_attr *attr,
				   uint64_t flags, struct fid_mr **mr)
{
	struct fi_ibv_domain *domain;
	struct fi_ibv_mem_desc *md;
	int ret;

	ret = fi_ibv_mr_regattr_check_args(fid, attr, flags);
	if (OFI_UNLIKELY(ret))
		return ret;

	domain = container_of(fid, struct fi_ibv_domain,
			      util_domain.domain_fid.fid);

	md = fi_ibv_mr_common_cache_reg(domain, (struct fi_mr_attr *)attr);
	if (OFI_UNLIKELY(!md))
		return -FI_EAVAIL;
	*mr = &md->mr_fid;
	return FI_SUCCESS;
}

FI_IBV_DEFINE_MR_REG_OPS(_cache_)
