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
#include <stdbool.h>

#include <ofi.h>
#include <ofi_atom.h>
#include <ofi_lock.h>
#include <ofi_list.h>
#include <rbtree.h>


#define OFI_MR_BASIC_MAP (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)

/* FI_LOCAL_MR is valid in pre-libfaric-1.5 and can be valid in
 * post-libfabric-1.5 */
static inline int ofi_mr_local(const struct fi_info *info)
{
	if (!info)
		return 0;

	if (!info->domain_attr)
		goto check_local_mr;

	if (info->domain_attr->mr_mode & FI_MR_LOCAL)
		return 1;

	if (info->domain_attr->mr_mode & ~(FI_MR_BASIC | FI_MR_SCALABLE))
		return 0;

check_local_mr:
	return (info->mode & FI_LOCAL_MR) ? 1 : 0;
}

#define OFI_MR_MODE_RMA_TARGET (FI_MR_RAW | FI_MR_VIRT_ADDR |			\
				 FI_MR_PROV_KEY | FI_MR_RMA_EVENT)

/* If the app sets FI_MR_LOCAL, we ignore FI_LOCAL_MR.  So, if the
 * app doesn't set FI_MR_LOCAL, we need to check for FI_LOCAL_MR.
 * The provider is assumed only to set FI_MR_LOCAL correctly.
 */
static inline uint64_t ofi_mr_get_prov_mode(uint32_t version,
					    const struct fi_info *user_info,
					    const struct fi_info *prov_info)
{
	if (FI_VERSION_LT(version, FI_VERSION(1, 5)) ||
	    (user_info->domain_attr &&
	     !(user_info->domain_attr->mr_mode & FI_MR_LOCAL))) {
		return (prov_info->domain_attr->mr_mode & FI_MR_LOCAL) ?
			prov_info->mode | FI_LOCAL_MR : prov_info->mode;
	} else {
		return prov_info->mode;
	}
}


/*
 * Memory notifier - Report memory mapping changes to address ranges
 */

struct ofi_subscription;
struct ofi_notification_queue;

struct ofi_mem_monitor {
	ofi_atomic32_t			refcnt;

	int (*subscribe)(struct ofi_mem_monitor *notifier,
			 struct ofi_subscription *subscription);
	void (*unsubscribe)(struct ofi_mem_monitor *notifier,
			    struct ofi_subscription *subscription);
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
	struct iovec			iov;
};

void ofi_monitor_init(struct ofi_mem_monitor *monitor);
void ofi_monitor_cleanup(struct ofi_mem_monitor *monitor);
void ofi_monitor_add_queue(struct ofi_mem_monitor *monitor,
			   struct ofi_notification_queue *nq);
void ofi_monitor_del_queue(struct ofi_notification_queue *nq);

int ofi_monitor_subscribe(struct ofi_notification_queue *nq,
			  void *addr, size_t len,
			  struct ofi_subscription *subscription);
void ofi_monitor_unsubscribe(struct ofi_subscription *subscription);
struct ofi_subscription *ofi_monitor_get_event(struct ofi_notification_queue *nq);
static inline void
ofi_monitor_add_event_to_nq(struct ofi_subscription *subscription)
{
	FI_DBG(&core_prov, FI_LOG_MR,
	       "Add event to NQ, context=%p, addr=%p, len=%"PRIu64" nq=%p\n",
	       subscription, subscription->iov.iov_base,
	       subscription->iov.iov_len, subscription->nq);
	fastlock_acquire(&subscription->nq->lock);
	if (dlist_empty(&subscription->entry))
		dlist_insert_tail(&subscription->entry,
				  &subscription->nq->list);
	fastlock_release(&subscription->nq->lock);
}

/*
 * MR map
 */

struct ofi_mr_map {
	const struct fi_provider *prov;
	void			*rbtree;
	uint64_t		key;
	enum fi_mr_mode		mode;
};

int ofi_mr_map_init(const struct fi_provider *in_prov, int mode,
		    struct ofi_mr_map *map);
void ofi_mr_map_close(struct ofi_mr_map *map);

int ofi_mr_map_insert(struct ofi_mr_map *map,
		      const struct fi_mr_attr *attr,
		      uint64_t *key, void *context);
int ofi_mr_map_remove(struct ofi_mr_map *map, uint64_t key);
void *ofi_mr_map_get(struct ofi_mr_map *map,  uint64_t key);

int ofi_mr_map_verify(struct ofi_mr_map *map, uintptr_t *io_addr,
		      size_t len, uint64_t key, uint64_t access,
		      void **context);

struct ofi_mr {
	struct fid_mr mr_fid;
	struct util_domain *domain;
	uint64_t key;
	uint64_t flags;
};

int ofi_mr_close(struct fid *fid);
int ofi_mr_regattr(struct fid *fid, const struct fi_mr_attr *attr,
		   uint64_t flags, struct fid_mr **mr_fid);
int ofi_mr_regv(struct fid *fid, const struct iovec *iov,
	        size_t count, uint64_t access, uint64_t offset,
		uint64_t requested_key, uint64_t flags,
		struct fid_mr **mr_fid, void *context);
int ofi_mr_reg(struct fid *fid, const void *buf, size_t len,
	       uint64_t access, uint64_t offset, uint64_t requested_key,
	       uint64_t flags, struct fid_mr **mr_fid, void *context);
int ofi_mr_verify(struct ofi_mr_map *map, ssize_t len,
		  uintptr_t *addr, uint64_t key, uint64_t access);

/*
 * Memory registration cache
 */

struct ofi_mr_entry {
	struct iovec			iov;
	unsigned int			cached:1;
	unsigned int			subscribed:1;
	int				use_cnt;
	struct dlist_entry		lru_entry;
	struct ofi_subscription		subscription;
	uint8_t				data[];
};

struct ofi_mr_storage;

typedef void (*ofi_mr_destroy_t)(struct ofi_mr_storage *storage);
typedef struct ofi_mr_entry *(*ofi_mr_find_t)(struct ofi_mr_storage *storage,
					      const struct iovec *key);
typedef int (*ofi_mr_insert_t)(struct ofi_mr_storage *storage,
			       struct iovec *key,
			       struct ofi_mr_entry *entry);
typedef int (*ofi_mr_erase_t)(struct ofi_mr_storage *storage,
			      struct ofi_mr_entry *entry);

enum ofi_mr_storage_type {
	OFI_MR_STORAGE_DEFAULT = 0,
	OFI_MR_STORAGE_RBT,
	OFI_MR_STORAGE_USER,
};

struct ofi_mr_storage {
	enum ofi_mr_storage_type type;
	void *storage;
	ofi_mr_destroy_t destroy;
	ofi_mr_find_t find;
	ofi_mr_insert_t insert;
	ofi_mr_erase_t erase;
};

struct ofi_mr_cache {
	struct util_domain		*domain;
	struct ofi_notification_queue	nq;
	size_t				max_cached_cnt;
	size_t				max_cached_size;
	int				merge_regions;
	size_t				entry_data_size;

	struct ofi_mr_storage		mr_storage;
	struct dlist_entry		lru_list;

	size_t				cached_cnt;
	size_t				cached_size;
	size_t				search_cnt;
	size_t				delete_cnt;
	size_t				hit_cnt;
	struct util_buf_pool		*entry_pool;

	int				(*add_region)(struct ofi_mr_cache *cache,
						      struct ofi_mr_entry *entry);
	void				(*delete_region)(struct ofi_mr_cache *cache,
							 struct ofi_mr_entry *entry);
};

int ofi_mr_cache_init(struct util_domain *domain, struct ofi_mem_monitor *monitor,
		      struct ofi_mr_cache *cache);
void ofi_mr_cache_cleanup(struct ofi_mr_cache *cache);

/* Caller must provide locking around calls */
bool ofi_mr_cache_flush(struct ofi_mr_cache *cache);
int ofi_mr_cache_search(struct ofi_mr_cache *cache, const struct fi_mr_attr *attr,
			struct ofi_mr_entry **entry);
void ofi_mr_cache_delete(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry);


#endif /* _OFI_MR_H_ */
