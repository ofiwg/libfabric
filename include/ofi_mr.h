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

#define OFI_MR_BASIC_MAP (FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)

/* FI_LOCAL_MR is valid in pre-libfaric-1.5 and can be valid in
 * post-libfabric-1.5 */
#define OFI_CHECK_MR_LOCAL(info)						\
	((info->domain_attr->mr_mode & FI_MR_LOCAL) ||				\
	 (!(info->domain_attr->mr_mode & ~(FI_MR_BASIC | FI_MR_SCALABLE)) &&	\
	  (info->mode & FI_LOCAL_MR)))

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

#endif /* _OFI_MR_H_ */
