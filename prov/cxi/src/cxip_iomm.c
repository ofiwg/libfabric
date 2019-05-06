/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_MR, __VA_ARGS__)

size_t iomm_max_cached_cnt = 4*1024ULL;
size_t iomm_max_cached_size = 256*1024*1024*1024ULL;
int iomm_merge_regions = 1;

/**
 * cxip_do_map() - IO map a buffer.
 */
static int cxip_do_map(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry)
{
	int ret;
	struct cxip_md *md = (struct cxip_md *)entry->data;
	struct cxip_domain *dom;
	uint32_t map_flags = CXI_MAP_READ | CXI_MAP_WRITE |
			     CXI_MAP_PIN | CXI_MAP_NTA;

	dom = container_of(cache, struct cxip_domain, iomm);

	ret = cxil_map(dom->dev_if->if_lni, entry->iov.iov_base,
		       entry->iov.iov_len, map_flags, NULL, &md->md);
	if (ret)
		CXIP_LOG_ERROR("cxil_map() failed: %d\n", ret);
	else
		md->dom = dom;

	return ret;
}

/**
 * cxip_do_unmap() - IO unmap a buffer.
 */
static void cxip_do_unmap(struct ofi_mr_cache *cache,
			  struct ofi_mr_entry *entry)
{
	int ret;
	struct cxip_md *md = (struct cxip_md *)entry->data;

	ret = cxil_unmap(md->md);
	if (ret)
		CXIP_LOG_ERROR("cxil_unmap failed: %d\n", ret);
}

/*
 * iomm_mon_unsubscribe() - Subscribe to map events for a region.
 */
int iomm_mon_subscribe(struct ofi_mem_monitor *notifier,
		       struct ofi_subscription *subscription)
{
	return FI_SUCCESS;
}

/*
 * iomm_mon_unsubscribe() - Unsubscribe from map events for a region.
 */
void iomm_mon_unsubscribe(struct ofi_mem_monitor *notifier,
			  struct ofi_subscription *subscription)
{
}

/*
 * cxip_iomm_init() - Initialize domain IO memory map.
 */
int cxip_iomm_init(struct cxip_domain *dom)
{
	int ret;

	dom->iomm_mon.subscribe = iomm_mon_subscribe;
	dom->iomm_mon.unsubscribe = iomm_mon_unsubscribe;
	ofi_monitor_init(&dom->iomm_mon);

	dom->iomm.max_cached_cnt = iomm_max_cached_cnt;
	dom->iomm.max_cached_size = iomm_max_cached_size;
	dom->iomm.merge_regions = iomm_merge_regions;
	dom->iomm.entry_data_size = sizeof(struct cxip_md);
	dom->iomm.add_region = cxip_do_map;
	dom->iomm.delete_region = cxip_do_unmap;
	ret = ofi_mr_cache_init(&dom->util_domain, &dom->iomm_mon, &dom->iomm);
	if (ret)
		CXIP_LOG_ERROR("ofi_mr_cache_init failed: %d\n", ret);

	fastlock_init(&dom->iomm_lock);

	return ret;
}

/*
 * cxip_iomm_fini() - Finalize domain IO memory map.
 */
void cxip_iomm_fini(struct cxip_domain *dom)
{
	fastlock_destroy(&dom->iomm_lock);
	ofi_mr_cache_cleanup(&dom->iomm);
}

/*
 * cxip_map() - Acquire IO mapping for buf.
 *
 * The IO memory map is searched for a IO mapping which covers buf. If no
 * mapping has been established, create one and cache it.
 */
int cxip_map(struct cxip_domain *dom, void *buf, unsigned long len,
	     struct cxip_md **md)
{
	int ret;
	struct iovec iov;
	unsigned long buf_adj;
	const struct fi_mr_attr attr = {
		.iov_count = 1,
		.mr_iov = &iov,
	};
	struct ofi_mr_entry *entry;

	/* TODO align buffer inside cache so driver can control mapping
	 * size.
	 */
	buf_adj = FLOOR(buf, C_PAGE_SIZE);
	iov.iov_base = (void *)buf_adj;

	buf_adj = (unsigned long)buf - buf_adj;
	iov.iov_len = CEILING(len + buf_adj, C_PAGE_SIZE);

	fastlock_acquire(&dom->iomm_lock);
	ret = ofi_mr_cache_search(&dom->iomm, &attr, &entry);
	if (ret) {
		CXIP_LOG_DBG("Failed to acquire mapping (%p, %lu): %d\n",
			     buf, len, ret);
	} else {
		*md = (struct cxip_md *)entry->data;
	}
	fastlock_release(&dom->iomm_lock);

	return ret;
}

/*
 * cxip_unmap() - Release an IO mapping.
 *
 * Drop a refernce to the IO mapping. If this was the last reference, the
 * buffer may be unmapped.
 */
void cxip_unmap(struct cxip_md *md)
{
	struct ofi_mr_entry *entry;

	entry = container_of(md, struct ofi_mr_entry, data);

	fastlock_acquire(&md->dom->iomm_lock);
	ofi_mr_cache_delete(&md->dom->iomm, entry);
	fastlock_release(&md->dom->iomm_lock);
}
