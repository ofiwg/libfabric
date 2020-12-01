/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_MR, __VA_ARGS__)
#define CXIP_INFO(...) _CXIP_INFO(FI_LOG_MR, __VA_ARGS__)

/**
 * cxip_do_map() - IO map a buffer.
 */
static int cxip_do_map(struct ofi_mr_cache *cache, struct ofi_mr_entry *entry)
{
	int ret;
	struct cxip_md *md = (struct cxip_md *)entry->data;
	struct cxip_domain *dom;
	uint32_t map_flags = CXI_MAP_READ | CXI_MAP_WRITE;

	dom = container_of(cache, struct cxip_domain, iomm);

	if (entry->info.iface == FI_HMEM_SYSTEM) {
		if (dom->ats)
			map_flags |= CXI_MAP_ATS;

		if (!dom->odp)
			map_flags |= CXI_MAP_PIN;
	} else {
		map_flags |= CXI_MAP_DEVICE | CXI_MAP_PIN;
	}

	ret = cxil_map(dom->lni->lni, entry->info.iov.iov_base,
		       entry->info.iov.iov_len, map_flags, NULL, &md->md);
	if (ret) {
		CXIP_WARN("cxil_map() failed: %d\n", ret);
	} else {
		md->dom = dom;
		md->info = entry->info;
	}

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
		CXIP_WARN("cxil_unmap failed: %d\n", ret);
}

static int cxip_scalable_iomm_init(struct cxip_domain *dom)
{
	int ret;
	uint32_t map_flags = (CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_ATS);

	/* Create ATS MD to cover all addresses. */
	fastlock_acquire(&dom->iomm_lock);

	ret = cxil_map(dom->lni->lni, 0, -1, map_flags, NULL,
		       &dom->scalable_md.md);
	if (!ret) {
		dom->scalable_md.dom = dom;
		dom->scalable_iomm = true;

		CXIP_DBG("Scalable IOMM enabled.\n");

		if (cxip_env.ats_mlock_mode == CXIP_ATS_MLOCK_ALL) {
			ret = mlockall(MCL_CURRENT | MCL_FUTURE);
			if (ret) {
				CXIP_WARN("mlockall(MCL_CURRENT | MCL_FUTURE) failed: %d\n",
					  -errno);
			}
		}

		ret = FI_SUCCESS;
	} else {
		ret = -FI_ENOSYS;
	}

	fastlock_release(&dom->iomm_lock);

	return ret;
}

static void cxip_scalable_iomm_fini(struct cxip_domain *dom)
{
	cxil_unmap(dom->scalable_md.md);
}

static int cxip_ats_check(struct cxip_domain *dom)
{
	uint32_t map_flags = CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_ATS;
	int stack_var;
	struct cxi_md *md;
	int ret;

	ret = cxil_map(dom->lni->lni, &stack_var, sizeof(stack_var), map_flags,
		       NULL, &md);
	if (!ret) {
		cxil_unmap(md);
		CXIP_INFO("PCIe ATS supported.\n");
		return 1;
	}

	CXIP_INFO("PCIe ATS not supported.\n");
	return 0;
}

/*
 * cxip_iomm_init() - Initialize domain IO memory map.
 */
int cxip_iomm_init(struct cxip_domain *dom)
{
	struct ofi_mem_monitor *memory_monitors[OFI_HMEM_MAX] = {
		[FI_HMEM_SYSTEM] = default_monitor,
		[FI_HMEM_CUDA] = default_cuda_monitor,
		[FI_HMEM_ROCR] = default_rocr_monitor,
	};
	int ret;
	bool scalable;

	fastlock_init(&dom->iomm_lock);

	/* Check if ATS is supported */
	if (cxip_env.ats && cxip_ats_check(dom))
		dom->ats = true;

	if (cxip_env.odp)
		dom->odp = true;

	if (dom->util_domain.info_domain_caps & FI_HMEM)
		dom->hmem = true;

	scalable = dom->ats && dom->odp;

	/* Unpinned ATS translation is scalable. A single MD covers all
	 * memory addresses and a cache isn't necessary.
	 */
	if (scalable) {
		ret = cxip_scalable_iomm_init(dom);
		if (ret)
			CXIP_WARN("cxip_scalable_iomm_init() returned: %d\n",
				  ret);
			return ret;
	}

	if (!scalable || dom->hmem) {
		dom->iomm.entry_data_size = sizeof(struct cxip_md);
		dom->iomm.add_region = cxip_do_map;
		dom->iomm.delete_region = cxip_do_unmap;
		ret = ofi_mr_cache_init(&dom->util_domain, memory_monitors,
					&dom->iomm);
		if (ret) {
			CXIP_WARN("ofi_mr_cache_init failed: %d\n", ret);
			if (scalable)
				cxip_scalable_iomm_fini(dom);
			return ret;
		}
	}

	return FI_SUCCESS;
}

/*
 * cxip_iomm_fini() - Finalize domain IO memory map.
 */
void cxip_iomm_fini(struct cxip_domain *dom)
{
	fastlock_destroy(&dom->iomm_lock);

	if (dom->scalable_iomm)
		cxip_scalable_iomm_fini(dom);

	if (!dom->scalable_iomm || dom->hmem)
		ofi_mr_cache_cleanup(&dom->iomm);
}

/*
 * cxip_map() - Acquire IO mapping for buf.
 *
 * The IO memory map is searched for a IO mapping which covers buf. If no
 * mapping has been established, create one and cache it.
 */
int cxip_map(struct cxip_domain *dom, const void *buf, unsigned long len,
	     struct cxip_md **md)
{
	int ret;
	struct iovec iov;
	unsigned long buf_adj;
	struct fi_mr_attr attr = {
		.iov_count = 1,
		.mr_iov = &iov,
	};
	struct ofi_mr_entry *entry;

	/* Always need to query memory type if HMEM support is enabled. */
	if (dom->hmem)
		attr.iface = ofi_get_hmem_iface(buf);

	if (dom->scalable_iomm && attr.iface == FI_HMEM_SYSTEM) {
		*md = &dom->scalable_md;
		return FI_SUCCESS;
	}

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
		CXIP_DBG("Failed to acquire mapping (%p, %lu): %d\n",
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

	if (md == &md->dom->scalable_md)
		return;

	fastlock_acquire(&md->dom->iomm_lock);
	ofi_mr_cache_delete(&md->dom->iomm, entry);
	fastlock_release(&md->dom->iomm_lock);
}
