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
	struct cxi_md_hints hints;
	void *ze_handle;
	void *ze_base_addr;
	size_t ze_base_size;

	dom = container_of(cache, struct cxip_domain, iomm);

	/* Prefer the ATS (scalable MD) whenever possible
	 *
	 * TODO: ATS (scalable MD) can only support CPU page sizes and should be
	 * avoided for non-standard page sizes.
	 */
	if (dom->scalable_iomm && entry->info.iface == FI_HMEM_SYSTEM) {
		md->md = dom->scalable_md.md;
		md->dom = dom;
		md->info = entry->info;

		return FI_SUCCESS;
	}

	memset(&hints, 0, sizeof(hints));

	if (entry->info.iface == FI_HMEM_SYSTEM) {
		if (dom->ats)
			map_flags |= CXI_MAP_ATS;

		if (!dom->odp)
			map_flags |= CXI_MAP_PIN;
	} else {
		/* TODO: Remove PIN when DMA buf move_notify is supported. */
		map_flags |= CXI_MAP_DEVICE | CXI_MAP_PIN;

		/* ZE support requires the use of the DMA buf FD and offset
		 * hints fields.
		 */
		if (entry->info.iface == FI_HMEM_ZE) {
			ret = ze_hmem_get_handle(entry->info.iov.iov_base,
						 &ze_handle);
			if (ret) {
				CXIP_WARN("ze_hmem_get_handle failed: rc=%d\n",
					  ret);
				return ret;
			}

			ret = ze_hmem_get_base_addr(entry->info.iov.iov_base,
						    &ze_base_addr,
						    &ze_base_size);
			if (ret) {
				CXIP_WARN("ze_hmem_get_base_addr failed: rc=%d\n",
					  ret);
				return ret;
			}

			hints.dmabuf_fd = (int)(uintptr_t)ze_handle;
			hints.dmabuf_offset =
				(uintptr_t)entry->info.iov.iov_base -
				(uintptr_t)ze_base_addr;
			hints.dmabuf_valid = true;
		}
	}

	if (!cxip_env.iotlb)
		map_flags |= CXI_MAP_NOCACHE;

	ret = cxil_map(dom->lni->lni, entry->info.iov.iov_base,
		       entry->info.iov.iov_len, map_flags, &hints, &md->md);
	if (ret) {
		md->dom = NULL;
		CXIP_WARN("cxil_map() failed: %d\n", ret);
	} else {
		/* If the md len is larger than the iov_len, the va and
		 * length have been aligned to a larger page size.
		 * Update the cache memory region by returning -EAGAIN.
		 */
		if (entry->info.iov.iov_len < md->md->len) {
			entry->info.iov.iov_base = (void*)md->md->va;
			entry->info.iov.iov_len = md->md->len;
			ret = -FI_EAGAIN;
		}
		md->dom = dom;
		md->info = entry->info;
		md->cached = true;
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

	if (!md || !md->dom || md->md == md->dom->scalable_md.md)
		return;

	ret = cxil_unmap(md->md);
	if (ret)
		CXIP_WARN("cxil_unmap failed: %d\n", ret);
}

static int cxip_scalable_iomm_init(struct cxip_domain *dom)
{
	int ret;
	uint32_t map_flags = (CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_ATS);

	if (!cxip_env.iotlb)
		map_flags |= CXI_MAP_NOCACHE;

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

	return ret;
}

static void cxip_scalable_iomm_fini(struct cxip_domain *dom)
{
	cxil_unmap(dom->scalable_md.md);
}

static int cxip_ats_check(struct cxip_domain *dom)
{
	uint32_t map_flags = CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_ATS |
			     CXI_MAP_PIN;
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

static void cxip_iomm_set_rocr_dev_mem_only(struct cxip_domain *dom)
{
	int dev_hmem_count = 0;
	bool rocr_support = false;
	int i;

	if (!dom->hmem) {
		dom->rocr_dev_mem_only = false;
		return;
	}

	for (i = 0; i < OFI_HMEM_MAX; i++) {
		if (i == FI_HMEM_SYSTEM)
			continue;

		if (hmem_ops[i].initialized) {
			dev_hmem_count++;

			if (i == FI_HMEM_ROCR)
				rocr_support = true;
		}
	}

	/* If FI_HMEM_ROCR is the ONLY device supported by libfabric and the
	 * core ROCR memory monitor is used, cxip_map can be optimized to avoid
	 * pointer queries.
	 */
	if (dev_hmem_count == 1 && rocr_support &&
	    default_rocr_monitor == rocr_monitor)
		dom->rocr_dev_mem_only = true;
	else
		dom->rocr_dev_mem_only = false;
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
		[FI_HMEM_ZE] = default_ze_monitor,
	};
	enum fi_hmem_iface iface;
	int ret;
	bool scalable;

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
		if (ret) {
			CXIP_WARN("cxip_scalable_iomm_init() returned: %d\n",
				  ret);
			return ret;
		}
	}

	if (!scalable || dom->hmem) {
		dom->iomm.entry_data_size = sizeof(struct cxip_md);
		dom->iomm.add_region = cxip_do_map;
		dom->iomm.delete_region = cxip_do_unmap;
		ret = ofi_mr_cache_init(&dom->util_domain, memory_monitors,
					&dom->iomm);
		if (ret) {
			CXIP_INFO("MR cache init failed: %s. MR caching disabled.\n",
				  fi_strerror(-ret));
		} else {
			for (iface = 0; iface < OFI_HMEM_MAX; iface++) {
				if (dom->iomm.monitors[iface])
					CXIP_INFO("MR cache enabled for %s memory\n",
						  fi_tostr(&iface, FI_TYPE_HMEM_IFACE));
			}
		}
	}

	cxip_iomm_set_rocr_dev_mem_only(dom);

	return FI_SUCCESS;
}

/*
 * cxip_iomm_fini() - Finalize domain IO memory map.
 */
void cxip_iomm_fini(struct cxip_domain *dom)
{
	if (dom->scalable_iomm)
		cxip_scalable_iomm_fini(dom);

	if (!dom->scalable_iomm || dom->hmem)
		ofi_mr_cache_cleanup(&dom->iomm);
}

static int cxip_map_cache(struct cxip_domain *dom, struct fi_mr_attr *attr,
			  struct cxip_md **md)
{
	struct ofi_mr_entry *entry;
	int ret;

	ret = ofi_mr_cache_search(&dom->iomm, attr, &entry);
	if (ret) {
		CXIP_WARN("Failed to acquire mapping (%p, %lu): %d\n",
			  attr->mr_iov->iov_base, attr->mr_iov->iov_len, ret);
		return ret;
	}

	*md = (struct cxip_md *)entry->data;

	return FI_SUCCESS;
}

static int cxip_map_nocache(struct cxip_domain *dom, struct fi_mr_attr *attr,
			    struct cxip_md **md)
{
	struct cxip_md *uncached_md;
	uint32_t map_flags;
	int ret;
	struct cxi_md_hints hints;
	void *ze_handle;
	void *ze_base_addr;
	size_t ze_base_size;

	/* Prefer the ATS (scalable MD) whenever possible
	 *
	 * TODO: ATS (scalable MD) can only support CPU page sizes and should be
	 * avoided for non-standard page sizes.
	 */
	if (dom->scalable_iomm && attr->iface == FI_HMEM_SYSTEM) {
		*md = &dom->scalable_md;
		return FI_SUCCESS;
	}

	memset(&hints, 0, sizeof(hints));

	uncached_md = calloc(1, sizeof(*uncached_md));
	if (!uncached_md)
		return -FI_ENOMEM;

	map_flags = CXI_MAP_READ | CXI_MAP_WRITE;
	if (attr->iface == FI_HMEM_SYSTEM) {
		if (dom->ats)
			map_flags |= CXI_MAP_ATS;

		if (!dom->odp)
			map_flags |= CXI_MAP_PIN;
	} else {
		/* TODO: Remove PIN when DMA buf move_notify is supported. */
		map_flags |= CXI_MAP_DEVICE | CXI_MAP_PIN;

		/* ZE support requires the use of the DMA buf FD and offset
		 * hints fields.
		 */
		if (attr->iface == FI_HMEM_ZE) {
			ret = ze_hmem_get_handle(attr->mr_iov->iov_base,
						 &ze_handle);
			if (ret) {
				CXIP_WARN("ze_hmem_get_handle failed: rc=%d\n",
					  ret);
				return ret;
			}

			ret = ze_hmem_get_base_addr(attr->mr_iov->iov_base,
						    &ze_base_addr,
						    &ze_base_size);
			if (ret) {
				CXIP_WARN("ze_hmem_get_base_addr failed: rc=%d\n",
					  ret);
				return ret;
			}

			hints.dmabuf_fd = (int)(uintptr_t)ze_handle;
			hints.dmabuf_offset =
				(uintptr_t)attr->mr_iov->iov_base -
				(uintptr_t)ze_base_addr;
			hints.dmabuf_valid = true;
		}
	}

	if (!cxip_env.iotlb)
		map_flags |= CXI_MAP_NOCACHE;

	ret = cxil_map(dom->lni->lni, attr->mr_iov->iov_base,
		       attr->mr_iov->iov_len, map_flags, &hints,
		       &uncached_md->md);
	if (ret) {
		CXIP_WARN("cxil_map() failed: %d\n", ret);
		goto err_free_uncached_md;
	}

	uncached_md->dom = dom;
	uncached_md->info.iov.iov_base = (void *)uncached_md->md->va;
	uncached_md->info.iov.iov_len = uncached_md->md->len;
	uncached_md->info.iface = attr->iface;

	*md = uncached_md;

	return FI_SUCCESS;

err_free_uncached_md:
	free(uncached_md);

	return ret;
}

static bool __attribute__ ((noinline)) cxip_map_is_stack_addr(const void *buf)
{
	char *sp = NULL;

	return (char *)buf >= (char *)&sp;
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
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct fi_mr_attr attr = {
		.iov_count = 1,
		.mr_iov = &iov,
	};
	struct ofi_mr_entry *entry;

	/* TODO: ATS (scalable MD) can only support CPU page sizes and should be
	 * avoided for non-standard page sizes.
	 */
	if (dom->scalable_iomm && !dom->hmem) {
		*md = &dom->scalable_md;
		return FI_SUCCESS;
	}

	/* Since the MR cache find operates on virtual addresses and all device
	 * memory must support a unified virtual address space with system
	 * memory, the buffer pointer query can be avoided completely if the
	 * corresponding entry is in the cache.
	 */
	if (cxip_domain_mr_cache_enabled(dom) && dom->rocr_dev_mem_only) {
		entry = ofi_mr_cache_find(&dom->iomm, &attr);
		if (entry) {
			*md = (struct cxip_md *)entry->data;
			return FI_SUCCESS;
		}
	}

	/* Since the MR cache search will allocate a new entry, the MR iface
	 * attribute must be defined for the proper MR cache memory monitor to
	 * be selected.
	 */
	if (dom->hmem)
		attr.iface = ofi_get_hmem_iface(buf);

	if (cxip_env.thread_safe_mr_reg && cxip_map_is_stack_addr(buf))
		goto nocache;

	if (cxip_domain_mr_cache_iface_enabled(dom, attr.iface))
		return cxip_map_cache(dom, &attr, md);

nocache:
	return cxip_map_nocache(dom, &attr, md);
}

static void cxip_unmap_cache(struct cxip_md *md)
{
	struct ofi_mr_entry *entry =
		container_of(md, struct ofi_mr_entry, data);

	ofi_mr_cache_delete(&md->dom->iomm, entry);
}

static void cxip_unmap_nocache(struct cxip_md *md)
{
	int ret;

	ret = cxil_unmap(md->md);
	if (ret)
		CXIP_WARN("cxil_unmap failed: %d\n", ret);

	free(md);
}

/*
 * cxip_unmap() - Release an IO mapping.
 *
 * Drop a refernce to the IO mapping. If this was the last reference, the
 * buffer may be unmapped.
 */
void cxip_unmap(struct cxip_md *md)
{
	/* Scalable MD is owned by the CXIP domain and thus will be freed when
	 * the domain is closed.
	 */
	if (md == &md->dom->scalable_md)
		return;

	if (md->cached)
		cxip_unmap_cache(md);
	else
		cxip_unmap_nocache(md);
}
