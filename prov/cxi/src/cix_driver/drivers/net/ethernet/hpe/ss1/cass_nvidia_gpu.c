// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021 Hewlett Packard Enterprise Development LP */

#include "cass_core.h"

#ifdef HAVE_NVIDIA_P2P
#include "nv-p2p.h"

static bool nv_p2p_persistent = true;
module_param(nv_p2p_persistent, bool, 0644);
MODULE_PARM_DESC(nv_p2p_persistent, "Use persistent P2P mapping");

struct nv_p2p_info {
	u64 va;
	size_t size;
	unsigned long npages;
	unsigned long page_size;
	struct nvidia_p2p_page_table *page_table;
	struct nvidia_p2p_dma_mapping *dma_mapping;
};

int (*p2p_get_pages)(u64 p2p_token, u32 va_space, u64 virtual_address,
		     u64 length, struct nvidia_p2p_page_table **page_table,
		     void (*free_callback)(void *data), void *data);
int (*p2p_put_pages)(u64 p2p_token, u32 va_space, u64 virtual_address,
		     struct nvidia_p2p_page_table *page_table);
int (*p2p_get_pages_pers)(u64 virtual_address, u64 length,
			  struct nvidia_p2p_page_table **page_table, u32 flags);
int (*p2p_put_pages_pers)(u64 virtual_address,
			  struct nvidia_p2p_page_table *page_table, u32 flags);
int (*p2p_dma_map_pages)(struct pci_dev *peer,
			 struct nvidia_p2p_page_table *page_table,
			 struct nvidia_p2p_dma_mapping **dma_mapping);
int (*p2p_dma_unmap_pages)(struct pci_dev *peer,
			   struct nvidia_p2p_page_table *page_table,
			   struct nvidia_p2p_dma_mapping *dma_mapping);

/**
 * nv_free_callback() - Invalidate an md_priv address range
 *
 * This callback is called while the gpu driver is holding its gpu-lock.
 * If the pages are cleaned up by this callback, p2p_put_pages
 * will just return.
 *
 * @data: callback data
 */
static void nv_free_callback(void *data)
{
	const struct cxi_md_priv *md_priv = (struct cxi_md_priv *)data;

	if (!md_priv) {
		pr_warn("No MD found\n");
		return;
	}

	if (!md_priv->cac) {
		pr_warn("cac is null\n");
		return;
	}

	if (!md_priv->p2p_info) {
		pr_warn("p2p_info is null\n");
		return;
	}

	pr_debug("ac:%d md:%d iova:%llx len:%lx\n", md_priv->cac->ac.acid,
		 md_priv->md.id, md_priv->md.iova, md_priv->md.len);

	mutex_lock(&md_priv->cac->ac_mutex);
	cass_clear_range(md_priv, md_priv->md.iova, md_priv->md.len);
	mutex_unlock(&md_priv->cac->ac_mutex);

	cass_invalidate_range(md_priv->cac, md_priv->md.iova, md_priv->md.len);
}

static int nv_get_pages(u64 va, u64 len, struct nv_p2p_info *p2p_info,
		 struct cxi_md_priv *md_priv)
{
	if (nv_p2p_persistent && p2p_get_pages_pers)
		return p2p_get_pages_pers(va, len, &p2p_info->page_table, 0);

	return p2p_get_pages(0, 0, va, len, &p2p_info->page_table,
		!nv_p2p_persistent ? nv_free_callback : NULL, md_priv);
}

static int nv_put_pages(struct nv_p2p_info *p2p_info)
{
	if (nv_p2p_persistent && p2p_put_pages_pers)
		return p2p_put_pages_pers(p2p_info->va, p2p_info->page_table,
					  0);

	return p2p_put_pages(0, 0, p2p_info->va, p2p_info->page_table);
}

static int n_contig(dma_addr_t *daddr, size_t size, int n)
{
	int i;
	dma_addr_t taddr;

	for (i = 1, taddr = daddr[0]; i < n; i++) {
		if (taddr + size != daddr[i])
			return i;

		taddr = daddr[i];
	}

	return i;
}

static int alloc_sg_table(struct nvidia_p2p_dma_mapping *dma_mapping,
			  struct ac_map_opts *m_opts)
{
	int i;
	int nents;
	int ret;
	int ncontig;
	int max_contig = 0;
	struct sg_table *sgt;
	struct scatterlist *sg;
	size_t page_size = BIT(m_opts->page_shift);
	int entries = dma_mapping->entries;
	dma_addr_t *dma_addr = dma_mapping->dma_addresses;
	int ncontig_max = ((u32)INT_MIN) >> m_opts->page_shift;

	sgt = kmalloc(sizeof(struct sg_table), GFP_KERNEL);
	if (!sgt)
		return -ENOMEM;

	ret = sg_alloc_table(sgt, entries, GFP_KERNEL);
	if (ret) {
		pr_warn("sg_alloc_table failed:%d\n", ret);
		kfree(sgt);
		return ret;
	}

	sg = sgt->sgl;

	for (i = 0, nents = 0; sg && i < entries; i += ncontig, nents++) {
		ncontig = n_contig(&dma_addr[i], page_size, entries - i);
		ncontig = __rounddown_pow_of_two(ncontig);
		/* verify dma_addr is aligned to the contiguous count */
		ncontig = min_t(u64, ncontig, BIT(ffsl(dma_addr[i])));
		/* Limit ncontig so that length is not > sizeof(int)
		 * since sg_dma_len() is unsigned int.
		 */
		ncontig = min_t(int, ncontig, ncontig_max);

		sg_dma_address(sg) = dma_addr[i];
		sg_dma_len(sg) = page_size * ncontig;
		sg = sg_next(sg);

		if (max_contig < ncontig)
			max_contig = ncontig;
	}

	sgt->nents = nents;
	m_opts->md_priv->sgt = sgt;

	cass_device_hugepage_size(max_contig, m_opts);

	return 0;
}

/**
 * nvidia_get_pages - Check if address range belongs to the GPU.
 *                    Get GPU pages and return a list of pfns.
 *
 * @m_opts: Map options
 *
 * Return:  0 on success, < 0 on error
 */
static int nvidia_get_pages(struct ac_map_opts *m_opts)
{
	int ret;
	struct cxi_md_priv *md_priv = m_opts->md_priv;
	struct cass_dev *hw = container_of(md_priv->lni_priv->dev,
					   struct cass_dev, cdev);
	struct pci_dev *pdev = hw->cdev.pdev;
	u64 va = m_opts->va_start;
	u64 len = m_opts->va_len;
	int npages = len >> m_opts->page_shift;
	struct nv_p2p_info *p2p_info;
	struct nvidia_p2p_dma_mapping *dma_mapping;

	p2p_info = kmalloc(sizeof(*p2p_info), GFP_KERNEL);
	if (!p2p_info)
		return -ENOMEM;

	p2p_info->va = va;
	p2p_info->size = len;

	ret = nv_get_pages(va, len, p2p_info, md_priv);
	if (ret)
		goto get_pages_fail;

	if (p2p_info->page_table->entries > npages) {
		pr_info("entries:%d returned greater than npages:%d\n",
			 p2p_info->page_table->entries, npages);
		ret = -EINVAL;
		goto error;
	}

	ret = p2p_dma_map_pages(pdev, p2p_info->page_table, &dma_mapping);
	if (ret) {
		pr_err("nvidia_p2p_dma_map_pages error:%d\n", ret);
		goto error;
	}

	if (dma_mapping->entries != npages)
		pr_warn("md:%d va:%llx len:%llx npages:%d entries:%d\n",
			 md_priv->md.id, va, len, npages, dma_mapping->entries);

	ret = alloc_sg_table(dma_mapping, m_opts);
	if (ret)
		goto alloc_sg_table_error;

	p2p_info->dma_mapping = dma_mapping;
	p2p_info->npages = dma_mapping->entries;

	md_priv->p2p_info = p2p_info;
	refcount_inc(&md_priv->refcount);

	return ret;

alloc_sg_table_error:
	p2p_dma_unmap_pages(pdev, p2p_info->page_table, dma_mapping);
error:
	nv_put_pages(p2p_info);
get_pages_fail:
	kfree(p2p_info);

	return ret;
}

/**
 * nvidia_put_pages - Check if put_pages is needed via cleanup_done.
 *                    Put GPU pages and free memory.
 *
 * A gpu-lock is held during put_pages and when it returns, the callback
 * will not be called any longer. If the pages are cleaned up by the
 * callback, put_pages will just return. When this function returns,
 * the md_priv will be freed.
 *
 * @md_priv: Private memory descriptor
 */
static void nvidia_put_pages(struct cxi_md_priv *md_priv)
{
	int rc;
	struct nv_p2p_info *p2p_info = (struct nv_p2p_info *)md_priv->p2p_info;
	struct cass_dev *hw = container_of(md_priv->lni_priv->dev,
					   struct cass_dev, cdev);
	struct pci_dev *pdev = hw->cdev.pdev;

	BUG_ON(p2p_info->page_table == NULL);

	p2p_dma_unmap_pages(pdev, p2p_info->page_table, p2p_info->dma_mapping);

	rc = nv_put_pages(p2p_info);
	if (rc)
		pr_info("%s rc:%d md:%d va:%llx\n", __func__, rc,
			md_priv->md.id, p2p_info->va);

	pr_debug("md:%d va:%llx\n", md_priv->md.id, p2p_info->va);

	sg_free_table(md_priv->sgt);
	kfree(md_priv->sgt);
	md_priv->sgt = NULL;
	kfree(md_priv->p2p_info);
	refcount_dec(&md_priv->refcount);
}

#define NV_DEF_PAGE_SHIFT 16

/**
 * nvidia_is_device_mem - Check if this virtual address is a allocated in
 *                        device memory and return the page size as a shift
 *                        value.
 *
 * @va: Virtual address
 * @len: Length
 * @page_shift: Page shift of this address
 *
 * Return: 0 if this address is device memory, < 0 if not device memory
 *         or error
 *
 * Note: nvidia_get_pages() does not return the page size and currently
 *       only supports a page size of 64k.
 */
static int nvidia_is_device_mem(uintptr_t va, size_t len, int *page_shift)
{
	*page_shift = NV_DEF_PAGE_SHIFT;

	return 0;
}

int nvidia_p2p_init(void)
{
	p2p_get_pages = symbol_request(nvidia_p2p_get_pages);
	p2p_put_pages = symbol_request(nvidia_p2p_put_pages);
	p2p_dma_map_pages = symbol_request(nvidia_p2p_dma_map_pages);
	p2p_dma_unmap_pages = symbol_request(nvidia_p2p_dma_unmap_pages);
	if (!p2p_get_pages || !p2p_put_pages || !p2p_dma_map_pages ||
	    !p2p_dma_unmap_pages) {
		pr_info("%s: Failed to find Nvidia GPU symbols. Nvidia RDMA support disabled.\n",
			CXI_MODULE_NAME);
		return -ENODEV;
	}

	pr_info("%s: Have Nvidia P2P interface\n", CXI_MODULE_NAME);

#ifdef NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API
#pragma message "Have NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API"
	p2p_get_pages_pers = symbol_request(nvidia_p2p_get_pages_persistent);
	p2p_put_pages_pers = symbol_request(nvidia_p2p_put_pages_persistent);
	if (!p2p_get_pages_pers || !p2p_put_pages_pers)
		pr_info("No persistent Nvidia GPU symbols. Using original NVIDIA P2P API\n");
	else
		pr_info("Have NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API\n");
#else
#pragma message "Have original NVIDIA P2P API"
	pr_info("Have original NVIDIA P2P API\n");

	p2p_get_pages_pers = NULL;
	p2p_put_pages_pers = NULL;
#endif

	if (nv_p2p_persistent)
		pr_info("Using persistent mode\n");
	else
		pr_info("Using non-persistent mode\n");

	p2p_ops.get_pages = nvidia_get_pages;
	p2p_ops.put_pages = nvidia_put_pages;
	p2p_ops.is_device_mem = nvidia_is_device_mem;

	return 0;
}

void nvidia_p2p_fini(void)
{
	if (p2p_get_pages) {
		p2p_ops.get_pages = NULL;
		p2p_ops.put_pages = NULL;
		p2p_ops.is_device_mem = NULL;

		symbol_put(nvidia_p2p_get_pages);
		symbol_put(nvidia_p2p_put_pages);
		symbol_put(nvidia_p2p_dma_map_pages);
		symbol_put(nvidia_p2p_dma_unmap_pages);

#ifdef NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API
		if (p2p_get_pages_pers)
			symbol_put(nvidia_p2p_get_pages_persistent);
		if (p2p_put_pages_pers)
			symbol_put(nvidia_p2p_put_pages_persistent);
#endif
		p2p_get_pages = NULL;
		p2p_put_pages = NULL;
	}
}

#else

int nvidia_p2p_init(void)
{
	pr_info("%s: Nvidia GPU P2P not supported\n", CXI_MODULE_NAME);
	return -ENOSYS;
}

void nvidia_p2p_fini(void)
{
	/* NOOP */
}

#endif /* HAVE_NVIDIA_P2P */
