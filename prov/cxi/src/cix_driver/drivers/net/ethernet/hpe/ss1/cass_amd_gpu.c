// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021 Hewlett Packard Enterprise Development LP */

#include "cass_core.h"

#ifdef HAVE_AMD_RDMA
/*
 * See https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver for details
 * on the AMD RDMA interface.
 */
#include "drm/amd_rdma.h"

#define AMDGPU_MD_DEBUG(md_priv, fmt, ...) \
	pr_debug("AMDGPU buf MD va=%#llx iova=%#llx len=%lu: " fmt "", \
		(md_priv)->md.va, (md_priv)->md.iova, (md_priv)->md.len, \
		##__VA_ARGS__)

int (*query_rdma_interface)(const struct amd_rdma_interface **ops);
static const struct amd_rdma_interface *amd_rdma_ops;

static void amd_free_callback(void *client_priv)
{
	struct cxi_md_priv *md_priv = (struct cxi_md_priv *)client_priv;

	if (!md_priv) {
		pr_warn("No MD found\n");
		return;
	}

	AMDGPU_MD_DEBUG(md_priv, "AMD free callback called\n");

	if (md_priv->cac) {
		mutex_lock(&md_priv->cac->ac_mutex);

		cass_clear_range(md_priv, md_priv->md.iova,
				 md_priv->md.len);

		mutex_unlock(&md_priv->cac->ac_mutex);
		cass_invalidate_range(md_priv->cac, md_priv->md.iova,
				      md_priv->md.len);
	}

	/* AMD driver will free resources when we return from this callback.
	 * Set cleanup_done flag for "amd_put_pages".
	 */
	WRITE_ONCE(md_priv->cleanup_done, true);
}

/**
 * amd_get_pages - Check if address range belongs to the GPU.
 *                 Get GPU pages and return a list of pfns.
 *
 * @m_opts: Map options
 *
 * Return:  0 on success, < 0 on error
 */
static int amd_get_pages(struct ac_map_opts *m_opts)
{
	int ret;
	struct cxi_md_priv *md_priv = m_opts->md_priv;
	u64 va = m_opts->va_start;
	u64 len = m_opts->va_len;
	struct amd_p2p_info *p2p_info;
	struct pid *pid = get_task_pid(current, PIDTYPE_PID);

#ifndef AMD_RDMA_MAJOR
	ret = amd_rdma_ops->get_pages(va, len, pid, &p2p_info,
				      amd_free_callback, md_priv);
#elif AMD_RDMA_MAJOR == 2
	ret = amd_rdma_ops->get_pages(va, len, pid,
				      &md_priv->lni_priv->dev->pdev->dev,
				      &p2p_info, amd_free_callback, md_priv);
#else
#error "AMD RDMA API version not supported"
#endif
	if (ret) {
		AMDGPU_MD_DEBUG(md_priv, "Get pages failed: rc=%d\n", ret);
		return ret;
	}

	refcount_inc(&md_priv->refcount);
	md_priv->p2p_info = p2p_info;
	md_priv->sgt = p2p_info->pages;

	AMDGPU_MD_DEBUG(md_priv, "Buf mapped\n");

	return 0;

	amd_rdma_ops->put_pages(&p2p_info);
	return ret;
}

/**
 * amd_put_pages - Check if put_pages is needed via cleanup_done.
 *                 Put GPU pages and free memory.
 *
 * @md_priv: Private memory descriptor
 */
static void amd_put_pages(struct cxi_md_priv *md_priv)
{
	struct amd_p2p_info *p2p_info =
		(struct amd_p2p_info *)md_priv->p2p_info;

	if (!p2p_info)
		return;

	refcount_dec(&md_priv->refcount);

	if (!READ_ONCE(md_priv->cleanup_done)) {
		amd_rdma_ops->put_pages(&p2p_info);
		AMDGPU_MD_DEBUG(md_priv, "Buf unmapped\n");
	} else {
		AMDGPU_MD_DEBUG(md_priv, "Skipping unmapped\n");
	}

	md_priv->sgt = NULL;
}

static int amd_is_device_mem(uintptr_t va, size_t len, int *page_shift)
{
	int ret;
	ulong page_size;
	struct pid *pid = get_task_pid(current, PIDTYPE_PID);

	ret = amd_rdma_ops->is_gpu_address(va, pid);
	if (!ret)
		return -ENOENT;

	ret = amd_rdma_ops->get_page_size(va, len, pid, &page_size);
	if (ret)
		return ret;

	*page_shift = __ffs(page_size);

	return 0;
}

int amd_p2p_init(void)
{
	int ret;

	query_rdma_interface = symbol_request(amdkfd_query_rdma_interface);
	if (!query_rdma_interface) {
		pr_info("%s: Failed to find AMD GPU symbols. AMD RDMA support disabled.\n",
			CXI_MODULE_NAME);
		return -ENODEV;
	}

	ret = query_rdma_interface(&amd_rdma_ops);
	if (ret < 0) {
		pr_err("%s: Query AMD RDMA Interface failed (%d)\n",
		       CXI_MODULE_NAME, ret);
		return -ENODEV;
	}

	pr_info("%s: Have AMD GPU RDMA interface\n", CXI_MODULE_NAME);

	p2p_ops.get_pages = amd_get_pages;
	p2p_ops.put_pages = amd_put_pages;
	p2p_ops.is_device_mem = amd_is_device_mem;

	return 0;
}

void amd_p2p_fini(void)
{
	if (query_rdma_interface) {
		p2p_ops.get_pages = NULL;
		p2p_ops.put_pages = NULL;
		p2p_ops.is_device_mem = NULL;

		symbol_put(amdkfd_query_rdma_interface);

		query_rdma_interface = NULL;
	}
}

#else

int amd_p2p_init(void)
{
	pr_info("%s: AMD GPU P2P not supported\n", CXI_MODULE_NAME);
	return -ENOSYS;
}

void amd_p2p_fini(void)
{
	/* NOOP */
}

#endif /* HAVE_AMD_RDMA */
