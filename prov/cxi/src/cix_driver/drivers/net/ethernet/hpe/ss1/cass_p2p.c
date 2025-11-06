// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021 Hewlett Packard Enterprise Development LP */

/* GPU direct (P2P) interface */

#include "cass_core.h"

static bool device_hugepage_enable = true;
module_param(device_hugepage_enable, bool, 0644);
MODULE_PARM_DESC(device_hugepage_enable, "Enable device hugepages");

#define DEVICE_HS_ARRAY_SIZE 2
static int device_hs_array_n = DEVICE_HS_ARRAY_SIZE;
/* Default to 2MB and 64M. */
static int device_hs_array[DEVICE_HS_ARRAY_SIZE] = {21, 26};

#define MIN_DEVICE_HP_SHIFT (MIN_DEVICE_PG_TABLE_SIZE + PAGE_SHIFT)
#define MAX_DEVICE_HP_SHIFT (MIN_DEVICE_HP_SHIFT + MAX_PG_TABLE_SIZE)
#define DEVICE_HS_ARRAY_BUFFER_SIZE (DEVICE_HS_ARRAY_SIZE * 3)

static int device_hs_array_get(char *buffer, const struct kernel_param *kp)
{
	return param_array_ops.get(buffer, kp);
}

static int device_hs_array_set(const char *val, const struct kernel_param *kp)
{
	int i;
	int ret;
	int prev = 0;
	char buf[DEVICE_HS_ARRAY_BUFFER_SIZE];

	/* save the current list in case there is an invalid value */
	ret = device_hs_array_get(buf, kp);
	if (ret < 0)
		return ret;

	/* set to the new value */
	ret = param_array_ops.set(val, kp);
	if (ret)
		return ret;

	/* check for out of range and that they are ascending in value */
	for (i = 0; i < device_hs_array_n; i++) {
		if (device_hs_array[i] < MIN_DEVICE_HP_SHIFT ||
				device_hs_array[i] > MAX_DEVICE_HP_SHIFT) {
			pr_info("Value (%d) out of range of %d-%d\n",
				device_hs_array[i], MIN_DEVICE_HP_SHIFT,
				MAX_DEVICE_HP_SHIFT);
			goto invalid;
		}

		if (device_hs_array[i] < prev) {
			pr_info("Values should be ascending\n");
			goto invalid;
		}

		prev = device_hs_array[i];
	}

	return 0;
invalid:
	/* restore previous values */
	param_array_ops.set(buf, kp);
	return -EINVAL;
}

static const struct kernel_param_ops device_hs_array_ops = {
	.set = device_hs_array_set,
	.get = device_hs_array_get,
};

/* Modified version of module_param_array_named to include the ops */
#define module_param_array_named_cb(name, array, type, _ops, nump, perm) \
	param_check_##type(name, &(array)[0]);				\
	static const struct kparam_array __param_arr_##name		\
	= { .max = ARRAY_SIZE(array), .num = nump,			\
	    .ops = &param_ops_##type,					\
	    .elemsize = sizeof(array[0]), .elem = array };		\
	__module_param_call(MODULE_PARAM_PREFIX, name, &_ops,		\
			    .arr = &__param_arr_##name, perm, -1, 0);	\
	__MODULE_PARM_TYPE(name, "array of " #type)

module_param_array_named_cb(device_hs_array, device_hs_array, int,
			    device_hs_array_ops, &device_hs_array_n, 0644);
MODULE_PARM_DESC(device_hs_array, "Array of allowed hugeshift values");

enum cxi_p2p_type {
	CXI_P2P_AMD_GPU = 1,
	CXI_P2P_NVIDIA_GPU,
};

static enum cxi_p2p_type cxi_supported_p2p_type;
static DEFINE_MUTEX(p2p_init_lock);

struct cass_p2p_ops p2p_ops = {
	.get_pages = NULL,
	.put_pages = NULL,
	.is_device_mem = NULL,
};

/**
 * cxi_p2p_init - Discover a GPU direct interface and initialize it.
 */
static int cxi_p2p_init(void)
{
	int ret = 0;

	if (cxi_supported_p2p_type > 0)
		return 0;

	mutex_lock(&p2p_init_lock);

	if (cxi_supported_p2p_type > 0)
		goto out_unlock;

#ifdef HAVE_NVIDIA_P2P
	ret = nvidia_p2p_init();
	if (!ret) {
		cxi_supported_p2p_type = CXI_P2P_NVIDIA_GPU;
		goto out_unlock;
	}
#endif

#ifdef HAVE_AMD_RDMA
	ret = amd_p2p_init();
	if (!ret)
		cxi_supported_p2p_type = CXI_P2P_AMD_GPU;
#endif

out_unlock:
	mutex_unlock(&p2p_init_lock);

	return ret;
}

/**
 * cxi_p2p_fini - Clean up a GPU direct interface.
 */
void cxi_p2p_fini(void)
{
	switch (cxi_supported_p2p_type) {
	case CXI_P2P_AMD_GPU:
#ifdef HAVE_AMD_RDMA
		amd_p2p_fini();
#endif
		break;
	case CXI_P2P_NVIDIA_GPU:
#ifdef HAVE_NVIDIA_P2P
		nvidia_p2p_fini();
#endif
		break;
	}
}

/**
 * cass_device_md() - Allocate a memory descriptor and get GPU device pages
 *
 * @lni_priv: LNI
 * @va: device virtual address
 * @len: length
 * @dma_addr: dma address of va
 * @return: memory descriptor
 */
struct cxi_md_priv *cass_device_md(struct cxi_lni_priv *lni_priv, u64 va,
				   size_t len, dma_addr_t *dma_addr)
{
	int rc;
	struct ac_map_opts m_opts = {};
	struct cxi_md_priv *md_priv;
	struct cass_dev *hw = container_of(lni_priv->dev, struct cass_dev,
					   cdev);

	rc = cass_is_device_memory(hw, &m_opts, va, len);
	if (rc)
		return ERR_PTR(rc);

	md_priv = kzalloc(sizeof(*md_priv), GFP_KERNEL);
	if (!md_priv)
		return ERR_PTR(-ENOMEM);

	m_opts.md_priv = md_priv;
	cass_align_start_len(&m_opts, va, len, m_opts.page_shift);

	md_priv->lni_priv = lni_priv;
	md_priv->md.va = m_opts.va_start;
	md_priv->md.len = m_opts.va_len;
	md_priv->md.page_shift = m_opts.page_shift;

	refcount_set(&md_priv->refcount, 1);

	rc = cass_device_get_pages(&m_opts);
	if (rc) {
		kfree(md_priv);
		return ERR_PTR(rc);
	}

	/* Get the first entry of the sg_table */
	*dma_addr = sg_dma_address(md_priv->sgt->sgl);

	return md_priv;
}

/**
 * cass_device_hugepage_size - Determine an appropriate hugepage size for
 *                             the dma address list based on the number of
 *                             contiguous entries.
 *
 * @contig_cnt: Number contiguous entries
 * @m_opts: User map options
 */
void cass_device_hugepage_size(int contig_cnt, struct ac_map_opts *m_opts)
{
	int i;
	int va_shift;
	int len_shift;
	int cntg_shift;
	int daddr_shift;
	int align_shift;
	dma_addr_t dma_addr = sg_dma_address(m_opts->md_priv->sgt->sgl);

	if (!device_hugepage_enable)
		return;

	cntg_shift = flsl(contig_cnt) + m_opts->page_shift;
	len_shift = ffsl(__rounddown_pow_of_two(m_opts->va_len));
	va_shift = ffsl(m_opts->va_start);
	daddr_shift = ffsl(dma_addr);
	align_shift = min_t(int, va_shift, len_shift);
	align_shift = min_t(int, daddr_shift, align_shift);
	align_shift = min_t(int, cntg_shift, align_shift);

	/* From the list of usable hugepage sizes, pick the largest one
	 * that the registration consumes.
	 */
	for (i = device_hs_array_n - 1; i >= 0; i--) {
		int page_table_size = device_hs_array[i] - m_opts->page_shift;

		if (device_hs_array[i] <= align_shift &&
				page_table_size <= MAX_PG_TABLE_SIZE &&
				page_table_size >= MIN_DEVICE_PG_TABLE_SIZE) {
			m_opts->is_huge_page = true;
			m_opts->huge_shift = device_hs_array[i];
			break;
		}
	}

	pr_debug("len:%lx align_shift:%d hs:%d is_hp:%d\n", m_opts->va_len,
		 align_shift, m_opts->huge_shift, m_opts->is_huge_page);
}

/**
 * cass_is_device_memory - Check if va is backed by device memory
 *
 * Also, return the page size.
 *
 * @hw: The cassini device
 * @m_opts: User map options
 * @va: Device virtual address
 * @len: Length of address to map in bytes
 * @return: 0 on success or < 0 on failure
 */
int cass_is_device_memory(struct cass_dev *hw, struct ac_map_opts *m_opts,
			  uintptr_t va, size_t len)
{
	int ret;

	if (m_opts->md_priv &&
	    m_opts->md_priv->dmabuf_fd != INVALID_DMABUF_FD) {
		m_opts->page_shift = PAGE_SHIFT;

		return 0;
	}

	if (p2p_ops.is_device_mem == NULL) {
		ret = cxi_p2p_init();
		if (ret) {
			cxidev_info_ratelimited(&hw->cdev, "p2p_init failed %d\n",
						ret);
			return ret;
		}
	}

	ret = p2p_ops.is_device_mem(va, len, &m_opts->page_shift);
	if (ret) {
		cxidev_info_ratelimited(&hw->cdev, "Expected device memory ret:%d\n",
					ret);
		return ret;
	}

	return 0;
}

/**
 * cass_device_get_pages - Get device pages
 *
 * @m_opts: User map options
 * @return: 0 on success or < 0 on failure
 */
int cass_device_get_pages(struct ac_map_opts *m_opts)
{
	if (m_opts->md_priv->dmabuf_fd != INVALID_DMABUF_FD)
		return cxi_dmabuf_get_pages(m_opts);

	if (p2p_ops.get_pages)
		return p2p_ops.get_pages(m_opts);

	return -EINVAL;
}

/**
 * cass_device_put_pages - Put device pages returned from cass_device_get_pages
 *
 * @md_priv: Private memory descriptor
 */
void cass_device_put_pages(struct cxi_md_priv *md_priv)
{
	if (md_priv->dmabuf_fd != INVALID_DMABUF_FD) {
		cxi_dmabuf_put_pages(md_priv);
		return;
	}

	p2p_ops.put_pages(md_priv);
}
