// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/* Handle differences in the HMM interface for different kernel versions. */

#include "cass_core.h"
#include "linux/sched/mm.h"

int cass_vma_write_flag(struct mm_struct *mm, ulong start, ulong end, u32 flags)
{
	int write;
	struct vm_area_struct *vma;

	vma = find_vma_intersection(mm, start, end);

	/* When prefetching, indicate to caller there is no VMA */
	if (!vma && (flags & CXI_MAP_PREFETCH))
		return -ENOENT;

	if (!vma || start < vma->vm_start) {
		pr_warn("No VMA covering 0x%016lx)\n", start);
		return -EINVAL;
	}

	write = vma->vm_flags & VM_WRITE;
	if (!write && (flags & CXI_MAP_WRITE)) {
		pr_warn("VMA does not have write permissions\n");
		return -EINVAL;
	}

	return write;
}

void cass_notifier_cleanup(struct cxi_md_priv *md_priv)
{
	if (!md_priv->mn_sub.mm)
		return;

	mmu_interval_notifier_remove(&md_priv->mn_sub);
	md_priv->mn_sub.mm = NULL;
}

/* Fault in pages from all VMAs in the address range.
 * When prefetching, ignore failures of VMAs that may have errors.
 * If -EBUSY is returned, continue at the address where we left off on
 * the next invocation.
 *
 * When faulting pages, the full range must be backed by contiguous VMAs.
 * Return a failure if all pages cannot be faulted.
 *
 * When prefetching, if no VMAs are found, return 0.
 */
static int cass_range_fault(struct hmm_range *range, struct mm_struct *mm,
			    const struct ac_map_opts *m_opts, ulong end)
{
	int ret = 0;
	bool success = false;
	ulong start = range->start;
	ulong *pfns = range->hmm_pfns;
	int flags = m_opts->flags;
	int page_shift = m_opts->page_shift;
	struct vm_area_struct *vma;
#ifdef VMA_ITERATOR
	VMA_ITERATOR(vmi, mm, start);
#endif

	vma = find_vma_intersection(mm, range->start, range->end);
	if (!vma) {
		if (flags & CXI_MAP_FAULT) {
			pr_debug("Faulting region not backed by VMA\n");
			return -EINVAL;
		}

		return 0;
	}

	/* When faulting, we expect the full range to be covered by VMA(s) */
	if (range->default_flags & HMM_PFN_REQ_FAULT)
		return hmm_range_fault(range);

	for_each_vma(vmi, vma) {
		if (vma->vm_start >= end)
			break;

		/* When prefetching, skip over VMAs that don't match
		 * the requested permissions.
		 */
		if ((flags & CXI_MAP_WRITE && !(vma->vm_flags & VM_WRITE)) ||
		    (flags & CXI_MAP_READ && !(vma->vm_flags & VM_READ))) {
			continue;
		}

		if (vma->vm_end < end)
			range->end = vma->vm_end;
		else
			range->end = end;

		if (range->start < vma->vm_start)
			range->start = vma->vm_start;

		range->hmm_pfns = pfns + ((range->start - start) >> page_shift);

		ret = hmm_range_fault(range);
		if (ret == -EBUSY || ret == -ENOMEM)
			return ret;

		if (!ret || success)
			success = true;

		atu_debug("start:%lx end:%lx len:%lx vm_start:%lx vm_end:%lx vm_flags:%lx ret:%d\n",
			  range->start, range->end, range->end - range->start,
			  vma->vm_start, vma->vm_end, vma->vm_flags, ret);
	}

	return success ? 0 : ret;
}

/**
 * cass_mirror_fault() - Fault and mirror an address range
 *
 * @m_opts: map options
 * @pfns: pfns to mirror
 * @count: number of pages to mirror
 * @addr: start virtual address to fault
 * @len: length of address range to fault
 *
 * @return: 0 on success or negative error value
 */
int cass_mirror_fault(const struct ac_map_opts *m_opts, u64 *pfns, int count,
		      uintptr_t addr, size_t len)
{
	long ret;
	bool is_huge_page = false;
	unsigned long timeout;
	struct hmm_range range = {};
	u64 end = addr + len;
	struct cxi_md_priv *md_priv = m_opts->md_priv;
	struct cass_ac *cac = md_priv->cac;
	ulong addr_mask = MASK(m_opts->huge_shift);
	struct mmu_interval_notifier *mn_sub = &m_opts->md_priv->mn_sub;
	struct mm_struct *mm = mn_sub->mm;

	range.start = addr;
	range.end = end;
	range.hmm_pfns = (unsigned long *)pfns;
	range.pfn_flags_mask = 0;
	range.notifier = mn_sub;

	if (!(m_opts->flags & CXI_MAP_PREFETCH))
		range.default_flags = HMM_PFN_REQ_FAULT;

	if (!mn_sub->mm) {
		pr_debug("Invalid md addr:%lx\n", addr);
		return -ENOENT;
	}

	if (!mmget_not_zero(mm)) {
		pr_debug("mm_user is 0 addr:%lx\n", addr);
		return -ESRCH;
	}

	timeout = jiffies + msecs_to_jiffies(HMM_RANGE_DEFAULT_TIMEOUT);
again:
	range.notifier_seq = mmu_interval_read_begin(mn_sub);
	mmap_read_lock(mm);

	ret = cass_vma_write_flag(mm, addr, end, m_opts->flags);
	if (ret < 0) {
		/* An -ENOENT indicates there is no VMA when prefetching. */
		if (ret == -ENOENT)
			ret = 0;

		goto out_unlock;
	}

	range.default_flags |= ret ? HMM_PFN_REQ_WRITE : 0;

	ret = cass_range_fault(&range, mm, m_opts, end);
	mmap_read_unlock(mm);
	if (ret) {
		if (ret == -EBUSY && !time_after(jiffies, timeout))
			goto again;

		goto out;
	}

	/* ATS mode just needs to fault the pages. */
	if (cac->flags & CXI_MAP_ATS) {
		if (mmu_interval_read_retry(mn_sub, range.notifier_seq))
			goto again;

		ret = count;
		goto out;
	}

	mutex_lock(&cac->ac_mutex);

	if (mmu_interval_read_retry(mn_sub, range.notifier_seq)) {
		mutex_unlock(&cac->ac_mutex);
		goto again;
	}

	if (m_opts->is_huge_page && !(addr & addr_mask) &&
			(len >= BIT(m_opts->huge_shift)))
		is_huge_page = true;

	ret = cass_pfns_mirror(md_priv, m_opts, pfns, count, is_huge_page);

	mutex_unlock(&cac->ac_mutex);

	mmput(mm);

	return ret;

out_unlock:
	mmap_read_unlock(mm);
out:
	mmput(mm);

	return ret;
}

/**
 * cass_sync_pagetables - Called by the hmm notifier to clear page table
 *                        entries and invalidate cache entries.
 *                        mmap_sem is held during this call.
 *
 * @mn_sub: MMU interval notifier
 * @update: update information for callback
 * @cur_seq:
 */
static bool cass_sync_pagetables(struct mmu_interval_notifier *mn_sub,
				 const struct mmu_notifier_range *update,
				 unsigned long cur_seq)
{
	struct cxi_md_priv *md_priv = container_of(mn_sub, struct cxi_md_priv,
						   mn_sub);
	struct cass_dev *hw = container_of(md_priv->cac->lni_priv->dev,
					   struct cass_dev, cdev);
	struct cxi_md md = md_priv->md;
	unsigned long start = update->start;
	unsigned long end = update->end;
	u64 va_end = md.va + md.len;
	u64 s = max_t(u64, md.va, start);
	u64 e = min_t(u64, va_end, end);
	u64 offset = s - md.va;

	if (md_priv->flags & CXI_MAP_ATS)
		return true;

	if (!mmu_notifier_range_blockable(update))
		return false;

	pr_debug("md:%d va:%llx s:%llx e:%llx update start:%lx end:%lx\n",
		 md.id, md.va, s, e, start, end);

	mutex_lock(&md_priv->cac->ac_mutex);
	mmu_interval_set_seq(mn_sub, cur_seq);

	cass_clear_range(md_priv, md.iova + offset, e - s);
	mutex_unlock(&md_priv->cac->ac_mutex);
	cass_invalidate_range(md_priv->cac, md.iova + offset, e - s);
	atomic_inc(&hw->dcpl_nta_mn_inval);

	return true;
}

static const struct mmu_interval_notifier_ops cass_notifier_ops = {
	.invalidate = cass_sync_pagetables,
};

/**
 * cass_mmu_notifier_insert() - Set up notifier for this descriptor
 *
 * @md_priv: Private memory descriptor
 * @m_opts:  User options
 *
 * @return: 0 on success or negative value on error
 */
int cass_mmu_notifier_insert(struct cxi_md_priv *md_priv,
			     const struct ac_map_opts *m_opts)
{
	int ret;

	ret = mmu_interval_notifier_insert(&md_priv->mn_sub,
					   current->mm,
					   m_opts->va_start,
					   m_opts->va_len,
					   &cass_notifier_ops);
	if (ret)
		return ret;

	if (m_opts->flags & (CXI_MAP_FAULT | CXI_MAP_PREFETCH)) {
		ret = cass_mirror_odp(m_opts, md_priv->cac,
				      m_opts->va_len >> m_opts->page_shift,
				      m_opts->va_start);
		if (ret)
			cass_notifier_cleanup(md_priv);
	}

	return ret;
}

int cass_odp_supported(struct cass_dev *hw, u32 flags)
{
	if (cass_version(hw, CASSINI_1_0) && !(flags & CXI_MAP_PIN) &&
	    (flags & CXI_MAP_USER_ADDR)) {
		cxidev_info(&hw->cdev, "Cassini 1.0 does not support ODP\n");
		return -EOPNOTSUPP;
	}

	if (cass_version(hw, CASSINI_1) && (flags & CXI_MAP_ATS) &&
			!hw->ats_c1_odp_enable && !ats_c1_override) {
		cxidev_dbg(&hw->cdev,
			   "ATS support requires invalidate interface for Cassini 1.x\n");
		return -EOPNOTSUPP;
	}

	return 0;
}
