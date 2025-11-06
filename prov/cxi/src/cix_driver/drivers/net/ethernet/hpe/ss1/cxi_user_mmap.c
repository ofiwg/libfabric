// SPDX-License-Identifier: GPL-2.0 OR Linux-OpenIB
/*
 * Copyright (c) 2016 Mellanox Technologies Ltd. All rights reserved.
 * Copyright (c) 2015 System Fabric Works, Inc. All rights reserved.
 * Copyright 2018 Hewlett Packard Enterprise Development LP
 */

/* mmap resources to userspace. */

#include <linux/version.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/vmalloc.h>

#include "cxi_prov_hw.h"
#include "cxi_user.h"

static void cxi_vma_open(struct vm_area_struct *vma)
{
	struct cxi_mmap_info *ip = vma->vm_private_data;

	if (ip->obj)
		atomic_inc(&ip->obj->mappings);
}

static void cxi_vma_close(struct vm_area_struct *vma)
{
	struct cxi_mmap_info *ip = vma->vm_private_data;

	ip->vm_start = 0;
	ip->vm_end = 0;
	if (ip->obj)
		atomic_dec(&ip->obj->mappings);
}

static struct vm_operations_struct cxi_vm_ops = {
	.open = cxi_vma_open,
	.close = cxi_vma_close,
};

/* This is a copy of vm_iomap_memory() with a small change. It appears
 * that vm_iomap_memory only works when the physical page are mapped
 * from the first page of the mapping. If we map something else before
 * it, it breaks because pg_off is not 0.
 */
static int ucxi_iomap_memory(struct vm_area_struct *vma, phys_addr_t start,
			     unsigned long len)
{
	unsigned long vm_len, pfn, pages;

	/* Check that the physical memory area passed in looks valid */
	if (start + len < start)
		return -EINVAL;
	/*
	 * You *really* shouldn't map things that aren't page-aligned,
	 * but we've historically allowed it because IO memory might
	 * just have smaller alignment.
	 */
	len += start & ~PAGE_MASK;
	pfn = start >> PAGE_SHIFT;
	pages = (len + ~PAGE_MASK) >> PAGE_SHIFT;
	if (pfn + pages < pfn)
		return -EINVAL;

	/* Can we fit all of the mapping? */
	vm_len = vma->vm_end - vma->vm_start;
	if (vm_len >> PAGE_SHIFT > pages)
		return -EINVAL;

	/* Ok, let it rip */
	return io_remap_pfn_range(vma, vma->vm_start, pfn, vm_len,
				  vma->vm_page_prot);
}

/**
 * ucxi_mmap() - create a new mmap region
 *
 * @filp: file pointer
 * @vma: virtual memory description
 *
 * Return zero if the mmap is OK. Otherwise, return an negative errno.
 */
int ucxi_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct user_client *client = filp->private_data;
	unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
	unsigned long size = vma->vm_end - vma->vm_start;
	struct cxi_mmap_info *ip, *pp;
	int ret;
	const pgprot_t pgprot_ro = PAGE_READONLY;

	/*
	 * Search the device's list of objects waiting for a mmap call.
	 * Normally, this list is very short since a call to create a
	 * CQ, EQ, or CT is soon followed by a call to mmap().
	 */
	spin_lock_bh(&client->pending_lock);
	list_for_each_entry_safe(ip, pp, &client->pending_mmaps,
				 pending_mmaps) {
		if ((__u64)offset != ip->mminfo.offset)
			continue;

		if (size != ip->mminfo.size) {
			pr_err("mmap region has not the right size!\n");
			spin_unlock_bh(&client->pending_lock);
			ret = -EINVAL;
			goto done;
		}

		goto found_it;
	}

	spin_unlock_bh(&client->pending_lock);
	pr_warn("unable to find pending mmap info\n");
	ret = -EINVAL;
	goto done;

found_it:
	list_del_init(&ip->pending_mmaps);
	spin_unlock_bh(&client->pending_lock);

	switch (ip->mmap_type) {
	case MMAP_PHYSICAL:
		if (ip->wc)
#ifdef CONFIG_ARM64
			if (static_branch_unlikely(&avoid_writecombine)) {
				vma->vm_page_prot =
					pgprot_device(vma->vm_page_prot);
				pr_warn("Broken PCIe write-combine detected on this platform, LL_RING will use slow workaround and should be avoided in userspace.");
			} else
#endif
			vma->vm_page_prot =
					pgprot_writecombine(vma->vm_page_prot);
		else
			vma->vm_page_prot =
					pgprot_noncached(vma->vm_page_prot);
		ret = ucxi_iomap_memory(vma, ip->phys, ip->mminfo.size);
		pr_debug("ret %d from ucxi_iomap_memory\n", ret);
		break;
	case MMAP_LOGICAL:
		ret = remap_pfn_range(vma, vma->vm_start,
				      page_to_pfn(ip->pages), ip->mminfo.size,
				      PAGE_SHARED);
		break;
	case MMAP_VIRTUAL:
		ret = remap_vmalloc_range(vma, ip->vma_addr, 0);
		break;
	case MMAP_LOGICAL_RO:
		if (vma->vm_page_prot.pgprot != pgprot_ro.pgprot) {
			ret = -EPERM;
			pr_err("mmap: requires read-only!\n");
			goto done;
		}
		ret = remap_pfn_range(vma, vma->vm_start,
				      page_to_pfn(ip->pages), ip->mminfo.size,
				      PAGE_READONLY);
		break;
	default:
		ret = -EINVAL;
		pr_err("mmap: invalid type %d!\n", ip->mmap_type);
		goto done;
	}

	vma->vm_ops = &cxi_vm_ops;
	vma->vm_private_data = ip;

done:
	if (ret == 0) {
		ip->vm_start = vma->vm_start;
		ip->vm_end = vma->vm_end;
		if (ip->obj)
			atomic_inc(&ip->obj->mappings);
	}
	return ret;
}

/*
 * Allocate information for cxi_mmap
 */
void fill_mmap_info(struct user_client *client, struct cxi_mmap_info *mminfo,
		    unsigned long addr, size_t size, enum cxiu_mmap_type type)
{
	size = PAGE_ALIGN(size);

	spin_lock_bh(&client->mmap_offset_lock);

	if (client->mmap_offset == 0)
		client->mmap_offset = ALIGN(PAGE_SIZE, SHMLBA);

	mminfo->mminfo.offset = client->mmap_offset;
	client->mmap_offset += ALIGN(size, SHMLBA);

	spin_unlock_bh(&client->mmap_offset_lock);

	INIT_LIST_HEAD(&mminfo->pending_mmaps);
	mminfo->mminfo.size = size;
	mminfo->mmap_type = type;
	switch (type) {
	case MMAP_PHYSICAL:
		mminfo->phys = addr;
		break;
	case MMAP_LOGICAL:
	case MMAP_LOGICAL_RO:
		mminfo->pages = (void *)addr;
		break;
	case MMAP_VIRTUAL:
		mminfo->vma_addr = (void *)addr;
		break;
	}
}
