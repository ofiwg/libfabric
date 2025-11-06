// SPDX-License-Identifier: GPL-2.0
/* Copyright 2022 Hewlett Packard Enterprise Development LP */

#ifdef CONFIG_AMD_IOMMU

#include <linux/module.h>
#include <linux/amd-iommu.h>

#include "cass_core.h"

/*
 * ATS invalidate interface for Cassini 1.1
 */

typedef void (*amd_iommu_invalidate_range)(struct pci_dev *pdev,
					   int pasid, unsigned long start,
					   unsigned long end);
int amd_iommu_set_invalidate_range_cb(struct pci_dev *pdev,
				      amd_iommu_invalidate_range cb);

/**
 * cass_ats_invalidate_range_cb() - Callback from the AMD IOMMU to invalidate
 *				    and address range.
 *
 * @pdev: PCI device
 * @pasid: The process address space ID.
 * @start: Start of address range
 * @end: End of address range
 */
static void cass_ats_invalidate_range_cb(struct pci_dev *pdev, int pasid,
					 unsigned long start, unsigned long end)
{

	struct cass_dev *hw = pci_get_drvdata(pdev);
	struct cass_ac *cac;

	cac = hw->cac_table[pasid];
	if (!cac)
		return;

	pr_debug("pasid:%d start:%lx end:%lx\n", pasid, start, end);

	cass_invalidate_range(cac, start, end - start);
	atomic_inc(&hw->dcpl_ats_mn_inval);
}

/**
 * cass_amd_iommu_inval_cb_init() - Initialize the invalidation callback.
 *
 * @pdev: PCI device
 */
void cass_amd_iommu_inval_cb_init(struct pci_dev *pdev)
{
	int (*set_inval_range_cb)(struct pci_dev *pdev,
				  amd_iommu_invalidate_range cb);
	struct cass_dev *hw = pci_get_drvdata(pdev);
	int ret;

	/* Only set up invalidation callback for Cassini 1 */
	if (!cass_version(hw, CASSINI_1))
		return;

	set_inval_range_cb = symbol_get(amd_iommu_set_invalidate_range_cb);
	if (set_inval_range_cb) {
		ret = set_inval_range_cb(pdev, cass_ats_invalidate_range_cb);
		if (ret) {
			dev_WARN(&pdev->dev,
				 "Error setting invalidation callback (%d)\n",
				 ret);
		} else {
			hw->ats_c1_odp_enable = true;
		}

		symbol_put(amd_iommu_set_invalidate_range_cb);
	}
}

#endif /* CONFIG_AMD_IOMMU */
