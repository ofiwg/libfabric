// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/*
 * Address Translation Unit (ATU) management
 * ATS functionality
 */

#include <linux/pci.h>
#include <linux/types.h>
#include <linux/iommu.h>
#include <linux/amd-iommu.h>

#include "cass_core.h"

/* PCIe IP PF0_PORT_LOGIC register */
#define SYMBOL_TIMER_FILTER_1_OFF 0x71c
#define CX_FLT_MASK_CPL_LEN_MATCH (1 << 26)
#define FILTER_MASK_2_OFF 0x720
#define CX_FLT_UNMASK_ATS_SPECIFIC_RULES (1 << 10)

#define ATU_NUM_PASIDS ATU_PHYS_AC
#define MAX_PAGE_REQS 512

/* PPR_FAULT_EXEC was removed with the iommu_v2 module */
#ifdef PPR_FAULT_EXEC

static bool enable_ats = true;
module_param(enable_ats, bool, 0644);
MODULE_PARM_DESC(enable_ats, "Enable Address Translation Services (PCIe ATS)");

static bool odp_mode_nic_pri;
module_param(odp_mode_nic_pri, bool, 0644);
MODULE_PARM_DESC(odp_mode_nic_pri,
		 "ODP mode used for ATS. ATS_PRS:0, NIC_PRI:1");

/**
 * cass_bind_ac() - Bind an address context to the current task.
 *
 * @pdev: pci device
 * @index: Use the AC index as the Process address space ID.
 *
 * @return: 0 on success
 */
static int cass_bind_ac(struct pci_dev *pdev, int index)
{
	return amd_iommu_bind_pasid(pdev, index, current);
}

/**
 * cass_unbind_ac() - Unbind an address context. Called during AC cleanup.
 *
 * @pdev: pci device
 * @pasid: The AC index is used as the Process address space ID.
 */
void cass_unbind_ac(struct pci_dev *pdev, int pasid)
{
	if (!enable_ats)
		return;

	amd_iommu_unbind_pasid(pdev, pasid);
}

void cass_iommu_fini(struct pci_dev *pdev)
{
	if (!enable_ats)
		return;

	amd_iommu_free_device(pdev);
}

/**
 * cass_iommu_init() - Initialize the AMD IOMMU for the physical device
 *
 * If the AMD IOMMU is not enabled or available, ATS mode is not
 * supported.
 *
 * @hw: Cassini device
 */
void cass_iommu_init(struct cass_dev *hw)
{
	int ret;
	int pos;
	u32 filter_mask;
	u32 max_requests;
	struct pci_dev *pdev = hw->cdev.pdev;
	struct amd_iommu_device_info info;
	const u32 flags_req = AMD_IOMMU_DEVICE_FLAG_ATS_SUP |
				AMD_IOMMU_DEVICE_FLAG_PRI_SUP |
				AMD_IOMMU_DEVICE_FLAG_PASID_SUP;

	if (!enable_ats)
		return;

	ret = amd_iommu_device_info(pdev, &info);
	if (ret < 0)
		return;

	if ((info.flags & flags_req) != flags_req) {
		dev_WARN(&pdev->dev,
			 "Error: missing required iommu flag(s) %s %s %s %x\n",
			 (info.flags & AMD_IOMMU_DEVICE_FLAG_ATS_SUP) ?
				"" : "ATS",
			 (info.flags & AMD_IOMMU_DEVICE_FLAG_PRI_SUP) ?
				"" : "PRI",
			 (info.flags & AMD_IOMMU_DEVICE_FLAG_PASID_SUP) ?
				"" : "PASID",
			 info.flags);
		return;
	}

	if (info.max_pasids < ATU_NUM_PASIDS) {
		dev_WARN(&pdev->dev,
			 "Error max_pasids is %d, required %d\n",
			 info.max_pasids, ATU_NUM_PASIDS);
		return;
	}

	ret = amd_iommu_init_device(pdev, ATU_NUM_PASIDS);
	if (ret < 0) {
		dev_WARN(&pdev->dev,
			 "Error initializing iommu device (%d)\n", ret);
		return;
	}

	cass_amd_iommu_inval_cb_init(pdev);


	/*
	 * Mask length match for completions.
	 * The PCI interface checks if the completion length matches
	 * the requested length. Unfortunately, this is not the case for
	 * ATS transactions as only pages that are present are returned
	 * so enable Mask length match for Cassini 1.
	 */
	if (cass_version(hw, CASSINI_1)) {
		pci_read_config_dword(pdev, SYMBOL_TIMER_FILTER_1_OFF,
				      &filter_mask);
		filter_mask |= CX_FLT_MASK_CPL_LEN_MATCH;
		pci_write_config_dword(pdev, SYMBOL_TIMER_FILTER_1_OFF,
				       filter_mask);
	}

	/*
	 * Enable CX_FLT_UNMASK_ATS_SPECIFIC_RULES for C2.
	 *
	 * Cassini 2 has a new capability which uses
	 * CX_FLT_UNMASK_ATS_SPECIFIC_RULES. The PCI controller used by
	 * C2 has a reference section which states the following:
	 * Lower Address is not checked for Cpls related to ATS Requests.
	 * An ATS-related Cpl completes the request if it has a Byte Count
	 * that is equal to four times the Length field.
	 */
	if (cass_version(hw, CASSINI_2)) {
		pci_read_config_dword(pdev, FILTER_MASK_2_OFF, &filter_mask);
		filter_mask |= CX_FLT_UNMASK_ATS_SPECIFIC_RULES;
		pci_write_config_dword(pdev, FILTER_MASK_2_OFF, filter_mask);
	}

	/*
	 * The AMD IOMMU is currently hardcoding max requests to 32.
	 * Set to MAX_PAGE_REQS.
	 */
	pos = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_PRI);
	if (!pos) {
		dev_WARN(&pdev->dev, "Error setting PRI max requests\n");
		return;
	}

	pci_read_config_dword(pdev, pos + PCI_PRI_MAX_REQ, &max_requests);
	max_requests = min_t(u32, max_requests, (u32)MAX_PAGE_REQS);
	pdev->pri_reqs_alloc = max_requests;
	pci_write_config_dword(pdev, pos + PCI_PRI_ALLOC_REQ, max_requests);
}

/**
 * cass_ats_init() - Initialize an AC using either ATS or ATS passthrough
 *                   and bind the AC to the current task
 *
 * @lni_priv: Private LNI struct
 * @m_opts: User map options
 * @cac:    Cassini AC struct
 *
 * @return: 0 on success, negative value on failure
 */
int cass_ats_init(struct cxi_lni_priv *lni_priv,
		  struct ac_map_opts *m_opts,
		  struct cass_ac *cac)
{
	int ret;
	int pasid = 0;
	bool privileged = !(m_opts->flags & CXI_MAP_USER_ADDR);
	union c_atu_cfg_ac_table *ac = &cac->cfg_ac;
	struct cxi_dev *cdev = lni_priv->dev;

	if (!enable_ats)
		return -EOPNOTSUPP;

	ret = cass_bind_ac(cdev->pdev, cac->ac.acid);
	if (ret)
		return ret;

	pasid = cac->ac.acid;

	if (m_opts->flags & CXI_MAP_ATS_PT)
		ac->ta_mode = C_ATU_PASSTHROUGH_MODE;
	else if (m_opts->flags & CXI_MAP_ATS_DYN)
		ac->ta_mode = C_ATU_ATS_DYNAMIC_MODE;
	else
		ac->ta_mode = C_ATU_ATS_MODE;

	m_opts->iova = m_opts->va_start;
	cac->iova_base = 0;
	cac->iova_len = m_opts->flags & CXI_MAP_USER_ADDR ?
				TASK_SIZE : ULONG_MAX;
	cac->iova_end = cac->iova_base + cac->iova_len - 1;

	ac->cntr_pool_id = 1;
	ac->ats_vf_en = 0;
	ac->ats_vf_num = 0;
	ac->ats_no_write = 0;
	ac->ats_pasid = pasid;
	ac->ats_pasid_en = !!pasid;
	ac->ats_pasid_er = 0;
	ac->ats_pasid_pmr = !!privileged;
	ac->mem_base = cac->iova_base >> ATU_CFG_AC_TABLE_MB_SHIFT;
	ac->mem_size = cac->iova_len >> C_ADDR_SHIFT;
	ac->odp_mode = odp_mode_nic_pri ?
			C_ATU_ODP_MODE_NIC_PRI : C_ATU_ODP_MODE_ATS_PRS;

	return 0;
}

/**
 * cass_ats_md_init() - Pin ATS pages or set up notifier
 *
 * When using ODP with NIC_PRI mode, the mmu notifier will be initialized
 * for this descriptor. If pinning, just pin the pages.
 *
 * @md_priv: Private memory descriptor
 * @m_opts:  User options containing the IOVA range
 *
 * @return: Success or -ENOMEM
 */
int cass_ats_md_init(struct cxi_md_priv *md_priv,
		     const struct ac_map_opts *m_opts)
{
	int ret;
	int write = m_opts->flags & CXI_MAP_WRITE;
	int npages = md_priv->md.len >> md_priv->md.page_shift;
	struct page **pages;
	size_t size = npages * sizeof(*pages);

	if (!enable_ats)
		return -EOPNOTSUPP;

	if (!(m_opts->flags & CXI_MAP_PIN)) {
		if (odp_mode_nic_pri)
			return cass_mmu_notifier_insert(md_priv, m_opts);

		return 0;
	}

	pages = kvmalloc(size, GFP_KERNEL);
	if (!pages)
		return -ENOMEM;

	ret = cass_pin(md_priv->cac, pages, npages, md_priv->md.va, write);

	if (ret) {
		kvfree(pages);
		return ret;
	}

	md_priv->pages = pages;

	return ret;
}

#else
void cass_unbind_ac(struct pci_dev *pdev, int pasid)
{
}

void cass_iommu_fini(struct pci_dev *pdev)
{
}

void cass_iommu_init(struct cass_dev *hw)
{
}

int cass_ats_init(struct cxi_lni_priv *lni_priv,
		  struct ac_map_opts *m_opts,
		  struct cass_ac *cac)
{
	return -ENODEV;
}

int cass_ats_md_init(struct cxi_md_priv *md_priv,
		     const struct ac_map_opts *m_opts)
{
	return -ENODEV;
}
#endif /* CONFIG_AMD_IOMMU */
