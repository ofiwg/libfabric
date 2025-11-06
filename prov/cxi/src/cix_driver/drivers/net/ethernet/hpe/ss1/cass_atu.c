// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

/* Address Translation Unit (ATU) management */

#include <linux/interrupt.h>
#include <linux/types.h>
#include <linux/hugetlb.h>
#include <linux/iova.h>
#include <linux/bvec.h>
#include <linux/iopoll.h>
#include <linux/workqueue.h>
#include <linux/random.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

int more_debug;
module_param(more_debug, int, 0644);
MODULE_PARM_DESC(more_debug, "More debug");

static int default_ptg_mode = C_ATU_PTG_MODE_DBL_C;
module_param(default_ptg_mode, int, 0644);
MODULE_PARM_DESC(default_ptg_mode, "Default PTG Mode");

/* Invalidation interval is 2^(inval_process_interval+1) - 10 is ~2 uS */
static int inval_process_interval = 10;
module_param(inval_process_interval, int, 0644);
MODULE_PARM_DESC(inval_process_interval, "ATS invalidate processing interval");

/* Relaxed ordering bits test */
static bool acxt_mem_rd_ro;
module_param(acxt_mem_rd_ro, bool, 0644);
MODULE_PARM_DESC(acxt_mem_rd_ro, "Memory Read Relaxed Ordering");
static bool acxt_flush_req_ro;
module_param(acxt_flush_req_ro, bool, 0644);
MODULE_PARM_DESC(acxt_flush_req_ro, "Flush Request Relaxed Ordering");
static bool acxt_translation_req_ro;
module_param(acxt_translation_req_ro, bool, 0644);
MODULE_PARM_DESC(acxt_translation_req_ro, "Translation Request Relaxed Ordering");
static bool acxt_fetching_amo_ro;
module_param(acxt_fetching_amo_ro, bool, 0644);
MODULE_PARM_DESC(acxt_fetching_amo_ro, "Fetching AMO Relaxed Ordering");
static bool acxt_ido;
module_param(acxt_ido, bool, 0644);
MODULE_PARM_DESC(acxt_ido, "ID-Based Ordering");

static bool ac_salt = true;
module_param(ac_salt, bool, 0644);
MODULE_PARM_DESC(ac_salt, "Add salt to ACs");

static bool acxt_ph_en;
module_param(acxt_ph_en, bool, 0644);
MODULE_PARM_DESC(acxt_ph_en, "Enable use of Processing Hints and Steering Tag");
static u8 acxt_phints;
module_param(acxt_phints, byte, 0644);
MODULE_PARM_DESC(acxt_phints, "TLP Processing Hints");
static u8 acxt_steering_tag;
module_param(acxt_steering_tag, byte, 0644);
MODULE_PARM_DESC(acxt_steering_tag, "Steering Tag");

bool ats_c1_override;
module_param(ats_c1_override, bool, 0644);
MODULE_PARM_DESC(ats_c1_override, "Override ODP Enable for ATS with C1");

/* Use a larger base page size when huge_shift - page_shift is greater
 * than the maximum page table size (max-threshold) allowed. This allows
 * us to support 256, 512 and 1024 MB Cray hugepage sizes.
 */
static int hugepage_threshold = 1;
module_param(hugepage_threshold, int, 0644);
MODULE_PARM_DESC(hugepage_threshold, "Threshold to increase base page size.");

void cass_atu_fini(struct cass_dev *hw)
{
	cxi_unregister_hw_errors(hw, &hw->atu_err);
	iova_cache_put();
	dma_unmap_single(&hw->cdev.pdev->dev, hw->oxe_dummy_dma_addr,
			 PAGE_SIZE, DMA_TO_DEVICE);
	kfree(hw->oxe_dummy_addr);
	ida_destroy(&hw->atu_table);
	cass_iommu_fini(hw->cdev.pdev);
	cass_nta_pri_fini(hw);
	cass_nta_cq_fini(hw);
	refcount_dec(&hw->refcount);
}

/* Atomically write an Address Context */
static void cass_write_ac(struct cass_dev *hw, union c_atu_cfg_ac_table *ac,
			  unsigned int index, bool user)
{
	union c_pi_cfg_acxt pi_cfg_acxt = {
		.ac_en = ac->context_en,
	};
	struct cass_ac *cac = container_of(ac, struct cass_ac, cfg_ac);

	if (ac->ta_mode != C_ATU_NTA_MODE) {
		pi_cfg_acxt.vf_en = ac->ats_vf_en;
		pi_cfg_acxt.vf_num = ac->ats_vf_num;
		pi_cfg_acxt.pasid = ac->ats_pasid;
		pi_cfg_acxt.pasid_en = ac->ats_pasid_en;
		pi_cfg_acxt.pasid_er = ac->ats_pasid_er;
		pi_cfg_acxt.pasid_pmr = ac->ats_pasid_pmr;
	}

	pi_cfg_acxt.mem_rd_ro = acxt_mem_rd_ro;
	pi_cfg_acxt.flush_req_ro = acxt_flush_req_ro;
	pi_cfg_acxt.translation_req_ro = acxt_translation_req_ro;
	pi_cfg_acxt.fetching_amo_ro = acxt_fetching_amo_ro;
	pi_cfg_acxt.ido = acxt_ido;

	if (acxt_ph_en && user) {
		pi_cfg_acxt.ph_en = acxt_ph_en;
		pi_cfg_acxt.phints = acxt_phints;
		pi_cfg_acxt.steering_tag = acxt_steering_tag;
	}

	if (ac_salt) {
		u32 rand = get_random_u32();

		ac->ptn_salt = rand;
		ac->idx_salt = rand >> 2;
	}

	spin_lock(&hw->atu_shadow_lock);
	cass_config_ac(hw, index, ac);
	spin_unlock(&hw->atu_shadow_lock);

	/* WORKAROUND: Cassini ERRATA-3277
	 * When disabling an AC, invalidate before disabling the ACXT.
	 */
	if (cass_version(hw, CASSINI_1) && !ac->context_en)
		cass_invalidate_range(cac, 0, ATUCQ_INVALIDATE_ALL);

	cass_config_pi_acxt(hw, index, &pi_cfg_acxt);
}

/* Callback registered with the interrupt error handler, executed when
 * C_ATU_ERR_FLG.PRB_EXPIRED error interrupt is detected.
 */
static void cass_prb_exp_cb(struct cass_dev *hw, unsigned int irq,
			    bool is_ext, unsigned int bitn)
{
	queue_work(hw->prb_wq, &hw->prb_work);
}

int cass_atu_init(struct cass_dev *hw)
{
	int i;
	int ret;
	union c_atu_cfg_nta cfg_nta = {
		.acid = ATU_NTA_DEF_PI_ACXT,
	};
	union c_atu_cfg_ac_table ac = {
		.context_en = 1,
		.ta_mode = C_ATU_PASSTHROUGH_MODE,
	};
	union c_oxe_cfg_dummy_addr oxe_dummy = {
		.ac = ATU_PHYS_AC
	};
	union c_atu_cfg_ats cfg_ats;

	ac.mem_size = ~ac.mem_size;

	hw->oxe_dummy_addr = kcalloc(1, PAGE_SIZE, GFP_KERNEL);
	if (hw->oxe_dummy_addr == NULL)
		return -ENOMEM;

	ida_init(&hw->atu_table);
	refcount_inc(&hw->refcount);
	spin_lock_init(&hw->atu_shadow_lock);
	mutex_init(&hw->odpq_mutex);
	atomic_set(&hw->stats.md, 0);
	atomic_set(&hw->atu_error_inject, 0);
	atomic_set(&hw->atu_odp_requests, 0);
	atomic_set(&hw->atu_odp_fails, 0);
	atomic_set(&hw->atu_prb_expired, 0);
	atomic_set(&hw->dcpl_md_clear_inval, 0);
	atomic_set(&hw->dcpl_nta_mn_inval, 0);
	atomic_set(&hw->dcpl_ats_mn_inval, 0);
	atomic_set(&hw->dcpl_comp_wait, 0);
	atomic_set(&hw->dcpl_entered, 0);
	atomic_set(&hw->dcpl_ixe_cntr_0, 0);
	atomic_set(&hw->dcpl_ixe_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_ee_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_ee_cntr_stuck, 0);
	atomic_set(&hw->dcpl_oxe_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_oxe_cntr_stuck, 0);
	atomic_set(&hw->dcpl_ixe_cntr_stuck, 0);
	atomic_set(&hw->dcpl_success, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_62, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_62_count, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_64, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_64_count, 0);
	atomic_set(&hw->dcpl_ibw_active_stuck, 0);
	atomic_set(&hw->dcpl_ibw_idle_wait, 0);
	atomic_set(&hw->dcpl_ibw_cntr_is_0, 0);
	atomic_set(&hw->dcpl_ibw_issued, 0);
	hw->dcpl_max_time = 0;
	for (i = 0; i < DBINS; i++)
		hw->dcpl_time[i] = 0;
	hw->pri_max_md_time = 0;
	hw->pri_max_fault_time = 0;
	for (i = 0; i < MDBINS; i++)
		hw->pri_md_time[i] = 0;
	for (i = 0; i < FBINS; i++)
		hw->pri_fault_time[i] = 0;


	cass_write_ac(hw, &ac, ATU_PHYS_AC, false);
	cass_write(hw, C_ATU_CFG_NTA, &cfg_nta, sizeof(union c_atu_cfg_nta));

	hw->oxe_dummy_dma_addr = dma_map_single(&hw->cdev.pdev->dev,
						hw->oxe_dummy_addr, PAGE_SIZE,
						DMA_TO_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, hw->oxe_dummy_dma_addr)) {
		ret = -ENOMEM;
		goto free_dummy;
	}
	oxe_dummy.subaddr = hw->oxe_dummy_dma_addr >> C_ADDR_SHIFT;
	cass_write(hw, C_OXE_CFG_DUMMY_ADDR, &oxe_dummy,
		   sizeof(union c_oxe_cfg_dummy_addr));

	cass_read(hw, C_ATU_CFG_ATS, &cfg_ats.qw, sizeof(cfg_ats));

	/* The simulator runs much slower so we need to change the
	 * invalidate processing interval to a lower value.
	 * The AMD IOMMU (in Epyc processors earlier than the 7002 series)
	 * has bug where it returns multiple translations
	 * with the same address if the page size is larger than the
	 * STU size. Setting truncate_gt_stu_rsp is a workaround.
	 * Cassini ERRATA-2431
	 */
	if (HW_PLATFORM_Z1(hw)) {
		cfg_ats.inval_process_interval = 5;
		cfg_ats.truncate_gt_stu_rsp = 1;
	}

	/* The inval_process_interval default causes IOMMU timeouts. */
	if (HW_PLATFORM_ASIC(hw))
		cfg_ats.inval_process_interval = inval_process_interval;

	if (HW_PLATFORM_Z1(hw) || HW_PLATFORM_ASIC(hw))
		cass_write(hw, C_ATU_CFG_ATS, &cfg_ats, sizeof(cfg_ats));

	if (!cass_version(hw, CASSINI_1))
		odp_sw_decouple = false;

	if (cass_version(hw, CASSINI_1)) {
		/* Errata 3117 */
		union c1_atu_cfg_odp_decouple odp_decouple;

		cass_read(hw, C1_ATU_CFG_ODP_DECOUPLE, &odp_decouple,
			  sizeof(odp_decouple));
		odp_decouple.flush_req_delay = ATU_FLUSH_REQ_DELAY;
		odp_decouple.enable = odp_sw_decouple ? 0 : 1;
		cass_write(hw, C1_ATU_CFG_ODP_DECOUPLE, &odp_decouple,
			   sizeof(odp_decouple));
	}

	ret = cass_nta_cq_init(hw);
	if (ret)
		goto cq_init_fail;

	ret = cass_nta_pri_init(hw);
	if (ret)
		goto pri_init_fail;

	iova_cache_get();
	cass_iommu_init(hw);

	/* Register callback for C_ATU_ERR_FLG.PRB_EXPIRED bit */
	hw->atu_err.irq = C_ATU_IRQA_MSIX_INT;
	hw->atu_err.is_ext = false;
	hw->atu_err.err_flags.atu.prb_expired = 1;
	hw->atu_err.cb = cass_prb_exp_cb;
	cxi_register_hw_errors(hw, &hw->atu_err);

	return 0;

pri_init_fail:
	cass_nta_cq_fini(hw);

cq_init_fail:
	refcount_dec(&hw->refcount);
	dma_unmap_single(&hw->cdev.pdev->dev, hw->oxe_dummy_dma_addr,
			 PAGE_SIZE, DMA_TO_DEVICE);
free_dummy:
	kfree(hw->oxe_dummy_addr);

	return ret;
}

static void cass_md_list_free(struct cass_ac *cac)
{
	struct cxi_md_priv *tmp;
	struct cxi_md_priv *le;

	mutex_lock(&cac->md_mutex);

	list_for_each_entry_safe(le, tmp, &cac->md_list, md_entry) {
		list_del(&le->md_entry);
		refcount_dec(&cac->refcount);
		cass_md_clear(le, false, true);
		kfree(le);
	}

	mutex_unlock(&cac->md_mutex);
}

/**
 * cass_cac_free - Free an Address Context
 *
 * @hw:  Cassini device
 * @cac: Address Context container
 */
static void cass_cac_free(struct cass_dev *hw, struct cass_ac *cac)
{
	int acid = cac->ac.acid;

	cxidev_WARN_ONCE(&hw->cdev, !refcount_dec_and_test(&cac->refcount),
			 "Resource leaks - AC refcount not zero: %d\n",
			 refcount_read(&cac->refcount));

	kfree(cac);
	hw->cac_table[acid] = NULL;
	ida_simple_remove(&hw->atu_table, acid);
}

static void cass_ac_init(struct cass_ac *cac, const struct ac_map_opts *m_opts)
{
	union c_atu_cfg_ac_table *ac = &cac->cfg_ac;
	struct cass_dev *hw = container_of(cac->lni_priv->dev,
					   struct cass_dev, cdev);

	ac->pg_table_size = cac->huge_shift - cac->page_shift;
	ac->base_pg_size = cac->page_shift - C_ADDR_SHIFT;
	ac->ptg_mode = cac->ptg_mode;
	ac->do_not_cache = !!(cac->flags & CXI_MAP_NOCACHE);
	ac->odp_en = !(cass_version(hw, CASSINI_1_0) ||
		       (cac->flags & CXI_MAP_PIN));
	ac->do_not_filter = hw->ac_filter_disabled;
	ac->context_en = 1;

	pr_debug("page_shift:%d huge_shift:%d base_pg_sz:%d pg_table_sz:%d\n",
		 cac->page_shift, cac->huge_shift, ac->base_pg_size,
		 ac->pg_table_size);
}

/**
 * cass_lac_free() - Unmap an AC from the LNI's resource group.
 *
 * @lni_priv: the Logical Network Interface
 * @lac: LAC to free
 */
static void cass_lac_free(struct cxi_lni_priv *lni_priv, int lac)
{
	struct cass_dev *hw = container_of(lni_priv->dev, struct cass_dev, cdev);

	cass_set_acid(hw, lni_priv->lni.rgid, lac, C_AC_NONE);
	cass_lac_put(hw, lni_priv->lni.rgid, lac);
}

/**
 * cass_ac_alloc() - Allocate an Address Context
 *
 * Once the ACID is written to hardware, it cannot be freed until the LNI
 * is freed.
 *
 * @lni_priv: the Logical Network Interface
 * @m_opts: User map options
 */
static struct cass_ac *cass_ac_alloc(struct cxi_lni_priv *lni_priv,
				     struct ac_map_opts *m_opts)
{
	int ret;
	struct cass_ac *cac;
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	/* ac->pg_table_size is limited to 4 bits and a minimum of 8 entries */
	if ((m_opts->huge_shift - m_opts->page_shift > MAX_PG_TABLE_SIZE) ||
	    ((m_opts->ptg_mode != C_ATU_PTG_MODE_SGL) &&
	     ((m_opts->huge_shift - m_opts->page_shift) < MIN_PG_TABLE_SIZE))) {
		pr_debug("Invalid huge_shift:%d/page_shift:%d combination\n",
			 m_opts->huge_shift, m_opts->page_shift);
		return ERR_PTR(-EINVAL);
	}

	/* Check the associated service to see if this AC can be allocated */
	ret = cxi_rgroup_alloc_resource(lni_priv->rgroup, CXI_RESOURCE_AC);
	if (ret)
		return ERR_PTR(ret);

	cac = kzalloc(sizeof(*cac), GFP_KERNEL);
	if (!cac) {
		ret = -ENOMEM;
		goto dec_rsrc_use;
	}

	/* AC_NONE (0) is invalid, ATU_PHYS_AC is reserved for physical
	 * mappings
	 */
	ret = ida_simple_get(&hw->atu_table, 1, ATU_PHYS_AC, GFP_KERNEL);
	if (ret < 0)
		goto cac_free;

	cac->ac.acid = ret;

	ret = cass_lac_get(hw, lni_priv->lni.rgid);
	if (ret < 0)
		goto ac_put;

	cac->ac.lac = ret;

	refcount_set(&cac->refcount, 1);
	mutex_init(&cac->ac_mutex);
	cac->lni_priv = lni_priv;
	cac->flags = m_opts->flags;
	cac->ptg_mode = m_opts->ptg_mode;
	cac->page_shift = m_opts->page_shift;
	cac->huge_shift = m_opts->huge_shift;
	cac->hugepage_test = m_opts->hugepage_test;

	cass_ac_init(cac, m_opts);

	if (m_opts->flags & CXI_MAP_ATS)
		ret = cass_ats_init(lni_priv, m_opts, cac);
	else
		ret = cass_nta_init(lni_priv, m_opts, cac);

	if (ret)
		goto lac_put;

	hw->cac_table[cac->ac.acid] = cac;

	cass_write_ac(hw, &cac->cfg_ac, cac->ac.acid,
		      m_opts->flags & CXI_MAP_USER_ADDR);
	cass_set_acid(hw, lni_priv->lni.rgid, cac->ac.lac, cac->ac.acid);

	INIT_LIST_HEAD(&cac->md_list);
	mutex_init(&cac->md_mutex);

	list_add_tail(&cac->list, &lni_priv->ac_list);
	refcount_inc(&lni_priv->refcount);

	atomic_inc(&hw->stats.ac);

	ac_debugfs_create(cac->ac.acid, cac, hw, lni_priv);

	return cac;

lac_put:
	cass_lac_put(hw, lni_priv->lni.rgid, cac->ac.lac);
ac_put:
	ida_simple_remove(&hw->atu_table, cac->ac.acid);
cac_free:
	kfree(cac);
dec_rsrc_use:
	cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_AC);

	pr_debug("failed rc %d\n", ret);
	return ERR_PTR(ret);
}

static void cass_ac_free(struct cass_dev *hw, struct cass_ac *cac)
{
	struct cxi_lni_priv *lni_priv = cac->lni_priv;
	int pasid = cac->cfg_ac.ats_pasid;

	pr_debug("freeing ac %d\n", cac->ac.acid);

	debugfs_remove(cac->lni_dir);
	debugfs_remove_recursive(cac->debug_dir);

	atomic_dec(&hw->stats.ac);
	cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_AC);

	cass_md_list_free(cac);
	cass_nta_iova_fini(cac);

	if (cac->nta) {
		cass_l1_tables_free(hw, cac->nta);
		cass_invalidate_range(cac, 0, ATUCQ_INVALIDATE_ALL);
		cass_nta_free(hw, cac->nta);
	}

	memset(&cac->cfg_ac, 0, sizeof(cac->cfg_ac));
	cass_write_ac(hw, &cac->cfg_ac, cac->ac.acid, false);

	if ((cac->flags & CXI_MAP_ATS) && pasid)
		cass_unbind_ac(lni_priv->dev->pdev, pasid);

	cass_lac_free(lni_priv, cac->ac.lac);
	cass_cac_free(hw, cac);
	refcount_dec(&lni_priv->refcount);
}

/**
 * cass_acs_disable() - Disable all the ACs of an LNI
 *
 * The LNI is getting released. This is the first part of the cleanup.
 *
 * @lni_priv: the Logical Network Interface
 */
void cass_acs_disable(struct cxi_lni_priv *lni_priv)
{
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cass_ac *cac;

	mutex_lock(&lni_priv->ac_list_mutex);

	list_for_each_entry(cac, &lni_priv->ac_list, list) {
		cac->cfg_ac.context_en = 0;
		cass_write_ac(hw, &cac->cfg_ac, cac->ac.acid, false);
	}

	mutex_unlock(&lni_priv->ac_list_mutex);

	cass_flush_pci(hw);
}

/**
 * cass_acs_free() - Free all ACs associated with an LNI
 *
 * The ac_list is used by cass_pri_fault() and cass_sync_pagetables().
 * LNI cleanup disables all the ACs associated with the LNI and then calls
 * inbound wait so cass_pri_fault() will not be called while cleaning up.
 * cass_hmm_fini() will remove this process from the hmm mirrors list so
 * cass_sync_pagetables() will not be called while cleaning up.
 *
 * @lni_priv: the Logical Network Interface
 */
void cass_acs_free(struct cxi_lni_priv *lni_priv)
{
	struct cass_ac *cac;
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	mutex_lock(&lni_priv->ac_list_mutex);

	while ((cac = list_first_entry_or_null(&lni_priv->ac_list,
					       struct cass_ac, list))) {
		list_del(&cac->list);
		mutex_unlock(&lni_priv->ac_list_mutex);
		cass_ac_free(hw, cac);
		mutex_lock(&lni_priv->ac_list_mutex);
	}

	mutex_unlock(&lni_priv->ac_list_mutex);

	if (dev->is_physfn)
		cass_flush_pci(hw);
}

/**
 * cass_iova_alloc() - Allocate an IO virtual address in an Address Context
 *
 * Find an AC with matching characteristics and allocate an IOVA.
 *
 * @lni_priv: the Logical Network Interface
 * @m_opts: User map options
 */
static struct cass_ac *cass_iova_alloc(struct cxi_lni_priv *lni_priv,
				       struct ac_map_opts *m_opts)
{
	u64 iova;
	struct list_head *ptr;
	u32 masked_flags = m_opts->flags & ATU_FLAGS_MASK;

	list_for_each(ptr, &lni_priv->ac_list) {
		struct cass_ac *cac = list_entry(ptr, struct cass_ac, list);
		u32 cac_msk_flags = cac->flags & ATU_FLAGS_MASK;

		if ((m_opts->page_shift == cac->page_shift) &&
				(m_opts->huge_shift == cac->huge_shift) &&
				(m_opts->ptg_mode == cac->ptg_mode) &&
				(masked_flags == cac_msk_flags) &&
				!m_opts->hugepage_test) {
			if (m_opts->flags & CXI_MAP_ATS) {
				iova = m_opts->va_start;
			} else {
				iova = cass_nta_iova_alloc(cac, m_opts);
				if (!iova)
					continue;
			}

			m_opts->iova = iova;

			return cac;
		}
	}

	return ERR_PTR(-ENOSPC);
}

/* Find or allocate an AC matching the options */
static struct cass_ac *get_matching_ac(struct cxi_lni_priv *lni_priv,
				       struct ac_map_opts *m_opts)
{
	struct cass_ac *cac;

	mutex_lock(&lni_priv->ac_list_mutex);

	cac = cass_iova_alloc(lni_priv, m_opts);
	if (IS_ERR(cac))
		cac = cass_ac_alloc(lni_priv, m_opts);

	mutex_unlock(&lni_priv->ac_list_mutex);

	return cac;
}

static void cass_init_md(const struct ac_map_opts *m_opts,
			 struct cxi_md_priv *md_priv)
{
	md_priv->md.iova = m_opts->iova;
	md_priv->md.va = m_opts->va_start;
	md_priv->md.len = m_opts->va_len;
	md_priv->md.page_shift = m_opts->page_shift;
	md_priv->md.huge_shift = m_opts->huge_shift;
	md_priv->md.lac = md_priv->cac->ac.lac;
	md_priv->olen = m_opts->va_len;
	md_priv->flags = m_opts->flags & ~(CXI_MAP_FAULT | CXI_MAP_PREFETCH);

	/* Since we don't mirror the page tables when CXI_MAP_ALLOC_MD
	 * is set, the first call to cxi_update_iov needs to lock.
	 */
	if (m_opts->flags & CXI_MAP_ALLOC_MD)
		md_priv->need_lock = true;
}

/* Add an MD to an LNI */
static void cass_add_md(struct cxi_lni_priv *lni_priv,
			const struct ac_map_opts *m_opts,
			struct cxi_md_priv *md_priv)
{
	struct cass_ac *cac = md_priv->cac;
	struct cass_dev *hw = container_of(cac->lni_priv->dev, struct cass_dev, cdev);

	mutex_lock(&cac->md_mutex);
	list_add_tail(&md_priv->md_entry, &cac->md_list);
	refcount_inc(&cac->refcount);
	atomic_inc(&hw->stats.md);
	mutex_unlock(&cac->md_mutex);
}

/**
 * cass_remove_md() - Remove an MD from its LNI, clear and invalidate
 *                    page table entries.
 *
 * @md_priv: Private memory descriptor
 */
static void cass_remove_md(struct cxi_md_priv *md_priv)
{
	struct cass_ac *cac = md_priv->cac;
	struct cass_dev *hw = container_of(cac->lni_priv->dev, struct cass_dev, cdev);

	mutex_lock(&md_priv->cac->md_mutex);
	list_del(&md_priv->md_entry);
	refcount_dec(&cac->refcount);
	atomic_dec(&hw->stats.md);
	mutex_unlock(&md_priv->cac->md_mutex);

	cass_md_clear(md_priv, true, true);
}

static void cass_md_iova_free(struct cxi_md_priv *md_priv)
{
	struct cass_ac *cac = md_priv->cac;

	if (cac->iovad)
		FREE_IOVA_FAST(cac->iovad, md_priv->md.iova, md_priv->olen);
}

static int cass_bvec(struct cass_dev *hw, const struct iov_iter *iter,
		     size_t *len, u64 *va)
{
	int i;
	int last_seg = iter->nr_segs - 1;
	const struct bio_vec *bvec = iter->bvec;

	*va = (u64)page_address(bvec->bv_page);

	for (i = 0; i < iter->nr_segs; i++) {
		if (i && (bvec->bv_offset & ~PAGE_MASK)) {
			cxidev_err(&hw->cdev, "bvec[%d] offset not aligned %d\n", i,
				   bvec->bv_offset);
			return -EINVAL;
		}

		if (i < last_seg && ((bvec->bv_offset + bvec->bv_len) &
						~PAGE_MASK)) {
			cxidev_err(&hw->cdev, "bvec[%d] contains a hole\n", i);
			return -EINVAL;
		}
		bvec++;
	}

	*len = iter->nr_segs * PAGE_SIZE;

	return 0;
}

static int cass_kvec(struct cass_dev *hw, const struct iov_iter *iter,
		     size_t *len, u64 *va)
{
	int i;
	int alen;
	int tlen = 0;
	int last_seg = iter->nr_segs - 1;
	const struct kvec *kvec = iter->kvec;

	*va = (u64)kvec->iov_base & PAGE_MASK;

	for (i = 0; i < iter->nr_segs; i++) {
		/*
		 * Only the first kvec may be unaligned. The first and last
		 * kvec's lengths can be less that a page but the rest need
		 * to be full pages. The first base + len must be aligned.
		 */
		if (i && ((u64)kvec->iov_base & ~PAGE_MASK)) {
			cxidev_err(&hw->cdev, "kvec[%d] base not aligned %p\n", i,
				   kvec->iov_base);
			return -EINVAL;
		}

		if (i < last_seg && (((u64)kvec->iov_base + kvec->iov_len) &
					~PAGE_MASK)) {
			cxidev_err(&hw->cdev, "kvec[%d] contains a hole\n", i);
			return -EINVAL;
		}

		alen = PAGE_ALIGN(kvec->iov_len);
		tlen += alen;
		kvec++;
	}

	*len = tlen;

	return 0;
}
/* cass_sgtable_is_valid() - Check for invalid sgt
 *
 * Requirements for entries in sgt:
 * one entry:
 *     offset < PAGE_SIZE
 * multiple entries:
 *     first:  offset < PAGE_SIZE, (len + offset) % PAGE_SIZE
 *     middle: offset == 0, len % PAGE_SIZE
 *     last:   offset == 0
 */
static bool cass_sgtable_is_valid(struct cass_dev *hw,
				  const struct sg_table *sgt, size_t *length)
{
	int i;
	struct scatterlist *sg;
	int last_entry = sgt->nents - 1;

	*length = 0;

	for_each_sgtable_dma_sg(sgt, sg, i) {
		size_t len = sg_dma_len(sg);

		if (sg->offset > PAGE_SIZE) {
			cxidev_err(&hw->cdev,
				   "Entry %d offset:%x > PAGE_SIZE\n",
				   i, sg->offset);

			return false;
		}

		/* first of multiple entries */
		if ((!i && i != last_entry && ((len + sg->offset) % PAGE_SIZE)) ||
		    /* last entry */
		    (i && i == last_entry && sg->offset) ||
		    /* middle entries */
		    (i && i < last_entry && (sg->offset ||
					     len % PAGE_SIZE))) {
			cxidev_err(&hw->cdev, "Hole in sg_table at entry %d offset:%x len:%lx\n",
				   i, sg->offset, len);

			return false;
		}

		*length += ALIGN(len + sg->offset, PAGE_SIZE);
	}

	return true;
}

/**
 * cass_mirror_range() - Mirror an address range
 *
 * @md_priv: Private memory descriptor
 * @m_opts:  User options
 *
 * @return: 0 on success or -ENOMEM
 */
int cass_mirror_range(struct cxi_md_priv *md_priv, struct ac_map_opts *m_opts)
{
	if (!(m_opts->flags & CXI_MAP_USER_ADDR))
		return cass_nta_mirror_kern(md_priv, NULL, true);

	if (m_opts->flags & CXI_MAP_ATS)
		return cass_ats_md_init(md_priv, m_opts);

	if (m_opts->flags & CXI_MAP_DEVICE)
		return cass_mirror_device(md_priv, m_opts);

	if (m_opts->hugepage_test)
		return cass_mirror_hp(m_opts, md_priv);

	if (m_opts->flags & CXI_MAP_PIN)
		return cass_pin_mirror(md_priv, m_opts);

	return cass_mmu_notifier_insert(md_priv, m_opts);
}

static void cass_unpin(struct cxi_md_priv *md_priv)
{
	int npages = md_priv->md.len >> PAGE_SHIFT;

	if (!(md_priv->flags & CXI_MAP_PIN))
		return;

	if (!md_priv->pages)
		return;

	unpin_user_pages(md_priv->pages, npages);
	kvfree(md_priv->pages);
	md_priv->pages = NULL;
}

/**
 * cass_md_clear() - Clear a memory descriptor
 *
 * Remove page table entries for NTA mode
 * Unpin ATS or device pinned pages
 *
 * @md_priv: Private memory descriptor
 * @inval: Invalidate this iova range
 * @need_lock: Indicates whether the ac_mutex is required
 */
void cass_md_clear(struct cxi_md_priv *md_priv, bool inval, bool need_lock)
{
	struct cass_dev *hw = container_of(md_priv->cac->lni_priv->dev,
					   struct cass_dev, cdev);

	cass_notifier_cleanup(md_priv);

	if (md_priv->flags & CXI_MAP_ATS)
		return cass_unpin(md_priv);

	/* If inval is false we have been called from cass_md_list_free()
	 * which is only called when the AC is cleaning up so we don't
	 * need to invalidate here as we will invalidate the whole AC range.
	 */
	cass_cond_lock(&md_priv->cac->ac_mutex, need_lock);
	cass_clear_range(md_priv, md_priv->md.iova,
			 md_priv->md.len);
	cass_cond_unlock(&md_priv->cac->ac_mutex, need_lock);

	if (inval) {
		cass_invalidate_range(md_priv->cac, md_priv->md.iova,
				      md_priv->md.len);
		atomic_inc(&hw->dcpl_md_clear_inval);
	}

	if (md_priv->flags & CXI_MAP_DEVICE)
		cass_device_put_pages(md_priv);
	else
		cass_unpin(md_priv);
}

static int largest_hp_size(struct vm_area_struct *vma, u64 end)
{
	int vma_hs;
	int hs = INT_MAX;
	ulong vm_end_prev;
#ifdef VMA_ITERATOR
	VMA_ITERATOR(vmi, vma->vm_mm, vma->vm_start);
#endif

	while (true) {
		if (!is_vm_hugetlb_page(vma))
			return PMD_SHIFT;

		vma_hs = huge_page_shift(hstate_vma(vma));

		hs = vma_hs < hs ? vma_hs : hs;

		if (vma->vm_end >= end)
			return hs;

		vm_end_prev = vma->vm_end;
#ifdef VMA_ITERATOR
		vma = vma_next(&vmi);
#else
		vma = vma->vm_next;
#endif
		/* full range is not backed by vmas */
		if (!vma)
			return PMD_SHIFT;

		/* found a hole */
		if (vma->vm_start > vm_end_prev)
			return PMD_SHIFT;

		/* not in range */
		if (vma->vm_start >= end)
			break;
	}

	return hs;
}

static int align_and_page_shift(struct ac_map_opts *m_opts)
{
	/* To support hugepage sizes larger than 128MB, we need
	 * a larger base page size.
	 */
	if ((m_opts->huge_shift - m_opts->page_shift) >
			(MAX_PG_TABLE_SIZE - hugepage_threshold))
		m_opts->page_shift = m_opts->huge_shift -
				MAX_PG_TABLE_SIZE + hugepage_threshold;

	/* A hugepage is backing this address so align
	 * to and map the full hugepage.
	 */
	return m_opts->huge_shift;
}

int cass_cpu_page_size(struct cass_dev *hw, struct ac_map_opts *m_opts,
		       struct mm_struct *mm, uintptr_t va,
		       const struct cxi_md_hints *hints, int *align_shift)
{
	int ret = 0;
	struct vm_area_struct *vma;

	/* ODP hint from user for the hugepage size */
	if (hints && !hints->page_shift && hints->huge_shift &&
	    !(m_opts->flags & CXI_MAP_PIN & CXI_MAP_FAULT & CXI_MAP_PREFETCH)) {
		m_opts->huge_shift = hints->huge_shift;
		*align_shift = align_and_page_shift(m_opts);

		return 0;
	}

	mmap_read_lock(mm);
	vma = find_vma(mm, va);
	if (!vma) {
		cxidev_warn(&hw->cdev, "No VMA covering 0x%016lx)\n", va);
		ret = -EFAULT;
		goto unlock;
	}

	/* Test arbitrary page sizes supplied by the user. The contiguous
	 * pages will be mmapped from a 1G hugepage and combined into larger
	 * base pages and smaller hugepages.
	 */
	if (hints && hints->page_shift && is_vm_hugetlb_page(vma)) {
		m_opts->huge_shift = huge_page_shift(hstate_vma(vma));

		pr_debug("Testing huge shift:%d bs:%d hs:%d\n",
			 m_opts->huge_shift, hints->page_shift,
			 hints->huge_shift);

		if (hints->huge_shift > m_opts->huge_shift) {
			cxidev_warn(&hw->cdev, "Hints huge_shift (%d) > vma's huge_shift (%d)\n",
				    hints->huge_shift, m_opts->huge_shift);
			ret = -EINVAL;
			goto unlock;
		}

		m_opts->page_shift = hints->page_shift;
		m_opts->huge_shift = hints->huge_shift;
		*align_shift = hints->huge_shift;
		m_opts->hugepage_test = true;
		goto unlock;
	}

	if (is_vm_hugetlb_page(vma)) {
		m_opts->huge_shift = largest_hp_size(vma, m_opts->va_end);
		m_opts->is_huge_page = true;
		*align_shift = align_and_page_shift(m_opts);
	} else {
		int vms_fs = ffsl(vma->vm_start);
		int vml_fs = ffsl(vma->vm_end - vma->vm_start);

		/* Small regions may not be aligned to the default alignment
		 * passed in so limit alignment to the VMA.
		 */
		if (vms_fs < *align_shift || vml_fs < *align_shift)
			*align_shift = min_t(int, vms_fs, vml_fs);
	}

unlock:
	mmap_read_unlock(mm);

	return ret;
}

/**
 * cass_align_start_len() - Align the va and len to align_shift page size
 *
 * @m_opts: User map options
 * @va: Virtual address
 * @len: Length of address
 * @align_shift: Power of two to align to
 */
void cass_align_start_len(struct ac_map_opts *m_opts, uintptr_t va, size_t len,
			  int align_shift)
{
	size_t page_size = BIT(align_shift);

	m_opts->va_start = round_down(va, page_size);
	m_opts->va_len = round_up(va + len, page_size) - m_opts->va_start;
	m_opts->va_end = m_opts->va_start + m_opts->va_len;
}

/**
 * cxi_phys_lac_alloc() - Map an AC into a resource group for use in DMA
 *                        commands. Exported to allow clients to perform
 *                        DMA using physical addresses.
 * @lni: the Logical Network Interface
 * @return: LAC or negative errno
 */
int cxi_phys_lac_alloc(struct cxi_lni *lni)
{
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv, lni);
	struct cass_dev *hw = container_of(lni_priv->dev, struct cass_dev, cdev);
	int lac;

	lac = cass_lac_get(hw, lni_priv->lni.rgid);
	if (lac < 0)
		return lac;

	cass_set_acid(hw, lni_priv->lni.rgid, lac, ATU_PHYS_AC);

	return lac;
}
EXPORT_SYMBOL(cxi_phys_lac_alloc);

/**
 * cxi_phys_lac_free() - Unmap a physical LAC
 *
 * @lni: the Logical Network Interface
 * @lac: LAC to free
 */
void cxi_phys_lac_free(struct cxi_lni *lni, int lac)
{
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv, lni);

	cass_lac_free(lni_priv, lac);
}
EXPORT_SYMBOL(cxi_phys_lac_free);

/**
 * cxi_clear_md() - Zero page table entries and invalidate a memory descriptor
 *
 * @md: Memory Descriptor used in network operations
 */
int cxi_clear_md(struct cxi_md *md)
{
	struct cxi_md_priv *md_priv;

	if (!md)
		return -EINVAL;

	md_priv = container_of(md, struct cxi_md_priv, md);

	pr_debug("va:%llx iova:%llx len:%lx\n", md->va, md->iova, md->len);

	cass_md_clear(md_priv, true, false);
	cass_dma_unmap_pages(md_priv);
	md_priv->external_sgt_owner = false;
	md_priv->sgt = NULL;

	return 0;
}
EXPORT_SYMBOL(cxi_clear_md);

/**
 * cxi_update_iov() - Map a list of virtual address or pages in an IO
 *                    virtual address space into an existing memory
 *                    descriptor.
 *
 * @md: Memory descriptor
 * @iter: List of virtual addresses to map - supports ITER_KVEC and ITER_BVEC
 */
int cxi_update_iov(struct cxi_md *md, const struct iov_iter *iter)
{
	u64 va;
	int ret;
	size_t len;
	struct cxi_md_priv *md_priv;
	struct cxi_lni_priv *lni_priv;
	struct cxi_dev *dev;
	struct cass_dev *hw;

	if (!md)
		return -EINVAL;

	md_priv = container_of(md, struct cxi_md_priv, md);
	lni_priv = md_priv->lni_priv;
	dev = lni_priv->dev;
	hw = container_of(dev, struct cass_dev, cdev);

	if (iov_iter_is_bvec(iter))
		ret = cass_bvec(hw, iter, &len, &va);
	else
		ret = cass_kvec(hw, iter, &len, &va);
	if (ret)
		return ret;

	if (len > md_priv->olen)
		return -EINVAL;

	md->len = len;

	ret = cass_nta_mirror_kern(md_priv, iter, md_priv->need_lock);

	pr_debug("iovs:%ld len:0x%lx iova:%llx\n", iter->nr_segs, md->len,
		 md->iova);

	return ret;

}
EXPORT_SYMBOL(cxi_update_iov);

/**
 * cxi_map_iov() - Map a list of virtual address or pages in an IO
 *                 virtual address space
 *
 * @lni: the Logical Network Interface
 * @iter: List of virtual addresses to map - supports ITER_KVEC and ITER_BVEC
 * @flags: various options affecting the map
 */
struct cxi_md *cxi_map_iov(struct cxi_lni *lni, const struct iov_iter *iter,
			   u32 flags)
{
	u64 va;
	int ret;
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cxi_md_priv *md_priv;
	struct cxi_md *md;
	struct ac_map_opts m_opts = {
		.page_shift = PAGE_SHIFT,
		.huge_shift = PMD_SHIFT
	};
	struct cass_ac *cac;

	/* currently unsupported flags */
	if (flags & CXI_MAP_ATS)
		return ERR_PTR(-EINVAL);

	if (iov_iter_is_bvec(iter))
		ret = cass_bvec(hw, iter, &m_opts.va_len, &va);
	else
		ret = cass_kvec(hw, iter, &m_opts.va_len, &va);
	if (ret)
		return ERR_PTR(ret);

	m_opts.va_start = 0;
	m_opts.flags = flags;
	m_opts.va_end = m_opts.va_start + m_opts.va_len;
	m_opts.ptg_mode = default_ptg_mode;

	cac = get_matching_ac(lni_priv, &m_opts);
	if (IS_ERR(cac))
		return ERR_PTR(PTR_ERR(cac));

	md_priv = kzalloc(sizeof(*md_priv), GFP_KERNEL);
	if (md_priv == NULL)
		return ERR_PTR(-ENOMEM);
	refcount_set(&md_priv->refcount, 1);
	md = &md_priv->md;
	md_priv->device = &hw->cdev.pdev->dev;

	/* Get an MD ID, above CXI_MD_NONE */
	ret = ida_simple_get(&hw->md_index_table, 1, 0, GFP_KERNEL);
	if (ret < 0) {
		cxidev_err(cdev, "ida_simple_get failed %d\n", ret);
		goto md_free;
	}
	md->id = ret;

	md_priv->cac = cac;
	md_priv->lni_priv = lni_priv;

	cass_init_md(&m_opts, md_priv);

	ret = cass_nta_mirror_kern(md_priv, iter, true);
	if (ret)
		goto mirror_range_error;

	cass_add_md(lni_priv, &m_opts, md_priv);

	pr_debug("iovs:%ld len:0x%lx iova:%llx\n", iter->nr_segs, md->len,
		 md->iova);

	refcount_inc(&lni_priv->refcount);

	return md;

mirror_range_error:
	cass_md_iova_free(md_priv);
	ida_simple_remove(&hw->md_index_table, md->id);
md_free:
	kfree(md_priv);

	return ERR_PTR(ret);
}
EXPORT_SYMBOL(cxi_map_iov);

/**
 * cxi_map_sgtable() - Map an sg table to an IO virtual address space
 *
 * @lni: The Logical Network Interface
 * @sgt: The sg_table object. It is expected to be dma mapped.
 * @flags: Various options affecting the map
 *
 * @return: memory descriptor or error code
 */
struct cxi_md *cxi_map_sgtable(struct cxi_lni *lni, struct sg_table *sgt,
			       u32 flags)
{
	int ret;
	size_t len;
	struct cxi_md *md;
	struct cass_ac *cac;
	struct cxi_md_priv *md_priv;
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct ac_map_opts m_opts = {
		.page_shift = PAGE_SHIFT,
		.huge_shift = PMD_SHIFT
	};

	/* unsupported flags */
	if (flags & CXI_MAP_ATS)
		return ERR_PTR(-EINVAL);

	if (!cass_sgtable_is_valid(hw, sgt, &len))
		return ERR_PTR(-EINVAL);

	m_opts.va_start = 0;
	m_opts.flags = flags;
	m_opts.va_len = len;
	m_opts.va_end = m_opts.va_len;
	m_opts.ptg_mode = default_ptg_mode;

	cac = get_matching_ac(lni_priv, &m_opts);
	if (IS_ERR(cac))
		return ERR_PTR(PTR_ERR(cac));

	md_priv = kzalloc(sizeof(*md_priv), GFP_KERNEL);
	if (md_priv == NULL)
		return ERR_PTR(-ENOMEM);
	refcount_set(&md_priv->refcount, 1);
	md = &md_priv->md;
	md_priv->device = &hw->cdev.pdev->dev;

	/* Get an MD ID, above CXI_MD_NONE */
	ret = ida_simple_get(&hw->md_index_table, 1, 0, GFP_KERNEL);
	if (ret < 0) {
		cxidev_err(cdev, "ida_simple_get failed %d\n", ret);
		goto md_free;
	}
	md->id = ret;

	md_priv->sgt = sgt;
	md_priv->external_sgt_owner = true;
	md_priv->cac = cac;
	md_priv->lni_priv = lni_priv;

	cass_init_md(&m_opts, md_priv);

	pr_debug("va:%llx va_len:%lx iova:%0llx iova_base:%llx md:%d ac:%d flags:%x\n",
		 m_opts.va_start, m_opts.va_len, m_opts.iova, cac->iova_base,
		 md->id, md_priv->cac->ac.acid, flags);

	ret = cass_nta_mirror_sgt(md_priv, true);
	if (ret)
		goto mirror_range_error;

	cass_add_md(lni_priv, &m_opts, md_priv);

	pr_debug("nents:%u len:0x%lx iova:%llx\n", sgt->nents, md->len,
		 md->iova);

	refcount_inc(&lni_priv->refcount);

	return md;

mirror_range_error:
	cass_md_iova_free(md_priv);
	ida_simple_remove(&hw->md_index_table, md->id);
md_free:
	kfree(md_priv);

	return ERR_PTR(ret);
}
EXPORT_SYMBOL(cxi_map_sgtable);

/**
 * cxi_update_sgtable() - Update the specified address range contained in MD
 *
 * @md: Memory Descriptor used in network operations
 * @sgt: The sg_table object. It is expected to be dma mapped.
 *
 * @return: 0 on success or negative error value
 */
int cxi_update_sgtable(struct cxi_md *md, struct sg_table *sgt)
{
	int ret;
	size_t len;
	struct cxi_md_priv *md_priv = container_of(md, struct cxi_md_priv, md);
	struct cxi_dev *cdev = md_priv->lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	if (!cass_sgtable_is_valid(hw, sgt, &len))
		return -EINVAL;

	if (len > md_priv->olen) {
		pr_debug("Address range not bounded by MD\n");
		return -EINVAL;
	}

	md_priv->sgt = sgt;
	md_priv->external_sgt_owner = true;

	pr_debug("nents:%u len:0x%lx iova:%llx\n", sgt->nents, md->len,
		 md->iova);

	ret = cass_nta_mirror_sgt(md_priv, md_priv->need_lock);
	if (ret)
		return ret;

	md_priv->need_lock = false;

	return 0;
}
EXPORT_SYMBOL(cxi_update_sgtable);

/**
 * cxi_map() - Map virtual addresses into IO  address space
 *
 * If CXI_MAP_ALLOC_MD is set, just allocate an IOVA and return the MD.
 *
 * @lni: the Logical Network Interface
 * @va: kernel or user virtual address
 * @len: length of address to map in bytes
 * @flags: various options affecting the map
 * @hints: hints used for debugging
 *
 * @return: memory descriptor or error code
 */
struct cxi_md *cxi_map(struct cxi_lni *lni, uintptr_t va, size_t len,
		       u32 flags, const struct cxi_md_hints *hints)
{
	struct cxi_lni_priv *lni_priv = container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *cdev = lni_priv->dev;
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int ret;
	struct cxi_md_priv *md_priv;
	struct cxi_md *md;
	struct ac_map_opts m_opts = {
		.huge_shift = PMD_SHIFT,
		.page_shift = PAGE_SHIFT
	};
	struct cass_ac *cac;
	u64 ova = va;
	size_t olen = len;

	if (!len) {
		pr_debug("Length is 0\n");
		return ERR_PTR(-EINVAL);
	}

	if (flags & CXI_MAP_PIN &&
	    (flags & CXI_MAP_FAULT || flags & CXI_MAP_PREFETCH)) {
		pr_debug("Invalid flags combination:0x%x\n", flags);
		return ERR_PTR(-EINVAL);
	}

	if (hints && (hints->ptg_mode_valid)) {
		/* Invalid combination. hugepage_test builds two
		 * level tables.
		 */
		if ((hints->ptg_mode == C_ATU_PTG_MODE_SGL) &&
				m_opts.hugepage_test) {
			cxidev_info(cdev, "Hugepage test cannot be used with PTG Mode Single\n");
			return ERR_PTR(-EINVAL);
		}

		if ((hints->ptg_mode < C_ATU_PTG_MODE_SGL) ||
				(hints->ptg_mode > C_ATU_PTG_MODE_DBL_C)) {
			cxidev_info(cdev, "Invalid PTG_MODE supplied:%d\n",
				    hints->ptg_mode);
			return ERR_PTR(-EINVAL);
		}
		m_opts.ptg_mode = hints->ptg_mode;
	} else {
		m_opts.ptg_mode = default_ptg_mode;
	}

	ret = cass_odp_supported(hw, flags);
	if (ret)
		return ERR_PTR(ret);

	md_priv = kzalloc(sizeof(*md_priv), GFP_KERNEL);
	if (md_priv == NULL)
		return ERR_PTR(-ENOMEM);

	refcount_set(&md_priv->refcount, 1);
	md = &md_priv->md;
	md_priv->lni_priv = lni_priv;
	md_priv->device = &hw->cdev.pdev->dev;
	m_opts.flags = flags;
	m_opts.md_priv = md_priv;

	if (flags & CXI_MAP_ATS) {
		cass_align_start_len(&m_opts, va, len, PAGE_SHIFT);
	} else if (flags & CXI_MAP_DEVICE) {
		if (hints) {
			if (hints->dmabuf_valid) {
				md_priv->dmabuf_fd = hints->dmabuf_fd;
				md_priv->dmabuf_offset = hints->dmabuf_offset;
				md_priv->dmabuf_length = olen;
			} else {
				md_priv->dmabuf_fd = INVALID_DMABUF_FD;
			}
		}

		ret = cass_is_device_memory(hw, &m_opts, va, len);
		if (ret)
			goto md_free;

		cass_align_start_len(&m_opts, va, len, m_opts.page_shift);

		ret = cass_device_get_pages(&m_opts);
		if (ret)
			goto md_free;

	} else if (flags & CXI_MAP_USER_ADDR) {
		int align_shift = m_opts.page_shift;

		m_opts.va_end = va + len;
		ret = cass_cpu_page_size(hw, &m_opts, current->mm, va, hints,
					 &align_shift);
		if (ret)
			goto md_free;

		cass_align_start_len(&m_opts, va, len, align_shift);
	} else {
		/* VA is a kernel address which is expected to be aligned
		 * to the PAGE_SIZE.
		 */
		if (va & ~PAGE_MASK) {
			cxidev_err(cdev, "va not aligned 0x%lx\n", va);
			ret = -EINVAL;
			goto md_free;
		}

		cass_align_start_len(&m_opts, va, len, m_opts.page_shift);
	}

	if (m_opts.ptg_mode == C_ATU_PTG_MODE_SGL)
		m_opts.huge_shift = m_opts.page_shift;

	cac = get_matching_ac(lni_priv, &m_opts);
	if (IS_ERR(cac)) {
		cxidev_dbg(cdev, "Failure to get AC:%ld\n", PTR_ERR(cac));
		ret = PTR_ERR(cac);
		goto put_device_pages;
	}

	md_priv->cac = cac;

	/* Get an MD ID, above CXI_MD_NONE */
	ret = ida_simple_get(&hw->md_index_table, 1, 0, GFP_KERNEL);
	if (ret < 0) {
		cxidev_err(cdev, "ida_simple_get failed %d\n", ret);
		goto iova_free;
	}

	md->id = ret;

	cass_init_md(&m_opts, md_priv);

	pr_debug("ova:%llx olen:%lx va:%llx iova:%0llx iova_base:%llx md:%d ac:%d len:%lx flags:%x\n",
		 ova, olen, m_opts.va_start, md->iova, cac->iova_base,
		 md->id, md_priv->cac->ac.acid, md->len, flags);

	if (!(flags & CXI_MAP_ALLOC_MD)) {
		ret = cass_mirror_range(md_priv, &m_opts);
		if (ret) {
			pr_debug("Mirror address range failed:%d\n", ret);
			goto mirror_range_error;
		}
	}

	cass_add_md(lni_priv, &m_opts, md_priv);
	refcount_inc(&lni_priv->refcount);

	return md;

mirror_range_error:
	ida_simple_remove(&hw->md_index_table, md->id);
iova_free:
	cass_md_iova_free(md_priv);
put_device_pages:
	if (flags & CXI_MAP_DEVICE)
		cass_device_put_pages(md_priv);
md_free:
	kfree(md_priv);

	return ERR_PTR(ret);
}
EXPORT_SYMBOL(cxi_map);

/**
 * cxi_unmap() - Unmap a virtual address
 *
 * @md: Memory Descriptor used in network operations
 */
int cxi_unmap(struct cxi_md *md)
{
	struct cxi_md_priv *md_priv;
	struct cxi_lni_priv *lni_priv;
	struct cxi_dev *dev;
	struct cass_dev *hw;

	if (!md || md->lac >= C_NUM_LACS)
		return -EINVAL;

	md_priv = container_of(md, struct cxi_md_priv, md);
	lni_priv = md_priv->lni_priv;
	dev = lni_priv->dev;
	hw = container_of(dev, struct cass_dev, cdev);

	/* One reference for CAC allocation, one reference for MD. */
	if (refcount_read(&md_priv->cac->refcount) < 2)
		return -EINVAL;

	pr_debug("md:%d ac:%d lac:%d va:%llx iova:%llx len:%lx\n",
		 md->id, md_priv->cac->ac.acid, md->lac, md->va, md->iova,
		 md->len);

	cass_remove_md(md_priv);
	cass_md_iova_free(md_priv);
	ida_simple_remove(&hw->md_index_table, md->id);

	cxidev_WARN_ONCE(dev, !refcount_dec_and_test(&md_priv->refcount),
			 "Resource leaks - MD refcount not zero: %d\n",
			 refcount_read(&md_priv->refcount));

	cass_dma_unmap_pages(md_priv);
	kfree(md_priv);

	refcount_dec(&lni_priv->refcount);

	return 0;
}
EXPORT_SYMBOL(cxi_unmap);

/**
 * cxi_update_md() - Update the specified address range contained in MD
 *
 * @md: Memory Descriptor used in network operations
 * @va: User virtual address
 * @len: Length of address to fault in bytes
 * @flags: Currently only supports faulting pages with CXI_MAP_FAULT flag
 *         for an ODP MD.
 *
 * @return: 0 on success or negative error value
 */
int cxi_update_md(struct cxi_md *md, uintptr_t va, size_t len, u32 flags)
{
	int ret;
	struct cxi_md_priv *md_priv;
	struct cass_dev *hw;
	int align_shift;
	struct ac_map_opts m_opts = {};

	if (!md || md->lac >= C_NUM_LACS)
		return -EINVAL;

	if (va < md->va || (va + len) > (md->va + md->len)) {
		pr_debug("Address range not bounded by MD\n");
		return -EINVAL;
	}

	/* Just support CXI_MAP_FAULT for now */
	if (!(flags & CXI_MAP_FAULT)) {
		pr_debug("Only CXI_MAP_FAULT is supported\n");
		return -EINVAL;
	}

	md_priv = container_of(md, struct cxi_md_priv, md);
	hw = container_of(md_priv->lni_priv->dev, struct cass_dev, cdev);

	if (md_priv->flags & CXI_MAP_PIN) {
		pr_debug("Must be ODP MD\n");
		return -EINVAL;
	}

	m_opts.va_end = va + len;
	m_opts.md_priv = md_priv;
	m_opts.page_shift = md->page_shift;
	m_opts.huge_shift = md->huge_shift;
	/* Use the permissions from the MD */
	m_opts.flags = md_priv->flags & (CXI_MAP_READ | CXI_MAP_WRITE);
	m_opts.flags |= flags;
	align_shift = m_opts.page_shift;

	ret = cass_cpu_page_size(hw, &m_opts, current->mm, va, NULL,
				 &align_shift);
	if (ret)
		return ret;

	/* We only support VMAs that match the MD's hugepage size. */
	if (m_opts.huge_shift != md->huge_shift) {
		pr_debug("Error: VMA huge_shift (%d) must match MD (%d)\n",
			 m_opts.huge_shift, md->huge_shift);
		return -EINVAL;
	}

	cass_align_start_len(&m_opts, va, len, align_shift);
	m_opts.iova = md->iova + (m_opts.va_start - md->va);

	pr_debug("md:%d ac:%d md.va:%llx va:%llx iova:%llx len:%lx flags:%x\n",
		 md->id, md_priv->cac->ac.acid, md->va, m_opts.va_start,
		 m_opts.iova, m_opts.va_len, m_opts.flags);

	ret = cass_mirror_odp(&m_opts, md_priv->cac,
			      m_opts.va_len >> m_opts.page_shift,
			      m_opts.va_start);

	return ret;
}
EXPORT_SYMBOL(cxi_update_md);
