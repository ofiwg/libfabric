// SPDX-License-Identifier: GPL-2.0
/*
 * Create and destroy Cassini counting events.
 * Copyright 2018 Hewlett Packard Enterprise Development LP
 */
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/iopoll.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

/* Initializes (if init is true) or cancel (if init is false) pending
 * triggered operations on a CT.
 */
static int set_hw_state(struct cass_dev *hw, unsigned int ctn,
			bool init)
{
	void __iomem *csr = cass_csr(hw, C_CQ_CFG_INIT_CT_HW_STATE);
	union c_cq_cfg_init_ct_hw_state ct_hw_state;
	int rc;

	mutex_lock(&hw->ct_init_lock);

	ct_hw_state.qw = 0;
	ct_hw_state.ct_handle = ctn;
	if (init)
		ct_hw_state.init = 1;
	else
		ct_hw_state.cancel = 1;
	cass_write(hw, C_CQ_CFG_INIT_CT_HW_STATE, &ct_hw_state.qw,
		   sizeof(ct_hw_state));

	rc = readq_poll_timeout(csr, ct_hw_state.qw,
				ct_hw_state.pending == 0, 1, 1000000);
	cxidev_WARN_ONCE(&hw->cdev, rc, "Timeout waiting for CT operation (%d)\n",
			 init);

	/* Need to zero cancel and init fields so that they can be reused. */
	memset(&ct_hw_state, 0, sizeof(ct_hw_state));
	cass_write(hw, C_CQ_CFG_INIT_CT_HW_STATE, &ct_hw_state.qw,
		   sizeof(ct_hw_state));

	mutex_unlock(&hw->ct_init_lock);

	return rc;
}

/* Map the wb buffer using DMA address. Pin buffer if from user space. */
static int ct_wb_map(struct cass_dev *hw, const struct cxi_ct_priv *ct_priv,
		     struct ct_wb_desc *desc, struct c_ct_writeback *wb)
{
	static const struct c_ct_writeback wb_init = {};
	struct device *device = &hw->cdev.pdev->dev;
	dma_addr_t dma_addrs;
	int rc;

	desc->md_priv = NULL;
	desc->page = NULL;

	if (!ct_priv->is_user) {
		*wb = wb_init;

		if (is_vmalloc_addr(wb)) {
			desc->wb_dma_addr = dma_map_page(device,
							 vmalloc_to_page(wb),
							 offset_in_page(wb),
							 sizeof(*wb), DMA_FROM_DEVICE);
			desc->is_map_page = 1;
		} else {
			desc->wb_dma_addr = dma_map_single(device, wb, sizeof(*wb),
							   DMA_FROM_DEVICE);
		}
		if (dma_mapping_error(&hw->cdev.pdev->dev, desc->wb_dma_addr))
			return -ENOMEM;

		return 0;
	}

	rc = pin_user_pages_fast((unsigned long)wb, 1, 1, &desc->page);
	if (rc == 1) {
		desc->wb_dma_addr = dma_map_single(device,
						   page_address(desc->page) +
						   offset_in_page(wb),
						   sizeof(*wb), DMA_FROM_DEVICE);
		if (dma_mapping_error(device, desc->wb_dma_addr))
			return -ENOMEM;

		return 0;
	}

	desc->md_priv = cass_device_md(ct_priv->lni_priv, (u64)wb, sizeof(*wb),
				       &dma_addrs);
	if (IS_ERR(desc->md_priv))
		return PTR_ERR(desc->md_priv);

	desc->wb_dma_addr = dma_addrs + offset_in_page(wb);

	return 0;
}

static void ct_wb_unmap(struct cass_dev *hw, const struct cxi_ct_priv *ct_priv,
			struct ct_wb_desc *desc)
{
	if (desc->wb_dma_addr == 0)
		return;

	/* The presence of md_priv indicates device memory which is unmapped
	 * when the pages are released.
	 */
	if (desc->md_priv == NULL) {
		if (desc->is_map_page)
			dma_unmap_page(&hw->cdev.pdev->dev, desc->wb_dma_addr,
				       sizeof(struct c_ct_writeback),
				       DMA_FROM_DEVICE);
		else
			dma_unmap_single(&hw->cdev.pdev->dev, desc->wb_dma_addr,
					 sizeof(struct c_ct_writeback),
					 DMA_FROM_DEVICE);
	}

	if (!ct_priv->is_user)
		return;

	if (desc->page) {
		unpin_user_page(desc->page);
		desc->page = NULL;
	} else if (desc->md_priv) {
		p2p_ops.put_pages(desc->md_priv);
		kfree(desc->md_priv);
		desc->md_priv = NULL;
	}
}

/**
 * cxi_ct_alloc() - Allocate a counting event
 *
 * @lni: LNI counting event is allocated against
 * @wb: User defined writeback buffer, must be DMA-able.
 * @is_user: Non-zero value if the counting event is for userspace.
 *
 * Writeback buffer is optional. If @wb is NULL, the wb field in the returned
 * cxi_ct will also be NULL.
 *
 * Return: Valid pointer on success. Else, negative errno pointer.
 */
struct cxi_ct *cxi_ct_alloc(struct cxi_lni *lni, struct c_ct_writeback *wb,
			    bool is_user)
{
	struct cxi_lni_priv *lni_priv =
		container_of(lni, struct cxi_lni_priv, lni);
	struct cxi_dev *dev = lni_priv->dev;
	struct cass_dev *hw =
		container_of(dev, struct cass_dev, cdev);
	struct cxi_ct_priv *ct_priv;
	u16 wb_ac;
	int rc;
	int ctn;
	struct cxi_md_priv *md_priv = NULL;

	/* WB address must be 8-byte aligned. */
	if (!IS_ALIGNED((u64)wb, 8))
		return ERR_PTR(-EINVAL);

	ct_priv = kzalloc(sizeof(*ct_priv), GFP_KERNEL);
	if (!ct_priv)
		return ERR_PTR(-ENOMEM);
	ct_priv->lni_priv = lni_priv;
	ct_priv->is_user = is_user;

	/* Check the associated service to see if this CT can be allocated */
	rc = cxi_rgroup_alloc_resource(lni_priv->rgroup, CXI_RESOURCE_CT);
	if (rc)
		goto free_ct_priv;

	/*
	 * A WB address context of zero means that WB is disabled for
	 * the CT.
	 */
	if (wb) {
		rc = ct_wb_map(hw, ct_priv, &ct_priv->wb_desc, wb);
		if (rc)
			goto dec_rsrc_use;
		wb_ac = ATU_PHYS_AC;
	} else {
		ct_priv->wb_desc.wb_dma_addr = 0;
		wb_ac = C_AC_NONE;
	}

	/* Allocate and configure the counting event. */
	rc = ida_simple_get(&hw->ct_table, 1, C_NUM_CTS, GFP_KERNEL);
	if (rc < 0)
		goto free_md_priv;
	ctn = rc;

	if (!is_user) {
		phys_addr_t mmio_phys = hw->regs_base + C_MEMORG_CQ_TOU +
				ctn * C_TOU_LAUNCH_PAGE_SIZE;
		ct_priv->ct_mmio = ioremap(mmio_phys, PAGE_SIZE);
		if (!ct_priv->ct_mmio) {
			cxidev_warn_once(dev, "ioremap failed\n");
			rc = -ENOMEM;
			goto ct_release;
		}
	}

	/* Reset the hardware CT. */
	rc = set_hw_state(hw, ctn, true);
	if (rc)
		goto ct_unmap;

	/* Finally the CT can be enabled */
	cxi_ct_init(&ct_priv->ct, wb, ctn, (__force u64 *)ct_priv->ct_mmio);
	cass_ct_enable(hw, ctn, lni_priv->lni.rgid, ct_priv->wb_desc.wb_dma_addr, wb_ac);

	/* Track the counting event. */
	spin_lock(&lni_priv->res_lock);
	list_add_tail(&ct_priv->entry, &lni_priv->ct_list);
	refcount_inc(&lni_priv->refcount);
	atomic_inc(&hw->stats.ct);
	spin_unlock(&lni_priv->res_lock);

	ct_debugfs_setup(ct_priv->ct.ctn, ct_priv, hw, lni_priv);

	return &ct_priv->ct;

ct_unmap:
	if (!is_user)
		iounmap(ct_priv->ct_mmio);
ct_release:
	ida_simple_remove(&hw->ct_table, ctn);
free_md_priv:
	kfree(md_priv);
	if (wb)
		ct_wb_unmap(hw, ct_priv, &ct_priv->wb_desc);
dec_rsrc_use:
	cxi_rgroup_free_resource(lni_priv->rgroup, CXI_RESOURCE_CT);
free_ct_priv:
	kfree(ct_priv);

	return ERR_PTR(rc);
}
EXPORT_SYMBOL(cxi_ct_alloc);

/* Wait for credits_in_use.tou_wb_credits to drain to 0. We then know that
 * no more writebacks are in progress.
 */
static int drain_wb_credits(struct cass_dev *hw)
{
	ktime_t timeout;
	int timeout_us = 1000;
	union c_cq_sts_credits_in_use in_use;

	timeout = ktime_add_us(ktime_get(), timeout_us);

	for (;;) {
		cass_read(hw, C_CQ_STS_CREDITS_IN_USE, &in_use,
			  sizeof(in_use));

		if (in_use.tou_wb_credits == 0)
			return 0;

		if (ktime_compare(ktime_get(), timeout) > 0)
			return -ETIMEDOUT;

		usleep_range(5, 20);
	}
}

/**
 * cxi_ct_wb_update() - Update a counting event with a new writeback address.
 *
 * Flush by setting c_cq_cfg_credit_limits.tou_wb_credits to 0. When
 * c_cq_sts_credits_in_use.tou_wb_credits is 0, then no writebacks are
 * in progress. Re-enable the c_cq_cfg_credit_limits.tou_wb_credits.
 *
 * @ct: Counting event
 * @wb: User defined writeback buffer, must be DMA-able.
 */
int cxi_ct_wb_update(struct cxi_ct *ct, struct c_ct_writeback *wb)
{
	struct cxi_ct_priv *ct_priv = container_of(ct, struct cxi_ct_priv, ct);
	struct cxi_lni_priv *lni_priv = ct_priv->lni_priv;
	struct cass_dev *hw =
		container_of(lni_priv->dev, struct cass_dev, cdev);
	union c_cq_cfg_credit_limits limits = {};
	union c_cq_cfg_ct_wb_mem_addr mem = {
		.wb_mem_context = ATU_PHYS_AC,
	};
	struct ct_wb_desc new_desc;
	int wb_credits = 0;
	int rc;

	// TODO: write pycxi test for this fn + driver test

	if (!wb)
		return -EINVAL;

	rc = ct_wb_map(hw, ct_priv, &new_desc, wb);
	if (rc)
		return rc;

	mutex_lock(&hw->ct_init_lock);

	/* If wb is valid, it may be in use. */
	if (ct->wb) {
		/* stop wb writes */
		cass_read(hw, C_CQ_CFG_CREDIT_LIMITS, &limits, sizeof(limits));
		wb_credits = limits.tou_wb_credits;
		limits.tou_wb_credits = 0;
		cass_write(hw, C_CQ_CFG_CREDIT_LIMITS, &limits, sizeof(limits));

		rc = drain_wb_credits(hw);
		if (rc)
			goto set_limit;
	}

	mem.wb_mem_addr = new_desc.wb_dma_addr >> 3,
	cass_write(hw, C_CQ_CFG_CT_WB_MEM_ADDR(ct->ctn), &mem, sizeof(mem));
	cass_flush_pci(hw);

	/* Previous pinned pages need to be cleaned up. */
	ct_wb_unmap(hw, ct_priv, &ct_priv->wb_desc);
	ct_priv->wb_desc = new_desc;

set_limit:
	if (ct->wb) {
		limits.tou_wb_credits = wb_credits;
		cass_write(hw, C_CQ_CFG_CREDIT_LIMITS, &limits, sizeof(limits));
	}

	mutex_unlock(&hw->ct_init_lock);

	/* finally, save the new wb */
	if (!rc)
		ct->wb = wb;

	return rc;
}
EXPORT_SYMBOL(cxi_ct_wb_update);

/**
 * cxi_ct_free() - Free a counting event.
 * @ct: Counting event
 */
void cxi_ct_free(struct cxi_ct *ct)
{
	struct cxi_ct_priv *ct_priv = container_of(ct, struct cxi_ct_priv, ct);
	struct cxi_lni_priv *lni_priv = ct_priv->lni_priv;
	struct cass_dev *hw =
		container_of(lni_priv->dev, struct cass_dev, cdev);

	/* Change ownership to the driver. */
	cass_ct_disable(hw, ct_priv->ct.ctn);

	/* Cancel all pending operations. */
	set_hw_state(hw, ct_priv->ct.ctn, false);

	spin_lock(&lni_priv->res_lock);
	list_del(&ct_priv->entry);
	spin_unlock(&lni_priv->res_lock);

	if (!ct_priv->is_user)
		iounmap(ct_priv->ct_mmio);

	ct_wb_unmap(hw, ct_priv, &ct_priv->wb_desc);

	/* Hang the CT on the LNI pending deletion list, to be
	 * released later
	 */
	spin_lock(&lni_priv->res_lock);
	list_add_tail(&ct_priv->entry, &lni_priv->ct_cleanups_list);
	spin_unlock(&lni_priv->res_lock);
}
EXPORT_SYMBOL(cxi_ct_free);

/* Finish releasing all the CTs on the deletion pending list */
void finalize_ct_cleanups(struct cxi_lni_priv *lni)
{
	struct cxi_dev *dev = lni->dev;
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	struct cxi_ct_priv *ct;

	while ((ct = list_first_entry_or_null(&lni->ct_cleanups_list,
					      struct cxi_ct_priv, entry))) {
		list_del(&ct->entry);

		debugfs_remove(ct->lni_dir);
		debugfs_remove_recursive(ct->debug_dir);

		refcount_dec(&lni->refcount);
		atomic_dec(&hw->stats.ct);
		cxi_rgroup_free_resource(lni->rgroup, CXI_RESOURCE_CT);
		ida_simple_remove(&hw->ct_table, ct->ct.ctn);
		kfree(ct);
	}
}

/**
 * cxi_ct_user_info() - Get userspace mmap info for counting event doorbell.
 * @ct: Counting event
 * @doorbell_addr: On success, will contain the address of the doorbell.
 * @doorbell_size: On success, will contain the size of the doorbell.
 *
 * Return: On success, returns zero. On error, returns a negative error number.
 */
int cxi_ct_user_info(struct cxi_ct *ct, phys_addr_t *doorbell_addr,
		     size_t *doorbell_size)
{
	struct cxi_ct_priv *ct_priv = container_of(ct, struct cxi_ct_priv, ct);
	struct cass_dev *hw = container_of(ct_priv->lni_priv->dev,
					   struct cass_dev, cdev);

	if (!ct_priv->is_user)
		return -EINVAL;

	*doorbell_addr = hw->regs_base + C_MEMORG_CQ_TOU +
		(ct->ctn * C_TOU_LAUNCH_PAGE_SIZE);
	*doorbell_size = PAGE_SIZE;

	return 0;
}
EXPORT_SYMBOL(cxi_ct_user_info);
