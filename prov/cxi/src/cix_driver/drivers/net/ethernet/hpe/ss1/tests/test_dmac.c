// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020,2024 Hewlett Packard Enterprise Development LP */

/* Test driver for DMAC functionality. */

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/hpe/cxi/cxi.h>
#include <linux/delay.h>

#include "cass_core.h"

static void dmac_desc_zero(struct cass_dev *hw)
{
	static const union c_pi_cfg_dmac_desc desc = {
		.reserved0  = 0,
		.dst_addr   = 0,
		.reserved1  = 0,
		.src_addr   = 0,
		.reserved2  = 0,
		.length     = 0,
		.cont       = 0,
		.reserved3  = 0
	};
	int i;

	for ((i = 0) ; (i < C_PI_CFG_DMAC_DESC_ENTRIES) ; (++i))
		cass_write(hw, C_PI_CFG_DMAC_DESC(i), &desc, sizeof(desc));
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static void testcase_dmac_0(struct cxi_dev *dev)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	union c_pi_cfg_dmac_desc desc = {};
	union c_pi_cfg_dmac dmac = {};
	union c_pi_cfg_dmac_cpl_addr dmac_cpl = {};
	const int first_desc = 1022; /* enough to wrap */
	int timeout_count;
	struct device *device = &hw->cdev.pdev->dev;
	struct dest {
		union c_mb_sts_rev rev;
		union c_ixe_cfg_bth_opcode cfg_bth;
		union c_ee_cfg_eq_descriptor eq_desc;
		struct c_pi_dmac_cdesc cpl_desc;
	} *dest;
	struct page *dest_page;
	dma_addr_t dest_dma_addr;

	pr_err("TESTCASE_DMAC_0: START\n");

	/* Disable DMAC */
	dmac.desc_index		= 0;
	dmac.enable		= 0;
	dmac.start		= 0;
	dmac.cpl_event_enable	= 0;
	dmac.irq_on_done	= 0;
	dmac.irq_on_error	= 0;
	cass_write(hw, C_PI_CFG_DMAC, &dmac, sizeof(dmac));

	/* zero the dmac descriptors */
	dmac_desc_zero(hw);

	/* zero the completion address */
	dmac_cpl.address = 0;
	cass_write(hw, C_PI_CFG_DMAC_CPL_ADDR, &dmac_cpl, sizeof(dmac_cpl));

	/* These must be in DMA-able memory. Not the stack. */
	dest_page = alloc_page(GFP_KERNEL | GFP_DMA);
	if (!dest_page)
		return;
	dest_dma_addr = dma_map_page(device, dest_page, 0, sizeof(*dest), DMA_FROM_DEVICE);
	if (dma_mapping_error(&hw->cdev.pdev->dev, dest_dma_addr))
		goto free_dest;
	dest = page_address(dest_page);

	/* Configure the DMAC to write a completion event to cpl_desc */
	dmac_cpl.address = (dest_dma_addr + offsetof(struct dest, cpl_desc)) >> 3;
	cass_write(hw, C_PI_CFG_DMAC_CPL_ADDR, &dmac_cpl, sizeof(dmac_cpl));

	/* Prepare 3 descriptors and write to them */

	desc.dst_addr = (dest_dma_addr + offsetof(struct dest, rev)) >> 3;
	desc.src_addr = (C_MB_STS_REV - C_CQ_BASE) >> 3;
	desc.length = sizeof(dest->rev) / 8 - 1;
	desc.cont = 1;

	cass_write(hw, C_PI_CFG_DMAC_DESC((first_desc + 0) % C_PI_CFG_DMAC_DESC_ENTRIES),
		   &desc, sizeof(desc));

	desc.dst_addr = (dest_dma_addr + offsetof(struct dest, eq_desc)) >> 3;
	desc.src_addr = (C_EE_CFG_EQ_DESCRIPTOR(50) - C_CQ_BASE) >> 3;
	desc.length = sizeof(dest->eq_desc) / 8 - 1;
	desc.cont = 1;

	cass_write(hw, C_PI_CFG_DMAC_DESC((first_desc + 1) % C_PI_CFG_DMAC_DESC_ENTRIES),
		   &desc, sizeof(desc));

	desc.dst_addr = (dest_dma_addr + offsetof(struct dest, cfg_bth)) >> 3;
	desc.src_addr = (C_IXE_CFG_BTH_OPCODE(0) - C_CQ_BASE) >> 3;
	desc.length = sizeof(dest->cfg_bth) / 8 - 1;
	desc.cont = 0;

	cass_write(hw, C_PI_CFG_DMAC_DESC((first_desc + 2) % C_PI_CFG_DMAC_DESC_ENTRIES),
		   &desc, sizeof(desc));

	/* Trigger the operation */
	dmac.desc_index = first_desc;
	dmac.enable = 1;
	dmac.start = 1;
	dmac.cpl_event_enable = 1; /* write result to cpl_desc */

	cass_write(hw, C_PI_CFG_DMAC, &dmac, sizeof(dmac));
	cass_flush_pci(hw);

	/* Wait for completion */
	timeout_count = 10;
	while (timeout_count) {
		msleep(200);

		if (dest->cpl_desc.done == 1)
			break;

		timeout_count--;
	}

	if (dest->cpl_desc.done == 0)
		pr_err("TEST-ERROR: failed to get DMAC completion\n");
	if (dest->cpl_desc.status != 0)
		pr_err("TEST-ERROR: DMAC cpl status is %d\n", dest->cpl_desc.status);

	/* Check that known values in the CSR that were read are
	 * correct. We can't really check the EQ descriptor.
	 */
	if (dest->rev.vendor_id != 0x17db || dest->rev.device_id != 0x0501)
		pr_err("TEST-ERROR: failed to get good sts_rev\n");

	if (dest->cfg_bth.len_10 != 0x7 || dest->cfg_bth.len_5 != 0x4)
		pr_err("TEST-ERROR: failed to get good cfg_bth\n");

	/* Check that the DMAC is usable again */
	cass_read(hw, C_PI_CFG_DMAC, &dmac, sizeof(dmac));
	if (dmac.start != 0)
		pr_err("TEST-ERROR: dmac.start not 0\n");
	if (dmac.done != 1)
		pr_err("TEST-ERROR: dmac.done not 1\n");
	if (dmac.status != 0)
		pr_err("TEST-ERROR: dmac.status is %d\n", dmac.status);

	/* Disable DMAC */
	dmac.desc_index		= 0;
	dmac.enable		= 0;
	dmac.start		= 0;
	dmac.cpl_event_enable	= 0;
	dmac.irq_on_done	= 0;
	dmac.irq_on_error	= 0;
	cass_write(hw, C_PI_CFG_DMAC, &dmac, sizeof(dmac));

	/* zero the dmac descriptors */
	dmac_desc_zero(hw);

	/* zero the completion address */
	dmac_cpl.address = 0;
	cass_write(hw, C_PI_CFG_DMAC_CPL_ADDR, &dmac_cpl, sizeof(dmac_cpl));

	pr_err("TESTCASE_DMAC_0: FINISH\n");

	dma_unmap_single(device, dest_dma_addr, sizeof(*dest), DMA_FROM_DEVICE);
free_dest:
	__free_page(dest_page);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* Core is adding a new device.
 *
 * Use DMAC to read 2 CSRs with well known and invariant values, as
 * well as a 64-byte EQ descriptor.
 */
static int add_device(struct cxi_dev *dev)
{
	pr_err("TESTSUITE_DMAC: START\n");
	testcase_dmac_0(dev);
	pr_err("TESTSUITE_DMAC: FINISH\n");
	return 0;
}

static void remove_device(struct cxi_dev *dev)
{
}

static struct cxi_client cxiu_client = {
	.add = add_device,
	.remove = remove_device,
};

static int __init init(void)
{
	int ret;

	ret = cxi_register_client(&cxiu_client);
	if (ret) {
		pr_err("Couldn't register client\n");
		goto out;
	}

	return 0;

out:
	return ret;
}

static void __exit cleanup(void)
{
	cxi_unregister_client(&cxiu_client);
}

module_init(init);
module_exit(cleanup);

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("DMAC test driver");
MODULE_AUTHOR("Cray Inc.");
