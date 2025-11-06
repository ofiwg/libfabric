// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021 Hewlett Packard Enterprise Development LP */

#include <linux/iopoll.h>
#include <linux/bitmap.h>

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

#define DMAC_DESC_INDEX_UNSPECIFIED		((u16)(~0))

/*
 * _must_ only be called when holding mutex hw->dmac.lock
 * TODO: add kernel 'requires_lock' annotations
 */
static int cass_dmac_desc_set_id_verify(struct cass_dev *hw, int set_id)
{
	if (set_id < 0 || set_id >= DMAC_DESC_SET_COUNT ||
	    hw->dmac.desc_sets[set_id].count == 0)
		return -EINVAL;
	return 0;
}

static void dmac_desc_zero_n(struct cass_dev *hw, u16 desc_index, u16 count)
{
	cass_clear(hw, C_PI_CFG_DMAC_DESC(desc_index),
		   sizeof(union c_pi_cfg_dmac_desc) * count);
	cass_flush_pci(hw);
}

static void dmac_desc_zero_all(struct cass_dev *hw)
{
	dmac_desc_zero_n(hw, 0, C_PI_CFG_DMAC_DESC_NUM_AVAILABLE_ENTRIES -
			 C_PI_CFG_DMAC_DESC_NUM_RSVD_ENTRIES);
}

static void dmac_disable(struct cass_dev *hw)
{
	/* Disable DMAC */
	cass_clear(hw, C_PI_CFG_DMAC, C_PI_CFG_DMAC_SIZE);
	cass_flush_pci(hw);
}

static void dmac_enable(struct cass_dev *hw, u16 desc_index)
{
	union c_pi_cfg_dmac dmac = {
		.desc_index	  = desc_index,
		.enable		  = 1,
		.start		  = 1,
		.cpl_event_enable = 1,
		.irq_on_done	  = 1,
		.irq_on_error	  = 1
	};

	/* Enable DMAC */
	cass_write(hw, C_PI_CFG_DMAC, &dmac, sizeof(dmac));
	cass_flush_pci(hw);
}

/**
 * cxi_dmac_desc_set_reset() - reset number used to zero
 * @cdev: the device
 * @set_id: dmac descriptor set id
 *
 * reset the number of used descriptors associated with the supplied
 * dmac descriptor set id to zero
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_dmac_desc_set_reset(struct cxi_dev *cdev, int set_id)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval;

	mutex_lock(&hw->dmac.lock);

	retval = cass_dmac_desc_set_id_verify(hw, set_id);
	if (retval == 0)
		hw->dmac.desc_sets[set_id].numused = 0;

	mutex_unlock(&hw->dmac.lock);
	return retval;
}
EXPORT_SYMBOL(cxi_dmac_desc_set_reset);

/**
 * cxi_dmac_desc_set_add() - write the next descriptor in the set
 * @cdev: the device
 * @set_id: dmac descriptor set id
 * @dst: DMA address of host buffer
 * @src: starting nic csr offset
 * @len: length to read, in bytes. must be a multiple of 8 (sizeof(u64))
 *
 * use the supplied parameters to write the appropriate values to the
 * next descriptor in the supplied dmac descriptor set id and
 * if this is not the first descriptor in the set written then set
 * the continue flag of previous descriptor to true
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_dmac_desc_set_add(struct cxi_dev *cdev, int set_id, dma_addr_t dst,
			  u32 src, size_t len)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval = 0;
	u16 count;
	u16 index;
	u16 numused;
	union c_pi_cfg_dmac_desc desc = {};
	u32 src_end;
	const u32 u64count = len / sizeof(u64);

	/*
	 * dst address and src offset must be 64-bit aligned
	 *
	 * len must be a multiple of 8 (sizeof(u64))
	 *
	 * u64count can not be zero and u64count - 1 must fit in 20-bits
	 *
	 * cxi csr offsets are within 29 bits
	 *      the starting & ending src offsets
	 *      must be contained in 29 bits.
	 *
	 * if src offset crosses 23 bit addr boundary,
	 * then src must be 128 byte aligned
	 */

	if (!IS_ALIGNED(dst, 8)) {
		cxidev_err(cdev,
			   "destination address is not 64-bit aligned: 0x%016llx\n",
			   dst);
		return -EINVAL;
	}

	if ((len % sizeof(u64)) != 0) {
		cxidev_err(cdev,
			   "length is not multiple of 8 (sizeof(u64)): %ld\n",
			   len);
		return -EINVAL;
	}
	if (u64count == 0) {
		cxidev_err(cdev, "length is zero\n");
		return -EINVAL;
	}
	if (u64count > 0x00100000UL || len != u64count * sizeof(u64)) {
		cxidev_err(cdev, "length is too large: %ld\n", len);
		return -EINVAL;
	}

	if (!IS_ALIGNED(src, 8)) {
		cxidev_err(cdev,
			   "source offset is not 64-bit aligned: 0x%08x\n",
			   src);
		return -EINVAL;
	}
	if (src >= C_MEMORG_CSR_SIZE) {
		cxidev_err(cdev,
			   "starting source offset is out of range: 0x%08x\n",
			   src);
		return -EINVAL;
	}
	src_end = src + (u64count - 1) * sizeof(u64);
	if (src_end >= C_MEMORG_CSR_SIZE) {
		cxidev_err(cdev,
			   "ending source offset is out of range: 0x%08x\n",
			   src_end);
		return -EINVAL;
	}
	if ((src & 0xff000000UL) != (src_end & 0xff000000UL) &&
	    (src & 0x7fUL) != 0) {
		cxidev_err(cdev,
			   "source offset should be 128-byte aligned when transfer crosses bit-23: 0x%08x 0x%08x\n",
			   src, src_end);
		return -EINVAL;
	}

	mutex_lock(&hw->dmac.lock);

	retval = cass_dmac_desc_set_id_verify(hw, set_id);
	if (retval != 0)
		goto __cxi_dmac_desc_set_add_0;

	count = hw->dmac.desc_sets[set_id].count;
	index = hw->dmac.desc_sets[set_id].index;
	numused  = hw->dmac.desc_sets[set_id].numused;

	if (numused == count) {
		retval = -ENOSPC;
		goto __cxi_dmac_desc_set_add_0;
	}

	desc.dst_addr = dst >> 3;
	desc.src_addr = src >> 3;
	desc.length   = u64count - 1;

	/*
	 * write the descriptor
	 */
	cass_write(hw, C_PI_CFG_DMAC_DESC(index + numused), &desc, sizeof(desc));

	/*
	 * if the added descriptor is not the first one (0th index) of the set,
	 * then modify previous descriptor's cont flag to be 1 indicating to
	 * chain to the next descriptor.
	 */
	if (numused != 0) {
		cass_read(hw, C_PI_CFG_DMAC_DESC(index + numused - 1) +
			  sizeof(desc.qw[0]), &desc.qw[1],
			  sizeof(desc.qw[1]));
		desc.cont = 1;
		cass_write(hw, C_PI_CFG_DMAC_DESC(index + numused - 1) +
			   sizeof(desc.qw[0]), &desc.qw[1],
			   sizeof(desc.qw[1]));
	}

	++hw->dmac.desc_sets[set_id].numused;

__cxi_dmac_desc_set_add_0:
	mutex_unlock(&hw->dmac.lock);
	return retval;
}
EXPORT_SYMBOL(cxi_dmac_desc_set_add);

/*
 * _must_ only be called when holding mutex hw->dmac.lock
 */
static void cass_dmac_desc_set_id_fill(struct cass_dev *hw, int set_id,
				       u16 num_descs, u16 index,
				       const char *name)
{
	cxidev_dbg(&hw->cdev, "set_id=%d num_descs=%u index=%u\n",
		   set_id, num_descs, index);
	hw->dmac.desc_sets[set_id].count = num_descs;
	hw->dmac.desc_sets[set_id].index = index;
	hw->dmac.desc_sets[set_id].numused  = 0;
	hw->dmac.desc_sets[set_id].name = name;
}

/*
 * _must_ only be called when holding mutex hw->dmac.lock
 */
static void cass_dmac_desc_set_id_clear(struct cass_dev *hw, int set_id)
{
	cass_dmac_desc_set_id_fill(hw, set_id, 0, DMAC_DESC_INDEX_UNSPECIFIED,
				   NULL);
}

static int cass_dmac_desc_set_common_alloc(struct cass_dev *hw,
					   u16 num_descs, u16 desc_idx,
					   const char *name)
{
	int retval;
	int i;
	int set_id;
	unsigned long index;
	u16 idx = ((desc_idx == DMAC_DESC_INDEX_UNSPECIFIED) ? 0 : desc_idx);

	if (num_descs == 0 || num_descs >= C_PI_CFG_DMAC_DESC_ENTRIES)
		return -EINVAL;

	mutex_lock(&hw->dmac.lock);

	for (i = 0; i < DMAC_DESC_SET_COUNT; ++i)
		if (hw->dmac.desc_sets[i].count == 0)
			break;
	if (i == DMAC_DESC_SET_COUNT) {
		retval = -ENOSPC;
		goto __cass_dmac_desc_set_common_alloc_0;
	} else {
		set_id = i;
	}

	index = bitmap_find_next_zero_area(hw->dmac.desc_map,
					   C_PI_CFG_DMAC_DESC_ENTRIES,
					   idx, num_descs, 0);

	if (index >= C_PI_CFG_DMAC_DESC_ENTRIES) {
		retval = -ENOSPC;
		goto __cass_dmac_desc_set_common_alloc_0;
	}

	if (desc_idx != DMAC_DESC_INDEX_UNSPECIFIED && index != desc_idx) {
		retval = -ENOSPC;
		goto __cass_dmac_desc_set_common_alloc_0;
	}

	bitmap_set(hw->dmac.desc_map, index, num_descs);
	cass_dmac_desc_set_id_fill(hw, set_id, num_descs, index, name);

	retval = set_id;

__cass_dmac_desc_set_common_alloc_0:
	mutex_unlock(&hw->dmac.lock);
	return retval;
}

/**
 * cxi_dmac_desc_set_reserve() - reserve dmac descriptors
 * @cdev: the device
 * @num_descs: number of descriptors to reserve
 * @desc_idx: starting NIC dmac descriptor
 * @name: descriptive name to associate the dmac descriptor set
 *        where the string must remain valid for the lifetime of the set
 *
 * reserve 'num_descs' contiguous from the NIC dmac descriptor pool
 * starting with index 'desc_idx'
 *
 * Return: dmac descriptor set id on success, negative errno on failure
 */
int cxi_dmac_desc_set_reserve(struct cxi_dev *cdev, u16 num_descs,
			      u16 desc_idx, const char *name)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval;

	cxidev_dbg(cdev, "hw=%p num_descs=%u desc_idx=%u\n", hw, num_descs,
		   desc_idx);

	if (desc_idx >= C_PI_CFG_DMAC_DESC_ENTRIES)
		return -EINVAL;

	retval = cass_dmac_desc_set_common_alloc(hw, num_descs, desc_idx,
						 name);
	return retval;
}
EXPORT_SYMBOL(cxi_dmac_desc_set_reserve);

/**
 * cxi_dmac_desc_set_alloc() - allocate dmac descriptors
 * @cdev: the device
 * @num_descs: number of descriptors to reserve
 * @name: descriptive name to associate the dmac descriptor set
 *        where the string must remain valid for the lifetime of the set
 *
 * allocate 'num_descs' contiguous from the NIC dmac descriptor free pool
 *
 * Return: dmac descriptor set id on success, negative errno on failure
 */
int cxi_dmac_desc_set_alloc(struct cxi_dev *cdev, u16 num_descs,
			    const char *name)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval;

	cxidev_dbg(cdev, "hw=%p num_descs=%u\n", hw, num_descs);

	retval = cass_dmac_desc_set_common_alloc(hw, num_descs,
						 DMAC_DESC_INDEX_UNSPECIFIED,
						 name);

	return retval;
}
EXPORT_SYMBOL(cxi_dmac_desc_set_alloc);

/**
 * cxi_dmac_desc_set_free() - free dmac descriptors
 * @cdev: the device
 * @set_id: dmac descriptor set id
 *
 * free the descriptors associated with 'set_id' returning them
 * back to the free pool
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_dmac_desc_set_free(struct cxi_dev *cdev, int set_id)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval = 0;
	u16 count;
	u16 index;

	cxidev_dbg(cdev, "hw=%p set_id=%d\n", hw, set_id);

	mutex_lock(&hw->dmac.lock);

	retval = cass_dmac_desc_set_id_verify(hw, set_id);
	if (retval != 0)
		goto __cxi_dmac_desc_set_free_0;

	count = hw->dmac.desc_sets[set_id].count;
	index = hw->dmac.desc_sets[set_id].index;

	dmac_desc_zero_n(hw, index, count);
	bitmap_clear(hw->dmac.desc_map, index, count);

	cass_dmac_desc_set_id_clear(hw, set_id);

__cxi_dmac_desc_set_free_0:
	mutex_unlock(&hw->dmac.lock);
	return retval;
}
EXPORT_SYMBOL(cxi_dmac_desc_set_free);

/**
 * cxi_dmac_xfer() - start dma & wait/block for completion
 * @cdev: the device
 * @set_id: dmac descriptor set id
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_dmac_xfer(struct cxi_dev *cdev, int set_id)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	volatile struct c_pi_dmac_cdesc *cpl_desc = hw->dmac.cpl_desc;
	int retval;

	cxidev_dbg(&hw->cdev, "hw=%p set_id=%d\n", hw, set_id);

	mutex_lock(&hw->dmac.lock);

	retval = cass_dmac_desc_set_id_verify(hw, set_id);
	if (retval != 0)
		goto __cxi_dmac_xfer_0;

	if (hw->dmac.desc_sets[set_id].numused == 0) {
		retval = -EINVAL;
		goto __cxi_dmac_xfer_0;
	}

	dmac_enable(hw, hw->dmac.desc_sets[set_id].index);

	wait_for_completion(&hw->dmac.irupt);

	if (cpl_desc->done == 0) {
		cxidev_err(cdev,
			   "ERROR: failed to get DMAC completion\n");
		retval = -ETIME;
	}
	if (cpl_desc->status != C_PI_DMAC_CDESC_STATUS_AXI_RESP_OKAY) {
		cxidev_err(cdev,
			   "ERROR: DMAC_CDESC_STATUS=%x is 'not OK'\n",
			   cpl_desc->status);
		retval = -EBADE;
	}

	cxidev_dbg(&hw->cdev, "hw->dmac.cpl_desc=%016llx\n", *((u64 *)cpl_desc));

	memset((void *)cpl_desc, 0, sizeof(*cpl_desc));

	/*
	 * Being paranoid and disabling the "done hw" for the normal case
	 * but in error case, including the timeout case, the dmac engine
	 * may not be done & could be still enabled.
	 *
	 * The next version of this becomes interrupt driven and the
	 * further paranoia wants to disable irupts in dmac when done
	 * to prevent 'erroneous spurious interrupts'.
	 */
	dmac_disable(hw);

__cxi_dmac_xfer_0:
	mutex_unlock(&hw->dmac.lock);
	return retval;
}
EXPORT_SYMBOL(cxi_dmac_xfer);

static irqreturn_t cass_dmac_irupt_handler(int irq, void *p)
{
	struct completion *x = (struct completion *)p;

	complete(x);
	return IRQ_HANDLED;
}

/*
 * Initialize DMAC (DMA Controller)
 */
int cass_dmac_init(struct cass_dev *hw)
{
	struct device *dma_dev = &hw->cdev.pdev->dev;
	union c_pi_cfg_dmac_cpl_addr dmac_cpl = {};
	int i;
	int retval;
	int vec;

	dmac_disable(hw);
	/*
	 * Configure the DMAC to write a completion event to cpl_desc
	 */
	hw->dmac.cpl_desc = dma_alloc_coherent(dma_dev, sizeof(*hw->dmac.cpl_desc),
					       &hw->dmac.cpl_desc_dma_addr,
					       GFP_KERNEL);
	memset(hw->dmac.cpl_desc, 0, sizeof(*hw->dmac.cpl_desc));
	dmac_cpl.address = hw->dmac.cpl_desc_dma_addr >> 3;
	cass_write(hw, C_PI_CFG_DMAC_CPL_ADDR, &dmac_cpl, sizeof(dmac_cpl));

	dmac_desc_zero_all(hw);	/* the zero will cass_flush_pci() */

	mutex_init(&hw->dmac.lock);

	init_completion(&hw->dmac.irupt);

	hw->dmac.desc_sets = kcalloc(DMAC_DESC_SET_COUNT,
				     sizeof(*hw->dmac.desc_sets), GFP_KERNEL);

	if (hw->dmac.desc_sets == NULL) {
		retval = -ENOMEM;
		goto __cass_dmac_init_0;
	}

	for (i = 0; i < DMAC_DESC_SET_COUNT; ++i)
		cass_dmac_desc_set_id_clear(hw, i);

	hw->dmac.desc_map = bitmap_zalloc(C_PI_CFG_DMAC_DESC_ENTRIES,
					  GFP_KERNEL);
	if (hw->dmac.desc_map == NULL) {
		retval = -ENOMEM;
		goto __cass_dmac_init_1;
	}
	scnprintf(hw->dmac.irupt_name, sizeof(hw->dmac.irupt_name),
		  "%s_irupt_dmac", hw->cdev.name);
	vec = pci_irq_vector(hw->cdev.pdev, C_PI_DMAC_MSIX_INT);
	retval = request_irq(vec, cass_dmac_irupt_handler, 0,
			     hw->dmac.irupt_name, &hw->dmac.irupt);
	if (retval != 0)
		goto __cass_dmac_init_2;

	return 0;

__cass_dmac_init_2:
	bitmap_free(hw->dmac.desc_map);

__cass_dmac_init_1:
	kfree(hw->dmac.desc_sets);

__cass_dmac_init_0:
	mutex_destroy(&hw->dmac.lock);
	dmac_cpl.address = 0;
	cass_write(hw, C_PI_CFG_DMAC_CPL_ADDR, &dmac_cpl, sizeof(dmac_cpl));

	return retval;
}

/*
 * Finish/finalize DMAC (DMA Controller)
 */
void cass_dmac_fini(struct cass_dev *hw)
{
	struct device *dma_dev = &hw->cdev.pdev->dev;
	int vec;

	dmac_disable(hw);
	vec = pci_irq_vector(hw->cdev.pdev, C_PI_DMAC_MSIX_INT);
	free_irq(vec, &hw->dmac.irupt);
	cass_clear(hw, C_PI_CFG_DMAC_CPL_ADDR, C_PI_CFG_DMAC_CPL_ADDR_SIZE);
	dmac_desc_zero_all(hw);
	dma_free_coherent(dma_dev, sizeof(*hw->dmac.cpl_desc),
			  hw->dmac.cpl_desc, hw->dmac.cpl_desc_dma_addr);
	hw->dmac.cpl_desc = NULL;
	bitmap_free(hw->dmac.desc_map);
	kfree(hw->dmac.desc_sets);
	mutex_destroy(&hw->dmac.lock);
}
