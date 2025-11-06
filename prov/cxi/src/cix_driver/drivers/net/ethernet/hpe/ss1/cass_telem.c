// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2021,2024 Hewlett Packard Enterprise Development LP */

/* Cassini Telemetry */

  /*
   * this affects this file and cassini-telemetry-sysfs-defs.h
   */
//# define CASSINI_TELEM_DEBUG (1)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <linux/iopoll.h>

#include "cass_core.h"
#include "cassini-telemetry-items.h"
#include "cassini-telemetry-sysfs-defs.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static int telem_item_retrieve(struct cass_dev *hw, const unsigned int item,
			       u64 *value)
{
	loff_t offset;
	u8 lsb;
	u8 numbits;
	u8 type;
	u64 rawval;
	u64 val;
	u64 mask;

	if (item >= hw->telemetry.info->total_num_items)
		return -EINVAL;

	offset	= ((loff_t)hw->telemetry.info->items[item].offset);
	lsb	= hw->telemetry.info->items[item].lsb;
	numbits = hw->telemetry.info->items[item].numbits;
	type	= hw->telemetry.info->items[item].type;

#	if (defined(CASSINI_TELEM_DEBUG))
	{
		cxidev_info(&hw->cdev,
			    "%s():\n"
			    "       item=%u\n"
			    "       name=%s\n"
			    "     offset=0x%08lx=%ld\n"
			    "        lsb=%u\n"
			    "    numbits=%u\n"
			    "       type=%u\n",
			    __func__, item, ctelem_item_names[item],
			    ((long)offset), ((long)offset),
			    lsb, numbits, type);
	}
#	endif

	if (type == CASSINI_TELEMETRY_ITEM_TYPE_UNDEFINED
			|| lsb >= 64
			|| numbits == 0
			|| numbits > 64)
		return -EINVAL;

	cass_read(hw, offset, &rawval, sizeof(rawval));

	mask = (~0ULL >> (64 - numbits));
	val = rawval;
	val >>= lsb;
	val &= mask;

#	if (defined(CASSINI_TELEM_DEBUG))
	{
		cxidev_info(&hw->cdev,
			    "%s():\n"
			    "     rawval=0x%016llx=%llu\n"
			    "        val=0x%016llx=%llu\n"
			    "        lsb=%u\n"
			    "       mask=0x%016llx=%llu\n",
			    __func__, rawval, rawval, val, val,
			    lsb, mask, mask);
	}
#	endif

	*value = val;
	return 0;
}

/* sysfs telemetry file output method. */
static ssize_t telem_sysfs_item_retrieve(struct kobject *kobj,
					 struct kobj_attribute *kattr,
					 char *buf)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev,
					   telemetry.kobj_items);
	struct ctelem_attr *attr = container_of(kattr, struct ctelem_attr,
						kattr);
	int retval;
	u64 value;
	struct timespec64 ts;

	retval = telem_item_retrieve(hw, attr->item, &value);
	if (retval == 0) {
		ts = ktime_to_timespec64(ktime_get_real());
		retval = snprintf(buf, PAGE_SIZE, "%llu@%lld.%09lu\n",
				  value, ts.tv_sec, ts.tv_nsec);
	}
	return retval;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * cxi_telem_get_selected() - get values for supplied telemetry items
 * @cdev: the device
 * @items: array indicating which items to retrieve
 * @data: array to hold the items's values
 * @count: number of elements in each array (items and data)
 *
 * retrieve the values for the requested telemetry items.
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_telem_get_selected(struct cxi_dev *cdev, const unsigned int *items,
			   u64 *data, unsigned int count)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	int retval = 0;
	int i;
	unsigned int item;

	if (count == 0 || count > hw->telemetry.info->valid_num_items)
		return -EINVAL;

	/*
	 * Verify the requested items before retrieving any data from
	 * the pcie bus.
	 */
	for (i = 0; i < count; ++i) {
		/*
		 * Items's values range from [0..(valid_num_items-1)]
		 * inclusively
		 */
		item = items[i];
		if (item == C_TELEM_UNDEFINED
				|| item >= hw->telemetry.info->total_num_items)
			return -EINVAL;

		/*
		 * Requesting an UNDEFINED item for this cassini asic version
		 * is wrong since the caller should know which would be valid.
		 */
		if (hw->telemetry.info->items[item].type
				== CASSINI_TELEMETRY_ITEM_TYPE_UNDEFINED)
			return -EINVAL;
	}
	/*
	 * Get the actual values now.
	 */
	for (i = 0; i < count; ++i) {
		retval = telem_item_retrieve(hw, items[i], &data[i]);
		if (retval != 0)
			break;
	}

	return retval;
}
EXPORT_SYMBOL(cxi_telem_get_selected);

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* Toggle all counter group enable and init bits. */
static int telem_cntr_group_toggle(struct cass_dev *hw, bool enable, bool init)
{
	/*
	 * While the c_*_cfg_event_cnts registers & unions are very
	 * similar, there are two variants one with and one without
	 * the warm_rst_disable field, and since each counter group
	 * defines its own cfg reg & union and may theoretically change
	 * without corresponding changes in the other groups cfg reg,
	 * use each group cfg reg independently versus using a single
	 * u64 (qw) for all.   Let the compiler determine if any
	 * optimizations may be done.
	 */

	union c_mb_cfg_event_cnts	cfg_mb;
	union c_pi_ipd_cfg_event_cnts	cfg_pi_ipd;
	union c_cq_cfg_event_cnts	cfg_cq;
	union c_lpe_cfg_event_cnts	cfg_lpe;
	union c_hni_cfg_event_cnts	cfg_hni;

	cfg_mb.qw	  = 0;
	cfg_mb.enable	  = enable;
	cfg_mb.init	  = init;

	cfg_pi_ipd.qw	  = 0;
	cfg_pi_ipd.enable = enable;
	cfg_pi_ipd.init	  = init;

	cfg_cq.qw	  = 0;
	cfg_cq.enable	  = enable;
	cfg_cq.init	  = init;

	cfg_lpe.qw	  = 0;
	cfg_lpe.enable	  = enable;
	cfg_lpe.init	  = init;

	cfg_hni.qw	  = 0;
	cfg_hni.enable	  = enable;
	cfg_hni.init	  = init;

	cass_write(hw, C_MB_CFG_EVENT_CNTS,	&cfg_mb,     sizeof(cfg_mb));
	cass_write(hw, C_PI_IPD_CFG_EVENT_CNTS, &cfg_pi_ipd, sizeof(cfg_pi_ipd));
	cass_write(hw, C_CQ_CFG_EVENT_CNTS,	&cfg_cq,     sizeof(cfg_cq));
	cass_write(hw, C_LPE_CFG_EVENT_CNTS,	&cfg_lpe,    sizeof(cfg_lpe));
	cass_write(hw, C_HNI_CFG_EVENT_CNTS,	&cfg_hni,    sizeof(cfg_hni));

	cass_flush_pci(hw);

	if (!init)
		return 0;

	if (readq_poll_timeout(cass_csr(hw, C_MB_CFG_EVENT_CNTS),
			       cfg_mb.qw, cfg_mb.init_done == 1, 1, 1000000)) {
		cxidev_err(&hw->cdev,
			   "Telemetry counter group reset failed: %x\n",
			   C_MB_CFG_EVENT_CNTS);
		return -1;
	}

	if (readq_poll_timeout(cass_csr(hw, C_PI_IPD_CFG_EVENT_CNTS),
			       cfg_pi_ipd.qw,
			       cfg_pi_ipd.init_done == 1, 1, 1000000)) {
		cxidev_err(&hw->cdev,
			   "Telemetry counter group reset failed: %x\n",
			   C_PI_IPD_CFG_EVENT_CNTS);
		return -1;
	}

	if (readq_poll_timeout(cass_csr(hw, C_CQ_CFG_EVENT_CNTS),
			       cfg_cq.qw, cfg_cq.init_done == 1, 1, 1000000)) {
		cxidev_err(&hw->cdev,
			   "Telemetry counter group reset failed: %x\n",
			   C_CQ_CFG_EVENT_CNTS);
		return -1;
	}

	if (readq_poll_timeout(cass_csr(hw, C_LPE_CFG_EVENT_CNTS),
			       cfg_lpe.qw, cfg_lpe.init_done == 1, 1, 1000000)) {
		cxidev_err(&hw->cdev,
			   "Telemetry counter group reset failed: %x\n",
			   C_LPE_CFG_EVENT_CNTS);
		return -1;
	}

	if (readq_poll_timeout(cass_csr(hw, C_HNI_CFG_EVENT_CNTS),
			       cfg_hni.qw, cfg_hni.init_done == 1, 1, 1000000)) {
		cxidev_err(&hw->cdev,
			   "Telemetry counter group reset failed: %x\n",
			   C_HNI_CFG_EVENT_CNTS);
		return -1;
	}

	return 0;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int cass_telem_init(struct cass_dev *hw)
{
	int rc;
	struct kobj_type *pkt;

	if (cass_version(hw, CASSINI_1)) {
		cxidev_info(&hw->cdev, "telemetry for cassini1\n");
		hw->telemetry.info = &c1;
		pkt = &c1telem_kobj_type;
	} else if (cass_version(hw, CASSINI_2)) {
		cxidev_info(&hw->cdev, "telemetry for cassini2\n");
		hw->telemetry.info = &c2;
		pkt = &c2telem_kobj_type;
	} else {
		cxidev_err(&hw->cdev,
			   "telemetry: unsupported cassini version = %x\n",
			   hw->cdev.prop.cassini_version);
		rc = -EOPNOTSUPP;
		goto __cass_telem_init_err_ret_rc;
	}

	if (hw->cdev.is_physfn) {
		rc = telem_cntr_group_toggle(hw, true, true);
		if (rc != 0) {
			cxidev_err(&hw->cdev,
				   "telem_cntr_group_toggle failed with %d\n", rc);
			/*
			 * going to __cass_telem_init_err_cntr_group_toggle to
			 * have driver try to write disable for the functionality.
			 */
			goto __cass_telem_init_err_cntr_group_toggle;
		}
	}

	rc = kobject_init_and_add(&hw->telemetry.kobj_items, pkt,
				  &hw->cdev.pdev->dev.kobj, "telemetry");
	if (rc != 0)
		goto __cass_telem_init_err_cntr_group_toggle;

	return 0;

__cass_telem_init_err_cntr_group_toggle:
	if (hw->cdev.is_physfn) {
		/*
		 * since the 'init parameter' is false, the following will not fail.
		 */
		(void)telem_cntr_group_toggle(hw, false, false);
	}
	hw->telemetry.info = NULL;

__cass_telem_init_err_ret_rc:
	return rc;
}

void cass_telem_fini(struct cass_dev *hw)
{
	kobject_put(&hw->telemetry.kobj_items);
	if (hw->cdev.is_physfn) {
		/*
		 * since the 'init parameter' is false, the following will not fail.
		 */
		(void)telem_cntr_group_toggle(hw, false, false);
	}
	hw->telemetry.info = NULL;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
