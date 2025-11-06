// SPDX-License-Identifier: GPL-2.0
/* Copyright 2024 Hewlett Packard Enterprise Development LP */

#include "cass_core.h"

int cxi_dev_info_get(struct cxi_dev *dev,
		     struct cxi_dev_info_use *devinfo)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);
	int speed;
	int state;

	devinfo->nid = hw->cdev.prop.nid;
	devinfo->pid_bits = hw->cdev.prop.pid_bits;
	devinfo->pid_count = hw->cdev.prop.pid_count;
	devinfo->pid_granule = hw->cdev.prop.pid_granule;
	devinfo->min_free_shift = hw->cdev.prop.min_free_shift;
	devinfo->rdzv_get_idx = hw->cdev.prop.rdzv_get_idx;
	devinfo->device_rev = hw->cdev.prop.device_rev;
	devinfo->device_proto = hw->cdev.prop.device_proto;
	devinfo->device_platform = hw->cdev.prop.device_platform;
	devinfo->vendor_id = hw->rev.vendor_id;
	devinfo->device_id = hw->rev.device_id;
	devinfo->pct_eq = hw->pct_eq_n;
	devinfo->uc_nic = hw->uc_nic;
	devinfo->link_mtu = C_MAX_HPC_MTU;
	devinfo->num_ptes = hw->cdev.prop.rsrcs.ptes.max;
	devinfo->num_txqs = hw->cdev.prop.rsrcs.txqs.max;
	devinfo->num_tgqs = hw->cdev.prop.rsrcs.tgqs.max;
	devinfo->num_eqs = hw->cdev.prop.rsrcs.eqs.max;
	devinfo->num_cts = hw->cdev.prop.rsrcs.cts.max;
	devinfo->num_acs = hw->cdev.prop.rsrcs.acs.max;
	devinfo->num_tles = hw->cdev.prop.rsrcs.tles.max;
	devinfo->num_les = hw->cdev.prop.rsrcs.les.max;
	devinfo->cassini_version = hw->cdev.prop.cassini_version;
	devinfo->system_type_identifier = hw->cdev.system_type_identifier;

	if (hw->port)
		cass_sbl_link_mode_to_speed(hw->port->lattr.bl.link_mode, &speed);
	else
		speed = 0;
	devinfo->link_speed = speed;

	state = hw->phy.state == CASS_PHY_RUNNING;
	devinfo->link_state = state;

	if (!hw->fru_info_valid)
		uc_cmd_get_fru(hw);

	if (hw->fru_info[PLDM_FRU_FIELD_DESCRIPTION])
		strscpy(devinfo->fru_description, hw->fru_info[PLDM_FRU_FIELD_DESCRIPTION],
			sizeof(devinfo->fru_description) - 1);
	else
		strscpy(devinfo->fru_description, "Not Available",
			sizeof(devinfo->fru_description) - 1);
	devinfo->fru_description[sizeof(devinfo->fru_description) - 1] = '\0';

	return 0;
}
EXPORT_SYMBOL(cxi_dev_info_get);
