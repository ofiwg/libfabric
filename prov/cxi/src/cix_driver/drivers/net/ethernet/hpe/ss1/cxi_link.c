// SPDX-License-Identifier: GPL-2.0
/*
 * Cassini LINK
 * Copyright 2022,2024 Hewlett Packard Enterprise Development LP
 */

#include <linux/kernel.h>
#include <linux/hpe/cxi/cxi.h>

#include "cass_core.h"
#include "cass_sbl.h"
#include "cass_sl.h"

static const struct cxi_link_ops cxi_link_ops_sbl = {
	.init = cass_sbl_init,
	.link_start = cass_sbl_link_start,
	.link_fini = cass_sbl_link_fini,
	.mode_get  = cass_sbl_mode_get,
	.mode_set  = cass_sbl_mode_set,
	.flags_get = cass_sbl_get_debug_flags,
	.flags_set = cass_sbl_set_debug_flags,
	.link_up   = cass_sbl_power_on,
	.link_down = cass_sbl_power_off,
	.is_pcs_aligned = cass_sbl_pml_pcs_aligned,
	.media_config = cass_sbl_media_config,
	.media_unconfig = cass_sbl_media_unconfig,
	.link_config  = cass_sbl_link_config,
	.link_reset = cass_sbl_reset,
	.link_exit = cass_sbl_exit,
	.pml_recovery_set = cass_sbl_pml_recovery_set,
	.is_link_up = cass_sbl_is_link_up,
	.eth_name_set = cass_sbl_set_eth_name,
};

static const struct cxi_link_ops cxi_link_ops_sl = {
	.init               = cass_sl_init,
	.link_start         = cass_sbl_link_start, /* TODO - temporary */
	.link_fini          = cass_sl_link_fini,
	.mode_get           = cass_sl_mode_get,
	.mode_set           = cass_sl_mode_set,
	.flags_get          = cass_sl_flags_get,
	.flags_set          = cass_sl_flags_set,
	.link_up            = cass_sl_link_up,
	.link_down          = cass_sl_link_down,
	.is_pcs_aligned     = cass_sl_is_pcs_aligned,
	.media_config       = cass_sl_media_config,
	.media_unconfig     = cass_sl_media_unconfig,
	.link_config        = cass_sl_link_config,
	.link_reset         = cass_sl_link_down,
	.link_exit          = cass_sl_exit,
	.pml_recovery_set   = cass_sl_pml_recovery_set,
	.is_link_up	    = cass_sl_is_link_up,
	.eth_name_set	    = cass_sl_connect_id_set,
};

void cxi_set_link_ops(struct cass_dev *hw)
{
	if (cass_version(hw, CASSINI_1))
		hw->link_ops = &cxi_link_ops_sbl;
	else
		hw->link_ops = &cxi_link_ops_sl;
}

void cxi_link_mode_get(struct cxi_dev *cxi_dev, struct cxi_link_info *link_info)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	hw->link_ops->mode_get(hw, link_info);
}
EXPORT_SYMBOL(cxi_link_mode_get);

void cxi_link_mode_set(struct cxi_dev *cxi_dev, const struct cxi_link_info *link_info)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	hw->link_ops->mode_set(hw, link_info);
}
EXPORT_SYMBOL(cxi_link_mode_set);

void cxi_link_flags_get(struct cxi_dev *cxi_dev, u32 *flags)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	hw->link_ops->flags_get(hw, flags);
}
EXPORT_SYMBOL(cxi_link_flags_get);

void cxi_link_flags_set(struct cxi_dev *cxi_dev, u32 clr_flags, u32 set_flags)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	hw->link_ops->flags_set(hw, clr_flags, set_flags);
}
EXPORT_SYMBOL(cxi_link_flags_set);

void cxi_link_use_unsupported_cable(struct cxi_dev *cxi_dev, bool use)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	cxidev_dbg(&hw->cdev, "use unsupported cable\n");

	if (use)
		hw->sl.link_policy.options |= SL_LINK_POLICY_OPT_USE_UNSUPPORTED_CABLE;
	else
		hw->sl.link_policy.options &= ~SL_LINK_POLICY_OPT_USE_UNSUPPORTED_CABLE;
}
EXPORT_SYMBOL(cxi_link_use_unsupported_cable);

void cxi_link_use_supported_ss200_cable(struct cxi_dev *cxi_dev, bool use)
{
        struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

        cxidev_dbg(&hw->cdev, "use supported ss200 cable\n");

        if (use)
                hw->sl.link_policy.options |= SL_LINK_POLICY_OPT_USE_SUPPORTED_SS200_CABLE;
        else
                hw->sl.link_policy.options &= ~SL_LINK_POLICY_OPT_USE_SUPPORTED_SS200_CABLE;
}
EXPORT_SYMBOL(cxi_link_use_supported_ss200_cable);

void cxi_link_ignore_media_error(struct cxi_dev *cxi_dev, bool ignore)
{
        struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

        cxidev_dbg(&hw->cdev, "ignore media error\n");

        if (ignore)
                hw->sl.link_policy.options |= SL_LINK_POLICY_OPT_IGNORE_MEDIA_ERROR;
        else
                hw->sl.link_policy.options &= ~SL_LINK_POLICY_OPT_IGNORE_MEDIA_ERROR;
}
EXPORT_SYMBOL(cxi_link_ignore_media_error);

void cxi_link_auto_lane_degrade(struct cxi_dev *cxi_dev, bool enable)
{
       struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

       cxidev_dbg(&hw->cdev, "auto lane degrade\n");

       if (enable)
               hw->sl.link_config.options |= SL_LINK_CONFIG_OPT_ALD_ENABLE;
       else
               hw->sl.link_config.options &= ~SL_LINK_CONFIG_OPT_ALD_ENABLE;
}
EXPORT_SYMBOL(cxi_link_auto_lane_degrade);

void cxi_link_fec_monitor(struct cxi_dev *cxi_dev, bool on)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	cxidev_dbg(&hw->cdev, "fec_monitor\n");

	hw->sl.link_policy.fec_mon_period_ms = on ? -1 : 0;
}
EXPORT_SYMBOL(cxi_link_fec_monitor);

void cxi_pml_recovery_set(struct cxi_dev *cxi_dev, bool set)
{
	struct cass_dev *hw = container_of(cxi_dev, struct cass_dev, cdev);

	hw->link_ops->pml_recovery_set(hw, set);
}
EXPORT_SYMBOL(cxi_pml_recovery_set);

/**
 * cxi_is_link_up() - Returns whether the link is up or down
 *
 * @cdev: the device
 *
 * Return: true if the link is up, false otherwise.
 */
bool cxi_is_link_up(struct cxi_dev *cdev)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	return hw->link_ops->is_link_up(hw);
}
EXPORT_SYMBOL(cxi_is_link_up);
