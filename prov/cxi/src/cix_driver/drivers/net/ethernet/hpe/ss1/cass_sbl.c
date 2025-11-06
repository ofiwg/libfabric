// SPDX-License-Identifier: GPL-2.0
/*
 * Cassini device handler
 * Copyright 2020-2024 Hewlett Packard Enterprise Development LP
 */

#include <linux/iopoll.h>
#include <linux/hpe/cxi/cxi.h>
#include <sbl/sbl_mb.h>
#include <linux/sbl.h>
#include <uapi/sbl.h>
#include <uapi/sbl_serdes.h>
#include <uapi/sbl_cassini.h>

#include "cass_core.h"
#include "cass_cable.h"
#include "cxi_ethtool.h"
#include "cass_ss1_debugfs.h"

#define SYSFS_BUFSIZE        1000

static unsigned int pml_rec_timeout = SBL_DFLT_PML_REC_TIMEOUT;
module_param(pml_rec_timeout, uint, 0644);
MODULE_PARM_DESC(pml_rec_timeout, "PML recovery timeout in ms");

static unsigned int pml_rec_rl_max_dur = SBL_DFLT_PML_REC_RL_MAX_DURATION;
module_param(pml_rec_rl_max_dur, uint, 0644);
MODULE_PARM_DESC(pml_rec_rl_max_dur, "PML recovery max time in ms per window");

static unsigned int pml_rec_rl_win_size = SBL_DFLT_PML_REC_RL_WINDOW_SIZE;
module_param(pml_rec_rl_win_size, uint, 0644);
MODULE_PARM_DESC(pml_rec_rl_win_size, "PML recovery rate limiter window in ms");

/**
 * cass_sbl_link_attr_cmp() - Check if input attr matches device attr
 *
 *  Checks each member of input cass_link_attr against value stored in
 *  device. If any values differ, return -1, else return 0.
 *
 * @hw: the device
 * @tgt_val: comparison values
 *
 * Return: 0 on if the structs are the same, else -1
 */
static int cass_sbl_link_attr_cmp(const struct cass_dev *hw,
				  const struct cass_link_attr *tgt_val)
{
	const struct cass_port *port = hw->port;

	if ((tgt_val->options != port->lattr.options) ||
	    (tgt_val->bl.options != port->lattr.bl.options) ||
	    (tgt_val->bl.magic != port->lattr.bl.magic) ||
	    (tgt_val->bl.start_timeout != port->lattr.bl.start_timeout) ||
	    (tgt_val->bl.config_target != port->lattr.bl.config_target) ||
	    (tgt_val->bl.pec.an_mode != port->lattr.bl.pec.an_mode) ||
	    (tgt_val->bl.pec.an_retry_timeout != port->lattr.bl.pec.an_retry_timeout) ||
	    (tgt_val->bl.pec.an_max_retry != port->lattr.bl.pec.an_max_retry) ||
	    (tgt_val->bl.lpd_timeout != port->lattr.bl.lpd_timeout) ||
	    (tgt_val->bl.lpd_poll_interval != port->lattr.bl.lpd_poll_interval) ||
	    (tgt_val->bl.link_mode != port->lattr.bl.link_mode) ||
	    (tgt_val->bl.loopback_mode != port->lattr.bl.loopback_mode) ||
	    (tgt_val->bl.link_partner != port->lattr.bl.link_partner) ||
	    (tgt_val->bl.tuning_pattern != port->lattr.bl.tuning_pattern) ||
	    (tgt_val->bl.precoding != port->lattr.bl.precoding) ||
	    (tgt_val->bl.dfe_pre_delay != port->lattr.bl.dfe_pre_delay) ||
	    (tgt_val->bl.dfe_timeout != port->lattr.bl.dfe_timeout) ||
	    (tgt_val->bl.dfe_poll_interval != port->lattr.bl.dfe_poll_interval) ||
	    (tgt_val->bl.pcal_eyecheck_holdoff != port->lattr.bl.pcal_eyecheck_holdoff) ||
	    (tgt_val->bl.nrz_min_eye_height != port->lattr.bl.nrz_min_eye_height) ||
	    (tgt_val->bl.nrz_max_eye_height != port->lattr.bl.nrz_max_eye_height) ||
	    (tgt_val->bl.pam4_min_eye_height != port->lattr.bl.pam4_min_eye_height) ||
	    (tgt_val->bl.pam4_max_eye_height != port->lattr.bl.pam4_max_eye_height) ||
	    (tgt_val->bl.fec_mode != port->lattr.bl.fec_mode) ||
	    (tgt_val->bl.enable_autodegrade != port->lattr.bl.enable_autodegrade) ||
	    (tgt_val->bl.llr_mode != port->lattr.bl.llr_mode) ||
	    (tgt_val->bl.ifg_config != port->lattr.bl.ifg_config) ||
	    (tgt_val->bl.pml_recovery.timeout != port->lattr.bl.pml_recovery.timeout) ||
	    (tgt_val->bl.pml_recovery.rl_max_duration !=
		port->lattr.bl.pml_recovery.rl_max_duration) ||
	    (tgt_val->bl.pml_recovery.rl_window_size !=
		port->lattr.bl.pml_recovery.rl_window_size))
		return -1;

	return 0;
}

/* Initializes the sbl module. */
int cass_sbl_link_start(struct cass_dev *hw)
{
	int ret;

	cxidev_dbg(&hw->cdev, "initializing SBL...\n");

	ret = cass_port_new_port_db(hw);
	if (ret) {
		cxidev_err(&hw->cdev, "cass_port_new_port_db failed [%d]\n", ret);
		goto out;
	}

	ret = hw->link_ops->init(hw);
	if (ret != 0) {
		cxidev_err(&hw->cdev, "init failed [%d]\n", ret);
		goto out_ports;
	}

	hw->port->hstate = CASS_HEADSHELL_STATUS_NOT_PRESENT;

	ret = cass_lmon_start_all(hw);
	if (ret) {
		cxidev_err(&hw->cdev, "cass_lmon_start_all failed [%d]\n", ret);
		goto out_sbl;
	}

	ret = cass_cable_scan(hw);
	if (ret) {
		cxidev_err(&hw->cdev, "cass_cable_scan failed [%d]\n", ret);
		goto out_lmon;
	}

	cass_sbl_set_defaults(hw);
	ret = cass_sbl_configure(hw);
	if (ret != 0) {
		cxidev_err(&hw->cdev, "cass_sbl_configure failed [%d]\n", ret);
		goto out_lmon;
	}
	cxidev_dbg(&hw->cdev, "SBL initialized\n");

	return 0;

out_lmon:
	cass_lmon_kill_all(hw);
out_sbl:
	hw->link_ops->link_fini(hw);
out_ports:
	cass_port_del_port_db(hw);
out:
	return ret;
}

void cass_sbl_link_fini(struct cass_dev *hw)
{
	cass_lmon_kill_all(hw);
	cass_sbl_exit(hw);
	cass_port_del_port_db(hw);
}

/* getters/setters follow */

/**
 * cass_sbl_mode_set() - Setter for various link configuration options
 *
 * Updates the cass_dev cass_link_attr with the target value.
 * If there are any differences between the requested config and the
 * running config, the link will be torn down and brought back up with
 * the new config.
 *
 * @hw: the device
 * @link_info: CXI ethtool private flags
 */
void cass_sbl_mode_set(struct cass_dev *hw, const struct cxi_link_info *link_info)
{
	struct cass_link_attr attrs = hw->port->lattr;

	switch (link_info->flags & LOOPBACK_MODE) {
	case 0:
		attrs.bl.loopback_mode = SBL_LOOPBACK_MODE_OFF;
		break;
	case CXI_ETH_PF_INTERNAL_LOOPBACK:
		attrs.bl.loopback_mode = SBL_LOOPBACK_MODE_LOCAL;
		break;
	case CXI_ETH_PF_EXTERNAL_LOOPBACK:
		attrs.bl.loopback_mode = SBL_LOOPBACK_MODE_REMOTE;
		break;
	default:
		/* Invalid configuration, no change */
		break;
	}

	if (link_info->flags & CXI_ETH_PF_LLR)
		attrs.bl.llr_mode = SBL_LLR_MODE_ON;
	else
		attrs.bl.llr_mode = SBL_LLR_MODE_OFF;

	if (link_info->flags & CXI_ETH_PF_PRECODING)
		attrs.bl.precoding = SBL_PRECODING_ON;
	else
		attrs.bl.precoding = SBL_PRECODING_OFF;

	if (link_info->flags & CXI_ETH_PF_IFG_HPC)
		attrs.bl.ifg_config = SBL_IFG_CONFIG_HPC;
	else
		attrs.bl.ifg_config = SBL_IFG_CONFIG_IEEE_200G;

	if (link_info->flags & CXI_ETH_PF_DISABLE_PML_RECOVERY)
		attrs.bl.options |= SBL_DISABLE_PML_RECOVERY;
	else
		attrs.bl.options &= ~SBL_DISABLE_PML_RECOVERY;

	if (attrs.mattr.media != SBL_LINK_MEDIA_OPTICAL) {
		if (link_info->autoneg == AUTONEG_DISABLE)
			attrs.bl.pec.an_mode = SBL_AN_MODE_OFF;
		else
			attrs.bl.pec.an_mode = SBL_AN_MODE_ON;
	}

	if (!cass_sbl_link_attr_cmp(hw, &attrs))
		hw->link_config_dirty = false;
	else
		hw->link_config_dirty = true;

	if (hw->link_config_dirty) {
		hw->port->lattr = attrs;
		/* Bounce the link to apply the new config */
		cass_phy_bounce(hw);
	}
}

void cass_sbl_pml_recovery_set(struct cass_dev *hw, bool set)
{
	sbl_disable_pml_recovery(hw->sbl, 0, set);
}

/**
 * cass_sbl_mode_get() - Getter for various link configuration options
 *
 * Updates the input tgt_val with the cass_dev cass_link_attr contents.
 *
 * @hw: the device
 * @link_info: location to write cass_link_attr struct
 */
void cass_sbl_mode_get(struct cass_dev *hw, struct cxi_link_info *link_info)
{
	const struct cass_link_attr *attrs = &hw->port->lattr;
	int sstate;
	int blstate;
	int link_mode;

	memset(link_info, 0, sizeof(*link_info));

	sbl_base_link_get_status(hw->sbl, 0, &blstate,
		NULL, &sstate, NULL, NULL, &link_mode);

	switch (blstate) {
	case SBL_BASE_LINK_STATUS_UNKNOWN:
	case SBL_BASE_LINK_STATUS_UNCONFIGURED:
	case SBL_BASE_LINK_STATUS_RESETTING:
	case SBL_BASE_LINK_STATUS_ERROR:
		link_info->speed = SPEED_UNKNOWN;
		break;
	default:
		switch (sstate) {
		case SBL_SERDES_STATUS_UNKNOWN:
		case SBL_SERDES_STATUS_RESETTING:
		case SBL_SERDES_STATUS_ERROR:
			link_info->speed = SPEED_UNKNOWN;
			break;
		default:
			cass_sbl_link_mode_to_speed(link_mode, &link_info->speed);
		}
	}

	if (attrs->bl.loopback_mode == SBL_LOOPBACK_MODE_LOCAL)
		link_info->flags |= CXI_ETH_PF_INTERNAL_LOOPBACK;
	else if (attrs->bl.loopback_mode == SBL_LOOPBACK_MODE_REMOTE)
		link_info->flags |= CXI_ETH_PF_EXTERNAL_LOOPBACK;

	if (attrs->bl.llr_mode == SBL_LLR_MODE_ON)
		link_info->flags |= CXI_ETH_PF_LLR;

	if (attrs->bl.precoding == SBL_PRECODING_ON)
		link_info->flags |= CXI_ETH_PF_PRECODING;

	if (attrs->bl.ifg_config == SBL_IFG_CONFIG_HPC)
		link_info->flags |= CXI_ETH_PF_IFG_HPC;

	if (attrs->bl.options & SBL_DISABLE_PML_RECOVERY)
		link_info->flags |= CXI_ETH_PF_DISABLE_PML_RECOVERY;

	if (attrs->mattr.media != SBL_LINK_MEDIA_OPTICAL) {
		if (attrs->bl.pec.an_mode == SBL_AN_MODE_OFF)
			link_info->autoneg = AUTONEG_DISABLE;
		else
			link_info->autoneg = AUTONEG_ENABLE;
	}

	switch (attrs->mattr.media) {
	case SBL_LINK_MEDIA_ELECTRICAL:
		link_info->port_type = PORT_DA;
		break;
	case SBL_LINK_MEDIA_OPTICAL:
		link_info->port_type = PORT_FIBRE;
		break;
	default:
		link_info->port_type = PORT_OTHER;
		break;
	}
}

/**
 * cass_sbl_get_debug_flags() - Getter for debug flags
 *
 * Updates the input flags with the current debug flags.
 *
 * @hw: the device
 * @debug_flags: storage for private CXI flags (CXI_ETH_PF_xxx)
 */
void cass_sbl_get_debug_flags(struct cass_dev *hw, u32 *debug_flags)
{
	u32 flags;

	sbl_debug_get_config(hw->sbl, 0, &flags);

	*debug_flags = 0;

	if (flags & SBL_DEBUG_IGNORE_ALIGN)
		*debug_flags |= CXI_ETH_PF_IGNORE_ALIGN;

	if (flags & SBL_DEBUG_REMOTE_FAULT_RECOVERY)
		*debug_flags |= CXI_ETH_PF_REMOTE_FAULT_RECOVERY;
}

/**
 * cass_sbl_set_debug_flags() - Setter for debug flags
 *
 * Updates the debug flags.
 *
 * @hw: the device
 * @clr_flags: desired clear flags
 * @set_flags: desired set flags
 */
void cass_sbl_set_debug_flags(struct cass_dev *hw, u32 clr_flags, u32 set_flags)
{
	u32 clr_flags_sbl = 0;
	u32 set_flags_sbl = 0;

	if (clr_flags & CXI_ETH_PF_IGNORE_ALIGN)
		clr_flags_sbl |= SBL_DEBUG_IGNORE_ALIGN;

	if (set_flags & CXI_ETH_PF_IGNORE_ALIGN)
		set_flags_sbl |= SBL_DEBUG_IGNORE_ALIGN;

	if (clr_flags & CXI_ETH_PF_REMOTE_FAULT_RECOVERY)
		clr_flags_sbl |= SBL_DEBUG_REMOTE_FAULT_RECOVERY;

	if (set_flags & CXI_ETH_PF_REMOTE_FAULT_RECOVERY)
		set_flags_sbl |= SBL_DEBUG_REMOTE_FAULT_RECOVERY;

	sbl_debug_update_config(hw->sbl, 0, clr_flags_sbl, set_flags_sbl);
}

/* Return whether the link is up or down. */
bool cass_sbl_is_link_up(struct cass_dev *hw)
{
	enum cass_link_status state = cass_link_get_state(hw);

	return state == CASS_LINK_STATUS_UP;
}

/* sbl_ops follow */
/**
 * cass_read64() - wrapper around Cassini read function
 *
 * Read a CSR on BAR0.
 *
 * @pci_accessor: the SBL device
 * @offset: offset of the CSR in the region
 *
 * Return: u64 containing the read result
 */
static u64 cass_read64(void *pci_accessor, long offset)
{
	struct cass_dev *hw = (struct cass_dev *)pci_accessor;
	size_t len = sizeof(u64);
	u64 data_out;

	cass_read(hw, (u64)offset, &data_out, len);

	return data_out;
}

/**
 * cass_write64() - wrapper around Cassini write function
 *
 * Modify a control status register (CSR) on BAR0.
 *
 * @pci_accessor: the SBL device
 * @offset: offset of the CSR in the region
 * @val: data to write
 */
static void cass_write64(void *pci_accessor, long offset, u64 val)
{
	struct cass_dev *hw = (struct cass_dev *)pci_accessor;
	size_t len = sizeof(u64);

	cass_write(hw, (u64)offset, &val, len);
}

/**
 * cass_read32() - wrapper around Cassini 64-bit read function
 *
 * Read a CSR on BAR0.
 *
 * @pci_accessor: the SBL device
 * @offset: offset of the CSR in the region
 *
 * Return: u32 containing the read result
 */
static u32 cass_read32(void *pci_accessor, long offset)
{
	struct cass_dev *hw = (struct cass_dev *)pci_accessor;
	size_t len = sizeof(u64);
	u64 data_out;

	cass_read(hw, (u64)offset, &data_out, len);

	/* Explicit loss of precision here - Cassini doe not have 32-bit CSRs.
	 *  CSRs which are described as 32 bits can be read/written
	 *  Using the 64-bit access functions.
	 */
	return (u32)data_out;
}

/**
 * cass_write32() - wrapper around Cassini 64-bit write function
 *
 * Modify a control status register (CSR) on BAR0.
 *
 * @pci_accessor: the SBL device
 * @offset: offset of the CSR in the region
 * @val: data to write
 */
static void cass_write32(void *pci_accessor, long offset, u32 val)
{
	struct cass_dev *hw = (struct cass_dev *)pci_accessor;
	size_t len = sizeof(u64);
	u64 wval = 0;

	/* Explicit loss of precision here - Cassini doe not have 32-bit CSRs.
	 *  CSRs which are described as 32 bits can be read/written
	 *  Using the 64-bit access functions.
	 */
	wval |= val;
	cass_write(hw, (u64)offset, &wval, len);
}

static bool cass_sbl_is_fabric_link(void *accessor, int port_num)
{
	return false;
}

/**
 * cass_sbl_get_max_frame_size() - Get MTU
 *
 * Return the MTU value from the cxi_dev
 *
 * @accessor: the SBL device
 * @port_num: unused
 *
 * Return: int containing the MTU
 */
static int cass_sbl_get_max_frame_size(void *accessor, int port_num)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;
	u32 size = C_MAX_HPC_MTU;

	/* SBL uses MTU to set LLR buffer size. Use the largest of the fixed
	 * HPC frame size and configurable Ethernet (standard or optimized)
	 * frame size(s).
	 */
	if (hw->max_eth_rxsize > size)
		size = hw->max_eth_rxsize;

	cxidev_dbg(&hw->cdev, "sbl get_max_frame_size returning %u\n", size);

	return size;
}

/**
 * cass_sbl_pml_hdlr() - The PML block intr handler
 *
 * This just calls into the sbl block handler, which determines the irq itself
 *
 * @hw: the device
 * @irq: unused
 * @is_ext: unused
 * @bitn: unused
 */
static void cass_sbl_pml_hdlr(struct cass_dev *hw, unsigned int irq,
			      bool is_ext, unsigned int bitn)
{
	int err;

	err = sbl_pml_hdlr(hw->sbl, 0, NULL);
	if (err)
		cxidev_err(&hw->cdev, "(sbl): call to sbl_pml_hdl failed [%d]\n",
			   err);
}

/**
 * cass_sbl_pml_install_intr_handler() - Installs the PML block intr handler
 *
 * Wrapper around the Cassini IRQ registration function
 *
 * @accessor: the SBL device
 * @port_num: unused
 * @err_flags: bitmask of interrupts to enable
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_pml_install_intr_handler(void *accessor, int port_num,
					     u64 err_flags)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;

	cxidev_dbg(&hw->cdev, "(sbl): installing intr handlers\n");

	hw->pml_err.irq = C_HNI_PML_IRQA_MSIX_INT;
	hw->pml_err.is_ext = false;
	hw->pml_err.err_flags.c1_hni_pml.qw = err_flags;
	hw->pml_err.cb = cass_sbl_pml_hdlr;
	cxi_register_hw_errors(hw, &hw->pml_err);

	return 0;
}

/**
 * cass_sbl_pml_enable_intr_handler() - Enable bits in PML block intr handler
 *
 * Wrapper around the Cassini IRQ enable function
 *
 * @accessor: the SBL device
 * @port_num: unused
 * @err_flags: bitmask of interrupts to enable
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_pml_enable_intr_handler(void *accessor, int port_num,
					    u64 err_flags)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;
	unsigned int irq = C_HNI_PML_IRQA_MSIX_INT;
	bool is_ext = false;

	cxidev_dbg(&hw->cdev, "(sbl): enabling intr handler\n");
	cxi_enable_hw_errors(hw, irq, is_ext, (unsigned long *)&err_flags);

	return 0;
}

/**
 * cass_sbl_pml_remove_intr_handler() - Remove PML block intr handler
 *
 * Wrapper around the Cassini IRQ unregister function
 *
 * @accessor: the SBL device
 * @port_num: unused
 * @err_flags: unused
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_pml_remove_intr_handler(void *accessor, int port_num,
					    u64 err_flags)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;

	cxidev_dbg(&hw->cdev, "(sbl): removing intr handlers\n");
	cxi_unregister_hw_errors(hw, &hw->pml_err);

	return 0;
}

/**
 * cass_sbl_pml_disable_intr_handler() - Disable bits in PML block intr handler
 *
 * Wrapper around the Cassini IRQ disable function
 *
 * @accessor: the SBL device
 * @port_num: unused
 * @err_flags: bitmask of interrupts to disable
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_pml_disable_intr_handler(void *accessor, int port_num,
					     u64 err_flags)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;
	unsigned int irq = C_HNI_PML_IRQA_MSIX_INT;
	bool is_ext = false;

	cxidev_dbg(&hw->cdev, "(sbl): disabling intr handler\n");
	cxi_disable_hw_errors(hw, irq, is_ext, (unsigned long *)&err_flags);

	return 0;
}

/**
 * cass_sbl_link_down_async_handler() - link down handler
 *
 * Handler function called by the SBL async alert if the alert type is
 * a link down event
 *
 * @hw: Cassini device
 * @port_num: unused
 * @down_origin: one of enum sbl_link_down_origin
 */
static void cass_sbl_link_down_async_handler(struct cass_dev *hw, int port_num,
					     int down_origin)
{
	int down_reason;

	cxidev_dbg(&hw->cdev, "(sbl): async link down\n");

	switch (down_origin) {
	case SBL_LINK_DOWN_ORIGIN_LOCAL_FAULT:
		down_reason = CASS_DOWN_ORIGIN_BL_LFAULT;
		break;
	case SBL_LINK_DOWN_ORIGIN_REMOTE_FAULT:
		down_reason = CASS_DOWN_ORIGIN_BL_RFAULT;
		break;
	case SBL_LINK_DOWN_ORIGIN_ALIGN:
		down_reason = CASS_DOWN_ORIGIN_BL_ALIGN;
		break;
	case SBL_LINK_DOWN_ORIGIN_LINK_DOWN:
		down_reason = CASS_DOWN_ORIGIN_BL_DOWN;
		break;
	case SBL_LINK_DOWN_ORIGIN_HISER:
		down_reason = CASS_DOWN_ORIGIN_BL_HISER;
		break;
	case SBL_LINK_DOWN_ORIGIN_LLR_MAX:
		down_reason = CASS_DOWN_ORIGIN_BL_LLR;
		break;
	case SBL_LINK_DOWN_ORIGIN_UCW:
		down_reason = CASS_DOWN_ORIGIN_LMON_UCW;
		break;
	case SBL_LINK_DOWN_ORIGIN_CCW:
		down_reason = CASS_DOWN_ORIGIN_LMON_CCW;
		break;
	case SBL_LINK_DOWN_ORIGIN_LLR_TX_REPLAY:
		down_reason = CASS_DOWN_ORIGIN_LLR_TX_REPLAY;
		break;
	default:
		down_reason = CASS_DOWN_ORIGIN_BL_UNKNOWN;
		break;
	}

	cass_link_async_down(hw, down_reason);
}

/**
 * cass_sbl_async_alert() - async alert
 *
 * Notification function asynchronously called by the base link when it sends
 * an alert
 *
 * @accessor: the SBL device
 * @port_num: port number
 * @alert_type: one of enum sbl_async_alert_type
 * @alert_data: opaque pointer to alert data
 * @size: size of the data
 */
static void cass_sbl_async_alert(void *accessor, int port_num, int alert_type,
				 void *alert_data, int size)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;
	int down_origin;

	cxidev_dbg(&hw->cdev, "(sbl): %s alert", sbl_async_alert_str(alert_type));

	switch (alert_type) {
	case SBL_ASYNC_ALERT_LINK_DOWN:
		cxidev_warn(&hw->cdev, "(sbl): %d: link down alert\n",
			    port_num);
		down_origin = (int)(uintptr_t)alert_data;
		cass_sbl_link_down_async_handler(hw, port_num, down_origin);
		break;
	case SBL_ASYNC_ALERT_SERDES_FW_CORRUPTION:
		cxidev_crit(&hw->cdev, "(sbl): %d: serdes fw corruption alert\n",
			    port_num);
		break;
	case SBL_ASYNC_ALERT_TX_DEGRADE:
		cxidev_warn(&hw->cdev, "(sbl): %d: tx degrade alert\n",
				port_num);
		break;
	case SBL_ASYNC_ALERT_RX_DEGRADE:
		cxidev_warn(&hw->cdev, "(sbl): %d: rx degrade alert\n",
				port_num);
		break;
	case SBL_ASYNC_ALERT_TX_DEGRADE_FAILURE:
		cxidev_warn(&hw->cdev, "(sbl): %d: tx degrade failure alert\n",
				port_num);
		break;
	case SBL_ASYNC_ALERT_RX_DEGRADE_FAILURE:
		cxidev_warn(&hw->cdev, "(sbl): %d: rx degrade failure alert\n",
				port_num);
		break;
	default:
		cxidev_warn(&hw->cdev, "(sbl): %d: unknown alert reported: %d\n",
			    port_num, alert_type);
		break;
	}
}

/**
 * cass_sbl_sbus_op() - Perform an sbus operation
 *
 * Performs target sbus operation on this NIC's Cassini Sbus ring
 *
 * @accessor: the SBL device
 * @ring: unused
 * @req_data: Data for request
 * @data_addr: Data address within the SBus Receiver
 * @rx_addr: Address of destination SBus Receiver
 * @command: Command for request
 * @rsp_data: pointer to store response data
 * @result_code: pointer to store result code from SBus request
 * @overrun: pointer to store request overrun condition
 * @timeout: timeout in ms
 * @flags: specified polling delay and interval from cass_timing_flag
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_sbus_op(void *accessor, int ring, u32 req_data,
			    u8 data_addr, u8 rx_addr, u8 command,
			    u32 *rsp_data, u8 *result_code, u8 *overrun,
			    int timeout, unsigned int flags)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;
	struct cxi_sbus_op_params params = {
		.req_data = req_data,
		.data_addr = data_addr,
		.rx_addr = rx_addr,
		.command = command,
		.timeout = timeout,
	};

	if (flags & SBL_FLAG_DELAY_3US)
		params.delay = 3;
	else if (flags & SBL_FLAG_DELAY_4US)
		params.delay = 4;
	else if (flags & SBL_FLAG_DELAY_5US)
		params.delay = 5;
	else if (flags & SBL_FLAG_DELAY_10US)
		params.delay = 10;
	else if (flags & SBL_FLAG_DELAY_20US)
		params.delay = 20;
	else if (flags & SBL_FLAG_DELAY_50US)
		params.delay = 50;
	else if (flags & SBL_FLAG_DELAY_100US)
		params.delay = 100;
	else
		return -EINVAL;

	if (flags & SBL_FLAG_INTERVAL_1MS)
		params.poll_interval = 1;
	else if (flags & SBL_FLAG_INTERVAL_10MS)
		params.poll_interval = 10;
	else if (flags & SBL_FLAG_INTERVAL_100MS)
		params.poll_interval = 100;
	else if (flags & SBL_FLAG_INTERVAL_1S)
		params.poll_interval = 1000;
	else
		return -EINVAL;

	return cxi_sbus_op(&hw->cdev, &params, rsp_data, result_code, overrun);
}

/**
 * cass_sbl_sbus_op_reset() - Perform an sbus op reset
 *
 * Note this doesn't reset the sbus - all it does is clear the MB accessor
 * registers
 *
 * @accessor: the SBL device
 * @ring: unused
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_sbus_op_reset(void *accessor, int ring)
{
	struct cass_dev *hw = (struct cass_dev *)accessor;

	return cxi_sbus_op_reset(&hw->cdev);
}
/* End sbl_ops */

/* phy_ops follow */
/**
 * cass_sbl_init() - Create a base link device
 *
 * Implements phy_ops' .init function
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_sbl_init(struct cass_dev *hw)
{
	struct sbl_inst *sbl;
	struct sbl_ops ops;
	struct sbl_instance_attr sbl_attr = SBL_INSTANCE_ATTR_INITIALIZER;
	struct sbl_init_attr sbl_iattr;
	int err = 0;

	ops.sbl_read32			 = cass_read32;
	ops.sbl_write32			 = cass_write32;
	ops.sbl_read64			 = cass_read64;
	ops.sbl_write64			 = cass_write64;
	ops.sbl_sbus_op			 = cass_sbl_sbus_op;
	ops.sbl_sbus_op_reset		 = cass_sbl_sbus_op_reset;
	ops.sbl_is_fabric_link		 = cass_sbl_is_fabric_link;
	ops.sbl_get_max_frame_size	 = cass_sbl_get_max_frame_size;
	ops.sbl_pml_install_intr_handler = cass_sbl_pml_install_intr_handler;
	ops.sbl_pml_enable_intr_handler	 = cass_sbl_pml_enable_intr_handler;
	ops.sbl_pml_remove_intr_handler	 = cass_sbl_pml_remove_intr_handler;
	ops.sbl_pml_disable_intr_handler = cass_sbl_pml_disable_intr_handler;
	ops.sbl_async_alert              = cass_sbl_async_alert;

	sbl_iattr.magic = SBL_INIT_ATTR_MAGIC;
	sbl_iattr.uc_nic = hw->uc_nic;

	switch (hw->uc_platform) {
	case CUC_BOARD_TYPE_SAWTOOTH:
		sbl_iattr.uc_platform = SBL_UC_PLATFORM_SAWTOOTH;
		break;
	case CUC_BOARD_TYPE_BRAZOS:
		sbl_iattr.uc_platform = SBL_UC_PLATFORM_BRAZOS;
		break;
	default:
		sbl_iattr.uc_platform = SBL_UC_PLATFORM_UNKNOWN;
	}

	sbl = sbl_new_instance(hw, hw, &ops, &sbl_iattr);

	if (IS_ERR(sbl)) {
		cxidev_err(&hw->cdev, "new sbl instance failed [%ld]\n",
			   PTR_ERR(sbl));
		err = PTR_ERR(sbl);
		goto out_bad_sbl;
	}
	hw->sbl = sbl;

	/* initialise the base link instance */
	strscpy(sbl_attr.inst_name, hw->cdev.name,
		sizeof(sbl_attr.inst_name));
	err = sbl_initialise_instance(hw->sbl, &sbl_attr);
	if (err) {
		cxidev_err(&hw->cdev, "sbl init failed [%d]\n", err);
		sbl_delete_instance(hw->sbl);
		goto out_bad_sbl;
	}

	hw->sbl_counters = NULL;
	cass_sbl_counters_init(hw);

 out_bad_sbl:
	if (err)
		hw->sbl = NULL;

	return err;
}

/**
 * cass_sbl_exit() - Delete a base link device
 *
 * Implements phy_ops' .exit function
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
void cass_sbl_exit(struct cass_dev *hw)
{
	int err;

	cass_sbl_counters_term(hw);
	err = sbl_delete_instance(hw->sbl);
	if (err)
		cxidev_err(&hw->cdev, "delete sbl instance failed [%d]\n", err);
	hw->sbl = NULL;
}

/**
 * cass_sbl_power_on() - Bring a link up
 *
 * Implements phy_ops' .power_on function
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_sbl_power_on(struct cass_dev *hw)
{
	bool auto_restart;
	int blstate;
	int err = 0;
	bool auto_neg_timedout = false;

	cxidev_dbg(&hw->cdev, "link: up starting\n");
	atomic_inc(&hw->sbl_counters[link_up_tries]);

	hw->port->config_state |= CASS_PORT_CONFIGURED;

	if (cass_link_get_state(hw) == CASS_LINK_STATUS_UP)
		goto out;
	else
		cass_link_set_state(hw, CASS_LINK_STATUS_STARTING, 0);

	cass_link_set_down_origin(hw, CASS_DOWN_ORIGIN_NONE);
	auto_restart = hw->port->lattr.options & CASS_LINK_OPT_UP_AUTO_RESTART;

	if (hw->port->subtype != CASS_PORT_SUBTYPE_IEEE)
		sbl_enable_opt_lane_degrade(hw->sbl, 0, true);
	/*
	 * bring up the base link (serdes,pcs,mac,[llr])
	 */
	err = sbl_base_link_start(hw->sbl, 0);
	if (err) {
		if (err == -ETIME)
			auto_neg_timedout = true;

		if (auto_restart || (err == -ECANCELED)) {
			/*
			 * link did not come up
			 * try to cleanup base link for another restart attempt
			 */
			cxidev_dbg(&hw->cdev, "link: base link start failed [%d]\n",
				   err);
			sbl_base_link_try_start_fail_cleanup(hw->sbl, 0);
			sbl_base_link_get_status(hw->sbl, 0, &blstate,
						 NULL, NULL, NULL, NULL, NULL);
			cxidev_dbg(&hw->cdev, "link: base link status = %d\n",
					blstate);
			if (blstate == SBL_BASE_LINK_STATUS_DOWN)
				goto out_non_fatal;
			else if (blstate == SBL_BASE_LINK_STATUS_UNCONFIGURED)
				goto out_non_fatal;
			else
				goto out_fatal;
		} else {
			cxidev_err(&hw->cdev, "link: base link start failed [%d]\n",
				   err);
			err = -ENOLINK;
			goto out_fatal;
		}
	}

	atomic_inc(&hw->sbl_counters[link_up]);

	// TODO_LLR

	// TODO_START

	// TODO_CB LINK_STATE_UP

	/* report success */
out:
	cass_link_set_state(hw, CASS_LINK_STATUS_UP, 0);
	cass_phy_link_up(hw);

	hw->port->start_time = ktime_get_seconds();

	return 0;

out_non_fatal:
	if (!auto_neg_timedout) {
		cass_link_set_state(hw, CASS_LINK_STATUS_DOWN, 0);
		cass_phy_link_down(hw);
	}

	return err;

	/*
	 * fatal errors result in the error state and require a reset to clear
	 */
out_fatal:
	cass_link_set_state(hw, CASS_LINK_STATUS_ERROR, err);

	return err;
}

/**
 * cass_sbl_power_off() - Bring a link down
 *
 * Implements phy_ops' .power_off function
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_sbl_power_off(struct cass_dev *hw)
{
	int err = 0;

	cxidev_dbg(&hw->cdev, "link: down starting (from %s)\n",
		   cass_link_state_str(cass_link_get_state(hw)));

	atomic_inc(&hw->sbl_counters[link_down_tries]);

	if (cass_link_get_state(hw) == CASS_LINK_STATUS_DOWN)
		goto out;
	else
		cass_link_set_state(hw, CASS_LINK_STATUS_STOPPING, 0);

	/*
	 * ethernet links need to make sure all flows are empty
	 */
	sbl_pml_mac_stop(hw->sbl, 0);

	/*
	 * stop base link
	 */
	if (hw->port->link_down_origin == CASS_DOWN_ORIGIN_BL_HISER ||
	    hw->port->link_down_origin == CASS_DOWN_ORIGIN_LMON_UCW ||
	    hw->port->link_down_origin == CASS_DOWN_ORIGIN_BL_DOWN)
		sbl_serdes_invalidate_tuning_params(hw->sbl, 0);

	err = sbl_base_link_stop(hw->sbl, 0);
	if (err) {
		cxidev_err(&hw->cdev,
			   "link: fabric stop, sbl_link_down failed [%d]\n",
			   err);
		goto out_fatal;
	}

	atomic_inc(&hw->sbl_counters[link_down]);

	// TODO_CB LINK_STATE_DOWN

	/* report success */
out:
	cass_link_set_state(hw, CASS_LINK_STATUS_DOWN, 0);
	hw->port->config_state &= ~CASS_PORT_CONFIGURED;
	cass_phy_link_down(hw);

	return 0;

out_fatal:
	cass_link_set_state(hw, CASS_LINK_STATUS_ERROR, err);

	return err;
}

/**
 * cass_sbl_serdes_config_defaults() - Writes a Cassini-specific config to SBL
 *
 * Updates SBL config to include an additional SerDes config value set.
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_sbl_serdes_config_defaults(struct cass_dev *hw)
{
	struct sbl_sc_values vals = { 0 };
	int rc, err;

	/* CONFIG 1: 200G and 100G PEC to switch */
	u64 port_mask = 0x1;
	u8  serdes_mask = 0xf;
	u64 tp_state_mask0 = sbl_create_tp_hash0((u8)CASS_SBL_SC_MASKED_IN,
						 (u8)CASS_SBL_SC_MASKED_IN,
						 (u8)CASS_SBL_SC_MASKED_OUT,
						 (u8)CASS_SBL_SC_MASKED_OUT,
						 (u8)CASS_SBL_SC_MASKED_IN,
						 (u16)CASS_SBL_SC_MASKED_OUT);
	u64 tp_state_mask1 = sbl_create_tp_hash1(CASS_SBL_SC_MASKED_IN);
	u64 tp_state_match0 = sbl_create_tp_hash0(CASS_SBL_SC_LP_SWITCH,
						 CASS_SBL_SC_LBM_ANY,
						 CASS_SBL_SC_NA,
						 CASS_SBL_SC_NA,
						 CASS_SBL_SC_MT_ELEC,
						 CASS_SBL_SC_NA);
	u64 tp_state_match1 = sbl_create_tp_hash1(CASS_SBL_SC_NA);

	vals.magic    = SBL_SERDES_CONFIG_MAGIC;
	vals.atten    = SBL_DFLT_PORT_CONFIG_ATTEN;
	vals.pre      = CASS_SBL_PORT_CONFIG1_PRE;
	vals.post     = SBL_DFLT_PORT_CONFIG_POST;
	vals.pre2     = SBL_DFLT_PORT_CONFIG_PRE2;
	vals.pre3     = SBL_DFLT_PORT_CONFIG_PRE3;
	vals.gs1      = SBL_DFLT_PORT_CONFIG_GS1_OPTICAL;
	vals.gs2      = SBL_DFLT_PORT_CONFIG_GS2_OPTICAL;
	vals.num_intr = SBL_DFLT_PORT_CONFIG_NUM_INTR;

	err = sbl_serdes_add_config(hw->sbl, tp_state_mask0, tp_state_mask1,
				   tp_state_match0, tp_state_match1, port_mask,
				   serdes_mask, &vals, false);
	/* If we failed trying to add a duplicate config - we still got what
	 * we wanted. Return 0.
	 */
	if ((err == 0) || (err == -EEXIST))
		rc = 0;
	else
		rc = err;

	// CONFIG 2: 100G AOC to switch
	// All the same mask/match params, save for the following:
	tp_state_match0 = sbl_create_tp_hash0((u8)CASS_SBL_SC_LP_SWITCH,
					      (u8)CASS_SBL_SC_LBM_ANY,
					      (u8)CASS_SBL_SC_NA,
					      (u8)CASS_SBL_SC_LM_100,
					      (u8)CASS_SBL_SC_MT_OPT_ANY,
					      (u16)CASS_SBL_SC_NA);
	vals.atten    = CASS_SBL_PORT_CONFIG2_ATTEN;
	vals.pre      = SBL_DFLT_PORT_CONFIG_PRE;
	vals.post     = SBL_DFLT_PORT_CONFIG_POST;
	vals.pre2     = SBL_DFLT_PORT_CONFIG_PRE2;
	vals.pre3     = SBL_DFLT_PORT_CONFIG_PRE3;
	vals.gs1      = CASS_SBL_PORT_CONFIG2_GS1;
	vals.gs2      = SBL_DFLT_PORT_CONFIG_GS2_OPTICAL;
	vals.num_intr = CASS_SBL_PORT_CONFIG2_NUM_INTR;
	vals.intr_val[0] = CASS_SBL_PORT_CONFIG2_INTR0;
	vals.data_val[0] = CASS_SBL_PORT_CONFIG2_DATA0;
	vals.intr_val[1] = CASS_SBL_PORT_CONFIG2_INTR1;
	vals.data_val[1] = CASS_SBL_PORT_CONFIG2_DATA1;
	vals.intr_val[2] = CASS_SBL_PORT_CONFIG2_INTR2;
	vals.data_val[2] = CASS_SBL_PORT_CONFIG2_DATA2;
	vals.intr_val[3] = CASS_SBL_PORT_CONFIG2_INTR3;
	vals.data_val[3] = CASS_SBL_PORT_CONFIG2_DATA3;

	err = sbl_serdes_add_config(hw->sbl, tp_state_mask0, tp_state_mask1,
				   tp_state_match0, tp_state_match1, port_mask,
				   serdes_mask, &vals, false);
	if ((err == 0) || (err == -EEXIST))
		rc += 0;
	else
		rc += err;

	// CONFIG 3: 200G AOC to switch
	// All the same mask/match params, save for the following:
	tp_state_match0 = sbl_create_tp_hash0((u8)CASS_SBL_SC_LP_SWITCH,
					      (u8)CASS_SBL_SC_LBM_ANY,
					      (u8)CASS_SBL_SC_NA,
					      (u8)CASS_SBL_SC_LM_200,
					      (u8)CASS_SBL_SC_MT_OPT_ANY,
					      (u16)CASS_SBL_SC_NA);
	vals.atten    = CASS_SBL_PORT_CONFIG3_ATTEN;
	vals.pre      = CASS_SBL_PORT_CONFIG3_PRE;
	vals.post     = SBL_DFLT_PORT_CONFIG_POST;
	vals.pre2     = SBL_DFLT_PORT_CONFIG_PRE2;
	vals.pre3     = SBL_DFLT_PORT_CONFIG_PRE3;
	vals.gs1      = CASS_SBL_PORT_CONFIG3_GS1;
	vals.gs2      = SBL_DFLT_PORT_CONFIG_GS2_OPTICAL;
	vals.num_intr = SBL_DFLT_PORT_CONFIG_NUM_INTR;

	err = sbl_serdes_add_config(hw->sbl, tp_state_mask0, tp_state_mask1,
				   tp_state_match0, tp_state_match1, port_mask,
				   serdes_mask, &vals, false);
	if ((err == 0) || (err == -EEXIST))
		rc += 0;
	else
		rc += err;

	if (rc)
		cxidev_err(&hw->cdev, "Failed to update SBL config [%d]\n", rc);
	return rc;
}

/**
 * cass_sbl_set_defaults() - Set link defaults
 *
 * Sets reasonable defaults for 200G Electrical link
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
void cass_sbl_set_defaults(struct cass_dev *hw)
{
	struct sbl_media_attr mattr;

	/* The media attribute can be configured during cable scan, which has
	 * already happened. Don't blow away this information.
	 */
	mattr = hw->port->lattr.mattr;
	memset(&hw->port->lattr, 0, sizeof(struct cass_link_attr));
	hw->port->lattr.mattr = mattr;

	/* base link config */
	hw->port->lattr.bl.magic = SBL_LINK_ATTR_MAGIC;
	hw->port->lattr.bl.options = SBL_OPT_ENABLE_PCAL | SBL_DISABLE_PML_RECOVERY;
	hw->port->lattr.options = CASS_LINK_ATTR_DEFAULT_OPT_FLAGS
		| CASS_LINK_OPT_UP_AUTO_RESTART;

	switch (hw->port->lattr.mattr.media) {
	case SBL_LINK_MEDIA_ELECTRICAL:
	default:
		hw->port->lattr.bl.options |= SBL_OPT_AUTONEG_TIMEOUT_SSHOT |
			SBL_OPT_AUTONEG_100CR4_FIXUP;
		hw->port->lattr.bl.start_timeout =
			SBL_LINK_START_TIMEOUT_PEC;
		hw->port->lattr.bl.config_target =
			SBL_BASE_LINK_CONFIG_PEC;
		hw->port->lattr.bl.pec.an_mode =
			SBL_AN_MODE_ON;
		hw->port->lattr.bl.pec.an_retry_timeout =
			SBL_LINK_DFLT_AN_RETRY_TIMEOUT;
		hw->port->lattr.bl.pec.an_max_retry =
			SBL_LINK_DFLT_AN_MAX_RETRY;
		hw->port->lattr.bl.lpd_timeout = 0;
		hw->port->lattr.bl.lpd_poll_interval = 0;
		hw->port->lattr.bl.dfe_pre_delay =
			SBL_DFLT_DFE_PRE_DELAY_PEC;
		hw->port->lattr.bl.dfe_timeout =
			SBL_DFLT_DFE_TIMEOUT_PEC;
		hw->port->lattr.bl.dfe_poll_interval =
			SBL_DFLT_DFE_POLL_INTERVAL;
		hw->port->lattr.bl.pcal_eyecheck_holdoff =
			SBL_DFLT_PCAL_EYECHECK_HOLDOFF_PEC;
		hw->port->lattr.bl.nrz_min_eye_height =
			SBL_DFLT_NRZ_PEC_MIN_EYE_HEIGHT;
		hw->port->lattr.bl.nrz_max_eye_height =
			SBL_DFLT_NRZ_PEC_MAX_EYE_HEIGHT;
		hw->port->lattr.bl.pam4_min_eye_height =
			SBL_DFLT_PAM4_PEC_MIN_EYE_HEIGHT;
		hw->port->lattr.bl.pam4_max_eye_height =
			SBL_DFLT_PAM4_PEC_MAX_EYE_HEIGHT;
		break;
	case SBL_LINK_MEDIA_OPTICAL:
		hw->port->lattr.bl.start_timeout =
			SBL_LINK_START_TIMEOUT_AOC;
		hw->port->lattr.bl.config_target =
			SBL_BASE_LINK_CONFIG_AOC;
		hw->port->lattr.bl.aoc.reserved = 0;
		hw->port->lattr.bl.aoc.optical_lock_delay =
			SBL_DFLT_OPTICAL_LOCK_DELAY;
		hw->port->lattr.bl.aoc.optical_lock_interval =
			SBL_DFLT_OPTICAL_LOCK_INTERVAL;
		/* hw->port->lattr.bl.options |= */
		/*	SBL_OPT_SERDES_LPD; */
		hw->port->lattr.bl.lpd_timeout =
			SBL_DFLT_LPD_TIMEOUT;
		hw->port->lattr.bl.lpd_poll_interval =
			SBL_DFLT_LPD_POLL_INTERVAL;
		hw->port->lattr.bl.dfe_pre_delay =
			SBL_DFLT_DFE_PRE_DELAY_AOC;
		hw->port->lattr.bl.dfe_timeout =
			SBL_DFLT_DFE_TIMEOUT_AOC;
		hw->port->lattr.bl.dfe_poll_interval =
			SBL_DFLT_DFE_POLL_INTERVAL;
		hw->port->lattr.bl.pcal_eyecheck_holdoff =
			SBL_DFLT_PCAL_EYECHECK_HOLDOFF_AOC;
		hw->port->lattr.bl.nrz_min_eye_height =
			SBL_DFLT_NRZ_AOC_MIN_EYE_HEIGHT;
		hw->port->lattr.bl.nrz_max_eye_height =
			SBL_DFLT_NRZ_AOC_MAX_EYE_HEIGHT;
		hw->port->lattr.bl.pam4_min_eye_height =
			SBL_DFLT_PAM4_AOC_MIN_EYE_HEIGHT;
		hw->port->lattr.bl.pam4_max_eye_height =
			SBL_DFLT_PAM4_AOC_MAX_EYE_HEIGHT;
		break;
	}

	if ((HW_PLATFORM_Z1(hw)) &&
	    (hw->port->lattr.mattr.media == SBL_LINK_MEDIA_ELECTRICAL)) {
		hw->port->lattr.bl.pec.an_mode = SBL_AN_MODE_FIXED;
	}

	hw->port->lattr.bl.link_mode          = SBL_LINK_MODE_BS_200G;
	hw->port->lattr.bl.loopback_mode      = SBL_LOOPBACK_MODE_OFF;
	hw->port->lattr.bl.link_partner       = SBL_LINK_PARTNER_SWITCH;
	hw->port->lattr.bl.tuning_pattern     = SBL_TUNING_PATTERN_CORE;
	hw->port->lattr.bl.fec_mode           = SBL_RS_MODE_ON;
	hw->port->lattr.bl.enable_autodegrade = 0;
	if (HW_PLATFORM_ASIC(hw))
		hw->port->lattr.bl.llr_mode   = SBL_LLR_MODE_ON;
	else
		hw->port->lattr.bl.llr_mode   = SBL_LLR_MODE_OFF;
	hw->port->lattr.bl.ifg_config         = SBL_IFG_CONFIG_IEEE_200G;
	hw->port->lattr.bl.precoding          = SBL_PRECODING_OFF;

	hw->port->lattr.bl.pml_recovery.timeout         = pml_rec_timeout;
	hw->port->lattr.bl.pml_recovery.rl_max_duration = pml_rec_rl_max_dur;
	hw->port->lattr.bl.pml_recovery.rl_window_size  = pml_rec_rl_win_size;
}

/**
 * cass_sbl_configure() - Send configuration to SBL
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_sbl_configure(struct cass_dev *hw)
{
	int err;

	cxidev_dbg(&hw->cdev, "Configuring SBL\n");

	if ((hw->port->lattr.bl.llr_mode == SBL_LLR_MODE_ON) ||
	    (hw->port->lattr.bl.ifg_config == SBL_IFG_CONFIG_HPC))
		hw->port->subtype = CASS_PORT_SUBTYPE_CASSINI;
	else
		hw->port->subtype = CASS_PORT_SUBTYPE_IEEE;

	err = hw->link_ops->link_config(hw);
	if (err) {
		cxidev_err(&hw->cdev, "configure failed [%d]\n", err);
		return err;
	}

	cass_link_set_state(hw, CASS_LINK_STATUS_DOWN, 0);
	hw->link_config_dirty = false;
	hw->port->config_state |= CASS_TYPE_CONFIGURED | CASS_LINK_CONFIGURED;

	return err;
}

/**
 * cass_sbl_reset() - Reset a link
 *
 * Implements phy_ops' .reset function
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_sbl_reset(struct cass_dev *hw)
{
	int bl_state;
	int err;

	cxidev_dbg(&hw->cdev, "link: reset (from %s)\n",
		   cass_link_state_str(cass_link_get_state(hw)));

	atomic_inc(&hw->sbl_counters[link_reset_tries]);

	cass_link_set_state(hw, CASS_LINK_STATUS_RESETTING, 0);

	/* reset the base link */
	sbl_base_link_get_status(hw->sbl, 0, &bl_state, NULL, NULL, NULL,
				 NULL, NULL);

	if (bl_state != SBL_BASE_LINK_STATUS_UNCONFIGURED) {
		err = sbl_base_link_reset(hw->sbl, 0);
		if (err)
			cxidev_err(&hw->cdev,
				   "link: reset, base link reset failed [%d]\n",
				   err);
	}

	spin_lock(&hw->sbl_state_lock);
	hw->port->prev_lstate = hw->port->lstate;
	hw->port->lstate = CASS_LINK_STATUS_UNCONFIGURED;
	hw->port->lerr = 0;
	hw->port->config_state &= ~CASS_LINK_CONFIGURED;
	spin_unlock(&hw->sbl_state_lock);

	atomic_inc(&hw->sbl_counters[link_reset]);

	return 0;
}
/* end phy_ops */

/* sysfs */
static ssize_t link_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = cass_link_sysfs_sprint(hw, buf, PAGE_SIZE);

	return rc;
}

static ssize_t link_layer_retry_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = sbl_base_link_llr_sysfs_sprint(hw->sbl, 0, buf, PAGE_SIZE);

	return rc;
}

static ssize_t loopback_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = sbl_base_link_loopback_sysfs_sprint(hw->sbl, 0, buf, PAGE_SIZE);

	return rc;
}

static ssize_t media_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = sbl_media_type_sysfs_sprint(hw->sbl, 0, buf, PAGE_SIZE);

	return rc;
}

static ssize_t pause_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = cass_pause_sysfs_sprint(hw, buf, PAGE_SIZE);

	return rc;
}

static ssize_t speed_show(struct kobject *kobj,
				struct kobj_attribute *kattr, char *buf)
{
	int rc;
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, port_kobj);

	if (!cass_version(hw, CASSINI_1))
		return snprintf(buf, SYSFS_BUFSIZE, "NOT IMPLEMENTED!!\n");

	rc = sbl_pml_pcs_speed_sysfs_sprint(hw->sbl, 0, buf, PAGE_SIZE);

	return rc;
}

static struct kobj_attribute dev_attr_link = __ATTR_RO(link);
static struct kobj_attribute dev_attr_link_layer_retry =
	__ATTR_RO(link_layer_retry);
static struct kobj_attribute dev_attr_loopback = __ATTR_RO(loopback);
static struct kobj_attribute dev_attr_media = __ATTR_RO(media);
static struct kobj_attribute dev_attr_pause = __ATTR_RO(pause);
static struct kobj_attribute dev_attr_speed = __ATTR_RO(speed);

static struct attribute *port_attrs[] = {
	&dev_attr_link.attr,
	&dev_attr_link_layer_retry.attr,
	&dev_attr_loopback.attr,
	&dev_attr_media.attr,
	&dev_attr_pause.attr,
	&dev_attr_speed.attr,
	NULL,
};
ATTRIBUTE_GROUPS(port);

static struct kobj_type port_sysfs_entries = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = port_groups,
};

int cass_register_port(struct cass_dev *hw)
{
	int rc;

	/* Create the port sysfs entries */
	rc = kobject_init_and_add(&hw->port_kobj, &port_sysfs_entries,
				  &hw->cdev.pdev->dev.kobj,
				  "port");
	if (rc)
		kobject_put(&hw->port_kobj);

	return rc;
}

void cass_unregister_port(struct cass_dev *hw)
{
	kobject_put(&hw->port_kobj);
}

void cass_sbl_link_mode_to_speed(u32 link_mode, int *speed)
{
	switch (link_mode) {
	case SBL_LINK_MODE_BS_200G:
		*speed = SPEED_200000;
		return;
	case SBL_LINK_MODE_BJ_100G:
	case SBL_LINK_MODE_CD_100G:
		*speed = SPEED_100000;
		return;
	case SBL_LINK_MODE_CD_50G:
		*speed = SPEED_50000;
		return;
	default:
		*speed = SPEED_UNKNOWN;
		return;
	}
}

void cass_sbl_link_speed_to_mode(int speed, u32 *link_mode)
{
	switch (speed) {
	case SPEED_200000:
	case SPEED_UNKNOWN:
		*link_mode = SBL_LINK_MODE_BS_200G;
		return;
	case SPEED_100000:
		*link_mode = SBL_LINK_MODE_BJ_100G;
		return;
	case SPEED_50000:
		*link_mode = SBL_LINK_MODE_CD_50G;
		return;
	default:
		*link_mode = SBL_LINK_MODE_INVALID;
		return;
	}
}

int cass_sbl_media_config(struct cass_dev *hw, void *attr)
{
	int err;

	cxidev_dbg(&hw->cdev, "sbl media config\n");

	err = sbl_media_config(hw->sbl, 0, attr);
	if (err) {
		cxidev_err(&hw->cdev, "sbl media config failed [%d]\n", err);
		return err;
	}

	return 0;
}

int cass_sbl_media_unconfig(struct cass_dev *hw)
{
	int err;

	cxidev_dbg(&hw->cdev, "sbl media unconfig\n");

	err = sbl_media_unconfig(hw->sbl, 0);
	if (err) {
		cxidev_err(&hw->cdev, "sbl media unconfig failed [%d]\n", err);
		return err;
	}

	return 0;
}

int cass_sbl_link_config(struct cass_dev *hw)
{
	int err;

	cxidev_dbg(&hw->cdev, "sbl link config\n");

	err = sbl_base_link_config(hw->sbl, 0, &hw->port->lattr.bl);
	if (err) {
		cxidev_err(&hw->cdev, "Failed to set SBL config defaults\n");
		return err;
	}

	err = cass_sbl_serdes_config_defaults(hw);
	if (err) {
		cxidev_err(&hw->cdev, "Failed to set SerDes config defaults\n");
		return err;
	}

	return 0;
}

bool cass_sbl_pml_pcs_aligned(struct cass_dev *hw)
{
	return sbl_pml_pcs_aligned(hw->sbl, 0);
}

/*
 * Create and initialise the port's counter array
 */
void cass_sbl_counters_init(struct cass_dev *hw)
{
	int i;

	hw->sbl_counters = kzalloc(sizeof(atomic_t)*CASS_SBL_NUM_COUNTERS, GFP_KERNEL);

	for (i = 0; i < CASS_SBL_NUM_COUNTERS; ++i)
		atomic_set(&hw->sbl_counters[i], 0);
}

/*
 * Destroy the port's counter array
 */
void cass_sbl_counters_term(struct cass_dev *hw)
{
	kfree(hw->sbl_counters);
	hw->sbl_counters = NULL;
}

/*
 * Update down origin counters
 */
void cass_sbl_counters_down_origin_inc(struct cass_dev *hw, int down_origin)
{
	switch (down_origin) {
	case CASS_DOWN_ORIGIN_CONFIG:
		atomic_inc(&hw->sbl_counters[link_down_origin_config]);
		break;
	case CASS_DOWN_ORIGIN_BL_LFAULT:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_lfault]);
		break;
	case CASS_DOWN_ORIGIN_BL_RFAULT:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_rfault]);
		break;
	case CASS_DOWN_ORIGIN_BL_ALIGN:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_align]);
		break;
	case CASS_DOWN_ORIGIN_BL_DOWN:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_down]);
		break;
	case CASS_DOWN_ORIGIN_BL_HISER:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_hiser]);
		break;
	case CASS_DOWN_ORIGIN_BL_LLR:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_llr]);
		break;
	case CASS_DOWN_ORIGIN_BL_UNKNOWN:
		atomic_inc(&hw->sbl_counters[link_down_origin_bl_unknown]);
		break;
	case CASS_DOWN_ORIGIN_LMON_UCW:
		atomic_inc(&hw->sbl_counters[link_down_origin_lmon_ucw]);
		break;
	case CASS_DOWN_ORIGIN_LMON_CCW:
		atomic_inc(&hw->sbl_counters[link_down_origin_lmon_ccw]);
		break;
	case CASS_DOWN_ORIGIN_LLR_TX_REPLAY:
		atomic_inc(&hw->sbl_counters[link_down_origin_lmon_llr_tx_replay]);
		break;
	case CASS_DOWN_ORIGIN_HEADSHELL_REMOVED:
		atomic_inc(&hw->sbl_counters[link_down_origin_headshell_removed]);
		break;
	case CASS_DOWN_ORIGIN_HEADSHELL_ERROR:
		atomic_inc(&hw->sbl_counters[link_down_origin_headshell_error]);
		break;
	case CASS_DOWN_ORIGIN_MEDIA_REMOVED:
		atomic_inc(&hw->sbl_counters[link_down_origin_media_removed]);
		break;
	case CASS_DOWN_ORIGIN_CMD:
		atomic_inc(&hw->sbl_counters[link_down_origin_cmd]);
		break;
	default:
		cxidev_err(&hw->cdev, "down origin (%d) with no counter", down_origin);
		break;
	}
}

/*
 * Update sbl counters
 */
void cass_sbl_counters_update(struct cass_dev *hw)
{
	int sbl_counters[SBL_NUM_COUNTERS];
	int i;
	int err;

	err = sbl_link_counters_get(hw->sbl, 0, sbl_counters, 0, SBL_NUM_COUNTERS);
	if (err)
		return;

	for (i = 0; i < SBL_NUM_COUNTERS; ++i)
		atomic_set(&hw->sbl_counters[SBL_COUNTERS_FIRST + i], sbl_counters[i]);
}

int cass_sbl_set_eth_name(struct cass_dev *hw, const char *name)
{
	sbl_set_eth_name(hw->sbl, name);

	return 0;
}
