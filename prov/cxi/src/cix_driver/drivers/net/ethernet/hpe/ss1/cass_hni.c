// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019-2022,2024 Hewlett Packard Enterprise Development LP */

/* Cassini device handler */

#include <linux/iopoll.h>

#include <linux/hpe/cxi/cxi.h>
#include <linux/sbl.h>
#include "cass_core.h"
#include "cass_cable.h"

static unsigned int pause_too_long_timeout = 1000;
module_param(pause_too_long_timeout, uint, 0444);
MODULE_PARM_DESC(pause_too_long_timeout,
		 "Pause too long error timeout in ms, up to 17 min (rounded up to pow 2)");
static bool disable_on_error = true;
module_param(disable_on_error, bool, 0644);
MODULE_PARM_DESC(disable_on_error, "Disable cxi device on certain errors");

/* 1ms at a 1GHz clock is 1000000 cycles, so we would need to monitor
 * bit 20 for a 1ms to 2ms timer.
 */
#define PAUSE_MSEC_BIT_OFFSET 20U
#define RX_CTRL_TIMER_BIT_OFFSET 9U
#define RX_CTRL_TIMER_BIT_MAX 31

/* Increase pause repeat period to repeat every 156 quantas. With the default
 * pause quanta configuration, repeat period is ~400 nsecs.
 */
#define PAUSE_REPEAT_PERIOD 156U

/* Try to clear a pause. */
static void pause_timeout_worker(struct work_struct *work)
{
	struct cass_dev *hw =
		container_of(work, struct cass_dev, pause_timeout_work);
	union c_hni_cfg_pause_rx_ctrl rx_ctrl_cfg;
	union c_hni_cfg_pause_rx_ctrl old_rx_ctrl_cfg;
	union c_hni_sts_pause_timeout pause_timeout;
	void __iomem *csr;
	unsigned int rc;

	/* Clear pfc_rec_enable to prevent an RX_PAUSE_ERR. */
	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &old_rx_ctrl_cfg,
		  sizeof(old_rx_ctrl_cfg));
	rx_ctrl_cfg = old_rx_ctrl_cfg;
	rx_ctrl_cfg.pfc_rec_enable = 0;
	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg, sizeof(rx_ctrl_cfg));

	/* Wait until pause clears for every PCP, ie bit 3 of each PCP
	 * is 0.
	 */
	csr = cass_csr(hw, C_HNI_STS_PAUSE_TIMEOUT);
	rc = readq_poll_timeout(csr, pause_timeout.qw,
				(pause_timeout.qw & 0x88888888ULL) == 0,
				1, 5000);
	if (rc)
		cxidev_warn(&hw->cdev,
			    "Pause too long not cleared after timeout: %llx\n",
			    pause_timeout.qw);

	/* Restore pfc_rec_enable */
	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &old_rx_ctrl_cfg,
		   sizeof(old_rx_ctrl_cfg));
}

static void pause_timeout_cb(struct cass_dev *hw, unsigned int irq,
			     bool is_ext, unsigned int bitn)
{
	queue_work(system_long_wq, &hw->pause_timeout_work);
}

static void uncor_cb(struct cass_dev *hw, unsigned int irq, bool is_ext,
		     unsigned int bitn)
{
	if (!disable_on_error)
		return;

	cass_disable_device(hw->cdev.pdev);
}

/* Re-program the HNI and OXE pause quanta when the link is going up,
 * as they are dependent on the link speed, which may have changed.
 */
static const struct quantas {
	unsigned int speed;
	union c_hni_cfg_pause_quanta q;
} c1_quantas[] = {
	{ .speed = SPEED_50000, .q = { .sub_period = 80, .sub_value = 0x7d000 }},
	{ .speed = SPEED_100000, .q = { .sub_period = 40, .sub_value = 0x7d000 }},
	{ .speed = SPEED_200000, .q = { .sub_period = 20, .sub_value = 0x7d000 }},
	{ .speed = SPEED_400000, .q = { .sub_period = 10, .sub_value = 0x7d000 }},
	{ }
}, c2_quantas[] = {
	{ .speed = SPEED_25000, .q = { .sub_period = 176, .sub_value = 0x7d000 }},
	{ .speed = SPEED_50000,  .q = { .sub_period = 88, .sub_value = 0x7d000 }},
	{ .speed = SPEED_100000, .q = { .sub_period = 44, .sub_value = 0x7d000 }},
	{ .speed = SPEED_200000, .q = { .sub_period = 22, .sub_value = 0x7d000 }},
	{ .speed = SPEED_400000, .q = { .sub_period = 11, .sub_value = 0x7d000 }},
	{ }
};

void update_hni_link_up(struct cass_dev *hw)
{
	union c_cq_cfg_fq_thresh_table fq_thresh = {};
	struct cxi_link_info link_info;
	const struct quantas *quanta;
	int i;

	cxi_link_mode_get(&hw->cdev, &link_info);

	/* Update the Pause Quanta. */
	if (cass_version(hw, CASSINI_1))
		quanta = c1_quantas;
	else
		quanta = c2_quantas;

	while (quanta->speed && quanta->speed != link_info.speed)
		quanta++;

	if (quanta->speed == 0) {
		cxidev_warn(&hw->cdev, "Missing pause quanta values for speed %u\n",
			    link_info.speed);
	} else {
		cass_write(hw, C_OXE_CFG_PAUSE_QUANTA, &quanta->q, sizeof(quanta->q));
		cass_write(hw, C_HNI_CFG_PAUSE_QUANTA, &quanta->q, sizeof(quanta->q));
	}

	/* and the FQ threshold */
	if (link_info.speed == SPEED_400000)
		fq_thresh.thresh = 8 * 1024 * 1024;
	else
		fq_thresh.thresh = 4 * 1024 * 1024;

	for (i = 0; i < C_CQ_CFG_FQ_THRESH_TABLE_ENTRIES; i++)
		cass_write(hw, C_CQ_CFG_FQ_THRESH_TABLE(i),
			   &fq_thresh, sizeof(fq_thresh));
}

void cass_hni_init(struct cass_dev *hw)
{
	int pause_timeout_bit;
	union c_hni_cfg_pause_rx_ctrl rx_ctrl;
	union c_hni_cfg_pause_timing timing;

	/* Clear all default reserved packet buffer space. */
	cass_clear(hw, C_HNI_CFG_PBUF(0), C_HNI_CFG_PBUF_SIZE);

	if (pause_too_long_timeout) {
		/* Up align pause too long timeout to a power of 2
		 * since hardware timeout operates on a specific bit.
		 */
		pause_too_long_timeout =
			roundup_pow_of_two(pause_too_long_timeout);

		/* By HW design, the pause_timeout interrupt is only
		 * generated after 8 timeouts. Account for that.
		 */
		pause_too_long_timeout = max_t(unsigned int,
					       pause_too_long_timeout / 8, 1);

		pause_timeout_bit = ffs(pause_too_long_timeout) - 1 +
			PAUSE_MSEC_BIT_OFFSET - RX_CTRL_TIMER_BIT_OFFSET;

		if (pause_timeout_bit > RX_CTRL_TIMER_BIT_MAX) {
			cxidev_info(&hw->cdev,
				    "pause_too_long_timeout is too big. Setting to max.\n");
			pause_timeout_bit = RX_CTRL_TIMER_BIT_MAX;
		}
	} else {
		pause_timeout_bit = 0;
	}

	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl, sizeof(rx_ctrl));
	rx_ctrl.timer_bit = pause_timeout_bit;
	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl, sizeof(rx_ctrl));

	cass_read(hw, C_HNI_CFG_PAUSE_TIMING, &timing, sizeof(timing));
	timing.pause_repeat_period = PAUSE_REPEAT_PERIOD;
	cass_write(hw, C_HNI_CFG_PAUSE_TIMING, &timing, sizeof(timing));

	INIT_WORK(&hw->pause_timeout_work, pause_timeout_worker);

	/* Register pause timeout callback */
	hw->hni_pause_err.irq = C_HNI_IRQA_MSIX_INT;
	hw->hni_pause_err.is_ext = false;
	if (cass_version(hw, CASSINI_1))
		hw->hni_pause_err.err_flags.c1_hni.pause_timeout = 1;
	else
		hw->hni_pause_err.err_flags.c2_hni.pause_timeout = 1;

	hw->hni_pause_err.cb = pause_timeout_cb;
	cxi_register_hw_errors(hw, &hw->hni_pause_err);

	/* Register uncor callback */
	hw->hni_pml_uncor_err.irq = C_HNI_PML_IRQA_MSIX_INT;
	hw->hni_pml_uncor_err.is_ext = false;
	hw->hni_uncor_err.irq = C_HNI_IRQA_MSIX_INT;
	hw->hni_uncor_err.is_ext = false;
	if (cass_version(hw, CASSINI_1)) {
		hw->hni_pml_uncor_err.err_flags.c1_hni_pml.llr_tx_dp_mbe = 1;
		hw->hni_uncor_err.err_flags.c1_hni.tx_flit_ucor = 1;
		hw->hni_uncor_err.err_flags.c1_hni.credit_uflw = 1;
	} else {
		hw->hni_pml_uncor_err.err_flags.ss2_port_pml.llr_tx_dp_mbe = 1;
		hw->hni_uncor_err.err_flags.c2_hni.tx_flit_ucor = 1;
		hw->hni_uncor_err.err_flags.c2_hni.credit_uflw = 1;
	}

	hw->hni_pml_uncor_err.cb = uncor_cb;
	hw->hni_uncor_err.cb = uncor_cb;
	cxi_register_hw_errors(hw, &hw->hni_pml_uncor_err);
	cxi_register_hw_errors(hw, &hw->hni_uncor_err);
}

void cass_hni_fini(struct cass_dev *hw)
{
	cxi_unregister_hw_errors(hw, &hw->hni_pause_err);
	cxi_unregister_hw_errors(hw, &hw->hni_pml_uncor_err);
	cxi_unregister_hw_errors(hw, &hw->hni_uncor_err);
	cancel_work_sync(&hw->pause_timeout_work);
}

/**
 * cass_cable_scan() - Read cable info from microcontroller
 *
 * Read data of the cable headshell and pass it down to the phy
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_cable_scan(struct cass_dev *hw)
{
	struct sbl_media_attr attr = SBL_MEDIA_ATTR_INITIALIZER;

	cxidev_dbg(&hw->cdev, "cable scan\n");

	/* This cable scan may be called before hw->port
	 * is initialized, so check for that first. In this case, a cable scan
	 * will be run as part of probe, so we can just return for now.
	 */
	if (!hw->port)
		return 0;

	/* Check if a cable is present */
	switch (hw->uc_platform) {
	case CUC_BOARD_TYPE_BRAZOS:
		cxidev_dbg(&hw->cdev, "Platform: BRAZOS\n");
		if (!cass_is_cable_present(hw)) {
			cxidev_warn(&hw->cdev, "HSN cable is not plugged\n");
			cass_link_headshell_remove(hw);
			return 0;
		}

		if (cass_parse_heashell_data(hw, &attr, &hw->qsfp_format))
			return cass_link_headshell_error(hw);

		/* AOCs need to be put into high-power mode */
		if (attr.media != SBL_LINK_MEDIA_ELECTRICAL)
			if (cass_headshell_power_up(hw, hw->qsfp_format))
				return cass_link_headshell_error(hw);

		/* Squirrel away the media info in device struct -
		 * we will make decisions based off it later
		 */
		hw->port->lattr.mattr = attr;
		return cass_link_headshell_insert(hw, &attr);

	case CUC_BOARD_TYPE_SAWTOOTH:
		cxidev_dbg(&hw->cdev, "Platform: SAWTOOTH\n");
		attr.media = SBL_LINK_MEDIA_ELECTRICAL;
		attr.len = SBL_LINK_LEN_BACKPLANE;
		attr.info = SBL_MEDIA_INFO_SUPPORTS_BS_200G |
			SBL_MEDIA_INFO_SUPPORTS_BJ_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_50G;

		/* Squirrel away the media info in device struct -
		 * we will make decisions based off it later
		 */
		hw->port->lattr.mattr = attr;

		return cass_link_media_config(hw, &attr);

	case CUC_BOARD_TYPE_KENNEBEC:
	case CUC_BOARD_TYPE_SOUHEGAN:
		cxidev_dbg(&hw->cdev, "Platform: KENNEBEC | SOUHEGAN\n");
		if (!cass_is_cable_present(hw)) {
			cxidev_warn(&hw->cdev, "HSN cable is not plugged\n");
			cass_link_headshell_remove(hw);
			return 0;
		}
		if (cass_parse_heashell_data(hw, &attr, &hw->qsfp_format))
			return cass_link_headshell_error(hw);
		return cass_link_headshell_insert(hw, &attr);

	case CUC_BOARD_TYPE_WASHINGTON:
		cxidev_dbg(&hw->cdev, "Platform: WASHINGTON\n");
		attr.media = SBL_LINK_MEDIA_ELECTRICAL;
		attr.len   = SBL_LINK_LEN_BACKPLANE;
		attr.info  = SBL_MEDIA_INFO_SUPPORTS_BS_200G;
		return cass_link_media_config(hw, &attr);

	default:
		cxidev_err(&hw->cdev, "Platform %d is not supported\n",
			   hw->uc_platform);
		return -EINVAL;
	}
}

/**
 * cxi_set_max_eth_rxsize() - Configure maximum frame size on the receive side
 *
 * @cdev: CXI device
 * @max_std_size: requested size, up to 9022.
 */
int cxi_set_max_eth_rxsize(struct cxi_dev *cdev, unsigned int max_std_size)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_hni_cfg_max_sizes new_rx;

	if (max_std_size > ETHERNET_MAX_FRAME_SIZE)
		return -EINVAL;

	/* This is only called by the Ethernet driver, so locking is
	 * not needed around the CSR.
	 */
	cass_read(hw, C_HNI_CFG_MAX_SIZES, &new_rx, sizeof(new_rx));
	new_rx.max_std_size = max_std_size;
	cass_write(hw, C_HNI_CFG_MAX_SIZES, &new_rx, sizeof(new_rx));

	/* Store max Ethernet frame size for port configuration. */
	hw->max_eth_rxsize = max_std_size;

	return 0;
}
EXPORT_SYMBOL(cxi_set_max_eth_rxsize);
