// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020,2022 Hewlett Packard Enterprise Development LP */

/* Cassini SBUS driver */

#include <linux/hpe/cxi/cxi.h>
#include <linux/iopoll.h>

#include "cass_core.h"

#include <linux/sbl.h>

/**
 * cxi_sbus_op_reset() - Perform an SBus op reset
 *
 * Note this doesn't reset the SBus - all it does is clear the MB
 * accessor registers.
 *
 * @cdev: CXI device
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_sbus_op_reset(struct cxi_dev *cdev)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	union c_mb_dbg_cfg_sbus_master cfg = {};
	union c_mb_dbg_sts_sbus_master sts;
	int rc;

	rc = mutex_lock_interruptible(&hw->sbus_mutex);
	if (rc) {
		mutex_unlock(&hw->sbus_mutex);
		return -ERESTARTSYS;
	}

	/* clear config register */
	cass_write(hw, C_MB_DBG_CFG_SBUS_MASTER, &cfg, sizeof(cfg));

	/* clear any sticky status bits */
	cass_read(hw, C_MB_DBG_STS_SBUS_MASTER, &sts, sizeof(sts));
	if (sts.overrun || sts.write_error) {
		cass_write(hw, C_MB_DBG_STS_SBUS_MASTER, &sts, sizeof(sts));
		cass_flush_pci(hw);
	}

	mutex_unlock(&hw->sbus_mutex);

	return 0;
}
EXPORT_SYMBOL(cxi_sbus_op_reset);

/**
 * cxi_sbus_op() - Perform an SBus operation
 *
 * Performs target SBus operation on the single Cassini SBus ring.
 *
 * This function will also correct the overrun and write_error
 * conditions if they happen.
 *
 * @cdev: CXI device
 * @params: command parameters
 * @rsp_data: pointer to store response data
 * @result_code: pointer to store result code from SBus request
 * @overrun: pointer to store request overrun condition
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_sbus_op(struct cxi_dev *cdev, const struct cxi_sbus_op_params *params,
		u32 *rsp_data, u8 *result_code,	u8 *overrun)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	const union c_mb_dbg_cfg_sbus_master cfg = {
		.execute = 1,
		.mode = 1,
		.command = params->command,
		.receiver_address = params->rx_addr,
		.data_address = params->data_addr,
		.data = params->req_data,
	};
	union c_mb_dbg_sts_sbus_master sts;
	void __iomem *sts_csr = cass_csr(hw, C_MB_DBG_STS_SBUS_MASTER);
	int rc;

	if (!rsp_data || !result_code || !overrun)
		return -EINVAL;

	if (params->timeout == 0 || params->timeout > 5000)
		return -EINVAL;

	if (params->poll_interval == 0 || params->poll_interval > 1000)
		return -EINVAL;

	if (params->delay > 100)
		return -EINVAL;

	cxidev_dbg(cdev, "sbus op: %d, %d, %d %d\n", params->req_data,
		   params->data_addr, params->rx_addr, params->command);

	rc = mutex_lock_interruptible(&hw->sbus_mutex);
	if (rc)
		return -ERESTARTSYS;

	/* Start the operation */
	cass_write(hw, C_MB_DBG_CFG_SBUS_MASTER, &cfg, sizeof(cfg));
	cass_flush_pci(hw);

	if (params->delay)
		udelay(params->delay);

	/* Poll for completion */
	rc = readq_poll_timeout(sts_csr, sts.qw,
				sts.rcv_data_valid == 1,
				(params->poll_interval * USEC_PER_MSEC),
				(params->timeout * USEC_PER_MSEC));
	if (rc) {
		mutex_unlock(&hw->sbus_mutex);
		return rc;
	}

	if (sts.overrun || sts.write_error) {
		cass_write(hw, C_MB_DBG_STS_SBUS_MASTER, &sts, sizeof(sts));
		cass_flush_pci(hw);
	}

	mutex_unlock(&hw->sbus_mutex);

	*rsp_data = sts.data;
	*result_code = sts.result_code;
	*overrun = sts.overrun;

	return 0;
}
EXPORT_SYMBOL(cxi_sbus_op);

/**
 * cxi_serdes_op() - Perform a SERDES operation
 *
 * Passthrough to the SBL driver.
 *
 * @cdev: CXI device
 * @serdes_sel:
 * @op:
 * @data:
 * @timeout:
 * @flags:
 * @result:
 *
 * Return: 0 on success, negative errno on failure
 */
int cxi_serdes_op(struct cxi_dev *cdev, u64 serdes_sel, u64 op, u64 data,
		  int timeout, unsigned int flags, u16 *result)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	if (!cass_version(hw, CASSINI_1)) {
		cxidev_err(cdev, "%s NOT IMPLEMENTED!\n", __func__);
		return -ENOTSUPP;
	}

	return sbl_pml_serdes_op(hw->sbl, 0, serdes_sel, op, data, result,
				 timeout, flags);
}
EXPORT_SYMBOL(cxi_serdes_op);
