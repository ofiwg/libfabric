// SPDX-License-Identifier: GPL-2.0
/* Copyright 2025 Hewlett Packard Enterprise Development LP */

#include "cass_core.h"
#include "cass_sl_io.h"

int cass_sl_uc_read8(void *uc_accessor, u32 offset, u32 page, u8 *data)
{
	struct cass_dev *cass_dev = uc_accessor;
	int              rtn;

	rtn = cxi_get_qsfp_data(&cass_dev->cdev, offset, sizeof(*data), page, data);

	cxidev_dbg(&cass_dev->cdev, "sl_uc_read8 0x%x = 0x%x\n", offset, *data);

	return rtn;
}

int cass_sl_uc_write8(void *uc_accessor, u8 page, u8 addr, u8 data)
{
	struct cass_dev *cass_dev = uc_accessor;
	int              rtn;

	rtn = uc_cmd_qsfp_write(cass_dev, page, addr, &data, sizeof(data));

	cxidev_dbg(&cass_dev->cdev, "sl_uc_write8 0x%x <- 0x%x\n", addr, data);

	return rtn;
}

u64 cass_sl_read64(void *pci_accessor, long addr)
{
	struct cass_dev *cass_dev = pci_accessor;
	u64              data64;

	cass_read(cass_dev, addr, &data64, sizeof(u64));

	cxidev_dbg(&cass_dev->cdev, "sl read64 0x%lX = 0x%llX\n", addr, data64);

	return data64;
}

void cass_sl_write64(void *pci_accessor, long addr, u64 data64)
{
	struct cass_dev *cass_dev = pci_accessor;

	cass_write(cass_dev, addr, &data64, sizeof(u64));

	cxidev_dbg(&cass_dev->cdev, "sl write64 0x%lX <- 0x%llX\n", addr, data64);
}

#define CASS_SL_SBUS_RD_CMD           0x22
#define CASS_SL_SBUS_RD_DELAY_US      50
#define CASS_SL_SBUS_RD_TIMOUT_MS     500
#define CASS_SL_SBUS_RD_POLL_INTVL_MS 10
int cass_sl_sbus_rd(void *sbus_accessor, u32 sbus_addr, u8 reg_addr, u32 *rd_data)
{
	int                        rtn;
	struct cass_dev           *cass_dev = sbus_accessor;
	struct cxi_sbus_op_params  params;
	u8                         result_code;
	u8                         overrun;

	params.req_data      = 0;
	params.data_addr     = reg_addr;
	params.rx_addr       = (sbus_addr & 0xFF);
	params.command       = CASS_SL_SBUS_RD_CMD;
	params.delay         = CASS_SL_SBUS_RD_DELAY_US;
	params.timeout       = CASS_SL_SBUS_RD_TIMOUT_MS;
	params.poll_interval = CASS_SL_SBUS_RD_POLL_INTVL_MS;

	mutex_lock(&cass_dev->sl.sbus_lock);

	rtn = cxi_sbus_op(&cass_dev->cdev, &params, rd_data, &result_code, &overrun);
	if (rtn != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus rd - cxi_sbus_op failed [%d]\n", rtn);
		goto out;
	}
	if (result_code != 4) {
		cxidev_err(&cass_dev->cdev, "sl sbus rd - cxi_sbus_op (rc = %d)\n", result_code);
		rtn = -EBADRQC;
		goto out;
	}
	if (overrun != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus rd - cxi_sbus_op (overrun = %d)\n", overrun);
		rtn = -EIO;
		goto out;
	}

	cxidev_dbg(&cass_dev->cdev, "sl sbus rd (sbus_addr = 0x%08X, reg = 0x%02X, data = 0x%08X)\n",
		sbus_addr, reg_addr, *rd_data);

	rtn = 0;

out:
	mutex_unlock(&cass_dev->sl.sbus_lock);

	return rtn;
}

#define CASS_SL_SBUS_WR_CMD           0x21
#define CASS_SL_SBUS_WR_DELAY_US      50
#define CASS_SL_SBUS_WR_TIMOUT_MS     500
#define CASS_SL_SBUS_WR_POLL_INTVL_MS 10
int cass_sl_sbus_wr(void *sbus_accessor, u32 sbus_addr, u8 reg_addr, u32 req_data)
{
	int                        rtn;
	struct cass_dev           *cass_dev = sbus_accessor;
	struct cxi_sbus_op_params  params;
	u32                        rsp_data;
	u8                         result_code;
	u8                         overrun;

	params.req_data      = req_data;
	params.data_addr     = reg_addr;
	params.rx_addr       = (sbus_addr & 0xFF);
	params.command       = CASS_SL_SBUS_WR_CMD;
	params.delay         = CASS_SL_SBUS_WR_DELAY_US;
	params.timeout       = CASS_SL_SBUS_WR_TIMOUT_MS;
	params.poll_interval = CASS_SL_SBUS_WR_POLL_INTVL_MS;

	mutex_lock(&cass_dev->sl.sbus_lock);

	rtn = cxi_sbus_op(&cass_dev->cdev, &params, &rsp_data, &result_code, &overrun);
	if (rtn != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus wr - cxi_sbus_op failed [%d]\n", rtn);
		goto out;
	}
	if (result_code != 1) {
		cxidev_err(&cass_dev->cdev, "sl sbus wr - cxi_sbus_op (rc = %d)\n", result_code);
		rtn = -EBADRQC;
		goto out;
	}
	if (overrun != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus wr - cxi_sbus_op (overrun = %d)\n", overrun);
		rtn = -EIO;
		goto out;
	}

	cxidev_dbg(&cass_dev->cdev, "sl sbus wr (sbus = 0x%08X, reg = 0x%02X, data = 0x%08X)\n",
		sbus_addr, reg_addr, req_data);

	rtn = 0;

out:
	mutex_unlock(&cass_dev->sl.sbus_lock);

	return rtn;
}

#define CASS_SL_SBUS_OP_DELAY_US      50
#define CASS_SL_SBUS_OP_TIMOUT_MS     500
#define CASS_SL_SBUS_OP_POLL_INTVL_MS 10
int cass_sl_sbus_cmd(void *sbus_accessor, int ring, u32 req_data,
			    u8 reg_addr, u8 dev_addr, u8 op,
			    u32 *rsp_data, u8 *result_code, u8 *overrun,
			    u8 *error, int timeout_ms, u32 flags)
{
	int                        rtn;
	struct cass_dev           *cass_dev = sbus_accessor;
	struct cxi_sbus_op_params  params;

	params.req_data      = req_data;
	params.data_addr     = reg_addr;
	params.rx_addr       = dev_addr;
	params.command       = op;
	params.delay         = CASS_SL_SBUS_OP_DELAY_US;
	params.timeout       = CASS_SL_SBUS_OP_TIMOUT_MS;
	params.poll_interval = CASS_SL_SBUS_OP_POLL_INTVL_MS;

	cxidev_dbg(&cass_dev->cdev, "sl sbus cmd\n");

	mutex_lock(&cass_dev->sl.sbus_lock);

	rtn = cxi_sbus_op(&cass_dev->cdev, &params, rsp_data, result_code, overrun);
	if (rtn != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus cmd - cxi_sbus_op failed [%d]\n", rtn);
		goto out;
	}

	cxidev_dbg(&cass_dev->cdev,
		"sl sbus cmd (op - 0x%X, dev = 0x%02X, reg = 0x%02X, data = 0x%08X, result = %u)\n",
		op, dev_addr, reg_addr, req_data, *result_code);

	// FIXME: for now just checking reset
	if (((op & 0xF) == 0) && (*result_code != 0)) {
		cxidev_err(&cass_dev->cdev, "sl sbus cmd - cxi_sbus_op (result = %u)\n",
			   *result_code);
		rtn = -EBADRQC;
		goto out;
	}
	if (*overrun != 0) {
		cxidev_err(&cass_dev->cdev, "sl sbus cmd - cxi_sbus_op (overrun = %u)\n", *overrun);
		rtn = -EIO;
		goto out;
	}

	rtn = 0;

out:
	mutex_unlock(&cass_dev->sl.sbus_lock);

	return rtn;
}

#define CASS_SL_PMI_RD_READY_MAX_COUNT 2
#define CASS_SL_PMI_RD_READY_DELAY_MS  1000
#define CASS_SL_PMI_RD_MAX_COUNT 10
#define CASS_SL_PMI_RD_DELAY_MS  100
int cass_sl_pmi_rd(void *pmi_accessor, u8 lgrp_num, u32 addr, u16 *data)
{
	int                           rtn;
	struct cass_dev              *cass_dev = pmi_accessor;
	int                           x;
	union c2_hni_serdes_pmi_ctl   pmi_ctl;
	union c2_hni_serdes_pmi_data  pmi_data;

	mutex_lock(&cass_dev->sl.pmi_lock);

	/* check ready */
	for (x = 0; x < CASS_SL_PMI_RD_READY_MAX_COUNT; ++x) {
		cass_read(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));
		if (!pmi_ctl.pmi_lp_ack)
			break;
		msleep(CASS_SL_PMI_RD_READY_DELAY_MS);
	}
	if (x >= CASS_SL_PMI_RD_READY_MAX_COUNT) {
		cxidev_err(&cass_dev->cdev,
			"sl pmi rd - ready timeout (ctl = 0x%016llX)\n", pmi_ctl.qw);
		rtn = -1;
		goto out;
	}

	/* start */
	pmi_ctl.qw          = 0ULL;
	pmi_ctl.pmi_lp_addr = addr;
	pmi_ctl.pmi_lp_en   = 1;
	cxidev_dbg(&cass_dev->cdev, "sl pmi rd - ctrl 0x%016llX -> 0x%08X\n",
		pmi_ctl.qw, C2_HNI_SERDES_PMI_CTL);
	cass_write(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));

	/* poll for done */
	for (x = 0; x < CASS_SL_PMI_RD_MAX_COUNT; ++x) {
		cass_read(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));
		cxidev_dbg(&cass_dev->cdev, "sl pmi rd - check #%d 0x%08X = 0x%016llX\n",
			x, C2_HNI_SERDES_PMI_CTL, pmi_ctl.qw);
		if (pmi_ctl.pmi_lp_error) {
			cxidev_err(&cass_dev->cdev, "sl pml rd - error\n");
			rtn = -1;
			goto out;
		}
		if (pmi_ctl.pmi_lp_ack)
			break;
		msleep(CASS_SL_PMI_RD_DELAY_MS);
	}
	if (x >= CASS_SL_PMI_RD_MAX_COUNT) {
		cxidev_err(&cass_dev->cdev, "sl pmi rd - timeout\n");
		rtn = -ETIMEDOUT;
		goto out;
	}

	/* check valid */
	if (!pmi_ctl.pmi_lp_read_vld) {
		cxidev_err(&cass_dev->cdev, "sl pmi rd - data not valid\n");
		rtn = -1;
		goto out;
	}

	/* get data */
	cass_read(cass_dev, C2_HNI_SERDES_PMI_DATA, &pmi_data, sizeof(pmi_data));

	cxidev_dbg(&cass_dev->cdev, "sl pmi rd - read 0x%08X = 0x%016llX\n",
		C2_HNI_SERDES_PMI_DATA, pmi_data.qw);

	*data = pmi_data.pmi_lp_rddata;

	cxidev_dbg(&cass_dev->cdev, "sl pmi rd (addr = 0x%08X, data = 0x%04X)\n", addr, *data);

	rtn = 0;

out:
	/* finish transaction */
	pmi_ctl.qw = 0ULL;
	cass_write(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));

	mutex_unlock(&cass_dev->sl.pmi_lock);

	return rtn;
}

#define CASS_SL_PMI_WR_READY_MAX_COUNT 2
#define CASS_SL_PMI_WR_READY_DELAY_MS  1000
#define CASS_SL_PMI_WR_MAX_COUNT 10
#define CASS_SL_PMI_WR_DELAY_MS  100
int cass_sl_pmi_wr(void *pmi_accessor, u8 lgrp_num, u32 addr, u16 data, u16 mask)
{
	int                           rtn;
	struct cass_dev              *cass_dev = pmi_accessor;
	int                           x;
	union c2_hni_serdes_pmi_ctl   pmi_ctl;
	union c2_hni_serdes_pmi_data  pmi_data;

	mutex_lock(&cass_dev->sl.pmi_lock);

	/* check ready */
	for (x = 0; x < CASS_SL_PMI_WR_READY_MAX_COUNT; ++x) {
		cass_read(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));
		if (!pmi_ctl.pmi_lp_ack)
			break;
		msleep(CASS_SL_PMI_WR_READY_DELAY_MS);
	}
	if (x >= CASS_SL_PMI_WR_READY_MAX_COUNT) {
		cxidev_err(&cass_dev->cdev,
			"sl pmi wr - ready timeout (ctl = 0x%016llX)\n", pmi_ctl.qw);
		rtn = -1;
		goto out;
	}

	/* load data */
	pmi_data.qw              = 0ULL;
	pmi_data.pmi_lp_wrdata   = data;
	pmi_data.pmi_lp_maskdata = ~mask;
	cxidev_dbg(&cass_dev->cdev, "sl pmi wr - data 0x%016llX -> 0x%08X\n",
		pmi_data.qw, C2_HNI_SERDES_PMI_DATA);
	cass_write(cass_dev, C2_HNI_SERDES_PMI_DATA, &pmi_data, sizeof(pmi_data));

	/* start */
	pmi_ctl.qw           = 0ULL;
	pmi_ctl.pmi_lp_addr  = addr;
	pmi_ctl.pmi_lp_en    = 1;
	pmi_ctl.pmi_lp_write = 1;
	cxidev_dbg(&cass_dev->cdev, "sl pmi wr - ctrl 0x%016llX -> 0x%08X\n",
		pmi_ctl.qw, C2_HNI_SERDES_PMI_CTL);
	cass_write(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));

	/* poll for done */
	for (x = 0; x < CASS_SL_PMI_WR_MAX_COUNT; ++x) {
		cass_read(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));
		cxidev_dbg(&cass_dev->cdev, "sl pmi wr - check #%d 0x%08X = 0x%016llX\n",
			x, C2_HNI_SERDES_PMI_CTL, pmi_ctl.qw);
		if (pmi_ctl.pmi_lp_error) {
			cxidev_err(&cass_dev->cdev, "sl pmi wr - error\n");
			rtn = -1;
			goto out;
		}
		if (pmi_ctl.pmi_lp_ack)
			break;
		msleep(CASS_SL_PMI_WR_DELAY_MS);
	}
	if (x >= CASS_SL_PMI_WR_MAX_COUNT) {
		cxidev_err(&cass_dev->cdev, "sl pmi wr - timeout\n");
		rtn = -ETIMEDOUT;
		goto out;
	}

	cxidev_dbg(&cass_dev->cdev, "sl pmi wr (addr = 0x%08X, data = 0x%04X, mask = 0x%04X)\n",
		addr, data, mask);

	rtn = 0;

out:
	/* finish transaction */
	pmi_ctl.qw = 0ULL;
	cass_write(cass_dev, C2_HNI_SERDES_PMI_CTL, &pmi_ctl, sizeof(pmi_ctl));

	mutex_unlock(&cass_dev->sl.pmi_lock);

	return rtn;
}

