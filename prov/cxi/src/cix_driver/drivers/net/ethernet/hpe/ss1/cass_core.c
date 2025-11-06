// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020-2024,2025 Hewlett Packard Enterprise Development LP */

/* Cassini device handler */

#include <linux/interrupt.h>
#include <linux/pci.h>
#include <linux/types.h>
#include <linux/iopoll.h>
#include <linux/seq_file.h>
#include <linux/phy/phy.h>
#include <linux/if_vlan.h>
#include <linux/aer.h>

#include "cass_core.h"
#include "cass_sbl.h"
#include <linux/sbl.h>
#include "cass_ss1_debugfs.h"

/* Range of available number of IRQs, depending on the number of VFs
 * configured in Cassini.
 */
#define MIN_NUM_IRQS 63
#define MAX_NUM_IRQS 2048

static unsigned int sct_pid_mask = 0x7;
module_param(sct_pid_mask, uint, 0444);
MODULE_PARM_DESC(sct_pid_mask, "Source Connection PID Mask");

static unsigned int hw_rev_override;
module_param(hw_rev_override, uint, 0644);
MODULE_PARM_DESC(hw_rev_override, "Hardware revision override");

static bool enable_fgfc = true;
module_param(enable_fgfc, bool, 0444);
MODULE_PARM_DESC(enable_fgfc, "Enable/disable fine grain flow control");

#define OUTSTANDING_LIMIT_MAX 2047U
#define OUTSTANDING_LIMIT_MIN 1U
#define OUTSTANDING_LIMIT_DEFAULT 64U

static unsigned int get_pkt_limit;
module_param(get_pkt_limit, uint, 0444);
MODULE_PARM_DESC(get_pkt_limit,
		 "Maximum number of outstanding GET packets (1-2047) per MCU");

static unsigned int put_pkt_limit;
module_param(put_pkt_limit, uint, 0444);
MODULE_PARM_DESC(put_pkt_limit,
		 "Maximum number of outstanding PUT packets (1-2047) per MCU");

static unsigned int ioi_ord_limit;
module_param(ioi_ord_limit, uint, 0444);
MODULE_PARM_DESC(ioi_ord_limit,
		 "Maximum number of outstanding ordered packets (1-2047) per MCU");

static unsigned int ioi_unord_limit;
module_param(ioi_unord_limit, uint, 0444);
MODULE_PARM_DESC(ioi_unord_limit,
		 "Maximum number of outstanding unordered packets (1-2047) per MCU");

static bool ioi_enable = true;
module_param(ioi_enable, bool, 0444);
MODULE_PARM_DESC(ioi_enable,
		 "Enable the In-Out-In protocol for puts.");

#ifndef PCI_EXT_CAP_ID_DVSEC
#define PCI_EXT_CAP_ID_DVSEC 0x23
#define PCI_DVSEC_HEADER1       0x4 /* Designated Vendor-Specific Header1 */
#endif
#ifndef PCI_DVSEC_HEADER1_VID
#define PCI_DVSEC_HEADER1_VID(x)   ((x) & 0xffff)
#define PCI_DVSEC_HEADER1_REV(x)   (((x) >> 16) & 0xf)
#define PCI_DVSEC_HEADER1_LEN(x)   (((x) >> 20) & 0xfff)
#endif

static void cass_dev_release(struct device *dev)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, class_dev);

	kfree(hw);
}

/**
 * cxi_get_csrs_range() - Return the whole physical range of the CSRs
 *
 * The BAR0 contains amongst other things the CSRs. At least the retry
 * handler needs an access to them. The CXI user module needs to mmap
 * it.
 *
 * @cdev: the device
 * @base: base address
 * @len: length of the CSRs
 */
void cxi_get_csrs_range(struct cxi_dev *cdev, phys_addr_t *base, size_t *len)
{
	struct cass_dev *hw = container_of(cdev, struct
					   cass_dev, cdev);

	*base = hw->regs_base + C_MEMORG_CSR;
	*len = C_MEMORG_CSR_SIZE;

}
EXPORT_SYMBOL(cxi_get_csrs_range);

void traffic_shaping_cfg(struct cass_dev *hw)
{
	static const union c_oxe_cfg_arb_config arb_config = {
		.mfs_out_sel = 1,
		.fill_rate = 9,
	};
	const union c_oxe_cfg_arb_head_out_c head_out_c = {
		.enable = 1,
		.fill_qty = hw->tsc_fill_qty,
		.limit = DIV_ROUND_UP(8 * ETHERNET_MAX_FRAME_SIZE, 1024),
	};
	const union c_oxe_cfg_arb_head_in_c head_in_c = {
		.enable = 1,
		.fill_qty = hw->tsc_fill_qty,
		.limit = DIV_ROUND_UP(8 * ETHERNET_MAX_FRAME_SIZE, 1024),
	};

	/* 16 traffic shaping credits are returned at 100MHz frequency 200 Gbps
	 * performance.
	 */
	cass_write(hw, C_OXE_CFG_ARB_CONFIG, &arb_config, sizeof(arb_config));
	cass_write(hw, C_OXE_CFG_ARB_HEAD_OUT_C, &head_out_c,
		   sizeof(head_out_c));
	cass_write(hw, C_OXE_CFG_ARB_HEAD_IN_C, &head_in_c, sizeof(head_in_c));
}

/* Set the number of packets inflight based on link speed.
 */
static unsigned int calculate_packets_inflight(struct cass_dev *hw)
{
	unsigned int value;
	struct cxi_link_info link_info;
	bool is_link_up;

	is_link_up = cass_sbl_is_link_up(hw);
	if (!is_link_up) {
		pr_err("cxi link info error: link is not up yet.\n");
		return 0;
	}
	cxi_link_mode_get(&hw->cdev, &link_info);

	if (link_info.speed == SPEED_400000)
		value = CASS2_MAX_PACKETS_INFLIGHT;
	else
		value = CASS1_MAX_PACKETS_INFLIGHT;

	return value;
}

/* Set outstanding limits based on kernel module parameters.
 * If no input, use the calculated packets_inflight.
 */
void cass_set_outstanding_limit(struct cass_dev *hw)
{
	int i;
	unsigned int packets_inflight;
	union c_oxe_cfg_outstanding_limit outstanding_limit;

	packets_inflight = calculate_packets_inflight(hw);

	if (packets_inflight < OUTSTANDING_LIMIT_MIN ||
	    packets_inflight > OUTSTANDING_LIMIT_MAX) {
		packets_inflight = OUTSTANDING_LIMIT_DEFAULT;
	}

	if (get_pkt_limit == 0)
		get_pkt_limit = packets_inflight;
	if (get_pkt_limit < OUTSTANDING_LIMIT_MIN ||
	    get_pkt_limit > OUTSTANDING_LIMIT_MAX) {
		pr_err("Invalid get_pkt_limit %u. Setting to %u\n",
		       get_pkt_limit, OUTSTANDING_LIMIT_DEFAULT);
		get_pkt_limit = OUTSTANDING_LIMIT_DEFAULT;
	}

	if (put_pkt_limit == 0)
		put_pkt_limit = packets_inflight;
	if (put_pkt_limit < OUTSTANDING_LIMIT_MIN ||
	    put_pkt_limit > OUTSTANDING_LIMIT_MAX) {
		pr_err("Invalid put_pkt_limit %u. Setting to %u\n",
		       put_pkt_limit, OUTSTANDING_LIMIT_DEFAULT);
		put_pkt_limit = OUTSTANDING_LIMIT_DEFAULT;
	}

	if (ioi_ord_limit == 0)
		ioi_ord_limit = packets_inflight;
	if (ioi_ord_limit < OUTSTANDING_LIMIT_MIN ||
	    ioi_ord_limit > OUTSTANDING_LIMIT_MAX) {
		pr_err("Invalid ioi_ord_limit %u. Setting to %u\n",
		       ioi_ord_limit, OUTSTANDING_LIMIT_DEFAULT);
		ioi_ord_limit = OUTSTANDING_LIMIT_DEFAULT;
	}

	if (ioi_unord_limit == 0)
		ioi_unord_limit = packets_inflight;
	if (ioi_unord_limit < OUTSTANDING_LIMIT_MIN ||
		ioi_unord_limit > OUTSTANDING_LIMIT_MAX) {
		pr_err("Invalid ioi_unord_limit %u. Setting to %u\n",
		       ioi_unord_limit, OUTSTANDING_LIMIT_DEFAULT);
		ioi_unord_limit = OUTSTANDING_LIMIT_DEFAULT;
	}

	for (i = 0; i < C_OXE_CFG_OUTSTANDING_LIMIT_ENTRIES; i++) {
		cass_read(hw, C_OXE_CFG_OUTSTANDING_LIMIT(i),
			  &outstanding_limit, sizeof(outstanding_limit));
		outstanding_limit.get_limit = get_pkt_limit;
		outstanding_limit.put_limit = put_pkt_limit;
		outstanding_limit.ioi_ord_limit = ioi_ord_limit;
		outstanding_limit.ioi_unord_limit = ioi_unord_limit;
		cass_write(hw, C_OXE_CFG_OUTSTANDING_LIMIT(i),
			   &outstanding_limit, sizeof(outstanding_limit));
	}
}

/* Initialize OXE registers. */
static void cass_oxe_init(struct cass_dev *hw)
{
	int i;
	union c_oxe_cfg_common cfg_common;
	union c_oxe_cfg_fgfc fgfc;
	static const union c_oxe_cfg_arb_drr_out wdrr_out = {
		.quanta_val0 = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 32),
		.quanta_val1 = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 32),
	};
	static const union c_oxe_cfg_arb_drr_in wdrr_in = {
		.quanta_val0 = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 32),
	};
	static const union c_oxe_cfg_arb_mfs_out mfs_out = {
		.mfs0 = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 32),
		.mfs1 = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 32),
	};
	static const union c_oxe_cfg_arb_mfs_in mfs_in = {
		.mfs0 = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 32),
	};
	static const union c_oxe_cfg_mcu_priority_limit mcu_priority_limit = {
		.occ_climit = 8,
		.occ_blimit = 4,
	};
	static const union c_oxe_cfg_fgfc_cnt fgfc_cnt = {
		.vni_mask = 0xffff,
	};
	union c_oxe_cfg_fab_param fab_param;

	/* Note: All MCUs must use quanta and mfs value zero. */
	cass_write(hw, C_OXE_CFG_ARB_DRR_OUT, &wdrr_out, sizeof(wdrr_out));
	cass_write(hw, C_OXE_CFG_ARB_DRR_IN, &wdrr_in, sizeof(wdrr_in));
	cass_write(hw, C_OXE_CFG_ARB_MFS_OUT, &mfs_out, sizeof(mfs_out));
	cass_write(hw, C_OXE_CFG_ARB_MFS_IN, &mfs_in, sizeof(mfs_in));

	traffic_shaping_cfg(hw);

	/* Clear all default reserved buffer class values. */
	cass_clear(hw, C_OXE_CFG_PCT_BC_CDT(0), C_OXE_CFG_PCT_BC_CDT_SIZE);
	cass_clear(hw, C_OXE_CFG_BUF_BC_PARAM(0), C_OXE_CFG_BUF_BC_PARAM_SIZE);

	/* Lower MCU arbitration priority limits to prevent a single MCU
	 * starving other MCUs within a traffic class.
	 */
	for (i = 0; i < C_OXE_CFG_MCU_PRIORITY_LIMIT_ENTRIES; i++)
		cass_write(hw, C_OXE_CFG_MCU_PRIORITY_LIMIT(i),
			   &mcu_priority_limit, sizeof(mcu_priority_limit));

	for (i = 0; i < C_OXE_CFG_FGFC_CNT_ENTRIES; i++)
		cass_write(hw, C_OXE_CFG_FGFC_CNT(i), &fgfc_cnt,
			   sizeof(fgfc_cnt));

	cass_read(hw, C_OXE_CFG_FAB_PARAM, &fab_param, sizeof(fab_param));
	fab_param.req_32b_enabled = 0;
	fab_param.rsp_32b_enabled = 0;
	cass_write(hw, C_OXE_CFG_FAB_PARAM, &fab_param, sizeof(fab_param));

	/* Disable In-Out-In for C1 and Mix system */
	if (system_type_identifier == CASSINI_1_ONLY ||
	    system_type_identifier == CASSINI_MIX) {
		ioi_enable = false;
	}
	cass_read(hw, C_OXE_CFG_COMMON, &cfg_common,
		      sizeof(cfg_common));
	cfg_common.ioi_enable = ioi_enable;
	cass_write(hw, C_OXE_CFG_COMMON, &cfg_common,
		       sizeof(cfg_common));

	cass_read(hw, C_OXE_CFG_FGFC, &fgfc, sizeof(fgfc));
	fgfc.enable = enable_fgfc;
	cass_write(hw, C_OXE_CFG_FGFC, &fgfc, sizeof(fgfc));
}

#define DFA_NID_MASK 0xfffff000U

/* TODO: Allow multiple request and response PCP configurations. Currently only
 * request PCP 0 and response PCP 1 configuration is used.
 */
static void cass_pct_init(struct cass_dev *hw)
{
	union c_pct_cfg_ixe_rsp_fifo_limits rsp_fifo_limits;
	union c_pct_cfg_tgt_clr_req_fifo_limit req_fifo_limit;
	u32 pid_mask = sct_pid_mask;
	union c_cq_cfg_ocu_hash_mask hash_mask;
	union c_pct_cfg_sct_tag_dfa_mask dfa_mask;
	union c_pct_cfg_timing timing;
	int i;

	/* Errata 2848 */
	cass_read(hw, C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT,
		  &req_fifo_limit, sizeof(req_fifo_limit));
	req_fifo_limit.mst_dealloc_fifo_limit = 7;
	cass_write(hw, C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT,
		   &req_fifo_limit, sizeof(req_fifo_limit));

	cass_read(hw, C_PCT_CFG_IXE_RSP_FIFO_LIMITS,
		  &rsp_fifo_limits, sizeof(rsp_fifo_limits));
	rsp_fifo_limits.smt_cdt_ack_limit = 7;
	rsp_fifo_limits.spt_cdt_ack_limit = 7;
	rsp_fifo_limits.sct_cdt_ack_limit = 7;
	cass_write(hw, C_PCT_CFG_IXE_RSP_FIFO_LIMITS,
		   &rsp_fifo_limits, sizeof(rsp_fifo_limits));

	/* Set default PCT timeouts */
	cass_read(hw, C_PCT_CFG_TIMING, &timing, sizeof(timing));
	timing.spt_timeout_epoch_sel = C1_DEFAULT_SPT_TIMEOUT_EPOCH;
	timing.sct_idle_epoch_sel = C1_DEFAULT_SCT_IDLE_EPOCH;
	timing.sct_close_epoch_sel = C1_DEFAULT_SCT_CLOSE_EPOCH;
	timing.tct_timeout_epoch_sel = C1_DEFAULT_TCT_TIMEOUT_EPOCH;
	cass_write(hw, C_PCT_CFG_TIMING, &timing, sizeof(timing));

	/* Set SCT PID Mask. CQ and PCT DFA masks must be consistent. */
	cass_read(hw, C_CQ_CFG_OCU_HASH_MASK, &hash_mask, sizeof(hash_mask));

	/* Mask off PID granule space as a part of SCT mask selection. */
	pid_mask &= BIT(pid_bits) - 1;

	/* Software uses the top bit of the PID granule space to differentiate
	 * between unrestricted puts and gets. To support having unrestricted
	 * puts and gets on different SCTs, the top bit of the PID granule
	 * space must always be included in SCT selection. This bit is the top
	 * bit in the endpoint define.
	 */
	pid_mask |= BIT(C_DFA_ENDPOINT_DEFINED_BITS - 1);

	hash_mask.dfa = DFA_NID_MASK | pid_mask;
	cass_write(hw, C_CQ_CFG_OCU_HASH_MASK, &hash_mask, sizeof(hash_mask));

	for (i = 0; i < 4; i++)
		dfa_mask.dscp[i].dfa_mask = pid_mask;

	for (i = 0; i < C_PCT_CFG_SCT_TAG_DFA_MASK_ENTRIES; i++) {
		cass_write(hw, C_PCT_CFG_SCT_TAG_DFA_MASK(i), &dfa_mask,
			   sizeof(dfa_mask));

		if (cass_version(hw, CASSINI_2))
			cass_write(hw, C2_LPE_CFG_GET_FQ_DFA_MASK(i),
				   &dfa_mask, sizeof(dfa_mask));
	}

	/* Clear the default target resource settings. This is configured later
	 * as a part of traffic class configuration.
	 */
	if (cass_version(hw, CASSINI_1)) {
		cass_clear(hw, C1_PCT_CFG_TRS_TC_DED_LIMIT(0),
			   C1_PCT_CFG_TRS_TC_DED_LIMIT_SIZE);
		cass_clear(hw, C1_PCT_CFG_TRS_TC_MAX_LIMIT(0),
			   C1_PCT_CFG_TRS_TC_MAX_LIMIT_SIZE);
	} else {
		cass_clear(hw, C2_PCT_CFG_TRS_TC_DED_LIMIT(0),
			   C2_PCT_CFG_TRS_TC_DED_LIMIT_SIZE);
		cass_clear(hw, C2_PCT_CFG_TRS_TC_MAX_LIMIT(0),
			   C2_PCT_CFG_TRS_TC_MAX_LIMIT_SIZE);
	}
	cass_clear(hw, C_PCT_CFG_MST_TC_DED_LIMIT(0),
		   C_PCT_CFG_MST_TC_DED_LIMIT_SIZE);
	cass_clear(hw, C_PCT_CFG_MST_TC_MAX_LIMIT(0),
		   C_PCT_CFG_MST_TC_MAX_LIMIT_SIZE);
	cass_clear(hw, C_PCT_CFG_TCT_TC_DED_LIMIT(0),
		   C_PCT_CFG_TCT_TC_DED_LIMIT_SIZE);
	cass_clear(hw, C_PCT_CFG_TCT_TC_MAX_LIMIT(0),
		   C_PCT_CFG_TCT_TC_MAX_LIMIT_SIZE);
};

/* Wait for the hardware to be ready after a reset. */
static int wait_hw_ready(struct cass_dev *hw)
{
	void __iomem *csr = cass_csr(hw, C_RMU_STS_INIT_DONE);
	union {
		u64 value;
		union c_atu_sts_init_done atu_sts_init_done;
		union c_cq_sts_init_done cq_sts_init_done;
		union c_lpe_sts_init_done lpe_sts_init_done;
		union c_oxe_sts_common oxe_sts_common;
		union c_rmu_sts_init_done rmu_sts_init_done;
		union c_ee_sts_init_done ee_sts_init_done;
	} x;
	int rc;

	csr = cass_csr(hw, C_ATU_STS_INIT_DONE);
	rc = readq_poll_timeout(csr, x.value,
				(x.atu_sts_init_done.anchor_init_done == 1 &&
				 x.atu_sts_init_done.tag_init_done == 1 &&
				 x.atu_sts_init_done.prb_init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	csr = cass_csr(hw, C_CQ_STS_INIT_DONE);
	rc = readq_poll_timeout(csr, x.value,
				(x.cq_sts_init_done.tou_init_done == 1 &&
				 x.cq_sts_init_done.cq_init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	csr = cass_csr(hw, C_LPE_STS_INIT_DONE);
	rc = readq_poll_timeout(csr, x.value,
				(x.lpe_sts_init_done.free_mask_init_done == 0xf &&
				 x.lpe_sts_init_done.ptlte_init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	csr = cass_csr(hw, C_OXE_STS_COMMON);
	rc = readq_poll_timeout(csr, x.value,
				(x.oxe_sts_common.init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	csr = cass_csr(hw, C_RMU_STS_INIT_DONE);
	rc = readq_poll_timeout(csr, x.value,
				(x.rmu_sts_init_done.warm_reset == 0 &&
				 x.rmu_sts_init_done.init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	csr = cass_csr(hw, C_EE_STS_INIT_DONE);
	rc = readq_poll_timeout(csr, x.value,
				(x.ee_sts_init_done.warm_reset == 0 &&
				 x.ee_sts_init_done.init_done == 1),
				1, 1000000);
	if (rc)
		return rc;

	return 0;
}

/* Program the device NID. */
static void program_nid(struct cass_dev *hw, u32 nid)
{
	struct cxi_dev *cdev = &hw->cdev;
	union c_oxe_cfg_fab_sfa cfg_fab_sfa = {};
	union c_lpe_cfg_get_ctrl cfg_get_ctrl;
	union c_ixe_cfg_dfa cfg_dfa = {};
	union c_pct_cfg_pkt_misc cfg_pkt_misc;

	cdev->prop.nid = nid;

	cass_read(hw, C_PCT_CFG_PKT_MISC, &cfg_pkt_misc, sizeof(cfg_pkt_misc));
	cfg_pkt_misc.sfa_nid = cdev->prop.nid;
	cass_write(hw, C_PCT_CFG_PKT_MISC, &cfg_pkt_misc, sizeof(cfg_pkt_misc));

	cfg_fab_sfa.nid = cdev->prop.nid;
	cass_write(hw, C_OXE_CFG_FAB_SFA, &cfg_fab_sfa, sizeof(cfg_fab_sfa));

	cass_read(hw, C_LPE_CFG_GET_CTRL, &cfg_get_ctrl, sizeof(cfg_get_ctrl));
	cfg_get_ctrl.dfa_nid = cdev->prop.nid;
	cass_write(hw, C_LPE_CFG_GET_CTRL, &cfg_get_ctrl, sizeof(cfg_get_ctrl));

	cfg_dfa.dfa_chk_en = 0xFF;
	cfg_dfa.dfa_nid = cdev->prop.nid;
	cass_write(hw, C_IXE_CFG_DFA, &cfg_dfa, sizeof(cfg_dfa));

	cxi_send_async_event(cdev, CXI_EVENT_NID_CHANGED);
}

void cxi_set_nid(struct cxi_dev *cdev, u32 nid)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	program_nid(hw, nid);
	hw->cdev.prop.nid_configured = true;
}
EXPORT_SYMBOL(cxi_set_nid);

void cxi_set_nid_from_mac(struct cxi_dev *cdev, const u8 *addr)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	u32 nid;

	/* If the NID has already been explicitly configured, do not override it. */
	if (hw->cdev.prop.nid_configured)
		return;

	nid = ether_addr_to_u64(addr) & 0xFFFFFULL;
	program_nid(hw, nid);
	cxidev_warn(&hw->cdev,
		    "NID has not been explicitly set. NID derived from MAC: 0x%x\n", nid);
}
EXPORT_SYMBOL(cxi_set_nid_from_mac);

/* Determine the Cassini link speed. When CCIX Extended Speed Mode
 * (ESM) is enabled, non-standard speeds exist.
 */
static int get_pcie_link_speed(struct cass_dev *hw)
{
	struct pci_dev *pdev = hw->cdev.pdev;
	enum pci_bus_speed speed;
	u32 val;
	int pos;
	int rc;

	/* Get the regular PCIe speed first */
	speed = pcie_get_speed_cap(pdev);
	switch (speed) {
	case PCIE_SPEED_2_5GT:
	case PCIE_SPEED_5_0GT:
	case PCIE_SPEED_8_0GT:
	case PCIE_SPEED_16_0GT:
	case PCIE_SPEED_32_0GT:
		hw->pcie_link_speed = speed;
		break;

	default:
		hw->pcie_link_speed = CASS_SPEED_UNKNOWN;
		break;
	}

	/* ESM is only present on first generation Cassini hardware. */
	if (cass_version(hw, CASSINI_2))
		return 0;

	pos = 0;
	while ((pos = pci_find_next_ext_capability(pdev, pos,
						   PCI_EXT_CAP_ID_DVSEC))) {
		pci_read_config_dword(pdev, pos + PCI_DVSEC_HEADER1, &val);
		if (PCI_DVSEC_HEADER1_VID(val) == 0x0001 &&
		    PCI_DVSEC_HEADER1_REV(val) == 0x1 &&
		    PCI_DVSEC_HEADER1_LEN(val) == 0x040)
			break;
	}
	if (pos == 0) {
		/* The header is not present when using PCIe
		 * passthrough with the vfio-pci driver.
		 */
		cxidev_warn(&hw->cdev, "Cannot find the PCIe DVSEC header\n");
		return 0;
	}

	/* ESM control register */
	rc = pci_read_config_dword(pdev, pos + 0x18, &val);
	if (rc) {
		cxidev_err(&hw->cdev, "Cannot get ESM status\n");
		return -EIO;
	}
	hw->esm_active = !!(val & BIT(15));

	if (!hw->esm_active)
		return 0;

	/* ESM PCIe link speed, from the ESM status register */
	rc = pci_read_config_dword(pdev, pos + 0x14, &val);
	if (rc) {
		cxidev_err(&hw->cdev, "Cannot get PCIe link speed\n");
		return -EIO;
	}

	switch (val & 0x7f) {
	case 0x1:
		hw->pcie_link_speed = PCIE_SPEED_2_5GT;
		break;
	case 0x2:
		hw->pcie_link_speed = PCIE_SPEED_5_0GT;
		break;
	case 0x3:
		hw->pcie_link_speed = PCIE_SPEED_8_0GT;
		break;
	case 0x6:
		hw->pcie_link_speed = PCIE_SPEED_16_0GT;
		break;
	case 0xa:
		hw->pcie_link_speed = CASS_SPEED_20_0GT;
		break;
	case 0xf:
		hw->pcie_link_speed = CASS_SPEED_25_0GT;
		break;
	default:
		hw->pcie_link_speed = CASS_SPEED_UNKNOWN;
		break;
	}

	return 0;
}

static const unsigned int cfg_crnc_csr[] = {
	C_ATU_CFG_CRNC,
	C_CQ_CFG_CRNC,
	C_EE_CFG_CRNC,
	C_HNI_CFG_CRNC,
	C_HNI_PML_CFG_CRNC,
	C_IXE_CFG_CRNC,
	C_LPE_CFG_CRNC,
	C_MB_CFG_CRNC,
	C_MST_CFG_CRNC,
	C_OXE_CFG_CRNC,
	C_PARBS_CFG_CRNC,
	C_PCT_CFG_CRNC,
	C_PI_CFG_CRNC,
	C_PI_IPD_CFG_CRNC,
	C_RMU_CFG_CRNC,
};

/* Initializes the hardware. */
static int init_hw(struct cass_dev *hw)
{
	int ret;
	int i;
	u64 do_dump = 0;
	union c_lpe_cfg_min_free_shft lpe_cfg_min_free_shift = {
		.min_free_shft = min_free_shift,
	};
	static const union c_mb_cfg_crmc cfg_crmc = {
		.ring_timeout = 0xff,
		.rd_addr_invalid_decerr_disable = 1,
	};
	static const union c_mb_cfg_misc cfg_mask = {
		.nic_warm_rst0 = 1,
		.nic_warm_rst1 = 1,
		.nic_warm_rst2 = 1,
		.nic_warm_rst3 = 1,
		.nic_warm_rst4 = 1,
		.nic_warm_rst5 = 1,
		.nic_warm_rst6 = 1,
		.nic_warm_rst7 = 1,
		.nic_warm_rst8 = 1,
		.nic_warm_rst9 = 1,
	};
	union c_mb_cfg_misc cfg;

	/* Trigger a warm reset of the hardware - Only the bits common
	 * to Cassini 1 & 2 are used here.
	 */
	cass_read(hw, C_MB_CFG_MISC, &cfg, sizeof(cfg));
	cfg.qw |= cfg_mask.qw;
	cass_write(hw, C_MB_CFG_MISC, &cfg, sizeof(cfg));

	/* This CSR must be cleared while warm reset is asserted. */
	cass_clear(hw, C_EE_DBG_LM_CB_TO_EQ_MAP(0), C_EE_DBG_LM_CB_TO_EQ_MAP_SIZE);

	/* De-assert warm reset */
	cfg.qw &= ~cfg_mask.qw;
	cass_write(hw, C_MB_CFG_MISC, &cfg, sizeof(cfg));

	cass_flush_pci(hw);

	if (wait_hw_ready(hw)) {
		cxidev_err(&hw->cdev, "chip not ready\n");
		return -ENOTSUPP;
	}

	cass_init(hw);

	/* Fake hardware initialization */
	if (HW_PLATFORM_NETSIM(hw))
		cass_write(hw, NICSIM_CONFIG_DUMP, &do_dump, sizeof(do_dump));

	for (i = 0; i < C_MB_CFG_CRMC_ENTRIES; i++)
		cass_write(hw, C_MB_CFG_CRMC(i), &cfg_crmc, sizeof(cfg_crmc));

	for (i = 0; i < ARRAY_SIZE(cfg_crnc_csr); i++) {
		static const union c_atu_cfg_crnc cfg_crnc = {
			.csr_timeout = 0xfd,
		};

		/* Use c_atu_cfg_crnc as it's the same structure for
		 * all CSRs
		 */
		cass_write(hw, cfg_crnc_csr[i], &cfg_crnc, sizeof(cfg_crnc));
	}

	cass_write(hw, C_LPE_CFG_MIN_FREE_SHFT, &lpe_cfg_min_free_shift,
		   sizeof(lpe_cfg_min_free_shift));

	if (cass_version(hw, CASSINI_2)) {
		union c2_lpe_cfg_match_sel cfg_match_sel = {
			.vni_cfg = vni_matching,
		};

		cass_write(hw, C2_LPE_CFG_MATCH_SEL, &cfg_match_sel,
			   sizeof(cfg_match_sel));
	}

	ret = cass_dmac_init(hw);
	if (ret)
		goto __init_hw_err_0;

	ret = cass_telem_init(hw);
	if (ret)
		goto __init_hw_err_1;

	cass_pct_init(hw);
	cass_hni_init(hw);
	cass_tle_init(hw);
	cass_cq_init(hw);
	cass_oxe_init(hw);

	ret = cass_ixe_init(hw);
	if (ret)
		goto __init_hw_err_2;

	cass_pte_init(hw);
	ret = cass_atu_init(hw);
	if (ret)
		goto __init_hw_err_2;

	cass_ee_init(hw);
	ret = cass_tc_init(hw);
	if (ret)
		goto __init_hw_err_3;

	cxi_dev_init_eth_tx_profile(hw);

	return 0;

__init_hw_err_3:
	cass_atu_fini(hw);

__init_hw_err_2:
	cass_hni_fini(hw);
	cass_telem_fini(hw);

__init_hw_err_1:
	cass_dmac_fini(hw);

__init_hw_err_0:
	return ret;
}

static void fini_hw(struct cass_dev *hw)
{
	cxi_eth_tx_profile_cleanup(hw);
	cass_tc_fini(hw);
	cass_atu_fini(hw);
	cass_hni_fini(hw);
	cass_telem_fini(hw);
	cass_dmac_fini(hw);
}

/* Check whether the hardware is supported, and set the version. */
static int is_supported(struct cass_dev *hw)
{
	cass_read(hw, C_MB_STS_REV, &hw->rev, sizeof(hw->rev));

	if (!HW_PLATFORM_ASIC(hw) && !HW_PLATFORM_Z1(hw) &&
	    !HW_PLATFORM_NETSIM(hw)) {
		cxidev_err(&hw->cdev, "device %x/%x/%x/%x/%x is not supported\n",
			   hw->rev.vendor_id, hw->rev.device_id,
			   hw->rev.platform, hw->rev.proto, hw->rev.rev);
		return -ENOTSUPP;
	}

	if (hw->rev.vendor_id == PCI_VENDOR_ID_CRAY &&
	    hw->rev.device_id == PCI_DEVICE_ID_CASSINI_1) {
		if (hw->rev.rev == 1)
			hw->cdev.prop.cassini_version = CASSINI_1_0;
		else if (hw->rev.rev == 2)
			hw->cdev.prop.cassini_version = CASSINI_1_1;
		else
			goto unsupported;
	} else if (hw->rev.vendor_id == PCI_VENDOR_ID_HPE &&
		   hw->rev.device_id == PCI_DEVICE_ID_CASSINI_2) {
		if (hw->rev.rev == 1)
			hw->cdev.prop.cassini_version = CASSINI_2_0;
		else
			goto unsupported;
	}

	if (hw_rev_override) {
		if (hw_rev_override != CASSINI_1_0 &&
		    hw_rev_override != CASSINI_1_1 &&
		    hw_rev_override != CASSINI_2_0)
			goto unsupported;

		hw->cdev.prop.cassini_version = hw_rev_override;
	}

	if (cass_version(hw, CASSINI_2) && HW_PLATFORM_Z1(hw) &&
	    (hw->rev.proto & 0x001) == 1) {
		cxidev_info(&hw->cdev, "device has reduced PCT tables\n");
		hw->reduced_pct_tables = true;
	}

	return 0;

unsupported:
	cxidev_err(&hw->cdev, "device %x/%x/%x/%x/%x is not supported\n",
		   hw->rev.vendor_id, hw->rev.device_id,
		   hw->rev.platform, hw->rev.proto, hw->rev.rev);
	return -ENOTSUPP;
}

static int cass_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	int i;
	int rc = 0;
	struct cass_dev *hw;
	bool is_physfn;
	int pos;
	u32 nid;

	hw = kzalloc(sizeof(*hw), GFP_KERNEL);
	if (!hw)
		return -ENOMEM;


	pci_set_drvdata(pdev, hw);
	hw->cdev.pdev = pdev;
	INIT_LIST_HEAD(&hw->lni_list);
	spin_lock_init(&hw->lni_lock);
	refcount_set(&hw->refcount, 1);

	atomic_set(&hw->stats.lni, 0);

	/* Initializes the RMU domains */
	atomic_set(&hw->stats.domain, 0);
	ida_init(&hw->rmu_index_table);
	ida_init(&hw->domain_table);
	spin_lock_init(&hw->domain_lock);
	spin_lock_init(&hw->rmu_lock);

	/* Initialize the EQs */
	atomic_set(&hw->stats.eq, 0);
	ida_init(&hw->eq_index_table);
	spin_lock_init(&hw->eq_shadow_lock);
	mutex_init(&hw->init_eq_hw_state);

	/* Initialize the communication profiles. */
	ida_init(&hw->cp_table);
	mutex_init(&hw->cp_lock);

	/* Initialize the transmit and target CQs. */
	atomic_set(&hw->stats.txq, 0);
	atomic_set(&hw->stats.tgq, 0);
	ida_init(&hw->cq_table);
	spin_lock_init(&hw->cq_shadow_lock);
	mutex_init(&hw->cq_init_lock);

	/* Initialize the PTs */
	atomic_set(&hw->stats.pt, 0);
	ida_init(&hw->pte_table);
	ida_init(&hw->pt_index_table);
	ida_init(&hw->set_list_table);
	ida_init(&hw->multicast_table);
	spin_lock_init(&hw->lpe_shadow_lock);
	spin_lock_init(&hw->rmu_portal_list_lock);
	mutex_init(&hw->mst_table_lock);
	mutex_init(&hw->pte_transition_sm_lock);
	atomic_set(&hw->plec_count, 0);

	/* Initialize the CTs. */
	atomic_set(&hw->stats.ct, 0);
	mutex_init(&hw->ct_init_lock);
	ida_init(&hw->ct_table);

	ida_init(&hw->lni_table);
	cass_rgid_init(hw);

	ida_init(&hw->md_index_table);

	for (i = 0; i < C_PE_COUNT; i++)
		ida_init(&hw->le_pool_ids[i]);

	ida_init(&hw->tle_pool_ids);
	mutex_init(&hw->msg_relay_lock);

	mutex_init(&hw->err_flg_mutex);
	INIT_LIST_HEAD(&hw->err_flg_list);
	spin_lock_init(&hw->sfs_err_flg_lock);

	spin_lock_init(&hw->mst_match_done_lock);
	mutex_init(&hw->mst_update_lock);

	spin_lock_init(&hw->sbl_state_lock);

	mutex_init(&hw->qsfp_eeprom_lock);
	spin_lock_init(&hw->phy.lock);

	INIT_DELAYED_WORK(&hw->lni_cleanups_work, lni_cleanups_work);
	mutex_init(&hw->lni_cleanups_lock);
	INIT_LIST_HEAD(&hw->lni_cleanups_list);

	mutex_init(&hw->sbus_mutex);
	mutex_init(&hw->get_ctrl_mutex);
	mutex_init(&hw->amo_remap_to_pcie_fadd_mutex);

	hw->tsc_fill_qty = TSC_DEFAULT_FILL_QTY;
	hw->uc_platform = CUC_BOARD_TYPE_UNKNOWN;

	/* Enable Device */
	rc = pci_enable_device(pdev);
	if (rc) {
		cxidev_err(&hw->cdev, "pci_enable_device() failed.\n");
		goto hw_free;
	}

	/* The is_physfn and is_virtfn fields of struct pci_dev are only set if
	 * SR-IOV has been successfully enabled. This means that we cannot rely on
	 * them to identify physical vs. virtual functions in all cases. A PF on
	 * real hardware without SR-IOV support, a passthrough PF attached to a VM,
	 * or a VF attached to a VM, will all have is_physfn = is_virtfn = 0.
	 *
	 * Many devices have distinct PCI IDs for VF and PF, but since Cassini 1 and
	 * 2 do not, we identify them using the size of BAR 0: 512M for VF, 2G for
	 * PF.
	 */
	is_physfn = pci_resource_len(pdev, MMIO_BAR) >= MMIO_BAR_LEN_PF;
	hw->cdev.is_physfn = is_physfn;

	hw->pci_disabled = false;

	rc = pci_enable_pcie_error_reporting(pdev);
	if (rc)
		cxidev_err(&hw->cdev, "PCIe AER not enabled: %d\n", rc);

	/* Set PID space variables for NIC. */
	hw->cdev.prop.pid_bits = pid_bits;
	hw->cdev.prop.pid_count = (1 << pid_bits) - 1; /* -1 is wildcard */
	hw->cdev.prop.pid_granule =
		(1 << (DFA_EP_BITS - hw->cdev.prop.pid_bits));

	hw->cdev.prop.min_free_shift = min_free_shift;

	/* Enable DMA */
	pci_set_master(pdev);

	dma_set_max_seg_size(&hw->cdev.pdev->dev, UINT_MAX);

	rc = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(57));
	if (rc) {
		cxidev_err(&hw->cdev, "dma_set_mask_and_coherent failed: %d\n", rc);
		goto dev_disable;
	}

	/* Enable PCIe AMO requests */
	pcie_capability_set_word(pdev, PCI_EXP_DEVCTL2,
				 PCI_EXP_DEVCTL2_ATOMIC_REQ);

	/* Make sure VF 10-bit tag support bit is set for correct counting of
	 * PI_IPD_pti_tarb_blocked_on_tag.
	 */
#define PCI_SRIOV_CTRL_VF_10BIT_TAG 0x20
	pos = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_SRIOV);
	if (pos) {
		u16 word;

		pci_read_config_word(pdev, pos + PCI_SRIOV_CTRL, &word);
		word |= PCI_SRIOV_CTRL_VF_10BIT_TAG;
		pci_write_config_word(pdev, pos + PCI_SRIOV_CTRL, word);
	}

	/* Sanity Check. Check BAR 0 is defined and mapped */
	if (!(pci_resource_flags(pdev, MMIO_BAR) & IORESOURCE_MEM)) {
		cxidev_err(&hw->cdev, "Incorrect Bar configuration\n");
		goto dev_disable;
	}

	/* Take ownership of memory region */
	rc = pci_request_regions(pdev, "cxi");
	if (rc) {
		cxidev_err(&hw->cdev, "pci_request_regions() failed.\n");
		goto dev_disable;
	}

	/* Map the base registers */
	if (is_physfn) {
		hw->regs_base = pci_resource_start(pdev, MMIO_BAR);
		hw->regs = ioremap(hw->regs_base + C_MEMORG_CSR,
				   C_MEMORG_CSR_SIZE);
		if (!hw->regs) {
			cxidev_err(&hw->cdev, "ioremap failed\n");
			goto free_regions;
		}

		/* Check hardware is supported */
		rc = is_supported(hw);
		if (rc)
			goto unmap_regions;

		/* Check for input and hardware version consistency */
		if ((system_type_identifier == CASSINI_1_ONLY && !cass_version(hw, CASSINI_1)) ||
		    (system_type_identifier == CASSINI_2_ONLY && !cass_version(hw, CASSINI_2))) {
			cxidev_err(&hw->cdev, "NIC Version and system_type_identifier (%u) are inconsistent\n",
					   system_type_identifier);
			goto unmap_regions;
		}

		hw->cdev.system_type_identifier = system_type_identifier;

		rc = get_pcie_link_speed(hw);
		if (rc)
			goto unmap_regions;

#if defined(CXI_DISABLE_SRIOV)
		rc = pci_sriov_set_totalvfs(pdev, 0);
		if (rc)
			goto unmap_regions;
#endif
	}

	hw->with_vf_support = pdev->is_physfn || pci_msix_vec_count(pdev) != 2;

	hw->cdev.cxi_num = atomic_inc_return(&cxi_num);
	dev_set_name(&hw->class_dev, "cxi%u", hw->cdev.cxi_num);
	strcpy(hw->cdev.name, dev_name(&hw->class_dev));

	cxi_set_link_ops(hw);

	rc = cass_irq_init(hw);
	if (rc) {
		cxidev_err(&hw->cdev, "cass_irq_init failed: %d\n", rc);
		goto unmap_regions;
	}

	if (is_physfn) {
		/* NIC address needs to be assigned before initialization. */
		rc = init_hw(hw);
		if (rc) {
			cxidev_err(&hw->cdev, "init_hw failed: %d\n", rc);
			goto free_irqs;
		}

		rc = register_error_handlers(hw);
		if (rc)
			goto fini_hw;

		rc = cass_register_uc(hw);
		if (rc) {
			cxidev_err(&hw->cdev, "Cannot register the uC: %d\n", rc);
			goto free_error_handlers;
		}

		switch (hw->uc_platform) {
		case CUC_BOARD_TYPE_SAWTOOTH:
		case CUC_BOARD_TYPE_BRAZOS:
			if (!cass_version(hw, CASSINI_1)) {
				cxidev_err(&hw->cdev, "Invalid Cassini 1 platform: %d\n",
					   hw->uc_platform);
				goto unregister_uc;
			}
			break;
		case CUC_BOARD_TYPE_WASHINGTON:
		case CUC_BOARD_TYPE_KENNEBEC:
		case CUC_BOARD_TYPE_SOUHEGAN:
			if (!cass_version(hw, CASSINI_2)) {
				cxidev_err(&hw->cdev, "Invalid Cassini 2 platform: %d\n",
					   hw->uc_platform);
				goto unregister_uc;
			}
			break;
		default:
			cxidev_err(&hw->cdev, "Unknown uC platform %d\n",
				   hw->uc_platform);
			goto unregister_uc;
		}

		if (HW_PLATFORM_ASIC(hw) || HW_PLATFORM_NETSIM(hw)) {
			/* These platforms have a micro-controller to fetch the
			 * NIC ID from
			 */
			rc = uc_cmd_get_nic_id(hw);
			if (rc)
				goto unregister_uc;
		} else {
			hw->uc_nic = 0;
		}

		rc = uc_cmd_get_mac(hw);
		if (rc)
			eth_random_addr(hw->default_mac_addr);
		uc_cmd_set_link_leds(hw, LED_FAST_GRN);

		if (HW_PLATFORM_ASIC(hw)) {
			/* Program MAC and NID with burned-in MAC. */
			ether_addr_copy(hw->cdev.mac_addr, hw->default_mac_addr);
		} else {
			u64 reg;
			u8 mac_addr[ETH_ALEN];

			/* Pull MAC from a scratch register on netsim and derive NID from it. */
			if (HW_PLATFORM_NETSIM(hw))
				cass_read(hw, NICSIM_NIC_ID, &reg, 8);
			else
				cass_read(hw, C_MB_DBG_SCRATCH(1), &reg, 8);

			u64_to_ether_addr(reg, mac_addr);
			ether_addr_copy(hw->cdev.mac_addr, mac_addr);
		}

		nid = ether_addr_to_u64(hw->cdev.mac_addr) & 0xFFFFFULL;
		program_nid(hw, nid);

		/* SBL init must happen after uC init, since SBL needs to know
		 * which NIC is associated with this driver instance
		 */
		rc = hw->link_ops->link_start(hw);
		if (rc)
			goto unregister_uc;

		cass_tc_get_hni_pause_cfg(hw);

		if (cass_version(hw, CASSINI_1)) {
			rc = cass_register_port(hw);
			if (rc)
				goto link_fini;
		}

		rc = cass_ptp_init(hw);
		if (rc)
			goto unregister_port;

		rc = cxi_dmac_desc_set_alloc(&hw->cdev, 1, "pt_id");
		if (rc < 0)
			goto cass_ptp_fini;
		hw->dmac_pt_id = rc;

		hw->mst_entries = dma_alloc_coherent(&pdev->dev,
						     C_MST_DBG_MST_TABLE_SIZE,
						     &hw->mst_entries_dma_addr,
						     GFP_KERNEL);
		if (!hw->mst_entries)
			goto cass_dmac_fini;

		rc = cxi_dmac_desc_set_add(&hw->cdev, hw->dmac_pt_id,
					   hw->mst_entries_dma_addr,
					   C_MST_DBG_MST_TABLE(0),
					   C_MST_DBG_MST_TABLE_SIZE);
		if (rc)
			goto cass_mst_fini;
	}

	/* Set after TC/QOS init */
	hw->cdev.untagged_eth_pcp = hw->qos.untagged_eth_pcp;

	hw->cdev.prop.device_rev = hw->rev.rev;
	hw->cdev.prop.device_proto = hw->rev.proto;
	hw->cdev.prop.device_platform = hw->rev.platform;

	hw->max_eth_rxsize = VLAN_ETH_FRAME_LEN;

	cass_probe_debugfs_init(hw);

	/* RX and TX profile setup */
	cass_dev_rx_tx_profiles_init(hw);

	/* Resource Group setup */
	cass_dev_rgroup_init(hw);

	/* Service setup */
	rc = cass_svc_init(hw);
	if (rc)
		goto rgroup_fini;

	rc = cxi_configfs_device_init(hw);
	if (rc) {
		pr_err("configfs initialize failed\n");
		goto svc_fini;
	}

	if (!is_physfn) {
		struct cxi_get_dev_properties_cmd cmd = {
			.op = CXI_OP_GET_DEV_PROPERTIES,
		};
		struct cxi_properties_info resp;
		size_t reply_len = sizeof(resp);

		rc = cass_vf_init(hw);
		if (rc)
			goto configfs_fini;

		/* Retrieve the device properties */
		rc = cxi_send_msg_to_pf(&hw->cdev, &cmd, sizeof(cmd),
					&resp, &reply_len);
		if (rc != 0) {
			cxidev_err(&hw->cdev, "BAD: Reply has return code %d\n", rc);
			goto vf_fini;
		} else if (reply_len != sizeof(resp)) {
			cxidev_err(&hw->cdev,
				   "BAD: Reply has unexpected length %zu\n",
				   reply_len);
			goto vf_fini;
		} else {
			cxidev_dbg(&hw->cdev, "Reply is valid\n");

			hw->cdev.prop = resp;
		}
	}

	/* Export device information. */
	rc = create_sysfs_properties(hw);
	if (rc)
		goto vf_fini;

	cxi_add_device(&hw->cdev);

	if (is_physfn)
		cass_phy_start(hw, true);

	if (is_physfn)
		start_pcie_monitoring(hw);

	/* Register the device with sysfs */
	hw->class_dev.class = &cxi_class;
	hw->class_dev.parent = &pdev->dev;
	hw->class_dev.release = cass_dev_release;
	dev_set_drvdata(&hw->class_dev, hw);
	if (device_register(&hw->class_dev)) {
		put_device(&hw->class_dev);
		goto vf_fini;
	}

	return 0;

vf_fini:
	if (!is_physfn)
		cass_vf_fini(hw);

configfs_fini:
	cxi_configfs_cleanup(hw);
svc_fini:
	cass_svc_fini(hw);
rgroup_fini:
	cass_dev_rgroup_fini(hw);
	cass_dev_rx_tx_profiles_fini(hw);
	debugfs_remove_recursive(hw->debug_dir);
cass_mst_fini:
	dma_free_coherent(&pdev->dev, C_MST_DBG_MST_TABLE_SIZE,
			  hw->mst_entries, hw->mst_entries_dma_addr);
cass_dmac_fini:
	if (is_physfn)
		cxi_dmac_desc_set_free(&hw->cdev, hw->dmac_pt_id);
cass_ptp_fini:
	if (is_physfn)
		cass_ptp_fini(hw);
unregister_port:
	if (is_physfn)
		if (cass_version(hw, CASSINI_1))
			cass_unregister_port(hw);
link_fini:
	if (is_physfn)
		hw->link_ops->link_fini(hw);
unregister_uc:
	if (is_physfn) {
		uc_cmd_set_link_leds(hw, LED_OFF);
		cass_unregister_uc(hw);
	}
free_error_handlers:
	if (is_physfn)
		deregister_error_handlers(hw);
fini_hw:
	if (is_physfn)
		fini_hw(hw);
free_irqs:
	cass_irq_fini(hw);
unmap_regions:
	if (is_physfn)
		iounmap(hw->regs);
free_regions:
	pci_release_regions(pdev);
dev_disable:
	cxidev_err(&hw->cdev, "deregistering device\n");

	pci_clear_master(pdev);
	pci_disable_pcie_error_reporting(pdev);
	pci_disable_device(pdev);
hw_free:
	kfree(hw);

	return -ENODEV;
}

static void cass_remove(struct pci_dev *pdev)
{
	struct cass_dev *hw = pci_get_drvdata(pdev);

	if (hw->cdev.is_physfn) {
		stop_pcie_monitoring(hw);
		cass_phy_stop(hw, true);
		hw->link_ops->link_fini(hw);
	}

	cxi_remove_device(&hw->cdev);

	cancel_delayed_work_sync(&hw->lni_cleanups_work);
	lni_cleanups(hw, true);

	cass_rgid_fini(hw);
	cxi_configfs_cleanup(hw);
	cass_svc_fini(hw);
	cass_dev_rgroup_fini(hw);
	cass_dev_rx_tx_profiles_fini(hw);

	destroy_sysfs_properties(hw);
	debugfs_remove_recursive(hw->debug_dir);

	if (hw->cdev.is_physfn) {
		if (hw->num_vfs)
			cass_sriov_configure(hw->cdev.pdev, 0);
		dma_free_coherent(&pdev->dev, C_MST_DBG_MST_TABLE_SIZE,
				  hw->mst_entries, hw->mst_entries_dma_addr);
		cxi_dmac_desc_set_free(&hw->cdev, hw->dmac_pt_id);
		cass_ptp_fini(hw);
		uc_cmd_set_link_leds(hw, LED_OFF);
		cass_unregister_uc(hw);
		if (cass_version(hw, CASSINI_1))
			cass_unregister_port(hw);
		deregister_error_handlers(hw);
		fini_hw(hw);
	} else {
		cass_vf_fini(hw);
	}

	cass_irq_fini(hw);

	if (hw->cdev.is_physfn)
		iounmap(hw->regs);
	pci_release_regions(pdev);
	pci_clear_master(pdev);
	pci_disable_pcie_error_reporting(pdev);
	if (!hw->pci_disabled)
		pci_disable_device(pdev);

	cxidev_WARN_ONCE(&hw->cdev, !refcount_dec_and_test(&hw->refcount),
			 "Resource leaks - HW refcount not zero: %d\n",
			 refcount_read(&hw->refcount));
	device_unregister(&hw->class_dev);
}

void cass_disable_device(struct pci_dev *pdev)
{
	struct cass_dev *hw = pci_get_drvdata(pdev);

	cxidev_warn(&hw->cdev, "disabling device");

	pci_clear_master(pdev);
	pci_disable_device(pdev);

	hw->pci_disabled = true;
}

static struct pci_device_id cass_ids[] = {
	{ PCI_DEVICE(PCI_VENDOR_ID_CRAY, PCI_DEVICE_ID_CASSINI_1) },
	{ PCI_DEVICE(PCI_VENDOR_ID_HPE, PCI_DEVICE_ID_CASSINI_2) },
	{ 0, },
};

static struct pci_driver cass_pci_driver = {
	.name = KBUILD_MODNAME,
	.id_table = cass_ids,
	.probe = cass_probe,
	.remove = cass_remove,
	.sriov_configure = cass_sriov_configure,
};

MODULE_DEVICE_TABLE(pci, cass_ids);

int hw_register(void)
{
	int rc;

	rc = genl_register_family(&cxierr_genl_family);
	if (rc)
		return rc;

	rc = pci_register_driver(&cass_pci_driver);

	if (rc)
		genl_unregister_family(&cxierr_genl_family);

	return rc;
}

void hw_unregister(void)
{
	pci_unregister_driver(&cass_pci_driver);
	genl_unregister_family(&cxierr_genl_family);
}

/**
 * cxi_retry_handler_running() - Check if retry handler is running.
 *
 * @cdev: CXI device.
 *
 * Return: True if retry handler is running. Else, false.
 */
bool cxi_retry_handler_running(struct cxi_dev *cdev)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);

	return hw->pct_eq_n != C_EQ_NONE;
}
EXPORT_SYMBOL(cxi_retry_handler_running);
