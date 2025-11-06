// SPDX-License-Identifier: GPL-2.0
/* Copyright 2020 Hewlett Packard Enterprise Development LP */

/* Traffic Class (TC) Management */

#include "cass_core.h"
#include "cass_ss1_debugfs.h"

static unsigned int pfc_buf_skid_space = 65280;
module_param(pfc_buf_skid_space, uint, 0444);
MODULE_PARM_DESC(pfc_buf_skid_space,
		 "PFC buffer skid space (high water threshold) in bytes");

/* PCP that PCT Traffic will use. By default we use the PCP specified
 * in the QoS Profile.
 */
static int pct_control_traffic_pcp = -1;
module_param(pct_control_traffic_pcp, int, 0444);
MODULE_PARM_DESC(pct_control_traffic_pcp,
		 "PCP used for PCT control traffic.\n"
		 "Values 0-7 are valid\n"
		 "Value -1 - Use PCP Specified in QoS Profile\n");

/* Prevent (or not) usage of PCT PCP by an RDMA TC */
static bool disable_user_pct_tc_usage = true;
module_param(disable_user_pct_tc_usage, bool, 0644);
MODULE_PARM_DESC(disable_user_pct_tc_usage,
		 "Prevent users from using traffic class assigned to PCT control traffic");

/* PCT Control Traffic needs a dedicated PCP that is configured to ignore
 * pause from Rosetta. However if the provided PCP is in use by an RDMA TC
 * we may disallow use of this RDMA TC. This value will only be set if
 * an RDMA TC is in fact using the PCP used by PCP.
 */
static int pct_control_traffic_tc = -1;

static unsigned int active_qos_profile = 2;
module_param(active_qos_profile, uint, 0444);
MODULE_PARM_DESC(active_qos_profile,
		 "QoS Profile to load. Must match fabric QoS Profile\n"
		 "1 - HPC\n"
		 "2 - LL_BE_BD_ET\n"
		 "3 - LL_BE_BD_ET1_ET2\n");

static const char * const cxi_qos_strs[] = {
	[CXI_QOS_HPC] = "HPC",
	[CXI_QOS_LL_BE_BD_ET] = "LL_BE_BD_ET",
	[CXI_QOS_LL_BE_BD_ET1_ET2] = "LL_BE_BD_ET1_ET2",
};

static const char * const cxi_tc_strs_lc[] = {
	[CXI_TC_DEDICATED_ACCESS]	= "dedicated_access",
	[CXI_TC_LOW_LATENCY]            = "low_latency",
	[CXI_TC_BULK_DATA]              = "bulk_data",
	[CXI_TC_BEST_EFFORT]            = "best_effort",
	[CXI_ETH_SHARED]		= "ethernet_shared",
	[CXI_ETH_TC1]			= "ethernet1",
	[CXI_ETH_TC2]			= "ethernet2",
};

static const char * const cxi_tc_strs_uc[] = {
	[CXI_TC_DEDICATED_ACCESS]	= "DEDICATED_ACCESS",
	[CXI_TC_LOW_LATENCY]            = "LOW_LATENCY",
	[CXI_TC_BULK_DATA]              = "BULK_DATA",
	[CXI_TC_BEST_EFFORT]            = "BEST_EFFORT",
	[CXI_ETH_SHARED]		= "ETHERNET_SHARED",
	[CXI_ETH_TC1]			= "ETHERNET1",
	[CXI_ETH_TC2]			= "ETHERNET2",
};

/**
 * cxi_get_tc_req_pcp() - Get the request PCP associated with a TC.
 * If the TC is not enabled return -1.
 *
 * @dev: CXI device
 * @tc: Traffic Class
 */
int cxi_get_tc_req_pcp(struct cxi_dev *dev, unsigned int tc)
{
	struct cass_dev *hw = container_of(dev, struct cass_dev, cdev);

	if (tc >= CXI_ETH_TC_MAX)
		return -EINVAL;

	if (!hw->qos.tcs_active[tc])
		return -1;

	if (is_eth_tc(tc))
		return hw->qos.tcs[tc].eth_settings.pcp;
	else
		return hw->qos.tcs[tc].dscp_pcp_settings.req_pcp;
}
EXPORT_SYMBOL(cxi_get_tc_req_pcp);

static int cass_tc_get_oxe_bc(struct cass_dev *hw)
{
	int bc = hw->next_oxe_bc;

	if (bc >= C_OXE_CFG_BUF_BC_PARAM_ENTRIES)
		return -ENOSPC;

	hw->next_oxe_bc++;

	return bc;
}

static int cass_tc_req_oxe_bc_cfg(struct cass_dev *hw, unsigned int spt_rsvd,
				  unsigned int smt_rsvd, unsigned int sct_rsvd,
				  unsigned int srb_rsvd, unsigned int pbuf_rsvd)
{
	const union c_oxe_cfg_pct_bc_cdt pct_bc_cdt_cfg = {
		.spt_cdt_rsvd = hw->reduced_pct_tables ? 8 : spt_rsvd,
		.smt_cdt_rsvd = hw->reduced_pct_tables ? 8 : smt_rsvd,
		.sct_cdt_rsvd = hw->reduced_pct_tables ? 8 : sct_rsvd,
		.srb_cdt_rsvd = hw->reduced_pct_tables ? 8 : srb_rsvd,
	};
	const union c_oxe_cfg_buf_bc_param buf_bc_param_cfg = {
		.bc_reserv_buf_cdt = pbuf_rsvd,
		.bc_max_mtu = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 256),
	};
	int bc;

	bc = cass_tc_get_oxe_bc(hw);
	if (bc < 0)
		goto out;

	cass_write(hw, C_OXE_CFG_PCT_BC_CDT(bc), &pct_bc_cdt_cfg,
		   sizeof(pct_bc_cdt_cfg));
	cass_write(hw, C_OXE_CFG_BUF_BC_PARAM(bc), &buf_bc_param_cfg,
		   sizeof(buf_bc_param_cfg));

	if (hw->reduced_pct_tables) {
		static const union c_oxe_cfg_pct_cdt oxe_cfg_pct_cdt = {
			.spt_cdt_limit = 208,
			.smt_cdt_limit = 112,
			.srb_cdt_limit = 336,
			.sct_cdt_limit = 112,
		};

		cass_write(hw, C_OXE_CFG_PCT_CDT, &oxe_cfg_pct_cdt,
			   sizeof(oxe_cfg_pct_cdt));
	}

out:
	return bc;
}

static int cass_tc_rsp_oxe_bc_cfg(struct cass_dev *hw, unsigned int pbuf_rsvd)
{
	const union c_oxe_cfg_pct_bc_cdt pct_bc_cdt_cfg = {
		.ignore_spt = 1,
		.ignore_smt = 1,
		.ignore_sct = 1,
		.ignore_srb = 1,
	};
	const union c_oxe_cfg_buf_bc_param buf_bc_param_cfg = {
		.bc_reserv_buf_cdt = pbuf_rsvd,
		.bc_max_mtu = DIV_ROUND_UP(PORTALS_MAX_FRAME_SIZE, 256),
	};
	int bc;

	bc = cass_tc_get_oxe_bc(hw);
	if (bc < 0)
		goto out;

	cass_write(hw, C_OXE_CFG_PCT_BC_CDT(bc), &pct_bc_cdt_cfg,
		   sizeof(pct_bc_cdt_cfg));
	cass_write(hw, C_OXE_CFG_BUF_BC_PARAM(bc), &buf_bc_param_cfg,
		   sizeof(buf_bc_param_cfg));

out:
	return bc;
}

static int cass_tc_eth_oxe_bc_cfg(struct cass_dev *hw, unsigned int spt_rsvd,
				  unsigned int pbuf_rsvd)
{
	const union c_oxe_cfg_pct_bc_cdt pct_bc_cdt_cfg = {
		.spt_cdt_rsvd = spt_rsvd,
		.ignore_smt = 1,
		.ignore_sct = 1,
		.ignore_srb = 1,
	};
	const union c_oxe_cfg_buf_bc_param buf_bc_param_cfg = {
		.bc_reserv_buf_cdt = pbuf_rsvd,
		.bc_max_mtu = DIV_ROUND_UP(ETHERNET_MAX_FRAME_SIZE, 256),
	};
	int bc;

	bc = cass_tc_get_oxe_bc(hw);
	if (bc < 0)
		goto out;

	cass_write(hw, C_OXE_CFG_PCT_BC_CDT(bc), &pct_bc_cdt_cfg,
		   sizeof(pct_bc_cdt_cfg));
	cass_write(hw, C_OXE_CFG_BUF_BC_PARAM(bc), &buf_bc_param_cfg,
		   sizeof(buf_bc_param_cfg));

out:
	return bc;
}

static int cass_tc_get_branch_bucket(struct cass_dev *hw)
{
	int bucket = hw->next_oxe_branch_bucket;

	if (bucket >= C_OXE_CFG_ARB_BRANCH_ENTRIES)
		return -ENOSPC;

	hw->next_oxe_branch_bucket++;

	return bucket;
}

static int cass_tc_get_leaf_bucket(struct cass_dev *hw)
{
	int bucket = hw->next_oxe_leaf_bucket;

	if (bucket >= C_OXE_CFG_ARB_LEAF_ENTRIES)
		return -ENOSPC;

	hw->next_oxe_leaf_bucket++;

	return bucket;
}

static int cass_tc_oxe_mcu_cfg(struct cass_dev *hw, unsigned int mcu_base,
			       unsigned int mcu_count, unsigned int pcp,
			       unsigned int tsc, unsigned int bc,
			       unsigned int mfs_index)
{
	const union c_oxe_cfg_mcu_param mcu_cfg = {
		.bc_map = bc,
		.tsc_map = tsc,
		.pcp_map = pcp,
		.wdrr_in_sel = mfs_index,
		.wdrr_out_sel = mfs_index,
	};
	int i;

	if (mcu_base >= C_OXE_MCU_COUNT ||
	    (mcu_base + mcu_count) >  C_OXE_MCU_COUNT)
		return -EINVAL;

	for (i = 0; i < mcu_count; i++)
		cass_write(hw, C_OXE_CFG_MCU_PARAM(mcu_base + i), &mcu_cfg,
			   sizeof(mcu_cfg));

	return 0;
}

static int cass_tc_oxe_pcp_pause_cfg(struct cass_dev *hw, unsigned int pcp,
				     unsigned int tsc)
{
	union c_oxe_cfg_arb_pcp_mask pcp_mask;

	if (pcp >= C_OXE_CFG_ARB_PCP_MASK_ENTRIES)
		return -EINVAL;

	cass_read(hw, C_OXE_CFG_ARB_PCP_MASK(pcp), &pcp_mask, sizeof(pcp_mask));

	pcp_mask.tsc_mask |= BIT(tsc);

	cass_write(hw, C_OXE_CFG_ARB_PCP_MASK(pcp), &pcp_mask,
		   sizeof(pcp_mask));

	return 0;
}

static void cass_tc_oxe_branch_bucket_cfg(struct cass_dev *hw,
					  int tc)
{
	const struct cass_tc_oxe_settings *oxe_settings =
		&hw->qos.tcs[tc].oxe_settings;
	int assured_fill_qty =
		max_t(int, 1, hw->tsc_fill_qty * oxe_settings->assured_percent / 100);
	int ceiling_fill_qty =
		max_t(int, 1, hw->tsc_fill_qty * oxe_settings->ceiling_percent / 100);
	int branch = hw->qos.tcs[tc].branch;
	const union c_oxe_cfg_arb_branch branch_cfg = {
		.pri = oxe_settings->branch_priority,
		.mfs_in_sel = oxe_settings->mfs_index,
		.mfs_out_sel = oxe_settings->mfs_index,
	};
	const union c_oxe_cfg_arb_branch_out_a branch_out_a_cfg = {
		.fill_qty = assured_fill_qty,
		.limit = oxe_settings->bucket_limit,
	};
	const union c_oxe_cfg_arb_branch_out_c branch_out_c_cfg = {
		.fill_qty = ceiling_fill_qty,
		.limit = oxe_settings->bucket_limit,
		.enable = 1,
	};
	const union c_oxe_cfg_arb_branch_in_a branch_in_a_cfg = {
		.fill_qty = assured_fill_qty,
		.limit = oxe_settings->bucket_limit,
	};
	const union c_oxe_cfg_arb_branch_in_c branch_in_c_cfg = {
		.fill_qty = ceiling_fill_qty,
		.limit = oxe_settings->bucket_limit,
		.enable = 1,
	};

	if (branch >= C_OXE_CFG_ARB_BRANCH_ENTRIES)
		return;

	cass_write(hw, C_OXE_CFG_ARB_BRANCH(branch), &branch_cfg,
		   sizeof(branch_cfg));
	cass_write(hw, C_OXE_CFG_ARB_BRANCH_OUT_A(branch), &branch_out_a_cfg,
		   sizeof(branch_out_a_cfg));
	cass_write(hw, C_OXE_CFG_ARB_BRANCH_OUT_C(branch), &branch_out_c_cfg,
		   sizeof(branch_out_c_cfg));
	cass_write(hw, C_OXE_CFG_ARB_BRANCH_IN_A(branch), &branch_in_a_cfg,
		   sizeof(branch_in_a_cfg));
	cass_write(hw, C_OXE_CFG_ARB_BRANCH_IN_C(branch), &branch_in_c_cfg,
		   sizeof(branch_in_c_cfg));
}

static void cass_tc_oxe_leaf_bucket_cfg(struct cass_dev *hw,
					int tc, int leaf, bool req)
{
	const struct cass_tc_oxe_settings *oxe_settings =
		&hw->qos.tcs[tc].oxe_settings;
	bool is_ethernet = is_eth_tc(tc);
	int assured_fill_qty = is_ethernet ?
		max_t(int, 1, hw->tsc_fill_qty * oxe_settings->assured_percent / 100) :
		1;
	int ceiling_fill_qty = is_ethernet ?
		max_t(int, 1, hw->tsc_fill_qty * oxe_settings->ceiling_percent / 100) :
		hw->tsc_fill_qty;
	unsigned int priority = req ? oxe_settings->leaf_request_priority :
				oxe_settings->leaf_response_priority;
	const union c_oxe_cfg_arb_leaf leaf_cfg = {
		.pri = priority,
		.parent = hw->qos.tcs[tc].branch,
		.enable_in = !is_ethernet,
		.mfs_in_sel = oxe_settings->mfs_index,
		.mfs_out_sel = oxe_settings->mfs_index,
	};
	const union c_oxe_cfg_arb_leaf_out_a leaf_out_a_cfg = {
		.fill_qty = assured_fill_qty,
		.limit = oxe_settings->bucket_limit,
	};
	const union c_oxe_cfg_arb_leaf_out_c leaf_out_c_cfg = {
		.fill_qty = ceiling_fill_qty,
		.limit = oxe_settings->bucket_limit,
		.enable = 1,
	};
	const union c_oxe_cfg_arb_leaf_in_a leaf_in_a_cfg = {
		.fill_qty = assured_fill_qty,
		.limit = oxe_settings->bucket_limit,
	};
	const union c_oxe_cfg_arb_leaf_in_c leaf_in_c_cfg = {
		.fill_qty = ceiling_fill_qty,
		.limit = oxe_settings->bucket_limit,
		.enable = 1,
	};

	cass_write(hw, C_OXE_CFG_ARB_LEAF(leaf), &leaf_cfg, sizeof(leaf_cfg));
	cass_write(hw, C_OXE_CFG_ARB_LEAF_OUT_A(leaf), &leaf_out_a_cfg,
		   sizeof(leaf_out_a_cfg));
	cass_write(hw, C_OXE_CFG_ARB_LEAF_OUT_C(leaf), &leaf_out_c_cfg,
		   sizeof(leaf_out_c_cfg));
	cass_write(hw, C_OXE_CFG_ARB_LEAF_IN_A(leaf), &leaf_in_a_cfg,
		   sizeof(leaf_in_a_cfg));
	cass_write(hw, C_OXE_CFG_ARB_LEAF_IN_C(leaf), &leaf_in_c_cfg,
		   sizeof(leaf_in_c_cfg));
}

/* Configure OXE for RDMA classes */
static int cass_tc_oxe_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			   unsigned int req_pcp, unsigned int rsp_pcp,
			   unsigned int cq_mcu_base,
			   unsigned int cq_mcu_count,
			   unsigned int lpe_mcu_base,
			   unsigned int lpe_mcu_count,
			   unsigned int ixe_mcu_base,
			   unsigned int ixe_mcu_count,
			   unsigned int tou_mcu)
{
	const struct cass_tc_oxe_settings *oxe_settings =
		&hw->qos.tcs[tc].oxe_settings;
	int req_bc;
	int rsp_bc;
	int branch_bucket;
	int req_leaf_bucket;
	int rsp_leaf_bucket;
	int ret;
	int spt_rsvd = oxe_settings->spt_rsvd;

	/* If PCT control traffic needs to be remapped to a different traffic
	 * class, reuse the response PCP of the remapped traffic class.
	 */
	int mcu_pcp = hw->qos.pct_control_pcp;

	/* Calculate SPT reserved back on number of MCUs and ordered packets
	 * inflight. Only 3 MCUs worth of SPT entries are currently supported
	 * being reserved.
	 */
	if (!spt_rsvd) {
		spt_rsvd = min_t(int, 3, cq_mcu_count);
		if (cass_version(hw, CASSINI_2))
			spt_rsvd *= CASS2_MAX_PACKETS_INFLIGHT;
		else
			spt_rsvd *= CASS1_MAX_PACKETS_INFLIGHT;
	}

	req_bc = cass_tc_req_oxe_bc_cfg(hw, spt_rsvd, oxe_settings->smt_rsvd,
					oxe_settings->sct_rsvd,
					oxe_settings->srb_rsvd,
					oxe_settings->pbuf_rsvd);
	if (req_bc < 0)
		return req_bc;

	hw->qos.tcs[tc].req_bc = req_bc;

	rsp_bc = cass_tc_rsp_oxe_bc_cfg(hw, oxe_settings->pbuf_rsvd);
	if (rsp_bc < 0)
		return rsp_bc;

	branch_bucket = cass_tc_get_branch_bucket(hw);
	if (branch_bucket < 0)
		return branch_bucket;

	hw->qos.tcs[tc].branch = branch_bucket;
	cass_tc_oxe_branch_bucket_cfg(hw, tc);

	req_leaf_bucket = cass_tc_get_leaf_bucket(hw);
	if (req_leaf_bucket < 0)
		return req_leaf_bucket;
	cass_tc_oxe_leaf_bucket_cfg(hw, tc, req_leaf_bucket, true);
	hw->qos.tcs[tc].leaf[0] = req_leaf_bucket;

	rsp_leaf_bucket = cass_tc_get_leaf_bucket(hw);
	if (rsp_leaf_bucket < 0)
		return rsp_leaf_bucket;
	cass_tc_oxe_leaf_bucket_cfg(hw, tc, rsp_leaf_bucket, false);
	hw->qos.tcs[tc].leaf[1] = rsp_leaf_bucket;

	ret = cass_tc_oxe_mcu_cfg(hw, cq_mcu_base, cq_mcu_count, mcu_pcp,
				  req_leaf_bucket, req_bc,
				  oxe_settings->mfs_index);
	if (ret)
		return ret;

	ret = cass_tc_oxe_mcu_cfg(hw, tou_mcu, 1, mcu_pcp, req_leaf_bucket,
				  req_bc, oxe_settings->mfs_index);
	if (ret)
		return ret;

	ret = cass_tc_oxe_mcu_cfg(hw, lpe_mcu_base, lpe_mcu_count, mcu_pcp,
				  req_leaf_bucket, req_bc,
				  oxe_settings->mfs_index);
	if (ret)
		return ret;

	ret = cass_tc_oxe_mcu_cfg(hw, ixe_mcu_base, ixe_mcu_count, rsp_pcp,
				  rsp_leaf_bucket, rsp_bc,
				  oxe_settings->mfs_index);
	if (ret)
		return ret;

	ret = cass_tc_oxe_pcp_pause_cfg(hw, req_pcp, req_leaf_bucket);
	if (ret)
		return ret;

	ret = cass_tc_oxe_pcp_pause_cfg(hw, rsp_pcp, rsp_leaf_bucket);
	if (ret)
		return ret;

	return 0;
}

static int cass_tc_get_cq_tc(struct cass_dev *hw)
{
	int tc = hw->next_cq_tc;

	if (tc >= C_CQ_CFG_FQ_RESRV_ENTRIES)
		return -ENOSPC;

	hw->next_cq_tc++;

	return tc;
}

static void cass_tc_cq_pfq_cfg(struct cass_dev *hw, unsigned int cq_tc,
			       unsigned int pfq, unsigned int high_thresh)
{
	const union c_cq_cfg_pfq_tc_map tc_map_cfg = {
		.pfq_tc = cq_tc,
	};
	const union c_cq_cfg_pfq_thresh_table thresh_table_cfg = {
		.high_thresh = high_thresh,
	};

	cass_write(hw, C_CQ_CFG_PFQ_TC_MAP(pfq), &tc_map_cfg,
		   sizeof(tc_map_cfg));
	cass_write(hw, C_CQ_CFG_PFQ_THRESH_TABLE(pfq), &thresh_table_cfg,
		   sizeof(thresh_table_cfg));
}

static void cass_tc_cq_fq_cfg(struct cass_dev *hw, unsigned int cq_tc,
			      unsigned int buf_rsvd)
{
	const union c_cq_cfg_fq_resrv fq_resrv_cfg = {
		.num_reserved = buf_rsvd,
		.max_alloc = 256,
	};

	cass_write(hw, C_CQ_CFG_FQ_RESRV(cq_tc), &fq_resrv_cfg,
		   sizeof(fq_resrv_cfg));
};

static int cass_tc_get_ocuset(struct cass_dev *hw, unsigned int fq_count)
{
	int ocuset = hw->next_cq_ocuset;

	if (hw->next_cq_ocuset >= C_CQ_CFG_OCUSET_OCU_TABLE_ENTRIES ||
	    (hw->next_cq_ocuset + fq_count) >
	     C_CQ_CFG_OCUSET_OCU_TABLE_ENTRIES)
		return -EINVAL;

	hw->next_cq_ocuset += fq_count;

	return ocuset;
}

static int cass_tc_get_ocu(struct cass_dev *hw, unsigned int ocu_count)
{
	int ocu = hw->next_cq_ocu;

	if (hw->next_cq_ocu >= C_CQ_CFG_OCU_TABLE_ENTRIES ||
	    (hw->next_cq_ocu + ocu_count) > C_CQ_CFG_OCU_TABLE_ENTRIES)
		return -EINVAL;

	hw->next_cq_ocu += ocu_count;

	return ocu;
}

static int cass_tc_cq_dynamic_ocuset_cfg(struct cass_dev *hw,
					 unsigned int ocu_base,
					 unsigned int ocu_count,
					 unsigned int fq_count)
{
	const union c_cq_cfg_ocuset_ocu_table ocu_table_cfg = {
		.ocu_base = ocu_base,
		.ocu_count = ilog2(ocu_count),
	};
	const union c_cq_cfg_ocuset_fq_table fq_table_cfg = {
		.fq_count = min_t(unsigned int, fq_count, 8),
	};
	int ocuset;

	ocuset = cass_tc_get_ocuset(hw, fq_table_cfg.fq_count);
	if (ocuset < 0)
		return ocuset;

	cass_write(hw, C_CQ_CFG_OCUSET_OCU_TABLE(ocuset), &ocu_table_cfg,
		   sizeof(ocu_table_cfg));
	cass_write(hw, C_CQ_CFG_OCUSET_FQ_TABLE(ocuset), &fq_table_cfg,
		   sizeof(fq_table_cfg));

	return ocuset;
}

static int cass_tc_cq_static_ocuset_cfg(struct cass_dev *hw,
					unsigned int ocu_base,
					unsigned int ocu_count,
					unsigned int fq_count)
{
	const union c_cq_cfg_ocuset_ocu_table ocu_table_cfg = {
		.ocu_base = ocu_base,
		.ocu_count = ilog2(ocu_count),
		.ocu_static = 1,
	};
	union c_cq_cfg_ocu_table ocu_cfg = {};
	int ocuset;
	int ocu;

	ocuset = cass_tc_get_ocuset(hw, fq_count);
	if (ocuset < 0)
		return ocuset;

	cass_write(hw, C_CQ_CFG_OCUSET_OCU_TABLE(ocuset), &ocu_table_cfg,
		   sizeof(ocu_table_cfg));

	for (ocu = ocu_base; ocu < (ocu_base + ocu_count); ocu++) {
		ocu_cfg.fq = ocuset + (ocu % fq_count);

		cass_write(hw, C_CQ_CFG_OCU_TABLE(ocu), &ocu_cfg,
			   sizeof(ocu_cfg));
	}

	return ocuset;
}

static int cass_tc_cq_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			  unsigned int *cq_mcu_base,
			  unsigned int *cq_mcu_count,
			  unsigned int *cq_tc, bool is_static)
{
	const struct cass_tc_cq_settings *cq_settings =
		&hw->qos.tcs[tc].cq_settings;
	unsigned int fq_count;
	unsigned int ocu_count;
	int ocu_base;
	int tc_cq;
	int ocuset;

	fq_count = is_static ? cq_settings->static_fq_count :
		cq_settings->dynamic_fq_count;
	if (!fq_count)
		return -EINVAL;

	ocu_count = is_static ? fq_count : fq_count * 4;

	ocu_base = cass_tc_get_ocu(hw, ocu_count);
	if (ocu_base < 0)
		return ocu_base;

	tc_cq = cass_tc_get_cq_tc(hw);
	if (tc_cq < 0)
		return tc_cq;

	if (is_static)
		ocuset = cass_tc_cq_static_ocuset_cfg(hw, ocu_base, ocu_count,
						      fq_count);
	else
		ocuset = cass_tc_cq_dynamic_ocuset_cfg(hw, ocu_base, ocu_count,
						       fq_count);
	if (ocuset < 0)
		return ocuset;

	cass_tc_cq_pfq_cfg(hw, tc_cq, ocuset, cq_settings->pfq_high_thresh);
	cass_tc_cq_fq_cfg(hw, tc_cq, cq_settings->fq_buf_reserved);

	*cq_mcu_base = ocuset + C_OXE_CQ_MCU_START;
	*cq_mcu_count = fq_count;
	*cq_tc = tc_cq;

	return ocuset;
}

static void cass_tc_get_ixe_wrq(struct cass_dev *hw,
				unsigned int *wrq_base, unsigned int *wrq_mask)
{
	*wrq_base = hw->next_ixe_wrq;
	*wrq_mask = 3;

	hw->next_ixe_wrq += 4;
	if (hw->next_ixe_wrq >= 32)
		hw->next_ixe_wrq = 0;
}

static void cass_tc_ixe_dscp_pcp_map(struct cass_dev *hw, unsigned int dscp,
				     unsigned int pcp, unsigned int rsp_dscp)
{
	union c_ixe_cfg_dscp_wrq_map wrq_map_cfg;
	union c_ixe_cfg_dscp_dscp_tc_map tc_map_cfg;
	unsigned int wrq_base;
	unsigned int wrq_mask;
	unsigned int index = dscp / C_IXE_CFG_DSCP_DSCP_TC_MAP_ARRAY_SIZE;
	unsigned int offset = dscp % C_IXE_CFG_DSCP_DSCP_TC_MAP_ARRAY_SIZE;

	cass_tc_get_ixe_wrq(hw, &wrq_base, &wrq_mask);

	cass_read(hw, C_IXE_CFG_DSCP_WRQ_MAP(dscp), &wrq_map_cfg,
		  sizeof(wrq_map_cfg));

	wrq_map_cfg.base = wrq_base;
	wrq_map_cfg.hash_mask = wrq_mask;

	cass_write(hw, C_IXE_CFG_DSCP_WRQ_MAP(dscp), &wrq_map_cfg,
		   sizeof(wrq_map_cfg));

	cass_read(hw, C_IXE_CFG_DSCP_DSCP_TC_MAP(index), &tc_map_cfg,
		  sizeof(tc_map_cfg));

	tc_map_cfg.map[offset].dscp = rsp_dscp;
	tc_map_cfg.map[offset].tc = pcp;

	cass_write(hw, C_IXE_CFG_DSCP_DSCP_TC_MAP(index), &tc_map_cfg,
		  sizeof(tc_map_cfg));
}

static int cass_tc_get_ixe_fq(struct cass_dev *hw, unsigned int fq_count)
{
	int fq = hw->next_ixe_fq;

	if (hw->next_ixe_fq >= 24 ||
	    (hw->next_ixe_fq + fq_count) > 24)
		return -EINVAL;

	hw->next_ixe_fq += fq_count;

	return fq;
}

static int cass_tc_ixe_fq_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			      unsigned int req_pcp, unsigned int *ixe_mcu_base,
			      unsigned int *ixe_mcu_count)
{
	const struct cass_tc_ixe_settings *ixe_settings =
		&hw->qos.tcs[tc].ixe_settings;
	union c_ixe_cfg_tc_fq_map fq_map_cfg = {};
	union c_ixe_cfg_mp_tc_fq_map mp_fq_map_cfg = {};
	unsigned int i;
	int fq_base;

	fq_base = cass_tc_get_ixe_fq(hw, ixe_settings->fq_count);
	if (fq_base < 0)
		return fq_base;

	for (i = 0; i < ixe_settings->fq_count; i++)
		fq_map_cfg.put_resp_mask |= BIT(fq_base + i);
	fq_map_cfg.get_req_mask = fq_map_cfg.put_resp_mask;

	cass_write(hw, C_IXE_CFG_TC_FQ_MAP(req_pcp), &fq_map_cfg,
		   sizeof(fq_map_cfg));

	mp_fq_map_cfg.base = fq_base;
	mp_fq_map_cfg.num_fqs_mask =
		min_t(unsigned int, 4,
		      rounddown_pow_of_two(ixe_settings->fq_count)) - 1;

	cass_write(hw, C_IXE_CFG_MP_TC_FQ_MAP(req_pcp), &mp_fq_map_cfg,
		   sizeof(mp_fq_map_cfg));

	*ixe_mcu_base = fq_base + C_OXE_IXE_MCU_START;
	*ixe_mcu_count = ixe_settings->fq_count;

	return 0;
}

static int cass_tc_get_lpe_fq(struct cass_dev *hw, unsigned int fq_count)
{
	int fq = hw->next_lpe_fq;

	if (hw->next_lpe_fq >= 16 ||
	    (hw->next_lpe_fq + fq_count) > 16)
		return -EINVAL;

	hw->next_lpe_fq += fq_count;

	return fq;
}

static int cass_tc_lpe_fq_cfg(struct cass_dev *hw, unsigned int req_pcp,
			      unsigned int fq_count)
{
	union c_lpe_cfg_fq_tc_map fq_tc_map_cfg;
	int fq_base;

	fq_base = cass_tc_get_lpe_fq(hw, fq_count);
	if (fq_base < 0)
		return fq_base;

	cass_read(hw, C_LPE_CFG_FQ_TC_MAP, &fq_tc_map_cfg,
		  sizeof(fq_tc_map_cfg));

	fq_tc_map_cfg.tc[req_pcp].fq_base = fq_base;
	fq_tc_map_cfg.tc[req_pcp].fq_mask =
		min_t(unsigned int, 8, rounddown_pow_of_two(fq_count)) - 1;

	cass_write(hw, C_LPE_CFG_FQ_TC_MAP, &fq_tc_map_cfg,
		   sizeof(fq_tc_map_cfg));

	return fq_base;
}

static int cass_tc_lpe_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			   unsigned int req_pcp, unsigned int *lpe_mcu_base,
			   unsigned int *lpe_mcu_count)
{
	const struct cass_tc_lpe_settings *lpe_settings =
		&hw->qos.tcs[tc].lpe_settings;
	int fq_base;

	fq_base = cass_tc_lpe_fq_cfg(hw, req_pcp, lpe_settings->fq_count);
	if (fq_base < 0)
		return fq_base;

	*lpe_mcu_base = fq_base + C_OXE_LPE_MCU_START;
	*lpe_mcu_count = lpe_settings->fq_count;

	return 0;
}

static void cass_tc_pct_pcp_cfg(struct cass_dev *hw, unsigned int req_pcp,
				unsigned int trs_rsvd, unsigned int trs_max,
				unsigned int tct_rsvd, unsigned int tct_max,
				unsigned int mst_rsvd, unsigned int mst_max)
{
	const union c_pct_cfg_mst_tc_ded_limit mst_ded_limit_cfg = {
		.ded_limit = mst_rsvd,
	};
	const union c_pct_cfg_mst_tc_max_limit mst_max_limit_cfg = {
		.max_limit = mst_max,
	};
	const union c_pct_cfg_tct_tc_ded_limit tct_ded_limit_cfg = {
		.ded_limit = tct_rsvd,
	};
	const union c_pct_cfg_tct_tc_max_limit tct_max_limit_cfg = {
		.max_limit = tct_max,
	};

	if (cass_version(hw, CASSINI_1)) {
		const union c1_pct_cfg_trs_tc_ded_limit trs_ded_limit_cfg = {
			.ded_limit = trs_rsvd,
		};
		const union c1_pct_cfg_trs_tc_max_limit trs_max_limit_cfg = {
			.max_limit = trs_max,
		};

		cass_write(hw, C1_PCT_CFG_TRS_TC_DED_LIMIT(req_pcp),
			   &trs_ded_limit_cfg, sizeof(trs_ded_limit_cfg));
		cass_write(hw, C1_PCT_CFG_TRS_TC_MAX_LIMIT(req_pcp),
			   &trs_max_limit_cfg, sizeof(trs_max_limit_cfg));
	} else {
		const union c2_pct_cfg_trs_tc_ded_limit trs_ded_limit_cfg = {
			.ded_limit = trs_rsvd,
		};
		const union c2_pct_cfg_trs_tc_max_limit trs_max_limit_cfg = {
			.max_limit = trs_max,
		};

		cass_write(hw, C2_PCT_CFG_TRS_TC_DED_LIMIT(req_pcp),
			   &trs_ded_limit_cfg, sizeof(trs_ded_limit_cfg));
		cass_write(hw, C2_PCT_CFG_TRS_TC_MAX_LIMIT(req_pcp),
			   &trs_max_limit_cfg, sizeof(trs_max_limit_cfg));
	}

	cass_write(hw, C_PCT_CFG_MST_TC_DED_LIMIT(req_pcp), &mst_ded_limit_cfg,
			sizeof(mst_ded_limit_cfg));
	cass_write(hw, C_PCT_CFG_MST_TC_MAX_LIMIT(req_pcp), &mst_max_limit_cfg,
			sizeof(mst_max_limit_cfg));
	cass_write(hw, C_PCT_CFG_TCT_TC_DED_LIMIT(req_pcp), &tct_ded_limit_cfg,
			sizeof(tct_ded_limit_cfg));
	cass_write(hw, C_PCT_CFG_TCT_TC_MAX_LIMIT(req_pcp), &tct_max_limit_cfg,
			sizeof(tct_max_limit_cfg));
}

static void cass_tc_pct_dscp_pcp_map(struct cass_dev *hw, unsigned int req_pcp,
				     unsigned int rsp_pcp,
				     unsigned int req_dscp,
				     unsigned int rsp_dscp)
{
	union c_pct_cfg_req_tc_rsp_tc_map req_tc_rsp_tc_map;
	union c_pct_cfg_req_dscp_rsp_dscp_map req_dscp_rsp_dscp_map;
	union c_pct_cfg_rsp_dscp_req_tc_map rsp_dscp_req_tc_map;
	int req_dscp_index = req_dscp / 8;
	int req_dscp_offset = req_dscp % 8;
	int rsp_dscp_index = rsp_dscp / 16;

	cass_read(hw, C_PCT_CFG_REQ_TC_RSP_TC_MAP, &req_tc_rsp_tc_map,
		  sizeof(req_tc_rsp_tc_map));

	/* Map the provided req_pcp to the provided rsp_pcp */
	switch (req_pcp) {
	case 0:
		req_tc_rsp_tc_map.req_tc0 = rsp_pcp;
		break;
	case 1:
		req_tc_rsp_tc_map.req_tc1 = rsp_pcp;
		break;
	case 2:
		req_tc_rsp_tc_map.req_tc2 = rsp_pcp;
		break;
	case 3:
		req_tc_rsp_tc_map.req_tc3 = rsp_pcp;
		break;
	case 4:
		req_tc_rsp_tc_map.req_tc4 = rsp_pcp;
		break;
	case 5:
		req_tc_rsp_tc_map.req_tc5 = rsp_pcp;
		break;
	case 6:
		req_tc_rsp_tc_map.req_tc6 = rsp_pcp;
		break;
	default:
		req_tc_rsp_tc_map.req_tc7 = rsp_pcp;
		break;
	}

	cass_write(hw, C_PCT_CFG_REQ_TC_RSP_TC_MAP, &req_tc_rsp_tc_map,
		   sizeof(req_tc_rsp_tc_map));

	cass_read(hw, C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP(req_dscp_index),
		  &req_dscp_rsp_dscp_map, sizeof(req_dscp_rsp_dscp_map));

	req_dscp_rsp_dscp_map.map[req_dscp_offset].rsp_dscp = rsp_dscp;

	cass_write(hw, C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP(req_dscp_index),
		   &req_dscp_rsp_dscp_map, sizeof(req_dscp_rsp_dscp_map));

	cass_read(hw, C_PCT_CFG_RSP_DSCP_REQ_TC_MAP(rsp_dscp_index),
		  &rsp_dscp_req_tc_map, sizeof(rsp_dscp_req_tc_map));

	switch (rsp_dscp % 16) {
	case 0:
		rsp_dscp_req_tc_map.dscp0_tc = req_pcp;
		break;
	case 1:
		rsp_dscp_req_tc_map.dscp1_tc = req_pcp;
		break;
	case 2:
		rsp_dscp_req_tc_map.dscp2_tc = req_pcp;
		break;
	case 3:
		rsp_dscp_req_tc_map.dscp3_tc = req_pcp;
		break;
	case 4:
		rsp_dscp_req_tc_map.dscp4_tc = req_pcp;
		break;
	case 5:
		rsp_dscp_req_tc_map.dscp5_tc = req_pcp;
		break;
	case 6:
		rsp_dscp_req_tc_map.dscp6_tc = req_pcp;
		break;
	case 7:
		rsp_dscp_req_tc_map.dscp7_tc = req_pcp;
		break;
	case 8:
		rsp_dscp_req_tc_map.dscp8_tc = req_pcp;
		break;
	case 9:
		rsp_dscp_req_tc_map.dscp9_tc = req_pcp;
		break;
	case 10:
		rsp_dscp_req_tc_map.dscp10_tc = req_pcp;
		break;
	case 11:
		rsp_dscp_req_tc_map.dscp11_tc = req_pcp;
		break;
	case 12:
		rsp_dscp_req_tc_map.dscp12_tc = req_pcp;
		break;
	case 13:
		rsp_dscp_req_tc_map.dscp13_tc = req_pcp;
		break;
	case 14:
		rsp_dscp_req_tc_map.dscp14_tc = req_pcp;
		break;
	default:
		rsp_dscp_req_tc_map.dscp15_tc = req_pcp;
		break;
	}

	cass_write(hw, C_PCT_CFG_RSP_DSCP_REQ_TC_MAP(rsp_dscp_index),
		   &rsp_dscp_req_tc_map, sizeof(rsp_dscp_req_tc_map));
}

static unsigned int cass_tc_get_pct_shared_trs_limit(struct cass_dev *hw)
{
	enum cxi_traffic_class tc;
	int rsvd = 0;

	for (tc = 0; tc < CXI_MAX_RDMA_TCS; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		rsvd += hw->qos.tcs[tc].pct_settings.trs_rsvd;
	}

	if (cass_version(hw, CASSINI_1))
		return max_t(unsigned int, C1_PCT_CFG_TRS_CAM_ENTRIES - rsvd, 0);
	else
		return max_t(unsigned int, C2_PCT_CFG_TRS_CAM_ENTRIES - rsvd, 0);
}

static unsigned int cass_tc_get_pct_shared_mst_limit(struct cass_dev *hw)
{
	enum cxi_traffic_class tc;
	int rsvd = 0;

	for (tc = 0; tc < CXI_MAX_RDMA_TCS; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		rsvd += hw->qos.tcs[tc].pct_settings.mst_rsvd;
	}

	return max_t(unsigned int, C_MST_DBG_MST_TABLE_ENTRIES - rsvd, 0);
}

static unsigned int cass_tc_get_pct_shared_tct_limit(struct cass_dev *hw)
{
	enum cxi_traffic_class tc;
	int rsvd = 0;

	for (tc = 0; tc < CXI_MAX_RDMA_TCS; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		rsvd += hw->qos.tcs[tc].pct_settings.tct_rsvd;
	}

	return max_t(unsigned int, C_PCT_CFG_TCT_RAM_ENTRIES - rsvd, 0);
}

static void cass_tc_pct_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			    unsigned int req_pcp)
{
	const struct cass_tc_pct_settings *pct_settings =
		&hw->qos.tcs[tc].pct_settings;
	unsigned int shared_trs_limit = cass_tc_get_pct_shared_trs_limit(hw);
	unsigned int shared_mst_limit = cass_tc_get_pct_shared_mst_limit(hw);
	unsigned int shared_tct_limit = cass_tc_get_pct_shared_tct_limit(hw);

	if (hw->reduced_pct_tables) {
		cass_tc_pct_pcp_cfg(hw, req_pcp, 8,
				    8 + 64,
				    8,
				    8 + 32,
				    8,
				    8 + 64);
	} else {
		cass_tc_pct_pcp_cfg(hw, req_pcp, pct_settings->trs_rsvd,
				    pct_settings->trs_rsvd + shared_trs_limit,
				    pct_settings->tct_rsvd,
				    pct_settings->tct_rsvd + shared_tct_limit,
				    pct_settings->mst_rsvd,
				    pct_settings->mst_rsvd + shared_mst_limit);
	}
}

static void cass_tc_pct_global_cfg(struct cass_dev *hw)
{
	const union c_pct_cfg_mst_crdt_limits mst_crdt_limits = {
		.total_limit = hw->reduced_pct_tables ?
			128 : C_MST_DBG_MST_TABLE_ENTRIES,
		.shared_limit = hw->reduced_pct_tables ?
			64 : cass_tc_get_pct_shared_mst_limit(hw),
	};
	const union c_pct_cfg_tct_crdt_limits tct_crdt_limits = {
		.total_limit = hw->reduced_pct_tables ?
			96 : C_PCT_CFG_TCT_RAM_ENTRIES,
		.shared_limit = hw->reduced_pct_tables ?
			32 : cass_tc_get_pct_shared_tct_limit(hw),
	};

	if (cass_version(hw, CASSINI_1)) {
		const union c1_pct_cfg_trs_crdt_limits trs_crdt_limits = {
			.total_limit = C1_PCT_CFG_TRS_CAM_ENTRIES,
			.shared_limit = cass_tc_get_pct_shared_trs_limit(hw),
		};

		cass_write(hw, C1_PCT_CFG_TRS_CRDT_LIMITS, &trs_crdt_limits,
			   sizeof(trs_crdt_limits));
	} else {
		const union c2_pct_cfg_trs_crdt_limits trs_crdt_limits = {
			.total_limit = hw->reduced_pct_tables ?
				128 : C2_PCT_CFG_TRS_CAM_ENTRIES,
			.shared_limit = hw->reduced_pct_tables ?
				64 : cass_tc_get_pct_shared_trs_limit(hw),
		};

		cass_write(hw, C2_PCT_CFG_TRS_CRDT_LIMITS, &trs_crdt_limits,
			   sizeof(trs_crdt_limits));
	}

	cass_write(hw, C_PCT_CFG_MST_CRDT_LIMITS, &mst_crdt_limits,
		   sizeof(mst_crdt_limits));
	cass_write(hw, C_PCT_CFG_TCT_CRDT_LIMITS, &tct_crdt_limits,
		   sizeof(tct_crdt_limits));
}


#define PFC_BUF_UNITS 64U

#define C1_ETHERNET_PAUSE_RESPONSE_TIME_BYTES (15 * 1024)
#define C2_ETHERNET_PAUSE_RESPONSE_TIME_BYTES (57 * 1024)

#define C1_MAX_CABLE_LENGTH_BYTES 2560
#define C2_MAX_CABLE_LENGTH_BYTES (5 * 1024)

#define C1_MAC_LATENCY_BYTES (4 * 1024)
#define C2_MAC_LATENCY_BYTES (10 * 1024)

#define MAX_PACKET_SIZE (9 * 1024)
#define STORE_AND_FORWARD_FIFO (9 * 1024)

#define C1_SKID_BUF_SIZE (64 * 1024)
#define C2_SKID_BUF_SIZE (128 * 1024)

static unsigned int cass_tc_get_pfc_high_water(const struct cass_dev *hw)
{
	unsigned int skid_space;
	unsigned int skid_buf_size;

	if (cass_version(hw, CASSINI_1)) {
		skid_space = C1_ETHERNET_PAUSE_RESPONSE_TIME_BYTES +
			C1_MAX_CABLE_LENGTH_BYTES + C1_MAC_LATENCY_BYTES +
			(2 * MAX_PACKET_SIZE) + STORE_AND_FORWARD_FIFO;
		skid_buf_size = C1_SKID_BUF_SIZE;
	} else {
		skid_space = C2_ETHERNET_PAUSE_RESPONSE_TIME_BYTES +
			C2_MAX_CABLE_LENGTH_BYTES + C2_MAC_LATENCY_BYTES +
			(2 * MAX_PACKET_SIZE) + STORE_AND_FORWARD_FIFO;
		skid_buf_size = C2_SKID_BUF_SIZE;
	}

	/* NIC-to-NIC needs extra store and forward fifo space. */
	if (!switch_connected)
		skid_space += STORE_AND_FORWARD_FIFO;

	/* Only use the user-defined PFC buffer skid space if the requested skid
	 * space is at least the minimal value.
	 */
	if (pfc_buf_skid_space) {
		if (pfc_buf_skid_space < skid_space)
			pr_info_once("PFC buf skid space %u too small. Using default %u\n",
				     pfc_buf_skid_space, skid_space);
		else if (pfc_buf_skid_space > (skid_buf_size - PFC_BUF_UNITS))
			pr_info_once("PFC buf skid space %u too large. Using default %u\n",
				     pfc_buf_skid_space, skid_space);
		else
			skid_space = pfc_buf_skid_space;
	}

	return (skid_buf_size - skid_space) / PFC_BUF_UNITS;
}

#define HIGH_LOW_WATER_SKID_SPACE_DELTA 4096U

static unsigned int cass_tc_get_pfc_low_water(const struct cass_dev *hw)
{
	return max_t(int,
		     (int)cass_tc_get_pfc_high_water(hw) -
		     (int)(HIGH_LOW_WATER_SKID_SPACE_DELTA / PFC_BUF_UNITS), 0);
}

static void cass_tc_hni_pcp_cfg(struct cass_dev *hw, unsigned int pcp,
				unsigned int pbuf_rsvd,
				unsigned int max_frame_size)
{
	union c_hni_cfg_pbuf cfg_pbuf_cfg;

	/* Since PCP packet buffer space is shared between Portals and Ethernet
	 * traffic and my be updated twice for Portals and Ethernet
	 * configuration, always take the max configured value.
	 */
	cass_read(hw, C_HNI_CFG_PBUF(pcp), &cfg_pbuf_cfg,
		  sizeof(cfg_pbuf_cfg));

	cfg_pbuf_cfg.mtu = max_t(unsigned int, cfg_pbuf_cfg.mtu,
				 DIV_ROUND_UP(max_frame_size, 128));
	cfg_pbuf_cfg.static_rsvd =
		max_t(unsigned int, cfg_pbuf_cfg.static_rsvd, pbuf_rsvd);

	cass_write(hw, C_HNI_CFG_PBUF(pcp), &cfg_pbuf_cfg,
		   sizeof(cfg_pbuf_cfg));

	if (cass_version(hw, CASSINI_1)) {
		const union c1_hni_cfg_pfc_buf pfc_buf_cfg = {
			.high_water = cass_tc_get_pfc_high_water(hw),
			.low_water = cass_tc_get_pfc_low_water(hw),
		};

		cass_write(hw, C1_HNI_CFG_PFC_BUF(pcp), &pfc_buf_cfg,
			   sizeof(pfc_buf_cfg));
	} else {
		const union c2_hni_cfg_pfc_buf pfc_buf_cfg = {
			.high_water = cass_tc_get_pfc_high_water(hw),
			.low_water = cass_tc_get_pfc_low_water(hw),
		};

		cass_write(hw, C2_HNI_CFG_PFC_BUF(pcp), &pfc_buf_cfg,
			   sizeof(pfc_buf_cfg));
	}
}

static void cass_tc_hni_dscp_pcp_map(struct cass_dev *hw, unsigned int dscp,
				     unsigned int pcp)
{
	const union c_hni_cfg_dscp_pcp dscp_pcp_cfg = {
		.pcp = pcp,
	};

	cass_write(hw, C_HNI_CFG_DSCP_PCP(dscp), &dscp_pcp_cfg,
		   sizeof(dscp_pcp_cfg));
}

static void cass_tc_enable_tx_pause(struct cass_dev *hw, unsigned int pcp)
{
	union c_hni_cfg_pause_tx_ctrl tx_ctrl_cfg;

	cass_read(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));

	tx_ctrl_cfg.enable_send_pause |= BIT(pcp);
	tx_ctrl_cfg.pev |= BIT(pcp);

	cass_write(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));
}

static void cass_tc_disable_tx_pause(struct cass_dev *hw, unsigned int pcp)
{
	union c_hni_cfg_pause_tx_ctrl tx_ctrl_cfg;

	cass_read(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));

	tx_ctrl_cfg.enable_send_pause &= ~BIT(pcp);
	tx_ctrl_cfg.pev &= ~BIT(pcp);

	cass_write(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));
}

static void cass_tc_enable_rx_pause(struct cass_dev *hw, unsigned int pcp)
{
	union c_hni_cfg_pause_rx_ctrl rx_ctrl_cfg;

	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg, sizeof(rx_ctrl_cfg));

	if (pcp == hw->qos.pct_control_pcp) {
		/* Never enable the receiving of PFC pause if the
		 * response PCP of a given traffic class is to be used
		 * for PCT control traffic.  Failure to do this
		 * results in packet drops and performance issues.
		 */
		rx_ctrl_cfg.enable_rec_pause &= ~BIT(pcp);
	} else {
		rx_ctrl_cfg.enable_rec_pause |= BIT(pcp);
	}

	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg, sizeof(rx_ctrl_cfg));
}

static void cass_tc_disable_rx_pause(struct cass_dev *hw, unsigned int pcp)
{
	union c_hni_cfg_pause_rx_ctrl rx_ctrl_cfg;

	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg,
		  sizeof(rx_ctrl_cfg));

	rx_ctrl_cfg.enable_rec_pause &= ~(1U << pcp);

	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg,
		   sizeof(rx_ctrl_cfg));
}

static void cass_tc_enable_pauses(struct cass_dev *hw, unsigned int pcp)
{
	cass_tc_enable_tx_pause(hw, pcp);
	cass_tc_enable_rx_pause(hw, pcp);
}

void cass_tc_set_tx_pause_all(struct cass_dev *hw, bool enable)
{
	int i;

	for (i = 0; i < PCP_COUNT; i++) {
		if (enable)
			cass_tc_enable_tx_pause(hw, i);
		else
			cass_tc_disable_tx_pause(hw, i);
	}
}

void cass_tc_set_rx_pause_all(struct cass_dev *hw, bool enable)
{
	int i;

	for (i = 0; i < PCP_COUNT; i++) {
		if (enable)
			cass_tc_enable_rx_pause(hw, i);
		else
			cass_tc_disable_rx_pause(hw, i);
	}
}

void cass_tc_get_hni_pause_cfg(struct cass_dev *hw)
{
	u32 pause_type;
	bool tx_pause;
	bool rx_pause;
	union c_hni_cfg_pause_tx_ctrl tx_ctrl_cfg;
	union c_hni_cfg_pause_rx_ctrl rx_ctrl_cfg;

	cass_read(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg,
		  sizeof(tx_ctrl_cfg));

	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg,
		  sizeof(rx_ctrl_cfg));

	tx_pause = ((tx_ctrl_cfg.enable_pfc_pause && tx_ctrl_cfg.pev) ||
		    tx_ctrl_cfg.enable_pause) && tx_ctrl_cfg.enable_send_pause;

	rx_pause = (rx_ctrl_cfg.pfc_rec_enable ||
		    rx_ctrl_cfg.pause_rec_enable) &&
		   rx_ctrl_cfg.enable_rec_pause;

	if (!tx_ctrl_cfg.enable_pfc_pause && !tx_ctrl_cfg.enable_pause &&
			!rx_ctrl_cfg.pfc_rec_enable && !rx_ctrl_cfg.pause_rec_enable)
		pause_type = CASS_PAUSE_TYPE_NONE;
	else if (!tx_ctrl_cfg.enable_pfc_pause && tx_ctrl_cfg.enable_pause &&
				!rx_ctrl_cfg.pfc_rec_enable && rx_ctrl_cfg.pause_rec_enable)
		pause_type = CASS_PAUSE_TYPE_GLOBAL;
	else if (tx_ctrl_cfg.enable_pfc_pause &&
				rx_ctrl_cfg.pfc_rec_enable && !rx_ctrl_cfg.pause_rec_enable)
		pause_type = CASS_PAUSE_TYPE_PFC;
	else
		pause_type = CASS_PAUSE_TYPE_INVALID;

	spin_lock(&hw->port->pause_lock);
	hw->port->pause_type = pause_type;
	hw->port->tx_pause = tx_pause;
	hw->port->rx_pause = rx_pause;
	spin_unlock(&hw->port->pause_lock);
}

static void cass_tc_hni_cfg(struct cass_dev *hw, enum cxi_traffic_class tc,
			    unsigned int req_pcp, unsigned int rsp_pcp)
{
	const struct cass_tc_hni_settings *hni_settings =
		&hw->qos.tcs[tc].hni_settings;

	cass_tc_hni_pcp_cfg(hw, req_pcp, hni_settings->pbuf_rsvd,
			    hni_settings->mfs);

	/* Response PCP is ignored for Ethernet configuration. */
	if (tc != CXI_TC_ETH)
		cass_tc_hni_pcp_cfg(hw, rsp_pcp, hni_settings->pbuf_rsvd,
				    hni_settings->mfs);

	/* TODO: Support disabling pause and global pause. */
	cass_tc_enable_pauses(hw, req_pcp);

	if (tc != CXI_TC_ETH)
		cass_tc_enable_pauses(hw, rsp_pcp);
}

#define C1_RESTRICTED_BC_SPT_RSVD 1500U
#define RESTRICTED_BC_SMT_RSVD 0U
#define RESTRICTED_BC_SCT_RSVD 0U
#define RESTRICTED_BC_SRB_RSVD 39U
#define RESTRICTED_BC_PBUF_RSVD 39U

/**
 * cass_tc_init_res_req_oxe_bc() - Initialize global OXE buffer class to be
 * used for restricted request MCUs.
 *
 * @hw: Cassini device
 *
 * This buffer class is to be used by all CXI_TC_TYPE_HRP and
 * CXI_TC_TYPE_RESTRICTED traffic class types.
 *
 * This only applies for C1.
 */
static int cass_tc_init_res_req_oxe_bc(struct cass_dev *hw)
{
	int bc;

	bc = cass_tc_req_oxe_bc_cfg(hw, C1_RESTRICTED_BC_SPT_RSVD,
				    RESTRICTED_BC_SMT_RSVD,
				    RESTRICTED_BC_SCT_RSVD,
				    RESTRICTED_BC_SRB_RSVD,
				    RESTRICTED_BC_PBUF_RSVD);

	return bc;
}

/* Set the DSCP DFA mask. */
static void set_dscp_dfa_mask(struct cass_dev *hw,
			      int dscp, unsigned int new_mask)
{
	union c_pct_cfg_sct_tag_dfa_mask dfa_mask;
	unsigned int idx = dscp / 4;
	unsigned int offset = dscp % 4;

	if (dscp == -1)
		return;

	cass_read(hw, C_PCT_CFG_SCT_TAG_DFA_MASK(idx),
		  &dfa_mask, sizeof(dfa_mask));

	dfa_mask.dscp[offset].dfa_mask = new_mask;

	cass_write(hw, C_PCT_CFG_SCT_TAG_DFA_MASK(idx),
		   &dfa_mask, sizeof(dfa_mask));

	if (cass_version(hw, CASSINI_2))
		cass_write(hw, C2_LPE_CFG_GET_FQ_DFA_MASK(idx),
			   &dfa_mask, sizeof(dfa_mask));
}

/**
 * cass_tc_dscp_pcp_map() - For all traffic classes, map DSCP and PCP values in
 * hardware.
 *
 * @hw: Cassini device
 */
static void cass_tc_dscp_pcp_map(struct cass_dev *hw)
{
	int i;
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings;
	const struct qos_profile *prof = &hw->qos;

	/* For each TC program mappings into HW */
	for (i = 0; i < CXI_MAX_RDMA_TCS; i++) {
		/* Don't configure inactive TCs */
		if (!prof->tcs_active[i])
			continue;
		dscp_pcp_settings = &prof->tcs[i].dscp_pcp_settings;

		/* IXE Mappings */
		cass_tc_ixe_dscp_pcp_map(hw, dscp_pcp_settings->res_req_dscp,
					 dscp_pcp_settings->req_pcp,
					 dscp_pcp_settings->res_rsp_dscp);
		cass_tc_ixe_dscp_pcp_map(hw, dscp_pcp_settings->res_rsp_dscp,
					 dscp_pcp_settings->rsp_pcp, 0);
		cass_tc_ixe_dscp_pcp_map(hw, dscp_pcp_settings->unres_req_dscp,
					 dscp_pcp_settings->req_pcp,
					 dscp_pcp_settings->unres_rsp_dscp);
		cass_tc_ixe_dscp_pcp_map(hw, dscp_pcp_settings->unres_rsp_dscp,
					 dscp_pcp_settings->rsp_pcp, 0);

		/* PCT Mapping Restricted Request DSCP */
		cass_tc_pct_dscp_pcp_map(hw, dscp_pcp_settings->req_pcp,
					 prof->pct_control_pcp,
					 dscp_pcp_settings->res_req_dscp,
					 dscp_pcp_settings->res_rsp_dscp);

		/* PCT Mapping Unrestricted Request DSCP */
		cass_tc_pct_dscp_pcp_map(hw, dscp_pcp_settings->req_pcp,
					 prof->pct_control_pcp,
					 dscp_pcp_settings->unres_req_dscp,
					 dscp_pcp_settings->unres_rsp_dscp);

		/* HNI Mappings */
		cass_tc_hni_dscp_pcp_map(hw, dscp_pcp_settings->res_req_dscp,
					 dscp_pcp_settings->req_pcp);
		cass_tc_hni_dscp_pcp_map(hw, dscp_pcp_settings->res_rsp_dscp,
					 dscp_pcp_settings->rsp_pcp);
		cass_tc_hni_dscp_pcp_map(hw, dscp_pcp_settings->unres_req_dscp,
					 dscp_pcp_settings->req_pcp);
		cass_tc_hni_dscp_pcp_map(hw, dscp_pcp_settings->unres_rsp_dscp,
					 dscp_pcp_settings->rsp_pcp);

		/* Optionally configure TC "Type" CXI_TC_TYPE_HRP. */
		if (dscp_pcp_settings->hrp_res_req_dscp >= 0) {
			cass_tc_ixe_dscp_pcp_map(hw,
						 dscp_pcp_settings->hrp_res_req_dscp,
						 dscp_pcp_settings->req_pcp,
						 dscp_pcp_settings->res_rsp_dscp);
			cass_tc_pct_dscp_pcp_map(hw, dscp_pcp_settings->req_pcp,
						 prof->pct_control_pcp,
						 dscp_pcp_settings->hrp_res_req_dscp,
						 dscp_pcp_settings->res_rsp_dscp);
			cass_tc_hni_dscp_pcp_map(hw,
						 dscp_pcp_settings->hrp_res_req_dscp,
						 dscp_pcp_settings->req_pcp);
		}

		/* Optionally configure TC "Type" CXI_TC_TYPE_COLL_LEAF. */
		if (dscp_pcp_settings->coll_leaf_res_req_dscp >= 0) {
			cass_tc_ixe_dscp_pcp_map(hw,
						 dscp_pcp_settings->coll_leaf_res_req_dscp,
						 dscp_pcp_settings->req_pcp,
						 dscp_pcp_settings->res_rsp_dscp);
			cass_tc_pct_dscp_pcp_map(hw, dscp_pcp_settings->req_pcp,
						 prof->pct_control_pcp,
						 dscp_pcp_settings->coll_leaf_res_req_dscp,
						 dscp_pcp_settings->res_rsp_dscp);
			cass_tc_hni_dscp_pcp_map(hw,
						 dscp_pcp_settings->coll_leaf_res_req_dscp,
						 dscp_pcp_settings->req_pcp);
		}

		/* Optionally configure TC "Type" CXI_TC_TYPE_RESTRICTED. */
		if (dscp_pcp_settings->restricted_unres_req_dscp >= 0) {
			cass_tc_ixe_dscp_pcp_map(hw,
						 dscp_pcp_settings->restricted_unres_req_dscp,
						 dscp_pcp_settings->req_pcp,
						 dscp_pcp_settings->unres_rsp_dscp);
			cass_tc_pct_dscp_pcp_map(hw, dscp_pcp_settings->req_pcp,
						 prof->pct_control_pcp,
						 dscp_pcp_settings->restricted_unres_req_dscp,
						 dscp_pcp_settings->unres_rsp_dscp);
			cass_tc_hni_dscp_pcp_map(hw,
						 dscp_pcp_settings->restricted_unres_req_dscp,
						 dscp_pcp_settings->req_pcp);
		}

		/* Errata 3176. */
		set_dscp_dfa_mask(hw, dscp_pcp_settings->res_req_dscp, 0xfff);
		set_dscp_dfa_mask(hw, dscp_pcp_settings->hrp_res_req_dscp,
				  0xfff);
		set_dscp_dfa_mask(hw, dscp_pcp_settings->coll_leaf_res_req_dscp,
				  0xfff);
	}
}

static void hrp_enable_mask(struct cass_dev *hw, u64 *dscp_mask, u64 *hrp_mask)
{
	struct cass_tc_cfg *tc_cfg;

	*dscp_mask = *hrp_mask = 0;
	list_for_each_entry(tc_cfg, &hw->tc_list, tc_entry) {
		*dscp_mask |= BIT(tc_cfg->res_req_dscp);
		if (tc_cfg->tc_type == CXI_TC_TYPE_HRP)
			*hrp_mask |= BIT(tc_cfg->res_req_dscp);
	}
}

/**
 * cass_tc_cfg() - Configure a traffic class (TC) definition which can be used
 * to allocate a communication profile.
 *
 * @hw: Cassini device
 * @tc: Traffic Class label
 * @tc_type: Traffic Class type
 *
 * @return: 0 on success. Else negative errno value.
 */
static int cass_tc_cfg(struct cass_dev *hw, unsigned int tc,
		       enum cxi_traffic_class_type tc_type)
{
	struct cass_tc_cfg *tc_cfg;
	unsigned int i;
	unsigned int req_pcp;
	unsigned int res_req_dscp;
	unsigned int res_rsp_dscp;
	unsigned int unres_req_dscp;
	unsigned int unres_rsp_dscp;
	unsigned int rsp_pcp;
	unsigned int ocuset;
	unsigned int cq_tc;
	u64 res_req_dscp_mask;
	struct cxi_tc *cxi_tc = &hw->qos.tcs[tc];
	bool eth_tc = is_eth_tc(tc);
	unsigned int iterations = (tc == CXI_ETH_SHARED) ? PCP_COUNT : 1;
	u64 dscp_mask = 0;
	u64 hrp_mask = 0;
	union c_ixe_cfg_hrp cfg_hrp = {};
	bool hrp = tc_type == CXI_TC_TYPE_HRP;

	if (eth_tc || tc_type == CXI_TC_TYPE_DEFAULT) {
		ocuset = cxi_tc->default_ocuset;
		cq_tc = cxi_tc->default_cq_tc;
	} else {
		ocuset = cxi_tc->restricted_ocuset;
		cq_tc = cxi_tc->restricted_cq_tc;
	}

	/* Common defaults */
	res_req_dscp = cxi_tc->dscp_pcp_settings.res_req_dscp;
	res_rsp_dscp = cxi_tc->dscp_pcp_settings.res_rsp_dscp;
	unres_rsp_dscp = cxi_tc->dscp_pcp_settings.unres_rsp_dscp;
	req_pcp = cxi_tc->dscp_pcp_settings.req_pcp;
	rsp_pcp = cxi_tc->dscp_pcp_settings.rsp_pcp;

	/* Changed various fields based on TC Type */
	if (tc_type == CXI_TC_TYPE_DEFAULT) {
		unres_req_dscp = cxi_tc->dscp_pcp_settings.unres_req_dscp;
	} else if (tc_type == CXI_TC_TYPE_RESTRICTED) {
		unres_req_dscp = cxi_tc->dscp_pcp_settings.restricted_unres_req_dscp;
	} else if (tc_type == CXI_TC_TYPE_HRP) {
		res_req_dscp = cxi_tc->dscp_pcp_settings.hrp_res_req_dscp;
		unres_req_dscp = cxi_tc->dscp_pcp_settings.unres_req_dscp;
	} else if (tc_type == CXI_TC_TYPE_COLL_LEAF) {
		res_req_dscp = cxi_tc->dscp_pcp_settings.coll_leaf_res_req_dscp;
		unres_req_dscp = cxi_tc->dscp_pcp_settings.unres_req_dscp;
	} else {
		return -EINVAL;
	}

	res_req_dscp_mask = BIT(res_req_dscp);
	/* rdscp must be used exclusively with or without HRP */
	if (!eth_tc) {
		hrp_enable_mask(hw, &dscp_mask, &hrp_mask);
		if (res_req_dscp_mask & dscp_mask) {
			if (((hrp && !(res_req_dscp_mask & hrp_mask))) ||
			(!hrp && (res_req_dscp_mask & hrp_mask))) {
				return -EINVAL;
			}
		}
	}

	/* Run once for most TC / TC Type Combinations.
	 * Run multiple times for the Shared Eth Class
	 */
	for (i = 0; i < iterations; i++) {
		/* Skip PCPs not owned by the Shared Eth Class */
		if (tc == CXI_ETH_SHARED &&
		    !cxi_tc->eth_settings.tc_pcps[i])
			continue;
		/* Use PCP field from settings for non-Shared Eth Classes */
		if (eth_tc) {
			if (tc == CXI_ETH_SHARED)
				req_pcp = i;
			else
				req_pcp = cxi_tc->eth_settings.pcp;
		}

		/* All traffic class configs have a unique traffic class label
		 * and traffic class type. Should never be configured twice.
		 */
		list_for_each_entry(tc_cfg, &hw->tc_list, tc_entry) {
			if (eth_tc) {
				if (tc_cfg->tc == tc && tc_cfg->req_pcp == req_pcp &&
				    tc_cfg->tc_type == tc_type)
					return -EEXIST;
			} else {
				if (tc_cfg->tc == tc && tc_cfg->tc_type == tc_type)
					return -EEXIST;
			}
		}

		tc_cfg = kzalloc(sizeof(*tc_cfg), GFP_KERNEL);
		if (!tc_cfg)
			return -ENOMEM;

		list_add_tail(&tc_cfg->tc_entry, &hw->tc_list);

		/* Update mapping for TC Label */
		tc_cfg->tc = tc;
		tc_cfg->tc_type = tc_type;
		tc_cfg->res_req_dscp = res_req_dscp;
		tc_cfg->unres_req_dscp = unres_req_dscp;
		tc_cfg->res_rsp_dscp = res_rsp_dscp;
		tc_cfg->unres_rsp_dscp = unres_rsp_dscp;
		tc_cfg->req_pcp = req_pcp;
		tc_cfg->rsp_pcp = rsp_pcp;
		tc_cfg->ocuset = ocuset;
		tc_cfg->cq_tc = cq_tc;

		/* For C1 Restricted and HRP Labels, use the global restricted
		 * buffer class.
		 * For all other labels, or C2, use the parent TC req_bc
		 */
		if (!cass_version(hw, CASSINI_2) &&
		    (tc_type == CXI_TC_TYPE_RESTRICTED ||
		     tc_type == CXI_TC_TYPE_HRP))
			tc_cfg->req_bc = hw->qos.tc_restricted_oxe_req_bc;
		else
			tc_cfg->req_bc = cxi_tc->req_bc;
	}

	/* Configure IXE to not generate responses to requests from HRP DSCPs */
	if (!eth_tc) {
		hrp_enable_mask(hw, &dscp_mask, &hrp_mask);
		cfg_hrp.hrp_enable = hrp_mask;
		cass_write(hw, C_IXE_CFG_HRP, &cfg_hrp, sizeof(cfg_hrp));
	}

	return 0;
}

/**
 * cass_tc_default_type_cfg() - Configure traffic class type CXI_TC_TYPE_DEFAULT
 * for a given traffic class.
 *
 * Each RDMA traffic class MUST have the CXI_TC_TYPE_DEFAULT type configured. This
 * performs the initial allocation and configuration of required resources which
 * can later be shared with other traffic class types with the same traffic
 * class.
 *
 * @hw: Cassini device
 * @tc: Traffic Class to initialize
 */
static int cass_tc_default_type_cfg(struct cass_dev *hw,
				    enum cxi_traffic_class tc)
{
	struct cxi_tc *cxi_tc = &hw->qos.tcs[tc];
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings =
		&cxi_tc->dscp_pcp_settings;
	unsigned int cq_mcu_base;
	unsigned int cq_mcu_count;
	unsigned int ixe_mcu_base;
	unsigned int ixe_mcu_count;
	unsigned int lpe_mcu_base;
	unsigned int lpe_mcu_count;
	unsigned int tou_mcu;
	int ret;

	cxi_tc->default_ocuset = cass_tc_cq_cfg(hw, tc, &cq_mcu_base,
						&cq_mcu_count,
						&cxi_tc->default_cq_tc,
						false);
	if (cxi_tc->default_ocuset < 0)
		return cxi_tc->default_ocuset;

	tou_mcu = cxi_tc->default_cq_tc + C_OXE_TOU_MCU_START;

	ret = cass_tc_ixe_fq_cfg(hw, tc, dscp_pcp_settings->req_pcp,
				 &ixe_mcu_base, &ixe_mcu_count);
	if (ret)
		return ret;

	ret = cass_tc_lpe_cfg(hw, tc, dscp_pcp_settings->req_pcp,
			      &lpe_mcu_base, &lpe_mcu_count);
	if (ret)
		return ret;

	ret = cass_tc_oxe_cfg(hw, tc, dscp_pcp_settings->req_pcp,
			      dscp_pcp_settings->rsp_pcp,
			      cq_mcu_base, cq_mcu_count, lpe_mcu_base,
			      lpe_mcu_count, ixe_mcu_base, ixe_mcu_count,
			      tou_mcu);
	if (ret)
		return ret;

	cass_tc_pct_cfg(hw, tc, dscp_pcp_settings->req_pcp);


	cass_tc_hni_cfg(hw, tc, dscp_pcp_settings->req_pcp,
			dscp_pcp_settings->rsp_pcp);

	ret = cass_tc_cfg(hw, tc, CXI_TC_TYPE_DEFAULT);

	return ret;
}

/**
 * cass_tc_restricted_cfg() - Configuration of restricted resources needed for
 * CXI_TC_TYPE_RESTRICTED, CXI_TC_TYPE_HRP, and CXI_TC_TYPE_COLL_LEAF traffic
 * class types.
 *
 * If a traffic class has CXI_TC_TYPE_RESTRICTED, CXI_TC_TYPE_HRP, and
 * CXI_TC_TYPE_COLL_LEAF disabled, this restricted resource configuration is a
 * NOOP.
 *
 * @hw: Cassini device
 * @tc: Traffic Class to configure
 */
static int cass_tc_restricted_cfg(struct cass_dev *hw,
				  enum cxi_traffic_class tc)
{
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings =
		&hw->qos.tcs[tc].dscp_pcp_settings;
	const struct cass_tc_oxe_settings *oxe_settings =
		&hw->qos.tcs[tc].oxe_settings;
	int ocuset;
	int ret;
	unsigned int cq_mcu_base;
	unsigned int cq_mcu_count;
	unsigned int cq_tc;
	unsigned int tou_mcu;
	unsigned int restricted_bc;

	/* If PCT control traffic needs to be remapped to a different traffic
	 * class, reuse the response PCP of the remapped traffic class.
	 */
	int mcu_pcp = hw->qos.pct_control_pcp;

	/* NOOP if CXI_TC_TYPE_RESTRICTED, CXI_TC_TYPE_HRP, and
	 * CXI_TC_TYPE_COLL_LEAF are disabled.
	 */
	if (dscp_pcp_settings->restricted_unres_req_dscp < 0 &&
	    dscp_pcp_settings->hrp_res_req_dscp < 0 &&
	    dscp_pcp_settings->coll_leaf_res_req_dscp < 0)
		return 0;

	ocuset = cass_tc_cq_cfg(hw, tc, &cq_mcu_base, &cq_mcu_count, &cq_tc,
				true);
	if (ocuset < 0)
		return ocuset;

	tou_mcu = cq_tc + C_OXE_TOU_MCU_START;

	/* For Restricted TC "Type" use the same BC as the parent TC for C2.
	 * For C1 use the global restricted buffer class.
	 */
	restricted_bc = cass_version(hw, CASSINI_2) ?
			hw->qos.tcs[tc].req_bc :
			hw->qos.tc_restricted_oxe_req_bc;

	ret = cass_tc_oxe_mcu_cfg(hw, cq_mcu_base, cq_mcu_count, mcu_pcp,
				  hw->qos.tcs[tc].leaf[0],
				  restricted_bc,
				  oxe_settings->mfs_index);
	if (ret)
		return ret;

	ret = cass_tc_oxe_mcu_cfg(hw, tou_mcu, 1, mcu_pcp,
				  hw->qos.tcs[tc].leaf[0],
				  restricted_bc,
				  oxe_settings->mfs_index);
	if (ret)
		return ret;

	hw->qos.tcs[tc].restricted_ocuset = ocuset;
	hw->qos.tcs[tc].restricted_cq_tc = cq_tc;

	return 0;
}

/**
 * cass_tc_restricted_type_cfg() - Configure traffic class type
 * CXI_TC_TYPE_RESTRICTED for a given traffic class.
 *
 * This traffic class type requires a unique OCUset. The resulting MCU
 * configuration share the same request traffic shaping and PCP values as the
 * CXI_TC_TYPE_DEFAULT traffic class type configuration for respective traffic
 * class. But, all CXI_TC_TYPE_RESTRICTED traffic class configurations will
 * share the same buffer class regardless of traffic class.
 *
 * In addition, the OCUset configured for CXI_TC_TYPE_RESTRICTED is shared with
 * CXI_TC_TYPE_HRP and CXI_TC_TYPE_COLL_LEAF configurations.
 *
 * @hw: Cassini device
 * @tc: Traffic Class label
 */
static int cass_tc_restricted_type_cfg(struct cass_dev *hw,
				       enum cxi_traffic_class tc)
{
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings =
		&hw->qos.tcs[tc].dscp_pcp_settings;
	int ret;

	if (dscp_pcp_settings->restricted_unres_req_dscp < 0)
		return 0;

	ret = cass_tc_cfg(hw, tc, CXI_TC_TYPE_RESTRICTED);

	return ret;
}

/**
 * cass_tc_hrp_type_cfg() - Configure traffic class type CXI_TC_TYPE_HRP for a
 * given traffic class.
 *
 * This traffic class type requires a unique OCUset. The resulting MCU
 * configuration share the same request traffic shaping and PCP values as the
 * CXI_TC_TYPE_DEFAULT traffic class type configuration for respective traffic
 * class. But, all CXI_TC_TYPE_HRP traffic class configurations will
 * share the same buffer class regardless of traffic class.
 *
 * In addition, the OCUset configured for CXI_TC_TYPE_HRP is shared with
 * CXI_TC_TYPE_RESTRICTED and CXI_TC_TYPE_COLL_LEAF configurations.
 *
 * @hw: Cassini device
 * @tc: Traffic Class label
 */
static int cass_tc_hrp_type_cfg(struct cass_dev *hw, enum cxi_traffic_class tc)
{
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings =
		&hw->qos.tcs[tc].dscp_pcp_settings;
	int ret;

	if (dscp_pcp_settings->hrp_res_req_dscp < 0)
		return 0;

	ret = cass_tc_cfg(hw, tc, CXI_TC_TYPE_HRP);

	return ret;
}

/**
 * cass_tc_coll_leaf_type_cfg() - Configure traffic class type
 * CXI_TC_TYPE_COLL_LEAF for a given traffic class.
 *
 * This traffic class type requires a unique OCUset. The resulting MCU
 * configuration share the same request traffic shaping and PCP values as the
 * CXI_TC_TYPE_DEFAULT traffic class type configuration for respective traffic
 * class. But, all CXI_TC_TYPE_COLL_LEAF traffic class configurations will share
 * the same buffer class regardless of traffic class.
 *
 * In addition, the OCUset configured for CXI_TC_TYPE_COLL_LEAF is shared with
 * CXI_TC_TYPE_RESTRICTED and CXI_TC_TYPE_HRP configurations.
 *
 * @hw: Cassini device
 * @tc: Traffic Class label
 */
static int cass_tc_coll_leaf_type_cfg(struct cass_dev *hw, enum cxi_traffic_class tc)
{
	const struct cass_tc_dscp_pcp_settings *dscp_pcp_settings =
		&hw->qos.tcs[tc].dscp_pcp_settings;
	int ret;

	if (dscp_pcp_settings->coll_leaf_res_req_dscp < 0)
		return 0;

	ret = cass_tc_cfg(hw, tc, CXI_TC_TYPE_COLL_LEAF);

	return ret;
}

static void cass_tc_eth_hni_cfg(struct cass_dev *hw)
{
	const union c_hni_cfg_gen cfg_gen = {
		.default_pcp = hw->qos.untagged_eth_pcp,
	};
	int i;

	cass_write(hw, C_HNI_CFG_GEN, &cfg_gen, sizeof(cfg_gen));

	for (i = 0; i < PCP_COUNT; i++)
		cass_tc_hni_cfg(hw, CXI_TC_ETH, i, 0);
}

static int cass_tc_eth_oxe_cfg(struct cass_dev *hw,
			       enum cxi_eth_traffic_class tc,
			       unsigned int cq_mcu_base,
			       unsigned int cq_mcu_count, unsigned int req_bc)
{
	struct cxi_tc *cxi_tc = &hw->qos.tcs[tc];
	const struct cass_tc_oxe_settings *oxe_settings =
		&cxi_tc->oxe_settings;
	unsigned int leaf = cxi_tc->leaf[0];
	int ret;
	int i;

	cass_tc_oxe_leaf_bucket_cfg(hw, tc, leaf, true);

	ret = cass_tc_oxe_mcu_cfg(hw, cq_mcu_base, cq_mcu_count,
				  hw->qos.untagged_eth_pcp,
				  leaf, req_bc, oxe_settings->mfs_index);
	if (ret)
		return ret;

	/* Configure PCP associated with an Eth TC */
	if (tc != CXI_ETH_SHARED) {
		ret = cass_tc_oxe_pcp_pause_cfg(hw, cxi_tc->eth_settings.pcp,
						leaf);
		if (ret)
			return ret;
	} else {
		for (i = 0; i < PCP_COUNT; i++) {
			/* Configure each PCP in the shared class */
			if (cxi_tc->eth_settings.tc_pcps[i]) {
				ret = cass_tc_oxe_pcp_pause_cfg(hw, i, leaf);
				if (ret)
					return ret;
			}
		}
	}

	return ret;
}

static int cass_tc_eth_init(struct cass_dev *hw)
{
	unsigned int cq_mcu_base;
	unsigned int cq_mcu_count;
	unsigned int shared_cq_mcu_base;
	unsigned int shared_cq_mcu_count;
	int req_bc;
	int ret;
	const struct cass_tc_oxe_settings *oxe_settings =
		&hw->qos.tcs[CXI_TC_ETH].oxe_settings;
	struct cxi_tc *cxi_tc;
	int leaf;
	int tc;
	int i;
	bool pcp_used[PCP_COUNT] =  {};

	/* Get TSC for each Eth TC */
	for (tc = CXI_TC_MAX; tc < CXI_ETH_TC_MAX; tc++) {

		/* Skip TCs that aren't enabled */
		if (!hw->qos.tcs_active[tc])
			continue;

		cxi_tc = &hw->qos.tcs[tc];

		/* Use the head bucket for Ethernet, not a branch. */
		cxi_tc->branch = 7;

		leaf = cass_tc_get_leaf_bucket(hw);
		if (leaf < 0)
			return leaf;

		cxi_tc->leaf[0] = leaf;
	}

	/* Due to buffer class exhaustion, a single buffer class is shared
	 * between all Ethernet OXE configurations.
	 */
	req_bc = cass_tc_eth_oxe_bc_cfg(hw, oxe_settings->spt_rsvd,
					oxe_settings->pbuf_rsvd);
	if (req_bc < 0)
		return req_bc;

	/* Configure a traffic class configuration using a unique communication
	 * profile and OCUset for the untagged Ethernet PCP. This will allow
	 * for all traffic using the untagged Ethernet PCP to have unique
	 * resources allocated.
	 */
	/* Get OCUset for each Eth TC */
	for (tc = CXI_TC_MAX; tc < CXI_ETH_TC_MAX; tc++) {

		/* Skip TCs that aren't enabled */
		if (!hw->qos.tcs_active[tc])
			continue;
		cxi_tc = &hw->qos.tcs[tc];

		cxi_tc->req_bc = req_bc;

		cxi_tc->default_ocuset = cass_tc_cq_cfg(hw, CXI_TC_ETH,
							&cq_mcu_base,
							&cq_mcu_count,
							&cxi_tc->default_cq_tc, true);
		if (cxi_tc->default_ocuset < 0)
			return cxi_tc->default_ocuset;

		/* We will configure the Shared Eth class later */
		if (tc == CXI_ETH_SHARED) {
			shared_cq_mcu_base = cq_mcu_base;
			shared_cq_mcu_count = cq_mcu_count;
			continue;
		}

		/* Store info about each PCP that is used. The catch all
		 * Eth class will be configured to use the rest.
		 */
		pcp_used[cxi_tc->eth_settings.pcp] = true;

		ret = cass_tc_eth_oxe_cfg(hw, tc, cq_mcu_base, cq_mcu_count,
					  req_bc);
		if (ret)
			return ret;

		ret = cass_tc_cfg(hw, tc, CXI_TC_TYPE_DEFAULT);
	}

	/* Configure a catch-all traffic class configuration for all the
	 * remaining Ethernet PCPs. This will allow for functionality of
	 * packets using these PCPs at the cost of sharing resources.
	 * In addition, any pause issued against one of the PCPs within the
	 * catch-all will impact all PCPs.
	 */

	/* Determine which PCPs are "Shared" */
	for (i = 0; i < PCP_COUNT; i++) {
		if (!pcp_used[i])
			hw->qos.tcs[CXI_ETH_SHARED].eth_settings.tc_pcps[i] = true;
	}

	cxi_tc = &hw->qos.tcs[CXI_ETH_SHARED];
	ret = cass_tc_eth_oxe_cfg(hw, CXI_ETH_SHARED,
				  shared_cq_mcu_base, shared_cq_mcu_count,
				  req_bc);
	if (ret)
		return ret;

	ret = cass_tc_cfg(hw, CXI_ETH_SHARED, CXI_TC_TYPE_DEFAULT);
	if (ret)
		return ret;

	cass_tc_eth_hni_cfg(hw);

	return ret;
}

int cass_tc_init(struct cass_dev *hw)
{
	union c_hni_cfg_pause_tx_ctrl tx_ctrl_cfg;
	union c_hni_cfg_pause_rx_ctrl rx_ctrl_cfg;
	int ret;
	int tc;

	if (!active_qos_profile ||
	    active_qos_profile > CXI_QOS_NUM_PROF) {
		pr_err("Invalid active_qos_profile: %d\n",
		       active_qos_profile);
		return -EINVAL;
	}
	pr_info("QoS Profile: %s\n", cxi_qos_strs[active_qos_profile]);

	/* Choose QoS Profile */
	hw->qos	= profiles[active_qos_profile];
	mutex_init(&hw->qos.mutex);
	for (tc = 0; tc < CXI_ETH_TC_MAX; tc++) {
		/* Store TC idx for easy access */
		hw->qos.tcs[tc].tc = tc;
	}

	/* Update QoS Profile Based on Device Generation */
	cxi_qos_calculate_limits(&hw->qos, (cass_version(hw, CASSINI_2)));

	/* Use user provided Untagged Eth PCP if applicable */
	if (untagged_eth_pcp > -1)
		hw->qos.untagged_eth_pcp = untagged_eth_pcp;

	if (pct_control_traffic_pcp < -1 ||
	    pct_control_traffic_pcp > 7) {
		pr_err("Invalid PCT control traffic pcp: %d\n",
		       pct_control_traffic_pcp);
		return -EINVAL;
	}

	if (pct_control_traffic_pcp > -1)
		hw->qos.pct_control_pcp = pct_control_traffic_pcp;

	/* Check if any RDMA TCs are using the PCT PCP */
	for (tc = 0; tc < CXI_TC_MAX; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		if (hw->qos.tcs[tc].dscp_pcp_settings.req_pcp == hw->qos.pct_control_pcp ||
		    hw->qos.tcs[tc].dscp_pcp_settings.rsp_pcp == hw->qos.pct_control_pcp)
			pct_control_traffic_tc = tc;
	}

	INIT_LIST_HEAD(&hw->tc_list);

	cass_clear(hw, C_OXE_CFG_ARB_PCP_MASK(0), C_OXE_CFG_ARB_PCP_MASK_SIZE);

	cass_tc_dscp_pcp_map(hw);

	/* Configure for PFC pauses */
	cass_read(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));
	tx_ctrl_cfg.enable_pfc_pause = 1;
	tx_ctrl_cfg.enable_pause = 0;
	tx_ctrl_cfg.mac_cntl_opcode = 0x0101;
	cass_write(hw, C_HNI_CFG_PAUSE_TX_CTRL, &tx_ctrl_cfg, sizeof(tx_ctrl_cfg));

	cass_read(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg, sizeof(rx_ctrl_cfg));
	rx_ctrl_cfg.pfc_rec_enable = 1;
	rx_ctrl_cfg.pause_rec_enable = 0;
	cass_write(hw, C_HNI_CFG_PAUSE_RX_CTRL, &rx_ctrl_cfg, sizeof(rx_ctrl_cfg));

	/* Setup Global OXE BC for restricted request MCUs for C1 */
	if (!cass_version(hw, CASSINI_2)) {
		ret = cass_tc_init_res_req_oxe_bc(hw);
		if (ret)
			return ret;
	}

	for (tc = 0; tc < CXI_MAX_RDMA_TCS; tc++) {
		/* Don't configure inactive TCs */
		if (!hw->qos.tcs_active[tc])
			continue;

		/* Setup CXI_TC_TYPE_DEFAULT for each RDMA TC */
		ret = cass_tc_default_type_cfg(hw, tc);
		if (ret)
			return ret;

		/* Must be run before CXI_TC_TYPE_RESTRICTED and
		 * CXI_TC_TYPE_HRP configuration.
		 */
		ret = cass_tc_restricted_cfg(hw, tc);
		if (ret)
			return ret;

		ret = cass_tc_restricted_type_cfg(hw, tc);
		if (ret)
			return ret;

		ret = cass_tc_hrp_type_cfg(hw, tc);
		if (ret)
			return ret;

		ret = cass_tc_coll_leaf_type_cfg(hw, tc);
		if (ret)
			return ret;
	}

	/* Set up global PCT Credit limits*/
	cass_tc_pct_global_cfg(hw);

	ret = cass_tc_eth_init(hw);
	if (ret)
		return ret;

	return 0;
}

/* Re-program the ARB CFG CSRs when the link is going up, as they are
 * dependent on the link speed, which may have changed.
 *
 * Matching formula between bandwidth, fill_qty and
 * fill_rate. bandwidth should be slightly above the real
 * value.
 *   FILL_QTY * 16/(FILL_RATE+1) * clock = bandwidth
 *
 * For instance, for Cassini 1, if desired bandwidth is
 * 200Gbps, assuming a FILL_RATE of 9, then FILL_QTY of 16
 * does the job:
 *   16 * 16/10 * 1_000_000_000 * 8 = 204_800_000_000
 *
 * Current fill_qty for Cassini 1 depending on the link speed, and a
 * clock speed of 1GHz:
 *    50: 4  ->  51.2
 *   100: 8  -> 102.4
 *   200: 16 -> 204.8
 *
 * Current fill_qty for Cassini 2 depending on the link speed, and a
 * clock speed of 1.1GHz:
 *    25: 2  ->  28.16
 *    50: 4  ->  56.32
 *   100: 8  -> 112.64
 *   200: 15 -> 211.2
 *   400: 29 -> 408.32
 */
void update_oxe_link_up(struct cass_dev *hw)
{
	struct cxi_link_info link_info;
	int tc;

	cxi_link_mode_get(&hw->cdev, &link_info);

	if (cass_version(hw, CASSINI_1)) {
		switch (link_info.speed) {
		case SPEED_50000:
			hw->tsc_fill_qty = 4;
			break;
		case SPEED_100000:
			hw->tsc_fill_qty = 8;
			break;
		case SPEED_200000:
		default:
			hw->tsc_fill_qty = 16;
			break;
		}
	} else {
		switch (link_info.speed) {
		case SPEED_25000:
			hw->tsc_fill_qty = 2;
			break;
		case SPEED_50000:
			hw->tsc_fill_qty = 4;
			break;
		case SPEED_100000:
			hw->tsc_fill_qty = 8;
			break;
		case SPEED_200000:
			hw->tsc_fill_qty = 15;
			break;
		case SPEED_400000:
		default:
			hw->tsc_fill_qty = 29;
			break;
		}
	}

	traffic_shaping_cfg(hw);

	/* Configure BW for each TC */
	for (tc = 0; tc < CXI_ETH_TC_MAX; tc++) {

		/* CXI_TC_ETH is used for "general" eth config. The specific
		 * Ethernet classes are the ones that actually will have BW
		 * settings.
		 */
		if (tc == CXI_TC_ETH)
			continue;

		if (!hw->qos.tcs_active[tc])
			continue;

		/* Configure Branch specified in TC */
		cass_tc_oxe_branch_bucket_cfg(hw, tc);

		/* Configure leaf for RDMA and Ethernet */
		cass_tc_oxe_leaf_bucket_cfg(hw, tc,
					    hw->qos.tcs[tc].leaf[0], true);

		/* Configure second leaf for RDMA. Not used for Ethernet */
		if (tc < CXI_TC_ETH)
			cass_tc_oxe_leaf_bucket_cfg(hw, tc,
						    hw->qos.tcs[tc].leaf[1],
						    false);
	}
}

/**
 * cass_tc_find() - Look up Traffic Class definition
 *
 * @hw: Cassini device
 * @tc: Traffic Class label
 * @tc_type: Traffic Class type
 * @ethernet_pcp: Ethernet PCP (only valid for CXI_TC_ETH)
 * @tc_cfg: Traffic Class description values
 *
 * @return: 0 on success. -EINVAL if a matching definition was not found.
 */
int cass_tc_find(struct cass_dev *hw, enum cxi_traffic_class tc,
		 enum cxi_traffic_class_type tc_type, u8 ethernet_pcp,
		 struct cass_tc_cfg *tc_cfg)
{
	struct cass_tc_cfg *cfg;

	if (disable_user_pct_tc_usage && pct_control_traffic_tc == tc) {
		pr_err_once("%s reserved for PCT control traffic\n",
			    cxi_tc_strs[tc]);
		return -EINVAL;
	}

	list_for_each_entry(cfg, &hw->tc_list, tc_entry) {
		if (tc == CXI_TC_ETH) {
			if (cfg->tc == tc && cfg->tc_type == tc_type &&
			    cfg->req_pcp == ethernet_pcp) {
				*tc_cfg = *cfg;
				return 0;
			}
		} else {
			if (cfg->tc == tc && cfg->tc_type == tc_type) {
				/* Return config by value */
				*tc_cfg = *cfg;
				return 0;
			}
		}
	}

	return -EINVAL;
}

/**
 * cass_tc_fini() - Finalize Traffic Classes
 *
 * @hw: Cassini device to be finalized
 */
void cass_tc_fini(struct cass_dev *hw)
{
	struct cass_tc_cfg *tc_cfg;

	/* Remove all TC definitions */
	while ((tc_cfg = list_first_entry_or_null(&hw->tc_list,
						  struct cass_tc_cfg,
						  tc_entry))) {
		list_del(&tc_cfg->tc_entry);
		kfree(tc_cfg);
	}
}

/*
 * print out pause state for sysfs diags
 *
 */
int cass_pause_sysfs_sprint(struct cass_dev *hw, char *buf, size_t size)
{
	int rc;
	u32 pause_type;

	spin_lock(&hw->port->pause_lock);
	pause_type = hw->port->pause_type;
	spin_unlock(&hw->port->pause_lock);

	rc = scnprintf(buf, size, "%s", cass_pause_type_str(pause_type));

	return rc;
}

static ssize_t active_qos_profile_show(struct kobject *kobj,
				       struct kobj_attribute *kattr, char *buf)
{
	return scnprintf(buf, PAGE_SIZE, "%s\n",
			 cxi_qos_strs[active_qos_profile]);
}
static struct kobj_attribute active_qos_profile_attribute = __ATTR_RO(active_qos_profile);

static ssize_t untagged_eth_pcp_show(struct kobject *kobj,
				     struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw = container_of(kobj, struct cass_dev, tcs_kobj);

	return scnprintf(buf, PAGE_SIZE, "%d\n",
			 hw->qos.untagged_eth_pcp);
}
static struct kobj_attribute untagged_eth_pcp_attribute = __ATTR_RO(untagged_eth_pcp);

static ssize_t assured_bw_show(struct kobject *kobj,
			       struct kobj_attribute *kattr, char *buf)
{
	const struct cxi_tc *tc_obj;
	const struct cass_tc_oxe_settings *oxe_settings;

	tc_obj = container_of(kobj, struct cxi_tc, kobj);
	oxe_settings = &tc_obj->oxe_settings;

	return scnprintf(buf, PAGE_SIZE, "%d\n",
			 oxe_settings->assured_percent);
}

static ssize_t assured_bw_store(struct kobject *kobj,
				struct kobj_attribute *attr, const char *buf,
				size_t count)
{
	struct cxi_tc *tc_obj;
	struct qos_profile *qos;
	struct cass_tc_oxe_settings *oxe_settings;
	struct cass_dev *hw;
	int ret, assured;
	unsigned int tc;
	unsigned int i;
	unsigned int sum = 0;

	tc_obj = container_of(kobj, struct cxi_tc, kobj);
	tc = tc_obj->tc;
	oxe_settings = &tc_obj->oxe_settings;
	qos = container_of(tc_obj, struct qos_profile, tcs[tc]);
	hw = container_of(qos, struct cass_dev, qos);

	/* Validate input */
	ret = kstrtoint(buf, 10, &assured);
	if (ret)
		return ret;

	/* Input range check */
	if (assured < 0 || assured > 100)
		return -EINVAL;

	/* Compare to ceiling */
	if (assured > oxe_settings->ceiling_percent) {
		pr_err("TC: %s requested assured_bw (%d) cannot be greater than ceiling_bw (%u)\n",
		       cxi_tc_strs_uc[tc], assured,
		       oxe_settings->ceiling_percent);
		return -EINVAL;
	}

	/* Calculate the sum of current BW Percentages */
	for (i = 0; i < CXI_ETH_TC_MAX; i++) {
		/* Not a "real" TC that can have BW */
		if (i == CXI_TC_ETH)
			continue;
		if (!qos->tcs_active[i])
			continue;
		if (i == tc)
			continue;
		sum += qos->tcs[i].oxe_settings.assured_percent;
	}

	/* Ensure requested assured value + sum of other BW Percentages
	 * does not exceed 100
	 */
	sum += assured;
	if (sum > 100) {
		pr_err("Updating assured_bw for TC: %s to %d would exceed total available bandwidth. (calculated sum: %d)",
		       cxi_tc_strs_uc[tc], assured, sum);
		return -EINVAL;
	}

	mutex_lock(&hw->qos.mutex);
	oxe_settings->assured_percent = assured;

	/* Reconfigure branch bucket and leaf buckets for req/rsp */
	cass_tc_oxe_branch_bucket_cfg(hw, tc);
	cass_tc_oxe_leaf_bucket_cfg(hw, tc, hw->qos.tcs[tc].leaf[0], true);
	if (tc < CXI_TC_ETH)
		cass_tc_oxe_leaf_bucket_cfg(hw, tc, hw->qos.tcs[tc].leaf[1],
					    false);
	mutex_unlock(&hw->qos.mutex);

	return count;
}
static struct kobj_attribute assured_bw_attribute = __ATTR_RW(assured_bw);

static ssize_t ceiling_bw_show(struct kobject *kobj,
			       struct kobj_attribute *kattr, char *buf)
{
	const struct cxi_tc *tc_obj;
	const struct cass_tc_oxe_settings *oxe_settings;

	tc_obj = container_of(kobj, struct cxi_tc, kobj);
	oxe_settings = &tc_obj->oxe_settings;

	return scnprintf(buf, PAGE_SIZE, "%d\n",
			 oxe_settings->ceiling_percent);
}

static ssize_t ceiling_bw_store(struct kobject *kobj,
				struct kobj_attribute *attr, const char *buf,
				size_t count)
{
	struct cxi_tc *tc_obj;
	struct qos_profile *qos;
	struct cass_tc_oxe_settings *oxe_settings;
	struct cass_dev *hw;
	int ret, ceiling;
	unsigned int tc;

	tc_obj = container_of(kobj, struct cxi_tc, kobj);
	tc = tc_obj->tc;
	oxe_settings = &tc_obj->oxe_settings;
	qos = container_of(tc_obj, struct qos_profile, tcs[tc]);
	hw = container_of(qos, struct cass_dev, qos);

	/* Validate input */
	ret = kstrtoint(buf, 10, &ceiling);
	if (ret)
		return ret;

	if (ceiling < 0 || ceiling > 100)
		return -EINVAL;

	/* Compare to assured bw */
	if (ceiling < oxe_settings->assured_percent) {
		pr_err("TC: %s requested ceiling_bw (%d) cannot be lower than assured_bw (%u)\n",
		       cxi_tc_strs_uc[tc], ceiling,
		       oxe_settings->assured_percent);
		return -EINVAL;
	}

	mutex_lock(&hw->qos.mutex);
	oxe_settings->ceiling_percent = ceiling;

	/* Reconfigure branch bucket and leaf buckets for req/rsp */
	cass_tc_oxe_branch_bucket_cfg(hw, tc);
	cass_tc_oxe_leaf_bucket_cfg(hw, tc, hw->qos.tcs[tc].leaf[0], true);
	if (tc < CXI_TC_ETH)
		cass_tc_oxe_leaf_bucket_cfg(hw, tc, hw->qos.tcs[tc].leaf[1],
					    false);
	mutex_unlock(&hw->qos.mutex);

	return count;
}
static struct kobj_attribute ceiling_bw_attribute = __ATTR_RW(ceiling_bw);

static struct attribute *tc_attrs[] = {
	&assured_bw_attribute.attr,
	&ceiling_bw_attribute.attr,
	NULL,
};
ATTRIBUTE_GROUPS(tc);

static struct kobj_type tc_settings = {
	.sysfs_ops	= &kobj_sysfs_ops,
	.default_groups = tc_groups,
};

static struct kobj_type tc_dir = {
	.sysfs_ops	= &kobj_sysfs_ops,
};

static struct attribute *tc_top_attrs[] = {
	&active_qos_profile_attribute.attr,
	&untagged_eth_pcp_attribute.attr,
	NULL,
};
ATTRIBUTE_GROUPS(tc_top);

static struct kobj_type tc_top_info = {
	.sysfs_ops = &kobj_sysfs_ops,
	.default_groups = tc_top_groups,
};

int cass_create_tc_sysfs(struct cass_dev *hw)
{
	int tc, rc, bad_tc;

	/* Create top level directory and files */
	rc = kobject_init_and_add(&hw->tcs_kobj, &tc_top_info,
				  &hw->cdev.pdev->dev.kobj,
				  "traffic_classes");
	if (rc)
		goto destroy;

	/* Create directory for each RDMA TC */
	for (tc = 0; tc < CXI_TC_ETH; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		rc = kobject_init_and_add(&hw->qos.tcs[tc].kobj,
					  &tc_settings, &hw->tcs_kobj,
					  "%s", cxi_tc_strs_lc[tc]);
		if (rc)
			goto put_rdma_tcs;
	}

	/* Create top level directory for ETH */
	rc = kobject_init_and_add(&hw->qos.tcs[CXI_TC_ETH].kobj,
				  &tc_dir, &hw->tcs_kobj,
				  "%s", "ethernet");
	if (rc) {
		bad_tc = -1;
		goto put_rdma_tcs;
	}

	/* Create directory for each Eth TC */
	for (tc = CXI_TC_MAX; tc < CXI_ETH_TC_MAX; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		rc = kobject_init_and_add(&hw->qos.tcs[tc].kobj,
					  &tc_settings, &hw->qos.tcs[CXI_TC_ETH].kobj,
					  "%s", cxi_tc_strs_lc[tc]);
		if (rc)
			goto put_eth_tcs;
	}

	return 0;

put_eth_tcs:
	bad_tc = tc;
	for (tc = CXI_TC_MAX; tc < CXI_ETH_TC_MAX; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;

		kobject_put(&hw->qos.tcs[tc].kobj);

		if (tc == bad_tc)
			break;
	}
	bad_tc = -1;
put_rdma_tcs:
	bad_tc = tc;
	for (tc = 0; tc < CXI_MAX_RDMA_TCS; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;

		kobject_put(&hw->qos.tcs[tc].kobj);

		if (tc == bad_tc)
			break;
	}
destroy:
	kobject_put(&hw->tcs_kobj);
	return rc;
}

void cass_destroy_tc_sysfs(struct cass_dev *hw)
{
	int tc;

	for (tc = CXI_TC_MAX; tc < CXI_ETH_TC_MAX; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		kobject_put(&hw->qos.tcs[tc].kobj);
	}

	kobject_put(&hw->qos.tcs[CXI_TC_ETH].kobj);

	for (tc = 0; tc < CXI_TC_ETH; tc++) {
		if (!hw->qos.tcs_active[tc])
			continue;
		kobject_put(&hw->qos.tcs[tc].kobj);
	}
	sysfs_remove_file(&hw->tcs_kobj, &active_qos_profile_attribute.attr);
	kobject_put(&hw->tcs_kobj);
}

int tc_cfg_show(struct seq_file *s, void *unused)
{
	struct cass_dev *hw = s->private;
	struct cass_tc_cfg *tc_cfg;
	const char *label;

	seq_printf(s, "Active QoS Profile: %s\n\n",
		   cxi_qos_strs[active_qos_profile]);
	seq_printf(s, "%18s%18s%15s%15s%15s%15s%8s%8s%7s%7s\n", "LABEL", "TYPE",
		   "RES_REQ_DSCP", "UNRES_REQ_DSCP", "RES_RSP_DSCP",
		   "UNRES_RSP_DSCP", "REQ_PCP", "RSP_PCP", "OCUSET", "BC");

	list_for_each_entry(tc_cfg, &hw->tc_list, tc_entry) {
		if (tc_cfg->tc < 0 || tc_cfg->tc > CXI_ETH_TC_MAX)
			label = "(invalid)";
		else
			label = cxi_tc_strs_uc[tc_cfg->tc];
		seq_printf(s, "%18s%18s%15d%15d%15d%15d%8d%8d%7d%7d\n",
			   label, cxi_tc_type_to_str(tc_cfg->tc_type),
			   tc_cfg->res_req_dscp, tc_cfg->unres_req_dscp,
			   tc_cfg->res_rsp_dscp, tc_cfg->unres_rsp_dscp,
			   tc_cfg->req_pcp, tc_cfg->rsp_pcp, tc_cfg->ocuset,
			   tc_cfg->req_bc);
	}
	seq_printf(s, "%18s%18s%15d%15d%15d%15d%8d%8d%7d%7d\n",
		   "PCT", "N/A", -1, -1, -1, -1,
		   hw->qos.pct_control_pcp, hw->qos.pct_control_pcp, -1, -1);

	return 0;
}
