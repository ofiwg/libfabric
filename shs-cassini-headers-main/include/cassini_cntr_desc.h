// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
* Copyright 2018-2021,2024 Hewlett Packard Enterprise Development LP
*
* This file is for temporary backwards compatibility with the libcxi API
* for cassini telemetry/counters.  That backwards compatibility will be
* removed in a future release.	This file was previously generated and
* released as part of "cassini-headers".
*/

#ifndef __CASSINI_CNTR_DESC_H
#define __CASSINI_CNTR_DESC_H

#include "cassini_cntr_defs.h"

#ifndef __cplusplus
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverride-init"

static const struct {
	const char *name;
} c1_cntr_descs[C1_CNTR_SIZE] = {
	[0 ... (C1_CNTR_SIZE - 1)] = {
		.name = NULL
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_TLPS] = {
		.name = "pi_ipd_pri_rtrgt_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_MWR_TLPS] = {
		.name = "pi_ipd_pri_rtrgt_mwr_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_MRD_TLPS] = {
		.name = "pi_ipd_pri_rtrgt_mrd_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_MSG_TLPS] = {
		.name = "pi_ipd_pri_rtrgt_msg_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_MSGD_TLPS] = {
		.name = "pi_ipd_pri_rtrgt_msgd_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_TLPS_ABORTED] = {
		.name = "pi_ipd_pri_rtrgt_tlps_aborted"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_BLOCKED_ON_MB] = {
		.name = "pi_ipd_pri_rtrgt_blocked_on_mb"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_BLOCKED_ON_RARB] = {
		.name = "pi_ipd_pri_rtrgt_blocked_on_rarb"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_HDR_PARITY_ERRORS] = {
		.name = "pi_ipd_pri_rtrgt_hdr_parity_errors"
	},
	[C_CNTR_PI_IPD_PRI_RTRGT_DATA_PARITY_ERRORS] = {
		.name = "pi_ipd_pri_rtrgt_data_parity_errors"
	},
	[C_CNTR_PI_IPD_PRI_RBYP_TLPS] = {
		.name = "pi_ipd_pri_rbyp_tlps"
	},
	[C_CNTR_PI_IPD_PRI_RBYP_TLPS_ABORTED] = {
		.name = "pi_ipd_pri_rbyp_tlps_aborted"
	},
	[C_CNTR_PI_IPD_PRI_RBYP_HDR_PARITY_ERRORS] = {
		.name = "pi_ipd_pri_rbyp_hdr_parity_errors"
	},
	[C_CNTR_PI_IPD_PRI_RBYP_DATA_PARITY_ERRORS] = {
		.name = "pi_ipd_pri_rbyp_data_parity_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_XALI_POSTED_TLPS] = {
		.name = "pi_ipd_pti_tarb_xali_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_TARB_XALI_NON_POSTED_TLPS] = {
		.name = "pi_ipd_pti_tarb_xali_non_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_MSIXC_XALI_POSTED_TLPS] = {
		.name = "pi_ipd_pti_msixc_xali_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_DBG_XALI_POSTED_TLPS] = {
		.name = "pi_ipd_pti_dbg_xali_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_DBG_XALI_NON_POSTED_TLPS] = {
		.name = "pi_ipd_pti_dbg_xali_non_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_DMAC_XALI_POSTED_TLPS] = {
		.name = "pi_ipd_pti_dmac_xali_posted_tlps"
	},
	[C_CNTR_PI_IPD_PTI_TARB_P_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_p_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_P_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_p_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_NP_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_np_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_NP_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_np_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_PH_CRD] = {
		.name = "pi_ipd_pti_tarb_blocked_on_ph_crd"
	},
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_PD_CRD] = {
		.name = "pi_ipd_pti_tarb_blocked_on_pd_crd"
	},
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_NPH_CRD] = {
		.name = "pi_ipd_pti_tarb_blocked_on_nph_crd"
	},
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_NPD_CRD] = {
		.name = "pi_ipd_pti_tarb_blocked_on_npd_crd"
	},
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_TAG] = {
		.name = "pi_ipd_pti_tarb_blocked_on_tag"
	},
	[C_CNTR_PI_IPD_PTI_DMAC_BLOCKED_ON_PH_CRD] = {
		.name = "pi_ipd_pti_dmac_blocked_on_ph_crd"
	},
	[C_CNTR_PI_IPD_PTI_DMAC_BLOCKED_ON_PD_CRD] = {
		.name = "pi_ipd_pti_dmac_blocked_on_pd_crd"
	},
	[C_CNTR_PI_IPD_PTI_MSIXC_BLOCKED_ON_PH_CRD] = {
		.name = "pi_ipd_pti_msixc_blocked_on_ph_crd"
	},
	[C_CNTR_PI_IPD_PTI_MSIXC_BLOCKED_ON_PD_CRD] = {
		.name = "pi_ipd_pti_msixc_blocked_on_pd_crd"
	},
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_PH_CRD] = {
		.name = "pi_ipd_pti_dbg_blocked_on_ph_crd"
	},
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_PD_CRD] = {
		.name = "pi_ipd_pti_dbg_blocked_on_pd_crd"
	},
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_NPH_CRD] = {
		.name = "pi_ipd_pti_dbg_blocked_on_nph_crd"
	},
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_TAG] = {
		.name = "pi_ipd_pti_dbg_blocked_on_tag"
	},
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO] = {
		.name = "pi_ipd_pti_tarb_cmpl_to"
	},
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_cmpl_to_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_pti_tarb_cmpl_to_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQS_SENT] = {
		.name = "pi_ipd_msixc_oob_irqs_sent"
	},
	[C_CNTR_PI_IPD_MSIXC_OOB_LEGACY_IRQS_SENT] = {
		.name = "pi_ipd_msixc_oob_legacy_irqs_sent"
	},
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQS_DISCARDED] = {
		.name = "pi_ipd_msixc_oob_irqs_discarded"
	},
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQ_PBAS] = {
		.name = "pi_ipd_msixc_oob_irq_pbas"
	},
	[C_CNTR_PI_IPD_MSIXC_IB_IRQS_SENT] = {
		.name = "pi_ipd_msixc_ib_irqs_sent"
	},
	[C_CNTR_PI_IPD_MSIXC_IB_LEGACY_IRQS_SENT] = {
		.name = "pi_ipd_msixc_ib_legacy_irqs_sent"
	},
	[C_CNTR_PI_IPD_MSIXC_IB_IRQS_DISCARDED] = {
		.name = "pi_ipd_msixc_ib_irqs_discarded"
	},
	[C_CNTR_PI_IPD_MSIXC_IB_IRQ_PBAS] = {
		.name = "pi_ipd_msixc_ib_irq_pbas"
	},
	[C_CNTR_PI_IPD_MSIXC_PBA_COR_ERRORS] = {
		.name = "pi_ipd_msixc_pba_cor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_PBA_UCOR_ERRORS] = {
		.name = "pi_ipd_msixc_pba_ucor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_TABLE_COR_ERRORS] = {
		.name = "pi_ipd_msixc_table_cor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_TABLE_UCOR_ERRORS] = {
		.name = "pi_ipd_msixc_table_ucor_errors"
	},
	[C_CNTR_PI_IPD_PTI_MSIXC_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_pti_msixc_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_PTI_MSIXC_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_pti_msixc_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_PTI_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_msixc_pti_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_MSIXC_PTI_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_msixc_pti_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_DMAC_P_FIFO_COR_ERRORS] = {
		.name = "pi_ipd_dmac_p_fifo_cor_errors"
	},
	[C_CNTR_PI_IPD_DMAC_P_FIFO_UCOR_ERRORS] = {
		.name = "pi_ipd_dmac_p_fifo_ucor_errors"
	},
	[C_CNTR_PI_IPD_DBIC_DBI_REQUESTS] = {
		.name = "pi_ipd_dbic_dbi_requests"
	},
	[C_CNTR_PI_IPD_DBIC_DBI_ACKS] = {
		.name = "pi_ipd_dbic_dbi_acks"
	},
	[C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_0] = {
		.name = "pi_ipd_ipd_trigger_events_0"
	},
	[C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1] = {
		.name = "pi_ipd_ipd_trigger_events_1"
	},
	[C_CNTR_MB_CRMC_RING_SBE_0] = {
		.name = "mb_crmc_ring_sbe_0"
	},
	[C_CNTR_MB_CRMC_RING_SBE_1] = {
		.name = "mb_crmc_ring_sbe_1"
	},
	[C_CNTR_MB_CRMC_RING_SBE_2] = {
		.name = "mb_crmc_ring_sbe_2"
	},
	[C_CNTR_MB_CRMC_RING_MBE_0] = {
		.name = "mb_crmc_ring_mbe_0"
	},
	[C_CNTR_MB_CRMC_RING_MBE_1] = {
		.name = "mb_crmc_ring_mbe_1"
	},
	[C_CNTR_MB_CRMC_RING_MBE_2] = {
		.name = "mb_crmc_ring_mbe_2"
	},
	[C_CNTR_MB_CRMC_WR_ERROR_0] = {
		.name = "mb_crmc_wr_error_0"
	},
	[C_CNTR_MB_CRMC_WR_ERROR_1] = {
		.name = "mb_crmc_wr_error_1"
	},
	[C_CNTR_MB_CRMC_WR_ERROR_2] = {
		.name = "mb_crmc_wr_error_2"
	},
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_0] = {
		.name = "mb_crmc_axi_wr_requests_0"
	},
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_1] = {
		.name = "mb_crmc_axi_wr_requests_1"
	},
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_2] = {
		.name = "mb_crmc_axi_wr_requests_2"
	},
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_0] = {
		.name = "mb_crmc_ring_wr_requests_0"
	},
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_1] = {
		.name = "mb_crmc_ring_wr_requests_1"
	},
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_2] = {
		.name = "mb_crmc_ring_wr_requests_2"
	},
	[C_CNTR_MB_CRMC_RD_ERROR_0] = {
		.name = "mb_crmc_rd_error_0"
	},
	[C_CNTR_MB_CRMC_RD_ERROR_1] = {
		.name = "mb_crmc_rd_error_1"
	},
	[C_CNTR_MB_CRMC_RD_ERROR_2] = {
		.name = "mb_crmc_rd_error_2"
	},
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_0] = {
		.name = "mb_crmc_axi_rd_requests_0"
	},
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_1] = {
		.name = "mb_crmc_axi_rd_requests_1"
	},
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_2] = {
		.name = "mb_crmc_axi_rd_requests_2"
	},
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_0] = {
		.name = "mb_crmc_ring_rd_requests_0"
	},
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_1] = {
		.name = "mb_crmc_ring_rd_requests_1"
	},
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_2] = {
		.name = "mb_crmc_ring_rd_requests_2"
	},
	[C_CNTR_MB_JAI_AXI_WR_REQUESTS] = {
		.name = "mb_jai_axi_wr_requests"
	},
	[C_CNTR_MB_JAI_AXI_RD_REQUESTS] = {
		.name = "mb_jai_axi_rd_requests"
	},
	[C_CNTR_MB_MB_LSA0_TRIGGER_EVENTS_0] = {
		.name = "mb_mb_lsa0_trigger_events_0"
	},
	[C_CNTR_MB_MB_LSA0_TRIGGER_EVENTS_1] = {
		.name = "mb_mb_lsa0_trigger_events_1"
	},
	[C_CNTR_MB_IXE_LSA0_TRIGGER_EVENTS_0] = {
		.name = "mb_ixe_lsa0_trigger_events_0"
	},
	[C_CNTR_MB_IXE_LSA0_TRIGGER_EVENTS_1] = {
		.name = "mb_ixe_lsa0_trigger_events_1"
	},
	[C_CNTR_MB_CQ_LSA0_TRIGGER_EVENTS_0] = {
		.name = "mb_cq_lsa0_trigger_events_0"
	},
	[C_CNTR_MB_CQ_LSA0_TRIGGER_EVENTS_1] = {
		.name = "mb_cq_lsa0_trigger_events_1"
	},
	[C_CNTR_MB_EE_LSA0_TRIGGER_EVENTS_0] = {
		.name = "mb_ee_lsa0_trigger_events_0"
	},
	[C_CNTR_MB_EE_LSA0_TRIGGER_EVENTS_1] = {
		.name = "mb_ee_lsa0_trigger_events_1"
	},
	[C_CNTR_MB_MB_LSA1_TRIGGER_EVENTS_0] = {
		.name = "mb_mb_lsa1_trigger_events_0"
	},
	[C_CNTR_MB_MB_LSA1_TRIGGER_EVENTS_1] = {
		.name = "mb_mb_lsa1_trigger_events_1"
	},
	[C_CNTR_MB_IXE_LSA1_TRIGGER_EVENTS_0] = {
		.name = "mb_ixe_lsa1_trigger_events_0"
	},
	[C_CNTR_MB_IXE_LSA1_TRIGGER_EVENTS_1] = {
		.name = "mb_ixe_lsa1_trigger_events_1"
	},
	[C_CNTR_MB_CQ_LSA1_TRIGGER_EVENTS_0] = {
		.name = "mb_cq_lsa1_trigger_events_0"
	},
	[C_CNTR_MB_CQ_LSA1_TRIGGER_EVENTS_1] = {
		.name = "mb_cq_lsa1_trigger_events_1"
	},
	[C_CNTR_MB_EE_LSA1_TRIGGER_EVENTS_0] = {
		.name = "mb_ee_lsa1_trigger_events_0"
	},
	[C_CNTR_MB_EE_LSA1_TRIGGER_EVENTS_1] = {
		.name = "mb_ee_lsa1_trigger_events_1"
	},
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_0] = {
		.name = "mb_cmc_axi_wr_requests_0"
	},
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_1] = {
		.name = "mb_cmc_axi_wr_requests_1"
	},
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_2] = {
		.name = "mb_cmc_axi_wr_requests_2"
	},
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_3] = {
		.name = "mb_cmc_axi_wr_requests_3"
	},
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_0] = {
		.name = "mb_cmc_axi_rd_requests_0"
	},
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_1] = {
		.name = "mb_cmc_axi_rd_requests_1"
	},
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_2] = {
		.name = "mb_cmc_axi_rd_requests_2"
	},
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_3] = {
		.name = "mb_cmc_axi_rd_requests_3"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_TLPS] = {
		.name = "pi_pri_mb_rtrgt_tlps"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_MWR_TLPS] = {
		.name = "pi_pri_mb_rtrgt_mwr_tlps"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_MRD_TLPS] = {
		.name = "pi_pri_mb_rtrgt_mrd_tlps"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_TLP_DISCARDS] = {
		.name = "pi_pri_mb_rtrgt_tlp_discards"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_POSTED_TLP_DISCARDS] = {
		.name = "pi_pri_mb_rtrgt_posted_tlp_discards"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_NON_POSTED_TLP_DISCARDS] = {
		.name = "pi_pri_mb_rtrgt_non_posted_tlp_discards"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_POSTED_TLP_PARTIAL_DISCARDS] = {
		.name = "pi_pri_mb_rtrgt_posted_tlp_partial_discards"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_FIFO_COR_ERRORS] = {
		.name = "pi_pri_mb_rtrgt_fifo_cor_errors"
	},
	[C_CNTR_PI_PRI_MB_RTRGT_FIFO_UCOR_ERRORS] = {
		.name = "pi_pri_mb_rtrgt_fifo_ucor_errors"
	},
	[C_CNTR_PI_PRI_MB_XALI_CMPL_TLPS] = {
		.name = "pi_pri_mb_xali_cmpl_tlps"
	},
	[C_CNTR_PI_PRI_MB_XALI_CMPL_CA_TLPS] = {
		.name = "pi_pri_mb_xali_cmpl_ca_tlps"
	},
	[C_CNTR_PI_PRI_MB_XALI_CMPL_UR_TLPS] = {
		.name = "pi_pri_mb_xali_cmpl_ur_tlps"
	},
	[C_CNTR_PI_PRI_MB_AXI_WR_REQUESTS] = {
		.name = "pi_pri_mb_axi_wr_requests"
	},
	[C_CNTR_PI_PRI_MB_AXI_RD_REQUESTS] = {
		.name = "pi_pri_mb_axi_rd_requests"
	},
	[C_CNTR_PI_PRI_MB_AXI_WR_RSP_DECERRS] = {
		.name = "pi_pri_mb_axi_wr_rsp_decerrs"
	},
	[C_CNTR_PI_PRI_MB_AXI_WR_RSP_SLVERRS] = {
		.name = "pi_pri_mb_axi_wr_rsp_slverrs"
	},
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_DECERRS] = {
		.name = "pi_pri_mb_axi_rd_rsp_decerrs"
	},
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_SLVERRS] = {
		.name = "pi_pri_mb_axi_rd_rsp_slverrs"
	},
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_PARERRS] = {
		.name = "pi_pri_mb_axi_rd_rsp_parerrs"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_TLPS] = {
		.name = "pi_pri_rarb_rtrgt_tlps"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_MWR_TLPS] = {
		.name = "pi_pri_rarb_rtrgt_mwr_tlps"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_MSG_TLPS] = {
		.name = "pi_pri_rarb_rtrgt_msg_tlps"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_MSGD_TLPS] = {
		.name = "pi_pri_rarb_rtrgt_msgd_tlps"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_TLP_DISCARDS] = {
		.name = "pi_pri_rarb_rtrgt_tlp_discards"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_FIFO_COR_ERRORS] = {
		.name = "pi_pri_rarb_rtrgt_fifo_cor_errors"
	},
	[C_CNTR_PI_PRI_RARB_RTRGT_FIFO_UCOR_ERRORS] = {
		.name = "pi_pri_rarb_rtrgt_fifo_ucor_errors"
	},
	[C_CNTR_PI_PRI_RARB_RBYP_TLPS] = {
		.name = "pi_pri_rarb_rbyp_tlps"
	},
	[C_CNTR_PI_PRI_RARB_RBYP_TLP_DISCARDS] = {
		.name = "pi_pri_rarb_rbyp_tlp_discards"
	},
	[C_CNTR_PI_PRI_RARB_RBYP_FIFO_COR_ERRORS] = {
		.name = "pi_pri_rarb_rbyp_fifo_cor_errors"
	},
	[C_CNTR_PI_PRI_RARB_RBYP_FIFO_UCOR_ERRORS] = {
		.name = "pi_pri_rarb_rbyp_fifo_ucor_errors"
	},
	[C_CNTR_PI_PRI_RARB_SRIOVT_COR_ERRORS] = {
		.name = "pi_pri_rarb_sriovt_cor_errors"
	},
	[C_CNTR_PI_PRI_RARB_SRIOVT_UCOR_ERRORS] = {
		.name = "pi_pri_rarb_sriovt_ucor_errors"
	},
	[C_CNTR_PI_PTI_TARB_PKTS] = {
		.name = "pi_pti_tarb_pkts"
	},
	[C_CNTR_PI_PTI_TARB_PKT_DISCARDS] = {
		.name = "pi_pti_tarb_pkt_discards"
	},
	[C_CNTR_PI_PTI_TARB_RARB_CA_PKTS] = {
		.name = "pi_pti_tarb_rarb_ca_pkts"
	},
	[C_CNTR_PI_PTI_TARB_MWR_PKTS] = {
		.name = "pi_pti_tarb_mwr_pkts"
	},
	[C_CNTR_PI_PTI_TARB_MRD_PKTS] = {
		.name = "pi_pti_tarb_mrd_pkts"
	},
	[C_CNTR_PI_PTI_TARB_FLUSH_PKTS] = {
		.name = "pi_pti_tarb_flush_pkts"
	},
	[C_CNTR_PI_PTI_TARB_IRQ_PKTS] = {
		.name = "pi_pti_tarb_irq_pkts"
	},
	[C_CNTR_PI_PTI_TARB_TRANSLATION_PKTS] = {
		.name = "pi_pti_tarb_translation_pkts"
	},
	[C_CNTR_PI_PTI_TARB_INV_RSP_PKTS] = {
		.name = "pi_pti_tarb_inv_rsp_pkts"
	},
	[C_CNTR_PI_PTI_TARB_PG_REQ_PKTS] = {
		.name = "pi_pti_tarb_pg_req_pkts"
	},
	[C_CNTR_PI_PTI_TARB_AMO_PKTS] = {
		.name = "pi_pti_tarb_amo_pkts"
	},
	[C_CNTR_PI_PTI_TARB_FETCHING_AMO_PKTS] = {
		.name = "pi_pti_tarb_fetching_amo_pkts"
	},
	[C_CNTR_PI_PTI_TARB_HDR_COR_ERRORS] = {
		.name = "pi_pti_tarb_hdr_cor_errors"
	},
	[C_CNTR_PI_PTI_TARB_HDR_UCOR_ERRORS] = {
		.name = "pi_pti_tarb_hdr_ucor_errors"
	},
	[C_CNTR_PI_PTI_TARB_DATA_COR_ERRORS] = {
		.name = "pi_pti_tarb_data_cor_errors"
	},
	[C_CNTR_PI_PTI_TARB_DATA_UCOR_ERROR] = {
		.name = "pi_pti_tarb_data_ucor_error"
	},
	[C_CNTR_PI_PTI_TARB_ACXT_COR_ERRORS] = {
		.name = "pi_pti_tarb_acxt_cor_errors"
	},
	[C_CNTR_PI_PTI_TARB_ACXT_UCOR_ERRORS] = {
		.name = "pi_pti_tarb_acxt_ucor_errors"
	},
	[C_CNTR_PI_DMAC_AXI_RD_REQUESTS] = {
		.name = "pi_dmac_axi_rd_requests"
	},
	[C_CNTR_PI_DMAC_AXI_RD_RSP_DECERRS] = {
		.name = "pi_dmac_axi_rd_rsp_decerrs"
	},
	[C_CNTR_PI_DMAC_AXI_RD_RSP_SLVERRS] = {
		.name = "pi_dmac_axi_rd_rsp_slverrs"
	},
	[C_CNTR_PI_DMAC_AXI_RD_RSP_PARERRS] = {
		.name = "pi_dmac_axi_rd_rsp_parerrs"
	},
	[C_CNTR_PI_DMAC_PAYLOAD_MWR_TLPS] = {
		.name = "pi_dmac_payload_mwr_tlps"
	},
	[C_CNTR_PI_DMAC_CE_MWR_TLPS] = {
		.name = "pi_dmac_ce_mwr_tlps"
	},
	[C_CNTR_PI_DMAC_IRQS] = {
		.name = "pi_dmac_irqs"
	},
	[C_CNTR_PI_DMAC_DESC_COR_ERRORS] = {
		.name = "pi_dmac_desc_cor_errors"
	},
	[C_CNTR_PI_DMAC_DESC_UCOR_ERRORS] = {
		.name = "pi_dmac_desc_ucor_errors"
	},
	[C_CNTR_ATU_MEM_COR_ERR_CNTR] = {
		.name = "atu_mem_cor_err_cntr"
	},
	[C_CNTR_ATU_MEM_UCOR_ERR_CNTR] = {
		.name = "atu_mem_ucor_err_cntr"
	},
	[C_CNTR_ATU_CACHE_MISS_0] = {
		.name = "atu_cache_miss_0"
	},
	[C_CNTR_ATU_CACHE_MISS_1] = {
		.name = "atu_cache_miss_1"
	},
	[C_CNTR_ATU_CACHE_MISS_2] = {
		.name = "atu_cache_miss_2"
	},
	[C_CNTR_ATU_CACHE_MISS_3] = {
		.name = "atu_cache_miss_3"
	},
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_0] = {
		.name = "atu_cache_hit_base_page_size_0"
	},
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_1] = {
		.name = "atu_cache_hit_base_page_size_1"
	},
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_2] = {
		.name = "atu_cache_hit_base_page_size_2"
	},
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_3] = {
		.name = "atu_cache_hit_base_page_size_3"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_0] = {
		.name = "atu_cache_hit_derivative1_page_size_0"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_1] = {
		.name = "atu_cache_hit_derivative1_page_size_1"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_2] = {
		.name = "atu_cache_hit_derivative1_page_size_2"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_3] = {
		.name = "atu_cache_hit_derivative1_page_size_3"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_0] = {
		.name = "atu_cache_hit_derivative2_page_size_0"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_1] = {
		.name = "atu_cache_hit_derivative2_page_size_1"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_2] = {
		.name = "atu_cache_hit_derivative2_page_size_2"
	},
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_3] = {
		.name = "atu_cache_hit_derivative2_page_size_3"
	},
	[C_CNTR_ATU_CACHE_MISS_OXE] = {
		.name = "atu_cache_miss_oxe"
	},
	[C_CNTR_ATU_CACHE_MISS_IXE] = {
		.name = "atu_cache_miss_ixe"
	},
	[C_CNTR_ATU_CACHE_MISS_EE] = {
		.name = "atu_cache_miss_ee"
	},
	[C_CNTR_ATU_ATS_TRANS_LATENCY_0] = {
		.name = "atu_ats_trans_latency_0"
	},
	[C_CNTR_ATU_ATS_TRANS_LATENCY_1] = {
		.name = "atu_ats_trans_latency_1"
	},
	[C_CNTR_ATU_ATS_TRANS_LATENCY_2] = {
		.name = "atu_ats_trans_latency_2"
	},
	[C_CNTR_ATU_ATS_TRANS_LATENCY_3] = {
		.name = "atu_ats_trans_latency_3"
	},
	[C_CNTR_ATU_NTA_TRANS_LATENCY_0] = {
		.name = "atu_nta_trans_latency_0"
	},
	[C_CNTR_ATU_NTA_TRANS_LATENCY_1] = {
		.name = "atu_nta_trans_latency_1"
	},
	[C_CNTR_ATU_NTA_TRANS_LATENCY_2] = {
		.name = "atu_nta_trans_latency_2"
	},
	[C_CNTR_ATU_NTA_TRANS_LATENCY_3] = {
		.name = "atu_nta_trans_latency_3"
	},
	[C_CNTR_ATU_CLIENT_REQ_OXE] = {
		.name = "atu_client_req_oxe"
	},
	[C_CNTR_ATU_CLIENT_REQ_IXE] = {
		.name = "atu_client_req_ixe"
	},
	[C_CNTR_ATU_CLIENT_REQ_EE] = {
		.name = "atu_client_req_ee"
	},
	[C_CNTR_ATU_FILTERED_REQUESTS] = {
		.name = "atu_filtered_requests"
	},
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_0] = {
		.name = "atu_ats_prs_odp_latency_0"
	},
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_1] = {
		.name = "atu_ats_prs_odp_latency_1"
	},
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_2] = {
		.name = "atu_ats_prs_odp_latency_2"
	},
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_3] = {
		.name = "atu_ats_prs_odp_latency_3"
	},
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_0] = {
		.name = "atu_nic_pri_odp_latency_0"
	},
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_1] = {
		.name = "atu_nic_pri_odp_latency_1"
	},
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_2] = {
		.name = "atu_nic_pri_odp_latency_2"
	},
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_3] = {
		.name = "atu_nic_pri_odp_latency_3"
	},
	[C_CNTR_ATU_ODP_REQUESTS_0] = {
		.name = "atu_odp_requests_0"
	},
	[C_CNTR_ATU_ODP_REQUESTS_1] = {
		.name = "atu_odp_requests_1"
	},
	[C_CNTR_ATU_ODP_REQUESTS_2] = {
		.name = "atu_odp_requests_2"
	},
	[C_CNTR_ATU_ODP_REQUESTS_3] = {
		.name = "atu_odp_requests_3"
	},
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_0] = {
		.name = "atu_client_rsp_not_ok_0"
	},
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_1] = {
		.name = "atu_client_rsp_not_ok_1"
	},
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_2] = {
		.name = "atu_client_rsp_not_ok_2"
	},
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_3] = {
		.name = "atu_client_rsp_not_ok_3"
	},
	[C_CNTR_ATU_CACHE_EVICTIONS] = {
		.name = "atu_cache_evictions"
	},
	[C_CNTR_ATU_ATS_INVAL_CNTR] = {
		.name = "atu_ats_inval_cntr"
	},
	[C_CNTR_ATU_IMPLICIT_FLR_INVAL_CNTR] = {
		.name = "atu_implicit_flr_inval_cntr"
	},
	[C_CNTR_ATU_IMPLICIT_ATS_EN_INVAL_CNTR] = {
		.name = "atu_implicit_ats_en_inval_cntr"
	},
	[C_CNTR_ATU_ATUCQ_INVAL_CNTR] = {
		.name = "atu_atucq_inval_cntr"
	},
	[C_CNTR_ATU_PCIE_UNSUCCESS_CMPL] = {
		.name = "atu_pcie_unsuccess_cmpl"
	},
	[C_CNTR_ATU_ATS_TRANS_ERR] = {
		.name = "atu_ats_trans_err"
	},
	[C_CNTR_ATU_PCIE_ERR_POISONED] = {
		.name = "atu_pcie_err_poisoned"
	},
	[C_CNTR_ATU_ATS_DYNAMIC_STATE_CHANGE_CNTR] = {
		.name = "atu_ats_dynamic_state_change_cntr"
	},
	[C_CNTR_ATU_AT_STALL_CCP] = {
		.name = "atu_at_stall_ccp"
	},
	[C_CNTR_ATU_AT_STALL_NO_PTID] = {
		.name = "atu_at_stall_no_ptid"
	},
	[C_CNTR_ATU_AT_STALL_NP_CDTS] = {
		.name = "atu_at_stall_np_cdts"
	},
	[C_CNTR_ATU_AT_STALL_TARB_ARB] = {
		.name = "atu_at_stall_tarb_arb"
	},
	[C_CNTR_ATU_INVAL_STALL_ARB] = {
		.name = "atu_inval_stall_arb"
	},
	[C_CNTR_ATU_INVAL_STALL_CMPL_WAIT] = {
		.name = "atu_inval_stall_cmpl_wait"
	},
	[C_CNTR_CQ_SUCCESS_TX_CNTR] = {
		.name = "cq_success_tx_cntr"
	},
	[C_CNTR_CQ_SUCCESS_TGT_CNTR] = {
		.name = "cq_success_tgt_cntr"
	},
	[C_CNTR_CQ_FAIL_TX_CNTR] = {
		.name = "cq_fail_tx_cntr"
	},
	[C_CNTR_CQ_FAIL_TGT_CNTR] = {
		.name = "cq_fail_tgt_cntr"
	},
	[C_CNTR_CQ_MEM_COR_ERR_CNTR] = {
		.name = "cq_mem_cor_err_cntr"
	},
	[C_CNTR_CQ_MEM_UCOR_ERR_CNTR] = {
		.name = "cq_mem_ucor_err_cntr"
	},
	[C_CNTR_CQ_RARB_LL_ERR_CNTR] = {
		.name = "cq_rarb_ll_err_cntr"
	},
	[C_CNTR_CQ_RARB_ERR_CNTR] = {
		.name = "cq_rarb_err_cntr"
	},
	[C_CNTR_CQ_WR_PTR_UPDT_ERR_CNTR] = {
		.name = "cq_wr_ptr_updt_err_cntr"
	},
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_0] = {
		.name = "cq_num_txq_cmd_reads_0"
	},
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_1] = {
		.name = "cq_num_txq_cmd_reads_1"
	},
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_2] = {
		.name = "cq_num_txq_cmd_reads_2"
	},
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_3] = {
		.name = "cq_num_txq_cmd_reads_3"
	},
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_0] = {
		.name = "cq_num_tgq_cmd_reads_0"
	},
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_1] = {
		.name = "cq_num_tgq_cmd_reads_1"
	},
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_2] = {
		.name = "cq_num_tgq_cmd_reads_2"
	},
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_3] = {
		.name = "cq_num_tgq_cmd_reads_3"
	},
	[C_CNTR_CQ_NUM_TOU_CMD_READS_0] = {
		.name = "cq_num_tou_cmd_reads_0"
	},
	[C_CNTR_CQ_NUM_TOU_CMD_READS_1] = {
		.name = "cq_num_tou_cmd_reads_1"
	},
	[C_CNTR_CQ_NUM_TOU_CMD_READS_2] = {
		.name = "cq_num_tou_cmd_reads_2"
	},
	[C_CNTR_CQ_NUM_TOU_CMD_READS_3] = {
		.name = "cq_num_tou_cmd_reads_3"
	},
	[C_CNTR_CQ_TX_WAITING_ON_READ_0] = {
		.name = "cq_tx_waiting_on_read_0"
	},
	[C_CNTR_CQ_TX_WAITING_ON_READ_1] = {
		.name = "cq_tx_waiting_on_read_1"
	},
	[C_CNTR_CQ_TX_WAITING_ON_READ_2] = {
		.name = "cq_tx_waiting_on_read_2"
	},
	[C_CNTR_CQ_TX_WAITING_ON_READ_3] = {
		.name = "cq_tx_waiting_on_read_3"
	},
	[C_CNTR_CQ_TGT_WAITING_ON_READ_0] = {
		.name = "cq_tgt_waiting_on_read_0"
	},
	[C_CNTR_CQ_TGT_WAITING_ON_READ_1] = {
		.name = "cq_tgt_waiting_on_read_1"
	},
	[C_CNTR_CQ_TGT_WAITING_ON_READ_2] = {
		.name = "cq_tgt_waiting_on_read_2"
	},
	[C_CNTR_CQ_TGT_WAITING_ON_READ_3] = {
		.name = "cq_tgt_waiting_on_read_3"
	},
	[C_CNTR_CQ_NUM_TX_CMD_ALIGN_ERRORS] = {
		.name = "cq_num_tx_cmd_align_errors"
	},
	[C_CNTR_CQ_NUM_TX_CMD_OP_ERRORS] = {
		.name = "cq_num_tx_cmd_op_errors"
	},
	[C_CNTR_CQ_NUM_TX_CMD_ARG_ERRORS] = {
		.name = "cq_num_tx_cmd_arg_errors"
	},
	[C_CNTR_CQ_NUM_TX_CMD_PERM_ERRORS] = {
		.name = "cq_num_tx_cmd_perm_errors"
	},
	[C_CNTR_CQ_NUM_TGT_CMD_ARG_ERRORS] = {
		.name = "cq_num_tgt_cmd_arg_errors"
	},
	[C_CNTR_CQ_NUM_TGT_CMD_PERM_ERRORS] = {
		.name = "cq_num_tgt_cmd_perm_errors"
	},
	[C_CNTR_CQ_CQ_OXE_NUM_FLITS] = {
		.name = "cq_cq_oxe_num_flits"
	},
	[C_CNTR_CQ_CQ_OXE_NUM_STALLS] = {
		.name = "cq_cq_oxe_num_stalls"
	},
	[C_CNTR_CQ_CQ_OXE_NUM_IDLES] = {
		.name = "cq_cq_oxe_num_idles"
	},
	[C_CNTR_CQ_CQ_TOU_NUM_FLITS] = {
		.name = "cq_cq_tou_num_flits"
	},
	[C_CNTR_CQ_CQ_TOU_NUM_STALLS] = {
		.name = "cq_cq_tou_num_stalls"
	},
	[C_CNTR_CQ_CQ_TOU_NUM_IDLES] = {
		.name = "cq_cq_tou_num_idles"
	},
	[C_CNTR_CQ_NUM_IDC_CMDS_0] = {
		.name = "cq_num_idc_cmds_0"
	},
	[C_CNTR_CQ_NUM_IDC_CMDS_1] = {
		.name = "cq_num_idc_cmds_1"
	},
	[C_CNTR_CQ_NUM_IDC_CMDS_2] = {
		.name = "cq_num_idc_cmds_2"
	},
	[C_CNTR_CQ_NUM_IDC_CMDS_3] = {
		.name = "cq_num_idc_cmds_3"
	},
	[C_CNTR_CQ_NUM_DMA_CMDS_0] = {
		.name = "cq_num_dma_cmds_0"
	},
	[C_CNTR_CQ_NUM_DMA_CMDS_1] = {
		.name = "cq_num_dma_cmds_1"
	},
	[C_CNTR_CQ_NUM_DMA_CMDS_2] = {
		.name = "cq_num_dma_cmds_2"
	},
	[C_CNTR_CQ_NUM_DMA_CMDS_3] = {
		.name = "cq_num_dma_cmds_3"
	},
	[C_CNTR_CQ_NUM_CQ_CMDS_0] = {
		.name = "cq_num_cq_cmds_0"
	},
	[C_CNTR_CQ_NUM_CQ_CMDS_1] = {
		.name = "cq_num_cq_cmds_1"
	},
	[C_CNTR_CQ_NUM_CQ_CMDS_2] = {
		.name = "cq_num_cq_cmds_2"
	},
	[C_CNTR_CQ_NUM_CQ_CMDS_3] = {
		.name = "cq_num_cq_cmds_3"
	},
	[C_CNTR_CQ_NUM_LL_CMDS_0] = {
		.name = "cq_num_ll_cmds_0"
	},
	[C_CNTR_CQ_NUM_LL_CMDS_1] = {
		.name = "cq_num_ll_cmds_1"
	},
	[C_CNTR_CQ_NUM_LL_CMDS_2] = {
		.name = "cq_num_ll_cmds_2"
	},
	[C_CNTR_CQ_NUM_LL_CMDS_3] = {
		.name = "cq_num_ll_cmds_3"
	},
	[C_CNTR_CQ_NUM_TGT_CMDS_0] = {
		.name = "cq_num_tgt_cmds_0"
	},
	[C_CNTR_CQ_NUM_TGT_CMDS_1] = {
		.name = "cq_num_tgt_cmds_1"
	},
	[C_CNTR_CQ_NUM_TGT_CMDS_2] = {
		.name = "cq_num_tgt_cmds_2"
	},
	[C_CNTR_CQ_NUM_TGT_CMDS_3] = {
		.name = "cq_num_tgt_cmds_3"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_0] = {
		.name = "cq_dma_cmd_counts_0"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_1] = {
		.name = "cq_dma_cmd_counts_1"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_2] = {
		.name = "cq_dma_cmd_counts_2"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_3] = {
		.name = "cq_dma_cmd_counts_3"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_4] = {
		.name = "cq_dma_cmd_counts_4"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_5] = {
		.name = "cq_dma_cmd_counts_5"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_6] = {
		.name = "cq_dma_cmd_counts_6"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_7] = {
		.name = "cq_dma_cmd_counts_7"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_8] = {
		.name = "cq_dma_cmd_counts_8"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_9] = {
		.name = "cq_dma_cmd_counts_9"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_10] = {
		.name = "cq_dma_cmd_counts_10"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_11] = {
		.name = "cq_dma_cmd_counts_11"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_12] = {
		.name = "cq_dma_cmd_counts_12"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_13] = {
		.name = "cq_dma_cmd_counts_13"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_14] = {
		.name = "cq_dma_cmd_counts_14"
	},
	[C_CNTR_CQ_DMA_CMD_COUNTS_15] = {
		.name = "cq_dma_cmd_counts_15"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_0] = {
		.name = "cq_cq_cmd_counts_0"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_1] = {
		.name = "cq_cq_cmd_counts_1"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_2] = {
		.name = "cq_cq_cmd_counts_2"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_3] = {
		.name = "cq_cq_cmd_counts_3"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_4] = {
		.name = "cq_cq_cmd_counts_4"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_5] = {
		.name = "cq_cq_cmd_counts_5"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_6] = {
		.name = "cq_cq_cmd_counts_6"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_7] = {
		.name = "cq_cq_cmd_counts_7"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_8] = {
		.name = "cq_cq_cmd_counts_8"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_9] = {
		.name = "cq_cq_cmd_counts_9"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_10] = {
		.name = "cq_cq_cmd_counts_10"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_11] = {
		.name = "cq_cq_cmd_counts_11"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_12] = {
		.name = "cq_cq_cmd_counts_12"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_13] = {
		.name = "cq_cq_cmd_counts_13"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_14] = {
		.name = "cq_cq_cmd_counts_14"
	},
	[C_CNTR_CQ_CQ_CMD_COUNTS_15] = {
		.name = "cq_cq_cmd_counts_15"
	},
	[C_CNTR_CQ_CYCLES_BLOCKED_0] = {
		.name = "cq_cycles_blocked_0"
	},
	[C_CNTR_CQ_CYCLES_BLOCKED_1] = {
		.name = "cq_cycles_blocked_1"
	},
	[C_CNTR_CQ_CYCLES_BLOCKED_2] = {
		.name = "cq_cycles_blocked_2"
	},
	[C_CNTR_CQ_CYCLES_BLOCKED_3] = {
		.name = "cq_cycles_blocked_3"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_0] = {
		.name = "cq_num_ll_ops_successful_0"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_1] = {
		.name = "cq_num_ll_ops_successful_1"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_2] = {
		.name = "cq_num_ll_ops_successful_2"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_3] = {
		.name = "cq_num_ll_ops_successful_3"
	},
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_0] = {
		.name = "cq_num_ll_ops_rejected_0"
	},
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_1] = {
		.name = "cq_num_ll_ops_rejected_1"
	},
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_2] = {
		.name = "cq_num_ll_ops_rejected_2"
	},
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_3] = {
		.name = "cq_num_ll_ops_rejected_3"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_0] = {
		.name = "cq_num_ll_ops_split_0"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_1] = {
		.name = "cq_num_ll_ops_split_1"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_2] = {
		.name = "cq_num_ll_ops_split_2"
	},
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_3] = {
		.name = "cq_num_ll_ops_split_3"
	},
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_0] = {
		.name = "cq_num_ll_ops_received_0"
	},
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_1] = {
		.name = "cq_num_ll_ops_received_1"
	},
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_2] = {
		.name = "cq_num_ll_ops_received_2"
	},
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_3] = {
		.name = "cq_num_ll_ops_received_3"
	},
	[C_CNTR_TOU_SUCCESS_CNTR] = {
		.name = "tou_success_cntr"
	},
	[C_CNTR_TOU_FAIL_CNTR] = {
		.name = "tou_fail_cntr"
	},
	[C_CNTR_TOU_CQ_TOU_NUM_CMDS] = {
		.name = "tou_cq_tou_num_cmds"
	},
	[C_CNTR_TOU_CQ_TOU_NUM_STALLS] = {
		.name = "tou_cq_tou_num_stalls"
	},
	[C_CNTR_TOU_TOU_OXE_NUM_FLITS] = {
		.name = "tou_tou_oxe_num_flits"
	},
	[C_CNTR_TOU_TOU_OXE_NUM_STALLS] = {
		.name = "tou_tou_oxe_num_stalls"
	},
	[C_CNTR_TOU_TOU_OXE_NUM_IDLES] = {
		.name = "tou_tou_oxe_num_idles"
	},
	[C_CNTR_TOU_LIST_REBUILD_CYCLES] = {
		.name = "tou_list_rebuild_cycles"
	},
	[C_CNTR_TOU_CMD_FIFO_FULL_CYCLES] = {
		.name = "tou_cmd_fifo_full_cycles"
	},
	[C_CNTR_TOU_NUM_DOORBELL_WRITES] = {
		.name = "tou_num_doorbell_writes"
	},
	[C_CNTR_TOU_NUM_CT_UPDATES] = {
		.name = "tou_num_ct_updates"
	},
	[C_CNTR_TOU_NUM_TRIG_CMDS_0] = {
		.name = "tou_num_trig_cmds_0"
	},
	[C_CNTR_TOU_NUM_TRIG_CMDS_1] = {
		.name = "tou_num_trig_cmds_1"
	},
	[C_CNTR_TOU_NUM_TRIG_CMDS_2] = {
		.name = "tou_num_trig_cmds_2"
	},
	[C_CNTR_TOU_NUM_TRIG_CMDS_3] = {
		.name = "tou_num_trig_cmds_3"
	},
	[C_CNTR_TOU_NUM_LIST_REBUILDS_0] = {
		.name = "tou_num_list_rebuilds_0"
	},
	[C_CNTR_TOU_NUM_LIST_REBUILDS_1] = {
		.name = "tou_num_list_rebuilds_1"
	},
	[C_CNTR_TOU_NUM_LIST_REBUILDS_2] = {
		.name = "tou_num_list_rebuilds_2"
	},
	[C_CNTR_TOU_NUM_LIST_REBUILDS_3] = {
		.name = "tou_num_list_rebuilds_3"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_0] = {
		.name = "tou_ct_cmd_counts_0"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_1] = {
		.name = "tou_ct_cmd_counts_1"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_2] = {
		.name = "tou_ct_cmd_counts_2"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_3] = {
		.name = "tou_ct_cmd_counts_3"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_4] = {
		.name = "tou_ct_cmd_counts_4"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_5] = {
		.name = "tou_ct_cmd_counts_5"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_6] = {
		.name = "tou_ct_cmd_counts_6"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_7] = {
		.name = "tou_ct_cmd_counts_7"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_8] = {
		.name = "tou_ct_cmd_counts_8"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_9] = {
		.name = "tou_ct_cmd_counts_9"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_10] = {
		.name = "tou_ct_cmd_counts_10"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_11] = {
		.name = "tou_ct_cmd_counts_11"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_12] = {
		.name = "tou_ct_cmd_counts_12"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_13] = {
		.name = "tou_ct_cmd_counts_13"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_14] = {
		.name = "tou_ct_cmd_counts_14"
	},
	[C_CNTR_TOU_CT_CMD_COUNTS_15] = {
		.name = "tou_ct_cmd_counts_15"
	},
	[C_CNTR_PCT_MEM_COR_ERR_CNTR] = {
		.name = "pct_mem_cor_err_cntr"
	},
	[C_CNTR_PCT_MEM_UCOR_ERR_CNTR] = {
		.name = "pct_mem_ucor_err_cntr"
	},
	[C_CNTR_PCT_REQ_ORDERED] = {
		.name = "pct_req_ordered"
	},
	[C_CNTR_PCT_REQ_UNORDERED] = {
		.name = "pct_req_unordered"
	},
	[C_CNTR_PCT_REQ_NO_RESPONSE] = {
		.name = "pct_req_no_response"
	},
	[C_CNTR_PCT_ETH_PACKETS] = {
		.name = "pct_eth_packets"
	},
	[C_CNTR_PCT_OPTIP_PACKETS] = {
		.name = "pct_optip_packets"
	},
	[C_CNTR_PCT_PTLS_RESPONSE] = {
		.name = "pct_ptls_response"
	},
	[C_CNTR_PCT_RESPONSES_RECEIVED] = {
		.name = "pct_responses_received"
	},
	[C_CNTR_PCT_HRP_RESPONSES_RECEIVED] = {
		.name = "pct_hrp_responses_received"
	},
	[C_CNTR_PCT_HRP_RSP_DISCARD_RECEIVED] = {
		.name = "pct_hrp_rsp_discard_received"
	},
	[C_CNTR_PCT_HRP_RSP_ERR_RECEIVED] = {
		.name = "pct_hrp_rsp_err_received"
	},
	[C_CNTR_PCT_CONN_SCT_OPEN] = {
		.name = "pct_conn_sct_open"
	},
	[C_CNTR_PCT_CONN_TCT_OPEN] = {
		.name = "pct_conn_tct_open"
	},
	[C_CNTR_PCT_MST_HIT_ON_SOM] = {
		.name = "pct_mst_hit_on_som"
	},
	[C_CNTR_PCT_TRS_HIT_ON_REQ] = {
		.name = "pct_trs_hit_on_req"
	},
	[C_CNTR_PCT_CLS_REQ_MISS_TCT] = {
		.name = "pct_cls_req_miss_tct"
	},
	[C_CNTR_PCT_CLEAR_SENT] = {
		.name = "pct_clear_sent"
	},
	[C_CNTR_PCT_CLOSE_SENT] = {
		.name = "pct_close_sent"
	},
	[C_CNTR_PCT_ACCEL_CLOSE] = {
		.name = "pct_accel_close"
	},
	[C_CNTR_PCT_CLEAR_CLOSE_DROP] = {
		.name = "pct_clear_close_drop"
	},
	[C_CNTR_PCT_REQ_SRC_ERROR] = {
		.name = "pct_req_src_error"
	},
	[C_CNTR_PCT_BAD_SEQ_NACKS] = {
		.name = "pct_bad_seq_nacks"
	},
	[C_CNTR_PCT_NO_TCT_NACKS] = {
		.name = "pct_no_tct_nacks"
	},
	[C_CNTR_PCT_NO_MST_NACKS] = {
		.name = "pct_no_mst_nacks"
	},
	[C_CNTR_PCT_NO_TRS_NACKS] = {
		.name = "pct_no_trs_nacks"
	},
	[C_CNTR_PCT_NO_MATCHING_TCT] = {
		.name = "pct_no_matching_tct"
	},
	[C_CNTR_PCT_RESOURCE_BUSY] = {
		.name = "pct_resource_busy"
	},
	[C_CNTR_PCT_ERR_NO_MATCHING_TRS] = {
		.name = "pct_err_no_matching_trs"
	},
	[C_CNTR_PCT_ERR_NO_MATCHING_MST] = {
		.name = "pct_err_no_matching_mst"
	},
	[C_CNTR_PCT_SPT_TIMEOUTS] = {
		.name = "pct_spt_timeouts"
	},
	[C_CNTR_PCT_SCT_TIMEOUTS] = {
		.name = "pct_sct_timeouts"
	},
	[C_CNTR_PCT_TCT_TIMEOUTS] = {
		.name = "pct_tct_timeouts"
	},
	[C_CNTR_PCT_RETRY_SRB_REQUESTS] = {
		.name = "pct_retry_srb_requests"
	},
	[C_CNTR_PCT_RETRY_TRS_PUT] = {
		.name = "pct_retry_trs_put"
	},
	[C_CNTR_PCT_RETRY_MST_GET] = {
		.name = "pct_retry_mst_get"
	},
	[C_CNTR_PCT_CLR_CLS_STALLS] = {
		.name = "pct_clr_cls_stalls"
	},
	[C_CNTR_PCT_CLOSE_RSP_DROPS] = {
		.name = "pct_close_rsp_drops"
	},
	[C_CNTR_PCT_TRS_RSP_NACK_DROPS] = {
		.name = "pct_trs_rsp_nack_drops"
	},
	[C_CNTR_PCT_REQ_BLOCKED_CLOSING] = {
		.name = "pct_req_blocked_closing"
	},
	[C_CNTR_PCT_REQ_BLOCKED_CLEARING] = {
		.name = "pct_req_blocked_clearing"
	},
	[C_CNTR_PCT_REQ_BLOCKED_RETRY] = {
		.name = "pct_req_blocked_retry"
	},
	[C_CNTR_PCT_RSP_ERR_RCVD] = {
		.name = "pct_rsp_err_rcvd"
	},
	[C_CNTR_PCT_RSP_DROPPED_TIMEOUT] = {
		.name = "pct_rsp_dropped_timeout"
	},
	[C_CNTR_PCT_RSP_DROPPED_TRY] = {
		.name = "pct_rsp_dropped_try"
	},
	[C_CNTR_PCT_RSP_DROPPED_CLS_TRY] = {
		.name = "pct_rsp_dropped_cls_try"
	},
	[C_CNTR_PCT_RSP_DROPPED_INACTIVE] = {
		.name = "pct_rsp_dropped_inactive"
	},
	[C_CNTR_PCT_PCT_HNI_FLITS] = {
		.name = "pct_pct_hni_flits"
	},
	[C_CNTR_PCT_PCT_HNI_STALLS] = {
		.name = "pct_pct_hni_stalls"
	},
	[C_CNTR_PCT_PCT_EE_EVENTS] = {
		.name = "pct_pct_ee_events"
},
	[C_CNTR_PCT_PCT_EE_STALLS] = {
		.name = "pct_pct_ee_stalls"
	},
	[C_CNTR_PCT_PCT_CQ_NOTIFICATIONS] = {
		.name = "pct_pct_cq_notifications"
	},
	[C_CNTR_PCT_PCT_CQ_STALLS] = {
		.name = "pct_pct_cq_stalls"
	},
	[C_CNTR_PCT_PCT_MST_CMD0] = {
		.name = "pct_pct_mst_cmd0"
	},
	[C_CNTR_PCT_PCT_MST_STALLS0] = {
		.name = "pct_pct_mst_stalls0"
	},
	[C_CNTR_PCT_PCT_MST_CMD1] = {
		.name = "pct_pct_mst_cmd1"
	},
	[C_CNTR_PCT_PCT_MST_STALLS1] = {
		.name = "pct_pct_mst_stalls1"
	},
	[C_CNTR_PCT_SCT_STALL_STATE] = {
		.name = "pct_sct_stall_state"
	},
	[C_CNTR_PCT_TGT_CLS_ABORT] = {
		.name = "pct_tgt_cls_abort"
	},
	[C_CNTR_PCT_CLS_REQ_BAD_SEQNO_DROPS] = {
		.name = "pct_cls_req_bad_seqno_drops"
	},
	[C_CNTR_PCT_REQ_TCT_TMOUT_DROPS] = {
		.name = "pct_req_tct_tmout_drops"
	},
	[C_CNTR_PCT_TRS_REPLAY_PEND_DROPS] = {
		.name = "pct_trs_replay_pend_drops"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_0] = {
		.name = "pct_req_rsp_latency_0"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_1] = {
		.name = "pct_req_rsp_latency_1"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_2] = {
		.name = "pct_req_rsp_latency_2"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_3] = {
		.name = "pct_req_rsp_latency_3"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_4] = {
		.name = "pct_req_rsp_latency_4"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_5] = {
		.name = "pct_req_rsp_latency_5"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_6] = {
		.name = "pct_req_rsp_latency_6"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_7] = {
		.name = "pct_req_rsp_latency_7"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_8] = {
		.name = "pct_req_rsp_latency_8"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_9] = {
		.name = "pct_req_rsp_latency_9"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_10] = {
		.name = "pct_req_rsp_latency_10"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_11] = {
		.name = "pct_req_rsp_latency_11"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_12] = {
		.name = "pct_req_rsp_latency_12"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_13] = {
		.name = "pct_req_rsp_latency_13"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_14] = {
		.name = "pct_req_rsp_latency_14"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_15] = {
		.name = "pct_req_rsp_latency_15"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_16] = {
		.name = "pct_req_rsp_latency_16"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_17] = {
		.name = "pct_req_rsp_latency_17"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_18] = {
		.name = "pct_req_rsp_latency_18"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_19] = {
		.name = "pct_req_rsp_latency_19"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_20] = {
		.name = "pct_req_rsp_latency_20"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_21] = {
		.name = "pct_req_rsp_latency_21"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_22] = {
		.name = "pct_req_rsp_latency_22"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_23] = {
		.name = "pct_req_rsp_latency_23"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_24] = {
		.name = "pct_req_rsp_latency_24"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_25] = {
		.name = "pct_req_rsp_latency_25"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_26] = {
		.name = "pct_req_rsp_latency_26"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_27] = {
		.name = "pct_req_rsp_latency_27"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_28] = {
		.name = "pct_req_rsp_latency_28"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_29] = {
		.name = "pct_req_rsp_latency_29"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_30] = {
		.name = "pct_req_rsp_latency_30"
	},
	[C_CNTR_PCT_REQ_RSP_LATENCY_31] = {
		.name = "pct_req_rsp_latency_31"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_0] = {
		.name = "pct_host_access_latency_0"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_1] = {
		.name = "pct_host_access_latency_1"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_2] = {
		.name = "pct_host_access_latency_2"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_3] = {
		.name = "pct_host_access_latency_3"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_4] = {
		.name = "pct_host_access_latency_4"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_5] = {
		.name = "pct_host_access_latency_5"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_6] = {
		.name = "pct_host_access_latency_6"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_7] = {
		.name = "pct_host_access_latency_7"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_8] = {
		.name = "pct_host_access_latency_8"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_9] = {
		.name = "pct_host_access_latency_9"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_10] = {
		.name = "pct_host_access_latency_10"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_11] = {
		.name = "pct_host_access_latency_11"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_12] = {
		.name = "pct_host_access_latency_12"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_13] = {
		.name = "pct_host_access_latency_13"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_14] = {
		.name = "pct_host_access_latency_14"
	},
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_15] = {
		.name = "pct_host_access_latency_15"
	},
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_0] = {
		.name = "ee_events_enqueued_cntr_0"
	},
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_1] = {
		.name = "ee_events_enqueued_cntr_1"
	},
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_2] = {
		.name = "ee_events_enqueued_cntr_2"
	},
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_3] = {
		.name = "ee_events_enqueued_cntr_3"
	},
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_0] = {
		.name = "ee_events_dropped_rsrvn_cntr_0"
	},
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_1] = {
		.name = "ee_events_dropped_rsrvn_cntr_1"
	},
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_2] = {
		.name = "ee_events_dropped_rsrvn_cntr_2"
	},
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_3] = {
		.name = "ee_events_dropped_rsrvn_cntr_3"
	},
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_0] = {
		.name = "ee_events_dropped_fc_sc_cntr_0"
	},
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_1] = {
		.name = "ee_events_dropped_fc_sc_cntr_1"
	},
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_2] = {
		.name = "ee_events_dropped_fc_sc_cntr_2"
	},
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_3] = {
		.name = "ee_events_dropped_fc_sc_cntr_3"
	},
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_0] = {
		.name = "ee_events_dropped_ordinary_cntr_0"
	},
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_1] = {
		.name = "ee_events_dropped_ordinary_cntr_1"
	},
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_2] = {
		.name = "ee_events_dropped_ordinary_cntr_2"
	},
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_3] = {
		.name = "ee_events_dropped_ordinary_cntr_3"
	},
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_0] = {
		.name = "ee_eq_status_update_cntr_0"
	},
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_1] = {
		.name = "ee_eq_status_update_cntr_1"
	},
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_2] = {
		.name = "ee_eq_status_update_cntr_2"
	},
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_3] = {
		.name = "ee_eq_status_update_cntr_3"
	},
	[C_CNTR_EE_CBS_WRITTEN_CNTR_0] = {
		.name = "ee_cbs_written_cntr_0"
	},
	[C_CNTR_EE_CBS_WRITTEN_CNTR_1] = {
		.name = "ee_cbs_written_cntr_1"
	},
	[C_CNTR_EE_CBS_WRITTEN_CNTR_2] = {
		.name = "ee_cbs_written_cntr_2"
	},
	[C_CNTR_EE_CBS_WRITTEN_CNTR_3] = {
		.name = "ee_cbs_written_cntr_3"
	},
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_0] = {
		.name = "ee_partial_cbs_written_cntr_0"
	},
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_1] = {
		.name = "ee_partial_cbs_written_cntr_1"
	},
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_2] = {
		.name = "ee_partial_cbs_written_cntr_2"
	},
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_3] = {
		.name = "ee_partial_cbs_written_cntr_3"
	},
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_0] = {
		.name = "ee_expired_cbs_written_cntr_0"
	},
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_1] = {
		.name = "ee_expired_cbs_written_cntr_1"
	},
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_2] = {
		.name = "ee_expired_cbs_written_cntr_2"
	},
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_3] = {
		.name = "ee_expired_cbs_written_cntr_3"
	},
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_0] = {
		.name = "ee_eq_buffer_switch_cntr_0"
	},
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_1] = {
		.name = "ee_eq_buffer_switch_cntr_1"
	},
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_2] = {
		.name = "ee_eq_buffer_switch_cntr_2"
	},
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_3] = {
		.name = "ee_eq_buffer_switch_cntr_3"
	},
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_0] = {
		.name = "ee_deferred_eq_switch_cntr_0"
	},
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_1] = {
		.name = "ee_deferred_eq_switch_cntr_1"
	},
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_2] = {
		.name = "ee_deferred_eq_switch_cntr_2"
	},
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_3] = {
		.name = "ee_deferred_eq_switch_cntr_3"
	},
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_0] = {
		.name = "ee_addr_trans_prefetch_cntr_0"
	},
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_1] = {
		.name = "ee_addr_trans_prefetch_cntr_1"
	},
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_2] = {
		.name = "ee_addr_trans_prefetch_cntr_2"
	},
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_3] = {
		.name = "ee_addr_trans_prefetch_cntr_3"
	},
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_0] = {
		.name = "ee_eq_sw_state_wr_cntr_0"
	},
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_1] = {
		.name = "ee_eq_sw_state_wr_cntr_1"
	},
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_2] = {
		.name = "ee_eq_sw_state_wr_cntr_2"
	},
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_3] = {
		.name = "ee_eq_sw_state_wr_cntr_3"
	},
	[C_CNTR_EE_LPE_EVENT_REQ_CNTR] = {
		.name = "ee_lpe_event_req_cntr"
	},
	[C_CNTR_EE_LPE_FE_CNTR] = {
		.name = "ee_lpe_fe_cntr"
	},
	[C_CNTR_EE_LPE_DD_CNTR] = {
		.name = "ee_lpe_dd_cntr"
	},
	[C_CNTR_EE_LPE_CE_CNTR] = {
		.name = "ee_lpe_ce_cntr"
	},
	[C_CNTR_EE_LPE_FE_STALL_CNTR] = {
		.name = "ee_lpe_fe_stall_cntr"
	},
	[C_CNTR_EE_LPE_CE_STALL_CNTR] = {
		.name = "ee_lpe_ce_stall_cntr"
	},
	[C_CNTR_EE_IXE_EVENT_REQ_CNTR] = {
		.name = "ee_ixe_event_req_cntr"
	},
	[C_CNTR_EE_IXE_FE_CNTR] = {
		.name = "ee_ixe_fe_cntr"
	},
	[C_CNTR_EE_IXE_DD_CNTR] = {
		.name = "ee_ixe_dd_cntr"
	},
	[C_CNTR_EE_IXE_CE_CNTR] = {
		.name = "ee_ixe_ce_cntr"
	},
	[C_CNTR_EE_IXE_FE_STALL_CNTR] = {
		.name = "ee_ixe_fe_stall_cntr"
	},
	[C_CNTR_EE_IXE_CE_STALL_CNTR] = {
		.name = "ee_ixe_ce_stall_cntr"
	},
	[C_CNTR_EE_MST_EVENT_REQ_CNTR] = {
		.name = "ee_mst_event_req_cntr"
	},
	[C_CNTR_EE_MST_FE_CNTR] = {
		.name = "ee_mst_fe_cntr"
	},
	[C_CNTR_EE_MST_DD_CNTR] = {
		.name = "ee_mst_dd_cntr"
	},
	[C_CNTR_EE_MST_CE_CNTR] = {
		.name = "ee_mst_ce_cntr"
	},
	[C_CNTR_EE_MST_FE_STALL_CNTR] = {
		.name = "ee_mst_fe_stall_cntr"
	},
	[C_CNTR_EE_MST_CE_STALL_CNTR] = {
		.name = "ee_mst_ce_stall_cntr"
	},
	[C_CNTR_EE_PCT_EVENT_REQ_CNTR] = {
		.name = "ee_pct_event_req_cntr"
	},
	[C_CNTR_EE_PCT_FE_CNTR] = {
		.name = "ee_pct_fe_cntr"
	},
	[C_CNTR_EE_PCT_DD_CNTR] = {
		.name = "ee_pct_dd_cntr"
	},
	[C_CNTR_EE_PCT_CE_CNTR] = {
		.name = "ee_pct_ce_cntr"
	},
	[C_CNTR_EE_PCT_FE_STALL_CNTR] = {
		.name = "ee_pct_fe_stall_cntr"
	},
	[C_CNTR_EE_PCT_CE_STALL_CNTR] = {
		.name = "ee_pct_ce_stall_cntr"
	},
	[C_CNTR_EE_HNI_EVENT_REQ_CNTR] = {
		.name = "ee_hni_event_req_cntr"
	},
	[C_CNTR_EE_HNI_FE_CNTR] = {
		.name = "ee_hni_fe_cntr"
	},
	[C_CNTR_EE_HNI_FE_STALL_CNTR] = {
		.name = "ee_hni_fe_stall_cntr"
	},
	[C_CNTR_EE_CQ_EVENT_REQ_CNTR] = {
		.name = "ee_cq_event_req_cntr"
	},
	[C_CNTR_EE_CQ_FE_CNTR] = {
		.name = "ee_cq_fe_cntr"
	},
	[C_CNTR_EE_CQ_FE_STALL_CNTR] = {
		.name = "ee_cq_fe_stall_cntr"
	},
	[C_CNTR_EE_TS_FE_CNTR] = {
		.name = "ee_ts_fe_cntr"
	},
	[C_CNTR_EE_TS_FE_STALL_CNTR] = {
		.name = "ee_ts_fe_stall_cntr"
	},
	[C_CNTR_EE_FE_ARB_OUT_EVENT_CNTR] = {
		.name = "ee_fe_arb_out_event_cntr"
	},
	[C_CNTR_EE_FE_ARB_OUT_STALL_CNTR] = {
		.name = "ee_fe_arb_out_stall_cntr"
	},
	[C_CNTR_EE_CE_ARB_OUT_EVENT_CNTR] = {
		.name = "ee_ce_arb_out_event_cntr"
	},
	[C_CNTR_EE_CE_ARB_OUT_STALL_CNTR] = {
		.name = "ee_ce_arb_out_stall_cntr"
	},
	[C_CNTR_EE_LPE_QUERY_EQ_NONE_CNTR] = {
		.name = "ee_lpe_query_eq_none_cntr"
	},
	[C_CNTR_EE_EQS_LPE_QUERY_REQ_CNTR] = {
		.name = "ee_eqs_lpe_query_req_cntr"
	},
	[C_CNTR_EE_EQS_CSR_REQ_CNTR] = {
		.name = "ee_eqs_csr_req_cntr"
	},
	[C_CNTR_EE_EQS_CSR_STALL_CNTR] = {
		.name = "ee_eqs_csr_stall_cntr"
	},
	[C_CNTR_EE_EQS_HWS_INIT_REQ_CNTR] = {
		.name = "ee_eqs_hws_init_req_cntr"
	},
	[C_CNTR_EE_EQS_HWS_INIT_STALL_CNTR] = {
		.name = "ee_eqs_hws_init_stall_cntr"
	},
	[C_CNTR_EE_EQS_EXPIRED_CB_REQ_CNTR] = {
		.name = "ee_eqs_expired_cb_req_cntr"
	},
	[C_CNTR_EE_EQS_EXPIRED_CB_STALL_CNTR] = {
		.name = "ee_eqs_expired_cb_stall_cntr"
	},
	[C_CNTR_EE_EQS_STS_UPDT_REQ_CNTR] = {
		.name = "ee_eqs_sts_updt_req_cntr"
	},
	[C_CNTR_EE_EQS_STS_UPDT_STALL_CNTR] = {
		.name = "ee_eqs_sts_updt_stall_cntr"
	},
	[C_CNTR_EE_EQS_SWS_WR_REQ_CNTR] = {
		.name = "ee_eqs_sws_wr_req_cntr"
	},
	[C_CNTR_EE_EQS_SWS_WR_STALL_CNTR] = {
		.name = "ee_eqs_sws_wr_stall_cntr"
	},
	[C_CNTR_EE_EQS_EVENT_REQ_CNTR] = {
		.name = "ee_eqs_event_req_cntr"
	},
	[C_CNTR_EE_EQS_EVENT_STALL_CNTR] = {
		.name = "ee_eqs_event_stall_cntr"
	},
	[C_CNTR_EE_EQS_ARB_OUT_REQ_CNTR] = {
		.name = "ee_eqs_arb_out_req_cntr"
	},
	[C_CNTR_EE_EQS_FREE_CB_STALL_CNTR] = {
		.name = "ee_eqs_free_cb_stall_cntr"
	},
	[C_CNTR_EE_ADDR_TRANS_REQ_CNTR] = {
		.name = "ee_addr_trans_req_cntr"
	},
	[C_CNTR_EE_TARB_WR_REQ_CNTR] = {
		.name = "ee_tarb_wr_req_cntr"
	},
	[C_CNTR_EE_TARB_IRQ_REQ_CNTR] = {
		.name = "ee_tarb_irq_req_cntr"
	},
	[C_CNTR_EE_TARB_STALL_CNTR] = {
		.name = "ee_tarb_stall_cntr"
	},
	[C_CNTR_EE_EVENT_PUT_CNTR] = {
		.name = "ee_event_put_cntr"
	},
	[C_CNTR_EE_EVENT_GET_CNTR] = {
		.name = "ee_event_get_cntr"
	},
	[C_CNTR_EE_EVENT_ATOMIC_CNTR] = {
		.name = "ee_event_atomic_cntr"
	},
	[C_CNTR_EE_EVENT_FETCH_ATOMIC_CNTR] = {
		.name = "ee_event_fetch_atomic_cntr"
	},
	[C_CNTR_EE_EVENT_PUT_OVERFLOW_CNTR] = {
		.name = "ee_event_put_overflow_cntr"
	},
	[C_CNTR_EE_EVENT_GET_OVERFLOW_CNTR] = {
		.name = "ee_event_get_overflow_cntr"
	},
	[C_CNTR_EE_EVENT_ATOMIC_OVERFLOW_CNTR] = {
		.name = "ee_event_atomic_overflow_cntr"
	},
	[C_CNTR_EE_EVENT_FETCH_ATOMIC_OVERFLOW_CNTR] = {
		.name = "ee_event_fetch_atomic_overflow_cntr"
	},
	[C_CNTR_EE_EVENT_SEND_CNTR] = {
		.name = "ee_event_send_cntr"
	},
	[C_CNTR_EE_EVENT_ACK_CNTR] = {
		.name = "ee_event_ack_cntr"
	},
	[C_CNTR_EE_EVENT_REPLY_CNTR] = {
		.name = "ee_event_reply_cntr"
	},
	[C_CNTR_EE_EVENT_LINK_CNTR] = {
		.name = "ee_event_link_cntr"
	},
	[C_CNTR_EE_EVENT_SEARCH_CNTR] = {
		.name = "ee_event_search_cntr"
	},
	[C_CNTR_EE_EVENT_STATE_CHANGE_CNTR] = {
		.name = "ee_event_state_change_cntr"
	},
	[C_CNTR_EE_EVENT_UNLINK_CNTR] = {
		.name = "ee_event_unlink_cntr"
	},
	[C_CNTR_EE_EVENT_RENDEZVOUS_CNTR] = {
		.name = "ee_event_rendezvous_cntr"
	},
	[C_CNTR_EE_EVENT_ETHERNET_CNTR] = {
		.name = "ee_event_ethernet_cntr"
	},
	[C_CNTR_EE_EVENT_COMMAND_FAILURE_CNTR] = {
		.name = "ee_event_command_failure_cntr"
	},
	[C_CNTR_EE_EVENT_TRIGGERED_OP_CNTR] = {
		.name = "ee_event_triggered_op_cntr"
	},
	[C_CNTR_EE_EVENT_ETHERNET_FGFC_CNTR] = {
		.name = "ee_event_ethernet_fgfc_cntr"
	},
	[C_CNTR_EE_EVENT_PCT_CNTR] = {
		.name = "ee_event_pct_cntr"
	},
	[C_CNTR_EE_EVENT_MATCH_CNTR] = {
		.name = "ee_event_match_cntr"
	},
	[C_CNTR_EE_EVENT_ERROR_CNTR] = {
		.name = "ee_event_error_cntr"
	},
	[C_CNTR_EE_EVENT_TIMESTAMP_CNTR] = {
		.name = "ee_event_timestamp_cntr"
	},
	[C_CNTR_EE_EVENT_EQ_SWITCH_CNTR] = {
		.name = "ee_event_eq_switch_cntr"
	},
	[C_CNTR_EE_EVENT_NULL_EVENT_CNTR] = {
		.name = "ee_event_null_event_cntr"
	},
	[C_CNTR_EE_MEM_COR_ERR_CNTR] = {
		.name = "ee_mem_cor_err_cntr"
	},
	[C_CNTR_EE_MEM_UCOR_ERR_CNTR] = {
		.name = "ee_mem_ucor_err_cntr"
	},
	[C_CNTR_EE_EQ_DSABLD_EVENT_ERR_CNTR] = {
		.name = "ee_eq_dsabld_event_err_cntr"
	},
	[C_CNTR_EE_EQ_DSABLD_SWS_ERR_CNTR] = {
		.name = "ee_eq_dsabld_sws_err_cntr"
	},
	[C_CNTR_EE_EQ_DSABLD_LPEQ_ERR_CNTR] = {
		.name = "ee_eq_dsabld_lpeq_err_cntr"
	},
	[C_CNTR_EE_EQ_RSRVN_UFLW_ERR_CNTR] = {
		.name = "ee_eq_rsrvn_uflw_err_cntr"
	},
	[C_CNTR_EE_UNXPCTD_TRNSLTN_RSP_ERR_CNTR] = {
		.name = "ee_unxpctd_trnsltn_rsp_err_cntr"
	},
	[C_CNTR_EE_RARB_HW_ERR_CNTR] = {
		.name = "ee_rarb_hw_err_cntr"
	},
	[C_CNTR_EE_RARB_SW_ERR_CNTR] = {
		.name = "ee_rarb_sw_err_cntr"
	},
	[C_CNTR_EE_TARB_ERR_CNTR] = {
		.name = "ee_tarb_err_cntr"
	},
	[C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR] = {
		.name = "ee_eq_state_ucor_err_cntr"
	},
	[C_CNTR_LPE_SUCCESS_CNTR] = {
		.name = "lpe_success_cntr"
	},
	[C_CNTR_LPE_FAIL_CNTR] = {
		.name = "lpe_fail_cntr"
	},
	[C_CNTR_LPE_MEM_COR_ERR_CNTR] = {
		.name = "lpe_mem_cor_err_cntr"
	},
	[C_CNTR_LPE_MEM_UCOR_ERR_CNTR] = {
		.name = "lpe_mem_ucor_err_cntr"
	},
	[C_CNTR_LPE_PUT_CMDS] = {
		.name = "lpe_put_cmds"
	},
	[C_CNTR_LPE_RENDEZVOUS_PUT_CMDS] = {
		.name = "lpe_rendezvous_put_cmds"
	},
	[C_CNTR_LPE_GET_CMDS] = {
		.name = "lpe_get_cmds"
	},
	[C_CNTR_LPE_AMO_CMDS] = {
		.name = "lpe_amo_cmds"
	},
	[C_CNTR_LPE_FAMO_CMDS] = {
		.name = "lpe_famo_cmds"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_ETHERNET] = {
		.name = "lpe_err_entry_not_found_ethernet"
	},
	[C_CNTR_LPE_ERR_PT_DISABLED_EQ_FULL] = {
		.name = "lpe_err_pt_disabled_eq_full"
	},
	[C_CNTR_LPE_ERR_PT_DISABLED_REQ_EMPTY] = {
		.name = "lpe_err_pt_disabled_req_empty"
	},
	[C_CNTR_LPE_ERR_PT_DISABLED_CANT_ALLOC_UNEXPECTED] = {
		.name = "lpe_err_pt_disabled_cant_alloc_unexpected"
	},
	[C_CNTR_LPE_ERR_PT_DISABLED_NO_MATCH_IN_OVERFLOW] = {
		.name = "lpe_err_pt_disabled_no_match_in_overflow"
	},
	[C_CNTR_LPE_ERR_EQ_DISABLED] = {
		.name = "lpe_err_eq_disabled"
	},
	[C_CNTR_LPE_ERR_OP_VIOLATION] = {
		.name = "lpe_err_op_violation"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_UNLINK_FAILED] = {
		.name = "lpe_err_entry_not_found_unlink_failed"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_EQ_FULL] = {
		.name = "lpe_err_entry_not_found_eq_full"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_REQ_EMPTY] = {
		.name = "lpe_err_entry_not_found_req_empty"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_NO_MATCH_IN_OVERFLOW] = {
		.name = "lpe_err_entry_not_found_no_match_in_overflow"
	},
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_CANT_ALLOC_UNEXPECTED] = {
		.name = "lpe_err_entry_not_found_cant_alloc_unexpected"
	},
	[C_CNTR_LPE_ERR_INVALID_ENDPOINT] = {
		.name = "lpe_err_invalid_endpoint"
	},
	[C_CNTR_LPE_ERR_NO_SPACE_APPEND] = {
		.name = "lpe_err_no_space_append"
	},
	[C_CNTR_LPE_ERR_NO_SPACE_NET] = {
		.name = "lpe_err_no_space_net"
	},
	[C_CNTR_LPE_ERR_SEARCH_NO_MATCH] = {
		.name = "lpe_err_search_no_match"
	},
	[C_CNTR_LPE_ERR_SETSTATE_NO_MATCH] = {
		.name = "lpe_err_setstate_no_match"
	},
	[C_CNTR_LPE_ERR_SRC_ERROR] = {
		.name = "lpe_err_src_error"
	},
	[C_CNTR_LPE_ERR_PTLTE_SW_MANAGED] = {
		.name = "lpe_err_ptlte_sw_managed"
	},
	[C_CNTR_LPE_ERR_ILLEGAL_OP] = {
		.name = "lpe_err_illegal_op"
	},
	[C_CNTR_LPE_ERR_RESTRICTED_UNICAST] = {
		.name = "lpe_err_restricted_unicast"
	},
	[C_CNTR_LPE_EVENT_PUT_OVERFLOW] = {
		.name = "lpe_event_put_overflow"
	},
	[C_CNTR_LPE_EVENT_GET_OVERFLOW] = {
		.name = "lpe_event_get_overflow"
	},
	[C_CNTR_LPE_EVENT_ATOMIC_OVERFLOW] = {
		.name = "lpe_event_atomic_overflow"
	},
	[C_CNTR_LPE_PLEC_FREES_CSR] = {
		.name = "lpe_plec_frees_csr"
	},
	[C_CNTR_LPE_EVENT_FETCH_ATOMIC_OVERFLOW] = {
		.name = "lpe_event_fetch_atomic_overflow"
	},
	[C_CNTR_LPE_EVENT_SEARCH] = {
		.name = "lpe_event_search"
	},
	[C_CNTR_LPE_EVENT_RENDEZVOUS] = {
		.name = "lpe_event_rendezvous"
	},
	[C_CNTR_LPE_EVENT_LINK] = {
		.name = "lpe_event_link"
	},
	[C_CNTR_LPE_EVENT_UNLINK] = {
		.name = "lpe_event_unlink"
	},
	[C_CNTR_LPE_EVENT_STATECHANGE] = {
		.name = "lpe_event_statechange"
	},
	[C_CNTR_LPE_PLEC_ALLOCS] = {
		.name = "lpe_plec_allocs"
	},
	[C_CNTR_LPE_PLEC_FREES] = {
		.name = "lpe_plec_frees"
	},
	[C_CNTR_LPE_PLEC_HITS] = {
		.name = "lpe_plec_hits"
	},
	[C_CNTR_LPE_CYC_NO_RDY_CDTS] = {
		.name = "lpe_cyc_no_rdy_cdts"
	},
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_0] = {
		.name = "lpe_cyc_rrq_blocked_0"
	},
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_1] = {
		.name = "lpe_cyc_rrq_blocked_1"
	},
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_2] = {
		.name = "lpe_cyc_rrq_blocked_2"
	},
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_3] = {
		.name = "lpe_cyc_rrq_blocked_3"
	},
	[C_CNTR_LPE_CYC_BLOCKED_LOSSLESS] = {
		.name = "lpe_cyc_blocked_lossless"
	},
	[C_CNTR_LPE_CYC_BLOCKED_IXE_PUT] = {
		.name = "lpe_cyc_blocked_ixe_put"
	},
	[C_CNTR_LPE_CYC_BLOCKED_IXE_GET] = {
		.name = "lpe_cyc_blocked_ixe_get"
	},
	[C_CNTR_LPE_SEARCH_CMDS] = {
		.name = "lpe_search_cmds"
	},
	[C_CNTR_LPE_SEARCH_SUCCESS] = {
		.name = "lpe_search_success"
	},
	[C_CNTR_LPE_SEARCH_FAIL] = {
		.name = "lpe_search_fail"
	},
	[C_CNTR_LPE_SEARCH_DELETE_CMDS] = {
		.name = "lpe_search_delete_cmds"
	},
	[C_CNTR_LPE_SEARCH_DELETE_SUCCESS] = {
		.name = "lpe_search_delete_success"
	},
	[C_CNTR_LPE_SEARCH_DELETE_FAIL] = {
		.name = "lpe_search_delete_fail"
	},
	[C_CNTR_LPE_UNLINK_CMDS] = {
		.name = "lpe_unlink_cmds"
	},
	[C_CNTR_LPE_UNLINK_SUCCESS] = {
		.name = "lpe_unlink_success"
	},
	[C_CNTR_LPE_NET_MATCH_REQUESTS_0] = {
		.name = "lpe_net_match_requests_0"
	},
	[C_CNTR_LPE_NET_MATCH_REQUESTS_1] = {
		.name = "lpe_net_match_requests_1"
	},
	[C_CNTR_LPE_NET_MATCH_REQUESTS_2] = {
		.name = "lpe_net_match_requests_2"
	},
	[C_CNTR_LPE_NET_MATCH_REQUESTS_3] = {
		.name = "lpe_net_match_requests_3"
	},
	[C_CNTR_LPE_NET_MATCH_SUCCESS_0] = {
		.name = "lpe_net_match_success_0"
	},
	[C_CNTR_LPE_NET_MATCH_SUCCESS_1] = {
		.name = "lpe_net_match_success_1"
	},
	[C_CNTR_LPE_NET_MATCH_SUCCESS_2] = {
		.name = "lpe_net_match_success_2"
	},
	[C_CNTR_LPE_NET_MATCH_SUCCESS_3] = {
		.name = "lpe_net_match_success_3"
	},
	[C_CNTR_LPE_NET_MATCH_USEONCE_0] = {
		.name = "lpe_net_match_useonce_0"
	},
	[C_CNTR_LPE_NET_MATCH_USEONCE_1] = {
		.name = "lpe_net_match_useonce_1"
	},
	[C_CNTR_LPE_NET_MATCH_USEONCE_2] = {
		.name = "lpe_net_match_useonce_2"
	},
	[C_CNTR_LPE_NET_MATCH_USEONCE_3] = {
		.name = "lpe_net_match_useonce_3"
	},
	[C_CNTR_LPE_NET_MATCH_LOCAL_0] = {
		.name = "lpe_net_match_local_0"
	},
	[C_CNTR_LPE_NET_MATCH_LOCAL_1] = {
		.name = "lpe_net_match_local_1"
	},
	[C_CNTR_LPE_NET_MATCH_LOCAL_2] = {
		.name = "lpe_net_match_local_2"
	},
	[C_CNTR_LPE_NET_MATCH_LOCAL_3] = {
		.name = "lpe_net_match_local_3"
	},
	[C_CNTR_LPE_NET_MATCH_PRIORITY_0] = {
		.name = "lpe_net_match_priority_0"
	},
	[C_CNTR_LPE_NET_MATCH_PRIORITY_1] = {
		.name = "lpe_net_match_priority_1"
	},
	[C_CNTR_LPE_NET_MATCH_PRIORITY_2] = {
		.name = "lpe_net_match_priority_2"
	},
	[C_CNTR_LPE_NET_MATCH_PRIORITY_3] = {
		.name = "lpe_net_match_priority_3"
	},
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_0] = {
		.name = "lpe_net_match_overflow_0"
	},
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_1] = {
		.name = "lpe_net_match_overflow_1"
	},
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_2] = {
		.name = "lpe_net_match_overflow_2"
	},
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_3] = {
		.name = "lpe_net_match_overflow_3"
	},
	[C_CNTR_LPE_NET_MATCH_REQUEST_0] = {
		.name = "lpe_net_match_request_0"
	},
	[C_CNTR_LPE_NET_MATCH_REQUEST_1] = {
		.name = "lpe_net_match_request_1"
	},
	[C_CNTR_LPE_NET_MATCH_REQUEST_2] = {
		.name = "lpe_net_match_request_2"
	},
	[C_CNTR_LPE_NET_MATCH_REQUEST_3] = {
		.name = "lpe_net_match_request_3"
	},
	[C_CNTR_LPE_APPEND_CMDS_0] = {
		.name = "lpe_append_cmds_0"
	},
	[C_CNTR_LPE_APPEND_CMDS_1] = {
		.name = "lpe_append_cmds_1"
	},
	[C_CNTR_LPE_APPEND_CMDS_2] = {
		.name = "lpe_append_cmds_2"
	},
	[C_CNTR_LPE_APPEND_CMDS_3] = {
		.name = "lpe_append_cmds_3"
	},
	[C_CNTR_LPE_APPEND_SUCCESS_0] = {
		.name = "lpe_append_success_0"
	},
	[C_CNTR_LPE_APPEND_SUCCESS_1] = {
		.name = "lpe_append_success_1"
	},
	[C_CNTR_LPE_APPEND_SUCCESS_2] = {
		.name = "lpe_append_success_2"
	},
	[C_CNTR_LPE_APPEND_SUCCESS_3] = {
		.name = "lpe_append_success_3"
	},
	[C_CNTR_LPE_SETSTATE_CMDS_0] = {
		.name = "lpe_setstate_cmds_0"
	},
	[C_CNTR_LPE_SETSTATE_CMDS_1] = {
		.name = "lpe_setstate_cmds_1"
	},
	[C_CNTR_LPE_SETSTATE_CMDS_2] = {
		.name = "lpe_setstate_cmds_2"
	},
	[C_CNTR_LPE_SETSTATE_CMDS_3] = {
		.name = "lpe_setstate_cmds_3"
	},
	[C_CNTR_LPE_SETSTATE_SUCCESS_0] = {
		.name = "lpe_setstate_success_0"
	},
	[C_CNTR_LPE_SETSTATE_SUCCESS_1] = {
		.name = "lpe_setstate_success_1"
	},
	[C_CNTR_LPE_SETSTATE_SUCCESS_2] = {
		.name = "lpe_setstate_success_2"
	},
	[C_CNTR_LPE_SETSTATE_SUCCESS_3] = {
		.name = "lpe_setstate_success_3"
	},
	[C_CNTR_LPE_SEARCH_NID_ANY_0] = {
		.name = "lpe_search_nid_any_0"
	},
	[C_CNTR_LPE_SEARCH_NID_ANY_1] = {
		.name = "lpe_search_nid_any_1"
	},
	[C_CNTR_LPE_SEARCH_NID_ANY_2] = {
		.name = "lpe_search_nid_any_2"
	},
	[C_CNTR_LPE_SEARCH_NID_ANY_3] = {
		.name = "lpe_search_nid_any_3"
	},
	[C_CNTR_LPE_SEARCH_PID_ANY_0] = {
		.name = "lpe_search_pid_any_0"
	},
	[C_CNTR_LPE_SEARCH_PID_ANY_1] = {
		.name = "lpe_search_pid_any_1"
	},
	[C_CNTR_LPE_SEARCH_PID_ANY_2] = {
		.name = "lpe_search_pid_any_2"
	},
	[C_CNTR_LPE_SEARCH_PID_ANY_3] = {
		.name = "lpe_search_pid_any_3"
	},
	[C_CNTR_LPE_SEARCH_RANK_ANY_0] = {
		.name = "lpe_search_rank_any_0"
	},
	[C_CNTR_LPE_SEARCH_RANK_ANY_1] = {
		.name = "lpe_search_rank_any_1"
	},
	[C_CNTR_LPE_SEARCH_RANK_ANY_2] = {
		.name = "lpe_search_rank_any_2"
	},
	[C_CNTR_LPE_SEARCH_RANK_ANY_3] = {
		.name = "lpe_search_rank_any_3"
	},
	[C_CNTR_LPE_RNDZV_PUTS_0] = {
		.name = "lpe_rndzv_puts_0"
	},
	[C_CNTR_LPE_RNDZV_PUTS_1] = {
		.name = "lpe_rndzv_puts_1"
	},
	[C_CNTR_LPE_RNDZV_PUTS_2] = {
		.name = "lpe_rndzv_puts_2"
	},
	[C_CNTR_LPE_RNDZV_PUTS_3] = {
		.name = "lpe_rndzv_puts_3"
	},
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_0] = {
		.name = "lpe_rndzv_puts_offloaded_0"
	},
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_1] = {
		.name = "lpe_rndzv_puts_offloaded_1"
	},
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_2] = {
		.name = "lpe_rndzv_puts_offloaded_2"
	},
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_3] = {
		.name = "lpe_rndzv_puts_offloaded_3"
	},
	[C_CNTR_LPE_NUM_TRUNCATED_0] = {
		.name = "lpe_num_truncated_0"
	},
	[C_CNTR_LPE_NUM_TRUNCATED_1] = {
		.name = "lpe_num_truncated_1"
	},
	[C_CNTR_LPE_NUM_TRUNCATED_2] = {
		.name = "lpe_num_truncated_2"
	},
	[C_CNTR_LPE_NUM_TRUNCATED_3] = {
		.name = "lpe_num_truncated_3"
	},
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_0] = {
		.name = "lpe_unexpected_get_amo_0"
	},
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_1] = {
		.name = "lpe_unexpected_get_amo_1"
	},
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_2] = {
		.name = "lpe_unexpected_get_amo_2"
	},
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_3] = {
		.name = "lpe_unexpected_get_amo_3"
	},
	[C_CNTR_IXE_MEM_COR_ERR_CNTR] = {
		.name = "ixe_mem_cor_err_cntr"
	},
	[C_CNTR_IXE_MEM_UCOR_ERR_CNTR] = {
		.name = "ixe_mem_ucor_err_cntr"
	},
	[C_CNTR_IXE_PORT_DFA_MISMATCH] = {
		.name = "ixe_port_dfa_mismatch"
	},
	[C_CNTR_IXE_HDR_CHECKSUM_ERRORS] = {
		.name = "ixe_hdr_checksum_errors"
	},
	[C_CNTR_IXE_IPV4_CHECKSUM_ERRORS] = {
		.name = "ixe_ipv4_checksum_errors"
	},
	[C_CNTR_IXE_HRP_REQ_ERRORS] = {
		.name = "ixe_hrp_req_errors"
	},
	[C_CNTR_IXE_IP_OPTIONS_ERRORS] = {
		.name = "ixe_ip_options_errors"
	},
	[C_CNTR_IXE_GET_LEN_ERRORS] = {
		.name = "ixe_get_len_errors"
	},
	[C_CNTR_IXE_ROCE_ICRC_ERROR] = {
		.name = "ixe_roce_icrc_error"
	},
	[C_CNTR_IXE_PARSER_PAR_ERRORS] = {
		.name = "ixe_parser_par_errors"
	},
	[C_CNTR_IXE_PBUF_RD_ERRORS] = {
		.name = "ixe_pbuf_rd_errors"
	},
	[C_CNTR_IXE_HDR_ECC_ERRORS] = {
		.name = "ixe_hdr_ecc_errors"
	},
	[C_CNTR_IXE_RX_UDP_PKT] = {
		.name = "ixe_rx_udp_pkt"
	},
	[C_CNTR_IXE_RX_TCP_PKT] = {
		.name = "ixe_rx_tcp_pkt"
	},
	[C_CNTR_IXE_RX_IPV4_PKT] = {
		.name = "ixe_rx_ipv4_pkt"
	},
	[C_CNTR_IXE_RX_IPV6_PKT] = {
		.name = "ixe_rx_ipv6_pkt"
	},
	[C_CNTR_IXE_RX_ROCE_PKT] = {
		.name = "ixe_rx_roce_pkt"
	},
	[C_CNTR_IXE_RX_PTL_GEN_PKT] = {
		.name = "ixe_rx_ptl_gen_pkt"
	},
	[C_CNTR_IXE_RX_PTL_SML_PKT] = {
		.name = "ixe_rx_ptl_sml_pkt"
	},
	[C_CNTR_IXE_RX_PTL_UNRESTRICTED_PKT] = {
		.name = "ixe_rx_ptl_unrestricted_pkt"
	},
	[C_CNTR_IXE_RX_PTL_SMALLMSG_PKT] = {
		.name = "ixe_rx_ptl_smallmsg_pkt"
	},
	[C_CNTR_IXE_RX_PTL_CONTINUATION_PKT] = {
		.name = "ixe_rx_ptl_continuation_pkt"
	},
	[C_CNTR_IXE_RX_PTL_RESTRICTED_PKT] = {
		.name = "ixe_rx_ptl_restricted_pkt"
	},
	[C_CNTR_IXE_RX_PTL_CONNMGMT_PKT] = {
		.name = "ixe_rx_ptl_connmgmt_pkt"
	},
	[C_CNTR_IXE_RX_PTL_RESPONSE_PKT] = {
		.name = "ixe_rx_ptl_response_pkt"
	},
	[C_CNTR_IXE_RX_UNRECOGNIZED_PKT] = {
		.name = "ixe_rx_unrecognized_pkt"
	},
	[C_CNTR_IXE_RX_PTL_SML_AMO_PKT] = {
		.name = "ixe_rx_ptl_sml_amo_pkt"
	},
	[C_CNTR_IXE_RX_PTL_MSGS] = {
		.name = "ixe_rx_ptl_msgs"
	},
	[C_CNTR_IXE_RX_PTL_MULTI_MSGS] = {
		.name = "ixe_rx_ptl_multi_msgs"
	},
	[C_CNTR_IXE_RX_PTL_MR_MSGS] = {
		.name = "ixe_rx_ptl_mr_msgs"
	},
	[C_CNTR_IXE_RX_PKT_DROP_PCT] = {
		.name = "ixe_rx_pkt_drop_pct"
	},
	[C_CNTR_IXE_RX_PKT_DROP_RMU_NORSP] = {
		.name = "ixe_rx_pkt_drop_rmu_norsp"
	},
	[C_CNTR_IXE_RX_PKT_DROP_RMU_WRSP] = {
		.name = "ixe_rx_pkt_drop_rmu_wrsp"
	},
	[C_CNTR_IXE_RX_PKT_DROP_IXE_PARSER] = {
		.name = "ixe_rx_pkt_drop_ixe_parser"
	},
	[C_CNTR_IXE_RX_PKT_IPV4_OPTIONS] = {
		.name = "ixe_rx_pkt_ipv4_options"
	},
	[C_CNTR_IXE_RX_PKT_IPV6_OPTIONS] = {
		.name = "ixe_rx_pkt_ipv6_options"
	},
	[C_CNTR_IXE_RX_ETH_SEG] = {
		.name = "ixe_rx_eth_seg"
	},
	[C_CNTR_IXE_RX_ROCE_SEG] = {
		.name = "ixe_rx_roce_seg"
	},
	[C_CNTR_IXE_RX_ROCE_SPSEG] = {
		.name = "ixe_rx_roce_spseg"
	},
	[C_CNTR_IXE_POOL_ECN_PKTS_0] = {
		.name = "ixe_pool_ecn_pkts_0"
	},
	[C_CNTR_IXE_POOL_ECN_PKTS_1] = {
		.name = "ixe_pool_ecn_pkts_1"
	},
	[C_CNTR_IXE_POOL_ECN_PKTS_2] = {
		.name = "ixe_pool_ecn_pkts_2"
	},
	[C_CNTR_IXE_POOL_ECN_PKTS_3] = {
		.name = "ixe_pool_ecn_pkts_3"
	},
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_0] = {
		.name = "ixe_pool_no_ecn_pkts_0"
	},
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_1] = {
		.name = "ixe_pool_no_ecn_pkts_1"
	},
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_2] = {
		.name = "ixe_pool_no_ecn_pkts_2"
	},
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_3] = {
		.name = "ixe_pool_no_ecn_pkts_3"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_0] = {
		.name = "ixe_tc_req_ecn_pkts_0"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_1] = {
		.name = "ixe_tc_req_ecn_pkts_1"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_2] = {
		.name = "ixe_tc_req_ecn_pkts_2"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_3] = {
		.name = "ixe_tc_req_ecn_pkts_3"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_4] = {
		.name = "ixe_tc_req_ecn_pkts_4"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_5] = {
		.name = "ixe_tc_req_ecn_pkts_5"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_6] = {
		.name = "ixe_tc_req_ecn_pkts_6"
	},
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_7] = {
		.name = "ixe_tc_req_ecn_pkts_7"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_0] = {
		.name = "ixe_tc_req_no_ecn_pkts_0"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_1] = {
		.name = "ixe_tc_req_no_ecn_pkts_1"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_2] = {
		.name = "ixe_tc_req_no_ecn_pkts_2"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_3] = {
		.name = "ixe_tc_req_no_ecn_pkts_3"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_4] = {
		.name = "ixe_tc_req_no_ecn_pkts_4"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_5] = {
		.name = "ixe_tc_req_no_ecn_pkts_5"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_6] = {
		.name = "ixe_tc_req_no_ecn_pkts_6"
	},
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_7] = {
		.name = "ixe_tc_req_no_ecn_pkts_7"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_0] = {
		.name = "ixe_tc_rsp_ecn_pkts_0"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_1] = {
		.name = "ixe_tc_rsp_ecn_pkts_1"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_2] = {
		.name = "ixe_tc_rsp_ecn_pkts_2"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_3] = {
		.name = "ixe_tc_rsp_ecn_pkts_3"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_4] = {
		.name = "ixe_tc_rsp_ecn_pkts_4"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_5] = {
		.name = "ixe_tc_rsp_ecn_pkts_5"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_6] = {
		.name = "ixe_tc_rsp_ecn_pkts_6"
	},
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_7] = {
		.name = "ixe_tc_rsp_ecn_pkts_7"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_0] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_0"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_1] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_1"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_2] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_2"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_3] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_3"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_4] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_4"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_5] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_5"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_6] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_6"
	},
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_7] = {
		.name = "ixe_tc_rsp_no_ecn_pkts_7"
	},
	[C_CNTR_IXE_DISP_LPE_PUTS] = {
		.name = "ixe_disp_lpe_puts"
	},
	[C_CNTR_IXE_DISP_LPE_PUTS_OK] = {
		.name = "ixe_disp_lpe_puts_ok"
	},
	[C_CNTR_IXE_DISP_LPE_AMOS] = {
		.name = "ixe_disp_lpe_amos"
	},
	[C_CNTR_IXE_DISP_LPE_AMOS_OK] = {
		.name = "ixe_disp_lpe_amos_ok"
	},
	[C_CNTR_IXE_DISP_MST_PUTS] = {
		.name = "ixe_disp_mst_puts"
	},
	[C_CNTR_IXE_DISP_MST_PUTS_OK] = {
		.name = "ixe_disp_mst_puts_ok"
	},
	[C_CNTR_IXE_DISP_LPE_GETS] = {
		.name = "ixe_disp_lpe_gets"
	},
	[C_CNTR_IXE_DISP_LPE_GETS_OK] = {
		.name = "ixe_disp_lpe_gets_ok"
	},
	[C_CNTR_IXE_DISP_MST_GETS] = {
		.name = "ixe_disp_mst_gets"
	},
	[C_CNTR_IXE_DISP_MST_GETS_OK] = {
		.name = "ixe_disp_mst_gets_ok"
	},
	[C_CNTR_IXE_DISP_RPU_RESPS] = {
		.name = "ixe_disp_rpu_resps"
	},
	[C_CNTR_IXE_DISP_RPU_ERR_REQS] = {
		.name = "ixe_disp_rpu_err_reqs"
	},
	[C_CNTR_IXE_DISP_DMAWR_REQS] = {
		.name = "ixe_disp_dmawr_reqs"
	},
	[C_CNTR_IXE_DISP_DMAWR_RESPS] = {
		.name = "ixe_disp_dmawr_resps"
	},
	[C_CNTR_IXE_DISP_DMAWR_RESPS_OK] = {
		.name = "ixe_disp_dmawr_resps_ok"
	},
	[C_CNTR_IXE_DISP_OXE_RESPS] = {
		.name = "ixe_disp_oxe_resps"
	},
	[C_CNTR_IXE_DISP_PCT_GETCOMP] = {
		.name = "ixe_disp_pct_getcomp"
	},
	[C_CNTR_IXE_DISP_ATU_REQS] = {
		.name = "ixe_disp_atu_reqs"
	},
	[C_CNTR_IXE_DISP_ATU_RESPS] = {
		.name = "ixe_disp_atu_resps"
	},
	[C_CNTR_IXE_DISP_ATU_RESPS_OK] = {
		.name = "ixe_disp_atu_resps_ok"
	},
	[C_CNTR_IXE_DISP_ETH_EVENTS] = {
		.name = "ixe_disp_eth_events"
	},
	[C_CNTR_IXE_DISP_PUT_EVENTS] = {
		.name = "ixe_disp_put_events"
	},
	[C_CNTR_IXE_DISP_AMO_EVENTS] = {
		.name = "ixe_disp_amo_events"
	},
	[C_CNTR_IXE_DISP_WR_CONFLICTS] = {
		.name = "ixe_disp_wr_conflicts"
	},
	[C_CNTR_IXE_DISP_AMO_CONFLICTS] = {
		.name = "ixe_disp_amo_conflicts"
	},
	[C_CNTR_IXE_DISP_STALL_CONFLICT] = {
		.name = "ixe_disp_stall_conflict"
	},
	[C_CNTR_IXE_DISP_STALL_RESP_FIFO] = {
		.name = "ixe_disp_stall_resp_fifo"
	},
	[C_CNTR_IXE_DISP_STALL_ERR_FIFO] = {
		.name = "ixe_disp_stall_err_fifo"
	},
	[C_CNTR_IXE_DISP_STALL_ATU_CDT] = {
		.name = "ixe_disp_stall_atu_cdt"
	},
	[C_CNTR_IXE_DISP_STALL_ATU_FIFO] = {
		.name = "ixe_disp_stall_atu_fifo"
	},
	[C_CNTR_IXE_DISP_STALL_GCOMP_FIFO] = {
		.name = "ixe_disp_stall_gcomp_fifo"
	},
	[C_CNTR_IXE_DISP_STALL_MDID_CDTS] = {
		.name = "ixe_disp_stall_mdid_cdts"
	},
	[C_CNTR_IXE_DISP_STALL_MST_MATCH_FIFO] = {
		.name = "ixe_disp_stall_mst_match_fifo"
	},
	[C_CNTR_IXE_DISP_STALL_GRR_ID] = {
		.name = "ixe_disp_stall_grr_id"
	},
	[C_CNTR_IXE_DISP_STALL_PUT_RESP] = {
		.name = "ixe_disp_stall_put_resp"
	},
	[C_CNTR_IXE_DISP_ATU_FLUSH_REQS] = {
		.name = "ixe_disp_atu_flush_reqs"
	},
	[C_CNTR_IXE_DMAWR_STALL_P_CDT] = {
		.name = "ixe_dmawr_stall_p_cdt"
	},
	[C_CNTR_IXE_DMAWR_STALL_NP_CDT] = {
		.name = "ixe_dmawr_stall_np_cdt"
	},
	[C_CNTR_IXE_DMAWR_STALL_NP_REQ_CNT] = {
		.name = "ixe_dmawr_stall_np_req_cnt"
	},
	[C_CNTR_IXE_DMAWR_STALL_FTCH_AMO_CNT] = {
		.name = "ixe_dmawr_stall_ftch_amo_cnt"
	},
	[C_CNTR_IXE_DMAWR_P_PASS_NP_CNT] = {
		.name = "ixe_dmawr_p_pass_np_cnt"
	},
	[C_CNTR_IXE_DMAWR_WRITE_REQS] = {
		.name = "ixe_dmawr_write_reqs"
	},
	[C_CNTR_IXE_DMAWR_NIC_AMO_REQS] = {
		.name = "ixe_dmawr_nic_amo_reqs"
	},
	[C_CNTR_IXE_DMAWR_CPU_AMO_REQS] = {
		.name = "ixe_dmawr_cpu_amo_reqs"
	},
	[C_CNTR_IXE_DMAWR_CPU_FTCH_AMO_REQS] = {
		.name = "ixe_dmawr_cpu_ftch_amo_reqs"
	},
	[C_CNTR_IXE_DMAWR_FLUSH_REQS] = {
		.name = "ixe_dmawr_flush_reqs"
	},
	[C_CNTR_IXE_DMAWR_REQ_NO_WRITE_REQS] = {
		.name = "ixe_dmawr_req_no_write_reqs"
	},
	[C_CNTR_IXE_DMAWR_AMO_EX_INVALID] = {
		.name = "ixe_dmawr_amo_ex_invalid"
	},
	[C_CNTR_IXE_DMAWR_AMO_EX_OVERFLOW] = {
		.name = "ixe_dmawr_amo_ex_overflow"
	},
	[C_CNTR_IXE_DMAWR_AMO_EX_UNDERFLOW] = {
		.name = "ixe_dmawr_amo_ex_underflow"
	},
	[C_CNTR_IXE_DMAWR_AMO_EX_INEXACT] = {
		.name = "ixe_dmawr_amo_ex_inexact"
	},
	[C_CNTR_IXE_DMAWR_AMO_MISALIGNED] = {
		.name = "ixe_dmawr_amo_misaligned"
	},
	[C_CNTR_IXE_DMAWR_AMO_INVALID_OP] = {
		.name = "ixe_dmawr_amo_invalid_op"
	},
	[C_CNTR_IXE_DMAWR_AMO_LEN_ERR] = {
		.name = "ixe_dmawr_amo_len_err"
	},
	[C_CNTR_IXE_DMAWR_PCIE_UNSUCCESS_CMPL] = {
		.name = "ixe_dmawr_pcie_unsuccess_cmpl"
	},
	[C_CNTR_IXE_DMAWR_PCIE_ERR_POISONED] = {
		.name = "ixe_dmawr_pcie_err_poisoned"
	},
	[C_CNTR_RMU_PTL_SUCCESS_CNTR] = {
		.name = "rmu_ptl_success_cntr"
	},
	[C_CNTR_RMU_ENET_SUCCESS_CNTR] = {
		.name = "rmu_enet_success_cntr"
	},
	[C_CNTR_RMU_MEM_COR_ERR_CNTR] = {
		.name = "rmu_mem_cor_err_cntr"
	},
	[C_CNTR_RMU_MEM_UCOR_ERR_CNTR] = {
		.name = "rmu_mem_ucor_err_cntr"
	},
	[C_CNTR_RMU_PTL_INVLD_DFA_CNTR] = {
		.name = "rmu_ptl_invld_dfa_cntr"
	},
	[C_CNTR_RMU_PTL_INVLD_VNI_CNTR] = {
		.name = "rmu_ptl_invld_vni_cntr"
	},
	[C_CNTR_RMU_PTL_NO_PTLTE_CNTR] = {
		.name = "rmu_ptl_no_ptlte_cntr"
	},
	[C_CNTR_RMU_ENET_FRAME_REJECTED_CNTR] = {
		.name = "rmu_enet_frame_rejected_cntr"
	},
	[C_CNTR_RMU_ENET_PTLTE_SET_CTRL_ERR_CNTR] = {
		.name = "rmu_enet_ptlte_set_ctrl_err_cntr"
	},
	[C_CNTR_PARBS_TARB_CQ_POSTED_PKTS] = {
		.name = "parbs_tarb_cq_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_PKTS] = {
		.name = "parbs_tarb_cq_non_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_CQ_SBE_CNT] = {
		.name = "parbs_tarb_cq_sbe_cnt"
	},
	[C_CNTR_PARBS_TARB_CQ_MBE_CNT] = {
		.name = "parbs_tarb_cq_mbe_cnt"
	},
	[C_CNTR_PARBS_TARB_EE_POSTED_PKTS] = {
		.name = "parbs_tarb_ee_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_EE_SBE_CNT] = {
		.name = "parbs_tarb_ee_sbe_cnt"
	},
	[C_CNTR_PARBS_TARB_EE_MBE_CNT] = {
		.name = "parbs_tarb_ee_mbe_cnt"
	},
	[C_CNTR_PARBS_TARB_IXE_POSTED_PKTS] = {
		.name = "parbs_tarb_ixe_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_PKTS] = {
		.name = "parbs_tarb_ixe_non_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_IXE_SBE_CNT] = {
		.name = "parbs_tarb_ixe_sbe_cnt"
	},
	[C_CNTR_PARBS_TARB_IXE_MBE_CNT] = {
		.name = "parbs_tarb_ixe_mbe_cnt"
	},
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_PKTS] = {
		.name = "parbs_tarb_oxe_non_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_OXE_SBE_CNT] = {
		.name = "parbs_tarb_oxe_sbe_cnt"
	},
	[C_CNTR_PARBS_TARB_OXE_MBE_CNT] = {
		.name = "parbs_tarb_oxe_mbe_cnt"
	},
	[C_CNTR_PARBS_TARB_ATU_POSTED_PKTS] = {
		.name = "parbs_tarb_atu_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_PKTS] = {
		.name = "parbs_tarb_atu_non_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_ATU_SBE_CNT] = {
		.name = "parbs_tarb_atu_sbe_cnt"
	},
	[C_CNTR_PARBS_TARB_ATU_MBE_CNT] = {
		.name = "parbs_tarb_atu_mbe_cnt"
	},
	[C_CNTR_PARBS_TARB_PI_POSTED_PKTS] = {
		.name = "parbs_tarb_pi_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_PI_NON_POSTED_PKTS] = {
		.name = "parbs_tarb_pi_non_posted_pkts"
	},
	[C_CNTR_PARBS_TARB_PI_POSTED_BLOCKED_CNT] = {
		.name = "parbs_tarb_pi_posted_blocked_cnt"
	},
	[C_CNTR_PARBS_TARB_PI_NON_POSTED_BLOCKED_CNT] = {
		.name = "parbs_tarb_pi_non_posted_blocked_cnt"
	},
	[C_CNTR_PARBS_TARB_CQ_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_cq_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_cq_non_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_EE_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_ee_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_IXE_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_ixe_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_ixe_non_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_oxe_non_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_ATU_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_atu_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_FIFO_SBE] = {
		.name = "parbs_tarb_atu_non_posted_fifo_sbe"
	},
	[C_CNTR_PARBS_TARB_CQ_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_cq_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_cq_non_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_EE_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_ee_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_IXE_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_ixe_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_ixe_non_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_oxe_non_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_ATU_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_atu_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_FIFO_MBE] = {
		.name = "parbs_tarb_atu_non_posted_fifo_mbe"
	},
	[C_CNTR_PARBS_RARB_PI_POSTED_PKTS] = {
		.name = "parbs_rarb_pi_posted_pkts"
	},
	[C_CNTR_PARBS_RARB_PI_COMPLETION_PKTS] = {
		.name = "parbs_rarb_pi_completion_pkts"
	},
	[C_CNTR_PARBS_RARB_PI_SBE_CNT] = {
		.name = "parbs_rarb_pi_sbe_cnt"
	},
	[C_CNTR_PARBS_RARB_PI_MBE_CNT] = {
		.name = "parbs_rarb_pi_mbe_cnt"
	},
	[C_CNTR_PARBS_RARB_CQ_POSTED_PKTS] = {
		.name = "parbs_rarb_cq_posted_pkts"
	},
	[C_CNTR_PARBS_RARB_CQ_COMPLETION_PKTS] = {
		.name = "parbs_rarb_cq_completion_pkts"
	},
	[C_CNTR_PARBS_RARB_EE_POSTED_PKTS] = {
		.name = "parbs_rarb_ee_posted_pkts"
	},
	[C_CNTR_PARBS_RARB_IXE_POSTED_PKTS] = {
		.name = "parbs_rarb_ixe_posted_pkts"
	},
	[C_CNTR_PARBS_RARB_IXE_COMPLETION_PKTS] = {
		.name = "parbs_rarb_ixe_completion_pkts"
	},
	[C_CNTR_PARBS_RARB_OXE_COMPLETION_PKTS] = {
		.name = "parbs_rarb_oxe_completion_pkts"
	},
	[C_CNTR_PARBS_RARB_ATU_POSTED_PKTS] = {
		.name = "parbs_rarb_atu_posted_pkts"
	},
	[C_CNTR_PARBS_RARB_ATU_COMPLETION_PKTS] = {
		.name = "parbs_rarb_atu_completion_pkts"
	},
	[C_CNTR_MST_MEM_COR_ERR_CNTR] = {
		.name = "mst_mem_cor_err_cntr"
	},
	[C_CNTR_MST_MEM_UCOR_ERR_CNTR] = {
		.name = "mst_mem_ucor_err_cntr"
	},
	[C_CNTR_MST_ALLOCATIONS] = {
		.name = "mst_allocations"
	},
	[C_CNTR_MST_REQUESTS] = {
		.name = "mst_requests"
	},
	[C_CNTR_MST_STALLED_WAITING_FOR_LPE] = {
		.name = "mst_stalled_waiting_for_lpe"
	},
	[C_CNTR_MST_STALLED_WAITING_GET_CRDTS] = {
		.name = "mst_stalled_waiting_get_crdts"
	},
	[C_CNTR_MST_STALLED_WAITING_PUT_CRDTS] = {
		.name = "mst_stalled_waiting_put_crdts"
	},
	[C_CNTR_MST_STALLED_WAITING_EE_CRDTS] = {
		.name = "mst_stalled_waiting_ee_crdts"
	},
	[C_CNTR_MST_ERR_CANCELLED] = {
		.name = "mst_err_cancelled"
	},
	[C_CNTR_HNI_COR_ECC_ERR_CNTR] = {
		.name = "hni_cor_ecc_err_cntr"
	},
	[C_CNTR_HNI_UCOR_ECC_ERR_CNTR] = {
		.name = "hni_ucor_ecc_err_cntr"
	},
	[C_CNTR_HNI_TX_STALL_LLR] = {
		.name = "hni_tx_stall_llr"
	},
	[C_CNTR_HNI_RX_STALL_IXE_FIFO] = {
		.name = "hni_rx_stall_ixe_fifo"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_0] = {
		.name = "hni_rx_stall_ixe_pktbuf_0"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_1] = {
		.name = "hni_rx_stall_ixe_pktbuf_1"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_2] = {
		.name = "hni_rx_stall_ixe_pktbuf_2"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_3] = {
		.name = "hni_rx_stall_ixe_pktbuf_3"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_4] = {
		.name = "hni_rx_stall_ixe_pktbuf_4"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_5] = {
		.name = "hni_rx_stall_ixe_pktbuf_5"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_6] = {
		.name = "hni_rx_stall_ixe_pktbuf_6"
	},
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_7] = {
		.name = "hni_rx_stall_ixe_pktbuf_7"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_0] = {
		.name = "hni_pfc_fifo_oflw_cntr_0"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_1] = {
		.name = "hni_pfc_fifo_oflw_cntr_1"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_2] = {
		.name = "hni_pfc_fifo_oflw_cntr_2"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_3] = {
		.name = "hni_pfc_fifo_oflw_cntr_3"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_4] = {
		.name = "hni_pfc_fifo_oflw_cntr_4"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_5] = {
		.name = "hni_pfc_fifo_oflw_cntr_5"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_6] = {
		.name = "hni_pfc_fifo_oflw_cntr_6"
	},
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_7] = {
		.name = "hni_pfc_fifo_oflw_cntr_7"
	},
	[C_CNTR_HNI_DISCARD_CNTR_0] = {
		.name = "hni_discard_cntr_0"
	},
	[C_CNTR_HNI_DISCARD_CNTR_1] = {
		.name = "hni_discard_cntr_1"
	},
	[C_CNTR_HNI_DISCARD_CNTR_2] = {
		.name = "hni_discard_cntr_2"
	},
	[C_CNTR_HNI_DISCARD_CNTR_3] = {
		.name = "hni_discard_cntr_3"
	},
	[C_CNTR_HNI_DISCARD_CNTR_4] = {
		.name = "hni_discard_cntr_4"
	},
	[C_CNTR_HNI_DISCARD_CNTR_5] = {
		.name = "hni_discard_cntr_5"
	},
	[C_CNTR_HNI_DISCARD_CNTR_6] = {
		.name = "hni_discard_cntr_6"
	},
	[C_CNTR_HNI_DISCARD_CNTR_7] = {
		.name = "hni_discard_cntr_7"
	},
	[C_CNTR_HNI_TX_OK_IEEE] = {
		.name = "hni_tx_ok_ieee"
	},
	[C_CNTR_HNI_TX_OK_OPT] = {
		.name = "hni_tx_ok_opt"
	},
	[C_CNTR_HNI_TX_POISONED_IEEE] = {
		.name = "hni_tx_poisoned_ieee"
	},
	[C_CNTR_HNI_TX_POISONED_OPT] = {
		.name = "hni_tx_poisoned_opt"
	},
	[C_CNTR_HNI_TX_OK_BROADCAST] = {
		.name = "hni_tx_ok_broadcast"
	},
	[C_CNTR_HNI_TX_OK_MULTICAST] = {
		.name = "hni_tx_ok_multicast"
	},
	[C_CNTR_HNI_TX_OK_TAGGED] = {
		.name = "hni_tx_ok_tagged"
	},
	[C_CNTR_HNI_TX_OK_27] = {
		.name = "hni_tx_ok_27"
	},
	[C_CNTR_HNI_TX_OK_35] = {
		.name = "hni_tx_ok_35"
	},
	[C_CNTR_HNI_TX_OK_36_TO_63] = {
		.name = "hni_tx_ok_36_to_63"
	},
	[C_CNTR_HNI_TX_OK_64] = {
		.name = "hni_tx_ok_64"
	},
	[C_CNTR_HNI_TX_OK_65_TO_127] = {
		.name = "hni_tx_ok_65_to_127"
	},
	[C_CNTR_HNI_TX_OK_128_TO_255] = {
		.name = "hni_tx_ok_128_to_255"
	},
	[C_CNTR_HNI_TX_OK_256_TO_511] = {
		.name = "hni_tx_ok_256_to_511"
	},
	[C_CNTR_HNI_TX_OK_512_TO_1023] = {
		.name = "hni_tx_ok_512_to_1023"
	},
	[C_CNTR_HNI_TX_OK_1024_TO_2047] = {
		.name = "hni_tx_ok_1024_to_2047"
	},
	[C_CNTR_HNI_TX_OK_2048_TO_4095] = {
		.name = "hni_tx_ok_2048_to_4095"
	},
	[C_CNTR_HNI_TX_OK_4096_TO_8191] = {
		.name = "hni_tx_ok_4096_to_8191"
	},
	[C_CNTR_HNI_TX_OK_8192_TO_MAX] = {
		.name = "hni_tx_ok_8192_to_max"
	},
	[C_CNTR_HNI_RX_OK_IEEE] = {
		.name = "hni_rx_ok_ieee"
	},
	[C_CNTR_HNI_RX_OK_OPT] = {
		.name = "hni_rx_ok_opt"
	},
	[C_CNTR_HNI_RX_BAD_IEEE] = {
		.name = "hni_rx_bad_ieee"
	},
	[C_CNTR_HNI_RX_BAD_OPT] = {
		.name = "hni_rx_bad_opt"
	},
	[C_CNTR_HNI_RX_OK_BROADCAST] = {
		.name = "hni_rx_ok_broadcast"
	},
	[C_CNTR_HNI_RX_OK_MULTICAST] = {
		.name = "hni_rx_ok_multicast"
	},
	[C_CNTR_HNI_RX_OK_MAC_CONTROL] = {
		.name = "hni_rx_ok_mac_control"
	},
	[C_CNTR_HNI_RX_OK_BAD_OP_CODE] = {
		.name = "hni_rx_ok_bad_op_code"
	},
	[C_CNTR_HNI_RX_OK_FLOW_CONTROL] = {
		.name = "hni_rx_ok_flow_control"
	},
	[C_CNTR_HNI_RX_OK_TAGGED] = {
		.name = "hni_rx_ok_tagged"
	},
	[C_CNTR_HNI_RX_OK_LEN_TYPE_ERROR] = {
		.name = "hni_rx_ok_len_type_error"
	},
	[C_CNTR_HNI_RX_OK_TOO_LONG] = {
		.name = "hni_rx_ok_too_long"
	},
	[C_CNTR_HNI_RX_OK_UNDERSIZE] = {
		.name = "hni_rx_ok_undersize"
	},
	[C_CNTR_HNI_RX_FRAGMENT] = {
		.name = "hni_rx_fragment"
	},
	[C_CNTR_HNI_RX_JABBER] = {
		.name = "hni_rx_jabber"
	},
	[C_CNTR_HNI_RX_OK_27] = {
		.name = "hni_rx_ok_27"
	},
	[C_CNTR_HNI_RX_OK_35] = {
		.name = "hni_rx_ok_35"
	},
	[C_CNTR_HNI_RX_OK_36_TO_63] = {
		.name = "hni_rx_ok_36_to_63"
	},
	[C_CNTR_HNI_RX_OK_64] = {
		.name = "hni_rx_ok_64"
	},
	[C_CNTR_HNI_RX_OK_65_TO_127] = {
		.name = "hni_rx_ok_65_to_127"
	},
	[C_CNTR_HNI_RX_OK_128_TO_255] = {
		.name = "hni_rx_ok_128_to_255"
	},
	[C_CNTR_HNI_RX_OK_256_TO_511] = {
		.name = "hni_rx_ok_256_to_511"
	},
	[C_CNTR_HNI_RX_OK_512_TO_1023] = {
		.name = "hni_rx_ok_512_to_1023"
	},
	[C_CNTR_HNI_RX_OK_1024_TO_2047] = {
		.name = "hni_rx_ok_1024_to_2047"
	},
	[C_CNTR_HNI_RX_OK_2048_TO_4095] = {
		.name = "hni_rx_ok_2048_to_4095"
	},
	[C_CNTR_HNI_RX_OK_4096_TO_8191] = {
		.name = "hni_rx_ok_4096_to_8191"
	},
	[C_CNTR_HNI_RX_OK_8192_TO_MAX] = {
		.name = "hni_rx_ok_8192_to_max"
	},
	[C_CNTR_HNI_HRP_ACK] = {
		.name = "hni_hrp_ack"
	},
	[C_CNTR_HNI_HRP_RESP] = {
		.name = "hni_hrp_resp"
	},
	[C_CNTR_HNI_FGFC_PORT] = {
		.name = "hni_fgfc_port"
	},
	[C_CNTR_HNI_FGFC_L2] = {
		.name = "hni_fgfc_l2"
	},
	[C_CNTR_HNI_FGFC_IPV4] = {
		.name = "hni_fgfc_ipv4"
	},
	[C_CNTR_HNI_FGFC_IPV6] = {
		.name = "hni_fgfc_ipv6"
	},
	[C_CNTR_HNI_FGFC_MATCH] = {
		.name = "hni_fgfc_match"
	},
	[C_CNTR_HNI_FGFC_EVENT_XON] = {
		.name = "hni_fgfc_event_xon"
	},
	[C_CNTR_HNI_FGFC_EVENT_XOFF] = {
		.name = "hni_fgfc_event_xoff"
	},
	[C_CNTR_HNI_FGFC_DISCARD] = {
		.name = "hni_fgfc_discard"
	},
	[C_CNTR_HNI_PAUSE_RECV_0] = {
		.name = "hni_pause_recv_0"
	},
	[C_CNTR_HNI_PAUSE_RECV_1] = {
		.name = "hni_pause_recv_1"
	},
	[C_CNTR_HNI_PAUSE_RECV_2] = {
		.name = "hni_pause_recv_2"
	},
	[C_CNTR_HNI_PAUSE_RECV_3] = {
		.name = "hni_pause_recv_3"
	},
	[C_CNTR_HNI_PAUSE_RECV_4] = {
		.name = "hni_pause_recv_4"
	},
	[C_CNTR_HNI_PAUSE_RECV_5] = {
		.name = "hni_pause_recv_5"
	},
	[C_CNTR_HNI_PAUSE_RECV_6] = {
		.name = "hni_pause_recv_6"
	},
	[C_CNTR_HNI_PAUSE_RECV_7] = {
		.name = "hni_pause_recv_7"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_0] = {
		.name = "hni_pause_xoff_sent_0"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_1] = {
		.name = "hni_pause_xoff_sent_1"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_2] = {
		.name = "hni_pause_xoff_sent_2"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_3] = {
		.name = "hni_pause_xoff_sent_3"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_4] = {
		.name = "hni_pause_xoff_sent_4"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_5] = {
		.name = "hni_pause_xoff_sent_5"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_6] = {
		.name = "hni_pause_xoff_sent_6"
	},
	[C_CNTR_HNI_PAUSE_XOFF_SENT_7] = {
		.name = "hni_pause_xoff_sent_7"
	},
	[C_CNTR_HNI_RX_PAUSED_0] = {
		.name = "hni_rx_paused_0"
	},
	[C_CNTR_HNI_RX_PAUSED_1] = {
		.name = "hni_rx_paused_1"
	},
	[C_CNTR_HNI_RX_PAUSED_2] = {
		.name = "hni_rx_paused_2"
	},
	[C_CNTR_HNI_RX_PAUSED_3] = {
		.name = "hni_rx_paused_3"
	},
	[C_CNTR_HNI_RX_PAUSED_4] = {
		.name = "hni_rx_paused_4"
	},
	[C_CNTR_HNI_RX_PAUSED_5] = {
		.name = "hni_rx_paused_5"
	},
	[C_CNTR_HNI_RX_PAUSED_6] = {
		.name = "hni_rx_paused_6"
	},
	[C_CNTR_HNI_RX_PAUSED_7] = {
		.name = "hni_rx_paused_7"
	},
	[C_CNTR_HNI_TX_PAUSED_0] = {
		.name = "hni_tx_paused_0"
	},
	[C_CNTR_HNI_TX_PAUSED_1] = {
		.name = "hni_tx_paused_1"
	},
	[C_CNTR_HNI_TX_PAUSED_2] = {
		.name = "hni_tx_paused_2"
	},
	[C_CNTR_HNI_TX_PAUSED_3] = {
		.name = "hni_tx_paused_3"
	},
	[C_CNTR_HNI_TX_PAUSED_4] = {
		.name = "hni_tx_paused_4"
	},
	[C_CNTR_HNI_TX_PAUSED_5] = {
		.name = "hni_tx_paused_5"
	},
	[C_CNTR_HNI_TX_PAUSED_6] = {
		.name = "hni_tx_paused_6"
	},
	[C_CNTR_HNI_TX_PAUSED_7] = {
		.name = "hni_tx_paused_7"
	},
	[C_CNTR_HNI_RX_PAUSED_STD] = {
		.name = "hni_rx_paused_std"
	},
	[C_CNTR_HNI_PAUSE_SENT] = {
		.name = "hni_pause_sent"
	},
	[C_CNTR_HNI_PAUSE_REFRESH] = {
		.name = "hni_pause_refresh"
	},
	[C_CNTR_HNI_TX_STD_PADDED] = {
		.name = "hni_tx_std_padded"
	},
	[C_CNTR_HNI_TX_OPT_PADDED] = {
		.name = "hni_tx_opt_padded"
	},
	[C_CNTR_HNI_TX_STD_SIZE_ERR] = {
		.name = "hni_tx_std_size_err"
	},
	[C_CNTR_HNI_TX_OPT_SIZE_ERR] = {
		.name = "hni_tx_opt_size_err"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_0] = {
		.name = "hni_pkts_sent_by_tc_0"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_1] = {
		.name = "hni_pkts_sent_by_tc_1"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_2] = {
		.name = "hni_pkts_sent_by_tc_2"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_3] = {
		.name = "hni_pkts_sent_by_tc_3"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_4] = {
		.name = "hni_pkts_sent_by_tc_4"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_5] = {
		.name = "hni_pkts_sent_by_tc_5"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_6] = {
		.name = "hni_pkts_sent_by_tc_6"
	},
	[C_CNTR_HNI_PKTS_SENT_BY_TC_7] = {
		.name = "hni_pkts_sent_by_tc_7"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_0] = {
		.name = "hni_pkts_recv_by_tc_0"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_1] = {
		.name = "hni_pkts_recv_by_tc_1"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_2] = {
		.name = "hni_pkts_recv_by_tc_2"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_3] = {
		.name = "hni_pkts_recv_by_tc_3"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_4] = {
		.name = "hni_pkts_recv_by_tc_4"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_5] = {
		.name = "hni_pkts_recv_by_tc_5"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_6] = {
		.name = "hni_pkts_recv_by_tc_6"
	},
	[C_CNTR_HNI_PKTS_RECV_BY_TC_7] = {
		.name = "hni_pkts_recv_by_tc_7"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_0] = {
		.name = "hni_multicast_pkts_sent_by_tc_0"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_1] = {
		.name = "hni_multicast_pkts_sent_by_tc_1"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_2] = {
		.name = "hni_multicast_pkts_sent_by_tc_2"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_3] = {
		.name = "hni_multicast_pkts_sent_by_tc_3"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_4] = {
		.name = "hni_multicast_pkts_sent_by_tc_4"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_5] = {
		.name = "hni_multicast_pkts_sent_by_tc_5"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_6] = {
		.name = "hni_multicast_pkts_sent_by_tc_6"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_7] = {
		.name = "hni_multicast_pkts_sent_by_tc_7"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_0] = {
		.name = "hni_multicast_pkts_recv_by_tc_0"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_1] = {
		.name = "hni_multicast_pkts_recv_by_tc_1"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_2] = {
		.name = "hni_multicast_pkts_recv_by_tc_2"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_3] = {
		.name = "hni_multicast_pkts_recv_by_tc_3"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_4] = {
		.name = "hni_multicast_pkts_recv_by_tc_4"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_5] = {
		.name = "hni_multicast_pkts_recv_by_tc_5"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_6] = {
		.name = "hni_multicast_pkts_recv_by_tc_6"
	},
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_7] = {
		.name = "hni_multicast_pkts_recv_by_tc_7"
	},
	[C1_CNTR_HNI_PCS_PML_MBE_ERROR_CNT] = {
		.name = "hni_pcs_pml_mbe_error_cnt"
	},
	[C1_CNTR_HNI_PCS_PML_SBE_ERROR_CNT] = {
		.name = "hni_pcs_pml_sbe_error_cnt"
	},
	[C1_CNTR_HNI_PCS_CORRECTED_CW] = {
		.name = "hni_pcs_corrected_cw"
	},
	[C1_CNTR_HNI_PCS_UNCORRECTED_CW] = {
		.name = "hni_pcs_uncorrected_cw"
	},
	[C1_CNTR_HNI_PCS_GOOD_CW] = {
		.name = "hni_pcs_good_cw"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_0] = {
		.name = "hni_pcs_fecl_errors_0"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_1] = {
		.name = "hni_pcs_fecl_errors_1"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_2] = {
		.name = "hni_pcs_fecl_errors_2"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_3] = {
		.name = "hni_pcs_fecl_errors_3"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_4] = {
		.name = "hni_pcs_fecl_errors_4"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_5] = {
		.name = "hni_pcs_fecl_errors_5"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_6] = {
		.name = "hni_pcs_fecl_errors_6"
	},
	[C1_CNTR_HNI_PCS_FECL_ERRORS_7] = {
		.name = "hni_pcs_fecl_errors_7"
	},
	[C1_CNTR_HNI_MAC_RX_ILLEGAL_SIZE] = {
		.name = "hni_mac_rx_illegal_size"
	},
	[C1_CNTR_HNI_MAC_RX_FRAGMENT] = {
		.name = "hni_mac_rx_fragment"
	},
	[C1_CNTR_HNI_MAC_RX_PREAMBLE_ERROR] = {
		.name = "hni_mac_rx_preamble_error"
	},
	[C1_CNTR_HNI_LLR_TX_LOOP_TIME_REQ_CTL_FR] = {
		.name = "hni_llr_tx_loop_time_req_ctl_fr"
	},
	[C1_CNTR_HNI_LLR_TX_LOOP_TIME_RSP_CTL_FR] = {
		.name = "hni_llr_tx_loop_time_rsp_ctl_fr"
	},
	[C1_CNTR_HNI_LLR_TX_INIT_CTL_OS] = {
		.name = "hni_llr_tx_init_ctl_os"
	},
	[C1_CNTR_HNI_LLR_TX_INIT_ECHO_CTL_OS] = {
		.name = "hni_llr_tx_init_echo_ctl_os"
	},
	[C1_CNTR_HNI_LLR_TX_ACK_CTL_OS] = {
		.name = "hni_llr_tx_ack_ctl_os"
	},
	[C1_CNTR_HNI_LLR_TX_NACK_CTL_OS] = {
		.name = "hni_llr_tx_nack_ctl_os"
	},
	[C1_CNTR_HNI_LLR_TX_DISCARD] = {
		.name = "hni_llr_tx_discard"
	},
	[C1_CNTR_HNI_LLR_TX_OK_LOSSY] = {
		.name = "hni_llr_tx_ok_lossy"
	},
	[C1_CNTR_HNI_LLR_TX_OK_LOSSLESS] = {
		.name = "hni_llr_tx_ok_lossless"
	},
	[C1_CNTR_HNI_LLR_TX_OK_LOSSLESS_RPT] = {
		.name = "hni_llr_tx_ok_lossless_rpt"
	},
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSY] = {
		.name = "hni_llr_tx_poisoned_lossy"
	},
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSLESS] = {
		.name = "hni_llr_tx_poisoned_lossless"
	},
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSLESS_RPT] = {
		.name = "hni_llr_tx_poisoned_lossless_rpt"
	},
	[C1_CNTR_HNI_LLR_TX_OK_BYPASS] = {
		.name = "hni_llr_tx_ok_bypass"
	},
	[C1_CNTR_HNI_LLR_TX_REPLAY_EVENT] = {
		.name = "hni_llr_tx_replay_event"
	},
	[C1_CNTR_HNI_LLR_RX_LOOP_TIME_REQ_CTL_FR] = {
		.name = "hni_llr_rx_loop_time_req_ctl_fr"
	},
	[C1_CNTR_HNI_LLR_RX_LOOP_TIME_RSP_CTL_FR] = {
		.name = "hni_llr_rx_loop_time_rsp_ctl_fr"
	},
	[C1_CNTR_HNI_LLR_RX_BAD_CTL_FR] = {
		.name = "hni_llr_rx_bad_ctl_fr"
	},
	[C1_CNTR_HNI_LLR_RX_INIT_CTL_OS] = {
		.name = "hni_llr_rx_init_ctl_os"
	},
	[C1_CNTR_HNI_LLR_RX_INIT_ECHO_CTL_OS] = {
		.name = "hni_llr_rx_init_echo_ctl_os"
	},
	[C1_CNTR_HNI_LLR_RX_ACK_CTL_OS] = {
		.name = "hni_llr_rx_ack_ctl_os"
	},
	[C1_CNTR_HNI_LLR_RX_NACK_CTL_OS] = {
		.name = "hni_llr_rx_nack_ctl_os"
	},
	[C1_CNTR_HNI_LLR_RX_ACK_NACK_SEQ_ERR] = {
		.name = "hni_llr_rx_ack_nack_seq_err"
	},
	[C1_CNTR_HNI_LLR_RX_OK_LOSSY] = {
		.name = "hni_llr_rx_ok_lossy"
	},
	[C1_CNTR_HNI_LLR_RX_POISONED_LOSSY] = {
		.name = "hni_llr_rx_poisoned_lossy"
	},
	[C1_CNTR_HNI_LLR_RX_BAD_LOSSY] = {
		.name = "hni_llr_rx_bad_lossy"
	},
	[C1_CNTR_HNI_LLR_RX_OK_LOSSLESS] = {
		.name = "hni_llr_rx_ok_lossless"
	},
	[C1_CNTR_HNI_LLR_RX_POISONED_LOSSLESS] = {
		.name = "hni_llr_rx_poisoned_lossless"
	},
	[C1_CNTR_HNI_LLR_RX_BAD_LOSSLESS] = {
		.name = "hni_llr_rx_bad_lossless"
	},
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_GOOD] = {
		.name = "hni_llr_rx_expected_seq_good"
	},
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_POISONED] = {
		.name = "hni_llr_rx_expected_seq_poisoned"
	},
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_BAD] = {
		.name = "hni_llr_rx_expected_seq_bad"
	},
	[C1_CNTR_HNI_LLR_RX_UNEXPECTED_SEQ] = {
		.name = "hni_llr_rx_unexpected_seq"
	},
	[C1_CNTR_HNI_LLR_RX_DUPLICATE_SEQ] = {
		.name = "hni_llr_rx_duplicate_seq"
	},
	[C1_CNTR_HNI_LLR_RX_REPLAY_EVENT] = {
		.name = "hni_llr_rx_replay_event"
	},
	[C1_CNTR_OXE_MEM_COR_ERR_CNTR] = {
		.name = "oxe_mem_cor_err_cntr"
	},
	[C1_CNTR_OXE_MEM_UCOR_ERR_CNTR] = {
		.name = "oxe_mem_ucor_err_cntr"
	},
	[C1_CNTR_OXE_ERR_NO_TRANSLATION] = {
		.name = "oxe_err_no_translation"
	},
	[C1_CNTR_OXE_ERR_INVALID_AC] = {
		.name = "oxe_err_invalid_ac"
	},
	[C1_CNTR_OXE_ERR_PAGE_PERM] = {
		.name = "oxe_err_page_perm"
	},
	[C1_CNTR_OXE_ERR_TA_ERROR] = {
		.name = "oxe_err_ta_error"
	},
	[C1_CNTR_OXE_ERR_PAGE_REQ] = {
		.name = "oxe_err_page_req"
	},
	[C1_CNTR_OXE_ERR_PCI_EP] = {
		.name = "oxe_err_pci_ep"
	},
	[C1_CNTR_OXE_ERR_PCI_CMPL] = {
		.name = "oxe_err_pci_cmpl"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_0] = {
		.name = "oxe_stall_pct_bc_0"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_1] = {
		.name = "oxe_stall_pct_bc_1"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_2] = {
		.name = "oxe_stall_pct_bc_2"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_3] = {
		.name = "oxe_stall_pct_bc_3"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_4] = {
		.name = "oxe_stall_pct_bc_4"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_5] = {
		.name = "oxe_stall_pct_bc_5"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_6] = {
		.name = "oxe_stall_pct_bc_6"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_7] = {
		.name = "oxe_stall_pct_bc_7"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_8] = {
		.name = "oxe_stall_pct_bc_8"
	},
	[C1_CNTR_OXE_STALL_PCT_BC_9] = {
		.name = "oxe_stall_pct_bc_9"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_0] = {
		.name = "oxe_stall_pbuf_bc_0"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_1] = {
		.name = "oxe_stall_pbuf_bc_1"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_2] = {
		.name = "oxe_stall_pbuf_bc_2"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_3] = {
		.name = "oxe_stall_pbuf_bc_3"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_4] = {
		.name = "oxe_stall_pbuf_bc_4"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_5] = {
		.name = "oxe_stall_pbuf_bc_5"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_6] = {
		.name = "oxe_stall_pbuf_bc_6"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_7] = {
		.name = "oxe_stall_pbuf_bc_7"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_8] = {
		.name = "oxe_stall_pbuf_bc_8"
	},
	[C1_CNTR_OXE_STALL_PBUF_BC_9] = {
		.name = "oxe_stall_pbuf_bc_9"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_0] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_0"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_1] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_1"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_2] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_2"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_3] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_3"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_4] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_4"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_5] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_5"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_6] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_6"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_7] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_7"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_8] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_8"
	},
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_9] = {
		.name = "oxe_stall_ts_no_out_crd_tsc_9"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_0] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_0"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_1] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_1"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_2] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_2"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_3] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_3"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_4] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_4"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_5] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_5"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_6] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_6"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_7] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_7"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_8] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_8"
	},
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_9] = {
		.name = "oxe_stall_ts_no_in_crd_tsc_9"
	},
	[C1_CNTR_OXE_STALL_WITH_NO_PCI_TAG] = {
		.name = "oxe_stall_with_no_pci_tag"
	},
	[C1_CNTR_OXE_STALL_WITH_NO_ATU_TAG] = {
		.name = "oxe_stall_with_no_atu_tag"
	},
	[C1_CNTR_OXE_STALL_STFWD_EOP] = {
		.name = "oxe_stall_stfwd_eop"
	},
	[C1_CNTR_OXE_STALL_PCIE_SCOREBOARD] = {
		.name = "oxe_stall_pcie_scoreboard"
	},
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_0] = {
		.name = "oxe_stall_wr_conflict_pkt_buff_bnk_0"
	},
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_1] = {
		.name = "oxe_stall_wr_conflict_pkt_buff_bnk_1"
	},
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_2] = {
		.name = "oxe_stall_wr_conflict_pkt_buff_bnk_2"
	},
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_3] = {
		.name = "oxe_stall_wr_conflict_pkt_buff_bnk_3"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_0] = {
		.name = "oxe_stall_idc_no_buff_bc_0"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_1] = {
		.name = "oxe_stall_idc_no_buff_bc_1"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_2] = {
		.name = "oxe_stall_idc_no_buff_bc_2"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_3] = {
		.name = "oxe_stall_idc_no_buff_bc_3"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_4] = {
		.name = "oxe_stall_idc_no_buff_bc_4"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_5] = {
		.name = "oxe_stall_idc_no_buff_bc_5"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_6] = {
		.name = "oxe_stall_idc_no_buff_bc_6"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_7] = {
		.name = "oxe_stall_idc_no_buff_bc_7"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_8] = {
		.name = "oxe_stall_idc_no_buff_bc_8"
	},
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_9] = {
		.name = "oxe_stall_idc_no_buff_bc_9"
	},
	[C1_CNTR_OXE_STALL_IDC_CMD_NO_DEQ] = {
		.name = "oxe_stall_idc_cmd_no_deq"
	},
	[C1_CNTR_OXE_STALL_NON_IDC_CMD_NO_DEQ] = {
		.name = "oxe_stall_non_idc_cmd_no_deq"
	},
	[C1_CNTR_OXE_STALL_FGFC_BLK_0] = {
		.name = "oxe_stall_fgfc_blk_0"
	},
	[C1_CNTR_OXE_STALL_FGFC_BLK_1] = {
		.name = "oxe_stall_fgfc_blk_1"
	},
	[C1_CNTR_OXE_STALL_FGFC_BLK_2] = {
		.name = "oxe_stall_fgfc_blk_2"
	},
	[C1_CNTR_OXE_STALL_FGFC_BLK_3] = {
		.name = "oxe_stall_fgfc_blk_3"
	},
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_0] = {
		.name = "oxe_stall_fgfc_cntrl_0"
	},
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_1] = {
		.name = "oxe_stall_fgfc_cntrl_1"
	},
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_2] = {
		.name = "oxe_stall_fgfc_cntrl_2"
	},
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_3] = {
		.name = "oxe_stall_fgfc_cntrl_3"
	},
	[C1_CNTR_OXE_STALL_FGFC_START_0] = {
		.name = "oxe_stall_fgfc_start_0"
	},
	[C1_CNTR_OXE_STALL_FGFC_START_1] = {
		.name = "oxe_stall_fgfc_start_1"
	},
	[C1_CNTR_OXE_STALL_FGFC_START_2] = {
		.name = "oxe_stall_fgfc_start_2"
	},
	[C1_CNTR_OXE_STALL_FGFC_START_3] = {
		.name = "oxe_stall_fgfc_start_3"
	},
	[C1_CNTR_OXE_STALL_FGFC_END_0] = {
		.name = "oxe_stall_fgfc_end_0"
	},
	[C1_CNTR_OXE_STALL_FGFC_END_1] = {
		.name = "oxe_stall_fgfc_end_1"
	},
	[C1_CNTR_OXE_STALL_FGFC_END_2] = {
		.name = "oxe_stall_fgfc_end_2"
	},
	[C1_CNTR_OXE_STALL_FGFC_END_3] = {
		.name = "oxe_stall_fgfc_end_3"
	},
	[C1_CNTR_OXE_STALL_HDR_ARB] = {
		.name = "oxe_stall_hdr_arb"
	},
	[C1_CNTR_OXE_STALL_IOI_LAST_ORDERED] = {
		.name = "oxe_stall_ioi_last_ordered"
	},
	[C1_CNTR_OXE_IGNORE_ERRS] = {
		.name = "oxe_ignore_errs"
	},
	[C1_CNTR_OXE_IOI_DMA] = {
		.name = "oxe_ioi_dma"
	},
	[C1_CNTR_OXE_IOI_PKTS_ORDERED] = {
		.name = "oxe_ioi_pkts_ordered"
	},
	[C1_CNTR_OXE_IOI_PKTS_UNORDERED] = {
		.name = "oxe_ioi_pkts_unordered"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_0] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_0"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_1] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_1"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_2] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_2"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_3] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_3"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_4] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_4"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_5] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_5"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_6] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_6"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_7] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_7"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_8] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_8"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_9] = {
		.name = "oxe_ptl_tx_put_msgs_tsc_9"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_0] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_0"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_1] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_1"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_2] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_2"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_3] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_3"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_4] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_4"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_5] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_5"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_6] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_6"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_7] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_7"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_8] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_8"
	},
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_9] = {
		.name = "oxe_ptl_tx_get_msgs_tsc_9"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_0] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_0"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_1] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_1"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_2] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_2"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_3] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_3"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_4] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_4"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_5] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_5"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_6] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_6"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_7] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_7"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_8] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_8"
	},
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_9] = {
		.name = "oxe_ptl_tx_put_pkts_tsc_9"
	},
	[C1_CNTR_OXE_PTL_TX_MR_MSGS] = {
		.name = "oxe_ptl_tx_mr_msgs"
	},
	[C1_CNTR_OXE_NUM_HRP_CMDS] = {
		.name = "oxe_num_hrp_cmds"
	},
	[C1_CNTR_OXE_CHANNEL_IDLE] = {
		.name = "oxe_channel_idle"
	},
	[C1_CNTR_OXE_MCU_MEAS_0] = {
		.name = "oxe_mcu_meas_0"
	},
	[C1_CNTR_OXE_MCU_MEAS_1] = {
		.name = "oxe_mcu_meas_1"
	},
	[C1_CNTR_OXE_MCU_MEAS_2] = {
		.name = "oxe_mcu_meas_2"
	},
	[C1_CNTR_OXE_MCU_MEAS_3] = {
		.name = "oxe_mcu_meas_3"
	},
	[C1_CNTR_OXE_MCU_MEAS_4] = {
		.name = "oxe_mcu_meas_4"
	},
	[C1_CNTR_OXE_MCU_MEAS_5] = {
		.name = "oxe_mcu_meas_5"
	},
	[C1_CNTR_OXE_MCU_MEAS_6] = {
		.name = "oxe_mcu_meas_6"
	},
	[C1_CNTR_OXE_MCU_MEAS_7] = {
		.name = "oxe_mcu_meas_7"
	},
	[C1_CNTR_OXE_MCU_MEAS_8] = {
		.name = "oxe_mcu_meas_8"
	},
	[C1_CNTR_OXE_MCU_MEAS_9] = {
		.name = "oxe_mcu_meas_9"
	},
	[C1_CNTR_OXE_MCU_MEAS_10] = {
		.name = "oxe_mcu_meas_10"
	},
	[C1_CNTR_OXE_MCU_MEAS_11] = {
		.name = "oxe_mcu_meas_11"
	},
	[C1_CNTR_OXE_MCU_MEAS_12] = {
		.name = "oxe_mcu_meas_12"
	},
	[C1_CNTR_OXE_MCU_MEAS_13] = {
		.name = "oxe_mcu_meas_13"
	},
	[C1_CNTR_OXE_MCU_MEAS_14] = {
		.name = "oxe_mcu_meas_14"
	},
	[C1_CNTR_OXE_MCU_MEAS_15] = {
		.name = "oxe_mcu_meas_15"
	},
	[C1_CNTR_OXE_MCU_MEAS_16] = {
		.name = "oxe_mcu_meas_16"
	},
	[C1_CNTR_OXE_MCU_MEAS_17] = {
		.name = "oxe_mcu_meas_17"
	},
	[C1_CNTR_OXE_MCU_MEAS_18] = {
		.name = "oxe_mcu_meas_18"
	},
	[C1_CNTR_OXE_MCU_MEAS_19] = {
		.name = "oxe_mcu_meas_19"
	},
	[C1_CNTR_OXE_MCU_MEAS_20] = {
		.name = "oxe_mcu_meas_20"
	},
	[C1_CNTR_OXE_MCU_MEAS_21] = {
		.name = "oxe_mcu_meas_21"
	},
	[C1_CNTR_OXE_MCU_MEAS_22] = {
		.name = "oxe_mcu_meas_22"
	},
	[C1_CNTR_OXE_MCU_MEAS_23] = {
		.name = "oxe_mcu_meas_23"
	},
	[C1_CNTR_OXE_MCU_MEAS_24] = {
		.name = "oxe_mcu_meas_24"
	},
	[C1_CNTR_OXE_MCU_MEAS_25] = {
		.name = "oxe_mcu_meas_25"
	},
	[C1_CNTR_OXE_MCU_MEAS_26] = {
		.name = "oxe_mcu_meas_26"
	},
	[C1_CNTR_OXE_MCU_MEAS_27] = {
		.name = "oxe_mcu_meas_27"
	},
	[C1_CNTR_OXE_MCU_MEAS_28] = {
		.name = "oxe_mcu_meas_28"
	},
	[C1_CNTR_OXE_MCU_MEAS_29] = {
		.name = "oxe_mcu_meas_29"
	},
	[C1_CNTR_OXE_MCU_MEAS_30] = {
		.name = "oxe_mcu_meas_30"
	},
	[C1_CNTR_OXE_MCU_MEAS_31] = {
		.name = "oxe_mcu_meas_31"
	},
	[C1_CNTR_OXE_MCU_MEAS_32] = {
		.name = "oxe_mcu_meas_32"
	},
	[C1_CNTR_OXE_MCU_MEAS_33] = {
		.name = "oxe_mcu_meas_33"
	},
	[C1_CNTR_OXE_MCU_MEAS_34] = {
		.name = "oxe_mcu_meas_34"
	},
	[C1_CNTR_OXE_MCU_MEAS_35] = {
		.name = "oxe_mcu_meas_35"
	},
	[C1_CNTR_OXE_MCU_MEAS_36] = {
		.name = "oxe_mcu_meas_36"
	},
	[C1_CNTR_OXE_MCU_MEAS_37] = {
		.name = "oxe_mcu_meas_37"
	},
	[C1_CNTR_OXE_MCU_MEAS_38] = {
		.name = "oxe_mcu_meas_38"
	},
	[C1_CNTR_OXE_MCU_MEAS_39] = {
		.name = "oxe_mcu_meas_39"
	},
	[C1_CNTR_OXE_MCU_MEAS_40] = {
		.name = "oxe_mcu_meas_40"
	},
	[C1_CNTR_OXE_MCU_MEAS_41] = {
		.name = "oxe_mcu_meas_41"
	},
	[C1_CNTR_OXE_MCU_MEAS_42] = {
		.name = "oxe_mcu_meas_42"
	},
	[C1_CNTR_OXE_MCU_MEAS_43] = {
		.name = "oxe_mcu_meas_43"
	},
	[C1_CNTR_OXE_MCU_MEAS_44] = {
		.name = "oxe_mcu_meas_44"
	},
	[C1_CNTR_OXE_MCU_MEAS_45] = {
		.name = "oxe_mcu_meas_45"
	},
	[C1_CNTR_OXE_MCU_MEAS_46] = {
		.name = "oxe_mcu_meas_46"
	},
	[C1_CNTR_OXE_MCU_MEAS_47] = {
		.name = "oxe_mcu_meas_47"
	},
	[C1_CNTR_OXE_MCU_MEAS_48] = {
		.name = "oxe_mcu_meas_48"
	},
	[C1_CNTR_OXE_MCU_MEAS_49] = {
		.name = "oxe_mcu_meas_49"
	},
	[C1_CNTR_OXE_MCU_MEAS_50] = {
		.name = "oxe_mcu_meas_50"
	},
	[C1_CNTR_OXE_MCU_MEAS_51] = {
		.name = "oxe_mcu_meas_51"
	},
	[C1_CNTR_OXE_MCU_MEAS_52] = {
		.name = "oxe_mcu_meas_52"
	},
	[C1_CNTR_OXE_MCU_MEAS_53] = {
		.name = "oxe_mcu_meas_53"
	},
	[C1_CNTR_OXE_MCU_MEAS_54] = {
		.name = "oxe_mcu_meas_54"
	},
	[C1_CNTR_OXE_MCU_MEAS_55] = {
		.name = "oxe_mcu_meas_55"
	},
	[C1_CNTR_OXE_MCU_MEAS_56] = {
		.name = "oxe_mcu_meas_56"
	},
	[C1_CNTR_OXE_MCU_MEAS_57] = {
		.name = "oxe_mcu_meas_57"
	},
	[C1_CNTR_OXE_MCU_MEAS_58] = {
		.name = "oxe_mcu_meas_58"
	},
	[C1_CNTR_OXE_MCU_MEAS_59] = {
		.name = "oxe_mcu_meas_59"
	},
	[C1_CNTR_OXE_MCU_MEAS_60] = {
		.name = "oxe_mcu_meas_60"
	},
	[C1_CNTR_OXE_MCU_MEAS_61] = {
		.name = "oxe_mcu_meas_61"
	},
	[C1_CNTR_OXE_MCU_MEAS_62] = {
		.name = "oxe_mcu_meas_62"
	},
	[C1_CNTR_OXE_MCU_MEAS_63] = {
		.name = "oxe_mcu_meas_63"
	},
	[C1_CNTR_OXE_MCU_MEAS_64] = {
		.name = "oxe_mcu_meas_64"
	},
	[C1_CNTR_OXE_MCU_MEAS_65] = {
		.name = "oxe_mcu_meas_65"
	},
	[C1_CNTR_OXE_MCU_MEAS_66] = {
		.name = "oxe_mcu_meas_66"
	},
	[C1_CNTR_OXE_MCU_MEAS_67] = {
		.name = "oxe_mcu_meas_67"
	},
	[C1_CNTR_OXE_MCU_MEAS_68] = {
		.name = "oxe_mcu_meas_68"
	},
	[C1_CNTR_OXE_MCU_MEAS_69] = {
		.name = "oxe_mcu_meas_69"
	},
	[C1_CNTR_OXE_MCU_MEAS_70] = {
		.name = "oxe_mcu_meas_70"
	},
	[C1_CNTR_OXE_MCU_MEAS_71] = {
		.name = "oxe_mcu_meas_71"
	},
	[C1_CNTR_OXE_MCU_MEAS_72] = {
		.name = "oxe_mcu_meas_72"
	},
	[C1_CNTR_OXE_MCU_MEAS_73] = {
		.name = "oxe_mcu_meas_73"
	},
	[C1_CNTR_OXE_MCU_MEAS_74] = {
		.name = "oxe_mcu_meas_74"
	},
	[C1_CNTR_OXE_MCU_MEAS_75] = {
		.name = "oxe_mcu_meas_75"
	},
	[C1_CNTR_OXE_MCU_MEAS_76] = {
		.name = "oxe_mcu_meas_76"
	},
	[C1_CNTR_OXE_MCU_MEAS_77] = {
		.name = "oxe_mcu_meas_77"
	},
	[C1_CNTR_OXE_MCU_MEAS_78] = {
		.name = "oxe_mcu_meas_78"
	},
	[C1_CNTR_OXE_MCU_MEAS_79] = {
		.name = "oxe_mcu_meas_79"
	},
	[C1_CNTR_OXE_MCU_MEAS_80] = {
		.name = "oxe_mcu_meas_80"
	},
	[C1_CNTR_OXE_MCU_MEAS_81] = {
		.name = "oxe_mcu_meas_81"
	},
	[C1_CNTR_OXE_MCU_MEAS_82] = {
		.name = "oxe_mcu_meas_82"
	},
	[C1_CNTR_OXE_MCU_MEAS_83] = {
		.name = "oxe_mcu_meas_83"
	},
	[C1_CNTR_OXE_MCU_MEAS_84] = {
		.name = "oxe_mcu_meas_84"
	},
	[C1_CNTR_OXE_MCU_MEAS_85] = {
		.name = "oxe_mcu_meas_85"
	},
	[C1_CNTR_OXE_MCU_MEAS_86] = {
		.name = "oxe_mcu_meas_86"
	},
	[C1_CNTR_OXE_MCU_MEAS_87] = {
		.name = "oxe_mcu_meas_87"
	},
	[C1_CNTR_OXE_MCU_MEAS_88] = {
		.name = "oxe_mcu_meas_88"
	},
	[C1_CNTR_OXE_MCU_MEAS_89] = {
		.name = "oxe_mcu_meas_89"
	},
	[C1_CNTR_OXE_MCU_MEAS_90] = {
		.name = "oxe_mcu_meas_90"
	},
	[C1_CNTR_OXE_MCU_MEAS_91] = {
		.name = "oxe_mcu_meas_91"
	},
	[C1_CNTR_OXE_MCU_MEAS_92] = {
		.name = "oxe_mcu_meas_92"
	},
	[C1_CNTR_OXE_MCU_MEAS_93] = {
		.name = "oxe_mcu_meas_93"
	},
	[C1_CNTR_OXE_MCU_MEAS_94] = {
		.name = "oxe_mcu_meas_94"
	},
	[C1_CNTR_OXE_MCU_MEAS_95] = {
		.name = "oxe_mcu_meas_95"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_READY] = {
		.name = "oxe_prf_set0_st_ready"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_PCT_IDX_WAIT] = {
		.name = "oxe_prf_set0_st_pct_idx_wait"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_PKTBUF_REQ] = {
		.name = "oxe_prf_set0_st_pktbuf_req"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_PKTBUF_GNT] = {
		.name = "oxe_prf_set0_st_pktbuf_gnt"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_HEADER] = {
		.name = "oxe_prf_set0_st_header"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_DMA_MDR] = {
		.name = "oxe_prf_set0_st_dma_mdr"
	},
	[C1_CNTR_OXE_PRF_SET0_ST_DMA_UPD] = {
		.name = "oxe_prf_set0_st_dma_upd"
	},
	[C1_CNTR_OXE_PRF_SET0_PKTBUF_NA] = {
		.name = "oxe_prf_set0_pktbuf_na"
	},
	[C1_CNTR_OXE_PRF_SET0_SPT_NA] = {
		.name = "oxe_prf_set0_spt_na"
	},
	[C1_CNTR_OXE_PRF_SET0_SMT_NA] = {
		.name = "oxe_prf_set0_smt_na"
	},
	[C1_CNTR_OXE_PRF_SET0_SRB_NA] = {
		.name = "oxe_prf_set0_srb_na"
	},
	[C1_CNTR_OXE_PRF_SET0_TS_SELECT] = {
		.name = "oxe_prf_set0_ts_select"
	},
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN0] = {
		.name = "oxe_prf_set0_occ_hist_bin0"
	},
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN1] = {
		.name = "oxe_prf_set0_occ_hist_bin1"
	},
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN2] = {
		.name = "oxe_prf_set0_occ_hist_bin2"
	},
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN3] = {
		.name = "oxe_prf_set0_occ_hist_bin3"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_READY] = {
		.name = "oxe_prf_set1_st_ready"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_PCT_IDX_WAIT] = {
		.name = "oxe_prf_set1_st_pct_idx_wait"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_PKTBUF_REQ] = {
		.name = "oxe_prf_set1_st_pktbuf_req"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_PKTBUF_GNT] = {
		.name = "oxe_prf_set1_st_pktbuf_gnt"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_HEADER] = {
		.name = "oxe_prf_set1_st_header"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_DMA_MDR] = {
		.name = "oxe_prf_set1_st_dma_mdr"
	},
	[C1_CNTR_OXE_PRF_SET1_ST_DMA_UPD] = {
		.name = "oxe_prf_set1_st_dma_upd"
	},
	[C1_CNTR_OXE_PRF_SET1_PKTBUF_NA] = {
		.name = "oxe_prf_set1_pktbuf_na"
	},
	[C1_CNTR_OXE_PRF_SET1_SPT_NA] = {
		.name = "oxe_prf_set1_spt_na"
	},
	[C1_CNTR_OXE_PRF_SET1_SMT_NA] = {
		.name = "oxe_prf_set1_smt_na"
	},
	[C1_CNTR_OXE_PRF_SET1_SRB_NA] = {
		.name = "oxe_prf_set1_srb_na"
	},
	[C1_CNTR_OXE_PRF_SET1_TS_SELECT] = {
		.name = "oxe_prf_set1_ts_select"
	},
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN0] = {
		.name = "oxe_prf_set1_occ_hist_bin0"
	},
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN1] = {
		.name = "oxe_prf_set1_occ_hist_bin1"
	},
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN2] = {
		.name = "oxe_prf_set1_occ_hist_bin2"
	},
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3] = {
		.name = "oxe_prf_set1_occ_hist_bin3"
	},
};

static const char * const c1_cntr_descs_str[C1_CNTR_SIZE] = {
	[0 ... (C1_CNTR_SIZE - 1)] = NULL,
	[C_CNTR_PI_IPD_PRI_RTRGT_TLPS] = "The number of TLPs the PI_PRI module received from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_MWR_TLPS] = "The number of memory write TLPs the PI_PRI module received from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_MRD_TLPS] = "The number of memory read TLPs the PI_PRI module received from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_MSG_TLPS] = "The number of message TLPs the PI_PRI module received from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_MSGD_TLPS] = "The number of message with data TLPs the PI_PRI module received from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_TLPS_ABORTED] = "The number of TLPs the PI_PRI module received from the RTRGT interface that were aborted.",
	[C_CNTR_PI_IPD_PRI_RTRGT_BLOCKED_ON_MB] = "The number of cycles that the RTRGT interface was blocked due to a full MB FIFO.",
	[C_CNTR_PI_IPD_PRI_RTRGT_BLOCKED_ON_RARB] = "The number of cycles that the RTRGT interface was blocked due to a full RARB FIFO.",
	[C_CNTR_PI_IPD_PRI_RTRGT_HDR_PARITY_ERRORS] = "The number of header parity errors the PI_PRI module detected from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RTRGT_DATA_PARITY_ERRORS] = "The number of data parity errors the PI_PRI module detected from the RTRGT interface.",
	[C_CNTR_PI_IPD_PRI_RBYP_TLPS] = "The number of TLPs the PI_PRI_RARB_XLAT module received from the RBYP interface.",
	[C_CNTR_PI_IPD_PRI_RBYP_TLPS_ABORTED] = "The number of data parity errors the PI_PRI_RARB_XLAT module detected from the RBYP interface.",
	[C_CNTR_PI_IPD_PRI_RBYP_HDR_PARITY_ERRORS] = "The number of header parity errors the PI_PRI_RARB_XLAT module detected from the RBYP interface.",
	[C_CNTR_PI_IPD_PRI_RBYP_DATA_PARITY_ERRORS] = "The number of data parity errors the PI_PRI_RARB_XLAT module detected from the RBYP interface.",
	[C_CNTR_PI_IPD_PTI_TARB_XALI_POSTED_TLPS] = "The number of posted TLPs sourced by the TARB that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_TARB_XALI_NON_POSTED_TLPS] = "The number of non-posted TLPs sourced by the TARB that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_MSIXC_XALI_POSTED_TLPS] = "The number of posted TLPs sourced by the MSIXC that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_DBG_XALI_POSTED_TLPS] = "The number of posted TLPs sourced by the DBG interface that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_DBG_XALI_NON_POSTED_TLPS] = "The number of non-posted TLPs sourced by the PI_PTI_DBG module that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_DMAC_XALI_POSTED_TLPS] = "The number of posted TLPs sourced by the PI_DMAC module that the PI_PTI_TARB_XLAT module sent to the XALI interface.",
	[C_CNTR_PI_IPD_PTI_TARB_P_FIFO_COR_ERRORS] = "The number of posted FIFO correctable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_IPD_PTI_TARB_P_FIFO_UCOR_ERRORS] = "The number of posted FIFO uncorrectable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_IPD_PTI_TARB_NP_FIFO_COR_ERRORS] = "The number of non-posted FIFO correctable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_IPD_PTI_TARB_NP_FIFO_UCOR_ERRORS] = "The number of non-posted FIFO uncorrectable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_PH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted header credits for a TARB posted request.",
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_PD_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted data credits for a TARB posted request.",
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_NPH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient non-posted header credits for a TARB non-posted request.",
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_NPD_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient non-posted data credits for a TARB non-posted request.",
	[C_CNTR_PI_IPD_PTI_TARB_BLOCKED_ON_TAG] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient tags for a TARB non-posted request.",
	[C_CNTR_PI_IPD_PTI_DMAC_BLOCKED_ON_PH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted header credits for a DMAC posted request.",
	[C_CNTR_PI_IPD_PTI_DMAC_BLOCKED_ON_PD_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted data credits for a DMAC posted request.",
	[C_CNTR_PI_IPD_PTI_MSIXC_BLOCKED_ON_PH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted header credits for a MSIXC posted request.",
	[C_CNTR_PI_IPD_PTI_MSIXC_BLOCKED_ON_PD_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted data credits for a MSIXC posted request.",
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_PH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted header credits for a DBG posted request.",
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_PD_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient posted data credits for a DBG posted request.",
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_NPH_CRD] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient non-posted header credits for a DBG non-posted request.",
	[C_CNTR_PI_IPD_PTI_DBG_BLOCKED_ON_TAG] = "The number of cycles that the PI_PTI_TARB_XLAT module is blocked on insufficient tags for a DBG non-posted request.",
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO] = "The number of completion timeouts that the PI_PTI_TARB_XLAT module received from the completion timeout interface.",
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO_FIFO_COR_ERRORS] = "The number of completion timeout FIFO correctable errors detected by the PI_PTI_TARB_XLAT module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_PTI_TARB_CMPL_TO_FIFO_UCOR_ERRORS] = "The number of completion timeout FIFO uncorrectable errors detected by the PI_PTI_TARB_XLAT module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQS_SENT] = "The number of out-of-band IRQs sent to the PTI_TARB_XLAT module to be sent out the XALI interface.",
	[C_CNTR_PI_IPD_MSIXC_OOB_LEGACY_IRQS_SENT] = "The number of out-of-band legacy IRQs sent to the PI_IP core.",
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQS_DISCARDED] = "The number of out-of-band IRQs discrarded due to the function being disabled.",
	[C_CNTR_PI_IPD_MSIXC_OOB_IRQ_PBAS] = "The number of times out-of-band IRQs set the pending bit due to the interrupt being masked.",
	[C_CNTR_PI_IPD_MSIXC_IB_IRQS_SENT] = "The number of in-band IRQs sent to the PTI_TARB_XLAT module to be sent out the XALI interface.",
	[C_CNTR_PI_IPD_MSIXC_IB_LEGACY_IRQS_SENT] = "The number of in-band legacy IRQs sent to the PI_IP core.",
	[C_CNTR_PI_IPD_MSIXC_IB_IRQS_DISCARDED] = "The number of in-band IRQs discrarded due to the function being disabled.",
	[C_CNTR_PI_IPD_MSIXC_IB_IRQ_PBAS] = "The number of times in-band IRQs set the pending bit due to the interrupt being masked.",
	[C_CNTR_PI_IPD_MSIXC_PBA_COR_ERRORS] = "The number of PBA correctable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_MSIXC_PBA_UCOR_ERRORS] = "The number of PBA uncorrectable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_MSIXC_TABLE_COR_ERRORS] = "The number of MSIX table correctable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_MSIXC_TABLE_UCOR_ERRORS] = "The number of MSIX table uncorrectable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_PTI_MSIXC_FIFO_COR_ERRORS] = "The number of pti_msixc_fifo correctable errors detected by the MSIXC module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_PTI_MSIXC_FIFO_UCOR_ERRORS] = "The number of pti_msixc_fifo uncorrectable errors detected by the MSIXC module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_MSIXC_PTI_FIFO_COR_ERRORS] = "The number of msixc_pti_fifo correctable errors detected by the MSIXC module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_MSIXC_PTI_FIFO_UCOR_ERRORS] = "The number of msixc_pti_fifo uncorrectable errors detected by the MSIXC module. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_PI_IPD_DMAC_P_FIFO_COR_ERRORS] = "The number of dmac_p_fifo correctable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_DMAC_P_FIFO_UCOR_ERRORS] = "The number of dmac_p_fifo uncorrectable errors detected by the MSIXC module.",
	[C_CNTR_PI_IPD_DBIC_DBI_REQUESTS] = "The number of DBI requests issued by the DBIC module.",
	[C_CNTR_PI_IPD_DBIC_DBI_ACKS] = "The number of DBI acknowledgements received by the DBIC module.",
	[C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the PI IPD LSA trigger module is asserted.",
	[C_CNTR_PI_IPD_IPD_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the PI IPD LSA trigger module is asserted.",
	[C_CNTR_MB_CRMC_RING_SBE_0] = "The number of CRMC ring in single bit errors",
	[C_CNTR_MB_CRMC_RING_SBE_1] = "The number of CRMC ring in single bit errors",
	[C_CNTR_MB_CRMC_RING_SBE_2] = "The number of CRMC ring in single bit errors",
	[C_CNTR_MB_CRMC_RING_MBE_0] = "The number of CRMC ring in multi bit errors",
	[C_CNTR_MB_CRMC_RING_MBE_1] = "The number of CRMC ring in multi bit errors",
	[C_CNTR_MB_CRMC_RING_MBE_2] = "The number of CRMC ring in multi bit errors",
	[C_CNTR_MB_CRMC_WR_ERROR_0] = "The number of CRMC write errors",
	[C_CNTR_MB_CRMC_WR_ERROR_1] = "The number of CRMC write errors",
	[C_CNTR_MB_CRMC_WR_ERROR_2] = "The number of CRMC write errors",
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_0] = "The number of CRMC AXI write requests",
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_1] = "The number of CRMC AXI write requests",
	[C_CNTR_MB_CRMC_AXI_WR_REQUESTS_2] = "The number of CRMC AXI write requests",
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_0] = "The number of CRMC ring out write requests",
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_1] = "The number of CRMC ring out write requests",
	[C_CNTR_MB_CRMC_RING_WR_REQUESTS_2] = "The number of CRMC ring out write requests",
	[C_CNTR_MB_CRMC_RD_ERROR_0] = "The number of CRMC read errors",
	[C_CNTR_MB_CRMC_RD_ERROR_1] = "The number of CRMC read errors",
	[C_CNTR_MB_CRMC_RD_ERROR_2] = "The number of CRMC read errors",
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_0] = "The number of CRMC AXI read requests",
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_1] = "The number of CRMC AXI read requests",
	[C_CNTR_MB_CRMC_AXI_RD_REQUESTS_2] = "The number of CRMC AXI read requests",
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_0] = "The number of CRMC ring out read requests",
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_1] = "The number of CRMC ring out read requests",
	[C_CNTR_MB_CRMC_RING_RD_REQUESTS_2] = "The number of CRMC ring out read requests",
	[C_CNTR_MB_JAI_AXI_WR_REQUESTS] = "The number of JAI AXI write requests",
	[C_CNTR_MB_JAI_AXI_RD_REQUESTS] = "The number of JAI AXI read requests",
	[C_CNTR_MB_MB_LSA0_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the MB LSA0 trigger module is asserted.",
	[C_CNTR_MB_MB_LSA0_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the MB LSA0 trigger module is asserted.",
	[C_CNTR_MB_IXE_LSA0_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the IXE LSA0 trigger module is asserted.",
	[C_CNTR_MB_IXE_LSA0_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the IXE LSA0 trigger module is asserted.",
	[C_CNTR_MB_CQ_LSA0_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the CQ LSA0 trigger module is asserted.",
	[C_CNTR_MB_CQ_LSA0_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the CQ LSA0 trigger module is asserted.",
	[C_CNTR_MB_EE_LSA0_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the EE LSA0 trigger module is asserted.",
	[C_CNTR_MB_EE_LSA0_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the EE LSA0 trigger module is asserted.",
	[C_CNTR_MB_MB_LSA1_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the MB LSA1 trigger module is asserted.",
	[C_CNTR_MB_MB_LSA1_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the MB LSA1 trigger module is asserted.",
	[C_CNTR_MB_IXE_LSA1_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the IXE LSA1 trigger module is asserted.",
	[C_CNTR_MB_IXE_LSA1_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the IXE LSA1 trigger module is asserted.",
	[C_CNTR_MB_CQ_LSA1_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the CQ LSA1 trigger module is asserted.",
	[C_CNTR_MB_CQ_LSA1_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the CQ LSA1 trigger module is asserted.",
	[C_CNTR_MB_EE_LSA1_TRIGGER_EVENTS_0] = "The number of cycles when the trigger output of the EE LSA1 trigger module is asserted.",
	[C_CNTR_MB_EE_LSA1_TRIGGER_EVENTS_1] = "The number of cycles when the trigger output of the EE LSA1 trigger module is asserted.",
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_0] = "The number of AXI write requests received at the CMC channels.CMC_AXI_WR_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_WR_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_WR_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_WR_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_1] = "The number of AXI write requests received at the CMC channels.CMC_AXI_WR_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_WR_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_WR_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_WR_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_2] = "The number of AXI write requests received at the CMC channels.CMC_AXI_WR_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_WR_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_WR_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_WR_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_WR_REQUESTS_3] = "The number of AXI write requests received at the CMC channels.CMC_AXI_WR_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_WR_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_WR_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_WR_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_0] = "The number of AXI read requests received at the CMC channels.CMC_AXI_RD_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_RD_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_RD_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_RD_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_1] = "The number of AXI read requests received at the CMC channels.CMC_AXI_RD_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_RD_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_RD_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_RD_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_2] = "The number of AXI read requests received at the CMC channels.CMC_AXI_RD_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_RD_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_RD_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_RD_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_MB_CMC_AXI_RD_REQUESTS_3] = "The number of AXI read requests received at the CMC channels.CMC_AXI_RD_REQUESTS[0] are sourced by c_mb_jai.CMC_AXI_RD_REQUESTS[1] are sourced by c_pi_pri_mb_xlat,CMC_AXI_RD_REQUESTS[2] are sourced by c_pi_dmac.CMC_AXI_RD_REQUESTS[3] are sourced by c_mb_flm.",
	[C_CNTR_PI_PRI_MB_RTRGT_TLPS] = "The number of TLPs the PI_PRI_MB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_MB_RTRGT_MWR_TLPS] = "The number of memory write TLPs the PI_PRI_MB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_MB_RTRGT_MRD_TLPS] = "The number of memory read TLPs the PI_PRI_MB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_MB_RTRGT_TLP_DISCARDS] = "The number of TLPs received from the RTRGT interface that the PI_PRI_MB_XLAT module discarded. This count includes PRI_MB_RTRGT_POSTED_TLP_DISCARDS and PRI_MB_RTRGT_NON_POSTED_TLP_DISCARDS but also includes TLPs that are aborted prior to decoding and/or unknown TLPs.",
	[C_CNTR_PI_PRI_MB_RTRGT_POSTED_TLP_DISCARDS] = "The number of posted TLPs received from the RTRGT interface that the PI_PRI_MB_XLAT module discarded.",
	[C_CNTR_PI_PRI_MB_RTRGT_NON_POSTED_TLP_DISCARDS] = "The number of non posted TLPs received from the RTRGT interface that the PI_PRI_MB_XLAT module discarded.",
	[C_CNTR_PI_PRI_MB_RTRGT_POSTED_TLP_PARTIAL_DISCARDS] = "The number of TLPs received from the RTRGT interface that the PI_PRI_MB_XLAT module discarded after writing some portion of the data to the MB AXI interface and recieved a write response error. This count is only incremented if there was more data in the TLP to be written when the write response error was detected.",
	[C_CNTR_PI_PRI_MB_RTRGT_FIFO_COR_ERRORS] = "The number of RTRGT FIFO correctable errors detected by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_RTRGT_FIFO_UCOR_ERRORS] = "The number of RTRGT FIFO uncorrectable errors detected by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_XALI_CMPL_TLPS] = "The number of completion TLPs sent to the XALI interface by the PI_PRI_MB_XLAT module. This count includes PRI_MB_XALI_CMPL_CA_TLPS and PRI_MB_XALI_CMPL_UR_TLPS .",
	[C_CNTR_PI_PRI_MB_XALI_CMPL_CA_TLPS] = "The number of completion TLPs sent to the XALI interface by the PI_PRI_MB_XLAT module with completer abort status to the XALI interface.",
	[C_CNTR_PI_PRI_MB_XALI_CMPL_UR_TLPS] = "The number of completion TLPs sent to the XALI interface by the PI_PRI_MB_XLAT module with unsupported request status to the XALI interface.",
	[C_CNTR_PI_PRI_MB_AXI_WR_REQUESTS] = "The number of write requests sent to the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_RD_REQUESTS] = "The number of read requests sent to the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_WR_RSP_DECERRS] = "The number of write response decode errors received from the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_WR_RSP_SLVERRS] = "The number of write response slave errors received from the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_DECERRS] = "The number of read response decode errors received from the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_SLVERRS] = "The number of read response slave errors received from the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_MB_AXI_RD_RSP_PARERRS] = "The number of read response parity errors received from the AXI interface by the PI_PRI_MB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_RTRGT_TLPS] = "The number of TLPs the PI_PRI_RARB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_RARB_RTRGT_MWR_TLPS] = "The number of memory write TLPs the PI_PRI_RARB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_RARB_RTRGT_MSG_TLPS] = "The number of message TLPs the PI_PRI_RARB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_RARB_RTRGT_MSGD_TLPS] = "The number of message with data TLPs the PI_PRI_RARB_XLAT module received from the RTRGT interface.",
	[C_CNTR_PI_PRI_RARB_RTRGT_TLP_DISCARDS] = "The number of TLPs received from the RTRGT interface that the PI_PRI_RARB_XLAT module discarded.",
	[C_CNTR_PI_PRI_RARB_RTRGT_FIFO_COR_ERRORS] = "The number of RTRGT FIFO correctable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_RTRGT_FIFO_UCOR_ERRORS] = "The number of RTRGT FIFO uncorrectable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_RBYP_TLPS] = "The number of TLPs the PI_PRI_RARB_XLAT module received from the RBYP interface. The RBYP interface is used for completions.",
	[C_CNTR_PI_PRI_RARB_RBYP_TLP_DISCARDS] = "The number of TLPs received from the RBYP interface that the PI_PRI_RARB_XLAT module discarded.",
	[C_CNTR_PI_PRI_RARB_RBYP_FIFO_COR_ERRORS] = "The number of RBYP FIFO correctable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_RBYP_FIFO_UCOR_ERRORS] = "The number of RBYP FIFO uncorrectable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_SRIOVT_COR_ERRORS] = "The number of SRIOVT correctable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PRI_RARB_SRIOVT_UCOR_ERRORS] = "The number of SRIOVT uncorrectable errors detected by the PI_PRI_RARB_XLAT module.",
	[C_CNTR_PI_PTI_TARB_PKTS] = "The number of packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_PKT_DISCARDS] = "The number of packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface that were discarded.",
	[C_CNTR_PI_PTI_TARB_RARB_CA_PKTS] = "The number of completion abort packets the PI_PTI_TARB_XLAT module sent back to the TARB client due to a discard. The completion aborts are sent over the PI_RARB interface.",
	[C_CNTR_PI_PTI_TARB_MWR_PKTS] = "The number of memory write packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_MRD_PKTS] = "The number of memory read packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_FLUSH_PKTS] = "The number of flush request packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_IRQ_PKTS] = "The number of interrupt request packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_TRANSLATION_PKTS] = "The number of translation request packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_INV_RSP_PKTS] = "The number of invalidate response packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_PG_REQ_PKTS] = "The number of page request packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_AMO_PKTS] = "The number of atomic operation packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_FETCHING_AMO_PKTS] = "The number of fetching atomic operation packets received by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_HDR_COR_ERRORS] = "The number of header correctable errors detected by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_HDR_UCOR_ERRORS] = "The number of header uncorrectable errors detected by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_DATA_COR_ERRORS] = "The number of data correctable errors detected by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_DATA_UCOR_ERROR] = "The number of data uncorrectable errors detected by the PI_PTI_TARB_XLAT module from the PI_TARB interface.",
	[C_CNTR_PI_PTI_TARB_ACXT_COR_ERRORS] = "The number of ACXT correctable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_PTI_TARB_ACXT_UCOR_ERRORS] = "The number of ACXT uncorrectable errors detected by the PI_PTI_TARB_XLAT module.",
	[C_CNTR_PI_DMAC_AXI_RD_REQUESTS] = "The number of read requests sent to the AXI interface by the PI_DMAC module.",
	[C_CNTR_PI_DMAC_AXI_RD_RSP_DECERRS] = "The number of read response decode errors received from the AXI interface by the PI_DMAC module.",
	[C_CNTR_PI_DMAC_AXI_RD_RSP_SLVERRS] = "The number of read response slave errors received from the AXI interface by the PI_DMAC module.",
	[C_CNTR_PI_DMAC_AXI_RD_RSP_PARERRS] = "The number of read response parity errors received from the AXI interface by the PI_DMAC module.",
	[C_CNTR_PI_DMAC_PAYLOAD_MWR_TLPS] = "The number of payload memory write TLPS sent to the XALI interface by the PI_DMAC module. Note that the TLPs must pass through the CDC FIFO before being sent to the XALI interface.",
	[C_CNTR_PI_DMAC_CE_MWR_TLPS] = "The number of completion event memory write TLPS sent to the XALI interface by the PI_DMAC module. Note that the TLPs must pass through the CDC FIFO before being sent to the XALI interface.",
	[C_CNTR_PI_DMAC_IRQS] = "The number of interrupt requests sent to the MSIXC module by the PI_DMAC module. Note that the TLPs must pass through the CDC FIFO before being sent to the MSIXC module.",
	[C_CNTR_PI_DMAC_DESC_COR_ERRORS] = "The number of DMAC DESC correctable errors detected by the PI_DMAC module.",
	[C_CNTR_PI_DMAC_DESC_UCOR_ERRORS] = "The number of DMAC DESC uncorrectable errors detected by the PI_DMAC module.",
	[C_CNTR_ATU_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_ATU_EXT_ERR_INFO_MSK ( Section 13.16.15 on page 1147 ). The errors which can contribute to this counter are: AC_MEM_COR, OTB_MEM_COR, TAG_MEM_COR, INGRESS_BUS_COR, DATA_MEM_COR, CHAIN_TAIL_MEM_COR, CHAIN_NEXT_MEM_COR, MISSQ_MEM_COR, UNIQQ_MEM_COR, CHAIN_MEM_COR, POINTER_MEM_COR, REPLAYQ_MEM_COR, RETRYQ_MEM_COR, PRB_MEM_COR, INVALQ_MEM_COR, RSPINFOQ_MEM_COR, CMPLDATA_MEM_COR, ATUCQ_MEM_COR, ATS_PRQ_MEM_COR, RARB_HDR_BUS_COR, RARB_DATA_BUS_COR, ODPQ_MEM_COR;",
	[C_CNTR_ATU_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_ATU_EXT_ERR_INFO_MSK ( Section 13.16.15 on page 1147 ). The errors which can contribute to this counter are: AC_MEM_UCOR, OTB_MEM_UCOR, TAG_MEM_UCOR, INGRESS_BUS_UCOR, DATA_MEM_UCOR, CHAIN_TAIL_MEM_UCOR, CHAIN_NEXT_MEM_UCOR, MISSQ_MEM_UCOR, UNIQQ_MEM_UCOR, CHAIN_MEM_UCOR, POINTER_MEM_UCOR, REPLAYQ_MEM_UCOR, RETRYQ_MEM_UCOR, PRB_MEM_UCOR, INVALQ_MEM_UCOR, RSPINFOQ_MEM_UCOR, CMPLDATA_MEM_UCOR, ATUCQ_MEM_UCOR, ATS_PRQ_MEM_UCOR, RARB_HDR_BUS_UCOR, RARB_DATA_BUS_UCOR, ODPQ_MEM_UCOR, ANCHOR_VLD_MEM_UCOR, ANCHOR_MEM_UCOR;",
	[C_CNTR_ATU_CACHE_MISS_0] = "The number of cache misses that have been observed. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_MISS_1] = "The number of cache misses that have been observed. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_MISS_2] = "The number of cache misses that have been observed. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_MISS_3] = "The number of cache misses that have been observed. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_0] = "The number of cache hits that have been observed on the Base Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_1] = "The number of cache hits that have been observed on the Base Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_2] = "The number of cache hits that have been observed on the Base Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_BASE_PAGE_SIZE_3] = "The number of cache hits that have been observed on the Base Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_0] = "The number of cache hits that have been observed on the Derivative 1 Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_1] = "The number of cache hits that have been observed on the Derivative 1 Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_2] = "The number of cache hits that have been observed on the Derivative 1 Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE1_PAGE_SIZE_3] = "The number of cache hits that have been observed on the Derivative 1 Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_0] = "The number of cache hits that have been observed on the Derivative 2Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_1] = "The number of cache hits that have been observed on the Derivative 2Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_2] = "The number of cache hits that have been observed on the Derivative 2Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_HIT_DERIVATIVE2_PAGE_SIZE_3] = "The number of cache hits that have been observed on the Derivative 2Page Size. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_MISS_OXE] = "The number of cache misses that have been observed by the OXE.",
	[C_CNTR_ATU_CACHE_MISS_IXE] = "The number of cache misses that have been observed by the IXE.",
	[C_CNTR_ATU_CACHE_MISS_EE] = "The number of cache misses that have been observed by the EE.",
	[C_CNTR_ATU_ATS_TRANS_LATENCY_0] = "ATS Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_ATS_TRANS_LATENCY_1] = "ATS Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_ATS_TRANS_LATENCY_2] = "ATS Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_ATS_TRANS_LATENCY_3] = "ATS Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_NTA_TRANS_LATENCY_0] = "NTA Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_NTA_TRANS_LATENCY_1] = "NTA Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_NTA_TRANS_LATENCY_2] = "NTA Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_NTA_TRANS_LATENCY_3] = "NTA Translation latency histogram. Four bins defined in C_ATU_CFG_XLATION_HIST .",
	[C_CNTR_ATU_CLIENT_REQ_OXE] = "The number of address translation requests from the OXE.",
	[C_CNTR_ATU_CLIENT_REQ_IXE] = "The number of address translation requests from the IXE.",
	[C_CNTR_ATU_CLIENT_REQ_EE] = "The number of address translation requests from the EE.",
	[C_CNTR_ATU_FILTERED_REQUESTS] = "The number of address translation requests, from all Clients, that have been filtered by the Filter block.",
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_0] = "ATS Page Request Services On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_1] = "ATS Page Request Services On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_2] = "ATS Page Request Services On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_ATS_PRS_ODP_LATENCY_3] = "ATS Page Request Services On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_0] = "NIC Page Request Interface On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_1] = "NIC Page Request Interface On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_2] = "NIC Page Request Interface On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_NIC_PRI_ODP_LATENCY_3] = "NIC Page Request Interface On-Demand Paging latency histogram. Four bins defined in C_ATU_CFG_ODP_HIST .",
	[C_CNTR_ATU_ODP_REQUESTS_0] = "Number of On-Demand Paging requests. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_ODP_REQUESTS_1] = "Number of On-Demand Paging requests. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_ODP_REQUESTS_2] = "Number of On-Demand Paging requests. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_ODP_REQUESTS_3] = "Number of On-Demand Paging requests. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_0] = "The number of client responses that were not RC_OK. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_1] = "The number of client responses that were not RC_OK. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_2] = "The number of client responses that were not RC_OK. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CLIENT_RSP_NOT_OK_3] = "The number of client responses that were not RC_OK. Pooled counter, pool determined by the CNTR_POOL_ID in the AC.",
	[C_CNTR_ATU_CACHE_EVICTIONS] = "The number of times a Tag was evicted from the cache to make way for a new one.",
	[C_CNTR_ATU_ATS_INVAL_CNTR] = "The number of ATS Invalidations performed, triggered by ATS Invalidation requests received from the TA",
	[C_CNTR_ATU_IMPLICIT_FLR_INVAL_CNTR] = "The number of Implicit Invalidations performed, triggered by Functional Level Resets",
	[C_CNTR_ATU_IMPLICIT_ATS_EN_INVAL_CNTR] = "The number of Implicit Invalidations performed, triggered by ATS Enable being set",
	[C_CNTR_ATU_ATUCQ_INVAL_CNTR] = "The number of Invalidations performed, triggered by the INVALIDATE_PAGES command",
	[C_CNTR_ATU_PCIE_UNSUCCESS_CMPL] = "The number of PCIe Reads that resulted in a Completion Status other than Successful Completion",
	[C_CNTR_ATU_ATS_TRANS_ERR] = "The number of ATS Translation Requests that resulted in a Completion Status other than Successful Completion",
	[C_CNTR_ATU_PCIE_ERR_POISONED] = "Number of PCIe Reads that returned with Error Poisoned",
	[C_CNTR_ATU_ATS_DYNAMIC_STATE_CHANGE_CNTR] = "Number of times the ATS Dynamic Flag state changed",
	[C_CNTR_ATU_AT_STALL_CCP] = "The number of clocks an Address Translation Request was stalled from entering the Cache Check Pipeline. An Address Translation Request can be stalled in this way due to an in-progress update to the Cache, an Cache invalidation, or loss of arbitration due to Replayed requests coming from the Filter block.",
	[C_CNTR_ATU_AT_STALL_NO_PTID] = "The number of clocks an Address Translation Request was stalled due to no PTID being available.",
	[C_CNTR_ATU_AT_STALL_NP_CDTS] = "The number of clocks an Address Translation Request was stalled due to no Non-Posted credits being available to a message to the TARB.",
	[C_CNTR_ATU_AT_STALL_TARB_ARB] = "The number of clocks an Address Translation Request was stalled due to losing arbitration to access the interface to TARB. Other type of requests with equal access to the TARB are Writebacks, In-Band Interrupts, Invalidation Completions, and On-Demand Paging Requests.",
	[C_CNTR_ATU_INVAL_STALL_ARB] = "The number of clocks an Invalidation, either initiated from the ATU Command Queue or from ATS, is stalled in waiting for an invalidation from the opposite source to complete.",
	[C_CNTR_ATU_INVAL_STALL_CMPL_WAIT] = "The number of clocks an invalidation, either initiated from the ATU Command Queue or from ATS, is stalled in waiting for the associated Completion Wait to finish.",
	[C_CNTR_CQ_SUCCESS_TX_CNTR] = "The number of valid Transmit commands received by CQ. Note that this counter is incremented for all valid commands (including Fence and NoOp that are not passed to OXE). Note that this counter is not incremented for commands sent to TOU as they are counted in C_TOU_CNTRS_T",
	[C_CNTR_CQ_SUCCESS_TGT_CNTR] = "The number of valid Target commands received by CQ. Note that this counter is not incremented for commands sent to TOU as they are counted in C_TOU_CNTRS_T Note that this counter is not incremented for commands sent to TOU as they are counted in C_TOU_CNTRS_T",
	[C_CNTR_CQ_FAIL_TX_CNTR] = "The number of invalid Transmit commands received by CQ. Incremented each time the CQ command parser generates a full event (COMMAND_FAILURE) with one of the following return codes: RC_CMD_ALIGN_ERROR, RC_PERM_VIOLATION, RC_CMD_INVALID_ARG, RC_OP_VIOLATION, RC_UNCOR, RC_UNCOR_TRNSNT, RC_PCIE_UNSUCCESS_CMPL, RC_PCIE_ERROR_POISONED; Note that this counter is not incremented for errors detected by TOU as they are counted in C_TOU_CNTRS_T",
	[C_CNTR_CQ_FAIL_TGT_CNTR] = "The number of invalid Target commands received by CQ. Incremented each time the CQ command parser generates a full event (COMMAND_FAILURE) with one of the following return codes: RC_PERM_VIOLATION, RC_CMD_INVALID_ARG, RC_UNCOR, RC_UNCOR_TRNSNT, RC_PCIE_UNSUCCESS_CMPL, RC_PCIE_ERROR_POISONED; Note that this counter is not incremented for errors detected by TOU as they are counted in C_TOU_CNTRS_T",
	[C_CNTR_CQ_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_CQ_ERR_INFO_MSK ( Section 13.8.14 on page 595 ). Includes memory errors detected by TOU. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_CQ_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_CQ_ERR_INFO_MSK ( Section 13.8.14 on page 595 ). Includes memory errors detected by TOU. See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_CQ_RARB_LL_ERR_CNTR] = "This is a count of the number of RARB low-latency transaction errors that have been detected. This counter is only incremented for RARB low-latency related errors that are not masked in CSR C_CQ_ERR_INFO_MSK ( Section 13.8.14 on page 595 ). The errors which can contribute to this counter are: RARB_LL_WR_HW_ERR, RARB_LL_WR_SW_ERR;",
	[C_CNTR_CQ_RARB_ERR_CNTR] = "This is a count of the number of RARB transaction errors that have been detected. This count excludes errors associated with RARB low-latency transactions. This counter is only incremented for RARB related errors that are not masked in CSR C_CQ_ERR_INFO_MSK ( Section 13.8.14 on page 595 ). The errors which can contribute to this counter are: RARB_BAD_XACTN_ERR, RARB_CMPLTN_ERR, RARB_TOU_WR_HW_ERR, RARB_INVLD_WR_SW_ERR, RARB_TOU_WR_SW_ERR;",
	[C_CNTR_CQ_WR_PTR_UPDT_ERR_CNTR] = "This is a count of attempts to update a transmit queue or a target queue write pointer with an invalid value. The error which contributes to this counter is WR_PTR_UPDT_ERR .",
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_0] = "Number of PCIe command reads issued for transmit prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_1] = "Number of PCIe command reads issued for transmit prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_2] = "Number of PCIe command reads issued for transmit prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TXQ_CMD_READS_3] = "Number of PCIe command reads issued for transmit prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_0] = "Number of PCIe command reads issued for target prefetch queue. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_1] = "Number of PCIe command reads issued for target prefetch queue. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_2] = "Number of PCIe command reads issued for target prefetch queue. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TGQ_CMD_READS_3] = "Number of PCIe command reads issued for target prefetch queue. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TOU_CMD_READS_0] = "Number of PCIe command reads issued for TOU prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TOU_CMD_READS_1] = "Number of PCIe command reads issued for TOU prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TOU_CMD_READS_2] = "Number of PCIe command reads issued for TOU prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_NUM_TOU_CMD_READS_3] = "Number of PCIe command reads issued for TOU prefetch queues. Four counters, one each for reads of 64, 128, 192, or 256 bytes.",
	[C_CNTR_CQ_TX_WAITING_ON_READ_0] = "Cycles on which transmit prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. CQ would maintain a count of the number of command read requests pending for each of the four counter pools The prefetch unit would increment these counts as a PCIe read is issued and decrement them as they complete. On cycles for which there is no command to process, the head of the CQ pipeline would increment the TX_WAITING_ON_READ counter for each pool that has read requests pending.",
	[C_CNTR_CQ_TX_WAITING_ON_READ_1] = "Cycles on which transmit prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. CQ would maintain a count of the number of command read requests pending for each of the four counter pools The prefetch unit would increment these counts as a PCIe read is issued and decrement them as they complete. On cycles for which there is no command to process, the head of the CQ pipeline would increment the TX_WAITING_ON_READ counter for each pool that has read requests pending.",
	[C_CNTR_CQ_TX_WAITING_ON_READ_2] = "Cycles on which transmit prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. CQ would maintain a count of the number of command read requests pending for each of the four counter pools The prefetch unit would increment these counts as a PCIe read is issued and decrement them as they complete. On cycles for which there is no command to process, the head of the CQ pipeline would increment the TX_WAITING_ON_READ counter for each pool that has read requests pending.",
	[C_CNTR_CQ_TX_WAITING_ON_READ_3] = "Cycles on which transmit prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. CQ would maintain a count of the number of command read requests pending for each of the four counter pools The prefetch unit would increment these counts as a PCIe read is issued and decrement them as they complete. On cycles for which there is no command to process, the head of the CQ pipeline would increment the TX_WAITING_ON_READ counter for each pool that has read requests pending.",
	[C_CNTR_CQ_TGT_WAITING_ON_READ_0] = "Cycles on which target prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.45 on page 641 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. Implementation as described for TX_WAITING_ON_READ",
	[C_CNTR_CQ_TGT_WAITING_ON_READ_1] = "Cycles on which target prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.45 on page 641 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. Implementation as described for TX_WAITING_ON_READ",
	[C_CNTR_CQ_TGT_WAITING_ON_READ_2] = "Cycles on which target prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.45 on page 641 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. Implementation as described for TX_WAITING_ON_READ",
	[C_CNTR_CQ_TGT_WAITING_ON_READ_3] = "Cycles on which target prefetch buffers are empty and pool has read requests pending. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.45 on page 641 ). Note that this counter does not increment on cycles for which commands in another pool are being processed. Implementation as described for TX_WAITING_ON_READ",
	[C_CNTR_CQ_NUM_TX_CMD_ALIGN_ERRORS] = "Number of Transmit commands with incorrect alignment, incremented when RC_CMD_ALIGN_ERROR is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_NUM_TX_CMD_OP_ERRORS] = "Number of Transmit commands containing invalid operations, incremented when RC_OP_VIOLATION is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_NUM_TX_CMD_ARG_ERRORS] = "Number of Transmit commands containing invalid arguments, incremented when RC_CMD_INVALID_ARG is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_NUM_TX_CMD_PERM_ERRORS] = "Number of Transmit commands containing invalid resource identifiers, incremented when RC_PERM_VIOLATION is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_NUM_TGT_CMD_ARG_ERRORS] = "Number of Target commands containing invalid arguments, incremented when RC_CMD_INVALID_ARG is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_NUM_TGT_CMD_PERM_ERRORS] = "Number of Target commands containing invalid resource identifiers, incremented when RC_PERM_VIOLATION is returned to the user in a COMMAND_FAILURE event. Errors in TOU commands are counted in C_TOU_CNTRS_T .",
	[C_CNTR_CQ_CQ_OXE_NUM_FLITS] = "Number of command flits sent from CQ to OXE. Each valid command will result in between one and four flits. Incremented as command flits are sent to OXE.",
	[C_CNTR_CQ_CQ_OXE_NUM_STALLS] = "Number of cycles on which a command flit is available for OXE but can not be sent because the flow queues are full.",
	[C_CNTR_CQ_CQ_OXE_NUM_IDLES] = "Number of cycles on which interface to OXE is idle.",
	[C_CNTR_CQ_CQ_TOU_NUM_FLITS] = "Number of command flits sent from CQ to TOU. Incremented as command flits are set to TOU",
	[C_CNTR_CQ_CQ_TOU_NUM_STALLS] = "Number of cycles on which a command flit is available for TOU but can not be sent because the Prefetch unit does not have credits to send commands to the TOU command queue (in_ct_fifo_cq).",
	[C_CNTR_CQ_CQ_TOU_NUM_IDLES] = "Number of cycles on which interface to TOU is idle",
	[C_CNTR_CQ_NUM_IDC_CMDS_0] = "Number of successfully parsed immediate data commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of an IDC command is sent to OXE",
	[C_CNTR_CQ_NUM_IDC_CMDS_1] = "Number of successfully parsed immediate data commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of an IDC command is sent to OXE",
	[C_CNTR_CQ_NUM_IDC_CMDS_2] = "Number of successfully parsed immediate data commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of an IDC command is sent to OXE",
	[C_CNTR_CQ_NUM_IDC_CMDS_3] = "Number of successfully parsed immediate data commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of an IDC command is sent to OXE",
	[C_CNTR_CQ_NUM_DMA_CMDS_0] = "Number of successfully parsed DMA commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of a DMA command is sent to OXE",
	[C_CNTR_CQ_NUM_DMA_CMDS_1] = "Number of successfully parsed DMA commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of a DMA command is sent to OXE",
	[C_CNTR_CQ_NUM_DMA_CMDS_2] = "Number of successfully parsed DMA commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of a DMA command is sent to OXE",
	[C_CNTR_CQ_NUM_DMA_CMDS_3] = "Number of successfully parsed DMA commands. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ). Incremented as the first flit of a DMA command is sent to OXE",
	[C_CNTR_CQ_NUM_CQ_CMDS_0] = "Number of successfully parsed CQ commands processed by the CQ block. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as Fence and CQ commands are processed.",
	[C_CNTR_CQ_NUM_CQ_CMDS_1] = "Number of successfully parsed CQ commands processed by the CQ block. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as Fence and CQ commands are processed.",
	[C_CNTR_CQ_NUM_CQ_CMDS_2] = "Number of successfully parsed CQ commands processed by the CQ block. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as Fence and CQ commands are processed.",
	[C_CNTR_CQ_NUM_CQ_CMDS_3] = "Number of successfully parsed CQ commands processed by the CQ block. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as Fence and CQ commands are processed.",
	[C_CNTR_CQ_NUM_LL_CMDS_0] = "Number of successfully parsed commands delivered using the low latency command issue mechanism. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as the first flit of a command is sent to OXE when pref_q_out_is_low_latency is set.",
	[C_CNTR_CQ_NUM_LL_CMDS_1] = "Number of successfully parsed commands delivered using the low latency command issue mechanism. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as the first flit of a command is sent to OXE when pref_q_out_is_low_latency is set.",
	[C_CNTR_CQ_NUM_LL_CMDS_2] = "Number of successfully parsed commands delivered using the low latency command issue mechanism. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as the first flit of a command is sent to OXE when pref_q_out_is_low_latency is set.",
	[C_CNTR_CQ_NUM_LL_CMDS_3] = "Number of successfully parsed commands delivered using the low latency command issue mechanism. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ) Incremented as the first flit of a command is sent to OXE when pref_q_out_is_low_latency is set.",
	[C_CNTR_CQ_NUM_TGT_CMDS_0] = "Number of successfully parsed CQ commands processed by target command queues. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the target queue descriptor base table ( Section 13.8.45 on page 641 . All target commands are single flit. Incremented as target commands are sent to LPE",
	[C_CNTR_CQ_NUM_TGT_CMDS_1] = "Number of successfully parsed CQ commands processed by target command queues. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the target queue descriptor base table ( Section 13.8.45 on page 641 . All target commands are single flit. Incremented as target commands are sent to LPE",
	[C_CNTR_CQ_NUM_TGT_CMDS_2] = "Number of successfully parsed CQ commands processed by target command queues. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the target queue descriptor base table ( Section 13.8.45 on page 641 . All target commands are single flit. Incremented as target commands are sent to LPE",
	[C_CNTR_CQ_NUM_TGT_CMDS_3] = "Number of successfully parsed CQ commands processed by target command queues. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the target queue descriptor base table ( Section 13.8.45 on page 641 . All target commands are single flit. Incremented as target commands are sent to LPE",
	[C_CNTR_CQ_DMA_CMD_COUNTS_0] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_1] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_2] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_3] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_4] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_5] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_6] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_7] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_8] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_9] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_10] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_11] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_12] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_13] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_14] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_DMA_CMD_COUNTS_15] = "Counts for each of the successfully parsed commands in C_DMA_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is sent to OXE. These counters include commands issued using the DMA path, but not those issuing using the IDC path (see Errata 3194 ).",
	[C_CNTR_CQ_CQ_CMD_COUNTS_0] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_1] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_2] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_3] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_4] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_5] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_6] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_7] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_8] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_9] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_10] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_11] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_12] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_13] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_14] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CQ_CMD_COUNTS_15] = "Counts for each of the successfully parsed commands in C_CQ_OP_T and C_TGT_OP_T . Offset into the counter array is equal to the OpCode. Incremented as the first flit of command is processed by CQ (for CQ commands) or sent to LPE (target commands). Note that the OpCodes used by CQ commands and target commands do not overlap. CQ uses 0-2 and target uses 8-12.",
	[C_CNTR_CQ_CYCLES_BLOCKED_0] = "Number of cycles on which pool had a command ready to send to OXE that could not make progress because another OCUSET won arbitration. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_CYCLES_BLOCKED_1] = "Number of cycles on which pool had a command ready to send to OXE that could not make progress because another OCUSET won arbitration. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_CYCLES_BLOCKED_2] = "Number of cycles on which pool had a command ready to send to OXE that could not make progress because another OCUSET won arbitration. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_CYCLES_BLOCKED_3] = "Number of cycles on which pool had a command ready to send to OXE that could not make progress because another OCUSET won arbitration. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_0] = "The number of low-latency operations for which the data of the operation was accepted. Barring an uncorrectable error being subsequently detected in the low latency data buffer (error flag LL_BUF_UCOR ), low-latency data that is accepted is forwarded to the CQ's transmit command parser. Each 64-byte ( LL_WR64A0123 ) or 128-byte ( LL_WR128A02 , LL_WR128A13 ) block of data written to the command issue image ( Table 17 on page 125 ) is counted as one low-latency operation. As each such block of data may contain more than one CQ command, there is not a one-to-one correspondence between this NUM_LL_OPS_SUCCESSFUL counter and the NUM_LL_CMDS counter. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_1] = "The number of low-latency operations for which the data of the operation was accepted. Barring an uncorrectable error being subsequently detected in the low latency data buffer (error flag LL_BUF_UCOR ), low-latency data that is accepted is forwarded to the CQ's transmit command parser. Each 64-byte ( LL_WR64A0123 ) or 128-byte ( LL_WR128A02 , LL_WR128A13 ) block of data written to the command issue image ( Table 17 on page 125 ) is counted as one low-latency operation. As each such block of data may contain more than one CQ command, there is not a one-to-one correspondence between this NUM_LL_OPS_SUCCESSFUL counter and the NUM_LL_CMDS counter. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_2] = "The number of low-latency operations for which the data of the operation was accepted. Barring an uncorrectable error being subsequently detected in the low latency data buffer (error flag LL_BUF_UCOR ), low-latency data that is accepted is forwarded to the CQ's transmit command parser. Each 64-byte ( LL_WR64A0123 ) or 128-byte ( LL_WR128A02 , LL_WR128A13 ) block of data written to the command issue image ( Table 17 on page 125 ) is counted as one low-latency operation. As each such block of data may contain more than one CQ command, there is not a one-to-one correspondence between this NUM_LL_OPS_SUCCESSFUL counter and the NUM_LL_CMDS counter. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SUCCESSFUL_3] = "The number of low-latency operations for which the data of the operation was accepted. Barring an uncorrectable error being subsequently detected in the low latency data buffer (error flag LL_BUF_UCOR ), low-latency data that is accepted is forwarded to the CQ's transmit command parser. Each 64-byte ( LL_WR64A0123 ) or 128-byte ( LL_WR128A02 , LL_WR128A13 ) block of data written to the command issue image ( Table 17 on page 125 ) is counted as one low-latency operation. As each such block of data may contain more than one CQ command, there is not a one-to-one correspondence between this NUM_LL_OPS_SUCCESSFUL counter and the NUM_LL_CMDS counter. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_0] = "The number of low-latency operations received for which the data of the operation was not accepted because the corresponding transmit queue was not empty, or the transmit queue's low-latency data buffer was not empty when the low-latency operation was received. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_1] = "The number of low-latency operations received for which the data of the operation was not accepted because the corresponding transmit queue was not empty, or the transmit queue's low-latency data buffer was not empty when the low-latency operation was received. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_2] = "The number of low-latency operations received for which the data of the operation was not accepted because the corresponding transmit queue was not empty, or the transmit queue's low-latency data buffer was not empty when the low-latency operation was received. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_REJECTED_3] = "The number of low-latency operations received for which the data of the operation was not accepted because the corresponding transmit queue was not empty, or the transmit queue's low-latency data buffer was not empty when the low-latency operation was received. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_0] = "The number of low-latency operations received for which the data of the operation was not accepted because delivery of the operation's data was split into multiple writes to the command issue image, with some or all of those writes containing less than 64 bytes of data. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_1] = "The number of low-latency operations received for which the data of the operation was not accepted because delivery of the operation's data was split into multiple writes to the command issue image, with some or all of those writes containing less than 64 bytes of data. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_2] = "The number of low-latency operations received for which the data of the operation was not accepted because delivery of the operation's data was split into multiple writes to the command issue image, with some or all of those writes containing less than 64 bytes of data. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_SPLIT_3] = "The number of low-latency operations received for which the data of the operation was not accepted because delivery of the operation's data was split into multiple writes to the command issue image, with some or all of those writes containing less than 64 bytes of data. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_0] = "The number of error-free low-latency operations received. This counter increments whenever any of counters NUM_LL_OPS_SUCCESSFUL , NUM_LL_OPS_REJECTED , or NUM_LL_OPS_SPLIT increments. However, this counter is not necessarily equal to the sum of these other three counters because counters NUM_LL_OPS_REJECTED and NUM_LL_OPS_SPLIT can both increment for the same low-latency operation. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_1] = "The number of error-free low-latency operations received. This counter increments whenever any of counters NUM_LL_OPS_SUCCESSFUL , NUM_LL_OPS_REJECTED , or NUM_LL_OPS_SPLIT increments. However, this counter is not necessarily equal to the sum of these other three counters because counters NUM_LL_OPS_REJECTED and NUM_LL_OPS_SPLIT can both increment for the same low-latency operation. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_2] = "The number of error-free low-latency operations received. This counter increments whenever any of counters NUM_LL_OPS_SUCCESSFUL , NUM_LL_OPS_REJECTED , or NUM_LL_OPS_SPLIT increments. However, this counter is not necessarily equal to the sum of these other three counters because counters NUM_LL_OPS_REJECTED and NUM_LL_OPS_SPLIT can both increment for the same low-latency operation. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_CQ_NUM_LL_OPS_RECEIVED_3] = "The number of error-free low-latency operations received. This counter increments whenever any of counters NUM_LL_OPS_SUCCESSFUL , NUM_LL_OPS_REJECTED , or NUM_LL_OPS_SPLIT increments. However, this counter is not necessarily equal to the sum of these other three counters because counters NUM_LL_OPS_REJECTED and NUM_LL_OPS_SPLIT can both increment for the same low-latency operation. Pooled counter, pool determined by MEM_Q_STAT_CNT_POOL in the transmit queue descriptor base table ( Section 13.8.38 on page 636 ).",
	[C_CNTR_TOU_SUCCESS_CNTR] = "The number of valid commands processed by TOU",
	[C_CNTR_TOU_FAIL_CNTR] = "The number of commands on which the TOU command parser detects an error. Incremented each time the TOU generates a full event (COMMAND_FAILURE) with one of the following return codes: RC_CMD_ALIGN_ERROR, RC_PERM_VIOLATION, RC_CMD_INVALID_ARG, RC_OP_VIOLATION, RC_UNCOR_TRNSNT, RC_UNCOR;",
	[C_CNTR_TOU_CQ_TOU_NUM_CMDS] = "Number of commands received from CQ",
	[C_CNTR_TOU_CQ_TOU_NUM_STALLS] = "Number of cycles on which a command is available from CQ but can not be taken because the TOU pipeline is stalled.",
	[C_CNTR_TOU_TOU_OXE_NUM_FLITS] = "Number of command flits sent to OXE. DMA commands are single flit. DMA AMO commands are two flits",
	[C_CNTR_TOU_TOU_OXE_NUM_STALLS] = "Number of cycles stalled when sending command to OXE",
	[C_CNTR_TOU_TOU_OXE_NUM_IDLES] = "Number of cycles on which interface to OXE is idle",
	[C_CNTR_TOU_LIST_REBUILD_CYCLES] = "Number of cycles that TOU spends on list rebuilds",
	[C_CNTR_TOU_CMD_FIFO_FULL_CYCLES] = "Number of cycles in which the CT Command FIFO is full",
	[C_CNTR_TOU_NUM_DOORBELL_WRITES] = "Number of commands issued as doorbell writes.",
	[C_CNTR_TOU_NUM_CT_UPDATES] = "Number of counting event updates received from EE. .",
	[C_CNTR_TOU_NUM_TRIG_CMDS_0] = "Number of triggered commands. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_TRIG_CMDS_1] = "Number of triggered commands. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_TRIG_CMDS_2] = "Number of triggered commands. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_TRIG_CMDS_3] = "Number of triggered commands. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_LIST_REBUILDS_0] = "Number of list CT list rebuilds. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_LIST_REBUILDS_1] = "Number of list CT list rebuilds. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_LIST_REBUILDS_2] = "Number of list CT list rebuilds. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_NUM_LIST_REBUILDS_3] = "Number of list CT list rebuilds. Pooled counter, pool is equal to the PFQ number.",
	[C_CNTR_TOU_CT_CMD_COUNTS_0] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_1] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_2] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_3] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_4] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_5] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_6] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_7] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_8] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_9] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_10] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_11] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_12] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_13] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_14] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_TOU_CT_CMD_COUNTS_15] = "Counts for each of the successfully validated commands in C_CT_OP_T . Offset into the counter array is equal to the OpCode.",
	[C_CNTR_PCT_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_PCT_ERR_INFO_MSK ( Section 13.11.23 on page 833 ). Due to Errata 2932 , the value of this counter might be inaccurate. Also see erratum 3325 for cases where errors may be over-counted. The errors which can contribute to this counter are: FLOP_FIFO_COR, SCT_CAM_FREE_COR, TCT_CAM_FREE_COR, MST_CAM_FREE_COR, TRS_CAM_FREE_COR, SCT_TBL_COR, TCT_TBL_COR, SMT_TBL_COR, SPT_TBL_COR, TRS_TBL_COR, REQ_REDO_COR, SRB_DATA_RAM_COR, SRB_NLIST_RAM_COR, CLS_CLR_REQ_GEN_COR;",
	[C_CNTR_PCT_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_PCT_ERR_INFO_MSK ( Section 13.11.23 on page 833 ). Due to Errata 2932 , the value of this counter might be inaccurate. Also see erratum 3325 for cases where errors may be over-counted. The errors which can contribute to this counter are: FLOP_FIFO_UCOR, SCT_CAM_UCOR, TCT_CAM_UCOR, MST_CAM_UCOR, TRS_CAM_UCOR, SCT_CAM_FREE_UCOR, TCT_CAM_FREE_UCOR, MST_CAM_FREE_UCOR, TRS_CAM_FREE_UCOR, SCT_TBL_UCOR, TCT_TBL_UCOR, SMT_TBL_UCOR, SPT_TBL_UCOR, TRS_TBL_UCOR, REQ_REDO_UCOR, SRB_DATA_RAM_UCOR, SRB_NLIST_RAM_UCOR, CLS_CLR_REQ_GEN_UCOR;",
	[C_CNTR_PCT_REQ_ORDERED] = "Count of the number of ordered Portals request packets sent to HNI from OXE. This only includes Portals request packets that have responses. This does not include retry packets that are resent out of the SRB. This also does not include management packets generated by PCT.",
	[C_CNTR_PCT_REQ_UNORDERED] = "Count of the number of unordered Portals request packets sent to HNI from OXE. This only includes Portals request packets that have responses.",
	[C_CNTR_PCT_REQ_NO_RESPONSE] = "Number of Portals request packets sent to HNI from OXE for which no response was expected.",
	[C_CNTR_PCT_ETH_PACKETS] = "Number of Ethernet packets sent to HNI from OXE.",
	[C_CNTR_PCT_OPTIP_PACKETS] = "Number of Optimized non-Portals IP packets sent to HNI from OXE.",
	[C_CNTR_PCT_PTLS_RESPONSE] = "Number of Portals response packets sent to HNI from OXE.",
	[C_CNTR_PCT_RESPONSES_RECEIVED] = "Number of response packets received",
	[C_CNTR_PCT_HRP_RESPONSES_RECEIVED] = "Number of high rate Put responses received",
	[C_CNTR_PCT_HRP_RSP_DISCARD_RECEIVED] = "Number of HRP responses received with the discard flag set",
	[C_CNTR_PCT_HRP_RSP_ERR_RECEIVED] = "Number of HRP responses received with the error flag set",
	[C_CNTR_PCT_CONN_SCT_OPEN] = "Count of the number of times a connection is opened at the source",
	[C_CNTR_PCT_CONN_TCT_OPEN] = "Count of the number of times a connection is opened at the target",
	[C_CNTR_PCT_MST_HIT_ON_SOM] = "Occurrences of an MST already existed for an SOM request. The reasons of these occurrences can be found in Section 7.5.23.1 Due to CAS 3001 , this counter also incorrectly increments when a new request that needs a new MST entry fails to obtain one because of lack of credits.",
	[C_CNTR_PCT_TRS_HIT_ON_REQ] = "Occurrences of a TRS already existed for a new request. The reasons of these occurrences can be found in Section 7.5.23.2 Due to CAS 3001 , this counter also incorrectly increments when a new request that needs a new TRS entry fails to obtain one because of lack of credits.",
	[C_CNTR_PCT_CLS_REQ_MISS_TCT] = "Occurrences where a Close Request misses on TCT",
	[C_CNTR_PCT_CLEAR_SENT] = "Count of the number of times clear is sent",
	[C_CNTR_PCT_CLOSE_SENT] = "Count of the number of times close is sent. This includes SW generated close requests.",
	[C_CNTR_PCT_ACCEL_CLOSE] = "Count of the number of times the accelerated close process is used",
	[C_CNTR_PCT_CLEAR_CLOSE_DROP] = "Count of number of times a clear or close is dropped due to internal congestion to the output FIFO.",
	[C_CNTR_PCT_REQ_SRC_ERROR] = "Number of requests sent that have the SRC_ERROR flag set",
	[C_CNTR_PCT_BAD_SEQ_NACKS] = "Count of the number of NACKS generated where there was a sequence number mismatch at the target",
	[C_CNTR_PCT_NO_TCT_NACKS] = "Count of the number of NACKS received because no TCT entry was available at the target",
	[C_CNTR_PCT_NO_MST_NACKS] = "Count of the number of NACKS received because no MST entry was available at the target",
	[C_CNTR_PCT_NO_TRS_NACKS] = "Count of the number of NACKS received because no TRS entry was available at the target",
	[C_CNTR_PCT_NO_MATCHING_TCT] = "Count of the number of NACKS received because no matching TCT connection entry was found at the target",
	[C_CNTR_PCT_RESOURCE_BUSY] = "Count of the number of NACKS received because either the MST or TRS resource was busy at the target, usually meant the entry was in the process of being deallocated",
	[C_CNTR_PCT_ERR_NO_MATCHING_TRS] = "Count of the number of errors received because no matching TRS connection entry was found for a retried operation at the target",
	[C_CNTR_PCT_ERR_NO_MATCHING_MST] = "Count of the number of errors received because no matching MST connection entry was found for a retried operation at the target",
	[C_CNTR_PCT_SPT_TIMEOUTS] = "Count of the number of request timeouts generated",
	[C_CNTR_PCT_SCT_TIMEOUTS] = "Count of the number of SCT timeouts generated",
	[C_CNTR_PCT_TCT_TIMEOUTS] = "Count of the number of TCT timeouts generated",
	[C_CNTR_PCT_RETRY_SRB_REQUESTS] = "Number of retry requests generated from the SRB",
	[C_CNTR_PCT_RETRY_TRS_PUT] = "Number of retry Put/AMO responses generated from the TRS",
	[C_CNTR_PCT_RETRY_MST_GET] = "Number of retry Get requests generated from the MST",
	[C_CNTR_PCT_CLR_CLS_STALLS] = "Count of the number of IXE requests were stalled due to internal Clear/Close request outbound FIFO congestion.",
	[C_CNTR_PCT_CLOSE_RSP_DROPS] = "Count of the number of close responses were dropped due to internal Close response outbound FIFO congestion. Due to errata 3217 , the value of this counter can be unreliable.",
	[C_CNTR_PCT_TRS_RSP_NACK_DROPS] = "Count of the number of NACKs and TRS responses that were dropped due to internal TRS retry and NACK response outbound FIFO congestion.",
	[C_CNTR_PCT_REQ_BLOCKED_CLOSING] = "Number of requests denied (blocked temporarily) because connection was closing",
	[C_CNTR_PCT_REQ_BLOCKED_CLEARING] = "Number of requests denied (blocked temporarily) because connection was clearing",
	[C_CNTR_PCT_REQ_BLOCKED_RETRY] = "Number of requests denied (blocked temporarily) because earlier requests are being retried",
	[C_CNTR_PCT_RSP_ERR_RCVD] = "Number of responses received at the source that contains a non-OK and non-NACK return_code.",
	[C_CNTR_PCT_RSP_DROPPED_TIMEOUT] = "Number of responses dropped because the SPT entry had timed out when the response arrived",
	[C_CNTR_PCT_RSP_DROPPED_TRY] = "Number of responses dropped because the Try value in the response did not match that in the SPT entry.",
	[C_CNTR_PCT_RSP_DROPPED_CLS_TRY] = "Number of Close responses dropped because the Try value in the response did not match that in the SCT entry",
	[C_CNTR_PCT_RSP_DROPPED_INACTIVE] = "Number of responses dropped because the SPT entry was inactive",
	[C_CNTR_PCT_PCT_HNI_FLITS] = "Cycles on which flits were sent from PCT to HNI",
	[C_CNTR_PCT_PCT_HNI_STALLS] = "Cycles on which interface from PCT to HNI is stalled",
	[C_CNTR_PCT_PCT_EE_EVENTS] = "Cycles on which events were sent from PCT to EE",
	[C_CNTR_PCT_PCT_EE_STALLS] = "Cycles on which interface from PCT to EE is stalled",
	[C_CNTR_PCT_PCT_CQ_NOTIFICATIONS] = "Cycles on which notifications were sent from PCT to CQ",
	[C_CNTR_PCT_PCT_CQ_STALLS] = "Cycles on which interface from PCT to CQ is stalled",
	[C_CNTR_PCT_PCT_MST_CMD0] = "Cycles on which commands were sent from PCT to MST on interface bus 0",
	[C_CNTR_PCT_PCT_MST_STALLS0] = "Cycles on which interface 0 from PCT to MST is stalled",
	[C_CNTR_PCT_PCT_MST_CMD1] = "Cycles on which commands were sent from PCT to MST on interface bus 1",
	[C_CNTR_PCT_PCT_MST_STALLS1] = "Cycles on which interface 1 from PCT to MST is stalled",
	[C_CNTR_PCT_SCT_STALL_STATE] = "Number of times an ordered OXE request encounters an SCT connection that is in the SCT_STALL state and needs to be stalled.",
	[C_CNTR_PCT_TGT_CLS_ABORT] = "Number of times the TCT closing engine has to abort due to a TRS entry still being in the PEND_RSP state.",
	[C_CNTR_PCT_CLS_REQ_BAD_SEQNO_DROPS] = "Number of close/clear requests dropped due to bad seqno at the target.",
	[C_CNTR_PCT_REQ_TCT_TMOUT_DROPS] = "Number of requests dropped due to TCT timeout at the target. This includes both normal requests and close/clear requests",
	[C_CNTR_PCT_TRS_REPLAY_PEND_DROPS] = "Number of replay requests dropped due to TRS pend_rsp bit is set",
	[C_CNTR_PCT_REQ_RSP_LATENCY_0] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_1] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_2] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_3] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_4] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_5] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_6] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_7] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_8] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_9] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_10] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_11] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_12] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_13] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_14] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_15] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_16] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_17] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_18] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_19] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_20] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_21] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_22] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_23] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_24] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_25] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_26] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_27] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_28] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_29] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_30] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_REQ_RSP_LATENCY_31] = "Request response latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_0] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_1] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_2] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_3] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_4] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_5] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_6] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_7] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_8] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_9] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_10] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_11] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_12] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_13] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_14] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_PCT_HOST_ACCESS_LATENCY_15] = "Host access latency histogram. Due to errata 2930 , the configuration of the bins affects the accuracy of the histogram.",
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_0] = "The number of full events enqueued to an event queue. This count does not include null events ( EVENT_NULL_EVENT ) unless an error is reported in the event's return code. The EE inserts error-free null events into event queues as required to maintain alignment requirements for real (non-null) events. ( Section 5.2.2 on page 168 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_1] = "The number of full events enqueued to an event queue. This count does not include null events ( EVENT_NULL_EVENT ) unless an error is reported in the event's return code. The EE inserts error-free null events into event queues as required to maintain alignment requirements for real (non-null) events. ( Section 5.2.2 on page 168 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_2] = "The number of full events enqueued to an event queue. This count does not include null events ( EVENT_NULL_EVENT ) unless an error is reported in the event's return code. The EE inserts error-free null events into event queues as required to maintain alignment requirements for real (non-null) events. ( Section 5.2.2 on page 168 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_ENQUEUED_CNTR_3] = "The number of full events enqueued to an event queue. This count does not include null events ( EVENT_NULL_EVENT ) unless an error is reported in the event's return code. The EE inserts error-free null events into event queues as required to maintain alignment requirements for real (non-null) events. ( Section 5.2.2 on page 168 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_0] = "The number of full events, subject to an event queue space reservation, that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_RSRVN provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_1] = "The number of full events, subject to an event queue space reservation, that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_RSRVN provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_2] = "The number of full events, subject to an event queue space reservation, that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_RSRVN provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_RSRVN_CNTR_3] = "The number of full events, subject to an event queue space reservation, that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_RSRVN provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_0] = "The number of flow-control state-change full events that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_FC_SC provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_1] = "The number of flow-control state-change full events that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_FC_SC provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_2] = "The number of flow-control state-change full events that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_FC_SC provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_FC_SC_CNTR_3] = "The number of flow-control state-change full events that should have been enqueued, but were not because the event queue was full. The description of EE error flag EQ_OVFLW_FC_SC provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_0] = "The number of full events that should have been enqueued, but were not because the event queue was full. This count includes all dropped full events that are not included in either EVENTS_DROPPED_RSRVN_CNTR or EVENTS_DROPPED_FC_SC_CNTR . The sum of those two counters and this one is the total number of dropped full events. The description of EE error flag EQ_OVFLW_ORDINARY provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_1] = "The number of full events that should have been enqueued, but were not because the event queue was full. This count includes all dropped full events that are not included in either EVENTS_DROPPED_RSRVN_CNTR or EVENTS_DROPPED_FC_SC_CNTR . The sum of those two counters and this one is the total number of dropped full events. The description of EE error flag EQ_OVFLW_ORDINARY provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_2] = "The number of full events that should have been enqueued, but were not because the event queue was full. This count includes all dropped full events that are not included in either EVENTS_DROPPED_RSRVN_CNTR or EVENTS_DROPPED_FC_SC_CNTR . The sum of those two counters and this one is the total number of dropped full events. The description of EE error flag EQ_OVFLW_ORDINARY provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EVENTS_DROPPED_ORDINARY_CNTR_3] = "The number of full events that should have been enqueued, but were not because the event queue was full. This count includes all dropped full events that are not included in either EVENTS_DROPPED_RSRVN_CNTR or EVENTS_DROPPED_FC_SC_CNTR . The sum of those two counters and this one is the total number of dropped full events. The description of EE error flag EQ_OVFLW_ORDINARY provides additional information. ( Section 13.15.8 on page 1035 ) This count also includes such events dropped because the event queue buffer, to which the events should have been written, was disabled. This situation should not occur unless software has incorrectly configured the event queue descriptor. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_0] = "The number of status updates written to an event queue. Status updates are written to report the fill-level of an event queue crossing a configured threshold and to report dropped events. ( Section 5.2.5 on page 171 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_1] = "The number of status updates written to an event queue. Status updates are written to report the fill-level of an event queue crossing a configured threshold and to report dropped events. ( Section 5.2.5 on page 171 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_2] = "The number of status updates written to an event queue. Status updates are written to report the fill-level of an event queue crossing a configured threshold and to report dropped events. ( Section 5.2.5 on page 171 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_STATUS_UPDATE_CNTR_3] = "The number of status updates written to an event queue. Status updates are written to report the fill-level of an event queue crossing a configured threshold and to report dropped events. ( Section 5.2.5 on page 171 ) Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_CBS_WRITTEN_CNTR_0] = "The number of combining buffers written to an event queue. Full events are first written into a 64-byte combining buffer, and then, possibly after some delay, the combining buffer is written to the event queue. If the full events already in the combining buffer do not fill the buffer, additional events may be added to it before the combining buffer is written to the event queue. The number of combining buffers written will be less than or equal to the number of events enqueued ( EVENTS_ENQUEUED_CNTR ). CBS_WRITTEN_CNTR does not include status updates written ( EQ_STATUS_UPDATE_CNTR ). Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_CBS_WRITTEN_CNTR_1] = "The number of combining buffers written to an event queue. Full events are first written into a 64-byte combining buffer, and then, possibly after some delay, the combining buffer is written to the event queue. If the full events already in the combining buffer do not fill the buffer, additional events may be added to it before the combining buffer is written to the event queue. The number of combining buffers written will be less than or equal to the number of events enqueued ( EVENTS_ENQUEUED_CNTR ). CBS_WRITTEN_CNTR does not include status updates written ( EQ_STATUS_UPDATE_CNTR ). Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_CBS_WRITTEN_CNTR_2] = "The number of combining buffers written to an event queue. Full events are first written into a 64-byte combining buffer, and then, possibly after some delay, the combining buffer is written to the event queue. If the full events already in the combining buffer do not fill the buffer, additional events may be added to it before the combining buffer is written to the event queue. The number of combining buffers written will be less than or equal to the number of events enqueued ( EVENTS_ENQUEUED_CNTR ). CBS_WRITTEN_CNTR does not include status updates written ( EQ_STATUS_UPDATE_CNTR ). Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_CBS_WRITTEN_CNTR_3] = "The number of combining buffers written to an event queue. Full events are first written into a 64-byte combining buffer, and then, possibly after some delay, the combining buffer is written to the event queue. If the full events already in the combining buffer do not fill the buffer, additional events may be added to it before the combining buffer is written to the event queue. The number of combining buffers written will be less than or equal to the number of events enqueued ( EVENTS_ENQUEUED_CNTR ). CBS_WRITTEN_CNTR does not include status updates written ( EQ_STATUS_UPDATE_CNTR ). Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_0] = "The number of partially full combining buffers written to an event queue. A partially full combining buffer is one that still had room for one or more additional event to be added to it at the time it was released to be written to the event queue. Therefore, the end of the buffer was padded with one or more null events. A partially full combining can be written to the event queue because either the size of the next event to be enqueued is such that, to maintain alignment requirements, the next event will need to start a new combining buffer, or because too much time has elapsed without additional events arriving to fill the partially full combining buffer. Note, combining buffers containing null events between real events at the start and end of the buffer are not counted as being partially full buffers by this counter. For example, if the first event to arrive into a buffer is a 16-byte event, and this is followed by a 32-byte event, the buffer will contain the 16-byte real event, followed by a 16-byte null event, followed by the 32-byte real event. The buffer does not have room for any additional events, and so is not counted as a partially full buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_1] = "The number of partially full combining buffers written to an event queue. A partially full combining buffer is one that still had room for one or more additional event to be added to it at the time it was released to be written to the event queue. Therefore, the end of the buffer was padded with one or more null events. A partially full combining can be written to the event queue because either the size of the next event to be enqueued is such that, to maintain alignment requirements, the next event will need to start a new combining buffer, or because too much time has elapsed without additional events arriving to fill the partially full combining buffer. Note, combining buffers containing null events between real events at the start and end of the buffer are not counted as being partially full buffers by this counter. For example, if the first event to arrive into a buffer is a 16-byte event, and this is followed by a 32-byte event, the buffer will contain the 16-byte real event, followed by a 16-byte null event, followed by the 32-byte real event. The buffer does not have room for any additional events, and so is not counted as a partially full buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_2] = "The number of partially full combining buffers written to an event queue. A partially full combining buffer is one that still had room for one or more additional event to be added to it at the time it was released to be written to the event queue. Therefore, the end of the buffer was padded with one or more null events. A partially full combining can be written to the event queue because either the size of the next event to be enqueued is such that, to maintain alignment requirements, the next event will need to start a new combining buffer, or because too much time has elapsed without additional events arriving to fill the partially full combining buffer. Note, combining buffers containing null events between real events at the start and end of the buffer are not counted as being partially full buffers by this counter. For example, if the first event to arrive into a buffer is a 16-byte event, and this is followed by a 32-byte event, the buffer will contain the 16-byte real event, followed by a 16-byte null event, followed by the 32-byte real event. The buffer does not have room for any additional events, and so is not counted as a partially full buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_PARTIAL_CBS_WRITTEN_CNTR_3] = "The number of partially full combining buffers written to an event queue. A partially full combining buffer is one that still had room for one or more additional event to be added to it at the time it was released to be written to the event queue. Therefore, the end of the buffer was padded with one or more null events. A partially full combining can be written to the event queue because either the size of the next event to be enqueued is such that, to maintain alignment requirements, the next event will need to start a new combining buffer, or because too much time has elapsed without additional events arriving to fill the partially full combining buffer. Note, combining buffers containing null events between real events at the start and end of the buffer are not counted as being partially full buffers by this counter. For example, if the first event to arrive into a buffer is a 16-byte event, and this is followed by a 32-byte event, the buffer will contain the 16-byte real event, followed by a 16-byte null event, followed by the 32-byte real event. The buffer does not have room for any additional events, and so is not counted as a partially full buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_0] = "The number of partially full combining buffers that were written to their event queue because too much time elapsed without additional events arriving to fill the buffer. Field LATENCY_TOLERANCE in CSR C_EE_CFG_EQ_DESCRIPTOR[2048] ( Section 13.15.37 on page 1090 ) controls how long the EE waits for more events to arrive to fill a partially full combining buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_1] = "The number of partially full combining buffers that were written to their event queue because too much time elapsed without additional events arriving to fill the buffer. Field LATENCY_TOLERANCE in CSR C_EE_CFG_EQ_DESCRIPTOR[2048] ( Section 13.15.37 on page 1090 ) controls how long the EE waits for more events to arrive to fill a partially full combining buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_2] = "The number of partially full combining buffers that were written to their event queue because too much time elapsed without additional events arriving to fill the buffer. Field LATENCY_TOLERANCE in CSR C_EE_CFG_EQ_DESCRIPTOR[2048] ( Section 13.15.37 on page 1090 ) controls how long the EE waits for more events to arrive to fill a partially full combining buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EXPIRED_CBS_WRITTEN_CNTR_3] = "The number of partially full combining buffers that were written to their event queue because too much time elapsed without additional events arriving to fill the buffer. Field LATENCY_TOLERANCE in CSR C_EE_CFG_EQ_DESCRIPTOR[2048] ( Section 13.15.37 on page 1090 ) controls how long the EE waits for more events to arrive to fill a partially full combining buffer. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_0] = "The number of event queue buffer switches that have been performed. Buffer switching is described in Section 5.2.1.1 on page 167 . Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_1] = "The number of event queue buffer switches that have been performed. Buffer switching is described in Section 5.2.1.1 on page 167 . Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_2] = "The number of event queue buffer switches that have been performed. Buffer switching is described in Section 5.2.1.1 on page 167 . Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_BUFFER_SWITCH_CNTR_3] = "The number of event queue buffer switches that have been performed. Buffer switching is described in Section 5.2.1.1 on page 167 . Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_0] = "The number of event queue buffer switches that could not be performed as soon as requested because there was not sufficient free space immediately available in the old event queue buffer to enqueue the buffer switch event. The buffer switch is eventually performed when space becomes available in the old event queue buffer. Deferred event queue buffer switches may be an indication that software is waiting too long before scheduling buffer switch requests. One or more other events are likely to be dropped when an event queue buffer switch must be deferred. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_1] = "The number of event queue buffer switches that could not be performed as soon as requested because there was not sufficient free space immediately available in the old event queue buffer to enqueue the buffer switch event. The buffer switch is eventually performed when space becomes available in the old event queue buffer. Deferred event queue buffer switches may be an indication that software is waiting too long before scheduling buffer switch requests. One or more other events are likely to be dropped when an event queue buffer switch must be deferred. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_2] = "The number of event queue buffer switches that could not be performed as soon as requested because there was not sufficient free space immediately available in the old event queue buffer to enqueue the buffer switch event. The buffer switch is eventually performed when space becomes available in the old event queue buffer. Deferred event queue buffer switches may be an indication that software is waiting too long before scheduling buffer switch requests. One or more other events are likely to be dropped when an event queue buffer switch must be deferred. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_DEFERRED_EQ_SWITCH_CNTR_3] = "The number of event queue buffer switches that could not be performed as soon as requested because there was not sufficient free space immediately available in the old event queue buffer to enqueue the buffer switch event. The buffer switch is eventually performed when space becomes available in the old event queue buffer. Deferred event queue buffer switches may be an indication that software is waiting too long before scheduling buffer switch requests. One or more other events are likely to be dropped when an event queue buffer switch must be deferred. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_0] = "The number of address translations that have been prefetched. When address translation prefetching is enabled for an event queue, the EE requests translations for upcoming memory pages of the event queue before events are written to those pages. This is done to load translations into the ATU's address translation cache in advance of when the translation will be required. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_1] = "The number of address translations that have been prefetched. When address translation prefetching is enabled for an event queue, the EE requests translations for upcoming memory pages of the event queue before events are written to those pages. This is done to load translations into the ATU's address translation cache in advance of when the translation will be required. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_2] = "The number of address translations that have been prefetched. When address translation prefetching is enabled for an event queue, the EE requests translations for upcoming memory pages of the event queue before events are written to those pages. This is done to load translations into the ATU's address translation cache in advance of when the translation will be required. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_ADDR_TRANS_PREFETCH_CNTR_3] = "The number of address translations that have been prefetched. When address translation prefetching is enabled for an event queue, the EE requests translations for upcoming memory pages of the event queue before events are written to those pages. This is done to load translations into the ATU's address translation cache in advance of when the translation will be required. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ).",
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_0] = "The number of times the event queue software state ( Section 13.15.38 on page 1098 ) has been updated using a fast-path write. The rate of increase of this counter relative to CBS_WRITTEN_CNTR may provide an indication of how frequently software is servicing event queues. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ). Note, as with counter EQS_SWS_WR_REQ_CNTR , this counter will not count every event queue software state fast-path write that occurs if the processing of writes received in quick succession, and targeting the same event queue, becomes coalesced in the EE's event queue state pipeline. Unlike counter EQS_SWS_WR_REQ_CNTR , this counter does not increment for fast path writes that target a disabled event queue.",
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_1] = "The number of times the event queue software state ( Section 13.15.38 on page 1098 ) has been updated using a fast-path write. The rate of increase of this counter relative to CBS_WRITTEN_CNTR may provide an indication of how frequently software is servicing event queues. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ). Note, as with counter EQS_SWS_WR_REQ_CNTR , this counter will not count every event queue software state fast-path write that occurs if the processing of writes received in quick succession, and targeting the same event queue, becomes coalesced in the EE's event queue state pipeline. Unlike counter EQS_SWS_WR_REQ_CNTR , this counter does not increment for fast path writes that target a disabled event queue.",
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_2] = "The number of times the event queue software state ( Section 13.15.38 on page 1098 ) has been updated using a fast-path write. The rate of increase of this counter relative to CBS_WRITTEN_CNTR may provide an indication of how frequently software is servicing event queues. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ). Note, as with counter EQS_SWS_WR_REQ_CNTR , this counter will not count every event queue software state fast-path write that occurs if the processing of writes received in quick succession, and targeting the same event queue, becomes coalesced in the EE's event queue state pipeline. Unlike counter EQS_SWS_WR_REQ_CNTR , this counter does not increment for fast path writes that target a disabled event queue.",
	[C_CNTR_EE_EQ_SW_STATE_WR_CNTR_3] = "The number of times the event queue software state ( Section 13.15.38 on page 1098 ) has been updated using a fast-path write. The rate of increase of this counter relative to CBS_WRITTEN_CNTR may provide an indication of how frequently software is servicing event queues. Pooled counter, pool determined by CNTR_POOL_ID in the event queue descriptor ( Section 13.15.37 on page 1090 ). Note, as with counter EQS_SWS_WR_REQ_CNTR , this counter will not count every event queue software state fast-path write that occurs if the processing of writes received in quick succession, and targeting the same event queue, becomes coalesced in the EE's event queue state pipeline. Unlike counter EQS_SWS_WR_REQ_CNTR , this counter does not increment for fast path writes that target a disabled event queue.",
	[C_CNTR_EE_LPE_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the LPE. This count includes all event requests, including those that do not result in any full or counting events being generated. Depending on event option, eq_handle and ct_handle configuration, an event request received by the EE may result in a full event being generated, and/or a counting event being generated, or no events being generated.",
	[C_CNTR_EE_LPE_FE_CNTR] = "The number of requests to generate full events that the EE has received from the LPE. This count includes deferred-drop events ( LPE_DD_CNTR ).",
	[C_CNTR_EE_LPE_DD_CNTR] = "The number of deferred-drop event requests the EE has received from the LPE. These are the number of event requests the EE has received from the LPE that, based on event option configuration, will not result in a full event being enqueued, but still require the event request to arbitrate for access to, and be processed by the EE's event queue state pipeline. The event request consumes bandwidth in the EE's event queue state pipeline, where the request is eventually dropped.",
	[C_CNTR_EE_LPE_CE_CNTR] = "The number of requests to generate counting events that the EE has received from the LPE",
	[C_CNTR_EE_LPE_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in LPE_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_LPE_CE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in LPE_CE_CNTR is available and is stalled from advancing in the EE's counting event processing pipeline. Stalls result from arbitration with counting event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's counting event processing pipeline.",
	[C_CNTR_EE_IXE_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the IXE. This count includes all event requests, including those that do not result in any full or counting events being generated. Depending on event option, eq_handle and ct_handle configuration, an event request received by the EE may result in a full event being generated, and/or a counting event being generated, or no events being generated.",
	[C_CNTR_EE_IXE_FE_CNTR] = "The number of requests to generate full events that the EE has received from the IXE. This count includes deferred-drop events ( IXE_DD_CNTR ).",
	[C_CNTR_EE_IXE_DD_CNTR] = "The number of deferred-drop event requests the EE has received from the IXE. These are the number of event requests the EE has received from the IXE that, based on event option configuration, will not result in a full event being enqueued, but still require the event request to arbitrate for access to, and be processed by the EE's event queue state pipeline. The event request consumes bandwidth in the EE's event queue state pipeline, where the request is eventually dropped.",
	[C_CNTR_EE_IXE_CE_CNTR] = "The number of requests to generate counting events that the EE has received from the IXE",
	[C_CNTR_EE_IXE_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in IXE_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_IXE_CE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in IXE_CE_CNTR is available and is stalled from advancing in the EE's counting event processing pipeline. Stalls result from arbitration with counting event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's counting event processing pipeline.",
	[C_CNTR_EE_MST_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the MST. This count includes all event requests, including those that do not result in any full or counting events being generated. Depending on event option, eq_handle and ct_handle configuration, an event request received by the EE may result in a full event being generated, and/or a counting event being generated, or no events being generated.",
	[C_CNTR_EE_MST_FE_CNTR] = "The number of requests to generate full events that the EE has received from the MST. This count includes deferred-drop events ( MST_DD_CNTR ).",
	[C_CNTR_EE_MST_DD_CNTR] = "The number of deferred-drop event requests the EE has received from the MST. These are the number of event requests the EE has received from the MST that, based on event option configuration, will not result in a full event being enqueued, but still require the event request to arbitrate for access to, and be processed by the EE's event queue state pipeline. The event request consumes bandwidth in the EE's event queue state pipeline, where the request is eventually dropped.",
	[C_CNTR_EE_MST_CE_CNTR] = "The number of requests to generate counting events that the EE has received from the MST",
	[C_CNTR_EE_MST_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in MST_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_MST_CE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in MST_CE_CNTR is available and is stalled from advancing in the EE's counting event processing pipeline. Stalls result from arbitration with counting event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's counting event processing pipeline.",
	[C_CNTR_EE_PCT_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the PCT. This count includes all event requests, including those that do not result in any full or counting events being generated. Depending on event option, eq_handle and ct_handle configuration, an event request received by the EE may result in a full event being generated, and/or a counting event being generated, or no events being generated.",
	[C_CNTR_EE_PCT_FE_CNTR] = "The number of requests to generate full events that the EE has received from the PCT. This count includes deferred-drop events ( PCT_DD_CNTR ).",
	[C_CNTR_EE_PCT_DD_CNTR] = "The number of deferred-drop event requests the EE has received from the PCT. These are the number of event requests the EE has received from the PCT that, based on event option configuration, will not result in a full event being enqueued, but still require the event request to arbitrate for access to, and be processed by the EE's event queue state pipeline. The event request consumes bandwidth in the EE's event queue state pipeline, where the request is eventually dropped.",
	[C_CNTR_EE_PCT_CE_CNTR] = "The number of requests to generate counting events that the EE has received from the PCT",
	[C_CNTR_EE_PCT_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in PCT_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_PCT_CE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in PCT_CE_CNTR is available and is stalled from advancing in the EE's counting event processing pipeline. Stalls result from arbitration with counting event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's counting event processing pipeline.",
	[C_CNTR_EE_HNI_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the HNI. This count includes all event requests, including those that do not result in any full events being generated. Depending on eq_handle configuration, an event request received by the EE may result in a full event being generated, or no event being generated. The HNI does not request counting events.",
	[C_CNTR_EE_HNI_FE_CNTR] = "The number of requests to generate full events that the EE has received from the HNI. Each request included in this count is expected to result in a full event being enqueued. Unlike requests generated by some other blocks, such as the LPE ( LPE_DD_CNTR ), HNI-generated event requests are not subject to deferred dropping.",
	[C_CNTR_EE_HNI_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in HNI_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_CQ_EVENT_REQ_CNTR] = "The number of event requests the EE has received from the CQ. This count includes all event requests, including those that do not result in any full events being generated. Depending on eq_handle configuration, an event request received by the EE may result in a full event being generated, or no event being generated. The CQ does not request counting events.",
	[C_CNTR_EE_CQ_FE_CNTR] = "The number of requests to generate full events that the EE has received from the CQ. Each request included in this count is expected to result in a full event being enqueued. Unlike requests generated by some other blocks, such as the LPE ( LPE_DD_CNTR ), CQ-generated event requests are not subject to deferred dropping.",
	[C_CNTR_EE_CQ_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in CQ_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_TS_FE_CNTR] = "The number of timestamp full events the EE has generated across all event queues that are enabled to received timestamp events ( Section 13.15.41 on page 1107 ).",
	[C_CNTR_EE_TS_FE_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in TS_FE_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from arbitration with full event requests the EE is receiving from other blocks and from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_FE_ARB_OUT_EVENT_CNTR] = "The aggregate number of full event requests received by the EE, from all of its client blocks and including timestamp events. This value is measured at the output of the EE's full event arbiter. Under quiescent conditions, and barring differences arising from differing initialization, this counter should be equal to the sum of: LPE_FE_CNTR, IXE_FE_CNTR, MST_FE_CNTR, PCT_FE_CNTR, HNI_FE_CNTR, CQ_FE_CNTR, TS_FE_CNTR;",
	[C_CNTR_EE_FE_ARB_OUT_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in FE_ARB_OUT_EVENT_CNTR is available and is stalled from advancing in the EE's full event processing pipeline. Stalls result from back-pressure limiting the rate at which event requests can enter the EE's full event processing pipeline.",
	[C_CNTR_EE_CE_ARB_OUT_EVENT_CNTR] = "The aggregate number of counting event requests received by the EE from all of its client blocks. This value is measured at the output of the EE's counting event arbiter. Under quiescent conditions, and barring differences arising from differing initialization, this counter should be equal to the sum of: LPE_CE_CNTR, IXE_CE_CNTR, MST_CE_CNTR, PCT_CE_CNTR;",
	[C_CNTR_EE_CE_ARB_OUT_STALL_CNTR] = "The number of core clock (clk) cycles during which an event included in the CE_ARB_OUT_EVENT_CNTR is available and is stalled from advancing in the EE's counting event processing pipeline. Stalls result from back-pressure limiting the rate at which event requests can enter the EE's counting event processing pipeline.",
	[C_CNTR_EE_LPE_QUERY_EQ_NONE_CNTR] = "The number of LPE event queue space query requests, targeting the null event queue ( EQ_NONE ), received by the EE.",
	[C_CNTR_EE_EQS_LPE_QUERY_REQ_CNTR] = "The number of LPE event queue space query requests processed by the EE as measured at the input to the EE's event queue state pipeline. This count excludes requests counted in LPE_QUERY_EQ_NONE_CNTR . LPE event queue space queries have the highest priority among all the request types that arbitrate for access to the event queue state pipeline; therefore, they are never stalled.",
	[C_CNTR_EE_EQS_CSR_REQ_CNTR] = "The number of bubbles inserted into the EE's event queue state pipeline to allow CSR access to memories used while processing other types of requests in the event queue state pipeline.",
	[C_CNTR_EE_EQS_CSR_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_CSR_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline.",
	[C_CNTR_EE_EQS_HWS_INIT_REQ_CNTR] = "The number of event queue state initialization requests ( Section 13.15.34 on page 1077 ) processed by the EE as measured at the input to the EE's event queue state pipeline.",
	[C_CNTR_EE_EQS_HWS_INIT_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_HWS_INIT_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline.",
	[C_CNTR_EE_EQS_EXPIRED_CB_REQ_CNTR] = "The number of expired combining buffers processed by the EE, as measured at the input to the EE's event queue state pipeline. Expired combining buffers are partially full combining buffers that are being released for writing to their event queue because too much time elapsed without additional events arriving to fill the buffer.",
	[C_CNTR_EE_EQS_EXPIRED_CB_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_EXPIRED_CB_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline.",
	[C_CNTR_EE_EQS_STS_UPDT_REQ_CNTR] = "The number of secondary event queue operations processed by the EE, as measured at the input to the EE's event queue state pipeline. Secondary event queue operations are used to: update the status area of the event queue, perform an event queue buffer switch, and, prefetch an event queue address translation;",
	[C_CNTR_EE_EQS_STS_UPDT_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_STS_UPDT_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline. Stalls also occur, for this type of request, if a free combining buffer is not available to allocate to the request.",
	[C_CNTR_EE_EQS_SWS_WR_REQ_CNTR] = "The number of event queue software state fast-path writes processed by the EE, as measured at the input to the EE's event queue state pipeline. Note, if multiple event queue software state fast-path writes targeting the same event queue are received by the EE is quick succession, the processing in the EE's event queue state pipeline of writes targeting the same event queue can be coalesced. When this occurs, this count will be less than the actual number of event queue software state fast-path writes received.",
	[C_CNTR_EE_EQS_SWS_WR_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_SWS_WR_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline. Stalls also occur, for this type of request, if a free combining buffer is not available to allocate to the request.",
	[C_CNTR_EE_EQS_EVENT_REQ_CNTR] = "The number of full event requests processed by the EE, as measured at the input to the EE's event queue state pipeline. Under quiescent conditions and, and barring differences arising from differing initialization, this counter should be equal to FE_ARB_OUT_EVENT_CNTR .",
	[C_CNTR_EE_EQS_EVENT_STALL_CNTR] = "The number of core clock (clk) cycles during which a request included in EQS_EVENT_REQ_CNTR is available and is stalled from advancing in the EE's event queue state pipeline. Stalls result from arbitration with other types of requests entering the event queue state pipeline. Stalls also occur, for this type of request, if a free combining buffer is not available to allocate to the request. This counter is closely related to FE_ARB_OUT_STALL_CNTR . FE_ARB_OUT_STALL_CNTR includes stalls due to certain CSR accesses that are not included in this counter.",
	[C_CNTR_EE_EQS_ARB_OUT_REQ_CNTR] = "The number of requests processed by the EE's event queue state pipeline. Under quiescent conditions, and barring differences arising from differing initialization, this counter should be equal to the sum of: EQS_LPE_QUERY_REQ_CNTR, EQS_CSR_REQ_CNTR, EQS_HWS_INIT_REQ_CNTR, EQS_EXPIRED_CB_REQ_CNTR, EQS_STS_UPDT_REQ_CNTR, EQS_SWS_WR_REQ_CNTR, EQS_EVENT_REQ_CNTR;",
	[C_CNTR_EE_EQS_FREE_CB_STALL_CNTR] = "The number of core clock (clk) cycles during which the only types of requests that are available to enter the EE's event queue state pipeline are requests that can only advance if a free combining buffer is available, and there are no free combining buffers available. This is a measure of the amount of time the event queue state pipeline is stalled due to there not being a free combining buffers as soon as one is needed.",
	[C_CNTR_EE_ADDR_TRANS_REQ_CNTR] = "The aggregate number of address translation requests issued to the ATU",
	[C_CNTR_EE_TARB_WR_REQ_CNTR] = "The number of memory write requests the EE has issued to the TARB. Each write request writes the data of one combining buffer, containing one or more full events, or an update of an event queue status area.",
	[C_CNTR_EE_TARB_IRQ_REQ_CNTR] = "The number of interrupt requests the EE has issued to the TARB",
	[C_CNTR_EE_TARB_STALL_CNTR] = "The number of core clock (clk) cycles during which the EE has either a TARB memory write request or interrupt request pending, which it is unable to forward to the TARB because of back-pressure from the TARB",
	[C_CNTR_EE_EVENT_PUT_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_GET_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ATOMIC_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_FETCH_ATOMIC_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_PUT_OVERFLOW_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_GET_OVERFLOW_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ATOMIC_OVERFLOW_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_FETCH_ATOMIC_OVERFLOW_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_SEND_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ACK_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_REPLY_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_LINK_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_SEARCH_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_STATE_CHANGE_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_UNLINK_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_RENDEZVOUS_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ETHERNET_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_COMMAND_FAILURE_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_TRIGGERED_OP_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ETHERNET_FGFC_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_PCT_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_MATCH_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_ERROR_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_TIMESTAMP_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_EQ_SWITCH_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_EVENT_NULL_EVENT_CNTR] = "For each possible type of event, the number of events of that type that have been processed in the EE's event queue state pipeline. These are events that will be written to an event queue, but at the time this counter is incremented, the event is still resident in a combining buffer which has not yet been written to the event queue. These counters exclude deferred-drop events. Null events are not counted unless an error is reported in the event's return code. The lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 0. The second to lowest indexed counter, in this vector of counters, corresponds to an event with field EVENT_TYPE equal to 1, and so on. Some possible EVENT_TYPE values are unassigned, and so some counters in this vector are not expected to ever increment.",
	[C_CNTR_EE_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory, bus, and state vector errors that have occurred. This counter is only incremented for correctable memory, bus, and state vector errors that are not masked in CSR C_EE_EXT_ERR_INFO_MSK ( Section 13.15.23 on page 1061 ). The errors which can contribute to this counter are: LEO_MEM_COR, PTS_MEM_COR, IXE_BUS_COR, LPE_BUS_COR, MST_BUS_COR, PCT_BUS_COR, HNI_BUS_COR, CQ_BUS_COR, EQD_MEM_COR, EQSWS_MEM_COR, EQHWS_MEM_COR, ECBSB_MEM_COR, ECB_MEM_COR, RARB_HDR_BUS_COR, RARB_DATA_BUS_COR, ACB_VECT_COR, EQSWUPD_VECT_COR, LM_CBEQ_MEM_COR, LM_STATE_VECT_COR, TRNRSP_MEM_COR, TRNRSP_BUS_COR, RLSCB_VECT_COR, TRNCB_VECT_COR;",
	[C_CNTR_EE_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory, bus, and state vector errors that have been detected. This counter is only incremented for uncorrectable memory, bus, and state vector errors that are not masked in CSR C_EE_EXT_ERR_INFO_MSK ( Section 13.15.23 on page 1061 ). The errors which can contribute to this counter are: LEO_MEM_UCOR, PTS_MEM_UCOR, IXE_BUS_UCOR, LPE_BUS_UCOR, MST_BUS_UCOR, PCT_BUS_UCOR, HNI_BUS_UCOR, CQ_BUS_UCOR, EQD_MEM_UCOR, EQSWS_MEM_UCOR, EQHWS_MEM_UCOR, ECBSB_MEM_UCOR, ECB_MEM_UCOR, RARB_HDR_BUS_UCOR, RARB_DATA_BUS_UCOR, ACB_VECT_UCOR, LM_CBEQ_MEM_UCOR, TRNRSP_MEM_UCOR, TRNRSP_BUS_UCOR, RLSCB_VECT_UCOR, TRNCB_VECT_UCOR;",
	[C_CNTR_EE_EQ_DSABLD_EVENT_ERR_CNTR] = "This is a count of the number of requests to generate full events that have been received, but dropped because the request targeted a disabled event queue. Conditions that cause this counter to increment also set error flag EQ_DSABLD_EVENT , described in Section 13.15.8 on page 1035 .",
	[C_CNTR_EE_EQ_DSABLD_SWS_ERR_CNTR] = "This is a count of the number of event queue software state fast-path writes that software has performed and that target an event queue that is disabled. Conditions that cause this counter to increment also set error flag EQ_DSABLD_SW_STATE_WR , described in Section 13.15.8 on page 1035 .",
	[C_CNTR_EE_EQ_DSABLD_LPEQ_ERR_CNTR] = "This is a count of the number of LPE Queries, targeting disabled event queues, that have been received. Conditions that cause this counter to increment also set error flag EQ_DSABLD_LPE_QUERY , described in Section 13.15.8 on page 1035 .",
	[C_CNTR_EE_EQ_RSRVN_UFLW_ERR_CNTR] = "This is a count of the number of full event requests received, for which space should have been reserved by the LPE in the targeted event queue, but for which sufficient reserved space was not available in the event queue. Conditions that cause this counter to increment also set error flag EQ_RSRVN_UFLW , described in Section 13.15.8 on page 1035 .",
	[C_CNTR_EE_UNXPCTD_TRNSLTN_RSP_ERR_CNTR] = "This is a count of the number of translation responses received from the ATU that were unexpected. Conditions that cause this counter to increment also set error flag UNXPCTD_TRNSLTN_RSP , described in Section 13.15.8 on page 1035 . More information about the conditions that cause this error are included in the description of this error flag.",
	[C_CNTR_EE_RARB_HW_ERR_CNTR] = "This is a count of the number of corrupt requests received from the RARB and for which the corruption is likely caused by a hardware problem. Conditions that cause this counter to increment also set error flag RARB_HW_ERR , described in Section 13.15.8 on page 1035 . More information about the conditions that cause this error are included in the description of this error flag.",
	[C_CNTR_EE_RARB_SW_ERR_CNTR] = "This is a count of the number of invalid fast-path write requests received from the RARB. The problem is caused by an MMIO request that has been incorrectly formed by software. Conditions that cause this counter to increment also set error flag RARB_SW_ERR , described in Section 13.15.8 on page 1035 . More information about the conditions that cause this error are included in the description of this error flag.",
	[C_CNTR_EE_TARB_ERR_CNTR] = "This is a count of the number of requests to the TARB that have been dropped because an error prevents the request from being correctly issued. Conditions that cause this counter to increment also set error flag TARB_ERR , described in Section 13.15.8 on page 1035 . More information about the conditions that cause this error are included in the description of this error flag.",
	[C_CNTR_EE_EQ_STATE_UCOR_ERR_CNTR] = "This is a count of the number of operations that were dropped by the event queue state pipeline, without being processed, because an uncorrectable error was detected in the event queue state while performing the operation. Conditions that cause this counter to increment also set error flag EQ_STATE_UCOR , described in Section 13.15.8 on page 1035 . More information about the conditions that cause this error are included in the description of this error flag.",
	[C_CNTR_LPE_SUCCESS_CNTR] = "The total number of network requests and commands completed successfully by LPE",
	[C_CNTR_LPE_FAIL_CNTR] = "The total number of network requests and commands unsuccessfully completed by LPE",
	[C_CNTR_LPE_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_LPE_ERR_INFO_MSK ( Section 13.13.14 on page 967 ). The errors which can contribute to this counter are listed in Table 334 .",
	[C_CNTR_LPE_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_LPE_ERR_INFO_MSK ( Section 13.13.14 on page 967 ). The errors which can contribute to this counter are listed in Table 334 .",
	[C_CNTR_LPE_PUT_CMDS] = "Number of Put commands received by LPE",
	[C_CNTR_LPE_RENDEZVOUS_PUT_CMDS] = "Number of Rendezvous Put commands received by LPE",
	[C_CNTR_LPE_GET_CMDS] = "Number of Get commands received by LPE",
	[C_CNTR_LPE_AMO_CMDS] = "Number of non-fetching AMO commands received by LPE",
	[C_CNTR_LPE_FAMO_CMDS] = "Number of fetching AMO commands received by LPE",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_ETHERNET] = "Count of the number of Ethernet requests dropped because an entry was not found in the searched list (Priority or Request)",
	[C_CNTR_LPE_ERR_PT_DISABLED_EQ_FULL] = "Count of LPE errors. PtlTE disabled because event queue was full.",
	[C_CNTR_LPE_ERR_PT_DISABLED_REQ_EMPTY] = "Count of LPE errors. PtlTE disabled because request queue was empty",
	[C_CNTR_LPE_ERR_PT_DISABLED_CANT_ALLOC_UNEXPECTED] = "Count of LPE errors. PtlTE disabled because space could not be allocated on the Unexpected list and the Request queue was empty.",
	[C_CNTR_LPE_ERR_PT_DISABLED_NO_MATCH_IN_OVERFLOW] = "Count of LPE errors. PtlTE disabled because there was no match on the Overflow list. i.e. no space for the payload of an Unexpected message",
	[C_CNTR_LPE_ERR_EQ_DISABLED] = "Count of LPE errors. Request was dropped because target event queue was disabled.",
	[C_CNTR_LPE_ERR_OP_VIOLATION] = "Count of LPE errors. Request was dropped because opcode did not match the options on the matched buffer.",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_UNLINK_FAILED] = "Count of Unlink commands that failed to match an entry",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_EQ_FULL] = "Count of requests dropped because the event queue was full and flow control was disabled",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_REQ_EMPTY] = "Count of requests dropped because request queue was empty and flow control was disabled",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_NO_MATCH_IN_OVERFLOW] = "Count of the number of requests dropped because no match was found on the Overflow list and flow control was disabled.",
	[C_CNTR_LPE_ERR_ENTRY_NOT_FOUND_CANT_ALLOC_UNEXPECTED] = "Count of the number of requests dropped because it was not possible to allocate a list entry on the Unexpected list, the Request list was empty, and flow control was disabled,",
	[C_CNTR_LPE_ERR_INVALID_ENDPOINT] = "Count of invalid endpoint errors",
	[C_CNTR_LPE_ERR_NO_SPACE_APPEND] = "Count of Append commands rejected because a list entry could not be allocated",
	[C_CNTR_LPE_ERR_NO_SPACE_NET] = "Count of requests dropped due to matching a list entry without enough space on a non-space-checking interface",
	[C_CNTR_LPE_ERR_SEARCH_NO_MATCH] = "Count of search events that return NO_MATCH",
	[C_CNTR_LPE_ERR_SETSTATE_NO_MATCH] = "Count of failed SetState commands",
	[C_CNTR_LPE_ERR_SRC_ERROR] = "Count of the number of requests dropped because an error was detected at the source",
	[C_CNTR_LPE_ERR_PTLTE_SW_MANAGED] = "Count of Append command rejected for targeting a software managed interface",
	[C_CNTR_LPE_ERR_ILLEGAL_OP] = "Count of illegal opcodes in Portals packets",
	[C_CNTR_LPE_ERR_RESTRICTED_UNICAST] = "A unicast restricted operation targets a locally managed list entry and EN_RESTRICTED_UNICAST_LM is clear",
	[C_CNTR_LPE_EVENT_PUT_OVERFLOW] = "Count of the number of Put overflow events sent to EE",
	[C_CNTR_LPE_EVENT_GET_OVERFLOW] = "Count of the number of Get overflow events sent to EE",
	[C_CNTR_LPE_EVENT_ATOMIC_OVERFLOW] = "Count of the number of Atomic overflow events sent to EE",
	[C_CNTR_LPE_PLEC_FREES_CSR] = "Count of the number of frees in the Persistent List Entry Cache resulting from CSR writes to the PtlTE",
	[C_CNTR_LPE_EVENT_FETCH_ATOMIC_OVERFLOW] = "Count of the number of Fetching Atomic overflow events sent to EE",
	[C_CNTR_LPE_EVENT_SEARCH] = "Count of the number of Search events sent to EE",
	[C_CNTR_LPE_EVENT_RENDEZVOUS] = "Count of the number of Rendezvous events sent to EE",
	[C_CNTR_LPE_EVENT_LINK] = "Count of the number of Link events sent to EE",
	[C_CNTR_LPE_EVENT_UNLINK] = "Count of the number of Unlink events sent to EE",
	[C_CNTR_LPE_EVENT_STATECHANGE] = "Count of the number of State Change events sent to EE",
	[C_CNTR_LPE_PLEC_ALLOCS] = "Count of the number of allocations in the Persistent List Entry Cache",
	[C_CNTR_LPE_PLEC_FREES] = "Count of the number of frees in the Persistent List Entry Cache due to functional Unlinks or replacements",
	[C_CNTR_LPE_PLEC_HITS] = "Count of the number of hits in the Persistent List Entry Cache",
	[C_CNTR_LPE_CYC_NO_RDY_CDTS] = "Count of the number of cycles in which an IXE MRQ dequeue could not be initiated because no Ready PLEQ credits were available",
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_0] = "Count of the number of cycles on which a PE Match Request Queue dequeue was blocked because a Ready Request Queue was full. One counter for each of the four PEs",
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_1] = "Count of the number of cycles on which a PE Match Request Queue dequeue was blocked because a Ready Request Queue was full. One counter for each of the four PEs",
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_2] = "Count of the number of cycles on which a PE Match Request Queue dequeue was blocked because a Ready Request Queue was full. One counter for each of the four PEs",
	[C_CNTR_LPE_CYC_RRQ_BLOCKED_3] = "Count of the number of cycles on which a PE Match Request Queue dequeue was blocked because a Ready Request Queue was full. One counter for each of the four PEs",
	[C_CNTR_LPE_CYC_BLOCKED_LOSSLESS] = "Count of the number of cycles on which a PtlTE was blocked because the required list was empty and the PtlTE was in lossless mode",
	[C_CNTR_LPE_CYC_BLOCKED_IXE_PUT] = "Count of the number of cycles on which LPE was blocked sending a Put request to IXE",
	[C_CNTR_LPE_CYC_BLOCKED_IXE_GET] = "Count of the number of cycles on which LPE was blocked sending a Get request to IXE",
	[C_CNTR_LPE_SEARCH_CMDS] = "The number of Search commands LPE received",
	[C_CNTR_LPE_SEARCH_SUCCESS] = "The number of successful matches generated by Search commands",
	[C_CNTR_LPE_SEARCH_FAIL] = "The number of Search commands that did not match",
	[C_CNTR_LPE_SEARCH_DELETE_CMDS] = "The number of Search&Delete commands LPE received",
	[C_CNTR_LPE_SEARCH_DELETE_SUCCESS] = "The number of match list entries deleted by Search&Delete command",
	[C_CNTR_LPE_SEARCH_DELETE_FAIL] = "The number of Search&Delete commands that did not find a match.",
	[C_CNTR_LPE_UNLINK_CMDS] = "The number of Unlink commands LPE received",
	[C_CNTR_LPE_UNLINK_SUCCESS] = "The number of Unlink commands successfully matched",
	[C_CNTR_LPE_NET_MATCH_REQUESTS_0] = "The number of network requests LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUESTS_1] = "The number of network requests LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUESTS_2] = "The number of network requests LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUESTS_3] = "The number of network requests LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_SUCCESS_0] = "The number of network requests LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_SUCCESS_1] = "The number of network requests LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_SUCCESS_2] = "The number of network requests LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_SUCCESS_3] = "The number of network requests LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_USEONCE_0] = "The number of network requests LPE successfully matched to use once buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_USEONCE_1] = "The number of network requests LPE successfully matched to use once buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_USEONCE_2] = "The number of network requests LPE successfully matched to use once buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_USEONCE_3] = "The number of network requests LPE successfully matched to use once buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_LOCAL_0] = "The number of network requests LPE successfully matched to locally managed buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_LOCAL_1] = "The number of network requests LPE successfully matched to locally managed buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_LOCAL_2] = "The number of network requests LPE successfully matched to locally managed buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_LOCAL_3] = "The number of network requests LPE successfully matched to locally managed buffers. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_PRIORITY_0] = "The number of network requests LPE successfully matched on the Priority list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_PRIORITY_1] = "The number of network requests LPE successfully matched on the Priority list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_PRIORITY_2] = "The number of network requests LPE successfully matched on the Priority list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_PRIORITY_3] = "The number of network requests LPE successfully matched on the Priority list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_0] = "The number of network requests LPE successfully matched on the Overflow list. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_1] = "The number of network requests LPE successfully matched on the Overflow list. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_2] = "The number of network requests LPE successfully matched on the Overflow list. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_OVERFLOW_3] = "The number of network requests LPE successfully matched on the Overflow list. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUEST_0] = "The number of network requests LPE successfully matched on the Request list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUEST_1] = "The number of network requests LPE successfully matched on the Request list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUEST_2] = "The number of network requests LPE successfully matched on the Request list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NET_MATCH_REQUEST_3] = "The number of network requests LPE successfully matched on the Request list.One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_CMDS_0] = "The number of Append commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_CMDS_1] = "The number of Append commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_CMDS_2] = "The number of Append commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_CMDS_3] = "The number of Append commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_SUCCESS_0] = "The number of Append commands LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_SUCCESS_1] = "The number of Append commands LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_SUCCESS_2] = "The number of Append commands LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_APPEND_SUCCESS_3] = "The number of Append commands LPE successfully completed. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_CMDS_0] = "The number of SetState commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_CMDS_1] = "The number of SetState commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_CMDS_2] = "The number of SetState commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_CMDS_3] = "The number of SetState commands LPE received. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_SUCCESS_0] = "The number of successful SetState commands. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_SUCCESS_1] = "The number of successful SetState commands. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_SUCCESS_2] = "The number of successful SetState commands. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SETSTATE_SUCCESS_3] = "The number of successful SetState commands. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_NID_ANY_0] = "The number of searches of all kinds received by LPE that used NID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_NID_ANY_1] = "The number of searches of all kinds received by LPE that used NID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_NID_ANY_2] = "The number of searches of all kinds received by LPE that used NID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_NID_ANY_3] = "The number of searches of all kinds received by LPE that used NID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_PID_ANY_0] = "The number of searches of all kinds received by LPE that used PID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_PID_ANY_1] = "The number of searches of all kinds received by LPE that used PID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_PID_ANY_2] = "The number of searches of all kinds received by LPE that used PID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_PID_ANY_3] = "The number of searches of all kinds received by LPE that used PID_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_RANK_ANY_0] = "The number of searches of all kinds received by LPE that used RANK_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_RANK_ANY_1] = "The number of searches of all kinds received by LPE that used RANK_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_RANK_ANY_2] = "The number of searches of all kinds received by LPE that used RANK_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_SEARCH_RANK_ANY_3] = "The number of searches of all kinds received by LPE that used RANK_ANY. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_0] = "The number of Rendezvous Puts received by LPE. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_1] = "The number of Rendezvous Puts received by LPE. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_2] = "The number of Rendezvous Puts received by LPE. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_3] = "The number of Rendezvous Puts received by LPE. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_0] = "The number of Rendezvous Puts that LPE was able to offload. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_1] = "The number of Rendezvous Puts that LPE was able to offload. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_2] = "The number of Rendezvous Puts that LPE was able to offload. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_RNDZV_PUTS_OFFLOADED_3] = "The number of Rendezvous Puts that LPE was able to offload. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NUM_TRUNCATED_0] = "The number of packets that were truncated. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NUM_TRUNCATED_1] = "The number of packets that were truncated. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NUM_TRUNCATED_2] = "The number of packets that were truncated. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_NUM_TRUNCATED_3] = "The number of packets that were truncated. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_0] = "The number of Get and AMO packets that match on Overflow or Request, resulting in RC_DELAYED. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_1] = "The number of Get and AMO packets that match on Overflow or Request, resulting in RC_DELAYED. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_2] = "The number of Get and AMO packets that match on Overflow or Request, resulting in RC_DELAYED. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_LPE_UNEXPECTED_GET_AMO_3] = "The number of Get and AMO packets that match on Overflow or Request, resulting in RC_DELAYED. One counter for each of the four counter pools ( CNTR_POOL ) defined in C_LPE_CFG_PTL_TABLE[2048] .",
	[C_CNTR_IXE_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_IXE_ERR_INFO_MSK (see Section 13.10.14 on page 757 ). The errors which can contribute to this counter are: FLOP_FIFO_COR, PUT_RESP_MEM_COR, GET_REQ_MEM_COR, RDFQ_LIST_MEM_COR, WR_MD_MEM_COR, EVENT_FIFO_COR, WR_EV_MEM_COR, PKT_FLIT_FIFO_COR, PBUF_LIST_MEM_COR, PBUF_SB_MEM_COR, ATU_RESP_COR, RDOQ_COR, INGRESSQ_COR, INFO_COR, CMD_INFO_COR, SIDEBAND_COR, P_CMD_COR, NP_CMD_COR, AMO_INFO_COR, ADDR_COR, RD_BUF_COR, ACK_RD_BUF_COR, HNI_FLIT_COR, RPU_LIST_MEM_COR, RPU_ST_MEM_COR, RARB_HDR_BUS_COR, RARB_DATA_BUS_COR, PBUF_FLIT_MEM_COR;",
	[C_CNTR_IXE_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_IXE_ERR_INFO_MSK (see Section 13.10.14 on page 757 ). The errors which can contribute to this counter are: FLOP_FIFO_UCOR, PUT_RESP_MEM_UCOR, GET_REQ_MEM_UCOR, RDFQ_LIST_MEM_UCOR, WR_MD_MEM_UCOR, EVENT_FIFO_UCOR, WR_EV_MEM_UCOR, PKT_FLIT_FIFO_UCOR, PBUF_LIST_MEM_UCOR, PBUF_SB_MEM_UCOR, ATU_RESP_UCOR, RDOQ_UCOR, INGRESSQ_UCOR, INFO_UCOR, CMD_INFO_UCOR, SIDEBAND_UCOR, P_CMD_UCOR, NP_CMD_UCOR, AMO_INFO_UCOR, ADDR_UCOR, RD_BUF_UCOR, ACK_RD_BUF_UCOR, HNI_FLIT_UCOR, RPU_LIST_MEM_UCOR, RPU_ST_MEM_UCOR, RARB_HDR_BUS_UCOR, RARB_DATA_BUS_UCOR, PBUF_FLIT_MEM_UCOR;",
	[C_CNTR_IXE_PORT_DFA_MISMATCH] = "Packets discarded because the DFA was incorrect ( PORT_DFA_MISMATCH )",
	[C_CNTR_IXE_HDR_CHECKSUM_ERRORS] = "Packets discarded because header checksum was incorrect ( HDR_CHECKSUM_ERR )",
	[C_CNTR_IXE_IPV4_CHECKSUM_ERRORS] = "Standard IPv4 packets where a header checksum error was detected ( IPV4_CHECKSUM_ERR )",
	[C_CNTR_IXE_HRP_REQ_ERRORS] = "Total number of packets where a HRP request error was detected ( HRP_REQ_ERR )",
	[C_CNTR_IXE_IP_OPTIONS_ERRORS] = "Total number of IP packets where a options error was detected ( IPV4_OPTIONS_ERR , IPV6_OPTIONS_ERR )",
	[C_CNTR_IXE_GET_LEN_ERRORS] = "Total number of packets where a get packet length error was detected ( GET_LEN_ERR )",
	[C_CNTR_IXE_ROCE_ICRC_ERROR] = "Total number of packets where ROCE_ICRC_ERR was detected",
	[C_CNTR_IXE_PARSER_PAR_ERRORS] = "Total number of parity errors detected in the parser ( PBUF_PKT_PERR , PKT_FLIT_FIFO_PERR )",
	[C_CNTR_IXE_PBUF_RD_ERRORS] = "Total number of packet buffer read errors detected ( PBUF_RD_ERR )",
	[C_CNTR_IXE_HDR_ECC_ERRORS] = "Total number of header ECC errors detected ( HDR_ECC_ERR )",
	[C_CNTR_IXE_RX_UDP_PKT] = "Total number of UDP packets received",
	[C_CNTR_IXE_RX_TCP_PKT] = "Total number of TCP packets received",
	[C_CNTR_IXE_RX_IPV4_PKT] = "Total number of IPv4 packets received",
	[C_CNTR_IXE_RX_IPV6_PKT] = "Total number of IPv6 packets received",
	[C_CNTR_IXE_RX_ROCE_PKT] = "Total number of RoCE packets received",
	[C_CNTR_IXE_RX_PTL_GEN_PKT] = "Total number of generic portals packets received",
	[C_CNTR_IXE_RX_PTL_SML_PKT] = "Total number of small portals packets received",
	[C_CNTR_IXE_RX_PTL_UNRESTRICTED_PKT] = "Total number of portals unrestricted packets received",
	[C_CNTR_IXE_RX_PTL_SMALLMSG_PKT] = "Total number of portals small message packets received",
	[C_CNTR_IXE_RX_PTL_CONTINUATION_PKT] = "Total number of portals continuation packets received",
	[C_CNTR_IXE_RX_PTL_RESTRICTED_PKT] = "Total number of portals restricted packets received",
	[C_CNTR_IXE_RX_PTL_CONNMGMT_PKT] = "Total number of portals connection management packets received",
	[C_CNTR_IXE_RX_PTL_RESPONSE_PKT] = "Total number of portals response packets received",
	[C_CNTR_IXE_RX_UNRECOGNIZED_PKT] = "Total number of packets received which were not UDP, TCP, or Portals",
	[C_CNTR_IXE_RX_PTL_SML_AMO_PKT] = "Total number of small AMO packets received",
	[C_CNTR_IXE_RX_PTL_MSGS] = "Total number of Portals messages received",
	[C_CNTR_IXE_RX_PTL_MULTI_MSGS] = "Number of Portals multi-packet messages received",
	[C_CNTR_IXE_RX_PTL_MR_MSGS] = "Number of Portals multicast/reduction messages received",
	[C_CNTR_IXE_RX_PKT_DROP_PCT] = "Total number of received packets which were dropped due to a drop indication from the PCT. This may be due to a sequence number mismatch, lack of target resources, or an error within the PCT.",
	[C_CNTR_IXE_RX_PKT_DROP_RMU_NORSP] = "Total number of received packets which were dropped due to the RMU returned a non-OK return code where no response is returned indicating an error. This applies to Ethernet, multicast, and HRP requests. Due to erratum 3115 this counter will incorrectly not increment for multicast or HRP requests.",
	[C_CNTR_IXE_RX_PKT_DROP_RMU_WRSP] = "Total number of received packets which were dropped due to the RMU returning a non-OK return code where a response is returned indicating an error. This only applies to Portals request packets. Due to erratum 3115 this counter will also incorrectly increment for multicast or HRP requests.",
	[C_CNTR_IXE_RX_PKT_DROP_IXE_PARSER] = "Total number of received packets which were dropped due to an error detected within the IXE parser. No response will be returned for these drops.",
	[C_CNTR_IXE_RX_PKT_IPV4_OPTIONS] = "Total number of IPv4 packets received which have optional headers",
	[C_CNTR_IXE_RX_PKT_IPV6_OPTIONS] = "Total number of IPv6 packets received which have optional headers",
	[C_CNTR_IXE_RX_ETH_SEG] = "Total number of Ethernet packets which were segmented into multiple buffers",
	[C_CNTR_IXE_RX_ROCE_SEG] = "Total number of RoCE packets which were segmented like a Ethernet packet",
	[C_CNTR_IXE_RX_ROCE_SPSEG] = "Total number of RoCE packets which match an enabled payload length (1K, 2K, 4K) and were segmented using the special RoCE segmentation (header in one segment, payload in the other, ICRC stripped)",
	[C_CNTR_IXE_POOL_ECN_PKTS_0] = "Number of packets with ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_ECN_PKTS_1] = "Number of packets with ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_ECN_PKTS_2] = "Number of packets with ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_ECN_PKTS_3] = "Number of packets with ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_0] = "Number of packets without ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_1] = "Number of packets without ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_2] = "Number of packets without ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_POOL_NO_ECN_PKTS_3] = "Number of packets without ECN set. One counter for each of the four pools of PtlTEs.",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_0] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_1] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_2] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_3] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_4] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_5] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_6] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_ECN_PKTS_7] = "Number of request packets with ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_0] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_1] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_2] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_3] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_4] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_5] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_6] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_REQ_NO_ECN_PKTS_7] = "Number of request packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_0] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_1] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_2] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_3] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_4] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_5] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_6] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_ECN_PKTS_7] = "Number of response packets with ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_0] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_1] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_2] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_3] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_4] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_5] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_6] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_TC_RSP_NO_ECN_PKTS_7] = "Number of response packets without ECN set, by TC",
	[C_CNTR_IXE_DISP_LPE_PUTS] = "Number of Put headers received from LPE",
	[C_CNTR_IXE_DISP_LPE_PUTS_OK] = "Number of RC_OK Put headers received from LPE",
	[C_CNTR_IXE_DISP_LPE_AMOS] = "Number of AMO headers received from LPE",
	[C_CNTR_IXE_DISP_LPE_AMOS_OK] = "Number of RC_OK AMO headers received from LPE",
	[C_CNTR_IXE_DISP_MST_PUTS] = "Number of Put headers received from MST",
	[C_CNTR_IXE_DISP_MST_PUTS_OK] = "Number of RC_OK Put headers received from MST",
	[C_CNTR_IXE_DISP_LPE_GETS] = "Number of Get headers received from LPE",
	[C_CNTR_IXE_DISP_LPE_GETS_OK] = "Number of RC_OK Get headers received from LPE",
	[C_CNTR_IXE_DISP_MST_GETS] = "Number of Get headers received from MST",
	[C_CNTR_IXE_DISP_MST_GETS_OK] = "Number of RC_OK Get headers received from MST",
	[C_CNTR_IXE_DISP_RPU_RESPS] = "Number of responses from RPU",
	[C_CNTR_IXE_DISP_RPU_ERR_REQS] = "Number of errored requests from RPU",
	[C_CNTR_IXE_DISP_DMAWR_REQS] = "Number of requests to DMA Write Controller",
	[C_CNTR_IXE_DISP_DMAWR_RESPS] = "Number of responses from DMA Write Controller",
	[C_CNTR_IXE_DISP_DMAWR_RESPS_OK] = "Number of RC_OK responses from DMA Write Controller",
	[C_CNTR_IXE_DISP_OXE_RESPS] = "Number of responses to OXE",
	[C_CNTR_IXE_DISP_PCT_GETCOMP] = "Number of Get completions to PCT",
	[C_CNTR_IXE_DISP_ATU_REQS] = "Number of requests to ATU",
	[C_CNTR_IXE_DISP_ATU_RESPS] = "Number of responses from ATU",
	[C_CNTR_IXE_DISP_ATU_RESPS_OK] = "Number of RC_OK responses from ATU",
	[C_CNTR_IXE_DISP_ETH_EVENTS] = "Number of Ethernet events sent to EE",
	[C_CNTR_IXE_DISP_PUT_EVENTS] = "Number of single-packet Put events sent to EE",
	[C_CNTR_IXE_DISP_AMO_EVENTS] = "Number of AMO events sent to EE",
	[C_CNTR_IXE_DISP_WR_CONFLICTS] = "Number of write requests that fail conflict checking",
	[C_CNTR_IXE_DISP_AMO_CONFLICTS] = "Number of AMO requests that fail conflict checking",
	[C_CNTR_IXE_DISP_STALL_CONFLICT] = "Number of cycles when at least one conflicted header is stalled",
	[C_CNTR_IXE_DISP_STALL_RESP_FIFO] = "Number of stalls due to no response FIFO credits",
	[C_CNTR_IXE_DISP_STALL_ERR_FIFO] = "Number of stalls due to no error FIFO credits",
	[C_CNTR_IXE_DISP_STALL_ATU_CDT] = "Number of stalls due to no ATU credits",
	[C_CNTR_IXE_DISP_STALL_ATU_FIFO] = "Number of stalls due to no ATU FIFO credits",
	[C_CNTR_IXE_DISP_STALL_GCOMP_FIFO] = "Number of stalls due to no PCT Get completions FIFO credits",
	[C_CNTR_IXE_DISP_STALL_MDID_CDTS] = "Number of stalls due to no MDID credits",
	[C_CNTR_IXE_DISP_STALL_MST_MATCH_FIFO] = "Number of stalls due to no MST Match FIFO credits",
	[C_CNTR_IXE_DISP_STALL_GRR_ID] = "Number of stalls due to no Get Response Request IDs",
	[C_CNTR_IXE_DISP_STALL_PUT_RESP] = "Number of stalls due to no Put Response credits",
	[C_CNTR_IXE_DISP_ATU_FLUSH_REQS] = "Number of ATU flush requests",
	[C_CNTR_IXE_DMAWR_STALL_P_CDT] = "Number of stalls due to no Posted Credits",
	[C_CNTR_IXE_DMAWR_STALL_NP_CDT] = "Number of stalls due to no Non-Posted Credits",
	[C_CNTR_IXE_DMAWR_STALL_NP_REQ_CNT] = "Number of stalls due to outstanding Non-Posted Request count",
	[C_CNTR_IXE_DMAWR_STALL_FTCH_AMO_CNT] = "Number of stalls due to outstanding Fetching AMO Request count",
	[C_CNTR_IXE_DMAWR_P_PASS_NP_CNT] = "Number of times a Posted Request passed a Non-Posted request because the Non-Posted request was stalled due to no Credits or at outstanding request count",
	[C_CNTR_IXE_DMAWR_WRITE_REQS] = "Number of Write requests to the DMA Write Controller",
	[C_CNTR_IXE_DMAWR_NIC_AMO_REQS] = "Number of NIC AMO requests to the DMA Write Controller",
	[C_CNTR_IXE_DMAWR_CPU_AMO_REQS] = "Number of CPU AMO requests to the DMA Write Controller",
	[C_CNTR_IXE_DMAWR_CPU_FTCH_AMO_REQS] = "Number of Fetching CPU AMO requests to the DMA Write Controller",
	[C_CNTR_IXE_DMAWR_FLUSH_REQS] = "Number of requests to the DMA Write Controller that required a Flush",
	[C_CNTR_IXE_DMAWR_REQ_NO_WRITE_REQS] = "Number of requests to the DMA Write Controller that were not Ok To Write",
	[C_CNTR_IXE_DMAWR_AMO_EX_INVALID] = "Number of invalid exceptions detected by the AMO unit",
	[C_CNTR_IXE_DMAWR_AMO_EX_OVERFLOW] = "Number of overflow exceptions detected by the AMO unit",
	[C_CNTR_IXE_DMAWR_AMO_EX_UNDERFLOW] = "Number of underflow exceptions detected by the AMO unit",
	[C_CNTR_IXE_DMAWR_AMO_EX_INEXACT] = "Number of inexact exceptions detected by the AMO unit",
	[C_CNTR_IXE_DMAWR_AMO_MISALIGNED] = "Number of misaligned AMO operations",
	[C_CNTR_IXE_DMAWR_AMO_INVALID_OP] = "Number of invalid AMO Operations",
	[C_CNTR_IXE_DMAWR_AMO_LEN_ERR] = "Number of AMO Operations with incorrect lengths",
	[C_CNTR_IXE_DMAWR_PCIE_UNSUCCESS_CMPL] = "Number of PCIe Reads that resulted in Unsuccessful Completion",
	[C_CNTR_IXE_DMAWR_PCIE_ERR_POISONED] = "Number of PCIe Reads that returned with Error Poisoned",
	[C_CNTR_RMU_PTL_SUCCESS_CNTR] = "This is a count of the number of Portals requests that have been received and for which the RMU was able to successfully identify a physical Portal Table entry to be used to process the request.",
	[C_CNTR_RMU_ENET_SUCCESS_CNTR] = "This is a count of the number of Ethernet frames that have been received and for which the RMU was able to successfully identify a physical Portal Table entry to be used to process the frame.",
	[C_CNTR_RMU_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_RMU_ERR_INFO_MSK ( Section 13.12.14 on page 915 ). The errors which can contribute to this counter are: PTL_IDX_MEM_COR, PTLTE_SET_CTRL_MEM_COR, PTL_IDX_INDIR_MEM_COR;",
	[C_CNTR_RMU_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_RMU_ERR_INFO_MSK ( Section 13.12.14 on page 915 ). The errors which can contribute to this counter are: VNI_LIST_MEM_UCOR, VNI_LIST_VLD_MEM_UCOR, PTL_LIST_MEM_UCOR, PTL_LIST_VLD_MEM_UCOR, PTL_IDX_MEM_UCOR, PTLTE_SET_LIST_MEM_UCOR, PTLTE_SET_LIST_VLD_MEM_UCOR, PTLTE_SET_CTRL_MEM_UCOR, PTL_IDX_INDIR_MEM_UCOR;",
	[C_CNTR_RMU_PTL_INVLD_DFA_CNTR] = "This is a count of the number of Portals requests that have been received and rejected because they contained an invalid DFA. The DFAs of Portals requests must be of either the unicast or multicast format. The corresponding error flag is PTL_INVLD_DFA and the conditions that cause Portals requests to be dropped for this reason are described in Section 13.12.7 on page 908 , in the definition of this error flag.",
	[C_CNTR_RMU_PTL_INVLD_VNI_CNTR] = "This is a count of the number of Portals requests that have been received and rejected because they contained a VNI that is not present in the VNI List ( Section 13.12.21 on page 924 ). The corresponding error flag is PTL_INVLD_VNI and the conditions that cause Portals requests to be dropped for this reason are described in Section 13.12.7 on page 908 , in the definition of this error flag.",
	[C_CNTR_RMU_PTL_NO_PTLTE_CNTR] = "This is a count of the number of Portals requests that have been received and rejected because there was no matching entry in the Portal List ( Section 13.12.23 on page 928 ). The corresponding error flag is PTL_NO_PTLTE and the conditions that cause Portals requests to be dropped for this reason are described in Section 13.12.7 on page 908 , in the definition of this error flag.",
	[C_CNTR_RMU_ENET_FRAME_REJECTED_CNTR] = "This is a count of the number of Ethernet frames that have been received and rejected because there were no matching entry in the Portal Table Entry Set List ( Section 13.12.28 on page 943 ). The corresponding error flag is ENET_FRAME_REJECTED and the conditions that cause Ethernet frames to be dropped for this reason are described in Section 13.12.7 on page 908 , in the definition of this error flag.",
	[C_CNTR_RMU_ENET_PTLTE_SET_CTRL_ERR_CNTR] = "This is a count of the number of Ethernet frames that have been received and rejected because they encountered a PtlTE Set Control Table entry ( Section 13.12.29 on page 944 ) with an invalid configuration ( Section 13.12.20 on page 923 ). The corresponding error flag is ENET_PTLTE_SET_CTRL_ERR and the conditions that cause Ethernet frames to be dropped for this reason are described in Section 13.12.7 on page 908 , in the definition of this error flag.",
	[C_CNTR_PARBS_TARB_CQ_POSTED_PKTS] = "The number of posted packets received from the CQ",
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_PKTS] = "The number of non-posted packets received from the CQ",
	[C_CNTR_PARBS_TARB_CQ_SBE_CNT] = "The number of SBEs received from the CQ",
	[C_CNTR_PARBS_TARB_CQ_MBE_CNT] = "The number of MBEs received from the CQ",
	[C_CNTR_PARBS_TARB_EE_POSTED_PKTS] = "The number of posted packets received from the EE",
	[C_CNTR_PARBS_TARB_EE_SBE_CNT] = "The number of SBEs received from the EE",
	[C_CNTR_PARBS_TARB_EE_MBE_CNT] = "The number of MBEs received from the EE",
	[C_CNTR_PARBS_TARB_IXE_POSTED_PKTS] = "The number of posted packets received from the IXE",
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_PKTS] = "The number of non-posted packets received from the IXE",
	[C_CNTR_PARBS_TARB_IXE_SBE_CNT] = "The number of SBEs received from the IXE",
	[C_CNTR_PARBS_TARB_IXE_MBE_CNT] = "The number of MBEs received from the IXE",
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_PKTS] = "The number of non-posted packets received from the OXE",
	[C_CNTR_PARBS_TARB_OXE_SBE_CNT] = "The number of SBEs received from the OXE",
	[C_CNTR_PARBS_TARB_OXE_MBE_CNT] = "The number of MBEs received from the OXE",
	[C_CNTR_PARBS_TARB_ATU_POSTED_PKTS] = "The number of posted packets received from the ATU",
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_PKTS] = "The number of non-posted packets received from the ATU",
	[C_CNTR_PARBS_TARB_ATU_SBE_CNT] = "The number of SBEs received from the ATU",
	[C_CNTR_PARBS_TARB_ATU_MBE_CNT] = "The number of MBEs received from the ATU",
	[C_CNTR_PARBS_TARB_PI_POSTED_PKTS] = "The number of posted packets sent to the PI",
	[C_CNTR_PARBS_TARB_PI_NON_POSTED_PKTS] = "The number of non-posted packets sent to the PI",
	[C_CNTR_PARBS_TARB_PI_POSTED_BLOCKED_CNT] = "The number of cycles in which the TARB was blocked waiting for posted credits to send to the PI",
	[C_CNTR_PARBS_TARB_PI_NON_POSTED_BLOCKED_CNT] = "The number of cycles in which the TARB was blocked waiting for non-posted credits to send to the PI",
	[C_CNTR_PARBS_TARB_CQ_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the CQ posted FIFO",
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the CQ non-posted FIFO",
	[C_CNTR_PARBS_TARB_EE_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the EE posted FIFO",
	[C_CNTR_PARBS_TARB_IXE_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the IXE posted FIFO",
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the IXE non-posted FIFO",
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the OXE non-posted FIFO",
	[C_CNTR_PARBS_TARB_ATU_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the ATU posted FIFO",
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_FIFO_SBE] = "The number of SBEs detected while reading from the ATU non-posted FIFO",
	[C_CNTR_PARBS_TARB_CQ_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the CQ posted FIFO",
	[C_CNTR_PARBS_TARB_CQ_NON_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the CQ non-posted FIFO",
	[C_CNTR_PARBS_TARB_EE_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the EE posted FIFO",
	[C_CNTR_PARBS_TARB_IXE_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the IXE posted FIFO",
	[C_CNTR_PARBS_TARB_IXE_NON_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the IXE non-posted FIFO",
	[C_CNTR_PARBS_TARB_OXE_NON_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the OXE non-posted FIFO",
	[C_CNTR_PARBS_TARB_ATU_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the ATU posted FIFO",
	[C_CNTR_PARBS_TARB_ATU_NON_POSTED_FIFO_MBE] = "The number of MBEs detected while reading from the ATU non-posted FIFO",
	[C_CNTR_PARBS_RARB_PI_POSTED_PKTS] = "The number of posted packets received from the PI.",
	[C_CNTR_PARBS_RARB_PI_COMPLETION_PKTS] = "The number of completion packets received from the PI.",
	[C_CNTR_PARBS_RARB_PI_SBE_CNT] = "The number of SBEs received from the PI",
	[C_CNTR_PARBS_RARB_PI_MBE_CNT] = "The number of SBEs received from the PI",
	[C_CNTR_PARBS_RARB_CQ_POSTED_PKTS] = "The number of posted packets sent to the CQ",
	[C_CNTR_PARBS_RARB_CQ_COMPLETION_PKTS] = "Then number of completion packets sent to the CQ",
	[C_CNTR_PARBS_RARB_EE_POSTED_PKTS] = "The number of posted packets sent to the EE",
	[C_CNTR_PARBS_RARB_IXE_POSTED_PKTS] = "The number of posted packets sent to the IXE",
	[C_CNTR_PARBS_RARB_IXE_COMPLETION_PKTS] = "Then number of completion packets sent to the IXE",
	[C_CNTR_PARBS_RARB_OXE_COMPLETION_PKTS] = "Then number of completion packets sent to the OXE",
	[C_CNTR_PARBS_RARB_ATU_POSTED_PKTS] = "The number of posted packets sent to the ATU",
	[C_CNTR_PARBS_RARB_ATU_COMPLETION_PKTS] = "Then number of completion packets sent to the ATU",
	[C_CNTR_MST_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_MST_ERR_INFO_MSK ( Section 13.14.15 on page 1013 ). See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_MST_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_MST_ERR_INFO_MSK ( Section 13.14.15 on page 1013 ). See erratum 3325 for cases where errors may be over-counted.",
	[C_CNTR_MST_ALLOCATIONS] = "Count of the number of MST allocations",
	[C_CNTR_MST_REQUESTS] = "Count of the number of MST requests",
	[C_CNTR_MST_STALLED_WAITING_FOR_LPE] = "Cycles on which MST has requests, but could not return message data because it was waiting for match results from LPE.",
	[C_CNTR_MST_STALLED_WAITING_GET_CRDTS] = "Number of cycles stalled waiting for GET credits from IXE Due to errata 3137 this counter is no longer usable. See errata for alternate counters.",
	[C_CNTR_MST_STALLED_WAITING_PUT_CRDTS] = "Number of cycles stalled waiting for PUT credits from IXE Due to errata 3137 this counter is no longer usable. See errata for alternate counters.",
	[C_CNTR_MST_STALLED_WAITING_EE_CRDTS] = "Number of cycles stalled waiting for credits from EE. Due to errata 3137 this counter is no longer usable. See errata for alternate counters.",
	[C_CNTR_MST_ERR_CANCELLED] = "Count of the number of times an error occurred because the MST entry was in the process of being canceled by software (RC_MST_CANCELLED)",
	[C_CNTR_HNI_COR_ECC_ERR_CNTR] = "This is a count of the number of correctable ECC errors that have occurred. This counter is only incremented for correctable ECC errors that are not masked in CSR C_HNI_ERR_INFO_MSK ( Section 13.6.15 on page 485 ). The errors which can contribute to this counter are: LLR_FLIT_COR, RX_STAT_FIFO_COR, RX_PKT_FIFO_COR, TIMESTAMP_COR, HRP_FIFO_COR, FGFC_FIFO_COR, TX_FLIT_COR, EVENT_RAM_COR;",
	[C_CNTR_HNI_UCOR_ECC_ERR_CNTR] = "This is a count of the number of uncorrectable ECC errors that have been detected. This counter is only incremented for uncorrectable ECC errors that are not masked in CSR C_HNI_ERR_INFO_MSK ( Section 13.6.15 on page 485 ). The errors which can contribute to this counter are: LLR_FLIT_UCOR, RX_STAT_FIFO_UCOR, RX_PKT_FIFO_UCOR, TIMESTAMP_UCOR, HRP_FIFO_UCOR, FGFC_FIFO_UCOR, TX_FLIT_UCOR, EVENT_RAM_UCOR;",
	[C_CNTR_HNI_TX_STALL_LLR] = "Count for each clock the Tx path is stalled due to lack of credits to the LLR block (port macro).",
	[C_CNTR_HNI_RX_STALL_IXE_FIFO] = "Count for each clock the Rx path is stalled due to lack of credits to the input FIFO within the IXE. This counter does not work properly as described in the errata for ID 2527 .",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_0] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_1] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_2] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_3] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_4] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_5] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_6] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_RX_STALL_IXE_PKTBUF_7] = "Count for each clock the Rx path of the corresponding traffic class is stalled due to lack of space in the IXE Packet Buffer.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_0] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_1] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_2] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_3] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_4] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_5] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_6] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_PFC_FIFO_OFLW_CNTR_7] = "Count for each packet discarded at the tail of the PFC FIFO. This can happen either if it has exceed the MFS setting for that traffic class or the FIFO is nearly full. This equates to each time the PFC_FIFO_OFLW error flag sets for a particular TC.",
	[C_CNTR_HNI_DISCARD_CNTR_0] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_1] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_2] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_3] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_4] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_5] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_6] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_DISCARD_CNTR_7] = "Count for each packet discard due to a timeout for each traffic class as indicated by DISCARD_ERR . In Cassini 1.1, due to erratum 2964 this counter will be the number of times discard has started for this traffic class. It can be used to approximate the number of packets discarded by multiplying it by ROUNDUP(( HIGH_WATER - LOW_WATER )/avg_pkt_size).",
	[C_CNTR_HNI_TX_OK_IEEE] = "IEEE frames transmitted with a good FCS.",
	[C_CNTR_HNI_TX_OK_OPT] = "OPT frames transmitted with a good FCS.",
	[C_CNTR_HNI_TX_POISONED_IEEE] = "IEEE frames transmitted with a poisoned FCS.",
	[C_CNTR_HNI_TX_POISONED_OPT] = "OPT frames transmitted with a poisoned FCS.",
	[C_CNTR_HNI_TX_OK_BROADCAST] = "Broadcast frames transmitted with a good FCS. Broadcast is indicated by a DMAC of all 1's.",
	[C_CNTR_HNI_TX_OK_MULTICAST] = "Multicast frames transmitted with a good FCS.Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's",
	[C_CNTR_HNI_TX_OK_TAGGED] = "Frames transmitted with a VLAN tag and a good FCS.",
	[C_CNTR_HNI_TX_OK_27] = "Frames transmitted with a length of 27 bytes and a good FCS. Counts small packets with no payload.",
	[C_CNTR_HNI_TX_OK_35] = "Frames transmitted with a length of 35 bytes and a good FCS. Counts small packets with an 8-byte payload.",
	[C_CNTR_HNI_TX_OK_36_TO_63] = "Frames transmitted with a length of between 36 and 63 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_64] = "Frames transmitted with a length of 64 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_65_TO_127] = "Frames transmitted with a length of between 65 and 127 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_128_TO_255] = "Frames transmitted with a length of between 128 and 255 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_256_TO_511] = "Frames transmitted with a length of between 256 and 511 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_512_TO_1023] = "Frames transmitted with a length of between 512 and 1023 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_1024_TO_2047] = "Frames transmitted with a length of between 1024 and 2047 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_2048_TO_4095] = "Frames transmitted with a length of between 2048 and 4095 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_4096_TO_8191] = "Frames transmitted with a length of between 4096 and 8191 bytes and a good FCS.",
	[C_CNTR_HNI_TX_OK_8192_TO_MAX] = "Frames transmitted with a length of between 8192 and MAX_LEN bytes and a good FCS.",
	[C_CNTR_HNI_RX_OK_IEEE] = "IEEE frames received with a good FCS, no EOPB, and no MBE",
	[C_CNTR_HNI_RX_OK_OPT] = "OPT frames received with a good FCS, no EOPB, and no MBE",
	[C_CNTR_HNI_RX_BAD_IEEE] = "IEEE frames received with a bad FCS, EOPB, PERR, or MBE. The error info mask does not effect this count. The error flags which may cause this counter to increment are: LLR_FCS_BAD, LLR_EOPB, LLR_CTRL_PERR, LLR_FLIT_UCOR; This counter will also increment when a poisoned FCS is received (which does not set LLR_FCS_BAD ).",
	[C_CNTR_HNI_RX_BAD_OPT] = "OPT frames received with a bad FCS, EOPB, PERR, or MBE. The error info mask does not effect this count. The error flags which may cause this counter to increment are: LLR_CHKSUM_BAD, LLR_FCS_BAD, LLR_EOPB, LLR_CTRL_PERR, LLR_FLIT_UCOR; This counter will also increment when a poisoned FCS is received (which does not set LLR_FCS_BAD ).",
	[C_CNTR_HNI_RX_OK_BROADCAST] = "Broadcast frames received with a good FCS, no EOPB, and no MBE. Broadcast is indicated by a DMAC of all 1's.",
	[C_CNTR_HNI_RX_OK_MULTICAST] = "Multicast frames received with a good FCS, no EOPB, and no MBE. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's",
	[C_CNTR_HNI_RX_OK_MAC_CONTROL] = "MAC control frames received with a good FCS, no EOPB, and no MBE.",
	[C_CNTR_HNI_RX_OK_BAD_OP_CODE] = "MAC control frames received with an opcode that is not pause or pfc pause opcode and a good FCS, no EOPB, and no MBE.",
	[C_CNTR_HNI_RX_OK_FLOW_CONTROL] = "Flow control (pause, pfc pause) frames received with a good FCS, no EOPB, and no MBE.",
	[C_CNTR_HNI_RX_OK_TAGGED] = "VLAN tagged frames received with a good FCS, no EOPB, and no MBE.",
	[C_CNTR_HNI_RX_OK_LEN_TYPE_ERROR] = "Frames received with a length/type error and a good FCS, no EOPB, and no MBE. The ethertype is <=1500 and therefore represents the length and that length does not match the received byte length.",
	[C_CNTR_HNI_RX_OK_TOO_LONG] = "Frames received that were too long and with a good FCS, no EOPB, and no MBE. The size is configured by C_HNI_CFG_MAX_SIZES with a violation also setting TX_SIZE_ERR .",
	[C_CNTR_HNI_RX_OK_UNDERSIZE] = "Frames received that were too short and with a good FCS, no EOPB, and no MBE. These are IEEE frames which are less than 64 bytes including the FCS32.",
	[C_CNTR_HNI_RX_FRAGMENT] = "Frames received that were too short and with a bad FCS, EOPB, or an MBE. These are IEEE frames which are less than 64 bytes including the FCS32.",
	[C_CNTR_HNI_RX_JABBER] = "Frames received that were too long and with a bad FCS, EOPB, or an MBE. The size is configured by C_HNI_CFG_MAX_SIZES .",
	[C_CNTR_HNI_RX_OK_27] = "Frames received with a length of 27 bytes and a good FCS, no EOPB and no MBE. Counts small packets with no payload.",
	[C_CNTR_HNI_RX_OK_35] = "Frames received with a length of 35 bytes and a good FCS.Counts small packets with an 8-byte payload.",
	[C_CNTR_HNI_RX_OK_36_TO_63] = "Frames received with a length of between 36 and 63 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_64] = "Frames received with a length of 64 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_65_TO_127] = "Frames received with a length of between 65 and 127 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_128_TO_255] = "Frames received with a length of between 128 and 255 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_256_TO_511] = "Frames received with a length of between 256 and 511 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_512_TO_1023] = "Frames received with a length of between 512 and 1023 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_1024_TO_2047] = "Frames received with a length of between 1024 and 2047 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_2048_TO_4095] = "Frames received with a length of between 2048 and 4095 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_4096_TO_8191] = "Frames received with a length of between 4096 and 8191 bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_RX_OK_8192_TO_MAX] = "Frames received with a length of between 8192 and MAX_LEN bytes and a good FCS, no EOPB and no MBE.",
	[C_CNTR_HNI_HRP_ACK] = "Count of the number of HRP Ack frames received",
	[C_CNTR_HNI_HRP_RESP] = "Count of the number of HRP responses generated as a result of HRP Ack frames being received",
	[C_CNTR_HNI_FGFC_PORT] = "Count of number of FGFC frames received with type of Portals",
	[C_CNTR_HNI_FGFC_L2] = "Count of number of FGFC frames received with type of L2",
	[C_CNTR_HNI_FGFC_IPV4] = "Count of number of FGFC frames received with type of IPv4",
	[C_CNTR_HNI_FGFC_IPV6] = "Count of number of FGFC frames received with type of IPv6",
	[C_CNTR_HNI_FGFC_MATCH] = "Count of number of FGFC frames of type L2, IPv4, IPv6 matched an entry in the FGFC Cache",
	[C_CNTR_HNI_FGFC_EVENT_XON] = "Count of the number of FGFC events generated with XON",
	[C_CNTR_HNI_FGFC_EVENT_XOFF] = "Count of the number of FGFC events generated with XOFF",
	[C_CNTR_HNI_FGFC_DISCARD] = "Count of the number of FGFC frames that are discarded due to lack of a FGFC cache entry (TBD error flag).",
	[C_CNTR_HNI_PAUSE_RECV_0] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_1] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_2] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_3] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_4] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_5] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_6] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_RECV_7] = "Count of the number of pause frames received for each enabled PCP (as identified by the PEV field of the pause frame) when PFC pause is enabled. Reception of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_0] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_1] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_2] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_3] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_4] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_5] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_6] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_PAUSE_XOFF_SENT_7] = "Count of the number of pause frames sent where XOFF is indicated for each PCP when PFC pause is enabled. Transmission of a standard pause frame will cause all counts to increment.",
	[C_CNTR_HNI_RX_PAUSED_0] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_1] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_2] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_3] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_4] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_5] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_6] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_RX_PAUSED_7] = "Count for each system clock the Rx PFC buffer indicates that pause should be set for that PCP.",
	[C_CNTR_HNI_TX_PAUSED_0] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_1] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_2] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_3] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_4] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_5] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_6] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_TX_PAUSED_7] = "Count for each system clock the Tx path is being paused for that PCP. For standard pause all PCPs are paused at the same time so these counters will all count the same.",
	[C_CNTR_HNI_RX_PAUSED_STD] = "Count for each system clock that the Rx PFC buffer indicates that at least one PCP is indicating pause. This can be used to track how many clocks pause is asserted when standard pause is being used.",
	[C_CNTR_HNI_PAUSE_SENT] = "Count of the number of pause frames sent",
	[C_CNTR_HNI_PAUSE_REFRESH] = "Count of the number of pause frames sent to refresh the count",
	[C_CNTR_HNI_TX_STD_PADDED] = "IEEE frame transmitted which was padded by HNI",
	[C_CNTR_HNI_TX_OPT_PADDED] = "OPT frame transmitted which was padded by HNI",
	[C_CNTR_HNI_TX_STD_SIZE_ERR] = "IEEE frame discarded due to it violating the size settings",
	[C_CNTR_HNI_TX_OPT_SIZE_ERR] = "OPT frame discarded due to it violating the size settings",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_0] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_1] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_2] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_3] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_4] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_5] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_6] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_SENT_BY_TC_7] = "Count of the number of packets sent with good FCS by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI.",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_0] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_1] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_2] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_3] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_4] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_5] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_6] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_PKTS_RECV_BY_TC_7] = "Count of the number of packets with good FCS received by TC. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_0] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_1] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_2] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_3] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_4] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_5] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_6] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_SENT_BY_TC_7] = "Count of the number of multicast packets with good FCS sent by TC. The TC is determined by parsing the packet and using the DSCP to PCP mapping in HNI. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's.",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_0] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_1] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_2] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_3] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_4] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_5] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_6] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C_CNTR_HNI_MULTICAST_PKTS_RECV_BY_TC_7] = "Count of the number of multicast packets with good FCS received by TC. Multicast is indicated by DMAC bit 40 being set and the DMAC not being all 1's. Due to errata item 2840 this counter will also increment for received pause frames using the TC pointed to by the DEFAULT_PCP setting within C_HNI_CFG_GEN .",
	[C1_CNTR_HNI_PCS_PML_MBE_ERROR_CNT] = "A count of the number of MBEs across all the memories in the PML. (It is a count of the OR of Err Flags LLR_SEQ_MEM_MBE , LLR_REPLAY_MEM_MBE , PCS_FEC_MEM0_MBE , PCS_FEC_MEM1_MBE in Section 13.7.5 on page 523 )",
	[C1_CNTR_HNI_PCS_PML_SBE_ERROR_CNT] = "A count of the number of SBEs across all the memories in the PML. (It is a count of the OR of Err Flags LLR_SEQ_MEM_SBE , LLR_REPLAY_MEM_SBE , PCS_FEC_MEM0_SBE , PCS_FEC_MEM1_SBE in Section 13.7.5 on page 523 )",
	[C1_CNTR_HNI_PCS_CORRECTED_CW] = "This counter indicates the number of RS codewords that contained errors and that have been corrected.",
	[C1_CNTR_HNI_PCS_UNCORRECTED_CW] = "This counter indicates the number of RS codewords that contained errors that were uncorrectable.",
	[C1_CNTR_HNI_PCS_GOOD_CW] = "This counter indicates the number of codewords received with no errors.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_0] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_1] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_2] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_3] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_4] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_5] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_6] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_PCS_FECL_ERRORS_7] = "This array of counters indicate the number of errors in each FECL.",
	[C1_CNTR_HNI_MAC_RX_ILLEGAL_SIZE] = "This counter indicates that a frame with an illegal size has been received and filtered.",
	[C1_CNTR_HNI_MAC_RX_FRAGMENT] = "This counter indicates that a frame fragment has been received.",
	[C1_CNTR_HNI_MAC_RX_PREAMBLE_ERROR] = "his counter indicates that a frame with a bad preamble has been received.",
	[C1_CNTR_HNI_LLR_TX_LOOP_TIME_REQ_CTL_FR] = "This counter indicates the number of loop time request control frames transmitted.",
	[C1_CNTR_HNI_LLR_TX_LOOP_TIME_RSP_CTL_FR] = "This counter indicates the number of loop time response control frames transmitted.",
	[C1_CNTR_HNI_LLR_TX_INIT_CTL_OS] = "This counter indicates the number of INIT CtlOS transmitted.",
	[C1_CNTR_HNI_LLR_TX_INIT_ECHO_CTL_OS] = "This counter indicates the number of INIT_ECHO CtlOS transmitted.",
	[C1_CNTR_HNI_LLR_TX_ACK_CTL_OS] = "This counter indicates the number of ACK CtlOS transmitted.",
	[C1_CNTR_HNI_LLR_TX_NACK_CTL_OS] = "This counter indicates the number of NACK CtlOS transmitted.",
	[C1_CNTR_HNI_LLR_TX_DISCARD] = "This counter indicates the number of frames that have been discarded by the TX LLR.",
	[C1_CNTR_HNI_LLR_TX_OK_LOSSY] = "This counter indicates the number of lossy frames transmitted with a good FCS.",
	[C1_CNTR_HNI_LLR_TX_OK_LOSSLESS] = "This counter indicates the number of lossless frames transmitted with a good FCS.",
	[C1_CNTR_HNI_LLR_TX_OK_LOSSLESS_RPT] = "This counter indicates the number of repeat lossless frames transmitted with a good FCS.",
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSY] = "This counter indicates the number of lossy frames transmitted with a poisoned FCS.",
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSLESS] = "This counter indicates the number of lossless frames transmitted with a poisoned FCS.",
	[C1_CNTR_HNI_LLR_TX_POISONED_LOSSLESS_RPT] = "This counter indicates the number of repeat lossless frames transmitted with a poisoned FCS.",
	[C1_CNTR_HNI_LLR_TX_OK_BYPASS] = "This counter indicates the number of LLR bypass frames transmitted with a good FCS.",
	[C1_CNTR_HNI_LLR_TX_REPLAY_EVENT] = "This counter indicates the number of replays that the transmitter has performed.",
	[C1_CNTR_HNI_LLR_RX_LOOP_TIME_REQ_CTL_FR] = "This counter indicates the number of loop time request control frames received.",
	[C1_CNTR_HNI_LLR_RX_LOOP_TIME_RSP_CTL_FR] = "This counter indicates the number of loop time response control frames received.",
	[C1_CNTR_HNI_LLR_RX_BAD_CTL_FR] = "This counter indicates the number of control frames received with a bad FCS.",
	[C1_CNTR_HNI_LLR_RX_INIT_CTL_OS] = "This counter indicates the number of INIT CtlOS received.",
	[C1_CNTR_HNI_LLR_RX_INIT_ECHO_CTL_OS] = "This counter indicates the number of INIT_ECHO CtlOS received.",
	[C1_CNTR_HNI_LLR_RX_ACK_CTL_OS] = "This counter indicates the number of ACK CtlOS received.",
	[C1_CNTR_HNI_LLR_RX_NACK_CTL_OS] = "This counter indicates the number of NACK CtlOS received.",
	[C1_CNTR_HNI_LLR_RX_ACK_NACK_SEQ_ERR] = "This counter indicates the number of ACKs and NACKs received with an invalid sequence number",
	[C1_CNTR_HNI_LLR_RX_OK_LOSSY] = "This counter indicates the number of lossy frames received with a good FCS.",
	[C1_CNTR_HNI_LLR_RX_POISONED_LOSSY] = "This counter indicates the number of lossy frames received with a poisoned FCS.",
	[C1_CNTR_HNI_LLR_RX_BAD_LOSSY] = "This counter indicates the number of lossy frames received with a bad FCS.",
	[C1_CNTR_HNI_LLR_RX_OK_LOSSLESS] = "This counter indicates the number of lossless frames received with a good FCS.",
	[C1_CNTR_HNI_LLR_RX_POISONED_LOSSLESS] = "This counter indicates the number of lossless frames received with a poisoned FCS.",
	[C1_CNTR_HNI_LLR_RX_BAD_LOSSLESS] = "This counter indicates the number of lossless frames received with a bad FCS.",
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_GOOD] = "This counter indicates the number of frames received with the expected sequence number and a good FCS.",
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_POISONED] = "This counter indicates the number of frames received with the expected sequence number and a poisoned FCS.",
	[C1_CNTR_HNI_LLR_RX_EXPECTED_SEQ_BAD] = "This counter indicates the number of frames received with the expected sequence number and a bad FCS.",
	[C1_CNTR_HNI_LLR_RX_UNEXPECTED_SEQ] = "This counter indicates the number of frames received with an unexpected sequence number.",
	[C1_CNTR_HNI_LLR_RX_DUPLICATE_SEQ] = "This counter indicates the number of frames received with a duplicate sequence number.",
	[C1_CNTR_HNI_LLR_RX_REPLAY_EVENT] = "This counter indicates the number of replays that the receiver has detected.",
	[C1_CNTR_OXE_MEM_COR_ERR_CNTR] = "This is a count of the number of correctable memory errors that have occurred. This counter is only incremented for correctable memory errors that are not masked in CSR C_OXE_ERR_INFO_MSK ( Section 13.9.15 on page 701 ). The errors which can contribute to this counter are: ..., ...;",
	[C1_CNTR_OXE_MEM_UCOR_ERR_CNTR] = "This is a count of the number of uncorrectable memory errors that have been detected. This counter is only incremented for uncorrectable memory errors that are not masked in CSR C_OXE_ERR_INFO_MSK ( Section 13.9.15 on page 701 ). The errors which can contribute to this counter are: ..., ...;",
	[C1_CNTR_OXE_ERR_NO_TRANSLATION] = "Count of the number of times the  ATS/NTA failed to obtain a translation (RC_NO_TRANSLATION)",
	[C1_CNTR_OXE_ERR_INVALID_AC] = "Count of the number of times the  AC was invalid (RC_INVALID_AC)",
	[C1_CNTR_OXE_ERR_PAGE_PERM] = "Count of the number of times   page permission check failed (RC_PAGE_PERM_ERRROR)",
	[C1_CNTR_OXE_ERR_TA_ERROR] = "Count of the number of times  there was  a translation agent error (RC_ATS_ERROR)",
	[C1_CNTR_OXE_ERR_PAGE_REQ] = "Count of the number of times  there was a page request error (RC_PAGE_REQ_ERROR)",
	[C1_CNTR_OXE_ERR_PCI_EP] = "count number of PCIe returns with client_tlp_ep set",
	[C1_CNTR_OXE_ERR_PCI_CMPL] = "count number of PCIe returns with completion status not successful",
	[C1_CNTR_OXE_STALL_PCT_BC_0] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_1] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_2] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_3] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_4] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_5] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_6] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_7] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_8] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PCT_BC_9] = "The number of clocks a BC  is not allowed to participate in MCUOBC arbitration because it is waiting for PCT resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_0] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_1] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_2] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_3] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_4] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_5] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_6] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_7] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_8] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_PBUF_BC_9] = "The number of clocks a BC is not allowed to participate in MCUOBC arbitration because it is waiting for packet buffer resources. One counter for each BC.Count in PKTBUFF_REQ state.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_0] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_1] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_2] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_3] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_4] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_5] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_6] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_7] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_8] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_OUT_CRD_TSC_9] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for outbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_0] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_1] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_2] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_3] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_4] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_5] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_6] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_7] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_8] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_TS_NO_IN_CRD_TSC_9] = "The number of clocks a TSC is not allowed to participate in traffic shaping arbitration because it is waiting for inbound shaping tokens. One counter for each TSC.",
	[C1_CNTR_OXE_STALL_WITH_NO_PCI_TAG] = "Cycles blocked waiting for PCIe tag",
	[C1_CNTR_OXE_STALL_WITH_NO_ATU_TAG] = "Cycles blocked waiting for an ATU tag",
	[C1_CNTR_OXE_STALL_STFWD_EOP] = "Count of the number of cycles the store-and-forward FIFO is stalled waiting for an EOP.",
	[C1_CNTR_OXE_STALL_PCIE_SCOREBOARD] = "Num cycles PCIe request delayed due to scoreboard (header write not complete)",
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_0] = "Num cycles header write or IDC write collides with PCIe data return. Per bank of Packet buffer",
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_1] = "Num cycles header write or IDC write collides with PCIe data return. Per bank of Packet buffer",
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_2] = "Num cycles header write or IDC write collides with PCIe data return. Per bank of Packet buffer",
	[C1_CNTR_OXE_STALL_WR_CONFLICT_PKT_BUFF_BNK_3] = "Num cycles header write or IDC write collides with PCIe data return. Per bank of Packet buffer",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_0] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_1] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_2] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_3] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_4] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_5] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_6] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_7] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_8] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_NO_BUFF_BC_9] = "Num cycles IDC command has been enqueued (into ENQ FIFO) but cannot get buffer. Counted per BC. This is an OR of all MCUs belonging to the same BC with an IDC command but does not have a cell (buffer) allocated.",
	[C1_CNTR_OXE_STALL_IDC_CMD_NO_DEQ] = "Num cycles IDC command cannot be dequeued after obtaining cell (buffer).This is an OR of all MCUs with an IDC command that have been allocated a cell (buffer) but have not been granted by the command dequeue arbiter (to pop the flow queue in CQ).",
	[C1_CNTR_OXE_STALL_NON_IDC_CMD_NO_DEQ] = "Num cycles non IDC (DMA) commands cannot be dequeued from CQ.This is an OR of all MCUs (from CQ) that have a DMA command enqueued but have not been granted by the command dequeue arbiter (to pop the flow queue in CQ).",
	[C1_CNTR_OXE_STALL_FGFC_BLK_0] = "Cycles blocked for FGFC (matching entry exists and Credit >=0).Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_BLK_1] = "Cycles blocked for FGFC (matching entry exists and Credit >=0).Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_BLK_2] = "Cycles blocked for FGFC (matching entry exists and Credit >=0).Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_BLK_3] = "Cycles blocked for FGFC (matching entry exists and Credit >=0).Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_0] = "Number of cycles a matching FGFC entry exists.Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_1] = "Number of cycles a matching FGFC entry exists.Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_2] = "Number of cycles a matching FGFC entry exists.Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_CNTRL_3] = "Number of cycles a matching FGFC entry exists.Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_START_0] = "Number of FGFC frames received with matching VNI that start or continue FGFC (period != 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_START_1] = "Number of FGFC frames received with matching VNI that start or continue FGFC (period != 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_START_2] = "Number of FGFC frames received with matching VNI that start or continue FGFC (period != 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_START_3] = "Number of FGFC frames received with matching VNI that start or continue FGFC (period != 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_END_0] = "Number of FGFC frames received with matching VNI that end FGFC (period == 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_END_1] = "Number of FGFC frames received with matching VNI that end FGFC (period == 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_END_2] = "Number of FGFC frames received with matching VNI that end FGFC (period == 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_FGFC_END_3] = "Number of FGFC frames received with matching VNI that end FGFC (period == 0)Index given by CSR C_OXE_CFG_FGFC_CNT[4]",
	[C1_CNTR_OXE_STALL_HDR_ARB] = "Cycles waiting for header arb. This is counted in the HEADER state and is the OR of all MCUs that are contending to write their header into the Packet Buffer.",
	[C1_CNTR_OXE_STALL_IOI_LAST_ORDERED] = "Cycles waiting for the last ordered packet to go out. To be counted by doing an OR of all MCUs that are processing a message in IOI mode and are waiting for the last (ordered) packet to go out.",
	[C1_CNTR_OXE_IGNORE_ERRS] = "Count of the number of times an ignore error has been detected in the MCUOBC arb ( SRB_IGNORE_ERR , SCT_IGNORE_ERR , SMT_IGNORE_ERR , SPT_IGNORE_ERR ). This counter is only incremented for ignore errors that are not masked in CSR C_OXE_ERR_INFO_MSK ( Section 13.9.15 on page 701 ).",
	[C1_CNTR_OXE_IOI_DMA] = "Number of IOI DMA messages (not packets). This excludes ethernet, response, restricted and any single MTU messages. Count only when IOI_enable=true. If IOI turns off in the middle of the command (due to assertion of FGFC for the MCU), still count as IOI_DMA.",
	[C1_CNTR_OXE_IOI_PKTS_ORDERED] = "Number of ordered packets sent by IOI DMAs. This is to be counted in the PKTBUFF_GNT state (in the MCUOBC allocator)",
	[C1_CNTR_OXE_IOI_PKTS_UNORDERED] = "Number of unordered packets sent by IOI DMAs. This is to be counted in the PKTBUFF_GNT state (in the MCUOBC allocator).",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_0] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_1] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_2] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_3] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_4] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_5] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_6] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_7] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_8] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_MSGS_TSC_9] = "Count of the number of Portals Put messages sent by traffic shaping class. The following OXE pkt types (c_oxe_pkt_type) are included. PKT_PUT_REQ, PKT_PUT_REQ_NORSP, PKT_AMO_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_0] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_1] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_2] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_3] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_4] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_5] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_6] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_7] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_8] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_GET_MSGS_TSC_9] = "Count of the number of Portals Get messages sent by traffic shaping class.The following OXE pkt types (c_oxe_pkt_type) are included: PKT_GET_REQwhere SOM is set; To be counted in the MCUOBC",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_0] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_1] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_2] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_3] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_4] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_5] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_6] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_7] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_8] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_PUT_PKTS_TSC_9] = "Count of the number of Portals packets by traffic shaping class. To be counted either in the MCUOBC after the MCU is granted or at the Traffic Shaper after packet is selected",
	[C1_CNTR_OXE_PTL_TX_MR_MSGS] = "Number of Portals multicast/reduction messages sent. Will not increment for multicast messages which have been flagged for HRP.",
	[C1_CNTR_OXE_NUM_HRP_CMDS] = "Number of HRP messages sent.To be counted in the MCUOBC",
	[C1_CNTR_OXE_CHANNEL_IDLE] = "Number of cycles traffic shaper head buckets have MTU ceiling tokens available but there is nothing to send. This is an indication of cycles where available channel bandwidth is not being used.To be counted in the Traffic Shaper/Output stage",
	[C1_CNTR_OXE_MCU_MEAS_0] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_1] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_2] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_3] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_4] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_5] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_6] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_7] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_8] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_9] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_10] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_11] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_12] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_13] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_14] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_15] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_16] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_17] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_18] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_19] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_20] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_21] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_22] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_23] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_24] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_25] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_26] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_27] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_28] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_29] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_30] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_31] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_32] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_33] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_34] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_35] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_36] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_37] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_38] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_39] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_40] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_41] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_42] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_43] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_44] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_45] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_46] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_47] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_48] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_49] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_50] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_51] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_52] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_53] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_54] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_55] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_56] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_57] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_58] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_59] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_60] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_61] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_62] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_63] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_64] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_65] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_66] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_67] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_68] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_69] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_70] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_71] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_72] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_73] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_74] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_75] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_76] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_77] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_78] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_79] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_80] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_81] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_82] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_83] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_84] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_85] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_86] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_87] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_88] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_89] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_90] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_91] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_92] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_93] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_94] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_MCU_MEAS_95] = "Count of number of Flit/packets/messages sent by each MCU.The unit of measurement is set by MCU_MEAS_UNIT in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET0_ST_READY] = "count number of cyles MCU in READY state.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_PCT_IDX_WAIT] = "count number of cycles MCU in PCT_IDX_WAIT stateEnabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_PKTBUF_REQ] = "count number of cycles MCU in PKTBUF_REQ stateEnabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_PKTBUF_GNT] = "count number of cycles MCU in PKTBUF_GNT state Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_HEADER] = "count number of cycles MCU in HEADER state. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_DMA_MDR] = "count number of cycles MCU in DMA_MDR state. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_ST_DMA_UPD] = "count number of cycle MCU in DMA_UPD state. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_PKTBUF_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to MTU size buffer not available.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_SPT_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SPT credit not available. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_SMT_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SMT credit not available.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_SRB_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SRB credits not available.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET0_TS_SELECT] = "count number of packets Traffic shaper selects MCU.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96] Due to errata item 2719 this counter is not usable.",
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN0] = "assign current OCC to histogram bin0. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin0 defined by OCC_BIN0_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN1] = "assign current OCC to histogram bin1.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96] .Bin1 defined by OCC_BIN1_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN2] = "assign current OCC to histogram bin2. Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin2 defined by OCC_BIN2_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET0_OCC_HIST_BIN3] = "assign current OCC to histogram bin3.Enabled by STALL_CNT_EN0 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin3 defined by OCC_BIN3_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET1_ST_READY] = "count number of cyles MCU in READY state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_PCT_IDX_WAIT] = "count number of cycles MCU in PCT_IDX_WAIT state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_PKTBUF_REQ] = "count number of cycles MCU in PKTBUF_REQ state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_PKTBUF_GNT] = "count number of cycles MCU in PKTBUF_GNT state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_HEADER] = "count number of cycles MCU in HEADER state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_DMA_MDR] = "count number of cycles MCU in DMA_MDR state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_ST_DMA_UPD] = "count number of cycle MCU in DMA_UPD state. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_PKTBUF_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to MTU size buffer not available. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_SPT_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SPT credit not available. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_SMT_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SMT credit not available. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_SRB_NA] = "count cycles when MCU does not qualify for PKTBUF arb due to SRB credits not available. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96]",
	[C1_CNTR_OXE_PRF_SET1_TS_SELECT] = "count number of packets Traffic shaper selects MCU.Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96] Due to errata item 2719 this counter is not usable.",
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN0] = "assign current OCC to histogram bin0.Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin0 defined by OCC_BIN0_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN1] = "assign current OCC to histogram bin1. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin1 defined by OCC_BIN1_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN2] = "assign current OCC to histogram bin2. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin2 defined by OCC_BIN2_SZ in CSR C_OXE_CFG_COMMON",
	[C1_CNTR_OXE_PRF_SET1_OCC_HIST_BIN3] = "assign current OCC to histogram bin3. Enabled by STALL_CNT_EN1 given by CSR C_OXE_CFG_MCU_PARAM[96] Bin3 defined by OCC_BIN3_SZ in CSR C_OXE_CFG_COMMON",
};

#pragma GCC diagnostic pop
#endif /* __cplusplus */

#endif
