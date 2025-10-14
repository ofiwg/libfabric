// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
 * Platform-specific definitions of SBL_* macros.
 * Copyright 2020-2023 Cray Inc. All rights reserved
 */
#ifndef _SBL_PML_H_
#define _SBL_PML_H_

#include "cassini_user_defs.h"
#include "cassini_csr_defaults.h"
#include "sbl_csr_common.h"
#include "cassini_cntr_defs.h"

/* BASE defines */
#define SBL_PML_BASE(a) C_HNI_PML_BASE

#define SBL_PCS_FECL_ERRORS_00_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_0 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_01_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_1 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_02_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_2 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_03_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_3 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_04_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_4 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_05_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_5 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_06_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_6 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_FECL_ERRORS_07_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_FECL_ERRORS_7 - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_CORRECTED_CW_ADDR(a)    C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_CORRECTED_CW - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_PCS_UNCORRECTED_CW_ADDR(a)  C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_PCS_UNCORRECTED_CW - C_CNTR_HNI_COR_ECC_ERR_CNTR)
#define SBL_LLR_TX_REPLAY_EVENT_ADDR(a) C_HNI_STS_EVENT_CNTS(C1_CNTR_HNI_LLR_TX_REPLAY_EVENT - C_CNTR_HNI_COR_ECC_ERR_CNTR)

/* OFFSET defines */
#define SBL_PML_CFG_LLR_CAPACITY_OFFSET          C1_HNI_PML_CFG_LLR_CAPACITY_OFFSET
#define SBL_PML_CFG_LLR_CF_ETYPE_OFFSET          C1_HNI_PML_CFG_LLR_CF_ETYPE_OFFSET
#define SBL_PML_CFG_LLR_CF_SMAC_OFFSET           C1_HNI_PML_CFG_LLR_CF_SMAC_OFFSET
#define SBL_PML_CFG_LLR_CF_RATES_OFFSET          C1_HNI_PML_CFG_LLR_CF_RATES_OFFSET
#define SBL_PML_CFG_LLR_OFFSET                   C1_HNI_PML_CFG_LLR_OFFSET
#define SBL_PML_CFG_LLR_SM_OFFSET                C1_HNI_PML_CFG_LLR_SM_OFFSET
#define SBL_PML_CFG_LLR_TIMEOUTS_OFFSET          C1_HNI_PML_CFG_LLR_TIMEOUTS_OFFSET
#define SBL_PML_CFG_PCS_AMS_OFFSET               C1_HNI_PML_CFG_PCS_AMS_OFFSET
#define SBL_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET
#define SBL_PML_CFG_PCS_AUTONEG_OFFSET           C1_HNI_PML_CFG_PCS_AUTONEG_OFFSET
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET    C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET
#define SBL_PML_CFG_PCS_CM_OFFSET                C1_HNI_PML_CFG_PCS_CM_OFFSET
#define SBL_PML_CFG_PCS_OFFSET                   C1_HNI_PML_CFG_PCS_OFFSET
#define SBL_PML_CFG_PCS_UM_OFFSET                C1_HNI_PML_CFG_PCS_UM_OFFSET
#define SBL_PML_CFG_RX_MAC_OFFSET                C1_HNI_PML_CFG_RX_MAC_OFFSET
#define SBL_PML_CFG_RX_PCS_AMS_OFFSET            C1_HNI_PML_CFG_RX_PCS_AMS_OFFSET
#define SBL_PML_CFG_RX_PCS_OFFSET                C1_HNI_PML_CFG_RX_PCS_OFFSET
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_OFFSET C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT_OFFSET
#define SBL_PML_CFG_TX_MAC_OFFSET                C1_HNI_PML_CFG_TX_MAC_OFFSET
#define SBL_PML_CFG_TX_PCS_OFFSET                C1_HNI_PML_CFG_TX_PCS_OFFSET
#define SBL_PML_DBG_PCS_OFFSET                   C1_HNI_PML_DBG_PCS_OFFSET
#define SBL_PML_ERR_CLR_OFFSET                   C1_HNI_PML_ERR_CLR_OFFSET
#define SBL_PML_ERR_FLG_OFFSET                   C1_HNI_PML_ERR_FLG_OFFSET
#define SBL_PML_SERDES_CORE_INTERRUPT_OFFSET     C1_HNI_PML_SERDES_CORE_INTERRUPT_OFFSET
#define SBL_PML_SERDES_CORE_STATUS_OFFSET        C1_HNI_PML_SERDES_CORE_STATUS_OFFSET
#define SBL_PML_STS_LLR_LOOP_TIME_OFFSET         C1_HNI_PML_STS_LLR_LOOP_TIME_OFFSET
#define SBL_PML_STS_LLR_OFFSET                   C1_HNI_PML_STS_LLR_OFFSET
#define SBL_PML_STS_LLR_USAGE_OFFSET             C1_HNI_PML_STS_LLR_USAGE_OFFSET
#define SBL_PML_STS_LLR_MAX_USAGE_OFFSET         C1_HNI_PML_STS_LLR_MAX_USAGE_OFFSET
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET C1_HNI_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET C1_HNI_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET
#define SBL_PML_STS_RX_PCS_OFFSET                C1_HNI_PML_STS_RX_PCS_OFFSET
#define SBL_PML_STS_PCS_LANE_DEGRADE_OFFSET      C1_HNI_PML_STS_PCS_LANE_DEGRADE_OFFSET
#define SBL_PML_STS_RX_PCS_DESKEW_DEPTHS_OFFSET  C1_HNI_PML_STS_RX_PCS_DESKEW_DEPTHS_OFFSET
#define SBL_PML_STS_RX_PCS_AM_MATCH_OFFSET       C1_HNI_PML_STS_RX_PCS_AM_MATCH_OFFSET
#define SBL_PML_STS_RX_PCS_FECL_SOURCES_OFFSET   C1_HNI_PML_STS_RX_PCS_FECL_SOURCES_OFFSET

/* UPDATE defines */
#define SBL_PML_CFG_LLR_ACK_NACK_ERR_CHECK_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_llr,ack_nack_err_check)
#define SBL_PML_CFG_LLR_ENABLE_LOOP_TIMING_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_llr,enable_loop_timing)
#define SBL_PML_CFG_LLR_FILTER_CTL_FRAMES_UPDATE(a,b)                          C_UPDATE(a,b,c1_hni_pml_cfg_llr,filter_ctl_frames)
#define SBL_PML_CFG_LLR_FILTER_LOSSLESS_WHEN_OFF_UPDATE(a,b)                   C_UPDATE(a,b,c1_hni_pml_cfg_llr,filter_lossless_when_off)
#define SBL_PML_CFG_LLR_LINK_DOWN_BEHAVIOR_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_llr,link_down_behavior)
#define SBL_PML_CFG_LLR_MAC_IF_CREDITS_UPDATE(a,b)                             C_UPDATE(a,b,c1_hni_pml_cfg_llr,mac_if_credits)
#define SBL_PML_CFG_LLR_LLR_MODE_UPDATE(a,b)                                   C_UPDATE(a,b,c1_hni_pml_cfg_llr,llr_mode)
#define SBL_PML_CFG_LLR_CF_RATES_LOOP_TIMING_PERIOD_UPDATE(a,b)                C_UPDATE(a,b,c_hni_pml_cfg_llr_cf_rates, loop_timing_period)
#define SBL_PML_CFG_LLR_PREAMBLE_SEQ_CHECK_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_llr,preamble_seq_check)
#define SBL_PML_CFG_LLR_SIZE_UPDATE(a,b)                                       C_UPDATE(a,b,c1_hni_pml_cfg_llr,size)
#define SBL_PML_CFG_LLR_SM_REPLAY_CT_MAX_UPDATE(a,b)                           C_UPDATE(a,b,c_hni_pml_cfg_llr_sm,replay_ct_max)
#define SBL_PML_CFG_LLR_SM_REPLAY_TIMER_MAX_UPDATE(a,b)                        C_UPDATE(a,b,c_hni_pml_cfg_llr_sm,replay_timer_max)
#define SBL_PML_CFG_LLR_TIMEOUTS_DATA_AGE_TIMER_MAX_UPDATE(a,b)                C_UPDATE(a,b,c_hni_pml_cfg_llr_timeouts,data_age_timer_max)
#define SBL_PML_CFG_LLR_TIMEOUTS_PCS_LINK_DN_TIMER_MAX_UPDATE(a,b)             C_UPDATE(a,b,c_hni_pml_cfg_llr_timeouts,pcs_link_dn_timer_max)
#define SBL_PML_CFG_PCS_AMS_USE_PROGRAMMABLE_AMS_UPDATE(a,b)                   C_UPDATE(a,b,c_hni_pml_cfg_pcs_ams,use_programmable_ams)
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_LOADED_UPDATE(a,b)                   C_UPDATE(a,b,c_hni_pml_cfg_pcs_autoneg,next_page_loaded)
#define SBL_PML_CFG_RX_PCS_ALLOW_AUTO_DEGRADE_UPDATE(a,b)                      C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,allow_auto_degrade)
#define SBL_PML_CFG_PCS_AUTONEG_RESET_UPDATE(a,b)                              C_UPDATE(a,b,c_hni_pml_cfg_pcs_autoneg,reset)
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_BREAK_LINK_TIMER_MAX_UPDATE(a,b)        C_UPDATE(a,b,c1_hni_pml_cfg_pcs_autoneg_timers,break_link_timer_max)
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_LINK_FAIL_INHIBIT_TIMER_MAX_UPDATE(a,b) C_UPDATE(a,b,c1_hni_pml_cfg_pcs_autoneg_timers,link_fail_inhibit_timer_max)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_UPDATE(a,b)                   C_UPDATE(a,b,c1_hni_pml_cfg_pcs,enable_auto_lane_degrade)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_NEG_UPDATE(a,b)                            C_UPDATE(a,b,c1_hni_pml_cfg_pcs,enable_auto_neg)
#define SBL_PML_CFG_PCS_PCS_ENABLE_UPDATE(a,b)                                 C_UPDATE(a,b,c1_hni_pml_cfg_pcs,pcs_enable)
#define SBL_PML_CFG_PCS_PCS_MODE_UPDATE(a,b)                                   C_UPDATE(a,b,c1_hni_pml_cfg_pcs,pcs_mode)
#define SBL_PML_CFG_PCS_BY_25G_FEC_MODE_UPDATE(a,b)                            C_UPDATE(a,b,c1_hni_pml_cfg_pcs, by_25g_fec_mode)
#define SBL_PML_CFG_PCS_TIMESTAMP_SHIFT_UPDATE(a,b)                            C_UPDATE(a,b,c1_hni_pml_cfg_pcs, timestamp_shift)
#define SBL_PML_CFG_RX_MAC_FILTER_ILLEGAL_SIZE_UPDATE(a,b)                     C_UPDATE(a,b,c1_hni_pml_cfg_rx_mac,filter_illegal_size)
#define SBL_PML_CFG_RX_MAC_MAC_OPERATIONAL_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_rx_mac,mac_operational)
#define SBL_PML_CFG_RX_MAC_SHORT_PREAMBLE_UPDATE(a,b)                          C_UPDATE(a,b,c1_hni_pml_cfg_rx_mac,short_preamble)
#define SBL_PML_CFG_RX_PCS_ACTIVE_LANES_UPDATE(a,b)                            C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,active_lanes)
#define SBL_PML_CFG_RX_PCS_ENABLE_CTL_OS_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,enable_ctl_os)
#define SBL_PML_CFG_RX_PCS_ENABLE_LOCK_UPDATE(a,b)                             C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,enable_lock)
#define SBL_PML_CFG_RX_PCS_ENABLE_RX_SM_UPDATE(a,b)                            C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,enable_rx_sm)
#define SBL_PML_CFG_RX_PCS_HEALTH_BAD_SENSITIVITY_UPDATE(a,b)                  C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,health_bad_sensitivity)
#define SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_AMS_UPDATE(a,b)                 C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,restart_lock_on_bad_ams)
#define SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_CWS_UPDATE(a,b)                 C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,restart_lock_on_bad_cws)
#define SBL_PML_CFG_RX_PCS_RS_MODE_UPDATE(a,b)                                 C_UPDATE(a,b,c1_hni_pml_cfg_rx_pcs,rs_mode)
#define SBL_PML_CFG_TX_MAC_IEEE_IFG_ADJUSTMENT_UPDATE(a,b)                     C_UPDATE(a,b,c1_hni_pml_cfg_tx_mac,ieee_ifg_adjustment)
#define SBL_PML_CFG_TX_MAC_IFG_MODE_UPDATE(a,b)                                C_UPDATE(a,b,c1_hni_pml_cfg_tx_mac,ifg_mode)
#define SBL_PML_CFG_TX_MAC_MAC_OPERATIONAL_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_tx_mac,mac_operational)
#define SBL_PML_CFG_TX_MAC_PCS_CREDITS_UPDATE(a,b)                             C_UPDATE(a,b,c1_hni_pml_cfg_tx_mac,pcs_credits)
#define SBL_PML_CFG_TX_MAC_SHORT_PREAMBLE_UPDATE(a,b)                          C_UPDATE(a,b,c1_hni_pml_cfg_tx_mac,short_preamble)
#define SBL_PML_CFG_TX_PCS_CDC_READY_LEVEL_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,cdc_ready_level)
#define SBL_PML_CFG_TX_PCS_ENABLE_CTL_OS_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,enable_ctl_os)
#define SBL_PML_CFG_TX_PCS_GEARBOX_CREDITS_UPDATE(a,b)                         C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,gearbox_credits)
#define SBL_PML_CFG_TX_PCS_LANE_0_SOURCE_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,lane_0_source)
#define SBL_PML_CFG_TX_PCS_LANE_1_SOURCE_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,lane_1_source)
#define SBL_PML_CFG_TX_PCS_LANE_2_SOURCE_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,lane_2_source)
#define SBL_PML_CFG_TX_PCS_LANE_3_SOURCE_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,lane_3_source)
#define SBL_PML_CFG_TX_PCS_ALLOW_AUTO_DEGRADE_UPDATE(a,b)                      C_UPDATE(a,b,c1_hni_pml_cfg_tx_pcs,allow_auto_degrade)
#define SBL_PML_DBG_PCS_ENABLE_RAPID_ALIGNMENT_UPDATE(a,b)                     C_UPDATE(a,b,c1_hni_pml_dbg_pcs,enable_rapid_alignment)
#define SBL_PML_DBG_PCS_FORCE_TX_DATA_RF_UPDATE(a,b)                           C_UPDATE(a,b,c1_hni_pml_dbg_pcs,force_tx_data_rf)
#define SBL_PML_DBG_PCS_FORCE_TX_DATA_UPDATE(a,b)                              C_UPDATE(a,b,c1_hni_pml_dbg_pcs,force_tx_data)

/* DFLT defines */
#define SBL_PML_CFG_LLR_MAC_IF_CREDITS C1_HNI_CFG_LLR_MAC__IF_CREDITS
#define SBL_PML_CFG_TX_MAC_PCS_CREDITS C1_HNI_CFG_TX_MAC__PCS_CREDITS
#define SBL_PML_CFG_TX_PCS_CDC_READY_LEVEL C1_HNI_CFG_TX_PCS__CDC_READY_LEVEL
#define SBL_PML_CFG_TX_PCS_GEARBOX_CREDITS C1_HNI_CFG_TX_PCS__GEARBOX_CREDITS
#define SBL_PML_CFG_TX_PCS_CDC_READY_LEVEL_DFLT C1_HNI_PML_CFG_TX_PCS__CDC_READY_LEVEL
#define SBL_PML_CFG_TX_PCS_GEARBOX_CREDITS_DFLT C1_HNI_PML_CFG_TX_PCS__GEARBOX_CREDITS
#define SBL_PML_CFG_TX_MAC_PCS_CREDITS_DFLT C1_HNI_PML_CFG_TX_MAC__PCS_CREDITS
#define SBL_PML_CFG_LLR_MAC_IF_CREDITS_DFLT C1_HNI_PML_CFG_LLR__MAC_IF_CREDITS
#define SBL_PML_CFG_LLR_CAPACITY_MAX_SEQ_DFLT C1_HNI_PML_CFG_LLR_CAPACITY__MAX_SEQ

/* TODO - cassini_csr_defaults.h skips defaults when they are 0? */
#define C1_HNI_PML_CFG_LLR__PREAMBLE_SEQ_CHECK 0x0
#define C1_HNI_PML_CFG_LLR__ACK_NACK_ERR_CHECK 0x0
#define C1_HNI_PML_CFG_LLR__LLR_MODE 0x0
#define C1_HNI_PML_CFG_LLR__ENABLE_LOOP_TIMING 0x0
#define C1_HNI_PML_CFG_LLR_SM__ALLOW_RE_INIT 0x0
#define C_HNI_PML_CFG_LLR_CF_SMAC__CTL_FRAME_SMAC 0x0
#define C1_HNI_PML_CFG_TX_MAC__IEEE_IFG_ADJUSTMENT 0ull
#define C1_HNI_PML_CFG_TX_MAC__SHORT_PREAMBLE 0ull
#define C1_HNI_PML_CFG_TX_MAC__MAC_OPERATIONAL 0ull
#define C1_HNI_PML_CFG_RX_MAC__SHORT_PREAMBLE 0x0
#define C1_HNI_PML_CFG_RX_MAC__MAC_OPERATIONAL 0x0
#define C1_HNI_PML_CFG_RX_MAC__FILTER_ILLEGAL_SIZE 0x0
#define C1_HNI_PML_DBG_PCS__ENABLE_RAPID_ALIGNMENT 0x0
#define C1_HNI_PML_DBG_PCS__ENABLE_IDLE_CHECK 0x0
#define C1_HNI_PML_DBG_PCS__FORCE_TX_DATA 0x0
#define C1_HNI_PML_DBG_PCS__FORCE_TX_DATA_RF 0x0
#define C1_HNI_PML_DBG_PCS__PPM_TEST_MODE 0x0
#define C1_HNI_PML_CFG_RX_PCS__ACTIVE_LANES 0x0
#define C1_HNI_PML_CFG_RX_PCS__ENABLE_LOCK 0x0
#define C1_HNI_PML_CFG_RX_PCS__RESTART_LOCK_ON_BAD_AMS 0x0
#define C1_HNI_PML_CFG_RX_PCS__RESTART_LOCK_ON_BAD_CWS 0x0
#define C1_HNI_PML_CFG_RX_PCS__ENABLE_CTL_OS 0x0
#define C1_HNI_PML_CFG_RX_PCS__LP_HEALTH_BAD_SENSITIVITY 0x0
#define C1_HNI_PML_CFG_RX_PCS__ALLOW_AUTO_DEGRADE 0x0
#define C1_HNI_PML_CFG_RX_PCS__ENABLE_RX_SM 0x0
#define C1_HNI_PML_CFG_TX_PCS__LANE_0_SOURCE 0x0
#define C1_HNI_PML_CFG_TX_PCS__ENABLE_CTL_OS 0x0
#define C1_HNI_PML_CFG_TX_PCS__ALLOW_AUTO_DEGRADE 0x0
#define C1_HNI_PML_CFG_PCS_AMS__USE_PROGRAMMABLE_AM_SPACING 0x0
#define C1_HNI_PML_CFG_PCS_AMS__USE_PROGRAMMABLE_AMS 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE__NEXT_PAGE 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE__BASE_PAGE 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG__NEXT_PAGE_LOADED 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG__RESTART 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG__TX_LANE 0x0
#define C1_HNI_PML_CFG_PCS_AUTONEG__RX_LANE 0x0
#define C1_HNI_PML_CFG_PCS__PCS_ENABLE 0x0
#define C1_HNI_PML_CFG_PCS__ENABLE_AUTO_NEG 0x0
#define C1_HNI_PML_CFG_PCS__PCS_MODE 0x0
#define C1_HNI_PML_CFG_PCS__ENABLE_AUTO_LANE_DEGRADE 0x0
#define C1_HNI_PML_CFG_PCS__LL_FEC 0x0
#define C1_HNI_PML_CFG_PCS__TIMESTAMP_SHIFT 0x0
#define SBL_PML_CFG_LLR_CAPACITY_DFLT					\
    (SBL_PML_CFG_LLR_CAPACITY_MAX_DATA_SET(C1_HNI_PML_CFG_LLR_CAPACITY__MAX_DATA)| \
     SBL_PML_CFG_LLR_CAPACITY_MAX_SEQ_SET(C1_HNI_PML_CFG_LLR_CAPACITY__MAX_SEQ))
#define SBL_PML_CFG_LLR_CF_ETYPE_DFLT                                  \
    (SBL_PML_CFG_LLR_CF_ETYPE_CTL_FRAME_ETHERTYPE_SET(C1_HNI_PML_CFG_LLR_CF_ETYPE__CTL_FRAME_ETHERTYPE))
#define SBL_PML_CFG_LLR_CF_SMAC_DFLT                                   \
    (SBL_PML_CFG_LLR_CF_SMAC_CTL_FRAME_SMAC_SET(C_HNI_PML_CFG_LLR_CF_SMAC__CTL_FRAME_SMAC))
#define SBL_PML_CFG_LLR_DFLT						\
    (SBL_PML_CFG_LLR_PREAMBLE_SEQ_CHECK_SET(C1_HNI_PML_CFG_LLR__PREAMBLE_SEQ_CHECK)| \
     SBL_PML_CFG_LLR_ACK_NACK_ERR_CHECK_SET(C1_HNI_PML_CFG_LLR__ACK_NACK_ERR_CHECK)| \
     SBL_PML_CFG_LLR_FILTER_LOSSLESS_WHEN_OFF_SET(C1_HNI_PML_CFG_LLR__FILTER_LOSSLESS_WHEN_OFF)| \
     SBL_PML_CFG_LLR_FILTER_CTL_FRAMES_SET(C1_HNI_PML_CFG_LLR__FILTER_CTL_FRAMES)| \
     SBL_PML_CFG_LLR_SIZE_SET(C1_HNI_PML_CFG_LLR__SIZE)| \
     SBL_PML_CFG_LLR_ENABLE_LOOP_TIMING_SET(C1_HNI_PML_CFG_LLR__ENABLE_LOOP_TIMING)| \
     SBL_PML_CFG_LLR_LINK_DOWN_BEHAVIOR_SET(C1_HNI_PML_CFG_LLR__LINK_DOWN_BEHAVIOR)| \
     SBL_PML_CFG_LLR_MAC_IF_CREDITS_SET(C1_HNI_PML_CFG_LLR__MAC_IF_CREDITS)| \
     SBL_PML_CFG_LLR_LLR_MODE_SET(C1_HNI_PML_CFG_LLR__LLR_MODE))
#define SBL_PML_CFG_LLR_SM_DFLT						\
    (SBL_PML_CFG_LLR_SM_RETRY_THRESHOLD_SET(C1_HNI_PML_CFG_LLR_SM__RETRY_THRESHOLD)| \
     SBL_PML_CFG_LLR_SM_ALLOW_RE_INIT_SET(C1_HNI_PML_CFG_LLR_SM__ALLOW_RE_INIT)| \
     SBL_PML_CFG_LLR_SM_REPLAY_CT_MAX_SET(C1_HNI_PML_CFG_LLR_SM__REPLAY_CT_MAX)| \
     SBL_PML_CFG_LLR_SM_REPLAY_TIMER_MAX_SET(C1_HNI_PML_CFG_LLR_SM__REPLAY_TIMER_MAX))
#define SBL_PML_CFG_LLR_TIMEOUTS_DFLT                                  \
    (SBL_PML_CFG_LLR_TIMEOUTS_DATA_AGE_TIMER_MAX_SET(C1_HNI_PML_CFG_LLR_TIMEOUTS__DATA_AGE_TIMER_MAX)| \
     SBL_PML_CFG_LLR_TIMEOUTS_PCS_LINK_DN_TIMER_MAX_SET(C1_HNI_PML_CFG_LLR_TIMEOUTS__PCS_LINK_DN_TIMER_MAX))
#define SBL_PML_CFG_PCS_AMS_DFLT                                       \
    (SBL_PML_CFG_PCS_AMS_USE_PROGRAMMABLE_AMS_SET(C1_HNI_PML_CFG_PCS_AMS__USE_PROGRAMMABLE_AMS)| \
     SBL_PML_CFG_PCS_AMS_USE_PROGRAMMABLE_AM_SPACING_SET(C1_HNI_PML_CFG_PCS_AMS__USE_PROGRAMMABLE_AM_SPACING)| \
     SBL_PML_CFG_PCS_AMS_RAM_SPACING_SET(C1_HNI_PML_CFG_PCS_AMS__RAM_SPACING)| \
     SBL_PML_CFG_PCS_AMS_AM_SPACING_SET(C1_HNI_PML_CFG_PCS_AMS__AM_SPACING))
#define SBL_PML_CFG_PCS_AUTONEG_BASE_PAGE_DFLT                         \
    (SBL_PML_CFG_PCS_AUTONEG_BASE_PAGE_BASE_PAGE_SET(C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE__BASE_PAGE))
#define SBL_PML_CFG_PCS_AUTONEG_DFLT                                   \
    (SBL_PML_CFG_PCS_AUTONEG_RX_LANE_SET(C1_HNI_PML_CFG_PCS_AUTONEG__RX_LANE)| \
     SBL_PML_CFG_PCS_AUTONEG_TX_LANE_SET(C1_HNI_PML_CFG_PCS_AUTONEG__TX_LANE)| \
     SBL_PML_CFG_PCS_AUTONEG_RESET_SET(C1_HNI_PML_CFG_PCS_AUTONEG__RESET)| \
     SBL_PML_CFG_PCS_AUTONEG_RESTART_SET(C1_HNI_PML_CFG_PCS_AUTONEG__RESTART)| \
     SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_LOADED_SET(C1_HNI_PML_CFG_PCS_AUTONEG__NEXT_PAGE_LOADED))
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_DFLT                         \
    (SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_NEXT_PAGE_SET(C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE__NEXT_PAGE))
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_DFLT                            \
    (SBL_PML_CFG_PCS_AUTONEG_TIMERS_LINK_FAIL_INHIBIT_TIMER_MAX_SET(C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS__LINK_FAIL_INHIBIT_TIMER_MAX)| \
     SBL_PML_CFG_PCS_AUTONEG_TIMERS_BREAK_LINK_TIMER_MAX_SET(C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS__BREAK_LINK_TIMER_MAX))
#define SBL_PML_CFG_PCS_DFLT                                           \
    (SBL_PML_CFG_PCS_TIMESTAMP_SHIFT_SET(C1_HNI_PML_CFG_PCS__TIMESTAMP_SHIFT)| \
     SBL_PML_CFG_PCS_LL_FEC_SET(C1_HNI_PML_CFG_PCS__LL_FEC)| \
     SBL_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_SET(C1_HNI_PML_CFG_PCS__ENABLE_AUTO_LANE_DEGRADE)| \
     SBL_PML_CFG_PCS_PCS_MODE_SET(C1_HNI_PML_CFG_PCS__PCS_MODE)| \
     SBL_PML_CFG_PCS_ENABLE_AUTO_NEG_SET(C1_HNI_PML_CFG_PCS__ENABLE_AUTO_NEG)| \
     SBL_PML_CFG_PCS_PCS_ENABLE_SET(C1_HNI_PML_CFG_PCS__PCS_ENABLE))
#define SBL_PML_CFG_RX_MAC_DFLT						\
    (SBL_PML_CFG_RX_MAC_FILTER_ILLEGAL_SIZE_SET(C1_HNI_PML_CFG_RX_MAC__FILTER_ILLEGAL_SIZE)| \
     SBL_PML_CFG_RX_MAC_SHORT_PREAMBLE_SET(C1_HNI_PML_CFG_RX_MAC__SHORT_PREAMBLE)| \
     SBL_PML_CFG_RX_MAC_MAC_OPERATIONAL_SET(C1_HNI_PML_CFG_RX_MAC__MAC_OPERATIONAL))
#define SBL_PML_CFG_RX_PCS_DFLT                                        \
    (SBL_PML_CFG_RX_PCS_ENABLE_RX_SM_SET(C1_HNI_PML_CFG_RX_PCS__ENABLE_RX_SM)| \
     SBL_PML_CFG_RX_PCS_ALLOW_AUTO_DEGRADE_SET(C1_HNI_PML_CFG_RX_PCS__ALLOW_AUTO_DEGRADE)| \
     SBL_PML_CFG_RX_PCS_LP_HEALTH_BAD_SENSITIVITY_SET(C1_HNI_PML_CFG_RX_PCS__LP_HEALTH_BAD_SENSITIVITY)| \
     SBL_PML_CFG_RX_PCS_HEALTH_BAD_SENSITIVITY_SET(C1_HNI_PML_CFG_RX_PCS__HEALTH_BAD_SENSITIVITY)| \
     SBL_PML_CFG_RX_PCS_ENABLE_CTL_OS_SET(C1_HNI_PML_CFG_RX_PCS__ENABLE_CTL_OS)| \
     SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_CWS_SET(C1_HNI_PML_CFG_RX_PCS__RESTART_LOCK_ON_BAD_CWS)| \
     SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_AMS_SET(C1_HNI_PML_CFG_RX_PCS__RESTART_LOCK_ON_BAD_AMS)| \
     SBL_PML_CFG_RX_PCS_ENABLE_LOCK_SET(C1_HNI_PML_CFG_RX_PCS__ENABLE_LOCK)| \
     SBL_PML_CFG_RX_PCS_RS_MODE_SET(C1_HNI_PML_CFG_RX_PCS__RS_MODE)| \
     SBL_PML_CFG_RX_PCS_ACTIVE_LANES_SET(C1_HNI_PML_CFG_RX_PCS__ACTIVE_LANES))
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_DFLT                         \
    (SBL_PML_CFG_SERDES_CORE_INTERRUPT_CAPTURE_INTERRUPT_DATA_DELAY_SET(C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT__CAPTURE_INTERRUPT_DATA_DELAY)| \
     SBL_PML_CFG_SERDES_CORE_INTERRUPT_CLEAR_INTERRUPT_DELAY_SET(C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT__CLEAR_INTERRUPT_DELAY)| \
     SBL_PML_CFG_SERDES_CORE_INTERRUPT_SET_INTERRUPT_DELAY_SET(C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT__SET_INTERRUPT_DELAY))
#define SBL_PML_CFG_TX_MAC_DFLT						\
    (SBL_PML_CFG_TX_MAC_PORTALS_PROTOCOL_SET(C1_HNI_PML_CFG_TX_MAC__PORTALS_PROTOCOL)| \
     SBL_PML_CFG_TX_MAC_ANY_FRAME_MAX_SOF_1_SET(C1_HNI_PML_CFG_TX_MAC__ANY_FRAME_MAX_SOF_1)| \
     SBL_PML_CFG_TX_MAC_ANY_FRAME_SOF_WINDOW_1_SET(C1_HNI_PML_CFG_TX_MAC__ANY_FRAME_SOF_WINDOW_1)| \
     SBL_PML_CFG_TX_MAC_ANY_FRAME_MAX_SOF_0_SET(C1_HNI_PML_CFG_TX_MAC__ANY_FRAME_MAX_SOF_0)| \
     SBL_PML_CFG_TX_MAC_ANY_FRAME_SOF_WINDOW_0_SET(C1_HNI_PML_CFG_TX_MAC__ANY_FRAME_SOF_WINDOW_0)| \
     SBL_PML_CFG_TX_MAC_NON_PORTALS_MAX_SOF_SET(C1_HNI_PML_CFG_TX_MAC__NON_PORTALS_MAX_SOF)| \
     SBL_PML_CFG_TX_MAC_NON_PORTALS_SOF_WINDOW_SET(C1_HNI_PML_CFG_TX_MAC__NON_PORTALS_SOF_WINDOW)| \
     SBL_PML_CFG_TX_MAC_IEEE_IFG_ADJUSTMENT_SET(C1_HNI_PML_CFG_TX_MAC__IEEE_IFG_ADJUSTMENT)| \
     SBL_PML_CFG_TX_MAC_VS_VERSION_SET(C1_HNI_PML_CFG_TX_MAC__VS_VERSION)| \
     SBL_PML_CFG_TX_MAC_IFG_MODE_SET(C1_HNI_PML_CFG_TX_MAC__IFG_MODE)| \
     SBL_PML_CFG_TX_MAC_PCS_CREDITS_SET(C1_HNI_PML_CFG_TX_MAC__PCS_CREDITS)| \
     SBL_PML_CFG_TX_MAC_SHORT_PREAMBLE_SET(C1_HNI_PML_CFG_TX_MAC__SHORT_PREAMBLE)| \
     SBL_PML_CFG_TX_MAC_MAC_OPERATIONAL_SET(C1_HNI_PML_CFG_TX_MAC__MAC_OPERATIONAL))
#define SBL_PML_CFG_TX_PCS_DFLT                                        \
    (SBL_PML_CFG_TX_PCS_CDC_READY_LEVEL_SET(C1_HNI_PML_CFG_TX_PCS__CDC_READY_LEVEL)| \
     SBL_PML_CFG_TX_PCS_GEARBOX_CREDITS_SET(C1_HNI_PML_CFG_TX_PCS__GEARBOX_CREDITS)| \
     SBL_PML_CFG_TX_PCS_ALLOW_AUTO_DEGRADE_SET(C1_HNI_PML_CFG_TX_PCS__ALLOW_AUTO_DEGRADE)| \
     SBL_PML_CFG_TX_PCS_ENABLE_CTL_OS_SET(C1_HNI_PML_CFG_TX_PCS__ENABLE_CTL_OS)| \
     SBL_PML_CFG_TX_PCS_LANE_3_SOURCE_SET(C1_HNI_PML_CFG_TX_PCS__LANE_3_SOURCE)| \
     SBL_PML_CFG_TX_PCS_LANE_2_SOURCE_SET(C1_HNI_PML_CFG_TX_PCS__LANE_2_SOURCE)| \
     SBL_PML_CFG_TX_PCS_LANE_1_SOURCE_SET(C1_HNI_PML_CFG_TX_PCS__LANE_1_SOURCE)| \
     SBL_PML_CFG_TX_PCS_LANE_0_SOURCE_SET(C1_HNI_PML_CFG_TX_PCS__LANE_0_SOURCE))
#define SBL_PML_DBG_PCS_DFLT                                           \
    (SBL_PML_DBG_PCS_PPM_TEST_MODE_SET(C1_HNI_PML_DBG_PCS__PPM_TEST_MODE)| \
     SBL_PML_DBG_PCS_FORCE_TX_DATA_RF_SET(C1_HNI_PML_DBG_PCS__FORCE_TX_DATA_RF)| \
     SBL_PML_DBG_PCS_FORCE_TX_DATA_SET(C1_HNI_PML_DBG_PCS__FORCE_TX_DATA)| \
     SBL_PML_DBG_PCS_ENABLE_IDLE_CHECK_SET(C1_HNI_PML_DBG_PCS__ENABLE_IDLE_CHECK)| \
     SBL_PML_DBG_PCS_FORCE_BAD_BIP_SET(C1_HNI_PML_DBG_PCS__FORCE_BAD_BIP)| \
     SBL_PML_DBG_PCS_ENABLE_RAPID_ALIGNMENT_SET(C1_HNI_PML_DBG_PCS__ENABLE_RAPID_ALIGNMENT)| \
     SBL_PML_DBG_PCS_RX_DESCRAMBLER_EN_SET(C1_HNI_PML_DBG_PCS__RX_DESCRAMBLER_EN)| \
     SBL_PML_DBG_PCS_TX_SCRAMBLER_EN_SET(C1_HNI_PML_DBG_PCS__TX_SCRAMBLER_EN))

/* SET defines */
#define SBL_PML_CFG_LLR_CAPACITY_MAX_DATA_SET(a)                              C_SET(a, c1_hni_pml_cfg_llr_capacity, max_data)
#define SBL_PML_CFG_LLR_CAPACITY_MAX_SEQ_SET(a)                               C_SET(a, c1_hni_pml_cfg_llr_capacity, max_seq)
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_LOADED_SET(a)                       C_SET(a, c_hni_pml_cfg_pcs_autoneg, next_page_loaded)
#define SBL_PML_CFG_PCS_AUTONEG_RESET_SET(a)                                  C_SET(a, c_hni_pml_cfg_pcs_autoneg, reset)
#define SBL_PML_CFG_PCS_AUTONEG_RESTART_SET(a)                                C_SET(a, c_hni_pml_cfg_pcs_autoneg, restart)
#define SBL_PML_CFG_PCS_AUTONEG_RX_LANE_SET(a)                                C_SET(a, c_hni_pml_cfg_pcs_autoneg, rx_lane)
#define SBL_PML_CFG_PCS_AUTONEG_TX_LANE_SET(a)                                C_SET(a, c_hni_pml_cfg_pcs_autoneg, tx_lane)
#define SBL_PML_CFG_RX_PCS_AMS_UM_MATCH_MASK_SET(a)                           C_SET(a, c_hni_pml_cfg_rx_pcs_ams, um_match_mask)
#define SBL_PML_CFG_RX_PCS_AMS_CM_MATCH_MASK_SET(a)                           C_SET(a, c_hni_pml_cfg_rx_pcs_ams, cm_match_mask)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_CAPTURE_INTERRUPT_DATA_DELAY_SET(a) C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, capture_interrupt_data_delay)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_CLEAR_INTERRUPT_DELAY_SET(a)        C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, clear_interrupt_delay)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_SET_INTERRUPT_DELAY_SET(a)          C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, set_interrupt_delay)
#define SBL_PML_ERR_FLG_AUTONEG_COMPLETE_SET(a)                               C_SET(a, c1_hni_pml_err_flg, autoneg_complete)
#define SBL_PML_ERR_FLG_AUTONEG_PAGE_RECEIVED_SET(a)                          C_SET(a, c1_hni_pml_err_flg, autoneg_page_received)
#define SBL_PML_ERR_FLG_LLR_REPLAY_AT_MAX_SET(a)                              C_SET(a, c1_hni_pml_err_flg, llr_replay_at_max)
#define SBL_PML_ERR_FLG_MAC_RX_DP_ERR_SET(a)                                  C_SET(a, c1_hni_pml_err_flg, mac_rx_dp_err)
#define SBL_PML_ERR_FLG_MAC_TX_DP_ERR_SET(a)                                  C_SET(a, c1_hni_pml_err_flg, mac_tx_dp_err)
#define SBL_PML_ERR_FLG_PCS_HI_SER_SET(a)                                     C_SET(a, c1_hni_pml_err_flg, pcs_hi_ser)
#define SBL_PML_ERR_FLG_PCS_LINK_DOWN_SET(a)                                  C_SET(a, c1_hni_pml_err_flg, pcs_link_down)
#define SBL_PML_ERR_FLG_PCS_TX_DEGRADE_SET(a)                                 C_SET(a, c1_hni_pml_err_flg, pcs_tx_degrade)
#define SBL_PML_ERR_FLG_PCS_RX_DEGRADE_SET(a)                                 C_SET(a, c1_hni_pml_err_flg, pcs_rx_degrade)
#define SBL_PML_ERR_FLG_PCS_TX_DEGRADE_FAILURE_SET(a)                         C_SET(a, c1_hni_pml_err_flg, pcs_tx_degrade_failure)
#define SBL_PML_ERR_FLG_PCS_RX_DEGRADE_FAILURE_SET(a)                         C_SET(a, c1_hni_pml_err_flg, pcs_rx_degrade_failure)
#define SBL_PML_SERDES_CORE_INTERRUPT_SERDES_SEL_SET(a)                       C_SET(a, c1_hni_pml_serdes_core_interrupt, serdes_sel)
#define SBL_PML_SERDES_CORE_INTERRUPT_DO_CORE_INTERRUPT_SET(a)                C_SET(a, c1_hni_pml_serdes_core_interrupt, do_core_interrupt)
#define SBL_PML_SERDES_CORE_INTERRUPT_CORE_INTERRUPT_CODE_SET(a)              C_SET(a, c1_hni_pml_serdes_core_interrupt, core_interrupt_code)
#define SBL_PML_SERDES_CORE_INTERRUPT_CORE_INTERRUPT_DATA_SET(a)              C_SET(a, c1_hni_pml_serdes_core_interrupt, core_interrupt_data)

#define SBL_PML_CFG_LLR_CAPACITY_SET(MAX_DATA,MAX_SEQ)  \
    (SBL_PML_CFG_LLR_CAPACITY_MAX_DATA_SET(MAX_DATA)| \
     SBL_PML_CFG_LLR_CAPACITY_MAX_SEQ_SET(MAX_SEQ))
#define SBL_PML_CFG_RX_PCS_AMS_SET(UM_MATCH_MASK,CM_MATCH_MASK)                         \
    (SBL_PML_CFG_RX_PCS_AMS_UM_MATCH_MASK_SET(UM_MATCH_MASK)| \
     SBL_PML_CFG_RX_PCS_AMS_CM_MATCH_MASK_SET(CM_MATCH_MASK))
#define SBL_PML_SERDES_CORE_INTERRUPT_SET(SERDES_SEL,DO_CORE_INTERRUPT,CORE_INTERRUPT_CODE,CORE_INTERRUPT_DATA) \
    (SBL_PML_SERDES_CORE_INTERRUPT_SERDES_SEL_SET(SERDES_SEL)| \
     SBL_PML_SERDES_CORE_INTERRUPT_DO_CORE_INTERRUPT_SET(DO_CORE_INTERRUPT)| \
     SBL_PML_SERDES_CORE_INTERRUPT_CORE_INTERRUPT_CODE_SET(CORE_INTERRUPT_CODE)| \
     SBL_PML_SERDES_CORE_INTERRUPT_CORE_INTERRUPT_DATA_SET(CORE_INTERRUPT_DATA))

#define SBL_PML_CFG_LLR_CF_ETYPE_CTL_FRAME_ETHERTYPE_SET(a)                   C_SET(a, c_hni_pml_cfg_llr_cf_etype, ctl_frame_ethertype)
#define SBL_PML_CFG_LLR_CF_SMAC_CTL_FRAME_SMAC_SET(a)                         C_SET(a, c_hni_pml_cfg_llr_cf_smac, ctl_frame_smac)
#define SBL_PML_CFG_LLR_PREAMBLE_SEQ_CHECK_SET(a)                             C_SET(a, c1_hni_pml_cfg_llr, preamble_seq_check)
#define SBL_PML_CFG_LLR_ACK_NACK_ERR_CHECK_SET(a)                             C_SET(a, c1_hni_pml_cfg_llr, ack_nack_err_check)
#define SBL_PML_CFG_LLR_FILTER_LOSSLESS_WHEN_OFF_SET(a)                       C_SET(a, c1_hni_pml_cfg_llr, filter_lossless_when_off)
#define SBL_PML_CFG_LLR_FILTER_CTL_FRAMES_SET(a)                              C_SET(a, c1_hni_pml_cfg_llr, filter_ctl_frames)
#define SBL_PML_CFG_LLR_SIZE_SET(a)                                           C_SET(a, c1_hni_pml_cfg_llr, size)
#define SBL_PML_CFG_LLR_ENABLE_LOOP_TIMING_SET(a)                             C_SET(a, c1_hni_pml_cfg_llr, enable_loop_timing)
#define SBL_PML_CFG_LLR_LINK_DOWN_BEHAVIOR_SET(a)                             C_SET(a, c1_hni_pml_cfg_llr, link_down_behavior)
#define SBL_PML_CFG_LLR_MAC_IF_CREDITS_SET(a)                                 C_SET(a, c1_hni_pml_cfg_llr, mac_if_credits)
#define SBL_PML_CFG_LLR_LLR_MODE_SET(a)                                       C_SET(a, c1_hni_pml_cfg_llr, llr_mode)
#define SBL_PML_CFG_LLR_SM_RETRY_THRESHOLD_SET(a)                             C_SET(a, c_hni_pml_cfg_llr_sm, retry_threshold)
#define SBL_PML_CFG_LLR_SM_ALLOW_RE_INIT_SET(a)                               C_SET(a, c_hni_pml_cfg_llr_sm, allow_re_init)
#define SBL_PML_CFG_LLR_SM_REPLAY_CT_MAX_SET(a)                               C_SET(a, c_hni_pml_cfg_llr_sm, replay_ct_max)
#define SBL_PML_CFG_LLR_SM_REPLAY_TIMER_MAX_SET(a)                            C_SET(a, c_hni_pml_cfg_llr_sm, replay_timer_max)
#define SBL_PML_CFG_LLR_TIMEOUTS_DATA_AGE_TIMER_MAX_SET(a)                    C_SET(a, c_hni_pml_cfg_llr_timeouts, data_age_timer_max)
#define SBL_PML_CFG_LLR_TIMEOUTS_PCS_LINK_DN_TIMER_MAX_SET(a)                 C_SET(a, c_hni_pml_cfg_llr_timeouts, pcs_link_dn_timer_max)
#define SBL_PML_CFG_PCS_AMS_USE_PROGRAMMABLE_AMS_SET(a)                       C_SET(a, c_hni_pml_cfg_pcs_ams, use_programmable_ams)
#define SBL_PML_CFG_PCS_AMS_USE_PROGRAMMABLE_AM_SPACING_SET(a)                C_SET(a, c_hni_pml_cfg_pcs_ams, use_programmable_am_spacing)
#define SBL_PML_CFG_PCS_AMS_RAM_SPACING_SET(a)                                C_SET(a, c_hni_pml_cfg_pcs_ams, ram_spacing)
#define SBL_PML_CFG_PCS_AMS_AM_SPACING_SET(a)                                 C_SET(a, c_hni_pml_cfg_pcs_ams, am_spacing)
#define SBL_PML_CFG_PCS_AUTONEG_BASE_PAGE_BASE_PAGE_SET(a)                    C_SET(a, c_hni_pml_cfg_pcs_autoneg_base_page, base_page)
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_NEXT_PAGE_SET(a)                    C_SET(a, c_hni_pml_cfg_pcs_autoneg_next_page, next_page)
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_LINK_FAIL_INHIBIT_TIMER_MAX_SET(a)     C_SET(a, c1_hni_pml_cfg_pcs_autoneg_timers, link_fail_inhibit_timer_max)
#define SBL_PML_CFG_PCS_AUTONEG_TIMERS_BREAK_LINK_TIMER_MAX_SET(a)            C_SET(a, c1_hni_pml_cfg_pcs_autoneg_timers, break_link_timer_max)
#define SBL_PML_CFG_PCS_TIMESTAMP_SHIFT_SET(a)                                C_SET(a, c1_hni_pml_cfg_pcs, timestamp_shift)
#define SBL_PML_CFG_PCS_LL_FEC_SET(a)                                         C_SET(a, c1_hni_pml_cfg_pcs, ll_fec)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_SET(a)                       C_SET(a, c1_hni_pml_cfg_pcs, enable_auto_lane_degrade)
#define SBL_PML_CFG_PCS_PCS_MODE_SET(a)                                       C_SET(a, c1_hni_pml_cfg_pcs, pcs_mode)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_NEG_SET(a)                                C_SET(a, c1_hni_pml_cfg_pcs, enable_auto_neg)
#define SBL_PML_CFG_PCS_PCS_ENABLE_SET(a)                                     C_SET(a, c1_hni_pml_cfg_pcs, pcs_enable)
#define SBL_PML_CFG_RX_MAC_FILTER_ILLEGAL_SIZE_SET(a)                         C_SET(a, c1_hni_pml_cfg_rx_mac, filter_illegal_size)
#define SBL_PML_CFG_RX_MAC_SHORT_PREAMBLE_SET(a)                              C_SET(a, c1_hni_pml_cfg_rx_mac, short_preamble)
#define SBL_PML_CFG_RX_MAC_MAC_OPERATIONAL_SET(a)                             C_SET(a, c1_hni_pml_cfg_rx_mac, mac_operational)
#define SBL_PML_CFG_RX_PCS_ENABLE_RX_SM_SET(a)                                C_SET(a, c1_hni_pml_cfg_rx_pcs, enable_rx_sm)
#define SBL_PML_CFG_RX_PCS_ALLOW_AUTO_DEGRADE_SET(a)                          C_SET(a, c1_hni_pml_cfg_rx_pcs, allow_auto_degrade)
#define SBL_PML_CFG_RX_PCS_LP_HEALTH_BAD_SENSITIVITY_SET(a)                   C_SET(a, c1_hni_pml_cfg_rx_pcs, lp_health_bad_sensitivity)
#define SBL_PML_CFG_RX_PCS_HEALTH_BAD_SENSITIVITY_SET(a)                      C_SET(a, c1_hni_pml_cfg_rx_pcs, health_bad_sensitivity)
#define SBL_PML_CFG_RX_PCS_ENABLE_CTL_OS_SET(a)                               C_SET(a, c1_hni_pml_cfg_rx_pcs, enable_ctl_os)
#define SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_CWS_SET(a)                     C_SET(a, c1_hni_pml_cfg_rx_pcs, restart_lock_on_bad_cws)
#define SBL_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_AMS_SET(a)                     C_SET(a, c1_hni_pml_cfg_rx_pcs, restart_lock_on_bad_ams)
#define SBL_PML_CFG_RX_PCS_ENABLE_LOCK_SET(a)                                 C_SET(a, c1_hni_pml_cfg_rx_pcs, enable_lock)
#define SBL_PML_CFG_RX_PCS_RS_MODE_SET(a)                                     C_SET(a, c1_hni_pml_cfg_rx_pcs, rs_mode)
#define SBL_PML_CFG_RX_PCS_ACTIVE_LANES_SET(a)                                C_SET(a, c1_hni_pml_cfg_rx_pcs, active_lanes)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_CAPTURE_INTERRUPT_DATA_DELAY_SET(a) C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, capture_interrupt_data_delay)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_CLEAR_INTERRUPT_DELAY_SET(a)        C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, clear_interrupt_delay)
#define SBL_PML_CFG_SERDES_CORE_INTERRUPT_SET_INTERRUPT_DELAY_SET(a)          C_SET(a, c1_hni_pml_cfg_serdes_core_interrupt, set_interrupt_delay)
#define SBL_PML_CFG_TX_MAC_PORTALS_PROTOCOL_SET(a)                            C_SET(a, c1_hni_pml_cfg_tx_mac, portals_protocol)
#define SBL_PML_CFG_TX_MAC_ANY_FRAME_MAX_SOF_1_SET(a)                         C_SET(a, c1_hni_pml_cfg_tx_mac, any_frame_max_sof_1)
#define SBL_PML_CFG_TX_MAC_ANY_FRAME_SOF_WINDOW_1_SET(a)                      C_SET(a, c1_hni_pml_cfg_tx_mac, any_frame_sof_window_1)
#define SBL_PML_CFG_TX_MAC_ANY_FRAME_MAX_SOF_0_SET(a)                         C_SET(a, c1_hni_pml_cfg_tx_mac, any_frame_max_sof_0)
#define SBL_PML_CFG_TX_MAC_ANY_FRAME_SOF_WINDOW_0_SET(a)                      C_SET(a, c1_hni_pml_cfg_tx_mac, any_frame_sof_window_0)
#define SBL_PML_CFG_TX_MAC_NON_PORTALS_MAX_SOF_SET(a)                         C_SET(a, c1_hni_pml_cfg_tx_mac, non_portals_max_sof)
#define SBL_PML_CFG_TX_MAC_NON_PORTALS_SOF_WINDOW_SET(a)                      C_SET(a, c1_hni_pml_cfg_tx_mac, non_portals_sof_window)
#define SBL_PML_CFG_TX_MAC_VS_VERSION_SET(a)                                  C_SET(a, c1_hni_pml_cfg_tx_mac, vs_version)
#define SBL_PML_CFG_TX_MAC_IEEE_IFG_ADJUSTMENT_SET(a)                         C_SET(a, c1_hni_pml_cfg_tx_mac, ieee_ifg_adjustment)
#define SBL_PML_CFG_TX_MAC_IFG_MODE_SET(a)                                    C_SET(a, c1_hni_pml_cfg_tx_mac, ifg_mode)
#define SBL_PML_CFG_TX_MAC_PCS_CREDITS_SET(a)                                 C_SET(a, c1_hni_pml_cfg_tx_mac, pcs_credits)
#define SBL_PML_CFG_TX_MAC_SHORT_PREAMBLE_SET(a)                              C_SET(a, c1_hni_pml_cfg_tx_mac, short_preamble)
#define SBL_PML_CFG_TX_MAC_MAC_OPERATIONAL_SET(a)                             C_SET(a, c1_hni_pml_cfg_tx_mac, mac_operational)
#define SBL_PML_CFG_TX_PCS_CDC_READY_LEVEL_SET(a)                             C_SET(a, c1_hni_pml_cfg_tx_pcs, cdc_ready_level)
#define SBL_PML_CFG_TX_PCS_GEARBOX_CREDITS_SET(a)                             C_SET(a, c1_hni_pml_cfg_tx_pcs, gearbox_credits)
#define SBL_PML_CFG_TX_PCS_ALLOW_AUTO_DEGRADE_SET(a)                          C_SET(a, c1_hni_pml_cfg_tx_pcs, allow_auto_degrade)
#define SBL_PML_CFG_TX_PCS_ENABLE_CTL_OS_SET(a)                               C_SET(a, c1_hni_pml_cfg_tx_pcs, enable_ctl_os)
#define SBL_PML_CFG_TX_PCS_LANE_3_SOURCE_SET(a)                               C_SET(a, c1_hni_pml_cfg_tx_pcs, lane_3_source)
#define SBL_PML_CFG_TX_PCS_LANE_2_SOURCE_SET(a)                               C_SET(a, c1_hni_pml_cfg_tx_pcs, lane_2_source)
#define SBL_PML_CFG_TX_PCS_LANE_1_SOURCE_SET(a)                               C_SET(a, c1_hni_pml_cfg_tx_pcs, lane_1_source)
#define SBL_PML_CFG_TX_PCS_LANE_0_SOURCE_SET(a)                               C_SET(a, c1_hni_pml_cfg_tx_pcs, lane_0_source)
#define SBL_PML_DBG_PCS_PPM_TEST_MODE_SET(a)                                  C_SET(a, c1_hni_pml_dbg_pcs, ppm_test_mode)
#define SBL_PML_DBG_PCS_FORCE_TX_DATA_RF_SET(a)                               C_SET(a, c1_hni_pml_dbg_pcs, force_tx_data_rf)
#define SBL_PML_DBG_PCS_FORCE_TX_DATA_SET(a)                                  C_SET(a, c1_hni_pml_dbg_pcs, force_tx_data)
#define SBL_PML_DBG_PCS_ENABLE_IDLE_CHECK_SET(a)                              C_SET(a, c1_hni_pml_dbg_pcs, enable_idle_check)
#define SBL_PML_DBG_PCS_FORCE_BAD_BIP_SET(a)                                  C_SET(a, c1_hni_pml_dbg_pcs, force_bad_bip)
#define SBL_PML_DBG_PCS_ENABLE_RAPID_ALIGNMENT_SET(a)                         C_SET(a, c1_hni_pml_dbg_pcs, enable_rapid_alignment)
#define SBL_PML_DBG_PCS_RX_DESCRAMBLER_EN_SET(a)                              C_SET(a, c1_hni_pml_dbg_pcs, rx_descrambler_en)
#define SBL_PML_DBG_PCS_TX_SCRAMBLER_EN_SET(a)                                C_SET(a, c1_hni_pml_dbg_pcs, tx_scrambler_en)

/* GET defines */
#define SBL_PML_CFG_LLR_LINK_DOWN_BEHAVIOR_GET(a)                C_GET(a, c1_hni_pml_cfg_llr, link_down_behavior)
#define SBL_PML_CFG_LLR_LLR_MODE_GET(a)                          C_GET(a, c1_hni_pml_cfg_llr, llr_mode)
#define SBL_PML_CFG_LLR_SM_REPLAY_CT_MAX_GET(a)                  C_GET(a, c_hni_pml_cfg_llr_sm, replay_ct_max)
#define SBL_PML_CFG_PCS_AUTONEG_NEXT_PAGE_LOADED_GET(a)          C_GET(a, c1_hni_pml_cfg_pcs_autoneg, next_page_loaded)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_NEG_GET(a)                   C_GET(a, c1_hni_pml_cfg_pcs, enable_auto_neg)
#define SBL_PML_CFG_PCS_PCS_ENABLE_GET(a)                        C_GET(a, c1_hni_pml_cfg_pcs, pcs_enable)
#define SBL_PML_CFG_RX_PCS_ENABLE_LOCK_GET(a)                    C_GET(a, c1_hni_pml_cfg_rx_pcs, enable_lock)
#define SBL_PML_ERR_FLG_AUTONEG_COMPLETE_GET(a)                  C_GET(a, c1_hni_pml_err_flg, autoneg_complete)
#define SBL_PML_ERR_FLG_AUTONEG_PAGE_RECEIVED_GET(a)             C_GET(a, c1_hni_pml_err_flg, autoneg_page_received)
#define SBL_PML_ERR_FLG_LLR_REPLAY_AT_MAX_GET(a)                 C_GET(a, c1_hni_pml_err_flg, llr_replay_at_max)
#define SBL_PML_ERR_FLG_PCS_HI_SER_GET(a)                        C_GET(a, c1_hni_pml_err_flg, pcs_hi_ser)
#define SBL_PML_ERR_FLG_PCS_LINK_DOWN_GET(a)                     C_GET(a, c1_hni_pml_err_flg, pcs_link_down)
#define SBL_PML_ERR_FLG_PCS_TX_DEGRADE_GET(a)                    C_GET(a, c1_hni_pml_err_flg, pcs_tx_degrade)
#define SBL_PML_ERR_FLG_PCS_RX_DEGRADE_GET(a)                    C_GET(a, c1_hni_pml_err_flg, pcs_rx_degrade)
#define SBL_PML_ERR_FLG_PCS_TX_DEGRADE_FAILURE_GET(a)            C_GET(a, c1_hni_pml_err_flg, pcs_tx_degrade_failure)
#define SBL_PML_ERR_FLG_PCS_RX_DEGRADE_FAILURE_GET(a)            C_GET(a, c1_hni_pml_err_flg, pcs_rx_degrade_failure)
#define SBL_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_GET(a)          C_GET(a, c1_hni_pml_cfg_pcs, enable_auto_lane_degrade)
#define SBL_PML_SERDES_CORE_INTERRUPT_CORE_INTERRUPT_DATA_GET(a) C_GET(a, c1_hni_pml_serdes_core_interrupt, core_interrupt_data)
#define SBL_PML_SERDES_CORE_INTERRUPT_DO_CORE_INTERRUPT_GET(a)   C_GET(a, c1_hni_pml_serdes_core_interrupt, do_core_interrupt)
#define SBL_PML_STS_LLR_LLR_STATE_GET(a)                         C_GET(a, c_hni_pml_sts_llr, llr_state)
#define SBL_PML_STS_LLR_LOOP_TIME_LOOP_TIME_GET(a)               C_GET(a, c_hni_pml_sts_llr_loop_time, loop_time)
#define SBL_PML_STS_LLR_MAX_USAGE_BUFF_SPC_STARVED_GET(a)        C_GET(a, c_hni_pml_sts_llr_max_usage, buffer_space_starved)
#define SBL_PML_STS_LLR_MAX_USAGE_SEQ_STARVED_GET(a)             C_GET(a, c_hni_pml_sts_llr_max_usage, seq_starved)
#define SBL_PML_CFG_RX_MAC_MAC_OPERATIONAL_GET(a)                C_GET(a, c1_hni_pml_cfg_rx_mac, mac_operational)
#define SBL_PML_CFG_TX_MAC_IEEE_IFG_ADJUSTMENT_GET(a)            C_GET(a, c1_hni_pml_cfg_tx_mac, ieee_ifg_adjustment)
#define SBL_PML_CFG_TX_MAC_IFG_MODE_GET(a)                       C_GET(a, c1_hni_pml_cfg_tx_mac, ifg_mode)
#define SBL_PML_CFG_TX_MAC_MAC_OPERATIONAL_GET(a)                C_GET(a, c1_hni_pml_cfg_tx_mac, mac_operational)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_BASE_PAGE_GET(a)       C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, base_page)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_LP_BASE_PAGE_GET(a)    C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, lp_base_page)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_LP_ABILITY_GET(a)      C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, lp_ability)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_STATE_GET(a)           C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, state)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_COMPLETE_GET(a)        C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, complete)
#define SBL_PML_STS_PCS_AUTONEG_BASE_PAGE_PAGE_RECEIVED_GET(a)   C_GET(a, c_hni_pml_sts_pcs_autoneg_base_page, page_received)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_BASE_PAGE_GET(a)       C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, base_page)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_LP_NEXT_PAGE_GET(a)    C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, lp_next_page)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_LP_ABILITY_GET(a)      C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, lp_ability)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_STATE_GET(a)           C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, state)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_COMPLETE_GET(a)        C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, complete)
#define SBL_PML_STS_PCS_AUTONEG_NEXT_PAGE_PAGE_RECEIVED_GET(a)   C_GET(a, c_hni_pml_sts_pcs_autoneg_next_page, page_received)
#define SBL_PML_STS_RX_PCS_ALIGN_STATUS_GET(a)                   C_GET(a, c1_hni_pml_sts_rx_pcs, align_status)
#define SBL_PML_STS_RX_PCS_AM_LOCK_GET(a)                        C_GET(a, c1_hni_pml_sts_rx_pcs, am_lock)
#define SBL_PML_STS_RX_PCS_FAULT_GET(a)                          C_GET(a, c1_hni_pml_sts_rx_pcs, fault)
#define SBL_PML_STS_RX_PCS_HI_SER_GET(a)                         C_GET(a, c1_hni_pml_sts_rx_pcs, hi_ser)
#define SBL_PML_STS_RX_PCS_LOCAL_FAULT_GET(a)                    C_GET(a, c1_hni_pml_sts_rx_pcs, local_fault)
#define SBL_PML_STS_PCS_LANE_DEGRADE_RX_PLS_AVAILABLE_GET(a)     C_GET(a, c1_hni_pml_sts_pcs_lane_degrade, rx_pls_available)
#define SBL_PML_STS_PCS_LANE_DEGRADE_LP_PLS_AVAILABLE_GET(a)     C_GET(a, c1_hni_pml_sts_pcs_lane_degrade, lp_pls_available)

/* Other defines */
#define SBL_PML_AUTONEG_STATE_AUTONEG_OFF    0x0
#define SBL_PML_AUTONEG_STATE_AUTONEG_ENABLE 0x1
#define SBL_PML_AUTONEG_STATE_TX_DISABLE     0x2
#define SBL_PML_AUTONEG_STATE_ABILITY_DETECT 0x3
#define SBL_PML_AUTONEG_STATE_ACK_DETECT     0x4
#define SBL_PML_AUTONEG_STATE_COMPLETE_ACK   0x5
#define SBL_PML_AUTONEG_STATE_NEXT_PAGE_WAIT 0x6
#define SBL_PML_AUTONEG_STATE_AN_GOOD_CHECK  0x9
#define SBL_PML_AUTONEG_STATE_AN_GOOD        0xa

/* Cassini 2 macros */

#define SS2_PORT_PML_CFG_SUBPORT_RESET_WARM_RST_FROM_CSR_SET(a)                      C_SET(a, ss2_port_pml_cfg_subport_reset, warm_rst_from_csr)

#define SS2_PORT_PML_CFG_GENERAL_CLOCK_PERIOD_PS_UPDATE(a,b)                         C_UPDATE(a,b,ss2_port_pml_cfg_general,clock_period_ps)

#define SS2_PORT_PML_CFG_LLR_CAPACITY_MAX_DATA_UPDATE(a,b)                           C_UPDATE(a,b,ss2_port_pml_cfg_llr_capacity,max_data)
#define SS2_PORT_PML_CFG_LLR_CAPACITY_MAX_SEQ_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_llr_capacity,max_seq)
#define SS2_PORT_PML_CFG_LLR_SIZE_UPDATE(a,b)                                        C_UPDATE(a,b,ss2_port_pml_cfg_llr,size)
#define SS2_PORT_PML_CFG_LLR_ACK_NACK_ERR_CHECK_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_llr,ack_nack_err_check)
#define SS2_PORT_PML_CFG_LLR_PREAMBLE_SEQ_CHECK_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_llr,preamble_seq_check)
#define SS2_PORT_PML_CFG_LLR_CF_RATES_LOOP_TIMING_PERIOD_UPDATE(a,b)                 C_UPDATE(a,b,ss2_port_pml_cfg_llr_cf_rates, loop_timing_period)
#define SS2_PORT_PML_CFG_LLR_CF_SMAC_CTL_FRAME_SMAC_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_llr_cf_smac,ctl_frame_smac)
#define SS2_PORT_PML_CFG_LLR_CF_ETYPE_CTL_FRAME_ETHERTYPE_UPDATE(a,b)                C_UPDATE(a,b,ss2_port_pml_cfg_llr_cf_etype,ctl_frame_ethertype)
#define SS2_PORT_PML_CFG_LLR_SM_REPLAY_CT_MAX_GET(a)                                 C_GET(a,ss2_port_pml_cfg_llr_sm,replay_ct_max)
#define SS2_PORT_PML_CFG_LLR_SM_REPLAY_CT_MAX_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_llr_sm,replay_ct_max)
#define SS2_PORT_PML_CFG_LLR_SM_REPLAY_TIMER_MAX_UPDATE(a,b)                         C_UPDATE(a,b,ss2_port_pml_cfg_llr_sm,replay_timer_max)
#define SS2_PORT_PML_CFG_LLR_SM_RETRY_THRESHOLD_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_llr_sm,retry_threshold)
#define SS2_PORT_PML_CFG_LLR_SM_ALLOW_RE_INIT_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_llr_sm,allow_re_init)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_FILTER_LOSSLESS_WHEN_OFF_UPDATE(a,b)            C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,filter_lossless_when_off)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_MAX_STARVATION_LIMIT_UPDATE(a,b)                C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,max_starvation_limit)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_LINK_DOWN_BEHAVIOR_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,link_down_behavior)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_MAC_IF_CREDITS_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,mac_if_credits)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_FILTER_CTL_FRAMES_UPDATE(a,b)                   C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,filter_ctl_frames)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_ENABLE_LOOP_TIMING_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,enable_loop_timing)
#define SS2_PORT_PML_CFG_LLR_SUBPORT_LLR_MODE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_llr_subport,llr_mode)
#define SS2_PORT_PML_CFG_LLR_TIMEOUTS_DATA_AGE_TIMER_MAX_UPDATE(a,b)                 C_UPDATE(a,b,ss2_port_pml_cfg_llr_timeouts,data_age_timer_max)
#define SS2_PORT_PML_CFG_LLR_TIMEOUTS_PCS_LINK_DN_TIMER_MAX_UPDATE(a,b)              C_UPDATE(a,b,ss2_port_pml_cfg_llr_timeouts,pcs_link_dn_timer_max)

#define SS2_PORT_PML_CFG_PCS_AUTONEG_RESET_UPDATE(a,b)                               C_UPDATE(a,b,ss2_port_pml_cfg_pcs_autoneg, reset)
#define SS2_PORT_PML_CFG_PCS_AUTONEG_RESTART_UPDATE(a,b)                             C_UPDATE(a,b,ss2_port_pml_cfg_pcs_autoneg, restart)
#define SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE_LOADED_UPDATE(a,b)                    C_UPDATE(a,b,ss2_port_pml_cfg_pcs_autoneg, next_page_loaded)

#define SS2_PORT_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_UPDATE(a,b)                    C_UPDATE(a,b,ss2_port_pml_cfg_pcs,enable_auto_lane_degrade)
#define SS2_PORT_PML_CFG_PCS_ENABLE_AUTO_LANE_DEGRADE_GET(a)                         C_GET(a,ss2_port_pml_cfg_pcs,enable_auto_lane_degrade)
#define SS2_PORT_PML_CFG_PCS_PCS_MODE_UPDATE(a,b)                                    C_UPDATE(a,b,ss2_port_pml_cfg_pcs,pcs_mode)
#define SS2_PORT_PML_CFG_PCS_TIMESTAMP_SHIFT_UPDATE(a,b)                             C_UPDATE(a,b,ss2_port_pml_cfg_pcs, timestamp_shift)
#define SS2_PORT_PML_CFG_PCS_SUBPORT_PCS_ENABLE_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_pcs_subport, pcs_enable)
#define SS2_PORT_PML_CFG_PCS_SUBPORT_ENABLE_AUTO_NEG_UPDATE(a,b)                     C_UPDATE(a,b,ss2_port_pml_cfg_pcs_subport, enable_auto_neg)

#define SS2_PORT_PML_CFG_RX_MAC_FLIT_PACKING_CNT_UPDATE(a,b)                         C_UPDATE(a,b,ss2_port_pml_cfg_rx_mac,flit_packing_cnt)
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_MAC_OPERATIONAL_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_rx_mac_subport,mac_operational)
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_MAC_OPERATIONAL_GET(a)                       C_GET(a, ss2_port_pml_cfg_rx_mac_subport, mac_operational)
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_SHORT_PREAMBLE_UPDATE(a,b)                   C_UPDATE(a,b,ss2_port_pml_cfg_rx_mac_subport,short_preamble)

#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_ENABLE_CTL_OS_UPDATE(a,b)                    C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs_subport,enable_ctl_os)
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_ENABLE_LOCK_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs_subport,enable_lock)
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_ENABLE_RX_SM_UPDATE(a,b)                     C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs_subport,enable_rx_sm)
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_RS_MODE_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs_subport,rs_mode)
#define SS2_PORT_PML_CFG_RX_PCS_ACTIVE_LANES_UPDATE(a,b)                             C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,active_lanes)
#define SS2_PORT_PML_CFG_RX_PCS_CW_GAP_544_UPDATE(a,b)                               C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,cw_gap_544)
#define SS2_PORT_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_AMS_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,restart_lock_on_bad_ams)
#define SS2_PORT_PML_CFG_RX_PCS_RESTART_LOCK_ON_BAD_CWS_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,restart_lock_on_bad_cws)
#define SS2_PORT_PML_CFG_RX_PCS_ALLOW_AUTO_DEGRADE_UPDATE(a,b)                       C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs, allow_auto_degrade)
#define SS2_PORT_PML_CFG_RX_PCS_LANE_0_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,lane_0_source)
#define SS2_PORT_PML_CFG_RX_PCS_LANE_1_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,lane_1_source)
#define SS2_PORT_PML_CFG_RX_PCS_LANE_2_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,lane_2_source)
#define SS2_PORT_PML_CFG_RX_PCS_LANE_3_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_rx_pcs,lane_3_source)

#define SS2_PORT_PML_CFG_TX_MAC_IEEE_IFG_ADJUSTMENT_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac,ieee_ifg_adjustment)
#define SS2_PORT_PML_CFG_TX_MAC_MAC_PAD_IDLE_THRESH_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac,mac_pad_idle_thresh)
#define SS2_PORT_PML_CFG_TX_MAC_IFG_MODE_UPDATE(a,b)                                 C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac,ifg_mode)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_MAC_OPERATIONAL_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac_subport,mac_operational)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_MAC_OPERATIONAL_GET(a)                       C_GET(a, ss2_port_pml_cfg_tx_mac_subport, mac_operational)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_MAC_CDT_THRESH_UPDATE(a,b)                   C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac_subport,mac_cdt_thresh)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_MAC_CDT_INIT_VAL_UPDATE(a,b)                 C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac_subport,mac_cdt_init_val)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_PCS_CREDITS_UPDATE(a,b)                      C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac_subport,pcs_credits)
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_SHORT_PREAMBLE_UPDATE(a,b)                   C_UPDATE(a,b,ss2_port_pml_cfg_tx_mac_subport,short_preamble)

#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT_ENABLE_CTL_OS_UPDATE(a,b)                    C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs_subport,enable_ctl_os)
#define SS2_PORT_PML_CFG_TX_PCS_CDC_READY_LEVEL_UPDATE(a,b)                          C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,cdc_ready_level)
#define SS2_PORT_PML_CFG_TX_PCS_EN_PK_BW_LIMITER_UPDATE(a,b)                         C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,en_pk_bw_limiter)
#define SS2_PORT_PML_CFG_TX_PCS_LANE_0_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,lane_0_source)
#define SS2_PORT_PML_CFG_TX_PCS_LANE_1_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,lane_1_source)
#define SS2_PORT_PML_CFG_TX_PCS_LANE_2_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,lane_2_source)
#define SS2_PORT_PML_CFG_TX_PCS_LANE_3_SOURCE_UPDATE(a,b)                            C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,lane_3_source)
#define SS2_PORT_PML_CFG_TX_PCS_ALLOW_AUTO_DEGRADE_UPDATE(a,b)                       C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,allow_auto_degrade)
#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT_GEARBOX_CREDITS_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs_subport,gearbox_credits)
#define SS2_PORT_PML_CFG_TX_PCS_KEEP_ALL_LANES_ACTIVE_UPDATE(a,b)                    C_UPDATE(a,b,ss2_port_pml_cfg_tx_pcs,keep_all_lanes_active)

#define SS2_PORT_PML_ERR_INFO_PCS_TX_DP_TX_CDC_UNDERRUN_UPDATE(a,b)                  C_UPDATE(a,b,ss2_port_pml_err_info_pcs_tx_dp,tx_cdc_underrun)

#define SS2_PORT_PML_STS_LLR_LOOP_TIME_LOOP_TIME_GET(a)                              C_GET(a, ss2_port_pml_sts_llr_loop_time, loop_time)
#define SS2_PORT_PML_STS_LLR_LLR_STATE_GET(a)                                        C_GET(a, ss2_port_pml_sts_llr, llr_state)

#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_ALIGN_STATUS_GET(a)                          C_GET(a, ss2_port_pml_sts_rx_pcs_subport, align_status)
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_FAULT_GET(a)                                 C_GET(a, ss2_port_pml_sts_rx_pcs_subport, fault)
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_LOCAL_FAULT_GET(a)                           C_GET(a, ss2_port_pml_sts_rx_pcs_subport, local_fault)
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_HI_SER_GET(a)                                C_GET(a, ss2_port_pml_sts_rx_pcs_subport, hi_ser)

#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_LP_BASE_PAGE_GET(a)                   C_GET(a, ss2_port_pml_sts_pcs_autoneg_base_page, lp_base_page)
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_LP_ABILITY_GET(a)                     C_GET(a, ss2_port_pml_sts_pcs_autoneg_base_page, lp_ability)
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_BASE_PAGE_GET(a)                      C_GET(a, ss2_port_pml_sts_pcs_autoneg_base_page, base_page)
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_COMPLETE_GET(a)                       C_GET(a, ss2_port_pml_sts_pcs_autoneg_base_page, complete)
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_STATE_GET(a)                          C_GET(a, ss2_port_pml_sts_pcs_autoneg_base_page, state)
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_LP_NEXT_PAGE_GET(a)                   C_GET(a, ss2_port_pml_sts_pcs_autoneg_next_page, lp_next_page)
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_LP_ABILITY_GET(a)                     C_GET(a, ss2_port_pml_sts_pcs_autoneg_next_page, lp_ability)
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_BASE_PAGE_GET(a)                      C_GET(a, ss2_port_pml_sts_pcs_autoneg_next_page, base_page)
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_COMPLETE_GET(a)                       C_GET(a, ss2_port_pml_sts_pcs_autoneg_next_page, complete)
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_STATE_GET(a)                          C_GET(a, ss2_port_pml_sts_pcs_autoneg_next_page, state)

#define SS2_PORT_PML_CFG_RX_PCS_ACTIVE_LANES_GET(a)                                  C_GET(a, ss2_port_pml_cfg_rx_pcs, active_lanes)

#define SS2_PORT_PML_LINK_DN_BEHAVIOR_T_LD_DISCARD                                   C2_HNI_PML_LD_DISCARD
#define SS2_PORT_PML_LINK_DN_BEHAVIOR_T_LD_BLOCK                                     C2_HNI_PML_LD_BLOCK
#define SS2_PORT_PML_LINK_DN_BEHAVIOR_T_LD_BEST_EFFORT                               C2_HNI_PML_LD_BEST_EFFORT

#define SS2_PORT_SERDES_PMI_CTL                                                      C2_HNI_SERDES_PMI_CTL
#define SS2_PORT_SERDES_PMI_CTL_LP_ADDR_SET(a)                                       C_SET(a, c2_hni_serdes_pmi_ctl, pmi_lp_addr)
#define SS2_PORT_SERDES_PMI_CTL_LP_WRITE_SET(a)                                      C_SET(a, c2_hni_serdes_pmi_ctl, pmi_lp_write)
#define SS2_PORT_SERDES_PMI_CTL_LP_READ_VLD_GET(a)                                   C_GET(a, c2_hni_serdes_pmi_ctl, pmi_lp_read_vld)
#define SS2_PORT_SERDES_PMI_CTL_LP_ACK_GET(a)                                        C_GET(a, c2_hni_serdes_pmi_ctl, pmi_lp_ack)
#define SS2_PORT_SERDES_PMI_CTL_LP_ERROR_GET(a)                                      C_GET(a, c2_hni_serdes_pmi_ctl, pmi_lp_error)
#define SS2_PORT_SERDES_PMI_CTL_LP_EN_SET(a)                                         C_SET(a, c2_hni_serdes_pmi_ctl, pmi_lp_en)
#define SS2_PORT_SERDES_PMI_DATA                                                     C2_HNI_SERDES_PMI_DATA
#define SS2_PORT_SERDES_PMI_DATA_LP_RDDATA_GET(a)                                    C_GET(a, c2_hni_serdes_pmi_data, pmi_lp_rddata)
#define SS2_PORT_SERDES_PMI_DATA_LP_WRDATA_SET(a)                                    C_SET(a, c2_hni_serdes_pmi_data, pmi_lp_wrdata)
#define SS2_PORT_SERDES_PMI_DATA_LP_MASKDATA_SET(a)                                  C_SET(a, c2_hni_serdes_pmi_data, pmi_lp_maskdata)

#define SS2_PORT_PML_CFG_SERDES_RX_PMD_RX_LANE_MODE_UPDATE(a,b)                      C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_rx_lane_mode)
#define SS2_PORT_PML_CFG_SERDES_RX_PMD_RX_OSR_MODE_UPDATE(a,b)                       C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_rx_osr_mode)
#define SS2_PORT_PML_CFG_SERDES_RX_PMD_EXT_LOS_UPDATE(a,b)                           C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_ext_los)
#define SS2_PORT_PML_CFG_SERDES_RX_PMD_LN_RX_H_PWRDN_UPDATE(a,b)                     C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_ln_rx_h_pwrdn)
#define SS2_PORT_PML_CFG_SERDES_RX_PMD_LN_RX_DP_H_RSTB_UPDATE(a,b)                   C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_ln_rx_dp_h_rstb)
#define SS2_PORT_PML_CFG_SERDES_RX_PMD_LN_RX_H_RSTB_UPDATE(a,b)                      C_UPDATE(a, b, ss2_port_pml_cfg_serdes_rx, pmd_ln_rx_h_rstb)

#define SS2_PORT_PML_CFG_SERDES_RX_PMD_RX_OSR_MODE_GET(a)                            C_GET(a, ss2_port_pml_cfg_serdes_rx, pmd_rx_osr_mode)

#define SS2_PORT_PML_CFG_SERDES_TX_PMD_TX_LANE_MODE_UPDATE(a,b)                      C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_tx_lane_mode)
#define SS2_PORT_PML_CFG_SERDES_TX_PMD_TX_OSR_MODE_UPDATE(a,b)                       C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_tx_osr_mode)
#define SS2_PORT_PML_CFG_SERDES_TX_PMD_TX_DISABLE_UPDATE(a,b)                        C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_tx_disable)
#define SS2_PORT_PML_CFG_SERDES_TX_PMD_LN_TX_H_PWRDN_UPDATE(a,b)                     C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_ln_tx_h_pwrdn)
#define SS2_PORT_PML_CFG_SERDES_TX_PMD_LN_TX_DP_H_RSTB_UPDATE(a,b)                   C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_ln_tx_dp_h_rstb)
#define SS2_PORT_PML_CFG_SERDES_TX_PMD_LN_TX_H_RSTB_UPDATE(a,b)                      C_UPDATE(a, b, ss2_port_pml_cfg_serdes_tx, pmd_ln_tx_h_rstb)

#define SS2_PORT_PML_CFG_SERDES_TX_PMD_TX_OSR_MODE_GET(a)                            C_GET(a, ss2_port_pml_cfg_serdes_tx, pmd_tx_osr_mode)

#define SS2_PORT_PML_STS_SERDES_PMD_TX_DATA_VLD_GET(a)                               C_GET(a, ss2_port_pml_sts_serdes, pmd_tx_data_vld)
#define SS2_PORT_PML_STS_SERDES_PMD_TX_CLK_VLD_GET(a)                                C_GET(a, ss2_port_pml_sts_serdes, pmd_tx_clk_vld)
#define SS2_PORT_PML_STS_SERDES_PMD_RX_DATA_VLD_GET(a)                               C_GET(a, ss2_port_pml_sts_serdes, pmd_rx_data_vld)
#define SS2_PORT_PML_STS_SERDES_PMD_RX_CLK_VLD_GET(a)                                C_GET(a, ss2_port_pml_sts_serdes, pmd_rx_clk_vld)
#define SS2_PORT_PML_STS_SERDES_PMD_RX_LOCK_GET(a)                                   C_GET(a, ss2_port_pml_sts_serdes, pmd_rx_lock)
#define SS2_PORT_PML_STS_SERDES_PMD_SIGNAL_DETECTGET(a)                              C_GET(a, ss2_port_pml_sts_serdes, pmd_signal_detect)

#define SS2_PORT_PML_ERR_FLG_WORD1_LLR_REPLAY_AT_MAX_0_GET(a)                        C_WORD_GET(a, ss2_port_pml_err_flg, llr_replay_at_max_0, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_LLR_REPLAY_AT_MAX_1_GET(a)                        C_WORD_GET(a, ss2_port_pml_err_flg, llr_replay_at_max_1, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_LLR_REPLAY_AT_MAX_2_GET(a)                        C_WORD_GET(a, ss2_port_pml_err_flg, llr_replay_at_max_2, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_LLR_REPLAY_AT_MAX_3_GET(a)                        C_WORD_GET(a, ss2_port_pml_err_flg, llr_replay_at_max_3, 1)

#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_0_GET(a)                            C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_0, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_1_GET(a)                            C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_1, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_2_GET(a)                            C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_2, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_3_GET(a)                            C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_3, 1)

#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_LF_0_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_lf_0, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_LF_1_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_lf_1, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_LF_2_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_lf_2, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_LF_3_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_lf_3, 1)

#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_RF_0_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_rf_0, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_RF_1_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_rf_1, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_RF_2_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_rf_2, 1)
#define SS2_PORT_PML_ERR_FLG_WORD1_PCS_LINK_DOWN_RF_3_GET(a)                         C_WORD_GET(a, ss2_port_pml_err_flg, pcs_link_down_rf_3, 1)

#define SS2_PORT_PML_STS_PCS_LANE_DEGRADE_WORD0_LP_PLS_AVAILABLE_GET(a)              C_WORD_GET(a, ss2_port_pml_sts_pcs_lane_degrade, lp_pls_available, 0)
#define SS2_PORT_PML_STS_PCS_LANE_DEGRADE_WORD0_RX_PLS_AVAILABLE_GET(a)              C_WORD_GET(a, ss2_port_pml_sts_pcs_lane_degrade, rx_pls_available, 0)

#define SS2_PORT_PML_LLR_STATE_T_OFF_LLR                                             C2_HNI_PML_OFF_LLR
#define SS2_PORT_PML_LLR_STATE_T_INIT                                                C2_HNI_PML_INIT
#define SS2_PORT_PML_LLR_STATE_T_ADVANCE                                             C2_HNI_PML_ADVANCE
#define SS2_PORT_PML_LLR_STATE_T_HALT                                                C2_HNI_PML_HALT
#define SS2_PORT_PML_LLR_STATE_T_REPLAY                                              C2_HNI_PML_REPLAY
#define SS2_PORT_PML_LLR_STATE_T_DISCARD                                             C2_HNI_PML_DISCARD
#define SS2_PORT_PML_LLR_STATE_T_MONITOR                                             C2_HNI_PML_MONITOR

#define SS2_MB_STS_REV_PLATFORM_GET(a)                                               C_GET(a, c_mb_sts_rev, platform)

#endif /* _SBL_PML_H_ */
