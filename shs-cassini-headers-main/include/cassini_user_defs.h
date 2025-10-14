// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
* Cassini hardware definitions
* Copyright 2018-2021 Hewlett Packard Enterprise Development LP
*
* This file is generated. Do not modify.
*/

#ifndef __CASSINI_USER_DEFS_H
#define __CASSINI_USER_DEFS_H

#include <asm/byteorder.h>

#ifndef __LITTLE_ENDIAN
#error "Non-little endian builds not supported"
#endif

#ifndef __KERNEL__
#include <endian.h>
#define be64_to_cpu be64toh
#define be32_to_cpu be32toh
#define be16_to_cpu be16toh
#define cpu_to_be64 htobe64
#define cpu_to_be32 htobe32
#define cpu_to_be16 htobe16
#endif

#define C_CQ_BASE 0x00000000
#define C_PCT_BASE 0x00800000
#define C_HNI_BASE 0x01000000
#define C_HNI_PML_BASE 0x01800000
#define C_RMU_BASE 0x02000000
#define C_IXE_BASE 0x02800000
#define C_ATU_BASE 0x08000000
#define C_EE_BASE 0x08800000
#define C_PARBS_BASE 0x09000000
#define C_LPE_BASE 0x09800000
#define C_MST_BASE 0x0a000000
#define C_OXE_BASE 0x0a800000
#define C_MB_BASE 0x10000000
#define C_PI_BASE 0x10800000
#define C_PI_IPD_BASE 0x11000000
#define C_MEMORG_CSR 0x40000000u
#define C_MEMORG_CSR_SIZE 0x28000000u
#define C_MEMORG_CQ_LAUNCH 0x00000000u
#define C_MEMORG_CQ_LAUNCH_SIZE 0x08000000u
#define C_MEMORG_CQ_TOU 0x08000000u
#define C_MEMORG_EE 0x10000000u
#define C_CQ_LAUNCH_TXQ_BASE (C_MEMORG_CQ_LAUNCH + 0x00000000)
#define C_CQ_LAUNCH_TGT_BASE (C_MEMORG_CQ_LAUNCH + 0x04000000)
#define C_CQ_LAUNCH_PAGE_SIZE 0x10000
#define C_TOU_LAUNCH_PAGE_SIZE 0x10000
#define C_EE_SW_STATE_PAGE_SIZE 0x10000
#define C_NUM_CTS 2048
#define C_CT_NONE 0
#define C_AC_NONE 0
#define C_CID_NONE 0
#define C_NUM_EQS 2048
#define C_EQ_NONE 0
#define C_NUM_ACS 1024
#define C_NUM_PTLTES 2048
#define C_NUM_TRANSMIT_CQS 1024
#define C_NUM_TARGET_CQS 512
#define C_NID_ANY 1048575
#define NICSIM_CONFIG_DUMP 0x27fff008
#define NICSIM_CONFIG_LOG 0x27fff010
#define NICSIM_CONFIG_STATE 0x27fff018
#define NICSIM_NIC_ID 0x27fff060
#define C_PLATFORM_ASIC 0
#define C_PLATFORM_NETSIM 1
#define C_PLATFORM_Z1 2
#define C_PLATFORM_FPGA 3
#define C_MST_TABLE_SIMPLE_OFFSET 16
#define C_PE_COUNT 4
#define C_COMM_PROF_PER_CQ 16
#define C_DFA_PID_BITS_MIN 6
#define C_DFA_PID_BITS_MAX 9
#define C_PID_ANY ((1 << C_DFA_PID_BITS_MAX) - 1)
#define C_RANK_ANY ((1 << (32 - C_DFA_PID_BITS_MAX)) - 1)
#define C_DFA_NIC_BITS 20
#define C_DFA_MULTICAST_ID_BITS 13
#define C_DFA_INDEX_EXT_BITS 5
#define C_CT_SUCCESS_BITS 48
#define C_CT_FAILURE_BITS 7
#define C_DFA_ENDPOINT_DEFINED_BITS 12
#define C_DFA_ENDPOINT_BITS (C_DFA_ENDPOINT_DEFINED_BITS + C_DFA_INDEX_EXT_BITS)
#define C_NUM_RGIDS 256
#define C_RESERVED_RGID 0
#define C_OXE_CQ_MCU_START 0
#define C_OXE_CQ_MCU_COUNT 48
#define C_OXE_LPE_MCU_START (C_OXE_CQ_MCU_START + C_OXE_CQ_MCU_COUNT)
#define C_OXE_LPE_MCU_COUNT 16
#define C_OXE_IXE_MCU_START (C_OXE_LPE_MCU_START + C_OXE_LPE_MCU_COUNT)
#define C_OXE_IXE_MCU_COUNT 24
#define C_OXE_TOU_MCU_START (C_OXE_IXE_MCU_START + C_OXE_IXE_MCU_COUNT)
#define C_OXE_TOU_MCU_COUNT 8
#define C_OXE_MCU_COUNT (C_OXE_CQ_MCU_COUNT + C_OXE_IXE_MCU_COUNT + C_OXE_LPE_MCU_COUNT + C_OXE_TOU_MCU_COUNT)
#define C_NUM_LACS 8
#define C_NUM_TLES 2048
#define C_NUM_VFS 64
#define C_MAX_IDC_PAYLOAD_RES 224
#define C_MAX_IDC_PAYLOAD_UNR 192
#define C_ATU_PRB_ENTRIES 512

enum c_rsp_status {
	C_RSP_PEND = 0,
	C_RSP_NACK_RCVD = 1,
	C_RSP_GET_ACK_RCVD = 2,
	C_RSP_OP_COMP = 3,
};

enum c_sct_status {
	C_SCT_NOT_USED = 0,
	C_SCT_NORMAL = 1,
	C_SCT_STALL = 2,
	C_SCT_CLEAR_PEND = 3,
	C_SCT_CLOSE_PEND = 4,
	C_SCT_CLOSE_TIMEOUT = 5,
	C_SCT_CLOSE_COMP = 6,
	C_SCT_RETRY = 7,
};

enum c_port_v4_pkt_type {
	C_V4_PKT_UNRESTRICTED = 0,
	C_V4_PKT_SMALLMSG = 1,
	C_V4_PKT_CONTINUATION = 2,
	C_V4_PKT_CONN_MGMT = 3,
	C_V4_PKT_RESTRICTED = 4,
	C_RESERVED5 = 5,
	C_RESERVED6 = 6,
	C_V4_PKT_HRP_ACK = 7,
};

enum c_port_vs_pkt_type {
	C_VS_PKT_PUT64 = 0,
	C_VS_PKT_GET64 = 1,
	C_VS_PKT_PUT32 = 2,
	C_VS_PKT_GET32 = 3,
	C_VS_PKT_AMO64 = 4,
	C_VS_PKT_FAMO64 = 5,
	C_VS_PKT_AMO32 = 6,
	C_VS_PKT_FAMO32 = 7,
};

enum c_port_rsp_pkt_type {
	C_RSP_PKT_ANY = 0,
	C_RSP_PKT_32B = 1,
};

enum c_format {
	C_PKT_FORMAT_OPT = 0,
	C_PKT_FORMAT_STD = 1,
};

enum c_checksum_ctrl {
	C_CHECKSUM_CTRL_NONE = 0,
	C_CHECKSUM_CTRL_ROCE = 1,
	C_CHECKSUM_CTRL_UDP = 2,
	C_CHECKSUM_CTRL_TCP = 3,
};

enum c_cmd_size {
	C_CMD_SIZE_32B = 0,
	C_CMD_SIZE_64B = 1,
	C_CMD_SIZE_128B = 2,
	C_CMD_SIZE_256B = 3,
};

enum c_ptl_list {
	C_PTL_LIST_PRIORITY = 0,
	C_PTL_LIST_OVERFLOW = 1,
	C_PTL_LIST_REQUEST = 2,
	C_PTL_LIST_UNEXPECTED = 3,
};

enum c_atomic_op {
	C_AMO_OP_MIN = 0,
	C_AMO_OP_MAX = 1,
	C_AMO_OP_SUM = 2,
	C_AMO_OP_LOR = 4,
	C_AMO_OP_LAND = 5,
	C_AMO_OP_BOR = 6,
	C_AMO_OP_BAND = 7,
	C_AMO_OP_LXOR = 8,
	C_AMO_OP_BXOR = 9,
	C_AMO_OP_SWAP = 10,
	C_AMO_OP_CSWAP = 11,
	C_AMO_OP_AXOR = 12,
};

enum c_cswap_op {
	C_AMO_OP_CSWAP_EQ = 0,
	C_AMO_OP_CSWAP_NE = 1,
	C_AMO_OP_CSWAP_LE = 2,
	C_AMO_OP_CSWAP_LT = 3,
	C_AMO_OP_CSWAP_GE = 4,
	C_AMO_OP_CSWAP_GT = 5,
};

enum c_atomic_type {
	C_AMO_TYPE_INT8_T = 0,
	C_AMO_TYPE_UINT8_T = 1,
	C_AMO_TYPE_INT16_T = 2,
	C_AMO_TYPE_UINT16_T = 3,
	C_AMO_TYPE_INT32_T = 4,
	C_AMO_TYPE_UINT32_T = 5,
	C_AMO_TYPE_INT64_T = 6,
	C_AMO_TYPE_UINT64_T = 7,
	C_AMO_TYPE_FLOAT_T = 8,
	C_AMO_TYPE_FLOAT_COMPLEX_T = 9,
	C_AMO_TYPE_DOUBLE_T = 10,
	C_AMO_TYPE_DOUBLE_COMPLEX_T = 11,
	C_AMO_TYPE_UINT128_T = 12,
};

enum c_ptlte_state {
	C_PTLTE_RESET = 0,
	C_PTLTE_DISABLED = 1,
	C_PTLTE_ENABLED = 2,
	C_PTLTE_SOFTWARE_MANAGED = 3,
	C_PTLTE_ETHERNET = 4,
};

enum c_cmd_type {
	C_CMD_TYPE_IDC = 0,
	C_CMD_TYPE_DMA = 1,
	C_CMD_TYPE_CT = 2,
	C_CMD_TYPE_CQ = 3,
};

enum c_dma_op {
	C_CMD_NOOP = 0,
	C_CMD_PUT = 1,
	C_CMD_GET = 2,
	C_CMD_RENDEZVOUS_PUT = 3,
	C_CMD_ATOMIC = 4,
	C_CMD_FETCHING_ATOMIC = 5,
	C_CMD_ETHERNET_TX = 6,
	C_CMD_SMALL_MESSAGE = 7,
	C_CMD_NOMATCH_PUT = 8,
	C_CMD_NOMATCH_GET = 9,
	C_CMD_CSTATE = 10,
	C_CMD_CLOSE = 11,
	C_CMD_CLEAR = 12,
	C_CMD_REDUCTION = 13,
};

enum c_ct_op {
	C_CMD_CT_NOOP = 0,
	C_CMD_CT_SET = 1,
	C_CMD_CT_GET = 2,
	C_CMD_CT_INC = 3,
	C_CMD_CT_CANCEL = 4,
	C_CMD_CT_TRIG_DMA = 5,
	C_CMD_CT_TRIG_SET = 6,
	C_CMD_CT_TRIG_INC = 7,
	C_CMD_CT_TRIG_EVENT = 8,
};

enum c_cq_op {
	C_CMD_CQ_NOOP = 0,
	C_CMD_CQ_FENCE = 1,
	C_CMD_CQ_LCID = 2,
};

enum c_tgt_op {
	C_CMD_TGT_APPEND = 8,
	C_CMD_TGT_SEARCH = 9,
	C_CMD_TGT_SEARCH_AND_DELETE = 10,
	C_CMD_TGT_UNLINK = 11,
	C_CMD_TGT_SETSTATE = 12,
};

enum c_event_size {
	C_EVENT_SIZE_NO_EVENT = 0,
	C_EVENT_SIZE_16_BYTE = 1,
	C_EVENT_SIZE_32_BYTE = 2,
	C_EVENT_SIZE_64_BYTE = 3,
};

enum c_event_type {
	C_EVENT_PUT = 0,
	C_EVENT_GET = 1,
	C_EVENT_ATOMIC = 2,
	C_EVENT_FETCH_ATOMIC = 3,
	C_EVENT_PUT_OVERFLOW = 4,
	C_EVENT_GET_OVERFLOW = 5,
	C_EVENT_ATOMIC_OVERFLOW = 6,
	C_EVENT_FETCH_ATOMIC_OVERFLOW = 7,
	C_EVENT_SEND = 8,
	C_EVENT_ACK = 9,
	C_EVENT_REPLY = 10,
	C_EVENT_LINK = 11,
	C_EVENT_SEARCH = 12,
	C_EVENT_STATE_CHANGE = 13,
	C_EVENT_UNLINK = 14,
	C_EVENT_RENDEZVOUS = 15,
	C_EVENT_ETHERNET = 16,
	C_EVENT_COMMAND_FAILURE = 17,
	C_EVENT_TRIGGERED_OP = 18,
	C_EVENT_ETHERNET_FGFC = 19,
	C_EVENT_PCT = 20,
	C_EVENT_MATCH = 21,
	C_EVENT_ERROR = 28,
	C_EVENT_TIMESTAMP = 29,
	C_EVENT_EQ_SWITCH = 30,
	C_EVENT_NULL_EVENT = 31,
};

enum c_return_code {
	C_RC_NO_EVENT = 0,
	C_RC_OK = 1,
	C_RC_UNDELIVERABLE = 2,
	C_RC_PT_DISABLED = 3,
	C_RC_DROPPED = 4,
	C_RC_PERM_VIOLATION = 5,
	C_RC_OP_VIOLATION = 6,
	C_RC_NO_MATCH = 8,
	C_RC_UNCOR = 9,
	C_RC_UNCOR_TRNSNT = 10,
	C_RC_NO_SPACE = 16,
	C_RC_ENTRY_NOT_FOUND = 18,
	C_RC_NO_TARGET_CONN = 19,
	C_RC_NO_TARGET_MST = 20,
	C_RC_NO_TARGET_TRS = 21,
	C_RC_SEQUENCE_ERROR = 22,
	C_RC_NO_MATCHING_CONN = 23,
	C_RC_INVALID_DFA_FORMAT = 24,
	C_RC_VNI_NOT_FOUND = 25,
	C_RC_PTLTE_NOT_FOUND = 26,
	C_RC_PTLTE_SW_MANAGED = 27,
	C_RC_SRC_ERROR = 28,
	C_RC_MST_CANCELLED = 29,
	C_RC_HRP_CONFIG_ERROR = 30,
	C_RC_HRP_RSP_ERROR = 31,
	C_RC_HRP_RSP_DISCARD = 32,
	C_RC_INVALID_AC = 33,
	C_RC_PAGE_PERM_ERROR = 34,
	C_RC_ATS_ERROR = 35,
	C_RC_NO_TRANSLATION = 36,
	C_RC_PAGE_REQ_ERROR = 37,
	C_RC_PCIE_ERROR_POISONED = 38,
	C_RC_PCIE_UNSUCCESS_CMPL = 39,
	C_RC_AMO_INVAL_OP_ERROR = 40,
	C_RC_AMO_ALIGN_ERROR = 41,
	C_RC_AMO_FP_INVALID = 42,
	C_RC_AMO_FP_UNDERFLOW = 43,
	C_RC_AMO_FP_OVERFLOW = 44,
	C_RC_AMO_FP_INEXACT = 45,
	C_RC_ILLEGAL_OP = 46,
	C_RC_INVALID_ENDPOINT = 47,
	C_RC_RESTRICTED_UNICAST = 48,
	C_RC_CMD_ALIGN_ERROR = 49,
	C_RC_CMD_INVALID_ARG = 50,
	C_RC_INVALID_EVENT = 51,
	C_RC_ADDR_OUT_OF_RANGE = 52,
	C_RC_CONN_CLOSED = 53,
	C_RC_CANCELED = 54,
	C_RC_NO_MATCHING_TRS = 55,
	C_RC_NO_MATCHING_MST = 56,
	C_RC_DELAYED = 57,
	C_RC_AMO_LENGTH_ERROR = 58,
	C_RC_PKTBUF_ERROR = 59,
	C_RC_RESOURCE_BUSY = 60,
	C_RC_FLUSH_TRANSLATION = 61,
	C_RC_TRS_PEND_RSP = 62,
};

enum c_rss_hash_type {
	C_RSS_HASH_NONE = 0,
	C_RSS_HASH_IPV4 = 1,
	C_RSS_HASH_IPV4_TCP = 2,
	C_RSS_HASH_IPV4_UDP = 3,
	C_RSS_HASH_IPV4_PROTOCOL = 4,
	C_RSS_HASH_IPV4_PROTOCOL_TCP = 5,
	C_RSS_HASH_IPV4_PROTOCOL_UDP = 6,
	C_RSS_HASH_IPV4_PROTOCOL_UDP_ROCE = 7,
	C_RSS_HASH_IPV4_FLOW_LABEL = 8,
	C_RSS_HASH_IPV6 = 9,
	C_RSS_HASH_IPV6_TCP = 10,
	C_RSS_HASH_IPV6_UDP = 11,
	C_RSS_HASH_IPV6_PROTOCOL = 12,
	C_RSS_HASH_IPV6_PROTOCOL_TCP = 13,
	C_RSS_HASH_IPV6_PROTOCOL_UDP = 14,
	C_RSS_HASH_IPV6_PROTOCOL_UDP_ROCE = 15,
	C_RSS_HASH_IPV6_FLOW_LABEL = 16,
};

enum c_sc_reason {
	C_SC_FC_EQ_FULL = 0,
	C_SC_FC_NO_MATCH = 1,
	C_SC_FC_UNEXPECTED_FAIL = 2,
	C_SC_FC_REQUEST_FULL = 3,
	C_SC_SM_UNEXPECTED_FAIL = 4,
	C_SC_SM_APPEND_FAIL = 5,
	C_SC_DIS_UNCOR = 6,
};

enum c_pct_event_type {
	C_PCT_REQUEST_NACK = 0,
	C_PCT_REQUEST_TIMEOUT = 1,
	C_PCT_SCT_TIMEOUT = 2,
	C_PCT_TCT_TIMEOUT = 3,
	C_PCT_ACCEL_CLOSE_COMPLETE = 4,
	C_PCT_RETRY_COMPLETE = 5,
};

enum c_error_class {
	C_EC_SFTWR = 0,
	C_EC_INFO = 1,
	C_EC_TRS_NS = 2,
	C_EC_TRS_S = 3,
	C_EC_TRNSNT_NS = 4,
	C_EC_TRNSNT_S = 5,
	C_EC_BADCON_NS = 6,
	C_EC_BADCON_S = 7,
	C_EC_DEGRD_NS = 8,
	C_EC_COR = 9,
	C_EC_UNCOR_NS = 10,
	C_EC_UNCOR_S = 11,
	C_EC_CRIT = 12,
};

enum c_odp_status {
	C_ATU_ODP_SUCCESS = 0,
	C_ATU_ODP_FAILURE = 1,
};

enum c_ta_mode {
	C_ATU_ATS_MODE = 0,
	C_ATU_NTA_MODE = 1,
	C_ATU_ATS_DYNAMIC_MODE = 2,
	C_ATU_PASSTHROUGH_MODE = 3,
};

enum c_ptg_mode {
	C_ATU_PTG_MODE_SGL = 0,
	C_ATU_PTG_MODE_DBL_A = 1,
	C_ATU_PTG_MODE_DBL_B = 2,
	C_ATU_PTG_MODE_DBL_C = 3,
	C_ATU_PTG_MODE_TPL_A = 4,
	C_ATU_PTG_MODE_TPL_B = 5,
};

enum c_odp_mode {
	C_ATU_ODP_MODE_ATS_PRS = 0,
	C_ATU_ODP_MODE_NIC_PRI = 1,
};

enum c_atucq_cmd {
	C_ATU_NOP = 0,
	C_ATU_INVALIDATE_PAGES = 1,
	C_ATU_COMPLETION_WAIT = 2,
	C_ATU_RESERVED = 3,
};

enum c_enet_frame_type {
	C_RMU_ENET_802_3 = 1,
	C_RMU_ENET_OPT_IPV4 = 2,
	C_RMU_ENET_OPT_IPV6 = 3,
};

enum c_msix_ints {
	C_PF_TO_VF_MSIX_INT = 0,
	C_PI_IPD_IRQA_MSIX_INT = 0,
	C_PI_IRQA_MSIX_INT = 1,
	C_MB_IRQA_MSIX_INT = 2,
	C_CQ_IRQA_MSIX_INT = 3,
	C_PCT_IRQA_MSIX_INT = 4,
	C_HNI_IRQA_MSIX_INT = 5,
	C_HNI_PML_IRQA_MSIX_INT = 6,
	C_RMU_IRQA_MSIX_INT = 7,
	C_IXE_IRQA_MSIX_INT = 8,
	C_ATU_IRQA_MSIX_INT = 9,
	C_EE_IRQA_MSIX_INT = 10,
	C_PARBS_IRQA_MSIX_INT = 11,
	C_LPE_IRQA_MSIX_INT = 12,
	C_MST_IRQA_MSIX_INT = 13,
	C_OXE_IRQA_MSIX_INT = 14,
	C_PI_IPD_IRQB_MSIX_INT = 15,
	C_PI_IRQB_MSIX_INT = 16,
	C_MB_IRQB_MSIX_INT = 17,
	C_CQ_IRQB_MSIX_INT = 18,
	C_PCT_IRQB_MSIX_INT = 19,
	C_HNI_IRQB_MSIX_INT = 20,
	C_HNI_PML_IRQB_MSIX_INT = 21,
	C_RMU_IRQB_MSIX_INT = 22,
	C_IXE_IRQB_MSIX_INT = 23,
	C_ATU_IRQB_MSIX_INT = 24,
	C_EE_IRQB_MSIX_INT = 25,
	C_PARBS_IRQB_MSIX_INT = 26,
	C_LPE_IRQB_MSIX_INT = 27,
	C_MST_IRQB_MSIX_INT = 28,
	C_OXE_IRQB_MSIX_INT = 29,
	C_PI_DMAC_MSIX_INT = 30,
	C_PI_IPD_VF_PF_MSIX_INT = 31,
	C_FIRST_AVAIL_MSIX_INT = 32,
};

enum c_pi_err_flg_bit {
	C_PI_ERR_FLG__UC_ATTENTION_0 = 62,
	C_PI_ERR_FLG__UC_ATTENTION_1 = 63,
};

enum c_pct_bitmask {
	C_PCT_CFG_SRB_RETRY_GRP_CTRL__GRP_LOADED_MSK = 0x1,
	C_PCT_CFG_SRB_RETRY_GRP_CTRL__GRP_ACTIVE_MSK = 0x2,
	C_PCT_CFG_SW_SIM_SRC_RSP__LOADED_MSK = 0x1000000000000,
	C_PCT_CFG_SW_RETRY_SRC_CLS_REQ__LOADED_MSK = 0x10000,
	C_PCT_CFG_SW_SIM_TGT_CLS_REQ__LOADED_MSK = 0x10000,
};

enum c2_link_dn_behavior {
	C2_HNI_PML_LD_DISCARD = 0,
	C2_HNI_PML_LD_BLOCK = 1,
	C2_HNI_PML_LD_BEST_EFFORT = 2,
};

enum c2_llr_state {
	C2_HNI_PML_OFF_LLR = 0,
	C2_HNI_PML_INIT = 1,
	C2_HNI_PML_ADVANCE = 2,
	C2_HNI_PML_HALT = 3,
	C2_HNI_PML_REPLAY = 4,
	C2_HNI_PML_DISCARD = 5,
	C2_HNI_PML_MONITOR = 6,
};

enum c2_link_function {
	C2_LF_ETH_ACC = 0,
	C2_LF_ETH_TRUNK = 1,
	C2_LF_C3_R2 = 2,
	C2_LF_C2_R2 = 3,
	C2_LF_C1_R2 = 4,
	C2_LF_C3_R1 = 5,
	C2_LF_R1_R2 = 6,
	C2_LF_R2_R2_FAB_NRML = 7,
	C2_LF_R2_R2_FAB_TRANS = 8,
	C2_LF_R2_R2_FAB_TO_FAB = 9,
	C2_LF_R2_R2_FAB_RTR = 10,
	C2_LF_C2_R1 = 11,
};

struct c_fa_anon {
	uint8_t action      :  3;
	uint32_t            : 20;
	uint16_t fa_format  :  9;
};

struct c_fa_unicast {
	uint16_t endpoint_defined  : 12;
	uint32_t nid               : 20;
};

struct c_fa_multicast {
	uint8_t action         :  3;
	uint16_t multicast_id  : 13;
	uint8_t                :  2;
	uint8_t reduction_id   :  3;
	uint8_t                :  2;
	uint16_t format        :  9;
};

union c_fab_addr {
	struct c_fa_multicast multicast;
	struct c_fa_unicast unicast;
	struct c_fa_anon anon;
};

struct c_process {
	uint32_t process;
};

struct c_hdr_data {
	uint64_t hdr_data;
};

struct c_padded_ct_success {
	uint64_t value      : 48;
	uint16_t unused_0;
};

struct c_padded_ct_failure {
	uint8_t value  :  7;
	uint64_t       : 57;
};

struct c_tct_idx {
	uint16_t tct_idx  : 13;
	uint8_t           :  3;
};

#ifdef __LITTLE_ENDIAN
struct c_pkt_type {
	uint8_t opcode        :  4;
	uint8_t ver_pkt_type  :  3;
	uint8_t req_not_rsp   :  1;
};
#else
struct c_pkt_type {
	uint8_t req_not_rsp   :  1;
	uint8_t ver_pkt_type  :  3;
	uint8_t opcode        :  4;
};
#endif

struct c_ts {
	uint32_t ns   : 30;
	uint8_t rsvd  :  2;
	uint64_t sec  : 48;
} __attribute__((packed));

struct c_padded_sct_idx {
	uint16_t sct_idx  : 12;
	uint8_t           :  1;
	uint8_t           :  3;
};

union c_conn_idx {
	struct c_tct_idx tct_idx;
	struct c_padded_sct_idx padded_sct_idx;
};

struct c_ule_entry {
	uint32_t                    : 17;
	uint8_t restricted          :  1;
	uint8_t tc                  :  3;
	uint8_t use_offset_for_get  :  1;
	uint16_t rendezvous_id      :  8;
	uint16_t vni                : 16;
	uint16_t dscp               :  6;
	uint32_t sfa_nid            : 20;
	uint64_t hdr_data           : 64;
	uint64_t match_bits         : 64;
	uint32_t initiator          : 32;
	uint64_t start_or_offset    : 57;
	uint32_t mlength            : 32;
	uint32_t request_len        : 32;
	uint8_t opcode              :  4;
	uint8_t src_error           :  1;
	uint16_t addr               : 14;
	uint8_t vld                 :  1;
	uint16_t                    : 11;
} __attribute__((packed));

struct c_amo_payload {
	uint8_t amo_op       :  4;
	uint8_t              :  4;
	uint8_t amo_type     :  4;
	uint8_t              :  4;
	uint8_t cswap_op     :  3;
	uint8_t              :  5;
	uint64_t             : 40;
	uint64_t op1_word1;
	uint64_t op1_word2;
	uint64_t op2_word1;
	uint64_t op2_word2;
};

struct c_small_amo_payload {
	uint8_t amo_op       :  4;
	uint8_t              :  4;
	uint8_t amo_type     :  4;
	uint8_t              :  4;
	uint64_t             : 48;
	uint64_t op1_word1;
};

struct c_page_table_entry {
	uint8_t       :  6;
	uint64_t ptr  : 51;
	uint8_t r     :  1;
	uint8_t w     :  1;
	uint8_t leaf  :  1;
	uint8_t p     :  1;
	uint8_t       :  3;
};

struct c_page_request_entry {
	uint64_t addr   : 45;
	uint16_t index  :  9;
	uint16_t acid   : 10;
} __attribute__((packed));

struct c_cmd {
	uint8_t cmd_type  :  2;
	uint8_t cmd_size  :  2;
	uint8_t opcode    :  4;
};

struct c_cmdq {
	uint16_t wr_ptr;
	uint16_t unused_0[511];
	uint64_t ll_wr64a0123[128];
	uint64_t ll_wr128a02[128];
	uint64_t ll_wr128a13[128];
};

struct c_cq_status {
	uint16_t rd_ptr;
	uint16_t num_write_cmds_rejected;
	uint8_t fail_command;
	uint8_t unused_0;
	uint8_t return_code                :  6;
	uint16_t                           :  7;
	uint8_t fail_loc_32                :  1;
	uint8_t fail_ll                    :  1;
	uint8_t ll_disabled                :  1;
};

struct c_idc_hdr {
	struct c_cmd command;
	uint8_t length;
	uint16_t unused_0;
	union c_fab_addr dfa;
	uint64_t remote_offset  : 56;
	uint8_t unused_1;
};

struct c_idc_amo_cmd {
	struct c_idc_hdr idc_header;
	uint64_t local_addr           : 57;
	uint8_t                       :  7;
	uint8_t atomic_op             :  4;
	uint8_t                       :  4;
	uint8_t atomic_type           :  4;
	uint8_t                       :  4;
	uint8_t cswap_op              :  3;
	uint8_t                       :  5;
	uint64_t                      : 40;
	uint64_t op1_word1;
	uint64_t op1_word2;
	uint64_t op2_word1;
	uint64_t op2_word2;
};

struct c_idc_msg_hdr {
	struct c_cmd command;
	uint8_t length;
	uint16_t unused_0;
	union c_fab_addr dfa;
	uint64_t match_bits;
	uint64_t header_data;
	uint64_t user_ptr;
	uint8_t data[0];
};

struct c_cstate_cmd {
	struct c_cmd command;
	uint8_t length;
	uint8_t write_lac              :  3;
	uint8_t event_send_disable     :  1;
	uint8_t event_success_disable  :  1;
	uint8_t event_ct_send          :  1;
	uint8_t event_ct_reply         :  1;
	uint8_t event_ct_ack           :  1;
	uint8_t event_ct_bytes         :  1;
	uint8_t get_with_local_flag    :  1;
	uint8_t restricted             :  1;
	uint8_t reduction              :  1;
	uint8_t flush                  :  1;
	uint8_t use_offset_for_get     :  1;
	uint8_t                        :  1;
	uint8_t                        :  1;
	uint32_t initiator;
	uint8_t index_ext              :  5;
	uint8_t                        :  3;
	uint64_t                       : 56;
	uint64_t user_ptr;
	uint32_t unused_0;
	uint16_t ct                    : 11;
	uint8_t                        :  5;
	uint16_t eq                    : 11;
	uint8_t                        :  5;
};

struct c_idc_eth_cmd {
	struct c_cmd command;
	uint8_t length;
	uint8_t flow_hash;
	uint8_t checksum_ctrl     :  2;
	uint8_t                   :  2;
	uint8_t fmt               :  1;
	uint8_t                   :  3;
	uint16_t checksum_start   : 10;
	uint8_t                   :  2;
	uint16_t checksum_offset  :  6;
	uint16_t                  : 14;
	uint64_t unused_0;
	uint8_t data[0];
} __attribute__((packed));

struct c_dma_eth_cmd {
	struct c_cmd command;
	uint8_t read_lac               :  3;
	uint8_t                        :  4;
	uint8_t fmt                    :  1;
	uint8_t flow_hash;
	uint8_t checksum_ctrl          :  2;
	uint8_t                        :  3;
	uint8_t num_segments           :  3;
	uint16_t checksum_start        : 10;
	uint8_t                        :  2;
	uint16_t checksum_offset       :  6;
	uint16_t                       : 14;
	uint64_t user_ptr;
	uint16_t len[7];
	uint16_t unused_0;
	uint64_t addr[7];
	uint16_t total_len;
	uint8_t event_send_disable     :  1;
	uint8_t event_success_disable  :  1;
	uint8_t event_ct_send          :  1;
	uint8_t event_ct_reply         :  1;
	uint8_t event_ct_ack           :  1;
	uint8_t event_ct_bytes         :  1;
	uint8_t get_with_local_flag    :  1;
	uint8_t restricted             :  1;
	uint8_t reduction              :  1;
	uint8_t flush                  :  1;
	uint8_t use_offset_for_get     :  1;
	uint8_t                        :  1;
	uint8_t                        :  1;
	uint8_t                        :  3;
	uint16_t ct                    : 11;
	uint8_t                        :  5;
	uint16_t eq                    : 11;
	uint8_t                        :  5;
} __attribute__((packed));

struct c_nomatch_dma_cmd {
	struct c_cmd command;
	uint8_t index_ext              :  5;
	uint8_t                        :  3;
	uint8_t lac                    :  3;
	uint8_t event_send_disable     :  1;
	uint8_t event_success_disable  :  1;
	uint8_t event_ct_send          :  1;
	uint8_t event_ct_reply         :  1;
	uint8_t event_ct_ack           :  1;
	uint8_t event_ct_bytes         :  1;
	uint8_t get_with_local_flag    :  1;
	uint8_t restricted             :  1;
	uint8_t reduction              :  1;
	uint8_t flush                  :  1;
	uint8_t use_offset_for_get     :  1;
	uint8_t                        :  1;
	uint8_t                        :  1;
	union c_fab_addr dfa;
	uint64_t remote_offset         : 56;
	uint8_t unused_0;
	uint64_t local_addr            : 57;
	uint8_t                        :  7;
	uint32_t request_len;
	uint16_t ct                    : 11;
	uint8_t                        :  5;
	uint16_t eq                    : 11;
	uint8_t                        :  5;
};

struct c_full_dma_cmd {
	struct c_cmd command;
	uint8_t index_ext              :  5;
	uint8_t                        :  3;
	uint8_t lac                    :  3;
	uint8_t event_send_disable     :  1;
	uint8_t event_success_disable  :  1;
	uint8_t event_ct_send          :  1;
	uint8_t event_ct_reply         :  1;
	uint8_t event_ct_ack           :  1;
	uint8_t event_ct_bytes         :  1;
	uint8_t get_with_local_flag    :  1;
	uint8_t restricted             :  1;
	uint8_t reduction              :  1;
	uint8_t flush                  :  1;
	uint8_t use_offset_for_get     :  1;
	uint8_t                        :  1;
	uint8_t                        :  1;
	union c_fab_addr dfa;
	uint64_t remote_offset         : 56;
	uint8_t unused_0;
	uint64_t local_addr            : 57;
	uint8_t                        :  7;
	uint32_t request_len;
	uint16_t ct                    : 11;
	uint8_t                        :  5;
	uint16_t eq                    : 11;
	uint8_t                        :  5;
	uint32_t eager_length          : 20;
	uint8_t                        :  4;
	uint8_t rendezvous_id;
	uint32_t initiator;
	uint64_t match_bits;
	uint64_t header_data;
	uint64_t user_ptr;
};

struct c_dma_amo_cmd {
	struct c_cmd command;
	uint8_t index_ext              :  5;
	uint8_t                        :  3;
	uint8_t lac                    :  3;
	uint8_t event_send_disable     :  1;
	uint8_t event_success_disable  :  1;
	uint8_t event_ct_send          :  1;
	uint8_t event_ct_reply         :  1;
	uint8_t event_ct_ack           :  1;
	uint8_t event_ct_bytes         :  1;
	uint8_t get_with_local_flag    :  1;
	uint8_t restricted             :  1;
	uint8_t reduction              :  1;
	uint8_t flush                  :  1;
	uint8_t use_offset_for_get     :  1;
	uint8_t                        :  1;
	uint8_t                        :  1;
	union c_fab_addr dfa;
	uint64_t remote_offset         : 56;
	uint8_t unused_0;
	uint64_t local_read_addr       : 57;
	uint8_t                        :  7;
	uint32_t request_len;
	uint16_t ct                    : 11;
	uint8_t                        :  5;
	uint16_t eq                    : 11;
	uint8_t                        :  5;
	uint32_t unused_1;
	uint32_t initiator;
	uint64_t match_bits;
	uint64_t header_data;
	uint64_t user_ptr;
	uint64_t local_write_addr      : 57;
	uint8_t                        :  7;
	uint8_t atomic_op              :  4;
	uint8_t                        :  4;
	uint8_t atomic_type            :  4;
	uint8_t                        :  4;
	uint8_t cswap_op               :  3;
	uint8_t                        :  5;
	uint8_t unused_2;
	uint8_t write_lac              :  3;
	uint32_t                       : 29;
	uint64_t op2_word1;
	uint64_t op2_word2;
};

struct c_ct_cmd {
	struct c_cmd command;
	uint8_t unused_0;
	uint16_t ct               : 11;
	uint8_t                   :  5;
	uint16_t trig_ct          : 11;
	uint8_t                   :  5;
	uint16_t eq               : 11;
	uint8_t                   :  5;
	uint64_t ct_success       : 48;
	uint8_t ct_failure        :  7;
	uint16_t                  :  7;
	uint8_t set_ct_success    :  1;
	uint8_t set_ct_failure    :  1;
	uint64_t threshold        : 48;
	uint16_t unused_1;
	uint8_t embedded_dma[0];
} __attribute__((packed));

struct c_cq_cmd {
	struct c_cmd command;
	uint32_t               : 24;
	uint8_t lcid           :  4;
	uint8_t                :  4;
	uint32_t               : 24;
	uint64_t unused_0;
	uint64_t unused_1;
	uint64_t unused_2;
	uint64_t unused_3;
	uint64_t unused_4;
	uint64_t unused_5;
	uint64_t unused_6;
};

struct c_target_cmd {
	struct c_cmd command;
	uint8_t ptl_list                :  2;
	uint8_t                         :  6;
	uint16_t ptlte_index            : 11;
	uint8_t                         :  5;
	uint8_t event_ct_bytes          :  1;
	uint8_t event_ct_overflow       :  1;
	uint8_t event_ct_comm           :  1;
	uint8_t event_unlink_disable    :  1;
	uint8_t event_success_disable   :  1;
	uint8_t event_comm_disable      :  1;
	uint8_t event_link_disable      :  1;
	uint8_t unrestricted_end_ro     :  1;
	uint8_t unrestricted_body_ro    :  1;
	uint8_t unexpected_hdr_disable  :  1;
	uint8_t use_once                :  1;
	uint8_t no_truncate             :  1;
	uint8_t manage_local            :  1;
	uint8_t op_get                  :  1;
	uint8_t op_put                  :  1;
	uint8_t                         :  1;
	uint16_t ct                     : 11;
	uint8_t                         :  5;
	uint16_t buffer_id;
	uint8_t lac                     :  3;
	uint8_t                         :  5;
	uint8_t restart_seq             :  1;
	uint8_t                         :  7;
	uint32_t unused_0;
	uint64_t start                  : 57;
	uint8_t                         :  7;
	uint64_t length                 : 56;
	uint8_t unused_1;
	uint32_t min_free               : 24;
	uint8_t unused_2;
	uint32_t match_id;
	uint64_t match_bits;
	uint64_t ignore_bits;
	uint64_t unused_3;
};

struct c_set_state_cmd {
	struct c_cmd command;
	uint8_t ptlte_state    :  3;
	uint8_t                :  5;
	uint16_t ptlte_index   : 11;
	uint8_t                :  5;
	uint32_t unused_0;
	uint64_t current_addr  : 57;
	uint8_t                :  7;
	uint32_t drop_count    : 24;
	uint8_t unused_1;
	uint32_t unused_2;
	uint64_t unused_3;
	uint64_t unused_4;
	uint64_t unused_5;
	uint64_t unused_6;
	uint64_t unused_7;
};

struct c_eq_status {
	uint32_t timestamp_ns_cpy     : 30;
	uint64_t                      : 34;
	uint64_t timestamp_sec_cpy    : 48;
	uint8_t thld_sts              :  1;
	uint8_t                       :  3;
	uint8_t thld_id               :  2;
	uint8_t                       :  2;
	uint8_t unackd_dropped_event  :  1;
	uint8_t                       :  3;
	uint8_t event_drop_seq_no     :  1;
	uint8_t                       :  3;
	uint32_t event_slots_free     : 26;
	uint8_t                       :  6;
	uint32_t event_slots_rsrvd    : 26;
	uint8_t                       :  6;
	uint32_t wr_ptr               : 26;
	uint8_t                       :  6;
	uint8_t using_buffer_b        :  1;
	uint32_t                      : 31;
	uint64_t unused_0;
	uint64_t unused_1;
	uint32_t unused_2;
	uint32_t timestamp_ns         : 30;
	uint8_t                       :  2;
	uint64_t timestamp_sec        : 48;
	uint16_t unused_3;
};

struct c_event_initiator_short {
	uint64_t user_ptr;
	uint32_t mlength;
	uint8_t ptl_list     :  2;
	uint8_t              :  2;
	uint8_t rendezvous   :  1;
	uint8_t retry        :  1;
	uint8_t tgt_error    :  1;
	uint16_t             :  9;
	uint8_t event_size   :  2;
	uint8_t event_type   :  5;
	uint8_t              :  1;
	uint8_t return_code  :  6;
	uint8_t              :  2;
};

struct c_event_initiator_long {
	uint64_t user_ptr;
	uint32_t mlength;
	uint8_t ptl_list        :  2;
	uint8_t                 :  2;
	uint8_t rendezvous      :  1;
	uint8_t retry           :  1;
	uint8_t tgt_error       :  1;
	uint16_t                :  9;
	uint8_t event_size      :  2;
	uint8_t event_type      :  5;
	uint8_t                 :  1;
	uint8_t unused_0;
	uint32_t timestamp_ns   : 30;
	uint64_t                : 34;
	uint64_t timestamp_sec  : 48;
	uint8_t unused_1;
	uint8_t return_code     :  6;
	uint8_t                 :  2;
};

struct c_event_trig_op_short {
	uint64_t threshold   : 48;
	uint16_t unused_0;
	uint16_t ct_handle   : 11;
	uint64_t             : 37;
	uint8_t event_size   :  2;
	uint8_t event_type   :  5;
	uint8_t              :  1;
	uint8_t return_code  :  6;
	uint8_t              :  2;
};

struct c_event_trig_op_long {
	uint64_t threshold      : 48;
	uint16_t unused_0;
	uint16_t ct_handle      : 11;
	uint64_t                : 37;
	uint8_t event_size      :  2;
	uint8_t event_type      :  5;
	uint8_t                 :  1;
	uint8_t unused_1;
	uint32_t timestamp_ns   : 30;
	uint64_t                : 34;
	uint64_t timestamp_sec  : 48;
	uint8_t unused_2;
	uint8_t return_code     :  6;
	uint8_t                 :  2;
};

struct c_event_cmd_fail {
	uint16_t cq_id              : 10;
	uint8_t is_target           :  1;
	uint8_t                     :  5;
	uint16_t fail_loc;
	struct c_cmd fail_command;
	uint32_t                    : 24;
	uint64_t                    : 48;
	uint8_t event_size          :  2;
	uint8_t event_type          :  5;
	uint8_t                     :  1;
	uint8_t return_code         :  6;
	uint8_t                     :  2;
};

struct c_state_chg_fields {
	uint8_t ptlte_state  :  3;
	uint8_t              :  5;
	uint8_t sc_reason    :  4;
	uint8_t              :  4;
	uint8_t sc_nic_auto  :  1;
	uint16_t             : 15;
};

union c_tgt_ol_initiator {
	struct c_process initiator;
	struct c_state_chg_fields state_change;
};

struct c_event_target_long {
	uint64_t header_data;
	union c_tgt_ol_initiator initiator;
	uint16_t ptlte_index                 : 11;
	uint8_t                              :  5;
	uint8_t event_size                   :  2;
	uint8_t event_type                   :  5;
	uint8_t                              :  1;
	uint8_t ptl_list                     :  2;
	uint8_t rendezvous                   :  1;
	uint8_t get_issued                   :  1;
	uint8_t auto_unlinked                :  1;
	uint8_t                              :  3;
	uint64_t match_bits;
	uint16_t buffer_id;
	uint16_t vni;
	uint32_t timestamp_ns                : 30;
	uint8_t timestamp_sec_low            :  2;
	uint32_t mlength;
	uint32_t rlength;
	uint64_t remote_offset               : 56;
	uint8_t unused_0;
	uint64_t start                       : 57;
	uint8_t                              :  7;
	uint16_t lpe_stat_1;
	uint16_t lpe_stat_2;
	uint8_t atomic_type                  :  4;
	uint8_t atomic_op                    :  4;
	uint8_t cswap_op                     :  3;
	uint8_t                              :  5;
	uint8_t rendezvous_id;
	uint8_t return_code                  :  6;
	uint8_t                              :  2;
};

struct c_event_target_short {
	uint64_t header_data;
	uint32_t initiator;
	uint16_t ptlte_index   : 11;
	uint8_t                :  5;
	uint8_t event_size     :  2;
	uint8_t event_type     :  5;
	uint8_t                :  1;
	uint8_t ptl_list       :  2;
	uint8_t                :  2;
	uint8_t auto_unlinked  :  1;
	uint8_t                :  3;
	uint64_t match_bits;
	uint16_t buffer_id;
	uint16_t vni;
	uint32_t length        : 20;
	uint8_t                :  4;
	uint8_t return_code    :  6;
	uint8_t                :  2;
};

struct c_event_target_enet {
	uint16_t checksum;
	uint64_t pkt_cnt          : 46;
	uint8_t seg_cnt           :  2;
	uint32_t rss_hash_value;
	uint16_t ptlte_index      : 11;
	uint8_t                   :  5;
	uint8_t event_size        :  2;
	uint8_t event_type        :  5;
	uint8_t                   :  1;
	uint8_t ptl_list          :  2;
	uint8_t                   :  2;
	uint8_t auto_unlinked     :  1;
	uint8_t                   :  3;
	uint64_t start            : 57;
	uint8_t                   :  7;
	uint16_t buffer_id;
	uint8_t rss_hash_type     :  5;
	uint8_t                   :  3;
	uint8_t more_frags        :  1;
	uint8_t                   :  1;
	uint8_t is_roce           :  1;
	uint8_t roce_icrc_ok      :  1;
	uint8_t timestamp         :  1;
	uint8_t fmt               :  1;
	uint8_t parser_err        :  1;
	uint8_t                   :  1;
	uint16_t length;
	uint8_t unused_0;
	uint8_t return_code       :  6;
	uint8_t                   :  2;
};

struct c_event_enet_fgfc {
	uint32_t identifier;
	uint32_t unused_0;
	uint8_t fgfc_type     :  2;
	uint8_t               :  6;
	uint8_t xoff          :  1;
	uint8_t               :  7;
	uint8_t srcip_tag     :  2;
	uint8_t               :  6;
	uint8_t unused_1;
	uint16_t credits;
	uint8_t event_size    :  2;
	uint8_t event_type    :  5;
	uint8_t               :  1;
	uint8_t return_code   :  6;
	uint8_t               :  2;
};

struct c_event_pct {
	union c_fab_addr dfa;
	uint16_t seq_num            : 12;
	uint8_t                     :  4;
	uint16_t unused_0;
	uint8_t pct_event_type      :  3;
	uint8_t                     :  5;
	uint8_t unused_1;
	uint16_t spt_idx            : 11;
	uint8_t                     :  5;
	union c_conn_idx conn_idx;
	uint8_t event_size          :  2;
	uint8_t event_type          :  5;
	uint8_t                     :  1;
	uint8_t return_code         :  6;
	uint8_t                     :  2;
};

struct c_event_timestamp {
	uint32_t timestamp_ns   : 30;
	uint64_t                : 34;
	uint64_t timestamp_sec  : 48;
	uint8_t event_size      :  2;
	uint8_t event_type      :  5;
	uint8_t                 :  1;
	uint8_t return_code     :  6;
	uint8_t                 :  2;
};

struct c_event_eq_switch {
	uint64_t unused_0;
	uint64_t             : 48;
	uint8_t event_size   :  2;
	uint8_t event_type   :  5;
	uint8_t              :  1;
	uint8_t return_code  :  6;
	uint8_t              :  2;
};

struct c_cq_cntrs {
	uint64_t success_tx_cntr;
	uint64_t success_tgt_cntr;
	uint64_t fail_tx_cntr;
	uint64_t fail_tgt_cntr;
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t rarb_ll_err_cntr;
	uint64_t rarb_err_cntr;
	uint64_t wr_ptr_updt_err_cntr;
	uint64_t cntr0[3];
	uint64_t num_txq_cmd_reads[4];
	uint64_t num_tgq_cmd_reads[4];
	uint64_t num_tou_cmd_reads[4];
	uint64_t tx_waiting_on_read[4];
	uint64_t tgt_waiting_on_read[4];
	uint64_t num_tx_cmd_align_errors;
	uint64_t num_tx_cmd_op_errors;
	uint64_t num_tx_cmd_arg_errors;
	uint64_t num_tx_cmd_perm_errors;
	uint64_t num_tgt_cmd_arg_errors;
	uint64_t num_tgt_cmd_perm_errors;
	uint64_t cq_oxe_num_flits;
	uint64_t cq_oxe_num_stalls;
	uint64_t cq_oxe_num_idles;
	uint64_t cq_tou_num_flits;
	uint64_t cq_tou_num_stalls;
	uint64_t cq_tou_num_idles;
	uint64_t num_idc_cmds[4];
	uint64_t num_dma_cmds[4];
	uint64_t num_cq_cmds[4];
	uint64_t num_ll_cmds[4];
	uint64_t num_tgt_cmds[4];
	uint64_t dma_cmd_counts[16];
	uint64_t cq_cmd_counts[16];
	uint64_t cycles_blocked[4];
	uint64_t cntr2[12];
	uint64_t num_ll_ops_successful[4];
	uint64_t num_ll_ops_rejected[4];
	uint64_t num_ll_ops_split[4];
	uint64_t num_ll_ops_received[4];
};

struct c_tou_cntrs {
	uint64_t success_cntr;
	uint64_t fail_cntr;
	uint64_t cntr0[2];
	uint64_t cq_tou_num_cmds;
	uint64_t cq_tou_num_stalls;
	uint64_t tou_oxe_num_flits;
	uint64_t tou_oxe_num_stalls;
	uint64_t tou_oxe_num_idles;
	uint64_t list_rebuild_cycles;
	uint64_t cmd_fifo_full_cycles;
	uint64_t cntr1[7];
	uint64_t num_doorbell_writes;
	uint64_t num_ct_updates;
	uint64_t num_trig_cmds[4];
	uint64_t num_list_rebuilds[4];
	uint64_t ct_cmd_counts[16];
	uint64_t cntr2[20];
};

struct c1_oxe_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t err_no_translation;
	uint64_t err_invalid_ac;
	uint64_t err_page_perm;
	uint64_t err_ta_error;
	uint64_t err_page_req;
	uint64_t err_pci_ep;
	uint64_t err_pci_cmpl;
	uint64_t rsvd0;
	uint64_t stall_pct_bc[10];
	uint64_t stall_pbuf_bc[10];
	uint64_t stall_ts_no_out_crd_tsc[10];
	uint64_t stall_ts_no_in_crd_tsc[10];
	uint64_t stall_with_no_pci_tag;
	uint64_t stall_with_no_atu_tag;
	uint64_t stall_stfwd_eop;
	uint64_t stall_pcie_scoreboard;
	uint64_t stall_wr_conflict_pkt_buff_bnk[4];
	uint64_t stall_idc_no_buff_bc[10];
	uint64_t stall_idc_cmd_no_deq;
	uint64_t stall_non_idc_cmd_no_deq;
	uint64_t stall_fgfc_blk[4];
	uint64_t stall_fgfc_cntrl[4];
	uint64_t stall_fgfc_start[4];
	uint64_t stall_fgfc_end[4];
	uint64_t stall_hdr_arb;
	uint64_t stall_ioi_last_ordered;
	uint64_t rsvd1[3];
	uint64_t ignore_errs;
	uint64_t ioi_dma;
	uint64_t ioi_pkts_ordered;
	uint64_t ioi_pkts_unordered;
	uint64_t ptl_tx_put_msgs_tsc[10];
	uint64_t ptl_tx_get_msgs_tsc[10];
	uint64_t ptl_tx_put_pkts_tsc[10];
	uint64_t ptl_tx_mr_msgs;
	uint64_t num_hrp_cmds;
	uint64_t channel_idle;
	uint64_t mcu_meas[96];
	uint64_t prf_set0_st_ready;
	uint64_t prf_set0_st_pct_idx_wait;
	uint64_t prf_set0_st_pktbuf_req;
	uint64_t prf_set0_st_pktbuf_gnt;
	uint64_t prf_set0_st_header;
	uint64_t prf_set0_st_dma_mdr;
	uint64_t prf_set0_st_dma_upd;
	uint64_t prf_set0_pktbuf_na;
	uint64_t prf_set0_spt_na;
	uint64_t prf_set0_smt_na;
	uint64_t prf_set0_srb_na;
	uint64_t prf_set0_ts_select;
	uint64_t prf_set0_occ_hist_bin0;
	uint64_t prf_set0_occ_hist_bin1;
	uint64_t prf_set0_occ_hist_bin2;
	uint64_t prf_set0_occ_hist_bin3;
	uint64_t prf_set1_st_ready;
	uint64_t prf_set1_st_pct_idx_wait;
	uint64_t prf_set1_st_pktbuf_req;
	uint64_t prf_set1_st_pktbuf_gnt;
	uint64_t prf_set1_st_header;
	uint64_t prf_set1_st_dma_mdr;
	uint64_t prf_set1_st_dma_upd;
	uint64_t prf_set1_pktbuf_na;
	uint64_t prf_set1_spt_na;
	uint64_t prf_set1_smt_na;
	uint64_t prf_set1_srb_na;
	uint64_t prf_set1_ts_select;
	uint64_t prf_set1_occ_hist_bin0;
	uint64_t prf_set1_occ_hist_bin1;
	uint64_t prf_set1_occ_hist_bin2;
	uint64_t prf_set1_occ_hist_bin3;
};

struct c2_oxe_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t err_no_translation;
	uint64_t err_invalid_ac;
	uint64_t err_page_perm;
	uint64_t err_ta_error;
	uint64_t err_page_req;
	uint64_t err_pci_ep;
	uint64_t err_pci_cmpl;
	uint64_t rsvd0;
	uint64_t stall_pct_bc[10];
	uint64_t stall_pbuf_bc[10];
	uint64_t stall_ts_no_out_crd_tsc[10];
	uint64_t stall_ts_no_in_crd_tsc[10];
	uint64_t stall_with_no_pci_tag;
	uint64_t stall_with_no_atu_tag;
	uint64_t stall_stfwd_eop;
	uint64_t stall_pcie_scoreboard;
	uint64_t stall_wr_conflict_pkt_buff_bnk[4];
	uint64_t stall_idc_no_buff_bc[10];
	uint64_t stall_idc_cmd_no_deq;
	uint64_t stall_non_idc_cmd_no_deq;
	uint64_t stall_fgfc_blk[4];
	uint64_t stall_fgfc_cntrl[4];
	uint64_t stall_fgfc_start[4];
	uint64_t stall_fgfc_end[4];
	uint64_t stall_hdr_arb;
	uint64_t stall_ioi_last_ordered;
	uint64_t spt_cdt_errs;
	uint64_t smt_cdt_errs;
	uint64_t rsvd1;
	uint64_t ignore_errs;
	uint64_t ioi_dma;
	uint64_t ioi_pkts_ordered;
	uint64_t ioi_pkts_unordered;
	uint64_t ptl_tx_put_msgs_tsc[10];
	uint64_t ptl_tx_get_msgs_tsc[10];
	uint64_t ptl_tx_put_pkts_tsc[10];
	uint64_t ptl_tx_mr_msgs;
	uint64_t num_hrp_cmds;
	uint64_t channel_idle;
	uint64_t mcu_meas[96];
	uint64_t prf_set0_st_ready;
	uint64_t prf_set0_st_pct_idx_wait;
	uint64_t prf_set0_st_pktbuf_req;
	uint64_t prf_set0_st_pktbuf_gnt;
	uint64_t prf_set0_st_header;
	uint64_t prf_set0_st_dma_mdr;
	uint64_t prf_set0_st_dma_upd;
	uint64_t prf_set0_pktbuf_na;
	uint64_t prf_set0_spt_na;
	uint64_t prf_set0_smt_na;
	uint64_t prf_set0_srb_na;
	uint64_t prf_set0_ts_select;
	uint64_t prf_set0_occ_hist_bin0;
	uint64_t prf_set0_occ_hist_bin1;
	uint64_t prf_set0_occ_hist_bin2;
	uint64_t prf_set0_occ_hist_bin3;
	uint64_t prf_set1_st_ready;
	uint64_t prf_set1_st_pct_idx_wait;
	uint64_t prf_set1_st_pktbuf_req;
	uint64_t prf_set1_st_pktbuf_gnt;
	uint64_t prf_set1_st_header;
	uint64_t prf_set1_st_dma_mdr;
	uint64_t prf_set1_st_dma_upd;
	uint64_t prf_set1_pktbuf_na;
	uint64_t prf_set1_spt_na;
	uint64_t prf_set1_smt_na;
	uint64_t prf_set1_srb_na;
	uint64_t prf_set1_ts_select;
	uint64_t prf_set1_occ_hist_bin0;
	uint64_t prf_set1_occ_hist_bin1;
	uint64_t prf_set1_occ_hist_bin2;
	uint64_t prf_set1_occ_hist_bin3;
};

struct c_ixe_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t cntr1[2];
	uint64_t port_dfa_mismatch;
	uint64_t hdr_checksum_errors;
	uint64_t ipv4_checksum_errors;
	uint64_t hrp_req_errors;
	uint64_t ip_options_errors;
	uint64_t get_len_errors;
	uint64_t roce_icrc_error;
	uint64_t parser_par_errors;
	uint64_t pbuf_rd_errors;
	uint64_t hdr_ecc_errors;
	uint64_t cntr2[2];
	uint64_t rx_udp_pkt;
	uint64_t rx_tcp_pkt;
	uint64_t rx_ipv4_pkt;
	uint64_t rx_ipv6_pkt;
	uint64_t rx_roce_pkt;
	uint64_t rx_ptl_gen_pkt;
	uint64_t rx_ptl_sml_pkt;
	uint64_t rx_ptl_unrestricted_pkt;
	uint64_t rx_ptl_smallmsg_pkt;
	uint64_t rx_ptl_continuation_pkt;
	uint64_t rx_ptl_restricted_pkt;
	uint64_t rx_ptl_connmgmt_pkt;
	uint64_t rx_ptl_response_pkt;
	uint64_t rx_unrecognized_pkt;
	uint64_t rx_ptl_sml_amo_pkt;
	uint64_t rx_ptl_msgs;
	uint64_t rx_ptl_multi_msgs;
	uint64_t rx_ptl_mr_msgs;
	uint64_t rx_pkt_drop_pct;
	uint64_t rx_pkt_drop_rmu_norsp;
	uint64_t rx_pkt_drop_rmu_wrsp;
	uint64_t rx_pkt_drop_ixe_parser;
	uint64_t rx_pkt_ipv4_options;
	uint64_t rx_pkt_ipv6_options;
	uint64_t rx_eth_seg;
	uint64_t rx_roce_seg;
	uint64_t rx_roce_spseg;
	uint64_t cntr3[21];
	uint64_t pool_ecn_pkts[4];
	uint64_t pool_no_ecn_pkts[4];
	uint64_t tc_req_ecn_pkts[8];
	uint64_t tc_req_no_ecn_pkts[8];
	uint64_t tc_rsp_ecn_pkts[8];
	uint64_t tc_rsp_no_ecn_pkts[8];
	uint64_t disp_lpe_puts;
	uint64_t disp_lpe_puts_ok;
	uint64_t disp_lpe_amos;
	uint64_t disp_lpe_amos_ok;
	uint64_t disp_mst_puts;
	uint64_t disp_mst_puts_ok;
	uint64_t disp_lpe_gets;
	uint64_t disp_lpe_gets_ok;
	uint64_t disp_mst_gets;
	uint64_t disp_mst_gets_ok;
	uint64_t disp_rpu_resps;
	uint64_t disp_rpu_err_reqs;
	uint64_t disp_dmawr_reqs;
	uint64_t disp_dmawr_resps;
	uint64_t disp_dmawr_resps_ok;
	uint64_t disp_oxe_resps;
	uint64_t disp_pct_getcomp;
	uint64_t disp_atu_reqs;
	uint64_t disp_atu_resps;
	uint64_t disp_atu_resps_ok;
	uint64_t disp_eth_events;
	uint64_t disp_put_events;
	uint64_t disp_amo_events;
	uint64_t disp_wr_conflicts;
	uint64_t disp_amo_conflicts;
	uint64_t disp_stall_conflict;
	uint64_t disp_stall_resp_fifo;
	uint64_t disp_stall_err_fifo;
	uint64_t disp_stall_atu_cdt;
	uint64_t disp_stall_atu_fifo;
	uint64_t disp_stall_gcomp_fifo;
	uint64_t disp_stall_mdid_cdts;
	uint64_t disp_stall_mst_match_fifo;
	uint64_t disp_stall_grr_id;
	uint64_t disp_stall_put_resp;
	uint64_t disp_atu_flush_reqs;
	uint64_t cntr4[4];
	uint64_t dmawr_stall_p_cdt;
	uint64_t dmawr_stall_np_cdt;
	uint64_t dmawr_stall_np_req_cnt;
	uint64_t dmawr_stall_ftch_amo_cnt;
	uint64_t dmawr_p_pass_np_cnt;
	uint64_t dmawr_write_reqs;
	uint64_t dmawr_nic_amo_reqs;
	uint64_t dmawr_cpu_amo_reqs;
	uint64_t dmawr_cpu_ftch_amo_reqs;
	uint64_t dmawr_flush_reqs;
	uint64_t dmawr_req_no_write_reqs;
	uint64_t dmawr_amo_ex_invalid;
	uint64_t dmawr_amo_ex_overflow;
	uint64_t dmawr_amo_ex_underflow;
	uint64_t dmawr_amo_ex_inexact;
	uint64_t dmawr_amo_misaligned;
	uint64_t dmawr_amo_invalid_op;
	uint64_t dmawr_amo_len_err;
	uint64_t dmawr_pcie_unsuccess_cmpl;
	uint64_t dmawr_pcie_err_poisoned;
	uint64_t cntr5[4];
};

struct c_pct_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t req_ordered;
	uint64_t req_unordered;
	uint64_t req_no_response;
	uint64_t eth_packets;
	uint64_t optip_packets;
	uint64_t ptls_response;
	uint64_t responses_received;
	uint64_t hrp_responses_received;
	uint64_t hrp_rsp_discard_received;
	uint64_t hrp_rsp_err_received;
	uint64_t conn_sct_open;
	uint64_t conn_tct_open;
	uint64_t mst_hit_on_som;
	uint64_t trs_hit_on_req;
	uint64_t cls_req_miss_tct;
	uint64_t clear_sent;
	uint64_t close_sent;
	uint64_t accel_close;
	uint64_t clear_close_drop;
	uint64_t req_src_error;
	uint64_t bad_seq_nacks;
	uint64_t no_tct_nacks;
	uint64_t no_mst_nacks;
	uint64_t no_trs_nacks;
	uint64_t no_matching_tct;
	uint64_t resource_busy;
	uint64_t err_no_matching_trs;
	uint64_t err_no_matching_mst;
	uint64_t spt_timeouts;
	uint64_t sct_timeouts;
	uint64_t tct_timeouts;
	uint64_t retry_srb_requests;
	uint64_t retry_trs_put;
	uint64_t retry_mst_get;
	uint64_t clr_cls_stalls;
	uint64_t close_rsp_drops;
	uint64_t trs_rsp_nack_drops;
	uint64_t req_blocked_closing;
	uint64_t req_blocked_clearing;
	uint64_t req_blocked_retry;
	uint64_t rsp_err_rcvd;
	uint64_t rsp_dropped_timeout;
	uint64_t rsp_dropped_try;
	uint64_t rsp_dropped_cls_try;
	uint64_t rsp_dropped_inactive;
	uint64_t pct_hni_flits;
	uint64_t pct_hni_stalls;
	uint64_t pct_ee_events;
	uint64_t pct_ee_stalls;
	uint64_t pct_cq_notifications;
	uint64_t pct_cq_stalls;
	uint64_t pct_mst_cmd0;
	uint64_t pct_mst_stalls0;
	uint64_t pct_mst_cmd1;
	uint64_t pct_mst_stalls1;
	uint64_t sct_stall_state;
	uint64_t tgt_cls_abort;
	uint64_t cls_req_bad_seqno_drops;
	uint64_t req_tct_tmout_drops;
	uint64_t trs_replay_pend_drops;
	uint64_t cntr_rsvd_1[2];
	uint64_t req_rsp_latency[32];
	uint64_t host_access_latency[16];
	uint64_t cntr_rsvd_2[16];
};

struct c_rmu_cntrs {
	uint64_t ptl_success_cntr;
	uint64_t enet_success_cntr;
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t ptl_invld_dfa_cntr;
	uint64_t ptl_invld_vni_cntr;
	uint64_t ptl_no_ptlte_cntr;
	uint64_t enet_frame_rejected_cntr;
	uint64_t enet_ptlte_set_ctrl_err_cntr;
	uint64_t cntr_rsvd[7];
};

struct c_lpe_cntrs {
	uint64_t success_cntr;
	uint64_t fail_cntr;
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t cntr0;
	uint64_t put_cmds;
	uint64_t rendezvous_put_cmds;
	uint64_t get_cmds;
	uint64_t amo_cmds;
	uint64_t famo_cmds;
	uint64_t cntr1[5];
	uint64_t err_entry_not_found_ethernet;
	uint64_t err_pt_disabled_eq_full;
	uint64_t err_pt_disabled_req_empty;
	uint64_t err_pt_disabled_cant_alloc_unexpected;
	uint64_t err_pt_disabled_no_match_in_overflow;
	uint64_t err_eq_disabled;
	uint64_t err_op_violation;
	uint64_t err_entry_not_found_unlink_failed;
	uint64_t err_entry_not_found_eq_full;
	uint64_t err_entry_not_found_req_empty;
	uint64_t err_entry_not_found_no_match_in_overflow;
	uint64_t err_entry_not_found_cant_alloc_unexpected;
	uint64_t err_invalid_endpoint;
	uint64_t err_no_space_append;
	uint64_t err_no_space_net;
	uint64_t err_search_no_match;
	uint64_t err_setstate_no_match;
	uint64_t err_src_error;
	uint64_t err_ptlte_sw_managed;
	uint64_t err_illegal_op;
	uint64_t err_restricted_unicast;
	uint64_t cntr2[5];
	uint64_t event_put_overflow;
	uint64_t cntr3;
	uint64_t event_get_overflow;
	uint64_t cntr4;
	uint64_t event_atomic_overflow;
	uint64_t plec_frees_csr;
	uint64_t event_fetch_atomic_overflow;
	uint64_t event_search;
	uint64_t event_rendezvous;
	uint64_t event_link;
	uint64_t event_unlink;
	uint64_t event_statechange;
	uint64_t plec_allocs;
	uint64_t plec_frees;
	uint64_t plec_hits;
	uint64_t cyc_no_rdy_cdts;
	uint64_t cyc_rrq_blocked[4];
	uint64_t cyc_blocked_lossless;
	uint64_t cyc_blocked_ixe_put;
	uint64_t cyc_blocked_ixe_get;
	uint64_t search_cmds;
	uint64_t search_success;
	uint64_t search_fail;
	uint64_t search_delete_cmds;
	uint64_t search_delete_success;
	uint64_t search_delete_fail;
	uint64_t unlink_cmds;
	uint64_t unlink_success;
	uint64_t net_match_requests[4];
	uint64_t net_match_success[4];
	uint64_t net_match_useonce[4];
	uint64_t net_match_local[4];
	uint64_t net_match_priority[4];
	uint64_t net_match_overflow[4];
	uint64_t net_match_request[4];
	uint64_t append_cmds[4];
	uint64_t append_success[4];
	uint64_t setstate_cmds[4];
	uint64_t setstate_success[4];
	uint64_t search_nid_any[4];
	uint64_t search_pid_any[4];
	uint64_t search_rank_any[4];
	uint64_t rndzv_puts[4];
	uint64_t rndzv_puts_offloaded[4];
	uint64_t num_truncated[4];
	uint64_t unexpected_get_amo[4];
	uint64_t cntr_rsvd[16];
};

struct c_mst_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t allocations;
	uint64_t requests;
	uint64_t stalled_waiting_for_lpe;
	uint64_t stalled_waiting_get_crdts;
	uint64_t stalled_waiting_put_crdts;
	uint64_t stalled_waiting_ee_crdts;
	uint64_t cntr1[8];
	uint64_t err_cancelled;
	uint64_t cntr2[15];
};

struct c_ee_cntrs {
	uint64_t events_enqueued_cntr[4];
	uint64_t events_dropped_rsrvn_cntr[4];
	uint64_t events_dropped_fc_sc_cntr[4];
	uint64_t events_dropped_ordinary_cntr[4];
	uint64_t eq_status_update_cntr[4];
	uint64_t cbs_written_cntr[4];
	uint64_t partial_cbs_written_cntr[4];
	uint64_t expired_cbs_written_cntr[4];
	uint64_t eq_buffer_switch_cntr[4];
	uint64_t deferred_eq_switch_cntr[4];
	uint64_t addr_trans_prefetch_cntr[4];
	uint64_t eq_sw_state_wr_cntr[4];
	uint64_t lpe_event_req_cntr;
	uint64_t lpe_fe_cntr;
	uint64_t lpe_dd_cntr;
	uint64_t lpe_ce_cntr;
	uint64_t lpe_fe_stall_cntr;
	uint64_t lpe_ce_stall_cntr;
	uint64_t ixe_event_req_cntr;
	uint64_t ixe_fe_cntr;
	uint64_t ixe_dd_cntr;
	uint64_t ixe_ce_cntr;
	uint64_t ixe_fe_stall_cntr;
	uint64_t ixe_ce_stall_cntr;
	uint64_t mst_event_req_cntr;
	uint64_t mst_fe_cntr;
	uint64_t mst_dd_cntr;
	uint64_t mst_ce_cntr;
	uint64_t mst_fe_stall_cntr;
	uint64_t mst_ce_stall_cntr;
	uint64_t pct_event_req_cntr;
	uint64_t pct_fe_cntr;
	uint64_t pct_dd_cntr;
	uint64_t pct_ce_cntr;
	uint64_t pct_fe_stall_cntr;
	uint64_t pct_ce_stall_cntr;
	uint64_t hni_event_req_cntr;
	uint64_t hni_fe_cntr;
	uint64_t hni_fe_stall_cntr;
	uint64_t cq_event_req_cntr;
	uint64_t cq_fe_cntr;
	uint64_t cq_fe_stall_cntr;
	uint64_t ts_fe_cntr;
	uint64_t ts_fe_stall_cntr;
	uint64_t fe_arb_out_event_cntr;
	uint64_t fe_arb_out_stall_cntr;
	uint64_t ce_arb_out_event_cntr;
	uint64_t ce_arb_out_stall_cntr;
	uint64_t lpe_query_eq_none_cntr;
	uint64_t eqs_lpe_query_req_cntr;
	uint64_t eqs_csr_req_cntr;
	uint64_t eqs_csr_stall_cntr;
	uint64_t eqs_hws_init_req_cntr;
	uint64_t eqs_hws_init_stall_cntr;
	uint64_t eqs_expired_cb_req_cntr;
	uint64_t eqs_expired_cb_stall_cntr;
	uint64_t eqs_sts_updt_req_cntr;
	uint64_t eqs_sts_updt_stall_cntr;
	uint64_t eqs_sws_wr_req_cntr;
	uint64_t eqs_sws_wr_stall_cntr;
	uint64_t eqs_event_req_cntr;
	uint64_t eqs_event_stall_cntr;
	uint64_t eqs_arb_out_req_cntr;
	uint64_t eqs_free_cb_stall_cntr;
	uint64_t addr_trans_req_cntr;
	uint64_t tarb_wr_req_cntr;
	uint64_t tarb_irq_req_cntr;
	uint64_t tarb_stall_cntr;
	uint64_t event_type_cntr[32];
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t eq_dsabld_event_err_cntr;
	uint64_t eq_dsabld_sws_err_cntr;
	uint64_t eq_dsabld_lpeq_err_cntr;
	uint64_t eq_rsrvn_uflw_err_cntr;
	uint64_t unxpctd_trnsltn_rsp_err_cntr;
	uint64_t rarb_hw_err_cntr;
	uint64_t rarb_sw_err_cntr;
	uint64_t tarb_err_cntr;
	uint64_t eq_state_ucor_err_cntr;
	uint64_t cntr_rsvd[13];
};

struct c_atu_cntrs {
	uint64_t mem_cor_err_cntr;
	uint64_t mem_ucor_err_cntr;
	uint64_t cache_miss[4];
	uint64_t cache_hit_base_page_size[4];
	uint64_t cache_hit_derivative1_page_size[4];
	uint64_t cache_hit_derivative2_page_size[4];
	uint64_t cache_miss_oxe;
	uint64_t cache_miss_ixe;
	uint64_t cache_miss_ee;
	uint64_t ats_trans_latency[4];
	uint64_t nta_trans_latency[4];
	uint64_t client_req_oxe;
	uint64_t client_req_ixe;
	uint64_t client_req_ee;
	uint64_t filtered_requests;
	uint64_t ats_prs_odp_latency[4];
	uint64_t nic_pri_odp_latency[4];
	uint64_t odp_requests[4];
	uint64_t client_rsp_not_ok[4];
	uint64_t cache_evictions;
	uint64_t ats_inval_cntr;
	uint64_t implicit_flr_inval_cntr;
	uint64_t implicit_ats_en_inval_cntr;
	uint64_t atucq_inval_cntr;
	uint64_t pcie_unsuccess_cmpl;
	uint64_t ats_trans_err;
	uint64_t pcie_err_poisoned;
	uint64_t ats_dynamic_state_change_cntr;
	uint64_t at_stall_ccp;
	uint64_t at_stall_no_ptid;
	uint64_t at_stall_np_cdts;
	uint64_t at_stall_tarb_arb;
	uint64_t inval_stall_arb;
	uint64_t inval_stall_cmpl_wait;
};

struct c_parbs_cntrs {
	uint64_t tarb_cq_posted_pkts;
	uint64_t tarb_cq_non_posted_pkts;
	uint64_t tarb_cq_sbe_cnt;
	uint64_t tarb_cq_mbe_cnt;
	uint64_t tarb_ee_posted_pkts;
	uint64_t tarb_ee_sbe_cnt;
	uint64_t tarb_ee_mbe_cnt;
	uint64_t tarb_ixe_posted_pkts;
	uint64_t tarb_ixe_non_posted_pkts;
	uint64_t tarb_ixe_sbe_cnt;
	uint64_t tarb_ixe_mbe_cnt;
	uint64_t tarb_oxe_non_posted_pkts;
	uint64_t tarb_oxe_sbe_cnt;
	uint64_t tarb_oxe_mbe_cnt;
	uint64_t tarb_atu_posted_pkts;
	uint64_t tarb_atu_non_posted_pkts;
	uint64_t tarb_atu_sbe_cnt;
	uint64_t tarb_atu_mbe_cnt;
	uint64_t tarb_pi_posted_pkts;
	uint64_t tarb_pi_non_posted_pkts;
	uint64_t tarb_pi_posted_blocked_cnt;
	uint64_t tarb_pi_non_posted_blocked_cnt;
	uint64_t tarb_cq_posted_fifo_sbe;
	uint64_t tarb_cq_non_posted_fifo_sbe;
	uint64_t tarb_ee_posted_fifo_sbe;
	uint64_t tarb_ixe_posted_fifo_sbe;
	uint64_t tarb_ixe_non_posted_fifo_sbe;
	uint64_t tarb_oxe_non_posted_fifo_sbe;
	uint64_t tarb_atu_posted_fifo_sbe;
	uint64_t tarb_atu_non_posted_fifo_sbe;
	uint64_t tarb_cq_posted_fifo_mbe;
	uint64_t tarb_cq_non_posted_fifo_mbe;
	uint64_t tarb_ee_posted_fifo_mbe;
	uint64_t tarb_ixe_posted_fifo_mbe;
	uint64_t tarb_ixe_non_posted_fifo_mbe;
	uint64_t tarb_oxe_non_posted_fifo_mbe;
	uint64_t tarb_atu_posted_fifo_mbe;
	uint64_t tarb_atu_non_posted_fifo_mbe;
	uint64_t tarb_reserved_cnts[10];
	uint64_t rarb_pi_posted_pkts;
	uint64_t rarb_pi_completion_pkts;
	uint64_t rarb_pi_sbe_cnt;
	uint64_t rarb_pi_mbe_cnt;
	uint64_t rarb_cq_posted_pkts;
	uint64_t rarb_cq_completion_pkts;
	uint64_t rarb_ee_posted_pkts;
	uint64_t rarb_ixe_posted_pkts;
	uint64_t rarb_ixe_completion_pkts;
	uint64_t rarb_oxe_completion_pkts;
	uint64_t rarb_atu_posted_pkts;
	uint64_t rarb_atu_completion_pkts;
	uint64_t rarb_reserved_cnts[4];
};

struct c_pi_cntrs {
	uint64_t pri_mb_rtrgt_tlps;
	uint64_t pri_mb_rtrgt_mwr_tlps;
	uint64_t pri_mb_rtrgt_mrd_tlps;
	uint64_t pri_mb_rtrgt_tlp_discards;
	uint64_t pri_mb_rtrgt_posted_tlp_discards;
	uint64_t pri_mb_rtrgt_non_posted_tlp_discards;
	uint64_t pri_mb_rtrgt_posted_tlp_partial_discards;
	uint64_t pri_mb_rtrgt_fifo_cor_errors;
	uint64_t pri_mb_rtrgt_fifo_ucor_errors;
	uint64_t pri_mb_xali_cmpl_tlps;
	uint64_t pri_mb_xali_cmpl_ca_tlps;
	uint64_t pri_mb_xali_cmpl_ur_tlps;
	uint64_t pri_mb_axi_wr_requests;
	uint64_t pri_mb_axi_rd_requests;
	uint64_t pri_mb_axi_wr_rsp_decerrs;
	uint64_t pri_mb_axi_wr_rsp_slverrs;
	uint64_t pri_mb_axi_rd_rsp_decerrs;
	uint64_t pri_mb_axi_rd_rsp_slverrs;
	uint64_t pri_mb_axi_rd_rsp_parerrs;
	uint64_t pri_rarb_rtrgt_tlps;
	uint64_t pri_rarb_rtrgt_mwr_tlps;
	uint64_t pri_rarb_rtrgt_msg_tlps;
	uint64_t pri_rarb_rtrgt_msgd_tlps;
	uint64_t pri_rarb_rtrgt_tlp_discards;
	uint64_t pri_rarb_rtrgt_fifo_cor_errors;
	uint64_t pri_rarb_rtrgt_fifo_ucor_errors;
	uint64_t pri_rarb_rbyp_tlps;
	uint64_t pri_rarb_rbyp_tlp_discards;
	uint64_t pri_rarb_rbyp_fifo_cor_errors;
	uint64_t pri_rarb_rbyp_fifo_ucor_errors;
	uint64_t pri_rarb_sriovt_cor_errors;
	uint64_t pri_rarb_sriovt_ucor_errors;
	uint64_t pti_tarb_pkts;
	uint64_t pti_tarb_pkt_discards;
	uint64_t pti_tarb_rarb_ca_pkts;
	uint64_t pti_tarb_mwr_pkts;
	uint64_t pti_tarb_mrd_pkts;
	uint64_t pti_tarb_flush_pkts;
	uint64_t pti_tarb_irq_pkts;
	uint64_t pti_tarb_translation_pkts;
	uint64_t pti_tarb_inv_rsp_pkts;
	uint64_t pti_tarb_pg_req_pkts;
	uint64_t pti_tarb_amo_pkts;
	uint64_t pti_tarb_fetching_amo_pkts;
	uint64_t pti_tarb_hdr_cor_errors;
	uint64_t pti_tarb_hdr_ucor_errors;
	uint64_t pti_tarb_data_cor_errors;
	uint64_t pti_tarb_data_ucor_error;
	uint64_t pti_tarb_acxt_cor_errors;
	uint64_t pti_tarb_acxt_ucor_errors;
	uint64_t pti_tarb_reserved[5];
	uint64_t dmac_axi_rd_requests;
	uint64_t dmac_axi_rd_rsp_decerrs;
	uint64_t dmac_axi_rd_rsp_slverrs;
	uint64_t dmac_axi_rd_rsp_parerrs;
	uint64_t dmac_payload_mwr_tlps;
	uint64_t dmac_ce_mwr_tlps;
	uint64_t dmac_irqs;
	uint64_t dmac_desc_cor_errors;
	uint64_t dmac_desc_ucor_errors;
};

struct c_pi_ipd_cntrs {
	uint64_t pri_rtrgt_tlps;
	uint64_t pri_rtrgt_mwr_tlps;
	uint64_t pri_rtrgt_mrd_tlps;
	uint64_t pri_rtrgt_msg_tlps;
	uint64_t pri_rtrgt_msgd_tlps;
	uint64_t pri_rtrgt_tlps_aborted;
	uint64_t pri_rtrgt_blocked_on_mb;
	uint64_t pri_rtrgt_blocked_on_rarb;
	uint64_t pri_rtrgt_hdr_parity_errors;
	uint64_t pri_rtrgt_data_parity_errors;
	uint64_t pri_rbyp_tlps;
	uint64_t pri_rbyp_tlps_aborted;
	uint64_t pri_rbyp_hdr_parity_errors;
	uint64_t pri_rbyp_data_parity_errors;
	uint64_t pti_tarb_xali_posted_tlps;
	uint64_t pti_tarb_xali_non_posted_tlps;
	uint64_t pti_msixc_xali_posted_tlps;
	uint64_t pti_dbg_xali_posted_tlps;
	uint64_t pti_dbg_xali_non_posted_tlps;
	uint64_t pti_dmac_xali_posted_tlps;
	uint64_t pti_tarb_p_fifo_cor_errors;
	uint64_t pti_tarb_p_fifo_ucor_errors;
	uint64_t pti_tarb_np_fifo_cor_errors;
	uint64_t pti_tarb_np_fifo_ucor_errors;
	uint64_t pti_tarb_blocked_on_ph_crd;
	uint64_t pti_tarb_blocked_on_pd_crd;
	uint64_t pti_tarb_blocked_on_nph_crd;
	uint64_t pti_tarb_blocked_on_npd_crd;
	uint64_t pti_tarb_blocked_on_tag;
	uint64_t pti_dmac_blocked_on_ph_crd;
	uint64_t pti_dmac_blocked_on_pd_crd;
	uint64_t pti_msixc_blocked_on_ph_crd;
	uint64_t pti_msixc_blocked_on_pd_crd;
	uint64_t pti_dbg_blocked_on_ph_crd;
	uint64_t pti_dbg_blocked_on_pd_crd;
	uint64_t pti_dbg_blocked_on_nph_crd;
	uint64_t pti_dbg_blocked_on_tag;
	uint64_t pti_tarb_cmpl_to;
	uint64_t pti_tarb_cmpl_to_fifo_cor_errors;
	uint64_t pti_tarb_cmpl_to_fifo_ucor_errors;
	uint64_t msixc_oob_irqs_sent;
	uint64_t msixc_oob_legacy_irqs_sent;
	uint64_t msixc_oob_irqs_discarded;
	uint64_t msixc_oob_irq_pbas;
	uint64_t msixc_ib_irqs_sent;
	uint64_t msixc_ib_legacy_irqs_sent;
	uint64_t msixc_ib_irqs_discarded;
	uint64_t msixc_ib_irq_pbas;
	uint64_t msixc_pba_cor_errors;
	uint64_t msixc_pba_ucor_errors;
	uint64_t msixc_table_cor_errors;
	uint64_t msixc_table_ucor_errors;
	uint64_t pti_msixc_fifo_cor_errors;
	uint64_t pti_msixc_fifo_ucor_errors;
	uint64_t msixc_pti_fifo_cor_errors;
	uint64_t msixc_pti_fifo_ucor_errors;
	uint64_t dmac_p_fifo_cor_errors;
	uint64_t dmac_p_fifo_ucor_errors;
	uint64_t dbic_dbi_requests;
	uint64_t dbic_dbi_acks;
	uint64_t ipd_trigger_events[2];
	uint64_t ipd_cnts_reserved[66];
};

struct c_mb_cntrs {
	uint64_t crmc_ring_sbe[3];
	uint64_t crmc_ring_mbe[3];
	uint64_t crmc_wr_error[3];
	uint64_t crmc_axi_wr_requests[3];
	uint64_t crmc_ring_wr_requests[3];
	uint64_t crmc_rd_error[3];
	uint64_t crmc_axi_rd_requests[3];
	uint64_t crmc_ring_rd_requests[3];
	uint64_t jai_axi_wr_requests;
	uint64_t jai_axi_rd_requests;
	uint64_t mb_lsa0_trigger_events[2];
	uint64_t ixe_lsa0_trigger_events[2];
	uint64_t cq_lsa0_trigger_events[2];
	uint64_t ee_lsa0_trigger_events[2];
	uint64_t mb_lsa1_trigger_events[2];
	uint64_t ixe_lsa1_trigger_events[2];
	uint64_t cq_lsa1_trigger_events[2];
	uint64_t ee_lsa1_trigger_events[2];
	uint64_t cmc_axi_wr_requests[4];
	uint64_t cmc_axi_rd_requests[4];
	uint64_t mb_todo_cntrs[14];
};

#ifdef __LITTLE_ENDIAN
struct c_port_fab_hdr {
	uint8_t ihl            :  4;
	uint8_t ver            :  4;
	uint8_t ecn            :  2;
	uint8_t dscp           :  6;
	uint16_t tl;
	uint16_t vni;
	uint8_t spt_idx_hi     :  6;
	uint8_t                :  2;
	uint8_t try_           :  3;
	uint8_t spt_idx_low    :  5;
	uint8_t ttl;
	uint8_t protocol;
	uint16_t hdr_chksum;
	union c_fab_addr sfa;
	union c_fab_addr dfa;
} __attribute__((packed));
#else
struct c_port_fab_hdr {
	uint8_t ver            :  4;
	uint8_t ihl            :  4;
	uint8_t dscp           :  6;
	uint8_t ecn            :  2;
	uint16_t tl;
	uint16_t vni;
	uint8_t                :  2;
	uint16_t spt_idx       : 11;
	uint8_t try_           :  3;
	uint8_t ttl;
	uint8_t protocol;
	uint16_t hdr_chksum;
	union c_fab_addr sfa;
	union c_fab_addr dfa;
} __attribute__((packed));
#endif

static inline void c_port_fab_hdr_set_tl(struct c_port_fab_hdr *p, uint16_t tl)
{
	p->tl = cpu_to_be16(tl);
}

static inline uint16_t c_port_fab_hdr_get_tl(const struct c_port_fab_hdr *p)
{
	return be16_to_cpu(p->tl);
}

static inline void c_port_fab_hdr_set_vni(struct c_port_fab_hdr *p, uint16_t vni)
{
	p->vni = cpu_to_be16(vni);
}

static inline uint16_t c_port_fab_hdr_get_vni(const struct c_port_fab_hdr *p)
{
	return be16_to_cpu(p->vni);
}

static inline void c_port_fab_hdr_set_spt_idx(struct c_port_fab_hdr *p, uint16_t spt_idx)
{
#ifdef __LITTLE_ENDIAN
	p->spt_idx_low = spt_idx;
	p->spt_idx_hi = spt_idx >> 5;
#else
	p->spt_idx = spt_idx;
#endif
}

static inline uint16_t c_port_fab_hdr_get_spt_idx(const struct c_port_fab_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->spt_idx_low | (uint16_t)p->spt_idx_hi << 5;
#else
	return p->spt_idx;
#endif
}

static inline void c_port_fab_hdr_set_hdr_chksum(struct c_port_fab_hdr *p, uint16_t hdr_chksum)
{
	p->hdr_chksum = cpu_to_be16(hdr_chksum);
}

static inline uint16_t c_port_fab_hdr_get_hdr_chksum(const struct c_port_fab_hdr *p)
{
	return be16_to_cpu(p->hdr_chksum);
}

#ifdef __LITTLE_ENDIAN
struct c_port_fab_vs_hdr {
	uint8_t dscp_hi        :  4;
	uint8_t ver            :  4;
	uint8_t                :  2;
	uint8_t                :  4;
	uint8_t dscp_low       :  2;
	uint8_t spt_idx_hi;
	uint8_t                :  2;
	uint8_t try_           :  3;
	uint8_t spt_idx_low    :  3;
	uint16_t vni;
	union c_fab_addr sfa;
	union c_fab_addr dfa;
} __attribute__((packed));
#else
struct c_port_fab_vs_hdr {
	uint8_t ver            :  4;
	uint16_t dscp          :  6;
	uint8_t                :  4;
	uint8_t                :  2;
	uint16_t spt_idx       : 11;
	uint8_t try_           :  3;
	uint8_t                :  2;
	uint16_t vni;
	union c_fab_addr sfa;
	union c_fab_addr dfa;
} __attribute__((packed));
#endif

static inline void c_port_fab_vs_hdr_set_dscp(struct c_port_fab_vs_hdr *p, uint8_t dscp)
{
#ifdef __LITTLE_ENDIAN
	p->dscp_low = dscp;
	p->dscp_hi = dscp >> 2;
#else
	p->dscp = dscp;
#endif
}

static inline uint8_t c_port_fab_vs_hdr_get_dscp(const struct c_port_fab_vs_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint8_t)p->dscp_low | (uint8_t)p->dscp_hi << 2;
#else
	return p->dscp;
#endif
}

static inline void c_port_fab_vs_hdr_set_spt_idx(struct c_port_fab_vs_hdr *p, uint16_t spt_idx)
{
#ifdef __LITTLE_ENDIAN
	p->spt_idx_low = spt_idx;
	p->spt_idx_hi = spt_idx >> 3;
#else
	p->spt_idx = spt_idx;
#endif
}

static inline uint16_t c_port_fab_vs_hdr_get_spt_idx(const struct c_port_fab_vs_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->spt_idx_low | (uint16_t)p->spt_idx_hi << 3;
#else
	return p->spt_idx;
#endif
}

static inline void c_port_fab_vs_hdr_set_vni(struct c_port_fab_vs_hdr *p, uint16_t vni)
{
	p->vni = cpu_to_be16(vni);
}

static inline uint16_t c_port_fab_vs_hdr_get_vni(const struct c_port_fab_vs_hdr *p)
{
	return be16_to_cpu(p->vni);
}

#ifdef __LITTLE_ENDIAN
struct c_port_unrestricted_hdr {
	uint8_t opcode              :  4;
	uint8_t ver_pkt_type        :  3;
	uint8_t req_not_rsp         :  1;
	uint8_t index_ext           :  5;
	uint8_t pktid_ext           :  2;
	uint8_t flush               :  1;
	uint64_t remote_offset      : 56;
	uint8_t msg_id;
	uint8_t seq_num_hi;
	uint8_t clr_seq_num_hi      :  4;
	uint8_t seq_num_low         :  4;
	uint8_t clr_seq_num_low;
	uint8_t sct_idx_hi;
	uint8_t rendezvous_id_hi    :  4;
	uint8_t sct_idx_low         :  4;
	uint8_t                     :  2;
	uint8_t use_offset_for_get  :  1;
	uint8_t src_error           :  1;
	uint8_t rendezvous_id_low   :  4;
	uint32_t request_len;
	uint32_t initiator;
	uint64_t match_bits;
	uint64_t header_data;
	uint8_t eg_len_hi;
	uint8_t eg_len_mid;
	uint8_t                     :  4;
	uint8_t eg_len_low          :  4;
} __attribute__((packed));
#else
struct c_port_unrestricted_hdr {
	uint8_t req_not_rsp         :  1;
	uint8_t ver_pkt_type        :  3;
	uint8_t opcode              :  4;
	uint8_t flush               :  1;
	uint8_t pktid_ext           :  2;
	uint8_t index_ext           :  5;
	uint64_t remote_offset      : 56;
	uint8_t msg_id;
	uint16_t seq_num            : 12;
	uint16_t clr_seq_num        : 12;
	uint16_t sct_idx            : 12;
	uint16_t rendezvous_id      :  8;
	uint8_t src_error           :  1;
	uint8_t use_offset_for_get  :  1;
	uint8_t                     :  2;
	uint32_t request_len;
	uint32_t initiator;
	uint64_t match_bits;
	uint64_t header_data;
	uint32_t eg_len             : 20;
	uint8_t                     :  4;
} __attribute__((packed));
#endif

static inline void c_port_unrestricted_hdr_set_remote_offset(struct c_port_unrestricted_hdr *p, uint64_t remote_offset)
{
	p->remote_offset = cpu_to_be64(remote_offset << 8);
}

static inline uint64_t c_port_unrestricted_hdr_get_remote_offset(const struct c_port_unrestricted_hdr *p)
{
	return be64_to_cpu(p->remote_offset) >> 8;
}

static inline void c_port_unrestricted_hdr_set_seq_num(struct c_port_unrestricted_hdr *p, uint16_t seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->seq_num_low = seq_num;
	p->seq_num_hi = seq_num >> 4;
#else
	p->seq_num = seq_num;
#endif
}

static inline uint16_t c_port_unrestricted_hdr_get_seq_num(const struct c_port_unrestricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->seq_num_low | (uint16_t)p->seq_num_hi << 4;
#else
	return p->seq_num;
#endif
}

static inline void c_port_unrestricted_hdr_set_clr_seq_num(struct c_port_unrestricted_hdr *p, uint16_t clr_seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->clr_seq_num_low = clr_seq_num;
	p->clr_seq_num_hi = clr_seq_num >> 8;
#else
	p->clr_seq_num = clr_seq_num;
#endif
}

static inline uint16_t c_port_unrestricted_hdr_get_clr_seq_num(const struct c_port_unrestricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->clr_seq_num_low | (uint16_t)p->clr_seq_num_hi << 8;
#else
	return p->clr_seq_num;
#endif
}

static inline void c_port_unrestricted_hdr_set_sct_idx(struct c_port_unrestricted_hdr *p, uint16_t sct_idx)
{
#ifdef __LITTLE_ENDIAN
	p->sct_idx_low = sct_idx;
	p->sct_idx_hi = sct_idx >> 4;
#else
	p->sct_idx = sct_idx;
#endif
}

static inline uint16_t c_port_unrestricted_hdr_get_sct_idx(const struct c_port_unrestricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->sct_idx_low | (uint16_t)p->sct_idx_hi << 4;
#else
	return p->sct_idx;
#endif
}

static inline void c_port_unrestricted_hdr_set_rendezvous_id(struct c_port_unrestricted_hdr *p, uint8_t rendezvous_id)
{
#ifdef __LITTLE_ENDIAN
	p->rendezvous_id_low = rendezvous_id;
	p->rendezvous_id_hi = rendezvous_id >> 4;
#else
	p->rendezvous_id = rendezvous_id;
#endif
}

static inline uint8_t c_port_unrestricted_hdr_get_rendezvous_id(const struct c_port_unrestricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint8_t)p->rendezvous_id_low | (uint8_t)p->rendezvous_id_hi << 4;
#else
	return p->rendezvous_id;
#endif
}

static inline void c_port_unrestricted_hdr_set_request_len(struct c_port_unrestricted_hdr *p, uint32_t request_len)
{
	p->request_len = cpu_to_be32(request_len);
}

static inline uint32_t c_port_unrestricted_hdr_get_request_len(const struct c_port_unrestricted_hdr *p)
{
	return be32_to_cpu(p->request_len);
}

static inline void c_port_unrestricted_hdr_set_initiator(struct c_port_unrestricted_hdr *p, uint32_t initiator)
{
	p->initiator = cpu_to_be32(initiator);
}

static inline uint32_t c_port_unrestricted_hdr_get_initiator(const struct c_port_unrestricted_hdr *p)
{
	return be32_to_cpu(p->initiator);
}

static inline void c_port_unrestricted_hdr_set_match_bits(struct c_port_unrestricted_hdr *p, uint64_t match_bits)
{
	p->match_bits = cpu_to_be64(match_bits);
}

static inline uint64_t c_port_unrestricted_hdr_get_match_bits(const struct c_port_unrestricted_hdr *p)
{
	return be64_to_cpu(p->match_bits);
}

static inline void c_port_unrestricted_hdr_set_header_data(struct c_port_unrestricted_hdr *p, uint64_t header_data)
{
	p->header_data = cpu_to_be64(header_data);
}

static inline uint64_t c_port_unrestricted_hdr_get_header_data(const struct c_port_unrestricted_hdr *p)
{
	return be64_to_cpu(p->header_data);
}

static inline void c_port_unrestricted_hdr_set_eg_len(struct c_port_unrestricted_hdr *p, uint32_t eg_len)
{
#ifdef __LITTLE_ENDIAN
	p->eg_len_low = eg_len;
	p->eg_len_mid = eg_len >> 4;
	p->eg_len_hi = eg_len >> 12;
#else
	p->eg_len = eg_len;
#endif
}

static inline uint32_t c_port_unrestricted_hdr_get_eg_len(const struct c_port_unrestricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint32_t)p->eg_len_low | (uint32_t)p->eg_len_mid << 4 | (uint32_t)p->eg_len_hi << 12;
#else
	return p->eg_len;
#endif
}

#ifdef __LITTLE_ENDIAN
struct c_port_small_msg_hdr {
	uint8_t opcode            :  4;
	uint8_t ver_pkt_type      :  3;
	uint8_t req_not_rsp       :  1;
	uint8_t index_ext         :  5;
	uint8_t pktid_ext         :  2;
	uint8_t flush             :  1;
	uint64_t header_data      : 64;
	uint8_t seq_num_hi;
	uint8_t clr_seq_num_hi    :  4;
	uint8_t seq_num_low       :  4;
	uint8_t clr_seq_num_low;
	uint8_t sct_idx_hi;
	uint8_t request_len_hi    :  4;
	uint8_t sct_idx_low       :  4;
	uint8_t request_len_low;
	uint32_t initiator;
	uint64_t match_bits       : 64;
	uint8_t                   :  7;
	uint8_t src_error         :  1;
} __attribute__((packed));
#else
struct c_port_small_msg_hdr {
	uint8_t req_not_rsp   :  1;
	uint8_t ver_pkt_type  :  3;
	uint8_t opcode        :  4;
	uint8_t flush         :  1;
	uint8_t pktid_ext     :  2;
	uint8_t index_ext     :  5;
	uint64_t header_data  : 64;
	uint16_t seq_num      : 12;
	uint16_t clr_seq_num  : 12;
	uint16_t sct_idx      : 12;
	uint16_t request_len  : 12;
	uint32_t initiator;
	uint64_t match_bits   : 64;
	uint8_t src_error     :  1;
	uint8_t               :  7;
} __attribute__((packed));
#endif

static inline void c_port_small_msg_hdr_set_header_data(struct c_port_small_msg_hdr *p, uint64_t header_data)
{
	p->header_data = cpu_to_be64(header_data);
}

static inline uint64_t c_port_small_msg_hdr_get_header_data(const struct c_port_small_msg_hdr *p)
{
	return be64_to_cpu(p->header_data);
}

static inline void c_port_small_msg_hdr_set_seq_num(struct c_port_small_msg_hdr *p, uint16_t seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->seq_num_low = seq_num;
	p->seq_num_hi = seq_num >> 4;
#else
	p->seq_num = seq_num;
#endif
}

static inline uint16_t c_port_small_msg_hdr_get_seq_num(const struct c_port_small_msg_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->seq_num_low | (uint16_t)p->seq_num_hi << 4;
#else
	return p->seq_num;
#endif
}

static inline void c_port_small_msg_hdr_set_clr_seq_num(struct c_port_small_msg_hdr *p, uint16_t clr_seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->clr_seq_num_low = clr_seq_num;
	p->clr_seq_num_hi = clr_seq_num >> 8;
#else
	p->clr_seq_num = clr_seq_num;
#endif
}

static inline uint16_t c_port_small_msg_hdr_get_clr_seq_num(const struct c_port_small_msg_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->clr_seq_num_low | (uint16_t)p->clr_seq_num_hi << 8;
#else
	return p->clr_seq_num;
#endif
}

static inline void c_port_small_msg_hdr_set_sct_idx(struct c_port_small_msg_hdr *p, uint16_t sct_idx)
{
#ifdef __LITTLE_ENDIAN
	p->sct_idx_low = sct_idx;
	p->sct_idx_hi = sct_idx >> 4;
#else
	p->sct_idx = sct_idx;
#endif
}

static inline uint16_t c_port_small_msg_hdr_get_sct_idx(const struct c_port_small_msg_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->sct_idx_low | (uint16_t)p->sct_idx_hi << 4;
#else
	return p->sct_idx;
#endif
}

static inline void c_port_small_msg_hdr_set_request_len(struct c_port_small_msg_hdr *p, uint16_t request_len)
{
#ifdef __LITTLE_ENDIAN
	p->request_len_low = request_len;
	p->request_len_hi = request_len >> 8;
#else
	p->request_len = request_len;
#endif
}

static inline uint16_t c_port_small_msg_hdr_get_request_len(const struct c_port_small_msg_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->request_len_low | (uint16_t)p->request_len_hi << 8;
#else
	return p->request_len;
#endif
}

static inline void c_port_small_msg_hdr_set_initiator(struct c_port_small_msg_hdr *p, uint32_t initiator)
{
	p->initiator = cpu_to_be32(initiator);
}

static inline uint32_t c_port_small_msg_hdr_get_initiator(const struct c_port_small_msg_hdr *p)
{
	return be32_to_cpu(p->initiator);
}

static inline void c_port_small_msg_hdr_set_match_bits(struct c_port_small_msg_hdr *p, uint64_t match_bits)
{
	p->match_bits = cpu_to_be64(match_bits);
}

static inline uint64_t c_port_small_msg_hdr_get_match_bits(const struct c_port_small_msg_hdr *p)
{
	return be64_to_cpu(p->match_bits);
}

#ifdef __LITTLE_ENDIAN
struct c_port_continuation_hdr {
	uint8_t opcode             :  4;
	uint8_t ver_pkt_type       :  3;
	uint8_t req_not_rsp        :  1;
	uint8_t                    :  2;
	uint8_t src_error          :  1;
	uint8_t pkt_ordered        :  1;
	uint8_t eom                :  1;
	uint8_t pktid_ext          :  2;
	uint8_t flush              :  1;
	uint32_t offset            : 32;
	uint8_t msg_id;
	uint8_t seq_num_hi;
	uint8_t clr_seq_num_hi     :  4;
	uint8_t seq_num_low        :  4;
	uint8_t clr_seq_num_low;
	uint8_t sct_idx_hi;
	uint8_t pkt_xfer_len_hi    :  4;
	uint8_t sct_idx_low        :  4;
	uint8_t pkt_xfer_len_low;
} __attribute__((packed));
#else
struct c_port_continuation_hdr {
	uint8_t req_not_rsp    :  1;
	uint8_t ver_pkt_type   :  3;
	uint8_t opcode         :  4;
	uint8_t flush          :  1;
	uint8_t pktid_ext      :  2;
	uint8_t eom            :  1;
	uint8_t pkt_ordered    :  1;
	uint8_t src_error      :  1;
	uint8_t                :  2;
	uint32_t offset        : 32;
	uint8_t msg_id;
	uint16_t seq_num       : 12;
	uint16_t clr_seq_num   : 12;
	uint16_t sct_idx       : 12;
	uint16_t pkt_xfer_len  : 12;
} __attribute__((packed));
#endif

static inline void c_port_continuation_hdr_set_offset(struct c_port_continuation_hdr *p, uint32_t offset)
{
	p->offset = cpu_to_be32(offset);
}

static inline uint32_t c_port_continuation_hdr_get_offset(const struct c_port_continuation_hdr *p)
{
	return be32_to_cpu(p->offset);
}

static inline void c_port_continuation_hdr_set_seq_num(struct c_port_continuation_hdr *p, uint16_t seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->seq_num_low = seq_num;
	p->seq_num_hi = seq_num >> 4;
#else
	p->seq_num = seq_num;
#endif
}

static inline uint16_t c_port_continuation_hdr_get_seq_num(const struct c_port_continuation_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->seq_num_low | (uint16_t)p->seq_num_hi << 4;
#else
	return p->seq_num;
#endif
}

static inline void c_port_continuation_hdr_set_clr_seq_num(struct c_port_continuation_hdr *p, uint16_t clr_seq_num)
{
#ifdef __LITTLE_ENDIAN
	p->clr_seq_num_low = clr_seq_num;
	p->clr_seq_num_hi = clr_seq_num >> 8;
#else
	p->clr_seq_num = clr_seq_num;
#endif
}

static inline uint16_t c_port_continuation_hdr_get_clr_seq_num(const struct c_port_continuation_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->clr_seq_num_low | (uint16_t)p->clr_seq_num_hi << 8;
#else
	return p->clr_seq_num;
#endif
}

static inline void c_port_continuation_hdr_set_sct_idx(struct c_port_continuation_hdr *p, uint16_t sct_idx)
{
#ifdef __LITTLE_ENDIAN
	p->sct_idx_low = sct_idx;
	p->sct_idx_hi = sct_idx >> 4;
#else
	p->sct_idx = sct_idx;
#endif
}

static inline uint16_t c_port_continuation_hdr_get_sct_idx(const struct c_port_continuation_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->sct_idx_low | (uint16_t)p->sct_idx_hi << 4;
#else
	return p->sct_idx;
#endif
}

static inline void c_port_continuation_hdr_set_pkt_xfer_len(struct c_port_continuation_hdr *p, uint16_t pkt_xfer_len)
{
#ifdef __LITTLE_ENDIAN
	p->pkt_xfer_len_low = pkt_xfer_len;
	p->pkt_xfer_len_hi = pkt_xfer_len >> 8;
#else
	p->pkt_xfer_len = pkt_xfer_len;
#endif
}

static inline uint16_t c_port_continuation_hdr_get_pkt_xfer_len(const struct c_port_continuation_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->pkt_xfer_len_low | (uint16_t)p->pkt_xfer_len_hi << 8;
#else
	return p->pkt_xfer_len;
#endif
}

#ifdef __LITTLE_ENDIAN
struct c_port_restricted_hdr {
	uint8_t opcode            :  4;
	uint8_t ver_pkt_type      :  3;
	uint8_t req_not_rsp       :  1;
	uint8_t index_ext         :  5;
	uint8_t pktid_ext         :  2;
	uint8_t flush             :  1;
	uint64_t remote_offset    : 56;
	uint8_t request_len_hi    :  4;
	uint8_t src_error         :  1;
	uint8_t                   :  3;
	uint8_t request_len_low;
} __attribute__((packed));
#else
struct c_port_restricted_hdr {
	uint8_t req_not_rsp     :  1;
	uint8_t ver_pkt_type    :  3;
	uint8_t opcode          :  4;
	uint8_t flush           :  1;
	uint8_t pktid_ext       :  2;
	uint8_t index_ext       :  5;
	uint64_t remote_offset  : 56;
	uint8_t                 :  3;
	uint8_t src_error       :  1;
	uint16_t request_len    : 12;
} __attribute__((packed));
#endif

static inline void c_port_restricted_hdr_set_remote_offset(struct c_port_restricted_hdr *p, uint64_t remote_offset)
{
	p->remote_offset = cpu_to_be64(remote_offset << 8);
}

static inline uint64_t c_port_restricted_hdr_get_remote_offset(const struct c_port_restricted_hdr *p)
{
	return be64_to_cpu(p->remote_offset) >> 8;
}

static inline void c_port_restricted_hdr_set_request_len(struct c_port_restricted_hdr *p, uint16_t request_len)
{
#ifdef __LITTLE_ENDIAN
	p->request_len_low = request_len;
	p->request_len_hi = request_len >> 8;
#else
	p->request_len = request_len;
#endif
}

static inline uint16_t c_port_restricted_hdr_get_request_len(const struct c_port_restricted_hdr *p)
{
#ifdef __LITTLE_ENDIAN
	return (uint16_t)p->request_len_low | (uint16_t)p->request_len_hi << 8;
#else
	return p->request_len;
#endif
}

#ifdef __LITTLE_ENDIAN
struct c_port_amo_payload {
	uint8_t amo_op          :  4;
	uint8_t                 :  4;
	uint8_t amo_type        :  4;
	uint8_t                 :  4;
	uint8_t cswap_op        :  3;
	uint8_t                 :  5;
	uint64_t                : 40;
	uint64_t op1_word1_le;
	uint64_t op1_word2_le;
	uint64_t op2_word1_le;
	uint64_t op2_word2_le;
} __attribute__((packed));
#else
struct c_port_amo_payload {
	uint8_t                 :  4;
	uint8_t amo_op          :  4;
	uint8_t                 :  4;
	uint8_t amo_type        :  4;
	uint8_t                 :  5;
	uint8_t cswap_op        :  3;
	uint64_t                : 40;
	uint64_t op1_word1_le;
	uint64_t op1_word2_le;
	uint64_t op2_word1_le;
	uint64_t op2_word2_le;
} __attribute__((packed));
#endif

static inline void c_port_amo_payload_set_op1_word1_le(struct c_port_amo_payload *p, uint64_t op1_word1_le)
{
	p->op1_word1_le = cpu_to_be64(op1_word1_le);
}

static inline uint64_t c_port_amo_payload_get_op1_word1_le(const struct c_port_amo_payload *p)
{
	return be64_to_cpu(p->op1_word1_le);
}

static inline void c_port_amo_payload_set_op1_word2_le(struct c_port_amo_payload *p, uint64_t op1_word2_le)
{
	p->op1_word2_le = cpu_to_be64(op1_word2_le);
}

static inline uint64_t c_port_amo_payload_get_op1_word2_le(const struct c_port_amo_payload *p)
{
	return be64_to_cpu(p->op1_word2_le);
}

static inline void c_port_amo_payload_set_op2_word1_le(struct c_port_amo_payload *p, uint64_t op2_word1_le)
{
	p->op2_word1_le = cpu_to_be64(op2_word1_le);
}

static inline uint64_t c_port_amo_payload_get_op2_word1_le(const struct c_port_amo_payload *p)
{
	return be64_to_cpu(p->op2_word1_le);
}

static inline void c_port_amo_payload_set_op2_word2_le(struct c_port_amo_payload *p, uint64_t op2_word2_le)
{
	p->op2_word2_le = cpu_to_be64(op2_word2_le);
}

static inline uint64_t c_port_amo_payload_get_op2_word2_le(const struct c_port_amo_payload *p)
{
	return be64_to_cpu(p->op2_word2_le);
}

struct c_hni_cntrs {
	uint64_t cor_ecc_err_cntr;
	uint64_t ucor_ecc_err_cntr;
	uint64_t tx_stall_llr;
	uint64_t cntr0;
	uint64_t rx_stall_ixe_fifo;
	uint64_t cntr1[3];
	uint64_t rx_stall_ixe_pktbuf[8];
	uint64_t pfc_fifo_oflw_cntr[8];
	uint64_t discard_cntr[8];
	uint64_t tx_ok_ieee;
	uint64_t tx_ok_opt;
	uint64_t tx_poisoned_ieee;
	uint64_t tx_poisoned_opt;
	uint64_t tx_ok_broadcast;
	uint64_t tx_ok_multicast;
	uint64_t tx_ok_tagged;
	uint64_t tx_ok_27;
	uint64_t tx_ok_35;
	uint64_t tx_ok_36_to_63;
	uint64_t tx_ok_64;
	uint64_t tx_ok_65_to_127;
	uint64_t tx_ok_128_to_255;
	uint64_t tx_ok_256_to_511;
	uint64_t tx_ok_512_to_1023;
	uint64_t tx_ok_1024_to_2047;
	uint64_t tx_ok_2048_to_4095;
	uint64_t tx_ok_4096_to_8191;
	uint64_t tx_ok_8192_to_max;
	uint64_t rx_ok_ieee;
	uint64_t rx_ok_opt;
	uint64_t rx_bad_ieee;
	uint64_t rx_bad_opt;
	uint64_t rx_ok_broadcast;
	uint64_t rx_ok_multicast;
	uint64_t rx_ok_mac_control;
	uint64_t rx_ok_bad_op_code;
	uint64_t rx_ok_flow_control;
	uint64_t rx_ok_tagged;
	uint64_t rx_ok_len_type_error;
	uint64_t rx_ok_too_long;
	uint64_t rx_ok_undersize;
	uint64_t rx_fragment;
	uint64_t rx_jabber;
	uint64_t rx_ok_27;
	uint64_t rx_ok_35;
	uint64_t rx_ok_36_to_63;
	uint64_t rx_ok_64;
	uint64_t rx_ok_65_to_127;
	uint64_t rx_ok_128_to_255;
	uint64_t rx_ok_256_to_511;
	uint64_t rx_ok_512_to_1023;
	uint64_t rx_ok_1024_to_2047;
	uint64_t rx_ok_2048_to_4095;
	uint64_t rx_ok_4096_to_8191;
	uint64_t rx_ok_8192_to_max;
	uint64_t hrp_ack;
	uint64_t hrp_resp;
	uint64_t fgfc_port;
	uint64_t fgfc_l2;
	uint64_t fgfc_ipv4;
	uint64_t fgfc_ipv6;
	uint64_t fgfc_match;
	uint64_t fgfc_event_xon;
	uint64_t fgfc_event_xoff;
	uint64_t fgfc_discard;
	uint64_t pause_recv[8];
	uint64_t pause_xoff_sent[8];
	uint64_t rx_paused[8];
	uint64_t tx_paused[8];
	uint64_t rx_paused_std;
	uint64_t pause_sent;
	uint64_t pause_refresh;
	uint64_t cntr2;
	uint64_t tx_std_padded;
	uint64_t tx_opt_padded;
	uint64_t tx_std_size_err;
	uint64_t tx_opt_size_err;
	uint64_t pkts_sent_by_tc[8];
	uint64_t pkts_recv_by_tc[8];
	uint64_t multicast_pkts_sent_by_tc[8];
	uint64_t multicast_pkts_recv_by_tc[8];
};

struct c1_hni_pcs_cntrs {
	uint64_t pml_mbe_error_cnt;
	uint64_t pml_sbe_error_cnt;
	uint64_t corrected_cw;
	uint64_t uncorrected_cw;
	uint64_t good_cw;
	uint64_t fecl_errors[8];
};

struct c2_hni_pcs_cntrs {
	uint64_t pml_mbe_error_cnt;
	uint64_t pml_sbe_error_cnt;
	uint64_t corrected_cw;
	uint64_t uncorrected_cw;
	uint64_t good_cw;
	uint64_t fecl_errors[16];
	uint64_t corrected_cw_bin[15];
};

struct c1_hni_mac_cntrs {
	uint64_t rx_illegal_size;
	uint64_t rx_fragment;
	uint64_t rx_preamble_error;
};

struct c2_hni_mac_cntrs {
	uint64_t rx_illegal_size;
	uint64_t rx_fragment;
	uint64_t rx_preamble_error;
	uint64_t tx_ifg_adjustment_sync;
};

struct c1_hni_llr_cntrs {
	uint64_t tx_loop_time_req_ctl_fr;
	uint64_t tx_loop_time_rsp_ctl_fr;
	uint64_t tx_init_ctl_os;
	uint64_t tx_init_echo_ctl_os;
	uint64_t tx_ack_ctl_os;
	uint64_t tx_nack_ctl_os;
	uint64_t tx_discard;
	uint64_t tx_ok_lossy;
	uint64_t tx_ok_lossless;
	uint64_t tx_ok_lossless_rpt;
	uint64_t tx_poisoned_lossy;
	uint64_t tx_poisoned_lossless;
	uint64_t tx_poisoned_lossless_rpt;
	uint64_t tx_ok_bypass;
	uint64_t tx_replay_event;
	uint64_t rx_loop_time_req_ctl_fr;
	uint64_t rx_loop_time_rsp_ctl_fr;
	uint64_t rx_bad_ctl_fr;
	uint64_t rx_init_ctl_os;
	uint64_t rx_init_echo_ctl_os;
	uint64_t rx_ack_ctl_os;
	uint64_t rx_nack_ctl_os;
	uint64_t rx_ack_nack_seq_err;
	uint64_t rx_ok_lossy;
	uint64_t rx_poisoned_lossy;
	uint64_t rx_bad_lossy;
	uint64_t rx_ok_lossless;
	uint64_t rx_poisoned_lossless;
	uint64_t rx_bad_lossless;
	uint64_t rx_expected_seq_good;
	uint64_t rx_expected_seq_poisoned;
	uint64_t rx_expected_seq_bad;
	uint64_t rx_unexpected_seq;
	uint64_t rx_duplicate_seq;
	uint64_t rx_replay_event;
};

struct c2_hni_llr_cntrs {
	uint64_t tx_loop_time_req_ctl_fr;
	uint64_t tx_loop_time_rsp_ctl_fr;
	uint64_t tx_init_ctl_os;
	uint64_t tx_init_echo_ctl_os;
	uint64_t tx_ack_ctl_os;
	uint64_t tx_nack_ctl_os;
	uint64_t tx_discard;
	uint64_t tx_ok_lossy;
	uint64_t tx_ok_lossless;
	uint64_t tx_ok_lossless_rpt;
	uint64_t tx_poisoned_lossy;
	uint64_t tx_poisoned_lossless;
	uint64_t tx_poisoned_lossless_rpt;
	uint64_t tx_bad_lossy;
	uint64_t tx_bad_lossless;
	uint64_t tx_bad_lossless_rpt;
	uint64_t tx_ok_bypass;
	uint64_t tx_replay_event;
	uint64_t tx_llr_blocked;
	uint64_t tx_llr_unblocked;
	uint64_t rx_loop_time_req_ctl_fr;
	uint64_t rx_loop_time_rsp_ctl_fr;
	uint64_t rx_bad_ctl_fr;
	uint64_t rx_init_ctl_os;
	uint64_t rx_init_echo_ctl_os;
	uint64_t rx_ack_ctl_os;
	uint64_t rx_nack_ctl_os;
	uint64_t rx_ack_nack_seq_err;
	uint64_t rx_ok_lossy;
	uint64_t rx_poisoned_lossy;
	uint64_t rx_bad_lossy;
	uint64_t rx_ok_lossless;
	uint64_t rx_poisoned_lossless;
	uint64_t rx_bad_lossless;
	uint64_t rx_expected_seq_good;
	uint64_t rx_expected_seq_poisoned;
	uint64_t rx_expected_seq_bad;
	uint64_t rx_unexpected_seq;
	uint64_t rx_duplicate_seq;
	uint64_t rx_replay_event;
	uint64_t rx_ack_nack_error;
};

struct c_pi_dmac_cdesc {
	uint8_t done    :  1;
	uint8_t status  :  2;
	uint64_t        : 61;
};

struct c_idc_put_cmd {
	struct c_idc_hdr idc_header;
	uint8_t data[0];
};

struct c_list_entry_simple {
	uint64_t       : 38;
	uint16_t addr  : 14;
	uint8_t vld    :  1;
	uint16_t       : 11;
} __attribute__((packed));

struct c_mst_table_simple {
	uint16_t portal_index  : 11;
	uint8_t                :  1;
	uint8_t ptl_list       :  2;
	uint8_t                :  2;
	uint16_t buffer_id;
	uint32_t initiator;
};

struct c_ct_writeback {
	uint64_t ct_success   : 48;
	uint8_t ct_failure    :  7;
	uint16_t unused       :  8;
	uint8_t ct_writeback  :  1;
};
#define C_ATU_CFG_CRNC_OFFSET	0x00000000
#define C_ATU_CFG_CRNC	(C_ATU_BASE + C_ATU_CFG_CRNC_OFFSET)
#define C_ATU_CFG_CRNC_SIZE	0x00000008

union c_atu_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_ATU_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_ATU_MSC_SHADOW_ACTION	(C_ATU_BASE + C_ATU_MSC_SHADOW_ACTION_OFFSET)
#define C_ATU_MSC_SHADOW_ACTION_SIZE	0x00000008

union c_msc_shadow_action {
	uint64_t qw;
	struct {
		uint64_t addr_offset  : 23;
		uint64_t              :  8;
		uint64_t write        :  1;
		uint64_t              : 32;
	};
};

#define C_ATU_MSC_SHADOW_OFFSET	0x00000040
#define C_ATU_MSC_SHADOW	(C_ATU_BASE + C_ATU_MSC_SHADOW_OFFSET)
#define C_ATU_MSC_SHADOW_SIZE	0x00000020

union c_atu_msc_shadow {
	uint64_t qw[4];
};

#define C_ATU_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_ATU_ERR_ELAPSED_TIME	(C_ATU_BASE + C_ATU_ERR_ELAPSED_TIME_OFFSET)
#define C_ATU_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_atu_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_ATU_ERR_FLG_OFFSET	0x00000108
#define C_ATU_ERR_FLG	(C_ATU_BASE + C_ATU_ERR_FLG_OFFSET)
#define C_ATU_ERR_FLG_SIZE	0x00000008

union c_atu_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                   :  1;
		uint64_t crnc_ring_sync_error      :  1;
		uint64_t crnc_ring_ecc_sbe         :  1;
		uint64_t crnc_ring_ecc_mbe         :  1;
		uint64_t crnc_ring_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_unknown      :  1;
		uint64_t crnc_csr_cmd_incomplete   :  1;
		uint64_t crnc_buf_ecc_sbe          :  1;
		uint64_t crnc_buf_ecc_mbe          :  1;
		uint64_t                           : 23;
		uint64_t fifo_err                  :  1;
		uint64_t page_perm_err             :  1;
		uint64_t ats_trans_err             :  1;
		uint64_t pcie_unsuccess_cmpl       :  1;
		uint64_t pcie_error_poisoned       :  1;
		uint64_t ats_page_req_err          :  1;
		uint64_t no_translation            :  1;
		uint64_t invalid_ac                :  1;
		uint64_t prb_expired               :  1;
		uint64_t nta_pte_align_err         :  1;
		uint64_t ats_odp_config_err        :  1;
		uint64_t addr_out_of_range         :  1;
		uint64_t ats_page_size_config_err  :  1;
		uint64_t ats_page_size_err         :  1;
		uint64_t p_tarb_cdt_uflow          :  1;
		uint64_t np_tarb_cdt_uflow         :  1;
		uint64_t ee_odp_req_err            :  1;
		uint64_t ats_fn_disabled_err       :  1;
		uint64_t rarb_hw_err               :  1;
		uint64_t epoch_cnt_err             :  1;
		uint64_t rspq_cmpl_err             :  1;
		uint64_t odp_unexp_idx_err         :  1;
		uint64_t in_progress_cnt_err       :  1;
		uint64_t                           :  9;
	};
};

#define C_ATU_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_ATU_ERR_FIRST_FLG	(C_ATU_BASE + C_ATU_ERR_FIRST_FLG_OFFSET)
#define C_ATU_ERR_FIRST_FLG_SIZE	0x00000008
#define C_ATU_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_ATU_ERR_FIRST_FLG_TS	(C_ATU_BASE + C_ATU_ERR_FIRST_FLG_TS_OFFSET)
#define C_ATU_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_ATU_ERR_CLR_OFFSET	0x00000120
#define C_ATU_ERR_CLR	(C_ATU_BASE + C_ATU_ERR_CLR_OFFSET)
#define C_ATU_ERR_CLR_SIZE	0x00000008
#define C_ATU_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_ATU_ERR_IRQA_MSK	(C_ATU_BASE + C_ATU_ERR_IRQA_MSK_OFFSET)
#define C_ATU_ERR_IRQA_MSK_SIZE	0x00000008
#define C_ATU_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_ATU_ERR_IRQB_MSK	(C_ATU_BASE + C_ATU_ERR_IRQB_MSK_OFFSET)
#define C_ATU_ERR_IRQB_MSK_SIZE	0x00000008
#define C_ATU_ERR_INFO_MSK_OFFSET	0x00000140
#define C_ATU_ERR_INFO_MSK	(C_ATU_BASE + C_ATU_ERR_INFO_MSK_OFFSET)
#define C_ATU_ERR_INFO_MSK_SIZE	0x00000008
#define C_ATU_EXT_ERR_FLG_OFFSET	0x00000148
#define C_ATU_EXT_ERR_FLG	(C_ATU_BASE + C_ATU_EXT_ERR_FLG_OFFSET)
#define C_ATU_EXT_ERR_FLG_SIZE	0x00000008

union c_atu_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag              :  1;
		uint64_t ac_mem_cor           :  1;
		uint64_t otb_mem_cor          :  1;
		uint64_t tag_mem_cor          :  1;
		uint64_t ingress_bus_cor      :  1;
		uint64_t data_mem_cor         :  1;
		uint64_t chain_tail_mem_cor   :  1;
		uint64_t chain_next_mem_cor   :  1;
		uint64_t missq_mem_cor        :  1;
		uint64_t uniqq_mem_cor        :  1;
		uint64_t chain_mem_cor        :  1;
		uint64_t pointer_mem_cor      :  1;
		uint64_t replayq_mem_cor      :  1;
		uint64_t retryq_mem_cor       :  1;
		uint64_t prb_mem_cor          :  1;
		uint64_t invalq_mem_cor       :  1;
		uint64_t rspinfoq_mem_cor     :  1;
		uint64_t cmpldata_mem_cor     :  1;
		uint64_t atucq_mem_cor        :  1;
		uint64_t ats_prq_mem_cor      :  1;
		uint64_t rarb_hdr_bus_cor     :  1;
		uint64_t rarb_data_bus_cor    :  1;
		uint64_t odpq_mem_cor         :  1;
		uint64_t                      :  9;
		uint64_t                      :  1;
		uint64_t ac_mem_ucor          :  1;
		uint64_t otb_mem_ucor         :  1;
		uint64_t tag_mem_ucor         :  1;
		uint64_t ingress_bus_ucor     :  1;
		uint64_t data_mem_ucor        :  1;
		uint64_t chain_tail_mem_ucor  :  1;
		uint64_t chain_next_mem_ucor  :  1;
		uint64_t missq_mem_ucor       :  1;
		uint64_t uniqq_mem_ucor       :  1;
		uint64_t chain_mem_ucor       :  1;
		uint64_t pointer_mem_ucor     :  1;
		uint64_t replayq_mem_ucor     :  1;
		uint64_t retryq_mem_ucor      :  1;
		uint64_t prb_mem_ucor         :  1;
		uint64_t invalq_mem_ucor      :  1;
		uint64_t rspinfoq_mem_ucor    :  1;
		uint64_t cmpldata_mem_ucor    :  1;
		uint64_t atucq_mem_ucor       :  1;
		uint64_t ats_prq_mem_ucor     :  1;
		uint64_t rarb_hdr_bus_ucor    :  1;
		uint64_t rarb_data_bus_ucor   :  1;
		uint64_t odpq_mem_ucor        :  1;
		uint64_t                      :  1;
		uint64_t anchor_vld_mem_ucor  :  1;
		uint64_t anchor_mem_ucor      :  1;
		uint64_t                      :  2;
		uint64_t                      :  4;
	};
};

#define C_ATU_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C_ATU_EXT_ERR_FIRST_FLG	(C_ATU_BASE + C_ATU_EXT_ERR_FIRST_FLG_OFFSET)
#define C_ATU_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_ATU_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C_ATU_EXT_ERR_FIRST_FLG_TS	(C_ATU_BASE + C_ATU_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_ATU_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_ATU_EXT_ERR_CLR_OFFSET	0x00000160
#define C_ATU_EXT_ERR_CLR	(C_ATU_BASE + C_ATU_EXT_ERR_CLR_OFFSET)
#define C_ATU_EXT_ERR_CLR_SIZE	0x00000008
#define C_ATU_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C_ATU_EXT_ERR_IRQA_MSK	(C_ATU_BASE + C_ATU_EXT_ERR_IRQA_MSK_OFFSET)
#define C_ATU_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_ATU_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C_ATU_EXT_ERR_IRQB_MSK	(C_ATU_BASE + C_ATU_EXT_ERR_IRQB_MSK_OFFSET)
#define C_ATU_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_ATU_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C_ATU_EXT_ERR_INFO_MSK	(C_ATU_BASE + C_ATU_EXT_ERR_INFO_MSK_OFFSET)
#define C_ATU_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_ATU_ERR_INFO_MEM_OFFSET	0x00000200
#define C_ATU_ERR_INFO_MEM	(C_ATU_BASE + C_ATU_ERR_INFO_MEM_OFFSET)
#define C_ATU_ERR_INFO_MEM_SIZE	0x00000008

union c_atu_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 12;
		uint64_t cor_mem_id         :  6;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 12;
		uint64_t ucor_mem_id        :  6;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_ATU_ERR_INFO_AC_OFFSET	0x00000208
#define C_ATU_ERR_INFO_AC	(C_ATU_BASE + C_ATU_ERR_INFO_AC_OFFSET)
#define C_ATU_ERR_INFO_AC_SIZE	0x00000008

union c_atu_err_info_ac {
	uint64_t qw;
	struct {
		uint64_t acid        : 10;
		uint64_t             :  2;
		uint64_t client      :  2;
		uint64_t             :  2;
		uint64_t error_type  :  3;
		uint64_t             : 45;
	};
};

#define C_ATU_ERR_INFO_TRANSLATION_OFFSET	0x00000210
#define C_ATU_ERR_INFO_TRANSLATION	(C_ATU_BASE + C_ATU_ERR_INFO_TRANSLATION_OFFSET)
#define C_ATU_ERR_INFO_TRANSLATION_SIZE	0x00000008

union c_atu_err_info_translation {
	uint64_t qw;
	struct {
		uint64_t acid    : 10;
		uint64_t client  :  2;
		uint64_t addr    : 45;
		uint64_t         :  7;
	};
};

#define C_ATU_ERR_INFO_ATS_PAGE_REQ_OFFSET	0x00000218
#define C_ATU_ERR_INFO_ATS_PAGE_REQ	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_PAGE_REQ_OFFSET)
#define C_ATU_ERR_INFO_ATS_PAGE_REQ_SIZE	0x00000008

union c_atu_err_info_ats_page_req {
	uint64_t qw;
	struct {
		uint64_t acid          : 10;
		uint64_t client        :  2;
		uint64_t prg_rsp_code  :  4;
		uint64_t prb_ucor      :  1;
		uint64_t               : 47;
	};
};

#define C_ATU_ERR_INFO_PCIE_EP_OFFSET	0x00000220
#define C_ATU_ERR_INFO_PCIE_EP	(C_ATU_BASE + C_ATU_ERR_INFO_PCIE_EP_OFFSET)
#define C_ATU_ERR_INFO_PCIE_EP_SIZE	0x00000008

union c_atu_err_info_pcie_ep {
	uint64_t qw;
	struct {
		uint64_t acid    : 10;
		uint64_t         :  2;
		uint64_t client  :  2;
		uint64_t         : 50;
	};
};

#define C_ATU_ERR_INFO_PCIE_UC_OFFSET	0x00000228
#define C_ATU_ERR_INFO_PCIE_UC	(C_ATU_BASE + C_ATU_ERR_INFO_PCIE_UC_OFFSET)
#define C_ATU_ERR_INFO_PCIE_UC_SIZE	0x00000008

union c_atu_err_info_pcie_uc {
	uint64_t qw;
	struct {
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t client       :  2;
		uint64_t              :  2;
		uint64_t cmpl_status  :  3;
		uint64_t              : 45;
	};
};

#define C_ATU_ERR_INFO_ATS_TRANS_OFFSET	0x00000230
#define C_ATU_ERR_INFO_ATS_TRANS	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_TRANS_OFFSET)
#define C_ATU_ERR_INFO_ATS_TRANS_SIZE	0x00000008

union c_atu_err_info_ats_trans {
	uint64_t qw;
	struct {
		uint64_t acid         : 10;
		uint64_t client       :  2;
		uint64_t addr         : 45;
		uint64_t cmpl_status  :  3;
		uint64_t              :  4;
	};
};

#define C_ATU_ERR_INFO_PAGE_PERM_OFFSET	0x00000238
#define C_ATU_ERR_INFO_PAGE_PERM	(C_ATU_BASE + C_ATU_ERR_INFO_PAGE_PERM_OFFSET)
#define C_ATU_ERR_INFO_PAGE_PERM_SIZE	0x00000008

union c_atu_err_info_page_perm {
	uint64_t qw;
	struct {
		uint64_t acid               : 10;
		uint64_t                    :  2;
		uint64_t client             :  2;
		uint64_t                    :  2;
		uint64_t perm_ats_priv      :  1;
		uint64_t perm_ats_exec      :  1;
		uint64_t perm_w             :  1;
		uint64_t perm_r             :  1;
		uint64_t perm_req_ats_priv  :  1;
		uint64_t perm_req_ats_exec  :  1;
		uint64_t perm_req_w         :  1;
		uint64_t perm_req_r         :  1;
		uint64_t                    : 40;
	};
};

#define C_ATU_ERR_INFO_NTA_PTE_ALIGN_OFFSET	0x00000240
#define C_ATU_ERR_INFO_NTA_PTE_ALIGN	(C_ATU_BASE + C_ATU_ERR_INFO_NTA_PTE_ALIGN_OFFSET)
#define C_ATU_ERR_INFO_NTA_PTE_ALIGN_SIZE	0x00000008

union c_atu_err_info_nta_pte_align {
	uint64_t qw;
	struct {
		uint64_t acid       : 10;
		uint64_t client     :  2;
		uint64_t addr       : 45;
		uint64_t page_size  :  5;
		uint64_t            :  2;
	};
};

#define C_ATU_ERR_INFO_FIFO_OFFSET	0x00000248
#define C_ATU_ERR_INFO_FIFO	(C_ATU_BASE + C_ATU_ERR_INFO_FIFO_OFFSET)
#define C_ATU_ERR_INFO_FIFO_SIZE	0x00000008

union c_atu_err_info_fifo {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           : 14;
		uint64_t fifo_id   :  4;
		uint64_t           : 44;
	};
};

#define C_ATU_ERR_INFO_ADDR_OUT_OF_RANGE_OFFSET	0x00000250
#define C_ATU_ERR_INFO_ADDR_OUT_OF_RANGE	(C_ATU_BASE + C_ATU_ERR_INFO_ADDR_OUT_OF_RANGE_OFFSET)
#define C_ATU_ERR_INFO_ADDR_OUT_OF_RANGE_SIZE	0x00000008

union c_atu_err_info_addr_out_of_range {
	uint64_t qw;
	struct {
		uint64_t acid    : 10;
		uint64_t client  :  2;
		uint64_t addr    : 45;
		uint64_t         :  7;
	};
};

#define C_ATU_ERR_INFO_ATS_PAGE_SIZE_CONFIG_OFFSET	0x00000258
#define C_ATU_ERR_INFO_ATS_PAGE_SIZE_CONFIG	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_PAGE_SIZE_CONFIG_OFFSET)
#define C_ATU_ERR_INFO_ATS_PAGE_SIZE_CONFIG_SIZE	0x00000008

union c_atu_err_info_ats_page_size_config {
	uint64_t qw;
	struct {
		uint64_t acid     : 10;
		uint64_t          :  2;
		uint64_t client   :  2;
		uint64_t          :  2;
		uint64_t pg_size  :  5;
		uint64_t          : 43;
	};
};

#define C_ATU_ERR_INFO_ATS_PAGE_SIZE_OFFSET	0x00000260
#define C_ATU_ERR_INFO_ATS_PAGE_SIZE	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_PAGE_SIZE_OFFSET)
#define C_ATU_ERR_INFO_ATS_PAGE_SIZE_SIZE	0x00000008

union c_atu_err_info_ats_page_size {
	uint64_t qw;
	struct {
		uint64_t acid     : 10;
		uint64_t          :  2;
		uint64_t client   :  2;
		uint64_t          :  2;
		uint64_t pg_size  :  6;
		uint64_t          : 42;
	};
};

#define C_ATU_ERR_INFO_EE_ODP_REQ_OFFSET	0x00000268
#define C_ATU_ERR_INFO_EE_ODP_REQ	(C_ATU_BASE + C_ATU_ERR_INFO_EE_ODP_REQ_OFFSET)
#define C_ATU_ERR_INFO_EE_ODP_REQ_SIZE	0x00000008

union c_atu_err_info_ee_odp_req {
	uint64_t qw;
	struct {
		uint64_t acid  : 10;
		uint64_t       :  2;
		uint64_t addr  : 45;
		uint64_t       :  7;
	};
};

#define C_ATU_ERR_INFO_ATS_ODP_CONFIG_OFFSET	0x00000270
#define C_ATU_ERR_INFO_ATS_ODP_CONFIG	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_ODP_CONFIG_OFFSET)
#define C_ATU_ERR_INFO_ATS_ODP_CONFIG_SIZE	0x00000008

union c_atu_err_info_ats_odp_config {
	uint64_t qw;
	struct {
		uint64_t acid    : 10;
		uint64_t         :  2;
		uint64_t client  :  2;
		uint64_t         : 50;
	};
};

#define C_ATU_ERR_INFO_PRB_EXPIRED_OFFSET	0x00000278
#define C_ATU_ERR_INFO_PRB_EXPIRED	(C_ATU_BASE + C_ATU_ERR_INFO_PRB_EXPIRED_OFFSET)
#define C_ATU_ERR_INFO_PRB_EXPIRED_SIZE	0x00000008

union c_atu_err_info_prb_expired {
	uint64_t qw;
	struct {
		uint64_t index  :  9;
		uint64_t        : 55;
	};
};

#define C_ATU_ERR_INFO_ATS_FN_DISABLED_OFFSET	0x00000280
#define C_ATU_ERR_INFO_ATS_FN_DISABLED	(C_ATU_BASE + C_ATU_ERR_INFO_ATS_FN_DISABLED_OFFSET)
#define C_ATU_ERR_INFO_ATS_FN_DISABLED_SIZE	0x00000008

union c_atu_err_info_ats_fn_disabled {
	uint64_t qw;
	struct {
		uint64_t acid    : 10;
		uint64_t         :  2;
		uint64_t client  :  2;
		uint64_t         : 50;
	};
};

#define C_ATU_ERR_INFO_RARB_OFFSET	0x00000288
#define C_ATU_ERR_INFO_RARB	(C_ATU_BASE + C_ATU_ERR_INFO_RARB_OFFSET)
#define C_ATU_ERR_INFO_RARB_SIZE	0x00000008

union c_atu_err_info_rarb {
	uint64_t qw;
	struct {
		uint64_t hdr_ucor               :  1;
		uint64_t data_ucor              :  1;
		uint64_t bad_vld                :  1;
		uint64_t bad_eop                :  1;
		uint64_t bad_cmd                :  1;
		uint64_t unexpected_cmpl_ptid   :  1;
		uint64_t cmpl_tag_out_of_range  :  1;
		uint64_t                        : 57;
	};
};

#define C_ATU_ERR_INFO_EPOCH_CNT_OFFSET	0x00000290
#define C_ATU_ERR_INFO_EPOCH_CNT	(C_ATU_BASE + C_ATU_ERR_INFO_EPOCH_CNT_OFFSET)
#define C_ATU_ERR_INFO_EPOCH_CNT_SIZE	0x00000008

union c_atu_err_info_epoch_cnt {
	uint64_t qw;
	struct {
		uint64_t oxe_at_epoch0_cnt_underrun  :  1;
		uint64_t oxe_at_epoch0_cnt_overrun   :  1;
		uint64_t oxe_at_epoch1_cnt_underrun  :  1;
		uint64_t oxe_at_epoch1_cnt_overrun   :  1;
		uint64_t ixe_at_epoch0_cnt_underrun  :  1;
		uint64_t ixe_at_epoch0_cnt_overrun   :  1;
		uint64_t ixe_at_epoch1_cnt_underrun  :  1;
		uint64_t ixe_at_epoch1_cnt_overrun   :  1;
		uint64_t ee_at_epoch0_cnt_underrun   :  1;
		uint64_t ee_at_epoch0_cnt_overrun    :  1;
		uint64_t ee_at_epoch1_cnt_underrun   :  1;
		uint64_t ee_at_epoch1_cnt_overrun    :  1;
		uint64_t ib_epoch0_cnt_underrun      :  1;
		uint64_t ib_epoch0_cnt_overrun       :  1;
		uint64_t ib_epoch1_cnt_underrun      :  1;
		uint64_t ib_epoch1_cnt_overrun       :  1;
		uint64_t                             : 48;
	};
};

#define C_ATU_ERR_INFO_RSPQ_CMPL_OFFSET	0x00000298
#define C_ATU_ERR_INFO_RSPQ_CMPL	(C_ATU_BASE + C_ATU_ERR_INFO_RSPQ_CMPL_OFFSET)
#define C_ATU_ERR_INFO_RSPQ_CMPL_SIZE	0x00000008

union c_atu_err_info_rspq_cmpl {
	uint64_t qw;
	struct {
		uint64_t cmpl_last_surplus  :  1;
		uint64_t cmpl_last_deficit  :  1;
		uint64_t cmpl_missing       :  1;
		uint64_t                    : 61;
	};
};

#define C_ATU_ERR_INFO_ODP_UNEXP_IDX_OFFSET	0x000002a0
#define C_ATU_ERR_INFO_ODP_UNEXP_IDX	(C_ATU_BASE + C_ATU_ERR_INFO_ODP_UNEXP_IDX_OFFSET)
#define C_ATU_ERR_INFO_ODP_UNEXP_IDX_SIZE	0x00000008

union c_atu_err_info_odp_unexp_idx {
	uint64_t qw;
	struct {
		uint64_t prb_index   :  9;
		uint64_t ats_pg_rsp  :  1;
		uint64_t             : 54;
	};
};

#define C_ATU_ERR_INFO_IN_PROGRESS_CNT_OFFSET	0x000002a8
#define C_ATU_ERR_INFO_IN_PROGRESS_CNT	(C_ATU_BASE + C_ATU_ERR_INFO_IN_PROGRESS_CNT_OFFSET)
#define C_ATU_ERR_INFO_IN_PROGRESS_CNT_SIZE	0x00000008

union c_atu_err_info_in_progress_cnt {
	uint64_t qw;
	struct {
		uint64_t oxe_epoch0_cnt_underrun  :  1;
		uint64_t oxe_epoch0_cnt_overrun   :  1;
		uint64_t oxe_epoch1_cnt_underrun  :  1;
		uint64_t oxe_epoch1_cnt_overrun   :  1;
		uint64_t ixe_epoch0_cnt_underrun  :  1;
		uint64_t ixe_epoch0_cnt_overrun   :  1;
		uint64_t ixe_epoch1_cnt_underrun  :  1;
		uint64_t ixe_epoch1_cnt_overrun   :  1;
		uint64_t ee_epoch0_cnt_underrun   :  1;
		uint64_t ee_epoch0_cnt_overrun    :  1;
		uint64_t ee_epoch1_cnt_underrun   :  1;
		uint64_t ee_epoch1_cnt_overrun    :  1;
		uint64_t                          : 52;
	};
};

#define C_ATU_CFG_NTA_OFFSET	0x00000400
#define C_ATU_CFG_NTA	(C_ATU_BASE + C_ATU_CFG_NTA_OFFSET)
#define C_ATU_CFG_NTA_SIZE	0x00000008

union c_atu_cfg_nta {
	uint64_t qw;
	struct {
		uint64_t acid  : 10;
		uint64_t       : 54;
	};
};

#define C_ATU_CFG_ATS_OFFSET	0x00000408
#define C_ATU_CFG_ATS	(C_ATU_BASE + C_ATU_CFG_ATS_OFFSET)
#define C_ATU_CFG_ATS_SIZE	0x00000008

union c_atu_cfg_ats {
	uint64_t qw;
	struct {
		uint64_t inval_batch_mode        :  1;
		uint64_t                         :  3;
		uint64_t num_xlations_req        :  2;
		uint64_t                         :  2;
		uint64_t inval_process_interval  :  5;
		uint64_t                         :  3;
		uint64_t truncate_gt_stu_rsp     :  1;
		uint64_t                         : 47;
	};
};

#define C_ATU_CFG_CMDPROC_OFFSET	0x00000410
#define C_ATU_CFG_CMDPROC	(C_ATU_BASE + C_ATU_CFG_CMDPROC_OFFSET)
#define C_ATU_CFG_CMDPROC_SIZE	0x00000008

union c_atu_cfg_cmdproc {
	uint64_t qw;
	struct {
		uint64_t         :  4;
		uint64_t rd_ptr  :  5;
		uint64_t         :  7;
		uint64_t         :  4;
		uint64_t wr_ptr  :  5;
		uint64_t         : 39;
	};
};

#define C_ATU_CFG_WB_DATA_OFFSET	0x00000418
#define C_ATU_CFG_WB_DATA	(C_ATU_BASE + C_ATU_CFG_WB_DATA_OFFSET)
#define C_ATU_CFG_WB_DATA_SIZE	0x00000008

union c_atu_cfg_wb_data {
	uint64_t qw;
	struct {
		uint64_t data;
	};
};

#define C_ATU_CFG_CMDS_OFFSET	0x00000430
#define C_ATU_CFG_CMDS	(C_ATU_BASE + C_ATU_CFG_CMDS_OFFSET)
#define C_ATU_CFG_CMDS_SIZE	0x00000008

union c_atu_cfg_cmds {
	uint64_t qw;
	struct {
		uint64_t writeback_acid        : 10;
		uint64_t                       :  6;
		uint64_t ib_wait_int_idx       : 11;
		uint64_t                       :  1;
		uint64_t ib_wait_int_enable    :  1;
		uint64_t                       :  3;
		uint64_t cmpl_wait_int_idx     : 11;
		uint64_t                       :  1;
		uint64_t cmpl_wait_int_enable  :  1;
		uint64_t                       : 19;
	};
};

#define C_ATU_CFG_ODPQ_OFFSET	0x00000438
#define C_ATU_CFG_ODPQ	(C_ATU_BASE + C_ATU_CFG_ODPQ_OFFSET)
#define C_ATU_CFG_ODPQ_SIZE	0x00000008

union c_atu_cfg_odpq {
	uint64_t qw;
	struct {
		uint64_t prb_index  :  9;
		uint64_t            :  7;
		uint64_t status     :  1;
		uint64_t            : 47;
	};
};

#define C_ATU_CFG_ODP_OFFSET	0x00000440
#define C_ATU_CFG_ODP	(C_ATU_BASE + C_ATU_CFG_ODP_OFFSET)
#define C_ATU_CFG_ODP_SIZE	0x00000008

union c_atu_cfg_odp {
	uint64_t qw;
	struct {
		uint64_t acid        : 10;
		uint64_t             :  6;
		uint64_t int_idx     : 11;
		uint64_t             :  1;
		uint64_t int_enable  :  1;
		uint64_t             : 31;
		uint64_t odp_en      :  1;
		uint64_t             :  3;
	};
};

#define C_ATU_CFG_PRI_OFFSET	0x00000448
#define C_ATU_CFG_PRI	(C_ATU_BASE + C_ATU_CFG_PRI_OFFSET)
#define C_ATU_CFG_PRI_SIZE	0x00000008

union c_atu_cfg_pri {
	uint64_t qw;
	struct {
		uint64_t                      : 12;
		uint64_t base_addr            : 45;
		uint64_t                      :  3;
		uint64_t relaxed_ordering_en  :  1;
		uint64_t                      :  3;
	};
};

#define C_ATU_CFG_PRB_OFFSET	0x00000450
#define C_ATU_CFG_PRB	(C_ATU_BASE + C_ATU_CFG_PRB_OFFSET)
#define C_ATU_CFG_PRB_SIZE	0x00000008

union c_atu_cfg_prb {
	uint64_t qw;
	struct {
		uint64_t timeout  :  4;
		uint64_t          : 60;
	};
};

#define C1_ATU_CFG_ODP_DECOUPLE_OFFSET	0x00000478
#define C1_ATU_CFG_ODP_DECOUPLE	(C_ATU_BASE + C1_ATU_CFG_ODP_DECOUPLE_OFFSET)
#define C1_ATU_CFG_ODP_DECOUPLE_SIZE	0x00000008

union c1_atu_cfg_odp_decouple {
	uint64_t qw;
	struct {
		uint64_t flush_req_delay  : 10;
		uint64_t                  :  2;
		uint64_t enable           :  1;
		uint64_t                  : 51;
	};
};

#define C2_ATU_CFG_ODP_DECOUPLE_OFFSET	0x00000478
#define C2_ATU_CFG_ODP_DECOUPLE	(C_ATU_BASE + C2_ATU_CFG_ODP_DECOUPLE_OFFSET)
#define C2_ATU_CFG_ODP_DECOUPLE_SIZE	0x00000008

union c2_atu_cfg_odp_decouple {
	uint64_t qw;
	struct {
		uint64_t flush_req_delay   : 32;
		uint64_t enable            :  1;
		uint64_t wait_no_trigger   :  1;
		uint64_t ip_all_client     :  1;
		uint64_t tcam_mask         :  2;
		uint64_t flush_to_trigger  :  1;
		uint64_t trigger_wo_odp    :  1;
		uint64_t                   : 25;
	};
};

#define C_ATU_CFG_INBOUND_WAIT_OFFSET	0x00000480
#define C_ATU_CFG_INBOUND_WAIT	(C_ATU_BASE + C_ATU_CFG_INBOUND_WAIT_OFFSET)
#define C_ATU_CFG_INBOUND_WAIT_SIZE	0x00000008

union c_atu_cfg_inbound_wait {
	uint64_t qw;
	struct {
		uint64_t           :  3;
		uint64_t rsp_addr  : 54;
		uint64_t rsp_en    :  1;
		uint64_t           :  6;
	};
};

#define C_ATU_CFG_AC_TABLE_OFFSET(idx)	(0x00010000+((idx)*32))
#define C_ATU_CFG_AC_TABLE_ENTRIES	1024
#define C_ATU_CFG_AC_TABLE(idx)	(C_ATU_BASE + C_ATU_CFG_AC_TABLE_OFFSET(idx))
#define C_ATU_CFG_AC_TABLE_SIZE	0x00008000

union c_atu_cfg_ac_table {
	uint64_t qw[4];
	struct {
		union {
			struct {
				uint64_t		  : 12;
				uint64_t nta_root_ptr     : 45;
				uint64_t nta_use_req_acid :  1;
				uint64_t		  :  6;
			};
			struct {
				uint64_t                        : 12;
				uint64_t ats_vf_num             : 6;
				uint64_t ats_vf_en              : 1;
				uint64_t ats_no_write           : 1;
				uint64_t ats_pasid              : 20;
				uint64_t ats_pasid_er           : 1;
				uint64_t ats_pasid_pmr          : 1;
				uint64_t ats_pasid_en           : 1;
				uint64_t ats_filter_optimistic  : 1;
				uint64_t                        : 20;
			};
			struct {
				uint64_t	      : 12;
				uint64_t ta_mode_opts : 46;
				uint64_t	      : 6;
			};
		};
		uint64_t		: 12;
		uint64_t mem_size       : 45;
		uint64_t		:  7;
		uint64_t		: 15;
		uint64_t mem_base       : 42;
		uint64_t		:  7;
		uint64_t		: 32;
		uint64_t idx_salt	:  8;
		uint64_t ptn_salt	:  2;
		uint64_t cntr_pool_id	:  2;
		uint64_t odp_mode       :  1;
		uint64_t odp_en		:  1;
		uint64_t do_not_filter  :  1;
		uint64_t		:  1;
		uint64_t ptg_mode       :  3;
		uint64_t		:  1;
		uint64_t pg_table_size  :  4;
		uint64_t base_pg_size   :  4;
		uint64_t do_not_cache   :  1;
		uint64_t ta_mode	:  2;
		uint64_t context_en     :  1;
	};
};

#define C_ATU_CFG_CQ_TABLE_OFFSET(idx)	(0x0001a000+((idx)*8))
#define C_ATU_CFG_CQ_TABLE_ENTRIES	32
#define C_ATU_CFG_CQ_TABLE(idx)	(C_ATU_BASE + C_ATU_CFG_CQ_TABLE_OFFSET(idx))
#define C_ATU_CFG_CQ_TABLE_SIZE	0x00000100

union c_atu_cfg_cq_table {
	uint64_t qw;
	union {
		struct {
			uint64_t cmd_args  : 62;
			uint64_t cmd       :  2;
		};
		struct {
			uint64_t		    :  3;
			uint64_t comp_wait_rsp_addr : 54;
			uint64_t comp_wait_rsp_en   :  1;
			uint64_t		    :  4;
			uint64_t comp_wait_cmd	    :  2;
		};
		struct {
			uint64_t inval_acid     : 10;
			uint64_t		:  1;
			uint64_t inval_size     :  1;
			uint64_t inval_addr     : 45;
			uint64_t		:  5;
			uint64_t inval_addr_cmd :  2;
		};
	};
};

#define C_ATU_STS_INIT_DONE_OFFSET	0x00000800
#define C_ATU_STS_INIT_DONE	(C_ATU_BASE + C_ATU_STS_INIT_DONE_OFFSET)
#define C_ATU_STS_INIT_DONE_SIZE	0x00000008

union c_atu_sts_init_done {
	uint64_t qw;
	struct {
		uint64_t tag_init_done     :  1;
		uint64_t anchor_init_done  :  1;
		uint64_t prb_init_done     :  1;
		uint64_t                   : 61;
	};
};

#define C_ATU_STS_CMDPROC_OFFSET	0x00000808
#define C_ATU_STS_CMDPROC	(C_ATU_BASE + C_ATU_STS_CMDPROC_OFFSET)
#define C_ATU_STS_CMDPROC_SIZE	0x00000008

union c_atu_sts_cmdproc {
	uint64_t qw;
	struct {
		uint64_t atucq_cmd_count    : 16;
		uint64_t                    :  8;
		uint64_t atucq_cmd          :  2;
		uint64_t                    :  2;
		uint64_t atucq_running      :  1;
		uint64_t atucq_idle         :  1;
		uint64_t atucq_waiting_arb  :  1;
		uint64_t inbound_wait_idle  :  1;
		uint64_t unused             : 32;
	};
};

#define C_ATU_STS_AT_EPOCH_OFFSET	0x00000810
#define C_ATU_STS_AT_EPOCH	(C_ATU_BASE + C_ATU_STS_AT_EPOCH_OFFSET)
#define C_ATU_STS_AT_EPOCH_SIZE	0x00000008

union c_atu_sts_at_epoch {
	uint64_t qw;
	struct {
		uint64_t current_epoch        :  1;
		uint64_t                      :  7;
		uint64_t ee_epoch0_cntr_ne0   :  1;
		uint64_t ee_epoch1_cntr_ne0   :  1;
		uint64_t                      :  2;
		uint64_t oxe_epoch0_cntr_ne0  :  1;
		uint64_t oxe_epoch1_cntr_ne0  :  1;
		uint64_t                      :  2;
		uint64_t ixe_epoch0_cntr_ne0  :  1;
		uint64_t ixe_epoch1_cntr_ne0  :  1;
		uint64_t                      : 46;
	};
};

#define C_ATU_STS_IXE_AT_EPOCH_CNTR_OFFSET	0x00000818
#define C_ATU_STS_IXE_AT_EPOCH_CNTR	(C_ATU_BASE + C_ATU_STS_IXE_AT_EPOCH_CNTR_OFFSET)
#define C_ATU_STS_IXE_AT_EPOCH_CNTR_SIZE	0x00000008

union c_atu_sts_ixe_at_epoch_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  : 16;
		uint64_t epoch1  : 16;
		uint64_t unused  : 32;
	};
};

#define C_ATU_STS_OXE_AT_EPOCH_CNTR_OFFSET	0x00000820
#define C_ATU_STS_OXE_AT_EPOCH_CNTR	(C_ATU_BASE + C_ATU_STS_OXE_AT_EPOCH_CNTR_OFFSET)
#define C_ATU_STS_OXE_AT_EPOCH_CNTR_SIZE	0x00000008

union c_atu_sts_oxe_at_epoch_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  : 16;
		uint64_t epoch1  : 16;
		uint64_t unused  : 32;
	};
};

#define C_ATU_STS_EE_AT_EPOCH_CNTR_OFFSET	0x00000828
#define C_ATU_STS_EE_AT_EPOCH_CNTR	(C_ATU_BASE + C_ATU_STS_EE_AT_EPOCH_CNTR_OFFSET)
#define C_ATU_STS_EE_AT_EPOCH_CNTR_SIZE	0x00000008

union c_atu_sts_ee_at_epoch_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  : 16;
		uint64_t epoch1  : 16;
		uint64_t unused  : 32;
	};
};

#define C_ATU_STS_IB_EPOCH_OFFSET	0x00000830
#define C_ATU_STS_IB_EPOCH	(C_ATU_BASE + C_ATU_STS_IB_EPOCH_OFFSET)
#define C_ATU_STS_IB_EPOCH_SIZE	0x00000008

union c_atu_sts_ib_epoch {
	uint64_t qw;
	struct {
		uint64_t current_epoch    :  1;
		uint64_t mirrored_epoch   :  1;
		uint64_t                  :  6;
		uint64_t epoch0_cntr_ne0  :  1;
		uint64_t epoch1_cntr_ne0  :  1;
		uint64_t                  : 54;
	};
};

#define C_ATU_STS_IB_EPOCH_CNTR_OFFSET	0x00000838
#define C_ATU_STS_IB_EPOCH_CNTR	(C_ATU_BASE + C_ATU_STS_IB_EPOCH_CNTR_OFFSET)
#define C_ATU_STS_IB_EPOCH_CNTR_SIZE	0x00000008

union c_atu_sts_ib_epoch_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  : 16;
		uint64_t epoch1  : 16;
		uint64_t unused  : 32;
	};
};

#define C_ATU_STS_ODPQ_OFFSET	0x00000848
#define C_ATU_STS_ODPQ	(C_ATU_BASE + C_ATU_STS_ODPQ_OFFSET)
#define C_ATU_STS_ODPQ_SIZE	0x00000008

union c_atu_sts_odpq {
	uint64_t qw;
	struct {
		uint64_t odpq_space  :  4;
		uint64_t             : 60;
	};
};

#define C_ATU_STS_ATS_VF_DISABLED_OFFSET	0x00000858
#define C_ATU_STS_ATS_VF_DISABLED	(C_ATU_BASE + C_ATU_STS_ATS_VF_DISABLED_OFFSET)
#define C_ATU_STS_ATS_VF_DISABLED_SIZE	0x00000008

union c_atu_sts_ats_vf_disabled {
	uint64_t qw;
	struct {
		uint64_t vf_disabled;
	};
};

#define C_ATU_STS_ATS_PF_DISABLED_OFFSET	0x00000860
#define C_ATU_STS_ATS_PF_DISABLED	(C_ATU_BASE + C_ATU_STS_ATS_PF_DISABLED_OFFSET)
#define C_ATU_STS_ATS_PF_DISABLED_SIZE	0x00000008

union c_atu_sts_ats_pf_disabled {
	uint64_t qw;
	struct {
		uint64_t pf_disabled  :  1;
		uint64_t              : 63;
	};
};

#define C_ATU_STS_ATS_INVALQ_OFFSET	0x00000918
#define C_ATU_STS_ATS_INVALQ	(C_ATU_BASE + C_ATU_STS_ATS_INVALQ_OFFSET)
#define C_ATU_STS_ATS_INVALQ_SIZE	0x00000008

union c_atu_sts_ats_invalq {
	uint64_t qw;
	struct {
		uint64_t requester_id        : 16;
		uint64_t vf_num              :  6;
		uint64_t                     :  2;
		uint64_t vf_en               :  1;
		uint64_t                     :  3;
		uint64_t invals_to_process   :  6;
		uint64_t                     :  2;
		uint64_t invalq_count        :  6;
		uint64_t invalq_full         :  1;
		uint64_t invalq_vld          :  1;
		uint64_t state_drive_rsp     :  1;
		uint64_t state_cmpl_wait     :  1;
		uint64_t state_invalidating  :  1;
		uint64_t state_wait_arb      :  1;
		uint64_t state_wait_trig     :  1;
		uint64_t                     : 15;
	};
};

#define C_ATU_STS_IXE_IN_PROGRESS_CNTR_OFFSET	0x00000920
#define C_ATU_STS_IXE_IN_PROGRESS_CNTR	(C_ATU_BASE + C_ATU_STS_IXE_IN_PROGRESS_CNTR_OFFSET)
#define C_ATU_STS_IXE_IN_PROGRESS_CNTR_SIZE	0x00000008

union c_atu_sts_ixe_in_progress_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  :  9;
		uint64_t         :  7;
		uint64_t epoch1  :  9;
		uint64_t         : 39;
	};
};

#define C_ATU_STS_OXE_IN_PROGRESS_CNTR_OFFSET	0x00000928
#define C_ATU_STS_OXE_IN_PROGRESS_CNTR	(C_ATU_BASE + C_ATU_STS_OXE_IN_PROGRESS_CNTR_OFFSET)
#define C_ATU_STS_OXE_IN_PROGRESS_CNTR_SIZE	0x00000008

union c_atu_sts_oxe_in_progress_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  :  9;
		uint64_t         :  7;
		uint64_t epoch1  :  9;
		uint64_t         : 39;
	};
};

#define C_ATU_STS_EE_IN_PROGRESS_CNTR_OFFSET	0x00000930
#define C_ATU_STS_EE_IN_PROGRESS_CNTR	(C_ATU_BASE + C_ATU_STS_EE_IN_PROGRESS_CNTR_OFFSET)
#define C_ATU_STS_EE_IN_PROGRESS_CNTR_SIZE	0x00000008

union c_atu_sts_ee_in_progress_cntr {
	uint64_t qw;
	struct {
		uint64_t epoch0  :  9;
		uint64_t         :  7;
		uint64_t epoch1  :  9;
		uint64_t         : 39;
	};
};

#define C_ATU_STS_ODP_DECOUPLE_OFFSET	0x00000938
#define C_ATU_STS_ODP_DECOUPLE	(C_ATU_BASE + C_ATU_STS_ODP_DECOUPLE_OFFSET)
#define C_ATU_STS_ODP_DECOUPLE_SIZE	0x00000008

union c_atu_sts_odp_decouple {
	uint64_t qw;
	struct {
		uint64_t previous_epoch                :  1;
		uint64_t cmpl_wait_active              :  1;
		uint64_t ixe_at_prev_epoch_cnt_ne0     :  1;
		uint64_t ixe_at_prev_epoch_ip_cnt_eq0  :  1;
		uint64_t ixe_prb_vld                   :  1;
		uint64_t state                         :  2;
		uint64_t                               : 57;
	};
};

#define C_ATU_STS_PRB_TABLE_OFFSET(idx)	(0x00020000+((idx)*8))
#define C_ATU_STS_PRB_TABLE_ENTRIES	512
#define C_ATU_STS_PRB_TABLE(idx)	(C_ATU_BASE + C_ATU_STS_PRB_TABLE_OFFSET(idx))
#define C_ATU_STS_PRB_TABLE_SIZE	0x00001000

union c_atu_sts_prb_table {
	uint64_t qw;
	struct {
		uint64_t acid      : 10;
		uint64_t client    :  2;
		uint64_t addr      : 45;
		uint64_t w         :  1;
		uint64_t r         :  1;
		uint64_t ta_mode   :  2;
		uint64_t odp_mode  :  1;
		uint64_t expired   :  1;
		uint64_t valid     :  1;
	};
};

#define C_CQ_CFG_CRNC_OFFSET	0x00000000
#define C_CQ_CFG_CRNC	(C_CQ_BASE + C_CQ_CFG_CRNC_OFFSET)
#define C_CQ_CFG_CRNC_SIZE	0x00000008

union c_cq_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_CQ_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_CQ_MSC_SHADOW_ACTION	(C_CQ_BASE + C_CQ_MSC_SHADOW_ACTION_OFFSET)
#define C_CQ_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_CQ_MSC_SHADOW_OFFSET	0x00000040
#define C_CQ_MSC_SHADOW	(C_CQ_BASE + C_CQ_MSC_SHADOW_OFFSET)
#define C_CQ_MSC_SHADOW_SIZE	0x00000088

union c_cq_msc_shadow {
	uint64_t qw[17];
};

#define C_CQ_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_CQ_ERR_ELAPSED_TIME	(C_CQ_BASE + C_CQ_ERR_ELAPSED_TIME_OFFSET)
#define C_CQ_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_cq_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_CQ_ERR_FLG_OFFSET	0x00000108
#define C_CQ_ERR_FLG	(C_CQ_BASE + C_CQ_ERR_FLG_OFFSET)
#define C_CQ_ERR_FLG_SIZE	0x00000008

union c_cq_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          :  7;
		uint64_t fifo_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t c_state_cor              :  1;
		uint64_t rarb_hdr_cor             :  1;
		uint64_t rarb_data_cor            :  1;
		uint64_t event_ram_cor            :  1;
		uint64_t ll_state_mem_cor         :  1;
		uint64_t ll_status_mem_cor        :  1;
		uint64_t c_state_ucor             :  1;
		uint64_t rarb_hdr_ucor            :  1;
		uint64_t rarb_data_ucor           :  1;
		uint64_t event_ram_ucor           :  1;
		uint64_t ll_state_mem_ucor        :  1;
		uint64_t ll_status_mem_ucor       :  1;
		uint64_t ll_nempty_vectr_ucor     :  1;
		uint64_t ll_full_vectr_ucor       :  1;
		uint64_t event_cnt_ovf            :  1;
		uint64_t tle_oversubscribed       :  1;
		uint64_t rarb_bad_xactn_err       :  1;
		uint64_t rarb_cmpltn_err          :  1;
		uint64_t rarb_tou_wr_hw_err       :  1;
		uint64_t rarb_ll_wr_hw_err        :  1;
		uint64_t rarb_invld_wr_sw_err     :  1;
		uint64_t rarb_tou_wr_sw_err       :  1;
		uint64_t rarb_ll_wr_sw_err        :  1;
		uint64_t msg_cnt_uflw             :  1;
		uint64_t fq_load_uflw             :  1;
		uint64_t wr_ptr_updt_err          :  1;
		uint64_t ard_ptr_update_ucor      :  1;
		uint64_t tx_fqu_free_list_ovf     :  1;
		uint64_t tou_fqu_free_list_ovf    :  1;
		uint64_t                          : 17;
	};
};

#define C_CQ_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_CQ_ERR_FIRST_FLG	(C_CQ_BASE + C_CQ_ERR_FIRST_FLG_OFFSET)
#define C_CQ_ERR_FIRST_FLG_SIZE	0x00000008
#define C_CQ_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_CQ_ERR_FIRST_FLG_TS	(C_CQ_BASE + C_CQ_ERR_FIRST_FLG_TS_OFFSET)
#define C_CQ_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_CQ_ERR_CLR_OFFSET	0x00000120
#define C_CQ_ERR_CLR	(C_CQ_BASE + C_CQ_ERR_CLR_OFFSET)
#define C_CQ_ERR_CLR_SIZE	0x00000008
#define C_CQ_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_CQ_ERR_IRQA_MSK	(C_CQ_BASE + C_CQ_ERR_IRQA_MSK_OFFSET)
#define C_CQ_ERR_IRQA_MSK_SIZE	0x00000008
#define C_CQ_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_CQ_ERR_IRQB_MSK	(C_CQ_BASE + C_CQ_ERR_IRQB_MSK_OFFSET)
#define C_CQ_ERR_IRQB_MSK_SIZE	0x00000008
#define C_CQ_ERR_INFO_MSK_OFFSET	0x00000140
#define C_CQ_ERR_INFO_MSK	(C_CQ_BASE + C_CQ_ERR_INFO_MSK_OFFSET)
#define C_CQ_ERR_INFO_MSK_SIZE	0x00000008
#define C_CQ_EXT_ERR_FLG_OFFSET	0x00000148
#define C_CQ_EXT_ERR_FLG	(C_CQ_BASE + C_CQ_EXT_ERR_FLG_OFFSET)
#define C_CQ_EXT_ERR_FLG_SIZE	0x00000008

union c_cq_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                 :  1;
		uint64_t remap_array_cor         :  1;
		uint64_t cprof_table_cor         :  1;
		uint64_t ac_remap_array_cor      :  1;
		uint64_t tou_fq_cmd_cor          :  1;
		uint64_t cq_fq_cmd_cor           :  1;
		uint64_t tou_fid_list_ptrs_cor   :  1;
		uint64_t cq_fid_list_ptrs_cor    :  1;
		uint64_t ll_buf_cor              :  1;
		uint64_t tx_ptr_cor              :  1;
		uint64_t tx_base_cor             :  1;
		uint64_t tx_ocuset_cor           :  1;
		uint64_t tg_ptr_cor              :  1;
		uint64_t tg_base_cor             :  1;
		uint64_t pfq_bufs_cor            :  1;
		uint64_t pfq_bufs_info_cor       :  1;
		uint64_t pfq_ptrs_cor            :  1;
		uint64_t target_pfq_cor          :  1;
		uint64_t tu_pfq_bufs_cor         :  1;
		uint64_t tu_pfq_info_cor         :  1;
		uint64_t tu_pfq_list_ptrs_cor    :  1;
		uint64_t ct_table_cor            :  1;
		uint64_t ee_table_cor            :  1;
		uint64_t ptlte_table_cor         :  1;
		uint64_t ct_events_cor           :  1;
		uint64_t thresh_table_cor        :  1;
		uint64_t trig_ct_field_cor       :  1;
		uint64_t trig_list_nodes_cor     :  1;
		uint64_t tou_ll_reorder_cor      :  1;
		uint64_t tou_cmd_ptrs_cor        :  1;
		uint64_t in_fifo_cor             :  1;
		uint64_t wb_addr_cor             :  1;
		uint64_t                         :  1;
		uint64_t remap_array_ucor        :  1;
		uint64_t cprof_table_ucor        :  1;
		uint64_t ac_remap_array_ucor     :  1;
		uint64_t tou_fq_cmd_ucor         :  1;
		uint64_t cq_fq_cmd_ucor          :  1;
		uint64_t tou_fid_list_ptrs_ucor  :  1;
		uint64_t cq_fid_list_ptrs_ucor   :  1;
		uint64_t ll_buf_ucor             :  1;
		uint64_t tx_ptr_ucor             :  1;
		uint64_t tx_base_ucor            :  1;
		uint64_t tx_ocuset_ucor          :  1;
		uint64_t tg_ptr_ucor             :  1;
		uint64_t tg_base_ucor            :  1;
		uint64_t pfq_bufs_ucor           :  1;
		uint64_t pfq_bufs_info_ucor      :  1;
		uint64_t pfq_ptrs_ucor           :  1;
		uint64_t target_pfq_ucor         :  1;
		uint64_t tu_pfq_bufs_ucor        :  1;
		uint64_t tu_pfq_info_ucor        :  1;
		uint64_t tu_pfq_list_ptrs_ucor   :  1;
		uint64_t ct_table_ucor           :  1;
		uint64_t ee_table_ucor           :  1;
		uint64_t ptlte_table_ucor        :  1;
		uint64_t ct_events_ucor          :  1;
		uint64_t thresh_table_ucor       :  1;
		uint64_t trig_ct_field_ucor      :  1;
		uint64_t trig_list_nodes_ucor    :  1;
		uint64_t tou_ll_reorder_ucor     :  1;
		uint64_t tou_cmd_ptrs_ucor       :  1;
		uint64_t in_fifo_ucor            :  1;
		uint64_t wb_addr_ucor            :  1;
	};
};

#define C_CQ_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C_CQ_EXT_ERR_FIRST_FLG	(C_CQ_BASE + C_CQ_EXT_ERR_FIRST_FLG_OFFSET)
#define C_CQ_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_CQ_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C_CQ_EXT_ERR_FIRST_FLG_TS	(C_CQ_BASE + C_CQ_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_CQ_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_CQ_EXT_ERR_CLR_OFFSET	0x00000160
#define C_CQ_EXT_ERR_CLR	(C_CQ_BASE + C_CQ_EXT_ERR_CLR_OFFSET)
#define C_CQ_EXT_ERR_CLR_SIZE	0x00000008
#define C_CQ_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C_CQ_EXT_ERR_IRQA_MSK	(C_CQ_BASE + C_CQ_EXT_ERR_IRQA_MSK_OFFSET)
#define C_CQ_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_CQ_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C_CQ_EXT_ERR_IRQB_MSK	(C_CQ_BASE + C_CQ_EXT_ERR_IRQB_MSK_OFFSET)
#define C_CQ_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_CQ_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C_CQ_EXT_ERR_INFO_MSK	(C_CQ_BASE + C_CQ_EXT_ERR_INFO_MSK_OFFSET)
#define C_CQ_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_CQ_ERR_INFO_MEM_OFFSET	0x00000190
#define C_CQ_ERR_INFO_MEM	(C_CQ_BASE + C_CQ_ERR_INFO_MEM_OFFSET)
#define C_CQ_ERR_INFO_MEM_SIZE	0x00000008

union c_cq_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 12;
		uint64_t cor_mem_id         :  7;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 12;
		uint64_t ucor_mem_id        :  7;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_CQ_ERR_INFO_FIFO_ERR_OFFSET	0x00000198
#define C_CQ_ERR_INFO_FIFO_ERR	(C_CQ_BASE + C_CQ_ERR_INFO_FIFO_ERR_OFFSET)
#define C_CQ_ERR_INFO_FIFO_ERR_SIZE	0x00000008

union c_cq_err_info_fifo_err {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           :  6;
		uint64_t fifo_id   :  6;
		uint64_t           : 50;
	};
};

#define C_CQ_ERR_INFO_CC_OFFSET	0x000001a0
#define C_CQ_ERR_INFO_CC	(C_CQ_BASE + C_CQ_ERR_INFO_CC_OFFSET)
#define C_CQ_ERR_INFO_CC_SIZE	0x00000008

union c_cq_err_info_cc {
	uint64_t qw;
	struct {
		uint64_t cc_id   :  5;
		uint64_t         :  3;
		uint64_t sub_id  :  3;
		uint64_t         : 53;
	};
};

#define C_CQ_ERR_INFO_RARB_OFFSET	0x000001b0
#define C_CQ_ERR_INFO_RARB	(C_CQ_BASE + C_CQ_ERR_INFO_RARB_OFFSET)
#define C_CQ_ERR_INFO_RARB_SIZE	0x00000010

union c_cq_err_info_rarb {
	uint64_t qw[2];
	struct {
		uint64_t bad_xactn_err        :  1;
		uint64_t cmpltn_sts_err       :  1;
		uint64_t cmpltn_ep_err        :  1;
		uint64_t cmpltn_len_err       :  1;
		uint64_t cmpltn_addr_err      :  1;
		uint64_t cmpltn_dv_err        :  1;
		uint64_t cmpltn_eop_err       :  1;
		uint64_t cmpltn_data_mbe_err  :  1;
		uint64_t tou_wr_hw_err        :  1;
		uint64_t invld_wr_sw_err      :  1;
		uint64_t tou_wr_sw_err        :  1;
		uint64_t                      :  5;
		uint64_t msg_type             :  3;
		uint64_t                      :  1;
		uint64_t cmpl_status          :  3;
		uint64_t                      :  1;
		uint64_t ep                   :  1;
		uint64_t dv                   :  1;
		uint64_t eop                  :  1;
		uint64_t data_mbe             :  1;
		uint64_t                      :  4;
		uint64_t tag                  :  9;
		uint64_t                      :  7;
		uint64_t lower_address        :  7;
		uint64_t                      :  9;
		uint64_t first_dw_be          :  4;
		uint64_t last_dw_be           :  4;
		uint64_t                      :  8;
		uint64_t length               : 11;
		uint64_t                      :  7;
		uint64_t addr_29_2            : 28;
		uint64_t                      :  2;
	};
};

#define C_CQ_ERR_INFO_RARB_LL_OFFSET	0x000001c0
#define C_CQ_ERR_INFO_RARB_LL	(C_CQ_BASE + C_CQ_ERR_INFO_RARB_LL_OFFSET)
#define C_CQ_ERR_INFO_RARB_LL_SIZE	0x00000010

union c_cq_err_info_rarb_ll {
	uint64_t qw[2];
	struct {
		uint64_t hw_data_uncorr_err  :  1;
		uint64_t hw_early_eop_err    :  1;
		uint64_t hw_missing_eop_err  :  1;
		uint64_t hw_data_valid_err   :  1;
		uint64_t sw_addr_err         :  1;
		uint64_t sw_align_err        :  1;
		uint64_t sw_op_chgd_err      :  1;
		uint64_t sw_offset_err       :  1;
		uint64_t sw_offset_chgd_err  :  1;
		uint64_t sw_overlap_err      :  1;
		uint64_t                     :  6;
		uint64_t ll_op               :  3;
		uint64_t                     : 13;
		uint64_t cq_id               : 10;
		uint64_t                     :  6;
		uint64_t is_tgq              :  1;
		uint64_t                     : 15;
		uint64_t first_dw_be         :  4;
		uint64_t last_dw_be          :  4;
		uint64_t                     :  8;
		uint64_t length              : 11;
		uint64_t                     :  7;
		uint64_t addr_in_pg_15_2     : 14;
		uint64_t                     : 16;
	};
};

#define C_CQ_ERR_INFO_OCU_MAP_OFFSET	0x000001d0
#define C_CQ_ERR_INFO_OCU_MAP	(C_CQ_BASE + C_CQ_ERR_INFO_OCU_MAP_OFFSET)
#define C_CQ_ERR_INFO_OCU_MAP_SIZE	0x00000008

union c_cq_err_info_ocu_map {
	uint64_t qw;
	struct {
		uint64_t ocu_id  :  8;
		uint64_t fq_id   :  6;
		uint64_t         : 50;
	};
};

#define C_CQ_ERR_INFO_WR_PTR_UPDT_OFFSET	0x000001d8
#define C_CQ_ERR_INFO_WR_PTR_UPDT	(C_CQ_BASE + C_CQ_ERR_INFO_WR_PTR_UPDT_OFFSET)
#define C_CQ_ERR_INFO_WR_PTR_UPDT_SIZE	0x00000008

union c_cq_err_info_wr_ptr_updt {
	uint64_t qw;
	struct {
		uint64_t invalid_wr_ptr  : 16;
		uint64_t max_ptr         : 16;
		uint64_t cq_id           : 10;
		uint64_t                 :  6;
		uint64_t is_tgq          :  1;
		uint64_t                 : 15;
	};
};

#define C_CQ_CFG_INIT_TXQ_HW_STATE_OFFSET	0x00000400
#define C_CQ_CFG_INIT_TXQ_HW_STATE	(C_CQ_BASE + C_CQ_CFG_INIT_TXQ_HW_STATE_OFFSET)
#define C_CQ_CFG_INIT_TXQ_HW_STATE_SIZE	0x00000008

union c_cq_cfg_init_txq_hw_state {
	uint64_t qw;
	struct {
		uint64_t cq_handle  : 10;
		uint64_t            :  6;
		uint64_t pending    :  1;
		uint64_t            : 47;
	};
};

#define C_CQ_CFG_INIT_TGQ_HW_STATE_OFFSET	0x00000440
#define C_CQ_CFG_INIT_TGQ_HW_STATE	(C_CQ_BASE + C_CQ_CFG_INIT_TGQ_HW_STATE_OFFSET)
#define C_CQ_CFG_INIT_TGQ_HW_STATE_SIZE	0x00000008

union c_cq_cfg_init_tgq_hw_state {
	uint64_t qw;
	struct {
		uint64_t cq_handle  :  9;
		uint64_t            :  7;
		uint64_t pending    :  1;
		uint64_t            : 47;
	};
};

#define C_CQ_CFG_INIT_CT_HW_STATE_OFFSET	0x00000480
#define C_CQ_CFG_INIT_CT_HW_STATE	(C_CQ_BASE + C_CQ_CFG_INIT_CT_HW_STATE_OFFSET)
#define C_CQ_CFG_INIT_CT_HW_STATE_SIZE	0x00000008

union c_cq_cfg_init_ct_hw_state {
	uint64_t qw;
	struct {
		uint64_t ct_handle  : 11;
		uint64_t init       :  1;
		uint64_t cancel     :  1;
		uint64_t            :  3;
		uint64_t pending    :  1;
		uint64_t            : 47;
	};
};

#define C_CQ_CFG_TLE_POOL_OFFSET(idx)	(0x00000500+((idx)*8))
#define C_CQ_CFG_TLE_POOL_ENTRIES	4
#define C_CQ_CFG_TLE_POOL(idx)	(C_CQ_BASE + C_CQ_CFG_TLE_POOL_OFFSET(idx))
#define C_CQ_CFG_TLE_POOL_SIZE	0x00000020

union c_cq_cfg_tle_pool {
	uint64_t qw;
	struct {
		uint64_t num_reserved  : 12;
		uint64_t               :  4;
		uint64_t in_use        :  8;
		uint64_t               :  8;
		uint64_t max_alloc     : 12;
		uint64_t               : 20;
	};
};

#define C_CQ_CFG_STS_TLE_SHARED_OFFSET	0x00000520
#define C_CQ_CFG_STS_TLE_SHARED	(C_CQ_BASE + C_CQ_CFG_STS_TLE_SHARED_OFFSET)
#define C_CQ_CFG_STS_TLE_SHARED_SIZE	0x00000008

union c_cq_cfg_sts_tle_shared {
	uint64_t qw;
	struct {
		uint64_t num_shared            : 12;
		uint64_t                       : 20;
		uint64_t num_shared_allocated  : 12;
		uint64_t                       :  4;
		uint64_t num_total_allocated   : 12;
		uint64_t                       :  4;
	};
};

#define C_CQ_CFG_CREDIT_LIMITS_OFFSET	0x00000530
#define C_CQ_CFG_CREDIT_LIMITS	(C_CQ_BASE + C_CQ_CFG_CREDIT_LIMITS_OFFSET)
#define C_CQ_CFG_CREDIT_LIMITS_SIZE	0x00000010

union c_cq_cfg_credit_limits {
	uint64_t qw[2];
	struct {
		uint64_t lpe_cmd_credits             : 10;
		uint64_t                             :  6;
		uint64_t lpe_rcv_fifo_credits        :  4;
		uint64_t                             :  4;
		uint64_t ee_event_credits            :  5;
		uint64_t                             : 11;
		uint64_t tou_fq_credits              :  7;
		uint64_t                             :  1;
		uint64_t tarb_p_credits              :  5;
		uint64_t                             :  3;
		uint64_t tarb_np_credits             :  5;
		uint64_t                             :  3;
		uint64_t prefetch_tou_credits        :  5;
		uint64_t                             :  3;
		uint64_t tou_wb_credits              :  5;
		uint64_t                             :  3;
		uint64_t tou_event_credits           :  3;
		uint64_t                             :  5;
		uint64_t tgq_cmd_fail_event_credits  :  3;
		uint64_t                             :  5;
		uint64_t txq_cmd_fail_event_credits  :  4;
		uint64_t                             :  4;
		uint64_t tou_cmd_fail_event_credits  :  3;
		uint64_t                             :  5;
		uint64_t fq_tot_credits              :  9;
		uint64_t                             :  7;
	};
};

#define C_CQ_CFG_EVENT_CNTS_OFFSET	0x00000540
#define C_CQ_CFG_EVENT_CNTS	(C_CQ_BASE + C_CQ_CFG_EVENT_CNTS_OFFSET)
#define C_CQ_CFG_EVENT_CNTS_SIZE	0x00000008

union c_cq_cfg_event_cnts {
	uint64_t qw;
	struct {
		uint64_t enable     :  1;
		uint64_t init       :  1;
		uint64_t init_done  :  1;
		uint64_t            :  1;
		uint64_t ovf_cnt    : 16;
		uint64_t            : 44;
	};
};

#define C_CQ_TXQ_BASE_TABLE_OFFSET(idx)	(0x00010000+((idx)*16))
#define C_CQ_TXQ_BASE_TABLE_ENTRIES	1024
#define C_CQ_TXQ_BASE_TABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_BASE_TABLE_OFFSET(idx))
#define C_CQ_TXQ_BASE_TABLE_SIZE	0x00004000

union c_cq_txq_base_table {
	uint64_t qw[2];
	struct {
		uint64_t mem_q_eq             : 11;
		uint64_t                      :  1;
		uint64_t mem_q_base           : 45;
		uint64_t                      :  7;
		uint64_t mem_q_max_ptr        : 16;
		uint64_t mem_q_rgid           :  8;
		uint64_t                      :  1;
		uint64_t                      :  1;
		uint64_t mem_q_stat_cnt_pool  :  2;
		uint64_t                      :  8;
		uint64_t mem_q_pfq            :  6;
		uint64_t                      :  2;
		uint64_t mem_q_lcid           :  4;
		uint64_t mem_q_acid           : 10;
		uint64_t mem_q_policy         :  2;
		uint64_t                      :  4;
	};
};

#define C_CQ_TXQ_ENABLE_OFFSET(idx)	(0x00002000+((idx)*8))
#define C_CQ_TXQ_ENABLE_ENTRIES	1024
#define C_CQ_TXQ_ENABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_ENABLE_OFFSET(idx))
#define C_CQ_TXQ_ENABLE_SIZE	0x00002000

union c_cq_txq_enable {
	uint64_t qw;
	struct {
		uint64_t txq_enable  :  1;
		uint64_t drain       :  1;
		uint64_t fence       :  1;
		uint64_t             : 61;
	};
};

#define C_CQ_TXQ_TC_TABLE_OFFSET(idx)	(0x00016000+((idx)*8))
#define C_CQ_TXQ_TC_TABLE_ENTRIES	1024
#define C_CQ_TXQ_TC_TABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_TC_TABLE_OFFSET(idx))
#define C_CQ_TXQ_TC_TABLE_SIZE	0x00002000

union c_cq_txq_tc_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_tc    :  3;
		uint64_t mem_q_trig  :  1;
		uint64_t mem_q_eth   :  1;
		uint64_t             : 59;
	};
};

#define C_CQ_TXQ_WRPTR_TABLE_OFFSET(idx)	(0x00018000+((idx)*8))
#define C_CQ_TXQ_WRPTR_TABLE_ENTRIES	1024
#define C_CQ_TXQ_WRPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_WRPTR_TABLE_OFFSET(idx))
#define C_CQ_TXQ_WRPTR_TABLE_SIZE	0x00002000

union c_cq_txq_wrptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_wr_ptr  : 16;
		uint64_t               : 48;
	};
};

#define C_CQ_TXQ_RDPTR_TABLE_OFFSET(idx)	(0x00090000+((idx)*8))
#define C_CQ_TXQ_RDPTR_TABLE_ENTRIES	1024
#define C_CQ_TXQ_RDPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_RDPTR_TABLE_OFFSET(idx))
#define C_CQ_TXQ_RDPTR_TABLE_SIZE	0x00002000

union c_cq_txq_rdptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_rd_ptr  : 16;
		uint64_t               : 48;
	};
};

#define C_CQ_TXQ_PREPTR_TABLE_OFFSET(idx)	(0x00070000+((idx)*8))
#define C_CQ_TXQ_PREPTR_TABLE_ENTRIES	1024
#define C_CQ_TXQ_PREPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TXQ_PREPTR_TABLE_OFFSET(idx))
#define C_CQ_TXQ_PREPTR_TABLE_SIZE	0x00002000

union c_cq_txq_preptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_pre_ptr      : 16;
		uint64_t mem_q_pre_ptr_par  :  1;
		uint64_t                    : 47;
	};
};

#define C_CQ_TXQ_ACK_CTR_OFFSET(idx)	(0x00094000+((idx)*8))
#define C_CQ_TXQ_ACK_CTR_ENTRIES	1024
#define C_CQ_TXQ_ACK_CTR(idx)	(C_CQ_BASE + C_CQ_TXQ_ACK_CTR_OFFSET(idx))
#define C_CQ_TXQ_ACK_CTR_SIZE	0x00002000

union c_cq_txq_ack_ctr {
	uint64_t qw;
	struct {
		uint64_t mem_q_ack_ctr  : 12;
		uint64_t                : 52;
	};
};

#define C_CQ_TGQ_TABLE_OFFSET(idx)	(0x0001a000+((idx)*16))
#define C_CQ_TGQ_TABLE_ENTRIES	512
#define C_CQ_TGQ_TABLE(idx)	(C_CQ_BASE + C_CQ_TGQ_TABLE_OFFSET(idx))
#define C_CQ_TGQ_TABLE_SIZE	0x00002000

union c_cq_tgq_table {
	uint64_t qw[2];
	struct {
		uint64_t mem_q_eq             : 11;
		uint64_t                      :  1;
		uint64_t mem_q_base           : 45;
		uint64_t                      :  7;
		uint64_t mem_q_max_ptr        : 16;
		uint64_t mem_q_rgid           :  8;
		uint64_t                      :  1;
		uint64_t                      :  1;
		uint64_t mem_q_stat_cnt_pool  :  2;
		uint64_t mem_q_tc_reg         :  3;
		uint64_t                      : 17;
		uint64_t mem_q_acid           : 10;
		uint64_t mem_q_policy         :  2;
		uint64_t                      :  4;
	};
};

#define C_CQ_TGQ_ENABLE_OFFSET(idx)	(0x00004000+((idx)*8))
#define C_CQ_TGQ_ENABLE_ENTRIES	512
#define C_CQ_TGQ_ENABLE(idx)	(C_CQ_BASE + C_CQ_TGQ_ENABLE_OFFSET(idx))
#define C_CQ_TGQ_ENABLE_SIZE	0x00001000

union c_cq_tgq_enable {
	uint64_t qw;
	struct {
		uint64_t tgq_enable  :  1;
		uint64_t drain       :  1;
		uint64_t             : 62;
	};
};

#define C_CQ_TGQ_WRPTR_TABLE_OFFSET(idx)	(0x0001c000+((idx)*8))
#define C_CQ_TGQ_WRPTR_TABLE_ENTRIES	512
#define C_CQ_TGQ_WRPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TGQ_WRPTR_TABLE_OFFSET(idx))
#define C_CQ_TGQ_WRPTR_TABLE_SIZE	0x00001000

union c_cq_tgq_wrptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_wr_ptr  : 16;
		uint64_t               : 48;
	};
};

#define C_CQ_TGQ_RDPTR_TABLE_OFFSET(idx)	(0x00092000+((idx)*8))
#define C_CQ_TGQ_RDPTR_TABLE_ENTRIES	512
#define C_CQ_TGQ_RDPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TGQ_RDPTR_TABLE_OFFSET(idx))
#define C_CQ_TGQ_RDPTR_TABLE_SIZE	0x00001000

union c_cq_tgq_rdptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_rd_ptr  : 16;
		uint64_t               : 48;
	};
};

#define C_CQ_TGQ_PREPTR_TABLE_OFFSET(idx)	(0x00072000+((idx)*8))
#define C_CQ_TGQ_PREPTR_TABLE_ENTRIES	512
#define C_CQ_TGQ_PREPTR_TABLE(idx)	(C_CQ_BASE + C_CQ_TGQ_PREPTR_TABLE_OFFSET(idx))
#define C_CQ_TGQ_PREPTR_TABLE_SIZE	0x00001000

union c_cq_tgq_preptr_table {
	uint64_t qw;
	struct {
		uint64_t mem_q_pre_ptr      : 16;
		uint64_t mem_q_pre_ptr_par  :  1;
		uint64_t                    : 47;
	};
};

#define C_CQ_TGQ_ACK_CTR_OFFSET(idx)	(0x00096000+((idx)*8))
#define C_CQ_TGQ_ACK_CTR_ENTRIES	512
#define C_CQ_TGQ_ACK_CTR(idx)	(C_CQ_BASE + C_CQ_TGQ_ACK_CTR_OFFSET(idx))
#define C_CQ_TGQ_ACK_CTR_SIZE	0x00001000

union c_cq_tgq_ack_ctr {
	uint64_t qw;
	struct {
		uint64_t mem_q_ack_ctr  : 12;
		uint64_t                :  4;
		uint64_t append_ctr     : 14;
		uint64_t                : 34;
	};
};

#define C_CQ_CFG_CQ_POLICY_OFFSET	0x00000640
#define C_CQ_CFG_CQ_POLICY	(C_CQ_BASE + C_CQ_CFG_CQ_POLICY_OFFSET)
#define C_CQ_CFG_CQ_POLICY_SIZE	0x00000008

union c_cq_cfg_cq_policy {
	uint64_t qw;
	struct {
		struct {
			uint8_t shift  :  4;
			uint8_t empty  :  1;
			uint8_t        :  3;
		} policy[4];
		uint32_t unused;
	};
};

#define C_CQ_CFG_CP_TABLE_OFFSET(idx)	(0x0001d000+((idx)*8))
#define C_CQ_CFG_CP_TABLE_ENTRIES	256
#define C_CQ_CFG_CP_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_CP_TABLE_OFFSET(idx))
#define C_CQ_CFG_CP_TABLE_SIZE	0x00000800

union c_cq_cfg_cp_table {
	uint64_t qw;
	struct {
		uint64_t vni           : 16;
		uint64_t dscp_unrsto   :  6;
		uint64_t               :  2;
		uint64_t dscp_rstuno   :  6;
		uint64_t               :  2;
		uint64_t hrp_vld       :  1;
		uint64_t smt_disabled  :  1;
		uint64_t sct_disabled  :  1;
		uint64_t srb_disabled  :  1;
		uint64_t valid         :  1;
		uint64_t               : 27;
	};
};

#define C_CQ_CFG_CP_FL_TABLE_OFFSET(idx)	(0x0001d800+((idx)*8))
#define C_CQ_CFG_CP_FL_TABLE_ENTRIES	256
#define C_CQ_CFG_CP_FL_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_CP_FL_TABLE_OFFSET(idx))
#define C_CQ_CFG_CP_FL_TABLE_SIZE	0x00000800

union c_cq_cfg_cp_fl_table {
	uint64_t qw;
	struct {
		uint64_t pfq  :  6;
		uint64_t      :  2;
		uint64_t tc   :  3;
		uint64_t      : 53;
	};
};

#define C_CQ_CFG_CT_WB_MEM_ADDR_OFFSET(idx)	(0x00080000+((idx)*8))
#define C_CQ_CFG_CT_WB_MEM_ADDR_ENTRIES	2048
#define C_CQ_CFG_CT_WB_MEM_ADDR(idx)	(C_CQ_BASE + C_CQ_CFG_CT_WB_MEM_ADDR_OFFSET(idx))
#define C_CQ_CFG_CT_WB_MEM_ADDR_SIZE	0x00004000

union c_cq_cfg_ct_wb_mem_addr {
	uint64_t qw;
	struct {
		uint64_t wb_mem_context  : 10;
		uint64_t wb_mem_addr     : 54;
	};
};

#define C_CQ_CFG_CT_ENABLE_OFFSET(idx)	(0x00088000+((idx)*8))
#define C_CQ_CFG_CT_ENABLE_ENTRIES	2048
#define C_CQ_CFG_CT_ENABLE(idx)	(C_CQ_BASE + C_CQ_CFG_CT_ENABLE_OFFSET(idx))
#define C_CQ_CFG_CT_ENABLE_SIZE	0x00004000

union c_cq_cfg_ct_enable {
	uint64_t qw;
	struct {
		uint64_t ct_enable  :  1;
		uint64_t            : 63;
	};
};

#define C_CQ_CFG_CID_ARRAY_OFFSET(idx)	(0x00050000+((idx)*8))
#define C_CQ_CFG_CID_ARRAY_ENTRIES	4096
#define C_CQ_CFG_CID_ARRAY(idx)	(C_CQ_BASE + C_CQ_CFG_CID_ARRAY_OFFSET(idx))
#define C_CQ_CFG_CID_ARRAY_SIZE	0x00008000

union c_cq_cfg_cid_array {
	uint64_t qw;
	struct {
		uint64_t cid  :  8;
		uint64_t      : 56;
	};
};

#define C_CQ_CFG_AC_ARRAY_OFFSET(idx)	(0x00060000+((idx)*8))
#define C_CQ_CFG_AC_ARRAY_ENTRIES	2048
#define C_CQ_CFG_AC_ARRAY(idx)	(C_CQ_BASE + C_CQ_CFG_AC_ARRAY_OFFSET(idx))
#define C_CQ_CFG_AC_ARRAY_SIZE	0x00004000

union c_cq_cfg_ac_array {
	uint64_t qw;
	struct {
		uint64_t ac  : 10;
		uint64_t     : 54;
	};
};

#define C_CQ_CFG_PFQ_THRESH_TABLE_OFFSET(idx)	(0x0001e000+((idx)*8))
#define C_CQ_CFG_PFQ_THRESH_TABLE_ENTRIES	48
#define C_CQ_CFG_PFQ_THRESH_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_PFQ_THRESH_TABLE_OFFSET(idx))
#define C_CQ_CFG_PFQ_THRESH_TABLE_SIZE	0x00000180

union c_cq_cfg_pfq_thresh_table {
	uint64_t qw;
	struct {
		uint64_t high_thresh  :  9;
		uint64_t              :  7;
		uint64_t buf_cnt      : 10;
		uint64_t              : 38;
	};
};

#define C_CQ_CFG_FQ_THRESH_TABLE_OFFSET(idx)	(0x0001e200+((idx)*8))
#define C_CQ_CFG_FQ_THRESH_TABLE_ENTRIES	48
#define C_CQ_CFG_FQ_THRESH_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_FQ_THRESH_TABLE_OFFSET(idx))
#define C_CQ_CFG_FQ_THRESH_TABLE_SIZE	0x00000180

union c_cq_cfg_fq_thresh_table {
	uint64_t qw;
	struct {
		uint64_t thresh  : 41;
		uint64_t         : 23;
	};
};

#define C_CQ_CFG_PFQ_TC_MAP_OFFSET(idx)	(0x0001e400+((idx)*8))
#define C_CQ_CFG_PFQ_TC_MAP_ENTRIES	48
#define C_CQ_CFG_PFQ_TC_MAP(idx)	(C_CQ_BASE + C_CQ_CFG_PFQ_TC_MAP_OFFSET(idx))
#define C_CQ_CFG_PFQ_TC_MAP_SIZE	0x00000180

union c_cq_cfg_pfq_tc_map {
	uint64_t qw;
	struct {
		uint64_t pfq_tc  :  3;
		uint64_t         : 61;
	};
};

#define C_CQ_CFG_OCUSET_OCU_TABLE_OFFSET(idx)	(0x0001e800+((idx)*8))
#define C_CQ_CFG_OCUSET_OCU_TABLE_ENTRIES	48
#define C_CQ_CFG_OCUSET_OCU_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_OCUSET_OCU_TABLE_OFFSET(idx))
#define C_CQ_CFG_OCUSET_OCU_TABLE_SIZE	0x00000180

union c_cq_cfg_ocuset_ocu_table {
	uint64_t qw;
	struct {
		uint64_t ocu_base    :  8;
		uint64_t ocu_count   :  4;
		uint64_t ocu_static  :  1;
		uint64_t             : 51;
	};
};

#define C_CQ_CFG_OCU_HASH_MASK_OFFSET	0x00039050
#define C_CQ_CFG_OCU_HASH_MASK	(C_CQ_BASE + C_CQ_CFG_OCU_HASH_MASK_OFFSET)
#define C_CQ_CFG_OCU_HASH_MASK_SIZE	0x00000008

union c_cq_cfg_ocu_hash_mask {
	uint64_t qw;
	struct {
		uint64_t dfa   : 32;
		uint64_t vni   : 16;
		uint64_t dscp  :  6;
		uint64_t       : 10;
	};
};

#define C_CQ_CFG_OCUSET_FQ_TABLE_OFFSET(idx)	(0x0001f000+((idx)*8))
#define C_CQ_CFG_OCUSET_FQ_TABLE_ENTRIES	48
#define C_CQ_CFG_OCUSET_FQ_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_OCUSET_FQ_TABLE_OFFSET(idx))
#define C_CQ_CFG_OCUSET_FQ_TABLE_SIZE	0x00000180

union c_cq_cfg_ocuset_fq_table {
	uint64_t qw;
	struct {
		uint64_t fq_count  :  4;
		uint64_t           : 60;
	};
};

#define C_CQ_CFG_OCU_TABLE_OFFSET(idx)	(0x0001f800+((idx)*8))
#define C_CQ_CFG_OCU_TABLE_ENTRIES	256
#define C_CQ_CFG_OCU_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_OCU_TABLE_OFFSET(idx))
#define C_CQ_CFG_OCU_TABLE_SIZE	0x00000800

union c_cq_cfg_ocu_table {
	uint64_t qw;
	struct {
		uint64_t msg_count  : 11;
		uint64_t            :  1;
		uint64_t fq_saved   :  6;
		uint64_t            :  2;
		uint64_t fq         :  6;
		uint64_t            : 38;
	};
};

#define C_CQ_CFG_CT_RGID_TABLE_OFFSET(idx)	(0x00020000+((idx)*8))
#define C_CQ_CFG_CT_RGID_TABLE_ENTRIES	2048
#define C_CQ_CFG_CT_RGID_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_CT_RGID_TABLE_OFFSET(idx))
#define C_CQ_CFG_CT_RGID_TABLE_SIZE	0x00004000

union c_cq_cfg_ct_rgid_table {
	uint64_t qw;
	struct {
		uint64_t rgid  :  8;
		uint64_t       : 56;
	};
};

#define C_CQ_CFG_PTLTE_RGID_TABLE_OFFSET(idx)	(0x00028000+((idx)*8))
#define C_CQ_CFG_PTLTE_RGID_TABLE_ENTRIES	2048
#define C_CQ_CFG_PTLTE_RGID_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_PTLTE_RGID_TABLE_OFFSET(idx))
#define C_CQ_CFG_PTLTE_RGID_TABLE_SIZE	0x00004000

union c_cq_cfg_ptlte_rgid_table {
	uint64_t qw;
	struct {
		uint64_t rgid  :  8;
		uint64_t       : 56;
	};
};

#define C_CQ_CFG_EQ_RGID_TABLE_OFFSET(idx)	(0x00030000+((idx)*8))
#define C_CQ_CFG_EQ_RGID_TABLE_ENTRIES	2048
#define C_CQ_CFG_EQ_RGID_TABLE(idx)	(C_CQ_BASE + C_CQ_CFG_EQ_RGID_TABLE_OFFSET(idx))
#define C_CQ_CFG_EQ_RGID_TABLE_SIZE	0x00004000

union c_cq_cfg_eq_rgid_table {
	uint64_t qw;
	struct {
		uint64_t rgid  :  8;
		uint64_t       : 56;
	};
};

#define C_CQ_CFG_TG_THRESH_OFFSET	0x00038000
#define C_CQ_CFG_TG_THRESH	(C_CQ_BASE + C_CQ_CFG_TG_THRESH_OFFSET)
#define C_CQ_CFG_TG_THRESH_SIZE	0x00000008

union c_cq_cfg_tg_thresh {
	uint64_t qw;
	struct {
		struct {
			uint16_t thresh : 14;
			uint16_t	:  2;
		} tg[4];
	};
};

#define C_CQ_CFG_FQ_RESRV_OFFSET(idx)	(0x00039000+((idx)*8))
#define C_CQ_CFG_FQ_RESRV_ENTRIES	8
#define C_CQ_CFG_FQ_RESRV(idx)	(C_CQ_BASE + C_CQ_CFG_FQ_RESRV_OFFSET(idx))
#define C_CQ_CFG_FQ_RESRV_SIZE	0x00000040

union c_cq_cfg_fq_resrv {
	uint64_t qw;
	struct {
		uint64_t num_reserved  :  8;
		uint64_t max_alloc     :  9;
		uint64_t               :  3;
		uint64_t num_alloc     :  9;
		uint64_t               : 35;
	};
};

#define C_CQ_CFG_THRESH_MAP_OFFSET(idx)	(0x00040000+((idx)*8))
#define C_CQ_CFG_THRESH_MAP_ENTRIES	512
#define C_CQ_CFG_THRESH_MAP(idx)	(C_CQ_BASE + C_CQ_CFG_THRESH_MAP_OFFSET(idx))
#define C_CQ_CFG_THRESH_MAP_SIZE	0x00001000

union c_cq_cfg_thresh_map {
	uint64_t qw;
	struct {
		uint64_t thresh_id  :  2;
		uint64_t            : 62;
	};
};

#define C_CQ_STS_INIT_DONE_OFFSET	0x00049ff8
#define C_CQ_STS_INIT_DONE	(C_CQ_BASE + C_CQ_STS_INIT_DONE_OFFSET)
#define C_CQ_STS_INIT_DONE_SIZE	0x00000008

union c_cq_sts_init_done {
	uint64_t qw;
	struct {
		uint64_t cq_init_done   :  1;
		uint64_t tou_init_done  :  1;
		uint64_t                : 62;
	};
};

#define C_CQ_STS_TLE_IN_USE_OFFSET(idx)	(0x00000850+((idx)*8))
#define C_CQ_STS_TLE_IN_USE_ENTRIES	4
#define C_CQ_STS_TLE_IN_USE(idx)	(C_CQ_BASE + C_CQ_STS_TLE_IN_USE_OFFSET(idx))
#define C_CQ_STS_TLE_IN_USE_SIZE	0x00000020

union c_cq_sts_tle_in_use {
	uint64_t qw;
	struct {
		uint64_t count  : 12;
		uint64_t        : 52;
	};
};

#define C_CQ_STS_MAX_TLE_IN_USE_OFFSET(idx)	(0x00000870+((idx)*8))
#define C_CQ_STS_MAX_TLE_IN_USE_ENTRIES	4
#define C_CQ_STS_MAX_TLE_IN_USE(idx)	(C_CQ_BASE + C_CQ_STS_MAX_TLE_IN_USE_OFFSET(idx))
#define C_CQ_STS_MAX_TLE_IN_USE_SIZE	0x00000020

union c_cq_sts_max_tle_in_use {
	uint64_t qw;
	struct {
		uint64_t max  : 12;
		uint64_t      : 52;
	};
};

#define C_CQ_STS_CREDITS_IN_USE_OFFSET	0x000008b0
#define C_CQ_STS_CREDITS_IN_USE	(C_CQ_BASE + C_CQ_STS_CREDITS_IN_USE_OFFSET)
#define C_CQ_STS_CREDITS_IN_USE_SIZE	0x00000010

union c_cq_sts_credits_in_use {
	uint64_t qw[2];
	struct {
		uint64_t lpe_cmd_credits             : 10;
		uint64_t                             :  6;
		uint64_t lpe_rcv_fifo_credits        :  4;
		uint64_t                             :  4;
		uint64_t ee_event_credits            :  5;
		uint64_t                             : 11;
		uint64_t tou_fq_credits              :  7;
		uint64_t                             :  1;
		uint64_t tarb_p_credits              :  5;
		uint64_t                             :  3;
		uint64_t tarb_np_credits             :  5;
		uint64_t                             :  3;
		uint64_t prefetch_tou_credits        :  5;
		uint64_t                             :  3;
		uint64_t tou_wb_credits              :  5;
		uint64_t                             :  3;
		uint64_t tou_event_credits           :  3;
		uint64_t                             :  5;
		uint64_t tgq_cmd_fail_event_credits  :  3;
		uint64_t                             :  5;
		uint64_t txq_cmd_fail_event_credits  :  4;
		uint64_t tou_cmd_fail_event_credits  :  3;
		uint64_t                             :  1;
		uint64_t fq_tot_credits              :  9;
		uint64_t                             :  3;
		uint64_t fq_sh_credits               :  9;
		uint64_t                             :  3;
	};
};

#define C_CQ_STS_EVENT_CNTS_OFFSET(idx)	(0x00100000+((idx)*8))
#ifndef C_CQ_STS_EVENT_CNTS_ENTRIES
#define C_CQ_STS_EVENT_CNTS_ENTRIES	512
#endif
#define C_CQ_STS_EVENT_CNTS(idx)	(C_CQ_BASE + C_CQ_STS_EVENT_CNTS_OFFSET(idx))
#define C_CQ_STS_EVENT_CNTS_SIZE	0x00001000

union c_cq_sts_event_cnts {
	uint64_t qw;
	struct {
		uint64_t cnt  : 56;
		uint64_t      :  8;
	};
};

#define C_EE_CFG_CRNC_OFFSET	0x00000000
#define C_EE_CFG_CRNC	(C_EE_BASE + C_EE_CFG_CRNC_OFFSET)
#define C_EE_CFG_CRNC_SIZE	0x00000008

union c_ee_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_EE_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_EE_CFG_RT_OFFSET	(C_EE_BASE + C_EE_CFG_RT_OFFSET_OFFSET)
#define C_EE_CFG_RT_OFFSET_SIZE	0x00000008

union c_ee_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_EE_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_EE_MSC_SHADOW_ACTION	(C_EE_BASE + C_EE_MSC_SHADOW_ACTION_OFFSET)
#define C_EE_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_EE_MSC_SHADOW_OFFSET	0x00000040
#define C_EE_MSC_SHADOW	(C_EE_BASE + C_EE_MSC_SHADOW_OFFSET)
#define C_EE_MSC_SHADOW_SIZE	0x00000040

union c_ee_msc_shadow {
	uint64_t qw[8];
};

#define C_EE_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_EE_ERR_ELAPSED_TIME	(C_EE_BASE + C_EE_ERR_ELAPSED_TIME_OFFSET)
#define C_EE_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_ee_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_EE_ERR_FLG_OFFSET	0x00000108
#define C_EE_ERR_FLG	(C_EE_BASE + C_EE_ERR_FLG_OFFSET)
#define C_EE_ERR_FLG_SIZE	0x00000008

union c_ee_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          : 23;
		uint64_t eq_ovflw_rsrvn           :  1;
		uint64_t eq_ovflw_fc_sc           :  1;
		uint64_t eq_ovflw_ordinary        :  1;
		uint64_t eq_dsabld_event          :  1;
		uint64_t eq_dsabld_sw_state_wr    :  1;
		uint64_t eq_dsabld_lpe_query      :  1;
		uint64_t eq_rsrvn_uflw            :  1;
		uint64_t unxpctd_trnsltn_rsp      :  1;
		uint64_t rarb_hw_err              :  1;
		uint64_t rarb_sw_err              :  1;
		uint64_t tarb_err                 :  1;
		uint64_t eq_state_ucor            :  1;
		uint64_t                          :  4;
		uint64_t unxpctd_cq_cdt           :  1;
		uint64_t unxpctd_tarb_cdt         :  1;
		uint64_t eq_state_updt_urun       :  1;
		uint64_t eq_state_updt_orun       :  1;
		uint64_t                          : 12;
	};
};

#define C_EE_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_EE_ERR_FIRST_FLG	(C_EE_BASE + C_EE_ERR_FIRST_FLG_OFFSET)
#define C_EE_ERR_FIRST_FLG_SIZE	0x00000008
#define C_EE_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_EE_ERR_FIRST_FLG_TS	(C_EE_BASE + C_EE_ERR_FIRST_FLG_TS_OFFSET)
#define C_EE_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_EE_ERR_CLR_OFFSET	0x00000120
#define C_EE_ERR_CLR	(C_EE_BASE + C_EE_ERR_CLR_OFFSET)
#define C_EE_ERR_CLR_SIZE	0x00000008
#define C_EE_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_EE_ERR_IRQA_MSK	(C_EE_BASE + C_EE_ERR_IRQA_MSK_OFFSET)
#define C_EE_ERR_IRQA_MSK_SIZE	0x00000008
#define C_EE_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_EE_ERR_IRQB_MSK	(C_EE_BASE + C_EE_ERR_IRQB_MSK_OFFSET)
#define C_EE_ERR_IRQB_MSK_SIZE	0x00000008
#define C_EE_ERR_INFO_MSK_OFFSET	0x00000140
#define C_EE_ERR_INFO_MSK	(C_EE_BASE + C_EE_ERR_INFO_MSK_OFFSET)
#define C_EE_ERR_INFO_MSK_SIZE	0x00000008
#define C_EE_EXT_ERR_FLG_OFFSET	0x00000148
#define C_EE_EXT_ERR_FLG	(C_EE_BASE + C_EE_EXT_ERR_FLG_OFFSET)
#define C_EE_EXT_ERR_FLG_SIZE	0x00000008

union c_ee_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag             :  1;
		uint64_t leo_mem_cor         :  1;
		uint64_t pts_mem_cor         :  1;
		uint64_t ixe_bus_cor         :  1;
		uint64_t lpe_bus_cor         :  1;
		uint64_t mst_bus_cor         :  1;
		uint64_t pct_bus_cor         :  1;
		uint64_t hni_bus_cor         :  1;
		uint64_t cq_bus_cor          :  1;
		uint64_t eqd_mem_cor         :  1;
		uint64_t eqsws_mem_cor       :  1;
		uint64_t eqhws_mem_cor       :  1;
		uint64_t ecbsb_mem_cor       :  1;
		uint64_t ecb_mem_cor         :  1;
		uint64_t rarb_hdr_bus_cor    :  1;
		uint64_t rarb_data_bus_cor   :  1;
		uint64_t acb_vect_cor        :  1;
		uint64_t eqswupd_vect_cor    :  1;
		uint64_t lm_cbeq_mem_cor     :  1;
		uint64_t lm_state_vect_cor   :  1;
		uint64_t trnrsp_mem_cor      :  1;
		uint64_t trnrsp_bus_cor      :  1;
		uint64_t rlscb_vect_cor      :  1;
		uint64_t trncb_vect_cor      :  1;
		uint64_t                     :  2;
		uint64_t leo_mem_ucor        :  1;
		uint64_t pts_mem_ucor        :  1;
		uint64_t ixe_bus_ucor        :  1;
		uint64_t lpe_bus_ucor        :  1;
		uint64_t mst_bus_ucor        :  1;
		uint64_t pct_bus_ucor        :  1;
		uint64_t hni_bus_ucor        :  1;
		uint64_t cq_bus_ucor         :  1;
		uint64_t eqd_mem_ucor        :  1;
		uint64_t eqsws_mem_ucor      :  1;
		uint64_t eqhws_mem_ucor      :  1;
		uint64_t ecbsb_mem_ucor      :  1;
		uint64_t ecb_mem_ucor        :  1;
		uint64_t rarb_hdr_bus_ucor   :  1;
		uint64_t rarb_data_bus_ucor  :  1;
		uint64_t acb_vect_ucor       :  1;
		uint64_t lm_cbeq_mem_ucor    :  1;
		uint64_t trnrsp_mem_ucor     :  1;
		uint64_t trnrsp_bus_ucor     :  1;
		uint64_t rlscb_vect_ucor     :  1;
		uint64_t trncb_vect_ucor     :  1;
		uint64_t                     :  2;
		uint64_t ixe_to_ee_urun      :  1;
		uint64_t ixe_to_ee_orun      :  1;
		uint64_t lpe_to_ee_urun      :  1;
		uint64_t lpe_to_ee_orun      :  1;
		uint64_t mst_to_ee_urun      :  1;
		uint64_t mst_to_ee_orun      :  1;
		uint64_t pct_to_ee_urun      :  1;
		uint64_t pct_to_ee_orun      :  1;
		uint64_t hni_to_ee_urun      :  1;
		uint64_t hni_to_ee_orun      :  1;
		uint64_t cq_to_ee_urun       :  1;
		uint64_t cq_to_ee_orun       :  1;
		uint64_t                     :  3;
	};
};

#define C_EE_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C_EE_EXT_ERR_FIRST_FLG	(C_EE_BASE + C_EE_EXT_ERR_FIRST_FLG_OFFSET)
#define C_EE_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_EE_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C_EE_EXT_ERR_FIRST_FLG_TS	(C_EE_BASE + C_EE_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_EE_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_EE_EXT_ERR_CLR_OFFSET	0x00000160
#define C_EE_EXT_ERR_CLR	(C_EE_BASE + C_EE_EXT_ERR_CLR_OFFSET)
#define C_EE_EXT_ERR_CLR_SIZE	0x00000008
#define C_EE_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C_EE_EXT_ERR_IRQA_MSK	(C_EE_BASE + C_EE_EXT_ERR_IRQA_MSK_OFFSET)
#define C_EE_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_EE_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C_EE_EXT_ERR_IRQB_MSK	(C_EE_BASE + C_EE_EXT_ERR_IRQB_MSK_OFFSET)
#define C_EE_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_EE_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C_EE_EXT_ERR_INFO_MSK	(C_EE_BASE + C_EE_EXT_ERR_INFO_MSK_OFFSET)
#define C_EE_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_EE_ERR_INFO_MEM_OFFSET	0x00000190
#define C_EE_ERR_INFO_MEM	(C_EE_BASE + C_EE_ERR_INFO_MEM_OFFSET)
#define C_EE_ERR_INFO_MEM_SIZE	0x00000008

union c_ee_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 12;
		uint64_t cor_address        : 11;
		uint64_t                    :  1;
		uint64_t cor_mem_id         :  5;
		uint64_t                    :  2;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 12;
		uint64_t ucor_address       : 11;
		uint64_t                    :  1;
		uint64_t ucor_mem_id        :  5;
		uint64_t                    :  2;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_EE_ERR_INFO_EQ_CFG_OFFSET	0x00000198
#define C_EE_ERR_INFO_EQ_CFG	(C_EE_BASE + C_EE_ERR_INFO_EQ_CFG_OFFSET)
#define C_EE_ERR_INFO_EQ_CFG_SIZE	0x00000008

union c_ee_err_info_eq_cfg {
	uint64_t qw;
	struct {
		uint64_t eq_ovflw_rsrvn         :  1;
		uint64_t eq_ovflw_fc_sc         :  1;
		uint64_t eq_ovflw_ordinary      :  1;
		uint64_t eq_dsabld_event        :  1;
		uint64_t eq_dsabld_sw_state_wr  :  1;
		uint64_t eq_dsabld_lpe_query    :  1;
		uint64_t                        :  2;
		uint64_t event_type             :  5;
		uint64_t                        :  3;
		uint64_t eq_handle              : 11;
		uint64_t                        : 37;
	};
};

#define C_EE_ERR_INFO_RSRVN_UFLW_OFFSET	0x000001a0
#define C_EE_ERR_INFO_RSRVN_UFLW	(C_EE_BASE + C_EE_ERR_INFO_RSRVN_UFLW_OFFSET)
#define C_EE_ERR_INFO_RSRVN_UFLW_SIZE	0x00000008

union c_ee_err_info_rsrvn_uflw {
	uint64_t qw;
	struct {
		uint64_t                      :  8;
		uint64_t event_type           :  5;
		uint64_t                      :  3;
		uint64_t eq_handle            : 11;
		uint64_t                      :  5;
		uint64_t pending_events       : 26;
		uint64_t                      :  2;
		uint64_t required_evnt_slots  :  3;
		uint64_t                      :  1;
	};
};

#define C_EE_ERR_INFO_TRNSLTN_OFFSET	0x000001a8
#define C_EE_ERR_INFO_TRNSLTN	(C_EE_BASE + C_EE_ERR_INFO_TRNSLTN_OFFSET)
#define C_EE_ERR_INFO_TRNSLTN_SIZE	0x00000008

union c_ee_err_info_trnsltn {
	uint64_t qw;
	struct {
		uint64_t cb_id            :  8;
		uint64_t free             :  1;
		uint64_t already_trnsltd  :  1;
		uint64_t eq_wr_pending    :  1;
		uint64_t                  : 53;
	};
};

#define C_EE_ERR_INFO_RARB_OFFSET	0x000001b0
#define C_EE_ERR_INFO_RARB	(C_EE_BASE + C_EE_ERR_INFO_RARB_OFFSET)
#define C_EE_ERR_INFO_RARB_SIZE	0x00000008

union c_ee_err_info_rarb {
	uint64_t qw;
	struct {
		uint64_t hdr_ucor        :  1;
		uint64_t data_ucor       :  1;
		uint64_t bad_vld         :  1;
		uint64_t bad_eop         :  1;
		uint64_t bad_cmd         :  1;
		uint64_t invalid_access  :  1;
		uint64_t eq_none         :  1;
		uint64_t                 :  1;
		uint64_t first_dw_be     :  4;
		uint64_t last_dw_be      :  4;
		uint64_t length          : 11;
		uint64_t                 :  5;
		uint64_t                 :  2;
		uint64_t addr_29_2       : 28;
		uint64_t                 :  2;
	};
};

#define C_EE_ERR_INFO_TARB_OFFSET	0x000001b8
#define C_EE_ERR_INFO_TARB	(C_EE_BASE + C_EE_ERR_INFO_TARB_OFFSET)
#define C_EE_ERR_INFO_TARB_SIZE	0x00000008

union c_ee_err_info_tarb {
	uint64_t qw;
	struct {
		uint64_t ecb_mem_ucor       :  1;
		uint64_t ecbsb_mem_ucor     :  1;
		uint64_t trnrsp_mem_ucor    :  1;
		uint64_t trnsltn_err        :  1;
		uint64_t evnt_dropped       :  1;
		uint64_t irq_dropped        :  1;
		uint64_t trnsltn_performed  :  1;
		uint64_t                    :  1;
		uint64_t trnsltn_rc         :  6;
		uint64_t                    :  2;
		uint64_t acid               : 10;
		uint64_t                    :  6;
		uint64_t eq_handle          : 11;
		uint64_t                    : 21;
	};
};

#define C_EE_ERR_INFO_EQ_UCOR_OFFSET	0x000001c0
#define C_EE_ERR_INFO_EQ_UCOR	(C_EE_BASE + C_EE_ERR_INFO_EQ_UCOR_OFFSET)
#define C_EE_ERR_INFO_EQ_UCOR_SIZE	0x00000008

union c_ee_err_info_eq_ucor {
	uint64_t qw;
	struct {
		uint64_t eqd_mem_ucor    :  1;
		uint64_t eqsws_mem_ucor  :  1;
		uint64_t eqhws_mem_ucor  :  1;
		uint64_t                 :  5;
		uint64_t aes_rsn         :  4;
		uint64_t                 : 20;
		uint64_t eq_handle       : 11;
		uint64_t                 : 21;
	};
};

#define C_EE_CFG_TIMESTAMP_FREQ_OFFSET	0x00000410
#define C_EE_CFG_TIMESTAMP_FREQ	(C_EE_BASE + C_EE_CFG_TIMESTAMP_FREQ_OFFSET)
#define C_EE_CFG_TIMESTAMP_FREQ_SIZE	0x00000008

union c_ee_cfg_timestamp_freq {
	uint64_t qw;
	struct {
		uint64_t clk_divider  : 32;
		uint64_t              : 32;
	};
};

#define C_EE_CFG_INIT_EQ_HW_STATE_OFFSET	0x00000418
#define C_EE_CFG_INIT_EQ_HW_STATE	(C_EE_BASE + C_EE_CFG_INIT_EQ_HW_STATE_OFFSET)
#define C_EE_CFG_INIT_EQ_HW_STATE_SIZE	0x00000008

union c_ee_cfg_init_eq_hw_state {
	uint64_t qw;
	struct {
		uint64_t eq_handle  : 11;
		uint64_t            :  5;
		uint64_t pending    :  1;
		uint64_t            : 47;
	};
};

#define C_EE_CFG_LATENCY_MONITOR_OFFSET	0x00000420
#define C_EE_CFG_LATENCY_MONITOR	(C_EE_BASE + C_EE_CFG_LATENCY_MONITOR_OFFSET)
#define C_EE_CFG_LATENCY_MONITOR_SIZE	0x00000020

union c_ee_cfg_latency_monitor {
	uint64_t qw[4];
	struct {
		uint64_t granularity     : 16;
		uint64_t                 : 16;
		uint64_t lt_limit        :  9;
		uint64_t                 : 23;
		uint64_t granularity_a   : 16;
		uint64_t lt_reduction_a  :  9;
		uint64_t                 :  7;
		uint64_t lt_limit_a      :  9;
		uint64_t                 : 23;
		uint64_t granularity_b   : 16;
		uint64_t lt_reduction_b  :  9;
		uint64_t                 :  7;
		uint64_t lt_limit_b      :  9;
		uint64_t                 : 23;
		uint64_t thresh_a        :  9;
		uint64_t                 :  7;
		uint64_t hysteresis_a    :  9;
		uint64_t                 :  7;
		uint64_t thresh_b        :  9;
		uint64_t                 :  7;
		uint64_t hysteresis_b    :  9;
		uint64_t                 :  7;
	};
};

#define C_EE_CFG_EQ_DESCRIPTOR_OFFSET(idx)	(0x00020000+((idx)*64))
#define C_EE_CFG_EQ_DESCRIPTOR_ENTRIES	2048
#define C_EE_CFG_EQ_DESCRIPTOR(idx)	(C_EE_BASE + C_EE_CFG_EQ_DESCRIPTOR_OFFSET(idx))
#define C_EE_CFG_EQ_DESCRIPTOR_SIZE	0x00020000

union c_ee_cfg_eq_descriptor {
	uint64_t qw[7];
	struct {
		uint64_t eq_enable          :  1;
		uint64_t                    :  7;
		uint64_t cntr_pool_id       :  2;
		uint64_t                    :  6;
		uint64_t reserved_fc        : 14;
		uint64_t                    :  2;
		uint64_t event_int_idx      : 11;
		uint64_t                    :  4;
		uint64_t event_int_en       :  1;
		uint64_t latency_tolerance  :  9;
		uint64_t                    :  7;
		uint64_t eq_sts_num_thld    :  2;
		uint64_t                    :  6;
		uint64_t eq_sts_thld_base   :  6;
		uint64_t                    :  2;
		uint64_t eq_sts_thld_offst  :  6;
		uint64_t                    :  2;
		uint64_t eq_sts_thld_shift  :  5;
		uint64_t                    :  1;
		uint64_t eq_sts_dropped_en  :  1;
		uint64_t use_buffer_b       :  1;
		uint64_t eq_sts_int_idx     : 11;
		uint64_t                    :  4;
		uint64_t eq_sts_int_en      :  1;
		uint64_t                    : 16;
		uint64_t buffer_a_size      : 20;
		uint64_t                    : 11;
		uint64_t buffer_a_en        :  1;
		uint64_t buffer_b_size      : 20;
		uint64_t                    : 11;
		uint64_t buffer_b_en        :  1;
		uint64_t                    : 12;
		uint64_t buffer_a_addr      : 45;
		uint64_t                    :  7;
		uint64_t                    : 12;
		uint64_t buffer_b_addr      : 45;
		uint64_t                    :  7;
		uint64_t acid               : 10;
		uint64_t                    :  6;
		uint64_t pf_match           :  8;
		uint64_t pf_match_shift     :  6;
		uint64_t                    :  2;
		uint64_t pf_mask_shift      :  6;
		uint64_t                    : 26;
		uint64_t                    :  3;
		uint64_t                    : 61;
	};
};

#define C_EE_CFG_EQ_SW_STATE_OFFSET(idx)	(0x00004000+((idx)*8))
#define C_EE_CFG_EQ_SW_STATE_ENTRIES	2048
#define C_EE_CFG_EQ_SW_STATE(idx)	(C_EE_BASE + C_EE_CFG_EQ_SW_STATE_OFFSET(idx))
#define C_EE_CFG_EQ_SW_STATE_SIZE	0x00004000

union c_ee_cfg_eq_sw_state {
	uint64_t qw;
	struct {
		uint64_t rd_ptr              : 26;
		uint64_t                     :  5;
		uint64_t reading_buffer_b    :  1;
		uint64_t event_int_disable   :  1;
		uint64_t eq_sts_int_disable  :  1;
		uint64_t                     :  6;
		uint64_t event_drop_seq_no   :  1;
		uint64_t                     :  3;
		uint64_t                     :  4;
		uint64_t                     : 16;
	};
};

#define C_EE_CFG_STS_EQ_HW_STATE_OFFSET(idx)	(0x00008000+((idx)*16))
#define C_EE_CFG_STS_EQ_HW_STATE_ENTRIES	2048
#define C_EE_CFG_STS_EQ_HW_STATE(idx)	(C_EE_BASE + C_EE_CFG_STS_EQ_HW_STATE_OFFSET(idx))
#define C_EE_CFG_STS_EQ_HW_STATE_SIZE	0x00008000

union c_ee_cfg_sts_eq_hw_state {
	uint64_t qw[2];
	struct {
		uint64_t wr_ptr             : 26;
		uint64_t                    :  5;
		uint64_t writing_buffer_b   :  1;
		uint64_t pending_events     : 26;
		uint64_t                    :  6;
		uint64_t ecb_idx            :  8;
		uint64_t ecb_state          :  3;
		uint64_t                    :  5;
		uint64_t eq_switch_dfrd     :  1;
		uint64_t eq_switch_pending  :  1;
		uint64_t prefetch_pending   :  1;
		uint64_t event_int_blkd     :  1;
		uint64_t event_dropped      :  1;
		uint64_t event_drop_seq_no  :  1;
		uint64_t                    : 10;
		uint64_t                    :  2;
		uint64_t                    : 30;
	};
};

#define C_EE_CFG_LONG_EVNT_OVR_TABLE_OFFSET(idx)	(0x00001000+((idx)*8))
#define C_EE_CFG_LONG_EVNT_OVR_TABLE_ENTRIES	64
#define C_EE_CFG_LONG_EVNT_OVR_TABLE(idx)	(C_EE_BASE + C_EE_CFG_LONG_EVNT_OVR_TABLE_OFFSET(idx))
#define C_EE_CFG_LONG_EVNT_OVR_TABLE_SIZE	0x00000200

union c_ee_cfg_long_evnt_ovr_table {
	uint64_t qw;
};

#define C_EE_CFG_PERIODIC_TSTAMP_TABLE_OFFSET(idx)	(0x00001200+((idx)*8))
#define C_EE_CFG_PERIODIC_TSTAMP_TABLE_ENTRIES	32
#define C_EE_CFG_PERIODIC_TSTAMP_TABLE(idx)	(C_EE_BASE + C_EE_CFG_PERIODIC_TSTAMP_TABLE_OFFSET(idx))
#define C_EE_CFG_PERIODIC_TSTAMP_TABLE_SIZE	0x00000100

union c_ee_cfg_periodic_tstamp_table {
	uint64_t qw;
	struct {
		uint64_t n63_n0_enable_periodic_tstamp;
	};
};

#define C_EE_STS_INIT_DONE_OFFSET	0x00000828
#define C_EE_STS_INIT_DONE	(C_EE_BASE + C_EE_STS_INIT_DONE_OFFSET)
#define C_EE_STS_INIT_DONE_SIZE	0x00000008

union c_ee_sts_init_done {
	uint64_t qw;
	struct {
		uint64_t init_done   :  1;
		uint64_t warm_reset  :  1;
		uint64_t             : 62;
	};
};

#define C_EE_DBG_ECB_SIDEBAND_OFFSET(idx)	(0x00001800+((idx)*8))
#define C_EE_DBG_ECB_SIDEBAND_ENTRIES	256
#define C_EE_DBG_ECB_SIDEBAND(idx)	(C_EE_BASE + C_EE_DBG_ECB_SIDEBAND_OFFSET(idx))
#define C_EE_DBG_ECB_SIDEBAND_SIZE	0x00000800

union c_ee_dbg_ecb_sideband {
	uint64_t qw;
	struct {
		uint64_t               :  6;
		uint64_t addr_11_6     :  6;
		uint64_t               :  4;
		uint64_t acid          : 10;
		uint64_t               :  4;
		uint64_t trnsltn       :  1;
		uint64_t data_present  :  1;
		uint64_t eq_handle     : 11;
		uint64_t               :  5;
		uint64_t int_idx       : 11;
		uint64_t               :  2;
		uint64_t int_present   :  1;
		uint64_t block_tarb    :  1;
		uint64_t               :  1;
	};
};

#define C_EE_DBG_ECB_OFFSET(idx)	(0x00010000+((idx)*64))
#define C_EE_DBG_ECB_ENTRIES	256
#define C_EE_DBG_ECB(idx)	(C_EE_BASE + C_EE_DBG_ECB_OFFSET(idx))
#define C_EE_DBG_ECB_SIZE	0x00004000

union c_ee_dbg_ecb {
	uint64_t qw[8];
};

#define C_EE_DBG_TRNSLTN_RSP_OFFSET(idx)	(0x00002000+((idx)*8))
#define C_EE_DBG_TRNSLTN_RSP_ENTRIES	256
#define C_EE_DBG_TRNSLTN_RSP(idx)	(C_EE_BASE + C_EE_DBG_TRNSLTN_RSP_OFFSET(idx))
#define C_EE_DBG_TRNSLTN_RSP_SIZE	0x00000800

union c_ee_dbg_trnsltn_rsp {
	uint64_t qw;
	struct {
		uint64_t return_code  :  6;
		uint64_t              :  2;
		uint64_t at_type      :  1;
		uint64_t nsa          :  1;
		uint64_t at_epoch     :  1;
		uint64_t              :  1;
		uint64_t addr_pg      : 45;
		uint64_t              :  4;
		uint64_t              :  3;
	};
};

#define C_EE_DBG_LM_CB_TO_EQ_MAP_OFFSET(idx)	(0x00002800+((idx)*8))
#define C_EE_DBG_LM_CB_TO_EQ_MAP_ENTRIES	256
#define C_EE_DBG_LM_CB_TO_EQ_MAP(idx)	(C_EE_BASE + C_EE_DBG_LM_CB_TO_EQ_MAP_OFFSET(idx))
#define C_EE_DBG_LM_CB_TO_EQ_MAP_SIZE	0x00000800

union c_ee_dbg_lm_cb_to_eq_map {
	uint64_t qw;
	struct {
		uint64_t eq_handle  : 11;
		uint64_t            : 53;
	};
};

#define C_HNI_CFG_CRNC_OFFSET	0x00000000
#define C_HNI_CFG_CRNC	(C_HNI_BASE + C_HNI_CFG_CRNC_OFFSET)
#define C_HNI_CFG_CRNC_SIZE	0x00000008

union c_hni_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_HNI_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_HNI_CFG_RT_OFFSET	(C_HNI_BASE + C_HNI_CFG_RT_OFFSET_OFFSET)
#define C_HNI_CFG_RT_OFFSET_SIZE	0x00000008

union c_hni_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_HNI_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_HNI_MSC_SHADOW_ACTION	(C_HNI_BASE + C_HNI_MSC_SHADOW_ACTION_OFFSET)
#define C_HNI_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_HNI_MSC_SHADOW_OFFSET	0x00000040
#define C_HNI_MSC_SHADOW	(C_HNI_BASE + C_HNI_MSC_SHADOW_OFFSET)
#define C_HNI_MSC_SHADOW_SIZE	0x00000008

union c_hni_msc_shadow {
	uint64_t qw[1];
};

#define C_HNI_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_HNI_ERR_ELAPSED_TIME	(C_HNI_BASE + C_HNI_ERR_ELAPSED_TIME_OFFSET)
#define C_HNI_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_hni_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C1_HNI_ERR_FLG_OFFSET	0x00000108
#define C1_HNI_ERR_FLG	(C_HNI_BASE + C1_HNI_ERR_FLG_OFFSET)
#define C1_HNI_ERR_FLG_SIZE	0x00000008

union c1_hni_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          :  1;
		uint64_t event_cnt_ovf            :  1;
		uint64_t llr_flit_cor             :  1;
		uint64_t rx_stat_fifo_cor         :  1;
		uint64_t rx_pkt_fifo_cor          :  1;
		uint64_t timestamp_cor            :  1;
		uint64_t hrp_fifo_cor             :  1;
		uint64_t fgfc_fifo_cor            :  1;
		uint64_t tx_flit_cor              :  1;
		uint64_t event_ram_cor            :  1;
		uint64_t llr_flit_ucor            :  1;
		uint64_t rx_stat_fifo_ucor        :  1;
		uint64_t rx_pkt_fifo_ucor         :  1;
		uint64_t timestamp_ucor           :  1;
		uint64_t hrp_fifo_ucor            :  1;
		uint64_t fgfc_fifo_ucor           :  1;
		uint64_t tx_flit_ucor             :  1;
		uint64_t event_ram_ucor           :  1;
		uint64_t fifo_err                 :  1;
		uint64_t llr_chksum_bad           :  1;
		uint64_t llr_fcs_bad              :  1;
		uint64_t llr_ctrl_perr            :  1;
		uint64_t llr_exceed_mfs           :  1;
		uint64_t llr_eopb                 :  1;
		uint64_t rx_ctrl_perr             :  1;
		uint64_t pause_timeout            :  1;
		uint64_t fgfc_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t tx_ctrl_perr             :  1;
		uint64_t tx_size_err              :  1;
		uint64_t rx_pause_err             :  1;
		uint64_t pfc_fifo_oflw            :  8;
		uint64_t discard_err              :  8;
		uint64_t disable_err              :  8;
	};
};

#define C2_HNI_ERR_FLG_OFFSET	0x00000108
#define C2_HNI_ERR_FLG	(C_HNI_BASE + C2_HNI_ERR_FLG_OFFSET)
#define C2_HNI_ERR_FLG_SIZE	0x00000008

union c2_hni_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          :  1;
		uint64_t event_cnt_ovf            :  1;
		uint64_t llr_flit_cor             :  1;
		uint64_t rx_stat_fifo_cor         :  1;
		uint64_t rx_pkt_fifo_cor          :  1;
		uint64_t timestamp_cor            :  1;
		uint64_t hrp_fifo_cor             :  1;
		uint64_t fgfc_fifo_cor            :  1;
		uint64_t tx_flit_cor              :  1;
		uint64_t event_ram_cor            :  1;
		uint64_t ixe_flit_cor             :  1;
		uint64_t llr_flit_ucor            :  1;
		uint64_t rx_stat_fifo_ucor        :  1;
		uint64_t rx_pkt_fifo_ucor         :  1;
		uint64_t timestamp_ucor           :  1;
		uint64_t hrp_fifo_ucor            :  1;
		uint64_t fgfc_fifo_ucor           :  1;
		uint64_t tx_flit_ucor             :  1;
		uint64_t event_ram_ucor           :  1;
		uint64_t ixe_flit_ucor            :  1;
		uint64_t fifo_err                 :  1;
		uint64_t llr_chksum_bad           :  1;
		uint64_t llr_fcs_bad              :  1;
		uint64_t llr_ctrl_perr            :  1;
		uint64_t llr_exceed_mfs           :  1;
		uint64_t llr_eopb                 :  1;
		uint64_t pause_timeout            :  1;
		uint64_t fgfc_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t tx_size_err              :  1;
		uint64_t rx_pause_err             :  1;
		uint64_t pfc_fifo_oflw            :  8;
		uint64_t discard_err              :  8;
		uint64_t disable_err              :  8;
	};
};

#define C_HNI_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_HNI_ERR_FIRST_FLG	(C_HNI_BASE + C_HNI_ERR_FIRST_FLG_OFFSET)
#define C_HNI_ERR_FIRST_FLG_SIZE	0x00000008
#define C_HNI_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_HNI_ERR_FIRST_FLG_TS	(C_HNI_BASE + C_HNI_ERR_FIRST_FLG_TS_OFFSET)
#define C_HNI_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_HNI_ERR_CLR_OFFSET	0x00000120
#define C_HNI_ERR_CLR	(C_HNI_BASE + C_HNI_ERR_CLR_OFFSET)
#define C_HNI_ERR_CLR_SIZE	0x00000008
#define C_HNI_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_HNI_ERR_IRQA_MSK	(C_HNI_BASE + C_HNI_ERR_IRQA_MSK_OFFSET)
#define C_HNI_ERR_IRQA_MSK_SIZE	0x00000008
#define C_HNI_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_HNI_ERR_IRQB_MSK	(C_HNI_BASE + C_HNI_ERR_IRQB_MSK_OFFSET)
#define C_HNI_ERR_IRQB_MSK_SIZE	0x00000008
#define C_HNI_ERR_INFO_MSK_OFFSET	0x00000140
#define C_HNI_ERR_INFO_MSK	(C_HNI_BASE + C_HNI_ERR_INFO_MSK_OFFSET)
#define C_HNI_ERR_INFO_MSK_SIZE	0x00000008
#define C1_HNI_ERR_INFO_ECC_OFFSET	0x00000180
#define C1_HNI_ERR_INFO_ECC	(C_HNI_BASE + C1_HNI_ERR_INFO_ECC_OFFSET)
#define C1_HNI_ERR_INFO_ECC_SIZE	0x00000008

union c1_hni_err_info_ecc {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        :  9;
		uint64_t                    :  3;
		uint64_t cor_id             :  3;
		uint64_t                    :  4;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       :  9;
		uint64_t                    :  3;
		uint64_t ucor_id            :  3;
		uint64_t                    :  4;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C2_HNI_ERR_INFO_ECC_OFFSET	0x00000188
#define C2_HNI_ERR_INFO_ECC	(C_HNI_BASE + C2_HNI_ERR_INFO_ECC_OFFSET)
#define C2_HNI_ERR_INFO_ECC_SIZE	0x00000008

union c2_hni_err_info_ecc {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        :  9;
		uint64_t                    :  3;
		uint64_t cor_id             :  4;
		uint64_t                    :  3;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       :  9;
		uint64_t                    :  3;
		uint64_t ucor_id            :  4;
		uint64_t                    :  3;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C1_HNI_ERR_INFO_FIFO_OFFSET	0x00000188
#define C1_HNI_ERR_INFO_FIFO	(C_HNI_BASE + C1_HNI_ERR_INFO_FIFO_OFFSET)
#define C_HNI_ERR_INFO_FIFO_SIZE	0x00000008
#define C1_HNI_ERR_INFO_FIFO_SIZE	C_HNI_ERR_INFO_FIFO_SIZE

union c_hni_err_info_fifo {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           :  2;
		uint64_t fifo_id   :  3;
		uint64_t           : 57;
	};
};

#define C2_HNI_ERR_INFO_FIFO_OFFSET	0x00000190
#define C2_HNI_ERR_INFO_FIFO	(C_HNI_BASE + C2_HNI_ERR_INFO_FIFO_OFFSET)
#define C2_HNI_ERR_INFO_FIFO_SIZE	C_HNI_ERR_INFO_FIFO_SIZE

#define C1_HNI_ERR_INFO_CC_OFFSET	0x00000190
#define C1_HNI_ERR_INFO_CC	(C_HNI_BASE + C1_HNI_ERR_INFO_CC_OFFSET)
#define C_HNI_ERR_INFO_CC_SIZE	0x00000008
#define C1_HNI_ERR_INFO_CC_SIZE	C_HNI_ERR_INFO_CC_SIZE

union c_hni_err_info_cc {
	uint64_t qw;
	struct {
		uint64_t cc_id  :  3;
		uint64_t        :  1;
		uint64_t cc_tc  :  3;
		uint64_t        : 57;
	};
};

#define C2_HNI_ERR_INFO_CC_OFFSET	0x00000198
#define C2_HNI_ERR_INFO_CC	(C_HNI_BASE + C2_HNI_ERR_INFO_CC_OFFSET)
#define C2_HNI_ERR_INFO_CC_SIZE	C_HNI_ERR_INFO_CC_SIZE

#define C1_HNI_ERR_INFO_FGFC_OFFSET	0x00000198
#define C1_HNI_ERR_INFO_FGFC	(C_HNI_BASE + C1_HNI_ERR_INFO_FGFC_OFFSET)
#define C_HNI_ERR_INFO_FGFC_SIZE	0x00000008
#define C1_HNI_ERR_INFO_FGFC_SIZE	C_HNI_ERR_INFO_FGFC_SIZE

union c_hni_err_info_fgfc {
	uint64_t qw;
	struct {
		uint64_t invalid_type  :  1;
		uint64_t no_match      :  1;
		uint64_t cache_full    :  1;
		uint64_t fifo_full     :  1;
		uint64_t               : 60;
	};
};

#define C2_HNI_ERR_INFO_FGFC_OFFSET	0x000001a0
#define C2_HNI_ERR_INFO_FGFC	(C_HNI_BASE + C2_HNI_ERR_INFO_FGFC_OFFSET)
#define C2_HNI_ERR_INFO_FGFC_SIZE	C_HNI_ERR_INFO_FGFC_SIZE

#define C1_HNI_ERR_INFO_SIZE_ERR_OFFSET	0x000001a0
#define C1_HNI_ERR_INFO_SIZE_ERR	(C_HNI_BASE + C1_HNI_ERR_INFO_SIZE_ERR_OFFSET)
#define C_HNI_ERR_INFO_SIZE_ERR_SIZE	0x00000008
#define C1_HNI_ERR_INFO_SIZE_ERR_SIZE	C_HNI_ERR_INFO_SIZE_ERR_SIZE

union c_hni_err_info_size_err {
	uint64_t qw;
	struct {
		uint64_t opt_too_small  :  1;
		uint64_t std_too_small  :  1;
		uint64_t opt_too_big    :  1;
		uint64_t std_too_big    :  1;
		uint64_t                : 60;
	};
};

#define C2_HNI_ERR_INFO_SIZE_ERR_OFFSET	0x000001a8
#define C2_HNI_ERR_INFO_SIZE_ERR	(C_HNI_BASE + C2_HNI_ERR_INFO_SIZE_ERR_OFFSET)
#define C2_HNI_ERR_INFO_SIZE_ERR_SIZE	C_HNI_ERR_INFO_SIZE_ERR_SIZE

#define C1_HNI_ERR_INFO_DISCARD_OFFSET	0x00000200
#define C1_HNI_ERR_INFO_DISCARD	(C_HNI_BASE + C1_HNI_ERR_INFO_DISCARD_OFFSET)
#define C_HNI_ERR_INFO_DISCARD_SIZE	0x00000018
#define C1_HNI_ERR_INFO_DISCARD_SIZE	C_HNI_ERR_INFO_DISCARD_SIZE

union c_hni_err_info_discard {
	uint64_t qw[3];
	struct {
		uint64_t addr_low;
		uint64_t addr_high;
		uint64_t tc          :  3;
		uint64_t             :  1;
		uint64_t pkt_type    :  2;
		uint64_t             : 58;
	};
};

#define C2_HNI_ERR_INFO_DISCARD_OFFSET	0x000001b0
#define C2_HNI_ERR_INFO_DISCARD	(C_HNI_BASE + C2_HNI_ERR_INFO_DISCARD_OFFSET)
#define C2_HNI_ERR_INFO_DISCARD_SIZE	C_HNI_ERR_INFO_DISCARD_SIZE

#define C_HNI_CFG_GEN_OFFSET	0x00000400
#define C_HNI_CFG_GEN	(C_HNI_BASE + C_HNI_CFG_GEN_OFFSET)
#define C_HNI_CFG_GEN_SIZE	0x00000008

union c_hni_cfg_gen {
	uint64_t qw;
	struct {
		uint64_t tx_timestamp_en  :  1;
		uint64_t rx_timestamp_en  :  1;
		uint64_t timestamp_fmt    :  2;
		uint64_t default_pcp      :  3;
		uint64_t ip_enable_dscp   :  1;
		uint64_t default_dscp     :  6;
		uint64_t                  : 50;
	};
};

#define C_HNI_CFG_CDT_OFFSET	0x00000408
#define C_HNI_CFG_CDT	(C_HNI_BASE + C_HNI_CFG_CDT_OFFSET)
#define C_HNI_CFG_CDT_SIZE	0x00000008

union c_hni_cfg_cdt {
	uint64_t qw;
	struct {
		uint64_t ixe_cdts       :  6;
		uint64_t                :  2;
		uint64_t hrp_cdts       :  5;
		uint64_t                :  3;
		uint64_t llr_cdts       :  4;
		uint64_t                :  4;
		uint64_t ee_cdts        :  5;
		uint64_t                :  3;
		uint64_t ixe_pbuf_size  : 13;
		uint64_t                : 19;
	};
};

#define C_HNI_CFG_TXTS_L2_MASK_OFFSET	0x00000430
#define C_HNI_CFG_TXTS_L2_MASK	(C_HNI_BASE + C_HNI_CFG_TXTS_L2_MASK_OFFSET)
#define C_HNI_CFG_TXTS_L2_MASK_SIZE	0x00000008

union c_hni_cfg_txts_l2_mask {
	uint64_t qw;
	struct {
		uint64_t dmac       : 48;
		uint64_t ethertype  : 16;
	};
};

#define C_HNI_CFG_TXTS_L2_MATCH_OFFSET	0x00000438
#define C_HNI_CFG_TXTS_L2_MATCH	(C_HNI_BASE + C_HNI_CFG_TXTS_L2_MATCH_OFFSET)
#define C_HNI_CFG_TXTS_L2_MATCH_SIZE	0x00000008

union c_hni_cfg_txts_l2_match {
	uint64_t qw;
	struct {
		uint64_t dmac       : 48;
		uint64_t ethertype  : 16;
	};
};

#define C_HNI_CFG_TXTS_L3_MASK_OFFSET	0x00000440
#define C_HNI_CFG_TXTS_L3_MASK	(C_HNI_BASE + C_HNI_CFG_TXTS_L3_MASK_OFFSET)
#define C_HNI_CFG_TXTS_L3_MASK_SIZE	0x00000008

union c_hni_cfg_txts_l3_mask {
	uint64_t qw;
	struct {
		uint64_t hdr;
	};
};

#define C_HNI_CFG_TXTS_L3_MATCH_OFFSET	0x00000448
#define C_HNI_CFG_TXTS_L3_MATCH	(C_HNI_BASE + C_HNI_CFG_TXTS_L3_MATCH_OFFSET)
#define C_HNI_CFG_TXTS_L3_MATCH_SIZE	0x00000008

union c_hni_cfg_txts_l3_match {
	uint64_t qw;
	struct {
		uint64_t hdr;
	};
};

#define C_HNI_CFG_RXTS_L2_MASK_OFFSET	0x00000450
#define C_HNI_CFG_RXTS_L2_MASK	(C_HNI_BASE + C_HNI_CFG_RXTS_L2_MASK_OFFSET)
#define C_HNI_CFG_RXTS_L2_MASK_SIZE	0x00000008

union c_hni_cfg_rxts_l2_mask {
	uint64_t qw;
	struct {
		uint64_t dmac       : 48;
		uint64_t ethertype  : 16;
	};
};

#define C_HNI_CFG_RXTS_L2_MATCH_OFFSET	0x00000458
#define C_HNI_CFG_RXTS_L2_MATCH	(C_HNI_BASE + C_HNI_CFG_RXTS_L2_MATCH_OFFSET)
#define C_HNI_CFG_RXTS_L2_MATCH_SIZE	0x00000008

union c_hni_cfg_rxts_l2_match {
	uint64_t qw;
	struct {
		uint64_t dmac       : 48;
		uint64_t ethertype  : 16;
	};
};

#define C_HNI_CFG_RXTS_L3_MASK_OFFSET	0x00000460
#define C_HNI_CFG_RXTS_L3_MASK	(C_HNI_BASE + C_HNI_CFG_RXTS_L3_MASK_OFFSET)
#define C_HNI_CFG_RXTS_L3_MASK_SIZE	0x00000008

union c_hni_cfg_rxts_l3_mask {
	uint64_t qw;
	struct {
		uint64_t hdr;
	};
};

#define C_HNI_CFG_RXTS_L3_MATCH_OFFSET	0x00000468
#define C_HNI_CFG_RXTS_L3_MATCH	(C_HNI_BASE + C_HNI_CFG_RXTS_L3_MATCH_OFFSET)
#define C_HNI_CFG_RXTS_L3_MATCH_SIZE	0x00000008

union c_hni_cfg_rxts_l3_match {
	uint64_t qw;
	struct {
		uint64_t hdr;
	};
};

#define C_HNI_CFG_PAUSE_QUANTA_OFFSET	0x00000470
#define C_HNI_CFG_PAUSE_QUANTA	(C_HNI_BASE + C_HNI_CFG_PAUSE_QUANTA_OFFSET)
#define C_HNI_CFG_PAUSE_QUANTA_SIZE	0x00000008

union c_hni_cfg_pause_quanta {
	uint64_t qw;
	struct {
		uint64_t sub_value   : 24;
		uint64_t sub_period  :  8;
		uint64_t             : 32;
	};
};

#define C_HNI_CFG_PAUSE_TIMING_OFFSET	0x00000478
#define C_HNI_CFG_PAUSE_TIMING	(C_HNI_BASE + C_HNI_CFG_PAUSE_TIMING_OFFSET)
#define C_HNI_CFG_PAUSE_TIMING_SIZE	0x00000008

union c_hni_cfg_pause_timing {
	uint64_t qw;
	struct {
		uint64_t pause_period         : 16;
		uint64_t pause_repeat_period  : 16;
		uint64_t                      : 32;
	};
};

#define C_HNI_CFG_PAUSE_TX_CTRL_OFFSET	0x00000490
#define C_HNI_CFG_PAUSE_TX_CTRL	(C_HNI_BASE + C_HNI_CFG_PAUSE_TX_CTRL_OFFSET)
#define C_HNI_CFG_PAUSE_TX_CTRL_SIZE	0x00000008

union c_hni_cfg_pause_tx_ctrl {
	uint64_t qw;
	struct {
		uint64_t mac_cntl_opcode    : 16;
		uint64_t mac_cntl_type      : 16;
		uint64_t enable_send_pause  :  8;
		uint64_t pev                :  8;
		uint64_t enable_pause       :  1;
		uint64_t enable_pfc_pause   :  1;
		uint64_t                    : 14;
	};
};

#define C_HNI_CFG_PAUSE_RX_CTRL_OFFSET	0x00000498
#define C_HNI_CFG_PAUSE_RX_CTRL	(C_HNI_BASE + C_HNI_CFG_PAUSE_RX_CTRL_OFFSET)
#define C_HNI_CFG_PAUSE_RX_CTRL_SIZE	0x00000008

union c_hni_cfg_pause_rx_ctrl {
	uint64_t qw;
	struct {
		uint64_t enable_rec_pause  :  8;
		uint64_t pause_rec_enable  :  1;
		uint64_t pfc_rec_enable    :  1;
		uint64_t                   :  2;
		uint64_t timer_bit         :  5;
		uint64_t                   : 47;
	};
};

#define C_HNI_CFG_MAX_SIZES_OFFSET	0x000004a8
#define C_HNI_CFG_MAX_SIZES	(C_HNI_BASE + C_HNI_CFG_MAX_SIZES_OFFSET)
#define C_HNI_CFG_MAX_SIZES_SIZE	0x00000008

union c_hni_cfg_max_sizes {
	uint64_t qw;
	struct {
		uint64_t max_std_size      : 14;
		uint64_t                   :  2;
		uint64_t max_opt_size      : 14;
		uint64_t                   :  2;
		uint64_t max_portals_size  : 14;
		uint64_t                   : 18;
	};
};

#define C_HNI_CFG_EVENT_CNTS_OFFSET	0x000004b8
#define C_HNI_CFG_EVENT_CNTS	(C_HNI_BASE + C_HNI_CFG_EVENT_CNTS_OFFSET)
#define C_HNI_CFG_EVENT_CNTS_SIZE	0x00000008

union c_hni_cfg_event_cnts {
	uint64_t qw;
	struct {
		uint64_t enable     :  1;
		uint64_t init       :  1;
		uint64_t init_done  :  1;
		uint64_t            :  1;
		uint64_t ovf_cnt    : 16;
		uint64_t            : 44;
	};
};

#define C_HNI_CFG_PBUF_OFFSET(idx)	(0x00000500+((idx)*8))
#define C_HNI_CFG_PBUF_ENTRIES	8
#define C_HNI_CFG_PBUF(idx)	(C_HNI_BASE + C_HNI_CFG_PBUF_OFFSET(idx))
#define C_HNI_CFG_PBUF_SIZE	0x00000040

union c_hni_cfg_pbuf {
	uint64_t qw;
	struct {
		uint64_t static_rsvd  : 12;
		uint64_t              :  4;
		uint64_t mtu          :  7;
		uint64_t              : 41;
	};
};

#define C1_HNI_CFG_PFC_BUF_OFFSET(idx)	(0x00000580+((idx)*8))
#define C1_HNI_CFG_PFC_BUF_ENTRIES	8
#define C1_HNI_CFG_PFC_BUF(idx)	(C_HNI_BASE + C1_HNI_CFG_PFC_BUF_OFFSET(idx))
#define C1_HNI_CFG_PFC_BUF_SIZE	0x00000040

union c1_hni_cfg_pfc_buf {
	uint64_t qw;
	struct {
		uint64_t low_water    : 11;
		uint64_t              :  1;
		uint64_t high_water   : 11;
		uint64_t              :  1;
		uint64_t timer_bit    :  5;
		uint64_t              :  3;
		uint64_t discard_max  :  3;
		uint64_t              :  1;
		uint64_t resume       :  1;
		uint64_t              : 27;
	};
};

#define C2_HNI_CFG_PFC_BUF_OFFSET(idx)	(0x00000580+((idx)*8))
#define C2_HNI_CFG_PFC_BUF_ENTRIES	8
#define C2_HNI_CFG_PFC_BUF(idx)	(C_HNI_BASE + C2_HNI_CFG_PFC_BUF_OFFSET(idx))
#define C2_HNI_CFG_PFC_BUF_SIZE	0x00000040

union c2_hni_cfg_pfc_buf {
	uint64_t qw;
	struct {
		uint64_t low_water    : 12;
		uint64_t high_water   : 12;
		uint64_t timer_bit    :  5;
		uint64_t              :  3;
		uint64_t discard_max  :  3;
		uint64_t              :  1;
		uint64_t resume       :  1;
		uint64_t              : 27;
	};
};

#define C_HNI_CFG_DSCP_PCP_OFFSET(idx)	(0x00000800+((idx)*8))
#define C_HNI_CFG_DSCP_PCP_ENTRIES	64
#define C_HNI_CFG_DSCP_PCP(idx)	(C_HNI_BASE + C_HNI_CFG_DSCP_PCP_OFFSET(idx))
#define C_HNI_CFG_DSCP_PCP_SIZE	0x00000200

union c_hni_cfg_dscp_pcp {
	uint64_t qw;
	struct {
		uint64_t pcp  :  3;
		uint64_t      : 61;
	};
};

#define C_HNI_STS_CDT_OFFSET	0x00001000
#define C_HNI_STS_CDT	(C_HNI_BASE + C_HNI_STS_CDT_OFFSET)
#define C_HNI_STS_CDT_SIZE	0x00000008

union c_hni_sts_cdt {
	uint64_t qw;
	struct {
		uint64_t ixe_cdts  :  6;
		uint64_t           :  2;
		uint64_t hrp_cdts  :  5;
		uint64_t           :  3;
		uint64_t llr_cdts  :  4;
		uint64_t           :  4;
		uint64_t ee_cdts   :  5;
		uint64_t           :  3;
		uint64_t buf_cdts  : 13;
		uint64_t           : 19;
	};
};

#define C_HNI_STS_PAUSE_TIMEOUT_OFFSET	0x00001008
#define C_HNI_STS_PAUSE_TIMEOUT	(C_HNI_BASE + C_HNI_STS_PAUSE_TIMEOUT_OFFSET)
#define C_HNI_STS_PAUSE_TIMEOUT_SIZE	0x00000008

union c_hni_sts_pause_timeout {
	uint64_t qw;
	struct {
		uint64_t pfc_0  :  4;
		uint64_t pfc_1  :  4;
		uint64_t pfc_2  :  4;
		uint64_t pfc_3  :  4;
		uint64_t pfc_4  :  4;
		uint64_t pfc_5  :  4;
		uint64_t pfc_6  :  4;
		uint64_t pfc_7  :  4;
		uint64_t        : 32;
	};
};

#define C_HNI_STS_EVENT_CNTS_OFFSET(idx)	(0x00002000+((idx)*8))
#ifndef C_HNI_STS_EVENT_CNTS_ENTRIES
#define C_HNI_STS_EVENT_CNTS_ENTRIES	512
#endif
#define C_HNI_STS_EVENT_CNTS(idx)	(C_HNI_BASE + C_HNI_STS_EVENT_CNTS_OFFSET(idx))
#define C_HNI_STS_EVENT_CNTS_SIZE	0x00001000

union c_hni_sts_event_cnts {
	uint64_t qw;
	struct {
		uint64_t cnt  : 56;
		uint64_t      :  8;
	};
};

#define C_HNI_UC_SBR_INTR_OFFSET	0x00004310
#define C_HNI_UC_SBR_INTR	(C_HNI_BASE + C_HNI_UC_SBR_INTR_OFFSET)
#define C_HNI_UC_SBR_INTR_SIZE	0x00000008

union c_hni_uc_sbr_intr {
	uint64_t qw;
	struct {
		uint64_t intr          :  8;
		uint64_t send_sbr      :  1;
		uint64_t send_sbr_dis  :  1;
		uint64_t               : 54;
	};
};

#define C_HNI_PML_CFG_CRNC_OFFSET	0x00000000
#define C_HNI_PML_CFG_CRNC	(C_HNI_PML_BASE + C_HNI_PML_CFG_CRNC_OFFSET)
#define C_HNI_PML_CFG_CRNC_SIZE	0x00000008

union c_hni_pml_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_HNI_PML_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_HNI_PML_ERR_ELAPSED_TIME	(C_HNI_PML_BASE + C_HNI_PML_ERR_ELAPSED_TIME_OFFSET)
#define C_HNI_PML_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_hni_pml_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C1_HNI_PML_ERR_FLG_OFFSET	0x00000108
#define C1_HNI_PML_ERR_FLG	(C_HNI_PML_BASE + C1_HNI_PML_ERR_FLG_OFFSET)
#define C1_HNI_PML_ERR_FLG_SIZE	0x00000008

union c1_hni_pml_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          : 22;
		uint64_t mac_rx_frame_err         :  1;
		uint64_t autoneg_page_received    :  1;
		uint64_t autoneg_complete         :  1;
		uint64_t autoneg_failed           :  1;
		uint64_t pcs_tx_dp_err            :  1;
		uint64_t pcs_rx_dp_err            :  1;
		uint64_t pcs_rx_deskew_overflow   :  1;
		uint64_t pcs_rx_deskew_pe         :  1;
		uint64_t pcs_fec_mem0_sbe         :  1;
		uint64_t pcs_fec_mem0_mbe         :  1;
		uint64_t pcs_fec_mem1_sbe         :  1;
		uint64_t pcs_fec_mem1_mbe         :  1;
		uint64_t pcs_tx_degrade           :  1;
		uint64_t pcs_rx_degrade           :  1;
		uint64_t pcs_rx_degrade_failure   :  1;
		uint64_t pcs_tx_degrade_failure   :  1;
		uint64_t pcs_hi_ser               :  1;
		uint64_t pcs_link_down            :  1;
		uint64_t mac_tx_dp_err            :  1;
		uint64_t mac_rx_dp_err            :  1;
		uint64_t llr_tx_dp_sbe            :  1;
		uint64_t llr_tx_dp_mbe            :  1;
		uint64_t llr_tx_dp_err            :  1;
		uint64_t llr_seq_mem_sbe          :  1;
		uint64_t llr_seq_mem_mbe          :  1;
		uint64_t llr_replay_mem_sbe       :  1;
		uint64_t llr_replay_mem_mbe       :  1;
		uint64_t llr_ack_nack_error       :  1;
		uint64_t llr_ack_nack_seq_err     :  1;
		uint64_t llr_replay_at_max        :  1;
		uint64_t pcs_lanes_locked         :  1;
		uint64_t pcs_aligned              :  1;
		uint64_t pcs_ready                :  1;
	};
};

#define SS2_PORT_PML_ERR_FLG_OFFSET	0x00000120
#define SS2_PORT_PML_ERR_FLG	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_FLG_OFFSET)
#define SS2_PORT_PML_ERR_FLG_SIZE	0x00000020

union ss2_port_pml_err_flg {
	uint64_t qw[4];
	struct {
		uint64_t sw_diag                     :  1;
		uint64_t crnc_ring_sync_error        :  1;
		uint64_t crnc_ring_ecc_sbe           :  1;
		uint64_t crnc_ring_ecc_mbe           :  1;
		uint64_t crnc_ring_cmd_unknown       :  1;
		uint64_t crnc_csr_cmd_unknown        :  1;
		uint64_t crnc_csr_cmd_incomplete     :  1;
		uint64_t crnc_buf_ecc_sbe            :  1;
		uint64_t crnc_buf_ecc_mbe            :  1;
		uint64_t csrc_ring_sync_error        :  1;
		uint64_t csrc_ring_ecc_sbe           :  1;
		uint64_t csrc_ring_ecc_mbe           :  1;
		uint64_t csrc_ring_cmd_unknown       :  1;
		uint64_t csrc_sring_sync_error       :  1;
		uint64_t csrc_sring_ecc_sbe          :  1;
		uint64_t csrc_sring_ecc_mbe          :  1;
		uint64_t csrc_sring_cmd_unknown      :  1;
		uint64_t csrc_buf_ecc_sbe            :  1;
		uint64_t csrc_buf_ecc_mbe            :  1;
		uint64_t                             : 13;
		uint64_t pcs_tx_dp_err               :  1;
		uint64_t pcs_rx_dp_err               :  1;
		uint64_t pcs_rx_deskew_overflow      :  1;
		uint64_t pcs_rx_deskew_pe            :  1;
		uint64_t pcs_fec0_mem_sbe            :  1;
		uint64_t pcs_fec0_mem_mbe            :  1;
		uint64_t pcs_fec1_mem_sbe            :  1;
		uint64_t pcs_fec1_mem_mbe            :  1;
		uint64_t mac_tx_dp_err               :  1;
		uint64_t mac_rx_dp_err               :  1;
		uint64_t llr_tx_dp_sbe               :  1;
		uint64_t llr_tx_dp_mbe               :  1;
		uint64_t llr_tx_dp_err               :  1;
		uint64_t llr_rx_dp_err               :  1;
		uint64_t llr_seq_mem_sbe             :  1;
		uint64_t llr_seq_mem_mbe             :  1;
		uint64_t llr_replay_mem_sbe          :  1;
		uint64_t llr_replay_mem_mbe          :  1;
		uint64_t event_mem_sbe               :  1;
		uint64_t event_mem_mbe               :  1;
		uint64_t mac_rx_fcs32_err_0          :  1;
		uint64_t mac_rx_fcs32_err_1          :  1;
		uint64_t mac_rx_fcs32_err_2          :  1;
		uint64_t mac_rx_fcs32_err_3          :  1;
		uint64_t mac_rx_framing_err_0        :  1;
		uint64_t mac_rx_framing_err_1        :  1;
		uint64_t mac_rx_framing_err_2        :  1;
		uint64_t mac_rx_framing_err_3        :  1;
		uint64_t mac_rx_preamble_err_0       :  1;
		uint64_t mac_rx_preamble_err_1       :  1;
		uint64_t mac_rx_preamble_err_2       :  1;
		uint64_t mac_rx_preamble_err_3       :  1;
		uint64_t pcs_link_down_0             :  1;
		uint64_t pcs_link_down_1             :  1;
		uint64_t pcs_link_down_2             :  1;
		uint64_t pcs_link_down_3             :  1;
		uint64_t pcs_link_down_lf_0          :  1;
		uint64_t pcs_link_down_lf_1          :  1;
		uint64_t pcs_link_down_lf_2          :  1;
		uint64_t pcs_link_down_lf_3          :  1;
		uint64_t pcs_link_down_rf_0          :  1;
		uint64_t pcs_link_down_rf_1          :  1;
		uint64_t pcs_link_down_rf_2          :  1;
		uint64_t pcs_link_down_rf_3          :  1;
		uint64_t pcs_hi_ser_0                :  1;
		uint64_t pcs_hi_ser_1                :  1;
		uint64_t pcs_hi_ser_2                :  1;
		uint64_t pcs_hi_ser_3                :  1;
		uint64_t pcs_tx_degrade              :  1;
		uint64_t pcs_rx_degrade              :  1;
		uint64_t pcs_rx_degrade_failure      :  1;
		uint64_t pcs_tx_degrade_failure      :  1;
		uint64_t                             :  8;
		uint64_t autoneg_failed_0            :  1;
		uint64_t autoneg_failed_1            :  1;
		uint64_t autoneg_failed_2            :  1;
		uint64_t autoneg_failed_3            :  1;
		uint64_t llr_ack_nack_error_0        :  1;
		uint64_t llr_ack_nack_error_1        :  1;
		uint64_t llr_ack_nack_error_2        :  1;
		uint64_t llr_ack_nack_error_3        :  1;
		uint64_t llr_ack_nack_seq_err_0      :  1;
		uint64_t llr_ack_nack_seq_err_1      :  1;
		uint64_t llr_ack_nack_seq_err_2      :  1;
		uint64_t llr_ack_nack_seq_err_3      :  1;
		uint64_t llr_replay_at_max_0         :  1;
		uint64_t llr_replay_at_max_1         :  1;
		uint64_t llr_replay_at_max_2         :  1;
		uint64_t llr_replay_at_max_3         :  1;
		uint64_t llr_expected_frame_bad_0    :  1;
		uint64_t llr_expected_frame_bad_1    :  1;
		uint64_t llr_expected_frame_bad_2    :  1;
		uint64_t llr_expected_frame_bad_3    :  1;
		uint64_t llr_duplicate_frame_0       :  1;
		uint64_t llr_duplicate_frame_1       :  1;
		uint64_t llr_duplicate_frame_2       :  1;
		uint64_t llr_duplicate_frame_3       :  1;
		uint64_t llr_unexpected_frame_0      :  1;
		uint64_t llr_unexpected_frame_1      :  1;
		uint64_t llr_unexpected_frame_2      :  1;
		uint64_t llr_unexpected_frame_3      :  1;
		uint64_t llr_loop_time_fail_0        :  1;
		uint64_t llr_loop_time_fail_1        :  1;
		uint64_t llr_loop_time_fail_2        :  1;
		uint64_t llr_loop_time_fail_3        :  1;
		uint64_t llr_init_fail_0             :  1;
		uint64_t llr_init_fail_1             :  1;
		uint64_t llr_init_fail_2             :  1;
		uint64_t llr_init_fail_3             :  1;
		uint64_t llr_starved_0               :  1;
		uint64_t llr_starved_1               :  1;
		uint64_t llr_starved_2               :  1;
		uint64_t llr_starved_3               :  1;
		uint64_t llr_max_starvation_limit_0  :  1;
		uint64_t llr_max_starvation_limit_1  :  1;
		uint64_t llr_max_starvation_limit_2  :  1;
		uint64_t llr_max_starvation_limit_3  :  1;
		uint64_t                             : 56;
		uint64_t pcs_lanes_locked_0          :  1;
		uint64_t pcs_lanes_locked_1          :  1;
		uint64_t pcs_lanes_locked_2          :  1;
		uint64_t pcs_lanes_locked_3          :  1;
		uint64_t pcs_aligned_0               :  1;
		uint64_t pcs_aligned_1               :  1;
		uint64_t pcs_aligned_2               :  1;
		uint64_t pcs_aligned_3               :  1;
		uint64_t pcs_link_up_0               :  1;
		uint64_t pcs_link_up_1               :  1;
		uint64_t pcs_link_up_2               :  1;
		uint64_t pcs_link_up_3               :  1;
		uint64_t pcs_rx_degrade_rdy          :  1;
		uint64_t pcs_lp_degrade_rdy          :  1;
		uint64_t                             :  2;
		uint64_t autoneg_page_received_0     :  1;
		uint64_t autoneg_page_received_1     :  1;
		uint64_t autoneg_page_received_2     :  1;
		uint64_t autoneg_page_received_3     :  1;
		uint64_t autoneg_complete_0          :  1;
		uint64_t autoneg_complete_1          :  1;
		uint64_t autoneg_complete_2          :  1;
		uint64_t autoneg_complete_3          :  1;
		uint64_t llr_loop_time_0             :  1;
		uint64_t llr_loop_time_1             :  1;
		uint64_t llr_loop_time_2             :  1;
		uint64_t llr_loop_time_3             :  1;
		uint64_t time_stamp_valid_0          :  1;
		uint64_t time_stamp_valid_1          :  1;
		uint64_t time_stamp_valid_2          :  1;
		uint64_t time_stamp_valid_3          :  1;
		uint64_t fec_err_vector_valid        :  1;
		uint64_t llr_init_complete_0         :  1;
		uint64_t llr_init_complete_1         :  1;
		uint64_t llr_init_complete_2         :  1;
		uint64_t llr_init_complete_3         :  1;
		uint64_t event_cnt_ovf               :  1;
		uint64_t rx_signal_ok_0              :  1;
		uint64_t rx_signal_ok_1              :  1;
		uint64_t rx_signal_ok_2              :  1;
		uint64_t rx_signal_ok_3              :  1;
		uint64_t                             : 22;
	};
};

#define C1_HNI_PML_ERR_FIRST_FLG_OFFSET	0x00000110
#define C1_HNI_PML_ERR_FIRST_FLG	(C_HNI_PML_BASE + C1_HNI_PML_ERR_FIRST_FLG_OFFSET)
#define C1_HNI_PML_ERR_FIRST_FLG_SIZE	0x00000008
#define SS2_PORT_PML_ERR_FIRST_FLG_OFFSET	0x00000140
#define SS2_PORT_PML_ERR_FIRST_FLG	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_FIRST_FLG_OFFSET)
#define SS2_PORT_PML_ERR_FIRST_FLG_SIZE	0x00000020
#define C1_HNI_PML_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C1_HNI_PML_ERR_FIRST_FLG_TS	(C_HNI_PML_BASE + C1_HNI_PML_ERR_FIRST_FLG_TS_OFFSET)
#define C_HNI_PML_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C1_HNI_PML_ERR_FIRST_FLG_TS_SIZE	C_HNI_PML_ERR_FIRST_FLG_TS_SIZE
#define SS2_PORT_PML_ERR_FIRST_FLG_TS_OFFSET	0x00000160
#define SS2_PORT_PML_ERR_FIRST_FLG_TS	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_FIRST_FLG_TS_OFFSET)
#define SS2_PORT_PML_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C1_HNI_PML_ERR_CLR_OFFSET	0x00000120
#define C1_HNI_PML_ERR_CLR	(C_HNI_PML_BASE + C1_HNI_PML_ERR_CLR_OFFSET)
#define C1_HNI_PML_ERR_CLR_SIZE	0x00000008
#define SS2_PORT_PML_ERR_CLR_OFFSET	0x00000180
#define SS2_PORT_PML_ERR_CLR	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_CLR_OFFSET)
#define SS2_PORT_PML_ERR_CLR_SIZE	0x00000020
#define C1_HNI_PML_ERR_IRQA_MSK_OFFSET	0x00000128
#define C1_HNI_PML_ERR_IRQA_MSK	(C_HNI_PML_BASE + C1_HNI_PML_ERR_IRQA_MSK_OFFSET)
#define C1_HNI_PML_ERR_IRQA_MSK_SIZE	0x00000008
#define SS2_PORT_PML_ERR_IRQA_MSK_OFFSET	0x000001a0
#define SS2_PORT_PML_ERR_IRQA_MSK	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_IRQA_MSK_OFFSET)
#define SS2_PORT_PML_ERR_IRQA_MSK_SIZE	0x00000020
#define C1_HNI_PML_ERR_IRQB_MSK_OFFSET	0x00000130
#define C1_HNI_PML_ERR_IRQB_MSK	(C_HNI_PML_BASE + C1_HNI_PML_ERR_IRQB_MSK_OFFSET)
#define C1_HNI_PML_ERR_IRQB_MSK_SIZE	0x00000008
#define SS2_PORT_PML_ERR_IRQB_MSK_OFFSET	0x000001c0
#define SS2_PORT_PML_ERR_IRQB_MSK	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_IRQB_MSK_OFFSET)
#define SS2_PORT_PML_ERR_IRQB_MSK_SIZE	0x00000020
#define C1_HNI_PML_ERR_INFO_MSK_OFFSET	0x00000140
#define C1_HNI_PML_ERR_INFO_MSK	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_MSK_OFFSET)
#define C1_HNI_PML_ERR_INFO_MSK_SIZE	0x00000008
#define SS2_PORT_PML_ERR_INFO_MSK_OFFSET	0x00000200
#define SS2_PORT_PML_ERR_INFO_MSK	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_MSK_OFFSET)
#define SS2_PORT_PML_ERR_INFO_MSK_SIZE	0x00000020
#define C1_HNI_PML_ERR_INFO_MEM_OFFSET	0x00000180
#define C1_HNI_PML_ERR_INFO_MEM	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_MEM_OFFSET)
#define C1_HNI_PML_ERR_INFO_MEM_SIZE	0x00000008

union c1_hni_pml_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t csr_detected_sbe  :  1;
		uint64_t csr_detected_mbe  :  1;
		uint64_t                   :  2;
		uint64_t sbe_syndrome      : 10;
		uint64_t                   :  2;
		uint64_t sbe_address       : 11;
		uint64_t                   :  1;
		uint64_t sbe_mem_id        :  2;
		uint64_t                   :  6;
		uint64_t mbe_syndrome      : 10;
		uint64_t                   :  2;
		uint64_t mbe_address       : 11;
		uint64_t                   :  1;
		uint64_t mbe_mem_id        :  2;
		uint64_t                   :  2;
	};
};

#define C1_HNI_PML_ERR_INFO_MBE_CNTS_OFFSET	0x00000188
#define C1_HNI_PML_ERR_INFO_MBE_CNTS	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_MBE_CNTS_OFFSET)
#define C1_HNI_PML_ERR_INFO_MBE_CNTS_SIZE	0x00000008

union c1_hni_pml_err_info_mbe_cnts {
	uint64_t qw;
	struct {
		uint64_t pcs_fec0_mbe_cnt    :  4;
		uint64_t                     :  4;
		uint64_t pcs_fec1_mbe_cnt    :  4;
		uint64_t                     :  4;
		uint64_t llr_seq_mbe_cnt     :  4;
		uint64_t                     :  4;
		uint64_t llr_replay_mbe_cnt  :  4;
		uint64_t                     : 36;
	};
};

#define C1_HNI_PML_ERR_INFO_SBE_CNTS_OFFSET	0x00000190
#define C1_HNI_PML_ERR_INFO_SBE_CNTS	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_SBE_CNTS_OFFSET)
#define C1_HNI_PML_ERR_INFO_SBE_CNTS_SIZE	0x00000008

union c1_hni_pml_err_info_sbe_cnts {
	uint64_t qw;
	struct {
		uint64_t pcs_fec0_sbe_cnt    :  8;
		uint64_t pcs_fec1_sbe_cnt    :  8;
		uint64_t llr_seq_sbe_cnt     :  8;
		uint64_t llr_replay_sbe_cnt  :  8;
		uint64_t                     : 32;
	};
};

#define C1_HNI_PML_ERR_INFO_PCS_TX_DP_OFFSET	0x000001a0
#define C1_HNI_PML_ERR_INFO_PCS_TX_DP	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_PCS_TX_DP_OFFSET)
#define C1_HNI_PML_ERR_INFO_PCS_TX_DP_SIZE	0x00000008

union c1_hni_pml_err_info_pcs_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_cdc_underrun             :  4;
		uint64_t tx_cdc_overflow             :  4;
		uint64_t tx_mac_if_fifo_overflow     :  1;
		uint64_t tx_mac_if_fifo_starved      :  1;
		uint64_t tx_serdes_if_fifo_overflow  :  1;
		uint64_t                             : 53;
	};
};

#define SS2_PORT_PML_ERR_INFO_PCS_TX_DP_OFFSET	0x00000320
#define SS2_PORT_PML_ERR_INFO_PCS_TX_DP	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_PCS_TX_DP_OFFSET)
#define SS2_PORT_PML_ERR_INFO_PCS_TX_DP_SIZE	0x00000008

union ss2_port_pml_err_info_pcs_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_cdc_underrun             :  4;
		uint64_t tx_cdc_overflow             :  4;
		uint64_t tx_mac_if_fifo_overflow     :  1;
		uint64_t tx_mac_if_fifo_starved      :  1;
		uint64_t tx_serdes_if_fifo_overflow  :  1;
		uint64_t                             :  5;
		uint64_t subport                     :  2;
		uint64_t                             : 46;
	};
};

#define C1_HNI_PML_ERR_INFO_PCS_RX_DP_OFFSET	0x000001a8
#define C1_HNI_PML_ERR_INFO_PCS_RX_DP	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_PCS_RX_DP_OFFSET)
#define C1_HNI_PML_ERR_INFO_PCS_RX_DP_SIZE	0x00000008

union c1_hni_pml_err_info_pcs_rx_dp {
	uint64_t qw;
	struct {
		uint64_t rx_cdc_overflow        :  4;
		uint64_t rx_rs_buffer_overflow  :  1;
		uint64_t                        : 59;
	};
};

#define SS2_PORT_PML_ERR_INFO_PCS_RX_DP_OFFSET	0x00000328
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DP	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_PCS_RX_DP_OFFSET)
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DP_SIZE	0x00000008

union ss2_port_pml_err_info_pcs_rx_dp {
	uint64_t qw;
	struct {
		uint64_t rx_cdc_overflow        :  4;
		uint64_t rx_rs_buffer_overflow  :  1;
		uint64_t                        :  3;
		uint64_t subport                :  2;
		uint64_t                        : 54;
	};
};

#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_OFFSET	0x000001b0
#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_OFFSET)
#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_SIZE	0x00000008

union c1_hni_pml_err_info_pcs_rx_deskew {
	uint64_t qw;
	struct {
		uint64_t rx_deskew_overflow  :  8;
		uint64_t                     : 56;
	};
};

#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_OFFSET	0x00000330
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_OFFSET)
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_SIZE	0x00000008

union ss2_port_pml_err_info_pcs_rx_deskew {
	uint64_t qw;
	struct {
		uint64_t rx_deskew_overflow  : 16;
		uint64_t                     : 48;
	};
};

#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_PES_OFFSET	0x000001b8
#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_PES	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_PES_OFFSET)
#define C1_HNI_PML_ERR_INFO_PCS_RX_DESKEW_PES_SIZE	0x00000008

union c1_hni_pml_err_info_pcs_rx_deskew_pes {
	uint64_t qw;
	struct {
		uint64_t deskew_0_pe_cnt  :  8;
		uint64_t deskew_1_pe_cnt  :  8;
		uint64_t deskew_2_pe_cnt  :  8;
		uint64_t deskew_3_pe_cnt  :  8;
		uint64_t deskew_4_pe_cnt  :  8;
		uint64_t deskew_5_pe_cnt  :  8;
		uint64_t deskew_6_pe_cnt  :  8;
		uint64_t deskew_7_pe_cnt  :  8;
	};
};

#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_PES_OFFSET	0x00000338
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_PES	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_PES_OFFSET)
#define SS2_PORT_PML_ERR_INFO_PCS_RX_DESKEW_PES_SIZE	0x00000008

union ss2_port_pml_err_info_pcs_rx_deskew_pes {
	uint64_t qw;
	struct {
		uint64_t rx_deskew_parity_err  : 16;
		uint64_t                       : 48;
	};
};

#define C1_HNI_PML_ERR_INFO_MAC_TX_DP_OFFSET	0x000001c0
#define C1_HNI_PML_ERR_INFO_MAC_TX_DP	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_MAC_TX_DP_OFFSET)
#define C1_HNI_PML_ERR_INFO_MAC_TX_DP_SIZE	0x00000008

union c1_hni_pml_err_info_mac_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_llr_if_credit_error   :  1;
		uint64_t tx_llr_if_ctl_par_err    :  1;
		uint64_t tx_llr_if_llr_par_err    :  1;
		uint64_t tx_llr_if_bytes_vld_err  :  1;
		uint64_t tx_llr_if_framing_err    :  1;
		uint64_t tx_llr_if_fcs_16_err     :  1;
		uint64_t tx_mac_gb_underrun       :  1;
		uint64_t                          : 57;
	};
};

#define SS2_PORT_PML_ERR_INFO_MAC_TX_DP_OFFSET	0x00000348
#define SS2_PORT_PML_ERR_INFO_MAC_TX_DP	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_MAC_TX_DP_OFFSET)
#define SS2_PORT_PML_ERR_INFO_MAC_TX_DP_SIZE	0x00000008

union ss2_port_pml_err_info_mac_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_llr_if_credit_error     :  1;
		uint64_t tx_llr_if_ctl_par_err      :  1;
		uint64_t tx_llr_if_llr_par_err      :  1;
		uint64_t tx_llr_if_llr_seg_par_err  :  4;
		uint64_t tx_llr_if_bytes_vld_err    :  4;
		uint64_t tx_llr_if_framing_err      :  1;
		uint64_t tx_llr_if_fcs_16_err       :  1;
		uint64_t tx_mac_gb_underrun         :  1;
		uint64_t tx_mac_gb_overflow         :  1;
		uint64_t tx_mac_fifo_overflow       :  1;
		uint64_t subport                    :  2;
		uint64_t                            : 46;
	};
};

#define C1_HNI_PML_ERR_INFO_MAC_RX_DP_OFFSET	0x000001c8
#define C1_HNI_PML_ERR_INFO_MAC_RX_DP	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_MAC_RX_DP_OFFSET)
#define C1_HNI_PML_ERR_INFO_MAC_RX_DP_SIZE	0x00000008

union c1_hni_pml_err_info_mac_rx_dp {
	uint64_t qw;
	struct {
		uint64_t rx_mac_gb_err          :  1;
		uint64_t rx_mac_illegal_size    :  1;
		uint64_t rx_mac_fragment        :  1;
		uint64_t rx_mac_preamble_error  :  1;
		uint64_t rx_mac_framing_error   :  1;
		uint64_t rx_mac_fcs32_error     :  1;
		uint64_t                        : 58;
	};
};

#define SS2_PORT_PML_ERR_INFO_MAC_RX_DP_OFFSET	0x00000350
#define SS2_PORT_PML_ERR_INFO_MAC_RX_DP	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_MAC_RX_DP_OFFSET)
#define SS2_PORT_PML_ERR_INFO_MAC_RX_DP_SIZE	0x00000008

union ss2_port_pml_err_info_mac_rx_dp {
	uint64_t qw;
	struct {
		uint64_t rx_mac_illegal_size        :  1;
		uint64_t rx_mac_fragment            :  1;
		uint64_t rx_mac_fifo_overflow       :  1;
		uint64_t rx_mac_short_pkt_buf_ovfw  :  1;
		uint64_t                            :  4;
		uint64_t subport                    :  2;
		uint64_t                            : 54;
	};
};

#define C1_HNI_PML_ERR_INFO_LLR_TX_DP_OFFSET	0x000001d0
#define C1_HNI_PML_ERR_INFO_LLR_TX_DP	(C_HNI_PML_BASE + C1_HNI_PML_ERR_INFO_LLR_TX_DP_OFFSET)
#define C1_HNI_PML_ERR_INFO_LLR_TX_DP_SIZE	0x00000008

union c1_hni_pml_err_info_llr_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_llr_sbe_syndrome            : 10;
		uint64_t tx_llr_bypass_sbe              :  1;
		uint64_t tx_llr_if_fifo_sbe             :  1;
		uint64_t                                :  4;
		uint64_t tx_llr_mbe_syndrome            : 10;
		uint64_t tx_llr_bypass_mbe              :  1;
		uint64_t tx_llr_if_fifo_mbe             :  1;
		uint64_t                                :  4;
		uint64_t tx_llr_bypass_par_err          :  1;
		uint64_t tx_llr_bypass_overflow_err     :  1;
		uint64_t tx_llr_bypass_illegal_len_err  :  1;
		uint64_t tx_llr_if_fifo_par_err         :  1;
		uint64_t tx_llr_if_fifo_overrun_err     :  1;
		uint64_t tx_llr_if_fifo_len_err         :  1;
		uint64_t                                : 26;
	};
};

#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_OFFSET	0x00000358
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_LLR_TX_DP_OFFSET)
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_SIZE	0x00000008

union ss2_port_pml_err_info_llr_tx_dp {
	uint64_t qw;
	struct {
		uint64_t tx_llr_bypass_par_err             :  1;
		uint64_t tx_llr_bypass_overflow_err        :  1;
		uint64_t tx_llr_bypass_illegal_len_err     :  1;
		uint64_t tx_llr_if_fifo_seg0_par_err       :  1;
		uint64_t tx_llr_if_fifo_seg1_par_err       :  1;
		uint64_t tx_llr_if_fifo_seg2_par_err       :  1;
		uint64_t tx_llr_if_fifo_seg3_par_err       :  1;
		uint64_t tx_llr_if_fifo_par_err            :  1;
		uint64_t tx_llr_if_fifo_overrun_err        :  1;
		uint64_t tx_llr_if_fifo_len_err            :  1;
		uint64_t tx_llr_bypass_illegal_frame_type  :  1;
		uint64_t tx_llr_input_bus_err              :  1;
		uint64_t                                   : 12;
		uint64_t subport                           :  2;
		uint64_t                                   : 38;
	};
};

#define C_HNI_PML_CFG_GENERAL_OFFSET	0x00000400
#define C_HNI_PML_CFG_GENERAL	(C_HNI_PML_BASE + C_HNI_PML_CFG_GENERAL_OFFSET)
#define C_HNI_PML_CFG_GENERAL_SIZE	0x00000008

union c_hni_pml_cfg_general {
	uint64_t qw;
	struct {
		uint64_t clock_period_ps  : 11;
		uint64_t                  : 53;
	};
};

#define C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT_OFFSET	0x00000408
#define C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT	(C_HNI_PML_BASE + C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT_OFFSET)
#define C1_HNI_PML_CFG_SERDES_CORE_INTERRUPT_SIZE	0x00000008

union c1_hni_pml_cfg_serdes_core_interrupt {
	uint64_t qw;
	struct {
		uint64_t set_interrupt_delay           :  8;
		uint64_t clear_interrupt_delay         :  8;
		uint64_t capture_interrupt_data_delay  :  8;
		uint64_t                               : 40;
	};
};

#define C1_HNI_PML_SERDES_CORE_INTERRUPT_OFFSET	0x00000410
#define C1_HNI_PML_SERDES_CORE_INTERRUPT	(C_HNI_PML_BASE + C1_HNI_PML_SERDES_CORE_INTERRUPT_OFFSET)
#define C1_HNI_PML_SERDES_CORE_INTERRUPT_SIZE	0x00000008

union c1_hni_pml_serdes_core_interrupt {
	uint64_t qw;
	struct {
		uint64_t core_interrupt_data  : 16;
		uint64_t core_interrupt_code  : 16;
		uint64_t do_core_interrupt    :  1;
		uint64_t serdes_sel           :  2;
		uint64_t                      : 29;
	};
};

#define C1_HNI_PML_CFG_PCS_OFFSET	0x00000500
#define C1_HNI_PML_CFG_PCS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_OFFSET)
#define C1_HNI_PML_CFG_PCS_SIZE	0x00000008

union c1_hni_pml_cfg_pcs {
	uint64_t qw;
	struct {
		uint64_t pcs_enable                :  1;
		uint64_t enable_auto_neg           :  1;
		uint64_t                           :  6;
		uint64_t pcs_mode                  :  2;
		uint64_t enable_auto_lane_degrade  :  1;
		uint64_t                           :  5;
		uint64_t ll_fec                    :  1;
		uint64_t                           :  7;
		uint64_t timestamp_shift           :  2;
		uint64_t                           : 38;
	};
};

#define SS2_PORT_PML_CFG_PCS_OFFSET	0x00000500
#define SS2_PORT_PML_CFG_PCS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_OFFSET)
#define SS2_PORT_PML_CFG_PCS_SIZE	0x00000008

union ss2_port_pml_cfg_pcs {
	uint64_t qw;
	struct {
		uint64_t pcs_mode                  :  3;
		uint64_t enable_auto_lane_degrade  :  1;
		uint64_t by_25g_fec_mode           :  2;
		uint64_t                           : 10;
		uint64_t timestamp_shift           :  2;
		uint64_t                           : 46;
	};
};

#define C1_HNI_PML_CFG_PCS_AUTONEG_OFFSET	0x00000508
#define C1_HNI_PML_CFG_PCS_AUTONEG	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_AUTONEG_OFFSET)
#define C1_HNI_PML_CFG_PCS_AUTONEG_SIZE	0x00000008

union c_hni_pml_cfg_pcs_autoneg {
	uint64_t qw;
	struct {
		uint64_t next_page_loaded  :  1;
		uint64_t                   :  7;
		uint64_t restart           :  1;
		uint64_t                   :  7;
		uint64_t reset             :  1;
		uint64_t                   :  7;
		uint64_t tx_lane           :  2;
		uint64_t rx_lane           :  2;
		uint64_t                   : 36;
	};
};

#define SS2_PORT_PML_CFG_PCS_AUTONEG_OFFSET(idx)	(0x00000540+((idx)*8))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_ENTRIES	4
#define SS2_PORT_PML_CFG_PCS_AUTONEG(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_AUTONEG_OFFSET(idx))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_SIZE	0x00000020

union ss2_port_pml_cfg_pcs_autoneg {
	uint64_t qw;
	struct {
		uint64_t next_page_loaded  :  1;
		uint64_t                   :  7;
		uint64_t restart           :  1;
		uint64_t                   :  7;
		uint64_t reset             :  1;
		uint64_t                   :  7;
		uint64_t tx_lane           :  2;
		uint64_t rx_lane           :  2;
		uint64_t                   : 36;
	};
};

#define C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET	0x00000510
#define C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET)
#define C1_HNI_PML_CFG_PCS_AUTONEG_TIMERS_SIZE	0x00000008

union c1_hni_pml_cfg_pcs_autoneg_timers {
	uint64_t qw;
	struct {
		uint64_t break_link_timer_max         : 32;
		uint64_t link_fail_inhibit_timer_max  : 32;
	};
};

#define SS2_PORT_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET	0x00000560
#define SS2_PORT_PML_CFG_PCS_AUTONEG_TIMERS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_AUTONEG_TIMERS_OFFSET)
#define SS2_PORT_PML_CFG_PCS_AUTONEG_TIMERS_SIZE	0x00000010

union ss2_port_pml_cfg_pcs_autoneg_timers {
	uint64_t qw[2];
	struct {
		uint64_t break_link_timer_max         : 32;
		uint64_t                              : 32;
		uint64_t link_fail_inhibit_timer_max  : 34;
		uint64_t                              : 30;
	};
};

#define C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET	0x00000518
#define C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET)
#define C1_HNI_PML_CFG_PCS_AUTONEG_BASE_PAGE_SIZE	0x00000008

union c_hni_pml_cfg_pcs_autoneg_base_page {
	uint64_t qw;
	struct {
		uint64_t base_page  : 48;
		uint64_t            : 16;
	};
};

#define SS2_PORT_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET(idx)	(0x00000580+((idx)*8))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_BASE_PAGE_ENTRIES	4
#define SS2_PORT_PML_CFG_PCS_AUTONEG_BASE_PAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_AUTONEG_BASE_PAGE_OFFSET(idx))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_BASE_PAGE_SIZE	0x00000020

union ss2_port_pml_cfg_pcs_autoneg_base_page {
	uint64_t qw;
	struct {
		uint64_t base_page  : 48;
		uint64_t            : 16;
	};
};

#define C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET	0x00000520
#define C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET)
#define C1_HNI_PML_CFG_PCS_AUTONEG_NEXT_PAGE_SIZE	0x00000008

union c_hni_pml_cfg_pcs_autoneg_next_page {
	uint64_t qw;
	struct {
		uint64_t next_page  : 48;
		uint64_t            : 16;
	};
};

#define SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET(idx)	(0x000005a0+((idx)*8))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE_ENTRIES	4
#define SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE_OFFSET(idx))
#define SS2_PORT_PML_CFG_PCS_AUTONEG_NEXT_PAGE_SIZE	0x00000020

union ss2_port_pml_cfg_pcs_autoneg_next_page {
	uint64_t qw;
	struct {
		uint64_t next_page  : 48;
		uint64_t            : 16;
	};
};

#define C1_HNI_PML_CFG_PCS_AMS_OFFSET	0x00000528
#define C1_HNI_PML_CFG_PCS_AMS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_AMS_OFFSET)
#define C_HNI_PML_CFG_PCS_AMS_SIZE	0x00000008
#define C1_HNI_PML_CFG_PCS_AMS_SIZE	C_HNI_PML_CFG_PCS_AMS_SIZE

union c_hni_pml_cfg_pcs_ams {
	uint64_t qw;
	struct {
		uint64_t am_spacing                   :  4;
		uint64_t                              :  4;
		uint64_t ram_spacing                  :  4;
		uint64_t                              : 12;
		uint64_t use_programmable_am_spacing  :  1;
		uint64_t use_programmable_ams         :  1;
		uint64_t                              : 38;
	};
};

#define SS2_PORT_PML_CFG_PCS_AMS_OFFSET	0x000005c0
#define SS2_PORT_PML_CFG_PCS_AMS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_AMS_OFFSET)
#define SS2_PORT_PML_CFG_PCS_AMS_SIZE	0x00000008

union ss2_port_pml_cfg_pcs_ams {
	uint64_t qw;
	struct {
		uint64_t am_spacing                   :  4;
		uint64_t                              :  4;
		uint64_t ram_spacing                  :  4;
		uint64_t                              : 12;
		uint64_t use_programmable_am_spacing  :  1;
		uint64_t use_programmable_ams         :  1;
		uint64_t                              : 38;
	};
};

#define C1_HNI_PML_CFG_PCS_UM_OFFSET(idx)	(0x00000530+((idx)*8))
#define C1_HNI_PML_CFG_PCS_UM_ENTRIES	8
#define C1_HNI_PML_CFG_PCS_UM(idx)	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_UM_OFFSET(idx))
#define C1_HNI_PML_CFG_PCS_UM_SIZE	0x00000040

union c_hni_pml_cfg_pcs_um {
	uint64_t qw;
	struct {
		uint64_t um  : 56;
		uint64_t     :  8;
	};
};

#define SS2_PORT_PML_CFG_PCS_UM_OFFSET(idx)	(0x00000600+((idx)*8))
#define SS2_PORT_PML_CFG_PCS_UM_ENTRIES	16
#define SS2_PORT_PML_CFG_PCS_UM(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_UM_OFFSET(idx))
#define SS2_PORT_PML_CFG_PCS_UM_SIZE	0x00000080

union ss2_port_pml_cfg_pcs_um {
	uint64_t qw;
	struct {
		uint64_t um  : 56;
		uint64_t     :  8;
	};
};

#define C1_HNI_PML_CFG_PCS_CM_OFFSET	0x00000570
#define C1_HNI_PML_CFG_PCS_CM	(C_HNI_PML_BASE + C1_HNI_PML_CFG_PCS_CM_OFFSET)
#define C_HNI_PML_CFG_PCS_CM_SIZE	0x00000008
#define C1_HNI_PML_CFG_PCS_CM_SIZE	C_HNI_PML_CFG_PCS_CM_SIZE

union c_hni_pml_cfg_pcs_cm {
	uint64_t qw;
	struct {
		uint64_t cm;
	};
};

#define SS2_PORT_PML_CFG_PCS_CM_OFFSET	0x00000680
#define SS2_PORT_PML_CFG_PCS_CM	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_CM_OFFSET)
#define SS2_PORT_PML_CFG_PCS_CM_SIZE	0x00000008

union ss2_port_pml_cfg_pcs_cm {
	uint64_t qw;
	struct {
		uint64_t cm;
	};
};

#define C1_HNI_PML_CFG_TX_PCS_OFFSET	0x00000578
#define C1_HNI_PML_CFG_TX_PCS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_TX_PCS_OFFSET)
#define C1_HNI_PML_CFG_TX_PCS_SIZE	0x00000008

union c1_hni_pml_cfg_tx_pcs {
	uint64_t qw;
	struct {
		uint64_t lane_0_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_1_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_2_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_3_source       :  2;
		uint64_t                     :  6;
		uint64_t enable_ctl_os       :  1;
		uint64_t allow_auto_degrade  :  1;
		uint64_t                     :  6;
		uint64_t gearbox_credits     :  4;
		uint64_t                     :  4;
		uint64_t cdc_ready_level     :  4;
		uint64_t                     : 12;
	};
};

#define SS2_PORT_PML_CFG_TX_PCS_OFFSET	0x00000688
#define SS2_PORT_PML_CFG_TX_PCS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_TX_PCS_OFFSET)
#define SS2_PORT_PML_CFG_TX_PCS_SIZE	0x00000008

union ss2_port_pml_cfg_tx_pcs {
	uint64_t qw;
	struct {
		uint64_t lane_0_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_1_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_2_source       :  2;
		uint64_t                     :  6;
		uint64_t lane_3_source       :  2;
		uint64_t                     :  6;
		uint64_t allow_auto_degrade  :  1;
		uint64_t                     :  7;
		uint64_t cdc_ready_level     :  4;
		uint64_t en_pk_bw_limiter    :  1;
		uint64_t                     : 19;
	};
};

#define C1_HNI_PML_CFG_RX_PCS_OFFSET	0x00000580
#define C1_HNI_PML_CFG_RX_PCS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_RX_PCS_OFFSET)
#define C1_HNI_PML_CFG_RX_PCS_SIZE	0x00000008

union c1_hni_pml_cfg_rx_pcs {
	uint64_t qw;
	struct {
		uint64_t active_lanes               :  4;
		uint64_t                            :  4;
		uint64_t rs_mode                    :  3;
		uint64_t                            :  5;
		uint64_t enable_lock                :  1;
		uint64_t restart_lock_on_bad_ams    :  1;
		uint64_t restart_lock_on_bad_cws    :  1;
		uint64_t                            :  5;
		uint64_t enable_ctl_os              :  1;
		uint64_t                            :  7;
		uint64_t health_bad_sensitivity     :  4;
		uint64_t lp_health_bad_sensitivity  :  1;
		uint64_t                            :  3;
		uint64_t allow_auto_degrade         :  1;
		uint64_t                            :  7;
		uint64_t enable_rx_sm               :  1;
		uint64_t                            : 15;
	};
};

#define SS2_PORT_PML_CFG_RX_PCS_OFFSET	0x000006c0
#define SS2_PORT_PML_CFG_RX_PCS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_RX_PCS_OFFSET)
#define SS2_PORT_PML_CFG_RX_PCS_SIZE	0x00000008

union ss2_port_pml_cfg_rx_pcs {
	uint64_t qw;
	struct {
		uint64_t active_lanes               :  4;
		uint64_t cw_gap_544                 :  5;
		uint64_t cw_gap_ll                  :  5;
		uint64_t cw_gap_one_fec             :  5;
		uint64_t restart_lock_on_bad_ams    :  1;
		uint64_t restart_lock_on_bad_cws    :  1;
		uint64_t health_bad_sensitivity     :  4;
		uint64_t lp_health_bad_sensitivity  :  1;
		uint64_t allow_auto_degrade         :  1;
		uint64_t lane_0_source              :  2;
		uint64_t lane_1_source              :  2;
		uint64_t lane_2_source              :  2;
		uint64_t lane_3_source              :  2;
		uint64_t signal_detect_enn          :  4;
		uint64_t rx_lock_enn                :  4;
		uint64_t rx_clk_vld_enn             :  4;
		uint64_t                            : 17;
	};
};

#define C1_HNI_PML_CFG_RX_PCS_AMS_OFFSET	0x00000588
#define C1_HNI_PML_CFG_RX_PCS_AMS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_RX_PCS_AMS_OFFSET)
#define C_HNI_PML_CFG_RX_PCS_AMS_SIZE	0x00000008
#define C1_HNI_PML_CFG_RX_PCS_AMS_SIZE	C_HNI_PML_CFG_RX_PCS_AMS_SIZE

union c_hni_pml_cfg_rx_pcs_ams {
	uint64_t qw;
	struct {
		uint64_t cm_match_mask  : 16;
		uint64_t um_match_mask  : 14;
		uint64_t                : 34;
	};
};

#define SS2_PORT_PML_CFG_RX_PCS_AMS_OFFSET	0x00000700
#define SS2_PORT_PML_CFG_RX_PCS_AMS	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_RX_PCS_AMS_OFFSET)
#define SS2_PORT_PML_CFG_RX_PCS_AMS_SIZE	0x00000008

union ss2_port_pml_cfg_rx_pcs_ams {
	uint64_t qw;
	struct {
		uint64_t cm_match_mask  : 16;
		uint64_t um_match_mask  : 14;
		uint64_t                : 34;
	};
};

#define C1_HNI_PML_CFG_TX_MAC_OFFSET	0x00000600
#define C1_HNI_PML_CFG_TX_MAC	(C_HNI_PML_BASE + C1_HNI_PML_CFG_TX_MAC_OFFSET)
#define C1_HNI_PML_CFG_TX_MAC_SIZE	0x00000008

union c1_hni_pml_cfg_tx_mac {
	uint64_t qw;
	struct {
		uint64_t mac_operational         :  1;
		uint64_t short_preamble          :  1;
		uint64_t                         :  6;
		uint64_t pcs_credits             :  4;
		uint64_t                         :  4;
		uint64_t ifg_mode                :  1;
		uint64_t                         :  3;
		uint64_t vs_version              :  4;
		uint64_t ieee_ifg_adjustment     :  2;
		uint64_t non_portals_sof_window  :  5;
		uint64_t non_portals_max_sof     :  4;
		uint64_t any_frame_sof_window_0  :  5;
		uint64_t any_frame_max_sof_0     :  4;
		uint64_t any_frame_sof_window_1  :  5;
		uint64_t any_frame_max_sof_1     :  4;
		uint64_t portals_protocol        :  8;
		uint64_t                         :  3;
	};
};

#define SS2_PORT_PML_CFG_TX_MAC_OFFSET	0x00000800
#define SS2_PORT_PML_CFG_TX_MAC	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_TX_MAC_OFFSET)
#define SS2_PORT_PML_CFG_TX_MAC_SIZE	0x00000008

union ss2_port_pml_cfg_tx_mac {
	uint64_t qw;
	struct {
		uint64_t ifg_mode                :  1;
		uint64_t min_ifg                 :  3;
		uint64_t mac_pad_idle_thresh     :  8;
		uint64_t                         :  1;
		uint64_t ieee_ifg_adjustment     :  3;
		uint64_t vs_version              :  4;
		uint64_t                         :  6;
		uint64_t non_st_sof_window       :  6;
		uint64_t non_st_max_sof          :  3;
		uint64_t any_frame_sof_window_0  :  6;
		uint64_t any_frame_max_sof_0     :  3;
		uint64_t                         : 10;
		uint64_t st_protocol             :  8;
		uint64_t                         :  2;
	};
};

#define C1_HNI_PML_CFG_RX_MAC_OFFSET	0x00000608
#define C1_HNI_PML_CFG_RX_MAC	(C_HNI_PML_BASE + C1_HNI_PML_CFG_RX_MAC_OFFSET)
#define C1_HNI_PML_CFG_RX_MAC_SIZE	0x00000008

union c1_hni_pml_cfg_rx_mac {
	uint64_t qw;
	struct {
		uint64_t mac_operational      :  1;
		uint64_t short_preamble       :  1;
		uint64_t                      :  6;
		uint64_t filter_illegal_size  :  1;
		uint64_t                      : 55;
	};
};

#define SS2_PORT_PML_CFG_RX_MAC_OFFSET	0x00000840
#define SS2_PORT_PML_CFG_RX_MAC	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_RX_MAC_OFFSET)
#define SS2_PORT_PML_CFG_RX_MAC_SIZE	0x00000008

union ss2_port_pml_cfg_rx_mac {
	uint64_t qw;
	struct {
		uint64_t flit_packing_cnt  :  4;
		uint64_t stop_packing      :  1;
		uint64_t                   : 59;
	};
};

#define C1_HNI_PML_CFG_LLR_OFFSET	0x00000700
#define C1_HNI_PML_CFG_LLR	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_OFFSET)
#define C1_HNI_PML_CFG_LLR_SIZE	0x00000008

union c1_hni_pml_cfg_llr {
	uint64_t qw;
	struct {
		uint64_t llr_mode                  :  2;
		uint64_t                           :  6;
		uint64_t mac_if_credits            :  4;
		uint64_t                           :  4;
		uint64_t link_down_behavior        :  2;
		uint64_t                           :  6;
		uint64_t enable_loop_timing        :  1;
		uint64_t                           :  7;
		uint64_t size                      :  2;
		uint64_t                           :  6;
		uint64_t filter_ctl_frames         :  1;
		uint64_t filter_lossless_when_off  :  1;
		uint64_t ack_nack_err_check        :  1;
		uint64_t preamble_seq_check        :  1;
		uint64_t                           : 20;
	};
};

#define SS2_PORT_PML_CFG_LLR_OFFSET	0x00000900
#define SS2_PORT_PML_CFG_LLR	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_OFFSET)
#define SS2_PORT_PML_CFG_LLR_SIZE	0x00000008

union ss2_port_pml_cfg_llr {
	uint64_t qw;
	struct {
		uint64_t size                :  1;
		uint64_t                     :  7;
		uint64_t ack_nack_err_check  :  1;
		uint64_t preamble_seq_check  :  1;
		uint64_t                     :  6;
		uint64_t stop_packing        :  1;
		uint64_t                     : 47;
	};
};

#define C1_HNI_PML_CFG_LLR_CF_SMAC_OFFSET	0x00000710
#define C1_HNI_PML_CFG_LLR_CF_SMAC	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_CF_SMAC_OFFSET)
#define C_HNI_PML_CFG_LLR_CF_SMAC_SIZE	0x00000008
#define C1_HNI_PML_CFG_LLR_CF_SMAC_SIZE	C_HNI_PML_CFG_LLR_CF_SMAC_SIZE

union c_hni_pml_cfg_llr_cf_smac {
	uint64_t qw;
	struct {
		uint64_t ctl_frame_smac  : 48;
		uint64_t                 : 16;
	};
};

#define SS2_PORT_PML_CFG_LLR_CF_SMAC_OFFSET	0x00000948
#define SS2_PORT_PML_CFG_LLR_CF_SMAC	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_CF_SMAC_OFFSET)
#define SS2_PORT_PML_CFG_LLR_CF_SMAC_SIZE	0x00000008

union ss2_port_pml_cfg_llr_cf_smac {
	uint64_t qw;
	struct {
		uint64_t ctl_frame_smac  : 48;
		uint64_t                 : 16;
	};
};

#define C1_HNI_PML_CFG_LLR_CF_ETYPE_OFFSET	0x00000718
#define C1_HNI_PML_CFG_LLR_CF_ETYPE	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_CF_ETYPE_OFFSET)
#define C_HNI_PML_CFG_LLR_CF_ETYPE_SIZE	0x00000008
#define C1_HNI_PML_CFG_LLR_CF_ETYPE_SIZE	C_HNI_PML_CFG_LLR_CF_ETYPE_SIZE

union c_hni_pml_cfg_llr_cf_etype {
	uint64_t qw;
	struct {
		uint64_t ctl_frame_ethertype  : 16;
		uint64_t                      : 48;
	};
};

#define SS2_PORT_PML_CFG_LLR_CF_ETYPE_OFFSET	0x00000950
#define SS2_PORT_PML_CFG_LLR_CF_ETYPE	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_CF_ETYPE_OFFSET)
#define SS2_PORT_PML_CFG_LLR_CF_ETYPE_SIZE	0x00000008

union ss2_port_pml_cfg_llr_cf_etype {
	uint64_t qw;
	struct {
		uint64_t ctl_frame_ethertype  : 16;
		uint64_t                      : 48;
	};
};

#define C1_HNI_PML_CFG_LLR_CF_RATES_OFFSET	0x00000720
#define C1_HNI_PML_CFG_LLR_CF_RATES	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_CF_RATES_OFFSET)
#define C1_HNI_PML_CFG_LLR_CF_RATES_SIZE	0x00000008

union c_hni_pml_cfg_llr_cf_rates {
	uint64_t qw;
	struct {
		uint64_t loop_timing_period  : 16;
		uint64_t init_ctl_os_period  :  4;
		uint64_t                     : 44;
	};
};

#define SS2_PORT_PML_CFG_LLR_CF_RATES_OFFSET(idx)	(0x00000960+((idx)*8))
#define SS2_PORT_PML_CFG_LLR_CF_RATES_ENTRIES	4
#define SS2_PORT_PML_CFG_LLR_CF_RATES(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_CF_RATES_OFFSET(idx))
#define SS2_PORT_PML_CFG_LLR_CF_RATES_SIZE	0x00000020

union ss2_port_pml_cfg_llr_cf_rates {
	uint64_t qw;
	struct {
		uint64_t loop_timing_period  : 16;
		uint64_t init_ctl_os_period  :  4;
		uint64_t                     : 44;
	};
};

#define C1_HNI_PML_CFG_LLR_SM_OFFSET	0x00000728
#define C1_HNI_PML_CFG_LLR_SM	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_SM_OFFSET)
#define C1_HNI_PML_CFG_LLR_SM_SIZE	0x00000008

union c_hni_pml_cfg_llr_sm {
	uint64_t qw;
	struct {
		uint64_t replay_timer_max  : 16;
		uint64_t replay_ct_max     :  8;
		uint64_t allow_re_init     :  1;
		uint64_t                   :  7;
		uint64_t retry_threshold   :  8;
		uint64_t                   : 24;
	};
};

#define SS2_PORT_PML_CFG_LLR_SM_OFFSET(idx)	(0x00000980+((idx)*8))
#define SS2_PORT_PML_CFG_LLR_SM_ENTRIES	4
#define SS2_PORT_PML_CFG_LLR_SM(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_SM_OFFSET(idx))
#define SS2_PORT_PML_CFG_LLR_SM_SIZE	0x00000020

union ss2_port_pml_cfg_llr_sm {
	uint64_t qw;
	struct {
		uint64_t replay_timer_max  : 16;
		uint64_t replay_ct_max     :  8;
		uint64_t allow_re_init     :  1;
		uint64_t                   :  7;
		uint64_t retry_threshold   :  8;
		uint64_t                   : 24;
	};
};

#define C1_HNI_PML_CFG_LLR_CAPACITY_OFFSET	0x00000730
#define C1_HNI_PML_CFG_LLR_CAPACITY	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_CAPACITY_OFFSET)
#define C1_HNI_PML_CFG_LLR_CAPACITY_SIZE	0x00000008

union c1_hni_pml_cfg_llr_capacity {
	uint64_t qw;
	struct {
		uint64_t max_seq   : 12;
		uint64_t           :  4;
		uint64_t max_data  : 12;
		uint64_t           : 36;
	};
};

#define SS2_PORT_PML_CFG_LLR_CAPACITY_OFFSET(idx)	(0x000009a0+((idx)*8))
#define SS2_PORT_PML_CFG_LLR_CAPACITY_ENTRIES	4
#define SS2_PORT_PML_CFG_LLR_CAPACITY(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_CAPACITY_OFFSET(idx))
#define SS2_PORT_PML_CFG_LLR_CAPACITY_SIZE	0x00000020

union ss2_port_pml_cfg_llr_capacity {
	uint64_t qw;
	struct {
		uint64_t max_seq   : 12;
		uint64_t           :  4;
		uint64_t max_data  : 12;
		uint64_t           :  4;
		uint64_t           : 32;
	};
};

#define C1_HNI_PML_CFG_LLR_TIMEOUTS_OFFSET	0x00000738
#define C1_HNI_PML_CFG_LLR_TIMEOUTS	(C_HNI_PML_BASE + C1_HNI_PML_CFG_LLR_TIMEOUTS_OFFSET)
#define C1_HNI_PML_CFG_LLR_TIMEOUTS_SIZE	0x00000008

union c_hni_pml_cfg_llr_timeouts {
	uint64_t qw;
	struct {
		uint64_t pcs_link_dn_timer_max  : 32;
		uint64_t data_age_timer_max     : 32;
	};
};

#define SS2_PORT_PML_CFG_LLR_TIMEOUTS_OFFSET(idx)	(0x000009c0+((idx)*8))
#define SS2_PORT_PML_CFG_LLR_TIMEOUTS_ENTRIES	4
#define SS2_PORT_PML_CFG_LLR_TIMEOUTS(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_TIMEOUTS_OFFSET(idx))
#define SS2_PORT_PML_CFG_LLR_TIMEOUTS_SIZE	0x00000020

union ss2_port_pml_cfg_llr_timeouts {
	uint64_t qw;
	struct {
		uint64_t pcs_link_dn_timer_max  : 32;
		uint64_t data_age_timer_max     : 32;
	};
};

#define C1_HNI_PML_SERDES_CORE_STATUS_OFFSET(idx)	(0x00000800+((idx)*8))
#define C1_HNI_PML_SERDES_CORE_STATUS_ENTRIES	4
#define C1_HNI_PML_SERDES_CORE_STATUS(idx)	(C_HNI_PML_BASE + C1_HNI_PML_SERDES_CORE_STATUS_OFFSET(idx))
#define C1_HNI_PML_SERDES_CORE_STATUS_SIZE	0x00000020

union c1_hni_pml_serdes_core_status {
	uint64_t qw;
	struct {
		uint64_t core_status     : 32;
		uint64_t rx_idle_detect  :  1;
		uint64_t rx_rdy          :  1;
		uint64_t tx_rdy          :  1;
		uint64_t                 : 29;
	};
};

#define C1_HNI_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET	0x00000820
#define C1_HNI_PML_STS_PCS_AUTONEG_BASE_PAGE	(C_HNI_PML_BASE + C1_HNI_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET)
#define C1_HNI_PML_STS_PCS_AUTONEG_BASE_PAGE_SIZE	0x00000008

union c_hni_pml_sts_pcs_autoneg_base_page {
	uint64_t qw;
	struct {
		uint64_t lp_base_page      : 48;
		uint64_t lp_ability        :  1;
		uint64_t page_received     :  1;
		uint64_t base_page         :  1;
		uint64_t state             :  4;
		uint64_t complete          :  1;
		uint64_t remaining_ack_ct  :  3;
		uint64_t ack_match_ct      :  2;
		uint64_t ability_match_ct  :  2;
		uint64_t                   :  1;
	};
};

#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET(idx)	(0x00001300+((idx)*8))
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_ENTRIES	4
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_OFFSET(idx))
#define SS2_PORT_PML_STS_PCS_AUTONEG_BASE_PAGE_SIZE	0x00000020

union ss2_port_pml_sts_pcs_autoneg_base_page {
	uint64_t qw;
	struct {
		uint64_t lp_base_page      : 48;
		uint64_t lp_ability        :  1;
		uint64_t page_received     :  1;
		uint64_t base_page         :  1;
		uint64_t state             :  4;
		uint64_t complete          :  1;
		uint64_t remaining_ack_ct  :  3;
		uint64_t ack_match_ct      :  2;
		uint64_t ability_match_ct  :  2;
		uint64_t                   :  1;
	};
};

#define C1_HNI_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET	0x00000828
#define C1_HNI_PML_STS_PCS_AUTONEG_NEXT_PAGE	(C_HNI_PML_BASE + C1_HNI_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET)
#define C1_HNI_PML_STS_PCS_AUTONEG_NEXT_PAGE_SIZE	0x00000008

union c_hni_pml_sts_pcs_autoneg_next_page {
	uint64_t qw;
	struct {
		uint64_t lp_next_page      : 48;
		uint64_t lp_ability        :  1;
		uint64_t page_received     :  1;
		uint64_t base_page         :  1;
		uint64_t state             :  4;
		uint64_t complete          :  1;
		uint64_t remaining_ack_ct  :  3;
		uint64_t ack_match_ct      :  2;
		uint64_t ability_match_ct  :  2;
		uint64_t                   :  1;
	};
};

#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET(idx)	(0x00001320+((idx)*8))
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_ENTRIES	4
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_OFFSET(idx))
#define SS2_PORT_PML_STS_PCS_AUTONEG_NEXT_PAGE_SIZE	0x00000020

union ss2_port_pml_sts_pcs_autoneg_next_page {
	uint64_t qw;
	struct {
		uint64_t lp_next_page      : 48;
		uint64_t lp_ability        :  1;
		uint64_t page_received     :  1;
		uint64_t base_page         :  1;
		uint64_t state             :  4;
		uint64_t complete          :  1;
		uint64_t remaining_ack_ct  :  3;
		uint64_t ack_match_ct      :  2;
		uint64_t ability_match_ct  :  2;
		uint64_t                   :  1;
	};
};

#define C1_HNI_PML_STS_TX_PCS_OFFSET	0x00000830
#define C1_HNI_PML_STS_TX_PCS	(C_HNI_PML_BASE + C1_HNI_PML_STS_TX_PCS_OFFSET)
#define C1_HNI_PML_STS_TX_PCS_SIZE	0x00000008

union c_hni_pml_sts_tx_pcs {
	uint64_t qw;
	struct {
		uint64_t timestamp        : 32;
		uint64_t timestamp_valid  :  1;
		uint64_t                  : 31;
	};
};

#define SS2_PORT_PML_STS_TX_PCS_OFFSET(idx)	(0x00001340+((idx)*8))
#define SS2_PORT_PML_STS_TX_PCS_ENTRIES	4
#define SS2_PORT_PML_STS_TX_PCS(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_TX_PCS_OFFSET(idx))
#define SS2_PORT_PML_STS_TX_PCS_SIZE	0x00000020

union ss2_port_pml_sts_tx_pcs {
	uint64_t qw;
	struct {
		uint64_t timestamp        : 32;
		uint64_t timestamp_valid  :  1;
		uint64_t                  : 31;
	};
};

#define C1_HNI_PML_STS_RX_PCS_OFFSET	0x00000838
#define C1_HNI_PML_STS_RX_PCS	(C_HNI_PML_BASE + C1_HNI_PML_STS_RX_PCS_OFFSET)
#define C1_HNI_PML_STS_RX_PCS_SIZE	0x00000008

union c1_hni_pml_sts_rx_pcs {
	uint64_t qw;
	struct {
		uint64_t am_identity_0  :  3;
		uint64_t                :  1;
		uint64_t am_identity_1  :  3;
		uint64_t                :  1;
		uint64_t am_identity_2  :  3;
		uint64_t                :  1;
		uint64_t am_identity_3  :  3;
		uint64_t                :  1;
		uint64_t am_identity_4  :  3;
		uint64_t                :  1;
		uint64_t am_identity_5  :  3;
		uint64_t                :  1;
		uint64_t am_identity_6  :  3;
		uint64_t                :  1;
		uint64_t am_identity_7  :  3;
		uint64_t                :  1;
		uint64_t am_lock        :  8;
		uint64_t align_status   :  1;
		uint64_t                :  7;
		uint64_t fault          :  1;
		uint64_t local_fault    :  1;
		uint64_t hi_ser         :  1;
		uint64_t                : 13;
	};
};

#define C1_HNI_PML_STS_RX_PCS_AM_MATCH_OFFSET	0x00000840
#define C1_HNI_PML_STS_RX_PCS_AM_MATCH	(C_HNI_PML_BASE + C1_HNI_PML_STS_RX_PCS_AM_MATCH_OFFSET)
#define C1_HNI_PML_STS_RX_PCS_AM_MATCH_SIZE	0x00000008

union c1_hni_pml_sts_rx_pcs_am_match {
	uint64_t qw;
	struct {
		uint64_t am_match_state_0  :  2;
		uint64_t am_match_pos_0    :  6;
		uint64_t am_match_state_1  :  2;
		uint64_t am_match_pos_1    :  6;
		uint64_t am_match_state_2  :  2;
		uint64_t am_match_pos_2    :  6;
		uint64_t am_match_state_3  :  2;
		uint64_t am_match_pos_3    :  6;
		uint64_t am_match_state_4  :  2;
		uint64_t am_match_pos_4    :  6;
		uint64_t am_match_state_5  :  2;
		uint64_t am_match_pos_5    :  6;
		uint64_t am_match_state_6  :  2;
		uint64_t am_match_pos_6    :  6;
		uint64_t am_match_state_7  :  2;
		uint64_t am_match_pos_7    :  6;
	};
};

#define C1_HNI_PML_STS_RX_PCS_FECL_SOURCES_OFFSET	0x00000848
#define C1_HNI_PML_STS_RX_PCS_FECL_SOURCES	(C_HNI_PML_BASE + C1_HNI_PML_STS_RX_PCS_FECL_SOURCES_OFFSET)
#define C1_HNI_PML_STS_RX_PCS_FECL_SOURCES_SIZE	0x00000008

union c1_hni_pml_sts_rx_pcs_fecl_sources {
	uint64_t qw;
	struct {
		uint64_t fecl_source_0      :  3;
		uint64_t fecl_source_vld_0  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_1      :  3;
		uint64_t fecl_source_vld_1  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_2      :  3;
		uint64_t fecl_source_vld_2  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_3      :  3;
		uint64_t fecl_source_vld_3  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_4      :  3;
		uint64_t fecl_source_vld_4  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_5      :  3;
		uint64_t fecl_source_vld_5  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_6      :  3;
		uint64_t fecl_source_vld_6  :  1;
		uint64_t                    :  4;
		uint64_t fecl_source_7      :  3;
		uint64_t fecl_source_vld_7  :  1;
		uint64_t                    :  4;
	};
};

#define C1_HNI_PML_STS_RX_PCS_DESKEW_DEPTHS_OFFSET	0x00000850
#define C1_HNI_PML_STS_RX_PCS_DESKEW_DEPTHS	(C_HNI_PML_BASE + C1_HNI_PML_STS_RX_PCS_DESKEW_DEPTHS_OFFSET)
#define C1_HNI_PML_STS_RX_PCS_DESKEW_DEPTHS_SIZE	0x00000008

union c1_hni_pml_sts_rx_pcs_deskew_depths {
	uint64_t qw;
	struct {
		uint64_t df_depth_0  :  8;
		uint64_t df_depth_1  :  8;
		uint64_t df_depth_2  :  8;
		uint64_t df_depth_3  :  8;
		uint64_t df_depth_4  :  8;
		uint64_t df_depth_5  :  8;
		uint64_t df_depth_6  :  8;
		uint64_t df_depth_7  :  8;
	};
};

#define C1_HNI_PML_STS_PCS_LANE_DEGRADE_OFFSET	0x00000868
#define C1_HNI_PML_STS_PCS_LANE_DEGRADE	(C_HNI_PML_BASE + C1_HNI_PML_STS_PCS_LANE_DEGRADE_OFFSET)
#define C1_HNI_PML_STS_PCS_LANE_DEGRADE_SIZE	0x00000008

union c1_hni_pml_sts_pcs_lane_degrade {
	uint64_t qw;
	struct {
		uint64_t rx_fecl_health_good  :  8;
		uint64_t lp_pls_available     :  4;
		uint64_t                      :  4;
		uint64_t rx_state             :  2;
		uint64_t                      :  2;
		uint64_t rx_pls_available     :  4;
		uint64_t tx_state             :  2;
		uint64_t                      :  2;
		uint64_t tx_lane              :  2;
		uint64_t                      :  2;
		uint64_t lp_health_bad_0      :  4;
		uint64_t lp_health_bad_1      :  4;
		uint64_t lp_health_bad_2      :  4;
		uint64_t lp_health_bad_3      :  4;
		uint64_t lp_health_good_0     :  4;
		uint64_t lp_health_good_1     :  4;
		uint64_t lp_health_good_2     :  4;
		uint64_t lp_health_good_3     :  4;
	};
};

#define SS2_PORT_PML_STS_PCS_LANE_DEGRADE_OFFSET	0x00001460
#define SS2_PORT_PML_STS_PCS_LANE_DEGRADE	(C_HNI_PML_BASE + SS2_PORT_PML_STS_PCS_LANE_DEGRADE_OFFSET)
#define SS2_PORT_PML_STS_PCS_LANE_DEGRADE_SIZE	0x00000010

union ss2_port_pml_sts_pcs_lane_degrade {
	uint64_t qw[2];
	struct {
		uint64_t rx_fecl_health_good  : 16;
		uint64_t lp_pls_available     :  4;
		uint64_t rx_state             :  3;
		uint64_t                      :  1;
		uint64_t rx_pls_available     :  4;
		uint64_t tx_state             :  3;
		uint64_t                      :  1;
		uint64_t tx_lane              :  2;
		uint64_t                      :  2;
		uint64_t degrade_failure      :  1;
		uint64_t degrading            :  1;
		uint64_t                      : 26;
		uint64_t lp_health_bad_0      :  5;
		uint64_t                      :  3;
		uint64_t lp_health_bad_1      :  5;
		uint64_t                      :  3;
		uint64_t lp_health_bad_2      :  5;
		uint64_t                      :  3;
		uint64_t lp_health_bad_3      :  5;
		uint64_t                      :  3;
		uint64_t lp_health_good_0     :  5;
		uint64_t                      :  3;
		uint64_t lp_health_good_1     :  5;
		uint64_t                      :  3;
		uint64_t lp_health_good_2     :  5;
		uint64_t                      :  3;
		uint64_t lp_health_good_3     :  5;
		uint64_t                      :  3;
	};
};

#define C1_HNI_PML_STS_LLR_OFFSET	0x00000a00
#define C1_HNI_PML_STS_LLR	(C_HNI_PML_BASE + C1_HNI_PML_STS_LLR_OFFSET)
#define C1_HNI_PML_STS_LLR_SIZE	0x00000008

union c_hni_pml_sts_llr {
	uint64_t qw;
	struct {
		uint64_t llr_state                  :  3;
		uint64_t                            :  1;
		uint64_t tx_seq                     : 20;
		uint64_t timestamp_frame_in_buffer  :  1;
		uint64_t timestamp_frame_ackd       :  1;
		uint64_t                            :  6;
		uint64_t ack_state                  :  2;
		uint64_t                            :  2;
		uint64_t ackd_seq                   : 20;
		uint64_t                            :  8;
	};
};

#define SS2_PORT_PML_STS_LLR_OFFSET(idx)	(0x00001500+((idx)*8))
#define SS2_PORT_PML_STS_LLR_ENTRIES	4
#define SS2_PORT_PML_STS_LLR(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_LLR_OFFSET(idx))
#define SS2_PORT_PML_STS_LLR_SIZE	0x00000020

union ss2_port_pml_sts_llr {
	uint64_t qw;
	struct {
		uint64_t llr_state                  :  3;
		uint64_t                            :  1;
		uint64_t tx_seq                     : 20;
		uint64_t timestamp_frame_in_buffer  :  1;
		uint64_t timestamp_frame_ackd       :  1;
		uint64_t                            :  6;
		uint64_t ack_state                  :  2;
		uint64_t                            :  2;
		uint64_t ackd_seq                   : 20;
		uint64_t                            :  8;
	};
};

#define C1_HNI_PML_STS_LLR_LOOP_TIME_OFFSET	0x00000a08
#define C1_HNI_PML_STS_LLR_LOOP_TIME	(C_HNI_PML_BASE + C1_HNI_PML_STS_LLR_LOOP_TIME_OFFSET)
#define C1_HNI_PML_STS_LLR_LOOP_TIME_SIZE	0x00000008

union c_hni_pml_sts_llr_loop_time {
	uint64_t qw;
	struct {
		uint64_t loop_time  : 16;
		uint64_t            : 48;
	};
};

#define SS2_PORT_PML_STS_LLR_LOOP_TIME_OFFSET(idx)	(0x00001520+((idx)*8))
#define SS2_PORT_PML_STS_LLR_LOOP_TIME_ENTRIES	4
#define SS2_PORT_PML_STS_LLR_LOOP_TIME(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_LLR_LOOP_TIME_OFFSET(idx))
#define SS2_PORT_PML_STS_LLR_LOOP_TIME_SIZE	0x00000020

union ss2_port_pml_sts_llr_loop_time {
	uint64_t qw;
	struct {
		uint64_t loop_time  : 16;
		uint64_t            : 48;
	};
};

#define C1_HNI_PML_STS_LLR_USAGE_OFFSET	0x00000a18
#define C1_HNI_PML_STS_LLR_USAGE	(C_HNI_PML_BASE + C1_HNI_PML_STS_LLR_USAGE_OFFSET)
#define C1_HNI_PML_STS_LLR_USAGE_SIZE	0x00000008

union c_hni_pml_sts_llr_usage {
	uint64_t qw;
	struct {
		uint64_t buffer_space_used  : 12;
		uint64_t                    :  4;
		uint64_t outstanding_seq    : 20;
		uint64_t                    : 28;
	};
};

#define SS2_PORT_PML_STS_LLR_USAGE_OFFSET(idx)	(0x00001560+((idx)*8))
#define SS2_PORT_PML_STS_LLR_USAGE_ENTRIES	4
#define SS2_PORT_PML_STS_LLR_USAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_LLR_USAGE_OFFSET(idx))
#define SS2_PORT_PML_STS_LLR_USAGE_SIZE	0x00000020

union ss2_port_pml_sts_llr_usage {
	uint64_t qw;
	struct {
		uint64_t buffer_space_used  : 12;
		uint64_t                    :  4;
		uint64_t outstanding_seq    : 20;
		uint64_t                    : 28;
	};
};

#define C1_HNI_PML_STS_LLR_MAX_USAGE_OFFSET	0x00000a20
#define C1_HNI_PML_STS_LLR_MAX_USAGE	(C_HNI_PML_BASE + C1_HNI_PML_STS_LLR_MAX_USAGE_OFFSET)
#define C1_HNI_PML_STS_LLR_MAX_USAGE_SIZE	0x00000008

union c_hni_pml_sts_llr_max_usage {
	uint64_t qw;
	struct {
		uint64_t max_buffer_space_used  : 12;
		uint64_t max_outstanding_seq    : 20;
		uint64_t max_starvation_time    : 16;
		uint64_t seq_starved            :  1;
		uint64_t buffer_space_starved   :  1;
		uint64_t                        : 14;
	};
};

#define SS2_PORT_PML_STS_LLR_MAX_USAGE_OFFSET(idx)	(0x00001580+((idx)*8))
#define SS2_PORT_PML_STS_LLR_MAX_USAGE_ENTRIES	4
#define SS2_PORT_PML_STS_LLR_MAX_USAGE(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_LLR_MAX_USAGE_OFFSET(idx))
#define SS2_PORT_PML_STS_LLR_MAX_USAGE_SIZE	0x00000020

union ss2_port_pml_sts_llr_max_usage {
	uint64_t qw;
	struct {
		uint64_t max_buffer_space_used  : 12;
		uint64_t max_outstanding_seq    : 20;
		uint64_t max_starvation_time    : 16;
		uint64_t seq_starved            :  1;
		uint64_t buffer_space_starved   :  1;
		uint64_t                        : 14;
	};
};

#define C1_HNI_PML_DBG_PCS_OFFSET	0x00000c10
#define C1_HNI_PML_DBG_PCS	(C_HNI_PML_BASE + C1_HNI_PML_DBG_PCS_OFFSET)
#define C1_HNI_PML_DBG_PCS_SIZE	0x00000008

union c1_hni_pml_dbg_pcs {
	uint64_t qw;
	struct {
		uint64_t tx_scrambler_en         :  1;
		uint64_t rx_descrambler_en       :  1;
		uint64_t                         :  6;
		uint64_t enable_rapid_alignment  :  1;
		uint64_t                         :  7;
		uint64_t force_bad_bip           :  5;
		uint64_t                         :  3;
		uint64_t enable_idle_check       :  1;
		uint64_t                         :  7;
		uint64_t force_tx_data           :  1;
		uint64_t force_tx_data_rf        :  1;
		uint64_t                         :  6;
		uint64_t ppm_test_mode           :  1;
		uint64_t                         : 23;
	};
};

#define SS2_PORT_PML_DBG_PCS_OFFSET(idx)	(0x00001740+((idx)*8))
#define SS2_PORT_PML_DBG_PCS_ENTRIES	4
#define SS2_PORT_PML_DBG_PCS(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_DBG_PCS_OFFSET(idx))
#define SS2_PORT_PML_DBG_PCS_SIZE	0x00000020

union ss2_port_pml_dbg_pcs {
	uint64_t qw;
	struct {
		uint64_t tx_scrambler_en         :  1;
		uint64_t rx_descrambler_en       :  1;
		uint64_t                         :  6;
		uint64_t enable_rapid_alignment  :  1;
		uint64_t                         :  7;
		uint64_t force_bad_bip           :  5;
		uint64_t                         :  3;
		uint64_t enable_idle_check       :  1;
		uint64_t                         :  7;
		uint64_t force_tx_data           :  1;
		uint64_t force_tx_data_rf        :  1;
		uint64_t                         :  6;
		uint64_t ppm_test_mode           :  1;
		uint64_t tx_clk_select           :  2;
		uint64_t                         : 21;
	};
};

#define C_IXE_CFG_CRNC_OFFSET	0x00000000
#define C_IXE_CFG_CRNC	(C_IXE_BASE + C_IXE_CFG_CRNC_OFFSET)
#define C_IXE_CFG_CRNC_SIZE	0x00000008

union c_ixe_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_IXE_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_IXE_MSC_SHADOW_ACTION	(C_IXE_BASE + C_IXE_MSC_SHADOW_ACTION_OFFSET)
#define C_IXE_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_IXE_MSC_SHADOW_OFFSET	0x00000080
#define C_IXE_MSC_SHADOW	(C_IXE_BASE + C_IXE_MSC_SHADOW_OFFSET)
#define C_IXE_MSC_SHADOW_SIZE	0x00000048

union c_ixe_msc_shadow {
	uint64_t qw[9];
};

#define C_IXE_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_IXE_ERR_ELAPSED_TIME	(C_IXE_BASE + C_IXE_ERR_ELAPSED_TIME_OFFSET)
#define C_IXE_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_ixe_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_IXE_ERR_FLG_OFFSET	0x00000108
#define C_IXE_ERR_FLG	(C_IXE_BASE + C_IXE_ERR_FLG_OFFSET)
#define C_IXE_ERR_FLG_SIZE	0x00000008

union c_ixe_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          : 23;
		uint64_t pafa_perr                :  1;
		uint64_t fifo_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t pcie_unsuccess_cmpl      :  1;
		uint64_t pcie_error_poisoned      :  1;
		uint64_t port_dfa_mismatch        :  1;
		uint64_t port_discard_trig        :  1;
		uint64_t hdr_checksum_err         :  1;
		uint64_t hdr_ecc_err              :  1;
		uint64_t get_len_err              :  1;
		uint64_t ipv4_checksum_err        :  1;
		uint64_t ipv4_options_err         :  1;
		uint64_t ipv6_options_err         :  1;
		uint64_t hrp_req_err              :  1;
		uint64_t parser_disable           :  1;
		uint64_t pbuf_pkt_perr            :  1;
		uint64_t pbuf_rd_err              :  1;
		uint64_t pkt_flit_fifo_perr       :  1;
		uint64_t amo_align_err            :  1;
		uint64_t amo_op_err               :  1;
		uint64_t amo_fp_inexact           :  1;
		uint64_t amo_fp_overflow          :  1;
		uint64_t amo_fp_underflow         :  1;
		uint64_t amo_fp_invalid           :  1;
		uint64_t rarb_hw_err              :  1;
		uint64_t amo_length_err           :  1;
		uint64_t pbuf_overflow            :  1;
		uint64_t                          :  5;
	};
};

#define C_IXE_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_IXE_ERR_FIRST_FLG	(C_IXE_BASE + C_IXE_ERR_FIRST_FLG_OFFSET)
#define C_IXE_ERR_FIRST_FLG_SIZE	0x00000008
#define C_IXE_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_IXE_ERR_FIRST_FLG_TS	(C_IXE_BASE + C_IXE_ERR_FIRST_FLG_TS_OFFSET)
#define C_IXE_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_IXE_ERR_CLR_OFFSET	0x00000120
#define C_IXE_ERR_CLR	(C_IXE_BASE + C_IXE_ERR_CLR_OFFSET)
#define C_IXE_ERR_CLR_SIZE	0x00000008
#define C_IXE_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_IXE_ERR_IRQA_MSK	(C_IXE_BASE + C_IXE_ERR_IRQA_MSK_OFFSET)
#define C_IXE_ERR_IRQA_MSK_SIZE	0x00000008
#define C_IXE_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_IXE_ERR_IRQB_MSK	(C_IXE_BASE + C_IXE_ERR_IRQB_MSK_OFFSET)
#define C_IXE_ERR_IRQB_MSK_SIZE	0x00000008
#define C_IXE_ERR_INFO_MSK_OFFSET	0x00000140
#define C_IXE_ERR_INFO_MSK	(C_IXE_BASE + C_IXE_ERR_INFO_MSK_OFFSET)
#define C_IXE_ERR_INFO_MSK_SIZE	0x00000008
#define C_IXE_EXT_ERR_FLG_OFFSET	0x00000148
#define C_IXE_EXT_ERR_FLG	(C_IXE_BASE + C_IXE_EXT_ERR_FLG_OFFSET)
#define C_IXE_EXT_ERR_FLG_SIZE	0x00000008

union c_ixe_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag             :  1;
		uint64_t flop_fifo_cor       :  1;
		uint64_t put_resp_mem_cor    :  1;
		uint64_t get_req_mem_cor     :  1;
		uint64_t rdfq_list_mem_cor   :  1;
		uint64_t wr_md_mem_cor       :  1;
		uint64_t event_fifo_cor      :  1;
		uint64_t wr_ev_mem_cor       :  1;
		uint64_t pkt_flit_fifo_cor   :  1;
		uint64_t pbuf_list_mem_cor   :  1;
		uint64_t pbuf_sb_mem_cor     :  1;
		uint64_t atu_resp_cor        :  1;
		uint64_t rdoq_cor            :  1;
		uint64_t ingressq_cor        :  1;
		uint64_t info_cor            :  1;
		uint64_t cmd_info_cor        :  1;
		uint64_t sideband_cor        :  1;
		uint64_t p_cmd_cor           :  1;
		uint64_t np_cmd_cor          :  1;
		uint64_t amo_info_cor        :  1;
		uint64_t addr_cor            :  1;
		uint64_t rd_buf_cor          :  1;
		uint64_t ack_rd_buf_cor      :  1;
		uint64_t hni_flit_cor        :  1;
		uint64_t rpu_list_mem_cor    :  1;
		uint64_t rpu_st_mem_cor      :  1;
		uint64_t rarb_hdr_bus_cor    :  1;
		uint64_t rarb_data_bus_cor   :  1;
		uint64_t pbuf_flit_mem_cor   :  1;
		uint64_t                     :  3;
		uint64_t                     :  1;
		uint64_t flop_fifo_ucor      :  1;
		uint64_t put_resp_mem_ucor   :  1;
		uint64_t get_req_mem_ucor    :  1;
		uint64_t rdfq_list_mem_ucor  :  1;
		uint64_t wr_md_mem_ucor      :  1;
		uint64_t event_fifo_ucor     :  1;
		uint64_t wr_ev_mem_ucor      :  1;
		uint64_t pkt_flit_fifo_ucor  :  1;
		uint64_t pbuf_list_mem_ucor  :  1;
		uint64_t pbuf_sb_mem_ucor    :  1;
		uint64_t atu_resp_ucor       :  1;
		uint64_t rdoq_ucor           :  1;
		uint64_t ingressq_ucor       :  1;
		uint64_t info_ucor           :  1;
		uint64_t cmd_info_ucor       :  1;
		uint64_t sideband_ucor       :  1;
		uint64_t p_cmd_ucor          :  1;
		uint64_t np_cmd_ucor         :  1;
		uint64_t amo_info_ucor       :  1;
		uint64_t addr_ucor           :  1;
		uint64_t rd_buf_ucor         :  1;
		uint64_t ack_rd_buf_ucor     :  1;
		uint64_t hni_flit_ucor       :  1;
		uint64_t rpu_list_mem_ucor   :  1;
		uint64_t rpu_st_mem_ucor     :  1;
		uint64_t rarb_hdr_bus_ucor   :  1;
		uint64_t rarb_data_bus_ucor  :  1;
		uint64_t pbuf_flit_mem_ucor  :  1;
		uint64_t                     :  3;
	};
};

#define C_IXE_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C_IXE_EXT_ERR_FIRST_FLG	(C_IXE_BASE + C_IXE_EXT_ERR_FIRST_FLG_OFFSET)
#define C_IXE_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_IXE_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C_IXE_EXT_ERR_FIRST_FLG_TS	(C_IXE_BASE + C_IXE_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_IXE_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_IXE_EXT_ERR_CLR_OFFSET	0x00000160
#define C_IXE_EXT_ERR_CLR	(C_IXE_BASE + C_IXE_EXT_ERR_CLR_OFFSET)
#define C_IXE_EXT_ERR_CLR_SIZE	0x00000008
#define C_IXE_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C_IXE_EXT_ERR_IRQA_MSK	(C_IXE_BASE + C_IXE_EXT_ERR_IRQA_MSK_OFFSET)
#define C_IXE_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_IXE_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C_IXE_EXT_ERR_IRQB_MSK	(C_IXE_BASE + C_IXE_EXT_ERR_IRQB_MSK_OFFSET)
#define C_IXE_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_IXE_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C_IXE_EXT_ERR_INFO_MSK	(C_IXE_BASE + C_IXE_EXT_ERR_INFO_MSK_OFFSET)
#define C_IXE_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_IXE_ERR_INFO_MEM_OFFSET	0x00000190
#define C_IXE_ERR_INFO_MEM	(C_IXE_BASE + C_IXE_ERR_INFO_MEM_OFFSET)
#define C_IXE_ERR_INFO_MEM_SIZE	0x00000008

union c_ixe_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 11;
		uint64_t                    :  1;
		uint64_t cor_mem_id         :  6;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 11;
		uint64_t                    :  1;
		uint64_t ucor_mem_id        :  6;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_IXE_ERR_INFO_FIFO_ERR_OFFSET	0x00000198
#define C_IXE_ERR_INFO_FIFO_ERR	(C_IXE_BASE + C_IXE_ERR_INFO_FIFO_ERR_OFFSET)
#define C_IXE_ERR_INFO_FIFO_ERR_SIZE	0x00000008

union c_ixe_err_info_fifo_err {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           :  6;
		uint64_t fifo_id   :  5;
		uint64_t           : 51;
	};
};

#define C_IXE_ERR_INFO_CC_OFFSET	0x000001a0
#define C_IXE_ERR_INFO_CC	(C_IXE_BASE + C_IXE_ERR_INFO_CC_OFFSET)
#define C_IXE_ERR_INFO_CC_SIZE	0x00000008

union c_ixe_err_info_cc {
	uint64_t qw;
	struct {
		uint64_t cc_id   :  4;
		uint64_t         :  4;
		uint64_t sub_id  :  5;
		uint64_t         : 51;
	};
};

#define C_IXE_ERR_INFO_PCIE_EP_OFFSET	0x00000200
#define C_IXE_ERR_INFO_PCIE_EP	(C_IXE_BASE + C_IXE_ERR_INFO_PCIE_EP_OFFSET)
#define C_IXE_ERR_INFO_PCIE_EP_SIZE	0x00000008

union c_ixe_err_info_pcie_ep {
	uint64_t qw;
	struct {
		uint64_t pcie_tag  :  9;
		uint64_t           : 55;
	};
};

#define C_IXE_ERR_INFO_PCIE_UC_OFFSET	0x00000208
#define C_IXE_ERR_INFO_PCIE_UC	(C_IXE_BASE + C_IXE_ERR_INFO_PCIE_UC_OFFSET)
#define C_IXE_ERR_INFO_PCIE_UC_SIZE	0x00000008

union c_ixe_err_info_pcie_uc {
	uint64_t qw;
	struct {
		uint64_t pcie_tag     :  9;
		uint64_t              :  3;
		uint64_t cmpl_status  :  3;
		uint64_t              : 49;
	};
};

#define C_IXE_ERR_INFO_ETH_PARSE_OFFSET	0x00000210
#define C_IXE_ERR_INFO_ETH_PARSE	(C_IXE_BASE + C_IXE_ERR_INFO_ETH_PARSE_OFFSET)
#define C_IXE_ERR_INFO_ETH_PARSE_SIZE	0x00000008

union c_ixe_err_info_eth_parse {
	uint64_t qw;
	struct {
		uint64_t ptlte_index        : 11;
		uint64_t                    :  1;
		uint64_t ipv4_checksum_err  :  1;
		uint64_t ipv4_options_err   :  1;
		uint64_t ipv6_options_err   :  1;
		uint64_t                    : 49;
	};
};

#define C_IXE_ERR_INFO_RARB_OFFSET	0x00000218
#define C_IXE_ERR_INFO_RARB	(C_IXE_BASE + C_IXE_ERR_INFO_RARB_OFFSET)
#define C_IXE_ERR_INFO_RARB_SIZE	0x00000008

union c_ixe_err_info_rarb {
	uint64_t qw;
	struct {
		uint64_t hdr_ucor               :  1;
		uint64_t data_ucor              :  1;
		uint64_t bad_cmpl_last          :  1;
		uint64_t bad_vld                :  1;
		uint64_t bad_eop                :  1;
		uint64_t bad_cmd                :  1;
		uint64_t unexpected_cmpl_tag    :  1;
		uint64_t cmpl_tag_out_of_range  :  1;
		uint64_t                        : 56;
	};
};

#define C_IXE_ERR_INFO_FP_INVALID_OFFSET	0x00000220
#define C_IXE_ERR_INFO_FP_INVALID	(C_IXE_BASE + C_IXE_ERR_INFO_FP_INVALID_OFFSET)
#define C_IXE_ERR_INFO_FP_INVALID_SIZE	0x00000010

union c_ixe_err_info_fp_invalid {
	uint64_t qw[2];
	struct {
		uint64_t              :  2;
		uint64_t addr_56_2    : 55;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t              : 48;
	};
};

#define C_IXE_ERR_INFO_FP_UNDERFLOW_OFFSET	0x00000230
#define C_IXE_ERR_INFO_FP_UNDERFLOW	(C_IXE_BASE + C_IXE_ERR_INFO_FP_UNDERFLOW_OFFSET)
#define C_IXE_ERR_INFO_FP_UNDERFLOW_SIZE	0x00000010

union c_ixe_err_info_fp_underflow {
	uint64_t qw[2];
	struct {
		uint64_t              :  2;
		uint64_t addr_56_2    : 55;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t              : 48;
	};
};

#define C_IXE_ERR_INFO_FP_OVERFLOW_OFFSET	0x00000240
#define C_IXE_ERR_INFO_FP_OVERFLOW	(C_IXE_BASE + C_IXE_ERR_INFO_FP_OVERFLOW_OFFSET)
#define C_IXE_ERR_INFO_FP_OVERFLOW_SIZE	0x00000010

union c_ixe_err_info_fp_overflow {
	uint64_t qw[2];
	struct {
		uint64_t              :  2;
		uint64_t addr_56_2    : 55;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t              : 48;
	};
};

#define C_IXE_ERR_INFO_FP_INEXACT_OFFSET	0x00000250
#define C_IXE_ERR_INFO_FP_INEXACT	(C_IXE_BASE + C_IXE_ERR_INFO_FP_INEXACT_OFFSET)
#define C_IXE_ERR_INFO_FP_INEXACT_SIZE	0x00000010

union c_ixe_err_info_fp_inexact {
	uint64_t qw[2];
	struct {
		uint64_t              :  2;
		uint64_t addr_56_2    : 55;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t              : 48;
	};
};

#define C_IXE_ERR_INFO_AMO_ALIGN_OFFSET	0x00000260
#define C_IXE_ERR_INFO_AMO_ALIGN	(C_IXE_BASE + C_IXE_ERR_INFO_AMO_ALIGN_OFFSET)
#define C_IXE_ERR_INFO_AMO_ALIGN_SIZE	0x00000010

union c_ixe_err_info_amo_align {
	uint64_t qw[2];
	struct {
		uint64_t addr_56_0    : 57;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t              : 48;
	};
};

#define C_IXE_ERR_INFO_AMO_OP_OFFSET	0x00000270
#define C_IXE_ERR_INFO_AMO_OP	(C_IXE_BASE + C_IXE_ERR_INFO_AMO_OP_OFFSET)
#define C_IXE_ERR_INFO_AMO_OP_SIZE	0x00000010

union c_ixe_err_info_amo_op {
	uint64_t qw[2];
	struct {
		uint64_t addr_56_0    : 57;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t cswap_op     :  3;
		uint64_t              :  1;
		uint64_t atomic_op    :  4;
		uint64_t              : 40;
	};
};

#define C_IXE_ERR_INFO_AMO_LENGTH_OFFSET	0x00000280
#define C_IXE_ERR_INFO_AMO_LENGTH	(C_IXE_BASE + C_IXE_ERR_INFO_AMO_LENGTH_OFFSET)
#define C_IXE_ERR_INFO_AMO_LENGTH_SIZE	0x00000010

union c_ixe_err_info_amo_length {
	uint64_t qw[2];
	struct {
		uint64_t addr_56_0    : 57;
		uint64_t              :  7;
		uint64_t acid         : 10;
		uint64_t              :  2;
		uint64_t atomic_type  :  4;
		uint64_t cmd_length   : 13;
		uint64_t              : 35;
	};
};

#define C_IXE_CFG_DSCP_WRQ_MAP_OFFSET(idx)	(0x00000400+((idx)*8))
#define C_IXE_CFG_DSCP_WRQ_MAP_ENTRIES	64
#define C_IXE_CFG_DSCP_WRQ_MAP(idx)	(C_IXE_BASE + C_IXE_CFG_DSCP_WRQ_MAP_OFFSET(idx))
#define C_IXE_CFG_DSCP_WRQ_MAP_SIZE	0x00000200

union c_ixe_cfg_dscp_wrq_map {
	uint64_t qw;
	struct {
		uint64_t mask0      : 22;
		uint64_t            :  2;
		uint64_t mask1      : 22;
		uint64_t            :  2;
		uint64_t hash_mask  :  2;
		uint64_t            :  6;
		uint64_t base       :  5;
		uint64_t            :  3;
	};
};

#define C_IXE_CFG_DSCP_DSCP_TC_MAP_OFFSET(idx)	(0x00000600+((idx)*8))
#define C_IXE_CFG_DSCP_DSCP_TC_MAP_ENTRIES	16
#define C_IXE_CFG_DSCP_DSCP_TC_MAP(idx)	(C_IXE_BASE + C_IXE_CFG_DSCP_DSCP_TC_MAP_OFFSET(idx))
#define C_IXE_CFG_DSCP_DSCP_TC_MAP_SIZE	0x00000080

struct c_dscp_map {
	uint8_t dscp    : 6;
	uint8_t         : 2;
	uint8_t tc      : 3;
	uint8_t         : 5;
};
#define C_IXE_CFG_DSCP_DSCP_TC_MAP_ARRAY_SIZE 4
union c_ixe_cfg_dscp_dscp_tc_map {
	uint64_t qw;
	struct c_dscp_map map[C_IXE_CFG_DSCP_DSCP_TC_MAP_ARRAY_SIZE];
};

#define C_IXE_CFG_TC_FQ_MAP_OFFSET(idx)	(0x00000700+((idx)*8))
#define C_IXE_CFG_TC_FQ_MAP_ENTRIES	8
#define C_IXE_CFG_TC_FQ_MAP(idx)	(C_IXE_BASE + C_IXE_CFG_TC_FQ_MAP_OFFSET(idx))
#define C_IXE_CFG_TC_FQ_MAP_SIZE	0x00000040

union c_ixe_cfg_tc_fq_map {
	uint64_t qw;
	struct {
		uint64_t put_resp_mask  : 24;
		uint64_t                :  8;
		uint64_t get_req_mask   : 24;
		uint64_t                :  8;
	};
};

#define C_IXE_CFG_MP_TC_FQ_MAP_OFFSET(idx)	(0x00000740+((idx)*8))
#define C_IXE_CFG_MP_TC_FQ_MAP_ENTRIES	8
#define C_IXE_CFG_MP_TC_FQ_MAP(idx)	(C_IXE_BASE + C_IXE_CFG_MP_TC_FQ_MAP_OFFSET(idx))
#define C_IXE_CFG_MP_TC_FQ_MAP_SIZE	0x00000040

union c_ixe_cfg_mp_tc_fq_map {
	uint64_t qw;
	struct {
		uint64_t num_fqs_mask  :  2;
		uint64_t               : 14;
		uint64_t base          :  5;
		uint64_t               : 43;
	};
};

#define C_IXE_CFG_DISP_CDT_LIM_OFFSET	0x00000788
#define C_IXE_CFG_DISP_CDT_LIM	(C_IXE_BASE + C_IXE_CFG_DISP_CDT_LIM_OFFSET)
#define C_IXE_CFG_DISP_CDT_LIM_SIZE	0x00000008

union c_ixe_cfg_disp_cdt_lim {
	uint64_t qw;
	struct {
		uint64_t mst_match_cdts  :  6;
		uint64_t                 :  2;
		uint64_t wrq_ff_cdts     :  2;
		uint64_t                 :  2;
		uint64_t atu_req_cdts    :  3;
		uint64_t                 :  1;
		uint64_t rpu_err_cdts    :  3;
		uint64_t                 :  1;
		uint64_t rpu_resp_cdts   :  3;
		uint64_t                 :  1;
		uint64_t pct_gcomp_cdts  :  6;
		uint64_t                 :  2;
		uint64_t put_resp_cdts   : 10;
		uint64_t                 :  6;
		uint64_t put_atu_cdts    :  9;
		uint64_t ee_cdts         :  5;
		uint64_t                 :  2;
	};
};

#define C_IXE_CFG_AMO_OFFLOAD_EN_OFFSET	0x00000800
#define C_IXE_CFG_AMO_OFFLOAD_EN	(C_IXE_BASE + C_IXE_CFG_AMO_OFFLOAD_EN_OFFSET)
#define C_IXE_CFG_AMO_OFFLOAD_EN_SIZE	0x00000008

union c_ixe_cfg_amo_offload_en {
	uint64_t qw;
	struct {
		uint64_t op_min       :  4;
		uint64_t op_max       :  4;
		uint64_t op_sum       :  4;
		uint64_t op_lor       :  4;
		uint64_t op_land      :  4;
		uint64_t op_bor       :  4;
		uint64_t op_band      :  4;
		uint64_t op_lxor      :  4;
		uint64_t op_bxor      :  4;
		uint64_t op_swap      :  4;
		uint64_t op_cswap_eq  :  3;
		uint64_t              :  1;
		uint64_t op_cswap_ne  :  2;
		uint64_t op_cswap_le  :  2;
		uint64_t op_cswap_lt  :  2;
		uint64_t op_cswap_ge  :  2;
		uint64_t op_cswap_gt  :  2;
		uint64_t op_axor      :  2;
		uint64_t              :  8;
	};
};

#define C_IXE_CFG_AMO_OFFLOAD_CODE_1OP_OFFSET	0x00000808
#define C_IXE_CFG_AMO_OFFLOAD_CODE_1OP	(C_IXE_BASE + C_IXE_CFG_AMO_OFFLOAD_CODE_1OP_OFFSET)
#define C_IXE_CFG_AMO_OFFLOAD_CODE_1OP_SIZE	0x00000008

union c_ixe_cfg_amo_offload_code_1op {
	uint64_t qw;
	struct {
		uint64_t op_min_code   :  4;
		uint64_t op_max_code   :  4;
		uint64_t op_sum_code   :  4;
		uint64_t op_lor_code   :  4;
		uint64_t op_land_code  :  4;
		uint64_t op_bor_code   :  4;
		uint64_t op_band_code  :  4;
		uint64_t op_lxor_code  :  4;
		uint64_t op_bxor_code  :  4;
		uint64_t op_swap_code  :  4;
		uint64_t               : 24;
	};
};

#define C_IXE_CFG_AMO_OFFLOAD_OFFSET	0x00000818
#define C_IXE_CFG_AMO_OFFLOAD	(C_IXE_BASE + C_IXE_CFG_AMO_OFFLOAD_OFFSET)
#define C_IXE_CFG_AMO_OFFLOAD_SIZE	0x00000008

union c_ixe_cfg_amo_offload {
	uint64_t qw;
	struct {
		uint64_t base_amo_req           :  1;
		uint64_t force_native_swap      :  1;
		uint64_t force_native_fetchadd  :  1;
		uint64_t                        : 13;
		uint64_t first_dw_be_32bit      :  4;
		uint64_t last_dw_be_32bit       :  4;
		uint64_t first_dw_be_64bit      :  4;
		uint64_t last_dw_be_64bit       :  4;
		uint64_t                        : 32;
	};
};

#define C_IXE_CFG_PARSER_OFFSET	0x00000908
#define C_IXE_CFG_PARSER	(C_IXE_BASE + C_IXE_CFG_PARSER_OFFSET)
#define C_IXE_CFG_PARSER_SIZE	0x00000008

union c_ixe_cfg_parser {
	uint64_t qw;
	struct {
		uint64_t ipv4_flowlab_type  :  8;
		uint64_t ipv6_frag_type     :  8;
		uint64_t udp_type           :  8;
		uint64_t tcp_type           :  8;
		uint64_t eth_segment        :  5;
		uint64_t                    :  3;
		uint64_t rpu_wrdisp_cdt     :  4;
		uint64_t                    : 20;
	};
};

#define C_IXE_CFG_ROCE_OFFSET	0x00000910
#define C_IXE_CFG_ROCE	(C_IXE_BASE + C_IXE_CFG_ROCE_OFFSET)
#define C_IXE_CFG_ROCE_SIZE	0x00000008

union c_ixe_cfg_roce {
	uint64_t qw;
	struct {
		uint64_t udp_dstport_bth  : 16;
		uint64_t enable_1k        :  1;
		uint64_t enable_2k        :  1;
		uint64_t enable_4k        :  1;
		uint64_t                  : 45;
	};
};

#define C_IXE_CFG_HRP_OFFSET	0x00000918
#define C_IXE_CFG_HRP	(C_IXE_BASE + C_IXE_CFG_HRP_OFFSET)
#define C_IXE_CFG_HRP_SIZE	0x00000008

union c_ixe_cfg_hrp {
	uint64_t qw;
	struct {
		uint64_t hrp_enable;
	};
};

#define C_IXE_CFG_DFA_OFFSET	0x00000920
#define C_IXE_CFG_DFA	(C_IXE_BASE + C_IXE_CFG_DFA_OFFSET)
#define C_IXE_CFG_DFA_SIZE	0x00000008

union c_ixe_cfg_dfa {
	uint64_t qw;
	struct {
		uint64_t             : 12;
		uint64_t dfa_nid     : 20;
		uint64_t dfa_chk_en  :  8;
		uint64_t             : 24;
	};
};

#define C_IXE_CFG_DISCARD_OFFSET	0x00000928
#define C_IXE_CFG_DISCARD	(C_IXE_BASE + C_IXE_CFG_DISCARD_OFFSET)
#define C_IXE_CFG_DISCARD_SIZE	0x00000008

union c_ixe_cfg_discard {
	uint64_t qw;
	struct {
		uint64_t v4_pkt_type_mask   :  8;
		uint64_t v4_pkt_type_match  :  8;
		uint64_t vs_pkt_type_mask   :  8;
		uint64_t vs_pkt_type_match  :  8;
		uint64_t armed              :  8;
		uint64_t                    : 24;
	};
};

#define C_IXE_CFG_IPV6_OPT_ABORT_OFFSET(idx)	(0x00000960+((idx)*8))
#define C_IXE_CFG_IPV6_OPT_ABORT_ENTRIES	4
#define C_IXE_CFG_IPV6_OPT_ABORT(idx)	(C_IXE_BASE + C_IXE_CFG_IPV6_OPT_ABORT_OFFSET(idx))
#define C_IXE_CFG_IPV6_OPT_ABORT_SIZE	0x00000020

union c_ixe_cfg_ipv6_opt_abort {
	uint64_t qw;
	struct {
		uint64_t abort;
	};
};

#define C_IXE_CFG_BTH_OPCODE_OFFSET(idx)	(0x00000980+((idx)*8))
#define C_IXE_CFG_BTH_OPCODE_ENTRIES	16
#define C_IXE_CFG_BTH_OPCODE(idx)	(C_IXE_BASE + C_IXE_CFG_BTH_OPCODE_OFFSET(idx))
#define C_IXE_CFG_BTH_OPCODE_SIZE	0x00000080

union c_ixe_cfg_bth_opcode {
	uint64_t qw;
	struct {
		uint64_t len_0   :  4;
		uint64_t len_1   :  4;
		uint64_t len_2   :  4;
		uint64_t len_3   :  4;
		uint64_t len_4   :  4;
		uint64_t len_5   :  4;
		uint64_t len_6   :  4;
		uint64_t len_7   :  4;
		uint64_t len_8   :  4;
		uint64_t len_9   :  4;
		uint64_t len_10  :  4;
		uint64_t len_11  :  4;
		uint64_t len_12  :  4;
		uint64_t len_13  :  4;
		uint64_t len_14  :  4;
		uint64_t len_15  :  4;
	};
};

#define C_IXE_STS_PARSER_DECODE_OFFSET	0x00000c40
#define C_IXE_STS_PARSER_DECODE	(C_IXE_BASE + C_IXE_STS_PARSER_DECODE_OFFSET)
#define C_IXE_STS_PARSER_DECODE_SIZE	0x00000008

union c_ixe_sts_parser_decode {
	uint64_t qw;
	struct {
		uint64_t flit_fifo_depth     :  6;
		uint64_t                     :  2;
		uint64_t pkt_fifo_depth      :  6;
		uint64_t                     :  2;
		uint64_t pct_req_fifo_depth  :  6;
		uint64_t                     :  2;
		uint64_t pct_rsp_fifo_depth  :  6;
		uint64_t                     :  2;
		uint64_t rmu_fifo_depth      :  6;
		uint64_t                     : 26;
	};
};

#define C_LPE_CFG_CRNC_OFFSET	0x00000000
#define C_LPE_CFG_CRNC	(C_LPE_BASE + C_LPE_CFG_CRNC_OFFSET)
#define C_LPE_CFG_CRNC_SIZE	0x00000008

union c_lpe_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_LPE_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_LPE_MSC_SHADOW_ACTION	(C_LPE_BASE + C_LPE_MSC_SHADOW_ACTION_OFFSET)
#define C_LPE_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_LPE_MSC_SHADOW_OFFSET	0x00000040
#define C_LPE_MSC_SHADOW	(C_LPE_BASE + C_LPE_MSC_SHADOW_OFFSET)
#define C_LPE_MSC_SHADOW_SIZE	0x00000040

union c_lpe_msc_shadow {
	uint64_t qw[8];
};

#define C_LPE_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_LPE_ERR_ELAPSED_TIME	(C_LPE_BASE + C_LPE_ERR_ELAPSED_TIME_OFFSET)
#define C_LPE_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_lpe_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_LPE_ERR_FLG_OFFSET	0x00000108
#define C_LPE_ERR_FLG	(C_LPE_BASE + C_LPE_ERR_FLG_OFFSET)
#define C_LPE_ERR_FLG_SIZE	0x00000008

union c_lpe_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t event_cnt_ovf            :  1;
		uint64_t le_oversubscribed        :  1;
		uint64_t                          :  1;
		uint64_t illegal_op               :  1;
		uint64_t src_error                :  1;
		uint64_t delayed                  :  1;
		uint64_t restricted_unicast       :  1;
		uint64_t no_space                 :  1;
		uint64_t invalid_endpoint         :  1;
		uint64_t entry_not_found          :  1;
		uint64_t op_violation             :  1;
		uint64_t eq_disabled              :  1;
		uint64_t pt_disabled_fc           :  1;
		uint64_t pt_disabled_dis          :  1;
		uint64_t pt_disabled_rst          :  1;
		uint64_t illegal_cq_op            :  1;
		uint64_t fifo_err                 :  1;
		uint64_t cq_fifo_overrun          :  1;
		uint64_t cdt_cnt_uflow            :  1;
		uint64_t mrr_mem_cor              :  1;
		uint64_t free_list_mem_cor        :  1;
		uint64_t event_ram_cor            :  1;
		uint64_t plec_mem_cor             :  1;
		uint64_t ptlte_mem_cor            :  1;
		uint64_t ixemrq_front_mem_cor     :  1;
		uint64_t ixemrq_back_mem_cor      :  1;
		uint64_t ixemrq_list_mem_cor      :  1;
		uint64_t petcq_list_mem_cor       :  1;
		uint64_t list_entry_mem_cor       :  1;
		uint64_t cqmrq_front_mem_cor      :  1;
		uint64_t cqmrq_back_mem_cor       :  1;
		uint64_t cqmrq_list_mem_cor       :  1;
		uint64_t rget_ram_mem_cor         :  1;
		uint64_t rrq_fifo_cor             :  1;
		uint64_t cq_fifo_cor              :  1;
		uint64_t                          :  1;
		uint64_t rdy_pleq_cor             :  1;
		uint64_t mrr_mem_ucor             :  1;
		uint64_t free_list_mem_ucor       :  1;
		uint64_t event_ram_ucor           :  1;
		uint64_t plec_mem_ucor            :  1;
		uint64_t ptlte_mem_ucor           :  1;
		uint64_t ixemrq_front_mem_ucor    :  1;
		uint64_t ixemrq_back_mem_ucor     :  1;
		uint64_t ixemrq_list_mem_ucor     :  1;
		uint64_t petcq_list_mem_ucor      :  1;
		uint64_t list_entry_mem_ucor      :  1;
		uint64_t cqmrq_front_mem_ucor     :  1;
		uint64_t cqmrq_back_mem_ucor      :  1;
		uint64_t cqmrq_list_mem_ucor      :  1;
		uint64_t rget_ram_mem_ucor        :  1;
		uint64_t rrq_fifo_ucor            :  1;
		uint64_t cq_fifo_ucor             :  1;
		uint64_t                          :  1;
		uint64_t rdy_pleq_ucor            :  1;
	};
};

#define C_LPE_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_LPE_ERR_FIRST_FLG	(C_LPE_BASE + C_LPE_ERR_FIRST_FLG_OFFSET)
#define C_LPE_ERR_FIRST_FLG_SIZE	0x00000008
#define C_LPE_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_LPE_ERR_FIRST_FLG_TS	(C_LPE_BASE + C_LPE_ERR_FIRST_FLG_TS_OFFSET)
#define C_LPE_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_LPE_ERR_CLR_OFFSET	0x00000120
#define C_LPE_ERR_CLR	(C_LPE_BASE + C_LPE_ERR_CLR_OFFSET)
#define C_LPE_ERR_CLR_SIZE	0x00000008
#define C_LPE_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_LPE_ERR_IRQA_MSK	(C_LPE_BASE + C_LPE_ERR_IRQA_MSK_OFFSET)
#define C_LPE_ERR_IRQA_MSK_SIZE	0x00000008
#define C_LPE_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_LPE_ERR_IRQB_MSK	(C_LPE_BASE + C_LPE_ERR_IRQB_MSK_OFFSET)
#define C_LPE_ERR_IRQB_MSK_SIZE	0x00000008
#define C_LPE_ERR_INFO_MSK_OFFSET	0x00000140
#define C_LPE_ERR_INFO_MSK	(C_LPE_BASE + C_LPE_ERR_INFO_MSK_OFFSET)
#define C_LPE_ERR_INFO_MSK_SIZE	0x00000008
#define C_LPE_ERR_INFO_MEM_OFFSET	0x00000180
#define C_LPE_ERR_INFO_MEM	(C_LPE_BASE + C_LPE_ERR_INFO_MEM_OFFSET)
#define C_LPE_ERR_INFO_MEM_SIZE	0x00000008

union c_lpe_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t cor_address        : 14;
		uint64_t cor_mem_id         :  5;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t ucor_address       : 14;
		uint64_t ucor_mem_id        :  5;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_LPE_ERR_INFO_CDT_CNT_UFLOW_OFFSET	0x00000188
#define C_LPE_ERR_INFO_CDT_CNT_UFLOW	(C_LPE_BASE + C_LPE_ERR_INFO_CDT_CNT_UFLOW_OFFSET)
#define C_LPE_ERR_INFO_CDT_CNT_UFLOW_SIZE	0x00000008

union c_lpe_err_info_cdt_cnt_uflow {
	uint64_t qw;
	struct {
		uint64_t cdt_cnt_id  :  4;
		uint64_t             : 60;
	};
};

#define C_LPE_ERR_INFO_NET_HDR_OFFSET	0x00000190
#define C_LPE_ERR_INFO_NET_HDR	(C_LPE_BASE + C_LPE_ERR_INFO_NET_HDR_OFFSET)
#define C_LPE_ERR_INFO_NET_HDR_SIZE	0x00000008

union c_lpe_err_info_net_hdr {
	uint64_t qw;
	struct {
		uint64_t initiator       : 32;
		uint64_t ptlte_index     : 11;
		uint64_t src_error       :  1;
		uint64_t tc              :  3;
		uint64_t                 :  1;
		uint64_t opcode          :  4;
		uint64_t net_hdr_err_id  :  4;
		uint64_t                 :  8;
	};
};

#define C_LPE_CFG_FQ_TC_MAP_OFFSET	0x00000410
#define C_LPE_CFG_FQ_TC_MAP	(C_LPE_BASE + C_LPE_CFG_FQ_TC_MAP_OFFSET)
#define C_LPE_CFG_FQ_TC_MAP_SIZE	0x00000008

struct c_lpe_cfg_tc_fqs {
	uint8_t fq_base	: 4;
	uint8_t fq_mask	: 3;
	uint8_t		: 1;
};
union c_lpe_cfg_fq_tc_map {
	uint64_t qw;
	struct c_lpe_cfg_tc_fqs tc[8];
};

#define C_LPE_CFG_CDT_LIMITS_OFFSET	0x00000418
#define C_LPE_CFG_CDT_LIMITS	(C_LPE_BASE + C_LPE_CFG_CDT_LIMITS_OFFSET)
#define C_LPE_CFG_CDT_LIMITS_SIZE	0x00000008

union c_lpe_cfg_cdt_limits {
	uint64_t qw;
	struct {
		uint64_t ixemrq_wc_cdts  :  4;
		uint64_t cqmrq_wc_cdts   :  4;
		uint64_t rdy_ple_cdts    :  4;
		uint64_t event_cdts      :  5;
		uint64_t                 :  3;
		uint64_t ixe_put_cdts    :  4;
		uint64_t ixe_get_cdts    :  4;
		uint64_t ixe_mst_cdts    :  4;
		uint64_t rrq_cdts        :  3;
		uint64_t                 : 29;
	};
};

#define C_LPE_CFG_GET_CTRL_OFFSET	0x00000428
#define C_LPE_CFG_GET_CTRL	(C_LPE_BASE + C_LPE_CFG_GET_CTRL_OFFSET)
#define C_LPE_CFG_GET_CTRL_SIZE	0x00000008

union c_lpe_cfg_get_ctrl {
	uint64_t qw;
	struct {
		uint64_t get_index         : 12;
		uint64_t                   :  4;
		uint64_t get_index_ext     :  5;
		uint64_t get_pid_shift     :  3;
		uint64_t transaction_type  :  4;
		uint64_t                   :  3;
		uint64_t get_en            :  1;
		uint64_t dfa_nid           : 20;
		uint64_t                   : 12;
	};
};

#define C_LPE_CFG_INITIATOR_CTRL_OFFSET	0x00000430
#define C_LPE_CFG_INITIATOR_CTRL	(C_LPE_BASE + C_LPE_CFG_INITIATOR_CTRL_OFFSET)
#define C_LPE_CFG_INITIATOR_CTRL_SIZE	0x00000008

union c_lpe_cfg_initiator_ctrl {
	uint64_t qw;
	struct {
		uint64_t pid_bits  :  4;
		uint64_t           : 60;
	};
};

#define C_LPE_CFG_ETHERNET_THRESHOLD_OFFSET	0x00000438
#define C_LPE_CFG_ETHERNET_THRESHOLD	(C_LPE_BASE + C_LPE_CFG_ETHERNET_THRESHOLD_OFFSET)
#define C_LPE_CFG_ETHERNET_THRESHOLD_SIZE	0x00000008

union c_lpe_cfg_ethernet_threshold {
	uint64_t qw;
	struct {
		uint64_t threshold  : 14;
		uint64_t            : 50;
	};
};

#define C_LPE_CFG_MIN_FREE_SHFT_OFFSET	0x00000440
#define C_LPE_CFG_MIN_FREE_SHFT	(C_LPE_BASE + C_LPE_CFG_MIN_FREE_SHFT_OFFSET)
#define C_LPE_CFG_MIN_FREE_SHFT_SIZE	0x00000008

union c_lpe_cfg_min_free_shft {
	uint64_t qw;
	struct {
		uint64_t min_free_shft  :  4;
		uint64_t                : 60;
	};
};

#define C_LPE_CFG_EVENT_CNTS_OFFSET	0x00000448
#define C_LPE_CFG_EVENT_CNTS	(C_LPE_BASE + C_LPE_CFG_EVENT_CNTS_OFFSET)
#define C_LPE_CFG_EVENT_CNTS_SIZE	0x00000008

union c_lpe_cfg_event_cnts {
	uint64_t qw;
	struct {
		uint64_t enable     :  1;
		uint64_t init       :  1;
		uint64_t init_done  :  1;
		uint64_t            :  1;
		uint64_t ovf_cnt    : 16;
		uint64_t            : 44;
	};
};

#define C_LPE_CFG_PE_LE_POOLS_OFFSET(idx)	(0x00000480+((idx)*8))
#define C_LPE_CFG_PE_LE_POOLS_ENTRIES	64
#define C_LPE_CFG_PE_LE_POOLS(idx)	(C_LPE_BASE + C_LPE_CFG_PE_LE_POOLS_OFFSET(idx))
#define C_LPE_CFG_PE_LE_POOLS_SIZE	0x00000200

union c_lpe_cfg_pe_le_pools {
	uint64_t qw;
	struct {
		uint64_t num_reserved  : 15;
		uint64_t               : 17;
		uint64_t max_alloc     : 15;
		uint64_t               : 17;
	};
};

#define C_LPE_STS_PE_LE_ALLOC_OFFSET(idx)	(0x00000800+((idx)*8))
#define C_LPE_STS_PE_LE_ALLOC_ENTRIES	64
#define C_LPE_STS_PE_LE_ALLOC(idx)	(C_LPE_BASE + C_LPE_STS_PE_LE_ALLOC_OFFSET(idx))
#define C_LPE_STS_PE_LE_ALLOC_SIZE	0x00000200

union c_lpe_sts_pe_le_alloc {
	uint64_t qw;
	struct {
		uint64_t num_allocated  : 15;
		uint64_t                : 49;
	};
};

#define C_LPE_CFG_PE_LE_SHARED_OFFSET(idx)	(0x00000a00+((idx)*8))
#define C_LPE_CFG_PE_LE_SHARED_ENTRIES	4
#define C_LPE_CFG_PE_LE_SHARED(idx)	(C_LPE_BASE + C_LPE_CFG_PE_LE_SHARED_OFFSET(idx))
#define C_LPE_CFG_PE_LE_SHARED_SIZE	0x00000020

union c_lpe_cfg_pe_le_shared {
	uint64_t qw;
	struct {
		uint64_t num_shared_allocated  : 15;
		uint64_t                       :  1;
		uint64_t num_total_allocated   : 15;
		uint64_t                       :  1;
		uint64_t num_shared            : 15;
		uint64_t                       :  1;
		uint64_t num_total             : 15;
		uint64_t                       :  1;
	};
};

#define C_LPE_STS_INIT_DONE_OFFSET	0x00000a28
#define C_LPE_STS_INIT_DONE	(C_LPE_BASE + C_LPE_STS_INIT_DONE_OFFSET)
#define C_LPE_STS_INIT_DONE_SIZE	0x00000008

union c_lpe_sts_init_done {
	uint64_t qw;
	struct {
		uint64_t ptlte_init_done      :  1;
		uint64_t free_mask_init_done  :  4;
		uint64_t                      : 59;
	};
};

#define C_LPE_STS_CDTS_IN_USE_OFFSET	0x00000a30
#define C_LPE_STS_CDTS_IN_USE	(C_LPE_BASE + C_LPE_STS_CDTS_IN_USE_OFFSET)
#define C_LPE_STS_CDTS_IN_USE_SIZE	0x00000008

union c_lpe_sts_cdts_in_use {
	uint64_t qw;
	struct {
		uint64_t ixemrq_wc_cdts_in_use  :  4;
		uint64_t cqmrq_wc_cdts_in_use   :  4;
		uint64_t rdy_ple_cdts_in_use    :  4;
		uint64_t event_cdts_in_use      :  5;
		uint64_t                        :  3;
		uint64_t ixe_put_cdts_in_use    :  4;
		uint64_t ixe_get_cdts_in_use    :  4;
		uint64_t ixe_mst_cdts_in_use    :  4;
		uint64_t pe0_rrq_cdts_in_use    :  3;
		uint64_t                        :  1;
		uint64_t pe1_rrq_cdts_in_use    :  3;
		uint64_t                        :  1;
		uint64_t pe2_rrq_cdts_in_use    :  3;
		uint64_t                        :  1;
		uint64_t pe3_rrq_cdts_in_use    :  3;
		uint64_t                        : 17;
	};
};

#define C_LPE_MSC_LE_POOL_FREE_OFFSET(idx)	(0x00000a40+((idx)*8))
#define C_LPE_MSC_LE_POOL_FREE_ENTRIES	4
#define C_LPE_MSC_LE_POOL_FREE(idx)	(C_LPE_BASE + C_LPE_MSC_LE_POOL_FREE_OFFSET(idx))
#define C_LPE_MSC_LE_POOL_FREE_SIZE	0x00000020

union c_lpe_msc_le_pool_free {
	uint64_t qw;
	struct {
		uint64_t le_pool  :  4;
		uint64_t free_en  :  1;
		uint64_t          : 59;
	};
};

#define C_LPE_MSC_LOAD_STATS_DOORBELL_OFFSET	0x00000a60
#define C_LPE_MSC_LOAD_STATS_DOORBELL	(C_LPE_BASE + C_LPE_MSC_LOAD_STATS_DOORBELL_OFFSET)
#define C_LPE_MSC_LOAD_STATS_DOORBELL_SIZE	0x00000008

union c_lpe_msc_load_stats_doorbell {
	uint64_t qw;
	struct {
		uint64_t doorbell  :  1;
		uint64_t           : 63;
	};
};

#define C_LPE_STS_PE_FREE_MASK_OFFSET(idx)	(0x00001000+((idx)*8))
#define C_LPE_STS_PE_FREE_MASK_ENTRIES	1024
#define C_LPE_STS_PE_FREE_MASK(idx)	(C_LPE_BASE + C_LPE_STS_PE_FREE_MASK_OFFSET(idx))
#define C_LPE_STS_PE_FREE_MASK_SIZE	0x00002000

union c_lpe_sts_pe_free_mask {
	uint64_t qw;
	struct {
		uint64_t free_mask;
	};
};

#define C_LPE_STS_LOAD_STATS_OFFSET(idx)	(0x00003000+((idx)*8))
#define C_LPE_STS_LOAD_STATS_ENTRIES	4
#define C_LPE_STS_LOAD_STATS(idx)	(C_LPE_BASE + C_LPE_STS_LOAD_STATS_OFFSET(idx))
#define C_LPE_STS_LOAD_STATS_SIZE	0x00000020

union c_lpe_sts_load_stats {
	uint64_t qw;
	struct {
		uint64_t load  : 48;
		uint64_t       : 12;
		uint64_t max   :  4;
	};
};

#define C_LPE_CFG_PTL_TABLE_OFFSET(idx)	(0x00010000+((idx)*32))
#define C_LPE_CFG_PTL_TABLE_ENTRIES	2048
#define C_LPE_CFG_PTL_TABLE(idx)	(C_LPE_BASE + C_LPE_CFG_PTL_TABLE_OFFSET(idx))
#define C_LPE_CFG_PTL_TABLE_SIZE	0x00010000

struct c_lpe_cfg_ptl_table_ptrs {
	uint32_t tail		  : 14;
	uint32_t                  :  2;
	uint32_t head             : 14;
	uint32_t append_disabled  :  1; // not for unexpected
	uint32_t list_vld         :  1;
};
union c_lpe_cfg_ptl_table {
	uint64_t qw[4];
	struct {
		uint64_t clr_remote_offset      :  1;
		uint64_t en_flowctrl            :  1;
		uint64_t use_long_event         :  1;
		uint64_t lossless               :  1;
		uint64_t en_event_match         :  1;
		uint64_t                        : 11;
		uint64_t eq_handle              : 11;
		uint64_t                        :  5;
		uint64_t eq_handle_no_rget      : 11;
		uint64_t                        :  5;
		uint64_t le_pool                :  4;
		uint64_t en_restricted_unicast_lm :  1;
		uint64_t use_logical            :  1;
		uint64_t is_matching            :  1;
		uint64_t do_space_check         :  1;
		uint64_t cntr_pool              :  2;
		uint64_t pe_num                 :  2;
		uint64_t en_align_lm            :  1;
		uint64_t en_match_on_vni        :  1; /* Cassini 2 only */
		uint64_t                        :  1;
		uint64_t en_sw_hw_st_chng       :  1;
		struct c_lpe_cfg_ptl_table_ptrs l[4];
		uint64_t ptl_state              :  3;
		uint64_t                        : 29;
		uint64_t drop_count             : 24;
		uint64_t signal_invalid         :  1;
		uint64_t signal_underflow       :  1;
		uint64_t signal_overflow        :  1;
		uint64_t signal_inexact         :  1;
		uint64_t                        :  4;
	};
};

#define C_LPE_DBG_MRR_OFFSET(idx)	(0x00040000+((idx)*64))
#define C_LPE_DBG_MRR_ENTRIES	2560
#define C_LPE_DBG_MRR(idx)	(C_LPE_BASE + C_LPE_DBG_MRR_OFFSET(idx))
#define C_LPE_DBG_MRR_SIZE	0x00028000

union c_lpe_dbg_mrr {
	uint64_t qw[7];
};

#define C_LPE_DBG_RGET_RAM_OFFSET(idx)	(0x00068000+((idx)*64))
#define C_LPE_DBG_RGET_RAM_ENTRIES	256
#define C_LPE_DBG_RGET_RAM(idx)	(C_LPE_BASE + C_LPE_DBG_RGET_RAM_OFFSET(idx))
#define C_LPE_DBG_RGET_RAM_SIZE	0x00004000

union c_lpe_dbg_rget_ram {
	uint64_t qw[7];
	struct {
		uint64_t vni             : 16;
		uint64_t buffer_id       : 16;
		uint64_t eq              : 11;
		uint64_t ct              : 11;
		uint64_t ac              : 10;
		uint64_t ptlte_index     : 11;
		uint64_t                 :  1;
		uint64_t sfa_nid         : 20;
		uint64_t eager_len       : 20;
		uint64_t                 :  4;
		uint64_t rendezvous_id   :  8;
		uint64_t hdr_data;
		uint64_t addr            : 57;
		uint64_t                 :  7;
		uint64_t remote_offset   : 56;
		uint64_t event_ct_reply  :  1;
		uint64_t event_ct_bytes  :  1;
		uint64_t dscp            :  6;
		uint64_t request_len     : 32;
		uint64_t                 : 23;
		uint64_t initiator_pid   :  9;
		uint64_t cmdq_id         :  9;
		uint64_t                 : 39;
		uint64_t match_bits_up   : 16;
	};
};

#define C_LPE_STS_EVENT_CNTS_OFFSET(idx)	(0x00070000+((idx)*8))
#ifndef C_LPE_STS_EVENT_CNTS_ENTRIES
#define C_LPE_STS_EVENT_CNTS_ENTRIES	512
#endif
#define C_LPE_STS_EVENT_CNTS(idx)	(C_LPE_BASE + C_LPE_STS_EVENT_CNTS_OFFSET(idx))
#define C_LPE_STS_EVENT_CNTS_SIZE	0x00001000

union c_lpe_sts_event_cnts {
	uint64_t qw;
	struct {
		uint64_t cnt  : 56;
		uint64_t      :  8;
	};
};

#define C_LPE_STS_LIST_ENTRIES_OFFSET(idx)	(0x00400000+((idx)*64))
#define C_LPE_STS_LIST_ENTRIES_ENTRIES	65536
#define C_LPE_STS_LIST_ENTRIES(idx)	(C_LPE_BASE + C_LPE_STS_LIST_ENTRIES_OFFSET(idx))
#define C_LPE_STS_LIST_ENTRIES_SIZE	0x00400000

union c_lpe_sts_list_entries {
	uint64_t qw[6];
};

#define C_MB_CFG_CRNC_OFFSET	0x00000000
#define C_MB_CFG_CRNC	(C_MB_BASE + C_MB_CFG_CRNC_OFFSET)
#define C_MB_CFG_CRNC_SIZE	0x00000008

union c_mb_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_MB_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_MB_CFG_RT_OFFSET	(C_MB_BASE + C_MB_CFG_RT_OFFSET_OFFSET)
#define C_MB_CFG_RT_OFFSET_SIZE	0x00000008

union c_mb_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_MB_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_MB_MSC_SHADOW_ACTION	(C_MB_BASE + C_MB_MSC_SHADOW_ACTION_OFFSET)
#define C_MB_MSC_SHADOW_ACTION_SIZE	0x00000008

union c_any_msc_shadow_action {
	uint64_t qw;
	struct {
		uint64_t addr_offset  : 23;
		uint64_t              :  8;
		uint64_t write        :  1;
		uint64_t              : 32;
	};
};

#define C_MB_MSC_SHADOW_OFFSET	0x00000040
#define C_MB_MSC_SHADOW	(C_MB_BASE + C_MB_MSC_SHADOW_OFFSET)
#define C_MB_MSC_SHADOW_SIZE	0x00000028

union c_mb_msc_shadow {
	uint64_t qw[5];
};

#define C_MB_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_MB_ERR_ELAPSED_TIME	(C_MB_BASE + C_MB_ERR_ELAPSED_TIME_OFFSET)
#define C_MB_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_mb_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_MB_ERR_FLG_OFFSET	0x00000108
#define C_MB_ERR_FLG	(C_MB_BASE + C_MB_ERR_FLG_OFFSET)
#define C_MB_ERR_FLG_SIZE	0x00000008

union c_mb_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                   :  1;
		uint64_t crnc_ring_sync_error      :  1;
		uint64_t crnc_ring_ecc_sbe         :  1;
		uint64_t crnc_ring_ecc_mbe         :  1;
		uint64_t crnc_ring_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_unknown      :  1;
		uint64_t crnc_csr_cmd_incomplete   :  1;
		uint64_t crnc_buf_ecc_sbe          :  1;
		uint64_t crnc_buf_ecc_mbe          :  1;
		uint64_t                           :  7;
		uint64_t crmc_ring_sync_error      :  1;
		uint64_t crmc_ring_ecc_sbe         :  1;
		uint64_t crmc_ring_ecc_mbe         :  1;
		uint64_t crmc_ring_cmd_unknown     :  1;
		uint64_t crmc_wr_error             :  1;
		uint64_t crmc_wr_addr_invalid      :  1;
		uint64_t crmc_wr_cmd_invalid       :  1;
		uint64_t crmc_wr_be_invalid        :  1;
		uint64_t crmc_wr_cor_error         :  1;
		uint64_t crmc_wr_uncor_error       :  1;
		uint64_t crmc_wr_burst_error       :  1;
		uint64_t crmc_wr_mcast_nack        :  1;
		uint64_t crmc_wr_rsvd_error        :  1;
		uint64_t crmc_wr_wrapped           :  1;
		uint64_t crmc_wr_timeout           :  1;
		uint64_t crmc_wr_axi_req_error     :  1;
		uint64_t crmc_wr_axi_ring_error    :  1;
		uint64_t crmc_wr_axi_wstrb_error   :  1;
		uint64_t crmc_wr_axi_parity_error  :  1;
		uint64_t crmc_rd_error             :  1;
		uint64_t crmc_rd_addr_invalid      :  1;
		uint64_t crmc_rd_cmd_invalid       :  1;
		uint64_t crmc_rd_be_invalid        :  1;
		uint64_t crmc_rd_cor_error         :  1;
		uint64_t crmc_rd_uncor_error       :  1;
		uint64_t crmc_rd_burst_error       :  1;
		uint64_t crmc_rd_mcast_nack        :  1;
		uint64_t crmc_rd_rsvd_error        :  1;
		uint64_t crmc_rd_wrapped           :  1;
		uint64_t crmc_rd_timeout           :  1;
		uint64_t crmc_rd_axi_req_error     :  1;
		uint64_t crmc_rd_axi_ring_error    :  1;
		uint64_t cmc_wr_error              :  4;
		uint64_t cmc_rd_error              :  4;
		uint64_t event_cnt_ovf             :  1;
		uint64_t event_ram_cor             :  1;
		uint64_t event_ram_ucor            :  1;
		uint64_t flash_cache0_cor          :  1;
		uint64_t flash_cache0_ucor         :  1;
		uint64_t flash_cache1_cor          :  1;
		uint64_t flash_cache1_ucor         :  1;
		uint64_t spi_ecc_error             :  1;
	};
};

#define C_MB_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_MB_ERR_FIRST_FLG	(C_MB_BASE + C_MB_ERR_FIRST_FLG_OFFSET)
#define C_MB_ERR_FIRST_FLG_SIZE	0x00000008

union c_any_err_first_flg {
	uint64_t qw;
	struct {
		uint64_t err_first_flg;
	};
};

#define C_MB_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_MB_ERR_FIRST_FLG_TS	(C_MB_BASE + C_MB_ERR_FIRST_FLG_TS_OFFSET)
#define C_MB_ERR_FIRST_FLG_TS_SIZE	0x00000008

union c_any_err_first_flg_ts {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_MB_ERR_CLR_OFFSET	0x00000120
#define C_MB_ERR_CLR	(C_MB_BASE + C_MB_ERR_CLR_OFFSET)
#define C_MB_ERR_CLR_SIZE	0x00000008

union c_any_err_clr {
	uint64_t qw;
	struct {
		uint64_t err_clr;
	};
};

#define C_MB_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_MB_ERR_IRQA_MSK	(C_MB_BASE + C_MB_ERR_IRQA_MSK_OFFSET)
#define C_MB_ERR_IRQA_MSK_SIZE	0x00000008

union c_any_err_irqa_msk {
	uint64_t qw;
	struct {
		uint64_t err_irq_msk;
	};
};

#define C_MB_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_MB_ERR_IRQB_MSK	(C_MB_BASE + C_MB_ERR_IRQB_MSK_OFFSET)
#define C_MB_ERR_IRQB_MSK_SIZE	0x00000008

union c_any_err_irqb_msk {
	uint64_t qw;
	struct {
		uint64_t err_irq_msk;
	};
};

#define C_MB_ERR_INFO_MSK_OFFSET	0x00000140
#define C_MB_ERR_INFO_MSK	(C_MB_BASE + C_MB_ERR_INFO_MSK_OFFSET)
#define C_MB_ERR_INFO_MSK_SIZE	0x00000008

union c_any_err_info_msk {
	uint64_t qw;
	struct {
		uint64_t err_info_msk;
	};
};

#define C_MB_ERR_INFO_MEM_OFFSET	0x00000150
#define C_MB_ERR_INFO_MEM	(C_MB_BASE + C_MB_ERR_INFO_MEM_OFFSET)
#define C_MB_ERR_INFO_MEM_SIZE	0x00000008

union c_mb_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       :  9;
		uint64_t                    :  3;
		uint64_t cor_address        : 13;
		uint64_t                    :  3;
		uint64_t cor_mem_id         :  2;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      :  9;
		uint64_t                    :  3;
		uint64_t ucor_address       : 13;
		uint64_t                    :  3;
		uint64_t ucor_mem_id        :  2;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_MB_ERR_INFO_CRMC_OFFSET	0x00000158
#define C_MB_ERR_INFO_CRMC	(C_MB_BASE + C_MB_ERR_INFO_CRMC_OFFSET)
#define C_MB_ERR_INFO_CRMC_SIZE	0x00000008

union c_mb_err_info_crmc {
	uint64_t qw;
	struct {
		uint64_t ring_id        : 10;
		uint64_t                :  2;
		uint64_t ring           :  3;
		uint64_t                :  1;
		uint64_t crnc_ring_sts  :  4;
		uint64_t crmc_ring_sts  :  1;
		uint64_t                : 43;
	};
};

#define C_MB_ERR_INFO_CMC_WR_OFFSET	0x00000160
#define C_MB_ERR_INFO_CMC_WR	(C_MB_BASE + C_MB_ERR_INFO_CMC_WR_OFFSET)
#define C_MB_ERR_INFO_CMC_WR_SIZE	0x00000018

union c_mb_err_info_cmc_wr {
	uint64_t qw[3];
	struct {
		uint64_t mst_awaddr    : 32;
		uint64_t mst_awuser    : 32;
		uint64_t mst_wdata;
		uint64_t mst_wderr     :  8;
		uint64_t mst_wstrb     :  8;
		uint64_t mst_awid      :  4;
		uint64_t mst_awregion  :  4;
		uint64_t mst_awlen     :  8;
		uint64_t mst_awsize    :  3;
		uint64_t mst_awburst   :  2;
		uint64_t mst_awmid     :  4;
		uint64_t mst_awlock    :  1;
		uint64_t mst_awqos     :  4;
		uint64_t mst_awprot    :  3;
		uint64_t mst_awcache   :  4;
		uint64_t slv_bresp     :  2;
		uint64_t slv_bid       :  4;
		uint64_t cmc_port      :  2;
		uint64_t ring          :  3;
	};
};

#define C_MB_ERR_INFO_CMC_RD_OFFSET	0x00000180
#define C_MB_ERR_INFO_CMC_RD	(C_MB_BASE + C_MB_ERR_INFO_CMC_RD_OFFSET)
#define C_MB_ERR_INFO_CMC_RD_SIZE	0x00000018

union c_mb_err_info_cmc_rd {
	uint64_t qw[3];
	struct {
		uint64_t mst_araddr    : 32;
		uint64_t mst_aruser    : 32;
		uint64_t slv_rdata;
		uint64_t slv_rderr     :  8;
		uint64_t slv_ruser     :  8;
		uint64_t mst_arid      :  4;
		uint64_t mst_arregion  :  4;
		uint64_t mst_arlen     :  8;
		uint64_t mst_arsize    :  3;
		uint64_t mst_arburst   :  2;
		uint64_t mst_armid     :  4;
		uint64_t mst_arlock    :  1;
		uint64_t mst_arqos     :  4;
		uint64_t mst_arprot    :  3;
		uint64_t mst_arcache   :  4;
		uint64_t slv_rresp     :  2;
		uint64_t slv_rid       :  4;
		uint64_t cmc_port      :  2;
		uint64_t ring          :  3;
	};
};

#define C_MB_ERR_INFO_SPI_OFFSET	0x000001a0
#define C_MB_ERR_INFO_SPI	(C_MB_BASE + C_MB_ERR_INFO_SPI_OFFSET)
#define C_MB_ERR_INFO_SPI_SIZE	0x00000008

union c_mb_err_info_spi {
	uint64_t qw;
	struct {
		uint64_t          :  3;
		uint64_t address  : 29;
		uint64_t status   :  8;
		uint64_t          : 24;
	};
};

#define C_MB_EXT_ERR_FLG_OFFSET	0x00000208
#define C_MB_EXT_ERR_FLG	(C_MB_BASE + C_MB_EXT_ERR_FLG_OFFSET)
#define C_MB_EXT_ERR_FLG_SIZE	0x00000008

union c_mb_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                      :  1;
		uint64_t jai_axi_cadr_unexpected_cmd  :  1;
		uint64_t jai_axi_cadr_unknown_cmd     :  1;
		uint64_t jai_axi_wr_decerr            :  1;
		uint64_t jai_axi_wr_slverr            :  1;
		uint64_t jai_axi_rd_decerr            :  1;
		uint64_t jai_axi_rd_slverr            :  1;
		uint64_t jai_axi_rd_parity_error      :  1;
		uint64_t flm_csr_init_p1_spi_err      :  1;
		uint64_t flm_csr_init_p2_spi_err      :  1;
		uint64_t flm_csr_init_p1_fail         :  1;
		uint64_t flm_csr_init_p2_fail         :  1;
		uint64_t flm_csr_init_p1_abort        :  1;
		uint64_t flm_csr_init_p2_abort        :  1;
		uint64_t flm_csr_init_p1_len_err      :  1;
		uint64_t flm_csr_init_p2_len_err      :  1;
		uint64_t flm_csr_init_p1_cont_err     :  1;
		uint64_t flm_csr_init_p2_cont_err     :  1;
		uint64_t flm_csr_init_p1_axi_decerr   :  1;
		uint64_t flm_csr_init_p2_axi_decerr   :  1;
		uint64_t flm_csr_init_p1_axi_slverr   :  1;
		uint64_t flm_csr_init_p2_axi_slverr   :  1;
		uint64_t flm_serdes_rom_load_spi_err  :  1;
		uint64_t flm_serdes_rom_load_error    :  1;
		uint64_t flm_serdes_rom_load_abort    :  1;
		uint64_t flm_serdes_rom_load_fail     :  1;
		uint64_t flm_hg_spi_error             :  1;
		uint64_t flm_hg_addr_error            :  1;
		uint64_t flm_sbus_wr_error            :  1;
		uint64_t flm_sbus_rd_error            :  1;
		uint64_t                              : 34;
	};
};

#define C_MB_EXT_ERR_FIRST_FLG_OFFSET	0x00000210
#define C_MB_EXT_ERR_FIRST_FLG	(C_MB_BASE + C_MB_EXT_ERR_FIRST_FLG_OFFSET)
#define C_MB_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_MB_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000218
#define C_MB_EXT_ERR_FIRST_FLG_TS	(C_MB_BASE + C_MB_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_MB_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_MB_EXT_ERR_CLR_OFFSET	0x00000220
#define C_MB_EXT_ERR_CLR	(C_MB_BASE + C_MB_EXT_ERR_CLR_OFFSET)
#define C_MB_EXT_ERR_CLR_SIZE	0x00000008
#define C_MB_EXT_ERR_IRQA_MSK_OFFSET	0x00000228
#define C_MB_EXT_ERR_IRQA_MSK	(C_MB_BASE + C_MB_EXT_ERR_IRQA_MSK_OFFSET)
#define C_MB_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_MB_EXT_ERR_IRQB_MSK_OFFSET	0x00000230
#define C_MB_EXT_ERR_IRQB_MSK	(C_MB_BASE + C_MB_EXT_ERR_IRQB_MSK_OFFSET)
#define C_MB_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_MB_EXT_ERR_INFO_MSK_OFFSET	0x00000240
#define C_MB_EXT_ERR_INFO_MSK	(C_MB_BASE + C_MB_EXT_ERR_INFO_MSK_OFFSET)
#define C_MB_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_MB_CFG_MISC_OFFSET	0x00004000
#define C_MB_CFG_MISC	(C_MB_BASE + C_MB_CFG_MISC_OFFSET)
#define C_MB_CFG_MISC_SIZE	0x00000008

union c_mb_cfg_misc {
	uint64_t qw;
	struct {
		uint64_t nic_warm_rst0                     :  1;
		uint64_t nic_warm_rst1                     :  1;
		uint64_t nic_warm_rst2                     :  1;
		uint64_t nic_warm_rst3                     :  1;
		uint64_t nic_warm_rst4                     :  1;
		uint64_t nic_warm_rst5                     :  1;
		uint64_t nic_warm_rst6                     :  1;
		uint64_t nic_warm_rst7                     :  1;
		uint64_t nic_warm_rst8                     :  1;
		uint64_t nic_warm_rst9                     :  1;
		uint64_t                                   :  6;
		uint64_t nic_warm_rst_mask                 : 10;
		uint64_t mb_warm_rst_mask                  :  1;
		uint64_t pi_warm_rst_mask                  :  1;
		uint64_t                                   :  3;
		uint64_t halt_clock_en                     :  1;
		uint64_t cold_rst_enable                   : 16;
		uint64_t cold_rst                          :  1;
		uint64_t mb_cold_rst_enable                :  1;
		uint64_t cold_rst_on_perst                 :  1;
		uint64_t cold_rst_on_hot_rst               :  1;
		uint64_t skip_rom_load_on_forced_cold_rst  :  1;
		uint64_t ignore_link_req_rst               :  1;
		uint64_t ignore_gpout_all_done             :  1;
		uint64_t ignore_gpout_efuse_done           :  1;
		uint64_t ignore_gpout_temp_pushed          :  1;
		uint64_t ignore_ancil_osc_clk_rdy          :  1;
		uint64_t                                   :  6;
	};
};

#define C_MB_STS_REV_OFFSET	0x00004008
#define C_MB_STS_REV	(C_MB_BASE + C_MB_STS_REV_OFFSET)
#define C_MB_STS_REV_SIZE	0x00000008

union c_mb_sts_rev {
	uint64_t qw;
	struct {
		uint64_t rev        : 16;
		uint64_t device_id  : 16;
		uint64_t vendor_id  : 16;
		uint64_t proto      : 12;
		uint64_t platform   :  4;
	};
};

#define C_MB_CFG_EVENT_CNTS_OFFSET	0x00004010
#define C_MB_CFG_EVENT_CNTS	(C_MB_BASE + C_MB_CFG_EVENT_CNTS_OFFSET)
#define C_MB_CFG_EVENT_CNTS_SIZE	0x00000008

union c_mb_cfg_event_cnts {
	uint64_t qw;
	struct {
		uint64_t enable            :  1;
		uint64_t init              :  1;
		uint64_t init_done         :  1;
		uint64_t warm_rst_disable  :  1;
		uint64_t ovf_cnt           : 16;
		uint64_t                   : 44;
	};
};

#define C_MB_DBG_SCRATCH_OFFSET(idx)	(0x00004020+((idx)*8))
#define C_MB_DBG_SCRATCH_ENTRIES	8
#define C_MB_DBG_SCRATCH(idx)	(C_MB_BASE + C_MB_DBG_SCRATCH_OFFSET(idx))
#define C_MB_DBG_SCRATCH_SIZE	0x00000040

union c_mb_dbg_scratch {
	uint64_t qw;
	struct {
		uint64_t scratch;
	};
};

#define C_MB_CFG_CRMC_OFFSET(idx)	(0x0000d000+((idx)*8))
#define C_MB_CFG_CRMC_ENTRIES	3
#define C_MB_CFG_CRMC(idx)	(C_MB_BASE + C_MB_CFG_CRMC_OFFSET(idx))
#define C_MB_CFG_CRMC_SIZE	0x00000018

union c_mb_cfg_crmc {
	uint64_t qw;
	struct {
		uint64_t ring_timeout                    :  8;
		uint64_t rready_timeout_disable          :  1;
		uint64_t bready_timeout_disable          :  1;
		uint64_t rd_addr_invalid_decerr_disable  :  1;
		uint64_t                                 : 53;
	};
};

#define C_MB_DBG_CFG_SBUS_MASTER_OFFSET	0x0000e000
#define C_MB_DBG_CFG_SBUS_MASTER	(C_MB_BASE + C_MB_DBG_CFG_SBUS_MASTER_OFFSET)
#define C_MB_DBG_CFG_SBUS_MASTER_SIZE	0x00000008

union c_mb_dbg_cfg_sbus_master {
	uint64_t qw;
	struct {
		uint64_t execute              :  1;
		uint64_t rcv_data_valid_sel   :  1;
		uint64_t mode                 :  1;
		uint64_t stop_on_overrun      :  1;
		uint64_t stop_on_write_error  :  1;
		uint64_t                      :  3;
		uint64_t command              :  8;
		uint64_t receiver_address     :  8;
		uint64_t data_address         :  8;
		uint64_t data                 : 32;
	};
};

#define C_MB_DBG_STS_SBUS_MASTER_OFFSET	0x0000e010
#define C_MB_DBG_STS_SBUS_MASTER	(C_MB_BASE + C_MB_DBG_STS_SBUS_MASTER_OFFSET)
#define C_MB_DBG_STS_SBUS_MASTER_SIZE	0x00000008

union c_mb_dbg_sts_sbus_master {
	uint64_t qw;
	struct {
		uint64_t done              :  1;
		uint64_t rcv_data_valid    :  1;
		uint64_t overrun           :  1;
		uint64_t write_error       :  1;
		uint64_t result_code       :  3;
		uint64_t                   :  1;
		uint64_t command           :  8;
		uint64_t receiver_address  :  8;
		uint64_t data_address      :  8;
		uint64_t data              : 32;
	};
};

#define C_MB_CFG_RTC_OFFSET	0x0000f000
#define C_MB_CFG_RTC	(C_MB_BASE + C_MB_CFG_RTC_OFFSET)
#define C_MB_CFG_RTC_SIZE	0x00000008

union c_mb_cfg_rtc {
	uint64_t qw;
	struct {
		uint64_t rtc_enable               :  1;
		uint64_t rtc_load                 :  1;
		uint64_t rtc_update               :  1;
		uint64_t                          :  1;
		uint64_t rtc_etc_sec_change_mask  :  3;
		uint64_t                          :  9;
		uint64_t etc_enable               :  1;
		uint64_t etc_load                 :  1;
		uint64_t                          :  2;
		uint64_t etc_ns_lock_resolution   :  3;
		uint64_t                          : 41;
	};
};

#define C_MB_STS_RTC_OFFSET	0x0000f008
#define C_MB_STS_RTC	(C_MB_BASE + C_MB_STS_RTC_OFFSET)
#define C_MB_STS_RTC_SIZE	0x00000008

union c_mb_sts_rtc {
	uint64_t qw;
	struct {
		uint64_t rtc_minus_etc_sec         : 48;
		uint64_t rtc_lt_etc_sec            :  1;
		uint64_t rtc_minus_etc_sec_change  :  1;
		uint64_t etc_ns_locked             :  1;
		uint64_t etc_ns_lost_lock          :  1;
		uint64_t                           : 12;
	};
};

#define C_MB_CFG_RTC_TIME_OFFSET	0x0000f010
#define C_MB_CFG_RTC_TIME	(C_MB_BASE + C_MB_CFG_RTC_TIME_OFFSET)
#define C_MB_CFG_RTC_TIME_SIZE	0x00000010

union c_mb_cfg_rtc_time {
	uint64_t qw[2];
	struct {
		uint64_t nanoseconds    : 30;
		uint64_t seconds_33_0   : 34;
		uint64_t seconds_47_34  : 14;
		uint64_t                : 50;
	};
};

#define C_MB_CFG_RTC_INC_OFFSET	0x0000f020
#define C_MB_CFG_RTC_INC	(C_MB_BASE + C_MB_CFG_RTC_INC_OFFSET)
#define C_MB_CFG_RTC_INC_SIZE	0x00000008

union c_mb_cfg_rtc_inc {
	uint64_t qw;
	struct {
		uint64_t fractional_ns  : 32;
		uint64_t nanoseconds    : 10;
		uint64_t                : 22;
	};
};

#define C_MB_STS_RTC_CURRENT_TIME_OFFSET	0x0000f030
#define C_MB_STS_RTC_CURRENT_TIME	(C_MB_BASE + C_MB_STS_RTC_CURRENT_TIME_OFFSET)
#define C_MB_STS_RTC_CURRENT_TIME_SIZE	0x00000010

union c_mb_sts_rtc_current_time {
	uint64_t qw[2];
	struct {
		uint64_t nanoseconds    : 30;
		uint64_t seconds_33_0   : 34;
		uint64_t seconds_47_34  : 14;
		uint64_t                : 50;
	};
};

#define C_MB_STS_RTC_LAST_TIME_OFFSET	0x0000f040
#define C_MB_STS_RTC_LAST_TIME	(C_MB_BASE + C_MB_STS_RTC_LAST_TIME_OFFSET)
#define C_MB_STS_RTC_LAST_TIME_SIZE	0x00000010

union c_mb_sts_rtc_last_time {
	uint64_t qw[2];
	struct {
		uint64_t nanoseconds    : 30;
		uint64_t seconds_33_0   : 34;
		uint64_t seconds_47_34  : 14;
		uint64_t                : 50;
	};
};

#define C_MB_STS_RTC_ETC_CUR_TIME_OFFSET	0x0000f050
#define C_MB_STS_RTC_ETC_CUR_TIME	(C_MB_BASE + C_MB_STS_RTC_ETC_CUR_TIME_OFFSET)
#define C_MB_STS_RTC_ETC_CUR_TIME_SIZE	0x00000008

union c_mb_sts_rtc_etc_cur_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_MB_STS_RTC_ETC_LAST_TIME_OFFSET	0x0000f058
#define C_MB_STS_RTC_ETC_LAST_TIME	(C_MB_BASE + C_MB_STS_RTC_ETC_LAST_TIME_OFFSET)
#define C_MB_STS_RTC_ETC_LAST_TIME_SIZE	0x00000008

union c_mb_sts_rtc_etc_last_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_MB_STS_EVENT_CNTS_OFFSET(idx)	(0x00020000+((idx)*8))
#ifndef C_MB_STS_EVENT_CNTS_ENTRIES
#define C_MB_STS_EVENT_CNTS_ENTRIES	256
#endif
#define C_MB_STS_EVENT_CNTS(idx)	(C_MB_BASE + C_MB_STS_EVENT_CNTS_OFFSET(idx))
#define C_MB_STS_EVENT_CNTS_SIZE	0x00000800

union c_mb_sts_event_cnts {
	uint64_t qw;
	struct {
		uint64_t cnt  : 56;
		uint64_t      :  8;
	};
};

#define C_MST_CFG_CRNC_OFFSET	0x00000000
#define C_MST_CFG_CRNC	(C_MST_BASE + C_MST_CFG_CRNC_OFFSET)
#define C_MST_CFG_CRNC_SIZE	0x00000008

union c_mst_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_MST_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_MST_CFG_RT_OFFSET	(C_MST_BASE + C_MST_CFG_RT_OFFSET_OFFSET)
#define C_MST_CFG_RT_OFFSET_SIZE	0x00000008

union c_mst_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_MST_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_MST_MSC_SHADOW_ACTION	(C_MST_BASE + C_MST_MSC_SHADOW_ACTION_OFFSET)
#define C_MST_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_MST_MSC_SHADOW_OFFSET	0x00000040
#define C_MST_MSC_SHADOW	(C_MST_BASE + C_MST_MSC_SHADOW_OFFSET)
#define C_MST_MSC_SHADOW_SIZE	0x00000040

union c_mst_msc_shadow {
	uint64_t qw[8];
};

#define C_MST_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_MST_ERR_ELAPSED_TIME	(C_MST_BASE + C_MST_ERR_ELAPSED_TIME_OFFSET)
#define C_MST_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_mst_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_MST_ERR_FLG_OFFSET	0x00000108
#define C_MST_ERR_FLG	(C_MST_BASE + C_MST_ERR_FLG_OFFSET)
#define C_MST_ERR_FLG_SIZE	0x00000008

union c_mst_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          : 23;
		uint64_t lnk_list_ptrs_cor        :  1;
		uint64_t lnk_list_ptrs_ucor       :  1;
		uint64_t                          :  2;
		uint64_t req_fifo_unrun           :  1;
		uint64_t req_fifo_ovrun           :  1;
		uint64_t req_fifo_crdt_unflw      :  1;
		uint64_t                          :  5;
		uint64_t match_crdt_unflw_get     :  1;
		uint64_t match_crdt_unflw_put     :  1;
		uint64_t evnt_fifo_cor            :  1;
		uint64_t evnt_fifo_ucor           :  1;
		uint64_t evnt_fifo_unrun          :  1;
		uint64_t evnt_fifo_ovrun          :  1;
		uint64_t ee_crdt_unflw            :  1;
		uint64_t req_fifo_cor             :  1;
		uint64_t req_fifo_ucor            :  1;
		uint64_t mst_table_cor            :  1;
		uint64_t mst_table_ucor           :  1;
		uint64_t ll_data_ram_cor          :  1;
		uint64_t ll_data_ram_ucor         :  1;
		uint64_t                          :  7;
	};
};

#define C_MST_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_MST_ERR_FIRST_FLG	(C_MST_BASE + C_MST_ERR_FIRST_FLG_OFFSET)
#define C_MST_ERR_FIRST_FLG_SIZE	0x00000008
#define C_MST_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_MST_ERR_FIRST_FLG_TS	(C_MST_BASE + C_MST_ERR_FIRST_FLG_TS_OFFSET)
#define C_MST_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_MST_ERR_CLR_OFFSET	0x00000120
#define C_MST_ERR_CLR	(C_MST_BASE + C_MST_ERR_CLR_OFFSET)
#define C_MST_ERR_CLR_SIZE	0x00000008
#define C_MST_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_MST_ERR_IRQA_MSK	(C_MST_BASE + C_MST_ERR_IRQA_MSK_OFFSET)
#define C_MST_ERR_IRQA_MSK_SIZE	0x00000008
#define C_MST_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_MST_ERR_IRQB_MSK	(C_MST_BASE + C_MST_ERR_IRQB_MSK_OFFSET)
#define C_MST_ERR_IRQB_MSK_SIZE	0x00000008
#define C_MST_ERR_INFO_MSK_OFFSET	0x00000140
#define C_MST_ERR_INFO_MSK	(C_MST_BASE + C_MST_ERR_INFO_MSK_OFFSET)
#define C_MST_ERR_INFO_MSK_SIZE	0x00000008
#define C_MST_ERR_INFO_MEM_OFFSET	0x00000180
#define C_MST_ERR_INFO_MEM	(C_MST_BASE + C_MST_ERR_INFO_MEM_OFFSET)
#define C_MST_ERR_INFO_MEM_SIZE	0x00000008

union c_mst_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 12;
		uint64_t cor_mem_id         :  6;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 12;
		uint64_t ucor_mem_id        :  6;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_MST_CFG_RC_UPDATE_OFFSET	0x00000400
#define C_MST_CFG_RC_UPDATE	(C_MST_BASE + C_MST_CFG_RC_UPDATE_OFFSET)
#define C_MST_CFG_RC_UPDATE_SIZE	0x00000008

union c_mst_cfg_rc_update {
	uint64_t qw;
	struct {
		uint64_t return_code  :  6;
		uint64_t              :  2;
		uint64_t mst_idx      : 11;
		uint64_t              :  5;
		uint64_t buffer_idx   : 16;
		uint64_t portal_idx   : 11;
		uint64_t              :  1;
		uint64_t ptl_list     :  2;
		uint64_t              :  2;
		uint64_t ptl_vld      :  1;
		uint64_t portal_vld   :  1;
		uint64_t buffer_vld   :  1;
		uint64_t              :  5;
	};
};

#define C_MST_STS_RC_STS_OFFSET	0x00010000
#define C_MST_STS_RC_STS	(C_MST_BASE + C_MST_STS_RC_STS_OFFSET)
#define C_MST_STS_RC_STS_SIZE	0x00000008

union c_mst_sts_rc_sts {
	uint64_t qw;
	struct {
		uint64_t done   :  1;
		uint64_t match  :  1;
		uint64_t        : 62;
	};
};

#define C_MST_STS_NUM_REQ_OFFSET	0x00010018
#define C_MST_STS_NUM_REQ	(C_MST_BASE + C_MST_STS_NUM_REQ_OFFSET)
#define C_MST_STS_NUM_REQ_SIZE	0x00000008

union c_mst_sts_num_req {
	uint64_t qw;
	struct {
		uint64_t cnt  : 12;
		uint64_t      : 52;
	};
};

#define C_MST_DBG_MST_TABLE_OFFSET(idx)	(0x00020000+((idx)*64))
#define C_MST_DBG_MST_TABLE_ENTRIES	2048
#define C_MST_DBG_MST_TABLE(idx)	(C_MST_BASE + C_MST_DBG_MST_TABLE_OFFSET(idx))
#define C_MST_DBG_MST_TABLE_SIZE	0x00020000

union c_mst_dbg_mst_table {
	uint64_t qw[8];
	struct {
		uint64_t cntr_pool_id           :  2;
		uint64_t                        :  2;
		uint64_t rendezvous             :  1;
		uint64_t                        :  3;
		uint64_t rendezvous_id          :  8;
		uint64_t vni                    : 16;
		uint64_t lpe_stat_1             : 16;
		uint64_t lpe_stat_2             : 16;
		uint64_t auto_unlinked          :  1;
		uint64_t                        :  7;
		uint64_t dma_hdr_len            :  6;
		uint64_t                        :  2;
		uint64_t event_ct_bytes         :  1;
		uint64_t event_ct_comm          :  1;
		uint64_t event_success_disable  :  1;
		uint64_t event_comm_disable     :  1;
		uint64_t use_once               :  1;
		uint64_t event_use_long_fmt     :  1;
		uint64_t event_start_at_base    :  1;
		uint64_t unrestricted_end_ro    :  1;
		uint64_t unrestricted_body_ro   :  1;
		uint64_t acid                   : 10;
		uint64_t                        :  5;
		uint64_t ct_handle              : 11;
		uint64_t                        :  1;
		uint64_t eq_handle              : 11;
		uint64_t                        :  1;
		uint64_t portal_index           : 11;
		uint64_t                        :  1;
		uint64_t ptl_list               :  2;
		uint64_t                        :  2;
		uint64_t buffer_id              : 16;
		uint64_t initiator              : 32;
		uint64_t hdr_data;
		uint64_t match_bits;
		uint64_t mlength                : 32;
		uint64_t rlength                : 32;
		uint64_t remote_offset          : 56;
		uint64_t return_code            :  6;
		uint64_t                        :  2;
		uint64_t start_addr             : 57;
		uint64_t                        :  7;
	};
};

#define C_MST_DBG_MATCH_DONE_OFFSET	0x00064100
#define C_MST_DBG_MATCH_DONE	(C_MST_BASE + C_MST_DBG_MATCH_DONE_OFFSET)
#define C_MST_DBG_MATCH_DONE_SIZE	0x00000008

union c_mst_dbg_match_done {
	uint64_t qw;
	struct {
		uint64_t mst_idx     : 11;
		uint64_t             :  5;
		uint64_t wr_en       :  1;
		uint64_t             :  3;
		uint64_t mst_idx_st  :  1;
		uint64_t             : 43;
	};
};

#define C_OXE_CFG_CRNC_OFFSET	0x00000000
#define C_OXE_CFG_CRNC	(C_OXE_BASE + C_OXE_CFG_CRNC_OFFSET)
#define C_OXE_CFG_CRNC_SIZE	0x00000008

union c_oxe_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_OXE_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_OXE_CFG_RT_OFFSET	(C_OXE_BASE + C_OXE_CFG_RT_OFFSET_OFFSET)
#define C_OXE_CFG_RT_OFFSET_SIZE	0x00000008

union c_oxe_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_OXE_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_OXE_MSC_SHADOW_ACTION	(C_OXE_BASE + C_OXE_MSC_SHADOW_ACTION_OFFSET)
#define C_OXE_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_OXE_MSC_SHADOW_OFFSET	0x00000040
#define C_OXE_MSC_SHADOW	(C_OXE_BASE + C_OXE_MSC_SHADOW_OFFSET)
#define C_OXE_MSC_SHADOW_SIZE	0x00000040

union c_oxe_msc_shadow {
	uint64_t qw[8];
};

#define C_OXE_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_OXE_ERR_ELAPSED_TIME	(C_OXE_BASE + C_OXE_ERR_ELAPSED_TIME_OFFSET)
#define C_OXE_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_oxe_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_OXE_ERR_FLG_OFFSET	0x00000108
#define C_OXE_ERR_FLG	(C_OXE_BASE + C_OXE_ERR_FLG_OFFSET)
#define C_OXE_ERR_FLG_SIZE	0x00000008

union c_oxe_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t                          :  5;
		uint64_t spt_cdt_err              :  1;
		uint64_t smt_cdt_err              :  1;
		uint64_t cells_mem_cor            :  1;
		uint64_t pkl_mem_cor              :  1;
		uint64_t pct_sb_mcu_mem_cor       :  1;
		uint64_t pct_sb_sct_mem_cor       :  1;
		uint64_t dmu_mem_cor              :  1;
		uint64_t rec_mem_cor              :  1;
		uint64_t atreq_mem_cor            :  1;
		uint64_t atrsp_mem_cor            :  1;
		uint64_t tso_flit_cor             :  1;
		uint64_t sb_fifo_cor              :  1;
		uint64_t rarb_hdr_cor             :  1;
		uint64_t rarb_data_cor            :  1;
		uint64_t at_resp_bus_cor          :  1;
		uint64_t ixe_cmd_cor              :  1;
		uint64_t lpe_cmd_cor              :  1;
		uint64_t tou_cmd_cor              :  1;
		uint64_t cq_cmd_cor               :  1;
		uint64_t cells_mem_ucor           :  1;
		uint64_t pkl_mem_ucor             :  1;
		uint64_t pct_sb_mcu_mem_ucor      :  1;
		uint64_t pct_sb_sct_mem_ucor      :  1;
		uint64_t dmu_mem_ucor             :  1;
		uint64_t rec_mem_ucor             :  1;
		uint64_t atreq_mem_ucor           :  1;
		uint64_t atrsp_mem_ucor           :  1;
		uint64_t tso_flit_ucor            :  1;
		uint64_t sb_fifo_ucor             :  1;
		uint64_t rarb_hdr_ucor            :  1;
		uint64_t rarb_data_ucor           :  1;
		uint64_t at_resp_bus_ucor         :  1;
		uint64_t ixe_cmd_ucor             :  1;
		uint64_t lpe_cmd_ucor             :  1;
		uint64_t tou_cmd_ucor             :  1;
		uint64_t cq_cmd_ucor              :  1;
		uint64_t fifo_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t spt_ignore_err           :  1;
		uint64_t smt_ignore_err           :  1;
		uint64_t sct_ignore_err           :  1;
		uint64_t srb_ignore_err           :  1;
		uint64_t cmd_unexpected           :  1;
		uint64_t rarb_hw_err              :  1;
		uint64_t pcie_unsuccess_cmpl      :  1;
		uint64_t pcie_error_poisoned      :  1;
		uint64_t at_overrun_err           :  1;
		uint64_t mdr_pipe_ucor            :  1;
		uint64_t cell_xlate_parity_err    :  1;
		uint64_t                          :  1;
	};
};

#define C_OXE_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_OXE_ERR_FIRST_FLG	(C_OXE_BASE + C_OXE_ERR_FIRST_FLG_OFFSET)
#define C_OXE_ERR_FIRST_FLG_SIZE	0x00000008
#define C_OXE_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_OXE_ERR_FIRST_FLG_TS	(C_OXE_BASE + C_OXE_ERR_FIRST_FLG_TS_OFFSET)
#define C_OXE_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_OXE_ERR_CLR_OFFSET	0x00000120
#define C_OXE_ERR_CLR	(C_OXE_BASE + C_OXE_ERR_CLR_OFFSET)
#define C_OXE_ERR_CLR_SIZE	0x00000008
#define C_OXE_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_OXE_ERR_IRQA_MSK	(C_OXE_BASE + C_OXE_ERR_IRQA_MSK_OFFSET)
#define C_OXE_ERR_IRQA_MSK_SIZE	0x00000008
#define C_OXE_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_OXE_ERR_IRQB_MSK	(C_OXE_BASE + C_OXE_ERR_IRQB_MSK_OFFSET)
#define C_OXE_ERR_IRQB_MSK_SIZE	0x00000008
#define C_OXE_ERR_INFO_MSK_OFFSET	0x00000140
#define C_OXE_ERR_INFO_MSK	(C_OXE_BASE + C_OXE_ERR_INFO_MSK_OFFSET)
#define C_OXE_ERR_INFO_MSK_SIZE	0x00000008
#define C_OXE_ERR_INFO_MEM_OFFSET	0x00000180
#define C_OXE_ERR_INFO_MEM	(C_OXE_BASE + C_OXE_ERR_INFO_MEM_OFFSET)
#define C_OXE_ERR_INFO_MEM_SIZE	0x00000008

union c_oxe_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 12;
		uint64_t cor_mem_id         :  5;
		uint64_t                    :  2;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 12;
		uint64_t ucor_mem_id        :  5;
		uint64_t                    :  2;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_OXE_ERR_INFO_FIFO_ERR_OFFSET	0x00000188
#define C_OXE_ERR_INFO_FIFO_ERR	(C_OXE_BASE + C_OXE_ERR_INFO_FIFO_ERR_OFFSET)
#define C_OXE_ERR_INFO_FIFO_ERR_SIZE	0x00000008

union c_oxe_err_info_fifo_err {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           :  6;
		uint64_t fifo_id   :  4;
		uint64_t           : 52;
	};
};

#define C_OXE_ERR_INFO_CC_OFFSET	0x00000190
#define C_OXE_ERR_INFO_CC	(C_OXE_BASE + C_OXE_ERR_INFO_CC_OFFSET)
#define C_OXE_ERR_INFO_CC_SIZE	0x00000008

union c_oxe_err_info_cc {
	uint64_t qw;
	struct {
		uint64_t cc_id   :  5;
		uint64_t         :  3;
		uint64_t sub_id  :  4;
		uint64_t         : 52;
	};
};

#define C_OXE_ERR_INFO_PCIE_UC_OFFSET	0x00000198
#define C_OXE_ERR_INFO_PCIE_UC	(C_OXE_BASE + C_OXE_ERR_INFO_PCIE_UC_OFFSET)
#define C_OXE_ERR_INFO_PCIE_UC_SIZE	0x00000008

union c_oxe_err_info_pcie_uc {
	uint64_t qw;
	struct {
		uint64_t pcie_tag     :  9;
		uint64_t              :  3;
		uint64_t cmpl_status  :  3;
		uint64_t              : 49;
	};
};

#define C_OXE_ERR_INFO_PCIE_EP_OFFSET	0x000001a0
#define C_OXE_ERR_INFO_PCIE_EP	(C_OXE_BASE + C_OXE_ERR_INFO_PCIE_EP_OFFSET)
#define C_OXE_ERR_INFO_PCIE_EP_SIZE	0x00000008

union c_oxe_err_info_pcie_ep {
	uint64_t qw;
	struct {
		uint64_t pcie_tag  :  9;
		uint64_t           : 55;
	};
};

#define C_OXE_ERR_INFO_RARB_OFFSET	0x000001a8
#define C_OXE_ERR_INFO_RARB	(C_OXE_BASE + C_OXE_ERR_INFO_RARB_OFFSET)
#define C_OXE_ERR_INFO_RARB_SIZE	0x00000008

union c_oxe_err_info_rarb {
	uint64_t qw;
	struct {
		uint64_t hdr_ucor               :  1;
		uint64_t data_ucor              :  1;
		uint64_t bad_vld                :  1;
		uint64_t bad_size               :  1;
		uint64_t bad_cmd                :  1;
		uint64_t unexpected_cmpl_tag    :  1;
		uint64_t cmpl_tag_out_of_range  :  1;
		uint64_t pcie_tag               :  9;
		uint64_t                        : 48;
	};
};

#define C_OXE_ERR_INFO_UNEXPCTD_OFFSET	0x000001b0
#define C_OXE_ERR_INFO_UNEXPCTD	(C_OXE_BASE + C_OXE_ERR_INFO_UNEXPCTD_OFFSET)
#define C_OXE_ERR_INFO_UNEXPCTD_SIZE	0x00000008

union c_oxe_err_info_unexpctd {
	uint64_t qw;
	struct {
		uint64_t mcu_unexpc_cmd  :  7;
		uint64_t                 :  1;
		uint64_t ixe_unexp_cmd   :  1;
		uint64_t lpe_unexp_cmd   :  1;
		uint64_t tou_unexp_cmd   :  1;
		uint64_t cq_unexp_cmd    :  1;
		uint64_t                 : 52;
	};
};

#define C_OXE_ERR_INFO_CELL_XLATE_OFFSET	0x000001b8
#define C_OXE_ERR_INFO_CELL_XLATE	(C_OXE_BASE + C_OXE_ERR_INFO_CELL_XLATE_OFFSET)
#define C_OXE_ERR_INFO_CELL_XLATE_SIZE	0x00000008

union c_oxe_err_info_cell_xlate {
	uint64_t qw;
	struct {
		uint64_t free_list_perr   :  1;
		uint64_t vptr_table_perr  :  1;
		uint64_t tt1_perr         :  1;
		uint64_t tt2_perr         :  1;
		uint64_t tt3_perr         :  1;
		uint64_t                  : 59;
	};
};

#define C_OXE_ERR_INFO_IGNORE_ERR_OFFSET	0x000001e0
#define C_OXE_ERR_INFO_IGNORE_ERR	(C_OXE_BASE + C_OXE_ERR_INFO_IGNORE_ERR_OFFSET)
#define C_OXE_ERR_INFO_IGNORE_ERR_SIZE	0x00000008

union c_oxe_err_info_ignore_err {
	uint64_t qw;
	struct {
		uint64_t spt_err  :  1;
		uint64_t smt_err  :  1;
		uint64_t sct_err  :  1;
		uint64_t srb_err  :  1;
		uint64_t bc       :  4;
		uint64_t mcu_num  :  7;
		uint64_t          : 49;
	};
};

#define C_OXE_CFG_COMMON_OFFSET	0x00000400
#define C_OXE_CFG_COMMON	(C_OXE_BASE + C_OXE_CFG_COMMON_OFFSET)
#define C_OXE_CFG_COMMON_SIZE	0x00000008

union c_oxe_cfg_common {
	uint64_t qw;
	struct {
		uint64_t occ_bin0_sz    : 10;
		uint64_t occ_bin1_sz    : 10;
		uint64_t occ_bin2_sz    : 10;
		uint64_t occ_bin3_sz    : 10;
		uint64_t ioi_enable     :  1;
		uint64_t mcu_meas_unit  :  2;
		uint64_t                : 21;
	};
};

#define C_OXE_CFG_FAB_PARAM_OFFSET	0x00000410
#define C_OXE_CFG_FAB_PARAM	(C_OXE_BASE + C_OXE_CFG_FAB_PARAM_OFFSET)
#define C_OXE_CFG_FAB_PARAM_SIZE	0x00000008

union c_oxe_cfg_fab_param {
	uint64_t qw;
	struct {
		uint64_t ver              :  4;
		uint64_t req_32b_enabled  :  1;
		uint64_t rsp_32b_enabled  :  1;
		uint64_t req_40b_enabled  :  1;
		uint64_t rsp_40b_enabled  :  1;
		uint64_t ttl              :  8;
		uint64_t protocol         :  8;
		uint64_t ecn_enable       :  8;
		uint64_t min_req_size     :  6;
		uint64_t                  :  2;
		uint64_t min_rsp_size     :  6;
		uint64_t                  : 18;
	};
};

#define C_OXE_CFG_FAB_SFA_OFFSET	0x00000418
#define C_OXE_CFG_FAB_SFA	(C_OXE_BASE + C_OXE_CFG_FAB_SFA_OFFSET)
#define C_OXE_CFG_FAB_SFA_SIZE	0x00000008

union c_oxe_cfg_fab_sfa {
	uint64_t qw;
	struct {
		uint64_t nid  : 20;
		uint64_t      : 44;
	};
};

#define C_OXE_CFG_DUMMY_ADDR_OFFSET	0x00000420
#define C_OXE_CFG_DUMMY_ADDR	(C_OXE_BASE + C_OXE_CFG_DUMMY_ADDR_OFFSET)
#define C_OXE_CFG_DUMMY_ADDR_SIZE	0x00000008

union c_oxe_cfg_dummy_addr {
	uint64_t qw;
	struct {
		uint64_t ac       : 10;
		uint64_t          :  2;
		uint64_t subaddr  : 45;
		uint64_t          :  7;
	};
};

#define C_OXE_CFG_PAUSE_QUANTA_OFFSET	0x00000438
#define C_OXE_CFG_PAUSE_QUANTA	(C_OXE_BASE + C_OXE_CFG_PAUSE_QUANTA_OFFSET)
#define C_OXE_CFG_PAUSE_QUANTA_SIZE	0x00000008

union c_oxe_cfg_pause_quanta {
	uint64_t qw;
	struct {
		uint64_t sub_value   : 24;
		uint64_t sub_period  :  8;
		uint64_t             : 32;
	};
};

#define C_OXE_CFG_FGFC_OFFSET	0x00000440
#define C_OXE_CFG_FGFC	(C_OXE_BASE + C_OXE_CFG_FGFC_OFFSET)
#define C_OXE_CFG_FGFC_SIZE	0x00000008

union c_oxe_cfg_fgfc {
	uint64_t qw;
	struct {
		uint64_t extra_bytes  :  4;
		uint64_t round_pos    :  4;
		uint64_t enable       :  1;
		uint64_t              :  7;
		uint64_t cdt_adj      : 14;
		uint64_t              :  2;
		uint64_t cdt_limit    : 24;
		uint64_t              :  8;
	};
};

#define C_OXE_CFG_PCT_CDT_OFFSET	0x00000448
#define C_OXE_CFG_PCT_CDT	(C_OXE_BASE + C_OXE_CFG_PCT_CDT_OFFSET)
#define C_OXE_CFG_PCT_CDT_SIZE	0x00000008

union c_oxe_cfg_pct_cdt {
	uint64_t qw;
	struct {
		uint64_t spt_cdt_limit  : 11;
		uint64_t                :  5;
		uint64_t smt_cdt_limit  :  9;
		uint64_t                :  7;
		uint64_t sct_cdt_limit  : 13;
		uint64_t                :  3;
		uint64_t srb_cdt_limit  : 12;
		uint64_t                :  4;
	};
};

#define C_OXE_CFG_BUF_BC_PARAM_OFFSET(idx)	(0x00000500+((idx)*8))
#define C_OXE_CFG_BUF_BC_PARAM_ENTRIES	10
#define C_OXE_CFG_BUF_BC_PARAM(idx)	(C_OXE_BASE + C_OXE_CFG_BUF_BC_PARAM_OFFSET(idx))
#define C_OXE_CFG_BUF_BC_PARAM_SIZE	0x00000050

union c_oxe_cfg_buf_bc_param {
	uint64_t qw;
	struct {
		uint64_t bc_reserv_buf_cdt  : 11;
		uint64_t                    :  5;
		uint64_t bc_max_mtu         :  8;
		uint64_t                    : 40;
	};
};

#define C_OXE_CFG_PCT_BC_CDT_OFFSET(idx)	(0x00000600+((idx)*8))
#define C_OXE_CFG_PCT_BC_CDT_ENTRIES	10
#define C_OXE_CFG_PCT_BC_CDT(idx)	(C_OXE_BASE + C_OXE_CFG_PCT_BC_CDT_OFFSET(idx))
#define C_OXE_CFG_PCT_BC_CDT_SIZE	0x00000050

union c_oxe_cfg_pct_bc_cdt {
	uint64_t qw;
	struct {
		uint64_t spt_cdt_rsvd  : 11;
		uint64_t               :  5;
		uint64_t smt_cdt_rsvd  :  9;
		uint64_t               :  7;
		uint64_t sct_cdt_rsvd  : 13;
		uint64_t               :  3;
		uint64_t srb_cdt_rsvd  : 12;
		uint64_t ignore_spt    :  1;
		uint64_t ignore_smt    :  1;
		uint64_t ignore_sct    :  1;
		uint64_t ignore_srb    :  1;
	};
};

#define C_OXE_CFG_BUF_SH_CDT_OFFSET(idx)	(0x00000700+((idx)*8))
#define C_OXE_CFG_BUF_SH_CDT_ENTRIES	10
#define C_OXE_CFG_BUF_SH_CDT(idx)	(C_OXE_BASE + C_OXE_CFG_BUF_SH_CDT_OFFSET(idx))
#define C_OXE_CFG_BUF_SH_CDT_SIZE	0x00000050

union c_oxe_cfg_buf_sh_cdt {
	uint64_t qw;
	struct {
		uint64_t buf_sh_cdt         : 11;
		uint64_t                    :  5;
		uint64_t buf_sh_bc_cdt      : 11;
		uint64_t                    :  5;
		uint64_t buf_sh_bc_idc_cdt  : 11;
		uint64_t                    : 21;
	};
};

#define C_OXE_CFG_MCU_PARAM_OFFSET(idx)	(0x00001000+((idx)*8))
#define C_OXE_CFG_MCU_PARAM_ENTRIES	96
#define C_OXE_CFG_MCU_PARAM(idx)	(C_OXE_BASE + C_OXE_CFG_MCU_PARAM_OFFSET(idx))
#define C_OXE_CFG_MCU_PARAM_SIZE	0x00000300

union c_oxe_cfg_mcu_param {
	uint64_t qw;
	struct {
		uint64_t bc_map         :  4;
		uint64_t tsc_map        :  4;
		uint64_t wdrr_out_sel   :  2;
		uint64_t wdrr_in_sel    :  2;
		uint64_t stall_cnt_en0  :  1;
		uint64_t stall_cnt_en1  :  1;
		uint64_t pcp_map        :  3;
		uint64_t limit_sel      :  2;
		uint64_t                : 45;
	};
};

#define C_OXE_CFG_MCU_PRIORITY_LIMIT_OFFSET(idx)	(0x00001400+((idx)*8))
#define C_OXE_CFG_MCU_PRIORITY_LIMIT_ENTRIES	96
#define C_OXE_CFG_MCU_PRIORITY_LIMIT(idx)	(C_OXE_BASE + C_OXE_CFG_MCU_PRIORITY_LIMIT_OFFSET(idx))
#define C_OXE_CFG_MCU_PRIORITY_LIMIT_SIZE	0x00000300

union c_oxe_cfg_mcu_priority_limit {
	uint64_t qw;
	struct {
		uint64_t occ_blimit  : 11;
		uint64_t             :  5;
		uint64_t occ_climit  : 11;
		uint64_t             : 37;
	};
};

#define C_OXE_CFG_ARB_CONFIG_OFFSET	0x00001808
#define C_OXE_CFG_ARB_CONFIG	(C_OXE_BASE + C_OXE_CFG_ARB_CONFIG_OFFSET)
#define C_OXE_CFG_ARB_CONFIG_SIZE	0x00000008

union c_oxe_cfg_arb_config {
	uint64_t qw;
	struct {
		uint64_t fill_rate    : 10;
		uint64_t              :  2;
		uint64_t mfs_out_sel  :  2;
		uint64_t mfs_in_sel   :  2;
		uint64_t              : 48;
	};
};

#define C_OXE_CFG_ARB_MFS_OUT_OFFSET	0x00001810
#define C_OXE_CFG_ARB_MFS_OUT	(C_OXE_BASE + C_OXE_CFG_ARB_MFS_OUT_OFFSET)
#define C_OXE_CFG_ARB_MFS_OUT_SIZE	0x00000008

union c_oxe_cfg_arb_mfs_out {
	uint64_t qw;
	struct {
		uint64_t mfs0  :  9;
		uint64_t       :  3;
		uint64_t mfs1  :  9;
		uint64_t       :  3;
		uint64_t mfs2  :  9;
		uint64_t       :  3;
		uint64_t mfs3  :  9;
		uint64_t       : 19;
	};
};

#define C_OXE_CFG_ARB_MFS_IN_OFFSET	0x00001818
#define C_OXE_CFG_ARB_MFS_IN	(C_OXE_BASE + C_OXE_CFG_ARB_MFS_IN_OFFSET)
#define C_OXE_CFG_ARB_MFS_IN_SIZE	0x00000008

union c_oxe_cfg_arb_mfs_in {
	uint64_t qw;
	struct {
		uint64_t mfs0  :  9;
		uint64_t       :  3;
		uint64_t mfs1  :  9;
		uint64_t       :  3;
		uint64_t mfs2  :  9;
		uint64_t       :  3;
		uint64_t mfs3  :  9;
		uint64_t       : 19;
	};
};

#define C_OXE_CFG_ARB_DRR_OUT_OFFSET	0x00001820
#define C_OXE_CFG_ARB_DRR_OUT	(C_OXE_BASE + C_OXE_CFG_ARB_DRR_OUT_OFFSET)
#define C_OXE_CFG_ARB_DRR_OUT_SIZE	0x00000008

union c_oxe_cfg_arb_drr_out {
	uint64_t qw;
	struct {
		uint64_t quanta_val0  : 12;
		uint64_t quanta_val1  : 12;
		uint64_t quanta_val2  : 12;
		uint64_t quanta_val3  : 12;
		uint64_t              : 16;
	};
};

#define C_OXE_CFG_ARB_DRR_IN_OFFSET	0x00001828
#define C_OXE_CFG_ARB_DRR_IN	(C_OXE_BASE + C_OXE_CFG_ARB_DRR_IN_OFFSET)
#define C_OXE_CFG_ARB_DRR_IN_SIZE	0x00000008

union c_oxe_cfg_arb_drr_in {
	uint64_t qw;
	struct {
		uint64_t quanta_val0  : 12;
		uint64_t quanta_val1  : 12;
		uint64_t quanta_val2  : 12;
		uint64_t quanta_val3  : 12;
		uint64_t              : 16;
	};
};

#define C_OXE_CFG_ARB_HEAD_OUT_C_OFFSET	0x00001830
#define C_OXE_CFG_ARB_HEAD_OUT_C	(C_OXE_BASE + C_OXE_CFG_ARB_HEAD_OUT_C_OFFSET)
#define C_OXE_CFG_ARB_HEAD_OUT_C_SIZE	0x00000008

union c_oxe_cfg_arb_head_out_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_ARB_HEAD_IN_C_OFFSET	0x00001838
#define C_OXE_CFG_ARB_HEAD_IN_C	(C_OXE_BASE + C_OXE_CFG_ARB_HEAD_IN_C_OFFSET)
#define C_OXE_CFG_ARB_HEAD_IN_C_SIZE	0x00000008

union c_oxe_cfg_arb_head_in_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_ARB_PCP_MASK_OFFSET(idx)	(0x00001840+((idx)*8))
#define C_OXE_CFG_ARB_PCP_MASK_ENTRIES	8
#define C_OXE_CFG_ARB_PCP_MASK(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_PCP_MASK_OFFSET(idx))
#define C_OXE_CFG_ARB_PCP_MASK_SIZE	0x00000040

union c_oxe_cfg_arb_pcp_mask {
	uint64_t qw;
	struct {
		uint64_t tsc_mask  : 10;
		uint64_t           : 54;
	};
};

#define C_OXE_CFG_OUTSTANDING_LIMIT_OFFSET(idx)	(0x00001880+((idx)*8))
#define C_OXE_CFG_OUTSTANDING_LIMIT_ENTRIES	4
#define C_OXE_CFG_OUTSTANDING_LIMIT(idx)	(C_OXE_BASE + C_OXE_CFG_OUTSTANDING_LIMIT_OFFSET(idx))
#define C_OXE_CFG_OUTSTANDING_LIMIT_SIZE	0x00000020

union c_oxe_cfg_outstanding_limit {
	uint64_t qw;
	struct {
		uint64_t put_limit        : 11;
		uint64_t get_limit        : 11;
		uint64_t ioi_ord_limit    : 11;
		uint64_t ioi_unord_limit  : 13;
		uint64_t                  : 18;
	};
};

#define C_OXE_CFG_ARB_LEAF_OFFSET(idx)	(0x000018a0+((idx)*8))
#define C_OXE_CFG_ARB_LEAF_ENTRIES	10
#define C_OXE_CFG_ARB_LEAF(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_LEAF_OFFSET(idx))
#define C_OXE_CFG_ARB_LEAF_SIZE	0x00000050

union c_oxe_cfg_arb_leaf {
	uint64_t qw;
	struct {
		uint64_t pri          :  3;
		uint64_t              :  1;
		uint64_t parent       :  3;
		uint64_t              :  1;
		uint64_t mfs_out_sel  :  2;
		uint64_t mfs_in_sel   :  2;
		uint64_t enable_in    :  1;
		uint64_t              : 51;
	};
};

#define C_OXE_CFG_ARB_LEAF_OUT_A_OFFSET(idx)	(0x00001900+((idx)*8))
#define C_OXE_CFG_ARB_LEAF_OUT_A_ENTRIES	10
#define C_OXE_CFG_ARB_LEAF_OUT_A(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_LEAF_OUT_A_OFFSET(idx))
#define C_OXE_CFG_ARB_LEAF_OUT_A_SIZE	0x00000050

union c_oxe_cfg_arb_leaf_out_a {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t           : 34;
	};
};

#define C_OXE_CFG_ARB_LEAF_OUT_C_OFFSET(idx)	(0x00001980+((idx)*8))
#define C_OXE_CFG_ARB_LEAF_OUT_C_ENTRIES	10
#define C_OXE_CFG_ARB_LEAF_OUT_C(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_LEAF_OUT_C_OFFSET(idx))
#define C_OXE_CFG_ARB_LEAF_OUT_C_SIZE	0x00000050

union c_oxe_cfg_arb_leaf_out_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_ARB_LEAF_IN_A_OFFSET(idx)	(0x00001a00+((idx)*8))
#define C_OXE_CFG_ARB_LEAF_IN_A_ENTRIES	10
#define C_OXE_CFG_ARB_LEAF_IN_A(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_LEAF_IN_A_OFFSET(idx))
#define C_OXE_CFG_ARB_LEAF_IN_A_SIZE	0x00000050

union c_oxe_cfg_arb_leaf_in_a {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t           : 34;
	};
};

#define C_OXE_CFG_ARB_LEAF_IN_C_OFFSET(idx)	(0x00001a80+((idx)*8))
#define C_OXE_CFG_ARB_LEAF_IN_C_ENTRIES	10
#define C_OXE_CFG_ARB_LEAF_IN_C(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_LEAF_IN_C_OFFSET(idx))
#define C_OXE_CFG_ARB_LEAF_IN_C_SIZE	0x00000050

union c_oxe_cfg_arb_leaf_in_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_ARB_BRANCH_OFFSET(idx)	(0x00001b00+((idx)*8))
#define C_OXE_CFG_ARB_BRANCH_ENTRIES	5
#define C_OXE_CFG_ARB_BRANCH(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_BRANCH_OFFSET(idx))
#define C_OXE_CFG_ARB_BRANCH_SIZE	0x00000028

union c_oxe_cfg_arb_branch {
	uint64_t qw;
	struct {
		uint64_t pri          :  3;
		uint64_t              :  1;
		uint64_t mfs_out_sel  :  2;
		uint64_t mfs_in_sel   :  2;
		uint64_t              : 56;
	};
};

#define C_OXE_CFG_ARB_BRANCH_OUT_A_OFFSET(idx)	(0x00001b40+((idx)*8))
#define C_OXE_CFG_ARB_BRANCH_OUT_A_ENTRIES	5
#define C_OXE_CFG_ARB_BRANCH_OUT_A(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_BRANCH_OUT_A_OFFSET(idx))
#define C_OXE_CFG_ARB_BRANCH_OUT_A_SIZE	0x00000028

union c_oxe_cfg_arb_branch_out_a {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t           : 34;
	};
};

#define C_OXE_CFG_ARB_BRANCH_OUT_C_OFFSET(idx)	(0x00001b80+((idx)*8))
#define C_OXE_CFG_ARB_BRANCH_OUT_C_ENTRIES	5
#define C_OXE_CFG_ARB_BRANCH_OUT_C(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_BRANCH_OUT_C_OFFSET(idx))
#define C_OXE_CFG_ARB_BRANCH_OUT_C_SIZE	0x00000028

union c_oxe_cfg_arb_branch_out_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_ARB_BRANCH_IN_A_OFFSET(idx)	(0x00001bc0+((idx)*8))
#define C_OXE_CFG_ARB_BRANCH_IN_A_ENTRIES	5
#define C_OXE_CFG_ARB_BRANCH_IN_A(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_BRANCH_IN_A_OFFSET(idx))
#define C_OXE_CFG_ARB_BRANCH_IN_A_SIZE	0x00000028

union c_oxe_cfg_arb_branch_in_a {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t           : 34;
	};
};

#define C_OXE_CFG_ARB_BRANCH_IN_C_OFFSET(idx)	(0x00001c00+((idx)*8))
#define C_OXE_CFG_ARB_BRANCH_IN_C_ENTRIES	5
#define C_OXE_CFG_ARB_BRANCH_IN_C(idx)	(C_OXE_BASE + C_OXE_CFG_ARB_BRANCH_IN_C_OFFSET(idx))
#define C_OXE_CFG_ARB_BRANCH_IN_C_SIZE	0x00000028

union c_oxe_cfg_arb_branch_in_c {
	uint64_t qw;
	struct {
		uint64_t fill_qty  : 11;
		uint64_t           :  1;
		uint64_t limit     : 18;
		uint64_t enable    :  1;
		uint64_t           : 33;
	};
};

#define C_OXE_CFG_FGFC_CNT_OFFSET(idx)	(0x00001c40+((idx)*8))
#define C_OXE_CFG_FGFC_CNT_ENTRIES	4
#define C_OXE_CFG_FGFC_CNT(idx)	(C_OXE_BASE + C_OXE_CFG_FGFC_CNT_OFFSET(idx))
#define C_OXE_CFG_FGFC_CNT_SIZE	0x00000020

union c_oxe_cfg_fgfc_cnt {
	uint64_t qw;
	struct {
		uint64_t vni_match  : 16;
		uint64_t vni_mask   : 16;
		uint64_t            : 32;
	};
};

#define C_OXE_STS_COMMON_OFFSET	0x00002000
#define C_OXE_STS_COMMON	(C_OXE_BASE + C_OXE_STS_COMMON_OFFSET)
#define C_OXE_STS_COMMON_SIZE	0x00000008

union c_oxe_sts_common {
	uint64_t qw;
	struct {
		uint64_t init_done  :  1;
		uint64_t            : 63;
	};
};

#define C_PARBS_CFG_CRNC_OFFSET	0x00000000
#define C_PARBS_CFG_CRNC	(C_PARBS_BASE + C_PARBS_CFG_CRNC_OFFSET)
#define C_PARBS_CFG_CRNC_SIZE	0x00000008

union c_parbs_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_PARBS_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_PARBS_CFG_RT_OFFSET	(C_PARBS_BASE + C_PARBS_CFG_RT_OFFSET_OFFSET)
#define C_PARBS_CFG_RT_OFFSET_SIZE	0x00000008

union c_parbs_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_PARBS_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_PARBS_ERR_ELAPSED_TIME	(C_PARBS_BASE + C_PARBS_ERR_ELAPSED_TIME_OFFSET)
#define C_PARBS_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_parbs_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_PARBS_ERR_FLG_OFFSET	0x00000108
#define C_PARBS_ERR_FLG	(C_PARBS_BASE + C_PARBS_ERR_FLG_OFFSET)
#define C_PARBS_ERR_FLG_SIZE	0x00000008

union c_parbs_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                   :  1;
		uint64_t crnc_ring_sync_error      :  1;
		uint64_t crnc_ring_ecc_sbe         :  1;
		uint64_t crnc_ring_ecc_mbe         :  1;
		uint64_t crnc_ring_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_unknown      :  1;
		uint64_t crnc_csr_cmd_incomplete   :  1;
		uint64_t crnc_buf_ecc_sbe          :  1;
		uint64_t crnc_buf_ecc_mbe          :  1;
		uint64_t                           :  7;
		uint64_t cq_tarb_hdr_cor           :  1;
		uint64_t cq_tarb_hdr_ucor          :  1;
		uint64_t cq_tarb_data_cor          :  1;
		uint64_t cq_tarb_data_ucor         :  1;
		uint64_t ee_tarb_hdr_cor           :  1;
		uint64_t ee_tarb_hdr_ucor          :  1;
		uint64_t ee_tarb_data_cor          :  1;
		uint64_t ee_tarb_data_ucor         :  1;
		uint64_t ixe_tarb_hdr_cor          :  1;
		uint64_t ixe_tarb_hdr_ucor         :  1;
		uint64_t ixe_tarb_data_cor         :  1;
		uint64_t ixe_tarb_data_ucor        :  1;
		uint64_t oxe_tarb_hdr_cor          :  1;
		uint64_t oxe_tarb_hdr_ucor         :  1;
		uint64_t atu_tarb_hdr_cor          :  1;
		uint64_t atu_tarb_hdr_ucor         :  1;
		uint64_t atu_tarb_data_cor         :  1;
		uint64_t atu_tarb_data_ucor        :  1;
		uint64_t tarb_pi_hdr_cor           :  1;
		uint64_t tarb_pi_hdr_ucor          :  1;
		uint64_t tarb_pi_data_cor          :  1;
		uint64_t tarb_pi_data_ucor         :  1;
		uint64_t cq_p_fifo_overrun         :  1;
		uint64_t cq_p_fifo_underrun        :  1;
		uint64_t cq_np_fifo_overrun        :  1;
		uint64_t cq_np_fifo_underrun       :  1;
		uint64_t ee_p_fifo_overrun         :  1;
		uint64_t ee_p_fifo_underrun        :  1;
		uint64_t ixe_p_fifo_overrun        :  1;
		uint64_t ixe_p_fifo_underrun       :  1;
		uint64_t ixe_np_fifo_overrun       :  1;
		uint64_t ixe_np_fifo_underrun      :  1;
		uint64_t oxe_np_fifo_overrun       :  1;
		uint64_t oxe_np_fifo_underrun      :  1;
		uint64_t atu_p_fifo_overrun        :  1;
		uint64_t atu_p_fifo_underrun       :  1;
		uint64_t atu_np_fifo_overrun       :  1;
		uint64_t atu_np_fifo_underrun      :  1;
		uint64_t ord_p_fifo_overrun        :  1;
		uint64_t ord_p_fifo_underrun       :  1;
		uint64_t ord_np_fifo_overrun       :  1;
		uint64_t ord_np_fifo_underrun      :  1;
		uint64_t ord_np_cnt_overrun        :  1;
		uint64_t ord_np_cnt_underrun       :  1;
		uint64_t tarb_pi_p_cdt_underflow   :  1;
		uint64_t tarb_pi_np_cdt_underflow  :  1;
		uint64_t                           :  2;
	};
};

#define C_PARBS_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_PARBS_ERR_FIRST_FLG	(C_PARBS_BASE + C_PARBS_ERR_FIRST_FLG_OFFSET)
#define C_PARBS_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PARBS_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_PARBS_ERR_FIRST_FLG_TS	(C_PARBS_BASE + C_PARBS_ERR_FIRST_FLG_TS_OFFSET)
#define C_PARBS_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PARBS_ERR_CLR_OFFSET	0x00000120
#define C_PARBS_ERR_CLR	(C_PARBS_BASE + C_PARBS_ERR_CLR_OFFSET)
#define C_PARBS_ERR_CLR_SIZE	0x00000008
#define C_PARBS_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_PARBS_ERR_IRQA_MSK	(C_PARBS_BASE + C_PARBS_ERR_IRQA_MSK_OFFSET)
#define C_PARBS_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PARBS_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_PARBS_ERR_IRQB_MSK	(C_PARBS_BASE + C_PARBS_ERR_IRQB_MSK_OFFSET)
#define C_PARBS_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PARBS_ERR_INFO_MSK_OFFSET	0x00000140
#define C_PARBS_ERR_INFO_MSK	(C_PARBS_BASE + C_PARBS_ERR_INFO_MSK_OFFSET)
#define C_PARBS_ERR_INFO_MSK_SIZE	0x00000008
#define C_PARBS_ERR_INFO_MEM_OFFSET	0x00000180
#define C_PARBS_ERR_INFO_MEM	(C_PARBS_BASE + C_PARBS_ERR_INFO_MEM_OFFSET)
#define C_PARBS_ERR_INFO_MEM_SIZE	0x00000008

union c_parbs_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 11;
		uint64_t                    :  1;
		uint64_t cor_mem_id         :  4;
		uint64_t                    :  3;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 11;
		uint64_t                    :  1;
		uint64_t ucor_mem_id        :  4;
		uint64_t                    :  3;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_PARBS_EXT_ERR_FLG_OFFSET	0x00000208
#define C_PARBS_EXT_ERR_FLG	(C_PARBS_BASE + C_PARBS_EXT_ERR_FLG_OFFSET)
#define C_PARBS_EXT_ERR_FLG_SIZE	0x00000008

union c_parbs_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag            :  1;
		uint64_t pi_rarb_hdr_cor    :  1;
		uint64_t pi_rarb_hdr_ucor   :  1;
		uint64_t pi_rarb_data_cor   :  1;
		uint64_t pi_rarb_data_ucor  :  1;
		uint64_t                    : 59;
	};
};

#define C_PARBS_EXT_ERR_FIRST_FLG_OFFSET	0x00000210
#define C_PARBS_EXT_ERR_FIRST_FLG	(C_PARBS_BASE + C_PARBS_EXT_ERR_FIRST_FLG_OFFSET)
#define C_PARBS_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PARBS_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000218
#define C_PARBS_EXT_ERR_FIRST_FLG_TS	(C_PARBS_BASE + C_PARBS_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_PARBS_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PARBS_EXT_ERR_CLR_OFFSET	0x00000220
#define C_PARBS_EXT_ERR_CLR	(C_PARBS_BASE + C_PARBS_EXT_ERR_CLR_OFFSET)
#define C_PARBS_EXT_ERR_CLR_SIZE	0x00000008
#define C_PARBS_EXT_ERR_IRQA_MSK_OFFSET	0x00000228
#define C_PARBS_EXT_ERR_IRQA_MSK	(C_PARBS_BASE + C_PARBS_EXT_ERR_IRQA_MSK_OFFSET)
#define C_PARBS_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PARBS_EXT_ERR_IRQB_MSK_OFFSET	0x00000230
#define C_PARBS_EXT_ERR_IRQB_MSK	(C_PARBS_BASE + C_PARBS_EXT_ERR_IRQB_MSK_OFFSET)
#define C_PARBS_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PARBS_EXT_ERR_INFO_MSK_OFFSET	0x00000240
#define C_PARBS_EXT_ERR_INFO_MSK	(C_PARBS_BASE + C_PARBS_EXT_ERR_INFO_MSK_OFFSET)
#define C_PARBS_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_PCT_CFG_CRNC_OFFSET	0x00000000
#define C_PCT_CFG_CRNC	(C_PCT_BASE + C_PCT_CFG_CRNC_OFFSET)
#define C_PCT_CFG_CRNC_SIZE	0x00000008

union c_pct_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_PCT_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_PCT_MSC_SHADOW_ACTION	(C_PCT_BASE + C_PCT_MSC_SHADOW_ACTION_OFFSET)
#define C_PCT_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_PCT_MSC_SHADOW_OFFSET	0x00000040
#define C_PCT_MSC_SHADOW	(C_PCT_BASE + C_PCT_MSC_SHADOW_OFFSET)
#define C_PCT_MSC_SHADOW_SIZE	0x00000080

union c_pct_msc_shadow {
	uint64_t qw[16];
};

#define C_PCT_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_PCT_ERR_ELAPSED_TIME	(C_PCT_BASE + C_PCT_ERR_ELAPSED_TIME_OFFSET)
#define C_PCT_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_pct_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C1_PCT_ERR_FLG_OFFSET	0x00000108
#define C1_PCT_ERR_FLG	(C_PCT_BASE + C1_PCT_ERR_FLG_OFFSET)
#define C1_PCT_ERR_FLG_SIZE	0x00000008

union c1_pct_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t srb_alloc_dup            :  1;
		uint64_t srb_dealloc_dup          :  1;
		uint64_t                          :  1;
		uint64_t flop_fifo_cor            :  1;
		uint64_t mst_cam_free_cor         :  1;
		uint64_t sct_cam_free_cor         :  1;
		uint64_t tct_cam_free_cor         :  1;
		uint64_t trs_cam_free_cor         :  1;
		uint64_t sct_tbl_cor              :  1;
		uint64_t smt_tbl_cor              :  1;
		uint64_t spt_tbl_cor              :  1;
		uint64_t tct_tbl_cor              :  1;
		uint64_t trs_tbl_cor              :  1;
		uint64_t req_redo_cor             :  1;
		uint64_t cls_clr_req_gen_cor      :  1;
		uint64_t srb_nlist_ram_cor        :  1;
		uint64_t srb_data_ram_cor         :  1;
		uint64_t flop_fifo_ucor           :  1;
		uint64_t mst_cam_free_ucor        :  1;
		uint64_t sct_cam_free_ucor        :  1;
		uint64_t tct_cam_free_ucor        :  1;
		uint64_t trs_cam_free_ucor        :  1;
		uint64_t sct_tbl_ucor             :  1;
		uint64_t smt_tbl_ucor             :  1;
		uint64_t spt_tbl_ucor             :  1;
		uint64_t tct_tbl_ucor             :  1;
		uint64_t trs_tbl_ucor             :  1;
		uint64_t req_redo_ucor            :  1;
		uint64_t cls_clr_req_gen_ucor     :  1;
		uint64_t srb_nlist_ram_ucor       :  1;
		uint64_t srb_data_ram_ucor        :  1;
		uint64_t mst_cam_ucor             :  1;
		uint64_t sct_cam_ucor             :  1;
		uint64_t tct_cam_ucor             :  1;
		uint64_t trs_cam_ucor             :  1;
		uint64_t fifo_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t                          : 18;
	};
};

#define C2_PCT_ERR_FLG_OFFSET	0x00000108
#define C2_PCT_ERR_FLG	(C_PCT_BASE + C2_PCT_ERR_FLG_OFFSET)
#define C2_PCT_ERR_FLG_SIZE	0x00000008

union c2_pct_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                  :  1;
		uint64_t crnc_ring_sync_error     :  1;
		uint64_t crnc_ring_ecc_sbe        :  1;
		uint64_t crnc_ring_ecc_mbe        :  1;
		uint64_t crnc_ring_cmd_unknown    :  1;
		uint64_t crnc_csr_cmd_unknown     :  1;
		uint64_t crnc_csr_cmd_incomplete  :  1;
		uint64_t crnc_buf_ecc_sbe         :  1;
		uint64_t crnc_buf_ecc_mbe         :  1;
		uint64_t srb_alloc_dup            :  1;
		uint64_t srb_dealloc_dup          :  1;
		uint64_t                          :  1;
		uint64_t flop_fifo_cor            :  1;
		uint64_t mst_cam_free_cor         :  1;
		uint64_t sct_cam_free_cor         :  1;
		uint64_t tct_cam_free_cor         :  1;
		uint64_t trs_cam_free_cor         :  1;
		uint64_t sct_tbl_cor              :  1;
		uint64_t smt_tbl_cor              :  1;
		uint64_t spt_tbl_cor              :  1;
		uint64_t tct_tbl_cor              :  1;
		uint64_t trs_tbl_cor              :  1;
		uint64_t req_redo_cor             :  1;
		uint64_t cls_clr_req_gen_cor      :  1;
		uint64_t srb_nlist_ram_cor        :  1;
		uint64_t srb_data_ram_cor         :  1;
		uint64_t cls_rsp_gen_cor          :  1;
		uint64_t nack_trs_rsp_gen_cor     :  1;
		uint64_t flop_fifo_ucor           :  1;
		uint64_t mst_cam_free_ucor        :  1;
		uint64_t sct_cam_free_ucor        :  1;
		uint64_t tct_cam_free_ucor        :  1;
		uint64_t trs_cam_free_ucor        :  1;
		uint64_t sct_tbl_ucor             :  1;
		uint64_t smt_tbl_ucor             :  1;
		uint64_t spt_tbl_ucor             :  1;
		uint64_t tct_tbl_ucor             :  1;
		uint64_t trs_tbl_ucor             :  1;
		uint64_t req_redo_ucor            :  1;
		uint64_t cls_clr_req_gen_ucor     :  1;
		uint64_t srb_nlist_ram_ucor       :  1;
		uint64_t srb_data_ram_ucor        :  1;
		uint64_t mst_cam_ucor             :  1;
		uint64_t sct_cam_ucor             :  1;
		uint64_t tct_cam_ucor             :  1;
		uint64_t trs_cam_ucor             :  1;
		uint64_t nack_trs_rsp_gen_ucor    :  1;
		uint64_t cls_rsp_gen_ucor         :  1;
		uint64_t fifo_err                 :  1;
		uint64_t credit_uflw              :  1;
		uint64_t                          : 14;
	};
};

#define C_PCT_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_PCT_ERR_FIRST_FLG	(C_PCT_BASE + C_PCT_ERR_FIRST_FLG_OFFSET)
#define C_PCT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PCT_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_PCT_ERR_FIRST_FLG_TS	(C_PCT_BASE + C_PCT_ERR_FIRST_FLG_TS_OFFSET)
#define C_PCT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PCT_ERR_CLR_OFFSET	0x00000120
#define C_PCT_ERR_CLR	(C_PCT_BASE + C_PCT_ERR_CLR_OFFSET)
#define C_PCT_ERR_CLR_SIZE	0x00000008
#define C_PCT_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_PCT_ERR_IRQA_MSK	(C_PCT_BASE + C_PCT_ERR_IRQA_MSK_OFFSET)
#define C_PCT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PCT_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_PCT_ERR_IRQB_MSK	(C_PCT_BASE + C_PCT_ERR_IRQB_MSK_OFFSET)
#define C_PCT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PCT_ERR_INFO_MSK_OFFSET	0x00000140
#define C_PCT_ERR_INFO_MSK	(C_PCT_BASE + C_PCT_ERR_INFO_MSK_OFFSET)
#define C_PCT_ERR_INFO_MSK_SIZE	0x00000008
#define C_PCT_EXT_ERR_FLG_OFFSET	0x00000148
#define C_PCT_EXT_ERR_FLG	(C_PCT_BASE + C_PCT_EXT_ERR_FLG_OFFSET)
#define C_PCT_EXT_ERR_FLG_SIZE	0x00000008

union c_pct_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                         :  1;
		uint64_t sct_tbl_dealloc                 :  1;
		uint64_t sct_tbl_alloc                   :  1;
		uint64_t sct_tbl_rd_misc_unused          :  1;
		uint64_t sct_tbl_wr_misc_unused          :  1;
		uint64_t sct_tbl_rd_ram_unused           :  1;
		uint64_t sct_tbl_wr_ram_unused           :  1;
		uint64_t smt_tbl_dealloc                 :  1;
		uint64_t smt_tbl_alloc                   :  1;
		uint64_t smt_tbl_rd_ram_unused           :  1;
		uint64_t smt_tbl_wr_ram_unused           :  1;
		uint64_t spt_tbl_dealloc                 :  1;
		uint64_t spt_tbl_alloc                   :  1;
		uint64_t spt_tbl_rd_misc_unused          :  1;
		uint64_t spt_tbl_wr_misc_unused          :  1;
		uint64_t spt_tbl_rd_ram_unused           :  1;
		uint64_t spt_tbl_wr_ram_unused           :  1;
		uint64_t spt_tbl_set_wrong_status        :  1;
		uint64_t tct_tbl_dealloc                 :  1;
		uint64_t tct_tbl_alloc                   :  1;
		uint64_t tct_tbl_rd_misc_unused          :  1;
		uint64_t tct_tbl_rd_ram_unused           :  1;
		uint64_t tct_tbl_wr_ram_unused           :  1;
		uint64_t trs_tbl_dealloc                 :  1;
		uint64_t trs_tbl_alloc                   :  1;
		uint64_t trs_tbl_rd_ram_unused           :  1;
		uint64_t trs_tbl_wr_ram_unused           :  1;
		uint64_t oxe_sct_miss_on_mid             :  1;
		uint64_t oxe_failed_sct_alloc            :  1;
		uint64_t oxe_illegal_req                 :  1;
		uint64_t oxe_new_conn_not_som            :  1;
		uint64_t oxe_invalid_smt_not_som         :  1;
		uint64_t oxe_illegal_access2_fsm         :  1;
		uint64_t oxe_smt_req_cnt_exceed          :  1;
		uint64_t oxe_sct_req_cnt_exceed          :  1;
		uint64_t ixe_req_tct_not_in_use          :  1;
		uint64_t ixe_req_tct_miss_unordered_mid  :  1;
		uint64_t ixe_req_mst_missing             :  1;
		uint64_t ixe_req_clr_out_of_range        :  1;
		uint64_t ixe_rsp_spt_not_in_use          :  1;
		uint64_t ixe_rsp_spt_status_not_pend     :  1;
		uint64_t ixe_rsp_smt_pend_cnt_exceed     :  1;
		uint64_t ixe_rsp_sct_pend_cnt_exceed     :  1;
		uint64_t ixe_rsp_smt_invalid             :  1;
		uint64_t ixe_rsp_spt_not_get_vld         :  1;
		uint64_t ixe_rsp_cls_not_pend            :  1;
		uint64_t ixe_gcomp_spt_not_in_use        :  1;
		uint64_t ixe_gcomp_not_get_ack_rcvd      :  1;
		uint64_t ixe_gcomp_smt_pend_cnt_exceed   :  1;
		uint64_t ixe_gcomp_sct_pend_cnt_exceed   :  1;
		uint64_t ixe_gcomp_smt_invalid           :  1;
		uint64_t ixe_gcomp_spt_not_get_vld       :  1;
		uint64_t src_cls_spt_wrong_sct           :  1;
		uint64_t src_cls_spt_not_used            :  1;
		uint64_t src_cls_spt_not_opcomp          :  1;
		uint64_t src_cls_bad_close               :  1;
		uint64_t pkto_illegal_req                :  1;
		uint64_t pkto_illegal_pkttype            :  1;
		uint64_t pkto_head_fsm                   :  1;
		uint64_t srb_wr_rd_same_cycle            :  1;
		uint64_t sct_miss_unrestricted_mid       :  1;
		uint64_t mbe_in_ctrls                    :  1;
		uint64_t sct_cam_err                     :  1;
		uint64_t mst_cam_err                     :  1;
	};
};

#define C_PCT_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C_PCT_EXT_ERR_FIRST_FLG	(C_PCT_BASE + C_PCT_EXT_ERR_FIRST_FLG_OFFSET)
#define C_PCT_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PCT_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C_PCT_EXT_ERR_FIRST_FLG_TS	(C_PCT_BASE + C_PCT_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_PCT_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PCT_EXT_ERR_CLR_OFFSET	0x00000160
#define C_PCT_EXT_ERR_CLR	(C_PCT_BASE + C_PCT_EXT_ERR_CLR_OFFSET)
#define C_PCT_EXT_ERR_CLR_SIZE	0x00000008
#define C_PCT_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C_PCT_EXT_ERR_IRQA_MSK	(C_PCT_BASE + C_PCT_EXT_ERR_IRQA_MSK_OFFSET)
#define C_PCT_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PCT_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C_PCT_EXT_ERR_IRQB_MSK	(C_PCT_BASE + C_PCT_EXT_ERR_IRQB_MSK_OFFSET)
#define C_PCT_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PCT_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C_PCT_EXT_ERR_INFO_MSK	(C_PCT_BASE + C_PCT_EXT_ERR_INFO_MSK_OFFSET)
#define C_PCT_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_PCT_ERR_INFO_MEM_OFFSET	0x00000188
#define C_PCT_ERR_INFO_MEM	(C_PCT_BASE + C_PCT_ERR_INFO_MEM_OFFSET)
#define C_PCT_ERR_INFO_MEM_SIZE	0x00000008

union c_pct_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t cor_address        : 13;
		uint64_t cor_mem_id         :  7;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t ucor_address       : 13;
		uint64_t ucor_mem_id        :  7;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_PCT_ERR_INFO_FIFO_OFFSET	0x00000190
#define C_PCT_ERR_INFO_FIFO	(C_PCT_BASE + C_PCT_ERR_INFO_FIFO_OFFSET)
#define C_PCT_ERR_INFO_FIFO_SIZE	0x00000008

union c_pct_err_info_fifo {
	uint64_t qw;
	struct {
		uint64_t overrun   :  1;
		uint64_t underrun  :  1;
		uint64_t           :  6;
		uint64_t fifo_id   :  6;
		uint64_t           : 50;
	};
};

#define C_PCT_ERR_INFO_CC_OFFSET	0x00000198
#define C_PCT_ERR_INFO_CC	(C_PCT_BASE + C_PCT_ERR_INFO_CC_OFFSET)
#define C_PCT_ERR_INFO_CC_SIZE	0x00000008

union c_pct_err_info_cc {
	uint64_t qw;
	struct {
		uint64_t cc_id   :  6;
		uint64_t         :  2;
		uint64_t sub_id  :  3;
		uint64_t         : 53;
	};
};

#define C_PCT_ERR_INFO_EXT_OFFSET	0x000001a0
#define C_PCT_ERR_INFO_EXT	(C_PCT_BASE + C_PCT_ERR_INFO_EXT_OFFSET)
#define C_PCT_ERR_INFO_EXT_SIZE	0x00000008

union c_pct_err_info_ext {
	uint64_t qw;
	struct {
		uint64_t sct_wr_err_id  :  5;
		uint64_t sct_rd_err_id  :  5;
		uint64_t                :  2;
		uint64_t smt_wr_err_id  :  4;
		uint64_t smt_rd_err_id  :  4;
		uint64_t spt_wr_err_id  :  5;
		uint64_t spt_rd_err_id  :  5;
		uint64_t                :  2;
		uint64_t tct_wr_err_id  :  3;
		uint64_t tct_rd_err_id  :  3;
		uint64_t                :  2;
		uint64_t trs_wr_err_id  :  3;
		uint64_t trs_rd_err_id  :  3;
		uint64_t                :  2;
		uint64_t mbe_err_id     :  5;
		uint64_t                : 11;
	};
};

#define C_PCT_CFG_EQ_RETRY_Q_HANDLE_OFFSET	0x00000300
#define C_PCT_CFG_EQ_RETRY_Q_HANDLE	(C_PCT_BASE + C_PCT_CFG_EQ_RETRY_Q_HANDLE_OFFSET)
#define C_PCT_CFG_EQ_RETRY_Q_HANDLE_SIZE	0x00000008

union c_pct_cfg_eq_retry_q_handle {
	uint64_t qw;
	struct {
		uint64_t eq_handle  : 11;
		uint64_t            : 53;
	};
};

#define C_PCT_CFG_EQ_TGT_Q_HANDLE_OFFSET	0x00000308
#define C_PCT_CFG_EQ_TGT_Q_HANDLE	(C_PCT_BASE + C_PCT_CFG_EQ_TGT_Q_HANDLE_OFFSET)
#define C_PCT_CFG_EQ_TGT_Q_HANDLE_SIZE	0x00000008

union c_pct_cfg_eq_tgt_q_handle {
	uint64_t qw;
	struct {
		uint64_t eq_handle  : 11;
		uint64_t            : 53;
	};
};

#define C_PCT_CFG_EQ_CONN_LD_Q_HANDLE_OFFSET	0x00000310
#define C_PCT_CFG_EQ_CONN_LD_Q_HANDLE	(C_PCT_BASE + C_PCT_CFG_EQ_CONN_LD_Q_HANDLE_OFFSET)
#define C_PCT_CFG_EQ_CONN_LD_Q_HANDLE_SIZE	0x00000008

union c_pct_cfg_eq_conn_ld_q_handle {
	uint64_t qw;
	struct {
		uint64_t eq_handle  : 11;
		uint64_t            : 53;
	};
};

#define C_PCT_CFG_SRB_RETRY_PTRS_OFFSET(idx)	(0x000003e0+((idx)*8))
#define C_PCT_CFG_SRB_RETRY_PTRS_ENTRIES	4
#define C_PCT_CFG_SRB_RETRY_PTRS(idx)	(C_PCT_BASE + C_PCT_CFG_SRB_RETRY_PTRS_OFFSET(idx))
#define C_PCT_CFG_SRB_RETRY_PTRS_SIZE	0x00000020

struct c_pct_cfg_srb_retry_ptrs_entry {
	uint32_t ptr     : 11;
	uint32_t	 :  1;
	uint32_t vld     :  1;
	uint32_t	 :  3;
	uint32_t spt_idx : 11;
	uint32_t	 :  1;
	uint32_t try_num :  3;
	uint32_t	 :  1;
};

union c_pct_cfg_srb_retry_ptrs {
	uint64_t qw;
	struct c_pct_cfg_srb_retry_ptrs_entry e[2];
};

#define C_PCT_CFG_SRB_RETRY_CTRL_OFFSET	0x00000400
#define C_PCT_CFG_SRB_RETRY_CTRL	(C_PCT_BASE + C_PCT_CFG_SRB_RETRY_CTRL_OFFSET)
#define C_PCT_CFG_SRB_RETRY_CTRL_SIZE	0x00000008

union c_pct_cfg_srb_retry_ctrl {
	uint64_t qw;
	struct {
		uint64_t pcp_pause_st  :  8;
		uint64_t current_ptr   :  3;
		uint64_t               :  5;
		uint64_t grp_nxt       :  1;
		uint64_t               :  7;
		uint64_t error         :  1;
		uint64_t               :  3;
		uint64_t sw_abort      :  1;
		uint64_t               :  3;
		uint64_t hw_sw_sync    :  1;
		uint64_t               :  3;
		uint64_t pause_mode    :  1;
		uint64_t               : 27;
	};
};

#define C1_PCT_CFG_SRB_RETRY_GRP_CTRL_OFFSET(idx)	(0x00000408+((idx)*8))
#define C1_PCT_CFG_SRB_RETRY_GRP_CTRL_ENTRIES	2
#define C1_PCT_CFG_SRB_RETRY_GRP_CTRL(idx)	(C_PCT_BASE + C1_PCT_CFG_SRB_RETRY_GRP_CTRL_OFFSET(idx))
#define C1_PCT_CFG_SRB_RETRY_GRP_CTRL_SIZE	0x00000010

union c1_pct_cfg_srb_retry_grp_ctrl {
	uint64_t qw;
	struct {
		uint64_t grp_loaded            :  1;
		uint64_t grp_active            :  1;
		uint64_t grp_paused            :  1;
		uint64_t grp_sct_idx_vld       :  1;
		uint64_t grp_sct_idx           : 12;
		uint64_t grp_pcp               :  3;
		uint64_t                       :  5;
		uint64_t grp_unset_clr_seqnum  :  1;
		uint64_t                       : 39;
	};
};

#define C2_PCT_CFG_SRB_RETRY_GRP_CTRL_OFFSET(idx)	(0x00000408+((idx)*8))
#define C2_PCT_CFG_SRB_RETRY_GRP_CTRL_ENTRIES	2
#define C2_PCT_CFG_SRB_RETRY_GRP_CTRL(idx)	(C_PCT_BASE + C2_PCT_CFG_SRB_RETRY_GRP_CTRL_OFFSET(idx))
#define C2_PCT_CFG_SRB_RETRY_GRP_CTRL_SIZE	0x00000010

union c2_pct_cfg_srb_retry_grp_ctrl {
	uint64_t qw;
	struct {
		uint64_t grp_loaded                :  1;
		uint64_t grp_active                :  1;
		uint64_t grp_paused                :  1;
		uint64_t grp_sct_idx_vld           :  1;
		uint64_t grp_sct_idx               : 12;
		uint64_t grp_pcp                   :  3;
		uint64_t                           :  5;
		uint64_t grp_update_clr_seqnum_en  :  1;
		uint64_t grp_update_clr_seqnum     : 12;
		uint64_t                           : 27;
	};
};

#define C_PCT_CFG_PKT_MISC_OFFSET	0x00000430
#define C_PCT_CFG_PKT_MISC	(C_PCT_BASE + C_PCT_CFG_PKT_MISC_OFFSET)
#define C_PCT_CFG_PKT_MISC_SIZE	0x00000008

union c_pct_cfg_pkt_misc {
	uint64_t qw;
	struct {
		uint64_t sm_fab_hdr                 :  4;
		uint64_t                            :  4;
		uint64_t ptl_proto                  :  8;
		uint64_t pktid_4000                 :  1;
		uint64_t                            :  7;
		uint64_t simple_mode                :  1;
		uint64_t                            :  3;
		uint64_t trs_pend_rsp_nack_disable  :  1;
		uint64_t                            : 15;
		uint64_t sfa_nid                    : 20;
	};
};

#define C_PCT_CFG_TIMING_OFFSET	0x00000440
#define C_PCT_CFG_TIMING	(C_PCT_BASE + C_PCT_CFG_TIMING_OFFSET)
#define C_PCT_CFG_TIMING_SIZE	0x00000008

union c_pct_cfg_timing {
	uint64_t qw;
	struct {
		uint64_t stall_ctr              :  5;
		uint64_t                        :  3;
		uint64_t spt_timeout_epoch_sel  :  6;
		uint64_t                        :  2;
		uint64_t sct_idle_epoch_sel     :  6;
		uint64_t                        :  2;
		uint64_t sct_close_epoch_sel    :  6;
		uint64_t                        :  2;
		uint64_t tct_timeout_epoch_sel  :  6;
		uint64_t                        :  2;
		uint64_t put_get_same_sct_en    :  1;
		uint64_t                        :  7;
		uint64_t accept_late_spt_rsp    :  1;
		uint64_t                        : 15;
	};
};

#define C1_PCT_CFG_IXE_REQ_FIFO_LIMITS_OFFSET	0x00000508
#define C1_PCT_CFG_IXE_REQ_FIFO_LIMITS	(C_PCT_BASE + C1_PCT_CFG_IXE_REQ_FIFO_LIMITS_OFFSET)
#define C1_PCT_CFG_IXE_REQ_FIFO_LIMITS_SIZE	0x00000008

union c1_pct_cfg_ixe_req_fifo_limits {
	uint64_t qw;
	struct {
		uint64_t ixe_req_clr_limit   :  6;
		uint64_t                     :  2;
		uint64_t ixe_req_clr1_limit  :  3;
		uint64_t                     :  5;
		uint64_t ixe_req_clr2_limit  :  3;
		uint64_t                     :  5;
		uint64_t ixe_req_rsp_limit   :  5;
		uint64_t                     : 35;
	};
};

#define C2_PCT_CFG_IXE_REQ_FIFO_LIMITS_OFFSET	0x00000508
#define C2_PCT_CFG_IXE_REQ_FIFO_LIMITS	(C_PCT_BASE + C2_PCT_CFG_IXE_REQ_FIFO_LIMITS_OFFSET)
#define C2_PCT_CFG_IXE_REQ_FIFO_LIMITS_SIZE	0x00000008

union c2_pct_cfg_ixe_req_fifo_limits {
	uint64_t qw;
	struct {
		uint64_t ixe_req_clr_limit   :  6;
		uint64_t                     :  2;
		uint64_t ixe_req_clr1_limit  :  3;
		uint64_t                     :  5;
		uint64_t ixe_req_clr2_limit  :  3;
		uint64_t                     :  5;
		uint64_t nack_trs_rsp_limit  :  4;
		uint64_t                     : 36;
	};
};

#define C_PCT_CFG_IXE_RSP_FIFO_LIMITS_OFFSET	0x00000528
#define C_PCT_CFG_IXE_RSP_FIFO_LIMITS	(C_PCT_BASE + C_PCT_CFG_IXE_RSP_FIFO_LIMITS_OFFSET)
#define C_PCT_CFG_IXE_RSP_FIFO_LIMITS_SIZE	0x00000008

union c_pct_cfg_ixe_rsp_fifo_limits {
	uint64_t qw;
	struct {
		uint64_t clr_limit          :  4;
		uint64_t ee_limit           :  5;
		uint64_t                    :  3;
		uint64_t cq_limit           :  5;
		uint64_t                    :  3;
		uint64_t srb_dealloc_limit  :  5;
		uint64_t                    :  3;
		uint64_t smt_cdt_ack_limit  :  4;
		uint64_t spt_cdt_ack_limit  :  4;
		uint64_t sct_cdt_ack_limit  :  4;
		uint64_t                    : 24;
	};
};

#define C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT_OFFSET	0x00000538
#define C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT	(C_PCT_BASE + C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT_OFFSET)
#define C_PCT_CFG_TGT_CLR_REQ_FIFO_LIMIT_SIZE	0x00000008

union c_pct_cfg_tgt_clr_req_fifo_limit {
	uint64_t qw;
	struct {
		uint64_t tgt_clr_req_limit       :  6;
		uint64_t                         :  2;
		uint64_t mst_dealloc_fifo_limit  :  4;
		uint64_t                         : 52;
	};
};

#define C1_PCT_CFG_TRS_TC_DED_LIMIT_OFFSET(idx)	(0x00000600+((idx)*8))
#define C1_PCT_CFG_TRS_TC_DED_LIMIT_ENTRIES	8
#define C1_PCT_CFG_TRS_TC_DED_LIMIT(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_TC_DED_LIMIT_OFFSET(idx))
#define C1_PCT_CFG_TRS_TC_DED_LIMIT_SIZE	0x00000040

union c1_pct_cfg_trs_tc_ded_limit {
	uint64_t qw;
	struct {
		uint64_t ded_limit  : 12;
		uint64_t            : 52;
	};
};

#define C2_PCT_CFG_TRS_TC_DED_LIMIT_OFFSET(idx)	(0x00000600+((idx)*8))
#define C2_PCT_CFG_TRS_TC_DED_LIMIT_ENTRIES	8
#define C2_PCT_CFG_TRS_TC_DED_LIMIT(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_TC_DED_LIMIT_OFFSET(idx))
#define C2_PCT_CFG_TRS_TC_DED_LIMIT_SIZE	0x00000040

union c2_pct_cfg_trs_tc_ded_limit {
	uint64_t qw;
	struct {
		uint64_t ded_limit  : 14;
		uint64_t            : 50;
	};
};

#define C1_PCT_CFG_TRS_TC_MAX_LIMIT_OFFSET(idx)	(0x00000640+((idx)*8))
#define C1_PCT_CFG_TRS_TC_MAX_LIMIT_ENTRIES	8
#define C1_PCT_CFG_TRS_TC_MAX_LIMIT(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_TC_MAX_LIMIT_OFFSET(idx))
#define C1_PCT_CFG_TRS_TC_MAX_LIMIT_SIZE	0x00000040

union c1_pct_cfg_trs_tc_max_limit {
	uint64_t qw;
	struct {
		uint64_t max_limit  : 12;
		uint64_t            : 52;
	};
};

#define C2_PCT_CFG_TRS_TC_MAX_LIMIT_OFFSET(idx)	(0x00000640+((idx)*8))
#define C2_PCT_CFG_TRS_TC_MAX_LIMIT_ENTRIES	8
#define C2_PCT_CFG_TRS_TC_MAX_LIMIT(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_TC_MAX_LIMIT_OFFSET(idx))
#define C2_PCT_CFG_TRS_TC_MAX_LIMIT_SIZE	0x00000040

union c2_pct_cfg_trs_tc_max_limit {
	uint64_t qw;
	struct {
		uint64_t max_limit  : 14;
		uint64_t            : 50;
	};
};

#define C1_PCT_CFG_TRS_CRDT_LIMITS_OFFSET	0x00000680
#define C1_PCT_CFG_TRS_CRDT_LIMITS	(C_PCT_BASE + C1_PCT_CFG_TRS_CRDT_LIMITS_OFFSET)
#define C1_PCT_CFG_TRS_CRDT_LIMITS_SIZE	0x00000008

union c1_pct_cfg_trs_crdt_limits {
	uint64_t qw;
	struct {
		uint64_t total_limit   : 12;
		uint64_t               :  4;
		uint64_t shared_limit  : 12;
		uint64_t               : 36;
	};
};

#define C2_PCT_CFG_TRS_CRDT_LIMITS_OFFSET	0x00000680
#define C2_PCT_CFG_TRS_CRDT_LIMITS	(C_PCT_BASE + C2_PCT_CFG_TRS_CRDT_LIMITS_OFFSET)
#define C2_PCT_CFG_TRS_CRDT_LIMITS_SIZE	0x00000008

union c2_pct_cfg_trs_crdt_limits {
	uint64_t qw;
	struct {
		uint64_t total_limit   : 14;
		uint64_t               :  2;
		uint64_t shared_limit  : 14;
		uint64_t               : 34;
	};
};

#define C_PCT_CFG_MST_TC_DED_LIMIT_OFFSET(idx)	(0x00000690+((idx)*8))
#define C_PCT_CFG_MST_TC_DED_LIMIT_ENTRIES	8
#define C_PCT_CFG_MST_TC_DED_LIMIT(idx)	(C_PCT_BASE + C_PCT_CFG_MST_TC_DED_LIMIT_OFFSET(idx))
#define C_PCT_CFG_MST_TC_DED_LIMIT_SIZE	0x00000040

union c_pct_cfg_mst_tc_ded_limit {
	uint64_t qw;
	struct {
		uint64_t ded_limit  : 12;
		uint64_t            : 52;
	};
};

#define C_PCT_CFG_MST_TC_MAX_LIMIT_OFFSET(idx)	(0x000006d0+((idx)*8))
#define C_PCT_CFG_MST_TC_MAX_LIMIT_ENTRIES	8
#define C_PCT_CFG_MST_TC_MAX_LIMIT(idx)	(C_PCT_BASE + C_PCT_CFG_MST_TC_MAX_LIMIT_OFFSET(idx))
#define C_PCT_CFG_MST_TC_MAX_LIMIT_SIZE	0x00000040

union c_pct_cfg_mst_tc_max_limit {
	uint64_t qw;
	struct {
		uint64_t max_limit  : 12;
		uint64_t            : 52;
	};
};

#define C_PCT_CFG_MST_CRDT_LIMITS_OFFSET	0x00000710
#define C_PCT_CFG_MST_CRDT_LIMITS	(C_PCT_BASE + C_PCT_CFG_MST_CRDT_LIMITS_OFFSET)
#define C_PCT_CFG_MST_CRDT_LIMITS_SIZE	0x00000008

union c_pct_cfg_mst_crdt_limits {
	uint64_t qw;
	struct {
		uint64_t total_limit   : 12;
		uint64_t               :  4;
		uint64_t shared_limit  : 12;
		uint64_t               : 36;
	};
};

#define C_PCT_CFG_TCT_TC_DED_LIMIT_OFFSET(idx)	(0x00000720+((idx)*8))
#define C_PCT_CFG_TCT_TC_DED_LIMIT_ENTRIES	8
#define C_PCT_CFG_TCT_TC_DED_LIMIT(idx)	(C_PCT_BASE + C_PCT_CFG_TCT_TC_DED_LIMIT_OFFSET(idx))
#define C_PCT_CFG_TCT_TC_DED_LIMIT_SIZE	0x00000040

union c_pct_cfg_tct_tc_ded_limit {
	uint64_t qw;
	struct {
		uint64_t ded_limit  : 14;
		uint64_t            : 50;
	};
};

#define C_PCT_CFG_TCT_TC_MAX_LIMIT_OFFSET(idx)	(0x00000760+((idx)*8))
#define C_PCT_CFG_TCT_TC_MAX_LIMIT_ENTRIES	8
#define C_PCT_CFG_TCT_TC_MAX_LIMIT(idx)	(C_PCT_BASE + C_PCT_CFG_TCT_TC_MAX_LIMIT_OFFSET(idx))
#define C_PCT_CFG_TCT_TC_MAX_LIMIT_SIZE	0x00000040

union c_pct_cfg_tct_tc_max_limit {
	uint64_t qw;
	struct {
		uint64_t max_limit  : 14;
		uint64_t            : 50;
	};
};

#define C_PCT_CFG_TCT_CRDT_LIMITS_OFFSET	0x000007a0
#define C_PCT_CFG_TCT_CRDT_LIMITS	(C_PCT_BASE + C_PCT_CFG_TCT_CRDT_LIMITS_OFFSET)
#define C_PCT_CFG_TCT_CRDT_LIMITS_SIZE	0x00000008

union c_pct_cfg_tct_crdt_limits {
	uint64_t qw;
	struct {
		uint64_t total_limit   : 14;
		uint64_t               :  2;
		uint64_t shared_limit  : 14;
		uint64_t               : 34;
	};
};

#define C_PCT_CFG_RSP_DSCP_REQ_TC_MAP_OFFSET(idx)	(0x00000800+((idx)*8))
#define C_PCT_CFG_RSP_DSCP_REQ_TC_MAP_ENTRIES	4
#define C_PCT_CFG_RSP_DSCP_REQ_TC_MAP(idx)	(C_PCT_BASE + C_PCT_CFG_RSP_DSCP_REQ_TC_MAP_OFFSET(idx))
#define C_PCT_CFG_RSP_DSCP_REQ_TC_MAP_SIZE	0x00000020

union c_pct_cfg_rsp_dscp_req_tc_map {
	uint64_t qw;
	struct {
		uint64_t dscp0_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp1_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp2_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp3_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp4_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp5_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp6_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp7_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp8_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp9_tc   :  3;
		uint64_t            :  1;
		uint64_t dscp10_tc  :  3;
		uint64_t            :  1;
		uint64_t dscp11_tc  :  3;
		uint64_t            :  1;
		uint64_t dscp12_tc  :  3;
		uint64_t            :  1;
		uint64_t dscp13_tc  :  3;
		uint64_t            :  1;
		uint64_t dscp14_tc  :  3;
		uint64_t            :  1;
		uint64_t dscp15_tc  :  3;
		uint64_t            :  1;
	};
};

#define C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_OFFSET(idx)	(0x00000820+((idx)*8))
#define C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_ENTRIES	8
#define C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP(idx)	(C_PCT_BASE + C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_OFFSET(idx))
#define C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_SIZE	0x00000040

struct c_rsp_dscp {
	uint8_t rsp_dscp    : 6;
	uint8_t             : 2;
};
#define C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_ARRAY_SIZE 8
union c_pct_cfg_req_dscp_rsp_dscp_map {
	uint64_t qw;
	struct c_rsp_dscp map[C_PCT_CFG_REQ_DSCP_RSP_DSCP_MAP_ARRAY_SIZE];
};

#define C_PCT_CFG_REQ_TC_RSP_TC_MAP_OFFSET	0x00000860
#define C_PCT_CFG_REQ_TC_RSP_TC_MAP	(C_PCT_BASE + C_PCT_CFG_REQ_TC_RSP_TC_MAP_OFFSET)
#define C_PCT_CFG_REQ_TC_RSP_TC_MAP_SIZE	0x00000008

union c_pct_cfg_req_tc_rsp_tc_map {
	uint64_t qw;
	struct {
		uint64_t req_tc0  :  3;
		uint64_t          :  1;
		uint64_t req_tc1  :  3;
		uint64_t          :  1;
		uint64_t req_tc2  :  3;
		uint64_t          :  1;
		uint64_t req_tc3  :  3;
		uint64_t          :  1;
		uint64_t req_tc4  :  3;
		uint64_t          :  1;
		uint64_t req_tc5  :  3;
		uint64_t          :  1;
		uint64_t req_tc6  :  3;
		uint64_t          :  1;
		uint64_t req_tc7  :  3;
		uint64_t          : 33;
	};
};

#define C_PCT_CFG_SCT_TAG_DFA_MASK_OFFSET(idx)	(0x00000878+((idx)*8))
#define C_PCT_CFG_SCT_TAG_DFA_MASK_ENTRIES	16
#define C_PCT_CFG_SCT_TAG_DFA_MASK(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_TAG_DFA_MASK_OFFSET(idx))
#define C_PCT_CFG_SCT_TAG_DFA_MASK_SIZE	0x00000080

union c_pct_cfg_sct_tag_dfa_mask {
	uint64_t qw;
	struct {
		uint16_t dfa_mask : 12;
		uint16_t          : 4;
	} dscp[4];
};

#define C_PCT_CFG_SW_RECYCLE_SPT_OFFSET	0x00000908
#define C_PCT_CFG_SW_RECYCLE_SPT	(C_PCT_BASE + C_PCT_CFG_SW_RECYCLE_SPT_OFFSET)
#define C_PCT_CFG_SW_RECYCLE_SPT_SIZE	0x00000008

union c_pct_cfg_sw_recycle_spt {
	uint64_t qw;
	struct {
		uint64_t spt_idx  : 11;
		uint64_t          :  5;
		uint64_t bc       :  4;
		uint64_t          : 44;
	};
};

#define C_PCT_CFG_SW_RECYCLE_SCT_OFFSET	0x00000910
#define C_PCT_CFG_SW_RECYCLE_SCT	(C_PCT_BASE + C_PCT_CFG_SW_RECYCLE_SCT_OFFSET)
#define C_PCT_CFG_SW_RECYCLE_SCT_SIZE	0x00000008

union c_pct_cfg_sw_recycle_sct {
	uint64_t qw;
	struct {
		uint64_t sct_idx  : 12;
		uint64_t          :  4;
		uint64_t bc       :  4;
		uint64_t          : 44;
	};
};

#define C_PCT_CFG_SW_SIM_SRC_RSP_OFFSET	0x00000918
#define C_PCT_CFG_SW_SIM_SRC_RSP	(C_PCT_BASE + C_PCT_CFG_SW_SIM_SRC_RSP_OFFSET)
#define C_PCT_CFG_SW_SIM_SRC_RSP_SIZE	0x00000008

union c_pct_cfg_sw_sim_src_rsp {
	uint64_t qw;
	struct {
		uint64_t spt_sct_idx    : 12;
		uint64_t                : 12;
		uint64_t return_code    :  6;
		uint64_t                :  2;
		uint64_t opcode         :  4;
		uint64_t                :  4;
		uint64_t rsp_not_gcomp  :  1;
		uint64_t                :  7;
		uint64_t loaded         :  1;
		uint64_t                : 15;
	};
};

#define C_PCT_CFG_SW_RETRY_SRC_CLS_REQ_OFFSET	0x00000920
#define C_PCT_CFG_SW_RETRY_SRC_CLS_REQ	(C_PCT_BASE + C_PCT_CFG_SW_RETRY_SRC_CLS_REQ_OFFSET)
#define C_PCT_CFG_SW_RETRY_SRC_CLS_REQ_SIZE	0x00000008

union c_pct_cfg_sw_retry_src_cls_req {
	uint64_t qw;
	struct {
		uint64_t sct_idx          : 12;
		uint64_t                  :  3;
		uint64_t close_not_clear  :  1;
		uint64_t loaded           :  1;
		uint64_t                  : 47;
	};
};

#define C_PCT_CFG_SW_SIM_TGT_CLS_REQ_OFFSET	0x00000928
#define C_PCT_CFG_SW_SIM_TGT_CLS_REQ	(C_PCT_BASE + C_PCT_CFG_SW_SIM_TGT_CLS_REQ_OFFSET)
#define C_PCT_CFG_SW_SIM_TGT_CLS_REQ_SIZE	0x00000008

union c_pct_cfg_sw_sim_tgt_cls_req {
	uint64_t qw;
	struct {
		uint64_t tct_idx  : 13;
		uint64_t          :  3;
		uint64_t loaded   :  1;
		uint64_t          : 47;
	};
};

#define C_PCT_CFG_SCT_RAM0_OFFSET(idx)	(0x00028000+((idx)*8))
#define C_PCT_CFG_SCT_RAM0_ENTRIES	4096
#define C_PCT_CFG_SCT_RAM0(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_RAM0_OFFSET(idx))
#define C_PCT_CFG_SCT_RAM0_SIZE	0x00008000

union c_pct_cfg_sct_ram0 {
	uint64_t qw;
	struct {
		uint64_t clr_tail      : 11;
		uint64_t               :  5;
		uint64_t req_pend_cnt  : 11;
		uint64_t               : 37;
	};
};

#define C_PCT_CFG_SCT_RAM1_OFFSET(idx)	(0x00030000+((idx)*8))
#define C_PCT_CFG_SCT_RAM1_ENTRIES	4096
#define C_PCT_CFG_SCT_RAM1(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_RAM1_OFFSET(idx))
#define C_PCT_CFG_SCT_RAM1_SIZE	0x00008000

union c_pct_cfg_sct_ram1 {
	uint64_t qw;
	struct {
		uint64_t put_comp_cnt  : 11;
		uint64_t               :  5;
		uint64_t mcu           :  7;
		uint64_t               :  1;
		uint64_t pcp           :  3;
		uint64_t               :  1;
		uint64_t bc            :  4;
		uint64_t               :  8;
		uint64_t sw_recycle    :  1;
		uint64_t               : 23;
	};
};

#define C_PCT_CFG_SCT_RAM2_OFFSET(idx)	(0x00040000+((idx)*8))
#define C_PCT_CFG_SCT_RAM2_ENTRIES	4096
#define C_PCT_CFG_SCT_RAM2(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_RAM2_OFFSET(idx))
#define C_PCT_CFG_SCT_RAM2_SIZE	0x00008000

union c_pct_cfg_sct_ram2 {
	uint64_t qw;
	struct {
		uint64_t            : 16;
		uint64_t clr_head   : 11;
		uint64_t            :  5;
		uint64_t req_seqno  : 12;
		uint64_t            : 20;
	};
};

#define C_PCT_CFG_SCT_RAM3_OFFSET(idx)	(0x00050000+((idx)*8))
#define C_PCT_CFG_SCT_RAM3_ENTRIES	4096
#define C_PCT_CFG_SCT_RAM3(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_RAM3_OFFSET(idx))
#define C_PCT_CFG_SCT_RAM3_SIZE	0x00008000

union c_pct_cfg_sct_ram3 {
	uint64_t qw;
	struct {
		uint64_t close_try       :  3;
		uint64_t hw_clr_cls_dis  :  1;
		uint64_t                 : 60;
	};
};

#define C_PCT_CFG_SCT_RAM4_OFFSET(idx)	(0x00060000+((idx)*8))
#define C_PCT_CFG_SCT_RAM4_ENTRIES	4096
#define C_PCT_CFG_SCT_RAM4(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_RAM4_OFFSET(idx))
#define C_PCT_CFG_SCT_RAM4_SIZE	0x00008000

union c_pct_cfg_sct_ram4 {
	uint64_t qw;
	struct {
		uint64_t get_comp_cnt  : 11;
		uint64_t               : 53;
	};
};

#define C_PCT_CFG_SCT_CAM_OFFSET(idx)	(0x00070000+((idx)*8))
#define C_PCT_CFG_SCT_CAM_ENTRIES	4096
#define C_PCT_CFG_SCT_CAM(idx)	(C_PCT_BASE + C_PCT_CFG_SCT_CAM_OFFSET(idx))
#define C_PCT_CFG_SCT_CAM_SIZE	0x00008000

union c_pct_cfg_sct_cam {
	uint64_t qw;
	struct {
		uint64_t dfa        : 32;
		uint64_t vni        : 16;
		uint64_t mcu_group  :  2;
		uint64_t            :  2;
		uint64_t dscp       :  6;
		uint64_t            :  2;
		uint64_t vld        :  1;
		uint64_t            :  3;
	};
};

#define C_PCT_CFG_SMT_RAM0_OFFSET(idx)	(0x00080000+((idx)*8))
#define C_PCT_CFG_SMT_RAM0_ENTRIES	256
#define C_PCT_CFG_SMT_RAM0(idx)	(C_PCT_BASE + C_PCT_CFG_SMT_RAM0_OFFSET(idx))
#define C_PCT_CFG_SMT_RAM0_SIZE	0x00000800

union c_pct_cfg_smt_ram0 {
	uint64_t qw;
	struct {
		uint64_t sw_retry      :  1;
		uint64_t return_code   :  6;
		uint64_t tgt_error     :  1;
		uint64_t req_comp_cnt  : 11;
		uint64_t               :  5;
		uint64_t mcu           :  7;
		uint64_t               :  1;
		uint64_t bc            :  4;
		uint64_t               :  4;
		uint64_t unrestricted  :  1;
		uint64_t               : 23;
	};
};

#define C_PCT_CFG_SMT_RAM1_OFFSET(idx)	(0x00088000+((idx)*8))
#define C_PCT_CFG_SMT_RAM1_ENTRIES	256
#define C_PCT_CFG_SMT_RAM1(idx)	(C_PCT_BASE + C_PCT_CFG_SMT_RAM1_OFFSET(idx))
#define C_PCT_CFG_SMT_RAM1_SIZE	0x00000800

union c_pct_cfg_smt_ram1 {
	uint64_t qw;
	struct {
		uint64_t last_req      :  1;
		uint64_t req_pend_cnt  : 11;
		uint64_t               : 52;
	};
};

#define C_PCT_CFG_SPT_RAM0_OFFSET(idx)	(0x000a0000+((idx)*64))
#define C_PCT_CFG_SPT_RAM0_ENTRIES	2048
#define C_PCT_CFG_SPT_RAM0(idx)	(C_PCT_BASE + C_PCT_CFG_SPT_RAM0_OFFSET(idx))
#define C_PCT_CFG_SPT_RAM0_SIZE	0x00020000

union c_pct_cfg_spt_ram0 {
	uint64_t qw[6];
	struct {
		uint64_t sw_recycle             :  1;
		uint64_t hrp                    :  1;
		uint64_t req_mp                 :  1;
		uint64_t req_order              :  1;
		uint64_t eom                    :  1;
		uint64_t                        :  3;
		uint64_t event_send_disable     :  1;
		uint64_t event_success_disable  :  1;
		uint64_t event_ct_send          :  1;
		uint64_t event_ct_reply         :  1;
		uint64_t event_ct_ack           :  1;
		uint64_t event_ct_bytes         :  1;
		uint64_t lpe_gen_rndzvs_get     :  1;
		uint64_t                        :  1;
		uint64_t mlength                : 32;
		uint64_t ptl_list               :  2;
		uint64_t                        : 14;
		uint64_t comp_data;
		uint64_t comp_cq                : 11;
		uint64_t                        :  5;
		uint64_t comp_ct                : 11;
		uint64_t                        :  5;
		uint64_t comp_eq                : 11;
		uint64_t                        : 21;
		uint64_t smt_idx                :  8;
		uint64_t                        : 56;
		uint64_t get_addr               : 57;
		uint64_t                        :  6;
		uint64_t get_with_local_flag    :  1;
		uint64_t get_atc                : 10;
		uint64_t                        :  2;
		uint64_t get_vld                :  1;
		uint64_t                        :  3;
		uint64_t pcp                    :  3;
		uint64_t                        :  1;
		uint64_t bc                     :  4;
		uint64_t mcu                    :  7;
		uint64_t                        :  1;
		uint64_t timestamp              : 20;
		uint64_t                        : 12;
	};
};

#define C_PCT_CFG_SPT_RAM1_OFFSET(idx)	(0x000c0000+((idx)*8))
#define C_PCT_CFG_SPT_RAM1_ENTRIES	2048
#define C_PCT_CFG_SPT_RAM1(idx)	(C_PCT_BASE + C_PCT_CFG_SPT_RAM1_OFFSET(idx))
#define C_PCT_CFG_SPT_RAM1_SIZE	0x00004000

union c_pct_cfg_spt_ram1 {
	uint64_t qw;
	struct {
		uint64_t clr_nxt  : 11;
		uint64_t          :  5;
		uint64_t sct_idx  : 12;
		uint64_t          : 36;
	};
};

#define C_PCT_CFG_SPT_RAM2_OFFSET(idx)	(0x000c8000+((idx)*8))
#define C_PCT_CFG_SPT_RAM2_ENTRIES	2048
#define C_PCT_CFG_SPT_RAM2(idx)	(C_PCT_BASE + C_PCT_CFG_SPT_RAM2_OFFSET(idx))
#define C_PCT_CFG_SPT_RAM2_SIZE	0x00004000

union c_pct_cfg_spt_ram2 {
	uint64_t qw;
	struct {
		uint64_t srb_ptr      : 11;
		uint64_t              :  5;
		uint64_t sct_seqno    : 12;
		uint64_t              :  4;
		uint64_t return_code  :  6;
		uint64_t              : 26;
	};
};

#define C_PCT_CFG_TCT_RAM_OFFSET(idx)	(0x000d0000+((idx)*8))
#define C_PCT_CFG_TCT_RAM_ENTRIES	8192
#define C_PCT_CFG_TCT_RAM(idx)	(C_PCT_BASE + C_PCT_CFG_TCT_RAM_OFFSET(idx))
#define C_PCT_CFG_TCT_RAM_SIZE	0x00010000

union c_pct_cfg_tct_ram {
	uint64_t qw;
	struct {
		uint64_t last_active_mst         : 11;
		uint64_t                         :  3;
		uint64_t last_active_mst_is_put  :  1;
		uint64_t last_active_mst_vld     :  1;
		uint64_t clr_seqno               : 12;
		uint64_t                         :  4;
		uint64_t exp_seqno               : 12;
		uint64_t                         :  4;
		uint64_t tc                      :  3;
		uint64_t                         : 13;
	};
};

#define C_PCT_CFG_TCT_CAM_OFFSET(idx)	(0x000e0000+((idx)*8))
#define C_PCT_CFG_TCT_CAM_ENTRIES	8192
#define C_PCT_CFG_TCT_CAM(idx)	(C_PCT_BASE + C_PCT_CFG_TCT_CAM_OFFSET(idx))
#define C_PCT_CFG_TCT_CAM_SIZE	0x00010000

union c_pct_cfg_tct_cam {
	uint64_t qw;
	struct {
		uint64_t sfa_nid  : 20;
		uint64_t          :  4;
		uint64_t sct_idx  : 12;
		uint64_t          : 24;
		uint64_t vld      :  1;
		uint64_t          :  3;
	};
};

#define C1_PCT_CFG_TRS_RAM0_OFFSET(idx)	(0x000f0000+((idx)*8))
#define C1_PCT_CFG_TRS_RAM0_ENTRIES	2048
#define C1_PCT_CFG_TRS_RAM0(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_RAM0_OFFSET(idx))
#define C1_PCT_CFG_TRS_RAM0_SIZE	0x00004000

union c_pct_cfg_trs_ram0 {
	uint64_t qw;
	struct {
		uint64_t mst_idx              : 11;
		uint64_t                      :  5;
		uint64_t put_not_get          :  1;
		uint64_t                      :  7;
		uint64_t mst_idx_dealloc_vld  :  1;
		uint64_t                      : 39;
	};
};

#define C2_PCT_CFG_TRS_RAM0_OFFSET(idx)	(0x000f0000+((idx)*8))
#define C2_PCT_CFG_TRS_RAM0_ENTRIES	8192
#define C2_PCT_CFG_TRS_RAM0(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_RAM0_OFFSET(idx))
#define C2_PCT_CFG_TRS_RAM0_SIZE	0x00010000

#define C1_PCT_CFG_TRS_RAM1_OFFSET(idx)	(0x00100000+((idx)*32))
#define C1_PCT_CFG_TRS_RAM1_ENTRIES	2048
#define C1_PCT_CFG_TRS_RAM1(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_RAM1_OFFSET(idx))
#define C1_PCT_CFG_TRS_RAM1_SIZE	0x00010000

union c_pct_cfg_trs_ram1 {
	uint64_t qw[3];
	struct {
		uint64_t data_lower;
		uint64_t data_upper;
		uint64_t famo_data_vld  :  1;
		uint64_t                :  7;
		uint64_t return_code    :  6;
		uint64_t                :  2;
		uint64_t ptl_list       :  2;
		uint64_t                :  6;
		uint64_t rsp_len        :  5;
		uint64_t                :  3;
		uint64_t pend_rsp       :  1;
		uint64_t                : 31;
	};
};

#define C2_PCT_CFG_TRS_RAM1_OFFSET(idx)	(0x00100000+((idx)*32))
#define C2_PCT_CFG_TRS_RAM1_ENTRIES	8192
#define C2_PCT_CFG_TRS_RAM1(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_RAM1_OFFSET(idx))
#define C2_PCT_CFG_TRS_RAM1_SIZE	0x00040000

#define C1_PCT_CFG_TRS_CAM_OFFSET(idx)	(0x00110000+((idx)*8))
#define C1_PCT_CFG_TRS_CAM_ENTRIES	2048
#define C1_PCT_CFG_TRS_CAM(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_CAM_OFFSET(idx))
#define C1_PCT_CFG_TRS_CAM_SIZE	0x00004000

union c_pct_cfg_trs_cam {
	uint64_t qw;
	struct {
		uint64_t req_seqno  : 12;
		uint64_t            :  4;
		uint64_t tct_idx    : 13;
		uint64_t            :  3;
		uint64_t vld        :  1;
		uint64_t            : 31;
	};
};

#define C2_PCT_CFG_TRS_CAM_OFFSET(idx)	(0x00180000+((idx)*8))
#define C2_PCT_CFG_TRS_CAM_ENTRIES	8192
#define C2_PCT_CFG_TRS_CAM(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_CAM_OFFSET(idx))
#define C2_PCT_CFG_TRS_CAM_SIZE	0x00010000

#define C1_PCT_CFG_MST_CAM_OFFSET(idx)	(0x00114000+((idx)*8))
#define C_PCT_CFG_MST_CAM_ENTRIES	2048
#define C1_PCT_CFG_MST_CAM_ENTRIES	C_PCT_CFG_MST_CAM_ENTRIES
#define C1_PCT_CFG_MST_CAM(idx)	(C_PCT_BASE + C1_PCT_CFG_MST_CAM_OFFSET(idx))
#define C_PCT_CFG_MST_CAM_SIZE	0x00004000
#define C1_PCT_CFG_MST_CAM_SIZE	C_PCT_CFG_MST_CAM_SIZE

union c_pct_cfg_mst_cam {
	uint64_t qw;
	struct {
		uint64_t msg_id     : 11;
		uint64_t            :  5;
		uint64_t multi_pkt  :  1;
		uint64_t            :  7;
		uint64_t sfa_nid    : 20;
		uint64_t vld        :  1;
		uint64_t            : 19;
	};
};

#define C2_PCT_CFG_MST_CAM_OFFSET(idx)	(0x00190000+((idx)*8))
#define C2_PCT_CFG_MST_CAM_ENTRIES	C_PCT_CFG_MST_CAM_ENTRIES
#define C2_PCT_CFG_MST_CAM(idx)	(C_PCT_BASE + C2_PCT_CFG_MST_CAM_OFFSET(idx))
#define C2_PCT_CFG_MST_CAM_SIZE	C_PCT_CFG_MST_CAM_SIZE

#define C1_PCT_CFG_SPT_MISC_INFO_OFFSET(idx)	(0x00118000+((idx)*8))
#define C_PCT_CFG_SPT_MISC_INFO_ENTRIES	2048
#define C1_PCT_CFG_SPT_MISC_INFO_ENTRIES	C_PCT_CFG_SPT_MISC_INFO_ENTRIES
#define C1_PCT_CFG_SPT_MISC_INFO(idx)	(C_PCT_BASE + C1_PCT_CFG_SPT_MISC_INFO_OFFSET(idx))
#define C_PCT_CFG_SPT_MISC_INFO_SIZE	0x00004000
#define C1_PCT_CFG_SPT_MISC_INFO_SIZE	C_PCT_CFG_SPT_MISC_INFO_SIZE

union c_pct_cfg_spt_misc_info {
	uint64_t qw;
	struct {
		uint64_t start_epoch  :  2;
		uint64_t              :  6;
		uint64_t to_flag      :  1;
		uint64_t              :  7;
		uint64_t req_try      :  3;
		uint64_t              :  5;
		uint64_t rsp_status   :  2;
		uint64_t              :  6;
		uint64_t sw_retry     :  1;
		uint64_t              :  7;
		uint64_t pkt_sent     :  1;
		uint64_t              :  7;
		uint64_t vld          :  1;
		uint64_t              : 15;
	};
};

#define C2_PCT_CFG_SPT_MISC_INFO_OFFSET(idx)	(0x00194000+((idx)*8))
#define C2_PCT_CFG_SPT_MISC_INFO_ENTRIES	C_PCT_CFG_SPT_MISC_INFO_ENTRIES
#define C2_PCT_CFG_SPT_MISC_INFO(idx)	(C_PCT_BASE + C2_PCT_CFG_SPT_MISC_INFO_OFFSET(idx))
#define C2_PCT_CFG_SPT_MISC_INFO_SIZE	C_PCT_CFG_SPT_MISC_INFO_SIZE

#define C1_PCT_CFG_SCT_MISC_INFO_OFFSET(idx)	(0x00120000+((idx)*8))
#define C_PCT_CFG_SCT_MISC_INFO_ENTRIES	4096
#define C1_PCT_CFG_SCT_MISC_INFO_ENTRIES	C_PCT_CFG_SCT_MISC_INFO_ENTRIES
#define C1_PCT_CFG_SCT_MISC_INFO(idx)	(C_PCT_BASE + C1_PCT_CFG_SCT_MISC_INFO_OFFSET(idx))
#define C_PCT_CFG_SCT_MISC_INFO_SIZE	0x00008000
#define C1_PCT_CFG_SCT_MISC_INFO_SIZE	C_PCT_CFG_SCT_MISC_INFO_SIZE

union c_pct_cfg_sct_misc_info {
	uint64_t qw;
	struct {
		uint64_t req_redo_vld       :  1;
		uint64_t                    :  7;
		uint64_t deny_new_msg       :  1;
		uint64_t                    :  7;
		uint64_t at_msg_boundary    :  1;
		uint64_t                    :  7;
		uint64_t clr_empty          :  1;
		uint64_t                    :  7;
		uint64_t last_active_epoch  :  2;
		uint64_t                    :  6;
		uint64_t sct_status         :  3;
		uint64_t                    : 21;
	};
};

#define C2_PCT_CFG_SCT_MISC_INFO_OFFSET(idx)	(0x00198000+((idx)*8))
#define C2_PCT_CFG_SCT_MISC_INFO_ENTRIES	C_PCT_CFG_SCT_MISC_INFO_ENTRIES
#define C2_PCT_CFG_SCT_MISC_INFO(idx)	(C_PCT_BASE + C2_PCT_CFG_SCT_MISC_INFO_OFFSET(idx))
#define C2_PCT_CFG_SCT_MISC_INFO_SIZE	C_PCT_CFG_SCT_MISC_INFO_SIZE

#define C1_PCT_CFG_TCT_MISC_INFO_OFFSET(idx)	(0x00130000+((idx)*8))
#define C1_PCT_CFG_TCT_MISC_INFO_ENTRIES	8192
#define C1_PCT_CFG_TCT_MISC_INFO(idx)	(C_PCT_BASE + C1_PCT_CFG_TCT_MISC_INFO_OFFSET(idx))
#define C1_PCT_CFG_TCT_MISC_INFO_SIZE	0x00010000

union c1_pct_cfg_tct_misc_info {
	uint64_t qw;
	struct {
		uint64_t last_active_epoch  :  2;
		uint64_t                    :  6;
		uint64_t to_flag            :  1;
		uint64_t                    :  7;
		uint64_t vld                :  1;
		uint64_t                    : 47;
	};
};

#define C2_PCT_CFG_TCT_MISC_INFO_OFFSET(idx)	(0x001a0000+((idx)*8))
#define C2_PCT_CFG_TCT_MISC_INFO_ENTRIES	8192
#define C2_PCT_CFG_TCT_MISC_INFO(idx)	(C_PCT_BASE + C2_PCT_CFG_TCT_MISC_INFO_OFFSET(idx))
#define C2_PCT_CFG_TCT_MISC_INFO_SIZE	0x00010000

union c2_pct_cfg_tct_misc_info {
	uint64_t qw;
	struct {
		uint64_t last_active_epoch  :  2;
		uint64_t                    :  6;
		uint64_t to_flag            :  1;
		uint64_t                    :  3;
		uint64_t closing            :  1;
		uint64_t                    :  3;
		uint64_t vld                :  1;
		uint64_t                    : 47;
	};
};

#define C1_PCT_CFG_SCT_INVALIDATE_OFFSET(idx)	(0x00140000+((idx)*8))
#define C_PCT_CFG_SCT_INVALIDATE_ENTRIES	4096
#define C1_PCT_CFG_SCT_INVALIDATE_ENTRIES	C_PCT_CFG_SCT_INVALIDATE_ENTRIES
#define C1_PCT_CFG_SCT_INVALIDATE(idx)	(C_PCT_BASE + C1_PCT_CFG_SCT_INVALIDATE_OFFSET(idx))
#define C_PCT_CFG_SCT_INVALIDATE_SIZE	0x00008000
#define C1_PCT_CFG_SCT_INVALIDATE_SIZE	C_PCT_CFG_SCT_INVALIDATE_SIZE

union c_pct_cfg_sct_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C2_PCT_CFG_SCT_INVALIDATE_OFFSET(idx)	(0x001d0000+((idx)*8))
#define C2_PCT_CFG_SCT_INVALIDATE_ENTRIES	C_PCT_CFG_SCT_INVALIDATE_ENTRIES
#define C2_PCT_CFG_SCT_INVALIDATE(idx)	(C_PCT_BASE + C2_PCT_CFG_SCT_INVALIDATE_OFFSET(idx))
#define C2_PCT_CFG_SCT_INVALIDATE_SIZE	C_PCT_CFG_SCT_INVALIDATE_SIZE

#define C1_PCT_CFG_TCT_INVALIDATE_OFFSET(idx)	(0x00150000+((idx)*8))
#define C_PCT_CFG_TCT_INVALIDATE_ENTRIES	8192
#define C1_PCT_CFG_TCT_INVALIDATE_ENTRIES	C_PCT_CFG_TCT_INVALIDATE_ENTRIES
#define C1_PCT_CFG_TCT_INVALIDATE(idx)	(C_PCT_BASE + C1_PCT_CFG_TCT_INVALIDATE_OFFSET(idx))
#define C_PCT_CFG_TCT_INVALIDATE_SIZE	0x00010000
#define C1_PCT_CFG_TCT_INVALIDATE_SIZE	C_PCT_CFG_TCT_INVALIDATE_SIZE

union c_pct_cfg_tct_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C2_PCT_CFG_TCT_INVALIDATE_OFFSET(idx)	(0x001b0000+((idx)*8))
#define C2_PCT_CFG_TCT_INVALIDATE_ENTRIES	C_PCT_CFG_TCT_INVALIDATE_ENTRIES
#define C2_PCT_CFG_TCT_INVALIDATE(idx)	(C_PCT_BASE + C2_PCT_CFG_TCT_INVALIDATE_OFFSET(idx))
#define C2_PCT_CFG_TCT_INVALIDATE_SIZE	C_PCT_CFG_TCT_INVALIDATE_SIZE

#define C1_PCT_CFG_TRS_INVALIDATE_OFFSET(idx)	(0x00168000+((idx)*8))
#define C1_PCT_CFG_TRS_INVALIDATE_ENTRIES	2048
#define C1_PCT_CFG_TRS_INVALIDATE(idx)	(C_PCT_BASE + C1_PCT_CFG_TRS_INVALIDATE_OFFSET(idx))
#define C1_PCT_CFG_TRS_INVALIDATE_SIZE	0x00004000

union c_pct_cfg_trs_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C2_PCT_CFG_TRS_INVALIDATE_OFFSET(idx)	(0x001c0000+((idx)*8))
#define C2_PCT_CFG_TRS_INVALIDATE_ENTRIES	8192
#define C2_PCT_CFG_TRS_INVALIDATE(idx)	(C_PCT_BASE + C2_PCT_CFG_TRS_INVALIDATE_OFFSET(idx))
#define C2_PCT_CFG_TRS_INVALIDATE_SIZE	0x00010000

#define C1_PCT_CFG_MST_INVALIDATE_OFFSET(idx)	(0x00170000+((idx)*8))
#define C_PCT_CFG_MST_INVALIDATE_ENTRIES	2048
#define C1_PCT_CFG_MST_INVALIDATE_ENTRIES	C_PCT_CFG_MST_INVALIDATE_ENTRIES
#define C1_PCT_CFG_MST_INVALIDATE(idx)	(C_PCT_BASE + C1_PCT_CFG_MST_INVALIDATE_OFFSET(idx))
#define C_PCT_CFG_MST_INVALIDATE_SIZE	0x00004000
#define C1_PCT_CFG_MST_INVALIDATE_SIZE	C_PCT_CFG_MST_INVALIDATE_SIZE

union c_pct_cfg_mst_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C2_PCT_CFG_MST_INVALIDATE_OFFSET(idx)	(0x001d8000+((idx)*8))
#define C2_PCT_CFG_MST_INVALIDATE_ENTRIES	C_PCT_CFG_MST_INVALIDATE_ENTRIES
#define C2_PCT_CFG_MST_INVALIDATE(idx)	(C_PCT_BASE + C2_PCT_CFG_MST_INVALIDATE_OFFSET(idx))
#define C2_PCT_CFG_MST_INVALIDATE_SIZE	C_PCT_CFG_MST_INVALIDATE_SIZE

#define C1_PCT_CFG_SRB_LINK_LIST_OFFSET(idx)	(0x00178000+((idx)*8))
#define C_PCT_CFG_SRB_LINK_LIST_ENTRIES	2048
#define C1_PCT_CFG_SRB_LINK_LIST_ENTRIES	C_PCT_CFG_SRB_LINK_LIST_ENTRIES
#define C1_PCT_CFG_SRB_LINK_LIST(idx)	(C_PCT_BASE + C1_PCT_CFG_SRB_LINK_LIST_OFFSET(idx))
#define C_PCT_CFG_SRB_LINK_LIST_SIZE	0x00004000
#define C1_PCT_CFG_SRB_LINK_LIST_SIZE	C_PCT_CFG_SRB_LINK_LIST_SIZE

union c_pct_cfg_srb_link_list {
	uint64_t qw;
	struct {
		uint64_t tail      :  1;
		uint64_t           :  7;
		uint64_t next_ptr  : 11;
		uint64_t           : 45;
	};
};

#define C2_PCT_CFG_SRB_LINK_LIST_OFFSET(idx)	(0x001e0000+((idx)*8))
#define C2_PCT_CFG_SRB_LINK_LIST_ENTRIES	C_PCT_CFG_SRB_LINK_LIST_ENTRIES
#define C2_PCT_CFG_SRB_LINK_LIST(idx)	(C_PCT_BASE + C2_PCT_CFG_SRB_LINK_LIST_OFFSET(idx))
#define C2_PCT_CFG_SRB_LINK_LIST_SIZE	C_PCT_CFG_SRB_LINK_LIST_SIZE

#define C_PCT_CFG_SRB_CELL_DATA_OFFSET(idx)	(0x00200000+((idx)*128))
#define C_PCT_CFG_SRB_CELL_DATA_ENTRIES	8192
#define C_PCT_CFG_SRB_CELL_DATA(idx)	(C_PCT_BASE + C_PCT_CFG_SRB_CELL_DATA_OFFSET(idx))
#define C_PCT_CFG_SRB_CELL_DATA_SIZE	0x00100000

union c_pct_cfg_srb_cell_data {
	uint64_t qw[9];
	struct {
		uint64_t byte_len  : 14;
		uint64_t           :  2;
		uint64_t fmt       :  1;
		uint64_t eopb      :  1;
		uint64_t eop       :  1;
		uint64_t sop       :  1;
		uint64_t           : 44;
		uint64_t word0;
		uint64_t word1;
		uint64_t word2;
		uint64_t word3;
		uint64_t word4;
		uint64_t word5;
		uint64_t word6;
		uint64_t word7;
	};
};

#define C_PCT_PRF_SCT_STATUS_OFFSET	0x00300100
#define C_PCT_PRF_SCT_STATUS	(C_PCT_BASE + C_PCT_PRF_SCT_STATUS_OFFSET)
#define C_PCT_PRF_SCT_STATUS_SIZE	0x00000008

union c_pct_prf_sct_status {
	uint64_t qw;
	struct {
		uint64_t sct_in_use      : 13;
		uint64_t                 : 19;
		uint64_t max_sct_in_use  : 13;
		uint64_t                 : 19;
	};
};

#define C_PCT_PRF_TCT_STATUS_OFFSET	0x00300108
#define C_PCT_PRF_TCT_STATUS	(C_PCT_BASE + C_PCT_PRF_TCT_STATUS_OFFSET)
#define C_PCT_PRF_TCT_STATUS_SIZE	0x00000008

union c_pct_prf_tct_status {
	uint64_t qw;
	struct {
		uint64_t tct_in_use      : 14;
		uint64_t                 : 18;
		uint64_t max_tct_in_use  : 14;
		uint64_t                 : 18;
	};
};

#define C_PCT_PRF_SPT_STATUS_OFFSET	0x00300110
#define C_PCT_PRF_SPT_STATUS	(C_PCT_BASE + C_PCT_PRF_SPT_STATUS_OFFSET)
#define C_PCT_PRF_SPT_STATUS_SIZE	0x00000008

union c_pct_prf_spt_status {
	uint64_t qw;
	struct {
		uint64_t spt_in_use      : 12;
		uint64_t                 : 20;
		uint64_t max_spt_in_use  : 12;
		uint64_t                 : 20;
	};
};

#define C_PCT_PRF_SMT_STATUS_OFFSET	0x00300118
#define C_PCT_PRF_SMT_STATUS	(C_PCT_BASE + C_PCT_PRF_SMT_STATUS_OFFSET)
#define C_PCT_PRF_SMT_STATUS_SIZE	0x00000008

union c_pct_prf_smt_status {
	uint64_t qw;
	struct {
		uint64_t smt_in_use      :  9;
		uint64_t                 : 23;
		uint64_t max_smt_in_use  :  9;
		uint64_t                 : 23;
	};
};

#define C1_PCT_PRF_TRS_STATUS_OFFSET	0x00300120
#define C1_PCT_PRF_TRS_STATUS	(C_PCT_BASE + C1_PCT_PRF_TRS_STATUS_OFFSET)
#define C1_PCT_PRF_TRS_STATUS_SIZE	0x00000008

union c1_pct_prf_trs_status {
	uint64_t qw;
	struct {
		uint64_t trs_in_use      : 12;
		uint64_t                 : 20;
		uint64_t max_trs_in_use  : 12;
		uint64_t                 : 20;
	};
};

#define C2_PCT_PRF_TRS_STATUS_OFFSET	0x00300120
#define C2_PCT_PRF_TRS_STATUS	(C_PCT_BASE + C2_PCT_PRF_TRS_STATUS_OFFSET)
#define C2_PCT_PRF_TRS_STATUS_SIZE	0x00000008

union c2_pct_prf_trs_status {
	uint64_t qw;
	struct {
		uint64_t trs_in_use      : 14;
		uint64_t                 : 18;
		uint64_t max_trs_in_use  : 14;
		uint64_t                 : 18;
	};
};

#define C_PCT_PRF_SRB_STATUS_OFFSET	0x00300128
#define C_PCT_PRF_SRB_STATUS	(C_PCT_BASE + C_PCT_PRF_SRB_STATUS_OFFSET)
#define C_PCT_PRF_SRB_STATUS_SIZE	0x00000008

union c_pct_prf_srb_status {
	uint64_t qw;
	struct {
		uint64_t srb_in_use      : 12;
		uint64_t                 : 20;
		uint64_t max_srb_in_use  : 12;
		uint64_t                 : 20;
	};
};

#define C_PCT_PRF_MST_STATUS_OFFSET	0x00300130
#define C_PCT_PRF_MST_STATUS	(C_PCT_BASE + C_PCT_PRF_MST_STATUS_OFFSET)
#define C_PCT_PRF_MST_STATUS_SIZE	0x00000008

union c_pct_prf_mst_status {
	uint64_t qw;
	struct {
		uint64_t mst_in_use      : 12;
		uint64_t                 : 20;
		uint64_t max_mst_in_use  : 12;
		uint64_t                 : 20;
	};
};

#define C_PI_CFG_CRNC_OFFSET	0x00000000
#define C_PI_CFG_CRNC	(C_PI_BASE + C_PI_CFG_CRNC_OFFSET)
#define C_PI_CFG_CRNC_SIZE	0x00000008

union c_pi_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_PI_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_PI_CFG_RT_OFFSET	(C_PI_BASE + C_PI_CFG_RT_OFFSET_OFFSET)
#define C_PI_CFG_RT_OFFSET_SIZE	0x00000008

union c_pi_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_PI_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_PI_MSC_SHADOW_ACTION	(C_PI_BASE + C_PI_MSC_SHADOW_ACTION_OFFSET)
#define C_PI_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_PI_MSC_SHADOW_OFFSET	0x00000040
#define C_PI_MSC_SHADOW	(C_PI_BASE + C_PI_MSC_SHADOW_OFFSET)
#define C_PI_MSC_SHADOW_SIZE	0x00000010

union c_pi_msc_shadow {
	uint64_t qw[2];
};

#define C_PI_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_PI_ERR_ELAPSED_TIME	(C_PI_BASE + C_PI_ERR_ELAPSED_TIME_OFFSET)
#define C_PI_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_pi_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_PI_ERR_FLG_OFFSET	0x00000108
#define C_PI_ERR_FLG	(C_PI_BASE + C_PI_ERR_FLG_OFFSET)
#define C_PI_ERR_FLG_SIZE	0x00000008

union c_pi_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                         :  1;
		uint64_t crnc_ring_sync_error            :  1;
		uint64_t crnc_ring_ecc_sbe               :  1;
		uint64_t crnc_ring_ecc_mbe               :  1;
		uint64_t crnc_ring_cmd_unknown           :  1;
		uint64_t crnc_csr_cmd_unknown            :  1;
		uint64_t crnc_csr_cmd_incomplete         :  1;
		uint64_t crnc_buf_ecc_sbe                :  1;
		uint64_t crnc_buf_ecc_mbe                :  1;
		uint64_t                                 :  7;
		uint64_t acxt_cor                        :  1;
		uint64_t acxt_ucor                       :  1;
		uint64_t sriovt_cor                      :  1;
		uint64_t sriovt_ucor                     :  1;
		uint64_t dmac_desc_cor                   :  1;
		uint64_t dmac_desc_ucor                  :  1;
		uint64_t pri_mb_xlat_rtrgt_fifo_cor      :  4;
		uint64_t pri_mb_xlat_rtrgt_fifo_ucor     :  4;
		uint64_t pri_rarb_xlat_rtrgt_fifo_cor    :  4;
		uint64_t pri_rarb_xlat_rtrgt_fifo_ucor   :  4;
		uint64_t pri_rarb_xlat_rbyp_fifo_cor     :  4;
		uint64_t pri_rarb_xlat_rbyp_fifo_ucor    :  4;
		uint64_t pti_tarb_xlat_pipe0_hdr_cor     :  1;
		uint64_t pti_tarb_xlat_pipe0_hdr_ucor    :  1;
		uint64_t pti_tarb_xlat_pipe3_hdr_cor     :  1;
		uint64_t pti_tarb_xlat_pipe3_hdr_ucor    :  1;
		uint64_t pti_tarb_xlat_pipe3_data_cor    :  1;
		uint64_t pti_tarb_xlat_pipe3_data_ucor   :  1;
		uint64_t pti_tarb_xlat_unexpected_dv     :  1;
		uint64_t pti_tarb_xlat_ac_en_error       :  1;
		uint64_t pti_tarb_xlat_cfg_vf_en_error   :  1;
		uint64_t pti_tarb_xlat_cfg_vf_num_error  :  1;
		uint64_t pti_tarb_xlat_cfg_vf_bme_error  :  1;
		uint64_t pti_tarb_xlat_cfg_bme_error     :  1;
		uint64_t pti_tarb_xlat_cfg_amo_error     :  1;
		uint64_t                                 :  3;
		uint64_t uc_attention                    :  2;
	};
};

#define C_PI_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_PI_ERR_FIRST_FLG	(C_PI_BASE + C_PI_ERR_FIRST_FLG_OFFSET)
#define C_PI_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PI_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_PI_ERR_FIRST_FLG_TS	(C_PI_BASE + C_PI_ERR_FIRST_FLG_TS_OFFSET)
#define C_PI_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PI_ERR_CLR_OFFSET	0x00000120
#define C_PI_ERR_CLR	(C_PI_BASE + C_PI_ERR_CLR_OFFSET)
#define C_PI_ERR_CLR_SIZE	0x00000008
#define C_PI_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_PI_ERR_IRQA_MSK	(C_PI_BASE + C_PI_ERR_IRQA_MSK_OFFSET)
#define C_PI_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PI_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_PI_ERR_IRQB_MSK	(C_PI_BASE + C_PI_ERR_IRQB_MSK_OFFSET)
#define C_PI_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PI_ERR_INFO_MSK_OFFSET	0x00000140
#define C_PI_ERR_INFO_MSK	(C_PI_BASE + C_PI_ERR_INFO_MSK_OFFSET)
#define C_PI_ERR_INFO_MSK_SIZE	0x00000008
#define C_PI_ERR_INFO_MEM_OFFSET	0x00000180
#define C_PI_ERR_INFO_MEM	(C_PI_BASE + C_PI_ERR_INFO_MEM_OFFSET)
#define C_PI_ERR_INFO_MEM_SIZE	0x00000008

union c_pi_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 11;
		uint64_t                    :  1;
		uint64_t cor_address        : 13;
		uint64_t                    :  1;
		uint64_t cor_mem_id         :  5;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 11;
		uint64_t                    :  1;
		uint64_t ucor_address       : 13;
		uint64_t                    :  1;
		uint64_t ucor_mem_id        :  5;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_PI_ERR_INFO_TARB_OFFSET	0x00000190
#define C_PI_ERR_INFO_TARB	(C_PI_BASE + C_PI_ERR_INFO_TARB_OFFSET)
#define C_PI_ERR_INFO_TARB_SIZE	0x00000010

union c_pi_err_info_tarb {
	uint64_t qw[2];
	struct {
		uint64_t msg_type   :  4;
		uint64_t acid       : 10;
		uint64_t msg_49_0   : 50;
		uint64_t msg_84_50  : 35;
		uint64_t source     :  3;
		uint64_t            : 26;
	};
};

#define C_PI_EXT_ERR_FLG_OFFSET	0x00000208
#define C_PI_EXT_ERR_FLG	(C_PI_BASE + C_PI_EXT_ERR_FLG_OFFSET)
#define C_PI_EXT_ERR_FLG_SIZE	0x00000008

union c_pi_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                                     :  1;
		uint64_t pri_mb_xlat_rtrgt_abort_error               :  1;
		uint64_t pri_mb_xlat_rtrgt_hdr_parity_error          :  1;
		uint64_t pri_mb_xlat_rtrgt_data_parity_error         :  1;
		uint64_t pri_mb_xlat_axi_wr_decerr                   :  1;
		uint64_t pri_mb_xlat_axi_wr_slverr                   :  1;
		uint64_t pri_mb_xlat_axi_rd_decerr                   :  1;
		uint64_t pri_mb_xlat_axi_rd_slverr                   :  1;
		uint64_t pri_mb_xlat_axi_rd_parity_error             :  1;
		uint64_t pri_mb_xlat_msg_tlp_discard                 :  1;
		uint64_t pri_mb_xlat_unexpected_np_tlp               :  1;
		uint64_t pri_mb_xlat_unexpected_tlp                  :  1;
		uint64_t pri_mb_xlat_pf_addr_error                   :  1;
		uint64_t pri_mb_xlat_vf_addr_error                   :  1;
		uint64_t pri_mb_xlat_rom_wr_error                    :  1;
		uint64_t pri_mb_xlat_partial_discard                 :  1;
		uint64_t pri_mb_xlat_rtrgt_fifo_underrun             :  1;
		uint64_t pri_mb_xlat_rtrgt_rdy_fifo_underrun         :  1;
		uint64_t pri_mb_xlat_comp_fifo_overrun               :  1;
		uint64_t pri_mb_xlat_comp_ca                         :  1;
		uint64_t pri_mb_xlat_comp_ur                         :  1;
		uint64_t pri_rarb_xlat_rtrgt_abort_error             :  1;
		uint64_t pri_rarb_xlat_rtrgt_hdr_parity_error        :  1;
		uint64_t pri_rarb_xlat_rtrgt_data_parity_error       :  1;
		uint64_t pri_rarb_xlat_rtrgt_unexpected_tlp          :  1;
		uint64_t pri_rarb_xlat_rtrgt_unexpected_msg          :  1;
		uint64_t pri_rarb_xlat_rtrgt_vf_addr_error           :  1;
		uint64_t pri_rarb_xlat_rtrgt_vf_en_num_error         :  1;
		uint64_t pri_rarb_xlat_rtrgt_vf_en_error             :  1;
		uint64_t pri_rarb_xlat_rtrgt_inv_req_vf_error        :  1;
		uint64_t pri_rarb_xlat_rtrgt_inv_req_dev_id_error    :  1;
		uint64_t pri_rarb_xlat_rtrgt_prgr_vf_error           :  1;
		uint64_t pri_rarb_xlat_rtrgt_prgr_dev_id_error       :  1;
		uint64_t pri_rarb_xlat_rtrgt_fifo_underrun           :  1;
		uint64_t pri_rarb_xlat_rtrgt_rdy_fifo_underrun       :  1;
		uint64_t pri_rarb_xlat_rbyp_abort_error              :  1;
		uint64_t pri_rarb_xlat_rbyp_hdr_parity_error         :  1;
		uint64_t pri_rarb_xlat_rbyp_data_parity_error        :  1;
		uint64_t pri_rarb_xlat_rbyp_tag_error                :  1;
		uint64_t pri_rarb_xlat_rbyp_unexpected_tlp_error     :  1;
		uint64_t pri_rarb_xlat_rbyp_fifo_underrun            :  1;
		uint64_t pri_rarb_xlat_rbyp_rdy_fifo_underrun        :  1;
		uint64_t pti_tarb_xlat_p_fifo_overrun                :  1;
		uint64_t pti_tarb_xlat_np_fifo_overrun               :  1;
		uint64_t pti_tarb_xlat_ord_fifo_overrun              :  1;
		uint64_t pti_tarb_xlat_np_err_fifo_overrun           :  1;
		uint64_t pti_tarb_xlat_np_err_fifo_underrun          :  1;
		uint64_t pti_tarb_xlat_pti_pri_cpl_to_fifo_underrun  :  1;
		uint64_t dmac_axi_rd_decerr                          :  1;
		uint64_t dmac_axi_rd_slverr                          :  1;
		uint64_t dmac_axi_rd_parity_error                    :  1;
		uint64_t ahb_overwrite                               :  1;
		uint64_t ahb_resp_err                                :  1;
		uint64_t                                             : 11;
	};
};

#define C_PI_EXT_ERR_FIRST_FLG_OFFSET	0x00000210
#define C_PI_EXT_ERR_FIRST_FLG	(C_PI_BASE + C_PI_EXT_ERR_FIRST_FLG_OFFSET)
#define C_PI_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PI_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000218
#define C_PI_EXT_ERR_FIRST_FLG_TS	(C_PI_BASE + C_PI_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_PI_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PI_EXT_ERR_CLR_OFFSET	0x00000220
#define C_PI_EXT_ERR_CLR	(C_PI_BASE + C_PI_EXT_ERR_CLR_OFFSET)
#define C_PI_EXT_ERR_CLR_SIZE	0x00000008
#define C_PI_EXT_ERR_IRQA_MSK_OFFSET	0x00000228
#define C_PI_EXT_ERR_IRQA_MSK	(C_PI_BASE + C_PI_EXT_ERR_IRQA_MSK_OFFSET)
#define C_PI_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PI_EXT_ERR_IRQB_MSK_OFFSET	0x00000230
#define C_PI_EXT_ERR_IRQB_MSK	(C_PI_BASE + C_PI_EXT_ERR_IRQB_MSK_OFFSET)
#define C_PI_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PI_EXT_ERR_INFO_MSK_OFFSET	0x00000240
#define C_PI_EXT_ERR_INFO_MSK	(C_PI_BASE + C_PI_EXT_ERR_INFO_MSK_OFFSET)
#define C_PI_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_PI_ERR_INFO_RTRGT_OFFSET	0x00000250
#define C_PI_ERR_INFO_RTRGT	(C_PI_BASE + C_PI_ERR_INFO_RTRGT_OFFSET)
#define C_PI_ERR_INFO_RTRGT_SIZE	0x00000018

union c_pi_err_info_rtrgt {
	uint64_t qw[3];
	struct {
		uint64_t addr;
		uint64_t tlp_type      :  5;
		uint64_t fmt           :  2;
		uint64_t tc            :  3;
		uint64_t attr          :  3;
		uint64_t td            :  1;
		uint64_t th            :  1;
		uint64_t poisoned      :  1;
		uint64_t reqid         : 16;
		uint64_t tag           : 10;
		uint64_t dw_len        : 10;
		uint64_t first_be      :  4;
		uint64_t last_be       :  4;
		uint64_t ats           :  2;
		uint64_t ph            :  2;
		uint64_t st            :  8;
		uint64_t vfunc_num     :  6;
		uint64_t vfunc_active  :  1;
		uint64_t nw            :  1;
		uint64_t ecrc_err      :  1;
		uint64_t dllp_abort    :  1;
		uint64_t tlp_abort     :  1;
		uint64_t               : 13;
		uint64_t tlp_prfx      : 32;
	};
};

#define C_PI_ERR_INFO_RBYP_OFFSET	0x00000280
#define C_PI_ERR_INFO_RBYP	(C_PI_BASE + C_PI_ERR_INFO_RBYP_OFFSET)
#define C_PI_ERR_INFO_RBYP_SIZE	0x00000018

union c_pi_err_info_rbyp {
	uint64_t qw[3];
	struct {
		uint64_t addr          :  7;
		uint64_t               :  9;
		uint64_t ats           :  2;
		uint64_t ph            :  2;
		uint64_t               : 12;
		uint64_t byte_cnt      : 12;
		uint64_t               :  4;
		uint64_t completer_id  : 16;
		uint64_t tlp_type      :  5;
		uint64_t fmt           :  2;
		uint64_t tc            :  3;
		uint64_t attr          :  3;
		uint64_t td            :  1;
		uint64_t th            :  1;
		uint64_t poisoned      :  1;
		uint64_t reqid         : 16;
		uint64_t tag           : 10;
		uint64_t dw_len        : 10;
		uint64_t cpl_status    :  3;
		uint64_t cpl_last      :  1;
		uint64_t bcm           :  1;
		uint64_t ecrc_err      :  1;
		uint64_t dllp_abort    :  1;
		uint64_t tlp_abort     :  1;
		uint64_t               :  4;
		uint64_t tlp_prfx      : 32;
		uint64_t               : 32;
	};
};

#define C_PI_ERR_INFO_DMAC_OFFSET	0x000002a0
#define C_PI_ERR_INFO_DMAC	(C_PI_BASE + C_PI_ERR_INFO_DMAC_OFFSET)
#define C_PI_ERR_INFO_DMAC_SIZE	0x00000008

union c_pi_err_info_dmac {
	uint64_t qw;
	struct {
		uint64_t desc_addr  : 10;
		uint64_t            : 54;
	};
};

#define C_PI_CFG_PRI_SRIOV_OFFSET	0x00001100
#define C_PI_CFG_PRI_SRIOV	(C_PI_BASE + C_PI_CFG_PRI_SRIOV_OFFSET)
#define C_PI_CFG_PRI_SRIOV_SIZE	0x00000008

union c_pi_cfg_pri_sriov {
	uint64_t qw;
	struct {
		uint64_t vf_offset  : 16;
		uint64_t vf_stride  : 16;
		uint64_t            : 32;
	};
};

#define C_PI_CFG_DMAC_CPL_ADDR_OFFSET	0x00002000
#define C_PI_CFG_DMAC_CPL_ADDR	(C_PI_BASE + C_PI_CFG_DMAC_CPL_ADDR_OFFSET)
#define C_PI_CFG_DMAC_CPL_ADDR_SIZE	0x00000008

union c_pi_cfg_dmac_cpl_addr {
	uint64_t qw;
	struct {
		uint64_t          :  3;
		uint64_t address  : 61;
	};
};

#define C_PI_CFG_DMAC_OFFSET	0x00002008
#define C_PI_CFG_DMAC	(C_PI_BASE + C_PI_CFG_DMAC_OFFSET)
#define C_PI_CFG_DMAC_SIZE	0x00000008

union c_pi_cfg_dmac {
	uint64_t qw;
	struct {
		uint64_t desc_index        : 10;
		uint64_t                   :  2;
		uint64_t enable            :  1;
		uint64_t start             :  1;
		uint64_t done              :  1;
		uint64_t cpl_event_enable  :  1;
		uint64_t status            :  2;
		uint64_t irq_on_done       :  1;
		uint64_t irq_on_error      :  1;
		uint64_t                   : 44;
	};
};

#define C_PI_CFG_SRIOVT_OFFSET(idx)	(0x00010000+((idx)*8))
#define C_PI_CFG_SRIOVT_ENTRIES	6144
#define C_PI_CFG_SRIOVT(idx)	(C_PI_BASE + C_PI_CFG_SRIOVT_OFFSET(idx))
#define C_PI_CFG_SRIOVT_SIZE	0x0000c000

union c_pi_cfg_sriovt {
	uint64_t qw;
	struct {
		uint64_t vf_num  :  6;
		uint64_t vf_en   :  1;
		uint64_t         : 57;
	};
};

#define C_PI_CFG_ACXT_OFFSET(idx)	(0x00020000+((idx)*8))
#define C_PI_CFG_ACXT_ENTRIES	1024
#define C_PI_CFG_ACXT(idx)	(C_PI_BASE + C_PI_CFG_ACXT_OFFSET(idx))
#define C_PI_CFG_ACXT_SIZE	0x00002000

union c_pi_cfg_acxt {
	uint64_t qw;
	struct {
		uint64_t ac_en               :  1;
		uint64_t vf_en               :  1;
		uint64_t vf_num              :  6;
		uint64_t steering_tag        :  8;
		uint64_t ph_en               :  1;
		uint64_t phints              :  2;
		uint64_t no_snoop            :  1;
		uint64_t mem_rd_ro           :  1;
		uint64_t flush_req_ro        :  1;
		uint64_t translation_req_ro  :  1;
		uint64_t fetching_amo_ro     :  1;
		uint64_t ido                 :  1;
		uint64_t at1                 :  1;
		uint64_t pasid_en            :  1;
		uint64_t pasid               : 20;
		uint64_t pasid_rsvd          :  2;
		uint64_t pasid_er            :  1;
		uint64_t pasid_pmr           :  1;
		uint64_t reserved            : 13;
	};
};

#define C_PI_CFG_DMAC_DESC_OFFSET(idx)	(0x00030000+((idx)*16))
#define C_PI_CFG_DMAC_DESC_ENTRIES	1024
/*
 * Since there are an 'abundance' of DMAC descriptors, some of them have
 * been repurposed & reserved as 'scratch registers' for other purposes.
 *
 * Such as for communication between UC & HOST (see cass_uc.c).
 *
 * The rule is a contiguous block of descriptors from
 *
 *	I ... (C_PI_CFG_DMAC_DESC_ENTRIES - 1)
 *
 * will be the reserved pool not to be used by the DMAC for DMAing.
 */
#define C_PI_CFG_DMAC_DESC_NUM_RSVD_ENTRIES	32
#define C_PI_CFG_DMAC_DESC_NUM_AVAILABLE_ENTRIES \
	(C_PI_CFG_DMAC_DESC_ENTRIES - C_PI_CFG_DMAC_DESC_NUM_RSVD_ENTRIES)

#define C_PI_CFG_DMAC_DESC(idx)	(C_PI_BASE + C_PI_CFG_DMAC_DESC_OFFSET(idx))
#define C_PI_CFG_DMAC_DESC_SIZE	0x00004000

union c_pi_cfg_dmac_desc {
	uint64_t qw[2];
	struct {
		uint64_t reserved0  :  3;
		uint64_t dst_addr   : 61;
		uint64_t reserved1  :  3;
		uint64_t src_addr   : 27;
		uint64_t reserved2  :  2;
		uint64_t length     : 20;
		uint64_t cont       :  1;
		uint64_t reserved3  : 11;
	};
};

#define C_PI_IPD_CFG_CRNC_OFFSET	0x00000000
#define C_PI_IPD_CFG_CRNC	(C_PI_IPD_BASE + C_PI_IPD_CFG_CRNC_OFFSET)
#define C_PI_IPD_CFG_CRNC_SIZE	0x00000008

union c_pi_ipd_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_PI_IPD_CFG_RT_OFFSET_OFFSET	0x00000020
#define C_PI_IPD_CFG_RT_OFFSET	(C_PI_IPD_BASE + C_PI_IPD_CFG_RT_OFFSET_OFFSET)
#define C_PI_IPD_CFG_RT_OFFSET_SIZE	0x00000008

union c_pi_ipd_cfg_rt_offset {
	uint64_t qw;
	struct {
		uint64_t seconds  : 48;
		uint64_t          : 16;
	};
};

#define C_PI_IPD_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_PI_IPD_MSC_SHADOW_ACTION	(C_PI_IPD_BASE + C_PI_IPD_MSC_SHADOW_ACTION_OFFSET)
#define C_PI_IPD_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_PI_IPD_MSC_SHADOW_OFFSET	0x00000040
#define C_PI_IPD_MSC_SHADOW	(C_PI_IPD_BASE + C_PI_IPD_MSC_SHADOW_OFFSET)
#define C_PI_IPD_MSC_SHADOW_SIZE	0x00000028

union c_pi_ipd_msc_shadow {
	uint64_t qw[5];
};

#define C_PI_IPD_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_PI_IPD_ERR_ELAPSED_TIME	(C_PI_IPD_BASE + C_PI_IPD_ERR_ELAPSED_TIME_OFFSET)
#define C_PI_IPD_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_pi_ipd_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_PI_IPD_ERR_FLG_OFFSET	0x00000108
#define C_PI_IPD_ERR_FLG	(C_PI_IPD_BASE + C_PI_IPD_ERR_FLG_OFFSET)
#define C_PI_IPD_ERR_FLG_SIZE	0x00000008

union c_pi_ipd_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                         :  1;
		uint64_t crnc_ring_sync_error            :  1;
		uint64_t crnc_ring_ecc_sbe               :  1;
		uint64_t crnc_ring_ecc_mbe               :  1;
		uint64_t crnc_ring_cmd_unknown           :  1;
		uint64_t crnc_csr_cmd_unknown            :  1;
		uint64_t crnc_csr_cmd_incomplete         :  1;
		uint64_t crnc_buf_ecc_sbe                :  1;
		uint64_t crnc_buf_ecc_mbe                :  1;
		uint64_t                                 :  7;
		uint64_t event_cnt_ovf                   :  1;
		uint64_t event_ram_cor                   :  1;
		uint64_t event_ram_ucor                  :  1;
		uint64_t msix_table_cor                  :  1;
		uint64_t msix_table_ucor                 :  1;
		uint64_t msix_pba_cor                    :  1;
		uint64_t msix_pba_ucor                   :  1;
		uint64_t pti_tarb_xlat_p_fifo_cor        :  4;
		uint64_t pti_tarb_xlat_p_fifo_ucor       :  4;
		uint64_t pti_tarb_xlat_np_fifo_cor       :  3;
		uint64_t pti_tarb_xlat_np_fifo_ucor      :  3;
		uint64_t pti_tarb_xlat_tbuf_cor          :  1;
		uint64_t pti_tarb_xlat_tbuf_ucor         :  1;
		uint64_t pti_tarb_xlat_cpl_to_fifo_cor   :  1;
		uint64_t pti_tarb_xlat_cpl_to_fifo_ucor  :  1;
		uint64_t pri_mb_xlat_comp_fifo_cor       :  4;
		uint64_t pri_mb_xlat_comp_fifo_ucor      :  4;
		uint64_t pti_msixc_fifo_cor              :  1;
		uint64_t pti_msixc_fifo_ucor             :  1;
		uint64_t msixc_pti_fifo_cor              :  1;
		uint64_t msixc_pti_fifo_ucor             :  1;
		uint64_t dmac_p_fifo_cor                 :  4;
		uint64_t dmac_p_fifo_ucor                :  4;
		uint64_t                                 :  3;
	};
};

#define C_PI_IPD_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_PI_IPD_ERR_FIRST_FLG	(C_PI_IPD_BASE + C_PI_IPD_ERR_FIRST_FLG_OFFSET)
#define C_PI_IPD_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PI_IPD_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_PI_IPD_ERR_FIRST_FLG_TS	(C_PI_IPD_BASE + C_PI_IPD_ERR_FIRST_FLG_TS_OFFSET)
#define C_PI_IPD_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PI_IPD_ERR_CLR_OFFSET	0x00000120
#define C_PI_IPD_ERR_CLR	(C_PI_IPD_BASE + C_PI_IPD_ERR_CLR_OFFSET)
#define C_PI_IPD_ERR_CLR_SIZE	0x00000008
#define C_PI_IPD_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_PI_IPD_ERR_IRQA_MSK	(C_PI_IPD_BASE + C_PI_IPD_ERR_IRQA_MSK_OFFSET)
#define C_PI_IPD_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PI_IPD_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_PI_IPD_ERR_IRQB_MSK	(C_PI_IPD_BASE + C_PI_IPD_ERR_IRQB_MSK_OFFSET)
#define C_PI_IPD_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PI_IPD_ERR_INFO_MSK_OFFSET	0x00000140
#define C_PI_IPD_ERR_INFO_MSK	(C_PI_IPD_BASE + C_PI_IPD_ERR_INFO_MSK_OFFSET)
#define C_PI_IPD_ERR_INFO_MSK_SIZE	0x00000008
#define C_PI_IPD_ERR_INFO_MEM_OFFSET	0x00000180
#define C_PI_IPD_ERR_INFO_MEM	(C_PI_IPD_BASE + C_PI_IPD_ERR_INFO_MEM_OFFSET)
#define C_PI_IPD_ERR_INFO_MEM_SIZE	0x00000008

union c_pi_ipd_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 10;
		uint64_t                    :  2;
		uint64_t cor_address        : 12;
		uint64_t                    :  2;
		uint64_t cor_mem_id         :  5;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 10;
		uint64_t                    :  2;
		uint64_t ucor_address       : 12;
		uint64_t                    :  2;
		uint64_t ucor_mem_id        :  5;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_PI_IPD_EXT_ERR_FLG_OFFSET	0x00000208
#define C_PI_IPD_EXT_ERR_FLG	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_FLG_OFFSET)
#define C_PI_IPD_EXT_ERR_FLG_SIZE	0x00000008

union c_pi_ipd_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                                    :  1;
		uint64_t ip_app_parity_errs                         :  3;
		uint64_t ip_radm_qoverflow                          :  1;
		uint64_t ip_radm_msg_unlock                         :  1;
		uint64_t ip_radm_pm_turnoff                         :  1;
		uint64_t ip_wake                                    :  1;
		uint64_t ip_cfg_uncor_internal_err_sts              :  1;
		uint64_t ip_cfg_corrected_internal_err_sts          :  1;
		uint64_t ip_cfg_rcvr_overflow_err_sts               :  1;
		uint64_t ip_cfg_fc_protocol_err_sts                 :  1;
		uint64_t ip_cfg_mlf_tlp_err_sts                     :  1;
		uint64_t ip_cfg_surprise_down_er_sts                :  1;
		uint64_t ip_cfg_dl_protocol_err_sts                 :  1;
		uint64_t ip_cfg_ecrc_err_sts                        :  1;
		uint64_t ip_cfg_replay_number_rollover_err_sts      :  1;
		uint64_t ip_cfg_replay_timer_timeout_err_sts        :  1;
		uint64_t ip_cfg_bad_dllp_err_sts                    :  1;
		uint64_t ip_cfg_bad_tlp_err_sts                     :  1;
		uint64_t ip_cfg_rcvr_err_sts                        :  1;
		uint64_t                                            :  5;
		uint64_t pti_tarb_xlat_xali_ph_crd_warn             :  1;
		uint64_t pti_tarb_xlat_xali_pd_crd_warn             :  1;
		uint64_t pti_tarb_xlat_xali_nph_crd_warn            :  1;
		uint64_t pti_tarb_xlat_xali_npd_crd_warn            :  1;
		uint64_t pti_tarb_xlat_ten_bit_tag_msb_error        :  1;
		uint64_t pti_tarb_xlat_p_fifo_underrun              :  1;
		uint64_t pti_tarb_xlat_np_fifo_underrun             :  1;
		uint64_t pti_tarb_xlat_ord_fifo_underrun            :  1;
		uint64_t pti_tarb_xlat_cpl_to_fifo_overrun          :  1;
		uint64_t pti_tarb_xlat_cpl_to_fifo_underrun         :  1;
		uint64_t pti_tarb_xlat_pti_pri_cpl_to_fifo_overrun  :  1;
		uint64_t pti_tarb_xlat_ord_np_cnt_overrun           :  1;
		uint64_t pti_tarb_xlat_ord_np_cnt_underrun          :  1;
		uint64_t pri_rtrgt_hdr_parity_error                 :  1;
		uint64_t pri_rtrgt_data_parity_error                :  1;
		uint64_t pri_rtrgt_abort                            :  1;
		uint64_t pri_rbyp_hdr_parity_error                  :  1;
		uint64_t pri_rbyp_data_parity_error                 :  1;
		uint64_t pri_rbyp_abort                             :  1;
		uint64_t pri_rarb_xlat_rtrgt_fifo_overrun           :  1;
		uint64_t pri_rarb_xlat_rtrgt_rdy_fifo_overrun       :  1;
		uint64_t pri_rarb_xlat_rbyp_fifo_overrun            :  1;
		uint64_t pri_rarb_xlat_rbyp_rdy_fifo_overrun        :  1;
		uint64_t pri_mb_xlat_rtrgt_fifo_overrun             :  1;
		uint64_t pri_mb_xlat_rtrgt_rdy_fifo_overrun         :  1;
		uint64_t pri_mb_xlat_comp_fifo_underrun             :  1;
		uint64_t msixc_pti_fifo_overrun                     :  1;
		uint64_t msixc_pti_fifo_underrun                    :  1;
		uint64_t pti_msixc_fifo_overrun                     :  1;
		uint64_t pti_msixc_fifo_underrun                    :  1;
		uint64_t msix_disabled_error                        :  1;
		uint64_t msix_hg_vf_size_error                      :  1;
		uint64_t msix_pti_idx_error                         :  1;
		uint64_t dmac_bme_error                             :  1;
		uint64_t                                            :  4;
	};
};

#define C_PI_IPD_EXT_ERR_FIRST_FLG_OFFSET	0x00000210
#define C_PI_IPD_EXT_ERR_FIRST_FLG	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_FIRST_FLG_OFFSET)
#define C_PI_IPD_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C_PI_IPD_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000218
#define C_PI_IPD_EXT_ERR_FIRST_FLG_TS	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C_PI_IPD_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_PI_IPD_EXT_ERR_CLR_OFFSET	0x00000220
#define C_PI_IPD_EXT_ERR_CLR	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_CLR_OFFSET)
#define C_PI_IPD_EXT_ERR_CLR_SIZE	0x00000008
#define C_PI_IPD_EXT_ERR_IRQA_MSK_OFFSET	0x00000228
#define C_PI_IPD_EXT_ERR_IRQA_MSK	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_IRQA_MSK_OFFSET)
#define C_PI_IPD_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C_PI_IPD_EXT_ERR_IRQB_MSK_OFFSET	0x00000230
#define C_PI_IPD_EXT_ERR_IRQB_MSK	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_IRQB_MSK_OFFSET)
#define C_PI_IPD_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C_PI_IPD_EXT_ERR_INFO_MSK_OFFSET	0x00000240
#define C_PI_IPD_EXT_ERR_INFO_MSK	(C_PI_IPD_BASE + C_PI_IPD_EXT_ERR_INFO_MSK_OFFSET)
#define C_PI_IPD_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C_PI_IPD_CFG_MSIXC_OFFSET	0x00004000
#define C_PI_IPD_CFG_MSIXC	(C_PI_IPD_BASE + C_PI_IPD_CFG_MSIXC_OFFSET)
#define C_PI_IPD_CFG_MSIXC_SIZE	0x00000008

union c_pi_ipd_cfg_msixc {
	uint64_t qw;
	struct {
		uint64_t legacy_irq_enable    :  1;
		uint64_t legacy_irq_mask      :  1;
		uint64_t legacy_irq_asserted  :  1;
		uint64_t                      :  5;
		uint64_t vf_pf_irq_enable     :  1;
		uint64_t                      : 55;
	};
};

#define C_PI_IPD_CFG_VF_PF_IRQ_MASK_OFFSET	0x00004010
#define C_PI_IPD_CFG_VF_PF_IRQ_MASK	(C_PI_IPD_BASE + C_PI_IPD_CFG_VF_PF_IRQ_MASK_OFFSET)
#define C_PI_IPD_CFG_VF_PF_IRQ_MASK_SIZE	0x00000008

union c_pi_ipd_cfg_vf_pf_irq_mask {
	uint64_t qw;
	struct {
		uint64_t mask;
	};
};

#define C_PI_IPD_STS_VF_PF_IRQ_OFFSET	0x00004018
#define C_PI_IPD_STS_VF_PF_IRQ	(C_PI_IPD_BASE + C_PI_IPD_STS_VF_PF_IRQ_OFFSET)
#define C_PI_IPD_STS_VF_PF_IRQ_SIZE	0x00000008

union c_pi_ipd_sts_vf_pf_irq {
	uint64_t qw;
	struct {
		uint64_t irq;
	};
};

#define C_PI_IPD_CFG_VF_PF_IRQ_CLR_OFFSET	0x00004020
#define C_PI_IPD_CFG_VF_PF_IRQ_CLR	(C_PI_IPD_BASE + C_PI_IPD_CFG_VF_PF_IRQ_CLR_OFFSET)
#define C_PI_IPD_CFG_VF_PF_IRQ_CLR_SIZE	0x00000008

union c_pi_ipd_cfg_vf_pf_irq_clr {
	uint64_t qw;
	struct {
		uint64_t clr;
	};
};

#define C_PI_IPD_CFG_PF_VF_IRQ_OFFSET	0x00004028
#define C_PI_IPD_CFG_PF_VF_IRQ	(C_PI_IPD_BASE + C_PI_IPD_CFG_PF_VF_IRQ_OFFSET)
#define C_PI_IPD_CFG_PF_VF_IRQ_SIZE	0x00000008

union c_pi_ipd_cfg_pf_vf_irq {
	uint64_t qw;
	struct {
		uint64_t irq;
	};
};

#define C_PI_IPD_CFG_EVENT_CNTS_OFFSET	0x00005000
#define C_PI_IPD_CFG_EVENT_CNTS	(C_PI_IPD_BASE + C_PI_IPD_CFG_EVENT_CNTS_OFFSET)
#define C_PI_IPD_CFG_EVENT_CNTS_SIZE	0x00000008

union c_pi_ipd_cfg_event_cnts {
	uint64_t qw;
	struct {
		uint64_t enable            :  1;
		uint64_t init              :  1;
		uint64_t init_done         :  1;
		uint64_t warm_rst_disable  :  1;
		uint64_t ovf_cnt           : 16;
		uint64_t                   : 44;
	};
};

#define C_PI_IPD_STS_EVENT_CNTS_OFFSET(idx)	(0x00006000+((idx)*8))
#ifndef C_PI_IPD_STS_EVENT_CNTS_ENTRIES
#define C_PI_IPD_STS_EVENT_CNTS_ENTRIES	128
#endif
#define C_PI_IPD_STS_EVENT_CNTS(idx)	(C_PI_IPD_BASE + C_PI_IPD_STS_EVENT_CNTS_OFFSET(idx))
#define C_PI_IPD_STS_EVENT_CNTS_SIZE	0x00000400

union c_pi_ipd_sts_event_cnts {
	uint64_t qw;
	struct {
		uint64_t cnt  : 56;
		uint64_t      :  8;
	};
};

#define C_PI_IPD_CFG_MSIX_TABLE_OFFSET(idx)	(0x00010000+((idx)*16))
#define C_PI_IPD_CFG_MSIX_TABLE_ENTRIES	4096
#define C_PI_IPD_CFG_MSIX_TABLE(idx)	(C_PI_IPD_BASE + C_PI_IPD_CFG_MSIX_TABLE_OFFSET(idx))
#define C_PI_IPD_CFG_MSIX_TABLE_SIZE	0x00010000

union c_pi_ipd_cfg_msix_table {
	uint64_t qw[2];
	struct {
		uint64_t msg_addr;
		uint64_t msg_data         : 32;
		uint64_t vector_ctl_mask  :  1;
		uint64_t vector_ctl_rsvd  : 31;
	};
};

#define C_PI_IPD_CFG_VF_MSIX_TABLE_OFFSET(idx)	(0x00400000+((idx)*16))
#define C_PI_IPD_CFG_VF_MSIX_TABLE_ENTRIES	131072
#define C_PI_IPD_CFG_VF_MSIX_TABLE(idx)	(C_PI_IPD_BASE + C_PI_IPD_CFG_VF_MSIX_TABLE_OFFSET(idx))
#define C_PI_IPD_CFG_VF_MSIX_TABLE_SIZE	0x00200000

union c_pi_ipd_cfg_vf_msix_table {
	uint64_t qw[2];
	struct {
		uint64_t msg_addr;
		uint64_t msg_data         : 32;
		uint64_t vector_ctl_mask  :  1;
		uint64_t vector_ctl_rsvd  : 31;
	};
};

#define C_RMU_CFG_CRNC_OFFSET	0x00000000
#define C_RMU_CFG_CRNC	(C_RMU_BASE + C_RMU_CFG_CRNC_OFFSET)
#define C_RMU_CFG_CRNC_SIZE	0x00000008

union c_rmu_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define C_RMU_MSC_SHADOW_ACTION_OFFSET	0x00000038
#define C_RMU_MSC_SHADOW_ACTION	(C_RMU_BASE + C_RMU_MSC_SHADOW_ACTION_OFFSET)
#define C_RMU_MSC_SHADOW_ACTION_SIZE	0x00000008
#define C_RMU_MSC_SHADOW_OFFSET	0x00000040
#define C_RMU_MSC_SHADOW	(C_RMU_BASE + C_RMU_MSC_SHADOW_OFFSET)
#define C_RMU_MSC_SHADOW_SIZE	0x00000020

union c_rmu_msc_shadow {
	uint64_t qw[4];
};

#define C_RMU_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define C_RMU_ERR_ELAPSED_TIME	(C_RMU_BASE + C_RMU_ERR_ELAPSED_TIME_OFFSET)
#define C_RMU_ERR_ELAPSED_TIME_SIZE	0x00000008

union c_rmu_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define C_RMU_ERR_FLG_OFFSET	0x00000108
#define C_RMU_ERR_FLG	(C_RMU_BASE + C_RMU_ERR_FLG_OFFSET)
#define C_RMU_ERR_FLG_SIZE	0x00000008

union c_rmu_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag                      :  1;
		uint64_t crnc_ring_sync_error         :  1;
		uint64_t crnc_ring_ecc_sbe            :  1;
		uint64_t crnc_ring_ecc_mbe            :  1;
		uint64_t crnc_ring_cmd_unknown        :  1;
		uint64_t crnc_csr_cmd_unknown         :  1;
		uint64_t crnc_csr_cmd_incomplete      :  1;
		uint64_t crnc_buf_ecc_sbe             :  1;
		uint64_t crnc_buf_ecc_mbe             :  1;
		uint64_t                              : 23;
		uint64_t ptl_idx_mem_cor              :  1;
		uint64_t ptlte_set_ctrl_mem_cor       :  1;
		uint64_t ptl_idx_indir_mem_cor        :  1;
		uint64_t                              :  1;
		uint64_t vni_list_mem_ucor            :  1;
		uint64_t vni_list_vld_mem_ucor        :  1;
		uint64_t ptl_list_mem_ucor            :  1;
		uint64_t ptl_list_vld_mem_ucor        :  1;
		uint64_t ptl_idx_mem_ucor             :  1;
		uint64_t ptlte_set_list_mem_ucor      :  1;
		uint64_t ptlte_set_list_vld_mem_ucor  :  1;
		uint64_t ptlte_set_ctrl_mem_ucor      :  1;
		uint64_t ptl_idx_indir_mem_ucor       :  1;
		uint64_t                              :  3;
		uint64_t ptl_invld_dfa                :  1;
		uint64_t ptl_invld_vni                :  1;
		uint64_t ptl_no_ptlte                 :  1;
		uint64_t enet_frame_rejected          :  1;
		uint64_t enet_ptlte_set_ctrl_err      :  1;
		uint64_t                              : 11;
	};
};

#define C_RMU_ERR_FIRST_FLG_OFFSET	0x00000110
#define C_RMU_ERR_FIRST_FLG	(C_RMU_BASE + C_RMU_ERR_FIRST_FLG_OFFSET)
#define C_RMU_ERR_FIRST_FLG_SIZE	0x00000008
#define C_RMU_ERR_FIRST_FLG_TS_OFFSET	0x00000118
#define C_RMU_ERR_FIRST_FLG_TS	(C_RMU_BASE + C_RMU_ERR_FIRST_FLG_TS_OFFSET)
#define C_RMU_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C_RMU_ERR_CLR_OFFSET	0x00000120
#define C_RMU_ERR_CLR	(C_RMU_BASE + C_RMU_ERR_CLR_OFFSET)
#define C_RMU_ERR_CLR_SIZE	0x00000008
#define C_RMU_ERR_IRQA_MSK_OFFSET	0x00000128
#define C_RMU_ERR_IRQA_MSK	(C_RMU_BASE + C_RMU_ERR_IRQA_MSK_OFFSET)
#define C_RMU_ERR_IRQA_MSK_SIZE	0x00000008
#define C_RMU_ERR_IRQB_MSK_OFFSET	0x00000130
#define C_RMU_ERR_IRQB_MSK	(C_RMU_BASE + C_RMU_ERR_IRQB_MSK_OFFSET)
#define C_RMU_ERR_IRQB_MSK_SIZE	0x00000008
#define C_RMU_ERR_INFO_MSK_OFFSET	0x00000140
#define C_RMU_ERR_INFO_MSK	(C_RMU_BASE + C_RMU_ERR_INFO_MSK_OFFSET)
#define C_RMU_ERR_INFO_MSK_SIZE	0x00000008
#define C_RMU_ERR_INFO_MEM_OFFSET	0x00000180
#define C_RMU_ERR_INFO_MEM	(C_RMU_BASE + C_RMU_ERR_INFO_MEM_OFFSET)
#define C_RMU_ERR_INFO_MEM_SIZE	0x00000008

union c_rmu_err_info_mem {
	uint64_t qw;
	struct {
		uint64_t cor_syndrome       : 12;
		uint64_t cor_address        : 13;
		uint64_t                    :  1;
		uint64_t cor_mem_id         :  4;
		uint64_t                    :  1;
		uint64_t cor_csr_detected   :  1;
		uint64_t ucor_syndrome      : 12;
		uint64_t ucor_address       : 13;
		uint64_t                    :  1;
		uint64_t ucor_mem_id        :  4;
		uint64_t                    :  1;
		uint64_t ucor_csr_detected  :  1;
	};
};

#define C_RMU_ERR_INFO_PTL_INVLD_DFA_OFFSET	0x00000188
#define C_RMU_ERR_INFO_PTL_INVLD_DFA	(C_RMU_BASE + C_RMU_ERR_INFO_PTL_INVLD_DFA_OFFSET)
#define C_RMU_ERR_INFO_PTL_INVLD_DFA_SIZE	0x00000008

union c_rmu_err_info_ptl_invld_dfa {
	uint64_t qw;
	struct {
		uint64_t dfa  : 32;
		uint64_t sfa  : 32;
	};
};

#define C_RMU_ERR_INFO_PTL_INVLD_VNI_OFFSET	0x00000190
#define C_RMU_ERR_INFO_PTL_INVLD_VNI	(C_RMU_BASE + C_RMU_ERR_INFO_PTL_INVLD_VNI_OFFSET)
#define C_RMU_ERR_INFO_PTL_INVLD_VNI_SIZE	0x00000008

union c_rmu_err_info_ptl_invld_vni {
	uint64_t qw;
	struct {
		uint64_t vni  : 16;
		uint64_t      : 16;
		uint64_t sfa  : 32;
	};
};

#define C_RMU_ERR_INFO_PTL_NO_PTLTE_OFFSET	0x00000198
#define C_RMU_ERR_INFO_PTL_NO_PTLTE	(C_RMU_BASE + C_RMU_ERR_INFO_PTL_NO_PTLTE_OFFSET)
#define C_RMU_ERR_INFO_PTL_NO_PTLTE_SIZE	0x00000008

union c_rmu_err_info_ptl_no_ptlte {
	uint64_t qw;
	struct {
		uint64_t index_ext     :  5;
		uint64_t multicast_id  : 13;
		uint64_t               :  2;
		uint64_t is_multicast  :  1;
		uint64_t               :  3;
		uint64_t vni_list_idx  :  8;
		uint64_t sfa           : 32;
	};
};

#define C_RMU_ERR_INFO_ENET_NO_PTLTE_SET_OFFSET	0x000001c0
#define C_RMU_ERR_INFO_ENET_NO_PTLTE_SET	(C_RMU_BASE + C_RMU_ERR_INFO_ENET_NO_PTLTE_SET_OFFSET)
#define C_RMU_ERR_INFO_ENET_NO_PTLTE_SET_SIZE	0x00000028

union c_rmu_err_info_enet_no_ptlte_set {
	uint64_t qw[5];
	struct {
		union {
			struct {
				uint64_t dipv6_lower;
				uint64_t dipv6_upper;
				uint64_t sipv6_lower;
				uint64_t sipv6_upper;
			};
			struct {
				uint32_t dipv4;
				uint32_t unused_opt_ipv4_0;
				uint64_t unused_opt_ipv4_1;
				uint32_t sipv4;
				uint32_t unused_opt_ipv4_2;
				uint64_t unused_opt_ipv4_3;
			};
			struct {
				uint64_t dmac : 48;
				uint64_t vid  : 12;
				uint64_t dei  :  1;
				uint64_t pcp  :  3;
				uint64_t vlan_present        :  1;
				uint64_t unused_ieee_802_3_0 : 63;
			};
		};
		uint64_t smac          : 48;
		uint64_t tc            :  3;
		uint64_t               :  1;
		uint64_t lossless      :  1;
		uint64_t               :  3;
		uint64_t frame_type    :  2;
		uint64_t ip_v4_present :  1;
		uint64_t ip_v6_present :  1;
		uint64_t               :  4;
	};
};

#define C_RMU_ERR_INFO_ENET_BAD_PTLTE_SET_CTRL_OFFSET	0x000001e8
#define C_RMU_ERR_INFO_ENET_BAD_PTLTE_SET_CTRL	(C_RMU_BASE + C_RMU_ERR_INFO_ENET_BAD_PTLTE_SET_CTRL_OFFSET)
#define C_RMU_ERR_INFO_ENET_BAD_PTLTE_SET_CTRL_SIZE	0x00000008

union c_rmu_err_info_enet_bad_ptlte_set_ctrl {
	uint64_t qw;
	struct {
		uint64_t ptlte_set_ctrl_entry     :  7;
		uint64_t                          :  1;
		uint64_t hash_type                :  5;
		uint64_t                          :  3;
		uint64_t portal_index_indir_base  : 11;
		uint64_t                          :  1;
		uint64_t hash_bits                :  4;
		uint64_t hash_value               : 32;
	};
};

#define C_RMU_CFG_VNI_LIST_X_OFFSET(idx)	(0x00001000+((idx)*8))
#define C_RMU_CFG_VNI_LIST_X(idx)	(C_RMU_BASE + C_RMU_CFG_VNI_LIST_X_OFFSET(idx))
#define C_RMU_CFG_VNI_LIST_X_SIZE	0x00000800
#define C_RMU_CFG_VNI_LIST_Y_OFFSET(idx)	(0x00001800+((idx)*8))
#define C_RMU_CFG_VNI_LIST_ENTRIES	256
#define C_RMU_CFG_VNI_LIST_Y(idx)	(C_RMU_BASE + C_RMU_CFG_VNI_LIST_Y_OFFSET(idx))
#define C_RMU_CFG_VNI_LIST_Y_SIZE	0x00000800

union c_rmu_cfg_vni_list {
	uint64_t qw;
	struct {
		uint64_t vni    : 16;
		uint64_t valid  :  1;
		uint64_t        : 47;
	};
};

#define C_RMU_CFG_VNI_LIST_INVALIDATE_OFFSET(idx)	(0x00002000+((idx)*8))
#define C_RMU_CFG_VNI_LIST_INVALIDATE(idx)	(C_RMU_BASE + C_RMU_CFG_VNI_LIST_INVALIDATE_OFFSET(idx))
#define C_RMU_CFG_VNI_LIST_INVALIDATE_SIZE	0x00000800

union c_rmu_cfg_vni_list_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C_RMU_CFG_VNI_LIST_CHK_OFFSET	0x00000400
#define C_RMU_CFG_VNI_LIST_CHK	(C_RMU_BASE + C_RMU_CFG_VNI_LIST_CHK_OFFSET)
#define C_RMU_CFG_VNI_LIST_CHK_SIZE	0x00000008

union c_rmu_cfg_vni_list_chk {
	uint64_t qw;
	struct {
		uint64_t intrvl  : 32;
		uint64_t         : 32;
	};
};

#define C_RMU_CFG_PORTAL_LIST_X_OFFSET(idx)	(0x00010000+((idx)*8))
#define C_RMU_CFG_PORTAL_LIST_X(idx)	(C_RMU_BASE + C_RMU_CFG_PORTAL_LIST_X_OFFSET(idx))
#define C_RMU_CFG_PORTAL_LIST_X_SIZE	0x00005000
#define C_RMU_CFG_PORTAL_LIST_Y_OFFSET(idx)	(0x00018000+((idx)*8))
#define C_RMU_CFG_PORTAL_LIST_ENTRIES	2560
#define C_RMU_CFG_PORTAL_LIST_Y(idx)	(C_RMU_BASE + C_RMU_CFG_PORTAL_LIST_Y_OFFSET(idx))
#define C_RMU_CFG_PORTAL_LIST_Y_SIZE	0x00005000

union c_rmu_cfg_portal_list {
	uint64_t qw;
	struct {
		union {
			struct {
				uint64_t index_ext     :  5;
				uint64_t multicast_id  : 13;
				uint64_t	       :  2;
				uint64_t is_multicast  :  1;
				uint64_t	       :  3;
				uint64_t vni_list_idx  :  8;
				uint64_t valid	       :  1;
				uint64_t	       : 31;
			};
			struct {
				uint64_t		  : 5;
				uint64_t endpoint_defined : 12;
				uint64_t		  : 7;
			};
		};
	};
};

#define C_RMU_CFG_PORTAL_LIST_INVALIDATE_OFFSET(idx)	(0x00020000+((idx)*8))
#define C_RMU_CFG_PORTAL_LIST_INVALIDATE(idx)	(C_RMU_BASE + C_RMU_CFG_PORTAL_LIST_INVALIDATE_OFFSET(idx))
#define C_RMU_CFG_PORTAL_LIST_INVALIDATE_SIZE	0x00005000

union c_rmu_cfg_portal_list_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C_RMU_CFG_PORTAL_LIST_CHK_OFFSET	0x00000408
#define C_RMU_CFG_PORTAL_LIST_CHK	(C_RMU_BASE + C_RMU_CFG_PORTAL_LIST_CHK_OFFSET)
#define C_RMU_CFG_PORTAL_LIST_CHK_SIZE	0x00000008

union c_rmu_cfg_portal_list_chk {
	uint64_t qw;
	struct {
		uint64_t intrvl  : 32;
		uint64_t         : 32;
	};
};

#define C_RMU_CFG_PORTAL_INDEX_TABLE_OFFSET(idx)	(0x00004000+((idx)*8))
#define C_RMU_CFG_PORTAL_INDEX_TABLE_ENTRIES	640
#define C_RMU_CFG_PORTAL_INDEX_TABLE(idx)	(C_RMU_BASE + C_RMU_CFG_PORTAL_INDEX_TABLE_OFFSET(idx))
#define C_RMU_CFG_PORTAL_INDEX_TABLE_SIZE	0x00001400

#define C_RMU_CFG_PORTAL_INDEX_TABLE_ARRAY_SIZE 4
union c_rmu_cfg_portal_index_table {
	uint64_t qw;
	struct {
		uint16_t phys_portal_table_idx : 11;
		uint16_t 	               :  5;
	} e[C_RMU_CFG_PORTAL_INDEX_TABLE_ARRAY_SIZE];
};

#define C_RMU_CFG_PTLTE_SET_LIST_X_OFFSET(idx)	(0x00006000+((idx)*32))
#define C_RMU_CFG_PTLTE_SET_LIST_X(idx)	(C_RMU_BASE + C_RMU_CFG_PTLTE_SET_LIST_X_OFFSET(idx))
#define C_RMU_CFG_PTLTE_SET_LIST_X_SIZE	0x00001000
#define C_RMU_CFG_PTLTE_SET_LIST_Y_OFFSET(idx)	(0x00007000+((idx)*32))
#define C_RMU_CFG_PTLTE_SET_LIST_ENTRIES	128
#define C_RMU_CFG_PTLTE_SET_LIST_Y(idx)	(C_RMU_BASE + C_RMU_CFG_PTLTE_SET_LIST_Y_OFFSET(idx))
#define C_RMU_CFG_PTLTE_SET_LIST_Y_SIZE	0x00001000

union c_rmu_cfg_ptlte_set_list {
	uint64_t qw[4];
	struct {
		union {
			struct {
				uint64_t	dmac		: 48;
				uint64_t	vid		: 12;
				uint64_t	dei		:  1;
				uint64_t	pcp		:  3;
			};
			uint64_t	dipv6[2];
			struct {
				uint32_t	dipv4;
				uint32_t	opt_ipv4_unused_0;
				union {
					struct {
						uint64_t	vlan_present	  :  1;
						uint64_t	ieee_802_3_unused : 63;
					};
					uint64_t opt_ipv4_unused_1;
				};
			};
		};
		uint64_t	lossless	:  1;
		uint64_t			:  7;
		uint64_t	frame_type	:  2;
		uint64_t			: 54;
		uint64_t	valid		:  1;
		uint64_t			: 63;
	};
};

#define C_RMU_CFG_PTLTE_SET_LIST_INVALIDATE_OFFSET(idx)	(0x00008000+((idx)*32))
#define C_RMU_CFG_PTLTE_SET_LIST_INVALIDATE(idx)	(C_RMU_BASE + C_RMU_CFG_PTLTE_SET_LIST_INVALIDATE_OFFSET(idx))
#define C_RMU_CFG_PTLTE_SET_LIST_INVALIDATE_SIZE	0x00001000

union c_rmu_cfg_ptlte_set_list_invalidate {
	uint64_t qw;
	struct {
		uint64_t invalidate  :  1;
		uint64_t             : 63;
	};
};

#define C_RMU_CFG_PTLTE_SET_LIST_CHK_OFFSET	0x00000418
#define C_RMU_CFG_PTLTE_SET_LIST_CHK	(C_RMU_BASE + C_RMU_CFG_PTLTE_SET_LIST_CHK_OFFSET)
#define C_RMU_CFG_PTLTE_SET_LIST_CHK_SIZE	0x00000008

union c_rmu_cfg_ptlte_set_list_chk {
	uint64_t qw;
	struct {
		uint64_t intrvl  : 32;
		uint64_t         : 32;
	};
};

#define C_RMU_CFG_PTLTE_SET_CTRL_TABLE_OFFSET(idx)	(0x00009000+((idx)*8))
#define C_RMU_CFG_PTLTE_SET_CTRL_TABLE_ENTRIES	64
#define C_RMU_CFG_PTLTE_SET_CTRL_TABLE(idx)	(C_RMU_BASE + C_RMU_CFG_PTLTE_SET_CTRL_TABLE_OFFSET(idx))
#define C_RMU_CFG_PTLTE_SET_CTRL_TABLE_SIZE	0x00000200

struct c_rmu_cfg_ptlte_set_ctrl_table_entry {
	uint32_t portal_index_indir_base  : 11;
	uint32_t                          :  1;
	uint32_t hash_bits                :  4;
	uint32_t hash_types_enabled       : 16;
};

#define C_RMU_CFG_PTLTE_SET_CTRL_TABLE_ARRAY_SIZE 2
union c_rmu_cfg_ptlte_set_ctrl_table {
	uint64_t qw;
	struct c_rmu_cfg_ptlte_set_ctrl_table_entry e[C_RMU_CFG_PTLTE_SET_CTRL_TABLE_ARRAY_SIZE];
};

#define C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_OFFSET(idx)	(0x0000a000+((idx)*8))
#define C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_ENTRIES	544
#define C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE(idx)	(C_RMU_BASE + C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_OFFSET(idx))
#define C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_SIZE	0x00001100

#define C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_ARRAY_SIZE 4
union c_rmu_cfg_portal_index_indir_table {
	uint64_t qw;
	struct {
		uint16_t phys_portal_table_idx : 11;
		uint16_t 	               :  5;
	} e[C_RMU_CFG_PORTAL_INDEX_INDIR_TABLE_ARRAY_SIZE];
};

#define C_RMU_CFG_HASH_KEY_OFFSET	0x00000440
#define C_RMU_CFG_HASH_KEY	(C_RMU_BASE + C_RMU_CFG_HASH_KEY_OFFSET)
#define C_RMU_CFG_HASH_KEY_SIZE	0x00000030

union c_rmu_cfg_hash_key {
	uint64_t qw[6];
	struct {
		uint64_t key_wd_0;
		uint64_t key_wd_1;
		uint64_t key_wd_2;
		uint64_t key_wd_3;
		uint64_t key_wd_4;
		uint64_t key_wd_5   : 31;
		uint64_t            : 33;
	};
};

#define C_RMU_STS_INIT_DONE_OFFSET	0x00000818
#define C_RMU_STS_INIT_DONE	(C_RMU_BASE + C_RMU_STS_INIT_DONE_OFFSET)
#define C_RMU_STS_INIT_DONE_SIZE	0x00000008

union c_rmu_sts_init_done {
	uint64_t qw;
	struct {
		uint64_t init_done   :  1;
		uint64_t warm_reset  :  1;
		uint64_t             : 62;
	};
};

#define C2_HNI_EXT_ERR_FLG_OFFSET	0x00000148
#define C2_HNI_EXT_ERR_FLG	(C_HNI_BASE + C2_HNI_EXT_ERR_FLG_OFFSET)
#define C2_HNI_EXT_ERR_FLG_SIZE	0x00000008

union c2_hni_ext_err_flg {
	uint64_t qw;
	struct {
		uint64_t sw_diag       :  1;
		uint64_t pmi_err       :  1;
		uint64_t fsm_err       :  1;
		uint64_t pmi_ack       :  1;
		uint64_t rx_ctrl_perr  :  1;
		uint64_t tx_ctrl_perr  :  1;
		uint64_t               : 58;
	};
};

#define C2_HNI_EXT_ERR_FIRST_FLG_OFFSET	0x00000150
#define C2_HNI_EXT_ERR_FIRST_FLG	(C_HNI_BASE + C2_HNI_EXT_ERR_FIRST_FLG_OFFSET)
#define C2_HNI_EXT_ERR_FIRST_FLG_SIZE	0x00000008
#define C2_HNI_EXT_ERR_FIRST_FLG_TS_OFFSET	0x00000158
#define C2_HNI_EXT_ERR_FIRST_FLG_TS	(C_HNI_BASE + C2_HNI_EXT_ERR_FIRST_FLG_TS_OFFSET)
#define C2_HNI_EXT_ERR_FIRST_FLG_TS_SIZE	0x00000008
#define C2_HNI_EXT_ERR_CLR_OFFSET	0x00000160
#define C2_HNI_EXT_ERR_CLR	(C_HNI_BASE + C2_HNI_EXT_ERR_CLR_OFFSET)
#define C2_HNI_EXT_ERR_CLR_SIZE	0x00000008
#define C2_HNI_EXT_ERR_IRQA_MSK_OFFSET	0x00000168
#define C2_HNI_EXT_ERR_IRQA_MSK	(C_HNI_BASE + C2_HNI_EXT_ERR_IRQA_MSK_OFFSET)
#define C2_HNI_EXT_ERR_IRQA_MSK_SIZE	0x00000008
#define C2_HNI_EXT_ERR_IRQB_MSK_OFFSET	0x00000170
#define C2_HNI_EXT_ERR_IRQB_MSK	(C_HNI_BASE + C2_HNI_EXT_ERR_IRQB_MSK_OFFSET)
#define C2_HNI_EXT_ERR_IRQB_MSK_SIZE	0x00000008
#define C2_HNI_EXT_ERR_INFO_MSK_OFFSET	0x00000180
#define C2_HNI_EXT_ERR_INFO_MSK	(C_HNI_BASE + C2_HNI_EXT_ERR_INFO_MSK_OFFSET)
#define C2_HNI_EXT_ERR_INFO_MSK_SIZE	0x00000008
#define C2_HNI_EXT_ERR_INFO_PMI_OFFSET	0x000001c8
#define C2_HNI_EXT_ERR_INFO_PMI	(C_HNI_BASE + C2_HNI_EXT_ERR_INFO_PMI_OFFSET)
#define C2_HNI_EXT_ERR_INFO_PMI_SIZE	0x00000008

union c2_hni_ext_err_info_pmi {
	uint64_t qw;
	struct {
		uint64_t pmi_lp_error  :  1;
		uint64_t               : 63;
	};
};

#define C2_HNI_EXT_ERR_INFO_FSM_OFFSET	0x000001d0
#define C2_HNI_EXT_ERR_INFO_FSM	(C_HNI_BASE + C2_HNI_EXT_ERR_INFO_FSM_OFFSET)
#define C2_HNI_EXT_ERR_INFO_FSM_SIZE	0x00000008

union c2_hni_ext_err_info_fsm {
	uint64_t qw;
	struct {
		uint64_t fsm_timeout         :  1;
		uint64_t fsm_illegal_access  :  1;
		uint64_t                     : 62;
	};
};

#define C2_HNI_SERDES_PMI_DATA_OFFSET	0x00000a08
#define C2_HNI_SERDES_PMI_DATA	(C_HNI_BASE + C2_HNI_SERDES_PMI_DATA_OFFSET)
#define C2_HNI_SERDES_PMI_DATA_SIZE	0x00000008

union c2_hni_serdes_pmi_data {
	uint64_t qw;
	struct {
		uint64_t pmi_lp_rddata    : 16;
		uint64_t pmi_lp_wrdata    : 16;
		uint64_t pmi_lp_maskdata  : 16;
		uint64_t                  : 16;
	};
};

#define C2_HNI_SERDES_PMI_CTL_OFFSET	0x00000a10
#define C2_HNI_SERDES_PMI_CTL	(C_HNI_BASE + C2_HNI_SERDES_PMI_CTL_OFFSET)
#define C2_HNI_SERDES_PMI_CTL_SIZE	0x00000008

union c2_hni_serdes_pmi_ctl {
	uint64_t qw;
	struct {
		uint64_t pmi_lp_addr      : 32;
		uint64_t pmi_lp_write     :  1;
		uint64_t pmi_lp_read_vld  :  1;
		uint64_t pmi_lp_ack       :  1;
		uint64_t pmi_lp_error     :  1;
		uint64_t pmi_lp_en        :  1;
		uint64_t                  : 27;
	};
};

#define SS2_PORT_PML_CFG_CRNC_OFFSET	0x00000000
#define SS2_PORT_PML_CFG_CRNC	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_CRNC_OFFSET)
#define SS2_PORT_PML_CFG_CRNC_SIZE	0x00000008

union ss2_port_pml_cfg_crnc {
	uint64_t qw;
	struct {
		uint64_t csr_timeout   :  8;
		uint64_t mcast_id      : 10;
		uint64_t               :  1;
		uint64_t mcast_enable  :  1;
		uint64_t mcast_mask    : 10;
		uint64_t               : 34;
	};
};

#define SS2_PORT_PML_MSC_SHADOW_ACTION_OFFSET	0x00000028
#define SS2_PORT_PML_MSC_SHADOW_ACTION	(C_HNI_PML_BASE + SS2_PORT_PML_MSC_SHADOW_ACTION_OFFSET)
#define SS2_PORT_PML_MSC_SHADOW_ACTION_SIZE	0x00000008
#define SS2_PORT_PML_MSC_SHADOW_OFFSET	0x00000030
#define SS2_PORT_PML_MSC_SHADOW	(C_HNI_PML_BASE + SS2_PORT_PML_MSC_SHADOW_OFFSET)
#define SS2_PORT_PML_MSC_SHADOW_SIZE	0x00000040

union ss2_port_pml_msc_shadow {
	uint64_t qw[5];
};

#define SS2_PORT_PML_ERR_ELAPSED_TIME_OFFSET	0x00000100
#define SS2_PORT_PML_ERR_ELAPSED_TIME	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_ELAPSED_TIME_OFFSET)
#define SS2_PORT_PML_ERR_ELAPSED_TIME_SIZE	0x00000008

union ss2_port_pml_err_elapsed_time {
	uint64_t qw;
	struct {
		uint64_t nanoseconds  : 30;
		uint64_t seconds      : 34;
	};
};

#define SS2_PORT_PML_ERR_INFO_EDC_COR_OFFSET	0x00000300
#define SS2_PORT_PML_ERR_INFO_EDC_COR	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_EDC_COR_OFFSET)
#define SS2_PORT_PML_ERR_INFO_EDC_COR_SIZE	0x00000008

union ss2_port_pml_err_info_edc_cor {
	uint64_t qw;
	struct {
		uint64_t syndrome      : 12;
		uint64_t address       : 14;
		uint64_t               :  1;
		uint64_t csr_detected  :  1;
		uint64_t mem_id        :  5;
		uint64_t               : 31;
	};
};

#define SS2_PORT_PML_ERR_INFO_EDC_UCOR_OFFSET	0x00000308
#define SS2_PORT_PML_ERR_INFO_EDC_UCOR	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_EDC_UCOR_OFFSET)
#define SS2_PORT_PML_ERR_INFO_EDC_UCOR_SIZE	0x00000008

union ss2_port_pml_err_info_edc_ucor {
	uint64_t qw;
	struct {
		uint64_t syndrome      : 12;
		uint64_t address       : 14;
		uint64_t               :  1;
		uint64_t csr_detected  :  1;
		uint64_t mem_id        :  5;
		uint64_t               : 31;
	};
};

#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_SBE_OFFSET	0x00000360
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_SBE	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_LLR_TX_DP_SBE_OFFSET)
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_SBE_SIZE	0x00000008

union ss2_port_pml_err_info_llr_tx_dp_sbe {
	uint64_t qw;
	struct {
		uint64_t syndrome  : 11;
		uint64_t           :  5;
		uint64_t subport   :  2;
		uint64_t           :  6;
		uint64_t dp_id     :  3;
		uint64_t           : 37;
	};
};

#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_MBE_OFFSET	0x00000368
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_MBE	(C_HNI_PML_BASE + SS2_PORT_PML_ERR_INFO_LLR_TX_DP_MBE_OFFSET)
#define SS2_PORT_PML_ERR_INFO_LLR_TX_DP_MBE_SIZE	0x00000008

union ss2_port_pml_err_info_llr_tx_dp_mbe {
	uint64_t qw;
	struct {
		uint64_t syndrome  : 11;
		uint64_t           :  5;
		uint64_t subport   :  2;
		uint64_t           :  6;
		uint64_t dp_id     :  3;
		uint64_t           : 37;
	};
};

#define SS2_PORT_PML_CFG_GENERAL_OFFSET	0x00000400
#define SS2_PORT_PML_CFG_GENERAL	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_GENERAL_OFFSET)
#define SS2_PORT_PML_CFG_GENERAL_SIZE	0x00000008

union ss2_port_pml_cfg_general {
	uint64_t qw;
	struct {
		uint64_t clock_period_ps  : 11;
		uint64_t                  : 53;
	};
};

#define SS2_PORT_PML_CFG_PORT_GROUP_OFFSET	0x00000408
#define SS2_PORT_PML_CFG_PORT_GROUP	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PORT_GROUP_OFFSET)
#define SS2_PORT_PML_CFG_PORT_GROUP_SIZE	0x00000008

union ss2_port_pml_cfg_port_group {
	uint64_t qw;
	struct {
		uint64_t pg_cfg         :  2;
		uint64_t                :  6;
		uint64_t link_function  :  4;
		uint64_t                : 52;
	};
};

#define SS2_PORT_PML_CFG_SUBPORT_RESET_OFFSET(idx)	(0x00000420+((idx)*8))
#define SS2_PORT_PML_CFG_SUBPORT_RESET_ENTRIES	4
#define SS2_PORT_PML_CFG_SUBPORT_RESET(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_SUBPORT_RESET_OFFSET(idx))
#define SS2_PORT_PML_CFG_SUBPORT_RESET_SIZE	0x00000020

union ss2_port_pml_cfg_subport_reset {
	uint64_t qw;
	struct {
		uint64_t warm_rst_from_csr  :  1;
		uint64_t                    : 63;
	};
};

#define SS2_PORT_PML_CFG_PCS_SUBPORT_OFFSET(idx)	(0x00000520+((idx)*8))
#define SS2_PORT_PML_CFG_PCS_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_PCS_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_PCS_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_PCS_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_pcs_subport {
	uint64_t qw;
	struct {
		uint64_t pcs_enable       :  1;
		uint64_t enable_auto_neg  :  1;
		uint64_t                  : 14;
		uint64_t ll_fec           :  1;
		uint64_t                  : 47;
	};
};

#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT_OFFSET(idx)	(0x000006a0+((idx)*8))
#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_TX_PCS_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_TX_PCS_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_tx_pcs_subport {
	uint64_t qw;
	struct {
		uint64_t enable_ctl_os    :  1;
		uint64_t ctl_os_rate      :  1;
		uint64_t                  :  2;
		uint64_t gearbox_credits  :  5;
		uint64_t                  : 55;
	};
};

#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_OFFSET(idx)	(0x000006e0+((idx)*8))
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_RX_PCS_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_RX_PCS_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_rx_pcs_subport {
	uint64_t qw;
	struct {
		uint64_t enable_rx_sm   :  1;
		uint64_t enable_ctl_os  :  1;
		uint64_t enable_lock    :  1;
		uint64_t                :  5;
		uint64_t rs_mode        :  3;
		uint64_t                : 53;
	};
};

#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_OFFSET(idx)	(0x00000820+((idx)*8))
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_TX_MAC_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_TX_MAC_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_tx_mac_subport {
	uint64_t qw;
	struct {
		uint64_t mac_operational   :  1;
		uint64_t short_preamble    :  1;
		uint64_t                   :  6;
		uint64_t pcs_credits       :  4;
		uint64_t                   :  4;
		uint64_t mac_cdt_init_val  :  8;
		uint64_t mac_cdt_thresh    :  8;
		uint64_t                   : 32;
	};
};

#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_OFFSET(idx)	(0x00000860+((idx)*8))
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_RX_MAC_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_RX_MAC_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_rx_mac_subport {
	uint64_t qw;
	struct {
		uint64_t mac_operational  :  1;
		uint64_t short_preamble   :  1;
		uint64_t                  : 62;
	};
};

#define SS2_PORT_PML_CFG_LLR_SUBPORT_OFFSET(idx)	(0x00000920+((idx)*8))
#define SS2_PORT_PML_CFG_LLR_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_CFG_LLR_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_LLR_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_CFG_LLR_SUBPORT_SIZE	0x00000020

union ss2_port_pml_cfg_llr_subport {
	uint64_t qw;
	struct {
		uint64_t llr_mode                  :  2;
		uint64_t                           :  6;
		uint64_t enable_loop_timing        :  1;
		uint64_t                           :  7;
		uint64_t filter_ctl_frames         :  1;
		uint64_t                           :  3;
		uint64_t filter_lossless_when_off  :  1;
		uint64_t                           :  3;
		uint64_t mac_if_credits            :  4;
		uint64_t                           :  4;
		uint64_t link_down_behavior        :  2;
		uint64_t                           :  6;
		uint64_t max_starvation_limit      : 16;
		uint64_t                           :  8;
	};
};

#define SS2_PORT_PML_CFG_SERDES_RX_OFFSET(idx)	(0x00001000+((idx)*8))
#define SS2_PORT_PML_CFG_SERDES_RX_ENTRIES	4
#define SS2_PORT_PML_CFG_SERDES_RX(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_SERDES_RX_OFFSET(idx))
#define SS2_PORT_PML_CFG_SERDES_RX_SIZE	0x00000020

union ss2_port_pml_cfg_serdes_rx {
	uint64_t qw;
	struct {
		uint64_t pmd_rx_lane_mode     : 16;
		uint64_t pmd_rx_osr_mode      :  9;
		uint64_t                      :  3;
		uint64_t pmd_ext_los          :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_rx_h_pwrdn    :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_rx_dp_h_rstb  :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_rx_h_rstb     :  1;
		uint64_t                      : 23;
	};
};

#define SS2_PORT_PML_CFG_SERDES_TX_OFFSET(idx)	(0x00001020+((idx)*8))
#define SS2_PORT_PML_CFG_SERDES_TX_ENTRIES	4
#define SS2_PORT_PML_CFG_SERDES_TX(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_CFG_SERDES_TX_OFFSET(idx))
#define SS2_PORT_PML_CFG_SERDES_TX_SIZE	0x00000020

union ss2_port_pml_cfg_serdes_tx {
	uint64_t qw;
	struct {
		uint64_t pmd_tx_lane_mode     : 16;
		uint64_t pmd_tx_osr_mode      :  9;
		uint64_t                      :  3;
		uint64_t pmd_tx_disable       :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_tx_h_pwrdn    :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_tx_dp_h_rstb  :  1;
		uint64_t                      :  3;
		uint64_t pmd_ln_tx_h_rstb     :  1;
		uint64_t                      : 23;
	};
};

#define SS2_PORT_PML_STS_SERDES_OFFSET(idx)	(0x00001200+((idx)*8))
#define SS2_PORT_PML_STS_SERDES_ENTRIES	4
#define SS2_PORT_PML_STS_SERDES(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_SERDES_OFFSET(idx))
#define SS2_PORT_PML_STS_SERDES_SIZE	0x00000020

union ss2_port_pml_sts_serdes {
	uint64_t qw;
	struct {
		uint64_t pmd_signal_detect  :  1;
		uint64_t                    :  3;
		uint64_t pmd_rx_lock        :  1;
		uint64_t                    :  3;
		uint64_t pmd_rx_clk_vld     :  1;
		uint64_t                    :  3;
		uint64_t pmd_rx_data_vld    :  1;
		uint64_t                    :  3;
		uint64_t pmd_tx_clk_vld     :  1;
		uint64_t                    :  3;
		uint64_t pmd_tx_data_vld    :  1;
		uint64_t                    :  3;
		uint64_t rx_signal_ok       :  1;
		uint64_t                    : 39;
	};
};

#define SS2_PORT_PML_STS_RX_PCS_AM_LOCK_OFFSET	0x00001380
#define SS2_PORT_PML_STS_RX_PCS_AM_LOCK	(C_HNI_PML_BASE + SS2_PORT_PML_STS_RX_PCS_AM_LOCK_OFFSET)
#define SS2_PORT_PML_STS_RX_PCS_AM_LOCK_SIZE	0x00000008

union ss2_port_pml_sts_rx_pcs_am_lock {
	uint64_t qw;
	struct {
		uint64_t am_lock  : 16;
		uint64_t          : 48;
	};
};

#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_OFFSET(idx)	(0x000013a0+((idx)*8))
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_ENTRIES	4
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT(idx)	(C_HNI_PML_BASE + SS2_PORT_PML_STS_RX_PCS_SUBPORT_OFFSET(idx))
#define SS2_PORT_PML_STS_RX_PCS_SUBPORT_SIZE	0x00000020

union ss2_port_pml_sts_rx_pcs_subport {
	uint64_t qw;
	struct {
		uint64_t               :  4;
		uint64_t align_status  :  1;
		uint64_t               :  3;
		uint64_t fault         :  1;
		uint64_t               :  3;
		uint64_t local_fault   :  1;
		uint64_t               :  3;
		uint64_t hi_ser        :  1;
		uint64_t               : 47;
	};
};

#define C2_IXE_CFG_DISP_ORD_OFFSET	0x000007a8
#define C2_IXE_CFG_DISP_ORD	(C_IXE_BASE + C2_IXE_CFG_DISP_ORD_OFFSET)
#define C2_IXE_CFG_DISP_ORD_SIZE	0x00000008

union c2_ixe_cfg_disp_ord {
	uint64_t qw;
	struct {
		uint64_t wr_atomic_op  :  1;
		uint64_t get_ro_last   :  1;
		uint64_t get_ro_body   :  1;
		uint64_t               : 61;
	};
};

#define C2_LPE_CFG_MATCH_SEL_OFFSET	0x00000a20
#define C2_LPE_CFG_MATCH_SEL	(C_LPE_BASE + C2_LPE_CFG_MATCH_SEL_OFFSET)
#define C2_LPE_CFG_MATCH_SEL_SIZE	0x00000008

union c2_lpe_cfg_match_sel {
	uint64_t qw;
	struct {
		uint64_t vni_cfg  :  2;
		uint64_t          : 62;
	};
};

#define C2_LPE_CFG_GET_FQ_DFA_MASK_OFFSET(idx)	(0x00000b00+((idx)*8))
#define C2_LPE_CFG_GET_FQ_DFA_MASK_ENTRIES	16
#define C2_LPE_CFG_GET_FQ_DFA_MASK(idx)	(C_LPE_BASE + C2_LPE_CFG_GET_FQ_DFA_MASK_OFFSET(idx))
#define C2_LPE_CFG_GET_FQ_DFA_MASK_SIZE	0x00000080

union c2_lpe_cfg_get_fq_dfa_mask {
	uint64_t qw;
	struct {
		uint64_t dscp0_dfa_mask  : 12;
		uint64_t                 :  4;
		uint64_t dscp1_dfa_mask  : 12;
		uint64_t                 :  4;
		uint64_t dscp2_dfa_mask  : 12;
		uint64_t                 :  4;
		uint64_t dscp3_dfa_mask  : 12;
		uint64_t                 :  4;
	};
};

#define C2_OXE_ERR_INFO_SPT_CDT_ERR_OFFSET	0x000001e8
#define C2_OXE_ERR_INFO_SPT_CDT_ERR	(C_OXE_BASE + C2_OXE_ERR_INFO_SPT_CDT_ERR_OFFSET)
#define C2_OXE_ERR_INFO_SPT_CDT_ERR_SIZE	0x00000008

union c2_oxe_err_info_spt_cdt_err {
	uint64_t qw;
	struct {
		uint64_t index  : 11;
		uint64_t        :  5;
		uint64_t bc     :  4;
		uint64_t        : 44;
	};
};

#define C2_OXE_ERR_INFO_SMT_CDT_ERR_OFFSET	0x000001f0
#define C2_OXE_ERR_INFO_SMT_CDT_ERR	(C_OXE_BASE + C2_OXE_ERR_INFO_SMT_CDT_ERR_OFFSET)
#define C2_OXE_ERR_INFO_SMT_CDT_ERR_SIZE	0x00000008

union c2_oxe_err_info_smt_cdt_err {
	uint64_t qw;
	struct {
		uint64_t index  :  8;
		uint64_t        :  8;
		uint64_t bc     :  4;
		uint64_t        : 44;
	};
};

#define C2_OXE_CFG_SPT_BC_LIMIT_OFFSET(idx)	(0x00000680+((idx)*8))
#define C2_OXE_CFG_SPT_BC_LIMIT_ENTRIES	10
#define C2_OXE_CFG_SPT_BC_LIMIT(idx)	(C_OXE_BASE + C2_OXE_CFG_SPT_BC_LIMIT_OFFSET(idx))
#define C2_OXE_CFG_SPT_BC_LIMIT_SIZE	0x00000050

union c2_oxe_cfg_spt_bc_limit {
	uint64_t qw;
	struct {
		uint64_t limit  : 11;
		uint64_t        : 53;
	};
};

#define C2_PI_ERR_INFO_AHB_OFFSET	0x00000300
#define C2_PI_ERR_INFO_AHB	(C_PI_BASE + C2_PI_ERR_INFO_AHB_OFFSET)
#define C2_PI_ERR_INFO_AHB_SIZE	0x00000018

union c2_pi_err_info_ahb {
	uint64_t qw[3];
	struct {
		uint64_t hwrite      :  1;
		uint64_t htrans      :  2;
		uint64_t hsize       :  3;
		uint64_t hrdata_sel  :  4;
		uint64_t             :  5;
		uint64_t ahb_err_id  :  1;
		uint64_t haddr       : 14;
		uint64_t             :  2;
		uint64_t hsel        : 16;
		uint64_t             : 16;
		uint64_t req         :  1;
		uint64_t             : 31;
		uint64_t hwdata      : 32;
		uint64_t done        : 16;
		uint64_t hresp       : 16;
		uint64_t hrdata      : 32;
	};
};



struct c_cq_cntrs_group {
	struct c_cq_cntrs cq;
	struct c_tou_cntrs tou;
	struct c_pct_cntrs pct;
	struct c_ee_cntrs ee;
};

struct c_lpe_cntrs_group {
	struct c_lpe_cntrs lpe;
	struct c_ixe_cntrs ixe;
	struct c_rmu_cntrs rmu;
	struct c_parbs_cntrs parbs;
	struct c_mst_cntrs mst;
};

struct c_hni_cntrs_group {
	struct c_hni_cntrs hni;
	struct c1_hni_pcs_cntrs pcs;
	struct c1_hni_mac_cntrs mac;
	struct c1_hni_llr_cntrs llr;
	struct c1_oxe_cntrs oxe;
};

struct c_mb_cntrs_group {
	struct c_mb_cntrs mb;
	struct c_pi_cntrs pi;
	struct c_atu_cntrs atu;
};

struct c_pi_ipd_cntrs_group {
	struct c_pi_ipd_cntrs pi_ipd;
};



#define  C_PI_CFG_DMAC_STATUS_AXI_RESP_OKAY             (0)
#define  C_PI_CFG_DMAC_STATUS_AXI_RESP_EXOKAY           (1)
#define  C_PI_CFG_DMAC_STATUS_AXI_RESP_SLVERR           (2)
#define  C_PI_CFG_DMAC_STATUS_AXI_RESP_DECERR           (3)

#define  C_PI_DMAC_CDESC_STATUS_AXI_RESP_OKAY           \
	C_PI_CFG_DMAC_STATUS_AXI_RESP_OKAY
#define  C_PI_DMAC_CDESC_STATUS_AXI_RESP_EXOKAY         \
	C_PI_CFG_DMAC_STATUS_AXI_RESP_EXOKAY
#define  C_PI_DMAC_CDESC_STATUS_AXI_RESP_SLVERR         \
	C_PI_CFG_DMAC_STATUS_AXI_RESP_SLVERR
#define  C_PI_DMAC_CDESC_STATUS_AXI_RESP_DECERR         \
	C_PI_CFG_DMAC_STATUS_AXI_RESP_DECERR


#endif
