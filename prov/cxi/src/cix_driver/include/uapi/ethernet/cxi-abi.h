/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/* Copyright 2021,2024,2025 Hewlett Packard Enterprise Development LP */

#ifndef __UAPI_CXI_ABI_H
#define __UAPI_CXI_ABI_H

#include <linux/types.h>
#include <cassini_user_defs.h>

#ifndef __KERNEL__

#ifndef __user
#define __user
#endif

#include <stdbool.h>

#endif

/* Information for mmap */
struct cxi_mminfo {
    __u64 offset;
    __u64 size;
};

enum cxi_command_opcode {
		CXI_OP_INVALID,

		CXI_OP_LNI_ALLOC,
		CXI_OP_LNI_FREE,
		CXI_OP_DOMAIN_RESERVE,
		CXI_OP_DOMAIN_ALLOC,
		CXI_OP_DOMAIN_FREE,
		CXI_OP_CP_ALLOC,
		CXI_OP_CP_FREE,
		CXI_OP_CQ_ALLOC,
		CXI_OP_CQ_FREE,
		CXI_OP_CQ_ACK_COUNTER,
		CXI_OP_ATU_MAP,
		CXI_OP_ATU_UNMAP,
		CXI_OP_EQ_ALLOC,
		CXI_OP_EQ_FREE,
		CXI_OP_EQ_RESIZE,
		CXI_OP_EQ_RESIZE_COMPLETE,
		CXI_OP_PTE_ALLOC,
		CXI_OP_PTE_FREE,
		CXI_OP_PTE_MAP,
		CXI_OP_PTE_UNMAP,
		CXI_OP_PTE_LE_INVALIDATE,
		CXI_OP_PTE_STATUS,
		CXI_OP_WAIT_ALLOC,
		CXI_OP_WAIT_FREE,
		CXI_OP_CT_ALLOC,
		CXI_OP_CT_FREE,
		CXI_OP_SVC_LIST_GET,
		CXI_OP_SVC_ALLOC,
		CXI_OP_SVC_DESTROY,
		CXI_OP_MAP_CSRS,
		CXI_OP_SBUS_OP_RESET,
		CXI_OP_SBUS_OP,
		CXI_OP_SERDES_OP,
		CXI_OP_GET_DEV_PROPERTIES,
		_OBSOLETED_CXI_OP_MAP_TELEMETRY,
		_OBSOLETED_CXI_OP_UNMAP_TELEMETRY,
		CXI_OP_EQ_ADJUST_RESERVED_FC,
		CXI_OP_CT_WB_UPDATE,
		_OBSOLETED_CXI_OP_RESERVE_CNTR_POOL_ID,
		_OBSOLETED_CXI_OP_RELEASE_CNTR_POOL_ID,
		_OBSOLETED_CXI_OP_GET_TELEMETRY_REFRESH_INTERVAL,
		_OBSOLETED_CXI_OP_SET_TELEMETRY_REFRESH_INTERVAL,
		CXI_OP_INBOUND_WAIT,
		CXI_OP_SVC_UPDATE,
		CXI_OP_SVC_GET,
		_OBSOLETED_CXI_OP_START_TELEMETRY,
		_OBSOLETED_CXI_OP_STOP_TELEMETRY,
		CXI_OP_PTE_TRANSITION_SM,
		CXI_OP_SVC_RSRC_LIST_GET,
		CXI_OP_SVC_RSRC_GET,
		CXI_OP_ATU_UPDATE_MD,
		CXI_OP_DEV_INFO_GET,

		CXI_OP_SVC_SET_LPR,
		CXI_OP_SVC_GET_LPR,

		CXI_OP_DEV_ALLOC_RX_PROFILE,
		CXI_OP_DEV_GET_RX_PROFILE_IDS,

		CXI_OP_RX_PROFILE_RELEASE,
		CXI_OP_RX_PROFILE_REVOKE,
		CXI_OP_RX_PROFILE_GET_INFO,

		CXI_OP_RX_PROFILE_ADD_AC_ENTRY,
		CXI_OP_RX_PROFILE_REMOVE_AC_ENTRY,
		CXI_OP_RX_PROFILE_GET_AC_ENTRY_IDS,
		CXI_OP_RX_PROFILE_GET_AC_ENTRY_DATA_BY_ID,
		CXI_OP_RX_PROFILE_GET_AC_ENTRY_ID_BY_DATA,
		CXI_OP_RX_PROFILE_GET_AC_ENTRY_ID_BY_USER,

		CXI_OP_DEV_ALLOC_TX_PROFILE,
		CXI_OP_DEV_GET_TX_PROFILE_IDS,

		CXI_OP_TX_PROFILE_RELEASE,
		CXI_OP_TX_PROFILE_REVOKE,
		CXI_OP_TX_PROFILE_GET_INFO,

		CXI_OP_TX_PROFILE_ADD_AC_ENTRY,
		CXI_OP_TX_PROFILE_REMOVE_AC_ENTRY,
		CXI_OP_TX_PROFILE_GET_AC_ENTRY_IDS,
		CXI_OP_TX_PROFILE_GET_AC_ENTRY_DATA_BY_ID,
		CXI_OP_TX_PROFILE_GET_AC_ENTRY_ID_BY_DATA,
		CXI_OP_TX_PROFILE_GET_AC_ENTRY_ID_BY_USER,

		CXI_OP_CP_MODIFY,
		CXI_OP_SVC_SET_EXCLUSIVE_CP,
		CXI_OP_SVC_GET_EXCLUSIVE_CP,
		CXI_OP_SVC_ENABLE,
		CXI_OP_SVC_SET_VNI_RANGE,
		CXI_OP_SVC_GET_VNI_RANGE,

		CXI_OP_MAX,
};

struct cxi_common_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
};

struct cxi_lni_alloc_cmd {
	enum cxi_command_opcode op;
	unsigned int svc_id;
	void __user  *resp;
};

struct cxi_lni_alloc_resp {
	unsigned int lni;
};

struct cxi_lni_free_cmd {
	enum cxi_command_opcode op;
	unsigned int lni;
};

struct cxi_domain_reserve_cmd {
	enum cxi_command_opcode op;
	void __user  *resp;

	unsigned int lni;
	unsigned int vni;
	unsigned int pid;
	unsigned int count;
};

struct cxi_domain_reserve_resp {
	unsigned int pid;
};

struct cxi_domain_alloc_cmd {
	enum cxi_command_opcode op;
	void __user  *resp;

	unsigned int lni;
	unsigned int vni;
	unsigned int pid;
};

struct cxi_domain_alloc_resp {
	unsigned int domain;
	unsigned int vni;
	unsigned int pid;
};

struct cxi_domain_free_cmd {
	enum cxi_command_opcode op;
	unsigned int domain;
};

struct cxi_ct_alloc_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	/* LNI to associate with the counting event. */
	unsigned int lni;

	/* User writeback buffer used by counting event. */
	struct c_ct_writeback __user *wb;
};

struct cxi_ct_wb_update_cmd {
	enum cxi_command_opcode op;
	unsigned int ctn;

	/* User writeback buffer used by counting event. */
	struct c_ct_writeback __user *wb;
};

struct cxi_ct_alloc_resp {
	/* Allocated counting event number. */
	unsigned int ctn;
	struct cxi_mminfo doorbell;
};

struct cxi_ct_free_cmd {
	enum cxi_command_opcode op;
	unsigned int ctn;
};

enum cxi_traffic_class {
	/* HRP traffic classes. */
	CXI_TC_DEDICATED_ACCESS,
	CXI_TC_LOW_LATENCY,
	CXI_TC_BULK_DATA,
	CXI_TC_BEST_EFFORT,

	/* Ethernet specific traffic class. */
	CXI_TC_ETH,
	CXI_TC_MAX,
};

enum cxi_eth_traffic_class {
	CXI_ETH_TC1 = CXI_TC_MAX,
	CXI_ETH_SHARED,
	CXI_ETH_TC2,
};

enum cxi_traffic_class_type {
	CXI_TC_TYPE_DEFAULT,
	CXI_TC_TYPE_HRP,
	CXI_TC_TYPE_RESTRICTED,
	CXI_TC_TYPE_COLL_LEAF,
	CXI_TC_TYPE_MAX,
};

static const char * const cxi_tc_strs[] = {
	[CXI_TC_DEDICATED_ACCESS]	= "DEDICATED_ACCESS",
	[CXI_TC_LOW_LATENCY]		= "LOW_LATENCY",
	[CXI_TC_BULK_DATA]		= "BULK_DATA",
	[CXI_TC_BEST_EFFORT]		= "BEST_EFFORT",
	[CXI_TC_ETH]			= "ETH",
};

static const char * const cxi_tc_type_strs[] = {
	[CXI_TC_TYPE_DEFAULT] = "DEFAULT",
	[CXI_TC_TYPE_HRP] = "HRP",
	[CXI_TC_TYPE_RESTRICTED] = "RESTRICTED",
	[CXI_TC_TYPE_COLL_LEAF] = "COLL_LEAF",
};

static inline const char *cxi_tc_to_str(enum cxi_traffic_class tc)
{
	if (tc >= CXI_TC_DEDICATED_ACCESS && tc < CXI_TC_MAX)
		return cxi_tc_strs[tc];
	return "(invalid)";
}

static inline
const char *cxi_tc_type_to_str(enum cxi_traffic_class_type tc_type)
{
	if (tc_type >= CXI_TC_TYPE_DEFAULT && tc_type < CXI_TC_TYPE_MAX)
		return cxi_tc_type_strs[tc_type];
	return "(invalid)";
}

/* Helper structure for mapping multicast addresses to a PtlTE */
union cxi_pte_map_offset {
	struct {
		unsigned int mcast_id:C_DFA_MULTICAST_ID_BITS;
		unsigned int mcast_pte_index:C_DFA_INDEX_EXT_BITS;
	};
	unsigned int uintval;
};

/* CXI CQ status update policies. Policies are listed from most aggressive to
 * least.
 *
 * CXI_CQ_UPDATE_ALWAYS: Update when the hardware read pointer advances.
 * CXI_CQ_UPDATE_HIGH_FREQ_EMPTY: Update at a high frequency and when the CQ is
 * empty.
 * CXI_CQ_UPDATE_LOW_FREQ_EMPTY: Update at a low frequency and when the CQ is
 * empty.
 * CXI_CQ_UPDATE_LOW_FREQ: Update at a low frequency.
 */
enum cxi_cq_update_policy {
	CXI_CQ_UPDATE_ALWAYS,
	CXI_CQ_UPDATE_HIGH_FREQ_EMPTY,
	CXI_CQ_UPDATE_LOW_FREQ_EMPTY,
	CXI_CQ_UPDATE_LOW_FREQ,
};

struct cxi_tc_cfg_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	/* TC Label (Key) */
	unsigned int tc;

	/* Restricted packet HRP */
	bool hrp;

	/* Restricted DSCP */
	unsigned int rdscp;

	/* Unrestricted DSCP */
	unsigned int udscp;

	/* OCU SET Index */
	unsigned int ocu_set_idx;

	/* CQ specific TC value. */
	unsigned int cq_tc;
};

struct cxi_tc_cfg_resp {
	/* Traffic Class ID */
	unsigned int id;
};

struct cxi_tc_clear_cmd {
	enum cxi_command_opcode op;

	/* Traffic Class ID */
	unsigned int id;
};

struct cxi_cp_alloc_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	/* LNI to associate with the CP. */
	unsigned int lni;

	/* VNI to associate with the CP. */
	unsigned int vni;

	/* Traffic class to associate with the CP. */
	enum cxi_traffic_class tc;

	/* Traffic class type to associated with the CP. */
	enum cxi_traffic_class_type tc_type;
};

struct cxi_cp_alloc_resp {
	unsigned int cp_hndl;
	unsigned int lcid;
};

struct cxi_cp_free_cmd {
	enum cxi_command_opcode op;
	unsigned int cp_hndl;
};

struct cxi_cp_modify_cmd {
	enum cxi_command_opcode op;
	unsigned int cp_hndl;

	/* VNI to associate with the CP. */
	unsigned int vni;
};

#define CXI_MAX_CQ_COUNT (1 << 16)

enum {
	/* CQ is allocated for the current user process. Driver use only. */
	CXI_CQ_USER = (1 << 0),

	/* Whether this is a transmit or target CQ */
	CXI_CQ_IS_TX = (1 << 1),

	/* Whether that CQ can send RAW ethernet packets. For TX CQ in
	 * privileged clients only.
	 */
	CXI_CQ_TX_ETHERNET = (1 << 2),

	/* If set, the TX queue is reserved for triggered commands. */
	CXI_CQ_TX_WITH_TRIG_CMDS= (1 << 3),
};

struct cxi_cq_alloc_opts {
	/* Requested number of 64-byte entries in the command
	 * queue. Up to CXI_MAX_CQ_COUNT.
	 */
	unsigned int count;

	/* CQ status update policy. */
	enum cxi_cq_update_policy policy;

	/* Statistics counter pool for the CQ. */
	unsigned int stat_cnt_pool;

	/* Flags controlling CQ behavior (See CXI_CQ_*). */
	uint32_t flags;

	/* Initial Local Communication ID, for transmit CQ only. */
	unsigned int lcid;

	/* Index of one of the 4 LPE credits defined in the CXI driver
	 * by the tg_threshold module parameter. Must be 0 to 3, with
	 * 0 being the default.
	 */
	unsigned int lpe_cdt_thresh_id;
};

struct cxi_cq_alloc_cmd {
	enum cxi_command_opcode op;
	void  __user *resp;

	unsigned int lni;  /* LNI to associate with the event queue */

	/* EQ handle to report CQ errors to. May be set to C_EQ_NONE. */
	unsigned int eq;

	struct cxi_cq_alloc_opts opts;
};

struct cxi_cq_alloc_resp {
	unsigned int cq;

	/* Number of 64 bytes entries in the queue */
	unsigned int count;

	/* mmap info for the array of commands and the write pointer
	 * for that CQ
	 */
	struct cxi_mminfo cmds;
	struct cxi_mminfo wp_addr;
};

struct cxi_cq_free_cmd {
	enum cxi_command_opcode op;
	unsigned int cq;
};

struct cxi_cq_ack_counter_resp {
	unsigned int ack_counter;
};

struct cxi_cq_ack_counter_cmd {
	enum cxi_command_opcode op;
	void  __user *resp;
	unsigned int cq;
};

enum cxi_atu_map_flags {
	CXI_MAP_PIN       = (1 << 0),
	CXI_MAP_ATS       = (1 << 1),
	CXI_MAP_WRITE     = (1 << 2),
	CXI_MAP_READ      = (1 << 3),
	CXI_MAP_FAULT     = (1 << 4),
	CXI_MAP_NOCACHE   = (1 << 5),
	CXI_MAP_USER_ADDR = (1 << 6),
	CXI_MAP_ATS_PT    = (1 << 7),
	CXI_MAP_ATS_DYN   = (1 << 8),
	CXI_MAP_DEVICE    = (1 << 9),
	CXI_MAP_UNUSED    = (1 << 10),
	CXI_MAP_ALLOC_MD  = (1 << 11),
	CXI_MAP_PREFETCH  = (1 << 12),
};

/* PTE allocation options */
struct cxi_pt_alloc_opts {
	uint64_t en_event_match         :  1;
	uint64_t clr_remote_offset      :  1;
	uint64_t en_flowctrl            :  1;
	uint64_t use_long_event         :  1;
	uint64_t lossless               :  1;
	uint64_t en_restricted_unicast_lm :  1;
	uint64_t use_logical            :  1;
	uint64_t is_matching            :  1;
	uint64_t do_space_check         :  1;
	uint64_t en_align_lm            :  1;
	uint64_t en_sw_hw_st_chng       :  1;
	uint64_t ethernet               :  1;
	uint64_t signal_invalid         :  1;
	uint64_t signal_underflow       :  1;
	uint64_t signal_overflow        :  1;
	uint64_t signal_inexact         :  1;
	uint64_t en_match_on_vni        :  1;
};

/* CXI Memory Descriptor */
struct cxi_md {
	__u64    iova;       /* aligned IO virtual address */
	__u64    va;         /* aligned virtual address */
	size_t   len;        /* aligned length */
	__u8     lac;        /* logical address context */
	int      page_shift; /* base page size */
	int      huge_shift; /* huge page size */
	unsigned int id;
};

/**
 * struct cxi_cp - Communication profile.
 *
 * @vni_pcp: Virtual Network Identifier or PCP
 * @tc: Best practice traffic class
 * @hrp: Restricted packet HRP enable
 * @lcid: Communication profile LCID
 */
struct cxi_cp {
	union {
		unsigned int vni;	/* Deprecated */
		unsigned int vni_pcp;
	};
	enum cxi_traffic_class tc;
	enum cxi_traffic_class_type tc_type;
	unsigned int lcid;
};

struct cxi_md_hints {
	int page_shift; /* base page size */
	int huge_shift; /* huge page size */
	int ptg_mode;
	bool ptg_mode_valid;
	int dmabuf_fd;
	unsigned long dmabuf_offset;
	bool dmabuf_valid;
};

#define CXI_VA_TO_IOVA(_md, _va) \
		((_md)->iova + ((__u64)(_va) - (_md)->va))
#define CXI_IOVA_TO_VA(_md, _iova) \
		((_md)->va + ((__u64)(_iova) - (_md)->iova))
#define CXI_MD_CONTAINS(_md, _va, _len) \
		((_va) >= (void *)(_md)->va && \
		 (_va) + (_len) <= (void *)(_md)->va + (_md)->len)

struct cxi_atu_map_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int lni;
	__u64 va;
	__u64 len;
	__u32 flags;

	struct cxi_md_hints hints;
};

struct cxi_atu_map_resp {
	unsigned int id;
	struct cxi_md md;
};

struct cxi_atu_unmap_cmd {
	enum cxi_command_opcode op;

	unsigned int id;
};

struct cxi_atu_update_md_cmd {
	enum cxi_command_opcode op;

	unsigned int id;
	__u64 va;
	__u64 len;
	__u32 flags;
};

enum {
	/* EQ is allocated for the current user process. Driver use only. */
	CXI_EQ_USER = (1 << 0),

	/* Configure the EQ buffer for passthrough (no address translation).
	 * When using passthrough, the queue buffer provided must be backed by
	 * contiguous physical pages and the MD is ignored. When using
	 * passthrough, physical pages backing the queue buffer are pinned.
	 * Using passthrough can alleviate IOMMU cache pressure and eliminate
	 * translation overhead for event writes.
	 *
	 * Kernel threads must use kmalloc'ed addresses. User processes have
	 * limited visibility into how virtual addresses are mapped. Using a
	 * single-page sized queue is one way to guarantee that a process'
	 * queue buffer is contiguous. A mapped hugepage may be used to support
	 * passthrough with large queue buffers.
	 */
	CXI_EQ_PASSTHROUGH = (1 << 1),

	/* Force all initiator events to use the long event format. */
	CXI_EQ_INIT_LONG = (1 << 2),

	/* Force all target events to use the long event format. */
	CXI_EQ_TGT_LONG = (1 << 3),

	/* By default, an EQ generates a status write-back when one or more
	 * events are dropped. Disable status write-backs when events are
	 * dropped.
	 */
	CXI_EQ_DROP_STATUS_DISABLE = (1 << 4),

	/* Enable EQ timestamp events. Timestamp events are periodically
	 * inserted into event queues, providing a means to bound the time
	 * interval during which other events arrive in the queue.
	 */
	CXI_EQ_TIMESTAMP_EVENTS = (1 << 5),

	/* Disable EQ event combining. */
	CXI_EQ_EC_DISABLE = (1 << 6),

	/* EQ is to be registered with the PCT block */
	CXI_EQ_REGISTER_PCT = (1 << 7),
};

struct cxi_eq_attr {
	/* Pointer to the event queue buffer. */
	void *queue;

	/* Length of the event queue buffer. */
	size_t queue_len;

	/* Flags controlling EQ behavior (See CXI_EQ_*). */
	uint64_t flags;

	/* An EQ may be configured to generate a status write-back when the
	 * event queue reaches a defined fill level. Status information lets
	 * applications better understand the current condition of the EQ.
	 *
	 * An EQ may have up to four fill level status update thresholds
	 * enabled. The first threshold is defined as a base percentage.
	 * Subsequent thresholds are defined as the base percentage minus the
	 * Nth multiple of a percentage delta.  Given a base of 95 and delta of
	 * 15, the four thresholds are: 95%, 80%, 65% and 50% filled.
	 *
	 * Each EQ may specify the number of enabled thresholds, N, where only
	 * thresholds 0-(N-1) are enabled.  For N of 2, an EQ will trigger a
	 * fill level status update when it becomes 80% and 95% filled.
	 */

	/* Status threshold base percentage. */
	unsigned int status_thresh_base;

	/* Status threshold percentage delta. */
	unsigned int status_thresh_delta;

	/* Count of enabled status thresholds. */
	unsigned int status_thresh_count;

	/* EQ event combining buffer expiration delay in nanoseconds. Specify
	 * 0 to use the default delay. Provide the CXI_EQ_EC_DISABLE flag to
	 * disable event combining.
	 */
	unsigned int ec_delay;

	/* CPU affinity hint used for interrupt generation. */
	unsigned int cpu_affinity;

	/* Number of event slots which cannot be used by LPE for incoming
	 * messages.
	 */
	unsigned int reserved_slots;
};

#define CXI_MD_NONE 0
#define CXI_WAIT_NONE 0

struct cxi_eq_alloc_cmd {
	enum cxi_command_opcode op;
	void  __user *resp;

	unsigned int lni; /* LNI to associate with the event queue */

	 /* Memory Descriptor covering event queue buffer. CXI_MD_NONE
	  * for none.
	  */
	unsigned int queue_md;

	/* Event wait object handle. CXI_WAIT_NONE for none. */
	unsigned int event_wait;

	/* Status update wait object handle. CXI_WAIT_NONE for none. */
	unsigned int status_wait;

	struct cxi_eq_attr attr;
};

struct cxi_eq_alloc_resp {
	unsigned int eq; /* event queue handle */
	struct cxi_mminfo csr; /* mmap info for the event queue CSRs */
};

struct cxi_eq_free_cmd {
	enum cxi_command_opcode op;
	unsigned int eq; /* event queue handle to free */
};

struct cxi_eq_resize_cmd {
	enum cxi_command_opcode op;

	/* EQ to resize */
	unsigned int eq_hndl;

	/* Pointer to the new event queue buffer. */
	void *queue;

	/* New event queue buffer length in bytes. */
	size_t queue_len;

	/* Memory Descriptor covering the new event queue buffer. */
	unsigned int queue_md;
};

struct cxi_eq_resize_complete_cmd {
	enum cxi_command_opcode op;

	unsigned int eq_hndl; /* EQ to resize */
};

/* PtlTE operations */
struct cxi_pte_alloc_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int lni_hndl;
	unsigned int evtq_hndl;
	struct cxi_pt_alloc_opts opts;
};

struct cxi_pte_alloc_resp {
	unsigned int pte_number;
};

struct cxi_pte_free_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int pte_number;
};

struct cxi_pte_map_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int pte_number;
	unsigned int domain_hndl;
	unsigned int pid_offset;
	bool is_multicast;
};

struct cxi_pte_map_resp {
	unsigned int pte_index;
};

struct cxi_pte_unmap_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int pte_index;
};

struct cxi_pte_le_invalidate_cmd {
	enum cxi_command_opcode op;
	unsigned int pte_index;
	unsigned int buffer_id;
	enum c_ptl_list list;
};

struct cxi_pte_status {
	__u32 drop_count;
	__u8 state;
	__u16 les_reserved;
	__u16 les_allocated;
	__u16 les_max;

	__u64 __user *ule_offsets;
	__u16 ule_count;
};

struct cxi_pte_status_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int pte_index;
};

struct cxi_pte_status_resp {
	struct cxi_pte_status status;
};

struct cxi_pte_transition_sm_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int pte_index;
	unsigned int drop_count;
};

struct cxi_wait_alloc_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int lni;

};

struct cxi_wait_alloc_resp {
	unsigned int client_id;
	unsigned int wait;
};

struct cxi_wait_free_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int wait;
};

struct cxi_map_csrs_cmd {
	enum cxi_command_opcode op;
	void  __user *resp;
};

struct cxi_map_csrs_resp {
	struct cxi_mminfo csr; /* mmap info for the CSRs */
};

enum cxi_rsrc_type {
	CXI_RSRC_TYPE_PTE,
	CXI_RSRC_TYPE_TXQ,
	CXI_RSRC_TYPE_TGQ,
	CXI_RSRC_TYPE_EQ,
	CXI_RSRC_TYPE_CT,
	CXI_RSRC_TYPE_LE,
	CXI_RSRC_TYPE_TLE,
	CXI_RSRC_TYPE_AC,

	CXI_RSRC_TYPE_MAX,
};

static const char * const cxi_rsrc_type_strs[] = {
	[CXI_RSRC_TYPE_PTE] = "PTEs",
	[CXI_RSRC_TYPE_TXQ] = "TXQs",
	[CXI_RSRC_TYPE_TGQ] = "TGQs",
	[CXI_RSRC_TYPE_EQ] = "EQs",
	[CXI_RSRC_TYPE_CT] = "CTs",
	[CXI_RSRC_TYPE_LE] = "LEs",
	[CXI_RSRC_TYPE_TLE] = "TLEs",
	[CXI_RSRC_TYPE_AC] = "ACs",
};

static inline
const char *cxi_rsrc_type_to_str(enum cxi_rsrc_type rsrc_type)
{
	if (rsrc_type >= CXI_RSRC_TYPE_PTE && rsrc_type < CXI_RSRC_TYPE_MAX)
		return cxi_rsrc_type_strs[rsrc_type];
	return "(invalid)";
}

struct cxi_limits {
	uint16_t max;
	uint16_t res;
};

struct cxi_rsrc_limits {
	union {
		struct {
			struct cxi_limits ptes;
			struct cxi_limits txqs;
			struct cxi_limits tgqs;
			struct cxi_limits eqs;
			struct cxi_limits cts;
			struct cxi_limits les;
			struct cxi_limits tles;
			struct cxi_limits acs;
		};
		struct cxi_limits type[CXI_RSRC_TYPE_MAX];
	};
};

struct cxi_rsrc_use {
	/* SVC_ID = 0 is not a valid svc_id. But will be used to indicate total
	 * resources used on the device
	 */
	unsigned int svc_id;
	unsigned int tle_pool_id;
	uint16_t in_use[CXI_RSRC_TYPE_MAX];
};

enum cxi_svc_member_type {
	CXI_SVC_MEMBER_IGNORE,
	CXI_SVC_MEMBER_UID,
	CXI_SVC_MEMBER_GID,

	CXI_SVC_MEMBER_MAX,
};

struct cxi_svc_fail_info {
	/* If a reservation was requested for a CXI_RSRC_TYPE_X and allocation
	 * failed, its entry in this array will reflect how many of said
	 * resource were actually available to reserve.
	 */
	uint16_t rsrc_avail[CXI_RSRC_TYPE_MAX];

	/* True if relevant pool ids were requested, but none were available. */
	bool no_le_pools;
	bool no_tle_pools;
	bool no_cntr_pools;
};

#define CXI_SVC_MAX_MEMBERS 2
#define CXI_SVC_MAX_VNIS 4
#define CXI_DEFAULT_SVC_ID 1

struct cxi_svc_desc {
	uint8_t
		/* Limit access to member processes */
		restricted_members:1,

		/* Limit access to defined VNIs */
		restricted_vnis:1,

		/* Limit access to defined TCs */
		restricted_tcs:1,

		/* Limit access to resources */
		resource_limits:1,

		/* Whether a service should be enabled.
		 * Services are enabled by default upon creation.
		 */
		enable:1,

		/* Differentiates system and user services */
		is_system_svc:1,

		/* Counter Pool ID */
		cntr_pool_id:2;

	/* How many VNIs provided by the user are valid.
	 * Must be non-zero if restricted_vnis is true.
	 */
	uint8_t num_vld_vnis;

	/* VNIs allowed by this service */
	uint16_t vnis[CXI_SVC_MAX_VNIS];

	bool tcs[CXI_TC_MAX];

	struct {
		union cxi_svc_member {
			__kernel_uid_t  uid;
			__kernel_gid_t  gid;
		} svc_member;
		enum cxi_svc_member_type type;
	} members[CXI_SVC_MAX_MEMBERS];

	struct cxi_rsrc_limits limits;

	unsigned int svc_id;
};

struct cxi_svc_rsrc_list_get_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int count;
	struct cxi_rsrc_use *rsrc_list;
};

struct cxi_svc_rsrc_list_get_resp {
	unsigned int count;
};

struct cxi_svc_rsrc_get_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int svc_id;
};

struct cxi_svc_rsrc_get_resp {
	struct cxi_rsrc_use rsrcs;
};

struct cxi_svc_list_get_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int count;
	struct cxi_svc_desc *svc_list;
};

struct cxi_svc_list_get_resp {
	unsigned int count;
};

struct cxi_svc_get_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	unsigned int svc_id;
};

struct cxi_svc_get_resp {
	struct cxi_svc_desc svc_desc;
};

struct cxi_svc_alloc_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	struct cxi_svc_desc svc_desc;
};

struct cxi_svc_alloc_resp {
	unsigned int svc_id;
	struct cxi_svc_fail_info fail_info;
};

struct cxi_svc_destroy_cmd {
	enum cxi_command_opcode op;
	unsigned int svc_id;
};

struct cxi_svc_update_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	struct cxi_svc_desc svc_desc;
};

struct cxi_svc_enable_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
	unsigned int svc_id;
	bool enable;
};

struct cxi_svc_lpr_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
	unsigned int svc_id;
	unsigned int lnis_per_rgid;
};

struct cxi_svc_set_exclusive_cp_cmd {
	enum cxi_command_opcode op;
	unsigned int svc_id;
	bool exclusive_cp;
};

struct cxi_svc_get_exclusive_cp_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
	unsigned int svc_id;
};

struct cxi_svc_get_value_resp {
	unsigned int value;
};

struct cxi_svc_vni_range_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
	unsigned int svc_id;
	unsigned int vni_min;
	unsigned int vni_max;
};

struct cxi_svc_get_vni_range_resp {
	unsigned int vni_min;
	unsigned int vni_max;
};

struct cxi_svc_update_resp {
	struct cxi_svc_fail_info fail_info;
};

struct cxi_inbound_wait_cmd {
	enum cxi_command_opcode op;
};

struct cxi_svc_get_exclusive_cp_resp {
	bool exclusive_cp;
};

/**
 * struct cxi_sbus_op_params - SBus operation parameters
 *
 * @req_data: Data for request
 * @data_addr: Data address within the SBus Receiver
 * @rx_addr: Address of destination SBus Receiver
 * @command: Command for request
 * @delay: time to wait after issuing the command, 0 to 100, in us
 * @timeout: timeout for the whole command, 1 to 5000, in ms
 * @poll_interval: polling interval for result, 1 to 1000, in ms
 */
struct cxi_sbus_op_params {
	__u32 req_data;
	__u8 data_addr;
	__u8 rx_addr;
	__u8 command;
	unsigned int delay;
	unsigned int timeout;
	unsigned int poll_interval;
};

struct cxi_sbus_op_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	struct cxi_sbus_op_params params;
};

struct cxi_sbus_op_resp {
	__u32 rsp_data;
	__u8 result_code;
	__u8 overrun;
};

struct cxi_sbus_op_reset_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
};

/**
 * struct cxi_serdes_op_cmd - SERDES operation parameters
 *
 * @serdes_sel:
 * @serdes_op:
 * @data:
 * @timeout: non-zero timeout for the whole command, in ms
 * @flags:
 */
struct cxi_serdes_op_cmd {
	enum cxi_command_opcode op;
	void __user *resp;

	uint64_t serdes_sel;
	uint64_t serdes_op;
	uint64_t data;
	int timeout;
	unsigned int flags;
};

struct cxi_serdes_op_resp {
	uint16_t result;
};

/* Versions and revisions of the chip */
enum cassini_version {
	CASSINI_1 = 0x10,
	CASSINI_2 = 0x20,

	CASSINI_1_0 = (CASSINI_1 | 0x01),
	CASSINI_1_1 = (CASSINI_1 | 0x02),

	CASSINI_2_0 = (CASSINI_2 | 0x01),
};

/* System info to determine if a system is mixed (C1 & C2)
 * or homogeneous (only C1 or only C2).
 */
enum system_type_identifier {
	CASSINI_MIX = 0,
	CASSINI_1_ONLY = 1,
	CASSINI_2_ONLY = 2,
};

struct cxi_properties_info {
	/* Physical NIC Address of the device. */
	union {
		uint32_t nic_addr; /* obsolete */
		uint32_t nid;
	};

	/* NID explicitly configured by user */
	bool nid_configured;

	/* Width of PID field */
	unsigned int pid_bits;

	/* Count of PIDs per VNI */
	unsigned int pid_count;

	/* Granularity of device PID space. */
	unsigned int pid_granule;

	/* min_free shift value */
	unsigned int min_free_shift;

	/* Source PTE index for offloaded rendezvous Gets */
	unsigned int rdzv_get_idx;

	/* Device resource limit info */
	struct cxi_rsrc_limits rsrcs;

	unsigned int device_rev;
	unsigned int device_proto;
	unsigned int device_platform;

	/* Hardware version (1.0, 1.1, 2.0, ...) */
	enum cassini_version cassini_version;

	/* Rendezvous get hardware enable state */
	bool rdzv_get_en;

	/* NIC AMO operation opcode being remapped to PCIe fetch add. */
	int amo_remap_to_pcie_fadd;
};

struct cxi_get_dev_properties_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
};

struct cxi_eq_adjust_reserved_fc_cmd {
	enum cxi_command_opcode op;
	unsigned int eq_hndl;
	int value;
	void __user *resp;
};

struct cxi_eq_adjust_reserved_fc_resp {
	int reserved_fc;
};

struct cxi_dev_info_use {
	unsigned int nid;
	unsigned int pid_bits;
	unsigned int pid_count;
	unsigned int pid_granule;
	unsigned int min_free_shift;
	unsigned int rdzv_get_idx;
	unsigned int vendor_id;
	unsigned int device_id;
	unsigned int device_rev;
	unsigned int device_proto;
	unsigned int device_platform;
	unsigned int pct_eq;
	int uc_nic;

	size_t link_mtu;
	size_t link_speed;
	uint8_t link_state;
	uint16_t num_ptes;
	uint16_t num_txqs;
	uint16_t num_tgqs;
	uint16_t num_eqs;
	uint16_t num_cts;
	uint16_t num_acs;
	uint16_t num_tles;
	uint16_t num_les;

	char fru_description[16];

	enum cassini_version cassini_version;

	enum system_type_identifier system_type_identifier;
};

struct cxi_dev_info_get_cmd {
	enum cxi_command_opcode op;
	void __user *resp;
};

struct cxi_dev_info_get_resp {
	struct cxi_dev_info_use devinfo;
};

#define CXIERR_GENL_FAMILY_NAME "cxierr"
#define CXIERR_GENL_VERSION	1
#define CXIERR_GENL_MCAST_GROUP_NAME "cxierr_mc_group"

enum {
	CXIERR_ERR_UNSPEC,
	CXIERR_ERR_MSG,
};

enum cxierr_nldev_attr {
	CXIERR_GENL_ATTR_UNSPEC, /* always first */

	CXIERR_GENL_ATTR_DEV_NUM,
	CXIERR_GENL_ATTR_CSR_FLG,
	CXIERR_GENL_ATTR_BIT,

	CXIERR_GENL_ATTR_MAX	/* always last */
};

/* Expose driver default PCT Timings that will be
 * referenced by the Retry Handler
 */
#define C1_DEFAULT_SPT_TIMEOUT_EPOCH 30
#define C1_DEFAULT_SCT_IDLE_EPOCH 15
#define C1_DEFAULT_SCT_CLOSE_EPOCH 30
#define C1_DEFAULT_TCT_TIMEOUT_EPOCH 35

/* Check whether Cassini is of a specific revision or version.
 *
 * For instance, CASSINI_1_0 will match only for Cassini 1.0, while
 * CASSINI_1 will match both Cassini 1.0 and 1.1.
 */
static inline bool cassini_version(const struct cxi_properties_info *prop,
				   enum cassini_version version)
{
	return (prop->cassini_version & version) == version;
}

#endif	/* __UAPI_CXI_ABI_H */
