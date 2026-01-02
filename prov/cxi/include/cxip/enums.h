/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_ENUMS_H_
#define _CXIP_ENUMS_H_


/* All enum type definitions */
/* Included first because many structs embed enum fields */

enum cxip_ats_mlock_mode {
	CXIP_ATS_MLOCK_OFF,
	CXIP_ATS_MLOCK_CACHE,
	CXIP_ATS_MLOCK_ALL,
};

enum cxip_llring_mode {
	CXIP_LLRING_NEVER,
	CXIP_LLRING_IDLE,
	CXIP_LLRING_ALWAYS,
};

enum cxip_ep_ptle_mode {
	CXIP_PTLTE_HARDWARE_MODE,
	CXIP_PTLTE_DEFAULT_MODE = CXIP_PTLTE_HARDWARE_MODE,
	CXIP_PTLTE_SOFTWARE_MODE,
	CXIP_PTLTE_HYBRID_MODE,
};

enum cxip_rdzv_proto {
	CXIP_RDZV_PROTO_DEFAULT,	/* unrestricted gets */
	CXIP_RDZV_PROTO_ALT_READ,	/* restricted gets */
	CXIP_RDZV_PROTO_ALT_WRITE,	/* restricted puts */
};

enum cxip_mr_target_ordering {
	/* Sets MR target ordering based on message and target RMA ordering
	 * options.
	 */
	MR_ORDER_DEFAULT,

	/* Force ordering to always be strict. */
	MR_ORDER_STRICT,

	/* Force ordering to always be relaxed. */
	MR_ORDER_RELAXED,
};

enum cxip_le_type {
	CXIP_LE_TYPE_RX = 0,	/* RX data LE */
	CXIP_LE_TYPE_ZBP,	/* Zero-byte Put control message LE. Used to
				 * exchange data in the EQ header_data and
				 * match_bits fields. Unexpected headers are
				 * disabled.
				 */
};

enum cxip_ctrl_le_type {
	CXIP_CTRL_LE_TYPE_MR = 0,	/* Memory Region LE */
	CXIP_CTRL_LE_TYPE_CTRL_MSG,	/* Control Message LE */
};

enum cxip_ctrl_msg_type {
	CXIP_CTRL_MSG_FC_NOTIFY = 0,
	CXIP_CTRL_MSG_FC_RESUME,
	CXIP_CTRL_MSG_ZB_DATA,
	CXIP_CTRL_MSG_ZB_DATA_RDMA_LAC,
};

enum cxip_req_type {
	CXIP_REQ_RMA,
	CXIP_REQ_AMO,
	CXIP_REQ_OFLOW,
	CXIP_REQ_RECV,
	CXIP_REQ_SEND,
	CXIP_REQ_RDZV_SRC,
	CXIP_REQ_SEARCH,
	CXIP_REQ_COLL,
	CXIP_REQ_RBUF,
};

enum cxip_rxc_state {
	/* Initial state of an RXC. All user posted receives are rejected until
	 * the RXC has been enabled.
	 *
	 * Note that an RXC can be transitioned from any state into
	 * RXC_DISABLED.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: User has successfully enabled the RXC.
	 * RXC_ENABLED_SOFTWARE: User has successfully initialized the RXC
	 * in a software only RX matching mode.
	 */
	RXC_DISABLED = 0,

	/* User posted receives are matched against the software unexpected
	 * list before being offloaded to hardware. Hardware matches against
	 * the corresponding PtlTE priority and overflow list.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL: Several scenarios can initiate this state
	 * change.
	 *    1. Hardware fails to allocate an LE for an unexpected message
	 *    or a priority list LE append fails, and hybrid mode is not
	 *    enabled. Hardware transitions the PtlTE from enabled to disabled.
	 *    2. Hardware fails to allocate an LE during an overflow list
	 *    append. The PtlTE remains in the enabled state but appends to
	 *    the overflow list are disabled. Software manually disables
	 *    the PtlTE.
	 *    3. Hardware fails to successfully match on the overflow list.
	 *    Hardware automatically transitions the PtlTE from enabled to
	 *    disabled.
	 * RXC_ONLOAD_FLOW_CONTROL_REENABLE: Several scenarios can initiate
	 * it this state change:
	 *    1. The hardware EQ is full, hardware transitions the PtlTE from
	 *    enabled/software managed to disabled to recover drops, but it
	 *    can re-enable if an LE resource is not recovered.
	 *    2. Running "hardware" RX match mode and matching failed because
	 *    the overflow list buffers were full. Hardware transitions the
	 *    PtlTE from enabled to disabled. The overflow list must be
	 *    replenished and processing can continue if an LE resource is not
	 *    recovered.
	 *    3. Running "hybrid" or "software" RX match mode and a message
	 *    is received, but there is not a buffer available on the request
	 *    list. Hardware transitions the PtlTE from software managed to
	 *    disabled. The request list must be replenished and processing
	 *    can continue if an LE resource is not recovered.
	 * RXC_PENDING_PTLTE_SOFTWARE_MANAGED: When the provider is configured
	 * to run in "hybrid" RX match mode and hardware fails to allocate an
	 * LE for an unexpected message match or an priority list append fails.
	 * Hardware will automatically transition the PtlTE from enabled to
	 * software managed and onload of UX messages will be initiated.
	 */
	RXC_ENABLED,

	/* The NIC has initiated a transition to software managed EP matching.
	 *
	 * Software must onload/reonload the hardware unexpected list while
	 * creating a pending unexpected list from entries received on the PtlTE
	 * request list. Any in flight appends will fail and be added to
	 * a receive replay list, further attempts to post receive operations
	 * will return -FI_EAGAIN. When onloading completes, the pending
	 * UX list is appended to the onloaded UX list and then failed appends
	 * are replayed prior to enabling the posting of receive operations.
	 *
	 * Validate state changes:
	 * RXC_ENABLED_SOFTWARE: The HW to SW transition onloading has
	 * completed and the onloaded and pending request UX list have been
	 * combined.
	 */
	RXC_PENDING_PTLTE_SOFTWARE_MANAGED,

	/* Executing as a software managed PtlTE either due to hybrid
	 * transition from hardware or initial startup in software
	 * RX matching mode.
	 *
	 * Validate state changes:
	 * RXC_PENDING_PTLTE_HARDWARE: TODO: When able, software may
	 * initiate a transition from software managed mode back to
	 * fully offloaded operation.
	 * RXC_ONLODAD_FLOW_CONTROL_REENABLE: Hardware was unable to match
	 * on the request list or the EQ is full. Hardware has disabled the
	 * PtlTE initiating flow control. Operation can continue if LE
	 * resources are not recovered as long as request buffers can be
	 * replenished.
	 */
	RXC_ENABLED_SOFTWARE,

	/* TODO: Hybrid RX match mode PtlTE is transitioning from software
	 * managed operation back to fully offloaded operation.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: Hybrid software managed PtlTE successfully
	 * transitions back to fully offloaded operation.
	 * RXC_ENABLED_SOFTWARE: Hybrid software managed PtlTE was
	 * not able to transition to fully offloaded operation.
	 */
	RXC_PENDING_PTLTE_HARDWARE,

	/* Software has encountered a condition which requires manual transition
	 * of the PtlTE into disable. This state change occurs when a posted
	 * receive could not be appended due to LE exhaustion and software
	 * managed EP PtlTE operation has been disabled or is not possible.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL: PtlTE disabled event has successfully been
	 * received and onloading can begin.
	 */
	RXC_PENDING_PTLTE_DISABLE,

	/* Flow control has occurred and the PtlTE is disabled. Software is
	 * in the process of onloading the hardware unexpected headers to free
	 * up LEs. User posted receives are matched against the software
	 * unexpected list. If a match is not found on the software unexpected
	 * list, -FI_EAGAIN is returned to the user. Hardware matching is
	 * disabled.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL_REENABLE: An unexpected list entry matched
	 * a user posted receive, the search and delete command free a
	 * unexpected list entry, or a transition to software managed EP is
	 * occuring.
	 */
	RXC_ONLOAD_FLOW_CONTROL,

	/* PtlTE is in the same state as RXC_ONLOAD_FLOW_CONTROL, but the RXC
	 * should attempt to be re-enabled.
	 *
	 * Validate state changes:
	 * RXC_FLOW_CONTROL: Onloading of the unexpected headers has completed.
	 */
	RXC_ONLOAD_FLOW_CONTROL_REENABLE,

	/* Software is performing sideband communication to recover the dropped
	 * messages. User posted receives are matched against the software
	 * unexpected list. If a match is not found on the software unexpected
	 * list, -FI_EAGAIN is returned to the user. Hardware matching is
	 * disabled.
	 *
	 * If an append fails due to RC_NO_SPACE while in the RXC_FLOW_CONTROL
	 * state, hardware LEs are exhausted and no more LEs can be freed by
	 * onloading unexpected headers into software. This is a fatal event
	 * which requires software endpoint mode to workaround.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: Sideband communication is complete and PtlTE is
	 * successfully re-enabled.
	 * RXC_SOFTWARE_MANAGED: When executing in "hybrid" or "software"
	 * RX match mode and processing has requested to re-enable as a
	 * software managed EP.
	 */
	RXC_FLOW_CONTROL,
};

enum cxip_mr_state {
	CXIP_MR_DISABLED = 1,
	CXIP_MR_ENABLED,
	CXIP_MR_LINKED,
	CXIP_MR_UNLINKED,
	CXIP_MR_LINK_ERR,
};

enum cxip_coll_redtype {
	REDTYPE_BYT,
	REDTYPE_INT,
	REDTYPE_FLT,
	REDTYPE_IMINMAX,
	REDTYPE_FMINMAX,
	REDTYPE_REPSUM
};

enum cxip_coll_state {
	CXIP_COLL_STATE_NONE,
	CXIP_COLL_STATE_READY,
	CXIP_COLL_STATE_FAULT,
};

typedef enum cxip_coll_rc {
	CXIP_COLL_RC_SUCCESS = 0,		// good
	CXIP_COLL_RC_FLT_INEXACT = 1,		// result was rounded
	CXIP_COLL_RC_FLT_OVERFLOW = 3,		// result too large to represent
	CXIP_COLL_RC_FLT_INVALID = 4,		// op was signalling NaN, or
						// infinities subtracted
	CXIP_COLL_RC_REP_INEXACT = 5,		// reproducible sum was rounded
	CXIP_COLL_RC_INT_OVERFLOW = 6,		// reproducible sum overflow
	CXIP_COLL_RC_CONTR_OVERFLOW = 7,	// too many contributions seen
	CXIP_COLL_RC_OP_MISMATCH = 8,		// conflicting opcodes
	CXIP_COLL_RC_TX_FAILURE = 9,		// internal send error
	CXIP_COLL_RC_RDMA_FAILURE = 10,		// leaf rdma read error
	CXIP_COLL_RC_RDMA_DATA_FAILURE = 11,	// leaf rdma read data misc
	CXIP_COLL_RC_MAX = 12
} cxip_coll_rc_t;

enum curl_ops {
	CURL_GET,
	CURL_PUT,
	CURL_POST,
	CURL_PATCH,
	CURL_DELETE,
	CURL_MAX
};

enum cxip_amo_req_type {
	CXIP_RQ_AMO,
	CXIP_RQ_AMO_FETCH,
	CXIP_RQ_AMO_SWAP,
	CXIP_RQ_AMO_PCIE_FETCH,
	CXIP_RQ_AMO_LAST,
};

enum cxip_coll_trace_module {
	CXIP_TRC_CTRL,
	CXIP_TRC_ZBCOLL,
	CXIP_TRC_COLL_CURL,
	CXIP_TRC_COLL_PKT,
	CXIP_TRC_COLL_JOIN,
	CXIP_TRC_COLL_DEBUG,
	CXIP_TRC_TEST_CODE,
	CXIP_TRC_MAX
};

#endif /* _CXIP_ENUMS_H_ */
