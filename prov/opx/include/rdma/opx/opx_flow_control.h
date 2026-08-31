/*
 * Copyright (C) 2026 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _FI_PROV_OPX_FLOW_CONTROL_H_
#define _FI_PROV_OPX_FLOW_CONTROL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum opx_flow_control_mode {
	OPX_FLOW_CONTROL_MODE_NONE	  = 0,
	OPX_FLOW_CONTROL_MODE_RCCC	  = 1,
	OPX_FLOW_CONTROL_MODE_CONFORMANCE = 254
};

#define OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL)  OPX_FLOW_CONTROL_MODE_VALUE_(FLOW_CONTROL)
#define OPX_FLOW_CONTROL_MODE_VALUE_(FLOW_CONTROL) OPX_FLOW_CONTROL_MODE_##FLOW_CONTROL

enum opx_flow_control_result { OPX_FLOW_CONTROL_OK = 0, OPX_FLOW_CONTROL_DEFER = 1, OPX_FLOW_CONTROL_FAULT = 2 };

enum opx_flow_control_carrier {
	OPX_FLOW_CONTROL_CARRIER_UNSPECIFIED	   = 0,
	OPX_FLOW_CONTROL_CARRIER_RELIABILITY_PING  = 1,
	OPX_FLOW_CONTROL_CARRIER_RELIABILITY_ACK   = 2,
	OPX_FLOW_CONTROL_CARRIER_MSG_RZV_RTS	   = 3,
	OPX_FLOW_CONTROL_CARRIER_TAG_RZV_RTS	   = 4,
	OPX_FLOW_CONTROL_CARRIER_RMA_RTS	   = 5,
	OPX_FLOW_CONTROL_CARRIER_MEMORY_REGION_CTS = 6,
	OPX_FLOW_CONTROL_CARRIER_RMA_CTS	   = 7,
	OPX_FLOW_CONTROL_CARRIER_FENCE_CTS	   = 8,
	OPX_FLOW_CONTROL_CARRIER_VADDR_CTS	   = 9,
	OPX_FLOW_CONTROL_CARRIER_TID_VADDR_CTS	   = 10,
	OPX_FLOW_CONTROL_CARRIER_HFISVC_RZV_RTS	   = 11,
	OPX_FLOW_CONTROL_CARRIER_DPUT		   = 12,
	OPX_FLOW_CONTROL_CARRIER_REPLAYED_ORIGINAL = 13,
	OPX_FLOW_CONTROL_CARRIER_COUNT		   = 14
};

enum opx_flow_control_lifecycle {
	OPX_FLOW_CONTROL_UNINITIALIZED = 0,
	OPX_FLOW_CONTROL_IDLE,
	OPX_FLOW_CONTROL_ACTIVE,
	OPX_FLOW_CONTROL_RESETTING,
	OPX_FLOW_CONTROL_QUARANTINED,
	OPX_FLOW_CONTROL_FAILED
};

enum opx_flow_control_fault_scope {
	OPX_FLOW_CONTROL_FAULT_NONE = 0,
	OPX_FLOW_CONTROL_FAULT_FLOW,
	OPX_FLOW_CONTROL_FAULT_CONTEXT,
	OPX_FLOW_CONTROL_FAULT_PROCESS
};

enum opx_flow_control_packet_class {
	OPX_FLOW_CONTROL_PACKET_INJECT = 0,
	OPX_FLOW_CONTROL_PACKET_EAGER,
	OPX_FLOW_CONTROL_PACKET_MP_EAGER,
	OPX_FLOW_CONTROL_PACKET_RENDEZVOUS,
	OPX_FLOW_CONTROL_PACKET_RMA,
	OPX_FLOW_CONTROL_PACKET_ATOMIC,
	OPX_FLOW_CONTROL_PACKET_CLASS_COUNT
};

struct opx_flow_control_reservation_record;
struct opx_flow_control_wire_envelope;
struct opx_flow_control_rccc_tx_profile;
struct opx_flow_control_rccc_scheduler_profile;

struct opx_flow_control_rccc_feedback_counters {
	uint64_t newer;
	uint64_t duplicate;
	uint64_t stale;
	uint64_t ambiguous;
	uint64_t invalid;
};

struct fi_opx_ep;
struct fi_opx_hfi1_context;
struct fi_opx_reliability_tx_flow;
struct fi_opx_reliability_rx_flow;
union fi_opx_reliability_service_flow_key;
union opx_hfi1_packet_hdr;

#define OPX_FLOW_CONTROL_NONE_STATE_SIZE  0u
#define OPX_FLOW_CONTROL_STATE_ALIGNMENT  8u
#define OPX_FLOW_CONTROL_MODE_STATE_BYTES 112u
#define OPX_FLOW_CONTROL_STATE_SIZE	  128u

struct opx_flow_control_state {
	uint8_t	 lifecycle;
	uint8_t	 fault_class;
	uint8_t	 fault_scope;
	uint8_t	 flags;
	uint16_t transition_reason;
	uint16_t reserved;
	uint64_t transition_time_ns;
	uint64_t mode_state[OPX_FLOW_CONTROL_MODE_STATE_BYTES / sizeof(uint64_t)];
};

_Static_assert(OPX_FLOW_CONTROL_STATE_ALIGNMENT != 0u, "flow-control state alignment must be nonzero");
_Static_assert((OPX_FLOW_CONTROL_STATE_ALIGNMENT & (OPX_FLOW_CONTROL_STATE_ALIGNMENT - 1u)) == 0u,
	       "flow-control state alignment must be a power of two");
_Static_assert(OPX_FLOW_CONTROL_MODE_STATE_BYTES % sizeof(uint64_t) == 0u,
	       "flow-control mode state must use whole qwords");
_Static_assert(OPX_FLOW_CONTROL_STATE_SIZE == 16u + OPX_FLOW_CONTROL_MODE_STATE_BYTES,
	       "flow-control state size must match its fields");

_Static_assert(OPX_FLOW_CONTROL_MODE_NONE == 0, "flow-control NONE mode identity");
_Static_assert(OPX_FLOW_CONTROL_MODE_RCCC == 1, "flow-control RCCC mode identity");
_Static_assert(OPX_FLOW_CONTROL_MODE_CONFORMANCE == 254, "flow-control conformance mode identity");

_Static_assert(OPX_FLOW_CONTROL_OK == 0, "flow-control success result identity");
_Static_assert(OPX_FLOW_CONTROL_DEFER == 1, "flow-control defer result identity");
_Static_assert(OPX_FLOW_CONTROL_FAULT == 2, "flow-control fault result identity");

_Static_assert(OPX_FLOW_CONTROL_CARRIER_UNSPECIFIED == 0, "flow-control unspecified carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_RELIABILITY_PING == 1, "flow-control reliability PING carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_RELIABILITY_ACK == 2, "flow-control reliability ACK carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_MSG_RZV_RTS == 3, "flow-control message RTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_TAG_RZV_RTS == 4, "flow-control tagged RTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_RMA_RTS == 5, "flow-control RMA RTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_MEMORY_REGION_CTS == 6, "flow-control memory-region CTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_RMA_CTS == 7, "flow-control RMA CTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_FENCE_CTS == 8, "flow-control fence CTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_VADDR_CTS == 9, "flow-control vaddr CTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_TID_VADDR_CTS == 10, "flow-control TID vaddr CTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_HFISVC_RZV_RTS == 11, "flow-control HFISVC RTS carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_DPUT == 12, "flow-control DPUT carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_REPLAYED_ORIGINAL == 13, "flow-control replayed-original carrier identity");
_Static_assert(OPX_FLOW_CONTROL_CARRIER_COUNT == 14, "flow-control carrier count");

_Static_assert(OPX_FLOW_CONTROL_UNINITIALIZED == 0, "flow-control uninitialized lifecycle identity");
_Static_assert(OPX_FLOW_CONTROL_IDLE == 1, "flow-control idle lifecycle identity");
_Static_assert(OPX_FLOW_CONTROL_ACTIVE == 2, "flow-control active lifecycle identity");
_Static_assert(OPX_FLOW_CONTROL_RESETTING == 3, "flow-control resetting lifecycle identity");
_Static_assert(OPX_FLOW_CONTROL_QUARANTINED == 4, "flow-control quarantined lifecycle identity");
_Static_assert(OPX_FLOW_CONTROL_FAILED == 5, "flow-control failed lifecycle identity");

_Static_assert(OPX_FLOW_CONTROL_FAULT_NONE == 0, "flow-control no-fault scope identity");
_Static_assert(OPX_FLOW_CONTROL_FAULT_FLOW == 1, "flow-control flow-fault scope identity");
_Static_assert(OPX_FLOW_CONTROL_FAULT_CONTEXT == 2, "flow-control context-fault scope identity");
_Static_assert(OPX_FLOW_CONTROL_FAULT_PROCESS == 3, "flow-control process-fault scope identity");

_Static_assert(OPX_FLOW_CONTROL_PACKET_INJECT == 0, "flow-control inject packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_EAGER == 1, "flow-control eager packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_MP_EAGER == 2, "flow-control MP eager packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_RENDEZVOUS == 3, "flow-control rendezvous packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_RMA == 4, "flow-control RMA packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_ATOMIC == 5, "flow-control atomic packet-class identity");
_Static_assert(OPX_FLOW_CONTROL_PACKET_CLASS_COUNT == 6, "flow-control packet-class count");

_Static_assert(offsetof(struct opx_flow_control_state, lifecycle) == 0u, "flow-control lifecycle offset");
_Static_assert(offsetof(struct opx_flow_control_state, fault_class) == 1u, "flow-control fault-class offset");
_Static_assert(offsetof(struct opx_flow_control_state, fault_scope) == 2u, "flow-control fault-scope offset");
_Static_assert(offsetof(struct opx_flow_control_state, flags) == 3u, "flow-control flags offset");
_Static_assert(offsetof(struct opx_flow_control_state, transition_reason) == 4u,
	       "flow-control transition-reason offset");
_Static_assert(offsetof(struct opx_flow_control_state, reserved) == 6u, "flow-control reserved offset");
_Static_assert(offsetof(struct opx_flow_control_state, transition_time_ns) == 8u,
	       "flow-control transition-time offset");
_Static_assert(offsetof(struct opx_flow_control_state, mode_state) == 16u, "flow-control mode-state offset");
_Static_assert(sizeof(((struct opx_flow_control_state *) 0)->mode_state) == OPX_FLOW_CONTROL_MODE_STATE_BYTES,
	       "flow-control mode-state width");
_Static_assert(sizeof(struct opx_flow_control_state) == OPX_FLOW_CONTROL_STATE_SIZE, "flow-control state size");
_Static_assert(_Alignof(struct opx_flow_control_state) == OPX_FLOW_CONTROL_STATE_ALIGNMENT,
	       "flow-control state alignment");

_Static_assert(OPX_FLOW_CONTROL_NONE_STATE_SIZE == 0u, "flow-control NONE state-size contract");

#endif
