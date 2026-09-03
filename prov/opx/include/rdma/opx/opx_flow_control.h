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

/* Every _OFFSET macro below is a hard ABI mirror of a struct field layout
 * owned by opx_flow_control_rccc_private.h, consumed by inline fast-path
 * memcpy leaves in TUs that do not include the private header.  Each one is
 * pinned by a _Static_assert against the real offsetof() in the private
 * header, which is compiled unconditionally via opx_flow_control_rccc.c, so
 * any struct reorder fails the build rather than silently corrupting the
 * fast path.  Never change an offset here without its private-header assert. */
#define OPX_FLOW_CONTROL_NONE_STATE_SIZE			0u
#define OPX_FLOW_CONTROL_STATE_ALIGNMENT			8u
#define OPX_FLOW_CONTROL_MODE_STATE_BYTES			112u
#define OPX_FLOW_CONTROL_STATE_SIZE				128u
#define OPX_FLOW_CONTROL_RCCC_TX_AVAILABLE_CREDITS_OFFSET	12u
#define OPX_FLOW_CONTROL_RCCC_TX_HELD_CREDITS_OFFSET		16u
#define OPX_FLOW_CONTROL_RCCC_TX_DESIRED_WINDOW_OFFSET		24u
#define OPX_FLOW_CONTROL_RCCC_TX_BLOCK_REASON_OFFSET		26u
#define OPX_FLOW_CONTROL_RCCC_TX_FLAGS_OFFSET			27u
#define OPX_FLOW_CONTROL_RCCC_RX_GRANT_OFFSET			8u
#define OPX_FLOW_CONTROL_RCCC_RX_OBSERVED_OFFSET		12u
#define OPX_FLOW_CONTROL_RCCC_RX_RETIRED_ACCEPTED_OFFSET	20u
#define OPX_FLOW_CONTROL_RCCC_RX_SCHEDULER_STATE_OFFSET		24u
#define OPX_FLOW_CONTROL_RCCC_RX_FLAGS_OFFSET			25u
#define OPX_FLOW_CONTROL_RCCC_RX_PERMISSION_CONTRIBUTION_OFFSET 48u
#define OPX_FLOW_CONTROL_RCCC_RX_CONSUMED_EPOCH_OFFSET		108u
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_OUTSTANDING_OFFSET	0u
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_SHARED_RELEASE_OFFSET	8u
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_SHARED_LEDGER_OFFSET	16u
#define OPX_FLOW_CONTROL_RCCC_SCHEDULER_CONSUMED_TOTAL_OFFSET	24u

#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_REQUEST_PENDING	UINT8_C(0x01)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_RESETTING		UINT8_C(0x04)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_ACTIVITY_PENDING	UINT8_C(0x08)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_CAPACITY_REJECTED UINT8_C(0x20)
#define OPX_FLOW_CONTROL_RCCC_TX_FLAG_ON_WORK_LIST	UINT8_C(0x80)

#define OPX_FLOW_CONTROL_RCCC_RX_FLAG_RECLAIMED UINT8_C(0x01)

#define OPX_FLOW_CONTROL_RCCC_FAST_BLOCK_NONE	   0u
#define OPX_FLOW_CONTROL_RCCC_FAST_SCHEDULER_BOUND 2u

#define OPX_FLOW_CONTROL_RCCC_SERIAL_MASK	UINT32_C(0x00FFFFFF)
#define OPX_FLOW_CONTROL_RCCC_SERIAL_HALF_RANGE UINT32_C(0x00800000)

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

int						      opx_flow_control_process_acquire(void);
void						      opx_flow_control_process_release(void);
enum opx_flow_control_mode			      opx_flow_control_selected_mode(void);
int						      opx_flow_control_selected_state_size(size_t *state_size);
const struct opx_flow_control_rccc_tx_profile	     *opx_flow_control_rccc_tx_profile_get(void);
const struct opx_flow_control_rccc_scheduler_profile *opx_flow_control_rccc_scheduler_profile_get(void);
uint16_t					      opx_flow_control_rccc_control_headroom_get(void);
uint16_t					      opx_flow_control_rccc_max_control_emits_get(void);
const char					     *opx_flow_control_mode_name(enum opx_flow_control_mode mode);

int  opx_flow_control_context_init(struct fi_opx_hfi1_context *context);
void opx_flow_control_context_fini(struct fi_opx_hfi1_context *context);
int  opx_flow_control_endpoint_init(struct fi_opx_ep *opx_ep);
void opx_flow_control_endpoint_fini(struct fi_opx_ep *opx_ep);
int  opx_flow_control_tx_flow_init(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow,
				   const union fi_opx_reliability_service_flow_key *key);
void opx_flow_control_tx_flow_fini(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow);
enum opx_flow_control_result opx_flow_control_rccc_tx_admit(struct fi_opx_ep		      *opx_ep,
							    struct fi_opx_reliability_tx_flow *flow,
							    uint32_t			       requested_packets);
enum opx_flow_control_result opx_flow_control_rccc_tx_eager_admit(struct fi_opx_ep *opx_ep, uint32_t dlid,
								  uint16_t dst_subctxt_rx, uint8_t tx_index,
								  uint32_t requested_packets);
void opx_flow_control_rccc_tx_work_enqueue(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow);
void opx_flow_control_rccc_control_sweep(struct fi_opx_ep *opx_ep, uint64_t now_ns);
int  opx_flow_control_rx_flow_init(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow);
void opx_flow_control_rx_flow_fini(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow);
/* Called once per accepted governed-class PSN (original or replay) at each
 * reliability acceptance site; PSN-ordered acceptance (the next_psn advance)
 * guarantees exactly-once, so the release table needs no per-PSN dedup. */
void opx_flow_control_rx_first_accept_slow(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_rx_flow *flow,
					   uint32_t psn);

/* Single-packet reservations skip the record pool entirely: completion-side
 * accounting never touches the flow's credit terms (records only drive
 * recycling), so a 1-packet record is pure per-send overhead (measured 3x
 * message-rate loss at tiny sizes).  The reservation seam instead carries the
 * flow pointer with bit 0 set - records are cacheline-aligned so the bit is
 * free - and commit/cancel branch on the tag.  The replay's record field
 * stays NULL for singles, so ACK settle no-ops and NACK replay cannot
 * double-complete by construction. */
#define OPX_FLOW_CONTROL_UNGOVERNED_RECORD     ((void *) 2ul)
#define OPX_FLOW_CONTROL_IS_UNGOVERNED(record) ((record) == OPX_FLOW_CONTROL_UNGOVERNED_RECORD)
#define OPX_FLOW_CONTROL_SINGLE_TAG(flow)      ((void *) (((uintptr_t) (flow)) | 1ul))
#define OPX_FLOW_CONTROL_IS_SINGLE(record)     ((((uintptr_t) (record)) & 1ul) != 0u)
#define OPX_FLOW_CONTROL_SINGLE_FLOW(record)   ((struct fi_opx_reliability_tx_flow *) (((uintptr_t) (record)) & ~1ul))

enum opx_flow_control_result opx_flow_control_tx_admit_reserve_single(struct fi_opx_ep			*opx_ep,
								      struct fi_opx_reliability_tx_flow *flow,
								      void **reservation_record_out);
enum opx_flow_control_result opx_flow_control_tx_reserve(struct fi_opx_ep		   *opx_ep,
							 struct fi_opx_reliability_tx_flow *flow,
							 enum opx_flow_control_packet_class packet_class,
							 uint32_t requested_packets, void **reservation_record_out);
void	 opx_flow_control_tx_reduce_reservation(struct fi_opx_ep *opx_ep, void *reservation_record, uint32_t new_count);
void	 opx_flow_control_tx_register_packet(struct fi_opx_ep *opx_ep, void *reservation_record);
void	 opx_flow_control_tx_complete_packet(struct fi_opx_ep *opx_ep, void *reservation_record);
void	 opx_flow_control_tx_cancel(struct fi_opx_ep *opx_ep, void *reservation_record);
uint64_t opx_flow_control_tx_record_generation(const void *reservation_record);
bool	 opx_flow_control_tx_capacity_rejected(const struct fi_opx_reliability_tx_flow *flow);

uint64_t opx_flow_control_piggyback_write(enum opx_flow_control_carrier carrier, struct fi_opx_ep *opx_ep,
					  const union fi_opx_reliability_service_flow_key *key);
enum opx_flow_control_result opx_flow_control_piggyback_read(enum opx_flow_control_carrier    carrier,
							     struct fi_opx_ep		     *opx_ep,
							     const union opx_hfi1_packet_hdr *header);
enum opx_flow_control_result opx_flow_control_wire_send(struct fi_opx_ep *opx_ep, bool request_direction,
							const union fi_opx_reliability_service_flow_key *key,
							const struct opx_flow_control_wire_envelope	*envelope);
enum opx_flow_control_result opx_flow_control_wire_receive(struct fi_opx_ep *opx_ep, bool request_direction,
							   const union opx_hfi1_packet_hdr *header, uint32_t outer_slid,
							   uint32_t outer_dlid, uint32_t primary_slid);
void			     opx_flow_control_progress(struct fi_opx_ep *opx_ep, uint64_t now_ns);
void			     opx_flow_control_context_progress(struct fi_opx_ep *opx_ep, uint64_t now_ns);
void opx_flow_control_report_fault(struct fi_opx_ep *opx_ep, enum opx_flow_control_fault_scope scope, uint32_t reason);

void opx_flow_control_dump_process(FILE *output);
void opx_flow_control_dump_context(const struct fi_opx_hfi1_context *context, FILE *output);
void opx_flow_control_dump_tx_flow(const struct fi_opx_reliability_tx_flow *flow, FILE *output);
void opx_flow_control_dump_rx_flow(const struct fi_opx_reliability_rx_flow *flow, FILE *output);

#endif
