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

#ifndef _FI_PROV_OPX_FLOW_CONTROL_INLINE_H_
#define _FI_PROV_OPX_FLOW_CONTROL_INLINE_H_

#include <stdbool.h>

#include "rdma/opx/fi_opx_compiler.h"
#include "rdma/opx/opx_flow_control.h"

struct fi_opx_ep;
struct fi_opx_reliability_tx_flow;
struct fi_opx_reliability_rx_flow;
union fi_opx_reliability_service_flow_key;
union opx_hfi1_packet_hdr;

__OPX_FORCE_INLINE__
int opx_flow_control_tx_flow_init_NONE(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow,
				       const union fi_opx_reliability_service_flow_key *key)
{
	(void) opx_ep;
	(void) flow;
	(void) key;
	return 0;
}

__OPX_FORCE_INLINE__
int opx_flow_control_tx_flow_init_RCCC(struct fi_opx_ep *opx_ep, struct fi_opx_reliability_tx_flow *flow,
				       const union fi_opx_reliability_service_flow_key *key)
{
	return opx_flow_control_tx_flow_init(opx_ep, flow, key);
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_tx_reserve_NONE(struct fi_opx_ep			*opx_ep,
							      struct fi_opx_reliability_tx_flow *flow,
							      enum opx_flow_control_packet_class packet_class,
							      uint32_t requested_packets, void **reservation_record_out)
{
	(void) opx_ep;
	(void) flow;
	(void) packet_class;
	(void) requested_packets;
	(void) reservation_record_out;
	return OPX_FLOW_CONTROL_OK;
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_reduce_reservation_NONE(struct fi_opx_ep *opx_ep, void *reservation_record, uint32_t new_count)
{
	(void) opx_ep;
	(void) reservation_record;
	(void) new_count;
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_register_packet_NONE(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	(void) opx_ep;
	(void) reservation_record;
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_complete_packet_NONE(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	(void) opx_ep;
	(void) reservation_record;
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_cancel_NONE(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	(void) opx_ep;
	(void) reservation_record;
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_tx_reserve_RCCC(struct fi_opx_ep			*opx_ep,
							      struct fi_opx_reliability_tx_flow *flow,
							      enum opx_flow_control_packet_class packet_class,
							      uint32_t requested_packets, void **reservation_record_out)
{
	return opx_flow_control_tx_reserve(opx_ep, flow, packet_class, requested_packets, reservation_record_out);
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_reduce_reservation_RCCC(struct fi_opx_ep *opx_ep, void *reservation_record, uint32_t new_count)
{
	opx_flow_control_tx_reduce_reservation(opx_ep, reservation_record, new_count);
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_register_packet_RCCC(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	opx_flow_control_tx_register_packet(opx_ep, reservation_record);
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_complete_packet_RCCC(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	opx_flow_control_tx_complete_packet(opx_ep, reservation_record);
}

__OPX_FORCE_INLINE__
void opx_flow_control_tx_cancel_RCCC(struct fi_opx_ep *opx_ep, void *reservation_record)
{
	opx_flow_control_tx_cancel(opx_ep, reservation_record);
}

__OPX_FORCE_INLINE__
uint64_t opx_flow_control_tx_record_generation_NONE(const void *reservation_record)
{
	(void) reservation_record;
	return 0u;
}

__OPX_FORCE_INLINE__
uint64_t opx_flow_control_tx_record_generation_RCCC(const void *reservation_record)
{
	return opx_flow_control_tx_record_generation(reservation_record);
}

__OPX_FORCE_INLINE__
uint64_t opx_flow_control_piggyback_write_NONE(enum opx_flow_control_carrier carrier, struct fi_opx_ep *opx_ep,
					       const union fi_opx_reliability_service_flow_key *key)
{
	(void) carrier;
	(void) opx_ep;
	(void) key;
	return 0u;
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_piggyback_read_NONE(enum opx_flow_control_carrier	   carrier,
								  struct fi_opx_ep		  *opx_ep,
								  const union opx_hfi1_packet_hdr *header)
{
	(void) carrier;
	(void) opx_ep;
	(void) header;
	return OPX_FLOW_CONTROL_OK;
}

__OPX_FORCE_INLINE__
uint64_t opx_flow_control_piggyback_write_RCCC(enum opx_flow_control_carrier carrier, struct fi_opx_ep *opx_ep,
					       const union fi_opx_reliability_service_flow_key *key)
{
	return opx_flow_control_piggyback_write(carrier, opx_ep, key);
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_piggyback_read_RCCC(enum opx_flow_control_carrier	   carrier,
								  struct fi_opx_ep		  *opx_ep,
								  const union opx_hfi1_packet_hdr *header)
{
	return opx_flow_control_piggyback_read(carrier, opx_ep, header);
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_wire_send_NONE(struct fi_opx_ep *opx_ep, bool request_direction,
							     const union fi_opx_reliability_service_flow_key *key,
							     const struct opx_flow_control_wire_envelope     *envelope)
{
	(void) opx_ep;
	(void) request_direction;
	(void) key;
	(void) envelope;
	return OPX_FLOW_CONTROL_OK;
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_wire_receive_NONE(struct fi_opx_ep *opx_ep, bool request_direction,
								const union opx_hfi1_packet_hdr *header,
								uint32_t outer_slid, uint32_t outer_dlid,
								uint32_t primary_slid)
{
	(void) opx_ep;
	(void) request_direction;
	(void) header;
	(void) outer_slid;
	(void) outer_dlid;
	(void) primary_slid;
	return OPX_FLOW_CONTROL_OK;
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_wire_send_RCCC(struct fi_opx_ep *opx_ep, bool request_direction,
							     const union fi_opx_reliability_service_flow_key *key,
							     const struct opx_flow_control_wire_envelope     *envelope)
{
	return opx_flow_control_wire_send(opx_ep, request_direction, key, envelope);
}

__OPX_FORCE_INLINE__
enum opx_flow_control_result opx_flow_control_wire_receive_RCCC(struct fi_opx_ep *opx_ep, bool request_direction,
								const union opx_hfi1_packet_hdr *header,
								uint32_t outer_slid, uint32_t outer_dlid,
								uint32_t primary_slid)
{
	return opx_flow_control_wire_receive(opx_ep, request_direction, header, outer_slid, outer_dlid, primary_slid);
}

__OPX_FORCE_INLINE__
void opx_flow_control_progress_NONE(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	(void) opx_ep;
	(void) now_ns;
}

__OPX_FORCE_INLINE__
void opx_flow_control_context_progress_NONE(struct fi_opx_ep *opx_ep, uint64_t now_ns)
{
	(void) opx_ep;
	(void) now_ns;
}

__OPX_FORCE_INLINE__
void opx_flow_control_report_fault_NONE(struct fi_opx_ep *opx_ep, enum opx_flow_control_fault_scope scope,
					uint32_t reason)
{
	(void) opx_ep;
	(void) scope;
	(void) reason;
}

#endif
