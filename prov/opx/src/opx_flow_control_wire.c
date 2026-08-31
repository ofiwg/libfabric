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

#include "rdma/opx/opx_flow_control_wire.h"

#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_hfi1_packet.h"
#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_reliability.h"
#include "rdma/opx/opx_flow_control_rccc_private.h"

#include <stddef.h>
#include <string.h>

#define OPX_FLOW_CONTROL_WIRE_LOW_24_MASK  UINT64_C(0x0000000000ffffff)
#define OPX_FLOW_CONTROL_WIRE_MODE_MASK	   UINT64_C(0x00000000000000ff)
#define OPX_FLOW_CONTROL_WIRE_VERSION_MASK UINT64_C(0x000000000000000f)

#define OPX_FLOW_CONTROL_WIRE_QW6_VALUE_SHIFT	   0u
#define OPX_FLOW_CONTROL_WIRE_QW6_GENERATION_SHIFT 24u
#define OPX_FLOW_CONTROL_WIRE_QW6_MODE_SHIFT	   48u
#define OPX_FLOW_CONTROL_WIRE_QW6_VERSION_SHIFT	   56u
#define OPX_FLOW_CONTROL_WIRE_QW6_VALID_SHIFT	   60u
#define OPX_FLOW_CONTROL_WIRE_QW6_RESERVED_SHIFT   61u

#define OPX_FLOW_CONTROL_WIRE_QW5_VALUE_SHIFT	   0u
#define OPX_FLOW_CONTROL_WIRE_QW5_MODE_SHIFT	   24u
#define OPX_FLOW_CONTROL_WIRE_QW5_GENERATION_SHIFT 32u
#define OPX_FLOW_CONTROL_WIRE_QW5_KIND_SHIFT	   56u

#define OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE1_SHIFT	  0u
#define OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE2_SHIFT	  24u
#define OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_FLAGS_SHIFT	  48u
#define OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VERSION_SHIFT  56u
#define OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_RESERVED_SHIFT 60u

_Static_assert(OPX_FLOW_CONTROL_WIRE_QW5_VALUE_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW5_VALUE_0_SHIFT,
	       "generic flow-control QW5 value zero shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_QW5_MODE_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW5_MODE_ID_SHIFT,
	       "generic flow-control QW5 mode shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_QW5_GENERATION_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW5_GENERATION_SHIFT,
	       "generic flow-control QW5 generation shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_QW5_KIND_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW5_KIND_SHIFT,
	       "generic flow-control QW5 kind shift");
_Static_assert((OPX_FLOW_CONTROL_WIRE_LOW_24_MASK << OPX_FLOW_CONTROL_WIRE_QW5_VALUE_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW5_VALUE_0_MASK,
	       "generic flow-control QW5 value zero mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_MODE_MASK << OPX_FLOW_CONTROL_WIRE_QW5_MODE_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW5_MODE_ID_MASK,
	       "generic flow-control QW5 mode mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_LOW_24_MASK << OPX_FLOW_CONTROL_WIRE_QW5_GENERATION_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW5_GENERATION_MASK,
	       "generic flow-control QW5 generation mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_MODE_MASK << OPX_FLOW_CONTROL_WIRE_QW5_KIND_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW5_KIND_MASK,
	       "generic flow-control QW5 kind mask");

_Static_assert(OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE1_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VALUE_1_SHIFT,
	       "generic flow-control QW6 value one shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE2_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VALUE_2_SHIFT,
	       "generic flow-control QW6 value two shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_FLAGS_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW6_FLAGS_SHIFT,
	       "generic flow-control QW6 flags shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VERSION_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VERSION_SHIFT,
	       "generic flow-control QW6 version shift");
_Static_assert(OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_RESERVED_SHIFT == FI_OPX_HFI_UD_FLOW_CONTROL_QW6_RESERVED_SHIFT,
	       "generic flow-control QW6 reserved shift");
_Static_assert((OPX_FLOW_CONTROL_WIRE_LOW_24_MASK << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE1_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VALUE_1_MASK,
	       "generic flow-control QW6 value one mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_LOW_24_MASK << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE2_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VALUE_2_MASK,
	       "generic flow-control QW6 value two mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_MODE_MASK << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_FLAGS_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW6_FLAGS_MASK,
	       "generic flow-control QW6 flags mask");
_Static_assert((OPX_FLOW_CONTROL_WIRE_VERSION_MASK << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VERSION_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW6_VERSION_MASK,
	       "generic flow-control QW6 version mask");
_Static_assert((UINT64_C(0xf) << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_RESERVED_SHIFT) ==
		       FI_OPX_HFI_UD_FLOW_CONTROL_QW6_RESERVED_MASK,
	       "generic flow-control QW6 reserved mask");
_Static_assert(OPX_FLOW_CONTROL_WIRE_VERSION == FI_OPX_HFI_UD_FLOW_CONTROL_VERSION,
	       "generic flow-control wire version");

static int opx_flow_control_wire_mode_valid(uint8_t mode)
{
	return mode != UINT8_MAX;
}

static int opx_flow_control_wire_flags_valid(uint8_t direction, uint8_t flags)
{
	const uint8_t allowed = OPX_FLOW_CONTROL_WIRE_FLAG_RESPONSE_REQUESTED | OPX_FLOW_CONTROL_WIRE_FLAG_TERMINAL;

	if ((flags & (uint8_t) ~allowed) != 0u) {
		return 0;
	}
	if (direction == OPX_FLOW_CONTROL_WIRE_DIRECTION_REQUEST) {
		return (flags & OPX_FLOW_CONTROL_WIRE_FLAG_TERMINAL) == 0u;
	}
	if (direction == OPX_FLOW_CONTROL_WIRE_DIRECTION_FEEDBACK) {
		return (flags & OPX_FLOW_CONTROL_WIRE_FLAG_RESPONSE_REQUESTED) == 0u;
	}
	return 0;
}

static int opx_flow_control_wire_bytes_zero(const uint8_t *bytes, size_t count)
{
	size_t index;

	for (index = 0u; index < count; ++index) {
		if (bytes[index] != 0u) {
			return 0;
		}
	}
	return 1;
}

void opx_flow_control_wire_qw6_absent(uint64_t *qw6)
{
	if (qw6 != NULL) {
		*qw6 = 0u;
	}
}

enum opx_flow_control_wire_result opx_flow_control_wire_qw6_pack(const struct opx_flow_control_wire_qw6 *fields,
								 uint64_t				*qw6)
{
	uint64_t encoded;

	if (fields == NULL || qw6 == NULL || fields->value > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK ||
	    fields->generation > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK || !opx_flow_control_wire_mode_valid(fields->mode) ||
	    !opx_flow_control_wire_bytes_zero(fields->reserved, sizeof(fields->reserved))) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	encoded = ((uint64_t) fields->value << OPX_FLOW_CONTROL_WIRE_QW6_VALUE_SHIFT) |
		  ((uint64_t) fields->generation << OPX_FLOW_CONTROL_WIRE_QW6_GENERATION_SHIFT) |
		  ((uint64_t) fields->mode << OPX_FLOW_CONTROL_WIRE_QW6_MODE_SHIFT) |
		  ((uint64_t) OPX_FLOW_CONTROL_WIRE_VERSION << OPX_FLOW_CONTROL_WIRE_QW6_VERSION_SHIFT) |
		  (UINT64_C(1) << OPX_FLOW_CONTROL_WIRE_QW6_VALID_SHIFT);
	*qw6 = encoded;
	return OPX_FLOW_CONTROL_WIRE_OK;
}

enum opx_flow_control_wire_result opx_flow_control_wire_qw6_unpack(uint64_t			     qw6,
								   struct opx_flow_control_wire_qw6 *fields)
{
	struct opx_flow_control_wire_qw6 decoded = {0};
	uint8_t				 version;

	if (fields == NULL) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}
	if (qw6 == 0u) {
		return OPX_FLOW_CONTROL_WIRE_ABSENT;
	}

	version = (uint8_t) ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_VERSION_SHIFT) & OPX_FLOW_CONTROL_WIRE_VERSION_MASK);
	decoded.mode = (uint8_t) ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_MODE_SHIFT) & OPX_FLOW_CONTROL_WIRE_MODE_MASK);
	if ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_RESERVED_SHIFT) != 0u ||
	    ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_VALID_SHIFT) & UINT64_C(1)) == 0u ||
	    version != OPX_FLOW_CONTROL_WIRE_VERSION || !opx_flow_control_wire_mode_valid(decoded.mode)) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	decoded.value = (uint32_t) ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_VALUE_SHIFT) & OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	decoded.generation =
		(uint32_t) ((qw6 >> OPX_FLOW_CONTROL_WIRE_QW6_GENERATION_SHIFT) & OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	*fields = decoded;
	return OPX_FLOW_CONTROL_WIRE_OK;
}

enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_validate(const struct opx_flow_control_wire_envelope *envelope)
{
	if (envelope == NULL || envelope->generation > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK ||
	    envelope->value[0] > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK ||
	    envelope->value[1] > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK ||
	    envelope->value[2] > OPX_FLOW_CONTROL_WIRE_LOW_24_MASK ||
	    !opx_flow_control_wire_mode_valid(envelope->mode) || envelope->kind == 0u || envelope->kind == UINT8_MAX ||
	    !opx_flow_control_wire_flags_valid(envelope->direction, envelope->flags) ||
	    !opx_flow_control_wire_bytes_zero(envelope->reserved, sizeof(envelope->reserved))) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}
	return OPX_FLOW_CONTROL_WIRE_OK;
}

enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_pack(const struct opx_flow_control_wire_envelope *envelope,
				    uint64_t qwords[OPX_FLOW_CONTROL_WIRE_QWORD_COUNT])
{
	uint64_t encoded[OPX_FLOW_CONTROL_WIRE_QWORD_COUNT];

	if (qwords == NULL || opx_flow_control_wire_envelope_validate(envelope) != OPX_FLOW_CONTROL_WIRE_OK) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	encoded[0] = ((uint64_t) envelope->value[0] << OPX_FLOW_CONTROL_WIRE_QW5_VALUE_SHIFT) |
		     ((uint64_t) envelope->mode << OPX_FLOW_CONTROL_WIRE_QW5_MODE_SHIFT) |
		     ((uint64_t) envelope->generation << OPX_FLOW_CONTROL_WIRE_QW5_GENERATION_SHIFT) |
		     ((uint64_t) envelope->kind << OPX_FLOW_CONTROL_WIRE_QW5_KIND_SHIFT);
	encoded[1] = ((uint64_t) envelope->value[1] << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE1_SHIFT) |
		     ((uint64_t) envelope->value[2] << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE2_SHIFT) |
		     ((uint64_t) envelope->flags << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_FLAGS_SHIFT) |
		     ((uint64_t) OPX_FLOW_CONTROL_WIRE_VERSION << OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VERSION_SHIFT);
	encoded[2] = envelope->key;
	memcpy(qwords, encoded, sizeof(encoded));
	return OPX_FLOW_CONTROL_WIRE_OK;
}

enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_unpack(const uint64_t			     qwords[OPX_FLOW_CONTROL_WIRE_QWORD_COUNT],
				      enum opx_flow_control_wire_direction   direction,
				      struct opx_flow_control_wire_envelope *envelope)
{
	struct opx_flow_control_wire_envelope decoded = {0};
	uint8_t				      version;

	if (qwords == NULL || envelope == NULL) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	version = (uint8_t) ((qwords[1] >> OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VERSION_SHIFT) &
			     OPX_FLOW_CONTROL_WIRE_VERSION_MASK);
	if ((qwords[1] >> OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_RESERVED_SHIFT) != 0u ||
	    version != OPX_FLOW_CONTROL_WIRE_VERSION) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	decoded.value[0] =
		(uint32_t) ((qwords[0] >> OPX_FLOW_CONTROL_WIRE_QW5_VALUE_SHIFT) & OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	decoded.mode =
		(uint8_t) ((qwords[0] >> OPX_FLOW_CONTROL_WIRE_QW5_MODE_SHIFT) & OPX_FLOW_CONTROL_WIRE_MODE_MASK);
	decoded.generation = (uint32_t) ((qwords[0] >> OPX_FLOW_CONTROL_WIRE_QW5_GENERATION_SHIFT) &
					 OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	decoded.kind =
		(uint8_t) ((qwords[0] >> OPX_FLOW_CONTROL_WIRE_QW5_KIND_SHIFT) & OPX_FLOW_CONTROL_WIRE_MODE_MASK);
	decoded.value[1]  = (uint32_t) ((qwords[1] >> OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE1_SHIFT) &
					OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	decoded.value[2]  = (uint32_t) ((qwords[1] >> OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_VALUE2_SHIFT) &
					OPX_FLOW_CONTROL_WIRE_LOW_24_MASK);
	decoded.flags	  = (uint8_t) ((qwords[1] >> OPX_FLOW_CONTROL_WIRE_ENVELOPE_QW6_FLAGS_SHIFT) &
				       OPX_FLOW_CONTROL_WIRE_MODE_MASK);
	decoded.direction = (uint8_t) direction;
	decoded.key	  = qwords[2];
	if (opx_flow_control_wire_envelope_validate(&decoded) != OPX_FLOW_CONTROL_WIRE_OK) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	*envelope = decoded;
	return OPX_FLOW_CONTROL_WIRE_OK;
}

static struct opx_flow_control_wire_carrier_metadata
opx_flow_control_wire_make_carrier_metadata(enum opx_flow_control_wire_carrier_category	 category,
					    enum opx_flow_control_wire_generation_source generation_source,
					    uint8_t extension_eligible, uint8_t preserve_base_semantics,
					    uint8_t preserve_encoded_extension)
{
	struct opx_flow_control_wire_carrier_metadata metadata = {0};

	metadata.category		    = (uint8_t) category;
	metadata.generation_source	    = (uint8_t) generation_source;
	metadata.extension_eligible	    = extension_eligible;
	metadata.preserve_base_semantics    = preserve_base_semantics;
	metadata.preserve_encoded_extension = preserve_encoded_extension;
	return metadata;
}

enum opx_flow_control_wire_result
opx_flow_control_wire_carrier_metadata(enum opx_flow_control_carrier		      carrier,
				       struct opx_flow_control_wire_carrier_metadata *metadata)
{
	struct opx_flow_control_wire_carrier_metadata selected;

	if (metadata == NULL) {
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	switch (carrier) {
	case OPX_FLOW_CONTROL_CARRIER_RELIABILITY_PING:
	case OPX_FLOW_CONTROL_CARRIER_RELIABILITY_ACK:
		selected = opx_flow_control_wire_make_carrier_metadata(OPX_FLOW_CONTROL_WIRE_CARRIER_RELIABILITY,
								       OPX_FLOW_CONTROL_WIRE_GENERATION_EMBEDDED, 1u,
								       1u, 0u);
		break;
	case OPX_FLOW_CONTROL_CARRIER_MSG_RZV_RTS:
	case OPX_FLOW_CONTROL_CARRIER_TAG_RZV_RTS:
	case OPX_FLOW_CONTROL_CARRIER_RMA_RTS:
		selected = opx_flow_control_wire_make_carrier_metadata(OPX_FLOW_CONTROL_WIRE_CARRIER_REQUEST_BINDING,
								       OPX_FLOW_CONTROL_WIRE_GENERATION_INDEPENDENT, 0u,
								       1u, 0u);
		break;
	case OPX_FLOW_CONTROL_CARRIER_MEMORY_REGION_CTS:
	case OPX_FLOW_CONTROL_CARRIER_RMA_CTS:
	case OPX_FLOW_CONTROL_CARRIER_FENCE_CTS:
		selected = opx_flow_control_wire_make_carrier_metadata(OPX_FLOW_CONTROL_WIRE_CARRIER_RESPONSE_BINDING,
								       OPX_FLOW_CONTROL_WIRE_GENERATION_INDEPENDENT, 0u,
								       1u, 0u);
		break;
	case OPX_FLOW_CONTROL_CARRIER_VADDR_CTS:
	case OPX_FLOW_CONTROL_CARRIER_TID_VADDR_CTS:
	case OPX_FLOW_CONTROL_CARRIER_HFISVC_RZV_RTS:
	case OPX_FLOW_CONTROL_CARRIER_DPUT:
		selected = opx_flow_control_wire_make_carrier_metadata(
			OPX_FLOW_CONTROL_WIRE_CARRIER_INELIGIBLE, OPX_FLOW_CONTROL_WIRE_GENERATION_NONE, 0u, 1u, 0u);
		break;
	case OPX_FLOW_CONTROL_CARRIER_REPLAYED_ORIGINAL:
		selected = opx_flow_control_wire_make_carrier_metadata(
			OPX_FLOW_CONTROL_WIRE_CARRIER_REPLAY, OPX_FLOW_CONTROL_WIRE_GENERATION_PRESERVED, 0u, 1u, 1u);
		break;
	case OPX_FLOW_CONTROL_CARRIER_UNSPECIFIED:
	case OPX_FLOW_CONTROL_CARRIER_COUNT:
	default:
		return OPX_FLOW_CONTROL_WIRE_INVALID;
	}

	*metadata = selected;
	return OPX_FLOW_CONTROL_WIRE_OK;
}

enum opx_flow_control_result opx_flow_control_piggyback_write(enum opx_flow_control_carrier	   carrier,
							      const struct opx_flow_control_state *state,
							      union opx_hfi1_packet_hdr		  *header)
{
	(void) carrier;
	(void) state;
	(void) header;
	return OPX_FLOW_CONTROL_FAULT;
}

enum opx_flow_control_result opx_flow_control_piggyback_read(enum opx_flow_control_carrier    carrier,
							     const union opx_hfi1_packet_hdr *header,
							     struct opx_flow_control_state   *state)
{
	(void) carrier;
	(void) header;
	(void) state;
	return OPX_FLOW_CONTROL_FAULT;
}

enum opx_flow_control_result opx_flow_control_wire_send(struct fi_opx_ep *opx_ep, bool request_direction,
							const union fi_opx_reliability_service_flow_key *key,
							const struct opx_flow_control_wire_envelope	*envelope)
{
	enum opx_flow_control_wire_direction direction;
	opx_lid_t			     dlid;
	uint16_t			     reliability_rx;
	ssize_t				     rc;

	assert(opx_ep != NULL);
	assert(opx_ep->flow_control_reservation_pool != NULL);
	assert(opx_ep->domain != NULL);
	assert(key != NULL);
	assert(envelope != NULL);

	direction =
		request_direction ? OPX_FLOW_CONTROL_WIRE_DIRECTION_REQUEST : OPX_FLOW_CONTROL_WIRE_DIRECTION_FEEDBACK;
	assert(envelope->direction == direction);
	assert(envelope->key == key->qw_prefix);
	assert(opx_flow_control_wire_envelope_validate(envelope) == OPX_FLOW_CONTROL_WIRE_OK);

	assert(OPX_RELIABILITY_DECODE_TX_INDEX(key->slid) < opx_ep->num_tx_contexts);
	assert(OPX_RELIABILITY_DECODE_TX_INDEX(key->dlid) == OPX_RELIABILITY_DECODE_TX_INDEX(key->slid));

	if (request_direction) {
		dlid	       = OPX_RELIABILITY_DECODE_LID(key->dlid);
		reliability_rx = key->dst_subctxt_rx;
	} else {
		dlid	       = OPX_RELIABILITY_DECODE_LID(key->slid);
		reliability_rx = key->src_subctxt_rx;
	}
	rc = fi_opx_hfi1_tx_flow_control_reliability_inject(&opx_ep->ep_fid, key, envelope, dlid, reliability_rx,
							    direction, OPX_SW_HFI1_TYPE(opx_ep->domain),
							    OPX_IS_CTX_SHARING_ENABLED);
	if (rc == FI_SUCCESS) {
		return OPX_FLOW_CONTROL_OK;
	}
	return OFI_LIKELY(rc == -FI_EAGAIN) ? OPX_FLOW_CONTROL_DEFER : OPX_FLOW_CONTROL_FAULT;
}

static inline bool opx_flow_control_rccc_wire_kind_valid(const struct opx_flow_control_wire_envelope *envelope,
							 bool					      request_direction)
{
	if (envelope->mode != OPX_FLOW_CONTROL_MODE_RCCC) {
		return false;
	}
	if (request_direction) {
		if (envelope->flags != OPX_FLOW_CONTROL_WIRE_FLAG_RESPONSE_REQUESTED) {
			return false;
		}
		switch (envelope->kind) {
		case OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST:
			return envelope->value[2] != 0u;
		case OPX_FLOW_CONTROL_RCCC_KIND_RETIREMENT:
		case OPX_FLOW_CONTROL_RCCC_KIND_RESET:
			return envelope->value[0] == 0u && envelope->value[2] == 0u;
		default:
			return false;
		}
	}
	switch (envelope->kind) {
	case OPX_FLOW_CONTROL_RCCC_KIND_FEEDBACK:
		return envelope->flags == 0u && envelope->value[2] == 0u;
	case OPX_FLOW_CONTROL_RCCC_KIND_RESET_ACK:
		return envelope->flags == 0u && envelope->value[0] == 0u && envelope->value[1] == 0u &&
		       envelope->value[2] == 0u;
	case OPX_FLOW_CONTROL_RCCC_KIND_CAPACITY_REJECT:
		return envelope->flags == OPX_FLOW_CONTROL_WIRE_FLAG_TERMINAL && envelope->value[1] == 0u &&
		       envelope->value[2] == 0u;
	default:
		return false;
	}
}

enum opx_flow_control_result opx_flow_control_wire_receive(struct fi_opx_ep *opx_ep, bool request_direction,
							   const union opx_hfi1_packet_hdr *header, uint32_t outer_slid,
							   uint32_t outer_dlid, uint32_t primary_slid)
{
	struct opx_flow_control_wire_envelope	  envelope;
	union fi_opx_reliability_service_flow_key key;
	enum opx_flow_control_wire_direction	  direction;
	void					 *iterator;
	uint8_t					  tx_index;

	if (opx_ep == NULL || opx_ep->flow_control_reservation_pool == NULL) {
		return OPX_FLOW_CONTROL_DEFER;
	}
	if (opx_ep->reli_service == NULL || header == NULL) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	direction =
		request_direction ? OPX_FLOW_CONTROL_WIRE_DIRECTION_REQUEST : OPX_FLOW_CONTROL_WIRE_DIRECTION_FEEDBACK;
	if (opx_flow_control_wire_envelope_unpack(&header->qw_16B[5], direction, &envelope) !=
	    OPX_FLOW_CONTROL_WIRE_OK) {
		return OPX_FLOW_CONTROL_FAULT;
	}

	key.qw_prefix = envelope.key;
	key.dw_suffix = header->service.key_dw_suffix;
	tx_index      = OPX_RELIABILITY_DECODE_TX_INDEX(key.slid);
	if (outer_slid == 0u || primary_slid == 0u || outer_dlid == 0u ||
	    OPX_RELIABILITY_DECODE_TX_INDEX(key.dlid) != tx_index || tx_index >= opx_ep->num_tx_contexts) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	/* Dual-plane: a tx_index=1 control packet egresses the secondary
	 * plane, so the arriving LRH slid is the sender's secondary-plane
	 * LID while the flow key holds primary LIDs.  The LRH slid is
	 * therefore not comparable to the key; validate the key's source
	 * against the sender's primary LID baked into QW[3] (the same
	 * mechanism reliability ping responses use for dual-plane routing)
	 * and the key's destination against the LRH dlid, which is always
	 * a primary LID because control senders address the peer's primary. */
	if (request_direction) {
		if (key.src_subctxt_rx !=
			    (header->service.origin_reliability_subctxt_rx & OPX_SERVICE_SUBCTXT_RX_MASK) ||
		    key.dst_subctxt_rx != opx_ep->reli_service->subctxt_rx ||
		    key.slid != OPX_RELIABILITY_ENCODE_LID_TX(primary_slid, tx_index) ||
		    key.dlid != OPX_RELIABILITY_ENCODE_LID_TX(outer_dlid, tx_index)) {
			return OPX_FLOW_CONTROL_FAULT;
		}
	} else if (key.dst_subctxt_rx !=
			   (header->service.origin_reliability_subctxt_rx & OPX_SERVICE_SUBCTXT_RX_MASK) ||
		   key.src_subctxt_rx != opx_ep->reli_service->subctxt_rx ||
		   key.dlid != OPX_RELIABILITY_ENCODE_LID_TX(primary_slid, tx_index) ||
		   key.slid != OPX_RELIABILITY_ENCODE_LID_TX(outer_dlid, tx_index)) {
		return OPX_FLOW_CONTROL_FAULT;
	}

	if (OFI_UNLIKELY(!opx_flow_control_rccc_wire_kind_valid(&envelope, request_direction))) {
		opx_ep->reli_service->flow_control_feedback_counters.invalid++;
		return OPX_FLOW_CONTROL_FAULT;
	}
	if (request_direction) {
		struct fi_opx_reliability_rx_flow *flow;
		enum opx_flow_control_result	   result;

		iterator = fi_opx_rbt_find(opx_ep->reli_service->rx.rx_flow_rbtree, (void *) &key);
		if (iterator == NULL) {
			if (envelope.kind != OPX_FLOW_CONTROL_RCCC_KIND_CREDIT_REQUEST) {
				return OPX_FLOW_CONTROL_DEFER;
			}
			flow = fi_opx_reliability_create_rx_flow(opx_ep->reli_service, &key);
			if (flow == NULL || opx_flow_control_rx_flow_init(opx_ep, flow) != FI_SUCCESS) {
				return OPX_FLOW_CONTROL_FAULT;
			}
		} else {
			flow = fi_opx_rbt_value(opx_ep->reli_service->rx.rx_flow_rbtree, iterator);
		}
		result = opx_flow_control_rccc_rx_control(opx_ep, flow, &envelope);
		if (result == OPX_FLOW_CONTROL_OK) {
			(void) opx_flow_control_rccc_rx_emit_pending(
				opx_ep, flow,
				(envelope.kind == OPX_FLOW_CONTROL_RCCC_KIND_RESET) ?
					OPX_FLOW_CONTROL_RCCC_RX_FLAG_RESET_ACK_PENDING :
					OPX_FLOW_CONTROL_RCCC_RX_FLAG_FEEDBACK_PENDING);
		}
		return result;
	}

	iterator = fi_opx_rbt_find(opx_ep->reli_service->tx.tx_flow_rbtree, (void *) &key);
	if (iterator == NULL) {
		opx_ep->reli_service->flow_control_feedback_counters.invalid++;
		return OPX_FLOW_CONTROL_DEFER;
	}
	struct fi_opx_reliability_tx_flow *flow = fi_opx_rbt_value(opx_ep->reli_service->tx.tx_flow_rbtree, iterator);
	struct opx_flow_control_rccc_tx_state *tx_state =
		(struct opx_flow_control_rccc_tx_state *) ((struct opx_flow_control_state *)
								   opx_flow_control_tx_flow_state(flow))
			->mode_state;
	const struct opx_flow_control_rccc_tx_profile *profile = opx_flow_control_rccc_tx_profile_get();
	if (profile == NULL) {
		return OPX_FLOW_CONTROL_FAULT;
	}
	switch (envelope.kind) {
	case OPX_FLOW_CONTROL_RCCC_KIND_FEEDBACK:
		(void) opx_flow_control_rccc_tx_feedback_apply(tx_state, profile, envelope.value[0], envelope.value[1],
							       envelope.generation,
							       &opx_ep->reli_service->flow_control_feedback_counters);
		break;
	case OPX_FLOW_CONTROL_RCCC_KIND_RESET_ACK:
		if (flow->flow_control.committed_head != NULL) {
			return OPX_FLOW_CONTROL_DEFER;
		}
		(void) opx_flow_control_rccc_tx_reset_ack(tx_state, profile, envelope.generation);
		break;
	case OPX_FLOW_CONTROL_RCCC_KIND_CAPACITY_REJECT:
		if (envelope.value[0] != tx_state->epoch) {
			opx_ep->reli_service->flow_control_feedback_counters.invalid++;
			return OPX_FLOW_CONTROL_FAULT;
		}
		(void) opx_flow_control_rccc_tx_capacity_reject(tx_state, profile, envelope.generation,
								&opx_ep->reli_service->flow_control_feedback_counters);
		break;
	default:
		return OPX_FLOW_CONTROL_FAULT;
	}
	opx_flow_control_rccc_tx_work_enqueue(opx_ep, flow);
	return OPX_FLOW_CONTROL_OK;
}
