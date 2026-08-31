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

#ifndef _FI_PROV_OPX_FLOW_CONTROL_WIRE_H_
#define _FI_PROV_OPX_FLOW_CONTROL_WIRE_H_

#include <stddef.h>
#include <stdint.h>

#include "rdma/opx/opx_flow_control.h"

#define OPX_FLOW_CONTROL_WIRE_QWORD_COUNT 3u
#define OPX_FLOW_CONTROL_WIRE_VERSION	  UINT8_C(1)

#define OPX_FLOW_CONTROL_WIRE_FLAG_RESPONSE_REQUESTED UINT8_C(0x01)
#define OPX_FLOW_CONTROL_WIRE_FLAG_TERMINAL	      UINT8_C(0x02)

enum opx_flow_control_wire_result {
	OPX_FLOW_CONTROL_WIRE_OK = 0,
	OPX_FLOW_CONTROL_WIRE_ABSENT,
	OPX_FLOW_CONTROL_WIRE_INVALID
};

enum opx_flow_control_wire_direction {
	OPX_FLOW_CONTROL_WIRE_DIRECTION_REQUEST = 1,
	OPX_FLOW_CONTROL_WIRE_DIRECTION_FEEDBACK
};

enum opx_flow_control_wire_carrier_category {
	OPX_FLOW_CONTROL_WIRE_CARRIER_RELIABILITY = 1,
	OPX_FLOW_CONTROL_WIRE_CARRIER_REQUEST_BINDING,
	OPX_FLOW_CONTROL_WIRE_CARRIER_RESPONSE_BINDING,
	OPX_FLOW_CONTROL_WIRE_CARRIER_INELIGIBLE,
	OPX_FLOW_CONTROL_WIRE_CARRIER_REPLAY
};

enum opx_flow_control_wire_generation_source {
	OPX_FLOW_CONTROL_WIRE_GENERATION_NONE = 0,
	OPX_FLOW_CONTROL_WIRE_GENERATION_EMBEDDED,
	OPX_FLOW_CONTROL_WIRE_GENERATION_INDEPENDENT,
	OPX_FLOW_CONTROL_WIRE_GENERATION_PRESERVED
};

struct opx_flow_control_wire_qw6 {
	uint32_t value;
	uint32_t generation;
	uint8_t	 mode;
	uint8_t	 reserved[3];
};

struct opx_flow_control_wire_envelope {
	uint32_t generation;
	uint32_t value[3];
	uint64_t key;
	uint8_t	 mode;
	uint8_t	 kind;
	uint8_t	 flags;
	uint8_t	 direction;
	uint8_t	 reserved[4];
};

struct opx_flow_control_wire_carrier_metadata {
	uint8_t category;
	uint8_t generation_source;
	uint8_t extension_eligible;
	uint8_t preserve_base_semantics;
	uint8_t preserve_encoded_extension;
	uint8_t reserved[3];
};

_Static_assert(offsetof(struct opx_flow_control_wire_qw6, value) == 0u, "generic QW6 value offset");
_Static_assert(offsetof(struct opx_flow_control_wire_qw6, generation) == 4u, "generic QW6 generation offset");
_Static_assert(offsetof(struct opx_flow_control_wire_qw6, mode) == 8u, "generic QW6 mode offset");
_Static_assert(offsetof(struct opx_flow_control_wire_qw6, reserved) == 9u, "generic QW6 reserved offset");
_Static_assert(sizeof(struct opx_flow_control_wire_qw6) == 12u, "generic QW6 field layout");

_Static_assert(offsetof(struct opx_flow_control_wire_envelope, generation) == 0u,
	       "generic wire-envelope generation offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, value) == 4u, "generic wire-envelope values offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, key) == 16u, "generic wire-envelope key offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, mode) == 24u, "generic wire-envelope mode offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, kind) == 25u, "generic wire-envelope kind offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, flags) == 26u, "generic wire-envelope flags offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, direction) == 27u,
	       "generic wire-envelope direction offset");
_Static_assert(offsetof(struct opx_flow_control_wire_envelope, reserved) == 28u,
	       "generic wire-envelope reserved offset");
_Static_assert(sizeof(struct opx_flow_control_wire_envelope) == 32u, "generic wire-envelope layout");

_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, category) == 0u,
	       "carrier metadata category offset");
_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, generation_source) == 1u,
	       "carrier metadata generation-source offset");
_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, extension_eligible) == 2u,
	       "carrier metadata eligibility offset");
_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, preserve_base_semantics) == 3u,
	       "carrier metadata base-semantics offset");
_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, preserve_encoded_extension) == 4u,
	       "carrier metadata preservation offset");
_Static_assert(offsetof(struct opx_flow_control_wire_carrier_metadata, reserved) == 5u,
	       "carrier metadata reserved offset");
_Static_assert(sizeof(struct opx_flow_control_wire_carrier_metadata) == 8u, "carrier metadata layout");

void				  opx_flow_control_wire_qw6_absent(uint64_t *qw6);
enum opx_flow_control_wire_result opx_flow_control_wire_qw6_pack(const struct opx_flow_control_wire_qw6 *fields,
								 uint64_t				*qw6);
enum opx_flow_control_wire_result opx_flow_control_wire_qw6_unpack(uint64_t			     qw6,
								   struct opx_flow_control_wire_qw6 *fields);

enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_validate(const struct opx_flow_control_wire_envelope *envelope);
enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_pack(const struct opx_flow_control_wire_envelope *envelope,
				    uint64_t qwords[OPX_FLOW_CONTROL_WIRE_QWORD_COUNT]);
enum opx_flow_control_wire_result
opx_flow_control_wire_envelope_unpack(const uint64_t			     qwords[OPX_FLOW_CONTROL_WIRE_QWORD_COUNT],
				      enum opx_flow_control_wire_direction   direction,
				      struct opx_flow_control_wire_envelope *envelope);

enum opx_flow_control_wire_result
opx_flow_control_wire_carrier_metadata(enum opx_flow_control_carrier		      carrier,
				       struct opx_flow_control_wire_carrier_metadata *metadata);

#endif
