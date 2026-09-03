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
#ifndef _FI_PROV_OPX_TELEMETRY_ORACLE_H_
#define _FI_PROV_OPX_TELEMETRY_ORACLE_H_

#ifdef OPX_TELEMETRY_ORACLE

#include <stdint.h>

#include "rdma/opx/fi_opx.h"

#define OPX_TELEMETRY_ORACLE_MAGIC   0x4C4341524F58504FULL /* "OPXORACL" little-endian */
#define OPX_TELEMETRY_ORACLE_VERSION 1u

struct opx_telemetry_oracle_file_header { /* 64 bytes */
	uint64_t magic;
	uint32_t version;
	uint32_t record_bytes;	/* sizeof(struct opx_telemetry_oracle_sample) == 40 */
	uint32_t elemcnt;	/* ring entries (rcvhdrq_cnt) */
	uint32_t elemsz_dws;	/* entry size in DWs */
	uint32_t elemlast_dws;	/* (elemcnt-1)*elemsz */
	uint32_t hfi1_type;	/* enum opx_hfi1_type value */
	uint64_t runtime_flags; /* driver ctxt runtime_flags (records HFI1_CAP_DMA_RTAIL availability) */
	uint32_t samples_written;
	uint32_t samples_dropped; /* samples lost after the buffer filled */
	uint64_t reserved[2];
};

struct opx_telemetry_oracle_sample { /* 40 bytes */
	uint64_t tsc;		     /* rdtsc */
	uint64_t sw_head_dws;	     /* opx_ep->rx->state.hdrq.head, RAW (un-normalized) */
	uint64_t rhf_seq;	     /* opx_ep->rx->state.hdrq.rhf_seq, RAW */
	uint32_t tail1_dws;	     /* MMIO ur_rcvhdrtail read BEFORE the RHF scan */
	uint32_t tail2_dws;	     /* MMIO ur_rcvhdrtail read AFTER the RHF scan */
	uint32_t backlog_entries;    /* opx_hfi1_sample_rx_backlog() result */
	uint32_t flags;		     /* bit0 = tail_stable (tail1 == tail2) */
};

_Static_assert(sizeof(struct opx_telemetry_oracle_file_header) == 64,
	       "opx telemetry oracle file header must remain 64 bytes");
_Static_assert(sizeof(struct opx_telemetry_oracle_sample) == 40, "opx telemetry oracle sample must remain 40 bytes");

struct fi_opx_ep;

void opx_telemetry_oracle_sample(struct fi_opx_ep *opx_ep, uint64_t hdrq_mask, const enum opx_hfi1_type hfi1_type);

#endif /* OPX_TELEMETRY_ORACLE */

#endif /* _FI_PROV_OPX_TELEMETRY_ORACLE_H_ */
