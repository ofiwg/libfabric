/*
 * Copyright (C) 2022-2026 by Cornelis Networks.
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

#include "rdma/opx/fi_opx_cq_ops_table.h"

/* HDRQ_MASK = 8k value (8191 * 0x20) */

#define FI_OPX_CQ_LOCKING_8192_SPECIALIZED_FUNCTIONS(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, \
						     CTX_SHARING, FLOW_CONTROL)                            \
	FI_OPX_CQ_SPECIALIZED_FUNC_LOCKING(FORMAT, RELIABILITY, MASK, CAPS, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL)

OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CATALOG(FI_OPX_CQ_LOCKING_8192_SPECIALIZED_FUNCTIONS, FI_OPX_HDRQ_MASK_8192,
					   FI_OPX_HDRQ_MASK_8192_CYR)

#define FI_OPX_CQ_LOCKING_8192_OPS_STRUCT(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, CTX_SHARING,     \
					  FLOW_CONTROL)                                                          \
	FI_OPX_CQ_OPS_STRUCT_INIT(FORMAT, FI_OPX_LOCK_REQUIRED, RELIABILITY, MASK, CAPS, HFI1_TYPE, CTX_SHARING, \
				  FLOW_CONTROL),

static struct fi_ops_cq fi_opx_cq_locking_8192_ops_table[] = {OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CATALOG(
	FI_OPX_CQ_LOCKING_8192_OPS_STRUCT, FI_OPX_HDRQ_MASK_8192, FI_OPX_HDRQ_MASK_8192_CYR)};

static op_matrix_t *fi_opx_cq_locking_8192_ops = (op_matrix_t *) &fi_opx_cq_locking_8192_ops_table;

struct fi_ops_cq *fi_opx_cq_select_locking_8192_ops(const enum fi_cq_format	    format,
						    const enum ofi_reliability_kind reliability,
						    const uint64_t comm_caps, const uint32_t hfi1_type,
						    const bool			     ctx_sharing,
						    const enum opx_flow_control_mode flow_control_mode)
{
	if ((unsigned int) flow_control_mode >= OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_COUNT) {
		return NULL;
	}

	return &(*fi_opx_cq_locking_8192_ops)[flow_control_mode][format][0][comm_caps][hfi1_type][ctx_sharing];
}
