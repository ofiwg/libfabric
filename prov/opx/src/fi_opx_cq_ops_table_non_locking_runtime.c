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

/* HDRQ_MASK = runtime value (not 2047 or 8191, won't be optimal) */

#define FI_OPX_CQ_NON_LOCKING_RUNTIME_SPECIALIZED_FUNCTIONS(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, \
							    CTX_SHARING, FLOW_CONTROL)                            \
	FI_OPX_CQ_SPECIALIZED_FUNC_NON_LOCKING(FORMAT, RELIABILITY, MASK, CAPS, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL)

OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_SPECIALIZATION_CATALOG(FI_OPX_CQ_NON_LOCKING_RUNTIME_SPECIALIZED_FUNCTIONS)

#define FI_OPX_CQ_NON_LOCKING_RUNTIME_OPS_STRUCT(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, CTX_SHARING,  \
						 FLOW_CONTROL)                                                       \
	FI_OPX_CQ_OPS_STRUCT_INIT(FORMAT, FI_OPX_LOCK_NOT_REQUIRED, RELIABILITY, MASK, CAPS, HFI1_TYPE, CTX_SHARING, \
				  FLOW_CONTROL),

static struct fi_ops_cq fi_opx_cq_non_locking_runtime_ops_table[] = {OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CATALOG(
	FI_OPX_CQ_NON_LOCKING_RUNTIME_OPS_STRUCT, FI_OPX_HDRQ_MASK_RUNTIME, FI_OPX_HDRQ_MASK_RUNTIME)};

ssize_t fi_opx_cq_read_FABRIC_DIRECT(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_opx_cq *opx_cq = container_of(cq, struct fi_opx_cq, cq_fid);

	/* Non-inlined functions should just use the runtime HFI1 type check, no optimizations */
	if (OPX_IS_CTX_SHARING_ENABLED) {
		if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_WFR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_MIXED_9B) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_MIXED_9B,
							       OPX_CTX_SHARING_ON, NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_JKR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_CYR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count);
		} else {
			/* should never get here */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Fatal -FI_EPERM\n");
			abort();
		}
	} else {
		if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_WFR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_MIXED_9B) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_MIXED_9B,
							       OPX_CTX_SHARING_OFF, NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_JKR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_CYR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_read, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count);
		} else {
			/* should never get here */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Fatal -FI_EPERM\n");
			abort();
		}
	}
	return (ssize_t) -FI_EPERM;
}

ssize_t fi_opx_cq_readfrom_FABRIC_DIRECT(struct fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr)
{
	struct fi_opx_cq *opx_cq = container_of(cq, struct fi_opx_cq, cq_fid);

	/* Non-inlined functions should just use the runtime HFI1 and ctx_sharing type check, no optimizations */
	if (OPX_IS_CTX_SHARING_ENABLED) {
		if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_WFR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_MIXED_9B) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_MIXED_9B,
							       OPX_CTX_SHARING_ON, NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_JKR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_CYR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,
							       NONE)(cq, buf, count, src_addr);
		} else {
			/* should never get here */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Fatal -FI_EPERM\n");
			abort();
		}
	} else {
		if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_WFR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_MIXED_9B) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_MIXED_9B,
							       OPX_CTX_SHARING_OFF, NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_JKR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count, src_addr);
		} else if (OPX_SW_HFI1_TYPE(opx_cq->domain) & OPX_HFI1_CYR) {
			return FI_OPX_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom, OPX_CQ_FORMAT, OPX_LOCK, OPX_RELIABILITY,
							       OPX_MASK, OPX_CQ_CAPS, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,
							       NONE)(cq, buf, count, src_addr);
		} else {
			/* should never get here */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Fatal -FI_EPERM\n");
			abort();
		}
	}
	return (ssize_t) -FI_EPERM;
}

static op_matrix_t *fi_opx_cq_non_locking_runtime_ops = (op_matrix_t *) &fi_opx_cq_non_locking_runtime_ops_table;

struct fi_ops_cq *fi_opx_cq_select_non_locking_runtime_ops(const enum fi_cq_format	   format,
							   const enum ofi_reliability_kind reliability,
							   const uint64_t comm_caps, const uint32_t hfi1_type,
							   const bool			    ctx_sharing,
							   const enum opx_flow_control_mode flow_control_mode)
{
	if ((unsigned int) flow_control_mode >= OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_COUNT) {
		return NULL;
	}

	return &(*fi_opx_cq_non_locking_runtime_ops)[flow_control_mode][format][0][comm_caps][hfi1_type][ctx_sharing];
}
