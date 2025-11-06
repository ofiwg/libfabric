// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */

/* IXE block */

#include <linux/hpe/cxi/cxi.h>
#include <rdma/ib_pack.h>

#include "cass_core.h"

/* AMO remap to PCIe fetching add is used to remap a single operand NIC AMO
 * as a PCIe fetch add AMO.
 */
static int amo_remap_to_pcie_fadd = C_AMO_OP_SWAP;
module_param(amo_remap_to_pcie_fadd, int, 0444);
MODULE_PARM_DESC(amo_remap_to_pcie_fadd,
		 "Remap NIC AMO operation as a PCIe fetch add operation. "
		 "The following are the value values.\n"
		 "AMO_OP_MIN == 0\n"
		 "AMO_OP_MAX == 1\n"
		 "AMO_OP_SUM == 2\n"
		 "AMO_OP_LOR == 4\n"
		 "AMO_OP_LAND == 5\n"
		 "AMO_OP_BOR == 6\n"
		 "AMO_OP_BAND == 7\n"
		 "AMO_OP_LXOR == 8\n"
		 "AMO_OP_BXOR == 9\n"
		 "AMO_OP_SWAP == 10\n"
		 "-1 disables remapping");


enum {
	RXE_BTH_BYTES       = 12,
	RXE_DETH_BYTES      = 8,
	RXE_IMMDT_BYTES     = 4,
	RXE_RETH_BYTES      = 16,
	RXE_AETH_BYTES      = 4,
	RXE_ATMACK_BYTES    = 8,
	RXE_ATMETH_BYTES    = 28,
	RXE_IETH_BYTES      = 4,
	RXE_RDETH_BYTES     = 4,
};

/* Length of a whole BTH depending on its opcode, not including the
 * base BTH itself. Extracted from the rdma_rxe driver's rxe_opcode
 * array.
 */
static const int rxe_op_length[256] = {
	/* RC */
	[IB_OPCODE_RC_SEND_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_RC_SEND_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_RC_RDMA_WRITE_FIRST] = RXE_RETH_BYTES,
	[IB_OPCODE_RC_RDMA_WRITE_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_RC_RDMA_WRITE_ONLY] = RXE_RETH_BYTES,
	[IB_OPCODE_RC_RDMA_WRITE_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_RETH_BYTES,
	[IB_OPCODE_RC_RDMA_READ_REQUEST] = RXE_RETH_BYTES,
	[IB_OPCODE_RC_RDMA_READ_RESPONSE_FIRST] = RXE_AETH_BYTES,
	[IB_OPCODE_RC_RDMA_READ_RESPONSE_LAST] = RXE_AETH_BYTES,
	[IB_OPCODE_RC_RDMA_READ_RESPONSE_ONLY] = RXE_AETH_BYTES,
	[IB_OPCODE_RC_ACKNOWLEDGE] = RXE_AETH_BYTES,
	[IB_OPCODE_RC_ATOMIC_ACKNOWLEDGE] = RXE_ATMACK_BYTES + RXE_AETH_BYTES,
	[IB_OPCODE_RC_COMPARE_SWAP] = RXE_ATMETH_BYTES,
	[IB_OPCODE_RC_FETCH_ADD] = RXE_ATMETH_BYTES,
	[IB_OPCODE_RC_SEND_LAST_WITH_INVALIDATE] = RXE_IETH_BYTES,
	[IB_OPCODE_RC_SEND_ONLY_WITH_INVALIDATE] = RXE_IETH_BYTES,

	/* UC */
	[IB_OPCODE_UC_SEND_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_UC_SEND_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_UC_RDMA_WRITE_FIRST] = RXE_RETH_BYTES,
	[IB_OPCODE_UC_RDMA_WRITE_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES,
	[IB_OPCODE_UC_RDMA_WRITE_ONLY] = RXE_RETH_BYTES,
	[IB_OPCODE_UC_RDMA_WRITE_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_RETH_BYTES,

	/* RD */
	[IB_OPCODE_RD_SEND_FIRST] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_SEND_MIDDLE] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_SEND_LAST] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_SEND_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_SEND_ONLY] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_SEND_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_FIRST] = RXE_RETH_BYTES + RXE_DETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_MIDDLE] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_LAST] = RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_LAST_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_ONLY] = RXE_RETH_BYTES + RXE_DETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_WRITE_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_RETH_BYTES + RXE_DETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_READ_REQUEST] = RXE_RETH_BYTES + RXE_DETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_READ_RESPONSE_FIRST] = RXE_AETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_READ_RESPONSE_MIDDLE] = RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_READ_RESPONSE_LAST] = RXE_AETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_RDMA_READ_RESPONSE_ONLY] = RXE_AETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_ACKNOWLEDGE] = RXE_AETH_BYTES + RXE_RDETH_BYTES,
	[IB_OPCODE_RD_ATOMIC_ACKNOWLEDGE] = RXE_ATMACK_BYTES + RXE_AETH_BYTES +
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_COMPARE_SWAP] = RXE_ATMETH_BYTES + RXE_DETH_BYTES	+
		RXE_RDETH_BYTES,
	[IB_OPCODE_RD_FETCH_ADD] = RXE_ATMETH_BYTES + RXE_DETH_BYTES +
		RXE_RDETH_BYTES,

	/* UD */
	[IB_OPCODE_UD_SEND_ONLY] = RXE_DETH_BYTES,
	[IB_OPCODE_UD_SEND_ONLY_WITH_IMMEDIATE] = RXE_IMMDT_BYTES +
		RXE_DETH_BYTES,
};

/* Program BTH opcodes for RXE, using definitions from the rxe driver */
static void set_bth_opcode(struct cass_dev *hw)
{
	unsigned int index;
	unsigned int offset;

	for (index = 0; index < C_IXE_CFG_BTH_OPCODE_ENTRIES; index++) {
		union c_ixe_cfg_bth_opcode bth_opcode = {};

		for (offset = 0; offset < 16; offset++) {
			u64 length;

			length = RXE_BTH_BYTES;
			length += rxe_op_length[index * 16 + offset];
			length /= 4;

			bth_opcode.qw |= length << (offset * 4);
		}

		cass_write(hw, C_IXE_CFG_BTH_OPCODE(index),
			   &bth_opcode, sizeof(bth_opcode));
	}
}

static bool cass_ixe_valid_amo_remap_to_pcie_fadd(int amo_remap)
{
	switch (amo_remap) {
	case -1:
	case C_AMO_OP_MIN:
	case C_AMO_OP_MAX:
	case C_AMO_OP_SUM:
	case C_AMO_OP_LOR:
	case C_AMO_OP_LAND:
	case C_AMO_OP_BOR:
	case C_AMO_OP_BAND:
	case C_AMO_OP_LXOR:
	case C_AMO_OP_BXOR:
	case C_AMO_OP_SWAP:
		return true;

	default:
		return false;
	}
}

#define OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP 0xf
#define OFFLOAD_INT32_INT64_UINT32_UINT64_FETCH_NON_FETCH_1OP 0xf

int cass_ixe_set_amo_remap_to_pcie_fadd(struct cass_dev *hw, int amo_remap)
{
	union c_ixe_cfg_amo_offload amo_offload;
	union c_ixe_cfg_amo_offload_code_1op amo_offload_code_1op = {};
	union c_ixe_cfg_amo_offload_en amo_offload_en = {};

	if (!cass_ixe_valid_amo_remap_to_pcie_fadd(amo_remap))
		return -EINVAL;

	/* Clear current AMO offload configuration. */
	cass_read(hw, C_IXE_CFG_AMO_OFFLOAD, &amo_offload,
		  sizeof(amo_offload));
	amo_offload.base_amo_req = 0;
	cass_write(hw, C_IXE_CFG_AMO_OFFLOAD, &amo_offload,
		   sizeof(amo_offload));

	cass_write(hw, C_IXE_CFG_AMO_OFFLOAD_CODE_1OP, &amo_offload_code_1op,
		   sizeof(amo_offload_code_1op));
	cass_write(hw, C_IXE_CFG_AMO_OFFLOAD_EN, &amo_offload_en,
		   sizeof(amo_offload_en));

	switch (amo_remap) {
	case -1:
		goto out;

	case C_AMO_OP_MIN:
		amo_offload_en.op_min =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_MAX:
		amo_offload_en.op_max =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_SUM:
		amo_offload_en.op_sum =
			OFFLOAD_INT32_INT64_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_LOR:
		amo_offload_en.op_lor =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_LAND:
		amo_offload_en.op_land =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_BOR:
		amo_offload_en.op_bor =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_BAND:
		amo_offload_en.op_band =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_LXOR:
		amo_offload_en.op_lxor =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_BXOR:
		amo_offload_en.op_bxor =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;

	case C_AMO_OP_SWAP:
		amo_offload_en.op_swap =
			OFFLOAD_UINT32_UINT64_FETCH_NON_FETCH_1OP;
		break;
	}

	cass_write(hw, C_IXE_CFG_AMO_OFFLOAD_EN, &amo_offload_en,
		   sizeof(amo_offload_en));

out:
	hw->cdev.prop.amo_remap_to_pcie_fadd = amo_remap;

	return 0;
}

#define ABORT_MASK_BITS (sizeof(union c_ixe_cfg_ipv6_opt_abort) * 8)
int cass_ixe_init(struct cass_dev *hw)
{
	int i;
	unsigned int next_header;
	unsigned int index;
	unsigned int bit;
	union c_ixe_cfg_ipv6_opt_abort masks[C_IXE_CFG_IPV6_OPT_ABORT_ENTRIES];
	static const unsigned int ipv6_extension_headers[] = {
		IPV6_EH_HOPOPTS,
		IPV6_EH_ROUTE,
		IPV6_EH_FRAG,
		IPV6_EH_ESP,
		IPV6_EH_AH,
		IPV6_EH_OPTS,
		IPV6_EH_MOBILITY,
		IPV6_EH_HIP,
		IPV6_EH_SHIM6,
		IPV6_EH_RSVD1,
		IPV6_EH_RSVD2,
	};
	union c_ixe_cfg_disp_cdt_lim cdt_limits;
	int ret;

	/* Initially mark all IPv6 next header values as ULP. */
	for (i = 0; i < ARRAY_SIZE(masks); i++)
		masks[i].qw = -1;

	/* Mask off IPv6 extension headers next header values. */
	for (i = 0; i < ARRAY_SIZE(ipv6_extension_headers); i++) {
		next_header = ipv6_extension_headers[i];
		index = next_header / ABORT_MASK_BITS;
		bit = next_header % ABORT_MASK_BITS;

		masks[index].qw &= ~BIT(bit);
	}

	/* Write the IPv6 next header configuration. */
	for (i = 0; i < ARRAY_SIZE(masks); i++)
		cass_write(hw, C_IXE_CFG_IPV6_OPT_ABORT(i), &masks[i],
			   sizeof(masks[i]));

	if (cass_version(hw, CASSINI_1_0)) {
		/* Errata 2973 */
		union c1_pct_cfg_ixe_req_fifo_limits req_limits;

		cass_read(hw, C1_PCT_CFG_IXE_REQ_FIFO_LIMITS, &req_limits,
			  sizeof(req_limits));

		req_limits.ixe_req_clr2_limit = 1;

		cass_write(hw, C1_PCT_CFG_IXE_REQ_FIFO_LIMITS, &req_limits,
			   sizeof(req_limits));

		cass_read(hw, C_IXE_CFG_DISP_CDT_LIM, &cdt_limits,
			  sizeof(cdt_limits));
		cdt_limits.atu_req_cdts = 3;
		cass_write(hw, C_IXE_CFG_DISP_CDT_LIM, &cdt_limits,
			   sizeof(cdt_limits));
	}

	if (cass_version(hw, CASSINI_2)) {
		union c2_ixe_cfg_disp_ord ord;

		cass_read(hw, C2_IXE_CFG_DISP_ORD, &ord, sizeof(ord));
		ord.get_ro_body = 1;
		ord.get_ro_last = 1;
		cass_write(hw, C2_IXE_CFG_DISP_ORD, &ord, sizeof(ord));

		cass_read(hw, C_IXE_CFG_DISP_CDT_LIM, &cdt_limits,
			  sizeof(cdt_limits));
		cdt_limits.atu_req_cdts = 4;
		cass_write(hw, C_IXE_CFG_DISP_CDT_LIM, &cdt_limits,
			   sizeof(cdt_limits));
	}

	/* RoCE */
	set_bth_opcode(hw);

	ret = cass_ixe_set_amo_remap_to_pcie_fadd(hw, amo_remap_to_pcie_fadd);
	if (ret)
		pr_err("Failed to remap NIC AMO to PCIe fetch add: %d\n", ret);

	return ret;
}
