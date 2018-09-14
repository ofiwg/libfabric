/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

/* Cassini supports ONLY 1-element vectors, and this code presumes that the
 * value is 1.
 */
_Static_assert(CXIP_AMO_MAX_IOV == 1, "Unexpected max IOV #");

/* Cassini supports ONLY 1-element packed IOVs.
 */
#define CXIP_AMO_MAX_PACKED_IOV (1)

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

/**
 * Request variants:
 *   CXIP_RQ_AMO
 *      Passes one argument (operand1), and applies that to a remote memory
 *      address content.
 *
 *   CXIP_RQ_AMO_FETCH
 *      Passes two arguments (operand1, resultptr), applies operand1 to a remote
 *      memory address content, and returns the prior content of the remote
 *      memory in resultptr.
 *
 *   CXIP_RQ_AMO_SWAP
 *      Passes three arguments (operand1, compare, resultptr). If remote memory
 *      address content satisfies the comparison operation with compare,
 *      replaces the remote memory content with operand1, and returns the prior
 *      content of the remote memory in resultptr.
 *
 */
enum cxip_amo_req_type {
	CXIP_RQ_AMO,
	CXIP_RQ_AMO_FETCH,
	CXIP_RQ_AMO_SWAP,
	CXIP_RQ_AMO_LAST
};

/**
 * Data type codes for all of the supported fi_datatype values.
 */
static enum c_atomic_type _cxip_amo_type_code[FI_DATATYPE_LAST] = {
	[FI_INT8]	  = C_AMO_TYPE_INT8_T,
	[FI_UINT8]	  = C_AMO_TYPE_UINT8_T,
	[FI_INT16]	  = C_AMO_TYPE_INT16_T,
	[FI_UINT16]	  = C_AMO_TYPE_UINT16_T,
	[FI_INT32]	  = C_AMO_TYPE_INT32_T,
	[FI_UINT32]	  = C_AMO_TYPE_UINT32_T,
	[FI_INT64]	  = C_AMO_TYPE_INT64_T,
	[FI_UINT64]	  = C_AMO_TYPE_UINT64_T,
	[FI_FLOAT]	  = C_AMO_TYPE_FLOAT_T,
	[FI_DOUBLE]	  = C_AMO_TYPE_DOUBLE_T,
	[FI_FLOAT_COMPLEX]	  = C_AMO_TYPE_FLOAT_COMPLEX_T,
	[FI_DOUBLE_COMPLEX]	  = C_AMO_TYPE_DOUBLE_COMPLEX_T,
};
//TODO: C_AMO_TYPE_UINT128_T

/**
 * AMO operation codes for all of the fi_op values.
 */
static enum c_atomic_op _cxip_amo_op_code[FI_ATOMIC_OP_LAST] = {
	[FI_MIN]	  = C_AMO_OP_MIN,
	[FI_MAX]	  = C_AMO_OP_MAX,
	[FI_SUM]	  = C_AMO_OP_SUM,
	[FI_LOR]	  = C_AMO_OP_LOR,
	[FI_LAND]	  = C_AMO_OP_LAND,
	[FI_BOR]	  = C_AMO_OP_BOR,
	[FI_BAND]	  = C_AMO_OP_BAND,
	[FI_LXOR]	  = C_AMO_OP_LXOR,
	[FI_BXOR]	  = C_AMO_OP_BXOR,
	[FI_ATOMIC_READ]  = C_AMO_OP_SUM,	/* special handling */
	[FI_ATOMIC_WRITE] = C_AMO_OP_SWAP,	/* special handling */
	[FI_CSWAP]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_NE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_LE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_LT]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_GE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_GT]	  = C_AMO_OP_CSWAP,
	[FI_MSWAP]	  = C_AMO_OP_AXOR,	/* special handling */
};

/**
 * AMO swap operation codes for the CSWAP comparison conditions.
 */
static enum c_cswap_op _cxip_amo_swpcode[FI_ATOMIC_OP_LAST] = {
	[FI_CSWAP]	  = C_AMO_OP_CSWAP_EQ,
	[FI_CSWAP_NE]	  = C_AMO_OP_CSWAP_NE,
	[FI_CSWAP_LE]	  = C_AMO_OP_CSWAP_LE,
	[FI_CSWAP_LT]	  = C_AMO_OP_CSWAP_LT,
	[FI_CSWAP_GE]	  = C_AMO_OP_CSWAP_GE,
	[FI_CSWAP_GT]	  = C_AMO_OP_CSWAP_GT,
};

/**
 * Multi-dimensional array defining supported/unsupported operations. Bits
 * correspond to the 14 possible fi_datatype values. The OP_VALID() macro will
 * return a 1 if the (request,op,dt) triple is supported by Cassini.
 */
static uint16_t _cxip_amo_valid[CXIP_RQ_AMO_LAST][FI_ATOMIC_OP_LAST] = {

	[CXIP_RQ_AMO] = {
		[FI_MIN]	  = 0x03ff,
		[FI_MAX]	  = 0x03ff,
		[FI_SUM]	  = 0x0fff,
		[FI_LOR]	  = 0x00ff,
		[FI_LAND]	  = 0x00ff,
		[FI_LXOR]	  = 0x00ff,
		[FI_BOR]	  = 0x00ff,
		[FI_BAND]	  = 0x00ff,
		[FI_BXOR]	  = 0x00ff,
		[FI_ATOMIC_WRITE] = 0x0fff,
	},

	[CXIP_RQ_AMO_FETCH] = {
		[FI_MIN]	  = 0x03ff,
		[FI_MAX]	  = 0x03ff,
		[FI_SUM]	  = 0x0fff,
		[FI_LOR]	  = 0x00ff,
		[FI_LAND]	  = 0x00ff,
		[FI_LXOR]	  = 0x00ff,
		[FI_BOR]	  = 0x00ff,
		[FI_BAND]	  = 0x00ff,
		[FI_BXOR]	  = 0x00ff,
		[FI_ATOMIC_WRITE] = 0x0fff,
		[FI_ATOMIC_READ]  = 0x0fff,
	},

	[CXIP_RQ_AMO_SWAP] = {
		[FI_CSWAP]	  = 0x0fff,
		[FI_CSWAP_NE]	  = 0x0fff,
		[FI_CSWAP_LE]	  = 0x03ff,
		[FI_CSWAP_LT]	  = 0x03ff,
		[FI_CSWAP_GE]	  = 0x03ff,
		[FI_CSWAP_GT]	  = 0x03ff,
		[FI_MSWAP]	  = 0x00ff,
	},
};
#define	OP_VALID(rq, op, dt)	(_cxip_amo_valid[rq][op] & (1 << dt))

/**
 * Supply opcodes for a request, and determine if the operation is supported.
 *
 * @param req_type basic, fetch, or swap
 * @param dt data type for operation
 * @param op operation
 * @param cop Cassini code for operation
 * @param cdt Cassini code for data type
 * @param copswp Cassini code for cswap operation
 * @param cdtlen Length of datatype in bytes
 *
 * @return int 0 on success, -FI_EOPNOTSUPP if operation is not supported
 */
static int _cxip_atomic_opcode(enum cxip_amo_req_type req_type,
			       enum fi_datatype dt, enum fi_op op,
			       enum c_atomic_op *cop, enum c_atomic_type *cdt,
			       enum c_cswap_op *copswp, int *cdtlen)
{
	int opcode;
	int dtcode;

	if (dt < 0 || dt >= FI_DATATYPE_LAST ||
	    op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	if (!OP_VALID(req_type, op, dt))
		return -FI_EOPNOTSUPP;

	opcode = _cxip_amo_op_code[op];
	dtcode = _cxip_amo_type_code[dt];
	if (opcode < 0 || dtcode < 0)
		return -FI_EOPNOTSUPP;

	if (cop)
		*cop = opcode;
	if (cdt)
		*cdt = dtcode;
	if (cdtlen)
		*cdtlen = ofi_datatype_size(dt);
	if (copswp)
		*copswp = _cxip_amo_swpcode[op];

	return 0;
}

/**
 * Implementation of the provider *_atomic_valid() functions.
 *
 * The returned count is the maximum number of atomic objects on which a single
 * atomic call can operate. For Cassini, this is 1.
 *
 * @param ep endpoint
 * @param req_type request type
 * @param datatype datatype
 * @param op operation
 * @param count returns count of operations supported
 *
 * @return int 0 on success, -FI_EOPNOTSUPP if operation not supported
 */
static inline int _cxip_ep_valid(struct fid_ep *ep,
				 enum cxip_amo_req_type req_type,
				 enum fi_datatype datatype,
				 enum fi_op op,
				 size_t *count)
{
	int ret;

	/* Endpoint must have atomics enabled */
	if (!ep->atomic)
		return -FI_EINVAL;

	/* Check for a valid opcode */
	ret = _cxip_atomic_opcode(req_type, datatype, op,
				  NULL, NULL, NULL, NULL);
	if (ret < 0)
		return ret;

	/* "Cassini implements single element atomics. There is no hardware
	 *  support for packed atomics or IOVECs." -- CSDG
	 */
	if (count)
		*count = CXIP_AMO_MAX_IOV;

	return 0;
}

/**
 * Callback for non-fetching AMO operations.
 *
 * @param req AMO request structure
 * @param event resulting event
 */
static void _cxip_amo_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;

	event_rc = event->init_short.return_code;
	if (event_rc == C_RC_OK) {
		ret = req->cq->report_completion(req->cq, FI_ADDR_UNSPEC, req);
		if (ret != req->cq->cq_entry_size)
			CXIP_LOG_ERROR("Failed to report completion: %d\n",
				       ret);
	} else {
		ret = cxip_cq_report_error(req->cq, req, 0, FI_EIO, event_rc,
					   NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}

	cxip_cq_req_free(req);
}

/**
 * Callback for fetching and swap AMO operations.
 *
 * @param req AMO request structure
 * @param event resulting event
 */
static void _cxip_famo_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;

	/* Free the local_result buffer, if allocated (free is NULL-safe) */
	free((void *)req->buf);
	req->buf = 0;

	ret = cxil_unmap(req->cq->domain->dev_if->if_lni,
			 &req->local_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

	_cxip_amo_cb(req, event);
}

/**
 * Return true if vector specification is valid.
 *
 * vn must be > 0 and <= 1 (CXIP_AMO_MAX_IOV). Formally, we could do this test,
 * but formally we would have to loop (once) over the vectors, and test each
 * count for being > 0 and <= 1 (CXIP_AMO_MAX_PACKED_IOV). Instead, we just test
 * to ensure that each is 1.
 *
 * @param vn vector element count
 * @param v vector pointer
 *
 * @return bool true if vector is valid, false otherwise
 */
static inline bool _vector_valid(size_t vn, const struct fi_ioc *v)
{
	return (vn == CXIP_AMO_MAX_IOV && v &&
		v[0].count == CXIP_AMO_MAX_PACKED_IOV &&
		v[0].addr);
}

/**
 * Return true if RMA vector specification is valid. Note that the address is
 * treated as an offset into an RMA MR window, so a value of zero is valid.
 *
 * @param vn vector element count
 * @param v vector pointer
 *
 * @return bool true if RMA vector is valid, false otherwise
 */
static inline bool _rma_vector_valid(size_t vn, const struct fi_rma_ioc *v)
{
	return (vn == CXIP_AMO_MAX_IOV && v &&
		v[0].count == CXIP_AMO_MAX_PACKED_IOV);
}

/**
 * Core implementation of all of the atomic operations.
 *
 * @param req_type basic, fetch, or swap
 * @param ep endpoint
 * @param msg atomic operation message
 * @param comparev compare value vector
 * @param comparedesc compare vector descriptors
 * @param compare_count compare vector count
 * @param resultv result pointer vector
 * @param resultdesc result vector descriptors
 * @param result_count result vector count
 * @param flags operation flags
 *
 * @return int FI_SUCCESS on success, negative value on failure
 */
static int _cxip_idc_amo(enum cxip_amo_req_type req_type, struct fid_ep *ep,
			 const struct fi_msg_atomic *msg,
			 const struct fi_ioc *comparev, void **comparedesc,
			 size_t compare_count,
			 const struct fi_ioc *resultv, void **resultdesc,
			 size_t result_count,
			 uint64_t flags)
{
	struct cxip_tx_ctx *txc;
	struct cxip_if *dev_if;
	struct cxi_iova result_md = {};
	struct cxip_addr caddr;
	struct cxip_req *req;
	enum c_atomic_op opcode;
	enum c_cswap_op swpcode;
	enum c_atomic_type dtcode;
	union c_cmdu cmd = {};
	union c_cmdu state = {};
	union c_fab_addr dfa;
	uint32_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;
	uint32_t result_lac = 0;
	uint64_t result_iova = 0;
	uint64_t report_flags = 0;
	void *local_result = NULL;
	void *compare = NULL;
	void *result = NULL;
	void *oper1 = NULL;
	uint64_t local_oper1[2];
	uint64_t local_compare[2];
	uint64_t off = 0;
	uint64_t key = 0;
	int len;
	int ret;

	if (!msg)
		return -FI_EINVAL;

	/* fi_atomic(3) documentation isn't pellucid, but it appears that the
	 * iov address itself can represent a sequential collection of datatype
	 * objects, and the iov 'count' is the number of these: an implicit
	 * packed vector of contiguous atomic values which is covered by a
	 * single desc descriptor. There can also be multiple vectors, each with
	 * its own descriptor. CASSINI has hardware supportfor single-value
	 * atomics ONLY, so there is no hardware boost for vectors, and thus no
	 * real point to using the vectorized calls. We enforce that by limiting
	 * calls to a single value.
	 */

	switch (req_type) {
	case CXIP_RQ_AMO_SWAP:
		/* Must have a valid compare address */
		if (!_vector_valid(compare_count, comparev))
			return -FI_EINVAL;
		compare = comparev[0].addr;
		/* FALLTHRU */
	case CXIP_RQ_AMO_FETCH:
		/* Must have a valid result address */
		if (!_vector_valid(result_count, resultv))
			return -FI_EINVAL;
		result = resultv[0].addr;
		/* FALLTHRU */
	case CXIP_RQ_AMO:
		if (!_vector_valid(msg->iov_count, msg->msg_iov))
			return -FI_EINVAL;
		/* The supplied RMA address is actually an offset into a
		 * registered MR. A value of 0 is valid.
		 */
		if (!_rma_vector_valid(msg->rma_iov_count, msg->rma_iov))
			return -FI_EINVAL;
		oper1 = msg->msg_iov[0].addr;
		off = msg->rma_iov[0].addr;
		key = msg->rma_iov[0].key;
		break;
	default:
		return -FI_EINVAL;
	}

	/* Convert FI to CXI codes, fail if operation not supported */
	ret = _cxip_atomic_opcode(req_type, msg->datatype, msg->op,
				  &opcode, &dtcode, &swpcode, &len);
	if (ret < 0)
		return ret;

	/* Cassini doesn't supply AMO WRITE.
	 *
	 * We implement a fetching AMO WRITE with C_AMO_OP_SWAP, which does
	 * exactly what we want and works for all data types.
	 *
	 * For the non-fetching AMO WRITE, we use C_AMO_OP_AXOR when we can.
	 * Otherwise, we use C_AMO_OP_SWAP with a throwaway buffer.
	 */
	if (msg->op == FI_ATOMIC_WRITE && !result) {
		/* AXOR for float, double, and float complex are not supported
		 * by Cassini, but since we are only trying to write the value
		 * unchanged, we can pretend these are arbitrary bit patterns in
		 * UINT32 or UINT64, and take advantage of AXOR.
		 */
		switch (dtcode) {
		case C_AMO_TYPE_FLOAT_T:
			dtcode = C_AMO_TYPE_UINT32_T;
			break;
		case C_AMO_TYPE_FLOAT_COMPLEX_T:
		case C_AMO_TYPE_DOUBLE_T:
			dtcode = C_AMO_TYPE_UINT64_T;
			break;
		default:
			break;
		}

		switch (dtcode) {
		case C_AMO_TYPE_DOUBLE_COMPLEX_T:
		case C_AMO_TYPE_UINT128_T:
			/* 128-bit quantities must be SWAPPED */
			local_result = calloc(1, len);
			if (!local_result) {
				CXIP_LOG_ERROR("Failed local result alloc\n");
				return -FI_ENOMEM;
			}
			result = local_result;
			break;
		default:
			/* Anything else can use AXOR with a set mask */
			opcode = C_AMO_OP_AXOR;
			memset(local_compare, -1, len);
			compare = local_compare;
			break;
		}
	}

	/* Cassini doesn't supply AMO READ. We implement it with a fetching AMO
	 * SUM, with the operand forced to zero. There is no non-fetching
	 * version of READ, so we don't need to worry about creating a local
	 * buffer. The AMO SUM works for all data types.
	 */
	if (msg->op == FI_ATOMIC_READ) {
		memset(local_oper1, 0, len);
		oper1 = local_oper1;
		report_flags |= FI_READ;
	} else {
		report_flags |= FI_WRITE;
	}

	/* Cassini doesn't supply AMO MSWAP. It instead supplies AMO AXOR.
	 *
	 * AXOR  = (*addr & ~mask) ^ data
	 * MSWAP = (*addr & ~mask) | (data & mask)
	 * data  = oper1
	 * mask  = compare
	 *
	 * However,
	 *
	 * (*addr & ~mask) | (data & mask) == (*addr & ~mask) ^ (data & mask)
	 *
	 * since no bit can be 1 on both sides, due to the masking.
	 */
	if (msg->op == FI_MSWAP) {
		uint64_t *mask = compare;

		memcpy(local_oper1, oper1, len);
		local_oper1[0] &= mask[0];
		if (len > sizeof(uint64_t))
			local_oper1[1] &= mask[1];
		oper1 = local_oper1;
	}

	/* The input FID could be a standard endpoint (containing a TX
	 * context), or a TX context itself.
	 */
	switch (ep->fid.fclass) {
		struct cxip_ep *cxi_ep;
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		txc = cxi_ep->attr->tx_ctx;
		break;
	case FI_CLASS_TX_CTX:
		txc = container_of(ep, struct cxip_tx_ctx, fid.ctx);
		break;
	default:
		CXIP_LOG_ERROR("Invalid EP type: %zd\n", ep->fid.fclass);
		return -FI_EINVAL;
	}

	dev_if = txc->domain->dev_if;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->av, msg->addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up dst FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer for fetching AMO */
	if (result) {
		/* Map local buffer */
		ret = cxil_map(dev_if->if_lni, result, len,
			       CXI_MAP_PIN | CXI_MAP_NTA |
			       CXI_MAP_READ | CXI_MAP_WRITE,
			       &result_md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map result buffer: %d\n", ret);
			return ret;
		}
		result_iova = CXI_VA_TO_IOVA(&result_md, result);
		result_lac = result_md.lac;
	}

	/* Allocate a CQ request */
	req = cxip_cq_req_alloc(txc->comp.send_cq, 0);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto unmap_amo;
	}

	/* Values set here are passed back to the user through the CQ */
	req->context = (uint64_t)msg->context;
	req->flags = FI_ATOMIC | report_flags;
	req->data_len = 0;
	req->buf = (uint64_t)local_result;
	req->data = 0;
	req->tag = 0;
	req->local_md = result_md;
	req->cb = (result) ? _cxip_famo_cb : _cxip_amo_cb;

	/* Build AMO command descriptor */
	pid_granule = dev_if->if_pid_granule;
	pid_idx = CXIP_ADDR_MR_IDX(pid_granule, key);
	cxi_build_dfa(caddr.nic, caddr.port, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	state.c_state.write_lac = result_lac;
	state.c_state.event_send_disable = 1;
	state.c_state.restricted = 1;
	state.c_state.index_ext = idx_ext;
	state.c_state.user_ptr = (uint64_t)req;
	state.c_state.eq = txc->comp.send_cq->evtq->eqn;

	cmd.idc_amo.idc_header.dfa = dfa;
	cmd.idc_amo.idc_header.remote_offset = off;
	cmd.idc_amo.atomic_op = opcode;
	cmd.idc_amo.atomic_type = dtcode;
	cmd.idc_amo.cswap_op = swpcode;
	cmd.idc_amo.local_addr = result_iova;
	/* Note: 16-byte value will overflow into op1_word2 */
	memcpy(&cmd.idc_amo.op1_word1, oper1, len);
	if (compare)
		memcpy(&cmd.idc_amo.op2_word1, compare, len);

	fastlock_acquire(&txc->lock);

	/* Issue a CSTATE command */
	ret = cxi_cq_emit_c_state(txc->tx_cmdq, &state.c_state);
	if (ret) {
		CXIP_LOG_DBG("Failed to issue CSTATE command: %d\n", ret);

		/* Return error according to Domain Resource Management */
		ret = -FI_EAGAIN;
		goto unlock_amo;
	}

	/* Issue IDC AMO command */
	ret = cxi_cq_emit_idc_amo(txc->tx_cmdq, &cmd.idc_amo, len,
				  result != NULL);
	if (ret) {
		CXIP_LOG_DBG("Failed to issue IDC AMO command: %d\n", ret);

		/* Return error according to Domain Resource Management */
		ret = -FI_EAGAIN;
		goto unlock_amo;
	}

	cxi_cq_ring(txc->tx_cmdq);

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

unlock_amo:
	fastlock_release(&txc->lock);
	cxip_cq_req_free(req);

unmap_amo:
	if (result_count)
		cxil_unmap(dev_if->if_lni, &result_md);

	return ret;
}

static ssize_t cxip_ep_atomic_write(struct fid_ep *ep, const void *buf,
				    size_t count, void *desc,
				    fi_addr_t dest_addr, uint64_t addr,
				    uint64_t key, enum fi_datatype datatype,
				    enum fi_op op, void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_ioc oper1 = {
		.addr = (void *)buf,
		.count = count
	};
	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = &oper1,
		.desc = &desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO, ep, &msg,
			     NULL, NULL, 0,
			     NULL, NULL, 0,
			     flags);
}

static ssize_t cxip_ep_atomic_writev(struct fid_ep *ep,
				     const struct fi_ioc *iov, void **desc,
				     size_t count, fi_addr_t dest_addr,
				     uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op,
				     void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO, ep, &msg,
			     NULL, NULL, 0,
			     NULL, NULL, 0,
			     flags);
}

static ssize_t cxip_ep_atomic_writemsg(struct fid_ep *ep,
				       const struct fi_msg_atomic *msg,
				       uint64_t flags)
{
	return _cxip_idc_amo(CXIP_RQ_AMO, ep, msg,
			     NULL, NULL, 0,
			     NULL, NULL, 0,
			     flags);
}

static ssize_t cxip_ep_atomic_inject(struct fid_ep *ep, const void *buf,
				     size_t count, fi_addr_t dest_addr,
				     uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op)
{
	uint64_t flags = FI_INJECT;

	struct fi_ioc oper1 = {
		.addr = (void *)buf,
		.count = count
	};
	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = &oper1,
		.desc = NULL,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = NULL
	};

	return _cxip_idc_amo(CXIP_RQ_AMO, ep, &msg,
			     NULL, NULL, 0,
			     NULL, NULL, 0,
			     flags);
}

static ssize_t cxip_ep_atomic_readwrite(struct fid_ep *ep, const void *buf,
					size_t count, void *desc, void *result,
					void *result_desc, fi_addr_t dest_addr,
					uint64_t addr, uint64_t key,
					enum fi_datatype datatype,
					enum fi_op op, void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_ioc oper1 = {
		.addr = (void *)buf,
		.count = count
	};
	struct fi_ioc resultv = {
		.addr = result,
		.count = count
	};
	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = &oper1,
		.desc = &desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, &msg,
			     NULL, NULL, 0,
			     &resultv, &result_desc, 1,
			     flags);
}

static ssize_t cxip_ep_atomic_readwritev(struct fid_ep *ep,
					 const struct fi_ioc *iov,
					 void **desc, size_t count,
					 struct fi_ioc *resultv,
					 void **result_desc,
					 size_t result_count,
					 fi_addr_t dest_addr, uint64_t addr,
					 uint64_t key,
					 enum fi_datatype datatype,
					 enum fi_op op, void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, &msg,
			     NULL, NULL, 0,
			     resultv, result_desc, result_count,
			     flags);
}

static ssize_t cxip_ep_atomic_readwritemsg(struct fid_ep *ep,
					   const struct fi_msg_atomic *msg,
					   struct fi_ioc *resultv,
					   void **result_desc,
					   size_t result_count, uint64_t flags)
{
	return _cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, msg,
			     NULL, NULL, 0,
			     resultv, result_desc, result_count,
			     flags);
}

static ssize_t cxip_ep_atomic_compwrite(struct fid_ep *ep, const void *buf,
					size_t count, void *desc,
					const void *compare, void *compare_desc,
					void *result, void *result_desc,
					fi_addr_t dest_addr, uint64_t addr,
					uint64_t key, enum fi_datatype datatype,
					enum fi_op op, void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_ioc oper1 = {
		.addr = (void *)buf,
		.count = count
	};
	struct fi_ioc comparev = {
		.addr = (void *)compare,
		.count = count
	};
	struct fi_ioc resultv = {
		.addr = result,
		.count = count
	};
	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = &oper1,
		.desc = &desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, &msg,
			     &comparev, &result_desc, 1,
			     &resultv, &result_desc, 1,
			     flags);
}

static ssize_t cxip_ep_atomic_compwritev(struct fid_ep *ep,
					 const struct fi_ioc *iov, void **desc,
					 size_t count,
					 const struct fi_ioc *comparev,
					 void **compare_desc,
					 size_t compare_count,
					 struct fi_ioc *resultv,
					 void **result_desc,
					 size_t result_count,
					 fi_addr_t dest_addr, uint64_t addr,
					 uint64_t key,
					 enum fi_datatype datatype,
					 enum fi_op op, void *context)
{
	uint64_t flags = FI_COMPLETION;

	struct fi_rma_ioc rma = {
		.addr = addr,
		.count = 1,
		.key = key
	};
	struct fi_msg_atomic msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.rma_iov = &rma,
		.rma_iov_count = 1,
		.datatype = datatype,
		.op = op,
		.context = context
	};

	return _cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, &msg,
			     comparev, compare_desc, compare_count,
			     resultv, result_desc, result_count,
			     flags);
}

static ssize_t
cxip_ep_atomic_compwritemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
			    const struct fi_ioc *comparev, void **compare_desc,
			    size_t compare_count, struct fi_ioc *resultv,
			    void **result_desc, size_t result_count,
			    uint64_t flags)
{
	return _cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, msg,
			     comparev, compare_desc, compare_count,
			     resultv, result_desc, result_count,
			     flags);
}

static int cxip_ep_atomic_valid(struct fid_ep *ep,
				enum fi_datatype datatype,
				enum fi_op op,
				size_t *count)
{
	return _cxip_ep_valid(ep, CXIP_RQ_AMO, datatype, op, count);
}

static int cxip_ep_fetch_atomic_valid(struct fid_ep *ep,
				      enum fi_datatype datatype, enum fi_op op,
				      size_t *count)
{
	return _cxip_ep_valid(ep, CXIP_RQ_AMO_FETCH, datatype, op, count);
}

static int cxip_ep_comp_atomic_valid(struct fid_ep *ep,

				     enum fi_datatype datatype,
				     enum fi_op op, size_t *count)
{
	return _cxip_ep_valid(ep, CXIP_RQ_AMO_SWAP, datatype, op, count);
}

struct fi_ops_atomic cxip_ep_atomic = {
	.size = sizeof(struct fi_ops_atomic),
	.write = cxip_ep_atomic_write,
	.writev = cxip_ep_atomic_writev,
	.writemsg = cxip_ep_atomic_writemsg,
	.inject = cxip_ep_atomic_inject,
	.readwrite = cxip_ep_atomic_readwrite,
	.readwritev = cxip_ep_atomic_readwritev,
	.readwritemsg = cxip_ep_atomic_readwritemsg,
	.compwrite = cxip_ep_atomic_compwrite,
	.compwritev = cxip_ep_atomic_compwritev,
	.compwritemsg = cxip_ep_atomic_compwritemsg,
	.writevalid = cxip_ep_atomic_valid,
	.readwritevalid = cxip_ep_fetch_atomic_valid,
	.compwritevalid = cxip_ep_comp_atomic_valid,
};

