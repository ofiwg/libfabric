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

/*
 * cxip_amo_inject_cb() - AMO inject event callback.
 */
static int cxip_amo_inject_cb(struct cxip_req *req, const union c_event *event)
{
	return cxip_cq_req_error(req, 0, FI_EIO, cxi_event_rc(event), NULL, 0);
}

/*
 * cxip_amo_inject_req() - Return request state associated with all AMO inject
 * transactions on the transmit context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_amo_inject_req(struct cxip_txc *txc)
{
	if (!txc->amo_inject_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_amo_inject_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_ATOMIC | FI_WRITE;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->addr = FI_ADDR_UNSPEC;

		txc->amo_inject_req = req;
	}

	return txc->amo_inject_req;
}

/**
 * Callback for non-fetching AMO operations.
 *
 * @param req AMO request structure
 * @param event resulting event
 */
static int _cxip_amo_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	int success_event = (req->flags & FI_COMPLETION);

	if (req->amo.result_md)
		cxip_unmap(req->amo.result_md);

	if (req->amo.oper1_md)
		cxip_unmap(req->amo.oper1_md);

	if (req->amo.ibuf)
		cxip_cq_ibuf_free(req->cq, req->amo.ibuf);

	req->flags &= (FI_ATOMIC | FI_READ | FI_WRITE);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report completion: %d\n",
					       ret);
		}
	} else {
		ret = cxip_cq_req_error(req, 0, FI_EIO, event_rc, NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}

	ofi_atomic_dec32(&req->amo.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
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
 * @param triggered is a triggered amo operation
 * @param trig_thresh triggered threshold
 * @param trig_cntr triggered counter
 * @param comp_cntr completion counter for triggered operation
 *
 * @return int FI_SUCCESS on success, negative value on failure
 */
int cxip_amo_common(enum cxip_amo_req_type req_type, struct cxip_txc *txc,
		    const struct fi_msg_atomic *msg,
		    const struct fi_ioc *comparev, void **comparedesc,
		    size_t compare_count, const struct fi_ioc *resultv,
		    void **resultdesc, size_t result_count, uint64_t flags,
		    bool triggered, uint64_t trig_thresh,
		    struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr)
{
	struct cxip_addr caddr;
	struct cxip_req *req = NULL;
	enum c_atomic_op opcode;
	enum c_cswap_op swpcode;
	enum c_atomic_type dtcode;
	union c_cmdu cmd = {};
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_idx;
	void *compare = NULL;
	void *result = NULL;
	void *oper1 = NULL;
	uint64_t local_compare[2];
	uint64_t off = 0;
	uint64_t key = 0;
	int len;
	int ret;
	bool idc;
	struct cxip_cmdq *cmdq =
		triggered ? txc->domain->trig_cmdq : txc->tx_cmdq;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_rma_initiate_allowed(txc->attr.caps & ~FI_RMA))
		return -FI_ENOPROTOOPT;

	if (!msg)
		return -FI_EINVAL;

	/* Restricted AMOs must target optimized MRs without target events */
	if (flags & FI_CXI_UNRELIABLE &&
	    (!cxip_mr_key_opt(key) || txc->ep_obj->caps & FI_RMA_EVENT))
		return -FI_EINVAL;

	if (flags & FI_CXI_HRP && !(flags & FI_CXI_UNRELIABLE))
		return -FI_EINVAL;

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

	/* IDCs cannot be used to target standard MRs. */
	idc = cxip_mr_key_opt(key) && !triggered;

	/* Convert FI to CXI codes, fail if operation not supported */
	ret = _cxip_atomic_opcode(req_type, msg->datatype, msg->op,
				  &opcode, &dtcode, &swpcode, &len);
	if (ret < 0)
		return ret;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, msg->addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up dst FI addr: %d\n", ret);
		return ret;
	}

	/* Allocate a CQ request if:
	 * 1. Performing a fetching transfer
	 * 2. Not using an IDC
	 * 3. Tracking completion for the user
	 * 4. Implementing Atomic Write in software (using a fetching swap)
	 *
	 * State is not tracked for non-fetching, inject-style transfers that
	 * can be implemented with an IDC.
	 */
	if (result || !idc || (flags & FI_COMPLETION) ||
	    msg->op == FI_ATOMIC_WRITE) {
		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req) {
			CXIP_LOG_DBG("Failed to allocate request\n");
			return -FI_ENOMEM;
		}

		/* Values set here are passed back to the user through the CQ */
		if (flags & FI_COMPLETION)
			req->context = (uint64_t)msg->context;
		else
			req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_ATOMIC;
		req->flags |= (req_type == CXIP_RQ_AMO ? FI_WRITE : FI_READ);
		req->flags |= (flags & FI_COMPLETION);
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->cb = _cxip_amo_cb;
		req->amo.txc = txc;
		req->amo.oper1_md = NULL;
		req->amo.result_md = NULL;
		req->type = CXIP_REQ_AMO;
		req->trig_cntr = trig_cntr;
	}

	/* Prepare special case AMOs. */
	if (msg->op == FI_ATOMIC_WRITE && !result) {
		/* Non-fetching Atomic Write is implemented with an AXOR
		 * command for datatypes less than or equal to 8 bytes wide.
		 * With larger datatypes, a Swap is used with a throwaway
		 * buffer.
		 *
		 * AXOR for float, double, and float complex are not supported
		 * by Cassini, but since we are only trying to write the value
		 * unchanged, we can pretend these are arbitrary bit patterns
		 * in UINT32 or UINT64, and take advantage of AXOR.
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
			result = &req->amo.result;
			memset(result, 0, len);
			break;
		default:
			/* Anything else can use AXOR with a set mask */
			opcode = C_AMO_OP_AXOR;
			memset(local_compare, -1, len);
			compare = local_compare;
			break;
		}
	} else if (msg->op == FI_ATOMIC_READ) {
		/* Atomic Read is implemented with an Add-zero command. */
		memset(req->amo.oper1, 0, len);
		oper1 = &req->amo.oper1;
	} else if (msg->op == FI_MSWAP) {
		/* Atomic MSWAP is implemented using an AXOR command. See:
		 *   (*addr & ~mask) | (data & mask) ==
		 *      (*addr & ~mask) ^ (data & mask)
		 * where:
		 *   data = oper1
		 *   mask = compare
		 */
		uint64_t *mask = compare;
		uint64_t *tmp_oper = (uint64_t *)&req->amo.oper1;

		memcpy(tmp_oper, oper1, len);
		tmp_oper[0] &= mask[0];
		if (len > sizeof(uint64_t))
			tmp_oper[1] &= mask[1];
		oper1 = tmp_oper;
	}

	if (!idc) {
		if (flags & FI_INJECT) {
			/* Allocate an internal buffer to hold source data. */
			req->amo.ibuf = cxip_cq_ibuf_alloc(txc->send_cq);
			if (!req->amo.ibuf)
				goto free_req;

			memcpy(req->amo.ibuf, oper1, len);
		} else {
			/* Map user buffer for DMA command. */
			ret = cxip_map(txc->domain, oper1, len,
				       &req->amo.oper1_md);
			if (ret) {
				CXIP_LOG_DBG("Failed to map operand buffer: %d\n",
					     ret);
				goto free_req;
			}
		}
	}

	/* Map result buffer for fetching AMOs. */
	if (result) {
		/* Map local buffer */
		ret = cxip_map(txc->domain, result, len, &req->amo.result_md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map result buffer: %d\n", ret);
			goto unmap_oper1;
		}
	}

	/* Build destination fabric address. */
	pid_idx = cxip_mr_key_to_ptl_idx(key);
	cxi_build_dfa(caddr.nic, caddr.pid, txc->pid_bits, pid_idx, &dfa,
		      &idx_ext);

	fastlock_acquire(&cmdq->lock);

	ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
			      cxip_ofi_to_cxi_tc(txc->tclass),
			      flags & FI_CXI_HRP);
	if (ret != FI_SUCCESS)
		goto unlock_cmdq;

	if (flags & FI_FENCE) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_LOG_DBG("Failed to issue CQ_FENCE command: %d\n",
				     ret);
			ret = -FI_EAGAIN;
			goto unlock_cmdq;
		}
	}

	/* Build AMO command descriptor and write command. */
	if (idc) {
		if (result)
			cmd.c_state.write_lac = req->amo.result_md->md->lac;

		cmd.c_state.event_send_disable = 1;
		cmd.c_state.index_ext = idx_ext;
		cmd.c_state.eq = txc->send_cq->evtq->eqn;

		if (flags & FI_CXI_UNRELIABLE)
			cmd.c_state.restricted = 1;

		if (req) {
			cmd.c_state.user_ptr = (uint64_t)req;
		} else {
			void *inject_req = cxip_amo_inject_req(txc);

			if (!inject_req) {
				ret = -FI_ENOMEM;
				goto unlock_cmdq;
			}

			cmd.c_state.user_ptr = (uint64_t)inject_req;
			cmd.c_state.event_success_disable = 1;
		}

		if (req_type == CXIP_RQ_AMO) {
			if (txc->write_cntr) {
				cmd.c_state.event_ct_ack = 1;
				cmd.c_state.ct = txc->write_cntr->ct->ctn;
			}
		} else {
			if (txc->read_cntr) {
				cmd.c_state.event_ct_reply = 1;
				cmd.c_state.ct = txc->read_cntr->ct->ctn;
			}
		}

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			cmd.c_state.flush = 1;

		if (memcmp(&cmdq->c_state, &cmd.c_state, sizeof(cmd.c_state))) {
			ret = cxi_cq_emit_c_state(cmdq->dev_cmdq, &cmd.c_state);
			if (ret) {
				CXIP_LOG_DBG("Failed to issue C_STATE command: %d\n",
					     ret);

				/* Return error according to Domain Resource
				 * Management
				 */
				ret = -FI_EAGAIN;
				goto unlock_cmdq;
			}

			/* Update TXQ C_STATE */
			cmdq->c_state = cmd.c_state;

			CXIP_LOG_DBG("Updated C_STATE: %p\n", req);
		}

		memset(&cmd.idc_amo, 0, sizeof(cmd.idc_amo));
		cmd.idc_amo.idc_header.dfa = dfa;
		cmd.idc_amo.idc_header.remote_offset = off;
		cmd.idc_amo.atomic_op = opcode;
		cmd.idc_amo.atomic_type = dtcode;
		cmd.idc_amo.cswap_op = swpcode;

		if (result)
			cmd.idc_amo.local_addr =
				CXI_VA_TO_IOVA(req->amo.result_md->md, result);

		/* Note: 16-byte value will overflow into op1_word2 */
		memcpy(&cmd.idc_amo.op1_word1, oper1, len);
		if (compare)
			memcpy(&cmd.idc_amo.op2_word1, compare, len);

		/* Issue IDC AMO command */
		ret = cxi_cq_emit_idc_amo(cmdq->dev_cmdq, &cmd.idc_amo,
					  result != NULL);
		if (ret) {
			CXIP_LOG_DBG("Failed to issue IDC AMO command: %d\n",
				     ret);

			/* Return error according to Domain Resource Mgmt */
			ret = -FI_EAGAIN;
			goto unlock_cmdq;
		}
	} else {
		struct c_dma_amo_cmd cmd = {};
		struct cxip_cntr *cntr;

		cmd.index_ext = idx_ext;
		cmd.event_send_disable = 1;
		cmd.dfa = dfa;
		cmd.remote_offset = off;

		if (!req->amo.ibuf) {
			cmd.local_read_addr =
					CXI_VA_TO_IOVA(req->amo.oper1_md->md,
						       oper1);
			cmd.lac = req->amo.oper1_md->md->lac;
		} else {
			struct cxip_md *ibuf_md =
					cxip_cq_ibuf_md(req->amo.ibuf);

			cmd.local_read_addr = CXI_VA_TO_IOVA(ibuf_md->md,
							     req->amo.ibuf);
			cmd.lac = ibuf_md->md->lac;
		}

		if (result) {
			cmd.local_write_addr =
				CXI_VA_TO_IOVA(req->amo.result_md->md, result);
			cmd.write_lac = req->amo.result_md->md->lac;
		}

		cmd.request_len = len;
		cmd.eq = txc->send_cq->evtq->eqn;
		cmd.user_ptr = (uint64_t)req;
		cmd.match_bits = key;
		cmd.atomic_op = opcode;
		cmd.atomic_type = dtcode;
		cmd.cswap_op = swpcode;

		/* Note: 16-byte value will overflow into op2_word2 */
		if (compare)
			memcpy(&cmd.op2_word1, compare, len);

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			cmd.flush = 1;

		if (req_type == CXIP_RQ_AMO) {
			cntr = triggered ? comp_cntr : txc->write_cntr;

			if (cntr) {
				cmd.event_ct_ack = 1;
				cmd.ct = cntr->ct->ctn;
			}
		} else {
			cntr = triggered ? comp_cntr : txc->read_cntr;

			if (cntr) {
				cmd.event_ct_reply = 1;
				cmd.ct = cntr->ct->ctn;
			}
		}

		if (flags & FI_CXI_UNRELIABLE)
			cmd.restricted = 1;

		if (triggered) {
			const struct c_ct_cmd ct_cmd = {
				.trig_ct = trig_cntr->ct->ctn,
				.threshold = trig_thresh,
			};

			ret = cxi_cq_emit_trig_dma_amo(cmdq->dev_cmdq, &ct_cmd,
						       &cmd, result);
		} else {
			ret = cxi_cq_emit_dma_amo(cmdq->dev_cmdq, &cmd, result);
		}

		if (ret) {
			CXIP_LOG_DBG("Failed to write DMA AMO command: %d\n",
				     ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto unlock_cmdq;
		}
	}

	cxip_txq_ring(cmdq, flags & FI_MORE, triggered,
		      ofi_atomic_get32(&txc->otx_reqs));

	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	fastlock_release(&cmdq->lock);

#if ENABLE_DEBUG
	/* TODO better expose tostr API to providers */
	char op_str[32];
	char *static_str = fi_tostr(&msg->op, FI_TYPE_ATOMIC_OP);

	strcpy(op_str, static_str);
	CXIP_LOG_DBG("%sreq: %p op: %s type: %s buf: %p dest_addr: %ld context %p\n",
		     idc ? "IDC " : "", req, op_str,
		     fi_tostr(&msg->datatype, FI_TYPE_ATOMIC_TYPE),
		     oper1, msg->addr, msg->context);
#endif

	return FI_SUCCESS;

unlock_cmdq:
	fastlock_release(&txc->tx_cmdq->lock);

	if (result)
		cxip_unmap(req->amo.result_md);
unmap_oper1:
	if (req && req->amo.ibuf)
		cxip_cq_ibuf_free(req->cq, req->amo.ibuf);
	if (req && req->amo.oper1_md)
		cxip_unmap(req->amo.oper1_md);
free_req:
	if (req)
		cxip_cq_req_free(req);

	return ret;
}

/*
 * Libfabric APIs
 */

static ssize_t cxip_ep_atomic_write(struct fid_ep *ep, const void *buf,
				    size_t count, void *desc,
				    fi_addr_t dest_addr, uint64_t addr,
				    uint64_t key, enum fi_datatype datatype,
				    enum fi_op op, void *context)
{
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO, txc, &msg, NULL, NULL, 0, NULL,
			       NULL, 0, txc->attr.op_flags, false, 0, NULL,
			       NULL);
}

static ssize_t cxip_ep_atomic_writev(struct fid_ep *ep,
				     const struct fi_ioc *iov, void **desc,
				     size_t count, fi_addr_t dest_addr,
				     uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op,
				     void *context)
{
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO, txc, &msg, NULL, NULL, 0, NULL,
			       NULL, 0, txc->attr.op_flags, false, 0, NULL,
			       NULL);
}

static ssize_t cxip_ep_atomic_writemsg(struct fid_ep *ep,
				       const struct fi_msg_atomic *msg,
				       uint64_t flags)
{
	struct cxip_txc *txc;

	if (flags & ~(CXIP_WRITEMSG_ALLOWED_FLAGS | FI_CXI_UNRELIABLE))
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_amo_common(CXIP_RQ_AMO, txc, msg, NULL, NULL, 0, NULL, NULL,
			       0, flags, false, 0, NULL, NULL);
}

static ssize_t cxip_ep_atomic_inject(struct fid_ep *ep, const void *buf,
				     size_t count, fi_addr_t dest_addr,
				     uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op)
{
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO, txc, &msg, NULL, NULL, 0, NULL,
			       NULL, 0, FI_INJECT, false, 0, NULL, NULL);
}

static ssize_t cxip_ep_atomic_readwrite(struct fid_ep *ep, const void *buf,
					size_t count, void *desc, void *result,
					void *result_desc, fi_addr_t dest_addr,
					uint64_t addr, uint64_t key,
					enum fi_datatype datatype,
					enum fi_op op, void *context)
{
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO_FETCH, txc, &msg, NULL, NULL, 0,
			       &resultv, &result_desc, 1, txc->attr.op_flags,
			       false, 0, NULL, NULL);
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
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO_FETCH, txc, &msg, NULL, NULL, 0,
			       resultv, result_desc, result_count,
			       txc->attr.op_flags, false, 0, NULL, NULL);
}

static ssize_t cxip_ep_atomic_readwritemsg(struct fid_ep *ep,
					   const struct fi_msg_atomic *msg,
					   struct fi_ioc *resultv,
					   void **result_desc,
					   size_t result_count, uint64_t flags)
{
	struct cxip_txc *txc;

	if (flags & ~CXIP_WRITEMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_amo_common(CXIP_RQ_AMO_FETCH, txc, msg, NULL, NULL, 0,
			       resultv, result_desc, result_count, flags, false,
			       0, NULL, NULL);
}

static ssize_t cxip_ep_atomic_compwrite(struct fid_ep *ep, const void *buf,
					size_t count, void *desc,
					const void *compare, void *compare_desc,
					void *result, void *result_desc,
					fi_addr_t dest_addr, uint64_t addr,
					uint64_t key, enum fi_datatype datatype,
					enum fi_op op, void *context)
{
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO_SWAP, txc, &msg, &comparev,
			       &result_desc, 1, &resultv, &result_desc, 1,
			       txc->attr.op_flags, false, 0, NULL, NULL);
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
	struct cxip_txc *txc;

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_amo_common(CXIP_RQ_AMO_SWAP, txc, &msg, comparev,
			       compare_desc, compare_count, resultv,
			       result_desc, result_count, txc->attr.op_flags,
			       false, 0, NULL, NULL);
}

static ssize_t
cxip_ep_atomic_compwritemsg(struct fid_ep *ep, const struct fi_msg_atomic *msg,
			    const struct fi_ioc *comparev, void **compare_desc,
			    size_t compare_count, struct fi_ioc *resultv,
			    void **result_desc, size_t result_count,
			    uint64_t flags)
{
	struct cxip_txc *txc;

	if (flags & ~CXIP_WRITEMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_amo_common(CXIP_RQ_AMO_SWAP, txc, msg, comparev,
			       compare_desc, compare_count, resultv,
			       result_desc, result_count, flags, false, 0, NULL,
			       NULL);
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
