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

#define CXIP_AMO_MAX_PACKED_IOV (1)

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

enum cxip_amo_req_type {
	CXIP_RQ_AMO,
	CXIP_RQ_AMO_FETCH,
	CXIP_RQ_AMO_SWAP,
	CXIP_RQ_AMO_LAST
};

static int _cxip_amo_type_code[FI_DATATYPE_LAST] = {
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
	[FI_LONG_DOUBLE]	  = -1,
	[FI_LONG_DOUBLE_COMPLEX]  = -1,
};
//C_AMO_TYPE_UINT128_T

static int _cxip_amo_op_code[FI_ATOMIC_OP_LAST] = {
	[FI_MIN]	  = C_AMO_OP_MIN,
	[FI_MAX]	  = C_AMO_OP_MAX,
	[FI_SUM]	  = C_AMO_OP_SUM,
	[FI_PROD]	  = -1,
	[FI_LOR]	  = C_AMO_OP_LOR,
	[FI_LAND]	  = C_AMO_OP_LAND,
	[FI_BOR]	  = C_AMO_OP_BOR,
	[FI_BAND]	  = C_AMO_OP_BAND,
	[FI_LXOR]	  = C_AMO_OP_LXOR,
	[FI_BXOR]	  = C_AMO_OP_BXOR,
	[FI_ATOMIC_READ]  = -1,
	[FI_ATOMIC_WRITE] = -1,
	[FI_CSWAP]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_NE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_LE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_LT]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_GE]	  = C_AMO_OP_CSWAP,
	[FI_CSWAP_GT]	  = C_AMO_OP_CSWAP,
	[FI_MSWAP]	  = C_AMO_OP_AXOR,
};
//C_AMO_OP_SWAP

static int _cxip_amo_swpcode[FI_ATOMIC_OP_LAST] = {
	[FI_CSWAP]	  = C_AMO_OP_CSWAP_EQ,
	[FI_CSWAP_NE]	  = C_AMO_OP_CSWAP_NE,
	[FI_CSWAP_LE]	  = C_AMO_OP_CSWAP_LE,
	[FI_CSWAP_LT]	  = C_AMO_OP_CSWAP_LT,
	[FI_CSWAP_GE]	  = C_AMO_OP_CSWAP_GE,
	[FI_CSWAP_GT]	  = C_AMO_OP_CSWAP_GT,
};

static int _cxip_amo_valid[CXIP_RQ_AMO_LAST]
			  [FI_ATOMIC_OP_LAST]
			  [FI_DATATYPE_LAST] = {
	/*
	 * Basic AMO types:
	 * FI_MIN, FI_MAX, FI_SUM, FI_PROD, FI_LOR, FI_LAND,  FI_BOR,  FI_BAND,
	 *       FI_LXOR, FI_BXOR, and FI_ATOMIC_WRITE.
	 */
	[CXIP_RQ_AMO] = {
	[FI_MIN]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_MAX]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_SUM]	  = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
	[FI_BOR]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_BAND]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_BXOR]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_ATOMIC_WRITE] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	},
	[CXIP_RQ_AMO_FETCH] = {
	[FI_MIN]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_MAX]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_SUM]	  = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
	[FI_BOR]	  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_BAND]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_BXOR]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_ATOMIC_READ]  = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_ATOMIC_WRITE] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	},
	[CXIP_RQ_AMO_SWAP] = {
	[FI_CSWAP]        = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_MSWAP]        = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_CSWAP_NE]     = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
	[FI_CSWAP_LE]     = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_CSWAP_LT]     = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_CSWAP_GE]     = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	[FI_CSWAP_GT]     = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	},
};

// TODO: should this be static?
int cxip_atomic_opcode(enum cxip_amo_req_type req_type,
		       enum fi_datatype dt, enum fi_op op,
		       enum c_atomic_op *cop, enum c_atomic_type *cdt,
		       enum c_cswap_op *copswp, int *cdtlen)
{
	int opcode;
	int dtcode;

	if (dt < 0 || dt >= FI_DATATYPE_LAST ||
	    op < 0 || op >= FI_ATOMIC_OP_LAST)
		return -FI_EINVAL;

	if (!_cxip_amo_valid[req_type][op][dt])
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
	ret = cxip_atomic_opcode(req_type, datatype, op,
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

static void cxip_amo_cb(struct cxip_req *req, const union c_event *event)
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

static void cxip_amo_cbf(struct cxip_req *req, const union c_event *event)
{
	int ret;

	ret = cxil_unmap(req->cq->domain->dev_if->if_lni,
			 &req->local_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

	cxip_amo_cb(req, event);
}


static int __cxip_idc_amo(enum cxip_amo_req_type req_type, struct fid_ep *ep,
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
	void *compare = NULL;
	void *result = NULL;
	void *oper1 = NULL;
	bool fetch = false;
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
		if (compare_count != CXIP_AMO_MAX_IOV || !comparev ||
		    comparev[0].count != CXIP_AMO_MAX_PACKED_IOV ||
		    !comparev[0].addr)
			return -FI_EINVAL;
		compare = comparev[0].addr;
		/* FALLTHRU */
	case CXIP_RQ_AMO_FETCH:
		/* Must have a valid result address */
		if (result_count != CXIP_AMO_MAX_IOV || !resultv ||
		    resultv[0].count != CXIP_AMO_MAX_PACKED_IOV ||
		    !resultv[0].addr)
			return -FI_EINVAL;
		result = resultv[0].addr;
		fetch = true;
		/* FALLTHRU */
	case CXIP_RQ_AMO:
		if (msg->iov_count != CXIP_AMO_MAX_IOV || !msg->msg_iov ||
		    msg->msg_iov[0].count != CXIP_AMO_MAX_PACKED_IOV ||
		    !msg->msg_iov[0].addr)
			return -FI_EINVAL;
		/* The supplied RMA address is actually an offset into a
		 * registered MR. A value of 0 is valid.
		 */
		if (msg->rma_iov_count != CXIP_AMO_MAX_IOV || !msg->rma_iov ||
		    msg->msg_iov[0].count != CXIP_AMO_MAX_PACKED_IOV)
			return -FI_EINVAL;
		oper1 = msg->msg_iov[0].addr;
		off = msg->rma_iov[0].addr;
		key = msg->rma_iov[0].key;
		break;
	default:
		return -FI_EINVAL;
	}

	/* Convert FI to CXI codes */
	ret = cxip_atomic_opcode(req_type, msg->datatype, msg->op,
				 &opcode, &dtcode, &swpcode, &len);
	if (ret < 0)
		return ret;

	/* The input FID could be a standard endpoint (containing a TX
	 * context), or a TX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP: {
		struct cxip_ep *cxi_ep;

		cxi_ep = container_of(ep, struct cxip_ep, ep);
		txc = cxi_ep->attr->tx_ctx;
		break;
	}
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
			       CXI_MAP_READ | CXI_MAP_WRITE |
			       CXI_MAP_NOCACHE,
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

	req->context = (uint64_t)msg->context;
	req->flags = FI_ATOMIC | FI_WRITE | FI_READ;
	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;
	req->local_md = result_md;
	req->cb = (result) ? cxip_amo_cbf : cxip_amo_cb;

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
	ret = cxi_cq_emit_idc_amo(txc->tx_cmdq, &cmd.idc_amo, fetch);
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
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO, ep, &msg,
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
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO, ep, &msg,
			      NULL, NULL, 0,
			      NULL, NULL, 0,
			      flags);
}

static ssize_t cxip_ep_atomic_writemsg(struct fid_ep *ep,
				       const struct fi_msg_atomic *msg,
				       uint64_t flags)
{
	return __cxip_idc_amo(CXIP_RQ_AMO, ep, msg,
			      NULL, NULL, 0,
			      NULL, NULL, 0,
			      flags);
}

static ssize_t cxip_ep_atomic_inject(struct fid_ep *ep, const void *buf,
				     size_t count, fi_addr_t dest_addr,
				     uint64_t addr, uint64_t key,
				     enum fi_datatype datatype, enum fi_op op)
{
	printf("ATOMIC WRITEINJECT\n"); fflush(stdout);
	return -FI_EOPNOTSUPP;
}

static ssize_t cxip_ep_atomic_readwrite(struct fid_ep *ep, const void *buf,
					size_t count, void *desc, void *result,
					void *result_desc, fi_addr_t dest_addr,
					uint64_t addr, uint64_t key,
					enum fi_datatype datatype,
					enum fi_op op, void *context)
{
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, &msg,
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
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, &msg,
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
	return __cxip_idc_amo(CXIP_RQ_AMO_FETCH, ep, msg,
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
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, &msg,
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
	uint64_t flags = 0;

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

	return __cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, &msg,
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
	return __cxip_idc_amo(CXIP_RQ_AMO_SWAP, ep, msg,
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

