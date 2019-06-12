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

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

/*
 * cxip_rma_inject_cb() - RMA inject event callback.
 */
static int cxip_rma_inject_cb(struct cxip_req *req, const union c_event *event)
{
	return cxip_cq_req_error(req, 0, FI_EIO, cxi_event_rc(event), NULL, 0);
}

/*
 * cxip_rma_inject_req() - Return request state associated with all RMA inject
 * transactions on the transmit context.
 *
 * The request is freed when the TXC send CQ is closed.
 */
static struct cxip_req *cxip_rma_inject_req(struct cxip_txc *txc)
{
	if (!txc->rma_inject_req) {
		struct cxip_req *req;

		req = cxip_cq_req_alloc(txc->comp.send_cq, 0, txc);
		if (!req)
			return NULL;

		req->cb = cxip_rma_inject_cb;
		req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->flags = FI_RMA | FI_WRITE;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->addr = FI_ADDR_UNSPEC;

		txc->rma_inject_req = req;
	}

	return txc->rma_inject_req;
}

/*
 * cxip_rma_cb() - RMA event callback.
 */
static int cxip_rma_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;
	int success_event = (req->flags & FI_COMPLETION);

	req->flags &= (FI_RMA | FI_READ | FI_WRITE);

	/* IDCs don't have an MD */
	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_LOG_ERROR("Failed to report completion: %d\n",
					       ret);
		}
	} else {
		ret = cxip_cq_req_error(req, 0, FI_EIO, event_rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}

	ofi_atomic_dec32(&req->rma.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
}

/*
 * _cxip_rma_op() - Perform an RMA operation.
 *
 * Common RMA function. Performs RMA reads and writes of all kinds.
 *
 * Generally, operations are supported by Cassini DMA commands. IDC commands
 * are used instead for Write operations smaller than the maximum IDC payload
 * size.
 *
 * If the FI_INJECT flag is specified, an IDC must be used in order to
 * guarantee that source buffer is unused on return. It is an error if
 * FI_INJECT is specified and the payload is longer than the IDC maximum.
 *
 * If the FI_COMPLETION flag is specified, the operation will generate a
 * libfabric completion event. If an event is not requested and an IDC command
 * is used, hardware success events will be suppressed. If a completion is
 * required but an IDC can't be used, the provider tracks the request
 * internally, but will suppress the libfabric event. The provider must track
 * DMA commands in order to clean up the source buffer mapping on completion.
 */
static ssize_t _cxip_rma_op(enum fi_op_type op, struct cxip_txc *txc,
			    const void *buf, size_t len, void *desc,
			    fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			    uint64_t data, uint64_t flags, void *context)
{
	struct cxip_domain *dom;
	int ret;
	struct cxip_md *md = NULL;
	struct cxip_req *req = NULL;
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;
	bool idc;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_rma_initiate_allowed(txc->attr.caps & ~FI_ATOMIC))
		return -FI_ENOPROTOOPT;

	if (!buf)
		return -FI_EINVAL;

	/* Always use IDCs when the payload fits */
	idc = (op == FI_OP_WRITE && len <= C_MAX_IDC_PAYLOAD_RES);

	if (((flags & FI_INJECT) && !idc) || len > CXIP_EP_MAX_MSG_SZ)
		return -FI_EMSGSIZE;

	dom = txc->domain;
	pid_granule = dom->dev_if->if_dev->info.pid_granule;

	if (key >= CXIP_PID_MR_CNT(pid_granule)) {
		CXIP_LOG_DBG("Invalid key: %lu\n", key);
		return -FI_EINVAL;
	}

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, tgt_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	if (!idc) {
		/* Map local buffer */
		ret = cxip_map(dom, buf, len, &md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map buffer: %d\n", ret);
			return ret;
		}
	}

	/* DMA commands must always be tracked. IDCs must be tracked if the
	 * user requested a completion event.
	 */
	if (!idc || (flags & FI_COMPLETION)) {
		req = cxip_cq_req_alloc(txc->comp.send_cq, 0, txc);
		if (!req) {
			CXIP_LOG_DBG("Failed to allocate request\n");
			ret = -FI_ENOMEM;
			goto unmap_op;
		}

		/* Populate request */
		req->context = (uint64_t)context;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | (op == FI_OP_READ ? FI_READ : FI_WRITE) |
				(flags & FI_COMPLETION);
		req->rma.local_md = md;
		req->rma.txc = txc;
	}

	/* Generate the destination fabric address */
	pid_idx = CXIP_MR_TO_IDX(key);
	cxi_build_dfa(caddr.nic, caddr.pid, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	/* Issue command */

	fastlock_acquire(&txc->tx_cmdq->lock);

	if (idc) {
		union c_cmdu cmd = {};

		cmd.c_state.event_send_disable = 1;
		cmd.c_state.restricted = 1;
		cmd.c_state.index_ext = idx_ext;
		cmd.c_state.eq = txc->comp.send_cq->evtq->eqn;

		if (req) {
			cmd.c_state.user_ptr = (uint64_t)req;
		} else {
			void *inject_req = cxip_rma_inject_req(txc);
			if (!inject_req) {
				ret = -FI_ENOMEM;
				goto unlock_op;
			}

			cmd.c_state.user_ptr = (uint64_t)inject_req;
			cmd.c_state.event_success_disable = 1;
		}

		if (memcmp(&txc->tx_cmdq->c_state, &cmd.c_state,
			   sizeof(cmd.c_state))) {
			/* Update TXQ C_STATE */
			txc->tx_cmdq->c_state = cmd.c_state;

			ret = cxi_cq_emit_c_state(txc->tx_cmdq->dev_cmdq,
						  &cmd.c_state);
			if (ret) {
				CXIP_LOG_DBG("Failed to issue C_STATE command: %d\n",
					     ret);

				/* Return error according to Domain Resource
				 * Management
				 */
				ret = -FI_EAGAIN;
				goto unlock_op;
			}

			CXIP_LOG_DBG("Updated C_STATE: %p\n", req);
		}

		memset(&cmd.idc_put, 0, sizeof(cmd.idc_put));
		cmd.idc_put.idc_header.dfa = dfa;
		cmd.idc_put.idc_header.remote_offset = addr;

		ret = cxi_cq_emit_idc_put(txc->tx_cmdq->dev_cmdq, &cmd.idc_put,
					  buf, len);
		if (ret) {
			CXIP_LOG_DBG("Failed to write IDC: %d\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto unlock_op;
		}
	} else {
		struct c_full_dma_cmd cmd = {};

		cmd.command.opcode =
				(op == FI_OP_READ ? C_CMD_GET : C_CMD_PUT);
		cmd.command.cmd_type = C_CMD_TYPE_DMA;
		cmd.index_ext = idx_ext;
		cmd.lac = md->md->lac;
		cmd.event_send_disable = 1;
		cmd.restricted = 1;
		cmd.dfa = dfa;
		cmd.remote_offset = addr;
		cmd.local_addr = CXI_VA_TO_IOVA(md->md, buf);
		cmd.request_len = len;
		cmd.eq = txc->comp.send_cq->evtq->eqn;
		cmd.user_ptr = (uint64_t)req;

		ret = cxi_cq_emit_dma(txc->tx_cmdq->dev_cmdq, &cmd);
		if (ret) {
			CXIP_LOG_DBG("Failed to write DMA command: %d\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto unlock_op;
		}
	}

	cxi_cq_ring(txc->tx_cmdq->dev_cmdq);

	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	fastlock_release(&txc->tx_cmdq->lock);

	CXIP_LOG_DBG("req: %p op: %s buf: %p len: %lu tgt_addr: %ld context %p\n",
		     req, fi_tostr(&op, FI_TYPE_OP_TYPE),
		     buf, len, tgt_addr, context);

	return FI_SUCCESS;

unlock_op:
	fastlock_release(&txc->tx_cmdq->lock);
	if (req)
		cxip_cq_req_free(req);
unmap_op:
	if (!idc)
		cxip_unmap(md);

	return ret;
}

/*
 * Libfabric APIs
 */

static ssize_t cxip_rma_write(struct fid_ep *ep, const void *buf, size_t len,
			      void *desc, fi_addr_t dest_addr, uint64_t addr,
			      uint64_t key, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_rma_op(FI_OP_WRITE, txc, buf, len, desc, dest_addr, addr,
			    key, 0, flags, context);
}

static ssize_t cxip_rma_writev(struct fid_ep *ep, const struct iovec *iov,
			       void **desc, size_t count, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_rma_op(FI_OP_WRITE, txc, iov[0].iov_base, iov[0].iov_len,
			    desc ? desc[0] : NULL, dest_addr, addr, key, 0,
			    flags, context);
}

#define CXIP_WRITEMSG_ALLOWED_FLAGS (FI_INJECT | FI_COMPLETION)

static ssize_t cxip_rma_writemsg(struct fid_ep *ep,
				 const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || !msg->rma_iov ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_WRITEMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return _cxip_rma_op(FI_OP_WRITE, txc,
			    msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			    msg->desc ? msg->desc[0] : NULL, msg->addr,
			    msg->rma_iov[0].addr, msg->rma_iov[0].key,
			    msg->data, flags, msg->context);
}

ssize_t cxip_rma_inject(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return _cxip_rma_op(FI_OP_WRITE, txc, buf, len, NULL, dest_addr, addr,
			    key, 0, FI_INJECT, NULL);
}

static ssize_t cxip_rma_read(struct fid_ep *ep, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, uint64_t addr,
			     uint64_t key, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_rma_op(FI_OP_READ, txc, buf, len, desc, src_addr, addr,
			    key, 0, flags, context);
}

static ssize_t cxip_rma_readv(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;
	uint64_t flags;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	if (txc->selective_completion)
		flags = txc->attr.op_flags & FI_COMPLETION;
	else
		flags = FI_COMPLETION;

	return _cxip_rma_op(FI_OP_READ, txc, iov[0].iov_base, iov[0].iov_len,
			    desc ? desc[0] : NULL, src_addr, addr, key, 0,
			    flags, context);
}

#define CXIP_READMSG_ALLOWED_FLAGS (FI_COMPLETION)

static ssize_t cxip_rma_readmsg(struct fid_ep *ep,
				const struct fi_msg_rma *msg, uint64_t flags)
{
	struct cxip_txc *txc;

	if (!msg || !msg->msg_iov || !msg->rma_iov ||
	    msg->iov_count != 1 || msg->rma_iov_count != 1)
		return -FI_EINVAL;

	if (flags & ~CXIP_READMSG_ALLOWED_FLAGS)
		return -FI_EBADFLAGS;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return _cxip_rma_op(FI_OP_READ, txc,
			    msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
			    msg->desc ? msg->desc[0] : NULL, msg->addr,
			    msg->rma_iov[0].addr, msg->rma_iov[0].key,
			    msg->data, flags, msg->context);
}

struct fi_ops_rma cxip_ep_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = cxip_rma_read,
	.readv = cxip_rma_readv,
	.readmsg = cxip_rma_readmsg,
	.write = cxip_rma_write,
	.writev = cxip_rma_writev,
	.writemsg = cxip_rma_writemsg,
	.inject = cxip_rma_inject,
	.injectdata = fi_no_rma_injectdata,
	.writedata = fi_no_rma_writedata,
};
