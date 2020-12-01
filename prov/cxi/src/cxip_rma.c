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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_DATA, __VA_ARGS__)

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

		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
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

	if (req->rma.local_md)
		cxip_unmap(req->rma.local_md);

	event_rc = cxi_init_event_rc(event);
	if (event_rc == C_RC_OK) {
		if (success_event) {
			ret = cxip_cq_req_complete(req);
			if (ret != FI_SUCCESS)
				CXIP_WARN("Failed to report completion: %d\n",
					  ret);
		}
	} else {
		ret = cxip_cq_req_error(req, 0, FI_EIO, event_rc,
					NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_WARN("Failed to report error: %d\n", ret);
	}

	ofi_atomic_dec32(&req->rma.txc->otx_reqs);
	cxip_cq_req_free(req);

	return FI_SUCCESS;
}

/*
 * cxip_rma_common() - Perform an RMA operation.
 *
 * Common RMA function. Performs RMA reads and writes of all kinds.
 *
 * Generally, operations are supported by Cassini DMA commands. IDC commands
 * are used instead for Write operations smaller than the maximum IDC payload
 * size.
 *
 * If the FI_COMPLETION flag is specified, the operation will generate a
 * libfabric completion event. If an event is not requested and an IDC command
 * is used, hardware success events will be suppressed. If a completion is
 * required but an IDC can't be used, the provider tracks the request
 * internally, but will suppress the libfabric event. The provider must track
 * DMA commands in order to clean up the source buffer mapping on completion.
 */
ssize_t cxip_rma_common(enum fi_op_type op, struct cxip_txc *txc,
			const void *buf, size_t len, void *desc,
			fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			uint64_t data, uint64_t flags, void *context,
			bool triggered, uint64_t trig_thresh,
			struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr)
{
	int ret;
	struct cxip_req *req = NULL;
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint8_t idx_ext;
	uint32_t pid_idx;
	bool idc;
	bool unr = false; /* use unrestricted command? */
	struct cxip_cmdq *cmdq =
		triggered ? txc->domain->trig_cmdq : txc->tx_cmdq;
	bool write = op == FI_OP_WRITE;
	enum cxi_traffic_class_type tc_type;
	void *hmem_buf = NULL;
	const void *idc_buf = buf;
	enum fi_hmem_iface iface;
	struct iovec hmem_iov;

	if (!txc->enabled)
		return -FI_EOPBADSTATE;

	if (!ofi_rma_initiate_allowed(txc->attr.caps & ~FI_ATOMIC))
		return -FI_ENOPROTOOPT;

	if (len && !buf)
		return -FI_EINVAL;

	if (((flags & FI_INJECT) && len > CXIP_INJECT_SIZE) ||
	    len > CXIP_EP_MAX_MSG_SZ)
		return -FI_EMSGSIZE;

	/* Use IDCs if the payload fits and targeting an optimized MR. */
	idc = (op == FI_OP_WRITE) && (len <= C_MAX_IDC_PAYLOAD_RES) &&
	       cxip_mr_key_opt(key) && !triggered;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->ep_obj->av, tgt_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* DMA commands must always be tracked. IDCs must be tracked if the
	 * user requested a completion event.
	 */
	if (!idc || (flags & FI_COMPLETION)) {
		req = cxip_cq_req_alloc(txc->send_cq, 0, txc);
		if (!req) {
			CXIP_DBG("Failed to allocate request\n");
			return -FI_ENOMEM;
		}

		/* Populate request */
		if (flags & FI_COMPLETION)
			req->context = (uint64_t)context;
		else
			req->context = (uint64_t)txc->fid.ctx.fid.context;
		req->data_len = 0;
		req->buf = 0;
		req->data = 0;
		req->tag = 0;
		req->cb = cxip_rma_cb;
		req->flags = FI_RMA | (op == FI_OP_READ ? FI_READ : FI_WRITE) |
				(flags & FI_COMPLETION);
		req->rma.txc = txc;
		req->type = CXIP_REQ_RMA;
		req->trig_cntr = trig_cntr;
	}

	if (len && !idc) {
		/* Map user buffer for DMA command. */
		ret = cxip_map(txc->domain, buf, len,
				&req->rma.local_md);
		if (ret) {
			CXIP_DBG("Failed to map buffer: %d\n", ret);
			goto req_free;
		}
	}

	/* Generate the destination fabric address */
	pid_idx = cxip_mr_key_to_ptl_idx(key, write);
	cxi_build_dfa(caddr.nic, caddr.pid, txc->pid_bits, pid_idx, &dfa,
		      &idx_ext);

	/* Unordered Puts are optimally supported with restricted commands.
	 * When Put ordering or remote events are required, or when targeting a
	 * standard MR, use unrestricted commands. Ordered Gets are never
	 * supported.
	 */
	unr = !cxip_mr_key_opt(key) || txc->ep_obj->caps & FI_RMA_EVENT;
	if (!unr && write)
		unr = txc->attr.msg_order & (FI_ORDER_WAW | FI_ORDER_RMA_WAW);

	if (!unr && (flags & FI_CXI_HRP))
		tc_type = CXI_TC_TYPE_HRP;
	else if (!unr)
		tc_type = CXI_TC_TYPE_RESTRICTED;
	else
		tc_type = CXI_TC_TYPE_DEFAULT;

	/* HMEM bounce buffer is required for IDCs and to non-system memory. */
	if (txc->hmem && idc) {
		iface = ofi_get_hmem_iface(buf);

		if (iface != FI_HMEM_SYSTEM) {
			hmem_iov.iov_base = (void *)buf;
			hmem_iov.iov_len = len;
			hmem_buf = cxip_cq_ibuf_alloc(txc->send_cq);
			if (!hmem_buf)
				goto md_unmap;

			ret = ofi_copy_from_hmem_iov(hmem_buf, len, iface, 0,
						     &hmem_iov, 1, 0);
			assert(ret == len);

			idc_buf = hmem_buf;
		}
	}

	/* Issue command */
	fastlock_acquire(&cmdq->lock);

	ret = cxip_txq_cp_set(cmdq, txc->ep_obj->auth_key.vni,
			      cxip_ofi_to_cxi_tc(txc->tclass), tc_type);
	if (ret != FI_SUCCESS)
		goto unlock_op;

	if (flags & FI_FENCE) {
		ret = cxi_cq_emit_cq_cmd(cmdq->dev_cmdq, C_CMD_CQ_FENCE);
		if (ret) {
			CXIP_DBG("Failed to issue CQ_FENCE command: %d\n",
				 ret);
			ret = -FI_EAGAIN;
			goto unlock_op;
		}
	}

	if (idc) {
		union c_cmdu cmd = {};

		cmd.c_state.event_send_disable = 1;
		cmd.c_state.index_ext = idx_ext;
		cmd.c_state.eq = txc->send_cq->evtq->eqn;

		if (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE))
			cmd.c_state.flush = 1;

		if (!unr)
			cmd.c_state.restricted = 1;

		if (txc->write_cntr) {
			cmd.c_state.event_ct_ack = 1;
			cmd.c_state.ct = txc->write_cntr->ct->ctn;
		}

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

		if (memcmp(&cmdq->c_state, &cmd.c_state, sizeof(cmd.c_state))) {
			ret = cxi_cq_emit_c_state(cmdq->dev_cmdq, &cmd.c_state);
			if (ret) {
				CXIP_DBG("Failed to issue C_STATE command: %d\n",
					 ret);

				/* Return error according to Domain Resource
				 * Management
				 */
				ret = -FI_EAGAIN;
				goto unlock_op;
			}

			/* Update TXQ C_STATE */
			cmdq->c_state = cmd.c_state;

			CXIP_DBG("Updated C_STATE: %p\n", req);
		}

		memset(&cmd.idc_put, 0, sizeof(cmd.idc_put));
		cmd.idc_put.idc_header.dfa = dfa;
		cmd.idc_put.idc_header.remote_offset = addr;

		ret = cxi_cq_emit_idc_put(cmdq->dev_cmdq, &cmd.idc_put, idc_buf,
					  len);
		if (ret) {
			CXIP_DBG("Failed to write IDC: %d\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto unlock_op;
		}
	} else {
		struct c_full_dma_cmd cmd = {};
		struct cxip_cntr *cntr;

		cmd.command.opcode =
				(op == FI_OP_READ ? C_CMD_GET : C_CMD_PUT);
		cmd.command.cmd_type = C_CMD_TYPE_DMA;
		cmd.index_ext = idx_ext;

		if (len) {
			cmd.lac = req->rma.local_md->md->lac;
			cmd.local_addr =
				CXI_VA_TO_IOVA(req->rma.local_md->md, buf);
		}

		cmd.event_send_disable = 1;
		cmd.dfa = dfa;
		cmd.remote_offset = addr;
		cmd.request_len = len;
		cmd.eq = txc->send_cq->evtq->eqn;
		cmd.user_ptr = (uint64_t)req;
		cmd.match_bits = key;

		if ((op == FI_OP_WRITE) &&
		    (flags & (FI_DELIVERY_COMPLETE | FI_MATCH_COMPLETE)))
			cmd.flush = 1;

		if (!unr)
			cmd.restricted = 1;

		if (op == FI_OP_WRITE) {
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

		if (triggered) {
			const struct c_ct_cmd ct_cmd = {
				.trig_ct = trig_cntr->ct->ctn,
				.threshold = trig_thresh,
			};

			ret = cxi_cq_emit_trig_full_dma(cmdq->dev_cmdq, &ct_cmd,
							&cmd);
		} else {
			ret = cxi_cq_emit_dma(cmdq->dev_cmdq, &cmd);
		}

		if (ret) {
			CXIP_DBG("Failed to write DMA command: %d\n", ret);

			/* Return error according to Domain Resource Management
			 */
			ret = -FI_EAGAIN;
			goto unlock_op;
		}
	}

	cxip_txq_ring(cmdq, flags & FI_MORE, triggered,
		      ofi_atomic_get32(&txc->otx_reqs));

	if (req)
		ofi_atomic_inc32(&txc->otx_reqs);

	fastlock_release(&cmdq->lock);

	if (hmem_buf)
		cxip_cq_ibuf_free(txc->send_cq, hmem_buf);

	CXIP_DBG("%sreq: %p op: %s buf: %p len: %lu tgt_addr: %ld context %p\n",
		 idc ? "IDC " : "", req, fi_tostr(&op, FI_TYPE_OP_TYPE),
		 buf, len, tgt_addr, context);

	return FI_SUCCESS;

unlock_op:
	fastlock_release(&txc->tx_cmdq->lock);
	if (hmem_buf)
		cxip_cq_ibuf_free(txc->send_cq, hmem_buf);
md_unmap:
	if (req && req->rma.local_md)
		cxip_unmap(req->rma.local_md);
req_free:
	if (req)
		cxip_cq_req_free(req);

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

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, buf, len, desc, dest_addr,
			       addr, key, 0, txc->attr.op_flags, context, false,
			       0, NULL, NULL);
}

static ssize_t cxip_rma_writev(struct fid_ep *ep, const struct iovec *iov,
			       void **desc, size_t count, fi_addr_t dest_addr,
			       uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, iov[0].iov_base,
			       iov[0].iov_len, desc ? desc[0] : NULL, dest_addr,
			       addr, key, 0, txc->attr.op_flags, context, false,
			       0, NULL, NULL);
}

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

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_WRITE, txc, msg->msg_iov[0].iov_base,
			       msg->msg_iov[0].iov_len,
			       msg->desc ? msg->desc[0] : NULL, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, msg->context, false, 0, NULL,
			       NULL);
}

ssize_t cxip_rma_inject(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_WRITE, txc, buf, len, NULL, dest_addr,
			       addr, key, 0, FI_INJECT, NULL, false, 0, NULL,
			       NULL);
}

static ssize_t cxip_rma_read(struct fid_ep *ep, void *buf, size_t len,
			     void *desc, fi_addr_t src_addr, uint64_t addr,
			     uint64_t key, void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_READ, txc, buf, len, desc, src_addr, addr,
			       key, 0, txc->attr.op_flags, context, false, 0,
			       NULL, NULL);
}

static ssize_t cxip_rma_readv(struct fid_ep *ep, const struct iovec *iov,
			      void **desc, size_t count, fi_addr_t src_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxip_txc *txc;

	if (cxip_fid_to_txc(ep, &txc) != FI_SUCCESS)
		return -FI_EINVAL;

	if (!iov || count != 1)
		return -FI_EINVAL;

	return cxip_rma_common(FI_OP_READ, txc, iov[0].iov_base, iov[0].iov_len,
			       desc ? desc[0] : NULL, src_addr, addr, key, 0,
			       txc->attr.op_flags, context, false, 0, NULL,
			       NULL);
}

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

	/* If selective completion is not requested, always generate
	 * completions.
	 */
	if (!txc->selective_completion)
		flags |= FI_COMPLETION;

	return cxip_rma_common(FI_OP_READ, txc, msg->msg_iov[0].iov_base,
			       msg->msg_iov[0].iov_len,
			       msg->desc ? msg->desc[0] : NULL, msg->addr,
			       msg->rma_iov[0].addr, msg->rma_iov[0].key,
			       msg->data, flags, msg->context, false, 0, NULL,
			       NULL);
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
