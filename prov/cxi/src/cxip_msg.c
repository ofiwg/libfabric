/*
 * SPDX-License-Identifier: GPL-2.0
 *
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

static void cxip_trecv_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int truncated;
	int err;

	CXIP_LOG_DBG("got event: %d\n", event->hdr.event_type);

	/* Netsim is currently giving events in the order: LINK-UNLINK-PUT.
	 * Assume this order is guaranteed for now.
	 */

	if (event->hdr.event_type == C_EVENT_LINK)
		return;

	if (event->hdr.event_type == C_EVENT_UNLINK)
		return;

	/* event_type == C_EVENT_PUT */
	req->rc = event->tgt_long.return_code;
	req->rlength = event->tgt_long.rlength;
	req->mlength = event->tgt_long.mlength;

	ret = cxil_unmap(req->cq->domain->dev_if->if_lni, &req->local_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

	truncated = req->rlength - req->mlength;
	if (req->rc == C_RC_OK && !truncated) {
		req->data_len = event->tgt_long.mlength;

		ret = req->cq->report_completion(req->cq, FI_ADDR_UNSPEC, req);
		if (ret != req->cq->cq_entry_size)
			CXIP_LOG_ERROR("Failed to report completion: %d\n",
				       ret);
	} else {
		err = truncated ? FI_EMSGSIZE : FI_EIO;

		ret = cxip_cq_report_error(req->cq, req, truncated, err,
					   req->rc, NULL, 0);
		if (ret != FI_SUCCESS)
			CXIP_LOG_ERROR("Failed to report error: %d\n", ret);
	}

	cxip_cq_req_free(req);
}

static ssize_t cxip_trecv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			  fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
			  void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_rx_ctx *rxc;
	struct cxip_domain *dom;
	int ret;
	struct cxi_iova recv_md;
	struct cxip_req *req;
	union c_cmdu cmd = {};

	if (!ep || !buf)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a RX
	 * context), or a RX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		rxc = cxi_ep->attr->rx_ctx;
		break;

	case FI_CLASS_RX_CTX:
		rxc = container_of(ep, struct cxip_rx_ctx, ctx);
		break;

	default:
		CXIP_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	dom = rxc->domain;

	/* Map local buffer */
	ret = cxil_map(dom->dev_if->if_lni, (void *)buf, len,
		       CXI_MAP_PIN | CXI_MAP_NTA |
		       CXI_MAP_WRITE | CXI_MAP_NOCACHE, &recv_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map recv buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(rxc->comp.recv_cq, 1);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto unmap;
	}

	/* req->data_len must be set later.  req->buf and req->data may be
	 * overwritten later.
	 */
	req->context = (uint64_t)context;
	req->flags = FI_TAGGED | FI_RECV;
	req->buf = 0;
	req->data = 0;
	req->tag = tag;

	req->local_md = recv_md;
	req->cb = cxip_trecv_cb;

	/* Build Append command descriptor */
	cmd.command.opcode     = C_CMD_TGT_APPEND;
	cmd.target.ptl_list    = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = rxc->pte_hw_id;
	cmd.target.op_put      = 1;
	cmd.target.buffer_id   = req->req_id;
	cmd.target.lac         = recv_md.lac;
	cmd.target.start       = CXI_VA_TO_IOVA(&recv_md, buf);
	cmd.target.length      = len;
	cmd.target.use_once    = 1;
	cmd.target.match_bits  = tag;
	cmd.target.ignore_bits = ignore;

	fastlock_acquire(&rxc->lock);

	/* Issue Append command */
	ret = cxi_cq_emit_target(rxc->rx_cmdq, &cmd);
	if (ret) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);

		/* Return error according to Domain Resource Management */
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cxi_cq_ring(rxc->rx_cmdq);

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&rxc->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&rxc->lock);
	cxip_cq_req_free(req);
unmap:
	cxil_unmap(dom->dev_if->if_lni, &recv_md);

	return ret;
}

static void cxip_tsend_cb(struct cxip_req *req, const union c_event *event)
{
	int ret;
	int event_rc;

	ret = cxil_unmap(req->cq->domain->dev_if->if_lni, &req->local_md);
	if (ret != FI_SUCCESS)
		CXIP_LOG_ERROR("Failed to free MD: %d\n", ret);

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

static ssize_t cxip_tsend(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, fi_addr_t dest_addr, uint64_t tag,
			  void *context)
{
	struct cxip_ep *cxi_ep;
	struct cxip_tx_ctx *txc;
	struct cxip_domain *dom;
	int ret;
	struct cxi_iova send_md;
	struct cxip_req *req;
	union c_cmdu cmd = {};
	struct cxip_addr caddr;
	union c_fab_addr dfa;
	uint32_t idx_ext;
	uint32_t pid_granule;
	uint32_t pid_idx;

	if (!ep || !buf)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a TX
	 * context), or a TX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		txc = cxi_ep->attr->tx_ctx;
		break;

	case FI_CLASS_TX_CTX:
		txc = container_of(ep, struct cxip_tx_ctx, fid.ctx);
		break;

	default:
		CXIP_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	dom = txc->domain;

	/* Look up target CXI address */
	ret = _cxip_av_lookup(txc->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer */
	ret = cxil_map(dom->dev_if->if_lni, (void *)buf, len,
		       CXI_MAP_PIN | CXI_MAP_NTA |
		       CXI_MAP_READ | CXI_MAP_NOCACHE, &send_md);
	if (ret) {
		CXIP_LOG_DBG("Failed to map send buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxip_cq_req_alloc(txc->comp.send_cq, 0);
	if (!req) {
		CXIP_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto unmap;
	}

	req->context = (uint64_t)context;
	req->flags = FI_TAGGED | FI_SEND;
	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;

	req->local_md = send_md;
	req->cb = cxip_tsend_cb;

	/* Build Put command descriptor */
	pid_granule = dom->dev_if->if_pid_granule;
	pid_idx = CXIP_ADDR_RX_IDX(pid_granule, 0);
	cxi_build_dfa(caddr.nic, caddr.port, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;
	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = send_md.lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.restricted = 0;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = 0;
	cmd.full_dma.local_addr = CXI_VA_TO_IOVA(&send_md, buf);
	cmd.full_dma.request_len = len;
	cmd.full_dma.eq = txc->comp.send_cq->evtq->eqn;
	cmd.full_dma.user_ptr = (uint64_t)req;
	cmd.full_dma.match_bits = tag;

	fastlock_acquire(&txc->lock);

	/* Issue Put command */
	ret = cxi_cq_emit_dma(txc->tx_cmdq, &cmd.full_dma);
	if (ret) {
		CXIP_LOG_DBG("Failed to write DMA command: %d\n", ret);

		/* Return error according to Domain Resource Management */
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cxi_cq_ring(txc->tx_cmdq);

	/* TODO take reference on EP or context for the outstanding request */
	fastlock_release(&txc->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&txc->lock);
	cxip_cq_req_free(req);
unmap:
	cxil_unmap(dom->dev_if->if_lni, &send_md);

	return ret;
}

struct fi_ops_tagged cxip_ep_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = cxip_trecv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = cxip_tsend,
	.sendv = fi_no_tagged_sendv,
	.sendmsg = fi_no_tagged_sendmsg,
	.inject = fi_no_tagged_inject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

struct fi_ops_msg cxip_ep_msg_ops = { 0 };

