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

#include "cxi_prov.h"

#define CXIX_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIX_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

static ssize_t cxix_rma_write(struct fid_ep *ep, const void *buf,
			      size_t len, void *desc, fi_addr_t dest_addr,
			      uint64_t addr, uint64_t key, void *context)
{
	struct cxi_ep *cxi_ep;
	struct cxi_tx_ctx *txc;
	struct cxi_domain *dom;
	int ret;
	struct cxi_iova write_md;
	struct cxi_req *req;
	union c_cmdu cmd = {};
	struct cxi_addr caddr;
	uint32_t dfa;
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
		cxi_ep = container_of(ep, struct cxi_ep, ep);
		txc = cxi_ep->attr->tx_ctx;
		break;

	case FI_CLASS_TX_CTX:
		txc = container_of(ep, struct cxi_tx_ctx, fid.ctx);
		break;

	default:
		CXIX_LOG_ERROR("Invalid EP type\n");
		return -FI_EINVAL;
	}

	dom = txc->domain;

	/* Look up target CXI address */
	ret = _cxi_av_lookup(txc->av, dest_addr, &caddr);
	if (ret != FI_SUCCESS) {
		CXIX_LOG_DBG("Failed to look up FI addr: %d\n", ret);
		return ret;
	}

	/* Map local buffer */
	ret = cxil_map(dom->dev_if->if_lni, (void *)buf, len,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_READ,
		       &write_md);
	if (ret) {
		CXIX_LOG_DBG("Failed to map write buffer: %d\n", ret);
		return ret;
	}

	/* Populate request */
	req = cxix_cq_req_alloc(txc->comp.send_cq, 0);
	if (!req) {
		CXIX_LOG_DBG("Failed to allocate request\n");
		ret = -FI_ENOMEM;
		goto unmap;
	}

	req->context = (uint64_t)context;
	req->flags = FI_RMA | FI_WRITE;
	req->data_len = 0;
	req->buf = 0;
	req->data = 0;
	req->tag = 0;

	req->local_md = write_md;

	/* Build Put command descriptor */
	pid_granule = dom->pid_granule;
	pid_idx = CXIX_ADDR_MR_IDX(pid_granule, key);
	cxi_build_dfa(caddr.nic, caddr.domain, pid_granule, pid_idx, &dfa,
		      &idx_ext);

	cmd.full_dma.command.cmd_type = C_CMD_TYPE_DMA;
	cmd.full_dma.command.opcode = C_CMD_PUT;

	cmd.full_dma.index_ext = idx_ext;
	cmd.full_dma.lac = write_md.lac;
	cmd.full_dma.event_send_disable = 1;
	cmd.full_dma.dfa = dfa;
	cmd.full_dma.remote_offset = addr;
	cmd.full_dma.local_addr = write_md.iova + ((uint64_t)buf - write_md.va);
	cmd.full_dma.request_len = len;
	cmd.full_dma.eq = txc->comp.send_cq->evtq->eqn;
	cmd.full_dma.user_ptr = (uint64_t)req;

	fastlock_acquire(&txc->lock);

	/* Issue Put command */
	ret = cxi_cq_emit_dma(txc->tx_cmdq, &cmd.full_dma);
	if (ret) {
		CXIX_LOG_DBG("Failed to write DMA command: %d\n", ret);

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
	cxix_cq_req_free(req);
unmap:
	cxil_unmap(dom->dev_if->if_lni, &write_md);

	return ret;
}

struct fi_ops_rma cxix_ep_rma = {
	.size  = sizeof(struct fi_ops_rma),
	.read = fi_no_rma_read,
	.readv = fi_no_rma_readv,
	.readmsg = fi_no_rma_readmsg,
	.write = cxix_rma_write,
	.writev = fi_no_rma_writev,
	.writemsg = fi_no_rma_writemsg,
	.inject = fi_no_rma_inject,
	.injectdata = fi_no_rma_injectdata,
	.writedata = fi_no_rma_writedata,
};

