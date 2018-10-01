/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <ofi_util.h>

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_MR, __VA_ARGS__)

#define MR_LINK_EVENT_ID 0x1e21

/* Caller must hold mr->lock */
int cxip_mr_enable(struct cxip_mr *mr)
{
	int ret;
	union c_cmdu cmd = {};
	const union c_event *event;
	uint32_t buffer_id = MR_LINK_EVENT_ID;
	struct cxi_pt_alloc_opts opts = {};

	if (mr->enabled)
		return FI_SUCCESS;

	/* Allocate a PTE */
	ret = cxil_alloc_pte(mr->domain->dev_if->if_lni,
			     mr->domain->dev_if->mr_evtq, &opts, &mr->pte);
	if (ret) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		return -FI_ENOSPC;
	}

	/* Reserve the logical endpoint (LEP) where the MR will be mapped */
	mr->pid_idx = CXIP_ADDR_MR_IDX(mr->domain->dev_if->if_pid_granule,
				       mr->key);

	ret = cxip_if_domain_lep_alloc(mr->ep->ep_obj->if_dom, mr->pid_idx);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to reserve LEP (%d): %d\n", mr->pid_idx,
			     ret);
		goto free_pte;
	}

	/* Map the PTE to the LEP */
	ret = cxil_map_pte(mr->pte, mr->ep->ep_obj->if_dom->cxil_if_dom,
			   mr->pid_idx, 0, &mr->pte_map);
	if (ret) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_EADDRINUSE;
		goto free_lep;
	}

	/* Map the window buffer into IO address space */
	ret = cxil_map(mr->domain->dev_if->if_lni, mr->buf, mr->len,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_READ | CXI_MAP_WRITE,
		       &mr->md);
	if (ret) {
		CXIP_LOG_DBG("Failed to IO map MR buffer: %d\n", ret);
		ret = -FI_EFAULT;
		goto unmap_pte;
	}

	/* Use the device CMDQ and EQ to enable the PTE and append an LE.
	 * This serializes creation of all MRs in the process.  We need to
	 * revisit.
	 *
	 * TODO: revisit this, deserialize MR enabling
	 */
	fastlock_acquire(&mr->domain->dev_if->lock);

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = mr->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unmap_buf;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != mr->pte->ptn) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto unmap_buf;
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

	/* Link Persistent LE */
	memset(&cmd, 0, sizeof(cmd));
	cmd.command.opcode     = C_CMD_TGT_APPEND;
	cmd.target.ptl_list    = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = mr->pte->ptn;
	cmd.target.no_truncate = 0;
	cmd.target.unexpected_hdr_disable = 0;
	cmd.target.buffer_id   = buffer_id;
	cmd.target.lac         = mr->md.lac;
	cmd.target.start       = mr->md.iova + ((uint64_t)mr->buf - mr->md.va);
	cmd.target.length      = mr->len;
	cmd.target.match_bits  = 0;

	if (mr->attr.access & FI_REMOTE_WRITE)
		cmd.target.op_put = 1;
	if (mr->attr.access & FI_REMOTE_READ)
		cmd.target.op_get = 1;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unmap_buf;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_LINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != buffer_id) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Link EQE\n");
		ret = -FI_EIO;
		goto unmap_buf;
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

	fastlock_release(&mr->domain->dev_if->lock);

	mr->enabled = 1;

	return FI_SUCCESS;

unmap_buf:
	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);
	fastlock_release(&mr->domain->dev_if->lock);

	cxil_unmap(mr->domain->dev_if->if_lni, &mr->md);
unmap_pte:
	cxil_unmap_pte(mr->pte_map);
free_lep:
	cxip_if_domain_lep_free(mr->ep->ep_obj->if_dom, mr->pid_idx);
free_pte:
	cxil_destroy_pte(mr->pte);

	return ret;
}

/* Caller must hold mr->lock */
int cxip_mr_disable(struct cxip_mr *mr)
{
	int ret;
	union c_cmdu cmd = {};
	const union c_event *event;
	uint32_t buffer_id = MR_LINK_EVENT_ID;

	if (!mr->enabled)
		return FI_SUCCESS;

	/* Use the device CMDQ and EQ to unlink the LE.  This serializes
	 * enable/disable of all MRs in the process.  We need to revisit.
	 */
	fastlock_acquire(&mr->domain->dev_if->lock);

	/* Unlink persistent LE */
	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = mr->pte->ptn;
	cmd.target.buffer_id = buffer_id;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a provider bug, we have exclusive access to this
		 * CMDQ.
		 */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unlock;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for unlink EQ event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_UNLINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != buffer_id) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Unlink EQE\n");
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

unlock:
	fastlock_release(&mr->domain->dev_if->lock);

	ret = cxil_unmap(mr->domain->dev_if->if_lni, &mr->md);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap MR buffer: %d\n", ret);

	ret = cxil_unmap_pte(mr->pte_map);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap PTE: %d\n", ret);

	ret = cxip_if_domain_lep_free(mr->ep->ep_obj->if_dom, mr->pid_idx);
	if (ret)
		CXIP_LOG_ERROR("Failed to free LEP: %d\n", ret);

	ret = cxil_destroy_pte(mr->pte);
	if (ret)
		CXIP_LOG_ERROR("Failed to free PTE: %d\n", ret);

	mr->enabled = 0;

	return FI_SUCCESS;
}

static int cxip_mr_close(struct fid *fid)
{
	struct cxip_mr *mr;
	int ret;

	if (!fid)
		return -FI_EINVAL;

	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	fastlock_acquire(&mr->lock);

	ret = cxip_mr_disable(mr);
	if (ret != FI_SUCCESS)
		CXIP_LOG_DBG("Failed to disable MR: %d\n", ret);

	if (mr->ep)
		ofi_atomic_dec32(&mr->ep->ep_obj->ref);

	ofi_atomic_dec32(&mr->domain->ref);

	fastlock_release(&mr->lock);

	free(mr);

	return FI_SUCCESS;
}

static int cxip_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxip_mr *mr;
	struct cxip_cntr *cntr;
	struct cxip_cq *cq;
	struct cxip_ep *ep;
	int ret = FI_SUCCESS;

	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	fastlock_acquire(&mr->lock);

	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct cxip_cq, cq_fid.fid);
		if (mr->domain != cq->domain) {
			ret = -FI_EINVAL;
			break;
		}

		if (flags & FI_REMOTE_WRITE)
			mr->cq = cq;
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxip_cntr, cntr_fid.fid);
		if (mr->domain != cntr->domain) {
			ret = -FI_EINVAL;
			break;
		}

		if (flags & FI_REMOTE_WRITE)
			mr->cntr = cntr;
		break;

	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		ep = container_of(bfid, struct cxip_ep, ep.fid);

		/* -An MR may only be bound once.
		 * -The EP and MR must be part of the same FI Domain.
		 * -An EP must be enabled before being bound.
		 */
		if (mr->ep ||
		    mr->domain != ep->ep_obj->domain ||
		    !ep->ep_obj->is_enabled) {
			ret = -FI_EINVAL;
			break;
		}

		mr->ep = ep;
		ofi_atomic_inc32(&ep->ep_obj->ref);
		break;

	default:
		ret = -FI_EINVAL;
	}

	fastlock_release(&mr->lock);

	return ret;
}

static int cxip_mr_control(struct fid *fid, int command, void *arg)
{
	struct cxip_mr *mr;
	int ret;

	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	fastlock_acquire(&mr->lock);

	switch (command) {
	case FI_ENABLE:
		/* An MR must be bound to an EP before being enabled. */
		if (!mr->ep) {
			ret = -FI_EINVAL;
			break;
		}

		ret = cxip_mr_enable(mr);
		if (ret != FI_SUCCESS)
			CXIP_LOG_DBG("Failed to enable MR: %d\n", ret);

		break;

	default:
		ret = -FI_EINVAL;
	}

	fastlock_release(&mr->lock);

	return ret;
}

static struct fi_ops cxip_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_mr_close,
	.bind = cxip_mr_bind,
	.control = cxip_mr_control,
	.ops_open = fi_no_ops_open,
};

static int cxip_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	//struct fi_eq_entry eq_entry;
	struct cxip_domain *dom;
	struct cxip_mr *_mr;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	if (attr->requested_key >= CXIP_ADDR_MR_IDX_CNT)
		return -FI_EINVAL;

	/* Only support length 1 IOVs for now */
	if (attr->iov_count != 1)
		return -FI_ENOSYS;

	dom = container_of(fid, struct cxip_domain, dom_fid.fid);

	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	fastlock_init(&_mr->lock);

	_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	_mr->mr_fid.fid.context = attr->context;
	_mr->mr_fid.fid.ops = &cxip_mr_fi_ops;

	_mr->domain = dom;
	_mr->flags = flags;
	_mr->attr = *attr;

	/* Support length 1 IOV only for now */
	_mr->buf = _mr->attr.mr_iov[0].iov_base;
	_mr->len = _mr->attr.mr_iov[0].iov_len;

	_mr->mr_fid.key = _mr->key = attr->requested_key;
	_mr->mr_fid.mem_desc = (void *)_mr;

	ofi_atomic_inc32(&dom->ref);

/* TODO EQs */
#if 0
	if (dom->mr_eq) {
		eq_entry.fid = &dom->dom_fid;
		eq_entry.context = attr->context;

		return cxi_eq_report_event(dom->mr_eq, FI_MR_COMPLETE,
					   &eq_entry, sizeof(eq_entry), 0);
	}
#endif

	*mr = &_mr->mr_fid;

	return 0;
}

static int cxip_regv(struct fid *fid, const struct iovec *iov, size_t count,
		     uint64_t access, uint64_t offset, uint64_t requested_key,
		     uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;

	return cxip_regattr(fid, &attr, flags, mr);
}

static int cxip_reg(struct fid *fid, const void *buf, size_t len,
		    uint64_t access, uint64_t offset, uint64_t requested_key,
		    uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	return cxip_regv(fid, &iov, 1, access, offset, requested_key, flags, mr,
			 context);
}

struct fi_ops_mr cxip_dom_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = cxip_reg,
	.regv = cxip_regv,
	.regattr = cxip_regattr,
};
