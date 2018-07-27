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

#include "cxi_prov.h"

#define CXIX_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIX_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_MR, __VA_ARGS__)

#define MR_LINK_EVENT_ID 0x1e21

int cxix_mr_enable(struct cxix_mr *mr)
{
	int ret;
	union c_cmdu cmd = {};
	const union c_event *event;
	uint32_t buffer_id = MR_LINK_EVENT_ID;
	struct cxi_pt_alloc_opts opts = {};

	/* Enable the Domain used by the MR */
	ret = cxix_domain_enable(mr->domain);
	if (ret != FI_SUCCESS) {
		CXIX_LOG_DBG("Failed to enable Domain: %d\n", ret);
		return ret;
	}

	/* Get the IF Domain where the MR will exist */
	ret = cxix_get_if_domain(mr->domain->dev_if,
				 mr->domain->vni,
				 mr->domain->pid,
				 mr->domain->pid_granule,
				 &mr->if_dom);
	if (ret != FI_SUCCESS) {
		CXIX_LOG_DBG("Failed to get IF Domain: %d\n", ret);
		return ret;
	}

	/* Allocate a PTE */
	ret = cxil_alloc_pte(mr->domain->dev_if->if_lni,
			     mr->domain->dev_if->mr_evtq,
			     &opts, &mr->pte, &mr->pte_hw_id);
	if (ret) {
		CXIX_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto put_if_dom;
	}

	/* Reserve the logical endpoint (LEP) where the MR will be mapped */
	mr->pid_off = CXIX_ADDR_MR_IDX(mr->domain->pid_granule, mr->key);

	ret = cxix_if_domain_lep_alloc(mr->if_dom, mr->pid_off);
	if (ret != FI_SUCCESS) {
		CXIX_LOG_DBG("Failed to reserve LEP (%d): %d\n",
			     mr->pid_off, ret);
		goto free_pte;
	}

	/* Map the PTE to the LEP */
	ret = cxil_map_pte(mr->pte, mr->if_dom->if_dom, mr->pid_off, 0,
			   &mr->pte_map);
	if (ret) {
		CXIX_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_EADDRINUSE;
		goto free_lep;
	}

	/* Map the window buffer into IO address space */
	ret = cxil_map(mr->domain->dev_if->if_lni, mr->buf, mr->len,
		       CXI_MAP_PIN | CXI_MAP_NTA | CXI_MAP_READ | CXI_MAP_WRITE,
		       &mr->md);
	if (ret) {
		CXIX_LOG_DBG("Failed to IO map MR buffer: %d\n", ret);
		ret = -FI_EFAULT;
		goto unmap_pte;
	}

	/* Use the device CMDQ and EQ to enable the PTE and append an LE.
	 * This serializes creation of all MRs in the process.  We need to
	 * revisit.
	 */
	fastlock_acquire(&mr->domain->dev_if->lock);

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = mr->pte_hw_id;
	cmd.set_state.ptlte_state  = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIX_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unmap_buf;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
			C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != mr->pte_hw_id) {
		/* This is a device malfunction */
		CXIX_LOG_ERROR("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto unmap_buf;
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

	/* Link Persistent LE */
	memset(&cmd, 0, sizeof(cmd));
	cmd.command.opcode     = C_CMD_TGT_APPEND;
	cmd.target.ptl_list    = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = mr->pte_hw_id;
	cmd.target.no_truncate = 0;
	cmd.target.unexpected_hdr_disable = 0;
	cmd.target.buffer_id   = buffer_id;
	cmd.target.lac         = mr->md.lac;
	cmd.target.start       = mr->md.iova + ((uint64_t)mr->buf - mr->md.va);
	cmd.target.length      = mr->len;
	cmd.target.match_bits  = 0;

	if (mr->flags & FI_REMOTE_WRITE)
		cmd.target.op_put = 1;
	if (mr->flags & FI_REMOTE_READ)
		cmd.target.op_get = 1;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIX_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unmap_buf;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->event_type != C_EVENT_LINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != buffer_id) {
		/* This is a device malfunction */
		CXIX_LOG_ERROR("Invalid Link EQE\n");
		ret = -FI_EIO;
		goto unmap_buf;
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

	fastlock_release(&mr->domain->dev_if->lock);

	return FI_SUCCESS;

unmap_buf:
	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);
	fastlock_release(&mr->domain->dev_if->lock);

	cxil_unmap(mr->domain->dev_if->if_lni, &mr->md);
unmap_pte:
	cxil_unmap_pte(mr->pte_map);
free_lep:
	cxix_if_domain_lep_free(mr->if_dom, mr->pid_off);
free_pte:
	cxil_destroy_pte(mr->pte);
put_if_dom:
	cxix_put_if_domain(mr->if_dom);

	return ret;
}

int cxix_mr_disable(struct cxix_mr *mr)
{
	int ret;
	union c_cmdu cmd = {};
	const union c_event *event;
	uint32_t buffer_id = MR_LINK_EVENT_ID;

	/* Use the device CMDQ and EQ to unlink the LE.  This serializes
	 * enable/disable of all MRs in the process.  We need to revisit.
	 */
	fastlock_acquire(&mr->domain->dev_if->lock);

	/* Unlink persistent LE */
	cmd.command.opcode = C_CMD_TGT_UNLINK;
	cmd.target.ptl_list = C_PTL_LIST_PRIORITY;
	cmd.target.ptlte_index = mr->pte_hw_id;
	cmd.target.buffer_id = buffer_id;

	ret = cxi_cq_emit_target(mr->domain->dev_if->mr_cmdq, &cmd);
	if (ret) {
		/* This is a provider bug, we have exclusive access to this
		 * CMDQ.
		 */
		CXIX_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto unlock;
	}

	cxi_cq_ring(mr->domain->dev_if->mr_cmdq);

	/* Wait for unlink EQ event */
	while (!(event = cxi_eq_get_event(mr->domain->dev_if->mr_evtq)))
		sched_yield();

	if (event->event_type != C_EVENT_UNLINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != buffer_id) {
		/* This is a device malfunction */
		CXIX_LOG_ERROR("Invalid Unlink EQE\n");
	}

	cxi_eq_ack_events(mr->domain->dev_if->mr_evtq);

unlock:
	fastlock_release(&mr->domain->dev_if->lock);

	ret = cxil_unmap(mr->domain->dev_if->if_lni, &mr->md);
	if (ret)
		CXIX_LOG_ERROR("Failed to unmap MR buffer: %d\n", ret);

	ret = cxil_unmap_pte(mr->pte_map);
	if (ret)
		CXIX_LOG_ERROR("Failed to unmap PTE: %d\n", ret);

	ret = cxix_if_domain_lep_free(mr->if_dom, mr->pid_off);
	if (ret)
		CXIX_LOG_ERROR("Failed to free LEP: %d\n", ret);

	ret = cxil_destroy_pte(mr->pte);
	if (ret)
		CXIX_LOG_ERROR("Failed to free PTE: %d\n", ret);

	cxix_put_if_domain(mr->if_dom);

	return FI_SUCCESS;
}

static int cxix_mr_close(struct fid *fid)
{
	struct cxix_mr *mr;
	struct cxi_domain *dom;
	int ret;

	mr = container_of(fid, struct cxix_mr, mr_fid.fid);
	dom = mr->domain;

	ret = cxix_mr_disable(mr);
	if (ret != FI_SUCCESS)
		CXIX_LOG_DBG("Failed to disable MR: %d\n", ret);

	ofi_atomic_dec32(&dom->ref);
	free(mr);

	return 0;
}

static int cxix_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_cntr *cntr;
	struct cxi_cq *cq;
	struct cxix_mr *mr;

	mr = container_of(fid, struct cxix_mr, mr_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct cxi_cq, cq_fid.fid);
		if (mr->domain != cq->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cq = cq;
		break;

	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxi_cntr, cntr_fid.fid);
		if (mr->domain != cntr->domain)
			return -FI_EINVAL;

		if (flags & FI_REMOTE_WRITE)
			mr->cntr = cntr;
		break;

	default:
		return -FI_EINVAL;
	}
	return 0;
}

static struct fi_ops cxix_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxix_mr_close,
	.bind = cxix_mr_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int cxix_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	//struct fi_eq_entry eq_entry;
	struct cxi_domain *dom;
	struct cxix_mr *_mr;
	int ret = 0;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	/* Only support length 1 IOVs for now */
	if (attr->iov_count != 1)
		return -FI_ENOSYS;

	dom = container_of(fid, struct cxi_domain, dom_fid.fid);

	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	_mr->mr_fid.fid.fclass = FI_CLASS_MR;
	_mr->mr_fid.fid.context = attr->context;
	_mr->mr_fid.fid.ops = &cxix_mr_fi_ops;

	_mr->domain = dom;
	_mr->flags = flags;
	_mr->attr = *attr;

	/* Support length 1 IOV only for now */
	_mr->buf = _mr->attr.mr_iov[0].iov_base;
	_mr->len = _mr->attr.mr_iov[0].iov_len;

	_mr->mr_fid.key = _mr->key = attr->requested_key;
	_mr->mr_fid.mem_desc = (void *)_mr;

	ofi_atomic_inc32(&dom->ref);

	ret = cxix_mr_enable(_mr);
	if (ret != FI_SUCCESS) {
		CXIX_LOG_DBG("Failed to enable MR: %d\n", ret);
		goto free_mr;
	}

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

free_mr:
	ofi_atomic_dec32(&dom->ref);
	free(_mr);

	return ret;
}

static int cxix_regv(struct fid *fid, const struct iovec *iov,
		     size_t count, uint64_t access,
		     uint64_t offset, uint64_t requested_key,
		     uint64_t flags, struct fid_mr **mr, void *context)
{
	struct fi_mr_attr attr;

	attr.mr_iov = iov;
	attr.iov_count = count;
	attr.access = access;
	attr.offset = offset;
	attr.requested_key = requested_key;
	attr.context = context;

	return cxix_regattr(fid, &attr, flags, mr);
}

static int cxix_reg(struct fid *fid, const void *buf, size_t len,
		    uint64_t access, uint64_t offset, uint64_t requested_key,
		    uint64_t flags, struct fid_mr **mr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	return cxix_regv(fid, &iov, 1, access, offset, requested_key,
			 flags, mr, context);
}

struct fi_ops_mr cxix_dom_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = cxix_reg,
	.regv = cxix_regv,
	.regattr = cxix_regattr,
};

