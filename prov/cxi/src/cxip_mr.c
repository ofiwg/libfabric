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

/*
 * cxip_ep_mr_insert() - Insert an MR key into the EP key space.
 *
 * Called during MR enable. The key space is a sparse 64 bits.
 */
static int cxip_ep_mr_insert(struct cxip_ep_obj *ep_obj, struct cxip_mr *mr)
{
	struct cxip_mr *tmp_mr;

	fastlock_acquire(&ep_obj->lock);

	dlist_foreach_container(&ep_obj->mr_list, struct cxip_mr, tmp_mr,
				ep_entry) {
		if (tmp_mr->key == mr->key) {
			fastlock_release(&ep_obj->lock);
			return -FI_ENOKEY;
		}
	}

	dlist_insert_tail(&mr->ep_entry, &ep_obj->mr_list);

	fastlock_release(&ep_obj->lock);

	return FI_SUCCESS;
}

/*
 * cxip_ep_mr_insert() - Remove an MR key from the EP key space.
 */
static void cxip_ep_mr_remove(struct cxip_mr *mr)
{
	fastlock_acquire(&mr->ep->ep_obj->lock);

	dlist_remove(&mr->ep_entry);

	fastlock_release(&mr->ep->ep_obj->lock);
}

/*
 * cxip_ep_mr_enable() - Initialize the endpoint for the use of standard MRs.
 */
static int cxip_ep_mr_enable(struct cxip_ep_obj *ep_obj)
{
	struct cxi_pt_alloc_opts pt_opts = {
		.use_long_event = 1,
		.is_matching = 1,
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY
	};
	struct cxi_eq_attr eq_attr = {};
	union c_cmdu cmd = {};
	const union c_event *event;
	struct cxi_cq_alloc_opts cq_opts = {};
	int ret;
	int tmp;

	fastlock_acquire(&ep_obj->lock);

	if (ep_obj->mr_init) {
		ret = FI_SUCCESS;
		goto unlock_ep;
	}

	cq_opts.count = 64;
	cq_opts.is_transmit = 0;
	ret = cxip_cmdq_alloc(ep_obj->domain->lni, NULL, &cq_opts,
			      &ep_obj->tgq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate MR CMDQ, ret: %d\n",
			     ret);
		ret = -FI_ENODEV;
		goto unlock_ep;
	}

	ep_obj->evtq_buf_len = C_PAGE_SIZE;
	ep_obj->evtq_buf = aligned_alloc(C_PAGE_SIZE, ep_obj->evtq_buf_len);
	if (!ep_obj->evtq_buf) {
		CXIP_LOG_DBG("Unable to allocate EVTQ buffer\n");
		goto free_tgq;
	}

	ret = cxil_map(ep_obj->domain->lni->lni, ep_obj->evtq_buf,
		       ep_obj->evtq_buf_len,
		       CXI_MAP_PIN | CXI_MAP_WRITE,
		       NULL, &ep_obj->evtq_buf_md);
	if (ret) {
		CXIP_LOG_DBG("Unable to map EVTQ buffer, ret: %d\n",
			     ret);
		goto free_evtq_buf;
	}

	eq_attr.queue = ep_obj->evtq_buf;
	eq_attr.queue_len = ep_obj->evtq_buf_len;
	eq_attr.queue_md = ep_obj->evtq_buf_md;
	eq_attr.flags = CXI_EQ_TGT_LONG;

	ret = cxil_alloc_evtq(ep_obj->domain->lni->lni, &eq_attr,
			      NULL, NULL, &ep_obj->evtq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to allocate EVTQ, ret: %d\n", ret);
		ret = -FI_ENODEV;
		goto free_evtq_md;
	}

	memset(&ep_obj->mr_table, 0, sizeof(ep_obj->mr_table));

	ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->evtq,
			     CXIP_PTL_IDX_MR_STD, &pt_opts, &ep_obj->mr_pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		ret = -FI_ENOSPC;
		goto free_evtq;
	}

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = ep_obj->mr_pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(ep_obj->tgq->dev_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto free_pte;
	}

	cxi_cq_ring(ep_obj->tgq->dev_cmdq);

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != ep_obj->mr_pte->pte->ptn) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto free_pte;
	}

	cxi_eq_ack_events(ep_obj->evtq);

	ep_obj->mr_init = true;

	CXIP_LOG_DBG("Standard MRs enabled: %p\n", ep_obj);

	fastlock_release(&ep_obj->lock);

	return FI_SUCCESS;

free_pte:
	cxip_pte_free(ep_obj->mr_pte);
free_evtq:
	ret = cxil_destroy_evtq(ep_obj->evtq);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy EVTQ: %d\n", ret);
free_evtq_md:
	tmp = cxil_unmap(ep_obj->evtq_buf_md);
	if (tmp)
		CXIP_LOG_ERROR("Failed to unmap EVTQ buffer: %d\n", ret);
free_evtq_buf:
	free(ep_obj->evtq_buf);
free_tgq:
	cxip_cmdq_free(ep_obj->tgq);
unlock_ep:
	fastlock_release(&ep_obj->lock);

	return ret;
}

/*
 * cxip_ep_mr_disable() - Free endpoint resources used for standard MRs.
 */
void cxip_ep_mr_disable(struct cxip_ep_obj *ep_obj)
{
	int ret;

	fastlock_acquire(&ep_obj->lock);

	if (!ep_obj->mr_init) {
		fastlock_release(&ep_obj->lock);
		return;
	}

	cxip_pte_free(ep_obj->mr_pte);

	ofi_idx_reset(&ep_obj->mr_table);

	ret = cxil_destroy_evtq(ep_obj->evtq);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy EVTQ: %d\n", ret);

	ret = cxil_unmap(ep_obj->evtq_buf_md);
	if (ret)
		CXIP_LOG_ERROR("Failed to unmap EVTQ buffer: %d\n",
			       ret);

	free(ep_obj->evtq_buf);

	cxip_cmdq_free(ep_obj->tgq);

	ep_obj->mr_init = false;

	fastlock_release(&ep_obj->lock);

	CXIP_LOG_DBG("Standard MRs disabled: %p\n", ep_obj);
}

/*
 * cxip_mr_enable_std() - Assign HW resources to the standard MR.
 *
 * Standard MRs are implemented by linking an LE describing the registered
 * buffer to a shared, matching PtlTE. The MR key is encoded in the LE match
 * bits. One PtlTE supports many standard MRs. The number of standard MR
 * supported is limited by the total number of NIC LEs. Because a matching LE
 * is used, unrestricted commands must be used to target standard MRs.
 *
 * Caller must hold mr->lock.
 */
static int cxip_mr_enable_std(struct cxip_mr *mr)
{
	const union c_event *event;
	int buffer_id;
	int ret;
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;

	if (mr->cntr) {
		ret = cxip_cntr_enable(mr->cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable() returned: %d\n",
				     ret);
			return ret;
		}
	}

	buffer_id = ofi_idx_insert(&ep_obj->mr_table, mr);
	if (buffer_id < 0 || buffer_id >= CXIP_BUFFER_ID_MAX) {
		CXIP_LOG_ERROR("Failed to allocate MR buffer ID: %d\n",
			       buffer_id);
		return -FI_ENOSPC;
	}
	mr->buffer_id = buffer_id;

	if (mr->len) {
		ret = cxip_map(mr->domain, (void *)mr->buf, mr->len, &mr->md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map MR buffer: %d\n", ret);
			goto err_free_idx;
		}
	}

	fastlock_acquire(&ep_obj->lock);

	le_flags = C_LE_EVENT_SUCCESS_DISABLE;
	if (mr->attr.access & FI_REMOTE_WRITE)
		le_flags |= C_LE_OP_PUT;
	if (mr->attr.access & FI_REMOTE_READ)
		le_flags |= C_LE_OP_GET;
	if (mr->cntr)
		le_flags |= C_LE_EVENT_CT_COMM;

	ret = cxip_pte_append(ep_obj->mr_pte,
			      mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0,
			      mr->len, mr->len ? mr->md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, mr->buffer_id,
			      mr->key, 0, CXI_MATCH_ID_ANY,
			      0, le_flags, mr->cntr, ep_obj->tgq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto err_unmap;
	}

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_LINK ||
	    event->tgt_long.buffer_id != mr->buffer_id) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Link EQE %u %u %u %u\n",
				event->hdr.event_type,
				event->tgt_long.return_code,
				event->tgt_long.buffer_id, mr->buffer_id);
		ret = -FI_EIO;
		goto err_unmap;
	}

	if (cxi_event_rc(event) != C_RC_OK) {
		CXIP_LOG_ERROR("Append failed: %s\n",
			       cxi_rc_to_str(cxi_event_rc(event)));
		ret = -FI_ENOSPC;
		goto err_unmap;
	}

	cxi_eq_ack_events(ep_obj->evtq);

	fastlock_release(&ep_obj->lock);

	mr->enabled = true;

	CXIP_LOG_DBG("Standard MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;

err_unmap:
	fastlock_release(&ep_obj->lock);

	if (mr->len)
		cxip_unmap(mr->md);
err_free_idx:
	ofi_idx_remove(&ep_obj->mr_table, mr->buffer_id);

	return ret;
}

/*
 * cxip_mr_disable_std() - Free HW resources from the standard MR.
 *
 * Caller must hold mr->lock.
 */
static int cxip_mr_disable_std(struct cxip_mr *mr)
{
	int ret;
	const union c_event *event;
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;

	if (!mr->enabled)
		return FI_SUCCESS;

	fastlock_acquire(&ep_obj->lock);

	ret = cxip_pte_unlink(ep_obj->mr_pte, C_PTL_LIST_PRIORITY,
			      mr->buffer_id, ep_obj->tgq);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n", ret);
		goto unlock;
	}

	/* Wait for unlink EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_UNLINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != mr->buffer_id) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Unlink EQE %s rc: %s (id: 0x%x)\n",
			       cxi_event_to_str(event),
			       cxi_rc_to_str(cxi_event_rc(event)),
			       event->tgt_long.buffer_id);
	}

	cxi_eq_ack_events(ep_obj->evtq);

	ret = cxil_invalidate_pte_le(ep_obj->mr_pte->pte, mr->key,
				     C_PTL_LIST_PRIORITY);
	if (ret)
		CXIP_LOG_ERROR("MR invalidate failed: %d (mr: %p key %lu)\n",
			       ret, mr, mr->key);

unlock:
	fastlock_release(&ep_obj->lock);

	if (mr->len)
		cxip_unmap(mr->md);

	ofi_idx_remove(&ep_obj->mr_table, mr->buffer_id);

	mr->enabled = false;

	CXIP_LOG_DBG("Standard MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

/*
 * cxip_mr_enable_opt() - Assign HW resources to the optimized MR.
 *
 * Optimized MRs are implemented by allocating a dedicated, non-matching PtlTE
 * and linking an LE describing the registered buffer. The MR key is used to
 * derive the PtlTE index. One PtlTE and one LE is required for each optimized
 * MR. Because a non-matching interface is used, optimized MRs can be targeted
 * with restricted commands. This may result in better performance.
 *
 * Caller must hold mr->lock.
 */
static int cxip_mr_enable_opt(struct cxip_mr *mr)
{
	int ret;
	union c_cmdu cmd = {};
	const union c_event *event;
	struct cxi_pt_alloc_opts opts = {
		.pe_num = CXI_PE_NUM_ANY,
		.le_pool = CXI_LE_POOL_ANY
	};
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;

	if (mr->enabled)
		return FI_SUCCESS;

	if (mr->cntr) {
		ret = cxip_cntr_enable(mr->cntr);
		if (ret != FI_SUCCESS) {
			CXIP_LOG_DBG("cxip_cntr_enable() returned: %d\n",
				     ret);
			return ret;
		}
	}

	if (mr->len) {
		ret = cxip_map(mr->domain, (void *)mr->buf, mr->len, &mr->md);
		if (ret) {
			CXIP_LOG_DBG("Failed to map MR buffer: %d\n", ret);
			return ret;
		}
	}

	ret = cxip_pte_alloc(ep_obj->if_dom, ep_obj->evtq,
			     CXIP_PTL_IDX_MR_OPT(mr->key), &opts, &mr->pte);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate PTE: %d\n", ret);
		goto err_unmap;
	}

	fastlock_acquire(&ep_obj->lock);

	/* Enable the PTE */
	cmd.command.opcode = C_CMD_TGT_SETSTATE;
	cmd.set_state.ptlte_index = mr->pte->pte->ptn;
	cmd.set_state.ptlte_state = C_PTLTE_ENABLED;

	ret = cxi_cq_emit_target(ep_obj->tgq->dev_cmdq, &cmd);
	if (ret) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_LOG_ERROR("Failed to enqueue command: %d\n", ret);
		goto err_pte_free;
	}

	cxi_cq_ring(ep_obj->tgq->dev_cmdq);

	/* Wait for Enable event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_STATE_CHANGE ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.initiator.state_change.ptlte_state !=
		    C_PTLTE_ENABLED ||
	    event->tgt_long.ptlte_index != mr->pte->pte->ptn) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Enable EQE\n");
		ret = -FI_EIO;
		goto err_pte_free;
	}

	le_flags = C_LE_EVENT_SUCCESS_DISABLE;
	if (mr->attr.access & FI_REMOTE_WRITE)
		le_flags |= C_LE_OP_PUT;
	if (mr->attr.access & FI_REMOTE_READ)
		le_flags |= C_LE_OP_GET;
	if (mr->cntr)
		le_flags |= C_LE_EVENT_CT_COMM;

	ret = cxip_pte_append(mr->pte,
			      mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0,
			      mr->len, mr->len ? mr->md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, mr->key,
			      mr->key, 0, CXI_MATCH_ID_ANY,
			      0, le_flags, mr->cntr, ep_obj->tgq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to write Append command: %d\n", ret);
		goto err_pte_free;
	}

	/* Wait for link EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_LINK ||
	    event->tgt_long.buffer_id != mr->key) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Link EQE\n");
		ret = -FI_EIO;
		goto err_pte_free;
	}

	if (cxi_event_rc(event) != C_RC_OK) {
		CXIP_LOG_ERROR("Append failed: %s\n",
			       cxi_rc_to_str(cxi_event_rc(event)));
		ret = -FI_ENOSPC;
		goto err_pte_free;
	}

	cxi_eq_ack_events(ep_obj->evtq);

	fastlock_release(&ep_obj->lock);

	mr->enabled = true;

	CXIP_LOG_DBG("Optimized MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;

err_pte_free:
	fastlock_release(&ep_obj->lock);

	cxip_pte_free(mr->pte);
err_unmap:
	if (mr->len)
		cxip_unmap(mr->md);

	return ret;
}

/*
 * cxip_mr_disable_opt() - Free hardware resources from the optimized MR.
 *
 * Caller must hold mr->lock.
 */
static int cxip_mr_disable_opt(struct cxip_mr *mr)
{
	int ret;
	const union c_event *event;
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;

	if (!mr->enabled)
		return FI_SUCCESS;

	fastlock_acquire(&ep_obj->lock);

	ret = cxip_pte_unlink(mr->pte, C_PTL_LIST_PRIORITY,
			      mr->key, ep_obj->tgq);
	if (ret) {
		CXIP_LOG_ERROR("Failed to enqueue Unlink: %d\n", ret);
		goto unlock;
	}

	/* Wait for unlink EQ event */
	while (!(event = cxi_eq_get_event(ep_obj->evtq)))
		sched_yield();

	if (event->hdr.event_type != C_EVENT_UNLINK ||
	    event->tgt_long.return_code != C_RC_OK ||
	    event->tgt_long.buffer_id != mr->key) {
		/* This is a device malfunction */
		CXIP_LOG_ERROR("Invalid Unlink EQE %s rc: %s (id: 0x%x)\n",
			       cxi_event_to_str(event),
			       cxi_rc_to_str(cxi_event_rc(event)),
			       event->tgt_long.buffer_id);
	}

	cxi_eq_ack_events(ep_obj->evtq);

unlock:
	fastlock_release(&ep_obj->lock);

	cxip_pte_free(mr->pte);

	if (mr->len)
		cxip_unmap(mr->md);

	mr->enabled = false;

	CXIP_LOG_DBG("Optimized MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

int cxip_mr_enable(struct cxip_mr *mr)
{
	int ret;

	ret = cxip_ep_mr_enable(mr->ep->ep_obj);
	if (ret != FI_SUCCESS)
		return ret;

	ret = cxip_ep_mr_insert(mr->ep->ep_obj, mr);
	if (ret) {
		CXIP_LOG_ERROR("Failed to insert MR key: %lu\n", mr->key);
		return ret;
	}

	if (cxip_mr_key_opt(mr->key))
		return cxip_mr_enable_opt(mr);
	else
		return cxip_mr_enable_std(mr);
}

int cxip_mr_disable(struct cxip_mr *mr)
{
	if (cxip_mr_key_opt(mr->key))
		return cxip_mr_disable_opt(mr);
	else
		return cxip_mr_disable_std(mr);

	cxip_ep_mr_remove(mr);
}

/*
 * cxip_mr_close() - fi_close implemented for MRs.
 */
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

	if (mr->cntr)
		ofi_atomic_dec32(&mr->cntr->ref);

	ofi_atomic_dec32(&mr->domain->ref);

	fastlock_release(&mr->lock);

	free(mr);

	return FI_SUCCESS;
}

/*
 * cxip_mr_bind() - fi_bind() implementation for MRs.
 */
static int cxip_mr_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxip_mr *mr;
	struct cxip_cntr *cntr;
	struct cxip_ep *ep;
	int ret = FI_SUCCESS;

	mr = container_of(fid, struct cxip_mr, mr_fid.fid);

	fastlock_acquire(&mr->lock);

	switch (bfid->fclass) {
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct cxip_cntr, cntr_fid.fid);
		if (mr->domain != cntr->domain || mr->enabled) {
			ret = -FI_EINVAL;
			break;
		}

		if (mr->cntr) {
			ret = -FI_EINVAL;
			break;
		}

		if (!(flags & FI_REMOTE_WRITE)) {
			ret = -FI_EINVAL;
			break;
		}

		mr->cntr = cntr;
		ofi_atomic_inc32(&cntr->ref);
		break;

	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		ep = container_of(bfid, struct cxip_ep, ep.fid);
		if (mr->domain != ep->ep_obj->domain || mr->enabled) {
			ret = -FI_EINVAL;
			break;
		}

		if (mr->ep || !ep->ep_obj->enabled) {
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

/*
 * cxip_mr_control() - fi_control() implementation for MRs.
 */
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

/*
 * Libfabric MR creation APIs
 */

static int cxip_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	//struct fi_eq_entry eq_entry;
	struct cxip_domain *dom;
	struct cxip_mr *_mr;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	/* Only support length 1 IOVs for now */
	if (attr->iov_count != 1)
		return -FI_ENOSYS;

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid);

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
