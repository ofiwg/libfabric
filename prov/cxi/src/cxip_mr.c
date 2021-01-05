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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_MR, __VA_ARGS__)

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
 * cxip_ep_mr_remove() - Remove an MR key from the EP key space.
 */
static void cxip_ep_mr_remove(struct cxip_mr *mr)
{
	fastlock_acquire(&mr->ep->ep_obj->lock);

	dlist_remove(&mr->ep_entry);

	fastlock_release(&mr->ep->ep_obj->lock);
}

/*
 * cxip_mr_cb() - Process MR LE events.
 */
int cxip_mr_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_mr *mr = req->mr.mr;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		assert(cxi_event_rc(event) == C_RC_OK);

		if (mr->optimized)
			assert(mr->mr_state == CXIP_MR_ENABLED);
		else
			assert(mr->mr_state == CXIP_MR_DISABLED);

		mr->mr_state = CXIP_MR_LINKED;

		CXIP_DBG("MR PTE linked: %p\n", mr);
		break;
	case C_EVENT_UNLINK:
		assert(cxi_event_rc(event) == C_RC_OK);

		assert(mr->mr_state == CXIP_MR_LINKED);
		mr->mr_state = CXIP_MR_UNLINKED;

		CXIP_DBG("MR PTE unlinked: %p\n", mr);
		break;
	default:
		CXIP_WARN("Unexpected event received: %s\n",
			  cxi_event_to_str(event));
	}

	return FI_SUCCESS;
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
	int buffer_id;
	int ret;
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;

	if (mr->cntr) {
		ret = cxip_cntr_enable(mr->cntr);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("cxip_cntr_enable() returned: %d\n", ret);
			return ret;
		}
	}

	fastlock_acquire(&ep_obj->lock);
	buffer_id = ofi_idx_insert(&ep_obj->req_ids, &mr->req);
	if (buffer_id < 0 || buffer_id >= CXIP_BUFFER_ID_MAX) {
		CXIP_WARN("Failed to allocate MR buffer ID: %d\n",
			  buffer_id);
		fastlock_release(&ep_obj->lock);
		return -FI_ENOSPC;
	}
	fastlock_release(&ep_obj->lock);

	mr->req.req_id = buffer_id;
	mr->req.cb = cxip_mr_cb;
	mr->req.mr.mr = mr;

	if (mr->len) {
		ret = cxip_map(mr->domain, (void *)mr->buf, mr->len, &mr->md);
		if (ret) {
			CXIP_WARN("Failed to map MR buffer: %d\n", ret);
			goto err_free_idx;
		}
	}

	le_flags = C_LE_EVENT_SUCCESS_DISABLE | C_LE_UNRESTRICTED_BODY_RO;
	if (mr->attr.access & FI_REMOTE_WRITE)
		le_flags |= C_LE_OP_PUT;
	if (mr->attr.access & FI_REMOTE_READ)
		le_flags |= C_LE_OP_GET;
	if (mr->cntr)
		le_flags |= C_LE_EVENT_CT_COMM;

	ret = cxip_pte_append(ep_obj->ctrl_pte,
			      mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0,
			      mr->len, mr->len ? mr->md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, mr->req.req_id,
			      mr->key, 0, CXI_MATCH_ID_ANY,
			      0, le_flags, mr->cntr, ep_obj->ctrl_tgq, true);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to write Append command: %d\n", ret);
		goto err_unmap;
	}

	/* Wait for Rendezvous PTE state changes */
	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_LINKED);

	mr->enabled = true;

	CXIP_DBG("Standard MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;

err_unmap:
	if (mr->len)
		cxip_unmap(mr->md);
err_free_idx:
	fastlock_acquire(&ep_obj->lock);
	ofi_idx_remove(&ep_obj->req_ids, mr->req.req_id);
	fastlock_release(&ep_obj->lock);

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
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;

	ret = cxip_pte_unlink(ep_obj->ctrl_pte, C_PTL_LIST_PRIORITY,
			      mr->req.req_id, ep_obj->ctrl_tgq);
	if (ret) {
		CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
		goto cleanup;
	}

	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_UNLINKED);

	ret = cxil_invalidate_pte_le(ep_obj->ctrl_pte->pte, mr->key,
				     C_PTL_LIST_PRIORITY);
	if (ret)
		CXIP_WARN("MR invalidate failed: %d (mr: %p key %lu)\n",
			  ret, mr, mr->key);

cleanup:
	if (mr->len)
		cxip_unmap(mr->md);

	fastlock_acquire(&ep_obj->lock);
	ofi_idx_remove(&ep_obj->req_ids, mr->req.req_id);
	fastlock_release(&ep_obj->lock);

	mr->enabled = false;

	CXIP_DBG("Standard MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

/*
 * cxip_mr_opt_pte_cb() - Process optimized MR state change events.
 */
void cxip_mr_opt_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state)
{
	struct cxip_mr *mr = (struct cxip_mr *)pte->ctx;

	switch (state) {
	case C_PTLTE_ENABLED:
		assert(mr->mr_state == CXIP_MR_DISABLED);
		mr->mr_state = CXIP_MR_ENABLED;

		CXIP_DBG("MR PTE enabled: %p\n", mr);
		break;
	default:
		CXIP_WARN("Unexpected state received: %u\n", state);
	}
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
	int buffer_id;
	struct cxi_pt_alloc_opts opts = {};
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;

	if (mr->cntr) {
		ret = cxip_cntr_enable(mr->cntr);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("cxip_cntr_enable() returned: %d\n", ret);
			return ret;
		}
	}

	fastlock_acquire(&ep_obj->lock);
	buffer_id = ofi_idx_insert(&ep_obj->req_ids, &mr->req);
	if (buffer_id < 0 || buffer_id >= CXIP_BUFFER_ID_MAX) {
		CXIP_WARN("Failed to allocate MR buffer ID: %d\n",
			  buffer_id);
		fastlock_release(&ep_obj->lock);
		return -FI_ENOSPC;
	}
	fastlock_release(&ep_obj->lock);

	mr->req.req_id = buffer_id;
	mr->req.cb = cxip_mr_cb;
	mr->req.mr.mr = mr;

	if (mr->len) {
		ret = cxip_map(mr->domain, (void *)mr->buf, mr->len, &mr->md);
		if (ret) {
			CXIP_WARN("Failed to map MR buffer: %d\n", ret);
			goto err_free_idx;
		}
	}

	ret = cxip_pte_alloc_nomap(ep_obj->if_dom[0], ep_obj->ctrl_evtq, &opts,
				   cxip_mr_opt_pte_cb, mr, &mr->pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate PTE: %d\n", ret);
		goto err_unmap;
	}

	ret = cxip_pte_map(mr->pte, CXIP_PTL_IDX_WRITE_MR_OPT(mr->key), false);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map write PTE: %d\n", ret);
		goto err_pte_free;
	}

	ret = cxip_pte_map(mr->pte, CXIP_PTL_IDX_READ_MR_OPT(mr->key), false);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to map write PTE: %d\n", ret);
		goto err_pte_free;
	}

	ret = cxip_pte_set_state(mr->pte, ep_obj->ctrl_tgq, C_PTLTE_ENABLED, 0);
	if (ret != FI_SUCCESS) {
		/* This is a bug, we have exclusive access to this CMDQ. */
		CXIP_WARN("Failed to enqueue command: %d\n", ret);
		goto err_pte_free;
	}

	le_flags = C_LE_EVENT_COMM_DISABLE | C_LE_EVENT_SUCCESS_DISABLE |
		   C_LE_UNRESTRICTED_BODY_RO;
	if (mr->attr.access & FI_REMOTE_WRITE)
		le_flags |= C_LE_OP_PUT;
	if (mr->attr.access & FI_REMOTE_READ)
		le_flags |= C_LE_OP_GET;
	if (mr->cntr)
		le_flags |= C_LE_EVENT_CT_COMM;

	ret = cxip_pte_append(mr->pte,
			      mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0,
			      mr->len, mr->len ? mr->md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, mr->req.req_id,
			      mr->key, 0, CXI_MATCH_ID_ANY,
			      0, le_flags, mr->cntr, ep_obj->ctrl_tgq, true);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to write Append command: %d\n", ret);
		goto err_pte_free;
	}

	/* Wait for Rendezvous PTE state changes */
	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_LINKED);

	mr->enabled = true;

	CXIP_DBG("Optimized MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;

err_pte_free:
	cxip_pte_free(mr->pte);
err_unmap:
	if (mr->len)
		cxip_unmap(mr->md);
err_free_idx:
	fastlock_acquire(&ep_obj->lock);
	ofi_idx_remove(&ep_obj->req_ids, mr->req.req_id);
	fastlock_release(&ep_obj->lock);

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
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;

	ret = cxip_pte_unlink(mr->pte, C_PTL_LIST_PRIORITY,
			      mr->req.req_id, ep_obj->ctrl_tgq);
	if (ret) {
		CXIP_WARN("Failed to enqueue Unlink: %d\n", ret);
		goto cleanup;
	}

	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_UNLINKED);

cleanup:
	cxip_pte_free(mr->pte);

	if (mr->len)
		cxip_unmap(mr->md);

	fastlock_acquire(&ep_obj->lock);
	ofi_idx_remove(&ep_obj->req_ids, mr->req.req_id);
	fastlock_release(&ep_obj->lock);

	mr->enabled = false;

	CXIP_DBG("Optimized MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

int cxip_mr_enable(struct cxip_mr *mr)
{
	int ret;

	if (mr->enabled)
		return FI_SUCCESS;

	ret = cxip_ep_mr_insert(mr->ep->ep_obj, mr);
	if (ret) {
		CXIP_WARN("Failed to insert MR key: %lu\n", mr->key);
		return ret;
	}

	if (mr->optimized)
		ret = cxip_mr_enable_opt(mr);
	else
		ret = cxip_mr_enable_std(mr);

	if (ret != FI_SUCCESS)
		cxip_ep_mr_remove(mr);

	return ret;
}

int cxip_mr_disable(struct cxip_mr *mr)
{
	int ret;

	if (!mr->enabled)
		return FI_SUCCESS;

	if (mr->optimized)
		ret = cxip_mr_disable_opt(mr);
	else
		ret = cxip_mr_disable_std(mr);

	cxip_ep_mr_remove(mr);

	return ret;
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
		CXIP_WARN("Failed to disable MR: %d\n", ret);

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
			CXIP_WARN("Failed to enable MR: %d\n", ret);

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

	_mr->optimized = cxip_mr_key_opt(_mr->key);

	_mr->mr_state = CXIP_MR_DISABLED;

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
