/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2017 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>
#include <fasthash.h>
#include <ofi_util.h>

#include "cxip.h"

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_MR, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_MR, __VA_ARGS__)

static int cxip_mr_init(struct cxip_mr *mr, struct cxip_domain *dom,
			const struct fi_mr_attr *attr, uint64_t flags);
static void cxip_mr_fini(struct cxip_mr *mr);

static void cxip_mr_domain_remove(struct cxip_mr *mr)
{
	ofi_spin_lock(&mr->domain->mr_domain.lock);
	dlist_remove(&mr->mr_domain_entry);
	ofi_spin_unlock(&mr->domain->mr_domain.lock);
}

static int cxip_mr_domain_insert(struct cxip_mr *mr)
{
	struct cxip_mr_domain *mr_domain = &mr->domain->mr_domain;
	int bucket;
	struct cxip_mr *clash_mr;

	if (!mr->domain->mr_util->key_is_valid(mr->key))
		return -FI_EKEYREJECTED;

	bucket = fasthash64(&mr->key, sizeof(mr->key), 0) %
		CXIP_MR_DOMAIN_HT_BUCKETS;

	ofi_spin_lock(&mr_domain->lock);

	dlist_foreach_container(&mr_domain->buckets[bucket], struct cxip_mr,
				clash_mr, mr_domain_entry) {
		if (clash_mr->key == mr->key) {
			ofi_spin_unlock(&mr_domain->lock);
			return -FI_ENOKEY;
		}
	}

	dlist_insert_tail(&mr->mr_domain_entry, &mr_domain->buckets[bucket]);

	ofi_spin_unlock(&mr_domain->lock);

	return FI_SUCCESS;
}

static void cxip_mr_domain_remove_prov(struct cxip_mr *mr)
{
}

static int cxip_mr_domain_insert_prov(struct cxip_mr *mr)
{
	return FI_SUCCESS;
}

void cxip_mr_domain_fini(struct cxip_mr_domain *mr_domain)
{
	int i;

	/* Assumption is this is only called when a domain is freed and only a
	 * single thread should be freeing a domain. Thus, no lock is taken.
	 */
	for (i = 0; i < CXIP_MR_DOMAIN_HT_BUCKETS; i++) {
		if (!dlist_empty(&mr_domain->buckets[i]))
			CXIP_WARN("MR domain bucket %d is not empty\n", i);
	}

	ofi_spin_destroy(&mr_domain->lock);
}

void cxip_mr_domain_init(struct cxip_mr_domain *mr_domain)
{
	int i;

	ofi_spin_init(&mr_domain->lock);

	for (i = 0; i < CXIP_MR_DOMAIN_HT_BUCKETS; i++)
		dlist_init(&mr_domain->buckets[i]);
}

/*
 * cxip_ep_mr_insert() - Insert an MR key into the EP key space.
 *
 * Called during MR enable. The key space is a sparse 64 bits.
 */
static void cxip_ep_mr_insert(struct cxip_ep_obj *ep_obj, struct cxip_mr *mr)
{
	ofi_mutex_lock(&ep_obj->lock);
	dlist_insert_tail(&mr->ep_entry, &ep_obj->mr_list);
	ofi_mutex_unlock(&ep_obj->lock);
}

/*
 * cxip_ep_mr_remove() - Remove an MR key from the EP key space.
 */
static void cxip_ep_mr_remove(struct cxip_mr *mr)
{
	ofi_mutex_lock(&mr->ep->ep_obj->lock);
	dlist_remove(&mr->ep_entry);
	ofi_mutex_unlock(&mr->ep->ep_obj->lock);
}

/*
 * cxip_mr_cb() - Process MR LE events.
 */
int cxip_mr_cb(struct cxip_ctrl_req *req, const union c_event *event)
{
	struct cxip_mr *mr = req->mr.mr;

	switch (event->hdr.event_type) {
	case C_EVENT_LINK:
		if (mr->optimized)
			assert(mr->mr_state == CXIP_MR_ENABLED);
		else
			assert(mr->mr_state == CXIP_MR_DISABLED);

		if (cxi_event_rc(event) == C_RC_OK) {
			mr->mr_state = CXIP_MR_LINKED;
			CXIP_DBG("MR PTE linked: %p\n", mr);
			break;
		}

		mr->mr_state = CXIP_MR_LINK_ERR;
		CXIP_WARN("MR PTE link: %p failed %d\n",
			  mr, cxi_event_rc(event));
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

static int cxip_mr_wait_append(struct cxip_mr *mr)
{
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;

	/* Wait for PTE LE append status update */
	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_LINKED &&
		 mr->mr_state != CXIP_MR_LINK_ERR);

	if (mr->mr_state == CXIP_MR_LINK_ERR)
		return -FI_ENOSPC;

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
	int ret;
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;

	mr->req.cb = cxip_mr_cb;

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
		return ret;
	}

	ret = cxip_mr_wait_append(mr);
	if (ret)
		return ret;

	mr->enabled = true;

	CXIP_DBG("Standard MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
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

	/* TODO: Handle -FI_EAGAIN. */
	ret = cxip_pte_unlink(ep_obj->ctrl_pte, C_PTL_LIST_PRIORITY,
			      mr->req.req_id, ep_obj->ctrl_tgq);
	assert(ret == FI_SUCCESS);

	do {
		sched_yield();
		cxip_ep_ctrl_progress(ep_obj);
	} while (mr->mr_state != CXIP_MR_UNLINKED);

	ret = cxil_invalidate_pte_le(ep_obj->ctrl_pte->pte, mr->key,
				     C_PTL_LIST_PRIORITY);
	if (ret)
		CXIP_WARN("MR invalidate failed: %d (mr: %p key %lu)\n",
			  ret, mr, mr->key);
	mr->enabled = false;

	CXIP_DBG("Standard MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

/*
 * cxip_mr_opt_pte_cb() - Process optimized MR state change events.
 */
void cxip_mr_opt_pte_cb(struct cxip_pte *pte, const union c_event *event)
{
	struct cxip_mr *mr = (struct cxip_mr *)pte->ctx;

	switch (pte->state) {
	case C_PTLTE_ENABLED:
		assert(mr->mr_state == CXIP_MR_DISABLED);
		mr->mr_state = CXIP_MR_ENABLED;

		CXIP_DBG("MR PTE enabled: %p\n", mr);
		break;
	default:
		CXIP_WARN("Unexpected state received: %u\n", pte->state);
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
	struct cxi_pt_alloc_opts opts = {};
	struct cxip_ep_obj *ep_obj = mr->ep->ep_obj;
	uint32_t le_flags;
	uint64_t ib = 0;

	mr->req.cb = cxip_mr_cb;

	ret = cxip_pte_alloc_nomap(ep_obj->if_dom[0], ep_obj->ctrl_tgt_evtq,
				   &opts, cxip_mr_opt_pte_cb, mr, &mr->pte);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate PTE: %d\n", ret);
		return ret;
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

	/* When FI_FENCE is not requested, restricted operations can used PCIe
	 * relaxed ordering. Unrestricted operations PCIe relaxed ordering is
	 * controlled by an env for now.
	 */
	if (!(ep_obj->caps & FI_FENCE)) {
		ib = 1;

		if (cxip_env.enable_unrestricted_end_ro)
			le_flags |= C_LE_UNRESTRICTED_END_RO;
	}

	ret = cxip_pte_append(mr->pte,
			      mr->len ? CXI_VA_TO_IOVA(mr->md->md, mr->buf) : 0,
			      mr->len, mr->len ? mr->md->md->lac : 0,
			      C_PTL_LIST_PRIORITY, mr->req.req_id,
			      0, ib, CXI_MATCH_ID_ANY,
			      0, le_flags, mr->cntr, ep_obj->ctrl_tgq, true);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to write Append command: %d\n", ret);
		goto err_pte_free;
	}

	ret = cxip_mr_wait_append(mr);
	if (ret)
		goto err_pte_free;

	mr->enabled = true;

	CXIP_DBG("Optimized MR enabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;

err_pte_free:
	cxip_pte_free(mr->pte);

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

	mr->enabled = false;

	CXIP_DBG("Optimized MR disabled: %p (key: %lu)\n", mr, mr->key);

	return FI_SUCCESS;
}

static int cxip_mr_prov_cache_enable_opt(struct cxip_mr *mr)
{
	/* TODO */
	return -FI_ENOSYS;
}

static int cxip_mr_prov_cache_disable_opt(struct cxip_mr *mr)
{
	/* TODO */
	return -FI_ENOSYS;
}

static int cxip_mr_prov_cache_enable_std(struct cxip_mr *mr)
{
	/* TODO */
	return -FI_ENOSYS;
}

static int cxip_mr_prov_cache_disable_std(struct cxip_mr *mr)
{
	/* TODO */
	return -FI_ENOSYS;
}

int cxip_mr_enable(struct cxip_mr *mr)
{
	int ret;

	/* MR which require remote access require additional resources. Locally
	 * access MRs only do not. Thus, return FI_SUCCESS.
	 */
	if (mr->enabled ||
	    !(mr->attr.access & (FI_REMOTE_READ | FI_REMOTE_WRITE)))
		return FI_SUCCESS;

	cxip_ep_mr_insert(mr->ep->ep_obj, mr);

	if (mr->optimized)
		ret = mr->domain->mr_util->enable_opt(mr);
	else
		ret = mr->domain->mr_util->enable_std(mr);

	if (ret != FI_SUCCESS)
		goto err_remove_mr;

	return FI_SUCCESS;

err_remove_mr:
	cxip_ep_mr_remove(mr);

	return ret;
}

int cxip_mr_disable(struct cxip_mr *mr)
{
	int ret;

	if (!mr->enabled ||
	    !(mr->attr.access & (FI_REMOTE_READ | FI_REMOTE_WRITE)))
		return FI_SUCCESS;

	if (mr->optimized)
		ret = mr->domain->mr_util->disable_opt(mr);
	else
		ret = mr->domain->mr_util->disable_std(mr);

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

	ofi_spin_lock(&mr->lock);

	ret = cxip_mr_disable(mr);
	if (ret != FI_SUCCESS)
		CXIP_WARN("Failed to disable MR: %d\n", ret);

	if (mr->len)
		cxip_unmap(mr->md);

	mr->domain->mr_util->domain_remove(mr);
	if (mr->ep)
		ofi_atomic_dec32(&mr->ep->ep_obj->ref);

	if (mr->cntr)
		ofi_atomic_dec32(&mr->cntr->ref);

	cxip_mr_fini(mr);
	ofi_atomic_dec32(&mr->domain->ref);

	ofi_spin_unlock(&mr->lock);

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

	ofi_spin_lock(&mr->lock);

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

	ofi_spin_unlock(&mr->lock);

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

	ofi_spin_lock(&mr->lock);

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

	ofi_spin_unlock(&mr->lock);

	return ret;
}

static struct fi_ops cxip_mr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_mr_close,
	.bind = cxip_mr_bind,
	.control = cxip_mr_control,
	.ops_open = fi_no_ops_open,
};

static void cxip_mr_fini(struct cxip_mr *mr)
{
	cxip_domain_ctrl_id_free(mr->domain, &mr->req);
}

static int cxip_mr_init(struct cxip_mr *mr, struct cxip_domain *dom,
			const struct fi_mr_attr *attr, uint64_t flags)
{
	int ret;

	ofi_spin_init(&mr->lock);

	mr->mr_fid.fid.fclass = FI_CLASS_MR;
	mr->mr_fid.fid.context = attr->context;
	mr->mr_fid.fid.ops = &cxip_mr_fi_ops;

	mr->domain = dom;
	mr->flags = flags;
	mr->attr = *attr;

	/* Support length 1 IOV only for now */
	mr->buf = mr->attr.mr_iov[0].iov_base;
	mr->len = mr->attr.mr_iov[0].iov_len;

	/* Allocate unique MR buffer ID */
	ret = cxip_domain_ctrl_id_alloc(dom, &mr->req);
	if (ret) {
		CXIP_WARN("Failed to allocate MR buffer ID: %d\n", ret);
		return -FI_ENOSPC;
	}
	mr->req.mr.mr = mr;

	ret = mr->domain->mr_util->init_key(mr, attr->requested_key);
	if (ret) {
		CXIP_WARN("Failed to initialize MR key: %d\n", ret);
		return ret;
	}
	mr->mr_fid.key = mr->key;
	mr->mr_fid.mem_desc = (void *)mr;

	mr->optimized = dom->mr_util->key_is_opt(mr->key);

	mr->mr_state = CXIP_MR_DISABLED;

	return FI_SUCCESS;
}

/*
 * Libfabric MR creation APIs
 */

static int cxip_regattr(struct fid *fid, const struct fi_mr_attr *attr,
			uint64_t flags, struct fid_mr **mr)
{
	struct cxip_domain *dom;
	struct cxip_mr *_mr;
	int ret;

	if (fid->fclass != FI_CLASS_DOMAIN || !attr || attr->iov_count <= 0)
		return -FI_EINVAL;

	if (attr->iov_count != 1)
		return -FI_ENOSYS;

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid);

	if (dom->mr_util->is_prov && attr->requested_key)
		return -FI_EINVAL;

	_mr = calloc(1, sizeof(*_mr));
	if (!_mr)
		return -FI_ENOMEM;

	ret = cxip_mr_init(_mr, dom, attr, flags);
	if (ret)
		goto err_free_mr;

	ret = dom->mr_util->domain_insert(_mr);
	if (ret)
		goto err_cleanup_mr;

	if (_mr->len) {
		ret = cxip_map(_mr->domain, (void *)_mr->buf, _mr->len, 0,
			       &_mr->md);
		if (ret) {
			CXIP_WARN("Failed to map MR buffer: %d\n", ret);
			goto err_remove_mr;
		}
	}

	ofi_atomic_inc32(&dom->ref);

	*mr = &_mr->mr_fid;

	return FI_SUCCESS;

err_remove_mr:
	cxip_mr_domain_remove(_mr);
err_cleanup_mr:
	cxip_mr_fini(_mr);
err_free_mr:
	free(_mr);

	return ret;
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

static int cxip_init_mr_key(struct cxip_mr *mr, uint64_t req_key)
{
	mr->key = req_key;
	return FI_SUCCESS;
}

static int cxip_prov_init_mr_key(struct cxip_mr *mr, uint64_t req_key)
{
	return cxip_init_mr_key(mr, mr->req.req_id);
}

static int cxip_prov_cache_init_mr_key(struct cxip_mr *mr,
				       uint64_t req_key)
{
	struct cxip_mr_key key = {};
	struct cxi_md *md = mr->md->md;

	key.lac = mr->len ? md->lac : 0;
	key.lac_off = mr->len ? CXI_VA_TO_IOVA(md, mr->buf) : 0;
	key.opt = cxip_env.optimized_mrs &&
			mr->req.req_id < CXIP_PTL_IDX_MR_OPT_CNT;
	mr->key = key.raw;

	return FI_SUCCESS;
}

static bool cxip_is_valid_mr_key(uint64_t key)
{
	if (key & ~CXIP_MR_KEY_MASK)
		return false;
	return true;
}

static bool cxip_is_valid_prov_mr_key(uint64_t key)
{
	return cxip_is_valid_mr_key(key);
}

static bool cxip_is_valid_prov_cache_mr_key(uint64_t key)
{
	/* TODO */
	return false;
}

static bool cxip_mr_key_opt(uint64_t key)
{
	return cxip_env.optimized_mrs && key < CXIP_PTL_IDX_MR_OPT_CNT;
}

static bool cxip_prov_mr_key_opt(uint64_t key)
{
	return cxip_mr_key_opt(key);
}

static bool cxip_prov_cache_mr_key_opt(uint64_t key)
{
	struct cxip_mr_key cxip_key = {
		.raw = key,
	};

	if (cxip_env.optimized_mrs && cxip_key.opt)
		return true;
	return false;
}

static int cxip_mr_key_to_ptl_idx(struct cxip_domain *dom,
				  uint64_t key, bool write)
{
	if (dom->mr_util->key_is_opt(key))
		return write ? CXIP_PTL_IDX_WRITE_MR_OPT(key) :
			CXIP_PTL_IDX_READ_MR_OPT(key);
	return write ? CXIP_PTL_IDX_WRITE_MR_STD : CXIP_PTL_IDX_READ_MR_STD;
}

static int cxip_prov_mr_key_to_ptl_idx(struct cxip_domain *dom,
				       uint64_t key, bool write)
{
	return cxip_mr_key_to_ptl_idx(dom, key, write);
}

static int cxip_prov_cache_mr_key_to_ptl_idx(struct cxip_domain *dom,
					     uint64_t key, bool write)
{
	struct cxip_mr_key cxip_key = {
		.raw = key,
	};

	/* TODO */
	if (dom->mr_util->key_is_opt(key))
		return CXIP_PTL_IDX_WRITE_MR_OPT(cxip_key.lac);

	return CXIP_PTL_IDX_WRITE_MR_STD;
}

struct cxip_mr_util_ops cxip_client_key_mr_util_ops = {
	.is_prov = false,
	.is_cached = false,
	.init_key = cxip_init_mr_key,
	.key_is_valid = cxip_is_valid_mr_key,
	.key_is_opt = cxip_mr_key_opt,
	.key_to_ptl_idx = cxip_mr_key_to_ptl_idx,
	.domain_insert = cxip_mr_domain_insert,
	.domain_remove = cxip_mr_domain_remove,
	.enable_opt = cxip_mr_enable_opt,
	.disable_opt = cxip_mr_disable_opt,
	.enable_std = cxip_mr_enable_std,
	.disable_std = cxip_mr_disable_std,
};

struct cxip_mr_util_ops cxip_prov_key_mr_util_ops = {
	.is_prov = true,
	.is_cached = false,
	.init_key = cxip_prov_init_mr_key,
	.key_is_valid = cxip_is_valid_prov_mr_key,
	.key_is_opt = cxip_prov_mr_key_opt,
	.key_to_ptl_idx = cxip_prov_mr_key_to_ptl_idx,
	.domain_insert = cxip_mr_domain_insert_prov,
	.domain_remove = cxip_mr_domain_remove_prov,
	.enable_opt = cxip_mr_enable_opt,
	.disable_opt = cxip_mr_disable_opt,
	.enable_std = cxip_mr_enable_std,
	.disable_std = cxip_mr_disable_std,
};

struct cxip_mr_util_ops cxip_prov_key_cache_mr_util_ops = {
	.is_prov = true,
	.is_cached = true,
	.init_key = cxip_prov_cache_init_mr_key,
	.key_is_valid = cxip_is_valid_prov_cache_mr_key,
	.key_is_opt = cxip_prov_cache_mr_key_opt,
	.key_to_ptl_idx = cxip_prov_cache_mr_key_to_ptl_idx,
	.domain_insert = cxip_mr_domain_insert_prov,
	.domain_remove = cxip_mr_domain_remove_prov,
	.enable_opt = cxip_mr_prov_cache_enable_opt,
	.disable_opt = cxip_mr_prov_cache_disable_opt,
	.enable_std = cxip_mr_prov_cache_enable_std,
	.disable_std = cxip_mr_prov_cache_disable_std,
};
