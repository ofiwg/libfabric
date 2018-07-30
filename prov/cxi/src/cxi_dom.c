/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include <ofi_util.h>

#include "cxi_prov.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_DOMAIN, __VA_ARGS__)

extern struct fi_ops_mr cxi_dom_mr_ops;

/* TODO define */
struct fi_ops_mr cxi_dom_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = fi_no_mr_reg,
	.regv = fi_no_mr_regv,
	.regattr = fi_no_mr_regattr
};

const struct fi_domain_attr cxi_domain_attr = {
	.name = NULL,
	.threading = FI_THREAD_SAFE,
	.control_progress = FI_PROGRESS_AUTO,
	.data_progress = FI_PROGRESS_AUTO,
	.resource_mgmt = FI_RM_ENABLED,
	.mr_mode = FI_MR_SCALABLE,
	.mr_key_size = sizeof(uint64_t),
	.cq_data_size = sizeof(uint64_t),
	.cq_cnt = CXI_EP_MAX_CQ_CNT,
	.ep_cnt = CXI_EP_MAX_EP_CNT,
	.tx_ctx_cnt = CXI_EP_MAX_TX_CNT,
	.rx_ctx_cnt = CXI_EP_MAX_RX_CNT,
	.max_ep_tx_ctx = CXI_EP_MAX_TX_CNT,
	.max_ep_rx_ctx = CXI_EP_MAX_RX_CNT,
	.max_ep_stx_ctx = CXI_EP_MAX_EP_CNT,
	.max_ep_srx_ctx = CXI_EP_MAX_EP_CNT,
	.cntr_cnt = CXI_EP_MAX_CNTR_CNT,
	.mr_iov_limit = CXI_EP_MAX_IOV_LIMIT,
	.max_err_data = CXI_MAX_ERR_CQ_EQ_DATA_SZ,
	.mr_cnt = CXI_DOMAIN_MR_CNT,
};

int cxi_verify_domain_attr(uint32_t version, const struct fi_info *info)
{
	const struct fi_domain_attr *attr = info->domain_attr;

	if (!attr)
		return 0;

	switch (attr->threading) {
	case FI_THREAD_UNSPEC:
	case FI_THREAD_SAFE:
	case FI_THREAD_FID:
	case FI_THREAD_DOMAIN:
	case FI_THREAD_COMPLETION:
	case FI_THREAD_ENDPOINT:
		break;
	default:
		CXI_LOG_DBG("Invalid threading model!\n");
		return -FI_ENODATA;
	}

	switch (attr->control_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;

	default:
		CXI_LOG_DBG("Control progress mode not supported!\n");
		return -FI_ENODATA;
	}

	switch (attr->data_progress) {
	case FI_PROGRESS_UNSPEC:
	case FI_PROGRESS_AUTO:
	case FI_PROGRESS_MANUAL:
		break;

	default:
		CXI_LOG_DBG("Data progress mode not supported!\n");
		return -FI_ENODATA;
	}

	switch (attr->resource_mgmt) {
	case FI_RM_UNSPEC:
	case FI_RM_DISABLED:
	case FI_RM_ENABLED:
		break;

	default:
		CXI_LOG_DBG("Resource mgmt not supported!\n");
		return -FI_ENODATA;
	}

	switch (attr->av_type) {
	case FI_AV_UNSPEC:
	case FI_AV_MAP:
	case FI_AV_TABLE:
		break;

	default:
		CXI_LOG_DBG("AV type not supported!\n");
		return -FI_ENODATA;
	}

	if (ofi_check_mr_mode(&cxi_prov, version,
			      cxi_domain_attr.mr_mode, info)) {
		FI_INFO(&cxi_prov, FI_LOG_CORE,
			"Invalid memory registration mode\n");
		return -FI_ENODATA;
	}

	if (attr->mr_key_size > cxi_domain_attr.mr_key_size)
		return -FI_ENODATA;

	if (attr->cq_data_size > cxi_domain_attr.cq_data_size)
		return -FI_ENODATA;

	if (attr->cq_cnt > cxi_domain_attr.cq_cnt)
		return -FI_ENODATA;

	if (attr->ep_cnt > cxi_domain_attr.ep_cnt)
		return -FI_ENODATA;

	if (attr->max_ep_tx_ctx > cxi_domain_attr.max_ep_tx_ctx)
		return -FI_ENODATA;

	if (attr->max_ep_rx_ctx > cxi_domain_attr.max_ep_rx_ctx)
		return -FI_ENODATA;

	if (attr->cntr_cnt > cxi_domain_attr.cntr_cnt)
		return -FI_ENODATA;

	if (attr->mr_iov_limit > cxi_domain_attr.mr_iov_limit)
		return -FI_ENODATA;

	if (attr->max_err_data > cxi_domain_attr.max_err_data)
		return -FI_ENODATA;

	if (attr->mr_cnt > cxi_domain_attr.mr_cnt)
		return -FI_ENODATA;

	return 0;
}

int cxix_domain_enable(struct cxi_domain *dom)
{
	int ret;

	fastlock_acquire(&dom->lock);

	ret = cxix_get_if(dom->nic_addr, &dom->dev_if);
	if (ret != FI_SUCCESS) {
		CXI_LOG_DBG("Unable to get IF\n");
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	dom->enabled = 1;
	fastlock_release(&dom->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&dom->lock);

	return ret;
}

static void cxix_domain_disable(struct cxi_domain *dom)
{
	fastlock_acquire(&dom->lock);

	if (!dom->enabled)
		goto unlock;

	cxix_put_if(dom->dev_if);

	dom->enabled = 0;
unlock:
	fastlock_release(&dom->lock);
}

static int cxi_dom_close(struct fid *fid)
{
	struct cxi_domain *dom;

	dom = container_of(fid, struct cxi_domain, dom_fid.fid);
	if (ofi_atomic_get32(&dom->ref))
		return -FI_EBUSY;

	cxix_domain_disable(dom);

	cxi_dom_remove_from_list(dom);
	fastlock_destroy(&dom->lock);
	free(dom);

	return 0;
}

static int cxi_dom_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxi_domain *dom;
	struct cxi_eq *eq;

	dom = container_of(fid, struct cxi_domain, dom_fid.fid);
	eq = container_of(bfid, struct cxi_eq, eq.fid);

	if (dom->eq)
		return -FI_EINVAL;

	dom->eq = eq;
	if (flags & FI_REG_MR)
		dom->mr_eq = eq;

	return 0;
}

#if 0
static int cxi_dom_ctrl(struct fid *fid, int command, void *arg)
{
	struct cxi_domain *dom;

	dom = container_of(fid, struct cxi_domain, dom_fid.fid);
	switch (command) {
	case FI_QUEUE_WORK:
		return cxi_queue_work(dom, arg);
	default:
		return -FI_ENOSYS;
	}
}
#endif

static int cxi_endpoint(struct fid_domain *domain, struct fi_info *info,
			struct fid_ep **ep, void *context)
{
	if (!info || !ep)
		return -FI_EINVAL;

	switch (info->ep_attr->type) {
	case FI_EP_RDM:
		return cxi_rdm_ep(domain, info, ep, context);
	default:
		return -FI_ENOPROTOOPT;
	}
}

static int cxi_scalable_ep(struct fid_domain *domain, struct fi_info *info,
			   struct fid_ep **sep, void *context)
{
	switch (info->ep_attr->type) {
	case FI_EP_RDM:
		return cxi_rdm_sep(domain, info, sep, context);
	default:
		return -FI_ENOPROTOOPT;
	}
}

static struct fi_ops cxi_dom_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_dom_close,
	.bind = cxi_dom_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain cxi_dom_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = cxi_av_open,
	.cq_open = cxi_cq_open,
	.endpoint = cxi_endpoint,
	.scalable_ep = cxi_scalable_ep,
	.cntr_open = fi_no_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};

int cxi_domain(struct fid_fabric *fabric, struct fi_info *info,
	       struct fid_domain **dom, void *context)
{
	struct cxi_domain *cxi_domain;
	struct cxi_fabric *fab;
	struct cxi_addr *src_addr;
	int ret;

	fab = container_of(fabric, struct cxi_fabric, fab_fid);
	if (info && info->domain_attr) {
		ret = cxi_verify_domain_attr(fabric->api_version, info);
		if (ret)
			return -FI_EINVAL;
	}

	cxi_domain = calloc(1, sizeof(*cxi_domain));
	if (!cxi_domain)
		return -FI_ENOMEM;

	fastlock_init(&cxi_domain->lock);
	ofi_atomic_initialize32(&cxi_domain->ref, 0);

	if (!info || !info->src_addr) {
		CXI_LOG_ERROR("Invalid fi_info\n");
		goto unlock;
	}

	cxi_domain->info = *info;

	cxi_domain->dom_fid.fid.fclass = FI_CLASS_DOMAIN;
	cxi_domain->dom_fid.fid.context = context;
	cxi_domain->dom_fid.fid.ops = &cxi_dom_fi_ops;
	cxi_domain->dom_fid.ops = &cxi_dom_ops;
	cxi_domain->dom_fid.mr = &cxi_dom_mr_ops;

	if (!info->domain_attr ||
	    info->domain_attr->data_progress == FI_PROGRESS_UNSPEC)
		cxi_domain->progress_mode = FI_PROGRESS_AUTO;
	else
		cxi_domain->progress_mode = info->domain_attr->data_progress;

	cxi_domain->fab = fab;

	src_addr = (struct cxi_addr *)info->src_addr;
	cxi_domain->nic_addr = src_addr->nic;
	cxi_domain->vni = 0; /* TODO set appropriately */
	cxi_domain->pid = src_addr->domain;
	cxi_domain->pid_granule = CXIX_PID_GRANULE_DEF;

	*dom = &cxi_domain->dom_fid;

	if (info->domain_attr)
		cxi_domain->attr = *(info->domain_attr);
	else
		cxi_domain->attr = cxi_domain_attr;

	cxi_dom_add_to_list(cxi_domain);
	return 0;

unlock:
	fastlock_destroy(&cxi_domain->lock);
	free(cxi_domain);
	return -FI_EINVAL;
}
