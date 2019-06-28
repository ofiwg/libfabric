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

#include "cxip.h"

#define CXIP_LOG_DBG(...) _CXIP_LOG_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_LOG_ERROR(...) _CXIP_LOG_ERROR(FI_LOG_DOMAIN, __VA_ARGS__)

extern struct fi_ops_mr cxip_dom_mr_ops;

/*
 * cxip_domain_enable() - Enable an FI Domain for use.
 *
 * Allocate hardware resources and initialize software to prepare the Domain
 * for use.
 */
int cxip_domain_enable(struct cxip_domain *dom)
{
	struct cxi_cq_alloc_opts cq_opts = {};
	int ret = FI_SUCCESS;

	fastlock_acquire(&dom->lock);

	if (dom->enabled)
		goto unlock;

	ret = cxip_get_if(dom->nic_addr, &dom->dev_if);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to get IF\n");
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	ret = cxip_iomm_init(dom);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to initialize IOMM: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto put_if;
	}

	cq_opts.count = 64;
	cq_opts.is_transmit = 1;
	cq_opts.with_trig_cmds = 1;

	ret = cxip_cmdq_alloc(dom->dev_if, NULL, &cq_opts, &dom->trig_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to allocate trig_cmdq: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto iomm_fini;
	}

	dom->enabled = true;
	fastlock_release(&dom->lock);

	return FI_SUCCESS;

iomm_fini:
	cxip_iomm_fini(dom);
put_if:
	cxip_put_if(dom->dev_if);
	dom->dev_if = NULL;
unlock:
	fastlock_release(&dom->lock);

	return ret;
}

/*
 * cxip_domain_disable() - Disable an FI Domain.
 */
static void cxip_domain_disable(struct cxip_domain *dom)
{
	fastlock_acquire(&dom->lock);

	if (!dom->enabled)
		goto unlock;

	cxip_cmdq_free(dom->trig_cmdq);

	cxip_iomm_fini(dom);

	cxip_put_if(dom->dev_if);

	dom->enabled = false;
unlock:
	fastlock_release(&dom->lock);
}

/*
 * cxip_dom_close() - Provider fi_close implementation for an FI Domain object.
 */
static int cxip_dom_close(struct fid *fid)
{
	struct cxip_domain *dom;

	dom = container_of(fid, struct cxip_domain,
			   util_domain.domain_fid.fid);
	if (ofi_atomic_get32(&dom->ref))
		return -FI_EBUSY;

	cxip_domain_disable(dom);

	fastlock_destroy(&dom->lock);
	ofi_domain_close(&dom->util_domain);
	free(dom);

	return 0;
}

/*
 * cxip_dom_bind() - Provider fi_domain_bind implementation.
 */
static int cxip_dom_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct cxip_domain *dom;
	struct cxip_eq *eq;

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid);
	eq = container_of(bfid, struct cxip_eq, eq.fid);

	if (dom->eq)
		return -FI_EINVAL;

	dom->eq = eq;
	if (flags & FI_REG_MR)
		dom->mr_eq = eq;

	return 0;
}

static struct fi_ops cxip_dom_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_dom_close,
	.bind = cxip_dom_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_domain cxip_dom_ops = {
	.size = sizeof(struct fi_ops_domain),
	.av_open = cxip_av_open,
	.cq_open = cxip_cq_open,
	.endpoint = cxip_endpoint,
	.scalable_ep = cxip_scalable_ep,
	.cntr_open = cxip_cntr_open,
	.poll_open = fi_no_poll_open,
	.stx_ctx = fi_no_stx_context,
	.srx_ctx = fi_no_srx_context,
	.query_atomic = fi_no_query_atomic,
};

/*
 * cxip_domain() - Provider fi_domain() implementation.
 */
int cxip_domain(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context)
{
	struct cxip_domain *cxi_domain;
	struct cxip_fabric *fab;
	struct cxip_addr *src_addr;
	int ret;

	ret = ofi_prov_check_info(&cxip_util_prov, CXIP_FI_VERSION, info);
	if (ret != FI_SUCCESS)
		return -FI_ENOPROTOOPT;

	fab = container_of(fabric, struct cxip_fabric, util_fabric.fabric_fid);

	cxi_domain = calloc(1, sizeof(*cxi_domain));
	if (!cxi_domain)
		return -FI_ENOMEM;

	ret = ofi_domain_init(&fab->util_fabric.fabric_fid, info,
			      &cxi_domain->util_domain, context);
	if (ret)
		goto unlock;

	if (!info || !info->src_addr) {
		CXIP_LOG_ERROR("Invalid fi_info\n");
		goto free_util_dom;
	}
	src_addr = (struct cxip_addr *)info->src_addr;
	cxi_domain->nic_addr = src_addr->nic;

	cxi_domain->util_domain.domain_fid.fid.ops = &cxip_dom_fi_ops;
	cxi_domain->util_domain.domain_fid.ops = &cxip_dom_ops;
	cxi_domain->util_domain.domain_fid.mr = &cxip_dom_mr_ops;

	fastlock_init(&cxi_domain->lock);
	ofi_atomic_initialize32(&cxi_domain->ref, 0);
	cxi_domain->fab = fab;

	*dom = &cxi_domain->util_domain.domain_fid;

	return 0;

free_util_dom:
	ofi_domain_close(&cxi_domain->util_domain);
unlock:
	fastlock_destroy(&cxi_domain->lock);
	free(cxi_domain);
	return -FI_EINVAL;
}
