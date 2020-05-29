/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018,2020 Cray Inc. All rights reserved.
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
	int ret = FI_SUCCESS;
	int tmp;

	fastlock_acquire(&dom->lock);

	if (dom->enabled)
		goto unlock;

	ret = cxip_get_if(dom->nic_addr, &dom->iface);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Unable to get IF\n");
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	ret = cxip_alloc_lni(dom->iface, &dom->lni);
	if (ret) {
		CXIP_LOG_DBG("cxip_alloc_lni returned: %d\n", ret);
		ret = -FI_ENODEV;
		goto put_if;
	}

	/* TODO Temporary CP setup, needed for CMDQ allocation */
	ret = cxil_alloc_cp(dom->lni->lni, 0, CXI_TC_LOW_LATENCY,
			    &dom->cps[0]);
	if (ret) {
		CXIP_LOG_DBG("Unable to allocate CP, ret: %d\n", ret);
		ret = -FI_ENODEV;
		goto free_lni;
	}
	dom->n_cps++;

	ret = cxip_iomm_init(dom);
	if (ret != FI_SUCCESS) {
		CXIP_LOG_DBG("Failed to initialize IOMM: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto free_cp;
	}

	dom->enabled = true;
	fastlock_release(&dom->lock);

	CXIP_LOG_DBG("Allocated interface, NIC[%u]: %u RGID: %u\n",
		     dom->iface->info.dev_id,
		     dom->iface->info.nic_addr,
		     dom->lni->lni->id);

	return FI_SUCCESS;

free_cp:
	tmp = cxil_destroy_cp(dom->cps[0]);
	if (tmp)
		CXIP_LOG_ERROR("Failed to destroy CP: %d\n", tmp);
free_lni:
	cxip_free_lni(dom->lni);
	dom->lni = NULL;
put_if:
	cxip_put_if(dom->iface);
	dom->iface = NULL;
unlock:
	fastlock_release(&dom->lock);

	return ret;
}

/*
 * cxip_domain_disable() - Disable an FI Domain.
 */
static void cxip_domain_disable(struct cxip_domain *dom)
{
	int ret;

	fastlock_acquire(&dom->lock);

	if (!dom->enabled)
		goto unlock;

	cxip_dom_cntr_disable(dom);

	cxip_iomm_fini(dom);

	ret = cxil_destroy_cp(dom->cps[0]);
	if (ret)
		CXIP_LOG_ERROR("Failed to destroy CP: %d\n", ret);

	CXIP_LOG_DBG("Releasing interface, NIC[%u]: %u RGID: %u\n",
		     dom->iface->info.dev_id,
		     dom->iface->info.nic_addr,
		     dom->lni->lni->id);

	cxip_free_lni(dom->lni);

	cxip_put_if(dom->iface);

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

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid.fid);
	eq = container_of(bfid, struct cxip_eq, util_eq.eq_fid.fid);

	if (dom->eq)
		return -FI_EINVAL;

	dom->eq = eq;
	if (flags & FI_REG_MR)
		dom->mr_eq = eq;

	return 0;
}

static int cxip_dom_control(struct fid *fid, int command, void *arg)
{
	struct cxip_domain *dom;
	struct fi_deferred_work *work;
	struct fi_op_msg *msg;
	struct cxip_txc *txc;
	struct cxip_cntr *trig_cntr;
	struct cxip_cntr *comp_cntr;
	const void *buf;
	size_t len;
	int ret;

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid.fid);

	if (command == FI_QUEUE_WORK) {
		work = arg;

		if (!work->triggering_cntr)
			return -FI_EINVAL;

		comp_cntr = work->completion_cntr ?
			container_of(work->completion_cntr,
				     struct cxip_cntr, cntr_fid) : NULL;
		trig_cntr = container_of(work->triggering_cntr,
					 struct cxip_cntr, cntr_fid);

		if (work->op_type == FI_OP_SEND) {
			msg = work->op.msg;

			if (msg->msg.iov_count > 1)
				return -FI_EINVAL;

			ret = cxip_fid_to_txc(msg->ep, &txc);
			if (ret)
				return ret;

			buf = msg->msg.iov_count ?
				msg->msg.msg_iov[0].iov_base : NULL;
			len = msg->msg.iov_count ?
				msg->msg.msg_iov[0].iov_len : 0;

			ret = cxip_dom_cntr_enable(dom);
			if (ret) {
				CXIP_LOG_DBG("Failed to enable domain for counters, ret=%d\n",
					     ret);
				return ret;
			}

			ret = cxip_send_common(txc, buf, len, NULL,
					       msg->msg.data, msg->msg.addr,
					       0, msg->msg.context, msg->flags,
					       false, true, work->threshold,
					       trig_cntr, comp_cntr);
			if (ret)
				CXIP_LOG_DBG("Failed to emit message triggered op, ret=%d\n",
					     ret);
			else
				CXIP_LOG_DBG("Queued triggered message operation with threshold %lu",
					     work->threshold);


			return ret;
		}
	}

	return -FI_EINVAL;
}

static struct fi_ops cxip_dom_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_dom_close,
	.bind = cxip_dom_bind,
	.control = cxip_dom_control,
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

	if (cxip_env.odp)
		cxi_domain->odp = true;

	if (cxip_env.ats)
		cxi_domain->ats_init = true;

	cxi_domain->util_domain.domain_fid.fid.ops = &cxip_dom_fi_ops;
	cxi_domain->util_domain.domain_fid.ops = &cxip_dom_ops;
	cxi_domain->util_domain.domain_fid.mr = &cxip_dom_mr_ops;

	dlist_init(&cxi_domain->txc_list);
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
