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

static int cxip_dom_dwq_op_send(struct cxip_domain *dom, struct fi_op_msg *msg,
				struct cxip_cntr *trig_cntr,
				struct cxip_cntr *comp_cntr,
				uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	const void *buf;
	size_t len;
	int ret;

	if (!msg || msg->msg.iov_count > 1)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(msg->ep, &txc);
	if (ret)
		return ret;

	buf = msg->msg.iov_count ? msg->msg.msg_iov[0].iov_base : NULL;
	len = msg->msg.iov_count ? msg->msg.msg_iov[0].iov_len : 0;

	ret = cxip_dom_cntr_enable(dom);
	if (ret) {
		CXIP_LOG_DBG("Failed to enable domain for counters, ret=%d\n",
			     ret);
		return ret;
	}

	ret = cxip_send_common(txc, buf, len, NULL, msg->msg.data,
			       msg->msg.addr, 0, msg->msg.context, msg->flags,
			       false, true, trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_LOG_DBG("Failed to emit message triggered op, ret=%d\n",
			     ret);
	else
		CXIP_LOG_DBG("Queued triggered message operation with threshold %lu",
			     trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_tsend(struct cxip_domain *dom,
				 struct fi_op_tagged *tagged,
				 struct cxip_cntr *trig_cntr,
				 struct cxip_cntr *comp_cntr,
				 uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	const void *buf;
	size_t len;
	int ret;

	if (!tagged || tagged->msg.iov_count > 1)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(tagged->ep, &txc);
	if (ret)
		return ret;

	buf = tagged->msg.iov_count ? tagged->msg.msg_iov[0].iov_base : NULL;
	len = tagged->msg.iov_count ? tagged->msg.msg_iov[0].iov_len : 0;

	ret = cxip_dom_cntr_enable(dom);
	if (ret) {
		CXIP_LOG_DBG("Failed to enable domain for counters, ret=%d\n",
			     ret);
		return ret;
	}

	ret = cxip_send_common(txc, buf, len, NULL, tagged->msg.data,
			       tagged->msg.addr, tagged->msg.tag,
			       tagged->msg.context, tagged->flags, true, true,
			       trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_LOG_DBG("Failed to emit tagged message triggered op, ret=%d\n",
			     ret);
	else
		CXIP_LOG_DBG("Queued triggered tagged message operation with threshold %lu",
			     trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_rma(struct cxip_domain *dom, struct fi_op_rma *rma,
			       enum fi_op_type op, struct cxip_cntr *trig_cntr,
			       struct cxip_cntr *comp_cntr,
			       uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	const void *buf;
	size_t len;
	int ret;

	if (!rma || !rma->msg.msg_iov || rma->msg.iov_count > 1 ||
	    !rma->msg.rma_iov || rma->msg.rma_iov_count != 1)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(rma->ep, &txc);
	if (ret)
		return ret;

	buf = rma->msg.iov_count ? rma->msg.msg_iov[0].iov_base : NULL;
	len = rma->msg.iov_count ? rma->msg.msg_iov[0].iov_len : 0;

	ret = cxip_dom_cntr_enable(dom);
	if (ret) {
		CXIP_LOG_DBG("Failed to enable domain for counters, ret=%d\n",
			     ret);
		return ret;
	}

	ret = cxip_rma_common(op, txc, buf, len, NULL, rma->msg.addr,
			      rma->msg.rma_iov[0].addr, rma->msg.rma_iov[0].key,
			      rma->msg.data, rma->flags, rma->msg.context, true,
			      trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_LOG_DBG("Failed to emit RMA triggered op, ret=%d\n",
			     ret);
	else
		CXIP_LOG_DBG("Queued triggered RMA operation with threshold %lu",
			     trig_thresh);

	return ret;
}

/* Must hold domain lock. */
static void cxip_dom_progress_all_cqs(struct cxip_domain *dom)
{
	struct cxip_cq *cq;

	dlist_foreach_container(&dom->cq_list, struct cxip_cq, cq,
				dom_entry)
		cxip_cq_progress(cq);
}

static int cxip_dom_control(struct fid *fid, int command, void *arg)
{
	struct cxip_domain *dom;
	struct cxip_txc *txc;
	struct fi_deferred_work *work;
	struct cxip_cntr *trig_cntr;
	struct cxip_cntr *comp_cntr;
	int ret;

	dom = container_of(fid, struct cxip_domain, util_domain.domain_fid.fid);

	switch (command) {
	case FI_QUEUE_WORK:
		work = arg;

		if (!work->triggering_cntr)
			return -FI_EINVAL;

		comp_cntr = work->completion_cntr ?
			container_of(work->completion_cntr,
				     struct cxip_cntr, cntr_fid) : NULL;
		trig_cntr = container_of(work->triggering_cntr,
					 struct cxip_cntr, cntr_fid);

		switch (work->op_type) {
		case FI_OP_SEND:
			return cxip_dom_dwq_op_send(dom, work->op.msg,
						    trig_cntr, comp_cntr,
						    work->threshold);

		case FI_OP_TSEND:
			return cxip_dom_dwq_op_tsend(dom, work->op.tagged,
						     trig_cntr, comp_cntr,
						     work->threshold);

		case FI_OP_READ:
		case FI_OP_WRITE:
			return cxip_dom_dwq_op_rma(dom, work->op.rma,
						   work->op_type, trig_cntr,
						   comp_cntr, work->threshold);

		default:
			CXIP_LOG_ERROR("Invalid FI_QUEUE_WORK op %s\n",
				       fi_tostr(&work->op_type,
						FI_TYPE_OP_TYPE));
			return -FI_EINVAL;
		}

	case FI_FLUSH_WORK:
		fastlock_acquire(&dom->lock);
		if (!dom->cntr_init) {
			fastlock_release(&dom->lock);
			return FI_SUCCESS;
		}

		fastlock_acquire(&dom->trig_cmdq->lock);

		/* Issue cancels to all allocated counters. */
		dlist_foreach_container(&dom->cntr_list, struct cxip_cntr,
					trig_cntr, dom_entry) {
			struct c_ct_cmd ct_cmd = {};

			if (!trig_cntr->ct)
				continue;

			ct_cmd.ct = trig_cntr->ct->ctn;
			ret = cxi_cq_emit_ct(dom->trig_cmdq->dev_cmdq,
					     C_CMD_CT_CANCEL, &ct_cmd);

			// TODO: Handle this assert. Multiple triggered CQs may
			// be required.
			assert(!ret);
			cxi_cq_ring(dom->trig_cmdq->dev_cmdq);
		};

		/* Rely on the triggered CQ ack counter to know when there are
		 * no more pending triggered operations. In-between, progress
		 * CQs to cleanup internal transaction state.
		 */
		while (true) {
			unsigned int ack_counter;

			ret = cxil_cmdq_ack_counter(dom->trig_cmdq->dev_cmdq,
						    &ack_counter);
			assert(!ret);

			if (!ack_counter)
				break;

			cxip_dom_progress_all_cqs(dom);
		}

		/* It is possible that the ack counter is zero and there are
		 * completion events in-flight meaning that the above
		 * progression may have missed events. Perform a sleep to help
		 * ensure events have arrived and progress all CQs one more
		 * time.
		 *
		 * TODO: Investigate better way to resolve this race condition.
		 */
		sleep(1);
		cxip_dom_progress_all_cqs(dom);

		/* At this point, all triggered operations should be cancelled
		 * or have completed. Due to special handling of message
		 * operations, flush any remaining message triggered requests
		 * from the TX context first.
		 */
		dlist_foreach_container(&dom->txc_list, struct cxip_txc, txc,
					dom_entry)
			cxip_txc_flush_msg_trig_reqs(txc);

		fastlock_release(&dom->trig_cmdq->lock);
		fastlock_release(&dom->lock);

		return FI_SUCCESS;
	default:
		return -FI_EINVAL;
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
	dlist_init(&cxi_domain->cntr_list);
	dlist_init(&cxi_domain->cq_list);
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
