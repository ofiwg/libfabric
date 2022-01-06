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

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_DOMAIN, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_DOMAIN, __VA_ARGS__)

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

	fastlock_acquire(&dom->lock);

	if (dom->enabled)
		goto unlock;

	ret = cxip_get_if(dom->nic_addr, &dom->iface);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Unable to get IF\n");
		ret = -FI_ENODEV;
		goto unlock;
	}

	ret = cxip_alloc_lni(dom->iface, dom->auth_key.svc_id, &dom->lni);
	if (ret) {
		CXIP_WARN("cxip_alloc_lni returned: %d\n", ret);
		ret = -FI_ENODEV;
		goto put_if;
	}

	ret = cxip_iomm_init(dom);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to initialize IOMM: %d\n", ret);
		assert(ret == -FI_ENOMEM);
		goto free_lni;
	}

	dom->enabled = true;
	fastlock_release(&dom->lock);

	CXIP_DBG("Allocated interface, %s: %u RGID: %u\n",
		 dom->iface->info->device_name,
		 dom->iface->info->nic_addr,
		 dom->lni->lni->id);

	return FI_SUCCESS;

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
	fastlock_acquire(&dom->lock);

	if (!dom->enabled)
		goto unlock;

	cxip_dom_cntr_disable(dom);

	cxip_iomm_fini(dom);

	CXIP_DBG("Releasing interface, %s: %u RGID: %u\n",
		 dom->iface->info->device_name,
		 dom->iface->info->nic_addr,
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

	ret = cxip_send_common(txc, buf, len, NULL, msg->msg.data,
			       msg->msg.addr, 0, msg->msg.context, msg->flags,
			       false, true, trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit message triggered op, ret=%d\n", ret);
	else
		CXIP_DBG("Queued triggered message operation with threshold %lu\n",
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

	ret = cxip_send_common(txc, buf, len, NULL, tagged->msg.data,
			       tagged->msg.addr, tagged->msg.tag,
			       tagged->msg.context, tagged->flags, true, true,
			       trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit tagged message triggered op, ret=%d\n",
			 ret);
	else
		CXIP_DBG("Queued triggered tagged message operation with threshold %lu\n",
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

	ret = cxip_rma_common(op, txc, buf, len, NULL, rma->msg.addr,
			      rma->msg.rma_iov[0].addr, rma->msg.rma_iov[0].key,
			      rma->msg.data, rma->flags, rma->msg.context, true,
			      trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit RMA triggered op, ret=%d\n", ret);
	else
		CXIP_DBG("Queued triggered RMA operation with threshold %lu\n",
			 trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_atomic(struct cxip_domain *dom,
				  struct fi_op_atomic *amo,
				  struct cxip_cntr *trig_cntr,
				  struct cxip_cntr *comp_cntr,
				  uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	int ret;

	if (!amo)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(amo->ep, &txc);
	if (ret)
		return ret;

	ret = cxip_amo_common(CXIP_RQ_AMO, txc, &amo->msg, NULL, NULL, 0, NULL,
			      NULL, 0, amo->flags, true, trig_thresh, trig_cntr,
			      comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit AMO triggered op, ret=%d\n", ret);
	else
		CXIP_DBG("Queued triggered AMO operation with threshold %lu\n",
			 trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_fetch_atomic(struct cxip_domain *dom,
					struct fi_op_fetch_atomic *fetch_amo,
					struct cxip_cntr *trig_cntr,
					struct cxip_cntr *comp_cntr,
					uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	int ret;

	if (!fetch_amo)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(fetch_amo->ep, &txc);
	if (ret)
		return ret;

	ret = cxip_amo_common(CXIP_RQ_AMO_FETCH, txc, &fetch_amo->msg, NULL,
			      NULL, 0, fetch_amo->fetch.msg_iov,
			      fetch_amo->fetch.desc, fetch_amo->fetch.iov_count,
			      fetch_amo->flags, true, trig_thresh, trig_cntr,
			      comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit fetching AMO triggered op, ret=%d\n",
			 ret);
	else
		CXIP_DBG("Queued triggered fetching AMO operation with threshold %lu\n",
			 trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_comp_atomic(struct cxip_domain *dom,
				       struct fi_op_compare_atomic *comp_amo,
				       struct cxip_cntr *trig_cntr,
				       struct cxip_cntr *comp_cntr,
				       uint64_t trig_thresh)
{
	struct cxip_txc *txc;
	int ret;

	if (!comp_amo)
		return -FI_EINVAL;

	ret = cxip_fid_to_txc(comp_amo->ep, &txc);
	if (ret)
		return ret;

	ret = cxip_amo_common(CXIP_RQ_AMO_SWAP, txc, &comp_amo->msg,
			      comp_amo->compare.msg_iov, comp_amo->compare.desc,
			      comp_amo->compare.iov_count,
			      comp_amo->fetch.msg_iov, comp_amo->fetch.desc,
			      comp_amo->fetch.iov_count, comp_amo->flags, true,
			      trig_thresh, trig_cntr, comp_cntr);
	if (ret)
		CXIP_DBG("Failed to emit compare AMO triggered op, ret=%d\n",
			 ret);
	else
		CXIP_DBG("Queued triggered compare AMO operation with threshold %lu\n",
			 trig_thresh);

	return ret;
}

static int cxip_dom_dwq_op_cntr(struct cxip_domain *dom,
				struct fi_op_cntr *cntr, enum fi_op_type op,
				struct cxip_cntr *trig_cntr,
				struct cxip_cntr *comp_cntr,
				uint64_t trig_thresh,
				bool cntr_wb)
{
	struct cxip_cntr *op_cntr;
	int ret;
	unsigned opcode;
	struct c_ct_cmd cmd = {};

	/* Completion counter must be NULL. */
	if (!cntr || !cntr->cntr || comp_cntr)
		return -FI_EINVAL;

	if (cntr_wb) {
		opcode = C_CMD_CT_TRIG_EVENT;
		cmd.eq = C_EQ_NONE;
	} else {
		opcode = op == FI_OP_CNTR_SET ?
			C_CMD_CT_TRIG_SET : C_CMD_CT_TRIG_INC;
	}

	op_cntr = container_of(cntr->cntr, struct cxip_cntr, cntr_fid);

	cmd.trig_ct = trig_cntr->ct->ctn;
	cmd.threshold = trig_thresh;
	cmd.ct = op_cntr->ct->ctn;
	cmd.set_ct_success = 1;
	cmd.ct_success = cntr->value;

	fastlock_acquire(&dom->trig_cmdq->lock);
	ret = cxi_cq_emit_ct(dom->trig_cmdq->dev_cmdq, opcode, &cmd);
	if (ret) {
		/* TODO: Handle this assert. */
		assert(!ret);
	}
	cxi_cq_ring(dom->trig_cmdq->dev_cmdq);
	fastlock_release(&dom->trig_cmdq->lock);

	return FI_SUCCESS;
}

static int cxip_dom_dwq_op_recv(struct cxip_domain *dom, struct fi_op_msg *msg,
				struct cxip_cntr *trig_cntr,
				struct cxip_cntr *comp_cntr,
				uint64_t trig_thresh)
{
	struct cxip_rxc *rxc;
	void *buf;
	size_t len;
	int ret;

	/* Non-zero thresholds for triggered receives are not supported. */
	if (!msg || msg->msg.iov_count > 1 || trig_thresh)
		return -FI_EINVAL;

	ret = cxip_fid_to_rxc(msg->ep, &rxc);
	if (ret)
		return ret;

	buf = msg->msg.iov_count ? msg->msg.msg_iov[0].iov_base : NULL;
	len = msg->msg.iov_count ? msg->msg.msg_iov[0].iov_len : 0;

	return cxip_recv_common(rxc, buf, len, NULL, msg->msg.addr, 0, 0,
				msg->msg.context, msg->flags, false, comp_cntr);
}

static int cxip_dom_dwq_op_trecv(struct cxip_domain *dom,
				 struct fi_op_tagged *tagged,
				 struct cxip_cntr *trig_cntr,
				 struct cxip_cntr *comp_cntr,
				 uint64_t trig_thresh)
{
	struct cxip_rxc *rxc;
	void *buf;
	size_t len;
	int ret;

	/* Non-zero thresholds for triggered receives are not supported. */
	if (!tagged || tagged->msg.iov_count > 1 || trig_thresh)
		return -FI_EINVAL;

	ret = cxip_fid_to_rxc(tagged->ep, &rxc);
	if (ret)
		return ret;

	buf = tagged->msg.iov_count ? tagged->msg.msg_iov[0].iov_base : NULL;
	len = tagged->msg.iov_count ? tagged->msg.msg_iov[0].iov_len : 0;

	return cxip_recv_common(rxc, buf, len, tagged->msg.desc,
				tagged->msg.addr, tagged->msg.tag,
				tagged->msg.ignore, tagged->msg.context,
				tagged->flags, true, comp_cntr);
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
	struct cxip_cq *cq;
	bool queue_wb_work = false;
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
			if (work->op.msg->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_send(dom, work->op.msg,
						   trig_cntr, comp_cntr,
						   work->threshold);
			break;

		case FI_OP_TSEND:
			if (work->op.tagged->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_tsend(dom, work->op.tagged,
						    trig_cntr, comp_cntr,
						    work->threshold);
			break;

		case FI_OP_RECV:
			if (work->op.msg->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_recv(dom, work->op.msg,
						   trig_cntr, comp_cntr,
						   work->threshold);
			break;

		case FI_OP_TRECV:
			if (work->op.tagged->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_trecv(dom, work->op.tagged,
						    trig_cntr, comp_cntr,
						    work->threshold);
			break;

		case FI_OP_READ:
		case FI_OP_WRITE:
			if (work->op.rma->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_rma(dom, work->op.rma,
						  work->op_type, trig_cntr,
						  comp_cntr, work->threshold);
			break;

		case FI_OP_ATOMIC:
			if (work->op.atomic->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_atomic(dom, work->op.atomic,
						     trig_cntr, comp_cntr,
						     work->threshold);
			break;

		case FI_OP_FETCH_ATOMIC:
			if (work->op.fetch_atomic->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_fetch_atomic(dom,
							   work->op.fetch_atomic,
							   trig_cntr,
							   comp_cntr,
							   work->threshold);
			break;

		case FI_OP_COMPARE_ATOMIC:
			if (work->op.compare_atomic->flags & FI_CXI_CNTR_WB)
				queue_wb_work = true;
			ret = cxip_dom_dwq_op_comp_atomic(dom,
							  work->op.compare_atomic,
							  trig_cntr, comp_cntr,
							  work->threshold);
			break;

		case FI_OP_CNTR_SET:
		case FI_OP_CNTR_ADD:
			return cxip_dom_dwq_op_cntr(dom, work->op.cntr,
						    work->op_type, trig_cntr,
						    comp_cntr, work->threshold,
						    false);

		default:
			CXIP_WARN("Invalid FI_QUEUE_WORK op %s\n",
				  fi_tostr(&work->op_type, FI_TYPE_OP_TYPE));
			return -FI_EINVAL;
		}

		if (ret)
			return ret;

		if (queue_wb_work) {
			struct fi_op_cntr op_cntr = {
				.cntr = &trig_cntr->cntr_fid,
			};

			/* no op_type needed for counter writeback */
			ret = cxip_dom_dwq_op_cntr(dom, &op_cntr, 0,
						   trig_cntr, NULL,
						   work->threshold + 1, true);
			/* TODO: If cxip_dom_dwq_op_cntr fails we need to
			 * cancel the above work queue.
			 */
		}

		return ret;

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

		/* Flush all the CQs of any remaining non-message triggered
		 * operation requests.
		 */
		dlist_foreach_container(&dom->cq_list, struct cxip_cq, cq,
				dom_entry)
			cxip_cq_flush_trig_reqs(cq);

		fastlock_release(&dom->trig_cmdq->lock);
		fastlock_release(&dom->lock);

		return FI_SUCCESS;
	default:
		return -FI_EINVAL;
	}

	return -FI_EINVAL;
}

static int cxip_domain_cntr_read(struct fid *fid, unsigned int cntr,
				 uint64_t *value, struct timespec *ts)
{
	struct cxip_domain *dom;
	int ret;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		CXIP_WARN("Invalid FID: %p\n", fid);
		return -FI_EINVAL;
	}

	dom = container_of(fid, struct cxip_domain,
			   util_domain.domain_fid.fid);

	if (!dom->enabled)
		return -FI_EOPBADSTATE;

	ret = cxil_read_cntr(dom->iface->dev, cntr, value, ts);

	return ret ? -FI_EINVAL : FI_SUCCESS;
}

static int cxip_domain_topology(struct fid *fid, unsigned int *group_id,
				unsigned int *switch_id, unsigned int *port_id)
{
	struct cxip_domain *dom;
	struct cxip_topo_addr topo;

	if (fid->fclass != FI_CLASS_DOMAIN) {
		CXIP_WARN("Invalid FID: %p\n", fid);
		return -FI_EINVAL;
	}

	dom = container_of(fid, struct cxip_domain,
			   util_domain.domain_fid.fid);
	topo.addr = dom->nic_addr;

	/* Only a dragonfly topology is supported at this time */
	if (group_id)
		*group_id = topo.dragonfly.group_num;
	if (switch_id)
		*switch_id = topo.dragonfly.switch_num;
	if (port_id)
		*port_id = topo.dragonfly.port_num;

	return FI_SUCCESS;
}

static struct fi_cxi_dom_ops cxip_dom_ops_ext = {
	.cntr_read = cxip_domain_cntr_read,
	.topology = cxip_domain_topology,
};

static int cxip_dom_ops_open(struct fid *fid, const char *ops_name,
			     uint64_t flags, void **ops, void *context)
{
	/* v2 only appended a new function */
	if (!strcmp(ops_name, FI_CXI_DOM_OPS_1) ||
	    !strcmp(ops_name, FI_CXI_DOM_OPS_2)) {
		*ops = &cxip_dom_ops_ext;
		return FI_SUCCESS;
	}

	return -FI_EINVAL;
}

static int cxip_domain_ops_set(struct fid *fid, const char *name,
			       uint64_t flags, void *ops, void *context)
{
	struct cxip_domain *domain =
		container_of(fid, struct cxip_domain,
			     util_domain.domain_fid.fid);
	struct fi_hmem_override_ops *hmem_ops;

	if (strcmp(FI_SET_OPS_HMEM_OVERRIDE, name) == 0) {
		hmem_ops = ops;

		if (!hmem_ops->copy_from_hmem_iov ||
		    !hmem_ops->copy_to_hmem_iov)
			return -FI_EINVAL;

		domain->hmem_ops = *hmem_ops;

		return FI_SUCCESS;
	}

	return -FI_ENOSYS;
}

static struct fi_ops cxip_dom_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_dom_close,
	.bind = cxip_dom_bind,
	.control = cxip_dom_control,
	.ops_open = cxip_dom_ops_open,
	.ops_set = cxip_domain_ops_set,
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

	/* The OFI check_info function does not verify that rx/tx attribute
	 * capabilities are a subset of the info capabilities. Currently
	 * MPI removes the FI_HMEM cap from info->caps but not the rx/tx
	 * caps. To avoided breaking MPI, the capabilities are removed
	 * here as a temporary work around.
	 * TODO: Remove this code when no longer required.
	 */
	if (info->caps && !(info->caps & FI_HMEM)) {
		if (info->tx_attr)
			info->tx_attr->caps &= ~FI_HMEM;
		if (info->rx_attr)
			info->rx_attr->caps &= ~FI_HMEM;
	}

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
		goto free_dom;

	if (!info || !info->src_addr) {
		CXIP_WARN("Invalid fi_info\n");
		goto close_util_dom;
	}
	src_addr = (struct cxip_addr *)info->src_addr;
	cxi_domain->nic_addr = src_addr->nic;

	if (info->domain_attr->auth_key_size) {
		if (info->domain_attr->auth_key &&
		    (info->domain_attr->auth_key_size ==
		     sizeof(struct cxi_auth_key))) {
			memcpy(&cxi_domain->auth_key,
			       info->domain_attr->auth_key,
			       sizeof(struct cxi_auth_key));
		} else {
			CXIP_WARN("Invalid auth_key\n");
			goto close_util_dom;
		}
	} else {
		/* Use default service and VNI */
		cxi_domain->auth_key.svc_id = CXI_DEFAULT_SVC_ID;
		cxi_domain->auth_key.vni = cxip_env.default_vni;
	}

	if (info->domain_attr->tclass != FI_TC_UNSPEC) {
		if (info->domain_attr->tclass >= FI_TC_LABEL &&
		    info->domain_attr->tclass <= FI_TC_SCAVENGER) {
			cxi_domain->tclass = info->domain_attr->tclass;
		} else {
			CXIP_WARN("Invalid tclass\n");
			goto close_util_dom;
		}
	} else {
		/* Use default tclass */
		cxi_domain->tclass = FI_TC_BEST_EFFORT;
	}

	cxi_domain->util_domain.domain_fid.fid.ops = &cxip_dom_fi_ops;
	cxi_domain->util_domain.domain_fid.ops = &cxip_dom_ops;
	cxi_domain->util_domain.domain_fid.mr = &cxip_dom_mr_ops;

	dlist_init(&cxi_domain->txc_list);
	dlist_init(&cxi_domain->cntr_list);
	dlist_init(&cxi_domain->cq_list);
	fastlock_init(&cxi_domain->lock);
	ofi_atomic_initialize32(&cxi_domain->ref, 0);
	cxi_domain->fab = fab;

	cxi_domain->hmem_ops.copy_from_hmem_iov = ofi_copy_from_hmem_iov;
	cxi_domain->hmem_ops.copy_to_hmem_iov = ofi_copy_to_hmem_iov;

	*dom = &cxi_domain->util_domain.domain_fid;

	return 0;

close_util_dom:
	ofi_domain_close(&cxi_domain->util_domain);
free_dom:
	free(cxi_domain);
	return -FI_EINVAL;
}
