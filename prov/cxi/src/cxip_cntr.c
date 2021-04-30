/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "cxip.h"

#include <ofi_util.h>

#define CXIP_DBG(...) _CXIP_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXIP_WARN(...) _CXIP_WARN(FI_LOG_EP_DATA, __VA_ARGS__)

static int cxip_dom_cntr_enable(struct cxip_domain *dom)
{
	struct cxi_cq_alloc_opts cq_opts = {};
	int ret;

	fastlock_acquire(&dom->lock);

	if (dom->cntr_init) {
		fastlock_release(&dom->lock);
		return FI_SUCCESS;
	}

	if (!dom->enabled) {
		fastlock_release(&dom->lock);

		ret = cxip_domain_enable(dom);
		if (ret != FI_SUCCESS) {
			CXIP_WARN("cxip_domain_enable returned: %d\n", ret);
			return ret;
		}

		fastlock_acquire(&dom->lock);
	}

	cq_opts.count = 64;
	cq_opts.flags = CXI_CQ_IS_TX | CXI_CQ_TX_WITH_TRIG_CMDS;
	cq_opts.policy = CXI_CQ_UPDATE_ALWAYS;

	ret = cxip_cmdq_alloc(dom->lni, NULL, &cq_opts,
			      dom->auth_key.vni,
			      cxip_ofi_to_cxi_tc(dom->tclass),
			      CXI_TC_TYPE_DEFAULT,
			      &dom->trig_cmdq);
	if (ret != FI_SUCCESS) {
		CXIP_WARN("Failed to allocate trig_cmdq: %d\n", ret);

		fastlock_release(&dom->lock);
		return -FI_ENOSPC;
	}

	dom->cntr_init = true;

	CXIP_DBG("Domain counters enabled: %p\n", dom);

	fastlock_release(&dom->lock);

	return FI_SUCCESS;
}

void cxip_dom_cntr_disable(struct cxip_domain *dom)
{
	if (dom->cntr_init)
		cxip_cmdq_free(dom->trig_cmdq);
}

const struct fi_cntr_attr cxip_cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.wait_obj = FI_WAIT_NONE,
	.wait_set = NULL,
	.flags = 0,
};

/*
 * cxip_cntr_mod() - Modify counter value.
 *
 * Set or increment the success or failure value of a counter by 'value'.
 */
int cxip_cntr_mod(struct cxip_cntr *cxi_cntr, uint64_t value, bool set,
		  bool err)
{
	struct c_ct_cmd cmd = {};
	struct cxip_cmdq *cmdq = cxi_cntr->domain->trig_cmdq;
	int ret;

	fastlock_acquire(&cxi_cntr->lock);

	/* Modifications are invalid before a counter is in use */
	if (!cxi_cntr->enabled) {
		fastlock_release(&cxi_cntr->lock);

		ret = cxip_cntr_enable(cxi_cntr);
		if (ret != FI_SUCCESS)
			return -FI_EOPBADSTATE;

		fastlock_acquire(&cxi_cntr->lock);
	}

	if (!set) {
		/* Doorbell supports counter increment */
		if (err)
			cxi_ct_inc_failure(cxi_cntr->ct, value);
		else
			cxi_ct_inc_success(cxi_cntr->ct, value);
	} else {
		/* Doorbell supports counter reset */
		if (!value) {
			if (err)
				cxi_ct_reset_failure(cxi_cntr->ct);
			else
				cxi_ct_reset_success(cxi_cntr->ct);

			/* Doorbell reset triggers a write-back */
			cxi_cntr->wb_pending = true;
		} else {
			/* Use CQ to set a specific counter value */
			cmd.ct = cxi_cntr->ct->ctn;
			if (err) {
				cmd.set_ct_failure = 1;
				cmd.ct_failure = value;
			} else {
				cmd.set_ct_success = 1;
				cmd.ct_success = value;
			}

			fastlock_acquire(&cmdq->lock);
			ret = cxi_cq_emit_ct(cmdq->dev_cmdq, C_CMD_CT_SET,
					     &cmd);
			if (ret) {
				fastlock_release(&cmdq->lock);
				fastlock_release(&cxi_cntr->lock);
				return -FI_EAGAIN;
			}
			cxi_cq_ring(cmdq->dev_cmdq);
			fastlock_release(&cmdq->lock);

			/* Set commands will trigger a write-back */
			cxi_cntr->wb_pending = true;
		}
	}

	fastlock_release(&cxi_cntr->lock);

	return FI_SUCCESS;
}

/*
 * cxip_cntr_get() - Schedule a counter write-back.
 *
 * Schedule hardware to write the value of a counter to memory. Avoid
 * scheduling multiple write-backs at once. The counter value will appear in
 * memory a small amount of time later.
 */
static int cxip_cntr_get(struct cxip_cntr *cxi_cntr)
{
	struct c_ct_cmd cmd = {};
	struct cxip_cmdq *cmdq = cxi_cntr->domain->trig_cmdq;
	int ret;

	fastlock_acquire(&cxi_cntr->lock);

	/* Get is a NOP when counter is not in use */
	if (!cxi_cntr->enabled) {
		fastlock_release(&cxi_cntr->lock);
		return -FI_EOPBADSTATE;
	}

	if (cxi_cntr->wb_pending) {
		if (cxi_cntr->wb.ct_writeback) {
			cxi_cntr->wb.ct_writeback = 0;
			cxi_cntr->wb_pending = false;
		}
		fastlock_release(&cxi_cntr->lock);
		return FI_SUCCESS;
	}

	/* Request a write-back */
	cmd.ct = cxi_cntr->ct->ctn;

	fastlock_acquire(&cmdq->lock);
	ret = cxi_cq_emit_ct(cmdq->dev_cmdq, C_CMD_CT_GET, &cmd);
	if (ret) {
		fastlock_release(&cmdq->lock);
		fastlock_release(&cxi_cntr->lock);
		return -FI_EAGAIN;
	}
	cxi_cq_ring(cmdq->dev_cmdq);
	fastlock_release(&cmdq->lock);

	/* Only schedule one write-back at a time */
	cxi_cntr->wb_pending = true;

	fastlock_release(&cxi_cntr->lock);

	return FI_SUCCESS;
}

/*
 * cxip_cntr_read() - fi_cntr_read() implementation.
 */
static uint64_t cxip_cntr_read(struct fid_cntr *fid_cntr)
{
	struct cxip_cntr *cxi_cntr;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	cxip_cntr_get(cxi_cntr);

	return cxi_cntr->wb.ct_success;
}

/*
 * cxip_cntr_readerr() - fi_cntr_readerr() implementation.
 */
static uint64_t cxip_cntr_readerr(struct fid_cntr *fid_cntr)
{
	struct cxip_cntr *cxi_cntr;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	cxip_cntr_get(cxi_cntr);

	return cxi_cntr->wb.ct_failure;
}

/*
 * cxip_cntr_add() - fi_cntr_add() implementation.
 */
static int cxip_cntr_add(struct fid_cntr *fid_cntr, uint64_t value)
{
	struct cxip_cntr *cxi_cntr;

	if (value > FI_CXI_CNTR_SUCCESS_MAX)
		return -FI_EINVAL;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	return cxip_cntr_mod(cxi_cntr, value, false, false);
}

/*
 * cxip_cntr_set() - fi_cntr_set() implementation.
 */
static int cxip_cntr_set(struct fid_cntr *fid_cntr, uint64_t value)
{
	struct cxip_cntr *cxi_cntr;

	if (value > FI_CXI_CNTR_SUCCESS_MAX)
		return -FI_EINVAL;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	return cxip_cntr_mod(cxi_cntr, value, true, false);
}

/*
 * cxip_cntr_adderr() - fi_cntr_adderr() implementation.
 */
static int cxip_cntr_adderr(struct fid_cntr *fid_cntr, uint64_t value)
{
	struct cxip_cntr *cxi_cntr;

	if (value > FI_CXI_CNTR_FAILURE_MAX)
		return -FI_EINVAL;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	return cxip_cntr_mod(cxi_cntr, value, false, true);
}

/*
 * cxip_cntr_seterr() - fi_cntr_seterr() implementation.
 */
static int cxip_cntr_seterr(struct fid_cntr *fid_cntr, uint64_t value)
{
	struct cxip_cntr *cxi_cntr;

	if (value > FI_CXI_CNTR_FAILURE_MAX)
		return -FI_EINVAL;

	cxi_cntr = container_of(fid_cntr, struct cxip_cntr, cntr_fid);

	return cxip_cntr_mod(cxi_cntr, value, true, true);
}

/*
 * cxip_cntr_wait() - fi_cntr_wait() implementation.
 */
__attribute__((unused))
static int cxip_cntr_wait(struct fid_cntr *fid_cntr, uint64_t threshold,
			  int timeout)
{
	return -FI_ENOSYS;
}

/*
 * cxip_cntr_control() - fi_control() implementation for counter objects.
 */
static int cxip_cntr_control(struct fid *fid, int command, void *arg)
{
	int ret = FI_SUCCESS;
	struct cxip_cntr *cntr;

	cntr = container_of(fid, struct cxip_cntr, cntr_fid);

	switch (command) {
	case FI_GETWAIT:
		if (cntr->wait)
			ret = fi_control(&cntr->wait->fid,
					 FI_GETWAIT, arg);
		else
			ret = -FI_EINVAL;
		break;

	case FI_GETOPSFLAG:
		memcpy(arg, &cntr->attr.flags, sizeof(uint64_t));
		break;

	case FI_SETOPSFLAG:
		memcpy(&cntr->attr.flags, arg, sizeof(uint64_t));
		break;

	default:
		ret = -FI_EINVAL;
		break;
	}

	return ret;
}

/*
 * cxip_cntr_enable() - Assign hardware resources to the Counter.
 */
int cxip_cntr_enable(struct cxip_cntr *cxi_cntr)
{
	int ret;

	fastlock_acquire(&cxi_cntr->lock);

	if (cxi_cntr->enabled) {
		ret = FI_SUCCESS;
		goto unlock;
	}

	ret = cxip_dom_cntr_enable(cxi_cntr->domain);
	if (ret != FI_SUCCESS)
		goto unlock;

	ret = cxil_alloc_ct(cxi_cntr->domain->lni->lni,
			    &cxi_cntr->wb, &cxi_cntr->ct);
	if (ret) {
		CXIP_WARN("Failed to allocate CT, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	cxi_cntr->enabled = true;

	CXIP_DBG("Counter enabled: %p (CT: %d)\n",
		 cxi_cntr, cxi_cntr->ct->ctn);

	fastlock_release(&cxi_cntr->lock);

	return FI_SUCCESS;

unlock:
	fastlock_release(&cxi_cntr->lock);

	return ret;
}

/*
 * cxip_cntr_disable() - Release hardware resources from the Counter.
 */
static void cxip_cntr_disable(struct cxip_cntr *cxi_cntr)
{
	int ret;

	fastlock_acquire(&cxi_cntr->lock);

	if (!cxi_cntr->enabled)
		goto unlock;

	ret = cxil_destroy_ct(cxi_cntr->ct);
	if (ret)
		CXIP_WARN("Failed to free CT, ret: %d\n", ret);

	cxi_cntr->enabled = false;

	CXIP_DBG("Counter disabled: %p\n", cxi_cntr);

unlock:
	fastlock_release(&cxi_cntr->lock);
}

/*
 * cxip_cntr_close() - fi_close() implementation for counter objects.
 */
static int cxip_cntr_close(struct fid *fid)
{
	struct cxip_cntr *cntr;

	cntr = container_of(fid, struct cxip_cntr, cntr_fid.fid);
	if (ofi_atomic_get32(&cntr->ref))
		return -FI_EBUSY;

	cxip_cntr_disable(cntr);

	fastlock_destroy(&cntr->lock);

	cxip_domain_remove_cntr(cntr->domain, cntr);

	free(cntr);
	return 0;
}

static struct fi_ops_cntr cxip_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.readerr = cxip_cntr_readerr,
	.read = cxip_cntr_read,
	.add = cxip_cntr_add,
	.set = cxip_cntr_set,
	.wait = fi_no_cntr_wait,
	.adderr = cxip_cntr_adderr,
	.seterr = cxip_cntr_seterr,
};

static struct fi_ops cxip_cntr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxip_cntr_close,
	.bind = fi_no_bind,
	.control = cxip_cntr_control,
	.ops_open = fi_no_ops_open,
};

/*
 * cxip_cntr_verify_attr() - Verify counter creation attributes.
 */
static int cxip_cntr_verify_attr(struct fi_cntr_attr *attr)
{
	if (!attr)
		return FI_SUCCESS;

	if (attr->events != FI_CNTR_EVENTS_COMP)
		return -FI_ENOSYS;

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	if (attr->flags)
		return -FI_ENOSYS;

	return FI_SUCCESS;
}

/*
 * cxip_cntr_open() - fi_cntr_open() implementation.
 */
int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context)
{
	int ret;
	struct cxip_domain *dom;
	struct cxip_cntr *_cntr;

	dom = container_of(domain, struct cxip_domain, util_domain.domain_fid);

	ret = cxip_cntr_verify_attr(attr);
	if (ret != FI_SUCCESS)
		return ret;

	_cntr = calloc(1, sizeof(*_cntr));
	if (!_cntr)
		return -FI_ENOMEM;

	if (!attr)
		memcpy(&_cntr->attr, &cxip_cntr_attr, sizeof(cxip_cntr_attr));
	else
		memcpy(&_cntr->attr, attr, sizeof(cxip_cntr_attr));

	ofi_atomic_initialize32(&_cntr->ref, 0);

	fastlock_init(&_cntr->lock);

	_cntr->cntr_fid.fid.fclass = FI_CLASS_CNTR;
	_cntr->cntr_fid.fid.context = context;
	_cntr->cntr_fid.fid.ops = &cxip_cntr_fi_ops;
	_cntr->cntr_fid.ops = &cxip_cntr_ops;

	_cntr->domain = dom;
	*cntr = &_cntr->cntr_fid;

	cxip_domain_add_cntr(dom, _cntr);

	return FI_SUCCESS;
}
