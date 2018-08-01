/*
 * Copyright (c) 2018 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <ofi.h>
#include "hook.h"
#include "hook_perf.h"


static uint32_t hooks_enabled;


struct fid *hook_to_hfid(const struct fid *fid)
{
	switch (fid->fclass) {
	case FI_CLASS_FABRIC:
		return &(container_of(fid, struct hook_fabric, fabric.fid)->
			 hfabric->fid);
	case FI_CLASS_DOMAIN:
		return &(container_of(fid, struct hook_domain, domain.fid)->
			 hdomain->fid);
	case FI_CLASS_AV:
		return &(container_of(fid, struct hook_av, av.fid)->
			 hav->fid);
	case FI_CLASS_WAIT:
		return &(container_of(fid, struct hook_wait, wait.fid)->
			 hwait->fid);
	case FI_CLASS_POLL:
		return &(container_of(fid, struct hook_poll, poll.fid)->
			 hpoll->fid);
	case FI_CLASS_EQ:
		return &(container_of(fid, struct hook_eq, eq.fid)->
			 heq->fid);
	case FI_CLASS_CQ:
		return &(container_of(fid, struct hook_cq, cq.fid)->
			 hcq->fid);
	case FI_CLASS_CNTR:
		return &(container_of(fid, struct hook_cntr, cntr.fid)->
			 hcntr->fid);
	case FI_CLASS_SEP:
	case FI_CLASS_EP:
	case FI_CLASS_RX_CTX:
	case FI_CLASS_SRX_CTX:
	case FI_CLASS_TX_CTX:
		return &(container_of(fid, struct hook_ep, ep.fid)->
			 hep->fid);
	case FI_CLASS_PEP:
		return &(container_of(fid, struct hook_pep, pep.fid)->
			 hpep->fid);
	case FI_CLASS_STX_CTX:
		return &(container_of(fid, struct hook_stx, stx.fid)->
			 hstx->fid);
	case FI_CLASS_MR:
		return &(container_of(fid, struct hook_mr, mr.fid)->
			 hmr->fid);
	default:
		return NULL;
	}
}

struct fid_wait *hook_to_hwait(const struct fid_wait *wait)
{
	return container_of(wait, struct hook_wait, wait)->hwait;
}


static int hook_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct fid *hfid, *hbfid;

	hfid = hook_to_hfid(fid);
	hbfid = hook_to_hfid(bfid);
	if (!hfid || !hbfid)
		return -FI_EINVAL;

	return hfid->ops->bind(hfid, hbfid, flags);
}

static int hook_control(struct fid *fid, int command, void *arg)
{
	struct fid *hfid;

	hfid = hook_to_hfid(fid);
	if (!hfid)
		return -FI_EINVAL;

	return hfid->ops->control(hfid, command, arg);
}

static int hook_ops_open(struct fid *fid, const char *name,
			 uint64_t flags, void **ops, void *context)
{
	struct fid *hfid;

	hfid = hook_to_hfid(fid);
	if (!hfid)
		return -FI_EINVAL;

	return hfid->ops->ops_open(hfid, name, flags, ops, context);
}

static int hook_close(struct fid *fid)
{
	struct fid *hfid;
	int ret;

	hfid = hook_to_hfid(fid);
	if (!hfid)
		return -FI_EINVAL;

	ret = hfid->ops->close(hfid);
	if (!ret)
		free(fid);
	return ret;
}

static int hook_fabric_close(struct fid *fid)
{
	struct hook_fabric *fab;

	fab = container_of(fid, struct hook_fabric, fabric.fid);
	switch (fab->hclass) {
	case HOOK_PERF:
		hook_perf_destroy(fab);
		break;
	default:
		break;
	}
	return hook_close(fid);
}

struct fi_ops hook_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = hook_close,
	.bind = hook_bind,
	.control = hook_control,
	.ops_open = hook_ops_open,
};

static struct fi_ops hook_fabric_fid_ops = {
	.size = sizeof(struct fi_ops),
	.close = hook_fabric_close,
	.bind = hook_bind,
	.control = hook_control,
	.ops_open = hook_ops_open,
};

static struct fi_ops_fabric hook_fabric_ops = {
	.size = sizeof(struct fi_ops_fabric),
	.domain = hook_domain,
	.passive_ep = hook_passive_ep,
	.eq_open = hook_eq_open,
	.wait_open = hook_wait_open,
	.trywait = hook_trywait,
};

static int hook_fabric(struct fid_fabric *hfabric, struct fid_fabric **fabric,
			enum hook_class hclass, struct fi_provider *prov)
{
	struct hook_fabric *fab;
	int ret = 0;

	switch (hclass) {
	case HOOK_NOOP:
		FI_TRACE(prov, FI_LOG_FABRIC, "Installing noop hook\n");
		fab = calloc(1, sizeof *fab);
		if (!fab)
			return -FI_ENOMEM;
		fab->prov = prov;
		break;
	case HOOK_PERF:
		FI_TRACE(prov, FI_LOG_FABRIC, "Installing perf hook\n");
		ret = hook_perf_create(&fab, prov);
		break;
	default:
		FI_WARN(&core_prov, FI_LOG_FABRIC, "Invalid hook specified\n");
		return -FI_ENOSYS;
	}

	if (ret)
		return ret;

	fab->hclass = hclass;
	fab->hfabric = hfabric;
	fab->fabric.fid.fclass = FI_CLASS_FABRIC;
	fab->fabric.fid.context = hfabric->fid.context;
	fab->fabric.fid.ops = &hook_fabric_fid_ops;
	fab->fabric.api_version = hfabric->api_version;
	fab->fabric.ops = &hook_fabric_ops;

	hfabric->fid.context = fab;
	*fabric = &fab->fabric;
	return 0;
}

void ofi_hook_install(struct fid_fabric *hfabric, struct fid_fabric **fabric,
		      struct fi_provider *prov)
{
	int hooks, hclass, ret;

	*fabric = hfabric;
	if (!hooks_enabled)
		return;

	for (hooks = hooks_enabled, hclass = 0; hooks;
	     hooks >>= 1, hclass++) {
		if (hooks & 0x1) {
			ret = hook_fabric(hfabric, fabric,
					  (enum hook_class) hclass, prov);
			if (ret)
				return;
			hfabric = *fabric;
		}
	}
}

void ofi_hook_init(void)
{
	char *param_val = NULL;

	fi_param_define(NULL, "hook", FI_PARAM_STRING,
			"Intercept calls to underlying provider and apply "
			"the specified functionality to them.  Hook option: "
			"perf (gather performance data)");
	fi_param_get_str(NULL, "hook", &param_val);

	hooks_enabled = 0;
	if (!param_val)
		return;

	if (!strcasecmp(param_val, "noop")) {
		FI_INFO(&core_prov, FI_LOG_CORE, "Noop hook requested\n");
		hooks_enabled |= (1 << HOOK_NOOP);
	}
	if (!strcasecmp(param_val, "perf")) {
		FI_INFO(&core_prov, FI_LOG_CORE, "Perf hook requested\n");
		hooks_enabled |= (1 << HOOK_PERF);
	}
}
