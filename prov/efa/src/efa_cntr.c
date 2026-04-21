/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright (c) 2013-2018 Intel Corporation, Inc.  All rights reserved. */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_util.h"
#include "efa.h"
#include "efa_cntr.h"
#include "efa_cq.h"

int efa_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout)
{
	struct util_cntr *cntr;
	uint64_t start, errcnt;
	int ret;
	int numtry = 5;
	int tryid = 0;
	int waitim = 1;
	struct efa_domain *domain;

	cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	domain = container_of(cntr->domain, struct efa_domain, util_domain);

	ofi_genlock_lock(&domain->srx_lock);

	assert(cntr->wait);
	errcnt = ofi_atomic_get64(&cntr->err);
	start = (timeout >= 0) ? ofi_gettime_ms() : 0;

	for (tryid = 0; tryid < numtry; ++tryid) {
		cntr->progress(cntr);
		if (threshold <= ofi_atomic_get64(&cntr->cnt)) {
		        ret = FI_SUCCESS;
			goto unlock;
		}

		if (errcnt != ofi_atomic_get64(&cntr->err)) {
			ret = -FI_EAVAIL;
			goto unlock;
		}

		if (timeout >= 0) {
			timeout -= (int)(ofi_gettime_ms() - start);
			if (timeout <= 0) {
				ret = -FI_ETIMEDOUT;
				goto unlock;
			}
		}

		ret = ofi_wait(&cntr->wait->wait_fid, waitim);
		if (ret == -FI_ETIMEDOUT)
			ret = 0;

		waitim *= 2;
	}

unlock:
	ofi_genlock_unlock(&domain->srx_lock);
	return ret;
}

uint64_t efa_cntr_read(struct fid_cntr *cntr_fid)
{
	struct efa_domain *domain;
	struct util_cntr *util_cntr;
	uint64_t ret;

	util_cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	domain = container_of(util_cntr->domain, struct efa_domain, util_domain);

	ofi_genlock_lock(&domain->srx_lock);
	ret = ofi_cntr_read(cntr_fid);
	ofi_genlock_unlock(&domain->srx_lock);

	return ret;
}

uint64_t efa_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct efa_domain *domain;
	struct util_cntr *util_cntr;
	uint64_t ret;

	util_cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	domain = container_of(util_cntr->domain, struct efa_domain, util_domain);

	ofi_genlock_lock(&domain->srx_lock);
	ret = ofi_cntr_readerr(cntr_fid);
	ofi_genlock_unlock(&domain->srx_lock);

	return ret;
}

static struct fi_ops_cntr efa_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = efa_cntr_read,
	.readerr = efa_cntr_readerr,
	.add = ofi_cntr_add,
	.adderr = ofi_cntr_adderr,
	.set = ofi_cntr_set,
	.seterr = ofi_cntr_seterr,
	.wait = efa_cntr_wait
};

static int efa_cntr_close(struct fid *fid)
{
	struct efa_cntr *cntr;

	cntr = container_of(fid, struct efa_cntr, util_cntr.cntr_fid.fid);

	efa_cntr_destruct(cntr);
	free(cntr);
	return 0;
}

static struct fi_ops efa_cntr_fi_ops = {
	.size = sizeof(efa_cntr_fi_ops),
	.close = efa_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

void efa_cntr_progress_ibv_cq_poll_list(struct efa_cntr *efa_cntr)
{
	struct dlist_entry *item;
	struct efa_ibv_cq_poll_list_entry *poll_list_entry;
	struct efa_cq *efa_cq;

	assert(ofi_genlock_held(&efa_cntr->util_cntr.ep_list_lock));

	dlist_foreach(&efa_cntr->ibv_cq_poll_list, item) {
		poll_list_entry = container_of(item, struct efa_ibv_cq_poll_list_entry, entry);
		efa_cq = container_of(poll_list_entry->cq, struct efa_cq, ibv_cq);
		ofi_genlock_lock(&efa_cq->util_cq.ep_list_lock);
		(void) efa_cq->poll_ibv_cq(efa_env.efa_cq_read_size, poll_list_entry->cq);
		ofi_genlock_unlock(&efa_cq->util_cq.ep_list_lock);
	}
}

static void efa_cntr_progress(struct util_cntr *cntr)
{
	struct efa_cntr *efa_cntr;

	efa_cntr = container_of(cntr, struct efa_cntr, util_cntr);

	ofi_genlock_lock(&cntr->ep_list_lock);
	efa_cntr_progress_ibv_cq_poll_list(efa_cntr);
	ofi_genlock_unlock(&cntr->ep_list_lock);
}

int efa_cntr_construct(struct efa_cntr *cntr, struct fid_domain *domain,
		       struct fi_cntr_attr *attr,
		       ofi_cntr_progress_func progress, void *context)
{
	dlist_init(&cntr->ibv_cq_poll_list);
	return ofi_cntr_init(&efa_prov, domain, attr, &cntr->util_cntr,
			     progress, context);
}

void efa_cntr_destruct(struct efa_cntr *cntr)
{
	ofi_cntr_cleanup(&cntr->util_cntr);
}

int efa_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr_fid, void *context)
{
	int ret;
	struct efa_cntr *cntr;

	cntr = calloc(1, sizeof(*cntr));
	if (!cntr)
		return -FI_ENOMEM;

	ret = efa_cntr_construct(cntr, domain, attr, efa_cntr_progress, context);
	if (ret) {
		free(cntr);
		return ret;
	}

	*cntr_fid = &cntr->util_cntr.cntr_fid;
	cntr->util_cntr.cntr_fid.ops = &efa_cntr_ops;
	cntr->util_cntr.cntr_fid.fid.ops = &efa_cntr_fi_ops;

	return FI_SUCCESS;
}

void efa_cntr_report_tx_completion(struct util_ep *ep, uint64_t flags)
{
	struct util_cntr *cntr;

	flags &= (FI_SEND | FI_WRITE | FI_READ);
	assert(flags == FI_SEND || flags == FI_WRITE || flags == FI_READ);

	if (flags == FI_SEND)
		cntr = ep->cntrs[CNTR_TX];
	else if (flags == FI_WRITE)
		cntr = ep->cntrs[CNTR_WR];
	else if (flags == FI_READ)
		cntr = ep->cntrs[CNTR_RD];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

void efa_cntr_report_rx_completion(struct util_ep *ep, uint64_t flags)
{
	struct util_cntr *cntr;

	flags &= (FI_RECV | FI_REMOTE_WRITE | FI_REMOTE_READ);
	assert(flags == FI_RECV || flags == FI_REMOTE_WRITE || flags == FI_REMOTE_READ);

	if (flags == FI_RECV)
		cntr = ep->cntrs[CNTR_RX];
	else if (flags == FI_REMOTE_READ)
		cntr = ep->cntrs[CNTR_REM_RD];
	else if (flags == FI_REMOTE_WRITE)
		cntr = ep->cntrs[CNTR_REM_WR];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
}

void efa_cntr_report_error(struct util_ep *ep, uint64_t flags)
{
	flags = flags & (FI_SEND | FI_READ | FI_WRITE | FI_ATOMIC |
			 FI_RECV | FI_REMOTE_READ | FI_REMOTE_WRITE);

	struct util_cntr *cntr;

	if (flags == FI_WRITE || flags == FI_ATOMIC)
		cntr = ep->cntrs[CNTR_WR];
	else if (flags == FI_READ)
		cntr = ep->cntrs[CNTR_RD];
	else if (flags == FI_SEND)
		cntr = ep->cntrs[CNTR_TX];
	else if (flags == FI_RECV)
		cntr = ep->cntrs[CNTR_RX];
	else if (flags == FI_REMOTE_READ)
		cntr = ep->cntrs[CNTR_REM_RD];
	else if (flags == FI_REMOTE_WRITE)
		cntr = ep->cntrs[CNTR_REM_WR];
	else
		cntr = NULL;

	if (cntr)
		cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
}
