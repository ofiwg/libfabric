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

#include "cxi_prov.h"

#include <ofi_util.h>

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

const struct fi_cntr_attr cxi_cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.wait_obj = FI_WAIT_MUTEX_COND,
	.wait_set = NULL,
	.flags = 0,
};

void cxi_cntr_add_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx)
{
	int ret;
	struct fid *fid = &tx_ctx->fid.ctx.fid;

	ret = fid_list_insert(&cntr->tx_list, &cntr->list_lock, fid);
	if (ret)
		CXI_LOG_ERROR("Error in adding ctx to progress list\n");
	else
		ofi_atomic_inc32(&cntr->ref);
}

void cxi_cntr_remove_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx)
{
	struct fid *fid = &tx_ctx->fid.ctx.fid;

	fid_list_remove(&cntr->tx_list, &cntr->list_lock, fid);
	ofi_atomic_dec32(&cntr->ref);
}

void cxi_cntr_add_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx)
{
	int ret;
	struct fid *fid = &rx_ctx->ctx.fid;

	ret = fid_list_insert(&cntr->rx_list, &cntr->list_lock, fid);
	if (ret)
		CXI_LOG_ERROR("Error in adding ctx to progress list\n");
	else
		ofi_atomic_inc32(&cntr->ref);
}

void cxi_cntr_remove_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx)
{
	struct fid *fid = &rx_ctx->ctx.fid;

	fid_list_remove(&cntr->rx_list, &cntr->list_lock, fid);
	ofi_atomic_dec32(&cntr->ref);
}

int cxi_cntr_progress(struct cxi_cntr *cntr)
{
	return 0;
}

static uint64_t cxi_cntr_read(struct fid_cntr *fid_cntr)
{
	return 0;
}

static uint64_t cxi_cntr_readerr(struct fid_cntr *fid_cntr)
{
	return 0;
}

void cxi_cntr_inc(struct cxi_cntr *cntr)
{
}

static int cxi_cntr_add(struct fid_cntr *fid_cntr, uint64_t value)
{
	return 0;
}

static int cxi_cntr_set(struct fid_cntr *fid_cntr, uint64_t value)
{
	return 0;
}

static int cxi_cntr_adderr(struct fid_cntr *fid_cntr, uint64_t value)
{
	return 0;
}

static int cxi_cntr_seterr(struct fid_cntr *fid_cntr, uint64_t value)
{
	return 0;
}

static int cxi_cntr_wait(struct fid_cntr *fid_cntr, uint64_t threshold,
			 int timeout)
{
	return 0;
}

static int cxi_cntr_control(struct fid *fid, int command, void *arg)
{
	int ret = 0;
	struct cxi_cntr *cntr;

	cntr = container_of(fid, struct cxi_cntr, cntr_fid);

	switch (command) {
	case FI_GETWAIT:
		if (cntr->domain->progress_mode == FI_PROGRESS_MANUAL)
			return -FI_ENOSYS;

		switch (cntr->attr.wait_obj) {
		case FI_WAIT_SET:
		case FI_WAIT_FD:
			cxi_wait_get_obj(cntr->waitset, arg);
			break;

		default:
			ret = -FI_EINVAL;
			break;
		}
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

static int cxi_cntr_close(struct fid *fid)
{
	struct cxi_cntr *cntr;

	cntr = container_of(fid, struct cxi_cntr, cntr_fid.fid);
	if (ofi_atomic_get32(&cntr->ref))
		return -FI_EBUSY;

	if (cntr->signal && cntr->attr.wait_obj == FI_WAIT_FD)
		cxi_wait_close(&cntr->waitset->fid);

	fastlock_destroy(&cntr->list_lock);

	ofi_atomic_dec32(&cntr->domain->ref);
	free(cntr);
	return 0;
}

static struct fi_ops_cntr cxi_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.readerr = cxi_cntr_readerr,
	.read = cxi_cntr_read,
	.add = cxi_cntr_add,
	.set = cxi_cntr_set,
	.wait = cxi_cntr_wait,
	.adderr = cxi_cntr_adderr,
	.seterr = cxi_cntr_seterr,
};

static struct fi_ops cxi_cntr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_cntr_close,
	.bind = fi_no_bind,
	.control = cxi_cntr_control,
	.ops_open = fi_no_ops_open,
};

static int cxi_cntr_verify_attr(struct fi_cntr_attr *attr)
{
	switch (attr->events) {
	case FI_CNTR_EVENTS_COMP:
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_SET:
	case FI_WAIT_FD:
		break;
	default:
		return -FI_ENOSYS;
	}
	if (attr->flags)
		return -FI_EINVAL;
	return 0;
}

int cxi_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr, void *context)
{
	int ret;
	struct cxi_domain *dom;
	struct cxi_cntr *_cntr;
	struct fi_wait_attr wait_attr;
	struct cxi_fid_list *list_entry;
	struct cxi_wait *wait;

	dom = container_of(domain, struct cxi_domain, dom_fid);
	if (attr && cxi_cntr_verify_attr(attr))
		return -FI_ENOSYS;

	_cntr = calloc(1, sizeof(*_cntr));
	if (!_cntr)
		return -FI_ENOMEM;

	if (attr == NULL)
		memcpy(&_cntr->attr, &cxi_cntr_attr, sizeof(cxi_cntr_attr));
	else
		memcpy(&_cntr->attr, attr, sizeof(cxi_cntr_attr));

	switch (_cntr->attr.wait_obj) {

	case FI_WAIT_FD:
		wait_attr.flags = 0;
		wait_attr.wait_obj = FI_WAIT_FD;
		ret = cxi_wait_open(&dom->fab->fab_fid, &wait_attr,
				     &_cntr->waitset);
		if (ret) {
			ret = FI_EINVAL;
			goto err;
		}
		_cntr->signal = 1;
		break;

	case FI_WAIT_SET:
		if (!attr) {
			ret = FI_EINVAL;
			goto err;
		}

		_cntr->waitset = attr->wait_set;
		_cntr->signal = 1;
		wait = container_of(attr->wait_set, struct cxi_wait, wait_fid);
		list_entry = calloc(1, sizeof(*list_entry));
		if (!list_entry) {
			ret = FI_ENOMEM;
			goto err;
		}
		dlist_init(&list_entry->entry);
		list_entry->fid = &_cntr->cntr_fid.fid;
		dlist_insert_after(&list_entry->entry, &wait->fid_list);
		break;

	default:
		break;
	}

	ofi_atomic_initialize32(&_cntr->ref, 0);

	dlist_init(&_cntr->tx_list);
	dlist_init(&_cntr->rx_list);
	fastlock_init(&_cntr->list_lock);

	_cntr->cntr_fid.fid.fclass = FI_CLASS_CNTR;
	_cntr->cntr_fid.fid.context = context;
	_cntr->cntr_fid.fid.ops = &cxi_cntr_fi_ops;
	_cntr->cntr_fid.ops = &cxi_cntr_ops;

	ofi_atomic_inc32(&dom->ref);
	_cntr->domain = dom;
	*cntr = &_cntr->cntr_fid;
	return 0;

err:
	free(_cntr);
	return -ret;
}

