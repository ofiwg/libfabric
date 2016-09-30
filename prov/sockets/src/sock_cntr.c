/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
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

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "sock.h"
#include "sock_util.h"

#include <fi_util.h>

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

const struct fi_cntr_attr sock_cntr_attr = {
	.events = FI_CNTR_EVENTS_COMP,
	.wait_obj = FI_WAIT_MUTEX_COND,
	.wait_set = NULL,
	.flags = 0,
};

void sock_cntr_add_tx_ctx(struct sock_cntr *cntr, struct sock_tx_ctx *tx_ctx)
{
	int ret;
	struct fid *fid = &tx_ctx->fid.ctx.fid;
	ret = fid_list_insert(&cntr->tx_list, &cntr->list_lock, fid);
	if (ret)
		SOCK_LOG_ERROR("Error in adding ctx to progress list\n");
}

void sock_cntr_remove_tx_ctx(struct sock_cntr *cntr, struct sock_tx_ctx *tx_ctx)
{
	struct fid *fid = &tx_ctx->fid.ctx.fid;
	fid_list_remove(&cntr->tx_list, &cntr->list_lock, fid);
}

void sock_cntr_add_rx_ctx(struct sock_cntr *cntr, struct sock_rx_ctx *rx_ctx)
{
	int ret;
	struct fid *fid = &rx_ctx->ctx.fid;
	ret = fid_list_insert(&cntr->rx_list, &cntr->list_lock, fid);
	if (ret)
		SOCK_LOG_ERROR("Error in adding ctx to progress list\n");
}

void sock_cntr_remove_rx_ctx(struct sock_cntr *cntr, struct sock_rx_ctx *rx_ctx)
{
	struct fid *fid = &rx_ctx->ctx.fid;
	fid_list_remove(&cntr->rx_list, &cntr->list_lock, fid);
}

int sock_cntr_progress(struct sock_cntr *cntr)
{
	struct sock_tx_ctx *tx_ctx;
	struct sock_rx_ctx *rx_ctx;
	struct dlist_entry *entry;

	struct fid_list_entry *fid_entry;

	if (cntr->domain->progress_mode == FI_PROGRESS_AUTO)
		return 0;

	fastlock_acquire(&cntr->list_lock);
	for (entry = cntr->tx_list.next; entry != &cntr->tx_list;
	     entry = entry->next) {
		fid_entry = container_of(entry, struct fid_list_entry, entry);
		tx_ctx = container_of(fid_entry->fid, struct sock_tx_ctx, fid.ctx.fid);
		if (tx_ctx->use_shared)
			sock_pe_progress_tx_ctx(cntr->domain->pe, tx_ctx->stx_ctx);
		else
			sock_pe_progress_tx_ctx(cntr->domain->pe, tx_ctx);
	}

	for (entry = cntr->rx_list.next; entry != &cntr->rx_list;
	     entry = entry->next) {
		fid_entry = container_of(entry, struct fid_list_entry, entry);
		rx_ctx = container_of(fid_entry->fid, struct sock_rx_ctx, ctx.fid);
		if (rx_ctx->use_shared)
			sock_pe_progress_rx_ctx(cntr->domain->pe, rx_ctx->srx_ctx);
		else
			sock_pe_progress_rx_ctx(cntr->domain->pe, rx_ctx);
	}

	fastlock_release(&cntr->list_lock);
	return 0;
}

void sock_cntr_check_trigger_list(struct sock_cntr *cntr)
{
	struct sock_trigger *trigger;
	struct dlist_entry *entry;
	int ret = 0;

	fastlock_acquire(&cntr->trigger_lock);
	for (entry = cntr->trigger_list.next;
	     entry != &cntr->trigger_list;) {

		trigger = container_of(entry, struct sock_trigger, entry);
		entry = entry->next;

		if (atomic_get(&cntr->value) < trigger->threshold)
			continue;

		switch (trigger->op_type) {
		case SOCK_OP_SEND:
			ret = sock_ep_sendmsg(trigger->ep, &trigger->op.msg.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_RECV:
			ret = sock_ep_recvmsg(trigger->ep, &trigger->op.msg.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_TSEND:
			ret = sock_ep_tsendmsg(trigger->ep,
					&trigger->op.tmsg.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_TRECV:
			ret = sock_ep_trecvmsg(trigger->ep,
					&trigger->op.tmsg.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_WRITE:
			ret = sock_ep_rma_writemsg(trigger->ep,
					&trigger->op.rma.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_READ:
			ret = sock_ep_rma_readmsg(trigger->ep,
					&trigger->op.rma.msg,
					trigger->flags & ~FI_TRIGGER);
			break;

		case SOCK_OP_ATOMIC:
			ret = sock_ep_tx_atomic(trigger->ep,
					&trigger->op.atomic.msg,
					trigger->op.atomic.comparev,
					NULL,
					trigger->op.atomic.compare_count,
					trigger->op.atomic.resultv,
					NULL,
					trigger->op.atomic.result_count,
					trigger->flags & ~FI_TRIGGER);
			break;

		default:
			SOCK_LOG_ERROR("unsupported op\n");
			ret = 0;
			break;
		}

		if (ret != -FI_EAGAIN) {
			dlist_remove(&trigger->entry);
			free(trigger);
		} else {
			break;
		}
	}
	fastlock_release(&cntr->trigger_lock);
}

static uint64_t sock_cntr_read(struct fid_cntr *cntr)
{
	struct sock_cntr *_cntr;
	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	sock_cntr_progress(_cntr);
	return atomic_get(&_cntr->value);
}

void sock_cntr_inc(struct sock_cntr *cntr)
{
	pthread_mutex_lock(&cntr->mut);
	atomic_inc(&cntr->value);
	if(atomic_get(&cntr->num_waiting))
		pthread_cond_broadcast(&cntr->cond);
	pthread_mutex_unlock(&cntr->mut);
	sock_cntr_check_trigger_list(cntr);
}

void sock_cntr_err_inc(struct sock_cntr *cntr)
{
	pthread_mutex_lock(&cntr->mut);
	atomic_inc(&cntr->err_cnt);
	if (!cntr->err_flag)
		cntr->err_flag = 1;
	pthread_cond_signal(&cntr->cond);
	pthread_mutex_unlock(&cntr->mut);
}

static int sock_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	uint64_t new_val;
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);
	new_val = atomic_add(&_cntr->value, value);
	atomic_set(&_cntr->last_read_val, new_val);
	if(atomic_get(&_cntr->num_waiting))
		pthread_cond_broadcast(&_cntr->cond);
	pthread_mutex_unlock(&_cntr->mut);
	sock_cntr_check_trigger_list(_cntr);
	return 0;
}

static int sock_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	uint64_t new_val;
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);
	new_val = atomic_set(&_cntr->value, value);
	atomic_set(&_cntr->last_read_val, new_val);
	if(atomic_get(&_cntr->num_waiting))
		pthread_cond_broadcast(&_cntr->cond);
	pthread_mutex_unlock(&_cntr->mut);
	sock_cntr_check_trigger_list(_cntr);
	return 0;
}

static int sock_cntr_wait(struct fid_cntr *cntr, uint64_t threshold,
			  int timeout)
{
	int last_read, ret = 0;
	uint64_t start_ms = 0, end_ms = 0, remaining_ms = 0;
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);

	if (_cntr->err_flag) {
		ret = -FI_EAVAIL;
		goto out;
	}

	if (atomic_get(&_cntr->value) >= threshold) {
		ret = 0;
		goto out;
	}

	atomic_inc(&_cntr->num_waiting);

	if (timeout >= 0) {
		start_ms = fi_gettime_ms();
		end_ms = start_ms + timeout;
	}

        last_read = atomic_get(&_cntr->value);
	remaining_ms = timeout;

	while (!ret && last_read < threshold) {
		if(_cntr->domain->progress_mode == FI_PROGRESS_MANUAL) {
			pthread_mutex_unlock(&_cntr->mut);

			ret = sock_cntr_progress(_cntr);

			pthread_mutex_lock(&_cntr->mut);

		} else {
			ret = fi_wait_cond(&_cntr->cond, &_cntr->mut, remaining_ms);
		}

		uint64_t curr_ms = fi_gettime_ms();
		if(timeout >= 0) {
			if(curr_ms >= end_ms) {
				ret = -FI_ETIMEDOUT;
				break;
			} else {
				remaining_ms = end_ms - curr_ms;
			}
		}

		last_read = atomic_get(&_cntr->value);
	}

	atomic_set(&_cntr->last_read_val, last_read);
	atomic_dec(&_cntr->num_waiting);

	pthread_mutex_unlock(&_cntr->mut);
	sock_cntr_check_trigger_list(_cntr);
	return (_cntr->err_flag) ? -FI_EAVAIL : ret;

out:
	pthread_mutex_unlock(&_cntr->mut);
	return ret;
}

static int sock_cntr_control(struct fid *fid, int command, void *arg)
{
	int ret = 0;
	struct sock_cntr *cntr;

	cntr = container_of(fid, struct sock_cntr, cntr_fid);

	switch (command) {
	case FI_GETWAIT:
		if (cntr->domain->progress_mode == FI_PROGRESS_MANUAL)
			return -FI_ENOSYS;

		switch (cntr->attr.wait_obj) {
		case FI_WAIT_NONE:
		case FI_WAIT_UNSPEC:
		case FI_WAIT_MUTEX_COND:
			memcpy(arg, &cntr->mut, sizeof(cntr->mut));
			memcpy((char *)arg + sizeof(cntr->mut), &cntr->cond,
			       sizeof(cntr->cond));
			break;

		case FI_WAIT_SET:
		case FI_WAIT_FD:
			sock_wait_get_obj(cntr->waitset, arg);
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

static int sock_cntr_close(struct fid *fid)
{
	struct sock_cntr *cntr;

	cntr = container_of(fid, struct sock_cntr, cntr_fid.fid);
	if (atomic_get(&cntr->ref))
		return -FI_EBUSY;

	if (cntr->signal && cntr->attr.wait_obj == FI_WAIT_FD)
		sock_wait_close(&cntr->waitset->fid);

	pthread_mutex_destroy(&cntr->mut);
	fastlock_destroy(&cntr->list_lock);
	fastlock_destroy(&cntr->trigger_lock);

	pthread_cond_destroy(&cntr->cond);
	atomic_dec(&cntr->domain->ref);
	free(cntr);
	return 0;
}

static uint64_t sock_cntr_readerr(struct fid_cntr *cntr)
{
	struct sock_cntr *_cntr;
	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	if (_cntr->domain->progress_mode == FI_PROGRESS_MANUAL)
		sock_cntr_progress(_cntr);
	if (_cntr->err_flag)
		_cntr->err_flag = 0;
	return atomic_get(&_cntr->err_cnt);
}

static struct fi_ops_cntr sock_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.readerr = sock_cntr_readerr,
	.read = sock_cntr_read,
	.add = sock_cntr_add,
	.set = sock_cntr_set,
	.wait = sock_cntr_wait,
};

static struct fi_ops sock_cntr_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_cntr_close,
	.bind = fi_no_bind,
	.control = sock_cntr_control,
	.ops_open = fi_no_ops_open,
};

static int sock_cntr_verify_attr(struct fi_cntr_attr *attr)
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
	case FI_WAIT_MUTEX_COND:
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

int sock_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context)
{
	int ret;
	struct sock_domain *dom;
	struct sock_cntr *_cntr;
	struct fi_wait_attr wait_attr;
	struct sock_fid_list *list_entry;
	struct sock_wait *wait;

	dom = container_of(domain, struct sock_domain, dom_fid);
	if (attr && sock_cntr_verify_attr(attr))
		return -FI_ENOSYS;

	_cntr = calloc(1, sizeof(*_cntr));
	if (!_cntr)
		return -FI_ENOMEM;

	ret = pthread_cond_init(&_cntr->cond, NULL);
	if (ret)
		goto err;

	if (attr == NULL)
		memcpy(&_cntr->attr, &sock_cntr_add, sizeof(sock_cntr_attr));
	else
		memcpy(&_cntr->attr, attr, sizeof(sock_cntr_attr));

	switch (_cntr->attr.wait_obj) {

	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_MUTEX_COND:
		_cntr->signal = 0;
		break;

	case FI_WAIT_FD:
		wait_attr.flags = 0;
		wait_attr.wait_obj = FI_WAIT_FD;
		ret = sock_wait_open(&dom->fab->fab_fid, &wait_attr,
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
		wait = container_of(attr->wait_set, struct sock_wait, wait_fid);
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

	pthread_mutex_init(&_cntr->mut, NULL);
	fastlock_init(&_cntr->list_lock);

	atomic_initialize(&_cntr->ref, 0);
	atomic_initialize(&_cntr->err_cnt, 0);

	atomic_initialize(&_cntr->value, 0);
	atomic_initialize(&_cntr->last_read_val, 0);
	atomic_initialize(&_cntr->num_waiting, 0);

	dlist_init(&_cntr->tx_list);
	dlist_init(&_cntr->rx_list);

	dlist_init(&_cntr->trigger_list);
	fastlock_init(&_cntr->trigger_lock);

	_cntr->cntr_fid.fid.fclass = FI_CLASS_CNTR;
	_cntr->cntr_fid.fid.context = context;
	_cntr->cntr_fid.fid.ops = &sock_cntr_fi_ops;
	_cntr->cntr_fid.ops = &sock_cntr_ops;

	atomic_inc(&dom->ref);
	_cntr->domain = dom;
	*cntr = &_cntr->cntr_fid;
	return 0;

err:
	free(_cntr);
	return -ret;
}

