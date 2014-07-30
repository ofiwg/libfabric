/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdlib.h>
#include <string.h>

#include "sock.h"


//static struct fi_ops sock_eq_fi_ops = {
//	.close = sock_eq_close,
//};

int sock_eq_open(struct fid_domain *domain, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context)
{
	return -FI_ENOSYS; /* TODO */
}

static uint64_t sock_cntr_read(struct fid_cntr *cntr)
{
	struct sock_cntr *_cntr;
	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	return _cntr->value;
}

static int sock_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);
	_cntr->value += value;
	if (_cntr->value >= _cntr->threshold)
		pthread_cond_signal(&_cntr->cond);
	pthread_mutex_unlock(&_cntr->mut);
	return 0;
}

static int sock_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);
	_cntr->value = value;
	if (_cntr->value >= _cntr->threshold)
		pthread_cond_signal(&_cntr->cond);
	pthread_mutex_unlock(&_cntr->mut);
	return 0;
}

static int sock_cntr_wait(struct fid_cntr *cntr, uint64_t threshold)
{
	struct sock_cntr *_cntr;

	_cntr = container_of(cntr, struct sock_cntr, cntr_fid);
	pthread_mutex_lock(&_cntr->mut);
	_cntr->threshold = threshold;
	while (_cntr->value < _cntr->threshold)
		pthread_cond_wait(&_cntr->cond, &_cntr->mut);
	_cntr->threshold = ~0;
	pthread_mutex_unlock(&_cntr->mut);
	return 0;
}

static int sock_cntr_close(struct fid *fid)
{
	struct sock_cntr *cntr;

	cntr = container_of(fid, struct sock_cntr, cntr_fid.fid);
	pthread_mutex_destroy(&cntr->mut);
	pthread_cond_destroy(&cntr->cond);
	atomic_dec(&cntr->dom->ref);
	free(cntr);
	return 0;
}

static struct fi_ops_cntr sock_cntr_ops = {
	.read = sock_cntr_read,
	.add = sock_cntr_add,
	.set = sock_cntr_set,
	.wait = sock_cntr_wait,
};

static struct fi_ops sock_cntr_fi_ops = {
	.close = sock_cntr_close,
};

int sock_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context)
{
	struct sock_domain *dom;
	struct sock_cntr *_cntr;
	int ret;

	if ((attr->events != FI_CNTR_EVENTS_COMP) ||
	    (attr->wait_obj != FI_WAIT_MUT_COND) || attr->flags)
		return -FI_ENOSYS;

	_cntr = calloc(1, sizeof(*_cntr));
	if (!_cntr)
		return -FI_ENOMEM;

	ret = pthread_cond_init(&_cntr->cond, NULL);
	if (ret)
		goto err1;

	ret = pthread_mutex_init(&_cntr->mut, NULL);
	if (ret)
		goto err2;

	_cntr->cntr_fid.fid.fclass = FID_CLASS_CNTR;
	_cntr->cntr_fid.fid.size = sizeof(struct fid_cntr);
	_cntr->cntr_fid.fid.context = context;
	_cntr->cntr_fid.fid.ops = &sock_cntr_fi_ops;
	_cntr->cntr_fid.ops = &sock_cntr_ops;
	_cntr->threshold = ~0;

	dom = container_of(domain, struct sock_domain, dom_fid);
	atomic_inc(&dom->ref);
	_cntr->dom = dom;
	*cntr = &_cntr->cntr_fid;
	return 0;

err2:
	pthread_cond_destroy(&_cntr->cond);
err1:
	free(_cntr);
	return -ret;
}
