/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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
#include <string.h>

#include "udpx.h"


static void util_cq_progress(struct util_cq *cq)
{
	struct util_ep *ep;
	struct fid_list_entry *fid_entry;
	struct dlist_entry *item;

	fastlock_acquire(&cq->list_lock);
	dlist_foreach(&cq->list, item) {
		fid_entry = container_of(item, struct fid_list_entry, entry);
		ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
		ep->progress(ep);

	}
	fastlock_release(&cq->list_lock);
}

static void util_cq_read_ctx(void **dst, void *src)
{
	*(struct fi_cq_entry *) *dst = *(struct fi_cq_entry *) src;
	*dst += sizeof(struct fi_cq_entry);
}

static void util_cq_read_msg(void **dst, void *src)
{
	*(struct fi_cq_msg_entry *) *dst = *(struct fi_cq_msg_entry *) src;
	*dst += sizeof(struct fi_cq_msg_entry);
}

static void util_cq_read_data(void **dst, void *src)
{
	*(struct fi_cq_data_entry *) *dst = *(struct fi_cq_data_entry *) src;
	*dst += sizeof(struct fi_cq_data_entry);
}

static void util_cq_read_tagged(void **dst, void *src)
{
	util_cq_read_data(dst, src);
	((struct fi_cq_tagged_entry *) *dst)->tag = 0;
	*dst += sizeof(struct fi_cq_tagged_entry);
}

static ssize_t util_cq_read(struct fid_cq *cq_fid, void *buf, size_t count)
{
	struct util_cq *cq;
	struct fi_cq_data_entry *entry;
	ssize_t i;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	fastlock_acquire(&cq->cq_lock);
	if (cirque_isempty(cq->cirq)) {
		fastlock_release(&cq->cq_lock);
		util_cq_progress(cq);
		fastlock_acquire(&cq->cq_lock);
		if (cirque_isempty(cq->cirq)) {
			i = -FI_EAGAIN;
			goto out;
		}
	}

	if (count > cirque_usedcnt(cq->cirq))
		count = cirque_usedcnt(cq->cirq);

	for (i = 0; i < count; i++) {
		entry = cirque_head(cq->cirq);
		if (entry->flags & UTIL_FLAG_ERROR) {
			if (!i)
				i = -FI_EAVAIL;
			break;
		}
		cq->read_entry(&buf, entry);
		cirque_discard(cq->cirq);
	}
out:
	fastlock_release(&cq->cq_lock);
	return i;
}

static ssize_t util_cq_readfrom(struct fid_cq *cq_fid, void *buf,
				size_t count, fi_addr_t *src_addr)
{
	struct util_cq *cq;
	struct fi_cq_data_entry *entry;
	ssize_t i;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	if (!cq->src) {
		i = util_cq_read(cq_fid, buf, count);
		if (i > 0) {
			for (count = 0; count < i; count++)
				src_addr[i] = FI_ADDR_NOTAVAIL;
		}
		return i;
	}

	fastlock_acquire(&cq->cq_lock);
	if (cirque_isempty(cq->cirq)) {
		fastlock_release(&cq->cq_lock);
		util_cq_progress(cq);
		fastlock_acquire(&cq->cq_lock);
		if (cirque_isempty(cq->cirq)) {
			i = -FI_EAGAIN;
			goto out;
		}
	}

	if (count > cirque_usedcnt(cq->cirq))
		count = cirque_usedcnt(cq->cirq);

	for (i = 0; i < count; i++) {
		entry = cirque_head(cq->cirq);
		if (entry->flags & UTIL_FLAG_ERROR) {
			if (!i)
				i = -FI_EAVAIL;
			break;
		}
		src_addr[i] = cq->src[cirque_rindex(cq->cirq)];
		cq->read_entry(&buf, entry);
		cirque_discard(cq->cirq);
	}
out:
	fastlock_release(&cq->cq_lock);
	return i;
}

static ssize_t util_cq_readerr(struct fid_cq *cq_fid, struct fi_cq_err_entry *buf,
			       uint64_t flags)
{
	struct util_cq *cq;
	struct util_cq_err_entry *err;
	struct slist_entry *entry;
	ssize_t ret;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	fastlock_acquire(&cq->cq_lock);
	if (!cirque_isempty(cq->cirq) &&
	    (cirque_head(cq->cirq)->flags & UTIL_FLAG_ERROR)) {
		cirque_discard(cq->cirq);
		entry = slist_remove_head(&cq->err_list);
		err = container_of(entry, struct util_cq_err_entry, list_entry);
		*buf = err->err_entry;
		free(err);
		ret = 0;
	} else {
		ret = -FI_EAGAIN;
	}
	fastlock_release(&cq->cq_lock);
	return ret;
}

static ssize_t util_cq_sread(struct fid_cq *cq_fid, void *buf, size_t count,
			     const void *cond, int timeout)
{
	struct util_cq *cq;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	assert(cq->wait && cq->internal_wait);
	fi_wait(&cq->wait->wait_fid, timeout);
	return util_cq_read(cq_fid, buf, count);
}

static ssize_t util_cq_sreadfrom(struct fid_cq *cq_fid, void *buf, size_t count,
				 fi_addr_t *src_addr, const void *cond,
				 int timeout)
{
	struct util_cq *cq;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	assert(cq->wait && cq->internal_wait);
	fi_wait(&cq->wait->wait_fid, timeout);
	return util_cq_readfrom(cq_fid, buf, count, src_addr);
}

static int util_cq_signal(struct fid_cq *cq_fid)
{
	struct util_cq *cq;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	assert(cq->wait);
	cq->wait->signal(cq->wait);
	return 0;
}

static const char *util_cq_strerror(struct fid_cq *cq, int prov_errno,
				    const void *err_data, char *buf, size_t len)
{
	return fi_strerror(prov_errno);
}

static struct fi_ops_cq util_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = util_cq_read,
	.readfrom = util_cq_readfrom,
	.readerr = util_cq_readerr,
	.sread = util_cq_sread,
	.sreadfrom = util_cq_sreadfrom,
	.signal = util_cq_signal,
	.strerror = util_cq_strerror,
};

static int util_cq_close(struct fid *fid)
{
	struct util_cq *cq;
	int ret;

	cq = container_of(fid, struct util_cq, cq_fid.fid);
	ret = fi_cq_cleanup(cq);
	if (ret)
		return ret;

	util_comp_cirq_free(cq->cirq);
	free(cq->src);
	free(cq);
	return 0;
}

static struct fi_ops util_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = util_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int util_cq_init(struct fid_domain *domain, struct fi_cq_attr *attr,
			struct util_cq *cq, void *context)
{
	fi_cq_read_func read_func;
	int ret;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
	case FI_CQ_FORMAT_CONTEXT:
		read_func = util_cq_read_ctx;
		break;
	case FI_CQ_FORMAT_MSG:
		read_func = util_cq_read_msg;
		break;
	case FI_CQ_FORMAT_DATA:
		read_func = util_cq_read_data;
		break;
	case FI_CQ_FORMAT_TAGGED:
		read_func = util_cq_read_tagged;
		break;
	default:
		assert(0);
		return -FI_EINVAL;
	}

	ret = fi_cq_init(domain, attr, read_func, cq, context);
	if (ret)
		return ret;

	cq->cirq = util_comp_cirq_create(attr->size);
	if (!cq->cirq) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	if (cq->domain->caps & FI_SOURCE) {
		cq->src = calloc(cq->cirq->size, sizeof *cq->src);
		if (!cq->src) {
			ret = -FI_ENOMEM;
			goto err2;
		}
	}
	return 0;

err2:
	util_comp_cirq_free(cq->cirq);
err1:
	fi_cq_cleanup(cq);
	return ret;
}

int util_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	struct util_cq *cq;
	int ret;

	ret = fi_check_cq_attr(&udpx_prov, attr);
	if (ret)
		return ret;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	ret = util_cq_init(domain, attr, cq, context);
	if (ret) {
		free(cq);
		return ret;
	}

	cq->cq_fid.fid.ops = &util_cq_fi_ops;
	cq->cq_fid.ops = &util_cq_ops;

	ret = fi_cq_ready(cq);
	if (ret) {
		util_cq_close(&cq->cq_fid.fid);
		return ret;
	}

	*cq_fid = &cq->cq_fid;
	return 0;
}
