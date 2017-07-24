/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include <fi_enosys.h>
#include <fi_util.h>

int ofi_check_cntr_attr(const struct fi_provider *prov,
			const struct fi_cntr_attr *attr)
{
	if (!attr)
		return FI_SUCCESS;

        if (attr->flags) {
		FI_WARN(prov, FI_LOG_CNTR, "unsupported flags\n");
		return -FI_EINVAL;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		break;
	case FI_WAIT_SET:
		if (!attr->wait_set) {
			FI_WARN(prov, FI_LOG_CNTR, "invalid wait set\n");
			return -FI_EINVAL;
		}
		/* fall through */
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		break;
	default:
		FI_WARN(prov, FI_LOG_CNTR, "unsupported wait object\n");
		return -FI_EINVAL;
	}

	return 0;
}

static uint64_t ofi_cntr_read(struct fid_cntr *cntr_fid)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	cntr->progress(cntr);

	return ofi_atomic_get64(&cntr->cnt);
}

static uint64_t ofi_cntr_readerr(struct fid_cntr *cntr_fid)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	cntr->progress(cntr);

	return ofi_atomic_get64(&cntr->err);
}

static int ofi_cntr_add(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	ofi_atomic_add64(&cntr->cnt, value);

	if(cntr->wait)
		cntr->wait->signal(cntr->wait);

	return FI_SUCCESS;
}

static int ofi_cntr_adderr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	ofi_atomic_add64(&cntr->err, value);

	if(cntr->wait)
		cntr->wait->signal(cntr->wait);

	return FI_SUCCESS;
}

static int ofi_cntr_set(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	ofi_atomic_initialize64(&cntr->cnt, value);

	if(cntr->wait)
		cntr->wait->signal(cntr->wait);

	return FI_SUCCESS;
}

static int ofi_cntr_seterr(struct fid_cntr *cntr_fid, uint64_t value)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	ofi_atomic_initialize64(&cntr->err, value);

	if(cntr->wait)
		cntr->wait->signal(cntr->wait);

	return FI_SUCCESS;
}

static int ofi_cntr_wait(struct fid_cntr *cntr_fid, uint64_t threshold, int timeout)
{
	struct util_cntr *cntr = container_of(cntr_fid, struct util_cntr, cntr_fid);
	uint64_t current_ms;
	uint64_t finish_ms;
	uint64_t err = ofi_cntr_readerr(cntr_fid);

	assert(cntr->cntr_fid.fid.fclass == FI_CLASS_CNTR);

	if (threshold <= ofi_cntr_read(cntr_fid))
		return FI_SUCCESS;

	assert(cntr->wait);

	current_ms = fi_gettime_ms();
	finish_ms = (timeout < 0) ? UINT64_MAX : current_ms + timeout;
	for (; timeout < 0 || current_ms < finish_ms;
	    current_ms = fi_gettime_ms()) {
		timeout = timeout < 0 ? timeout : (int)(finish_ms - current_ms);
		fi_wait(&cntr->wait->wait_fid, timeout);
		cntr->progress(cntr);
		if (threshold <= ofi_atomic_get64(&cntr->cnt))
			return FI_SUCCESS;
		else if (err != ofi_atomic_get64(&cntr->err))
			return -FI_EAVAIL;
	}

	return -FI_ETIMEDOUT;
}

static struct fi_ops_cntr util_cntr_ops = {
	.size = sizeof(struct fi_ops_cntr),
	.read = ofi_cntr_read,
	.readerr = ofi_cntr_readerr,
	.add = ofi_cntr_add,
	.adderr = ofi_cntr_adderr,
	.set = ofi_cntr_set,
	.seterr = ofi_cntr_seterr,
	.wait = ofi_cntr_wait
};

int ofi_cntr_cleanup(struct util_cntr *cntr)
{
	if (ofi_atomic_get32(&cntr->ref))
		return -FI_EBUSY;

	fastlock_destroy(&cntr->ep_list_lock);

	if (cntr->wait) {
		fi_poll_del(&cntr->wait->pollset->poll_fid,
			    &cntr->cntr_fid.fid, 0);
	}

	ofi_atomic_dec32(&cntr->domain->ref);
	return 0;
}

static int util_cntr_close(struct fid *fid)
{
	struct util_cntr *cntr;
	int ret;

	cntr = container_of(fid, struct util_cntr, cntr_fid.fid);
	ret = ofi_cntr_cleanup(cntr);
	if (ret)
		return ret;
	return 0;
}

int ofi_check_bind_cntr_flags(struct util_ep *ep, struct util_cntr *cntr,
			      uint64_t flags)
{
	const struct fi_provider *prov = ep->domain->fabric->prov;

	if (flags & ~(FI_TRANSMIT | FI_RECV | FI_RECV  | FI_WRITE |
		      FI_REMOTE_READ | FI_REMOTE_WRITE)) {
		FI_WARN(prov, FI_LOG_EP_CTRL,
			"Unsupported flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && ep->tx_cntr) ||
	    ((flags & FI_RECV) && ep->rx_cntr) ||
	    ((flags & FI_READ) && ep->rd_cntr) ||
	    ((flags & FI_WRITE) && ep->wr_cntr) ||
	    ((flags & FI_REMOTE_READ) && ep->rem_rd_cntr) ||
	    ((flags & FI_REMOTE_WRITE) && ep->rem_wr_cntr)) {
		FI_WARN(prov, FI_LOG_EP_CTRL,
			"Duplicate CNTR binding\n");
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

static int fi_cntr_init(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct util_cntr *cntr, void *context)
{
	struct fi_wait_attr wait_attr;
	struct fid_wait *wait;
	int ret;

	cntr->domain = container_of(domain, struct util_domain, domain_fid);
	ofi_atomic_initialize32(&cntr->ref, 0);
	dlist_init(&cntr->ep_list);
	fastlock_init(&cntr->ep_list_lock);

	cntr->cntr_fid.fid.fclass = FI_CLASS_CNTR;
	cntr->cntr_fid.fid.context = context;

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
		wait = NULL;
		break;
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		memset(&wait_attr, 0, sizeof wait_attr);
		wait_attr.wait_obj = attr->wait_obj;
		ret = fi_wait_open(&cntr->domain->fabric->fabric_fid,
				   &wait_attr, &wait);
		if (ret)
			return ret;
		break;
	case FI_WAIT_SET:
		wait = attr->wait_set;
		break;
	default:
		assert(0);
		return -FI_EINVAL;
	}

	if (wait)
		cntr->wait = container_of(wait, struct util_wait, wait_fid);

	ofi_atomic_inc32(&cntr->domain->ref);
	return 0;
}

void ofi_cntr_progress(struct util_cntr *cntr)
{
	struct util_ep *ep;
	struct fid_list_entry *fid_entry;
	struct dlist_entry *item;

	fastlock_acquire(&cntr->ep_list_lock);
	dlist_foreach(&cntr->ep_list, item) {
		fid_entry = container_of(item, struct fid_list_entry, entry);
		ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
		ep->progress(ep);
	}
	fastlock_release(&cntr->ep_list_lock);
}

static struct fi_ops util_cntr_fi_ops = {
	.size = sizeof(util_cntr_fi_ops),
	.close = util_cntr_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

int ofi_cntr_init(const struct fi_provider *prov, struct fid_domain *domain,
		  struct fi_cntr_attr *attr, struct util_cntr *cntr,
		  ofi_cntr_progress_func progress, void *context)
{
	int ret;

	assert(progress);
	ret = ofi_check_cntr_attr(prov, attr);
	if (ret)
		return ret;

	cntr->cntr_fid.fid.ops = &util_cntr_fi_ops;
	cntr->cntr_fid.ops = &util_cntr_ops;
	cntr->progress = progress;

	ret = fi_cntr_init(domain, attr, cntr, context);
	if (ret)
		return ret;

	/* CNTR must be fully operational before adding to wait set */
	if (cntr->wait) {
		ret = fi_poll_add(&cntr->wait->pollset->poll_fid,
				  &cntr->cntr_fid.fid, 0);
		if (ret) {
			ofi_cntr_cleanup(cntr);
			return ret;
		}
	}

	return 0;
}

