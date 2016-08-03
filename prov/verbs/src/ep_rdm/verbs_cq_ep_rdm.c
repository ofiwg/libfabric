/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#include <fi_enosys.h>
#include <fi_list.h>
#include "../fi_verbs.h"
#include "verbs_queuing.h"


struct util_buf_pool *fi_ibv_rdm_tagged_request_pool;
struct util_buf_pool *fi_ibv_rdm_postponed_pool;

/*
 * extra buffer size equal eager buffer size, it is used for any intermediate
 * needs like unexpected recv, pack/unpack noncontig messages, etc
 */
struct util_buf_pool *fi_ibv_rdm_tagged_extra_buffers_pool;

static ssize_t fi_ibv_rdm_tagged_cq_readfrom(struct fid_cq *cq, void *buf,
                                             size_t count, fi_addr_t * src_addr)
{
	struct fi_ibv_rdm_cq *_cq = 
		container_of(cq, struct fi_ibv_rdm_cq, cq_fid);
	struct fi_cq_tagged_entry *entry = buf;
	struct fi_ibv_rdm_tagged_request *cq_entry;
	size_t ret = 0;

	for (cq_entry = count ? fi_ibv_rdm_take_first_from_cq() : NULL;
	     cq_entry;
	     cq_entry = (ret < count) ? fi_ibv_rdm_take_first_from_cq() : NULL)
	{
		FI_DBG(&fi_ibv_prov, FI_LOG_CQ,
			"\t\t-> found in ready: %p op_ctx %p, len %lu, tag 0x%llx\n",
			cq_entry, cq_entry->context, cq_entry->len,
			cq_entry->minfo.tag);

		src_addr[ret] = (fi_addr_t) (uintptr_t) cq_entry->minfo.conn;
		entry[ret].op_context = cq_entry->context;
		entry[ret].flags = (cq_entry->comp_flags & ~FI_COMPLETION);
		entry[ret].len = cq_entry->len;
		entry[ret].data = cq_entry->imm;
		entry[ret].tag = cq_entry->minfo.tag;

		if (cq_entry->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
			FI_IBV_RDM_DBG_REQUEST("to_pool: ", cq_entry, 
						FI_LOG_DEBUG);
			util_buf_release(fi_ibv_rdm_tagged_request_pool, cq_entry);
		} else {
			cq_entry->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		}
		ret++;
	};

	if (ret == 0) {
		if (fi_ibv_rdm_tagged_poll(_cq->ep) < 0) {
			VERBS_INFO(FI_LOG_CQ, "fi_ibv_rdm_tagged_poll failed\n");
		}

		if (!dlist_empty(&fi_ibv_rdm_comp_queue.request_errcq)) {
			ret = -FI_EAVAIL;
		}
	}

	return ret;
}

static ssize_t fi_ibv_rdm_tagged_cq_read(struct fid_cq *cq, void *buf,
                                         size_t count)
{
	struct fi_ibv_rdm_cq *_cq =
		container_of(cq, struct fi_ibv_rdm_cq, cq_fid);
	const size_t _count = _cq->ep->n_buffs;
	fi_addr_t addr[_count];

	return fi_ibv_rdm_tagged_cq_readfrom(cq, buf, MIN(_count, count), addr);
}

ssize_t fi_ibv_rdm_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
				fi_addr_t *src_addr, const void *cond,
				int timeout)
{
	size_t threshold = count;
	uint64_t time_limit = fi_gettime_ms() + timeout;
	size_t counter = 0;
	ssize_t ret = 0;

	struct fi_ibv_rdm_cq *_cq = container_of(cq, struct fi_ibv_rdm_cq, cq_fid);
	switch (_cq->wait_cond) {
	case FI_CQ_COND_THRESHOLD:
		threshold = MIN((uintptr_t) cond, threshold);
	case FI_CQ_COND_NONE:
		break;
	default:
		assert(0);
		return -FI_EOTHER;
	}

	do {
		ret = fi_ibv_rdm_tagged_cq_readfrom(cq, buf, threshold,
						    src_addr);
		counter += (ret > 0) ? ret : 0;
	} while ((ret >= 0) && (counter < threshold ||
		(timeout >= 0 && fi_gettime_ms() < time_limit)));

	if (counter != 0 && ret >= 0) {
		ret = counter;
	} else if (ret == 0) {
		ret = -FI_EAGAIN;
	}

	return ret;
}

ssize_t fi_ibv_rdm_cq_sread(struct fid_cq *cq, void *buf, size_t count,
			    const void *cond, int timeout)
{
	struct fi_ibv_rdm_cq *_cq =
		container_of(cq, struct fi_ibv_rdm_cq, cq_fid);
	size_t chunk		= MIN(_cq->ep->n_buffs, count);
	uint64_t time_limit	= fi_gettime_ms() + timeout;
	size_t rest		= count;
	fi_addr_t addr[chunk];
	ssize_t	ret;

	do {
		ret = fi_ibv_rdm_cq_sreadfrom(cq, buf, chunk, addr, cond,
					      timeout);
		if (ret > 0) {
			rest -= ret;
			chunk = MIN(chunk, rest);
		}
	} while (((ret >=0) && (rest != 0)) ||
		 (timeout >= 0 && fi_gettime_ms() < time_limit));

	return (count != rest) ? (count - rest) : ret;
}

static const char *
fi_ibv_rdm_cq_strerror(struct fid_cq *eq, int prov_errno, const void *err_data,
		       char *buf, size_t len)
{
	/* TODO: */
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

static ssize_t
fi_ibv_rdm_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *entry,
                             uint64_t flags)
{
	ssize_t ret = 0;
	struct fi_ibv_rdm_tagged_request *err_request = 
		fi_ibv_rdm_take_first_from_errcq();

	if (err_request) {
		entry->op_context = err_request->context;
		entry->flags = (err_request->comp_flags & ~FI_COMPLETION);
		entry->len = err_request->len;
		entry->buf = err_request->unexp_rbuf;
		entry->data = err_request->imm;
		entry->tag = err_request->minfo.tag;
		entry->olen = -1; /* TODO: */
		entry->err = err_request->state.err;
		entry->prov_errno = -err_request->state.err;
		entry->err_data = NULL;

		if (err_request->state.eager == FI_IBV_STATE_EAGER_READY_TO_FREE) {
			FI_IBV_RDM_DBG_REQUEST("to_pool: ", err_request,
						FI_LOG_DEBUG);
			util_buf_release(fi_ibv_rdm_tagged_request_pool,
				err_request);
		} else {
			err_request->state.eager = FI_IBV_STATE_EAGER_READY_TO_FREE;
		}

		ret++;
	}

	return ret;
}

static struct fi_ops_cq fi_ibv_rdm_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_rdm_tagged_cq_read,
	.readfrom = fi_ibv_rdm_tagged_cq_readfrom,
	.readerr = fi_ibv_rdm_cq_readerr,
	.sread = fi_ibv_rdm_cq_sread,
	.sreadfrom = fi_ibv_rdm_cq_sreadfrom,
	.strerror = fi_ibv_rdm_cq_strerror
};

static int fi_ibv_rdm_cq_close(fid_t fid)
{
	struct fi_ibv_rdm_cq *cq =
		container_of(fid, struct fi_ibv_rdm_cq, cq_fid.fid);

	if(cq->ep) {
		return -FI_EBUSY;
	}

	/* TODO: move queues & related pools cleanup from close EP */
	/* fi_ibv_rdm_clean_queues(); */

	free(cq);

	return FI_SUCCESS;
}

static struct fi_ops fi_ibv_rdm_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_rdm_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};


int fi_ibv_rdm_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq, void *context)
{
	struct fi_ibv_rdm_cq *_cq;
	int ret;

	_cq = calloc(1, sizeof *_cq);
	if (!_cq)
		return -FI_ENOMEM;

	_cq->domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
		break;
	case FI_WAIT_SET:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
	default:
		assert(0);
		ret = -FI_ENOSYS;
		goto err;
	}

	_cq->flags |= attr->flags;
	_cq->wait_cond = attr->wait_cond;
	_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	_cq->cq_fid.fid.context = context;
	_cq->cq_fid.fid.ops = &fi_ibv_rdm_cq_fi_ops;
	_cq->cq_fid.ops = &fi_ibv_rdm_cq_ops;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
		assert(0);
		break;
	case FI_CQ_FORMAT_TAGGED:
		assert(_cq->domain->rdm);
		_cq->entry_size = sizeof(struct fi_cq_tagged_entry);
		break;
	default:
		ret = -FI_ENOSYS;
		goto err;
	}

	dlist_init(&fi_ibv_rdm_comp_queue.request_cq);
	dlist_init(&fi_ibv_rdm_comp_queue.request_errcq);

	*cq = &_cq->cq_fid;
	return 0;

err:
	free(_cq);
	return ret;
}
