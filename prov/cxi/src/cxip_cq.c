
/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>

#include <ofi_list.h>
#include <ofi.h>

#include "cxip.h"

#define CXI_LOG_DBG(...) _CXI_LOG_DBG(FI_LOG_CQ, __VA_ARGS__)
#define CXI_LOG_ERROR(...) _CXI_LOG_ERROR(FI_LOG_CQ, __VA_ARGS__)

static struct cxi_req *cxix_cq_req_find(struct cxi_cq *cq, int id)
{
	return ofi_idx_at(&cq->req_table, id);
}

struct cxi_req *cxix_cq_req_alloc(struct cxi_cq *cq, int remap)
{
	struct cxi_req *req;

	fastlock_acquire(&cq->req_lock);

	req = (struct cxi_req *)util_buf_alloc(cq->req_pool);
	if (!req) {
		CXI_LOG_ERROR("Failed to allocate request\n");
		goto out;
	}

	if (remap) {
		req->req_id = ofi_idx_insert(&cq->req_table, req);

		/* Target command buffer IDs are 16 bits wide. */
		if (req->req_id < 0 || req->req_id >= (1 << 16)) {
			CXI_LOG_ERROR("Failed to map request: %d\n",
				      req->req_id);
			util_buf_release(cq->req_pool, req);
			req = NULL;
			goto out;
		}
	} else {
		req->req_id = -1;
	}

	req->cq = cq;

out:
	fastlock_release(&cq->req_lock);

	return req;
}

void cxix_cq_req_free(struct cxi_req *req)
{
	struct cxi_req *table_req;
	struct cxi_cq *cq = req->cq;

	fastlock_acquire(&cq->req_lock);

	if (req->req_id >= 0) {
		table_req = (struct cxi_req *)
				ofi_idx_remove(&req->cq->req_table,
					       req->req_id);
		if (table_req != req)
			CXI_LOG_ERROR("Failed to free request\n");
	}

	util_buf_release(req->cq->req_pool, req);

	fastlock_release(&cq->req_lock);
}

void cxi_cq_add_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx)
{
	struct dlist_entry *entry;
	struct cxi_tx_ctx *curr_ctx;

	fastlock_acquire(&cq->list_lock);
	for (entry = cq->tx_list.next; entry != &cq->tx_list;
	     entry = entry->next) {
		curr_ctx = container_of(entry, struct cxi_tx_ctx, cq_entry);
		if (tx_ctx == curr_ctx)
			goto out;
	}
	dlist_insert_tail(&tx_ctx->cq_entry, &cq->tx_list);
	ofi_atomic_inc32(&cq->ref);
out:
	fastlock_release(&cq->list_lock);
}

void cxi_cq_remove_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx)
{
	fastlock_acquire(&cq->list_lock);
	dlist_remove(&tx_ctx->cq_entry);
	ofi_atomic_dec32(&cq->ref);
	fastlock_release(&cq->list_lock);
}

void cxi_cq_add_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx)
{
	struct dlist_entry *entry;
	struct cxi_rx_ctx *curr_ctx;

	fastlock_acquire(&cq->list_lock);

	for (entry = cq->rx_list.next; entry != &cq->rx_list;
	     entry = entry->next) {
		curr_ctx = container_of(entry, struct cxi_rx_ctx, cq_entry);
		if (rx_ctx == curr_ctx)
			goto out;
	}
	dlist_insert_tail(&rx_ctx->cq_entry, &cq->rx_list);
	ofi_atomic_inc32(&cq->ref);
out:
	fastlock_release(&cq->list_lock);
}

void cxi_cq_remove_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx)
{
	fastlock_acquire(&cq->list_lock);
	dlist_remove(&rx_ctx->cq_entry);
	ofi_atomic_dec32(&cq->ref);
	fastlock_release(&cq->list_lock);
}

static struct cxi_req *cxix_cq_event_req(const union c_event *event)
{
	switch (event->event_type) {
	case C_EVENT_ACK:
		return (struct cxi_req *)event->init_short.user_ptr;
	}

	CXI_LOG_ERROR("Invalid event type: %d\n", event->event_type);
	return NULL;
}

/* Caller must hold the cq->lock. */
void cxi_cq_progress(struct cxi_cq *cq)
{
	const union c_event *event;
	struct cxi_req *req;

	if (!cq->enabled)
		return;

	/* TODO Limit the maximum number of events processed */
	while ((event = cxi_eq_get_event(cq->evtq))) {
		req = cxix_cq_event_req(event);
		if (req)
			req->cb(req, event);

		cxi_eq_ack_events(cq->evtq);
	}
}

static ssize_t cxi_cq_entry_size(struct cxi_cq *cxi_cq)
{
	ssize_t size;

	switch (cxi_cq->attr.format) {
	case FI_CQ_FORMAT_CONTEXT:
		size = sizeof(struct fi_cq_entry);
		break;

	case FI_CQ_FORMAT_MSG:
		size = sizeof(struct fi_cq_msg_entry);
		break;

	case FI_CQ_FORMAT_DATA:
		size = sizeof(struct fi_cq_data_entry);
		break;

	case FI_CQ_FORMAT_TAGGED:
		size = sizeof(struct fi_cq_tagged_entry);
		break;

	case FI_CQ_FORMAT_UNSPEC:
	default:
		size = -1;
		CXI_LOG_ERROR("Invalid CQ format\n");
		break;
	}
	return size;
}

/* Caller must hold the cq->lock.  This is true for all EQE callbacks. */
static ssize_t _cxi_cq_write(struct cxi_cq *cq, fi_addr_t addr,
			     const void *buf, size_t len)
{
	ssize_t ret;
	struct cxi_cq_overflow_entry_t *overflow_entry;

	if (ofi_rbfdavail(&cq->cq_rbfd) < len) {
		CXI_LOG_ERROR("Not enough space in CQ\n");
		overflow_entry = calloc(1, sizeof(*overflow_entry) + len);
		if (!overflow_entry) {
			ret = -FI_ENOSPC;
			goto out;
		}

		memcpy(&overflow_entry->cq_entry[0], buf, len);
		overflow_entry->len = len;
		overflow_entry->addr = addr;
		dlist_insert_tail(&overflow_entry->entry, &cq->overflow_list);
		ret = len;
		goto out;
	}


	ofi_rbwrite(&cq->addr_rb, &addr, sizeof(addr));
	ofi_rbcommit(&cq->addr_rb);

	ofi_rbfdwrite(&cq->cq_rbfd, buf, len);
	if (cq->domain->progress_mode == FI_PROGRESS_MANUAL)
		ofi_rbcommit(&cq->cq_rbfd.rb);
	else
		ofi_rbfdcommit(&cq->cq_rbfd);

	ret = len;

	if (cq->signal)
		cxi_wait_signal(cq->waitset);
out:
	return ret;
}

static int cxi_cq_report_context(struct cxi_cq *cq, fi_addr_t addr,
				 struct cxi_req *req)
{
	struct fi_cq_entry cq_entry;

	cq_entry.op_context = (void *) (uintptr_t) req->context;

	return _cxi_cq_write(cq, addr, &cq_entry, sizeof(cq_entry));
}

static uint64_t cxi_cq_sanitize_flags(uint64_t flags)
{
	return (flags & (FI_SEND | FI_RECV | FI_RMA | FI_ATOMIC |
				FI_MSG | FI_TAGGED |
				FI_READ | FI_WRITE |
				FI_REMOTE_READ | FI_REMOTE_WRITE |
				FI_REMOTE_CQ_DATA | FI_MULTI_RECV));
}

static int cxi_cq_report_msg(struct cxi_cq *cq, fi_addr_t addr,
			     struct cxi_req *req)
{
	struct fi_cq_msg_entry cq_entry;

	cq_entry.op_context = (void *) (uintptr_t) req->context;
	cq_entry.flags = cxi_cq_sanitize_flags(req->flags);
	cq_entry.len = req->data_len;

	return _cxi_cq_write(cq, addr, &cq_entry, sizeof(cq_entry));
}

static int cxi_cq_report_data(struct cxi_cq *cq, fi_addr_t addr,
			      struct cxi_req *req)
{
	struct fi_cq_data_entry cq_entry;

	cq_entry.op_context = (void *) (uintptr_t) req->context;
	cq_entry.flags = cxi_cq_sanitize_flags(req->flags);
	cq_entry.len = req->data_len;
	cq_entry.buf = (void *) (uintptr_t) req->buf;
	cq_entry.data = req->data;

	return _cxi_cq_write(cq, addr, &cq_entry, sizeof(cq_entry));
}

static int cxi_cq_report_tagged(struct cxi_cq *cq, fi_addr_t addr,
				struct cxi_req *req)
{
	struct fi_cq_tagged_entry cq_entry;

	cq_entry.op_context = (void *) (uintptr_t) req->context;
	cq_entry.flags = cxi_cq_sanitize_flags(req->flags);
	cq_entry.len = req->data_len;
	cq_entry.buf = (void *) (uintptr_t) req->buf;
	cq_entry.data = req->data;
	cq_entry.tag = req->tag;

	return _cxi_cq_write(cq, addr, &cq_entry, sizeof(cq_entry));
}

static void cxi_cq_set_report_fn(struct cxi_cq *cxi_cq)
{
	switch (cxi_cq->attr.format) {
	case FI_CQ_FORMAT_CONTEXT:
		cxi_cq->report_completion = &cxi_cq_report_context;
		break;

	case FI_CQ_FORMAT_MSG:
		cxi_cq->report_completion = &cxi_cq_report_msg;
		break;

	case FI_CQ_FORMAT_DATA:
		cxi_cq->report_completion = &cxi_cq_report_data;
		break;

	case FI_CQ_FORMAT_TAGGED:
		cxi_cq->report_completion = &cxi_cq_report_tagged;
		break;

	case FI_CQ_FORMAT_UNSPEC:
	default:
		CXI_LOG_ERROR("Invalid CQ format\n");
		break;
	}
}

static inline void cxi_cq_copy_overflow_list(struct cxi_cq *cq, size_t count)
{
	size_t i;
	struct cxi_cq_overflow_entry_t *overflow_entry;

	for (i = 0; i < count && !dlist_empty(&cq->overflow_list); i++) {
		overflow_entry = container_of(cq->overflow_list.next,
					      struct cxi_cq_overflow_entry_t,
					      entry);
		ofi_rbwrite(&cq->addr_rb, &overflow_entry->addr,
			    sizeof(fi_addr_t));
		ofi_rbcommit(&cq->addr_rb);

		ofi_rbfdwrite(&cq->cq_rbfd, &overflow_entry->cq_entry[0],
			      overflow_entry->len);
		if (cq->domain->progress_mode == FI_PROGRESS_MANUAL)
			ofi_rbcommit(&cq->cq_rbfd.rb);
		else
			ofi_rbfdcommit(&cq->cq_rbfd);

		dlist_remove(&overflow_entry->entry);
		free(overflow_entry);
	}
}

static inline ssize_t cxi_cq_rbuf_read(struct cxi_cq *cq, void *buf,
				       size_t count, fi_addr_t *src_addr,
				       size_t cq_entry_len)
{
	size_t i;
	fi_addr_t addr;

	ofi_rbfdread(&cq->cq_rbfd, buf, cq_entry_len * count);
	for (i = 0; i < count; i++) {
		ofi_rbread(&cq->addr_rb, &addr, sizeof(addr));
		if (src_addr)
			src_addr[i] = addr;
	}
	cxi_cq_copy_overflow_list(cq, count);
	return count;
}

static ssize_t cxi_cq_sreadfrom(struct fid_cq *cq, void *buf, size_t count,
				fi_addr_t *src_addr, const void *cond,
				int timeout)
{
	int ret = 0;
	size_t threshold;
	struct cxi_cq *cxi_cq;
	uint64_t start_ms;
	ssize_t cq_entry_len, avail;

	cxi_cq = container_of(cq, struct cxi_cq, cq_fid);
	if (ofi_rbused(&cxi_cq->cqerr_rb))
		return -FI_EAVAIL;

	cq_entry_len = cxi_cq->cq_entry_size;
	if (cxi_cq->attr.wait_cond == FI_CQ_COND_THRESHOLD)
		threshold = MIN((uintptr_t) cond, count);
	else
		threshold = count;

	start_ms = (timeout >= 0) ? fi_gettime_ms() : 0;

	if (cxi_cq->domain->progress_mode == FI_PROGRESS_MANUAL) {
		while (1) {
			fastlock_acquire(&cxi_cq->lock);

			cxi_cq_progress(cxi_cq);

			avail = ofi_rbfdused(&cxi_cq->cq_rbfd);
			if (avail) {
				ret = cxi_cq_rbuf_read(cxi_cq, buf,
					MIN(threshold,
					    (size_t)(avail / cq_entry_len)),
					src_addr, cq_entry_len);
			}
			fastlock_release(&cxi_cq->lock);
			if (ret)
				return ret;

			if (timeout >= 0) {
				timeout -= (int) (fi_gettime_ms() - start_ms);
				if (timeout <= 0)
					return -FI_EAGAIN;
			}

			if (ofi_atomic_get32(&cxi_cq->signaled)) {
				ofi_atomic_set32(&cxi_cq->signaled, 0);
				return -FI_ECANCELED;
			}
		};
	} else {
		do {
			fastlock_acquire(&cxi_cq->lock);
			ret = 0;
			avail = ofi_rbfdused(&cxi_cq->cq_rbfd);
			if (avail) {
				ret = cxi_cq_rbuf_read(cxi_cq, buf,
					MIN(threshold,
					    (size_t)(avail / cq_entry_len)),
					src_addr, cq_entry_len);
			} else {
				ofi_rbfdreset(&cxi_cq->cq_rbfd);
			}
			fastlock_release(&cxi_cq->lock);
			if (ret && ret != -FI_EAGAIN)
				return ret;

			if (timeout >= 0) {
				timeout -= (int) (fi_gettime_ms() - start_ms);
				if (timeout <= 0)
					return -FI_EAGAIN;
			}

			if (ofi_atomic_get32(&cxi_cq->signaled)) {
				ofi_atomic_set32(&cxi_cq->signaled, 0);
				return -FI_ECANCELED;
			}
			ret = ofi_rbfdwait(&cxi_cq->cq_rbfd, timeout);
		} while (ret > 0);
	}

	return (ret == 0 || ret == -FI_ETIMEDOUT) ? -FI_EAGAIN : ret;
}

static ssize_t cxi_cq_sread(struct fid_cq *cq, void *buf, size_t len,
			    const void *cond, int timeout)
{
	return cxi_cq_sreadfrom(cq, buf, len, NULL, cond, timeout);
}

static ssize_t cxi_cq_readfrom(struct fid_cq *cq, void *buf, size_t count,
			       fi_addr_t *src_addr)
{
	return cxi_cq_sreadfrom(cq, buf, count, src_addr, NULL, -1);
}

static ssize_t cxi_cq_read(struct fid_cq *cq, void *buf, size_t count)
{
	return cxi_cq_readfrom(cq, buf, count, NULL);
}

static ssize_t cxi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
			      uint64_t flags)
{
	struct cxi_cq *cxi_cq;
	ssize_t ret;
	struct fi_cq_err_entry entry;
	uint32_t api_version;
	size_t err_data_size = 0;
	void *err_data = NULL;

	if (cq == NULL || buf == NULL)
		return -FI_EINVAL;

	cxi_cq = container_of(cq, struct cxi_cq, cq_fid);

	fastlock_acquire(&cxi_cq->lock);

	if (cxi_cq->domain->progress_mode == FI_PROGRESS_MANUAL)
		cxi_cq_progress(cxi_cq);

	if (ofi_rbused(&cxi_cq->cqerr_rb) >= sizeof(struct fi_cq_err_entry)) {
		api_version = cxi_cq->domain->fab->fab_fid.api_version;
		ofi_rbread(&cxi_cq->cqerr_rb, &entry, sizeof(entry));

		if ((FI_VERSION_GE(api_version, FI_VERSION(1, 5)))
			&& buf->err_data && buf->err_data_size) {
			err_data = buf->err_data;
			err_data_size = buf->err_data_size;
			*buf = entry;
			buf->err_data = err_data;

			/* Fill provided user's buffer */
			buf->err_data_size = MIN(entry.err_data_size,
						 err_data_size);
			memcpy(buf->err_data, entry.err_data,
			       buf->err_data_size);
		} else {
			memcpy(buf, &entry, sizeof(struct fi_cq_err_entry_1_0));
		}

		ret = 1;
	} else {
		ret = -FI_EAGAIN;
	}
	fastlock_release(&cxi_cq->lock);
	return ret;
}

static const char *cxi_cq_strerror(struct fid_cq *cq, int prov_errno,
				   const void *err_data, char *buf, size_t len)
{
	if (buf && len) {
		strncpy(buf, fi_strerror(-prov_errno), len);
		buf[len-1] = '\0';
		return buf;
	}

	return fi_strerror(-prov_errno);
}

int cxix_cq_enable(struct cxi_cq *cxi_cq)
{
	struct cxi_eq_alloc_opts evtq_opts;
	int ret = FI_SUCCESS;

	fastlock_acquire(&cxi_cq->lock);

	if (cxi_cq->enabled)
		goto unlock;

	/* TODO set EVTQ size with CQ attrs */
	evtq_opts.count = 1024;
	evtq_opts.reserved_fc = 1;

	ret = cxil_alloc_evtq(cxi_cq->domain->dev_if->if_lni, &evtq_opts,
			      &cxi_cq->evtq);
	if (ret != FI_SUCCESS) {
		CXI_LOG_DBG("Unable to allocate EVTQ, ret: %d\n", ret);
		ret = -FI_EDOMAIN;
		goto unlock;
	}

	/* TODO set buffer pool size with CQ attrs */
	ret = util_buf_pool_create(&cxi_cq->req_pool, sizeof(struct cxi_req),
				   8, 0, 64);
	if (ret) {
		ret = -FI_ENOMEM;
		goto free_evtq;
	}

	memset(&cxi_cq->req_table, 0, sizeof(cxi_cq->req_table));

	cxi_cq->enabled = 1;
	fastlock_release(&cxi_cq->lock);

	return FI_SUCCESS;

free_evtq:
	cxil_destroy_evtq(cxi_cq->evtq);
unlock:
	fastlock_release(&cxi_cq->lock);

	return ret;
}

static void cxix_cq_disable(struct cxi_cq *cxi_cq)
{
	fastlock_acquire(&cxi_cq->lock);

	if (!cxi_cq->enabled)
		goto unlock;

	util_buf_pool_destroy(cxi_cq->req_pool);

	cxil_destroy_evtq(cxi_cq->evtq);

	cxi_cq->enabled = 0;
unlock:
	fastlock_release(&cxi_cq->lock);
}

static int cxi_cq_close(struct fid *fid)
{
	struct cxi_cq *cq;

	cq = container_of(fid, struct cxi_cq, cq_fid.fid);
	if (ofi_atomic_get32(&cq->ref))
		return -FI_EBUSY;

	cxix_cq_disable(cq);

	if (cq->signal && cq->attr.wait_obj == FI_WAIT_MUTEX_COND)
		cxi_wait_close(&cq->waitset->fid);

	ofi_rbfree(&cq->addr_rb);
	ofi_rbfree(&cq->cqerr_rb);
	ofi_rbfdfree(&cq->cq_rbfd);

	fastlock_destroy(&cq->lock);
	fastlock_destroy(&cq->list_lock);
	fastlock_destroy(&cq->req_lock);
	ofi_atomic_dec32(&cq->domain->ref);

	free(cq);

	return 0;
}

static int cxi_cq_signal(struct fid_cq *cq)
{
	struct cxi_cq *cxi_cq;

	cxi_cq = container_of(cq, struct cxi_cq, cq_fid);

	ofi_atomic_set32(&cxi_cq->signaled, 1);
	fastlock_acquire(&cxi_cq->lock);
	ofi_rbfdsignal(&cxi_cq->cq_rbfd);
	fastlock_release(&cxi_cq->lock);
	return 0;
}

static struct fi_ops_cq cxi_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = cxi_cq_read,
	.readfrom = cxi_cq_readfrom,
	.readerr = cxi_cq_readerr,
	.sread = cxi_cq_sread,
	.sreadfrom = cxi_cq_sreadfrom,
	.signal = cxi_cq_signal,
	.strerror = cxi_cq_strerror,
};

static int cxi_cq_control(struct fid *fid, int command, void *arg)
{
	struct cxi_cq *cq;
	int ret = 0;

	cq = container_of(fid, struct cxi_cq, cq_fid);
	switch (command) {
	case FI_GETWAIT:
		if (cq->domain->progress_mode == FI_PROGRESS_MANUAL)
			return -FI_ENOSYS;

		switch (cq->attr.wait_obj) {
		case FI_WAIT_NONE:
		case FI_WAIT_FD:
		case FI_WAIT_UNSPEC:
			memcpy(arg, &cq->cq_rbfd.fd[OFI_RB_READ_FD],
			       sizeof(int));
			break;

		case FI_WAIT_SET:
		case FI_WAIT_MUTEX_COND:
			cxi_wait_get_obj(cq->waitset, arg);
			break;

		default:
			ret = -FI_EINVAL;
			break;
		}
		break;

	default:
		ret =  -FI_EINVAL;
		break;
	}

	return ret;
}

static struct fi_ops cxi_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = cxi_cq_close,
	.bind = fi_no_bind,
	.control = cxi_cq_control,
	.ops_open = fi_no_ops_open,
};

static int cxi_cq_verify_attr(struct fi_cq_attr *attr)
{
	if (!attr)
		return 0;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
	case FI_CQ_FORMAT_MSG:
	case FI_CQ_FORMAT_DATA:
	case FI_CQ_FORMAT_TAGGED:
		break;
	case FI_CQ_FORMAT_UNSPEC:
		attr->format = FI_CQ_FORMAT_CONTEXT;
		break;
	default:
		return -FI_ENOSYS;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_FD:
	case FI_WAIT_MUTEX_COND:
		break;
	case FI_WAIT_UNSPEC:
		attr->wait_obj = FI_WAIT_FD;
		break;
	case FI_WAIT_SET:
	default:
		return -FI_ENOSYS;
	}

	return 0;
}

static struct fi_cq_attr _cxi_cq_def_attr = {
	.size = CXI_CQ_DEF_SZ,
	.flags = 0,
	.format = FI_CQ_FORMAT_CONTEXT,
	.wait_obj = FI_WAIT_FD,
	.signaling_vector = 0,
	.wait_cond = FI_CQ_COND_NONE,
	.wait_set = NULL,
};

int cxi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context)
{
	struct cxi_domain *cxi_dom;
	struct cxi_cq *cxi_cq;
	struct fi_wait_attr wait_attr;
	struct cxi_fid_list *list_entry;
	struct cxi_wait *wait;
	int ret;

	if (cq == NULL)
		return -FI_EINVAL;

	cxi_dom = container_of(domain, struct cxi_domain, dom_fid);
	ret = cxi_cq_verify_attr(attr);
	if (ret)
		return ret;

	cxi_cq = calloc(1, sizeof(*cxi_cq));
	if (!cxi_cq)
		return -FI_ENOMEM;

	ofi_atomic_initialize32(&cxi_cq->ref, 0);
	ofi_atomic_initialize32(&cxi_cq->signaled, 0);
	cxi_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	cxi_cq->cq_fid.fid.context = context;
	cxi_cq->cq_fid.fid.ops = &cxi_cq_fi_ops;
	cxi_cq->cq_fid.ops = &cxi_cq_ops;

	if (attr == NULL) {
		cxi_cq->attr = _cxi_cq_def_attr;
	} else {
		cxi_cq->attr = *attr;
		if (attr->size == 0)
			cxi_cq->attr.size = _cxi_cq_def_attr.size;
	}

	cxi_cq->domain = cxi_dom;
	cxi_cq->cq_entry_size = cxi_cq_entry_size(cxi_cq);
	cxi_cq_set_report_fn(cxi_cq);

	dlist_init(&cxi_cq->tx_list);
	dlist_init(&cxi_cq->rx_list);
	dlist_init(&cxi_cq->ep_list);
	dlist_init(&cxi_cq->overflow_list);

	ret = ofi_rbfdinit(&cxi_cq->cq_rbfd, cxi_cq->attr.size *
			cxi_cq->cq_entry_size);
	if (ret)
		goto err1;

	ret = ofi_rbinit(&cxi_cq->addr_rb,
			cxi_cq->attr.size * sizeof(fi_addr_t));
	if (ret)
		goto err2;

	ret = ofi_rbinit(&cxi_cq->cqerr_rb, cxi_cq->attr.size *
			sizeof(struct fi_cq_err_entry));
	if (ret)
		goto err3;

	fastlock_init(&cxi_cq->lock);

	switch (cxi_cq->attr.wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		break;

	case FI_WAIT_MUTEX_COND:
		wait_attr.flags = 0;
		wait_attr.wait_obj = FI_WAIT_MUTEX_COND;
		ret = cxi_wait_open(&cxi_dom->fab->fab_fid, &wait_attr,
				     &cxi_cq->waitset);
		if (ret) {
			ret = -FI_EINVAL;
			goto err4;
		}
		cxi_cq->signal = 1;
		break;

	case FI_WAIT_SET:
		if (!attr) {
			ret = -FI_EINVAL;
			goto err4;
		}

		cxi_cq->waitset = attr->wait_set;
		cxi_cq->signal = 1;
		wait = container_of(attr->wait_set, struct cxi_wait, wait_fid);
		list_entry = calloc(1, sizeof(*list_entry));
		if (!list_entry) {
			ret = -FI_ENOMEM;
			goto err4;
		}
		dlist_init(&list_entry->entry);
		list_entry->fid = &cxi_cq->cq_fid.fid;
		dlist_insert_after(&list_entry->entry, &wait->fid_list);
		break;

	default:
		break;
	}

	*cq = &cxi_cq->cq_fid;
	ofi_atomic_inc32(&cxi_dom->ref);
	fastlock_init(&cxi_cq->list_lock);
	fastlock_init(&cxi_cq->req_lock);

	return 0;

err4:
	ofi_rbfree(&cxi_cq->cqerr_rb);
err3:
	ofi_rbfree(&cxi_cq->addr_rb);
err2:
	ofi_rbfdfree(&cxi_cq->cq_rbfd);
err1:
	free(cxi_cq);
	return ret;
}

int cxi_cq_report_error(struct cxi_cq *cq, struct cxi_req *req,
			size_t olen, int err, int prov_errno, void *err_data,
			size_t err_data_size)
{
	int ret;
	struct fi_cq_err_entry err_entry;

	fastlock_acquire(&cq->lock);
	if (ofi_rbavail(&cq->cqerr_rb) < sizeof(err_entry)) {
		ret = -FI_ENOSPC;
		goto out;
	}

	err_entry.err = err;
	err_entry.olen = olen;
	err_entry.err_data = err_data;
	err_entry.err_data_size = err_data_size;
	err_entry.len = req->data_len;
	err_entry.prov_errno = prov_errno;
	err_entry.flags = req->flags;
	err_entry.data = req->data;
	err_entry.tag = req->tag;
	err_entry.op_context = (void *) (uintptr_t) req->context;

#if 0
	if (entry->type == CXI_PE_RX)
		err_entry.buf = (void *) (uintptr_t) entry->pe.rx.rx_iov[0].iov.addr;
	else
		err_entry.buf = (void *) (uintptr_t) entry->pe.tx.tx_iov[0].src.iov.addr;
#endif

	ofi_rbwrite(&cq->cqerr_rb, &err_entry, sizeof(err_entry));
	ofi_rbcommit(&cq->cqerr_rb);
	ret = 0;

	ofi_rbfdsignal(&cq->cq_rbfd);

out:
	fastlock_release(&cq->lock);
	return ret;
}

