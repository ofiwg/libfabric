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

#include "config.h"

#include <fi_mem.h>

#include "fi_verbs.h"

static uint64_t fi_ibv_comp_flags(struct ibv_wc *wc)
{
	uint64_t flags = 0;

	if (wc->wc_flags & IBV_WC_WITH_IMM)
		flags |= FI_REMOTE_CQ_DATA;

	switch (wc->opcode) {
	case IBV_WC_SEND:
		flags |= FI_SEND | FI_MSG;
		break;
	case IBV_WC_RDMA_WRITE:
		flags |= FI_RMA | FI_WRITE;
		break;
	case IBV_WC_RDMA_READ:
		flags |= FI_RMA | FI_READ;
		break;
	case IBV_WC_COMP_SWAP:
		flags |= FI_ATOMIC;
		break;
	case IBV_WC_FETCH_ADD:
		flags |= FI_ATOMIC;
		break;
	case IBV_WC_RECV:
		flags |= FI_RECV | FI_MSG;
		break;
	case IBV_WC_RECV_RDMA_WITH_IMM:
		flags |= FI_RMA | FI_REMOTE_WRITE;
		break;
	default:
		break;
	}
	return flags;
}

static ssize_t
fi_ibv_cq_readerr(struct fid_cq *cq_fid, struct fi_cq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_cq *cq;
	struct fi_ibv_wce *wce;
	struct slist_entry *slist_entry;

	cq = container_of(cq_fid, struct fi_ibv_cq, cq_fid);

	fastlock_acquire(&cq->lock);
	if (slist_empty(&cq->wcq))
		goto err;

	wce = container_of(cq->wcq.head, struct fi_ibv_wce, entry);
	if (!wce->wc.status)
		goto err;

	slist_entry = slist_remove_head(&cq->wcq);
	fastlock_release(&cq->lock);

	wce = container_of(slist_entry, struct fi_ibv_wce, entry);

	entry->op_context = (void *) (uintptr_t) wce->wc.wr_id;
	entry->flags = fi_ibv_comp_flags(&wce->wc);
	entry->err = EIO;
	entry->prov_errno = wce->wc.status;
	memcpy(&entry->err_data, &wce->wc.vendor_err,
	       sizeof(wce->wc.vendor_err));

	util_buf_release(cq->domain->fab->wce_pool, wce);
	return sizeof(*entry);
err:
	fastlock_release(&cq->lock);
	return -FI_EAGAIN;
}

static inline int
fi_ibv_poll_events(struct fi_ibv_cq *_cq, int timeout)
{
	int ret, rc;
	void *context;
	struct pollfd fds[2];
	char data;

	fds[0].fd = _cq->channel->fd;
	fds[1].fd = _cq->signal_fd[0];

	fds[0].events = fds[1].events = POLLIN;

	rc = poll(fds, 2, timeout);
	if (rc == 0)
		return -FI_EAGAIN;
	else if (rc < 0)
		return -errno;

	if (fds[0].revents & POLLIN) {
		ret = ibv_get_cq_event(_cq->channel, &_cq->cq, &context);
		if (ret)
			return ret;

		atomic_inc(&_cq->nevents);
		rc--;
	}
	if (fds[1].revents & POLLIN) {
		do {
			ret = read(fds[1].fd, &data, 1);
		} while (ret > 0);
		ret = -FI_EAGAIN;
		rc--;
	}
	if (rc) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Unknown poll error: check revents\n");
		return -FI_EOTHER;
	}

	return ret;
}

static ssize_t
fi_ibv_cq_sread(struct fid_cq *cq, void *buf, size_t count, const void *cond,
		int timeout)
{
	ssize_t ret = 0, cur;
	ssize_t  threshold;
	struct fi_ibv_cq *_cq;
	uint8_t *p;

	p = buf;
	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);

	if (!_cq->channel)
		return -FI_ENOSYS;

	threshold = (_cq->wait_cond == FI_CQ_COND_THRESHOLD) ?
		MIN((ssize_t) cond, count) : 1;

	for (cur = 0; cur < threshold; ) {
		if (_cq->trywait(&cq->fid) == FI_SUCCESS) {
			ret = fi_ibv_poll_events(_cq, timeout);
			if (ret)
				break;
		}

		ret = _cq->cq_fid.ops->read(&_cq->cq_fid, p, count - cur);
		if (ret > 0) {
			p += ret * _cq->entry_size;
			cur += ret;
			if (cur >= threshold)
				break;
		} else if (ret != -FI_EAGAIN) {
			break;
		}
	}

	return cur ? cur : ret;
}

static void fi_ibv_cq_read_context_entry(struct ibv_wc *wc, int i, void *buf)
{
	struct fi_cq_entry *entry = buf;

	entry[i].op_context = (void *) (uintptr_t) wc->wr_id;
}

static void fi_ibv_cq_read_msg_entry(struct ibv_wc *wc, int i, void *buf)
{
	struct fi_cq_msg_entry *entry = buf;

	entry[i].op_context = (void *) (uintptr_t) wc->wr_id;
	entry[i].flags = fi_ibv_comp_flags(wc);
	entry[i].len = (uint64_t) wc->byte_len;
}

static void fi_ibv_cq_read_data_entry(struct ibv_wc *wc, int i, void *buf)
{
	struct fi_cq_data_entry *entry = buf;

	entry[i].op_context = (void *) (uintptr_t) wc->wr_id;
	entry[i].flags = fi_ibv_comp_flags(wc);

	entry[i].data = (wc->wc_flags & IBV_WC_WITH_IMM) ?
		ntohl(wc->imm_data) : 0;

	entry->len = (wc->opcode & (IBV_WC_RECV | IBV_WC_RECV_RDMA_WITH_IMM)) ?
		wc->byte_len : 0;
}

static int fi_ibv_match_ep_id(struct slist_entry *entry, const void *ep_id)
{
	struct fi_ibv_msg_epe *epe = container_of(entry, struct fi_ibv_msg_epe, entry);

	if (epe->ep->ep_id == (uint64_t) ep_id)
		return 1;

	return 0;
}

/* Must call with cq->lock held */
ssize_t fi_ibv_poll_cq(struct fi_ibv_cq *cq, struct ibv_wc *wc)
{
	struct fi_ibv_msg_epe *epe;
	struct slist_entry *entry;
	ssize_t ret;

	ret = ibv_poll_cq(cq->cq, 1, wc);
	if (ret <= 0)
		return ret;

	if (wc->opcode == IBV_WC_RECV || wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM)
		return ret;

	/* TODO Handle the case when app posts a send with same wr_id */
	if ((wc->wr_id & cq->wr_id_mask) != cq->send_signal_wr_id)
		return ret;

	entry = slist_remove_first_match(&cq->ep_list, fi_ibv_match_ep_id, (void *)wc->wr_id);
	if (!entry) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "No matching EP for :"
				"given signaled send completion\n");
		return -FI_EOTHER;
	}
	epe = container_of(entry, struct fi_ibv_msg_epe, entry);
	atomic_sub(&epe->ep->unsignaled_send_cnt,
			VERBS_SEND_SIGNAL_THRESH(epe->ep));
	atomic_dec(&epe->ep->comp_pending);
	util_buf_release(cq->domain->fab->epe_pool, epe);

	return 0;
}

static ssize_t fi_ibv_cq_read(struct fid_cq *cq_fid, void *buf, size_t count)
{
	struct fi_ibv_cq *cq;
	struct fi_ibv_wce *wce;
	struct slist_entry *entry;
	struct ibv_wc wc;
	ssize_t ret = 0, i;

	cq = container_of(cq_fid, struct fi_ibv_cq, cq_fid);

	fastlock_acquire(&cq->lock);

	for (i = 0; i < count; i++) {
		if (!slist_empty(&cq->wcq)) {
			wce = container_of(cq->wcq.head, struct fi_ibv_wce, entry);
			if (wce->wc.status) {
				ret = -FI_EAVAIL;
				break;
			}
			entry = slist_remove_head(&cq->wcq);
			wce = container_of(entry, struct fi_ibv_wce, entry);
			cq->read_entry(&wce->wc, i, buf);
			util_buf_release(cq->domain->fab->wce_pool, wce);
			continue;
		}

		ret = fi_ibv_poll_cq(cq, &wc);
		if (ret <= 0)
			break;

		/* Insert error entry into wcq */
		if (wc.status) {
			wce = util_buf_alloc(cq->domain->fab->wce_pool);
			if (!wce) {
				fastlock_release(&cq->lock);
				return -FI_ENOMEM;
			}
			memset(wce, 0, sizeof(*wce));
			memcpy(&wce->wc, &wc, sizeof wc);
			slist_insert_tail(&wce->entry, &cq->wcq);
			ret = -FI_EAVAIL;
			break;
		}

		cq->read_entry(&wc, i, buf);
	}

	fastlock_release(&cq->lock);
	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static const char *
fi_ibv_cq_strerror(struct fid_cq *eq, int prov_errno, const void *err_data,
		   char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, ibv_wc_status_str(prov_errno), len);
	return ibv_wc_status_str(prov_errno);
}

int fi_ibv_cq_signal(struct fid_cq *cq)
{
	struct fi_ibv_cq *_cq;
	char data = '0';

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);

	if (write(_cq->signal_fd[1], &data, 1) != 1) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Error signalling CQ\n");
		return -errno;
	}

	return 0;
}

static int fi_ibv_cq_trywait(struct fid *fid)
{
	struct fi_ibv_cq *cq;
	struct fi_ibv_wce *wce;
	void *context;
	int ret = -FI_EAGAIN, rc;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);

	if (!cq->channel) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "No wait object object associated with CQ\n");
		return -FI_EINVAL;
	}

	fastlock_acquire(&cq->lock);
	if (!slist_empty(&cq->wcq))
		goto out;

	wce = util_buf_alloc(cq->domain->fab->wce_pool);
	if (!wce) {
		ret = -FI_ENOMEM;
		goto out;
	}
	memset(wce, 0, sizeof(*wce));

	rc = fi_ibv_poll_cq(cq, &wce->wc);
	if (rc > 0) {
		slist_insert_tail(&wce->entry, &cq->wcq);
		goto out;
	} else if (rc < 0) {
		goto err;
	}

	while (!ibv_get_cq_event(cq->channel, &cq->cq, &context))
		atomic_inc(&cq->nevents);

	rc = ibv_req_notify_cq(cq->cq, 0);
	if (rc) {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "ibv_req_notify_cq error: %d\n", ret);
		ret = -errno;
		goto err;
	}

	/* Read again to fetch any completions that we might have missed
	 * while rearming */
	rc = fi_ibv_poll_cq(cq, &wce->wc);
	if (rc > 0) {
		slist_insert_tail(&wce->entry, &cq->wcq);
		goto out;
	} else if (rc < 0) {
		goto err;
	}

	ret = FI_SUCCESS;
err:
	util_buf_release(cq->domain->fab->wce_pool, wce);
out:
	fastlock_release(&cq->lock);
	return ret;
}

static struct fi_ops_cq fi_ibv_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static int fi_ibv_cq_control(fid_t fid, int command, void *arg)
{
	struct fi_ibv_cq *cq;
	int ret = 0;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);
	switch(command) {
	case FI_GETWAIT:
		if (!cq->channel) {
			ret = -FI_ENODATA;
			break;
		}
		*(int *) arg = cq->channel->fd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_cq_close(fid_t fid)
{
	struct fi_ibv_cq *cq;
	struct fi_ibv_msg_epe *epe;
	struct fi_ibv_wce *wce;
	struct slist_entry *entry;
	int ret;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);

	if (atomic_get(&cq->nevents))
		ibv_ack_cq_events(cq->cq, atomic_get(&cq->nevents));

	fastlock_acquire(&cq->lock);
	while (!slist_empty(&cq->wcq)) {
		entry = slist_remove_head(&cq->wcq);
		wce = container_of(entry, struct fi_ibv_wce, entry);
		util_buf_release(cq->domain->fab->wce_pool, wce);
	}

	while (!slist_empty(&cq->ep_list)) {
		entry = slist_remove_head(&cq->ep_list);
		epe = container_of(entry, struct fi_ibv_msg_epe, entry);
		util_buf_release(cq->domain->fab->epe_pool, epe);
	}
	fastlock_release(&cq->lock);

	fastlock_destroy(&cq->lock);

	if (cq->cq) {
		ret = ibv_destroy_cq(cq->cq);
		if (ret)
			return -ret;
	}

	if (cq->signal_fd[0]) {
		ofi_close_socket(cq->signal_fd[0]);
	}
	if (cq->signal_fd[1]) {
		ofi_close_socket(cq->signal_fd[1]);
	}

	if (cq->channel)
		ibv_destroy_comp_channel(cq->channel);

	free(cq);
	return 0;
}

static struct fi_ops fi_ibv_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_cq_close,
	.bind = fi_no_bind,
	.control = fi_ibv_cq_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq, void *context)
{
	struct fi_ibv_cq *_cq;
	int ep_cnt_bits = 0;
	size_t size;
	int ret;

	_cq = calloc(1, sizeof *_cq);
	if (!_cq)
		return -FI_ENOMEM;

	_cq->domain = container_of(domain, struct fi_ibv_domain, domain_fid);
	/*
	 * RDM functionality is moved to correspond separated functions
	 */
	assert(!_cq->domain->rdm);

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_cq->channel = ibv_create_comp_channel(_cq->domain->verbs);
		if (!_cq->channel) {
			ret = -errno;
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ,
					"Unable to create completion channel\n");
			goto err1;
		}

		ret = fi_fd_nonblock(_cq->channel->fd);
		if (ret)
			goto err2;

		if (socketpair(AF_UNIX, SOCK_STREAM, 0, _cq->signal_fd)) {
			ret = -errno;
			goto err2;
		}

		ret = fi_fd_nonblock(_cq->signal_fd[0]);
		if (ret)
			goto err3;

		break;
	case FI_WAIT_NONE:
		break;
	default:
		ret = -FI_ENOSYS;
		goto err3;
	}

	size = attr->size ? attr->size : VERBS_DEF_CQ_SIZE;

	_cq->cq = ibv_create_cq(_cq->domain->verbs, size, _cq, _cq->channel,
			attr->signaling_vector);

	if (!_cq->cq) {
		ret = -errno;
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Unable to create verbs CQ\n");
		goto err3;
	}

	if (_cq->channel) {
		ret = ibv_req_notify_cq(_cq->cq, 0);
		if (ret) {
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ,
				"ibv_req_notify_cq failed\n");
			goto err4;
		}
	}

	_cq->flags |= attr->flags;
	_cq->wait_cond = attr->wait_cond;
	_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	_cq->cq_fid.fid.context = context;
	_cq->cq_fid.fid.ops = &fi_ibv_cq_fi_ops;
	_cq->cq_fid.ops = &fi_ibv_cq_ops;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
	case FI_CQ_FORMAT_CONTEXT:
		_cq->read_entry = fi_ibv_cq_read_context_entry;
		_cq->entry_size = sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		_cq->read_entry = fi_ibv_cq_read_msg_entry;
		_cq->entry_size = sizeof(struct fi_cq_msg_entry);
		break;
	case FI_CQ_FORMAT_DATA:
		_cq->read_entry = fi_ibv_cq_read_data_entry;
		_cq->entry_size = sizeof(struct fi_cq_data_entry);
		break;
	case FI_CQ_FORMAT_TAGGED:
	default:
		ret = -FI_ENOSYS;
		goto err4;
	}

	fastlock_init(&_cq->lock);

	slist_init(&_cq->wcq);
	slist_init(&_cq->ep_list);

	while (_cq->domain->info->domain_attr->ep_cnt >> ++ep_cnt_bits);

	_cq->send_signal_wr_id = 0xffffffffffffc0de << ep_cnt_bits;
	_cq->wr_id_mask = (~_cq->wr_id_mask) << ep_cnt_bits;

	_cq->trywait = fi_ibv_cq_trywait;
	atomic_initialize(&_cq->nevents, 0);

	*cq = &_cq->cq_fid;
	return 0;

err4:
	ibv_destroy_cq(_cq->cq);
err3:
	ofi_close_socket(_cq->signal_fd[0]);
	ofi_close_socket(_cq->signal_fd[1]);
err2:
	if (_cq->channel)
		ibv_destroy_comp_channel(_cq->channel);
err1:
	free(_cq);
	return ret;
}
