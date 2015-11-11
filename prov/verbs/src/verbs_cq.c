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

#include "fi_verbs.h"


static ssize_t
fi_ibv_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_cq *_cq;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (!_cq->wc.status)
		return 0;

	entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
	entry->flags = 0;
	entry->err = EIO;
	entry->prov_errno = _cq->wc.status;
	memcpy(&entry->err_data, &_cq->wc.vendor_err,
	       sizeof(_cq->wc.vendor_err));

	_cq->wc.status = 0;
	return sizeof(*entry);
}

static inline int
fi_ibv_poll_events(struct fi_ibv_cq *_cq, int timeout)
{
	int ret;
	void *context;
	struct pollfd fds[2];
	char data;

	fds[0].fd = _cq->channel->fd;
	fds[1].fd = _cq->signal_fd[0];

	fds[0].events = fds[1].events = POLLIN;

	ret = poll(fds, 2, timeout);
	if (ret == 0)
		return -FI_EAGAIN;
	else if (ret < 0)
		return -errno;

	if (fds[1].revents & POLLIN) {
		do {
			ret = read(fds[1].fd, &data, 1);
		} while (ret > 0);
		return -FI_EAGAIN;
	} else if (fds[0].revents & POLLIN) {
		ret = ibv_get_cq_event(_cq->channel, &_cq->cq, &context);
		if (ret)
			return ret;

		ibv_ack_cq_events(_cq->cq, 1);
	} else {
		FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "Unknown poll error: check revents\n");
		return -FI_EOTHER;
	}

	return 0;
}

static ssize_t
fi_ibv_cq_sread(struct fid_cq *cq, void *buf, size_t count, const void *cond,
		int timeout)
{
	ssize_t ret = 0, cur;
	ssize_t  threshold;
	struct fi_ibv_cq *_cq;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);

	if (!_cq->channel)
		return -FI_ENOSYS;

	threshold = (_cq->wait_cond == FI_CQ_COND_THRESHOLD) ?
		MIN((ssize_t) cond, count) : 1;

	for (cur = 0; cur < threshold; ) {
		ret = _cq->cq_fid.ops->read(&_cq->cq_fid, buf, count - cur);
		if (ret > 0) {
			buf += ret * _cq->entry_size;
			cur += ret;
			if (cur >= threshold)
				break;
		} else if (ret != -FI_EAGAIN) {
			break;
		}

		ret = ibv_req_notify_cq(_cq->cq, 0);
		if (ret) {
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "ibv_req_notify_cq error: %d\n", ret);
			break;
		}

		/* Read again to fetch any completions that we might have missed
		 * while rearming */
		ret = _cq->cq_fid.ops->read(&_cq->cq_fid, buf, count - cur);
		if (ret > 0) {
			buf += ret * _cq->entry_size;
			cur += ret;
			if (cur >= threshold)
				break;
		} else if (ret != -FI_EAGAIN) {
			break;
		}

		ret = fi_ibv_poll_events(_cq, timeout);
		if (ret)
			break;
	}

	return cur ? cur : ret;
}

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

static ssize_t fi_ibv_cq_read_context(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry += 1;
	}

	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static ssize_t fi_ibv_cq_read_msg(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_msg_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry->flags = fi_ibv_comp_flags(&_cq->wc);
		entry->len = (uint64_t) _cq->wc.byte_len;
		entry += 1;
	}

	return i ? i : (ret ? ret : -FI_EAGAIN);
}

static ssize_t fi_ibv_cq_read_data(struct fid_cq *cq, void *buf, size_t count)
{
	struct fi_ibv_cq *_cq;
	struct fi_cq_data_entry *entry = buf;
	ssize_t ret = 0, i;

	_cq = container_of(cq, struct fi_ibv_cq, cq_fid);
	if (_cq->wc.status)
		return -FI_EAVAIL;

	for (i = 0; i < count; i++) {
		ret = ibv_poll_cq(_cq->cq, 1, &_cq->wc);
		if (ret <= 0)
			break;

		if (_cq->wc.status) {
			ret = -FI_EAVAIL;
			break;
		}

		entry->op_context = (void *) (uintptr_t) _cq->wc.wr_id;
		entry->flags = fi_ibv_comp_flags(&_cq->wc);
		if (_cq->wc.wc_flags & IBV_WC_WITH_IMM) {
			entry->data = ntohl(_cq->wc.imm_data);
		} else {
			entry->data = 0;
		}
		if (_cq->wc.opcode & (IBV_WC_RECV | IBV_WC_RECV_RDMA_WITH_IMM))
			entry->len = _cq->wc.byte_len;
		else
			entry->len = 0;

		entry += 1;
	}

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

static int
fi_ibv_cq_signal(struct fid_cq *cq)
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

static struct fi_ops_cq fi_ibv_cq_context_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_context,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static struct fi_ops_cq fi_ibv_cq_msg_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_msg,
	.readfrom = fi_no_cq_readfrom,
	.readerr = fi_ibv_cq_readerr,
	.sread = fi_ibv_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_ibv_cq_signal,
	.strerror = fi_ibv_cq_strerror
};

static struct fi_ops_cq fi_ibv_cq_data_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = fi_ibv_cq_read_data,
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
	int ret;

	cq = container_of(fid, struct fi_ibv_cq, cq_fid.fid);
	if (cq->cq) {
		ret = ibv_destroy_cq(cq->cq);
		if (ret)
			return -ret;
	}

	if (cq->signal_fd[0]) {
		close(cq->signal_fd[0]);
	}
	if (cq->signal_fd[1]) {
		close(cq->signal_fd[1]);
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
	int ret;

	_cq = calloc(1, sizeof *_cq);
	if (!_cq)
		return -FI_ENOMEM;

	_cq->domain = container_of(domain, struct fi_ibv_domain, domain_fid);

	switch (attr->wait_obj) {
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_cq->channel = ibv_create_comp_channel(_cq->domain->verbs);
		if (!_cq->channel) {
			ret = -errno;
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

	_cq->cq = ibv_create_cq(_cq->domain->verbs, attr->size, _cq,
				_cq->channel, attr->signaling_vector);
	if (!_cq->cq) {
		ret = -errno;
		goto err3;
	}

	if (_cq->channel) {
		ret = ibv_req_notify_cq(_cq->cq, 0);
		if (ret) {
			FI_WARN(&fi_ibv_prov, FI_LOG_CQ, "ibv_req_notify_cq failed\n");
			goto err4;
		}
	}

	_cq->flags |= attr->flags;
	_cq->wait_cond = attr->wait_cond;
	_cq->cq_fid.fid.fclass = FI_CLASS_CQ;
	_cq->cq_fid.fid.context = context;
	_cq->cq_fid.fid.ops = &fi_ibv_cq_fi_ops;

	switch (attr->format) {
	case FI_CQ_FORMAT_CONTEXT:
		assert(!_cq->domain->rdm);
		_cq->cq_fid.ops = &fi_ibv_cq_context_ops;
		_cq->entry_size = sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		assert(!_cq->domain->rdm);
		_cq->cq_fid.ops = &fi_ibv_cq_msg_ops;
		_cq->entry_size = sizeof(struct fi_cq_msg_entry);
		break;
	case FI_CQ_FORMAT_DATA:
		assert(!_cq->domain->rdm);
		_cq->cq_fid.ops = &fi_ibv_cq_data_ops;
		_cq->entry_size = sizeof(struct fi_cq_data_entry);
		break;
	case FI_CQ_FORMAT_TAGGED:
		assert(_cq->domain->rdm);
		_cq->cq_fid.ops = fi_ibv_cq_ops_tagged(_cq);
		_cq->entry_size = sizeof(struct fi_cq_tagged_entry);
		break;
	default:
		ret = -FI_ENOSYS;
		goto err4;
	}

	*cq = &_cq->cq_fid;
	return 0;

err4:
	ibv_destroy_cq(_cq->cq);
err3:
	close(_cq->signal_fd[0]);
	close(_cq->signal_fd[1]);
err2:
	if (_cq->channel)
		ibv_destroy_comp_channel(_cq->channel);
err1:
	free(_cq);
	return ret;
}
