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

#include <ofi_util.h>
#include "fi_verbs.h"

const struct fi_info *
fi_ibv_get_verbs_info(const struct fi_info *ilist, const char *domain_name)
{
	const struct fi_info *fi;

	for (fi = ilist; fi; fi = fi->next) {
		if (!strcmp(fi->domain_attr->name, domain_name))
			return fi;
	}

	return NULL;
}

static ssize_t
fi_ibv_eq_readerr(struct fid_eq *eq, struct fi_eq_err_entry *entry,
		  uint64_t flags)
{
	struct fi_ibv_eq *_eq;
	uint32_t api_version;
	void *err_data = NULL;
	size_t err_data_size = 0;

	_eq = container_of(eq, struct fi_ibv_eq, eq_fid.fid);
	if (!_eq->err.err)
		return 0;

	api_version = _eq->fab->util_fabric.fabric_fid.api_version;

	if ((FI_VERSION_GE(api_version, FI_VERSION(1, 5)))
		&& entry->err_data && entry->err_data_size) {
		err_data_size = MIN(entry->err_data_size, _eq->err.err_data_size);
		err_data = _eq->err.err_data;
	}

	*entry = _eq->err;
	if (err_data) {
		memcpy(entry->err_data, err_data, err_data_size);
		entry->err_data_size = err_data_size;
	}

	_eq->err.err = 0;
	_eq->err.prov_errno = 0;
	return sizeof(*entry);
}

static int
fi_ibv_eq_cm_getinfo(struct fi_ibv_fabric *fab, struct rdma_cm_event *event,
		     struct fi_info *pep_info, struct fi_info **info)
{
	struct fi_info *hints;
	struct fi_ibv_connreq *connreq;
	const char *devname = ibv_get_device_name(event->id->verbs->device);
	int ret = -FI_ENOMEM;

	if (!(hints = fi_dupinfo(pep_info)))
		return -FI_ENOMEM;

	/* Free src_addr info from pep to avoid addr reuse errors */
	free(hints->src_addr);
	hints->src_addr = NULL;
	hints->src_addrlen = 0;

	if (!strcmp(hints->domain_attr->name, VERBS_ANY_DOMAIN)) {
		free(hints->domain_attr->name);
		if (!(hints->domain_attr->name = strdup(devname)))
			goto err1;
	} else {
		if (strcmp(hints->domain_attr->name, devname)) {
			VERBS_WARN(FI_LOG_EQ, "Passive endpoint domain: %s does"
				   " not match device: %s where we got a "
				   "connection request\n",
				   hints->domain_attr->name, devname);
			ret = -FI_ENODATA;
			goto err1;
		}
	}

	if (!strcmp(hints->domain_attr->name, VERBS_ANY_FABRIC)) {
		free(hints->fabric_attr->name);
		hints->fabric_attr->name = NULL;
	}

	if (fi_ibv_getinfo(hints->fabric_attr->api_version, NULL, NULL, 0,
			   hints, info))
		goto err1;

	assert(!(*info)->dest_addr);

	free((*info)->src_addr);

	(*info)->src_addrlen = fi_ibv_sockaddr_len(rdma_get_local_addr(event->id));
	if (!((*info)->src_addr = malloc((*info)->src_addrlen)))
		goto err2;
	memcpy((*info)->src_addr, rdma_get_local_addr(event->id), (*info)->src_addrlen);

	(*info)->dest_addrlen = fi_ibv_sockaddr_len(rdma_get_peer_addr(event->id));
	if (!((*info)->dest_addr = malloc((*info)->dest_addrlen)))
		goto err2;
	memcpy((*info)->dest_addr, rdma_get_peer_addr(event->id), (*info)->dest_addrlen);

	ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_EQ, "src", (*info)->src_addr);
	ofi_straddr_dbg(&fi_ibv_prov, FI_LOG_EQ, "dst", (*info)->dest_addr);

	connreq = calloc(1, sizeof *connreq);
	if (!connreq)
		goto err2;

	connreq->handle.fclass = FI_CLASS_CONNREQ;
	connreq->id = event->id;
	(*info)->handle = &connreq->handle;
	fi_freeinfo(hints);
	return 0;
err2:
	fi_freeinfo(*info);
err1:
	fi_freeinfo(hints);
	return ret;
}

static ssize_t
fi_ibv_eq_cm_process_event(struct fi_ibv_eq *eq, struct rdma_cm_event *cma_event,
	uint32_t *event, struct fi_eq_cm_entry *entry, size_t len)
{
	struct fi_ibv_pep *pep;
	fid_t fid;
	size_t datalen;
	int ret;

	fid = cma_event->id->context;
	pep = container_of(fid, struct fi_ibv_pep, pep_fid);
	switch (cma_event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		*event = FI_CONNREQ;
		ret = fi_ibv_eq_cm_getinfo(eq->fab, cma_event, pep->info, &entry->info);
		if (ret) {
			rdma_destroy_id(cma_event->id);
			if (ret == -FI_ENODATA)
				return 0;
			eq->err.err = -ret;
			eq->err.prov_errno = ret;
			goto err;
		}
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		*event = FI_CONNECTED;
		entry->info = NULL;
		if (cma_event->id->qp->context->device->transport_type !=
		    IBV_TRANSPORT_IWARP) {
			ret = fi_ibv_set_rnr_timer(cma_event->id->qp);
			if (ret)
				return ret;
		}
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		*event = FI_SHUTDOWN;
		entry->info = NULL;
		break;
	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
		eq->err.err = -cma_event->status;
		goto err;
	case RDMA_CM_EVENT_REJECTED:
		eq->err.err = ECONNREFUSED;
		eq->err.prov_errno = -cma_event->status;
		goto err;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		eq->err.err = ENODEV;
		goto err;
	case RDMA_CM_EVENT_ADDR_CHANGE:
		eq->err.err = EADDRNOTAVAIL;
		goto err;
	default:
		return 0;
	}

	entry->fid = fid;
	datalen = MIN(len - sizeof(*entry), cma_event->param.conn.private_data_len);
	if (datalen)
		memcpy(entry->data, cma_event->param.conn.private_data, datalen);
	return sizeof(*entry) + datalen;
err:
	eq->err.fid = fid;
	return -FI_EAVAIL;
}

ssize_t fi_ibv_eq_write_event(struct fi_ibv_eq *eq, uint32_t event,
		const void *buf, size_t len)
{
	struct fi_ibv_eq_entry *entry;

	entry = calloc(1, sizeof(struct fi_ibv_eq_entry) + len);
	if (!entry)
		return -FI_ENOMEM;

	entry->event = event;
	entry->len = len;
	memcpy(entry->eq_entry, buf, len);

	fastlock_acquire(&eq->lock);
	dlistfd_insert_tail(&entry->item, &eq->list_head);
	fastlock_release(&eq->lock);

	return len;
}

static ssize_t fi_ibv_eq_write(struct fid_eq *eq_fid, uint32_t event,
			       const void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);
	if (!(eq->flags & FI_WRITE))
		return -FI_EINVAL;

	return fi_ibv_eq_write_event(eq, event, buf, len);
}

static size_t fi_ibv_eq_read_event(struct fi_ibv_eq *eq, uint32_t *event,
		void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq_entry *entry;
	ssize_t ret = 0;

	fastlock_acquire(&eq->lock);

	if (dlistfd_empty(&eq->list_head))
		goto out;

	entry = container_of(eq->list_head.list.next, struct fi_ibv_eq_entry, item);
	if (entry->len > len) {
		ret = -FI_ETOOSMALL;
		goto out;
	}

	ret = entry->len;
	*event = entry->event;
	memcpy(buf, entry->eq_entry, entry->len);

	if (!(flags & FI_PEEK)) {
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

out:
	fastlock_release(&eq->lock);
	return ret;
}

static ssize_t
fi_ibv_eq_read(struct fid_eq *eq_fid, uint32_t *event,
	       void *buf, size_t len, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct rdma_cm_event *cma_event;
	ssize_t ret = 0;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	if (eq->err.err)
		return -FI_EAVAIL;

	if ((ret = fi_ibv_eq_read_event(eq, event, buf, len, flags)))
		return ret;

	if (eq->channel) {
		ret = rdma_get_cm_event(eq->channel, &cma_event);
		if (ret)
			return -errno;

		if (len < sizeof(struct fi_eq_cm_entry)) {
			ret = -FI_ETOOSMALL;
			goto ack;
		}

		ret = fi_ibv_eq_cm_process_event(eq, cma_event, event,
				(struct fi_eq_cm_entry *)buf, len);
		if (ret < 0)
			goto ack;

		if (flags & FI_PEEK)
			ret = fi_ibv_eq_write_event(eq, *event, buf, len);
ack:
		rdma_ack_cm_event(cma_event);
		return ret;
	}

	return -FI_EAGAIN;
}

static ssize_t
fi_ibv_eq_sread(struct fid_eq *eq_fid, uint32_t *event,
		void *buf, size_t len, int timeout, uint64_t flags)
{
	struct fi_ibv_eq *eq;
	struct epoll_event events[2];
	ssize_t ret;

	eq = container_of(eq_fid, struct fi_ibv_eq, eq_fid.fid);

	while (1) {
		ret = fi_ibv_eq_read(eq_fid, event, buf, len, flags);
		if (ret && (ret != -FI_EAGAIN))
			return ret;

		ret = epoll_wait(eq->epfd, events, 2, timeout);
		if (ret == 0)
			return -FI_EAGAIN;
		else if (ret < 0)
			return -errno;
	};
}

static const char *
fi_ibv_eq_strerror(struct fid_eq *eq, int prov_errno, const void *err_data,
		   char *buf, size_t len)
{
	if (buf && len)
		strncpy(buf, strerror(prov_errno), len);
	return strerror(prov_errno);
}

static struct fi_ops_eq fi_ibv_eq_ops = {
	.size = sizeof(struct fi_ops_eq),
	.read = fi_ibv_eq_read,
	.readerr = fi_ibv_eq_readerr,
	.write = fi_ibv_eq_write,
	.sread = fi_ibv_eq_sread,
	.strerror = fi_ibv_eq_strerror
};

static int fi_ibv_eq_control(fid_t fid, int command, void *arg)
{
	struct fi_ibv_eq *eq;
	int ret = 0;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);
	switch (command) {
	case FI_GETWAIT:
		if (!eq->epfd) {
			ret = -FI_ENODATA;
			break;
		}
		*(int *) arg = eq->epfd;
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static int fi_ibv_eq_close(fid_t fid)
{
	struct fi_ibv_eq *eq;
	struct fi_ibv_eq_entry *entry;

	eq = container_of(fid, struct fi_ibv_eq, eq_fid.fid);
	/* TODO: use util code, if possible, and add ref counting */

	if (eq->channel)
		rdma_destroy_event_channel(eq->channel);

	close(eq->epfd);

	while (!dlistfd_empty(&eq->list_head)) {
		entry = container_of(eq->list_head.list.next,
				     struct fi_ibv_eq_entry, item);
		dlistfd_remove(eq->list_head.list.next, &eq->list_head);
		free(entry);
	}

	dlistfd_head_free(&eq->list_head);
	fastlock_destroy(&eq->lock);
	free(eq);

	return 0;
}

static struct fi_ops fi_ibv_eq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = fi_ibv_eq_close,
	.bind = fi_no_bind,
	.control = fi_ibv_eq_control,
	.ops_open = fi_no_ops_open,
};

int fi_ibv_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		   struct fid_eq **eq, void *context)
{
	struct fi_ibv_eq *_eq;
	struct epoll_event event;
	int ret;

	_eq = calloc(1, sizeof *_eq);
	if (!_eq)
		return -ENOMEM;

	_eq->fab = container_of(fabric, struct fi_ibv_fabric,
				util_fabric.fabric_fid);

	fastlock_init(&_eq->lock);
	ret = dlistfd_head_init(&_eq->list_head);
	if (ret) {
		VERBS_INFO(FI_LOG_EQ, "Unable to initialize dlistfd\n");
		goto err1;
	}

	_eq->epfd = epoll_create1(0);
	if (_eq->epfd < 0) {
		ret = -errno;
		goto err2;
	}

	memset(&event, 0, sizeof(event));
	event.events = EPOLLIN;

	if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD,
		      _eq->list_head.signal.fd[FI_READ_FD], &event)) {
		ret = -errno;
		goto err3;
	}

	switch (attr->wait_obj) {
	case FI_WAIT_NONE:
	case FI_WAIT_UNSPEC:
	case FI_WAIT_FD:
		_eq->channel = rdma_create_event_channel();
		if (!_eq->channel) {
			ret = -errno;
			goto err3;
		}

		ret = fi_fd_nonblock(_eq->channel->fd);
		if (ret)
			goto err4;

		if (epoll_ctl(_eq->epfd, EPOLL_CTL_ADD, _eq->channel->fd, &event)) {
			ret = -errno;
			goto err4;
		}

		break;
	default:
		ret = -FI_ENOSYS;
		goto err1;
	}

	_eq->flags = attr->flags;
	_eq->eq_fid.fid.fclass = FI_CLASS_EQ;
	_eq->eq_fid.fid.context = context;
	_eq->eq_fid.fid.ops = &fi_ibv_eq_fi_ops;
	_eq->eq_fid.ops = &fi_ibv_eq_ops;

	*eq = &_eq->eq_fid;
	return 0;
err4:
	if (_eq->channel)
		rdma_destroy_event_channel(_eq->channel);
err3:
	close(_eq->epfd);
err2:
	dlistfd_head_free(&_eq->list_head);
err1:
	fastlock_destroy(&_eq->lock);
	free(_eq);
	return ret;
}

