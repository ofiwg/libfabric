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


int udpx_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct udpx_ep *ep;
	int ret;

	ep = container_of(fid, struct udpx_ep, util_ep.ep_fid.fid);
	ret = bind(ep->sock, addr, addrlen);
	if (ret)
		return -errno;
	ep->is_bound = 1;
	return 0;
}

int udpx_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct udpx_ep *ep;
	socklen_t len;
	int ret;

	ep = container_of(fid, struct udpx_ep, util_ep.ep_fid.fid);
	len = *addrlen;
	ret = getsockname(ep->sock, addr, &len);
	*addrlen = len;
	return ret ? -errno : 0;
}

static struct fi_ops_cm udpx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = udpx_setname,
	.getname = udpx_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

int udpx_getopt(fid_t fid, int level, int optname,
		void *optval, size_t *optlen)
{
	return -FI_ENOPROTOOPT;
}

int udpx_setopt(fid_t fid, int level, int optname,
		const void *optval, size_t optlen)
{
	return -FI_ENOPROTOOPT;
}

static struct fi_ops_ep udpx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = udpx_getopt,
	.setopt = udpx_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void udpx_tx_comp(struct udpx_ep *ep, void *context)
{
	struct fi_cq_tagged_entry *comp;

	comp = cirque_tail(ep->util_ep.tx_cq->cirq);
	comp->op_context = context;
	comp->flags = FI_SEND;
	comp->len = 0;
	comp->buf = NULL;
	comp->data = 0;
	cirque_commit(ep->util_ep.tx_cq->cirq);
}

static void udpx_tx_comp_signal(struct udpx_ep *ep, void *context)
{
	udpx_tx_comp(ep, context);
	ep->util_ep.tx_cq->wait->signal(ep->util_ep.tx_cq->wait);
}

static void udpx_rx_comp(struct udpx_ep *ep, void *context, uint64_t flags,
			 size_t len, void *buf, void *addr)
{
	struct fi_cq_tagged_entry *comp;

	comp = cirque_tail(ep->util_ep.rx_cq->cirq);
	comp->op_context = context;
	comp->flags = FI_RECV | flags;
	comp->len = len;
	comp->buf = buf;
	comp->data = 0;
	cirque_commit(ep->util_ep.rx_cq->cirq);
}

static void udpx_rx_src_comp(struct udpx_ep *ep, void *context, uint64_t flags,
			     size_t len, void *buf, void *addr)
{
	ep->util_ep.rx_cq->src[cirque_windex(ep->util_ep.rx_cq->cirq)] =
			ip_av_get_index(ep->util_ep.av, addr);
	udpx_rx_comp(ep, context, flags, len, buf, addr);
}

static void udpx_rx_comp_signal(struct udpx_ep *ep, void *context,
			uint64_t flags, size_t len, void *buf, void *addr)
{
	udpx_rx_comp(ep, context, flags, len, buf, addr);
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
}

static void udpx_rx_src_comp_signal(struct udpx_ep *ep, void *context,
			uint64_t flags, size_t len, void *buf, void *addr)
{
	udpx_rx_src_comp(ep, context, flags, len, buf, addr);
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
}

void udpx_ep_progress(struct util_ep *util_ep)
{
	struct udpx_ep *ep;
	struct udpx_ep_entry *entry;
	struct msghdr hdr;
	struct sockaddr_in6 addr;
	int ret;

	ep = container_of(util_ep, struct udpx_ep, util_ep);
	hdr.msg_name = &addr;
	hdr.msg_namelen = sizeof(addr);
	hdr.msg_control = NULL;
	hdr.msg_controllen = 0;
	hdr.msg_flags = 0;

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (cirque_isempty(ep->rxq))
		goto out;

	entry = cirque_head(ep->rxq);
	hdr.msg_iov = entry->iov;
	hdr.msg_iovlen = entry->iov_count;

	ret = recvmsg(ep->sock, &hdr, 0);
	if (ret >= 0) {
		ep->rx_comp(ep, entry->context, 0, ret, NULL, &addr);
		cirque_discard(ep->rxq);
	}
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
}

ssize_t udpx_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct udpx_ep *ep;
	struct udpx_ep_entry *entry;
	ssize_t ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (cirque_isfull(ep->rxq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry = cirque_tail(ep->rxq);
	entry->context = msg->context;
	for (entry->iov_count = 0; entry->iov_count < msg->iov_count;
	     entry->iov_count++) {
		entry->iov[entry->iov_count] = msg->msg_iov[entry->iov_count];
	}
	entry->flags = 0;

	cirque_commit(ep->rxq);
	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t udpx_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.iov_count = count;
	msg.context = context;
	return udpx_recvmsg(ep_fid, &msg, 0);
}

ssize_t udpx_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct udpx_ep *ep;
	struct udpx_ep_entry *entry;
	ssize_t ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (cirque_isfull(ep->rxq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry = cirque_tail(ep->rxq);
	entry->context = context;
	entry->iov_count = 1;
	entry->iov[0].iov_base = buf;
	entry->iov[0].iov_len = len;
	entry->flags = 0;

	cirque_commit(ep->rxq);
	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t udpx_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct udpx_ep *ep;
	ssize_t ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	ret = sendto(ep->sock, buf, len, 0,
		     ip_av_get_addr(ep->util_ep.av, dest_addr),
		     ep->util_ep.av->addrlen);
	if (ret == len) {
		ep->tx_comp(ep, context);
		ret = 0;
	} else {
		ret = -errno;
	}
out:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
	return ret;
}

ssize_t udpx_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct udpx_ep *ep;
	struct msghdr hdr;
	ssize_t ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	hdr.msg_name = ip_av_get_addr(ep->util_ep.av, msg->addr);
	hdr.msg_namelen = ep->util_ep.av->addrlen;
	hdr.msg_iov = (struct iovec *) msg->msg_iov;
	hdr.msg_iovlen = msg->iov_count;
	hdr.msg_control = NULL;
	hdr.msg_controllen = 0;
	hdr.msg_flags = 0;

	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	ret = sendmsg(ep->sock, &hdr, 0);
	if (ret >= 0) {
		ep->tx_comp(ep, msg->context);
		ret = 0;
	} else {
		ret = -errno;
	}
out:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
	return ret;
}

ssize_t udpx_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;

	return udpx_sendmsg(ep_fid, &msg, 0);
}

ssize_t udpx_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	struct udpx_ep *ep;
	ssize_t ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	ret = sendto(ep->sock, buf, len, 0,
		     ip_av_get_addr(ep->util_ep.av, dest_addr),
		     ep->util_ep.av->addrlen);
	return ret == len ? 0 : -errno;
}

static struct fi_ops_msg udpx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = udpx_recv,
	.recvv = udpx_recvv,
	.recvmsg = udpx_recvmsg,
	.send = udpx_send,
	.sendv = udpx_sendv,
	.sendmsg = udpx_sendmsg,
	.inject = udpx_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

static int udpx_ep_close(struct fid *fid)
{
	struct udpx_ep *ep;
	struct util_wait_fd *wait;

	ep = container_of(fid, struct udpx_ep, util_ep.ep_fid.fid);

	if (ep->util_ep.av)
		atomic_dec(&ep->util_ep.av->ref);

	if (ep->util_ep.rx_cq) {
		if (ep->util_ep.rx_cq->wait) {
			wait = container_of(ep->util_ep.rx_cq->wait,
					    struct util_wait_fd, util_wait);
			fi_epoll_del(wait->epoll_fd, ep->sock);
		}
		fid_list_remove(&ep->util_ep.rx_cq->list,
				&ep->util_ep.rx_cq->list_lock,
				&ep->util_ep.ep_fid.fid);
		atomic_dec(&ep->util_ep.rx_cq->ref);
	}

	if (ep->util_ep.tx_cq)
		atomic_dec(&ep->util_ep.tx_cq->ref);

	udpx_rx_cirq_free(ep->rxq);
	close(ep->sock);
	atomic_dec(&ep->util_ep.domain->ref);
	free(ep);
	return 0;
}

static int udpx_ep_bind_cq(struct udpx_ep *ep, struct util_cq *cq, uint64_t flags)
{
	struct util_wait_fd *wait;
	int ret;

	if (flags & ~(FI_TRANSMIT | FI_RECV)) {
		FI_WARN(&udpx_prov, FI_LOG_EP_CTRL,
			"unsupported flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && ep->util_ep.tx_cq) ||
	    ((flags & FI_RECV) && ep->util_ep.rx_cq)) {
		FI_WARN(&udpx_prov, FI_LOG_EP_CTRL,
			"duplicate CQ binding\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		ep->util_ep.tx_cq = cq;
		atomic_inc(&cq->ref);
		ep->tx_comp = cq->wait ? udpx_tx_comp_signal : udpx_tx_comp;
	}

	if (flags & FI_RECV) {
		ep->util_ep.rx_cq = cq;
		atomic_inc(&cq->ref);

		if (cq->wait) {
			ep->rx_comp = (cq->domain->caps & FI_SOURCE) ?
				      udpx_rx_src_comp_signal :
				      udpx_rx_comp_signal;

			wait = container_of(cq->wait,
					    struct util_wait_fd, util_wait);
			ret = fi_epoll_add(wait->epoll_fd, ep->sock,
					   &ep->util_ep.ep_fid.fid);
			if (ret)
				return ret;
		} else {
			ep->rx_comp = (cq->domain->caps & FI_SOURCE) ?
				      udpx_rx_src_comp : udpx_rx_comp;
		}

		ret = fid_list_insert(&cq->list,
				      &cq->list_lock,
				      &ep->util_ep.ep_fid.fid);
		if (ret)
			return ret;
	}

	return 0;
}

static int udpx_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct udpx_ep *ep;
	struct util_av *av;
	int ret;

	ret = ofi_ep_bind_valid(&udpx_prov, bfid, flags);
	if (ret)
		return ret;

	ep = container_of(ep_fid, struct udpx_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		if (ep->util_ep.av) {
			FI_WARN(&udpx_prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
			return -FI_EINVAL;
		}
		av = container_of(bfid, struct util_av, av_fid.fid);
		atomic_inc(&av->ref);
		ep->util_ep.av = av;
		break;
	case FI_CLASS_CQ:
		ret = udpx_ep_bind_cq(ep, container_of(bfid, struct util_cq,
							cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&udpx_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static void udpx_bind_src_addr(struct udpx_ep *ep)
{
	int ret;
	struct addrinfo ai, *rai = NULL;
	char hostname[HOST_NAME_MAX];

	memset(&ai, 0, sizeof(ai));
	ai.ai_family = AF_INET;
	ai.ai_socktype = SOCK_DGRAM;

	ofi_getnodename(hostname, sizeof(hostname));
	ret = getaddrinfo(hostname, NULL, &ai, &rai);
	if (ret) {
		FI_WARN(&udpx_prov, FI_LOG_EP_CTRL,
			"getaddrinfo failed\n");
		return;
	}

	ret = udpx_setname(&ep->util_ep.ep_fid.fid, rai->ai_addr, rai->ai_addrlen);
	if (ret) {
		FI_WARN(&udpx_prov, FI_LOG_EP_CTRL, "failed to set addr\n");
	}
	freeaddrinfo(rai);
}

static int udpx_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct udpx_ep *ep;

	ep = container_of(fid, struct udpx_ep, util_ep.ep_fid.fid);
	switch (command) {
	case FI_ENABLE:
		if (!ep->util_ep.rx_cq || !ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!ep->util_ep.av)
			return -FI_EOPBADSTATE; /* TODO: Add FI_ENOAV */
		if (!ep->is_bound)
			udpx_bind_src_addr(ep);
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static struct fi_ops udpx_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = udpx_ep_close,
	.bind = udpx_ep_bind,
	.control = udpx_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int udpx_ep_init(struct udpx_ep *ep, struct fi_info *info)
{
	int family;
	int ret;

	ep->rxq = udpx_rx_cirq_create(info->rx_attr->size);
	if (!ep->rxq) {
		ret = -FI_ENOMEM;
		return ret;
	}

	family = info->src_addr ?
		 ((struct sockaddr *) info->src_addr)->sa_family : AF_INET;
	ep->sock = socket(family, SOCK_DGRAM, IPPROTO_UDP);
	if (ep->sock < 0) {
		ret = -errno;
		goto err1;
	}

	if (info->src_addr) {
		ret = bind(ep->sock, info->src_addr, info->src_addrlen);
		if (ret) {
			ret = -errno;
			goto err1;
		}
	}

	ret = fi_fd_nonblock(ep->sock);
	if (ret)
		goto err2;

	return 0;
err2:
	close(ep->sock);
err1:
	udpx_rx_cirq_free(ep->rxq);
	return ret;
}

int udpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct udpx_ep *ep;
	int ret;

	if (!info || !info->ep_attr || !info->rx_attr || !info->tx_attr)
		return -FI_EINVAL;

	ret = udpx_check_info(info);
	if (ret)
		return ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = udpx_ep_init(ep, info);
	if (ret) {
		free(ep);
		return ret;
	}

	ep->util_ep.ep_fid.fid.fclass = FI_CLASS_EP;
	ep->util_ep.ep_fid.fid.context = context;
	ep->util_ep.ep_fid.fid.ops = &udpx_ep_fi_ops;
	ep->util_ep.ep_fid.ops = &udpx_ep_ops;
	ep->util_ep.ep_fid.cm = &udpx_cm_ops;
	ep->util_ep.ep_fid.msg = &udpx_msg_ops;
	ep->util_ep.progress = udpx_ep_progress;

	ep->util_ep.domain = container_of(domain, struct util_domain, domain_fid);
	atomic_inc(&ep->util_ep.domain->ref);

	*ep_fid = &ep->util_ep.ep_fid;
	return 0;
}
