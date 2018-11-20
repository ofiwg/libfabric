/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *	   Redistribution and use in source and binary forms, with or
 *	   without modification, are permitted provided that the following
 *	   conditions are met:
 *
 *		- Redistributions of source code must retain the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer.
 *
 *		- Redistributions in binary form must reproduce the above
 *		  copyright notice, this list of conditions and the following
 *		  disclaimer in the documentation and/or other materials
 *		  provided with the distribution.
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

#include <rdma/fi_errno.h>
#include "rdma/fi_eq.h"
#include "ofi_iov.h"
#include <ofi_prov.h>
#include "tcpx.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <ofi_util.h>
#include <unistd.h>
#include <string.h>
#include <poll.h>
#include <arpa/inet.h>
#include <netdb.h>

extern struct fi_ops_rma tcpx_rma_ops;

static inline struct tcpx_xfer_entry *
tcpx_alloc_recv_entry(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_cq *tcpx_cq;

	tcpx_cq = container_of(tcpx_ep->util_ep.rx_cq, struct tcpx_cq,
			       util_cq);

	recv_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_MSG_RECV);
	if (recv_entry) {
		recv_entry->ep = tcpx_ep;
		recv_entry->done_len = 0;
	}
	return recv_entry;
}

static inline struct tcpx_xfer_entry *
tcpx_alloc_send_entry(struct tcpx_ep *tcpx_ep)
{
	struct tcpx_xfer_entry *send_entry;
	struct tcpx_cq *tcpx_cq;

	tcpx_cq = container_of(tcpx_ep->util_ep.tx_cq, struct tcpx_cq,
			       util_cq);

	send_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_MSG_SEND);
	if (send_entry) {
		send_entry->ep = tcpx_ep;
		send_entry->done_len = 0;
	}
	return send_entry;
}

static inline void tcpx_queue_recv(struct tcpx_ep *tcpx_ep,
				   struct tcpx_xfer_entry *recv_entry)
{
	fastlock_acquire(&tcpx_ep->lock);
	slist_insert_tail(&recv_entry->entry, &tcpx_ep->rx_queue);
	fastlock_release(&tcpx_ep->lock);
}

static ssize_t tcpx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_ep *tcpx_ep;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	recv_entry = tcpx_alloc_recv_entry(tcpx_ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = msg->iov_count;
	memcpy(&recv_entry->msg_data.iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof(struct iovec));

	recv_entry->flags = ((tcpx_ep->util_ep.rx_op_flags & FI_COMPLETION) |
			     flags | FI_MSG | FI_RECV);
	recv_entry->context = msg->context;

	tcpx_queue_recv(tcpx_ep, recv_entry);
	return FI_SUCCESS;
}

static ssize_t tcpx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_ep *tcpx_ep;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	recv_entry = tcpx_alloc_recv_entry(tcpx_ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = 1;
	recv_entry->msg_data.iov[0].iov_base = buf;
	recv_entry->msg_data.iov[0].iov_len = len;

	recv_entry->flags = ((tcpx_ep->util_ep.rx_op_flags & FI_COMPLETION) |
			     FI_MSG | FI_RECV);
	recv_entry->context = context;

	tcpx_queue_recv(tcpx_ep, recv_entry);
	return FI_SUCCESS;
}

static ssize_t tcpx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t src_addr, void *context)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_ep *tcpx_ep;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	assert(count <= TCPX_IOV_LIMIT);

	recv_entry = tcpx_alloc_recv_entry(tcpx_ep);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = count;
	memcpy(recv_entry->msg_data.iov, iov, count * sizeof(*iov));

	recv_entry->flags = ((tcpx_ep->util_ep.rx_op_flags & FI_COMPLETION) |
			     FI_MSG | FI_RECV);
	recv_entry->context = context;

	tcpx_queue_recv(tcpx_ep, recv_entry);
	return FI_SUCCESS;
}

static ssize_t tcpx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_cq *tcpx_cq;
	struct tcpx_xfer_entry *tx_entry;
	uint64_t data_len;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	tcpx_cq = container_of(tcpx_ep->util_ep.tx_cq, struct tcpx_cq,
			       util_cq);

	tx_entry = tcpx_xfer_entry_alloc(tcpx_cq, TCPX_OP_MSG_SEND);
	if (!tx_entry)
		return -FI_EAGAIN;

	assert(msg->iov_count <= TCPX_IOV_LIMIT);
	data_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	assert(!(flags & FI_INJECT) || (data_len <= TCPX_MAX_INJECT_SZ));
	tx_entry->msg_hdr.hdr.size = htonll(data_len + sizeof(tx_entry->msg_hdr));
	tx_entry->msg_hdr.hdr.flags = 0;

	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	tx_entry->msg_data.iov_cnt = msg->iov_count + 1;

	if (flags & FI_INJECT) {
		ofi_copy_iov_buf(msg->msg_iov, msg->iov_count, 0,
				 tx_entry->msg_data.inject,
				 data_len,
				 OFI_COPY_IOV_TO_BUF);

		tx_entry->msg_data.iov[1].iov_base = (void *)tx_entry->msg_data.inject;
		tx_entry->msg_data.iov[1].iov_len = data_len;
		tx_entry->msg_data.iov_cnt = 2;
	} else {
		memcpy(&tx_entry->msg_data.iov[1], &msg->msg_iov[0],
		       msg->iov_count * sizeof(struct iovec));

	}

	tx_entry->flags = ((tcpx_ep->util_ep.tx_op_flags & FI_COMPLETION) |
			    flags | FI_MSG | FI_SEND);

	if (flags & FI_REMOTE_CQ_DATA) {
		tx_entry->msg_hdr.hdr.flags |= OFI_REMOTE_CQ_DATA;
		tx_entry->msg_hdr.hdr.data = htonll(msg->data);
	}

	if (flags & (FI_TRANSMIT_COMPLETE | FI_DELIVERY_COMPLETE)) {
		tx_entry->msg_hdr.hdr.flags |= OFI_DELIVERY_COMPLETE;
		tx_entry->flags &= ~FI_COMPLETION;
	}

	tx_entry->msg_hdr.hdr.flags = htonl(tx_entry->msg_hdr.hdr.flags);
	tx_entry->ep = tcpx_ep;
	tx_entry->context = msg->context;
	tx_entry->done_len = 0;

	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			 fi_addr_t dest_addr, void *context)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_xfer_entry *tx_entry;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	tx_entry = tcpx_alloc_send_entry(tcpx_ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->msg_hdr.hdr.size = htonll(len + sizeof(tx_entry->msg_hdr));
	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	tx_entry->msg_data.iov[1].iov_base = (void *) buf;
	tx_entry->msg_data.iov[1].iov_len = len;
	tx_entry->msg_data.iov_cnt = 2;
	tx_entry->context = context;
	tx_entry->flags = ((tcpx_ep->util_ep.tx_op_flags & FI_COMPLETION) |
			   FI_MSG | FI_SEND);

	tx_entry->msg_hdr.hdr.flags = 0;
	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t dest_addr, void *context)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_xfer_entry *tx_entry;
	uint64_t data_len;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	tx_entry = tcpx_alloc_send_entry(tcpx_ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	assert(count <= TCPX_IOV_LIMIT);
	data_len = ofi_total_iov_len(iov, count);
	tx_entry->msg_hdr.hdr.size = htonll(data_len + sizeof(tx_entry->msg_hdr));
	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	tx_entry->msg_data.iov_cnt = count + 1;
	memcpy(&tx_entry->msg_data.iov[1], &iov[0],
	       count * sizeof(struct iovec));

	tx_entry->msg_hdr.hdr.flags = 0;
	tx_entry->context = context;
	tx_entry->flags = ((tcpx_ep->util_ep.tx_op_flags & FI_COMPLETION) |
			   FI_MSG | FI_SEND);

	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}


static ssize_t tcpx_inject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_xfer_entry *tx_entry;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	tx_entry = tcpx_alloc_send_entry(tcpx_ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	assert(len <= TCPX_MAX_INJECT_SZ);
	tx_entry->msg_hdr.hdr.size = htonll(len + sizeof(tx_entry->msg_hdr));
	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	memcpy(tx_entry->msg_data.inject, (char *) buf, len);
	tx_entry->msg_data.iov[1].iov_base = (void *)tx_entry->msg_data.inject;
	tx_entry->msg_data.iov[1].iov_len = len;
	tx_entry->msg_data.iov_cnt = 2;

	tx_entry->msg_hdr.hdr.flags = 0;
	tx_entry->flags = FI_MSG | FI_SEND;

	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			     uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_xfer_entry *tx_entry;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	tx_entry = tcpx_alloc_send_entry(tcpx_ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	tx_entry->msg_hdr.hdr.size = htonll(len + sizeof(tx_entry->msg_hdr));
	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	tx_entry->msg_data.iov[1].iov_base = (void *) buf;
	tx_entry->msg_data.iov[1].iov_len = len;
	tx_entry->msg_data.iov_cnt = 2;

	tx_entry->msg_hdr.hdr.flags = htonl(OFI_REMOTE_CQ_DATA);
	tx_entry->msg_hdr.hdr.data = htonll(data);

	tx_entry->context = context;
	tx_entry->flags = ((tcpx_ep->util_ep.tx_op_flags & FI_COMPLETION) |
			   FI_MSG | FI_SEND);

	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_xfer_entry *tx_entry;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	tx_entry = tcpx_alloc_send_entry(tcpx_ep);
	if (!tx_entry)
		return -FI_EAGAIN;

	assert(len <= TCPX_MAX_INJECT_SZ);
	tx_entry->msg_hdr.hdr.size = htonll(len + sizeof(tx_entry->msg_hdr));

	tx_entry->msg_data.iov[0].iov_base = (void *) &tx_entry->msg_hdr;
	tx_entry->msg_data.iov[0].iov_len = sizeof(tx_entry->msg_hdr);
	memcpy(tx_entry->msg_data.inject, (char *) buf, len);
	tx_entry->msg_data.iov[1].iov_base = (void *)tx_entry->msg_data.inject;
	tx_entry->msg_data.iov[1].iov_len = len;
	tx_entry->msg_data.iov_cnt = 2;

	tx_entry->msg_hdr.hdr.flags = htonl(OFI_REMOTE_CQ_DATA);
	tx_entry->msg_hdr.hdr.data = htonll(data);
	tx_entry->flags = FI_MSG | FI_SEND ;

	fastlock_acquire(&tcpx_ep->lock);
	tcpx_tx_queue_insert(tcpx_ep, tx_entry);
	fastlock_release(&tcpx_ep->lock);
	return FI_SUCCESS;
}

static struct fi_ops_msg tcpx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = tcpx_recv,
	.recvv = tcpx_recvv,
	.recvmsg = tcpx_recvmsg,
	.send = tcpx_send,
	.sendv = tcpx_sendv,
	.sendmsg = tcpx_sendmsg,
	.inject = tcpx_inject,
	.senddata = tcpx_senddata,
	.injectdata = tcpx_injectdata,
};

static int tcpx_setup_socket(SOCKET sock)
{
	int ret, optval = 1;

	ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &optval,
			 sizeof(optval));
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"setsockopt reuseaddr failed\n");
		return ret;
	}

	ret = setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char *) &optval,
			 sizeof(optval));
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"setsockopt nodelay failed\n");
		return ret;
	}

	return ret;
}

static int tcpx_ep_connect(struct fid_ep *ep, const void *addr,
			   const void *param, size_t paramlen)
{
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	struct tcpx_cm_context *cm_ctx;
	int ret;

	if (!addr || !tcpx_ep->conn_fd || paramlen > TCPX_MAX_CM_DATA_SIZE)
		return -FI_EINVAL;

	cm_ctx = calloc(1, sizeof(*cm_ctx));
	if (!cm_ctx) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		return -FI_ENOMEM;
	}

	ret = connect(tcpx_ep->conn_fd, (struct sockaddr *) addr,
		      (socklen_t) ofi_sizeofaddr(addr));
	if (ret && ofi_sockerr() != FI_EINPROGRESS) {
		ret =  -ofi_sockerr();
		goto err;
	}

	cm_ctx->fid = &tcpx_ep->util_ep.ep_fid.fid;
	cm_ctx->type = CLIENT_SEND_CONNREQ;

	if (paramlen) {
		cm_ctx->cm_data_sz = paramlen;
		memcpy(cm_ctx->cm_data, param, paramlen);
	}

	ret = ofi_wait_fd_add(tcpx_ep->util_ep.eq->wait, tcpx_ep->conn_fd,
			      FI_EPOLL_OUT, tcpx_eq_wait_try_func, NULL,cm_ctx);
	if (ret)
		goto err;

	tcpx_ep->util_ep.eq->wait->signal(tcpx_ep->util_ep.eq->wait);
	return 0;
err:
	free(cm_ctx);
	return ret;
}

static int tcpx_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	struct tcpx_cm_context *cm_ctx;
	int ret;

	if (tcpx_ep->conn_fd == INVALID_SOCKET)
		return -FI_EINVAL;

	cm_ctx = calloc(1, sizeof(*cm_ctx));
	if (!cm_ctx) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		return -FI_ENOMEM;
	}

	cm_ctx->fid = &tcpx_ep->util_ep.ep_fid.fid;
	cm_ctx->type = SERVER_SEND_CM_ACCEPT;
	if (paramlen) {
		cm_ctx->cm_data_sz = paramlen;
		memcpy(cm_ctx->cm_data, param, paramlen);
	}

	ret = ofi_wait_fd_add(tcpx_ep->util_ep.eq->wait, tcpx_ep->conn_fd,
			      FI_EPOLL_OUT, tcpx_eq_wait_try_func, NULL, cm_ctx);
	if (ret) {
		free(cm_ctx);
		return ret;
	}
	tcpx_ep->util_ep.eq->wait->signal(tcpx_ep->util_ep.eq->wait);
	return 0;
}

static int tcpx_ep_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct tcpx_ep *tcpx_ep;
	int ret;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	ret = ofi_shutdown(tcpx_ep->conn_fd, SHUT_RDWR);
	if (ret && ofi_sockerr() != ENOTCONN) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA, "ep shutdown unsuccessful\n");
	}

	fastlock_acquire(&tcpx_ep->lock);
	ret = tcpx_ep_shutdown_report(tcpx_ep, &ep->fid);
	fastlock_release(&tcpx_ep->lock);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA, "Error writing to EQ\n");
	}

	return ret;
}

static int tcpx_pep_sock_create(struct tcpx_pep *pep)
{
	int ret, af;

	switch (pep->info.addr_format) {
	case FI_SOCKADDR:
	case FI_SOCKADDR_IN:
	case FI_SOCKADDR_IN6:
		af = ((struct sockaddr *)pep->info.src_addr)->sa_family;
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"invalid source address format\n");
		return -FI_EINVAL;
	}

	pep->sock = ofi_socket(af, SOCK_STREAM, 0);
	if (pep->sock == INVALID_SOCKET) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to create listener: %s\n",
			strerror(ofi_sockerr()));
		return -FI_EIO;
	}

	ret = tcpx_setup_socket(pep->sock);
	if (ret) {
		goto err;
	}

	ret = bind(pep->sock, pep->info.src_addr,
		   (socklen_t) pep->info.src_addrlen);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to bind listener: %s\n",
			strerror(ofi_sockerr()));
		goto err;
	}
	return FI_SUCCESS;
err:
	ofi_close_socket(pep->sock);
	pep->sock = INVALID_SOCKET;
	return ret;
}

static int tcpx_ep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct tcpx_ep *tcpx_ep;
	size_t addrlen_in = *addrlen;
	int ret;

	tcpx_ep = container_of(fid, struct tcpx_ep, util_ep.ep_fid);
	ret = ofi_getsockname(tcpx_ep->conn_fd, addr, (socklen_t *)addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen)? -FI_ETOOSMALL: FI_SUCCESS;
}

static int tcpx_ep_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct tcpx_ep *tcpx_ep;
	size_t addrlen_in = *addrlen;
	int ret;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	ret = ofi_getpeername(tcpx_ep->conn_fd, addr, (socklen_t *)addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen)? -FI_ETOOSMALL: FI_SUCCESS;
}

static struct fi_ops_cm tcpx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = tcpx_ep_getname,
	.getpeer = tcpx_ep_getpeer,
	.connect = tcpx_ep_connect,
	.listen = fi_no_listen,
	.accept = tcpx_ep_accept,
	.reject = fi_no_reject,
	.shutdown = tcpx_ep_shutdown,
	.join = fi_no_join,
};

static void tcpx_ep_tx_rx_queues_release(struct tcpx_ep *ep)
{
	struct slist_entry *entry;
	struct tcpx_xfer_entry *xfer_entry;
	struct tcpx_cq *tcpx_cq;

	fastlock_acquire(&ep->lock);
	while (!slist_empty(&ep->tx_queue)) {
		entry = ep->tx_queue.head;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		slist_remove_head(&ep->tx_queue);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.tx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}

	while (!slist_empty(&ep->rx_queue)) {
		entry = ep->rx_queue.head;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		slist_remove_head(&ep->rx_queue);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.rx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}

	while (!slist_empty(&ep->rma_read_queue)) {
		entry = ep->rma_read_queue.head;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		slist_remove_head(&ep->rma_read_queue);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.tx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}

	while (!slist_empty(&ep->tx_rsp_pend_queue)) {
		entry = ep->tx_rsp_pend_queue.head;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		slist_remove_head(&ep->tx_rsp_pend_queue);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.tx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}

	fastlock_release(&ep->lock);
}

static int tcpx_ep_close(struct fid *fid)
{
	struct tcpx_ep *ep = container_of(fid, struct tcpx_ep,
					  util_ep.ep_fid.fid);

	tcpx_ep_tx_rx_queues_release(ep);
	tcpx_cq_wait_ep_del(ep);
	if (ep->util_ep.eq->wait)
		ofi_wait_fd_del(ep->util_ep.eq->wait, ep->conn_fd);

	ofi_close_socket(ep->conn_fd);
	ofi_endpoint_close(&ep->util_ep);
	fastlock_destroy(&ep->lock);

	free(ep);
	return 0;
}

static int tcpx_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct tcpx_ep *ep;

	ep = container_of(fid, struct tcpx_ep, util_ep.ep_fid.fid);
	switch (command) {
	case FI_ENABLE:
		if (!ep->util_ep.rx_cq || !ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
}
static int tcpx_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcpx_ep *tcpx_ep;
	struct tcpx_rx_ctx *rx_ctx;

	tcpx_ep = container_of(fid, struct tcpx_ep, util_ep.ep_fid.fid);

	if (bfid->fclass == FI_CLASS_SRX_CTX) {
		rx_ctx = container_of(bfid, struct tcpx_rx_ctx, rx_fid.fid);
		tcpx_ep->srx_ctx = rx_ctx;
		return FI_SUCCESS;
	}

	return ofi_ep_bind(&tcpx_ep->util_ep, bfid, flags);
}

static struct fi_ops tcpx_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_ep_close,
	.bind = tcpx_ep_bind,
	.control = tcpx_ep_ctrl,
	.ops_open = fi_no_ops_open,
};
static int tcpx_ep_getopt(fid_t fid, int level, int optname,
			  void *optval, size_t *optlen)
{
	if (level != FI_OPT_ENDPOINT)
		return -ENOPROTOOPT;

	switch (optname) {
	case FI_OPT_CM_DATA_SIZE:
		if (*optlen < sizeof(size_t)) {
			*optlen = sizeof(size_t);
			return -FI_ETOOSMALL;
		}
		*((size_t *) optval) = TCPX_MAX_CM_DATA_SIZE;
		*optlen = sizeof(size_t);
		break;
	default:
		return -FI_ENOPROTOOPT;
	}
	return FI_SUCCESS;
}

static struct fi_ops_ep tcpx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = tcpx_ep_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void tcpx_empty_progress(struct tcpx_ep *ep)
{
}

int tcpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct tcpx_ep *ep;
	struct tcpx_pep *pep;
	struct tcpx_conn_handle *handle;
	int ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &tcpx_util_prov, info, &ep->util_ep,
				context, tcpx_progress);
	if (ret)
		goto err1;

	if (info->handle) {
		if (((fid_t) info->handle)->fclass == FI_CLASS_PEP) {
			pep = container_of(info->handle, struct tcpx_pep,
					   util_pep.pep_fid.fid);

			ep->conn_fd = pep->sock;
			pep->sock = INVALID_SOCKET;
		} else {
			handle = container_of(info->handle,
					      struct tcpx_conn_handle, handle);
			ep->conn_fd = handle->conn_fd;
			free(handle);

			ret = tcpx_setup_socket(ep->conn_fd);
			if (ret)
				goto err3;
		}
	} else {
		ep->conn_fd = ofi_socket(ofi_get_sa_family(info), SOCK_STREAM, 0);
		if (ep->conn_fd == INVALID_SOCKET) {
			ret = -ofi_sockerr();
			goto err2;
		}

		ret = tcpx_setup_socket(ep->conn_fd);
		if (ret)
			goto err3;
	}

	ep->cm_state = TCPX_EP_CONNECTING;
	ep->progress_func = tcpx_empty_progress;
	ret = fastlock_init(&ep->lock);
	if (ret)
		goto err3;

	ep->stage_buf.size = STAGE_BUF_SIZE;
	ep->stage_buf.len = 0;
	ep->stage_buf.off = 0;

	slist_init(&ep->rx_queue);
	slist_init(&ep->tx_queue);
	slist_init(&ep->rma_read_queue);
	slist_init(&ep->tx_rsp_pend_queue);

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcpx_ep_fi_ops;
	(*ep_fid)->ops = &tcpx_ep_ops;
	(*ep_fid)->cm = &tcpx_cm_ops;
	(*ep_fid)->msg = &tcpx_msg_ops;
	(*ep_fid)->rma = &tcpx_rma_ops;

	ep->get_rx_entry[ofi_op_msg] = tcpx_get_rx_entry_op_msg;
	ep->get_rx_entry[ofi_op_tagged] = tcpx_get_rx_entry_op_invalid;
	ep->get_rx_entry[ofi_op_read_req] = tcpx_get_rx_entry_op_read_req;
	ep->get_rx_entry[ofi_op_read_rsp] = tcpx_get_rx_entry_op_read_rsp;
	ep->get_rx_entry[ofi_op_write] =tcpx_get_rx_entry_op_write;
	return 0;
err3:
	ofi_close_socket(ep->conn_fd);
err2:
	ofi_endpoint_close(&ep->util_ep);
err1:
	free(ep);
	return ret;
}

static int tcpx_pep_fi_close(struct fid *fid)
{
	struct tcpx_pep *pep;

	pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid.fid);
	if (pep->util_pep.eq)
		ofi_wait_fd_del(pep->util_pep.eq->wait, pep->sock);

	ofi_close_socket(pep->sock);
	ofi_pep_close(&pep->util_pep);
	free(pep);
	return 0;
}

static int tcpx_pep_fi_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcpx_pep *tcpx_pep = container_of(fid, struct tcpx_pep,
						 util_pep.pep_fid.fid);

	switch (bfid->fclass) {
	case FI_CLASS_EQ:
		return ofi_pep_bind_eq(&tcpx_pep->util_pep,
				       container_of(bfid, struct util_eq,
						    eq_fid.fid), flags);
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"invalid FID class for binding\n");
		return -FI_EINVAL;
	}
}

static struct fi_ops tcpx_pep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_pep_fi_close,
	.bind = tcpx_pep_fi_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int tcpx_pep_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct tcpx_pep *tcpx_pep;

	if ((addrlen != sizeof(struct sockaddr_in)) &&
	    (addrlen != sizeof(struct sockaddr_in6)))
		return -FI_EINVAL;

	tcpx_pep = container_of(fid, struct tcpx_pep,
				util_pep.pep_fid);

	if (tcpx_pep->sock != INVALID_SOCKET) {
		ofi_close_socket(tcpx_pep->sock);
		tcpx_pep->sock = INVALID_SOCKET;
	}

	if (tcpx_pep->info.src_addr) {
		free(tcpx_pep->info.src_addr);
		tcpx_pep->info.src_addrlen = 0;
	}


	tcpx_pep->info.src_addr = mem_dup(addr, addrlen);
	if (!tcpx_pep->info.src_addr)
		return -FI_ENOMEM;
	tcpx_pep->info.src_addrlen = addrlen;

	return tcpx_pep_sock_create(tcpx_pep);
}

static int tcpx_pep_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct tcpx_pep *tcpx_pep;
	size_t addrlen_in = *addrlen;
	int ret;

	tcpx_pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid);
	ret = ofi_getsockname(tcpx_pep->sock, addr, (socklen_t *)addrlen);
	if (ret)
		return -ofi_sockerr();

	return (addrlen_in < *addrlen)? -FI_ETOOSMALL: FI_SUCCESS;
}

static int tcpx_pep_listen(struct fid_pep *pep)
{
	struct tcpx_pep *tcpx_pep;
	int ret;

	tcpx_pep = container_of(pep,struct tcpx_pep, util_pep.pep_fid);

	if (listen(tcpx_pep->sock, SOMAXCONN)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"socket listen failed\n");
		return -ofi_sockerr();
	}

	ret = ofi_wait_fd_add(tcpx_pep->util_pep.eq->wait, tcpx_pep->sock,
			      FI_EPOLL_IN, tcpx_eq_wait_try_func,
			      NULL, &tcpx_pep->cm_ctx);

	tcpx_pep->util_pep.eq->wait->signal(tcpx_pep->util_pep.eq->wait);
	return ret;
}

static int tcpx_pep_reject(struct fid_pep *pep, fid_t handle,
			   const void *param, size_t paramlen)
{
	struct ofi_ctrl_hdr hdr;
	struct tcpx_conn_handle *tcpx_handle;
	int ret;

	tcpx_handle = container_of(handle, struct tcpx_conn_handle, handle);

	memset(&hdr, 0, sizeof(hdr));
	hdr.version = OFI_CTRL_VERSION;
	hdr.type = ofi_ctrl_nack;
	hdr.seg_size = htons((uint16_t) paramlen);

	ret = ofi_sendall_socket(tcpx_handle->conn_fd, &hdr, sizeof(hdr));
	if (!ret && paramlen)
		(void) ofi_sendall_socket(tcpx_handle->conn_fd, param, paramlen);

	ofi_shutdown(tcpx_handle->conn_fd, SHUT_RDWR);
	return ofi_close_socket(tcpx_handle->conn_fd);
}

static struct fi_ops_cm tcpx_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = tcpx_pep_setname,
	.getname = tcpx_pep_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = tcpx_pep_listen,
	.accept = fi_no_accept,
	.reject = tcpx_pep_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static int tcpx_verify_info(uint32_t version, struct fi_info *info)
{
	/* TODO: write me! */
	return 0;
}

static int  tcpx_pep_getopt(fid_t fid, int level, int optname,
			    void *optval, size_t *optlen)
{
	if ( level != FI_OPT_ENDPOINT ||
	     optname != FI_OPT_CM_DATA_SIZE)
		return -FI_ENOPROTOOPT;

	if (*optlen < sizeof(size_t)) {
		*optlen = sizeof(size_t);
		return -FI_ETOOSMALL;
	}

	*((size_t *) optval) = TCPX_MAX_CM_DATA_SIZE;
	*optlen = sizeof(size_t);
	return FI_SUCCESS;
}

static struct fi_ops_ep tcpx_pep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.getopt = tcpx_pep_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};


int tcpx_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context)
{
	struct tcpx_pep *_pep;
	int ret;

	if (!info) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"invalid info\n");
		return -FI_EINVAL;
	}

	ret = tcpx_verify_info(fabric->api_version, info);
	if (ret)
		return ret;

	_pep = calloc(1, sizeof(*_pep));
	if (!_pep)
		return -FI_ENOMEM;

	ret = ofi_pep_init(fabric, info, &_pep->util_pep, context);
	if (ret)
		goto err1;

	_pep->util_pep.pep_fid.fid.ops = &tcpx_pep_fi_ops;
	_pep->util_pep.pep_fid.cm = &tcpx_pep_cm_ops;
	_pep->util_pep.pep_fid.ops = &tcpx_pep_ops;


	_pep->info = *info;
	_pep->cm_ctx.fid = &_pep->util_pep.pep_fid.fid;
	_pep->cm_ctx.type = SERVER_SOCK_ACCEPT;
	_pep->cm_ctx.cm_data_sz = 0;
	_pep->sock = INVALID_SOCKET;

	*pep = &_pep->util_pep.pep_fid;

	if (info->src_addr) {
		ret = tcpx_pep_sock_create(_pep);
		if (ret)
			goto err2;
	}
	return FI_SUCCESS;
err2:
	ofi_pep_close(&_pep->util_pep);
err1:
	free(_pep);
	return ret;
}
