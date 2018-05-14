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

static ssize_t tcpx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	struct tcpx_xfer_entry *recv_entry;
	struct tcpx_ep *tcpx_ep;
	struct tcpx_cq *tcpx_cq;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	tcpx_cq = container_of(tcpx_ep->util_ep.rx_cq, struct tcpx_cq,
			       util_cq);

	assert(msg->iov_count < TCPX_IOV_LIMIT);

	recv_entry = tcpx_xfer_entry_alloc(tcpx_cq);
	if (!recv_entry)
		return -FI_EAGAIN;

	recv_entry->msg_data.iov_cnt = msg->iov_count;
	memcpy(&recv_entry->msg_data.iov[0], &msg->msg_iov[0],
	       msg->iov_count * sizeof(struct iovec));

	recv_entry->ep = tcpx_ep;
	recv_entry->flags = flags;
	recv_entry->context = msg->context;
	recv_entry->done_len = 0;

	fastlock_acquire(&tcpx_ep->queue_lock);
	dlist_insert_tail(&recv_entry->entry, &tcpx_ep->rx_queue);
	fastlock_release(&tcpx_ep->queue_lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;

	return tcpx_recvmsg(ep, &msg, 0);
}

static ssize_t tcpx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;
	return tcpx_recvmsg(ep, &msg, 0);
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

	tx_entry = tcpx_xfer_entry_alloc(tcpx_cq);
	if (!tx_entry)
		return -FI_EAGAIN;

	assert(msg->iov_count <= TCPX_IOV_LIMIT);

	data_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	assert(!(flags & FI_INJECT) || (data_len <= TCPX_MAX_INJECT_SZ));

	tx_entry->msg_hdr.hdr.version = OFI_CTRL_VERSION;
	tx_entry->msg_hdr.hdr.op = ofi_op_msg;
	tx_entry->msg_hdr.hdr.op_data = TCPX_OP_MSG_SEND;
	tx_entry->msg_hdr.hdr.size = htonll(data_len + sizeof(tx_entry->msg_hdr));

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

	if (flags & FI_REMOTE_CQ_DATA) {
		tx_entry->msg_hdr.hdr.flags |= OFI_REMOTE_CQ_DATA;
		tx_entry->msg_hdr.hdr.data = htonll(msg->data);
	}

	tx_entry->msg_hdr.hdr.flags = htonl(tx_entry->msg_hdr.hdr.flags);
	tx_entry->ep = tcpx_ep;
	tx_entry->context = msg->context;
	tx_entry->done_len = 0;

	fastlock_acquire(&tcpx_ep->queue_lock);
	if (dlist_empty(&tcpx_ep->tx_queue)) {
		dlist_insert_tail(&tx_entry->entry, &tcpx_ep->tx_queue);
		process_tx_entry(tx_entry);
	} else {
		dlist_insert_tail(&tx_entry->entry, &tcpx_ep->tx_queue);
	}
	fastlock_release(&tcpx_ep->queue_lock);
	return FI_SUCCESS;
}

static ssize_t tcpx_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			 fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;

	return tcpx_sendmsg(ep, &msg, 0);
}

static ssize_t tcpx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;

	return tcpx_sendmsg(ep, &msg, 0);
}


static ssize_t tcpx_inject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = NULL;

	return tcpx_sendmsg(ep, &msg, FI_INJECT | TCPX_NO_COMPLETION);
}

static ssize_t tcpx_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			     uint64_t data, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = NULL;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = NULL;
	msg.data = data;

	return tcpx_sendmsg(ep, &msg, FI_REMOTE_CQ_DATA);
}

static ssize_t tcpx_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = NULL;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = NULL;
	msg.data = 0;

	return tcpx_sendmsg(ep, &msg, FI_REMOTE_CQ_DATA | FI_INJECT |
			    TCPX_NO_COMPLETION);
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
	struct poll_fd_info *fd_info;
	struct util_fabric *util_fabric;
	struct tcpx_fabric *tcpx_fabric;
	int ret;

	util_fabric = tcpx_ep->util_ep.domain->fabric;
	tcpx_fabric = container_of(util_fabric, struct tcpx_fabric, util_fabric);

	if (!addr || !tcpx_ep->conn_fd || paramlen > TCPX_MAX_CM_DATA_SIZE)
		return -FI_EINVAL;

	fd_info = calloc(1, sizeof(*fd_info));
	if (!fd_info) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		return -FI_ENOMEM;
	}

	ret = connect(tcpx_ep->conn_fd, (struct sockaddr *) addr,
		      (socklen_t) ofi_sizeofaddr(addr));
	if (ret && errno != FI_EINPROGRESS) {
		free(fd_info);
		return -errno;
	}

	fd_info->fid = &tcpx_ep->util_ep.ep_fid.fid;
	fd_info->flags = POLL_MGR_FREE;
	fd_info->type = CONNECT_SOCK;
	fd_info->state = ESTABLISH_CONN;

	if (paramlen) {
		fd_info->cm_data_sz = paramlen;
		memcpy(fd_info->cm_data, param, paramlen);
	}

	fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
	dlist_insert_tail(&fd_info->entry, &tcpx_fabric->poll_mgr.list);
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);
	fastlock_release(&tcpx_fabric->poll_mgr.lock);
	return 0;
}

static int tcpx_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	struct poll_fd_info *fd_info;
	struct util_fabric *util_fabric;
	struct tcpx_fabric *tcpx_fabric;

	util_fabric = tcpx_ep->util_ep.domain->fabric;
	tcpx_fabric = container_of(util_fabric, struct tcpx_fabric, util_fabric);

	if (tcpx_ep->conn_fd == INVALID_SOCKET)
		return -FI_EINVAL;

	fd_info = calloc(1, sizeof(*fd_info));
	if (!fd_info) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory \n");
		return -FI_ENOMEM;
	}

	fd_info->fid = &tcpx_ep->util_ep.ep_fid.fid;
	fd_info->flags = POLL_MGR_FREE;
	fd_info->type = ACCEPT_SOCK;
	if (paramlen) {
		fd_info->cm_data_sz = paramlen;
		memcpy(fd_info->cm_data, param, paramlen);
	}

	fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
	dlist_insert_tail(&fd_info->entry, &tcpx_fabric->poll_mgr.list);
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);
	fastlock_release(&tcpx_fabric->poll_mgr.lock);
	return 0;
}

static int tcpx_ep_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct tcpx_ep *tcpx_ep;
	int ret;

	tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	ret = ofi_shutdown(tcpx_ep->conn_fd, SHUT_RDWR);
	if (ret && errno != ENOTCONN) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_DATA, "ep shutdown unsuccessful\n");
	}

	ret = tcpx_ep_shutdown_report(tcpx_ep, &ep->fid);
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
			"failed to create listener: %s\n", strerror(errno));
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
			"failed to bind listener: %s\n", strerror(errno));
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

	tcpx_ep = container_of(fid, struct tcpx_ep, util_ep.ep_fid);
	if (getsockname(tcpx_ep->conn_fd, addr, (socklen_t *)addrlen))
		return (addrlen_in < *addrlen)? -FI_ETOOSMALL: -errno;

	return FI_SUCCESS;
}

static struct fi_ops_cm tcpx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = tcpx_ep_getname,
	.getpeer = fi_no_getpeer,
	.connect = tcpx_ep_connect,
	.listen = fi_no_listen,
	.accept = tcpx_ep_accept,
	.reject = fi_no_reject,
	.shutdown = tcpx_ep_shutdown,
	.join = fi_no_join,
};

static void tcpx_ep_tx_rx_queues_release(struct tcpx_ep *ep)
{
	struct dlist_entry *entry;
	struct tcpx_xfer_entry *xfer_entry;
	struct tcpx_cq *tcpx_cq;

	fastlock_acquire(&ep->queue_lock);
	while (!dlist_empty(&ep->tx_queue)) {
		entry = ep->tx_queue.next;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		dlist_remove(entry);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.tx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}

	while (!dlist_empty(&ep->rx_queue)) {
		entry = ep->rx_queue.next;
		xfer_entry = container_of(entry, struct tcpx_xfer_entry, entry);
		dlist_remove(entry);
		tcpx_cq = container_of(xfer_entry->ep->util_ep.rx_cq,
				       struct tcpx_cq, util_cq);
		tcpx_xfer_entry_release(tcpx_cq, xfer_entry);
	}
	fastlock_release(&ep->queue_lock);
}

static int tcpx_ep_close(struct fid *fid)
{
	struct tcpx_ep *ep = container_of(fid, struct tcpx_ep,
					  util_ep.ep_fid.fid);

	tcpx_ep_tx_rx_queues_release(ep);
	tcpx_progress_ep_del(ep);
	ofi_close_socket(ep->conn_fd);
	fastlock_destroy(&ep->queue_lock);
	fastlock_destroy(&ep->cm_state_lock);
	ofi_endpoint_close(&ep->util_ep);

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
	struct util_ep *util_ep;

	util_ep = container_of(fid, struct util_ep, ep_fid.fid);
	return ofi_ep_bind(util_ep, bfid, flags);
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

static SOCKET create_ep_sock_from_pep_sock(SOCKET pep_sock)
{
	struct sockaddr_storage ss;
	socklen_t ss_len;
	SOCKET ep_sock;
	int ret, af;

	ss_len = sizeof(ss);
	ret = ofi_getsockname(pep_sock, (struct sockaddr *)&ss, &ss_len);
	if (ret)
		return INVALID_SOCKET;

	af = ss.ss_family;

	ep_sock = ofi_socket(af, SOCK_STREAM, 0);
	if (ep_sock == INVALID_SOCKET)
		return INVALID_SOCKET;

	ret = tcpx_setup_socket(ep_sock);
	if (ret)
		goto err;

	if (bind(ep_sock, (struct sockaddr *)&ss, ss_len) != 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"bind failed \n");
		goto err;
	}
	return ep_sock;
err:
	ofi_close_socket(ep_sock);
	return INVALID_SOCKET;
}

int tcpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct tcpx_ep *ep;
	struct tcpx_pep *pep;
	struct tcpx_conn_handle *handle;
	int af, ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &tcpx_util_prov, info, &ep->util_ep,
				context, tcpx_progress);
	if (ret)
		goto err1;

	if (info->handle) {
		handle = container_of(info->handle, struct tcpx_conn_handle,
				      handle);

		if (info->handle->fclass == FI_CLASS_PEP) {
			pep = container_of(info->handle, struct tcpx_pep,
					   util_pep.pep_fid.fid);

			ep->conn_fd = create_ep_sock_from_pep_sock(pep->sock);
			if (ep->conn_fd == INVALID_SOCKET) {
				ret = -errno;
				goto err2;
			}
		} else {
			ep->conn_fd = handle->conn_fd;
			free(handle);

			ret = tcpx_setup_socket(ep->conn_fd);
			if (ret)
				goto err3;
		}
	} else {
		if (info->src_addr)
			af = ((const struct sockaddr *) info->src_addr)->sa_family;
		else if (info->dest_addr)
			af = ((const struct sockaddr *) info->dest_addr)->sa_family;
		else
			af = ofi_get_sa_family(info->addr_format);

		ep->conn_fd = ofi_socket(af, SOCK_STREAM, 0);
		if (ep->conn_fd == INVALID_SOCKET) {
			ret = -errno;
			goto err2;
		}

		ret = tcpx_setup_socket(ep->conn_fd);
		if (ret)
			goto err3;
	}

	ret = fastlock_init(&ep->queue_lock);
	if (ret)
		goto err3;

	ep->cm_state = TCPX_EP_CONNECTING;

	ret = fastlock_init(&ep->cm_state_lock);
	if (ret)
		goto err4;

	dlist_init(&ep->rx_queue);
	dlist_init(&ep->tx_queue);
	dlist_init(&ep->rma_list.list);
	ep->rma_list.msg_id_tracker = 0;

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcpx_ep_fi_ops;
	(*ep_fid)->ops = &tcpx_ep_ops;
	(*ep_fid)->cm = &tcpx_cm_ops;
	(*ep_fid)->msg = &tcpx_msg_ops;
	(*ep_fid)->rma = &tcpx_rma_ops;

	return 0;
err4:
	fastlock_destroy(&ep->queue_lock);
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
	struct tcpx_fabric *tcpx_fabric;

	pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid.fid);

	tcpx_fabric = container_of(pep->util_pep.fabric, struct tcpx_fabric,
				   util_fabric);


	fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
	if (pep->state != TCPX_PEP_LISTENING) {
		pep->state = TCPX_PEP_CLOSED;
		fastlock_release(&tcpx_fabric->poll_mgr.lock);
		goto out;
	}

	pep->poll_info.flags = POLL_MGR_DEL;
	if (pep->poll_info.entry.next == pep->poll_info.entry.prev)
		dlist_insert_tail(&pep->poll_info.entry,
				  &tcpx_fabric->poll_mgr.list);
	pep->state = TCPX_PEP_CLOSED;
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);
	fastlock_release(&tcpx_fabric->poll_mgr.lock);

	while (!(pep->poll_info.flags & POLL_MGR_ACK))
		sleep(0);
out:
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

	tcpx_pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid);
	if (getsockname(tcpx_pep->sock, addr, (socklen_t *)addrlen))
		return (addrlen_in < *addrlen)? -FI_ETOOSMALL: -errno;

	return FI_SUCCESS;
}

static int tcpx_pep_listen(struct fid_pep *pep)
{
	struct tcpx_pep *tcpx_pep;
	struct tcpx_fabric *tcpx_fabric;

	tcpx_pep = container_of(pep,struct tcpx_pep, util_pep.pep_fid);
	tcpx_fabric = container_of(tcpx_pep->util_pep.fabric,
				   struct tcpx_fabric, util_fabric);

	if (listen(tcpx_pep->sock, SOMAXCONN)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"socket listen failed\n");
		return -errno;
	}

	fastlock_acquire(&tcpx_fabric->poll_mgr.lock);
	tcpx_pep->state = TCPX_PEP_LISTENING;
	dlist_insert_tail(&tcpx_pep->poll_info.entry, &tcpx_fabric->poll_mgr.list);
	fd_signal_set(&tcpx_fabric->poll_mgr.signal);
	fastlock_release(&tcpx_fabric->poll_mgr.lock);

	return 0;
}

static int tcpx_pep_reject(struct fid_pep *pep, fid_t handle,
			   const void *param, size_t paramlen)
{
	struct ofi_ctrl_hdr hdr;
	struct tcpx_conn_handle *tcpx_handle;

	tcpx_handle = container_of(handle, struct tcpx_conn_handle, handle);

	memset(&hdr, 0, sizeof(hdr));
	hdr.version = OFI_CTRL_VERSION;
	hdr.type = ofi_ctrl_nack;
	hdr.seg_size = paramlen;

	ofi_send_socket(tcpx_handle->conn_fd, &hdr,
			sizeof(hdr), 0);

	if (paramlen)
		ofi_send_socket(tcpx_handle->conn_fd, param,
				paramlen, 0);

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
	_pep->poll_info.fid = &_pep->util_pep.pep_fid.fid;
	_pep->poll_info.type = PASSIVE_SOCK;
	_pep->poll_info.flags = 0;
	_pep->poll_info.cm_data_sz = 0;
	dlist_init(&_pep->poll_info.entry);
	_pep->sock = INVALID_SOCKET;
	_pep->state = TCPX_PEP_CREATED;

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
