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
#include <prov.h>
#include "tcpx.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <fi_util.h>
#include <unistd.h>
#include <string.h>
#include <poll.h>
#include <arpa/inet.h>
#include <netdb.h>

static ssize_t tcpx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	return -FI_ENOSYS;
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
	return -FI_ENOSYS;
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

	return tcpx_sendmsg(ep, &msg, FI_INJECT | TCPX_NO_COMPLETION);
}

static ssize_t tcpx_senddata(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			     uint64_t data, fi_addr_t dest_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_injectdata(struct fid_ep *ep, const void *buf, size_t len,
			       uint64_t data, fi_addr_t dest_addr)
{
	return -FI_ENOSYS;
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

static void tcpx_set_sockopts(int sock)
{
	int optval;

	optval = 1;

	if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)))
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"setsockopt reuseaddr failed\n");

	if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)))
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"setsockopt nodelay failed\n");
}

static void tcpx_setup_socket(int sock)
{
	tcpx_set_sockopts(sock);
	fi_fd_nonblock(sock);
}

static int tcpx_ep_connect(struct fid_ep *ep, const void *addr,
		    const void *param, size_t paramlen)
{
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	struct sockaddr *serv_addr = (struct sockaddr *) addr;
	struct poll_fd_info *fd_info;
	struct util_fabric *util_fabric;
	struct tcpx_fabric *tcpx_fabric;
	size_t addrlen;
	int ret = FI_SUCCESS;

	util_fabric = tcpx_ep->util_ep.domain->fabric;
	tcpx_fabric = container_of(util_fabric, struct tcpx_fabric, util_fabric);

	if (!addr || !tcpx_ep->conn_fd)
		return -FI_EINVAL;

	switch (serv_addr->sa_family) {
	case AF_INET:
		addrlen = sizeof(struct sockaddr_in);
		break;
	case AF_INET6:
		addrlen = sizeof(struct sockaddr_in6);
		break;
	default:
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"server address family invalid \n");
		return -FI_EINVAL;
	}

	ret = connect(tcpx_ep->conn_fd, (struct sockaddr *) addr, addrlen);
	if (ret == -1 && errno != FI_EINPROGRESS) {

		/* alloc for fd_entry */
		fd_info = calloc(1, sizeof(*fd_info));
		if (!fd_info) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"cannot allocate memory \n");
		}
		fd_info->fid = &tcpx_ep->util_ep.ep_fid.fid;
		fd_info->info = &tcpx_ep->info;
		fd_info->flags = 0;

		/* Add to poll list in fabric thread */
		fastlock_acquire(&tcpx_fabric->fd_list_lock);
		dlist_insert_tail(&fd_info->entry,&tcpx_fabric->fd_list);
		fastlock_release(&tcpx_fabric->fd_list_lock);
		fd_signal_set(&tcpx_fabric->signal);
	}

	return ret;
}

static int tcpx_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct fi_eq_cm_entry *eq_entry;
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	if (tcpx_ep->conn_fd == 0)
		return -FI_EINVAL;


	eq_entry  = calloc(1, sizeof(*eq_entry));
	if (!eq_entry) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory\n");
		return -FI_ENOMEM;
	}

	eq_entry->fid = &ep->fid;
	eq_entry->info = &tcpx_ep->info;

	/* report FI_CONNECTED to eq  */
	if (fi_eq_write(&tcpx_ep->util_ep.eq->eq_fid, FI_CONNECTED, eq_entry,
			sizeof(*eq_entry), 0) < 0)
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"Eq write failed: %s \n", strerror(errno));
	return 0;
}

static struct fi_ops_cm tcpx_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = tcpx_ep_connect,
	.listen = fi_no_listen,
	.accept = tcpx_ep_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static int tcpx_ep_close(struct fid *fid)
{
	struct tcpx_ep *ep;

	ep = container_of(fid, struct tcpx_ep, util_ep.ep_fid.fid);
	ofi_close_socket(ep->conn_fd);
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

static struct fi_ops_ep tcpx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void tcpx_manual_progress(struct util_ep *util_ep)
{
	return;
}

int tcpx_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct tcpx_ep *ep;
	struct tcpx_conn_handle *handle;
	int ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	ret = ofi_endpoint_init(domain, &tcpx_util_prov, info, &ep->util_ep,
				context, tcpx_manual_progress);
	if (ret)
		goto err;

	ep->info = *info;
	if (info && info->handle) {
		handle = container_of(info->handle,
				      struct tcpx_conn_handle,
				      handle);
		ep->conn_fd = handle->conn_fd;
		free(handle);
	} else {
		ep->conn_fd = ofi_socket(AF_INET, SOCK_STREAM, 0);
		if (!ep->conn_fd) {
		  ret = -errno;
		  goto err;
		}
		tcpx_setup_socket(ep->conn_fd);
	}

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcpx_ep_fi_ops;
	(*ep_fid)->ops = &tcpx_ep_ops;
	(*ep_fid)->cm = &tcpx_cm_ops;
	(*ep_fid)->msg = &tcpx_msg_ops;

	return 0;
err:
	free(ep);
	return ret;
}

static int tcpx_pep_fi_close(struct fid *fid)
{
	struct tcpx_pep *pep;
	struct poll_fd_info *fd_info;
	struct tcpx_fabric *tcpx_fabric;

	pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid.fid);

	tcpx_fabric = container_of(pep->util_pep.fabric,
				   struct tcpx_fabric,
				   util_fabric);

	fd_info = calloc(1, sizeof(*fd_info));
	if (!fd_info) {
	  FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
		  "cannot allocate memory \n");
	}
	/* remove the pep socket from fabric thread */
	fd_info->fid = &pep->util_pep.pep_fid.fid;
	fd_info->info = &pep->info;
	fd_info->flags = TCPX_SOCK_DEL;

	/* Add to poll list in fabric thread */
	fastlock_acquire(&tcpx_fabric->fd_list_lock);
	dlist_insert_tail(&fd_info->entry,&tcpx_fabric->fd_list);
	fastlock_release(&tcpx_fabric->fd_list_lock);
	fd_signal_set(&tcpx_fabric->signal);

	/* waiting for actual removal before freeing pep */
	while (!pep->sock_fd_closed);

	free(pep);
	return 0;
}

static int tcpx_pep_fi_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcpx_pep *pep;
	struct util_eq *eq;

	pep = container_of(fid, struct tcpx_pep, util_pep.pep_fid.fid);

	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;
	eq = container_of(bfid, struct util_eq, eq_fid.fid);

	if(ofi_pep_bind_eq(&pep->util_pep, eq)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"Cannot bind Passive EP and EQ on different fabrics\n");
		return -FI_EINVAL;
	}

	return 0;
}

static struct fi_ops tcpx_pep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_pep_fi_close,
	.bind = tcpx_pep_fi_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static int tcpx_pep_listen(struct fid_pep *pep)
{
	struct tcpx_pep *tcpx_pep;
	struct poll_fd_info *fd_info;
	struct util_fabric *util_fabric;
	struct tcpx_fabric *tcpx_fabric;
	int ret = FI_SUCCESS;

	tcpx_pep = container_of(pep,struct tcpx_pep, util_pep.pep_fid);
	util_fabric = tcpx_pep->util_pep.fabric;
	tcpx_fabric = container_of(util_fabric,
				   struct tcpx_fabric,
				   util_fabric);

	if (listen(tcpx_pep->sock, SOMAXCONN) < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"socket listen failed\n");
		ret = -errno;
		goto out;
	}

	/* alloc for fd_entry */
	fd_info = calloc(1, sizeof(*fd_info));
	if (!fd_info) {
	  FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
		  "cannot allocate memory \n");
	}
	fd_info->fid = &tcpx_pep->util_pep.pep_fid.fid;
	fd_info->info = &tcpx_pep->info;
	fd_info->flags = TCPX_SOCK_ADD;

	/* Add to poll list in fabric thread */
	fastlock_acquire(&tcpx_fabric->fd_list_lock);
	dlist_insert_tail(&fd_info->entry,&tcpx_fabric->fd_list);
	fastlock_release(&tcpx_fabric->fd_list_lock);
	fd_signal_set(&tcpx_fabric->signal);

	return ret;
 out:
	tcpx_pep->sock = -1;
	return ret;
}

static struct fi_ops_cm tcpx_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = fi_no_setname,
	.getname = fi_no_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = tcpx_pep_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};


static int tcpx_verify_info(uint32_t version, struct fi_info *info)
{
	return 0;
}

static struct fi_ops_ep tcpx_pep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.getopt = fi_no_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};


int tcpx_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_pep **pep, void *context)
{
	int ret;
	struct tcpx_pep *_pep;
	struct addrinfo hints, *result, *iter;
	char sa_ip[INET_ADDRSTRLEN] = {0};
	char sa_port[NI_MAXSERV] = {0};

	if (info) {
		ret = tcpx_verify_info(fabric->api_version, info);
		if (ret) {
			return ret;
		}
	} else {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"invalid info\n");
		return -FI_EINVAL;

	}

	_pep = calloc(1, sizeof(*_pep));
	if (!_pep)
		return -FI_ENOMEM;

	if (ofi_pep_init(fabric, info, &_pep->util_pep, context)) {

		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"pep initialization failed\n");
		ret = -FI_EINVAL;
		goto err;
	}

	_pep->info = *info;

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;

	if (info->src_addr) {

		switch (info->addr_format) {
		case FI_SOCKADDR:
		case FI_SOCKADDR_IN:
		case FI_SOCKADDR_IN6:
			ret = getnameinfo((const struct sockaddr *) info->src_addr,
				    info->src_addrlen,
				    sa_ip, INET_ADDRSTRLEN,
				    sa_port, NI_MAXSERV,0);
			if (ret) {
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"pep initialization failed\n");
				goto err;
			}
			break;
		default:
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"invalid source address format\n");
			ret = -FI_EINVAL;
			goto err;
		}
		ret = getaddrinfo(sa_ip, sa_port, &hints, &result);
	} else {
		ret = getaddrinfo("localhost", NULL, &hints, &result);
	}

	if (ret) {
		ret = -FI_EINVAL;
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"getaddrinfo failed");
		goto err;
	}

	for (iter = result; iter; iter = iter->ai_next) {
		_pep->sock = ofi_socket(iter->ai_family, iter->ai_socktype,
				      iter->ai_protocol);
		if (_pep->sock >= 0) {
			tcpx_set_sockopts(_pep->sock);
			if (!bind(_pep->sock, result->ai_addr, result->ai_addrlen))
				break;
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"failed to bind listener: %s\n", strerror(errno));
			ofi_close_socket(_pep->sock);
			_pep->sock = -1;
		}
	}

	if (_pep->sock < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to create listener: %s\n", strerror(errno));
		ret = -FI_EIO;
		goto err;
	}
	freeaddrinfo(result);

	_pep->util_pep.pep_fid.fid.fclass = FI_CLASS_PEP;
	_pep->util_pep.pep_fid.fid.context = context;
	_pep->util_pep.pep_fid.fid.ops = &tcpx_pep_fi_ops;
	_pep->util_pep.pep_fid.cm = &tcpx_pep_cm_ops;
	_pep->util_pep.pep_fid.ops = &tcpx_pep_ops;

	_pep->util_pep.fabric = container_of(fabric, struct util_fabric, fabric_fid);

	*pep = &_pep->util_pep.pep_fid;
	return 0;
err:
	free(_pep);
	return ret;
}
