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


static ssize_t tcpx_recv(struct fid_ep *ep, void *buf, size_t len, void *desc,
			 fi_addr_t src_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_recvv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t src_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_send(struct fid_ep *ep, const void *buf, size_t len, void *desc,
			 fi_addr_t dest_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			  size_t count, fi_addr_t dest_addr, void *context)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			    uint64_t flags)
{
	return -FI_ENOSYS;
}

static ssize_t tcpx_inject(struct fid_ep *ep, const void *buf, size_t len,
			   fi_addr_t dest_addr)
{
	return -FI_ENOSYS;
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

static void *tcpx_ep_connect_handler(void *data)
{
	struct tcpx_conn_handle *handle;
	struct fi_eq_cm_entry *eq_entry;
	struct tcpx_ep *tcpx_ep = (struct tcpx_ep *) data;

	handle = tcpx_ep->handle;

	eq_entry  = calloc(1, sizeof(*eq_entry));
	if (eq_entry) {
	  FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
		  "cannot allocate memory\n");
	  goto err;
	}

	eq_entry->fid = &tcpx_ep->util_ep.ep_fid.fid;
	eq_entry->info = &tcpx_ep->info;

	tcpx_set_sockopts(handle->conn_fd);
	if (connect(handle->conn_fd,
		    (struct sockaddr *)handle->serv_addr,
		    sizeof(*handle->serv_addr))) {

	  FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
		  "connect failed : %s\n", strerror(errno));
	  goto err1;
	}

	/* report success to eq with FI_CONNECTED */
	fi_eq_write(&tcpx_ep->eq->eq_fid, FI_CONNECTED,
		    eq_entry, sizeof(eq_entry), 0);

	return NULL;
err1:
	free(eq_entry);
err:
	ofi_close_socket(handle->conn_fd);

	/* report the error to eq */
	fi_eq_write(&tcpx_ep->eq->eq_fid, FI_SHUTDOWN,
		    eq_entry, sizeof(eq_entry), 0);

	return NULL;
}

static int tcpx_ep_connect(struct fid_ep *ep, const void *addr,
		    const void *param, size_t paramlen)
{
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);
	struct sockaddr_in *serv_addr = (struct sockaddr_in *) addr;
	struct tcpx_conn_handle *handle;
	int ret = 0;

	if (!addr || tcpx_ep->handle)
		return -FI_EINVAL;


	handle  = calloc(1, sizeof(*handle));
	if (handle) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"cannot allocate memory\n");
		return -FI_ENOMEM;
	}

	handle->conn_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (handle->conn_fd < 0) {
		ret = handle->conn_fd;
		goto out;
	}
	tcpx_ep->handle = handle;
	tcpx_ep->handle->serv_addr = serv_addr;

	if (pthread_create(&tcpx_ep->cm_thread, NULL,
			   tcpx_ep_connect_handler, (void *) tcpx_ep)) {

		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to create connect thread\n");
		goto err;
	}

	return FI_SUCCESS;
err:
	ofi_close_socket(handle->conn_fd);
out:
	free(handle);
	return ret;

}

static int tcpx_ep_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct fi_eq_cm_entry *eq_entry;
	struct tcpx_ep *tcpx_ep = container_of(ep, struct tcpx_ep, util_ep.ep_fid);

	if (tcpx_ep->handle != NULL && tcpx_ep->handle->conn_fd != 9) {

		eq_entry  = calloc(1, sizeof(*eq_entry));
		if (eq_entry) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"cannot allocate memory\n");
			return -FI_ENOMEM;
		}

		eq_entry->fid = &ep->fid;
		eq_entry->info = &tcpx_ep->info;

		/* report FI_CONNECTED to eq  */
		fi_eq_write(&tcpx_ep->eq->eq_fid, FI_CONNECTED, eq_entry,
			    sizeof(eq_entry), 0);
		return 0;
	}
	return -FI_EINVAL;
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
	if (ofi_atomic_get32(&ep->ref)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL, "EP busy\n");
		return -FI_EBUSY;
	}

	ofi_close_socket(ep->sock);
	ofi_endpoint_close(&ep->util_ep);
	free(ep);
	return 0;
}

static int tcpx_ep_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	return -FI_ENOSYS;
}

static struct fi_ops tcpx_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_ep_close,
	.bind = tcpx_ep_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_ep tcpx_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = fi_no_getopt, //tcpx_getopt,
	.setopt = fi_no_setopt, //tcpx_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static void tcpx_ep_progress(struct util_ep *util_ep)
{
}

static int tcpx_ep_init(struct tcpx_ep *ep, struct fi_info *info)
{
	return -FI_ENOSYS;
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
				context, tcpx_ep_progress);
	if (ret)
		goto err;

	ret = tcpx_ep_init(ep, info);
	if (ret) {
		free(ep);
		return ret;
	}

	*ep_fid = &ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &tcpx_ep_fi_ops;
	(*ep_fid)->ops = &tcpx_ep_ops;
	(*ep_fid)->cm = &tcpx_cm_ops;
	(*ep_fid)->msg = &tcpx_msg_ops;

	if (info != NULL && info->handle != NULL){
		handle = container_of(info->handle,struct tcpx_conn_handle,
				      handle);
		ep->handle = handle;
	} else {
		ret = -FI_EINVAL;
		goto err;
	}
	return 0;
err:
	free(ep);
	return ret;
}

static int tcpx_pep_fi_close(struct fid *fid)
{
	struct tcpx_pep *pep;
	int ret;
	char c;

	pep = container_of(fid, struct tcpx_pep, pep.fid);
	pep->cm.do_listen = 0;

	ret = ofi_write_socket(pep->cm.signal_fds[0], &c, 1);
	if (ret != 1)
		FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL,"Failed to signal\n");

	if (pep->cm.listener_thread &&
	    pthread_join(pep->cm.listener_thread, NULL)) {
		FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL,"pthread join failed\n");
	}

	ofi_close_socket(pep->cm.signal_fds[0]);
	ofi_close_socket(pep->cm.signal_fds[1]);

	free(pep);
	return 0;

}

static int tcpx_pep_fi_bind(struct fid *fid, struct fid *bfid, uint64_t flags)
{
	struct tcpx_pep *pep;
	struct util_eq *eq;

	pep = container_of(fid, struct tcpx_pep, pep.fid);

	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;

	eq = container_of(bfid, struct util_eq, eq_fid.fid);
	if (pep->fabric != eq->fabric) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"Cannot bind Passive EP and EQ on different fabric\n");
		return -FI_EINVAL;
	}
	pep->eq = eq;
	return 0;
}

static struct fi_ops tcpx_pep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = tcpx_pep_fi_close,
	.bind = tcpx_pep_fi_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static void *tcpx_cm_listener_thread(void *pep_data)
{
	struct tcpx_pep *pep = (struct tcpx_pep *) pep_data;
	struct fi_eq_cm_entry *cm_entry = NULL;
	struct tcpx_conn_handle *handle;
	struct pollfd poll_fds[2];
	int ret = 0, conn_fd, entry_sz;
	char tmp = 0;

	FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL,
	       "Starting listener thread for PEP: %p\n", pep);

	poll_fds[0].fd = pep->cm.sock;
	poll_fds[1].fd = pep->cm.signal_fds[1];
	poll_fds[0].events = poll_fds[1].events = POLLIN;

	while (*((volatile int *) &pep->cm.do_listen)) {
		/* block on the listen fd until a req comes through */
		ret = poll(poll_fds, 1, -1);
		if (ret > 0) {
			if (poll_fds[1].revents & POLLIN) {
				ret = ofi_read_socket(pep->cm.signal_fds[1], &tmp, 1);
				if (ret != 1)
					FI_DBG(&tcpx_prov, FI_LOG_EP_CTRL,
					       "Invalid signal\n");
				continue;
			}

		} else {
			break;
		}

		conn_fd = accept(pep->cm.sock, NULL, 0);
		if (conn_fd < 0) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"failed to accept: %d\n", errno);
			continue;
		}

		handle = calloc(1, sizeof(*handle));
		if (!handle) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"cannot allocate memory\n");

			ofi_close_socket(conn_fd);
			break;
		}

		entry_sz = sizeof(*cm_entry);
		cm_entry = calloc(1, entry_sz);
		if (!cm_entry) {
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"cannot allocate memory\n");
			goto err;
		}

		cm_entry->fid = &pep->pep.fid;
		cm_entry->info = &pep->info;
		cm_entry->info->handle = &handle->handle;

		/* report the conn req to the associated eq */
		if (fi_eq_write(&pep->eq->eq_fid, FI_CONNREQ, cm_entry, entry_sz, 0))
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"Error in writing to EQ\n");
		free(cm_entry);
	}
	return NULL;
err:
	ofi_close_socket(conn_fd);
	free(cm_entry);
	free(handle);
	return NULL;
}

static int tcpx_pep_listen(struct fid_pep *pep)
{
	struct tcpx_pep *_pep;
	/* struct fi_info *fi; */
	struct addrinfo hints, *addr_result, *iter;
	char sa_ip[INET_ADDRSTRLEN] = {0};
	char sa_port[NI_MAXSERV] = {0};
	int ret;

	_pep = container_of(pep,struct tcpx_pep, pep);

	/* fi = &_pep->info; */

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;

	memcpy(sa_ip, inet_ntoa(_pep->src_addr.sin_addr), INET_ADDRSTRLEN);
	sprintf(sa_port, "%d", ntohs(_pep->src_addr.sin_port));

	ret = getaddrinfo(sa_ip, sa_port, &hints, &addr_result);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"no available AF_INET address service:%s, %s\n",
			sa_port, gai_strerror(ret));
		return -FI_EINVAL;
	}

	for (iter = addr_result; iter; iter = iter->ai_next) {
		_pep->cm.sock = socket(iter->ai_family, iter->ai_socktype,
				      iter->ai_protocol);
		if (_pep->cm.sock >= 0) {
			tcpx_set_sockopts(_pep->cm.sock);
			if (!bind(_pep->cm.sock, addr_result->ai_addr, addr_result->ai_addrlen))
				break;
			FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
				"failed to bind listener: f%s\n", strerror(errno));
			ofi_close_socket(_pep->cm.sock);
			_pep->cm.sock = -1;
		}
	}

	freeaddrinfo(addr_result);

	if (_pep->cm.sock < 0) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to create listener: %s\n", strerror(errno));
		return -FI_EIO;
	}

	if (listen(_pep->cm.sock, TCPX_MAX_SOCK_REQS)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"socket listen failed\n");
		goto out;
	}

	_pep->cm.do_listen = 1;

	if (pthread_create(&_pep->cm.listener_thread, NULL,
			   tcpx_cm_listener_thread, (void *)_pep)) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,
			"failed to create cm thread\n");
		goto out;
	}

	return FI_SUCCESS;
 out:
	ofi_close_socket(_pep->cm.sock);
	_pep->cm.sock = -1;
	return -FI_ENOMEM;
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
	return -FI_ENOSYS;
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
	struct addrinfo hints, *result;

	if (info) {
		ret = tcpx_verify_info(fabric->api_version, info);
		if (ret) {
			return ret;
		}
	}
	_pep = calloc(1, sizeof(*_pep));
	if (!_pep)
		return -FI_ENOMEM;

	if (info) {
		_pep->info = *info;

		if (info->src_addr) {
			memcpy(&_pep->src_addr, info->src_addr,
			       info->src_addrlen);
		} else {
			memset(&hints, 0, sizeof(hints));
			hints.ai_family = AF_INET;
			hints.ai_socktype = SOCK_STREAM;

			ret = getaddrinfo("localhost", NULL, &hints, &result);
			if (ret) {
				ret = -FI_EINVAL;
				FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"getaddrinfo failed");
				goto err;
			}
			memcpy(&_pep->src_addr, result->ai_addr,
			       result->ai_addrlen);
			freeaddrinfo(result);
		}

	} else {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"invalid info");
		ret = -FI_EINVAL;
		goto err;
	}

	ret = socketpair(AF_UNIX, SOCK_STREAM, 0, _pep->cm.signal_fds);
	if (ret) {
		ret = -errno;
		goto err;
	}

	ret = fi_fd_nonblock(_pep->cm.signal_fds[1]);
	if (ret) {
		FI_WARN(&tcpx_prov, FI_LOG_EP_CTRL,"fi_fd_nonblock failed\n");
	}

	_pep->pep.fid.fclass = FI_CLASS_PEP;
	_pep->pep.fid.context = context;
	_pep->pep.fid.ops = &tcpx_pep_fi_ops;
	_pep->pep.cm = &tcpx_pep_cm_ops;
	_pep->pep.ops = &tcpx_pep_ops;

	_pep->fabric = container_of(fabric, struct util_fabric, fabric_fid);

	*pep = &_pep->pep;
	return 0;

err:
	free(_pep);
	return ret;
}
