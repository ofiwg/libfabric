/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
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

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <limits.h>

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

static const struct fi_ep_attr sock_msg_ep_attr = {
	.type = FI_EP_MSG,
	.protocol = FI_PROTO_SOCK_TCP,
	.protocol_version = SOCK_WIRE_PROTO_VERSION,
	.max_msg_size = SOCK_EP_MAX_MSG_SZ,
	.msg_prefix_size = SOCK_EP_MSG_PREFIX_SZ,
	.max_order_raw_size = SOCK_EP_MAX_ORDER_RAW_SZ,
	.max_order_war_size = SOCK_EP_MAX_ORDER_WAR_SZ,
	.max_order_waw_size = SOCK_EP_MAX_ORDER_WAW_SZ,
	.mem_tag_format = SOCK_EP_MEM_TAG_FMT,
	.tx_ctx_cnt = SOCK_EP_MAX_TX_CNT,
	.rx_ctx_cnt = SOCK_EP_MAX_RX_CNT,
};

static const struct fi_tx_attr sock_msg_tx_attr = {
	.caps = SOCK_EP_MSG_CAP_BASE,
	.mode = SOCK_MODE,
	.op_flags = SOCK_EP_DEFAULT_OP_FLAGS,
	.msg_order = SOCK_EP_MSG_ORDER,
	.inject_size = SOCK_EP_MAX_INJECT_SZ,
	.size = SOCK_EP_TX_SZ,
	.iov_limit = SOCK_EP_MAX_IOV_LIMIT,
	.rma_iov_limit = SOCK_EP_MAX_IOV_LIMIT,
};

static const struct fi_rx_attr sock_msg_rx_attr = {
	.caps = SOCK_EP_MSG_CAP_BASE,
	.mode = SOCK_MODE,
	.op_flags = 0,
	.msg_order = SOCK_EP_MSG_ORDER,
	.comp_order = SOCK_EP_COMP_ORDER,
	.total_buffered_recv = SOCK_EP_MAX_BUFF_RECV,
	.size = SOCK_EP_RX_SZ,
	.iov_limit = SOCK_EP_MAX_IOV_LIMIT,
};

static int sock_msg_verify_rx_attr(const struct fi_rx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | SOCK_EP_MSG_CAP) != SOCK_EP_MSG_CAP)
		return -FI_ENODATA;

	if ((attr->msg_order | SOCK_EP_MSG_ORDER) != SOCK_EP_MSG_ORDER)
		return -FI_ENODATA;

	if ((attr->comp_order | SOCK_EP_COMP_ORDER) != SOCK_EP_COMP_ORDER)
		return -FI_ENODATA;

	if (attr->total_buffered_recv > sock_msg_rx_attr.total_buffered_recv)
		return -FI_ENODATA;

	if (sock_get_tx_size(attr->size) >
	     sock_get_tx_size(sock_msg_rx_attr.size))
		return -FI_ENODATA;

	if (attr->iov_limit > sock_msg_rx_attr.iov_limit)
		return -FI_ENODATA;

	return 0;
}

static int sock_msg_verify_tx_attr(const struct fi_tx_attr *attr)
{
	if (!attr)
		return 0;

	if ((attr->caps | SOCK_EP_MSG_CAP) != SOCK_EP_MSG_CAP)
		return -FI_ENODATA;

	if ((attr->msg_order | SOCK_EP_MSG_ORDER) != SOCK_EP_MSG_ORDER)
		return -FI_ENODATA;

	if (attr->inject_size > sock_msg_tx_attr.inject_size)
		return -FI_ENODATA;

	if (sock_get_tx_size(attr->size) >
	     sock_get_tx_size(sock_msg_tx_attr.size))
		return -FI_ENODATA;

	if (attr->iov_limit > sock_msg_tx_attr.iov_limit)
		return -FI_ENODATA;

	if (attr->rma_iov_limit > sock_msg_tx_attr.rma_iov_limit)
		return -FI_ENODATA;

	return 0;
}

int sock_msg_verify_ep_attr(const struct fi_ep_attr *ep_attr,
			    const struct fi_tx_attr *tx_attr,
			    const struct fi_rx_attr *rx_attr)
{
	if (ep_attr) {
		switch (ep_attr->protocol) {
		case FI_PROTO_UNSPEC:
		case FI_PROTO_SOCK_TCP:
			break;
		default:
			return -FI_ENODATA;
		}

		if (ep_attr->protocol_version &&
		    (ep_attr->protocol_version != sock_msg_ep_attr.protocol_version))
			return -FI_ENODATA;

		if (ep_attr->max_msg_size > sock_msg_ep_attr.max_msg_size)
			return -FI_ENODATA;

		if (ep_attr->msg_prefix_size > sock_msg_ep_attr.msg_prefix_size)
			return -FI_ENODATA;

		if (ep_attr->max_order_raw_size >
		   sock_msg_ep_attr.max_order_raw_size)
			return -FI_ENODATA;

		if (ep_attr->max_order_war_size >
		   sock_msg_ep_attr.max_order_war_size)
			return -FI_ENODATA;

		if (ep_attr->max_order_waw_size >
		   sock_msg_ep_attr.max_order_waw_size)
			return -FI_ENODATA;

		if ((ep_attr->tx_ctx_cnt > SOCK_EP_MAX_TX_CNT) &&
		    ep_attr->tx_ctx_cnt != FI_SHARED_CONTEXT)
			return -FI_ENODATA;

		if ((ep_attr->rx_ctx_cnt > SOCK_EP_MAX_RX_CNT) &&
		    ep_attr->rx_ctx_cnt != FI_SHARED_CONTEXT)
			return -FI_ENODATA;

		if (ep_attr->auth_key_size &&
		    (ep_attr->auth_key_size != sock_msg_ep_attr.auth_key_size))
			return -FI_ENODATA;
	}

	if (sock_msg_verify_tx_attr(tx_attr) || sock_msg_verify_rx_attr(rx_attr))
		return -FI_ENODATA;

	return 0;
}

int sock_msg_fi_info(uint32_t version, void *src_addr, void *dest_addr,
		     const struct fi_info *hints, struct fi_info **info)
{
	*info = sock_fi_info(version, FI_EP_MSG, hints, src_addr, dest_addr);
	if (!*info)
		return -FI_ENOMEM;

	*(*info)->tx_attr = sock_msg_tx_attr;
	(*info)->tx_attr->size = sock_get_tx_size(sock_msg_tx_attr.size);
	*(*info)->rx_attr = sock_msg_rx_attr;
	(*info)->rx_attr->size = sock_get_tx_size(sock_msg_rx_attr.size);
	*(*info)->ep_attr = sock_msg_ep_attr;

	if (hints && hints->ep_attr) {
		if (hints->ep_attr->rx_ctx_cnt)
			(*info)->ep_attr->rx_ctx_cnt = hints->ep_attr->rx_ctx_cnt;
		if (hints->ep_attr->tx_ctx_cnt)
			(*info)->ep_attr->tx_ctx_cnt = hints->ep_attr->tx_ctx_cnt;
	}

	if (hints && hints->rx_attr) {
		(*info)->rx_attr->op_flags |= hints->rx_attr->op_flags;
		if (hints->rx_attr->caps)
			(*info)->rx_attr->caps = SOCK_EP_MSG_SEC_CAP |
							hints->rx_attr->caps;
	}

	if (hints && hints->tx_attr) {
		(*info)->tx_attr->op_flags |= hints->tx_attr->op_flags;
		if (hints->tx_attr->caps)
			(*info)->tx_attr->caps = SOCK_EP_MSG_SEC_CAP |
							hints->tx_attr->caps;
	}

	(*info)->caps = SOCK_EP_MSG_CAP |
		(*info)->rx_attr->caps | (*info)->tx_attr->caps;
	if (hints && hints->caps) {
		(*info)->caps = SOCK_EP_MSG_SEC_CAP | hints->caps;
		(*info)->rx_attr->caps = SOCK_EP_MSG_SEC_CAP |
			((*info)->rx_attr->caps & (*info)->caps);
		(*info)->tx_attr->caps = SOCK_EP_MSG_SEC_CAP |
			((*info)->tx_attr->caps & (*info)->caps);
	}
	return 0;
}

static int sock_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct sock_ep *sock_ep = NULL;
	struct sock_pep *sock_pep = NULL;
	size_t len;

	len = MIN(*addrlen, sizeof(struct sockaddr_in));
	switch (fid->fclass) {
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		sock_ep = container_of(fid, struct sock_ep, ep.fid);
		if (sock_ep->attr->is_enabled == 0)
			return -FI_EOPBADSTATE;
		memcpy(addr, sock_ep->attr->src_addr, len);
		break;
	case FI_CLASS_PEP:
		sock_pep = container_of(fid, struct sock_pep, pep.fid);
		if (!sock_pep->name_set)
			return -FI_EOPBADSTATE;
		memcpy(addr, &sock_pep->src_addr, len);
		break;
	default:
		SOCK_LOG_ERROR("Invalid argument\n");
		return -FI_EINVAL;
	}

	*addrlen = sizeof(struct sockaddr_in);
	return (len == sizeof(struct sockaddr_in)) ? 0 : -FI_ETOOSMALL;
}

static int sock_pep_create_listener(struct sock_pep *pep)
{
	int ret;
	socklen_t addr_size;
	struct sockaddr_in addr;
	struct addrinfo *s_res = NULL, *p;
	struct addrinfo hints;
	char sa_ip[INET_ADDRSTRLEN];
	char sa_port[NI_MAXSERV];

	pep->cm.do_listen = 1;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;

	memcpy(sa_ip, inet_ntoa(pep->src_addr.sin_addr), INET_ADDRSTRLEN);
	sprintf(sa_port, "%d", ntohs(pep->src_addr.sin_port));
	sa_ip[INET_ADDRSTRLEN - 1] = '\0';
	sa_port[NI_MAXSERV - 1] = '\0';

	ret = getaddrinfo(sa_ip, sa_port, &hints, &s_res);
	if (ret) {
		SOCK_LOG_ERROR("no available AF_INET address service:%s, %s\n",
			       sa_port, gai_strerror(ret));
		return -FI_EINVAL;
	}

	SOCK_LOG_DBG("binding pep listener to %s\n", sa_port);
	for (p = s_res; p; p = p->ai_next) {
		pep->cm.sock = ofi_socket(p->ai_family, p->ai_socktype,
				     p->ai_protocol);
		if (pep->cm.sock >= 0) {
			sock_set_sockopts(pep->cm.sock, SOCK_OPTS_NONBLOCK);
			if (!bind(pep->cm.sock, s_res->ai_addr, s_res->ai_addrlen))
				break;
			SOCK_LOG_ERROR("failed to bind listener: %s\n",
				       strerror(ofi_sockerr()));
			ofi_close_socket(pep->cm.sock);
			pep->cm.sock = -1;
		}
	}

	freeaddrinfo(s_res);
	if (pep->cm.sock < 0) {
		SOCK_LOG_ERROR("failed to create listener: %s\n",
			       strerror(ofi_sockerr()));
		return -FI_EIO;
	}

	if (pep->src_addr.sin_port == 0) {
		addr_size = sizeof(addr);
		if (getsockname(pep->cm.sock, (struct sockaddr *)&addr, &addr_size))
			return -FI_EINVAL;
		pep->src_addr.sin_port = addr.sin_port;
		sprintf(sa_port, "%d", ntohs(pep->src_addr.sin_port));
	}

	if (pep->src_addr.sin_addr.s_addr == 0) {
		ret = sock_get_src_addr_from_hostname(&pep->src_addr, sa_port);
		if (ret)
			return -FI_EINVAL;
	}

	if (listen(pep->cm.sock, sock_cm_def_map_sz)) {
		SOCK_LOG_ERROR("failed to listen socket: %s\n",
			       strerror(ofi_sockerr()));
		return -ofi_sockerr();
	}

	pep->name_set = 1;
	SOCK_LOG_DBG("Listener thread bound to %s:%d\n",
		     sa_ip, ntohs(pep->src_addr.sin_port));
	return 0;
}

static int sock_ep_cm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct sock_ep *sock_ep = NULL;
	struct sock_pep *sock_pep = NULL;

	if (addrlen != sizeof(struct sockaddr_in))
		return -FI_EINVAL;

	switch (fid->fclass) {
	case FI_CLASS_EP:
	case FI_CLASS_SEP:
		sock_ep = container_of(fid, struct sock_ep, ep.fid);
		if (sock_ep->attr->conn_handle.do_listen)
			return -FI_EINVAL;
		memcpy(sock_ep->attr->src_addr, addr, addrlen);
		return sock_conn_listen(sock_ep->attr);
	case FI_CLASS_PEP:
		sock_pep = container_of(fid, struct sock_pep, pep.fid);
		if (sock_pep->cm.listener_thread)
			return -FI_EINVAL;
		memcpy(&sock_pep->src_addr, addr, addrlen);
		return sock_pep_create_listener(sock_pep);
	default:
		SOCK_LOG_ERROR("Invalid argument\n");
		return -FI_EINVAL;
	}
}

static int sock_ep_cm_getpeer(struct fid_ep *ep, void *addr, size_t *addrlen)
{
	struct sock_ep *sock_ep;
	size_t len;

	sock_ep = container_of(ep, struct sock_ep, ep);
	len = MIN(*addrlen, sizeof(struct sockaddr_in));
	memcpy(addr, sock_ep->attr->dest_addr, len);
	*addrlen = sizeof(struct sockaddr_in);
	return (len == sizeof(struct sockaddr_in)) ? 0 : -FI_ETOOSMALL;
}

static int sock_cm_send(int fd, const void *buf, int len)
{
	int ret, done = 0;

	while (done != len) {
		ret = ofi_send_socket(fd, (const char*) buf + done,
				      len - done, MSG_NOSIGNAL);
		if (ret < 0) {
			if (OFI_SOCK_TRY_SND_RCV_AGAIN(ofi_sockerr()))
				continue;
			SOCK_LOG_ERROR("failed to write to fd: %s\n",
				       strerror(ofi_sockerr()));
			return -FI_EIO;
		}
		done += ret;
	}
	return 0;
}

static int sock_cm_recv(int fd, void *buf, int len)
{
	int ret, done = 0;
	while (done != len) {
		ret = ofi_recv_socket(fd, (char*) buf + done, len - done, 0);
		if (ret <= 0) {
			if (ret < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
				continue;
			SOCK_LOG_ERROR("failed to read from fd: %s\n",
				       strerror(ofi_sockerr()));
			return -FI_EIO;
		}
		done += ret;
	}
	return 0;
}

static void sock_ep_wait_shutdown(struct sock_ep *ep)
{
	int ret, do_report = 0;
	char tmp = 0;
	struct pollfd poll_fds[2];
	struct sock_conn_hdr msg;
	struct fi_eq_cm_entry cm_entry = {0};

	poll_fds[0].fd = ep->attr->cm.sock;
	poll_fds[1].fd = ep->attr->cm.signal_fds[1];
	poll_fds[0].events = poll_fds[1].events = POLLIN;

	while (*((volatile int*) &ep->attr->cm.do_listen)) {
		ret = poll(poll_fds, 2, -1);
		if (ret > 0) {
			if (poll_fds[1].revents & POLLIN) {
				ret = ofi_read_socket(ep->attr->cm.signal_fds[1], &tmp, 1);
				if (ret != 1) {
					SOCK_LOG_DBG("Invalid signal\n");
					break;
				}
				continue;
			}
		} else {
			break;
		}

		if (sock_cm_recv(ep->attr->cm.sock, &msg, sizeof(msg)))
			break;

		if (msg.type == SOCK_CONN_SHUTDOWN)
			break;
	}

	fastlock_acquire(&ep->attr->cm.lock);
	if (ep->attr->cm.is_connected) {
		do_report = 1;
		ep->attr->cm.is_connected = 0;
	}
	fastlock_release(&ep->attr->cm.lock);

	if (do_report) {
		cm_entry.fid = &ep->ep.fid;
		SOCK_LOG_DBG("reporting FI_SHUTDOWN\n");
		if (sock_eq_report_event(ep->attr->eq, FI_SHUTDOWN,
					 &cm_entry, sizeof(cm_entry), 0))
			SOCK_LOG_ERROR("Error in writing to EQ\n");
	}
	ofi_close_socket(ep->attr->cm.sock);
}

static void sock_ep_cm_report_connect_fail(struct sock_ep *ep,
					   void *param, size_t paramlen)
{
	SOCK_LOG_DBG("reporting FI_REJECT\n");
	if (sock_eq_report_error(ep->attr->eq, &ep->ep.fid, NULL, 0,
				 FI_ECONNREFUSED, -FI_ECONNREFUSED,
				 param, paramlen))
		SOCK_LOG_ERROR("Error in writing to EQ\n");
}

static void *sock_ep_cm_connect_handler(void *data)
{
	int sock_fd, ret;
	struct sock_conn_req_handle *handle = data;
	struct sock_conn_req *req = handle->req;
	struct sock_conn_hdr response;
	struct sock_ep *ep = handle->ep;
	void *param = NULL;
	struct fi_eq_cm_entry *cm_entry = NULL;
	int cm_data_sz, response_port;

	sock_fd = ofi_socket(AF_INET, SOCK_STREAM, 0);
	if (sock_fd < 0) {
		SOCK_LOG_ERROR("no socket\n");
		sock_ep_cm_report_connect_fail(handle->ep, NULL, 0);
		goto out;
	}

	ofi_straddr_dbg(&sock_prov, FI_LOG_EP_CTRL, "Connecting to address",
			&handle->dest_addr);
	sock_set_sockopts(sock_fd, SOCK_OPTS_KEEPALIVE);
	ret = connect(sock_fd, (struct sockaddr *)&handle->dest_addr,
		      sizeof(handle->dest_addr));
	if (ret < 0) {
		SOCK_LOG_ERROR("connect failed : %s\n",
			       strerror(ofi_sockerr()));
		goto err;
	}

	if (sock_cm_send(sock_fd, req, sizeof(*req)))
		goto err;
	if (handle->paramlen && sock_cm_send(sock_fd, handle->cm_data, handle->paramlen))
		goto err;

	if (sock_cm_recv(sock_fd, &response, sizeof(response)))
		goto err;

	cm_data_sz = ntohs(response.cm_data_sz);
	response_port = ntohs(response.port);
	if (cm_data_sz) {
		param = calloc(1, cm_data_sz);
		if (!param)
			goto err;

		if (sock_cm_recv(sock_fd, param, cm_data_sz))
			goto err;
	}

	if (response.type == SOCK_CONN_REJECT) {
		sock_ep_cm_report_connect_fail(handle->ep, param, cm_data_sz);
		ofi_close_socket(sock_fd);
	} else {
		cm_entry = calloc(1, sizeof(*cm_entry) + SOCK_EP_MAX_CM_DATA_SZ);
		if (!cm_entry)
			goto err;

		cm_entry->fid = &ep->ep.fid;
		memcpy(&cm_entry->data, param, cm_data_sz);
		ep->attr->cm.is_connected = 1;
		ep->attr->cm.do_listen = 1;
		ep->attr->cm.sock = sock_fd;
		ep->attr->msg_dest_port = response_port;
		SOCK_LOG_DBG("got accept - port: %d\n", response_port);

		SOCK_LOG_DBG("Reporting FI_CONNECTED\n");
		if (sock_eq_report_event(ep->attr->eq, FI_CONNECTED, cm_entry,
					 sizeof(*cm_entry) + cm_data_sz, 0))
			SOCK_LOG_ERROR("Error in writing to EQ\n");
		sock_ep_wait_shutdown(ep);
	}
	goto out;
err:
	SOCK_LOG_ERROR("io failed : %s\n", strerror(ofi_sockerr()));
	sock_ep_cm_report_connect_fail(handle->ep, NULL, 0);
	ofi_close_socket(sock_fd);
out:
	free(param);
	free(cm_entry);
	free(handle->req);
	free(handle);
	return NULL;
}

static int sock_ep_cm_connect(struct fid_ep *ep, const void *addr,
			      const void *param, size_t paramlen)
{
	struct sock_conn_req *req = NULL;
	struct sock_conn_req_handle *handle = NULL;
	struct sock_ep *_ep;
	struct sock_eq *_eq;

	_ep = container_of(ep, struct sock_ep, ep);
	_eq = _ep->attr->eq;
	if (!_eq || !addr || (paramlen > SOCK_EP_MAX_CM_DATA_SZ))
		return -FI_EINVAL;

	if (!_ep->attr->conn_handle.do_listen && sock_conn_listen(_ep->attr))
		return -FI_EINVAL;

	if (!_ep->attr->dest_addr) {
		_ep->attr->dest_addr = calloc(1, sizeof(*_ep->attr->dest_addr));
		if (!_ep->attr->dest_addr)
			return -FI_ENOMEM;
	}
	memcpy(_ep->attr->dest_addr, addr, sizeof(*_ep->attr->dest_addr));

	req = calloc(1, sizeof(*req));
	if (!req)
		return -FI_ENOMEM;

	handle = calloc(1, sizeof(*handle));
	if (!handle)
		goto out;

	req->hdr.type = SOCK_CONN_REQ;
	req->hdr.port = htons(_ep->attr->msg_src_port);
	req->hdr.cm_data_sz = htons(paramlen);
	req->caps = _ep->attr->info.caps;
	memcpy(&req->src_addr, _ep->attr->src_addr, sizeof(req->src_addr));
	memcpy(&handle->dest_addr, addr, sizeof(handle->dest_addr));

	handle->ep = _ep;
	handle->req = req;
	if (paramlen) {
		handle->paramlen = paramlen;
		memcpy(handle->cm_data, param, paramlen);
	}

	if (_ep->attr->cm.listener_thread &&
	    pthread_join(_ep->attr->cm.listener_thread, NULL))
		SOCK_LOG_DBG("failed to join cm listener\n");

	if (pthread_create(&_ep->attr->cm.listener_thread, NULL,
			   sock_ep_cm_connect_handler, handle)) {
		SOCK_LOG_ERROR("failed to create cm thread\n");
		goto out;
	}
	return 0;
out:
	free(req);
	free(handle);
	return -FI_ENOMEM;
}

static void *sock_cm_accept_handler(void *data)
{
	int ret;
	struct sock_conn_hdr reply;
	struct sock_conn_req_handle *hreq = data;
	struct sock_ep_attr *ep_attr;
	struct fi_eq_cm_entry cm_entry;

	ep_attr = hreq->ep->attr;
	ep_attr->msg_dest_port = ntohs(hreq->req->hdr.port);

	reply.type = SOCK_CONN_ACCEPT;
	reply.port = htons(ep_attr->msg_src_port);
	reply.cm_data_sz = htons(hreq->paramlen);
	ret = sock_cm_send(hreq->sock_fd, &reply, sizeof(reply));
	if (ret) {
		SOCK_LOG_ERROR("failed to reply\n");
		return NULL;
	}

	if (hreq->paramlen && sock_cm_send(hreq->sock_fd, hreq->cm_data, hreq->paramlen)) {
		SOCK_LOG_ERROR("failed to send userdata\n");
		return NULL;
	}

	cm_entry.fid = &hreq->ep->ep.fid;
	SOCK_LOG_DBG("reporting FI_CONNECTED\n");
	if (sock_eq_report_event(ep_attr->eq, FI_CONNECTED, &cm_entry,
				 sizeof(cm_entry), 0))
		SOCK_LOG_ERROR("Error in writing to EQ\n");
	ep_attr->cm.is_connected = 1;
	ep_attr->cm.do_listen = 1;
	ep_attr->cm.sock = hreq->sock_fd;
	sock_ep_wait_shutdown(hreq->ep);

	if (pthread_join(hreq->req_handler, NULL))
		SOCK_LOG_DBG("failed to join req-handler\n");
	free(hreq->req);
	free(hreq);
	return NULL;
}

static int sock_ep_cm_accept(struct fid_ep *ep, const void *param, size_t paramlen)
{
	struct sock_conn_req_handle *handle;
	struct sock_ep *_ep;

	_ep = container_of(ep, struct sock_ep, ep);
	if (!_ep->attr->eq || paramlen > SOCK_EP_MAX_CM_DATA_SZ)
		return -FI_EINVAL;

	if (!_ep->attr->conn_handle.do_listen && sock_conn_listen(_ep->attr))
		return -FI_EINVAL;

	handle = container_of(_ep->attr->info.handle,
			      struct sock_conn_req_handle, handle);
	if (!handle || handle->handle.fclass != FI_CLASS_CONNREQ) {
		SOCK_LOG_ERROR("invalid handle for cm_accept\n");
		return -FI_EINVAL;
	}

	handle->ep = _ep;
	handle->paramlen = 0;
	handle->is_accepted = 1;
	if (paramlen) {
		handle->paramlen = paramlen;
		memcpy(handle->cm_data, param, paramlen);
	}

	if (_ep->attr->cm.listener_thread &&
	    pthread_join(_ep->attr->cm.listener_thread, NULL))
		SOCK_LOG_DBG("failed to join cm listener\n");

	if (pthread_create(&_ep->attr->cm.listener_thread, NULL,
			   sock_cm_accept_handler, handle)) {
		SOCK_LOG_ERROR("Couldnt create accept handler\n");
		return -FI_ENOMEM;
	}
	return 0;
}

static int sock_ep_cm_shutdown(struct fid_ep *ep, uint64_t flags)
{
	struct sock_ep *_ep;
	struct fi_eq_cm_entry cm_entry = {0};
	struct sock_conn_hdr msg = {0};
	char c = 0;

	_ep = container_of(ep, struct sock_ep, ep);
	fastlock_acquire(&_ep->attr->cm.lock);
	if (_ep->attr->cm.is_connected) {
		msg.type = SOCK_CONN_SHUTDOWN;
		if (sock_cm_send(_ep->attr->cm.sock, &msg, sizeof(msg)))
			SOCK_LOG_DBG("failed to send shutdown msg\n");
		_ep->attr->cm.is_connected = 0;
		_ep->attr->cm.do_listen = 0;
		if (ofi_write_socket(_ep->attr->cm.signal_fds[0], &c, 1) != 1)
			SOCK_LOG_DBG("Failed to signal\n");

		cm_entry.fid = &_ep->ep.fid;
		SOCK_LOG_DBG("reporting FI_SHUTDOWN\n");
		if (sock_eq_report_event(_ep->attr->eq, FI_SHUTDOWN,
					 &cm_entry, sizeof(cm_entry), 0))
			SOCK_LOG_ERROR("Error in writing to EQ\n");
	}
	fastlock_release(&_ep->attr->cm.lock);
	sock_ep_disable(ep);
	return 0;
}

struct fi_ops_cm sock_ep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = sock_ep_cm_setname,
	.getname = sock_ep_cm_getname,
	.getpeer = sock_ep_cm_getpeer,
	.connect = sock_ep_cm_connect,
	.listen = fi_no_listen,
	.accept = sock_ep_cm_accept,
	.reject = fi_no_reject,
	.shutdown = sock_ep_cm_shutdown,
	.join = fi_no_join,
};

static int sock_msg_endpoint(struct fid_domain *domain, struct fi_info *info,
		struct sock_ep **ep, void *context, size_t fclass)
{
	int ret;
	struct sock_pep *pep;

	if (info) {
		if (info->ep_attr) {
			ret = sock_msg_verify_ep_attr(info->ep_attr,
						      info->tx_attr,
						      info->rx_attr);
			if (ret)
				return -FI_EINVAL;
		}

		if (info->tx_attr) {
			ret = sock_msg_verify_tx_attr(info->tx_attr);
			if (ret)
				return -FI_EINVAL;
		}

		if (info->rx_attr) {
			ret = sock_msg_verify_rx_attr(info->rx_attr);
			if (ret)
				return -FI_EINVAL;
		}
	}

	ret = sock_alloc_endpoint(domain, info, ep, context, fclass);
	if (ret)
		return ret;

	if (info && info->handle && info->handle->fclass == FI_CLASS_PEP) {
		pep = container_of(info->handle, struct sock_pep, pep.fid);
		memcpy((*ep)->attr->src_addr, &pep->src_addr, sizeof *(*ep)->attr->src_addr);
	}

	if (!info || !info->ep_attr)
		(*ep)->attr->ep_attr = sock_msg_ep_attr;

	if (!info || !info->tx_attr)
		(*ep)->tx_attr = sock_msg_tx_attr;

	if (!info || !info->rx_attr)
		(*ep)->rx_attr = sock_msg_rx_attr;

	return 0;
}

int sock_msg_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context)
{
	int ret;
	struct sock_ep *endpoint;

	ret = sock_msg_endpoint(domain, info, &endpoint, context, FI_CLASS_EP);
	if (ret)
		return ret;

	*ep = &endpoint->ep;
	return 0;
}

static int sock_pep_fi_bind(fid_t fid, struct fid *bfid, uint64_t flags)
{
	struct sock_pep *pep;
	struct sock_eq *eq;

	pep = container_of(fid, struct sock_pep, pep.fid);

	if (bfid->fclass != FI_CLASS_EQ)
		return -FI_EINVAL;

	eq = container_of(bfid, struct sock_eq, eq.fid);
	if (pep->sock_fab != eq->sock_fab) {
		SOCK_LOG_ERROR("Cannot bind Passive EP and EQ on different fabric\n");
		return -FI_EINVAL;
	}
	pep->eq = eq;
	return 0;
}

static int sock_pep_fi_close(fid_t fid)
{
	int ret;
	char c = 0;
	struct sock_pep *pep;

	pep = container_of(fid, struct sock_pep, pep.fid);
	pep->cm.do_listen = 0;
	ret = ofi_write_socket(pep->cm.signal_fds[0], &c, 1);
	if (ret != 1)
		SOCK_LOG_DBG("Failed to signal\n");

	if (pep->cm.listener_thread &&
	    pthread_join(pep->cm.listener_thread, NULL)) {
		SOCK_LOG_DBG("pthread join failed\n");
	}

	ofi_close_socket(pep->cm.signal_fds[0]);
	ofi_close_socket(pep->cm.signal_fds[1]);
	fastlock_destroy(&pep->cm.lock);

	free(pep);
	return 0;
}

static struct fi_ops sock_pep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = sock_pep_fi_close,
	.bind = sock_pep_fi_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_info *sock_ep_msg_get_info(struct sock_pep *pep,
					    struct sock_conn_req *req)
{
	struct fi_info hints;
	uint64_t requested, supported;

	requested = req->caps & SOCK_EP_MSG_PRI_CAP;
	supported = pep->info.caps & SOCK_EP_MSG_PRI_CAP;
	supported = (supported & FI_RMA) ?
		(supported | FI_REMOTE_READ | FI_REMOTE_WRITE) : supported;
	if ((requested | supported) != supported)
		return NULL;

	hints = pep->info;
	hints.caps = req->caps;
	return sock_fi_info(pep->sock_fab->fab_fid.api_version, FI_EP_MSG,
			    &hints, &pep->src_addr, &req->src_addr);
}

static void *sock_pep_req_handler(void *data)
{
	int ret, entry_sz;
	struct fi_info *info;
	struct sock_conn_req *conn_req = NULL;
	struct fi_eq_cm_entry *cm_entry = NULL;
	struct sock_conn_req_handle *handle = data;
	int req_cm_data_sz;
	char c = 0;

	conn_req = calloc(1, sizeof(*conn_req) + SOCK_EP_MAX_CM_DATA_SZ);
	if (!conn_req) {
		SOCK_LOG_ERROR("cannot allocate memory\n");
		goto err;
	}

	ret = sock_cm_recv(handle->sock_fd, conn_req, sizeof(*conn_req));
	if (ret) {
		SOCK_LOG_ERROR("IO failed\n");
		goto err;
	}

	req_cm_data_sz = ntohs(conn_req->hdr.cm_data_sz);
	if (req_cm_data_sz) {
		ret = sock_cm_recv(handle->sock_fd, conn_req->cm_data,
				   req_cm_data_sz);
		if (ret) {
			SOCK_LOG_ERROR("IO failed for cm-data\n");
			goto err;
		}
	}

	info = sock_ep_msg_get_info(handle->pep, conn_req);
	if (info == NULL) {
		handle->paramlen = 0;
		fastlock_acquire(&handle->pep->cm.lock);
		dlist_insert_tail(&handle->entry, &handle->pep->cm.msg_list);
		fastlock_release(&handle->pep->cm.lock);

		if (ofi_write_socket(handle->pep->cm.signal_fds[0], &c, 1) != 1)
			SOCK_LOG_DBG("Failed to signal\n");
		free(conn_req);
		return NULL;
	}

	cm_entry = calloc(1, sizeof(*cm_entry) + req_cm_data_sz);
	if (!cm_entry) {
		SOCK_LOG_ERROR("cannot allocate memory\n");
		goto err;
	}

	handle->handle.fclass = FI_CLASS_CONNREQ;
	handle->req = conn_req;

	entry_sz = sizeof(*cm_entry) + req_cm_data_sz;
	cm_entry->fid = &handle->pep->pep.fid;
	cm_entry->info = info;
	cm_entry->info->handle = &handle->handle;
	memcpy(cm_entry->data, conn_req->cm_data, req_cm_data_sz);

	SOCK_LOG_DBG("reporting conn-req to EQ\n");
	if (sock_eq_report_event(handle->pep->eq, FI_CONNREQ, cm_entry, entry_sz, 0))
		SOCK_LOG_ERROR("Error in writing to EQ\n");

	free(cm_entry);
	return NULL;
err:
	ofi_close_socket(handle->sock_fd);
	free(cm_entry);
	free(conn_req);
	free(handle);
	return NULL;
}

static void sock_pep_check_msg_list(struct sock_pep *pep)
{
	struct dlist_entry *entry;
	struct sock_conn_req_handle *hreq;
	struct sock_conn_hdr reply;

	fastlock_acquire(&pep->cm.lock);
	while (!dlist_empty(&pep->cm.msg_list)) {
		entry = pep->cm.msg_list.next;
		dlist_remove(entry);
		hreq = container_of(entry, struct sock_conn_req_handle, entry);

		reply.type = SOCK_CONN_REJECT;
		reply.cm_data_sz = htons(hreq->paramlen);

		SOCK_LOG_DBG("sending reject message\n");
		if (sock_cm_send(hreq->sock_fd, &reply, sizeof(reply))) {
			SOCK_LOG_ERROR("failed to reply\n");
			break;
		}

		if (hreq->paramlen && sock_cm_send(hreq->sock_fd, hreq->cm_data,
						   hreq->paramlen)) {
			SOCK_LOG_ERROR("failed to send userdata\n");
			break;
		}

		if (pthread_join(hreq->req_handler, NULL))
			SOCK_LOG_DBG("failed to join req-handler\n");
		ofi_close_socket(hreq->sock_fd);
		free(hreq->req);
		free(hreq);
	}
	fastlock_release(&pep->cm.lock);
}

static void *sock_pep_listener_thread(void *data)
{
	struct sock_pep *pep = (struct sock_pep *) data;
	struct sock_conn_req_handle *handle = NULL;
	struct pollfd poll_fds[2];

	int ret = 0, conn_fd;
	char tmp = 0;

	SOCK_LOG_DBG("Starting listener thread for PEP: %p\n", pep);
	poll_fds[0].fd = pep->cm.sock;
	poll_fds[1].fd = pep->cm.signal_fds[1];
	poll_fds[0].events = poll_fds[1].events = POLLIN;
	while (*((volatile int *) &pep->cm.do_listen)) {
		ret = poll(poll_fds, 2, -1);
		if (ret > 0) {
			if (poll_fds[1].revents & POLLIN) {
				ret = ofi_read_socket(pep->cm.signal_fds[1], &tmp, 1);
				if (ret != 1)
					SOCK_LOG_DBG("Invalid signal\n");
				sock_pep_check_msg_list(pep);
				continue;
			}
		} else {
			break;
		}

		conn_fd = accept(pep->cm.sock, NULL, 0);
		if (conn_fd < 0) {
			SOCK_LOG_ERROR("failed to accept: %d\n", ofi_sockerr());
			continue;
		}

		sock_set_sockopts(conn_fd, SOCK_OPTS_KEEPALIVE);
		handle = calloc(1, sizeof(*handle));
		if (!handle) {
			SOCK_LOG_ERROR("cannot allocate memory\n");
			ofi_close_socket(conn_fd);
			break;
		}

		handle->sock_fd = conn_fd;
		handle->pep = pep;

		if (pthread_create(&handle->req_handler, NULL,
				   sock_pep_req_handler, handle)) {
			SOCK_LOG_ERROR("failed to create req handler\n");
			ofi_close_socket(conn_fd);
			free(handle);
		}
	}

	SOCK_LOG_DBG("PEP listener thread exiting\n");
	ofi_close_socket(pep->cm.sock);
	return NULL;
}

static int sock_pep_start_listener_thread(struct sock_pep *pep)
{
	if (pthread_create(&pep->cm.listener_thread, NULL,
			   sock_pep_listener_thread, (void *)pep)) {
		SOCK_LOG_ERROR("Couldn't create listener thread\n");
		return -FI_EINVAL;
	}
	return 0;
}

static int sock_pep_listen(struct fid_pep *pep)
{
	struct sock_pep *_pep;
	_pep = container_of(pep, struct sock_pep, pep);
	if (_pep->cm.listener_thread)
		return 0;

	if (!_pep->cm.do_listen && sock_pep_create_listener(_pep)) {
		SOCK_LOG_ERROR("Failed to create pep thread\n");
		return -FI_EINVAL;
	}

	return sock_pep_start_listener_thread(_pep);
}

static int sock_pep_reject(struct fid_pep *pep, fid_t handle,
		const void *param, size_t paramlen)
{
	struct sock_conn_req_handle *hreq;
	struct sock_conn_req *req;
	struct sock_pep *_pep;
	char c = 0;

	_pep = container_of(pep, struct sock_pep, pep);
	hreq = container_of(handle, struct sock_conn_req_handle, handle);
	req = hreq->req;
	if (!req || hreq->handle.fclass != FI_CLASS_CONNREQ || hreq->is_accepted)
		return -FI_EINVAL;

	hreq->paramlen = 0;
	if (paramlen) {
		memcpy(hreq->cm_data, param, paramlen);
		hreq->paramlen = paramlen;
	}

	fastlock_acquire(&_pep->cm.lock);
	dlist_insert_tail(&hreq->entry, &_pep->cm.msg_list);
	fastlock_release(&_pep->cm.lock);

	if (ofi_write_socket(_pep->cm.signal_fds[0], &c, 1) != 1)
		SOCK_LOG_DBG("Failed to signal\n");
	return 0;
}

static struct fi_ops_cm sock_pep_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = sock_ep_cm_setname,
	.getname = sock_ep_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = sock_pep_listen,
	.accept = fi_no_accept,
	.reject = sock_pep_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};


int sock_pep_getopt(fid_t fid, int level, int optname,
		      void *optval, size_t *optlen)
{
	if (level != FI_OPT_ENDPOINT || optname != FI_OPT_CM_DATA_SIZE)
		return -FI_ENOPROTOOPT;

	if (*optlen < sizeof(size_t)) {
		*optlen = sizeof(size_t);
		return -FI_ETOOSMALL;
	}
	*((size_t *) optval) = SOCK_EP_MAX_CM_DATA_SZ;
	*optlen = sizeof(size_t);
	return 0;
}

static struct fi_ops_ep sock_pep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.getopt = sock_pep_getopt,
	.setopt = fi_no_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

int sock_msg_sep(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **sep, void *context)
{
	int ret;
	struct sock_ep *endpoint;

	ret = sock_msg_endpoint(domain, info, &endpoint, context, FI_CLASS_SEP);
	if (ret)
		return ret;

	*sep = &endpoint->ep;
	return 0;
}

int sock_msg_passive_ep(struct fid_fabric *fabric, struct fi_info *info,
			struct fid_pep **pep, void *context)
{
	int ret = 0;
	struct sock_pep *_pep;
	struct addrinfo hints, *result;

	if (info) {
		ret = sock_verify_info(fabric->api_version, info);
		if (ret) {
			SOCK_LOG_DBG("Cannot support requested options!\n");
			return ret;
		}
	}

	_pep = calloc(1, sizeof(*_pep));
	if (!_pep)
		return -FI_ENOMEM;

	if (info) {
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
				SOCK_LOG_DBG("getaddrinfo failed!\n");
				goto err;
			}
			memcpy(&_pep->src_addr, result->ai_addr,
				result->ai_addrlen);
			freeaddrinfo(result);
		}
		_pep->info = *info;
	} else {
		SOCK_LOG_ERROR("invalid fi_info\n");
		goto err;
	}

	ret = socketpair(AF_UNIX, SOCK_STREAM, 0, _pep->cm.signal_fds);
	if (ret) {
		ret = -ofi_sockerr();
		goto err;
	}

	fd_set_nonblock(_pep->cm.signal_fds[1]);
	dlist_init(&_pep->cm.msg_list);

	_pep->pep.fid.fclass = FI_CLASS_PEP;
	_pep->pep.fid.context = context;
	_pep->pep.fid.ops = &sock_pep_fi_ops;
	_pep->pep.cm = &sock_pep_cm_ops;
	_pep->pep.ops = &sock_pep_ops;
	fastlock_init(&_pep->cm.lock);

	_pep->sock_fab = container_of(fabric, struct sock_fabric, fab_fid);
	*pep = &_pep->pep;
	return 0;
err:
	free(_pep);
	return ret;
}

