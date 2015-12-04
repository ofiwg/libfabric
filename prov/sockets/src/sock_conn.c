/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
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

#include <stdlib.h>
#include <stdio.h>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <poll.h>
#include <limits.h>

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_CTRL, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_CTRL, __VA_ARGS__)

ssize_t sock_conn_send_src_addr(struct sock_ep *sock_ep, struct sock_conn *conn)
{
	int ret;
	uint64_t total_len;
	struct sock_op tx_op;
	struct sock_tx_ctx *tx_ctx;

	if (sock_ep->ep.fid.fclass == FI_CLASS_SEP)
		tx_ctx = sock_ep->tx_array[0];
	else
		tx_ctx = sock_ep->tx_ctx;

	memset(&tx_op, 0, sizeof(struct sock_op));
	tx_op.op = SOCK_OP_CONN_MSG;
	SOCK_LOG_DBG("New conn msg on TX: %p using conn: %p\n", tx_ctx, conn);

	total_len = 0;
	tx_op.src_iov_len = sizeof(struct sockaddr_in);
	total_len = tx_op.src_iov_len + sizeof(struct sock_op_send);

	sock_tx_ctx_start(tx_ctx);
	if (rbavail(&tx_ctx->rb) < total_len) {
		ret = -FI_EAGAIN;
		goto err;
	}

	sock_tx_ctx_write_op_send(tx_ctx, &tx_op, 0, (uintptr_t) NULL, 0, 0, sock_ep,
					conn);
	sock_tx_ctx_write(tx_ctx, sock_ep->src_addr, sizeof(struct sockaddr_in));
	sock_tx_ctx_commit(tx_ctx);
	conn->address_published = 1;
	return 0;

err:
	sock_tx_ctx_abort(tx_ctx);
	return ret;
}

int sock_conn_map_init(struct sock_ep *ep, int init_size)
{
	struct sock_conn_map *map = &ep->cmap;
	map->table = calloc(init_size, sizeof(*map->table));
	if (!map->table)
		return -FI_ENOMEM;

	if (sock_epoll_create(&map->epoll_set, init_size) < 0) {
                SOCK_LOG_ERROR("failed to create epoll set\n");
                free(map->table);
                return -FI_ENOMEM;
        }

	fastlock_init(&map->lock);
	map->used = 0;
	map->size = init_size;
	return 0;
}

static int sock_conn_map_increase(struct sock_conn_map *map, int new_size)
{
	void *_table;

	_table = realloc(map->table, new_size * sizeof(*map->table));
	if (!_table) {
		SOCK_LOG_ERROR("*** realloc failed, use FI_SOCKETS_DEF_CONN_MAP_SZ for"
			"specifying conn-map-size\n");
		return -FI_ENOMEM;
	}

	map->size = new_size;
	map->table = _table;
	return 0;
}

void sock_conn_map_destroy(struct sock_conn_map *cmap)
{
	int i;

	for (i = 0; i < cmap->used; i++) {
		close(cmap->table[i].sock_fd);
	}
	free(cmap->table);
	cmap->table = NULL;
	cmap->used = cmap->size = 0;
	sock_epoll_close(&cmap->epoll_set);
	fastlock_destroy(&cmap->lock);
}

static struct sock_conn *sock_conn_map_insert(struct sock_ep *ep,
				struct sockaddr_in *addr, int conn_fd,
				int addr_published)
{
	int index;
	struct sock_conn_map *map = &ep->cmap;

	if (map->size == map->used) {
		if (sock_conn_map_increase(map, map->size * 2))
			return 0;
	}

	index = map->used;
	map->used++;

	map->table[index].addr = *addr;
	map->table[index].sock_fd = conn_fd;
	map->table[index].ep = ep;
	sock_set_sockopts(conn_fd);

	fastlock_acquire(&ep->lock);
	dlist_insert_tail(&map->table[index].ep_entry, &ep->conn_list);
	fastlock_release(&ep->lock);

	if (idm_set(&ep->conn_idm, conn_fd, &map->table[index]) < 0)
                SOCK_LOG_ERROR("idm_set failed\n");

	if (sock_epoll_add(&map->epoll_set, conn_fd))
                SOCK_LOG_ERROR("failed to add to epoll set: %d\n", conn_fd);

	map->table[index].address_published = addr_published;
	sock_pe_poll_add(ep->domain->pe, conn_fd);
	return &map->table[index];
}

int fd_set_nonblock(int fd)
{
	int flags, ret;

	flags = fcntl(fd, F_GETFL, 0);
	ret = fcntl(fd, F_SETFL, flags | O_NONBLOCK);
	if (ret) {
		SOCK_LOG_ERROR("fcntl failed\n");
		ret = -errno;
	}

	return ret;
}

static void sock_set_sockopt_reuseaddr(int sock)
{
	int optval;
	optval = 1;
	if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)))
		SOCK_LOG_ERROR("setsockopt reuseaddr failed\n");
}

void sock_set_sockopts_conn(int sock)
{
	int optval;
	optval = 1;
	sock_set_sockopt_reuseaddr(sock);
	if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)))
		SOCK_LOG_ERROR("setsockopt nodelay failed\n");
}

void sock_set_sockopts(int sock)
{
	int optval;
	optval = 1;

	sock_set_sockopt_reuseaddr(sock);
	if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval)))
		SOCK_LOG_ERROR("setsockopt nodelay failed\n");

	fd_set_nonblock(sock);
}

static void *_sock_conn_listen(void *arg)
{
	int conn_fd, ret;
	char tmp;
	socklen_t addr_size;
	struct sockaddr_in remote;
	struct pollfd poll_fds[2];

	struct sock_ep *ep = (struct sock_ep *)arg;
	struct sock_conn_listener *listener = &ep->listener;
	struct sock_conn_map *map = &ep->cmap;

	poll_fds[0].fd = listener->sock;
	poll_fds[1].fd = listener->signal_fds[1];
	poll_fds[0].events = poll_fds[1].events = POLLIN;
	listener->is_ready = 1;

	while (listener->do_listen) {
		if (poll(poll_fds, 2, -1) > 0) {
			if (poll_fds[1].revents & POLLIN) {
				ret = read(listener->signal_fds[1], &tmp, 1);
				if (ret != 1) {
					SOCK_LOG_ERROR("Invalid signal\n");
					goto err;
				}
				continue;
			}
		} else {
			goto err;
		}

		addr_size = sizeof(remote);
		conn_fd = accept(listener->sock, (struct sockaddr *) &remote,
					&addr_size);
		SOCK_LOG_DBG("CONN: accepted conn-req: %d\n", conn_fd);
		if (conn_fd < 0) {
			SOCK_LOG_ERROR("failed to accept: %d\n", errno);
			goto err;
		}

		SOCK_LOG_DBG("ACCEPT: %s, %d\n", inet_ntoa(remote.sin_addr),
				ntohs(remote.sin_port));

		fastlock_acquire(&map->lock);
		sock_conn_map_insert(ep, &remote, conn_fd, 1);
		fastlock_release(&map->lock);
		sock_pe_signal(ep->domain->pe);
	}

err:
	close(listener->sock);
	SOCK_LOG_DBG("Listener thread exited\n");
	return NULL;
}

int sock_conn_listen(struct sock_ep *ep)
{
	struct addrinfo *s_res = NULL, *p;
	struct addrinfo hints;
	int listen_fd = 0, ret;
	socklen_t addr_size;
	struct sockaddr_in addr;
	struct sock_conn_listener *listener = &ep->listener;
	struct sock_domain *domain = ep->domain;
	char service[NI_MAXSERV] = {0};

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_flags = AI_PASSIVE;

	if (getnameinfo((void *)ep->src_addr, sizeof(*ep->src_addr),
			NULL, 0, listener->service,
			sizeof(listener->service), NI_NUMERICSERV)) {
		SOCK_LOG_ERROR("could not resolve src_addr\n");
		return -FI_EINVAL;
	}

	if (!sock_fabric_check_service(domain->fab, atoi(listener->service))) {
		memset(listener->service, 0, NI_MAXSERV);
		((struct sockaddr_in *)ep->src_addr)->sin_port = 0;
	}

	ret = getaddrinfo(inet_ntoa(((struct sockaddr_in *)ep->src_addr)->sin_addr),
			  listener->service, &hints, &s_res);
	if (ret) {
		SOCK_LOG_ERROR("no available AF_INET address, service %s, %s\n",
			       listener->service, gai_strerror(ret));
		return -FI_EINVAL;
	}

	SOCK_LOG_DBG("Binding listener thread to port: %s\n", listener->service);
	for (p = s_res; p; p = p->ai_next) {
		listen_fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
		if (listen_fd >= 0) {
			sock_set_sockopts(listen_fd);

			if (!bind(listen_fd, s_res->ai_addr, s_res->ai_addrlen))
				break;
			close(listen_fd);
			listen_fd = -1;
		}
	}
	freeaddrinfo(s_res);

	if (listen_fd < 0) {
		SOCK_LOG_ERROR("failed to listen to port: %s\n",
			       listener->service);
		goto err;
	}

	if (atoi(listener->service) == 0) {
		addr_size = sizeof(addr);
		if (getsockname(listen_fd, (struct sockaddr *) &addr, &addr_size))
			goto err;
		snprintf(listener->service, sizeof listener->service, "%d",
			 ntohs(addr.sin_port));
		SOCK_LOG_DBG("Bound to port: %s - %d\n", listener->service, getpid());
	}

	if (ep->src_addr->sin_addr.s_addr == 0) {
		sprintf(service, "%s", listener->service);
		ret = sock_get_src_addr_from_hostname(ep->src_addr, service);
		if (ret)
			goto err;
	}

	if (listen(listen_fd, sock_cm_def_map_sz)) {
		SOCK_LOG_ERROR("failed to listen socket: %s\n", strerror(errno));
		goto err;
	}

	((struct sockaddr_in *) (ep->src_addr))->sin_port =
		htons(atoi(listener->service));
	listener->do_listen = 1;
	listener->sock = listen_fd;

	sock_fabric_add_service(domain->fab, atoi(listener->service));
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, listener->signal_fds) < 0)
		goto err;

	fd_set_nonblock(listener->signal_fds[1]);
	if (pthread_create(&listener->listener_thread, 0,
			   _sock_conn_listen, ep)) {
		SOCK_LOG_ERROR("failed to create conn listener thread\n");
		goto err;
	} while (!*((volatile int*)&listener->is_ready));
	return 0;
err:
	if (listen_fd >= 0)
		close(listen_fd);
	return -FI_EINVAL;
}

struct sock_conn *sock_ep_connect(struct sock_ep *ep, fi_addr_t index)
{
	int conn_fd = -1, ret;
	int do_retry = sock_conn_retry;
	struct sock_conn *conn, *new_conn;
	uint16_t idx;
	struct sockaddr_in *addr;
	socklen_t lon;
	int valopt = 0;
	struct pollfd poll_fd;

	if (ep->ep_type == FI_EP_MSG) {
		idx = 0;
		addr = ep->dest_addr;
	} else {
		idx = index & ep->av->mask;
		addr = (struct sockaddr_in *)&ep->av->table[idx].addr;
	}

do_connect:
	fastlock_acquire(&ep->cmap.lock);
	conn = sock_ep_lookup_conn(ep, index, addr);
	fastlock_release(&ep->cmap.lock);

	if (conn != SOCK_CM_CONN_IN_PROGRESS)
		return conn;

	conn_fd = socket(AF_INET, SOCK_STREAM, 0);
	if (conn_fd == -1) {
		SOCK_LOG_ERROR("failed to create conn_fd, errno: %d\n", errno);
		errno = FI_EOTHER;
		return NULL;
	}

	ret = fd_set_nonblock(conn_fd);
	if (ret) {
		SOCK_LOG_ERROR("failed to set conn_fd nonblocking, errno: %d\n", errno);
		errno = FI_EOTHER;
		close(conn_fd);
                return NULL;
	}

	SOCK_LOG_DBG("Connecting to: %s:%d\n", inet_ntoa(addr->sin_addr),
			ntohs(addr->sin_port));
	SOCK_LOG_DBG("Connecting using address:%s\n",
			inet_ntoa(ep->src_addr->sin_addr));

	ret = connect(conn_fd, (struct sockaddr *) addr, sizeof *addr);
	if (ret < 0) {
		if (errno == EINPROGRESS) {
			poll_fd.fd = conn_fd;
			poll_fd.events = POLLOUT;

			ret = poll(&poll_fd, 1, 15 * 1000);
			if (ret < 0) {
				SOCK_LOG_DBG("poll failed\n");
				goto retry;
			}

			lon = sizeof(int);
			ret = getsockopt(conn_fd, SOL_SOCKET, SO_ERROR, (void*)(&valopt), &lon);
			if (ret < 0) {
				SOCK_LOG_DBG("getsockopt failed: %d, %d\n", ret, conn_fd);
				goto retry;
			}

			if (valopt) {
				SOCK_LOG_DBG("Error in connection() %d - %s - %d\n", valopt, strerror(valopt), conn_fd);
				SOCK_LOG_DBG("Connecting to: %s:%d\n", inet_ntoa(addr->sin_addr),
						ntohs(addr->sin_port));
				SOCK_LOG_DBG("Connecting using address:%s\n",
				inet_ntoa(ep->src_addr->sin_addr));
				goto retry;
			}
			goto out;
		} else {
			SOCK_LOG_DBG("Timeout or error() - %s: %d\n", strerror(errno), conn_fd);
			SOCK_LOG_DBG("Connecting to: %s:%d\n", inet_ntoa(addr->sin_addr),
					ntohs(addr->sin_port));
			SOCK_LOG_DBG("Connecting using address:%s\n",
					inet_ntoa(ep->src_addr->sin_addr));
			goto retry;
		}
	} else {
		goto out;
	}

retry:
	do_retry--;
	sleep(10);
	if (!do_retry)
		goto err;

	if (conn_fd != -1) {
		close(conn_fd);
		conn_fd = -1;
	}

	SOCK_LOG_ERROR("Connect error, retrying - %s - %d\n", strerror(errno), conn_fd);
	SOCK_LOG_DBG("Connecting to: %s:%d\n", inet_ntoa(addr->sin_addr),
			ntohs(addr->sin_port));
	SOCK_LOG_DBG("Connecting using address:%s\n",
			inet_ntoa(ep->src_addr->sin_addr));
        goto do_connect;

out:
	fastlock_acquire(&ep->cmap.lock);
	new_conn = sock_conn_map_insert(ep, addr, conn_fd, 0);
	new_conn->av_index = (ep->ep_type == FI_EP_MSG) ? FI_ADDR_NOTAVAIL : (fi_addr_t) idx;
	conn = idm_lookup(&ep->av_idm, index);
	if (conn == SOCK_CM_CONN_IN_PROGRESS) {
		idm_set(&ep->av_idm, index, new_conn);
		conn = new_conn;
	}
	fastlock_release(&ep->cmap.lock);
	return conn;

err:
	close(conn_fd);
	return NULL;
}

