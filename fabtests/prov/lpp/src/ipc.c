/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdatomic.h>

#include "ipc.h"
#include "test_util.h"
#include "error.h"

static pthread_t info_server_pt;
static int server_running;

static int info_server_port = 9000;
static struct rank_info rank_info[MAX_RANK] = { 0 };
// TODO: do we need to do this? Or can we just keep the peer rank info on the stack of each test fn?
static struct rank_info peer_rank_info[MAX_RANK] = { 0 };
static const int max_wait_up_s = 120;

struct hello {
	uint64_t rank;
	int64_t iteration;
	int32_t test_num;
};

enum ipc_msg_type {
	IPC_INVALID = 0,
	IPC_HELLO,
	IPC_HELLO_YOURSELF,
	IPC_XCHG,
	IPC_BARRIER,

	IPC_MAX = IPC_BARRIER,
};

const char *ipc_msg_type_strs[] = {
	"INVALID",
	"HELLO",
	"HELLO_YOURSELF",
	"XCHG",
	"BARRIER",
};

struct ipc_msg {
	uint8_t type;
	uint32_t length;
	uint8_t payload[];
};

static struct sockaddr_in peer_info_addr;
socklen_t peeraddr_len;

static int send_ipc_msg(int socket, enum ipc_msg_type type, uint32_t length,
			void *payload)
{
	struct iovec iov[2];
	struct ipc_msg msg;
	ssize_t total_bytes;
	ssize_t ret;
	int iovcnt;

	msg.type = type;
	msg.length = length;

	iov[0].iov_base = &msg;
	iov[0].iov_len = sizeof(msg);
	total_bytes = iov[0].iov_len;
	if (length > 0) {
		iov[1].iov_base = payload;
		iov[1].iov_len = length;
		total_bytes += iov[1].iov_len;
		iovcnt = 2;
	} else {
		iovcnt = 1;
	}

	errno = 0;
	ret = writev(socket, iov, iovcnt);
	if (ret != total_bytes) {
		return -1;
	}
	return 0;
}

static int recv_ipc_msg(int socket, enum ipc_msg_type want_type, void *payload,
			uint32_t length, bool block)
{
	struct ipc_msg msg;
	uint8_t *recvbuf;
	int flags;
	ssize_t total;
	ssize_t recvd;
	ssize_t ret;

	flags = block ? MSG_WAITALL : MSG_DONTWAIT;
	total = sizeof(msg);
	recvd = 0;
	while (recvd != total) {
		recvbuf = (uint8_t*)&msg;
		errno = 0;
		ret = recv(socket, recvbuf, total - recvd, flags);
		if (ret <= 0) {
			return (ret == 0 ? -1 : ret);
		}
		recvd += ret;
		recvbuf += ret;
		// Our recv buffer is on the stack; we're committed now to
		// getting the rest of the message.
		flags = MSG_WAITALL;
	}

	if (msg.type != want_type) {
		errorx("invalid message type, want %d(%s), got %d(%s)\n",
		       want_type, ipc_msg_type_strs[want_type], msg.type,
		       msg.type <= IPC_MAX ? ipc_msg_type_strs[msg.type] :
					     "UNKNOWN");
	}

	if (length > 0) {
		errno = 0;
		ret = recv(socket, payload, length, MSG_WAITALL);
		if (ret <= 0) {
			return (ret == 0 ? -1 : ret);
		}
	}

	return 0;
}

static void get_peer_addrinfo(const char *peerhostname)
{
	struct addrinfo hints, *serverinfo;
	int ret;
	char portstr[1024];
	int portlist[] = { info_server_port };
	struct sockaddr_in *addrlist[] = { &peer_info_addr };

	for (unsigned int i = 0; i < sizeof(addrlist) / sizeof(addrlist[0]);
	     i++) {
		memset(&hints, 0, sizeof(hints));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_DGRAM;

		sprintf(portstr, "%d", portlist[i]);
		if ((ret = getaddrinfo(peerhostname, portstr, &hints, &serverinfo)) != 0) {
			errorx("getaddrinfo(%s,...) %s", peerhostname, gai_strerror(ret));
		}
		if (serverinfo == NULL) {
			errorx("getaddrinfo returned no results");
		}

		*addrlist[i] = *(struct sockaddr_in*)serverinfo->ai_addr;
		peeraddr_len = serverinfo->ai_addrlen;

		freeaddrinfo(serverinfo);
	}
}

static int setup_server(int port)
{
	int sockfd;
	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		error("socket() failed");
	}

	struct sockaddr_in serveraddr;
	serveraddr.sin_family = AF_INET;
	serveraddr.sin_addr.s_addr = INADDR_ANY;
	serveraddr.sin_port = htons(port);

	int enable = 1;
	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0) {
		error("setsocketopt");
	}

	enable = 1;
	if (setsockopt(sockfd, SOL_TCP, TCP_NODELAY, &enable, sizeof(enable)) < 0) {
		error("setsocketopt");
	}

	if (bind(sockfd, (struct sockaddr*)&serveraddr, sizeof(serveraddr)) < 0) {
		error("bind() failed");
	}

	if (listen(sockfd, 10) < 0) {
		error("listen() failed");
	}

	return sockfd;
}

static void *info_server_thread(void *arg)
{
	debug("starting info server on port %d\n", info_server_port);

	int info_server_sock = setup_server(info_server_port);

	while (server_running) {
		int clientsock = accept(info_server_sock, NULL, NULL);
		if (clientsock < 0) {
			error("accept");
		}

		struct hello hello;
		int ret = recv_ipc_msg(clientsock, IPC_HELLO, &hello,
				       sizeof(hello), true);
		if (ret) {
			error("recv IPC_HELLO failed: %d\n", ret);
		}
		if (hello.rank >= MAX_RANK) {
			errorx("rank %lu too large", hello.rank);
		}

		struct rank_info *ri = &rank_info[hello.rank];
		pthread_mutex_lock(&ri->lock);
		if (ri->valid != 1 || ri->iteration != hello.iteration ||
		    ri->peer_comm_sock >= 0) {
			if (verbose) {
				debug("close() valid %d iteration %ld %ld peer_comm_sock %d\n",
				      ri->valid, ri->iteration, hello.iteration,
				      ri->peer_comm_sock);
			}
			close(clientsock);
		} else {
			ri->cur_test_num = hello.test_num;
			ret = send_ipc_msg(clientsock, IPC_HELLO_YOURSELF, 0,
					   NULL);
			if (ret) {
				error("failed to send hello");
			}
			// This signals a thread in wait_for_peer_up() they can
			// proceed.
			atomic_store(&ri->peer_comm_sock, clientsock);
		}
		pthread_mutex_unlock(&ri->lock);
	}

	close(info_server_sock);

	return NULL;
}

static void clear_rank_info(struct rank_info *ri)
{
	ri->valid = 0;
	ri->rank = 0;
	ri->iteration = 0;
	ri->peer_comm_sock = 0;

	ri->cur_test_name = NULL;
	ri->cur_test_num = 0;

	ri->fabric = NULL;
	ri->domain = NULL;
	ri->fi = NULL;

	memset(&ri->sync_op_context, 0, sizeof(ri->sync_op_context));
	ri->context_tree_root = NULL;

	ri->n_mr_info = 0;
	memset(ri->mr_info, 0, sizeof(ri->mr_info));

	ri->n_ep_info = 0;
	memset(ri->ep_info, 0, sizeof(ri->ep_info));

	ri->tracei = 0;
	memset(ri->trace_lines, 0, sizeof(ri->trace_lines));
	memset(ri->trace_files, 0, sizeof(ri->trace_files));
	memset(ri->trace_funcs, 0, sizeof(ri->trace_funcs));
}

struct rank_info *get_rank_info(uint64_t rank, int64_t iteration)
{
	if (rank >= MAX_RANK) {
		errorx("rank %ld out of range", rank);
	}

	struct rank_info *ri = &rank_info[rank];

	pthread_mutex_lock(&ri->lock);
	if (ri->valid == 1) {
		errorx("duplicate get for rank %ld", rank);
	}
	clear_rank_info(ri);
	ri->iteration = iteration;
	ri->rank = rank;
	ri->peer_comm_sock = -1;
	ri->cur_test_name = "N/A";
	ri->cur_test_num = -1;
	ri->valid = 1;
	pthread_mutex_unlock(&ri->lock);

	return ri;
}

void put_rank_info(struct rank_info *ri)
{
	pthread_mutex_lock(&ri->lock);
	close(ri->peer_comm_sock);
	free_ctx_tree(ri);
	ri->valid = 0;
	pthread_mutex_unlock(&ri->lock);
}

struct rank_info *exchange_rank_info(struct rank_info *ri)
{
	int ret;

	ret = send_ipc_msg(ri->peer_comm_sock, IPC_XCHG, sizeof(*ri), ri);
	if (ret) {
		ERROR(ri, "send XCHG failed");
	}

	struct rank_info *pri = &peer_rank_info[ri->rank];
	ret = recv_ipc_msg(ri->peer_comm_sock, IPC_XCHG, pri, sizeof(*pri),
			   true);
	if (ret) {
		ERROR(ri, "recv (our peer likely failed)");
	}

	return pri;
}

void peer_barrier(struct rank_info *ri)
{
	const size_t max_wait_s = 60;
	size_t waited_us = 0;
	int ret;

	ret = send_ipc_msg(ri->peer_comm_sock, IPC_BARRIER, 0, NULL);
	if (ret) {
		ERROR(ri, "send BARRIER failed");
	}

	while (1) {
		ret = recv_ipc_msg(ri->peer_comm_sock, IPC_BARRIER, NULL, 0,
				   false);
		if (ret == 0) {
			break;
		} else if (ret < 0 && errno != EAGAIN) {
			ERROR(ri, "recv (our peer likely failed)");
		}

		// We must use our thread to make progress in FI_PROGRESS_MANUAL
		// mode.
		for (int i = 0; i < MAX_EP_INFO; i++) {
			struct ep_info *ep_info = &ri->ep_info[i];
			if (ep_info->valid) {
				fi_cntr_read(ep_info->tx_cntr_fid);
			}
		}
		usleep(2000);
		waited_us += 2000;
		if (waited_us > max_wait_s * 1000000) {
			ERROR(ri, "waited for peer for %zu seconds\n",
			      max_wait_s);
		}
	}
}

void put_peer_rank_info(struct rank_info *pri)
{
	memset(pri, 0, sizeof(*pri));
}

void announce_peer_up(struct rank_info *ri, int64_t iteration)
{
	int waited_ms = 0;

	while (1) {
		int sockfd = socket(AF_INET, SOCK_STREAM, 0);
		if (sockfd < 0) {
			ERROR(ri, "socket");
		}

		int enable = 1;
		if (setsockopt(sockfd, SOL_TCP, TCP_NODELAY, &enable,
			       sizeof(enable)) < 0) {
			ERROR(ri, "setsocketopt");
		}

		if (connect(sockfd, &peer_info_addr, peeraddr_len) >= 0) {
			struct hello hello;
			hello.rank = ri->rank;
			hello.iteration = iteration;
			hello.test_num = ri->cur_test_num;
			int ret = send_ipc_msg(sockfd, IPC_HELLO, sizeof(hello),
					       &hello);
			if (ret) {
				ERROR(ri, "send HELLO failed");
			}
			ret = recv_ipc_msg(sockfd, IPC_HELLO_YOURSELF, NULL, 0,
					   true);
			if (ret == 0) {
				ri->peer_comm_sock = sockfd;
				break;
			}
		}
		close(sockfd);
		usleep(10000);
		waited_ms += 10;
		if (waited_ms > max_wait_up_s * 1000) {
			ERRORX(ri, "waited for peer up for %ds\n",
					max_wait_up_s);
		}
	}
}

void wait_for_peer_up(struct rank_info *ri)
{
	int waited_ms = 0;

	 while (atomic_load(&ri->peer_comm_sock) < 0) {
		 usleep(10000);
		 waited_ms += 10;
		 if (waited_ms > max_wait_up_s * 1000) {
			 ERRORX(ri, "waited for peer up for %ds\n",
				max_wait_up_s);
		 }
	 }
	 if (ri->cur_test_num < 0) {
		 ERRORX(ri, "peer failed to provide valid test num: %d\n",
			ri->cur_test_num);
	 }
}

void server_init(const char *peerhostname, int server_port)
{
	for (int i = 0; i < MAX_RANK; i++) {
		pthread_mutex_init(&rank_info[i].lock, NULL);
	}

	if (server_port > 0) {
		info_server_port = server_port;
	}
	server_running = 1;
	get_peer_addrinfo(peerhostname);
	if (pthread_create(&info_server_pt, NULL, info_server_thread, NULL)) {
		errorx("Failed to create server thread errno: %d\n", errno);
	}
}
