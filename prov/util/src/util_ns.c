/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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

/*
 * A name server is started on each node as a thread within one of
 * the processes on that node. It maintains a database that maps
 * "services" to "endpoint names". Other processes on the same node
 * talk to this name server to update mapping information.
 *
 * To resolve a "node:service" pair into an provider internal endpoint name
 * that can be used as the input of fi_av_insert, a process needs to make
 * a query to the name server residing on "node".
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <fi_util.h>
#include <rdma/providers/fi_log.h>
#include <fi.h>

#include "rbtree.h"

#define OFI_NS_DEFAULT_HOSTNAME	"localhost"

#define OFI_NS_SOCKET_OP(op)							\
static inline									\
ssize_t util_ns_##op##_socket_op(SOCKET sock, void *buf, size_t len)		\
{										\
	ssize_t ret = 0, bytes = 0;						\
	while (bytes != (len) && ret >= 0) {					\
		ret = ofi_##op##_socket((sock),					\
					(void *)((char *) (buf) + bytes),	\
					(len) - bytes);				\
		bytes = ((ret < 0) ? -1 : bytes + ret);				\
	}									\
	return bytes;								\
}

OFI_NS_SOCKET_OP(write)
OFI_NS_SOCKET_OP(read)

enum {
	OFI_UTIL_NS_ADD,
	OFI_UTIL_NS_DEL,
	OFI_UTIL_NS_QUERY,
	OFI_UTIL_NS_ACK,
};

struct util_ns_cmd {
	int	op;
	int	status;
};

const size_t cmd_len = sizeof(struct util_ns_cmd);

static int util_ns_map_init(struct util_ns *ns)
{
	ns->ns_map = rbtNew(ns->service_cmp);
	return ns->ns_map ? 0 : -FI_ENOMEM;
}

static void util_ns_map_fini(struct util_ns *ns)
{
	rbtDelete(ns->ns_map);
}

static int util_ns_map_add(struct util_ns *ns, void *service_in,
			   void *name_in)
{
	void *name, *service;
	int ret;

	service = calloc(ns->service_len, 1);
	if (!service) {
		ret = -FI_ENOMEM;
		goto err1;
	}
	memcpy(service, service_in, ns->service_len);

	name = calloc(ns->name_len, 1);
	if (!name) {
		ret = -FI_ENOMEM;
		goto err2;
	}
	memcpy(name, name_in, ns->name_len);

	if (rbtFind(ns->ns_map, service)) {
		ret = -FI_EADDRINUSE;
		goto err3;
	}

	if (rbtInsert(ns->ns_map, service, name)) {
		ret = -FI_ENOMEM;
		goto err3;
	}
	return FI_SUCCESS;

err3:
	free(name);
err2:
	free(service);
err1:
	return ret;
}
 
static int util_ns_map_del(struct util_ns *ns, void *service_in,
			   void *name_in)
{
	RbtIterator it;
	int ret = -FI_ENOENT;
	void *service, *name;

        it = rbtFind(ns->ns_map, service_in);
        if (it) {
		rbtKeyValue(ns->ns_map, it, &service, &name);
		if (memcmp(name, name_in, ns->name_len))
			return ret;
		free(service);
		free(name);
		rbtErase(ns->ns_map, it);
		ret = FI_SUCCESS;
	}

	return ret;
}

static int util_ns_map_lookup(struct util_ns *ns, void *service_in,
			      void *name_out)
{
	RbtIterator it;
	void *key, *name;

        it = rbtFind(ns->ns_map, service_in);
	if (!it)
		return -FI_ENOENT;

	rbtKeyValue(ns->ns_map, it, &key, (void **)&name);
	memcpy(name_out, name, ns->name_len);

	if (ns->is_service_wildcard && ns->is_service_wildcard(service_in))
		memcpy(service_in, key, ns->service_len);

	return FI_SUCCESS;
}

static void util_ns_name_server_cleanup(void *args)
{
	void **cleanup_args = (void **)args;
	ofi_close_socket((uintptr_t)cleanup_args[0]);
	util_ns_map_fini((struct util_ns *)cleanup_args[1]);
}

static int util_ns_op_dispatcher(struct util_ns *ns,
				 struct util_ns_cmd *cmd,
				 SOCKET sock)
{
	int ret = FI_SUCCESS;
	size_t io_len = 0;
	void *io_buf = NULL, *service, *name;

	switch (cmd->op) {
	case OFI_UTIL_NS_ADD:
	case OFI_UTIL_NS_DEL:
		io_len = ns->name_len + ns->service_len;
		io_buf = calloc(io_len, 1);
		if (!io_buf) {
			ret = -FI_ENOMEM;
			goto fn1;
		}

		ret = util_ns_read_socket_op(sock, io_buf, io_len);
		if (ret == io_len) {
			service = io_buf;
			name = (void *)((char *)io_buf + ns->service_len);
			ret = (cmd->op == OFI_UTIL_NS_ADD) ?
			      util_ns_map_add(ns, service, name) :
			      util_ns_map_del(ns, service, name);
		} else {
			ret = -FI_ENODATA;
		}
		goto fn2;

	case OFI_UTIL_NS_QUERY:
		io_len = ns->service_len;
		/* allocate io_buf to be large enough */
		io_buf = calloc(
			cmd_len + ns->service_len + ns->name_len, 1
		);
		if (!io_buf) {
			ret = -FI_ENOMEM;
			goto fn1;
		}

		memcpy(io_buf, cmd, cmd_len);
		cmd = io_buf;
		service = (char *)io_buf + cmd_len;
		name = (char *)service + ns->service_len;

		ret = util_ns_read_socket_op(sock, service, io_len);
		if (ret == io_len) {
			cmd->op = OFI_UTIL_NS_ACK;
			cmd->status = util_ns_map_lookup(
				ns, service, name
			);
		} else {
			ret = -FI_ENODATA;
			goto fn2;
		}

		if (!cmd->status)
			io_len = cmd_len + ns->service_len + ns->name_len;
		else
			io_len = cmd_len;
		ret = util_ns_write_socket_op(sock, io_buf, io_len);
		ret = ((ret == cmd_len) ? FI_SUCCESS : -FI_ENODATA);
		goto fn2;

	default:
		ret = -FI_ENODATA;
		assert(0);
		goto fn1;
	}

fn2:
	free(io_buf);
fn1:
	return ret;
}

static void *util_ns_name_server_func(void *args)
{
	struct util_ns *ns;
	struct addrinfo hints = {
		.ai_flags = AI_PASSIVE,
		.ai_family = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	struct addrinfo *res, *p;
	void *cleanup_args[2];
	char *service;
	SOCKET listenfd = INVALID_SOCKET, connfd;
	int n, ret;
	struct util_ns_cmd cmd = (const struct util_ns_cmd){ 0 };

	ns = (struct util_ns *)args;

	if (asprintf(&service, "%d", ns->ns_port) < 0)
		return NULL;

	n = getaddrinfo(NULL, service, &hints, &res);
	if (n < 0) {
		free(service);
		return NULL;
	}

	for (p = res; p; p = p->ai_next) {
		listenfd = ofi_socket(p->ai_family, p->ai_socktype,
				      p->ai_protocol);
		if (listenfd != INVALID_SOCKET) {
			n = 1;
			(void) setsockopt(listenfd, SOL_SOCKET,
					  SO_REUSEADDR, &n, sizeof(n));
			if (!bind(listenfd, p->ai_addr, p->ai_addrlen))
				break;
			ofi_close_socket(listenfd);
			listenfd = INVALID_SOCKET;
		}
	}

	freeaddrinfo(res);
	free(service);

	if (listenfd == INVALID_SOCKET)
		return NULL;

	if (util_ns_map_init(ns))
		goto done;

	ret = listen(listenfd, 256);
	if (ret)
		goto done;

	cleanup_args[0] = (void *)(uintptr_t)listenfd;
	cleanup_args[1] = (void *)ns;
	pthread_cleanup_push(util_ns_name_server_cleanup,
			     (void *)cleanup_args);

	while (1) {
		connfd = accept(listenfd, NULL, 0);
		if (connfd != INVALID_SOCKET) {
			/* Read service data */
			ret = ofi_read_socket(connfd, &cmd, cmd_len);
			if (ret == cmd_len) {
				(void) util_ns_op_dispatcher(ns, &cmd,
							     connfd);
			}
			ofi_close_socket(connfd);
		}
	}

	pthread_cleanup_pop(1);

done:
	ofi_close_socket(listenfd);
	return NULL;
}


/*
 * Name server API: client side
 */

static int util_ns_connect_server(struct util_ns *ns, const char *server)
{
	struct addrinfo hints = {
		.ai_family   = AF_UNSPEC,
		.ai_socktype = SOCK_STREAM
	};
	struct addrinfo *res, *p;
	char *service;
	SOCKET sockfd = INVALID_SOCKET;
	int n;

	if (asprintf(&service, "%d", ns->ns_port) < 0)
		return -1;

	n = getaddrinfo(server, service, &hints, &res);
	if (n < 0) {
		free(service);
		return -1;
	}

	for (p = res; p; p = p->ai_next) {
		sockfd = ofi_socket(p->ai_family, p->ai_socktype, p->ai_protocol);
		if (sockfd != INVALID_SOCKET) {
			if (!connect(sockfd, p->ai_addr, p->ai_addrlen))
				break;
			ofi_close_socket(sockfd);
			sockfd = INVALID_SOCKET;
		}
	}

	freeaddrinfo(res);
	free(service);

	return sockfd;
}

int ofi_ns_add_local_name(struct util_ns *ns, void *service, void *name)
{
	SOCKET sockfd;
	int ret;
	char *server = (ns->ns_hostname ?
		ns->ns_hostname : OFI_NS_DEFAULT_HOSTNAME);
	void *write_buf;
	size_t write_len = 0;
	struct util_ns_cmd cmd = {
		.op = OFI_UTIL_NS_ADD,
		.status = 0,
	};

	write_buf = calloc(cmd_len + ns->service_len + ns->name_len, 1);
	if (!write_buf) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	memcpy(write_buf, &cmd, cmd_len);
	write_len += cmd_len;
	memcpy((void *)((char *)write_buf + write_len), service,
	       ns->service_len);
	write_len += ns->service_len;
	memcpy((void *)((char *)write_buf + write_len), name,
	       ns->name_len);
	write_len += ns->name_len;

	sockfd = util_ns_connect_server(ns, server);
	if (sockfd == INVALID_SOCKET) {
		ret = -FI_ENODATA;
		goto err2;
	}

	ret = util_ns_write_socket_op(sockfd, write_buf, write_len);
	ret = ((ret == write_len) ? FI_SUCCESS : -FI_ENODATA);

	ofi_close_socket(sockfd);
err2:
	free(write_buf);
err1:
	return ret;
}

int ofi_ns_del_local_name(struct util_ns *ns, void *service, void *name)
{
	SOCKET sockfd;
	int ret;
	const char *server_hostname = (ns->ns_hostname ?
		ns->ns_hostname : OFI_NS_DEFAULT_HOSTNAME);
	void *write_buf;
	size_t write_len = 0;
	struct util_ns_cmd cmd = {
		.op = OFI_UTIL_NS_DEL,
		.status = 0,
	};

	write_buf = calloc(cmd_len + ns->service_len + ns->name_len, 1);
	if (!write_buf) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	memcpy(write_buf, &cmd, cmd_len);
	write_len += cmd_len;
	memcpy((void *)((char *)write_buf + write_len), service,
	       ns->service_len);
	write_len += ns->service_len;
	memcpy((void *)((char *)write_buf + write_len), name,
	       ns->name_len);
	write_len += ns->name_len;

	sockfd = util_ns_connect_server(ns, server_hostname);
	if (sockfd == INVALID_SOCKET) {
		ret = -FI_ENODATA;
		goto err2;
	}

	ret = util_ns_write_socket_op(sockfd, write_buf, write_len);
	ret = ((ret == write_len) ? FI_SUCCESS : -FI_ENODATA);

	ofi_close_socket(sockfd);
err2:
	free(write_buf);
err1:
	return ret;
}

void *ofi_ns_resolve_name(struct util_ns *ns, const char *server_hostname,
			  void *service)
{
	void *dest_addr = NULL, *io_buf;
	size_t io_len = 0;
	SOCKET sockfd;
	ssize_t ret = 0;
	struct util_ns_cmd cmd = {
		.op = OFI_UTIL_NS_QUERY,
		.status = 0,
	};

	sockfd = util_ns_connect_server(ns, server_hostname);
	if (sockfd == INVALID_SOCKET)
		goto err1;

	io_buf = calloc(cmd_len + ns->service_len, 1);
	if (!io_buf)
		goto err2;

	memcpy(io_buf, &cmd, cmd_len);
	io_len += cmd_len;
	memcpy((void *)((char *)io_buf + io_len), service,
	       ns->service_len);
	io_len += ns->service_len;

	ret = util_ns_write_socket_op(sockfd, io_buf, io_len);
	if (ret < 0)
		goto err3;

	free(io_buf);

	io_len = ns->service_len + ns->name_len;
	io_buf = calloc(io_len, 1);
	if (!io_buf)
		goto err2;

	ret = util_ns_read_socket_op(sockfd, &cmd, cmd_len);
	if (ret < 0 || cmd.status)
		goto err3;

	ret = util_ns_read_socket_op(sockfd, io_buf, io_len);
	if (ret == io_len) {
		dest_addr = calloc(ns->name_len, 1);
		if (!dest_addr)
			goto err3;
		io_len = 0;
		memcpy(service, (void *)((char *)io_buf + io_len),
		       ns->service_len);
		io_len += ns->service_len;
		memcpy(dest_addr, (void *)((char *)io_buf + io_len),
		       ns->name_len);
	}

err3:
	free(io_buf);
err2:
	ofi_close_socket(sockfd);
err1:
	return dest_addr;
}

/*
 * Name server API: server side
 */

void ofi_ns_start_server(struct util_ns *ns)
{
	int ret;
	SOCKET sockfd;
	int sleep_usec = 1000;
	char *server_hostname = (ns->ns_hostname ?
		ns->ns_hostname : OFI_NS_DEFAULT_HOSTNAME);

	ofi_osd_init();

	ret = pthread_create(&ns->ns_thread, NULL,
			     util_ns_name_server_func, (void *)ns);
	if (ret) {
		/*
		 * use the main thread's ID as invalid
		 * value for the new thread
		 */
		ns->ns_thread = pthread_self();
	}

	/*
	 * Wait for the local name server to come up. It could be the thread
	 * created above, or the thread created by another process on the same
	 * node. The total wait time is about (1+2+4+...+8192)ms = 16 seconds.
	 */
	while (sleep_usec < 10000) {
		sockfd = util_ns_connect_server(ns, server_hostname);
		if (sockfd != INVALID_SOCKET) {
			ofi_close_socket(sockfd);
			return;
		}
		usleep(sleep_usec);
		sleep_usec *= 2;
	}
}

void ofi_ns_stop_server(struct util_ns *ns)
{
	ofi_osd_fini();

	if (pthread_equal(ns->ns_thread, pthread_self()))
		return;

	(void) pthread_cancel(ns->ns_thread);
	(void) pthread_join(ns->ns_thread, NULL);
}

int ofi_ns_init(struct util_ns_attr *attr, struct util_ns *ns)
{
	if (!ns || !attr || !attr->name_len ||
	    !attr->service_len || !attr->service_cmp)
		return -FI_EINVAL;

	ns->name_len = attr->name_len;
	ns->service_len = attr->service_len;
	ns->service_cmp = attr->service_cmp;
	ns->is_service_wildcard = attr->is_service_wildcard;
	ns->ns_port = attr->ns_port;
	if (attr->ns_hostname)
		ns->ns_hostname = strdup(attr->ns_hostname);

	return FI_SUCCESS;
}
