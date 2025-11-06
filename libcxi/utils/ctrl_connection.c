/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2021-2025 Hewlett Packard Enterprise Development LP
 */

/* IP/TCP control messaging for CXI benchmarks */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include "utils_common.h"

static struct timeval dflt_tmo = { .tv_sec = 5, .tv_usec = 0 };

/* Get IP/TCP address for the given host and port */
static int ctrl_getaddrinfo(const char *node, uint16_t port,
			    struct addrinfo **res)
{
	int rc;
	char service[6];

	struct addrinfo hints = { .ai_flags = AI_NUMERICSERV,
				  .ai_family = AF_INET,
				  .ai_socktype = SOCK_STREAM,
				  .ai_protocol = IPPROTO_TCP };

	snprintf(service, 6, "%" PRIu16, port);

	rc = getaddrinfo(node, service, &hints, res);
	if (rc != 0) {
		fprintf(stderr, "getaddrinfo() failed: %s\n", gai_strerror(rc));
		return -EIO;
	}
	if (!*res) {
		fprintf(stderr, "getaddrinfo returned NULL list\n");
		return -EIO;
	}

	return rc;
}

/* Initialize client and connect */
static int ctrl_init_client(struct ctrl_connection *ctrl)
{
	struct sockaddr_in addr_in = { 0 };
	struct addrinfo *res;
	struct addrinfo *next_res;
	int last_errno = 0;
	int rc;

	rc = ctrl_getaddrinfo(ctrl->dst_addr, ctrl->dst_port, &res);
	if (rc)
		return rc;

	for (next_res = res; next_res; next_res = next_res->ai_next) {
		ctrl->fd = socket(next_res->ai_family, next_res->ai_socktype,
				  next_res->ai_protocol);
		if (ctrl->fd == -1) {
			last_errno = errno;
			continue;
		}

		if (ctrl->src_port != 0) {
			addr_in.sin_family = AF_INET;
			addr_in.sin_port = htons(ctrl->src_port);
			addr_in.sin_addr.s_addr = htonl(INADDR_ANY);

			rc = bind(ctrl->fd, (struct sockaddr *)&addr_in,
				  sizeof(addr_in));
			if (rc == -1) {
				last_errno = errno;
				close(ctrl->fd);
				continue;
			}
		}

		rc = connect(ctrl->fd, next_res->ai_addr, next_res->ai_addrlen);
		if (rc == -1) {
			last_errno = errno;
			close(ctrl->fd);
		} else {
			break;
		}
	}

	if (!next_res || rc == -1) {
		rc = -last_errno;
		ctrl->fd = -1;
		fprintf(stderr, "Failed to connect to %s: %s\n", ctrl->dst_addr,
			strerror(last_errno));
	}

	freeaddrinfo(res);

	return rc;
}

/* Initialize server and wait for client to connect */
static int ctrl_init_server(struct ctrl_connection *ctrl)
{
	struct sockaddr_in addr_in = { 0 };
	int optval = 1;
	int fd;
	int rc;

	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd == -1) {
		rc = -errno;
		fprintf(stderr, "socket() failed: %s\n", strerror(-rc));
		return rc;
	}

	rc = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
	if (rc == -1) {
		rc = -errno;
		fprintf(stderr, "setsockopt(SO_REUSEADDR) failed: %s\n",
			strerror(-rc));
		goto done;
	}

	addr_in.sin_family = AF_INET;
	addr_in.sin_port = htons(ctrl->src_port);
	addr_in.sin_addr.s_addr = htonl(INADDR_ANY);

	rc = bind(fd, (struct sockaddr *)&addr_in, sizeof(addr_in));
	if (rc == -1) {
		rc = -errno;
		fprintf(stderr, "bind() failed: %s\n", strerror(-rc));
		goto done;
	}

	rc = listen(fd, 10);
	if (rc == -1) {
		rc = -errno;
		fprintf(stderr, "listen() failed: %s\n", strerror(-rc));
		goto done;
	}

	printf("Listening on port %u for client to connect...\n",
	       ctrl->src_port);

	ctrl->fd = accept(fd, NULL, NULL);
	if (ctrl->fd == -1) {
		rc = -errno;
		fprintf(stderr, "accept() failed: %s\n", strerror(-rc));
		goto done;
	}

done:
	if (fd != -1)
		close(fd);

	return rc;
}

/* Change the receive timeout */
static int ctrl_set_rcvtmo(struct ctrl_connection *ctrl, struct timeval *tv)
{
	int rc;

	if (!ctrl || !tv)
		return -EINVAL;

	if ((tv->tv_sec == ctrl->last_rcvtmo.tv_sec) &&
	    (tv->tv_usec == ctrl->last_rcvtmo.tv_usec))
		return 0;

	rc = setsockopt(ctrl->fd, SOL_SOCKET, SO_RCVTIMEO, tv, sizeof(*tv));
	if (rc == -1) {
		rc = -errno;
		fprintf(stderr, "setsockopt(SO_RCVTIMEO) failed: %s\n",
			strerror(-rc));
	}
	ctrl->last_rcvtmo.tv_sec = tv->tv_sec;
	ctrl->last_rcvtmo.tv_usec = tv->tv_usec;

	return rc;
}

/* Initialize control messaging over IP/TCP */
static int ctrl_init(struct ctrl_connection *ctrl)
{
	int rc;

	if (!ctrl)
		return -EINVAL;

	if (ctrl->dst_addr) {
		ctrl->is_server = false;
		if (!ctrl->dst_port)
			ctrl->dst_port = DFLT_PORT;
		rc = ctrl_init_client(ctrl);
		if (rc)
			goto done;
	} else {
		ctrl->is_server = true;
		if (!ctrl->src_port)
			ctrl->src_port = DFLT_PORT;
		rc = ctrl_init_server(ctrl);
		if (rc)
			goto done;
	}

	ctrl->last_rcvtmo.tv_sec = 0;
	ctrl->last_rcvtmo.tv_usec = 0;
	rc = ctrl_set_rcvtmo(ctrl, &dflt_tmo);
	if (rc)
		goto done;

	rc = setsockopt(ctrl->fd, SOL_SOCKET, SO_SNDTIMEO, &dflt_tmo,
			sizeof(dflt_tmo));
	if (rc == -1) {
		rc = -errno;
		fprintf(stderr, "setsockopt(SO_SNDTIMEO) failed: %s\n",
			strerror(-rc));
		goto done;
	}

done:
	if (rc != 0)
		ctrl_close(ctrl);
	return rc;
}

/* Exchange and compare client/server program names and versions. Share client
 * command line options with server.
 */
static int ctrl_exchange_config(struct ctrl_connection *ctrl, const char *name,
				const char *ver, struct util_opts *opts)
{
	int rc;
	char peer_buf[MAX_CTRL_DATA_BYTES];
	size_t buf_len;
	int maj;
	int min;
	int peer_maj;
	int peer_min;
	bool vers_match;
	struct util_opts *peer_opts;

	if (!ctrl || !name || !ver || !opts)
		return -EINVAL;

	/* Verify names match */
	buf_len = strlen(name) + 1;
	rc = ctrl_exchange_data(ctrl, name, buf_len, peer_buf,
				sizeof(peer_buf));
	if (rc < 0)
		return rc;
	if (rc != buf_len || strcmp(name, peer_buf)) {
		fprintf(stderr,
			"Client and server program names do not match!\n");
		return -EINVAL;
	}

	/* Verify MAJ.MIN versions match */
	buf_len = strlen(ver) + 1;
	rc = ctrl_exchange_data(ctrl, ver, buf_len, peer_buf,
				sizeof(peer_buf));
	if (rc < 0)
		return rc;
	vers_match = false;
	rc = sscanf(ver, "%u.%u", &maj, &min);
	if (rc == 2) {
		rc = sscanf(peer_buf, "%u.%u", &peer_maj, &peer_min);
		if (rc == 2 && min == peer_min && maj == peer_maj)
			vers_match = true;
	}
	if (!vers_match) {
		fprintf(stderr,
			"Client and server program versions do not match!\n");
		fprintf(stderr, "%s: %s\n", name, ver);
		return -EINVAL;
	}

	/* Exchange config options */
	rc = ctrl_exchange_data(ctrl, opts, sizeof(*opts), peer_buf,
				sizeof(peer_buf));
	if (rc < 0)
		return rc;
	if ((size_t)rc != sizeof(*opts)) {
		fprintf(stderr, "Bad opts size (exp %zu got %d)\n",
			sizeof(*opts), rc);
		return -EINVAL;
	}

	peer_opts = (struct util_opts *)peer_buf;
	if (ctrl->is_server)
		memcpy(opts, peer_opts,
		       (sizeof(*opts) - sizeof(opts->loc_opts)
			- sizeof(opts->rmt_opts)));

	opts->rmt_opts = peer_opts->loc_opts;

	return 0;
}

/* Exchange client/server NIC fabric addresses */
static int ctrl_exchange_addrs(struct ctrl_connection *ctrl,
			       struct cxi_ep_addr *loc_addr,
			       struct cxi_ep_addr *rmt_addr)
{
	int rc;

	if (!ctrl || !loc_addr || !rmt_addr)
		return -EINVAL;

	rc = ctrl_exchange_data(ctrl, loc_addr, sizeof(*loc_addr), rmt_addr,
				sizeof(*rmt_addr));
	if (rc < 0)
		return rc;
	if (rc != sizeof(*rmt_addr)) {
		fprintf(stderr, "Received undersized remote address\n");
		return -EIO;
	}

	return 0;
}

/* Connect to peer and exchange data */
int ctrl_connect(struct ctrl_connection *ctrl, const char *name,
		 const char *version, struct util_opts *opts,
		 struct cxi_ep_addr *loc_addr, struct cxi_ep_addr *rmt_addr)
{
	int rc;

	if (!ctrl || !name || !version || !opts || !loc_addr || !rmt_addr)
		return -EINVAL;

	/* Connect to peer */
	rc = ctrl_init(ctrl);
	if (rc) {
		fprintf(stderr, "Failed to init control messaging: %s\n",
			strerror(-rc));
		return rc;
	}

	/* Verify peer and sync config */
	rc = ctrl_exchange_config(ctrl, name, version, opts);
	if (rc < 0) {
		fprintf(stderr, "Failed to exchange client/server config: %s\n",
			strerror(-rc));
		return rc;
	}

	/* Exchange addresses */
	rc = ctrl_exchange_addrs(ctrl, loc_addr, rmt_addr);
	if (rc < 0) {
		fprintf(stderr, "Failed to exchange client/server addrs: %s\n",
			strerror(-rc));
		return rc;
	}
	ctrl->is_loopback = loc_addr->nic == rmt_addr->nic;

	ctrl->connected = true;
	return 0;
}

/* Close the connection */
int ctrl_close(struct ctrl_connection *ctrl)
{
	int rc = 0;

	if (ctrl->fd != -1) {
		rc = close(ctrl->fd);
		ctrl->fd = -1;
	}
	ctrl->connected = false;
	return rc;
}

/* Helper function for ctrl_send */
static ssize_t __ctrl_send(int fd, const void *buf, size_t len)
{
	ssize_t bytes;

	bytes = send(fd, buf, len, 0);
	if (bytes < 0) {
		int rc = -errno;

		fprintf(stderr, "send() failed: %s\n", strerror(-rc));
		return rc;
	} else if (bytes == 0) {
		fprintf(stderr, "No data sent or remote connection closed\n");
		return -ECONNABORTED;
	} else if (bytes != len) {
		fprintf(stderr, "Bad send size (exp %zu got %zu)\n", len, bytes);
		return -EIO;
	}

	return bytes;
}

/* Encapsulate and send a control message
 * returns -errno on error, 0 on success
 */
static int ctrl_send(struct ctrl_connection *ctrl, const void *buf, size_t size)
{
	struct ctrl_msg msg = {};
	ssize_t bytes;

	if (!ctrl || (!buf && size > 0))
		return -EINVAL;

	msg.pyld_len = size;

	bytes = __ctrl_send(ctrl->fd, &msg, sizeof(msg));
	if (bytes < 0)
		return bytes;

	if (size) {
		bytes = __ctrl_send(ctrl->fd, buf, size);
		if (bytes < 0)
			return bytes;
	}

	return 0;
}

/* Helper function for ctrl_recv */
static ssize_t __ctrl_recv(int fd, void *buf, size_t len)
{
	ssize_t bytes;

	bytes = recv(fd, buf, len, 0);
	if (bytes < 0) {
		int rc = -errno;

		/* recv will return EAGAIN when a timeout is set */
		if (rc == -EAGAIN)
			rc = -ETIMEDOUT;

		fprintf(stderr, "recv() failed: %s\n", strerror(-rc));
		return rc;
	} else if (bytes == 0) {
		fprintf(stderr, "No data received or remote connection closed\n");
		return -ECONNABORTED;
	} else if (bytes != len) {
		fprintf(stderr, "Bad recv size (exp %zu got %zu)\n", len, bytes);
		return -EIO;
	}

	return bytes;
}

/* Attempt to receive and de-encapsulate a control message
 * returns -errno on error, number of bytes received on success
 */
static int ctrl_recv(struct ctrl_connection *ctrl, void *buf, size_t size,
		struct timeval *tmo_tv)
{
	int rc;
	struct ctrl_msg msg = {};
	ssize_t bytes;

	if (!ctrl || (!buf && size > 0))
		return -EINVAL;

	rc = ctrl_set_rcvtmo(ctrl, tmo_tv);
	if (rc)
		return rc;

	bytes = __ctrl_recv(ctrl->fd, &msg, sizeof(msg));
	if (bytes < 0)
		return bytes;

	if (msg.pyld_len) {
		if (msg.pyld_len > size) {
			fprintf(stderr, "Received message (%u) larger than buf (%zu)\n",
				msg.pyld_len, size);
			return -EMSGSIZE;
		}

		bytes = __ctrl_recv(ctrl->fd, buf, msg.pyld_len);
		if (bytes < 0)
			return bytes;
	}

	return msg.pyld_len;
}

/* Helper function for ctrl_exchange_data and ctrl_barrier */
static int __ctrl_exchange_data(struct ctrl_connection *ctrl,
		const void *cbuf, size_t clen, void *sbuf, size_t slen,
		struct timeval *tmo)
{
	int rc;
	int rc_recv;
	if (!ctrl->is_server) {
		rc = ctrl_send(ctrl, cbuf, clen);
		if (rc < 0)
			return rc;
	}

	rc_recv = ctrl_recv(ctrl, sbuf, slen, tmo);
	if (rc_recv < 0)
		return rc_recv;

	if (ctrl->is_server) {
		rc = ctrl_send(ctrl, cbuf, clen);
		if (rc < 0)
			return rc;
	}

	return rc_recv;
}

/* Exchange client/server data
 * returns -errno on error, number of bytes received on success
 */
int ctrl_exchange_data(struct ctrl_connection *ctrl, const void *client_buf,
		       size_t cbuf_size, void *server_buf, size_t sbuf_size)
{
	int rc;

	if (!ctrl || !client_buf || !server_buf)
		return -EINVAL;

	rc = __ctrl_exchange_data(ctrl, client_buf, cbuf_size,
			server_buf, sbuf_size, &dflt_tmo);

	return rc;
}

/* Sync client and server */
int ctrl_barrier(struct ctrl_connection *ctrl, uint64_t tmo_usec, char *label)
{
	int rc;
	struct timeval tmo_tv;

	if (!ctrl || !label)
		return -EINVAL;

	tmo_tv.tv_sec = tmo_usec / SEC2USEC;
	tmo_tv.tv_usec = tmo_usec % SEC2USEC;

	rc = __ctrl_exchange_data(ctrl, NULL, 0, NULL, 0, &tmo_tv);
	if (rc < 0)
		fprintf(stderr, "%s handshake failed: %s\n", label,
			strerror(-rc));

	return rc;
}
