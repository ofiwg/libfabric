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

#include <errno.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "sock.h"
#include "sock_util.h"

#define SOCK_LOG_DBG(...) _SOCK_LOG_DBG(FI_LOG_EP_DATA, __VA_ARGS__)
#define SOCK_LOG_ERROR(...) _SOCK_LOG_ERROR(FI_LOG_EP_DATA, __VA_ARGS__)

ssize_t sock_comm_send(struct sock_conn *conn, const void *buf,
					size_t len)
{
	ssize_t ret;

	ret = write(conn->sock_fd, buf, len);
	if (ret < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			ret = 0;
		} else {
			SOCK_LOG_DBG("write %s\n", strerror(errno));
		}
	}
	SOCK_LOG_DBG("wrote to network: %lu\n", ret);
	return ret;
}

ssize_t sock_comm_recv(struct sock_conn *conn, void *buf,
					size_t len)
{
	ssize_t ret;

	ret = recv(conn->sock_fd, buf, len, 0);
	if (ret == 0) {
		conn->disconnected = 1;
		SOCK_LOG_DBG("Disconnected: %s:%d\n", inet_ntoa(conn->addr.sin_addr),
                               ntohs(conn->addr.sin_port));

		return ret;
	}

	if (ret < 0) {
		SOCK_LOG_DBG("read %s\n", strerror(errno));
		ret = 0;
	}

	if (ret > 0)
		SOCK_LOG_DBG("read from network: %lu\n", ret);
	return ret;
}

ssize_t sock_comm_peek(struct sock_conn *conn, void *buf, size_t len)
{
	ssize_t ret;
	ret = recv(conn->sock_fd, buf, len, MSG_PEEK);
	if (ret == 0) {
		conn->disconnected = 1;
		SOCK_LOG_ERROR("Disconnected\n");
		return ret;
	}

	if (ret < 0) {
		SOCK_LOG_DBG("peek %s\n", strerror(errno));
		ret = 0;
	}

	if (ret > 0)
		SOCK_LOG_DBG("peek from network: %lu\n", ret);
	return ret;
}

ssize_t sock_comm_discard(struct sock_conn *conn, size_t len)
{
	void *buf;
	int ret;

	buf = malloc(len);
	if (!buf)
		return 0;

	ret = sock_comm_recv(conn, buf, len);
	free(buf);
	return ret;
}
