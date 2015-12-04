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

static ssize_t sock_comm_send_socket(struct sock_conn *conn,
				     const void *buf, size_t len)
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

ssize_t sock_comm_flush(struct sock_pe_entry *pe_entry)
{
	ssize_t ret1, ret2 = 0;
	size_t endlen, len, xfer_len;

	len = rbused(&pe_entry->comm_buf);
	endlen = pe_entry->comm_buf.size -
		(pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask);

	xfer_len = MIN(len, endlen);
	ret1 = sock_comm_send_socket(pe_entry->conn, pe_entry->comm_buf.buf +
				     (pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask),
				     xfer_len);
	if (ret1 > 0)
		pe_entry->comm_buf.rcnt += ret1;

	if (ret1 == xfer_len && xfer_len < len) {
		ret2 = sock_comm_send_socket(pe_entry->conn, pe_entry->comm_buf.buf +
					     (pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask),
					     len - xfer_len);
		if (ret2 > 0)
			pe_entry->comm_buf.rcnt += ret2;
		else
			ret2 = 0;
	}

	return (ret1 > 0) ? ret1 + ret2 : 0;
}

ssize_t sock_comm_send(struct sock_pe_entry *pe_entry,
		       const void *buf, size_t len)
{
	ssize_t ret, used;

	if (len > SOCK_PE_COMM_BUFF_SZ) {
		used = rbused(&pe_entry->comm_buf);
		if (used == sock_comm_flush(pe_entry)) {
			return sock_comm_send_socket(pe_entry->conn, buf, len);
		} else {
			return 0;
		}
	}

	if (rbavail(&pe_entry->comm_buf) < len) {
		ret = sock_comm_flush(pe_entry);
		if (ret <= 0)
			return 0;
	}

	ret = MIN(rbavail(&pe_entry->comm_buf), len);
	rbwrite(&pe_entry->comm_buf, buf, ret);
	rbcommit(&pe_entry->comm_buf);
	SOCK_LOG_DBG("buffered %lu\n", ret);
	return ret;
}

int sock_comm_tx_done(struct sock_pe_entry *pe_entry)
{
	return rbempty(&pe_entry->comm_buf);
}

static ssize_t sock_comm_recv_socket(struct sock_conn *conn,
			      void *buf, size_t len)
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

static void sock_comm_recv_buffer(struct sock_pe_entry *pe_entry)
{
	int ret;
	size_t max_read, avail;

	avail = rbavail(&pe_entry->comm_buf);
	assert(avail == pe_entry->comm_buf.size);
	pe_entry->comm_buf.rcnt = 
		pe_entry->comm_buf.wcnt = 
		pe_entry->comm_buf.wpos = 0;

	max_read = pe_entry->total_len - pe_entry->done_len;
	ret = sock_comm_recv_socket(pe_entry->conn, (char *) pe_entry->comm_buf.buf,
				    MIN(max_read, avail));
	pe_entry->comm_buf.wpos += ret;
	rbcommit(&pe_entry->comm_buf);
}

ssize_t sock_comm_recv(struct sock_pe_entry *pe_entry, void *buf, size_t len)
{
	ssize_t read_len;
	if (rbempty(&pe_entry->comm_buf)) {
		if (len <= SOCK_PE_COMM_BUFF_SZ) {
			sock_comm_recv_buffer(pe_entry);
		} else {
			return sock_comm_recv_socket(pe_entry->conn, buf, len);
		}
	}

	read_len = MIN(len, rbused(&pe_entry->comm_buf));
	rbread(&pe_entry->comm_buf, buf, read_len);
	SOCK_LOG_DBG("read from buffer: %lu\n", read_len);
	return read_len;
}

ssize_t sock_comm_peek(struct sock_conn *conn, void *buf, size_t len)
{
	ssize_t ret;
	ret = recv(conn->sock_fd, buf, len, MSG_PEEK);
	if (ret == 0) {
		conn->disconnected = 1;
		SOCK_LOG_DBG("Disconnected\n");
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

ssize_t sock_comm_discard(struct sock_pe_entry *pe_entry, size_t len)
{
	void *buf;
	int ret;

	buf = malloc(len);
	if (!buf)
		return 0;

	ret = sock_comm_recv(pe_entry, buf, len);
	free(buf);
	return ret;
}
