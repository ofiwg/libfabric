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
#include <prov.h>
#include <sys/types.h>
#include <fi_util.h>
#include "tcpx.h"

static ssize_t tcpx_comm_recv_socket(SOCKET conn_fd, void *buf, size_t len)
{
	ssize_t ret;

	ret = ofi_recv_socket(conn_fd, buf, len, 0);
	if (ret > 0) {
		FI_DBG(&tcpx_prov, FI_LOG_EP_DATA,
		       "read from network: %lu\n", ret);
	} else {
		if (ret < 0) {
			FI_DBG(&tcpx_prov, FI_LOG_EP_DATA,
			       "read %s\n", strerror(ofi_sockerr()));
			ret = 0;
		} else {
			FI_DBG(&tcpx_prov, FI_LOG_EP_DATA," socket closed\n");
			return ret;
		}
	}
	return ret;
}

static void tcpx_comm_recv_buffer(struct tcpx_pe_entry *pe_entry)
{
	int ret;
	size_t max_read, avail;

	avail = ofi_rbavail(&pe_entry->comm_buf);
	assert(avail == pe_entry->comm_buf.size);
	pe_entry->comm_buf.rcnt =
		pe_entry->comm_buf.wcnt =
		pe_entry->comm_buf.wpos = 0;

	max_read = pe_entry->data_len - pe_entry->done_len;
	ret = tcpx_comm_recv_socket(pe_entry->ep->conn_fd, (char *) pe_entry->comm_buf.buf,
				    MIN(max_read, avail));
	pe_entry->comm_buf.wpos += ret;
	ofi_rbcommit(&pe_entry->comm_buf);
}

ssize_t tcpx_comm_recv(struct tcpx_pe_entry *pe_entry, void *buf, size_t len)
{
	ssize_t read_len;
	if (ofi_rbempty(&pe_entry->comm_buf)) {
		if (len <= pe_entry->cache_sz) {
			tcpx_comm_recv_buffer(pe_entry);
		} else {
			return tcpx_comm_recv_socket(pe_entry->ep->conn_fd, buf, len);
		}
	}

	read_len = MIN(len, ofi_rbused(&pe_entry->comm_buf));
	ofi_rbread(&pe_entry->comm_buf, buf, read_len);
	FI_DBG(&tcpx_prov, FI_LOG_EP_DATA, "read from buffer: %lu\n", read_len);
	return read_len;
}

static ssize_t tcpx_comm_send_socket(SOCKET conn_fd, const void *buf, size_t len)
{
	ssize_t ret;

	ret = ofi_send_socket(conn_fd, buf, len, MSG_NOSIGNAL);
	if (ret >= 0) {
		FI_DBG(&tcpx_prov, FI_LOG_EP_DATA, "wrote to network: %lu\n", ret);
		return ret;
	}

	if (OFI_SOCK_TRY_SND_RCV_AGAIN(ofi_sockerr())) {
		ret = 0;
	} else {
		FI_DBG(&tcpx_prov, FI_LOG_EP_DATA,
		       "write error: %s\n", strerror(ofi_sockerr()));
	}
	return ret;
}


ssize_t tcpx_comm_flush(struct tcpx_pe_entry *pe_entry)
{
	ssize_t ret1, ret2 = 0;
	size_t endlen, len, xfer_len;

	len = ofi_rbused(&pe_entry->comm_buf);
	endlen = pe_entry->comm_buf.size -
		(pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask);

	xfer_len = MIN(len, endlen);
	ret1 = tcpx_comm_send_socket(pe_entry->ep->conn_fd, (char*)pe_entry->comm_buf.buf +
				     (pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask),
				     xfer_len);
	if (ret1 > 0)
		pe_entry->comm_buf.rcnt += ret1;

	if (ret1 == xfer_len && xfer_len < len) {
		ret2 = tcpx_comm_send_socket(pe_entry->ep->conn_fd, (char*)pe_entry->comm_buf.buf +
					     (pe_entry->comm_buf.rcnt & pe_entry->comm_buf.size_mask),
					     len - xfer_len);
		if (ret2 > 0)
			pe_entry->comm_buf.rcnt += ret2;
		else
			ret2 = 0;
	}

	return (ret1 > 0) ? ret1 + ret2 : 0;
}

ssize_t tcpx_comm_send(struct tcpx_pe_entry *pe_entry,
		       const void *buf, size_t len)
{
	ssize_t ret, used;

	if (len > pe_entry->cache_sz) {
		used = ofi_rbused(&pe_entry->comm_buf);
		if (used == tcpx_comm_flush(pe_entry)) {
			return tcpx_comm_send_socket(pe_entry->ep->conn_fd,
						     buf, len);
		} else {
			return 0;
		}
	}

	if (ofi_rbavail(&pe_entry->comm_buf) < len) {
		ret = tcpx_comm_flush(pe_entry);
		if (ret <= 0)
			return 0;
	}

	ret = MIN(ofi_rbavail(&pe_entry->comm_buf), len);
	ofi_rbwrite(&pe_entry->comm_buf, buf, ret);
	ofi_rbcommit(&pe_entry->comm_buf);
	FI_DBG(&tcpx_prov, FI_LOG_EP_DATA, "buffered %lu\n", ret);
	return ret;
}
