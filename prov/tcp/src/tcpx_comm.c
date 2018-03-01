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
#include <ofi_prov.h>
#include <sys/types.h>
#include <ofi_util.h>
#include <ofi_iov.h>
#include "tcpx.h"

int tcpx_send_msg(struct tcpx_pe_entry *pe_entry)
{
	ssize_t bytes_sent;

	bytes_sent = ofi_writev_socket(pe_entry->ep->conn_fd,
				       pe_entry->msg_data.iov,
				       pe_entry->msg_data.iov_cnt);
	if (bytes_sent < 0)
		return -errno;

	if (pe_entry->done_len < ntohll(pe_entry->msg_hdr.size)) {
		ofi_consume_iov(pe_entry->msg_data.iov,
				&pe_entry->msg_data.iov_cnt,
				bytes_sent);
	}

	pe_entry->done_len += bytes_sent;
	return FI_SUCCESS;
}

static int tcpx_recv_msg_hdr(struct tcpx_pe_entry *pe_entry)
{
	ssize_t bytes_recvd;
	void *rem_hdr_buf;
	size_t rem_hdr_len;

	rem_hdr_buf = (uint8_t *)&pe_entry->msg_hdr + pe_entry->done_len;
	rem_hdr_len = sizeof(pe_entry->msg_hdr) - pe_entry->done_len;

	bytes_recvd = ofi_recv_socket(pe_entry->ep->conn_fd,
				      rem_hdr_buf, rem_hdr_len, 0);
	if (bytes_recvd <= 0)
		return (bytes_recvd)? -errno: -FI_ENOTCONN;

	pe_entry->done_len += bytes_recvd;

	if (pe_entry->done_len < sizeof(pe_entry->msg_hdr))
		return -FI_EAGAIN;

	pe_entry->msg_hdr.op_data = TCPX_OP_MSG_RECV;
	return ofi_truncate_iov(pe_entry->msg_data.iov,
				&pe_entry->msg_data.iov_cnt,
				(ntohll(pe_entry->msg_hdr.size) -
				 sizeof(pe_entry->msg_hdr)));
}

int tcpx_recv_msg(struct tcpx_pe_entry *pe_entry)
{
	ssize_t bytes_recvd;
	int ret;

	if (pe_entry->done_len < sizeof(pe_entry->msg_hdr)) {
		ret = tcpx_recv_msg_hdr(pe_entry);
		if (ret)
			return ret;
	}

	bytes_recvd = ofi_readv_socket(pe_entry->ep->conn_fd,
				       pe_entry->msg_data.iov,
				       pe_entry->msg_data.iov_cnt);
	if (bytes_recvd <= 0)
		return (bytes_recvd)? -errno: -FI_ENOTCONN;


	if (pe_entry->done_len < ntohll(pe_entry->msg_hdr.size)) {
		ofi_consume_iov(pe_entry->msg_data.iov,
				&pe_entry->msg_data.iov_cnt,
				bytes_recvd);
	}

	pe_entry->done_len += bytes_recvd;
	return FI_SUCCESS;
}
