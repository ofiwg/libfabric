/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "psmx.h"

static ssize_t psmx_recv(fid_t fid, void *buf, size_t len, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvmem(fid_t fid, void *buf, size_t len,
				uint64_t mem_desc, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvv(fid_t fid, const void *iov, size_t count,
				void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvfrom(fid_t fid, void *buf, size_t len,
				const void *src_addr, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvmemfrom(fid_t fid, void *buf, size_t len,
				uint64_t mem_desc, const void *src_addr,
				void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvmsg(fid_t fid, const struct fi_msg *msg,
				uint64_t flags)
{
	return -ENOSYS;
}

static ssize_t psmx_send(fid_t fid, const void *buf, size_t len,
				void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendmem(fid_t fid, const void *buf, size_t len,
				uint64_t mem_desc, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendv(fid_t fid, const void *iov, size_t count,
				void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendto(fid_t fid, const void *buf, size_t len,
				  const void *dest_addr, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendmemto(fid_t fid, const void *buf, size_t len,
				uint64_t mem_desc, const void *dest_addr,
				void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendmsg(fid_t fid, const struct fi_msg *msg,
				uint64_t flags)
{
	return -ENOSYS;
}

struct fi_ops_msg psmx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = psmx_recv,
	.recvmem = psmx_recvmem,
	.recvv = psmx_recvv,
	.recvfrom = psmx_recvfrom,
	.recvmemfrom = psmx_recvmemfrom,
	.recvmsg = psmx_recvmsg,
	.send = psmx_send,
	.sendmem = psmx_sendmem,
	.sendv = psmx_sendv,
	.sendto = psmx_sendto,
	.sendmemto = psmx_sendmemto,
	.sendmsg = psmx_sendmsg,
};

