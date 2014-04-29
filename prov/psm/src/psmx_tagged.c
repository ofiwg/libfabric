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

static ssize_t psmx_tagged_recv(fid_t fid, void *buf, size_t len,
				uint64_t tag, uint64_t ignore, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_recvv(fid_t fid, const void *iov, size_t len,
				 uint64_t tag, uint64_t ignore, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_recvfrom(fid_t fid, void *buf, size_t len,
				    const void *src_addr,
				    uint64_t tag, uint64_t ignore, void *context)
{
	struct psmx_fid_ep *fid_ep;
	psm_mq_req_t psm_req;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	err = psm_mq_irecv(fid_ep->domain->psm_mq, tag, ~ignore, 0, /* flags */
			   buf, len, context, &psm_req);
	if (err != PSM_OK)
		return psmx_errno(err);

	if (context && (fid_socket->flags & (FI_BUFFERED_RECV | FI_CANCEL)))
		((struct fi_context *)context)->internal[0] = psm_req;

	return 0;
}

static ssize_t psmx_tagged_recvmsg(fid_t fid, const struct fi_msg_tagged *msg,
				   uint64_t flags)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_send(fid_t fid, const void *buf, size_t len,
				uint64_t tag, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_sendv(fid_t fid, const void *iov, size_t len,
				 uint64_t tag, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_sendto(fid_t fid, const void *buf, size_t len,
				  const void *dest_addr,
				  uint64_t tag, void *context)
{
	struct psmx_fid_ep *fid_ep;
	int send_flag;
	psm_epaddr_t psm_epaddr;
	psm_mq_req_t psm_req;
	int err;
	int flags;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	psm_epaddr = (psm_epaddr_t) dest_addr;

	flags = fid_ep->flags;

	send_flag = (flags & FI_ACK) ? PSM_MQ_FLAG_SENDSYNC : 0;

	if (!(flags & FI_BLOCK)) {
		err = psm_mq_isend(fid_ep->domain->psm_mq, psm_epaddr,
				   send_flag, tag, buf, len, context, &psm_req);

		if (context && (flags & (FI_BUFFERED_RECV | FI_CANCEL)))
			((struct fi_context *)context)->internal[0] = NULL;
			 /* send cannot be canceled */
		return 0;
	} else {
		err = psm_mq_send(fid_ep->domain->psm_mq, psm_epaddr,
				  send_flag, tag, buf, len);
		if (err == PSM_OK)
			return len;
		else
			return psmx_errno(err);
	}
}

static ssize_t psmx_tagged_sendmsg(fid_t fid, const struct fi_msg_tagged *msg,
				   uint64_t flags)
{
	return -ENOSYS;
}

static ssize_t psmx_tagged_search(fid_t fid, uint64_t *tag, uint64_t ignore,
				  uint64_t flags, void *src_addr, 
				  size_t *src_addrlen, size_t *len,
				  void *context)
{
	struct psmx_fid_ep *fid_ep;
	psm_mq_status_t psm_status;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	err = psm_mq_iprobe(fid_ep->domain->psm_mq, *tag, ~ignore, &psm_status);
	switch (err) {
	case PSM_OK:
		*tag = psm_status.msg_tag;
		*len = psm_status.msg_length;
		/* FIXME: fill in src_addr and src_addrlen */
		return 1;

	case PSM_MQ_NO_COMPLETIONS:
		return -FI_ENOMSG;

	default:
		return psmx_errno(err);
	}
}

struct fi_ops_tagged psmx_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = psmx_tagged_recv,
	.recvv = psmx_tagged_recvv,
	.recvfrom = psmx_tagged_recvfrom,
	.recvmsg = psmx_tagged_recvmsg,
	.send = psmx_tagged_send,
	.sendv = psmx_tagged_sendv,
	.sendto = psmx_tagged_sendto,
	.sendmsg = psmx_tagged_sendmsg,
	.search = psmx_tagged_search,
};

