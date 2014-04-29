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

static inline ssize_t _psmx_recvfrom(fid_t fid, void *buf, size_t len,
					const void *src_addr, void *context,
					uint64_t flags)
{
	struct psmx_fid_ep *fid_ep;
	psm_mq_req_t psm_req;
	uint64_t psm_tag, psm_tagsel;
	struct fi_context *fi_context;
	int err;
	int recv_flag = 0;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	if (src_addr) {
		psm_tag = ((uint64_t)(uintptr_t)psm_epaddr_getctxt((void *)src_addr))
				| PSMX_MSG_BIT;
		psm_tagsel = -1ULL;
	}
	else {
		psm_tag = PSMX_MSG_BIT;
		psm_tagsel = PSMX_MSG_BIT;
	}

	if (fid_ep->use_fi_context) {
		if (!context)
			return -EINVAL;

		fi_context = context;
		fi_context->internal[1] =
			((fid_ep->completion_mask | (flags & FI_EVENT)) == PSMX_COMP_ON) ?
			0 : PSMX_NOCOMP_CONTEXT;
	}
	else {
		fi_context = NULL;
	}

	err = psm_mq_irecv(fid_ep->domain->psm_mq,
			   psm_tag, psm_tagsel, recv_flag,
			   buf, len, (void *)fi_context, &psm_req);
	if (err != PSM_OK)
		return psmx_errno(err);

	if (fi_context)
		fi_context->internal[0] = psm_req;

	return 0;
}

static ssize_t psmx_recvfrom(fid_t fid, void *buf, size_t len,
				const void *src_addr, void *context)
{
	return _psmx_recvfrom(fid, buf, len, src_addr, context, 0);
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
	struct iovec *iov;

	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!msg || msg->iov_count != 1)
		return -EINVAL;

	/* FIXME: check iov format */
	iov = (struct iovec *)msg->msg_iov;
	return _psmx_recvfrom(fid, iov[0].iov_base, iov[0].iov_len,
					msg->addr, msg->context, flags);
}

static ssize_t psmx_recv(fid_t fid, void *buf, size_t len, void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	if (fid_ep->connected)
		return psmx_recvfrom(fid, buf, len, fid_ep->peer_psm_epaddr, context);
	else
		return psmx_recvfrom(fid, buf, len, NULL, context);
}

static ssize_t psmx_recvmem(fid_t fid, void *buf, size_t len,
				uint64_t mem_desc, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_recvv(fid_t fid, const void *iov, size_t count,
				void *context)
{
	struct iovec *iov0;

	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!iov || count != 1)
		return -EINVAL;

	/* FIXME: check iov format */
	iov0 = (struct iovec *)iov;
	return psmx_recv(fid, iov0->iov_base, iov0->iov_len, context);
}

static inline ssize_t _psmx_sendto(fid_t fid, const void *buf, size_t len,
				const void *dest_addr, void *context,
				uint64_t flags)
{
	struct psmx_fid_ep *fid_ep;
	int send_flag = 0;
	psm_epaddr_t psm_epaddr;
	psm_mq_req_t psm_req;
	uint64_t psm_tag;
	struct fi_context * fi_context;
	int err;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	psm_epaddr = (psm_epaddr_t) dest_addr;
	psm_tag = fid_ep->domain->psm_epid | PSMX_MSG_BIT;

	if ((fid_ep->flags | flags) & FI_BLOCK) {
		err = psm_mq_send(fid_ep->domain->psm_mq, psm_epaddr,
				  send_flag, psm_tag, buf, len);
		if (err == PSM_OK)
			return len;
		else
			return psmx_errno(err);
	}

	if (fid_ep->use_fi_context) {
		if (!context)
			return -EINVAL;

		fi_context = context;
		fi_context->internal[1] =
			((fid_ep->completion_mask | (flags & FI_EVENT)) == PSMX_COMP_ON) ?
			0 : PSMX_NOCOMP_CONTEXT;
	}
	else {
		fi_context = NULL;
	}

	err = psm_mq_isend(fid_ep->domain->psm_mq, psm_epaddr, send_flag,
				psm_tag, buf, len, (void *)fi_context, &psm_req);

	if (fi_context)
		fi_context->internal[0] = psm_req;

	return 0;
}

static ssize_t psmx_sendto(fid_t fid, const void *buf, size_t len,
				  const void *dest_addr, void *context)
{
	return _psmx_sendto(fid, buf, len, dest_addr, context, 0);
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
	struct iovec *iov;

	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!msg || msg->iov_count != 1)
		return -EINVAL;

	/* FIXME: check iov format */
	iov = (struct iovec *)msg->msg_iov;
	return _psmx_sendto(fid, iov[0].iov_base, iov[0].iov_len,
					msg->addr, msg->context, flags);
}

static ssize_t psmx_send(fid_t fid, const void *buf, size_t len,
				void *context)
{
	struct psmx_fid_ep *fid_ep;

	fid_ep = container_of(fid, struct psmx_fid_ep, ep.fid);
	assert(fid_ep->domain);

	if (!fid_ep->connected)
		return -ENOTCONN;

	return psmx_sendto(fid, buf, len, fid_ep->peer_psm_epaddr, context);
}

static ssize_t psmx_sendmem(fid_t fid, const void *buf, size_t len,
				uint64_t mem_desc, void *context)
{
	return -ENOSYS;
}

static ssize_t psmx_sendv(fid_t fid, const void *iov, size_t count,
				void *context)
{
	struct iovec *iov0;

	/* FIXME: allow iov_count == 0? */
	/* FIXME: allow iov_count > 1 */
	if (!iov || count != 1)
		return -EINVAL;

	/* FIXME: check iov format */
	iov0 = (struct iovec *)iov;
	return psmx_send(fid, iov0->iov_base, iov0->iov_len, context);
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

