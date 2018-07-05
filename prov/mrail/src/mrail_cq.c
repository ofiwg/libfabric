/*
 * Copyright (c) 2018 Intel Corporation, Inc.  All rights reserved.
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
 *	- Redistributions of source code must retain the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer.
 *
 *	- Redistributions in binary form must reproduce the above
 *	  copyright notice, this list of conditions and the following
 *	  disclaimer in the documentation and/or other materials
 *	  provided with the distribution.
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

#include <ofi_iov.h>

#include "mrail.h"

int mrail_cq_write_recv_comp(struct mrail_ep *mrail_ep, struct mrail_hdr *hdr,
			     mrail_cq_entry_t *comp, struct mrail_recv *recv)
{
	FI_DBG(&mrail_prov, FI_LOG_CQ, "writing recv completion: length: %zu "
	       "tag: 0x%" PRIx64 "\n", comp->len - sizeof(struct mrail_pkt),
	       hdr->tag);
	return ofi_cq_write(mrail_ep->util_ep.rx_cq, recv->context,
			   recv->comp_flags |
			   (comp->flags & FI_REMOTE_CQ_DATA),
			   comp->len - sizeof(struct mrail_pkt),
			   NULL, comp->data, hdr->tag);
}

int mrail_cq_process_buf_recv(mrail_cq_entry_t *comp, struct mrail_recv *recv)
{
	struct fi_recv_context *recv_ctx = comp->op_context;
	struct fi_msg msg = {
		.context = recv_ctx,
	};
	struct mrail_ep *mrail_ep;
	struct mrail_pkt *mrail_pkt;
	size_t size, len;
	int ret, retv = 0;

	if (comp->flags & FI_MORE) {
		msg.msg_iov	= recv->iov;
		msg.iov_count	= recv->count;
		msg.addr	= recv->addr;

		recv_ctx->context = recv;

		ret = fi_recvmsg(recv_ctx->ep, &msg, FI_CLAIM);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_CQ,
				"Unable to claim buffered recv\n");
			assert(0);
			// TODO write cq error entry
		}
		return ret;
	}

	mrail_ep = recv_ctx->ep->fid.context;
	mrail_pkt = (struct mrail_pkt *)comp->buf;

	len = comp->len - sizeof(*mrail_pkt);

	size = ofi_copy_to_iov(&recv->iov[1], recv->count - 1, 0,
			       mrail_pkt->data, len);

	if (size < len) {
		FI_WARN(&mrail_prov, FI_LOG_CQ, "Message truncated recv buf "
			"size: %zu message length: %zu\n", size, len);
		ret = ofi_cq_write_error_trunc(
			mrail_ep->util_ep.rx_cq, recv->context,
			recv->comp_flags | (comp->flags & FI_REMOTE_CQ_DATA),
			0, NULL, comp->data, mrail_pkt->hdr.tag, comp->len - size);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_CQ,
				"Unable to write truncation error to util cq\n");
			retv = ret;
		}
		goto out;
	}
	ret = mrail_cq_write_recv_comp(mrail_ep, &mrail_pkt->hdr, comp, recv);
	if (ret)
		retv = ret;
out:
	ret = fi_recvmsg(recv_ctx->ep, &msg, FI_DISCARD);
	if (ret) {
		FI_WARN(&mrail_prov, FI_LOG_CQ,
			"Unable to discard buffered recv\n");
		retv = ret;
	}
	mrail_push_recv(recv);
	return retv;
}

static int mrail_cq_process_comp_buf_recv(struct util_cq *cq,
					  mrail_cq_entry_t *comp)
{
	struct fi_recv_context *recv_ctx;
	struct mrail_ep *mrail_ep;
	struct mrail_hdr *hdr;
	struct mrail_recv *recv;
	int ret;

	if (comp->flags & FI_CLAIM) {
		recv = comp->op_context;
		assert(recv->hdr.version == MRAIL_HDR_VERSION);
		ret =  mrail_cq_write_recv_comp(recv->ep, &recv->hdr, comp,
						recv->context);
		mrail_push_recv(recv);
		return ret;
	}

	recv_ctx = comp->op_context;
	mrail_ep = recv_ctx->ep->fid.context;
	hdr = comp->buf;

	// TODO make rxm send buffered recv amount of data for large message
	assert(hdr->version == MRAIL_HDR_VERSION);

	// TODO match seq number
	fastlock_acquire(&mrail_ep->util_ep.lock);
	if (hdr->op == ofi_op_msg) {
		FI_DBG(&mrail_prov, FI_LOG_CQ, "Got MSG op\n");
		// TODO pass the right address
		recv = mrail_match_recv_handle_unexp(&mrail_ep->recv_queue, 0,
						     FI_ADDR_UNSPEC,
						     (char *)comp, sizeof(*comp),
						     NULL);
	} else {
		assert(hdr->op == ofi_op_tagged);
		FI_DBG(&mrail_prov, FI_LOG_CQ, "Got TAGGED op\n");
		recv = mrail_match_recv_handle_unexp(&mrail_ep->trecv_queue,
						     hdr->tag, FI_ADDR_UNSPEC,
						     (char *)comp, sizeof(*comp),
						     NULL);
	}
	fastlock_release(&mrail_ep->util_ep.lock);
	if (OFI_UNLIKELY(!recv))
		return 0;

	return mrail_cq_process_buf_recv(comp, recv);
}

static int mrail_cq_close(fid_t fid)
{
	struct mrail_cq *mrail_cq = container_of(fid, struct mrail_cq, util_cq.cq_fid.fid);
	int ret, retv = 0;

	ret = mrail_close_fids((struct fid **)mrail_cq->cqs,
			       mrail_cq->num_cqs);
	if (ret)
		retv = ret;
	free(mrail_cq->cqs);

	ret = ofi_cq_cleanup(&mrail_cq->util_cq);
	if (ret)
		retv = ret;

	free(mrail_cq);
	return retv;
}

static struct fi_ops mrail_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = mrail_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq mrail_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = ofi_cq_sread,
	.sreadfrom = ofi_cq_sreadfrom,
	.signal = ofi_cq_signal,
	// TODO define cq strerror, may need to pass rail index
	// in err_data
	.strerror = fi_no_cq_strerror,
};

static void mrail_cq_progress(struct util_cq *cq)
{
	struct mrail_cq *mrail_cq;
	mrail_cq_entry_t comp;
	size_t i;
	int ret;

	mrail_cq = container_of(cq, struct mrail_cq, util_cq);

	for (i = 0; i < mrail_cq->num_cqs; i++) {
		ret = fi_cq_read(mrail_cq->cqs[i], &comp, 1);
		if (ret == -FI_EAGAIN || !ret)
			continue;
		if (ret < 0) {
			FI_WARN(&mrail_prov, FI_LOG_CQ,
				"Unable to read rail completion\n");
			goto err;
		}
		// TODO handle variable length message
		if (comp.flags & FI_RECV) {
			ret = mrail_cq->process_comp(cq, &comp);
			if (ret)
				goto err;
		} else {
			assert(comp.flags & (FI_SEND | FI_WRITE | FI_READ |
					     FI_REMOTE_READ | FI_REMOTE_WRITE));
			ret = ofi_cq_write(cq, comp.op_context, comp.flags,
					   0, NULL, 0, 0);
			if (ret) {
				FI_WARN(&mrail_prov, FI_LOG_CQ,
					"Unable to write to util cq\n");
				goto err;
			}
		}
	}
	return;
err:
	// TODO write error to cq
	assert(0);
}

int mrail_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		   struct fid_cq **cq_fid, void *context)
{
	struct mrail_domain *mrail_domain;
	struct mrail_cq *mrail_cq;
	struct fi_cq_attr rail_cq_attr = {
		.wait_obj = FI_WAIT_NONE,
		.format = MRAIL_RAIL_CQ_FORMAT,
		.size = attr->size,
	};
	size_t i;
	int ret;

	mrail_cq = calloc(1, sizeof(*mrail_cq));
	if (!mrail_cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&mrail_prov, domain, attr, &mrail_cq->util_cq,
			  &mrail_cq_progress, context);
	if (ret) {
		free(mrail_cq);
		return ret;
	}

	mrail_domain = container_of(domain, struct mrail_domain,
				    util_domain.domain_fid);

	mrail_cq->cqs = calloc(mrail_domain->num_domains,
			       sizeof(*mrail_cq->cqs));
	if (!mrail_cq->cqs)
		goto err;

	mrail_cq->num_cqs = mrail_domain->num_domains;

	for (i = 0; i < mrail_cq->num_cqs; i++) {
		ret = fi_cq_open(mrail_domain->domains[i], &rail_cq_attr,
				 &mrail_cq->cqs[i], NULL);
		if (ret) {
			FI_WARN(&mrail_prov, FI_LOG_EP_CTRL,
				"Unable to open rail CQ\n");
			goto err;
		}
	}

	// TODO add regular process comp when FI_BUFFERED_RECV not set
	mrail_cq->process_comp = mrail_cq_process_comp_buf_recv;

	*cq_fid = &mrail_cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &mrail_cq_fi_ops;
	(*cq_fid)->ops = &mrail_cq_ops;

	return 0;
err:
	mrail_cq_close(&mrail_cq->util_cq.cq_fid.fid);
	return ret;
}
