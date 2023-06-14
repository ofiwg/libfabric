/*
 * Copyright (c) 2013-2021 Intel Corporation. All rights reserved
 * (C) Copyright 	   Amazon.com, Inc. or its affiliates.
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "sm2.h"
#include "sm2_fifo.h"

void sm2_rma_handle_remote_error(struct sm2_ep *ep,
				 struct sm2_xfer_entry *xfer_entry,
				 struct sm2_sar_ctx *ctx)
{
	int ret = 0;
	if (ctx->op_flags & FI_DELIVERY_COMPLETE &&
	    !(ctx->status_flags & SM2_SAR_STATUS_COMPLETED)) {
		ret = sm2_write_err_comp(
			ep->util_ep.tx_cq, (void *) xfer_entry->hdr.context,
			xfer_entry->hdr.op_flags, 0, xfer_entry->hdr.cq_data,
			-FI_EREMOTEIO);
		if (!ret)
			ctx->status_flags |= SM2_SAR_STATUS_COMPLETED;
		else
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"Error in sm2_write_err_comp\n");
	}
	sm2_freestack_push(ep, xfer_entry);
	if (ctx->msgs_in_flight == 0)
		sm2_free_sar_ctx(ctx);
}

void sm2_rma_handle_local_error(struct sm2_ep *ep,
				struct sm2_xfer_entry *xfer_entry,
				struct sm2_sar_ctx *ctx, uint64_t err)
{
	int ret = 0;
	if (err == 0)
		return;
	ctx->status_flags |= SM2_SAR_ERROR_FLAG;

	if (!(ctx->status_flags & SM2_SAR_STATUS_COMPLETED)) {
		ret = sm2_write_err_comp(ep->util_ep.tx_cq,
					 (void *) xfer_entry->hdr.context,
					 xfer_entry->hdr.op_flags, 0,
					 xfer_entry->hdr.cq_data, err);
		if (!ret)
			ctx->status_flags |= SM2_SAR_STATUS_COMPLETED;
		else
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"Error in sm2_write_err_comp\n");
	}
	sm2_freestack_push(ep, xfer_entry);
	if (ctx->msgs_in_flight == 0)
		sm2_free_sar_ctx(ctx);
}

void sm2_fill_sar_ctx(struct sm2_ep *ep, const struct fi_msg_rma *msg,
		      uint32_t op, uint64_t op_flags, sm2_gid_t peer_gid,
		      struct sm2_sar_ctx *ctx)
{
	ctx->ep = ep;
	ctx->msg.context = msg->context;
	ctx->msg.rma_iov_count = msg->rma_iov_count;
	ctx->msg.iov_count = msg->iov_count;
	ctx->msg.addr = msg->addr;
	ctx->msg.data = msg->data;

	assert(msg->iov_count <= SM2_IOV_LIMIT);

	memset(ctx->msg.desc, 0, msg->iov_count * sizeof(*msg->desc));
	if (msg->desc) {
		memcpy(ctx->msg.desc, msg->desc,
		       msg->iov_count * sizeof(*msg->desc));
	}
	memcpy(ctx->msg.msg_iov, msg->msg_iov,
	       msg->iov_count * sizeof(*msg->msg_iov));
	memcpy(ctx->msg.rma_iov, msg->rma_iov,
	       msg->rma_iov_count * sizeof(*msg->rma_iov));

	ctx->bytes_sent = 0;
	ctx->bytes_acked = 0;
	ctx->bytes_total = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
	ctx->msgs_in_flight = 0;
	ctx->peer_gid = peer_gid;
	ctx->op = op;
	ctx->op_flags = op_flags;
	ctx->status_flags = 0;
}

ssize_t sm2_rma_cmd_fill_sar_xfer(struct sm2_xfer_entry *xfer_entry,
				  struct sm2_sar_ctx *ctx)
{
	ssize_t bytes_to_send, payload_used, rma_consumed;
	int ret;
	void *src, *dest;
	enum fi_hmem_iface iface;
	uint64_t device;
	struct sm2_rma_msg *msg;
	struct sm2_cmd_sar_rma_msg *cmd_rma;
	struct fi_rma_iov *cmd_iov;

	msg = &ctx->msg;

	xfer_entry->hdr.proto = sm2_proto_sar;
	sm2_generic_format(xfer_entry, ctx->ep->gid, ctx->op, 0, msg->data,
			   ctx->op_flags, msg->context);
	cmd_rma = (struct sm2_cmd_sar_rma_msg *) xfer_entry->user_data;
	memset(cmd_rma, 0, sizeof(*cmd_rma));
	cmd_rma->sar_hdr.proto_ctx = ctx;

	payload_used = 0;
	if (ctx->op == ofi_op_read_req) {
		cmd_rma->sar_hdr.request_offset = ctx->bytes_requested;
		ctx->bytes_requested += SM2_RMA_INJECT_SIZE;
		if (ctx->bytes_requested >= ctx->bytes_total) {
			ctx->bytes_requested = ctx->bytes_total;
			xfer_entry->hdr.proto_flags |=
				SM2_SAR_LAST_MESSAGE_FLAG;
		}
		memcpy(cmd_rma->rma_iov, msg->rma_iov,
		       sizeof(cmd_rma->rma_iov));
		ctx->msgs_in_flight++;

		return FI_SUCCESS;
	}

	while (payload_used < SM2_RMA_INJECT_SIZE &&
	       ctx->bytes_sent != ctx->bytes_total) {
		bytes_to_send = msg->msg_iov[0].iov_len;
		bytes_to_send =
			MIN(bytes_to_send, SM2_RMA_INJECT_SIZE - payload_used);

		src = (uint8_t *) msg->msg_iov[0].iov_base;
		sm2_get_iface_device(msg->desc[0], &iface, &device);
		dest = cmd_rma->user_data + payload_used;
		assert(payload_used < SM2_INJECT_SIZE);

		ret = ofi_copy_from_hmem(iface, device, dest, src,
					 bytes_to_send);
		if (ret)
			return ret;
		ofi_consume_iov_desc(msg->msg_iov, msg->desc, &msg->iov_count,
				     bytes_to_send);

		payload_used += bytes_to_send;
		ctx->bytes_sent += bytes_to_send;
	}

	rma_consumed = 0;
	cmd_iov = &cmd_rma->rma_iov[0];
	while (rma_consumed < payload_used) {
		cmd_iov->len =
			MIN(msg->rma_iov[0].len, payload_used - rma_consumed);
		cmd_iov->addr = msg->rma_iov[0].addr;
		cmd_iov->key = msg->rma_iov[0].key;
		ofi_consume_rma_iov(msg->rma_iov, &msg->rma_iov_count,
				    cmd_iov->len);
		rma_consumed += cmd_iov->len;
		cmd_iov++;
	}

	if (ctx->bytes_sent == ctx->bytes_total) {
		xfer_entry->hdr.proto_flags |= SM2_SAR_LAST_MESSAGE_FLAG;
	}
	ctx->msgs_in_flight++;

	return FI_SUCCESS;
}

/* takes rma command from application, and issues individual rma proto ops. */
ssize_t sm2_generic_rma(struct sm2_ep *ep, const struct fi_msg_rma *msg,
			uint32_t op, uint64_t op_flags)
{
	ssize_t ret = 0;
	sm2_gid_t peer_gid;
	struct sm2_sar_ctx *rma_ctx;
	struct sm2_xfer_entry *xfer_entry;

	assert(ofi_total_iov_len(msg->msg_iov, msg->iov_count) ==
	       ofi_total_rma_iov_len(msg->rma_iov, msg->rma_iov_count));

	assert(sizeof(rma_ctx->msg) < SM2_INJECT_SIZE);

	assert(op == ofi_op_read_req || op == ofi_op_write);

	ret = sm2_verify_peer(ep, msg->addr, &peer_gid);
	if (ret < 0)
		return -FI_EAGAIN;

	ofi_genlock_lock(&ep->util_ep.lock);

	ret = sm2_alloc_sar_ctx(ep, &rma_ctx);
	if (ret) {
		ret = -FI_EAGAIN;
		goto out;
	}
	sm2_fill_sar_ctx(ep, msg, op, op_flags, peer_gid, rma_ctx);

	do {
		ret = sm2_pop_xfer_entry(ep, &xfer_entry);
		if (ret) {
			if (rma_ctx->msgs_in_flight == 0) {
				sm2_free_sar_ctx(rma_ctx);
				ret = -FI_EAGAIN;
			} else {
				ret = FI_SUCCESS;
			}
			goto out;
		}
		ret = sm2_rma_cmd_fill_sar_xfer(xfer_entry, rma_ctx);
		if (!ret)
			sm2_fifo_write(ep, peer_gid, xfer_entry);
		else {
			sm2_rma_handle_local_error(ep, xfer_entry, rma_ctx,
						   ret);
			break;
		}
	} while (rma_ctx->msgs_in_flight < SM2_SAR_IN_FLIGHT_TARGET_RMA &&
		 rma_ctx->bytes_sent != rma_ctx->bytes_total);

	if (op == ofi_op_write && !(op_flags & FI_DELIVERY_COMPLETE) &&
	    rma_ctx->bytes_sent == rma_ctx->bytes_total &&
	    !(rma_ctx->status_flags & SM2_SAR_STATUS_COMPLETED)) {
		ret = sm2_complete_tx(ep, msg->context, op, op_flags);
		if (!ret)
			rma_ctx->status_flags |= SM2_SAR_STATUS_COMPLETED;
		else
			FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
		ret = FI_SUCCESS;
	}

out:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

/* ---- Start of fi_rma interfaces ----*/

static ssize_t sm2_read(struct fid_ep *ep_fid, void *buf, size_t len,
			void *desc, fi_addr_t src_addr, uint64_t addr,
			uint64_t key, void *context)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.desc = &desc;
	msg.context = context;
	msg.msg_iov = &msg_iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = 1;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_read_req, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_readv(struct fid_ep *ep_fid, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t src_addr,
			 uint64_t addr, uint64_t key, void *context)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.context = context;
	msg.desc = desc;
	msg.msg_iov = iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = count;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_read_req, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			   uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_rma(ep, msg, ofi_op_read_req,
			       flags | ep->util_ep.tx_msg_flags);
}

static ssize_t sm2_write(struct fid_ep *ep_fid, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, uint64_t addr,
			 uint64_t key, void *context)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.desc = &desc;
	msg.context = context;
	msg.msg_iov = &msg_iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = 1;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_write, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_writev(struct fid_ep *ep_fid, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.context = context;
	msg.desc = desc;
	msg.msg_iov = iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = count;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_write, sm2_ep_tx_flags(ep));
}

static ssize_t sm2_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			    uint64_t flags)
{
	struct sm2_ep *ep;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	return sm2_generic_rma(ep, msg, ofi_op_write,
			       flags | ep->util_ep.tx_msg_flags);
}

static ssize_t sm2_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.context = context;
	msg.desc = &desc;
	msg.data = data;
	msg.msg_iov = &msg_iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = 1;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_write,
			       FI_REMOTE_CQ_DATA | sm2_ep_tx_flags(ep));
}

static ssize_t sm2_rma_inject(struct fid_ep *ep_fid, const void *buf,
			      size_t len, fi_addr_t dest_addr, uint64_t addr,
			      uint64_t key)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	if (len > SM2_RMA_INJECT_SIZE)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &msg_iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = 1;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_write, FI_INJECT);
}

static ssize_t sm2_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				    size_t len, uint64_t data,
				    fi_addr_t dest_addr, uint64_t addr,
				    uint64_t key)
{
	struct sm2_ep *ep;
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	if (len > SM2_RMA_INJECT_SIZE)
		return -FI_EINVAL;

	ep = container_of(ep_fid, struct sm2_ep, util_ep.ep_fid.fid);

	memset(&msg, 0, sizeof(msg));
	msg.data = data;
	msg.msg_iov = &msg_iov;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.iov_count = 1;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return sm2_generic_rma(ep, &msg, ofi_op_write,
			       FI_REMOTE_CQ_DATA | FI_INJECT);
}

struct fi_ops_rma sm2_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = sm2_read,
	.readv = sm2_readv,
	.readmsg = sm2_readmsg,
	.write = sm2_write,
	.writev = sm2_writev,
	.writemsg = sm2_writemsg,
	.inject = sm2_rma_inject,
	.writedata = sm2_writedata,
	.injectdata = sm2_inject_writedata,
};
