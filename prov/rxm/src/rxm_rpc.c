/*
 * Copyright (c) Intel Corporation. All rights reserved.
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

#include <inttypes.h>
#include <math.h>

#include <rdma/fabric.h>
#include <rdma/fi_collective.h>
#include "ofi.h"
#include <ofi_util.h>

#include "rxm.h"


static ofi_atomic64_t current_rpc;

void rxm_rpc_init(void)
{
	ofi_atomic_initialize64(&current_rpc, 0);
}

static inline uint64_t rxm_next_rpc_tag(void)
{
	return ofi_atomic_inc64(&current_rpc) | RXM_RPC_TAG_FLAG;
}

static ssize_t
rxm_rpc_req(struct fid_ep *ep_fid, const void *req_buf, size_t req_len,
	    void *req_desc, void *resp_buf, size_t resp_len,
	    void *resp_desc, fi_addr_t dest_addr, int timeout,
	    void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	struct iovec req_iov = {
		.iov_base = (void *) req_buf,
		.iov_len = req_len,
	};
	struct iovec resp_iov = {
		.iov_base = resp_buf,
		.iov_len = resp_len,
	};
	uint64_t rpc_tag;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	/* the srx lock is the same lock as rxm_ep->util_ep.lock */
	rpc_tag = rxm_next_rpc_tag();
	ret = util_srx_generic_trecv_no_lock(
				&rxm_ep->srx->ep_fid, &resp_iov, &resp_desc,
				1, dest_addr, context, rpc_tag, 0,
				rxm_ep->util_ep.rx_op_flags | FI_RPC);
	if (ret)
		goto unlock;

	/* TODO: pass the timeout to the send */
	ret = rxm_send_common(rxm_ep, rxm_conn, &req_iov, &req_desc, 1, context,
			      0, rxm_ep->util_ep.tx_op_flags | FI_RPC,
			      rpc_tag, ofi_op_msg);

	if (ret)
		(void) fi_cancel(&rxm_ep->srx->ep_fid.fid, (void *)rpc_tag);

unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_reqv(struct fid_ep *ep_fid, const struct iovec *req_iov,
	     void **req_desc, size_t req_count, struct iovec *resp_iov,
	     void **resp_desc, size_t resp_count, fi_addr_t dest_addr,
	     int timeout, void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	uint64_t rpc_tag;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	/* the srx lock is the same lock as rxm_ep->util_ep.lock */
	rpc_tag = rxm_next_rpc_tag();
	ret = util_srx_generic_trecv_no_lock(
				&rxm_ep->srx->ep_fid, resp_iov, resp_desc,
				resp_count, dest_addr, context, rpc_tag, 0,
				rxm_ep->util_ep.rx_op_flags | FI_RPC);
	if (ret)
		goto unlock;

	/* TODO: pass the timeout to the send */
	ret = rxm_send_common(rxm_ep, rxm_conn, req_iov, req_desc, req_count,
			      context, 0,
			      rxm_ep->util_ep.tx_op_flags | FI_RPC,
			      rpc_tag, ofi_op_msg);
	if (ret)
		(void) fi_cancel(&rxm_ep->srx->ep_fid.fid, (void *)rpc_tag);

unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_reqdata(struct fid_ep *ep_fid, const void *req_buf, size_t req_len,
		void *req_desc, void *resp_buf, size_t resp_len,
		void *resp_desc, uint64_t data, fi_addr_t dest_addr,
		int timeout, void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	struct iovec req_iov = {
		.iov_base = (void *) req_buf,
		.iov_len = req_len,
	};
	struct iovec resp_iov = {
		.iov_base = resp_buf,
		.iov_len = resp_len,
	};
	uint64_t rpc_tag;
	uint64_t flags;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	/* the srx lock is the same lock as rxm_ep->util_ep.lock */
	rpc_tag = rxm_next_rpc_tag();
	ret = util_srx_generic_trecv_no_lock(
				&rxm_ep->srx->ep_fid, &resp_iov, &resp_desc,
				1, dest_addr, context, rpc_tag, 0,
				rxm_ep->util_ep.rx_op_flags | FI_RPC);
	if (ret)
		goto unlock;

	flags = rxm_ep->util_ep.tx_op_flags | FI_RPC | FI_REMOTE_CQ_DATA;

	/* TODO: pass the timeout to the send */
	ret = rxm_send_common(rxm_ep, rxm_conn, &req_iov, &req_desc, 1, context,
			      data, flags, rpc_tag, ofi_op_msg);

	if (ret)
		(void) fi_cancel(&rxm_ep->srx->ep_fid.fid, (void *)rpc_tag);

unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_reqmsg(struct fid_ep *ep_fid, const struct fi_msg_rpc *msg,
	       uint64_t flags)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	uint64_t rpc_tag;
	ssize_t ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, msg->addr, &rxm_conn);
	if (ret)
		goto unlock;

	/* the srx lock is the same lock as rxm_ep->util_ep.lock */
	rpc_tag = rxm_next_rpc_tag();
	ret = util_srx_generic_trecv_no_lock(
				&rxm_ep->srx->ep_fid, msg->resp_iov,
				msg->resp_desc, msg->resp_iov_count,
				msg->addr, msg->context, rpc_tag, 0,
				rxm_ep->util_ep.rx_op_flags | FI_RPC);
	if (ret)
		goto unlock;

	ret = rxm_send_common(rxm_ep, rxm_conn, msg->req_iov, msg->req_desc,
			      msg->req_iov_count, msg->context, msg->timeout,
			      flags | rxm_ep->util_ep.tx_msg_flags | FI_RPC,
			      rpc_tag, ofi_op_msg);
	if (ret)
		(void) fi_cancel(&rxm_ep->srx->ep_fid.fid, (void *)rpc_tag);

unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_resp(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
	     fi_addr_t dest_addr, uint64_t rpc_id, void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	struct iovec iov = {
		.iov_base = (void *) buf,
		.iov_len = len,
	};
	ssize_t ret;

	assert(rpc_id & RXM_RPC_TAG_FLAG);

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	ret = rxm_send_common(rxm_ep, rxm_conn, &iov, &desc, 1, context,
			      0, rxm_ep->util_ep.tx_op_flags | FI_RPC, rpc_id,
			      ofi_op_tagged);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_respv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	      size_t count, fi_addr_t dest_addr, uint64_t rpc_id, void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	ssize_t ret;

	assert(rpc_id & RXM_RPC_TAG_FLAG);

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	ret = rxm_send_common(rxm_ep, rxm_conn, iov, desc, count, context,
			      0, rxm_ep->util_ep.tx_op_flags | FI_RPC, rpc_id,
			      ofi_op_tagged);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_respdata(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		 uint64_t data, fi_addr_t dest_addr, uint64_t rpc_id,
		 void *context)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	struct iovec iov = {
		.iov_base = (void *) buf,
		.iov_len = len,
	};
	uint64_t flags;
	ssize_t ret;

	assert(rpc_id & RXM_RPC_TAG_FLAG);

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, dest_addr, &rxm_conn);
	if (ret)
		goto unlock;

	flags = rxm_ep->util_ep.tx_op_flags | FI_RPC | FI_REMOTE_CQ_DATA;
	ret = rxm_send_common(rxm_ep, rxm_conn, &iov, &desc, 1, context,
			      data, flags, rpc_id, ofi_op_tagged);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

static ssize_t
rxm_rpc_respmsg(struct fid_ep *ep_fid, const struct fi_msg_rpc_resp *msg,
	        uint64_t flags)
{
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	ssize_t ret;

	assert(msg->rpc_id & RXM_RPC_TAG_FLAG);

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	ofi_genlock_lock(&rxm_ep->util_ep.lock);
	ret = rxm_get_conn(rxm_ep, msg->addr, &rxm_conn);
	if (ret)
		goto unlock;

	ret = rxm_send_common(rxm_ep, rxm_conn, msg->iov, msg->desc,
			      msg->iov_count, msg->context, 0,
			      rxm_ep->util_ep.tx_op_flags | FI_RPC,
			      msg->rpc_id, ofi_op_tagged);
unlock:
	ofi_genlock_unlock(&rxm_ep->util_ep.lock);
	return ret;
}

ssize_t rxm_rpc_discard(struct fid_ep *ep_fid, uint64_t rpc_id)
{
	return 0;
}

struct fi_ops_rpc rxm_rpc_ops = {
	.size = sizeof(struct fi_ops_rpc),
	.req = rxm_rpc_req,
	.reqv = rxm_rpc_reqv,
	.reqdata = rxm_rpc_reqdata,
	.reqmsg = rxm_rpc_reqmsg,
	.resp = rxm_rpc_resp,
	.respv = rxm_rpc_respv,
	.respdata = rxm_rpc_respdata,
	.respmsg = rxm_rpc_respmsg,
	.discard = rxm_rpc_discard,
};
