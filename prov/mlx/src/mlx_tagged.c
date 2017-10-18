/*
 * Copyright (c) 2016 Intel Corporation. All rights reserved.
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
#include "mlx.h"

#define __mlx_get_dstep_from_fi_addr(EP, ADDR) ((ucp_ep_h)(ADDR))

static ssize_t mlx_tagged_recvmsg(
				struct fid_ep *ep,
				const struct fi_msg_tagged *msg,
				uint64_t flags)
{
	ucs_status_ptr_t status = NULL;
	ucp_tag_recv_callback_t cbf;
	struct mlx_ep *u_ep;
	struct mlx_request *req;
	struct util_cq *cq;
	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);

	if (flags & FI_REMOTE_CQ_DATA) {
		return -FI_EBADFLAGS;
	}

	cbf = ((!(u_ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION)) 
			|| (flags & FI_COMPLETION)) ? 
				mlx_recv_callback : mlx_recv_callback_no_compl;

	if (msg->iov_count == 1) {
		status = ucp_tag_recv_nb(u_ep->worker, msg->msg_iov[0].iov_base,
					 msg->msg_iov[0].iov_len,
					 ucp_dt_make_contig(1),
					 msg->tag, (~(msg->ignore)), cbf);
	} else {
		return -FI_EINVAL; /*Do not return IOV for a while*/
	}

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	req = (struct mlx_request *)status;
	cq = u_ep->ep.rx_cq;
	req->cq = cq;
	req->ep =u_ep;

	if (msg->context) {
		struct fi_context *_ctx =
			((struct fi_context *)(msg->context));
		_ctx->internal[0] = (void*)req;
	}
	req->completion.tagged.op_context = msg->context;
	req->completion.tagged.flags = FI_RECV;
	req->completion.tagged.buf = msg->msg_iov[0].iov_base;
	req->completion.tagged.data = 0;

	if (req->type == MLX_FI_REQ_UNINITIALIZED) {
		req->type = MLX_FI_REQ_REGULAR;
		req->completion.tagged.tag = msg->tag;
		req->completion.tagged.len = msg->msg_iov[0].iov_len;
		goto fence;
	}

	/*Unexpected path*/
	struct fi_cq_tagged_entry *t_entry;
	fastlock_acquire(&cq->cq_lock);
	t_entry = ofi_cirque_tail(cq->cirq);
	*t_entry = (req->completion.tagged);

	if (req->type == MLX_FI_REQ_UNEXPECTED_ERR) {
		struct util_cq_err_entry* err;
		req->completion.error.olen -= req->completion.tagged.len;
		t_entry->flags |= UTIL_FLAG_ERROR;

		err = calloc(1, sizeof(struct util_cq_err_entry));
		if (!err) {
			FI_WARN(&mlx_prov, FI_LOG_CQ,
				"out of memory, cannot report CQ error\n");
			fastlock_release(&cq->cq_lock);
			return -FI_ENOMEM;
		}
		err->err_entry = (req->completion.error);
		slist_insert_tail(&err->list_entry, &cq->err_list);
	}

	ofi_cirque_commit(cq->cirq);
	fastlock_release(&cq->cq_lock);

fence:
	if (flags & FI_FENCE) {
		ucs_status_t cstatus;
		cstatus = ucp_worker_flush(u_ep->worker);
		if (status != UCS_OK)
			return MLX_TRANSLATE_ERRCODE(cstatus);
	}
	return FI_SUCCESS;
}

static ssize_t mlx_tagged_sendmsg(
				struct fid_ep *ep,
				const struct fi_msg_tagged *msg,
				uint64_t flags)
{
	struct mlx_ep* u_ep;
	ucp_send_callback_t cbf;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	ucs_status_t cstatus;
	struct util_cq *cq;
	ucp_tag_recv_info_t info;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = __mlx_get_dstep_from_fi_addr(u_ep, msg->addr);
	cq = u_ep->ep.tx_cq;

	if(flags & FI_REMOTE_CQ_DATA) {
		return -FI_EBADFLAGS;
	}

	cbf = ((!(u_ep->ep.tx_op_flags & FI_SELECTIVE_COMPLETION)) 
			|| (flags & FI_COMPLETION)) ? 
				mlx_send_callback : mlx_send_callback_no_compl;
	if (msg->iov_count == 1) {
		if (flags & FI_TRANSMIT_COMPLETE) {
			status = ucp_tag_send_sync_nb (
						dst_ep,
						msg->msg_iov[0].iov_base,
						msg->msg_iov[0].iov_len,
						ucp_dt_make_contig(1),
						msg->tag, cbf);
		} else {
			status = ucp_tag_send_nb(
						dst_ep,
						msg->msg_iov[0].iov_base,
						msg->msg_iov[0].iov_len,
						ucp_dt_make_contig(1),
						msg->tag, cbf);
		}
	} else {
		return -FI_EINVAL; /*Do not return IOV for a while*/
	}

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	if ((flags & FI_INJECT) && (UCS_PTR_STATUS(status) == UCS_OK)) {
		while (ucp_request_test(status, &info) != UCS_INPROGRESS)
			ucp_worker_progress(u_ep->worker);
		goto fence;
	}

	if((u_ep->ep.tx_op_flags & FI_SELECTIVE_COMPLETION)
			&& !(flags & FI_COMPLETION)) {
		goto fence;
	}

	if (msg->context) {
		struct fi_context* _ctx =
			((struct fi_context*)(msg->context));
		_ctx->internal[0] = status;
	}

	if (UCS_PTR_STATUS(status) != UCS_OK) {
		struct mlx_request *req;
		req = (struct mlx_request *) status;
		req->cq = cq;
		req->ep = u_ep;
		req->type = MLX_FI_REQ_REGULAR;
		req->completion.tagged.op_context = msg->context;
		req->completion.tagged.flags = FI_SEND;
		req->completion.tagged.len = msg->msg_iov[0].iov_len;
		req->completion.tagged.buf = msg->msg_iov[0].iov_base;
		req->completion.tagged.data = 0;
		req->completion.tagged.tag = msg->tag;
	} else {
		struct fi_cq_tagged_entry *t_entry;
		fastlock_acquire(&cq->cq_lock);
		t_entry = ofi_cirque_tail(cq->cirq);
		t_entry->op_context = msg->context;
		t_entry->flags = FI_SEND;
		t_entry->len = msg->msg_iov[0].iov_len;
		t_entry->buf = msg->msg_iov[0].iov_base;
		t_entry->data = 0;
		t_entry->tag = msg->tag;
		ofi_cirque_commit(cq->cirq);
		fastlock_release(&cq->cq_lock);
	}

fence:
	if(flags & FI_FENCE) {
		cstatus = ucp_worker_flush(u_ep->worker);
		if(status != UCS_OK) {
			return MLX_TRANSLATE_ERRCODE(cstatus);
		}
	}
	return FI_SUCCESS;
}


static ssize_t mlx_tagged_inject(
			struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	struct mlx_ep* u_ep;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	ucp_tag_recv_info_t info;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = __mlx_get_dstep_from_fi_addr(u_ep, dest_addr);

	status = ucp_tag_send_nb(dst_ep, buf, len,
				 ucp_dt_make_contig(1),
				 tag, mlx_send_callback_no_compl);
	if (UCS_PTR_STATUS(status) == UCS_OK)
		return FI_SUCCESS;

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	/* `info` is left unitialized, because this is send operation */
	while (ucp_request_test(status, &info) != UCS_INPROGRESS)
		ucp_worker_progress(u_ep->worker);

	return FI_SUCCESS;
}

static ssize_t mlx_tagged_send(
				struct fid_ep *ep, const void *buf,
				size_t len, void *desc,
				fi_addr_t dest_addr,
				uint64_t tag, void *context)
{
	struct iovec iov = {
		.iov_base = (void*)buf,
		.iov_len = len,
	};

	struct fi_msg_tagged msg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = dest_addr,
		.tag = tag,
		.context = context,
	};

	return mlx_tagged_sendmsg( ep, &msg, 0);
}

static ssize_t mlx_tagged_sendv(
				struct fid_ep *ep, const struct iovec *iov,
				void **desc,
				size_t count, fi_addr_t dest_addr,
				uint64_t tag, void *context)
{
	struct fi_msg_tagged msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.tag = tag,
		.context = context,
	};

	return mlx_tagged_sendmsg( ep, &msg, 0);
}

static ssize_t mlx_tagged_recvv(
			struct fid_ep *ep, const struct iovec *iov, void **desc,
			size_t count, fi_addr_t src_addr,
			uint64_t tag, uint64_t ignore, void *context)
{
	struct fi_msg_tagged msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = src_addr,
		.tag = tag,
		.ignore = ignore,
		.context = context,
	};
	return mlx_tagged_recvmsg(ep, &msg, 0);
}

static ssize_t mlx_tagged_recv(
			struct fid_ep *ep, void *buf, size_t len, void *desc,
			fi_addr_t src_addr,
			uint64_t tag,
			uint64_t ignore,
			void *context)
{
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = len,
	};

	struct fi_msg_tagged msg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = src_addr,
		.tag = tag,
		.ignore = ignore,
		.context = context,
	};
	return mlx_tagged_recvmsg(ep, &msg, 0);
}

struct fi_ops_tagged mlx_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = mlx_tagged_recv,
	.recvv = mlx_tagged_recvv,
	.recvmsg = mlx_tagged_recvmsg,
	.send = mlx_tagged_send,
	.senddata = fi_no_tagged_senddata,
	.sendv = mlx_tagged_sendv,
	.inject = mlx_tagged_inject,
	.sendmsg = mlx_tagged_sendmsg,
	.injectdata = fi_no_tagged_injectdata,
};

