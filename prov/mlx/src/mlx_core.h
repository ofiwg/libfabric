/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
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




struct mlx_claimed_msg {
	struct dlist_entry dentry;
	uint64_t tag;
	ucp_tag_message_h ucp_msg;
};

static struct mlx_claimed_msg *mlx_dequeue_claimed(struct mlx_ep *u_ep,
		const struct fi_msg_tagged *msg)
{
	struct dlist_entry *dentry = NULL;
	struct mlx_claimed_msg *cmsg;

	dlist_foreach((&u_ep->claimed_list), dentry) {
		cmsg = (struct mlx_claimed_msg *) dentry;
		if ((cmsg->tag & ~msg->ignore) == (msg->tag & ~msg->ignore)) {
			dlist_remove(&cmsg->dentry);
			return cmsg;
		}
	}
	return NULL;
}

static void mlx_enqueue_claimed(struct mlx_ep *u_ep,
		ucp_tag_recv_info_t *recv_info,
		ucp_tag_message_h probed_msg)
{
	struct mlx_claimed_msg *cmsg;
	cmsg = malloc(sizeof(struct mlx_claimed_msg));
	dlist_init(&cmsg->dentry);
	cmsg->tag = recv_info->sender_tag;
	cmsg->ucp_msg = probed_msg;
	dlist_insert_tail(&cmsg->dentry, &u_ep->claimed_list);
}

static inline ssize_t mlx_generate_completion(struct mlx_mrecv_request *req) {
	struct mlx_mrecv_ctx *mctx = req->mrecv_ctx;
	struct util_cq *cq = req->cq;
	ssize_t ret = -FI_EAGAIN;

	if (req->status != UCS_OK) {
		MLX_PUSH_ERROR_COMPLETION(cq, mctx->context,
				FI_RECV | FI_MSG, (int)req->status,
				-MLX_TRANSLATE_ERRCODE(req->status),
				(req->last_recvd - mctx->remain),
				ret, mctx->tag);
		ret = -MLX_TRANSLATE_ERRCODE(req->status);
	} else {
		uint64_t flags = FI_RECV | FI_MSG;
		int buff_is_full = (req->ep->ep_opts.mrecv_min_size
					> (mctx->remain - req->last_recvd));
		if (buff_is_full)
			flags |= FI_MULTI_RECV;
		ofi_cq_write(cq, mctx->context, flags,  req->last_recvd,
				mctx->head, 0, 0);
		if (buff_is_full) {
			ret = FI_SUCCESS;
		}
	}

	if (cq->wait) {
		cq->wait->signal(cq->wait);
	}
	return ret;
}



static inline ssize_t mlx_do_recvmsg(struct fid_ep *ep, const struct fi_msg_tagged *msg, const uint64_t flags,
		uint64_t tag, uint64_t ignore, const mlx_comm_mode_t mode)
{
	ucs_status_ptr_t status = NULL;
	ucp_tag_recv_callback_t cbf;
	struct mlx_ep *u_ep;
	struct mlx_request *req;
	struct util_cq *cq;
	ucp_datatype_t recv_dt;
	size_t recv_cnt;
	void *recv_buf;
	int ret = FI_SUCCESS;
	ssize_t posted_size = 0;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	cq = u_ep->ep.rx_cq;

	if (msg->iov_count < 2) {
		recv_dt =  ucp_dt_make_contig(1);
		recv_buf = msg->msg_iov[0].iov_base;
		recv_cnt = msg->msg_iov[0].iov_len;
		posted_size = recv_cnt;
	} else {
		int i;
		recv_dt = ucp_dt_make_iov();
		recv_buf = (void*)msg->msg_iov;
		recv_cnt = msg->iov_count;
		for (i=0; i < msg->iov_count; ++i )
			posted_size += msg->msg_iov[i].iov_len;
	}

	if (mode != MLX_TAGGED) {
		return -FI_EINVAL;
	}

	if (flags & FI_CLAIM) {
		struct mlx_claimed_msg *cmsg = mlx_dequeue_claimed(u_ep, msg);
		if (!cmsg)
			return -FI_EINVAL;

		cbf = ((!(u_ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION))
				|| (flags & FI_COMPLETION)
				|| (flags & FI_DISCARD)) ?
					mlx_recv_callback : mlx_recv_callback_no_compl;
		status = ucp_tag_msg_recv_nb(u_ep->worker,
					 recv_buf,
					 recv_cnt,
					 recv_dt,
					 cmsg->ucp_msg, cbf);
		free(cmsg);
	} else {
		cbf = ((!(u_ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION)) 
				|| (flags & FI_COMPLETION)) ? 
					mlx_recv_callback : mlx_recv_callback_no_compl;
		status = ucp_tag_recv_nb(u_ep->worker, recv_buf,
					 recv_cnt,
					 recv_dt,
					 tag, ignore, cbf);
	}

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	req = (struct mlx_request *)status;
	req->cq = cq;
	req->ep =u_ep;

	if (msg->context) {
		struct fi_context *_ctx =
			((struct fi_context *)(msg->context));
		_ctx->internal[0] = (void*)req;
		_ctx->internal[1] = NULL;
	}

	req->completion.tagged.op_context = msg->context;
	req->completion.tagged.flags = FI_RECV | ((mode == MLX_MSG) ? FI_MSG : FI_TAGGED);
	req->completion.tagged.buf = msg->msg_iov[0].iov_base;
	req->completion.tagged.data = 0;

	req->posted_size = posted_size;
	if (req->type == MLX_FI_REQ_UNINITIALIZED) {
		req->type = MLX_FI_REQ_REGULAR;
		req->completion.tagged.tag = msg->tag;
		req->completion.tagged.len = msg->msg_iov[0].iov_len;
		return FI_SUCCESS;
	}

	/*Unexpected path*/
	struct fi_cq_tagged_entry *tc = &req->completion.tagged;
	if (req->type == MLX_FI_REQ_UNEXPECTED_ERR) {
		MLX_PUSH_ERROR_COMPLETION(cq, tc->op_context,
			tc->flags,
			(int)req->status, -MLX_TRANSLATE_ERRCODE((int)req->status) ,
			(tc->len - req->posted_size),
			ret, tc->tag);
	} else if ((!(u_ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION))
				|| (flags & FI_COMPLETION)
				|| (req->type == MLX_FI_REQ_UNEXPECTED_ERR)) {
		ofi_cq_write(cq, tc->op_context, tc->flags,
				tc->len, tc->buf, 0, tc->tag);
	}
	mlx_req_release(req);
	return ret;
}

static inline ssize_t mlx_do_inject(
			struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	struct mlx_ep* u_ep;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	ucs_status_t ret;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = MLX_GET_UCP_EP(u_ep, dest_addr);

	status = ucp_tag_send_nb(dst_ep, buf, len,
				 ucp_dt_make_contig(1),
				 tag, mlx_send_callback_no_compl);

	if (status == NULL)
		return FI_SUCCESS;

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	/* `info` is left unitialized, because this is send operation */
	while ((ret = ucp_request_check_status(status)) == UCS_INPROGRESS)
		ucp_worker_progress(u_ep->worker);

	return -MLX_TRANSLATE_ERRCODE((int)ret);
}

static inline  ssize_t mlx_do_sendmsg(
				struct fid_ep *ep,
				const struct fi_msg_tagged *msg,
				uint64_t flags, uint64_t tag, const mlx_comm_mode_t mode)
{
	struct mlx_ep* u_ep;
	ucp_send_callback_t cbf;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	struct util_cq *cq;
	ucs_status_t cstatus = UCS_OK;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = MLX_GET_UCP_EP(u_ep, msg->addr);
	cq = u_ep->ep.tx_cq;

	cbf = ((!(u_ep->ep.tx_op_flags & FI_SELECTIVE_COMPLETION)) 
			|| (flags & FI_COMPLETION)) ?
				mlx_send_callback : mlx_send_callback_no_compl;

	if (OFI_UNLIKELY(flags & FI_MATCH_COMPLETE)) {
		if (msg->iov_count < 2) {
			status = ucp_tag_send_sync_nb(
						dst_ep,
						msg->msg_iov[0].iov_base,
						msg->msg_iov[0].iov_len,
						ucp_dt_make_contig(1),
						tag, cbf);
		} else {
			status = ucp_tag_send_sync_nb(
						dst_ep,
						msg->msg_iov,
						msg->iov_count,
						ucp_dt_make_iov(),
						tag, cbf);
		}
	} else {
		if (msg->iov_count < 2) {
			status = ucp_tag_send_nb(
						dst_ep,
						msg->msg_iov[0].iov_base,
						msg->msg_iov[0].iov_len,
						ucp_dt_make_contig(1),
						tag, cbf);
		} else {
			status = ucp_tag_send_nb(
						dst_ep,
						msg->msg_iov,
						msg->iov_count,
						ucp_dt_make_iov(),
						tag, cbf);
		}
	}

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	if (flags & FI_INJECT) {
		if(UCS_PTR_STATUS(status) != UCS_OK) {
			while ((cstatus = ucp_request_check_status(status)) == UCS_INPROGRESS)
				ucp_worker_progress(u_ep->worker);
		}
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
		_ctx->internal[1] = NULL;
	}

	if (status != NULL) {
		struct mlx_request *req = (struct mlx_request *)status;
		req->completion.tagged.op_context = msg->context;
		req->completion.tagged.flags = FI_SEND | ((mode == MLX_MSG) ? FI_MSG : FI_TAGGED);
		req->completion.tagged.len = msg->msg_iov[0].iov_len;
		req->completion.tagged.buf = msg->msg_iov[0].iov_base;
		req->completion.tagged.tag = msg->tag;
		req->ep = u_ep;
		req->cq = cq;
	} else {
		ofi_cq_write(cq,  msg->context,
				FI_SEND | ((mode == MLX_MSG) ? FI_MSG : FI_TAGGED),
				msg->msg_iov[0].iov_len,
				msg->msg_iov[0].iov_base,
				0, msg->tag);
	}

fence:
	if(flags & (FI_FENCE | FI_TRANSMIT_COMPLETE)) {
		cstatus = ucp_worker_flush(u_ep->worker);
	}
	return MLX_TRANSLATE_ERRCODE(cstatus);
}
