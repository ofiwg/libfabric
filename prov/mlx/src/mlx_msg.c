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
#include "mlx_core.h"

ssize_t mlx_mrecv_repost(struct mlx_ep *ep, struct mlx_mrecv_ctx *mctx)
{
	ssize_t ret = -FI_EAGAIN;
	ucs_status_ptr_t status = NULL;
	struct mlx_mrecv_request *req;
	struct fi_context* _ctx;

	status = ucp_tag_recv_nb(ep->worker,
				 mctx->head,
				 mctx->remain,
				 ucp_dt_make_contig(1),
				 mctx->tag,
				 mctx->ignore,
				 mlx_multi_recv_callback);
	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Send operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	_ctx = (struct fi_context*)(mctx->context);
	_ctx->internal[0] = (void *) status;
	_ctx->internal[1] = (void *) mctx;

	req = (struct mlx_mrecv_request *)status;
	req->mrecv_ctx = mctx;
	req->cq = ep->ep.rx_cq;
	req->ep = ep;
	if (req->type == MLX_FI_REQ_UNINITIALIZED) {
		req->type = MLX_FI_REQ_MULTIRECV;
		return FI_SUCCESS;
	}
	req->type = MLX_FI_REQ_MULTIRECV;

	if( !(ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION)
			|| (mctx->flags & FI_COMPLETION)
			|| (req->status != UCS_OK)
			|| ((mctx->remain - req->last_recvd) < ep->ep_opts.mrecv_min_size)) {
		ret = mlx_generate_completion(req);
	}

	mctx->remain -= req->last_recvd;
	mctx->head = (void*)((char*)(mctx->head) + req->last_recvd);
	mlx_req_release((struct mlx_request*)req);

	if (ret == FI_SUCCESS) {
		dlist_insert_head(&(mctx->list), &(ep->mctx_freelist));
	} else {
		dlist_insert_tail(&(mctx->list), &(ep->mctx_repost));
	}

	return FI_SUCCESS;
}

static ssize_t mlx_recvmsg(struct fid_ep *ep, const struct fi_msg *msg, uint64_t flags)
{
	struct mlx_mrecv_ctx *mrecv_ctx;
	struct mlx_ep *u_ep;

	if (flags & (FI_REMOTE_CQ_DATA | FI_PEEK | FI_CLAIM)) {
		return -FI_EBADFLAGS;
	}

	if (msg->addr != FI_ADDR_UNSPEC) {
		return -FI_EADDRNOTAVAIL;
	}

	/* MLX msg supports multi-recv only */
	if (!(flags & (FI_MULTI_RECV))) {
		return -FI_EBADFLAGS;
	}
	if (msg->iov_count > 1) {
		return -FI_EINVAL;
	}

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	if (!dlist_empty(&u_ep->mctx_freelist)){
		dlist_pop_front(&u_ep->mctx_freelist,
					struct mlx_mrecv_ctx, mrecv_ctx, list);
	} else {
		mrecv_ctx = (struct mlx_mrecv_ctx*) calloc(1,
				sizeof(struct mlx_mrecv_ctx));
		if (!mrecv_ctx) {
			return -FI_ENOMEM;
		}
	}

	mrecv_ctx->head = msg->msg_iov[0].iov_base;
	mrecv_ctx->remain = msg->msg_iov[0].iov_len;
	mrecv_ctx->tag = MLX_EP_MSG_TAG;
	mrecv_ctx->ignore = MLX_AUX_TAG_MASK;
	mrecv_ctx->flags = flags;
	mrecv_ctx->context = msg->context;
	dlist_init(&mrecv_ctx->list);
	return mlx_mrecv_repost(u_ep, mrecv_ctx);
}


ssize_t mlx_inject(struct fid_ep *ep, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	return mlx_do_inject(ep, buf, len, dest_addr, tag);
}

ssize_t mlx_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
		uint64_t flags)
{
	uint64_t tag = MLX_EP_MSG_TAG;

	struct fi_msg_tagged tmsg = {
		.msg_iov = msg->msg_iov,
		.desc = msg->desc,
		.iov_count = msg->iov_count,
		.addr = msg->addr,
		.tag = 0,
		.ignore = 0,
		.context = msg->context,
		.data = 0,
	};

	if(flags & FI_REMOTE_CQ_DATA) {
		return -FI_EBADFLAGS;
	}

	return mlx_do_sendmsg(ep, &tmsg, flags, tag, MLX_MSG);
}


static ssize_t mlx_send(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct fi_msg_tagged tmsg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = dest_addr,
		.context = context,
		.data = 0,
	};

	return mlx_do_sendmsg(ep, &tmsg, 0, tag, MLX_MSG);
}

static ssize_t mlx_sendv(struct fid_ep *ep, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	uint64_t tag = MLX_EP_MSG_TAG;
	struct fi_msg_tagged tmsg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = count,
		.addr = dest_addr,
		.context = context,
		.data = 0,
	};

	return mlx_do_sendmsg(ep, &tmsg, 0, tag, MLX_MSG);
}


struct fi_ops_msg mlx_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = mlx_recvmsg,
	.send = mlx_send,
	.sendv = mlx_sendv,
	.sendmsg = mlx_sendmsg,
	.inject = mlx_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};
