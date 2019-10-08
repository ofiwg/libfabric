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
#include "mlx_core.h"

/*using for fi_tinject path*/
/*Using for selective completions scenario*/
void mlx_send_callback_no_compl(void *request, ucs_status_t status)
{
	mlx_req_release(request);
}

void mlx_send_callback(void *request,
		       ucs_status_t status)
{
	struct util_cq *cq;
	struct mlx_request *mlx_req = request;
	struct fi_cq_tagged_entry *tc = &mlx_req->completion.tagged;
	cq = mlx_req->cq;
	if (status != UCS_OK){
		int ret;
		MLX_PUSH_ERROR_COMPLETION(cq,
				tc->op_context, tc->flags,
				(int)status, -MLX_TRANSLATE_ERRCODE(status),
				0, ret, tc->tag);
	} else {
		ofi_cq_write(cq, tc->op_context, tc->flags,
				tc->len, tc->buf, 0, tc->tag);
	}
	mlx_req_release(request);
}

/*Using for selective completions scenario*/
void mlx_recv_callback_no_compl(void *request,
				ucs_status_t status,
				ucp_tag_recv_info_t *info)
{
	mlx_req_release(request);
}

void mlx_recv_callback(void *request,
		       ucs_status_t status,
		       ucp_tag_recv_info_t *info)
{
	struct util_cq *cq;
	struct mlx_request *mlx_req;
	struct fi_cq_tagged_entry *tc;
	mlx_req = (struct mlx_request*)request;
	mlx_req->completion.tagged.tag = info->sender_tag;
	mlx_req->completion.tagged.len = info->length;
	mlx_req->status = status;


	if (mlx_req->type == MLX_FI_REQ_UNINITIALIZED) {
		mlx_req->type = (status != UCS_OK) ?
					MLX_FI_REQ_UNEXPECTED_ERR
					: MLX_FI_REQ_UNEXPECTED;
		return;
	}

	cq = mlx_req->cq;
	tc = &mlx_req->completion.tagged;
	if (status != UCS_OK) {
		int ret;
		size_t olen = (info->length > mlx_req->posted_size)
				? info->length - mlx_req->posted_size : 0;
		MLX_PUSH_ERROR_COMPLETION(cq,
				tc->op_context,
				tc->flags,
				(int)status, -MLX_TRANSLATE_ERRCODE(status),
				olen, ret, tc->tag);
	} else {
		ofi_cq_write(cq, tc->op_context, tc->flags,
				tc->len, tc->buf, 0, tc->tag);
	}

	if (cq->wait) {
		cq->wait->signal(cq->wait);
	}

	mlx_req_release(request);
}

/* Multi-recv completion handlers*/
void mlx_multi_recv_callback(void *request,
				ucs_status_t ustatus,
				ucp_tag_recv_info_t *info)
{
	struct mlx_mrecv_request *mlx_req;
	struct mlx_mrecv_ctx *mctx;
	struct mlx_ep* ep;
	ssize_t status = -FI_EAGAIN;

	mlx_req = (struct mlx_mrecv_request*)request;
	mlx_req->last_recvd = info->length;
	mlx_req->status = ustatus;
	if (mlx_req->type == MLX_FI_REQ_UNINITIALIZED) {
		mlx_req->type = MLX_FI_REQ_MULTIRECV_UNEXP;
		return;
	}

	mctx = mlx_req->mrecv_ctx;
	ep = mlx_req->ep;

	if (!(ep->ep.rx_op_flags & FI_SELECTIVE_COMPLETION)
			|| (mctx->flags & FI_COMPLETION)
			|| (mlx_req->status != UCS_OK)
			|| ((mctx->remain - mlx_req->last_recvd) < ep->ep_opts.mrecv_min_size)) {
		status = mlx_generate_completion(mlx_req);
	}
	mctx->head = (void*)((char*)(mctx->head) + info->length);
	mctx->remain -= info->length;
	mlx_req_release((struct mlx_request*)mlx_req);

	if (status == FI_SUCCESS) {
		dlist_insert_head(&(mctx->list), &(ep->mctx_freelist));
	} else {
		dlist_insert_tail(&(mctx->list), &(ep->mctx_repost));
	}
}

