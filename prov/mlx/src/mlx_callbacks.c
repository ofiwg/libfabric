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

/*using for fi_tinject path*/
/*Using for selective completions scenario*/
void mlx_send_callback_no_compl(void *request, ucs_status_t status)
{
	ucp_request_release(request);
}

void mlx_send_callback(void *request,
		       ucs_status_t status)
{
	struct util_cq *cq;
	struct mlx_request *mlx_req = request;
	struct fi_cq_tagged_entry *t_entry;
	struct util_cq_err_entry *err;

	cq = mlx_req->cq;

	if (status == UCS_ERR_CANCELED) {
		ucp_request_release(request);
		return;
	}

	fastlock_acquire(&cq->cq_lock);

	t_entry = ofi_cirque_tail(cq->cirq);
	*t_entry = (mlx_req->completion.tagged);
	ofi_cirque_commit(cq->cirq);

	if (status != UCS_OK){
		t_entry->flags |= UTIL_FLAG_ERROR;
		err = calloc(1, sizeof(struct util_cq_err_entry));
		if (!err) {
			FI_WARN(&mlx_prov, FI_LOG_CQ,
				"out of memory, cannot report CQ error\n");
			goto fn;
		}

		err->err_entry = (mlx_req->completion.error);
		err->err_entry.prov_errno = (int)status;
		err->err_entry.err = MLX_TRANSLATE_ERRCODE(status);
		err->err_entry.olen = 0;
		slist_insert_tail(&err->list_entry, &cq->err_list);
	}
fn:
	mlx_req->type = MLX_FI_REQ_UNINITIALIZED;
	fastlock_release(&cq->cq_lock);
	ucp_request_release(request);
}

/*Using for selective completions scenario*/
void mlx_recv_callback_no_compl(void *request,
				ucs_status_t status,
				ucp_tag_recv_info_t *info)
{
	ucp_request_release(request);
}

void mlx_recv_callback(void *request,
		       ucs_status_t status,
		       ucp_tag_recv_info_t *info)
{
	struct util_cq *cq;
	struct mlx_request *mlx_req;

	mlx_req = (struct mlx_request*)request;
	if (status == UCS_ERR_CANCELED) {
		ucp_request_release(request);
		return;
	}

	cq = mlx_req->cq;

	mlx_req->completion.tagged.tag = info->sender_tag;
	mlx_req->completion.tagged.len = info->length;

	if (status != UCS_OK) {
		mlx_req->completion.error.prov_errno = (int)status;
		mlx_req->completion.error.err = MLX_TRANSLATE_ERRCODE(status);
	}

	fastlock_acquire(&cq->cq_lock);
	if (mlx_req->type == MLX_FI_REQ_UNINITIALIZED) {
		if (status != UCS_OK) {
			mlx_req->completion.error.olen = info->length;
			mlx_req->type = MLX_FI_REQ_UNEXPECTED_ERR;
		} else {
			mlx_req->type = MLX_FI_REQ_UNEXPECTED;
		}
		fastlock_release(&cq->cq_lock);
		return;
	} else {
		if (status != UCS_OK) {
			mlx_req->completion.error.olen = info->length -
						mlx_req->completion.error.len;
		}

		struct fi_cq_tagged_entry *t_entry;
		t_entry = ofi_cirque_tail(cq->cirq);
		*t_entry = (mlx_req->completion.tagged);

		if (status != UCS_OK) {
			struct util_cq_err_entry* err;
			t_entry->flags |= UTIL_FLAG_ERROR;

			err = calloc(1, sizeof(struct util_cq_err_entry));
			if (!err) {
				FI_WARN(&mlx_prov, FI_LOG_CQ,
					"out of memory, cannot report CQ error\n");
				mlx_req->type = MLX_FI_REQ_UNINITIALIZED;
				goto fn;
			}

			err->err_entry = (mlx_req->completion.error);
			slist_insert_tail(&err->list_entry, &cq->err_list);
		}

		if (cq->src){
			cq->src[ofi_cirque_windex((struct mlx_comp_cirq*)(cq->cirq))] =
					FI_ADDR_NOTAVAIL;
		}

		if (cq->wait) {
			cq->wait->signal(cq->wait);
		}

		mlx_req->type = MLX_FI_REQ_UNINITIALIZED;
		ofi_cirque_commit(cq->cirq);
	}
fn:
	fastlock_release(&cq->cq_lock);
	ucp_request_release(request);
}

