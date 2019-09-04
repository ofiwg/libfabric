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
#include "ofi_util.h"

#define MLX_DO_READ 0
#define MLX_DO_WRITE 1
#define __mlx_get_dstep_from_fi_addr(EP, ADDR) ((ucp_ep_h)(ADDR))

static ssize_t mlx_proc_rma_msg(struct fid_ep *ep,
			const struct fi_msg_rma *msg,
			uint64_t flags, int op_type);

static ssize_t mlx_write(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key, void *context)
{
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = len,
	};
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	return mlx_proc_rma_msg(ep, &msg, 0, MLX_DO_WRITE);
}

static ssize_t mlx_writev(struct fid_ep *ep, const struct iovec *iov,
			void **desc, size_t count, fi_addr_t dest_addr,
			uint64_t addr, uint64_t key, void *context)
{
	if (count != 1)
		return -FI_EINVAL;

	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = iov[0].iov_len,
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = 1,
		.addr = dest_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	return mlx_proc_rma_msg(ep, &msg, 0, MLX_DO_WRITE);
}

static ssize_t mlx_read(struct fid_ep *ep, const void *buf, size_t len,
			void *desc, fi_addr_t src_addr, uint64_t addr,
			uint64_t key, void *context)
{
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = len,
	};
	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = len,
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = &iov,
		.desc = desc,
		.iov_count = 1,
		.addr = src_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	return mlx_proc_rma_msg(ep, &msg, 0, MLX_DO_READ);
}

static ssize_t mlx_readv(struct fid_ep *ep, const struct iovec *iov,
			void **desc, size_t count, fi_addr_t src_addr,
			uint64_t addr, uint64_t key, void *context)
{
	if (count != 1)
		return -FI_EINVAL;

	struct fi_rma_iov rma_iov = {
		.addr = addr,
		.len = iov[0].iov_len,
		.key = key,
	};
	struct fi_msg_rma msg = {
		.msg_iov = iov,
		.desc = desc,
		.iov_count = 1,
		.addr = src_addr,
		.rma_iov = &rma_iov,
		.rma_iov_count = 1,
		.context = context,
		.data = 0,
	};
	return mlx_proc_rma_msg(ep, &msg, 0, MLX_DO_READ);
}

static ssize_t mlx_writemsg(struct fid_ep *ep,
			const struct fi_msg_rma *msg,
			uint64_t flags)
{
	return mlx_proc_rma_msg(ep, msg, flags, MLX_DO_WRITE);
}

static ssize_t mlx_readmsg(struct fid_ep *ep,
			const struct fi_msg_rma *msg,
			uint64_t flags)
{
	return mlx_proc_rma_msg(ep, msg, flags, MLX_DO_READ);
}
/*=======================================================*/


void mlx_rma_callback(void *request,
		       ucs_status_t status)
{
	struct mlx_request *mlx_req = request;
	if (status == UCS_OK){
		if (mlx_req->type == MLX_FI_REQ_WRITE) {
			ofi_ep_wr_cntr_inc(&(mlx_req->ep->ep));
		} else {
			ofi_ep_rd_cntr_inc(&(mlx_req->ep->ep));
		}
		if (mlx_req->completion.tagged.flags & FI_COMPLETION)
			ofi_cq_write(mlx_req->ep->ep.rx_cq, mlx_req->completion.tagged.op_context,
					mlx_req->completion.tagged.flags,
					0, NULL, 0, 0);
	} else {
		int ret;
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"RMA completion error %s\n",
			ucs_status_string(status));
		struct util_cntr *cntr = (mlx_req->type == MLX_FI_REQ_WRITE) ?
				mlx_req->ep->ep.wr_cntr : mlx_req->ep->ep.rd_cntr;
		if (cntr)
			cntr->cntr_fid.ops->adderr(&cntr->cntr_fid, 1);
		if (mlx_req->completion.tagged.flags & FI_COMPLETION) {
			MLX_PUSH_ERROR_COMPLETION(mlx_req->ep->ep.rx_cq,
					mlx_req->completion.tagged.op_context,
					mlx_req->completion.tagged.flags,
					status, -MLX_TRANSLATE_ERRCODE(status),
					0, ret);
		}
	}
	mlx_req_release(request);
}

static ssize_t mlx_proc_rma_msg(struct fid_ep *ep,
			const struct fi_msg_rma *msg,
			uint64_t flags, int op_type)
{
	struct mlx_ep* u_ep;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	struct mlx_mr_rkey tmp_rkey;
	struct mlx_domain *domain;
	struct mlx_mr_rkey *rkey;
	struct mlx_request * req;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = __mlx_get_dstep_from_fi_addr(u_ep, msg->addr);

	if (msg->rma_iov_count > 1 || msg->iov_count > 1) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"MLX/RMA: unsupported IOV len. Local: %lu, Remote %lu",
			msg->iov_count,
			msg->rma_iov_count);
		return -FI_EINVAL;
	}

	tmp_rkey.id.owner_addr = msg->addr;
	tmp_rkey.id.key = msg->rma_iov[0].key;
	domain = container_of(u_ep->ep.domain ,struct mlx_domain, u_domain);

	HASH_FIND(hh, domain->remote_keys, &tmp_rkey.id, sizeof(tmp_rkey.id),
				rkey);
	if (!rkey) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"MLX/RMA: key not found {%llu:%llu}\n",
			tmp_rkey.id.owner_addr,
			tmp_rkey.id.key);
		return -FI_EINVAL;
	}

	if (MLX_DO_READ == op_type) {
		status = ucp_get_nb(dst_ep,
				msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
				msg->rma_iov[0].addr, rkey->rkey, mlx_rma_callback);
	} else {
		status = ucp_put_nb(dst_ep,
				msg->msg_iov[0].iov_base, msg->msg_iov[0].iov_len,
				msg->rma_iov[0].addr, rkey->rkey, mlx_rma_callback);
	}
	if (status == UCS_OK) {
		if (MLX_DO_WRITE == op_type) {
			ofi_ep_wr_cntr_inc(&(u_ep->ep));
		} else {
			ofi_ep_rd_cntr_inc(&(u_ep->ep));
		}
		if (flags & FI_COMPLETION) {
			ofi_cq_write(u_ep->ep.rx_cq, msg->context, flags,
				0, NULL, 0, 0);
		}
		return FI_SUCCESS;
	}

	if (UCS_PTR_IS_ERR(status)) {
		FI_DBG( &mlx_prov,FI_LOG_CORE,
			"Write operation returns error: %s",
			ucs_status_string(*(ucs_status_t*)status));
		return MLX_TRANSLATE_ERRCODE(*(ucs_status_t*)status);
	}

	req = (struct mlx_request *)status;
	req->cq = NULL;
	req->ep = u_ep;
	req->completion.tagged.op_context = msg->context;
	req->completion.tagged.flags = flags;
	req->type = (MLX_DO_READ == op_type) ?
			MLX_FI_REQ_READ : MLX_FI_REQ_WRITE;
	return FI_SUCCESS;
}


ssize_t mlx_inject_write(struct fid_ep *ep, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct mlx_ep* u_ep;
	ucp_ep_h dst_ep;
	ucs_status_ptr_t status = NULL;
	ucs_status_t ret = UCS_OK;
	struct mlx_mr_rkey tmp_rkey;
	struct mlx_domain *domain;
	struct mlx_mr_rkey *rkey;

	u_ep = container_of(ep, struct mlx_ep, ep.ep_fid);
	dst_ep = __mlx_get_dstep_from_fi_addr(u_ep, dest_addr);
	domain = container_of(u_ep->ep.domain ,struct mlx_domain, u_domain);

	tmp_rkey.id.owner_addr = dest_addr;
	tmp_rkey.id.key = key;
	HASH_FIND(hh, domain->remote_keys, &tmp_rkey.id, sizeof(tmp_rkey.id),
				rkey);
	if (!rkey) {
		return -FI_EINVAL;
	}

	status = ucp_put_nb(dst_ep,
			buf, len, addr, rkey->rkey, mlx_send_callback_no_compl);

	if (status != UCS_OK) {
		if (UCS_PTR_IS_ERR(status))
			return MLX_TRANSLATE_ERRCODE((int)UCS_PTR_STATUS(status));

		while ((ret = ucp_request_check_status(status)) == UCS_INPROGRESS)
			ucp_worker_progress(u_ep->worker);
	}

	ofi_ep_wr_cntr_inc(&(u_ep->ep));

	return MLX_TRANSLATE_ERRCODE((int)ret);
}

struct fi_ops_rma mlx_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = mlx_read,
	.readv = mlx_readv,
	.readmsg = mlx_readmsg,
	.write = mlx_write,
	.writev = mlx_writev,
	.writemsg = mlx_writemsg,
	.inject = mlx_inject_write,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};
