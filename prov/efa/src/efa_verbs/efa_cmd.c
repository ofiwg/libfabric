/*
 * Copyright (c) 2018-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#include "infiniband/efa_verbs.h"

#include "efa_cmd.h"
#include "efa_ib_cmd.h"
#include "efa_io_defs.h" /* entry sizes */

int efa_cmd_alloc_ucontext(struct ibv_device *device, struct efa_context *ctx, int cmd_fd)
{
	struct efa_alloc_ucontext_resp resp;
	struct ibv_get_context cmd = {};
	struct ibv_context *ibctx;
	int ret;

	ibctx = &ctx->ibv_ctx;
	ibctx->device = device;
	ibctx->cmd_fd = cmd_fd;

	ret = efa_ib_cmd_get_context(ibctx, &cmd, sizeof(cmd),
				     &resp.ibv_resp, sizeof(resp));
	if (ret)
		return ret;

	ctx->cmds_supp_udata = resp.efa_resp.cmds_supp_udata_mask;
	ctx->sub_cqs_per_cq = resp.efa_resp.sub_cqs_per_cq;
	ctx->inject_size = resp.efa_resp.inline_buf_size;
	ctx->max_llq_size = resp.efa_resp.max_llq_size;

	return 0;
}

static int efa_everbs_cmd_get_ex_query_dev(struct efa_context *ctx,
					   struct efa_device_attr *attr)
{
	struct efa_everbs_get_ex_dev_attrs_resp resp;
	struct efa_everbs_get_ex_dev_attrs cmd = {};

	cmd.command     = EFA_EVERBS_CMD_GET_EX_DEV_ATTRS;
	cmd.in_words    = sizeof(cmd) / 4;
	cmd.out_words   = sizeof(resp) / 4;
	cmd.response    = (uintptr_t)&resp;

	if (write(ctx->efa_everbs_cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));

	attr->max_sq_wr         = resp.max_sq_wr;
	attr->max_rq_wr         = resp.max_rq_wr;
	attr->max_sq_sge        = resp.max_sq_sge;
	attr->max_rq_sge        = resp.max_rq_sge;

	return 0;
}

int efa_cmd_query_device(struct efa_context *ctx, struct efa_device_attr *attr)
{
	struct efa_ex_query_device_resp resp;
	unsigned int major, minor, sub_minor;
	struct ibv_ex_query_device cmd_ex;
	struct ibv_query_device cmd;
	uint64_t raw_fw_ver;
	int ret;

	if (ctx->cmds_supp_udata & EFA_USER_CMDS_SUPP_UDATA_QUERY_DEVICE) {
		ret = efa_ib_cmd_query_device_ex(&ctx->ibv_ctx, &attr->ibv_attr, &raw_fw_ver,
						 &cmd_ex, sizeof(cmd_ex), sizeof(cmd_ex),
						 &resp.ibv_resp, sizeof(resp.ibv_resp), sizeof(resp));
		if (ret)
			return ret;

		attr->max_sq_wr         = resp.efa_resp.max_sq_wr;
		attr->max_rq_wr         = resp.efa_resp.max_rq_wr;
		attr->max_sq_sge        = resp.efa_resp.max_sq_sge;
		attr->max_rq_sge        = resp.efa_resp.max_rq_sge;
	} else {
		ret = efa_ib_cmd_query_device(&ctx->ibv_ctx, &attr->ibv_attr, &raw_fw_ver, &cmd, sizeof(cmd));
		if (ret)
			return ret;

		ret = efa_everbs_cmd_get_ex_query_dev(ctx, attr);
		if (ret)
			return ret;
	}

	major     = (raw_fw_ver >> 32) & 0xffff;
	minor     = (raw_fw_ver >> 16) & 0xffff;
	sub_minor = raw_fw_ver & 0xffff;

	snprintf(attr->ibv_attr.fw_ver, sizeof(attr->ibv_attr.fw_ver),
		 "%u.%u.%03u", major, minor, sub_minor);

	return 0;
}

int efa_cmd_query_port(struct efa_context *ctx, uint8_t port, struct ibv_port_attr *attr)
{
	struct ibv_query_port cmd;

	return efa_ib_cmd_query_port(&ctx->ibv_ctx, port, attr, &cmd, sizeof(cmd));
}

struct efa_pd *efa_cmd_alloc_pd(struct efa_context *ctx)
{
	struct efa_alloc_pd_resp resp;
	struct ibv_alloc_pd cmd;
	struct efa_pd *pd;

	pd = malloc(sizeof(*pd));
	if (!pd)
		return NULL;

	if (efa_ib_cmd_alloc_pd(&ctx->ibv_ctx, &pd->ibv_pd, &cmd, sizeof(cmd),
				&resp.ibv_resp, sizeof(resp))) {
		free(pd);
		return NULL;
	}

	pd->context = ctx;
	pd->pdn = resp.efa_resp.pdn;

	return pd;
}

int efa_cmd_dealloc_pd(struct efa_pd *pd)
{
	int ret;

	ret = efa_ib_cmd_dealloc_pd(&pd->ibv_pd);
	if (ret)
		return ret;

	free(pd);
	return 0;
}

struct ibv_mr *efa_cmd_reg_mr(struct efa_pd *pd, void *addr,
			      size_t length, int access)
{
	struct ib_uverbs_reg_mr_resp resp;
	struct ibv_reg_mr cmd;
	struct ibv_mr *mr;
	int ret;

	mr = malloc(sizeof(*mr));
	if (!mr)
		return NULL;

	ret = efa_ib_cmd_reg_mr(&pd->ibv_pd, addr, length, (uintptr_t)addr,
				access, mr, &cmd, sizeof(cmd),
				&resp, sizeof(resp));
	if (ret) {
		free(mr);
		return NULL;
	}

	mr->context = pd->ibv_pd.context;
	mr->pd      = &pd->ibv_pd;
	mr->addr    = addr;
	mr->length  = length;

	return mr;
}

int efa_cmd_dereg_mr(struct ibv_mr *mr)
{
	int ret;

	ret = efa_ib_cmd_dereg_mr(mr);
	if (ret)
		return ret;

	free(mr);

	return ret;
}

/* context->mutex must be held */
int efa_cmd_create_cq(struct efa_cq *cq, int cq_size, uint64_t *q_mmap_key,
		      uint64_t *q_mmap_size, uint32_t *cqn)
{
	struct efa_context *ctx = container_of(cq->domain->ctx, struct efa_context, ibv_ctx);
	struct efa_create_cq cmd;
	struct efa_create_cq_resp resp;
	int err;

	memset(&cmd, 0, sizeof(struct efa_create_cq));
	cmd.efa_cmd.num_sub_cqs   = ctx->sub_cqs_per_cq;
	cmd.efa_cmd.cq_entry_size = ctx->cqe_size;
	err = efa_ib_cmd_create_cq(&ctx->ibv_ctx, cq_size,
				   &cq->ibv_cq, &cmd.ibv_cmd, sizeof(cmd),
				   &resp.ibv_resp, sizeof(resp));
	if (err) {
		EFA_WARN_ERRNO(FI_LOG_CQ, "Command failed to create cq", err);
		return err;
	}

	*q_mmap_size = resp.efa_resp.q_mmap_size;
	*q_mmap_key = resp.efa_resp.q_mmap_key;
	*cqn = resp.efa_resp.cq_idx;

	cq->ibv_cq.context = &ctx->ibv_ctx;
	cq->ibv_cq.cq_context = cq;
	cq->ibv_cq.comp_events_completed  = 0;
	cq->ibv_cq.async_events_completed = 0;
	pthread_mutex_init(&cq->ibv_cq.mutex, NULL);
	pthread_cond_init(&cq->ibv_cq.cond, NULL);

	return 0;
}

/* context->mutex must be held */
int efa_cmd_destroy_cq(struct efa_cq *cq)
{
	return efa_ib_cmd_destroy_cq(&cq->ibv_cq);
}

int efa_cmd_create_qp(struct efa_qp *qp, struct efa_pd *pd, struct ibv_qp_init_attr *init_attr,
		      uint32_t srd_qp, struct efa_create_qp_resp *resp)
{
	struct ibv_pd *ibpd = &pd->ibv_pd;
	struct efa_create_qp cmd;
	int err;

	init_attr->cap.max_send_wr = qp->sq.wq.wqe_cnt;
	init_attr->cap.max_recv_wr = qp->rq.wq.wqe_cnt;

	memset(&cmd, 0, sizeof(struct efa_create_qp));
	cmd.efa_cmd.rq_ring_size = (qp->rq.wq.desc_mask + 1) *
		sizeof(struct efa_io_rx_desc);
	cmd.efa_cmd.sq_ring_size = (qp->sq.wq.desc_mask + 1) *
		sizeof(struct efa_io_tx_wqe);
	cmd.efa_cmd.driver_qp_type = EFA_QP_DRIVER_TYPE_SRD; /* ignored on UD */
	err = efa_ib_cmd_create_qp(ibpd, &qp->ibv_qp, init_attr,
				   &cmd.ibv_cmd, sizeof(cmd),
				   &resp->ibv_resp, sizeof(*resp));
	if (err)
		return err;

	qp->ibv_qp.context	= ibpd->context;
	qp->ibv_qp.qp_context	= init_attr->qp_context;
	qp->ibv_qp.pd		= ibpd;
	qp->ibv_qp.send_cq	= init_attr->send_cq;
	qp->ibv_qp.recv_cq	= init_attr->recv_cq;
	qp->ibv_qp.srq		= init_attr->srq;
	qp->ibv_qp.qp_type	= init_attr->qp_type;
	qp->ibv_qp.state	= IBV_QPS_RESET;
	qp->ibv_qp.events_completed = 0;
	pthread_mutex_init(&qp->ibv_qp.mutex, NULL);
	pthread_cond_init(&qp->ibv_qp.cond, NULL);

	return 0;
}

int efa_cmd_destroy_qp(struct efa_qp *qp)
{
	return efa_ib_cmd_destroy_qp(&qp->ibv_qp);
}

int efa_cmd_query_gid(struct efa_context *ctx, uint8_t port_num,
		      int index, union ibv_gid *gid)
{
	struct ibv_context *context = &ctx->ibv_ctx;
	char name[24];
	char attr[41];
	uint16_t val;
	int i;

	snprintf(name, sizeof(name), "ports/%d/gids/%d", port_num, index);

	if (fi_read_file(context->device->ibdev_path, name,
			 attr, sizeof(attr)) < 0)
		return -1;

	for (i = 0; i < 8; ++i) {
		if (sscanf(attr + i * 5, "%hx", &val) != 1)
			return -1;
		gid->raw[i * 2] = val >> 8;
		gid->raw[i * 2 + 1] = val & 0xff;
	}

	return 0;
}

static int efa_everbs_cmd_get_ah(struct efa_context *ctx, struct efa_ah *efa_ah, struct ibv_pd *pd,
				 struct ibv_ah_attr *attr)
{
	struct efa_everbs_get_ah_resp resp;
	struct efa_everbs_get_ah cmd = {};

	cmd.command		= EFA_EVERBS_CMD_GET_AH;
	cmd.in_words		= sizeof(cmd) / 4;
	cmd.out_words		= sizeof(resp) / 4;
	cmd.response		= (uintptr_t)&resp;

	cmd.user_handle		   = (uintptr_t)&efa_ah->ibv_ah;
	cmd.pdn			   = to_efa_pd(pd)->pdn;
	memcpy(cmd.gid, attr->grh.dgid.raw, 16);

	if (write(ctx->efa_everbs_cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));
	efa_ah->efa_address_handle = resp.efa_address_handle;

	return 0;
}

struct efa_ah *efa_cmd_create_ah(struct efa_pd *pd, struct ibv_ah_attr *attr)
{
	struct efa_context *ctx = pd->context;
	struct efa_create_ah_resp resp = {};
	struct ibv_port_attr port_attr;
	struct efa_ah *ah;
	int err;

	err = efa_cmd_query_port(ctx, attr->port_num, &port_attr);
	if (err) {
		EFA_WARN_ERRNO(FI_LOG_AV, "Command failed to query port", err);
		return NULL;
	}

	ah = malloc(sizeof(*ah));
	if (!ah) {
		EFA_WARN(FI_LOG_AV, "Failed to allocate memory for AH\n");
		return NULL;
	}

	attr->is_global  = 1;

	err = efa_ib_cmd_create_ah(&pd->ibv_pd, &ah->ibv_ah, attr,
				   &resp.ibv_resp, sizeof(resp));
	if (err) {
		EFA_WARN_ERRNO(FI_LOG_AV, "Command failed to create ah", err);
		goto err_free_ah;
	}

	if (ctx->cmds_supp_udata & EFA_USER_CMDS_SUPP_UDATA_CREATE_AH) {
		ah->efa_address_handle = resp.efa_resp.efa_address_handle;
	} else {
		err = efa_everbs_cmd_get_ah(ctx, ah, &pd->ibv_pd, attr);
		if (err) {
			EFA_WARN_ERRNO(FI_LOG_AV, "Command failed to get ah attrs", err);
			goto err_destroy_ah;
		}
	}

	return ah;

err_destroy_ah:
	efa_ib_cmd_destroy_ah(&ah->ibv_ah);
err_free_ah:
	free(ah);
	return NULL;
}

int efa_cmd_destroy_ah(struct efa_ah *ah)
{
	int ret;

	ret = efa_ib_cmd_destroy_ah(&ah->ibv_ah);
	free(ah);

	return ret;
}
