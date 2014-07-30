/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005, 2006 Cisco Systems.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2013 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _FI_UVERBS_H_
#define _FI_UVERBS_H_


#include <linux/types.h>
#include <rdma/fabric.h>
#include <infiniband/kern-abi.h>


#ifdef __cplusplus
extern "C" {
#endif


struct fi_ops_uverbs {
	int	(*get_context)(fid_t fid,
				struct ibv_get_context *cmd, size_t cmd_size,
				struct ibv_get_context_resp *resp, size_t resp_size);
	int	(*query_device)(fid_t fid,
				struct ibv_query_device *cmd, size_t cmd_size,
				struct ibv_query_device_resp *resp, size_t resp_size);
	int	(*query_port)(fid_t fid,
				struct ibv_query_port *cmd, size_t cmd_size,
				struct ibv_query_port_resp *resp, size_t resp_size);
	int	(*alloc_pd)(fid_t fid,
				struct ibv_alloc_pd *cmd, size_t cmd_size,
				struct ibv_alloc_pd_resp *resp, size_t resp_size);
	int	(*dealloc_pd)(fid_t fid,
				struct ibv_dealloc_pd *cmd, size_t cmd_size);
	int	(*create_ah)(fid_t fid,
				struct ibv_create_ah *cmd, size_t cmd_size,
				struct ibv_create_ah_resp *resp, size_t resp_size);
	int	(*destroy_ah)(fid_t fid,
				struct ibv_destroy_ah *cmd, size_t cmd_size);
	int	(*open_xrcd)(fid_t fid,
				struct ibv_open_xrcd *cmd, size_t cmd_size,
				struct ibv_open_xrcd_resp *resp, size_t resp_size);
	int	(*close_xrcd)(fid_t fid,
				struct ibv_close_xrcd *cmd, size_t cmd_size);
	int	(*reg_mr)(fid_t fid,
				struct ibv_reg_mr *cmd, size_t cmd_size,
				struct ibv_reg_mr_resp *resp, size_t resp_size);
	int	(*dereg_mr)(fid_t fid,
				struct ibv_dereg_mr *cd, size_t cmd_size);
	int	(*create_comp_channel)(fid_t fid,
				struct ibv_create_comp_channel *cmd, size_t cmd_size,
				struct ibv_create_comp_channel_resp *resp, size_t resp_size);
	int	(*create_cq)(fid_t fid,
				struct ibv_create_cq *cmd, size_t cmd_size,
				struct ibv_create_cq_resp *resp, size_t resp_size);
	int	(*poll_cq)(fid_t fid,
				struct ibv_poll_cq *cmd, size_t cmd_size,
				struct ibv_poll_cq_resp *resp, size_t resp_size);
	int	(*req_notify_cq)(fid_t fid,
				struct ibv_req_notify_cq *cmd, size_t cmd_size);
	int	(*resize_cq)(fid_t fid,
				struct ibv_resize_cq *cmd, size_t cmd_size,
				struct ibv_resize_cq_resp *resp, size_t resp_size);
	int	(*destroy_cq)(fid_t fid,
				struct ibv_destroy_cq *cmd, size_t cmd_size,
				struct ibv_destroy_cq_resp *resp, size_t resp_size);
	int	(*create_srq)(fid_t fid,
				struct ibv_create_srq *cmd, size_t cmd_size,
				struct ibv_create_srq_resp *resp, size_t resp_size);
	int	(*modify_srq)(fid_t fid,
				struct ibv_modify_srq *cmd, size_t cmd_size);
	int	(*query_srq)(fid_t fid,
				struct ibv_query_srq *cmd, size_t cmd_size,
				struct ibv_query_srq_resp *resp, size_t resp_size);
	int	(*destroy_srq)(fid_t fid,
				struct ibv_destroy_srq *cmd, size_t cmd_size,
				struct ibv_destroy_srq_resp *resp, size_t resp_size);
	int	(*create_qp)(fid_t fid,
				struct ibv_create_qp *cmd, size_t cmd_size,
				struct ibv_create_qp_resp *resp, size_t resp_size);
	int	(*open_qp)(fid_t fid,
				struct ibv_open_qp *cmd, size_t cmd_size,
				struct ibv_create_qp_resp *resp, size_t resp_size);
	int	(*query_qp)(fid_t fid,
				struct ibv_query_qp *cmd, size_t cmd_size,
				struct ibv_query_qp_resp *resp, size_t resp_size);
	int	(*modify_qp)(fid_t fid,
				struct ibv_modify_qp *cmd, size_t cmd_size);
	int	(*destroy_qp)(fid_t fid,
				struct ibv_destroy_qp *cmd, size_t cmd_size,
				struct ibv_destroy_qp_resp *resp, size_t resp_size);
	int	(*post_send)(fid_t fid,
				struct ibv_post_send *cmd, size_t cmd_size,
				struct ibv_post_send_resp *resp, size_t resp_size);
	int	(*post_recv)(fid_t fid,
				struct ibv_post_recv *cmd, size_t cmd_size,
				struct ibv_post_recv_resp *resp, size_t resp_size);
	int	(*post_srq_recv)(fid_t fid,
				struct ibv_post_srq_recv *cmd, size_t cmd_size,
				struct ibv_post_srq_recv_resp *resp, size_t resp_size);
	int	(*attach_mcast)(fid_t fid,
				struct ibv_attach_mcast *cmd, size_t cmd_size);
	int	(*detach_mcast)(fid_t fid,
				struct ibv_detach_mcast *cmd, size_t cmd_size);
};

struct fid_uverbs {
	struct fid		fid;
	int			fd;
	struct fi_ops_uverbs	*ops;
};

#define FI_UVERBS_INTERFACE	"uverbs"

static inline int
uv_get_context(fid_t fid,
	struct ibv_get_context *cmd, size_t cmd_size,
	struct ibv_get_context_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, get_context);
	return uv->ops->get_context(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_device(fid_t fid,
	struct ibv_query_device *cmd, size_t cmd_size,
	struct ibv_query_device_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_device);
	return uv->ops->query_device(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_port(fid_t fid,
	struct ibv_query_port *cmd, size_t cmd_size,
	struct ibv_query_port_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_port);
	return uv->ops->query_port(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_alloc_pd(fid_t fid,
	struct ibv_alloc_pd *cmd, size_t cmd_size,
	struct ibv_alloc_pd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, alloc_pd);
	return uv->ops->alloc_pd(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_dealloc_pd(fid_t fid,
	struct ibv_dealloc_pd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, dealloc_pd);
	return uv->ops->dealloc_pd(fid, cmd, cmd_size);
}

static inline int
uv_create_ah(fid_t fid,
	struct ibv_create_ah *cmd, size_t cmd_size,
	struct ibv_create_ah_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_ah);
	return uv->ops->create_ah(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_ah(fid_t fid,
	struct ibv_destroy_ah *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_ah);
	return uv->ops->destroy_ah(fid, cmd, cmd_size);
}

static inline int
uv_open_xrcd(fid_t fid,
	struct ibv_open_xrcd *cmd, size_t cmd_size,
	struct ibv_open_xrcd_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, open_xrcd);
	return uv->ops->open_xrcd(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_close_xrcd(fid_t fid,
	struct ibv_close_xrcd *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, close_xrcd);
	return uv->ops->close_xrcd(fid, cmd, cmd_size);
}

static inline int
uv_reg_mr(fid_t fid,
	struct ibv_reg_mr *cmd, size_t cmd_size,
	struct ibv_reg_mr_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, reg_mr);
	return uv->ops->reg_mr(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_dereg_mr(fid_t fid,
	struct ibv_dereg_mr *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, dereg_mr);
	return uv->ops->dereg_mr(fid, cmd, cmd_size);
}

static inline int
uv_create_comp_channel(fid_t fid,
	struct ibv_create_comp_channel *cmd, size_t cmd_size,
	struct ibv_create_comp_channel_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_comp_channel);
	return uv->ops->create_comp_channel(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_cq(fid_t fid,
	struct ibv_create_cq *cmd, size_t cmd_size,
	struct ibv_create_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_cq);
	return uv->ops->create_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_poll_cq(fid_t fid,
	struct ibv_poll_cq *cmd, size_t cmd_size,
	struct ibv_poll_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, poll_cq);
	return uv->ops->poll_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_req_notify_cq(fid_t fid,
	struct ibv_req_notify_cq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, req_notify_cq);
	return uv->ops->req_notify_cq(fid, cmd, cmd_size);
}

static inline int
uv_resize_cq(fid_t fid,
	struct ibv_resize_cq *cmd, size_t cmd_size,
	struct ibv_resize_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, resize_cq);
	return uv->ops->resize_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_cq(fid_t fid,
	struct ibv_destroy_cq *cmd, size_t cmd_size,
	struct ibv_destroy_cq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_cq);
	return uv->ops->destroy_cq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_srq(fid_t fid,
	struct ibv_create_srq *cmd, size_t cmd_size,
	struct ibv_create_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_srq);
	return uv->ops->create_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_modify_srq(fid_t fid,
	struct ibv_modify_srq *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, modify_srq);
	return uv->ops->modify_srq(fid, cmd, cmd_size);
}

static inline int
uv_query_srq(fid_t fid,
	struct ibv_query_srq *cmd, size_t cmd_size,
	struct ibv_query_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_srq);
	return uv->ops->query_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_destroy_srq(fid_t fid,
	struct ibv_destroy_srq *cmd, size_t cmd_size,
	struct ibv_destroy_srq_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_srq);
	return uv->ops->destroy_srq(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_create_qp(fid_t fid,
	struct ibv_create_qp *cmd, size_t cmd_size,
	struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, create_qp);
	return uv->ops->create_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_open_qp(fid_t fid,
	struct ibv_open_qp *cmd, size_t cmd_size,
	struct ibv_create_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, open_qp);
	return uv->ops->open_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_query_qp(fid_t fid,
	struct ibv_query_qp *cmd, size_t cmd_size,
	struct ibv_query_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, query_qp);
	return uv->ops->query_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_modify_qp(fid_t fid,
	struct ibv_modify_qp *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, modify_qp);
	return uv->ops->modify_qp(fid, cmd, cmd_size);
}

static inline int
uv_destroy_qp(fid_t fid,
	struct ibv_destroy_qp *cmd, size_t cmd_size,
	struct ibv_destroy_qp_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, destroy_qp);
	return uv->ops->destroy_qp(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_send(fid_t fid,
	struct ibv_post_send *cmd, size_t cmd_size,
	struct ibv_post_send_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_send);
	return uv->ops->post_send(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_recv(fid_t fid,
	struct ibv_post_recv *cmd, size_t cmd_size,
	struct ibv_post_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_recv);
	return uv->ops->post_recv(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_post_srq_recv(fid_t fid,
	struct ibv_post_srq_recv *cmd, size_t cmd_size,
	struct ibv_post_srq_recv_resp *resp, size_t resp_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, post_srq_recv);
	return uv->ops->post_srq_recv(fid, cmd, cmd_size, resp, resp_size);
}

static inline int
uv_attach_mcast(fid_t fid,
	struct ibv_attach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, attach_mcast);
	return uv->ops->attach_mcast(fid, cmd, cmd_size);
}

static inline int
uv_detach_mcast(fid_t fid,
		struct ibv_detach_mcast *cmd, size_t cmd_size)
{
	struct fid_uverbs *uv = container_of(fid, struct fid_uverbs, fid);
	FI_ASSERT_CLASS(fid, FID_CLASS_INTERFACE);
	FI_ASSERT_OPS(fid, struct fid_uverbs, ops);
	FI_ASSERT_OP(uv->ops, struct fi_ops_uverbs, detach_mcast);
	return uv->ops->detach_mcast(fid, cmd, cmd_size);
}


#ifdef __cplusplus
}
#endif

#endif /* _FI_UVERBS_H_ */
