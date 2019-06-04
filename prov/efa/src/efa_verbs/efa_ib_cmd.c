/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017-2018 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <alloca.h>
#include <string.h>

#include "efa_ib.h"
#include "efa_ib_cmd.h"

#define IBV_INIT_CMD(cmd, size, opcode)					\
	do {								\
		(cmd)->hdr.command = IB_USER_VERBS_CMD_##opcode;	\
		(cmd)->hdr.in_words  = (size) / 4;			\
		(cmd)->hdr.out_words = 0;				\
	} while (0)

#define IBV_INIT_CMD_RESP(cmd, size, opcode, out, outsize)		\
	do {								\
		(cmd)->hdr.command = IB_USER_VERBS_CMD_##opcode;	\
		(cmd)->hdr.in_words  = (size) / 4;			\
		(cmd)->hdr.out_words = (outsize) / 4;			\
		(cmd)->ibcmd.response  = (uintptr_t)(out);			\
	} while (0)

static inline uint32_t _cmd_ex(uint32_t cmd)
{
	return IB_USER_VERBS_CMD_FLAG_EXTENDED | cmd;
}

#define IBV_INIT_CMD_RESP_EX_V(cmd, cmd_size, size, opcode, out, resp_size,      \
		outsize)						         \
	do {                                                                     \
		size_t c_size = cmd_size - sizeof(struct ib_uverbs_cmd_hdr)      \
					 - sizeof(struct ib_uverbs_ex_cmd_hdr);  \
		(cmd)->hdr.command =					         \
			_cmd_ex(IB_USER_VERBS_EX_CMD_##opcode);		         \
		(cmd)->hdr.in_words  = ((c_size) / 8);                           \
		(cmd)->hdr.out_words = ((resp_size) / 8);                        \
		(cmd)->ex_hdr.response  = (uintptr_t)(out);                      \
		(cmd)->ex_hdr.provider_in_words   = (((size) - (cmd_size)) / 8); \
		(cmd)->ex_hdr.provider_out_words  =			         \
			     (((outsize) - (resp_size)) / 8);                    \
		(cmd)->ex_hdr.cmd_hdr_reserved = 0;				 \
	} while (0)

int efa_ib_cmd_get_context(struct ibv_context *context, struct ibv_get_context *cmd,
			   size_t cmd_size, struct ib_uverbs_get_context_resp *resp,
			   size_t resp_size)
{
	if (abi_ver < IB_USER_VERBS_MIN_ABI_VERSION)
		return -ENOSYS;

	IBV_INIT_CMD_RESP(cmd, cmd_size, GET_CONTEXT, resp, resp_size);

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	context->async_fd         = resp->async_fd;
	context->num_comp_vectors = resp->num_comp_vectors;

	return 0;
}

static void copy_query_dev_fields(struct ibv_device_attr *device_attr,
				  struct ib_uverbs_query_device_resp *resp,
				  uint64_t *raw_fw_ver)
{
	*raw_fw_ver				= resp->fw_ver;
	device_attr->node_guid			= resp->node_guid;
	device_attr->sys_image_guid		= resp->sys_image_guid;
	device_attr->max_mr_size		= resp->max_mr_size;
	device_attr->page_size_cap		= resp->page_size_cap;
	device_attr->vendor_id			= resp->vendor_id;
	device_attr->vendor_part_id		= resp->vendor_part_id;
	device_attr->hw_ver			= resp->hw_ver;
	device_attr->max_qp			= resp->max_qp;
	device_attr->max_qp_wr			= resp->max_qp_wr;
	device_attr->device_cap_flags		= resp->device_cap_flags;
	device_attr->max_sge			= resp->max_sge;
	device_attr->max_sge_rd			= resp->max_sge_rd;
	device_attr->max_cq			= resp->max_cq;
	device_attr->max_cqe			= resp->max_cqe;
	device_attr->max_mr			= resp->max_mr;
	device_attr->max_pd			= resp->max_pd;
	device_attr->max_qp_rd_atom		= resp->max_qp_rd_atom;
	device_attr->max_ee_rd_atom		= resp->max_ee_rd_atom;
	device_attr->max_res_rd_atom		= resp->max_res_rd_atom;
	device_attr->max_qp_init_rd_atom	= resp->max_qp_init_rd_atom;
	device_attr->max_ee_init_rd_atom	= resp->max_ee_init_rd_atom;
	device_attr->atomic_cap			= resp->atomic_cap;
	device_attr->max_ee			= resp->max_ee;
	device_attr->max_rdd			= resp->max_rdd;
	device_attr->max_mw			= resp->max_mw;
	device_attr->max_raw_ipv6_qp		= resp->max_raw_ipv6_qp;
	device_attr->max_raw_ethy_qp		= resp->max_raw_ethy_qp;
	device_attr->max_mcast_grp		= resp->max_mcast_grp;
	device_attr->max_mcast_qp_attach	= resp->max_mcast_qp_attach;
	device_attr->max_total_mcast_qp_attach	= resp->max_total_mcast_qp_attach;
	device_attr->max_ah			= resp->max_ah;
	device_attr->max_fmr			= resp->max_fmr;
	device_attr->max_map_per_fmr		= resp->max_map_per_fmr;
	device_attr->max_srq			= resp->max_srq;
	device_attr->max_srq_wr			= resp->max_srq_wr;
	device_attr->max_srq_sge		= resp->max_srq_sge;
	device_attr->max_pkeys			= resp->max_pkeys;
	device_attr->local_ca_ack_delay		= resp->local_ca_ack_delay;
	device_attr->phys_port_cnt		= resp->phys_port_cnt;
}

int efa_ib_cmd_query_device(struct ibv_context *context,
			    struct ibv_device_attr *device_attr,
			    uint64_t *raw_fw_ver,
			    struct ibv_query_device *cmd, size_t cmd_size)
{
	struct ib_uverbs_query_device_resp resp;

	IBV_INIT_CMD_RESP(cmd, cmd_size, QUERY_DEVICE, &resp, sizeof(resp));

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));

	memset(device_attr->fw_ver, 0, sizeof(device_attr->fw_ver));
	copy_query_dev_fields(device_attr, &resp, raw_fw_ver);

	return 0;
}

int efa_ib_cmd_query_device_ex(struct ibv_context *context,
			       struct ibv_device_attr *device_attr,
			       uint64_t *raw_fw_ver,
			       struct ibv_ex_query_device *cmd,
			       size_t cmd_core_size,
			       size_t cmd_size,
			       struct ib_uverbs_ex_query_device_resp *resp,
			       size_t resp_core_size,
			       size_t resp_size)
{
	if (resp_core_size < offsetof(struct ib_uverbs_ex_query_device_resp,
				      response_length) +
			     sizeof(resp->response_length))
		return -EINVAL;

	IBV_INIT_CMD_RESP_EX_V(cmd, cmd_core_size, cmd_size,
			       QUERY_DEVICE, resp, resp_core_size,
			       resp_size);
	cmd->ibcmd.comp_mask = 0;
	cmd->ibcmd.reserved = 0;

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	memset(device_attr->fw_ver, 0, sizeof(device_attr->fw_ver));
	copy_query_dev_fields(device_attr, &resp->base, raw_fw_ver);

	return 0;
}

int efa_ib_cmd_query_port(struct ibv_context *context, uint8_t port_num,
			  struct ibv_port_attr *port_attr,
			  struct ibv_query_port *cmd, size_t cmd_size)
{
	struct ib_uverbs_query_port_resp resp;

	IBV_INIT_CMD_RESP(cmd, cmd_size, QUERY_PORT, &resp, sizeof(resp));
	cmd->ibcmd.port_num = port_num;
	memset(cmd->ibcmd.reserved, 0, sizeof(cmd->ibcmd.reserved));

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));

	port_attr->state	   = resp.state;
	port_attr->max_mtu         = resp.max_mtu;
	port_attr->active_mtu      = resp.active_mtu;
	port_attr->gid_tbl_len     = resp.gid_tbl_len;
	port_attr->port_cap_flags  = resp.port_cap_flags;
	port_attr->max_msg_sz      = resp.max_msg_sz;
	port_attr->bad_pkey_cntr   = resp.bad_pkey_cntr;
	port_attr->qkey_viol_cntr  = resp.qkey_viol_cntr;
	port_attr->pkey_tbl_len    = resp.pkey_tbl_len;
	port_attr->lid		   = resp.lid;
	port_attr->sm_lid	   = resp.sm_lid;
	port_attr->lmc		   = resp.lmc;
	port_attr->max_vl_num      = resp.max_vl_num;
	port_attr->sm_sl	   = resp.sm_sl;
	port_attr->subnet_timeout  = resp.subnet_timeout;
	port_attr->init_type_reply = resp.init_type_reply;
	port_attr->active_width    = resp.active_width;
	port_attr->active_speed    = resp.active_speed;
	port_attr->phys_state      = resp.phys_state;
	port_attr->link_layer      = resp.link_layer;

	return 0;
}

int efa_ib_cmd_alloc_pd(struct ibv_context *context, struct ibv_pd *pd,
			struct ibv_alloc_pd *cmd, size_t cmd_size,
			struct ib_uverbs_alloc_pd_resp *resp, size_t resp_size)
{
	IBV_INIT_CMD_RESP(cmd, cmd_size, ALLOC_PD, resp, resp_size);

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	pd->handle  = resp->pd_handle;
	pd->context = context;

	return 0;
}

int efa_ib_cmd_dealloc_pd(struct ibv_pd *pd)
{
	struct ibv_dealloc_pd cmd;

	IBV_INIT_CMD(&cmd, sizeof(cmd), DEALLOC_PD);
	cmd.ibcmd.pd_handle = pd->handle;

	if (write(pd->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	return 0;
}

int efa_ib_cmd_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
		      uint64_t hca_va, int access,
		      struct ibv_mr *mr, struct ibv_reg_mr *cmd,
		      size_t cmd_size,
		      struct ib_uverbs_reg_mr_resp *resp, size_t resp_size)
{
	IBV_INIT_CMD_RESP(cmd, cmd_size, REG_MR, resp, resp_size);

	cmd->ibcmd.start	  = (uintptr_t)addr;
	cmd->ibcmd.length	  = length;
	cmd->ibcmd.hca_va	  = hca_va;
	cmd->ibcmd.pd_handle	  = pd->handle;
	cmd->ibcmd.access_flags = access;

	if (write(pd->context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	mr->handle  = resp->mr_handle;
	mr->lkey    = resp->lkey;
	mr->rkey    = resp->rkey;
	mr->context = pd->context;

	return 0;
}

int efa_ib_cmd_dereg_mr(struct ibv_mr *mr)
{
	struct ibv_dereg_mr cmd;

	IBV_INIT_CMD(&cmd, sizeof(cmd), DEREG_MR);
	cmd.ibcmd.mr_handle = mr->handle;

	if (write(mr->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	return 0;
}

int efa_ib_cmd_create_cq(struct ibv_context *context, int cqe,
			 struct ibv_cq *cq,
			 struct ibv_create_cq *cmd, size_t cmd_size,
			 struct ib_uverbs_create_cq_resp *resp, size_t resp_size)
{
	IBV_INIT_CMD_RESP(cmd, cmd_size, CREATE_CQ, resp, resp_size);
	cmd->ibcmd.user_handle   = (uintptr_t)cq;
	cmd->ibcmd.cqe           = cqe;
	cmd->ibcmd.comp_vector   = 0;
	cmd->ibcmd.comp_channel  = -1;
	cmd->ibcmd.reserved      = 0;

	if (write(context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	cq->handle  = resp->cq_handle;
	cq->cqe     = resp->cqe;
	cq->context = context;

	return 0;
}

int efa_ib_cmd_destroy_cq(struct ibv_cq *cq)
{
	struct ibv_destroy_cq cmd;
	struct ib_uverbs_destroy_cq_resp resp;

	IBV_INIT_CMD_RESP(&cmd, sizeof(cmd), DESTROY_CQ, &resp, sizeof(resp));
	cmd.ibcmd.cq_handle = cq->handle;
	cmd.ibcmd.reserved  = 0;

	if (write(cq->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));

	pthread_mutex_lock(&cq->mutex);
	while (cq->comp_events_completed  != resp.comp_events_reported ||
	       cq->async_events_completed != resp.async_events_reported)
		pthread_cond_wait(&cq->cond, &cq->mutex);
	pthread_mutex_unlock(&cq->mutex);

	return 0;
}

int efa_ib_cmd_create_qp(struct ibv_pd *pd,
			 struct ibv_qp *qp, struct ibv_qp_init_attr *attr,
			 struct ibv_create_qp *cmd, size_t cmd_size,
			 struct ib_uverbs_create_qp_resp *resp, size_t resp_size)
{
	IBV_INIT_CMD_RESP(cmd, cmd_size, CREATE_QP, resp, resp_size);

	cmd->ibcmd.user_handle     = (uintptr_t)qp;
	cmd->ibcmd.pd_handle       = pd->handle;
	cmd->ibcmd.send_cq_handle  = attr->send_cq->handle;
	cmd->ibcmd.recv_cq_handle  = attr->recv_cq->handle;
	cmd->ibcmd.srq_handle      = attr->srq ? attr->srq->handle : 0;
	cmd->ibcmd.max_send_wr     = attr->cap.max_send_wr;
	cmd->ibcmd.max_recv_wr     = attr->cap.max_recv_wr;
	cmd->ibcmd.max_send_sge    = attr->cap.max_send_sge;
	cmd->ibcmd.max_recv_sge    = attr->cap.max_recv_sge;
	cmd->ibcmd.max_inline_data = attr->cap.max_inline_data;
	cmd->ibcmd.sq_sig_all      = attr->sq_sig_all;
	cmd->ibcmd.qp_type         = attr->qp_type;
	cmd->ibcmd.is_srq          = !!attr->srq;
	cmd->ibcmd.reserved        = 0;

	if (write(pd->context->cmd_fd, cmd, cmd_size) != cmd_size)
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(resp, resp_size);

	qp->handle		  = resp->qp_handle;
	qp->qp_num		  = resp->qpn;
	qp->context		  = pd->context;

	attr->cap.max_recv_sge    = resp->max_recv_sge;
	attr->cap.max_send_sge    = resp->max_send_sge;
	attr->cap.max_recv_wr     = resp->max_recv_wr;
	attr->cap.max_send_wr     = resp->max_send_wr;
	attr->cap.max_inline_data = resp->max_inline_data;

	return 0;
}

int efa_ib_cmd_destroy_qp(struct ibv_qp *qp)
{
	struct ibv_destroy_qp cmd;
	struct ib_uverbs_destroy_qp_resp resp;

	IBV_INIT_CMD_RESP(&cmd, sizeof(cmd), DESTROY_QP, &resp, sizeof(resp));
	cmd.ibcmd.qp_handle = qp->handle;
	cmd.ibcmd.reserved  = 0;

	if (write(qp->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, sizeof(resp));

	pthread_mutex_lock(&qp->mutex);
	while (qp->events_completed != resp.events_reported)
		pthread_cond_wait(&qp->cond, &qp->mutex);
	pthread_mutex_unlock(&qp->mutex);

	return 0;
}

int efa_ib_cmd_create_ah(struct ibv_pd *pd, struct ibv_ah *ah,
			 struct ibv_ah_attr *attr,
			 struct ib_uverbs_create_ah_resp *resp,
			 size_t resp_size)
{
	struct ibv_create_ah cmd;

	IBV_INIT_CMD_RESP(&cmd, sizeof(cmd), CREATE_AH, resp, resp_size);
	cmd.ibcmd.user_handle            = (uintptr_t)ah;
	cmd.ibcmd.pd_handle              = pd->handle;
	cmd.ibcmd.reserved		   = 0;
	cmd.ibcmd.attr.dlid              = attr->dlid;
	cmd.ibcmd.attr.sl                = attr->sl;
	cmd.ibcmd.attr.src_path_bits     = attr->src_path_bits;
	cmd.ibcmd.attr.static_rate       = attr->static_rate;
	cmd.ibcmd.attr.is_global         = attr->is_global;
	cmd.ibcmd.attr.port_num          = attr->port_num;
	cmd.ibcmd.attr.reserved	   = 0;
	cmd.ibcmd.attr.grh.flow_label    = attr->grh.flow_label;
	cmd.ibcmd.attr.grh.sgid_index    = attr->grh.sgid_index;
	cmd.ibcmd.attr.grh.hop_limit     = attr->grh.hop_limit;
	cmd.ibcmd.attr.grh.traffic_class = attr->grh.traffic_class;
	cmd.ibcmd.attr.grh.reserved	   = 0;
	memcpy(cmd.ibcmd.attr.grh.dgid, attr->grh.dgid.raw, 16);

	if (write(pd->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	VALGRIND_MAKE_MEM_DEFINED(&resp, resp_size);

	ah->handle  = resp->ah_handle;
	ah->context = pd->context;

	return 0;
}

int efa_ib_cmd_destroy_ah(struct ibv_ah *ah)
{
	struct ibv_destroy_ah cmd;

	IBV_INIT_CMD(&cmd, sizeof(cmd), DESTROY_AH);
	cmd.ibcmd.ah_handle = ah->handle;

	if (write(ah->context->cmd_fd, &cmd, sizeof(cmd)) != sizeof(cmd))
		return -errno;

	return 0;
}
