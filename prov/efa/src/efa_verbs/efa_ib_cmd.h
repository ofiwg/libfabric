/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2005, 2006 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 * Copyright (c) 2017-2019 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#ifndef EFA_IB_CMD_H_
#define EFA_IB_CMD_H_

#include "infiniband/efa_kern-abi.h"

struct ibv_get_context {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_get_context ibcmd;
};

struct ibv_query_device {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_query_device ibcmd;
};

struct ibv_ex_query_device {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_ex_cmd_hdr ex_hdr;
	struct ib_uverbs_ex_query_device ibcmd;
};

struct ibv_query_port {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_query_port ibcmd;
};

struct ibv_alloc_pd {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_alloc_pd ibcmd;
};

struct ibv_dealloc_pd {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_dealloc_pd ibcmd;
};

struct ibv_reg_mr {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_reg_mr ibcmd;
};

struct ibv_dereg_mr {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_dereg_mr ibcmd;
};

struct ibv_create_cq {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_create_cq ibcmd;
};

struct ibv_destroy_cq {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_destroy_cq ibcmd;
};

struct ibv_create_qp {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_create_qp ibcmd;
};

struct ibv_destroy_qp {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_destroy_qp ibcmd;
};

struct ibv_create_ah {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_create_ah ibcmd;
};

struct ibv_destroy_ah {
	struct ib_uverbs_cmd_hdr hdr;
	struct ib_uverbs_destroy_ah ibcmd;
};

int efa_ib_cmd_get_context(struct ibv_context *context, struct ibv_get_context *cmd,
			   size_t cmd_size, struct ib_uverbs_get_context_resp *resp,
			   size_t resp_size);
int efa_ib_cmd_query_device(struct ibv_context *context,
			    struct ibv_device_attr *device_attr,
			    uint64_t *raw_fw_ver,
			    struct ibv_query_device *cmd, size_t cmd_size);
int efa_ib_cmd_query_device_ex(struct ibv_context *context,
			       struct ibv_device_attr *device_attr,
			       uint64_t *raw_fw_ver,
			       struct ibv_ex_query_device *cmd,
			       size_t cmd_core_size,
			       size_t cmd_size,
			       struct ib_uverbs_ex_query_device_resp *resp,
			       size_t resp_core_size,
			       size_t resp_size);
int efa_ib_cmd_query_port(struct ibv_context *context, uint8_t port_num,
			  struct ibv_port_attr *port_attr,
			  struct ibv_query_port *cmd, size_t cmd_size);
int efa_ib_cmd_alloc_pd(struct ibv_context *context, struct ibv_pd *pd,
			struct ibv_alloc_pd *cmd, size_t cmd_size,
			struct ib_uverbs_alloc_pd_resp *resp, size_t resp_size);
int efa_ib_cmd_dealloc_pd(struct ibv_pd *pd);
int efa_ib_cmd_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
		      uint64_t hca_va, int access,
		      struct ibv_mr *mr, struct ibv_reg_mr *cmd,
		      size_t cmd_size,
		      struct ib_uverbs_reg_mr_resp *resp, size_t resp_size);
int efa_ib_cmd_dereg_mr(struct ibv_mr *mr);
int efa_ib_cmd_create_cq(struct ibv_context *context, int cqe,
			 struct ibv_cq *cq,
			 struct ibv_create_cq *cmd, size_t cmd_size,
			 struct ib_uverbs_create_cq_resp *resp, size_t resp_size);
int efa_ib_cmd_destroy_cq(struct ibv_cq *cq);
int efa_ib_cmd_create_qp(struct ibv_pd *pd,
			 struct ibv_qp *qp, struct ibv_qp_init_attr *attr,
			 struct ibv_create_qp *cmd, size_t cmd_size,
			 struct ib_uverbs_create_qp_resp *resp, size_t resp_size);
int efa_ib_cmd_destroy_qp(struct ibv_qp *qp);
int efa_ib_cmd_create_ah(struct ibv_pd *pd, struct ibv_ah *ah,
			 struct ibv_ah_attr *attr,
			 struct ib_uverbs_create_ah_resp *resp,
			 size_t resp_size);
int efa_ib_cmd_destroy_ah(struct ibv_ah *ah);

#endif /* EFA_IB_CMD_H_ */
