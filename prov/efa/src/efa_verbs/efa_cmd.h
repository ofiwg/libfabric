/*
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

#ifndef _EFA_CMD_H_
#define _EFA_CMD_H_

#include "efa-abi.h"
#include "efa_ib_cmd.h"
#include "efa.h"

struct efa_alloc_ucontext_resp {
	struct ib_uverbs_get_context_resp ibv_resp;
	struct efa_ibv_alloc_ucontext_resp efa_resp;
};

struct efa_ex_query_device_resp {
	struct ib_uverbs_ex_query_device_resp ibv_resp;
	struct efa_ibv_ex_query_device_resp efa_resp;
};

struct efa_alloc_pd_resp {
	struct ib_uverbs_alloc_pd_resp ibv_resp;
	struct efa_ibv_alloc_pd_resp efa_resp;
};

struct efa_create_cq {
	struct ibv_create_cq ibv_cmd;
	struct efa_ibv_create_cq efa_cmd;
};

struct efa_create_cq_resp {
	struct ib_uverbs_create_cq_resp ibv_resp;
	struct efa_ibv_create_cq_resp efa_resp;
};

struct efa_create_qp {
	struct ibv_create_qp ibv_cmd;
	struct efa_ibv_create_qp efa_cmd;
};

struct efa_create_qp_resp {
	struct ib_uverbs_create_qp_resp ibv_resp;
	struct efa_ibv_create_qp_resp efa_resp;
};

struct efa_create_ah_resp {
	struct ib_uverbs_create_ah_resp ibv_resp;
	struct efa_ibv_create_ah_resp efa_resp;
};

int efa_cmd_alloc_ucontext(struct ibv_device *device, struct efa_context *ctx, int cmd_fd);
int efa_cmd_query_device(struct efa_context *ctx, struct efa_device_attr *attr);
int efa_cmd_query_port(struct efa_context *ctx, uint8_t port, struct ibv_port_attr *attr);
struct efa_pd *efa_cmd_alloc_pd(struct efa_context *ctx);
int efa_cmd_dealloc_pd(struct efa_pd *pd);
struct ibv_mr *efa_cmd_reg_mr(struct efa_pd *pd, void *addr,
			      size_t length, int access);
int efa_cmd_dereg_mr(struct ibv_mr *mr);
int efa_cmd_create_cq(struct efa_cq *cq, int cq_size, uint64_t *q_mmap_key,
		      uint64_t *q_mmap_size, uint32_t *cqn);
int efa_cmd_destroy_cq(struct efa_cq *cq);
int efa_cmd_create_qp(struct efa_qp *qp, struct efa_pd *pd, struct ibv_qp_init_attr *init_attr,
		      uint32_t srd_qp, struct efa_create_qp_resp *resp);
int efa_cmd_destroy_qp(struct efa_qp *qp);
int efa_cmd_query_gid(struct efa_context *ctx, uint8_t port_num,
		      int index, union ibv_gid *gid);
struct efa_ah *efa_cmd_create_ah(struct efa_pd *pd, struct ibv_ah_attr *attr);
int efa_cmd_destroy_ah(struct efa_ah *ah);

#endif /* _EFA_CMD_H_ */
