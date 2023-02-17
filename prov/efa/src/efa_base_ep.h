/*
 * Copyright (c) 2018-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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

#ifndef EFA_BASE_EP_H
#define EFA_BASE_EP_H

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>

#include "ofi.h"
#include "ofi_util.h"
#include "rdm/rdm_proto_v4.h"

struct efa_qp {
	struct ibv_qp *ibv_qp;
	struct ibv_qp_ex *ibv_qp_ex;
	struct efa_base_ep *base_ep;
	uint32_t qp_num;
	uint32_t qkey;
};

struct efa_av;

struct efa_base_ep {
	struct util_ep util_ep;
	struct efa_domain *domain;
	struct efa_qp *qp;
	struct efa_av *av;
	struct fi_info *info;
	size_t rnr_retry;
	struct efa_ep_addr src_addr;
	struct ibv_ah *self_ah;

	bool util_ep_initialized;
	bool efa_qp_enabled;

	struct ibv_send_wr xmit_more_wr_head;
	struct ibv_send_wr *xmit_more_wr_tail;
	struct ibv_recv_wr recv_more_wr_head;
	struct ibv_recv_wr *recv_more_wr_tail;
};

int efa_base_ep_bind_av(struct efa_base_ep *base_ep, struct efa_av *av);

int efa_base_ep_destruct(struct efa_base_ep *base_ep);

int efa_base_ep_enable(struct efa_base_ep *base_ep);

int efa_base_ep_construct(struct efa_base_ep *base_ep,
			  struct fid_domain* domain_fid,
			  struct fi_info *info,
			  ofi_ep_progress_func progress,
			  void *context);

int efa_base_ep_getname(fid_t fid, void *addr, size_t *addrlen);

int efa_base_ep_create_qp(struct efa_base_ep *base_ep,
			  struct ibv_qp_init_attr_ex *init_attr_ex);

bool efa_base_ep_support_op_in_order_aligned_128_bytes(struct efa_base_ep *base_ep,
						       enum ibv_wr_opcode op);

#endif
