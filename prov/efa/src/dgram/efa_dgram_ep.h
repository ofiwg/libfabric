
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

#include "efa_base_ep.h"

#ifndef EFA_DGRAM_H
#define EFA_DGRAM_H

struct efa_dgram_ep {
	struct efa_base_ep base_ep;

	struct efa_dgram_cq	*rcq;
	struct efa_dgram_cq	*scq;

	struct ofi_bufpool	*send_wr_pool;
	struct ofi_bufpool	*recv_wr_pool;
};

struct efa_send_wr {
	/** @brief Work request struct used by rdma-core */
	struct ibv_send_wr wr;

	/** @brief Scatter gather element array
	 *
	 * @details
	 * EFA device supports a maximum of 2 iov/SGE
	 */
	struct ibv_sge sge[2];
};

struct efa_recv_wr {
	/** @brief Work request struct used by rdma-core */
	struct ibv_recv_wr wr;

	/** @brief Scatter gather element array
	 *
	 * @details
	 * EFA device supports a maximum of 2 iov/SGE
	 * For receive, we only use 1 SGE
	 */
	struct ibv_sge sge[1];
};


int efa_dgram_ep_open(struct fid_domain *domain_fid, struct fi_info *info,
		      struct fid_ep **ep_fid, void *context);

extern struct fi_ops_msg efa_dgram_ep_msg_ops;
extern struct fi_ops_rma efa_dgram_ep_rma_ops;
#endif
