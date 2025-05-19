/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _FI_EXT_EFA_H_
#define _FI_EXT_EFA_H_

#include <rdma/fi_domain.h>

#define FI_EFA_DOMAIN_OPS "efa domain ops"

struct fi_efa_mr_attr {
    uint16_t ic_id_validity;
    uint16_t recv_ic_id;
    uint16_t rdma_read_ic_id;
    uint16_t rdma_recv_ic_id;
};

enum {
    FI_EFA_MR_ATTR_RECV_IC_ID = 1 << 0,
    FI_EFA_MR_ATTR_RDMA_READ_IC_ID = 1 << 1,
    FI_EFA_MR_ATTR_RDMA_RECV_IC_ID = 1 << 2,
};

struct fi_efa_wq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
    uint32_t *doorbell;
    uint32_t max_batch;
};

struct fi_efa_cq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
};

struct fi_efa_ops_domain {
	int (*query_mr)(struct fid_mr *mr, struct fi_efa_mr_attr *mr_attr);
	int (*query_addr)(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
			  uint16_t *remote_qpn, uint32_t *remote_qkey);
	int (*query_qp_wqs)(struct fid_ep *ep_fid,
			    struct fi_efa_wq_attr *sq_attr,
			    struct fi_efa_wq_attr *rq_attr);
	int (*query_cq)(struct fid_cq *cq_fid, struct fi_efa_cq_attr *cq_attr);
};

#endif /* _FI_EXT_EFA_H_ */
