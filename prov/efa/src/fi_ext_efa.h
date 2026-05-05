/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _FI_EXT_EFA_H_
#define _FI_EXT_EFA_H_

#include <stdbool.h>
#include <rdma/fi_domain.h>

#define FI_EFA_DOMAIN_OPS "efa domain ops"
#define FI_EFA_GDA_OPS "efa gda ops"
#define FI_EFA_FEATURE_OPS "efa feature ops"

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

enum {
    FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF = 1 << 0,
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

struct fi_efa_cq_init_attr {
	uint64_t flags;
	struct {
		uint8_t  *buffer;
		uint64_t length;
		uint64_t offset;
		uint32_t fd;
	} ext_mem_dmabuf;
};

/* mirror efadv_memory_location_type */
enum fi_efa_memory_location_type {
	FI_EFA_MEMORY_LOCATION_VA,
	FI_EFA_MEMORY_LOCATION_DMABUF,
};

/* mirror efadv_memory_location */
struct fi_efa_memory_location {
	uint8_t *ptr;
	struct {
		uint64_t offset;
		int32_t fd;
		uint32_t reserved;
	} dmabuf;
	uint8_t type; /* Use fi_efa_memory_location_type */
	uint8_t reserved[7];
};

enum {
	FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM = 1 << 0,
	FI_EFA_COMP_CNTR_INIT_WITH_ERR_EXTERNAL_MEM = 1 << 1,
};

/* mirror efadv_comp_cntr_init_attr */
struct fi_efa_comp_cntr_init_attr {
	uint64_t comp_mask;
	uint32_t flags;
	uint32_t reserved;
	struct fi_efa_memory_location comp_cntr_ext_mem;
	struct fi_efa_memory_location err_cntr_ext_mem;
};

struct fi_efa_ops_domain {
	int (*query_mr)(struct fid_mr *mr, struct fi_efa_mr_attr *mr_attr);
};

struct fi_efa_ops_gda {
	int (*query_addr)(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
			  uint16_t *remote_qpn, uint32_t *remote_qkey);
	int (*query_qp_wqs)(struct fid_ep *ep_fid,
			    struct fi_efa_wq_attr *sq_attr,
			    struct fi_efa_wq_attr *rq_attr);
	int (*query_cq)(struct fid_cq *cq_fid, struct fi_efa_cq_attr *cq_attr);
	int (*cq_open_ext)(struct fid_domain *domain_fid,
			   struct fi_cq_attr *attr,
			   struct fi_efa_cq_init_attr *efa_cq_init_attr,
			   struct fid_cq **cq_fid, void *context);
	uint64_t (*get_mr_lkey)(struct fid_mr *mr);
	int (*cntr_open_ext)(struct fid_domain *domain,
			     struct fi_cntr_attr *attr,
			     struct fid_cntr **cntr,
			     void *context,
			     struct fi_efa_comp_cntr_init_attr *efa_attr);
};

/*
 * EFA feature flags
 *
 * Features are runtime-discoverable flags advertised by the provider,
 * letting consumers detect the presence of a given behavior or bug fix
 * independently of the libfabric API version (which cannot encode
 * patch releases).
 *
 * Currently defined feature strings:
 *
 *   "mixed_hmem_iov" - the provider correctly inspects every descriptor
 *                      in a multi-iov request for HMEM/iface, rather
 *                      than only the first descriptor.
 */
struct fi_efa_feature_ops {
	bool (*query)(const char *feature);
};

#endif /* _FI_EXT_EFA_H_ */
