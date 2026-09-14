/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _FI_EXT_EFA_H_
#define _FI_EXT_EFA_H_

#include <stdbool.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#define FI_EFA_DOMAIN_OPS "efa domain ops"
#define FI_EFA_GDA_OPS "efa gda ops"
#define FI_EFA_MEM_COMP_ACTION_OPS "efa mem comp action ops"
#define FI_EFA_FEATURE_OPS "efa feature ops"
#define FI_EFA_MODIFY_EP_OPS "efa modify ep ops"

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

enum fi_efa_wq_caps {
    FI_EFA_WQ_CAPS_64_BIT_REQ_ID = 1 << 0,
};

struct fi_efa_wq_attr {
    uint8_t *buffer;
    uint32_t entry_size;
    uint32_t num_entries;
    uint32_t *doorbell;
    uint32_t max_batch;
    uint16_t caps;
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

/*
 * Completion actions.
 *
 * A completion action lets the EFA NIC perform a registered action -- today a
 * write into a registered memory vector -- when a work request completes,
 * without host CPU involvement. Using one has three parts:
 *   1. Register a memory completion action (control path, below).
 *   2. Enable action support on the endpoint (FI_OPT_EFA_COMP_ACTION, see
 *      rdma/fi_ext.h) before the endpoint is enabled.
 *   3. Name the action from individual work requests on the data path via the
 *      fi_efa_msg_rma descriptor and the FI_EFA_EXTENDED_MSG op flag (below).
 */

/*
 * A registered completion action. It is an opaque fid: release it with
 * fi_close(&action->fid). The provider embeds it as the first member of its
 * internal object, recovered via container_of.
 *
 * action_id is the value a work request names the action by, retrievable via
 * fid_efa_comp_action_get_id(). For an action executed at the target, the id
 * must be communicated out of band to the initiator.
 */
struct fid_efa_comp_action {
	struct fid fid;
	uint32_t action_id;
};

static inline uint32_t
fid_efa_comp_action_get_id(struct fid_efa_comp_action *action)
{
	return action->action_id;
}

/* Operation a memory completion action performs. Mirrors efadv_comp_op:
 * SET_INITIATOR_VAL writes a value the initiator supplies per work request. */
enum fi_efa_mem_comp_action_op {
	FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL = 0,
};

/*
 * Attributes for a memory completion action, mirroring
 * efadv_mem_comp_action_init_attr.
 *
 * The target is num_entries slots of entry_size bytes each, starting at
 * location, which may be host memory (VA) or device memory (dmabuf) exactly as
 * for completion counters. entry_size is the width of the device's write and is
 * 1, 2 or 4, and location must be aligned to it. num_entries mirrors the
 * rdma-core attribute and must be 1: the device accepts no other value, and a
 * work request has no way to name a slot other than the first, so the target is
 * effectively a scalar. Should the device gain multi-slot targets, selecting a
 * slot per work request is a new data-path field and feature bit, not a change
 * here.
 *
 * comp_mask versions this struct alone: with one creator per action type, each
 * action type owns its own version namespace.
 */
struct fi_efa_mem_comp_action_attr {
	uint64_t comp_mask;
	uint64_t flags;				/* 0 today */
	struct fi_efa_memory_location location;
	uint32_t num_entries;
	uint32_t entry_size;			/* 1, 2 or 4 */
	enum fi_efa_mem_comp_action_op op;
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
 * Memory completion action control path (EFA-direct only), a domain-level ops
 * table obtained with
 * fi_open_ops(&domain->fid, FI_EFA_MEM_COMP_ACTION_OPS, 0, &ops, NULL).
 *
 * Every call names the domain explicitly, as the other EFA ops tables do. An
 * action and the endpoint whose work requests name it must come from the same
 * domain.
 *
 * create_mem_comp_action registers a memory completion action and returns a
 * struct fid_efa_comp_action; release it with fi_close(&action->fid), after
 * which the target memory is the application's to free.
 *
 * query_max_mem_comp_actions reports how many memory completion actions the
 * domain can have registered at once; 0 means the device does not support them
 * and create_mem_comp_action fails.
 *
 * Other action types -- a counter increment -- get their own creator and limit
 * query appended to this table once the device supports them.
 */
struct fi_efa_ops_mem_comp_action {
	int (*create_mem_comp_action)(struct fid_domain *domain,
				      struct fi_efa_mem_comp_action_attr *attr,
				      struct fid_efa_comp_action **action);
	int (*query_max_mem_comp_actions)(struct fid_domain *domain,
					  uint32_t *max_mem_comp_actions);
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


/**
 * EFA provider specific op flags (60 - 63 bits)
 * See rdma/fabric.h for 0-59 bit that apply to all providers
 */

 /*
 * Hint the device to optimize for higher message rate for rdma operations.
 * This flag can be passed in the 'flags' argument of data transfer calls
 * such as fi_writemsg().
 */
#define FI_EFA_WR_HIGH_PPS (1ULL << 60)

/*
 * Request relaxed ordering for a memory region. This flag can be passed in the
 * 'flags' argument of fi_mr_reg*().
 *
 * By default, the EFA device issues all data transactions to a memory region
 * (RDMA reads, RDMA writes, and receives) with strict ordering, so the TX
 * completion of an operation guarantees that subsequent operations to the same
 * endpoint appear at the target after it. When FI_EFA_MR_RELAXED_ORDERING is
 * set, that ordering guarantee is lost.
 */
#define FI_EFA_MR_RELAXED_ORDERING (1ULL << 61)


enum {
	FI_EFA_EP_ATTR_QKEY = 1 << 0,
};

#define FI_EFA_EP_ATTR_SUPPORTED_FLAGS (FI_EFA_EP_ATTR_QKEY)

struct fi_efa_ep_attr {
	uint32_t qkey;
};

struct fi_efa_ops_modify_ep {
	int (*modify_ep)(struct fid_ep *ep, struct fi_efa_ep_attr *ep_attr,
			 uint64_t attr_mask);
};

#endif /* _FI_EXT_EFA_H_ */
