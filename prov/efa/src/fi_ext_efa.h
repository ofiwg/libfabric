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
#define FI_EFA_SIGNAL_OPS "efa signal ops"
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
 * Completion with signal (EFA Extended Completion Signaling).
 *
 * A signal lets the EFA NIC perform a registered action (a memory write or a
 * counter increment) at completion time without host CPU involvement. Using it
 * has three parts:
 *   1. Register a signal (control path, below).
 *   2. Enable signal support on the endpoint (FI_OPT_EFA_COMP_SIGNAL, see
 *      rdma/fi_ext.h) before the endpoint is enabled.
 *   3. Attach the signal to individual work requests on the data path via the
 *      fi_efa_msg[_rma] descriptors and the FI_EFA_EXTENDED_MSG op flag (below).
 */

/* Operation performed on the completion memory target. Mirrors
 * efadv_comp_mem_op: NONE performs no memory write (the signal fires without a
 * MEMSET, e.g. a counter-increment-only signal); the SET_SIGNAL_VAL_* values
 * write an operand of the given width. */
enum fi_efa_comp_mem_op {
	FI_EFA_COMP_MEM_OP_NONE,
	FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_8,
	FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_16,
	FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_32,
};

/* fi_efa_comp_mem_op_attr flags. Mirror the efadv EFADV_COMP_MEM_OP_WITH_*
 * flags. A MEMSET completion target uses the comp flag; the error target is
 * optional and gated by the err flag. The provider forwards whichever flags
 * are set to the device, which enforces the supported combinations. */
enum {
	FI_EFA_COMP_MEM_OP_WITH_COMP_EXTERNAL_MEM = 1 << 0,
	FI_EFA_COMP_MEM_OP_WITH_ERR_EXTERNAL_MEM  = 1 << 1,
};

/* Attributes for creating a completion memory operation (step 1 of a MEMSET
 * signal). The target may live in host memory (VA) or device memory
 * (dmabuf/HMEM), described by struct fi_efa_memory_location.
 *
 * The error-operation fields (err_op / err_location / err_length, gated by
 * FI_EFA_COMP_MEM_OP_WITH_ERR_EXTERNAL_MEM) describe an optional second target
 * written on error completion. They mirror efadv_comp_mem_op_init_attr; the
 * provider forwards them to the device, which enforces whether the error
 * operation is supported on the current hardware. */
struct fi_efa_comp_mem_op_attr {
	uint64_t comp_mask;
	uint32_t flags;				/* FI_EFA_COMP_MEM_OP_WITH_* */
	enum fi_efa_comp_mem_op op;		/* completion value width to write */
	enum fi_efa_comp_mem_op err_op;		/* error value width */
	struct fi_efa_memory_location location;		/* comp target: VA or dmabuf */
	struct fi_efa_memory_location err_location;	/* error target: VA or dmabuf */
	uint64_t length;			/* comp target region length */
	uint64_t err_length;			/* error target region length */
};

/*
 * Completion memory operation handle, returned by create_comp_mem_op and
 * passed to register_signal as the MEM_OP backing resource. It is an opaque
 * fid: destroy it with fi_close(&mem_op->fid). The provider embeds it as the
 * first member of its internal object (recovered via container_of), which
 * holds the underlying device resource.
 */
struct fid_efa_comp_mem_op {
	struct fid fid;
};

/* Backing resource type for a signal. */
enum fi_efa_comp_signal_type {
	FI_EFA_COMP_SIGNAL_MEM_OP,	/* backed by a completion memory operation */
	FI_EFA_COMP_SIGNAL_CNTR_INC,	/* backed by an event counter */
};

/* Attributes for registering a signal (step 2). A MEMSET signal references a
 * completion memory op handle created via create_comp_mem_op; a counter signal
 * references an existing libfabric counter. */
struct fi_efa_comp_signal_attr {
	uint64_t comp_mask;
	enum fi_efa_comp_signal_type type;
	union {
		struct fid_efa_comp_mem_op *mem_op;	/* FI_EFA_COMP_SIGNAL_MEM_OP */
		struct fid_cntr *cntr;			/* FI_EFA_COMP_SIGNAL_CNTR_INC */
	};
};

/*
 * Signal handle, returned by register_signal. It is a fid; deregister it with
 * fi_close(&signal->fid). The id used in work requests
 * (fi_efa_msg_rma.{local,remote}_signal_id) is the "id" field, retrievable via
 * fid_efa_comp_signal_get_id(). For remote signals the id must be communicated
 * out-of-band to the sender.
 */
struct fid_efa_comp_signal {
	struct fid fid;
	uint32_t id;
};

static inline uint32_t fid_efa_comp_signal_get_id(struct fid_efa_comp_signal *signal)
{
	return signal->id;
}

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
 * Completion-with-signal control path (EFA-direct only), exposed as a
 * domain-level ops table obtained with fi_open_ops(FI_EFA_SIGNAL_OPS).
 *
 * create_comp_mem_op creates a completion memory operation (a MEMSET target)
 * and returns a struct fid_efa_comp_mem_op handle; destroy it with
 * fi_close(&mem_op->fid).
 *
 * register_signal wraps a completion memory op handle (or an event counter) in
 * a signal and returns a struct fid_efa_comp_signal handle whose id
 * (fid_efa_comp_signal_get_id) is referenced in work requests; deregister it
 * with fi_close(&signal->fid). For remote signals the id must be communicated
 * out-of-band to the sender.
 *
 * query_max_comp_mem_ops reports the maximum number of completion memory
 * operations (MEMSET-backed signals) that can be registered on the domain;
 * 0 means completion with signal is unsupported.
 */
struct fi_efa_ops_signal {
	int (*create_comp_mem_op)(struct fid_domain *domain,
				  struct fi_efa_comp_mem_op_attr *attr,
				  struct fid_efa_comp_mem_op **mem_op);
	int (*register_signal)(struct fid_domain *domain,
			       struct fi_efa_comp_signal_attr *attr,
			       struct fid_efa_comp_signal **signal);
	int (*query_max_comp_mem_ops)(struct fid_domain *domain,
				      uint32_t *max_comp_mem_ops);
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

/*
 * Reinterpret the message descriptor pointer passed to a message-form data
 * transfer call as the corresponding EFA-specific descriptor
 * (struct fi_efa_msg for fi_sendmsg, struct fi_efa_msg_rma for fi_writemsg),
 * so the provider reads the extra EFA per-WR metadata (completion signals).
 *
 * This flag is only accepted on fi_sendmsg and fi_writemsg (the calls that
 * take a descriptor pointer); passing it on any other data transfer variant
 * returns -FI_EINVAL. The endpoint must have signal support enabled via
 * FI_OPT_EFA_COMP_SIGNAL (see rdma/fi_ext.h) before the endpoint is enabled.
 */
#define FI_EFA_EXTENDED_MSG (1ULL << 62)

/*
 * Selects which EFA metadata fields the fi_efa_msg[_rma] struct carries. Each
 * bit gates exactly one field, so new EFA per-WR metadata can be added over
 * time without consuming bits in the common fi_writemsg/fi_sendmsg flags word.
 * A field whose bit is unset is ignored and need not be initialized.
 *
 * A signal's data bit (FI_EFA_*_SIGNAL_DATA) may only be set when its ID bit
 * (FI_EFA_*_SIGNAL_ID) is also set.
 */
enum {
	FI_EFA_LOCAL_SIGNAL_ID    = 1 << 0,	/* local_signal_id is valid */
	FI_EFA_REMOTE_SIGNAL_ID   = 1 << 1,	/* remote_signal_id is valid */
	FI_EFA_LOCAL_SIGNAL_DATA  = 1 << 2,	/* local_signal_data is valid */
	FI_EFA_REMOTE_SIGNAL_DATA = 1 << 3,	/* remote_signal_data is valid */
	/* future EFA per-WR metadata fields add bits here */
};

/*
 * EFA-specific message descriptors. The core descriptor is the first member,
 * so (struct fi_msg_rma *)&efa_msg is valid and vice versa. The feature_bits
 * word selects which of the fields below the provider should read.
 */
struct fi_efa_msg_rma {
	struct fi_msg_rma msg;		/* MUST be first -- castable to fi_msg_rma */
	uint64_t feature_bits;		/* which fields below are valid (FI_EFA_*) */
	uint32_t local_signal_id;	/* opaque ID from register_comp_signal */
	uint32_t remote_signal_id;
	uint32_t local_signal_data;	/* per-WR operand for the local signal */
	uint32_t remote_signal_data;	/* per-WR operand for the remote signal */
};

struct fi_efa_msg {
	struct fi_msg msg;		/* MUST be first -- castable to fi_msg */
	uint64_t feature_bits;		/* which fields below are valid (FI_EFA_*) */
	uint32_t local_signal_id;
	uint32_t remote_signal_id;
	uint32_t local_signal_data;
	uint32_t remote_signal_data;
};


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
