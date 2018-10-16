/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#ifndef _CXIP_PROV_H_
#define _CXIP_PROV_H_

#include "config.h"

#include <pthread.h>

#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>

#include <ofi.h>
#include <ofi_atom.h>
#include <ofi_atomic.h>
#include <ofi_mr.h>
#include <ofi_enosys.h>
#include <ofi_indexer.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_file.h>
#include <ofi_osd.h>
#include <ofi_util.h>

#include "libcxi/libcxi.h"

#define CXIP_EP_MAX_MSG_SZ (1 << 23)
#define CXIP_EP_MAX_INJECT_SZ ((1 << 8) - 1)
#define CXIP_EP_MAX_BUFF_RECV (1 << 26)
#define CXIP_EP_MAX_ORDER_RAW_SZ CXIP_EP_MAX_MSG_SZ
#define CXIP_EP_MAX_ORDER_WAR_SZ CXIP_EP_MAX_MSG_SZ
#define CXIP_EP_MAX_ORDER_WAW_SZ CXIP_EP_MAX_MSG_SZ
#define CXIP_EP_MEM_TAG_FMT FI_TAG_GENERIC
#define CXIP_EP_MAX_EP_CNT (128)
#define CXIP_EP_MAX_CQ_CNT (32)
#define CXIP_EP_MAX_CNTR_CNT (128)
#define CXIP_EP_MAX_TX_CNT (16)
#define CXIP_EP_MAX_RX_CNT (16)
#define CXIP_EP_MAX_IOV_LIMIT (8)
#define CXIP_EP_TX_SZ (256)
#define CXIP_EP_RX_SZ (256)
#define CXIP_EP_MIN_MULTI_RECV (64)
#define CXIP_EP_MAX_ATOMIC_SZ (4096)
#define CXIP_EP_MAX_CTX_BITS (4)
#define CXIP_EP_MSG_PREFIX_SZ (0)
#define CXIP_DOMAIN_MR_CNT (65535)
#define CXIP_DOMAIN_CAPS_FLAGS (FI_LOCAL_COMM | FI_REMOTE_COMM)

#define CXIP_EQ_DEF_SZ (1 << 8)
#define CXIP_CQ_DEF_SZ (1 << 8)
#define CXIP_AV_DEF_SZ (1 << 8)
#define CXIP_CMAP_DEF_SZ (1 << 10)
#define CXIP_EPOLL_WAIT_EVENTS (32)

#define CXIP_CQ_DATA_SIZE (sizeof(uint64_t))
#define CXIP_TAG_SIZE (sizeof(uint64_t))
#define CXIP_MAX_NETWORK_ADDR_SZ (35)

#define CXIP_PEP_LISTENER_TIMEOUT (10000)
#define CXIP_CM_COMM_TIMEOUT (2000)
#define CXIP_EP_MAX_RETRY (5)
#define CXIP_EP_MAX_CM_DATA_SZ (256)

#define CXIP_RMA_MAX_IOV (1)
#define CXIP_AMO_MAX_IOV (1)

#define CXIP_EP_RDM_PRI_CAP \
	(FI_RMA | FI_ATOMICS | FI_TAGGED | FI_RECV | FI_SEND | \
	 FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
	 /* TODO FI_MSG | FI_NAMED_RX_CTX | FI_DIRECTED_RECV */ 0 \
	)

#define CXIP_EP_RDM_SEC_CAP_BASE \
	(FI_SOURCE | FI_SHARED_AV | FI_LOCAL_COMM | FI_REMOTE_COMM | \
	 /* TODO FI_MULTI_RECV | FI_RMA_EVENT | FI_FENCE | FI_TRIGGER */ 0 \
	)
extern uint64_t CXIP_EP_RDM_SEC_CAP;

#define CXIP_EP_RDM_CAP_BASE (CXIP_EP_RDM_PRI_CAP | CXIP_EP_RDM_SEC_CAP_BASE)
extern uint64_t CXIP_EP_RDM_CAP;

#define CXIP_EP_MSG_ORDER \
	(FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS | FI_ORDER_WAR | \
	 FI_ORDER_WAW | FI_ORDER_WAS | FI_ORDER_SAR | FI_ORDER_SAW | \
	 FI_ORDER_SAS)

#define CXIP_EP_COMP_ORDER (FI_ORDER_STRICT | FI_ORDER_DATA)
#define CXIP_EP_DEFAULT_OP_FLAGS (FI_TRANSMIT_COMPLETE)

#define CXIP_EP_CQ_FLAGS \
	(FI_SEND | FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION)
#define CXIP_EP_CNTR_FLAGS \
	(FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | \
	 FI_REMOTE_WRITE)

#define CXIP_EP_SET_TX_OP_FLAGS(_flags) \
	do { \
		if (!((_flags) & FI_INJECT_COMPLETE)) \
			(_flags) |= FI_TRANSMIT_COMPLETE; \
	} while (0)

#define CXIP_MODE (0)

#define CXIP_MAX_ERR_CQ_EQ_DATA_SZ CXIP_EP_MAX_CM_DATA_SZ

#define CXIP_MAJOR_VERSION 0
#define CXIP_MINOR_VERSION 0

#define CXIP_WIRE_PROTO_VERSION (1)

extern const char cxip_fab_fmt[];
extern const char cxip_dom_fmt[];
extern const char cxip_prov_name[];
extern struct fi_provider cxip_prov;
extern int cxip_av_def_sz;
extern int cxip_cq_def_sz;
extern int cxip_eq_def_sz;
extern struct slist cxip_if_list;

/**
 * The CXI Provider Address format.
 *
 * A Cassini NIC Address and PID identify a libfabric Endpoint.  Cassini borrows
 * the name 'PID' from Portals. The maximum PID value in Cassini is 12 bits. We
 * allow/use only 9 bits.
 *
 * Pid -1 is reserved.  When used, the library auto-assigns a free PID value
 * when network resources are allocated.  Libfabric clients can achieve this by
 * not specifying a 'service' in a call to fi_getinfo() or by specifying the
 * reserved value -1.
 *
 * TODO: If NIC Address must be non-zero, the valid bit can be removed.
 * TODO: Is 22 bits enough for NIC Address?
 */
#define	CXIP_ADDR_VNI_BITS	17
#define CXIP_ADDR_PID_BITS	9
#define	CXIP_ADDR_IDX_BITS	(CXIP_ADDR_VNI_BITS - CXIP_ADDR_PID_BITS)
#define	CXIP_ADDR_PID_AUTO	((1 << CXIP_ADDR_PID_BITS) - 1)
#define	CXIP_ADDR_NIC_BITS	(32 - 1 - CXIP_ADDR_PID_BITS)

struct cxip_addr {
	union {
		struct {
			uint32_t pid	: CXIP_ADDR_PID_BITS;
			uint32_t nic	: CXIP_ADDR_NIC_BITS;
			uint32_t valid	: 1;
		};
		uint32_t raw;
	};
};

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

/* 256 slots total (2^(17-9) == 2^8)
 *  16 RX slots
 * 239 MR slots
 *   1 Rendesvous Send slot
 */
#define	CXIP_EP_MAX_IDX_CNT	(1 << CXIP_ADDR_IDX_BITS)
#define	CXIP_EP_MAX_MR_CNT	(CXIP_EP_MAX_IDX_CNT - CXIP_EP_MAX_RX_CNT - 1)

#define CXIP_MR_TO_IDX(key)	(CXIP_EP_MAX_RX_CNT + (key))
#define CXIP_RXC_TO_IDX(rx_id)	(rx_id)

#define	CXIP_AV_ADDR_IDX(av, fi_addr)	((uint64_t)fi_addr & av->mask)
#define	CXIP_AV_ADDR_RXC(av, fi_addr)	((uint64_t)fi_addr >> \
						(64 - av->rx_ctx_bits))

// TODO: comments are not yet complete, and may not be entirely correct
//       complete documentation and review thoroughly

/**
 * Local Interface Domain
 *
 * Support structure.
 *
 * Create/lookup in cxip_get_if_domain(), during EP creation.
 *
 * These are associated with the local Cassini chip referenced in the parent
 * dev_if. There must be one for every 'pid' that is active for communicating
 * with other devices on the network.
 *
 * This structure wraps a libcxi domain, or Cassini "granule" (of 256 soft
 * endpoints) for the local CXI chip.
 *
 * The vni value specifies a Cassini VNI, as supplied by the privileged WLM that
 * started the job/service that has activated this 'pid'.
 *
 * The pid value specifies the libcxi domain, or VNI granule, and is supplied
 * by the application. Each pid has its own set of RX and MR resources for
 * receiving data.
 *
 * Every EP will use one of these for the RX context.
 */
struct cxip_if_domain {
	struct dlist_entry if_dom_entry; // attach to cxip_if->if_doms
	struct cxip_if *dev_if;		// local Cassini device
	struct cxil_domain *cxil_if_dom; // cxil domain (dev, vni, pid)
	uint32_t vni;			// vni value (namespace)
	uint32_t pid;			// pid value (granule index)
	struct index_map lep_map;	// Cassini Logical EP bitmap
	ofi_atomic32_t ref;
	fastlock_t lock;
};

/**
 * Local Interface
 *
 * Support structure.
 *
 * Created by cxip_get_if().
 *
 * There will be one of these for every local Cassini device on the node,
 * typically 1 to 4 interfaces.
 *
 * This implements a single, dedicated Cassini cmdq for creating memory regions
 * and attaching them to Cassini PTEs.
 *
 * Statically initialized at library initialization, based on information from
 * the libcxi layer.
 */
struct cxip_if {
	struct slist_entry if_entry;	// attach to global cxip_if_list
	uint32_t if_nic;		// cxil NIC identifier
	uint32_t if_idx;		// cxil NIC index
	uint32_t if_fabric;		// cxil NIC fabric address
	struct cxil_dev *if_dev;	// cxil NIC DEV structure
	uint32_t if_pid_granule;	// cxil NIC granule size
	struct cxil_lni *if_lni;	// cxil NIC LNI structure
	struct cxi_cp cps[16];		// Cassini communication profiles
	int n_cps;
	struct dlist_entry if_doms;	// if_domain list
	struct dlist_entry ptes;	// PTE list
	ofi_atomic32_t ref;
	struct cxi_cmdq *mr_cmdq;	// used for all MR activation
	struct cxi_evtq *mr_evtq;	// used for async completion
	fastlock_t lock;
};

/**
 * Portal Table Entry
 *
 * Support structure.
 *
 * Created in cxip_pte_alloc().
 *
 * When the PTE object is created, the user specifies the desired pid_idx to
 * use, which implicitly defines the function of this PTE. It is an error to
 * attempt to multiply allocate the same pid_idx.
 */
struct cxip_pte {
	struct dlist_entry pte_entry;	// attaches to cxip_if->ptes
	struct cxip_if_domain *if_dom;	// parent domain
	uint64_t pid_idx;
	struct cxil_pte *pte;		// cxil PTE object
	struct cxil_pte_map *pte_map;	// cxil PTE mapped object
	enum c_ptlte_state state;	// Cassini PTE state
};

/**
 * Fabric object
 *
 * libfabric if_fabric implementation.
 *
 * Created in cxip_fabric().
 *
 * This is an anchor for all remote EPs on this fabric, and allows domains to
 * find common services.
 *
 */
struct cxip_fabric {
	struct fid_fabric fab_fid;
	ofi_atomic32_t ref;
	struct dlist_entry service_list;	// contains services (TODO)
	struct dlist_entry fab_list_entry;	// attaches to cxip_fab_list
	fastlock_t lock;
};

/**
 * Domain object
 *
 * libfabric if_domain implementation.
 *
 * Created in cxip_domain().
 */
struct cxip_domain {
	struct fid_domain dom_fid;
	struct fi_info info;		// copy of user-supplied domain info
	struct cxip_fabric *fab;	// parent cxip_fabric
	fastlock_t lock;
	ofi_atomic32_t ref;

	struct cxip_eq *eq;		// linked during cxip_dom_bind()
	struct cxip_eq *mr_eq;		// == eq || == NULL

	enum fi_progress progress_mode;
	struct dlist_entry dom_list_entry;
					// attaches to global cxip_dom_list
	struct fi_domain_attr attr;	// copy of user or default domain attr

	uint32_t nic_addr;		// dev address of source NIC
	int enabled;			// set when domain is enabled
	struct cxip_if *dev_if;		// looked when domain is enabled
};

// Apparently not yet in use (??)
// No reference count (?)
struct cxip_eq {
	struct fid_eq eq;
	struct fi_eq_attr attr;
	struct cxip_fabric *cxi_fab;

	struct dlistfd_head list;
	struct dlistfd_head err_list;
	struct dlist_entry err_data_list;
	fastlock_t lock;

	struct fid_wait *waitset;
	int signal;
	int wait_fd;
};

/**
 * RMA request
 *
 * Support structures, accumulated in a union.
 */
struct cxip_req_rma {
	struct cxi_iova local_md;	// RMA target buffer
};

struct cxip_req_amo {
	struct cxi_iova local_md;	// RMA target buffer
	void *result_buf;		// local buffer for fetch
};

struct cxip_req_recv {
	struct cxip_rx_ctx *rxc;	// receive context
	void *recv_buf;			// local receive buffer
	struct cxi_iova recv_md;	// local receive MD
	int rc;				// result code
	int rlength;			// receive length
	int mlength;			// message length
	uint64_t start;			// starting receive offset
};

struct cxip_req_send {
	struct cxi_iova send_md;	// message target buffer
};

struct cxip_req_oflow {
	struct cxip_rx_ctx *rxc;	// ??
	struct cxip_oflow_buf *oflow_buf;
					// ??
};

/**
 * Async Request
 *
 * Support structure.
 *
 * Created in cxip_cq_req_alloc().
 *
 * This implements an async-request/callback mechanism. It uses the libfabric
 * utility pool, which provides a pool of reusable memory objects that supports
 * a fast lookup through the req_id index value, and can be bound to a CQ.
 *
 * The request is allocated and bound to the CQ, and then the command is issued.
 * When the completion queue signals completion, this request is found, and the
 * callback function is called.
 */
struct cxip_req {
	/* Control info */
	struct dlist_entry list;	// attaches to utility pool
	struct cxip_cq *cq;		// request CQ
	int req_id;			// fast lookup in index table
	void (*cb)(struct cxip_req *req, const union c_event *evt);
					// completion event callback

	/* CQ event fields, set according to fi_cq.3
	 *   - set by provider
	 *   - returned to user in completion event
	 */
	uint64_t context;
	uint64_t flags;
	uint64_t data_len;
	uint64_t buf;
	uint64_t data;
	uint64_t tag;
	fi_addr_t addr;

	/* Request parameters */
	union {
		struct cxip_req_rma rma;
		struct cxip_req_amo amo;
		struct cxip_req_oflow oflow;
		struct cxip_req_recv recv;
		struct cxip_req_send send;
	};
};

/**
 * cxip_cq completion report callback typedef
 */
struct cxip_cq;
typedef int (*cxip_cq_report_fn)(struct cxip_cq *cq, fi_addr_t addr,
				 struct cxip_req *req);

/**
 * cxip_cq completion event overflow list if ring buffer fills.
 */
struct cxip_cq_overflow_entry_t {
	size_t len;			// data length
	fi_addr_t addr;			// data address
	struct dlist_entry entry;	// attaches to cxip_cq
	char cq_entry[0];		// data
};

/**
 * Completion Queue
 *
 * libfabric fi_cq implementation.
 *
 * Created in cxip_cq_open().
 */
struct cxip_cq {
	struct fid_cq cq_fid;
	struct cxip_domain *domain;	// parent domain
	ssize_t cq_entry_size;		// size of CQ entry (depends on type)
	ofi_atomic32_t ref;
	struct fi_cq_attr attr;		// copy of user or default attributes

	/* Ring buffer */
	struct ofi_ringbuf addr_rb;
	struct ofi_ringbuffd cq_rbfd;
	struct ofi_ringbuf cqerr_rb;
	struct dlist_entry overflow_list; // slower: used when ring overfills
	fastlock_t lock;
	fastlock_t list_lock;

	struct fid_wait *waitset;
	int signal;
	ofi_atomic32_t signaled;

	struct dlist_entry ep_list;	// contains endpoints (not used yet)
	struct dlist_entry rx_list;	// contains rx contexts
	struct dlist_entry tx_list;	// contains tx contexts

	cxip_cq_report_fn report_completion;
					// callback function

	int enabled;
	struct cxi_evtq *evtq;		// set when enabled
	fastlock_t req_lock;
	struct util_buf_pool *req_pool;	// utility pool for cxip_req
	struct indexer req_table;	// fast lookup index table for cxip_req
};

/**
 * Completion Counter
 *
 * libfabric if_cntr implementation.
 *
 * Created in cxip_cntr_open().
 */
struct cxip_cntr {
	struct fid_cntr cntr_fid;
	struct cxip_domain *domain;	// parent domain
	ofi_atomic32_t ref;
	struct fi_cntr_attr attr;	// copy of user or default attributes

	struct fid_wait *waitset;
	int signal;

	struct dlist_entry rx_list;	// contains rx contexts
	struct dlist_entry tx_list;	// contains tx contexts
	fastlock_t list_lock;
};

/**
 * TX/RX Completion
 *
 * Support structure.
 *
 * Initialized when binding TX/RX to EP.
 */
struct cxip_comp {
	uint8_t send_cq_event;
	uint8_t recv_cq_event;
	char reserved[2];

	struct cxip_cq *send_cq;
	struct cxip_cq *recv_cq;

	struct cxip_cntr *send_cntr;
	struct cxip_cntr *recv_cntr;
	struct cxip_cntr *read_cntr;
	struct cxip_cntr *write_cntr;
	struct cxip_cntr *rem_read_cntr;
	struct cxip_cntr *rem_write_cntr;

	struct cxip_eq *eq;
};

/**
 * Unexpected-Send buffer
 *
 * Support structure.
 */
struct cxip_ux_send {
	struct dlist_entry list;
	struct cxip_oflow_buf *oflow_buf;
	uint64_t start;
	uint64_t length;
};

/**
 * Overflow buffer
 *
 * Support structure.
 */
struct cxip_oflow_buf {
	struct dlist_entry list;
	struct cxip_rx_ctx *rxc;
	void *buf;
	struct cxi_iova md;
	ofi_atomic32_t ref;
	int exhausted;
	int buffer_id;
};

/**
 * Receive Context
 *
 * Support structure.
 *
 * Created in cxip_rx_ctx_alloc(), during EP creation.
 */
struct cxip_rx_ctx {
	struct fid_ep ctx;

	uint16_t rx_id;			// SEP index
	int enabled;
	int progress;			// unused
	int recv_cq_event;		// unused
	int use_shared;

	size_t num_left;		// unused (?) (set, never referenced)
	size_t min_multi_recv;
	uint64_t addr;
	struct cxip_comp comp;
	struct cxip_rx_ctx *srx_ctx;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry cq_entry;	// attaches to CQ RX list
	struct dlist_entry ep_list;	// contains EPs using shared context
	fastlock_t lock;

	struct fi_rx_attr attr;

	struct cxip_pte *rx_pte;
	struct cxi_cmdq *rx_cmdq;

	int eager_threshold;

	/* Unexpected message handling */
	ofi_atomic32_t oflow_buf_cnt;
	int oflow_bufs_max;
	int oflow_msgs_max;
	int oflow_buf_size;
	struct dlist_entry oflow_bufs;	// Overflow buffers
	struct dlist_entry ux_sends;	// Sends matched in overflow list
	struct dlist_entry ux_recvs;	// Recvs matched in overflow list
};

/**
 * Transmit Context
 *
 * Support structure.
 *
 * Created by cxip_tx_ctx_alloc(), during EP creation.
 *
 */
struct cxip_tx_ctx {
	union {
		struct fid_ep ctx;	// standard endpoint
		struct fid_stx stx;	// scalable endpoint
	} fid;
	size_t fclass;

	uint16_t tx_id;			// SEP index
	uint8_t enabled;
	uint8_t progress;		// unused

	int use_shared;
	struct cxip_comp comp;
	struct cxip_tx_ctx *stx_ctx;	// shared context (?)

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry cq_entry;	// attaches to CQ TX list
	struct dlist_entry ep_list;	// contains EPs using shared context
	fastlock_t lock;

	struct fi_tx_attr attr;		// attributes

	struct cxi_cmdq *tx_cmdq;	// added during cxip_tx_ctx_enable()
};

/**
 * Endpoint Internals
 *
 * Support structure, libfabric fi_endpoint implementation.
 *
 * Created in cxip_alloc_endpoint().
 *
 * This is the meat of the endpoint object. It has been separated from cxip_ep
 * to support aliasing, to allow different TX/RX attributes for a single TX or
 * RX object. TX/RX objects are tied to Cassini PTEs, and the number of bits
 * available to represent separate contexts is limited, so we want to reuse
 * these when the only difference is the attributes.
 */
struct cxip_ep_obj {
	size_t fclass;

	int tx_shared;
	int rx_shared;
	size_t min_multi_recv;

	ofi_atomic32_t ref;
	struct cxip_eq *eq;		// EQ for async EP add/rem/etc
	struct cxip_av *av;		// target AV (network address vector)
	struct cxip_domain *domain;	// parent domain

	/* TX/RX context pointers for standard EPs. */
	struct cxip_rx_ctx *rx_ctx;	// rx_array[0] || NULL
	struct cxip_tx_ctx *tx_ctx;	// tx_array[0] || NULL

	/* TX/RX contexts.  Standard EPs have 1 of each.  SEPs have many. */
	struct cxip_rx_ctx **rx_array;	// rx contexts
	struct cxip_tx_ctx **tx_array;	// tx contexts
	ofi_atomic32_t num_rx_ctx;	// num rx contexts (>= 1)
	ofi_atomic32_t num_tx_ctx;	// num tx contexts (>= 1)

	/* List of shared contexts associated with the EP.  Necessary? */
	struct dlist_entry rx_ctx_entry;
	struct dlist_entry tx_ctx_entry;

	struct fi_info info;		// TODO: use this properly
	struct fi_ep_attr ep_attr;

	int is_enabled;
	fastlock_t lock;

	struct cxip_addr src_addr;	// address of this NIC
	uint32_t vni;			// VNI all EP addressing
	struct cxip_if_domain *if_dom;
};

/**
 * Endpoint
 *
 * libfabric fi_endpoint implementation.
 *
 * Created in cxip_alloc_endpoint().
 *
 * This contains TX and RX attributes, and can share the cxip_ep_attr structure
 * among multiple EPs, to conserve Cassini resources.
 */
struct cxip_ep {
	struct fid_ep ep;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;
	struct cxip_ep_obj *ep_obj;
	int is_alias;
};

/**
 * Memory Region
 *
 * libfabric fi_mr implementation.
 *
 * Created in cxip_regattr().
 *
 */
struct cxip_mr {
	struct fid_mr mr_fid;
	struct cxip_domain *domain;	// parent domain
	struct cxip_ep *ep;		// endpoint for remote memory
	uint64_t key;			// memory key
	uint64_t flags;			// special flags
	struct fi_mr_attr attr;		// attributes
	struct cxip_cntr *cntr;		// if bound to cntr
	struct cxip_cq *cq;		// if bound to cq
	fastlock_t lock;

	/*
	 * A standard MR is implemented as a single persistent, non-matching
	 * list entry (LE) on the PTE mapped to the logical endpoint
	 * addressed with the four-tuple:
	 *
	 *    ( if_dom->dev_if->if_nic, if_dom->pid, vni, pid_idx )
	 */
	uint32_t pid_idx;

	int enabled;
	struct cxil_pte *pte;
	unsigned int pte_hw_id;
	struct cxil_pte_map *pte_map;

	void *buf;			// memory buffer VA
	uint64_t len;			// memory length
	struct cxi_iova md;		// memory buffer IOVA
};

/**
 * Address Vector header
 *
 * Support structure.
 */
struct cxip_av_table_hdr {
	uint64_t size;
	uint64_t stored;
};

/**
 * Address Vector
 *
 * libfabric fi_av implementation.
 *
 * Created in cxip_av_open().
 */
struct cxip_av {
	struct fid_av av_fid;
	struct cxip_domain *domain;	// parent domain
	ofi_atomic32_t ref;
	struct fi_av_attr attr;		// copy of user attributes
	uint64_t mask;			// mask with rx_ctx_bits MSbits clear
	int rx_ctx_bits;		// address bits needed for SEP RXs
	socklen_t addrlen;		// size of struct cxip_addr
	struct cxip_eq *eq;		// event queue
	struct cxip_av_table_hdr *table_hdr;
					// mapped AV table
	struct cxip_addr *table;	// address data in table_hdr memory
	uint64_t *idx_arr;		// valid only for shared AVs
	struct util_shm shm;		// OFI shared memory structure
	int shared;			// set if shared
	struct dlist_entry ep_list;	// contains EP fid objects
	fastlock_t list_lock;
};

/**
 * CNTR/CQ wait object file list element
 *
 * Support structure.
 *
 * Created in cxip_cntr_open(), cxip_cq_open().
 */
struct cxip_fid_list {
	struct dlist_entry entry;
	struct fid *fid;
};

/**
 * Wait object
 *
 * Support structure.
 *
 * Created in cxip_wait_get_obj().
 */
struct cxip_wait {
	struct fid_wait wait_fid;
	struct cxip_fabric *fab;
	struct dlist_entry fid_list;
	enum fi_wait_obj type;
	union {
		int fd[2];
		struct cxip_mutex_cond {
			pthread_mutex_t mutex;
			pthread_cond_t cond;
		} mutex_cond;
	} wobj;
};

struct cxip_if *cxip_if_lookup(uint32_t nic_addr);
int cxip_get_if(uint32_t nic_addr, struct cxip_if **dev_if);
void cxip_put_if(struct cxip_if *dev_if);
int cxip_get_if_domain(struct cxip_if *dev_if, uint32_t vni, uint32_t pid,
		       struct cxip_if_domain **if_dom);
void cxip_put_if_domain(struct cxip_if_domain *if_dom);
int cxip_if_domain_lep_alloc(struct cxip_if_domain *if_dom, uint64_t pid_idx);
int cxip_if_domain_lep_free(struct cxip_if_domain *if_dom, uint64_t pid_idx);
void cxip_if_init(void);
void cxip_if_fini(void);

int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_evtq *evtq,
		   uint64_t pid_idx, struct cxi_pt_alloc_opts *opts,
		   struct cxip_pte **pte);
void cxip_pte_free(struct cxip_pte *pte);
int cxip_pte_state_change(struct cxip_if *dev_if, uint32_t pte_num,
			  enum c_ptlte_state new_state);

int cxip_parse_addr(const char *node, const char *service,
		    struct cxip_addr *addr);

int cxip_domain_enable(struct cxip_domain *dom);
int cxip_domain(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

char *cxip_get_fabric_name(struct cxip_addr *src_addr);
char *cxip_get_domain_name(struct cxip_addr *src_addr);

void cxip_dom_add_to_list(struct cxip_domain *domain);
int cxip_dom_check_list(struct cxip_domain *domain);
void cxip_dom_remove_from_list(struct cxip_domain *domain);
struct cxip_domain *cxip_dom_list_head(void);
int cxip_dom_check_manual_progress(struct cxip_fabric *fabric);

void cxip_fab_add_to_list(struct cxip_fabric *fabric);
int cxip_fab_check_list(struct cxip_fabric *fabric);
void cxip_fab_remove_from_list(struct cxip_fabric *fabric);
struct cxip_fabric *cxip_fab_list_head(void);

int _cxip_av_lookup(struct cxip_av *av, fi_addr_t fi_addr,
		    struct cxip_addr *addr);
int cxip_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context);

struct fi_info *cxip_fi_info(uint32_t version, enum fi_ep_type ep_type,
			     const struct fi_info *hints, void *src_addr,
			     void *dest_addr);
int cxip_rdm_fi_info(uint32_t version, void *src_addr, void *dest_addr,
		     const struct fi_info *hints, struct fi_info **info);
int cxip_alloc_endpoint(struct fid_domain *domain, struct fi_info *info,
			struct cxip_ep **ep, void *context, size_t fclass);
int cxip_rdm_ep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **ep, void *context);
int cxip_rdm_sep(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **sep, void *context);

int cxip_verify_info(uint32_t version, const struct fi_info *hints);
int cxip_verify_fabric_attr(const struct fi_fabric_attr *attr);
int cxip_get_src_addr(struct cxip_addr *dest_addr, struct cxip_addr *src_addr);

int cxip_rdm_verify_ep_attr(const struct fi_ep_attr *ep_attr,
			    const struct fi_tx_attr *tx_attr,
			    const struct fi_rx_attr *rx_attr);

int cxip_verify_domain_attr(uint32_t version, const struct fi_info *info);

int cxip_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context);

int cxip_wait_get_obj(struct fid_wait *fid, void *arg);
void cxip_wait_signal(struct fid_wait *wait_fid);
int cxip_wait_close(fid_t fid);
int cxip_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		   struct fid_wait **waitset);

struct cxip_rx_ctx *cxip_rx_ctx_alloc(const struct fi_rx_attr *attr,
				      void *context, int use_shared);
void cxip_rx_ctx_free(struct cxip_rx_ctx *rx_ctx);

void cxip_rxc_oflow_replenish(struct cxip_rx_ctx *rxc);
void cxip_rxc_oflow_cleanup(struct cxip_rx_ctx *rxc);
int cxip_rx_ctx_enable(struct cxip_rx_ctx *rxc);
int cxip_tx_ctx_enable(struct cxip_tx_ctx *txc);
struct cxip_tx_ctx *cxip_tx_ctx_alloc(const struct fi_tx_attr *attr,
				      void *context, int use_shared);
struct cxip_tx_ctx *cxip_stx_ctx_alloc(const struct fi_tx_attr *attr,
				       void *context);
void cxip_tx_ctx_free(struct cxip_tx_ctx *tx_ctx);

struct cxip_req *cxip_cq_req_alloc(struct cxip_cq *cq, int remap);
void cxip_cq_req_free(struct cxip_req *req);
void cxip_cq_add_tx_ctx(struct cxip_cq *cq, struct cxip_tx_ctx *tx_ctx);
void cxip_cq_remove_tx_ctx(struct cxip_cq *cq, struct cxip_tx_ctx *tx_ctx);
void cxip_cq_add_rx_ctx(struct cxip_cq *cq, struct cxip_rx_ctx *rx_ctx);
void cxip_cq_remove_rx_ctx(struct cxip_cq *cq, struct cxip_rx_ctx *rx_ctx);
void cxip_cq_progress(struct cxip_cq *cq);
int cxip_cq_enable(struct cxip_cq *cxi_cq);
int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context);
int cxip_cq_report_error(struct cxip_cq *cq, struct cxip_req *req, size_t olen,
			 int err, int prov_errno, void *err_data,
			 size_t err_data_size);

void cxip_cntr_add_tx_ctx(struct cxip_cntr *cntr, struct cxip_tx_ctx *tx_ctx);
void cxip_cntr_remove_tx_ctx(struct cxip_cntr *cntr,
			     struct cxip_tx_ctx *tx_ctx);
void cxip_cntr_add_rx_ctx(struct cxip_cntr *cntr, struct cxip_rx_ctx *rx_ctx);
void cxip_cntr_remove_rx_ctx(struct cxip_cntr *cntr,
			     struct cxip_rx_ctx *rx_ctx);
int cxip_cntr_progress(struct cxip_cntr *cntr);
int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);

#define _CXIP_LOG_DBG(subsys, ...) FI_DBG(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_LOG_ERROR(subsys, ...) FI_WARN(&cxip_prov, subsys, __VA_ARGS__)

#endif
