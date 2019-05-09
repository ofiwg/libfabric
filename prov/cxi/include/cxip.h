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

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#ifndef FLOOR
#define FLOOR(a, b) ((long long)(a) - (((long long)(a)) % (b)))
#endif

#ifndef CEILING
#define CEILING(a, b) ((long long)(a) <= 0LL ? 0 : (FLOOR((a)-1, b) + (b)))
#endif

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
	 FI_DIRECTED_RECV | FI_MSG | FI_NAMED_RX_CTX)

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

#ifndef CXIP_EAGER_THRESHOLD
#define CXIP_EAGER_THRESHOLD (2048)
#endif

#ifndef CXIP_MAX_OFLOW_BUFS
#define CXIP_MAX_OFLOW_BUFS (3)
#endif

#ifndef CXIP_MAX_OFLOW_MSGS
#define CXIP_MAX_OFLOW_MSGS (1024)
#endif

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
 * A Cassini NIC Address and PID identify a libfabric Endpoint.  Cassini
 * borrows the name 'PID' from Portals. In CXI, a process can allocate several
 * PID values.
 *
 * The PID value C_PID_ANY is reserved. When used, the library auto-assigns
 * a free PID value. A PID value is assigned when network resources are
 * allocated. Libfabric clients can achieve this by not specifying a 'service'
 * in a call to fi_getinfo() or by not setting src_addr in the fi_info
 * structure used to allocate an Endpoint.
 *
 * TODO: If NIC Address must be non-zero, the valid bit can be removed.
 */
struct cxip_addr {
	union {
		struct {
			uint32_t pid		: C_DFA_PID_BITS_MAX;
			uint32_t nic		: C_DFA_NIC_BITS;
			uint32_t valid		: 1;
		};
		uint32_t raw;
	};
};

/* A PID contains "pid_granule" logical endpoints. The PID granule is set per
 * device and can be found in libCXI devinfo. These endpoints are partitioned
 * by the provider for the following use:
 *
 * 0-MAX_RXC       RX context queues
 * MAX_RXC-(MAX-1) MR key values
 * MAX             Rendezvous read queue
 *
 * The default pid_granule is 256. The default maximum RXC count is 16.
 * Therefore, the mapping is usually:
 *
 * 0-15   RX context queues 0-15
 * 16-254 MR keys 0-238
 * 255    Rendezvous read queue
 */
#define CXIP_PID_RXC_CNT CXIP_EP_MAX_RX_CNT
#define CXIP_PID_MR_CNT(pid_granule) ((pid_granule) - CXIP_PID_RXC_CNT - 1)

#define CXIP_MR_TO_IDX(key) (CXIP_PID_RXC_CNT + (key))
#define CXIP_RXC_TO_IDX(rx_id) (rx_id)

#define CXIP_AV_ADDR_IDX(av, fi_addr) ((uint64_t)fi_addr & av->mask)
#define CXIP_AV_ADDR_RXC(av, fi_addr) \
	(av->rxc_bits ? ((uint64_t)fi_addr >> (64 - av->rxc_bits)) : 0)

/* Messaging Match Bit layout */
#define RDZV_ID_LO_WIDTH 14
#define RDZV_ID_HI_WIDTH 8
#define RDZV_ID_WIDTH (RDZV_ID_LO_WIDTH + RDIVS_ID_HI_WIDTH)
#define RDZV_ID_LO(id) ((id) & ((1 << RDZV_ID_LO_WIDTH) - 1))
#define RDZV_ID_HI(id) \
	(((id) >> RDZV_ID_LO_WIDTH) & ((1 << RDZV_ID_HI_WIDTH) - 1))

union cxip_match_bits {
	struct {
		uint64_t tag        : 48; /* User tag value */
		uint64_t rdzv_id_lo : RDZV_ID_LO_WIDTH;
		uint64_t sink       : 1;  /* Long eager protocol */
		uint64_t tagged     : 1;  /* Tagged API */
	};
	struct {
		uint64_t rdzv_id_hi : RDZV_ID_HI_WIDTH;
	};
	uint64_t raw;
};

#define RDZV_ID(hi, lo) (((hi) << RDZV_ID_LO_WIDTH) | (lo))

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
	struct cxil_lni *if_lni;	// cxil NIC LNI structure
	struct cxi_cp *cps[16];		// Cassini communication profiles
	int n_cps;
	struct dlist_entry if_doms;	// if_domain list
	struct dlist_entry ptes;	// PTE list
	ofi_atomic32_t ref;
	struct cxip_cmdq *mr_cmdq;	// used for all MR activation
	struct cxi_evtq *mr_evtq;	// used for async completion
	void *evtq_buf;
	size_t evtq_buf_len;
	struct cxi_md *evtq_buf_md;
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
 * Command Queue
 *
 * Support structure.
 *
 * Created in cxip_cmdq_alloc().
 */
struct cxip_cmdq {
	struct cxi_cmdq *dev_cmdq;
	fastlock_t lock;
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
	struct util_fabric util_fabric;		// fabric used for MR cache
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
	struct util_domain util_domain; // domain used for MR cache
	struct cxip_fabric *fab;	// parent cxip_fabric
	fastlock_t lock;
	ofi_atomic32_t ref;

	struct cxip_eq *eq;		// linked during cxip_dom_bind()
	struct cxip_eq *mr_eq;		// == eq || == NULL
	struct ofi_mr_cache iomm;	// IO Memory Map
	struct ofi_mem_monitor iomm_mon;// IOMM monitor
	fastlock_t iomm_lock;

	enum fi_progress progress_mode;
	struct dlist_entry dom_list_entry;
					// attaches to global cxip_dom_list
	struct fi_domain_attr attr;	// copy of user or default domain attr

	uint32_t nic_addr;		// dev address of source NIC
	int enabled;			// set when domain is enabled
	struct cxip_if *dev_if;		// looked when domain is enabled
};

/* CXI Provider Memory Descriptor */
struct cxip_md {
	struct cxip_domain *dom;
	struct cxi_md *md;
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
	struct cxip_md *local_md;	// RMA target buffer
};

struct cxip_req_amo {
	struct cxip_md *local_md;	// RMA target buffer
	void *result_buf;		// local buffer for fetch
};

struct cxip_req_recv {
	struct dlist_entry ux_entry;	// UX event list entry
	struct cxip_rxc *rxc;		// receive context
	void *recv_buf;			// local receive buffer
	struct cxip_md *recv_md;	// local receive MD
	int rc;				// DMA return code
	uint32_t rlength;		// DMA requested length
	uint32_t mlength;		// DMA manipulated length
	uint64_t start;			// DMA Overflow buffer address
	uint32_t initiator;		// DMA initiator address
	uint32_t rdzv_id;		// DMA initiator rendezvous ID
	int rdzv_events;		// Processed rdzv event count
};

struct cxip_req_send {
	void *buf;			// local send buffer
	struct cxip_md *send_md;		// message target buffer
	struct cxip_txc *txc;
	size_t length;			// request length
	int rdzv_id;			// SW RDZV ID for long messages
	enum c_return_code event_failure;// SW RDZV Failure status on prev event
	int complete_on_unlink;		// SW RDZV state for expected messages
	union c_cmdu cmd;	// Rendezvous cmd to send after LE is linked
};

struct cxip_req_oflow {
	struct cxip_rxc *rxc;
	struct cxip_oflow_buf *oflow_buf;
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
	fastlock_t rb_lock;

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
	void *evtq_buf;
	size_t evtq_buf_len;
	struct cxi_md *evtq_buf_md;
	fastlock_t req_lock;
	struct ofi_bufpool *req_pool;	// utility pool for cxip_req
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
 *
 * Acts as a record of an unexpected send. Contains the fields from a Put event
 * necessary to correlate the send with an Overflow event.
 */
struct cxip_ux_send {
	struct dlist_entry ux_entry;		// UX event list entry
	struct cxip_oflow_buf *oflow_buf;
	uint64_t start;
	uint32_t initiator;
	uint32_t rdzv_id;
};

enum oflow_buf_type {
	OFLOW_BUF_EAGER = 1, /* Locally-managed eager data overflow buffer */
	OFLOW_BUF_SINK,      /* Truncating overflow buffer for long eager
			      * protocol
			      */
};

/**
 * Overflow buffer
 *
 * Support structure.
 */
struct cxip_oflow_buf {
	struct dlist_entry list;
	enum oflow_buf_type type;
	struct cxip_rxc *rxc;
	void *buf;
	struct cxip_md *md;
	ofi_atomic32_t ref;
	int exhausted;
	int buffer_id;
};

/**
 * Receive Context
 *
 * Support structure.
 *
 * Created in cxip_rxc(), during EP creation.
 */
struct cxip_rxc {
	struct fid_ep ctx;
	fastlock_t lock;		// Control ops lock

	uint16_t rx_id;			// SEP index
	int enabled;
	int progress;			// unused
	int recv_cq_event;		// unused
	int use_shared;

	size_t num_left;		// unused (?) (set, never referenced)
	size_t min_multi_recv;
	uint64_t addr;
	struct cxip_comp comp;
	struct cxip_rxc *srx;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry ep_list;	// contains EPs using shared context

	struct fi_rx_attr attr;

	struct cxip_pte *rx_pte;
	struct cxip_cmdq *rx_cmdq;	// RX CMDQ for posting recvs
	struct cxip_cmdq *tx_cmdq;	// TX CMDQ for Message Gets

	int eager_threshold;

	/* Unexpected message handling */
	ofi_atomic32_t oflow_buf_cnt;
	int oflow_bufs_max;
	int oflow_msgs_max;
	int oflow_buf_size;
	struct dlist_entry oflow_bufs;		// Overflow buffers
	struct dlist_entry ux_sends;		// UX sends records
	struct dlist_entry ux_recvs;		// UX recv records
	struct dlist_entry ux_rdzv_sends;	// UX RDZV send records
	struct dlist_entry ux_rdzv_recvs;	// UX RDZV recv records
	struct cxip_oflow_buf ux_sink_buf;	// Long UX sink buffer
};

#define CXIP_RDZV_BM_LEN (8)

struct cxip_rdzv_ids {
	uint16_t bitmap[CXIP_RDZV_BM_LEN];
};

/**
 * Transmit Context
 *
 * Support structure.
 *
 * Created by cxip_txc_alloc(), during EP creation.
 *
 */
struct cxip_txc {
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
	struct cxip_txc *stx;		// shared context (?)

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry ep_list;	// contains EPs using shared context
	fastlock_t lock;

	struct fi_tx_attr attr;		// attributes

	struct cxip_cmdq *tx_cmdq;	// added during cxip_txc_enable()

	/* Software Rendezvous related structures */
	struct cxip_pte *rdzv_pte;	// PTE for SW Rendezvous commands
	int eager_threshold;		// Threshold for eager IOs
	struct cxip_cmdq *rx_cmdq;	// Target cmdq for Rendezvous buffers
	struct cxip_rdzv_ids rdzv_ids;	// Set of Rendezvous IDs to be used
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

	/* TX/RX contexts. Standard EPs have 1 of each. SEPs have many. */
	struct cxip_rxc **rxcs;		// RX contexts
	struct cxip_txc **txcs;		// TX contexts
	ofi_atomic32_t num_rxc;		// num RX contexts (>= 1)
	ofi_atomic32_t num_txc;		// num TX contexts (>= 1)

	struct fi_info info;		// TODO: use this properly
	struct fi_ep_attr ep_attr;

	int is_enabled;
	fastlock_t lock;

	struct cxip_addr src_addr;	// address of this NIC
	fi_addr_t fi_addr;		// AV address of this EP
	uint32_t vni;			// VNI all EP addressing
	struct cxip_if_domain *if_dom;
	int rdzv_offload;
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
	struct cxi_md *md;		// buffer IO descriptor
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
	uint64_t mask;			// mask with rxc_bits MSbits clear
	int rxc_bits;			// address bits needed for SEP RXs
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

int cxip_cmdq_alloc(struct cxip_if *dev_if, struct cxi_evtq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts,
		    struct cxip_cmdq **cmdq);
void cxip_cmdq_free(struct cxip_cmdq *cmdq);

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

fi_addr_t _cxip_av_reverse_lookup(struct cxip_av *av, uint32_t nic,
				  uint32_t pid);
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

struct cxip_rxc *cxip_rxc_alloc(const struct fi_rx_attr *attr,
				      void *context, int use_shared);
void cxip_rxc_free(struct cxip_rxc *rxc);

int cxip_txc_alloc_rdzv_id(struct cxip_txc *txc);
int cxip_txc_free_rdzv_id(struct cxip_txc *txc, int tag);

int cxip_msg_oflow_init(struct cxip_rxc *rxc);
void cxip_msg_oflow_fini(struct cxip_rxc *rxc);

int cxip_rxc_enable(struct cxip_rxc *rxc);
int cxip_txc_enable(struct cxip_txc *txc);
struct cxip_txc *cxip_txc_alloc(const struct fi_tx_attr *attr, void *context,
				int use_shared);
struct cxip_txc *cxip_stx_alloc(const struct fi_tx_attr *attr, void *context);
void cxip_txc_free(struct cxip_txc *txc);

struct cxip_req *cxip_cq_req_alloc(struct cxip_cq *cq, int remap);
void cxip_cq_req_free(struct cxip_req *req);
void cxip_cq_progress(struct cxip_cq *cq);
int cxip_cq_enable(struct cxip_cq *cxi_cq);
int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context);
int cxip_cq_report_error(struct cxip_cq *cq, struct cxip_req *req, size_t olen,
			 int err, int prov_errno, void *err_data,
			 size_t err_data_size);

void cxip_cntr_add_txc(struct cxip_cntr *cntr, struct cxip_txc *txc);
void cxip_cntr_remove_txc(struct cxip_cntr *cntr, struct cxip_txc *txc);
void cxip_cntr_add_rxc(struct cxip_cntr *cntr, struct cxip_rxc *rxc);
void cxip_cntr_remove_rxc(struct cxip_cntr *cntr, struct cxip_rxc *rxc);
int cxip_cntr_progress(struct cxip_cntr *cntr);
int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);

int cxip_iomm_init(struct cxip_domain *dom);
void cxip_iomm_fini(struct cxip_domain *dom);
int cxip_map(struct cxip_domain *dom, void *buf, unsigned long len,
	     struct cxip_md **md);
void cxip_unmap(struct cxip_md *md);

#define _CXIP_LOG_DBG(subsys, ...) FI_DBG(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_LOG_ERROR(subsys, ...) FI_WARN(&cxip_prov, subsys, __VA_ARGS__)

#endif
