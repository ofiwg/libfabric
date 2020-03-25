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
#include "cxip_faults.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#ifndef FLOOR
#define FLOOR(a, b) ((long long)(a) - (((long long)(a)) % (b)))
#endif

#ifndef CEILING
#define CEILING(a, b) ((long long)(a) <= 0LL ? 0 : (FLOOR((a)-1, b) + (b)))
#endif

#define CXIP_REQ_CLEANUP_TO		3000

#define CXIP_BUFFER_ID_MAX		(1 << 16)

#define CXIP_EP_MAX_MSG_SZ		(1 << 30)
#define CXIP_EP_MAX_TX_CNT		16
#define CXIP_EP_MAX_RX_CNT		16
#define CXIP_EP_MIN_MULTI_RECV		64
#define CXIP_EP_MAX_MULTI_RECV		((1 << 24) - 1)
#define CXIP_EP_MAX_CTX_BITS		4

#define CXIP_TX_COMP_MODES		(FI_INJECT_COMPLETE | \
					 FI_TRANSMIT_COMPLETE | \
					 FI_DELIVERY_COMPLETE | \
					 FI_MATCH_COMPLETE)
#define CXIP_TX_OP_FLAGS		(FI_INJECT | \
					 FI_COMPLETION | \
					 CXIP_TX_COMP_MODES | \
					 FI_REMOTE_CQ_DATA | \
					 FI_MORE)
#define CXIP_RX_OP_FLAGS		(FI_COMPLETION | \
					 FI_MULTI_RECV | \
					 FI_MORE)
#define CXIP_WRITEMSG_ALLOWED_FLAGS	(FI_INJECT | \
					 FI_COMPLETION | \
					 FI_MORE | \
					 CXIP_TX_COMP_MODES)
#define CXIP_READMSG_ALLOWED_FLAGS	(FI_COMPLETION | \
					 FI_MORE | \
					 CXIP_TX_COMP_MODES)

#define CXIP_AMO_MAX_IOV		1
#define CXIP_EQ_DEF_SZ			(1 << 8)
#define CXIP_CQ_DEF_SZ			(1 << 8)
#define CXIP_AV_DEF_SZ			(1 << 8)

#define CXIP_RDZV_THRESHOLD		2048
#define CXIP_OFLOW_BUF_SIZE		(2*1024*1024)
#define CXIP_OFLOW_BUF_COUNT		3
#define CXIP_UX_BUFFER_SIZE		(CXIP_OFLOW_BUF_COUNT * \
					 CXIP_OFLOW_BUF_SIZE)

#define CXIP_EP_PRI_CAPS \
	(FI_RMA | FI_ATOMICS | FI_TAGGED | FI_RECV | FI_SEND | \
	 FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
	 FI_DIRECTED_RECV | FI_MSG | FI_NAMED_RX_CTX)
#define CXIP_EP_SEC_CAPS \
	(FI_SOURCE | FI_SHARED_AV | FI_LOCAL_COMM | FI_REMOTE_COMM | \
	 FI_RMA_EVENT | FI_MULTI_RECV)
	 /* TODO FI_FENCE | FI_TRIGGER */
#define CXIP_EP_CAPS (CXIP_EP_PRI_CAPS | CXIP_EP_SEC_CAPS)
#define CXIP_MSG_ORDER			(FI_ORDER_SAS | \
					 FI_ORDER_WAW | \
					 FI_ORDER_RMA_WAW | \
					 FI_ORDER_ATOMIC_WAW | \
					 FI_ORDER_ATOMIC_WAR | \
					 FI_ORDER_ATOMIC_RAW | \
					 FI_ORDER_ATOMIC_RAR)

#define CXIP_EP_CQ_FLAGS \
	(FI_SEND | FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION)
#define CXIP_EP_CNTR_FLAGS \
	(FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | \
	 FI_REMOTE_WRITE)

#define CXIP_INJECT_SIZE		C_MAX_IDC_PAYLOAD_UNR

#define CXIP_MAJOR_VERSION		0
#define CXIP_MINOR_VERSION		0
#define CXIP_PROV_VERSION		FI_VERSION(CXIP_MAJOR_VERSION, \
						   CXIP_MINOR_VERSION)
#define CXIP_FI_VERSION			FI_VERSION(1, 9)
#define CXIP_WIRE_PROTO_VERSION		1

#define CXIP_CNTR_SUCCESS_MAX ((1ULL << C_CT_SUCCESS_BITS) - 1)
#define CXIP_CNTR_FAILURE_MAX ((1ULL << C_CT_FAILURE_BITS) - 1)

static const char cxip_dom_fmt[] = "cxi%d";
extern char cxip_prov_name[];
extern struct fi_provider cxip_prov;
extern struct util_prov cxip_util_prov;

extern int cxip_av_def_sz;
extern int cxip_cq_def_sz;
extern int cxip_eq_def_sz;

extern struct slist cxip_if_list;

extern struct fi_fabric_attr cxip_fabric_attr;
extern struct fi_domain_attr cxip_domain_attr;
extern struct fi_ep_attr cxip_ep_attr;
extern struct fi_tx_attr cxip_tx_attr;
extern struct fi_rx_attr cxip_rx_attr;

struct cxip_environment {
	/* Translation */
	int odp;
	int ats;

	/* Messaging */
	int rdzv_offload;
	size_t rdzv_threshold;
	size_t oflow_buf_size;
	size_t oflow_buf_count;

	int optimized_mrs;
};

extern struct cxip_environment cxip_env;

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

#define CXIP_ADDR_EQUAL(a, b) ((a).nic == (b).nic && (a).pid == (b).pid)

#define CXIP_AV_ADDR_IDX(av, fi_addr) ((uint64_t)fi_addr & av->mask)
#define CXIP_AV_ADDR_RXC(av, fi_addr) \
	(av->rxc_bits ? ((uint64_t)fi_addr >> (64 - av->rxc_bits)) : 0)

/* A PID contains "pid_granule" logical endpoints. The PID granule is set per
 * device and can be found in libCXI devinfo. The default pid_granule is 256.
 * The default maximum RXC count is 16. These endpoints are partitioned by the
 * provider for the following use:
 *
 * 0-15   RX Queue PtlTEs 0-15
 * 50-250 Optimized MR PtlTEs 0-199
 * 254    Standard MR PtlTE
 * 255    Rendezvous source PtlTE
 */
#define CXIP_PTL_IDX_MR_OPT_BASE	50
#define CXIP_PTL_IDX_MR_OPT_CNT		200

#define CXIP_PTL_IDX_RXC(rx_id)		(rx_id)
#define CXIP_PTL_IDX_MR_OPT(key)	(CXIP_PTL_IDX_MR_OPT_BASE + (key))
#define CXIP_PTL_IDX_CTRL		254
#define CXIP_PTL_IDX_RDZV_SRC		255

static inline bool cxip_mr_key_opt(int key)
{
	return cxip_env.optimized_mrs && key < CXIP_PTL_IDX_MR_OPT_CNT;
}

static inline int cxip_mr_key_to_ptl_idx(int key)
{
	if (cxip_mr_key_opt(key))
		return CXIP_PTL_IDX_MR_OPT((key));
	return CXIP_PTL_IDX_CTRL;
}

/* Messaging Match Bit layout */
#define CXIP_TAG_WIDTH		48
#define CXIP_RDZV_ID_WIDTH	8
#define CXIP_TX_ID_WIDTH	12
#define CXIP_TAG_MASK		((1UL << CXIP_TAG_WIDTH) - 1)

/* Define several types of LEs */
enum cxip_le_type {
	CXIP_LE_TYPE_RX = 0,	/* RX data LE */
	CXIP_LE_TYPE_SINK,	/* Truncating RX LE */
	CXIP_LE_TYPE_ZBP,	/* Zero-byte Put control message LE. Used to
				 * exchange data in the EQ header_data and
				 * match_bits fields. Unexpected headers are
				 * disabled.
				 */
};

enum cxip_ctrl_le_type {
	CXIP_CTRL_LE_TYPE_MR = 0,	/* Memory Region LE */
};

union cxip_match_bits {
	struct {
		uint64_t tag        : CXIP_TAG_WIDTH; /* User tag value */
		uint64_t tx_id      : CXIP_TX_ID_WIDTH; /* Prov. tracked ID */
		uint64_t tagged     : 1;  /* Tagged API */
		uint64_t match_comp : 1;  /* Notify initiator on match */
		uint64_t le_type    : 2;
	};
	/* Split TX ID for rendezvous operations. */
	struct {
		uint64_t pad0       : CXIP_TAG_WIDTH; /* User tag value */
		uint64_t rdzv_id_hi : CXIP_RDZV_ID_WIDTH;
		uint64_t rdzv_lac   : 4;  /* Rendezvous Get LAC */
	};
	struct {
		uint64_t rdzv_id_lo : CXIP_RDZV_ID_WIDTH;
	};
	/* Control LE match bit format */
	struct {
		uint64_t mr_key       : 63;
		uint64_t ctrl_le_type : 1;
	};
	uint64_t raw;
};


/* libcxi Wrapper Structures */

/**
 * CXI Device wrapper
 *
 * There will be one of these for every local Cassini device on the node.
 */
struct cxip_if {
	struct slist_entry if_entry;

	/* Device description */
	struct cxil_devinfo info;
	struct cxil_dev *dev;

	/* PtlTEs (searched during state change events) */
	struct dlist_entry ptes;

	ofi_atomic32_t ref;
	fastlock_t lock;
};

/**
 * CXI Logical Network Interface (LNI) wrapper
 *
 * An LNI is a container used allocate resources from a NIC.
 */
struct cxip_lni {
	struct cxip_if *iface;
	struct cxil_lni *lni;

	/* Resource cache */
};

/**
 * CXI Device Domain wrapper
 *
 * A CXI domain is conceptually equivalent to a Portals table. The provider
 * assigns a unique domain to each OFI Endpoint. A domain is addressed using
 * the tuple { NIC, VNI, PID }.
 */
struct cxip_if_domain {
	struct cxip_lni *lni;
	struct cxil_domain *dom;
};

/**
 * CXI Portal Table Entry (PtlTE) wrapper
 *
 * Represents PtlTE mapped in a CXI domain.
 */
struct cxip_pte {
	struct dlist_entry pte_entry;
	struct cxip_if_domain *if_dom;
	uint64_t pid_idx;
	struct cxil_pte *pte;
	struct cxil_pte_map *pte_map;

	void (*state_change_cb)(struct cxip_pte *pte,
				enum c_ptlte_state state);
	void *ctx;
};

/**
 * CXI Command Queue wrapper
 */
struct cxip_cmdq {
	struct cxi_cmdq *dev_cmdq;
	fastlock_t lock;
	struct c_cstate_cmd c_state;
};


/* OFI Provider Structures */

/**
 * CXI Provider Fabric object
 */
struct cxip_fabric {
	struct util_fabric util_fabric;
	ofi_atomic32_t ref;
};

struct cxip_domain;

/*
 * CXI Provider Memory Descriptor
 */
struct cxip_md {
	struct cxip_domain *dom;
	struct cxi_md *md;
};

/*
 * CXI Provider Domain object
 */
struct cxip_domain {
	struct util_domain util_domain;
	struct cxip_fabric *fab;
	fastlock_t lock;
	ofi_atomic32_t ref;

	struct cxip_eq *eq; //unused
	struct cxip_eq *mr_eq; //unused

	/* Assigned NIC address */
	uint32_t nic_addr;

	/* Device info */
	struct cxip_if *iface;

	/* Device partition */
	struct cxip_lni *lni;

	/* Communication Profiles */
	struct cxi_cp *cps[16];
	int n_cps;

	/* Trigger and CT support */
	struct cxip_cmdq *trig_cmdq;
	bool cntr_init;

	/* Translation cache */
	struct ofi_mr_cache iomm;
	fastlock_t iomm_lock;
	bool odp;

	/* ATS translation support */
	struct cxip_md ats_md;
	bool ats_init;
	bool ats_enabled;

	/* Domain state */
	bool enabled;
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
	struct cxip_txc *txc;
	struct cxip_md *local_md;	// RMA target buffer
};

struct cxip_req_amo {
	struct cxip_txc *txc;
	struct cxip_md *result_md;
	struct cxip_md *oper1_md;
	char result[16];
	char oper1[16];
	bool tmp_result;
	bool tmp_oper1;
};

struct cxip_req_recv {
	/* Receive parameters */
	struct dlist_entry ux_entry;	// UX event list entry
	struct cxip_rxc *rxc;		// receive context
	void *recv_buf;			// local receive buffer
	struct cxip_md *recv_md;	// local receive MD
	uint32_t ulen;			// User buffer length
	bool tagged;
	uint64_t tag;
	uint64_t ignore;
	uint32_t match_id;
	uint64_t flags;

	/* Control info */
	int rc;				// DMA return code
	uint32_t rlen;			// Send length
	uint64_t oflow_start;		// Overflow buffer address
	uint32_t initiator;		// DMA initiator address
	uint32_t rdzv_id;		// DMA initiator rendezvous ID
	int rdzv_events;		// Processed rdzv event count
	bool canceled;			// Request canceled?
	bool multi_recv;
	bool rdzv_tgt_event;
	uint64_t start_offset;
	uint64_t mrecv_bytes;
	struct cxip_req *parent;
	struct dlist_entry children;
	uint64_t src_offset;
};

struct cxip_req_send {
	/* Send parameters */
	struct cxip_txc *txc;
	const void *buf;		// local send buffer
	size_t len;			// request length
	struct cxip_md *send_md;	// send buffer memory descriptor
	struct cxip_addr caddr;
	uint8_t rxc_id;
	bool tagged;
	uint64_t tag;
	uint64_t data;
	uint64_t flags;

	/* Control info */
	struct dlist_entry txc_entry;
	struct cxip_fc_peer *fc_peer;
	union {
		int rdzv_id;		// SW RDZV ID for long messages
		int tx_id;
	};
	int rc;				// DMA return code
	int long_send_events;		// Processed event count
};

struct cxip_req_oflow {
	union {
		struct cxip_txc *txc;
		struct cxip_rxc *rxc;
	};
	struct cxip_oflow_buf *oflow_buf;
};

struct cxip_req_rdzv_src {
	struct dlist_entry list;
	struct cxip_txc *txc;
	uint32_t lac;
	int rc;
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
	struct dlist_entry cq_entry;
	void *req_ctx;
	struct cxip_cq *cq;		// request CQ
	int req_id;			// fast lookup in index table
	int (*cb)(struct cxip_req *req, const union c_event *evt);
					// completion event callback
	bool discard;

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
		struct cxip_req_rdzv_src rdzv_src;
	};
};

struct cxip_ctrl_req_mr {
	struct cxip_mr *mr;
};

struct cxip_ctrl_req {
	struct dlist_entry ep_entry;
	struct cxip_ep_obj *ep_obj;
	int req_id;
	int (*cb)(struct cxip_ctrl_req *req, const union c_event *evt);

	union {
		struct cxip_ctrl_req_mr mr;
	};
};

/**
 * Completion Queue
 *
 * libfabric fi_cq implementation.
 *
 * Created in cxip_cq_open().
 */
struct cxip_cq {
	struct util_cq util_cq;
	struct fi_cq_attr attr;
	ofi_atomic32_t ref;

	/* CXI specific fields. */
	struct cxip_domain *domain;
	struct cxip_ep_obj *ep_obj;
	fastlock_t lock;
	bool enabled;
	struct cxi_evtq *evtq;
	void *evtq_buf;
	size_t evtq_buf_len;
	struct cxi_md *evtq_buf_md;
	fastlock_t req_lock;
	struct ofi_bufpool *req_pool;
	struct indexer req_table;
	struct dlist_entry req_list;
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

	fastlock_t lock;
	bool enabled;

	struct cxi_ct *ct;
	struct c_ct_writeback wb;
	bool wb_pending;
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
	struct cxip_req *req;
	uint64_t start;
	uint32_t initiator;
	uint32_t rdzv_id;
	uint64_t src_offset;
	uint32_t rlen;
	uint32_t mlen;
	union cxip_match_bits mb;
	uint64_t data;
};

/**
 * Overflow buffer
 *
 * Support structure.
 */
struct cxip_oflow_buf {
	struct dlist_entry list;
	enum cxip_le_type type;
	union {
		struct cxip_txc *txc;
		struct cxip_rxc *rxc;
	};
	void *buf;
	struct cxip_md *md;
	int min_bytes;
	int buffer_id;
};

enum cxip_pte_state {
	CXIP_PTE_DISABLED = 1,
	CXIP_PTE_ENABLED,
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
	bool enabled;

	int use_shared;
	struct cxip_rxc *srx;

	struct cxip_cq *recv_cq;
	struct cxip_cntr *recv_cntr;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry ep_list;	// contains EPs using shared context

	struct fi_rx_attr attr;
	bool selective_completion;

	struct cxip_pte *rx_pte;	// HW RX Queue
	enum cxip_pte_state pte_state;
	struct cxip_cmdq *rx_cmdq;	// RX CMDQ for posting receive buffers
	struct cxip_cmdq *tx_cmdq;	// TX CMDQ for Message Gets

	ofi_atomic32_t orx_reqs;	// outstanding receive requests

	int min_multi_recv;
	int rdzv_threshold;

	/* Unexpected message handling */
	fastlock_t rx_lock;			// RX message lock
	ofi_atomic32_t oflow_bufs_submitted;
	ofi_atomic32_t oflow_bufs_linked;
	ofi_atomic32_t oflow_bufs_in_use;
	int oflow_buf_size;
	int oflow_bufs_max;
	struct dlist_entry oflow_bufs;		// Overflow buffers
	struct dlist_entry ux_sends;		// UX sends records
	struct dlist_entry ux_recvs;		// UX recv records
	struct dlist_entry ux_rdzv_sends;	// UX RDZV send records
	struct dlist_entry ux_rdzv_recvs;	// UX RDZV recv records

	/* Long eager send handling */
	ofi_atomic32_t sink_le_linked;
	struct cxip_oflow_buf sink_le;		// Long UX sink buffer
};

#define CXIP_RDZV_IDS	(1 << CXIP_RDZV_ID_WIDTH)
#define CXIP_TX_IDS	(1 << CXIP_TX_ID_WIDTH)

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
	bool enabled;

	int use_shared;
	struct cxip_txc *stx;		// shared context (?)

	struct cxip_cq *send_cq;
	struct cxip_cntr *send_cntr;
	struct cxip_cntr *read_cntr;
	struct cxip_cntr *write_cntr;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain

	struct dlist_entry ep_list;	// contains EPs using shared context
	fastlock_t lock;

	struct fi_tx_attr attr;		// attributes
	bool selective_completion;

	struct cxip_cmdq *tx_cmdq;	// added during cxip_txc_enable()

	ofi_atomic32_t otx_reqs;	// outstanding transmit requests
	struct cxip_req *rma_inject_req;
	struct cxip_req *amo_inject_req;

	/* Software Rendezvous related structures */
	struct cxip_pte *rdzv_pte;	// PTE for SW Rendezvous commands
	enum cxip_pte_state pte_state;
	int rdzv_threshold;
	struct cxip_cmdq *rx_cmdq;	// Target cmdq for Rendezvous buffers
	ofi_atomic32_t rdzv_src_lacs;	// Bitmask of LACs
	struct dlist_entry rdzv_src_reqs;
	fastlock_t rdzv_src_lock;

	/* Header message handling */
	ofi_atomic32_t zbp_le_linked;
	struct cxip_oflow_buf zbp_le;	// Zero-byte Put LE

	/* Flow Control recovery */
	struct dlist_entry msg_queue;
	struct dlist_entry fc_peers;
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

	/* Shared context resources */
	struct cxip_cmdq *txqs[CXIP_EP_MAX_TX_CNT];
	ofi_atomic32_t txq_refs[CXIP_EP_MAX_TX_CNT];
	struct cxip_cmdq *tgqs[CXIP_EP_MAX_TX_CNT];
	ofi_atomic32_t tgq_refs[CXIP_EP_MAX_TX_CNT];

	struct fi_ep_attr ep_attr;

	bool enabled;
	fastlock_t lock;

	struct cxip_addr src_addr;	// address of this NIC
	fi_addr_t fi_addr;		// AV address of this EP
	uint32_t vni;			// VNI all EP addressing
	int rdzv_offload;

	struct cxip_if_domain *if_dom;

	struct indexer rdzv_ids;
	fastlock_t rdzv_id_lock;

	struct indexer tx_ids;
	fastlock_t tx_id_lock;

	/* Control resources */
	struct cxip_cmdq *ctrl_tgq;
	struct cxi_evtq *ctrl_evtq;
	void *ctrl_evtq_buf;
	size_t ctrl_evtq_buf_len;
	struct cxi_md *ctrl_evtq_buf_md;
	struct cxip_pte *ctrl_pte;
	struct indexer req_ids;
	struct dlist_entry mr_list;
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

enum cxip_mr_state {
	CXIP_MR_DISABLED = 1,
	CXIP_MR_ENABLED,
	CXIP_MR_LINKED,
	CXIP_MR_UNLINKED,
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
	fastlock_t lock;

	bool enabled;
	struct cxip_pte *pte;
	enum cxip_mr_state mr_state;
	struct cxip_ctrl_req req;
	bool optimized;

	void *buf;			// memory buffer VA
	uint64_t len;			// memory length
	struct cxip_md *md;		// buffer IO descriptor
	struct dlist_entry ep_entry;
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
int cxip_alloc_lni(struct cxip_if *iface, struct cxip_lni **if_lni);
void cxip_free_lni(struct cxip_lni *lni);
int cxip_alloc_if_domain(struct cxip_lni *lni, uint32_t vni, uint32_t pid,
			 struct cxip_if_domain **if_dom);
void cxip_free_if_domain(struct cxip_if_domain *if_dom);
void cxip_if_init(void);
void cxip_if_fini(void);

int cxip_pte_append(struct cxip_pte *pte, uint64_t iova, size_t len,
		    unsigned int lac, enum c_ptl_list list,
		    uint32_t buffer_id, uint64_t match_bits,
		    uint64_t ignore_bits, uint32_t match_id,
		    uint64_t min_free, uint32_t flags,
		    struct cxip_cntr *cntr, struct cxip_cmdq *cmdq,
		    bool ring);
int cxip_pte_unlink(struct cxip_pte *pte, enum c_ptl_list list,
		    int buffer_id, struct cxip_cmdq *cmdq);
int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_evtq *evtq,
		   uint64_t pid_idx, struct cxi_pt_alloc_opts *opts,
		   void (*state_change_cb)(struct cxip_pte *pte,
					   enum c_ptlte_state state),
		   void *ctx, struct cxip_pte **pte);
void cxip_pte_free(struct cxip_pte *pte);
int cxip_pte_state_change(struct cxip_if *dev_if, uint32_t pte_num,
			  enum c_ptlte_state new_state);

int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_evtq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts,
		    struct cxip_cmdq **cmdq);
void cxip_cmdq_free(struct cxip_cmdq *cmdq);

int cxip_domain_enable(struct cxip_domain *dom);
int cxip_domain(struct fid_fabric *fabric, struct fi_info *info,
		struct fid_domain **dom, void *context);

fi_addr_t _cxip_av_reverse_lookup(struct cxip_av *av, uint32_t nic,
				  uint32_t pid);
int _cxip_av_lookup(struct cxip_av *av, fi_addr_t fi_addr,
		    struct cxip_addr *addr);
int cxip_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context);

int cxip_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric,
		void *context);

int cxip_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep, void *context);
int cxip_scalable_ep(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **sep, void *context);

int cxip_wait_get_obj(struct fid_wait *fid, void *arg);
void cxip_wait_signal(struct fid_wait *wait_fid);
int cxip_wait_close(fid_t fid);
int cxip_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		   struct fid_wait **waitset);

int cxip_tx_id_alloc(struct cxip_ep_obj *ep_obj, void *ctx);
int cxip_tx_id_free(struct cxip_ep_obj *ep_obj, int id);
void *cxip_tx_id_lookup(struct cxip_ep_obj *ep_obj, int id);

int cxip_rdzv_id_alloc(struct cxip_ep_obj *ep_obj, void *ctx);
int cxip_rdzv_id_free(struct cxip_ep_obj *ep_obj, int id);
void *cxip_rdzv_id_lookup(struct cxip_ep_obj *ep_obj, int id);
int cxip_ep_cmdq(struct cxip_ep_obj *ep_obj,
		 uint32_t ctx_id, uint32_t size, bool transmit,
		 struct cxip_cmdq **cmdq);
void cxip_ep_cmdq_put(struct cxip_ep_obj *ep_obj,
		      uint32_t ctx_id, bool transmit);

int cxip_recv_cancel(struct cxip_req *req);
void cxip_recv_pte_cb(struct cxip_pte *pte, enum c_ptlte_state state);
int cxip_rxc_oflow_init(struct cxip_rxc *rxc);
void cxip_rxc_oflow_fini(struct cxip_rxc *rxc);
int cxip_txc_zbp_init(struct cxip_txc *txc);
int cxip_txc_zbp_fini(struct cxip_txc *txc);
int cxip_txc_rdzv_src_fini(struct cxip_txc *txc);

struct cxip_txc *cxip_txc_alloc(const struct fi_tx_attr *attr, void *context,
				int use_shared);
int cxip_txc_enable(struct cxip_txc *txc);
struct cxip_txc *cxip_stx_alloc(const struct fi_tx_attr *attr, void *context);
void cxip_txc_free(struct cxip_txc *txc);

struct cxip_rxc *cxip_rxc_alloc(const struct fi_rx_attr *attr,
				      void *context, int use_shared);
int cxip_rxc_enable(struct cxip_rxc *rxc);
void cxip_rxc_free(struct cxip_rxc *rxc);

int cxip_cq_req_cancel(struct cxip_cq *cq, void *req_ctx, void *op_ctx,
		       bool match);
void cxip_cq_req_discard(struct cxip_cq *cq, void *req_ctx);
int cxip_cq_req_complete(struct cxip_req *req);
int cxip_cq_req_complete_addr(struct cxip_req *req, fi_addr_t src);
int cxip_cq_req_error(struct cxip_req *req, size_t olen,
		      int err, int prov_errno, void *err_data,
		      size_t err_data_size);
struct cxip_req *cxip_cq_req_alloc(struct cxip_cq *cq, int remap,
				   void *req_ctx);
void cxip_cq_req_free(struct cxip_req *req);
void cxip_cq_progress(struct cxip_cq *cq);
int cxip_cq_enable(struct cxip_cq *cxi_cq);
int cxip_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq, void *context);

void cxip_dom_cntr_disable(struct cxip_domain *dom);
int cxip_cntr_mod(struct cxip_cntr *cxi_cntr, uint64_t value, bool set,
		  bool err);
int cxip_cntr_enable(struct cxip_cntr *cxi_cntr);
int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);

int cxip_iomm_init(struct cxip_domain *dom);
void cxip_iomm_fini(struct cxip_domain *dom);
int cxip_map(struct cxip_domain *dom, const void *buf, unsigned long len,
	     struct cxip_md **md);
void cxip_unmap(struct cxip_md *md);

void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj);
int cxip_ep_ctrl_init(struct cxip_ep_obj *ep_obj);
void cxip_ep_ctrl_fini(struct cxip_ep_obj *ep_obj);

/*
 * cxip_fid_to_txc() - Return TXC from FID provided to a transmit API.
 */
static inline int cxip_fid_to_txc(struct fid_ep *ep, struct cxip_txc **txc)
{
	struct cxip_ep *cxi_ep;

	if (!ep)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing a TX
	 * context), or a TX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		*txc = cxi_ep->ep_obj->txcs[0];
		return FI_SUCCESS;

	case FI_CLASS_TX_CTX:
		*txc = container_of(ep, struct cxip_txc, fid.ctx);
		return FI_SUCCESS;

	default:
		return -FI_EINVAL;
	}
}

/*
 * cxip_fid_to_rxc() - Return RXC from FID provided to a Receive API.
 */
static inline int cxip_fid_to_rxc(struct fid_ep *ep, struct cxip_rxc **rxc)
{
	struct cxip_ep *cxi_ep;

	if (!ep)
		return -FI_EINVAL;

	/* The input FID could be a standard endpoint (containing an RX
	 * context), or an RX context itself.
	 */
	switch (ep->fid.fclass) {
	case FI_CLASS_EP:
		cxi_ep = container_of(ep, struct cxip_ep, ep);
		*rxc = cxi_ep->ep_obj->rxcs[0];
		return FI_SUCCESS;

	case FI_CLASS_RX_CTX:
		*rxc = container_of(ep, struct cxip_rxc, ctx);
		return FI_SUCCESS;

	default:
		return -FI_EINVAL;
	}
}

#define _CXIP_LOG_DBG(subsys, ...) FI_DBG(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_LOG_ERROR(subsys, ...) FI_WARN(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_LOG_INFO(subsys, ...) FI_INFO(&cxip_prov, subsys, __VA_ARGS__)

#endif
