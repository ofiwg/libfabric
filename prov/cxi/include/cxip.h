/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018-2020 Cray Inc. All rights reserved.
 * Copyright (c) 2021 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_PROV_H_
#define _CXIP_PROV_H_

#include <netinet/ether.h>
#include "config.h"

#include <pthread.h>
#include <json-c/json.h>

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
#include <ofi_mem.h>
#include <unistd.h>

#include "libcxi/libcxi.h"
#include "cxip_faults.h"
#include "fi_cxi_ext.h"

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

#define CXIP_EP_MAX_CTX_BITS		8
#define CXIP_EP_MAX_TX_CNT		(1 << CXIP_EP_MAX_CTX_BITS)
#define CXIP_EP_MAX_RX_CNT		(1 << CXIP_EP_MAX_CTX_BITS)

#define CXIP_EP_MAX_MSG_SZ		(1 << 30)
#define CXIP_EP_MIN_MULTI_RECV		64
#define CXIP_EP_MAX_MULTI_RECV		((1 << 24) - 1)

#define CXIP_TX_COMP_MODES		(FI_INJECT_COMPLETE | \
					 FI_TRANSMIT_COMPLETE | \
					 FI_DELIVERY_COMPLETE | \
					 FI_MATCH_COMPLETE)
#define CXIP_TX_OP_FLAGS		(FI_INJECT | \
					 FI_COMPLETION | \
					 CXIP_TX_COMP_MODES | \
					 FI_REMOTE_CQ_DATA | \
					 FI_MORE | \
					 FI_FENCE)
#define CXIP_RX_OP_FLAGS		(FI_COMPLETION | \
					 FI_MULTI_RECV | \
					 FI_MORE)
/* Invalid OP flags for RX that can be silently ignored */
#define CXIP_RX_IGNORE_OP_FLAGS		(FI_REMOTE_CQ_DATA | \
					 FI_INJECT)
#define CXIP_WRITEMSG_ALLOWED_FLAGS	(FI_INJECT | \
					 FI_COMPLETION | \
					 FI_MORE | \
					 FI_FENCE | \
					 CXIP_TX_COMP_MODES)
#define CXIP_READMSG_ALLOWED_FLAGS	(FI_COMPLETION | \
					 FI_MORE | \
					 FI_FENCE | \
					 CXIP_TX_COMP_MODES)

#define CXIP_AMO_MAX_IOV		1
#define CXIP_EQ_DEF_SZ			(1 << 8)
#define CXIP_CQ_DEF_SZ			1024U
#define CXIP_AV_DEF_SZ			(1 << 8)

#define CXIP_PTE_IGNORE_DROPS		((1 << 24) - 1)
#define CXIP_RDZV_THRESHOLD		2048
#define CXIP_OFLOW_BUF_SIZE		(2*1024*1024)
#define CXIP_OFLOW_BUF_COUNT		3
#define CXIP_REQ_BUF_SIZE		(2*1024*1024)
#define CXIP_REQ_BUF_MIN_POSTED		3
#define CXIP_REQ_BUF_MAX_COUNT		10
#define CXIP_UX_BUFFER_SIZE		(CXIP_OFLOW_BUF_COUNT * \
					 CXIP_OFLOW_BUF_SIZE)

/* When device memory is safe to access via load/store then the
 * CPU will be used to move data below this threshold.
 */
#define CXIP_SAFE_DEVMEM_COPY_THRESH	4096

#define CXIP_EP_PRI_CAPS \
	(FI_RMA | FI_ATOMICS | FI_TAGGED | FI_RECV | FI_SEND | \
	 FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE | \
	 FI_DIRECTED_RECV | FI_MSG | FI_NAMED_RX_CTX | \
	 FI_COLLECTIVE | FI_HMEM)
#define CXIP_EP_SEC_CAPS \
	(FI_SOURCE | FI_SHARED_AV | FI_LOCAL_COMM | FI_REMOTE_COMM | \
	 FI_RMA_EVENT | FI_MULTI_RECV | FI_FENCE | FI_TRIGGER | \
	 FI_COLLECTIVE)
#define CXIP_EP_CAPS (CXIP_EP_PRI_CAPS | CXIP_EP_SEC_CAPS)
#define CXIP_MSG_ORDER			(FI_ORDER_SAS | \
					 FI_ORDER_WAW | \
					 FI_ORDER_RMA_WAW | \
					 FI_ORDER_ATOMIC_WAW | \
					 FI_ORDER_ATOMIC_WAR | \
					 FI_ORDER_ATOMIC_RAW | \
					 FI_ORDER_ATOMIC_RAR)

#define CXIP_EP_CQ_FLAGS \
	(FI_SEND | FI_TRANSMIT | FI_RECV | FI_SELECTIVE_COMPLETION | \
	 FI_COLLECTIVE)
#define CXIP_EP_CNTR_FLAGS \
	(FI_SEND | FI_RECV | FI_READ | FI_WRITE | FI_REMOTE_READ | \
	 FI_REMOTE_WRITE)

#define CXIP_INJECT_SIZE		C_MAX_IDC_PAYLOAD_UNR

/* Max TX size of 16,384 translate to a 4MiB command queue buffer. */
#define CXIP_MAX_TX_SIZE		16384U
#define CXIP_DEFAULT_TX_SIZE		256U

#define CXIP_MAJOR_VERSION		0
#define CXIP_MINOR_VERSION		0
#define CXIP_PROV_VERSION		FI_VERSION(CXIP_MAJOR_VERSION, \
						   CXIP_MINOR_VERSION)
#define CXIP_FI_VERSION			FI_VERSION(1, 14)
#define CXIP_WIRE_PROTO_VERSION		1

#define	CXIP_COLL_MAX_CONCUR		8
#define	CXIP_COLL_MIN_RX_BUFS		3
#define	CXIP_COLL_MIN_RX_SIZE		4096
#define	CXIP_COLL_MIN_FREE		64
#define	CXIP_COLL_MAX_TX_SIZE		32
#define	CXIP_COLL_MAX_SEQNO		(1 << 10)

#define CXIP_REQ_BUF_HEADER_MAX_SIZE (sizeof(struct c_port_fab_hdr) + \
	sizeof(struct c_port_unrestricted_hdr))
#define CXIP_REQ_BUF_HEADER_MIN_SIZE (sizeof(struct c_port_fab_hdr) + \
	sizeof(struct c_port_small_msg_hdr))

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

enum cxip_ats_mlock_mode {
	CXIP_ATS_MLOCK_OFF,
	CXIP_ATS_MLOCK_CACHE,
	CXIP_ATS_MLOCK_ALL,
};

enum cxip_llring_mode {
	CXIP_LLRING_NEVER,
	CXIP_LLRING_IDLE,
	CXIP_LLRING_ALWAYS,
};

enum cxip_ep_ptle_mode {
	CXIP_PTLTE_HARDWARE_MODE,
	CXIP_PTLTE_DEFAULT_MODE = CXIP_PTLTE_HARDWARE_MODE,
	CXIP_PTLTE_SOFTWARE_MODE,
	CXIP_PTLTE_HYBRID_MODE,
};

struct cxip_environment {
	/* Translation */
	int odp;
	int ats;
	int iotlb;
	enum cxip_ats_mlock_mode ats_mlock_mode;

	/* Messaging */
	enum cxip_ep_ptle_mode rx_match_mode;
	int msg_offload;
	int rdzv_offload;
	int hybrid_preemptive;
	int hybrid_recv_preemptive;
	size_t rdzv_threshold;
	size_t rdzv_get_min;
	size_t rdzv_eager_size;
	size_t oflow_buf_size;
	size_t oflow_buf_count;
	size_t safe_devmem_copy_threshold;
	size_t req_buf_size;
	size_t req_buf_min_posted;
	size_t req_buf_max_count;
	int msg_lossless;
	size_t default_cq_size;
	int optimized_mrs;
	int disable_cq_hugetlb;
	int zbcoll_radix;

	enum cxip_llring_mode llring_mode;

	int cq_policy;

	size_t default_vni;

	size_t eq_ack_batch_size;
	int fc_retry_usec_delay;
	size_t ctrl_rx_eq_max_size;
	char *device_name;
	size_t cq_fill_percent;
};

extern struct cxip_environment cxip_env;

static inline bool cxip_software_pte_allowed(void)
{
	return (cxip_env.rdzv_offload &&
		(cxip_env.rx_match_mode == CXIP_PTLTE_SOFTWARE_MODE ||
		 cxip_env.rx_match_mode == CXIP_PTLTE_HYBRID_MODE));
}

/*
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

/*
 * A PID contains "pid_granule" logical endpoints. The PID granule is set per
 * device and can be found in libCXI devinfo. The default pid_granule is 256.
 * The default maximum RXC count is 16. These endpoints are partitioned by the
 * provider for the following use:
 *
 * 0       RX Queue PtlTE
 * 16      Collective PtlTE entry
 * 17-116  Optimized write MR PtlTEs 0-99
 * 117     Standard write MR PtlTE / Control messaging
 * 128-227 Optimized read MR PtlTEs 0-99
 * 228     Standard read MR PtlTE
 * 255     Rendezvous source PtlTE
 *
 * Note: Any logical endpoint within a PID granule that issues unrestricted Puts
 * MUST be within the logical endpoint range 0 - 127 and unrestricted Gets MUST
 * be within the logical endpoint range 128 - 255.
 */
#define CXIP_PTL_IDX_RXQ		0
#define CXIP_PTL_IDX_WRITE_MR_OPT_BASE	17
#define CXIP_PTL_IDX_READ_MR_OPT_BASE	128
#define CXIP_PTL_IDX_MR_OPT_CNT		100

#define CXIP_PTL_IDX_WRITE_MR_OPT(key)	(CXIP_PTL_IDX_WRITE_MR_OPT_BASE + (key))
#define CXIP_PTL_IDX_READ_MR_OPT(key)	(CXIP_PTL_IDX_READ_MR_OPT_BASE + (key))
#define CXIP_PTL_IDX_WRITE_MR_STD	117
#define CXIP_PTL_IDX_COLL		6
#define CXIP_PTL_IDX_CTRL		CXIP_PTL_IDX_WRITE_MR_STD
#define CXIP_PTL_IDX_READ_MR_STD	228
#define CXIP_PTL_IDX_RDZV_SRC		255

static inline bool cxip_mr_key_opt(int key)
{
	return cxip_env.optimized_mrs && key < CXIP_PTL_IDX_MR_OPT_CNT;
}

static inline int cxip_mr_key_to_ptl_idx(int key, bool write)
{
	if (cxip_mr_key_opt(key))
		return write ? CXIP_PTL_IDX_WRITE_MR_OPT(key) :
			CXIP_PTL_IDX_READ_MR_OPT(key);
	return write ? CXIP_PTL_IDX_WRITE_MR_STD : CXIP_PTL_IDX_READ_MR_STD;
}

/* Messaging Match Bit layout */
#define CXIP_TAG_WIDTH		48
#define CXIP_RDZV_ID_WIDTH	8
#define CXIP_EAGER_RDZV_ID_WIDTH 7
#define CXIP_TX_ID_WIDTH	11
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
	CXIP_CTRL_LE_TYPE_CTRL_MSG,	/* Control Message LE */
};

enum cxip_ctrl_msg_type {
	CXIP_CTRL_MSG_FC_NOTIFY = 0,
	CXIP_CTRL_MSG_FC_RESUME,
	CXIP_CTRL_MSG_ZB_DATA,
};

union cxip_match_bits {
	struct {
		uint64_t tag        : CXIP_TAG_WIDTH; /* User tag value */
		uint64_t tx_id      : CXIP_TX_ID_WIDTH; /* Prov. tracked ID */
		uint64_t cq_data    : 1;  /* Header data is valid */
		uint64_t tagged     : 1;  /* Tagged API */
		uint64_t match_comp : 1;  /* Notify initiator on match */
		uint64_t le_type    : 2;
	};
	/* Split TX ID for rendezvous operations. */
	struct {
		uint64_t pad0       : CXIP_TAG_WIDTH; /* User tag value */
		uint64_t rdzv_id_hi : CXIP_EAGER_RDZV_ID_WIDTH;
		uint64_t rdzv_lac   : 4;  /* Rendezvous Get LAC */
	};
	struct {
		uint64_t rdzv_id_lo : CXIP_RDZV_ID_WIDTH;
	};
	/* Control LE match bit format for notify/resume */
	struct {
		uint64_t txc_id       : 8;
		uint64_t rxc_id       : 8;
		uint64_t drops        : 16;
		uint64_t pad1         : 29;
		uint64_t ctrl_msg_type: 2;
		uint64_t ctrl_le_type : 1;
	};
	/* Control LE match bit format for zbcollectives */
	struct {
		uint64_t zb_data       :61;
		uint64_t zb_pad        : 3;
		/* shares ctrl_le_type == CXIP_CTRL_LE_TYPE_CTRL_MSG
		 * shares ctrl_msg_type == CXIP_CTRL_MSG_ZB_BCAST
		 */
	};
	struct {
		uint64_t mr_key       : 63;
		uint64_t mr_pad       : 1;
		/* shares ctrl_le_type == CXIP_CTRL_LE_TYPE_MR */
	};
	uint64_t raw;
};

/* libcxi Wrapper Structures */

#define CXI_PLATFORM_ASIC 0
#define CXI_PLATFORM_NETSIM 1
#define CXI_PLATFORM_Z1 2
#define CXI_PLATFORM_FPGA 3

/*
 * CXI Device wrapper
 *
 * There will be one of these for every local Cassini device on the node.
 */
struct cxip_if {
	struct slist_entry if_entry;

	/* Device description */
	struct cxil_devinfo *info;
	int speed;
	int link;
	struct cxil_dev *dev;

	/* PtlTEs (searched during state change events) */
	struct dlist_entry ptes;

	ofi_atomic32_t ref;
	fastlock_t lock;
};

/*
 * CXI communication profile wrapper.
 *
 * The wrapper is used to remap user requested traffic class to a communication
 * profile which actually can be allocated.
 */
struct cxip_remap_cp {
	struct dlist_entry remap_entry;
	struct cxi_cp remap_cp;
	struct cxi_cp *hw_cp;
};

/*
 * CXI Logical Network Interface (LNI) wrapper
 *
 * An LNI is a container used allocate resources from a NIC.
 */
struct cxip_lni {
	struct cxip_if *iface;
	struct cxil_lni *lni;

	/* Hardware communication profiles */
	struct cxi_cp *hw_cps[16];
	int n_cps;

	/* Software remapped communication profiles. */
	struct dlist_entry remap_cps;

	fastlock_t lock;
};

/*
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

#define MAX_PTE_MAP_COUNT 2

/*
 * CXI Portal Table Entry (PtlTE) wrapper
 *
 * Represents PtlTE mapped in a CXI domain.
 */
struct cxip_pte {
	struct dlist_entry pte_entry;
	struct cxip_if_domain *if_dom;
	struct cxil_pte *pte;
	enum c_ptlte_state state;
	struct cxil_pte_map *pte_map[MAX_PTE_MAP_COUNT];
	unsigned int pte_map_count;

	void (*state_change_cb)(struct cxip_pte *pte,
				const union c_event *event);
	void *ctx;
};

/*
 * CXI Command Queue wrapper
 */
struct cxip_cmdq {
	struct cxi_cq *dev_cmdq;
	fastlock_t lock;
	struct c_cstate_cmd c_state;
	enum cxip_llring_mode llring_mode;

	struct cxi_cp *cur_cp;
	struct cxip_lni *lni;
};


/* OFI Provider Structures */

/*
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
	struct ofi_mr_info info;
	bool cached;
};

#define CXIP_MR_DOMAIN_HT_BUCKETS 16

struct cxip_mr_domain {
	struct dlist_entry buckets[CXIP_MR_DOMAIN_HT_BUCKETS];
	fastlock_t lock;
};

void cxip_mr_domain_init(struct cxip_mr_domain *mr_domain);
void cxip_mr_domain_fini(struct cxip_mr_domain *mr_domain);

/*
 * CXI Provider Domain object
 */
struct cxip_domain {
	struct util_domain util_domain;
	struct cxip_fabric *fab;
	fastlock_t lock;
	ofi_atomic32_t ref;

	struct cxi_auth_key auth_key;
	uint32_t tclass;

	struct cxip_eq *eq; //unused
	struct cxip_eq *mr_eq; //unused

	/* Assigned NIC address */
	uint32_t nic_addr;

	/* Device info */
	struct cxip_if *iface;

	/* Device partition */
	struct cxip_lni *lni;

	/* Trigger and CT support */
	struct cxip_cmdq *trig_cmdq;
	bool cntr_init;

	/* Translation cache */
	struct ofi_mr_cache iomm;
	bool odp;
	bool ats;
	bool hmem;

	/* ATS translation support */
	struct cxip_md scalable_md;
	bool scalable_iomm;
	bool rocr_dev_mem_only;

	/* Domain state */
	bool enabled;

	/* List of allocated resources used for deferred work queue processing.
	 */
	struct dlist_entry txc_list;
	struct dlist_entry cntr_list;
	struct dlist_entry cq_list;

	struct fi_hmem_override_ops hmem_ops;
	bool hybrid_mr_desc;

	/* Container of in-use MRs against this domain. */
	struct cxip_mr_domain mr_domain;
};

static inline bool cxip_domain_mr_cache_enabled(struct cxip_domain *dom)
{
	return dom->iomm.domain == &dom->util_domain;
}

static inline bool cxip_domain_mr_cache_iface_enabled(struct cxip_domain *dom,
						      enum fi_hmem_iface iface)
{
	return cxip_domain_mr_cache_enabled(dom) && dom->iomm.monitors[iface];
}

/* This structure implies knowledge about the breakdown of the NIC address,
 * which is taken from the AMA, that the provider does not know in a flexible
 * way. However, the domain fi_open_ops() API includes a topology function
 * that requires knowledge of the address breakdown into topology components.
 * TODO: Research a less restricted way to get this information.
 */
#define CXIP_ADDR_PORT_BITS 6
#define CXIP_ADDR_SWITCH_BITS 5
#define CXIP_ADDR_GROUP_BITS 9
#define CXIP_ADDR_FATTREE_PORT_BITS 6
#define CXIP_ADDR_FATTREE_SWITCH_BITS 14

struct cxip_topo_addr {
	union {
		uint32_t addr;
		struct {
			uint32_t port_num:CXIP_ADDR_PORT_BITS;
			uint32_t switch_num:CXIP_ADDR_SWITCH_BITS;
			uint32_t group_num:CXIP_ADDR_GROUP_BITS;
		} dragonfly;
		struct {
			uint32_t port_num:CXIP_ADDR_FATTREE_PORT_BITS;
			uint32_t switch_num:CXIP_ADDR_FATTREE_SWITCH_BITS;
		} fat_tree;
	};
};

static inline ssize_t
cxip_copy_to_hmem_iov(struct cxip_domain *domain, enum fi_hmem_iface hmem_iface,
		      uint64_t device, const struct iovec *hmem_iov,
		      size_t hmem_iov_count, uint64_t hmem_iov_offset,
		      const void *src, size_t size)
{
	return domain->hmem_ops.copy_to_hmem_iov(hmem_iface, device, hmem_iov,
						 hmem_iov_count,
						 hmem_iov_offset, src, size);
}

static inline ssize_t
cxip_domain_copy_to_hmem(struct cxip_domain *domain, uint64_t device,
			 void *hmem_dest, const void *src, size_t size,
			 enum fi_hmem_iface hmem_iface, bool hmem_iface_valid)
{
	struct iovec hmem_iov = {
		.iov_base = (void *)hmem_dest,
		.iov_len = size,
	};

	/* If device memory not supported or device supports access via
	 * load/store, just use memcpy to avoid expensive pointer query.
	 */
	if (!domain->hmem || (domain->rocr_dev_mem_only &&
	    size <= cxip_env.safe_devmem_copy_threshold)) {
		memcpy(hmem_dest, src, size);
		return size;
	}

	if (!hmem_iface_valid)
		hmem_iface = ofi_get_hmem_iface(hmem_dest);

	return cxip_copy_to_hmem_iov(domain, hmem_iface, device,
				     &hmem_iov, 1, 0, src, size);
}

static inline ssize_t
cxip_domain_copy_from_hmem(struct cxip_domain *domain, void *dest,
			   const void *hmem_src, size_t size,
			   enum fi_hmem_iface hmem_iface, bool hmem_iface_valid)
{
	struct iovec hmem_iov = {
		.iov_base = (void *)hmem_src,
		.iov_len = size,
	};

	/* If device memory not supported or device supports access via
	 * load/store, just use memcpy to avoid expensive pointer query.
	 */
	if (!domain->hmem || (domain->rocr_dev_mem_only &&
	    size <= cxip_env.safe_devmem_copy_threshold)) {
		memcpy(dest, hmem_src, size);
		return size;
	}

	if (!hmem_iface_valid)
		hmem_iface = ofi_get_hmem_iface(hmem_src);

	return domain->hmem_ops.copy_from_hmem_iov(dest, size, hmem_iface, 0,
						   &hmem_iov, 1, 0);
}

/*
 *  Event Queue
 *
 *  libfabric fi_eq implementation.
 *
 *  Created in cxip_eq_open().
 */
struct cxip_eq {
	struct util_eq util_eq;
	struct fi_eq_attr attr;
};

#define CXIP_EQ_MAP_FLAGS \
	(CXI_MAP_WRITE | CXI_MAP_PIN | CXI_MAP_IOVA_ALLOC)

/*
 * RMA request
 *
 * Support structures, accumulated in a union.
 */
struct cxip_req_rma {
	struct cxip_txc *txc;
	struct cxip_md *local_md;	// RMA target buffer
	void *ibuf;
};

struct cxip_req_amo {
	struct cxip_txc *txc;
	struct cxip_md *result_md;
	struct cxip_md *oper1_md;
	char result[16];
	char oper1[16];
	bool tmp_result;
	bool tmp_oper1;
	void *ibuf;
	bool fetching_amo_flush;
	uint8_t fetching_amo_flush_event_count;
	unsigned int fetching_amo_flush_event_rc;
	struct cxip_cntr *fetching_amo_flush_cntr;
};

struct cxip_req_recv {
	/* Receive parameters */
	struct dlist_entry rxc_entry;
	struct cxip_rxc *rxc;		// receive context
	struct cxip_cntr *cntr;
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
	uint8_t rdzv_lac;		// Rendezvous source LAC
	int rdzv_events;		// Processed rdzv event count
	uint32_t rdzv_initiator;	// Rendezvous initiator used of mrecvs
	uint32_t rget_nic;
	uint32_t rget_pid;
	bool software_list;		// Appended to HW or SW
	bool canceled;			// Request canceled?
	bool unlinked;
	bool multi_recv;
	bool tgt_event;
	uint64_t start_offset;
	uint64_t mrecv_bytes;
	struct cxip_req *parent;
	struct dlist_entry children;
	uint64_t src_offset;
	uint16_t rdzv_mlen;
};

struct cxip_req_send {
	/* Send parameters */
	struct cxip_txc *txc;
	struct cxip_cntr *cntr;
	const void *buf;		// local send buffer
	size_t len;			// request length
	struct cxip_md *send_md;	// send buffer memory descriptor
	struct cxip_addr caddr;
	uint8_t rxc_id;
	bool tagged;
	uint64_t tag;
	uint64_t data;
	uint64_t flags;
	void *ibuf;

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

struct cxip_req_search {
	struct cxip_rxc *rxc;
	bool complete;
	int puts_pending;
};

struct cxip_req_coll {
	struct cxip_coll_pte *coll_pte;
	struct cxip_coll_buf *coll_buf;
	uint32_t mrecv_space;
	size_t hw_req_len;
	bool isred;
	enum c_return_code cxi_rc;
};

enum cxip_req_type {
	CXIP_REQ_RMA,
	CXIP_REQ_AMO,
	CXIP_REQ_OFLOW,
	CXIP_REQ_RECV,
	CXIP_REQ_SEND,
	CXIP_REQ_RDZV_SRC,
	CXIP_REQ_SEARCH,
	CXIP_REQ_COLL,
	CXIP_REQ_RBUF,
};

/*
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
 * The request is allocated and bound to the CQ, and then the command is
 * issued. When the completion queue signals completion, this request is found,
 * and the callback function is called.
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

	/* Triggered related fields. */
	bool triggered;
	uint64_t trig_thresh;
	struct cxip_cntr *trig_cntr;

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
	enum cxip_req_type type;
	union {
		struct cxip_req_rma rma;
		struct cxip_req_amo amo;
		struct cxip_req_oflow oflow;
		struct cxip_req_recv recv;
		struct cxip_req_send send;
		struct cxip_req_rdzv_src rdzv_src;
		struct cxip_req_search search;
		struct cxip_req_coll coll;
	};
};

static inline bool cxip_is_trig_req(struct cxip_req *req)
{
	return req->trig_cntr != NULL;
}

struct cxip_ctrl_req_mr {
	struct cxip_mr *mr;
};

struct cxip_ctrl_send {
	uint32_t nic_addr;
	uint32_t pid;
	union cxip_match_bits mb;
};

struct cxip_ctrl_req {
	struct dlist_entry ep_entry;
	struct cxip_ep_obj *ep_obj;
	int req_id;
	int (*cb)(struct cxip_ctrl_req *req, const union c_event *evt);

	union {
		struct cxip_ctrl_req_mr mr;
		struct cxip_ctrl_send send;
	};
};

struct cxip_fc_peer {
	struct dlist_entry txc_entry;
	struct cxip_txc *txc;
	struct cxip_ctrl_req req;
	struct cxip_addr caddr;
	uint8_t rxc_id;
	struct dlist_entry msg_queue;
	uint16_t pending;
	uint16_t dropped;
	uint16_t pending_acks;
	bool replayed;
	unsigned int retry_count;
};

struct cxip_fc_drops {
	struct dlist_entry rxc_entry;
	struct cxip_rxc *rxc;
	struct cxip_ctrl_req req;
	uint32_t nic_addr;
	uint32_t pid;
	uint8_t txc_id;
	uint8_t rxc_id;
	uint16_t drops;
	unsigned int retry_count;
};

/* Completion queue specific wrapper around CXI event queue. */
struct cxip_cq_eq {
	struct cxi_eq *eq;
	void *buf;
	size_t len;
	struct cxi_md *md;
	bool mmap;
	unsigned int unacked_events;
	struct c_eq_status prev_eq_status;
	bool eq_saturated;
};

/*
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

	/* Wrapper for hardware EQ. */
	struct cxip_cq_eq eq;

	/* CXI specific fields. */
	struct cxip_domain *domain;
	struct cxip_ep_obj *ep_obj;
	fastlock_t lock;
	bool enabled;
	unsigned int ack_batch_size;
	fastlock_t req_lock;
	struct ofi_bufpool *req_pool;
	struct indexer req_table;
	struct dlist_entry req_list;

	struct ofi_bufpool *ibuf_pool;
	fastlock_t ibuf_lock;

	struct dlist_entry dom_entry;
};

static inline uint16_t cxip_cq_tx_eqn(struct cxip_cq *cq)
{
	return cq->eq.eq->eqn;
}

/*
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

	struct fid_wait *wait;

	fastlock_t lock;

	struct cxi_ct *ct;
	struct c_ct_writeback *wb;
	enum fi_hmem_iface wb_iface;
	struct c_ct_writeback lwb;

	struct dlist_entry dom_entry;
};

struct cxip_ux_send {
	struct dlist_entry rxc_entry;
	struct cxip_req *req;
	union c_event put_ev;
};

/* Key used to associate PUT and PUT_OVERFLOW events */
union cxip_def_event_key {
	struct {
		uint64_t initiator	: 32;
		uint64_t rdzv_id	: 8;
		uint64_t pad0		: 23;
		uint64_t rdzv		: 1;
	};
	struct {
		uint64_t start_addr	: 57;
		uint64_t pad1		: 7;
	};
	uint64_t raw;
};

struct cxip_deferred_event {
	struct dlist_entry rxc_entry;
	union cxip_def_event_key key;
	struct cxip_req *req;
	union c_event ev;
	uint64_t mrecv_start;
	uint32_t mrecv_len;

	struct cxip_ux_send *ux_send;
};

/* A very specific (non-generic) hash table is used to map
 * deferred CXI events to associate PUT and PUT_OVERFLOW events.
 * Hash entries are added and removed at a high rate and the
 * overhead of generic implementations is insufficient.
 */
#define CXIP_DEF_EVENT_HT_BUCKETS	256

struct def_event_ht {
	struct dlist_entry bh[CXIP_DEF_EVENT_HT_BUCKETS];
};

/*
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
	size_t sw_consumed;
	size_t hw_consumed;
	int buffer_id;
};

/*
 * Zero-buffer collectives context.
 */
struct cxip_zbcoll_state {
	uint64_t *dataptr;
	uint64_t dataval;
	uint32_t nid;
	uint32_t pid;
	uint32_t num_relatives;
	uint32_t *relatives;
	uint32_t contribs;
	int error;
	bool running;
	bool complete;
	ofi_atomic32_t err_count;
	ofi_atomic32_t ack_count;
	ofi_atomic32_t rcv_count;
};

struct cxip_zbcoll_obj {
	bool disable;
	uint32_t count;
	struct cxip_zbcoll_state *state;
};

/*
 * Collectives context.
 *
 * Support structure.
 *
 * Initialized in cxip_coll_init() during EP creation.
 */
struct cxip_ep_coll_obj {
	struct cxip_cmdq *rx_cmdq;	// shared with STD EP
	struct cxip_cmdq *tx_cmdq;	// shared with STD EP
	struct cxip_cntr *rx_cntr;	// shared with STD EP
	struct cxip_cntr *tx_cntr;	// shared with STD EP
	struct cxip_cq *rx_cq;		// shared with STD EP
	struct cxip_cq *tx_cq;		// shared with STD EP
	ofi_atomic32_t mc_count;	// count of MC objects
	fastlock_t lock;		// collectives lock
	size_t min_multi_recv;		// trigger value to rotate bufs
	size_t buffer_size;		// size of receive buffers
	size_t buffer_count;		// count of receive buffers
	bool enabled;			// enabled
};

/* Receive context state machine.
 * TODO: Handle unexpected RMA.
 */
enum cxip_rxc_state {
	/* Initial state of an RXC. All user posted receives are rejected until
	 * the RXC has been enabled.
	 *
	 * Note that an RXC can be transitioned from any state into
	 * RXC_DISABLED.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: User has successfully enabled the RXC.
	 * RXC_ENABLED_SOFTWARE: User has successfully enabled the RXC
	 * in a software only RX matching mode.
	 */
	RXC_DISABLED = 0,

	/* User posted receives are matched against the software unexpected list
	 * before being offloaded to hardware. Hardware matches against the
	 * corresponding PtlTE priority and overflow list.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL: Two scenarios can cause this state change.
	 *    1. Hardware fails to allocate an LE for an unexpected message,
	 *    or a priority list LE append fails. Hardware automatically
	 *    transitions the PtlTE from Enabled to Disabled.
	 *    2. Hardware fails to allocate an LE during an overflow list
	 *    append. The PtlTE remains in the Enabled state but appends to
	 *    the overflow list are disabled. Software manually disables
	 *    the PtlTE.
	 *    3. Hardware fails to successfully match on the overflow list or
	 *    the overflow buffer is full. Hardware automatically transitions
	 *    the PtlTE from Enabled to Disabled.
	 * RXC_PENDING_PTLTE_SOFTWARE_MANAGED: If the provider is configured
	 * to run in "hybrid" RX match mode and hardware fails to allocate an
	 * LE for an overflow list or priority list append. Hardware will
	 * automatically transition the PtlTE from Enabled to Software Managed.
	 */
	RXC_ENABLED,

	/* Hardware has initiated an automated transition from hardware
	 * to software managed PtlTE matching due to resource load.
	 *
	 * Software is onloading the hardware unexpected list while creating
	 * a pending unexpected list from entries received on the PtlTE
	 * request list. Any in flight appends will fail and be added to
	 * a receive replay list, further attempts to post receive operations
	 * will return -FI_EAGAIN. When onloading completes, the pending
	 * UX list is appended to the onloaded UX list and then failed appends
	 * are replayed prior to enabling the posting of receive operations.
	 *
	 * Validate state changes:
	 * RXC_ENABLED_SOFTWARE: The HW to SW transition completes and software
	 * managed endpoint PtlTE matching is operating.
	 */
	RXC_PENDING_PTLTE_SOFTWARE_MANAGED,

	/* Executing as a software managed PtlTE either due to hybrid
	 * transition from hardware or initial startup in software
	 * RX matching mode.
	 *
	 * Validate state changes:
	 * RXC_PENDING_PTLTE_HARDWARE: TODO: When able, software may
	 * initiate a transition from software managed mode back to
	 * fully offloaded operation.
	 * RXC_FLOW_CONTROL: Hardware was unable to match on the request
	 * list and Disabled the PtlTE initiating flow control.
	 */
	RXC_ENABLED_SOFTWARE,

	/* TODO: Hybrid RX match mode PtlTE is transitioning from software
	 * managed operation back to fully offloaded operation.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: Hybrid software managed PtlTE successfully
	 * transitions back to fully offloaded operation.
	 * RXC_ENABLED_SOFTWARE: Hybrid software managed PtlTE was
	 * not able to transition to fully offloaded operation.
	 */
	RXC_PENDING_PTLTE_HARDWARE,

	/* Software has encountered a condition which requires manual transition
	 * of the PtlTE into disable. This state change occurs when a posted
	 * receive could not be appended due to LE exhaustion and software
	 * managed EP PtlTE operation has been disabled or is not possible.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL: PtlTE disabled event has successfully been
	 * received and onloading can begin.
	 */
	RXC_PENDING_PTLTE_DISABLE,

	/* Flow control has occurred and the PtlTE is disabled. Software is
	 * in the process of onloading the hardware unexpected headers to free
	 * up LEs. User posted receives are matched against the software
	 * unexpected list. If a match is not found on the software unexpected
	 * list, -FI_EAGAIN is returned to the user. Hardware matching is
	 * disabled.
	 *
	 * Validate state changes:
	 * RXC_ONLOAD_FLOW_CONTROL_REENABLE: An unexpected list entry matched a
	 * user posted receive or the search and delete command free a
	 * unexpected list entry.
	 */
	RXC_ONLOAD_FLOW_CONTROL,

	/* PtlTE is in the same state as RXC_ONLOAD_FLOW_CONTROL, but the RXC
	 * should attempt to be re-enabled.
	 *
	 * Validate state changes:
	 * RXC_FLOW_CONTROL: Onloading of the unexpected headers has completed.
	 */
	RXC_ONLOAD_FLOW_CONTROL_REENABLE,

	/* Software is performing sideband communication to recover the dropped
	 * messages. User posted receives are matched against the software
	 * unexpected list. If a match is not found on the software unexpected
	 * list, -FI_EAGAIN is returned to the user. Hardware matching is
	 * disabled.
	 *
	 * If an append fails due to RC_NO_SPACE while in the RXC_FLOW_CONTROL
	 * state, hardware LEs are exhausted and no more LEs can be freed by
	 * onloading unexpected headers into software. This is a fatal event
	 * which requires software endpoint mode to workaround.
	 *
	 * Validate state changes:
	 * RXC_ENABLED: Sideband communication is complete and PtlTE is
	 * successfully re-enabled.
	 * RXC_ENABLED_SOFTWARE: Executing in hybrid or software mode,
	 * and sideband communication is completed; the PtlTE was
	 * successfully transitioned to software managed operation.
	 */
	RXC_FLOW_CONTROL,
};

/*
 * Receive Context
 *
 * Created in cxip_rxc(), during EP creation.
 */
struct cxip_rxc {
	struct fid_ep ctx;
	fastlock_t lock;		// Control ops lock

	uint16_t rx_id;			// SEP index

	int use_shared;
	struct cxip_rxc *srx;

	struct cxip_cq *recv_cq;
	struct cxip_cntr *recv_cntr;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain
	uint8_t pid_bits;

	struct dlist_entry ep_list;	// contains EPs using shared context

	struct fi_rx_attr attr;
	bool selective_completion;

	struct cxip_pte *rx_pte;	// HW RX Queue
	struct cxip_cmdq *rx_cmdq;	// RX CMDQ for posting receive buffers
	struct cxip_cmdq *tx_cmdq;	// TX CMDQ for Message Gets

	ofi_atomic32_t orx_reqs;	// outstanding receive requests
	unsigned int recv_appends;

	int min_multi_recv;
	int rdzv_threshold;
	int rdzv_get_min;

	/* Unexpected message handling */
	fastlock_t rx_lock;			// RX message lock
	ofi_atomic32_t oflow_bufs_submitted;
	ofi_atomic32_t oflow_bufs_linked;
	ofi_atomic32_t oflow_bufs_in_use;
	int oflow_buf_size;
	int oflow_bufs_max;
	struct dlist_entry oflow_bufs;		// Overflow buffers

	/* Defer events to wait for both put and put overflow */
	struct def_event_ht deferred_events;

	/* Order list of request buffers emitted to hardware. */
	struct dlist_entry active_req_bufs;

	/* List of consumed buffers which cannot be reposted yet since
	 * unexpected entries have not been matched yet.
	 */
	struct dlist_entry consumed_req_bufs;

	/*
	 * List of available buffers for which an append failed or have
	 * have been removed but not yet released.
	 */
	struct dlist_entry free_req_bufs;

	ofi_atomic32_t req_bufs_linked;
	ofi_atomic32_t req_bufs_allocated;
	size_t req_buf_size;
	size_t req_buf_max_count;
	size_t req_buf_min_posted;

	/* Long eager send handling */
	ofi_atomic32_t sink_le_linked;
	struct cxip_oflow_buf sink_le;		// Long UX sink buffer

	struct dlist_entry fc_drops;
	struct dlist_entry replay_queue;
	struct dlist_entry sw_ux_list;
	struct dlist_entry sw_pending_ux_list;
	int sw_ux_list_len;
	int sw_pending_ux_list_len;

	/* Array of 8-byte of unexpected headers remote offsets. */
	uint64_t *ule_offsets;

	/* Current remote offset to be processed. Incremented after processing
	 * a search and delete put event.
	 */
	unsigned int cur_ule_offsets;

	/* Software receive queue. User posted requests are queued here instead
	 * of on hardware if the RXC is in software endpoint mode.
	 */
	struct dlist_entry sw_recv_queue;

	enum cxip_rxc_state state;
	enum cxip_rxc_state prev_state;
	bool msg_offload;

	/* RXC drop count used for FC accounting. */
	int drop_count;
	bool hmem;
};

static inline ssize_t
cxip_rxc_copy_to_hmem(struct cxip_rxc *rxc, uint64_t device,
		      void *hmem_dest, const void *src, size_t size,
		      enum fi_hmem_iface hmem_iface)
{
	if (!rxc->hmem) {
		memcpy(hmem_dest, src, size);
		return size;
	}

	return cxip_domain_copy_to_hmem(rxc->domain, device, hmem_dest, src,
					size, hmem_iface, true);
}

/* Request buffer structure. */
struct cxip_req_buf {
	/* RX context the request buffer is posted on. */
	struct cxip_rxc *rxc;
	struct dlist_entry req_buf_entry;
	struct cxip_req *req;

	/* Memory mapping of req_buf field. */
	struct cxip_md *md;

	/* The number of bytes consume by hardware when the request buffer was
	 * unlinked.
	 */
	size_t unlink_length;

	/* Current offset into the req_buf where packets are landing. When
	 * cur_offset is equal to unlink_length, software has received all
	 * hardware put events for this request buffer.
	 */
	size_t cur_offset;

	/* Pending list of unexpected header entries which could not be placed
	 * on the RX context unexpected header list due to put events being
	 * receive out-of-order.
	 */
	struct dlist_entry pending_ux_list;

	/* The number of unexpected headers posted placed on the RX context
	 * unexpected header list which have not been matched.
	 */
	ofi_atomic32_t refcount;

	/* Buffer used to land packets matching on the request list. This field
	 * must remain at the bottom of this structure.
	 */
	char req_buf[0];
};

void cxip_req_buf_ux_free(struct cxip_ux_send *ux);
int cxip_req_buf_unlink(struct cxip_req_buf *buf);
int cxip_req_buf_link(struct cxip_req_buf *buf, bool seq_restart);
struct cxip_req_buf *cxip_req_buf_alloc(struct cxip_rxc *rxc);
void cxip_req_buf_free(struct cxip_req_buf *buf);
int cxip_req_buf_replenish(struct cxip_rxc *rxc, bool seq_restart);

#define CXIP_RDZV_IDS	(1 << CXIP_RDZV_ID_WIDTH)
#define CXIP_EAGER_RDZV_IDS (1 << CXIP_EAGER_RDZV_ID_WIDTH)
#define CXIP_TX_IDS	(1 << CXIP_TX_ID_WIDTH)

/*
 * Transmit Context
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

	bool hmem;

	int use_shared;
	struct cxip_txc *stx;

	struct cxip_cq *send_cq;
	struct cxip_cntr *send_cntr;
	struct cxip_cntr *read_cntr;
	struct cxip_cntr *write_cntr;

	struct cxip_ep_obj *ep_obj;	// parent EP object
	struct cxip_domain *domain;	// parent domain
	uint8_t pid_bits;

	struct dlist_entry ep_list;	// contains EPs using shared context
	fastlock_t lock;

	struct fi_tx_attr attr;		// attributes
	bool selective_completion;
	uint32_t tclass;

	struct cxip_cmdq *tx_cmdq;	// added during cxip_txc_enable()

	ofi_atomic32_t otx_reqs;	// outstanding transmit requests
	struct cxip_req *rma_inject_req;
	struct cxip_req *amo_inject_req;

	/* Software Rendezvous related structures */
	struct cxip_pte *rdzv_pte;	// PTE for SW Rendezvous commands
	int rdzv_threshold;
	int rdzv_get_min;
	int rdzv_eager_size;
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

	struct dlist_entry dom_entry;
};

static inline ssize_t
cxip_txc_copy_from_hmem(struct cxip_txc *txc, void *dest, const void *hmem_src,
			size_t size)
{
	if (!txc->hmem) {
		memcpy(dest, hmem_src, size);
		return size;
	}

	return cxip_domain_copy_from_hmem(txc->domain, dest, hmem_src, size,
					  FI_HMEM_SYSTEM, !txc->hmem);
}

void cxip_txc_flush_msg_trig_reqs(struct cxip_txc *txc);

/*
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
	int pids;
	struct cxip_cmdq *txqs[CXIP_EP_MAX_TX_CNT];
	ofi_atomic32_t txq_refs[CXIP_EP_MAX_TX_CNT];
	struct cxip_cmdq *tgqs[CXIP_EP_MAX_TX_CNT];
	ofi_atomic32_t tgq_refs[CXIP_EP_MAX_TX_CNT];
	fastlock_t cmdq_lock;

	uint64_t caps;
	struct fi_ep_attr ep_attr;
	size_t txq_size;
	size_t tgq_size;

	struct cxi_auth_key auth_key;

	bool enabled;
	fastlock_t lock;

	struct cxip_addr src_addr;	// address of this NIC
	fi_addr_t fi_addr;		// AV address of this EP
	int rdzv_offload;

	struct cxip_if_domain *if_dom[CXIP_EP_MAX_TX_CNT];

	/* collectives support */
	struct cxip_ep_coll_obj coll;
	struct cxip_zbcoll_obj zbcoll;

	struct indexer rdzv_ids;
	int max_rdzv_ids;
	fastlock_t rdzv_id_lock;

	struct indexer tx_ids;
	fastlock_t tx_id_lock;

	/* Control resources */
	struct cxip_cmdq *ctrl_tgq;
	struct cxip_cmdq *ctrl_txq;
	unsigned int ctrl_tx_credits;
	struct cxi_eq *ctrl_tgt_evtq;
	struct cxi_eq *ctrl_tx_evtq;
	void *ctrl_tgt_evtq_buf;
	struct cxi_md *ctrl_tgt_evtq_buf_md;
	void *ctrl_tx_evtq_buf;
	struct cxi_md *ctrl_tx_evtq_buf_md;
	struct cxip_pte *ctrl_pte;
	struct indexer req_ids;
	struct dlist_entry mr_list;
	struct cxip_ctrl_req ctrl_msg_req;
};

/*
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

/*
 * Memory Region
 *
 * libfabric fi_mr implementation.
 *
 * Created in cxip_regattr().
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

	struct dlist_entry mr_domain_entry;
};

/*
 * Address Vector header
 *
 * Support structure.
 */
struct cxip_av_table_hdr {
	uint64_t size;
	uint64_t stored;
};

/*
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

/*
 * AV Set communication key.
 *
 * For production, use COMM_KEY_MULTICAST type.
 * - The av_set should be joined once on each node.
 * - dest_addr is the 13-bit multicast ID value supplied by the Rosetta fabic
 *   service.
 * - hwroot_nic is the NIC address of the node that serves as the root of the
 *   multicast tree.
 * - The PTE will receive at the multicast ID value, index extension of zero.
 * - Sending to the multicast ID will cause delivery to nodes according to the
 *   tree topology.
 *
 * For testing on a multinode system without multicast, use COMM_KEY_UNICAST.
 * - The av_set should be joined once on each node.
 * - dest_addr is ignored.
 * - hwroot_nic is the NIC address of the node that serves as the simulated
 *   hardware root of the tree.
 * - The PTE will use the EP source NIC address and process PID, with a
 *   PID_IDX of CXIP_PTL_IDX_COLL.
 * - Sending to any (valid) node address with CXIP_PTL_IDX_COLL will target the
 *   collectives PTE on that node.
 * - The root/leaf send routines will distribute one or more packets to all
 *   fi_addr_t in the av_set as appropriate.
 *
 * For testing on a single node (typically under NETSIM), use COMM_KEY_RANK.
 * - The node must join N identical av_set objects, each representing a
 *   simulated "node" target in the av_set, to result in N separate MC objects,
 *   which are numbered (0, N-1).
 * - dest_addr is the MC object index.
 * - hwroot_nic is the MC object index for the MC object to serve as the
 *   simulated hardware root.
 * - The PTE will use the EP source NIC address and process PID, with a PID_IDX
 *   of 16 + dest_addr (MC object index).
 * - Sending to the node's own address with a PID_IDX of 16 + MC index will
 *   target the appropriate MC object.
 */
enum cxip_comm_key_type {
	COMM_KEY_NONE = 0,
	COMM_KEY_MULTICAST,
	COMM_KEY_UNICAST,
	COMM_KEY_RANK,
};

struct cxip_coll_mcast_key {
	uint32_t hwroot_idx;	// index of hwroot in av_set list
	uint32_t mcast_id;	// 13-bit multicast address id
	uint32_t mcast_ref;	// unique multicast reference number
};

struct cxip_coll_unicast_key {
	uint32_t hwroot_idx;	// index of hwroot in av_set list
};

struct cxip_coll_rank_key {
	uint32_t hwroot_idx;	// index of hwroot in av_set list
	uint32_t rank;
};

struct cxip_comm_key {
	enum cxip_comm_key_type type;
	union {
		struct cxip_coll_mcast_key mcast;
		struct cxip_coll_unicast_key ucast;
		struct cxip_coll_rank_key rank;
	};
};

/*
 * AV Set
 *
 * libfabric fi_av_set implementation.
 *
 * Created in cxip_av_set().
 */
struct cxip_av_set {
	struct fid_av_set av_set_fid;
	struct cxip_av *cxi_av;		// associated AV
	struct cxip_coll_mc *mc_obj;	// reference MC
	fi_addr_t *fi_addr_ary;		// addresses in set
	size_t fi_addr_cnt;		// count of addresses
	struct cxip_comm_key comm_key;	// communication key
	uint64_t flags;
	ofi_atomic32_t ref;
};

/* Collective operation states
 */
enum cxip_coll_state {
	CXIP_COLL_STATE_NONE,
	CXIP_COLL_STATE_READY,
	CXIP_COLL_STATE_FAULT,
};

/* Rosetta reduction engine error codes.
 */
enum cxip_coll_rc {
	CXIP_COLL_RC_SUCCESS = 0,		// good
	CXIP_COLL_RC_FLT_INEXACT = 1,		// result was rounded
	CXIP_COLL_RC_FLT_OVERFLOW = 3,		// result too large to represent
	CXIP_COLL_RC_FLT_INVALID = 4,           // operand was signalling NaN,
						//   or infinities subtracted
	CXIP_COLL_RC_REPSUM_INEXACT = 5,	// reproducible sum was rounded
	CXIP_COLL_RC_INT_OVERFLOW = 6,		// integer overflow
	CXIP_COLL_RC_CONTR_OVERFLOW = 7,	// too many contributions seen
	CXIP_COLL_RC_OP_MISMATCH = 8,		// conflicting opcodes
	CXIP_COLL_RC_MAX = 9
};

struct cxip_coll_buf {
	struct dlist_entry buf_entry;		// linked list of buffers
	struct cxip_req *req;			// associated LINK request
	struct cxip_md *cxi_md;			// buffer memory descriptor
	size_t bufsiz;				// buffer size in bytes
	uint8_t buffer[];			// buffer space itself
};

struct cxip_coll_pte {
	struct cxip_pte *pte;			// Collectives PTE
	struct cxip_ep_obj *ep_obj;		// Associated endpoint
	struct cxip_coll_mc *mc_obj;		// Associated multicast object
	struct dlist_entry buf_list;		// PTE receive buffers
	ofi_atomic32_t buf_cnt;			// count of linked buffers
	ofi_atomic32_t buf_swap_cnt;		// for diagnostics
	bool enabled;				// enabled
};

struct cxip_coll_reduction {
	struct cxip_coll_mc *mc_obj;		// parent mc_obj
	uint32_t red_id;			// reduction id
	uint16_t seqno;				// reduction sequence number
	uint16_t resno;				// reduction result number
	struct cxip_req *op_inject_req;		// active operation request
	enum cxip_coll_state coll_state;	// reduction state on node
	int op_code;				// requested CXI operation
	const void *op_send_data;		// user send buffer (or NULL)
	void *op_rslt_data;			// user recv buffer (or NULL)
	int op_data_len;			// bytes in send/recv buffers
	void *op_context;			// caller's context
	bool in_use;				// reduction is in-use
	bool completed;				// reduction is completed
	uint8_t red_data[CXIP_COLL_MAX_TX_SIZE];
	bool red_init;				// set by first rcvd pkt
	int red_op;				// set by first rcvd pkt
	int red_cnt;				// incremented by packet
	enum cxip_coll_rc red_rc;		// set by first error
	struct timespec armtime;		// timestamp at last arm
	uint8_t tx_msg[64];			// static packet memory
};

struct cxip_coll_mc {
	struct fid_mc mc_fid;
	struct cxip_ep_obj *ep_obj;		// Associated endpoint
	struct cxip_av_set *av_set;		// associated AV set
	struct cxip_coll_pte *coll_pte;		// collective PTE
	bool is_joined;				// true if joined
	unsigned int mynode_index;		// av_set index of this node
	unsigned int hwroot_index;		// av_set index of hwroot node
	uint32_t mc_unique;			// MC object id for cookie
	int tail_red_id;			// tail active red_id
	int next_red_id;			// next available red_id
	int max_red_id;				// limit total concurrency
	struct timespec timeout;		// state machine timeout
	int seqno;				// rolling seqno for packets
	bool arm_enable;			// arm-enable for root
	enum cxi_traffic_class tc;		// traffic class
	enum cxi_traffic_class_type tc_type;	// traffic class type
	ofi_atomic32_t send_cnt;		// for diagnostics
	ofi_atomic32_t recv_cnt;		// for diagnostics
	ofi_atomic32_t pkt_cnt;			// for diagnostics
	ofi_atomic32_t seq_err_cnt;		// for diagnostics
	fastlock_t lock;

	struct cxi_md *reduction_md;		// memory descriptor for DMA
	struct cxip_coll_reduction reduction[CXIP_COLL_MAX_CONCUR];
};

/**
 * Packed data structure used for all reductions.
 */
union cxip_coll_data {
	uint8_t databuf[CXIP_COLL_MAX_TX_SIZE];
	uint64_t ival[CXIP_COLL_MAX_TX_SIZE/(sizeof(uint64_t))];
	double fval[CXIP_COLL_MAX_TX_SIZE/(sizeof(double))];
	struct {
		uint64_t iminval;
		uint64_t iminidx;
		uint64_t imaxval;
		uint64_t imaxidx;

	} iminmax;
	struct {
		double fminval;
		uint64_t fminidx;
		double fmaxval;
		uint64_t fmaxidx;
	} fminmax;
} __attribute__((packed));

/* Our asynchronous handle for requests */
struct cxip_curl_handle;

typedef void (*curlcomplete_t)(struct cxip_curl_handle *);

struct cxip_curl_handle {
	long status;		// HTTP status, 0 for no server, -1 busy
	const char *endpoint;	// HTTP server endpoint address
	const char *request;	// HTTP request data
	const char *response;	// HTTP response data, NULL until complete
	curlcomplete_t usrfunc;	// user completion function
	const void *usrptr;	// user data pointer
	void *recv;		// opaque
	void *headers;		// opaque
};

/* Low-level CURL POST/DELETE async wrappers */
enum curl_ops {
	CURL_GET,
	CURL_PUT,
	CURL_POST,
	CURL_PATCH,
	CURL_DELETE,
	CURL_MAX
};
int cxip_curl_init(void);
void cxip_curl_fini(void);
const char *cxip_curl_opname(enum curl_ops op);
int cxip_curl_perform(const char *server, const char *request,
		      size_t rsp_init_size, enum curl_ops op, bool verbose,
		      curlcomplete_t usrfunc, void *usrptr);
int cxip_curl_progress(struct cxip_curl_handle **handleptr);
void cxip_curl_free(struct cxip_curl_handle *handle);

static inline void single_to_double_quote(char *str)
{
	do {if (*str == '\'') *str = '"';} while (*(++str));
}
enum json_type cxip_json_obj(const char *desc, struct json_object *jobj,
			     struct json_object **jval);
int cxip_json_bool(const char *desc, struct json_object *jobj, bool *val);
int cxip_json_int(const char *desc, struct json_object *jobj, int *val);
int cxip_json_int64(const char *desc, struct json_object *jobj, int64_t *val);
int cxip_json_double(const char *desc, struct json_object *jobj, double *val);
int cxip_json_string(const char *desc, struct json_object *jobj,
		     const char **val);

/* Acquire or delete a multicast address */
int cxip_request_mcast(const char *endpoint, const char *cfg_path,
		       unsigned int mcast_id, unsigned int root_port_idx,
		       bool verbose, curlcomplete_t usrfunc, void *usrptr);
int cxip_delete_mcast(const char *server, long reqid, bool verbose,
		      curlcomplete_t usrfunc, void *usrptr);
int cxip_progress_mcast(int *reqid, int *mcast_id, int *root_idx);

/* Perform zero-buffer collectives */
void cxip_tree_rowcol(int radix, int nodeidx, int *row, int *col, int *siz);
void cxip_tree_nodeidx(int radix, int row, int col, int *nodeidx);
int cxip_tree_relatives(int radix, int nodeidx, int maxnodes, int *rels);

int cxip_zbcoll_send(struct cxip_ep_obj *ep_obj, uint32_t srcnid,
		     uint32_t dstnid, uint64_t mb);
void cxip_zbcoll_reset_counters(struct fid_ep *ep);
int cxip_zbcoll_config(struct fid_ep *ep, int num_nids, uint32_t *nids,
			bool sim);
int cxip_zbcoll_recv(struct cxip_ep_obj *ep_obj, uint32_t init_nic,
		      uint32_t init_pid, uint64_t mbv);
int cxip_zbcoll_bcast(struct fid_ep *ep, uint64_t *dataptr);
int cxip_zbcoll_barrier(struct fid_ep *ep);
int cxip_zbcoll_progress(struct fid_ep *ep);

void cxip_zbcoll_fini(struct cxip_ep_obj *ep_obj);
int cxip_zbcoll_init(struct cxip_ep_obj *ep_obj);

/*
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

struct cxip_if *cxip_if_lookup_addr(uint32_t nic_addr);
struct cxip_if *cxip_if_lookup_name(const char *name);
int cxip_get_if(uint32_t nic_addr, struct cxip_if **dev_if);
void cxip_put_if(struct cxip_if *dev_if);
int cxip_alloc_lni(struct cxip_if *iface, uint32_t svc_id,
		   struct cxip_lni **if_lni);
void cxip_free_lni(struct cxip_lni *lni);
const char *cxi_tc_str(enum cxi_traffic_class tc);
enum cxi_traffic_class cxip_ofi_to_cxi_tc(uint32_t ofi_tclass);
int cxip_txq_cp_set(struct cxip_cmdq *cmdq, uint16_t vni,
		    enum cxi_traffic_class tc,
		    enum cxi_traffic_class_type tc_type);
int cxip_alloc_if_domain(struct cxip_lni *lni, uint32_t vni, uint32_t pid,
			 struct cxip_if_domain **if_dom);
void cxip_free_if_domain(struct cxip_if_domain *if_dom);
void cxip_if_init(void);
void cxip_if_fini(void);

int cxip_pte_set_state(struct cxip_pte *pte, struct cxip_cmdq *cmdq,
		       enum c_ptlte_state new_state, uint32_t drop_count);
int cxip_pte_set_state_wait(struct cxip_pte *pte, struct cxip_cmdq *cmdq,
			    struct cxip_cq *cq, enum c_ptlte_state new_state,
			    uint32_t drop_count);
int cxip_pte_append(struct cxip_pte *pte, uint64_t iova, size_t len,
		    unsigned int lac, enum c_ptl_list list,
		    uint32_t buffer_id, uint64_t match_bits,
		    uint64_t ignore_bits, uint32_t match_id,
		    uint64_t min_free, uint32_t flags,
		    struct cxip_cntr *cntr, struct cxip_cmdq *cmdq,
		    bool ring);
int cxip_pte_unlink(struct cxip_pte *pte, enum c_ptl_list list,
		    int buffer_id, struct cxip_cmdq *cmdq);
int cxip_pte_map(struct cxip_pte *pte, uint64_t pid_idx, bool is_multicast);
int cxip_pte_alloc_nomap(struct cxip_if_domain *if_dom, struct cxi_eq *evtq,
			 struct cxi_pt_alloc_opts *opts,
			 void (*state_change_cb)(struct cxip_pte *pte,
						 const union c_event *event),
			 void *ctx, struct cxip_pte **pte);
int cxip_pte_alloc(struct cxip_if_domain *if_dom, struct cxi_eq *evtq,
		   uint64_t pid_idx, bool is_multicast,
		   struct cxi_pt_alloc_opts *opts,
		   void (*state_change_cb)(struct cxip_pte *pte,
					   const union c_event *event),
		   void *ctx, struct cxip_pte **pte);
void cxip_pte_free(struct cxip_pte *pte);
int cxip_pte_state_change(struct cxip_if *dev_if, const union c_event *event);

int cxip_cmdq_alloc(struct cxip_lni *lni, struct cxi_eq *evtq,
		    struct cxi_cq_alloc_opts *cq_opts, uint16_t vni,
		    enum cxi_traffic_class tc,
		    enum cxi_traffic_class_type tc_type,
		    struct cxip_cmdq **cmdq);
void cxip_cmdq_free(struct cxip_cmdq *cmdq);
int cxip_cmdq_emit_c_state(struct cxip_cmdq *cmdq,
			   const struct c_cstate_cmd *cmd);

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

int cxip_tx_id_alloc(struct cxip_ep_obj *ep_obj, void *ctx);
int cxip_tx_id_free(struct cxip_ep_obj *ep_obj, int id);
void *cxip_tx_id_lookup(struct cxip_ep_obj *ep_obj, int id);

int cxip_rdzv_id_alloc(struct cxip_ep_obj *ep_obj, void *ctx);
int cxip_rdzv_id_free(struct cxip_ep_obj *ep_obj, int id);
void *cxip_rdzv_id_lookup(struct cxip_ep_obj *ep_obj, int id);
int cxip_ep_cmdq(struct cxip_ep_obj *ep_obj, uint32_t ctx_id, bool transmit,
		 uint32_t tclass, struct cxi_eq *evtq, struct cxip_cmdq **cmdq);
void cxip_ep_cmdq_put(struct cxip_ep_obj *ep_obj, uint32_t ctx_id,
		      bool transmit);

int cxip_recv_ux_sw_matcher(struct cxip_ux_send *ux);
int cxip_recv_req_sw_matcher(struct cxip_req *req);
int cxip_recv_cancel(struct cxip_req *req);
int cxip_fc_process_drops(struct cxip_ep_obj *ep_obj, uint8_t rxc_id,
			  uint32_t nic_addr, uint32_t pid, uint8_t txc_id,
			  uint16_t drops);
void cxip_recv_pte_cb(struct cxip_pte *pte, const union c_event *event);
void cxip_rxc_req_fini(struct cxip_rxc *rxc);
int cxip_rxc_oflow_init(struct cxip_rxc *rxc);
void cxip_rxc_oflow_fini(struct cxip_rxc *rxc);
int cxip_txc_zbp_init(struct cxip_txc *txc);
int cxip_txc_zbp_fini(struct cxip_txc *txc);
int cxip_txc_rdzv_src_fini(struct cxip_txc *txc);
int cxip_fc_resume(struct cxip_ep_obj *ep_obj, uint8_t txc_id,
		   uint32_t nic_addr, uint32_t pid, uint8_t rxc_id);

struct cxip_txc *cxip_txc_alloc(const struct fi_tx_attr *attr, void *context,
				int use_shared);
int cxip_txc_enable(struct cxip_txc *txc);
struct cxip_txc *cxip_stx_alloc(const struct fi_tx_attr *attr, void *context);
void cxip_txc_free(struct cxip_txc *txc);

int cxip_rxc_msg_enable(struct cxip_rxc *rxc, uint32_t drop_count);
int cxip_rxc_enable(struct cxip_rxc *rxc);
struct cxip_rxc *cxip_rxc_alloc(const struct fi_rx_attr *attr,
				      void *context, int use_shared);
void cxip_rxc_free(struct cxip_rxc *rxc);

int cxip_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		 struct fid_eq **eq, void *context);

bool cxip_cq_saturated(struct cxip_cq *cq);
struct cxip_md *cxip_cq_ibuf_md(void *ibuf);
void *cxip_cq_ibuf_alloc(struct cxip_cq *cq);
void cxip_cq_ibuf_free(struct cxip_cq *cq, void *ibuf);
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
int cxip_cq_adjust_reserved_fc_event_slots(struct cxip_cq *cq, int value);
void cxip_cq_flush_trig_reqs(struct cxip_cq *cq);

void cxip_dom_cntr_disable(struct cxip_domain *dom);
int cxip_cntr_mod(struct cxip_cntr *cxi_cntr, uint64_t value, bool set,
		  bool err);
int cxip_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		   struct fid_cntr **cntr, void *context);

int cxip_iomm_init(struct cxip_domain *dom);
void cxip_iomm_fini(struct cxip_domain *dom);
int cxip_map(struct cxip_domain *dom, const void *buf, unsigned long len,
	     struct cxip_md **md);
void cxip_unmap(struct cxip_md *md);

int cxip_ctrl_msg_send(struct cxip_ctrl_req *req);
void cxip_ep_ctrl_progress(struct cxip_ep_obj *ep_obj);
void cxip_ep_tx_ctrl_progress(struct cxip_ep_obj *ep_obj);
int cxip_ep_ctrl_init(struct cxip_ep_obj *ep_obj);
void cxip_ep_ctrl_fini(struct cxip_ep_obj *ep_obj);

int cxip_av_set(struct fid_av *av, struct fi_av_set_attr *attr,
	        struct fid_av_set **av_set_fid, void * context);

int cxip_coll_init(struct cxip_ep_obj *ep_obj);
int cxip_coll_enable(struct cxip_ep_obj *ep_obj);
int cxip_coll_disable(struct cxip_ep_obj *ep_obj);
int cxip_coll_close(struct cxip_ep_obj *ep_obj);
void cxip_coll_populate_opcodes(void);
int cxip_fi2cxi_opcode(int op, int datatype);
int cxip_coll_send(struct cxip_coll_reduction *reduction,
		   int av_set_idx, const void *buffer, size_t buflen,
		   struct cxi_md *md);
int cxip_coll_send_red_pkt(struct cxip_coll_reduction *reduction,
			   int arm, size_t redcnt, int op, const void *data,
			   int len, enum cxip_coll_rc red_rc, bool retry);
ssize_t cxip_coll_inject(struct cxip_coll_mc *mc_obj,
			 enum fi_datatype datatype, int cxi_opcode,
			 const void *op_send_data, void *op_rslt_data,
			 size_t op_count, void *context, int *reduction_id);
ssize_t cxip_barrier(struct fid_ep *ep, fi_addr_t coll_addr, void *context);
ssize_t cxip_broadcast(struct fid_ep *ep, void *buf, size_t count,
		       void *desc, fi_addr_t coll_addr, fi_addr_t root_addr,
		       enum fi_datatype datatype, uint64_t flags,
		       void *context);
ssize_t cxip_reduce(struct fid_ep *ep, const void *buf, size_t count,
		    void *desc, void *result, void *result_desc,
		    fi_addr_t coll_addr, fi_addr_t root_addr,
		    enum fi_datatype datatype, enum fi_op op, uint64_t flags,
		    void *context);
ssize_t cxip_allreduce(struct fid_ep *ep, const void *buf, size_t count, void *desc,
		       void *result, void *result_desc, fi_addr_t coll_addr,
		       enum fi_datatype datatype, enum fi_op op,
		       uint64_t flags, void *context);
int cxip_join_collective(struct fid_ep *ep, fi_addr_t coll_addr,
			 const struct fid_av_set *coll_av_set,
			 uint64_t flags, struct fid_mc **mc, void *context);

void cxip_coll_limit_red_id(struct fid_mc *mc, int max_red_id);
int cxip_coll_arm_enable(struct fid_mc *mc, bool enable);
void cxip_coll_reset_mc_ctrs(struct fid_mc *mc);

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

static inline void cxip_txq_ring(struct cxip_cmdq *cmdq, bool more,
				 bool triggered, int otx_reqs)
{
	if (triggered) {
		cxi_cq_ring(cmdq->dev_cmdq);
		return;
	}

	if (!more) {
		switch (cmdq->llring_mode) {
		case CXIP_LLRING_IDLE:
			if (!otx_reqs)
				cxi_cq_ll_ring(cmdq->dev_cmdq);
			else
				cxi_cq_ring(cmdq->dev_cmdq);
			break;
		case CXIP_LLRING_ALWAYS:
			cxi_cq_ll_ring(cmdq->dev_cmdq);
			break;
		case CXIP_LLRING_NEVER:
		default:
			cxi_cq_ring(cmdq->dev_cmdq);
			break;
		}
	}
}

ssize_t cxip_send_common(struct cxip_txc *txc, const void *buf, size_t len,
			void *desc, uint64_t data, fi_addr_t dest_addr,
			uint64_t tag, void *context, uint64_t flags,
			bool tagged, bool triggered, uint64_t trig_thresh,
			struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr);

ssize_t cxip_recv_common(struct cxip_rxc *rxc, void *buf, size_t len,
			 void *desc, fi_addr_t src_addr, uint64_t tag,
			 uint64_t ignore, void *context, uint64_t flags,
			 bool tagged, struct cxip_cntr *comp_cntr);

ssize_t cxip_rma_common(enum fi_op_type op, struct cxip_txc *txc,
			const void *buf, size_t len, void *desc,
			fi_addr_t tgt_addr, uint64_t addr, uint64_t key,
			uint64_t data, uint64_t flags, void *context,
			bool triggered, uint64_t trig_thresh,
			struct cxip_cntr *trig_cntr,
			struct cxip_cntr *comp_cntr);

/*
 * Request variants:
 *   CXIP_RQ_AMO
 *      Passes one argument (operand1), and applies that to a remote memory
 *      address content.
 *
 *   CXIP_RQ_AMO_FETCH
 *      Passes two arguments (operand1, resultptr), applies operand1 to a
 *      remote memory address content, and returns the prior content of the
 *      remote memory in resultptr.
 *
 *   CXIP_RQ_AMO_SWAP
 *      Passes three arguments (operand1, compare, resultptr). If remote memory
 *      address content satisfies the comparison operation with compare,
 *      replaces the remote memory content with operand1, and returns the prior
 *      content of the remote memory in resultptr.
 *
 */
enum cxip_amo_req_type {
	CXIP_RQ_AMO,
	CXIP_RQ_AMO_FETCH,
	CXIP_RQ_AMO_SWAP,
	CXIP_RQ_AMO_LAST,
};

int cxip_amo_common(enum cxip_amo_req_type req_type, struct cxip_txc *txc,
		    const struct fi_msg_atomic *msg,
		    const struct fi_ioc *comparev, void **comparedesc,
		    size_t compare_count, const struct fi_ioc *resultv,
		    void **resultdesc, size_t result_count, uint64_t flags,
		    bool triggered, uint64_t trig_thresh,
		    struct cxip_cntr *trig_cntr, struct cxip_cntr *comp_cntr);

static inline void
cxip_domain_add_txc(struct cxip_domain *dom, struct cxip_txc *txc)
{
	fastlock_acquire(&dom->lock);
	dlist_insert_tail(&txc->dom_entry, &dom->txc_list);
	fastlock_release(&dom->lock);
}

static inline void
cxip_domain_remove_txc(struct cxip_domain *dom, struct cxip_txc *txc)
{
	fastlock_acquire(&dom->lock);
	dlist_remove(&txc->dom_entry);
	fastlock_release(&dom->lock);
}

static inline void
cxip_domain_add_cntr(struct cxip_domain *dom, struct cxip_cntr *cntr)
{
	fastlock_acquire(&dom->lock);
	dlist_insert_tail(&cntr->dom_entry, &dom->cntr_list);
	ofi_atomic_inc32(&dom->ref);
	fastlock_release(&dom->lock);
}

static inline void
cxip_domain_remove_cntr(struct cxip_domain *dom, struct cxip_cntr *cntr)
{
	fastlock_acquire(&dom->lock);
	dlist_remove(&cntr->dom_entry);
	ofi_atomic_dec32(&dom->ref);
	fastlock_release(&dom->lock);
}

static inline void
cxip_domain_add_cq(struct cxip_domain *dom, struct cxip_cq *cq)
{
	fastlock_acquire(&dom->lock);
	dlist_insert_tail(&cq->dom_entry, &dom->cq_list);
	ofi_atomic_inc32(&dom->ref);
	fastlock_release(&dom->lock);
}

static inline void
cxip_domain_remove_cq(struct cxip_domain *dom, struct cxip_cq *cq)
{
	fastlock_acquire(&dom->lock);
	dlist_remove(&cq->dom_entry);
	ofi_atomic_dec32(&dom->ref);
	fastlock_release(&dom->lock);
}

static inline uint32_t cxip_mac_to_nic(struct ether_addr *mac)
{
	return mac->ether_addr_octet[5] |
			(mac->ether_addr_octet[4] << 8) |
			((mac->ether_addr_octet[3] & 0xF) << 16);
}

static inline bool is_netsim(struct cxip_ep_obj *ep_obj)
{
	return (ep_obj->domain->iface->info->device_platform ==
		CXI_PLATFORM_NETSIM);
}

#define _CXIP_DBG(subsys, ...) FI_DBG(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_INFO(subsys, ...) FI_INFO(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_WARN(subsys, ...) FI_WARN(&cxip_prov, subsys, __VA_ARGS__)
#define _CXIP_WARN_ONCE(subsys, ...) FI_WARN_ONCE(&cxip_prov, subsys, \
						  __VA_ARGS__)
#define CXIP_LOG(...)						\
	fi_log(&cxip_prov, FI_LOG_WARN, FI_LOG_CORE,		\
	       __func__, __LINE__, __VA_ARGS__)

#define CXIP_FATAL(...)						\
	do {							\
		CXIP_LOG(__VA_ARGS__);				\
		abort();					\
	} while (0)

#define TXC_DBG(txc, fmt, ...) \
	_CXIP_DBG(FI_LOG_EP_DATA, "TXC (%#x:%u:%u): " fmt "", \
		  (txc)->ep_obj->src_addr.nic, (txc)->ep_obj->src_addr.pid, \
		  (txc)->tx_id, ##__VA_ARGS__)
#define TXC_WARN(txc, fmt, ...) \
	_CXIP_WARN(FI_LOG_EP_DATA, "TXC (%#x:%u:%u): " fmt "", \
		   (txc)->ep_obj->src_addr.nic, (txc)->ep_obj->src_addr.pid, \
		   (txc)->tx_id, ##__VA_ARGS__)
#define TXC_FATAL(txc, fmt, ...) \
	CXIP_FATAL("TXC (%#x:%u:%u):: " fmt "", (txc)->ep_obj->src_addr.nic, \
		   (txc)->ep_obj->src_addr.pid, (txc)->tx_id, ##__VA_ARGS__)

#define RXC_DBG(rxc, fmt, ...) \
	_CXIP_DBG(FI_LOG_EP_DATA, "RXC (%#x:%u:%u): " fmt "", \
		  (rxc)->ep_obj->src_addr.nic, (rxc)->ep_obj->src_addr.pid, \
		  (rxc)->rx_id, ##__VA_ARGS__)
#define RXC_WARN(rxc, fmt, ...) \
	_CXIP_WARN(FI_LOG_EP_DATA, "RXC (%#x:%u:%u): " fmt "", \
		   (rxc)->ep_obj->src_addr.nic, (rxc)->ep_obj->src_addr.pid, \
		   (rxc)->rx_id, ##__VA_ARGS__)
#define RXC_FATAL(rxc, fmt, ...) \
	CXIP_FATAL("RXC (%#x:%u:%u): " fmt "", (rxc)->ep_obj->src_addr.nic, \
		   (rxc)->ep_obj->src_addr.pid, (rxc)->rx_id, ##__VA_ARGS__)

#endif
