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
#define CXIP_EP_MAX_CTX_BITS (16)
#define CXIP_EP_MSG_PREFIX_SZ (0)
#define CXIP_DOMAIN_MR_CNT (65535)

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

#define CXIP_EP_RDM_PRI_CAP \
	(FI_MSG | FI_RMA | FI_TAGGED | FI_ATOMICS | FI_NAMED_RX_CTX | \
	 FI_DIRECTED_RECV | FI_READ | FI_WRITE | FI_RECV | FI_SEND | \
	 FI_REMOTE_READ | FI_REMOTE_WRITE)

#define CXIP_EP_RDM_SEC_CAP_BASE \
	(FI_MULTI_RECV | FI_SOURCE | FI_RMA_EVENT | FI_SHARED_AV | FI_FENCE | \
	 FI_TRIGGER)
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

extern struct fi_provider cxip_prov;

/*
 * The CXI Provider Address format.
 *
 * A Cassini NIC Address and PID identify a libfabric Endpoint.  While Cassini
 * borrows the name 'PID' from Portals, we use the term 'Port' here since the
 * 1-1 mapping of PID to process does not exist.  The maximum PID value in
 * Cassini is 12 bits.  Practically, 9-10 bits will be used.  Therefore, we
 * could steal bits from the Port field if necessary.
 *
 * Port -1 is reserved.  When used, the library auto-assigns a free PID value
 * when network resources are allocated.  Libfabric clients can achieve this by
 * not specifying a 'service' in a call to fi_getinfo() or by specifying the
 * reserved value -1.
 *
 * TODO: If NIC Address must be non-zero, the valid bit can be removed.
 * TODO: Is 18 bits enough for NIC Address?
 */
struct cxip_addr {
	union {
		struct {
			uint32_t port		: 13;
			uint32_t nic		: 18;
			uint32_t valid		: 1;
		};
		uint32_t raw;
	};
};
#define CXIP_ADDR_PORT_AUTO 0x1fff

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#define CXIP_ADDR_MR_IDX(pid_granule, key) ((pid_granule) / 2 + (key))

struct cxip_if_domain {
	struct dlist_entry entry;
	struct cxip_if *dev_if;
	struct cxil_domain *if_dom;
	uint32_t vni;
	uint32_t pid;
	struct index_map lep_map; /* Cassini Logical EP Map */
	ofi_atomic32_t ref;
	fastlock_t lock;
};

struct cxip_if {
	struct slist_entry entry;
	uint32_t if_nic;
	uint32_t if_idx;
	uint32_t if_fabric;
	struct cxil_dev *if_dev;
	uint32_t if_pid_granule;
	struct cxil_lni *if_lni;
	struct cxi_cp cps[16];
	int n_cps;
	struct dlist_entry if_doms;
	ofi_atomic32_t ref;
	struct cxi_cmdq *mr_cmdq;
	struct cxi_evtq *mr_evtq;
	fastlock_t lock;
};

struct cxip_fabric {
	struct fid_fabric	fab_fid;
	ofi_atomic32_t		ref;
	struct dlist_entry	service_list;
	struct dlist_entry	fab_list_entry;
	fastlock_t		lock;
};

struct cxip_domain {
	struct fid_domain	dom_fid;
	struct fi_info		info;
	struct cxip_fabric	*fab;
	fastlock_t		lock;
	ofi_atomic32_t		ref;

	struct cxip_eq		*eq;
	struct cxip_eq		*mr_eq;

	enum fi_progress	progress_mode;
	struct dlist_entry	dom_list_entry;
	struct fi_domain_attr	attr;

	uint32_t		nic_addr;
	int			enabled;
	struct cxip_if		*dev_if;
};

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

struct cxip_req {
	/* Control info */
	struct cxip_cq *cq;
	int req_id;

	/* CQ event fields */
	uint64_t context;
	uint64_t flags;
	uint64_t data_len;
	uint64_t buf;
	uint64_t data;
	uint64_t tag;
	fi_addr_t addr;

	struct cxi_iova local_md;
	void (*cb)(struct cxip_req *req, const union c_event *evt);
};

struct cxip_cq;
typedef int (*cxip_cq_report_fn)(struct cxip_cq *cq, fi_addr_t addr,
				 struct cxip_req *req);

struct cxip_cq_overflow_entry_t {
	size_t len;
	fi_addr_t addr;
	struct dlist_entry entry;
	char cq_entry[0];
};

struct cxip_cq {
	struct fid_cq cq_fid;
	struct cxip_domain *domain;
	ssize_t cq_entry_size;
	ofi_atomic32_t ref;
	struct fi_cq_attr attr;

	struct ofi_ringbuf addr_rb;
	struct ofi_ringbuffd cq_rbfd;
	struct ofi_ringbuf cqerr_rb;
	struct dlist_entry overflow_list;
	fastlock_t lock;
	fastlock_t list_lock;

	struct fid_wait *waitset;
	int signal;
	ofi_atomic32_t signaled;

	struct dlist_entry ep_list;
	struct dlist_entry rx_list;
	struct dlist_entry tx_list;

	cxip_cq_report_fn report_completion;

	int enabled;
	struct cxi_evtq *evtq;
	fastlock_t req_lock;
	struct util_buf_pool *req_pool;
	struct indexer req_table;
};

struct cxip_cntr {
	struct fid_cntr		cntr_fid;
	struct cxip_domain	*domain;
	ofi_atomic32_t		ref;
	struct fi_cntr_attr	attr;

	struct fid_wait		*waitset;
	int			signal;

	struct dlist_entry	rx_list;
	struct dlist_entry	tx_list;
	fastlock_t		list_lock;
};

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

struct cxip_rx_ctx {
	struct fid_ep ctx;

	uint16_t rx_id;
	int enabled;
	int progress;
	int recv_cq_event;
	int use_shared;

	size_t num_left;
	size_t buffered_len;
	size_t min_multi_recv;
	uint64_t addr;
	struct cxip_comp comp;
	struct cxip_rx_ctx *srx_ctx;

	struct cxip_ep_attr *ep_attr;
	struct cxip_av *av;
	struct cxip_domain *domain;

	struct dlist_entry cq_entry;
	struct dlist_entry ep_list;
	fastlock_t lock;

	struct fi_rx_attr attr;
};

struct cxip_tx_ctx {
	union {
		struct fid_ep ctx;
		struct fid_stx stx;
	} fid;
	size_t fclass;

	uint16_t tx_id;
	uint8_t enabled;
	uint8_t progress;

	int use_shared;
	struct cxip_comp comp;
	struct cxip_tx_ctx *stx_ctx;

	struct cxip_ep_attr *ep_attr;
	struct cxip_av *av;
	struct cxip_domain *domain;

	struct dlist_entry cq_entry;
	struct dlist_entry ep_list;
	fastlock_t lock;

	struct fi_tx_attr attr;

	struct cxi_cmdq *tx_cmdq;
};

struct cxip_ep_attr {
	size_t fclass;

	int tx_shared;
	int rx_shared;
	size_t buffered_len;
	size_t min_multi_recv;

	ofi_atomic32_t ref;
	struct cxip_eq *eq;
	struct cxip_av *av;
	struct cxip_domain *domain;

	/* TX/RX context pointers for standard EPs. */
	struct cxip_rx_ctx *rx_ctx;
	struct cxip_tx_ctx *tx_ctx;

	/* TX/RX contexts.  Standard EPs have 1 of each.  SEPs have many. */
	struct cxip_rx_ctx **rx_array;
	struct cxip_tx_ctx **tx_array;
	ofi_atomic32_t num_rx_ctx;
	ofi_atomic32_t num_tx_ctx;

	/* List of contexts associated with the EP.  Necessary? */
	struct dlist_entry rx_ctx_entry;
	struct dlist_entry tx_ctx_entry;

	struct fi_info info;
	struct fi_ep_attr ep_attr;

	enum fi_ep_type ep_type;

	int is_enabled;
	fastlock_t lock;

	struct cxip_addr *src_addr;
	uint32_t vni;
	struct cxip_if_domain *if_dom;
};

struct cxip_ep {
	struct fid_ep ep;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;
	struct cxip_ep_attr *attr;
	int is_alias;
};

struct cxip_mr {
	struct fid_mr mr_fid;
	struct cxip_domain *domain;
	struct cxip_ep *ep;
	uint64_t key;
	uint64_t flags;
	struct fi_mr_attr attr;
	struct cxip_cntr *cntr;
	struct cxip_cq *cq;
	fastlock_t lock;

	/*
	 * A standard MR is implemented as a single persistent, non-matching
	 * list entry (LE) on the PtlTE mapped to the logical endpoint
	 * addressed with the four-tuple:
	 *
	 *    ( if_dom->dev_if->if_nic, if_dom->pid, vni, pid_idx )
	 */
	uint32_t pid_off;

	int enabled;
	struct cxil_pte *pte;
	unsigned int pte_hw_id;
	struct cxil_pte_map *pte_map;

	void *buf;
	uint64_t len;
	struct cxi_iova md;
};

struct cxip_av_table_hdr {
	uint64_t size;
	uint64_t stored;
};

struct cxip_av {
	struct fid_av av_fid;
	struct cxip_domain *domain;
	ofi_atomic32_t ref;
	struct fi_av_attr attr;
	uint64_t mask;
	int rx_ctx_bits;
	socklen_t addrlen;
	struct cxip_eq *eq;
	struct cxip_av_table_hdr *table_hdr;
	struct cxip_addr *table;
	uint64_t *idx_arr;
	struct util_shm shm;
	int shared;
	struct dlist_entry ep_list;
	fastlock_t list_lock;
};

struct cxip_fid_list {
	struct dlist_entry entry;
	struct fid *fid;
};

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
int cxip_if_domain_lep_alloc(struct cxip_if_domain *if_dom, uint64_t lep_idx);
int cxip_if_domain_lep_free(struct cxip_if_domain *if_dom, uint64_t lep_idx);
void cxip_if_init(void);
void cxip_if_fini(void);

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
