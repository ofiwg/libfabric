/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#ifndef _CXI_PROV_H_
#define _CXI_PROV_H_

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

#define CXI_EP_MAX_MSG_SZ (1<<23)
#define CXI_EP_MAX_INJECT_SZ ((1<<8) - 1)
#define CXI_EP_MAX_BUFF_RECV (1<<26)
#define CXI_EP_MAX_ORDER_RAW_SZ CXI_EP_MAX_MSG_SZ
#define CXI_EP_MAX_ORDER_WAR_SZ CXI_EP_MAX_MSG_SZ
#define CXI_EP_MAX_ORDER_WAW_SZ CXI_EP_MAX_MSG_SZ
#define CXI_EP_MEM_TAG_FMT FI_TAG_GENERIC
#define CXI_EP_MAX_EP_CNT (128)
#define CXI_EP_MAX_CQ_CNT (32)
#define CXI_EP_MAX_CNTR_CNT (128)
#define CXI_EP_MAX_TX_CNT (16)
#define CXI_EP_MAX_RX_CNT (16)
#define CXI_EP_MAX_IOV_LIMIT (8)
#define CXI_EP_TX_SZ (256)
#define CXI_EP_RX_SZ (256)
#define CXI_EP_MIN_MULTI_RECV (64)
#define CXI_EP_MAX_ATOMIC_SZ (4096)
#define CXI_EP_MAX_CTX_BITS (16)
#define CXI_EP_MSG_PREFIX_SZ (0)
#define CXI_DOMAIN_MR_CNT (65535)

#define CXI_EQ_DEF_SZ (1<<8)
#define CXI_CQ_DEF_SZ (1<<8)
#define CXI_AV_DEF_SZ (1<<8)
#define CXI_CMAP_DEF_SZ (1<<10)
#define CXI_EPOLL_WAIT_EVENTS 32

#define CXI_CQ_DATA_SIZE (sizeof(uint64_t))
#define CXI_TAG_SIZE (sizeof(uint64_t))
#define CXI_MAX_NETWORK_ADDR_SZ (35)

#define CXI_PEP_LISTENER_TIMEOUT (10000)
#define CXI_CM_COMM_TIMEOUT (2000)
#define CXI_EP_MAX_RETRY (5)
#define CXI_EP_MAX_CM_DATA_SZ (256)

#define CXI_EP_RDM_PRI_CAP (FI_MSG | FI_RMA | FI_TAGGED | FI_ATOMICS |	\
			 FI_NAMED_RX_CTX | \
			 FI_DIRECTED_RECV | \
			 FI_READ | FI_WRITE | FI_RECV | FI_SEND | \
			 FI_REMOTE_READ | FI_REMOTE_WRITE)

#define CXI_EP_RDM_SEC_CAP_BASE (FI_MULTI_RECV | FI_SOURCE | FI_RMA_EVENT | \
				  FI_SHARED_AV | FI_FENCE | FI_TRIGGER)
extern uint64_t CXI_EP_RDM_SEC_CAP;

#define CXI_EP_RDM_CAP_BASE (CXI_EP_RDM_PRI_CAP | CXI_EP_RDM_SEC_CAP_BASE)
extern uint64_t CXI_EP_RDM_CAP;

#define CXI_EP_MSG_ORDER (FI_ORDER_RAR | FI_ORDER_RAW | FI_ORDER_RAS|	\
			  FI_ORDER_WAR | FI_ORDER_WAW | FI_ORDER_WAS |	\
			  FI_ORDER_SAR | FI_ORDER_SAW | FI_ORDER_SAS)

#define CXI_EP_COMP_ORDER (FI_ORDER_STRICT | FI_ORDER_DATA)
#define CXI_EP_DEFAULT_OP_FLAGS (FI_TRANSMIT_COMPLETE)

#define CXI_EP_CQ_FLAGS (FI_SEND | FI_TRANSMIT | FI_RECV | \
			FI_SELECTIVE_COMPLETION)
#define CXI_EP_CNTR_FLAGS (FI_SEND | FI_RECV | FI_READ | \
			FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE)

#define CXI_EP_SET_TX_OP_FLAGS(_flags) do {			\
		if (!((_flags) & FI_INJECT_COMPLETE))		\
			(_flags) |= FI_TRANSMIT_COMPLETE;	\
	} while (0)

#define CXI_MODE (0)

#define CXI_MAX_ERR_CQ_EQ_DATA_SZ CXI_EP_MAX_CM_DATA_SZ

#define CXI_MAJOR_VERSION 0
#define CXI_MINOR_VERSION 0

#define CXI_WIRE_PROTO_VERSION (1)

#define CXIX_NUM_PIDS_DEF 128
#define CXIX_PID_GRANULE_DEF 1024

extern const char cxi_fab_fmt[];
extern const char cxi_dom_fmt[];
extern const char cxi_prov_name[];
extern struct fi_provider cxi_prov;
extern int cxi_av_def_sz;
extern int cxi_cq_def_sz;
extern int cxi_eq_def_sz;
extern struct slist cxix_if_list;
extern int cxix_num_pids;

extern struct fi_provider cxi_prov;

struct cxi_addr {
	union {
		struct {
			uint64_t port	: 20;
			uint64_t domain	: 20;
			uint64_t nic	: 20;
			uint64_t flags	: 4;
		};
		uint64_t qw;
	};
};

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#endif

#define CXI_ADDR_INIT {{{0}}}
#define CXI_ADDR_FLAG_AV_ENTRY_VALID 1
#define CXI_ADDR_AV_ENTRY_VALID(addr) \
		((addr)->flags & CXI_ADDR_FLAG_AV_ENTRY_VALID)
#define CXI_ADDR_AV_ENTRY_SET_VALID(addr) \
		((addr)->flags |= CXI_ADDR_FLAG_AV_ENTRY_VALID)
#define CXI_ADDR_AV_ENTRY_CLR_VALID(addr) \
		((addr)->flags &= ~CXI_ADDR_FLAG_AV_ENTRY_VALID)

struct cxix_if_domain {
	struct dlist_entry entry;
	struct cxix_if *dev_if;
	struct cxil_domain *if_dom;
	uint32_t vni;
	uint32_t pid;
	uint32_t pid_granule;
	struct index_map lep_map; /* Cassini Logical EP Map */
	ofi_atomic32_t ref;
	fastlock_t lock;
};

struct cxix_if {
	struct slist_entry entry;
	uint32_t if_nic;
	uint32_t if_idx;
	uint32_t if_fabric;
	struct cxil_dev *if_dev;
	struct cxil_lni *if_lni;
	struct cxi_cp cps[16];
	int n_cps;
	struct dlist_entry if_doms;
	ofi_atomic32_t ref;
	struct cxi_cmdq *mr_cmdq;
	struct cxi_evtq *mr_evtq;
	fastlock_t lock;
};

struct cxi_fabric {
	struct fid_fabric	fab_fid;
	ofi_atomic32_t		ref;
	struct dlist_entry	service_list;
	struct dlist_entry	fab_list_entry;
	fastlock_t		lock;
};

struct cxi_domain {
	struct fid_domain	dom_fid;
	struct fi_info		info;
	struct cxi_fabric	*fab;
	fastlock_t		lock;
	ofi_atomic32_t		ref;

	struct cxi_eq		*eq;
	struct cxi_eq		*mr_eq;

	enum fi_progress	progress_mode;
	struct dlist_entry	dom_list_entry;
	struct fi_domain_attr	attr;

	uint32_t		nic_addr;
	uint32_t		vni;
	uint32_t		pid;
	uint32_t		pid_granule;
	int			enabled;
	struct cxix_if		*dev_if;
};

struct cxi_eq {
	struct fid_eq eq;
	struct fi_eq_attr attr;
	struct cxi_fabric *cxi_fab;

	struct dlistfd_head list;
	struct dlistfd_head err_list;
	struct dlist_entry err_data_list;
	fastlock_t lock;

	struct fid_wait *waitset;
	int signal;
	int wait_fd;
};

struct cxi_req {
	uint8_t type;
	int req_id;

	uint64_t flags;
	uint64_t context;
	uint64_t addr;
	uint64_t data;
	uint64_t tag;
	uint64_t buf;
	uint64_t data_len;

	struct cxi_cq *cq;
};

struct cxi_cq;
typedef int (*cxi_cq_report_fn) (struct cxi_cq *cq, fi_addr_t addr,
				 struct cxi_req *req);

struct cxi_cq_overflow_entry_t {
	size_t len;
	fi_addr_t addr;
	struct dlist_entry entry;
	char cq_entry[0];
};

struct cxi_cq {
	struct fid_cq cq_fid;
	struct cxi_domain *domain;
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

	cxi_cq_report_fn report_completion;

	int enabled;
	struct cxi_evtq *evtq;
	fastlock_t req_lock;
	struct util_buf_pool *req_pool;
	struct indexer req_table;
};

struct cxi_cntr {
	struct fid_cntr		cntr_fid;
	struct cxi_domain	*domain;
	ofi_atomic32_t		ref;
	struct fi_cntr_attr	attr;

	struct fid_wait		*waitset;
	int			signal;

	struct dlist_entry	rx_list;
	struct dlist_entry	tx_list;
	fastlock_t		list_lock;
};

struct cxi_comp {
	uint8_t send_cq_event;
	uint8_t recv_cq_event;
	char reserved[2];

	struct cxi_cq	*send_cq;
	struct cxi_cq	*recv_cq;

	struct cxi_cntr *send_cntr;
	struct cxi_cntr *recv_cntr;
	struct cxi_cntr *read_cntr;
	struct cxi_cntr *write_cntr;
	struct cxi_cntr *rem_read_cntr;
	struct cxi_cntr *rem_write_cntr;

	struct cxi_eq *eq;
};

struct cxi_rx_ctx {
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
	struct cxi_comp comp;
	struct cxi_rx_ctx *srx_ctx;

	struct cxi_ep_attr *ep_attr;
	struct cxi_av *av;
	struct cxi_domain *domain;

	struct dlist_entry cq_entry;
	struct dlist_entry ep_list;
	fastlock_t lock;

	struct fi_rx_attr attr;
};

struct cxi_tx_ctx {
	union {
		struct fid_ep ctx;
		struct fid_stx stx;
	} fid;
	size_t fclass;

	uint16_t tx_id;
	uint8_t enabled;
	uint8_t progress;

	int use_shared;
	struct cxi_comp comp;
	struct cxi_tx_ctx *stx_ctx;

	struct cxi_ep_attr *ep_attr;
	struct cxi_av *av;
	struct cxi_domain *domain;

	struct dlist_entry cq_entry;
	struct dlist_entry ep_list;
	fastlock_t lock;

	struct fi_tx_attr attr;

	struct cxi_cmdq *tx_cmdq;
};

struct cxi_ep_attr {
	size_t fclass;

	int tx_shared;
	int rx_shared;
	size_t buffered_len;
	size_t min_multi_recv;

	ofi_atomic32_t ref;
	struct cxi_eq *eq;
	struct cxi_av *av;
	struct cxi_domain *domain;

	/* TX/RX context pointers for standard EPs. */
	struct cxi_rx_ctx *rx_ctx;
	struct cxi_tx_ctx *tx_ctx;

	/* TX/RX contexts.  Standard EPs have 1 of each.  SEPs have many. */
	struct cxi_rx_ctx **rx_array;
	struct cxi_tx_ctx **tx_array;
	ofi_atomic32_t num_rx_ctx;
	ofi_atomic32_t num_tx_ctx;

	/* List of contexts associated with the EP.  Necessary? */
	struct dlist_entry rx_ctx_entry;
	struct dlist_entry tx_ctx_entry;

	struct fi_info info;
	struct fi_ep_attr ep_attr;

	enum fi_ep_type ep_type;
	struct cxi_addr *src_addr;

	int is_enabled;
	fastlock_t lock;
};

struct cxi_ep {
	struct fid_ep ep;
	struct fi_tx_attr tx_attr;
	struct fi_rx_attr rx_attr;
	struct cxi_ep_attr *attr;
	int is_alias;
};

struct cxi_av_table_hdr {
	uint64_t size;
	uint64_t stored;
};

struct cxi_av {
	struct fid_av av_fid;
	struct cxi_domain *domain;
	ofi_atomic32_t ref;
	struct fi_av_attr attr;
	uint64_t mask;
	int rx_ctx_bits;
	socklen_t addrlen;
	struct cxi_eq *eq;
	struct cxi_av_table_hdr *table_hdr;
	struct cxi_addr *table;
	uint64_t *idx_arr;
	struct util_shm shm;
	int    shared;
	struct dlist_entry ep_list;
	fastlock_t list_lock;
};

struct cxi_fid_list {
	struct dlist_entry entry;
	struct fid *fid;
};

struct cxi_wait {
	struct fid_wait wait_fid;
	struct cxi_fabric *fab;
	struct dlist_entry fid_list;
	enum fi_wait_obj type;
	union {
		int fd[2];
		struct cxi_mutex_cond {
			pthread_mutex_t	mutex;
			pthread_cond_t	cond;
		} mutex_cond;
	} wobj;
};

struct cxix_if *cxix_if_lookup(uint32_t nic_addr);
int cxix_get_if(uint32_t nic_addr, struct cxix_if **dev_if);
void cxix_put_if(struct cxix_if *dev_if);
int cxix_get_if_domain(struct cxix_if *dev_if, uint32_t vni, uint32_t pid,
		       uint32_t pid_granule, struct cxix_if_domain **if_dom);
void cxix_put_if_domain(struct cxix_if_domain *if_dom);
int cxix_if_domain_lep_alloc(struct cxix_if_domain *if_dom, uint64_t lep_idx);
int cxix_if_domain_lep_free(struct cxix_if_domain *if_dom, uint64_t lep_idx);
void cxix_if_init(void);
void cxix_if_fini(void);

int cxi_parse_addr(const char *node, const char *service,
		   struct cxi_addr *addr);

int cxix_domain_enable(struct cxi_domain *dom);
int cxi_domain(struct fid_fabric *fabric, struct fi_info *info,
	       struct fid_domain **dom, void *context);

char *cxi_get_fabric_name(struct cxi_addr *src_addr);
char *cxi_get_domain_name(struct cxi_addr *src_addr);

void cxi_dom_add_to_list(struct cxi_domain *domain);
int cxi_dom_check_list(struct cxi_domain *domain);
void cxi_dom_remove_from_list(struct cxi_domain *domain);
struct cxi_domain *cxi_dom_list_head(void);
int cxi_dom_check_manual_progress(struct cxi_fabric *fabric);

void cxi_fab_add_to_list(struct cxi_fabric *fabric);
int cxi_fab_check_list(struct cxi_fabric *fabric);
void cxi_fab_remove_from_list(struct cxi_fabric *fabric);
struct cxi_fabric *cxi_fab_list_head(void);

int cxi_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		struct fid_av **av, void *context);

struct fi_info *cxi_fi_info(uint32_t version, enum fi_ep_type ep_type,
			    const struct fi_info *hints, void *src_addr,
			    void *dest_addr);
int cxi_rdm_fi_info(uint32_t version, void *src_addr, void *dest_addr,
		    const struct fi_info *hints, struct fi_info **info);
int cxi_alloc_endpoint(struct fid_domain *domain, struct fi_info *info,
		       struct cxi_ep **ep, void *context, size_t fclass);
int cxi_rdm_ep(struct fid_domain *domain, struct fi_info *info,
	       struct fid_ep **ep, void *context);
int cxi_rdm_sep(struct fid_domain *domain, struct fi_info *info,
		struct fid_ep **sep, void *context);
int cxi_ep_enable(struct fid_ep *ep);
int cxi_ep_disable(struct fid_ep *ep);

int cxi_verify_info(uint32_t version, const struct fi_info *hints);
int cxi_verify_fabric_attr(const struct fi_fabric_attr *attr);
int cxi_get_src_addr(struct cxi_addr *dest_addr, struct cxi_addr *src_addr);

int cxi_rdm_verify_ep_attr(const struct fi_ep_attr *ep_attr,
			   const struct fi_tx_attr *tx_attr,
			   const struct fi_rx_attr *rx_attr);

int cxi_verify_domain_attr(uint32_t version, const struct fi_info *info);
int cxi_domain(struct fid_fabric *fabric, struct fi_info *info,
	       struct fid_domain **dom, void *context);

int cxi_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
		struct fid_eq **eq, void *context);

int cxi_wait_get_obj(struct fid_wait *fid, void *arg);
void cxi_wait_signal(struct fid_wait *wait_fid);
int cxi_wait_close(fid_t fid);
int cxi_wait_open(struct fid_fabric *fabric, struct fi_wait_attr *attr,
		  struct fid_wait **waitset);

struct cxi_rx_ctx *cxi_rx_ctx_alloc(const struct fi_rx_attr *attr,
				    void *context, int use_shared);
void cxi_rx_ctx_free(struct cxi_rx_ctx *rx_ctx);

int cxix_tx_ctx_enable(struct cxi_tx_ctx *txc);
struct cxi_tx_ctx *cxi_tx_ctx_alloc(const struct fi_tx_attr *attr,
				    void *context, int use_shared);
struct cxi_tx_ctx *cxi_stx_ctx_alloc(const struct fi_tx_attr *attr, void *context);
void cxi_tx_ctx_free(struct cxi_tx_ctx *tx_ctx);

struct cxi_req *cxix_cq_req_alloc(struct cxi_cq *cq, int remap);
void cxix_cq_req_free(struct cxi_req *req);
void cxi_cq_add_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx);
void cxi_cq_remove_tx_ctx(struct cxi_cq *cq, struct cxi_tx_ctx *tx_ctx);
void cxi_cq_add_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx);
void cxi_cq_remove_rx_ctx(struct cxi_cq *cq, struct cxi_rx_ctx *rx_ctx);
int cxi_cq_progress(struct cxi_cq *cq);
int cxix_cq_enable(struct cxi_cq *cxi_cq);
int cxi_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq, void *context);
int cxi_cq_report_error(struct cxi_cq *cq, struct cxi_req *req,
			 size_t olen, int err, int prov_errno, void *err_data,
			 size_t err_data_size);

void cxi_cntr_add_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx);
void cxi_cntr_remove_tx_ctx(struct cxi_cntr *cntr, struct cxi_tx_ctx *tx_ctx);
void cxi_cntr_add_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx);
void cxi_cntr_remove_rx_ctx(struct cxi_cntr *cntr, struct cxi_rx_ctx *rx_ctx);
int cxi_cntr_progress(struct cxi_cntr *cntr);
int cxi_cntr_open(struct fid_domain *domain, struct fi_cntr_attr *attr,
		  struct fid_cntr **cntr, void *context);

#define _CXI_LOG_DBG(subsys, ...) FI_DBG(&cxi_prov, subsys, __VA_ARGS__)
#define _CXI_LOG_ERROR(subsys, ...) FI_WARN(&cxi_prov, subsys, __VA_ARGS__)

#endif
