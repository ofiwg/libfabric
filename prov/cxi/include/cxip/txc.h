/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_TXC_H_
#define _CXIP_TXC_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_cntr;
struct cxip_cq;
struct cxip_domain;
struct cxip_ep_obj;
struct cxip_md;
struct cxip_rdzv_match_pte;
struct cxip_rdzv_nomatch_pte;
struct cxip_req;

/* Macros */
#define CXIP_TXC_FORCE_ERR_ALT_READ_PROTO_ALLOC (1 << 0)

#define TXC_BASE(txc) ((struct cxip_txc *) (void *) (txc))

#define TXC_DBG(txc, fmt, ...)                             \
	_CXIP_DBG(FI_LOG_EP_DATA, "TXC (%#x:%u): " fmt "", \
		  TXC_BASE(txc)->ep_obj->src_addr.nic,     \
		  TXC_BASE(txc)->ep_obj->src_addr.pid, ##__VA_ARGS__)

#define TXC_INFO(txc, fmt, ...)                             \
	_CXIP_INFO(FI_LOG_EP_DATA, "TXC (%#x:%u): " fmt "", \
		   TXC_BASE(txc)->ep_obj->src_addr.nic,     \
		   TXC_BASE(txc)->ep_obj->src_addr.pid, ##__VA_ARGS__)

#define TXC_WARN(txc, fmt, ...)                             \
	_CXIP_WARN(FI_LOG_EP_DATA, "TXC (%#x:%u): " fmt "", \
		   TXC_BASE(txc)->ep_obj->src_addr.nic,     \
		   TXC_BASE(txc)->ep_obj->src_addr.pid, ##__VA_ARGS__)

#define TXC_WARN_RET(txc, ret, fmt, ...) \
	TXC_WARN(txc, "%d:%s: " fmt "", ret, fi_strerror(-ret), ##__VA_ARGS__)

#define TXC_FATAL(txc, fmt, ...)                        \
	CXIP_FATAL("TXC (%#x:%u):: " fmt "",            \
		   TXC_BASE(txc)->ep_obj->src_addr.nic, \
		   TXC_BASE(txc)->ep_obj->src_addr.pid, ##__VA_ARGS__)

/* Type definitions */
struct cxip_txc_ops {
	ssize_t (*send_common)(struct cxip_txc *txc, uint32_t tclass,
			       const void *buf, size_t len, void *desc,
			       uint64_t data, fi_addr_t dest_addr, uint64_t tag,
			       void *context, uint64_t flags, bool tagged,
			       bool triggered, uint64_t trig_thresh,
			       struct cxip_cntr *trig_cntr,
			       struct cxip_cntr *comp_cntr);
	void (*progress)(struct cxip_txc *txc, bool internal);
	int (*cancel_msg_send)(struct cxip_req *req);
	void (*init_struct)(struct cxip_txc *txc, struct cxip_ep_obj *ep_obj);
	void (*fini_struct)(struct cxip_txc *txc);
	void (*cleanup)(struct cxip_txc *txc);
	int (*msg_init)(struct cxip_txc *txc);
	int (*msg_fini)(struct cxip_txc *txc);
};

struct cxip_txc {
	void *context;

	uint32_t protocol;
	bool enabled;
	bool hrp_war_req; // Non-fetching 32-bit HRP
	bool hmem;
	bool trunc_ok;

	struct cxip_cq *send_cq;
	struct cxip_cntr *send_cntr;
	struct cxip_cntr *read_cntr;
	struct cxip_cntr *write_cntr;

	struct cxip_txc_ops ops;

	struct cxip_ep_obj *ep_obj; // parent EP object
	struct cxip_domain *domain; // parent domain
	uint8_t pid_bits;
	uint8_t recv_ptl_idx;

	struct fi_tx_attr attr; // attributes
	bool selective_completion;
	uint32_t tclass;

	/* TX H/W Event Queue */
	struct cxip_evtq tx_evtq;

	/* Inject buffers for EP, protected by ep_obj->lock */
	struct ofi_bufpool *ibuf_pool;

	struct cxip_cmdq *tx_cmdq; // added during cxip_txc_enable()
	int otx_reqs; // outstanding transmit requests

	/* Queue of TX messages in flight for the context */
	struct dlist_entry msg_queue;

	struct cxip_req *rma_write_selective_completion_req;
	struct cxip_req *rma_read_selective_completion_req;
	struct cxip_req *amo_selective_completion_req;
	struct cxip_req *amo_fetch_selective_completion_req;

	struct dlist_entry dom_entry;
};

struct cxip_txc_hpc {
	/* Must remain first */
	struct cxip_txc base;

	int max_eager_size;
	int rdzv_eager_size;

	/* Rendezvous messaging support */
	struct cxip_rdzv_match_pte *rdzv_pte;
	struct cxip_rdzv_nomatch_pte *rdzv_nomatch_pte[RDZV_NO_MATCH_PTES];
	struct indexer rdzv_ids;
	struct indexer msg_rdzv_ids;
	enum cxip_rdzv_proto rdzv_proto;

	struct cxip_cmdq *rx_cmdq; // Target cmdq for Rendezvous buffers

#if ENABLE_DEBUG
	uint64_t force_err;
#endif
	/* Flow Control recovery */
	struct dlist_entry fc_peers;

	/* Match complete IDs */
	struct indexer tx_ids;
};

struct cxip_txc_rnr {
	/* Must remain first */
	struct cxip_txc base;

	uint64_t max_retry_wait_us; /* Maximum time to retry any request */
	ofi_atomic32_t time_wait_reqs; /* Number of RNR time wait reqs */
	uint64_t next_retry_wait_us; /* Time of next retry in all queues */
	uint64_t total_retries;
	uint64_t total_rnr_nacks;
	bool hybrid_mr_desc;

	/* Used when success events are not required */
	struct cxip_req *req_selective_comp_msg;
	struct cxip_req *req_selective_comp_tag;

	/* There are CXIP_NUM_RNR_WAIT_QUEUE queues where each queue has
	 * a specified time wait value and where the last queue is has the
	 * maximum time wait value before retrying (and is used for all
	 * subsequent retries). This implementation allows each queue to
	 * be maintained in retry order with a simple append of the request.
	 */
	struct dlist_entry time_wait_queue[CXIP_NUM_RNR_WAIT_QUEUE];
};

/* Function declarations */
int cxip_txc_emit_idc_put(struct cxip_txc *txc, uint16_t vni,
			  enum cxi_traffic_class tc,
			  enum cxi_traffic_class_type tc_type,
			  const struct c_cstate_cmd *c_state,
			  const struct c_idc_put_cmd *put, const void *buf,
			  size_t len, uint64_t flags);

int cxip_txc_emit_dma(struct cxip_txc *txc, uint16_t vni,
		      enum cxi_traffic_class tc,
		      enum cxi_traffic_class_type tc_type,
		      struct cxip_cntr *trig_cntr, size_t trig_thresh,
		      struct c_full_dma_cmd *dma, uint64_t flags);

int cxip_txc_emit_idc_amo(struct cxip_txc *txc, uint16_t vni,
			  enum cxi_traffic_class tc,
			  enum cxi_traffic_class_type tc_type,
			  const struct c_cstate_cmd *c_state,
			  const struct c_idc_amo_cmd *amo, uint64_t flags,
			  bool fetching, bool flush);

int cxip_txc_emit_dma_amo(struct cxip_txc *txc, uint16_t vni,
			  enum cxi_traffic_class tc,
			  enum cxi_traffic_class_type tc_type,
			  struct cxip_cntr *trig_cntr, size_t trig_thresh,
			  struct c_dma_amo_cmd *amo, uint64_t flags,
			  bool fetching, bool flush);

int cxip_txc_emit_idc_msg(struct cxip_txc *txc, uint16_t vni,
			  enum cxi_traffic_class tc,
			  enum cxi_traffic_class_type tc_type,
			  const struct c_cstate_cmd *c_state,
			  const struct c_idc_msg_hdr *msg, const void *buf,
			  size_t len, uint64_t flags);

void cxip_txc_flush_msg_trig_reqs(struct cxip_txc *txc);

int cxip_tx_id_alloc(struct cxip_txc_hpc *txc, void *ctx);

int cxip_tx_id_free(struct cxip_txc_hpc *txc, int id);

void *cxip_tx_id_lookup(struct cxip_txc_hpc *txc, int id);

int cxip_rdzv_id_alloc(struct cxip_txc_hpc *txc, struct cxip_req *req);

int cxip_rdzv_id_free(struct cxip_txc_hpc *txc, int id);

void *cxip_rdzv_id_lookup(struct cxip_txc_hpc *txc, int id);

void cxip_txc_struct_init(struct cxip_txc *txc, const struct fi_tx_attr *attr,
			  void *context);

struct cxip_txc *cxip_txc_calloc(struct cxip_ep_obj *ep_obj, void *context);

void cxip_txc_free(struct cxip_txc *txc);

int cxip_txc_enable(struct cxip_txc *txc);

void cxip_txc_disable(struct cxip_txc *txc);

struct cxip_txc *cxip_stx_alloc(const struct fi_tx_attr *attr, void *context);

struct cxip_md *cxip_txc_ibuf_md(void *ibuf);

void *cxip_txc_ibuf_alloc(struct cxip_txc *txc);

void cxip_txc_ibuf_free(struct cxip_txc *txc, void *ibuf);

int cxip_ibuf_chunk_init(struct ofi_bufpool_region *region);

void cxip_ibuf_chunk_fini(struct ofi_bufpool_region *region);

#endif /* _CXIP_TXC_H_ */
