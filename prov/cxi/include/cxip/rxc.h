/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_RXC_H_
#define _CXIP_RXC_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <ofi_list.h>
#include <ofi_atom.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_cntr;
struct cxip_cq;
struct cxip_ctrl_req;
struct cxip_domain;
struct cxip_ep_obj;
struct cxip_pte;
struct cxip_ptelist_bufpool;
struct cxip_req;

/* Macros */
#define RXC_RESERVED_FC_SLOTS 1

#define RXC_BASE(rxc) ((struct cxip_rxc *)(void *)(rxc))

#define RXC_DBG(rxc, fmt, ...) \
	_CXIP_DBG(FI_LOG_EP_DATA, "RXC (%#x:%u) PtlTE %u: " fmt "", \
		  RXC_BASE(rxc)->ep_obj->src_addr.nic, \
		  RXC_BASE(rxc)->ep_obj->src_addr.pid, \
		  RXC_BASE(rxc)->rx_pte->pte->ptn, ##__VA_ARGS__)

#define RXC_INFO(rxc, fmt, ...) \
	_CXIP_INFO(FI_LOG_EP_DATA, "RXC (%#x:%u) PtlTE %u: " fmt "", \
		   RXC_BASE(rxc)->ep_obj->src_addr.nic, \
		   RXC_BASE(rxc)->ep_obj->src_addr.pid, \
		   RXC_BASE(rxc)->rx_pte->pte->ptn, ##__VA_ARGS__)

#define RXC_WARN(rxc, fmt, ...) \
	_CXIP_WARN(FI_LOG_EP_DATA, "RXC (%#x:%u) PtlTE %u: " fmt "", \
		   RXC_BASE(rxc)->ep_obj->src_addr.nic, \
		   RXC_BASE(rxc)->ep_obj->src_addr.pid, \
		   RXC_BASE(rxc)->rx_pte->pte->ptn, ##__VA_ARGS__)

#define RXC_WARN_ONCE(rxc, fmt, ...) \
	_CXIP_WARN_ONCE(FI_LOG_EP_DATA, "RXC (%#x:%u) PtlTE %u: " fmt "", \
			RXC_BASE(rxc)->ep_obj->src_addr.nic, \
			RXC_BASE(rxc)->ep_obj->src_addr.pid, \
			RXC_BASE(rxc)->rx_pte->pte->ptn, ##__VA_ARGS__)

#define RXC_FATAL(rxc, fmt, ...) \
	CXIP_FATAL("RXC (%#x:%u) PtlTE %u:[Fatal] " fmt "", \
		   RXC_BASE(rxc)->ep_obj->src_addr.nic, \
		   RXC_BASE(rxc)->ep_obj->src_addr.pid, \
		   RXC_BASE(rxc)->rx_pte->pte->ptn, ##__VA_ARGS__)

/* Type definitions */
struct cxip_rxc_ops {
	ssize_t (*recv_common)(struct cxip_rxc *rxc, void *buf, size_t len,
			       void *desc, fi_addr_t src_add, uint64_t tag,
			       uint64_t ignore, void *context, uint64_t flags,
			       bool tagged, struct cxip_cntr *comp_cntr);
	void (*progress)(struct cxip_rxc *rxc, bool internal);
	void (*recv_req_tgt_event)(struct cxip_req *req,
				   const union c_event *event);
	int (*cancel_msg_recv)(struct cxip_req *req);
	int (*ctrl_msg_cb)(struct cxip_ctrl_req *req,
			   const union c_event *event);
	void (*init_struct)(struct cxip_rxc *rxc, struct cxip_ep_obj *ep_obj);
	void (*fini_struct)(struct cxip_rxc *rxc);
	void (*cleanup)(struct cxip_rxc *rxc);
	int (*msg_init)(struct cxip_rxc *rxc);
	int (*msg_fini)(struct cxip_rxc *rxc);
};

struct cxip_rxc {
	void *context;
	uint32_t protocol;

	struct fi_rx_attr attr;
	bool selective_completion;
	bool hmem;
	bool trunc_ok;
	bool sw_ep_only;
	bool msg_offload;
	uint8_t pid_bits;		// Zero without SEP
	uint8_t recv_ptl_idx;

	enum cxip_rxc_state state;

	/* Reverse link to EP object that owns this context */
	struct cxip_ep_obj *ep_obj;

	struct cxip_cq *recv_cq;
	struct cxip_cntr *recv_cntr;

	struct cxip_rxc_ops ops;

	struct cxip_domain *domain;

	/* RXC receive portal table, event queue and hardware
	 * command queue.
	 */
	struct cxip_evtq rx_evtq;
	struct cxip_pte *rx_pte;
	struct cxip_cmdq *rx_cmdq;
	int orx_reqs;

	/* If FI_MULTI_RECV is supported, minimum receive size required
	 * for buffers posted.
	 */
	size_t min_multi_recv;

	/* If TX events are required by specialization, the maximum
	 * credits that can be used.
	 */
	int32_t max_tx;
	unsigned int recv_appends;

	struct cxip_msg_counters cntrs;
};

struct cxip_rxc_hpc {
	/* Must be first */
	struct cxip_rxc base;

	int max_eager_size;
	uint64_t rget_align_mask;

	/* Window when FI_CLAIM mutual exclusive access is required */
	bool hw_claim_in_progress;

	int sw_ux_list_len;
	int sw_pending_ux_list_len;

	/* Number of unexpected list entries in HW. */
	ofi_atomic32_t orx_hw_ule_cnt;

	/* RX context transmit queue is separated into two logical
	 * queues, one used for rendezvous get initiation and one
	 * used for notifications. Depending on the messaging protocols
	 * and traffic classes in use, the two logical queues could
	 * point to the same hardware queue or be distinct.
	 */
	struct cxip_cmdq *tx_rget_cmdq;
	struct cxip_cmdq *tx_cmdq;
	ofi_atomic32_t orx_tx_reqs;

	/* Software receive queue. User posted requests are queued here instead
	 * of on hardware if the RXC is in software endpoint mode.
	 */
	struct dlist_entry sw_recv_queue;

	/* Defer events to wait for both put and put overflow */
	struct def_event_ht deferred_events;

	/* Unexpected message handling */
	struct cxip_ptelist_bufpool *req_list_bufpool;
	struct cxip_ptelist_bufpool *oflow_list_bufpool;

	enum cxip_rxc_state prev_state;
	enum cxip_rxc_state new_state;
	enum c_sc_reason fc_reason;

	/* RXC drop count used for FC accounting. */
	int drop_count;

	/* Array of 8-byte of unexpected headers remote offsets. */
	uint64_t *ule_offsets;
	unsigned int num_ule_offsets;

	/* Current remote offset to be processed. Incremented after processing
	 * a search and delete put event.
	 */
	unsigned int cur_ule_offsets;

	struct dlist_entry fc_drops;
	struct dlist_entry replay_queue;
	struct dlist_entry sw_ux_list;
	struct dlist_entry sw_pending_ux_list;

	/* Flow control/software state change metrics */
	int num_fc_eq_full;
	int num_fc_no_match;
	int num_fc_unexp;
	int num_fc_append_fail;
	int num_fc_req_full;
	int num_sc_nic_hw2sw_append_fail;
	int num_sc_nic_hw2sw_unexp;
};

struct cxip_rxc_rnr {
	/* Must be first */
	struct cxip_rxc base;

	bool hybrid_mr_desc;
	/* Used when success events are not required */
	struct cxip_req *req_selective_comp_msg;
	struct cxip_req *req_selective_comp_tag;
};

/* Function declarations */
void cxip_rxc_req_fini(struct cxip_rxc *rxc);

int cxip_rxc_oflow_init(struct cxip_rxc *rxc);

void cxip_rxc_oflow_fini(struct cxip_rxc *rxc);

int cxip_rxc_msg_enable(struct cxip_rxc_hpc *rxc, uint32_t drop_count);

struct cxip_rxc *cxip_rxc_calloc(struct cxip_ep_obj *ep_obj, void *context);

void cxip_rxc_free(struct cxip_rxc *rxc);

int cxip_rxc_enable(struct cxip_rxc *rxc);

void cxip_rxc_disable(struct cxip_rxc *rxc);

void cxip_rxc_struct_init(struct cxip_rxc *rxc, const struct fi_rx_attr *attr,
			  void *context);

void cxip_rxc_recv_req_cleanup(struct cxip_rxc *rxc);

int cxip_rxc_emit_dma(struct cxip_rxc_hpc *rxc, struct cxip_cmdq *cmdq,
		      uint16_t vni, enum cxi_traffic_class tc,
		      enum cxi_traffic_class_type tc_type,
		      struct c_full_dma_cmd *dma, uint64_t flags);

int cxip_rxc_emit_idc_msg(struct cxip_rxc_hpc *rxc, struct cxip_cmdq *cmdq,
			  uint16_t vni, enum cxi_traffic_class tc,
			  enum cxi_traffic_class_type tc_type,
			  const struct c_cstate_cmd *c_state,
			  const struct c_idc_msg_hdr *msg, const void *buf,
			  size_t len, uint64_t flags);

void cxip_rxc_record_req_stat(struct cxip_rxc *rxc, enum c_ptl_list list,
			      size_t rlength, struct cxip_req *req);

#endif /* _CXIP_RXC_H_ */
