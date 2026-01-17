/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_REQ_H_
#define _CXIP_REQ_H_

#include <ofi_list.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_cntr;
struct cxip_coll_buf;
struct cxip_coll_pte;
struct cxip_coll_reduction;
struct cxip_cq;
struct cxip_evtq;
struct cxip_fc_peer;
struct cxip_md;
struct cxip_rxc;
struct cxip_rxc_hpc;
struct cxip_rxc_rnr;
struct cxip_txc;
struct cxip_txc_hpc;
struct cxip_txc_rnr;
struct cxip_ux_dump_state;

/* Macros */
#define CXIP_REQ_CLEANUP_TO 3000

/* Type definitions */
struct cxip_req_rma {
	struct cxip_txc *txc;
	struct cxip_md *local_md; // RMA target buffer
	void *ibuf;
	struct cxip_cntr *cntr;
	/* collectives leaf_rdma_get_callback context data */
	struct cxip_coll_reduction *reduction;
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
	struct cxip_cntr *cntr;
};

struct cxip_req_recv {
	/* Receive parameters */
	struct dlist_entry rxc_entry;
	union {
		struct cxip_rxc *rxc;
		struct cxip_rxc_hpc *rxc_hpc;
		struct cxip_rxc_rnr *rxc_rnr;
	};

	struct cxip_cntr *cntr;
	void *recv_buf; // local receive buffer
	struct cxip_md *recv_md; // local receive MD
	bool hybrid_md; // True if MD was provided
	bool success_disable;
	uint32_t ulen; // User buffer length
	bool tagged;
	uint64_t tag;
	uint64_t ignore;
	uint32_t match_id;
	uint64_t flags;

	/* FI_CLAIM work around to hold UX remote offsets for duration of
	 * H/W UX entry matching and deletion. Array of 8-byte unexpected
	 * headers remote offsets, and current remote offset used when
	 * processing search results to match remote offsets.
	 */
	uint64_t *ule_offsets;
	uint64_t ule_offset;
	unsigned int num_ule_offsets;
	unsigned int cur_ule_offsets;
	bool offset_found;

	/* UX list dump state */
	struct cxip_ux_dump_state *ux_dump;

	/* Control info */
	int rc; // DMA return code
	uint32_t rlen; // Send length
	uint64_t oflow_start; // Overflow buffer address
	uint16_t vni; // VNI operation came in on
	uint32_t initiator; // DMA initiator address
	uint32_t rdzv_id; // DMA initiator rendezvous ID
	uint8_t rdzv_lac; // Rendezvous source LAC
	bool done_notify; // Must send done notification
	enum cxip_rdzv_proto rdzv_proto;
	int rdzv_events; // Processed rdzv event count
	enum c_event_type rdzv_event_types[4];
	uint32_t rdzv_initiator; // Rendezvous initiator used for mrecvs
	uint32_t rget_nic;
	uint32_t rget_pid;
	int multirecv_inflight; // SW EP Multi-receives in progress
	bool canceled; // Request canceled?
	bool unlinked;
	bool multi_recv;
	bool tgt_event;
	uint64_t start_offset;
	uint64_t mrecv_bytes;
	uint64_t mrecv_unlink_bytes;
	bool auto_unlinked;
	bool hw_offloaded;
	struct cxip_req *parent;
	struct dlist_entry children;
	uint64_t src_offset;
	uint16_t rdzv_mlen;
};

struct cxip_req_send {
	/* Send parameters */
	union {
		struct cxip_txc *txc;
		struct cxip_txc_hpc *txc_hpc;
		struct cxip_txc_rnr *txc_rnr;
	};
	struct cxip_cntr *cntr;
	const void *buf; // local send buffer
	size_t len; // request length
	struct cxip_md *send_md; // send buffer memory descriptor
	struct cxip_addr caddr;
	fi_addr_t dest_addr;
	bool tagged;
	bool hybrid_md;
	bool success_disable;
	uint32_t tclass;
	uint64_t tag;
	uint64_t data;
	uint64_t flags;
	void *ibuf;

	/* Control info */
	struct dlist_entry txc_entry;
	struct cxip_fc_peer *fc_peer;
	union {
		int rdzv_id; // SW RDZV ID for long messages
		int tx_id;
	};
	int rc; // DMA return code
	int rdzv_send_events; // Processed event count
	uint64_t max_rnr_time;
	uint64_t retry_rnr_time;
	struct dlist_entry rnr_entry;
	int retries;
	bool canceled;
};

struct cxip_req_rdzv_src {
	struct dlist_entry list;
	struct cxip_txc *txc;
	uint32_t lac;
	int rc;
};

struct cxip_req_search {
	struct cxip_rxc_hpc *rxc;
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

struct cxip_req {
	/* Control info */
	struct dlist_entry evtq_entry;
	void *req_ctx;
	struct cxip_cq *cq; // request CQ
	struct cxip_evtq *evtq; // request event queue
	int req_id; // fast lookup in index table
	int (*cb)(struct cxip_req *req, const union c_event *evt);
	// completion event callback
	bool discard;

	/* Triggered related fields. */
	bool triggered;
	uint64_t trig_thresh;
	struct cxip_cntr *trig_cntr;

	struct fi_peer_rx_entry *rx_entry;

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
		struct cxip_req_recv recv;
		struct cxip_req_send send;
		struct cxip_req_rdzv_src rdzv_src;
		struct cxip_req_search search;
		struct cxip_req_coll coll;
	};
};

#endif /* _CXIP_REQ_H_ */
