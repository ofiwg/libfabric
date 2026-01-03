/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_MSG_H_
#define _CXIP_MSG_H_

#include <ofi_atom.h>
#include <ofi_list.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Forward declarations */
struct cxip_md;
struct cxip_pte;
struct cxip_req;
struct cxip_rxc;
struct cxip_rxc_hpc;
struct cxip_txc;

/* Macros */
#define CXIP_MSG_ORDER                                                       \
	(FI_ORDER_SAS | FI_ORDER_WAW | FI_ORDER_RMA_WAW | FI_ORDER_RMA_RAR | \
	 FI_ORDER_ATOMIC_WAW | FI_ORDER_ATOMIC_WAR | FI_ORDER_ATOMIC_RAW |   \
	 FI_ORDER_ATOMIC_RAR)

#define CXIP_TAG_WIDTH 48

#define CXIP_TAG_MASK ((1UL << CXIP_TAG_WIDTH) - 1)

/* Type definitions */
union cxip_match_bits {
	struct {
		uint64_t tag : CXIP_TAG_WIDTH; /* User tag value */
		uint64_t tx_id : CXIP_TX_ID_WIDTH; /* Prov. tracked ID */
		uint64_t cq_data : 1; /* Header data is valid */
		uint64_t tagged : 1; /* Tagged API */
		uint64_t match_comp : 1; /* Notify initiator on match */
		uint64_t rdzv_done : 1; /* Notify initiator when rdzv done */
		uint64_t le_type : 1;
	};
	/* Rendezvous protocol request, overloads match_comp and rdzv_done
	 * to specify requested protocol.
	 */
	struct {
		uint64_t pad0 : 61;
		uint64_t rdzv_proto : 2;
		uint64_t pad1 : 1;
	};
	/* Split TX ID for rendezvous operations. */
	struct {
		uint64_t pad2 : (CXIP_TAG_WIDTH - 1); /* User tag value */
		uint64_t coll_get : 1; /* leaf rdma get */
		uint64_t rdzv_id_hi : CXIP_RDZV_ID_HIGH_WIDTH;
		uint64_t rdzv_lac : 4; /* Rendezvous Get LAC */
	};
	struct {
		uint64_t rdzv_id_lo : CXIP_RDZV_ID_CMD_WIDTH;
	};
	/* Client/Server messaging match bits */
	struct {
		uint64_t rnr_tag : CXIP_CS_TAG_WIDTH; /* User tag value */
		uint64_t rnr_rsvd : 6; /* Unused, set to 0 */
		uint64_t rnr_cq_data : 1; /* Header data valid */
		uint64_t rnr_tagged : 1; /* Tagged API */
		uint64_t rnr_vni : CXIP_VNI_WIDTH; /* Source VNI */
	};
	/* Control LE match bit format for notify/resume */
	struct {
		uint64_t txc_id : 8;
		uint64_t rxc_id : 8;
		uint64_t drops : 16;
		uint64_t pad3 : 29;
		uint64_t ctrl_msg_type : 2;
		uint64_t ctrl_le_type : 1;
	};
	/* Control LE match bit format for zbcollectives */
	struct {
		uint64_t zb_data : 61;
		uint64_t zb_pad : 3;
		/* shares ctrl_le_type == CXIP_CTRL_LE_TYPE_CTRL_MSG
		 * shares ctrl_msg_type == CXIP_CTRL_MSG_ZB_BCAST
		 */
	};
	/* Control LE match bit format for cached MR */
	struct {
		uint64_t mr_lac : 3;
		uint64_t mr_lac_off : 58;
		uint64_t mr_opt : 1;
		uint64_t mr_cached : 1;
		uint64_t mr_unused : 1;
		/* shares ctrl_le_type == CXIP_CTRL_LE_TYPE_MR */
	};
	struct {
		uint64_t mr_key : 61;
		uint64_t mr_pad : 3;
		/* shares mr_opt
		 * shares mr_cached == 0
		 * shares ctrl_le_type == CXIP_CTRL_LE_TYPE_MR
		 */
	};
	struct {
		uint64_t unused2 : 63;
		uint64_t is_prov : 1;
		/* Indicates provider generated key and shares ctrl_le_type ==
		 * CXIP_CTRL_LE_TYPE_MR so it must be cleared before matching.
		 */
	};
	uint64_t raw;
};

struct cxip_ux_dump_state {
	bool done;

	size_t max_count; /* Number entries/src_addr provided */
	size_t ret_count; /* Number of UX entries returned */
	size_t ux_count; /* Total UX entries available */

	struct fi_cq_tagged_entry *entry;
	fi_addr_t *src_addr;
};

struct cxip_ux_send {
	struct dlist_entry rxc_entry;
	struct cxip_req *req;
	struct cxip_rxc *rxc;
	struct fi_peer_rx_entry *rx_entry;
	union c_event put_ev;
	bool claimed; /* Reserved with FI_PEEK | FI_CLAIM */
};

struct cxip_msg_counters {
	/* Histogram counting the number of messages based on priority, buffer
	 * type (HMEM), and message size.
	 */
	ofi_atomic32_t msg_count[CXIP_LIST_COUNTS][OFI_HMEM_MAX]
				[CXIP_COUNTER_BUCKETS];
};

/* Function declarations */
int cxip_recv_ux_sw_matcher(struct cxip_ux_send *ux);

int cxip_recv_req_sw_matcher(struct cxip_req *req);

int cxip_recv_cancel(struct cxip_req *req);

void cxip_recv_pte_cb(struct cxip_pte *pte, const union c_event *event);

fi_addr_t cxip_recv_req_src_addr(struct cxip_rxc *rxc, uint32_t init,
				 uint16_t vni, bool force);

int cxip_recv_req_alloc(struct cxip_rxc *rxc, void *buf, size_t len,
			struct cxip_md *md, struct cxip_req **cxip_req,
			int (*recv_cb)(struct cxip_req *req,
				       const union c_event *event));

void cxip_recv_req_free(struct cxip_req *req);

void cxip_recv_req_report(struct cxip_req *req);

void cxip_recv_req_peek_complete(struct cxip_req *req,
				 struct cxip_ux_send *ux_send);

struct cxip_req *cxip_mrecv_req_dup(struct cxip_req *mrecv_req);

int cxip_complete_put(struct cxip_req *req, const union c_event *event);

int cxip_recv_pending_ptlte_disable(struct cxip_rxc *rxc, bool check_fc);

int cxip_flush_appends(struct cxip_rxc_hpc *rxc,
		       int (*flush_cb)(struct cxip_req *req,
				       const union c_event *event));

int cxip_recv_req_dropped(struct cxip_req *req);

bool tag_match(uint64_t init_mb, uint64_t mb, uint64_t ib);

bool init_match(struct cxip_rxc *rxc, uint32_t init, uint32_t match_id);

/*
 * cxip_ux_bloom_hash() - Compute bloom filter hash for UX message.
 *
 * Creates a hash value from the tag and tagged flag. This is used
 * for bloom filter insert/lookup to accelerate UX matching.
 *
 * The hash combines:
 * - tagged flag (1 bit) in high position
 * - tag value (48 bits)
 *
 * Note: We don't include initiator because receives often use
 * FI_ADDR_UNSPEC. The bloom filter is just for quick rejection.
 */
static inline uint64_t cxip_ux_bloom_hash(bool tagged, uint64_t tag)
{
	/* Put tagged flag in bit 63, tag in bits 0-47 */
	return ((uint64_t)tagged << 63) | (tag & CXIP_TAG_MASK);
}

uint32_t cxip_msg_match_id(struct cxip_txc *txc);

void cxip_report_send_completion(struct cxip_req *req, bool sw_cntr);

bool cxip_send_eager_idc(struct cxip_req *req);

void cxip_send_buf_fini(struct cxip_req *req);

int cxip_send_buf_init(struct cxip_req *req);

#endif /* _CXIP_MSG_H_ */
