/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _RXR_H_
#define _RXR_H_

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
#include <ofi_iov.h>
#include <ofi_proto.h>
#include <ofi_enosys.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_util.h>
#include <ofi_tree.h>
#include <uthash.h>
#include <ofi_recvwin.h>
#include <ofi_perf.h>

#define RXR_MAJOR_VERSION	(2)
#define RXR_MINOR_VERSION	(0)
#define RXR_PROTOCOL_VERSION	(2)
#define RXR_FI_VERSION		FI_VERSION(1, 8)

#define RXR_IOV_LIMIT		(4)

#ifdef ENABLE_EFA_POISONING
extern const uint32_t rxr_poison_value;
#endif

/*
 * Set alignment to x86 cache line size.
 */
#define RXR_BUF_POOL_ALIGNMENT	(64)

/*
 * will add following parameters to env variable for tuning
 */
#define RXR_RECVWIN_SIZE		(16384)
#define RXR_DEF_CQ_SIZE			(8192)
#define RXR_REMOTE_CQ_DATA_LEN		(8)
#define RXR_MIN_AV_SIZE			(8192)
/* maximum timeout for RNR backoff (microseconds) */
#define RXR_DEF_RNR_MAX_TIMEOUT		(1000000)
/* bounds for random RNR backoff timeout */
#define RXR_RAND_MIN_TIMEOUT		(40)
#define RXR_RAND_MAX_TIMEOUT		(120)
#define RXR_DEF_MAX_RX_WINDOW		(16)
/*
 * maximum time (microseconds) we will allow available_bufs for large msgs to
 * be exhausted
 */
#define RXR_AVAILABLE_DATA_BUFS_TIMEOUT	(5000000)

#if ENABLE_DEBUG
#define RXR_TX_PKT_DBG_SIZE	(16384)
#define RXR_RX_PKT_DBG_SIZE	(16384)
#endif

/*
 * Based on size of tx_id and rx_id in headers, can be arbitrary once those are
 * removed.
 */
#define RXR_MAX_RX_QUEUE_SIZE (UINT32_MAX)
#define RXR_MAX_TX_QUEUE_SIZE (UINT32_MAX)

/*
 * The maximum supported source address length in bytes
 */
#define RXR_MAX_NAME_LENGTH	(32)

/*
 * RxR specific flags that are sent over the wire.
 */
#define RXR_TAGGED		BIT_ULL(0)
#define RXR_REMOTE_CQ_DATA	BIT_ULL(1)
#define RXR_REMOTE_SRC_ADDR	BIT_ULL(2)
/*
 * TODO: In future we will send RECV_CANCEL signal to sender,
 * to stop transmitting large message, this flag is also
 * used for fi_discard which has similar behavior.
 */
#define RXR_RECV_CANCEL		BIT_ULL(3)

/*
 * Flags to tell if the rx_entry is tracking FI_MULTI_RECV buffers
 */
#define RXR_MULTI_RECV_POSTED	BIT_ULL(4)
#define RXR_MULTI_RECV_CONSUMER	BIT_ULL(5)

/*
 * for RMA
 */
#define RXR_WRITE		(1 << 6)
#define RXR_READ_REQ		(1 << 7)
#define RXR_READ_DATA		(1 << 8)

/*
 * OFI flags
 * The 64-bit flag field is used as follows:
 * 1-grow up    common (usable with multiple operations)
 * 59-grow down operation specific (used for single call/class)
 * 60 - 63      provider specific
 */
#define RXR_NO_COMPLETION	BIT_ULL(60)

/*
 * RM flags
 */
#define RXR_RM_TX_CQ_FULL	BIT_ULL(0)
#define RXR_RM_RX_CQ_FULL	BIT_ULL(1)

#define RXR_MTU_MAX_LIMIT	BIT_ULL(15)

extern struct fi_provider *lower_efa_prov;
extern struct fi_provider rxr_prov;
extern struct fi_info rxr_info;
extern struct rxr_env rxr_env;
extern struct fi_fabric_attr rxr_fabric_attr;
extern struct util_prov rxr_util_prov;

struct rxr_env {
	int rx_window_size;
	int tx_queue_size;
	int enable_sas_ordering;
	int recvwin_size;
	int cq_size;
	size_t max_memcpy_size;
	size_t mtu_size;
	size_t tx_size;
	size_t rx_size;
	size_t tx_iov_limit;
	size_t rx_iov_limit;
	int rx_copy_unexp;
	int rx_copy_ooo;
	int max_timeout;
	int timeout_interval;
};

enum rxr_pkt_type {
	RXR_RTS_PKT = 1,
	RXR_CONNACK_PKT,
	/* Large message types */
	RXR_CTS_PKT,
	RXR_DATA_PKT,
	RXR_READRSP_PKT,
};

/* pkt_entry types for rx pkts */
enum rxr_pkt_entry_type {
	RXR_PKT_ENTRY_POSTED = 1,   /* entries that are posted to the core */
	RXR_PKT_ENTRY_UNEXP,        /* entries used to stage unexpected msgs */
	RXR_PKT_ENTRY_OOO	    /* entries used to stage out-of-order RTS */
};

/* pkt_entry state for retransmit tracking */
enum rxr_pkt_entry_state {
	RXR_PKT_ENTRY_FREE = 0,
	RXR_PKT_ENTRY_IN_USE,
	RXR_PKT_ENTRY_RNR_RETRANSMIT,
};

enum rxr_x_entry_type {
	RXR_TX_ENTRY = 1,
	RXR_RX_ENTRY,
};

enum rxr_tx_comm_type {
	RXR_TX_FREE = 0,	/* tx_entry free state */
	RXR_TX_RTS,		/* tx_entry sending RTS message */
	RXR_TX_SEND,		/* tx_entry sending data in progress */
	RXR_TX_QUEUED_RTS,	/* tx_entry was unable to send RTS */
	RXR_TX_QUEUED_RTS_RNR,  /* tx_entry RNR sending RTS packet */
	RXR_TX_QUEUED_DATA_RNR,	/* tx_entry RNR sending data packets */
	RXR_TX_SENT_READRSP,	/* tx_entry (on remote EP) sent
				 * read response (FI_READ only)
				 */
	RXR_TX_QUEUED_READRSP, /* tx_entry (on remote EP) was
				* unable to send read response
				* (FI_READ only)
				*/
	RXR_TX_WAIT_READ_FINISH, /* tx_entry (on initiating EP) wait
				  * for rx_entry to finish receiving
				  * (FI_READ only)
				  */
};

enum rxr_rx_comm_type {
	RXR_RX_FREE = 0,	/* rx_entry free state */
	RXR_RX_INIT,		/* rx_entry ready to recv RTS */
	RXR_RX_UNEXP,		/* rx_entry unexp msg waiting for post recv */
	RXR_RX_MATCHED,		/* rx_entry matched with RTS msg */
	RXR_RX_RECV,		/* rx_entry large msg recv data pkts */
	RXR_RX_QUEUED_CTS,	/* rx_entry was unable to send CTS */
	RXR_RX_QUEUED_CTS_RNR,	/* rx_entry RNR sending CTS */
	RXR_RX_WAIT_READ_FINISH, /* rx_entry wait for send to finish, FI_READ */
};

enum rxr_peer_state {
	RXR_PEER_FREE = 0,	/* rxr_peer free state */
	RXR_PEER_CONNREQ,	/* RTS with endpoint address sent to peer */
	RXR_PEER_ACKED,		/* RXR_CONNACK_PKT received from peer */
};

/* peer is in backoff, not allowed to send */
#define RXR_PEER_IN_BACKOFF (1ULL << 0)
/* peer backoff was increased during this loop of the progress engine */
#define RXR_PEER_BACKED_OFF (1ULL << 1)

struct rxr_fabric {
	struct util_fabric util_fabric;
	struct fid_fabric *lower_fabric;
#ifdef RXR_PERF_ENABLED
	struct ofi_perfset perf_set;
#endif
};

struct rxr_mr {
	struct fid_mr mr_fid;
	struct fid_mr *msg_mr;
	struct rxr_domain *domain;
};

struct rxr_av_entry {
	uint8_t addr[RXR_MAX_NAME_LENGTH];
	fi_addr_t rdm_addr;
	UT_hash_handle hh;
};

struct rxr_av {
	struct util_av util_av;
	struct fid_av *rdm_av;
	struct rxr_av_entry *av_map;

	int rdm_av_used;
	size_t rdm_addrlen;
};

struct rxr_peer {
	struct rxr_robuf *robuf;	/* tracks expected msg_id on rx */
	uint32_t next_msg_id;		/* sender's view of msg_id */
	enum rxr_peer_state state;	/* state of CM protocol with peer */
	unsigned int rnr_state;		/* tracks RNR backoff for peer */
	uint64_t rnr_ts;		/* timestamp for RNR backoff tracking */
	int rnr_queued_pkt_cnt;		/* queued RNR packet count */
	int timeout_interval;		/* initial RNR timeout value */
	int rnr_timeout_exp;		/* RNR timeout exponentation calc val */
	struct dlist_entry rnr_entry;	/* linked to rxr_ep peer_backoff_list */
	struct dlist_entry entry;	/* linked to rxr_ep peer_list */
};

struct rxr_rx_entry {
	/* Must remain at the top */
	enum rxr_x_entry_type type;

	fi_addr_t addr;

	/*
	 * freestack ids used to lookup rx_entry during pkt recv
	 */
	uint32_t tx_id;
	uint32_t rx_id;

	/*
	 * The following two varibales are for emulated RMA fi_read only
	 */
	uint32_t rma_loc_tx_id;
	uint32_t rma_initiator_rx_id;

	uint32_t msg_id;

	uint64_t tag;
	uint64_t ignore;

	uint64_t bytes_done;
	int64_t window;

	uint64_t total_len;

	enum rxr_rx_comm_type state;

	uint64_t fi_flags;
	uint16_t rxr_flags;

	size_t iov_count;
	struct iovec iov[RXR_IOV_LIMIT];

	struct fi_cq_tagged_entry cq_entry;

	/* entry is linked with rx entry lists in rxr_ep */
	struct dlist_entry entry;

	/* queued_entry is linked with rx_queued_ctrl_list in rxr_ep */
	struct dlist_entry queued_entry;

	/* Queued packets due to TX queue full or RNR backoff */
	struct dlist_entry queued_pkts;

	/*
	 * A list of rx_entries tracking FI_MULTI_RECV buffers. An rx_entry of
	 * type RXR_MULTI_RECV_POSTED that was created when the multi-recv
	 * buffer was posted is the list head, and the rx_entries of type
	 * RXR_MULTI_RECV_CONSUMER get added to the list as they consume the
	 * buffer.
	 */
	struct dlist_entry multi_recv_consumers;
	struct dlist_entry multi_recv_entry;
	struct rxr_rx_entry *master_entry;

	struct rxr_pkt_entry *unexp_rts_pkt;

#if ENABLE_DEBUG
	/* linked with rx_pending_list in rxr_ep */
	struct dlist_entry rx_pending_entry;
	/* linked with rx_entry_list in rxr_ep */
	struct dlist_entry rx_entry_entry;
#endif
};

struct rxr_tx_entry {
	/* Must remain at the top */
	enum rxr_x_entry_type type;

	fi_addr_t addr;

	/*
	 * freestack ids used to lookup tx_entry during ctrl pkt recv
	 */
	uint32_t tx_id;
	uint32_t rx_id;

	uint32_t msg_id;

	uint64_t tag;

	uint64_t bytes_acked;
	uint64_t bytes_sent;
	int64_t window;

	uint64_t total_len;

	enum rxr_tx_comm_type state;

	uint64_t fi_flags;
	uint64_t send_flags;
	size_t iov_count;
	size_t iov_index;
	size_t iov_offset;
	struct iovec iov[RXR_IOV_LIMIT];

	uint64_t rma_loc_rx_id;
	uint64_t rma_window;
	size_t rma_iov_count;
	struct fi_rma_iov rma_iov[RXR_IOV_LIMIT];

	/* Only used with mr threshold switch from memcpy */
	size_t iov_mr_start;
	struct fid_mr *mr[RXR_IOV_LIMIT];

	struct fi_cq_tagged_entry cq_entry;

	/* entry is linked with tx_pending_list in rxr_ep */
	struct dlist_entry entry;

	/* queued_entry is linked with tx_queued_ctrl_list in rxr_ep */
	struct dlist_entry queued_entry;

	/* Queued packets due to TX queue full or RNR backoff */
	struct dlist_entry queued_pkts;

#if ENABLE_DEBUG
	/* linked with tx_entry_list in rxr_ep */
	struct dlist_entry tx_entry_entry;
#endif
};

#define RXR_GET_X_ENTRY_TYPE(pkt_entry)	\
	(*((enum rxr_x_entry_type *)	\
	 ((unsigned char *)((pkt_entry)->x_entry))))

struct rxr_domain {
	struct util_domain util_domain;
	struct fid_domain *rdm_domain;

	size_t addrlen;
	uint8_t mr_local;
	uint64_t rdm_mode;
	int do_progress;
	size_t cq_size;
	enum fi_resource_mgmt resource_mgmt;
};

struct rxr_ep {
	struct util_ep util_ep;

	uint8_t core_addr[RXR_MAX_NAME_LENGTH];
	size_t core_addrlen;

	/* per-peer information */
	struct rxr_peer *peer;

	/* free stack for reorder buffer */
	struct rxr_robuf_fs *robuf_fs;

	/* core provider fid */
	struct fid_ep *rdm_ep;
	struct fid_cq *rdm_cq;

	/*
	 * RxR rx/tx queue sizes. These may be different from the core
	 * provider's rx/tx size and will either limit the number of possible
	 * receives/sends or allow queueing.
	 */
	size_t rx_size;
	size_t tx_size;
	size_t mtu_size;
	size_t rx_iov_limit;
	size_t tx_iov_limit;

	/* core's capabilities */
	uint64_t core_caps;

	/* rx/tx queue size of core provider */
	size_t core_rx_size;
	size_t max_outstanding_tx;
	size_t core_inject_size;
	size_t max_data_payload_size;

	/* Resource management flag */
	uint64_t rm_full;

	/* application's ordering requirements */
	uint64_t msg_order;
	/* core's supported tx/rx msg_order */
	uint64_t core_msg_order;

	/* tx iov limit of core provider */
	size_t core_iov_limit;

	/* threshold to release multi_recv buffer */
	size_t min_multi_recv_size;

	/* buffer pool for send & recv */
	struct ofi_bufpool *tx_pkt_pool;
	struct ofi_bufpool *rx_pkt_pool;

	/* staging area for unexpected and out-of-order packets */
	struct ofi_bufpool *rx_unexp_pkt_pool;
	struct ofi_bufpool *rx_ooo_pkt_pool;

#ifdef ENABLE_EFA_POISONING
	size_t tx_pkt_pool_entry_sz;
	size_t rx_pkt_pool_entry_sz;
#endif

	/* datastructure to maintain rxr send/recv states */
	struct ofi_bufpool *tx_entry_pool;
	struct ofi_bufpool *rx_entry_pool;
	/* datastructure to maintain read response */
	struct ofi_bufpool *readrsp_tx_entry_pool;

	/* rx_entries with recv buf */
	struct dlist_entry rx_list;
	/* rx_entries without recv buf (unexpected message) */
	struct dlist_entry rx_unexp_list;
	/* rx_entries with tagged recv buf */
	struct dlist_entry rx_tagged_list;
	/* rx_entries without tagged recv buf (unexpected message) */
	struct dlist_entry rx_unexp_tagged_list;
	/* list of pre-posted recv buffers */
	struct dlist_entry rx_posted_buf_list;
	/* tx entries with queued messages */
	struct dlist_entry tx_entry_queued_list;
	/* rx entries with queued messages */
	struct dlist_entry rx_entry_queued_list;
	/* tx_entries with data to be sent (large messages) */
	struct dlist_entry tx_pending_list;
	/* rxr_peer entries that are in backoff due to RNR */
	struct dlist_entry peer_backoff_list;
	/* rxr_peer entries with an allocated robuf */
	struct dlist_entry peer_list;

#if ENABLE_DEBUG
	/* rx_entries waiting for data to arrive (large messages) */
	struct dlist_entry rx_pending_list;
	/* count of rx_pending_list */
	size_t rx_pending;

	/* rx packets being processed or waiting to be processed */
	struct dlist_entry rx_pkt_list;

	/* tx packets waiting for send completion */
	struct dlist_entry tx_pkt_list;

	/* track allocated rx_entries and tx_entries for endpoint cleanup */
	struct dlist_entry rx_entry_list;
	struct dlist_entry tx_entry_list;

	size_t sends;
	size_t send_comps;
	size_t failed_send_comps;
	size_t recv_comps;
#endif

	/* number of posted buffers */
	size_t posted_bufs;
	size_t rx_bufs_to_post;
	/* number of buffers available for large messages */
	size_t available_data_bufs;
	/* Timestamp of when available_data_bufs was exhausted. */
	uint64_t available_data_bufs_ts;

	/* number of outstanding sends */
	size_t tx_pending;
};

#define rxr_rx_flags(rxr_ep) ((rxr_ep)->util_ep.rx_op_flags)
#define rxr_tx_flags(rxr_ep) ((rxr_ep)->util_ep.tx_op_flags)

/*
 * Packet fields common to all rxr packets. The other packet headers below must
 * be changed if this is updated.
 */
struct rxr_base_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
};

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_base_hdr) == 4, "rxr_base_hdr check");
#endif

/*
 * RTS packet structure: rts_hdr, cq_data (optional), src_addr(optional),  data.
 */
struct rxr_rts_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	/* TODO: need to add msg_id -> tx_id mapping to remove tx_id and pad */
	uint8_t pad[2];
	uint8_t addrlen;
	uint8_t rma_iov_count;
	uint32_t tx_id;
	uint32_t msg_id;
	uint64_t tag;
	uint64_t data_len;
}; /* 24 bytes without tx_id and padding for it */

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_rts_hdr) == 32, "rxr_rts_hdr check");
#endif

struct rxr_connack_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
}; /* 4 bytes */

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_base_hdr) == 4, "rxr_connack_hdr check");
#endif

struct rxr_cts_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint8_t pad[4];
	/* TODO: need to add msg_id -> tx_id/rx_id mapping */
	uint32_t tx_id;
	uint32_t rx_id;
	uint64_t window;
};

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_cts_hdr) == 24, "rxr_cts_hdr check");
#endif

struct rxr_data_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	/* TODO: need to add msg_id -> tx_id/rx_id mapping */
	uint32_t rx_id;
	uint64_t seg_size;
	uint64_t seg_offset;
};

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_data_hdr) == 24, "rxr_data_hdr check");
#endif

struct rxr_readrsp_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint8_t pad[4];
	uint32_t rx_id;
	uint32_t tx_id;
	uint64_t seg_size;
};

#if defined(static_assert) && defined(__X86_64__)
static_assert(sizeof(struct rxr_readrsp_hdr) == sizeof(struct rxr_data_hdr), "rxr_readrsp_hdr check");
#endif

/*
 * Control header without completion data. We will send more data with the RTS
 * packet if RXR_REMOTE_CQ_DATA is not set.
 */
struct rxr_ctrl_hdr {
	union {
		struct rxr_base_hdr base_hdr;
		struct rxr_rts_hdr rts_hdr;
		struct rxr_connack_hdr connack_hdr;
		struct rxr_cts_hdr cts_hdr;
	};
};

/*
 * Control header with completion data. CQ data length is static.
 */
#define RXR_CQ_DATA_SIZE (8)
struct rxr_ctrl_cq_hdr {
	union {
		struct rxr_base_hdr base_hdr;
		struct rxr_rts_hdr rts_hdr;
		struct rxr_connack_hdr connack_hdr;
		struct rxr_cts_hdr cts_hdr;
	};
	uint64_t cq_data;
};

/*
 * There are three packet types:
 * - Control packet with completion queue data
 * - Control packet without completion queue data
 * - Data packet
 *
 * All start with rxr_base_hdr so it is safe to cast between them to check
 * values in that structure.
 */
struct rxr_ctrl_cq_pkt {
	struct rxr_ctrl_cq_hdr hdr;
	char data[];
};

struct rxr_ctrl_pkt {
	struct rxr_ctrl_hdr hdr;
	char data[];
};

struct rxr_data_pkt {
	struct rxr_data_hdr hdr;
	char data[];
};

struct rxr_readrsp_pkt {
	struct rxr_readrsp_hdr hdr;
	char data[];
};

struct rxr_pkt_entry {
	/* for rx/tx_entry queued_pkts list */
	struct dlist_entry entry;
#if ENABLE_DEBUG
	/* for tx/rx debug list or posted buf list */
	struct dlist_entry dbg_entry;
#endif
	void *x_entry; /* pointer to rxr rx/tx entry */
	size_t pkt_size;
	struct fid_mr *mr;
	fi_addr_t addr;
	void *pkt; /* rxr_ctrl_*_pkt, or rxr_data_pkt */
	enum rxr_pkt_entry_type type;
	enum rxr_pkt_entry_state state;
#if ENABLE_DEBUG
/* pad to cache line size of 64 bytes */
	uint8_t pad[48];
#endif
};

#if defined(static_assert) && defined(__X86_64__)
#if ENABLE_DEBUG
static_assert(sizeof(struct rxr_pkt_entry) == 128, "rxr_pkt_entry check");
#else
static_assert(sizeof(struct rxr_pkt_entry) == 64, "rxr_pkt_entry check");
#endif
#endif

OFI_DECL_RECVWIN_BUF(struct rxr_pkt_entry*, rxr_robuf);
DECLARE_FREESTACK(struct rxr_robuf, rxr_robuf_fs);

#define RXR_CTRL_HDR_SIZE		(sizeof(struct rxr_ctrl_cq_hdr))

#define RXR_CTRL_HDR_SIZE_NO_CQ		(sizeof(struct rxr_ctrl_hdr))

#define RXR_CONNACK_HDR_SIZE		(sizeof(struct rxr_connack_hdr))

#define RXR_CTS_HDR_SIZE		(sizeof(struct rxr_cts_hdr))

#define RXR_DATA_HDR_SIZE		(sizeof(struct rxr_data_hdr))

#define RXR_READRSP_HDR_SIZE	(sizeof(struct rxr_readrsp_hdr))

static inline struct rxr_peer *rxr_ep_get_peer(struct rxr_ep *ep,
					       fi_addr_t addr)
{
	return &ep->peer[addr];
}

struct rxr_rx_entry *rxr_ep_get_rx_entry(struct rxr_ep *ep,
					 const struct iovec *iov,
					 size_t iov_count, uint64_t tag,
					 uint64_t ignore, void *context,
					 fi_addr_t addr, uint32_t op,
					 uint64_t flags);

struct rxr_rx_entry *rxr_ep_rx_entry_init(struct rxr_ep *ep,
					  struct rxr_rx_entry *rx_entry,
					  const struct iovec *iov,
					  size_t iov_count, uint64_t tag,
					  uint64_t ignore, void *context,
					  fi_addr_t addr, uint32_t op,
					  uint64_t flags);

void rxr_generic_tx_entry_init(struct rxr_tx_entry *tx_entry,
			       const struct iovec *iov,
			       size_t iov_count,
			       const struct fi_rma_iov *rma_iov,
			       size_t rma_iov_count,
			       fi_addr_t addr, uint64_t tag,
			       uint64_t data, void *context,
			       uint32_t op, uint64_t flags);

struct rxr_tx_entry *rxr_ep_tx_entry_init(struct rxr_ep *rxr_ep,
					  const struct iovec *iov,
					  size_t iov_count,
					  const struct fi_rma_iov *rma_iov,
					  size_t rma_iov_count,
					  fi_addr_t addr, uint64_t tag,
					  uint64_t data, void *context,
					  uint32_t op, uint64_t flags);

ssize_t rxr_tx(struct fid_ep *ep, const struct iovec *iov, size_t iov_count,
	       const struct fi_rma_iov *rma_iov, size_t rma_iov_count,
	       fi_addr_t addr, uint64_t tag, uint64_t data, void *context,
	       uint32_t op, uint64_t flags);

static inline void
rxr_copy_pkt_entry(struct rxr_ep *ep,
		   struct rxr_pkt_entry *dest,
		   struct rxr_pkt_entry *src,
		   enum rxr_pkt_entry_type type)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
	       "Copying packet (type %d) out of posted buffer\n", type);
	assert(src->type == RXR_PKT_ENTRY_POSTED);
	memcpy(dest, src, sizeof(struct rxr_pkt_entry));
	dest->pkt = (struct rxr_pkt *)((char *)dest + sizeof(*dest));
	memcpy(dest->pkt, src->pkt, ep->mtu_size);
	dest->type = type;
	dlist_init(&dest->entry);
#if ENABLE_DEBUG
	dlist_init(&dest->dbg_entry);
#endif
	dest->state = RXR_PKT_ENTRY_IN_USE;
}

static inline struct rxr_pkt_entry*
rxr_get_pkt_entry(struct rxr_ep *ep, struct ofi_bufpool *pkt_pool)
{
	struct rxr_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ofi_buf_alloc_ex(pkt_pool, &mr);
	if (!pkt_entry)
		return NULL;
#ifdef ENABLE_EFA_POISONING
	memset(pkt_entry, 0, sizeof(*pkt_entry));
#endif
	dlist_init(&pkt_entry->entry);
#if ENABLE_DEBUG
	dlist_init(&pkt_entry->dbg_entry);
#endif
	pkt_entry->mr = (struct fid_mr *)mr;
	pkt_entry->pkt = (struct rxr_pkt *)((char *)pkt_entry +
			  sizeof(*pkt_entry));
#ifdef ENABLE_EFA_POISONING
	memset(pkt_entry->pkt, 0, ep->mtu_size);
#endif
	pkt_entry->state = RXR_PKT_ENTRY_IN_USE;

	return pkt_entry;
}

#ifdef ENABLE_EFA_POISONING
static inline void rxr_poison_mem_region(uint32_t *ptr, size_t size)
{
	int i;

	for (i = 0; i < size / sizeof(rxr_poison_value); i++)
		memcpy(ptr + i, &rxr_poison_value, sizeof(rxr_poison_value));
}
#endif

static inline void rxr_release_tx_pkt_entry(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt)
{
	struct rxr_peer *peer;

#if ENABLE_DEBUG
	dlist_remove(&pkt->dbg_entry);
#endif
	/*
	 * Decrement rnr_queued_pkts counter and reset backoff for this peer if
	 * we get a send completion for a retransmitted packet.
	 */
	if (OFI_UNLIKELY(pkt->state == RXR_PKT_ENTRY_RNR_RETRANSMIT)) {
		peer = rxr_ep_get_peer(ep, pkt->addr);
		peer->rnr_queued_pkt_cnt--;
		peer->timeout_interval = 0;
		peer->rnr_timeout_exp = 0;
		if (peer->rnr_state & RXR_PEER_IN_BACKOFF)
			dlist_remove(&peer->rnr_entry);
		peer->rnr_state = 0;
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "reset backoff timer for peer: %" PRIu64 "\n",
		       pkt->addr);
	}
#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region((uint32_t *)pkt, ep->tx_pkt_pool_entry_sz);
#endif
	pkt->state = RXR_PKT_ENTRY_FREE;
	ofi_buf_free(pkt);
}

static inline void rxr_release_rx_pkt_entry(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt)
{
#if ENABLE_DEBUG
	dlist_remove(&pkt->dbg_entry);
#endif
#ifdef ENABLE_EFA_POISONING
	/* the same pool size is used for all types of rx pkt_entries */
	rxr_poison_mem_region((uint32_t *)pkt, ep->rx_pkt_pool_entry_sz);
#endif
	pkt->state = RXR_PKT_ENTRY_FREE;
	ofi_buf_free(pkt);
}

static inline void rxr_release_tx_entry(struct rxr_ep *ep,
					struct rxr_tx_entry *tx_entry)
{
#if ENABLE_DEBUG
	dlist_remove(&tx_entry->tx_entry_entry);
#endif
	assert(dlist_empty(&tx_entry->queued_pkts));
#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region((uint32_t *)tx_entry,
			      sizeof(struct rxr_tx_entry));
#endif
	tx_entry->state = RXR_TX_FREE;
	tx_entry->msg_id = ~0;
	ofi_buf_free(tx_entry);
}

static inline void rxr_release_rx_entry(struct rxr_ep *ep,
					struct rxr_rx_entry *rx_entry)
{
#if ENABLE_DEBUG
	dlist_remove(&rx_entry->rx_entry_entry);
#endif
	assert(dlist_empty(&rx_entry->queued_pkts));
#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region((uint32_t *)rx_entry,
			      sizeof(struct rxr_rx_entry));
#endif
	rx_entry->state = RXR_RX_FREE;
	rx_entry->msg_id = ~0;
	ofi_buf_free(rx_entry);
}

static inline void *rxr_pkt_start(struct rxr_pkt_entry *pkt_entry)
{
	return (void *)((char *)pkt_entry + sizeof(*pkt_entry));
}

static inline struct rxr_base_hdr *rxr_get_base_hdr(void *pkt)
{
	return (struct rxr_base_hdr *)pkt;
}

static inline struct rxr_rts_hdr *rxr_get_rts_hdr(void *pkt)
{
	return (struct rxr_rts_hdr *)pkt;
}

static inline struct rxr_connack_hdr *rxr_get_connack_hdr(void *pkt)
{
	return (struct rxr_connack_hdr *)pkt;
}

static inline struct rxr_cts_hdr *rxr_get_cts_hdr(void *pkt)
{
	return (struct rxr_cts_hdr *)pkt;
}

static inline struct rxr_readrsp_hdr *rxr_get_readrsp_hdr(void *pkt)
{
	return (struct rxr_readrsp_hdr *)pkt;
}

static inline struct rxr_ctrl_cq_pkt *rxr_get_ctrl_cq_pkt(void *pkt)
{
	return (struct rxr_ctrl_cq_pkt *)pkt;
}

static inline struct rxr_ctrl_pkt *rxr_get_ctrl_pkt(void *pkt)
{
	return (struct rxr_ctrl_pkt *)pkt;
}

static inline struct rxr_data_pkt *rxr_get_data_pkt(void *pkt)
{
	return (struct rxr_data_pkt *)pkt;
}

static inline int rxr_match_addr(fi_addr_t addr, fi_addr_t match_addr)
{
	return (addr == FI_ADDR_UNSPEC || addr == match_addr);
}

static inline int rxr_match_tag(uint64_t tag, uint64_t ignore,
				uint64_t match_tag)
{
	return ((tag | ignore) == (match_tag | ignore));
}

/*
 * Helper function to compute the maximum payload of the RTS header based on
 * the RTS header flags. The header may have a length greater than the possible
 * RTS payload size if it is a large message.
 */
static inline uint64_t rxr_get_rts_data_size(struct rxr_ep *ep,
					     struct rxr_rts_hdr *rts_hdr)
{
	/*
	 * for read request, rts packet contain no data
	 * because data is on remote host
	 */
	if (rts_hdr->flags & RXR_READ_REQ)
		return 0;

	size_t max_payload_size;

	if (rts_hdr->flags & RXR_REMOTE_CQ_DATA)
		max_payload_size = ep->mtu_size - RXR_CTRL_HDR_SIZE;
	else
		max_payload_size = ep->mtu_size - RXR_CTRL_HDR_SIZE_NO_CQ;

	if (rts_hdr->flags & RXR_REMOTE_SRC_ADDR)
		max_payload_size -= rts_hdr->addrlen;

	if (rts_hdr->flags & RXR_WRITE)
		max_payload_size -= rts_hdr->rma_iov_count *
					sizeof(struct fi_rma_iov);

	return (rts_hdr->data_len > max_payload_size)
		? max_payload_size : rts_hdr->data_len;
}

static inline size_t rxr_get_rx_pool_chunk_cnt(struct rxr_ep *ep)
{
	return MIN(ep->core_rx_size, ep->rx_size);
}

static inline size_t rxr_get_tx_pool_chunk_cnt(struct rxr_ep *ep)
{
	return MIN(ep->max_outstanding_tx, ep->tx_size);
}

static inline int rxr_need_sas_ordering(struct rxr_ep *ep)
{
	/*
	 * RxR needs to reorder RTS packets for send-after-send guarantees
	 * only if the application requested it and the core provider does not
	 * support it.
	 */
	return ((ep->msg_order & FI_ORDER_SAS) &&
		!(ep->core_msg_order & FI_ORDER_SAS) &&
		rxr_env.enable_sas_ordering);
}

/* Initialization functions */
void rxr_reset_rx_tx_to_core(const struct fi_info *user_info,
			     struct fi_info *core_info);
int rxr_get_lower_rdm_info(uint32_t version, const char *node, const char *service,
			   uint64_t flags, const struct util_prov *util_prov,
			   const struct fi_info *util_hints,
			   struct fi_info **core_info);
int rxr_fabric(struct fi_fabric_attr *attr,
	       struct fid_fabric **fabric, void *context);
int rxr_domain_open(struct fid_fabric *fabric, struct fi_info *info,
		    struct fid_domain **dom, void *context);
int rxr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context);
int rxr_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context);

/* AV sub-functions */
int rxr_av_insert_rdm_addr(struct rxr_av *av, const void *addr,
			   fi_addr_t *rdm_fiaddr, uint64_t flags,
			   void *context);
int rxr_av_open(struct fid_domain *domain_fid, struct fi_av_attr *attr,
		struct fid_av **av_fid, void *context);

/* EP sub-functions */
void rxr_ep_progress(struct util_ep *util_ep);
struct rxr_pkt_entry *rxr_ep_get_pkt_entry(struct rxr_ep *rxr_ep,
					   struct ofi_bufpool *pkt_pool);
int rxr_ep_post_buf(struct rxr_ep *ep, uint64_t flags);
ssize_t rxr_ep_send_msg(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			const struct fi_msg *msg, uint64_t flags);
ssize_t rxr_ep_post_data(struct rxr_ep *rxr_ep, struct rxr_tx_entry *tx_entry);
ssize_t rxr_ep_post_readrsp(struct rxr_ep *rxr_ep, struct rxr_tx_entry *tx_entry);
void rxr_ep_init_connack_pkt_entry(struct rxr_ep *ep,
				   struct rxr_pkt_entry *pkt_entry,
				   fi_addr_t addr);
void rxr_ep_calc_cts_window_credits(struct rxr_ep *ep, uint32_t max_window,
				    uint64_t size, int *window, int *credits);
void rxr_ep_init_cts_pkt_entry(struct rxr_ep *ep,
			       struct rxr_rx_entry *rx_entry,
			       struct rxr_pkt_entry *pkt_entry,
			       uint32_t max_window,
			       uint64_t size,
			       int *credits);
void rxr_ep_init_readrsp_pkt_entry(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
				   struct rxr_pkt_entry *pkt_entry);
struct rxr_rx_entry *rxr_ep_get_new_unexp_rx_entry(struct rxr_ep *ep,
						   struct rxr_pkt_entry *unexp_entry);
struct rxr_rx_entry *rxr_ep_split_rx_entry(struct rxr_ep *ep,
					   struct rxr_rx_entry *posted_entry,
					   struct rxr_rx_entry *consumer_entry,
					   struct rxr_pkt_entry *pkt_entry);
#if ENABLE_DEBUG
void rxr_ep_print_pkt(char *prefix,
		      struct rxr_ep *ep,
		      struct rxr_base_hdr *hdr);
#endif

/* CQ sub-functions */
int rxr_cq_handle_rx_error(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
			   ssize_t prov_errno);
int rxr_cq_handle_tx_error(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
			   ssize_t prov_errno);
int rxr_cq_handle_cq_error(struct rxr_ep *ep, ssize_t err);
ssize_t rxr_cq_post_cts(struct rxr_ep *ep,
			struct rxr_rx_entry *rx_entry,
			uint32_t max_window,
			uint64_t size);

int rxr_cq_handle_rx_completion(struct rxr_ep *ep,
				struct fi_cq_msg_entry *comp,
				struct rxr_pkt_entry *pkt_entry,
				struct rxr_rx_entry *rx_entry);

void rxr_cq_write_tx_completion(struct rxr_ep *ep,
				struct fi_cq_msg_entry *comp,
				struct rxr_tx_entry *tx_entry);

void rxr_cq_recv_rts_data(struct rxr_ep *ep,
			  struct rxr_rx_entry *rx_entry,
			  struct rxr_rts_hdr *rts_hdr);

void rxr_cq_handle_pkt_recv_completion(struct rxr_ep *ep,
				       struct fi_cq_msg_entry *comp,
				       fi_addr_t src_addr);

void rxr_cq_handle_pkt_send_completion(struct rxr_ep *rxr_ep,
				       struct fi_cq_msg_entry *comp);

/* Aborts if unable to write to the eq */
static inline void rxr_eq_write_error(struct rxr_ep *ep, ssize_t err,
				      ssize_t prov_errno)
{
	struct fi_eq_err_entry err_entry;
	int ret = -FI_ENOEQ;

	FI_WARN(&rxr_prov, FI_LOG_EQ, "Writing error %s to EQ.\n",
		fi_strerror(err));
	if (ep->util_ep.eq) {
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.err = err;
		err_entry.prov_errno = prov_errno;
		ret = fi_eq_write(&ep->util_ep.eq->eq_fid, FI_NOTIFY,
				  &err_entry, sizeof(err_entry),
				  UTIL_FLAG_ERROR);

		if (ret == sizeof(err_entry))
			return;
	}

	FI_WARN(&rxr_prov, FI_LOG_EQ,
		"Unable to write to EQ: %s. err: %s (%zd) prov_errno: %s (%zd)\n",
		fi_strerror(-ret), fi_strerror(err), err,
		fi_strerror(prov_errno), prov_errno);
	fprintf(stderr,
		"Unable to write to EQ: %s. err: %s (%zd) prov_errno: %s (%zd) %s:%d\n",
		fi_strerror(-ret), fi_strerror(err), err,
		fi_strerror(prov_errno), prov_errno, __FILE__, __LINE__);
	abort();
}

static inline struct rxr_av *rxr_ep_av(struct rxr_ep *ep)
{
	return container_of(ep->util_ep.av, struct rxr_av, util_av);
}

static inline struct rxr_domain *rxr_ep_domain(struct rxr_ep *ep)
{
	return container_of(ep->util_ep.domain, struct rxr_domain, util_domain);
}

static inline uint8_t rxr_ep_mr_local(struct rxr_ep *ep)
{
	struct rxr_domain *domain = container_of(ep->util_ep.domain,
						 struct rxr_domain,
						 util_domain);
	return domain->mr_local;
}

/*
 * today we have only cq res check, in future we will have ctx, and other
 * resource check as well.
 */
static inline uint64_t is_tx_res_full(struct rxr_ep *ep)
{
	return ep->rm_full & RXR_RM_TX_CQ_FULL;
}

static inline uint64_t is_rx_res_full(struct rxr_ep *ep)
{
	return ep->rm_full & RXR_RM_RX_CQ_FULL;
}

static inline void rxr_rm_rx_cq_check(struct rxr_ep *ep, struct util_cq *rx_cq)
{
	fastlock_acquire(&rx_cq->cq_lock);
	if (ofi_cirque_isfull(rx_cq->cirq))
		ep->rm_full |= RXR_RM_RX_CQ_FULL;
	else
		ep->rm_full &= ~RXR_RM_RX_CQ_FULL;
	fastlock_release(&rx_cq->cq_lock);
}

static inline void rxr_rm_tx_cq_check(struct rxr_ep *ep, struct util_cq *tx_cq)
{
	fastlock_acquire(&tx_cq->cq_lock);
	if (ofi_cirque_isfull(tx_cq->cirq))
		ep->rm_full |= RXR_RM_TX_CQ_FULL;
	else
		ep->rm_full &= ~RXR_RM_TX_CQ_FULL;
	fastlock_release(&tx_cq->cq_lock);
}

static inline ssize_t rxr_ep_sendv_pkt(struct rxr_ep *ep,
				       struct rxr_pkt_entry *pkt_entry,
				       fi_addr_t addr, const struct iovec *iov,
				       void **desc, size_t count,
				       uint64_t flags)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = addr;
	msg.context = pkt_entry;
	msg.data = 0;

	return rxr_ep_send_msg(ep, pkt_entry, &msg, flags);
}

/* rxr_pkt_start currently expects data pkt right after pkt hdr */
static inline ssize_t rxr_ep_send_pkt_flags(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry,
					    fi_addr_t addr, uint64_t flags)
{
	struct iovec iov;
	void *desc;

	iov.iov_base = rxr_pkt_start(pkt_entry);
	iov.iov_len = pkt_entry->pkt_size;

	desc = rxr_ep_mr_local(ep) ? fi_mr_desc(pkt_entry->mr) : NULL;

	return rxr_ep_sendv_pkt(ep, pkt_entry, addr, &iov, &desc, 1, flags);
}

static inline ssize_t rxr_ep_send_pkt(struct rxr_ep *ep,
				      struct rxr_pkt_entry *pkt_entry,
				      fi_addr_t addr)
{
	return rxr_ep_send_pkt_flags(ep, pkt_entry, addr, 0);
}

static inline int rxr_ep_post_cts_or_queue(struct rxr_ep *ep,
					   struct rxr_rx_entry *rx_entry,
					   uint64_t bytes_left)
{
	int ret;

	if (rx_entry->state == RXR_RX_QUEUED_CTS)
		return 0;

	ret = rxr_cq_post_cts(ep, rx_entry, rxr_env.rx_window_size,
			      bytes_left);
	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN) {
			rx_entry->state = RXR_RX_QUEUED_CTS;
			dlist_insert_tail(&rx_entry->queued_entry,
					  &ep->rx_entry_queued_list);
			ret = 0;
		} else {
			if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
				assert(0 && "failed to write err cq entry");
		}
	}
	return ret;
}

static inline bool rxr_peer_timeout_expired(struct rxr_ep *ep,
					    struct rxr_peer *peer,
					    uint64_t ts)
{
	return (ts >= (peer->rnr_ts + MIN(rxr_env.max_timeout,
					  peer->timeout_interval *
					  (1 << peer->rnr_timeout_exp))));
}

static inline bool
rxr_multi_recv_buffer_available(struct rxr_ep *ep,
				struct rxr_rx_entry *rx_entry)
{
	assert(rx_entry->fi_flags & FI_MULTI_RECV);
	assert(rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED);

	return (ofi_total_iov_len(rx_entry->iov, rx_entry->iov_count)
		>= ep->min_multi_recv_size);
}

static inline bool
rxr_multi_recv_buffer_complete(struct rxr_ep *ep,
			       struct rxr_rx_entry *rx_entry)
{
	assert(rx_entry->fi_flags & FI_MULTI_RECV);
	assert(rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED);

	return (!rxr_multi_recv_buffer_available(ep, rx_entry) &&
		dlist_empty(&rx_entry->multi_recv_consumers));
}

static inline void
rxr_multi_recv_free_posted_entry(struct rxr_ep *ep,
				 struct rxr_rx_entry *rx_entry)
{
	assert(!(rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED));

	if ((rx_entry->rxr_flags & RXR_MULTI_RECV_CONSUMER) &&
	    rxr_multi_recv_buffer_complete(ep, rx_entry->master_entry))
		rxr_release_rx_entry(ep, rx_entry->master_entry);
}

static inline void
rxr_cq_handle_multi_recv_completion(struct rxr_ep *ep,
				    struct rxr_rx_entry *rx_entry)
{
	assert(!(rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED) &&
	       (rx_entry->rxr_flags & RXR_MULTI_RECV_CONSUMER));

	dlist_remove(&rx_entry->multi_recv_entry);
	rx_entry->rxr_flags &= ~RXR_MULTI_RECV_CONSUMER;

	if (!rxr_multi_recv_buffer_complete(ep, rx_entry->master_entry))
		return;

	/*
	 * Buffer is consumed and all messages have been received. Update the
	 * last message to release the application buffer.
	 */
	rx_entry->cq_entry.flags |= FI_MULTI_RECV;
}

/* Performance counter declarations */
#ifdef RXR_PERF_ENABLED
#define RXR_PERF_FOREACH(DECL)	\
	DECL(perf_rxr_tx),	\
	DECL(perf_rxr_recv),	\
	DECL(rxr_perf_size)	\

enum rxr_perf_counters {
	RXR_PERF_FOREACH(OFI_ENUM_VAL)
};

extern const char *rxr_perf_counters_str[];

static inline void rxr_perfset_start(struct rxr_ep *ep, size_t index)
{
	struct rxr_domain *domain = rxr_ep_domain(ep);
	struct rxr_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct rxr_fabric,
						 util_fabric);
	ofi_perfset_start(&fabric->perf_set, index);
}

static inline void rxr_perfset_end(struct rxr_ep *ep, size_t index)
{
	struct rxr_domain *domain = rxr_ep_domain(ep);
	struct rxr_fabric *fabric = container_of(domain->util_domain.fabric,
						 struct rxr_fabric,
						 util_fabric);
	ofi_perfset_end(&fabric->perf_set, index);
}
#else
#define rxr_perfset_start(ep, index) do {} while (0)
#define rxr_perfset_end(ep, index) do {} while (0)
#endif
#endif
