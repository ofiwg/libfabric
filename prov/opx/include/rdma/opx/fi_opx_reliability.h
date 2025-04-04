/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2025 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_RELIABILITY_H_
#define _FI_PROV_OPX_RELIABILITY_H_

#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx_internal.h"
#include "rbtree.h"
#include "ofi_lock.h"
#include <ofi_mem.h>
#include "uthash.h"

#include "rdma/opx/fi_opx_timer.h"

#include "rdma/opx/opx_tracer.h"

enum ofi_reliability_kind {
	OFI_RELIABILITY_KIND_NONE = 0,
	OFI_RELIABILITY_KIND_OFFLOAD,
	OFI_RELIABILITY_KIND_ONLOAD,
	OFI_RELIABILITY_KIND_RUNTIME,
	OFI_RELIABILITY_KIND_COUNT,
	OFI_RELIABILITY_KIND_UNSET,
};

enum ofi_reliability_app_kind { OFI_RELIABILITY_APP_KIND_MPI = 0, OFI_RELIABILITY_APP_KIND_DAOS };

/* #define SKIP_RELIABILITY_PROTOCOL_RX */
/* #define SKIP_RELIABILITY_PROTOCOL_TX */
/* #define OPX_RELIABILITY_DEBUG */
/* #define OPX_RELIABILITY_TEST */

#ifdef OPX_RELIABILITY_TEST
#define FI_OPX_RELIABILITY_RX_DROP_PACKET(x, y) fi_opx_reliability_rx_drop_packet(x, y)
#else
#define FI_OPX_RELIABILITY_RX_DROP_PACKET(x, y) (0)
#endif

#define PENDING_RX_RELIABLITY_COUNT_MAX (1024) // Max depth of the pending Rx reliablity pool

/* Eager buffer size is 32*262144
 * We'll do 28 * 262144 (i.e. 7MB) as a starting point
 * Empirically this survives a message rate test
 * with no packet loss with a single pair
 * in biband
 * TODO:  fetch this from the context info
 * TODO:  This buffer should be dynamically adjusted
 * back when we get a nack, and dyamically adjusted
 * forward when we have nothing outstanding as
 * a starting point for per destination windowing
 * max outstanding could be adjusted to zero until
 * all the packets are replayed, then this can
 * be adjusted back to it's base value.
 * Either way, there should be knobs and controls
 * to make this dynamic when packets are lost
 */
#ifndef OPX_RELIABILITY_MAX_OUTSTANDING_BYTES_DEFAULT
#define OPX_RELIABILITY_MAX_OUTSTANDING_BYTES_DEFAULT (7u << 20)
#endif

/*
 * Absolute max outstanding =
     (16K IOV Replays + 2K non-IOV Replays) * (8K MAX_MTU) = 144MB
 */
#ifndef OPX_RELIABILITY_MAX_OUTSTANDING_BYTES_MAX
#define OPX_RELIABILITY_MAX_OUTSTANDING_BYTES_MAX (144u << 20)
#endif

struct fi_opx_completion_counter {
	struct fi_opx_completion_counter *next;
	union {
		uint64_t tag;
		ssize_t	 initial_byte_count;
	};
	ssize_t		    byte_counter;
	struct fi_opx_cntr *cntr;
	struct fi_opx_cq   *cq;
	union {
		struct opx_context *context;
		void		   *container;
	};
	void (*hit_zero)(struct fi_opx_completion_counter *);
};

union fi_opx_reliability_deferred_work;
struct fi_opx_reliability_work_elem {
	struct slist_entry slist_entry;
	ssize_t (*work_fn)(union fi_opx_reliability_deferred_work *work_state);
};

struct fi_opx_reliability_tx_sdma_replay_params {
	struct fi_opx_reliability_work_elem work_elem;
	void				   *opx_ep;
	struct slist			    sdma_reqs;
	uint64_t			    flow_key;
};

#define OPX_RELIABILITY_TX_MAX_REPLAYS (32)
struct fi_opx_reliability_tx_pio_replay_params {
	struct fi_opx_reliability_work_elem  work_elem;
	void				    *opx_ep;
	struct fi_opx_reliability_tx_replay *replays[OPX_RELIABILITY_TX_MAX_REPLAYS];
	uint64_t			     flow_key;
	uint16_t			     num_replays;
	uint16_t			     start_index;
};

union fi_opx_reliability_deferred_work {
	struct fi_opx_reliability_work_elem		work_elem;
	struct fi_opx_reliability_tx_sdma_replay_params sdma_replay;
	struct fi_opx_reliability_tx_pio_replay_params	pio_replay;
};

#define OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_MIN     (1)
#define OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_MAX     (65535)
#define OPX_RELIABILITY_MAX_UNCONGESTED_PINGS_DEFAULT (128)
#define OPX_RELIABILITY_MAX_CONGESTED_PINGS_MIN	      (1)
#define OPX_RELIABILITY_MAX_CONGESTED_PINGS_MAX	      (65535)
#define OPX_RELIABILITY_MAX_CONGESTED_PINGS_DEFAULT   (4)
struct fi_opx_reliability_service {
	uint64_t max_outstanding_bytes;
	uint64_t usec_next;
	uint32_t usec_max;
	uint32_t preemptive_ack_rate_mask;

	struct {
		union fi_opx_timer_stamp timestamp; /*  2 qws =  16 bytes */
		union fi_opx_timer_state timer;	    /*  5 bytes           */
		uint8_t			 unused_padding[3];

		RbtHandle flow; /*  1 qw  =   8 bytes */
		uint64_t  ping_start_key;

		/* == CACHE LINE == */
		uint16_t max_uncongested_pings;
		uint16_t max_congested_pings;
		uint8_t	 congested_flag;
		uint8_t	 unused_padding2[3];

		struct {
			uint64_t		     unused_cacheline_1[3];
			union fi_opx_hfi1_pio_state *pio_state;
			volatile uint64_t	    *pio_scb_sop_first;
			volatile uint64_t	    *pio_credits_addr;
			volatile uint64_t	    *pio_scb_first;

			/* == CACHE LINE == */
			struct fi_opx_hfi1_txe_scb_9B  ping_model_9B;
			struct fi_opx_hfi1_txe_scb_9B  ack_model_9B;
			struct fi_opx_hfi1_txe_scb_9B  nack_model_9B;
			struct fi_opx_hfi1_txe_scb_16B ping_model_16B;
			struct fi_opx_hfi1_txe_scb_16B ack_model_16B;
			struct fi_opx_hfi1_txe_scb_16B nack_model_16B;
		} hfi1;
	} tx __attribute__((__packed__));

	/* == CACHE LINE == */

	struct {
		RbtHandle flow; /*  1 qw  =   8 bytes */
		struct {
			struct fi_opx_hfi1_rxe_state state; /*  2 qws  =  16 bytes */

			struct {
				uint32_t	  *rhf_base;
				volatile uint64_t *head_register;
			} hdrq;

			/* -- not critical; can be moved to another cacheline */

			struct {
				uint32_t	  *base_addr;
				uint32_t	   elemsz;
				uint32_t	   last_egrbfr_index;
				volatile uint64_t *head_register;
			} egrq;
		} hfi1;
	} rx;

	/* == CACHE LINE == */

	struct fi_opx_hfi1_context		*context;
	volatile uint64_t			 enabled;
	volatile uint64_t			 active;
	pthread_t				 thread; /* 1 qw =  8 bytes */
	struct ofi_bufpool			*pending_rx_reliability_pool;
	struct fi_opx_pending_rx_reliability_op *pending_rx_reliability_ops_hashmap;
	uint64_t				 backoff_period;
	opx_lid_t				 lid;
	uint32_t				 preemptive_ack_rate;

	/* == CACHE LINE == */
	struct ofi_bufpool *uepkt_pool;
	struct slist	    work_pending; /* 16 bytes */
	struct ofi_bufpool *work_pending_pool;
	struct ofi_bufpool *sdma_replay_request_pool;

	int			  is_backoff_enabled;
	enum ofi_reliability_kind reliability_kind; /* 4 bytes */
	uint16_t		  unused[3];

} __attribute__((__aligned__(64))) __attribute__((__packed__));

union fi_opx_reliability_service_flow_key {
	uint64_t value;
	uint32_t value32b[2];
	struct {
		uint32_t slid : 24;
		uint32_t src_rx : 8;
		uint32_t dlid : 24;
		uint32_t dst_rx : 8;
	} __attribute__((__packed__));
};

struct fi_opx_reliability_flow {
	// opx_spin_t				lock;
	uint64_t				  next_psn;
	union fi_opx_reliability_service_flow_key key;
	struct fi_opx_reliability_rx_uepkt	 *uepkt;
};

struct fi_opx_pending_rx_reliability_op_key {
	uint64_t key;
	uint64_t psn_start;
};

struct fi_opx_pending_rx_reliability_op {
	struct fi_opx_pending_rx_reliability_op_key key;
	uint64_t				    psn_count;
	uint64_t				    psn_count_coalesce;
	opx_lid_t				    slid;
	uint32_t				    unused;
	uint64_t				    rx;
	uint64_t				    ud_opcode; // Only needs to be uint8_t
	UT_hash_handle				    hh;	       /* makes this structure hashable */
};

struct fi_opx_reliability_tx_replay {
	/* == CACHE LINE 0 == */
	struct fi_opx_reliability_tx_replay *next;
	struct fi_opx_reliability_tx_replay *prev;
	uint64_t			     target_reliability_rx;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	struct fi_opx_completion_counter    *cc_ptr;
	uint64_t			     cc_dec;
	union {
		uint64_t     *payload;
		struct iovec *iov;
	};
	uint16_t unused1;
	uint16_t nack_count;
	bool	 acked;
	bool	 pinned;
	bool	 use_sdma;
	bool	 use_iov;

	/* == CACHE LINE 1 == */
	void		  *sdma_we;
	uint32_t	   sdma_we_use_count;
	enum fi_hmem_iface hmem_iface;
	uint64_t	   hmem_device;
	uint64_t	   unused3[5];

#ifndef NDEBUG
	/* == CACHE LINE == */
	uint64_t orig_payload[8];
#endif

	/* == CACHE LINE == */

	/* --- MUST BE 64 BYTE ALIGNED --- */
	union opx_hfi1_txe_scb_union scb;

	uint8_t data[];
} __attribute__((__aligned__(64)));

#define OPX_REPLAY_HDR(_replay) OPX_REPLAY_HDR_TYPE(_replay, OPX_HFI1_TYPE)

#define OPX_REPLAY_HDR_TYPE(_replay, _hfi1_type)                                            \
	((_hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) ? (&((_replay)->scb.scb_9B.hdr)) : \
							   (&((_replay)->scb.scb_16B.hdr)))

OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_reliability_tx_replay, sdma_we) == FI_OPX_CACHE_LINE_SIZE,
			"Reliability Replay sdma_we should start on first cacheline!");
OPX_COMPILE_TIME_ASSERT((offsetof(struct fi_opx_reliability_tx_replay, scb) & (FI_OPX_CACHE_LINE_SIZE - 1)) == 0,
			"Reliability Replay scb must be 64-byte aligned!");

struct fi_opx_reliability_resynch_flow {
	bool					  client_ep;
	uint64_t				  resynch_counter;
	union fi_opx_reliability_service_flow_key key;
	/* client related fields */
	bool remote_ep_resynch_completed;
	bool resynch_client_ep;
};

// Begin rbtree implementation
// Import and inline data structures from the red-black tree implementation
// The RBTree functions cause a fairly large loss of message rate
// as the message rate code is instruction bound and the out of line implementations
// are small  These data structures mirror the out of line implementations
// but if those ever change, we need to update our copies of this
// or lobby for an interface change to rb tree to allow for inline implementations
// On skylake 18 core pairs, the rbtree implementation leads to a loss of about
// 10 mmps by not being in lined
// These substitute inline functions are copies of the implementation in rbtree.c
#include "ofi_tree.h"
typedef enum ofi_node_color NodeColor;

typedef struct NodeTag {
	struct NodeTag *left;	// left child
	struct NodeTag *right;	// right child
	struct NodeTag *parent; // parent
	NodeColor	color;	// node color (BLACK, RED)
	void	       *key;	// key used for searching
	void	       *val;	// user data
} NodeType;

typedef struct RbtTag {
	NodeType *root; // root of red-black tree
	NodeType  sentinel;
	int (*compare)(void *a, void *b); // compare keys
} RbtType;

__OPX_FORCE_INLINE__
int fi_opx_reliability_compare(void *a, void *b)
{
	const uintptr_t a_key = (uintptr_t) a;
	const uintptr_t b_key = (uintptr_t) b;

	if (a_key > b_key) {
		return 1;
	}
	if (a_key < b_key) {
		return -1;
	}

	return 0;
}

__OPX_FORCE_INLINE__
void *fi_opx_rbt_find(RbtHandle h, void *key)
{
	RbtType	 *rbt = h;
	NodeType *current;
	current = rbt->root;
	while (current != &rbt->sentinel) {
		int rc = fi_opx_reliability_compare(key, current->key);
		if (rc == 0) {
			return current;
		}
		current = (rc < 0) ? current->left : current->right;
	}
	return NULL;
}

__OPX_FORCE_INLINE__
void **fi_opx_rbt_value_ptr(RbtHandle h, RbtIterator it)
{
	NodeType *i = it;

	return &i->val;
}

__OPX_FORCE_INLINE__
RbtIterator fi_opx_rbt_begin(RbtHandle h)
{
	RbtType *rbt = h;

	// return pointer to first value
	NodeType *i;
	for (i = rbt->root; i->left != &rbt->sentinel; i = i->left)
		;
	return i != &rbt->sentinel ? i : NULL;
}

__OPX_FORCE_INLINE__
void fi_opx_rbt_key(RbtIterator it, uint64_t *key)
{
	NodeType *i = it;

	*key = (uint64_t) i->key;
}

__OPX_FORCE_INLINE__
void fi_opx_rbt_key_value(RbtHandle h, RbtIterator it, void **key, void **val)
{
	NodeType *i = it;

	*key = i->key;
	*val = i->val;
}

/*
 * Initialize the reliability service - and pthread
 *
 * \return reliability service hfi1 rx identifier
 */
uint8_t fi_opx_reliability_service_init(struct fi_opx_reliability_service *service, uuid_t unique_job_key,
					struct fi_opx_hfi1_context     *hfi1,
					const enum ofi_reliability_kind reliability_kind);
void	fi_opx_reliability_model_init_16B(struct fi_opx_reliability_service *service, struct fi_opx_hfi1_context *hfi1);
void	fi_opx_reliability_service_fini(struct fi_opx_reliability_service *service);

void	 fi_reliability_service_ping_remote(struct fid_ep *ep, struct fi_opx_reliability_service *service);
unsigned fi_opx_reliability_service_poll_hfi1(struct fid_ep *ep, struct fi_opx_reliability_service *service);

void fi_opx_reliability_service_process_pending(struct fi_opx_reliability_service *service);

#define RX_CMD (0x0000000000000008ul)
#define TX_CMD (0x0000000000000010ul)

enum opx_reliability_status { OPX_RELIABILITY_SUCCESS = 0, OPX_RELIABILITY_EXCEPTION, OPX_RELIABILITY_FLOW_ERROR };

struct fi_opx_reliability_rx_uepkt {
	struct fi_opx_reliability_rx_uepkt *prev;
	struct fi_opx_reliability_rx_uepkt *next;
	uint64_t			    psn;
	bool				    processed;
	uint8_t				    unused_b[7];
	uint64_t			    unused_0[4];

	/* == CACHE LINE == */

	uint64_t		  unused_1;
	union opx_hfi1_packet_hdr hdr; /* 56 bytes */

	/* == CACHE LINE == */
	uint8_t payload[FI_OPX_HFI1_PACKET_MTU];

} __attribute__((__packed__)) __attribute__((aligned(64)));

union fi_opx_reliability_tx_psn {
	uint64_t value;
	struct {
		uint64_t psn : 24;
		uint64_t throttle : 8;
		uint64_t nack_count : 8;
		uint64_t bytes_outstanding : 24;
	} psn;
} __attribute__((__packed__));

// TODO - make these tunable.
#define FI_OPX_RELIABILITY_TX_REPLAY_BLOCKS	(2048)
#define FI_OPX_RELIABILITY_TX_REPLAY_IOV_BLOCKS (8192)
#define OPX_MAX_OUTSTANDING_REPLAYS		(FI_OPX_RELIABILITY_TX_REPLAY_BLOCKS + FI_OPX_RELIABILITY_TX_REPLAY_IOV_BLOCKS)

#define OPX_REPLAY_BASE_SIZE		   (sizeof(struct fi_opx_reliability_tx_replay))
#define OPX_REPLAY_IOV_SIZE		   (sizeof(struct iovec) << 1)
#define OPX_REPLAY_PAYLOAD_SIZE		   (FI_OPX_HFI1_PACKET_MTU + 64)
#define OPX_RELIABILITY_TX_REPLAY_SIZE	   (OPX_REPLAY_BASE_SIZE + OPX_REPLAY_PAYLOAD_SIZE)
#define OPX_RELIABILITY_TX_REPLAY_IOV_SIZE (OPX_REPLAY_BASE_SIZE + OPX_REPLAY_IOV_SIZE)

// Maximum PSNs to NACK when responding to a ping
#ifndef OPX_RELIABILITY_RX_MAX_NACK
#define OPX_RELIABILITY_RX_MAX_NACK (32)
#endif

// Maximum PSNs to NACK when receiving an out of order packet
#ifndef OPX_RELIABILITY_RX_MAX_PRE_NACK
#define OPX_RELIABILITY_RX_MAX_PRE_NACK (1)
#endif

/*
 * For incoming packets:
 * Normally, we can simply add the high 40 bits of the receiver's 64-bit PSN
 * to the 24-bit PSN of an incoming packet to get the 64-bit PSN. However,
 * we have to deal with the case where the Sender's PSN has rolled over but
 * the receiver's hasn't. The difficulty is determining when this is the case.
 *
 * In the ideal case, the Sender's PSN will always be greater than or equal to
 * the PSN the receiver expects to see (the expected PSN). However, this is
 * not the case when the packet is a duplicate of one that was already
 * received. This can happen when a packet gets NACK'ed twice for some reason,
 * resulting in the sender sending two new copies. This is futher complicated
 * by the fact that the sender's replay list has a fixed size but does not
 * have to contain a continuous range of PSNs - there can be gaps. Thus it
 * is hard to recognize packets that are "too old" to be legitimate.
 *
 * Definitions:
 * R-PSN-64 = The next expected PSN.
 * R-PSN-24 = (R-PSN-64 & 0xFFFFFF)
 * S-PSN-24 = 24-bit OPA PSN from the incoming packet.
 * S-PSN-64 = 64-bit "scaled up" PSN of incoming packet.
 *
 * (Note: I'm deliberately not defining "Low", "High", or "Middle" yet.)
 *
 * S-PSN-24		R-PSN-24
 * Middle		Middle		S-PSN-64 = S-PSN-24 + (high 40 bits of R-PSN-64)
 *
 * Low			High		Sender has rolled over. Add an extra 2^24.
 * 							S-PSN-64 = S-PSN-24 + (high 40 bits of R-PSN-64) +
 * 							2^24
 *
 * High			Low			Packet is old and should be discarded. (The only
 * 							way for this to occur is if the R-PSN-24 has rolled
 * 							over but the incoming S-PSN-24 has not. Since
 * 							R-PSN-64 represents the highest PSN successfully
 * 							received, we must have already received this packet.
 *
 * The risks in defining "High" and "Low" is that if they are too generous we
 * might scale up S-PSN-24 incorrectly, causing us to discard a valid packet,
 * or assign a packet an invalid PSN (causing it to be processed in the wrong
 * order).
 *
 * First we will assume we will never have to deal with PSNs that are correctly
 * close to 2^24 apart because that would indicate much more fundamental
 * problems with the protocol - how did we manage to drop ~2^24 packets?
 *
 * This assumption allows us to be generous in how we define "Low" and "High"
 * for the Low/High case. If we define "Low" as < 10k and "High" as > "2^24-10k"
 * that should be adequate. Why 10K? We currently allocate 2048 replay buffers
 * with room for payloads, and 8192 replay buffers for IOV only, meaning a
 * sender could have up to 10K PSNs "in flight" at once.
 *
 * In the "High/Low" case, if we say S-PSN-64 = S-PSN-24 + (high 40 bits of
 * R-PSN-64) then S-PSN-64 will be much, much, higher than R-PSN-64. Presumably
 * close to 2^24 apart but it is hard to say "how close". Let's assume anything
 * greater than 2^23 is enough to reject a packet as "too old".
 *
 * Ping/Ack/Nack Handling:
 * Ping/Ack/Nack packets can hold 64-bit PSNs but the replay queue stores
 * packets it uses 24 bit PSNs. Therefore, a ping will contain a 24-bit PSN
 * and a count. To prevent rollover issues, code sending pings
 * will truncate the count in the case where the replay queue contains a
 * rollover. Because of this limitation the Ack and Nack replies are similarly
 * constrained.
 *
 */
#define MAX_PSN	     0x0000000000FFFFFFull
#define MAX_PSN_MASK 0xFFFFFFFFFF000000ull

#define PSN_WINDOW_SIZE 0x0000000001000000ull
#define PSN_LOW_WINDOW	OPX_MAX_OUTSTANDING_REPLAYS
#define PSN_HIGH_WINDOW (PSN_WINDOW_SIZE - PSN_LOW_WINDOW)
#define PSN_AGE_LIMIT	0x0000000000F00000ull

struct fi_opx_reliability_client_state {
	/* == CACHE LINE 0 == */
	enum ofi_reliability_kind kind; /* runtime check for fi_cq_read(), etc */
	uint32_t		  pad;

	RbtHandle tx_flow_rbtree;
	RbtHandle rx_flow_rbtree;

	struct ofi_bufpool *replay_pool;     // for main data path
	struct ofi_bufpool *replay_iov_pool; // for main data path

	struct fi_opx_reliability_service *service;
	void (*process_fn)(struct fid_ep *ep, const union opx_hfi1_packet_hdr *const hdr, const uint8_t *const payload);

	opx_lid_t lid;
	uint8_t	  rx;
	uint8_t	  unused1[3];

	/* == CACHE LINE 1 == */
	/* -- not critical; only for debug, init/fini, etc. -- */
	uint16_t		  drop_count;
	uint16_t		  drop_mask;
	enum ofi_reliability_kind reliability_kind;

	RbtHandle flow_rbtree_resynch;
	// 80 bytes
} __attribute__((__packed__)) __attribute__((aligned(64)));
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_reliability_client_state, drop_count) == FI_OPX_CACHE_LINE_SIZE,
			"Reliability state->drop_count should start on cacheline 1!");

void fi_opx_reliability_client_init(struct fi_opx_reliability_client_state *state,
				    struct fi_opx_reliability_service *service, const uint8_t rx,
				    void (*process_fn)(struct fid_ep *ep, const union opx_hfi1_packet_hdr *const hdr,
						       const uint8_t *const payload));

void fi_opx_reliability_client_fini(struct fi_opx_reliability_client_state *state);

__OPX_FORCE_INLINE__
unsigned fi_opx_reliability_client_active(struct fi_opx_reliability_client_state *state)
{
	return ((state->replay_pool && !ofi_bufpool_empty(state->replay_pool)) ||
		(state->replay_iov_pool && !ofi_bufpool_empty(state->replay_iov_pool)));
}

static inline void opx_reliability_service_append_replay(struct fi_opx_reliability_client_state *state,
							 struct fi_opx_reliability_tx_replay *replay, opx_lid_t slid,
							 opx_lid_t dlid, uint8_t src_rx, uint8_t dst_rx,
							 const enum opx_hfi1_type hfi1_type)
{
	union fi_opx_reliability_service_flow_key key = {.slid	 = slid,
							 .src_rx = src_rx,
							 .dlid	 = dlid,
							 .dst_rx = dst_rx};

	void *itr = NULL;

#ifdef OPX_RELIABILITY_DEBUG
	fprintf(stderr, "(tx) packet %016lx %08u posted.\n", key.value,
		FI_OPX_HFI1_PACKET_PSN(OPX_REPLAY_HDR_TYPE(replay, hfi1_type)));
#endif

#ifndef NDEBUG
	itr = fi_opx_rbt_find(state->tx_flow_rbtree, (void *) key.value);
	if (itr == NULL) {
		fprintf(stderr,
			"(%d) %s:%s():%d [%016lX] [slid=%08X tx=%08X dlid=%08X rx=%0hhX] Error trying to register replay for flow with no handshake!\n",
			getpid(), __FILE__, __func__, __LINE__, key.value, slid,
			FI_OPX_HFI1_PACKET_ORIGIN_RX(OPX_REPLAY_HDR_TYPE(replay, hfi1_type)), dlid,
			OPX_REPLAY_HDR_TYPE(replay, hfi1_type)->bth.rx);
		assert(itr);
	}
#endif
	/* search for existing unack'd flows */
	itr = fi_opx_rbt_find(state->service->tx.flow, (void *) key.value);
	if (OFI_UNLIKELY((itr == NULL))) {
		/* did not find an existing flow */
		replay->prev = replay;
		replay->next = replay;

		rbtInsert(state->service->tx.flow, (void *) key.value, (void *) replay);

#ifdef OPX_RELIABILITY_DEBUG
		fprintf(stderr, "\nNew flow: %016lx\n", key.value);
#endif
		return;
	}

	struct fi_opx_reliability_tx_replay **value_ptr =
		(struct fi_opx_reliability_tx_replay **) fi_opx_rbt_value_ptr(state->service->tx.flow, itr);

	struct fi_opx_reliability_tx_replay *head = *value_ptr;

	if (head == NULL) {
		/* the existing flow does not have any un-ack'd replay buffers */
		replay->prev = replay;
		replay->next = replay;
		*value_ptr   = replay;

	} else {
		/* insert this replay at the end of the list */
		replay->prev	 = head->prev;
		replay->next	 = head;
		head->prev->next = replay;
		head->prev	 = replay;
	}
}

#ifdef OPX_RELIABILITY_TEST

#define FI_PSN_TO_DROP 0xfffff0

// Debugging tool that deliberately drops packets.
static inline uint16_t fi_opx_reliability_rx_drop_packet(struct fi_opx_reliability_client_state *state,
							 const union opx_hfi1_packet_hdr *const	 hdr)
{
/*
 * Two variations of when to drop packets. The first drops a percentage of the
 * incoming packets. The other drops 1 specific packet every 2^24 packets (i.e.
 * according to the incoming PSN.) When debugging a reliability issue you can
 * use either of these or code up something different depending on what you're
 * trying to debug.
 */
#if 1
	// drops a percentage of the packets based on drop_mask.
	const uint16_t tmp = state->drop_count & state->drop_mask;

	if (tmp == 0) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "DEBUG: discarding packet %hu\n", state->drop_count);
	}

	state->drop_count = tmp + 1;
	return !tmp;
#else
	// drops every other version of this packet(so reliability can shove it through)
	const uint64_t psn = hdr->reliability.psn;
	if (psn == FI_PSN_TO_DROP && (state->drop_count == 0)) {
		fprintf(stderr, "Dropping packet %lx\n", psn);
		state->drop_count++;
		return 1;
	} else if (psn == FI_PSN_TO_DROP) {
		fprintf(stderr, "Allowing replay packet %lx\n", psn);
		state->drop_count = 0;
	}
	return 0;
#endif
}
#endif

#ifdef OPX_DEBUG_COUNTERS_RELIABILITY_PING
void dump_ping_counts();
#endif

ssize_t fi_opx_hfi1_tx_reliability_inject_ud_resynch(struct fid_ep *ep, const uint64_t key, const opx_lid_t dlid,
						     const uint64_t reliability_rx, const uint64_t opcode);

__OPX_FORCE_INLINE__
size_t fi_opx_reliability_replay_get_payload_size(struct fi_opx_reliability_tx_replay *replay)
{
	if (replay->use_iov) {
		return replay->iov->iov_len;
	}

	/* reported in LRH as the number of 4-byte words in the packet; header + payload + icrc */
	/* Inlined but called from non-inlined functions with no const hfi1 type, so just use the runtime check */
	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		const uint16_t lrh_pktlen_le = ntohs(replay->scb.scb_9B.hdr.lrh_9B.pktlen);
		const size_t   total_bytes   = (lrh_pktlen_le - 1) * 4; /* do not copy the trailing icrc */
		return (total_bytes - sizeof(struct fi_opx_hfi1_stl_packet_hdr_9B));
	} else {
		const uint16_t lrh_pktlen_le = replay->scb.scb_16B.hdr.lrh_16B.pktlen;
		const size_t   total_bytes   = (lrh_pktlen_le - 1) * 8; /* do not copy the trailing icrc */
		return (total_bytes - sizeof(struct fi_opx_hfi1_stl_packet_hdr_16B));
	}
}

__OPX_FORCE_INLINE__
union fi_opx_reliability_tx_psn *opx_reliability_create_tx_flow(struct fi_opx_reliability_client_state *state,
								const uint64_t				key)
{
	union fi_opx_reliability_tx_psn value = {.value = 0};

	RbtStatus rbt_rc = rbtInsert(state->tx_flow_rbtree, (void *) key, (void *) value.value);

	if (OFI_UNLIKELY(rbt_rc != RBT_STATUS_OK)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"Error creating TX Flow: Could not insert flow into tx_flow_rbtree, rbtInsert() returned %d\n",
			rbt_rc);
		return NULL;
	}

	void *itr = fi_opx_rbt_find(state->tx_flow_rbtree, (void *) key);
	assert(itr);
	return (union fi_opx_reliability_tx_psn *) fi_opx_rbt_value_ptr(state->tx_flow_rbtree, itr);
}

__OPX_FORCE_INLINE__
struct fi_opx_reliability_flow *fi_opx_reliability_create_rx_flow(struct fi_opx_reliability_client_state *state,
								  const uint64_t			  key)
{
	/* TODO - allocate from a pool of flow objects instead for better memory utilization */
	int				rc __attribute__((unused));
	struct fi_opx_reliability_flow *flow = NULL;
	rc				     = posix_memalign((void **) &flow, 32, sizeof(*flow));
	assert(rc == 0);

	flow->next_psn	= 0;
	flow->key.value = key;
	flow->uepkt	= NULL;
	// ofi_spin_init(&flow->lock);

	RbtStatus rbt_rc = rbtInsert(state->rx_flow_rbtree, (void *) key, (void *) flow);

	if (OFI_UNLIKELY(rbt_rc != RBT_STATUS_OK)) {
		free(flow);
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"Error creating RX Flow: Could not insert flow into rx_flow_rbtree, rbtInsert() returned %d\n",
			rbt_rc);
		return NULL;
	}

	rbt_rc = rbtInsert(state->service->rx.flow, (void *) flow->key.value, (void *) flow);

	if (OFI_UNLIKELY(rbt_rc != RBT_STATUS_OK)) {
		void *itr = fi_opx_rbt_find(state->rx_flow_rbtree, (void *) key);
		if (itr) {
			rbtErase(state->rx_flow_rbtree, itr);
		}
		free(flow);
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"Error creating RX Flow: Could not insert flow into rx.flow, rbtInsert() returned %d\n",
			rbt_rc);
		return NULL;
	}

	return flow;
}

/*
 * returns OPX_RELIABILITY_SUCCESS if this packet is expected
 * returns OPX_RELIABILITY_FLOW_ERROR if there was a problem creating the RX flow
 * returns OPX_RELIABILITY_EXCEPTION if this packet was not the next expected,
 *   and flow_ptr will be set to point to the RX flow
 */
static inline unsigned fi_opx_reliability_rx_check(struct fi_opx_reliability_client_state *state, opx_lid_t slid,
						   uint64_t src_origin_rx, uint32_t psn,
						   struct fi_opx_reliability_flow **flow_ptr)
{
	struct fi_opx_reliability_flow *flow;

	void *itr, *key_ptr;

	const union fi_opx_reliability_service_flow_key key = {.slid   = slid,
							       .src_rx = src_origin_rx,
							       .dlid   = state->lid,
							       .dst_rx = state->rx};

	itr = fi_opx_rbt_find(state->rx_flow_rbtree, (void *) key.value);
	if (OFI_UNLIKELY(itr == NULL)) {
#ifdef OPX_RELIABILITY_DEBUG
		fprintf(stderr, "(rx) packet %016lx %08u received but no flow for this found!\n", key.value, psn);
#endif
		flow = fi_opx_reliability_create_rx_flow(state, key.value);
		if (OFI_UNLIKELY(flow == NULL)) {
			(*flow_ptr) = NULL;
			return OPX_RELIABILITY_FLOW_ERROR;
		}
	} else {
		fi_opx_rbt_key_value(state->rx_flow_rbtree, itr, &key_ptr, (void **) &flow);
	}

	if (((flow->next_psn & MAX_PSN) == psn) && (flow->uepkt == NULL)) {
#ifdef OPX_RELIABILITY_DEBUG
		fprintf(stderr, "(rx) packet %016lx %08u received.FI_OPX_RELIABILITY_EXPECTED\n", key.value, psn);
#endif
		flow->next_psn += 1;
		(*flow_ptr) = NULL;
		return OPX_RELIABILITY_SUCCESS;
	}
#ifdef OPX_RELIABILITY_DEBUG
	fprintf(stderr, "(rx) packet %016lx %08u received.FI_OPX_RELIABILITY_EXCEPTION\n", key.value, psn);
#endif
	(*flow_ptr) = flow;
	return OPX_RELIABILITY_EXCEPTION;
}

void fi_opx_hfi1_rx_reliability_ping(struct fid_ep *ep, struct fi_opx_reliability_service *service, const uint64_t key,
				     uint64_t psn_count, uint64_t psn_start, const opx_lid_t slid, const uint64_t rx);

void fi_opx_hfi1_rx_reliability_ack(struct fid_ep *ep, struct fi_opx_reliability_service *service, const uint64_t key,
				    const uint64_t psn_count, const uint64_t psn_start);

void fi_opx_hfi1_rx_reliability_nack(struct fid_ep *ep, struct fi_opx_reliability_service *service, const uint64_t key,
				     const uint64_t psn_count, const uint64_t psn_start);

void fi_opx_reliability_rx_exception(struct fi_opx_reliability_client_state *state,
				     struct fi_opx_reliability_flow *flow, opx_lid_t slid, uint64_t origin_rx,
				     uint32_t psn, struct fid_ep *ep, const union opx_hfi1_packet_hdr *const hdr,
				     const uint8_t *const payload, const enum opx_hfi1_type hfi1_type,
				     const uint8_t opcode, const bool ctx_sharing);

ssize_t fi_opx_hfi1_tx_reliability_inject(struct fid_ep *ep, const uint64_t key, const opx_lid_t dlid,
					  const uint64_t reliability_rx, const uint64_t psn_start,
					  const uint64_t psn_count, const uint64_t opcode,
					  const enum opx_hfi1_type hfi1_type, const bool ctx_sharing);

void fi_opx_hfi1_rx_reliability_send_pre_acks(struct fid_ep *ep, const opx_lid_t dlid, const uint64_t reliability_rx,
					      const uint64_t psn_start, const uint64_t psn_count,
					      const union opx_hfi1_packet_hdr *const hdr, opx_lid_t slid,
					      const enum opx_hfi1_type hfi1_type, const bool ctx_sharing);

void fi_opx_hfi1_rx_reliability_resynch(struct fid_ep *ep, struct fi_opx_reliability_service *service,
					uint32_t origin_reliability_rx, const union opx_hfi1_packet_hdr *const hdr,
					const bool ctx_sharing);

void fi_opx_hfi1_rx_reliability_ack_resynch(struct fid_ep *ep, struct fi_opx_reliability_service *service,
					    const union opx_hfi1_packet_hdr *const hdr);

__OPX_FORCE_INLINE__
int32_t fi_opx_reliability_tx_max_nacks()
{
	// TODO, make this tunable.
	return 0;
}

void fi_opx_reliability_inc_throttle_count(struct fid_ep *ep);
void fi_opx_reliability_inc_throttle_nacks(struct fid_ep *ep);
void fi_opx_reliability_inc_throttle_maxo(struct fid_ep *ep);

__OPX_FORCE_INLINE__
int32_t fi_opx_reliability_tx_available_psns(struct fid_ep *ep, struct fi_opx_reliability_client_state *state,
					     const opx_lid_t lid, const uint64_t rx,
					     union fi_opx_reliability_tx_psn **psn_ptr, uint32_t psns_to_get,
					     uint32_t bytes_per_packet)
{
	OPX_TRACER_TRACE_SDMA(OPX_TRACER_BEGIN, "GET_PSNS");
	assert(psns_to_get && psns_to_get <= MAX(OPX_HFI1_SDMA_MAX_PKTS_TID, OPX_HFI1_SDMA_MAX_PKTS));
	union fi_opx_reliability_service_flow_key key = {
		.slid	= (uint32_t) state->lid,
		.src_rx = (uint32_t) state->rx,
		.dlid	= (uint32_t) lid,
		.dst_rx = (uint32_t) rx,
	};

	void *itr = fi_opx_rbt_find(state->tx_flow_rbtree, (void *) key.value);
	if (OFI_UNLIKELY(!itr)) {
		*psn_ptr = opx_reliability_create_tx_flow(state, key.value);
		if (OFI_UNLIKELY((*psn_ptr) == NULL)) {
			OPX_TRACER_TRACE_SDMA(OPX_TRACER_INSTANT, "GET_PSN_THROTTLE");
			return -1;
		}
	} else {
		*psn_ptr = (union fi_opx_reliability_tx_psn *) fi_opx_rbt_value_ptr(state->tx_flow_rbtree, itr);
	}

	/*
	 * We can leverage the fact that every packet needs a packet sequence
	 * number before it can be sent to implement some simply throttling.
	 *
	 * If the throttle is on, or if the # of bytes outstanding exceeds
	 * a threshold, return an error.
	 */
	if (OFI_UNLIKELY((*psn_ptr)->psn.throttle != 0)) {
		OPX_TRACER_TRACE_SDMA(OPX_TRACER_INSTANT, "GET_PSN_THROTTLE");
		return -1;
	}
	if (OFI_UNLIKELY((*psn_ptr)->psn.nack_count > fi_opx_reliability_tx_max_nacks())) {
		(*psn_ptr)->psn.throttle = 1;
		fi_opx_reliability_inc_throttle_count(ep);
		fi_opx_reliability_inc_throttle_nacks(ep);
		OPX_TRACER_TRACE_SDMA(OPX_TRACER_END_EAGAIN_SDMA_PSNS_MAX_NACKS, "GET_PSNS");
		return -1;
	}
	uint32_t max_outstanding = state->service->max_outstanding_bytes;
	if (OFI_UNLIKELY((*psn_ptr)->psn.bytes_outstanding > max_outstanding)) {
		(*psn_ptr)->psn.throttle = 1;
		fi_opx_reliability_inc_throttle_count(ep);
		fi_opx_reliability_inc_throttle_maxo(ep);
		OPX_TRACER_TRACE_SDMA(OPX_TRACER_END_EAGAIN_SDMA_PSNS_MAX_OUT, "GET_PSNS");
		return -1;
	}

	uint32_t bytes_avail = max_outstanding - (*psn_ptr)->psn.bytes_outstanding;
	OPX_TRACER_TRACE_SDMA(OPX_TRACER_END_SUCCESS, "GET_PSNS");
	return MIN(bytes_avail / bytes_per_packet, psns_to_get);
}

__OPX_FORCE_INLINE__
int32_t fi_opx_reliability_tx_next_psn(struct fid_ep *ep, struct fi_opx_reliability_client_state *state,
				       const opx_lid_t lid, const uint64_t rx,
				       union fi_opx_reliability_tx_psn **psn_ptr, uint32_t psns_to_get)
{
	assert(psns_to_get && psns_to_get <= MAX(OPX_HFI1_SDMA_MAX_PKTS_TID, OPX_HFI1_SDMA_MAX_PKTS));
	uint32_t psn = 0;

	union fi_opx_reliability_service_flow_key key = {
		.slid	= (uint32_t) state->lid,
		.src_rx = (uint32_t) state->rx,
		.dlid	= (uint32_t) lid,
		.dst_rx = (uint32_t) rx,
	};

	void *itr = fi_opx_rbt_find(state->tx_flow_rbtree, (void *) key.value);
	if (OFI_UNLIKELY(!itr)) {
		*psn_ptr = opx_reliability_create_tx_flow(state, key.value);
		if (OFI_UNLIKELY((*psn_ptr) == NULL)) {
			return -1;
		}
	} else {
		*psn_ptr = (union fi_opx_reliability_tx_psn *) fi_opx_rbt_value_ptr(state->tx_flow_rbtree, itr);
		union fi_opx_reliability_tx_psn psn_value = **psn_ptr;

		/*
		 * We can leverage the fact athat every packet needs a packet sequence
		 * number before it can be sent to implement some simply throttling.
		 *
		 * If the throttle is on, or if the # of bytes outstanding exceeds
		 * a threshold, return an error.
		 */
		if (OFI_UNLIKELY((*psn_ptr)->psn.throttle != 0)) {
			return -1;
		}
		if (OFI_UNLIKELY((*psn_ptr)->psn.nack_count > fi_opx_reliability_tx_max_nacks())) {
			(*psn_ptr)->psn.throttle = 1;
			fi_opx_reliability_inc_throttle_count(ep);
			fi_opx_reliability_inc_throttle_nacks(ep);
			return -1;
		}
		if (OFI_UNLIKELY((*psn_ptr)->psn.bytes_outstanding > state->service->max_outstanding_bytes)) {
			(*psn_ptr)->psn.throttle = 1;
			fi_opx_reliability_inc_throttle_count(ep);
			fi_opx_reliability_inc_throttle_maxo(ep);
			return -1;
		}

		psn		    = psn_value.psn.psn;
		(*psn_ptr)->psn.psn = (psn_value.psn.psn + psns_to_get) & MAX_PSN;
	}

	return psn;
}
__OPX_FORCE_INLINE__
struct fi_opx_reliability_tx_replay *
fi_opx_reliability_client_replay_allocate(struct fi_opx_reliability_client_state *state, const bool use_iov)
{
	struct fi_opx_reliability_tx_replay *return_value;

	if (!use_iov) {
		return_value = (struct fi_opx_reliability_tx_replay *) ofi_buf_alloc(state->replay_pool);
	} else {
		return_value = (struct fi_opx_reliability_tx_replay *) ofi_buf_alloc(state->replay_iov_pool);
	}
	if (OFI_LIKELY(return_value != NULL)) {
		return_value->nack_count = 0;
		return_value->pinned	 = false;
		return_value->acked	 = false;
		return_value->use_sdma	 = false;
		return_value->use_iov	 = use_iov;
		return_value->sdma_we	 = NULL;
#ifndef NDEBUG
		memset(return_value->orig_payload, 0x2B, 64);
#endif
		return_value->hmem_iface  = FI_HMEM_SYSTEM;
		return_value->hmem_device = 0;

		// This will implicitly set return_value->iov correctly
		return_value->payload = (uint64_t *) &return_value->data;
	}

	return return_value;
}

__OPX_FORCE_INLINE__
int32_t fi_opx_reliability_get_replay(struct fid_ep *ep, struct fi_opx_reliability_client_state *state,
				      const opx_lid_t lid, const uint64_t rx, union fi_opx_reliability_tx_psn **psn_ptr,
				      struct fi_opx_reliability_tx_replay **replay,
				      const enum ofi_reliability_kind reliability, const enum opx_hfi1_type hfi1_type)
{
	union fi_opx_reliability_service_flow_key key = {
		.slid	= (uint32_t) state->lid,
		.src_rx = (uint32_t) state->rx,
		.dlid	= (uint32_t) lid,
		.dst_rx = (uint32_t) rx,
	};

	void *itr = fi_opx_rbt_find(state->tx_flow_rbtree, (void *) key.value);
	if (OFI_UNLIKELY(!itr)) {
		*psn_ptr = opx_reliability_create_tx_flow(state, key.value);
		if (OFI_UNLIKELY((*psn_ptr) == NULL)) {
			return -1;
		}
	} else {
		*psn_ptr = (union fi_opx_reliability_tx_psn *) fi_opx_rbt_value_ptr(state->tx_flow_rbtree, itr);
	}
	union fi_opx_reliability_tx_psn psn_value = **psn_ptr;

	/*
	 * We can leverage the fact athat every packet needs a packet sequence
	 * number before it can be sent to implement some simply throttling.
	 *
	 * If the throttle is on, or if the # of bytes outstanding exceeds
	 * a threshold, return an error.
	 */
	if (OFI_UNLIKELY((*psn_ptr)->psn.throttle != 0)) {
		return -1;
	}
	if (OFI_UNLIKELY((*psn_ptr)->psn.nack_count > fi_opx_reliability_tx_max_nacks())) {
		(*psn_ptr)->psn.throttle = 1;
		fi_opx_reliability_inc_throttle_count(ep);
		fi_opx_reliability_inc_throttle_nacks(ep);
		return -1;
	}
	if (OFI_UNLIKELY((*psn_ptr)->psn.bytes_outstanding > state->service->max_outstanding_bytes)) {
		(*psn_ptr)->psn.throttle = 1;
		fi_opx_reliability_inc_throttle_count(ep);
		fi_opx_reliability_inc_throttle_maxo(ep);
		return -1;
	}

	*replay = fi_opx_reliability_client_replay_allocate(state, false);
	if (*replay == NULL) {
		return -1;
	}

	uint32_t psn;
	psn		    = psn_value.psn.psn;
	(*psn_ptr)->psn.psn = (psn_value.psn.psn + 1) & MAX_PSN;

	return psn;
}

static inline void fi_opx_reliability_client_replay_deallocate(struct fi_opx_reliability_client_state *state
							       __attribute__((unused)),
							       struct fi_opx_reliability_tx_replay *replay)
{
#ifdef OPX_RELIABILITY_DEBUG
	replay->next = replay->prev = 0;
#endif
	OPX_BUF_FREE(replay);
}

__OPX_FORCE_INLINE__
void fi_opx_reliability_client_replay_register_with_update(struct fi_opx_reliability_client_state *state,
							   const uint8_t rx, union fi_opx_reliability_tx_psn *psn_ptr,
							   struct fi_opx_reliability_tx_replay *replay,
							   struct fi_opx_completion_counter *counter, uint64_t value,
							   const enum ofi_reliability_kind reliability_kind,
							   const enum opx_hfi1_type	   hfi1_type)
{
	uint16_t  lrh_pktlen_le;
	size_t	  total_bytes;
	opx_lid_t hdr_dlid;
	uint8_t	  hdr_src_rx;
	uint8_t	  hdr_dst_rx;

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		lrh_pktlen_le = ntohs(replay->scb.scb_9B.hdr.lrh_9B.pktlen);
		total_bytes   = (lrh_pktlen_le - 1) * 4; /* do not copy the trailing icrc */
		hdr_dlid      = (opx_lid_t) __be16_to_cpu24((__be16) replay->scb.scb_9B.hdr.lrh_9B.dlid);
		hdr_src_rx    = FI_OPX_HFI1_PACKET_ORIGIN_RX(&replay->scb.scb_9B.hdr);
		hdr_dst_rx    = replay->scb.scb_9B.hdr.bth.rx;
	} else {
		lrh_pktlen_le = replay->scb.scb_16B.hdr.lrh_16B.pktlen;
		total_bytes   = (lrh_pktlen_le - 1) * 8; /* do not copy the trailing icrc */
		hdr_dlid      = (opx_lid_t) __le24_to_cpu(replay->scb.scb_16B.hdr.lrh_16B.dlid20 << 20 |
							  replay->scb.scb_16B.hdr.lrh_16B.dlid);
		hdr_src_rx    = FI_OPX_HFI1_PACKET_ORIGIN_RX(&replay->scb.scb_16B.hdr);
		hdr_dst_rx    = replay->scb.scb_16B.hdr.bth.rx;
	}

	psn_ptr->psn.bytes_outstanding += total_bytes;
	replay->target_reliability_rx = rx;
	replay->psn_ptr		      = psn_ptr;
	replay->cc_ptr		      = counter;
	replay->cc_dec		      = value;

#ifndef NDEBUG
	if (replay->use_iov && replay->hmem_iface == FI_HMEM_SYSTEM) {
		/* Copy up to 64 bytes of the current payload value for
		 * later comparison as a sanity check to make sure the
		 * user didn't alter the buffer */
		uint32_t copy_payload_qws = MIN(8, value >> 3);
		for (int i = 0; i < copy_payload_qws; ++i) {
			replay->orig_payload[i] = ((uint64_t *) replay->iov->iov_base)[i];
		}
	}
#endif
	opx_reliability_service_append_replay(state, replay, state->lid, hdr_dlid, hdr_src_rx, hdr_dst_rx, hfi1_type);
}

__OPX_FORCE_INLINE__
void fi_opx_reliability_client_replay_register_no_update(struct fi_opx_reliability_client_state *state,
							 const uint8_t rx, union fi_opx_reliability_tx_psn *psn_ptr,
							 struct fi_opx_reliability_tx_replay *replay,
							 const enum ofi_reliability_kind      reliability_kind,
							 const enum opx_hfi1_type	      hfi1_type)
{
	// If this replay is pointing to an outside buffer instead of us copying
	// the payload into the replay's bounce buffer, there needs to be a completion
	// counter associated with the replay, and replay_register_with_update should
	// be used instead.
	assert(!replay->use_iov);

	fi_opx_reliability_client_replay_register_with_update(state, rx, psn_ptr, replay, NULL, 0UL, reliability_kind,
							      hfi1_type);
}

ssize_t fi_opx_reliability_service_do_replay_sdma(struct fid_ep *ep, struct fi_opx_reliability_service *service,
						  struct fi_opx_reliability_tx_replay *start,
						  struct fi_opx_reliability_tx_replay *stop, uint32_t num_replays);

ssize_t fi_opx_reliability_service_do_replay(struct fi_opx_reliability_service	 *service,
					     struct fi_opx_reliability_tx_replay *replay);

void fi_opx_hfi_rx_reliablity_process_requests(struct fid_ep *ep, int max_to_send);

ssize_t fi_opx_reliability_do_remote_ep_resynch(struct fid_ep *ep, union fi_opx_addr dest_addr, void *context,
						const uint64_t caps);

#endif /* _FI_PROV_OPX_RELIABILITY_H_ */
