/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2024 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_ENDPOINT_H_
#define _FI_PROV_OPX_ENDPOINT_H_

#include <stdint.h>
#include <pthread.h>
#include <sys/uio.h>

#include "rdma/opx/fi_opx_domain.h"
#include "rdma/opx/fi_opx_internal.h"
#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_compiler.h"
#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx_reliability.h"
#include "rdma/opx/fi_opx_rma_ops.h"
#include "rdma/opx/fi_opx_match.h"
#include "rdma/opx/fi_opx_addr.h"
#include "rdma/opx/opx_tracer.h"
#include "rdma/opx/fi_opx_debug_counters.h"
#include "rdma/opx/fi_opx_flight_recorder.h"
#include "opx_shm.h"
#include "fi_opx_tid_cache.h"

void fi_opx_cq_debug(struct fid_cq *cq, char *func, const int line);

#define FI_OPX_KIND_TAG	(0)
#define FI_OPX_KIND_MSG	(1)

#define OPX_INTRANODE_TRUE		(1)
#define OPX_INTRANODE_FALSE		(0)

#define OPX_CONTIG_TRUE			(1)
#define OPX_CONTIG_FALSE		(0)

#define OPX_FLAGS_OVERRIDE_TRUE		(1)
#define OPX_FLAGS_OVERRIDE_FALSE	(0)

#define OPX_MULTI_RECV_TRUE		(1)
#define OPX_MULTI_RECV_FALSE		(0)

#define OPX_HMEM_TRUE			(1)
#define OPX_HMEM_FALSE			(0)


// #define FI_OPX_TRACE 1
// #define FI_OPX_REMOTE_COMPLETION

/* #define IS_MATCH_DEBUG */

/* Macro for declaring a compile/constant fi_opx_addr based on AV type.
 * const FI_AV_MAP/FI_AV_TABLE compile optimized.
 * const FI_AV_UNSPEC requires a conditional pulling it out of the endpoint   */

#define FI_OPX_EP_AV_ADDR(const_av_type, ep, addr)				\
{										\
	(const_av_type == FI_AV_TABLE) ? ep->tx->av_addr[addr].fi :		\
		( (const_av_type == FI_AV_MAP) ? addr :				\
		( (ep->av_type == FI_AV_TABLE) ?				\
		ep->tx->av_addr[addr].fi : addr) )				\
}

/* Macro indirection in order to support other macros as arguments
 * C requires another indirection for expanding macros since
 * operands of the token pasting operator are not expanded */

#define FI_OPX_MSG_SPECIALIZED_FUNC(LOCK,AV,CAPS,RELIABILITY,HFI1_TYPE)			\
	FI_OPX_MSG_SPECIALIZED_FUNC_(LOCK,AV,CAPS,RELIABILITY,HFI1_TYPE)

#define FI_OPX_MSG_SPECIALIZED_FUNC_(LOCK,AV,CAPS,RELIABILITY,HFI1_TYPE)		\
	static inline ssize_t							\
	fi_opx_send_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE	\
		(struct fid_ep *ep, const void *buf, size_t len,		\
			void *desc, fi_addr_t dest_addr, void *context)		\
	{									\
		return fi_opx_ep_tx_send(ep, buf, len, desc,			\
				dest_addr, 0, context, 0,			\
				LOCK,	/* lock_required */			\
				AV,	/* av_type */				\
				1,	/* is_contiguous */			\
				0,	/* override_flags */			\
				0,	/* flags */				\
				CAPS | FI_MSG,					\
				RELIABILITY,					\
				HFI1_TYPE);					\
	}									\
	static inline ssize_t							\
	fi_opx_recv_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE	\
		(struct fid_ep *ep, void *buf, size_t len,			\
			void *desc, fi_addr_t src_addr, void *context)		\
	{									\
		return fi_opx_recv_generic(ep, buf, len, desc,			\
				src_addr, 0, (uint64_t)-1, context,		\
				LOCK, AV, FI_MSG, RELIABILITY, HFI1_TYPE);	\
	}									\
	static inline ssize_t							\
	fi_opx_inject_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE	\
		(struct fid_ep *ep, const void *buf, size_t len,		\
			fi_addr_t dest_addr)					\
	{									\
		return fi_opx_ep_tx_inject(ep, buf, len,			\
				dest_addr, 0, 0,				\
				LOCK,	/* lock_required */			\
				AV,	/* av_type */				\
				0,	/* flags */				\
				CAPS | FI_MSG,					\
				RELIABILITY,					\
				HFI1_TYPE);					\
	}									\
	static inline ssize_t							\
	fi_opx_recvmsg_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY	## _ ## HFI1_TYPE\
		(struct fid_ep *ep, const struct fi_msg *msg,			\
			uint64_t flags)						\
	{									\
		return fi_opx_recvmsg_generic(ep, msg, flags,			\
				LOCK, AV, RELIABILITY, HFI1_TYPE);		\
	}									\
	static inline ssize_t							\
	fi_opx_senddata_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE	\
		(struct fid_ep *ep, const void *buf, size_t len,		\
			void *desc, uint64_t data, fi_addr_t dest_addr,		\
			void *context)						\
	{									\
		return fi_opx_ep_tx_send(ep, buf, len, desc,			\
				dest_addr, 0, context, data,			\
				LOCK,	/* lock_required */			\
				AV,	/* av_type */				\
				1,	/* is_contiguous */			\
				0,	/* override_flags */			\
				FI_REMOTE_CQ_DATA,	/* flags */		\
				CAPS | FI_MSG,					\
				RELIABILITY,					\
				HFI1_TYPE);					\
	}									\
	static inline ssize_t							\
	fi_opx_injectdata_ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE	\
		(struct fid_ep *ep, const void *buf, size_t len,		\
			uint64_t data, fi_addr_t dest_addr)			\
	{									\
		return fi_opx_ep_tx_inject(ep, buf, len,			\
				dest_addr, 0, data,				\
				LOCK,	/* lock_required */			\
				AV,	/* av_type */				\
				FI_REMOTE_CQ_DATA,	/* flags */		\
				CAPS | FI_MSG,					\
				RELIABILITY,					\
				HFI1_TYPE);					\
	}

#define FI_OPX_MSG_SPECIALIZED_FUNC_NAME(TYPE, LOCK, AV, CAPS, RELIABILITY,HFI1_TYPE)	\
	FI_OPX_MSG_SPECIALIZED_FUNC_NAME_(TYPE, LOCK, AV, CAPS, RELIABILITY,HFI1_TYPE)

#define FI_OPX_MSG_SPECIALIZED_FUNC_NAME_(TYPE, LOCK, AV, CAPS, RELIABILITY,HFI1_TYPE)	\
		fi_opx_ ## TYPE ## _ ## LOCK ## _ ## AV ## _ ## CAPS ## _ ## RELIABILITY ## _ ## HFI1_TYPE




enum fi_opx_ep_state {
	FI_OPX_EP_UNINITIALIZED = 0,
	FI_OPX_EP_INITITALIZED_DISABLED,
	FI_OPX_EP_INITITALIZED_ENABLED
};

enum opx_work_type {
	OPX_WORK_TYPE_SDMA = 0,		// SDMA should always be first value in enum
	OPX_WORK_TYPE_PIO,
	OPX_WORK_TYPE_SHM,
	OPX_WORK_TYPE_TID_SETUP,
	OPX_WORK_TYPE_LAST
};

OPX_COMPILE_TIME_ASSERT(OPX_WORK_TYPE_SDMA == 0,
			"OPX_WORK_TYPE_SDMA needs to be 0/first value in the enum!");

static const char * const OPX_WORK_TYPE_STR[] = {
	[OPX_WORK_TYPE_SDMA] = "SDMA",
	[OPX_WORK_TYPE_PIO] = "PIO",
	[OPX_WORK_TYPE_SHM] = "SHM",
	[OPX_WORK_TYPE_TID_SETUP] = "TID_SETUP",
	[OPX_WORK_TYPE_LAST] = "LAST"
};

struct fi_opx_stx {

	/* == CACHE LINE 0,1,2 == */

	struct fid_stx				stx_fid;	/* 80 bytes */
	struct fi_opx_domain *			domain;
	struct fi_tx_attr			attr;		/* 72 bytes */
	struct fi_opx_hfi1_context *		hfi;
	uint64_t				unused_cacheline_2[2];

	/* == CACHE LINE 3 == */

	struct fi_opx_reliability_client_state	reliability_state;	/* 56 bytes */
	struct fi_opx_reliability_service	reliability_service;	/* ONLOAD only */
	uint8_t					reliability_rx;		/* ONLOAD only */

	/* == CACHE LINE 4-9 == */

	struct {
		struct fi_opx_hfi1_txe_scb_9B	inject;
		struct fi_opx_hfi1_txe_scb_9B	send;
		struct fi_opx_hfi1_txe_scb_9B	rzv;
		struct fi_opx_hfi1_txe_scb_16B  inject_16B;
		struct fi_opx_hfi1_txe_scb_16B  send_16B;
		struct fi_opx_hfi1_txe_scb_16B  rzv_16B;
	} tx;

	/* == CACHE LINE 10 == */

	struct fi_opx_hfi1_rxe_state		rxe_state;	/* ignored for ofi tx */
	int64_t					ref_cnt;
};


/*
 * This structure layout ensures that the 'fi_tinject()' function will only
 * touch 2 cachelines - one from this structure and one to obtain the pio
 * state information.
 *
 * This structure layout ensures that the 'fi_tsend()' function will only
 * touch 3 cachelines - two from this structure and one to obtain the pio
 * state information. Additional cachelines will be touched if a completion
 * queue entry is requested.
 *
 * 'fi_inject()' -> 3 cachelines
 * 'fi_send()'   -> 4 cachelines
 */
struct fi_opx_ep_tx {

	/* == CACHE LINE 0 == */

	volatile union fi_opx_hfi1_pio_state	*pio_state;			/* 1 qw = 8 bytes */
	volatile uint64_t *			pio_scb_sop_first;
	uint32_t				sdma_bounce_buf_threshold;
	uint16_t 				pio_max_eager_tx_bytes;
	uint16_t 				pio_flow_eager_tx_bytes;

	volatile uint64_t *			pio_credits_addr;		/* const; only used to infrequently "refresh" credit information */
	volatile uint64_t *			pio_scb_first;			/* const; only eager and rendezvous */
	uint64_t				cq_bind_flags;
	struct slist *				cq_completed_ptr;
	uint32_t				do_cq_completion;
	uint16_t 				unused_cacheline1;
	uint8_t					force_credit_return;
	uint8_t					use_sdma;

	/* == CACHE LINE 1,2 == */
	struct fi_opx_hfi1_txe_scb_9B		inject_9B;				/* qws 5,6, and 7 specified at runtime */

	/* == CACHE LINE 3,4 == */
	struct fi_opx_hfi1_txe_scb_9B		send_9B;

	/* == CACHE LINE 5,6 == */
	struct fi_opx_hfi1_txe_scb_9B		rzv_9B;

	/* == CACHE LINE 7,8 == */
	struct fi_opx_hfi1_txe_scb_16B		inject_16B;

	/* == CACHE LINE 9,10 == */
	struct fi_opx_hfi1_txe_scb_16B		send_16B;

	/* == CACHE LINE 11,12 == */
	struct fi_opx_hfi1_txe_scb_16B		rzv_16B;

	/* == CACHE LINE 13 == */

	union fi_opx_addr *			av_addr;			/* only FI_ADDR_TABLE */
	uint64_t				av_count;			/* only FI_ADDR_TABLE */
	uint64_t				op_flags;
	uint64_t				caps;
	uint64_t				mode;
	struct slist *				cq_err_ptr;
	struct fi_opx_cq *			cq;
	struct slist *				cq_pending_ptr;			/* only rendezvous (typically) */

	/* == CACHE LINE 14 == */

	struct slist				work_pending[OPX_WORK_TYPE_LAST];

	/* == CACHE LINE 15 == */

	struct slist				work_pending_completion;
	struct ofi_bufpool			*work_pending_pool;
	struct ofi_bufpool			*rma_payload_pool;
	struct ofi_bufpool			*rma_request_pool;
	struct ofi_bufpool			*sdma_work_pool;
	uint32_t				sdma_min_payload_bytes;
	uint32_t				tid_min_payload_bytes;
	uint32_t				rzv_min_payload_bytes;
	uint16_t				mp_eager_max_payload_bytes;
	uint16_t				unused_cacheline6;

	/* == CACHE LINE 16 == */
	struct opx_sdma_queue			sdma_request_queue;
	struct slist				sdma_pending_queue;
	struct ofi_bufpool			*sdma_request_pool;
	uint64_t				unused_cacheline7[2];

	/* == CACHE LINE 17, ... == */
	int64_t					ref_cnt;
	struct fi_opx_stx			*stx;
	// struct opx_shm_tx is very large and should go last!
	struct opx_shm_tx			shm;
	void					*mem;
} __attribute__((__aligned__(L2_CACHE_LINE_SIZE))) __attribute__((__packed__));

OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, inject_9B) == (FI_OPX_CACHE_LINE_SIZE * 1),
			"Offset of fi_opx_ep_tx->inject_9B should start at cacheline 1!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, send_9B) == (FI_OPX_CACHE_LINE_SIZE * 3),
			"Offset of fi_opx_ep_tx->send_9B should start at cacheline 3!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, rzv_9B) == (FI_OPX_CACHE_LINE_SIZE * 5),
			"Offset of fi_opx_ep_tx->rzv_9B should start at cacheline 5!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, inject_16B) == (FI_OPX_CACHE_LINE_SIZE * 7),
			"Offset of fi_opx_ep_tx->inject_16B should start at cacheline 7!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, send_16B) == (FI_OPX_CACHE_LINE_SIZE * 9),
			"Offset of fi_opx_ep_tx->send_16B should start at cacheline 9!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, rzv_16B) == (FI_OPX_CACHE_LINE_SIZE * 11),
			"Offset of fi_opx_ep_tx->rzv_16B should start at cacheline 11!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, av_addr) == (FI_OPX_CACHE_LINE_SIZE * 13),
			"Offset of fi_opx_ep_tx->av_addr should start at cacheline 13!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, work_pending) == (FI_OPX_CACHE_LINE_SIZE * 14),
			"Offset of fi_opx_ep_tx->work_pending should start at cacheline 14!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, work_pending_completion) == (FI_OPX_CACHE_LINE_SIZE * 15),
			"Offset of fi_opx_ep_tx->work_pending_completion should start at cacheline 15!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, sdma_request_queue) == (FI_OPX_CACHE_LINE_SIZE * 16),
			"Offset of fi_opx_ep_tx->sdma_request_queue should start at cacheline 16!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_tx, ref_cnt) == (FI_OPX_CACHE_LINE_SIZE * 17),
			"Offset of fi_opx_ep_tx->ref_cnt should start at cacheline 17!");


struct fi_opx_ep_rx {

	/* == CACHE LINE 0 == */

	/*
	 * NOTE: This cacheline is used when a thread is INITIATING
	 * receive operations
	 */
	uint64_t			op_flags;
	uint64_t			unused_cacheline_0[5];
	uint64_t			av_count;
	union fi_opx_addr *		av_addr;

	/*
	 * NOTE: The following 2 cachelines are shared between the application-facing
	 * functions, such as 'fi_trecv()', and the progress functions, such as
	 * those invoked during 'fi_cq_read()'.
	 */

	/* == CACHE LINE 1 & 2 == */

	struct {
		struct fi_opx_hfi1_ue_packet_slist	ue;	/* 3 qws */
		struct slist				mq;	/* 2 qws */
	} queue[2];	/* 0 = FI_TAGGED, 1 = FI_MSG */

	struct {
		struct fi_opx_hfi1_ue_packet_slist	ue;	/* 3 qws */
		struct slist				mq;	/* 2 qws */
	} mp_egr_queue;

	struct fi_opx_match_ue_hash *			match_ue_tag_hash;

	/* == CACHE LINE 3 == */
	struct slist *					cq_pending_ptr;
	struct slist *					cq_completed_ptr;
	struct ofi_bufpool *				ue_packet_pool;
	struct ofi_bufpool *				ctx_pool;

	uint64_t					unused_cacheline_3[4];
	/* == CACHE LINE 4 == */

	/*
	 * NOTE: This cacheline is used when a thread is making PROGRESS to
	 * process fabric events.
	 */
	struct fi_opx_hfi1_rxe_state	state;			/* 2 qws */

	struct {
		uint32_t *		rhf_base;
		uint64_t *		rhe_base;
		volatile uint64_t *	head_register;
	} hdrq;

	struct {
		uint32_t *		base_addr;
		uint32_t		elemsz;
		uint32_t		last_egrbfr_index;
		volatile uint64_t *	head_register;
	} egrq __attribute__((__packed__));

	/* == CACHE LINE 5,6,7,8,9,10,11,12 == */

	/*
	 * NOTE: These cachelines are shared between the application-facing
	 * functions, such as 'fi_trecv()', and the progress functions, such as
	 * those invoked during 'fi_cq_read()'.
	 *
	 * This 'tx' information is used when sending acks, etc.
	 */
	struct {
		struct fi_opx_hfi1_txe_scb_9B	dput_9B;
		struct fi_opx_hfi1_txe_scb_9B	cts_9B;
		struct fi_opx_hfi1_txe_scb_16B	dput_16B;
		struct fi_opx_hfi1_txe_scb_16B	cts_16B;
	} tx;

	/* -- non-critical -- */
	uint64_t			min_multi_recv;
	struct fi_opx_domain		*domain;

	uint64_t			caps;
	uint64_t			mode;
	union fi_opx_addr		self;

	struct slist			*cq_err_ptr;
	struct fi_opx_cq		*cq;

	struct opx_shm_rx		shm;
	void				*mem;
	int64_t				ref_cnt;
	//ofi_spin_t			lock;

} __attribute__((__aligned__(L2_CACHE_LINE_SIZE))) __attribute__((__packed__));

OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_rx, queue) == FI_OPX_CACHE_LINE_SIZE,
			"Offset of fi_opx_ep_rx->queue should start at cacheline 1!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_rx, cq_pending_ptr) == (FI_OPX_CACHE_LINE_SIZE * 3),
			"Offset of fi_opx_ep_rx->queue should start at cacheline 3!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_rx, state) == (FI_OPX_CACHE_LINE_SIZE * 4),
			"Offset of fi_opx_ep_rx->queue should start at cacheline 4!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep_rx, tx) == (FI_OPX_CACHE_LINE_SIZE * 5),
			"Offset of fi_opx_ep_rx->tx should start at cacheline 5!");


struct fi_opx_ep_reliability {
	struct fi_opx_reliability_client_state	state;		/* 14 qws = 112 bytes */
	struct fi_opx_reliability_service	service;	/* ONLOAD only */
	uint8_t					rx;		/* ONLOAD only */
	void					*mem;
	int64_t					ref_cnt;
};


struct fi_opx_daos_av_rank_key {
	uint32_t rank;
	uint32_t rank_inst;
};


struct fi_opx_daos_av_rank {
	struct fi_opx_daos_av_rank_key key;
	uint32_t updated;
	fi_addr_t fi_addr;
	UT_hash_handle hh;	/* makes this structure hashable */
};


struct fi_opx_ep_daos_info {
	struct fi_opx_daos_av_rank * av_rank_hashmap;
	uint32_t rank_inst;
	uint32_t rank;
	bool do_resynch_remote_ep;
	bool hfi_rank_enabled;
} __attribute__((__packed__));

/*
 * The 'fi_opx_ep' struct defines an endpoint with a single tx context and a
 * single rx context. The tx context is only valid if the FI_READ, FI_WRITE,
 * or FI_SEND capability is specified. The rx context is only valid if the
 * FI_RECV, FI_REMOTE_READ, or FI_REMOTE_WRITE flags are specified.
 *
 * A 'scalable tx context' is simply an endpoint structure with only the
 * tx flags specified, and a 'scalable rx context' is simply an endpoint
 * structure with only the rx flags specified.
 *
 * As such, multiple OFI 'classes' share this endpoint structure:
 *   FI_CLASS_EP
 *   FI_CLASS_TX_CTX
 *   --- no FI_CLASS_STX_CTX
 *   FI_CLASS_RX_CTX
 *   -- no FI_CLASS_SRX_CTX
 */
struct fi_opx_ep {
	/* == CACHE LINE 0,1 == */
	struct fid_ep				ep_fid;		/* 10 qws */
	struct fi_opx_ep_tx			*tx;
	struct fi_opx_ep_rx			*rx;
	struct fi_opx_ep_reliability		*reliability;
	struct fi_opx_cntr			*read_cntr;
	struct fi_opx_cntr			*write_cntr;
	struct fi_opx_cntr			*send_cntr;

	/* == CACHE LINE 2 == */
	struct fi_opx_cntr			*recv_cntr;
	struct fi_opx_domain			*domain;
	struct opx_tid_domain			*tid_domain;
	struct ofi_bufpool			*rma_counter_pool;
	struct ofi_bufpool			*rzv_completion_pool;
	void					*mem;
	struct fi_opx_av			*av;
	struct fi_opx_sep			*sep;

	/* == CACHE LINE 3 == */
	struct fi_opx_hfi1_context		*hfi;
	uint8_t					*hmem_copy_buf;

	int					sep_index;
	enum fi_opx_ep_state			state;

	uint32_t				threading;
	uint32_t				av_type;
	uint32_t				mr_mode;
	enum fi_ep_type				type;
	uint64_t				unused_cacheline3[3];

	/* == CACHE LINE 4 == */
	// Only used for initialization
	// free these flags
	struct fi_info				*common_info;
	struct fi_info				*tx_info;
	struct fi_info				*rx_info;
	struct fi_opx_cq			*init_tx_cq;
	struct fi_opx_cq			*init_rx_cq;
	struct fi_opx_cntr			*init_read_cntr;
	struct fi_opx_cntr			*init_write_cntr;
	uint64_t				rx_cq_bflags;

	/* == CACHE LINE 5 == */
	struct fi_opx_cntr			*init_send_cntr;
	struct fi_opx_cntr			*init_recv_cntr;
	uint64_t				tx_cq_bflags;
	struct fi_opx_ep_daos_info		daos_info;	/* 18 bytes */
	bool					is_tx_cq_bound;
	bool					is_rx_cq_bound;
	bool					use_expected_tid_rzv;
	uint8_t					unused_cacheline5[3];
	uint32_t				unused_cacheline5_u32;
	ofi_spin_t				lock; /* lock size varies based on ENABLE_DEBUG*/

	/* == CACHE LINE 6 (if ENABLE_DEBUG) == */

#ifdef FLIGHT_RECORDER_ENABLE
	struct flight_recorder			*fr;
#endif

	FI_OPX_DEBUG_COUNTERS_DECLARE_COUNTERS;

} __attribute((aligned(L2_CACHE_LINE_SIZE)));

OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep, recv_cntr) == (FI_OPX_CACHE_LINE_SIZE * 2),
			"Offset of fi_opx_ep->recv_cntr should start at cacheline 2!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep, hfi) == (FI_OPX_CACHE_LINE_SIZE * 3),
			"Offset of fi_opx_ep->hfi should start at cacheline 3!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep, common_info) == (FI_OPX_CACHE_LINE_SIZE * 4),
			"Offset of fi_opx_ep->hfi should start at cacheline 4!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep, init_send_cntr) == (FI_OPX_CACHE_LINE_SIZE * 5),
			"Offset of fi_opx_ep->init_send_cntr should start at cacheline 5!");
OPX_COMPILE_TIME_ASSERT(offsetof(struct fi_opx_ep, lock) == ((FI_OPX_CACHE_LINE_SIZE * 5)+52),
			"Offset of fi_opx_ep->lock should start before cacheline 6!");


/*
 * A 'scalable endpoint' may not be directly specified in a data movement
 * functions, such as fi_tsend(), as it is only a container for multiple
 * tx and rx contexts.
 *
 * The scalable contexts share certain resources, such as the address vector.
 */
struct fi_opx_sep {
	struct fid_ep		ep_fid;

	struct fi_opx_domain	*domain;
	struct fi_opx_av	*av;
	struct fi_info		*info;
	void			*memptr;
	struct fi_opx_ep	*ep[FI_OPX_ADDR_SEP_RX_MAX];
	struct fi_opx_hfi1_context *hfi1[FI_OPX_ADDR_SEP_RX_MAX];
	struct fi_opx_ep_reliability *reliability[FI_OPX_ADDR_SEP_RX_MAX];
	struct fi_opx_ep_tx *tx[FI_OPX_ADDR_SEP_RX_MAX];
	struct fi_opx_ep_rx *rx[FI_OPX_ADDR_SEP_RX_MAX];

	int64_t		ref_cnt;

} __attribute((aligned(L2_CACHE_LINE_SIZE)));


struct fi_opx_rzv_completion {
	struct opx_context	*context;
	uint64_t		tid_length;
	uint64_t		tid_vaddr;
	uint64_t		byte_counter;
	uint64_t		bytes_accumulated;
};

struct fi_opx_rma_request {
	struct fi_opx_completion_counter	*cc;
	uint64_t				hmem_device;
	enum fi_hmem_iface			hmem_iface;
	uint32_t				padding;
};

/*
 * =========================== begin: no-inline functions ===========================
 */

__attribute__((noinline))
void fi_opx_ep_rx_process_context_noinline (struct fi_opx_ep * opx_ep,
		const uint64_t static_flags,
		struct opx_context *context,
		const uint64_t rx_op_flags,
		const uint64_t is_hmem,
		const int lock_required, const enum fi_av_type av_type,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hf1_type);

void fi_opx_ep_rx_process_header_tag (struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const uint8_t * const payload,
		const size_t payload_bytes,
		const uint8_t opcode,
		const uint8_t origin_rs,
		const unsigned is_intranode,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hf1_type,
		opx_lid_t slid);

void fi_opx_ep_rx_process_header_msg (struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const uint8_t * const payload,
		const size_t payload_bytes,
		const uint8_t opcode,
		const uint8_t origin_rs,
		const unsigned is_intranode,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hf1_type,
		opx_lid_t slid);

void fi_opx_ep_rx_reliability_process_packet (struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const uint8_t * const payload,
		const uint8_t origin_rs);

void fi_opx_ep_rx_append_ue_msg (struct fi_opx_ep_rx * const rx,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const uint32_t rank,
		const uint32_t rank_inst,
		const bool daos_enabled,
		struct fi_opx_debug_counters *debug_counters,
		const opx_lid_t slid);

void fi_opx_ep_rx_append_ue_tag (struct fi_opx_ep_rx * const rx,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const uint32_t rank,
		const uint32_t rank_inst,
		const bool daos_enabled,
		struct fi_opx_debug_counters *debug_counters,
		const opx_lid_t slid);

void fi_opx_ep_rx_append_ue_egr (struct fi_opx_ep_rx * const rx,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const opx_lid_t slid);

int fi_opx_ep_tx_check (struct fi_opx_ep_tx * tx, enum fi_av_type av_type);

/*
 * =========================== end: no-inline functions ===========================
 */

__OPX_FORCE_INLINE__
void fi_opx_ep_clear_credit_return(struct fi_opx_ep *opx_ep) {
	if (OFI_UNLIKELY(opx_ep->tx->force_credit_return)) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					"======================================= Forced a credit return\n");
		opx_ep->tx->force_credit_return = 0;
	}
}

#define FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep) fi_opx_ep_clear_credit_return(opx_ep)


#include "rdma/opx/fi_opx_fabric_transport.h"

#ifdef OPX_DAOS_DEBUG
static void fi_opx_dump_daos_av_addr_rank(struct fi_opx_ep *opx_ep,
	const union fi_opx_addr find_addr, const char *title)
{
	if (opx_ep->daos_info.av_rank_hashmap) {
		struct fi_opx_daos_av_rank *cur_av_rank = NULL;
		struct fi_opx_daos_av_rank *tmp_av_rank = NULL;
		int i = 0, found = 0;

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "%s Dump av_rank_hashmap (rank:%d LID:0x%x fi_addr:0x%08lx)\n",
			title, opx_ep->daos_info.rank, find_addr.lid, find_addr.fi);

		HASH_ITER(hh, opx_ep->daos_info.av_rank_hashmap, cur_av_rank, tmp_av_rank) {
			if (cur_av_rank) {
				union fi_opx_addr addr;
				addr.fi = cur_av_rank->fi_addr;

				if ((addr.lid == find_addr.lid) && (cur_av_rank->key.rank == opx_ep->daos_info.rank)) {
					found = 1;
					FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "Dump av_rank_hashmap[%d] = rank:%d LID:0x%x fi_addr:0x%08lx - Found.\n",
						i++, cur_av_rank->key.rank, addr.lid, addr.fi);
				} else {
					FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "Dump av_rank_hashmap[%d] = rank:%d LID:0x%x fi:0x%08lx.\n",
						i++, cur_av_rank->key.rank, addr.lid, addr.fi);
				}
			}
		}

		if (!found) {
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "Dump av_rank_hashmap - rank:%d LID:0x%x fi_addr:0x%08lx - Not found.\n",
				opx_ep->daos_info.rank, find_addr.lid, find_addr.fi);
		}
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "%s Dump av_rank_hashmap (completed)\n\n", title);
	}
}
#endif

static struct fi_opx_daos_av_rank * fi_opx_get_daos_av_rank(struct fi_opx_ep *opx_ep,
	uint32_t rank, uint32_t rank_inst)
{
	struct fi_opx_daos_av_rank_key key;
	struct fi_opx_daos_av_rank *av_rank = NULL;
	/*
	 * DAOS Persistent Address Support:
	 * No Context Resource Management Framework is supported by OPX to enable
	 * acquiring a context with attributes that exactly match the specified
	 * source address.
	 *
	 * Therefore, treat the source address as an ‘opaque’ ID, so reference the
	 * rank data associated with the source address, which maps to the appropriate
	 * HFI and HFI port.
	 */
	key.rank = rank;
	key.rank_inst = rank_inst;

	HASH_FIND(hh, opx_ep->daos_info.av_rank_hashmap, &key, sizeof(key), av_rank);

#ifdef IS_MATCH_DEBUG
	if (av_rank) {
		union fi_opx_addr addr;

		addr.fi = av_rank->fi_addr;
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"Found AV rank - rank:%d, LID:0x%x, fi_addr:%08lx.\n",
			av_rank->key.rank, addr.lid, addr.fi);
	} else if (opx_ep->daos_info.av_rank_hashmap) {
		struct fi_opx_daos_av_rank *cur_av_rank = NULL;
		struct fi_opx_daos_av_rank *tmp_av_rank = NULL;
		int i = 0;

		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "AV hash lookup of rank %d failed.\n", key.rank);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "GET Dump av_rank_hashmap (rank:%d)\n", key.rank);

		HASH_ITER(hh, opx_ep->daos_info.av_rank_hashmap, cur_av_rank, tmp_av_rank) {
			if (cur_av_rank) {
				union fi_opx_addr addr;
				addr.fi = cur_av_rank->fi_addr;

				FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					"GET Dump av_rank_hashmap[%d] = rank:%d LID:0x%x fi_addr:0x%08lx\n",
					i++, cur_av_rank->key.rank, addr.lid, addr.fi);

				if (cur_av_rank->key.rank == key.rank) {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"AV linear lookup of rank %d succeeded.\n", key.rank);
					return cur_av_rank;
				}
			}
		}
	}
#endif

	return av_rank;
}

__OPX_FORCE_INLINE__
uint64_t fi_opx_ep_is_matching_packet(const uint64_t origin_tag,
				      const opx_lid_t origin_lid,
				      const uint8_t origin_endpoint_id,
				      const uint64_t ignore,
				      const uint64_t target_tag_and_not_ignore,
				      const uint64_t any_addr,
				      const union fi_opx_addr src_addr,
				      struct fi_opx_ep *opx_ep,
				      uint32_t rank, uint32_t rank_inst,
				      const unsigned is_intranode)
{
	const uint64_t origin_tag_and_not_ignore = origin_tag & ~ignore;
	return	(origin_tag_and_not_ignore == target_tag_and_not_ignore) &&
		(
			(any_addr)					||
			((origin_lid == src_addr.lid) && (origin_endpoint_id == src_addr.endpoint_id))		||
			(
				opx_ep->daos_info.hfi_rank_enabled &&
				is_intranode &&
				fi_opx_get_daos_av_rank(opx_ep, rank, rank_inst)
			)
		);

}


__OPX_FORCE_INLINE__
struct fi_opx_hfi1_ue_packet *fi_opx_ep_find_matching_packet(struct fi_opx_ep *opx_ep,
							     struct opx_context *context,
							     const uint64_t kind,
							     const enum opx_hfi1_type hfi1_type)
{
	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.match.default_searches);
	struct fi_opx_hfi1_ue_packet *uepkt = opx_ep->rx->queue[kind].ue.head;

	if (!uepkt) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.match.default_not_found);
		return NULL;
	}

	const union fi_opx_addr src_addr = { .fi = context->src_addr };
	const uint64_t ignore = context->ignore;
	const uint64_t target_tag_and_not_ignore = context->tag & ~ignore;
	const uint64_t any_addr = (context->src_addr == FI_ADDR_UNSPEC);

	while (uepkt && !fi_opx_ep_is_matching_packet(uepkt->tag,
						      uepkt->lid,
						      uepkt->endpoint_id,
						      ignore,
						      target_tag_and_not_ignore,
						      any_addr,
						      src_addr,
						      opx_ep,
						      uepkt->daos_info.rank,
						      uepkt->daos_info.rank_inst,
						      opx_lrh_is_intranode(&(uepkt->hdr), hfi1_type))) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.match.default_misses);
		uepkt = uepkt->next;
	}

	FI_OPX_DEBUG_COUNTERS_INC_COND(uepkt, opx_ep->debug_counters.match.default_hits);
	FI_OPX_DEBUG_COUNTERS_INC_COND(!uepkt, opx_ep->debug_counters.match.default_not_found);

	return uepkt;
}

__OPX_FORCE_INLINE__
uint64_t is_match (struct fi_opx_ep * opx_ep,
	const union opx_hfi1_packet_hdr * const hdr,
	struct opx_context *context,
	uint32_t rank, uint32_t rank_inst,
	unsigned is_intranode,
	const opx_lid_t slid)
{

	const union fi_opx_addr src_addr = { .fi = context->src_addr };
	const uint64_t ignore = context->ignore;
	const uint64_t target_tag = context->tag;
	const uint64_t origin_tag = hdr->match.ofi_tag;
	const uint64_t target_tag_and_not_ignore = target_tag & ~ignore;
	const uint64_t origin_tag_and_not_ignore = origin_tag & ~ignore;

	const uint64_t answer =
		(
			(origin_tag_and_not_ignore == target_tag_and_not_ignore) &&
			(
				(context->src_addr == FI_ADDR_UNSPEC) 	||
				((slid == src_addr.lid)	&& (hdr->reliability.origin_tx == src_addr.endpoint_id))	||
				(
					opx_ep->daos_info.hfi_rank_enabled &&
					is_intranode &&
					fi_opx_get_daos_av_rank(opx_ep, rank, rank_inst)
				)
			)
		);

#ifdef IS_MATCH_DEBUG
	fprintf(stderr, "%s:%s():%d context = %p, context->src_addr = 0x%016lx, context->ignore = 0x%016lx, context->tag = 0x%016lx, src_addr.uid.fi = 0x%08x\n", __FILE__, __func__, __LINE__,
		context, context->src_addr, context->ignore, context->tag, src_addr.uid.fi);
	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		fprintf(stderr, "%s:%s():%d hdr->match.slid = 0x%04x (%u), hdr->match.origin_tx = 0x%02x (%u), origin_lid = 0x%08x, origin_endpoint_id = 0x%x\n", __FILE__, __func__, __LINE__,
			__be16_to_cpu24((__be16)hdr->lrh_9B.slid), __be16_to_cpu24((__be16)hdr->lrh_9B.slid),
			hdr->match.origin_tx, hdr->match.origin_tx, slid, hdr->reliability.origin_tx);
	} else {
		fprintf(stderr, "%s:%s():%d hdr->match.slid = 0x%lx (%u), hdr->match.origin_tx = 0x%02x (%u), origin_lid = 0x%08x, origin_endpoint_id = 0x%x\n", __FILE__, __func__, __LINE__,
			__le24_to_cpu((opx_lid_t)((hdr->lrh_16B.slid20 << 20) | (hdr->lrh_16B.slid))),
			__le24_to_cpu((opx_lid_t)((hdr->lrh_16B.slid20 << 20) | (hdr->lrh_16B.slid))),
			hdr->match.origin_tx, hdr->match.origin_tx, slid, hdr->reliability.origin_tx);
	}
	fprintf(stderr, "%s:%s():%d hdr->match.ofi_tag = 0x%016lx, target_tag_and_not_ignore = 0x%016lx, origin_tag_and_not_ignore = 0x%016lx, FI_ADDR_UNSPEC = 0x%08lx\n", __FILE__, __func__, __LINE__,
		hdr->match.ofi_tag, target_tag_and_not_ignore, origin_tag_and_not_ignore, FI_ADDR_UNSPEC);
	if (opx_ep->daos_info.hfi_rank_enabled && is_intranode) {
		struct fi_opx_daos_av_rank *av_rank =
			fi_opx_get_daos_av_rank(opx_ep, rank, rank_inst);

		if (av_rank) {
			fprintf(stderr, "%s:%s():%d AV - rank %d, rank_inst %d, fi_addr 0x%08lx\n",
				__FILE__, __func__, __LINE__,
				av_rank->key.rank, av_rank->key.rank_inst, av_rank->fi_addr);
		} else {
			fprintf(stderr, "%s:%s():%d AV - Not Found.\n", __FILE__, __func__, __LINE__);
			fprintf(stderr, "%s:%s():%d EP - rank %d, rank_inst %d, fi_addr 0x%08lx\n",
				__FILE__, __func__, __LINE__,
				opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst, av_rank->fi_addr);
		}
	}
	fprintf(stderr, "%s:%s():%d answer = %lu\n", __FILE__, __func__, __LINE__, answer);
#endif
	return answer;
}


__OPX_FORCE_INLINE__
uint32_t fi_opx_ep_get_u32_extended_rx (struct fi_opx_ep * opx_ep,
		const unsigned is_intranode,
		const uint8_t origin_rx) {

	return (is_intranode && opx_ep->daos_info.hfi_rank_enabled) ?
		opx_ep->daos_info.rank : origin_rx;
}

__OPX_FORCE_INLINE__
void fi_opx_enqueue_completed(struct slist *queue, struct opx_context *context, const int lock_required)
{
	assert(!lock_required);
	assert(context);
	context->flags &= ~FI_OPX_CQ_CONTEXT_HMEM;
	slist_insert_tail((struct slist_entry *) context, queue);
}

__OPX_FORCE_INLINE__
void opx_ep_copy_immediate_data(struct fi_opx_ep * opx_ep,
				const union fi_opx_hfi1_rzv_rts_immediate_info immediate_info,
				struct opx_payload_rzv_contig *contiguous,
				const uint64_t immediate_byte_count,
				const uint64_t immediate_qw_count,
				const uint64_t immediate_block,
				const uint64_t immediate_tail,
				const uint64_t immediate_total,
				const size_t xfer_len,
				const uint64_t is_hmem,
				const enum fi_hmem_iface rbuf_iface,
				const uint64_t rbuf_device,
				const uint64_t hmem_handle,
				uint8_t *rbuf_in)
{
	uint8_t *rbuf = is_hmem ? opx_ep->hmem_copy_buf : rbuf_in;

	for (int i = 0; i < immediate_byte_count; ++i) {
		rbuf[i] = contiguous->immediate_byte[i];
	}
	rbuf += immediate_byte_count;

	uint64_t * rbuf_qw = (uint64_t *)rbuf;
	for (int i = 0; i < immediate_qw_count; ++i) {
		rbuf_qw[i] = contiguous->immediate_qw[i];
	}
	rbuf += immediate_qw_count * sizeof(uint64_t);

	if (immediate_block) {
		const uint64_t immediate_fragment = (immediate_byte_count || immediate_qw_count) ? 1 : 0;
		#if (defined __GNUC__) && (__GNUC__ > 10)
			#pragma GCC diagnostic ignored "-Wstringop-overread"
		#endif
		#pragma GCC diagnostic ignored "-Warray-bounds"
		memcpy(rbuf, (void *) (&contiguous->cache_line_1 + immediate_fragment), FI_OPX_CACHE_LINE_SIZE);
	}

	if (is_hmem && immediate_total) {
		opx_copy_to_hmem(rbuf_iface, rbuf_device, hmem_handle,
			rbuf_in, opx_ep->hmem_copy_buf, immediate_total,
			OPX_HMEM_DEV_REG_RECV_THRESHOLD);
	}

	if (immediate_tail) {
		uint8_t *rbuf_start = rbuf_in + xfer_len - OPX_IMMEDIATE_TAIL_BYTE_COUNT;

		if (!is_hmem) {
			for (int i = 0; i < OPX_IMMEDIATE_TAIL_BYTE_COUNT; ++i) {
				rbuf_start[i] = immediate_info.tail_bytes[i];
			}
		} else {
			opx_copy_to_hmem(rbuf_iface, rbuf_device, hmem_handle, rbuf_start,
					immediate_info.tail_bytes, OPX_IMMEDIATE_TAIL_BYTE_COUNT,
					OPX_HMEM_DEV_REG_RECV_THRESHOLD);
		}
	}
}

__OPX_FORCE_INLINE__
void fi_opx_handle_recv_rts(const union opx_hfi1_packet_hdr * const hdr,
			    const union fi_opx_hfi1_packet_payload * const payload,
			    struct fi_opx_ep * opx_ep,
			    const uint64_t origin_tag,
			    const uint8_t opcode,
			    struct opx_context *context,
			    const uint64_t is_multi_receive,
			    const unsigned is_intranode,
			    const uint64_t is_hmem,
			    const int lock_required,
			    const enum ofi_reliability_kind reliability,
			    const enum opx_hfi1_type hfi1_type)
{
	assert(FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- RENDEZVOUS RTS (%X) (begin) context %p is_multi_recv (%lu)\n",
		opcode, context, is_multi_receive);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-RTS");

	const uint64_t ofi_data = hdr->match.ofi_data;
	const uint64_t niov = hdr->rendezvous.niov;
	const uint64_t xfer_len = hdr->rendezvous.message_length;
	const uint64_t is_noncontig = hdr->rendezvous.flags & FI_OPX_PKT_RZV_FLAGS_NONCONTIG;
	void *recv_buf = context->buf;
	struct fi_opx_ep_rx * const rx = opx_ep->rx;
	const uint64_t recv_len = context->len;

	if (is_multi_receive) {		/* compile-time constant expression */
		assert(FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode) == FI_MSG);
		const uint8_t u8_rx = hdr->rendezvous.origin_rx;
		const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->rendezvous.origin_rx);
		struct opx_context * original_multi_recv_context = context;
		context = (struct opx_context *)((uintptr_t)recv_buf - sizeof(struct opx_context));

		assert((((uintptr_t)context) & 0x07) == 0);
		context->flags = FI_RECV | FI_MSG | FI_OPX_CQ_CONTEXT_MULTIRECV;
		context->buf = recv_buf;
		context->len = xfer_len;
		context->data = ofi_data;
		context->tag = 0;	/* tag is not valid for multi-receives */
		context->multi_recv_context = original_multi_recv_context;
		context->byte_counter = xfer_len;
		context->next = NULL;
		uint8_t * rbuf = (uint8_t *)recv_buf;

		if (OFI_LIKELY(is_noncontig)) {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.recv.multi_recv_rzv_noncontig);
			FI_OPX_FABRIC_RX_RZV_RTS(opx_ep,
						hdr,
						payload,
						u8_rx, niov,
						payload->rendezvous.noncontiguous.origin_byte_counter_vaddr,
						context,
						(uintptr_t)(rbuf),		/* receive buffer virtual address */
						FI_HMEM_SYSTEM,			/* receive buffer iface */
						0UL,				/* receive buffer device */
						0UL,				/* immediate_data */
						0UL,				/* immediate_end_block_count */
						&payload->rendezvous.noncontiguous.iov[0],
						FI_OPX_HFI_DPUT_OPCODE_RZV_NONCONTIG,
						is_intranode,
						reliability,			/* compile-time constant expression */
						u32_ext_rx,
						hfi1_type);
		} else {
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.recv.multi_recv_rzv_contig);
			assert(niov == 1);
			struct opx_payload_rzv_contig *contiguous = (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B))
				? (struct opx_payload_rzv_contig *) &payload->rendezvous.contiguous
				: (struct opx_payload_rzv_contig *) &payload->rendezvous.contiguous_16B;
			const union fi_opx_hfi1_rzv_rts_immediate_info immediate_info = {
				.qw0 = contiguous->immediate_info
			};
			const uint64_t immediate_byte_count = (immediate_info.count & OPX_IMMEDIATE_BYTE_COUNT_MASK)
								>> OPX_IMMEDIATE_BYTE_COUNT_SHIFT;
			const uint64_t immediate_qw_count   = (immediate_info.count & OPX_IMMEDIATE_QW_COUNT_MASK)
								>> OPX_IMMEDIATE_QW_COUNT_SHIFT;
			const uint64_t immediate_block      = (immediate_info.count & OPX_IMMEDIATE_BLOCK_MASK)
								>> OPX_IMMEDIATE_BLOCK_SHIFT;
			const uint64_t immediate_tail       = (immediate_info.count & OPX_IMMEDIATE_TAIL_MASK)
								>> OPX_IMMEDIATE_TAIL_SHIFT;
			const uint64_t immediate_total      = immediate_byte_count +
								immediate_qw_count * sizeof(uint64_t) +
								immediate_block * sizeof(union cacheline);

			const struct fi_opx_hmem_iov src_dst_iov[1] = {
				{
					.buf = contiguous->src_vaddr,
					.len = contiguous->src_len,
					.device = contiguous->src_device_id,
					.iface = (enum fi_hmem_iface) contiguous->src_iface
				}
			};

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,"IMMEDIATE  RZV_RTS immediate_total %#lX, immediate_byte_count %#lX, immediate_qw_count %#lX, immediate_block_count %#lX\n",
					 immediate_total, immediate_byte_count, immediate_qw_count, immediate_block);

			context->byte_counter -= immediate_total;

			FI_OPX_FABRIC_RX_RZV_RTS(opx_ep,
						hdr,
						payload,
						u8_rx, niov,
						contiguous->origin_byte_counter_vaddr,
						context,
						(uintptr_t)(rbuf + immediate_total),	/* receive buffer virtual address */
						FI_HMEM_SYSTEM,				/* receive buffer iface */
						0UL,					/* receive buffer device */
						immediate_total,
						immediate_tail,
						src_dst_iov,
						FI_OPX_HFI_DPUT_OPCODE_RZV,
						is_intranode,
						reliability,			/* compile-time constant expression */
						u32_ext_rx,
						hfi1_type);

			opx_ep_copy_immediate_data(opx_ep, immediate_info, contiguous, immediate_byte_count,
						immediate_qw_count, immediate_block, immediate_tail,
						immediate_total, xfer_len, OPX_HMEM_FALSE, FI_HMEM_SYSTEM,
						0ul, OPX_HMEM_NO_HANDLE, rbuf);
		}

		uint64_t bytes_consumed = ((xfer_len + 8) & (~0x07ull)) + sizeof(struct opx_context);
		original_multi_recv_context->len -= bytes_consumed;
		original_multi_recv_context->byte_counter++;  // re-using the byte counter as a "pending flag"
		original_multi_recv_context->tag = (uintptr_t)opx_ep;  // re-using tag to store the ep
		original_multi_recv_context->buf = (void*)((uintptr_t)(original_multi_recv_context->buf) + bytes_consumed);
		assert(context->next == NULL);
		if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
		slist_insert_tail((struct slist_entry *) context, rx->cq_pending_ptr);

	} else if (OFI_LIKELY(xfer_len <= recv_len)) {

		context->len = xfer_len;
		context->data = ofi_data;
		context->tag = origin_tag;
		context->next = NULL;
		context->flags |= FI_RECV |
				FI_OPX_HFI_BTH_OPCODE_GET_CQ_FLAG(opcode) |
				FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode);

		const uint8_t u8_rx = hdr->rendezvous.origin_rx;
		const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->rendezvous.origin_rx);

		if (OFI_LIKELY(niov == 1)) {
			assert(!is_noncontig);

			uint64_t rbuf_device;
			enum fi_hmem_iface rbuf_iface;
			uint64_t hmem_handle;
			if (is_hmem) {		/* Branch should compile out */
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) context->hmem_info_qws;
				rbuf_device = hmem_info->device;
				rbuf_iface = hmem_info->iface;
				hmem_handle = hmem_info->hmem_dev_reg_handle;
				FI_OPX_DEBUG_COUNTERS_INC_COND(is_intranode, opx_ep->debug_counters.hmem.intranode
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.rzv);
				FI_OPX_DEBUG_COUNTERS_INC_COND(!is_intranode, opx_ep->debug_counters.hmem.hfi
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.rzv);
			} else {
				rbuf_device = 0;
				hmem_handle = 0;
				rbuf_iface = FI_HMEM_SYSTEM;
			}
			uint8_t * rbuf = (uint8_t *)recv_buf;

			struct opx_payload_rzv_contig *contiguous = (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B))
				? (struct opx_payload_rzv_contig *) &payload->rendezvous.contiguous
				: (struct opx_payload_rzv_contig *) &payload->rendezvous.contiguous_16B;
			const union fi_opx_hfi1_rzv_rts_immediate_info immediate_info = {
				.qw0 = contiguous->immediate_info
			};
			const uint64_t immediate_byte_count = (immediate_info.count & OPX_IMMEDIATE_BYTE_COUNT_MASK)
								>> OPX_IMMEDIATE_BYTE_COUNT_SHIFT;
			const uint64_t immediate_qw_count   = (immediate_info.count & OPX_IMMEDIATE_QW_COUNT_MASK)
								>> OPX_IMMEDIATE_QW_COUNT_SHIFT;
			const uint64_t immediate_block      = (immediate_info.count & OPX_IMMEDIATE_BLOCK_MASK)
								>> OPX_IMMEDIATE_BLOCK_SHIFT;
			const uint64_t immediate_tail       = (immediate_info.count & OPX_IMMEDIATE_TAIL_MASK)
								>> OPX_IMMEDIATE_TAIL_SHIFT;
			const uint64_t immediate_total      = immediate_byte_count +
								immediate_qw_count * sizeof(uint64_t) +
								immediate_block * sizeof(union cacheline);

			const struct fi_opx_hmem_iov src_dst_iov[1] = {
				{
					.buf = contiguous->src_vaddr,
					.len = contiguous->src_len,
					.device = contiguous->src_device_id,
					.iface = (enum fi_hmem_iface) contiguous->src_iface
				}
			};

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,"IMMEDIATE  RZV_RTS immediate_total %#lX, immediate_byte_count %#lX, immediate_qw_count %#lX, immediate_block_count %#lX\n",
				     immediate_total, immediate_byte_count, immediate_qw_count, immediate_block);
			context->byte_counter = xfer_len - immediate_total;

			FI_OPX_FABRIC_RX_RZV_RTS(opx_ep,
						 hdr,
						 payload,
						 u8_rx, 1,
						 contiguous->origin_byte_counter_vaddr,
						 context,
						 (uintptr_t) (rbuf + immediate_total),
						 rbuf_iface,
						 rbuf_device,
						 immediate_total,
						 immediate_tail,
						 src_dst_iov,
						 FI_OPX_HFI_DPUT_OPCODE_RZV,
						 is_intranode,
						 reliability,		/* compile-time constant expression */
						 u32_ext_rx,
						 hfi1_type);

			opx_ep_copy_immediate_data(opx_ep, immediate_info, contiguous, immediate_byte_count,
						immediate_qw_count, immediate_block, immediate_tail,
						immediate_total, xfer_len, is_hmem, rbuf_iface,
						rbuf_device, hmem_handle, rbuf);
		} else {
			/*fi_opx_hfi1_dump_packet_hdr(hdr, __func__, __LINE__); */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"rendezvous non-contiguous source data not implemented; abort\n");
			abort();
		}

		/* post a pending completion event for the individual receive */
		if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
		slist_insert_tail((struct slist_entry *) context, rx->cq_pending_ptr);

	} else {				/* truncation - unlikely */
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"RENDEZVOUS truncation - xfer_len %lu > recv_len %lu posting error\n", xfer_len, recv_len);

		/* Post a CTS Truncation error (FI_OPX_HFI_DPUT_OPCODE_RZV_ETRUNC) to unblock the Tx of RTS */

		context->len = xfer_len;
		context->data = ofi_data;
		context->tag = origin_tag;
		context->next = NULL;
		context->byte_counter = 0;
		context->flags = FI_RECV | FI_OPX_HFI_BTH_OPCODE_GET_CQ_FLAG(opcode) |
					FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode);
		const uint8_t u8_rx = hdr->rendezvous.origin_rx;
		const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->rendezvous.origin_rx);

		assert(payload != NULL);

		uintptr_t origin_byte_counter_vaddr = is_noncontig ?
				payload->rendezvous.noncontiguous.origin_byte_counter_vaddr :
				(hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) ?
					payload->rendezvous.contiguous.origin_byte_counter_vaddr :
					payload->rendezvous.contiguous_16B.origin_byte_counter_vaddr;
		FI_OPX_FABRIC_RX_RZV_RTS_ETRUNC(opx_ep,
						(const void * const)hdr,
						u8_rx,
						origin_byte_counter_vaddr,
						is_intranode,
						reliability,			/* compile-time constant expression */
						u32_ext_rx, hfi1_type);

		/* Post a E_TRUNC to our local RX error queue because a client called receive
		with too small a buffer.  Tell them about it via the error cq */

		context->err_entry.flags = context->flags;
		context->err_entry.len = recv_len;
		context->err_entry.buf = recv_buf;
		context->err_entry.data = ofi_data;
		context->err_entry.tag = origin_tag;
		context->err_entry.olen = xfer_len - recv_len;
		context->err_entry.err = FI_ETRUNC;
		context->err_entry.prov_errno = 0;
		context->err_entry.err_data = NULL;
		context->err_entry.err_data_size = 0;

		context->byte_counter = 0;
		context->next = NULL;

		/* post an 'error' completion event */
		if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
		slist_insert_tail((struct slist_entry *) context, rx->cq_err_ptr);
	}

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-RTS");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- RENDEZVOUS RTS (end) context %p\n",context);
}

/**
 * \brief Complete a receive operation that has matched the packet header with
 * 		the match information
 *
 * \param[in]		rx	Receive endoint
 * \param[in]		hdr	MU packet header that matched
 * \param[in,out]	entry	Completion entry
 */
__OPX_FORCE_INLINE__
void opx_ep_complete_receive_operation (struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const uint64_t origin_tag,
		struct opx_context *context,
		const uint8_t opcode,
		const uint64_t is_multi_receive,
		const unsigned is_intranode,
		const uint64_t is_hmem,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{
	assert((is_multi_receive && !is_hmem) || !is_multi_receive);

	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	struct fi_opx_ep_rx * const rx = opx_ep->rx;

	const uint64_t recv_len = context->len;
	/*
	 * The context buffer pointer has already been set to the appropriate
	 * value (NULL or receive data buffer) to be returned to the user
	 * application.  The value is based upon the type of receive operation
	 * done by the user application.
	 */
	void * recv_buf = context->buf;

	OPX_DEBUG_PRINT_HDR(hdr, hfi1_type);

	if (FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_INJECT) {

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- INJECT (begin)\n");
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-INJECT");

		const uint64_t ofi_data = hdr->match.ofi_data;
		const uint64_t send_len = hdr->inject.message_length;

		if (is_multi_receive) {		/* branch should compile out */

			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.recv.multi_recv_inject);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"INJECT is_multi_recv\n");

			if (send_len) memcpy(recv_buf, (void*)&hdr->inject.app_data_u8[0], send_len);

			struct opx_context * original_multi_recv_context = context;
			context = (struct opx_context *)((uintptr_t)recv_buf - sizeof(struct opx_context));
			assert((((uintptr_t)context) & 0x07) == 0);

			context->flags = FI_RECV | FI_MSG | FI_OPX_CQ_CONTEXT_MULTIRECV;
			context->buf = recv_buf;
			context->len = send_len;
			context->data = ofi_data;
			context->tag = 0;	/* tag is not valid for multi-receives */
			context->multi_recv_context = original_multi_recv_context;
			context->byte_counter = 0;
			context->next = NULL;

			/* the next 'fi_opx_context' must be 8-byte aligned */
			uint64_t bytes_consumed = ((send_len + 8) & (~0x07ull)) + sizeof(struct opx_context);
			original_multi_recv_context->len -= bytes_consumed;
			original_multi_recv_context->buf = (void*)((uintptr_t)(original_multi_recv_context->buf) + bytes_consumed);

			/* post a completion event for the individual receive */
			if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
			slist_insert_tail((struct slist_entry *) context, rx->cq_completed_ptr);

		} else if (OFI_LIKELY(send_len <= recv_len)) {
			if (is_hmem && send_len) {
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) context->hmem_info_qws;
				opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
						recv_buf, hdr->inject.app_data_u8, send_len,
						OPX_HMEM_DEV_REG_RECV_THRESHOLD);
				FI_OPX_DEBUG_COUNTERS_INC_COND(is_intranode, opx_ep->debug_counters.hmem.intranode
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.inject);
				FI_OPX_DEBUG_COUNTERS_INC_COND(!is_intranode, opx_ep->debug_counters.hmem.hfi
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.inject);
			} else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
				switch (send_len) {
				case 0:
					break;
				case 1:	*((uint8_t*)recv_buf) = hdr->inject.app_data_u8[0];
					break;
				case 2:	*((uint16_t*)recv_buf) = hdr->inject.app_data_u16[0];
					break;
				case 3:	memcpy(recv_buf, (void*)&hdr->inject.app_data_u8[0], send_len);
					break;
				case 4:	*((uint32_t*)recv_buf) = hdr->inject.app_data_u32[0];
					break;
				case 5:
				case 6:
				case 7:	memcpy(recv_buf, (void*)&hdr->inject.app_data_u8[0], send_len);
					break;
				case 8:	*((uint64_t*)recv_buf) = hdr->inject.app_data_u64[0];
					break;
				case 9:
				case 10:
				case 11:
				case 12:
				case 13:
				case 14:
				case 15: memcpy(recv_buf, (void*)&hdr->inject.app_data_u8[0], send_len);
					break;
				case 16:
					((uint64_t*)recv_buf)[0] = hdr->inject.app_data_u64[0];
					((uint64_t*)recv_buf)[1] = hdr->inject.app_data_u64[1];
					break;
				default:
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_CTRL, "Invalid send length for inject: %lu\n", send_len);
					abort();
					break;
				}
#pragma GCC diagnostic pop
			}

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"INJECT send_len %lu <= recv_len %lu; enqueue cq (completed) ofi_data = %ld tag = %ld\n",
					 send_len, recv_len, ofi_data, origin_tag);

			context->flags |= FI_RECV | FI_OPX_HFI_BTH_OPCODE_GET_CQ_FLAG(opcode) |
					FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode);
			context->len = send_len;
			context->data = ofi_data;
			context->tag = origin_tag;
			context->byte_counter = 0;
			context->next = NULL;

			/* post a completion event for the individual receive */
			fi_opx_enqueue_completed(rx->cq_completed_ptr, context, lock_required);

		} else {	/* truncation - unlikely */

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"INJECT truncation - send_len %lu > recv_len %lu posting error\n", send_len, recv_len);

			context->err_entry.flags = context->flags;
			context->err_entry.len = recv_len;
			context->err_entry.buf = recv_buf;
			context->err_entry.data = ofi_data;
			context->err_entry.tag = origin_tag;
			context->err_entry.olen = send_len - recv_len;
			context->err_entry.err = FI_ETRUNC;
			context->err_entry.prov_errno = 0;
			context->err_entry.err_data = NULL;
			context->err_entry.err_data_size = 0;

			context->byte_counter = 0;
			context->next = NULL;

			/* post an 'error' completion event for the receive */
			if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
			slist_insert_tail((struct slist_entry *) context, rx->cq_err_ptr);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-INJECT");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- INJECT (end)\n");

	} else if (FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_EAGER) {

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- EAGER (begin)\n");
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-EAGER");

		const uint64_t ofi_data = hdr->match.ofi_data;
		const uint64_t send_len = hdr->send.xfer_bytes_tail + hdr->send.payload_qws_total * sizeof(uint64_t);

		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			"hdr->send.xfer_bytes_tail = %u, hdr->send.payload_qws_total = %u, send_len = %lu\n",
			hdr->send.xfer_bytes_tail, hdr->send.payload_qws_total, send_len);

		if (is_multi_receive) {		/* branch should compile out */

			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.recv.multi_recv_eager);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"EAGER is_multi_recv\n");

			struct opx_context *original_multi_recv_context = context;

			context = (struct opx_context *)((uintptr_t)recv_buf - sizeof(struct opx_context));
			assert((((uintptr_t)context) & 0x07) == 0);
			context->flags = FI_RECV | FI_MSG | FI_OPX_CQ_CONTEXT_MULTIRECV;
			context->buf = recv_buf;
			context->len = send_len;
			context->data = ofi_data;
			context->tag = 0;	/* tag is not valid for multi-receives */
			context->multi_recv_context = original_multi_recv_context;
			context->byte_counter = 0;
			context->next = NULL;

			if (hdr->send.xfer_bytes_tail) {
				memcpy(recv_buf, (void*)&hdr->send.xfer_tail, hdr->send.xfer_bytes_tail);
				recv_buf = (void*)((uintptr_t)recv_buf + hdr->send.xfer_bytes_tail);
			}

			if (payload) {
				uint64_t * recv_buf_qw = (uint64_t *)recv_buf;
				uint64_t * payload_qw = (uint64_t *)payload;
				unsigned i;
				for (i=0; i<hdr->send.payload_qws_total; ++i) {
					recv_buf_qw[i] = payload_qw[i];
				}
			}

			/* the next 'fi_opx_context' must be 8-byte aligned */
			uint64_t bytes_consumed = ((send_len + 8) & (~0x07ull)) + sizeof(struct opx_context);
			original_multi_recv_context->len -= bytes_consumed;
			original_multi_recv_context->buf = (void*)((uintptr_t)(original_multi_recv_context->buf) + bytes_consumed);

			assert(context->next == NULL);
			/* post a completion event for the individual receive */
			if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
			slist_insert_tail((struct slist_entry *) context, rx->cq_completed_ptr);

		} else if (OFI_LIKELY(send_len <= recv_len)) {

			const size_t xfer_bytes_tail = hdr->send.xfer_bytes_tail;

			if (is_hmem) {
				recv_buf = (void *) opx_ep->hmem_copy_buf;
			}

			if (xfer_bytes_tail) {
				#pragma GCC diagnostic ignored "-Wnonnull"
				memcpy(recv_buf, (void*)&hdr->send.xfer_tail, xfer_bytes_tail);
				recv_buf = (void*)((uint8_t *)recv_buf + xfer_bytes_tail);
			}

			if (send_len != xfer_bytes_tail) {
				uint64_t * recv_buf_qw = (uint64_t *)recv_buf;
				uint64_t * payload_qw = (uint64_t *)payload;
				const unsigned payload_qws_total = hdr->send.payload_qws_total;
				unsigned i;
				for (i=0; i<payload_qws_total; ++i) {
					recv_buf_qw[i] = payload_qw[i];
				}
			}

			if (is_hmem) {
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) context->hmem_info_qws;
				opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
						context->buf, opx_ep->hmem_copy_buf, send_len,
						OPX_HMEM_DEV_REG_RECV_THRESHOLD);
				FI_OPX_DEBUG_COUNTERS_INC_COND(is_intranode, opx_ep->debug_counters.hmem.intranode
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.eager);
				FI_OPX_DEBUG_COUNTERS_INC_COND(!is_intranode, opx_ep->debug_counters.hmem.hfi
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
								? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.eager);
			}

			/* fi_opx_hfi1_dump_packet_hdr(hdr, __func__, __LINE__); */

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"EAGER send_len %lu <= recv_len %lu; enqueue cq (completed), tag %#lX/%#lX, ofi_data %#lX \n", send_len, recv_len, context->tag, origin_tag, ofi_data);

			context->flags |= FI_RECV | FI_OPX_HFI_BTH_OPCODE_GET_CQ_FLAG(opcode) |
					FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode);
			context->len = send_len;
			context->data = ofi_data;
			context->tag = origin_tag;
			context->byte_counter = 0;
			context->next = NULL;

			/* post a completion event for the individual receive */
			fi_opx_enqueue_completed(rx->cq_completed_ptr, context, lock_required);

		} else {	/* truncation - unlikely */

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"EAGER truncation - send_len %lu > recv_len %lu posting error\n", send_len, recv_len);

			context->err_entry.flags = context->flags;
			context->err_entry.len = recv_len;
			context->err_entry.buf = recv_buf;
			context->err_entry.data = ofi_data;
			context->err_entry.tag = origin_tag;
			context->err_entry.olen = send_len - recv_len;
			context->err_entry.err = FI_ETRUNC;
			context->err_entry.prov_errno = 0;
			context->err_entry.err_data = NULL;
			context->err_entry.err_data_size = 0;

			context->byte_counter = 0;
			context->next = NULL;

			/* post an 'error' completion event for the receive */
			if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
			slist_insert_tail((struct slist_entry *) context, rx->cq_err_ptr);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-EAGER");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- EAGER (end)\n");

	} else if (FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_MP_EAGER_FIRST) {

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- MULTI PACKET EAGER FIRST (begin)\n");
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-MP-EAGER-FIRST");

		const uint64_t ofi_data = hdr->match.ofi_data;

		uint64_t payload_qws_total;
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			payload_qws_total = (((uint64_t) ntohs(hdr->lrh_9B.pktlen)) - 15) >> 1;
		} else{
			payload_qws_total = (uint64_t)(hdr->lrh_16B.pktlen - 9);
		}
		const uint64_t packet_payload_len = hdr->mp_eager_first.xfer_bytes_tail + (payload_qws_total << 3);
		const uint64_t payload_total_len = hdr->mp_eager_first.payload_bytes_total & FI_OPX_HFI1_KDETH_VERSION_OFF_MASK;

		assert(packet_payload_len < payload_total_len);

		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "hdr->mp_eager_first.xfer_bytes_tail = %u, "
			"hdr->mp_eager_first.payload_bytes_total = %u, send_len = %lu, xfer_len = %lu\n",
			hdr->mp_eager_first.xfer_bytes_tail,
			hdr->mp_eager_first.payload_bytes_total & FI_OPX_HFI1_KDETH_VERSION_OFF_MASK,
			packet_payload_len, payload_total_len);

		if (OFI_UNLIKELY(is_multi_receive)) {		/* branch should compile out */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Multi-receive used with Multi-packet Eager not implemented. abort.");
			abort();
		} else if (OFI_LIKELY(payload_total_len <= recv_len)) {
#ifndef NDEBUG
			/* For non-optimized builds, fill in the entire buffer area we expect to use.
			   Then as we process multi-packet eager Nth packets, we can check that the
			   buffer area we're writing to contains the filled value. */
			if (!is_hmem) {
				memset(recv_buf, 0xAA, payload_total_len);
			}
#endif
			/* For the first MP-Eager packet, we expect all tail bytes in the packet
			   header to be used, as well as a full payload chunk */

			uint64_t * recv_buf_qw = is_hmem ? (uint64_t *) opx_ep->hmem_copy_buf : (uint64_t *) recv_buf;

			/* Tail size is 16 bytes for an eager first packet */
			recv_buf_qw[0] = hdr->mp_eager_first.xfer_tail[0];
			recv_buf_qw[1] = hdr->mp_eager_first.xfer_tail[1];

			recv_buf_qw += 2;
			uint64_t * payload_qw = (uint64_t *) payload;

			for (unsigned i = 0; i < payload_qws_total; ++i) {
				recv_buf_qw[i] = payload_qw[i];
			}

			context->flags |= FI_RECV | FI_OPX_HFI_BTH_OPCODE_GET_CQ_FLAG(opcode) |
					FI_OPX_HFI_BTH_OPCODE_GET_MSG_FLAG(opcode);
			context->len = payload_total_len;
			context->data = ofi_data;
			context->tag = origin_tag;
			context->byte_counter = payload_total_len - packet_payload_len;
			context->next = NULL;

			if (is_hmem) {
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) context->hmem_info_qws;
				opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
						recv_buf, opx_ep->hmem_copy_buf, packet_payload_len,
						OPX_HMEM_DEV_REG_RECV_THRESHOLD);

				/* MP Eager sends are never intranode */
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hmem.hfi
							.kind[FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)
									? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG]
							.recv.mp_eager);
			}
		} else {	/* truncation - unlikely */

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"EAGER truncation - xfer_len %lu > recv_len %lu posting error\n", payload_total_len, recv_len);

			context->err_entry.flags = context->flags;
			context->err_entry.len = recv_len;
			context->err_entry.buf = recv_buf;
			context->err_entry.data = ofi_data;
			context->err_entry.tag = origin_tag;
			context->err_entry.olen = payload_total_len - recv_len;
			context->err_entry.err = FI_ETRUNC;
			context->err_entry.prov_errno = 0;
			context->err_entry.err_data = NULL;
			context->err_entry.err_data_size = 0;

			context->byte_counter = payload_total_len - packet_payload_len;
			context->next = NULL;
		}
#ifndef NDEBUG
		if (context->byte_counter == 0) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "===================================== RECV -- MULTI PACKET EAGER FIRST UNEXPECTED COMPLETE\n");
		}
#endif

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-MP-EAGER-FIRST");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- MULTI PACKET EAGER FIRST byte counter %lu (end)\n",context->byte_counter);

	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_MP_EAGER_NTH) {

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- MULTI PACKET EAGER NTH (begin)\n");
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-MP-EAGER-NTH");

		uint64_t payload_qws_total;
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B))
			payload_qws_total = (((uint64_t) ntohs(hdr->lrh_9B.pktlen)) - 15) >> 1;
		else
			payload_qws_total = (uint64_t) hdr->lrh_16B.pktlen - 9;
		const uint64_t send_len = hdr->mp_eager_nth.xfer_bytes_tail + (payload_qws_total << 3);
		const uint64_t xfer_len = send_len + hdr->mp_eager_nth.payload_offset;

		assert(xfer_len <= context->len);

		/* If we flagged this context w/ an error, just decrement the byte counter that this
		 * nth packet would have filled in */
		if (OFI_UNLIKELY(context->err_entry.err == FI_ETRUNC)) {
			context->byte_counter -= send_len;
			return;
		}

		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			"hdr->mp_eager_nth.xfer_bytes_tail = %u, hdr->mp_eager_nth.payload_offset = %u, send_len = %lu, xfer_len = %lu\n",
			hdr->mp_eager_nth.xfer_bytes_tail, hdr->mp_eager_nth.payload_offset, send_len, xfer_len);

		if (OFI_UNLIKELY(is_multi_receive)) {		/* branch should compile out */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Multi-receive used with Multi-packet Eager not implemented. abort.");
			abort();
		} else if (OFI_LIKELY(xfer_len <= recv_len)) {

#ifndef NDEBUG
			uint8_t *orig_recv_buf_end = ((uint8_t *) recv_buf) + context->len;
			if (!is_hmem) {
				uint8_t *payload_end = ((uint8_t *)recv_buf) + send_len;
				for (uint8_t *buf_check = (uint8_t *)recv_buf + hdr->mp_eager_nth.payload_offset;
					buf_check < payload_end; ++buf_check) {

					if (*buf_check != 0xAA) {
						fprintf(stderr, "(%d) %s:%s:%d Multi-Packet Eager Nth packet encountered "
							"corrupted destination buffer! Initial PSN = %u, recv_buf=%p, "
							"offset=%0X, payload_length=%ld, buffer altered at %p, "
							"current value is %0hhX\n",
							getpid(), __FILE__, __func__, __LINE__,
							hdr->mp_eager_nth.mp_egr_uid, recv_buf,
							hdr->mp_eager_nth.payload_offset, send_len,
							buf_check, *buf_check);
						abort();
					}
				}
			}
#endif

			const size_t xfer_bytes_tail = hdr->mp_eager_nth.xfer_bytes_tail;
			recv_buf = is_hmem ? (void *) opx_ep->hmem_copy_buf
					   : (void*)((uint8_t*) recv_buf + hdr->mp_eager_nth.payload_offset);

			/* We'll never *not* have some bytes in the tail */
			if (OFI_LIKELY(xfer_bytes_tail == 16)) {
				uint64_t * recv_buf_qw = (uint64_t *)recv_buf;
				recv_buf_qw[0] = hdr->mp_eager_nth.xfer_tail[0];
				recv_buf_qw[1] = hdr->mp_eager_nth.xfer_tail[1];
				recv_buf = (void *) (recv_buf_qw + 2);
			} else {
				memcpy(recv_buf, hdr->mp_eager_nth.xfer_tail, xfer_bytes_tail);
				recv_buf = (void*)((uint8_t *)recv_buf + xfer_bytes_tail);
			}

			assert(is_hmem || ((uint8_t*)recv_buf) <= orig_recv_buf_end);

			if (OFI_LIKELY(send_len > xfer_bytes_tail)) {
				uint64_t * recv_buf_qw = (uint64_t *)recv_buf;
				uint64_t * payload_qw = (uint64_t *)payload;

				unsigned i;
				for (i=0; i<payload_qws_total; ++i) {
					recv_buf_qw[i] = payload_qw[i];
					assert(is_hmem || ((uint8_t*) &recv_buf_qw[i] ) <= orig_recv_buf_end);
				}
			}

			if (is_hmem) {
				recv_buf = (void*)((uint8_t*) context->buf + hdr->mp_eager_nth.payload_offset);
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) context->hmem_info_qws;
				opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
						recv_buf, opx_ep->hmem_copy_buf, send_len,
						OPX_HMEM_DEV_REG_RECV_THRESHOLD);
			}
			/* fi_opx_hfi1_dump_packet_hdr(hdr, __func__, __LINE__);*/

			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Multi-packet EAGER (nth) send_len %lu <= recv_len %lu; enqueue cq (pending)\n", send_len, recv_len);

			assert(context->byte_counter >= send_len);
			context->byte_counter -= send_len;
#ifndef NDEBUG
			if (context->byte_counter == 0) {
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "===================================== RECV -- MULTI PACKET EAGER NTH COMPLETE\n");
			}
#endif
		} else {	/* truncation - unlikely */
			/* We verified the context had enough buffer space for the entire multi-packet payload
			 * when we processed the first multi-egr packet. So if xver_len > recv_len, then something
			 * went wrong somewhere. Either the offset/payload size for this packet is incorrect, or
			 * something messed up the context. */
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Multi-packet Eager nth truncation error. Abort.");
			abort();
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-MP-EAGER-NTH");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- MULTI PACKET EAGER NTH byte counter %lu (end)\n",context->byte_counter);
	} else {
		fi_opx_handle_recv_rts(hdr, payload, opx_ep, origin_tag, opcode,
					context, is_multi_receive, is_intranode, is_hmem,
					lock_required, reliability, hfi1_type);
	}
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "\n");
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_shm_dynamic_tx_connect(const unsigned is_intranode,
				      struct fi_opx_ep * opx_ep,
				      const unsigned rx_id,
				      const uint8_t hfi1_unit)
{
	if (!is_intranode) {
		return FI_SUCCESS;
	}

	assert(hfi1_unit < FI_OPX_MAX_HFIS);
	assert(rx_id < OPX_SHM_MAX_CONN_NUM);

#ifdef OPX_DAOS
	uint32_t segment_index;

	if (!opx_ep->daos_info.hfi_rank_enabled) {
		assert(rx_id < 256);
		segment_index = OPX_SHM_SEGMENT_INDEX(hfi1_unit, rx_id);
	} else {
		segment_index = rx_id & OPX_SHM_MAX_CONN_MASK;
	}
#else
	uint32_t segment_index = OPX_SHM_SEGMENT_INDEX(hfi1_unit, rx_id);
#endif

	if (OFI_LIKELY(opx_ep->tx->shm.fifo_segment[segment_index] != NULL)) {
		/* Connection already established */
		return FI_SUCCESS;
	}

	/* Setup new connection */
	char buffer[OPX_JOB_KEY_STR_SIZE + 32];
	int inst = 0;

#ifdef OPX_DAOS
	if (opx_ep->daos_info.hfi_rank_enabled) {
		inst = opx_ep->daos_info.rank_inst;
	}
#endif

	snprintf(buffer, sizeof(buffer), OPX_SHM_FILE_NAME_PREFIX_FORMAT,
		opx_ep->domain->unique_job_key_str, hfi1_unit, inst);

	return opx_shm_tx_connect(&opx_ep->tx->shm, (const char * const) buffer,
				segment_index, rx_id, FI_OPX_SHM_FIFO_SIZE, FI_OPX_SHM_PACKET_SIZE);
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_header_rzv_cts(struct fi_opx_ep * opx_ep,
				const union opx_hfi1_packet_hdr * const hdr,
				const union fi_opx_hfi1_packet_payload * const payload,
				const uint8_t origin_rs,
				const unsigned is_intranode,
				const int lock_required,
				const enum ofi_reliability_kind reliability,
				const enum opx_hfi1_type hfi1_type)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- %s RENDEZVOUS CTS (begin)\n", is_intranode ? "SHM":"HFI");

	assert(payload != NULL || hdr->cts.target.opcode == FI_OPX_HFI_DPUT_OPCODE_RZV_ETRUNC);
	const uint8_t u8_rx = hdr->cts.origin_rx;
	const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->cts.origin_rx);

	switch(hdr->cts.target.opcode) {
	case FI_OPX_HFI_DPUT_OPCODE_RZV:
	case FI_OPX_HFI_DPUT_OPCODE_RZV_TID:
	{
		const union fi_opx_hfi1_dput_iov * const dput_iov = payload->cts.iov;
		const uintptr_t target_context_vaddr = hdr->cts.target.vaddr.target_context_vaddr;
		const uint32_t niov = hdr->cts.target.vaddr.niov;
		uint64_t * origin_byte_counter = (uint64_t *)hdr->cts.target.vaddr.origin_byte_counter_vaddr;
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-CTS-HFI:%p", (void *) target_context_vaddr);
		FI_OPX_FABRIC_RX_RZV_CTS(opx_ep, NULL, hdr, (const void * const) payload, 0,
					 u8_rx, origin_rs, niov, dput_iov,
					 (const uint8_t) (FI_NOOP - 1),
					 (const uint8_t) (FI_VOID - 1),
					 (uintptr_t) NULL, /* No RMA Request */
					 target_context_vaddr, origin_byte_counter,
					 hdr->cts.target.opcode, NULL,
					 is_intranode,	/* compile-time constant expression */
					 reliability,	/* compile-time constant expression */
					 u32_ext_rx,
					 hfi1_type);
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-CTS-HFI:%p", (void *) target_context_vaddr);
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_RZV_NONCONTIG:
	{
		const union fi_opx_hfi1_dput_iov * const dput_iov = payload->cts.iov;
		const uintptr_t target_context_vaddr = hdr->cts.target.vaddr.target_context_vaddr;
		const uint32_t niov = hdr->cts.target.vaddr.niov;
		uint64_t * origin_byte_counter = (uint64_t *)hdr->cts.target.vaddr.origin_byte_counter_vaddr;
		FI_OPX_FABRIC_RX_RZV_CTS(opx_ep, NULL, hdr, (const void * const) payload, 0,
					 u8_rx, origin_rs, niov, dput_iov,
					 (const uint8_t) (FI_NOOP - 1),
					 (const uint8_t) (FI_VOID - 1),
					 (uintptr_t) NULL, /* No RMA Request */
					 target_context_vaddr, origin_byte_counter,
					 FI_OPX_HFI_DPUT_OPCODE_RZV_NONCONTIG,
					 NULL,
					 is_intranode,	/* compile-time constant expression */
					 reliability,	/* compile-time constant expression */
					 u32_ext_rx,
					 hfi1_type);
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_RZV_ETRUNC:
	{
		uint64_t * origin_byte_counter = (uint64_t *)hdr->cts.target.vaddr.origin_byte_counter_vaddr;
		*origin_byte_counter = 0;
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_GET:
	{
		const uintptr_t rma_request_vaddr = hdr->cts.target.mr.rma_request_vaddr;
		struct fi_opx_mr *opx_mr = NULL;
		const uint32_t niov = hdr->cts.target.mr.niov;
		HASH_FIND(hh, opx_ep->domain->mr_hashmap,
			  &hdr->cts.target.mr.key,
			  sizeof(hdr->cts.target.mr.key),
			  opx_mr);
		// Permissions (TODO)
		// check MR permissions
		// nack on failed lookup
		assert(opx_mr != NULL);

#ifdef OPX_HMEM
		// Our MR code only supports 1 IOV per registration.
		uint64_t hmem_device;
		enum fi_hmem_iface hmem_iface = fi_opx_mr_get_iface(opx_mr, &hmem_device);
		assert(niov == 1);
		const union fi_opx_hfi1_dput_iov dput_iov =  {
			.rbuf = payload->cts.iov[0].rbuf,
			.sbuf = payload->cts.iov[0].sbuf,
			.bytes = payload->cts.iov[0].bytes,
			.rbuf_iface = payload->cts.iov[0].rbuf_iface,
			.rbuf_device = payload->cts.iov[0].rbuf_device,
			.sbuf_iface = hmem_iface,
			.sbuf_device = hmem_device
		};
		const union fi_opx_hfi1_dput_iov * const dput_iov_ptr = &dput_iov;
#else
		const union fi_opx_hfi1_dput_iov * const dput_iov_ptr = payload->cts.iov;
#endif
		FI_OPX_FABRIC_RX_RZV_CTS(opx_ep, opx_mr, hdr, (const void * const) payload, 0,
					 u8_rx, origin_rs, niov, dput_iov_ptr,
					 hdr->cts.target.mr.op,
					 hdr->cts.target.mr.dt,
					 rma_request_vaddr,
					 (uintptr_t) NULL, /* Target completion counter is in rma_request */
					 NULL, /* No origin byte counter here */
					 FI_OPX_HFI_DPUT_OPCODE_GET,
					 NULL,
					 is_intranode,	/* compile-time constant expression */
					 reliability,	/* compile-time constant expression */
					 u32_ext_rx,
					 hfi1_type);
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_FENCE:
	{
		opx_hfi1_dput_fence(opx_ep, hdr, u8_rx, u32_ext_rx, hfi1_type);
	}
	break;
	default:
		abort();
		break;
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- %s RENDEZVOUS CTS (end)\n", is_intranode ? "SHM":"HFI");
}

void fi_opx_atomic_completion_action(union fi_opx_hfi1_deferred_work * work_state);

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_header_rzv_data(struct fi_opx_ep * opx_ep,
				const union opx_hfi1_packet_hdr * const hdr,
				const union fi_opx_hfi1_packet_payload * const payload,
				const size_t payload_bytes,
				const uint8_t origin_rs,
				const unsigned is_intranode,
				const int lock_required,
				const enum ofi_reliability_kind reliability,
				const enum opx_hfi1_type hfi1_type)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV -- %s RENDEZVOUS DATA Opcode=%0hhX (begin)\n", is_intranode ? "SHM":"HFI", hdr->dput.target.opcode);
	switch(hdr->dput.target.opcode) {
	case FI_OPX_HFI_DPUT_OPCODE_RZV:
	case FI_OPX_HFI_DPUT_OPCODE_RZV_NONCONTIG:
	{
		struct fi_opx_rzv_completion * rzv_comp = (struct fi_opx_rzv_completion *)(hdr->dput.target.rzv.completion_vaddr);
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-DATA-HFI-DPUT:%p", rzv_comp);
		struct opx_context *target_context = rzv_comp->context;
		assert(target_context);
		uint64_t* rbuf_qws = (uint64_t *) fi_opx_dput_rbuf_in(hdr->dput.target.rzv.rbuf);

		/* In a multi-packet SDMA send, the driver sets the high bit on
		 * in the PSN to indicate this is the last packet. The payload
		 * size of the last packet may be smaller than the other packets
		 * in the multi-packet send, so set the payload bytes accordingly */
		const uint16_t bytes = (ntohl(hdr->bth.psn) & 0x80000000) ?
					hdr->dput.target.last_bytes :
					hdr->dput.target.bytes;

		assert(bytes <= FI_OPX_HFI1_PACKET_MTU);
#ifndef NDEBUG
		if (bytes == 0) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Received RZV (non-TID) data packet with 0-byte payload size. hdr->dput.target.last_bytes=%hd, hdr->dput.target.bytes=%hd. Based on PSN high bit (%s), bytes was set to %s\n",
				hdr->dput.target.last_bytes,
				hdr->dput.target.bytes,
				(ntohl(hdr->bth.psn) & 0x80000000) ? "ON" : "OFF",
				(ntohl(hdr->bth.psn) & 0x80000000) ? "last_bytes" : "bytes");
			abort();
		}
#endif
		const uint64_t *sbuf_qws = (uint64_t*)&payload->byte[0];
#ifdef OPX_HMEM
		if (target_context->flags & FI_OPX_CQ_CONTEXT_HMEM) {
			struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) target_context->hmem_info_qws;
			assert(hmem_info->iface > FI_HMEM_SYSTEM);
			opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
					rbuf_qws, sbuf_qws, bytes,
					OPX_HMEM_DEV_REG_RECV_THRESHOLD);
		} else
#endif
		{
			memcpy(rbuf_qws, sbuf_qws, bytes);
		}

		assert(rzv_comp->byte_counter >= bytes);
		rzv_comp->bytes_accumulated += bytes;
		rzv_comp->byte_counter -= bytes;

		if (rzv_comp->byte_counter == 0) {
			assert(target_context->byte_counter >= rzv_comp->bytes_accumulated);
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				"hdr->dput.target.last_bytes = %hu, hdr->dput.target.bytes = %u, bytes = %u, rzv_comp->bytes_accumulated=%lu, target_byte_counter = %p, %lu -> %lu\n",
				hdr->dput.target.last_bytes, hdr->dput.target.bytes, bytes,
				rzv_comp->bytes_accumulated, &target_context->byte_counter,
				target_context->byte_counter,
				target_context->byte_counter - rzv_comp->bytes_accumulated);

			target_context->byte_counter -= rzv_comp->bytes_accumulated;

			/* free the rendezvous completion structure */
			OPX_BUF_FREE(rzv_comp);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-DATA-HFI-DPUT:%p", rzv_comp);
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_RZV_TID:
	{
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RX_PROCESS_HEADER_RZV_TID");
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.expected_receive.tid_rcv_pkts);
		struct fi_opx_rzv_completion * rzv_comp = (struct fi_opx_rzv_completion *)(hdr->dput.target.rzv.completion_vaddr);
		struct opx_context *target_context = rzv_comp->context;
		assert(target_context);

		/* TID packets are mixed 4k/8k packets and length adjusted,
		 * so use actual packet size here reported in LRH as the
		 * number of 4-byte words in the packet; header + payload - icrc
		 */
		uint16_t lrh_pktlen_le;
		size_t total_bytes_to_copy;
		uint16_t bytes;

		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			lrh_pktlen_le = ntohs(hdr->lrh_9B.pktlen);
			total_bytes_to_copy =
				(lrh_pktlen_le - 1) * 4; /* do not copy the trailing icrc */
			bytes =	(uint16_t)(total_bytes_to_copy - sizeof(struct fi_opx_hfi1_stl_packet_hdr_9B));
		} else {
			lrh_pktlen_le = hdr->lrh_16B.pktlen;
			total_bytes_to_copy =
				(lrh_pktlen_le - 1) * 8; /* do not copy the trailing icrc */
			bytes =	(uint16_t)((total_bytes_to_copy -
				   sizeof(struct fi_opx_hfi1_stl_packet_hdr_16B)));
		}

		assert(bytes <= FI_OPX_HFI1_PACKET_MTU);

		/* SDMA expected receive w/TID will use CTRL 1, 2 or 3.
		   Replays should indicate we are not using TID (CTRL 0) */
		int tidctrl = KDETH_GET(hdr->kdeth.offset_ver_tid, TIDCTRL);
		assert((tidctrl == 0) || (tidctrl == 1) || (tidctrl == 2) || (tidctrl == 3));

		/* Copy only if there's a replay payload and TID direct rdma was NOT done.
		 * Note: out of order queued TID packets appear to have a
		 * payload when they don't, so checking tidctrl (not a replay) is necessary.
		 */
		if((payload != NULL) && (tidctrl == 0)) {
			uint64_t* rbuf_qws = (uint64_t *) fi_opx_dput_rbuf_in(hdr->dput.target.rzv.rbuf);
			const uint64_t *sbuf_qws = (uint64_t*)&payload->byte[0];
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				"TID REPLAY rbuf_qws %p, sbuf_qws %p, bytes %u/%#x, target_context->byte_counter %p\n",
				(void*)rbuf_qws, (void*)sbuf_qws, bytes, bytes, &target_context->byte_counter);
			if (target_context->flags & FI_OPX_CQ_CONTEXT_HMEM) {
				struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) target_context->hmem_info_qws;
				assert(hmem_info->iface > FI_HMEM_SYSTEM);
				opx_copy_to_hmem(hmem_info->iface, hmem_info->device, hmem_info->hmem_dev_reg_handle,
						rbuf_qws, sbuf_qws, bytes,
						OPX_HMEM_DEV_REG_RECV_THRESHOLD);
			} else {
				memcpy(rbuf_qws, sbuf_qws, bytes);
			}
			FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.expected_receive.tid_rcv_pkts_replays);
		}
#ifndef NDEBUG
		else { /* Debug, tracking where the TID wrote even though we don't memcpy here */
			uint64_t* rbuf_qws = (uint64_t *) fi_opx_dput_rbuf_in(hdr->dput.target.rzv.rbuf);
			const uint64_t *sbuf_qws = (uint64_t*)&payload->byte[0];
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				"NOT REPLAY tidctrl %#x, tid %#X, tid0M %#X, tidoffset %#X rbuf_qws %p, "
				"sbuf_qws %p, bytes %u/%#x, target_context->byte_counter %p\n",
				tidctrl, KDETH_GET(hdr->kdeth.offset_ver_tid, TID),
				KDETH_GET(hdr->kdeth.offset_ver_tid, OM),
				KDETH_GET(hdr->kdeth.offset_ver_tid, OFFSET),
				(void*)rbuf_qws, (void*)sbuf_qws, bytes, bytes,
				&target_context->byte_counter);
		}
#endif
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			"hdr->dput.target.last_bytes = %hu, hdr->dput.target.bytes = %u, bytes = %u, target_context->byte_counter = %p, %lu -> %lu\n",
			hdr->dput.target.last_bytes, hdr->dput.target.bytes,
			bytes, &target_context->byte_counter,
			rzv_comp->byte_counter, rzv_comp->byte_counter - bytes);
		assert(rzv_comp->byte_counter >= bytes);
		rzv_comp->bytes_accumulated += bytes;
		rzv_comp->byte_counter -= bytes;

		/* On completion, decrement TID refcount and maybe free the TID cache */
		if (rzv_comp->byte_counter == 0) {
			const uint64_t tid_vaddr = rzv_comp->tid_vaddr;
			const uint64_t tid_length = rzv_comp->tid_length;
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				"tid vaddr>buf [%p - %p] tid len %lu/%#lX\n",
				(void *)tid_vaddr,
				(void *)(tid_vaddr + tid_length),
				tid_length, tid_length);
			target_context->byte_counter -= rzv_comp->bytes_accumulated;

			opx_deregister_for_rzv(opx_ep, tid_vaddr, tid_length);

			/* free the rendezvous completion structure */
			OPX_BUF_FREE(rzv_comp);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RX_PROCESS_HEADER_RZV_TID");
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_PUT:
	{
		assert(payload != NULL);
		const uint64_t *sbuf_qws = (uint64_t*)&payload->byte[0];
		struct fi_opx_mr *opx_mr = NULL;
		HASH_FIND(hh, opx_ep->domain->mr_hashmap,
			&hdr->dput.target.mr.key,
			sizeof(hdr->dput.target.mr.key),
			opx_mr);

		if (opx_mr == NULL) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"lookup of key (%ld) failed; packet dropped\n", hdr->dput.target.mr.key);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RECV -- RENDEZVOUS DATA - failed (end)\n");
			assert(0);
			return;
		}

		uint64_t* rbuf_qws = (uint64_t *)((uint8_t*)opx_mr->iov.iov_base +
							fi_opx_dput_rbuf_in(hdr->dput.target.mr.offset));
		uint64_t hmem_device;
		enum fi_hmem_iface hmem_iface = fi_opx_mr_get_iface(opx_mr, &hmem_device);

		/* In a multi-packet SDMA send, the driver sets the high bit on
		 * in the PSN to indicate this is the last packet. The payload
		 * size of the last packet may be smaller than the other packets
		 * in the multi-packet send, so set the payload bytes accordingly */
		const uint16_t bytes = (ntohl(hdr->bth.psn) & 0x80000000) ?
					hdr->dput.target.last_bytes :
					hdr->dput.target.bytes;
		assert(bytes <= FI_OPX_HFI1_PACKET_MTU);

#ifndef NDEBUG
		if (bytes == 0) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Received RMA PUT data packet with 0-byte payload size. hdr->dput.target.last_bytes=%hd, hdr->dput.target.bytes=%hd. Based on PSN high bit (%s), bytes was set to %s\n",
				hdr->dput.target.last_bytes,
				hdr->dput.target.bytes,
				(ntohl(hdr->bth.psn) & 0x80000000) ? "ON" : "OFF",
				(ntohl(hdr->bth.psn) & 0x80000000) ? "last_bytes" : "bytes");
			abort();
		}
#endif
		// Optimize Memcpy
		if(hdr->dput.target.op == FI_NOOP - 1 &&
			hdr->dput.target.dt == FI_VOID - 1) {
			OPX_HMEM_COPY_TO(rbuf_qws, sbuf_qws, bytes,
					OPX_HMEM_NO_HANDLE,
					OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET,
					hmem_iface, hmem_device);
		} else {
			OPX_HMEM_ATOMIC_DISPATCH(sbuf_qws, rbuf_qws, bytes,
						hdr->dput.target.dt,
						hdr->dput.target.op,
						hmem_iface, hmem_device);
		}
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_GET:
	{
		assert(payload != NULL);
		struct fi_opx_rma_request *rma_req =
			(struct fi_opx_rma_request *) hdr->dput.target.get.rma_request_vaddr;
		struct fi_opx_completion_counter *cc = rma_req->cc;
		uint64_t* rbuf_qws = (uint64_t *) fi_opx_dput_rbuf_in(hdr->dput.target.get.rbuf);
		const uint64_t *sbuf_qws = (uint64_t*)&payload->byte[0];

		/* In a multi-packet SDMA send, the driver sets the high bit on
		 * in the PSN to indicate this is the last packet. The payload
		 * size of the last packet may be smaller than the other packets
		 * in the multi-packet send, so set the payload bytes accordingly */
		const uint16_t bytes = (ntohl(hdr->bth.psn) & 0x80000000) ?
					hdr->dput.target.last_bytes :
					hdr->dput.target.bytes;

		assert(cc);
		assert(bytes <= FI_OPX_HFI1_PACKET_MTU);

#ifndef NDEBUG
		if (bytes == 0) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Received RMA GET data packet with 0-byte payload size. hdr->dput.target.last_bytes=%hd, hdr->dput.target.bytes=%hd. Based on PSN high bit (%s), bytes was set to %s\n",
				hdr->dput.target.last_bytes,
				hdr->dput.target.bytes,
				(ntohl(hdr->bth.psn) & 0x80000000) ? "ON" : "OFF",
				(ntohl(hdr->bth.psn) & 0x80000000) ? "last_bytes" : "bytes");
			abort();
		}
#endif
		if (hdr->dput.target.dt == (FI_VOID - 1)) {
			OPX_HMEM_COPY_TO(rbuf_qws, sbuf_qws, bytes, OPX_HMEM_NO_HANDLE,
					 OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET,
					 rma_req->hmem_iface, rma_req->hmem_device);
		} else {
			OPX_HMEM_ATOMIC_DISPATCH(sbuf_qws, rbuf_qws, bytes,
						hdr->dput.target.dt,
						FI_ATOMIC_WRITE,
						rma_req->hmem_iface,
						rma_req->hmem_device);
		}
		assert(cc->byte_counter >= bytes);
		cc->byte_counter -= bytes;
		assert(cc->byte_counter >= 0);

		if(cc->byte_counter == 0) {
			OPX_BUF_FREE(rma_req);
			cc->hit_zero(cc);
		}
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH:
	{
		const uint8_t u8_rx = hdr->dput.origin_rx;
		const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->dput.origin_rx);
		struct fi_opx_mr *opx_mr = NULL;

		uint64_t key = hdr->dput.target.mr.key;
		HASH_FIND(hh, opx_ep->domain->mr_hashmap,
			&key,
			sizeof(key),
			opx_mr);
		assert(opx_mr != NULL);
		uintptr_t mr_offset = fi_opx_dput_rbuf_in(hdr->dput.target.mr.offset);
		uint64_t* rbuf_qws = (uint64_t *)((uint8_t*)opx_mr->iov.iov_base + mr_offset);
		const struct fi_opx_hfi1_dput_fetch *dput_fetch = (struct fi_opx_hfi1_dput_fetch *)&payload->byte[0];

		/* In a multi-packet SDMA send, the driver sets the high bit on
		 * in the PSN to indicate this is the last packet. The payload
		 * size of the last packet may be smaller than the other packets
		 * in the multi-packet send, so set the payload bytes accordingly */
		const uint16_t bytes = (ntohl(hdr->bth.psn) & 0x80000000) ?
					hdr->dput.target.last_bytes :
					hdr->dput.target.bytes;

#ifndef NDEBUG
		if (bytes == 0) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Received ATOMIC FETCH data packet with 0-byte payload size. hdr->dput.target.last_bytes=%hd, hdr->dput.target.bytes=%hd. Based on PSN high bit (%s), bytes was set to %s\n",
				hdr->dput.target.last_bytes,
				hdr->dput.target.bytes,
				(ntohl(hdr->bth.psn) & 0x80000000) ? "ON" : "OFF",
				(ntohl(hdr->bth.psn) & 0x80000000) ? "last_bytes" : "bytes");
			abort();
		}
#endif
		assert(bytes > sizeof(*dput_fetch));
		uint64_t hmem_device;
		enum fi_hmem_iface hmem_iface = fi_opx_mr_get_iface(opx_mr, &hmem_device);

		// rbuf_iface & rbuf_hmem are contained in the rma_request that
		// resides in the originating endpoint, so can just be set to
		// system/0 here.
		union fi_opx_hfi1_dput_iov dput_iov = {
			.sbuf = mr_offset,
			.rbuf = dput_fetch->fetch_rbuf,
			.bytes = bytes - sizeof(struct fi_opx_hfi1_dput_fetch),
			.rbuf_iface = FI_HMEM_SYSTEM,
			.sbuf_iface = hmem_iface,
			.rbuf_device = 0,
			.sbuf_device = hmem_device
		};
		assert(dput_iov.bytes <= FI_OPX_HFI1_PACKET_MTU - sizeof(*dput_fetch));
		assert(hdr->dput.target.op != (FI_NOOP-1));
		assert(hdr->dput.target.dt != (FI_VOID-1));

		// Do the FETCH part of this atomic fetch operation
		union fi_opx_hfi1_deferred_work *work =
		FI_OPX_FABRIC_RX_RZV_CTS(opx_ep, opx_mr, hdr,
					(const void * const) payload, bytes,
					u8_rx, origin_rs, 1, &dput_iov,
					hdr->dput.target.op,
					hdr->dput.target.dt,
					dput_fetch->rma_request_vaddr,
					(uintptr_t) NULL, /* target byte counter is in rma_request */
					NULL, /* No origin byte counter here */
					FI_OPX_HFI_DPUT_OPCODE_GET,
					fi_opx_atomic_completion_action,
					is_intranode,
					reliability,
					u32_ext_rx,
					hfi1_type);
		if(work == NULL) {
			// The FETCH completed without being deferred, now do
			// the actual atomic operation.
			const uint64_t *sbuf_qws = (uint64_t*)(dput_fetch + 1);
			OPX_HMEM_ATOMIC_DISPATCH(sbuf_qws, rbuf_qws, dput_iov.bytes,
						hdr->dput.target.dt,
						hdr->dput.target.op,
						hmem_iface, hmem_device);
		}
		// else the FETCH was deferred, so the atomic operation will
		// be done upon FETCH completion via fi_opx_atomic_completion_action
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH:
	{
		const uint8_t u8_rx = hdr->dput.origin_rx;
		const uint32_t u32_ext_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->dput.origin_rx);
		struct fi_opx_mr *opx_mr = NULL;

		uint64_t key = hdr->dput.target.mr.key;
		HASH_FIND(hh, opx_ep->domain->mr_hashmap,
			&key,
			sizeof(key),
			opx_mr);
		assert(opx_mr != NULL);
		uintptr_t mr_offset = fi_opx_dput_rbuf_in(hdr->dput.target.mr.offset);
		uint64_t* rbuf_qws = (uint64_t *)((uint8_t*)opx_mr->iov.iov_base + mr_offset);
		const struct fi_opx_hfi1_dput_fetch *dput_fetch = (struct fi_opx_hfi1_dput_fetch *)&payload->byte[0];

		/* In a multi-packet SDMA send, the driver sets the high bit on
		 * in the PSN to indicate this is the last packet. The payload
		 * size of the last packet may be smaller than the other packets
		 * in the multi-packet send, so set the payload bytes accordingly */
		const uint16_t bytes = (ntohl(hdr->bth.psn) & 0x80000000) ?
					hdr->dput.target.last_bytes :
					hdr->dput.target.bytes;

#ifndef NDEBUG
		if (bytes == 0) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
				"Received ATOMIC COMPARE FETCH data packet with 0-byte payload size. hdr->dput.target.last_bytes=%hd, hdr->dput.target.bytes=%hd. Based on PSN high bit (%s), bytes was set to %s\n",
				hdr->dput.target.last_bytes,
				hdr->dput.target.bytes,
				(ntohl(hdr->bth.psn) & 0x80000000) ? "ON" : "OFF",
				(ntohl(hdr->bth.psn) & 0x80000000) ? "last_bytes" : "bytes");
			abort();
		}
#endif
		assert(bytes > sizeof(*dput_fetch));
		uint64_t hmem_device;
		enum fi_hmem_iface hmem_iface = fi_opx_mr_get_iface(opx_mr, &hmem_device);

		// rbuf_iface & rbuf_hmem are contained in the rma_request that
		// resides in the originating endpoint, so can just be set to
		// system/0 here.
		union fi_opx_hfi1_dput_iov dput_iov = {
			.sbuf = mr_offset,
			.rbuf = dput_fetch->fetch_rbuf,
			.bytes = (bytes - sizeof(struct fi_opx_hfi1_dput_fetch)) >> 1,
			.rbuf_iface = FI_HMEM_SYSTEM,
			.sbuf_iface = hmem_iface,
			.rbuf_device = 0,
			.sbuf_device = hmem_device
		};
		assert(dput_iov.bytes <= ((FI_OPX_HFI1_PACKET_MTU - sizeof(*dput_fetch)) >> 1));
		assert(hdr->dput.target.op != (FI_NOOP-1));
		assert(hdr->dput.target.dt != (FI_VOID-1));

		// Do the FETCH part of this atomic fetch operation
		union fi_opx_hfi1_deferred_work *work =
		FI_OPX_FABRIC_RX_RZV_CTS(opx_ep, opx_mr, hdr,
					(const void * const) payload, bytes,
					u8_rx, origin_rs, 1, &dput_iov,
					hdr->dput.target.op,
					hdr->dput.target.dt,
					dput_fetch->rma_request_vaddr,
					(uintptr_t) NULL, /* Target completion counter is in rma request */
					NULL, /* No origin byte counter here */
					FI_OPX_HFI_DPUT_OPCODE_GET,
					fi_opx_atomic_completion_action,
					is_intranode,
					reliability,
					u32_ext_rx,
					hfi1_type);
		if(work == NULL) {
			// The FETCH completed without being deferred, now do
			// the actual atomic operation.
			const uint64_t *sbuf_qws = (uint64_t*)(dput_fetch + 1);
			OPX_HMEM_ATOMIC_DISPATCH(sbuf_qws, rbuf_qws, dput_iov.bytes,
						hdr->dput.target.dt,
						hdr->dput.target.op,
						hmem_iface, hmem_device);
		}
		// else the FETCH was deferred, so the atomic operation will
		// be done upon FETCH completion via fi_opx_atomic_completion_action
	}
	break;
	case FI_OPX_HFI_DPUT_OPCODE_FENCE:
	{
		assert(payload != NULL);
		struct fi_opx_completion_counter *cc =
				(struct fi_opx_completion_counter *)hdr->dput.target.fence.completion_counter;
		const uint32_t bytes = hdr->dput.target.fence.bytes_to_fence;

		assert(cc);
		assert(cc->byte_counter >= bytes);
		cc->byte_counter -= bytes;
		if (cc->byte_counter == 0) {
			cc->hit_zero(cc);
		}
	}
	break;
	default:
		abort();
		break;
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
	"===================================== RECV -- %s RENDEZVOUS DATA (end)\n", is_intranode ? "SHM":"HFI");
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_header_non_eager(struct fid_ep *ep,
				const union opx_hfi1_packet_hdr * const hdr,
				const union fi_opx_hfi1_packet_payload * const payload,
				const size_t payload_bytes,
				const uint64_t static_flags,
				const uint8_t opcode,
				const uint8_t origin_rs,
				const unsigned is_intranode,
				const int lock_required,
				const enum ofi_reliability_kind reliability,
				const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (opcode == FI_OPX_HFI_BTH_OPCODE_RZV_CTS) {
		fi_opx_ep_rx_process_header_rzv_cts(opx_ep, hdr, payload,
						origin_rs,
						is_intranode,
						lock_required, reliability,
						hfi1_type);
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_RZV_DATA) {
		fi_opx_ep_rx_process_header_rzv_data(opx_ep, hdr, payload, payload_bytes,
						origin_rs,
						is_intranode,
						lock_required, reliability,
						hfi1_type);
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_ACK) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"unimplemented opcode (%u); abort\n", opcode);
		abort();
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_RMA) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"unimplemented opcode (%u); abort\n", opcode);
		abort();
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_ATOMIC) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"unimplemented opcode (%u); abort\n", opcode);
		abort();
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_UD) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"reliability exception with opcode %d, dropped\n", opcode);
	} else {
		fprintf(stderr, "unimplemented opcode (%#x); abort\n", opcode);
		fprintf(stderr, "%s:%u payload %p, payload bytes %zu, is_instranode %u,  %#16.16llX %#16.16llX %#16.16llX %#16.16llX %#16.16llX %#16.16llX %#16.16llX \n",
			__func__, __LINE__, payload, payload_bytes, is_intranode,
		      (long long) hdr->qw_9B[0],
		      (long long) hdr->qw_9B[1],
		      (long long) hdr->qw_9B[2],
		      (long long) hdr->qw_9B[3],
		      (long long) hdr->qw_9B[4],
		      (long long) hdr->qw_9B[5],
		      (long long) hdr->qw_9B[6]);
		abort();
	}
}

__OPX_FORCE_INLINE__
uint64_t fi_opx_mp_egr_id_from_nth_packet(const union opx_hfi1_packet_hdr *hdr,
					  const opx_lid_t slid)
{
	return ((uint64_t) hdr->mp_eager_nth.mp_egr_uid) |
		((uint64_t) ((slid << 8) | hdr->reliability.origin_tx) << 32);
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_pending_mp_eager_ue(struct fid_ep *ep,
				struct opx_context *context,
				union fi_opx_mp_egr_id mp_egr_id,
				const unsigned is_intranode,
				const int lock_required,
				const enum ofi_reliability_kind reliability,
				const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const uint64_t is_hmem = context->flags & FI_OPX_CQ_CONTEXT_HMEM;
	struct fi_opx_hfi1_ue_packet *uepkt = opx_ep->rx->mp_egr_queue.ue.head;

	FI_OPX_DEBUG_COUNTERS_DECLARE_TMP(length);

	while (uepkt && context->byte_counter) {
		opx_lid_t slid;
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			slid = (opx_lid_t)__be16_to_cpu24(((__be16)(uepkt->hdr.lrh_9B.slid)));
		} else {
			slid = (opx_lid_t)__le24_to_cpu((uepkt->hdr.lrh_16B.slid20 << 20) | (uepkt->hdr.lrh_16B.slid));
		}

		if (fi_opx_mp_egr_id_from_nth_packet(&uepkt->hdr, slid) == mp_egr_id.id) {

			opx_ep_complete_receive_operation(ep,
				&uepkt->hdr,
				&uepkt->payload,
				0,	/* OFI Tag, N/A for multi-packet eager nth */
				context,
				FI_OPX_HFI_BTH_OPCODE_MP_EAGER_NTH,
				OPX_MULTI_RECV_FALSE,
				OPX_INTRANODE_FALSE,
				is_hmem,
				lock_required,
				reliability,
				hfi1_type);

			/* Remove this packet and get the next one */
			uepkt = fi_opx_hfi1_ue_packet_slist_remove_item(uepkt,
									&opx_ep->rx->mp_egr_queue.ue);
		} else {
			uepkt = uepkt->next;
		}
		FI_OPX_DEBUG_COUNTERS_INC(length);
	}

	FI_OPX_DEBUG_COUNTERS_MAX_OF(opx_ep->debug_counters.mp_eager.recv_max_ue_queue_length, length);
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_header_mp_eager_first(struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const uint64_t static_flags,
		const uint8_t opcode,
		const uint8_t origin_rs,
		const unsigned is_intranode,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type,
		const opx_lid_t slid)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_first_packets);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "search the match queue\n");

	const uint64_t kind = (static_flags & FI_TAGGED) ? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG;
	assert((kind == FI_OPX_KIND_TAG && FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)) ||
		(kind == FI_OPX_KIND_MSG && !FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)));
	struct opx_context *context = (struct opx_context *) opx_ep->rx->queue[kind].mq.head;
	struct opx_context *prev = NULL;

	while (
		context &&
		!is_match(opx_ep,
			hdr,
			context,
			opx_ep->daos_info.rank,
			opx_ep->daos_info.rank_inst,
			is_intranode,
			slid)
	) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "context = %p\n", context);
		prev = context;
		context = context->next;
	}

	if (!context) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"did not find a match .. add this packet to the unexpected queue\n");

		if (OFI_LIKELY(FI_OPX_HFI_BTH_OPCODE_IS_TAGGED(opcode)))
			fi_opx_ep_rx_append_ue_tag(opx_ep->rx, hdr, payload, payload_bytes,
				opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst,
				opx_ep->daos_info.hfi_rank_enabled,
				FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep), slid);
		else
			fi_opx_ep_rx_append_ue_msg(opx_ep->rx, hdr, payload, payload_bytes,
				opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst,
				opx_ep->daos_info.hfi_rank_enabled,
				FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep), slid);

		return;
	}

	/* Found a match. Remove from the match queue */
	slist_remove(&opx_ep->rx->queue[kind].mq,
		     (struct slist_entry *) context,
		     (struct slist_entry *) prev);

	uint64_t is_hmem = context->flags & FI_OPX_CQ_CONTEXT_HMEM;

	/* Copy this packet's payload to the context's buffer. */
	opx_ep_complete_receive_operation(ep, hdr, payload,
			hdr->match.ofi_tag, context,
			opcode,
			OPX_MULTI_RECV_FALSE,
			OPX_INTRANODE_FALSE,	/* Should always be false for mp_eager */
			is_hmem,
			lock_required,
			reliability,
			hfi1_type);

	const union fi_opx_mp_egr_id mp_egr_id = {
		.uid = hdr->reliability.psn,
		.slid_origin_tx = (slid << 8) | hdr->reliability.origin_tx };

	/* Process any other early arrival packets that are part of this multi-packet egr */
	fi_opx_ep_rx_process_pending_mp_eager_ue(ep, context, mp_egr_id, is_intranode, lock_required, reliability, hfi1_type);

	/* Only add this to the multi-packet egr queue if we still expect additional packets to come in */
	if (context->byte_counter) {
		context->mp_egr_id = mp_egr_id;
		slist_insert_tail((struct slist_entry *) context, &opx_ep->rx->mp_egr_queue.mq);
	} else {
		context->next = NULL;

		if (OFI_UNLIKELY(context->err_entry.err == FI_ETRUNC)) {
			slist_insert_tail((struct slist_entry *) context, opx_ep->rx->cq_err_ptr);
		} else {
			fi_opx_enqueue_completed(opx_ep->rx->cq_completed_ptr, context,
						lock_required);
		}
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_completed_eager_first);
	}
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_process_header_mp_eager_nth(struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const uint64_t static_flags,
		const uint8_t opcode,
		const uint8_t origin_rs,
		const unsigned is_intranode,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type,
		const opx_lid_t slid)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_nth_packets);

	/* Search mp-eager queue for the context w/ matching mp-eager ID */

	const uint64_t mp_egr_id = fi_opx_mp_egr_id_from_nth_packet(hdr, slid);
	struct opx_context *context = (struct opx_context *) opx_ep->rx->mp_egr_queue.mq.head;
	struct opx_context *prev = NULL;

	FI_OPX_DEBUG_COUNTERS_DECLARE_TMP(length);

	while (context && context->mp_egr_id.id != mp_egr_id) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			"process_header_mp_eager_nth: Searching mp_egr match queue, context = %p\n", context);
		prev = context;
		context = context->next;
		FI_OPX_DEBUG_COUNTERS_INC(length);
	}

	FI_OPX_DEBUG_COUNTERS_MAX_OF(opx_ep->debug_counters.mp_eager.recv_max_mq_queue_length, length);

	if (!context) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"process_header_mp_eager_nth: did not find a match .. add this packet to the unexpected queue\n");
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_nth_no_match);

		fi_opx_ep_rx_append_ue_egr(opx_ep->rx, hdr, payload, payload_bytes, slid);

		return;
	}

	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_nth_match);

	/* We found a match!  */
	opx_ep_complete_receive_operation(ep,
				hdr,
				payload,
				0,		/* OFI Tag, N/A for multi-packet eager nth */
				context,
				opcode, 	// FI_OPX_HFI_BTH_OPCODE_MP_EAGER_NTH
				OPX_MULTI_RECV_FALSE,
				is_intranode,
				context->flags & FI_OPX_CQ_CONTEXT_HMEM,
				lock_required,
				reliability,
				hfi1_type);

	if (!context->byte_counter) {
		/* Remove from the mp-eager queue */
		slist_remove(&opx_ep->rx->mp_egr_queue.mq,
			     (struct slist_entry *) context,
			     (struct slist_entry *) prev);

		if (OFI_UNLIKELY(context->err_entry.err == FI_ETRUNC)) {
			slist_insert_tail((struct slist_entry *) context, opx_ep->rx->cq_err_ptr);
		} else {
			fi_opx_enqueue_completed(opx_ep->rx->cq_completed_ptr, context,
						lock_required);
		}

		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_completed_eager_nth);
	}
}

static inline
void fi_opx_ep_rx_process_header (struct fid_ep *ep,
		const union opx_hfi1_packet_hdr * const hdr,
		const union fi_opx_hfi1_packet_payload * const payload,
		const size_t payload_bytes,
		const uint64_t static_flags,
		const uint8_t opcode,
		const uint8_t origin_rs,
		const unsigned is_intranode,
		const int lock_required,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type,
		const opx_lid_t slid)
{

	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	if (OFI_UNLIKELY(opcode < FI_OPX_HFI_BTH_OPCODE_MP_EAGER_NTH)) {
		fi_opx_ep_rx_process_header_non_eager(ep, hdr, payload, payload_bytes,
						static_flags, opcode,
						origin_rs,
						is_intranode,
						lock_required, reliability, hfi1_type);
		return;
	} else if (FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_MP_EAGER_FIRST) {
		fi_opx_ep_rx_process_header_mp_eager_first(ep, hdr, payload, payload_bytes,
							static_flags, opcode,
							origin_rs,
							is_intranode,
							lock_required, reliability,
							hfi1_type, slid);

		return;
	} else if (opcode == FI_OPX_HFI_BTH_OPCODE_MP_EAGER_NTH) {
		fi_opx_ep_rx_process_header_mp_eager_nth(ep, hdr, payload, payload_bytes,
							static_flags, opcode,
							origin_rs,
							is_intranode,
							lock_required, reliability, hfi1_type, slid);
		return;
	}

	assert(opcode >= FI_OPX_HFI_BTH_OPCODE_MSG_INJECT);

	/* search the match queue */
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "search the match queue\n");

	assert(static_flags & (FI_TAGGED | FI_MSG));
	const uint64_t kind = (static_flags & FI_TAGGED) ? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG;
	struct opx_context *context = (struct opx_context *) opx_ep->rx->queue[kind].mq.head;
	struct opx_context *prev = NULL;

	while (OFI_LIKELY(context != NULL) &&
		!is_match(opx_ep,
			hdr,
			context,
			opx_ep->daos_info.rank,
			opx_ep->daos_info.rank_inst,
			is_intranode,
			slid)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "context = %p\n", context);
		prev = context;
		context = context->next;
	}
	if (!context) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"did not find a match .. add this packet to the unexpected queue\n");

		if (OFI_LIKELY(kind == FI_OPX_KIND_TAG)) {
			fi_opx_ep_rx_append_ue_tag(opx_ep->rx, hdr, payload, payload_bytes,
				opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst,
				opx_ep->daos_info.hfi_rank_enabled,
				FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep), slid);
		} else {
			fi_opx_ep_rx_append_ue_msg(opx_ep->rx, hdr, payload, payload_bytes,
				opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst,
				opx_ep->daos_info.hfi_rank_enabled,
				FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep), slid);
		}

		return;
	}

	const uint64_t rx_op_flags = context->flags;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "found a match\n");

	if (OFI_LIKELY((static_flags & FI_TAGGED) ||	/* branch will compile out for tag */
			((rx_op_flags & FI_MULTI_RECV) == 0))) {

		assert((prev == NULL) || (prev->next == context));
		if (prev)
			prev->next = context->next;
		else {
			assert(opx_ep->rx->queue[kind].mq.head == (struct slist_entry *) context);
			opx_ep->rx->queue[kind].mq.head = (struct slist_entry *) context->next;
		}

		if (context->next == NULL){
			assert(opx_ep->rx->queue[kind].mq.tail == (struct slist_entry *) context);
			opx_ep->rx->queue[kind].mq.tail = (struct slist_entry *) prev;
		}

		context->next = NULL;

		opx_ep_complete_receive_operation(ep, hdr, payload,
			hdr->match.ofi_tag, context, opcode,
			OPX_MULTI_RECV_FALSE,
			is_intranode,
			rx_op_flags & FI_OPX_CQ_CONTEXT_HMEM,
			lock_required,
			reliability,
			hfi1_type);

		return;
	}

	/*
	 * verify that there is enough space available in
	 * the multi-receive buffer for the incoming data
	 */
	const uint64_t recv_len = context->len;
	const uint64_t send_len = fi_opx_hfi1_packet_hdr_message_length(hdr);

	assert(!(context->flags & FI_OPX_CQ_CONTEXT_HMEM));
	if (OFI_LIKELY(send_len <= recv_len)) {
		opx_ep_complete_receive_operation(ep, hdr, payload,
			0, context, opcode,
			OPX_MULTI_RECV_TRUE,
			is_intranode,
			OPX_HMEM_FALSE,
			lock_required,
			reliability,
			hfi1_type);

		if (context->len < opx_ep->rx->min_multi_recv) {
			/* after processing this message there is not
			 * enough space available in the multi-receive
			 * buffer to receive the next message; post a
			 * 'FI_MULTI_RECV' event to the completion
			 * queue and return. */

			/* remove context from match queue */
			if (prev)
				prev->next = context->next;
			else
				opx_ep->rx->queue[kind].mq.head = (struct slist_entry *) context->next;

			if (context->next == NULL)
				opx_ep->rx->queue[kind].mq.tail = NULL;

			context->next = NULL;

			// Signaling the userneeds to be deferred until the op is completed for rendezvous
			// reusing byte counter as a pending flag
			// to ensure that any pending ops are completed (eg rendezvous multi-receive)
			if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
			if(context->byte_counter == 0) {
				slist_insert_tail((struct slist_entry *) context, opx_ep->rx->cq_completed_ptr);
			}
		}
	} else {
		/*
		 * there is not enough space available in
		 * the multi-receive buffer;
		 */
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Not enough space in multi-receive buffer; abort\n");
		abort();
	}
}






#include "rdma/opx/fi_opx_fabric_progress.h"


void opx_hfi1_sdma_process_requests(struct fi_opx_ep *opx_ep);
void opx_hfi1_sdma_process_pending(struct fi_opx_ep *opx_ep);

__OPX_FORCE_INLINE__
void fi_opx_ep_do_pending_sdma_work(struct fi_opx_ep *opx_ep)
{
	/* Process pending (sent) SDMA requests to maximize free slots for new requests */
	if (!slist_empty(&opx_ep->tx->sdma_pending_queue)) {
		fi_opx_hfi1_poll_sdma_completion(opx_ep);
		opx_hfi1_sdma_process_pending(opx_ep);
	} else {
		/* If no SDMA requests were pending, then the SDMA completion
		   queue should also not have any outstanding requests */
		assert(opx_ep->hfi->info.sdma.done_index == opx_ep->hfi->info.sdma.fill_index);
	}

	struct opx_sdma_queue *sdma_queue = &opx_ep->tx->sdma_request_queue;
	struct slist_entry *sdma_work_prev = NULL;
	union fi_opx_hfi1_deferred_work *sdma_work =
		(union fi_opx_hfi1_deferred_work *) opx_ep->tx->work_pending[OPX_WORK_TYPE_SDMA].head;

	uint16_t iovs_left = sdma_queue->max_iovs - sdma_queue->num_iovs;
	/* Make a single pass through the SDMA work queue, trying each work item
	   and queuing up SDMA requests. Non-TID work items require a minimum of
	   2 IOV slots, and TID work items require 3. */
	while (sdma_work && (sdma_queue->num_reqs < sdma_queue->slots_avail)
			 && (iovs_left > 1)) {
		struct slist_entry *sdma_work_next = sdma_work->work_elem.slist_entry.next;

		if (OFI_UNLIKELY(iovs_left < 3) && sdma_work->dput.opcode == FI_OPX_HFI_DPUT_OPCODE_RZV_TID) {
			sdma_work_prev = &sdma_work->work_elem.slist_entry;
			sdma_work = (union fi_opx_hfi1_deferred_work *) sdma_work_next;
			continue;
		}

		int rc = sdma_work->work_elem.work_fn(sdma_work);
		if(rc == FI_SUCCESS) {
			if(sdma_work->work_elem.completion_action) {
				sdma_work->work_elem.completion_action(sdma_work);
			}
			if(sdma_work->work_elem.payload_copy) {
				OPX_BUF_FREE(sdma_work->work_elem.payload_copy);
			}
			slist_remove(&opx_ep->tx->work_pending[OPX_WORK_TYPE_SDMA],
				     &sdma_work->work_elem.slist_entry,
				     sdma_work_prev);
			OPX_BUF_FREE(sdma_work);
		} else if (sdma_work->work_elem.work_type == OPX_WORK_TYPE_LAST) {
			slist_remove(&opx_ep->tx->work_pending[OPX_WORK_TYPE_SDMA],
				     &sdma_work->work_elem.slist_entry,
				     sdma_work_prev);
			/* Move this to the pending completion queue,
				since there's nothing left to do but wait */
			slist_insert_tail(&sdma_work->work_elem.slist_entry,
					  &opx_ep->tx->work_pending_completion);
		} else {
			sdma_work_prev = &sdma_work->work_elem.slist_entry;
		}
		sdma_work = (union fi_opx_hfi1_deferred_work *) sdma_work_next;
		iovs_left = sdma_queue->max_iovs - sdma_queue->num_iovs;
	}

	/* Process new SDMA requests to egress */
	if (!slist_empty(&opx_ep->tx->sdma_request_queue.list)) {
		opx_hfi1_sdma_process_requests(opx_ep);
	}
}

__OPX_FORCE_INLINE__
void fi_opx_ep_do_pending_work(struct fi_opx_ep *opx_ep)
{
	/* Clean up all the pending completion work, but stop as soon as we
	   encounter one that isn't done (and requeue that one) */
	uintptr_t work_pending_completion = (uintptr_t) opx_ep->tx->work_pending_completion.head;
	while (work_pending_completion) {
		union fi_opx_hfi1_deferred_work *work =
			(union fi_opx_hfi1_deferred_work *) slist_remove_head(&opx_ep->tx->work_pending_completion);
		work->work_elem.slist_entry.next = NULL;
		int rc = work->work_elem.work_fn(work);
		if(rc == FI_SUCCESS) {
			if(work->work_elem.completion_action) {
				work->work_elem.completion_action(work);
			}
			if(work->work_elem.payload_copy) {
				OPX_BUF_FREE(work->work_elem.payload_copy);
			}
			OPX_BUF_FREE(work);
			work_pending_completion = (uintptr_t)opx_ep->tx->work_pending_completion.head;
		} else {
			assert(work->work_elem.slist_entry.next == NULL);
			slist_insert_head(&work->work_elem.slist_entry, &opx_ep->tx->work_pending_completion);
			work_pending_completion = 0;
		}
	}

	/* Note that SDMA work is not included in this loop, as it is done separately */
	for (enum opx_work_type work_type = OPX_WORK_TYPE_PIO; work_type < OPX_WORK_TYPE_LAST; ++work_type) {
		const uintptr_t work_pending = (const uintptr_t)opx_ep->tx->work_pending[work_type].head;
		if (work_pending) {
			OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "DO-DEFERRED-WORK-%s", OPX_WORK_TYPE_STR[work_type]);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POLL WORK PENDING type %d <start \n", work_type);
			union fi_opx_hfi1_deferred_work *work = (union fi_opx_hfi1_deferred_work *)
								slist_remove_head(&opx_ep->tx->work_pending[work_type]);
			work->work_elem.slist_entry.next = NULL;
			int rc = work->work_elem.work_fn(work);
			if(rc == FI_SUCCESS) {
				if(work->work_elem.completion_action) {
					work->work_elem.completion_action(work);
				}
				if(work->work_elem.payload_copy) {
					OPX_BUF_FREE(work->work_elem.payload_copy);
				}
				OPX_BUF_FREE(work);
			} else {
				assert(work->work_elem.slist_entry.next == NULL);
				if (work->work_elem.work_type == OPX_WORK_TYPE_LAST) {
					/* Move this to the pending completion queue,
					   since there's nothing left to do but wait */
					slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending_completion);
				} else {
					slist_insert_head(&work->work_elem.slist_entry,
							  &opx_ep->tx->work_pending[work->work_elem.work_type]);
				}
			}
			OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "DO-DEFERRED-WORK-%s", OPX_WORK_TYPE_STR[work_type]);
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POLL WORK PENDING type %d %u done>\n", work_type, rc);
		}
	}

	fi_opx_ep_do_pending_sdma_work(opx_ep);
}

__OPX_FORCE_INLINE__
void fi_opx_ep_rx_poll_internal (struct fid_ep *ep,
				 const uint64_t caps,
				 const enum ofi_reliability_kind reliability,
				 const uint64_t hdrq_mask,
				 const enum opx_hfi1_type hfi1_type)
{

	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	const uint64_t rx_caps = (caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) ? caps
				: opx_ep->rx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM);

	if (OFI_LIKELY(hdrq_mask == FI_OPX_HDRQ_MASK_RUNTIME)) {		/* constant compile-time expression */
		FI_OPX_FABRIC_POLL_MANY(ep, FI_OPX_LOCK_NOT_REQUIRED, rx_caps,
					OFI_RELIABILITY_KIND_ONLOAD, FI_OPX_HDRQ_MASK_RUNTIME,
					hfi1_type);
	} else if (hdrq_mask == FI_OPX_HDRQ_MASK_2048) {			/* constant compile-time expression */
		FI_OPX_FABRIC_POLL_MANY(ep, FI_OPX_LOCK_NOT_REQUIRED, rx_caps,
					OFI_RELIABILITY_KIND_ONLOAD, FI_OPX_HDRQ_MASK_2048,
					hfi1_type);
	} else if (hdrq_mask == FI_OPX_HDRQ_MASK_8192) {			/* constant compile-time expression */
		FI_OPX_FABRIC_POLL_MANY(ep, FI_OPX_LOCK_NOT_REQUIRED, rx_caps,
					OFI_RELIABILITY_KIND_ONLOAD, FI_OPX_HDRQ_MASK_8192,
					hfi1_type);
	} else {
		FI_OPX_FABRIC_POLL_MANY(ep, FI_OPX_LOCK_NOT_REQUIRED, rx_caps,
					OFI_RELIABILITY_KIND_ONLOAD, hdrq_mask,
					hfi1_type);
	}

	fi_opx_ep_do_pending_work(opx_ep);

	if (!slist_empty(&opx_ep->reliability->service.work_pending)) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== RELIABILITY SERVICE WORK PENDING\n");
		fi_opx_reliability_service_process_pending(&opx_ep->reliability->service);
	}
}

static inline
void fi_opx_ep_rx_poll (struct fid_ep *ep,
			const uint64_t caps,
			const enum ofi_reliability_kind reliability,
			const uint64_t hdrq_mask,
			const enum opx_hfi1_type hfi1_type)
{
	if (hfi1_type & OPX_HFI1_WFR) {
		fi_opx_ep_rx_poll_internal(ep, caps, reliability, hdrq_mask, OPX_HFI1_WFR);
	} else if (hfi1_type & OPX_HFI1_JKR) {
		fi_opx_ep_rx_poll_internal(ep, caps, reliability, hdrq_mask, OPX_HFI1_JKR);
	} else if (hfi1_type & OPX_HFI1_JKR_9B) {
		fi_opx_ep_rx_poll_internal(ep, caps, reliability, hdrq_mask, OPX_HFI1_JKR_9B);
	} else {
		abort();
	}
}

__OPX_FORCE_INLINE__
int fi_opx_ep_cancel_context(struct fi_opx_ep *opx_ep,
			const uint64_t cancel_context,
			struct opx_context *context,
			const uint64_t rx_op_flags,
			const int lock_required)
{
	FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "unimplemented; abort\n"); abort();

	const uint64_t compare_context = (uint64_t) context->err_entry.op_context;

	if (compare_context == cancel_context) {

		context->byte_counter = 0;
		context->err_entry.flags = rx_op_flags;
		context->err_entry.len = 0;
		context->err_entry.buf = 0;
		context->err_entry.data = 0;
		context->err_entry.tag = context->tag;
		context->err_entry.olen = 0;
		context->err_entry.err = FI_ECANCELED;
		context->err_entry.prov_errno = 0;
		context->err_entry.err_data = NULL;
		context->err_entry.err_data_size = 0;

		/* post an 'error' completion event for the canceled receive */
		if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
		slist_insert_tail((struct slist_entry *) context, opx_ep->rx->cq_err_ptr);

		return FI_ECANCELED;
	}

	return FI_SUCCESS;
}

__OPX_FORCE_INLINE__
int fi_opx_ep_process_context_match_ue_packets(struct fi_opx_ep * opx_ep,
				const uint64_t static_flags,
				struct opx_context * context,
				const uint64_t is_hmem,
				const int lock_required,
				const enum ofi_reliability_kind reliability,
				const enum opx_hfi1_type hfi1_type)
{
	assert(static_flags & (FI_TAGGED | FI_MSG));
	const uint64_t kind = (static_flags & FI_TAGGED) ? FI_OPX_KIND_TAG : FI_OPX_KIND_MSG;

	struct fid_ep * ep = &opx_ep->ep_fid;
	/*
	 * search the unexpected packet queue
	 */
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"searching unexpected queue\n");

	__attribute__((__unused__)) bool from_hash_queue = false;
	struct fi_opx_hfi1_ue_packet *uepkt = fi_opx_ep_find_matching_packet(opx_ep, context, kind, hfi1_type);

#ifndef FI_OPX_MATCH_HASH_DISABLE
	if (!uepkt && kind == FI_OPX_KIND_TAG) {
		from_hash_queue = true;
		uepkt = fi_opx_match_find_uepkt(opx_ep->rx->match_ue_tag_hash,
						context,
						FI_OPX_DEBUG_COUNTERS_GET_PTR(opx_ep));
	}
#endif

	if (uepkt) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "UEPKT found a match, uepkt = %p\n", uepkt);

		uint8_t is_mp_eager = (FI_OPX_HFI_BTH_OPCODE_BASE_OPCODE(uepkt->hdr.bth.opcode) == FI_OPX_HFI_BTH_OPCODE_MSG_MP_EAGER_FIRST);

		const unsigned is_intranode = opx_lrh_is_intranode(&(uepkt->hdr), hfi1_type);
		if (is_mp_eager) {
			opx_ep_complete_receive_operation(ep,
							 &uepkt->hdr,
							 &uepkt->payload,
							 uepkt->hdr.match.ofi_tag,
							 context,
							 uepkt->hdr.bth.opcode,
							 OPX_MULTI_RECV_FALSE,
							 is_intranode,
							 is_hmem,
							 lock_required,
							 reliability,
							 hfi1_type);

			/* Since this is the first multi-packet eager packet,
			   the uid portion of the mp_egr_id will be this packet's PSN */
			opx_lid_t slid;
			if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
				slid = (opx_lid_t)__be16_to_cpu24((__be16)uepkt->hdr.lrh_9B.slid);
			} else {
				slid = (opx_lid_t)__le24_to_cpu((__le24)(uepkt->hdr.lrh_16B.slid20 << 20) | (uepkt->hdr.lrh_16B.slid));
			}
			const union fi_opx_mp_egr_id mp_egr_id = {
					.uid = uepkt->hdr.reliability.psn,
					.slid_origin_tx = (slid << 8) | uepkt->hdr.reliability.origin_tx
			};

			fi_opx_ep_rx_process_pending_mp_eager_ue(ep, context, mp_egr_id, is_intranode,
					lock_required, reliability, hfi1_type);

			if (context->byte_counter) {
				context->mp_egr_id = mp_egr_id;
				slist_insert_tail((struct slist_entry *) context, &opx_ep->rx->mp_egr_queue.mq);
			} else {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.recv_completed_process_context);

				context->next = NULL;
				if (OFI_UNLIKELY(context->err_entry.err == FI_ETRUNC)) {
					slist_insert_tail((struct slist_entry *) context, opx_ep->rx->cq_err_ptr);
				} else {
					fi_opx_enqueue_completed(opx_ep->rx->cq_completed_ptr, context,
								lock_required);
				}
			}
		} else {
			opx_ep_complete_receive_operation(ep,
						   &uepkt->hdr,
						   &uepkt->payload,
						   uepkt->hdr.match.ofi_tag,
						   context,
						   uepkt->hdr.bth.opcode,
						   OPX_MULTI_RECV_FALSE,
						   is_intranode,
						   is_hmem,
						   lock_required,
						   reliability,
						   hfi1_type);
		}

#ifndef FI_OPX_MATCH_HASH_DISABLE
		if (from_hash_queue) {
			fi_opx_match_ue_hash_remove(uepkt, opx_ep->rx->match_ue_tag_hash);
		} else {
			fi_opx_hfi1_ue_packet_slist_remove_item(uepkt, &opx_ep->rx->queue[kind].ue);
		}
#else
		fi_opx_hfi1_ue_packet_slist_remove_item(uepkt, &opx_ep->rx->queue[kind].ue);
#endif

		return 0;
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"nothing found on unexpected queue; adding to match queue\n");
	/*
	 * no unexpected headers were matched; add this match information
	 * (context) to the appropriate match queue
	 */
	context->next = NULL;
	slist_insert_tail((struct slist_entry *) context, &opx_ep->rx->queue[kind].mq);

	return 0;
}

/* rx_op_flags is only checked for FI_PEEK | FI_CLAIM | FI_MULTI_RECV
 * rx_op_flags is only used if FI_PEEK | FI_CLAIM
 *
 * The "normal" data movement functions, such as fi_[t]recv(), can safely
 * specify '0' for rx_op_flags in order to reduce code path.
 *
 * TODO - use payload pointer? keep data in hfi eager buffer as long
 * as possible to avoid memcpy?
 */
__OPX_FORCE_INLINE__
int fi_opx_ep_rx_process_context (
		struct fi_opx_ep * opx_ep,
		const uint64_t static_flags,
		struct opx_context *context,
		const uint64_t rx_op_flags,
		const uint64_t is_hmem,
		const int lock_required, const enum fi_av_type av_type,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{

	if (OFI_LIKELY((rx_op_flags & (FI_PEEK | FI_CLAIM | FI_MULTI_RECV)) == 0)) {
		if (is_hmem) {	/* branch should compile out */
			return fi_opx_ep_process_context_match_ue_packets(opx_ep, static_flags, context,
									  OPX_HMEM_TRUE, lock_required,
									  reliability, hfi1_type);
		}

		return fi_opx_ep_process_context_match_ue_packets(opx_ep, static_flags, context,
								  OPX_HMEM_FALSE, lock_required,
								  reliability, hfi1_type);
	} else {

		/*
		 * Not for critical path: peek, or claim, or multi-receive
		 * context information
		 */
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"process peek, claim, or multi-receive context\n");

		fi_opx_ep_rx_process_context_noinline(opx_ep, static_flags,
			context, rx_op_flags, is_hmem, lock_required, av_type,
			reliability, hfi1_type);
	}

	return 0;
}

__OPX_FORCE_INLINE__
fi_addr_t fi_opx_ep_get_src_addr(struct fi_opx_ep *opx_ep,
				const enum fi_av_type av_type,
				const fi_addr_t msg_addr)
{
	if (av_type == FI_AV_MAP) {	/* constant compile-time expression */
		return msg_addr;
	}

	if (av_type == FI_AV_TABLE) {
		return OFI_LIKELY(msg_addr != FI_ADDR_UNSPEC) ?
				opx_ep->rx->av_addr[msg_addr].fi :
				FI_ADDR_UNSPEC;
	}

	assert(av_type == FI_AV_UNSPEC);

	if (opx_ep->av_type != FI_AV_TABLE) {
		return msg_addr;
	}

	/* use runtime endpoint value*/
	return OFI_LIKELY(msg_addr != FI_ADDR_UNSPEC) ?
			opx_ep->rx->av_addr[msg_addr].fi :
			FI_ADDR_UNSPEC;
}

/*
 * =========================== Application-facing ===========================
 */

__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_rx_recv_internal (struct fi_opx_ep *opx_ep,
		void *buf, size_t len, void *desc,
		fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		void *user_context,
		const int lock_required, const enum fi_av_type av_type,
		const uint64_t static_flags,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{
	assert(((static_flags & (FI_TAGGED | FI_MSG)) == FI_TAGGED) ||
		((static_flags & (FI_TAGGED | FI_MSG)) == FI_MSG));

	FI_OPX_DEBUG_COUNTERS_INC_COND(static_flags & FI_MSG, opx_ep->debug_counters.recv.posted_recv_msg);
	FI_OPX_DEBUG_COUNTERS_INC_COND(static_flags & FI_TAGGED, opx_ep->debug_counters.recv.posted_recv_tag);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== POST RECV: context = %p\n",
		user_context);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "POST-RECV");

	struct opx_context *context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
	if (OFI_UNLIKELY(context == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
		return -FI_ENOMEM;
	}

	const uint64_t rx_op_flags = opx_ep->rx->op_flags;
	uint64_t rx_caps = opx_ep->rx->caps;

	context->next = NULL;
	context->err_entry.err = 0;
	context->err_entry.op_context = user_context;
	context->flags = rx_op_flags;
	context->len = len;
	context->buf = buf;
	context->src_addr = (rx_caps & FI_DIRECTED_RECV)
		?  fi_opx_ep_get_src_addr(opx_ep, av_type, src_addr)
		: FI_ADDR_UNSPEC;
	context->tag = tag;
	context->ignore = ignore;
	context->byte_counter = (uint64_t)-1;


#ifdef FI_OPX_TRACE
	fprintf(stderr,"fi_opx_recv_generic from source addr:\n");
	FI_OPX_ADDR_DUMP(&context->src_addr);
#endif

	assert(IS_PROGRESS_MANUAL(opx_ep->domain));

	if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"process context (check unexpected queue, append match queue)\n");

#ifdef OPX_HMEM
	uint64_t hmem_device;
	enum fi_hmem_iface hmem_iface = desc ? fi_opx_hmem_get_iface(buf, desc, &hmem_device) : FI_HMEM_SYSTEM;
	if (hmem_iface != FI_HMEM_SYSTEM) {
		FI_OPX_DEBUG_COUNTERS_INC_COND(static_flags & FI_MSG, opx_ep->debug_counters.hmem.posted_recv_msg);
		FI_OPX_DEBUG_COUNTERS_INC_COND(static_flags & FI_TAGGED, opx_ep->debug_counters.hmem.posted_recv_tag);
		struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) &context->hmem_info_qws[0];
		hmem_info->iface = hmem_iface;
		hmem_info->device = hmem_device;
		hmem_info->hmem_dev_reg_handle = ((struct fi_opx_mr *)desc)->hmem_dev_reg_handle;

		context->flags |= FI_OPX_CQ_CONTEXT_HMEM;

		fi_opx_ep_rx_process_context(opx_ep,
					static_flags,
					context,
					0, // rx_op_flags
					OPX_HMEM_TRUE,
					lock_required,
					av_type,
					reliability,
					hfi1_type);
	} else
#endif
	{
		fi_opx_ep_rx_process_context(opx_ep,
					static_flags,
					context,
					0, // rx_op_flags
					OPX_HMEM_FALSE,
					lock_required,
					av_type,
					reliability,
					hfi1_type);
	}

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECV");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== POST RECV RETURN\n");

	return 0;
}

/*
 * \note The opx provider asserts the following mode bits which affect
 * 	the behavior of this routine:
 *
 * 	- 'FI_ASYNC_IOV' mode bit which requires the application to maintain
 * 	  the 'msg->msg_iov' iovec array until the operation completes
 *
 * 	- 'FI_LOCAL_MR' mode bit which allows the provider to ignore the 'desc'
 * 	  parameter .. no memory regions are required to access the local
 * 	  memory
 */
static inline
ssize_t fi_opx_ep_rx_recvmsg_internal (struct fi_opx_ep *opx_ep,
		const struct fi_msg *msg, uint64_t flags,
		const int lock_required, const enum fi_av_type av_type,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,"===================================== POST RECVMSG\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "POST-RECVMSG");
	FI_OPX_DEBUG_COUNTERS_INC_COND(!(flags & FI_MULTI_RECV), opx_ep->debug_counters.recv.posted_recv_msg);
	FI_OPX_DEBUG_COUNTERS_INC_COND((flags & FI_MULTI_RECV), opx_ep->debug_counters.recv.posted_multi_recv);
	assert(!lock_required);

	struct opx_context *context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
	if (OFI_UNLIKELY(context == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_ERROR, "POST-RECVMSG");
		return -FI_ENOMEM;
	}
	context->next = NULL;
	context->err_entry.err = 0;
	context->err_entry.op_context = msg->context;

	if (OFI_LIKELY(flags & FI_MULTI_RECV)) {
		uint64_t len = msg->msg_iov[0].iov_len;
		void * base = msg->msg_iov[0].iov_base;

		assert(msg->iov_count == 1);
		assert(base != NULL);
		if ((uintptr_t)base & 0x07ull) {
			uintptr_t new_base = (((uintptr_t)base + 8) & (~0x07ull));
			len -= (new_base - (uintptr_t)base);
			base = (void *)new_base;
		}
		assert(((uintptr_t)base & 0x07ull) == 0);
		assert(len >= (sizeof(struct opx_context) + opx_ep->rx->min_multi_recv));
		context->flags = FI_MULTI_RECV;
		context->len = len - sizeof(struct opx_context);
		context->buf = (void *)((uintptr_t)base + sizeof(struct opx_context));
		context->src_addr = fi_opx_ep_get_src_addr(opx_ep, av_type, msg->addr);
		context->byte_counter = 0;
		context->ignore = (uint64_t)-1;

		ssize_t rc = fi_opx_ep_rx_process_context(opx_ep, FI_MSG,
						context, flags,
						OPX_HMEM_FALSE,
						lock_required, av_type,
						reliability,
						hfi1_type);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POST RECVMSG RETURN\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECVMSG");
		return rc;

	} else if (msg->iov_count == 0) {
		context->flags = flags;
		context->len = 0;
		context->buf = NULL;
		context->src_addr = fi_opx_ep_get_src_addr(opx_ep, av_type, msg->addr);
		context->tag = 0;
		context->ignore = (uint64_t)-1;
		context->byte_counter = (uint64_t)-1;

		ssize_t rc = fi_opx_ep_rx_process_context(opx_ep, FI_MSG,
						context, flags,
						OPX_HMEM_FALSE,
						lock_required, av_type,
						reliability,
						hfi1_type);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POST RECVMSG RETURN\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECVMSG");
		return rc;
	}

#ifdef OPX_HMEM
	/* NOTE: Assume that all IOVs reside in the same HMEM space */
	uint64_t hmem_device;
	enum fi_hmem_iface hmem_iface = fi_opx_hmem_get_iface(msg->msg_iov[0].iov_base,
							      msg->desc ? msg->desc[0] : NULL,
							      &hmem_device);
#ifndef NDEBUG
	if (msg->iov_count > 1) {
		for (int i = 1; i < msg->iov_count; ++i) {
			uint64_t tmp_hmem_device;
			enum fi_hmem_iface tmp_hmem_iface =
				fi_opx_hmem_get_iface(msg->msg_iov[i].iov_base,
						      msg->desc ? msg->desc[i] : NULL,
						      &tmp_hmem_device);
			assert(tmp_hmem_iface == hmem_iface);
			assert(tmp_hmem_device == hmem_device);
		}
	}
#endif
	if (hmem_iface != FI_HMEM_SYSTEM) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.hmem.posted_recv_msg);
		context->flags = flags | FI_OPX_CQ_CONTEXT_HMEM;
		context->len = msg->msg_iov[0].iov_len;
		context->buf = msg->msg_iov[0].iov_base;
		context->byte_counter = (uint64_t)-1;
		context->src_addr = fi_opx_ep_get_src_addr(opx_ep, av_type, msg->addr);
		context->tag = 0;
		context->ignore = (uint64_t)-1;
		context->msg.iov_count = msg->iov_count;
		context->msg.iov = (struct iovec *)msg->msg_iov;

		struct fi_opx_hmem_info *hmem_info = (struct fi_opx_hmem_info *) &context->hmem_info_qws[0];
		hmem_info->iface = hmem_iface;
		hmem_info->device = hmem_device;

		ssize_t rc = fi_opx_ep_rx_process_context(opx_ep, FI_MSG,
						context, context->flags,
						OPX_HMEM_TRUE,
						lock_required, av_type,
						reliability,
						hfi1_type);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POST RECVMSG (HMEM) RETURN\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECVMSG");
		return rc;
	}
#endif

	if (msg->iov_count == 1) {
		context->flags = flags;
		context->len = msg->msg_iov[0].iov_len;
		context->buf = msg->msg_iov[0].iov_base;
		context->src_addr = fi_opx_ep_get_src_addr(opx_ep, av_type, msg->addr);
		context->tag = 0;
		context->ignore = (uint64_t)-1;
		context->byte_counter = (uint64_t)-1;

		ssize_t rc = fi_opx_ep_rx_process_context(opx_ep, FI_MSG,
						context, flags,
						OPX_HMEM_FALSE,
						lock_required, av_type,
						reliability,
						hfi1_type);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== POST RECVMSG RETURN\n");
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECVMSG");
		return rc;
	}

	/* msg->iov_count > 1 */

	context->flags = flags;
	context->byte_counter = (uint64_t)-1;
	context->src_addr = fi_opx_ep_get_src_addr(opx_ep, av_type, msg->addr);
	context->tag = 0;
	context->ignore = (uint64_t)-1;
	context->msg.iov_count = msg->iov_count;
	context->msg.iov = (struct iovec *)msg->msg_iov;

	ssize_t rc = fi_opx_ep_rx_process_context(opx_ep, FI_MSG,
					context, flags,
					OPX_HMEM_FALSE,
					lock_required, av_type,
					reliability,
					hfi1_type);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== POST RECVMSG RETURN\n");
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "POST-RECVMSG");
	return rc;
}

__OPX_FORCE_INLINE__
uint64_t fi_opx_ep_tx_do_cq_completion(struct fi_opx_ep *opx_ep,
					const unsigned override_flags,
					uint64_t tx_op_flags)
{
	/*
	 * ==== NOTE_SELECTIVE_COMPLETION ====
	 *
	 * FI_SELECTIVE_COMPLETION essentially changes the default from:
	 *
	 *   "generate a completion of some kind if FI_TRANSMIT is
	 *   also specified"
	 *
	 * to
	 *
	 *   "only generate a completion of some kind if FI_TRANSMIT
	 *   and FI_COMPLETION are also specified".
	 *
	 * and as specified in commit 8bf9bf74b719f265186a7dea1c1e1f26a24bfb5a:
	 *
	 *   "FI_COMPLETION is only needed in cases where an endpoint was
	 *   bound to a CQ or counter with the FI_SELECTIVE_COMPLETION flag."
	 */

	if (!override_flags) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "=================== DO TX CQ COMPLETION %u (tx)\n", opx_ep->tx->do_cq_completion);
		return opx_ep->tx->do_cq_completion;
	}

	const uint64_t selective_completion = FI_SELECTIVE_COMPLETION | FI_TRANSMIT | FI_COMPLETION;

	const uint64_t flags = tx_op_flags | opx_ep->tx->cq_bind_flags;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "=================== DO TX CQ COMPLETION %u (flags)\n", ((flags & selective_completion) == selective_completion) ||
		((flags & (FI_SELECTIVE_COMPLETION | FI_TRANSMIT)) == FI_TRANSMIT));
	return  ((flags & selective_completion) == selective_completion) ||
		((flags & (FI_SELECTIVE_COMPLETION | FI_TRANSMIT)) == FI_TRANSMIT);
}

__OPX_FORCE_INLINE__
void fi_opx_ep_tx_cq_completion_rzv(struct fid_ep *ep,
				void *context,
				const size_t len,
				const int lock_required,
				const uint64_t tag,
				const uint64_t caps)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	assert(context);
	assert(((uintptr_t)context & 0x07ull) == 0);	/* must be 8 byte aligned */
	struct opx_context *opx_context = (struct opx_context *) context;
	opx_context->flags =  FI_SEND | (caps & (FI_TAGGED | FI_MSG));
	opx_context->len = len;
	opx_context->buf = NULL;	/* receive data buffer */
	opx_context->tag = tag;
	opx_context->next = NULL;

	if (lock_required) { fprintf(stderr, "%s:%s():%d\n", __FILE__, __func__, __LINE__); abort(); }
	slist_insert_tail((struct slist_entry *) opx_context, opx_ep->tx->cq_pending_ptr);
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_hfi1_tx_send_try_mp_egr (struct fid_ep *ep,
		const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, uint64_t tag, void* context,
		const uint32_t data, int lock_required,
		const unsigned override_flags, uint64_t tx_op_flags,
		const uint64_t caps,
		const enum ofi_reliability_kind reliability,
		const uint64_t do_cq_completion,
		const enum fi_hmem_iface hmem_iface,
		const uint64_t hmem_device,
		const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr = { .fi = dest_addr };

	assert (!fi_opx_hfi1_tx_is_intranode(opx_ep, addr, caps));
	assert (len > FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type));

	const uint64_t bth_rx = ((uint64_t)addr.hfi1_rx) << OPX_BTH_RX_SHIFT;
	const uint64_t lrh_dlid = hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B) ? FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(addr.lid) : addr.lid;
	const uint64_t pbc_dlid = OPX_PBC_DLID_TO_PBC_DLID(addr.lid, hfi1_type);

	/* Write the first packet */
	uint32_t first_packet_psn;
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- MULTI-PACKET EAGER USER (begin)\n");

	uint8_t *buf_bytes_ptr = (uint8_t *) buf;
	ssize_t rc;
	rc = fi_opx_hfi1_tx_send_mp_egr_first_common (opx_ep, (void **) &buf_bytes_ptr, len, desc,
						      opx_ep->hmem_copy_buf, pbc_dlid, bth_rx, lrh_dlid,
						      addr, tag, data, lock_required,
						      tx_op_flags, caps, reliability, &first_packet_psn,
						      hmem_iface, hmem_device, hfi1_type);

	if (rc != FI_SUCCESS) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_fall_back_to_rzv);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, HFI -- MULTI-PACKET EAGER USER (return %zd)\n", rc);

		return rc;
	}

	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_first_packets);

	/* The first packet was successful. We're now committed to finishing this */
	ssize_t payload_remaining = len - FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);
	uint32_t payload_offset = FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);
	buf_bytes_ptr += FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- MULTI-PACKET EAGER USER FIRST NTH (payload_remaining %zu)\n", payload_remaining);

	/* Write all the full nth packets */
	while (payload_remaining >= FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type)) {
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			rc = fi_opx_hfi1_tx_send_mp_egr_nth(opx_ep, (void *)buf_bytes_ptr, payload_offset,
							first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
							lock_required, reliability, hfi1_type);
		} else {
			rc = fi_opx_hfi1_tx_send_mp_egr_nth_16B(opx_ep, (void *)buf_bytes_ptr, payload_offset,
							first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
							lock_required, reliability, hfi1_type);
		}

		if (rc != FI_SUCCESS) {
			if (rc == -FI_ENOBUFS) {
				/* Insufficient credits. Try forcing a credit return and retry. */
				fi_opx_force_credit_return(ep, addr.fi, addr.hfi1_rx, caps, hfi1_type);
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_nth_force_cr);
			} else {
				fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_full_replay_buffer_rx_poll);
			}

			do {
				if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
					rc = fi_opx_hfi1_tx_send_mp_egr_nth(opx_ep, (void *)buf_bytes_ptr, payload_offset,
								first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
								lock_required, reliability, hfi1_type);
				} else {
					rc = fi_opx_hfi1_tx_send_mp_egr_nth_16B(opx_ep, (void *)buf_bytes_ptr, payload_offset,
								first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
								lock_required, reliability, hfi1_type);
				}

				if (rc == -FI_EAGAIN) {
					FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_full_replay_buffer_rx_poll);
					fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
				}
			} while (rc != FI_SUCCESS);
		}
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_nth_packets);

		payload_remaining -= FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);
		buf_bytes_ptr += FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);
		payload_offset += FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, HFI -- MULTI-PACKET EAGER USER (payload_remaining %zu)\n", payload_remaining);
	}


	/* Write all the last packet (if necessary) */
	if (payload_remaining > 0) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, HFI -- MULTI-PACKET EAGER USER LAST (payload_remaining %zu)\n", payload_remaining);
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			rc = fi_opx_hfi1_tx_send_mp_egr_last(opx_ep, (void *)buf_bytes_ptr, payload_offset,
							payload_remaining,
							first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
							lock_required, reliability, hfi1_type);
		} else {
			rc = fi_opx_hfi1_tx_send_mp_egr_last_16B(opx_ep, (void *)buf_bytes_ptr, payload_offset,
							payload_remaining,
							first_packet_psn, pbc_dlid, bth_rx, lrh_dlid, addr,
							lock_required, reliability, hfi1_type);
		}

		if (rc != FI_SUCCESS) {
			if (rc == -FI_ENOBUFS) {
				/* Insufficient credits. Try forcing a credit return and retry. */
				fi_opx_force_credit_return(ep, addr.fi, addr.hfi1_rx, caps,hfi1_type);
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_nth_force_cr);
			} else {
				fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_full_replay_buffer_rx_poll);
			}

			do {
				if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
					rc = fi_opx_hfi1_tx_send_mp_egr_last(opx_ep, (void *)buf_bytes_ptr, payload_offset,
								payload_remaining, first_packet_psn, pbc_dlid, bth_rx, lrh_dlid,
								addr, lock_required, reliability, hfi1_type);
				} else {
					rc = fi_opx_hfi1_tx_send_mp_egr_last_16B(opx_ep, (void *)buf_bytes_ptr, payload_offset,
								payload_remaining, first_packet_psn, pbc_dlid, bth_rx, lrh_dlid,
								addr, lock_required, reliability, hfi1_type);
				}
				if (rc == -FI_EAGAIN) {
					FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_full_replay_buffer_rx_poll);
					fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
				}
			} while (rc != FI_SUCCESS);
		}
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.mp_eager.send_nth_packets);
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, HFI -- MULTI-PACKET EAGER USER LAST (payload_remaining %zu)\n", payload_remaining);

	}

	if (OFI_LIKELY(do_cq_completion)) {
		fi_opx_ep_tx_cq_inject_completion(ep, context, len, lock_required, tag, caps);
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- MULTI-PACKET EAGER USER (end)\n");

	return FI_SUCCESS;
}

#ifndef FI_OPX_EP_TX_SEND_EAGER_MAX_RETRIES
#define FI_OPX_EP_TX_SEND_EAGER_MAX_RETRIES 0x200000
#endif

__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_tx_send_try_eager(struct fid_ep *ep,
				const void *buf, size_t len, void *desc,
				const union fi_opx_addr addr, uint64_t tag, void *context,
				const struct iovec *local_iov, size_t niov, size_t total_len,
				const uint32_t data,
				const int lock_required,
				const unsigned is_contiguous,
				const unsigned override_flags,
				uint64_t tx_op_flags,
				const uint64_t caps,
				const enum ofi_reliability_kind reliability,
				const uint64_t do_cq_completion,
				const enum fi_hmem_iface hmem_iface,
				const uint64_t hmem_device,
				const bool mp_eager_fallback,
				const enum opx_hfi1_type hfi1_type)
{
	ssize_t rc;
	if(is_contiguous) {
		rc = FI_OPX_FABRIC_TX_SEND_EGR(ep, buf, len,
					       desc, addr.fi, tag, context, data,
					       lock_required,
					       override_flags, tx_op_flags, addr.hfi1_rx,
					       caps, reliability, do_cq_completion,
					       hmem_iface, hmem_device, hfi1_type);
	} else {
		rc = FI_OPX_FABRIC_TX_SENDV_EGR(ep, local_iov, niov, total_len,
						desc, addr.fi, tag, context, data,
						lock_required,
						override_flags, tx_op_flags, addr.hfi1_rx,
						caps, reliability, do_cq_completion,
						hmem_iface, hmem_device, hfi1_type);
	}
	if (OFI_LIKELY(rc == FI_SUCCESS)) {
		return rc;

#ifndef FI_OPX_MP_EGR_DISABLE
	} else if (rc == -FI_ENOBUFS && mp_eager_fallback) {
		/* Insufficient credits. If the payload is big enough,
		   fall back to Multi-packet eager to try sending this in
		   smaller chunks. */
		return rc;
#endif
	}

	if (rc == -FI_ENOBUFS) {
		/* Insufficient credits. Try forcing a credit return and retry. */
		fi_opx_force_credit_return(ep, addr.fi, addr.hfi1_rx, caps,hfi1_type);
	} else {
		/* Likely full replay buffers or waiting for reliability handshake init.
		   A poll might help */
		fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
	}

	/* Note that we'll only iterate this loop more than once if we got here
	   due to insufficient credits. */
	uint64_t loop = 0;
	do {
		if(is_contiguous) {
			rc = FI_OPX_FABRIC_TX_SEND_EGR(ep, buf, len,
						       desc, addr.fi, tag, context, data,
						       lock_required,
						       override_flags, tx_op_flags, addr.hfi1_rx,
						       caps, reliability, do_cq_completion,
							hmem_iface, hmem_device, hfi1_type);
		} else {
			rc = FI_OPX_FABRIC_TX_SENDV_EGR(ep, local_iov, niov, total_len,
							desc, addr.fi, tag, context, data,
							lock_required,
							override_flags, tx_op_flags, addr.hfi1_rx,
							caps, reliability, do_cq_completion,
							hmem_iface, hmem_device, hfi1_type);
		}
		fi_opx_ep_rx_poll(ep, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
	} while (rc == -FI_ENOBUFS && loop++ < FI_OPX_EP_TX_SEND_EAGER_MAX_RETRIES);

	return rc;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_tx_send_rzv(struct fid_ep *ep,
			const void *buf, size_t len, void *desc,
			const union fi_opx_addr addr, uint64_t tag, void *context,
			const struct iovec *local_iov, size_t niov, size_t total_len,
			const uint32_t data,
			const int lock_required,
			const unsigned is_contiguous,
			const unsigned override_flags,
			uint64_t tx_op_flags,
			const uint64_t caps,
			const enum ofi_reliability_kind reliability,
			const uint64_t do_cq_completion,
			const enum fi_hmem_iface hmem_iface,
			const uint64_t hmem_device,
			const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	ssize_t rc;

	do {
		if (is_contiguous) {
			rc = FI_OPX_FABRIC_TX_SEND_RZV(
				ep, buf, len, desc, addr.fi, tag, context, data,
				lock_required, override_flags, tx_op_flags, addr.hfi1_rx,
				caps, reliability,
				do_cq_completion,
				hmem_iface, hmem_device, hfi1_type);
		} else {
			rc = FI_OPX_FABRIC_TX_SENDV_RZV(
				ep, local_iov, niov, total_len, desc, addr.fi, tag,
				context, data, lock_required, override_flags, tx_op_flags,
				addr.hfi1_rx,
				caps, reliability,
				do_cq_completion,
				hmem_iface, hmem_device, hfi1_type);
		}

		if (OFI_UNLIKELY(rc == -EAGAIN)) {
			fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
		}
	} while (rc == -EAGAIN);

	return rc;
}

static inline
ssize_t fi_opx_ep_tx_send_internal (struct fid_ep *ep,
		const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, uint64_t tag, void *context,
		const uint32_t data,
		const int lock_required,
		const enum fi_av_type av_type,
		const unsigned is_contiguous,
		const unsigned override_flags,
		uint64_t tx_op_flags,
		const uint64_t caps,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND");

	uint64_t hmem_device;
	enum fi_hmem_iface hmem_iface;
	if (desc) {
		hmem_iface = fi_opx_hmem_get_iface(buf, desc, &hmem_device);
	} else {
		hmem_iface = FI_HMEM_SYSTEM;
		hmem_device = 0ul;
	}

	assert(is_contiguous == OPX_CONTIG_FALSE || is_contiguous == OPX_CONTIG_TRUE);

	// Exactly one of FI_MSG or FI_TAGGED should be on
	assert((caps & (FI_MSG | FI_TAGGED)) &&
		((caps & (FI_MSG | FI_TAGGED)) != (FI_MSG | FI_TAGGED)));

	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	ssize_t ret;
	ret = fi_opx_ep_tx_check(opx_ep->tx, av_type);
	if (ret) return ret;
#endif

	assert(dest_addr != FI_ADDR_UNSPEC);
	assert((FI_AV_TABLE == opx_ep->av_type) || (FI_AV_MAP == opx_ep->av_type));
	const union fi_opx_addr addr = FI_OPX_EP_AV_ADDR(av_type,opx_ep,dest_addr);

	ssize_t rc = 0;

	/* Resynch of all the Reliability Protocol(RP) related data maintained by the
	 * remote EP, must be done first before any other RP related operations are
	 * done with the remote EP.
	 */
	if (opx_ep->daos_info.do_resynch_remote_ep) {
		rc = fi_opx_reliability_do_remote_ep_resynch(ep, addr, context, caps);
		if (OFI_UNLIKELY(rc == -FI_EAGAIN)) {
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND");
			return rc;
		}
	}

	if (OFI_UNLIKELY(!opx_reliability_ready(ep,
			&opx_ep->reliability->state,
			addr.lid,
			addr.hfi1_rx,
			addr.reliability_rx,
			reliability))) {
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY, FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND");
		return -FI_EAGAIN;
	}

	size_t total_len = len;
	const struct iovec *local_iov = NULL;
	size_t niov = 0;
	if(!is_contiguous) { /* constant compile time expression */
		ssize_t i;
		local_iov = buf;
		niov = len;
		total_len = 0;
		for(i=0; i < niov; i++) {
			total_len += local_iov[i].iov_len;
		}
	}

	const uint64_t do_cq_completion =
		fi_opx_ep_tx_do_cq_completion(opx_ep, override_flags, tx_op_flags);

	if (total_len < opx_ep->tx->rzv_min_payload_bytes) {
		const bool mp_eager_fallback = (total_len >  FI_OPX_MP_EGR_CHUNK_PAYLOAD_SIZE(hfi1_type) &&
						total_len <= opx_ep->tx->mp_eager_max_payload_bytes);
		if (total_len <= opx_ep->tx->pio_max_eager_tx_bytes) {
			rc = fi_opx_ep_tx_send_try_eager(ep, buf, len, desc, addr, tag, context, local_iov,
							niov, total_len, data, lock_required, is_contiguous,
							override_flags, tx_op_flags, caps, reliability,
							do_cq_completion, hmem_iface, hmem_device,
							mp_eager_fallback, hfi1_type);
			if (OFI_LIKELY(rc == FI_SUCCESS)) {
				OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND");
				return rc;
			}
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND");
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== SEND -- Eager send failed, trying next method\n");
		}

#ifndef FI_OPX_MP_EGR_DISABLE
		/* If hmem_iface != FI_HMEM_SYSTEM, we skip MP EGR because RZV yields better performance for devices */
		if (is_contiguous &&
		    mp_eager_fallback &&
		    !fi_opx_hfi1_tx_is_intranode(opx_ep, addr, caps) &&
			(caps & FI_TAGGED) && hmem_iface == FI_HMEM_SYSTEM) {

			rc = fi_opx_hfi1_tx_send_try_mp_egr(ep, buf, len, desc, addr.fi, tag,
							context, data, lock_required, override_flags,
							tx_op_flags, caps, reliability, do_cq_completion,
						  FI_HMEM_SYSTEM, 0ul, hfi1_type);
			if (OFI_LIKELY(rc == FI_SUCCESS)) {
				OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND");
				return rc;
			}
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND");
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== SEND -- MP-Eager send failed, trying next method\n");
		}
#endif

		if (OFI_UNLIKELY(total_len < FI_OPX_HFI1_TX_MIN_RZV_PAYLOAD_BYTES)) {
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN,"SEND");
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				"===================================== SEND -- FI_EAGAIN Can't do RZV with payload length = %ld\n",len);
			return -FI_EAGAIN;
		}
	}

	rc = fi_opx_ep_tx_send_rzv(ep,
				buf, len, desc,
				addr, tag, context,
				local_iov, niov, total_len,
				data,
				lock_required,
				is_contiguous,
				override_flags,
				tx_op_flags,
				caps,
				reliability,
				do_cq_completion,
				hmem_iface, hmem_device,
				hfi1_type);

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND (end)\n");

	return rc;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_tx_send(struct fid_ep *ep,
			  const void *buf, size_t len, void *desc,
			  fi_addr_t dest_addr, uint64_t tag, void *context,
			  const uint32_t data,
			  const int lock_required,
			  const enum fi_av_type av_type,
			  const unsigned is_contiguous,
			  const unsigned override_flags,
			  uint64_t tx_op_flags,
			  const uint64_t caps,
			  const enum ofi_reliability_kind reliability,
			  const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	ssize_t rc = fi_opx_ep_tx_send_internal(ep, buf, len, desc, dest_addr,
						tag, context, data, FI_OPX_LOCK_NOT_REQUIRED, av_type,
						is_contiguous, override_flags,
						tx_op_flags, caps, reliability,
						hfi1_type);

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}


__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_tx_inject_internal (struct fid_ep *ep,
		const void *buf,
		size_t len,
		fi_addr_t dest_addr,
		uint64_t tag,
		const uint32_t data,
		const int lock_required,
		const enum fi_av_type av_type,
		uint64_t tx_op_flags,
		const uint64_t caps,
		const enum ofi_reliability_kind reliability,
		const enum opx_hfi1_type hfi1_type)
{
	// Exactly one of FI_MSG or FI_TAGGED should be on
	assert((caps & (FI_MSG | FI_TAGGED)) &&
		((caps & (FI_MSG | FI_TAGGED)) != (FI_MSG | FI_TAGGED)));

	// This message check is a workaround for some versions of MPI
	// that do not check or enforce inject limits for FI_MSG
	// Remove this workaround when MPI's are upgraded to obey these limits
	if(caps & FI_MSG && len > FI_OPX_HFI1_PACKET_IMM) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
					 "Downgrading inject to eager due to > inject limit (begin)\n");

		return fi_opx_ep_tx_send_internal (ep, buf, len, NULL, dest_addr, tag,
						   NULL, // context
						   0, // data
						   lock_required,
						   av_type,
						   OPX_CONTIG_TRUE,
						   OPX_FLAGS_OVERRIDE_TRUE,
						   FI_SELECTIVE_COMPLETION, // op flags to turn off context
						   caps,
						   reliability,
						   hfi1_type);
	} else {
		assert(len <= FI_OPX_HFI1_PACKET_IMM);
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== INJECT (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "INJECT");

	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

#ifndef NDEBUG
	ssize_t ret;
	ret = fi_opx_ep_tx_check(opx_ep->tx, av_type);
	if (ret) return ret;
#endif
	assert(dest_addr != FI_ADDR_UNSPEC);
	assert((FI_AV_TABLE == opx_ep->av_type) || (FI_AV_MAP == opx_ep->av_type));
	const union fi_opx_addr addr = FI_OPX_EP_AV_ADDR(av_type,opx_ep,dest_addr);

	const ssize_t rc = FI_OPX_FABRIC_TX_INJECT(ep, buf, len, addr.fi, tag, data,
			lock_required, addr.hfi1_rx, tx_op_flags, caps, reliability, hfi1_type);

	if (OFI_UNLIKELY(rc == -EAGAIN)) {
		// In this case we are probably out of replay buffers. To deal
		// with this, we do a poll which may send a ping and will
		// process any incoming ACKs, hopefully releasing a buffer for
		// reuse.
		fi_opx_ep_rx_poll(&opx_ep->ep_fid, 0, OPX_RELIABILITY,
			FI_OPX_HDRQ_MASK_RUNTIME, hfi1_type);
	}

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "INJECT");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== INJECT (end)\n");

	return rc;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_ep_tx_inject(struct fid_ep *ep,
			    const void *buf,
			    size_t len,
			    fi_addr_t dest_addr,
			    uint64_t tag,
			    const uint32_t data,
			    const int lock_required,
			    const enum fi_av_type av_type,
			    uint64_t tx_op_flags,
			    const uint64_t caps,
			    const enum ofi_reliability_kind reliability,
			    const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);

	ssize_t rc = fi_opx_ep_tx_inject_internal(ep, buf, len, dest_addr, tag,
						  data, FI_OPX_LOCK_NOT_REQUIRED, av_type,
						  tx_op_flags, caps, reliability, hfi1_type);

	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_recv_generic(struct fid_ep *ep,
			    void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context,
			    const int lock_required, const enum fi_av_type av_type,
			    const uint64_t static_flags,
			    const enum ofi_reliability_kind reliability,
			    const enum opx_hfi1_type hfi1_type)
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_ep_rx_recv_internal(opx_ep, buf, len, desc, src_addr, tag,
						ignore, context, FI_OPX_LOCK_NOT_REQUIRED, av_type,
						static_flags, reliability, hfi1_type);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}

__OPX_FORCE_INLINE__
ssize_t fi_opx_recvmsg_generic(struct fid_ep *ep,
			       const struct fi_msg *msg, uint64_t flags,
			       const int lock_required, const enum fi_av_type av_type,
			       const enum ofi_reliability_kind reliability,
			       const enum opx_hfi1_type hfi1_type )
{
	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);

	fi_opx_lock_if_required(&opx_ep->lock, lock_required);
	ssize_t rc = fi_opx_ep_rx_recvmsg_internal(opx_ep, msg, flags, FI_OPX_LOCK_NOT_REQUIRED, av_type,
					reliability, hfi1_type);
	fi_opx_unlock_if_required(&opx_ep->lock, lock_required);

	return rc;
}


#endif /* _FI_PROV_OPX_ENDPOINT_H_ */
