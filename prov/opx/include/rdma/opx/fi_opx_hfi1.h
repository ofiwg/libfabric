/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_HFI1_H_
#define _FI_PROV_OPX_HFI1_H_

#include "rdma/opx/fi_opx_hfi1_packet.h"
#include "rdma/opx/fi_opx_compiler.h"

#include <stdlib.h>
#include <stdio.h>

#include <arpa/inet.h>

#include "rdma/fi_errno.h"	// only for FI_* errno return codes
#include "rdma/fabric.h" // only for 'fi_addr_t' ... which is a typedef to uint64_t
#include <uuid/uuid.h>

// #define FI_OPX_TRACE 1


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
 * for the Low/High case. If we define "Low" as < 4k and "High" as > "2^24-4k"
 * that should be adequate.
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
#define MAX_PSN			0x0000000000FFFFFFull
#define PSN_WINDOW_SIZE 0x0000000001000000ull
#define MAX_PSN_MASK	0xFFFFFFFFFF000000ull
#define PSN_LOW_WINDOW 	0x0000000000001000ull
#define PSN_HIGH_WINDOW 0x0000000000FFEFFFull
#define PSN_AGE_LIMIT	0x0000000000F00000ull

#define FI_OPX_HFI1_RHF_EGRBFR_INDEX_SHIFT	(16)		/* a.k.a. "HFI_RHF_EGRBFR_INDEX_SHIFT" */
#define FI_OPX_HFI1_RHF_EGRBFR_INDEX_MASK	(0x7FF)		/* a.k.a. "HFI_RHF_EGRBFR_INDEX_MASK" */
#define FI_OPX_HFI1_PBC_VL_MASK		(0xf)		/* a.k.a. "HFI_PBC_VL_MASK" */
#define FI_OPX_HFI1_PBC_VL_SHIFT		(12)		/* a.k.a. "HFI_PBC_VL_SHIFT" */
#define FI_OPX_HFI1_PBC_CR_MASK		(0x1)		/* Force Credit return */
#define FI_OPX_HFI1_PBC_CR_SHIFT		(25)    /* Force Credit return */
#define FI_OPX_HFI1_PBC_SC4_SHIFT		(4)		/* a.k.a. "HFI_PBC_SC4_SHIFT" */
#define FI_OPX_HFI1_PBC_SC4_MASK		(0x1)		/* a.k.a. "HFI_PBC_SC4_MASK" */
#define FI_OPX_HFI1_PBC_DCINFO_SHIFT		(30)		/* a.k.a. "HFI_PBC_DCINFO_SHIFT" */
#define FI_OPX_HFI1_LRH_BTH			(0x0002)	/* a.k.a. "HFI_LRH_BTH" */
#define FI_OPX_HFI1_LRH_SL_MASK		(0xf)		/* a.k.a. "HFI_LRH_SL_MASK" */
#define FI_OPX_HFI1_LRH_SL_SHIFT		(4)		/* a.k.a. "HFI_LRH_SL_SHIFT" */
#define FI_OPX_HFI1_LRH_SC_MASK		(0xf)		/* a.k.a. "HFI_LRH_SC_MASK" */
#define FI_OPX_HFI1_LRH_SC_SHIFT		(12)		/* a.k.a. "HFI_LRH_SC_SHIFT" */
#define FI_OPX_HFI1_DEFAULT_P_KEY		(0x8001)	/* a.k.a. "HFI_DEFAULT_P_KEY" */

#define FI_OPX_HFI1_TX_SEND_RZV_CREDIT_MAX_WAIT	0x7fffffff
#define FI_OPX_HFI1_TX_DO_REPLAY_CREDIT_MAX_WAIT	0x0000ffff
#define FI_OPX_HFI1_TX_MIN_RZV_PAYLOAD_BYTES (64) /* The Minimum size of a data payload Rendezvous can send an RTS for. */
                                                  /* Normally, the payload should be larger than 8K */

#define FI_OPX_HFI1_TX_RELIABILITY_RESERVED_CREDITS (1)  //Todo not actually reserving a credit
#define FI_OPX_HFI1_TX_CREDIT_DELTA_THRESHOLD 		(63)  // If the incomming request asks for more credits than this, force a return.  Lower number here is more agressive 
#define FI_OPX_HFI1_TX_CREDIT_MIN_FORCE_CR			(130) // We won't force a credit return for FI_OPX_HFI1_TX_CREDIT_DELTA_THRESHOLD if the number 
                                                          // of avalible credits is above this number

static inline
uint32_t fi_opx_addr_calculate_base_rx (const uint32_t process_id, const uint32_t processes_per_node) {

abort();
	return 0;
}

struct fi_opx_hfi1_txe_scb {

	union {
		uint64_t		qw0;	/* a.k.a. 'struct hfi_pbc' */
		//struct hfi_pbc		pbc;
	};
	union fi_opx_hfi1_packet_hdr	hdr;

} __attribute__((__aligned__(8)));


struct fi_opx_hfi1_rxe_hdr {

	union fi_opx_hfi1_packet_hdr	hdr;
	uint64_t			rhf;

} __attribute__((__aligned__(64)));




#define HFI_TXE_CREDITS_COUNTER(credits)	((credits.raw16b[0] >> 0) & 0x07FFu)
#define HFI_TXE_CREDITS_STATUS(credits)		((credits.raw16b[0] >> 11) & 0x01u)
#define HFI_TXE_CREDITS_DUETOPBC(credits)	((credits.raw16b[0] >> 12) & 0x01u)
#define HFI_TXE_CREDITS_DUETOTHRESHOLD(credits)	((credits.raw16b[0] >> 13) & 0x01u)
#define HFI_TXE_CREDITS_DUETOERR(credits)	((credits.raw16b[0] >> 14) & 0x01u)
#define HFI_TXE_CREDITS_DUETOFORCE(credits)	((credits.raw16b[0] >> 15) & 0x01u)
union fi_opx_hfi1_txe_credits {

	uint16_t		raw16b[4];
	uint64_t		raw64b;

	struct {
		uint16_t	Counter				: 11;	/* use macros to access */
		uint16_t	Status				:  1;
		uint16_t	CreditReturnDueToPbc		:  1;
		uint16_t	CreditReturnDueToThreshold	:  1;
		uint16_t	CreditReturnDueToErr		:  1;
		uint16_t	CreditReturnDueToForce		:  1;

		uint16_t	pad[3];
	} __attribute__((packed));
};

#define FI_OPX_HFI1_DUMP_TXE_CREDITS(credits)	\
	fi_opx_hfi1_dump_txe_credits(credits, __FILE__, __func__, __LINE__);

static inline void fi_opx_hfi1_dump_txe_credits (union fi_opx_hfi1_txe_credits * credits,
		const char * file, const char * func, unsigned line)
{
	fprintf(stderr, "%s:%s():%d === dump hfi1 txe credits ===\n", file, func, line);
	fprintf(stderr, "%s:%s():%d .raw64b ...................... 0x%016lx\n", file, func, line, credits->raw64b);
	fprintf(stderr, "%s:%s():%d .Counter ..................... %hu\n", file, func, line, credits->Counter);
	fprintf(stderr, "%s:%s():%d .Status ...................... %hu\n", file, func, line, credits->Status);
	fprintf(stderr, "%s:%s():%d .CreditReturnDueToPbc ........ %hu\n", file, func, line, credits->CreditReturnDueToPbc);
	fprintf(stderr, "%s:%s():%d .CreditReturnDueToThreshold .. %hu\n", file, func, line, credits->CreditReturnDueToThreshold);
	fprintf(stderr, "%s:%s():%d .CreditReturnDueToErr ........ %hu\n", file, func, line, credits->CreditReturnDueToErr);
	fprintf(stderr, "%s:%s():%d .CreditReturnDueToForce ...... %hu\n", file, func, line, credits->CreditReturnDueToForce);
}





/* This 'state' information will update on each txe pio operation */
union fi_opx_hfi1_pio_state {

	uint64_t			qw0;

	struct {
		uint16_t		fill_counter;
		uint16_t		free_counter_shadow;
		uint16_t		scb_head_index;
		uint16_t		credits_total;	/* yeah, yeah .. THIS field is static, but there was an unused halfword at this spot, so .... */
	};
};

/* This 'static' information will not change after it is set by the driver
 * and can be safely copied into other structures to improve cache layout */
struct fi_opx_hfi1_pio_static {
	volatile uint64_t *		scb_sop_first;
	volatile uint64_t *		scb_first;

	/* pio credit return address. The HFI TXE periodically updates this
	 * host memory location with the current credit state. To avoid cache
	 * thrashing software should read from this location sparingly. */
	union {
		volatile uint64_t *				credits_addr;
		volatile union fi_opx_hfi1_txe_credits *	credits;
	};
};

/* This 'state' information will update on each txe sdma operation */
union fi_opx_hfi1_sdma_state {

	uint64_t			qw0;

//	struct {
//		uint16_t		pio_fill_counter;
//		uint16_t		pio_free_counter_shadow;
//		uint16_t		pio_scb_head_index;
//		uint16_t		unused;
//	};
};

/* This 'static' information will not change after it is set by the driver
 * and can be safely copied into other structures to improve cache layout */
struct fi_opx_hfi1_sdma_static {
	uint16_t			available_counter;
	uint16_t			fill_index;
	uint16_t			done_index;
	uint16_t			queue_size;
	struct hfi1_sdma_comp_entry *	completion_queue;
};


struct fi_opx_hfi1_rxe_state {

	struct {
		uint64_t		head;
		uint32_t		rhf_seq;
	} __attribute__((__packed__)) hdrq;

	struct {
		uint32_t		countdown;
	} __attribute__((__packed__)) egrq;

} __attribute__((__packed__));

struct fi_opx_hfi1_rxe_static {

	struct {
		uint32_t *		base_addr;
		uint32_t		rhf_off;
		int32_t			rhf_notail;


		uint32_t		elemsz;
		uint32_t		elemlast;
		uint32_t		elemcnt;
		uint64_t		rx_poll_mask;

		uint32_t *		rhf_base;


		volatile uint64_t *	head_register;
		volatile uint64_t *	tail_register;

	} hdrq;


	struct {
		uint32_t *		base_addr;
		uint32_t		elemsz;
		uint32_t        size;

		volatile uint64_t *	head_register;
		volatile uint64_t *	tail_register;

	} egrq;

	volatile uint64_t *		uregbase;
	uint8_t				id;		/* hfi receive context id [0..159] */
};




struct fi_opx_hfi1_context {

	struct {
		union fi_opx_hfi1_pio_state		pio;
		union fi_opx_hfi1_sdma_state		sdma;
		struct fi_opx_hfi1_rxe_state		rxe;
	} state;

	struct {
		struct fi_opx_hfi1_pio_static		pio;
		struct fi_opx_hfi1_sdma_static	sdma;
		struct fi_opx_hfi1_rxe_static		rxe;

	} info;

	int				fd;
	uint16_t			lid;
	//struct _hfi_ctrl *		ctrl;
	//struct hfi1_user_info_dep	user_info;
	uint32_t			hfi_unit;
	uint32_t			hfi_port;
	uint64_t			gid_hi;
	uint64_t			gid_lo;
	uint16_t			mtu;
	uint8_t				bthqp;
	uint16_t			jkey;
	uint16_t			send_ctxt;

	uint16_t			sl2sc[32];
	uint16_t			sc2vl[32];
	uint64_t			sl;
	uint64_t			sc;
	uint64_t			vl;

	uint64_t			runtime_flags;

};



#ifdef NDEBUG
#define FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(credits_addr)
#else
#define FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(credits_addr)	\
	fi_opx_hfi1_check_credits_for_error(credits_addr, __FILE__, __func__, __LINE__);
#endif

static inline void fi_opx_hfi1_check_credits_for_error (volatile uint64_t * credits_addr, const char * file, const char * func, unsigned line)
{
	const uint64_t credit_return = *credits_addr;
	if ((credit_return & 0x0000000000004800ul) != 0) {
		fprintf(stderr, "%s:%s():%d ########### PIO SEND ERROR!\n", file, func, line);
		fi_opx_hfi1_dump_txe_credits((union fi_opx_hfi1_txe_credits *)credits_addr, file, func, line);
		abort();
	}

	return;
}

__OPX_FORCE_INLINE__
uint16_t fi_opx_credits_in_use(union fi_opx_hfi1_pio_state *pio_state) {
	assert((pio_state->fill_counter - pio_state->free_counter_shadow) <= pio_state->credits_total);
	return ((pio_state->fill_counter - pio_state->free_counter_shadow) & 0x07FFu);
}

__OPX_FORCE_INLINE_AND_FLATTEN__
uint16_t fi_opx_credits_avail(union fi_opx_hfi1_pio_state *pio_state, uint8_t *force_credit_return, uint16_t credits_needed) {
	const uint16_t return_value =  pio_state->credits_total - fi_opx_credits_in_use(pio_state);
	if ((return_value < FI_OPX_HFI1_TX_CREDIT_MIN_FORCE_CR) && (credits_needed > FI_OPX_HFI1_TX_CREDIT_DELTA_THRESHOLD)) {
		*force_credit_return = 1;
	}
	return return_value;
}

__OPX_FORCE_INLINE_AND_FLATTEN__
uint16_t fi_opx_reliability_credits_avail(union fi_opx_hfi1_pio_state *pio_state) {
	return pio_state->credits_total - fi_opx_credits_in_use(pio_state);
}

__OPX_FORCE_INLINE__
volatile uint64_t * fi_opx_pio_scb_base(volatile uint64_t *pio_scb_base, union fi_opx_hfi1_pio_state *pio_state) {
	return ((pio_scb_base) + (pio_state->scb_head_index << 3));
}

__OPX_FORCE_INLINE__
void fi_opx_update_credits(union fi_opx_hfi1_pio_state *pio_state, volatile uint64_t *pio_credits_addr) {
	volatile uint64_t * credits_addr = (uint64_t *)(pio_credits_addr);
	const uint64_t credit_return = *credits_addr;
	pio_state->free_counter_shadow = (uint16_t)(credit_return & 0x00000000000007FFul);
}

__OPX_FORCE_INLINE__
void fi_opx_consume_credits(union fi_opx_hfi1_pio_state *pio_state, size_t count) {
	assert((pio_state->scb_head_index + count) <= pio_state->credits_total);
	pio_state->scb_head_index = (pio_state->scb_head_index + count) * (pio_state->credits_total != (pio_state->scb_head_index + count));
	pio_state->fill_counter = (pio_state->fill_counter + count) & 0x00000000000007FFul;
}

#define FI_OPX_HFI1_CREDITS_IN_USE(pio_state) fi_opx_credits_in_use(&pio_state)
#define FI_OPX_HFI1_UPDATE_CREDITS(pio_state, pio_credits_addr)	fi_opx_update_credits(&pio_state, pio_credits_addr);
#define FI_OPX_HFI1_PIO_SCB_HEAD(pio_scb_base, pio_state) fi_opx_pio_scb_base(pio_scb_base, &pio_state)
#define FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, force_credit_return, credits_needed) fi_opx_credits_avail(&pio_state, force_credit_return, credits_needed)
#define FI_OPX_HFI1_AVAILABLE_RELIABILITY_CREDITS(pio_state) fi_opx_reliability_credits_avail(&pio_state)
#define FI_OPX_HFI1_CONSUME_CREDITS(pio_state, count) fi_opx_consume_credits(&pio_state, count)
#define FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state) FI_OPX_HFI1_CONSUME_CREDITS(pio_state, 1);


struct fi_opx_hfi1_context * fi_opx_hfi1_context_open (struct fid_ep *ep, uuid_t unique_job_key);

int init_hfi1_rxe_state (struct fi_opx_hfi1_context * context,
		struct fi_opx_hfi1_rxe_state * rxe_state);

/*
 * Shared memory transport
 */
#define FI_OPX_SHM_FIFO_SIZE		(1024)
#define FI_OPX_SHM_BUFFER_MASK		(FI_OPX_SHM_FIFO_SIZE-1)
#define FI_OPX_SHM_PACKET_SIZE	(FI_OPX_HFI1_PACKET_MTU + sizeof(struct fi_opx_hfi1_stl_packet_hdr))

#endif /* _FI_PROV_OPX_HFI1_H_ */
