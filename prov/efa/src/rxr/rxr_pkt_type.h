/*
 * Copyright (c) 2019-2020 Amazon.com, Inc. or its affiliates.
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

#ifndef _RXR_PKT_TYPE_H
#define _RXR_PKT_TYPE_H

/* This header file contain the ID of all RxR packet types, and
 * the necessary data structures and functions for each packet type
 *
 * RxR packet types can be classified into 3 categories:
 *     data packet, control packet and context packet
 *
 * For each packet type, the following items are needed:
 *
 *   First, each packet type need to define a struct for its header,
 *       and the header must be start with ```struct rxr_base_hdr```.
 *
 *   Second, each control packet type need to define an init()
 *       function and a handle_sent() function. These functions
 *       are called by rxr_pkt_post_ctrl_or_queue().
 *
 *   Finally, each packet type (except context packet) need to
 *     define a handle_recv() functions which is called by
 *     rxr_pkt_handle_recv_completion().
 */

/* ID of each packet type. Changing ID would break inter
 * operability thus is strictly prohibited.
 */

#define RXR_RETIRED_RTS_PKT	1
#define RXR_RETIRED_CONNACK_PKT	2
#define RXR_CTS_PKT		3
#define RXR_DATA_PKT		4
#define RXR_READRSP_PKT		5
#define RXR_RMA_CONTEXT_PKT	6
#define RXR_EOR_PKT		7
#define RXR_ATOMRSP_PKT         8
#define RXR_HANDSHAKE_PKT	9
#define RXR_RECEIPT_PKT 10

#define RXR_REQ_PKT_BEGIN		64
#define RXR_BASELINE_REQ_PKT_BEGIN	64
#define RXR_EAGER_MSGRTM_PKT		64
#define RXR_EAGER_TAGRTM_PKT		65
#define RXR_MEDIUM_MSGRTM_PKT		66
#define RXR_MEDIUM_TAGRTM_PKT		67
#define RXR_LONGCTS_MSGRTM_PKT		68
#define RXR_LONGCTS_TAGRTM_PKT		69
#define RXR_EAGER_RTW_PKT		70
#define RXR_LONGCTS_RTW_PKT		71
#define RXR_SHORT_RTR_PKT		72
#define RXR_LONGCTS_RTR_PKT		73
#define RXR_WRITE_RTA_PKT		74
#define RXR_FETCH_RTA_PKT		75
#define RXR_COMPARE_RTA_PKT		76
#define RXR_BASELINE_REQ_PKT_END	77

#define RXR_EXTRA_REQ_PKT_BEGIN		128
#define RXR_LONGREAD_MSGRTM_PKT		128
#define RXR_LONGREAD_TAGRTM_PKT		129
#define RXR_LONGREAD_RTW_PKT		130
#define RXR_READ_RTR_PKT		131

#define RXR_DC_REQ_PKT_BEGIN		132
#define RXR_DC_EAGER_MSGRTM_PKT 	133
#define RXR_DC_EAGER_TAGRTM_PKT 	134
#define RXR_DC_MEDIUM_MSGRTM_PKT 	135
#define RXR_DC_MEDIUM_TAGRTM_PKT 	136
#define RXR_DC_LONGCTS_MSGRTM_PKT  	137
#define RXR_DC_LONGCTS_TAGRTM_PKT  	138
#define RXR_DC_EAGER_RTW_PKT    	139
#define RXR_DC_LONGCTS_RTW_PKT     	140
#define RXR_DC_WRITE_RTA_PKT    	141
#define RXR_DC_REQ_PKT_END		142
#define RXR_EXTRA_REQ_PKT_END   	142

/*
 *  Packet fields common to all rxr packets. The other packet headers below must
 *  be changed if this is updated.
 */
struct rxr_base_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_base_hdr) == 4, "rxr_base_hdr check");
#endif

static inline struct rxr_base_hdr *rxr_get_base_hdr(void *pkt)
{
	return (struct rxr_base_hdr *)pkt;
}

struct rxr_ep;
struct rdm_peer;
struct rxr_tx_entry;
struct rxr_rx_entry;
struct rxr_read_entry;

/*
 *  HANDSHAKE packet header and functions
 *  implementation of the functions are in rxr_pkt_type_misc.c
 */
struct rxr_handshake_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	/* nex_p3 is number of members in exinfo plus 3.
	 * The "p3" part was introduced for backward compatibility.
	 * See protocol v4 document section 2.1 for detail.
	 */
	uint32_t nex_p3;
	uint64_t exinfo[0];
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_handshake_hdr) == 8, "rxr_handshake_hdr check");
#endif

static inline
struct rxr_handshake_hdr *rxr_get_handshake_hdr(void *pkt)
{
	return (struct rxr_handshake_hdr *)pkt;
}

ssize_t rxr_pkt_init_handshake(struct rxr_ep *ep,
			       struct rxr_pkt_entry *pkt_entry,
			       fi_addr_t addr);

ssize_t rxr_pkt_post_handshake(struct rxr_ep *ep, struct rdm_peer *peer);

void rxr_pkt_post_handshake_or_queue(struct rxr_ep *ep,
				     struct rdm_peer *peer);

void rxr_pkt_handle_handshake_recv(struct rxr_ep *ep,
				   struct rxr_pkt_entry *pkt_entry);
/*
 * @breif format of CTS packet header
 *
 * CTS is used in long-CTS sub-protocols for flow control.
 *
 * It is sent from receiver to sender, and contains number of bytes
 * receiver is ready to receive.
 *
 * long-CTS is used not only by two-sided communication but also
 * by emulated write and emulated read protocols.
 *
 * In emulated write, requester is sender, and responder is receiver.
 *
 * In emulated read, requester is receiver, and responder is sender.
 */
struct rxr_cts_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint8_t pad[4];
	uint32_t send_id; /* ID of the send opertaion on sender side */
	uint32_t recv_id; /* ID of the receive operatin on receive side */
	uint64_t recv_length; /* number of bytes receiver is ready to receive */
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_cts_hdr) == 24, "rxr_cts_hdr check");
#endif

/* this flag is to indicated the CTS is the response of a RTR packet */
#define RXR_CTS_READ_REQ		BIT_ULL(7)
#define RXR_CTS_HDR_SIZE		(sizeof(struct rxr_cts_hdr))

static inline
struct rxr_cts_hdr *rxr_get_cts_hdr(void *pkt)
{
	return (struct rxr_cts_hdr *)pkt;
}

void rxr_pkt_calc_cts_window_credits(struct rxr_ep *ep, struct rdm_peer *peer,
				     uint64_t size, int request,
				     int *window, int *credits);

ssize_t rxr_pkt_init_cts(struct rxr_ep *ep,
			 struct rxr_rx_entry *rx_entry,
			 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_cts_sent(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_cts_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry);

/*
 * @brief format of DATA packet header.
 *
 * DATA is used in long-CTS sub-protocols.
 *
 * It is sent from sender to receiver, and contains a segment
 * of application data.
 *
 * long-CTS is used not only by two-sided communication but also
 * by emulated write and emulated read protocols.
 *
 * In emulated write, requester is sender, and responder is receiver.
 *
 * In emulated read, requester is receiver, and responder is sender.
 */
struct rxr_data_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t recv_id; /* ID of the receive operation on receiver */
	uint64_t seg_length;
	uint64_t seg_offset;
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_data_hdr) == 24, "rxr_data_hdr check");
#endif

#define RXR_DATA_HDR_SIZE		(sizeof(struct rxr_data_hdr))

struct rxr_data_pkt {
	struct rxr_data_hdr hdr;
	char data[];
};

static inline
struct rxr_data_pkt *rxr_get_data_pkt(void *pkt)
{
	return (struct rxr_data_pkt *)pkt;
}

ssize_t rxr_pkt_send_data(struct rxr_ep *ep,
			  struct rxr_tx_entry *tx_entry,
			  struct rxr_pkt_entry *pkt_entry);

ssize_t rxr_pkt_send_data_desc(struct rxr_ep *ep,
			       struct rxr_tx_entry *tx_entry,
			       struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_proc_data(struct rxr_ep *ep,
		       struct rxr_rx_entry *rx_entry,
		       struct rxr_pkt_entry *pkt_entry,
		       char *data, size_t seg_offset,
		       size_t seg_size);

void rxr_pkt_handle_data_send_completion(struct rxr_ep *ep,
					 struct rxr_pkt_entry *pkt_entry);


void rxr_pkt_handle_data_recv(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry);

/*
 *  @brief READRSP packet header
 *
 *  READRSP is sent from read responder to read requester, and it contains
 *  application data.
 */
struct rxr_readrsp_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t padding;
	uint32_t recv_id; /* ID of the receive operation on the read requester, from rtr packet */
	uint32_t send_id; /* ID of the send operation on the read responder, will be included in CTS packet */
	uint64_t seg_length;
};

static inline struct rxr_readrsp_hdr *rxr_get_readrsp_hdr(void *pkt)
{
	return (struct rxr_readrsp_hdr *)pkt;
}

#define RXR_READRSP_HDR_SIZE	(sizeof(struct rxr_readrsp_hdr))

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_readrsp_hdr) == sizeof(struct rxr_data_hdr), "rxr_readrsp_hdr check");
#endif

struct rxr_readrsp_pkt {
	struct rxr_readrsp_hdr hdr;
	char data[];
};

int rxr_pkt_init_readrsp(struct rxr_ep *ep,
			 struct rxr_tx_entry *tx_entry,
			 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_readrsp_sent(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_readrsp_send_completion(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_readrsp_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry);

/*
 *  RMA context packet, used to differentiate the normal RMA read, normal RMA
 *  write, and the RMA read in two-sided large message transfer
 *  Implementation of the function is in rxr_pkt_type_misc.c
 */
struct rxr_rma_context_pkt {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t context_type;
	uint32_t tx_id; /* used by write context */
	uint32_t read_id; /* used by read context */
	size_t seg_size; /* used by read context */
};

enum rxr_rma_context_pkt_type {
	RXR_READ_CONTEXT = 1,
	RXR_WRITE_CONTEXT,
};

void rxr_pkt_init_write_context(struct rxr_tx_entry *tx_entry,
				struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_init_read_context(struct rxr_ep *rxr_ep,
			       struct rxr_read_entry *read_entry,
			       size_t seg_size,
			       struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_rma_completion(struct rxr_ep *ep,
				   struct rxr_pkt_entry *pkt_entry);

/*
 * @brief format of the EOR packet.
 *
 * EOR packet is used in long-read sub-protocols.
 *
 * It is sent from receiver to sender, to notify
 * the finish of data transfer.
 *
 * long-read is used not only by two-sided communication but also
 * by emulated write.
 *
 * In emulated write, requester is sender, and responder is receiver.
 */
struct rxr_eor_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t send_id; /* ID of the send operation on sender */
	uint32_t recv_id; /* ID of the receive operation on receiver */
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_eor_hdr) == 12, "rxr_eor_hdr check");
#endif

static inline
struct rxr_eor_hdr *rxr_get_eor_hdr(void *pkt)
{
	return (struct rxr_eor_hdr *)pkt;
}

int rxr_pkt_init_eor(struct rxr_ep *ep,
		     struct rxr_rx_entry *rx_entry,
		     struct rxr_pkt_entry *pkt_entry);


void rxr_pkt_handle_eor_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_eor_send_completion(struct rxr_ep *ep,
					struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_eor_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry);

/* atomrsp types */
struct rxr_atomrsp_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint8_t pad[4];
	uint32_t rx_id;
	uint32_t tx_id;
	uint64_t seg_size;
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_atomrsp_hdr) == 24, "rxr_atomrsp_hdr check");
#endif

#define RXR_ATOMRSP_HDR_SIZE	(sizeof(struct rxr_atomrsp_hdr))

struct rxr_atomrsp_pkt {
	struct rxr_atomrsp_hdr hdr;
	char data[];
};

static inline struct rxr_atomrsp_hdr *rxr_get_atomrsp_hdr(void *pkt)
{
	return (struct rxr_atomrsp_hdr *)pkt;
}

/* receipt packet headers */
struct rxr_receipt_hdr {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t tx_id;
	uint32_t msg_id;
	int32_t padding;
};

static inline
struct rxr_receipt_hdr *rxr_get_receipt_hdr(void *pkt)
{
	return (struct rxr_receipt_hdr *)pkt;
}

/* receipt packet functions: init, handle_sent, handle_send_completion, recv*/
int rxr_pkt_init_receipt(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
			 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_receipt_sent(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_receipt_send_completion(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_receipt_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry);

/* atomrsp functions: init, handle_sent, handle_send_completion, recv */
int rxr_pkt_init_atomrsp(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
			 struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_atomrsp_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_atomrsp_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void rxr_pkt_handle_atomrsp_recv(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

#endif

#include "rxr_pkt_type_req.h"
