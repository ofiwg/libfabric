/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
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

#include "efa_rdm_ope.h"
#include "efa_rdm_protocol.h"

static inline struct rxr_base_hdr *rxr_get_base_hdr(void *pkt)
{
	return (struct rxr_base_hdr *)pkt;
}

struct efa_rdm_ep;
struct efa_rdm_peer;
struct rxr_read_entry;

/* HANDSHAKE packet related functions */
static inline
struct rxr_handshake_hdr *rxr_get_handshake_hdr(void *pkt)
{
	return (struct rxr_handshake_hdr *)pkt;
}

static inline
struct rxr_handshake_opt_connid_hdr *rxr_get_handshake_opt_connid_hdr(void *pkt)
{
	struct rxr_handshake_hdr *handshake_hdr;
	size_t base_hdr_size;

	handshake_hdr = (struct rxr_handshake_hdr *)pkt;
	assert(handshake_hdr->type == RXR_HANDSHAKE_PKT);
	assert(handshake_hdr->flags & RXR_PKT_CONNID_HDR);
	base_hdr_size = sizeof(struct rxr_handshake_hdr) +
			(handshake_hdr->nextra_p3 - 3) * sizeof(uint64_t);
	return (struct rxr_handshake_opt_connid_hdr *)((char *)pkt + base_hdr_size);
}

static inline
struct rxr_handshake_opt_host_id_hdr *rxr_get_handshake_opt_host_id_hdr(void *pkt)
{
	struct rxr_handshake_hdr *handshake_hdr;
	size_t offset = 0;

	handshake_hdr = (struct rxr_handshake_hdr *)pkt;
	assert(handshake_hdr->type == RXR_HANDSHAKE_PKT);

	offset += sizeof(struct rxr_handshake_hdr) +
					(handshake_hdr->nextra_p3 - 3) * sizeof(uint64_t);

	assert(handshake_hdr->flags & RXR_HANDSHAKE_HOST_ID_HDR);

	if (handshake_hdr->flags & RXR_PKT_CONNID_HDR) {
		/* HOST_ID_HDR is always immediately after CONNID_HDR(if present) */
		offset += sizeof(struct rxr_handshake_opt_connid_hdr);
	}

	return (struct rxr_handshake_opt_host_id_hdr *)((char *)pkt + offset);
}

ssize_t rxr_pkt_init_handshake(struct efa_rdm_ep *ep,
			       struct efa_rdm_pke *pkt_entry,
			       fi_addr_t addr);

ssize_t rxr_pkt_post_handshake(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer);

void rxr_pkt_post_handshake_or_queue(struct efa_rdm_ep *ep,
				     struct efa_rdm_peer *peer);

void rxr_pkt_handle_handshake_recv(struct efa_rdm_ep *ep,
				   struct efa_rdm_pke *pkt_entry);

/* CTS packet related functions */
static inline
struct rxr_cts_hdr *rxr_get_cts_hdr(void *pkt)
{
	return (struct rxr_cts_hdr *)pkt;
}

void rxr_pkt_calc_cts_window_credits(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
				     uint64_t size, int request,
				     int *window, int *credits);

ssize_t rxr_pkt_init_cts(struct efa_rdm_pke *pkt_entry,
			 struct efa_rdm_ope *ope);

void rxr_pkt_handle_cts_sent(struct efa_rdm_ep *ep,
			     struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_cts_recv(struct efa_rdm_ep *ep,
			     struct efa_rdm_pke *pkt_entry);

static inline
struct rxr_data_hdr *rxr_get_data_hdr(void *pkt)
{
	return (struct rxr_data_hdr *)pkt;
}

int rxr_pkt_init_data(struct efa_rdm_pke *pkt_entry,
		      struct efa_rdm_ope *ope,
		      size_t data_offset,
		      int data_size);

void rxr_pkt_handle_data_sent(struct efa_rdm_ep *ep,
			      struct efa_rdm_pke *pkt_entry);

void rxr_pkt_proc_data(struct efa_rdm_ep *ep,
		       struct efa_rdm_ope *ope,
		       struct efa_rdm_pke *pkt_entry,
		       char *data, size_t seg_offset,
		       size_t seg_size);

void rxr_pkt_handle_data_send_completion(struct efa_rdm_ep *ep,
					 struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_data_recv(struct efa_rdm_ep *ep,
			      struct efa_rdm_pke *pkt_entry);

/* READRSP packet related functions */
static inline struct rxr_readrsp_hdr *rxr_get_readrsp_hdr(void *pkt)
{
	return (struct rxr_readrsp_hdr *)pkt;
}

int rxr_pkt_init_readrsp(struct efa_rdm_pke *pkt_entry,
			 struct efa_rdm_ope *txe);

void rxr_pkt_handle_readrsp_sent(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_readrsp_send_completion(struct efa_rdm_ep *ep,
					    struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_readrsp_recv(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke *pkt_entry);

/*
 *  RMA context packet, used to differentiate the normal RMA read, normal RMA
 *  write, and the RMA read in two-sided large message transfer
 *  Implementation of the function is in rxr_pkt_type_misc.c
 */
struct efa_rdm_rma_context_pkt {
	uint8_t type;
	uint8_t version;
	uint16_t flags;
	/* end of rxr_base_hdr */
	uint32_t context_type;
	uint32_t tx_id; /* used by write context */
	uint32_t read_id; /* used by read context */
	size_t seg_size; /* used by read context */
};

enum efa_rdm_rma_context_pkt_type {
	RXR_READ_CONTEXT = 1,
	RXR_WRITE_CONTEXT,
};

void rxr_pkt_init_write_context(struct efa_rdm_ope *txe,
				struct efa_rdm_pke *pkt_entry);

void rxr_pkt_init_read_context(struct efa_rdm_ep *efa_rdm_ep,
			       void *x_entry,
			       fi_addr_t addr,
			       int read_id,
			       size_t seg_size,
			       struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_rma_completion(struct efa_rdm_ep *ep,
				   struct efa_rdm_pke *pkt_entry);

/* EOR packet related functions */
static inline
struct rxr_eor_hdr *rxr_get_eor_hdr(void *pkt)
{
	return (struct rxr_eor_hdr *)pkt;
}

int rxr_pkt_init_eor(struct efa_rdm_pke *pkt_entry,
		     struct efa_rdm_ope *rxe);

static inline
void rxr_pkt_handle_eor_sent(struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry)
{
}

void rxr_pkt_handle_eor_send_completion(struct efa_rdm_ep *ep,
					struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_eor_recv(struct efa_rdm_ep *ep,
			     struct efa_rdm_pke *pkt_entry);

/* ATOMRSP packet related functions */
static inline struct rxr_atomrsp_hdr *rxr_get_atomrsp_hdr(void *pkt)
{
	return (struct rxr_atomrsp_hdr *)pkt;
}

int rxr_pkt_init_atomrsp(struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *rxe);

void rxr_pkt_handle_atomrsp_sent(struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_atomrsp_send_completion(struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_atomrsp_recv(struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry);

/* RECEIPT packet related functions */
static inline
struct rxr_receipt_hdr *rxr_get_receipt_hdr(void *pkt)
{
	return (struct rxr_receipt_hdr *)pkt;
}

int rxr_pkt_init_receipt(struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *rxe);

void rxr_pkt_handle_receipt_sent(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_receipt_send_completion(struct efa_rdm_ep *ep,
					    struct efa_rdm_pke *pkt_entry);

void rxr_pkt_handle_receipt_recv(struct efa_rdm_ep *ep,
				 struct efa_rdm_pke *pkt_entry);

int rxr_pkt_type_readbase_rtm(struct efa_rdm_peer *peer, int op, uint64_t fi_flags, struct efa_hmem_info *hmem_info);

#endif

#include "rxr_pkt_type_req.h"
