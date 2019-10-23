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

#ifndef _RXR_RMA_H_
#define _RXR_RMA_H_

#include <rdma/fi_rma.h>

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

struct rxr_readrsp_pkt {
	struct rxr_readrsp_hdr hdr;
	char data[];
};

static inline struct rxr_readrsp_hdr *rxr_get_readrsp_hdr(void *pkt)
{
	return (struct rxr_readrsp_hdr *)pkt;
}

#define RXR_READRSP_HDR_SIZE	(sizeof(struct rxr_readrsp_hdr))

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_readrsp_hdr) == sizeof(struct rxr_data_hdr), "rxr_readrsp_hdr check");
#endif

struct rxr_rma_read_info {
	uint64_t rma_initiator_rx_id;
	uint64_t window;
};

#if defined(static_assert) && defined(__x86_64__)
static_assert(sizeof(struct rxr_rma_read_info) == 16, "rxr_rma_read_hdr check");
#endif

char *rxr_rma_init_rts_hdr(struct rxr_ep *ep,
			   struct rxr_tx_entry *tx_entry,
			   struct rxr_pkt_entry *pkt_entry,
			   char *hdr);

int rxr_rma_verified_copy_iov(struct rxr_ep *ep, struct fi_rma_iov *rma,
			      size_t count, uint32_t flags, struct iovec *iov);

char *rxr_rma_read_rts_hdr(struct rxr_ep *ep,
			   struct rxr_rx_entry *rx_entry,
			   struct rxr_pkt_entry *pkt_entry,
			   char *hdr);

int rxr_rma_proc_write_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

int rxr_rma_init_read_rts(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
			  struct rxr_pkt_entry *pkt_entry);

int rxr_rma_proc_read_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

/* read response related functions */
struct rxr_tx_entry *
rxr_rma_alloc_readrsp_tx_entry(struct rxr_ep *rxr_ep,
			       struct rxr_rx_entry *rx_entry);

int rxr_rma_init_readrsp_pkt(struct rxr_ep *ep,
			     struct rxr_tx_entry *tx_entry,
			     struct rxr_pkt_entry *pkt_entry);

void rxr_rma_handle_readrsp_sent(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry);

/* EOR related functions */
int rxr_rma_init_eor_pkt(struct rxr_ep *ep,
			 struct rxr_rx_entry *rx_entry,
			 struct rxr_pkt_entry *pkt_entry);

void rxr_rma_handle_eor_sent(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry);

size_t rxr_rma_post_shm_rma(struct rxr_ep *rxr_ep,
			    struct rxr_tx_entry *tx_entry);

extern struct fi_ops_rma rxr_ops_rma;

#endif
