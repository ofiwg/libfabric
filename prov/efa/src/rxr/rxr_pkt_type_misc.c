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

#include "efa.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_cntr.h"
#include "rxr_pkt_cmd.h"

/* This file define functons for the following packet type:
 *       CONNACK
 *       CTS
 *       READRSP
 *       RMA_CONTEXT
 *       EOR
 */

/*  CONNACK packet related functions */
ssize_t rxr_pkt_init_connack(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry,
			     fi_addr_t addr)
{
	struct rxr_connack_hdr *connack_hdr;

	connack_hdr = (struct rxr_connack_hdr *)pkt_entry->pkt;

	connack_hdr->type = RXR_CONNACK_PKT;
	connack_hdr->version = RXR_PROTOCOL_VERSION;
	connack_hdr->flags = 0;

	pkt_entry->pkt_size = RXR_CONNACK_HDR_SIZE;
	pkt_entry->addr = addr;
	return 0;
}

void rxr_pkt_post_connack(struct rxr_ep *ep,
			  struct rxr_peer *peer,
			  fi_addr_t addr)
{
	struct rxr_pkt_entry *pkt_entry;
	ssize_t ret;

	if (peer->state == RXR_PEER_ACKED)
		return;

	pkt_entry = rxr_pkt_entry_alloc(ep, ep->tx_pkt_efa_pool);
	if (OFI_UNLIKELY(!pkt_entry))
		return;

	rxr_pkt_init_connack(ep, pkt_entry, addr);

	/*
	 * TODO: Once we start using a core's selective completion capability,
	 * post the CONNACK packets without FI_COMPLETION.
	 */
	ret = rxr_pkt_entry_send(ep, pkt_entry, addr);

	/*
	 * Skip sending this connack on error and try again when processing the
	 * next RTS from this peer containing the source information
	 */
	if (OFI_UNLIKELY(ret)) {
		rxr_pkt_entry_release_tx(ep, pkt_entry);
		if (ret == -FI_EAGAIN)
			return;
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Failed to send a CONNACK packet: ret %zd\n", ret);
	} else {
		peer->state = RXR_PEER_ACKED;
	}
}

void rxr_pkt_handle_connack_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry,
				 fi_addr_t src_addr)
{
	struct rxr_peer *peer;

	/*
	 * We don't really need any information from the actual connack packet
	 * itself, just the src_addr from the CQE
	 */
	assert(src_addr != FI_ADDR_NOTAVAIL);
	peer = rxr_ep_get_peer(ep, src_addr);
	peer->state = RXR_PEER_ACKED;
	FI_DBG(&rxr_prov, FI_LOG_CQ,
	       "CONNACK received from %" PRIu64 "\n", src_addr);
	rxr_pkt_entry_release_rx(ep, pkt_entry);
}

/*  CTS packet related functions */
void rxr_pkt_calc_cts_window_credits(struct rxr_ep *ep, struct rxr_peer *peer,
				     uint64_t size, int request,
				     int *window, int *credits)
{
	struct efa_av *av;
	int num_peers;

	/*
	 * Adjust the peer credit pool based on the current AV size, which could
	 * have grown since the time this peer was initialized.
	 */
	av = rxr_ep_av(ep);
	num_peers = av->used - 1;
	if (num_peers && ofi_div_ceil(rxr_env.rx_window_size, num_peers) < peer->rx_credits)
		peer->rx_credits = ofi_div_ceil(peer->rx_credits, num_peers);

	/*
	 * Allocate credits for this transfer based on the request, the number
	 * of available data buffers, and the number of outstanding peers this
	 * endpoint is actively tracking in the AV. Also ensure that a minimum
	 * number of credits are allocated to the transfer so the sender can
	 * make progress.
	 */
	*credits = MIN(MIN(ep->available_data_bufs, ep->posted_bufs_efa),
		       peer->rx_credits);
	*credits = MIN(request, *credits);
	*credits = MAX(*credits, rxr_env.tx_min_credits);
	*window = MIN(size, *credits * ep->max_data_payload_size);
	if (peer->rx_credits > ofi_div_ceil(*window, ep->max_data_payload_size))
		peer->rx_credits -= ofi_div_ceil(*window, ep->max_data_payload_size);
}

ssize_t rxr_pkt_init_cts(struct rxr_ep *ep,
			 struct rxr_rx_entry *rx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	int window = 0;
	struct rxr_cts_hdr *cts_hdr;
	struct rxr_peer *peer;
	size_t bytes_left;

	cts_hdr = (struct rxr_cts_hdr *)pkt_entry->pkt;
	cts_hdr->type = RXR_CTS_PKT;
	cts_hdr->version = RXR_PROTOCOL_VERSION;
	cts_hdr->flags = 0;

	if (rx_entry->cq_entry.flags & FI_READ)
		cts_hdr->flags |= RXR_READ_REQ;

	cts_hdr->tx_id = rx_entry->tx_id;
	cts_hdr->rx_id = rx_entry->rx_id;

	bytes_left = rx_entry->total_len - rx_entry->bytes_done;
	peer = rxr_ep_get_peer(ep, rx_entry->addr);
	rxr_pkt_calc_cts_window_credits(ep, peer, bytes_left,
					rx_entry->credit_request,
					&window, &rx_entry->credit_cts);
	cts_hdr->window = window;
	pkt_entry->pkt_size = RXR_CTS_HDR_SIZE;
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->x_entry = (void *)rx_entry;
	return 0;
}

void rxr_pkt_handle_cts_sent(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;

	rx_entry = (struct rxr_rx_entry *)pkt_entry->x_entry;
	rx_entry->window = rxr_get_cts_hdr(pkt_entry->pkt)->window;
	ep->available_data_bufs -= rx_entry->credit_cts;

	/*
	 * Set a timer if available_bufs is exhausted. We may encounter a
	 * scenario where a peer has stopped responding so we need a fallback
	 * to replenish the credits.
	 */
	if (OFI_UNLIKELY(ep->available_data_bufs == 0))
		ep->available_data_bufs_ts = ofi_gettime_us();
}

void rxr_pkt_handle_cts_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_cts_hdr *cts_pkt;
	struct rxr_tx_entry *tx_entry;

	cts_pkt = (struct rxr_cts_hdr *)pkt_entry->pkt;
	if (cts_pkt->flags & RXR_READ_REQ)
		tx_entry = ofi_bufpool_get_ibuf(ep->readrsp_tx_entry_pool, cts_pkt->tx_id);
	else
		tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, cts_pkt->tx_id);

	tx_entry->rx_id = cts_pkt->rx_id;
	tx_entry->window = cts_pkt->window;

	/* Return any excess tx_credits that were borrowed for the request */
	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	tx_entry->credit_allocated = ofi_div_ceil(cts_pkt->window, ep->max_data_payload_size);
	if (tx_entry->credit_allocated < tx_entry->credit_request)
		peer->tx_credits += tx_entry->credit_request - tx_entry->credit_allocated;

	rxr_pkt_entry_release_rx(ep, pkt_entry);

	if (tx_entry->state != RXR_TX_SEND) {
		tx_entry->state = RXR_TX_SEND;
		dlist_insert_tail(&tx_entry->entry, &ep->tx_pending_list);
	}
}

/*  READRSP packet functions */
int rxr_pkt_init_readrsp(struct rxr_ep *ep,
			 struct rxr_tx_entry *tx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_pkt *readrsp_pkt;
	struct rxr_readrsp_hdr *readrsp_hdr;
	size_t mtu = ep->mtu_size;

	readrsp_pkt = (struct rxr_readrsp_pkt *)pkt_entry->pkt;
	readrsp_hdr = &readrsp_pkt->hdr;
	readrsp_hdr->type = RXR_READRSP_PKT;
	readrsp_hdr->version = RXR_PROTOCOL_VERSION;
	readrsp_hdr->flags = 0;
	readrsp_hdr->tx_id = tx_entry->tx_id;
	readrsp_hdr->rx_id = tx_entry->rx_id;
	readrsp_hdr->seg_size = ofi_copy_from_iov(readrsp_pkt->data,
						  mtu - RXR_READRSP_HDR_SIZE,
						  tx_entry->iov,
						  tx_entry->iov_count, 0);
	pkt_entry->pkt_size = RXR_READRSP_HDR_SIZE + readrsp_hdr->seg_size;
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = tx_entry;
	return 0;
}

void rxr_pkt_handle_readrsp_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;
	size_t data_len;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	data_len = rxr_get_readrsp_hdr(pkt_entry->pkt)->seg_size;
	tx_entry->state = RXR_TX_SENT_READRSP;
	tx_entry->bytes_sent += data_len;
	tx_entry->window -= data_len;
	assert(tx_entry->window >= 0);
	if (tx_entry->bytes_sent < tx_entry->total_len) {
		if (efa_mr_cache_enable && rxr_ep_mr_local(ep))
			rxr_inline_mr_reg(rxr_ep_domain(ep), tx_entry);

		tx_entry->state = RXR_TX_SEND;
		dlist_insert_tail(&tx_entry->entry,
				  &ep->tx_pending_list);
	}
}

void rxr_pkt_handle_readrsp_send_completion(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;
	struct rxr_readrsp_hdr *readrsp_hdr;

	readrsp_hdr = (struct rxr_readrsp_hdr *)pkt_entry->pkt;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	assert(tx_entry->cq_entry.flags & FI_READ);

	tx_entry->bytes_acked += readrsp_hdr->seg_size;
	if (tx_entry->total_len == tx_entry->bytes_acked)
		rxr_cq_handle_tx_completion(ep, tx_entry);
}

void rxr_pkt_handle_readrsp_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_pkt *readrsp_pkt = NULL;
	struct rxr_readrsp_hdr *readrsp_hdr = NULL;
	struct rxr_rx_entry *rx_entry = NULL;

	readrsp_pkt = (struct rxr_readrsp_pkt *)pkt_entry->pkt;
	readrsp_hdr = &readrsp_pkt->hdr;
	rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool, readrsp_hdr->rx_id);
	assert(rx_entry->cq_entry.flags & FI_READ);
	rx_entry->tx_id = readrsp_hdr->tx_id;
	rxr_pkt_proc_data(ep, rx_entry, pkt_entry,
			  readrsp_pkt->data,
			  0, readrsp_hdr->seg_size);
}

/*  RMA_CONTEXT packet functions */
void rxr_pkt_handle_rma_context_send_completion(struct rxr_ep *ep,
						struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry = NULL;
	struct rxr_rx_entry *rx_entry = NULL;
	struct rxr_rma_context_pkt *rma_context_pkt;
	int ret;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->version == RXR_PROTOCOL_VERSION);

	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->pkt;

	switch (rma_context_pkt->rma_context_type) {
	case RXR_SHM_RMA_READ:
	case RXR_SHM_RMA_WRITE:
		/* Completion of RMA READ/WRTITE operations that apps call */
		tx_entry = pkt_entry->x_entry;

		if (tx_entry->fi_flags & FI_COMPLETION) {
			rxr_cq_write_tx_completion(ep, tx_entry);
		} else {
			efa_cntr_report_tx_completion(&ep->util_ep, tx_entry->cq_entry.flags);
			rxr_release_tx_entry(ep, tx_entry);
		}
		rxr_pkt_entry_release_tx(ep, pkt_entry);
		break;
	case RXR_SHM_LARGE_READ:
		/*
		 * This must be on the receiver (remote) side of two-sided message
		 * transfer, which is also the initiator of RMA READ.
		 * We get RMA READ completion for previously issued
		 * fi_read operation over shm provider, which means
		 * receiver side has received all data from sender
		 */
		rx_entry = pkt_entry->x_entry;
		rx_entry->cq_entry.len = rx_entry->total_len;
		rx_entry->bytes_done = rx_entry->total_len;

		ret = rxr_pkt_post_ctrl_or_queue(ep, RXR_TX_ENTRY, rx_entry, RXR_EOR_PKT, 1);
		if (ret) {
			if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
				assert(0 && "error writing error cq entry for EOR\n");
		}

		if (rx_entry->fi_flags & FI_MULTI_RECV)
			rxr_msg_multi_recv_handle_completion(ep, rx_entry);
		rxr_cq_write_rx_completion(ep, rx_entry);
		rxr_msg_multi_recv_free_posted_entry(ep, rx_entry);
		if (OFI_LIKELY(!ret))
			rxr_release_rx_entry(ep, rx_entry);
		rxr_pkt_entry_release_rx(ep, pkt_entry);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "invalid rma_context_type in RXR_RMA_CONTEXT_PKT %d\n",
			rma_context_pkt->rma_context_type);
		assert(0 && "invalid RXR_RMA_CONTEXT_PKT rma_context_type\n");
	}
}

/*  EOR packet related functions */
int rxr_pkt_init_eor(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *eor_hdr;

	eor_hdr = (struct rxr_eor_hdr *)pkt_entry->pkt;
	eor_hdr->type = RXR_EOR_PKT;
	eor_hdr->version = RXR_PROTOCOL_VERSION;
	eor_hdr->flags = 0;
	eor_hdr->tx_id = rx_entry->tx_id;
	eor_hdr->rx_id = rx_entry->rx_id;
	pkt_entry->pkt_size = sizeof(struct rxr_eor_hdr);
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->x_entry = rx_entry;
	return 0;
}

/*
 *   Sender handles the acknowledgment (RXR_EOR_PKT) from receiver on the completion
 *   of the large message copy via fi_readmsg operation
 */
void rxr_pkt_handle_eor_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *shm_eor;
	struct rxr_tx_entry *tx_entry;

	shm_eor = (struct rxr_eor_hdr *)pkt_entry->pkt;

	/* pre-post buf used here, so can NOT track back to tx_entry with x_entry */
	tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, shm_eor->tx_id);
	rxr_cq_write_tx_completion(ep, tx_entry);
	rxr_pkt_entry_release_rx(ep, pkt_entry);
}

