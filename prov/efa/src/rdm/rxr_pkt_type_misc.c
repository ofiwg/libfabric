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

#include "efa.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_cntr.h"
#include "rxr_pkt_cmd.h"
#include "rxr_pkt_type_base.h"
#include "rxr_pkt_type_misc.h"

#include "rxr_tp.h"

/* This file define functons for the following packet type:
 *       HANDSHAKE
 *       CTS
 *       READRSP
 *       RMA_CONTEXT
 *       EOR
 */

/* HANDSHAKE packet related functions */

ssize_t rxr_pkt_init_handshake(struct rxr_ep *ep,
			       struct rxr_pkt_entry *pkt_entry,
			       fi_addr_t addr)
{
	int nex;
	struct rxr_handshake_hdr *handshake_hdr;
	struct rxr_handshake_opt_connid_hdr *connid_hdr;
	struct rxr_handshake_opt_host_id_hdr *host_id_hdr;

	handshake_hdr = (struct rxr_handshake_hdr *)pkt_entry->wiredata;
	handshake_hdr->type = RXR_HANDSHAKE_PKT;
	handshake_hdr->version = RXR_PROTOCOL_VERSION;
	handshake_hdr->flags = 0;

	nex = (RXR_NUM_EXTRA_FEATURE_OR_REQUEST-1)/64 + 1;
	/*
	 * The action of plus 3 is for backward compatibility.
	 * See section 2.1 of protocol v4 document for detail.
	 */
	handshake_hdr->nextra_p3 = nex + 3;
	memcpy(handshake_hdr->extra_info, ep->extra_info, nex * sizeof(uint64_t));
	pkt_entry->pkt_size = sizeof(struct rxr_handshake_hdr) + nex * sizeof(uint64_t);

	/*
	 * Always include connid at the end of a handshake packet.
	 * If peer cannot make use of connid, the connid will be ignored.
	 */
	connid_hdr = (struct rxr_handshake_opt_connid_hdr *)(pkt_entry->wiredata + pkt_entry->pkt_size);
	connid_hdr->connid = rxr_ep_raw_addr(ep)->qkey;
	handshake_hdr->flags |= RXR_PKT_CONNID_HDR;
	pkt_entry->pkt_size += sizeof(struct rxr_handshake_opt_connid_hdr);

	/**
	 * Include the optional host id if it is non-zero
	 */
	if (ep->host_id) {
		host_id_hdr = (struct rxr_handshake_opt_host_id_hdr *)(pkt_entry->wiredata + pkt_entry->pkt_size);
		host_id_hdr->host_id = ep->host_id;
		handshake_hdr->flags |= RXR_HANDSHAKE_HOST_ID_HDR;
		pkt_entry->pkt_size += sizeof(struct rxr_handshake_opt_host_id_hdr);
	}

	pkt_entry->addr = addr;
	return 0;
}

/** @brief Post a handshake packet to a peer.
 *
 * @param ep The endpoint on which the handshake packet is sent out.
 * @param peer The peer to which the handshake packet is posted.
 * @return 0 on success, fi_errno on error.
 */
ssize_t rxr_pkt_post_handshake(struct rxr_ep *ep, struct efa_rdm_peer *peer)
{
	struct rxr_pkt_entry *pkt_entry;
	fi_addr_t addr;
	ssize_t ret;

	addr = peer->efa_fiaddr;
	pkt_entry = rxr_pkt_entry_alloc(ep, ep->efa_tx_pkt_pool, RXR_PKT_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	rxr_pkt_init_handshake(ep, pkt_entry, addr);

	ret = rxr_pkt_entry_send(ep, pkt_entry, 0);
	if (OFI_UNLIKELY(ret)) {
		rxr_pkt_entry_release_tx(ep, pkt_entry);
	}
	return ret;
}

/** @brief Post a handshake packet to a peer.
 *
 * This function ensures an endpoint post one and only one handshake
 * to a peer.
 *
 * For a peer that the endpoint has not attempted to send handshake,
 * it will send a handshake packet.
 *
 * If the send succeeded, EFA_RDM_PEER_HANDSHAKE_SENT flag will be set to peer->flags.
 *
 * If the send encountered FI_EAGAIN failure, the peer will be added to
 * rxr_ep->handshake_queued_peer_list. The handshake will be resend later
 * by the progress engine.
 *
 * If the send encountered other failure, an EQ entry will be written.
 *
 * To ensure only one handshake is send to a peer, the function will not send
 * packet to a peer whose peer->flags has either EFA_RDM_PEER_HANDSHAKE_SENT or
 * EFA_RDM_PEER_HANDSHAKE_QUEUED.
 *
 * @param[in]	ep	The endpoint on which the handshake packet is sent out.
 * @param[in]	peer	The peer to which the handshake packet is posted.
 * @return 	void.
 */
void rxr_pkt_post_handshake_or_queue(struct rxr_ep *ep, struct efa_rdm_peer *peer)
{
	ssize_t err;

	if (peer->flags & (EFA_RDM_PEER_HANDSHAKE_SENT | EFA_RDM_PEER_HANDSHAKE_QUEUED))
		return;

	err = rxr_pkt_post_handshake(ep, peer);
	if (OFI_UNLIKELY(err == -FI_EAGAIN)) {
		/* add peer to handshake_queued_peer_list for retry later */
		peer->flags |= EFA_RDM_PEER_HANDSHAKE_QUEUED;
		dlist_insert_tail(&peer->handshake_queued_entry,
				  &ep->handshake_queued_peer_list);
		return;
	}

	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_EP_CTRL,
			"Failed to post HANDSHAKE to peer %ld: %s\n",
			peer->efa_fiaddr, fi_strerror(-err));
		efa_eq_write_error(&ep->base_ep.util_ep, FI_EIO, FI_EFA_ERR_PEER_HANDSHAKE);
		return;
	}

	peer->flags |= EFA_RDM_PEER_HANDSHAKE_SENT;
}

void rxr_pkt_handle_handshake_recv(struct rxr_ep *ep,
				   struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_peer *peer;
	struct rxr_handshake_hdr *handshake_pkt;
	uint64_t *host_id_ptr;

	assert(pkt_entry->addr != FI_ADDR_NOTAVAIL);
	EFA_DBG(FI_LOG_CQ,
			"HANDSHAKE received from %" PRIu64 "\n", pkt_entry->addr);

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	handshake_pkt = (struct rxr_handshake_hdr *)pkt_entry->wiredata;

	/* nextra_p3 is number of members in extra_info plus 3.
	 * See section 2.1 of protocol v4 document for detail
	 */
	peer->nextra_p3 = handshake_pkt->nextra_p3;
	memcpy(peer->extra_info, handshake_pkt->extra_info,
		   (handshake_pkt->nextra_p3 - 3) * sizeof(uint64_t));
	peer->flags |= EFA_RDM_PEER_HANDSHAKE_RECEIVED;

	host_id_ptr = rxr_pkt_handshake_host_id_ptr(pkt_entry);
	if (host_id_ptr) {
		peer->host_id = *host_id_ptr;
		EFA_INFO(FI_LOG_CQ, "Received peer host id: i-%017lx\n", peer->host_id);
	}

	rxr_pkt_entry_release_rx(ep, pkt_entry);
}

/*  CTS packet related functions */
ssize_t rxr_pkt_init_cts(struct rxr_ep *ep,
			 struct efa_rdm_ope *ope,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_cts_hdr *cts_hdr;
	size_t bytes_left;

	cts_hdr = (struct rxr_cts_hdr *)pkt_entry->wiredata;
	cts_hdr->type = RXR_CTS_PKT;
	cts_hdr->version = RXR_PROTOCOL_VERSION;
	cts_hdr->flags = 0;

	if (ope->cq_entry.flags & FI_READ)
		cts_hdr->flags |= RXR_CTS_READ_REQ;

	/* CTS is sent by different communication protocols.
	 * CTS is sent using tx_entry in the emulated longcts read
	 * protocol. The emulated longcts write and the longcts
	 * message protocols sends CTS using rx_entry.
	 * This check ensures appropriate tx_id and rx_id are
	 * assigned for the respective protocols */
	if (ope->type == EFA_RDM_TXE){
		cts_hdr->send_id = ope->rx_id;
		cts_hdr->recv_id = ope->tx_id;
	} else {
		assert(ope->type == EFA_RDM_RXE);
		cts_hdr->send_id = ope->tx_id;
		cts_hdr->recv_id = ope->rx_id;
	}

	bytes_left = ope->total_len - ope->bytes_received;
	cts_hdr->recv_length = MIN(bytes_left, rxr_env.tx_min_credits * ep->max_data_payload_size);
	assert(cts_hdr->recv_length > 0);
	pkt_entry->pkt_size = sizeof(struct rxr_cts_hdr);

	/*
	 * always set connid header. If the peer does not need it,
	 * it will be ignored.
	 */
	cts_hdr->flags |= RXR_PKT_CONNID_HDR;
	cts_hdr->connid = rxr_ep_raw_addr(ep)->qkey;

	pkt_entry->addr = ope->addr;
	pkt_entry->ope = (void *)ope;
	return 0;
}

void rxr_pkt_handle_cts_sent(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *ope;

	ope = (struct efa_rdm_ope *)pkt_entry->ope;
	ope->window = rxr_get_cts_hdr(pkt_entry->wiredata)->recv_length;
}

void rxr_pkt_handle_cts_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_cts_hdr *cts_pkt;
	struct efa_rdm_ope *ope;

	cts_pkt = (struct rxr_cts_hdr *)pkt_entry->wiredata;
	ope = ofi_bufpool_get_ibuf(ep->ope_pool, cts_pkt->send_id);

	ope->rx_id = cts_pkt->recv_id;
	ope->window = cts_pkt->recv_length;
	assert(ope->window > 0);

	rxr_pkt_entry_release_rx(ep, pkt_entry);

	if (ope->state != EFA_RDM_TXE_SEND) {
		ope->state = EFA_RDM_TXE_SEND;
		dlist_insert_tail(&ope->entry, &ep->ope_longcts_send_list);
	}
}

/*  READRSP packet functions */
int rxr_pkt_init_readrsp(struct rxr_ep *ep,
			 struct efa_rdm_ope *rx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_hdr *readrsp_hdr;
	int ret;

	readrsp_hdr = rxr_get_readrsp_hdr(pkt_entry->wiredata);
	readrsp_hdr->type = RXR_READRSP_PKT;
	readrsp_hdr->version = RXR_PROTOCOL_VERSION;
	readrsp_hdr->flags = 0;
	readrsp_hdr->send_id = rx_entry->rx_id;
	readrsp_hdr->recv_id = rx_entry->tx_id;
	readrsp_hdr->flags |= RXR_PKT_CONNID_HDR;
	readrsp_hdr->connid = rxr_ep_raw_addr(ep)->qkey;
	readrsp_hdr->seg_length = MIN(ep->mtu_size - sizeof(struct rxr_readrsp_hdr),
				      rx_entry->total_len);
	pkt_entry->addr = rx_entry->addr;
	ret = rxr_pkt_init_data_from_ope(ep, pkt_entry, sizeof(struct rxr_readrsp_hdr),
					      rx_entry, 0, readrsp_hdr->seg_length);
	return ret;
}

void rxr_pkt_handle_readrsp_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rx_entry;
	size_t data_len;

	rx_entry = (struct efa_rdm_ope *)pkt_entry->ope;
	data_len = rxr_get_readrsp_hdr(pkt_entry->wiredata)->seg_length;
	rx_entry->bytes_sent += data_len;
	rx_entry->window -= data_len;
	assert(rx_entry->window >= 0);
	if (rx_entry->bytes_sent < rx_entry->total_len) {
		if (efa_is_cache_available(rxr_ep_domain(ep)))
			efa_rdm_ope_try_fill_desc(rx_entry, 0, FI_SEND);

		rx_entry->state = EFA_RDM_TXE_SEND;
		dlist_insert_tail(&rx_entry->entry, &ep->ope_longcts_send_list);
	}
}

void rxr_pkt_handle_readrsp_send_completion(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rx_entry;
	struct rxr_readrsp_hdr *readrsp_hdr;

	readrsp_hdr = (struct rxr_readrsp_hdr *)pkt_entry->wiredata;

	rx_entry = (struct efa_rdm_ope *)pkt_entry->ope;
	assert(rx_entry->cq_entry.flags & FI_READ);
	rx_entry->bytes_acked += readrsp_hdr->seg_length;
	if (rx_entry->total_len == rx_entry->bytes_acked)
		efa_rdm_ope_handle_send_completed(rx_entry);
}

void rxr_pkt_handle_readrsp_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_pkt *readrsp_pkt = NULL;
	struct rxr_readrsp_hdr *readrsp_hdr = NULL;
	struct efa_rdm_ope *tx_entry = NULL;

	readrsp_pkt = (struct rxr_readrsp_pkt *)pkt_entry->wiredata;
	readrsp_hdr = &readrsp_pkt->hdr;
	tx_entry = ofi_bufpool_get_ibuf(ep->ope_pool, readrsp_hdr->recv_id);
	assert(tx_entry->cq_entry.flags & FI_READ);
	tx_entry->rx_id = readrsp_hdr->send_id;
	rxr_pkt_proc_data(ep, tx_entry, pkt_entry,
			  readrsp_pkt->data,
			  0, readrsp_hdr->seg_length);
}

/*  RMA_CONTEXT packet functions
 *
 *  RMA context packet is used a context of RMA operations and is not
 *  sent over wire. It is named packet because currently all EFA operation
 *  use a packet as context.
 */
void rxr_pkt_init_write_context(struct efa_rdm_ope *tx_entry,
				struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rma_context_pkt *rma_context_pkt;

	pkt_entry->ope = (void *)tx_entry;
	pkt_entry->addr = tx_entry->addr;
	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->wiredata;
	rma_context_pkt->type = RXR_RMA_CONTEXT_PKT;
	rma_context_pkt->version = RXR_PROTOCOL_VERSION;
	rma_context_pkt->context_type = RXR_WRITE_CONTEXT;
	rma_context_pkt->tx_id = tx_entry->tx_id;
}

void rxr_pkt_init_read_context(struct rxr_ep *rxr_ep,
			       void  *x_entry,
			       fi_addr_t addr,
			       int read_id,
			       size_t seg_size,
			       struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rma_context_pkt *ctx_pkt;

	pkt_entry->ope = x_entry;
	pkt_entry->addr = addr;
	pkt_entry->pkt_size = sizeof(struct rxr_rma_context_pkt);

	ctx_pkt = (struct rxr_rma_context_pkt *)pkt_entry->wiredata;
	ctx_pkt->type = RXR_RMA_CONTEXT_PKT;
	ctx_pkt->flags = 0;
	ctx_pkt->version = RXR_PROTOCOL_VERSION;
	ctx_pkt->context_type = RXR_READ_CONTEXT;
	ctx_pkt->read_id = read_id;
	ctx_pkt->seg_size = seg_size;
}

static
void rxr_pkt_handle_rma_read_completion(struct rxr_ep *ep,
					struct rxr_pkt_entry *context_pkt_entry)
{
	enum efa_rdm_ope_type x_entry_type;
	struct efa_rdm_ope *tx_entry;
	struct efa_rdm_ope *rx_entry;
	struct rxr_pkt_entry *data_pkt_entry;
	struct rxr_rma_context_pkt *rma_context_pkt;
	size_t data_size;
	int err;

	rma_context_pkt = (struct rxr_rma_context_pkt *)context_pkt_entry->wiredata;
	assert(rma_context_pkt->type == RXR_RMA_CONTEXT_PKT);
	assert(rma_context_pkt->context_type == RXR_READ_CONTEXT);

	x_entry_type = context_pkt_entry->ope->type;

	if (x_entry_type == EFA_RDM_TXE) {
		tx_entry = context_pkt_entry->ope;
		assert(tx_entry->op == ofi_op_read_req);
		tx_entry->bytes_read_completed += rma_context_pkt->seg_size;
		if (tx_entry->bytes_read_total_len == tx_entry->bytes_read_completed) {
			if (tx_entry->addr == FI_ADDR_NOTAVAIL) {
				data_pkt_entry = tx_entry->local_read_pkt_entry;
				data_size = rxr_pkt_data_size(data_pkt_entry);
				assert(data_size > 0);
				rxr_pkt_handle_data_copied(ep, data_pkt_entry, data_size);
			} else {
				assert(tx_entry && tx_entry->cq_entry.flags & FI_READ);
				efa_rdm_txe_report_completion(tx_entry);
			}

			efa_rdm_txe_release(tx_entry);
		}
	} else {
		assert(x_entry_type == EFA_RDM_RXE);
		rx_entry = context_pkt_entry->ope;
		rx_entry->bytes_read_completed += rma_context_pkt->seg_size;
		assert(rx_entry->bytes_read_completed <= rx_entry->bytes_read_total_len);
		if (rx_entry->bytes_read_completed == rx_entry->bytes_read_total_len) {
			rxr_tracepoint(read_completed,
				    rx_entry->msg_id, (size_t) rx_entry->cq_entry.op_context,
				    rx_entry->total_len, (size_t) rx_entry);
			err = rxr_pkt_post_or_queue(ep, rx_entry, RXR_EOR_PKT);
			if (OFI_UNLIKELY(err)) {
				EFA_WARN(FI_LOG_CQ,
					"Posting of EOR failed! err=%s(%d)\n",
					fi_strerror(-err), -err);
				efa_rdm_rxe_handle_error(rx_entry, -err, FI_EFA_ERR_PKT_POST);
				efa_rdm_rxe_release(rx_entry);
			}

			rx_entry->rxr_flags |= EFA_RDM_RXE_EOR_IN_FLIGHT;
			rx_entry->bytes_received += rx_entry->bytes_read_completed;
			rx_entry->bytes_copied += rx_entry->bytes_read_completed;
			if (rx_entry->bytes_copied == rx_entry->total_len) {
				efa_rdm_ope_handle_recv_completed(rx_entry);
			} else if(rx_entry->bytes_copied + rx_entry->bytes_queued_blocking_copy == rx_entry->total_len) {
				rxr_ep_flush_queued_blocking_copy_to_hmem(ep);
			}
		}
	}
}

/**
 * @brief Handle a single RMA completion (read or write)
 *
 * RMA Completion can either be caused by the completion of an RDMA Read,
 * an emulated RDMA Write, or a true RDMA Write.
 *
 * Note that if true RDMA Read/Write was used, the packet here was not sent over
 * but only used as the context for the request.
 *
 * @param ep[in,out]			Endpoint
 * @param context_pkt_entry[in,out]	The "Packet" which serves as context
 */
void rxr_pkt_handle_rma_completion(struct rxr_ep *ep,
				   struct rxr_pkt_entry *context_pkt_entry)
{
	struct efa_rdm_ope *tx_entry = NULL;
	struct rxr_rma_context_pkt *rma_context_pkt;

	assert(rxr_get_base_hdr(context_pkt_entry->wiredata)->version == RXR_PROTOCOL_VERSION);

	rma_context_pkt = (struct rxr_rma_context_pkt *)context_pkt_entry->wiredata;

	switch (rma_context_pkt->context_type) {
	case RXR_WRITE_CONTEXT:
		tx_entry = (struct efa_rdm_ope *)context_pkt_entry->ope;
		tx_entry->bytes_write_completed += rma_context_pkt->seg_size;
		if (tx_entry->bytes_write_completed == tx_entry->bytes_write_total_len) {
			if (tx_entry->fi_flags & FI_COMPLETION)
				efa_rdm_txe_report_completion(tx_entry);
			else
				efa_cntr_report_tx_completion(&ep->base_ep.util_ep, tx_entry->cq_entry.flags);
			efa_rdm_txe_release(tx_entry);
		}
		break;
	case RXR_READ_CONTEXT:
		rxr_pkt_handle_rma_read_completion(ep, context_pkt_entry);
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "invalid rma_context_type in RXR_RMA_CONTEXT_PKT %d\n",
			rma_context_pkt->context_type);
		assert(0 && "invalid RXR_RMA_CONTEXT_PKT rma_context_type\n");
	}

	rxr_ep_record_tx_op_completed(ep, context_pkt_entry);
	rxr_pkt_entry_release_tx(ep, context_pkt_entry);
}

/*  EOR packet related functions */
int rxr_pkt_init_eor(struct rxr_ep *ep, struct efa_rdm_ope *rx_entry, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *eor_hdr;

	eor_hdr = (struct rxr_eor_hdr *)pkt_entry->wiredata;
	eor_hdr->type = RXR_EOR_PKT;
	eor_hdr->version = RXR_PROTOCOL_VERSION;
	eor_hdr->flags = 0;
	eor_hdr->send_id = rx_entry->tx_id;
	eor_hdr->recv_id = rx_entry->rx_id;
	eor_hdr->flags |= RXR_PKT_CONNID_HDR;
	eor_hdr->connid = rxr_ep_raw_addr(ep)->qkey;
	pkt_entry->pkt_size = sizeof(struct rxr_eor_hdr);
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->ope = rx_entry;
	return 0;
}

void rxr_pkt_handle_eor_send_completion(struct rxr_ep *ep,
					struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rx_entry;

	rx_entry = pkt_entry->ope;
	assert(rx_entry && rx_entry->rx_id == rxr_get_eor_hdr(pkt_entry->wiredata)->recv_id);

	if (rx_entry->bytes_copied == rx_entry->total_len) {
		efa_rdm_rxe_release(rx_entry);
	} else {
		rx_entry->rxr_flags &= ~EFA_RDM_RXE_EOR_IN_FLIGHT;
	}
}

/*
 *   Sender handles the acknowledgment (RXR_EOR_PKT) from receiver on the completion
 *   of the large message copy via fi_readmsg operation
 */
void rxr_pkt_handle_eor_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *eor_hdr;
	struct efa_rdm_ope *tx_entry;
	struct efa_rdm_peer *peer;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	peer->num_read_msg_in_flight -= 1;

	eor_hdr = (struct rxr_eor_hdr *)pkt_entry->wiredata;

	/* pre-post buf used here, so can NOT track back to tx_entry with x_entry */
	tx_entry = ofi_bufpool_get_ibuf(ep->ope_pool, eor_hdr->send_id);

	tx_entry->bytes_acked += tx_entry->total_len - tx_entry->bytes_runt;
	if (tx_entry->bytes_acked == tx_entry->total_len) {
		efa_rdm_txe_report_completion(tx_entry);
		efa_rdm_txe_release(tx_entry);
	}

	rxr_pkt_entry_release_rx(ep, pkt_entry);

}

/* receipt packet related functions */
int rxr_pkt_init_receipt(struct rxr_ep *ep, struct efa_rdm_ope *rx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_receipt_hdr *receipt_hdr;

	receipt_hdr = rxr_get_receipt_hdr(pkt_entry->wiredata);
	receipt_hdr->type = RXR_RECEIPT_PKT;
	receipt_hdr->version = RXR_PROTOCOL_VERSION;
	receipt_hdr->flags = 0;
	receipt_hdr->tx_id = rx_entry->tx_id;
	receipt_hdr->msg_id = rx_entry->msg_id;
	receipt_hdr->flags |= RXR_PKT_CONNID_HDR;
	receipt_hdr->connid = rxr_ep_raw_addr(ep)->qkey;

	pkt_entry->pkt_size = sizeof(struct rxr_receipt_hdr);
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->ope = rx_entry;

	return 0;
}

void rxr_pkt_handle_receipt_sent(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry)
{
}

void rxr_pkt_handle_receipt_send_completion(struct rxr_ep *ep,
					    struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rx_entry;

	rx_entry = (struct efa_rdm_ope *)pkt_entry->ope;
	efa_rdm_rxe_release(rx_entry);
}

/* atomrsp packet related functions: init, handle_sent, handle_send_completion and recv
 *
 * initialize atomic response packet by creating a packet that hold original data
 * in rx_entry->iov. rx_entry->iov will then be changed by atomic operation.
 * release that packet entry until it is sent.
 */
int rxr_pkt_init_atomrsp(struct rxr_ep *ep, struct efa_rdm_ope *rx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_atomrsp_pkt *atomrsp_pkt;
	struct rxr_atomrsp_hdr *atomrsp_hdr;

	assert(rx_entry->atomrsp_data);
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->ope = rx_entry;

	atomrsp_pkt = (struct rxr_atomrsp_pkt *)pkt_entry->wiredata;
	atomrsp_hdr = &atomrsp_pkt->hdr;
	atomrsp_hdr->type = RXR_ATOMRSP_PKT;
	atomrsp_hdr->version = RXR_PROTOCOL_VERSION;
	atomrsp_hdr->flags = 0;
	atomrsp_hdr->recv_id = rx_entry->tx_id;
	atomrsp_hdr->seg_length = ofi_total_iov_len(rx_entry->iov, rx_entry->iov_count);
	atomrsp_hdr->flags |= RXR_PKT_CONNID_HDR;
	atomrsp_hdr->connid = rxr_ep_raw_addr(ep)->qkey;
	assert(sizeof(struct rxr_atomrsp_hdr) + atomrsp_hdr->seg_length < ep->mtu_size);
	/* rx_entry->atomrsp_data was filled in rxr_pkt_handle_req_recv() */
	memcpy(atomrsp_pkt->data, rx_entry->atomrsp_data, atomrsp_hdr->seg_length);
	pkt_entry->pkt_size = sizeof(struct rxr_atomrsp_hdr) + atomrsp_hdr->seg_length;
	return 0;
}

void rxr_pkt_handle_atomrsp_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
}

void rxr_pkt_handle_atomrsp_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rx_entry;

	rx_entry = (struct efa_rdm_ope *)pkt_entry->ope;
	ofi_buf_free(rx_entry->atomrsp_data);
	rx_entry->atomrsp_data = NULL;
	efa_rdm_rxe_release(rx_entry);
}

void rxr_pkt_handle_atomrsp_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_atomrsp_pkt *atomrsp_pkt = NULL;
	struct rxr_atomrsp_hdr *atomrsp_hdr = NULL;
	struct efa_rdm_ope *tx_entry = NULL;
	ssize_t ret;

	atomrsp_pkt = (struct rxr_atomrsp_pkt *)pkt_entry->wiredata;
	atomrsp_hdr = &atomrsp_pkt->hdr;
	tx_entry = ofi_bufpool_get_ibuf(ep->ope_pool, atomrsp_hdr->recv_id);

	ret = efa_copy_to_hmem_iov(tx_entry->atomic_ex.result_desc, tx_entry->atomic_ex.resp_iov,
	                           tx_entry->atomic_ex.resp_iov_count, atomrsp_pkt->data,
	                           atomrsp_hdr->seg_length);
	if (OFI_UNLIKELY(ret < 0)) {
		efa_eq_write_error(&ep->base_ep.util_ep, FI_EMSGSIZE, FI_EFA_LOCAL_ERROR_BAD_LENGTH);
		return;
	}

	if (tx_entry->fi_flags & FI_COMPLETION)
		efa_rdm_txe_report_completion(tx_entry);
	else
		efa_cntr_report_tx_completion(&ep->base_ep.util_ep, tx_entry->cq_entry.flags);

	efa_rdm_txe_release(tx_entry);
	rxr_pkt_entry_release_rx(ep, pkt_entry);
}

void rxr_pkt_handle_receipt_recv(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *tx_entry = NULL;
	struct rxr_receipt_hdr *receipt_hdr;

	receipt_hdr = rxr_get_receipt_hdr(pkt_entry->wiredata);
	/* Retrieve the tx_entry that will be written into TX CQ*/
	tx_entry = ofi_bufpool_get_ibuf(ep->ope_pool,
					receipt_hdr->tx_id);
	if (!tx_entry) {
		EFA_WARN(FI_LOG_CQ,
			"Failed to retrive the tx_entry when hadling receipt packet.\n");
		return;
	}

	efa_rdm_ope_handle_send_completed(tx_entry);
	rxr_pkt_entry_release_rx(ep, pkt_entry);
}
