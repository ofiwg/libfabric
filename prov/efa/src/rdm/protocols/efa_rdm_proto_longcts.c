/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_nonreq.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_LONGCTS_MSGRTM_PKT
 * EFA_RDM_LONGCTS_TAGRTM_PKT
 * EFA_RDM_DC_LONGCTS_MSGRTM_PKT
 * EFA_RDM_DC_LONGCTS_TAGRTM_PKT
 *
 * Sent by the receiver to open a send window
 * EFA_RDM_CTS_PKT
 *
 * Carries the rest of the message, one window at a time
 * EFA_RDM_CTSDATA_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#long-cts-message-featuresubprotocol
 */

static ssize_t
efa_rdm_proto_longcts_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;
	ssize_t ret;
#if ENABLE_DEBUG
	struct efa_rdm_ep *ep = pkt_entry->ep;
#endif

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);
	rxe->tx_id =
		efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)->send_id;
	rxe->bytes_received += pkt_entry->payload_size;
	ret = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
	if (ret)
		return ret;

#if ENABLE_DEBUG
	dlist_insert_tail(&rxe->pending_recv_entry, &ep->ope_recv_list);
	ep->pending_recv_counter++;
#endif
	rxe->state = EFA_RDM_RXE_RECV;
	return efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_CTS_PKT);
}

void efa_rdm_proto_longcts_handle_cts_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep = pkt_entry->ep;
	struct efa_rdm_cts_hdr *cts_pkt =
		(struct efa_rdm_cts_hdr *) pkt_entry->wiredata;
	struct efa_rdm_ope *ope =
		efa_rdm_ep_live_ope_from_id(ep, cts_pkt->send_id);

	/*
	 * Drop a CTS whose id no longer names the operation that created it.
	 */
	if (OFI_UNLIKELY(!ope)) {
		EFA_INFO(FI_LOG_CQ,
			 "CTS names ope id %" PRIu32 ", which no longer holds "
			 "the operation that requested it. Dropping the CTS.\n",
			 cts_pkt->send_id);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	ope->rx_id = cts_pkt->recv_id;
	ope->window = cts_pkt->recv_length;
	assert(ope->window > 0);

	efa_rdm_pke_release_rx(pkt_entry);

	if (ope->state != EFA_RDM_OPE_SEND) {
		ope->state = EFA_RDM_OPE_SEND;
		dlist_insert_tail(&ope->entry, &ep->ope_longcts_send_list);
		efa_rdm_ep_enqueue_progress_list(ep);
	}
}

void efa_rdm_proto_longcts_handle_ctsdata_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ctsdata_hdr *data_hdr =
		efa_rdm_pke_get_ctsdata_hdr(pkt_entry);
	struct efa_rdm_ope *ope =
		efa_rdm_ep_get_ope_from_ope_id(pkt_entry->ep, data_hdr->recv_id);
	size_t hdr_size = sizeof (struct efa_rdm_ctsdata_hdr);

	if (data_hdr->flags & EFA_RDM_PKT_CONNID_HDR)
		hdr_size += sizeof (struct efa_rdm_ctsdata_opt_connid_hdr);

	efa_rdm_pke_proc_ctsdata(pkt_entry, ope,
				 pkt_entry->wiredata + hdr_size,
				 data_hdr->seg_offset,
				 data_hdr->seg_length);
}

/**
 * @brief Long CTS takes any send the protocols before it decline, so it is
 * last in the registry.
 */
static bool efa_rdm_proto_longcts_can_use_for_send(struct efa_rdm_ope *txe,
						   int req_pkt_type,
						   uint16_t header_flags,
						   int iface, bool use_p2p)
{
	return true;
}

EFA_RDM_PROTO_DEF(longcts,
	.wants_mr = true,
	.can_use_protocol = &efa_rdm_proto_longcts_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longcts_construct_req_pke,
	.post_continuation = &efa_rdm_proto_post_ctsdata,
	.req_pkt_type = EFA_RDM_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGCTS_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_LONGCTS_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longcts_handle_req_posted,
	.handle_unexp_pke_match = &efa_rdm_proto_longcts_handle_matched_rtm,
);

/* TX path callbacks */

/**
 * @brief Record the user data carried by the REQ in bytes_sent.
 *
 * Only the REQ goes through this hook.
 */
void efa_rdm_proto_longcts_handle_req_posted(
	struct efa_rdm_ep *ep, struct efa_rdm_ope *txe)
{
	assert(ep->send_pkt_entry_vec_size == 1);

	/* A read NACK REQ carries no data, so bytes_sent stays as it is. */
	if (txe->internal_flags & EFA_RDM_OPE_READ_NACK)
		assert(ep->send_pkt_entry_vec[0]->payload_size == 0);
	else
		txe->bytes_sent = ep->send_pkt_entry_vec[0]->payload_size;

	assert(txe->bytes_sent < txe->total_len);

	/* Retry registering the source buffer so CTSDATA can send from it. */
	if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
		efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
}

/**
 * @brief Account for a completed send and complete the txe once the whole
 * message is done.
 */
static void efa_rdm_proto_longcts_account_send_completion(
	struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *txe)
{
	bool delivery_complete_requested =
		txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	/* Delivery complete tracks outstanding TX ops instead of bytes. */
	if (!delivery_complete_requested)
		txe->bytes_acked += pkt_entry->payload_size;

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (delivery_complete_requested) {
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else if (efa_rdm_ope_send_done(txe)) {
		efa_rdm_ope_handle_send_completed(txe);
	}
}

/**
 * @brief Handle send completion for a long CTS RTM packet.
 */
ssize_t efa_rdm_proto_longcts_handle_req_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	efa_rdm_proto_longcts_account_send_completion(pkt_entry, txe);

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief Build the long CTS REQ, filled with the start of the message.
 *
 * The rest of the message goes out as CTSDATA packets after the receiver's CTS.
 * A read NACK REQ carries no data.
 */
int efa_rdm_proto_longcts_construct_req_pke(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *txe,
					    uint64_t *pke_send_flags)
{
	int ret, iface;
	size_t hdr_size, rtm_payload_size, memory_alignment;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;

	/* The receiver answers the REQ with a CTS, so ring the doorbell now. */
	*pke_send_flags = 0;

	/* This protocol does not support injected sends. */
	assert(!(txe->fi_flags & FI_INJECT));

	if (efa_rdm_proto_get_dc(txe, &efa_rdm_proto_longcts))
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	pkt_entry = efa_rdm_proto_tx_pke_init_common(txe, txe->peer);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;

	pkt_entry->handle_pke = &efa_rdm_proto_longcts_handle_req_send_completion;

	/* The DC and non-DC long CTS headers have the same layout. */
	rtm_hdr = efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->credit_request = efa_env.tx_min_credits;

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	if (txe->internal_flags & EFA_RDM_OPE_READ_NACK) {
		/* A read NACK REQ continues a transfer and carries no data. */
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_READ_NACK;
		rtm_payload_size = 0;
	} else {
		/* Fill the packet with as much aligned payload as fits. */
		iface = txe->desc[0] ? ((struct efa_mr *) txe->desc[0])->iface :
				       FI_HMEM_SYSTEM;
		memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);
		rtm_payload_size = (ep->mtu_size - hdr_size) &
				   ~(memory_alignment - 1);
		assert(rtm_payload_size > 0);
		/* The REQ holds only part of the message. */
		assert(rtm_payload_size < txe->total_len);
	}

	ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, hdr_size, 0,
						rtm_payload_size);
	if (ret) {
		efa_rdm_pke_release_tx(pkt_entry);
		return ret;
	}

	EFA_DBG(FI_LOG_EP_DATA,
		"longcts protocol: posting 1 REQ pke, payload_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		pkt_entry->payload_size, txe->total_len, txe->msg_id);

	return FI_SUCCESS;
}

/* CTSDATA, shared by the long CTS protocols */

/**
 * @brief Handle the send completion of a CTSDATA packet.
 *
 * A delivery complete txe is released once its sends have all completed and
 * the peer's RECEIPT has arrived; otherwise the ope completes once all its data
 * has been acked and none of its packets are in flight or queued.
 */
ssize_t efa_rdm_proto_ctsdata_handle_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *ope = pkt_entry->ope;

	assert(ope);

	if (ope->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING) {
		efa_rdm_txe_progress_peer_abort_if_drained(ope);
	} else if (pkt_entry->flags & EFA_RDM_PKE_DC_LONGCTS_DATA) {
		if (efa_rdm_txe_with_remote_ack_ready_for_release(ope))
			efa_rdm_txe_release(ope);
	} else {
		ope->bytes_acked +=
			efa_rdm_pke_get_ctsdata_hdr(pkt_entry)->seg_length;
		if (efa_rdm_ope_send_done(ope))
			efa_rdm_ope_handle_send_completed(ope);
	}

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief Send the next window of an ope's data as CTSDATA packets.
 *
 * The window is split into full packets starting at bytes_sent, limited by the
 * TX packets available, and posted in one batch.
 */
ssize_t efa_rdm_proto_post_ctsdata(struct efa_rdm_ope *ope)
{
	struct efa_rdm_ep *ep = ope->ep;
	struct efa_rdm_pke *pkt_entry;
	size_t i, pkt_entry_cnt, pkt_entry_cnt_allocated = 0;
	size_t data_size, remaining;
	uint64_t segment_offset;
	ssize_t err;

	assert(ope->window > 0);

	pkt_entry_cnt = efa_rdm_ep_get_available_tx_pkts(ep);
	if (pkt_entry_cnt == 0)
		return -FI_EAGAIN;

	pkt_entry_cnt = MIN(pkt_entry_cnt,
			    (ope->window - 1) / ep->max_data_payload_size + 1);
	assert(pkt_entry_cnt <= efa_base_ep_get_tx_pool_size(&ep->base_ep));

	segment_offset = ope->bytes_sent;
	remaining = ope->window;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
					      EFA_RDM_PKE_FROM_EFA_TX_POOL);
		if (OFI_UNLIKELY(!pkt_entry)) {
			err = -FI_EAGAIN;
			goto release;
		}
		ep->send_pkt_entry_vec[i] = pkt_entry;
		pkt_entry_cnt_allocated++;

		data_size = MIN(remaining, ep->max_data_payload_size);
		err = efa_rdm_pke_init_ctsdata(pkt_entry, ope, segment_offset,
					       data_size);
		if (OFI_UNLIKELY(err))
			goto release;

		pkt_entry->handle_pke =
			&efa_rdm_proto_ctsdata_handle_send_completion;
		segment_offset += data_size;
		remaining -= data_size;
	}
	ep->send_pkt_entry_vec_size = pkt_entry_cnt;

	/* The window only shrinks once the packets are posted, so never batch. */
	err = efa_rdm_pke_sendv(ep->send_pkt_entry_vec, pkt_entry_cnt, 0);
	if (OFI_UNLIKELY(err))
		goto release;

	for (i = 0; i < pkt_entry_cnt; ++i) {
		data_size = efa_rdm_pke_get_ctsdata_hdr(
				    ep->send_pkt_entry_vec[i])->seg_length;
		ope->bytes_sent += data_size;
		ope->window -= data_size;
	}
	assert(ope->window >= 0);

	return 0;

release:
	for (i = 0; i < pkt_entry_cnt_allocated; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	return err;
}
