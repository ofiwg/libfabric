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
 * @brief Check if the long CTS protocol can handle this send operation.
 *
 * Long CTS needs nothing from the peer beyond the baseline protocol and nothing
 * from the source buffer: the sender copies the data into its own packet buffers
 * and the receiver paces it with CTS packets. It is therefore always usable,
 * which is why it is the last entry in efa_rdm_protocols[] -- anything after it
 * would be dead.
 */
static bool efa_rdm_proto_longcts_can_use_for_send(struct efa_rdm_ope *txe,
						   int req_pkt_type,
						   uint16_t header_flags,
						   int iface, bool use_p2p)
{
	return true;
}

/**
 * @brief Send the next window of the message as CTSDATA packets.
 *
 * Called from the progress engine when the peer has granted more window. The
 * remaining payload is carried on the shared CTSDATA path, which keeps its own
 * bytes_sent and window accounting.
 *
 * @return 0 on success, negative errno on failure
 */
static ssize_t efa_rdm_proto_longcts_post_continuation(struct efa_rdm_ope *ope)
{
	return efa_rdm_ope_post_send(ope, EFA_RDM_CTSDATA_PKT);
}

EFA_RDM_PROTO_DEF(longcts,
	.wants_mr = true,
	.can_use_protocol = &efa_rdm_proto_longcts_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longcts_construct_tx_pkes,
	.post_continuation = &efa_rdm_proto_longcts_post_continuation,
	.req_pkt_type = EFA_RDM_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGCTS_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_LONGCTS_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longcts_handle_tx_pkes_posted,
	.handle_unexp_pke_match = &efa_rdm_proto_longcts_handle_matched_rtm,
);

/**
 * @brief Account for the long CTS REQ that just reached the device.
 *
 * The REQ carries the head of the message; the receiver answers with a CTS and
 * the rest follows as CTSDATA packets on the shared post-send path, which keeps
 * its own accounting. This hook therefore only runs for the REQ.
 *
 * bytes_sent is assigned rather than accumulated on purpose: a txe queued before
 * the handshake is reposted through the same protocol entry points, and an
 * accumulating write would double count. That cannot collide with the CTSDATA
 * accounting, which only starts once a CTS has arrived for a REQ that did reach
 * the device.
 */
void efa_rdm_proto_longcts_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
	assert(ep->send_pkt_entry_vec_size == 1);

	txe->bytes_sent = ep->send_pkt_entry_vec[0]->payload_size;
	assert(txe->bytes_sent < txe->total_len);

	/*
	 * Try to register the source buffer again. The first attempt was made in
	 * efa_rdm_proto_select_send_protocol(); it can have failed because the
	 * device's memory registration limit was reached, and a later attempt may
	 * succeed. It is worth retrying because the CTSDATA packets that carry
	 * the rest of the message can then be sent from the user buffer instead
	 * of through a bounce copy.
	 */
	if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
		efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
}

/* TX path callbacks - one callback per packet kind this protocol posts, each of
 * which reads the txe to tell the transmit complete and delivery complete
 * variants apart
 */
/**
 * @brief Finish accounting for a long CTS packet whose send completed.
 *
 * One long CTS message is carried by the REQ plus a stream of CTSDATA packets
 * sharing one txe, so the operation is only complete once every one of them has
 * been accounted for: for a transmit complete send once bytes_acked reaches the
 * message length, and for a delivery complete send once every send completion has
 * arrived (efa_outstanding_tx_ops == 0, which is what
 * efa_rdm_txe_with_remote_ack_ready_for_release() tracks instead of bytes_acked)
 * and the peer's RECEIPT has been received.
 *
 * The peer-abort check is load bearing here, unlike in the single packet eager
 * protocol: an early packet of the message can fail (marking the txe
 * peer-aborting) while the rest are still in flight and go on to complete
 * successfully. An aborting txe's single completion and release are owned by the
 * peer-abort drain helper, so a successful completion on it is only a WR drain -
 * drive the helper (a no-op until the last WR drains, then it emits the
 * PEER_ERROR_PKT that unblocks the peer's reorder window) instead of the normal
 * completion path, whose efa_rdm_ope_handle_send_completed() asserts the flag is
 * clear. It has to come before the delivery complete check too: an aborting DC
 * transfer never receives its RECEIPT, so
 * efa_rdm_txe_with_remote_ack_ready_for_release() would stay false forever and
 * the txe would leak.
 */
static void efa_rdm_proto_longcts_account_send_completion(
	struct efa_rdm_pke *pkt_entry, struct efa_rdm_ope *txe)
{
	bool delivery_complete_requested =
		txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	/*
	 * A delivery complete transfer counts outstanding TX ops rather than
	 * bytes, so leave bytes_acked alone for it.
	 */
	if (!delivery_complete_requested)
		txe->bytes_acked += pkt_entry->payload_size;

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (delivery_complete_requested) {
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else if (txe->total_len == txe->bytes_acked) {
		efa_rdm_ope_handle_send_completed(txe);
	}
}

/**
 * @brief Handle send completion for a long CTS RTM packet.
 */
ssize_t efa_rdm_proto_longcts_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;
	int pkt_type;

	/*
	 * A payload-free long CTS REQ only happens on the read NACK fallback: a
	 * read protocol whose receiver could not register its buffer continues as
	 * long CTS, and since the runt packets already delivered the head of the
	 * message and the long CTS header has no segment offset field, that REQ
	 * carries no data.
	 *
	 * A transmit complete transfer must not account it, and its txe may
	 * already be gone, because the CTSDATA packets that finish the message can
	 * complete first -- so return without touching the txe. A delivery
	 * complete transfer cannot release its txe until every one of its send
	 * completions has arrived, this one included, so its payload-free REQ has
	 * to fall through to the release check below instead. Telling the two
	 * apart therefore cannot go through the txe, so read the REQ type out of
	 * the packet's own header, which this branch already reads for the assert
	 * and which construct_tx_pkes() wrote from the same value it set
	 * EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED from.
	 *
	 * That fallback still posts through the legacy path
	 * (efa_rdm_pke_handle_read_nack_recv), so nothing reaches this callback
	 * with a zero payload yet; the check is here for when it is ported.
	 */
	if (pkt_entry->payload_size == 0) {
		assert(efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->flags &
		       EFA_RDM_REQ_READ_NACK);
		pkt_type = efa_rdm_pkt_type_of(pkt_entry);
		if (pkt_type != efa_rdm_proto_longcts.req_pkt_type_dc &&
		    pkt_type != efa_rdm_proto_longcts.req_pkt_type_tagged_dc) {
			efa_rdm_pke_release_tx(pkt_entry);
			return 0;
		}
	}

	txe = pkt_entry->ope;
	assert(txe);

	efa_rdm_proto_longcts_account_send_completion(pkt_entry, txe);

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief Handle send completion for a long CTS CTSDATA packet.
 *
 * Shares the accounting with the REQ: both carry part of the same message on the
 * same txe, so whichever completes last completes the operation.
 */
ssize_t efa_rdm_proto_longcts_handle_ctsdata_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe = pkt_entry->ope;

	assert(txe);
	assert(txe->type == EFA_RDM_TXE);
	assert(pkt_entry->payload_size > 0);

	efa_rdm_proto_longcts_account_send_completion(pkt_entry, txe);

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief Build the single REQ packet the sender may send unsolicited.
 *
 * It carries as much of the head of the message as fits, plus the credit request
 * that tells the receiver how large a window to open; the receiver answers with a
 * CTS and the rest of the message follows as CTSDATA packets.
 */
static int efa_rdm_proto_longcts_construct_req_pke(struct efa_rdm_ep *ep,
						   struct efa_rdm_ope *txe)
{
	int ret, iface;
	size_t hdr_size, rtm_payload_size, memory_alignment;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;

	/*
	 * Protocol selection recorded the REQ packet type its predicate was
	 * evaluated against, and efa_rdm_proto_tx_pke_init_common() writes the
	 * header for exactly that type instead of deriving it a second time.
	 * Reading the recorded type is what keeps the header from drifting away
	 * from the type the message was sized against.
	 *
	 * The field is also what the peer-abort (MR abort) protocol reads back to
	 * tell a two-sided RTM from an operation it does not handle, so a send
	 * that reached this function with it unset would silently lose abort
	 * notification and park the peer's reorder window on this msg_id forever.
	 * See efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	if (efa_rdm_proto_get_dc(txe, &efa_rdm_proto_longcts))
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	pkt_entry = efa_rdm_proto_tx_pke_init_common(txe, txe->peer);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	ep->send_pkt_entry_vec[0] = pkt_entry;

	pkt_entry->handle_pke = &efa_rdm_proto_longcts_handle_rtm_send_completion;

	/*
	 * The DC and non-DC long CTS headers have the same layout -- the DC
	 * variant reuses the send_id the base header already carries -- so one
	 * accessor covers both.
	 */
	rtm_hdr = efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->credit_request = efa_env.tx_min_credits;

	/*
	 * The header size is only final once efa_rdm_pke_init_req_hdr_common()
	 * has set the optional header flags, so compute the payload size from it
	 * rather than from the packet type.
	 */
	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	iface = txe->desc[0] ? ((struct efa_mr *) txe->desc[0])->iface :
			       FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);
	rtm_payload_size = (ep->mtu_size - hdr_size) & ~(memory_alignment - 1);
	assert(rtm_payload_size > 0);
	/*
	 * Every protocol that can carry a whole message in REQ packets is tried
	 * before this one, so the REQ can only ever hold part of the message.
	 * efa_rdm_proto_longcts_handle_tx_pkes_posted() and the send completion
	 * callback both rely on that.
	 */
	assert(rtm_payload_size < txe->total_len);

	ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, hdr_size, 0,
						rtm_payload_size);
	if (ret) {
		/*
		 * Release only the packet entry this function allocated. The txe
		 * was allocated by the caller, which releases it and rolls back
		 * peer->next_msg_id when this function fails.
		 */
		efa_rdm_pke_release_tx(pkt_entry);
		return ret;
	}

	ep->send_pkt_entry_vec_size = 1;
	EFA_DBG(FI_LOG_EP_DATA,
		"longcts protocol: posting 1 REQ pke, payload_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		pkt_entry->payload_size, txe->total_len, txe->msg_id);

	return FI_SUCCESS;
}

/**
 * @brief Construct TX packet entries for the long CTS protocol.
 *
 * Builds the single REQ packet the sender may send unsolicited: it carries the
 * head of the message plus the credit request that opens the CTS handshake. The
 * rest of the message follows later as CTSDATA packets, posted from
 * efa_rdm_proto_longcts_post_continuation() when the peer grants a window.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake is reposted through this same function. Nothing here advances
 * bytes_sent; that is efa_rdm_proto_longcts_handle_tx_pkes_posted(), which only
 * runs once the REQ has reached the device.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entry and
 * ep->send_pkt_entry_vec_size is 1.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longcts_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *txe,
					    uint64_t *pke_send_flags)
{
	/*
	 * The REQ is what opens the CTS handshake that carries the rest of the
	 * message, so it does not honor the caller's FI_MORE: ring the doorbell
	 * immediately.
	 */
	*pke_send_flags = 0;

	/*
	 * An injected send always fits in a single eager packet, and eager is
	 * tried first, so FI_INJECT cannot reach this protocol.
	 *
	 * A peer in zero-copy (headerless) receive mode is not ruled out: only
	 * the eager REQ has a headerless form, so a message too large for eager
	 * goes to such a peer with ordinary long CTS headers on the ordinary QP,
	 * which is what the legacy path does too.
	 */
	assert(!(txe->fi_flags & FI_INJECT));

	return efa_rdm_proto_longcts_construct_req_pke(ep, txe);
}
