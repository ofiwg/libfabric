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

EFA_RDM_PROTO_DEF(longcts,
	.wants_mr = true,
	.can_use_protocol = &efa_rdm_proto_longcts_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longcts_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGCTS_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_LONGCTS_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longcts_handle_tx_pkes_posted,
	.handle_unexp_pke_match = &efa_rdm_proto_longcts_handle_matched_rtm,
);

/**
 * @brief Is this operation past its REQ and streaming CTSDATA packets?
 *
 * The receiver's first CTS moves the txe to EFA_RDM_OPE_SEND and puts it on
 * ep->ope_longcts_send_list, which is where every later packet of the message is
 * posted from. Before that the only packet the protocol may send is the REQ.
 */
static inline bool efa_rdm_proto_longcts_sending_ctsdata(struct efa_rdm_ope *txe)
{
	return txe->state == EFA_RDM_OPE_SEND;
}

/**
 * @brief Account for the long CTS packets that just reached the device.
 *
 * The REQ carries the head of the message and the CTSDATA packets carry the rest,
 * one receiver-granted window at a time, so this hook runs once per burst and has
 * to advance bytes_sent for whichever kind it just posted.
 *
 * For the REQ it is an assignment rather than an accumulation on purpose: a txe
 * queued before the handshake is reposted through the same protocol entry points,
 * and an accumulating write would double count. That cannot collide with the
 * CTSDATA accounting, which only starts once a CTS has arrived for a REQ that did
 * reach the device.
 *
 * For CTSDATA it has to accumulate, and may: a burst that fails to post is not
 * accounted at all, because the construct and the post both finish before this
 * runs. Closing the window by exactly what was posted is what paces the stream --
 * efa_rdm_ep_progress_peers_and_queues() keeps calling back while the window is
 * open, and the receiver opens the next one with another CTS.
 */
void efa_rdm_proto_longcts_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
	size_t i, payload_size;

	if (efa_rdm_proto_longcts_sending_ctsdata(txe)) {
		for (i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
			payload_size = ep->send_pkt_entry_vec[i]->payload_size;
			assert(payload_size > 0);
			txe->bytes_sent += payload_size;
			txe->window -= payload_size;
		}
		assert(txe->window >= 0);
		assert(txe->bytes_sent <= txe->total_len);
		return;
	}

	assert(ep->send_pkt_entry_vec_size == 1);

	/*
	 * A read NACK continuation's REQ carries no data, and txe->bytes_sent
	 * already covers what the read protocol's REQ packets delivered
	 * (bytes_runt for runt read, zero for long read). Leave it alone: the
	 * CTSDATA stream picks up from exactly there.
	 */
	if (txe->internal_flags & EFA_RDM_OPE_READ_NACK)
		assert(ep->send_pkt_entry_vec[0]->payload_size == 0);
	else
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
 *
 * With EFA_RDM_OPE_READ_NACK set on the txe this instead builds the REQ that
 * continues a read protocol whose receiver could not register its buffer. That
 * REQ carries no data at all: the read protocol's REQ packets already delivered
 * txe->bytes_sent bytes and the long CTS header has no segment offset field to
 * describe where a payload would belong, so everything still owed goes out as
 * CTSDATA packets. See #efa_rdm_msg_post_read_nack_rtm_proto.
 */
static int efa_rdm_proto_longcts_construct_req_pke(struct efa_rdm_ep *ep,
						   struct efa_rdm_ope *txe)
{
	int ret, iface;
	size_t hdr_size, rtm_payload_size, memory_alignment;
	bool read_nack;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;

	read_nack = txe->internal_flags & EFA_RDM_OPE_READ_NACK;

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
	 *
	 * A read NACK continuation is the one case where the recorded type is not
	 * the right one, so it derives its own and records that instead. The type
	 * still names the read protocol's REQ, because that is what protocol
	 * selection ran for, and efa_rdm_proto_req_pkt_type() cannot be re-run
	 * either: it declines the delivery complete variant for a peer in
	 * zero-copy receive mode, which is right for a fresh send (a headerless
	 * REQ has nowhere to put the send_id) but wrong here, where the REQ is
	 * always headered and must keep whatever delivery semantics the original
	 * send asked for. Leaving the stale read protocol type on the txe would
	 * name a transfer that is no longer on the wire.
	 */
	if (read_nack)
		txe->req_pkt_type =
			((txe->fi_flags & FI_DELIVERY_COMPLETE) ?
				 efa_rdm_proto_longcts.req_pkt_type_dc :
				 efa_rdm_proto_longcts.req_pkt_type) +
			efa_rdm_proto_get_tagged(txe);

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
	 * Tell the receiver this REQ continues a transfer it already has an rxe
	 * for. Without it the receiver would allocate a second rxe for this
	 * msg_id and slide its receive window again, since the read protocol's
	 * RTM already consumed this msg_id. See efa_rdm_pke_proc_msgrtm().
	 */
	if (read_nack)
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_READ_NACK;

	/*
	 * The header size is only final once efa_rdm_pke_init_req_hdr_common()
	 * has set the optional header flags, so compute the payload size from it
	 * rather than from the packet type.
	 */
	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	if (read_nack) {
		/*
		 * A continuation REQ carries no data: the read protocol's REQ
		 * packets already delivered txe->bytes_sent bytes and this header
		 * has no segment offset field to describe a payload at that
		 * offset, so the CTSDATA packets carry everything still owed.
		 */
		rtm_payload_size = 0;
	} else {
		iface = txe->desc[0] ? ((struct efa_mr *) txe->desc[0])->iface :
				       FI_HMEM_SYSTEM;
		memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);
		rtm_payload_size = (ep->mtu_size - hdr_size) &
				   ~(memory_alignment - 1);
		assert(rtm_payload_size > 0);
		/*
		 * Every protocol that can carry a whole message in REQ packets is
		 * tried before this one, so the REQ can only ever hold part of the
		 * message. efa_rdm_proto_longcts_handle_tx_pkes_posted() and the
		 * send completion callback both rely on that.
		 */
		assert(rtm_payload_size < txe->total_len);
	}

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
		"longcts protocol%s: posting 1 REQ pke, payload_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		read_nack ? " (read NACK continuation)" : "",
		pkt_entry->payload_size, txe->total_len, txe->msg_id);

	return FI_SUCCESS;
}

/**
 * @brief Build CTSDATA packets for as much of the open window as can be posted.
 *
 * The receiver's CTS grants txe->window bytes starting at txe->bytes_sent. Fill
 * that window with full sized data packets, bounded by the TX packets the
 * endpoint can supply right now: a short burst is not an error, because
 * efa_rdm_ep_progress_peers_and_queues() calls back while the window is still
 * open.
 */
static int efa_rdm_proto_longcts_construct_ctsdata_pkes(struct efa_rdm_ep *ep,
							struct efa_rdm_ope *txe)
{
	size_t i, pkt_entry_cnt, pkt_entry_cnt_allocated = 0;
	size_t max_pkt_entry_cnt, segment_offset, remainder;
	size_t *pkt_entry_data_size_vec = ep->send_pkt_entry_vec_data_sizes;
	struct efa_rdm_pke *pkt_entry;
	int ret;

	assert(txe->window > 0);
	assert(txe->bytes_sent < txe->total_len);
	assert(ep->efa_max_outstanding_tx_ops >=
	       ep->efa_outstanding_tx_ops + ep->efa_rnr_queued_pkt_cnt);

	max_pkt_entry_cnt = MIN(efa_rdm_ep_get_available_tx_pkts(ep),
				efa_base_ep_get_tx_pool_size(&ep->base_ep));
	if (max_pkt_entry_cnt == 0)
		return -FI_EAGAIN;

	pkt_entry_cnt = (txe->window - 1) / ep->max_data_payload_size + 1;
	pkt_entry_cnt = MIN(pkt_entry_cnt, max_pkt_entry_cnt);

	for (i = 0; i + 1 < pkt_entry_cnt; ++i)
		pkt_entry_data_size_vec[i] = ep->max_data_payload_size;

	/*
	 * Only the last packet of the window can be short, so clamp: a burst cut
	 * short by the TX packet supply leaves more than one packet's worth of
	 * window behind for the next one.
	 */
	remainder = txe->window - (pkt_entry_cnt - 1) * ep->max_data_payload_size;
	assert(remainder > 0);
	pkt_entry_data_size_vec[pkt_entry_cnt - 1] =
		MIN(remainder, ep->max_data_payload_size);

	segment_offset = txe->bytes_sent;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
					      EFA_RDM_PKE_FROM_EFA_TX_POOL);
		if (OFI_UNLIKELY(!pkt_entry)) {
			ret = -FI_EAGAIN;
			goto err_release_pkes;
		}

		ep->send_pkt_entry_vec[i] = pkt_entry;
		pkt_entry_cnt_allocated++;

		pkt_entry->handle_pke =
			&efa_rdm_proto_longcts_handle_ctsdata_send_completion;

		/*
		 * The CTSDATA header writer is shared with the emulated long CTS
		 * write and read protocols, which have not moved, so it stays
		 * where both can reach it.
		 */
		ret = efa_rdm_pke_init_ctsdata(pkt_entry, txe, segment_offset,
					       pkt_entry_data_size_vec[i]);
		if (ret)
			goto err_release_pkes;

		assert(pkt_entry->payload_size == pkt_entry_data_size_vec[i]);
		segment_offset += pkt_entry_data_size_vec[i];
	}

	assert(segment_offset <= txe->total_len);

	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_DBG(FI_LOG_EP_DATA,
		"longcts protocol: posting %zu CTSDATA pkes, bytes_sent %zu, window %" PRId64
		", total_len %zu, msg_id %" PRIu32 "\n",
		pkt_entry_cnt, txe->bytes_sent, txe->window, txe->total_len,
		txe->msg_id);

	return FI_SUCCESS;

err_release_pkes:
	for (i = 0; i < pkt_entry_cnt_allocated; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	return ret;
}

/**
 * @brief Construct TX packet entries for the long CTS protocol.
 *
 * A long CTS message goes out in two stages, so this function serves both: the
 * REQ that opens the CTS handshake, and then the CTSDATA bursts that carry the
 * rest of the message, one receiver-granted window at a time.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake -- or a CTSDATA burst or read NACK continuation REQ that hit
 * -FI_EAGAIN -- is reposted through this same function. Nothing here advances
 * bytes_sent or closes the window; that is
 * efa_rdm_proto_longcts_handle_tx_pkes_posted(), which only runs once the packets
 * have reached the device.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entries and
 * ep->send_pkt_entry_vec_size is the number of them.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longcts_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *txe,
					    uint64_t *pke_send_flags)
{
	/*
	 * Neither stage honors the caller's FI_MORE. The REQ is what opens the
	 * CTS handshake that carries the rest of the message, and a CTSDATA burst
	 * only closes its window once the packets are posted -- the progress
	 * engine keeps calling back until then -- so both ring the doorbell
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

	if (efa_rdm_proto_longcts_sending_ctsdata(txe))
		return efa_rdm_proto_longcts_construct_ctsdata_pkes(ep, txe);

	return efa_rdm_proto_longcts_construct_req_pke(ep, txe);
}
