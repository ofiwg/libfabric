/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_medium.h"
#include "efa.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_MEDIUM_MSGRTM_PKT
 * EFA_RDM_MEDIUM_TAGRTM_PKT
 * EFA_RDM_DC_MEDIUM_MSGRTM_PKT
 * EFA_RDM_DC_MEDIUM_TAGRTM_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#medium-message-featuresubprotocol
 */

/**
 * @brief Check if the medium protocol can handle this send operation.
 *
 * The medium protocol carries the whole message in REQ packets, so it is only
 * worth using up to the interface's medium threshold (64KB for system memory,
 * and 0 - i.e. never - for every HMEM interface, which use the read based
 * protocols instead).
 */
static bool efa_rdm_proto_medium_can_use_for_send(struct efa_rdm_ope *txe,
						  struct efa_rdm_peer *peer,
						  int req_pkt_type,
						  uint16_t header_flags,
						  int iface, bool use_p2p)
{
	/*
	 * A zero length message always fits in a single eager packet, and eager
	 * is tried first, so this cannot be reached today. Rule it out anyway:
	 * max_medium_msg_size is 0 where medium is unsupported, which a zero
	 * length message would otherwise satisfy, and the segmenting math below
	 * divides by the message size.
	 */
	if (!txe->total_len)
		return false;

	return txe->total_len <= g_efa_hmem_info[iface].max_medium_msg_size;
}

struct efa_rdm_proto efa_rdm_proto_medium = {
	.name = "medium",
	.can_use_protocol_for_send = &efa_rdm_proto_medium_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_medium_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_MEDIUM_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_MEDIUM_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_medium_handle_tx_pkes_posted,
};

/**
 * @brief Account for the medium packets that just reached the device.
 *
 * The medium protocol carries the whole message in its REQ packets and posts
 * all of them in one go, so bytes_sent goes straight to total_len -- the
 * construct step asserts that the segments add up to exactly that.
 *
 * It is an assignment rather than an accumulation on purpose: a txe queued
 * before the handshake is reposted through the same protocol entry points, and
 * an accumulating write would double count.
 */
void efa_rdm_proto_medium_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						struct efa_rdm_ope *txe)
{
	txe->bytes_sent = txe->total_len;
}

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
/**
 * @brief Handle send completion for a non-DC medium RTM packet.
 *
 * One medium message is carried by several packets sharing one txe, so the
 * operation is only complete once every packet has been acknowledged.
 *
 * The peer-abort check is load bearing here, unlike in the single packet eager
 * protocol: packet 3 of the message can fail (marking the txe peer-aborting)
 * while packets 4..n are still in flight and go on to complete successfully. An
 * aborting txe's single completion and release are owned by the peer-abort drain
 * helper, so a successful completion on it is only a WR drain - drive the helper
 * (a no-op until the last WR drains, then it emits the PEER_ERROR_PKT that
 * unblocks the peer's reorder window) instead of the normal completion path,
 * whose efa_rdm_ope_handle_send_completed() asserts the flag is clear.
 */
void efa_rdm_proto_medium_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	txe->bytes_acked += pkt_entry->payload_size;

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Handle send completion for a DC medium RTM packet.
 *
 * A delivery complete transfer is only done when both every send completion has
 * arrived (efa_outstanding_tx_ops == 0, which is what the DC variants track
 * instead of bytes_acked) and the peer's RECEIPT has been received.
 *
 * The peer-abort check must come first: an aborting DC transfer never receives
 * its RECEIPT, so efa_rdm_txe_with_remote_ack_ready_for_release() would stay
 * false forever and the txe would leak.
 */
void efa_rdm_proto_medium_handle_rtm_dc_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct TX packet entries for the medium protocol.
 *
 * Splits the message across as many REQ packets as it takes, balancing the
 * per-packet data sizes and honouring the interface's memory alignment, then
 * stamps each packet with its segment offset and the total message length.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake is reposted through this same function.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entries and
 * ep->send_pkt_entry_vec_size is the number of them.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_medium_construct_tx_pkes(struct efa_rdm_ep *ep,
					   struct efa_rdm_peer *peer,
					   const struct fi_msg *msg, uint32_t op,
					   uint64_t tag, uint64_t flags,
					   uint32_t internal_flags,
					   struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type;
	size_t i, pkt_entry_cnt = 0, pkt_entry_cnt_allocated = 0;
	size_t segment_offset;
	size_t *pkt_entry_data_size_vec = ep->send_pkt_entry_vec_data_sizes;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_medium_rtm_base_hdr *medium_rtm_hdr;
	struct efa_rdm_dc_medium_rtm_base_hdr *dc_medium_rtm_hdr;

	/*
	 * An injected send always fits in a single eager packet -- inject_size
	 * is the MTU minus the largest REQ header, which is never smaller than
	 * the eager REQ header -- and eager is tried first, so FI_INJECT cannot
	 * reach this protocol.
	 *
	 * A peer in zero-copy (headerless) receive mode is not ruled out: only
	 * the eager REQ has a headerless form, so a message too large for eager
	 * goes to such a peer with ordinary medium headers on the ordinary QP.
	 * That is what the legacy path does too -- efa_rdm_msg_select_rtm()
	 * picks the medium RTM and efa_rdm_pke_fill_data() only sets the
	 * headerless flags for EFA_RDM_EAGER_MSGRTM_PKT.
	 */
	assert(!(flags & FI_INJECT));

	tagged = (op == ofi_op_tagged);

	req_pkt_type = efa_rdm_proto_req_pkt_type(&efa_rdm_proto_medium, op,
						  flags, peer);
	delivery_complete_requested =
		(req_pkt_type == efa_rdm_proto_medium.req_pkt_type_dc ||
		 req_pkt_type == efa_rdm_proto_medium.req_pkt_type_tagged_dc);

	/*
	 * Record the wire protocol on the txe, next to the code that writes the
	 * matching headers. The peer-abort (MR abort) protocol reads it back to
	 * tell a two-sided RTM from an operation it does not handle; leaving it
	 * unset silently disables abort notification for this send, which parks
	 * the peer's reorder window on this msg_id forever. See
	 * efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	txe->protocol = req_pkt_type;

	if (delivery_complete_requested)
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	/*
	 * Decide how many packets to send and how much data each carries. This
	 * only reads the txe and writes endpoint scratch state, so it is safe to
	 * re-run on a repost. It returns -FI_EAGAIN when the TX packet pool
	 * cannot supply that many packets right now.
	 */
	ret = efa_rdm_ope_prepare_to_post_send(txe, req_pkt_type,
					       &pkt_entry_cnt,
					       pkt_entry_data_size_vec);
	if (ret)
		return ret;

	assert(pkt_entry_cnt > 0);
	assert(pkt_entry_cnt <= efa_base_ep_get_tx_pool_size(&ep->base_ep));

	segment_offset = 0;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		assert(pkt_entry_data_size_vec[i] > 0);

		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
					      EFA_RDM_PKE_FROM_EFA_TX_POOL);
		if (OFI_UNLIKELY(!pkt_entry)) {
			ret = -FI_EAGAIN;
			goto err_release_pkes;
		}

		ep->send_pkt_entry_vec[i] = pkt_entry;
		pkt_entry_cnt_allocated++;

		efa_rdm_pke_set_ope(pkt_entry, txe);
		pkt_entry->peer = peer;

		efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

		/* The DC and non-DC medium headers share this prefix. */
		rtm_hdr = efa_rdm_pke_get_rtm_base_hdr(pkt_entry);
		rtm_hdr->flags |= EFA_RDM_REQ_MSG;
		rtm_hdr->msg_id = txe->msg_id;

		if (tagged) {
			rtm_hdr->flags |= EFA_RDM_REQ_TAGGED;
			efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
		}

		/*
		 * msg_length and seg_offset live at different offsets in the two
		 * header layouts, because the DC header inserts send_id before
		 * them, so each variant must be written through its own type.
		 */
		if (delivery_complete_requested) {
			dc_medium_rtm_hdr =
				efa_rdm_pke_get_dc_medium_rtm_base_hdr(pkt_entry);
			dc_medium_rtm_hdr->send_id = txe->tx_id;
			dc_medium_rtm_hdr->msg_length = txe->total_len;
			dc_medium_rtm_hdr->seg_offset = segment_offset;
			pkt_entry->handle_pke =
				&efa_rdm_proto_medium_handle_rtm_dc_send_completion;
		} else {
			medium_rtm_hdr =
				efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry);
			medium_rtm_hdr->msg_length = txe->total_len;
			medium_rtm_hdr->seg_offset = segment_offset;
			pkt_entry->handle_pke =
				&efa_rdm_proto_medium_handle_rtm_send_completion;
		}

		ret = efa_rdm_pke_init_payload_from_ope(
			pkt_entry, txe, efa_rdm_pke_get_req_hdr_size(pkt_entry),
			segment_offset, pkt_entry_data_size_vec[i]);
		if (ret)
			goto err_release_pkes;

		assert(pkt_entry->payload_size == pkt_entry_data_size_vec[i]);
		segment_offset += pkt_entry_data_size_vec[i];
	}

	/* The medium protocol carries the whole message in its REQ packets */
	assert(segment_offset == txe->total_len);

	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_DBG(FI_LOG_EP_DATA,
		"medium protocol: posting %zu pkes, total_len %lu, msg_id %" PRIu32
		"\n",
		pkt_entry_cnt, txe->total_len, txe->msg_id);

	return FI_SUCCESS;

err_release_pkes:
	/*
	 * Release every packet entry this function allocated, and only those.
	 * The txe was allocated by the caller, which releases it and rolls back
	 * peer->next_msg_id when this function fails.
	 */
	for (i = 0; i < pkt_entry_cnt_allocated; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	return ret;
}
