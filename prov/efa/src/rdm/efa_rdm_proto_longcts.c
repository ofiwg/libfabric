/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ope.h"
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
						   struct efa_rdm_peer *peer,
						   int req_pkt_type,
						   uint16_t header_flags,
						   int iface, bool use_p2p)
{
	return true;
}

struct efa_rdm_proto efa_rdm_proto_longcts = {
	.name = "longcts",
	.can_use_protocol_for_send = &efa_rdm_proto_longcts_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longcts_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGCTS_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_LONGCTS_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longcts_handle_tx_pkes_posted,
};

/**
 * @brief Account for the long CTS REQ that just reached the device.
 *
 * The REQ carries the head of the message; the rest goes out as CTSDATA packets
 * once the receiver's CTS opens a window. Those are still posted by
 * efa_rdm_ope_post_send() from ep->ope_longcts_send_list, which reads
 * txe->bytes_sent to find where to continue and advances it in
 * efa_rdm_pke_handle_ctsdata_sent(). So this hook owns bytes_sent only up to the
 * end of the REQ's segment, and the CTSDATA loop owns it from there.
 *
 * It is an assignment rather than an accumulation on purpose: a txe queued
 * before the handshake is reposted through the same protocol entry points, and
 * an accumulating write would double count. That cannot collide with the CTSDATA
 * loop, which only starts once a CTS has arrived for a REQ that did reach the
 * device.
 */
void efa_rdm_proto_longcts_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
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

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
/**
 * @brief Handle send completion for a non-DC long CTS RTM packet.
 *
 * One long CTS message is carried by the REQ plus a stream of CTSDATA packets
 * sharing one txe, so the operation is only complete once every one of them has
 * been acknowledged. efa_rdm_pke_handle_ctsdata_send_completion() makes the same
 * bytes_acked bookkeeping for the CTSDATA half of the stream.
 *
 * The peer-abort check is load bearing here, unlike in the single packet eager
 * protocol: an early packet of the message can fail (marking the txe
 * peer-aborting) while the rest are still in flight and go on to complete
 * successfully. An aborting txe's single completion and release are owned by the
 * peer-abort drain helper, so a successful completion on it is only a WR drain -
 * drive the helper (a no-op until the last WR drains, then it emits the
 * PEER_ERROR_PKT that unblocks the peer's reorder window) instead of the normal
 * completion path, whose efa_rdm_ope_handle_send_completed() asserts the flag is
 * clear.
 */
void efa_rdm_proto_longcts_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	/*
	 * A payload-free long CTS REQ only happens on the read NACK fallback: a
	 * read protocol whose receiver could not register its buffer continues as
	 * long CTS, and since the runt packets already delivered the head of the
	 * message and the long CTS header has no segment offset field, that REQ
	 * carries no data. It must not be accounted, and the txe may already be
	 * gone, because the CTSDATA packets that finish the message can complete
	 * first.
	 *
	 */
	if (pkt_entry->payload_size == 0) {
		assert(efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->flags &
		       EFA_RDM_REQ_READ_NACK);
		efa_rdm_pke_release_tx(pkt_entry);
		return;
	}

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
 * @brief Handle send completion for a DC long CTS RTM packet.
 *
 * A delivery complete transfer is only done when both every send completion has
 * arrived (efa_outstanding_tx_ops == 0, which is what the DC variants track
 * instead of bytes_acked) and the peer's RECEIPT has been received.
 *
 * The peer-abort check must come first: an aborting DC transfer never receives
 * its RECEIPT, so efa_rdm_txe_with_remote_ack_ready_for_release() would stay
 * false forever and the txe would leak.
 */
void efa_rdm_proto_longcts_handle_rtm_dc_send_completion(
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
 * @brief Construct TX packet entries for the long CTS protocol.
 *
 * Builds the single REQ packet the sender may send unsolicited. It carries as
 * much of the head of the message as fits, plus the credit request that tells
 * the receiver how large a window to open; the receiver answers with a CTS and
 * the rest of the message follows as CTSDATA packets, which are not this
 * function's business.
 *
 * With EFA_RDM_OPE_READ_NACK set on the txe this instead builds the REQ that
 * continues a read protocol whose receiver could not register its buffer. That
 * REQ carries no data at all: the read protocol's REQ packets already delivered
 * txe->bytes_sent bytes and the long CTS header has no segment offset field to
 * describe where a payload would belong, so everything still owed goes out as
 * CTSDATA packets. See #efa_rdm_msg_post_read_nack_rtm_proto.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake -- or a continuation REQ that hit -FI_EAGAIN -- is reposted through
 * this same function.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entry and
 * ep->send_pkt_entry_vec_size is 1.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longcts_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_peer *peer,
					    const struct fi_msg *msg, uint32_t op,
					    uint64_t tag, uint64_t flags,
					    uint32_t internal_flags,
					    struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type, iface;
	size_t hdr_size, rtm_payload_size, memory_alignment;
	bool tagged, delivery_complete_requested, read_nack;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;

	/*
	 * An injected send always fits in a single eager packet, and eager is
	 * tried first, so FI_INJECT cannot reach this protocol.
	 *
	 * A peer in zero-copy (headerless) receive mode is not ruled out: only
	 * the eager REQ has a headerless form, so a message too large for eager
	 * goes to such a peer with ordinary long CTS headers on the ordinary QP,
	 * which is what the legacy path does too.
	 */
	assert(!(flags & FI_INJECT));

	tagged = (op == ofi_op_tagged);
	read_nack = txe->internal_flags & EFA_RDM_OPE_READ_NACK;

	if (read_nack) {
		/*
		 * A read NACK continuation must keep whatever delivery semantics
		 * the original send asked for, so derive the REQ type from the
		 * operation's own flags. efa_rdm_proto_req_pkt_type() cannot be
		 * used: it also declines the delivery complete variant for a peer
		 * in zero-copy receive mode, which is right for a fresh send
		 * (a headerless REQ has nowhere to put the send_id) but wrong
		 * here, where the REQ is always headered.
		 */
		req_pkt_type = ((flags & FI_DELIVERY_COMPLETE) ?
					efa_rdm_proto_longcts.req_pkt_type_dc :
					efa_rdm_proto_longcts.req_pkt_type) +
			       tagged;
	} else {
		req_pkt_type = efa_rdm_proto_req_pkt_type(
			&efa_rdm_proto_longcts, op, flags, peer);
	}

	delivery_complete_requested =
		(req_pkt_type == efa_rdm_proto_longcts.req_pkt_type_dc ||
		 req_pkt_type == efa_rdm_proto_longcts.req_pkt_type_tagged_dc);

	/*
	 * Record the wire protocol on the txe, next to the code that writes the
	 * matching header. The peer-abort (MR abort) protocol reads it back to
	 * tell a two-sided RTM from an operation it does not handle; leaving it
	 * unset silently disables abort notification for this send, which parks
	 * the peer's reorder window on this msg_id forever. See
	 * efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	txe->protocol = req_pkt_type;

	if (delivery_complete_requested)
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	ep->send_pkt_entry_vec[0] = pkt_entry;

	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;
	pkt_entry->handle_pke = delivery_complete_requested ?
				       &efa_rdm_proto_longcts_handle_rtm_dc_send_completion :
				       &efa_rdm_proto_longcts_handle_rtm_send_completion;

	efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

	/*
	 * The DC and non-DC long CTS headers have the same layout -- the DC
	 * variant reuses the send_id the base header already carries -- so one
	 * accessor covers both.
	 */
	rtm_hdr = efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry);
	rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
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

	if (tagged) {
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

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
	if (ret)
		goto err_release_pke;

	ep->send_pkt_entry_vec_size = 1;
	EFA_DBG(FI_LOG_EP_DATA,
		"longcts protocol%s: posting 1 pke, payload_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		read_nack ? " (read NACK continuation)" : "",
		pkt_entry->payload_size, txe->total_len, txe->msg_id);

	return FI_SUCCESS;

err_release_pke:
	/*
	 * Release only the packet entry this function allocated. The txe was
	 * allocated by the caller, which releases it and rolls back
	 * peer->next_msg_id when this function fails.
	 */
	efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}
