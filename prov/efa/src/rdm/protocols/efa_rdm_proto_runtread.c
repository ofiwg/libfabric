/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_runtread.h"
#include "efa.h"
#include "efa_rdm_domain.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_tracepoint.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_RUNTREAD_MSGRTM_PKT
 * EFA_RDM_RUNTREAD_TAGRTM_PKT
 *
 * Sent by the receiver once it has read the rest of the message
 * EFA_RDM_EOR_PKT
 *
 * Sent by the receiver when it cannot register the receive buffer
 * EFA_RDM_READ_NACK_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#45-runting-read-message-subprotocol
 */

static ssize_t
efa_rdm_proto_runtread_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr =
		efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
	struct fi_rma_iov *read_iov = (struct fi_rma_iov *)
		(pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry));
	ssize_t ret;

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);
	rxe->bytes_runt = rtm_hdr->runt_length;
	if (rxe->total_len > rxe->bytes_runt &&
	    rxe->bytes_read_total_len == 0) {
		rxe->tx_id = rtm_hdr->send_id;
		rxe->rma_iov_count = rtm_hdr->read_iov_count;
		memcpy(rxe->rma_iov, read_iov,
		       rxe->rma_iov_count * sizeof (struct fi_rma_iov));
		efa_rdm_tracepoint(runtread_read_posted, rxe->msg_id,
				   (size_t) rxe->cq_entry.op_context,
				   rxe->total_len);

		ret = efa_rdm_pke_post_remote_read_or_nack(
			pkt_entry->ep, pkt_entry, rxe);
		if (ret) {
			efa_rdm_pke_release_rx_list(pkt_entry);
			return ret;
		}
	}

	return efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
}

/**
 * @brief Check if the runt read protocol can handle this send operation.
 *
 * The runt read protocol puts the first bytes_runt bytes of the message in REQ
 * packets and lets the receiver RDMA read the rest, so it needs a message large
 * enough to be worth a read, a source buffer the device can read from directly,
 * a peer that supports RDMA read, and a non-zero runt allowance for that peer.
 *
 * This is the conjunction of the two legacy predicates it replaces: the read
 * based arm of the old RTM selector and the runt arm of the old read based
 * selector, both of which this series deletes.
 */
static bool efa_rdm_proto_runtread_can_use_for_send(struct efa_rdm_proto *proto,
						    struct efa_rdm_ope *txe,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	int i;

	if (!efa_rdm_proto_msg_size_in_bounds(proto, iface, txe->total_len))
		return false;

	/*
	 * The device reads the source buffer directly, so it must be reachable
	 * and registered. Every iov is checked, not just the first: the read
	 * iov array in the REQ header describes all of them, and a runt read
	 * whose later iovs are unregistered would have to be abandoned in
	 * construct_tx_pkes() with -FI_ENOMR. Ruling it out here instead lets
	 * the selection fall through to a protocol that needs no registration.
	 */
	if (!use_p2p)
		return false;

	for (i = 0; i < txe->iov_count; ++i)
		if (!txe->desc[i])
			return false;

	if (!efa_rdm_interop_rdma_read(txe->ep, txe->peer))
		return false;

	/*
	 * There is no delivery complete variant of the runt read REQ, so a DC
	 * send has to use a read protocol that has one. The legacy read based
	 * selector made the same call, falling back to long read when
	 * FI_DELIVERY_COMPLETE is set.
	 */
	if (txe->fi_flags & FI_DELIVERY_COMPLETE)
		return false;

	/*
	 * Only one read based message may be in flight per domain at a time, so
	 * that the receiver's reads are not competing for device read bandwidth.
	 */
	if (ofi_atomic_get64(
		    &efa_rdm_ep_rdm_domain(txe->ep)->num_read_msg_in_flight))
		return false;

	/*
	 * The runt allowance is shared across the messages in flight to this
	 * peer. A zero return means this peer has none left, in which case
	 * there is nothing to runt and a pure read protocol should be used.
	 */
	return efa_rdm_peer_get_runt_size(txe->peer, txe->ep, txe) > 0;
}

static bool efa_rdm_proto_runtread_set_bounds(struct efa_rdm_proto *proto, enum fi_hmem_iface target_iface, enum fi_hmem_iface bounds_profile_iface){
	switch (bounds_profile_iface) {
	case FI_HMEM_SYSTEM:
		proto->min_size[target_iface] = EFA_DEFAULT_INTER_MIN_READ_MESSAGE_SIZE;
		return true;
	case FI_HMEM_SYNAPSEAI:
		proto->min_size[target_iface] = 1;
		return false;
	default:
		proto->min_size[target_iface] = efa_rdm_proto_max_eager_msg_size_with_largest_header() + 1;
		return true;
	}
}

EFA_RDM_PROTO_DEF(runtread,
	.wants_mr = true,
	.min_param_name = "inter_min_read_message_size",
	.set_bounds = &efa_rdm_proto_runtread_set_bounds,
	.can_use_protocol = &efa_rdm_proto_runtread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_runtread_construct_tx_pkes,
	/*
	 * The runt read REQ has no delivery complete variant, so the DC entries
	 * point at the non-DC types to keep the
	 * req_pkt_type_tagged == req_pkt_type + 1 relation that
	 * efa_rdm_proto_req_pkt_type() asserts. can_use_protocol() rules a DC
	 * send out before those entries can be selected.
	 */
	.req_pkt_type = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_runtread_handle_tx_pkes_posted,
	.handle_unexp_pke_match = &efa_rdm_proto_runtread_handle_matched_rtm,
);

/**
 * @brief Account for the runt read packets that just reached the device.
 *
 * The runt portion is posted in one go, so bytes_sent goes straight to
 * bytes_runt -- construct_tx_pkes() asserts that the segments add up to exactly
 * that. It is an assignment rather than an accumulation because a txe queued
 * before the handshake is reposted through the same entry points and an
 * accumulating write would double count.
 *
 * The other two counters are true accumulators and cannot be written that way,
 * but they do not need to be: this hook only runs once the whole packet vector
 * reached the device, and a repost only happens for an attempt that never got
 * there, so it cannot run twice for one message. Each is balanced elsewhere --
 * the runt bytes by the per-packet send completions below, and the read message
 * slot by efa_rdm_txe_release_read_msg_slot() via whichever of the EOR,
 * READ_NACK, PEER_ERROR or sender-abort paths resolves the transfer.
 */
void efa_rdm_proto_runtread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						  struct efa_rdm_ope *txe)
{
	assert(txe->bytes_runt);

	txe->bytes_sent = txe->bytes_runt;
	txe->peer->num_runt_bytes_in_flight += txe->bytes_runt;

	/*
	 * The receiver only posts a read when the runt did not cover the whole
	 * message, so only then does this message occupy a read slot. The flag
	 * records the bump so exactly one release site decrements it; it also
	 * makes this idempotent should the hook ever run twice.
	 */
	if (txe->total_len > txe->bytes_runt &&
	    !(txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED)) {
		ofi_atomic_inc64(
			&efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight);
		txe->internal_flags |= EFA_RDM_TXE_READ_MSG_COUNTED;
	}
}

/* TX path callbacks - one callback per protocol, which reads the txe to tell
 * the transmit complete and delivery complete variants apart
 */
/**
 * @brief Handle send completion for a runt read RTM packet.
 *
 * Returns this packet's share of the peer's runt allowance so the next message
 * to that peer can runt again, then completes the operation once every packet
 * has been acknowledged. A runt read normally completes when the receiver's EOR
 * arrives rather than here, since the runt is only the head of the message; the
 * bytes_acked check matters for the case where the runt covered all of it.
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
ssize_t efa_rdm_proto_runtread_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	size_t pkt_data_size;

	txe = pkt_entry->ope;
	assert(txe);

	pkt_data_size = pkt_entry->payload_size;
	txe->bytes_acked += pkt_data_size;

	peer = txe->peer;
	assert(peer);
	assert(peer->num_runt_bytes_in_flight >= pkt_data_size);
	peer->num_runt_bytes_in_flight -= pkt_data_size;

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (efa_rdm_ope_send_done(txe))
		efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief Set the number of bytes a runt read message sends in its REQ packets.
 *
 * Only the first call computes the size, so a send that is queued and reposted
 * carries the same runt on every attempt.
 */
static void efa_rdm_proto_runtread_set_runt_size(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
	assert(txe->type == EFA_RDM_TXE);

	if (txe->bytes_runt > 0)
		return;

	assert(txe->peer);
	txe->bytes_runt = efa_rdm_peer_get_runt_size(txe->peer, ep, txe);

	assert(txe->bytes_runt);
}

/**
 * @brief Set the runt size and decide how the REQ packets share it.
 *
 * Each packet holds the read iov array, then a piece of the runt.
 *
 * @param[out] pkt_entry_cnt		number of packets to send
 * @param[out] pkt_entry_data_size_vec	data size of each of those packets
 * @return 0 on success, -FI_EAGAIN when the TX packet pool does not have
 *	   enough free packets
 */
static ssize_t efa_rdm_proto_runtread_plan_tx_pkes(struct efa_rdm_ep *ep,
						   struct efa_rdm_ope *txe,
						   int req_pkt_type,
						   size_t *pkt_entry_cnt,
						   size_t *pkt_entry_data_size_vec)
{
	size_t hdr_size, max_data_size;

	assert(efa_rdm_pkt_type_is_runtread(req_pkt_type));
	assert(ep->efa_max_outstanding_tx_ops >=
	       ep->efa_outstanding_tx_ops + ep->efa_rnr_queued_pkt_cnt);

	/* The runt size is fixed on the first attempt that has a free packet. */
	if (efa_rdm_ep_get_available_tx_pkts(ep) == 0)
		return -FI_EAGAIN;

	efa_rdm_proto_runtread_set_runt_size(ep, txe);

	/* Leave room in every packet for the read iov array. */
	hdr_size = efa_rdm_pkt_type_get_req_hdr_size(
			   req_pkt_type,
			   efa_rdm_proto_req_header_flags(txe->peer, txe->fi_flags),
			   0 /* rma_iov_count */) +
		   txe->iov_count * sizeof(struct fi_rma_iov);
	max_data_size = efa_rdm_proto_max_req_data_capacity(ep, hdr_size);

	return efa_rdm_proto_split_req_data(ep, txe, txe->bytes_runt,
					    max_data_size, pkt_entry_cnt,
					    pkt_entry_data_size_vec);
}

/**
 * @brief Construct TX packet entries for the runt read protocol.
 *
 * Splits the runt portion of the message across as many REQ packets as it takes,
 * balancing the per-packet data sizes and honouring the interface's memory
 * alignment. Every packet also carries the read iov array describing the whole
 * source buffer, which is how the receiver fetches the remainder.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake is reposted through this same function.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entries and
 * ep->send_pkt_entry_vec_size is the number of them.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_runtread_construct_tx_pkes(struct efa_rdm_ep *ep,
					     struct efa_rdm_ope *txe,
					     uint64_t *pke_send_flags)
{
	int ret, req_pkt_type;
	size_t i, pkt_entry_cnt = 0, pkt_entry_cnt_allocated = 0;
	size_t segment_offset, hdr_size, read_iov_size;
	size_t *pkt_entry_data_size_vec = ep->send_pkt_entry_vec_data_sizes;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;

	/*
	 * The runt read protocol posts multiple packets, so it does not honor
	 * FI_MORE; ring the doorbell immediately.
	 */
	*pke_send_flags = 0;

	/*
	 * A valid injected send fits the eager protocol, so FI_INJECT cannot
	 * reach this protocol.
	 */
	assert(!(txe->fi_flags & FI_INJECT));

	/*
	 * Protocol selection recorded the REQ packet type its predicate was
	 * evaluated against, so write the header for exactly that type instead of
	 * deriving it a second time. Reading the recorded type is what keeps the
	 * header from drifting away from the type the message was sized against.
	 *
	 * The field is also what the peer-abort (MR abort) protocol reads back to
	 * tell a two-sided RTM from an operation it does not handle, so a send
	 * that reached this function with it unset would silently lose abort
	 * notification and park the peer's reorder window on this msg_id forever.
	 * See efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	req_pkt_type = txe->req_pkt_type;

	/*
	 * There is no DC runt read REQ, and can_use_protocol() rules a DC send
	 * out, so the DC and non-DC entries can only resolve to the runt read
	 * types here.
	 */
	assert(efa_rdm_pkt_type_is_runtread(req_pkt_type));

	ret = efa_rdm_proto_runtread_plan_tx_pkes(ep, txe, req_pkt_type,
						  &pkt_entry_cnt,
						  pkt_entry_data_size_vec);
	if (ret)
		return ret;

	assert(pkt_entry_cnt > 0);
	assert(pkt_entry_cnt <= efa_base_ep_get_tx_pool_size(&ep->base_ep));

	read_iov_size = txe->iov_count * sizeof(struct fi_rma_iov);

	segment_offset = 0;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		assert(pkt_entry_data_size_vec[i] > 0);

		pkt_entry = efa_rdm_proto_tx_pke_init_common(txe, txe->peer);
		if (OFI_UNLIKELY(!pkt_entry)) {
			ret = -FI_EAGAIN;
			goto err_release_pkes;
		}

		ep->send_pkt_entry_vec[i] = pkt_entry;
		pkt_entry_cnt_allocated++;

		pkt_entry->handle_pke =
			&efa_rdm_proto_runtread_handle_rtm_send_completion;

		rtm_hdr = efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
		rtm_hdr->msg_length = txe->total_len;
		rtm_hdr->send_id = txe->tx_id;
		rtm_hdr->seg_offset = segment_offset;
		rtm_hdr->runt_length = txe->bytes_runt;
		rtm_hdr->read_iov_count = txe->iov_count;

		/*
		 * The read iov array sits between the header and the payload; the
		 * receiver skips past it with efa_rdm_pke_get_payload_offset().
		 */
		hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
		read_iov = (struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);
		ret = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
		if (OFI_UNLIKELY(ret))
			goto err_release_pkes;

		ret = efa_rdm_pke_init_payload_from_ope(
			pkt_entry, txe, hdr_size + read_iov_size,
			segment_offset, pkt_entry_data_size_vec[i]);
		if (ret)
			goto err_release_pkes;

		assert(pkt_entry->payload_size == pkt_entry_data_size_vec[i]);
		segment_offset += pkt_entry_data_size_vec[i];
	}

	/* The REQ packets carry exactly the runt portion of the message */
	assert(segment_offset == txe->bytes_runt);

	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_DBG(FI_LOG_EP_DATA,
		"runtread protocol: posting %zu pkes, runt_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		pkt_entry_cnt, txe->bytes_runt, txe->total_len, txe->msg_id);

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
