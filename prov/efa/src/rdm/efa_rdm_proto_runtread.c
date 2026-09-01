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
static bool efa_rdm_proto_runtread_can_use_for_send(struct efa_rdm_ope *txe,
						    struct efa_rdm_peer *peer,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	int i;

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

	if (txe->total_len < g_efa_hmem_info[iface].min_read_msg_size)
		return false;

	if (!efa_rdm_interop_rdma_read(txe->ep, peer))
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
	return efa_rdm_peer_get_runt_size(peer, txe->ep, txe) > 0;
}

struct efa_rdm_proto efa_rdm_proto_runtread = {
	.name = "runtread",
	.can_use_protocol_for_send = &efa_rdm_proto_runtread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_runtread_construct_tx_pkes,
	/*
	 * The runt read REQ has no delivery complete variant, so the DC entries
	 * point at the non-DC types to keep the
	 * req_pkt_type_tagged == req_pkt_type + 1 relation that
	 * efa_rdm_proto_req_pkt_type() asserts. can_use_protocol_for_send()
	 * rules a DC send out before those entries can be selected.
	 */
	.req_pkt_type = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_runtread_handle_tx_pkes_posted,
};

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

/* TX path callbacks - one callback for each packet type that this protocol uses
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
void efa_rdm_proto_runtread_handle_rtm_send_completion(
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
	else if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
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
					     struct efa_rdm_peer *peer,
					     const struct fi_msg *msg,
					     uint32_t op, uint64_t tag,
					     uint64_t flags,
					     uint32_t internal_flags,
					     struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type;
	size_t i, pkt_entry_cnt = 0, pkt_entry_cnt_allocated = 0;
	size_t segment_offset, hdr_size, read_iov_size;
	size_t *pkt_entry_data_size_vec = ep->send_pkt_entry_vec_data_sizes;
	bool tagged;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;

	/*
	 * An injected send always fits in a single eager packet, and eager is
	 * tried first, so FI_INJECT cannot reach this protocol.
	 */
	assert(!(flags & FI_INJECT));

	tagged = (op == ofi_op_tagged);

	req_pkt_type = efa_rdm_proto_req_pkt_type(&efa_rdm_proto_runtread, op,
						  flags, peer);
	/*
	 * There is no DC runt read REQ, and can_use_protocol_for_send() rules a
	 * DC send out, so the DC and non-DC entries can only resolve to the
	 * runt read types here.
	 */
	assert(efa_rdm_pkt_type_is_runtread(req_pkt_type));

	/*
	 * Record the wire protocol on the txe, next to the code that writes the
	 * matching headers. The peer-abort (MR abort) protocol reads it back to
	 * tell a two-sided RTM from an operation it does not handle; leaving it
	 * unset silently disables abort notification for this send, which parks
	 * the peer's reorder window on this msg_id forever. See
	 * efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	txe->protocol = req_pkt_type;

	/*
	 * Set txe->bytes_runt and decide how many packets to send and how much
	 * data each carries. Both are idempotent -- the runt size is only
	 * computed when it is still zero -- so it is safe to re-run on a repost.
	 * Returns -FI_EAGAIN when the TX packet pool cannot supply that many
	 * packets right now.
	 */
	ret = efa_rdm_ope_prepare_to_post_send(txe, req_pkt_type,
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
		pkt_entry->handle_pke =
			&efa_rdm_proto_runtread_handle_rtm_send_completion;

		efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

		rtm_hdr = efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
		rtm_hdr->hdr.msg_id = txe->msg_id;
		rtm_hdr->msg_length = txe->total_len;
		rtm_hdr->send_id = txe->tx_id;
		rtm_hdr->seg_offset = segment_offset;
		rtm_hdr->runt_length = txe->bytes_runt;
		rtm_hdr->read_iov_count = txe->iov_count;

		if (tagged) {
			rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
			efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
		}

		/*
		 * The read iov array sits between the header and the payload;
		 * efa_rdm_txe_max_req_data_capacity() already reserved room for
		 * it, and the receiver finds the payload the same way, via
		 * efa_rdm_pke_get_payload_offset().
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
