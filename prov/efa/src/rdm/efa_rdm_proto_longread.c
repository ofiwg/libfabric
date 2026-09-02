/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longread.h"
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
 * EFA_RDM_LONGREAD_MSGRTM_PKT
 * EFA_RDM_LONGREAD_TAGRTM_PKT
 *
 * Sent by the receiver once it has read the whole message
 * EFA_RDM_EOR_PKT
 *
 * Sent by the receiver when it cannot register the receive buffer
 * EFA_RDM_READ_NACK_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#long-read-message-featuresubprotocol
 */

/**
 * @brief Check if the long read protocol can handle this send operation.
 *
 * The long read protocol sends no message data at all: one REQ packet carries
 * the addresses and keys of the source buffer and the receiver RDMA reads the
 * whole message. So it needs a message large enough to be worth a read, a source
 * buffer the device can read from directly, and a peer that supports RDMA read.
 *
 * Unlike the runt read protocol, a FI_DELIVERY_COMPLETE send is not ruled out:
 * long read is delivery complete by nature -- the sender's completion is driven
 * by the receiver's EOR -- which is why it has no separate DC REQ type and why
 * the legacy read based selector fell back to it for a DC send.
 */
static bool efa_rdm_proto_longread_can_use_for_send(struct efa_rdm_ope *txe,
						    struct efa_rdm_peer *peer,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	int i;

	/*
	 * The device reads the source buffer directly, so it must be reachable
	 * and registered. Every iov is checked, not just the first: the read iov
	 * array in the REQ header describes all of them, and a long read whose
	 * later iovs are unregistered would have to be abandoned in
	 * construct_tx_pkes() with -FI_ENOMR. Ruling it out here instead lets the
	 * selection fall through to long CTS, which needs no registration and is
	 * where mainline ends up anyway once its own registration attempt fails.
	 */
	if (!use_p2p)
		return false;

	for (i = 0; i < txe->iov_count; ++i)
		if (!txe->desc[i])
			return false;

	if (txe->total_len < g_efa_hmem_info[iface].min_read_msg_size)
		return false;

	return efa_rdm_interop_rdma_read(txe->ep, peer);
}

struct efa_rdm_proto efa_rdm_proto_longread = {
	.name = "longread",
	.can_use_protocol_for_send = &efa_rdm_proto_longread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longread_construct_tx_pkes,
	/*
	 * The long read protocol is delivery complete by nature, so it has no
	 * separate DC REQ type. The DC entries point at the non-DC types to keep
	 * the req_pkt_type_tagged == req_pkt_type + 1 relation that
	 * efa_rdm_proto_req_pkt_type() asserts.
	 */
	.req_pkt_type = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longread_handle_tx_pkes_posted,
};

/**
 * @brief Account for the long read RTM that just reached the device.
 *
 * The message now occupies one of the domain's read message slots, which limits
 * how many receivers are reading at once. The flag records the bump so exactly
 * one release site decrements it -- efa_rdm_txe_release_read_msg_slot(), reached
 * from whichever of the EOR, READ_NACK, PEER_ERROR or sender-abort paths
 * resolves the transfer -- and makes this idempotent should a txe queued before
 * the handshake be reposted through here.
 */
void efa_rdm_proto_longread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						  struct efa_rdm_ope *txe)
{
	if (txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED)
		return;

	ofi_atomic_inc64(&efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight);
	txe->internal_flags |= EFA_RDM_TXE_READ_MSG_COUNTED;
}

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
/**
 * @brief Handle send completion for a long read RTM packet.
 *
 * The RTM carries no message data, so this completion only says the read iov
 * array reached the peer; the transfer itself finishes when the receiver's EOR
 * arrives. The txe is therefore released by whichever of the two happens last,
 * and efa_rdm_pke_handle_eor_recv() makes the mirror image of this check.
 *
 * A peer-aborting txe (the application closed the source MR, or the receiver's
 * READ failed and it sent a PEER_ERROR_PKT) never gets an EOR, so the
 * ready-for-release check would stay false forever and the txe would leak. Fall
 * back to the peer-abort drain helper, which owns that txe's single completion
 * and release: it is a no-op until the last WR drains, then it emits the
 * PEER_ERROR_PKT that unblocks the peer's reorder window.
 */
void efa_rdm_proto_longread_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
		efa_rdm_txe_release(txe);
	else if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct TX packet entries for the long read protocol.
 *
 * Builds the single REQ packet the protocol sends. It carries no payload: the
 * body is the read iov array describing the source buffer, which is how the
 * receiver fetches the whole message.
 *
 * Writes to the txe are all idempotent, because a txe queued before the
 * handshake is reposted through this same function.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entry and
 * ep->send_pkt_entry_vec_size is 1.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longread_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_peer *peer,
					    const struct fi_msg *msg, uint32_t op,
					    uint64_t tag, uint64_t flags,
					    uint32_t internal_flags,
					    struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type;
	bool tagged;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;

	/*
	 * An injected send always fits in a single eager packet, and eager is
	 * tried first, so FI_INJECT cannot reach this protocol.
	 */
	assert(!(flags & FI_INJECT));

	tagged = (op == ofi_op_tagged);

	req_pkt_type = efa_rdm_proto_req_pkt_type(&efa_rdm_proto_longread, op,
						  flags, peer);
	/*
	 * There is no DC long read REQ: the protocol is delivery complete by
	 * nature, so the DC and non-DC entries both resolve to the long read
	 * types here.
	 */
	assert(req_pkt_type == (tagged ? EFA_RDM_LONGREAD_TAGRTM_PKT :
					 EFA_RDM_LONGREAD_MSGRTM_PKT));

	/*
	 * Record the wire protocol on the txe, next to the code that writes the
	 * matching header. The peer-abort (MR abort) protocol reads it back to
	 * tell a two-sided RTM from an operation it does not handle; leaving it
	 * unset silently disables abort notification for this send, which parks
	 * the peer's reorder window on this msg_id forever. See
	 * efa_rdm_txe_mark_peer_abort_if_needed().
	 */
	txe->protocol = req_pkt_type;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	ep->send_pkt_entry_vec[0] = pkt_entry;

	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;
	pkt_entry->handle_pke =
		&efa_rdm_proto_longread_handle_rtm_send_completion;

	efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

	rtm_hdr = efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
	rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->read_iov_count = txe->iov_count;

	if (tagged) {
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	/*
	 * The read iov array sits immediately after the REQ header and is the
	 * entire body of the packet -- there is no payload, so pkt_size covers
	 * the header plus the array and payload_size stays zero.
	 */
	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);
	ret = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (OFI_UNLIKELY(ret))
		goto err_release_pke;

	pkt_entry->pkt_size =
		hdr_size + txe->iov_count * sizeof(struct fi_rma_iov);

	ep->send_pkt_entry_vec_size = 1;
	EFA_DBG(FI_LOG_EP_DATA,
		"longread protocol: posting 1 pke, pkt_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		pkt_entry->pkt_size, txe->total_len, txe->msg_id);

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
