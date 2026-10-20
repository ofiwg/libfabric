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
#include "efa_rdm_tracepoint.h"

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
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#long-read-message-subprotocol
 */

static ssize_t
efa_rdm_proto_longread_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr =
		efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
	struct fi_rma_iov *read_iov = (struct fi_rma_iov *)
		(pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry));
	ssize_t ret;

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);

	rxe->tx_id = rtm_hdr->send_id;
	rxe->rma_iov_count = rtm_hdr->read_iov_count;
	memcpy(rxe->rma_iov, read_iov,
	       rxe->rma_iov_count * sizeof (struct fi_rma_iov));

	efa_rdm_tracepoint(longread_read_posted, rxe->msg_id,
			   (size_t) rxe->cq_entry.op_context, rxe->total_len);

	ret = efa_rdm_pke_post_remote_read_or_nack(rxe->ep, pkt_entry, rxe);
	efa_rdm_pke_release_rx(pkt_entry);
	return ret;
}

/**
 * @brief Check if the long read protocol can be used for this send.
 *
 * The receiver RDMA reads the message straight from the source buffers, so they
 * must be reachable over p2p and registered, and the peer must support RDMA read.
 */
static bool efa_rdm_proto_longread_can_use_for_send(struct efa_rdm_ope *txe,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	int i;

	if (!use_p2p)
		return false;

	/* The REQ describes every iov to the receiver, so all must be registered. */
	for (i = 0; i < txe->iov_count; ++i)
		if (!txe->desc[i])
			return false;

	if (txe->total_len < g_efa_hmem_info[iface].min_read_msg_size)
		return false;

	return efa_rdm_interop_rdma_read(txe->ep, txe->peer);
}

EFA_RDM_PROTO_DEF(longread,
	.wants_mr = true,
	.can_use_protocol = &efa_rdm_proto_longread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longread_construct_tx_pkes,
	/* Long read has no DC variant, so the DC entries reuse the non-DC types. */
	.req_pkt_type = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longread_handle_tx_pkes_posted,
	.handle_unexp_pke_match = &efa_rdm_proto_longread_handle_matched_rtm,
);

/* TX path callbacks */

/**
 * @brief Count this message in the domain's read messages in flight.
 *
 * EFA_RDM_TXE_READ_MSG_COUNTED ensures it is counted only once per message.
 */
void efa_rdm_proto_longread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
	if (txe->internal_flags & EFA_RDM_TXE_READ_MSG_COUNTED)
		return;

	ofi_atomic_inc64(&efa_rdm_ep_rdm_domain(ep)->num_read_msg_in_flight);
	txe->internal_flags |= EFA_RDM_TXE_READ_MSG_COUNTED;
}

/**
 * @brief Handle send completion of the long read RTM.
 *
 * The user completion is written when the receiver's EOR arrives, not here. The
 * txe is released by whichever of this completion and the EOR comes last.
 */
ssize_t efa_rdm_proto_longread_handle_rtm_send_completion(
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
	return 0;
}

/**
 * @brief Build the single long read RTM packet.
 *
 * The packet holds the REQ header followed by the read iov array the receiver
 * uses to RDMA read the message.
 */
int efa_rdm_proto_longread_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *txe,
					    uint64_t *pke_send_flags)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;

	/* This one packet starts the transfer, so ring the doorbell right away. */
	*pke_send_flags = 0;

	/* This protocol does not support injected sends. */
	assert(!(txe->fi_flags & FI_INJECT));

	assert(txe->req_pkt_type == (efa_rdm_proto_get_tagged(txe) ?
					EFA_RDM_LONGREAD_TAGRTM_PKT :
					EFA_RDM_LONGREAD_MSGRTM_PKT));

	pkt_entry = efa_rdm_proto_tx_pke_init_common(txe, txe->peer);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;

	pkt_entry->handle_pke =
		&efa_rdm_proto_longread_handle_rtm_send_completion;

	rtm_hdr = efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->read_iov_count = txe->iov_count;

	/* The read iov array follows the REQ header and ends the packet. */
	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);
	ret = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (OFI_UNLIKELY(ret))
		goto err_release_pke;

	pkt_entry->pkt_size =
		hdr_size + txe->iov_count * sizeof(struct fi_rma_iov);

	EFA_DBG(FI_LOG_EP_DATA,
		"longread protocol: posting 1 pke, pkt_size %zu, total_len %zu, msg_id %" PRIu32
		"\n",
		pkt_entry->pkt_size, txe->total_len, txe->msg_id);

	return FI_SUCCESS;

err_release_pke:
	efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}
