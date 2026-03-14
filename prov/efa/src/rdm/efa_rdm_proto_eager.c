/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_eager.h"
#include "efa.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_EAGER_MSGRTM_PKT
 * EFA_RDM_EAGER_TAGRTM_PKT
 * EFA_RDM_DC_EAGER_MSGRTM_PKT
 * EFA_RDM_DC_EAGER_TAGRTM_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#eager-message-featuresubprotocol
 */

/**
 * @brief Check if the eager protocol can handle this send operation.
 *
 * Returns true if the message fits in a single MTU-sized packet after
 * accounting for the request header size.
 */
static bool efa_rdm_proto_eager_can_use_for_send(struct efa_rdm_ope *txe,
						 int req_pkt_type,
						 uint16_t header_flags,
						 int iface)
{
	size_t max_data_offset, max_rtm_data_capacity;

	/* TODO: For emulated read and atomics, need to consider RMA
	 * IOVs in the header
	 * https://github.com/ofiwg/libfabric/blob/cff899c9ef6dd823a1e3b35d3205622013c6eb6c/prov/efa/src/rdm/efa_rdm_pkt_type.c#L101-L103
	 */
	max_data_offset = efa_rdm_pkt_type_get_req_hdr_size(req_pkt_type,
							    header_flags, 0);
	max_rtm_data_capacity = txe->ep->mtu_size - max_data_offset;

	return txe->total_len <= max_rtm_data_capacity;
}

struct efa_rdm_proto efa_rdm_proto_eager = {
	.can_use_protocol_for_send = &efa_rdm_proto_eager_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_eager_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_EAGER_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_EAGER_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_EAGER_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_EAGER_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
/**
 * @brief Handle send completion for a non-DC eager RTM packet.
 *
 * Reports the CQ completion, releases the TXE, and releases the
 * TX packet entry.
 */
void efa_rdm_proto_eager_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);
	assert(txe->total_len == pkt_entry->payload_size);

	efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Handle send completion for a DC eager RTM packet.
 *
 * Only releases the TXE when both all send completions have arrived
 * (efa_outstanding_tx_ops == 0) and the receipt packet has been received.
 */
void efa_rdm_proto_eager_handle_rtm_dc_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);
	assert(txe->total_len == pkt_entry->payload_size);

	if (efa_rdm_txe_dc_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct TX packet entries for the eager protocol.
 *
 * Allocates a single TX packet entry, initializes the RTM header with
 * the message payload, and sets the per-packet send completion callback.
 * Supports both regular and delivery-complete (DC) eager packets.
 *
 * On success, ep->send_pkt_entry_vec[0] contains the packet entry and
 * ep->send_pkt_entry_vec_size is set to 1.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_eager_construct_tx_pkes(struct efa_rdm_ep *ep,
					  struct efa_rdm_peer *peer,
					  const struct fi_msg *msg, uint32_t op,
					  uint64_t tag, uint64_t flags,
					  struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type, pkt_entry_cnt;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_dc_eager_rtm_base_hdr *dc_base_hdr;

	efa_rdm_proto_txe_fill(txe, ep, peer, msg, op, tag, flags);

	txe->msg_id = peer->next_msg_id++;

	// Eager protocol sends 1 packet by definition
	pkt_entry_cnt = 1;

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	tagged = (op == ofi_op_tagged);

	if (flags & FI_INJECT ||
	    efa_both_support_zero_hdr_data_transfer(ep, peer))
		delivery_complete_requested = false;
	else
		delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	req_pkt_type = delivery_complete_requested ?
			       efa_rdm_proto_eager.req_pkt_type_dc + tagged :
			       efa_rdm_proto_eager.req_pkt_type + tagged;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);

	pkt_entry->ope = txe;
	pkt_entry->peer = peer;
	pkt_entry->handle_pke = &efa_rdm_proto_eager_handle_rtm_send_completion;

	// Refactored code path does not support zero copy
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) pkt_entry->wiredata;
	rtm_hdr->flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->msg_id = txe->msg_id;

	if (tagged) {
		rtm_hdr->flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	EFA_DBG(FI_LOG_EP_DATA,
		"eager protocol: dc_requested=%d tagged=%d req_pkt_type=%d\n",
		delivery_complete_requested, tagged, req_pkt_type);

	if (delivery_complete_requested) {
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
		dc_base_hdr = (struct efa_rdm_dc_eager_rtm_base_hdr *)
				      pkt_entry->wiredata;
		dc_base_hdr->send_id = txe->tx_id;
		pkt_entry->handle_pke =
			&efa_rdm_proto_eager_handle_rtm_dc_send_completion;
	}

	ret = efa_rdm_pke_init_payload_from_ope(
		pkt_entry, txe, efa_rdm_pke_get_req_hdr_size(pkt_entry), 0,
		txe->total_len);

	if (ret)
		goto out;

	// Verify that all of the data has been copied to the pke buffer
	assert(txe->total_len == pkt_entry->payload_size);

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_INFO(FI_LOG_EP_DATA,
		"eager protocol: posting 1 pke, size %lu, msg_id %" PRIu32 "\n",
		txe->total_len, txe->msg_id);

	return FI_SUCCESS;

out:
	if (txe) {
		peer->next_msg_id--;
		efa_rdm_txe_release(txe);
	}
	if (pkt_entry)
		efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}
