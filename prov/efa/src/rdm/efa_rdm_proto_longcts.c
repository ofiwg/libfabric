/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts.h"
#include "efa.h"
#include "efa_rdm_pke_req.h"
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
 * EFA_RDM_CTS_PKT
 * EFA_RDM_CTSDATA_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#long-cts-message-featuresubprotocol
 */

static bool efa_rdm_proto_longcts_can_use_for_send(struct efa_rdm_ope *txe,
						   struct efa_rdm_peer *peer,
						   int req_pkt_type,
						   uint16_t header_flags,
						   int iface, bool use_p2p)
{
	// Long CTS is always usable
	return true;
}

struct efa_rdm_proto efa_rdm_proto_longcts = {
	.name = "LONGCTS",
	.can_use_protocol_for_send = &efa_rdm_proto_longcts_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longcts_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGCTS_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_LONGCTS_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

void efa_rdm_proto_longcts_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						 struct efa_rdm_ope *txe)
{
	for (int i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
		txe->bytes_sent += ep->send_pkt_entry_vec[i]->payload_size;
		assert(txe->bytes_sent < txe->total_len);
	}

	/* Try to register application buffer again
	 * We first try to register application's buffer in
	 * efa_rdm_proto_select_send_protocol. We try here again in case the
	 * first attempt failed because of e.g. number of MRs reaching device
	 * limits */
	if (efa_is_cache_available(efa_rdm_ep_domain(ep)))
		efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
}

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
void efa_rdm_proto_longcts_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	/**
	 * A zero-payload longcts rtm pkt currently should only happen when it's
	 * used for the READ NACK protocol. In this case, this pkt doesn't
	 * contribute to the send completion, and the associated tx entry
	 * may be released earlier as the CTSDATA pkts have already kicked off
	 * and finished the send.
	 */
	if (pkt_entry->payload_size == 0) {
		assert(efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->flags &
		       EFA_RDM_REQ_READ_NACK);
		return;
	}

	txe = pkt_entry->ope;

	txe->bytes_acked += pkt_entry->payload_size;

	/* Long CTS protocol should not be used when the total buffer size can
	 * fit in one packet */
	assert(txe->total_len != txe->bytes_acked);

	efa_rdm_pke_release_tx(pkt_entry);
}

void efa_rdm_proto_longcts_handle_rtm_dc_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (efa_rdm_txe_dc_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

void efa_rdm_proto_longcts_handle_cts_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	return;
}

void efa_rdm_proto_longcts_handle_ctsdata_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
}

int efa_rdm_proto_longcts_construct_tx_pkes(struct efa_rdm_ep *ep,
					    struct efa_rdm_peer *peer,
					    const struct fi_msg *msg,
					    uint32_t op, uint64_t tag,
					    uint64_t flags,
					    struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type, iface, pkt_entry_cnt;
	size_t hdr_size, rtm_payload_size, memory_alignment;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_longcts_rtm_base_hdr *rtm_hdr;

	efa_rdm_proto_txe_fill(txe, ep, peer, msg, op, tag, flags);

	txe->msg_id = peer->next_msg_id++;

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	// Long CTS protocol sends 1 req packet by definition
	pkt_entry_cnt = 1;

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt > 0);

	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	// Refactored code path does not support zero copy
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	// Inject should use eager protocol
	assert(!(flags & FI_INJECT));

	delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	req_pkt_type = delivery_complete_requested ?
			       efa_rdm_proto_longcts.req_pkt_type_dc + tagged :
			       efa_rdm_proto_longcts.req_pkt_type + tagged;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);

	pkt_entry->ope = txe;
	pkt_entry->peer = peer;
	pkt_entry->callback = &efa_rdm_proto_longcts_handle_rtm_send_completion;

	// Zero copy path should use the eager protocol
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

	rtm_hdr = (struct efa_rdm_longcts_rtm_base_hdr *) pkt_entry->wiredata;
	rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;

	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->credit_request = efa_env.tx_min_credits;

	if (tagged) {
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	if (delivery_complete_requested) {
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
		rtm_hdr->send_id = txe->tx_id;
		pkt_entry->callback =
			&efa_rdm_proto_longcts_handle_rtm_dc_send_completion;
	}

	// Calculate hdr_size after initializing the flags in
	// efa_rdm_pke_init_req_hdr_common
	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	rtm_payload_size = txe->ep->mtu_size - hdr_size;

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->peer.iface :
			FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);

	rtm_payload_size &= ~(memory_alignment - 1);

	ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, hdr_size, 0,
						rtm_payload_size);

	if (ret)
		goto out;

	// Set ep->send_pkt_entry_vec and related fields
	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
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
