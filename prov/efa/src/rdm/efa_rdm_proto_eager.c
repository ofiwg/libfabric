/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto_eager.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_req.h"


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


struct efa_rdm_proto efa_rdm_proto_eager = {
	.construct_pkes = &efa_rdm_proto_eager_construct_pkes,
	.req_pkt_type = EFA_RDM_EAGER_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_EAGER_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_EAGER_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_EAGER_TAGRTM_PKT,
	.send_pkes_posted = &efa_rdm_proto_send_pkes_posted_no_op,
};


/* TX path callbacks - one callback for each packet type that this protocol uses */
void efa_rdm_proto_eager_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	return;
}

int efa_rdm_proto_eager_construct_txe(struct efa_rdm_ope **txe,
					  struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
					  const struct fi_msg *msg, uint32_t op, uint64_t tag,
					  uint64_t flags)
{
	struct efa_rdm_ope *new_txe;

	/* TODO: Move function to efa_rdm_proto.c
	 * TODO: Remove the switch statement on op in efa_rdm_txe_construct
	 * It's fine to have some of the txe construction logic in a shared function
	 * But we want to be fast, so we should have a different function for each op
	 */
	new_txe = efa_rdm_ep_alloc_txe(ep, peer, msg, op, tag, flags);

	if (!new_txe)
		return -FI_EAGAIN;

	new_txe->msg_id = peer->next_msg_id++;

	*txe = new_txe;
	return FI_SUCCESS;
}

int efa_rdm_proto_eager_construct_pkes(struct efa_rdm_ep *ep,
					   struct efa_rdm_peer *peer,
					   const struct fi_msg *msg, uint32_t op,
					   uint64_t tag, uint64_t flags, struct efa_rdm_ope **txe)
{
	int pkt_entry_cnt, available_tx_pkts;
	int *pkt_entry_data_size_vec = ep->send_pkt_entry_data_sizes;
	int ret, req_pkt_type;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_base_hdr *base_hdr;
	struct efa_rdm_ope *new_txe;
	struct efa_rdm_dc_eager_rtm_base_hdr *dc_base_hdr;

	ret = efa_rdm_proto_eager_construct_txe(txe, ep, peer, msg, op, tag, flags);
	if (ret)
		return ret;

	new_txe = *txe;

	// Eager protocol sends 1 packet by definition
	pkt_entry_cnt = 1;

	// Handle case when there are no TX packets available
	available_tx_pkts = ep->efa_max_outstanding_tx_ops -
			ep->efa_outstanding_tx_ops - ep->efa_rnr_queued_pkt_cnt;

	if (available_tx_pkts == 0)
		return -FI_EAGAIN;

	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

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

	pkt_entry->ope = new_txe;
	pkt_entry->peer = peer;

	if (efa_both_support_zero_hdr_data_transfer(ep, peer)) {
		/* zero hdr transfer only happens for eager msg (non-tagged) pkt */
		assert(req_pkt_type == EFA_RDM_EAGER_MSGRTM_PKT);
		pkt_entry->flags |= EFA_RDM_PKE_SEND_TO_USER_RECV_QP | EFA_RDM_PKE_HAS_NO_BASE_HDR;
		ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, new_txe, 0,
							0, new_txe->total_len);
	} else {
		ret = efa_rdm_pke_init_rtm_with_payload(pkt_entry, req_pkt_type, new_txe, 0, -1);
	}

	if (ret)
		goto out;

	assert(new_txe->total_len == pkt_entry->payload_size);

	if (tagged) {
		base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
		base_hdr->flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, new_txe->tag);
	}

	if (delivery_complete_requested) {
		new_txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
		dc_base_hdr = (struct efa_rdm_dc_eager_rtm_base_hdr *) pkt_entry->wiredata;
		dc_base_hdr->send_id = new_txe->tx_id;
	}

	pkt_entry->callback = &efa_rdm_proto_eager_handle_rtm_send_completion;

	// Set ep->send_pkt_entry_vec and related fields
	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	pkt_entry_data_size_vec[0] = -1;
	return FI_SUCCESS;

out:
	if (new_txe)
		efa_rdm_txe_release(new_txe);
	if (pkt_entry)
		efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}
