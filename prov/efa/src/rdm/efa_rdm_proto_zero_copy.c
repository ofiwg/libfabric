/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_zero_copy.h"
#include "efa.h"
#include "efa_rdm_pke_utils.h"

/*
 * This protocol exists for backwards compatibility with older peers that have
 * zero-copy receive enabled. Such a peer posts application buffers directly on
 * a dedicated user_recv_qp and rejects data packets sent to its control QP, so
 * the user data must be sent with no protocol header at all.
 *
 * The protocol is deliberately not registered in efa_rdm_protocols[] so that
 * it is not checked for all other message operations.
 */

struct efa_rdm_proto efa_rdm_proto_zero_copy = {
	.construct_tx_pkes = &efa_rdm_proto_zero_copy_construct_tx_pkes,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

void efa_rdm_proto_zero_copy_reselect_queued_before_handshake(
	struct efa_rdm_ope *txe)
{
	assert(txe->peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);

	if (efa_rdm_peer_expects_zero_hdr_data_transfer(txe->peer))
		txe->proto = &efa_rdm_proto_zero_copy;
}

/* TX path callbacks */

/**
 * @brief Handle the send completion of a zero-copy packet.
 */
void efa_rdm_proto_zero_copy_handle_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);
	assert(txe->total_len == pkt_entry->payload_size);
	assert(!(txe->internal_flags &
		 EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED));

	efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct TX packet entries for the zero-copy protocol.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_zero_copy_construct_tx_pkes(struct efa_rdm_ep *ep,
					      struct efa_rdm_peer *peer,
					      const struct fi_msg *msg,
					      uint32_t op, uint64_t tag,
					      uint64_t flags,
					      uint32_t internal_flags,
					      struct efa_rdm_ope *txe)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	assert(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED);
	assert(txe->total_len <= ep->mtu_size);
	assert(op == ofi_op_msg);

	txe->protocol = EFA_RDM_EAGER_MSGRTM_PKT;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;
	pkt_entry->handle_pke = &efa_rdm_proto_zero_copy_handle_send_completion;
	pkt_entry->flags |= EFA_RDM_PKE_SEND_TO_USER_RECV_QP |
			    EFA_RDM_PKE_HAS_NO_BASE_HDR;

	ret = efa_rdm_pke_init_payload_from_ope(pkt_entry, txe, 0, 0,
						txe->total_len);
	if (ret) {
		efa_rdm_pke_release_tx(pkt_entry);
		return ret;
	}

	// Verify that all of the data has been copied to the pke buffer
	assert(txe->total_len == pkt_entry->payload_size);

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;
	EFA_INFO(FI_LOG_EP_DATA,
		 "zero-copy protocol: posting 1 pke, size %lu, msg_id %" PRIu32
		 "\n",
		 txe->total_len, txe->msg_id);

	return FI_SUCCESS;
}
