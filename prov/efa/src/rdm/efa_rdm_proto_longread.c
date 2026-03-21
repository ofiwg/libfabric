/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longread.h"
#include "efa.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_LONGREAD_MSGRTM_PKT
 * EFA_RDM_LONGREAD_TAGRTM_PKT
 *
 * EFA_RDM_EOR_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#long-read-message-featuresubprotocol
 */

/**
 * @brief Check if the long read protocol can handle this send operation.
 *
 * Requires p2p availability, registered memory descriptors, the message
 * meeting the minimum read size threshold, and peer RDMA read support.
 */
static bool efa_rdm_proto_longread_can_use_for_send(struct efa_rdm_ope *txe,
						    struct efa_rdm_peer *peer,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	bool size, read_interop, mr_avail;

	mr_avail = (txe->desc[0] != NULL);
	size = txe->total_len >= g_efa_hmem_info[iface].min_read_msg_size;
	read_interop = efa_rdm_interop_rdma_read(txe->ep, peer);

	EFA_DBG(FI_LOG_EP_DATA,
		"longread eligibility: use_p2p=%d mr_avail=%d size=%d read_interop=%d\n",
		use_p2p, mr_avail, size, read_interop);

	if (use_p2p && mr_avail && size && read_interop)
		return true;

	return false;
}

struct efa_rdm_proto efa_rdm_proto_longread = {
	.name = "LONGREAD",
	.can_use_protocol_for_send = &efa_rdm_proto_longread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_longread_construct_tx_pkes,
	// Long read protocol is always delivery complete
	.req_pkt_type = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_LONGREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_LONGREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longread_handle_tx_pkes_posted,
};

void efa_rdm_proto_longread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						  struct efa_rdm_ope *txe)
{
	efa_rdm_ep_domain(ep)->num_read_msg_in_flight += 1;
}

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
/**
 * @brief Handle send completion for a long read RTM packet.
 *
 * Simply releases the TX packet entry since the actual data transfer
 * is driven by the receiver via RDMA reads.
 */
void efa_rdm_proto_longread_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	// Nothing to do except to release the pkt_entry
	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct TX packet entries for the long read protocol.
 *
 * Sends a single RTM packet containing the memory keys and addresses
 * of the sender's registered buffers. The receiver performs RDMA reads
 * to fetch the data.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longread_construct_tx_pkes(struct efa_rdm_ep *ep,
					     struct efa_rdm_peer *peer,
					     const struct fi_msg *msg,
					     uint32_t op, uint64_t tag,
					     uint64_t flags,
					     struct efa_rdm_ope *txe)
{
	int i, ret, req_pkt_type, pkt_entry_cnt;
	bool tagged;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;

	efa_rdm_proto_txe_fill(txe, ep, peer, msg, op, tag, flags);

	txe->msg_id = peer->next_msg_id++;

	/* Read based protocols shouldn't be chosen if the local buffer cannot
	 * be registered */
	assert(txe->desc[0]);

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	// Refactored code path does not support zero copy
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	// Inject should use eager protocol
	assert(!(flags & FI_INJECT));

	// Long read protocol sends 1 req packet by definition
	pkt_entry_cnt = 1;

	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	// Long read protocol is always delivery complete
	assert(efa_rdm_proto_longread.req_pkt_type ==
	       efa_rdm_proto_longread.req_pkt_type_dc);
	assert(efa_rdm_proto_longread.req_pkt_type_tagged ==
	       efa_rdm_proto_longread.req_pkt_type_tagged_dc);

	req_pkt_type = efa_rdm_proto_longread.req_pkt_type + tagged;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (!pkt_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	pkt_entry->ope = txe;
	pkt_entry->peer = peer;
	pkt_entry->callback =
		&efa_rdm_proto_longread_handle_rtm_send_completion;

	efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

	rtm_hdr = (struct efa_rdm_longread_rtm_base_hdr *) pkt_entry->wiredata;
	rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
	rtm_hdr->hdr.msg_id = txe->msg_id;
	rtm_hdr->msg_length = txe->total_len;
	rtm_hdr->send_id = txe->tx_id;
	rtm_hdr->read_iov_count = txe->iov_count;

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);

	pkt_entry->pkt_size =
		hdr_size + txe->iov_count * sizeof(struct fi_rma_iov);

	// Logic copied from efa_rdm_txe_prepare_to_be_read
	for (i = 0; i < txe->iov_count; ++i) {
		read_iov[i].addr = (uint64_t) txe->iov[i].iov_base;
		read_iov[i].len = txe->iov[i].iov_len;
		read_iov[i].key = fi_mr_key(txe->desc[i]);
	}

	if (tagged) {
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
		efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
	}

	// Set ep->send_pkt_entry_vec and related fields
	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_INFO(FI_LOG_EP_DATA,
		"longread protocol: posting 1 pke, pkt_size %lu, total_len %lu\n",
		pkt_entry->pkt_size, txe->total_len);

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
