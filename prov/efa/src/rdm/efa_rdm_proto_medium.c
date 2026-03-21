/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto_medium.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_req.h"


/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_MEDIUM_MSGRTM_PKT
 * EFA_RDM_MEDIUM_TAGRTM_PKT
 * EFA_RDM_DC_MEDIUM_MSGRTM_PKT
 * EFA_RDM_DC_MEDIUM_TAGRTM_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */


/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#medium-message-featuresubprotocol
 */

static bool efa_rdm_proto_medium_can_use_for_send(struct efa_rdm_ope *txe,
						  struct efa_rdm_peer *peer,
						  int req_pkt_type,
						  uint16_t header_flags,
						  int iface, bool use_p2p)
{
	return txe->total_len <= g_efa_hmem_info[iface].max_medium_msg_size;
}

struct efa_rdm_proto efa_rdm_proto_medium = {
	.name = "MEDIUM",
	.can_use_protocol_for_send = &efa_rdm_proto_medium_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_medium_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_MEDIUM_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_MEDIUM_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_medium_handle_tx_pkes_posted,
};

void efa_rdm_proto_medium_handle_tx_pkes_posted(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe)
{
	for (int i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
		txe->bytes_sent += ep->send_pkt_entry_vec[i]->payload_size;
	}

	// For medium protocol, all of the data is posted at once.
	assert(txe->bytes_sent == txe->total_len);
}

/* TX path callbacks - one callback for each packet type that this protocol uses */
void efa_rdm_proto_medium_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	txe->bytes_acked += pkt_entry->payload_size;
	if (txe->bytes_acked >= txe->total_len)
		efa_rdm_ope_handle_send_completed(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

void efa_rdm_proto_medium_handle_rtm_dc_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (efa_rdm_txe_dc_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_tx(pkt_entry);
}

int efa_rdm_proto_medium_construct_tx_pkes(struct efa_rdm_ep *ep,
					   struct efa_rdm_peer *peer,
					   const struct fi_msg *msg, uint32_t op,
					   uint64_t tag, uint64_t flags, struct efa_rdm_ope *txe)
{
	int pkt_entry_cnt, pkt_entry_cnt_allocated = 0, single_pkt_entry_max_data_size, memory_alignment;
	int i, ret, req_pkt_type, iface, available_tx_pkts, single_pkt_entry_data_size, remainder;
	size_t segment_offset, hdr_size;
	int *pkt_entry_data_size_vec = ep->send_pkt_entry_data_sizes;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_medium_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_dc_medium_rtm_base_hdr *dc_medium_rtm_base_hdr;

	efa_rdm_proto_txe_fill(txe, ep, peer, msg, op, tag, flags);

	txe->msg_id = peer->next_msg_id++;

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt > 0);


	/* Select req_pkt_type based on whether FI_TAGGED is set and whether
	 * delivery_complete is requested
	 */
	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	// Refactored code path does not support zero copy
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	// Inject should use eager protocol
	assert(!(flags & FI_INJECT));

	delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	req_pkt_type = delivery_complete_requested ?
		efa_rdm_proto_medium.req_pkt_type_dc + tagged :
		efa_rdm_proto_medium.req_pkt_type + tagged;


	single_pkt_entry_max_data_size = efa_rdm_txe_max_req_data_capacity(ep, txe, req_pkt_type);
	assert(single_pkt_entry_max_data_size > 0);

	iface = (msg->desc && msg->desc[0]) ? ((struct efa_mr*) msg->desc[0])->peer.iface : FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);

	pkt_entry_cnt = (txe->total_len - 1) / single_pkt_entry_max_data_size + 1;


	/* when sending multiple packets, it is more performant that the data
	 * size of each packet are close to achieve that, we calculate the
	 * single packet size
	 */
	single_pkt_entry_data_size =
		(txe->total_len - 1) / pkt_entry_cnt + 1;

	/* each packet must be aligned */
	single_pkt_entry_data_size =
		single_pkt_entry_data_size & ~(memory_alignment - 1);

	assert(single_pkt_entry_data_size);

	pkt_entry_cnt = txe->total_len / single_pkt_entry_data_size;
	for (i = 0; i < pkt_entry_cnt; ++i)
		pkt_entry_data_size_vec[i] = single_pkt_entry_data_size;

	remainder = txe->total_len -
		    pkt_entry_cnt * single_pkt_entry_data_size;
	if (single_pkt_entry_data_size + remainder <=
	    single_pkt_entry_max_data_size) {
		pkt_entry_data_size_vec[pkt_entry_cnt - 1] += remainder;
	} else {
		pkt_entry_data_size_vec[pkt_entry_cnt] = remainder;
		pkt_entry_cnt += 1;
	}



	available_tx_pkts = ep->efa_max_outstanding_tx_ops -
			ep->efa_outstanding_tx_ops - ep->efa_rnr_queued_pkt_cnt;

	if (pkt_entry_cnt > available_tx_pkts)
		return -FI_EAGAIN;

	assert(pkt_entry_cnt <= efa_base_ep_get_tx_pool_size(&ep->base_ep));


	segment_offset = 0;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool, EFA_RDM_PKE_FROM_EFA_TX_POOL);

		if (OFI_UNLIKELY(!pkt_entry)) {
			ret = -FI_EAGAIN;
			goto out;
		}

		pkt_entry_cnt_allocated++;

		pkt_entry->ope = txe;
		pkt_entry->peer = peer;

		assert(pkt_entry_data_size_vec[i] > 0);

		efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

		rtm_hdr = efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry);
		rtm_hdr->hdr.flags |= EFA_RDM_REQ_MSG;
		rtm_hdr->hdr.msg_id = txe->msg_id;

		if (tagged) {
			rtm_hdr->hdr.flags |= EFA_RDM_REQ_TAGGED;
			efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
		}

		if (delivery_complete_requested) {
			txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
			dc_medium_rtm_base_hdr = (struct efa_rdm_dc_medium_rtm_base_hdr *) pkt_entry->wiredata;
			dc_medium_rtm_base_hdr->send_id = txe->tx_id;
			dc_medium_rtm_base_hdr->msg_length = txe->total_len;
			dc_medium_rtm_base_hdr->seg_offset = segment_offset;
			pkt_entry->callback =
				&efa_rdm_proto_medium_handle_rtm_dc_send_completion;
		} else {
			rtm_hdr->msg_length = txe->total_len;
			rtm_hdr->seg_offset = segment_offset;
			pkt_entry->callback =
				&efa_rdm_proto_medium_handle_rtm_send_completion;
		}

		assert(pkt_entry->callback);

		hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);

		ret = efa_rdm_pke_init_payload_from_ope(
			pkt_entry, txe, hdr_size, segment_offset,
			pkt_entry_data_size_vec[i]);

		if (ret)
			goto out;

		ep->send_pkt_entry_vec[i] = pkt_entry;
		segment_offset += ep->send_pkt_entry_data_sizes[i];
	}

	assert(pkt_entry_cnt == pkt_entry_cnt_allocated);

	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	return FI_SUCCESS;

out:
	if (txe) {
		peer->next_msg_id--;
		efa_rdm_txe_release(txe);
	}
	for (i = 0; i < pkt_entry_cnt_allocated; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	return ret;
}
