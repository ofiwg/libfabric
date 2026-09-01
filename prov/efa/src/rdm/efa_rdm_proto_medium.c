/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_medium.h"
#include "efa.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

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

/**
 * @brief Check if the medium protocol can handle this send operation.
 *
 * The medium protocol carries the whole message in REQ packets, so it is only
 * worth using up to the interface's medium threshold (64KB for system memory,
 * and 0 - i.e. never - for every HMEM interface, which use the read based
 * protocols instead).
 */
static bool efa_rdm_proto_medium_can_use_for_send(struct efa_rdm_ope *txe,
						  int req_pkt_type,
						  uint16_t header_flags,
						  int iface)
{
	/* A zero sized message should use eager protocol */
	assert(txe->total_len > 0);

	return txe->total_len <= g_efa_hmem_info[iface].max_medium_msg_size;
}

struct efa_rdm_proto efa_rdm_proto_medium = {
	.name = "medium",
	.can_use_protocol_for_send = &efa_rdm_proto_medium_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_medium_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_MEDIUM_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_MEDIUM_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_MEDIUM_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_medium_handle_tx_pkes_posted,
};

/**
 * @brief Account for the medium packets that just reached the device.
 *
 * This function runs once per successful operation after all pkes are
 * posted. So it is OK to directly assign bytes_sent here.
 */
void efa_rdm_proto_medium_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						struct efa_rdm_ope *txe)
{
	txe->bytes_sent = txe->total_len;
}

/* TX path callbacks - one callback per protocol, which reads the txe to tell
 * the transmit complete and delivery complete variants apart
 */
/**
 * @brief Handle send completion for a medium RTM packet.
 *
 * Generate an application completion only when we receive send completions
 * for all of the medium pkes. Also handle the MR abort and delivery complete
 * cases.
 *
 */
void efa_rdm_proto_medium_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;
	bool delivery_complete_requested;

	txe = pkt_entry->ope;
	assert(txe);

	delivery_complete_requested =
		txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	txe->bytes_acked += pkt_entry->payload_size;

	if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(txe);
	else if (delivery_complete_requested) {
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else if (txe->total_len == txe->bytes_acked) {
		efa_rdm_ope_handle_send_completed(txe);
	}

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Calculate how many REQ packets are required and how much data each
 *        packet carries.
 *
 * The data is spread evenly over the fewest packets that can carry it.
 * because the EFA device is better at handling a set of similarly sized
 * sends.
 *
 * Each size is rounded down to the interface's memory alignment.
 *
 * @param[out] pkt_entry_cnt		number of packets to send
 * @param[out] pkt_entry_data_size_vec	data size of each of those packets
 * @return 0 on success, -FI_EAGAIN when that many packets cannot be supplied
 *	   right now
 */
ssize_t efa_rdm_proto_medium_plan_tx_pkes(struct efa_rdm_ep *ep,
					 struct efa_rdm_ope *txe,
					 int req_pkt_type,
					 size_t *pkt_entry_cnt,
					 size_t *pkt_entry_data_size_vec)
{
	size_t i, max_pkt_entry_cnt, max_data_size, data_size, remainder;
	size_t full_pkt_cnt, memory_alignment;
	enum fi_hmem_iface iface;

	assert(efa_rdm_pkt_type_is_medium(req_pkt_type));
	assert(txe->total_len);
	assert(ep->efa_max_outstanding_tx_ops >=
	       ep->efa_outstanding_tx_ops + ep->efa_rnr_queued_pkt_cnt);

	max_pkt_entry_cnt = MIN(efa_rdm_ep_get_available_tx_pkts(ep),
				efa_base_ep_get_tx_pool_size(&ep->base_ep));

	assert(max_pkt_entry_cnt > 0);

	max_data_size = efa_rdm_proto_max_req_data_capacity(
		ep, req_pkt_type,
		efa_rdm_proto_req_header_flags(txe->peer, txe->fi_flags));
	assert(max_data_size);

	iface = txe->desc[0] ? ((struct efa_mr *) txe->desc[0])->iface :
			       FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);

	*pkt_entry_cnt = (txe->total_len - 1) / max_data_size + 1;
	data_size = (txe->total_len - 1) / *pkt_entry_cnt + 1;
	data_size &= ~(memory_alignment - 1);
	assert(data_size);

	full_pkt_cnt = txe->total_len / data_size;
	remainder = txe->total_len - full_pkt_cnt * data_size;
	*pkt_entry_cnt = full_pkt_cnt +
			 (data_size + remainder > max_data_size ? 1 : 0);

	if (*pkt_entry_cnt > max_pkt_entry_cnt)
		return -FI_EAGAIN;

	assert(full_pkt_cnt > 0);
	for (i = 0; i < full_pkt_cnt; ++i)
		pkt_entry_data_size_vec[i] = data_size;

	if (*pkt_entry_cnt == full_pkt_cnt)
		pkt_entry_data_size_vec[full_pkt_cnt - 1] += remainder;
	else
		pkt_entry_data_size_vec[full_pkt_cnt] = remainder;

	return FI_SUCCESS;
}

/**
 * @brief Construct TX packet entries for the medium protocol.
 *
 * On success, ep->send_pkt_entry_vec holds the packet entries and
 * ep->send_pkt_entry_vec_size is the number of packets.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_medium_construct_tx_pkes(struct efa_rdm_ep *ep,
					   struct efa_rdm_peer *peer,
					   const struct fi_msg *msg, uint32_t op,
					   uint64_t tag, uint64_t flags,
					   uint32_t internal_flags,
					   struct efa_rdm_ope *txe)
{
	int ret, req_pkt_type;
	size_t i, pkt_entry_cnt = 0, pkt_entry_cnt_allocated = 0;
	size_t segment_offset;
	size_t *pkt_entry_data_size_vec = ep->send_pkt_entry_vec_data_sizes;
	bool tagged, delivery_complete_requested;
	struct efa_rdm_pke *pkt_entry;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	struct efa_rdm_medium_rtm_base_hdr *medium_rtm_hdr;
	struct efa_rdm_dc_medium_rtm_base_hdr *dc_medium_rtm_hdr;

	/*
	 * Inject should always use eager protocol
	 */
	assert(!(flags & FI_INJECT));

	tagged = (op == ofi_op_tagged);

	req_pkt_type = txe->req_pkt_type;
	delivery_complete_requested =
		(req_pkt_type == efa_rdm_proto_medium.req_pkt_type_dc ||
		 req_pkt_type == efa_rdm_proto_medium.req_pkt_type_tagged_dc);

	if (delivery_complete_requested)
		txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	ret = efa_rdm_proto_medium_plan_tx_pkes(ep, txe, req_pkt_type,
						&pkt_entry_cnt,
						pkt_entry_data_size_vec);
	if (ret)
		return ret;

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

		efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

		rtm_hdr = efa_rdm_pke_get_rtm_base_hdr(pkt_entry);
		rtm_hdr->flags |= EFA_RDM_REQ_MSG;
		rtm_hdr->msg_id = txe->msg_id;

		if (tagged) {
			rtm_hdr->flags |= EFA_RDM_REQ_TAGGED;
			efa_rdm_pke_set_rtm_tag(pkt_entry, txe->tag);
		}

		pkt_entry->handle_pke =
			&efa_rdm_proto_medium_handle_rtm_send_completion;

		if (delivery_complete_requested) {
			dc_medium_rtm_hdr =
				efa_rdm_pke_get_dc_medium_rtm_base_hdr(pkt_entry);
			dc_medium_rtm_hdr->send_id = txe->tx_id;
			dc_medium_rtm_hdr->msg_length = txe->total_len;
			dc_medium_rtm_hdr->seg_offset = segment_offset;
		} else {
			medium_rtm_hdr =
				efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry);
			medium_rtm_hdr->msg_length = txe->total_len;
			medium_rtm_hdr->seg_offset = segment_offset;
		}

		ret = efa_rdm_pke_init_payload_from_ope(
			pkt_entry, txe, efa_rdm_pke_get_req_hdr_size(pkt_entry),
			segment_offset, pkt_entry_data_size_vec[i]);
		if (ret)
			goto err_release_pkes;

		assert(pkt_entry->payload_size == pkt_entry_data_size_vec[i]);
		segment_offset += pkt_entry_data_size_vec[i];
	}

	/* Make sure all of the data is copied */
	assert(segment_offset == txe->total_len);

	ep->send_pkt_entry_vec_size = pkt_entry_cnt;
	EFA_DBG(FI_LOG_EP_DATA,
		"medium protocol: posting %zu pkes, total_len %lu, msg_id %" PRIu32
		"\n",
		pkt_entry_cnt, txe->total_len, txe->msg_id);

	return FI_SUCCESS;

err_release_pkes:
	for (i = 0; i < pkt_entry_cnt_allocated; ++i)
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[i]);
	return ret;
}
