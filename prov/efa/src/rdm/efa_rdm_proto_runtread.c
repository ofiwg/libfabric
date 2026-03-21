/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_runtread.h"
#include "efa.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_RUNTREAD_MSGRTM_PKT
 * EFA_RDM_RUNTREAD_TAGRTM_PKT
 *
 * EFA_RDM_EOR_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#45-runting-read-message-subprotocol
 */

static bool efa_rdm_proto_runtread_can_use_for_send(struct efa_rdm_ope *txe,
						    struct efa_rdm_peer *peer,
						    int req_pkt_type,
						    uint16_t header_flags,
						    int iface, bool use_p2p)
{
	bool size, read_interop, mr_avail, no_read_in_progress, runt_allowed;

	mr_avail = txe->desc && txe->desc[0];
	size = txe->total_len >= g_efa_hmem_info[iface].min_read_msg_size;
	read_interop = efa_rdm_interop_rdma_read(txe->ep, peer);
	no_read_in_progress =
		efa_rdm_ep_domain(txe->ep)->num_read_msg_in_flight == 0;
	runt_allowed = g_efa_hmem_info[iface].runt_size <
		       peer->num_runt_bytes_in_flight;

	if (use_p2p && mr_avail && size && read_interop &&
	    no_read_in_progress && runt_allowed)
		return true;

	return false;
}

struct efa_rdm_proto efa_rdm_proto_runtread = {
	.name = "RUNTREAD",
	.can_use_protocol_for_send = &efa_rdm_proto_runtread_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_runtread_construct_tx_pkes,
	// Runting read protocol is always delivery complete
	.req_pkt_type = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_RUNTREAD_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_RUNTREAD_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_runtread_handle_tx_pkes_posted,
};

void efa_rdm_proto_runtread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						  struct efa_rdm_ope *txe)
{
	size_t pkt_data_size;

	for (int i = 0; i < ep->send_pkt_entry_vec_size; ++i) {
		pkt_data_size = ep->send_pkt_entry_vec[i]->payload_size;
		txe->bytes_sent += pkt_data_size;
		txe->peer->num_runt_bytes_in_flight += pkt_data_size;
	}

	efa_rdm_ep_domain(txe->ep)->num_read_msg_in_flight++;
}

/* TX path callbacks - one callback for each packet type that this protocol uses
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

	/* If the entire buffer could be sent in RTM packets, we should have
	 * used the medium protocol instead of the runting read
	 */
	assert(txe->bytes_acked < txe->total_len);

	peer = txe->peer;
	assert(peer);
	assert(peer->num_runt_bytes_in_flight >= pkt_data_size);
	peer->num_runt_bytes_in_flight -= pkt_data_size;

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief calculate and set the bytes_runt field of a txe
 *
 * bytes_runt is number of bytes for a message to be sent by runting
 *
 * @param[in]		ep			endpoint
 * @param[in,out]	txe	txe to be set
 */
static inline void efa_rdm_proto_runtread_set_runt_size(struct efa_rdm_ep *ep,
							struct efa_rdm_ope *txe)
{
	assert(txe->type == EFA_RDM_TXE);

	if (txe->bytes_runt > 0)
		return;

	assert(txe->peer);
	txe->bytes_runt = efa_rdm_peer_get_runt_size(txe->peer, ep, txe);

	assert(txe->bytes_runt);
}

int efa_rdm_proto_runtread_construct_tx_pkes(struct efa_rdm_ep *ep,
					     struct efa_rdm_peer *peer,
					     const struct fi_msg *msg,
					     uint32_t op, uint64_t tag,
					     uint64_t flags,
					     struct efa_rdm_ope *txe)
{
	int pkt_entry_cnt, pkt_entry_cnt_allocated = 0,
			   single_pkt_entry_max_data_size, memory_alignment;
	int i, ret, req_pkt_type, iface, available_tx_pkts,
		single_pkt_entry_data_size, remainder;
	size_t segment_offset, hdr_size, payload_offset;
	struct fi_rma_iov *read_iov;
	int *pkt_entry_data_size_vec = ep->send_pkt_entry_data_sizes;
	bool tagged;
	struct efa_rdm_pke *pkt_entry = NULL;
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr;

	efa_rdm_proto_txe_fill(txe, ep, peer, msg, op, tag, flags);
	efa_rdm_proto_runtread_set_runt_size(ep, txe);

	// Should use medium protocol if the entire buffer can fit in the runt
	// size
	assert(txe->bytes_runt < txe->total_len);

	/* Read based protocols shouldn't be chosen if the local buffer cannot
	 * be registered */
	assert(txe->desc && txe->desc[0]);

	// Verify that the send queue is not full
	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt > 0);

	// Refactored code path does not support zero copy
	assert(!efa_both_support_zero_hdr_data_transfer(ep, peer));

	// Inject should use eager protocol
	assert(!(flags & FI_INJECT));

	txe->msg_id = peer->next_msg_id++;

	/* Select req_pkt_type based on whether FI_TAGGED is set and whether
	 * delivery_complete is requested
	 */
	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	// Runting read is always delivery complete
	assert(efa_rdm_proto_runtread.req_pkt_type ==
	       efa_rdm_proto_runtread.req_pkt_type_dc);
	assert(efa_rdm_proto_runtread.req_pkt_type_tagged ==
	       efa_rdm_proto_runtread.req_pkt_type_tagged_dc);

	req_pkt_type = efa_rdm_proto_runtread.req_pkt_type + tagged;

	single_pkt_entry_max_data_size =
		efa_rdm_txe_max_req_data_capacity(ep, txe, req_pkt_type);
	assert(single_pkt_entry_max_data_size > 0);

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->peer.iface :
			FI_HMEM_SYSTEM;
	memory_alignment = efa_rdm_ep_get_memory_alignment(ep, iface);

	pkt_entry_cnt =
		(txe->total_len - 1) / single_pkt_entry_max_data_size + 1;

	/* when sending multiple packets, it is more performant that the data
	 * size of each packet are close to achieve that, we calculate the
	 * single packet size
	 */
	single_pkt_entry_data_size = (txe->bytes_runt - 1) / pkt_entry_cnt + 1;

	/* each packet must be aligned */
	single_pkt_entry_data_size =
		single_pkt_entry_data_size & ~(memory_alignment - 1);

	assert(single_pkt_entry_data_size);

	pkt_entry_cnt = txe->total_len / single_pkt_entry_data_size;
	for (i = 0; i < pkt_entry_cnt; ++i)
		pkt_entry_data_size_vec[i] = single_pkt_entry_data_size;

	remainder = txe->total_len - pkt_entry_cnt * single_pkt_entry_data_size;
	if (single_pkt_entry_data_size + remainder <=
	    single_pkt_entry_max_data_size) {
		pkt_entry_data_size_vec[pkt_entry_cnt - 1] += remainder;
	} else {
		pkt_entry_data_size_vec[pkt_entry_cnt] = remainder;
		pkt_entry_cnt += 1;
	}

	available_tx_pkts = ep->efa_max_outstanding_tx_ops -
			    ep->efa_outstanding_tx_ops -
			    ep->efa_rnr_queued_pkt_cnt;

	if (pkt_entry_cnt > available_tx_pkts)
		return -FI_EAGAIN;

	assert(pkt_entry_cnt <= efa_base_ep_get_tx_pool_size(&ep->base_ep));

	segment_offset = 0;
	for (i = 0; i < pkt_entry_cnt; ++i) {
		pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
					      EFA_RDM_PKE_FROM_EFA_TX_POOL);

		if (OFI_UNLIKELY(!pkt_entry)) {
			ret = -FI_EAGAIN;
			goto out;
		}

		pkt_entry_cnt_allocated++;

		pkt_entry->ope = txe;
		pkt_entry->peer = peer;
		pkt_entry->callback =
			&efa_rdm_proto_runtread_handle_rtm_send_completion;

		assert(pkt_entry_data_size_vec[i] > 0);

		efa_rdm_pke_init_req_hdr_common(pkt_entry, req_pkt_type, txe);

		rtm_hdr = (struct efa_rdm_runtread_rtm_base_hdr *)
				  pkt_entry->wiredata;
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

		hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
		read_iov =
			(struct fi_rma_iov *) (pkt_entry->wiredata + hdr_size);

		// Logic copied from efa_rdm_txe_prepare_to_be_read
		for (i = 0; i < txe->iov_count; ++i) {
			read_iov[i].addr = (uint64_t) txe->iov[i].iov_base;
			read_iov[i].len = txe->iov[i].iov_len;
			read_iov[i].key = fi_mr_key(txe->desc[i]);
		}
		payload_offset =
			hdr_size + sizeof(struct fi_rma_iov) * txe->iov_count;

		assert(pkt_entry->callback);

		ret = efa_rdm_pke_init_payload_from_ope(
			pkt_entry, txe, payload_offset, segment_offset,
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
