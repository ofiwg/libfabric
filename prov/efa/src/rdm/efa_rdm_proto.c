/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto.h"
#include "efa_rdm_proto_eager.h"
#include "efa_rdm_proto_medium.h"

#define EFA_RDM_MAX_PROTO 8

/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 */
struct efa_rdm_proto *efa_rdm_protocols [EFA_RDM_MAX_PROTO] = {
	&efa_rdm_proto_eager,
	&efa_rdm_proto_medium,
	NULL,	/* Sentinel used to stop iteration */
};

int efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags, struct efa_rdm_ope *txe,
				       struct efa_rdm_proto **proto)
{
	struct efa_rdm_proto *selected_proto;
	int req_pkt_type, iface, err, use_p2p;
	uint16_t header_flags = 0;
	bool tagged, delivery_complete_requested;

	if (flags & FI_INJECT ||
		efa_both_support_zero_hdr_data_transfer(ep, peer))
		delivery_complete_requested = false;
	else
		delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	txe->total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	/* TODO: These fields are copied to the txe because the current
	 * implementation of efa_rdm_ope_try_fill_desc relies on it. Eliminate
	 * unncessary copies wherever possible. */
	txe->ep = ep;
	txe->iov_count = msg->iov_count;
	memcpy(txe->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);
	memset(txe->mr, 0, sizeof(*txe->mr) * msg->iov_count);
	if (msg->desc)
		memcpy(txe->desc, msg->desc, sizeof(*msg->desc) * msg->iov_count);
	else
		memset(txe->desc, 0, sizeof(*txe->desc) * msg->iov_count);

	iface = (msg->desc && msg->desc[0]) ? ((struct efa_mr*) msg->desc[0])->peer.iface : FI_HMEM_SYSTEM;

	err = efa_rdm_ep_use_p2p(ep, (msg->desc && msg->desc[0]) ? msg->desc[0] : NULL);
	if (err < 0)
		return err;
	use_p2p = err;

	/* Logic copied from efa_rdm_txe_max_req_data_capacity */
	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= EFA_RDM_PKT_CONNID_HDR;

	if (flags & FI_REMOTE_CQ_DATA)
		header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

	for (int i = 0; i < EFA_RDM_MAX_PROTO; ++i) {
		selected_proto = efa_rdm_protocols[i];

		if (!selected_proto)
			break;

		/*
		 * For performance consideration, this function assume the
		 * tagged rtm packet type id is always the correspondent message
		 * rtm packet type id + 1, thus the assertion here.
		 */
		assert(selected_proto->req_pkt_type_tagged ==
			   selected_proto->req_pkt_type + 1);
		assert(selected_proto->req_pkt_type_tagged_dc ==
			   selected_proto->req_pkt_type_dc + 1);

		/* TODO: The req_pkt_type is again needed in each protocol when allocating pkes
		 * Option 1: Make pkt headers independent of tag and DC to avoid these checks
		 * Option 2: Store the req_pkt_type in the txe
		 */
		req_pkt_type = delivery_complete_requested ?
			selected_proto->req_pkt_type_dc + tagged :
			selected_proto->req_pkt_type + tagged;

		/* All protocols other than the eager protocol can benefit from
		 * registering the application buffers.
		 * TODO: Move function to efa_rdm_proto.c
		 */
		if (selected_proto != &efa_rdm_proto_eager) {
			// Try to register buffer if MR cache is available
			if (efa_is_cache_available(efa_rdm_ep_domain(ep)))
				efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
		}

		if (selected_proto->can_use_protocol_for_send(txe, req_pkt_type,
							      header_flags, iface)) {
			*proto = selected_proto;
			return FI_SUCCESS;
		}
	}

	*proto = NULL;
	return FI_SUCCESS;
}


/* Utility funcions */

void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags)
{
	/* Logic copied from efa_rdm_txe_construct */
	uint64_t tx_op_flags;

	/* txe->mr, txe->desc, txe->total_len are filled by
	 * efa_rdm_ope_try_fill_desc in efa_rdm_proto_select_send_protocol
	 */

	txe->ep = ep;
	txe->type = EFA_RDM_TXE;
	txe->op = op;
	txe->tx_id = ofi_buf_index(txe);
	txe->state = EFA_RDM_TXE_REQ;
	txe->peer = peer;

	/* peer would be NULL for local read operation */
	if (txe->peer) {
		dlist_insert_tail(&txe->peer_entry, &txe->peer->txe_list);
	}

	txe->internal_flags = 0;
	txe->bytes_received = 0;
	txe->bytes_copied = 0;
	txe->bytes_acked = 0;
	txe->bytes_sent = 0;
	txe->window = 0;
	txe->iov_count = msg->iov_count;
	txe->rma_iov_count = 0;
	txe->msg_id = 0;
	txe->efa_outstanding_tx_ops = 0;
	dlist_init(&txe->queued_pkts);

	memcpy(txe->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);

	/* cq_entry on completion */
	txe->cq_entry.op_context = msg->context;
	txe->cq_entry.data = msg->data;
	txe->cq_entry.len = ofi_total_iov_len(txe->iov, txe->iov_count);
	txe->cq_entry.buf =
		OFI_LIKELY(txe->cq_entry.len > 0) ? txe->iov[0].iov_base : NULL;

	/* set flags */
	assert(ep->base_ep.util_ep.tx_msg_flags == 0 ||
	       ep->base_ep.util_ep.tx_msg_flags == FI_COMPLETION);
	tx_op_flags = ep->base_ep.util_ep.tx_op_flags;
	if (ep->base_ep.util_ep.tx_msg_flags == 0)
		tx_op_flags &= ~FI_COMPLETION;
	txe->fi_flags = flags | tx_op_flags;
	txe->bytes_runt = 0;
	dlist_init(&txe->entry);

	switch (op) {
	case ofi_op_tagged:
		txe->cq_entry.flags = FI_TRANSMIT | FI_MSG | FI_TAGGED;
		txe->cq_entry.tag = tag;
		txe->tag = tag;
		break;
	case ofi_op_write:
		txe->cq_entry.flags = FI_RMA | FI_WRITE;
		break;
	case ofi_op_read_req:
		txe->cq_entry.flags = FI_RMA | FI_READ;
		break;
	case ofi_op_msg:
		txe->cq_entry.flags = FI_TRANSMIT | FI_MSG;
		break;
	case ofi_op_atomic:
		txe->cq_entry.flags = (FI_WRITE | FI_ATOMIC);
		break;
	case ofi_op_atomic_fetch:
	case ofi_op_atomic_compare:
		txe->cq_entry.flags = (FI_READ | FI_ATOMIC);
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "invalid operation type\n");
		assert(0);
	}

	dlist_insert_tail(&txe->ep_entry, &ep->txe_list);
}
