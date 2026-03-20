/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"

int efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags, struct efa_rdm_ope *txe,
				       struct efa_rdm_proto **proto)
{
	return 0;
}

/* Utility funcions */

void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags)
{
	/* Logic copied from efa_rdm_txe_construct */
	uint64_t tx_op_flags;

	/* txe->mr, txe->desc, txe->total_len will be filled by
	 * efa_rdm_ope_try_fill_desc in efa_rdm_proto_select_send_protocol
	 */

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
