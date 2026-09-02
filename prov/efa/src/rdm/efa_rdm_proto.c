/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"
#include "efa_rdm_proto_eager.h"
#include "efa_rdm_msg.h"

/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 */
struct efa_rdm_proto *efa_rdm_protocols[] = {
	&efa_rdm_proto_eager,
};

void efa_rdm_proto_txe_init_buffers(struct efa_rdm_ep *ep,
						  const struct fi_msg *msg,
						  struct efa_rdm_ope *txe)
{
	txe->ep = ep;
	txe->iov_count = msg->iov_count;
	memcpy(txe->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);
	memset(txe->mr, 0, sizeof(*txe->mr) * msg->iov_count);

	efa_rdm_mr_gen_init_ope_desc(txe);
	if (msg->desc) {
		memcpy(txe->desc, msg->desc, sizeof(*msg->desc) * msg->iov_count);
		efa_rdm_mr_gen_capture_in_ope_desc(txe);
	} else {
		memset(txe->desc, 0, sizeof(*txe->desc) * msg->iov_count);
	}
	txe->total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);
}

void efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
					struct efa_rdm_peer *peer,
					const struct fi_msg *msg, uint32_t op,
					uint64_t flags, struct efa_rdm_ope *txe,
					struct efa_rdm_proto **proto)
{
	/* TODO: Handle memory registration of user buffers.
	 * If MR fails, switch to a different protocol.
	 */

	struct efa_rdm_proto *selected_proto;
	int req_pkt_type, iface;
	uint16_t header_flags = 0;
	uint64_t effective_flags;

	/*
	 * efa_rdm_msg_generic_send() sends a peer that only accepts headerless
	 * packets straight to the zero-copy protocol, so every protocol
	 * considered here writes a REQ header.
	 */
	assert(!efa_rdm_peer_expects_zero_hdr_data_transfer(peer));

	/*
	 * Resolve the endpoint's tx_op_flags now: an endpoint-level
	 * FI_DELIVERY_COMPLETE must steer protocol selection just as a
	 * per-operation flag does.
	 */
	effective_flags = efa_rdm_msg_get_tx_flags(ep, flags);

	efa_rdm_proto_txe_init_buffers(ep, msg, txe);

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->iface :
			FI_HMEM_SYSTEM;

	/* Logic copied from efa_rdm_txe_max_req_data_capacity */
	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= EFA_RDM_PKT_CONNID_HDR;

	if (flags & FI_REMOTE_CQ_DATA)
		header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

	for (int i = 0; i < ARRAY_SIZE(efa_rdm_protocols); ++i) {
		selected_proto = efa_rdm_protocols[i];

		req_pkt_type = efa_rdm_proto_req_pkt_type(
			selected_proto, op, effective_flags, peer);

		/* All protocols other than the eager protocol can benefit from
		 * registering the application buffers.
		 * TODO: Move function to efa_rdm_proto.c
		 */
		if (selected_proto->can_use_protocol_for_send(
			    txe, req_pkt_type, header_flags, iface)) {
			*proto = selected_proto;
			txe->proto = selected_proto;
			return;
		}
	}

	/*
	 * No protocol matched, so the message is larger than a single eager
	 * packet and the caller falls back to the old code path.
	 */
	*proto = NULL;
	txe->proto = NULL;
}

/* Utility funcions */

void efa_rdm_proto_txe_fill(struct efa_rdm_ope *txe, struct efa_rdm_ep *ep,
			    struct efa_rdm_peer *peer, const struct fi_msg *msg,
			    uint32_t op, uint64_t tag, uint64_t flags,
			    uint32_t internal_flags, struct efa_rdm_proto *proto)
{
	/*
	 * txe->mr, txe->desc and the MR generation snapshot were already
	 * populated by efa_rdm_proto_select_send_protocol(), which needs them
	 * to decide whether a protocol can be used, so use the construct
	 * helper that leaves them alone.
	 */
	efa_rdm_txe_construct_common(txe, ep, peer, msg, op, flags,
				     internal_flags);

	txe->proto = proto;

	if (op == ofi_op_tagged) {
		txe->cq_entry.tag = tag;
		txe->tag = tag;
	}
}
