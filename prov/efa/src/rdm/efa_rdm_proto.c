/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto.h"
#include "efa.h"
#include "efa_rdm_proto_eager.h"

/* We have total of 5 protocols in the EFA provider. Use a slightly larger
 * number to accomodate the NULL sentinel and future protocols. */
#define EFA_RDM_MAX_PROTO 8

/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 */
struct efa_rdm_proto *efa_rdm_protocols[EFA_RDM_MAX_PROTO] = {
	&efa_rdm_proto_eager,
	NULL, /* Sentinel used to stop iteration */
};

int efa_rdm_proto_select_send_protocol(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags, struct efa_rdm_ope *txe,
				       struct efa_rdm_proto **proto)
{
	/* TODO: Handle memory registration of user buffers.
	 * If MR fails, switch to a different protocol.
	 */

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

	iface = (msg->desc && msg->desc[0]) ?
			((struct efa_mr *) msg->desc[0])->iface :
			FI_HMEM_SYSTEM;

	err = efa_rdm_ep_use_p2p(
		ep, (msg->desc && msg->desc[0]) ? msg->desc[0] : NULL);
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

		/* TODO: The req_pkt_type is again needed in each protocol when
		 * allocating pkes Option 1: Make pkt headers independent of tag
		 * and DC to avoid these checks Option 2: Store the req_pkt_type
		 * in the txe
		 */
		req_pkt_type =
			delivery_complete_requested ?
				selected_proto->req_pkt_type_dc + tagged :
				selected_proto->req_pkt_type + tagged;

		/* All protocols other than the eager protocol can benefit from
		 * registering the application buffers.
		 * TODO: Move function to efa_rdm_proto.c
		 */
		if (selected_proto->can_use_protocol_for_send(
			    txe, req_pkt_type, header_flags, iface)) {
			*proto = selected_proto;
			return FI_SUCCESS;
		}
	}

	*proto = NULL;
	return FI_SUCCESS;
}
