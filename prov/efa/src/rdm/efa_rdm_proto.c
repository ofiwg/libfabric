/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto.h"

#define EFA_RDM_MAX_PROTO 8

/* List of supported protocols.
 * The protocols listed here will be tried in the order they're listed.
 * The first protocol that can be used for the TX operation will be used.
 */
struct efa_rdm_proto *efa_rdm_protocols [EFA_RDM_MAX_PROTO] = {
	&efa_rdm_proto_eager,
	NULL,	/* Sentinel used to stop iteration */
};

int efa_rdm_proto_select_send_protocol(struct efa_rdm_proto **proto,
				       struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags)
{
	/* TODO: Handle memory registration of user buffers.
	 * If MR fails, switch to a different protocol.
 	 */

	size_t max_data_offset, max_rtm_data_capacity;
	struct efa_rdm_proto *selected_proto;
	int req_pkt_type, iface, err, use_p2p;
	size_t total_len;
	uint16_t header_flags = 0;

	bool tagged, delivery_complete_requested;

	if (flags & FI_INJECT ||
	    efa_both_support_zero_hdr_data_transfer(ep, peer))
		delivery_complete_requested = false;
	else
		delivery_complete_requested = flags & FI_DELIVERY_COMPLETE;

	tagged = (op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	iface = (msg->desc && msg->desc[0]) ? ((struct efa_mr*) msg->desc[0])->peer.iface : FI_HMEM_SYSTEM;

	err = efa_rdm_ep_use_p2p(ep, (msg->desc && msg->desc[0]) ? msg->desc[0] : NULL);
	if (err < 0)
		return err;
	use_p2p = err;

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

		// TODO: Make pkt headers independent of tag and DC to avoid these checks
		req_pkt_type = delivery_complete_requested ?
			selected_proto->req_pkt_type_dc + tagged :
			selected_proto->req_pkt_type + tagged;

		/* Logic copied from efa_rdm_txe_max_req_data_capacity */
		if (efa_rdm_peer_need_raw_addr_hdr(peer))
			header_flags |= EFA_RDM_REQ_OPT_RAW_ADDR_HDR;

		else if (efa_rdm_peer_need_connid(peer))
			header_flags |= EFA_RDM_PKT_CONNID_HDR;

		if (flags & FI_REMOTE_CQ_DATA)
			header_flags |= EFA_RDM_REQ_OPT_CQ_DATA_HDR;

		/* TODO: For emulated read and atomics, need to consider RMA
		 * IOVs in the header
		 * https://github.com/ofiwg/libfabric/blob/cff899c9ef6dd823a1e3b35d3205622013c6eb6c/prov/efa/src/rdm/efa_rdm_pkt_type.c#L101-L103
		 */
		max_data_offset = efa_rdm_pkt_type_get_req_hdr_size(
			req_pkt_type, header_flags, 0);

		max_rtm_data_capacity = ep->mtu_size - max_data_offset;

		if (total_len <= max_rtm_data_capacity) {
			/* Can only be true for the eager protocol */
			assert(selected_proto == &efa_rdm_proto_eager);
			*proto = selected_proto;
			return FI_SUCCESS;
		}
	}

	*proto = NULL;
	return FI_SUCCESS;
}
