/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_eager.h"
#include "efa.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"

/*
 * List of packet types used by this protocol
 *
 * For send/recv operations
 * EFA_RDM_EAGER_MSGRTM_PKT
 * EFA_RDM_EAGER_TAGRTM_PKT
 * EFA_RDM_DC_EAGER_MSGRTM_PKT
 * EFA_RDM_DC_EAGER_TAGRTM_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#eager-message-featuresubprotocol
 */

/**
 * @brief Check if the eager protocol can handle this send operation.
 *
 * Returns true if the message fits in a single MTU-sized packet after
 * accounting for the request header size.
 */
static bool efa_rdm_proto_eager_can_use_for_send(struct efa_rdm_ope *txe,
						 int req_pkt_type,
						 uint16_t header_flags,
						 int iface)
{
	size_t max_data_offset, max_rtm_data_capacity;

	/* TODO: For emulated read and atomics, need to consider RMA
	 * IOVs in the header
	 * https://github.com/ofiwg/libfabric/blob/cff899c9ef6dd823a1e3b35d3205622013c6eb6c/prov/efa/src/rdm/efa_rdm_pkt_type.c#L101-L103
	 */
	max_data_offset = efa_rdm_pkt_type_get_req_hdr_size(req_pkt_type,
							    header_flags, 0);
	max_rtm_data_capacity = txe->ep->mtu_size - max_data_offset;

	return txe->total_len <= max_rtm_data_capacity;
}

struct efa_rdm_proto efa_rdm_proto_eager = {
	.can_use_protocol_for_send = &efa_rdm_proto_eager_can_use_for_send,
	.construct_tx_pkes = &efa_rdm_proto_eager_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_EAGER_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_EAGER_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_EAGER_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_EAGER_TAGRTM_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

/* TX path callbacks - one callback for each packet type that this protocol uses
 */
void efa_rdm_proto_eager_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	return;
}

int efa_rdm_proto_eager_construct_txe(struct efa_rdm_ope **txe,
				      struct efa_rdm_ep *ep,
				      struct efa_rdm_peer *peer,
				      const struct fi_msg *msg, uint32_t op,
				      uint64_t tag, uint64_t flags)
{
	return FI_SUCCESS;
}

int efa_rdm_proto_eager_construct_tx_pkes(struct efa_rdm_ep *ep,
					  struct efa_rdm_peer *peer,
					  const struct fi_msg *msg, uint32_t op,
					  uint64_t tag, uint64_t flags,
					  struct efa_rdm_ope *txe)
{
	return FI_SUCCESS;
}
