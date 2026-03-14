/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_req.h"


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


struct efa_rdm_proto efa_rdm_proto_eager = {
	.construct_pkes = &efa_rdm_proto_eager_construct_pkes,
	.req_pkt_type = EFA_RDM_EAGER_MSGRTM_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_EAGER_MSGRTM_PKT,
	.req_pkt_type_tagged = EFA_RDM_EAGER_TAGRTM_PKT,
	.req_pkt_type_tagged_dc = EFA_RDM_DC_EAGER_TAGRTM_PKT,
	.send_pkes_posted = &efa_rdm_proto_send_pkes_posted_no_op,
};


/* TX path callbacks - one callback for each packet type that this protocol uses */
void efa_rdm_proto_eager_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	return;
}

int efa_rdm_proto_eager_construct_txe(struct efa_rdm_ope **txe,
				      struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
				      const struct fi_msg *msg, uint32_t op, uint64_t tag,
				      uint64_t flags)
{
	return FI_SUCCESS;
}

int efa_rdm_proto_eager_construct_pkes(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t tag, uint64_t flags, struct efa_rdm_ope **txe)
{
	return FI_SUCCESS;
}
