/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_H
#define _EFA_RDM_PROTO_H

#include "efa.h"
#include "efa_rdm_pkt_type.h"

/* The protocol interface */
struct efa_rdm_proto {
	/* TX path handlers */

	/* This function will allocate the pkes that need to be sent for a given
	 * TX operation. At the end of this function, ep->send_pkt_entry_vec
	 * will be correctly populated with the all of the pkes that need to be
	 * sent including copying the application data into the pke buffer if
	 * necessary. Each pke will have an appropriate callback function set to
	 * handle the TX completion of that pke. This function also constructs
	 * and returns the txe
	 */
	int (*construct_pkes)(struct efa_rdm_ep *ep, struct efa_rdm_peer *peer,
			      const struct fi_msg *msg, uint32_t op, uint64_t tag,
			      uint64_t flags, struct efa_rdm_ope **txe);

	/* This function is called after all pkes are posted to the EFA device.
	 * It is useful for some protocols: e.g. to register the buffer after
	 * posting a Long CTS RTM pke or to update the number of in flight reads
	 * and read bytes
	 */
	int (*send_pkes_posted)(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe);

	/* TX utitlities */
    int req_pkt_type;
    int req_pkt_type_tagged;
    int req_pkt_type_dc;
    int req_pkt_type_tagged_dc;
};

/* This function will select the appropriate protocol for a given TX operation.
 * It will also handle memory registration of user buffers. If read based
 * protocols are appropriate but MR fails, it will automatically switch to a
 * different protocol.
 */
int efa_rdm_proto_select_send_protocol(struct efa_rdm_proto **proto,
				       struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags);

/* Utility funcions */
static inline int efa_rdm_proto_send_pkes_posted_no_op(struct efa_rdm_ep *ep,
						       struct efa_rdm_ope *txe)
{
	return FI_SUCCESS;
};

#endif /* _EFA_RDM_PROTO_H */
