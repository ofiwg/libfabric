/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_RUNTREAD_H
#define _EFA_RDM_PROTO_RUNTREAD_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_runtread;

int efa_rdm_proto_runtread_construct_tx_pkes(struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t tag, uint64_t flags,
				       struct efa_rdm_ope *txe);

void efa_rdm_proto_runtread_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry);

void efa_rdm_proto_runtread_handle_rtm_dc_send_completion(
	struct efa_rdm_pke *pkt_entry);

void efa_rdm_proto_runtread_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						struct efa_rdm_ope *txe);

#endif /* _EFA_RDM_PROTO_RUNTREAD_H */
