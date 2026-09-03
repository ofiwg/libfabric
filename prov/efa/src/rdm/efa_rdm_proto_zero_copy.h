/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_ZERO_COPY_H
#define _EFA_RDM_PROTO_ZERO_COPY_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_zero_copy;

void efa_rdm_proto_zero_copy_reselect_queued_before_handshake(
	struct efa_rdm_ope *txe);

int efa_rdm_proto_zero_copy_construct_tx_pkes(struct efa_rdm_ep *ep,
					      struct efa_rdm_peer *peer,
					      const struct fi_msg *msg,
					      uint32_t op, uint64_t tag,
					      uint64_t flags,
					      uint32_t internal_flags,
					      struct efa_rdm_ope *txe);

void efa_rdm_proto_zero_copy_handle_send_completion(
	struct efa_rdm_pke *pkt_entry);

#endif /* _EFA_RDM_PROTO_ZERO_COPY_H */
