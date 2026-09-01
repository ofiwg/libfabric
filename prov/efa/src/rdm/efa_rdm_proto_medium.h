/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_MEDIUM_H
#define _EFA_RDM_PROTO_MEDIUM_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_medium;

/* Exposed so the unit tests can drive the segmenting decision on its own, over
 * interfaces and alignments a test cannot set up real memory for.
 */
ssize_t efa_rdm_proto_medium_plan_tx_pkes(struct efa_rdm_ep *ep,
					 struct efa_rdm_ope *txe,
					 int req_pkt_type,
					 size_t *pkt_entry_cnt,
					 size_t *pkt_entry_data_size_vec);

int efa_rdm_proto_medium_construct_tx_pkes(struct efa_rdm_ep *ep,
					   struct efa_rdm_peer *peer,
					   const struct fi_msg *msg, uint32_t op,
					   uint64_t tag, uint64_t flags,
					   uint32_t internal_flags,
					   struct efa_rdm_ope *txe);

void efa_rdm_proto_medium_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
						struct efa_rdm_ope *txe);

void efa_rdm_proto_medium_handle_rtm_send_completion(
	struct efa_rdm_pke *pkt_entry);

#endif /* _EFA_RDM_PROTO_MEDIUM_H */
