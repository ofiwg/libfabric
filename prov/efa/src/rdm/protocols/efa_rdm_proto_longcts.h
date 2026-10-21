/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_LONGCTS_H
#define _EFA_RDM_PROTO_LONGCTS_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_longcts;

void efa_rdm_proto_longcts_handle_cts_recv(struct efa_rdm_pke *pkt_entry);
void efa_rdm_proto_longcts_handle_ctsdata_recv(struct efa_rdm_pke *pkt_entry);

int efa_rdm_proto_longcts_construct_req_pke(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *txe,
					    uint64_t *pke_send_flags);

void efa_rdm_proto_longcts_handle_req_posted(
	struct efa_rdm_ep *ep, struct efa_rdm_ope *txe);

ssize_t efa_rdm_proto_longcts_handle_req_send_completion(
	struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_proto_ctsdata_handle_send_completion(
	struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_proto_post_ctsdata(struct efa_rdm_ope *ope);

#endif /* _EFA_RDM_PROTO_LONGCTS_H */
