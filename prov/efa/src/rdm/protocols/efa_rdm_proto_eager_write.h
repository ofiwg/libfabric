/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_EAGER_WRITE_H
#define _EFA_RDM_PROTO_EAGER_WRITE_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_eager_write;

int efa_rdm_proto_eager_write_construct_tx_pkes(struct efa_rdm_ep *ep,
						struct efa_rdm_peer *peer,
						uint32_t op, uint64_t tag,
						uint64_t flags,
						uint32_t internal_flags,
						struct efa_rdm_ope *txe,
						uint64_t *pke_send_flags);

void efa_rdm_proto_eager_write_handle_rtw_send_completion(
	struct efa_rdm_pke *pkt_entry);

ssize_t efa_rdm_proto_eager_write_init_rtw(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe);

void efa_rdm_pke_handle_eager_rtw_send_completion(struct efa_rdm_pke *pkt_entry);

void efa_rdm_proto_eager_write_handle_rtw_recv(struct efa_rdm_pke *pkt_entry);

void efa_rdm_proto_eager_write_handle_dc_rtw_recv(struct efa_rdm_pke *pkt_entry);

#endif /* _EFA_RDM_PROTO_EAGER_WRITE_H */
