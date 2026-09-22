/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longread.h"

#include "efa_rdm_ope.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_tracepoint.h"

static ssize_t
efa_rdm_proto_longread_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr =
		efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
	struct fi_rma_iov *read_iov = (struct fi_rma_iov *)
		(pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry));
	ssize_t ret;

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);

	rxe->tx_id = rtm_hdr->send_id;
	rxe->rma_iov_count = rtm_hdr->read_iov_count;
	memcpy(rxe->rma_iov, read_iov,
	       rxe->rma_iov_count * sizeof (struct fi_rma_iov));

	efa_rdm_tracepoint(longread_read_posted, rxe->msg_id,
			   (size_t) rxe->cq_entry.op_context, rxe->total_len);

	ret = efa_rdm_pke_post_remote_read_or_nack(rxe->ep, pkt_entry, rxe);
	efa_rdm_pke_release_rx(pkt_entry);
	return ret;
}

EFA_RDM_PROTO_DEF(longread,
	.handle_unexp_pke_match = &efa_rdm_proto_longread_handle_matched_rtm,
);
