/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_runtread.h"

#include "efa_rdm_ope.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_tracepoint.h"

static ssize_t
efa_rdm_proto_runtread_process_received_pke_after_robuf(
	struct efa_rdm_pke *pkt_entry)
{
	return efa_rdm_pke_proc_rtm_after_robuf(pkt_entry,
					       &efa_rdm_proto_runtread);
}

static ssize_t
efa_rdm_proto_runtread_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);
	struct efa_rdm_runtread_rtm_base_hdr *rtm_hdr =
		efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry);
	rxe->bytes_runt = rtm_hdr->runt_length;
	if (rxe->total_len > rxe->bytes_runt &&
	    rxe->bytes_read_total_len == 0) {
		rxe->tx_id = rtm_hdr->send_id;
		struct fi_rma_iov *read_iov = (struct fi_rma_iov *)
			(pkt_entry->wiredata +
			 efa_rdm_pke_get_req_hdr_size(pkt_entry));
		rxe->rma_iov_count = rtm_hdr->read_iov_count;
		memcpy(rxe->rma_iov, read_iov,
		       rxe->rma_iov_count * sizeof(struct fi_rma_iov));
		efa_rdm_tracepoint(runtread_read_posted, rxe->msg_id,
				   (size_t) rxe->cq_entry.op_context,
				   rxe->total_len);

		ssize_t ret = efa_rdm_pke_post_remote_read_or_nack(
			pkt_entry->ep, pkt_entry, rxe);
		if (ret) {
			efa_rdm_pke_release_rx_list(pkt_entry);
			return ret;
		}
	}

	return efa_rdm_pke_proc_matched_mulreq_rtm(pkt_entry);
}

struct efa_rdm_proto efa_rdm_proto_runtread = {
	.process_received_pke_after_robuf =
		&efa_rdm_proto_runtread_process_received_pke_after_robuf,
	.handle_unexp_pke_match = &efa_rdm_proto_runtread_handle_matched_rtm,
};
