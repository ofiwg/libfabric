/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longread.h"

#include "efa_rdm_ope.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_tracepoint.h"

static ssize_t
efa_rdm_proto_longread_process_received_pke_after_robuf(
	struct efa_rdm_pke *pkt_entry)
{
	return efa_rdm_pke_proc_rtm_after_robuf(pkt_entry,
					       &efa_rdm_proto_longread);
}

static ssize_t
efa_rdm_proto_longread_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);
	struct efa_rdm_longread_rtm_base_hdr *rtm_hdr =
		efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry);
	struct fi_rma_iov *read_iov = (struct fi_rma_iov *)
		(pkt_entry->wiredata + efa_rdm_pke_get_req_hdr_size(pkt_entry));

	rxe->tx_id = rtm_hdr->send_id;
	rxe->rma_iov_count = rtm_hdr->read_iov_count;
	memcpy(rxe->rma_iov, read_iov,
	       rxe->rma_iov_count * sizeof(struct fi_rma_iov));

	efa_rdm_tracepoint(longread_read_posted, rxe->msg_id,
			   (size_t) rxe->cq_entry.op_context, rxe->total_len);

	ssize_t ret =
		efa_rdm_pke_post_remote_read_or_nack(rxe->ep, pkt_entry, rxe);
	efa_rdm_pke_release_rx(pkt_entry);
	return ret;
}

struct efa_rdm_proto efa_rdm_proto_longread = {
	.process_received_pke_after_robuf =
		&efa_rdm_proto_longread_process_received_pke_after_robuf,
	.handle_unexp_pke_match = &efa_rdm_proto_longread_handle_matched_rtm,
};

void efa_rdm_proto_longread_handle_eor_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_eor_hdr *eor_hdr =
		(struct efa_rdm_eor_hdr *) pkt_entry->wiredata;

	/* Bounce buffers cannot retain the originating TXE directly. */
	struct efa_rdm_ope *txe =
		efa_rdm_ep_live_txe_from_id(pkt_entry->ep, eor_hdr->send_id);
	if (!txe) {
		EFA_INFO(FI_LOG_CQ,
			 "EOR names a send that is no longer live, dropping it\n");
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	efa_rdm_txe_release_read_msg_slot(txe);

	txe->bytes_acked += txe->total_len - txe->bytes_runt;
	if (txe->bytes_acked == txe->total_len) {
		efa_rdm_txe_report_completion(txe);
		/*
		 * The TXE is released either here or when the request packet's
		 * send completes, whichever happens last.
		 */
		txe->internal_flags |= EFA_RDM_TXE_REMOTE_ACK_RECEIVED;
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	}

	efa_rdm_pke_release_rx(pkt_entry);
}
