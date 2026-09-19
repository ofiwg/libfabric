/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts.h"

#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke_nonreq.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"

static ssize_t
efa_rdm_proto_longcts_process_received_pke_after_robuf(
	struct efa_rdm_pke *pkt_entry)
{
	return efa_rdm_pke_proc_rtm_after_robuf(pkt_entry,
					       &efa_rdm_proto_longcts);
}

static ssize_t
efa_rdm_proto_longcts_handle_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe = pkt_entry->ope;
#if ENABLE_DEBUG
	struct efa_rdm_ep *ep = pkt_entry->ep;
#endif

	efa_rdm_pke_prepare_matched_rtm(pkt_entry);
	rxe->tx_id =
		efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)->send_id;
	rxe->bytes_received += pkt_entry->payload_size;
	ssize_t ret = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
	if (ret)
		return ret;

#if ENABLE_DEBUG
	dlist_insert_tail(&rxe->pending_recv_entry, &ep->ope_recv_list);
	ep->pending_recv_counter++;
#endif
	rxe->state = EFA_RDM_RXE_RECV;
	return efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_CTS_PKT);
}

void efa_rdm_proto_longcts_handle_cts_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep = pkt_entry->ep;
	struct efa_rdm_cts_hdr *cts_pkt =
		(struct efa_rdm_cts_hdr *) pkt_entry->wiredata;

	/*
	 * Drop a CTS whose id no longer names the operation that created it.
	 */
	struct efa_rdm_ope *ope =
		efa_rdm_ep_live_ope_from_id(ep, cts_pkt->send_id);
	if (OFI_UNLIKELY(!ope)) {
		EFA_INFO(FI_LOG_CQ,
			 "CTS names ope id %" PRIu32 ", which no longer holds "
			 "the operation that requested it. Dropping the CTS.\n",
			 cts_pkt->send_id);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	ope->rx_id = cts_pkt->recv_id;
	ope->window = cts_pkt->recv_length;
	assert(ope->window > 0);

	efa_rdm_pke_release_rx(pkt_entry);

	if (ope->state != EFA_RDM_OPE_SEND) {
		ope->state = EFA_RDM_OPE_SEND;
		dlist_insert_tail(&ope->entry, &ep->ope_longcts_send_list);
		efa_rdm_ep_enqueue_progress_list(ep);
	}
}

void efa_rdm_proto_longcts_handle_ctsdata_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ctsdata_hdr *data_hdr =
		efa_rdm_pke_get_ctsdata_hdr(pkt_entry);
	struct efa_rdm_ope *ope =
		efa_rdm_ep_get_ope_from_ope_id(pkt_entry->ep, data_hdr->recv_id);
	size_t hdr_size = sizeof(struct efa_rdm_ctsdata_hdr);

	if (data_hdr->flags & EFA_RDM_PKT_CONNID_HDR)
		hdr_size += sizeof(struct efa_rdm_ctsdata_opt_connid_hdr);

	efa_rdm_pke_proc_ctsdata(pkt_entry, ope,
				 pkt_entry->wiredata + hdr_size,
				 data_hdr->seg_offset,
				 data_hdr->seg_length);
}

struct efa_rdm_proto efa_rdm_proto_longcts = {
	.process_received_pke_after_robuf =
		&efa_rdm_proto_longcts_process_received_pke_after_robuf,
	.handle_unexp_pke_match = &efa_rdm_proto_longcts_handle_matched_rtm,
};
