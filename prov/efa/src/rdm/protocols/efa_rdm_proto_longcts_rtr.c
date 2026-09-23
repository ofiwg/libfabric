/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts_rtr.h"
#include "efa.h"
#include "efa_env.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtr.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_protocol.h"

/*
 * List of packet types used by this protocol
 *
 * EFA_RDM_LONGCTS_RTR_PKT
 *
 * The response data arrives on the shared CTS/CTSDATA path, which is not owned
 * by this protocol.
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#emulated-longcts-read-featuresubprotocol
 */

/**
 * @brief Check if the long CTS read protocol can handle this operation.
 *
 * Long CTS read is the fallback for reads that do not fit in a single response
 * packet, so it always applies once short read has been ruled out.
 */
static bool efa_rdm_proto_longcts_rtr_can_use(struct efa_rdm_ope *txe,
					      int req_pkt_type,
					      uint16_t header_flags, int iface,
					      bool use_p2p)
{
	assert(txe->total_len >=
	       txe->ep->mtu_size - sizeof(struct efa_rdm_readrsp_hdr));
	return true;
}

/**
 * @brief Handle the send completion of a long CTS RTR packet.
 *
 * An emulated read is driven by the response data, not by the RTR send, so the
 * RTR send completion does not finish the read. Release the txe here only if
 * the response was already fully received; otherwise
 * efa_rdm_ope_handle_recv_completed() reports the completion and releases it,
 * whichever happens last.
 */
ssize_t efa_rdm_proto_longcts_rtr_handle_send_completion(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (efa_rdm_txe_emulated_read_ready_for_release(txe))
		efa_rdm_txe_release(txe);

	efa_rdm_pke_release_tx(pkt_entry);
	return 0;
}

/**
 * @brief initialize a EFA_RDM_LONGCTS_RTR_PKT
 *
 * @param[in]		pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA read information
 */
static ssize_t efa_rdm_pke_init_longcts_rtr(struct efa_rdm_pke *pkt_entry,
					    struct efa_rdm_ope *txe)
{
	efa_rdm_pke_init_rtr_common(pkt_entry,
				    EFA_RDM_LONGCTS_RTR_PKT,
				    txe,
				    txe->window);
	return 0;
}

/**
 * @brief Construct the TX packet entry for the long CTS read protocol.
 *
 * Sets the flow-control window, allocates a single RTR packet entry describing
 * the remote memory to read, and sets the per-packet send completion callback.
 * The RTR has no payload, so FI_MORE is not honored. On success
 * ep->send_pkt_entry_vec[0] holds the packet entry.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longcts_rtr_construct_tx_pkes(struct efa_rdm_ep *ep,
						struct efa_rdm_ope *txe,
						uint64_t *pke_send_flags)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;

	*pke_send_flags = 0;

	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	assert(efa_env.tx_min_credits > 0);
	txe->window = MIN(txe->total_len,
			  efa_env.tx_min_credits * ep->max_data_payload_size);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = txe->peer;
	pkt_entry->handle_pke =
		&efa_rdm_proto_longcts_rtr_handle_send_completion;

	ret = efa_rdm_pke_init_longcts_rtr(pkt_entry, txe);
	if (ret)
		goto err;

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;
	return FI_SUCCESS;

err:
	efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}

/**
 * @brief Send the next batch of read-response data for a long CTS read.
 *
 * Called from the progress engine when the peer has granted more window. The
 * response data is carried on the shared CTSDATA path.
 *
 * @return 0 on success, negative errno on failure
 */
static ssize_t efa_rdm_proto_longcts_rtr_post_continuation(struct efa_rdm_ope *ope)
{
	return efa_rdm_ope_post_send(ope, EFA_RDM_CTSDATA_PKT);
}

struct efa_rdm_proto efa_rdm_proto_longcts_rtr = {
	.name = "longcts_rtr",
	.can_use_protocol = &efa_rdm_proto_longcts_rtr_can_use,
	.construct_tx_pkes = &efa_rdm_proto_longcts_rtr_construct_tx_pkes,
	.post_continuation = &efa_rdm_proto_longcts_rtr_post_continuation,
	.req_pkt_type = EFA_RDM_LONGCTS_RTR_PKT,
	.req_pkt_type_dc = EFA_RDM_LONGCTS_RTR_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};
