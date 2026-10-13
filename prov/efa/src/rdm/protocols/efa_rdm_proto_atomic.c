/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_rdm_proto_atomic.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_peer.h"
#include "efa_rdm_pke_rta.h"
#include <ofi_proto.h>
#include <rdma/fi_errno.h>

/*
 * List of packet types used by this protocol
 *
 * EFA_RDM_WRITE_RTA_PKT
 * EFA_RDM_DC_WRITE_RTA_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/**
 * @brief select the request packet type of an atomic TXE
 *
 * FI_DELIVERY_COMPLETE takes precedence over FI_INJECT: inject only
 * controls source-buffer reuse, while delivery complete controls when the
 * outcome is reported.
 */
static int efa_rdm_proto_atomic_req_pkt_type(struct efa_rdm_ope *txe){
	const struct efa_rdm_atomic_proto *proto = txe->atomic_proto;
	return (txe->fi_flags & FI_DELIVERY_COMPLETE) ? proto->req_pkt_type_dc :
							proto->req_pkt_type;
}

/**
 * @brief handle the send completion of a WRITE_RTA or DC_WRITE_RTA packet
 *
 * @param[in,out]	pkt_entry	request packet entry
 * @returns		FI_SUCCESS
 */
static ssize_t efa_rdm_proto_atomic_write_handle_send_completion(struct efa_rdm_pke *pkt_entry){
	struct efa_rdm_ope *txe = pkt_entry->ope;

	if (txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED) {
		/* RECEIPT reports completion. Send completion gates TXE
		 * release. */
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else {
		efa_rdm_ope_handle_send_completed(txe);
	}

	efa_rdm_pke_release_tx(pkt_entry);
	return FI_SUCCESS;
}

static size_t efa_rdm_proto_atomic_write_get_req_data_size(struct efa_rdm_ope *txe){
	return txe->total_len;
}

static ssize_t efa_rdm_proto_atomic_write_init_req_pke(struct efa_rdm_pke *pkt_entry,
					struct efa_rdm_ope *txe){
	if (txe->req_pkt_type == EFA_RDM_DC_WRITE_RTA_PKT)
		return efa_rdm_pke_init_dc_write_rta(pkt_entry, txe);

	assert(txe->req_pkt_type == EFA_RDM_WRITE_RTA_PKT);
	return efa_rdm_pke_init_write_rta(pkt_entry, txe);
}

const struct efa_rdm_atomic_proto efa_rdm_atomic_write_proto = {
	.req_pkt_type = EFA_RDM_WRITE_RTA_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_WRITE_RTA_PKT,
	.get_req_data_size = &efa_rdm_proto_atomic_write_get_req_data_size,
	.init_req_pke = &efa_rdm_proto_atomic_write_init_req_pke,
	.handle_send_completion =
		&efa_rdm_proto_atomic_write_handle_send_completion,
};

/**
 * @brief construct the single request PKE of an atomic TXE
 *
 * The whole operand must fit in one packet. On failure, only the PKE
 * allocated here is released; the caller still owns the TXE.
 *
 * @param[in,out]	txe	atomic TXE with req_pkt_type selected
 * @returns
 * 0 on success, with the PKE in ep->send_pkt_entry_vec[0]
 * -FI_EMSGSIZE if the operand does not fit in one packet
 * -FI_EAGAIN if no TX packet is available
 * other negative libfabric error code if the PKE initialization fails
 */
static ssize_t efa_rdm_proto_atomic_construct_tx_pke(struct efa_rdm_ope *txe){
	struct efa_rdm_ep *ep = txe->ep;
	struct efa_rdm_pke *pkt_entry;
	size_t max_payload;
	size_t req_data_size;
	ssize_t err;

	assert(txe->atomic_proto->get_req_data_size);
	assert(txe->atomic_proto->init_req_pke);
	assert(txe->atomic_proto->handle_send_completion);

	req_data_size = txe->atomic_proto->get_req_data_size(txe);
	max_payload =
		efa_rdm_txe_max_req_data_capacity(ep, txe, txe->req_pkt_type);
	if (OFI_UNLIKELY(req_data_size > max_payload))
		return -FI_EMSGSIZE;

	if (efa_rdm_ep_get_available_tx_pkts(ep) == 0)
		return -FI_EAGAIN;

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      ep->efa_tx_bounce_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	err = txe->atomic_proto->init_req_pke(pkt_entry, txe);
	if (OFI_UNLIKELY(err)) {
		efa_rdm_pke_release_tx(pkt_entry);
		return err;
	}

	/* Install the callback before posting: the completion may arrive at any
	 * time after. */
	pkt_entry->handle_pke = txe->atomic_proto->handle_send_completion;
	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;
	return FI_SUCCESS;
}

ssize_t efa_rdm_proto_atomic_post(struct efa_rdm_ope *txe)
{
	struct efa_rdm_ep *ep = txe->ep;
	ssize_t err;

	assert(txe->atomic_proto);
	assert(!txe->proto);
	assert(txe->op == ofi_op_atomic);

	txe->req_pkt_type = efa_rdm_proto_atomic_req_pkt_type(txe);
	err = efa_rdm_proto_atomic_construct_tx_pke(txe);
	if (err)
		return err;

	err = efa_rdm_pke_sendv(ep->send_pkt_entry_vec,
				ep->send_pkt_entry_vec_size, 0);
	if (OFI_UNLIKELY(err)) {
		/* No completion follows a failed post, so release the unposted
		 * PKE here. */
		efa_rdm_pke_release_tx(ep->send_pkt_entry_vec[0]);
		return err;
	}

	txe->peer->flags |= EFA_RDM_PEER_REQ_SENT;
	return FI_SUCCESS;
}
