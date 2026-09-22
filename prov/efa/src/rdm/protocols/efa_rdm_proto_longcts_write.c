/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longcts_write.h"
#include "ofi_iov.h"
#include "ofi_proto.h"
#include "efa_errno.h"
#include "efa.h"
#include "efa_env.h"
#include "efa_hmem.h"
#include "efa_base_ep.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_rma.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtw.h"
#include "efa_rdm_proto_write.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke_req.h"

/*
 * List of packet types used by this protocol
 *
 * EFA_RDM_LONGCTS_RTW_PKT
 * EFA_RDM_DC_LONGCTS_RTW_PKT
 *
 * The remaining payload is pulled by the peer's CTS on the shared CTSDATA path,
 * which is not owned by this protocol.
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#emulated-longcts-write-featuresubprotocol
 */

/* TX path functions */

/**
 * @brief initialize the header of a LONGCTS RTW packet
 *
 * This function applies to both EFA_RDM_LONGCTS_RTW_PKT and
 * EFA_RDM_DC_LONGCTS_RTW_PKT
 */
static inline
void efa_rdm_proto_longcts_write_init_rtw_hdr(struct efa_rdm_pke *pkt_entry,
					      int pkt_type,
					      struct efa_rdm_ope *txe)
{
	struct efa_rdm_longcts_rtw_hdr *rtw_hdr;

	rtw_hdr = (struct efa_rdm_longcts_rtw_hdr *)pkt_entry->wiredata;
	rtw_hdr->rma_iov_count = txe->rma_iov_count;
	rtw_hdr->msg_length = txe->total_len;
	rtw_hdr->send_id = txe->tx_id;
	rtw_hdr->credit_request = efa_env.tx_min_credits;
	efa_rdm_pke_init_req_hdr_common(pkt_entry, pkt_type, txe);
}

/**
 * @brief initialize a EFA_RDM_LONGCTS_RTW packet
 *
 *
 * @param[in,out]	pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA write information
 * @returns
 * 0 on success.
 * negative libfabric error code on failure
 */
static ssize_t efa_rdm_proto_longcts_write_init_rtw(struct efa_rdm_pke *pkt_entry,
						    struct efa_rdm_ope *txe)
{
	struct efa_rdm_longcts_rtw_hdr *rtw_hdr;

	assert(txe->op == ofi_op_write);

	rtw_hdr = (struct efa_rdm_longcts_rtw_hdr *)pkt_entry->wiredata;
	efa_rdm_proto_longcts_write_init_rtw_hdr(pkt_entry, EFA_RDM_LONGCTS_RTW_PKT, txe);
	return efa_rdm_proto_write_rtw_pke_init_common(pkt_entry, txe, rtw_hdr->rma_iov);
}

/**
 * @brief initialize a EFA_RDM_DC_LONGCTS_RTW packet
 *
 * DC means delivery complete
 * @param[in,out]	pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA write information
 * @returns
 * 0 on success.
 * negative libfabric error code on failure
 */
static ssize_t efa_rdm_proto_longcts_write_init_dc_rtw(struct efa_rdm_pke *pkt_entry,
						       struct efa_rdm_ope *txe)
{
	struct efa_rdm_longcts_rtw_hdr *rtw_hdr;

	assert(txe->op == ofi_op_write);

	txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	rtw_hdr = (struct efa_rdm_longcts_rtw_hdr *)pkt_entry->wiredata;
	efa_rdm_proto_longcts_write_init_rtw_hdr(pkt_entry, EFA_RDM_DC_LONGCTS_RTW_PKT, txe);
	return efa_rdm_proto_write_rtw_pke_init_common(pkt_entry, txe, rtw_hdr->rma_iov);
}

/**
 * @brief handle the "send completion" event of a non-DC LONGCTS RTW packet
 *
 * @param[in]	pkt_entry	LONGCTS RTW packet entry
 */
static void efa_rdm_proto_longcts_write_handle_rtw_send_completion(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	/**
	 * A zero-payload longcts rtw pkt currently should only happen when it's
	 * used for the READ NACK protocol. In this case, this pkt doesn't
	 * contribute to the send completion, and the associated tx entry
	 * may be released earlier as the CTSDATA pkts have already kicked off
	 * and finished the send.
	 */
	if (pkt_entry->payload_size == 0) {
		assert(efa_rdm_pke_get_rtw_base_hdr(pkt_entry)->flags & EFA_RDM_REQ_READ_NACK);
		return;
	}

	txe = pkt_entry->ope;
	txe->bytes_acked += pkt_entry->payload_size;
	if (txe->total_len == txe->bytes_acked)
		efa_rdm_ope_handle_send_completed(txe);
}

/**
 * @brief Handle the send completion of a LONGCTS RTW packet.
 *
 * Dispatches to the delivery complete or transmit complete completion path. A
 * delivery complete write must also wait for the peer's RECEIPT, so it only
 * releases the TXE here if that RECEIPT already arrived.
 */
void efa_rdm_proto_longcts_write_handle_send_completion(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);

	if (txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED) {
		if (txe->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
			efa_rdm_txe_progress_peer_abort_if_drained(txe);
		else if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else {
		efa_rdm_proto_longcts_write_handle_rtw_send_completion(pkt_entry);
	}

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Bookkeeping after the initial LONGCTS RTW packet has been posted.
 *
 * The peer pulls the remaining payload with CTS, so account for the bytes in
 * the initial packet and pre-fill the source descriptor for the CTSDATA pkts.
 */
static void efa_rdm_proto_longcts_write_handle_tx_pkes_posted(struct efa_rdm_ep *ep,
							      struct efa_rdm_ope *txe)
{
	struct efa_rdm_pke *pkt_entry = ep->send_pkt_entry_vec[0];

	txe->bytes_sent += pkt_entry->payload_size;
	assert(txe->bytes_sent < txe->total_len);
	if (efa_is_cache_available(efa_rdm_ep_rdm_domain(ep)))
		efa_rdm_ope_try_fill_desc(txe, 0, FI_SEND);
}

/**
 * @brief Construct the TX packet entry for the long CTS write protocol.
 *
 * Allocates the initial RTW packet entry, writes the remote memory region and
 * the first segment of payload into it, and sets the per-packet send completion
 * callback. The remaining payload is sent later on the CTSDATA path. On success
 * ep->send_pkt_entry_vec[0] holds the packet entry.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longcts_write_construct_tx_pkes(struct efa_rdm_ep *ep,
						  struct efa_rdm_peer *peer,
						  const struct fi_msg *msg,
						  uint32_t op, uint64_t tag,
						  uint64_t flags,
						  uint32_t internal_flags,
						  struct efa_rdm_ope *txe,
						  uint64_t *pke_send_flags)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;

	/* Long CTS write posts multiple packets, so it does not honor FI_MORE. */
	*pke_send_flags = 0;

	assert(ep->efa_max_outstanding_tx_ops - ep->efa_outstanding_tx_ops -
		       ep->efa_rnr_queued_pkt_cnt >
	       0);

	pkt_entry = efa_rdm_pke_alloc(ep, ep->efa_tx_pkt_pool,
				      EFA_RDM_PKE_FROM_EFA_TX_POOL);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	efa_rdm_pke_set_ope(pkt_entry, txe);
	pkt_entry->peer = peer;
	pkt_entry->handle_pke =
		&efa_rdm_proto_longcts_write_handle_send_completion;

	if (txe->req_pkt_type == efa_rdm_proto_longcts_write.req_pkt_type_dc)
		ret = efa_rdm_proto_longcts_write_init_dc_rtw(pkt_entry, txe);
	else
		ret = efa_rdm_proto_longcts_write_init_rtw(pkt_entry, txe);
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
 * @brief Check if the long CTS write protocol can handle this operation.
 *
 * Long CTS write is the fallback for writes that neither fit in a single packet
 * nor can use the read based protocol. SynapseAI is served by the read based
 * protocol, so it is rejected here.
 */
static bool efa_rdm_proto_longcts_write_can_use(struct efa_rdm_ope *txe,
						int req_pkt_type,
						uint16_t header_flags, int iface,
						bool use_p2p)
{
	/* TODO: remove interface-specific run-time checks for protocol usage */
	if (iface == FI_HMEM_SYNAPSEAI)
		return false;

	assert(txe->total_len > txe->ep->mtu_size -
	       efa_rdm_pkt_type_get_req_hdr_size(req_pkt_type, header_flags,
						 txe->rma_iov_count));
	assert(!efa_rdm_rma_should_write_using_longread(txe->ep, txe, txe->peer,
							use_p2p));
	return true;
}

struct efa_rdm_proto efa_rdm_proto_longcts_write = {
	.name = "longcts_write",
	.wants_mr = false,
	.can_use_protocol = &efa_rdm_proto_longcts_write_can_use,
	.construct_tx_pkes = &efa_rdm_proto_longcts_write_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_LONGCTS_RTW_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_LONGCTS_RTW_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_longcts_write_handle_tx_pkes_posted,
};

/* RX path functions */

/**
 * @brief handle the event that a LONGCTS RTW packet has been received
 *
 * applies to both EFA_RDM_LONGCTS_RTW_PKT and EFA_RDM_DC_LONGCTS_RTW_PKT
 *
 * @param[in]	pkt_entry	received LONGCTS RTW paket entry
 */
void efa_rdm_proto_longcts_write_handle_rtw_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_longcts_rtw_hdr *rtw_hdr;
	ssize_t err;
	uint32_t tx_id;

	ep = pkt_entry->ep;
	rxe = efa_rdm_pke_alloc_rtw_rxe(pkt_entry);
	if (!rxe) {
		EFA_WARN(FI_LOG_CQ,
			"RX entries exhausted.\n");
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep,
					   FI_ENOBUFS,
					   FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	rtw_hdr = (struct efa_rdm_longcts_rtw_hdr *)pkt_entry->wiredata;
	tx_id = rtw_hdr->send_id;
	if (rtw_hdr->type == EFA_RDM_DC_LONGCTS_RTW_PKT)
		rxe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	rxe->iov_count = rtw_hdr->rma_iov_count;
	err = efa_rdm_rma_verified_copy_iov(ep, rtw_hdr->rma_iov, rtw_hdr->rma_iov_count,
					FI_REMOTE_WRITE, rxe->iov, rxe->desc);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "RMA address verify failed!\n");
		efa_base_ep_write_eq_error(&ep->base_ep, err, FI_EFA_ERR_RMA_ADDR);
		efa_rdm_rxe_release(rxe);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	rxe->cq_entry.len = ofi_total_iov_len(rxe->iov, rxe->iov_count);
	rxe->cq_entry.buf = rxe->iov[0].iov_base;
	rxe->total_len = rxe->cq_entry.len;

	rxe->bytes_received += pkt_entry->payload_size;
	if (pkt_entry->payload_size >= rxe->total_len) {
		EFA_WARN(FI_LOG_CQ, "Long RTM size mismatch! payload_size: %ld total_len: %ld\n",
			 pkt_entry->payload_size, rxe->total_len);
		EFA_WARN(FI_LOG_CQ, "target buffer: %p length: %ld\n", rxe->iov[0].iov_base,
			rxe->iov[0].iov_len);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EINVAL, FI_EFA_ERR_RTM_MISMATCH);
		efa_rdm_rxe_release(rxe);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	} else {
		err = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
		if (OFI_UNLIKELY(err)) {
			/* copy_payload_to_ope() releases pkt_entry on error; do not release it here. */
			efa_base_ep_write_eq_error(&ep->base_ep, err, FI_EFA_ERR_RXE_COPY);
			efa_rdm_rxe_release(rxe);
			return;
		}
	}


#if ENABLE_DEBUG
	dlist_insert_tail(&rxe->pending_recv_entry, &ep->ope_recv_list);
	ep->pending_recv_counter++;
#endif
	rxe->state = EFA_RDM_RXE_RECV;
	rxe->tx_id = tx_id;
	err = efa_rdm_ope_post_send_or_queue(rxe, EFA_RDM_CTS_PKT);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "Cannot post CTS packet\n");
		efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_POST);
		efa_rdm_rxe_release(rxe);
	}
}
