/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_eager_write.h"
#include "efa.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtw.h"
#include "efa_rdm_proto_write.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_rma.h"

/*
 * List of packet types used by this protocol
 *
 * EFA_RDM_EAGER_RTW_PKT
 * EFA_RDM_DC_EAGER_RTW_PKT
 *
 * For FI_DELIVERY_COMPLETE - shared with other protocols
 * EFA_RDM_RECEIPT_PKT
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#emulated-eager-write-featuresubprotocol
 */

/* TX path functions */

/**
 * @brief initialize a EFA_RDM_EAGER_RTW packet
 *
 * @param[in,out]	pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA write information
 *
 * @returns
 * 0 on success
 * negative libfabric error code on failure
 */
ssize_t efa_rdm_proto_eager_write_init_rtw(struct efa_rdm_pke *pkt_entry,
				   struct efa_rdm_ope *txe)
{
	struct efa_rdm_eager_rtw_hdr *rtw_hdr;

	assert(txe->op == ofi_op_write);

	rtw_hdr = (struct efa_rdm_eager_rtw_hdr *)pkt_entry->wiredata;
	rtw_hdr->rma_iov_count = txe->rma_iov_count;
	efa_rdm_pke_init_req_hdr_common(pkt_entry, EFA_RDM_EAGER_RTW_PKT, txe);
	return efa_rdm_proto_write_rtw_pke_init_common(pkt_entry, txe, rtw_hdr->rma_iov);
}

/**
 * @brief initialize a EFA_RDM_DC_EAGER_RTW_PKT packet
 *
 * DC means delivery complete
 *
 * @param[in,out]	pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA write information
 * @returns
 * 0 on success.
 * negative libfabric error code on failure
 */
static ssize_t efa_rdm_proto_eager_write_init_dc_rtw(struct efa_rdm_pke *pkt_entry,
				      struct efa_rdm_ope *txe)
{
	struct efa_rdm_dc_eager_rtw_hdr *dc_eager_rtw_hdr;
	int ret;

	assert(txe->op == ofi_op_write);

	txe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	dc_eager_rtw_hdr = (struct efa_rdm_dc_eager_rtw_hdr *)pkt_entry->wiredata;
	dc_eager_rtw_hdr->rma_iov_count = txe->rma_iov_count;
	efa_rdm_pke_init_req_hdr_common(pkt_entry, EFA_RDM_DC_EAGER_RTW_PKT, txe);
	ret = efa_rdm_proto_write_rtw_pke_init_common(pkt_entry, txe,
					  dc_eager_rtw_hdr->rma_iov);
	dc_eager_rtw_hdr->send_id = txe->tx_id;
	return ret;
}

/**
 * @brief handle the send completion event of an EAGER RTW packet
 *
 * This function apply to both EFA_RDM_EAGER_RTW_PKT and
 * EFA_RDM_DC_EAGER_RTW_PKT
 *
 * @param[in]		pkt_entry	packet entry
 */
void efa_rdm_pke_handle_eager_rtw_send_completion(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe->total_len == pkt_entry->payload_size);
	efa_rdm_ope_handle_send_completed(txe);
}

/* RX path functions */

/**
 * @brief process an received EAGER RTW packet
 *
 * This function apply to both EFA_RDM_EAGER_RTW_PKT and
 * EFA_RDM_DC_EAGER_RTW_PKT
 *
 * @param[in]		pkt_entry	received EAGER RTW packet entry
 * @param[in,out]	rxe		RX entry
 * @param[in]		rma_iov		rma_iov array in RTW header
 * @param[in]		rma_iov_count	number of elements in rma_iov
 */
static void efa_rdm_proto_eager_write_proc_rtw(struct efa_rdm_pke *pkt_entry,
				struct efa_rdm_ope *rxe,
				struct efa_rma_iov *rma_iov,
				size_t rma_iov_count)
{
	ssize_t err;
	struct efa_rdm_ep *ep;

	ep = pkt_entry->ep;

	err = efa_rdm_rma_verified_copy_iov(ep, rma_iov, rma_iov_count,
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
	if (pkt_entry->payload_size != rxe->total_len) {
		EFA_WARN(FI_LOG_CQ, "Eager RTM size mismatch! payload_size: %ld total_len: %ld.\n",
			 pkt_entry->payload_size, rxe->total_len);
		EFA_WARN(FI_LOG_CQ, "target buffer: %p length: %ld\n", rxe->iov[0].iov_base,
			rxe->iov[0].iov_len);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EINVAL, FI_EFA_ERR_RTM_MISMATCH);
		efa_rdm_pke_release_rx(pkt_entry);
		efa_rdm_rxe_release(rxe);
	} else {
		err = efa_rdm_pke_copy_payload_to_ope(pkt_entry, rxe);
		if (OFI_UNLIKELY(err)) {
			/* copy_payload_to_ope() releases pkt_entry on error; do not release it here. */
			efa_base_ep_write_eq_error(&ep->base_ep, err, FI_EFA_ERR_RXE_COPY);
			efa_rdm_rxe_release(rxe);
		}
	}
}

/**
 * @brief handle the event that an EFA_RDM_EAGER_RTW packet has been received
 *
 * Calls #efa_rdm_proto_eager_write_proc_rtw()
 *
 * @param[in,out]	pkt_entry	received EFA_RDM_EAGER_RTW packet
 *
 */
void efa_rdm_proto_eager_write_handle_rtw_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_eager_rtw_hdr *rtw_hdr;

	ep = pkt_entry->ep;
	rxe = efa_rdm_pke_alloc_rtw_rxe(pkt_entry);

	if (!rxe) {
		EFA_WARN(FI_LOG_CQ,
			"RX entries exhausted.\n");
		efa_base_ep_write_eq_error(&ep->base_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	rtw_hdr = (struct efa_rdm_eager_rtw_hdr *)pkt_entry->wiredata;
	rxe->iov_count = rtw_hdr->rma_iov_count;
	efa_rdm_proto_eager_write_proc_rtw(pkt_entry,
				   rxe,
				   rtw_hdr->rma_iov,
				   rtw_hdr->rma_iov_count);
}

/**
 * @brief handle the event that an EFA_RDM_DC_EAGER_RTW packet has been received
 *
 * DC means delivery complete
 * Calls #efa_rdm_proto_eager_write_proc_rtw()
 *
 * @param[in,out]	pkt_entry	received EFA_RDM_DC_EAGER_RTW packet
 *
 */
void efa_rdm_proto_eager_write_handle_dc_rtw_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_dc_eager_rtw_hdr *rtw_hdr;

	rxe = efa_rdm_pke_alloc_rtw_rxe(pkt_entry);
	if (!rxe) {
		EFA_WARN(FI_LOG_CQ,
			"RX entries exhausted.\n");
		efa_base_ep_write_eq_error(&pkt_entry->ep->base_ep,
					   FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	rxe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;
	rtw_hdr = (struct efa_rdm_dc_eager_rtw_hdr *)pkt_entry->wiredata;
	rxe->tx_id = rtw_hdr->send_id;
	rxe->iov_count = rtw_hdr->rma_iov_count;
	efa_rdm_proto_eager_write_proc_rtw(pkt_entry,
				   rxe,
				   rtw_hdr->rma_iov,
				   rtw_hdr->rma_iov_count);
}

/**
 * @brief Check if the eager write protocol can handle this operation.
 *
 * The request carries the whole write in one packet, so it is only usable when
 * the payload fits after the header. Unlike a two-sided send, the header also
 * holds the remote memory region, which grows it.
 */
static bool efa_rdm_proto_eager_write_can_use(struct efa_rdm_ope *txe,
					      int req_pkt_type,
					      uint16_t header_flags, int iface,
					      bool use_p2p)
{
	size_t max_data_offset, max_rtw_data_capacity;

	max_data_offset = efa_rdm_pkt_type_get_req_hdr_size(
		req_pkt_type, header_flags, txe->rma_iov_count);
	max_rtw_data_capacity = txe->ep->mtu_size - max_data_offset;

	return txe->total_len <= max_rtw_data_capacity;
}

struct efa_rdm_proto efa_rdm_proto_eager_write = {
	.name = "eager_write",
	.wants_mr = false,
	.can_use_protocol = &efa_rdm_proto_eager_write_can_use,
	.construct_tx_pkes = &efa_rdm_proto_eager_write_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_EAGER_RTW_PKT,
	.req_pkt_type_dc = EFA_RDM_DC_EAGER_RTW_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

/**
 * @brief Handle the send completion of an eager RTW packet.
 *
 * A transmit complete write is done once the device reports the send, so it
 * reports the completion and releases the TXE here. A delivery complete write
 * must also wait for the peer's RECEIPT, so it only releases the TXE here if
 * that RECEIPT already arrived; otherwise efa_rdm_pke_handle_receipt_recv()
 * reports the completion and releases it.
 */
void efa_rdm_proto_eager_write_handle_rtw_send_completion(
	struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *txe;

	txe = pkt_entry->ope;
	assert(txe);
	assert(txe->total_len == pkt_entry->payload_size);

	if (txe->internal_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED) {
		if (efa_rdm_txe_with_remote_ack_ready_for_release(txe))
			efa_rdm_txe_release(txe);
	} else {
		efa_rdm_ope_handle_send_completed(txe);
	}

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct the TX packet entry for the eager write protocol.
 *
 * Allocates a single RTW packet entry, writes the remote memory region and
 * payload into it, and sets the per-packet send completion callback. On
 * success ep->send_pkt_entry_vec[0] holds the packet entry.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_eager_write_construct_tx_pkes(struct efa_rdm_ep *ep,
						struct efa_rdm_peer *peer,
						uint32_t op, uint64_t tag,
						uint64_t flags,
						uint32_t internal_flags,
						struct efa_rdm_ope *txe,
						uint64_t *pke_send_flags)
{
	int ret;
	struct efa_rdm_pke *pkt_entry;

	/*
	 * Eager write is a single packet, so it honors the caller's FI_MORE
	 * request; the doorbell can be deferred to batch with a following post.
	 */
	*pke_send_flags = (txe->fi_flags & FI_MORE) ? FI_MORE : 0;

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
		&efa_rdm_proto_eager_write_handle_rtw_send_completion;

	if (txe->req_pkt_type == efa_rdm_proto_eager_write.req_pkt_type_dc)
		ret = efa_rdm_proto_eager_write_init_dc_rtw(pkt_entry, txe);
	else
		ret = efa_rdm_proto_eager_write_init_rtw(pkt_entry, txe);
	if (ret)
		goto err;

	assert(txe->total_len == pkt_entry->payload_size);

	ep->send_pkt_entry_vec[0] = pkt_entry;
	ep->send_pkt_entry_vec_size = 1;
	return FI_SUCCESS;

err:
	efa_rdm_pke_release_tx(pkt_entry);
	return ret;
}
