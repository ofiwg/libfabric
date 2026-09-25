/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_rdm_proto_longread_write.h"
#include "ofi_iov.h"
#include "efa_errno.h"
#include "efa.h"
#include "efa_base_ep.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_rma.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtw.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_pke_req.h"

/*
 * List of packet types used by this protocol
 *
 * EFA_RDM_LONGREAD_RTW_PKT
 *
 * The peer RDMA-reads the source buffer and signals completion with an EOR on
 * the shared read path, which is not owned by this protocol.
 */

/*
 * Description of the protocol
 * https://github.com/ofiwg/libfabric/blob/main/prov/efa/docs/efa_rdm_protocol_v4.md#emulated-longread-write-featuresubprotocol
 */

/* TX path functions */

/**
 * @brief initialize a EFA_RDM_LONGREAD_RTW packet
 *
 * @param[in,out]	pkt_entry	packet entry to be initialized
 * @param[in]		txe		TX entry that has RMA write information
 * @returns
 * 0 on success.
 * negative libfabric error code on failure
 */
static ssize_t efa_rdm_proto_longread_write_init_rtw(struct efa_rdm_pke *pkt_entry,
						     struct efa_rdm_ope *txe)
{
	struct efa_rdm_longread_rtw_hdr *rtw_hdr;
	struct efa_rma_iov *rma_iov;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;
	int i, err;

	assert(txe->op == ofi_op_write);

	rtw_hdr = (struct efa_rdm_longread_rtw_hdr *)pkt_entry->wiredata;
	rtw_hdr->rma_iov_count = txe->rma_iov_count;
	rtw_hdr->msg_length = txe->total_len;
	rtw_hdr->send_id = txe->tx_id;
	rtw_hdr->read_iov_count = txe->iov_count;
	efa_rdm_pke_init_req_hdr_common(pkt_entry, EFA_RDM_LONGREAD_RTW_PKT, txe);

	rma_iov = rtw_hdr->rma_iov;
	for (i = 0; i < txe->rma_iov_count; ++i) {
		rma_iov[i].addr = txe->rma_iov[i].addr;
		rma_iov[i].len = txe->rma_iov[i].len;
		rma_iov[i].key = txe->rma_iov[i].key;
	}

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata + hdr_size);
	err = efa_rdm_txe_prepare_to_be_read(txe, read_iov);
	if (OFI_UNLIKELY(err))
		return err;

	pkt_entry->pkt_size = hdr_size + txe->iov_count * sizeof(struct efa_rma_iov);
	efa_rdm_pke_set_ope(pkt_entry, txe);
	return 0;
}

/**
 * @brief Handle the send completion of a LONGREAD RTW packet.
 *
 * For long read write, the txe is released either here or in
 * efa_rdm_pke_handle_eor_recv(), whichever happens last. Release here if the
 * EOR already arrived.
 */
void efa_rdm_proto_longread_write_handle_rtw_send_completion(struct efa_rdm_pke *pkt_entry)
{
	assert(pkt_entry->ope);
	if (efa_rdm_txe_with_remote_ack_ready_for_release(pkt_entry->ope))
		efa_rdm_txe_release(pkt_entry->ope);
	/* Peer-abort race: see the long read RTM send completion. */
	else if (pkt_entry->ope->internal_flags & EFA_RDM_OPE_PEER_ABORT_PENDING)
		efa_rdm_txe_progress_peer_abort_if_drained(pkt_entry->ope);

	efa_rdm_pke_release_tx(pkt_entry);
}

/**
 * @brief Construct the TX packet entry for the long read write protocol.
 *
 * Allocates the RTW packet entry that describes the remote memory region and
 * the source buffer for the peer to RDMA-read, and sets the per-packet send
 * completion callback. On success ep->send_pkt_entry_vec[0] holds the packet
 * entry.
 *
 * @return 0 on success, negative errno on failure
 */
int efa_rdm_proto_longread_write_construct_tx_pkes(struct efa_rdm_ep *ep,
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

	/* Long read write posts a single RTW, so it does not honor FI_MORE. */
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
		&efa_rdm_proto_longread_write_handle_rtw_send_completion;

	ret = efa_rdm_proto_longread_write_init_rtw(pkt_entry, txe);
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
 * @brief Check if the long read write protocol can handle this operation.
 *
 * The peer RDMA-reads the source buffer, so this is usable when the transfer
 * is large enough and the peer supports RDMA read over p2p.
 */
static bool efa_rdm_proto_longread_write_can_use(struct efa_rdm_ope *txe,
						 int req_pkt_type,
						 uint16_t header_flags, int iface,
						 bool use_p2p)
{
	return efa_rdm_rma_should_write_using_longread(txe->ep, txe, txe->peer,
						       use_p2p);
}

struct efa_rdm_proto efa_rdm_proto_longread_write = {
	.name = "longread_write",
	.wants_mr = true,
	.can_use_protocol = &efa_rdm_proto_longread_write_can_use,
	.construct_tx_pkes = &efa_rdm_proto_longread_write_construct_tx_pkes,
	.req_pkt_type = EFA_RDM_LONGREAD_RTW_PKT,
	.req_pkt_type_dc = EFA_RDM_LONGREAD_RTW_PKT,
	.handle_tx_pkes_posted = &efa_rdm_proto_handle_tx_pkes_posted_no_op,
};

/* RX path functions */

/**
 * @brief handle the event that a LONGREAD RTW packet has been received
 *
 * @param[in]	pkt_entry	received LONGREAD RTW packet entry
 */
void efa_rdm_proto_longread_write_handle_rtw_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_longread_rtw_hdr *rtw_hdr;
	struct fi_rma_iov *read_iov;
	size_t hdr_size;
	ssize_t err;

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

	rtw_hdr = (struct efa_rdm_longread_rtw_hdr *)pkt_entry->wiredata;
	rxe->iov_count = rtw_hdr->rma_iov_count;
	err = efa_rdm_rma_verified_copy_iov(pkt_entry->ep,
					    rtw_hdr->rma_iov,
					    rtw_hdr->rma_iov_count,
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

	hdr_size = efa_rdm_pke_get_req_hdr_size(pkt_entry);
	read_iov = (struct fi_rma_iov *)(pkt_entry->wiredata + hdr_size);
	rxe->peer = pkt_entry->peer;
	rxe->tx_id = rtw_hdr->send_id;
	rxe->rma_iov_count = rtw_hdr->read_iov_count;
	memcpy(rxe->rma_iov, read_iov,
	       rxe->rma_iov_count * sizeof(struct fi_rma_iov));

	err = efa_rdm_pke_post_remote_read_or_nack(rxe->ep, pkt_entry, rxe);

	efa_rdm_pke_release_rx(pkt_entry);

	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ,
			"RDMA post read or queue failed.\n");
		efa_base_ep_write_eq_error(&ep->base_ep, err, FI_EFA_ERR_RDMA_READ_POST);
		efa_rdm_rxe_release(rxe);
	}
}
