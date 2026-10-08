/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "ofi_iov.h"
#include "ofi_proto.h"
#include "efa_errno.h"
#include "efa.h"
#include "efa_env.h"
#include "efa_hmem.h"
#include "efa_base_ep.h"
#include "efa_rdm_ep.h"
#include "efa_rdm_msg.h"
#include "efa_rdm_rma.h"
#include "efa_rdm_ope.h"
#include "efa_rdm_rxe_map.h"
#include "efa_rdm_pke.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_rta.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_protocol.h"
#include "efa_rdm_proto.h"
#include "efa_rdm_tracepoint.h"
#include "efa_rdm_pke_req.h"

/**
 * @brief the total length of the message corresponds to a RTM packet
 *
 * @details
 * A RTM packet is sent/received for an user's message, this function
 * return the total length of that message
 *
 * @param[in]	pkt_entry	RTM packet entry
 *
 * @returns
 * a 64-bits integer
 */
size_t efa_rdm_pke_get_rtm_msg_length(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_base_hdr *base_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	switch (base_hdr->type) {
	case EFA_RDM_EAGER_MSGRTM_PKT:
	case EFA_RDM_EAGER_TAGRTM_PKT:
	case EFA_RDM_DC_EAGER_MSGRTM_PKT:
	case EFA_RDM_DC_EAGER_TAGRTM_PKT:
		return pkt_entry->payload_size;
	case EFA_RDM_MEDIUM_MSGRTM_PKT:
	case EFA_RDM_MEDIUM_TAGRTM_PKT:
		return efa_rdm_pke_get_medium_rtm_base_hdr(pkt_entry)->msg_length;
	case EFA_RDM_DC_MEDIUM_MSGRTM_PKT:
	case EFA_RDM_DC_MEDIUM_TAGRTM_PKT:
		return efa_rdm_pke_get_dc_medium_rtm_base_hdr(pkt_entry)->msg_length;
	case EFA_RDM_LONGCTS_MSGRTM_PKT:
	case EFA_RDM_LONGCTS_TAGRTM_PKT:
	case EFA_RDM_DC_LONGCTS_MSGRTM_PKT:
	case EFA_RDM_DC_LONGCTS_TAGRTM_PKT:
		return efa_rdm_pke_get_longcts_rtm_base_hdr(pkt_entry)->msg_length;
	case EFA_RDM_LONGREAD_MSGRTM_PKT:
	case EFA_RDM_LONGREAD_TAGRTM_PKT:
		return efa_rdm_pke_get_longread_rtm_base_hdr(pkt_entry)->msg_length;
	case EFA_RDM_RUNTREAD_MSGRTM_PKT:
	case EFA_RDM_RUNTREAD_TAGRTM_PKT:
		return efa_rdm_pke_get_runtread_rtm_base_hdr(pkt_entry)->msg_length;
	default:
		assert(0 && "Unknown REQ packet type\n");
	}

	return 0;
}

/**
 * @brief Update RX entry with the information in RTM packet entry.
 *
 * @details
 * The following field of RX entry is updated:
 *            address:       this is necessary because original address in
 *                           rxe can be FI_ADDR_UNSPEC
 *            cq_entry.data: for FI_REMOTE_CQ_DATA
 *            msg_id:        message id
 *            total_len:     application might provide a buffer that is larger
 *                           then incoming message size.
 *            tag:           sender's tag can be different from receiver's tag
 *                           becuase match only requires
 *                           (sender_tag | ignore) == (receiver_tag | ignore)
 *  This function is applied to both expected and unexpected RX entry
 *
 * @param[in]		pkt_entry	RTM packet entry
 * @param[in,out]	rxe		RX entry to be updated
 */
void efa_rdm_pke_rtm_update_rxe(struct efa_rdm_pke *pkt_entry,
				struct efa_rdm_ope *rxe)
{
	struct efa_rdm_base_hdr *base_hdr;

	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	if (base_hdr->flags & EFA_RDM_REQ_OPT_CQ_DATA_HDR) {
		rxe->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rxe->cq_entry.data = efa_rdm_pke_get_req_cq_data(pkt_entry);
	}

	rxe->msg_id = efa_rdm_pke_get_rtm_msg_id(pkt_entry);
	rxe->total_len = efa_rdm_pke_get_rtm_msg_length(pkt_entry);
	if (rxe->op == ofi_op_tagged) {
		rxe->tag = efa_rdm_pke_get_rtm_tag(pkt_entry);
		rxe->cq_entry.tag = rxe->tag;
	}
}

void efa_rdm_pke_prepare_matched_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	int pkt_type;

	rxe = pkt_entry->ope;
	assert(rxe && rxe->state == EFA_RDM_RXE_MATCHED);

	efa_rdm_tracepoint(rx_pke_proc_matched_msg_begin, (size_t) pkt_entry, pkt_entry->payload_size, rxe->msg_id, (size_t) rxe->cq_entry.op_context, rxe->total_len);
	if (!rxe->peer) {
		rxe->peer = pkt_entry->peer;
		assert(rxe->peer);
		dlist_insert_tail(&rxe->peer_entry, &rxe->peer->rxe_list);
	}

	/* Adjust rxe->cq_entry.len as needed.
	 * Initialy rxe->cq_entry.len is total recv buffer size.
	 * rxe->total_len is from REQ packet and is total send buffer size.
	 * if send buffer size < recv buffer size, we adjust value of rxe->cq_entry.len
	 * if send buffer size > recv buffer size, we have a truncated message and will
	 * write error CQ entry.
	 */
	if (rxe->cq_entry.len > rxe->total_len)
		rxe->cq_entry.len = rxe->total_len;

	pkt_type = efa_rdm_pke_get_base_hdr(pkt_entry)->type;

	if (pkt_type > EFA_RDM_DC_REQ_PKT_BEGIN &&
	    pkt_type < EFA_RDM_DC_REQ_PKT_END)
		rxe->internal_flags |= EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED;

	rxe->msg_id = efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->msg_id;
}

/**
 * @brief process a received non-tagged RTM packet
 *
 * @param[in,out]	pkt_entry	non-tagged RTM packet entry
 */
static ssize_t
efa_rdm_pke_proc_msgrtm_with_callback(
	struct efa_rdm_pke *pkt_entry,
	efa_rdm_pke_callback handle_unexp_pke_match)
{
	ssize_t err;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct fid_peer_srx *peer_srx;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;

	ep = pkt_entry->ep;

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *)pkt_entry->wiredata;
	if (rtm_hdr->flags & EFA_RDM_REQ_READ_NACK) {
		rxe = efa_rdm_rxe_map_lookup(&pkt_entry->peer->rxe_map, efa_rdm_pke_get_rtm_msg_id(pkt_entry));
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(
				&ep->base_ep, FI_EINVAL,
				FI_EFA_ERR_PKT_PROC_MSGRTM);
			efa_rdm_pke_release_rx(pkt_entry);
			return -FI_EINVAL;
		}
		rxe->internal_flags |= EFA_RDM_OPE_READ_NACK;
	} else {
		rxe = efa_rdm_msg_alloc_rxe_for_msgrtm(ep, &pkt_entry);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(
				&ep->base_ep, FI_ENOBUFS,
				FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			efa_rdm_pke_release_rx(pkt_entry);
			return -FI_ENOBUFS;
		}
	}

	efa_rdm_pke_set_ope(pkt_entry, rxe);

	if (rxe->state == EFA_RDM_RXE_MATCHED) {
		err = handle_unexp_pke_match(pkt_entry);
		if (OFI_UNLIKELY(err)) {
			efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_PROC_MSGRTM);
			efa_rdm_rxe_release(rxe);
			return err;
		}
	} else if (rxe->state == EFA_RDM_RXE_UNEXP) {
		pkt_entry->handle_pke = handle_unexp_pke_match;
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		return peer_srx->owner_ops->queue_msg(rxe->peer_rxe);
	}

	return 0;
}

/**
 * @brief process a received tagged RTM packet
 *
 * @param[in,out]	pkt_entry	non-tagged RTM packet entry
 */
static ssize_t
efa_rdm_pke_proc_tagrtm_with_callback(
	struct efa_rdm_pke *pkt_entry,
	efa_rdm_pke_callback handle_unexp_pke_match)
{
	ssize_t err;
	struct efa_rdm_ep *ep;
	struct efa_rdm_ope *rxe;
	struct fid_peer_srx *peer_srx;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;

	ep = pkt_entry->ep;

	rtm_hdr = (struct efa_rdm_rtm_base_hdr *) pkt_entry->wiredata;
	if (rtm_hdr->flags & EFA_RDM_REQ_READ_NACK) {
		rxe = efa_rdm_rxe_map_lookup(&pkt_entry->peer->rxe_map, efa_rdm_pke_get_rtm_msg_id(pkt_entry));
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(
				&ep->base_ep, FI_EINVAL,
				FI_EFA_ERR_PKT_PROC_TAGRTM);
			efa_rdm_pke_release_rx(pkt_entry);
			return -FI_EINVAL;
		}
		rxe->internal_flags |= EFA_RDM_OPE_READ_NACK;
	} else {
		rxe = efa_rdm_msg_alloc_rxe_for_tagrtm(ep, &pkt_entry);
		if (OFI_UNLIKELY(!rxe)) {
			efa_base_ep_write_eq_error(
				&ep->base_ep, FI_ENOBUFS,
				FI_EFA_ERR_RXE_POOL_EXHAUSTED);
			efa_rdm_pke_release_rx(pkt_entry);
			return -FI_ENOBUFS;
		}
	}

	efa_rdm_pke_set_ope(pkt_entry, rxe);

	if (rxe->state == EFA_RDM_RXE_MATCHED) {
		err = handle_unexp_pke_match(pkt_entry);
		if (OFI_UNLIKELY(err)) {
			if (err == -FI_ENOMR)
				return err;
			efa_rdm_rxe_handle_error(rxe, -err, FI_EFA_ERR_PKT_PROC_TAGRTM);
			efa_rdm_rxe_release(rxe);
			return err;
		}
	} else if (rxe->state == EFA_RDM_RXE_UNEXP) {
		pkt_entry->handle_pke = handle_unexp_pke_match;
		peer_srx = util_get_peer_srx(ep->peer_srx_ep);
		return peer_srx->owner_ops->queue_tag(rxe->peer_rxe);
	}

	return 0;
}

ssize_t efa_rdm_pke_proc_rtm_after_robuf(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_proto *proto = pkt_entry->proto;

	assert(proto);
	assert(proto->handle_unexp_pke_match);

	if (efa_rdm_pke_get_rtm_base_hdr(pkt_entry)->flags &
	    EFA_RDM_REQ_TAGGED)
		return efa_rdm_pke_proc_tagrtm_with_callback(
			pkt_entry, proto->handle_unexp_pke_match);

	return efa_rdm_pke_proc_msgrtm_with_callback(
		pkt_entry, proto->handle_unexp_pke_match);
}

ssize_t efa_rdm_pke_proc_msgrtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_proto *proto = pkt_entry->proto;

	assert(proto);
	return efa_rdm_pke_proc_msgrtm_with_callback(
		pkt_entry, proto->handle_unexp_pke_match);
}

/**
 * @brief process a received RTA packet entry
 *
 * @details
 * The RTA passed to this function is ordered
 * by msg_id in the packet header
 *
 * @param[in,out]	pkt_entry	received RTA packet entry
 */
ssize_t efa_rdm_pke_proc_rta(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_base_hdr *base_hdr;

	ep = pkt_entry->ep;
	base_hdr = efa_rdm_pke_get_base_hdr(pkt_entry);
	assert(base_hdr->type >= EFA_RDM_BASELINE_REQ_PKT_BEGIN);

	switch (base_hdr->type) {
	case EFA_RDM_WRITE_RTA_PKT:
		return efa_rdm_pke_proc_write_rta(pkt_entry);
	case EFA_RDM_DC_WRITE_RTA_PKT:
		return efa_rdm_pke_proc_dc_write_rta(pkt_entry);
	case EFA_RDM_FETCH_RTA_PKT:
		return efa_rdm_pke_proc_fetch_rta(pkt_entry);
	case EFA_RDM_COMPARE_RTA_PKT:
		return efa_rdm_pke_proc_compare_rta(pkt_entry);
	default:
		EFA_WARN(FI_LOG_EP_CTRL,
			"Unknown packet type ID: %d\n",
		       base_hdr->type);
		efa_base_ep_write_eq_error(&ep->base_ep, FI_EINVAL, FI_EFA_ERR_UNKNOWN_PKT_TYPE);
		efa_rdm_pke_release_rx(pkt_entry);
	}

	return -FI_EINVAL;
}

static void
efa_rdm_pke_handle_ordered_req_recv(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ep *ep;
	struct efa_rdm_peer *peer;
	struct efa_rdm_rtm_base_hdr *rtm_hdr;
	bool slide_recvwin;
	int ret;
	uint32_t exp_msg_id;

	ep = pkt_entry->ep;
	peer = pkt_entry->peer;
	assert(peer);
	assert(pkt_entry->handle_pke);

	ret = efa_rdm_peer_reorder_msg(peer, pkt_entry->ep, pkt_entry);
	if (OFI_UNLIKELY(ret != 0)) {
		if (ret < 0)
			/* reorder_msg already reported error to the EQ */
			efa_rdm_pke_release_rx(pkt_entry);
		return;
	}

	/* The condition below is false for long CTS RTM packet sent after the
	 * long read protocol failed. The message ID was marked as consumed when
	 * the long read RTM packet was processed. So we shouldn't slide the
	 * receive window again.
	 */
	rtm_hdr = (struct efa_rdm_rtm_base_hdr *)pkt_entry->wiredata;
	slide_recvwin = !(rtm_hdr->flags & EFA_RDM_REQ_READ_NACK);

	/*
	 * The callback writes an error CQ entry if needed. Even if processing
	 * fails, slide the receive window so progress can continue.
	 */
	pkt_entry->handle_pke(pkt_entry);

	if (OFI_LIKELY(slide_recvwin)) {
		ofi_recvwin_slide((&peer->robuf));
	}

	exp_msg_id = ofi_recvwin_next_exp_id((&peer->robuf));
	if (OFI_UNLIKELY(exp_msg_id % efa_env.recvwin_size == 0))
		efa_rdm_peer_move_overflow_pke_to_recvwin(peer);

	efa_rdm_peer_proc_pending_items_in_robuf(peer, ep);
}

static inline bool
efa_rdm_pke_handle_existing_mulreq_rxe(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_peer *peer = pkt_entry->peer;
	struct efa_rdm_proto *proto = pkt_entry->proto;
	struct efa_rdm_ope *rxe = efa_rdm_rxe_map_lookup(
		&peer->rxe_map, efa_rdm_pke_get_rtm_msg_id(pkt_entry));
	struct efa_rdm_pke *unexp_pkt_entry = NULL;

	if (!rxe)
		return false;

	if (rxe->state == EFA_RDM_RXE_MATCHED) {
		efa_rdm_pke_set_ope(pkt_entry, rxe);
		proto->handle_unexp_pke_match(pkt_entry);
	} else {
		assert(rxe->unexp_pkt);
		unexp_pkt_entry = efa_rdm_pke_get_unexp(&pkt_entry);
		efa_rdm_pke_append(rxe->unexp_pkt, unexp_pkt_entry);
		efa_rdm_pke_set_ope(unexp_pkt_entry, rxe);
	}

	return true;
}

void efa_rdm_pke_handle_rtm_recv(struct efa_rdm_pke *pkt_entry,
				 struct efa_rdm_proto *proto)
{
	assert(efa_rdm_pkt_type_is_rtm(
		efa_rdm_pke_get_base_hdr(pkt_entry)->type));
	assert(proto);
	assert(!pkt_entry->proto);
	assert(!pkt_entry->handle_pke);

	pkt_entry->proto = proto;
	pkt_entry->handle_pke = efa_rdm_pke_proc_rtm_after_robuf;
	efa_rdm_pke_handle_ordered_req_recv(pkt_entry);
}

void efa_rdm_pke_handle_mulreq_rtm_recv(struct efa_rdm_pke *pkt_entry,
					struct efa_rdm_proto *proto)
{
	assert(pkt_entry->peer);
	assert(efa_rdm_pkt_type_is_mulreq(
		efa_rdm_pke_get_base_hdr(pkt_entry)->type));
	assert(proto);
	assert(!pkt_entry->proto);
	assert(!pkt_entry->handle_pke);

	pkt_entry->proto = proto;
	pkt_entry->handle_pke = efa_rdm_pke_proc_rtm_after_robuf;
	if (efa_rdm_pke_handle_existing_mulreq_rxe(pkt_entry))
		return;

	efa_rdm_pke_handle_ordered_req_recv(pkt_entry);
}

void efa_rdm_pke_handle_rta_recv(struct efa_rdm_pke *pkt_entry)
{
	assert(efa_rdm_pkt_type_is_rta(
		efa_rdm_pke_get_base_hdr(pkt_entry)->type));
	assert(!pkt_entry->proto);
	assert(!pkt_entry->handle_pke);

	pkt_entry->handle_pke = efa_rdm_pke_proc_rta;
	efa_rdm_pke_handle_ordered_req_recv(pkt_entry);
}

/**
 * @brief process a matched MEDIUM or RUNTREAD RTM
 *
 * @details
 * This function applies to all 4 types of MEDIUM
 * RTM and 2 types of RUNTREAD RTM.
 *
 * @param[in,out]	pkt_entry	packet entry
 */
ssize_t efa_rdm_pke_proc_matched_mulreq_rtm(struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *cur, *nxt;
	int pkt_type;
	ssize_t ret, err;
	uint64_t msg_id;

	rxe = pkt_entry->ope;
	pkt_type = efa_rdm_pke_get_base_hdr(pkt_entry)->type;

	ret = 0;
	cur = pkt_entry;
	while (cur) {
		assert(cur->payload);
		assert(cur->payload_size);
		/* efa_rdm_pke_copy_payload_to_ope() can release rxe, so
		 * bytes_received must be calculated before it.
		 */
		rxe->bytes_received += cur->payload_size;
		rxe->bytes_received_via_mulreq += cur->payload_size;
		if (efa_rdm_ope_mulreq_total_data_size(rxe, pkt_type) ==
		    rxe->bytes_received_via_mulreq) {
			if (rxe->internal_flags & EFA_RDM_OPE_READ_NACK) {
				EFA_INFO(FI_LOG_EP_CTRL,
					 "Receiver sending long read NACK "
					 "packet because memory registration "
					 "limit was reached on the receiver\n");
				err = efa_rdm_ope_post_send_or_queue(
					rxe, EFA_RDM_READ_NACK_PKT);
				if (err) {
					efa_rdm_pke_release_rx_list(cur);
					return err;
				}
			} else {
				msg_id = efa_rdm_pke_get_rtm_msg_id(cur);
				efa_rdm_rxe_map_remove(&cur->peer->rxe_map, msg_id,
						       rxe);
			}
		}

		/* efa_rdm_pke_copy_data_to_ope() will release cur, so
		 * cur->next must be copied out before it.
		 */
		nxt = cur->next;
		cur->next = NULL;

		err = efa_rdm_pke_copy_payload_to_ope(cur, rxe);
		if (err) {
			/* efa_rdm_pke_copy_payload_to_ope() frees cur on error no matter where it
			 * sits in the chain; the rest are freed by later iterations. */
			ret = err;
		}

		cur = nxt;
	}

	return ret;
}
