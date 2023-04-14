/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "efa.h"
#include "efa_rdm_srx.h"
#include "rxr_msg.h"
#include "rxr_pkt_type_req.h"

/**
 * @brief Construct a packet entry that will be used as input of
 * rxr_pkt_get_msg(tag)rtm_rxe in efa_rdm_srx_get_msg(tag).
 *
 * @param[in,out] pkt_entry the pkt_entry to be constructed.
 * @param[in] addr the fi_addr_t of the pkt entry
 * @param[in] size the data size of the pkt entry
 * @param[in] tag the tag of the pkt entry, ignored unless the op is ofi_op_tagged
 * @param[in] op the ofi_op code, allowed values: ofi_op_msg and ofi_op_tagged
 */
static
void efa_rdm_srx_construct_pkt_entry(struct rxr_pkt_entry *pkt_entry,
				     fi_addr_t addr,
				     size_t size,
				     uint64_t tag,
				     int op)
{
	struct rxr_longread_rtm_base_hdr *rtm_hdr;
	int pkt_type;
	assert(op == ofi_op_msg || op == ofi_op_tagged);

	/*
	 * Use longread msg/tag rtm pkt type because it has the
	 * msg length in the pkt hdr that is used to derive the
	 * pkt size for the multi-recv path.
	 * And we cannot make the data size part of pkt_entry->pkt_size
	 * because we are not allocating memory for it.
	 */
	if (op == ofi_op_msg)
		pkt_type = RXR_LONGREAD_MSGRTM_PKT;
	else
		pkt_type = RXR_LONGREAD_TAGRTM_PKT;

	rtm_hdr = (struct rxr_longread_rtm_base_hdr *)pkt_entry->wiredata;
	rtm_hdr->hdr.type = pkt_type;
	rtm_hdr->hdr.version = RXR_PROTOCOL_VERSION;
	rtm_hdr->hdr.flags = RXR_REQ_MSG;
	rtm_hdr->msg_length = size;

	if (op == ofi_op_tagged) {
		rtm_hdr->hdr.flags |= RXR_REQ_TAGGED;
		rxr_pkt_rtm_settag(pkt_entry, tag);
	}

	pkt_entry->pkt_size = rxr_pkt_req_hdr_size(pkt_type, 0, 0);
	pkt_entry->addr = addr;
	pkt_entry->alloc_type = RXR_PKT_FROM_PEER_SRX;
	pkt_entry->flags = RXR_PKT_ENTRY_IN_USE;
	pkt_entry->next = NULL;
	pkt_entry->ope = NULL;
	pkt_entry->recv_wr.wr.next = NULL;
	pkt_entry->send_wr = NULL;
	pkt_entry->mr = NULL;
}

/**
 * @brief This call is invoked by the peer provider to obtain a
 * peer_rxe where an incoming message should be placed.
 * @param[in] srx the fid_peer_srx
 * @param[in] addr the source address of the incoming message
 * @param[in] size the size of the incoming message
 * @param[out] peer_rxe the obtained peer_rxe
 * @return int FI_SUCCESS when a matched rxe is found, -FI_ENOENT when the
 * match is not found, other negative integer for other errors.
 */
static int efa_rdm_srx_get_msg(struct fid_peer_srx *srx, fi_addr_t addr,
		       size_t size, struct fi_peer_rx_entry **peer_rxe)
{
	struct rxr_ep *rxr_ep;
	struct rxr_pkt_entry *pkt_entry;
	struct efa_rdm_ope *rxe;
	int ret;
	char buf[PEER_SRX_MSG_PKT_SIZE];

	pkt_entry = (struct rxr_pkt_entry *) &buf[0];

	rxr_ep = (struct rxr_ep *) srx->ep_fid.fid.context;
	/*
	 * TODO:
	 * In theory we should not need to create a pkt_entry to get a rxe,
	 * but the current EFA code needs an pkt_entry as input. A major refactor
	 * is needed to split the context (addr, size etc.) and data from pkt entry
	 * and make rxr_*_get_*_rxe needs context only.
	 */
	efa_rdm_srx_construct_pkt_entry(pkt_entry, addr, size, 0, ofi_op_msg);

	ofi_mutex_lock(&rxr_ep->base_ep.util_ep.lock);
	rxe = rxr_pkt_get_msgrtm_rxe(rxr_ep, &pkt_entry);
	if (OFI_UNLIKELY(!rxe)) {
		efa_eq_write_error(&rxr_ep->base_ep.util_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		ret = -FI_ENOBUFS;
		goto out;
	}
	ret = rxe->state == EFA_RDM_RXE_MATCHED ? FI_SUCCESS : -FI_ENOENT;
	/* Override this field to be peer provider's srx so the correct srx can be used by start_msg ops */
	rxe->peer_rxe.srx = srx;
	*peer_rxe = &rxe->peer_rxe;

out:
	ofi_mutex_unlock(&rxr_ep->base_ep.util_ep.lock);
	return ret;
}

/**
 * @brief This call is invoked by the peer provider to obtain a
 * peer_rxe where an incoming tagged message should be placed.
 *
 * @param[in] srx the fid_peer_srx
 * @param[in] addr the source address of the incoming message
 * @param[in] size the size of the incoming message
 * @param[out] peer_rxe the obtained peer_rxe
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_get_tag(struct fid_peer_srx *srx, fi_addr_t addr,
			size_t size, uint64_t tag, struct fi_peer_rx_entry **peer_rxe)
{
	struct rxr_ep *rxr_ep;
	struct rxr_pkt_entry *pkt_entry;
	struct efa_rdm_ope *rxe;
	int ret;
	char buf[PEER_SRX_TAG_PKT_SIZE];

	pkt_entry = (struct rxr_pkt_entry *) &buf[0];

	rxr_ep = (struct rxr_ep *) srx->ep_fid.fid.context;
	/*
	 * TODO:
	 * In theory we should not need to create a pkt_entry to get a rxe,
	 * but the current EFA code needs an pkt_entry as input. A major refactor
	 * is needed to split the context (addr, size etc.) and data from pkt entry
	 * and make rxr_*_get_*_rxe needs context only.
	 */
	efa_rdm_srx_construct_pkt_entry(pkt_entry, addr, size, tag, ofi_op_tagged);

	ofi_mutex_lock(&rxr_ep->base_ep.util_ep.lock);
	rxe = rxr_pkt_get_tagrtm_rxe(rxr_ep, &pkt_entry);
	if (OFI_UNLIKELY(!rxe)) {
		efa_eq_write_error(&rxr_ep->base_ep.util_ep, FI_ENOBUFS, FI_EFA_ERR_RXE_POOL_EXHAUSTED);
		ret = -FI_ENOBUFS;
		goto out;
	}
	ret = rxe->state == EFA_RDM_RXE_MATCHED ? FI_SUCCESS : -FI_ENOENT;
	/* Override this field to be peer provider's srx so the correct srx can be used by start_tag ops */
	rxe->peer_rxe.srx = srx;
	*peer_rxe = &rxe->peer_rxe;

out:
	ofi_mutex_unlock(&rxr_ep->base_ep.util_ep.lock);
	return ret;
}

/**
 * @brief This call is invoked by the peer provider to queue
 * a received message to owner provider's unexpected queue,
 * when the peer calls owner_ops->get_msg() but the owner fails to
 * find a matching receive buffer.
 *
 * @param[in] peer_rxe the entry to be queued
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_queue_msg(struct fi_peer_rx_entry *peer_rxe)
{
	struct efa_rdm_ope *rxe;
	struct rxr_ep *ep;

	rxe = container_of(peer_rxe, struct efa_rdm_ope, peer_rxe);
	ep = rxe->ep;

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	rxr_msg_queue_unexp_rxe_for_msgrtm(ep, rxe);
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);
	return FI_SUCCESS;
}

/**
 * @brief This call is invoked by the peer provider to queue
 * a received tagged message to owner provider's unexpected queue,
 * when the peer calls owner_ops->get_tag() but the owner fails to
 * find a matching receive buffer.
 *
 * @param[in] peer_rxe the entry to be queued
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_queue_tag(struct fi_peer_rx_entry *peer_rxe)
{
	struct efa_rdm_ope *rxe;
	struct rxr_ep *ep;

	rxe = container_of(peer_rxe, struct efa_rdm_ope, peer_rxe);
	ep = rxe->ep;

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	rxr_msg_queue_unexp_rxe_for_tagrtm(ep, rxe);
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);
	return FI_SUCCESS;
}

/**
 * @brief This call is invoked by the peer provider to release a peer_rxe
 * from owner provider's resource pool
 *
 * @param[in] peer_rxe the peer_rxe to be freed
 */
static void efa_rdm_srx_free_entry(struct fi_peer_rx_entry *peer_rxe)
{
	struct efa_rdm_ope *rxe;
	struct rxr_ep *ep;

	rxe = container_of(peer_rxe, struct efa_rdm_ope, peer_rxe);
	ep = rxe->ep;

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	if (rxe->fi_flags & FI_MULTI_RECV) {
		rxr_msg_multi_recv_handle_completion(rxe->ep, rxe);
		if (rxe->cq_entry.flags & FI_MULTI_RECV) {
			if (ofi_peer_cq_write(rxe->ep->base_ep.util_ep.rx_cq,
					      rxe->master_entry->cq_entry.op_context,
					      FI_MULTI_RECV, 0, NULL, 0, 0,
					      FI_ADDR_NOTAVAIL)) {
				EFA_WARN(FI_LOG_EP_CTRL,
					"unable to write rx MULTI_RECV completion\n");
			}
		}
	}
	rxr_msg_multi_recv_free_posted_entry(rxe->ep, rxe);
	efa_rdm_rxe_release(rxe);
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);
}

/**
 * @brief This call is invoked by the owner provider to start progressing
 * the peer_rxe that matches a received message.
 *
 * @param[in] peer_rxe the rxe to be progressed.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_start_msg(struct fi_peer_rx_entry *peer_rxe)
{
	struct efa_rdm_ope *rx_ope;
	int ret;
	struct rxr_ep *ep;

	rx_ope = container_of(peer_rxe, struct efa_rdm_ope, peer_rxe);
	ep = rx_ope->ep;

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	ret = rxr_pkt_proc_matched_rtm(ep, rx_ope, peer_rxe->owner_context);
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);

	return ret;
}

/**
 * @brief This call is invoked by the owner provider to start progressing
 * the peer_rxe that matches a received tagged message.
 *
 * @param[in] peer_rxe the fi_peer_rx_entry to be progressed.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_start_tag(struct fi_peer_rx_entry *peer_rxe)
{
	struct efa_rdm_ope *rx_ope;
	int ret;
	struct rxr_ep *ep;

	rx_ope = container_of(peer_rxe, struct efa_rdm_ope, peer_rxe);
	ep = rx_ope->ep;

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	ret = rxr_pkt_proc_matched_rtm(ep, rx_ope, peer_rxe->owner_context);
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);

	return ret;
}

/**
 * @brief This call is invoked by the owner provider to discard
 * the message and data associated with the specified fi_peer_rx_entry.
 * This often indicates that the application has canceled or discarded
 * the receive operation.
 *
 * @param[in] peer_rxe the fi_peer_rx_entry to be discarded.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_discard_msg(struct fi_peer_rx_entry *peer_rxe)
{
    return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the owner provider to discard
 * the tagged message and data associated with the specified fi_peer_rx_entry.
 * This often indicates that the application has canceled or discarded
 * the receive operation.
 *
 * @param[in] peer_rxe the fi_peer_rx_entry to be discarded.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_discard_tag(struct fi_peer_rx_entry *peer_rxe)
{
    return -FI_ENOSYS;
}


static struct fi_ops_srx_owner efa_rdm_srx_owner_ops = {
	.size = sizeof(struct fi_ops_srx_owner),
	.get_msg = efa_rdm_srx_get_msg,
	.get_tag = efa_rdm_srx_get_tag,
	.queue_msg = efa_rdm_srx_queue_msg,
	.queue_tag = efa_rdm_srx_queue_tag,
	.free_entry = efa_rdm_srx_free_entry,
};

static struct fi_ops_srx_peer efa_rdm_srx_peer_ops = {
	.size = sizeof(struct fi_ops_srx_peer),
	.start_msg = efa_rdm_srx_start_msg,
	.start_tag = efa_rdm_srx_start_tag,
	.discard_msg = efa_rdm_srx_discard_msg,
	.discard_tag = efa_rdm_srx_discard_tag,
};

/**
 * @brief Construct peer srx
 *
 * @param[in] rxr_ep rxr_ep
 * @param[out] peer_srx the constructed peer srx
 */
void efa_rdm_peer_srx_construct(struct rxr_ep *rxr_ep, struct fid_peer_srx *peer_srx)
{
	peer_srx->owner_ops = &efa_rdm_srx_owner_ops;
	peer_srx->peer_ops = &efa_rdm_srx_peer_ops;
	/* This is required to bind this srx to peer provider's ep */
	peer_srx->ep_fid.fid.fclass = FI_CLASS_SRX_CTX;
	/* This context will be used in the ops of peer SRX to access the owner provider resources */
	peer_srx->ep_fid.fid.context = rxr_ep;
}
