/*
 * Copyright (c) 2019-2023 Amazon.com, Inc. or its affiliates.
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
#include "efa_av.h"

/**
 * @brief initialize a rdm peer
 *
 * @param[in,out]	peer	rdm peer
 * @param[in]		ep	rdm endpoint
 * @param[in]		conn	efa conn object
 * @relates efa_rdm_peer
 */
void efa_rdm_peer_construct(struct efa_rdm_peer *peer, struct rxr_ep *ep, struct efa_conn *conn)
{
	memset(peer, 0, sizeof(struct efa_rdm_peer));

	peer->efa_fiaddr = conn->fi_addr;
	peer->is_self = efa_is_same_addr(&ep->base_ep.src_addr, conn->ep_addr);
	peer->host_id = peer->is_self ? ep->host_id : 0;	/* Peer host id is exchanged via handshake */
	peer->num_read_msg_in_flight = 0;
	peer->num_runt_bytes_in_flight = 0;
	ofi_recvwin_buf_alloc(&peer->robuf, rxr_env.recvwin_size);
	dlist_init(&peer->outstanding_tx_pkts);
	dlist_init(&peer->rx_unexp_list);
	dlist_init(&peer->rx_unexp_tagged_list);
	dlist_init(&peer->tx_entry_list);
	dlist_init(&peer->rx_entry_list);
}

/**
 * @brief clear resources accociated with a peer
 *
 * release reorder buffer, tx_entry list and rx_entry list of a peer
 *
 * @param[in,out]	peer 	rdm peer
 * @relates efa_rdm_peer
 */
void efa_rdm_peer_destruct(struct efa_rdm_peer *peer, struct rxr_ep *ep)
{
	struct dlist_entry *tmp;
	struct rxr_op_entry *tx_entry;
	struct rxr_op_entry *rx_entry;
	struct rxr_pkt_entry *pkt_entry;
	/*
	 * TODO: Add support for wait/signal until all pending messages have
	 * been sent/received so we do not attempt to complete a data transfer
	 * or internal transfer after the EP is shutdown.
	 */
	if ((peer->flags & EFA_RDM_PEER_REQ_SENT) &&
	    !(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED))
		EFA_WARN_ONCE(FI_LOG_EP_CTRL, "Closing EP with unacked CONNREQs in flight\n");

	if (peer->robuf.pending)
		ofi_recvwin_free(&peer->robuf);

	if (!ep) {
		/* ep is NULL means the endpoint has been closed.
		 * In this case there is no need to proceed because
		 * all the tx_entry, rx_entry, pkt_entry has been released.
		 */
		return;
	}

	/* we cannot release outstanding TX packets because device
	 * will report completion of these packets later. Setting
	 * the address to FI_ADDR_NOTAVAIL, so rxr_ep_get_peer()
	 * will return NULL for the address, so the completion will
	 * be ignored.
	 */
	dlist_foreach_container(&peer->outstanding_tx_pkts,
				struct rxr_pkt_entry,
				pkt_entry, entry) {
		pkt_entry->addr = FI_ADDR_NOTAVAIL;
	}

	dlist_foreach_container_safe(&peer->tx_entry_list,
				     struct rxr_op_entry,
				     tx_entry, peer_entry, tmp) {
		rxr_tx_entry_release(tx_entry);
	}

	dlist_foreach_container_safe(&peer->rx_entry_list,
				     struct rxr_op_entry,
				     rx_entry, peer_entry, tmp) {
		rxr_rx_entry_release(rx_entry);
	}

	if (peer->flags & EFA_RDM_PEER_HANDSHAKE_QUEUED)
		dlist_remove(&peer->handshake_queued_entry);

	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
		dlist_remove(&peer->rnr_backoff_entry);

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(peer, sizeof(struct efa_rdm_peer));
#endif
}

/**
 * @brief run incoming packet_entry through reorder buffer
 * queue the packe entry if msg_id is larger then expected.
 * If queue failed, abort the application and print error message.
 *
 * @param[in]		peer		peer
 * @param[in]		ep		endpoint
 * @param[in,out]	pkt_entry	packet entry, will be released if successfully queued
 * @returns
 * 0 if `msg_id` of `pkt_entry` matches expected msg_id.
 * 1 if `msg_id` of `pkt_entry` is larger then expected, and the packet entry is queued successfully
 * -FI_EALREADY if `msg_id` of `pkt_entry` is smaller than expected.
 */
int efa_rdm_peer_reorder_msg(struct efa_rdm_peer *peer, struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_pkt_entry *ooo_entry;
	struct rxr_pkt_entry *cur_ooo_entry;
	struct efa_rdm_robuf *robuf;
	uint32_t msg_id;

	assert(rxr_get_base_hdr(pkt_entry->wiredata)->type >= RXR_REQ_PKT_BEGIN);

	msg_id = rxr_pkt_msg_id(pkt_entry);

	robuf = &peer->robuf;
#if ENABLE_DEBUG
	if (msg_id != ofi_recvwin_next_exp_id(robuf))
		EFA_DBG(FI_LOG_EP_CTRL,
		       "msg OOO msg_id: %" PRIu32 " expected: %"
		       PRIu32 "\n", msg_id,
		       ofi_recvwin_next_exp_id(robuf));
#endif
	if (ofi_recvwin_is_exp(robuf, msg_id))
		return 0;
	else if (!ofi_recvwin_id_valid(robuf, msg_id)) {
		if (ofi_recvwin_id_processed(robuf, msg_id)) {
			EFA_WARN(FI_LOG_EP_CTRL,
			       "Error: message id has already been processed. received: %" PRIu32 " expected: %"
			       PRIu32 "\n", msg_id, ofi_recvwin_next_exp_id(robuf));
			return -FI_EALREADY;
		} else {
			fprintf(stderr,
				"Current receive window size (%d) is too small to hold incoming messages.\n"
				"As a result, you application cannot proceed.\n"
				"Receive window size can be increased by setting the environment variable:\n"
				"              FI_EFA_RECVWIN_SIZE\n"
				"\n"
				"Your job will now abort.\n\n", rxr_env.recvwin_size);
			abort();
		}
	}

	if (OFI_LIKELY(rxr_env.rx_copy_ooo)) {
		assert(pkt_entry->alloc_type == RXR_PKT_FROM_EFA_RX_POOL);
		ooo_entry = rxr_pkt_entry_clone(ep, ep->rx_ooo_pkt_pool, RXR_PKT_FROM_OOO_POOL, pkt_entry);
		if (OFI_UNLIKELY(!ooo_entry)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"Unable to allocate rx_pkt_entry for OOO msg\n");
			return -FI_ENOMEM;
		}
		rxr_pkt_entry_release_rx(ep, pkt_entry);
	} else {
		ooo_entry = pkt_entry;
	}

	cur_ooo_entry = *ofi_recvwin_get_msg(robuf, msg_id);
	if (cur_ooo_entry) {
		assert(rxr_pkt_type_is_mulreq(rxr_get_base_hdr(cur_ooo_entry->wiredata)->type));
		assert(rxr_pkt_msg_id(cur_ooo_entry) == msg_id);
		assert(rxr_pkt_rtm_total_len(cur_ooo_entry) == rxr_pkt_rtm_total_len(ooo_entry));
		rxr_pkt_entry_append(cur_ooo_entry, ooo_entry);
	} else {
		ofi_recvwin_queue_msg(robuf, &ooo_entry, msg_id);
	}

	return 1;
}

/**
 * @brief process packet entries in reorder buffer
 * This function is called after processing the expected packet entry
 *
 * @param[in,out]	peer 		peer
 * @param[in,out]	ep		end point
 */
void efa_rdm_peer_proc_pending_items_in_robuf(struct efa_rdm_peer *peer, struct rxr_ep *ep)
{
	struct rxr_pkt_entry *pending_pkt;
	int ret = 0;
	uint32_t msg_id;

	while (1) {
		pending_pkt = *ofi_recvwin_peek((&peer->robuf));
		if (!pending_pkt)
			return;

		msg_id = rxr_pkt_msg_id(pending_pkt);
		EFA_DBG(FI_LOG_EP_CTRL,
		       "Processing msg_id %d from robuf\n", msg_id);
		/* rxr_pkt_proc_rtm_rta will write error cq entry if needed */
		ret = rxr_pkt_proc_rtm_rta(ep, pending_pkt);
		*ofi_recvwin_get_next_msg((&peer->robuf)) = NULL;
		if (OFI_UNLIKELY(ret)) {
			EFA_WARN(FI_LOG_CQ,
				"Error processing msg_id %d from robuf: %s\n",
				msg_id, fi_strerror(-ret));
			return;
		}
	}
	return;
}
