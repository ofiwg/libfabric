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
#include "efa_rdm_pkt_type.h"
#include "efa_rdm_pke_rtm.h"
#include "efa_rdm_pke_utils.h"
#include "efa_rdm_pke_req.h"
#include "efa_rdm_pke_rtm.h"

/**
 * @brief initialize a rdm peer
 *
 * @param[in,out]	peer	rdm peer
 * @param[in]		ep	rdm endpoint
 * @param[in]		conn	efa conn object
 * @relates efa_rdm_peer
 */
void efa_rdm_peer_construct(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep, struct efa_conn *conn)
{
	memset(peer, 0, sizeof(struct efa_rdm_peer));

	peer->efa_fiaddr = conn->fi_addr;
	peer->is_self = efa_is_same_addr(&ep->base_ep.src_addr, conn->ep_addr);
	peer->host_id = peer->is_self ? ep->host_id : 0;	/* Peer host id is exchanged via handshake */
	peer->num_read_msg_in_flight = 0;
	peer->num_runt_bytes_in_flight = 0;
	ofi_recvwin_buf_alloc(&peer->robuf, efa_env.recvwin_size);
	dlist_init(&peer->outstanding_tx_pkts);
	dlist_init(&peer->txe_list);
	dlist_init(&peer->rxe_list);
}

/**
 * @brief clear resources accociated with a peer
 *
 * release reorder buffer, txe list and rxe list of a peer
 *
 * @param[in,out]	peer 	rdm peer
 * @relates efa_rdm_peer
 */
void efa_rdm_peer_destruct(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep)
{
	struct dlist_entry *tmp;
	struct efa_rdm_ope *txe;
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *pkt_entry;
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
		 * all the txe, rxe, pkt_entry has been released.
		 */
		return;
	}

	/* we cannot release outstanding TX packets because device
	 * will report completion of these packets later. Setting
	 * the address to FI_ADDR_NOTAVAIL, so efa_rdm_ep_get_peer()
	 * will return NULL for the address, so the completion will
	 * be ignored.
	 */
	dlist_foreach_container(&peer->outstanding_tx_pkts,
				struct efa_rdm_pke,
				pkt_entry, entry) {
		pkt_entry->addr = FI_ADDR_NOTAVAIL;
	}

	dlist_foreach_container_safe(&peer->txe_list,
				     struct efa_rdm_ope,
				     txe, peer_entry, tmp) {
		efa_rdm_txe_release(txe);
	}

	dlist_foreach_container_safe(&peer->rxe_list,
				     struct efa_rdm_ope,
				     rxe, peer_entry, tmp) {
		efa_rdm_rxe_release(rxe);
	}

	if (peer->flags & EFA_RDM_PEER_HANDSHAKE_QUEUED)
		dlist_remove(&peer->handshake_queued_entry);

	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
		dlist_remove(&peer->rnr_backoff_entry);

#ifdef ENABLE_EFA_POISONING
	efa_rdm_poison_mem_region(peer, sizeof(struct efa_rdm_peer));
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
int efa_rdm_peer_reorder_msg(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep,
			     struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_pke *ooo_entry;
	struct efa_rdm_pke *cur_ooo_entry;
	struct efa_rdm_robuf *robuf;
	uint32_t msg_id;

	assert(efa_rdm_pke_get_base_hdr(pkt_entry)->type >= EFA_RDM_REQ_PKT_BEGIN);

	msg_id = efa_rdm_pke_get_rtm_msg_id(pkt_entry);

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
				"Your job will now abort.\n\n", efa_env.recvwin_size);
			abort();
		}
	}

	if (OFI_LIKELY(efa_env.rx_copy_ooo)) {
		assert(pkt_entry->alloc_type == EFA_RDM_PKE_FROM_EFA_RX_POOL);
		ooo_entry = efa_rdm_pke_clone(pkt_entry, ep->rx_ooo_pkt_pool, EFA_RDM_PKE_FROM_OOO_POOL);
		if (OFI_UNLIKELY(!ooo_entry)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"Unable to allocate rx_pkt_entry for OOO msg\n");
			return -FI_ENOMEM;
		}
		efa_rdm_pke_release_rx(pkt_entry);
	} else {
		ooo_entry = pkt_entry;
	}

	cur_ooo_entry = *ofi_recvwin_get_msg(robuf, msg_id);
	if (cur_ooo_entry) {
		assert(efa_rdm_pkt_type_is_mulreq(efa_rdm_pke_get_base_hdr(cur_ooo_entry)->type));
		assert(efa_rdm_pke_get_rtm_msg_id(cur_ooo_entry) == msg_id);
		assert(efa_rdm_pke_get_rtm_msg_length(cur_ooo_entry) == efa_rdm_pke_get_rtm_msg_length(ooo_entry));
		efa_rdm_pke_append(cur_ooo_entry, ooo_entry);
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
void efa_rdm_peer_proc_pending_items_in_robuf(struct efa_rdm_peer *peer, struct efa_rdm_ep *ep)
{
	struct efa_rdm_pke *pending_pkt;
	int ret = 0;
	uint32_t msg_id;

	while (1) {
		pending_pkt = *ofi_recvwin_peek((&peer->robuf));
		if (!pending_pkt)
			return;

		msg_id = efa_rdm_pke_get_rtm_msg_id(pending_pkt);
		EFA_DBG(FI_LOG_EP_CTRL,
		       "Processing msg_id %d from robuf\n", msg_id);
		/* efa_rdm_pke_proc_rtm_rta will write error cq entry if needed */
		ret = efa_rdm_pke_proc_rtm_rta(pending_pkt);
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

/**
 * @brief Determine which Read based protocol to use for a given peer
 *
 * @param[in] peer		rdm peer
 * @param[in] op		operation type
 * @param[in] flags		the flags that the application used to call fi_* functions
 * @param[in] hmem_info	configured protocol limits
 * @return The read-based protocol to use based on inputs.
 */
int efa_rdm_peer_select_readbase_rtm(struct efa_rdm_peer *peer, int op, uint64_t fi_flags, struct efa_hmem_info *hmem_info)
{
	assert(op == ofi_op_tagged || op == ofi_op_msg);
	if (peer->num_read_msg_in_flight == 0 &&
	    hmem_info->runt_size > peer->num_runt_bytes_in_flight &&
	    !(fi_flags & FI_DELIVERY_COMPLETE)) {
		return (op == ofi_op_tagged) ? EFA_RDM_RUNTREAD_TAGRTM_PKT
					     : EFA_RDM_RUNTREAD_MSGRTM_PKT;
	} else {
		return (op == ofi_op_tagged) ? EFA_RDM_LONGREAD_RTA_TAGRTM_PKT
					     : EFA_RDM_LONGREAD_RTA_MSGRTM_PKT;
	}
}
