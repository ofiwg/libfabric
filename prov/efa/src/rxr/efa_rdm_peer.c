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
	peer->is_self = efa_is_same_addr((struct efa_ep_addr *)ep->core_addr,
					 conn->ep_addr);
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
		FI_WARN_ONCE(&rxr_prov, FI_LOG_EP_CTRL, "Closing EP with unacked CONNREQs in flight\n");

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
		rxr_release_tx_entry(ep, tx_entry);
	}

	dlist_foreach_container_safe(&peer->rx_entry_list,
				     struct rxr_op_entry,
				     rx_entry, peer_entry, tmp) {
		rxr_release_rx_entry(ep, rx_entry);
	}

	if (peer->flags & EFA_RDM_PEER_HANDSHAKE_QUEUED)
		dlist_remove(&peer->handshake_queued_entry);

	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
		dlist_remove(&peer->rnr_backoff_entry);

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(peer, sizeof(struct efa_rdm_peer));
#endif
}
