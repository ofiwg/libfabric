/*
 * Copyright (c) 2019-2020 Amazon.com, Inc. or its affiliates.
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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>

#include "rxr.h"
#include "efa.h"
#include "rxr_msg.h"
#include "rxr_rma.h"

/*
 *   General purpose utility functions
 */
struct rxr_pkt_entry *rxr_pkt_entry_alloc(struct rxr_ep *ep,
					  struct ofi_bufpool *pkt_pool)
{
	struct rxr_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ofi_buf_alloc_ex(pkt_pool, &mr);
	if (!pkt_entry)
		return NULL;
#ifdef ENABLE_EFA_POISONING
	memset(pkt_entry, 0, sizeof(*pkt_entry));
#endif
	dlist_init(&pkt_entry->entry);
#if ENABLE_DEBUG
	dlist_init(&pkt_entry->dbg_entry);
#endif
	pkt_entry->mr = (struct fid_mr *)mr;
	pkt_entry->pkt = (struct rxr_pkt *)((char *)pkt_entry +
			  sizeof(*pkt_entry));
#ifdef ENABLE_EFA_POISONING
	memset(pkt_entry->pkt, 0, ep->mtu_size);
#endif
	pkt_entry->state = RXR_PKT_ENTRY_IN_USE;

	return pkt_entry;
}

void rxr_pkt_entry_release_tx(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt)
{
	struct rxr_peer *peer;

#if ENABLE_DEBUG
	dlist_remove(&pkt->dbg_entry);
#endif
	/*
	 * Decrement rnr_queued_pkts counter and reset backoff for this peer if
	 * we get a send completion for a retransmitted packet.
	 */
	if (OFI_UNLIKELY(pkt->state == RXR_PKT_ENTRY_RNR_RETRANSMIT)) {
		peer = rxr_ep_get_peer(ep, pkt->addr);
		peer->rnr_queued_pkt_cnt--;
		peer->timeout_interval = 0;
		peer->rnr_timeout_exp = 0;
		if (peer->rnr_state & RXR_PEER_IN_BACKOFF)
			dlist_remove(&peer->rnr_entry);
		peer->rnr_state = 0;
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "reset backoff timer for peer: %" PRIu64 "\n",
		       pkt->addr);
	}
#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region((uint32_t *)pkt, ep->tx_pkt_pool_entry_sz);
#endif
	pkt->state = RXR_PKT_ENTRY_FREE;
	ofi_buf_free(pkt);
}

void rxr_pkt_entry_release_rx(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry)
{
	if (pkt_entry->type == RXR_PKT_ENTRY_POSTED) {
		struct rxr_peer *peer;

		peer = rxr_ep_get_peer(ep, pkt_entry->addr);
		assert(peer);
		if (peer->is_local)
			ep->rx_bufs_shm_to_post++;
		else
			ep->rx_bufs_efa_to_post++;
	}
#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
#ifdef ENABLE_EFA_POISONING
	/* the same pool size is used for all types of rx pkt_entries */
	rxr_poison_mem_region((uint32_t *)pkt_entry, ep->rx_pkt_pool_entry_sz);
#endif
	pkt_entry->state = RXR_PKT_ENTRY_FREE;
	ofi_buf_free(pkt_entry);
}

void rxr_pkt_entry_copy(struct rxr_ep *ep,
			struct rxr_pkt_entry *dest,
			struct rxr_pkt_entry *src,
			enum rxr_pkt_entry_type type)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
	       "Copying packet (type %d) out of posted buffer\n", type);
	assert(src->type == RXR_PKT_ENTRY_POSTED);
	memcpy(dest, src, sizeof(struct rxr_pkt_entry));
	dest->pkt = (struct rxr_pkt *)((char *)dest + sizeof(*dest));
	memcpy(dest->pkt, src->pkt, ep->mtu_size);
	dest->type = type;
	dlist_init(&dest->entry);
#if ENABLE_DEBUG
	dlist_init(&dest->dbg_entry);
#endif
	dest->state = RXR_PKT_ENTRY_IN_USE;
}

/*
 *   Utility functions used to send pkt over wire
 */
static inline
ssize_t rxr_pkt_entry_sendmsg(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			      const struct fi_msg *msg, uint64_t flags)
{
	struct rxr_peer *peer;
	size_t ret;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(ep->tx_pending <= ep->max_outstanding_tx);

	if (ep->tx_pending == ep->max_outstanding_tx)
		return -FI_EAGAIN;

	if (peer->rnr_state & RXR_PEER_IN_BACKOFF)
		return -FI_EAGAIN;

#if ENABLE_DEBUG
	dlist_insert_tail(&pkt_entry->dbg_entry, &ep->tx_pkt_list);
#ifdef ENABLE_RXR_PKT_DUMP
	rxr_pkt_print("Sent", ep, (struct rxr_base_hdr *)pkt_entry->pkt);
#endif
#endif
	if (rxr_env.enable_shm_transfer && peer->is_local) {
		ret = fi_sendmsg(ep->shm_ep, msg, flags);
	} else {
		ret = fi_sendmsg(ep->rdm_ep, msg, flags);
		if (OFI_LIKELY(!ret))
			rxr_ep_inc_tx_pending(ep, peer);
	}

	return ret;
}

ssize_t rxr_pkt_entry_sendv(struct rxr_ep *ep,
			    struct rxr_pkt_entry *pkt_entry,
			    fi_addr_t addr, const struct iovec *iov,
			    void **desc, size_t count, uint64_t flags)
{
	struct fi_msg msg;
	struct rxr_peer *peer;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	peer = rxr_ep_get_peer(ep, addr);
	msg.addr = peer->is_local ? peer->shm_fiaddr : addr;
	msg.context = pkt_entry;
	msg.data = 0;

	return rxr_pkt_entry_sendmsg(ep, pkt_entry, &msg, flags);
}

/* rxr_pkt_start currently expects data pkt right after pkt hdr */
ssize_t rxr_pkt_entry_send_with_flags(struct rxr_ep *ep,
				      struct rxr_pkt_entry *pkt_entry,
				      fi_addr_t addr, uint64_t flags)
{
	struct iovec iov;
	void *desc;

	iov.iov_base = rxr_pkt_start(pkt_entry);
	iov.iov_len = pkt_entry->pkt_size;

	if (rxr_ep_get_peer(ep, addr)->is_local)
		desc = NULL;
	else
		desc = rxr_ep_mr_local(ep) ? fi_mr_desc(pkt_entry->mr) : NULL;

	return rxr_pkt_entry_sendv(ep, pkt_entry, addr, &iov, &desc, 1, flags);
}

ssize_t rxr_pkt_entry_send(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry,
			   fi_addr_t addr)
{
	return rxr_pkt_entry_send_with_flags(ep, pkt_entry, addr, 0);
}

ssize_t rxr_pkt_entry_inject(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry,
			     fi_addr_t addr)
{
	struct rxr_peer *peer;

	/* currently only EOR packet is injected using shm ep */
	peer = rxr_ep_get_peer(ep, addr);
	assert(peer);
	assert(rxr_env.enable_shm_transfer && peer->is_local);
	return fi_inject(ep->shm_ep, rxr_pkt_start(pkt_entry), pkt_entry->pkt_size,
			 peer->shm_fiaddr);
}
