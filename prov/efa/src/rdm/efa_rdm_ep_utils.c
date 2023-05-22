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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>
#include "efa.h"
#include "efa_av.h"
#include "efa_cq.h"
#include "efa_rdm_msg.h"
#include "efa_rdm_rma.h"
#include "rxr_pkt_cmd.h"
#include "rxr_pkt_type_base.h"
#include "efa_rdm_atomic.h"
#include <infiniband/verbs.h>
#include "rxr_pkt_pool.h"
#include "rxr_tp.h"
#include "efa_cntr.h"
#include "efa_rdm_srx.h"
#include "efa_rdm_cq.h"

struct efa_ep_addr *efa_rdm_ep_raw_addr(struct efa_rdm_ep *ep)
{
	return &ep->base_ep.src_addr;
}

const char *efa_rdm_ep_raw_addr_str(struct efa_rdm_ep *ep, char *buf, size_t *buflen)
{
	return ofi_straddr(buf, buflen, FI_ADDR_EFA, efa_rdm_ep_raw_addr(ep));
}

/**
 * @brief return peer's raw address in #efa_ep_addr
 * 
 * @param[in] ep		end point 
 * @param[in] addr 		libfabric address
 * @returns
 * If peer exists, return peer's raw addrress as pointer to #efa_ep_addr;
 * Otherwise, return NULL
 * @relates efa_rdm_peer
 */
struct efa_ep_addr *efa_rdm_ep_get_peer_raw_addr(struct efa_rdm_ep *ep, fi_addr_t addr)
{
	struct efa_av *efa_av;
	struct efa_conn *efa_conn;

	efa_av = ep->base_ep.av;
	efa_conn = efa_av_addr_to_conn(efa_av, addr);
	return efa_conn ? efa_conn->ep_addr : NULL;
}

/**
 * @brief return peer's ahn
 *
 * @param[in] ep		end point
 * @param[in] addr 		libfabric address
 * @returns
 * If peer exists, return peer's ahn
 * Otherwise, return -1
 */
int32_t efa_rdm_ep_get_peer_ahn(struct efa_rdm_ep *ep, fi_addr_t addr)
{
	struct efa_av *efa_av;
	struct efa_conn *efa_conn;

	efa_av = ep->base_ep.av;
	efa_conn = efa_av_addr_to_conn(efa_av, addr);
	return efa_conn ? efa_conn->ah->ahn : -1;
}

/**
 * @brief return peer's raw address in a reable string
 * 
 * @param[in] ep		end point 
 * @param[in] addr 		libfabric address
 * @param[out] buf		a buffer tat to be used to store string
 * @param[in,out] buflen	length of `buf` as input. length of the string as output.
 * @relates efa_rdm_peer
 * @return a string with peer's raw address
 */
const char *efa_rdm_ep_get_peer_raw_addr_str(struct efa_rdm_ep *ep, fi_addr_t addr, char *buf, size_t *buflen)
{
	return ofi_straddr(buf, buflen, FI_ADDR_EFA, efa_rdm_ep_get_peer_raw_addr(ep, addr));
}

/**
 * @brief get pointer to efa_rdm_peer structure for a given libfabric address
 * 
 * @param[in]		ep		endpoint 
 * @param[in]		addr 		libfabric address
 * @returns if peer exists, return pointer to #efa_rdm_peer;
 *          otherwise, return NULL.
 */
struct efa_rdm_peer *efa_rdm_ep_get_peer(struct efa_rdm_ep *ep, fi_addr_t addr)
{
	struct util_av_entry *util_av_entry;
	struct efa_av_entry *av_entry;

	if (OFI_UNLIKELY(addr == FI_ADDR_NOTAVAIL))
		return NULL;

	util_av_entry = ofi_bufpool_get_ibuf(ep->base_ep.util_ep.av->av_entry_pool,
	                                     addr);
	av_entry = (struct efa_av_entry *)util_av_entry->data;
	return av_entry->conn.ep_addr ? &av_entry->conn.rdm_peer : NULL;
}

/**
 * @brief allocate an rxe for an operation
 *
 * @param ep[in]	end point
 * @param addr[in]	fi address of the sender/requester.
 * @param op[in]	operation type (ofi_op_msg/ofi_op_tagged/ofi_op_read/ofi_op_write/ofi_op_atomic_xxx)
 * @return		if allocation succeeded, return pointer to rxe
 * 			if allocation failed, return NULL
 */
struct efa_rdm_ope *efa_rdm_ep_alloc_rxe(struct efa_rdm_ep *ep, fi_addr_t addr, uint32_t op)
{
	struct efa_rdm_ope *rxe;

	rxe = ofi_buf_alloc(ep->ope_pool);
	if (OFI_UNLIKELY(!rxe)) {
		EFA_WARN(FI_LOG_EP_CTRL, "RX entries exhausted\n");
		return NULL;
	}
	memset(rxe, 0, sizeof(struct efa_rdm_ope));

	rxe->ep = ep;
	dlist_insert_tail(&rxe->ep_entry, &ep->rxe_list);
	rxe->type = EFA_RDM_RXE;
	rxe->rx_id = ofi_buf_index(rxe);
	dlist_init(&rxe->queued_pkts);

	rxe->state = EFA_RDM_RXE_INIT;
	rxe->addr = addr;
	if (addr != FI_ADDR_UNSPEC) {
		rxe->peer = efa_rdm_ep_get_peer(ep, addr);
		assert(rxe->peer);
		dlist_insert_tail(&rxe->peer_entry, &rxe->peer->rxe_list);
	} else {
		/*
		 * If msg->addr is not provided, rxe->peer will be set
		 * after it is matched with a message.
		 */
		assert(op == ofi_op_msg || op == ofi_op_tagged);
		rxe->peer = NULL;
	}

	rxe->bytes_runt = 0;
	rxe->bytes_received_via_mulreq = 0;
	rxe->cuda_copy_method = EFA_RDM_CUDA_COPY_UNSPEC;
	rxe->efa_outstanding_tx_ops = 0;
	rxe->op = op;
	rxe->peer_rxe = NULL;

	switch (op) {
	case ofi_op_tagged:
		rxe->cq_entry.flags = (FI_RECV | FI_MSG | FI_TAGGED);
		break;
	case ofi_op_msg:
		rxe->cq_entry.flags = (FI_RECV | FI_MSG);
		break;
	case ofi_op_read_rsp:
		rxe->cq_entry.flags = (FI_REMOTE_READ | FI_RMA);
		break;
	case ofi_op_write:
		rxe->cq_entry.flags = (FI_REMOTE_WRITE | FI_RMA);
		break;
	case ofi_op_atomic:
		rxe->cq_entry.flags = (FI_REMOTE_WRITE | FI_ATOMIC);
		break;
	case ofi_op_atomic_fetch:
	case ofi_op_atomic_compare:
		rxe->cq_entry.flags = (FI_REMOTE_READ | FI_ATOMIC);
		break;
	default:
		EFA_WARN(FI_LOG_EP_CTRL,
			"Unknown operation while %s\n", __func__);
		assert(0 && "Unknown operation");
	}

	return rxe;
}

/**
 * @brief post user provided receiving buffer to the device.
 *
 * The user receive buffer was converted to an RX packet, then posted to the device.
 *
 * @param[in]	ep		endpint
 * @param[in]	rxe	rxe that contain user buffer information
 * @param[in]	flags		user supplied flags passed to fi_recv
 */
int efa_rdm_ep_post_user_recv_buf(struct efa_rdm_ep *ep, struct efa_rdm_ope *rxe, uint64_t flags)
{
	struct rxr_pkt_entry *pkt_entry;
	struct efa_mr *mr;
	int err;

	assert(rxe->iov_count == 1);
	assert(rxe->iov[0].iov_len >= ep->msg_prefix_size);
	pkt_entry = (struct rxr_pkt_entry *)rxe->iov[0].iov_base;
	assert(pkt_entry);

	/*
	 * The ownership of the prefix buffer lies with the application, do not
	 * put it on the dbg list for cleanup during shutdown or poison it. The
	 * provider loses jurisdiction over it soon after writing the rx
	 * completion.
	 */
	dlist_init(&pkt_entry->entry);
	mr = (struct efa_mr *)rxe->desc[0];
	pkt_entry->mr = &mr->mr_fid;
	pkt_entry->alloc_type = RXR_PKT_FROM_USER_BUFFER;
	pkt_entry->flags = RXR_PKT_ENTRY_IN_USE;
	pkt_entry->next = NULL;
	/*
	 * The actual receiving buffer size (pkt_size) is
	 *    rxe->total_len - sizeof(struct rxr_pkt_entry)
	 * because the first part of user buffer was used to
	 * construct pkt_entry. The actual receiving buffer
	 * posted to device starts from pkt_entry->wiredata.
	 */
	pkt_entry->pkt_size = rxe->iov[0].iov_len - sizeof(struct rxr_pkt_entry);

	pkt_entry->ope = rxe;
	rxe->state = EFA_RDM_RXE_MATCHED;

	err = rxr_pkt_entry_recv(ep, pkt_entry, rxe->desc, flags);
	if (OFI_UNLIKELY(err)) {
		rxr_pkt_entry_release_rx(ep, pkt_entry);
		EFA_WARN(FI_LOG_EP_CTRL,
			"failed to post user supplied buffer %d (%s)\n", -err,
			fi_strerror(-err));
		return err;
	}

	ep->efa_rx_pkts_posted++;
	return 0;
}



/* create a new txe */
struct efa_rdm_ope *efa_rdm_ep_alloc_txe(struct efa_rdm_ep *efa_rdm_ep,
					   const struct fi_msg *msg,
					   uint32_t op,
					   uint64_t tag,
					   uint64_t flags)
{
	struct efa_rdm_ope *txe;

	txe = ofi_buf_alloc(efa_rdm_ep->ope_pool);
	if (OFI_UNLIKELY(!txe)) {
		EFA_DBG(FI_LOG_EP_CTRL, "TX entries exhausted.\n");
		return NULL;
	}

	efa_rdm_txe_construct(txe, efa_rdm_ep, msg, op, flags);
	if (op == ofi_op_tagged) {
		txe->cq_entry.tag = tag;
		txe->tag = tag;
	}

	dlist_insert_tail(&txe->ep_entry, &efa_rdm_ep->txe_list);
	return txe;
}

/*
 * For a given peer, trigger a handshake packet and determine if both peers
 * support rdma read.
 *
 * @param[in,out]	ep	efa_rdm_ep
 * @param[in]		addr	remote address
 * @param[in,out]	peer	remote peer
 * @return 		1 if supported, 0 if not, negative errno on error
 */
int efa_rdm_ep_determine_rdma_read_support(struct efa_rdm_ep *ep, fi_addr_t addr,
				       struct efa_rdm_peer *peer)
{
	int ret;

	if (!peer->is_local) {
		ret = rxr_pkt_trigger_handshake(ep, addr, peer);
		if (OFI_UNLIKELY(ret))
			return ret;

		if (!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED))
			return -FI_EAGAIN;
	}

	if (!efa_both_support_rdma_read(ep, peer))
		return 0;

	return 1;
}

/**
 * @brief determine if both peers support rdma write.
 *
 * If no prior communication with the given peer, and we support RDMA WRITE
 * locally, then trigger a handshake packet and determine if both peers
 * support rdma write.
 *
 * @param[in,out]	ep	efa_rdm_ep
 * @param[in]		addr	remote address
 * @param[in,out]	peer	remote peer
 * @return 		1 if supported, 0 if not, negative errno on error
 */
int efa_rdm_ep_determine_rdma_write_support(struct efa_rdm_ep *ep, fi_addr_t addr,
					struct efa_rdm_peer *peer)
{
	int ret;

	/* no need to trigger handshake if we don't support RDMA WRITE locally */
	if (!efa_rdm_ep_support_rdma_write(ep))
		return false;

	if (!peer->is_local) {
		ret = rxr_pkt_trigger_handshake(ep, addr, peer);
		if (OFI_UNLIKELY(ret))
			return ret;

		if (!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED))
			return -FI_EAGAIN;
	}

	if (!efa_both_support_rdma_write(ep, peer))
		return 0;

	return 1;
}

/**
 * @brief record the event that a TX op has been submitted
 *
 * This function is called after a TX operation has been posted
 * successfully. It will:
 *
 *  1. increase the outstanding tx_op counter in endpoint and
 *     in the peer structure.
 *
 *  2. add the TX packet to peer's outstanding TX packet list.
 *
 * Both send and read are considered TX operation.
 *
 * The tx_op counters used to prevent over posting the device
 * and used in flow control. They are also usefull for debugging.
 *
 * Peer's outstanding TX packet list is used when removing a peer
 * to invalidate address of these packets, so that the completion
 * of these packet is ignored.
 *
 * @param[in,out]	ep		endpoint
 * @param[in]		pkt_entry	TX pkt_entry, which contains
 * 					the info of the TX op.
 */
void efa_rdm_ep_record_tx_op_submitted(struct efa_rdm_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_peer *peer;
	struct efa_rdm_ope *ope;

	ope = pkt_entry->ope;
	/*
	 * peer can be NULL when the pkt_entry is a RMA_CONTEXT_PKT,
	 * and the RMA is a local read toward the endpoint itself
	 */
	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	if (peer)
		dlist_insert_tail(&pkt_entry->entry,
				  &peer->outstanding_tx_pkts);

	assert(pkt_entry->alloc_type == RXR_PKT_FROM_EFA_TX_POOL);
	ep->efa_outstanding_tx_ops++;
	if (peer)
		peer->efa_outstanding_tx_ops++;

	if (ope)
		ope->efa_outstanding_tx_ops++;
#if ENABLE_DEBUG
	ep->efa_total_posted_tx_ops++;
#endif
}

/**
 * @brief record the event that an TX op is completed
 *
 * This function is called when the completion of
 * a TX operation is received. It will
 *
 * 1. decrease the outstanding tx_op counter in the endpoint
 *    and in the peer.
 *
 * 2. remove the TX packet from peer's outstanding
 *    TX packet list.
 *
 * Both send and read are considered TX operation.
 *
 * One may ask why this function is not integrated
 * into rxr_pkt_entry_relase_tx()?
 *
 * The reason is the action of decrease tx_op counter
 * is not tied to releasing a TX pkt_entry.
 *
 * Sometimes we need to decreate the tx_op counter
 * without releasing a TX pkt_entry. For example,
 * we handle a TX pkt_entry encountered RNR. We need
 * to decrease the tx_op counter and queue the packet.
 *
 * Sometimes we need release TX pkt_entry without
 * decreasing the tx_op counter. For example, when
 * rxr_pkt_post() failed to post a pkt entry.
 *
 * @param[in,out]	ep		endpoint
 * @param[in]		pkt_entry	TX pkt_entry, which contains
 * 					the info of the TX op
 */
void efa_rdm_ep_record_tx_op_completed(struct efa_rdm_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *ope = NULL;
	struct efa_rdm_peer *peer;

	ope = pkt_entry->ope;
	/*
	 * peer can be NULL when:
	 *
	 * 1. the pkt_entry is a RMA_CONTEXT_PKT, and the RMA op is a local read
	 *    toward the endpoint itself.
	 * 2. peer's address has been removed from address vector. Either because
	 *    a new peer has the same GID+QPN was inserted to address, or because
	 *    application removed the peer from address vector.
	 */
	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	if (peer)
		dlist_remove(&pkt_entry->entry);

	assert(pkt_entry->alloc_type == RXR_PKT_FROM_EFA_TX_POOL);
	ep->efa_outstanding_tx_ops--;
	if (peer)
		peer->efa_outstanding_tx_ops--;

	if (ope)
		ope->efa_outstanding_tx_ops--;
}

/* @brief Queue a packet that encountered RNR error and setup RNR backoff
 *
 * We uses an exponential backoff strategy to handle RNR errors.
 *
 * `Backoff` means if a peer encountered RNR, an endpoint will
 * wait a period of time before sending packets to the peer again
 *
 * `Exponential` means the more RNR encountered, the longer the
 * backoff wait time will be.
 *
 * To quantify how long a peer stay in backoff mode, two parameters
 * are defined:
 *
 *    rnr_backoff_begin_ts (ts is timestamp) and rnr_backoff_wait_time.
 *
 * A peer stays in backoff mode until:
 *
 * current_timestamp >= (rnr_backoff_begin_ts + rnr_backoff_wait_time),
 *
 * with one exception: a peer can got out of backoff mode early if a
 * packet's send completion to this peer was reported by the device.
 *
 * Specifically, the implementation of RNR backoff is:
 *
 * For a peer, the first time RNR is encountered, the packet will
 * be resent immediately.
 *
 * The second time RNR is encountered, the endpoint will put the
 * peer in backoff mode, and initialize rnr_backoff_begin_timestamp
 * and rnr_backoff_wait_time.
 *
 * The 3rd and following time RNR is encounter, the RNR will be handled
 * like this:
 *
 *     If peer is already in backoff mode, rnr_backoff_begin_ts
 *     will be updated
 *
 *     Otherwise, peer will be put in backoff mode again,
 *     rnr_backoff_begin_ts will be updated and rnr_backoff_wait_time
 *     will be doubled until it reached maximum wait time.
 *
 * @param[in]	ep		endpoint
 * @param[in]	list		queued RNR packet list
 * @param[in]	pkt_entry	packet entry that encounter RNR
 */
void efa_rdm_ep_queue_rnr_pkt(struct efa_rdm_ep *ep,
			  struct dlist_entry *list,
			  struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_peer *peer;
	static const int random_min_timeout = 40;
	static const int random_max_timeout = 120;

#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
	dlist_insert_tail(&pkt_entry->entry, list);

	peer = efa_rdm_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	if (!(pkt_entry->flags & RXR_PKT_ENTRY_RNR_RETRANSMIT)) {
		/* This is the first time this packet encountered RNR,
		 * we are NOT going to put the peer in backoff mode just yet.
		 */
		pkt_entry->flags |= RXR_PKT_ENTRY_RNR_RETRANSMIT;
		peer->rnr_queued_pkt_cnt++;
		return;
	}

	/* This packet has encountered RNR multiple times, therefore the peer
	 * need to be in backoff mode.
	 *
	 * If the peer is already in backoff mode, we just need to update the
	 * RNR backoff begin time.
	 *
	 * Otherwise, we need to put the peer in backoff mode and set up backoff
	 * begin time and wait time.
	 */
	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF) {
		peer->rnr_backoff_begin_ts = ofi_gettime_us();
		return;
	}

	peer->flags |= EFA_RDM_PEER_IN_BACKOFF;
	dlist_insert_tail(&peer->rnr_backoff_entry,
			  &ep->peer_backoff_list);

	peer->rnr_backoff_begin_ts = ofi_gettime_us();
	if (peer->rnr_backoff_wait_time == 0) {
		if (efa_env.rnr_backoff_initial_wait_time > 0)
			peer->rnr_backoff_wait_time = efa_env.rnr_backoff_initial_wait_time;
		else
			peer->rnr_backoff_wait_time = MAX(random_min_timeout,
							  rand() %
							  random_max_timeout);

		EFA_DBG(FI_LOG_EP_DATA,
		       "initializing backoff timeout for peer: %" PRIu64
		       " timeout: %ld rnr_queued_pkts: %d\n",
		       pkt_entry->addr, peer->rnr_backoff_wait_time,
		       peer->rnr_queued_pkt_cnt);
	} else {
		peer->rnr_backoff_wait_time = MIN(peer->rnr_backoff_wait_time * 2,
						  efa_env.rnr_backoff_wait_time_cap);
		EFA_DBG(FI_LOG_EP_DATA,
		       "increasing backoff timeout for peer: %" PRIu64
		       " to %ld rnr_queued_pkts: %d\n",
		       pkt_entry->addr, peer->rnr_backoff_wait_time,
		       peer->rnr_queued_pkt_cnt);
	}
}