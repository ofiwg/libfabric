/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
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

#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <ofi_iov.h>
#include <ofi_recvwin.h>
#include "rxr.h"
#include "rxr_rma.h"
#include "rxr_cntr.h"
#include "efa.h"

static const char *rxr_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
				   const void *err_data, char *buf, size_t len)
{
	struct fid_list_entry *fid_entry;
	struct util_ep *util_ep;
	struct util_cq *cq;
	struct rxr_ep *ep;
	const char *str;

	cq = container_of(cq_fid, struct util_cq, cq_fid);

	fastlock_acquire(&cq->ep_list_lock);
	assert(!dlist_empty(&cq->ep_list));
	fid_entry = container_of(cq->ep_list.next,
				 struct fid_list_entry, entry);
	util_ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
	ep = container_of(util_ep, struct rxr_ep, util_ep);

	str = fi_cq_strerror(ep->rdm_cq, prov_errno, err_data, buf, len);
	fastlock_release(&cq->ep_list_lock);
	return str;
}

/*
 * Teardown rx_entry and write an error cq entry. With our current protocol we
 * will only encounter an RX error when sending a queued RTS or CTS packet or
 * if we are sending a CTS message. Because of this, the sender will not send
 * any additional data packets if the receiver encounters an error. If there is
 * a scenario in the future where the sender will continue to send data packets
 * we need to prevent rx_id mismatch. Ideally, we should add a NACK message and
 * tear down both RX and TX entires although whatever caused the error may
 * prevent that.
 *
 * TODO: add a NACK message to tear down state on sender side
 */
int rxr_cq_handle_rx_error(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
			   ssize_t prov_errno)
{
	struct fi_cq_err_entry err_entry;
	struct util_cq *util_cq;
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;

	memset(&err_entry, 0, sizeof(err_entry));

	util_cq = ep->util_ep.rx_cq;

	err_entry.err = FI_EIO;
	err_entry.prov_errno = (int)prov_errno;

	switch (rx_entry->state) {
	case RXR_RX_INIT:
	case RXR_RX_UNEXP:
		dlist_remove(&rx_entry->entry);
		break;
	case RXR_RX_MATCHED:
		break;
	case RXR_RX_RECV:
#if ENABLE_DEBUG
		dlist_remove(&rx_entry->rx_pending_entry);
#endif
		break;
	case RXR_RX_QUEUED_CTRL:
	case RXR_RX_QUEUED_CTS_RNR:
	case RXR_RX_QUEUED_SHM_LARGE_READ:
	case RXR_RX_QUEUED_EOR:
		dlist_remove(&rx_entry->queued_entry);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "rx_entry unknown state %d\n",
			rx_entry->state);
		assert(0 && "rx_entry unknown state");
	}

	dlist_foreach_container_safe(&rx_entry->queued_pkts,
				     struct rxr_pkt_entry,
				     pkt_entry, entry, tmp)
		rxr_release_tx_pkt_entry(ep, pkt_entry);

	if (rx_entry->unexp_rts_pkt) {
		rxr_release_rx_pkt_entry(ep, rx_entry->unexp_rts_pkt);
		rx_entry->unexp_rts_pkt = NULL;
	}

	if (rx_entry->fi_flags & FI_MULTI_RECV)
		rxr_cq_handle_multi_recv_completion(ep, rx_entry);

	err_entry.flags = rx_entry->cq_entry.flags;
	if (rx_entry->state != RXR_RX_UNEXP)
		err_entry.op_context = rx_entry->cq_entry.op_context;
	err_entry.buf = rx_entry->cq_entry.buf;
	err_entry.data = rx_entry->cq_entry.data;
	err_entry.tag = rx_entry->cq_entry.tag;

	rxr_multi_recv_free_posted_entry(ep, rx_entry);

        FI_WARN(&rxr_prov, FI_LOG_CQ,
		"rxr_cq_handle_rx_error: err: %d, prov_err: %s (%d)\n",
		err_entry.err, fi_strerror(-err_entry.prov_errno),
		err_entry.prov_errno);

	/*
	 * TODO: We can't free the rx_entry as we may receive additional
	 * packets for this entry. Add ref counting so the rx_entry can safely
	 * be freed once all packets are accounted for.
	 */
	//rxr_release_rx_entry(ep, rx_entry);

	rxr_cntr_report_error(ep, err_entry.flags);
	return ofi_cq_write_error(util_cq, &err_entry);
}

/*
 * Teardown tx_entry and write an error cq entry. With our current protocol the
 * receiver will only send a CTS once the window is exhausted, meaning that all
 * data packets for that window will have been received successfully. This
 * means that the receiver will not send any CTS packets if the sender
 * encounters and error sending data packets. If that changes in the future we
 * will need to be careful to prevent tx_id mismatch.
 *
 * TODO: add NACK message to tear down receive side state
 */
int rxr_cq_handle_tx_error(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
			   ssize_t prov_errno)
{
	struct fi_cq_err_entry err_entry;
	struct util_cq *util_cq;
	uint32_t api_version;
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;

	memset(&err_entry, 0, sizeof(err_entry));

	util_cq = ep->util_ep.tx_cq;
	api_version = util_cq->domain->fabric->fabric_fid.api_version;

	err_entry.err = FI_EIO;
	err_entry.prov_errno = (int)prov_errno;

	switch (tx_entry->state) {
	case RXR_TX_RTS:
	case RXR_TX_SHM_RMA:
		break;
	case RXR_TX_SEND:
		dlist_remove(&tx_entry->entry);
		break;
	case RXR_TX_QUEUED_CTRL:
	case RXR_TX_QUEUED_SHM_RMA:
	case RXR_TX_QUEUED_RTS_RNR:
	case RXR_TX_QUEUED_DATA_RNR:
		dlist_remove(&tx_entry->queued_entry);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "tx_entry unknown state %d\n",
			tx_entry->state);
		assert(0 && "tx_entry unknown state");
	}

	dlist_foreach_container_safe(&tx_entry->queued_pkts,
				     struct rxr_pkt_entry,
				     pkt_entry, entry, tmp)
		rxr_release_tx_pkt_entry(ep, pkt_entry);

	err_entry.flags = tx_entry->cq_entry.flags;
	err_entry.op_context = tx_entry->cq_entry.op_context;
	err_entry.buf = tx_entry->cq_entry.buf;
	err_entry.data = tx_entry->cq_entry.data;
	err_entry.tag = tx_entry->cq_entry.tag;
	if (FI_VERSION_GE(api_version, FI_VERSION(1, 5)))
		err_entry.err_data_size = 0;

	FI_WARN(&rxr_prov, FI_LOG_CQ,
		"rxr_cq_handle_tx_error: err: %d, prov_err: %s (%d)\n",
		err_entry.err, fi_strerror(-err_entry.prov_errno),
		err_entry.prov_errno);

	/*
	 * TODO: We can't free the tx_entry as we may receive a control packet
	 * packet for this entry. Add ref counting so the tx_entry can safely
	 * be freed once all packets are accounted for.
	 */
	//rxr_release_tx_entry(ep, tx_entry);

	rxr_cntr_report_error(ep, tx_entry->cq_entry.flags);
	return ofi_cq_write_error(util_cq, &err_entry);
}

/*
 * Queue a packet on the appropriate list when an RNR error is received.
 */
static inline void rxr_cq_queue_pkt(struct rxr_ep *ep,
				    struct dlist_entry *list,
				    struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);

	/*
	 * Queue the packet if it has not been retransmitted yet.
	 */
	if (pkt_entry->state != RXR_PKT_ENTRY_RNR_RETRANSMIT) {
		pkt_entry->state = RXR_PKT_ENTRY_RNR_RETRANSMIT;
		peer->rnr_queued_pkt_cnt++;
		goto queue_pkt;
	}

	/*
	 * Otherwise, increase the backoff if the peer is already not in
	 * backoff. Reset the timer when starting backoff or if another RNR for
	 * a retransmitted packet is received while waiting for the timer to
	 * expire.
	 */
	peer->rnr_ts = fi_gettime_us();
	if (peer->rnr_state & RXR_PEER_IN_BACKOFF)
		goto queue_pkt;

	peer->rnr_state |= RXR_PEER_IN_BACKOFF;

	if (!peer->timeout_interval) {
		if (rxr_env.timeout_interval)
			peer->timeout_interval = rxr_env.timeout_interval;
		else
			peer->timeout_interval = MAX(RXR_RAND_MIN_TIMEOUT,
						     rand() %
						     RXR_RAND_MAX_TIMEOUT);

		peer->rnr_timeout_exp = 1;
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "initializing backoff timeout for peer: %" PRIu64
		       " timeout: %d rnr_queued_pkts: %d\n",
		       pkt_entry->addr, peer->timeout_interval,
		       peer->rnr_queued_pkt_cnt);
	} else {
		/* Only backoff once per peer per progress thread loop. */
		if (!(peer->rnr_state & RXR_PEER_BACKED_OFF)) {
			peer->rnr_state |= RXR_PEER_BACKED_OFF;
			peer->rnr_timeout_exp++;
			FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
			       "increasing backoff for peer: %" PRIu64
			       " rnr_timeout_exp: %d rnr_queued_pkts: %d\n",
			       pkt_entry->addr, peer->rnr_timeout_exp,
			       peer->rnr_queued_pkt_cnt);
		}
	}
	dlist_insert_tail(&peer->rnr_entry,
			  &ep->peer_backoff_list);

queue_pkt:
#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
	dlist_insert_tail(&pkt_entry->entry, list);
}

int rxr_cq_handle_cq_error(struct rxr_ep *ep, ssize_t err)
{
	struct fi_cq_err_entry err_entry;
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	struct rxr_peer *peer;
	ssize_t ret;

	memset(&err_entry, 0, sizeof(err_entry));

	/*
	 * If the cq_read failed with another error besides -FI_EAVAIL or
	 * the cq_readerr fails we don't know if this is an rx or tx error.
	 * We'll write an error eq entry to the event queue instead.
	 */

	err_entry.err = FI_EIO;
	err_entry.prov_errno = (int)err;

	if (err != -FI_EAVAIL) {
		FI_WARN(&rxr_prov, FI_LOG_CQ, "fi_cq_read: %s\n",
			fi_strerror(-err));
		goto write_err;
	}

	ret = fi_cq_readerr(ep->rdm_cq, &err_entry, 0);
	if (ret != sizeof(err_entry)) {
		if (ret < 0) {
			FI_WARN(&rxr_prov, FI_LOG_CQ, "fi_cq_readerr: %s\n",
				fi_strerror(-ret));
			err_entry.prov_errno = ret;
		} else {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"fi_cq_readerr unexpected size %zu expected %zu\n",
				ret, sizeof(err_entry));
			err_entry.prov_errno = -FI_EIO;
		}
		goto write_err;
	}

	if (err_entry.err != -FI_EAGAIN)
		OFI_CQ_STRERROR(&rxr_prov, FI_LOG_WARN, FI_LOG_CQ, ep->rdm_cq,
				&err_entry);

	pkt_entry = (struct rxr_pkt_entry *)err_entry.op_context;
	peer = rxr_ep_get_peer(ep, pkt_entry->addr);

	/*
	 * A connack send could fail at the core provider if the peer endpoint
	 * is shutdown soon after it receives a send completion for the RTS
	 * packet that included src_address. The connack itself is irrelevant if
	 * that happens, so just squelch this error entry and move on without
	 * writing an error completion or event to the application.
	 */
	if (rxr_get_base_hdr(pkt_entry->pkt)->type == RXR_CONNACK_PKT) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Squelching error CQE for RXR_CONNACK_PKT\n");
		/*
		 * CONNACK packets do not have an associated rx/tx entry. Use
		 * the flags instead to determine if this is a send or recv.
		 */
		if (err_entry.flags & FI_SEND) {
			rxr_ep_dec_tx_pending(ep, peer, 1);
			rxr_release_tx_pkt_entry(ep, pkt_entry);
		} else if (err_entry.flags & FI_RECV) {
			rxr_release_rx_pkt_entry(ep, pkt_entry);
		} else {
			assert(0 && "unknown err_entry flags in CONNACK packet");
		}
		return 0;
	}

	if (!pkt_entry->x_entry) {
		/*
		 * A NULL x_entry means this is a recv posted buf pkt_entry.
		 * Since we don't have any context besides the error code,
		 * we will write to the eq instead.
		 */
		rxr_release_rx_pkt_entry(ep, pkt_entry);
		goto write_err;
	}

	/*
	 * If x_entry is set this rx or tx entry error is for a sent
	 * packet. Decrement the tx_pending counter and fall through to
	 * the rx or tx entry handlers.
	 */
	if (!peer->is_local)
		rxr_ep_dec_tx_pending(ep, peer, 1);
	if (RXR_GET_X_ENTRY_TYPE(pkt_entry) == RXR_TX_ENTRY) {
		tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
		if (err_entry.err != -FI_EAGAIN ||
		    rxr_ep_domain(ep)->resource_mgmt == FI_RM_ENABLED) {
			ret = rxr_cq_handle_tx_error(ep, tx_entry,
						     err_entry.prov_errno);
			rxr_release_tx_pkt_entry(ep, pkt_entry);
			return ret;
		}

		rxr_cq_queue_pkt(ep, &tx_entry->queued_pkts, pkt_entry);
		if (tx_entry->state == RXR_TX_SEND) {
			dlist_remove(&tx_entry->entry);
			tx_entry->state = RXR_TX_QUEUED_DATA_RNR;
			dlist_insert_tail(&tx_entry->queued_entry,
					  &ep->tx_entry_queued_list);
		} else if (tx_entry->state == RXR_TX_RTS) {
			tx_entry->state = RXR_TX_QUEUED_RTS_RNR;
			dlist_insert_tail(&tx_entry->queued_entry,
					  &ep->tx_entry_queued_list);
		}
		return 0;
	} else if (RXR_GET_X_ENTRY_TYPE(pkt_entry) == RXR_RX_ENTRY) {
		rx_entry = (struct rxr_rx_entry *)pkt_entry->x_entry;
		if (err_entry.err != -FI_EAGAIN ||
		    rxr_ep_domain(ep)->resource_mgmt == FI_RM_ENABLED) {
			ret = rxr_cq_handle_rx_error(ep, rx_entry,
						     err_entry.prov_errno);
			rxr_release_tx_pkt_entry(ep, pkt_entry);
			return ret;
		}
		rxr_cq_queue_pkt(ep, &rx_entry->queued_pkts, pkt_entry);
		if (rx_entry->state == RXR_RX_RECV) {
			rx_entry->state = RXR_RX_QUEUED_CTS_RNR;
			dlist_insert_tail(&rx_entry->queued_entry,
					  &ep->rx_entry_queued_list);
		}
		return 0;
	}

	FI_WARN(&rxr_prov, FI_LOG_CQ,
		"%s unknown x_entry state %d\n",
		__func__, RXR_GET_X_ENTRY_TYPE(pkt_entry));
	assert(0 && "unknown x_entry state");
write_err:
	rxr_eq_write_error(ep, err_entry.err, err_entry.prov_errno);
	return 0;
}

static int rxr_cq_match_recv(struct dlist_entry *item, const void *arg)
{
	const struct rxr_pkt_entry *pkt_entry = arg;
	struct rxr_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);

	return rxr_match_addr(rx_entry->addr, pkt_entry->addr);
}

static int rxr_cq_match_trecv(struct dlist_entry *item, const void *arg)
{
	struct rxr_pkt_entry *pkt_entry = (struct rxr_pkt_entry *)arg;
	struct rxr_rx_entry *rx_entry;
	uint64_t match_tag;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);

	match_tag = rxr_get_rts_hdr(pkt_entry->pkt)->tag;

	return rxr_match_addr(rx_entry->addr, pkt_entry->addr) &&
	       rxr_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     match_tag);
}

static void rxr_cq_post_connack(struct rxr_ep *ep,
				struct rxr_peer *peer,
				fi_addr_t addr)
{
	struct rxr_pkt_entry *pkt_entry;
	ssize_t ret;

	if (peer->state == RXR_PEER_ACKED)
		return;

	pkt_entry = rxr_get_pkt_entry(ep, ep->tx_pkt_efa_pool);
	if (OFI_UNLIKELY(!pkt_entry))
		return;

	rxr_ep_init_connack_pkt_entry(ep, pkt_entry, addr);

	/*
	 * TODO: Once we start using a core's selective completion capability,
	 * post the CONNACK packets without FI_COMPLETION.
	 */
	ret = rxr_ep_send_pkt(ep, pkt_entry, addr);

	/*
	 * Skip sending this connack on error and try again when processing the
	 * next RTS from this peer containing the source information
	 */
	if (OFI_UNLIKELY(ret)) {
		rxr_release_tx_pkt_entry(ep, pkt_entry);
		if (ret == -FI_EAGAIN)
			return;
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Failed to send a CONNACK packet: ret %zd\n", ret);
	} else {
		peer->state = RXR_PEER_ACKED;
	}

	return;
}

void rxr_cq_write_rx_completion(struct rxr_ep *ep,
				struct rxr_rx_entry *rx_entry)
{
	struct util_cq *rx_cq = ep->util_ep.rx_cq;
	int ret = 0;
	if (OFI_UNLIKELY(rx_entry->cq_entry.len < rx_entry->total_len)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Message truncated: tag: %"PRIu64" len: %"PRIu64" total_len: %zu\n",
			rx_entry->cq_entry.tag,	rx_entry->total_len,
			rx_entry->cq_entry.len);

		ret = ofi_cq_write_error_trunc(ep->util_ep.rx_cq,
					       rx_entry->cq_entry.op_context,
					       rx_entry->cq_entry.flags,
					       rx_entry->total_len,
					       rx_entry->cq_entry.buf,
					       rx_entry->cq_entry.data,
					       rx_entry->cq_entry.tag,
					       rx_entry->total_len -
					       rx_entry->cq_entry.len);

		rxr_rm_rx_cq_check(ep, rx_cq);

		if (OFI_UNLIKELY(ret))
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"Unable to write recv error cq: %s\n",
				fi_strerror(-ret));

		rxr_cntr_report_error(ep, rx_entry->cq_entry.flags);
		return;
	}

	if (!(rx_entry->rxr_flags & RXR_RECV_CANCEL) &&
	    (ofi_need_completion(rxr_rx_flags(ep), rx_entry->fi_flags) ||
	     (rx_entry->cq_entry.flags & FI_MULTI_RECV))) {
		FI_DBG(&rxr_prov, FI_LOG_CQ,
		       "Writing recv completion for rx_entry from peer: %"
		       PRIu64 " rx_id: %" PRIu32 " msg_id: %" PRIu32
		       " tag: %lx total_len: %" PRIu64 "\n",
		       rx_entry->addr, rx_entry->rx_id, rx_entry->msg_id,
		       rx_entry->cq_entry.tag, rx_entry->total_len);

		if (ep->util_ep.caps & FI_SOURCE)
			ret = ofi_cq_write_src(rx_cq,
					       rx_entry->cq_entry.op_context,
					       rx_entry->cq_entry.flags,
					       rx_entry->cq_entry.len,
					       rx_entry->cq_entry.buf,
					       rx_entry->cq_entry.data,
					       rx_entry->cq_entry.tag,
					       rx_entry->addr);
		else
			ret = ofi_cq_write(rx_cq,
					   rx_entry->cq_entry.op_context,
					   rx_entry->cq_entry.flags,
					   rx_entry->cq_entry.len,
					   rx_entry->cq_entry.buf,
					   rx_entry->cq_entry.data,
					   rx_entry->cq_entry.tag);

		rxr_rm_rx_cq_check(ep, rx_cq);

		if (OFI_UNLIKELY(ret)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"Unable to write recv completion: %s\n",
				fi_strerror(-ret));
			if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
				assert(0 && "failed to write err cq entry");
			return;
		}
	}

	rxr_cntr_report_rx_completion(ep, rx_entry);
}

void rxr_cq_handle_rx_completion(struct rxr_ep *ep,
				 struct rxr_pkt_entry *pkt_entry,
				 struct rxr_rx_entry *rx_entry)
{
	struct rxr_tx_entry *tx_entry = NULL;

	if (rx_entry->cq_entry.flags & FI_WRITE) {
		/*
		 * must be on the remote side, notify cq/counter
		 * if FI_RMA_EVENT is requested or REMOTE_CQ_DATA is on
		 */
		if (rx_entry->cq_entry.flags & FI_REMOTE_CQ_DATA)
			rxr_cq_write_rx_completion(ep, rx_entry);
		else if (ep->util_ep.caps & FI_RMA_EVENT)
			rxr_cntr_report_rx_completion(ep, rx_entry);

		rxr_release_rx_pkt_entry(ep, pkt_entry);
		return;
	}

	if (rx_entry->cq_entry.flags & FI_READ) {
		/* Note for emulated FI_READ, there is an rx_entry on
		 * both initiator side and on remote side.
		 * However, only on the initiator side,
		 * rxr_cq_handle_rx_completion() will be called.
		 * The following shows the sequence of events that
		 * is happening
		 *
		 * Initiator side                    Remote side
		 * create tx_entry
		 * create rx_entry
		 * send rts(with rx_id)
		 *                                receive rts
		 *                                create rx_entry
		 *                                create tx_entry
		 *                                tx_entry sending data
		 * rx_entry receiving data
		 * receive completed              send completed
		 * handle_rx_completion()         handle_pkt_send_completion()
		 * |->write_tx_completion()       |-> if (FI_RMA_EVENT)
		 *                                         write_rx_completion()
		 *
		 * As can be seen, although there is a rx_entry on remote side,
		 * the entry will not enter into rxr_cq_handle_rx_completion
		 * So at this point we must be on the initiator side, we
		 *     1. find the corresponding tx_entry
		 *     2. call rxr_cq_write_tx_completion()
		 */
		tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, rx_entry->rma_loc_tx_id);
		assert(tx_entry->state == RXR_TX_WAIT_READ_FINISH);
		if (tx_entry->fi_flags & FI_COMPLETION) {
			/* Note write_tx_completion() will release tx_entry */
			rxr_cq_write_tx_completion(ep, tx_entry);
		} else {
			rxr_cntr_report_tx_completion(ep, tx_entry);
			rxr_release_tx_entry(ep, tx_entry);
		}

		/*
		 * do not call rxr_release_rx_entry here because
		 * caller will release
		 */
		rxr_release_rx_pkt_entry(ep, pkt_entry);
		return;
	}

	if (rx_entry->fi_flags & FI_MULTI_RECV)
		rxr_cq_handle_multi_recv_completion(ep, rx_entry);

	rxr_cq_write_rx_completion(ep, rx_entry);
	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return;
}

ssize_t rxr_cq_recv_shm_large_message(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_rma_context_pkt *rma_context_pkt;
	struct fi_msg_rma msg;
	struct iovec msg_iov[RXR_IOV_LIMIT];
	struct fi_rma_iov rma_iov[RXR_IOV_LIMIT];
	fi_addr_t src_shm_fiaddr;
	uint64_t remain_len;
	struct rxr_peer *peer;
	int ret, i;

	if (rx_entry->state == RXR_RX_QUEUED_SHM_LARGE_READ)
		return 0;

	pkt_entry = rxr_get_pkt_entry(ep, ep->tx_pkt_shm_pool);
	assert(pkt_entry);

	pkt_entry->x_entry = (void *)rx_entry;
	pkt_entry->addr = rx_entry->addr;
	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->pkt;
	rma_context_pkt->type = RXR_RMA_CONTEXT_PKT;
	rma_context_pkt->version = RXR_PROTOCOL_VERSION;
	rma_context_pkt->rma_context_type = RXR_SHM_LARGE_READ;
	rma_context_pkt->tx_id = rx_entry->tx_id;

	peer = rxr_ep_get_peer(ep, rx_entry->addr);
	src_shm_fiaddr = peer->shm_fiaddr;

	memset(&msg, 0, sizeof(msg));

	remain_len = rx_entry->total_len;

	for (i = 0; i < rx_entry->rma_iov_count; i++) {
		rma_iov[i].addr = rx_entry->rma_iov[i].addr;
		rma_iov[i].len = rx_entry->rma_iov[i].len;
		rma_iov[i].key = 0;
	}

	/*
	 * shm provider will compare #bytes CMA copied with total length of recv buffer
	 * (msg_iov here). If they are not equal, an error is returned when reading shm
	 * provider's cq. So shrink the total length of recv buffer if applicable
	 */
	for (i = 0; i < rx_entry->iov_count; i++) {
		msg_iov[i].iov_base = (void *)rx_entry->iov[i].iov_base;
		msg_iov[i].iov_len = (remain_len < rx_entry->iov[i].iov_len) ?
					remain_len : rx_entry->iov[i].iov_len;
		remain_len -= msg_iov[i].iov_len;
		if (remain_len == 0)
			break;
	}

	msg.msg_iov = msg_iov;
	msg.iov_count = rx_entry->iov_count;
	msg.desc = NULL;
	msg.addr = src_shm_fiaddr;
	msg.context = pkt_entry;
	msg.rma_iov = rma_iov;
	msg.rma_iov_count = rx_entry->rma_iov_count;

	ret = fi_readmsg(ep->shm_ep, &msg, 0);

	return ret;
}

void rxr_cq_process_shm_large_message(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
				      struct rxr_rts_hdr *rts_hdr, char *data)
{
	struct iovec *iovec_ptr;
	int ret, i;

	/* get iov_count of sender first */
	memcpy(&rx_entry->rma_iov_count, data, sizeof(size_t));
	data += sizeof(size_t);

	iovec_ptr = (struct iovec *)data;
	for (i = 0; i < rx_entry->rma_iov_count; i++) {
		iovec_ptr = iovec_ptr + i;
		rx_entry->rma_iov[i].addr = (intptr_t) iovec_ptr->iov_base;
		rx_entry->rma_iov[i].len = iovec_ptr->iov_len;
		rx_entry->rma_iov[i].key = 0;
	}

	ret = rxr_cq_recv_shm_large_message(ep, rx_entry);

	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN) {
			rx_entry->state = RXR_RX_QUEUED_SHM_LARGE_READ;
			dlist_insert_tail(&rx_entry->queued_entry,  &ep->rx_entry_queued_list);
			return;
		}
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"A large message RMA READ failed over shm provider.\n");
		if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
			assert(0 && "failed to write err cq entry");
	}
}

char *rxr_cq_read_rts_hdr(struct rxr_ep *ep,
			  struct rxr_rx_entry *rx_entry,
			  struct rxr_pkt_entry *pkt_entry)
{
	char *data;
	struct rxr_rts_hdr *rts_hdr = NULL;
	/*
	 * Use the correct header and grab CQ data and data, but ignore the
	 * source_address since that has been fetched and processed already
	 */

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	rx_entry->addr = pkt_entry->addr;
	rx_entry->tx_id = rts_hdr->tx_id;
	rx_entry->msg_id = rts_hdr->msg_id;
	rx_entry->total_len = rts_hdr->data_len;
	rx_entry->cq_entry.tag = rts_hdr->tag;

	if (rts_hdr->flags & RXR_REMOTE_CQ_DATA) {
		rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		data = rxr_get_ctrl_cq_pkt(rts_hdr)->data + rts_hdr->addrlen;
		rx_entry->cq_entry.data =
				rxr_get_ctrl_cq_pkt(rts_hdr)->hdr.cq_data;
	} else {
		rx_entry->cq_entry.data = 0;
		data = rxr_get_ctrl_pkt(rts_hdr)->data + rts_hdr->addrlen;
	}

	return data;
}

int rxr_cq_process_msg_rts(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_rts_hdr *rts_hdr;
	struct dlist_entry *match;
	struct rxr_rx_entry *rx_entry;
	char *data;
	size_t data_size;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (rts_hdr->flags & RXR_TAGGED) {
		match = dlist_find_first_match(&ep->rx_tagged_list,
					       &rxr_cq_match_trecv,
					       (void *)pkt_entry);
	} else {
		match = dlist_find_first_match(&ep->rx_list,
					       &rxr_cq_match_recv,
					       (void *)pkt_entry);
	}

	if (OFI_UNLIKELY(!match)) {
		rx_entry = rxr_ep_get_new_unexp_rx_entry(ep, pkt_entry);
		if (!rx_entry) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
			return -FI_ENOBUFS;
		}

		/* we are not releasing pkt_entry here because it will be
		 * processed later
		 */
		pkt_entry = rx_entry->unexp_rts_pkt;
		rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
		rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
		return 0;
	}

	rx_entry = container_of(match, struct rxr_rx_entry, entry);
	if (rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED) {
		rx_entry = rxr_ep_split_rx_entry(ep, rx_entry,
						 NULL, pkt_entry);
		if (OFI_UNLIKELY(!rx_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
			return -FI_ENOBUFS;
		}
	}

	rx_entry->state = RXR_RX_MATCHED;

	if (!(rx_entry->fi_flags & FI_MULTI_RECV) ||
	    !rxr_multi_recv_buffer_available(ep, rx_entry->master_entry))
		dlist_remove(match);

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	data = rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
	if (peer->is_local && !(rts_hdr->flags & RXR_SHM_HDR_DATA)) {
		rxr_cq_process_shm_large_message(ep, rx_entry, rts_hdr, data);
		rxr_release_rx_pkt_entry(ep, pkt_entry);
		return 0;
	}

	data_size = rxr_get_rts_data_size(ep, rts_hdr);
	return rxr_cq_handle_rts_with_data(ep, rx_entry,
					   pkt_entry, data,
					   data_size);
}

static int rxr_cq_process_rts(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (rts_hdr->flags & RXR_READ_REQ)
		return rxr_rma_proc_read_rts(ep, pkt_entry);

	if (rts_hdr->flags & RXR_WRITE)
		return rxr_rma_proc_write_rts(ep, pkt_entry);

	return rxr_cq_process_msg_rts(ep, pkt_entry);
}

static int rxr_cq_reorder_msg(struct rxr_ep *ep,
			      struct rxr_peer *peer,
			      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_pkt_entry *ooo_entry;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	/*
	 * TODO: Initialize peer state  at the time of AV insertion
	 * where duplicate detection is available.
	 */
	if (!peer->rx_init)
		rxr_ep_peer_init(ep, peer);

#if ENABLE_DEBUG
	if (rts_hdr->msg_id != ofi_recvwin_next_exp_id(peer->robuf))
		FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
		       "msg OOO rts_hdr->msg_id: %" PRIu32 " expected: %"
		       PRIu32 "\n", rts_hdr->msg_id,
		       ofi_recvwin_next_exp_id(peer->robuf));
#endif
	if (ofi_recvwin_is_exp(peer->robuf, rts_hdr->msg_id))
		return 0;
	else if (!ofi_recvwin_id_valid(peer->robuf, rts_hdr->msg_id))
		return -FI_EALREADY;

	if (OFI_LIKELY(rxr_env.rx_copy_ooo)) {
		assert(pkt_entry->type == RXR_PKT_ENTRY_POSTED);
		ooo_entry = rxr_get_pkt_entry(ep, ep->rx_ooo_pkt_pool);
		if (OFI_UNLIKELY(!ooo_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unable to allocate rx_pkt_entry for OOO msg\n");
			return -FI_ENOMEM;
		}
		rxr_copy_pkt_entry(ep, ooo_entry, pkt_entry, RXR_PKT_ENTRY_OOO);
		rts_hdr = rxr_get_rts_hdr(ooo_entry->pkt);
		rxr_release_rx_pkt_entry(ep, pkt_entry);
	} else {
		ooo_entry = pkt_entry;
	}

	ofi_recvwin_queue_msg(peer->robuf, &ooo_entry, rts_hdr->msg_id);
	return 1;
}

static void rxr_cq_proc_pending_items_in_recvwin(struct rxr_ep *ep,
						 struct rxr_peer *peer)
{
	struct rxr_pkt_entry *pending_pkt;
	struct rxr_rts_hdr *rts_hdr;
	int ret = 0;

	while (1) {
		pending_pkt = *ofi_recvwin_peek(peer->robuf);
		if (!pending_pkt || !pending_pkt->pkt)
			return;

		rts_hdr = rxr_get_rts_hdr(pending_pkt->pkt);
		*ofi_recvwin_get_next_msg(peer->robuf) = NULL;

		FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
		       "Processing msg_id %d from robuf\n", rts_hdr->msg_id);

		/* rxr_cq_process_rts will write error cq entry if needed */
		ret = rxr_cq_process_rts(ep, pending_pkt);
		if (OFI_UNLIKELY(ret)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"Error processing msg_id %d from robuf: %s\n",
				rts_hdr->msg_id, fi_strerror(-ret));
			return;
		}
	}
	return;
}

/* Handle RMA writes with immediate data at remote endpoint, write a completion */
void rxr_cq_handle_shm_rma_write_data(struct rxr_ep *ep, struct fi_cq_data_entry *shm_comp, fi_addr_t src_addr)
{
	struct rxr_rx_entry *rx_entry;
	int ret;

	struct util_cq *rx_cq = ep->util_ep.rx_cq;
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, 0, 0, NULL, src_addr, 0, 0);
	rxr_copy_shm_cq_entry(&rx_entry->cq_entry, shm_comp);

	if (ep->util_ep.caps & FI_SOURCE)
		ret = ofi_cq_write_src(rx_cq,
				       rx_entry->cq_entry.op_context,
				       rx_entry->cq_entry.flags,
				       rx_entry->cq_entry.len,
				       rx_entry->cq_entry.buf,
				       rx_entry->cq_entry.data,
				       rx_entry->cq_entry.tag,
				       src_addr);
	else
		ret = ofi_cq_write(rx_cq,
				       rx_entry->cq_entry.op_context,
				       rx_entry->cq_entry.flags,
				       rx_entry->cq_entry.len,
				       rx_entry->cq_entry.buf,
				       rx_entry->cq_entry.data,
				       rx_entry->cq_entry.tag);

	rxr_rm_rx_cq_check(ep, rx_cq);

	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Unable to deliver immediate data of shm provider's RMA write operation: %s\n",
			fi_strerror(-ret));
		if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
			assert(0 && "failed to write err cq entry");
	}
	rxr_cntr_report_rx_completion(ep, rx_entry);
	rxr_release_rx_entry(ep, rx_entry);
}

/*
 * Sender handles the acknowledgment (RXR_EOR_PKT) from receiver on the completion
 * of the large message copy via fi_readmsg operation
 */
static void rxr_cq_handle_eor(struct rxr_ep *ep,
				  struct fi_cq_data_entry *comp,
				  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *shm_eor;
	struct rxr_tx_entry *tx_entry;

	shm_eor = (struct rxr_eor_hdr *)pkt_entry->pkt;

	/* pre-post buf used here, so can NOT track back to tx_entry with x_entry */
	tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, shm_eor->tx_id);
	rxr_cq_write_tx_completion(ep, tx_entry);
	rxr_release_rx_pkt_entry(ep, pkt_entry);
}

static void rxr_cq_handle_rts(struct rxr_ep *ep,
			      struct fi_cq_data_entry *comp,
			      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_peer *peer;
	int ret;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	if (rxr_env.enable_shm_transfer && peer->is_local) {
		/* no need to reorder msg for shm_ep
		 * rxr_cq_process_rts will write error cq entry if needed
		 */
		rxr_cq_process_rts(ep, pkt_entry);
		return;
	}

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (ep->core_caps & FI_SOURCE)
		rxr_cq_post_connack(ep, peer, pkt_entry->addr);

	if (rxr_need_sas_ordering(ep)) {
		ret = rxr_cq_reorder_msg(ep, peer, pkt_entry);
		if (ret == 1) {
			/* Packet was queued */
			return;
		} else if (OFI_UNLIKELY(ret == -FI_EALREADY)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Invalid msg_id: %" PRIu32
				" robuf->exp_msg_id: %" PRIu32 "\n",
			       rts_hdr->msg_id, peer->robuf->exp_msg_id);
			if (!rts_hdr->addrlen)
				rxr_eq_write_error(ep, FI_EIO, ret);
			rxr_release_rx_pkt_entry(ep, pkt_entry);
			return;
		} else if (OFI_UNLIKELY(ret == -FI_ENOMEM)) {
			rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
			return;
		} else if (OFI_UNLIKELY(ret < 0)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unknown error %d processing RTS packet msg_id: %"
				PRIu32 "\n", ret, rts_hdr->msg_id);
			rxr_eq_write_error(ep, FI_EIO, ret);
			return;
		}

		/* processing the expected packet */
		ofi_recvwin_slide(peer->robuf);
	}

	/* rxr_cq_process_rts will write error cq entry if needed */
	ret = rxr_cq_process_rts(ep, pkt_entry);
	if (OFI_UNLIKELY(ret))
		return;

	/* process pending items in reorder buff */
	if (rxr_need_sas_ordering(ep))
		rxr_cq_proc_pending_items_in_recvwin(ep, peer);
}

static void rxr_cq_handle_connack(struct rxr_ep *ep,
				  struct fi_cq_data_entry *comp,
				  struct rxr_pkt_entry *pkt_entry,
				  fi_addr_t src_addr)
{
	struct rxr_peer *peer;

	/*
	 * We don't really need any information from the actual connack packet
	 * itself, just the src_addr from the CQE
	 */
	assert(src_addr != FI_ADDR_NOTAVAIL);
	peer = rxr_ep_get_peer(ep, src_addr);
	peer->state = RXR_PEER_ACKED;
	FI_DBG(&rxr_prov, FI_LOG_CQ,
	       "CONNACK received from %" PRIu64 "\n", src_addr);
	rxr_release_rx_pkt_entry(ep, pkt_entry);
}

int rxr_cq_handle_rts_with_data(struct rxr_ep *ep,
				struct rxr_rx_entry *rx_entry,
				struct rxr_pkt_entry *pkt_entry,
				char *data, size_t data_size)
{
	struct rxr_rts_hdr *rts_hdr;
	int64_t bytes_left, bytes_copied;
	ssize_t ret;


	/* rx_entry->cq_entry.len is total recv buffer size.
	 * rx_entry->total_len is from rts_hdr and is total send buffer size.
	 * if send buffer size < recv buffer size, we adjust value of rx_entry->cq_entry.len.
	 * if send buffer size > recv buffer size, we have a truncated message.
	 */
	if (rx_entry->cq_entry.len > rx_entry->total_len)
		rx_entry->cq_entry.len = rx_entry->total_len;

	bytes_copied = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
				       0, data, data_size);

	if (OFI_UNLIKELY(bytes_copied < data_size)) {
		/* recv buffer is not big enough to hold rts, this must be a truncated message */
		assert(bytes_copied == rx_entry->cq_entry.len &&
		       rx_entry->cq_entry.len < rx_entry->total_len);
		rx_entry->bytes_done = bytes_copied;
		bytes_left = 0;
	} else {
		assert(bytes_copied == data_size);
		rx_entry->bytes_done = data_size;
		bytes_left = rx_entry->total_len - data_size;
	}

	assert(bytes_left >= 0);
	if (!bytes_left) {
		/* rxr_cq_handle_rx_completion() releases pkt_entry, thus
		 * we do not release it here.
		 */
		rxr_cq_handle_rx_completion(ep, pkt_entry, rx_entry);
		rxr_multi_recv_free_posted_entry(ep, rx_entry);
		rxr_release_rx_entry(ep, rx_entry);
		return 0;
	}

#if ENABLE_DEBUG
	dlist_insert_tail(&rx_entry->rx_pending_entry, &ep->rx_pending_list);
	ep->rx_pending++;
#endif
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rx_entry->state = RXR_RX_RECV;
	if (rts_hdr->flags & RXR_CREDIT_REQUEST)
		rx_entry->credit_request = rts_hdr->credit_request;
	else
		rx_entry->credit_request = rxr_env.tx_min_credits;
	ret = rxr_ep_post_ctrl_or_queue(ep, RXR_RX_ENTRY, rx_entry, RXR_CTS_PKT, 0);
	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return ret;
}

int rxr_cq_handle_pkt_with_data(struct rxr_ep *ep,
				struct rxr_rx_entry *rx_entry,
				struct rxr_pkt_entry *pkt_entry,
				char *data, size_t seg_offset,
				size_t seg_size)
{
	struct rxr_peer *peer;
	int64_t bytes_left, bytes_copied;
	ssize_t ret = 0;

#if ENABLE_DEBUG
	int pkt_type = rxr_get_base_hdr(pkt_entry->pkt)->type;
	assert(pkt_type == RXR_DATA_PKT || pkt_type == RXR_READRSP_PKT);
#endif
	/* we are sinking message for CANCEL/DISCARD entry */
	if (OFI_LIKELY(!(rx_entry->rxr_flags & RXR_RECV_CANCEL)) &&
	    rx_entry->cq_entry.len > seg_offset) {
		bytes_copied = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
					       seg_offset, data, seg_size);
		if (bytes_copied != MIN(seg_size, rx_entry->cq_entry.len - seg_offset)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ, "wrong size! bytes_copied: %ld\n",
				bytes_copied);
			if (rxr_cq_handle_rx_error(ep, rx_entry, -FI_EINVAL))
				assert(0 && "error writing error cq entry for EOR\n");
		}
	}

	rx_entry->bytes_done += seg_size;

	peer = rxr_ep_get_peer(ep, rx_entry->addr);
	peer->rx_credits += ofi_div_ceil(seg_size, ep->max_data_payload_size);

	rx_entry->window -= seg_size;
	if (ep->available_data_bufs < rxr_get_rx_pool_chunk_cnt(ep))
		ep->available_data_bufs++;

	/* bytes_done is total bytes sent/received, which could be larger than
	 * to bytes copied to recv buffer (for truncated messages).
	 * rx_entry->total_len is from rts_hdr and is the size of send buffer,
	 * thus we always have:
	 *             rx_entry->total >= rx_entry->bytes_done
	 */
	bytes_left = rx_entry->total_len - rx_entry->bytes_done;
	assert(bytes_left >= 0);
	if (!bytes_left) {
#if ENABLE_DEBUG
		dlist_remove(&rx_entry->rx_pending_entry);
		ep->rx_pending--;
#endif
		rxr_cq_handle_rx_completion(ep, pkt_entry, rx_entry);

		rxr_multi_recv_free_posted_entry(ep, rx_entry);
		rxr_release_rx_entry(ep, rx_entry);
		return 0;
	}

	if (!rx_entry->window) {
		assert(rx_entry->state == RXR_RX_RECV);
		ret = rxr_ep_post_ctrl_or_queue(ep, RXR_RX_ENTRY, rx_entry, RXR_CTS_PKT, 0);
	}

	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return ret;
}

static void rxr_cq_handle_readrsp(struct rxr_ep *ep,
				  struct fi_cq_data_entry *comp,
				  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_pkt *readrsp_pkt = NULL;
	struct rxr_readrsp_hdr *readrsp_hdr = NULL;
	struct rxr_rx_entry *rx_entry = NULL;

	readrsp_pkt = (struct rxr_readrsp_pkt *)pkt_entry->pkt;
	readrsp_hdr = &readrsp_pkt->hdr;
	rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool, readrsp_hdr->rx_id);
	assert(rx_entry->cq_entry.flags & FI_READ);
	rx_entry->tx_id = readrsp_hdr->tx_id;
	rxr_cq_handle_pkt_with_data(ep, rx_entry, pkt_entry,
				    readrsp_pkt->data,
				    0, readrsp_hdr->seg_size);
}

static void rxr_cq_handle_cts(struct rxr_ep *ep,
			      struct fi_cq_data_entry *comp,
			      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_cts_hdr *cts_pkt;
	struct rxr_tx_entry *tx_entry;

	cts_pkt = (struct rxr_cts_hdr *)pkt_entry->pkt;
	if (cts_pkt->flags & RXR_READ_REQ)
		tx_entry = ofi_bufpool_get_ibuf(ep->readrsp_tx_entry_pool, cts_pkt->tx_id);
	else
		tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, cts_pkt->tx_id);

	tx_entry->rx_id = cts_pkt->rx_id;
	tx_entry->window = cts_pkt->window;

	/* Return any excess tx_credits that were borrowed for the request */
	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	tx_entry->credit_allocated = ofi_div_ceil(cts_pkt->window, ep->max_data_payload_size);
	if (tx_entry->credit_allocated < tx_entry->credit_request)
		peer->tx_credits += tx_entry->credit_request - tx_entry->credit_allocated;

	rxr_release_rx_pkt_entry(ep, pkt_entry);

	if (tx_entry->state != RXR_TX_SEND) {
		tx_entry->state = RXR_TX_SEND;
		dlist_insert_tail(&tx_entry->entry, &ep->tx_pending_list);
	}
	return;
}

static void rxr_cq_handle_data(struct rxr_ep *ep,
			       struct fi_cq_data_entry *comp,
			       struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_data_pkt *data_pkt;
	struct rxr_rx_entry *rx_entry;
	data_pkt = (struct rxr_data_pkt *)pkt_entry->pkt;

	rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool,
					 data_pkt->hdr.rx_id);

	rxr_cq_handle_pkt_with_data(ep, rx_entry,
				    pkt_entry,
				    data_pkt->data,
				    data_pkt->hdr.seg_offset,
				    data_pkt->hdr.seg_size);
}

void rxr_cq_write_tx_completion(struct rxr_ep *ep,
				struct rxr_tx_entry *tx_entry)
{
	struct util_cq *tx_cq = ep->util_ep.tx_cq;
	int ret;

	if (!(tx_entry->fi_flags & RXR_NO_COMPLETION) &&
	    ofi_need_completion(rxr_tx_flags(ep), tx_entry->fi_flags)) {
		FI_DBG(&rxr_prov, FI_LOG_CQ,
		       "Writing send completion for tx_entry to peer: %" PRIu64
		       " tx_id: %" PRIu32 " msg_id: %" PRIu32 " tag: %lx len: %"
		       PRIu64 "\n",
		       tx_entry->addr, tx_entry->tx_id, tx_entry->msg_id,
		       tx_entry->cq_entry.tag, tx_entry->total_len);

		/* TX completions should not send peer address to util_cq */
		if (ep->util_ep.caps & FI_SOURCE)
			ret = ofi_cq_write_src(tx_cq,
					       tx_entry->cq_entry.op_context,
					       tx_entry->cq_entry.flags,
					       tx_entry->cq_entry.len,
					       tx_entry->cq_entry.buf,
					       tx_entry->cq_entry.data,
					       tx_entry->cq_entry.tag,
					       FI_ADDR_NOTAVAIL);
		else
			ret = ofi_cq_write(tx_cq,
					   tx_entry->cq_entry.op_context,
					   tx_entry->cq_entry.flags,
					   tx_entry->cq_entry.len,
					   tx_entry->cq_entry.buf,
					   tx_entry->cq_entry.data,
					   tx_entry->cq_entry.tag);

		rxr_rm_tx_cq_check(ep, tx_cq);

		if (OFI_UNLIKELY(ret)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"Unable to write send completion: %s\n",
				fi_strerror(-ret));
			if (rxr_cq_handle_tx_error(ep, tx_entry, ret))
				assert(0 && "failed to write err cq entry");
			return;
		}
	}

	rxr_cntr_report_tx_completion(ep, tx_entry);
	rxr_release_tx_entry(ep, tx_entry);
	return;
}

fi_addr_t rxr_cq_insert_addr_from_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	int i, ret;
	void *raw_address;
	fi_addr_t rdm_addr;
	struct rxr_rts_hdr *rts_hdr;
	struct efa_ep *efa_ep;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->type == RXR_RTS_PKT);

	efa_ep = container_of(ep->rdm_ep, struct efa_ep, ep_fid);
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	assert(rts_hdr->flags & RXR_REMOTE_SRC_ADDR);
	assert(rts_hdr->addrlen > 0);
	if (rxr_get_base_hdr(pkt_entry->pkt)->version !=
	    RXR_PROTOCOL_VERSION) {
		char buffer[ep->core_addrlen * 3];
		int length = 0;

		for (i = 0; i < ep->core_addrlen; i++)
			length += sprintf(&buffer[length], "%02x ",
					  ep->core_addr[i]);
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Host %s:Invalid protocol version %d. Expected protocol version %d.\n",
			buffer,
			rxr_get_base_hdr(pkt_entry->pkt)->version,
			RXR_PROTOCOL_VERSION);
		rxr_eq_write_error(ep, FI_EIO, -FI_EINVAL);
		fprintf(stderr, "Invalid protocol version %d. Expected protocol version %d. %s:%d\n",
			rxr_get_base_hdr(pkt_entry->pkt)->version,
			RXR_PROTOCOL_VERSION, __FILE__, __LINE__);
		abort();
	}

	raw_address = (rts_hdr->flags & RXR_REMOTE_CQ_DATA) ?
		      rxr_get_ctrl_cq_pkt(rts_hdr)->data
		      : rxr_get_ctrl_pkt(rts_hdr)->data;

	ret = efa_av_insert_addr(efa_ep->av, (struct efa_ep_addr *)raw_address,
				&rdm_addr, 0, NULL);
	if (OFI_UNLIKELY(ret != 0)) {
		rxr_eq_write_error(ep, FI_EINVAL, ret);
		return -1;
	}

	return rdm_addr;
}

void rxr_cq_handle_pkt_recv_completion(struct rxr_ep *ep,
				       struct fi_cq_data_entry *cq_entry,
				       fi_addr_t src_addr)
{
	struct rxr_peer *peer;
	struct rxr_pkt_entry *pkt_entry;

	pkt_entry = (struct rxr_pkt_entry *)cq_entry->op_context;

#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
	dlist_insert_tail(&pkt_entry->dbg_entry, &ep->rx_pkt_list);
#ifdef ENABLE_RXR_PKT_DUMP
	rxr_ep_print_pkt("Received", ep, (struct rxr_base_hdr *)pkt_entry->pkt);
#endif
#endif
	if (OFI_UNLIKELY(src_addr == FI_ADDR_NOTAVAIL))
		pkt_entry->addr = rxr_cq_insert_addr_from_rts(ep, pkt_entry);
	else
		pkt_entry->addr = src_addr;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->version ==
	       RXR_PROTOCOL_VERSION);

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);

	if (rxr_env.enable_shm_transfer && peer->is_local)
		ep->posted_bufs_shm--;
	else
		ep->posted_bufs_efa--;

	switch (rxr_get_base_hdr(pkt_entry->pkt)->type) {
	case RXR_RTS_PKT:
		rxr_cq_handle_rts(ep, cq_entry, pkt_entry);
		return;
	case RXR_EOR_PKT:
		rxr_cq_handle_eor(ep, cq_entry, pkt_entry);
		return;
	case RXR_CONNACK_PKT:
		rxr_cq_handle_connack(ep, cq_entry, pkt_entry, src_addr);
		return;
	case RXR_CTS_PKT:
		rxr_cq_handle_cts(ep, cq_entry, pkt_entry);
		return;
	case RXR_DATA_PKT:
		rxr_cq_handle_data(ep, cq_entry, pkt_entry);
		return;
	case RXR_READRSP_PKT:
		rxr_cq_handle_readrsp(ep, cq_entry, pkt_entry);
		return;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"invalid control pkt type %d\n",
			rxr_get_base_hdr(pkt_entry->pkt)->type);
		assert(0 && "invalid control pkt type");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
		return;
	}
	return;
}

static int rxr_send_completion_mr_dereg(struct rxr_tx_entry *tx_entry)
{
	int i, ret = 0;

	for (i = tx_entry->iov_mr_start; i < tx_entry->iov_count; i++) {
		if (tx_entry->mr[i]) {
			ret = fi_close((struct fid *)tx_entry->mr[i]);
			if (OFI_UNLIKELY(ret))
				return ret;
		}
	}
	return ret;
}

void rxr_cq_handle_rma_context_pkt(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry = NULL;
	struct rxr_rx_entry *rx_entry = NULL;
	struct rxr_rma_context_pkt *rma_context_pkt;
	int ret;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->version == RXR_PROTOCOL_VERSION);

	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->pkt;

	switch (rma_context_pkt->rma_context_type) {
	case RXR_SHM_RMA_READ:
	case RXR_SHM_RMA_WRITE:
		/* Completion of RMA READ/WRTITE operations that apps call */
		tx_entry = pkt_entry->x_entry;

		if (tx_entry->fi_flags & FI_COMPLETION) {
			rxr_cq_write_tx_completion(ep, tx_entry);
		} else {
			rxr_cntr_report_tx_completion(ep, tx_entry);
			rxr_release_tx_entry(ep, tx_entry);
		}
		rxr_release_tx_pkt_entry(ep, pkt_entry);
		break;
	case RXR_SHM_LARGE_READ:
		/*
		 * This must be on the receiver (remote) side of two-sided message
		 * transfer, which is also the initiator of RMA READ.
		 * We get RMA READ completion for previously issued
		 * fi_read operation over shm provider, which means
		 * receiver side has received all data from sender
		 */
		rx_entry = pkt_entry->x_entry;
		rx_entry->cq_entry.len = rx_entry->total_len;
		rx_entry->bytes_done = rx_entry->total_len;

		ret = rxr_ep_post_ctrl_or_queue(ep, RXR_TX_ENTRY, rx_entry, RXR_EOR_PKT, 1);
		if (ret) {
			if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
				assert(0 && "error writing error cq entry for EOR\n");
		}

		if (rx_entry->fi_flags & FI_MULTI_RECV)
			rxr_cq_handle_multi_recv_completion(ep, rx_entry);
		rxr_cq_write_rx_completion(ep, rx_entry);
		rxr_multi_recv_free_posted_entry(ep, rx_entry);
		if (OFI_LIKELY(!ret))
			rxr_release_rx_entry(ep, rx_entry);
		rxr_release_rx_pkt_entry(ep, pkt_entry);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "invalid rma_context_type in RXR_RMA_CONTEXT_PKT %d\n",
			rma_context_pkt->rma_context_type);
		assert(0 && "invalid RXR_RMA_CONTEXT_PKT rma_context_type\n");
	}
}

void rxr_cq_handle_pkt_send_completion(struct rxr_ep *ep, struct fi_cq_data_entry *comp)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_tx_entry *tx_entry = NULL;
	struct rxr_peer *peer;
	struct rxr_rts_hdr *rts_hdr = NULL;
	struct rxr_readrsp_hdr *readrsp_hdr = NULL;
	uint32_t tx_id;
	int ret;

	pkt_entry = (struct rxr_pkt_entry *)comp->op_context;
	assert(rxr_get_base_hdr(pkt_entry->pkt)->version ==
	       RXR_PROTOCOL_VERSION);
	peer = rxr_ep_get_peer(ep, pkt_entry->addr);

	switch (rxr_get_base_hdr(pkt_entry->pkt)->type) {
	case RXR_RTS_PKT:
		/*
		 * for FI_READ, it is possible (though does not happen very offen) that at the point
		 * tx_entry has been released. The reason is, for FI_READ:
		 *     1. only the initator side will send a RTS.
		 *     2. the initator side will receive data packet. When all data was received,
		 *        it will release the tx_entry
		 * Therefore, if it so happens that all data was received before we got the send
		 * completion notice, we will have a released tx_entry at this point.
		 * Nonetheless, because for FI_READ tx_entry will be release in rxr_handle_rx_completion,
		 * we will ignore it here.
		 *
		 * For shm provider, we will write completion for small & medium  message, as data has
		 * been sent in the RTS packet; for large message, will wait for the EOR packet
		 */
		rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
		if (!(rts_hdr->flags & RXR_READ_REQ)) {
			tx_id = rts_hdr->tx_id;
			tx_entry = ofi_bufpool_get_ibuf(ep->tx_entry_pool, tx_id);
			tx_entry->bytes_acked += rxr_get_rts_data_size(ep, rts_hdr);
		}
		break;
	case RXR_CONNACK_PKT:
		break;
	case RXR_CTS_PKT:
		break;
	case RXR_DATA_PKT:
		tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
		tx_entry->bytes_acked +=
			rxr_get_data_pkt(pkt_entry->pkt)->hdr.seg_size;
		break;
	case RXR_READRSP_PKT:
		readrsp_hdr = rxr_get_readrsp_hdr(pkt_entry->pkt);
		tx_id = readrsp_hdr->tx_id;
		tx_entry = ofi_bufpool_get_ibuf(ep->readrsp_tx_entry_pool, tx_id);
		assert(tx_entry->cq_entry.flags & FI_READ);
		tx_entry->bytes_acked += readrsp_hdr->seg_size;
		break;
	case RXR_RMA_CONTEXT_PKT:
		rxr_cq_handle_rma_context_pkt(ep, pkt_entry);
		return;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"invalid control pkt type %d\n",
			rxr_get_base_hdr(pkt_entry->pkt)->type);
		assert(0 && "invalid control pkt type");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
		return;
	}

	if (tx_entry && tx_entry->total_len == tx_entry->bytes_acked) {
		if (tx_entry->state == RXR_TX_SEND)
			dlist_remove(&tx_entry->entry);
		if (tx_entry->state == RXR_TX_SEND &&
		    efa_mr_cache_enable && rxr_ep_mr_local(ep)) {
			ret = rxr_send_completion_mr_dereg(tx_entry);
			if (OFI_UNLIKELY(ret)) {
				FI_WARN(&rxr_prov, FI_LOG_MR,
					"In-line memory deregistration failed with error: %s.\n",
					fi_strerror(-ret));
			}
		}

		peer->tx_credits += tx_entry->credit_allocated;

		if (tx_entry->cq_entry.flags & FI_READ) {
			/*
			 * this must be on remote side
			 * see explaination on rxr_cq_handle_rx_completion
			 */
			struct rxr_rx_entry *rx_entry = NULL;

			rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool, tx_entry->rma_loc_rx_id);
			assert(rx_entry);
			assert(rx_entry->state == RXR_RX_WAIT_READ_FINISH);

			if (ep->util_ep.caps & FI_RMA_EVENT) {
				rx_entry->cq_entry.len = rx_entry->total_len;
				rx_entry->bytes_done = rx_entry->total_len;
				rxr_cntr_report_rx_completion(ep, rx_entry);
			}

			rxr_release_rx_entry(ep, rx_entry);
			/* just release tx, do not write completion */
			rxr_release_tx_entry(ep, tx_entry);
		} else if (tx_entry->cq_entry.flags & FI_WRITE) {
			if (tx_entry->fi_flags & FI_COMPLETION) {
				rxr_cq_write_tx_completion(ep, tx_entry);
			} else {
				rxr_cntr_report_tx_completion(ep, tx_entry);
				rxr_release_tx_entry(ep, tx_entry);
			}
		} else {
			assert(tx_entry->cq_entry.flags & FI_SEND);
			rxr_cq_write_tx_completion(ep, tx_entry);
		}
	}

	rxr_release_tx_pkt_entry(ep, pkt_entry);
	if (!peer->is_local)
		rxr_ep_dec_tx_pending(ep, peer, 0);
	return;
}

static int rxr_cq_close(struct fid *fid)
{
	int ret;
	struct util_cq *cq;

	cq = container_of(fid, struct util_cq, cq_fid.fid);
	ret = ofi_cq_cleanup(cq);
	if (ret)
		return ret;
	free(cq);
	return 0;
}

static struct fi_ops rxr_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxr_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq rxr_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = fi_no_cq_sread,
	.sreadfrom = fi_no_cq_sreadfrom,
	.signal = fi_no_cq_signal,
	.strerror = rxr_cq_strerror,
};

int rxr_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct util_cq *cq;
	struct rxr_domain *rxr_domain;

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	rxr_domain = container_of(domain, struct rxr_domain,
				  util_domain.domain_fid);
	/* Override user cq size if it's less than recommended cq size */
	attr->size = MAX(rxr_domain->cq_size, attr->size);

	ret = ofi_cq_init(&rxr_prov, domain, attr, cq,
			  &ofi_cq_progress, context);

	if (ret)
		goto free;

	*cq_fid = &cq->cq_fid;
	(*cq_fid)->fid.ops = &rxr_cq_fi_ops;
	(*cq_fid)->ops = &rxr_cq_ops;
	return 0;
free:
	free(cq);
	return ret;
}
