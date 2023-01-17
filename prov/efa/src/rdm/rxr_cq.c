/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
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
#include "efa.h"
#include "rxr_rma.h"
#include "rxr_msg.h"
#include "rxr_cntr.h"
#include "rxr_read.h"
#include "rxr_atomic.h"
#include "rxr_pkt_cmd.h"

#include "rxr_tp.h"

static const char *rxr_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
				   const void *err_data, char *buf, size_t len)
{
	return efa_strerror(prov_errno);
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
void rxr_cq_queue_rnr_pkt(struct rxr_ep *ep,
			  struct dlist_entry *list,
			  struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_peer *peer;

#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
	dlist_insert_tail(&pkt_entry->entry, list);

	/*
	 * When the EFA RDM provider wants to send multiple packets,
	 * it connects the packets in a linked list through ibv_send_wr.next
	 * and calls ibv_post_send once on the head of the linked list.
	 *
	 * When a packet encounters RNR, we want to queue only the packet that
	 * encountered RNR. So we remove the other packets in the linked list
	 * by setting ibv_send_wr.next to NULL
	 */
	pkt_entry->send_wr.wr.next = NULL;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
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
		if (rxr_env.rnr_backoff_initial_wait_time > 0)
			peer->rnr_backoff_wait_time = rxr_env.rnr_backoff_initial_wait_time;
		else
			peer->rnr_backoff_wait_time = MAX(RXR_RAND_MIN_TIMEOUT,
							  rand() %
							  RXR_RAND_MAX_TIMEOUT);

		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "initializing backoff timeout for peer: %" PRIu64
		       " timeout: %ld rnr_queued_pkts: %d\n",
		       pkt_entry->addr, peer->rnr_backoff_wait_time,
		       peer->rnr_queued_pkt_cnt);
	} else {
		peer->rnr_backoff_wait_time = MIN(peer->rnr_backoff_wait_time * 2,
						  rxr_env.rnr_backoff_wait_time_cap);
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "increasing backoff timeout for peer: %" PRIu64
		       "to %ld rnr_queued_pkts: %d\n",
		       pkt_entry->addr, peer->rnr_backoff_wait_time,
		       peer->rnr_queued_pkt_cnt);
	}
}





/* Handle two scenarios:
 *  1. RMA writes with immediate data at remote endpoint,
 *  2. atomic completion on the requester
 * write completion for both
 */
void rxr_cq_handle_shm_completion(struct rxr_ep *ep, struct fi_cq_data_entry *cq_entry, fi_addr_t src_addr)
{
	struct util_cq *target_cq;
	int ret;

	if (cq_entry->flags & FI_ATOMIC) {
		target_cq = ep->base_ep.util_ep.tx_cq;
	} else {
		assert(cq_entry->flags & FI_REMOTE_CQ_DATA);
		target_cq = ep->base_ep.util_ep.rx_cq;
	}

	if (ep->base_ep.util_ep.caps & FI_SOURCE)
		ret = ofi_cq_write_src(target_cq,
				       cq_entry->op_context,
				       cq_entry->flags,
				       cq_entry->len,
				       cq_entry->buf,
				       cq_entry->data,
				       0,
				       src_addr);
	else
		ret = ofi_cq_write(target_cq,
				   cq_entry->op_context,
				   cq_entry->flags,
				   cq_entry->len,
				   cq_entry->buf,
				   cq_entry->data,
				   0);

	rxr_rm_rx_cq_check(ep, target_cq);

	if (OFI_UNLIKELY(ret)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Unable to write a cq entry for shm operation: %s\n",
			fi_strerror(-ret));
		efa_eq_write_error(&ep->base_ep.util_ep, FI_EIO, FI_EFA_ERR_WRITE_SHM_CQ_ENTRY);
	}

	if (cq_entry->flags & FI_ATOMIC) {
		efa_cntr_report_tx_completion(&ep->base_ep.util_ep, cq_entry->flags);
	} else {
		assert(cq_entry->flags & FI_REMOTE_CQ_DATA);
		efa_cntr_report_rx_completion(&ep->base_ep.util_ep, cq_entry->flags);
	}
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
	struct efa_domain *efa_domain;

	if (attr->wait_obj != FI_WAIT_NONE)
		return -FI_ENOSYS;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	efa_domain = container_of(domain, struct efa_domain,
				  util_domain.domain_fid);
	/* Override user cq size if it's less than recommended cq size */
	attr->size = MAX(efa_domain->rdm_cq_size, attr->size);

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
