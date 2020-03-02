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

#include "efa.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_rma.h"
#include "rxr_pkt_cmd.h"

/* This file contains RTS packet related functions.
 * RTS is used to post send, emulated read and emulated
 * write requests.
 */

/*
 *  Utility struct and functions
 */
struct rxr_pkt_rts_read_hdr {
	uint64_t rma_initiator_rx_id;
	uint64_t window;
};

/*
 *   Helper function to compute the maximum payload of the RTS header based on
 *   the RTS header flags. The header' data_len field may have a length greater
 *   than the possible RTS payload size if it is a large message.
 */
uint64_t rxr_get_rts_data_size(struct rxr_ep *ep,
			       struct rxr_rts_hdr *rts_hdr)
{
	size_t max_payload_size;

	/*
	 * read RTS contain no data, because data is on remote EP.
	 */
	if (rts_hdr->flags & RXR_READ_REQ)
		return 0;

	if (rts_hdr->flags & RXR_SHM_HDR)
		return (rts_hdr->flags & RXR_SHM_HDR_DATA) ? rts_hdr->data_len : 0;


	if (rts_hdr->flags & RXR_REMOTE_CQ_DATA)
		max_payload_size = ep->mtu_size - RXR_CTRL_HDR_SIZE;
	else
		max_payload_size = ep->mtu_size - RXR_CTRL_HDR_SIZE_NO_CQ;

	if (rts_hdr->flags & RXR_REMOTE_SRC_ADDR)
		max_payload_size -= rts_hdr->addrlen;

	if (rts_hdr->flags & RXR_WRITE)
		max_payload_size -= rts_hdr->rma_iov_count *
					sizeof(struct fi_rma_iov);

	return (rts_hdr->data_len > max_payload_size)
		? max_payload_size : rts_hdr->data_len;
}

/*
 *  rxr_pkt_init_rts() and related functions.
 */
static
char *rxr_pkt_init_rts_base_hdr(struct rxr_ep *ep,
				struct rxr_tx_entry *tx_entry,
				struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_peer *peer;
	char *src;

	rts_hdr = (struct rxr_rts_hdr *)pkt_entry->pkt;
	peer = rxr_ep_get_peer(ep, tx_entry->addr);

	rts_hdr->type = RXR_RTS_PKT;
	rts_hdr->version = RXR_PROTOCOL_VERSION;
	rts_hdr->tag = tx_entry->tag;

	rts_hdr->data_len = tx_entry->total_len;
	rts_hdr->tx_id = tx_entry->tx_id;
	rts_hdr->msg_id = tx_entry->msg_id;
	/*
	 * Even with protocol versions prior to v3 that did not include a
	 * request in the RTS, the receiver can test for this flag and decide if
	 * it should be used as a heuristic for credit calculation. If the
	 * receiver is on <3 protocol version, the flag and the request just get
	 * ignored.
	 */
	rts_hdr->flags |= RXR_CREDIT_REQUEST;
	rts_hdr->credit_request = tx_entry->credit_request;

	if (tx_entry->fi_flags & FI_REMOTE_CQ_DATA) {
		rts_hdr->flags = RXR_REMOTE_CQ_DATA;
		pkt_entry->pkt_size = RXR_CTRL_HDR_SIZE;
		rxr_get_ctrl_cq_pkt(rts_hdr)->hdr.cq_data =
			tx_entry->cq_entry.data;
		src = rxr_get_ctrl_cq_pkt(rts_hdr)->data;
	} else {
		rts_hdr->flags = 0;
		pkt_entry->pkt_size = RXR_CTRL_HDR_SIZE_NO_CQ;
		src = rxr_get_ctrl_pkt(rts_hdr)->data;
	}

	if (tx_entry->cq_entry.flags & FI_TAGGED)
		rts_hdr->flags |= RXR_TAGGED;

	rts_hdr->addrlen = 0;
	if (OFI_UNLIKELY(peer->state != RXR_PEER_ACKED)) {
		/*
		 * This is the first communication with this peer on this
		 * endpoint, so send the core's address for this EP in the RTS
		 * so the remote side can insert it into its address vector.
		 */
		rts_hdr->addrlen = ep->core_addrlen;
		rts_hdr->flags |= RXR_REMOTE_SRC_ADDR;
		memcpy(src, ep->core_addr, rts_hdr->addrlen);
		src += rts_hdr->addrlen;
		pkt_entry->pkt_size += rts_hdr->addrlen;
	}

	return src;
}

static
char *rxr_pkt_init_rts_rma_hdr(struct rxr_ep *ep,
			       struct rxr_tx_entry *tx_entry,
			       struct rxr_pkt_entry *pkt_entry,
			       char *hdr)
{
	int rmalen;
	struct rxr_rts_hdr *rts_hdr;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rts_hdr->rma_iov_count = 0;
	assert(tx_entry->cq_entry.flags & FI_RMA);
	if (tx_entry->op == ofi_op_write) {
		rts_hdr->flags |= RXR_WRITE;
	} else {
		assert(tx_entry->op == ofi_op_read_req);
		rts_hdr->flags |= RXR_READ_REQ;
	}

	rmalen = tx_entry->rma_iov_count * sizeof(struct fi_rma_iov);
	rts_hdr->rma_iov_count = tx_entry->rma_iov_count;
	memcpy(hdr, tx_entry->rma_iov, rmalen);
	hdr += rmalen;
	pkt_entry->pkt_size += rmalen;

	return hdr;
}

static
int rxr_pkt_init_read_rts(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
			  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_pkt_rts_read_hdr *read_hdr;
	char *hdr;

	hdr = rxr_pkt_init_rts_base_hdr(ep, tx_entry, pkt_entry);
	hdr = rxr_pkt_init_rts_rma_hdr(ep, tx_entry, pkt_entry, hdr);

	/* no data to send, but need to send rx_id and window */
	read_hdr = (struct rxr_pkt_rts_read_hdr *)hdr;
	read_hdr->rma_initiator_rx_id = tx_entry->rma_loc_rx_id;
	read_hdr->window = tx_entry->rma_window;
	hdr += sizeof(struct rxr_pkt_rts_read_hdr);
	pkt_entry->pkt_size += sizeof(struct rxr_pkt_rts_read_hdr);

	assert(pkt_entry->pkt_size <= ep->mtu_size);
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = (void *)tx_entry;
	return 0;
}

ssize_t rxr_pkt_init_rts(struct rxr_ep *ep,
			 struct rxr_tx_entry *tx_entry,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_rts_hdr *rts_hdr;
	char *data, *src;
	uint64_t data_len;
	size_t mtu = ep->mtu_size;

	if (tx_entry->op == ofi_op_read_req)
		return rxr_pkt_init_read_rts(ep, tx_entry, pkt_entry);

	src = rxr_pkt_init_rts_base_hdr(ep, tx_entry, pkt_entry);
	if (tx_entry->op == ofi_op_write)
		src = rxr_pkt_init_rts_rma_hdr(ep, tx_entry, pkt_entry, src);

	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);
	data = src;
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	if (rxr_env.enable_shm_transfer && peer->is_local) {
		rts_hdr->flags |= RXR_SHM_HDR;
		/* will be sent over shm provider */
		if (tx_entry->total_len <= rxr_env.shm_max_medium_size) {
			data_len = ofi_copy_from_iov(data, rxr_env.shm_max_medium_size,
						     tx_entry->iov, tx_entry->iov_count, 0);
			assert(data_len == tx_entry->total_len);
			rts_hdr->flags |= RXR_SHM_HDR_DATA;
			pkt_entry->pkt_size += data_len;
		} else {
			/* rendezvous protocol
			 * place iov_count first, then local iov
			 */
			memcpy(data, &tx_entry->iov_count, sizeof(size_t));
			data += sizeof(size_t);
			pkt_entry->pkt_size += sizeof(size_t);
			memcpy(data, tx_entry->iov, sizeof(struct iovec) * tx_entry->iov_count);
			pkt_entry->pkt_size += sizeof(struct iovec) * tx_entry->iov_count;
		}
	} else {
		/* will be sent over efa provider */
		data_len = ofi_copy_from_iov(data, mtu - pkt_entry->pkt_size,
					     tx_entry->iov, tx_entry->iov_count, 0);
		assert(data_len == rxr_get_rts_data_size(ep, rts_hdr));
		pkt_entry->pkt_size += data_len;
	}

	assert(pkt_entry->pkt_size <= mtu);
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = (void *)tx_entry;
	return 0;
}

void rxr_pkt_handle_rts_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_tx_entry *tx_entry;
	size_t data_sent;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	if (tx_entry->op == ofi_op_read_req) {
		tx_entry->bytes_sent = 0;
		tx_entry->state = RXR_TX_WAIT_READ_FINISH;
		return;
	}

	data_sent = rxr_get_rts_data_size(ep, rxr_get_rts_hdr(pkt_entry->pkt));

	tx_entry->bytes_sent += data_sent;

	if ((rxr_env.enable_shm_transfer && peer->is_local) ||
	    !(efa_mr_cache_enable && tx_entry->total_len > data_sent))
		return;

	/*
	 * Register the data buffers inline only if the application did not
	 * provide a descriptor with the tx op
	 */
	if (rxr_ep_mr_local(ep) && !tx_entry->desc[0])
		rxr_inline_mr_reg(rxr_ep_domain(ep), tx_entry);
}

void rxr_pkt_handle_rts_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_tx_entry *tx_entry;

	/*
	 * for FI_READ, it is possible (though does not happen very often) that at the point
	 * tx_entry has been released. The reason is, for FI_READ:
	 *     1. only the initator side will send a RTS.
	 *     2. the initator side will receive data packet. When all data was received,
	 *        it will release the tx_entry
	 * Therefore, if it so happens that all data was received before we got the send
	 * completion notice, we will have a released tx_entry at this point.
	 * Nonetheless, because for FI_READ tx_entry will be release in rxr_handle_rx_completion,
	 * we will ignore it here.
	 */
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	if (rts_hdr->flags & RXR_READ_REQ)
		return;

	/*
	 * For shm provider, we will write completion for small & medium  message, as data has
	 * been sent in the RTS packet; for large message, will wait for the EOR packet
	 */
	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	tx_entry->bytes_acked += rxr_get_rts_data_size(ep, rts_hdr);
	if (tx_entry->total_len == tx_entry->bytes_acked)
		rxr_cq_handle_tx_completion(ep, tx_entry);
}

/*
 *  The following section are rxr_pkt_handle_rts_recv() and
 *  its related functions.
 */
static
char *rxr_pkt_proc_rts_base_hdr(struct rxr_ep *ep,
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

static
char *rxr_pkt_proc_rts_rma_hdr(struct rxr_ep *ep,
			       struct rxr_rx_entry *rx_entry,
			       struct rxr_pkt_entry *pkt_entry,
			       char *rma_hdr)
{
	uint32_t rma_access;
	struct fi_rma_iov *rma_iov = NULL;
	struct rxr_rts_hdr *rts_hdr;
	int ret;

	rma_iov = (struct fi_rma_iov *)rma_hdr;
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	if (rts_hdr->flags & RXR_READ_REQ) {
		rma_access = FI_SEND;
		rx_entry->cq_entry.flags |= (FI_RMA | FI_READ);
	} else {
		assert(rts_hdr->flags & RXR_WRITE);
		rma_access = FI_RECV;
		rx_entry->cq_entry.flags |= (FI_RMA | FI_WRITE);
	}

	assert(rx_entry->iov_count == 0);

	rx_entry->iov_count = rts_hdr->rma_iov_count;
	ret = rxr_rma_verified_copy_iov(ep, rma_iov, rts_hdr->rma_iov_count,
					rma_access, rx_entry->iov);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_CQ, "RMA address verify failed!\n");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
	}

	rx_entry->cq_entry.len = ofi_total_iov_len(&rx_entry->iov[0],
						   rx_entry->iov_count);
	rx_entry->cq_entry.buf = rx_entry->iov[0].iov_base;
	return rma_hdr + rts_hdr->rma_iov_count * sizeof(struct fi_rma_iov);
}

static
int rxr_cq_match_recv(struct dlist_entry *item, const void *arg)
{
	const struct rxr_pkt_entry *pkt_entry = arg;
	struct rxr_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);

	return rxr_match_addr(rx_entry->addr, pkt_entry->addr);
}

static
int rxr_cq_match_trecv(struct dlist_entry *item, const void *arg)
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

int rxr_pkt_proc_rts_data(struct rxr_ep *ep,
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
		rxr_msg_multi_recv_free_posted_entry(ep, rx_entry);
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
	ret = rxr_pkt_post_ctrl_or_queue(ep, RXR_RX_ENTRY, rx_entry, RXR_CTS_PKT, 0);
	rxr_pkt_entry_release_rx(ep, pkt_entry);
	return ret;
}

ssize_t rxr_pkt_post_shm_rndzv_read(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry)
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

	pkt_entry = rxr_pkt_entry_alloc(ep, ep->tx_pkt_shm_pool);
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

void rxr_pkt_proc_shm_long_msg_rts(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry,
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

	ret = rxr_pkt_post_shm_rndzv_read(ep, rx_entry);

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

ssize_t rxr_pkt_proc_matched_msg_rts(struct rxr_ep *ep,
				     struct rxr_rx_entry *rx_entry,
				     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	struct rxr_rts_hdr *rts_hdr;
	char *data;
	size_t data_size;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	data = rxr_pkt_proc_rts_base_hdr(ep, rx_entry, pkt_entry);
	if (peer->is_local && !(rts_hdr->flags & RXR_SHM_HDR_DATA)) {
		rxr_pkt_proc_shm_long_msg_rts(ep, rx_entry, rts_hdr, data);
		rxr_pkt_entry_release_rx(ep, pkt_entry);
		return 0;
	}

	data_size = rxr_get_rts_data_size(ep, rts_hdr);
	return rxr_pkt_proc_rts_data(ep, rx_entry,
				     pkt_entry, data,
				     data_size);
}

static
int rxr_pkt_proc_msg_rts(struct rxr_ep *ep,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct dlist_entry *match;
	struct rxr_rx_entry *rx_entry;

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
			efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
			return -FI_ENOBUFS;
		}

		/* we are not releasing pkt_entry here because it will be
		 * processed later
		 */
		pkt_entry = rx_entry->unexp_rts_pkt;
		rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
		rxr_pkt_proc_rts_base_hdr(ep, rx_entry, pkt_entry);
		return 0;
	}

	rx_entry = container_of(match, struct rxr_rx_entry, entry);
	if (rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED) {
		rx_entry = rxr_ep_split_rx_entry(ep, rx_entry,
						 NULL, pkt_entry);
		if (OFI_UNLIKELY(!rx_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
			return -FI_ENOBUFS;
		}
	}

	rx_entry->state = RXR_RX_MATCHED;

	if (!(rx_entry->fi_flags & FI_MULTI_RECV) ||
	    !rxr_msg_multi_recv_buffer_available(ep, rx_entry->master_entry))
		dlist_remove(match);

	return rxr_pkt_proc_matched_msg_rts(ep, rx_entry, pkt_entry);
}

static
int rxr_pkt_proc_write_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_rts_hdr *rts_hdr;
	uint64_t tag = ~0;
	char *rma_hdr;
	char *data;
	size_t data_size;

	/*
	 * rma is one sided operation, match is not expected
	 * we need to create a rx entry upon receiving a rts
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, tag, 0, NULL, pkt_entry->addr, ofi_op_write, 0);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted.\n");
		efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rma_hdr = rxr_pkt_proc_rts_base_hdr(ep, rx_entry, pkt_entry);
	data = rxr_pkt_proc_rts_rma_hdr(ep, rx_entry, pkt_entry, rma_hdr);
	data_size = rxr_get_rts_data_size(ep, rts_hdr);
	return rxr_pkt_proc_rts_data(ep, rx_entry,
				     pkt_entry, data,
				     data_size);
}

static
int rxr_pkt_proc_read_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	uint64_t tag = ~0;
	int err = 0;
	char *hdr;
	struct rxr_pkt_rts_read_hdr *read_info;
	/*
	 * rma is one sided operation, match is not expected
	 * we need to create a rx entry upon receiving a rts
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, tag, 0, NULL, pkt_entry->addr, ofi_op_read_rsp, 0);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted.\n");
		efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	hdr = (char *)rxr_pkt_proc_rts_base_hdr(ep, rx_entry, pkt_entry);
	hdr = (char *)rxr_pkt_proc_rts_rma_hdr(ep, rx_entry, pkt_entry, hdr);
	read_info = (struct rxr_pkt_rts_read_hdr *)hdr;

	rx_entry->rma_initiator_rx_id = read_info->rma_initiator_rx_id;
	rx_entry->window = read_info->window;
	assert(rx_entry->window > 0);

	tx_entry = rxr_rma_alloc_readrsp_tx_entry(ep, rx_entry);
	assert(tx_entry);
	/* the only difference between a read response packet and
	 * a data packet is that read response packet has remote EP tx_id
	 * which initiator EP rx_entry need to send CTS back
	 */
	err = rxr_pkt_post_ctrl_or_queue(ep, RXR_TX_ENTRY, tx_entry, RXR_READRSP_PKT, 0);
	if (OFI_UNLIKELY(err)) {
		if (rxr_cq_handle_tx_error(ep, tx_entry, err))
			assert(0 && "failed to write err cq entry");
		rxr_release_tx_entry(ep, tx_entry);
		rxr_release_rx_entry(ep, rx_entry);
	} else {
		rx_entry->state = RXR_RX_WAIT_READ_FINISH;
	}

	rxr_pkt_entry_release_rx(ep, pkt_entry);
	return err;
}

ssize_t rxr_pkt_proc_rts(struct rxr_ep *ep,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (rts_hdr->flags & RXR_READ_REQ)
		return rxr_pkt_proc_read_rts(ep, pkt_entry);

	if (rts_hdr->flags & RXR_WRITE)
		return rxr_pkt_proc_write_rts(ep, pkt_entry);

	return rxr_pkt_proc_msg_rts(ep, pkt_entry);
}

void rxr_pkt_handle_rts_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_peer *peer;
	int ret;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	if (rxr_env.enable_shm_transfer && peer->is_local) {
		/* no need to reorder msg for shm_ep
		 * rxr_pkt_proc_rts will write error cq entry if needed
		 */
		rxr_pkt_proc_rts(ep, pkt_entry);
		return;
	}

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (ep->core_caps & FI_SOURCE)
		rxr_pkt_post_connack(ep, peer, pkt_entry->addr);

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
				efa_eq_write_error(&ep->util_ep, FI_EIO, ret);
			rxr_pkt_entry_release_rx(ep, pkt_entry);
			return;
		} else if (OFI_UNLIKELY(ret == -FI_ENOMEM)) {
			efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
			return;
		} else if (OFI_UNLIKELY(ret < 0)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unknown error %d processing RTS packet msg_id: %"
				PRIu32 "\n", ret, rts_hdr->msg_id);
			efa_eq_write_error(&ep->util_ep, FI_EIO, ret);
			return;
		}

		/* processing the expected packet */
		ofi_recvwin_slide(peer->robuf);
	}

	/* rxr_pkt_proc_rts will write error cq entry if needed */
	ret = rxr_pkt_proc_rts(ep, pkt_entry);
	if (OFI_UNLIKELY(ret))
		return;

	/* process pending items in reorder buff */
	if (rxr_need_sas_ordering(ep))
		rxr_cq_proc_pending_items_in_recvwin(ep, peer);
}

