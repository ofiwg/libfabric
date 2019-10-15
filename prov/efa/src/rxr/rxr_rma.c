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
#include <ofi_mem.h>
#include <ofi_iov.h>
#include "efa.h"
#include "rxr.h"
#include "rxr_rma.h"

char *rxr_rma_init_rts_hdr(struct rxr_ep *ep,
			   struct rxr_tx_entry *tx_entry,
			   struct rxr_pkt_entry *pkt_entry,
			   char *hdr)
{
	int rmalen;
	struct rxr_rts_hdr *rts_hdr;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rts_hdr->rma_iov_count = 0;
	assert (tx_entry->cq_entry.flags & FI_RMA);
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

int rxr_rma_verified_copy_iov(struct rxr_ep *ep, struct fi_rma_iov *rma,
			      size_t count, uint32_t flags, struct iovec *iov)
{
	struct util_domain *util_domain;
	int i, ret;

	util_domain = &rxr_ep_domain(ep)->util_domain;

	for (i = 0; i < count; i++) {
		ret = ofi_mr_verify(&util_domain->mr_map,
				    rma[i].len,
				    (uintptr_t *)(&rma[i].addr),
				    rma[i].key,
				    flags);
		if (ret) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"MR verification failed (%s)\n",
				fi_strerror(-ret));
			return -FI_EACCES;
		}

		iov[i].iov_base = (void *)rma[i].addr;
		iov[i].iov_len = rma[i].len;
	}
	return 0;
}

char *rxr_rma_read_rts_hdr(struct rxr_ep *ep,
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

int rxr_rma_proc_write_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
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
		rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);
	rma_hdr = rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
	data = rxr_rma_read_rts_hdr(ep, rx_entry, pkt_entry, rma_hdr);
	data_size = rxr_get_rts_data_size(ep, rts_hdr);
	return rxr_cq_handle_rts_with_data(ep, rx_entry,
					   pkt_entry, data,
					   data_size);
}

int rxr_rma_init_read_rts(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
			  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rma_read_info *read_info;
	char *hdr;

	hdr = rxr_ep_init_rts_hdr(ep, tx_entry, pkt_entry);
	hdr = rxr_rma_init_rts_hdr(ep, tx_entry, pkt_entry, hdr);

	/* no data to send, but need to send rx_id and window */
	read_info = (struct rxr_rma_read_info *)hdr;
	read_info->rma_initiator_rx_id = tx_entry->rma_loc_rx_id;
	read_info->window = tx_entry->rma_window;
	hdr += sizeof(struct rxr_rma_read_info);
	pkt_entry->pkt_size += sizeof(struct rxr_rma_read_info);

	assert(pkt_entry->pkt_size <= ep->mtu_size);
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = (void *)tx_entry;
	return 0;
}

int rxr_rma_proc_read_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	uint64_t tag = ~0;
	int err = 0;
	char *hdr;
	struct rxr_rma_read_info *read_info;
	/*
	 * rma is one sided operation, match is not expected
	 * we need to create a rx entry upon receiving a rts
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, tag, 0, NULL, pkt_entry->addr, ofi_op_read_rsp, 0);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted.\n");
		rxr_eq_write_error(ep, FI_ENOBUFS, -FI_ENOBUFS);
		return -FI_ENOBUFS;
	}

	rx_entry->bytes_done = 0;

	hdr = (char *)rxr_cq_read_rts_hdr(ep, rx_entry, pkt_entry);
	hdr = (char *)rxr_rma_read_rts_hdr(ep, rx_entry, pkt_entry, hdr);
	read_info = (struct rxr_rma_read_info *)hdr;

	rx_entry->rma_initiator_rx_id = read_info->rma_initiator_rx_id;
	rx_entry->window = read_info->window;
	assert(rx_entry->window > 0);

	tx_entry = rxr_rma_alloc_readrsp_tx_entry(ep, rx_entry);
	assert(tx_entry);
	/* the only difference between a read response packet and
	 * a data packet is that read response packet has remote EP tx_id
	 * which initiator EP rx_entry need to send CTS back
	 */
	err = rxr_ep_post_ctrl_or_queue(ep, RXR_TX_ENTRY, tx_entry, RXR_READRSP_PKT, 0);
	if (OFI_UNLIKELY(err)) {
		if (rxr_cq_handle_tx_error(ep, tx_entry, err))
			assert(0 && "failed to write err cq entry");
		rxr_release_tx_entry(ep, tx_entry);
		rxr_release_rx_entry(ep, rx_entry);
	} else {
		rx_entry->state = RXR_RX_WAIT_READ_FINISH;
	}

	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return err;
}

/* Upon receiving a read request, Remote EP call this function to create
 * a tx entry for sending data back.
 */
struct rxr_tx_entry *
rxr_rma_alloc_readrsp_tx_entry(struct rxr_ep *rxr_ep,
			       struct rxr_rx_entry *rx_entry)
{
	struct rxr_tx_entry *tx_entry;
	struct fi_msg msg;

	tx_entry = ofi_buf_alloc(rxr_ep->readrsp_tx_entry_pool);
	if (OFI_UNLIKELY(!tx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Read Response TX entries exhausted.\n");
		return NULL;
	}

	assert(tx_entry);
#if ENABLE_DEBUG
	dlist_insert_tail(&tx_entry->tx_entry_entry, &rxr_ep->tx_entry_list);
#endif

	msg.msg_iov = rx_entry->iov;
	msg.iov_count = rx_entry->iov_count;
	msg.addr = rx_entry->addr;
	msg.desc = NULL;
	msg.context = NULL;
	msg.data = 0;

	/*
	 * this tx_entry works similar to a send tx_entry thus its op was
	 * set to ofi_op_msg. Note this tx_entry will not write a completion
	 */
	rxr_tx_entry_init(rxr_ep, tx_entry, &msg, ofi_op_msg, 0);

	tx_entry->cq_entry.flags |= FI_READ;
	/* rma_loc_rx_id is for later retrieve of rx_entry
	 * to write rx_completion
	 */
	tx_entry->rma_loc_rx_id = rx_entry->rx_id;

	/* the following is essentially handle CTS */
	tx_entry->rx_id = rx_entry->rma_initiator_rx_id;
	tx_entry->window = rx_entry->window;

	/* this tx_entry does not send rts
	 * therefore should not increase msg_id
	 */
	tx_entry->msg_id = 0;
	return tx_entry;
}

int rxr_rma_init_readrsp_pkt(struct rxr_ep *ep,
			     struct rxr_tx_entry *tx_entry,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_readrsp_pkt *readrsp_pkt;
	struct rxr_readrsp_hdr *readrsp_hdr;
	size_t mtu = ep->mtu_size;

	readrsp_pkt = (struct rxr_readrsp_pkt *)pkt_entry->pkt;
	readrsp_hdr = &readrsp_pkt->hdr;
	readrsp_hdr->type = RXR_READRSP_PKT;
	readrsp_hdr->version = RXR_PROTOCOL_VERSION;
	readrsp_hdr->flags = 0;
	readrsp_hdr->tx_id = tx_entry->tx_id;
	readrsp_hdr->rx_id = tx_entry->rx_id;
	readrsp_hdr->seg_size = ofi_copy_from_iov(readrsp_pkt->data,
						  mtu - RXR_READRSP_HDR_SIZE,
						  tx_entry->iov,
						  tx_entry->iov_count, 0);
	pkt_entry->pkt_size = RXR_READRSP_HDR_SIZE + readrsp_hdr->seg_size;
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = tx_entry;
	return 0;
}

void rxr_rma_handle_readrsp_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;
	size_t data_len;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	data_len = rxr_get_readrsp_hdr(pkt_entry->pkt)->seg_size;
	tx_entry->state = RXR_TX_SENT_READRSP;
	tx_entry->bytes_sent += data_len;
	tx_entry->window -= data_len;
	assert(tx_entry->window >= 0);
	if (tx_entry->bytes_sent < tx_entry->total_len) {
		if (efa_mr_cache_enable && rxr_ep_mr_local(ep))
			rxr_inline_mr_reg(rxr_ep_domain(ep), tx_entry);

		tx_entry->state = RXR_TX_SEND;
		dlist_insert_tail(&tx_entry->entry,
				  &ep->tx_pending_list);
	}
}

/* EOR packet functions */
int rxr_rma_init_eor_pkt(struct rxr_ep *ep, struct rxr_rx_entry *rx_entry, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_eor_hdr *eor_hdr;

	eor_hdr = (struct rxr_eor_hdr *)pkt_entry->pkt;
	eor_hdr->type = RXR_EOR_PKT;
	eor_hdr->version = RXR_PROTOCOL_VERSION;
	eor_hdr->flags = 0;
	eor_hdr->tx_id = rx_entry->tx_id;
	eor_hdr->rx_id = rx_entry->rx_id;
	pkt_entry->pkt_size = sizeof(struct rxr_eor_hdr);
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->x_entry = rx_entry;
	return 0;
}

void rxr_rma_handle_eor_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
}

struct rxr_tx_entry *
rxr_rma_alloc_tx_entry(struct rxr_ep *rxr_ep,
		       const struct fi_msg_rma *msg_rma,
		       uint32_t op,
		       uint64_t flags)
{
	struct rxr_tx_entry *tx_entry;
	struct fi_msg msg;

	tx_entry = ofi_buf_alloc(rxr_ep->tx_entry_pool);
	if (OFI_UNLIKELY(!tx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "TX entries exhausted.\n");
		return NULL;
	}

	msg.addr = msg_rma->addr;
	msg.msg_iov = msg_rma->msg_iov;
	msg.context = msg_rma->context;
	msg.iov_count = msg_rma->iov_count;
	msg.data = msg_rma->data;
	msg.desc = msg_rma->desc;
	rxr_tx_entry_init(rxr_ep, tx_entry, &msg, op, flags);

	assert(msg_rma->rma_iov_count > 0);
	assert(msg_rma->rma_iov);
	tx_entry->rma_iov_count = msg_rma->rma_iov_count;
	memcpy(tx_entry->rma_iov, msg_rma->rma_iov,
	       sizeof(struct fi_rma_iov) * msg_rma->rma_iov_count);

#if ENABLE_DEBUG
	dlist_insert_tail(&tx_entry->tx_entry_entry, &rxr_ep->tx_entry_list);
#endif
	return tx_entry;
}

size_t rxr_rma_post_shm_rma(struct rxr_ep *rxr_ep, struct rxr_tx_entry *tx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	struct fi_msg_rma msg;
	struct rxr_rma_context_pkt *rma_context_pkt;
	struct rxr_peer *peer;
	fi_addr_t shm_fiaddr;
	int ret;

	tx_entry->state = RXR_TX_SHM_RMA;

	peer = rxr_ep_get_peer(rxr_ep, tx_entry->addr);
	shm_fiaddr = peer->shm_fiaddr;
	pkt_entry = rxr_get_pkt_entry(rxr_ep, rxr_ep->tx_pkt_shm_pool);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	pkt_entry->x_entry = (void *)tx_entry;
	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->pkt;
	rma_context_pkt->type = RXR_RMA_CONTEXT_PKT;
	rma_context_pkt->version = RXR_PROTOCOL_VERSION;
	rma_context_pkt->tx_id = tx_entry->tx_id;

	msg.msg_iov = tx_entry->iov;
	msg.iov_count = tx_entry->iov_count;
	msg.addr = shm_fiaddr;
	msg.rma_iov = tx_entry->rma_iov;
	msg.rma_iov_count = tx_entry->rma_iov_count;
	msg.context = pkt_entry;

	if (tx_entry->cq_entry.flags & FI_READ) {
		rma_context_pkt->rma_context_type = RXR_SHM_RMA_READ;
		msg.data = 0;
		ret = fi_readmsg(rxr_ep->shm_ep, &msg, tx_entry->fi_flags);
	} else {
		rma_context_pkt->rma_context_type = RXR_SHM_RMA_WRITE;
		msg.data = tx_entry->cq_entry.data;
		ret = fi_writemsg(rxr_ep->shm_ep, &msg, tx_entry->fi_flags);
	}

	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN) {
			tx_entry->state = RXR_TX_QUEUED_SHM_RMA;
			dlist_insert_tail(&tx_entry->queued_entry,
					  &rxr_ep->tx_entry_queued_list);
			return 0;
		}
		rxr_release_tx_entry(rxr_ep, tx_entry);
	}

	return ret;
}

/* rma_read functions */
ssize_t rxr_rma_post_efa_read(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry)
{
	int err, window, credits;
	struct rxr_peer *peer;
	struct rxr_rx_entry *rx_entry;

	/* create a rx_entry to receve data
	 * use ofi_op_msg for its op.
	 * it does not write a rx completion.
	 */
	rx_entry = rxr_ep_get_rx_entry(ep, tx_entry->iov,
				       tx_entry->iov_count,
				       0, ~0, NULL,
				       tx_entry->addr, ofi_op_msg, 0);
	if (!rx_entry) {
		rxr_release_tx_entry(ep, tx_entry);
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"RX entries exhausted for read.\n");
		rxr_ep_progress_internal(ep);
		return -FI_EAGAIN;
	}

	/*
	 * this rx_entry does not know its tx_id, because remote
	 * tx_entry has not been created yet.
	 * set tx_id to -1, and the correct one will be filled in
	 * rxr_cq_handle_readrsp()
	 */
	assert(rx_entry);
	rx_entry->tx_id = -1;
	rx_entry->cq_entry.flags |= FI_READ;
	rx_entry->total_len = rx_entry->cq_entry.len;

	/*
	 * there will not be a CTS for fi_read, we calculate CTS
	 * window here, and send it via RTS.
	 * meanwhile set rx_entry->state to RXR_RX_RECV so that
	 * this rx_entry is ready to receive.
	 */

	/* But if there is no available buffer, we do not even proceed.
	 * call rxr_ep_progress_internal() might release some buffer
	 */
	if (ep->available_data_bufs == 0) {
		rxr_release_tx_entry(ep, tx_entry);
		rxr_release_rx_entry(ep, rx_entry);
		rxr_ep_progress_internal(ep);
		return -FI_EAGAIN;
	}

	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);
	rxr_ep_calc_cts_window_credits(ep, peer,
				       tx_entry->total_len,
				       tx_entry->credit_request,
				       &window,
				       &credits);

	rx_entry->window = window;
	rx_entry->credit_cts = credits;

	rx_entry->state = RXR_RX_RECV;
	/* rma_loc_tx_id is used in rxr_cq_handle_rx_completion()
	 * to locate the tx_entry for tx completion.
	 */
	rx_entry->rma_loc_tx_id = tx_entry->tx_id;
#if ENABLE_DEBUG
	dlist_insert_tail(&rx_entry->rx_pending_entry,
			  &ep->rx_pending_list);
	ep->rx_pending++;
#endif
	/*
	 * this tx_entry does not need a rx_id, because it does not
	 * send any data.
	 * the rma_loc_rx_id and rma_window will be sent to remote EP
	 * via RTS
	 */
	tx_entry->rma_loc_rx_id = rx_entry->rx_id;
	tx_entry->rma_window = rx_entry->window;
	tx_entry->msg_id = (peer->next_msg_id != ~0) ?
			    peer->next_msg_id++ : ++peer->next_msg_id;

	err = rxr_ep_post_ctrl_or_queue(ep, RXR_TX_ENTRY, tx_entry, RXR_RTS_PKT, 0);
	if (OFI_UNLIKELY(err)) {
		rxr_release_tx_entry(ep, tx_entry);
		peer->next_msg_id--;
	}

	return err;
}

ssize_t rxr_rma_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags)
{
	ssize_t err;
	struct rxr_ep *rxr_ep;
	struct rxr_peer *peer;
	struct rxr_tx_entry *tx_entry;

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "read iov_len: %lu flags: %lx\n",
	       ofi_total_iov_len(msg->msg_iov, msg->iov_count),
	       flags);

	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);

	assert(msg->iov_count <= rxr_ep->tx_iov_limit);

	rxr_perfset_start(rxr_ep, perf_rxr_tx);
	fastlock_acquire(&rxr_ep->util_ep.lock);

	if (OFI_UNLIKELY(is_tx_res_full(rxr_ep)))
		return -FI_EAGAIN;

	tx_entry = rxr_rma_alloc_tx_entry(rxr_ep, msg, ofi_op_read_req, flags);
	if (OFI_UNLIKELY(!tx_entry)) {
		rxr_ep_progress_internal(rxr_ep);
		return -FI_EAGAIN;
	}

	peer = rxr_ep_get_peer(rxr_ep, msg->addr);
	assert(peer);
	if (rxr_env.enable_shm_transfer && peer->is_local) {
		err = rxr_rma_post_shm_rma(rxr_ep, tx_entry);
	} else {
		err = rxr_ep_set_tx_credit_request(rxr_ep, tx_entry);
		if (OFI_UNLIKELY(err)) {
			rxr_release_tx_entry(rxr_ep, tx_entry);
			goto out;
		}

		err = rxr_rma_post_efa_read(rxr_ep, tx_entry);
	}

out:
	fastlock_release(&rxr_ep->util_ep.lock);
	rxr_perfset_end(rxr_ep, perf_rxr_tx);
	return err;
}

ssize_t rxr_rma_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		      size_t iov_count, fi_addr_t src_addr, uint64_t addr,
		      uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, iov_count);
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = iov_count;
	msg.addr = src_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	return rxr_rma_readmsg(ep, &msg, 0);
}

ssize_t rxr_rma_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
		     fi_addr_t src_addr, uint64_t addr, uint64_t key,
		     void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_rma_readv(ep, &iov, &desc, 1, src_addr, addr, key, context);
}

/* rma_write functions */
ssize_t rxr_rma_writemsg(struct fid_ep *ep,
			 const struct fi_msg_rma *msg,
			 uint64_t flags)
{
	ssize_t err;
	struct rxr_ep *rxr_ep;
	struct rxr_peer *peer;
	struct rxr_tx_entry *tx_entry;

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "write iov_len %lu flags: %lx\n",
	       ofi_total_iov_len(msg->msg_iov, msg->iov_count),
	       flags);

	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);
	assert(msg->iov_count <= rxr_ep->tx_iov_limit);

	rxr_perfset_start(rxr_ep, perf_rxr_tx);
	fastlock_acquire(&rxr_ep->util_ep.lock);

	peer = rxr_ep_get_peer(rxr_ep, msg->addr);
	assert(peer);

	tx_entry = rxr_rma_alloc_tx_entry(rxr_ep, msg, ofi_op_write, flags);
	if (OFI_UNLIKELY(!tx_entry)) {
		rxr_ep_progress_internal(rxr_ep);
		err = -FI_EAGAIN;
		goto out;
	}

	if (rxr_env.enable_shm_transfer && peer->is_local) {
		err = rxr_rma_post_shm_rma(rxr_ep, tx_entry);
	}  else {
		err = rxr_ep_set_tx_credit_request(rxr_ep, tx_entry);
		if (OFI_UNLIKELY(err)) {
			rxr_release_tx_entry(rxr_ep, tx_entry);
			goto out;
		}

		tx_entry->msg_id = (peer->next_msg_id != ~0) ?
				    peer->next_msg_id++ : ++peer->next_msg_id;

		err = rxr_ep_post_ctrl_or_queue(rxr_ep, RXR_TX_ENTRY, tx_entry, RXR_RTS_PKT, 0);
		if (OFI_UNLIKELY(err)) {
			rxr_release_tx_entry(rxr_ep, tx_entry);
			peer->next_msg_id--;
		}
	}

out:
	fastlock_release(&rxr_ep->util_ep.lock);
	rxr_perfset_end(rxr_ep, perf_rxr_tx);
	return err;
}

ssize_t rxr_rma_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		       size_t iov_count, fi_addr_t dest_addr, uint64_t addr,
		       uint64_t key, void *context)
{
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, iov_count);
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = iov_count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;

	return rxr_rma_writemsg(ep, &msg, 0);
}

ssize_t rxr_rma_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		      fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		      void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_rma_writev(ep, &iov, &desc, 1, dest_addr, addr, key, context);
}

ssize_t rxr_rma_writedata(struct fid_ep *ep, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t addr, uint64_t key, void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	struct fi_msg_rma msg;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.desc = desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.data = data;

	return rxr_rma_writemsg(ep, &msg, FI_REMOTE_CQ_DATA);
}

ssize_t rxr_rma_inject_write(struct fid_ep *ep, const void *buf, size_t len,
			     fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.addr = dest_addr;

	return rxr_rma_writemsg(ep, &msg, FI_INJECT | RXR_NO_COMPLETION);
}

ssize_t rxr_rma_inject_writedata(struct fid_ep *ep, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr, uint64_t addr,
				 uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = &iov;
	msg.iov_count = 1;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.addr = dest_addr;
	msg.data = data;

	return rxr_rma_writemsg(ep, &msg, FI_INJECT | RXR_NO_COMPLETION |
				FI_REMOTE_CQ_DATA);
}

struct fi_ops_rma rxr_ops_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = rxr_rma_read,
	.readv = rxr_rma_readv,
	.readmsg = rxr_rma_readmsg,
	.write = rxr_rma_write,
	.writev = rxr_rma_writev,
	.writemsg = rxr_rma_writemsg,
	.inject = rxr_rma_inject_write,
	.writedata = rxr_rma_writedata,
	.injectdata = rxr_rma_inject_writedata,
};
