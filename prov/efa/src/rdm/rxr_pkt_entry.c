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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>

#include "efa.h"
#include "efa_av.h"
#include "efa_tp.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_rma.h"
#include "rxr_op_entry.h"
#include "rxr_pkt_cmd.h"
#include "rxr_pkt_pool.h"

/**
 * @brief allocate a packet entry
 *
 * Allocate a packet entry from given packet packet pool
 * @param[in,out] ep end point
 * @param[in,out] pkt_pool packet pool
 * @param[in] alloc_type allocation type see `enum rxr_pkt_entry_alloc_type`
 * @return on success return pointer of the allocated packet entry.
 *         on failure return NULL
 * @related rxr_pkt_entry
 */
struct rxr_pkt_entry *rxr_pkt_entry_alloc(struct rxr_ep *ep, struct rxr_pkt_pool *pkt_pool,
			enum rxr_pkt_entry_alloc_type alloc_type)
{
	struct rxr_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ofi_buf_alloc_ex(pkt_pool->entry_pool, &mr);
	if (!pkt_entry)
		return NULL;

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(pkt_entry, sizeof(struct rxr_pkt_entry) + ep->mtu_size);
#endif

	pkt_entry->send = NULL;
	if (pkt_pool->sendv_pool) {
		pkt_entry->send = ofi_buf_alloc(pkt_pool->sendv_pool);
		assert(pkt_entry->send);
		pkt_entry->send->iov_count = 0; /* rxr_pkt_init methods expect iov_count = 0 */
	}

	pkt_entry->send_wr = NULL;
	if (pkt_pool->efa_send_wr_pool)
		pkt_entry->send_wr = &pkt_pool->efa_send_wr_pool[ofi_buf_index(pkt_entry)];

	dlist_init(&pkt_entry->entry);

#if ENABLE_DEBUG
	dlist_init(&pkt_entry->dbg_entry);
#endif

	/* Initialize necessary fields in pkt_entry.
	 * The memory region allocated by ofi_buf_alloc_ex is not initalized.
	 */
	pkt_entry->mr = mr;
	pkt_entry->alloc_type = alloc_type;
	pkt_entry->flags = RXR_PKT_ENTRY_IN_USE;
	pkt_entry->next = NULL;
	pkt_entry->x_entry = NULL;
	pkt_entry->recv_wr.wr.next = NULL;
	return pkt_entry;
}

/**
 * @brief released packet entry
 *
 * @param[in] pkt_entry packet entry
 *
 * @related rxr_pkt_entry
 */
void rxr_pkt_entry_release(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	if (pkt_entry->send)
		ofi_buf_free(pkt_entry->send);

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(pkt_entry, sizeof(struct rxr_pkt_entry) + ep->mtu_size);
#endif
	pkt_entry->flags = 0;
	ofi_buf_free(pkt_entry);
}

/**
 * @brief release a packet entry used by an TX operation
 *
 * TX operation include send/read_req/write_req/atomic_req
 *
 * @param[in]     ep  the end point
 * @param[in,out] pkt_entry the pkt_entry to be released
 * @related rxr_pkt_entry
 */
void rxr_pkt_entry_release_tx(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_peer *peer;

#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
	/*
	 * Decrement rnr_queued_pkts counter and reset backoff for this peer if
	 * we get a send completion for a retransmitted packet.
	 */
	if (OFI_UNLIKELY(pkt_entry->flags & RXR_PKT_ENTRY_RNR_RETRANSMIT)) {
		peer = rxr_ep_get_peer(ep, pkt_entry->addr);
		assert(peer);
		peer->rnr_queued_pkt_cnt--;
		peer->rnr_backoff_wait_time = 0;
		if (peer->flags & EFA_RDM_PEER_IN_BACKOFF) {
			dlist_remove(&peer->rnr_backoff_entry);
			peer->flags &= ~EFA_RDM_PEER_IN_BACKOFF;
		}
		EFA_DBG(FI_LOG_EP_DATA,
		       "reset backoff timer for peer: %" PRIu64 "\n",
		       pkt_entry->addr);
	}

	rxr_pkt_entry_release(ep, pkt_entry);
}

/**
 * @brief release a packet entry used by a RX operation
 *
 * RX operation include receive/read_response/write_response/atomic_response
 * rxr_pkt_entry_release_rx() release a rx packet entry.
 * It requires input pkt_entry to be unlinked.
 *
 * RX packet entry can be linked when medium message protocol
 * is used.
 *
 * In that case, caller is responsible to unlink the pkt_entry
 * can call this function on next packet entry.
 * @param[in]     ep  the end point
 * @param[in,out] pkt_entry the pkt_entry to be released
 * @related rxr_pkt_entry
 */
void rxr_pkt_entry_release_rx(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry)
{
	assert(pkt_entry->next == NULL);

	if (ep->use_zcpy_rx && pkt_entry->alloc_type == RXR_PKT_FROM_USER_BUFFER)
		return;

	if (pkt_entry->alloc_type == RXR_PKT_FROM_EFA_RX_POOL) {
		ep->efa_rx_pkts_to_post++;
	} else if (pkt_entry->alloc_type == RXR_PKT_FROM_SHM_RX_POOL) {
		ep->shm_rx_pkts_to_post++;
	} else if (pkt_entry->alloc_type == RXR_PKT_FROM_READ_COPY_POOL) {
		assert(ep->rx_readcopy_pkt_pool_used > 0);
		ep->rx_readcopy_pkt_pool_used--;
	}

#if ENABLE_DEBUG
	dlist_remove(&pkt_entry->dbg_entry);
#endif
	rxr_pkt_entry_release(ep, pkt_entry);
}

void rxr_pkt_entry_copy(struct rxr_ep *ep,
			struct rxr_pkt_entry *dest,
			struct rxr_pkt_entry *src)
{
	EFA_DBG(FI_LOG_EP_CTRL,
	       "Copying packet out of posted buffer! src_entry_alloc_type: %d desc_entry_alloc_type: %d\n",
		src->alloc_type, dest->alloc_type);
	dlist_init(&dest->entry);
#if ENABLE_DEBUG
	dlist_init(&dest->dbg_entry);
#endif
	/* dest->mr was set in rxr_pkt_entry_alloc(), and
	 * is tied to the memory region, therefore should
	 * not be changed.
	 */
	dest->x_entry = src->x_entry;
	dest->pkt_size = src->pkt_size;
	dest->addr = src->addr;
	dest->flags = RXR_PKT_ENTRY_IN_USE;
	dest->next = NULL;
	assert(src->pkt_size > 0);
	memcpy(dest->wiredata, src->wiredata, src->pkt_size);
}

/**
 * @brief create a copy of unexpected packet entry
 *
 * Handle copying or updating the metadata for an unexpected packet.
 *
 * Packets from the EFA RX pool will be copied into a separate buffer not
 * registered with the device (if this option is enabled) so that we can repost
 * the registered buffer again to keep the EFA RX queue full. Packets from the
 * SHM RX pool will also be copied to reuse the unexpected message pool.
 *
 * @param[in]     ep  the end point
 * @param[in,out] pkt_entry_ptr unexpected packet, if this packet is copied to
 *                a new memory region this pointer will be updated.
 *
 * @return	  struct rxr_pkt_entry of the updated or copied packet, NULL on
 * 		  allocation failure.
 */
struct rxr_pkt_entry *rxr_pkt_get_unexp(struct rxr_ep *ep,
					struct rxr_pkt_entry **pkt_entry_ptr)
{
	struct rxr_pkt_entry *unexp_pkt_entry;
	enum rxr_pkt_entry_alloc_type type;

	type = (*pkt_entry_ptr)->alloc_type;

	if (rxr_env.rx_copy_unexp && (type == RXR_PKT_FROM_EFA_RX_POOL ||
				      type == RXR_PKT_FROM_SHM_RX_POOL)) {
		unexp_pkt_entry = rxr_pkt_entry_clone(ep, ep->rx_unexp_pkt_pool,
						      RXR_PKT_FROM_UNEXP_POOL,
						      *pkt_entry_ptr);
		if (OFI_UNLIKELY(!unexp_pkt_entry)) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"Unable to allocate rx_pkt_entry for unexp msg\n");
			return NULL;
		}
		rxr_pkt_entry_release_rx(ep, *pkt_entry_ptr);
		*pkt_entry_ptr = unexp_pkt_entry;
	} else {
		unexp_pkt_entry = *pkt_entry_ptr;
	}

	return unexp_pkt_entry;
}

void rxr_pkt_entry_release_cloned(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_pkt_entry *next;

	while (pkt_entry) {
		assert(pkt_entry->alloc_type == RXR_PKT_FROM_OOO_POOL ||
		       pkt_entry->alloc_type == RXR_PKT_FROM_UNEXP_POOL);
		rxr_pkt_entry_release(ep, pkt_entry);
		next = pkt_entry->next;
		pkt_entry = next;
	}
}

/**
 * @brief clone a packet entry
 *
 * This function is used on receive side to make a copy of a packet whose memory is on bounce
 * buffer using other buffer pool, so the original packet can be released and posted to device.
 *
 * @param ep
 * @param pkt_pool
 * @param alloc_type
 * @param src
 * @return struct rxr_pkt_entry*
 * @related rxr_pkt_entry
 */
struct rxr_pkt_entry *rxr_pkt_entry_clone(struct rxr_ep *ep,
					  struct rxr_pkt_pool *pkt_pool,
					  enum rxr_pkt_entry_alloc_type alloc_type,
					  struct rxr_pkt_entry *src)
{
	struct rxr_pkt_entry *root = NULL;
	struct rxr_pkt_entry *dst;

	assert(src);
	assert(alloc_type == RXR_PKT_FROM_OOO_POOL ||
	       alloc_type == RXR_PKT_FROM_UNEXP_POOL ||
	       alloc_type == RXR_PKT_FROM_READ_COPY_POOL);

	dst = rxr_pkt_entry_alloc(ep, pkt_pool, alloc_type);
	if (!dst)
		return NULL;

	if (alloc_type == RXR_PKT_FROM_READ_COPY_POOL) {
		assert(pkt_pool == ep->rx_readcopy_pkt_pool);
		ep->rx_readcopy_pkt_pool_used++;
		ep->rx_readcopy_pkt_pool_max_used = MAX(ep->rx_readcopy_pkt_pool_used,
							ep->rx_readcopy_pkt_pool_max_used);
	}

	rxr_pkt_entry_copy(ep, dst, src);
	root = dst;
	while (src->next) {
		dst->next = rxr_pkt_entry_alloc(ep, pkt_pool, alloc_type);
		if (!dst->next) {
			rxr_pkt_entry_release_cloned(ep, root);
			return NULL;
		}

		rxr_pkt_entry_copy(ep, dst->next, src->next);
		src = src->next;
		dst = dst->next;
	}

	assert(dst && !dst->next);
	return root;
}

void rxr_pkt_entry_append(struct rxr_pkt_entry *dst,
			  struct rxr_pkt_entry *src)
{
	assert(dst);

	while (dst->next)
		dst = dst->next;
	assert(dst && !dst->next);
	dst->next = src;
}

/**
 * @brief Populate pkt_entry->ibv_send_wr with the information stored in pkt_entry,
 * and send it out
 *
 * @param[in] ep	rxr endpoint
 * @param[in] pkt_entry	packet entry to be sent
 * @param[in] flags	flags to be applied to the send operation
 * @return		0 on success
 * 			On error, a negative value corresponding to fabric errno
 */
ssize_t rxr_pkt_entry_send(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			   uint64_t flags)
{
	assert(pkt_entry->pkt_size);

	struct efa_rdm_peer *peer;
	struct rxr_pkt_sendv *send = pkt_entry->send;
	struct ibv_send_wr *bad_wr, *send_wr;
	struct ibv_sge *sge;
	int ret, total_len;
	struct efa_conn *conn;

	/* EFA device supports a maximum of 2 iov/SGE
	 */
	assert(send->iov_count <= 2);

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);
	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF)
		return -FI_EAGAIN;

	conn = efa_av_addr_to_conn(ep->base_ep.av, pkt_entry->addr);
	assert(conn && conn->ep_addr);

	assert(send);
	if (send->iov_count == 0) {
		send->iov_count = 1;
		send->iov[0].iov_base = pkt_entry->wiredata;
		send->iov[0].iov_len = pkt_entry->pkt_size;
		send->desc[0] = (pkt_entry->alloc_type == RXR_PKT_FROM_SHM_TX_POOL) ? NULL : pkt_entry->mr;
	}

#if ENABLE_DEBUG
	dlist_insert_tail(&pkt_entry->dbg_entry, &ep->tx_pkt_list);
#ifdef ENABLE_RXR_PKT_DUMP
	rxr_pkt_print("Sent", ep, (struct rxr_base_hdr *)pkt_entry->wiredata);
#endif
#endif

	if (pkt_entry->alloc_type == RXR_PKT_FROM_SHM_TX_POOL) {
		ret = fi_sendv(ep->shm_ep, send->iov, NULL, send->iov_count, peer->shm_fiaddr, pkt_entry);
		goto out;
	}

	assert(pkt_entry->send_wr);
	send_wr = &pkt_entry->send_wr->wr;
	send_wr->num_sge = send->iov_count;
	send_wr->sg_list = pkt_entry->send_wr->sge;
	send_wr->next = NULL;
	send_wr->send_flags = 0;

	total_len = 0;
	for (int i = 0; i < send->iov_count; i++) {
		sge = &send_wr->sg_list[i];
		sge->addr = (uintptr_t)send->iov[i].iov_base;
		sge->length = send->iov[i].iov_len;
		sge->lkey = ((struct efa_mr *)send->desc[i])->ibv_mr->lkey;
		total_len += sge->length;
	}

	if (total_len <= rxr_ep_domain(ep)->device->efa_attr.inline_buf_size &&
	    !rxr_pkt_entry_has_hmem_mr(send))
		send_wr->send_flags |= IBV_SEND_INLINE;

	send_wr->opcode = IBV_WR_SEND;
	send_wr->wr_id = (uintptr_t)pkt_entry;
	send_wr->wr.ud.ah = conn->ah->ibv_ah;
	send_wr->wr.ud.remote_qpn = conn->ep_addr->qpn;
	send_wr->wr.ud.remote_qkey = conn->ep_addr->qkey;

	ep->base_ep.xmit_more_wr_tail->next = send_wr;
	ep->base_ep.xmit_more_wr_tail = send_wr;

	if (flags & FI_MORE) {
		rxr_ep_record_tx_op_submitted(ep, pkt_entry);
		return 0;
	}

	ret = efa_rdm_ep_post_flush(ep, &bad_wr);

out:
	if (OFI_UNLIKELY(ret)) {
		return ret;
	}

	rxr_ep_record_tx_op_submitted(ep, pkt_entry);
	return 0;
}

/**
 * @brief post one read request
 *
 * This function posts one read request.
 *
 * @param[in]		pkt_entry	read_entry that has information of the read request.
 * @param[in,out]	ep		endpoint
 * @param[in]		local_buf 	local buffer, where data will be copied to.
 * @param[in]		len		read size.
 * @param[in]		desc		memory descriptor of local buffer.
 * @param[in]		remote_buff	remote buffer, where data will be read from.
 * @param[in]		remote_key	memory key of remote buffer.
 * @return	On success, return 0
 * 		On failure, return a negative error code.
 */
int rxr_pkt_entry_read(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
		       void *local_buf, size_t len, void *desc,
		       uint64_t remote_buf, size_t remote_key)
{
	struct efa_rdm_peer *peer;
	struct efa_qp *qp;
	struct efa_conn *conn;
	struct ibv_sge sge;
	bool self_comm;
	int err = 0;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	if (peer && peer->is_local && ep->use_shm_for_tx) {
		err = fi_read(ep->shm_ep, local_buf, len, efa_mr_get_shm_desc(desc), peer->shm_fiaddr, remote_buf, remote_key, pkt_entry);
	} else {
		self_comm = (peer == NULL);
		if (self_comm)
			pkt_entry->flags |= RXR_PKT_ENTRY_LOCAL_READ;

		qp = ep->base_ep.qp;
		ibv_wr_start(qp->ibv_qp_ex);
		qp->ibv_qp_ex->wr_id = (uintptr_t)pkt_entry;
		ibv_wr_rdma_read(qp->ibv_qp_ex, remote_key, remote_buf);

		sge.addr = (uint64_t)local_buf;
		sge.length = len;
		sge.lkey = ((struct efa_mr *)desc)->ibv_mr->lkey;

		ibv_wr_set_sge_list(qp->ibv_qp_ex, 1, &sge);
		if (self_comm) {
			ibv_wr_set_ud_addr(qp->ibv_qp_ex, ep->base_ep.self_ah,
					   qp->qp_num, qp->qkey);
		} else {
			conn = efa_av_addr_to_conn(ep->base_ep.av, pkt_entry->addr);
			assert(conn && conn->ep_addr);
			ibv_wr_set_ud_addr(qp->ibv_qp_ex, conn->ah->ibv_ah,
					   conn->ep_addr->qpn, conn->ep_addr->qkey);
		}

		err = ibv_wr_complete(qp->ibv_qp_ex);
	}

	if (OFI_UNLIKELY(err))
		return err;

	rxr_ep_record_tx_op_submitted(ep, pkt_entry);
	return 0;
}

/**
 * @brief post one write request
 *
 * This function posts one write request.
 *
 * @param[in,out]	ep		endpoint
 * @param[in]		pkt_entry	write_entry that has information of the write request.
 * @param[in]		local_buf 	local buffer, where data will be copied from.
 * @param[in]		len		write size.
 * @param[in]		desc		memory descriptor of local buffer.
 * @param[in]		remote_buff	remote buffer, where data will be written to.
 * @param[in]		remote_key	memory key of remote buffer.
 * @return	On success, return 0
 * 		On failure, return a negative error code.
 */
int rxr_pkt_entry_write(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			void *local_buf, size_t len, void *desc,
			uint64_t remote_buf, size_t remote_key)
{
	struct efa_rdm_peer *peer;
	struct efa_qp *qp;
	struct efa_conn *conn;
	struct ibv_sge sge;
	struct rxr_rma_context_pkt *rma_context_pkt;
	struct rxr_op_entry *tx_entry;
	bool self_comm;
	int err = 0;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	tx_entry = (struct rxr_op_entry *)pkt_entry->x_entry;

	rma_context_pkt = (struct rxr_rma_context_pkt *)pkt_entry->wiredata;
	rma_context_pkt->seg_size = len;

	if (peer && peer->is_local && ep->use_shm_for_tx) {
		err = fi_write(ep->shm_ep, local_buf, len, efa_mr_get_shm_desc(desc), peer->shm_fiaddr, remote_buf, remote_key, pkt_entry);
	} else {
		assert(((struct efa_mr *)desc)->ibv_mr);

		self_comm = (peer == NULL);
		if (self_comm)
			pkt_entry->flags |= RXR_PKT_ENTRY_LOCAL_WRITE;

		qp = ep->base_ep.qp;
		ibv_wr_start(qp->ibv_qp_ex);
		qp->ibv_qp_ex->wr_id = (uintptr_t)pkt_entry;

		if (tx_entry->fi_flags & FI_REMOTE_CQ_DATA) {
			/* assert that we are sending the entire buffer as a
			   single IOV when immediate data is also included. */
			assert( len == tx_entry->bytes_write_total_len );
			ibv_wr_rdma_write_imm(qp->ibv_qp_ex, remote_key,
				remote_buf, tx_entry->cq_entry.data);
		} else {
			ibv_wr_rdma_write(qp->ibv_qp_ex, remote_key, remote_buf);
		}

		sge.addr = (uint64_t)local_buf;
		sge.length = len;
		sge.lkey = ((struct efa_mr *)desc)->ibv_mr->lkey;

		/* As an optimization, we should consider implementing multiple-
		   iov writes using an IBV wr with multiple sge entries.
		   For now, each WR contains only one sge. */
		ibv_wr_set_sge_list(qp->ibv_qp_ex, 1, &sge);
		if (self_comm) {
			ibv_wr_set_ud_addr(qp->ibv_qp_ex, ep->base_ep.self_ah,
					   qp->qp_num, qp->qkey);
		} else {
			conn = efa_av_addr_to_conn(ep->base_ep.av, pkt_entry->addr);
			assert(conn && conn->ep_addr);
			ibv_wr_set_ud_addr(qp->ibv_qp_ex, conn->ah->ibv_ah,
					   conn->ep_addr->qpn, conn->ep_addr->qkey);
		}

		err = ibv_wr_complete(qp->ibv_qp_ex);
	}

	if (OFI_UNLIKELY(err))
		return err;

	rxr_ep_record_tx_op_submitted(ep, pkt_entry);
	return 0;
}

/**
 * @brief Post a pkt_entry to receive message from EFA device
 *
 * @param[in] ep	rxr endpoint
 * @param[in] pkt_entry	packet entry to be posted
 * @param[in] desc	Memory registration key
 * @param[in] flags	flags to be applied to the receive operation
 * @return		0 on success
 * 			On error, a negative value corresponding to fabric errno
 *
 */
ssize_t rxr_pkt_entry_recv(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			   void **desc, uint64_t flags)
{
	struct ibv_recv_wr *bad_wr, *recv_wr = &pkt_entry->recv_wr.wr;
	int err;

	recv_wr->wr_id = (uintptr_t)pkt_entry;
	recv_wr->num_sge = 1;	/* Always post one iov/SGE */
	recv_wr->sg_list = pkt_entry->recv_wr.sge;

	recv_wr->sg_list[0].length = ep->mtu_size;
	recv_wr->sg_list[0].lkey = ((struct efa_mr *) desc[0])->ibv_mr->lkey;
	recv_wr->sg_list[0].addr = (uintptr_t)pkt_entry->wiredata;

	ep->base_ep.recv_more_wr_tail->next = recv_wr;
	ep->base_ep.recv_more_wr_tail = recv_wr;

	if (flags & FI_MORE)
		return 0;

#if HAVE_LTTNG
	struct ibv_recv_wr *head = ep->base_ep.recv_more_wr_head.next;
	while (head) {
		efa_tracepoint_wr_id_post_recv((void *) head->wr_id);
		head = head->next;
	}
#endif

	err = ibv_post_recv(ep->base_ep.qp->ibv_qp, ep->base_ep.recv_more_wr_head.next, &bad_wr);
	if (OFI_UNLIKELY(err)) {
		err = (err == ENOMEM) ? -FI_EAGAIN : -err;
	}

	ep->base_ep.recv_more_wr_head.next = NULL;
	ep->base_ep.recv_more_wr_tail = &ep->base_ep.recv_more_wr_head;

	return err;
}

ssize_t rxr_pkt_entry_inject(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry,
			     fi_addr_t addr)
{
	struct efa_rdm_peer *peer;
	ssize_t ret;

	/* currently only EOR packet is injected using shm ep */
	peer = rxr_ep_get_peer(ep, addr);
	assert(peer);

	assert(ep->use_shm_for_tx && peer->is_local);
	ret = fi_inject(ep->shm_ep, rxr_pkt_start(pkt_entry), pkt_entry->pkt_size,
			 peer->shm_fiaddr);

	if (OFI_UNLIKELY(ret))
		return ret;

	rxr_ep_record_tx_op_submitted(ep, pkt_entry);
	return 0;
}

/*
 * Functions for pkt_rx_map
 */
struct rxr_op_entry *rxr_pkt_rx_map_lookup(struct rxr_ep *ep,
					   struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_pkt_rx_map *entry = NULL;
	struct rxr_pkt_rx_key key;

	memset(&key, 0, sizeof(key));
	key.msg_id = rxr_pkt_msg_id(pkt_entry);
	key.addr = pkt_entry->addr;
	HASH_FIND(hh, ep->pkt_rx_map, &key, sizeof(struct rxr_pkt_rx_key), entry);
	return entry ? entry->rx_entry : NULL;
}

void rxr_pkt_rx_map_insert(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry,
			   struct rxr_op_entry *rx_entry)
{
	struct rxr_pkt_rx_map *entry;

	entry = ofi_buf_alloc(ep->map_entry_pool);
	if (OFI_UNLIKELY(!entry)) {
		EFA_WARN(FI_LOG_CQ,
			"Map entries for medium size message exhausted.\n");
		efa_eq_write_error(&ep->base_ep.util_ep, FI_ENOBUFS, FI_EFA_ERR_RX_ENTRIES_EXHAUSTED);
		return;
	}

	memset(&entry->key, 0, sizeof(entry->key));
	entry->key.msg_id = rxr_pkt_msg_id(pkt_entry);
	entry->key.addr = pkt_entry->addr;

#if ENABLE_DEBUG
	{
		struct rxr_pkt_rx_map *existing_entry = NULL;

		HASH_FIND(hh, ep->pkt_rx_map, &entry->key, sizeof(struct rxr_pkt_rx_key), existing_entry);
		assert(!existing_entry);
	}
#endif

	entry->rx_entry = rx_entry;
	HASH_ADD(hh, ep->pkt_rx_map, key, sizeof(struct rxr_pkt_rx_key), entry);
}

void rxr_pkt_rx_map_remove(struct rxr_ep *ep,
			   struct rxr_pkt_entry *pkt_entry,
			   struct rxr_op_entry *rx_entry)
{
	struct rxr_pkt_rx_map *entry;
	struct rxr_pkt_rx_key key;

	memset(&key, 0, sizeof(key));
	key.msg_id = rxr_pkt_msg_id(pkt_entry);
	key.addr = pkt_entry->addr;

	HASH_FIND(hh, ep->pkt_rx_map, &key, sizeof(key), entry);
	assert(entry && entry->rx_entry == rx_entry);
	HASH_DEL(ep->pkt_rx_map, entry);
	ofi_buf_free(entry);
}

