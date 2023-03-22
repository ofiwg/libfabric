/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
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
#include "efa_rdm_error.h"
#include "rxr_cntr.h"
#include "rxr_msg.h"
#include "rxr_read.h"
#include "rxr_pkt_cmd.h"
#include "rxr_tp.h"


void rxr_tx_entry_construct(struct rxr_op_entry *tx_entry,
			    struct rxr_ep *ep,
			    const struct fi_msg *msg,
			    uint32_t op, uint64_t flags)
{
	uint64_t tx_op_flags;

	tx_entry->ep = ep;
	tx_entry->type = RXR_TX_ENTRY;
	tx_entry->op = op;
	tx_entry->tx_id = ofi_buf_index(tx_entry);
	tx_entry->state = RXR_TX_REQ;
	tx_entry->addr = msg->addr;
	tx_entry->peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(tx_entry->peer);
	dlist_insert_tail(&tx_entry->peer_entry, &tx_entry->peer->tx_entry_list);

	tx_entry->rxr_flags = 0;
	tx_entry->bytes_received = 0;
	tx_entry->bytes_copied = 0;
	tx_entry->bytes_acked = 0;
	tx_entry->bytes_sent = 0;
	tx_entry->window = 0;
	tx_entry->iov_count = msg->iov_count;
	tx_entry->msg_id = 0;
	tx_entry->efa_outstanding_tx_ops = 0;
	tx_entry->shm_outstanding_tx_ops = 0;
	dlist_init(&tx_entry->queued_pkts);

	memcpy(tx_entry->iov, msg->msg_iov, sizeof(struct iovec) * msg->iov_count);
	memset(tx_entry->mr, 0, sizeof(*tx_entry->mr) * msg->iov_count);
	if (msg->desc)
		memcpy(tx_entry->desc, msg->desc, sizeof(*msg->desc) * msg->iov_count);
	else
		memset(tx_entry->desc, 0, sizeof(tx_entry->desc));

	/* cq_entry on completion */
	tx_entry->cq_entry.op_context = msg->context;
	tx_entry->cq_entry.data = msg->data;
	tx_entry->cq_entry.len = ofi_total_iov_len(tx_entry->iov, tx_entry->iov_count);
	tx_entry->cq_entry.buf = OFI_LIKELY(tx_entry->cq_entry.len > 0) ? tx_entry->iov[0].iov_base : NULL;

	if (ep->msg_prefix_size > 0) {
		assert(tx_entry->iov[0].iov_len >= ep->msg_prefix_size);
		tx_entry->iov[0].iov_base = (char *)tx_entry->iov[0].iov_base + ep->msg_prefix_size;
		tx_entry->iov[0].iov_len -= ep->msg_prefix_size;
	}
	tx_entry->total_len = ofi_total_iov_len(tx_entry->iov, tx_entry->iov_count);

	/* set flags */
	assert(ep->base_ep.util_ep.tx_msg_flags == 0 ||
	       ep->base_ep.util_ep.tx_msg_flags == FI_COMPLETION);
	tx_op_flags = ep->base_ep.util_ep.tx_op_flags;
	if (ep->base_ep.util_ep.tx_msg_flags == 0)
		tx_op_flags &= ~FI_COMPLETION;
	tx_entry->fi_flags = flags | tx_op_flags;
	tx_entry->bytes_runt = 0;
	tx_entry->max_req_data_size = 0;
	dlist_init(&tx_entry->entry);

	switch (op) {
	case ofi_op_tagged:
		tx_entry->cq_entry.flags = FI_TRANSMIT | FI_MSG | FI_TAGGED;
		break;
	case ofi_op_write:
		tx_entry->cq_entry.flags = FI_RMA | FI_WRITE;
		break;
	case ofi_op_read_req:
		tx_entry->cq_entry.flags = FI_RMA | FI_READ;
		break;
	case ofi_op_msg:
		tx_entry->cq_entry.flags = FI_TRANSMIT | FI_MSG;
		break;
	case ofi_op_atomic:
		tx_entry->cq_entry.flags = (FI_WRITE | FI_ATOMIC);
		break;
	case ofi_op_atomic_fetch:
	case ofi_op_atomic_compare:
		tx_entry->cq_entry.flags = (FI_READ | FI_ATOMIC);
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "invalid operation type\n");
		assert(0);
	}
}

void rxr_tx_entry_release(struct rxr_op_entry *tx_entry)
{
	int i, err = 0;
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;

	assert(tx_entry->peer);
	dlist_remove(&tx_entry->peer_entry);

	for (i = 0; i < tx_entry->iov_count; i++) {
		if (tx_entry->mr[i]) {
			err = fi_close((struct fid *)tx_entry->mr[i]);
			if (OFI_UNLIKELY(err)) {
				EFA_WARN(FI_LOG_CQ, "mr dereg failed. err=%d\n", err);
				efa_eq_write_error(&tx_entry->ep->base_ep.util_ep, err, FI_EFA_ERR_MR_DEREG);
			}

			tx_entry->mr[i] = NULL;
		}
	}

	dlist_remove(&tx_entry->ep_entry);

	dlist_foreach_container_safe(&tx_entry->queued_pkts,
				     struct rxr_pkt_entry,
				     pkt_entry, entry, tmp) {
		rxr_pkt_entry_release_tx(tx_entry->ep, pkt_entry);
	}

	if (tx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_RNR)
		dlist_remove(&tx_entry->queued_rnr_entry);

	if (tx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_CTRL)
		dlist_remove(&tx_entry->queued_ctrl_entry);

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(tx_entry,
			      sizeof(struct rxr_op_entry));
#endif
	tx_entry->state = RXR_OP_FREE;
	ofi_buf_free(tx_entry);
}

void rxr_rx_entry_release(struct rxr_op_entry *rx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	struct dlist_entry *tmp;
	int i, err;

	if (rx_entry->peer)
		dlist_remove(&rx_entry->peer_entry);

	dlist_remove(&rx_entry->ep_entry);

	for (i = 0; i < rx_entry->iov_count; i++) {
		if (rx_entry->mr[i]) {
			err = fi_close((struct fid *)rx_entry->mr[i]);
			if (OFI_UNLIKELY(err)) {
				EFA_WARN(FI_LOG_CQ, "mr dereg failed. err=%d\n", err);
				efa_eq_write_error(&rx_entry->ep->base_ep.util_ep, err, FI_EFA_ERR_MR_DEREG);
			}

			rx_entry->mr[i] = NULL;
		}
	}

	if (!dlist_empty(&rx_entry->queued_pkts)) {
		dlist_foreach_container_safe(&rx_entry->queued_pkts,
					     struct rxr_pkt_entry,
					     pkt_entry, entry, tmp) {
			rxr_pkt_entry_release_tx(rx_entry->ep, pkt_entry);
		}
		dlist_remove(&rx_entry->queued_rnr_entry);
	}

	if (rx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_CTRL)
		dlist_remove(&rx_entry->queued_ctrl_entry);

#ifdef ENABLE_EFA_POISONING
	rxr_poison_mem_region(rx_entry,
			      sizeof(struct rxr_op_entry));
#endif
	rx_entry->state = RXR_OP_FREE;
	ofi_buf_free(rx_entry);
}

/**
 * @brief try to fill the desc field of an op_entry by memory registration
 *
 * The desc field of op_entry contains the memory descriptors of
 * the user's data buffer.
 *
 * For EFA provider, a data buffer's memory descriptor is a pointer to an
 * efa_mr object, which contains the memory registration information
 * of the data buffer.
 *
 * EFA provider does not require user to provide a descriptor when
 * user's data buffer is on host memory (Though user can register
 * its buffer, and provide its descriptor as an optimization).
 *
 * However, there are a few occations that EFA device and shm
 * require memory to be register with them:
 *
 * First, when EFA device is used to send data:
 *
 *   If a non-read based protocol (such as eager, meidum, longcts)
 *   is used, the send buffer must be registered with EFA device.
 *
 *   If a read based protocol is used, both send buffer
 *   and receive buffer must be registered with EFA device.
 *
 * Second, when shm is used:
 *   If eager protocol is used, no registration is needed (because
 *   shm does not require registration for local buffer)
 *
 *   If a read based protocol is used, the send buffer must
 *   be registered with shm, because send buffer is used as
 *   remote buffer in a read based protocol.
 *
 * Therefore, when user did not provide descriptors for the buffer(s),
 * EFA provider need to bridge the gap.
 *
 * On sender side:
 *
 * First, EFA provider can copy the user data to a pre-registered bounce
 * buffer, then send data from bounce buffer. (this happens when
 * EFA device is used, and eager protocol is used)
 *
 * Second, EFA provider can register the user's buffer and fill desc
 * (by calling this function). then send directly from send buffer.
 * Because of the high cost of memory registration, this happens
 * only when MR cache is available, which is checked by the caller
 * of this function on sender side. (this happens when
 *
 * 1. EFA device is used with non-eager protocols and
 * 2. SHM is used with long read protocol
 *
 * This function is not guaranteed to fill all descriptors (which
 * is why the function name has try). When memory registration fail due
 * to limited resources, EFA provider may either fallback to
 * to use bounce buffer, or return -FI_EAGAIN to user and let
 * user run progress engine, which will release memory from
 * MR cache.
 *
 * On receiver side:
 *
 * The only occasion receiver buffer need to be registered is
 * when EFA device is used with a read base message protocol,
 * in which case this function is called for the registration.
 *
 * If the function did not fill desc due to temporarily out
 * of resource. EFA provider need to run progress engine to release
 * memory registration then retry.
 *
 * @param[in,out]	op_entry		contains the inforation of a TX/RX operation
 * @param[in]		mr_iov_start	the IOV index to start generating descriptors
 * @param[in]		access			the access flag for the memory registation.
 *
 */
void rxr_op_entry_try_fill_desc(struct rxr_op_entry *op_entry, int mr_iov_start, uint64_t access)
{
	int i, err;
	struct efa_rdm_peer *peer;

	peer = rxr_ep_get_peer(op_entry->ep, op_entry->addr);

	for (i = mr_iov_start; i < op_entry->iov_count; ++i) {
		if (op_entry->desc[i])
			continue;


		if (peer->is_local && op_entry->ep->use_shm_for_tx) {
			if (access == FI_REMOTE_READ) {
				/* this happens when longread protocol message protocl was used
				 * with shm. The send buffer is going to be read by receiver,
				 * therefore must be registered with shm provider.
				 */
				assert(op_entry->type == RXR_TX_ENTRY);
				err = efa_mr_reg_shm(&rxr_ep_domain(op_entry->ep)->util_domain.domain_fid,
						     op_entry->iov + i,
						     access, &op_entry->mr[i]);
			} else {
				assert(access == FI_SEND || access == FI_RECV);
				/* shm does not require registration for send and recv */
				err = 0;
			}
		} else {
			err = fi_mr_regv(&rxr_ep_domain(op_entry->ep)->util_domain.domain_fid,
					 op_entry->iov + i, 1,
					 access,
					 0, 0, 0, &op_entry->mr[i], NULL);
		}

		if (err) {
			EFA_WARN(FI_LOG_EP_CTRL,
				"fi_mr_reg failed! buf: %p len: %ld access: %lx",
				op_entry->iov[i].iov_base, op_entry->iov[i].iov_len,
				access);

			op_entry->mr[i] = NULL;
		} else {
			op_entry->desc[i] = fi_mr_desc(op_entry->mr[i]);
		}
	}
}

int rxr_tx_entry_prepare_to_be_read(struct rxr_op_entry *tx_entry, struct fi_rma_iov *read_iov)
{
	int i;

	rxr_op_entry_try_fill_desc(tx_entry, 0, FI_REMOTE_READ);

	for (i = 0; i < tx_entry->iov_count; ++i) {
		read_iov[i].addr = (uint64_t)tx_entry->iov[i].iov_base;
		read_iov[i].len = tx_entry->iov[i].iov_len;

		if (!tx_entry->desc[i]) {
			/* rxr_op_entry_try_fill_desc() did not register the memory */
			return -FI_ENOMEM;
		}

		read_iov[i].key = fi_mr_key(tx_entry->desc[i]);
	}

	return 0;
}

/**
 * @brief calculate and set the bytes_runt field of a tx_entry
 *
 * bytes_runt is number of bytes for a message to be sent by runting
 *
 * @param[in]		ep			endpoint
 * @param[in,out]	tx_entry	tx_entry to be set
 */
void rxr_tx_entry_set_runt_size(struct rxr_ep *ep, struct rxr_op_entry *tx_entry)
{
	int iface;
	struct efa_hmem_info *hmem_info;
	struct efa_rdm_peer *peer;

	assert(tx_entry->type == RXR_TX_ENTRY);

	if (tx_entry->bytes_runt > 0)
		return;

	peer = rxr_ep_get_peer(ep, tx_entry->addr);

	iface = tx_entry->desc[0] ? ((struct efa_mr*) tx_entry->desc[0])->peer.iface : FI_HMEM_SYSTEM;
	hmem_info = &rxr_ep_domain(ep)->hmem_info[iface];

	assert(peer);
	tx_entry->bytes_runt = MIN(hmem_info->runt_size - peer->num_runt_bytes_in_flight, tx_entry->total_len);
}

/**
 * @brief total data size that will be sent/received via the multiple REQ packets
 *
 * Multi-req protocols send/receive data via multiple REQ packets. Different
 * protocols have different behavior:
 *
 *     Medium protocols send/receive all data via REQ packets
 *
 *     Runting read protocols send/receive part of the data via REQ packets.
 *     The reminder of the data is sent via other type of packets or via RDMA operations.
 *
 * which is why this function is needed.
 *
 * @param[in]		op_entry		contains operation information
 * @param[in]		pkt_type		REQ packet type
 * @return			size of total data transfered by REQ packets
 */
size_t rxr_op_entry_mulreq_total_data_size(struct rxr_op_entry *op_entry, int pkt_type)
{
	assert(rxr_pkt_type_is_mulreq(pkt_type));

	if (rxr_pkt_type_is_medium(pkt_type)) {
		return op_entry->total_len;
	}

	assert(rxr_pkt_type_is_runt(pkt_type));
	return op_entry->bytes_runt;
}

/**
 * @brief return the maximum data capacity of a REQ packet for an send operation
 *
 * The REQ packet header length is a variable that depends on a number of factors,
 * including:
 *
 *   packet_type, peer_type, cq_data and number of rma iov.
 *
 * As a result the maximum data capacity of a REQ packet for a send operation,(
 * which is the number of bytes of data can be saved in a REQ packet) is different.
 *
 * This function is used to caculate the maxium data capacity.
 *
 * @param[in]		ep			endpoint
 * @param[in]		tx_entry	tx_entry that has all information of
 * 					a send operation
 * @param[in]		pkt_type	type of REQ packet
 *
 * @return		maxiumum number of bytes of data can be save in a REQ packet
 * 			for given send operation and REQ packet type.
 */
size_t rxr_tx_entry_max_req_data_capacity(struct rxr_ep *ep, struct rxr_op_entry *tx_entry, int pkt_type)
{
	struct efa_rdm_peer *peer;
	uint16_t header_flags = 0;
	int max_data_offset;

	assert(pkt_type >= RXR_REQ_PKT_BEGIN);

	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);

	if (peer->is_local && ep->use_shm_for_tx) {
		return rxr_env.shm_max_medium_size;
	}

	if (efa_rdm_peer_need_raw_addr_hdr(peer))
		header_flags |= RXR_REQ_OPT_RAW_ADDR_HDR;
	else if (efa_rdm_peer_need_connid(peer))
		header_flags |= RXR_PKT_CONNID_HDR;

	if (tx_entry->fi_flags & FI_REMOTE_CQ_DATA)
		header_flags |= RXR_REQ_OPT_CQ_DATA_HDR;

	max_data_offset = rxr_pkt_req_hdr_size(pkt_type, header_flags,
					       tx_entry->rma_iov_count);

	if (rxr_pkt_type_is_runtread(pkt_type)) {
		max_data_offset += tx_entry->iov_count * sizeof(struct fi_rma_iov);
	}

	return ep->mtu_size - max_data_offset;
}

/**
 * @brief set the max_req_data_size field of a tx_entry for multi-req
 *
 * Multi-REQ protocols send multiple REQ packets via one call to ibv_post_send() for efficiency.
 * Under such circumstance, it is better that the data size of mulitple REQ packets to be close.
 *
 * To achieve the closeness, the field of max_req_data_size was introduced to rxr_op_entry,
 * and used to limit data size when construct REQ packet.
 *
 * This function set the max_req_data_size properly.
 *
 *
 * @param[in]		ep			endpoint
 * @param[in,out]	tx_entry	tx_entry that has all information of
 * 					a send operation
 * @param[in]		pkt_type	type of REQ packet
 *
 */
void rxr_tx_entry_set_max_req_data_size(struct rxr_ep *ep, struct rxr_op_entry *tx_entry, int pkt_type)
{
	int max_req_data_capacity;
	int mulreq_total_data_size;
	int num_req;
	int memory_alignment = 8;

	assert(rxr_pkt_type_is_mulreq(pkt_type));

	max_req_data_capacity = rxr_tx_entry_max_req_data_capacity(ep, tx_entry, pkt_type);
	assert(max_req_data_capacity);

	mulreq_total_data_size = rxr_op_entry_mulreq_total_data_size(tx_entry, pkt_type);
	assert(mulreq_total_data_size);

	num_req = (mulreq_total_data_size - 1)/max_req_data_capacity + 1;

	if (efa_mr_is_cuda(tx_entry->desc[0])) {
		memory_alignment = 64;
	}

	tx_entry->max_req_data_size = ofi_get_aligned_size((mulreq_total_data_size - 1)/num_req + 1, memory_alignment);
	if (tx_entry->max_req_data_size > max_req_data_capacity)
		tx_entry->max_req_data_size = max_req_data_capacity;
}

/**
 * @brief return number of REQ packets needed to send a message using mulit-req protocol
 *
 * @param[in]		tx_entry		tx_entry with information of the message
 * @param[in]		pkt_type		packet type of the mulit-req protocol
 * @return			number of REQ packets
 */
size_t rxr_tx_entry_num_req(struct rxr_op_entry *tx_entry, int pkt_type)
{
	assert(rxr_pkt_type_is_mulreq(pkt_type));
	assert(tx_entry->max_req_data_size);

	size_t total_size = rxr_op_entry_mulreq_total_data_size(tx_entry, pkt_type);

	return (total_size - tx_entry->bytes_sent - 1)/tx_entry->max_req_data_size + 1;
}

/**
 * @brief handle the situation that an error has happened to an RX (receive) operation
 *
 * This function does the following to handle error:
 *
 * 1. write an error cq entry to notify application the rx
 *    operation failed. If write failed, it will write an eq entry.
 *
 * 2. increase error counter.
 *
 * 3. print warning message about the error with self and peer's
 *    raw address.
 *
 * 4. release resources owned by the RX entry, such as unexpected
 *    packet entry, because the RX operation is aborted.
 *
 * 5. remove the rx_entry from queued rx_entry list for the same reason.
 *
 * Note, It will NOT release the rx_entry because it is still possible to receive
 * packet for this rx_entry.
 *
 * @param[in]	rx_entry	rx_entry encountered error
 * @param[in]	err			positive libfabric error code
 * @param[in]	prov_errno	positive provider specific error code
 */
void rxr_rx_entry_handle_error(struct rxr_op_entry *rx_entry, int err, int prov_errno)
{
	struct rxr_ep *ep;
	struct fi_cq_err_entry err_entry;
	struct util_cq *util_cq;
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;
	int write_cq_err;

	assert(rx_entry->type == RXR_RX_ENTRY);

	memset(&err_entry, 0, sizeof(err_entry));

	ep = rx_entry->ep;
	util_cq = ep->base_ep.util_ep.rx_cq;

	err_entry.err = err;
	err_entry.prov_errno = prov_errno;

	switch (rx_entry->state) {
	case RXR_RX_INIT:
	case RXR_RX_UNEXP:
		dlist_remove(&rx_entry->entry);
		break;
	case RXR_RX_MATCHED:
		break;
	case RXR_RX_RECV:
#if ENABLE_DEBUG
		dlist_remove(&rx_entry->pending_recv_entry);
#endif
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "rx_entry unknown state %d\n",
			rx_entry->state);
		assert(0 && "rx_entry unknown state");
	}

	if (rx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_RNR) {
		dlist_foreach_container_safe(&rx_entry->queued_pkts,
					     struct rxr_pkt_entry,
					     pkt_entry, entry, tmp)
			rxr_pkt_entry_release_tx(ep, pkt_entry);
		dlist_remove(&rx_entry->queued_rnr_entry);
	}

	if (rx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_CTRL)
		dlist_remove(&rx_entry->queued_ctrl_entry);

	if (rx_entry->unexp_pkt) {
		rxr_pkt_entry_release_rx(ep, rx_entry->unexp_pkt);
		rx_entry->unexp_pkt = NULL;
	}

	if (rx_entry->fi_flags & FI_MULTI_RECV)
		rxr_msg_multi_recv_handle_completion(ep, rx_entry);

	err_entry.flags = rx_entry->cq_entry.flags;
	if (rx_entry->state != RXR_RX_UNEXP)
		err_entry.op_context = rx_entry->cq_entry.op_context;
	err_entry.buf = rx_entry->cq_entry.buf;
	err_entry.data = rx_entry->cq_entry.data;
	err_entry.tag = rx_entry->cq_entry.tag;
	if (OFI_UNLIKELY(efa_rdm_error_write_msg(ep, rx_entry->addr, err, prov_errno,
	                                         &err_entry.err_data, &err_entry.err_data_size))) {
		err_entry.err_data_size = 0;
	}

	rxr_msg_multi_recv_free_posted_entry(ep, rx_entry);

	EFA_WARN(FI_LOG_CQ, "err: %d, message: %s (%d)\n", err_entry.err,
	         efa_strerror(err_entry.prov_errno, err_entry.err_data), err_entry.prov_errno);
	/*
	 * TODO: We can't free the rx_entry as we may receive additional
	 * packets for this entry. Add ref counting so the rx_entry can safely
	 * be freed once all packets are accounted for.
	 */
	//rxr_rx_entry_release(rx_entry);

	efa_cntr_report_error(&ep->base_ep.util_ep, err_entry.flags);
	write_cq_err = ofi_cq_write_error(util_cq, &err_entry);
	if (write_cq_err) {
		EFA_WARN(FI_LOG_CQ,
			"Error writing error cq entry when handling RX error");
		efa_eq_write_error(&ep->base_ep.util_ep, err, prov_errno);
	}
}

/**
 * @brief handle the situation that a TX operation encountered error
 *
 * This function does the follow to handle error:
 *
 * 1. write an error cq entry for the TX operation, if writing
 *    CQ error entry failed, it will write eq entry.
 *
 * 2. increase error counter.
 *
 * 3. remove the TX entry from ep->tx_queued_list and ep->tx_pending_list
 *    if the tx_entry is on it.
 *
 * 4. print warning message with self and peer's raw address
 *
 * Note It does NOT release tx entry because it is still possible to receive
 * send completion for this TX entry
 *
 * @param[in]	tx_entry	tx_entry that encountered error
 * @param[in]	err			positive libfabric error code
 * @param[in]	prov_errno	positive EFA provider specific error code
 */
void rxr_tx_entry_handle_error(struct rxr_op_entry *tx_entry, int err, int prov_errno)
{
	struct rxr_ep *ep;
	struct fi_cq_err_entry err_entry;
	struct util_cq *util_cq;
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;
	int write_cq_err;

	ep = tx_entry->ep;
	memset(&err_entry, 0, sizeof(err_entry));

	util_cq = ep->base_ep.util_ep.tx_cq;

	err_entry.err = err;
	err_entry.prov_errno = prov_errno;

	switch (tx_entry->state) {
	case RXR_TX_REQ:
		break;
	case RXR_TX_SEND:
		dlist_remove(&tx_entry->entry);
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "tx_entry unknown state %d\n",
			tx_entry->state);
		assert(0 && "tx_entry unknown state");
	}

	if (tx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_RNR)
		dlist_remove(&tx_entry->queued_rnr_entry);

	if (tx_entry->rxr_flags & RXR_OP_ENTRY_QUEUED_CTRL)
		dlist_remove(&tx_entry->queued_ctrl_entry);

	dlist_foreach_container_safe(&tx_entry->queued_pkts,
				     struct rxr_pkt_entry,
				     pkt_entry, entry, tmp)
		rxr_pkt_entry_release_tx(ep, pkt_entry);

	err_entry.flags = tx_entry->cq_entry.flags;
	err_entry.op_context = tx_entry->cq_entry.op_context;
	err_entry.buf = tx_entry->cq_entry.buf;
	err_entry.data = tx_entry->cq_entry.data;
	err_entry.tag = tx_entry->cq_entry.tag;
	if (OFI_UNLIKELY(efa_rdm_error_write_msg(ep, tx_entry->addr, err, prov_errno,
	                                         &err_entry.err_data, &err_entry.err_data_size))) {
		err_entry.err_data_size = 0;
	}

	EFA_WARN(FI_LOG_CQ, "err: %d, message: %s (%d)\n", err_entry.err,
	         efa_strerror(err_entry.prov_errno, err_entry.err_data), err_entry.prov_errno);

	/*
	 * TODO: We can't free the tx_entry as we may receive a control packet
	 * for this entry. Add ref counting so the tx_entry can safely
	 * be freed once all packets are accounted for.
	 */
	//rxr_tx_entry_release(tx_entry);

	efa_cntr_report_error(&ep->base_ep.util_ep, tx_entry->cq_entry.flags);
	write_cq_err = ofi_cq_write_error(util_cq, &err_entry);
	if (write_cq_err) {
		EFA_WARN(FI_LOG_CQ,
			"Error writing error cq entry when handling TX error");
		efa_eq_write_error(&ep->base_ep.util_ep, err, prov_errno);
	}
}

/**
 * @brief report (to user) that an RX operation has completed
 *
 * An RX operation can be recevie, write response, read response
 * and atomic response.
 * This function will do the following:
 *
 * 1. write an CQ entry. If the message is truncated, (receiving
 *    buffer is too small to hold incming message), the CQ entry
 *    will be an error CQ entry, otherwise the CQ entry will be
 *    normal CQ entry. User will get the CQ entry will it call
 *    fi_cq_read()
 *
 * 2. increase counter. User will get the updated counter when
 *    it call fi_cntr_read()
 *
 * @param[in]		rx_entry	information of the completed RX operation
 */
void rxr_rx_entry_report_completion(struct rxr_op_entry *rx_entry)
{
	struct rxr_ep *ep = rx_entry->ep;
	struct util_cq *rx_cq = ep->base_ep.util_ep.rx_cq;
	int ret = 0;

	if (OFI_UNLIKELY(rx_entry->cq_entry.len < rx_entry->total_len)) {
		EFA_WARN(FI_LOG_CQ,
			"Message truncated! tag: %"PRIu64" incoming message size: %"PRIu64" receiving buffer size: %zu\n",
			rx_entry->cq_entry.tag,	rx_entry->total_len,
			rx_entry->cq_entry.len);

		ret = ofi_cq_write_error_trunc(ep->base_ep.util_ep.rx_cq,
					       rx_entry->cq_entry.op_context,
					       rx_entry->cq_entry.flags,
					       rx_entry->total_len,
					       rx_entry->cq_entry.buf,
					       rx_entry->cq_entry.data,
					       rx_entry->cq_entry.tag,
					       rx_entry->total_len -
					       rx_entry->cq_entry.len);

		rxr_rm_rx_cq_check(ep, rx_cq);

		if (OFI_UNLIKELY(ret)) {
			EFA_WARN(FI_LOG_CQ,
				"Unable to write recv error cq: %s\n",
				fi_strerror(-ret));
			return;
		}

		rx_entry->fi_flags |= RXR_TX_ENTRY_NO_COMPLETION;
		efa_cntr_report_error(&ep->base_ep.util_ep, rx_entry->cq_entry.flags);
		return;
	}

	if (!(rx_entry->rxr_flags & RXR_RX_ENTRY_RECV_CANCEL) &&
	    (ofi_need_completion(rxr_rx_flags(ep), rx_entry->fi_flags) ||
	     (rx_entry->cq_entry.flags & FI_MULTI_RECV))) {
		EFA_DBG(FI_LOG_CQ,
		       "Writing recv completion for rx_entry from peer: %"
		       PRIu64 " rx_id: %" PRIu32 " msg_id: %" PRIu32
		       " tag: %lx total_len: %" PRIu64 "\n",
		       rx_entry->addr, rx_entry->rx_id, rx_entry->msg_id,
		       rx_entry->cq_entry.tag, rx_entry->total_len);

		rxr_tracepoint(recv_end,
			    rx_entry->msg_id, (size_t) rx_entry->cq_entry.op_context,
			    rx_entry->total_len, rx_entry->cq_entry.tag, rx_entry->addr);


		if (ep->base_ep.util_ep.caps & FI_SOURCE)
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
			EFA_WARN(FI_LOG_CQ,
				"Unable to write recv completion: %s\n",
				fi_strerror(-ret));
			rxr_rx_entry_handle_error(rx_entry, -ret, FI_EFA_ERR_WRITE_RECV_COMP);
			return;
		}

		rx_entry->fi_flags |= RXR_TX_ENTRY_NO_COMPLETION;
	}

	efa_cntr_report_rx_completion(&ep->base_ep.util_ep, rx_entry->cq_entry.flags);
}

/**
 * @brief whether tx_entry should write CQ entry when completed
 *
 * There are two situations that tx_entry should not write CQ entry:
 *
 * 1. there are RXR_NO_COMPLETEION flag in tx_entry->fi_flags, which
 *    is because this tx_entry is for an emulated inject operation
 *
 * 2. user does not want CQ entry for this operation, this behavior
 *    is controled by the FI_SELECTIVE_COMPLETE flag when creating
 *    endpoint. However, this flag is override by the per message
 *    FI_COMPLETION flag.
 *
 * @param[in] 	tx_entry 	information of the TX operation
 * @return 		a boolean
 */
static inline
bool rxr_tx_entry_should_update_cq(struct rxr_op_entry *tx_entry)

{
	if (tx_entry->fi_flags & RXR_TX_ENTRY_NO_COMPLETION)
		return false;

	/*
	 * ep->util_ep.tx_msg_flags is either 0 or FI_COMPLETION, depend on
	 * whether app specfied FI_SELECTIVE_COMPLETION when binding CQ.
	 * (ep->util_ep.tx_msg_flags was set in ofi_ep_bind_cq())
	 *
	 * If tx_msg_flags is 0, we only write completion when app specify
	 * FI_COMPLETION in flags.
	 */
	return tx_entry->ep->base_ep.util_ep.tx_msg_flags == FI_COMPLETION ||
	       tx_entry->fi_flags & FI_COMPLETION;
}

/**
 * @brief report (to user) that a TX operation has completed
 *
 * This function does the following to report completion:
 *
 * 1. write a cq entry for the TX operation when necessary
 *    Sometimes it is not necessary to to write CQ entry
 *    See #rxr_op_entry_should_update_cq
 *
 * 2. update counter if necessary.
 *
 *
 * @param[in]	tx_entry	information of the completed TX operation
 */
void rxr_tx_entry_report_completion(struct rxr_op_entry *tx_entry)
{
	struct util_cq *tx_cq = tx_entry->ep->base_ep.util_ep.tx_cq;
	int ret;

	assert(tx_entry->type == RXR_TX_ENTRY);
	if (rxr_tx_entry_should_update_cq(tx_entry)) {
		EFA_DBG(FI_LOG_CQ,
		       "Writing send completion for tx_entry to peer: %" PRIu64
		       " tx_id: %" PRIu32 " msg_id: %" PRIu32 " tag: %lx len: %"
		       PRIu64 "\n",
		       tx_entry->addr, tx_entry->tx_id, tx_entry->msg_id,
		       tx_entry->cq_entry.tag, tx_entry->total_len);


	rxr_tracepoint(send_end,
		    tx_entry->msg_id, (size_t) tx_entry->cq_entry.op_context,
		    tx_entry->total_len, tx_entry->cq_entry.tag, tx_entry->addr);

		/* TX completions should not send peer address to util_cq */
		if (tx_entry->ep->base_ep.util_ep.caps & FI_SOURCE)
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

		rxr_rm_tx_cq_check(tx_entry->ep, tx_cq);

		if (OFI_UNLIKELY(ret)) {
			EFA_WARN(FI_LOG_CQ,
				"Unable to write send completion: %s\n",
				fi_strerror(-ret));
			rxr_tx_entry_handle_error(tx_entry, -ret, FI_EFA_ERR_WRITE_SEND_COMP);
			return;
		}
	}

	efa_cntr_report_tx_completion(&tx_entry->ep->base_ep.util_ep, tx_entry->cq_entry.flags);
	tx_entry->fi_flags |= RXR_TX_ENTRY_NO_COMPLETION;
	return;
}

/**
 * @brief handle the event that an op_entry has "sent all the data".
 *
 * Note that both tx_entry and rx_entry can send data:
 *
 * tx_entry will send data when operation type is send/write. In this case, the
 * exact timing of "all the data has been sent" event is:
 *
 * 	If the tx_entry requested delivery complete, "all the data has been sent"
 *      happens when tx_entry received a RECEIPT packet from receiver/write responder
 *
 * 	If the tx_entry requested delivery complete, "all the data has been sent"
 *      happens when the send completion of all packets that contains data has been
 *      received.
 *
 * rx_entry will send data when operation type is read and emulated read is used.
 * In this case, "all data has been sent" happens when tx_entry received the completion
 * of all the packets that contains data.
 *
 * In both cases, the "all data has been send" event mark the end of the operation,
 * therefore this function will call rxr_tx/rx_op_entry_report_completion(), and
 * release the op_entry
 *
 * @param[in]	op_entry	inforatminon of op entry that sends data
 */
void rxr_op_entry_handle_send_completed(struct rxr_op_entry *op_entry)
{
	struct rxr_ep *ep;
	struct rxr_op_entry *rx_entry;

	if (op_entry->state == RXR_TX_SEND)
		dlist_remove(&op_entry->entry);

	ep = op_entry->ep;

	if (op_entry->cq_entry.flags & FI_READ) {
		/*
		 * This is on responder side of an emulated read operation.
		 * In this case, we do not write any completion.
		 * The entry is allocated for emulated read, so no need to write tx completion.
		 * EFA does not support FI_RMA_EVENT, so no need to write rx completion.
		 */
		assert(op_entry->type == RXR_RX_ENTRY);
		rx_entry = op_entry;
		rxr_rx_entry_release(rx_entry);
		return;
	} else if (op_entry->cq_entry.flags & FI_WRITE) {
		if (op_entry->fi_flags & FI_COMPLETION) {
			rxr_tx_entry_report_completion(op_entry);
		} else {
			if (!(op_entry->fi_flags & RXR_TX_ENTRY_NO_COUNTER))
				efa_cntr_report_tx_completion(&ep->base_ep.util_ep, op_entry->cq_entry.flags);
		}

	} else {
		assert(op_entry->cq_entry.flags & FI_SEND);
		rxr_tx_entry_report_completion(op_entry);
	}

	assert(op_entry->type == RXR_TX_ENTRY);
	rxr_tx_entry_release(op_entry);
}

/**
 * @brief handle the event "all data has been received" for an op_entry
 *
 * Both tx_entry and rx_entry can receive data.
 *
 * tx_entry can receive data when the operation is read and emulated
 * read is used.
 *
 * rx_entry can receiver data when the operation is receive or write response.
 *
 * To complete a RX operation, this function does 3 things:
 *
 * 1. If necessary, write completion to application. (Not all
 *    completed RX action will cause a completion to be written).
 *
 * 2. If delievery complete is requested send a receipt packet back to the peer.
 *
 * 3. Release the op_entry unless the op_entry is rx_entry, and an RECEIPT/EOR
 *    packet has been sent. This is because rx_entry is needed to handle the
 *    send completion event of RECEIPT/EOR
 *
 * @param[in,out]	op_entry	op_entry that contains information of a data receive operation
 */
void rxr_op_entry_handle_recv_completed(struct rxr_op_entry *op_entry)
{
	struct rxr_op_entry *tx_entry = NULL;
	struct rxr_op_entry *rx_entry = NULL;
	struct efa_rdm_peer *peer;
	bool inject;
	int err;

	/* It is important to write completion before sending ctrl packet, because the
	 * action of sending ctrl packet may cause the release of RX entry (when inject
	 * was used on lower device).
	 */
	if (op_entry->cq_entry.flags & FI_REMOTE_WRITE) {
		/*
		 * For write, only write RX completion when REMOTE_CQ_DATA is on
		 */
		if (op_entry->cq_entry.flags & FI_REMOTE_CQ_DATA)
			rxr_rx_entry_report_completion(op_entry);
	} else if (op_entry->cq_entry.flags & FI_READ) {
		/* This op_entry is part of the for emulated read protocol,
		 * created on the read requester side.
		 * The following shows the sequence of events in an emulated
		 * read protocol.
		 *
		 * Requester                      Responder
		 * create tx_entry
		 * send rtr
		 *                                receive rtr
		 *                                create rx_entry
		 *                                rx_entry sending data
		 * tx_entry receiving data
		 * receive completed              send completed
		 * call rxr_cq_complete_recv()    call rxr_cq_handle_send_completion()
		 *
		 * As can be seen, in the emulated read protocol, this function is called only
		 * on the requester side, so we need to find the corresponding tx_entry and
		 * complete it.
		 */
		assert(op_entry->type == RXR_TX_ENTRY);
		tx_entry = op_entry; /* Intentionally assigned for easier understanding */

		assert(tx_entry->state == RXR_TX_REQ);
		if (tx_entry->fi_flags & FI_COMPLETION) {
			rxr_tx_entry_report_completion(tx_entry);
		} else {
			efa_cntr_report_tx_completion(&tx_entry->ep->base_ep.util_ep, tx_entry->cq_entry.flags);
		}
	} else {
		assert(op_entry->type == RXR_RX_ENTRY);
		rx_entry = op_entry; /* Intentionally assigned for easier understanding */

		assert(rx_entry->op == ofi_op_msg || rx_entry->op == ofi_op_tagged);
		if (rx_entry->fi_flags & FI_MULTI_RECV)
			rxr_msg_multi_recv_handle_completion(rx_entry->ep, rx_entry);

		rxr_rx_entry_report_completion(rx_entry);
		rxr_msg_multi_recv_free_posted_entry(rx_entry->ep, rx_entry);
	}

	/* As can be seen, this function does not release rx_entry when
	 * rxr_pkt_post_or_queue() was successful.
	 *
	 * This is because that rxr_pkt_post_or_queue() might have
	 * queued the ctrl packet (due to out of resource), and progress
	 * engine will resend the packet. In that case, progress engine
	 * needs the rx_entry to construct the ctrl packet.
	 *
	 * Hence, the rx_entry can be safely released only when we got
	 * the send completion of the ctrl packet.
	 *
	 * Another interesting point is that when inject was used, the
	 * rx_entry was released by rxr_pkt_post_or_queue(), because
	 * when inject was used, lower device will not provider send
	 * completion for the ctrl packet.
	 */
	if (op_entry->rxr_flags & RXR_TX_ENTRY_DELIVERY_COMPLETE_REQUESTED) {
		assert(op_entry->type == RXR_RX_ENTRY);
		rx_entry = op_entry; /* Intentionally assigned for easier understanding */
		peer = rxr_ep_get_peer(rx_entry->ep, rx_entry->addr);
		assert(peer);
		inject = peer->is_local && rx_entry->ep->use_shm_for_tx;
		err = rxr_pkt_post_or_queue(rx_entry->ep, rx_entry, RXR_RECEIPT_PKT, inject);
		if (OFI_UNLIKELY(err)) {
			EFA_WARN(FI_LOG_CQ,
				 "Posting of ctrl packet failed when complete rx! err=%s(%d)\n",
				 fi_strerror(-err), -err);
			rxr_rx_entry_handle_error(rx_entry, -err, FI_EFA_ERR_PKT_POST);
			rxr_rx_entry_release(rx_entry);
		}

		return;
	}

	/* An EOR (End Of Read) packet is sent when RDMA read operation has completed
	 * for read data. EOR was sent before "all data has been received". Therefore
	 * it is possible that when this function is called, EOR is still inflight
	 * (EOR has been sent, and the send completion has NOT been received).
	 *
	 * If EOR is inflight, the RX_entry cannot be released because the rx_entry
	 * is needed to handle the send completion of the EOR.
	 *
	 * see #rxr_pkt_handle_eor_send_completion
	 */
	if (op_entry->rxr_flags & RXR_RX_ENTRY_EOR_IN_FLIGHT) {
		return;
	}

	if (op_entry->type == RXR_TX_ENTRY) {
		rxr_tx_entry_release(op_entry);
	} else {
		assert(op_entry->type == RXR_RX_ENTRY);
		rxr_rx_entry_release(op_entry);
	}
}

/**
 * @brief prepare an op_entry such that it is ready to post read request(s)
 *
 * An op_entry can post read request(s) for two reasons:
 *
 * First, it can be because user directly initiated a read requst
 * (by calling fi_readxxx() API). In this case the op_entry argument
 * will be a tx_entry (op_entry->type == RXR_TX_ENTRY)
 *
 * Second, it can be part of a read-base message protocol, such as
 * the longread message protocol. In this case, the op_entry argument
 * will be a rx_entry (op_entry->type == RXR_RX_ENTRY)
 *
 * @param[in,out]		ep		endpoint
 * @param[in,out]		op_entry	information of the operation the needs to post a read
 * @return		0 if the read request is posted successfully.
 * 			negative libfabric error code on failure.
 */
int rxr_op_entry_prepare_to_post_read(struct rxr_op_entry *op_entry)
{
	int err;
	size_t total_iov_len, total_rma_iov_len;

	if (op_entry->type == RXR_RX_ENTRY) {
		/* Often times, application will provide a receiving buffer that is larger
		 * then the incoming message size. For read based message transfer, the
		 * receiving buffer need to be registered. Thus truncating rx_entry->iov to
		 * extact message size to save memory registration pages.
		 */
		err = ofi_truncate_iov(op_entry->iov, &op_entry->iov_count,
				       op_entry->total_len + op_entry->ep->msg_prefix_size);
		if (err) {
			EFA_WARN(FI_LOG_CQ,
				 "ofi_truncated_iov failed. new_size: %ld\n",
				 op_entry->total_len + op_entry->ep->msg_prefix_size);
			return err;
		}
	}

	total_iov_len = ofi_total_iov_len(op_entry->iov, op_entry->iov_count);
	total_rma_iov_len = ofi_total_rma_iov_len(op_entry->rma_iov, op_entry->rma_iov_count);

	if (op_entry->type == RXR_TX_ENTRY)
		op_entry->bytes_read_offset = 0;
	else
		op_entry->bytes_read_offset = op_entry->bytes_runt;

	op_entry->bytes_read_total_len = MIN(total_iov_len, total_rma_iov_len) - op_entry->bytes_read_offset;
	op_entry->bytes_read_submitted = 0;
	op_entry->bytes_read_completed = 0;
	return 0;
}

/**
 * @brief prepare an op_entry such that it is ready to post write request(s)
 *
 * An op_entry will post write request(s) after a user directly initiated a
 * write request (by calling fi_writexxx() API).  The op_entry->type will be
 * RXR_TX_ENTRY
 *
 * @param[in,out]	ep		endpoint
 * @param[in,out]	op_entry	information the operation needs to post a write
 */
void rxr_op_entry_prepare_to_post_write(struct rxr_op_entry *op_entry)
{
	size_t local_iov_len;

	assert(op_entry->type == RXR_TX_ENTRY);

	local_iov_len = ofi_total_iov_len(op_entry->iov, op_entry->iov_count);
#ifndef NDEBUG
	{
		size_t remote_iov_len;
		remote_iov_len = ofi_total_rma_iov_len(op_entry->rma_iov, op_entry->rma_iov_count);
		assert( local_iov_len == remote_iov_len );
	}
#endif

	op_entry->bytes_write_total_len = local_iov_len;
	op_entry->bytes_write_submitted = 0;
	op_entry->bytes_write_completed = 0;
}

/**
 * @brief post read request(s)
 *
 * This function posts read request(s) according to information in op_entry.
 * Depend on op_entry->bytes_read_total_len and max read size of device. This function
 * might issue multiple read requdsts.
 *
 * @param[in,out]	op_entry	op_entry that has information of the read request.
 * 					If read request is successfully submitted,
 * 					op_entry->bytes_read_submitted will be updated.
 * @return	On success, return 0
 * 		On failure, return a negative error code.
 */
int rxr_op_entry_post_remote_read(struct rxr_op_entry *op_entry)
{
	int err;
	int iov_idx = 0, rma_iov_idx = 0;
	size_t iov_offset = 0, rma_iov_offset = 0;
	size_t read_once_len, max_read_once_len;
	bool use_shm;
	struct rxr_ep *ep;
	struct efa_rdm_peer *peer;
	struct rxr_pkt_entry *pkt_entry;

	assert(op_entry->iov_count > 0);
	assert(op_entry->rma_iov_count > 0);

	ep = op_entry->ep;
	peer = rxr_ep_get_peer(ep, op_entry->addr);
	use_shm = peer->is_local && ep->use_shm_for_tx;

	if (op_entry->bytes_read_total_len == 0) {

		/* According to libfabric document
		 *     https://ofiwg.github.io/libfabric/main/man/fi_rma.3.html
		 * read with 0 byte is allowed.
		 *
		 * Note that because send operation used a pkt_entry as wr_id,
		 * we had to use a pkt_entry as context for read too.
		 */
		if (use_shm)
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->shm_tx_pkt_pool, RXR_PKT_FROM_SHM_TX_POOL);
		else
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->efa_tx_pkt_pool, RXR_PKT_FROM_EFA_TX_POOL);

		if (OFI_UNLIKELY(!pkt_entry))
			return -FI_EAGAIN;

		rxr_pkt_init_read_context(ep, op_entry, op_entry->addr, ofi_buf_index(op_entry), 0, pkt_entry);
		err = rxr_pkt_entry_read(ep, pkt_entry,
					 op_entry->iov[0].iov_base,
					 0,
					 op_entry->desc[0],
					 op_entry->rma_iov[0].addr,
					 op_entry->rma_iov[0].key);
		if (err)
			rxr_pkt_entry_release_tx(ep, pkt_entry);
		return err;
	}

	assert(op_entry->bytes_read_submitted < op_entry->bytes_read_total_len);

	if (!use_shm) {
		rxr_op_entry_try_fill_desc(op_entry, 0, FI_RECV);
	}

	max_read_once_len = use_shm ? SIZE_MAX : MIN(rxr_env.efa_read_segment_size, rxr_ep_domain(ep)->device->max_rdma_size);
	assert(max_read_once_len > 0);

	err = rxr_locate_iov_pos(op_entry->iov, op_entry->iov_count,
				 op_entry->bytes_read_offset + op_entry->bytes_read_submitted + ep->msg_prefix_size,
				 &iov_idx, &iov_offset);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "rxr_locate_iov_pos failed! err: %d\n", err);
		return err;
	}

	err = rxr_locate_rma_iov_pos(op_entry->rma_iov, op_entry->rma_iov_count,
				     op_entry->bytes_read_offset + op_entry->bytes_read_submitted,
				     &rma_iov_idx, &rma_iov_offset);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "rxr_locate_rma_iov_pos failed! err: %d\n", err);
		return err;
	}

	while (op_entry->bytes_read_submitted < op_entry->bytes_read_total_len) {

		assert(iov_idx < op_entry->iov_count);
		assert(iov_offset < op_entry->iov[iov_idx].iov_len);
		assert(rma_iov_idx < op_entry->rma_iov_count);
		assert(rma_iov_offset < op_entry->rma_iov[rma_iov_idx].len);

		if (!use_shm) {
			if (ep->efa_outstanding_tx_ops == ep->efa_max_outstanding_tx_ops)
				return -FI_EAGAIN;

			if (!op_entry->desc[iov_idx]) {
				/* rxr_op_entry_try_fill_desc() did not fill the desc,
				 * which means memory registration failed.
				 * return -FI_EAGAIN here will cause user to run progress
				 * engine, which will cause some memory registration
				 * in MR cache to be released.
				 */
				return -FI_EAGAIN;
			}
		}

		if (use_shm)
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->shm_tx_pkt_pool, RXR_PKT_FROM_SHM_TX_POOL);
		else
			pkt_entry = rxr_pkt_entry_alloc(ep, ep->efa_tx_pkt_pool, RXR_PKT_FROM_EFA_TX_POOL);

		if (OFI_UNLIKELY(!pkt_entry))
			return -FI_EAGAIN;

		read_once_len = MIN(op_entry->iov[iov_idx].iov_len - iov_offset,
				    op_entry->rma_iov[rma_iov_idx].len - rma_iov_offset);
		read_once_len = MIN(read_once_len, max_read_once_len);

		rxr_pkt_init_read_context(ep, op_entry, op_entry->addr, ofi_buf_index(op_entry), read_once_len, pkt_entry);
		err = rxr_pkt_entry_read(ep, pkt_entry,
					 (char *)op_entry->iov[iov_idx].iov_base + iov_offset,
					 read_once_len,
					 op_entry->desc[iov_idx],
					 op_entry->rma_iov[rma_iov_idx].addr + rma_iov_offset,
					 op_entry->rma_iov[rma_iov_idx].key);
		if (err) {
			EFA_WARN(FI_LOG_CQ, "rxr_pkt_entry_read failed! err: %d\n", err);
			rxr_pkt_entry_release_tx(ep, pkt_entry);
			return err;
		}

		op_entry->bytes_read_submitted += read_once_len;

		iov_offset += read_once_len;
		assert(iov_offset <= op_entry->iov[iov_idx].iov_len);
		if (iov_offset == op_entry->iov[iov_idx].iov_len) {
			iov_idx += 1;
			iov_offset = 0;
		}

		rma_iov_offset += read_once_len;
		assert(rma_iov_offset <= op_entry->rma_iov[rma_iov_idx].len);
		if (rma_iov_offset == op_entry->rma_iov[rma_iov_idx].len) {
			rma_iov_idx += 1;
			rma_iov_offset = 0;
		}
	}

	return 0;
}

/**
 * @brief post RDMA write request(s)
 *
 * This function posts write request(s) according to information in op_entry.
 * Depending on op_entry->bytes_write_total_len and max write size of device,
 * this function might issue multiple write requests.
 *
 * @param[in,out]	op_entry	op_entry that has information of the read request.
 * 					If write request is successfully submitted,
 * 					op_entry->bytes_write_submitted will be updated.
 * @return	On success, return 0
 * 		On failure, return a negative error code.
 */
int rxr_op_entry_post_remote_write(struct rxr_op_entry *op_entry)
{
	int err;
	int iov_idx = 0, rma_iov_idx = 0;
	size_t iov_offset = 0, rma_iov_offset = 0;
	size_t write_once_len, max_write_once_len;
	struct rxr_ep *ep;
	struct rxr_pkt_entry *pkt_entry;

	assert(op_entry->iov_count > 0);
	assert(op_entry->rma_iov_count > 0);
	assert(op_entry->bytes_write_submitted < op_entry->bytes_write_total_len);

	rxr_op_entry_try_fill_desc(op_entry, 0, FI_WRITE);
	ep = op_entry->ep;
	max_write_once_len = MIN(rxr_env.efa_write_segment_size, rxr_ep_domain(ep)->device->max_rdma_size);

	assert(max_write_once_len > 0);

	err = rxr_locate_iov_pos(op_entry->iov, op_entry->iov_count,
				 op_entry->bytes_write_submitted,
				 &iov_idx, &iov_offset);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "rxr_locate_iov_pos failed! err: %d\n", err);
		return err;
	}

	err = rxr_locate_rma_iov_pos(op_entry->rma_iov, op_entry->rma_iov_count,
				     op_entry->bytes_write_submitted,
				     &rma_iov_idx, &rma_iov_offset);
	if (OFI_UNLIKELY(err)) {
		EFA_WARN(FI_LOG_CQ, "rxr_locate_rma_iov_pos failed! err: %d\n", err);
		return err;
	}

	while (op_entry->bytes_write_submitted < op_entry->bytes_write_total_len) {

		assert(iov_idx < op_entry->iov_count);
		assert(iov_offset < op_entry->iov[iov_idx].iov_len);
		assert(rma_iov_idx < op_entry->rma_iov_count);
		assert(rma_iov_offset < op_entry->rma_iov[rma_iov_idx].len);

		if (ep->efa_outstanding_tx_ops == ep->efa_max_outstanding_tx_ops)
			return -FI_EAGAIN;

		if (!op_entry->desc[iov_idx]) {
			/* rxr_op_entry_try_fill_desc() did not fill the desc,
			 * which means memory registration failed.
			 * return -FI_EAGAIN here will cause user to run progress
			 * engine, which will cause some memory registration
			 * in MR cache to be released.
			 */
			return -FI_EAGAIN;
		}
		pkt_entry = rxr_pkt_entry_alloc(ep, ep->efa_tx_pkt_pool, RXR_PKT_FROM_EFA_TX_POOL);

		if (OFI_UNLIKELY(!pkt_entry))
			return -FI_EAGAIN;

		write_once_len = MIN(op_entry->iov[iov_idx].iov_len - iov_offset,
				    op_entry->rma_iov[rma_iov_idx].len - rma_iov_offset);
		write_once_len = MIN(write_once_len, max_write_once_len);

		rxr_pkt_init_write_context(op_entry, pkt_entry);
		err = rxr_pkt_entry_write(ep, pkt_entry,
					 (char *)op_entry->iov[iov_idx].iov_base + iov_offset,
					 write_once_len,
					 op_entry->desc[iov_idx],
					 op_entry->rma_iov[rma_iov_idx].addr + rma_iov_offset,
					 op_entry->rma_iov[rma_iov_idx].key);
		if (err) {
			EFA_WARN(FI_LOG_CQ, "rxr_pkt_entry_write failed! err: %d\n", err);
			rxr_pkt_entry_release_tx(ep, pkt_entry);
			return err;
		}

		op_entry->bytes_write_submitted += write_once_len;

		iov_offset += write_once_len;
		assert(iov_offset <= op_entry->iov[iov_idx].iov_len);
		if (iov_offset == op_entry->iov[iov_idx].iov_len) {
			iov_idx += 1;
			iov_offset = 0;
		}

		rma_iov_offset += write_once_len;
		assert(rma_iov_offset <= op_entry->rma_iov[rma_iov_idx].len);
		if (rma_iov_offset == op_entry->rma_iov[rma_iov_idx].len) {
			rma_iov_idx += 1;
			rma_iov_offset = 0;
		}
	}

	return 0;
}

int rxr_op_entry_post_remote_read_or_queue(struct rxr_op_entry *op_entry)
{
	int err;

	err = rxr_op_entry_prepare_to_post_read(op_entry);
	if (err) {
		EFA_WARN(FI_LOG_CQ, "Prepare to post read failed. err=%d\n", err);
		return err;
	}

	err = rxr_op_entry_post_remote_read(op_entry);
	if (err == -FI_EAGAIN) {
		dlist_insert_tail(&op_entry->queued_read_entry, &op_entry->ep->op_entry_queued_read_list);
		op_entry->rxr_flags |= RXR_OP_ENTRY_QUEUED_READ;
		err = 0;
	} else if(err) {
		EFA_WARN(FI_LOG_CQ,
			"RDMA post read failed. errno=%d.\n", err);
	}

	return err;
}
