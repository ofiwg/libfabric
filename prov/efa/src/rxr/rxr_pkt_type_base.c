/*
 * Copyright (c) 2021 Amazon.com, Inc. or its affiliates.
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
#include "rxr_read.h"
#include "rxr_pkt_cmd.h"

/**
 * @brief return the optional connid header pointer in a packet
 *
 * @param[in]	pkt_entry	an packet entry
 * @return	If the input has the optional connid header, return the pointer to connid header
 * 		Otherwise, return NULL
 */
uint32_t *rxr_pkt_connid_ptr(struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);

	if (base_hdr->type >= RXR_REQ_PKT_BEGIN)
		return rxr_pkt_req_connid_ptr(pkt_entry);

	if (!(base_hdr->flags & RXR_PKT_CONNID_HDR))
		return NULL;

	switch (base_hdr->type) {
	case RXR_CTS_PKT:
		return &(rxr_get_cts_hdr(pkt_entry->pkt)->connid);

	case RXR_RECEIPT_PKT:
		return &(rxr_get_receipt_hdr(pkt_entry->pkt)->connid);

	case RXR_DATA_PKT:
		return &(rxr_get_data_hdr(pkt_entry->pkt)->connid_hdr->connid);

	case RXR_READRSP_PKT:
		return &(rxr_get_readrsp_hdr(pkt_entry->pkt)->connid);

	case RXR_ATOMRSP_PKT:
		return &(rxr_get_atomrsp_hdr(pkt_entry->pkt)->connid);

	case RXR_EOR_PKT:
		return &rxr_get_eor_hdr(pkt_entry->pkt)->connid;

	case RXR_HANDSHAKE_PKT:
		return &(rxr_get_handshake_opt_connid_hdr(pkt_entry->pkt)->connid);

	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "unknown packet type: %d\n", base_hdr->type);
		assert(0 && "Unknown packet type");
	}

	return NULL;
}

/**
 * @brief set up data in a packet entry using tx_entry information, such that the packet is ready to be sent.
 *        Depend on the tx_entry, this function can either copy data to packet entry, or point
 *        pkt_entry->iov to tx_entry->iov.
 *        It requires the packet header to be set.
 *
 * @param[in]		ep		end point.
 * @param[in,out]	pkt_entry	packet entry. Header must have been set when the function is called
 * @param[in]		hdr_size	packet header size.
 * @param[in]		tx_entry	This function will use iov, iov_count and desc of tx_entry
 * @param[in]		data_offset	offset of the data to be set up. In reference to tx_entry->total_len.
 * @param[in]		data_size	length of the data to be set up. In reference to tx_entry->total_len.
 * @return		0 on success, negative FI code on error
 */
int rxr_pkt_init_data_from_tx_entry(struct rxr_ep *ep,
				    struct rxr_pkt_entry *pkt_entry,
				    size_t hdr_size,
				    struct rxr_tx_entry *tx_entry,
				    size_t data_offset,
				    size_t data_size)
{
	struct efa_ep *efa_ep;
	int tx_iov_index;
	char *data;
	size_t tx_iov_offset, copied;
	struct efa_mr *desc;
	int ret;

	efa_ep = container_of(ep->rdm_ep, struct efa_ep, util_ep.ep_fid);

	assert(hdr_size > 0);

	pkt_entry->x_entry = tx_entry;
	/* pkt_sendv_pool's size equal efa_tx_pkt_pool size +
	 * shm_tx_pkt_pool size. As long as we have a pkt_entry,
	 * pkt_entry->send should be allocated successfully
	 */
	pkt_entry->send = ofi_buf_alloc(ep->pkt_sendv_pool);
	if (!pkt_entry->send) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "allocate pkt_entry->send failed\n");
		assert(pkt_entry->send);
		return -FI_ENOMEM;
	}

	if (data_size == 0) {
		pkt_entry->send->iov_count = 0;
		pkt_entry->pkt_size = hdr_size;
		return 0;
	}

	rxr_locate_iov_pos(tx_entry->iov, tx_entry->iov_count, data_offset,
			   &tx_iov_index, &tx_iov_offset);
	desc = tx_entry->desc[0];
	assert(tx_iov_index < tx_entry->iov_count);
	assert(tx_iov_offset < tx_entry->iov[tx_iov_index].iov_len);

	ret = efa_ep_use_p2p(efa_ep, desc);
	if (ret < 0) {
		ofi_buf_free(pkt_entry->send);
		return -FI_ENOSYS;
	}
	if (ret == 0)
		goto copy;

	/*
	 * Copy can be avoid if the following 2 conditions are true:
	 * 1. user provided memory descriptor, or message is sent via shm provider
	 *    (which does not require a memory descriptor)
	 * 2. data to be send is in 1 iov, because device only support 2 iov, and we use
	 *    1st iov for header.
	 */
	if ((!pkt_entry->mr || tx_entry->desc[tx_iov_index]) &&
	    (tx_iov_offset + data_size <= tx_entry->iov[tx_iov_index].iov_len)) {

		assert(ep->core_iov_limit >= 2);
		pkt_entry->send->iov[0].iov_base = pkt_entry->pkt;
		pkt_entry->send->iov[0].iov_len = hdr_size;
		pkt_entry->send->desc[0] = pkt_entry->mr ? fi_mr_desc(pkt_entry->mr) : NULL;

		pkt_entry->send->iov[1].iov_base = (char *)tx_entry->iov[tx_iov_index].iov_base + tx_iov_offset;
		pkt_entry->send->iov[1].iov_len = data_size;
		pkt_entry->send->desc[1] = tx_entry->desc[tx_iov_index];
		pkt_entry->send->iov_count = 2;
		pkt_entry->pkt_size = hdr_size + data_size;
		return 0;
	}

copy:
	data = pkt_entry->pkt + hdr_size;
	copied = ofi_copy_from_hmem_iov(data,
					data_size,
					desc ? desc->peer.iface : FI_HMEM_SYSTEM,
					desc ? desc->peer.device.reserved : 0,
					tx_entry->iov,
					tx_entry->iov_count,
					data_offset);
	assert(copied == data_size);
	pkt_entry->send->iov_count = 0;
	pkt_entry->pkt_size = hdr_size + copied;
	return 0;
}

/* @brief return the data size in a packet entry
 *
 * @param[in]	pkt_entry		packet entry
 * @return	the data size in the packet entry.
 * 		if the packet entry does not contain data,
 * 		return 0.
 */
size_t rxr_pkt_data_size(struct rxr_pkt_entry *pkt_entry)
{
	int pkt_type;

	assert(pkt_entry);
	pkt_type = rxr_get_base_hdr(pkt_entry->pkt)->type;

	if (pkt_type == RXR_DATA_PKT)
		return rxr_get_data_hdr(pkt_entry->pkt)->seg_length;

	if (pkt_type == RXR_READRSP_PKT)
		return rxr_get_readrsp_hdr(pkt_entry->pkt)->seg_length;

	if (pkt_type >= RXR_REQ_PKT_BEGIN) {
		assert(pkt_type == RXR_EAGER_MSGRTM_PKT || pkt_type == RXR_EAGER_TAGRTM_PKT ||
		       pkt_type == RXR_MEDIUM_MSGRTM_PKT || pkt_type == RXR_MEDIUM_TAGRTM_PKT ||
		       pkt_type == RXR_LONGCTS_MSGRTM_PKT || pkt_type == RXR_LONGCTS_TAGRTM_PKT ||
		       pkt_type == RXR_EAGER_RTW_PKT ||
		       pkt_type == RXR_LONGCTS_RTW_PKT ||
		       pkt_type == RXR_DC_EAGER_MSGRTM_PKT ||
		       pkt_type == RXR_DC_EAGER_TAGRTM_PKT ||
		       pkt_type == RXR_DC_MEDIUM_MSGRTM_PKT ||
		       pkt_type == RXR_DC_MEDIUM_TAGRTM_PKT ||
		       pkt_type == RXR_DC_LONGCTS_MSGRTM_PKT ||
		       pkt_type == RXR_DC_LONGCTS_TAGRTM_PKT ||
		       pkt_type == RXR_DC_EAGER_RTW_PKT ||
		       pkt_type == RXR_DC_LONGCTS_RTW_PKT);

		return pkt_entry->pkt_size - rxr_pkt_req_hdr_size(pkt_entry);
	}

	/* other packet type does not contain data, thus return 0
	 */
	return 0;
}


/*
 * @brief copy data to hmem buffer
 *
 * This function queue multiple (up to RXR_EP_MAX_QUEUED_COPY) copies to
 * device memory, and do them at the same time. This is to avoid any memory
 * barrier between copies, which will cause a flush.
 *
 * @param[in]		ep		endpoint
 * @param[in]		pkt_entry	the packet entry that contains data, which
 *                                      x_entry pointing to the correct rx_entry.
 * @param[in]		data		the pointer pointing to the beginning of data
 * @param[in]		data_size	the length of data
 * @param[in]		data_offset	the offset of the data in the packet in respect
 *					of the receiving buffer.
 * @return		On success, return 0
 * 			On failure, return libfabric error code
 */
static inline
int rxr_pkt_copy_data_to_hmem(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry,
			      char *data,
			      size_t data_size,
			      size_t data_offset)
{
	struct efa_mr *desc;
	struct rxr_rx_entry *rx_entry;
	size_t bytes_copied[RXR_EP_MAX_QUEUED_COPY] = {0};
	size_t i;

	assert(ep->queued_copy_num < RXR_EP_MAX_QUEUED_COPY);
	ep->queued_copy_vec[ep->queued_copy_num].pkt_entry = pkt_entry;
	ep->queued_copy_vec[ep->queued_copy_num].data = data;
	ep->queued_copy_vec[ep->queued_copy_num].data_size = data_size;
	ep->queued_copy_vec[ep->queued_copy_num].data_offset = data_offset;
	ep->queued_copy_num += 1;

	rx_entry = pkt_entry->x_entry;
	assert(rx_entry);
	rx_entry->bytes_queued += data_size;

	if (ep->queued_copy_num < RXR_EP_MAX_QUEUED_COPY &&
	    rx_entry->bytes_copied + rx_entry->bytes_queued < rx_entry->total_len) {
		return 0;
	}

	for (i = 0; i < ep->queued_copy_num; ++i) {
		pkt_entry = ep->queued_copy_vec[i].pkt_entry;
		data = ep->queued_copy_vec[i].data;
		data_size = ep->queued_copy_vec[i].data_size;
		data_offset = ep->queued_copy_vec[i].data_offset;

		rx_entry = pkt_entry->x_entry;
		desc = rx_entry->desc[0];
		assert(desc && desc->peer.iface != FI_HMEM_SYSTEM);
		bytes_copied[i] = ofi_copy_to_hmem_iov(desc->peer.iface, desc->peer.device.reserved,
						       rx_entry->iov, rx_entry->iov_count,
						       data_offset + ep->msg_prefix_size,
						       data, data_size);
	}

	for (i = 0; i < ep->queued_copy_num; ++i) {
		pkt_entry = ep->queued_copy_vec[i].pkt_entry;
		data_size = ep->queued_copy_vec[i].data_size;
		data_offset = ep->queued_copy_vec[i].data_offset;
		rx_entry = pkt_entry->x_entry;

		if (bytes_copied[i] != MIN(data_size, rx_entry->cq_entry.len - data_offset)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ, "wrong size! bytes_copied: %ld\n",
				bytes_copied[i]);
			return -FI_EIO;
		}

		rx_entry->bytes_queued -= data_size;
		rxr_pkt_handle_data_copied(ep, pkt_entry, data_size);
	}

	ep->queued_copy_num = 0;
	return 0;
}

/**
 * @brief copy data to application's receive buffer and update counter in rx_entry.
 *
 * Depend on when application's receive buffer is located (host or device) and
 * the software stack, this function will select different, strategies to copy data.
 *
 * When application's receive buffer is on device, there are two scenarios:
 *
 *    If memory is on cuda GPU, and gdrcopy is not available, this function
 *    will post a local read request to copy data. (This is because NCCL forbids its
 *    plugin to make cuda calls). In this case, the data is not copied upon return of
 *    this function, and the function rxr_pkt_handle_copied() is not called. It will
 *    be called upon the completion of local read operation by the progress engine.
 *
 *    Otherwise, this function calls rxr_pkt_copy_data_to_hmem(), which will batch
 *    multiple copies, and perform the copy (then call rxr_pkt_handle_copied()) together
 *    to improve performance.
 *
 * When application's receive buffer is on host, data is copied immediately, and
 * rxr_pkt_handle_copied() is called.
 *
 * @param[in]		ep		endpoint
 * @param[in,out]	rx_entry	rx_entry contains information of the receive
 *                      	        op. This function uses receive buffer in it.
 * @param[in]		data_offset	the offset of the data in the packet in respect
 *					of the receiving buffer.
 * @param[in]		pkt_entry	the packet entry that contains data
 * @param[in]		data		the pointer pointing to the beginning of data
 * @param[in]		data_size	the length of data
 * @return		On success, return 0
 * 			On failure, return libfabric error code
 */
ssize_t rxr_pkt_copy_data_to_rx_entry(struct rxr_ep *ep,
				      struct rxr_rx_entry *rx_entry,
				      size_t data_offset,
				      struct rxr_pkt_entry *pkt_entry,
				      char *data, size_t data_size)
{
	struct efa_ep *efa_ep;
	struct efa_mr *desc;
	ssize_t bytes_copied;
	int use_p2p, err;

	pkt_entry->x_entry = rx_entry;

	/*
	 * Under 3 rare situations, this function does not perform the copy
	 * action, but still consider data is copied:
	 *
	 * 1. application cancelled the receive, thus the receive buffer is not
	 *    available for copying to. In the case, this function is still
	 *    called because sender will keep sending data as receiver did not
	 *    notify the sender about the cancelation,
	 *
	 * 2. application's receiving buffer is smaller than incoming message size,
	 *    and data in the packet is outside of receiving buffer (truncated message).
	 *    In this case, this function is still called because sender will
	 *    keep sending data as receiver did not notify the sender about the
	 *    truncation.
	 *
	 * 3. message size is 0, thus no data to copy.
	 */
	if (OFI_UNLIKELY((rx_entry->rxr_flags & RXR_RECV_CANCEL)) ||
	    OFI_UNLIKELY(data_offset >= rx_entry->cq_entry.len) ||
	    OFI_UNLIKELY(data_size == 0)) {
		rxr_pkt_handle_data_copied(ep, pkt_entry, data_size);
		return 0;
	}

	desc = rx_entry->desc[0];

	/* local read is used to copy data from receiving bounce buffer (on host memory)
	 * to device memory.
	 *
	 * It is an expensive operation thus should be used when CUDA memory is used
	 * and gdrcopy is not available.
	 * 
	 * CUDA memory is special because Nvidia Collective Communications Library (NCCL) forbids
	 * its plugins (hence libfabric) to call cudaMemcpy. Doing so will result in a deadlock.
	 *
	 * Therefore, if gdrcopy is not available, we will have to use local read to do the
	 * copy.
	 * Other types of device memory (neuron) does not have this limitation.
	 */
	if (efa_ep_is_cuda_mr(desc) && !cuda_is_gdrcopy_enabled()) {
		efa_ep = container_of(ep->rdm_ep, struct efa_ep, util_ep.ep_fid);
		use_p2p = efa_ep_use_p2p(efa_ep, desc);
		if (use_p2p < 0)
			return use_p2p;

		if (use_p2p == 0) {
			FI_WARN(&rxr_prov, FI_LOG_CQ, "Neither p2p nor gdrcopy is available,"
				"thus libfabric is not able to copy received data to Nvidia GPU");
			return -FI_EINVAL;
		}

		err = rxr_read_post_local_read_or_queue(ep, rx_entry, data_offset,
							pkt_entry, data, data_size);
		if (err)
			FI_WARN(&rxr_prov, FI_LOG_CQ, "cannot post read to copy data\n");

		/* At this point data has NOT been copied yet, thus we cannot call
		 * rxr_pkt_handle_data_copied(). The function will be called
		 * when the completion of the local read request is received
		 * (by progress engine).
		 */
		return err;
	}

	if (efa_ep_is_hmem_mr(desc))
		return rxr_pkt_copy_data_to_hmem(ep, pkt_entry, data, data_size, data_offset);

	assert( !desc || desc->peer.iface == FI_HMEM_SYSTEM);
	bytes_copied = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
				       data_offset + ep->msg_prefix_size,
				       data, data_size);

	if (bytes_copied != MIN(data_size, rx_entry->cq_entry.len - data_offset)) {
		FI_WARN(&rxr_prov, FI_LOG_CQ, "wrong size! bytes_copied: %ld\n",
			bytes_copied);
		return -FI_EIO;
	}

	rxr_pkt_handle_data_copied(ep, pkt_entry, data_size);
	return 0;
}
