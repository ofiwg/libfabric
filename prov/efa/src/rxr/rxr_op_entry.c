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

/**
 * @brief try to fill the desc field of a tx_entry
 *
 * The desc field of tx_entry contains the memory descriptors of
 * the user's data buffer.
 *
 * For EFA provider, a data buffer's memory descriptor is a pointer to an
 * efa_mr object, which contains the memory registration information
 * of the data buffer.
 *
 * EFA provider does not require user to provide a descriptor, when
 * user's data buffer is on host memory (Though user can register
 * its buffer, and provide its descriptor as an optimization).
 *
 * The EFA device requires send buffer to be registered.
 *
 * For a user that did not provide descriptors for the buffer,
 * EFA provider need to bridge the gap. It has 2 solutions for
 * this issue:
 *
 * First, EFA provider can copy the user data to a pre-registered bounce
 * buffer, then send data from bounce buffer.
 *
 * Second, EFA provider can register the user's buffer and fill tx_entry->desc
 * (by calling this function). then send directly from send buffer.
 *
 * Because of the high cost of memory registration, this function
 * check the availibity of MR cache, and only register memory
 * when MR cache is available.
 *
 * Also memory registration may fail due to limited resources, in which
 * case tx_entry->desc will not be filled either.
 *
 * Because this function is not guaranteed to fill tx_entry->desc,
 * it is used by protocols that does not rely on memory registration
 * such as the medium message protocol and long-cts protocol. These
 * protocol check tx_entry->desc, and when tx_entry->desc is not set,
 * they use bounce buffer.
 *
 * Among other protocols, eager protocol will not register memory,
 * so do not call this function. Read base protocol rely on memory
 * registration, hence cannot use function for memory registration either.
 *
 * @param[in,out]	tx_entry	contains the inforation of a TX operation
 * @param[in]		efa_domain	where memory regstration function operates from
 * @param[in]		mr_iov_start	the IOV index to start generating descriptors
 * @param[in]		access		the access flag for the memory registation.
 *
 */
void rxr_tx_entry_try_fill_desc(struct rxr_tx_entry *tx_entry,
				struct efa_domain *efa_domain,
				int mr_iov_start, uint64_t access)
{
	int i, err;

	if (!efa_is_cache_available(efa_domain))
		return;

	for (i = mr_iov_start; i < tx_entry->iov_count; ++i) {
		if (tx_entry->desc[i])
			continue;

		if (tx_entry->iov[i].iov_len <= rxr_env.max_memcpy_size) {
			assert(!tx_entry->mr[i]);
			continue;
		}

		err = fi_mr_reg(&efa_domain->util_domain.domain_fid,
				tx_entry->iov[i].iov_base,
				tx_entry->iov[i].iov_len,
				access, 0, 0, 0,
				&tx_entry->mr[i], NULL);
		if (err) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"fi_mr_reg failed! buf: %p len: %ld access: %lx",
				tx_entry->iov[i].iov_base, tx_entry->iov[i].iov_len,
				access);

			tx_entry->mr[i] = NULL;
		} else {
			tx_entry->desc[i] = fi_mr_desc(tx_entry->mr[i]);
		}
	}
}

/**
 * @brief calculate and set the bytes_runt field of a tx_entry
 *
 * bytes_runt is number of bytes for a message to be sent by runting
 *
 * @param		ep[in]			endpoint
 * @param		tx_entry[in,out]	tx_entry to be set
 */
void rxr_tx_entry_set_runt_size(struct rxr_ep *ep, struct rxr_op_entry *tx_entry)
{
	struct rdm_peer *peer;

	assert(tx_entry->type == RXR_TX_ENTRY);

	if (tx_entry->bytes_runt > 0)
		return;

	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);
	tx_entry->bytes_runt = MIN(rxr_env.efa_runt_size - peer->num_runt_bytes_in_flight, tx_entry->total_len);
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
 * param[in]		pkt_type		REQ packet type
 * param[in]		op_entry		contains operation information
 * return		size of total data transfered by REQ packets
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
 * @param[in]		ep		endpoint
 * @param[in]		tx_entry	tx_entry that has all information of
 * 					a send operation
 * @param[in]		pkt_type	type of REQ packet
 *
 * @return		maxiumum number of bytes of data can be save in a REQ packet
 * 			for given send operation and REQ packet type.
 */
size_t rxr_tx_entry_max_req_data_capacity(struct rxr_ep *ep, struct rxr_op_entry *tx_entry, int pkt_type)
{
	struct rdm_peer *peer;
	uint16_t header_flags = 0;

	assert(pkt_type >= RXR_REQ_PKT_BEGIN);

	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);

	if (peer->is_local && ep->use_shm_for_tx) {
		return rxr_env.shm_max_medium_size;
	}

	if (rxr_peer_need_raw_addr_hdr(peer))
		header_flags |= RXR_REQ_OPT_RAW_ADDR_HDR;
	else if (rxr_peer_need_connid(peer))
		header_flags |= RXR_PKT_CONNID_HDR;

	if (tx_entry->fi_flags & FI_REMOTE_CQ_DATA)
		header_flags |= RXR_REQ_OPT_CQ_DATA_HDR;

	return ep->mtu_size - rxr_pkt_req_header_size(pkt_type,
						      header_flags,
						      tx_entry->rma_iov_count);
}
