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
#include "rxr_pkt_cmd.h"
#include "rxr_pkt_type_base.h"

int rxr_pkt_init_data(struct rxr_ep *ep,
		      struct rxr_tx_entry *tx_entry,
		      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_data_hdr *data_hdr;
	struct rdm_peer *peer;
	size_t hdr_size;
	int ret;

	data_hdr = rxr_get_data_hdr(pkt_entry->pkt);
	data_hdr->type = RXR_DATA_PKT;
	data_hdr->version = RXR_PROTOCOL_VERSION;
	data_hdr->flags = 0;
	data_hdr->recv_id = tx_entry->rx_id;

	hdr_size = sizeof(struct rxr_data_hdr);
	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);
	if (rxr_peer_need_connid(peer)) {
		data_hdr->flags |= RXR_PKT_CONNID_HDR;
		data_hdr->connid_hdr->connid = rxr_ep_raw_addr(ep)->qkey;
		hdr_size += sizeof(struct rxr_data_opt_connid_hdr);
	}

	/*
	 * Data packets are sent in order so using bytes_sent is okay here.
	 */
	data_hdr->seg_offset = tx_entry->bytes_sent;
	data_hdr->seg_length = MIN(tx_entry->total_len - tx_entry->bytes_sent,
				   ep->max_data_payload_size);
	data_hdr->seg_length = MIN(data_hdr->seg_length, tx_entry->window);
	ret = rxr_pkt_init_data_from_tx_entry(ep, pkt_entry, hdr_size,
					      tx_entry, tx_entry->bytes_sent,
					      data_hdr->seg_length);
	if (ret)
		return ret;

	pkt_entry->x_entry = (void *)tx_entry;
	pkt_entry->addr = tx_entry->addr;

	return 0;
}

void rxr_pkt_handle_data_sent(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;
	struct rxr_data_hdr *data_hdr;

	data_hdr = rxr_get_data_hdr(pkt_entry->pkt);
	assert(data_hdr->seg_length > 0);

	tx_entry = pkt_entry->x_entry;
	tx_entry->bytes_sent += data_hdr->seg_length;
	tx_entry->window -= data_hdr->seg_length;
	assert(tx_entry->window >= 0);
}

void rxr_pkt_handle_data_send_completion(struct rxr_ep *ep,
					 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	tx_entry->bytes_acked +=
		rxr_get_data_hdr(pkt_entry->pkt)->seg_length;

	if (tx_entry->total_len == tx_entry->bytes_acked) {
		if (!(tx_entry->rxr_flags & RXR_DELIVERY_COMPLETE_REQUESTED))
			rxr_cq_handle_tx_completion(ep, tx_entry);
		else
			if (tx_entry->rxr_flags & RXR_RECEIPT_RECEIVED)
				/*
				 * For long message protocol,
				 * when FI_DELIVERY_COMPLETE
				 * is requested,
				 * we have to write tx completions
				 * in either
				 * rxr_pkt_handle_data_send_completion()
				 * or rxr_pkt_handle_receipt_recv()
				 * depending on which of them
				 * is called later due
				 * to avoid accessing released
				 * tx_entry.
				 */
				rxr_cq_handle_tx_completion(ep, tx_entry);
	}
}

/*
 *  rxr_pkt_handle_data_recv() and related functions
 */

/*
 * rxr_pkt_proc_data() processes data in a DATA/READRSP
 * pakcet entry.
 */
void rxr_pkt_proc_data(struct rxr_ep *ep,
		       struct rxr_rx_entry *rx_entry,
		       struct rxr_pkt_entry *pkt_entry,
		       char *data, size_t seg_offset,
		       size_t seg_size)
{
	struct rdm_peer *peer;
	bool all_received = 0;
	ssize_t err;

#if ENABLE_DEBUG
	int pkt_type = rxr_get_base_hdr(pkt_entry->pkt)->type;

	assert(pkt_type == RXR_DATA_PKT || pkt_type == RXR_READRSP_PKT);
#endif
	rx_entry->bytes_received += seg_size;
	assert(rx_entry->bytes_received <= rx_entry->total_len);
	all_received = (rx_entry->bytes_received == rx_entry->total_len);

	peer = rxr_ep_get_peer(ep, rx_entry->addr);
	assert(peer);
	peer->rx_credits += ofi_div_ceil(seg_size, ep->max_data_payload_size);

	rx_entry->window -= seg_size;
	if (ep->available_data_bufs < rxr_get_rx_pool_chunk_cnt(ep))
		ep->available_data_bufs++;

#if ENABLE_DEBUG
	/* rx_entry can be released by rxr_pkt_copy_data_to_rx_entry
	 * so the call to dlist_remove must happen before
	 * call to rxr_copy_data_to_rx_entry
	 */
	if (all_received) {
		dlist_remove(&rx_entry->rx_pending_entry);
		ep->rx_pending--;
	}
#endif
	err = rxr_pkt_copy_data_to_rx_entry(ep, rx_entry, seg_offset,
					    pkt_entry, data, seg_size);
	if (err) {
		rxr_pkt_entry_release_rx(ep, pkt_entry);
		rxr_cq_write_rx_error(ep, rx_entry, -err, -err);
	}

	if (all_received)
		return;

	if (!rx_entry->window) {
		assert(rx_entry->state == RXR_RX_RECV);
		err = rxr_pkt_post_ctrl_or_queue(ep, RXR_RX_ENTRY, rx_entry, RXR_CTS_PKT, 0);
		if (err) {
			FI_WARN(&rxr_prov, FI_LOG_CQ, "post CTS packet failed!\n");
			rxr_cq_write_rx_error(ep, rx_entry, -err, -err);
		}
	}
}

void rxr_pkt_handle_data_recv(struct rxr_ep *ep,
			      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_data_hdr *data_hdr;
	struct rxr_rx_entry *rx_entry;
	size_t hdr_size;

	data_hdr = rxr_get_data_hdr(pkt_entry->pkt);

	rx_entry = ofi_bufpool_get_ibuf(ep->rx_entry_pool,
					data_hdr->recv_id);

	hdr_size = sizeof(struct rxr_data_hdr);
	if (data_hdr->flags & RXR_PKT_CONNID_HDR)
		hdr_size += sizeof(struct rxr_data_opt_connid_hdr);

	rxr_pkt_proc_data(ep, rx_entry,
			  pkt_entry,
			  pkt_entry->pkt + hdr_size,
			  data_hdr->seg_offset,
			  data_hdr->seg_length);
}

