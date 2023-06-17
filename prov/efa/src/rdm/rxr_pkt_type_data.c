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

#include "efa_rdm_msg.h"
#include "rxr_pkt_cmd.h"
#include "rxr_pkt_type_base.h"

int rxr_pkt_init_data(struct efa_rdm_pke *pkt_entry,
		      struct efa_rdm_ope *ope,
		      size_t data_offset,
		      int data_size)
{
	struct rxr_data_hdr *data_hdr;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *ep;
	size_t hdr_size;
	int ret;

	ep = ope->ep;
	data_hdr = rxr_get_data_hdr(pkt_entry->wiredata);
	data_hdr->type = RXR_DATA_PKT;
	data_hdr->version = RXR_PROTOCOL_VERSION;
	data_hdr->flags = 0;

	/* Data is sent using rxe in the emulated longcts read 
	 * protocol. The emulated longcts write and the longcts 
	 * message protocols sends data using txe.
	 * This check ensures appropriate recv_id is 
	 * assigned for the respective protocols */
	if (ope->type == EFA_RDM_RXE) {
		data_hdr->recv_id = ope->tx_id;
	} else {
		assert(ope->type == EFA_RDM_TXE);
		data_hdr->recv_id = ope->rx_id;
		if (ope->rxr_flags & EFA_RDM_TXE_DELIVERY_COMPLETE_REQUESTED)
			pkt_entry->flags |= EFA_RDM_PKE_DC_LONGCTS_DATA;
	}

	hdr_size = sizeof(struct rxr_data_hdr);
	peer = efa_rdm_ep_get_peer(ep, ope->addr);
	assert(peer);
	if (efa_rdm_peer_need_connid(peer)) {
		data_hdr->flags |= RXR_PKT_CONNID_HDR;
		data_hdr->connid_hdr->connid = efa_rdm_ep_raw_addr(ep)->qkey;
		hdr_size += sizeof(struct rxr_data_opt_connid_hdr);
	}

	/*
	 * Data packets are sent in order so using bytes_sent is okay here.
	 */
	data_hdr->seg_offset = data_offset;
	data_hdr->seg_length = data_size;
	ret = rxr_pkt_init_data_from_ope(ep, pkt_entry, hdr_size,
					 ope, data_offset, data_size);
	if (ret)
		return ret;

	pkt_entry->ope = (void *)ope;
	pkt_entry->addr = ope->addr;

	return 0;
}

void rxr_pkt_handle_data_sent(struct efa_rdm_ep *ep, struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *ope;
	struct rxr_data_hdr *data_hdr;

	data_hdr = rxr_get_data_hdr(pkt_entry->wiredata);
	assert(data_hdr->seg_length > 0);

	ope = pkt_entry->ope;
	ope->bytes_sent += data_hdr->seg_length;
	ope->window -= data_hdr->seg_length;
	assert(ope->window >= 0);
}

void rxr_pkt_handle_data_send_completion(struct efa_rdm_ep *ep,
					 struct efa_rdm_pke *pkt_entry)
{
	struct efa_rdm_ope *ope;

	/* if this DATA packet is used by a DC protocol, the completion
	 * was (or will be) written when the receipt packet was received.
	 * The txe may have already been released. So nothing
	 * to do (or can be done) here.
	 */
	if (pkt_entry->flags & EFA_RDM_PKE_DC_LONGCTS_DATA)
		return;

	ope = pkt_entry->ope;
	ope->bytes_acked +=
		rxr_get_data_hdr(pkt_entry->wiredata)->seg_length;

	if (ope->total_len == ope->bytes_acked)
		efa_rdm_ope_handle_send_completed(ope);
}

/*
 *  rxr_pkt_handle_data_recv() and related functions
 */

/*
 * rxr_pkt_proc_data() processes data in a DATA/READRSP
 * packet entry.
 */
void rxr_pkt_proc_data(struct efa_rdm_ep *ep,
		       struct efa_rdm_ope *ope,
		       struct efa_rdm_pke *pkt_entry,
		       char *data, size_t seg_offset,
		       size_t seg_size)
{
	bool all_received = 0;
	ssize_t err;

#if ENABLE_DEBUG
	int pkt_type = rxr_get_base_hdr(pkt_entry->wiredata)->type;

	assert(pkt_type == RXR_DATA_PKT || pkt_type == RXR_READRSP_PKT);
#endif
	ope->bytes_received += seg_size;
	assert(ope->bytes_received <= ope->total_len);
	all_received = (ope->bytes_received == ope->total_len);

	ope->window -= seg_size;
#if ENABLE_DEBUG
	/* ope can be released by rxr_pkt_copy_data_to_ope
	 * so the call to dlist_remove must happen before
	 * call to rxr_copy_data_to_ope
	 */
	if (all_received) {
		dlist_remove(&ope->pending_recv_entry);
		ep->pending_recv_counter--;
	}
#endif
	err = rxr_pkt_copy_data_to_ope(ep, ope, seg_offset,
					    pkt_entry, data, seg_size);
	if (err) {
		efa_rdm_pke_release_rx(ep, pkt_entry);
		efa_rdm_rxe_handle_error(ope, -err, FI_EFA_ERR_RXE_COPY);
	}

	if (all_received)
		return;

	if (!ope->window) {
		err = rxr_pkt_post_or_queue(ep, ope, RXR_CTS_PKT);
		if (err) {
			EFA_WARN(FI_LOG_CQ, "post CTS packet failed!\n");
			efa_rdm_rxe_handle_error(ope, -err, FI_EFA_ERR_PKT_POST);
		}
	}
}

void rxr_pkt_handle_data_recv(struct efa_rdm_ep *ep,
			      struct efa_rdm_pke *pkt_entry)
{
	struct rxr_data_hdr *data_hdr;
	struct efa_rdm_ope *ope;
	size_t hdr_size;

	data_hdr = rxr_get_data_hdr(pkt_entry->wiredata);

	ope = ofi_bufpool_get_ibuf(ep->ope_pool,
					data_hdr->recv_id);

	hdr_size = sizeof(struct rxr_data_hdr);
	if (data_hdr->flags & RXR_PKT_CONNID_HDR)
		hdr_size += sizeof(struct rxr_data_opt_connid_hdr);

	rxr_pkt_proc_data(ep, ope,
			  pkt_entry,
			  pkt_entry->wiredata + hdr_size,
			  data_hdr->seg_offset,
			  data_hdr->seg_length);
}

