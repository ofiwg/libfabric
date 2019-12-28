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
#include "rxr_cntr.h"

/* This file implements 4 actions that can be applied to a packet:
 *          posting,
 *          handling send completion and,
 *          handing recv completion.
 *          dump (for debug only)
 */

/*
 *  Functions used to post a packet
 */
ssize_t rxr_pkt_post_data(struct rxr_ep *rxr_ep,
			  struct rxr_tx_entry *tx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_data_pkt *data_pkt;
	ssize_t ret;

	pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->tx_pkt_efa_pool);

	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_ENOMEM;

	pkt_entry->x_entry = (void *)tx_entry;
	pkt_entry->addr = tx_entry->addr;

	data_pkt = (struct rxr_data_pkt *)pkt_entry->pkt;

	data_pkt->hdr.type = RXR_DATA_PKT;
	data_pkt->hdr.version = RXR_PROTOCOL_VERSION;
	data_pkt->hdr.flags = 0;

	data_pkt->hdr.rx_id = tx_entry->rx_id;

	/*
	 * Data packets are sent in order so using bytes_sent is okay here.
	 */
	data_pkt->hdr.seg_offset = tx_entry->bytes_sent;

	if (efa_mr_cache_enable)
		ret = rxr_pkt_send_data_mr_cache(rxr_ep, tx_entry, pkt_entry);
	else
		ret = rxr_pkt_send_data(rxr_ep, tx_entry, pkt_entry);

	if (OFI_UNLIKELY(ret)) {
		rxr_pkt_entry_release_tx(rxr_ep, pkt_entry);
		return ret;
	}

	data_pkt = rxr_get_data_pkt(pkt_entry->pkt);
	tx_entry->bytes_sent += data_pkt->hdr.seg_size;
	tx_entry->window -= data_pkt->hdr.seg_size;
	return ret;
}

/*
 *   rxr_pkt_init_ctrl() uses init functions declared in rxr_pkt_type.h
 */
static
int rxr_pkt_init_ctrl(struct rxr_ep *rxr_ep, int entry_type, void *x_entry,
		      int ctrl_type, struct rxr_pkt_entry *pkt_entry)
{
	int ret = 0;

	switch (ctrl_type) {
	case RXR_RTS_PKT:
		ret = rxr_pkt_init_rts(rxr_ep, (struct rxr_tx_entry *)x_entry, pkt_entry);
		break;
	case RXR_READRSP_PKT:
		ret = rxr_pkt_init_readrsp(rxr_ep, (struct rxr_tx_entry *)x_entry, pkt_entry);
		break;
	case RXR_CTS_PKT:
		ret = rxr_pkt_init_cts(rxr_ep, (struct rxr_rx_entry *)x_entry, pkt_entry);
		break;
	case RXR_EOR_PKT:
		ret = rxr_pkt_init_eor(rxr_ep, (struct rxr_rx_entry *)x_entry, pkt_entry);
		break;
	default:
		ret = -FI_EINVAL;
		assert(0 && "unknown pkt type to init");
		break;
	}

	return ret;
}

/*
 *   rxr_pkt_handle_ctrl_sent() uses handle_sent() functions declared in rxr_pkt_type.h
 */
static
void rxr_pkt_handle_ctrl_sent(struct rxr_ep *rxr_ep, struct rxr_pkt_entry *pkt_entry)
{
	int ctrl_type = rxr_get_base_hdr(pkt_entry->pkt)->type;

	switch (ctrl_type) {
	case RXR_RTS_PKT:
		rxr_pkt_handle_rts_sent(rxr_ep, pkt_entry);
		break;
	case RXR_READRSP_PKT:
		rxr_pkt_handle_readrsp_sent(rxr_ep, pkt_entry);
		break;
	case RXR_CTS_PKT:
		rxr_pkt_handle_cts_sent(rxr_ep, pkt_entry);
		break;
	case RXR_EOR_PKT:
		rxr_pkt_handle_eor_sent(rxr_ep, pkt_entry);
		break;
	default:
		assert(0 && "Unknown packet type to handle sent");
		break;
	}
}

ssize_t rxr_pkt_post_ctrl(struct rxr_ep *rxr_ep, int entry_type, void *x_entry,
			  int ctrl_type, bool inject)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_tx_entry *tx_entry;
	struct rxr_rx_entry *rx_entry;
	struct rxr_peer *peer;
	ssize_t err;
	fi_addr_t addr;

	if (entry_type == RXR_TX_ENTRY) {
		tx_entry = (struct rxr_tx_entry *)x_entry;
		addr = tx_entry->addr;
	} else {
		rx_entry = (struct rxr_rx_entry *)x_entry;
		addr = rx_entry->addr;
	}

	peer = rxr_ep_get_peer(rxr_ep, addr);
	if (peer->is_local)
		pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->tx_pkt_shm_pool);
	else
		pkt_entry = rxr_pkt_entry_alloc(rxr_ep, rxr_ep->tx_pkt_efa_pool);

	if (!pkt_entry)
		return -FI_EAGAIN;

	err = rxr_pkt_init_ctrl(rxr_ep, entry_type, x_entry, ctrl_type, pkt_entry);
	if (OFI_UNLIKELY(err)) {
		rxr_pkt_entry_release_tx(rxr_ep, pkt_entry);
		return err;
	}

	/* if send, tx_pkt_entry will be released while handle completion
	 * if inject, there will not be completion, therefore tx_pkt_entry has to be
	 * released here
	 */
	if (inject)
		err = rxr_pkt_entry_inject(rxr_ep, pkt_entry, addr);
	else
		err = rxr_pkt_entry_send(rxr_ep, pkt_entry, addr);

	if (OFI_UNLIKELY(err)) {
		rxr_pkt_entry_release_tx(rxr_ep, pkt_entry);
		return err;
	}

	rxr_pkt_handle_ctrl_sent(rxr_ep, pkt_entry);

	if (inject)
		rxr_pkt_entry_release_tx(rxr_ep, pkt_entry);

	return 0;
}

ssize_t rxr_pkt_post_ctrl_or_queue(struct rxr_ep *ep, int entry_type, void *x_entry, int ctrl_type, bool inject)
{
	ssize_t err;
	struct rxr_tx_entry *tx_entry;
	struct rxr_rx_entry *rx_entry;

	err = rxr_pkt_post_ctrl(ep, entry_type, x_entry, ctrl_type, inject);
	if (err == -FI_EAGAIN) {
		if (entry_type == RXR_TX_ENTRY) {
			tx_entry = (struct rxr_tx_entry *)x_entry;
			tx_entry->state = RXR_TX_QUEUED_CTRL;
			tx_entry->queued_ctrl.type = ctrl_type;
			tx_entry->queued_ctrl.inject = inject;
			dlist_insert_tail(&tx_entry->queued_entry,
					  &ep->tx_entry_queued_list);
		} else {
			assert(entry_type == RXR_RX_ENTRY);
			rx_entry = (struct rxr_rx_entry *)x_entry;
			rx_entry->state = RXR_RX_QUEUED_CTRL;
			rx_entry->queued_ctrl.type = ctrl_type;
			rx_entry->queued_ctrl.inject = inject;
			dlist_insert_tail(&rx_entry->queued_entry,
					  &ep->rx_entry_queued_list);
		}

		err = 0;
	}

	return err;
}

/*
 *   Functions used to handle packet send completion
 */
void rxr_pkt_handle_send_completion(struct rxr_ep *ep, struct fi_cq_data_entry *comp)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_peer *peer;

	pkt_entry = (struct rxr_pkt_entry *)comp->op_context;
	assert(rxr_get_base_hdr(pkt_entry->pkt)->version ==
	       RXR_PROTOCOL_VERSION);

	switch (rxr_get_base_hdr(pkt_entry->pkt)->type) {
	case RXR_RTS_PKT:
		rxr_pkt_handle_rts_send_completion(ep, pkt_entry);
		break;
	case RXR_CONNACK_PKT:
		break;
	case RXR_CTS_PKT:
		break;
	case RXR_DATA_PKT:
		rxr_pkt_handle_data_send_completion(ep, pkt_entry);
		break;
	case RXR_READRSP_PKT:
		rxr_pkt_handle_readrsp_send_completion(ep, pkt_entry);
		break;
	case RXR_RMA_CONTEXT_PKT:
		rxr_pkt_handle_rma_context_send_completion(ep, pkt_entry);
		return;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"invalid control pkt type %d\n",
			rxr_get_base_hdr(pkt_entry->pkt)->type);
		assert(0 && "invalid control pkt type");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
		return;
	}
	rxr_pkt_entry_release_tx(ep, pkt_entry);
	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	if (!peer->is_local)
		rxr_ep_dec_tx_pending(ep, peer, 0);
}

/*
 *  Functions used to handle packet receive completion
 */
static
fi_addr_t rxr_pkt_insert_addr_from_rts(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	int i, ret;
	void *raw_address;
	fi_addr_t rdm_addr;
	struct rxr_rts_hdr *rts_hdr;
	struct efa_ep *efa_ep;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->type == RXR_RTS_PKT);

	efa_ep = container_of(ep->rdm_ep, struct efa_ep, util_ep.ep_fid);
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
		efa_eq_write_error(&ep->util_ep, FI_EIO, -FI_EINVAL);
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
		efa_eq_write_error(&ep->util_ep, FI_EINVAL, ret);
		return -1;
	}

	return rdm_addr;
}

void rxr_pkt_handle_recv_completion(struct rxr_ep *ep,
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
		pkt_entry->addr = rxr_pkt_insert_addr_from_rts(ep, pkt_entry);
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
		rxr_pkt_handle_rts_recv(ep, pkt_entry);
		return;
	case RXR_EOR_PKT:
		rxr_pkt_handle_eor_recv(ep, pkt_entry);
		return;
	case RXR_CONNACK_PKT:
		rxr_pkt_handle_connack_recv(ep, pkt_entry, src_addr);
		return;
	case RXR_CTS_PKT:
		rxr_pkt_handle_cts_recv(ep, pkt_entry);
		return;
	case RXR_DATA_PKT:
		rxr_pkt_handle_data_recv(ep, pkt_entry);
		return;
	case RXR_READRSP_PKT:
		rxr_pkt_handle_readrsp_recv(ep, pkt_entry);
		return;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"invalid control pkt type %d\n",
			rxr_get_base_hdr(pkt_entry->pkt)->type);
		assert(0 && "invalid control pkt type");
		rxr_cq_handle_cq_error(ep, -FI_EIO);
		return;
	}
}

#if ENABLE_DEBUG

/*
 *  Functions used to dump packets
 */

#define RXR_PKT_DUMP_DATA_LEN 64

static
void rxr_pkt_print_rts(struct rxr_ep *ep,
		       char *prefix, struct rxr_rts_hdr *rts_hdr)
{
	char str[RXR_PKT_DUMP_DATA_LEN * 4];
	size_t str_len = RXR_PKT_DUMP_DATA_LEN * 4, l;
	uint8_t *src;
	uint8_t *data;
	int i;

	str[str_len - 1] = '\0';

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR RTS packet - version: %"	PRIu8
	       " flags: %"	PRIu16
	       " tx_id: %"	PRIu32
	       " msg_id: %"	PRIu32
	       " tag: %lx data_len: %"	PRIu64 "\n",
	       prefix, rts_hdr->version, rts_hdr->flags, rts_hdr->tx_id,
	       rts_hdr->msg_id, rts_hdr->tag, rts_hdr->data_len);

	if ((rts_hdr->flags & RXR_REMOTE_CQ_DATA) &&
	    (rts_hdr->flags & RXR_REMOTE_SRC_ADDR)) {
		src = (uint8_t *)((struct rxr_ctrl_cq_pkt *)rts_hdr)->data;
		data = src + rts_hdr->addrlen;
	} else if (!(rts_hdr->flags & RXR_REMOTE_CQ_DATA) &&
		   (rts_hdr->flags & RXR_REMOTE_SRC_ADDR)) {
		src = (uint8_t *)((struct rxr_ctrl_pkt *)rts_hdr)->data;
		data = src + rts_hdr->addrlen;
	} else if ((rts_hdr->flags & RXR_REMOTE_CQ_DATA) &&
		   !(rts_hdr->flags & RXR_REMOTE_SRC_ADDR)) {
		data = (uint8_t *)((struct rxr_ctrl_cq_pkt *)rts_hdr)->data;
	} else {
		data = (uint8_t *)((struct rxr_ctrl_pkt *)rts_hdr)->data;
	}

	if (rts_hdr->flags & RXR_REMOTE_CQ_DATA)
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
		       "\tcq_data: %08lx\n",
		       ((struct rxr_ctrl_cq_hdr *)rts_hdr)->cq_data);

	if (rts_hdr->flags & RXR_REMOTE_SRC_ADDR) {
		l = snprintf(str, str_len, "\tsrc_addr: ");
		for (i = 0; i < rts_hdr->addrlen; i++)
			l += snprintf(str + l, str_len - l, "%02x ", src[i]);
		FI_DBG(&rxr_prov, FI_LOG_EP_DATA, "%s\n", str);
	}

	l = snprintf(str, str_len, ("\tdata:    "));
	for (i = 0; i < MIN(rxr_get_rts_data_size(ep, rts_hdr),
			    RXR_PKT_DUMP_DATA_LEN); i++)
		l += snprintf(str + l, str_len - l, "%02x ", data[i]);
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA, "%s\n", str);
}

static
void rxr_pkt_print_connack(char *prefix,
			   struct rxr_connack_hdr *connack_hdr)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR CONNACK packet - version: %" PRIu8
	       " flags: %x\n", prefix, connack_hdr->version,
	       connack_hdr->flags);
}

static
void rxr_pkt_print_cts(char *prefix, struct rxr_cts_hdr *cts_hdr)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR CTS packet - version: %"	PRIu8
	       " flags: %x tx_id: %" PRIu32
	       " rx_id: %"	   PRIu32
	       " window: %"	   PRIu64
	       "\n", prefix, cts_hdr->version, cts_hdr->flags,
	       cts_hdr->tx_id, cts_hdr->rx_id, cts_hdr->window);
}

static
void rxr_pkt_print_data(char *prefix, struct rxr_data_pkt *data_pkt)
{
	char str[RXR_PKT_DUMP_DATA_LEN * 4];
	size_t str_len = RXR_PKT_DUMP_DATA_LEN * 4, l;
	int i;

	str[str_len - 1] = '\0';

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR DATA packet -  version: %" PRIu8
	       " flags: %x rx_id: %" PRIu32
	       " seg_size: %"	     PRIu64
	       " seg_offset: %"	     PRIu64
	       "\n", prefix, data_pkt->hdr.version, data_pkt->hdr.flags,
	       data_pkt->hdr.rx_id, data_pkt->hdr.seg_size,
	       data_pkt->hdr.seg_offset);

	l = snprintf(str, str_len, ("\tdata:    "));
	for (i = 0; i < MIN(data_pkt->hdr.seg_size, RXR_PKT_DUMP_DATA_LEN);
	     i++)
		l += snprintf(str + l, str_len - l, "%02x ",
			      ((uint8_t *)data_pkt->data)[i]);
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA, "%s\n", str);
}

void rxr_pkt_print(char *prefix, struct rxr_ep *ep, struct rxr_base_hdr *hdr)
{
	switch (hdr->type) {
	case RXR_RTS_PKT:
		rxr_pkt_print_rts(ep, prefix, (struct rxr_rts_hdr *)hdr);
		break;
	case RXR_CONNACK_PKT:
		rxr_pkt_print_connack(prefix, (struct rxr_connack_hdr *)hdr);
		break;
	case RXR_CTS_PKT:
		rxr_pkt_print_cts(prefix, (struct rxr_cts_hdr *)hdr);
		break;
	case RXR_DATA_PKT:
		rxr_pkt_print_data(prefix, (struct rxr_data_pkt *)hdr);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "invalid ctl pkt type %d\n",
			rxr_get_base_hdr(hdr)->type);
		assert(0);
		return;
	}
}
#endif

