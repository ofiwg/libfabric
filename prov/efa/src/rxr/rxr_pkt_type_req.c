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

#include "efa.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_pkt_cmd.h"

/*
 * Utility constants and funnctions shared by all REQ packe
 * types.
 */

size_t RXR_REQ_HDR_SIZE_LIST[] = {
	[RXR_EAGER_MSGRTM_PKT] = sizeof(struct rxr_eager_msgrtm_hdr),
	[RXR_EAGER_TAGRTM_PKT] = sizeof(struct rxr_eager_tagrtm_hdr),
	[RXR_LONG_MSGRTM_PKT] = sizeof(struct rxr_long_msgrtm_hdr),
	[RXR_LONG_TAGRTM_PKT] = sizeof(struct rxr_long_tagrtm_hdr),
};

size_t rxr_pkt_req_data_size(struct rxr_pkt_entry *pkt_entry)
{
	assert(pkt_entry->hdr_size > 0);
	return pkt_entry->pkt_size - pkt_entry->hdr_size;
}

size_t rxr_pkt_req_total_len(struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	switch (base_hdr->type) {
	case RXR_EAGER_MSGRTM_PKT:
	case RXR_EAGER_TAGRTM_PKT:
		return rxr_pkt_req_data_size(pkt_entry);
	case RXR_LONG_MSGRTM_PKT:
	case RXR_LONG_TAGRTM_PKT:
		return rxr_get_long_rtm_base_hdr(pkt_entry->pkt)->data_len;
	default:
		assert(0 && "Unknown REQ packet type\n");
	}

	return 0;
}

static
void rxr_pkt_init_req_hdr(struct rxr_ep *ep,
			  struct rxr_tx_entry *tx_entry,
			  int pkt_type,
			  struct rxr_pkt_entry *pkt_entry)
{
	int i;
	char *opt_hdr;
	struct rxr_peer *peer;
	struct rxr_base_hdr *base_hdr;

	/* init the base header */
	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	base_hdr->type = pkt_type;
	base_hdr->version = RXR_PROTOCOL_VERSION;
	base_hdr->flags = 0;

	/* init the opt header */
	opt_hdr = (char *)base_hdr + RXR_REQ_HDR_SIZE_LIST[base_hdr->type];
	peer = rxr_ep_get_peer(ep, tx_entry->addr);
	assert(peer);
	if (OFI_UNLIKELY(peer->state != RXR_PEER_ACKED)) {
		/*
		 * This is the first communication with this peer on this
		 * endpoint, so send the core's address for this EP in the REQ
		 * so the remote side can insert it into its address vector.
		 */
		struct rxr_req_opt_raw_addr_hdr *raw_addr_hdr;

		raw_addr_hdr = (struct rxr_req_opt_raw_addr_hdr *)opt_hdr;
		raw_addr_hdr->addr_len = ep->core_addrlen;
		base_hdr->flags |= RXR_REQ_OPT_RAW_ADDR_HDR;
		for (i = 0; i < raw_addr_hdr->addr_len; ++i)
			raw_addr_hdr->raw_addr[i] = ep->core_addr[i];

		opt_hdr += sizeof(*raw_addr_hdr);
	}

	if (tx_entry->fi_flags & FI_REMOTE_CQ_DATA) {
		struct rxr_req_opt_cq_data_hdr *cq_data_hdr;

		base_hdr->flags |= RXR_REQ_OPT_CQ_DATA_HDR;
		cq_data_hdr = (struct rxr_req_opt_cq_data_hdr *)opt_hdr;
		cq_data_hdr->cq_data = tx_entry->cq_entry.data;
		opt_hdr += sizeof(*cq_data_hdr);
	}

	pkt_entry->addr = tx_entry->addr;
	pkt_entry->hdr_size = opt_hdr - (char *)pkt_entry->pkt;
}

void rxr_pkt_read_req_hdr(struct rxr_pkt_entry *pkt_entry,
			  struct rxr_rx_entry *rx_entry)
{
	char *opt_hdr;
	struct rxr_base_hdr *base_hdr;

	rx_entry->addr = pkt_entry->addr;

	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	opt_hdr = (char *)pkt_entry->pkt + RXR_REQ_HDR_SIZE_LIST[base_hdr->type];
	if (base_hdr->flags & RXR_REQ_OPT_RAW_ADDR_HDR) {
		rx_entry->rxr_flags |= RXR_REMOTE_SRC_ADDR;
		opt_hdr += sizeof(struct rxr_req_opt_raw_addr_hdr);
	}

	if (base_hdr->flags & RXR_REQ_OPT_CQ_DATA_HDR) {
		struct rxr_req_opt_cq_data_hdr *cq_data_hdr;

		cq_data_hdr = (struct rxr_req_opt_cq_data_hdr *)opt_hdr;
		rx_entry->rxr_flags |= RXR_REMOTE_CQ_DATA;
		rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rx_entry->cq_entry.data = cq_data_hdr->cq_data;
		opt_hdr += sizeof(struct rxr_req_opt_cq_data_hdr);
	}

	pkt_entry->hdr_size = opt_hdr - (char *)pkt_entry->pkt;
	rx_entry->total_len = rxr_pkt_req_total_len(pkt_entry);
	rx_entry->msg_id = rxr_pkt_rtm_msg_id(pkt_entry);
	rx_entry->tag = rxr_pkt_rtm_tag(pkt_entry);
	rx_entry->cq_entry.tag = rx_entry->tag;
}

size_t rxr_pkt_req_max_data_size(struct rxr_ep *ep, fi_addr_t addr, int pkt_id)
{
	struct rxr_peer *peer;

	peer = rxr_ep_get_peer(ep, addr);
	assert(peer);

	if (rxr_env.enable_shm_transfer && peer->is_local)
		return rxr_env.shm_max_medium_size;

	return ep->mtu_size - RXR_REQ_HDR_SIZE_LIST[pkt_id]
		- sizeof(struct rxr_req_opt_raw_addr_hdr)
		- sizeof(struct rxr_req_opt_cq_data_hdr);
}

static
size_t rxr_pkt_req_copy_data(struct rxr_rx_entry *rx_entry,
			     struct rxr_pkt_entry *pkt_entry,
			     char *data, size_t data_size)
{
	size_t bytes_copied;
	int bytes_left;
	/* rx_entry->cq_entry.len is total recv buffer size.
	 * rx_entry->total_len is from REQ packet and is total send buffer size.
	 * if send buffer size < recv buffer size, we adjust value of rx_entry->cq_entry.len.
	 * if send buffer size > recv buffer size, we have a truncated message.
	 */
	if (rx_entry->cq_entry.len > rx_entry->total_len)
		rx_entry->cq_entry.len = rx_entry->total_len;

	bytes_copied = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
				       0, data, data_size);

	if (OFI_UNLIKELY(bytes_copied < data_size)) {
		/* recv buffer is not big enough to hold req, this must be a truncated message */
		assert(bytes_copied == rx_entry->cq_entry.len &&
		       rx_entry->cq_entry.len < rx_entry->total_len);
		rx_entry->bytes_done = bytes_copied;
		bytes_left = 0;
	} else {
		assert(bytes_copied == data_size);
		rx_entry->bytes_done = data_size;
		bytes_left = rx_entry->total_len - rx_entry->bytes_done;
	}

	assert(bytes_left >= 0);
	return bytes_left;
}

/*
 * REQ packet type functions
 *
 *     init() functions
 */
void rxr_pkt_init_rtm(struct rxr_ep *ep,
		      struct rxr_tx_entry *tx_entry,
		      int pkt_type,
		      struct rxr_pkt_entry *pkt_entry)
{
	char *data;
	size_t data_size;
	struct rxr_rtm_base_hdr *rtm_hdr;

	/* this function set pkt_entry->hdr_size */
	rxr_pkt_init_req_hdr(ep, tx_entry, pkt_type, pkt_entry);

	rtm_hdr = (struct rxr_rtm_base_hdr *)pkt_entry->pkt;
	rtm_hdr->flags |= RXR_REQ_MSG;
	rtm_hdr->msg_id = tx_entry->msg_id;

	data = (char *)pkt_entry->pkt + pkt_entry->hdr_size;
	data_size = ofi_copy_from_iov(data, ep->mtu_size - pkt_entry->hdr_size,
				      tx_entry->iov, tx_entry->iov_count, 0);

	pkt_entry->pkt_size = pkt_entry->hdr_size + data_size;
	pkt_entry->x_entry = tx_entry;
}

ssize_t rxr_pkt_init_eager_msgrtm(struct rxr_ep *ep,
				  struct rxr_tx_entry *tx_entry,
				  struct rxr_pkt_entry *pkt_entry)
{
	rxr_pkt_init_rtm(ep, tx_entry, RXR_EAGER_MSGRTM_PKT, pkt_entry);
	return 0;
}

ssize_t rxr_pkt_init_eager_tagrtm(struct rxr_ep *ep,
				  struct rxr_tx_entry *tx_entry,
				  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	rxr_pkt_init_rtm(ep, tx_entry, RXR_EAGER_TAGRTM_PKT, pkt_entry);
	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, tx_entry->tag);
	return 0;
}

void rxr_pkt_init_long_rtm(struct rxr_ep *ep,
			   struct rxr_tx_entry *tx_entry,
			   int pkt_type,
			   struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_long_rtm_base_hdr *rtm_hdr;

	rxr_pkt_init_rtm(ep, tx_entry, pkt_type, pkt_entry);
	rtm_hdr = rxr_get_long_rtm_base_hdr(pkt_entry->pkt);
	rtm_hdr->data_len = tx_entry->total_len;
	rtm_hdr->tx_id = tx_entry->tx_id;
	rtm_hdr->credit_request = tx_entry->credit_request;
}

ssize_t rxr_pkt_init_long_msgrtm(struct rxr_ep *ep,
				 struct rxr_tx_entry *tx_entry,
				 struct rxr_pkt_entry *pkt_entry)
{
	rxr_pkt_init_long_rtm(ep, tx_entry, RXR_LONG_MSGRTM_PKT, pkt_entry);
	return 0;
}

ssize_t rxr_pkt_init_long_tagrtm(struct rxr_ep *ep,
				 struct rxr_tx_entry *tx_entry,
				 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	rxr_pkt_init_long_rtm(ep, tx_entry, RXR_LONG_TAGRTM_PKT, pkt_entry);
	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	base_hdr->flags |= RXR_REQ_TAGGED;
	rxr_pkt_rtm_settag(pkt_entry, tx_entry->tag);
	return 0;
}

/*
 *     handle_sent() functions
 */

/*
 *         rxr_pkt_handle_eager_rtm_sent() is empty and is defined in rxr_pkt_type_req.h
 */
void rxr_pkt_handle_long_rtm_sent(struct rxr_ep *ep,
				  struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	tx_entry->bytes_sent += rxr_pkt_req_data_size(pkt_entry);
	assert(tx_entry->bytes_sent < tx_entry->total_len);

	if (efa_mr_cache_enable && !tx_entry->desc[0])
		rxr_inline_mr_reg(rxr_ep_domain(ep), tx_entry);
}

/*
 *     handle_send_completion() functions
 */
void rxr_pkt_handle_eager_rtm_send_completion(struct rxr_ep *ep,
					      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	assert(tx_entry->total_len == rxr_pkt_req_data_size(pkt_entry));
	rxr_cq_handle_tx_completion(ep, tx_entry);
}

void rxr_pkt_handle_long_rtm_send_completion(struct rxr_ep *ep,
					     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_tx_entry *tx_entry;

	tx_entry = (struct rxr_tx_entry *)pkt_entry->x_entry;
	tx_entry->bytes_acked += rxr_pkt_req_data_size(pkt_entry);
	if (tx_entry->total_len == tx_entry->bytes_acked)
		rxr_cq_handle_tx_completion(ep, tx_entry);
}

/*
 *     proc() functions
 */
struct rxr_rx_entry *rxr_pkt_get_rtm_matched_rx_entry(struct rxr_ep *ep,
						      struct dlist_entry *match,
						      struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;

	assert(match);
	rx_entry = container_of(match, struct rxr_rx_entry, entry);
	if (rx_entry->rxr_flags & RXR_MULTI_RECV_POSTED) {
		rx_entry = rxr_ep_split_rx_entry(ep, rx_entry,
						 NULL, pkt_entry);
		if (OFI_UNLIKELY(!rx_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
			return NULL;
		}

	} else {
		rxr_pkt_read_req_hdr(pkt_entry, rx_entry);
	}

	rx_entry->state = RXR_RX_MATCHED;

	if (!(rx_entry->fi_flags & FI_MULTI_RECV) ||
	    !rxr_msg_multi_recv_buffer_available(ep, rx_entry->master_entry))
		dlist_remove(match);

	return rx_entry;
}

static
int rxr_pkt_rtm_match_recv(struct dlist_entry *item, const void *arg)
{
	const struct rxr_pkt_entry *pkt_entry = arg;
	struct rxr_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);
	return rxr_match_addr(rx_entry->addr, pkt_entry->addr);
}

static
int rxr_pkt_rtm_match_trecv(struct dlist_entry *item, const void *arg)
{
	struct rxr_pkt_entry *pkt_entry = (struct rxr_pkt_entry *)arg;
	struct rxr_rx_entry *rx_entry;
	uint64_t match_tag;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);
	match_tag = rxr_pkt_rtm_tag(pkt_entry);

	return rxr_match_addr(rx_entry->addr, pkt_entry->addr) &&
	       rxr_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     match_tag);
}

static
struct rxr_rx_entry *rxr_pkt_get_msgrtm_rx_entry(struct rxr_ep *ep,
						 struct rxr_pkt_entry **pkt_entry_ptr)
{
	struct rxr_rx_entry *rx_entry;
	struct dlist_entry *match;

	match = dlist_find_first_match(&ep->rx_list, &rxr_pkt_rtm_match_recv, *pkt_entry_ptr);
	if (OFI_UNLIKELY(!match)) {
		/*
		 * rxr_ep_alloc_unexp_rx_entry_for_msgrtm() might release pkt_entry,
		 * thus we have to use pkt_entry_ptr here
		 */
		rx_entry = rxr_ep_alloc_unexp_rx_entry_for_msgrtm(ep, pkt_entry_ptr);
		if (OFI_UNLIKELY(!rx_entry))
			return NULL;

	} else {
		rx_entry = rxr_pkt_get_rtm_matched_rx_entry(ep, match, *pkt_entry_ptr);
	}

	assert(rx_entry->total_len > 0);
	return rx_entry;
}

static
struct rxr_rx_entry *rxr_pkt_get_tagrtm_rx_entry(struct rxr_ep *ep,
						 struct rxr_pkt_entry **pkt_entry_ptr)
{
	struct rxr_rx_entry *rx_entry;
	struct dlist_entry *match;

	match = dlist_find_first_match(&ep->rx_tagged_list, &rxr_pkt_rtm_match_trecv, *pkt_entry_ptr);
	if (OFI_UNLIKELY(!match)) {
		/*
		 * rxr_ep_alloc_unexp_rx_entry_for_tagrtm() might release pkt_entry,
		 * thus we have to use pkt_entry_ptr here
		 */
		rx_entry = rxr_ep_alloc_unexp_rx_entry_for_tagrtm(ep, pkt_entry_ptr);
		if (OFI_UNLIKELY(!rx_entry))
			return NULL;

	} else {
		rx_entry = rxr_pkt_get_rtm_matched_rx_entry(ep, match, *pkt_entry_ptr);
	}

	return rx_entry;
}

ssize_t rxr_pkt_proc_matched_rtm(struct rxr_ep *ep,
				 struct rxr_rx_entry *rx_entry,
				 struct rxr_pkt_entry *pkt_entry)
{
	char *data;
	size_t data_size, bytes_left;
	ssize_t ret;

	assert(rx_entry->state == RXR_RX_MATCHED);
	data = (char *)pkt_entry->pkt + pkt_entry->hdr_size;
	data_size = pkt_entry->pkt_size - pkt_entry->hdr_size;
	bytes_left = rxr_pkt_req_copy_data(rx_entry, pkt_entry,
					   data, data_size);
	if (!bytes_left) {
		/*
		 * rxr_cq_handle_rx_completion() releases pkt_entry, thus
		 * we do not release it here.
		 */
		rxr_cq_handle_rx_completion(ep, pkt_entry, rx_entry);
		rxr_msg_multi_recv_free_posted_entry(ep, rx_entry);
		rxr_release_rx_entry(ep, rx_entry);
		ret = 0;
	} else {
		/*
		 * long message protocol
		 */
#if ENABLE_DEBUG
		dlist_insert_tail(&rx_entry->rx_pending_entry, &ep->rx_pending_list);
		ep->rx_pending++;
#endif
		rx_entry->state = RXR_RX_RECV;
		rx_entry->tx_id = rxr_get_long_rtm_base_hdr(pkt_entry->pkt)->tx_id;
		rx_entry->credit_request = rxr_env.tx_min_credits;
		ret = rxr_pkt_post_ctrl_or_queue(ep, RXR_RX_ENTRY, rx_entry, RXR_CTS_PKT, 0);
		rxr_pkt_entry_release_rx(ep, pkt_entry);
	}

	return ret;
}

ssize_t rxr_pkt_proc_msgrtm(struct rxr_ep *ep,
			    struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;

	rx_entry = rxr_pkt_get_msgrtm_rx_entry(ep, &pkt_entry);
	if (OFI_UNLIKELY(!rx_entry))
		return -FI_ENOBUFS;

	if (rx_entry->state == RXR_RX_MATCHED)
		return rxr_pkt_proc_matched_rtm(ep, rx_entry, pkt_entry);

	return 0;
}

ssize_t rxr_pkt_proc_tagrtm(struct rxr_ep *ep,
			    struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;

	rx_entry = rxr_pkt_get_tagrtm_rx_entry(ep, &pkt_entry);
	if (OFI_UNLIKELY(!rx_entry))
		return -FI_ENOBUFS;

	if (rx_entry->state == RXR_RX_MATCHED)
		return rxr_pkt_proc_matched_rtm(ep, rx_entry, pkt_entry);

	return 0;
}

/*
 * proc() functions called by rxr_pkt_handle_recv_completion()
 */
ssize_t rxr_pkt_proc_rtm(struct rxr_ep *ep,
			 struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_base_hdr *base_hdr;

	base_hdr = rxr_get_base_hdr(pkt_entry->pkt);
	assert(base_hdr->type >= RXR_BASELINE_REQ_PKT_BEGIN);

	switch (base_hdr->type) {
	case RXR_EAGER_MSGRTM_PKT:
	case RXR_LONG_MSGRTM_PKT:
		return rxr_pkt_proc_msgrtm(ep, pkt_entry);
	case RXR_EAGER_TAGRTM_PKT:
	case RXR_LONG_TAGRTM_PKT:
		return rxr_pkt_proc_tagrtm(ep, pkt_entry);
	default:
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Unknown packet type ID: %d\n",
		       base_hdr->type);
		if (rxr_cq_handle_cq_error(ep, -FI_EINVAL))
			assert(0 && "failed to write err cq entry");
	}

	return -FI_EINVAL;
}

void rxr_pkt_handle_rtm_recv(struct rxr_ep *ep,
			     struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_peer *peer;
	int ret, msg_id;

	assert(rxr_get_base_hdr(pkt_entry->pkt)->type >= RXR_BASELINE_REQ_PKT_BEGIN);

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	assert(peer);

	if (ep->core_caps & FI_SOURCE)
		rxr_pkt_post_connack(ep, peer, pkt_entry->addr);

	if (rxr_env.enable_shm_transfer && peer->is_local) {
		/* no need to reorder msg for shm_ep
		 * rxr_pkt_proc_rtm will write error cq entry if needed
		 */
		rxr_pkt_proc_rtm(ep, pkt_entry);
		return;
	}

	msg_id = rxr_pkt_rtm_msg_id(pkt_entry);
	if (rxr_need_sas_ordering(ep)) {
		ret = rxr_cq_reorder_msg(ep, peer, pkt_entry);
		if (ret == 1) {
			/* Packet was queued */
			return;
		} else if (OFI_UNLIKELY(ret == -FI_EALREADY)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Invalid msg_id: %" PRIu32
				" robuf->exp_msg_id: %" PRIu32 "\n",
			       msg_id, peer->robuf->exp_msg_id);
			efa_eq_write_error(&ep->util_ep, FI_EIO, ret);
			rxr_pkt_entry_release_rx(ep, pkt_entry);
			return;
		} else if (OFI_UNLIKELY(ret == -FI_ENOMEM)) {
			efa_eq_write_error(&ep->util_ep, FI_ENOBUFS, -FI_ENOBUFS);
			return;
		} else if (OFI_UNLIKELY(ret < 0)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unknown error %d processing REQ packet msg_id: %"
				PRIu32 "\n", ret, msg_id);
			efa_eq_write_error(&ep->util_ep, FI_EIO, ret);
			return;
		}

		/* processing the expected packet */
		ofi_recvwin_slide(peer->robuf);
	}

	/* rxr_pkt_proc_rtm will write error cq entry if needed */
	ret = rxr_pkt_proc_rtm(ep, pkt_entry);
	if (OFI_UNLIKELY(ret))
		return;

	/* process pending items in reorder buff */
	if (rxr_need_sas_ordering(ep))
		rxr_cq_proc_pending_items_in_recvwin(ep, peer);
}
