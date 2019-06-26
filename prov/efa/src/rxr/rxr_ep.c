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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>

#include "rxr.h"
#include "efa.h"
#include "rxr_rma.h"

#define RXR_PKT_DUMP_DATA_LEN 64

struct rxr_match_info {
	fi_addr_t addr;
	uint64_t tag;
	uint64_t ignore;
};

static void rxr_ep_progress_internal(struct rxr_ep *ep);

#if ENABLE_DEBUG
static void rxr_ep_print_rts_pkt(struct rxr_ep *ep,
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

static void rxr_ep_print_connack_pkt(char *prefix,
				     struct rxr_connack_hdr *connack_hdr)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR CONNACK packet - version: %" PRIu8
	       " flags: %x\n", prefix, connack_hdr->version,
	       connack_hdr->flags);
}

static void rxr_ep_print_cts_pkt(char *prefix, struct rxr_cts_hdr *cts_hdr)
{
	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s RxR CTS packet - version: %"	PRIu8
	       " flags: %x tx_id: %" PRIu32
	       " rx_id: %"	   PRIu32
	       " window: %"	   PRIu64
	       "\n", prefix, cts_hdr->version, cts_hdr->flags,
	       cts_hdr->tx_id, cts_hdr->rx_id, cts_hdr->window);
}

static void rxr_ep_print_data_pkt(char *prefix, struct rxr_data_pkt *data_pkt)
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

void rxr_ep_print_pkt(char *prefix, struct rxr_ep *ep, struct rxr_base_hdr *hdr)
{
	switch (hdr->type) {
	case RXR_RTS_PKT:
		rxr_ep_print_rts_pkt(ep, prefix, (struct rxr_rts_hdr *)hdr);
		break;
	case RXR_CONNACK_PKT:
		rxr_ep_print_connack_pkt(prefix, (struct rxr_connack_hdr *)hdr);
		break;
	case RXR_CTS_PKT:
		rxr_ep_print_cts_pkt(prefix, (struct rxr_cts_hdr *)hdr);
		break;
	case RXR_DATA_PKT:
		rxr_ep_print_data_pkt(prefix, (struct rxr_data_pkt *)hdr);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "invalid ctl pkt type %d\n",
			rxr_get_base_hdr(hdr)->type);
		assert(0);
		return;
	}
}
#endif

struct rxr_rx_entry *rxr_ep_rx_entry_init(struct rxr_ep *ep,
					  struct rxr_rx_entry *rx_entry,
					  const struct iovec *iov,
					  size_t iov_count, uint64_t tag,
					  uint64_t ignore, void *context,
					  fi_addr_t addr, uint32_t op,
					  uint64_t flags)
{
	rx_entry->type = RXR_RX_ENTRY;
	rx_entry->rx_id = ofi_buf_index(rx_entry);
	rx_entry->addr = addr;
	rx_entry->fi_flags = flags;
	rx_entry->rxr_flags = 0;
	rx_entry->bytes_done = 0;
	rx_entry->window = 0;
	rx_entry->iov_count = iov_count;
	rx_entry->tag = tag;
	rx_entry->ignore = ignore;
	rx_entry->unexp_rts_pkt = NULL;
	dlist_init(&rx_entry->queued_pkts);

	memset(&rx_entry->cq_entry, 0, sizeof(rx_entry->cq_entry));

	/* Handle case where we're allocating an unexpected rx_entry */
	if (iov) {
		memcpy(rx_entry->iov, iov, sizeof(*rx_entry->iov) * iov_count);
		rx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
		rx_entry->cq_entry.buf = iov[0].iov_base;
	}

	rx_entry->cq_entry.op_context = context;
	rx_entry->cq_entry.tag = 0;
	rx_entry->ignore = ~0;

	switch (op) {
	case ofi_op_tagged:
		rx_entry->cq_entry.flags = (FI_RECV | FI_MSG | FI_TAGGED);
		rx_entry->cq_entry.tag = tag;
		rx_entry->ignore = ignore;
		break;
	case ofi_op_msg:
		rx_entry->cq_entry.flags = (FI_RECV | FI_MSG);
		break;
	case ofi_op_read_rsp:
		rx_entry->cq_entry.flags = (FI_REMOTE_READ | FI_MSG);
		break;
	case ofi_op_write_async:
		rx_entry->cq_entry.flags = (FI_REMOTE_WRITE | FI_MSG);
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Unknown operation while %s\n", __func__);
		assert(0 && "Unknown operation");
	}

	return rx_entry;
}

struct rxr_rx_entry *rxr_ep_get_rx_entry(struct rxr_ep *ep,
					 const struct iovec *iov,
					 size_t iov_count, uint64_t tag,
					 uint64_t ignore, void *context,
					 fi_addr_t addr, uint32_t op,
					 uint64_t flags)
{
	struct rxr_rx_entry *rx_entry;

	rx_entry = ofi_buf_alloc(ep->rx_entry_pool);
	if (OFI_UNLIKELY(!rx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "RX entries exhausted\n");
		return NULL;
	}

#if ENABLE_DEBUG
	dlist_insert_tail(&rx_entry->rx_entry_entry, &ep->rx_entry_list);
#endif
	rx_entry = rxr_ep_rx_entry_init(ep, rx_entry, iov, iov_count, tag,
					ignore, context, addr, op, flags);
	rx_entry->state = RXR_RX_INIT;
	return rx_entry;
}

/*
 * Create a new rx_entry for an unexpected message. Store the packet for later
 * processing and put the rx_entry on the appropriate unexpected list.
 */
struct rxr_rx_entry *rxr_ep_get_new_unexp_rx_entry(struct rxr_ep *ep,
						   struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_pkt_entry *unexp_entry;
	struct rxr_rts_hdr *rts_pkt;
	uint32_t op;

	if (rxr_env.rx_copy_unexp && pkt_entry->type == RXR_PKT_ENTRY_POSTED) {
		unexp_entry = rxr_get_pkt_entry(ep, ep->rx_unexp_pkt_pool);
		if (OFI_UNLIKELY(!unexp_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unable to allocate rx_pkt_entry for unexp msg\n");
			return NULL;
		}
		rxr_copy_pkt_entry(ep, unexp_entry, pkt_entry,
				   RXR_PKT_ENTRY_UNEXP);
		rxr_release_rx_pkt_entry(ep, pkt_entry);
		ep->rx_bufs_to_post++;
	} else {
		unexp_entry = pkt_entry;
	}

	rts_pkt = rxr_get_rts_hdr(unexp_entry->pkt);

	if (rts_pkt->flags & RXR_TAGGED)
		op = ofi_op_tagged;
	else
		op = ofi_op_msg;

	rx_entry = rxr_ep_get_rx_entry(ep, NULL, 0, rts_pkt->tag, ~0, NULL,
				       unexp_entry->addr, op, 0);
	if (OFI_UNLIKELY(!rx_entry))
		return NULL;

	rx_entry->state = RXR_RX_UNEXP;
	rx_entry->total_len = rts_pkt->data_len;
	rx_entry->rxr_flags = rts_pkt->flags;
	rx_entry->unexp_rts_pkt = unexp_entry;

	if (op == ofi_op_tagged)
		dlist_insert_tail(&rx_entry->entry, &ep->rx_unexp_tagged_list);
	else
		dlist_insert_tail(&rx_entry->entry, &ep->rx_unexp_list);

	return rx_entry;
}

struct rxr_rx_entry *rxr_ep_split_rx_entry(struct rxr_ep *ep,
					   struct rxr_rx_entry *posted_entry,
					   struct rxr_rx_entry *consumer_entry,
					   struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rx_entry *rx_entry;
	struct rxr_rts_hdr *rts_pkt;
	size_t buf_len, consumed_len;

	rts_pkt = rxr_get_rts_hdr(pkt_entry->pkt);
	if (!consumer_entry) {
		rx_entry = rxr_ep_get_rx_entry(ep, posted_entry->iov,
					       posted_entry->iov_count,
					       rts_pkt->tag, 0, NULL,
					       pkt_entry->addr, ofi_op_msg,
					       posted_entry->fi_flags);
		if (OFI_UNLIKELY(!rx_entry))
			return NULL;

		FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
		       "Splitting into new multi_recv consumer rx_entry %d from rx_entry %d\n",
		       rx_entry->rx_id,
		       posted_entry->rx_id);
	} else {
		rx_entry = consumer_entry;
		memcpy(rx_entry->iov, posted_entry->iov,
		       sizeof(*posted_entry->iov) * posted_entry->iov_count);
		rx_entry->iov_count = posted_entry->iov_count;
	}

	buf_len = ofi_total_iov_len(rx_entry->iov,
				    rx_entry->iov_count);
	consumed_len = MIN(buf_len, rts_pkt->data_len);

	rx_entry->rxr_flags |= RXR_MULTI_RECV_CONSUMER;
	rx_entry->fi_flags |= FI_MULTI_RECV;
	rx_entry->master_entry = posted_entry;
	rx_entry->cq_entry.len = consumed_len;
	rx_entry->cq_entry.buf = rx_entry->iov[0].iov_base;
	rx_entry->cq_entry.op_context = posted_entry->cq_entry.op_context;
	rx_entry->cq_entry.flags = (FI_RECV | FI_MSG);

	ofi_consume_iov(posted_entry->iov, &posted_entry->iov_count,
			consumed_len);

	dlist_init(&rx_entry->multi_recv_entry);
	dlist_insert_tail(&rx_entry->multi_recv_entry,
			  &posted_entry->multi_recv_consumers);
	return rx_entry;
}

/* Post buf as undirected recv (FI_ADDR_UNSPEC) */
int rxr_ep_post_buf(struct rxr_ep *ep, uint64_t flags)
{
	struct fi_msg msg;
	struct iovec msg_iov;
	void *desc;
	struct rxr_pkt_entry *rx_pkt_entry;
	int ret = 0;

	rx_pkt_entry = rxr_get_pkt_entry(ep, ep->rx_pkt_pool);
	if (OFI_UNLIKELY(!rx_pkt_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Unable to allocate rx_pkt_entry\n");
		return -FI_ENOMEM;
	}

#if ENABLE_DEBUG
	dlist_insert_tail(&rx_pkt_entry->dbg_entry,
			  &ep->rx_posted_buf_list);
#endif
	rx_pkt_entry->x_entry = NULL;
	rx_pkt_entry->type = RXR_PKT_ENTRY_POSTED;

	msg_iov.iov_base = (void *)rxr_pkt_start(rx_pkt_entry);
	msg_iov.iov_len = ep->mtu_size;

	msg.msg_iov = &msg_iov;
	desc = rxr_ep_mr_local(ep) ? fi_mr_desc(rx_pkt_entry->mr) : NULL;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = FI_ADDR_UNSPEC;
	msg.context = rx_pkt_entry;
	msg.data = 0;

	ret = fi_recvmsg(ep->rdm_ep, &msg, flags);

	if (OFI_UNLIKELY(ret)) {
		rxr_release_rx_pkt_entry(ep, rx_pkt_entry);
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"failed to post buf %d (%s)\n", -ret,
			fi_strerror(-ret));
		return ret;
	}

	ep->posted_bufs++;
	return 0;
}

static int rxr_ep_match_unexp_msg(struct dlist_entry *item, const void *arg)
{
	const struct rxr_match_info *match_info = arg;
	struct rxr_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);

	return rxr_match_addr(match_info->addr, rx_entry->addr);
}

static int rxr_ep_match_unexp_tmsg(struct dlist_entry *item, const void *arg)
{
	const struct rxr_match_info *match_info = arg;
	struct rxr_rx_entry *rx_entry;

	rx_entry = container_of(item, struct rxr_rx_entry, entry);

	return rxr_match_addr(match_info->addr, rx_entry->addr) &&
	       rxr_match_tag(rx_entry->tag, match_info->ignore,
			     match_info->tag);
}

static int rxr_ep_handle_unexp_match(struct rxr_ep *ep,
				     struct rxr_rx_entry *rx_entry,
				     uint64_t tag, uint64_t ignore,
				     void *context, fi_addr_t addr,
				     uint32_t op, uint64_t flags)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_rts_hdr *rts_hdr;
	uint64_t bytes_left, len;
	int ret = 0;

	rx_entry->fi_flags = flags;
	rx_entry->ignore = ignore;
	rx_entry->state = RXR_RX_MATCHED;

	pkt_entry = rx_entry->unexp_rts_pkt;
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	rx_entry->cq_entry.op_context = context;
	/*
	 * we don't expect recv buf from application for discard,
	 * hence setting to NULL
	 */
	if (OFI_UNLIKELY(flags & FI_DISCARD)) {
		rx_entry->cq_entry.buf = NULL;
		rx_entry->cq_entry.len = rts_hdr->data_len;
	} else {
		rx_entry->cq_entry.buf = rx_entry->iov[0].iov_base;
		len = MIN(rx_entry->total_len,
			  ofi_total_iov_len(rx_entry->iov,
					    rx_entry->iov_count));
		rx_entry->cq_entry.len = len;
	}

	rx_entry->cq_entry.flags = (FI_RECV | FI_MSG);

	if (op == ofi_op_tagged) {
		rx_entry->cq_entry.flags |= FI_TAGGED;
		rx_entry->cq_entry.tag = rx_entry->tag;
		rx_entry->ignore = ignore;
	} else {
		rx_entry->cq_entry.tag = 0;
		rx_entry->ignore = ~0;
	}

	rxr_cq_recv_rts_data(ep, rx_entry, rts_hdr);

	/*
	 * TODO: Unsure how to handle fi_cq_msg_entry when writing completion
	 * events in the unexpected path. Right now this field is unused. If
	 * that changes we'll need to parse the flags as we get completion
	 * events from the provider in the recv path and save the flags in the
	 * rx_entry for the unexp message path to use when the app calls recv.
	 */
	if (rx_entry->total_len - rx_entry->bytes_done == 0) {
		ret = rxr_cq_handle_rx_completion(ep, NULL,
						  pkt_entry, rx_entry);
		if (!ret)
			rxr_release_rx_entry(ep, rx_entry);
		return 0;
	}

	rx_entry->state = RXR_RX_RECV;
#if ENABLE_DEBUG
	dlist_insert_tail(&rx_entry->rx_pending_entry, &ep->rx_pending_list);
	ep->rx_pending++;
#endif
	bytes_left = rx_entry->total_len - rx_entry->bytes_done;
	if (!rx_entry->window && bytes_left > 0)
		ret = rxr_ep_post_cts_or_queue(ep, rx_entry, bytes_left);

	if (pkt_entry->type == RXR_PKT_ENTRY_POSTED)
		ep->rx_bufs_to_post++;
	rxr_release_rx_pkt_entry(ep, pkt_entry);
	return ret;
}

/*
 * Search unexpected list for matching message and process it if found.
 *
 * Returns 0 if the message is processed, -FI_ENOMSG if no match is found.
 */
static int rxr_ep_check_unexp_msg_list(struct rxr_ep *ep,
				       const struct iovec *iov,
				       size_t iov_count, uint64_t tag,
				       uint64_t ignore, void *context,
				       fi_addr_t addr, uint32_t op,
				       uint64_t flags,
				       struct rxr_rx_entry *posted_entry)
{
	struct rxr_match_info match_info;
	struct dlist_entry *match;
	struct rxr_rx_entry *rx_entry;
	int ret;

	if (op == ofi_op_tagged) {
		match_info.addr = addr;
		match_info.tag = tag;
		match_info.ignore = ignore;
		match = dlist_remove_first_match(&ep->rx_unexp_tagged_list,
						 &rxr_ep_match_unexp_tmsg,
						 (void *)&match_info);
	} else {
		match_info.addr = addr;
		match = dlist_remove_first_match(&ep->rx_unexp_list,
						 &rxr_ep_match_unexp_msg,
						 (void *)&match_info);
	}

	if (!match)
		return -FI_ENOMSG;

	rx_entry = container_of(match, struct rxr_rx_entry, entry);

	/*
	 * Initialize the matched entry as a multi-recv consumer if the posted
	 * buffer is a multi-recv buffer.
	 */
	if (posted_entry) {
		/*
		 * rxr_ep_split_rx_entry will setup rx_entry iov and count
		 */
		rx_entry = rxr_ep_split_rx_entry(ep, posted_entry, rx_entry,
						 rx_entry->unexp_rts_pkt);
		if (OFI_UNLIKELY(!rx_entry)) {
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			return -FI_ENOBUFS;
		}
	} else {
		memcpy(rx_entry->iov, iov, sizeof(*rx_entry->iov) * iov_count);
		rx_entry->iov_count = iov_count;
	}

	FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
	       "Match found in unexp list for a posted recv msg_id: %" PRIu32
	       " total_len: %" PRIu64 " tag: %lx\n",
	       rx_entry->msg_id, rx_entry->total_len, rx_entry->tag);

	ret = rxr_ep_handle_unexp_match(ep, rx_entry, tag, ignore,
					context, addr, op, flags);
	return ret;
}

static ssize_t rxr_ep_discard_trecv(struct rxr_ep *ep,
				    struct rxr_rx_entry *rx_entry,
				    const struct fi_msg_tagged *msg,
				    int64_t flags)
{
	int ret;

	if ((flags & FI_DISCARD) && !(flags & (FI_PEEK | FI_CLAIM)))
		return -FI_EINVAL;

	rx_entry->fi_flags |= FI_DISCARD;
	rx_entry->rxr_flags |= RXR_RECV_CANCEL;
	ret = ofi_cq_write(ep->util_ep.rx_cq, msg->context,
			   FI_TAGGED | FI_RECV | FI_MSG,
			   0, NULL, rx_entry->cq_entry.data,
			   rx_entry->cq_entry.tag);
	rxr_rm_rx_cq_check(ep, ep->util_ep.rx_cq);
	return ret;
}

static ssize_t rxr_ep_claim_trecv(struct fid_ep *ep_fid,
				  const struct fi_msg_tagged *msg,
				  int64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *ep;
	struct rxr_rx_entry *rx_entry;
	struct fi_context *context;

	ep = container_of(ep_fid, struct rxr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.lock);

	context = (struct fi_context *)msg->context;
	rx_entry = (struct rxr_rx_entry *)context->internal[0];

	if (flags & FI_DISCARD) {
		ret = rxr_ep_discard_trecv(ep, rx_entry, msg, flags);
		if (OFI_UNLIKELY(ret))
			goto out;
	}

	/*
	 * Handle unexp match entry even for discard entry as we are sinking
	 * messages for that case
	 */
	memcpy(rx_entry->iov, msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	rx_entry->iov_count = msg->iov_count;

	ret = rxr_ep_handle_unexp_match(ep, rx_entry, msg->tag,
					msg->ignore, msg->context,
					msg->addr, ofi_op_tagged, flags);

out:
	fastlock_release(&ep->util_ep.lock);
	return ret;
}

static ssize_t rxr_ep_peek_trecv(struct fid_ep *ep_fid,
				 const struct fi_msg_tagged *msg,
				 uint64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *ep;
	struct dlist_entry *match;
	struct rxr_match_info match_info;
	struct rxr_rx_entry *rx_entry;
	struct fi_context *context;
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_rts_hdr *rts_hdr;

	ep = container_of(ep_fid, struct rxr_ep, util_ep.ep_fid.fid);

	fastlock_acquire(&ep->util_ep.lock);

	rxr_ep_progress_internal(ep);
	match_info.addr = msg->addr;
	match_info.tag = msg->tag;
	match_info.ignore = msg->ignore;

	match = dlist_find_first_match(&ep->rx_unexp_tagged_list,
				       &rxr_ep_match_unexp_tmsg,
				       (void *)&match_info);
	if (!match) {
		FI_DBG(&rxr_prov, FI_LOG_EP_CTRL,
		       "Message not found addr: %" PRIu64
		       " tag: %lx ignore %lx\n", msg->addr, msg->tag,
		       msg->ignore);
		ret = ofi_cq_write_error_peek(ep->util_ep.rx_cq, msg->tag,
					      msg->context);
		goto out;
	}

	rx_entry = container_of(match, struct rxr_rx_entry, entry);
	context = (struct fi_context *)msg->context;
	if (flags & FI_CLAIM) {
		context->internal[0] = rx_entry;
		dlist_remove(match);
	} else if (flags & FI_DISCARD) {
		dlist_remove(match);

		ret = rxr_ep_discard_trecv(ep, rx_entry, msg, flags);
		if (ret)
			goto out;

		memcpy(rx_entry->iov, msg->msg_iov,
		       sizeof(*msg->msg_iov) * msg->iov_count);
		rx_entry->iov_count = msg->iov_count;

		ret = rxr_ep_handle_unexp_match(ep, rx_entry,
						msg->tag, msg->ignore,
						msg->context, msg->addr,
						ofi_op_tagged, flags);

		goto out;
	}

	pkt_entry = rx_entry->unexp_rts_pkt;
	rts_hdr = rxr_get_rts_hdr(pkt_entry->pkt);

	if (rts_hdr->flags & RXR_REMOTE_CQ_DATA) {
		rx_entry->cq_entry.data =
			rxr_get_ctrl_cq_pkt(rts_hdr)->hdr.cq_data;
		rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
	}

	if (ep->util_ep.caps & FI_SOURCE)
		ret = ofi_cq_write_src(ep->util_ep.rx_cq, context,
				       FI_TAGGED | FI_RECV,
				       rts_hdr->data_len, NULL,
				       rx_entry->cq_entry.data, rts_hdr->tag,
				       rx_entry->addr);
	else
		ret = ofi_cq_write(ep->util_ep.rx_cq, context,
				   FI_TAGGED | FI_RECV,
				   rts_hdr->data_len, NULL,
				   rx_entry->cq_entry.data, rts_hdr->tag);
	rxr_rm_rx_cq_check(ep, ep->util_ep.rx_cq);
out:
	fastlock_release(&ep->util_ep.lock);
	return ret;
}

static ssize_t rxr_multi_recv(struct rxr_ep *rxr_ep, const struct iovec *iov,
			      size_t iov_count, fi_addr_t addr, uint64_t tag,
			      uint64_t ignore, void *context, uint32_t op,
			      uint64_t flags)
{
	struct rxr_rx_entry *rx_entry;
	int ret = 0;

	if ((ofi_total_iov_len(iov, iov_count)
	     < rxr_ep->min_multi_recv_size) || op != ofi_op_msg)
		return -FI_EINVAL;

	/*
	 * Always get new rx_entry of type RXR_MULTI_RECV_POSTED when in the
	 * multi recv path. The posted entry will not be used for receiving
	 * messages but will be used for tracking the application's buffer and
	 * when to write the completion to release the buffer.
	 */
	rx_entry = rxr_ep_get_rx_entry(rxr_ep, iov, iov_count, tag,
				       ignore, context,
				       (rxr_ep->util_ep.caps &
					FI_DIRECTED_RECV) ? addr :
				       FI_ADDR_UNSPEC, op, flags);
	if (OFI_UNLIKELY(!rx_entry)) {
		rxr_ep_progress_internal(rxr_ep);
		return -FI_EAGAIN;
	}

	rx_entry->rxr_flags |= RXR_MULTI_RECV_POSTED;
	dlist_init(&rx_entry->multi_recv_consumers);
	dlist_init(&rx_entry->multi_recv_entry);

	while (!dlist_empty(&rxr_ep->rx_unexp_list)) {
		ret = rxr_ep_check_unexp_msg_list(rxr_ep, NULL, 0, tag,
						  ignore, context,
						  (rxr_ep->util_ep.caps
						   & FI_DIRECTED_RECV) ?
						   addr : FI_ADDR_UNSPEC,
						  op, flags, rx_entry);

		if (!rxr_multi_recv_buffer_available(rxr_ep, rx_entry)) {
			/*
			 * Multi recv buffer consumed by short, unexp messages,
			 * free posted rx_entry.
			 */
			if (rxr_multi_recv_buffer_complete(rxr_ep, rx_entry))
				rxr_release_rx_entry(rxr_ep, rx_entry);
			/*
			 * Multi recv buffer has been consumed, but waiting on
			 * long msg completion. Last msg completion will free
			 * posted rx_entry.
			 */
			if (ret != -FI_ENOMSG || ret != 0)
				return ret;
			return 0;
		}

		if (ret == -FI_ENOMSG) {
			ret = 0;
			break;
		}

		/*
		 * Error was encountered when processing unexpected messages,
		 * but there is buffer space available. Add the posted entry to
		 * the rx_list.
		 */
		if (ret)
			break;
	}

	dlist_insert_tail(&rx_entry->entry, &rxr_ep->rx_list);
	return ret;
}
/*
 * create a rx entry and verify in unexpected message list
 * else add to posted recv list
 */
static ssize_t rxr_recv(struct fid_ep *ep, const struct iovec *iov,
			size_t iov_count, fi_addr_t addr, uint64_t tag,
			uint64_t ignore, void *context, uint32_t op,
			uint64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *rxr_ep;
	struct dlist_entry *unexp_list;
	struct rxr_rx_entry *rx_entry;
	uint64_t rx_op_flags;

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s: iov_len: %lu tag: %lx ignore: %lx op: %x flags: %lx\n",
	       __func__, ofi_total_iov_len(iov, iov_count), tag, ignore,
	       op, flags);

	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);

	assert(iov_count <= rxr_ep->rx_iov_limit);

	rxr_perfset_start(rxr_ep, perf_rxr_recv);

	assert(rxr_ep->util_ep.rx_msg_flags == 0 || rxr_ep->util_ep.rx_msg_flags == FI_COMPLETION);
	rx_op_flags = rxr_ep->util_ep.rx_op_flags;
	if (rxr_ep->util_ep.rx_msg_flags == 0)
		rx_op_flags &= ~FI_COMPLETION;
	flags = flags | rx_op_flags;

	fastlock_acquire(&rxr_ep->util_ep.lock);
	if (OFI_UNLIKELY(is_rx_res_full(rxr_ep))) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (flags & FI_MULTI_RECV) {
		ret = rxr_multi_recv(rxr_ep, iov, iov_count, addr, tag, ignore,
				     context, op, flags);
		goto out;
	}

	unexp_list = (op == ofi_op_tagged) ? &rxr_ep->rx_unexp_tagged_list :
		     &rxr_ep->rx_unexp_list;

	if (!dlist_empty(unexp_list)) {
		ret = rxr_ep_check_unexp_msg_list(rxr_ep, iov, iov_count, tag,
						  ignore, context,
						  (rxr_ep->util_ep.caps
						   & FI_DIRECTED_RECV) ?
						   addr : FI_ADDR_UNSPEC,
						  op, flags, NULL);

		if (ret != -FI_ENOMSG)
			goto out;
		ret = 0;
	}

	rx_entry = rxr_ep_get_rx_entry(rxr_ep, iov, iov_count, tag,
				       ignore, context,
				       (rxr_ep->util_ep.caps &
					FI_DIRECTED_RECV) ? addr :
				       FI_ADDR_UNSPEC, op, flags);

	if (OFI_UNLIKELY(!rx_entry)) {
		ret = -FI_EAGAIN;
		rxr_ep_progress_internal(rxr_ep);
		goto out;
	}

	if (op == ofi_op_tagged)
		dlist_insert_tail(&rx_entry->entry, &rxr_ep->rx_tagged_list);
	else
		dlist_insert_tail(&rx_entry->entry, &rxr_ep->rx_list);

out:
	fastlock_release(&rxr_ep->util_ep.lock);

	rxr_perfset_end(rxr_ep, perf_rxr_recv);
	return ret;
}

static ssize_t rxr_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	return rxr_recv(ep_fid, msg->msg_iov, msg->iov_count, msg->addr,
			0, 0, msg->context, ofi_op_msg, flags);
}

static ssize_t rxr_ep_recv(struct fid_ep *ep, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;

	return rxr_ep_recvmsg(ep, &msg, 0);
}

static ssize_t rxr_ep_recvv(struct fid_ep *ep, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t src_addr,
			    void *context)
{
	struct fi_msg msg;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.context = context;
	msg.data = 0;

	return rxr_ep_recvmsg(ep, &msg, 0);
}


void rxr_generic_tx_entry_init(struct rxr_tx_entry *tx_entry, const struct iovec *iov, size_t iov_count,
			       const struct fi_rma_iov *rma_iov, size_t rma_iov_count,
			       fi_addr_t addr, uint64_t tag, uint64_t data, void *context,
			       uint32_t op, uint64_t flags)
{
	tx_entry->type = RXR_TX_ENTRY;
	tx_entry->tx_id = ofi_buf_index(tx_entry);
	tx_entry->state = RXR_TX_RTS;
	tx_entry->addr = addr;
	tx_entry->tag = tag;

	tx_entry->send_flags = 0;
	tx_entry->bytes_acked = 0;
	tx_entry->bytes_sent = 0;
	tx_entry->window = 0;
	tx_entry->total_len = ofi_total_iov_len(iov, iov_count);
	tx_entry->iov_count = iov_count;
	tx_entry->iov_index = 0;
	tx_entry->iov_mr_start = 0;
	tx_entry->iov_offset = 0;
	tx_entry->msg_id = ~0;
	dlist_init(&tx_entry->queued_pkts);

	memcpy(&tx_entry->iov[0], iov, sizeof(*iov) * iov_count);

	/* cq_entry on completion */
	tx_entry->cq_entry.op_context = context;
	tx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	if (OFI_LIKELY(tx_entry->cq_entry.len > 0))
		tx_entry->cq_entry.buf = iov[0].iov_base;
	else
		tx_entry->cq_entry.buf = NULL;

	tx_entry->cq_entry.data = data;
	tx_entry->cq_entry.tag = 0;
	switch (op) {
	case ofi_op_tagged:
		tx_entry->cq_entry.flags = FI_TRANSMIT | FI_MSG | FI_TAGGED;
		tx_entry->cq_entry.tag = tag;
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
	default:
		FI_WARN(&rxr_prov, FI_LOG_CQ, "invalid operation type in %s\n",
			__func__);
		assert(0);
	}

	if (tx_entry->cq_entry.flags & FI_RMA) {
		assert(rma_iov_count>0);
		assert(rma_iov);
		tx_entry->rma_iov_count = rma_iov_count;
		memcpy(tx_entry->rma_iov, rma_iov, sizeof(struct fi_rma_iov) * rma_iov_count);
	}
}

/* create a new tx entry */
struct rxr_tx_entry *rxr_ep_tx_entry_init(struct rxr_ep *rxr_ep, const struct iovec *iov, size_t iov_count,
					  const struct fi_rma_iov *rma_iov, size_t rma_iov_count,
					  fi_addr_t addr, uint64_t tag, uint64_t data, void *context,
					  uint32_t op, uint64_t flags)
{
	struct rxr_tx_entry *tx_entry;
	uint64_t tx_op_flags;

	tx_entry = ofi_buf_alloc(rxr_ep->tx_entry_pool);
	if (OFI_UNLIKELY(!tx_entry)) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "TX entries exhausted.\n");
		return NULL;
	}

#if ENABLE_DEBUG
	dlist_insert_tail(&tx_entry->tx_entry_entry, &rxr_ep->tx_entry_list);
#endif

	rxr_generic_tx_entry_init(tx_entry, iov, iov_count, rma_iov, rma_iov_count,
				  addr, tag, data, context, op, flags);

	assert(rxr_ep->util_ep.tx_msg_flags == 0 || rxr_ep->util_ep.tx_msg_flags == FI_COMPLETION);
	tx_op_flags = rxr_ep->util_ep.tx_op_flags;
	if (rxr_ep->util_ep.tx_msg_flags == 0)
		tx_op_flags &= ~FI_COMPLETION;
	tx_entry->fi_flags = flags | tx_op_flags;

	return tx_entry;
}

/*
 * Copies all consecutive small iov's into one buffer. If the function reaches
 * an iov greater than the max memcpy size, it will end, only copying up to
 * that iov.
 */
static size_t rxr_copy_from_iov(void *buf, uint64_t remaining_len,
				struct rxr_tx_entry *tx_entry)
{
	struct iovec *tx_iov = tx_entry->iov;
	uint64_t done = 0, len;

	while (tx_entry->iov_index < tx_entry->iov_count &&
	       done < remaining_len) {
		len = tx_iov[tx_entry->iov_index].iov_len;
		if (tx_entry->mr[tx_entry->iov_index])
			break;

		len -= tx_entry->iov_offset;

		/*
		 * If the amount to be written surpasses the remaining length,
		 * copy up to the remaining length and return, else copy the
		 * entire iov and continue.
		 */
		if (done + len > remaining_len) {
			len = remaining_len - done;
			memcpy((char *)buf + done,
			       (char *)tx_iov[tx_entry->iov_index].iov_base +
			       tx_entry->iov_offset, len);
			tx_entry->iov_offset += len;
			done += len;
			break;
		}
		memcpy((char *)buf + done,
		       (char *)tx_iov[tx_entry->iov_index].iov_base +
		       tx_entry->iov_offset, len);
		tx_entry->iov_index++;
		tx_entry->iov_offset = 0;
		done += len;
	}
	return done;
}

ssize_t rxr_ep_send_msg(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry,
			const struct fi_msg *msg, uint64_t flags)
{
	struct rxr_peer *peer;
	size_t ret;

	assert(ep->tx_pending <= ep->max_outstanding_tx);

	if (ep->tx_pending == ep->max_outstanding_tx)
		return -FI_EAGAIN;

	peer = rxr_ep_get_peer(ep, pkt_entry->addr);
	if (peer->rnr_state & RXR_PEER_IN_BACKOFF)
		return -FI_EAGAIN;

#if ENABLE_DEBUG
	dlist_insert_tail(&pkt_entry->dbg_entry, &ep->tx_pkt_list);
#ifdef ENABLE_RXR_PKT_DUMP
	rxr_ep_print_pkt("Sent", ep, (struct rxr_base_hdr *)pkt_entry->pkt);
#endif
#endif
	ret = fi_sendmsg(ep->rdm_ep, msg, flags);

	if (OFI_LIKELY(!ret)) {
		ep->tx_pending++;
#if ENABLE_DEBUG
		ep->sends++;
#endif
	}

	return ret;
}

static ssize_t rxr_ep_send_data_pkt_entry(struct rxr_ep *ep,
					  struct rxr_tx_entry *tx_entry,
					  struct rxr_pkt_entry *pkt_entry,
					  struct rxr_data_pkt *data_pkt)
{
	uint64_t payload_size;

	payload_size = MIN(tx_entry->total_len - tx_entry->bytes_sent,
			   ep->max_data_payload_size);
	payload_size = MIN(payload_size, tx_entry->window);
	data_pkt->hdr.seg_size = payload_size;

	pkt_entry->pkt_size = ofi_copy_from_iov(data_pkt->data,
						payload_size,
						tx_entry->iov,
						tx_entry->iov_count,
						tx_entry->bytes_sent);
	assert(pkt_entry->pkt_size == payload_size);

	pkt_entry->pkt_size += RXR_DATA_HDR_SIZE;
	pkt_entry->addr = tx_entry->addr;

	return rxr_ep_send_pkt_flags(ep, pkt_entry, tx_entry->addr,
				     tx_entry->send_flags);
}

/* If mr local is not set, will skip copying and only send user buffers */
static ssize_t rxr_ep_mr_send_data_pkt_entry(struct rxr_ep *ep,
					     struct rxr_tx_entry *tx_entry,
					     struct rxr_pkt_entry *pkt_entry,
					     struct rxr_data_pkt *data_pkt)
{
	/* The user's iov */
	struct iovec *tx_iov = tx_entry->iov;
	/* The constructed iov to be passed to sendv
	 * and corresponding fid_mrs
	 */
	struct iovec iov[ep->core_iov_limit];
	struct fid_mr *mr[ep->core_iov_limit];
	/* Constructed iov's total size */
	uint64_t payload_size = 0;
	/* pkt_entry offset to write data into */
	uint64_t pkt_used = 0;
	/* Remaining size that can fit in the constructed iov */
	uint64_t remaining_len = MIN(tx_entry->window,
				     ep->max_data_payload_size);
	/* The constructed iov's index */
	size_t i = 0;
	size_t len = 0;

	ssize_t ret;

	/* Assign packet header in constructed iov */
	iov[i].iov_base = rxr_pkt_start(pkt_entry);
	iov[i].iov_len = RXR_DATA_HDR_SIZE;
	mr[i] = rxr_ep_mr_local(ep) ? fi_mr_desc(pkt_entry->mr) : NULL;
	i++;

	/*
	 * Loops until payload size is at max, all user iovs are sent, the
	 * constructed iov count is greater than the core iov limit, or the tx
	 * entry window is exhausted.  Each iteration fills one entry of the
	 * iov to be sent.
	 */
	while (tx_entry->iov_index < tx_entry->iov_count &&
	       remaining_len > 0 && i < ep->core_iov_limit) {
		/* If the iov was pre registered after the RTS */
		if (!rxr_ep_mr_local(ep) ||
		    tx_entry->mr[tx_entry->iov_index]) {
			iov[i].iov_base =
				(char *)tx_iov[tx_entry->iov_index].iov_base +
				tx_entry->iov_offset;
			mr[i] = rxr_ep_mr_local(ep) ?
				fi_mr_desc(tx_entry->mr[tx_entry->iov_index]) :
				NULL;

			len = tx_iov[tx_entry->iov_index].iov_len
			      - tx_entry->iov_offset;
			if (len > remaining_len) {
				len = remaining_len;
				tx_entry->iov_offset += len;
			} else {
				tx_entry->iov_index++;
				tx_entry->iov_offset = 0;
			}
			iov[i].iov_len = len;
		} else {
			/*
			 * Copies any consecutive small iov's, returning size
			 * written while updating iov index and offset
			 */
			len = rxr_copy_from_iov((char *)data_pkt->data +
						 pkt_used,
						 remaining_len,
						 tx_entry);

			iov[i].iov_base = (char *)data_pkt->data + pkt_used;
			iov[i].iov_len = len;
			mr[i] = fi_mr_desc(pkt_entry->mr);
			pkt_used += len;
		}
		payload_size += len;
		remaining_len -= len;
		i++;
	}
	data_pkt->hdr.seg_size = (uint16_t)payload_size;
	pkt_entry->pkt_size = payload_size + RXR_DATA_HDR_SIZE;
	pkt_entry->addr = tx_entry->addr;

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "Sending an iov count, %zu with payload size: %lu.\n",
	       i, payload_size);
	ret = rxr_ep_sendv_pkt(ep, pkt_entry, tx_entry->addr,
			       (const struct iovec *)iov,
			       (void **)mr, i, tx_entry->send_flags);
	return ret;
}

ssize_t rxr_ep_post_data(struct rxr_ep *rxr_ep,
			 struct rxr_tx_entry *tx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	struct rxr_data_pkt *data_pkt;
	ssize_t ret;

	pkt_entry = rxr_get_pkt_entry(rxr_ep, rxr_ep->tx_pkt_pool);

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

	if (efa_mr_cache_enable) {
		ret = rxr_ep_mr_send_data_pkt_entry(rxr_ep, tx_entry, pkt_entry,
						    data_pkt);
	} else {
		ret = rxr_ep_send_data_pkt_entry(rxr_ep, tx_entry, pkt_entry,
						 data_pkt);
	}

	if (OFI_UNLIKELY(ret)) {
		rxr_release_tx_pkt_entry(rxr_ep, pkt_entry);
		return ret;
	}
	data_pkt = rxr_get_data_pkt(pkt_entry->pkt);
	tx_entry->bytes_sent += data_pkt->hdr.seg_size;
	tx_entry->window -= data_pkt->hdr.seg_size;

	return ret;
}

ssize_t rxr_ep_post_readrsp(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	ssize_t ret;
	size_t data_len;

	pkt_entry = rxr_get_pkt_entry(ep, ep->tx_pkt_pool);
	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	rxr_ep_init_readrsp_pkt_entry(ep, tx_entry, pkt_entry);
	ret = rxr_ep_send_pkt(ep, pkt_entry, tx_entry->addr);
	if (OFI_UNLIKELY(ret)) {
		rxr_release_tx_pkt_entry(ep, pkt_entry);
		FI_WARN(&rxr_prov, FI_LOG_CQ,
			"Failed to send a read response packet: ret %zd\n", ret);
		return ret;
	}

	data_len = rxr_get_readrsp_hdr(pkt_entry->pkt)->seg_size;
	tx_entry->bytes_sent += data_len;
	tx_entry->window -= data_len;
	assert(tx_entry->window >= 0);
	assert(tx_entry->bytes_sent <= tx_entry->total_len);
	assert(tx_entry->bytes_acked == 0);
	return 0;
}

void rxr_ep_calc_cts_window_credits(struct rxr_ep *ep, uint32_t max_window, uint64_t size,
				    int *window, int *credits)
{
	*credits = ofi_div_ceil(size, ep->max_data_payload_size);
	*credits = MIN(MIN(ep->available_data_bufs, ep->posted_bufs),
		       MIN(*credits, max_window));
	*window = MIN(size, *credits * ep->max_data_payload_size);
}

void rxr_ep_init_cts_pkt_entry(struct rxr_ep *ep,
			       struct rxr_rx_entry *rx_entry,
			       struct rxr_pkt_entry *pkt_entry,
			       uint32_t max_window,
			       uint64_t size,
			       int *credits)
{
	int window = 0;
	struct rxr_cts_hdr *cts_hdr;

	cts_hdr = (struct rxr_cts_hdr *)pkt_entry->pkt;

	cts_hdr->type = RXR_CTS_PKT;
	cts_hdr->version = RXR_PROTOCOL_VERSION;
	cts_hdr->flags = 0;

	if (rx_entry->cq_entry.flags & FI_READ)
		cts_hdr->flags |= RXR_READ_REQ;

	cts_hdr->tx_id = rx_entry->tx_id;
	cts_hdr->rx_id = rx_entry->rx_id;

	rxr_ep_calc_cts_window_credits(ep, max_window, size, &window, credits);
	cts_hdr->window = window;

	pkt_entry->pkt_size = RXR_CTS_HDR_SIZE;
	pkt_entry->addr = rx_entry->addr;
	pkt_entry->x_entry = (void *)rx_entry;
}

void rxr_ep_init_connack_pkt_entry(struct rxr_ep *ep,
				   struct rxr_pkt_entry *pkt_entry,
				   fi_addr_t addr)
{
	struct rxr_connack_hdr *connack_hdr;

	connack_hdr = (struct rxr_connack_hdr *)pkt_entry->pkt;

	connack_hdr->type = RXR_CONNACK_PKT;
	connack_hdr->version = RXR_PROTOCOL_VERSION;
	connack_hdr->flags = 0;

	pkt_entry->pkt_size = RXR_CONNACK_HDR_SIZE;
	pkt_entry->addr = addr;
}

void rxr_ep_init_readrsp_pkt_entry(struct rxr_ep *ep, struct rxr_tx_entry *tx_entry,
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
}

/* Initialize RTS packet */
void rxr_init_rts_pkt_entry(struct rxr_ep *ep,
			    struct rxr_tx_entry *tx_entry,
			    struct rxr_pkt_entry *pkt_entry)
{
	struct rxr_rts_hdr *rts_hdr;
	struct rxr_peer *peer;
	char *data, *src;
	uint64_t data_len;
	size_t mtu = ep->mtu_size;
	int rmalen = 0;

	rts_hdr = (struct rxr_rts_hdr *)pkt_entry->pkt;
	peer = rxr_ep_get_peer(ep, tx_entry->addr);

	rts_hdr->type = RXR_RTS_PKT;
	rts_hdr->version = RXR_PROTOCOL_VERSION;
	rts_hdr->tag = tx_entry->tag;

	rts_hdr->data_len = tx_entry->total_len;
	rts_hdr->tx_id = tx_entry->tx_id;
	rts_hdr->msg_id = tx_entry->msg_id;

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

	rts_hdr->rma_iov_count = 0;
	if (tx_entry->cq_entry.flags & FI_RMA) {
		if (tx_entry->cq_entry.flags & FI_WRITE) {
			rts_hdr->flags |= RXR_WRITE;
		} else {
			assert(tx_entry->cq_entry.flags & FI_READ);
			rts_hdr->flags |= RXR_READ_REQ;
		}

		rmalen = tx_entry->rma_iov_count * sizeof(struct fi_rma_iov);
		rts_hdr->rma_iov_count = tx_entry->rma_iov_count;
		memcpy(src, tx_entry->rma_iov, rmalen);
		src += rmalen;
		pkt_entry->pkt_size += rmalen;
	}

	/*
	 * currently copying for both INJECT and SEND,
	 * need to optimize for SEND (small & large) messages
	 */
	if (rts_hdr->flags & RXR_READ_REQ) {
		/* no data to send, but need to send rx_id and window */
		memcpy(src, &tx_entry->rma_loc_rx_id, sizeof(uint64_t));
		src += sizeof(uint64_t);
		pkt_entry->pkt_size += sizeof(uint64_t);
		memcpy(src, &tx_entry->rma_window, sizeof(uint64_t));
		src += sizeof(uint64_t);
		pkt_entry->pkt_size += sizeof(uint64_t);
	} else {
		data = src;
		data_len = ofi_copy_from_iov(data, mtu - pkt_entry->pkt_size,
					     tx_entry->iov, tx_entry->iov_count, 0);
		assert(data_len == rxr_get_rts_data_size(ep, rts_hdr));

		pkt_entry->pkt_size += data_len;
	}

	assert(pkt_entry->pkt_size <= mtu);
	pkt_entry->addr = tx_entry->addr;
	pkt_entry->x_entry = (void *)tx_entry;

	if (tx_entry->cq_entry.flags & FI_TAGGED)
		rts_hdr->flags |= RXR_TAGGED;
}

static void rxr_inline_mr_reg(struct rxr_domain *rxr_domain,
			      struct rxr_tx_entry *tx_entry,
			      size_t index)
{
	ssize_t ret;

	tx_entry->iov_mr_start = index;
	while (index < tx_entry->iov_count) {
		if (tx_entry->iov[index].iov_len > rxr_env.max_memcpy_size) {
			ret = fi_mr_reg(rxr_domain->rdm_domain,
					tx_entry->iov[index].iov_base,
					tx_entry->iov[index].iov_len,
					FI_SEND, 0, 0, 0,
					&tx_entry->mr[index], NULL);
			if (ret)
				tx_entry->mr[index] = NULL;
		}
		index++;
	}

	return;
}

/* Post request to send */
static size_t rxr_ep_post_rts(struct rxr_ep *rxr_ep, struct rxr_tx_entry *tx_entry)
{
	struct rxr_pkt_entry *pkt_entry;
	ssize_t ret;
	uint64_t data_sent, offset;
	int i;

	pkt_entry = rxr_get_pkt_entry(rxr_ep, rxr_ep->tx_pkt_pool);

	if (OFI_UNLIKELY(!pkt_entry))
		return -FI_EAGAIN;

	rxr_init_rts_pkt_entry(rxr_ep, tx_entry, pkt_entry);

	ret = rxr_ep_send_pkt(rxr_ep, pkt_entry, tx_entry->addr);
	if (OFI_UNLIKELY(ret)) {
		rxr_release_tx_pkt_entry(rxr_ep, pkt_entry);
		return ret;
	}

	if (tx_entry->cq_entry.flags & FI_READ) {
		tx_entry->bytes_sent = 0;
		assert(tx_entry->state == RXR_TX_RTS ||
		       tx_entry->state == RXR_TX_QUEUED_RTS);
		tx_entry->state = RXR_TX_WAIT_READ_FINISH;
		return 0;
	}

	data_sent = rxr_get_rts_data_size(rxr_ep, rxr_get_rts_hdr(pkt_entry->pkt));

	tx_entry->bytes_sent += data_sent;

	if (!(efa_mr_cache_enable && tx_entry->total_len > data_sent))
		return ret;

	/* Set the iov index and iov offset from bytes sent */
	offset = data_sent;
	for (i = 0; i < tx_entry->iov_count; i++) {
		if (offset >= tx_entry->iov[i].iov_len) {
			offset -= tx_entry->iov[i].iov_len;
		} else {
			tx_entry->iov_index = i;
			tx_entry->iov_offset = offset;
			break;
		}
	}

	if (rxr_ep_mr_local(rxr_ep))
		rxr_inline_mr_reg(rxr_ep_domain(rxr_ep), tx_entry, i);

	return 0;
}



/* Generic send */
ssize_t rxr_tx(struct fid_ep *ep, const struct iovec *iov, size_t iov_count,
	       const struct fi_rma_iov *rma_iov, size_t rma_iov_count,
	       fi_addr_t addr, uint64_t tag, uint64_t data, void *context,
	       uint32_t op, uint64_t flags)
{
	struct rxr_ep *rxr_ep;
	ssize_t ret;
	struct rxr_tx_entry *tx_entry;
	struct rxr_peer *peer;

	FI_DBG(&rxr_prov, FI_LOG_EP_DATA,
	       "%s: iov_len: %lu tag: %lx op: %x flags: %lx\n",
	       __func__, ofi_total_iov_len(iov, iov_count), tag, op, flags);

	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);

	assert(iov_count <= rxr_ep->tx_iov_limit);

	rxr_perfset_start(rxr_ep, perf_rxr_tx);

	fastlock_acquire(&rxr_ep->util_ep.lock);

	if (OFI_UNLIKELY(is_tx_res_full(rxr_ep))) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry = rxr_ep_tx_entry_init(rxr_ep, iov, iov_count,
					rma_iov, rma_iov_count,
					addr, tag, data, context,
					op, flags);

	if (OFI_UNLIKELY(!tx_entry)) {
		ret = -FI_EAGAIN;
		rxr_ep_progress_internal(rxr_ep);
		goto out;
	}

	peer = rxr_ep_get_peer(rxr_ep, addr);
	tx_entry->msg_id = (peer->next_msg_id != ~0) ?
			    peer->next_msg_id++ : ++peer->next_msg_id;

	if (op == ofi_op_read_req) {
		int ignore = ~0;
		struct rxr_rx_entry *rx_entry = NULL;
		int credits = 0;
		int window = 0;
		/* this rx_entry works same as a receiving rx_entry thus
		 * we use ofi_op_msg for its op.
		 * it does not write a rx completion.
		 */
		rx_entry = rxr_ep_get_rx_entry(rxr_ep, iov, iov_count, tag,
					       ignore, context,
					       addr, ofi_op_msg, 0);
		if (!rx_entry) {
			rxr_release_tx_entry(rxr_ep, tx_entry);
			FI_WARN(&rxr_prov, FI_LOG_CQ,
				"RX entries exhausted.\n");
			rxr_eq_write_error(rxr_ep, FI_ENOBUFS, -FI_ENOBUFS);
			ret = -FI_ENOBUFS;
			goto out;
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
		 * this rx_entry is ready to receive
		 */

		/* If there is no available buffer, we do not proceed.
		 * It is important to decrease peer->next_msg_id by 1
		 * in this case because this message was not sent.
		 */
		if (rxr_ep->available_data_bufs==0) {
			rxr_release_tx_entry(rxr_ep, tx_entry);
			rxr_release_rx_entry(rxr_ep, rx_entry);
			peer->next_msg_id--;
			ret = -FI_EAGAIN;
			rxr_ep_progress_internal(rxr_ep);
			goto out;
		}

		rxr_ep_calc_cts_window_credits(rxr_ep, rxr_env.rx_window_size,
					       tx_entry->total_len, &window,
					       &credits);

		rx_entry->window = window;
		assert(rxr_ep->available_data_bufs >= credits);
		rxr_ep->available_data_bufs -= credits;

		rx_entry->state = RXR_RX_RECV;
		/* rma_loc_tx_id is used in rxr_cq_handle_rx_completion()
		 * to locate the tx_entry for tx completion.
		 */
		rx_entry->rma_loc_tx_id = tx_entry->tx_id;
#if ENABLE_DEBUG
		dlist_insert_tail(&rx_entry->rx_pending_entry,
				  &rxr_ep->rx_pending_list);
		rxr_ep->rx_pending++;
#endif
		/*
		 * this tx_entry does not need a rx_id, because it does not
		 * send any data.
		 * the rma_loc_rx_id and rma_window will be sent to remote EP
		 * via RTS
		 */
		tx_entry->rma_loc_rx_id = rx_entry->rx_id;
		tx_entry->rma_window = rx_entry->window;
	}

	ret = rxr_ep_post_rts(rxr_ep, tx_entry);

	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN) {
			tx_entry->state = RXR_TX_QUEUED_RTS;
			dlist_insert_tail(&tx_entry->queued_entry,
					  &rxr_ep->tx_entry_queued_list);
			ret = 0;
		} else {
			peer = rxr_ep_get_peer(rxr_ep, addr);
			peer->next_msg_id--;
		}
	}

out:
	fastlock_release(&rxr_ep->util_ep.lock);
	rxr_perfset_end(rxr_ep, perf_rxr_tx);
	return ret;
}

static ssize_t rxr_ep_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			      uint64_t flags)
{
	return rxr_tx(ep, msg->msg_iov, msg->iov_count, NULL, 0,
		      msg->addr, 0, msg->data, msg->context,
		      ofi_op_msg, flags);
}

static ssize_t rxr_ep_sendv(struct fid_ep *ep, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t dest_addr,
			    void *context)
{
	struct fi_msg msg;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;

	return rxr_ep_sendmsg(ep, &msg, 0);
}

static ssize_t rxr_ep_send(struct fid_ep *ep, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_ep_sendv(ep, &iov, desc, 1, dest_addr, context);
}

static ssize_t rxr_ep_senddata(struct fid_ep *ep, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	return rxr_tx(ep, &iov, 1, NULL, 0, dest_addr, 0, data, context,
		      ofi_op_msg, FI_REMOTE_CQ_DATA);
}

static ssize_t rxr_ep_inject(struct fid_ep *ep, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
#if ENABLE_DEBUG
	struct rxr_ep *rxr_ep;
#endif
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

#if ENABLE_DEBUG
	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);
	assert(len <= rxr_ep->core_inject_size - RXR_CTRL_HDR_SIZE_NO_CQ);
#endif

	return rxr_tx(ep, &iov, 1, NULL, 0, dest_addr, 0, 0, NULL, ofi_op_msg,
		      RXR_NO_COMPLETION | FI_INJECT);
}

static ssize_t rxr_ep_injectdata(struct fid_ep *ep, const void *buf,
				 size_t len, uint64_t data,
				 fi_addr_t dest_addr)
{
#if ENABLE_DEBUG
	struct rxr_ep *rxr_ep;
#endif
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

#if ENABLE_DEBUG
	rxr_ep = container_of(ep, struct rxr_ep, util_ep.ep_fid.fid);
	/*
	 * We advertise the largest possible inject size with no cq data or
	 * source address. This means that we may end up not using the core
	 * providers inject for this send.
	 */
	assert(len <= rxr_ep->core_inject_size - RXR_CTRL_HDR_SIZE_NO_CQ);
#endif

	return rxr_tx(ep, &iov, 1, NULL, 0, dest_addr, 0, data, NULL,
		      ofi_op_msg, RXR_NO_COMPLETION | FI_REMOTE_CQ_DATA | FI_INJECT);
}

static struct fi_ops_msg rxr_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxr_ep_recv,
	.recvv = rxr_ep_recvv,
	.recvmsg = rxr_ep_recvmsg,
	.send = rxr_ep_send,
	.sendv = rxr_ep_sendv,
	.sendmsg = rxr_ep_sendmsg,
	.inject = rxr_ep_inject,
	.senddata = rxr_ep_senddata,
	.injectdata = rxr_ep_injectdata,
};

ssize_t rxr_ep_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		     fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		     void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *)buf;
	msg_iov.iov_len = len;

	return rxr_recv(ep_fid, &msg_iov, 1, src_addr, tag, ignore,
			context, ofi_op_tagged, 0);
}

ssize_t rxr_ep_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t src_addr,
		      uint64_t tag, uint64_t ignore, void *context)
{
	return rxr_recv(ep_fid, iov, count, src_addr, tag, ignore,
			context, ofi_op_tagged, 0);
}

ssize_t rxr_ep_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			uint64_t flags)
{
	ssize_t ret;

	if (flags & FI_PEEK) {
		ret = rxr_ep_peek_trecv(ep_fid, msg, flags);
		goto out;
	} else if (flags & FI_CLAIM) {
		ret = rxr_ep_claim_trecv(ep_fid, msg, flags);
		goto out;
	}

	ret = rxr_recv(ep_fid, msg->msg_iov, msg->iov_count, msg->addr,
		       msg->tag, msg->ignore, msg->context,
		       ofi_op_tagged, flags);

out:
	return ret;
}

ssize_t rxr_ep_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			uint64_t flags)
{
	return rxr_tx(ep_fid, msg->msg_iov, msg->iov_count, NULL, 0,
		      msg->addr, msg->tag, msg->data, msg->context,
		      ofi_op_tagged, flags);
}

ssize_t rxr_ep_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t dest_addr,
		      uint64_t tag, void *context)
{
	struct fi_msg_tagged msg;

	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.tag = tag;

	return rxr_ep_tsendmsg(ep_fid, &msg, 0);
}

ssize_t rxr_ep_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr, uint64_t tag,
		     void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *)buf;
	msg_iov.iov_len = len;

	return rxr_ep_tsendv(ep_fid, &msg_iov, desc, 1, dest_addr, tag,
			     context);
}

ssize_t rxr_ep_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
		       fi_addr_t dest_addr, uint64_t tag)
{
#if ENABLE_DEBUG
	struct rxr_ep *rxr_ep;
#endif
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

#if ENABLE_DEBUG
	rxr_ep = container_of(ep_fid, struct rxr_ep, util_ep.ep_fid.fid);
	assert(len <= rxr_ep->core_inject_size - RXR_CTRL_HDR_SIZE_NO_CQ);
#endif

	return rxr_tx(ep_fid, &iov, 1, NULL, 0, dest_addr, tag, 0, NULL,
		      ofi_op_tagged, RXR_NO_COMPLETION | FI_INJECT);
}

ssize_t rxr_ep_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 uint64_t tag, void *context)
{
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	return rxr_tx(ep_fid, &iov, 1, NULL, 0, dest_addr, tag, data, context,
		      ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

ssize_t rxr_ep_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			   uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
#if ENABLE_DEBUG
	struct rxr_ep *rxr_ep;
#endif
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

#if ENABLE_DEBUG
	rxr_ep = container_of(ep_fid, struct rxr_ep, util_ep.ep_fid.fid);
	/*
	 * We advertise the largest possible inject size with no cq data or
	 * source address. This means that we may end up not using the core
	 * providers inject for this send.
	 */
	assert(len <= rxr_ep->core_inject_size - RXR_CTRL_HDR_SIZE_NO_CQ);
#endif

	return rxr_tx(ep_fid, &iov, 1, NULL, 0, dest_addr, tag, data, NULL,
		      ofi_op_tagged, RXR_NO_COMPLETION | FI_REMOTE_CQ_DATA | FI_INJECT);
}

static struct fi_ops_tagged rxr_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = rxr_ep_trecv,
	.recvv = rxr_ep_trecvv,
	.recvmsg = rxr_ep_trecvmsg,
	.send = rxr_ep_tsend,
	.sendv = rxr_ep_tsendv,
	.sendmsg = rxr_ep_tsendmsg,
	.inject = rxr_ep_tinject,
	.senddata = rxr_ep_tsenddata,
	.injectdata = rxr_ep_tinjectdata,
};

static void rxr_ep_free_res(struct rxr_ep *rxr_ep)
{
	struct rxr_peer *peer;
	struct dlist_entry *tmp;
#if ENABLE_DEBUG
	struct dlist_entry *entry;
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	struct rxr_pkt_entry *pkt;
#endif

	if (rxr_need_sas_ordering(rxr_ep)) {
		dlist_foreach_container_safe(&rxr_ep->peer_list,
					     struct rxr_peer,
					     peer, entry, tmp) {
			ofi_recvwin_free(peer->robuf);
		}

		if (rxr_ep->robuf_fs)
			rxr_robuf_fs_free(rxr_ep->robuf_fs);
	}

#if ENABLE_DEBUG
	dlist_foreach_container_safe(&rxr_ep->peer_list,
				     struct rxr_peer,
				     peer, entry, tmp) {
		/*
		 * TODO: Add support for wait/signal until all pending messages
		 * have been sent/received so the core does not attempt to
		 * complete a data operation or an internal RxR transfer after
		 * the EP is shutdown.
		 */
		if (peer->state == RXR_PEER_CONNREQ)
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Closing EP with unacked CONNREQs in flight\n");
	}

	dlist_foreach(&rxr_ep->rx_unexp_list, entry) {
		rx_entry = container_of(entry, struct rxr_rx_entry, entry);
		rxr_release_rx_pkt_entry(rxr_ep, rx_entry->unexp_rts_pkt);
	}

	dlist_foreach(&rxr_ep->rx_unexp_tagged_list, entry) {
		rx_entry = container_of(entry, struct rxr_rx_entry, entry);
		rxr_release_rx_pkt_entry(rxr_ep, rx_entry->unexp_rts_pkt);
	}

	dlist_foreach(&rxr_ep->rx_entry_queued_list, entry) {
		rx_entry = container_of(entry, struct rxr_rx_entry,
					queued_entry);
		dlist_foreach_container_safe(&rx_entry->queued_pkts,
					     struct rxr_pkt_entry,
					     pkt, entry, tmp)
			rxr_release_tx_pkt_entry(rxr_ep, pkt);
	}

	dlist_foreach(&rxr_ep->tx_entry_queued_list, entry) {
		tx_entry = container_of(entry, struct rxr_tx_entry,
					queued_entry);
		dlist_foreach_container_safe(&tx_entry->queued_pkts,
					     struct rxr_pkt_entry,
					     pkt, entry, tmp)
			rxr_release_tx_pkt_entry(rxr_ep, pkt);
	}

	dlist_foreach_safe(&rxr_ep->rx_pkt_list, entry, tmp) {
		pkt = container_of(entry, struct rxr_pkt_entry, dbg_entry);
		rxr_release_rx_pkt_entry(rxr_ep, pkt);
	}

	dlist_foreach_safe(&rxr_ep->tx_pkt_list, entry, tmp) {
		pkt = container_of(entry, struct rxr_pkt_entry, dbg_entry);
		rxr_release_tx_pkt_entry(rxr_ep, pkt);
	}

	dlist_foreach_safe(&rxr_ep->rx_posted_buf_list, entry, tmp) {
		pkt = container_of(entry, struct rxr_pkt_entry, dbg_entry);
		ofi_buf_free(pkt);
	}
	dlist_foreach_safe(&rxr_ep->rx_entry_list, entry, tmp) {
		rx_entry = container_of(entry, struct rxr_rx_entry,
					rx_entry_entry);
		rxr_release_rx_entry(rxr_ep, rx_entry);
	}
	dlist_foreach_safe(&rxr_ep->tx_entry_list, entry, tmp) {
		tx_entry = container_of(entry, struct rxr_tx_entry,
					tx_entry_entry);
		rxr_release_tx_entry(rxr_ep, tx_entry);
	}
#endif

	if (rxr_ep->rx_entry_pool)
		ofi_bufpool_destroy(rxr_ep->rx_entry_pool);

	if (rxr_ep->tx_entry_pool)
		ofi_bufpool_destroy(rxr_ep->tx_entry_pool);

	if (rxr_ep->readrsp_tx_entry_pool)
		ofi_bufpool_destroy(rxr_ep->readrsp_tx_entry_pool);

	if (rxr_ep->rx_ooo_pkt_pool)
		ofi_bufpool_destroy(rxr_ep->rx_ooo_pkt_pool);

	if (rxr_ep->rx_unexp_pkt_pool)
		ofi_bufpool_destroy(rxr_ep->rx_unexp_pkt_pool);

	if (rxr_ep->rx_pkt_pool)
		ofi_bufpool_destroy(rxr_ep->rx_pkt_pool);

	if (rxr_ep->tx_pkt_pool)
		ofi_bufpool_destroy(rxr_ep->tx_pkt_pool);
}

static int rxr_ep_close(struct fid *fid)
{
	int ret, retv = 0;
	struct rxr_ep *rxr_ep;

	rxr_ep = container_of(fid, struct rxr_ep, util_ep.ep_fid.fid);

	ret = fi_close(&rxr_ep->rdm_ep->fid);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Unable to close EP\n");
		retv = ret;
	}

	ret = fi_close(&rxr_ep->rdm_cq->fid);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
		retv = ret;
	}

	ret = ofi_endpoint_close(&rxr_ep->util_ep);
	if (ret) {
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Unable to close util EP\n");
		retv = ret;
	}
	rxr_ep_free_res(rxr_ep);
	free(rxr_ep->peer);
	free(rxr_ep);
	return retv;
}

static int rxr_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxr_ep *rxr_ep =
		container_of(ep_fid, struct rxr_ep, util_ep.ep_fid.fid);
	struct util_cq *cq;
	struct rxr_av *av;
	struct util_cntr *cntr;
	struct util_eq *eq;
	int ret = 0;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct rxr_av, util_av.av_fid.fid);
		/* Bind util provider endpoint and av */
		ret = ofi_ep_bind_av(&rxr_ep->util_ep, &av->util_av);
		if (ret)
			return ret;

		/* Bind core provider endpoint & av */
		ret = fi_ep_bind(rxr_ep->rdm_ep, &av->rdm_av->fid, flags);
		if (ret)
			return ret;

		rxr_ep->peer = calloc(av->util_av.count,
				      sizeof(struct rxr_peer));
		if (!rxr_ep->peer)
			return -FI_ENOMEM;

		rxr_ep->robuf_fs = rxr_robuf_fs_create(rxr_ep->rx_size,
						       NULL, NULL);
		if (!rxr_ep->robuf_fs)
			return -FI_ENOMEM;

		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&rxr_ep->util_ep, cq, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&rxr_ep->util_ep, cntr, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_EQ:
		eq = container_of(bfid, struct util_eq, eq_fid.fid);

		ret = ofi_ep_bind_eq(&rxr_ep->util_ep, eq);
		if (ret)
			return ret;
		break;
	default:
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int rxr_ep_ctrl(struct fid *fid, int command, void *arg)
{
	ssize_t ret;
	size_t i;
	struct rxr_ep *ep;
	uint64_t flags = FI_MORE;
	size_t rx_size;

	switch (command) {
	case FI_ENABLE:
		/* Enable core provider endpoint & post recv buff */
		ep = container_of(fid, struct rxr_ep, util_ep.ep_fid.fid);

		rx_size = rxr_get_rx_pool_chunk_cnt(ep);

		ret = fi_enable(ep->rdm_ep);
		if (ret)
			return ret;

		fastlock_acquire(&ep->util_ep.lock);
		for (i = 0; i < rx_size; i++) {
			if (i == rx_size - 1)
				flags = 0;

			ret = rxr_ep_post_buf(ep, flags);

			if (ret)
				goto out;
		}

		ep->available_data_bufs = rx_size;

		ep->core_addrlen = RXR_MAX_NAME_LENGTH;
		ret = fi_getname(&ep->rdm_ep->fid,
				 ep->core_addr,
				 &ep->core_addrlen);
		assert(ret != -FI_ETOOSMALL);
		FI_DBG(&rxr_prov, FI_LOG_EP_CTRL, "core_addrlen = %ld\n",
		       ep->core_addrlen);
out:
		fastlock_release(&ep->util_ep.lock);
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}

	return ret;
}

static struct fi_ops rxr_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxr_ep_close,
	.bind = rxr_ep_bind,
	.control = rxr_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int rxr_ep_cancel_match_recv(struct dlist_entry *item,
				    const void *context)
{
	struct rxr_rx_entry *rx_entry = container_of(item,
						     struct rxr_rx_entry,
						     entry);
	return rx_entry->cq_entry.op_context == context;
}

static ssize_t rxr_ep_cancel_recv(struct rxr_ep *ep,
				  struct dlist_entry *recv_list,
				  void *context)
{
	struct rxr_domain *domain;
	struct dlist_entry *entry;
	struct rxr_rx_entry *rx_entry;
	struct fi_cq_err_entry err_entry;
	uint32_t api_version;

	fastlock_acquire(&ep->util_ep.lock);
	entry = dlist_remove_first_match(recv_list,
					 &rxr_ep_cancel_match_recv,
					 context);
	if (entry) {
		rx_entry = container_of(entry, struct rxr_rx_entry, entry);
		rx_entry->rxr_flags |= RXR_RECV_CANCEL;
		if (rx_entry->fi_flags & FI_MULTI_RECV)
			rxr_cq_handle_multi_recv_completion(ep, rx_entry);
		fastlock_release(&ep->util_ep.lock);
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = rx_entry->cq_entry.op_context;
		err_entry.flags |= rx_entry->cq_entry.flags;
		err_entry.tag = rx_entry->tag;
		err_entry.err = FI_ECANCELED;
		err_entry.prov_errno = -FI_ECANCELED;

		domain = rxr_ep_domain(ep);
		api_version =
			 domain->util_domain.fabric->fabric_fid.api_version;
		if (FI_VERSION_GE(api_version, FI_VERSION(1, 5)))
			err_entry.err_data_size = 0;
		return ofi_cq_write_error(ep->util_ep.rx_cq, &err_entry);
	}

	fastlock_release(&ep->util_ep.lock);
	return 0;
}

static ssize_t rxr_ep_cancel(fid_t fid_ep, void *context)
{
	struct rxr_ep *ep;
	int ret;

	ep = container_of(fid_ep, struct rxr_ep, util_ep.ep_fid.fid);

	ret = rxr_ep_cancel_recv(ep, &ep->rx_list, context);
	if (ret)
		return ret;

	ret = rxr_ep_cancel_recv(ep, &ep->rx_tagged_list, context);
	return ret;
}

static int rxr_ep_getopt(fid_t fid, int level, int optname, void *optval,
			 size_t *optlen)
{
	struct rxr_ep *rxr_ep = container_of(fid, struct rxr_ep,
					     util_ep.ep_fid.fid);

	if (level != FI_OPT_ENDPOINT || optname != FI_OPT_MIN_MULTI_RECV)
		return -FI_ENOPROTOOPT;

	*(size_t *)optval = rxr_ep->min_multi_recv_size;
	*optlen = sizeof(size_t);

	return FI_SUCCESS;
}

static int rxr_ep_setopt(fid_t fid, int level, int optname,
			 const void *optval, size_t optlen)
{
	struct rxr_ep *rxr_ep = container_of(fid, struct rxr_ep,
					     util_ep.ep_fid.fid);

	if (level != FI_OPT_ENDPOINT || optname != FI_OPT_MIN_MULTI_RECV)
		return -FI_ENOPROTOOPT;

	if (optlen < sizeof(size_t))
		return -FI_EINVAL;

	rxr_ep->min_multi_recv_size = *(size_t *)optval;

	return FI_SUCCESS;
}

static struct fi_ops_ep rxr_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = rxr_ep_cancel,
	.getopt = rxr_ep_getopt,
	.setopt = rxr_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int rxr_buf_region_alloc_hndlr(struct ofi_bufpool_region *region)
{
	size_t ret;
	struct fid_mr *mr;
	struct rxr_domain *domain = region->pool->attr.context;

	ret = fi_mr_reg(domain->rdm_domain, region->mem_region,
			region->pool->region_size,
			FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);

	region->context = mr;
	return ret;
}

static void rxr_buf_region_free_hndlr(struct ofi_bufpool_region *region)
{
	ssize_t ret;

	ret = fi_close((struct fid *)region->context);
	if (ret)
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Unable to deregister memory in a buf pool: %s\n",
			fi_strerror(-ret));
}

static int rxr_create_pkt_pool(struct rxr_ep *ep, size_t size,
			       size_t chunk_count,
			       struct ofi_bufpool **buf_pool)
{
	struct ofi_bufpool_attr attr = {
		.size		= size,
		.alignment	= RXR_BUF_POOL_ALIGNMENT,
		.max_cnt	= chunk_count,
		.chunk_cnt	= chunk_count,
		.alloc_fn	= rxr_ep_mr_local(ep) ?
					rxr_buf_region_alloc_hndlr : NULL,
		.free_fn	= rxr_ep_mr_local(ep) ?
					rxr_buf_region_free_hndlr : NULL,
		.init_fn	= NULL,
		.context	= rxr_ep_domain(ep),
		.flags		= OFI_BUFPOOL_HUGEPAGES,
	};

	return ofi_bufpool_create_attr(&attr, buf_pool);
}

int rxr_ep_init(struct rxr_ep *ep)
{
	size_t entry_sz;
	int ret;

	entry_sz = ep->mtu_size + sizeof(struct rxr_pkt_entry);
#ifdef ENABLE_EFA_POISONING
	ep->tx_pkt_pool_entry_sz = entry_sz;
	ep->rx_pkt_pool_entry_sz = entry_sz;
#endif

	ret = rxr_create_pkt_pool(ep, entry_sz, rxr_get_tx_pool_chunk_cnt(ep),
				  &ep->tx_pkt_pool);
	if (ret)
		goto err_out;

	ret = rxr_create_pkt_pool(ep, entry_sz, rxr_get_rx_pool_chunk_cnt(ep),
				  &ep->rx_pkt_pool);
	if (ret)
		goto err_free_tx_pool;

	if (rxr_env.rx_copy_unexp) {
		ret = ofi_bufpool_create(&ep->rx_unexp_pkt_pool, entry_sz,
					 RXR_BUF_POOL_ALIGNMENT, 0,
					 rxr_get_rx_pool_chunk_cnt(ep), 0);

		if (ret)
			goto err_free_rx_pool;
	}

	if (rxr_env.rx_copy_ooo) {
		ret = ofi_bufpool_create(&ep->rx_ooo_pkt_pool, entry_sz,
					 RXR_BUF_POOL_ALIGNMENT, 0,
					 rxr_env.recvwin_size, 0);

		if (ret)
			goto err_free_rx_unexp_pool;
	}

	ret = ofi_bufpool_create(&ep->tx_entry_pool,
				 sizeof(struct rxr_tx_entry),
				 RXR_BUF_POOL_ALIGNMENT,
				 ep->tx_size, ep->tx_size, 0);
	if (ret)
		goto err_free_rx_ooo_pool;

	ret = ofi_bufpool_create(&ep->readrsp_tx_entry_pool,
				 sizeof(struct rxr_tx_entry),
				 RXR_BUF_POOL_ALIGNMENT,
				 RXR_MAX_RX_QUEUE_SIZE,
				 ep->rx_size, 0);
	if (ret)
		goto err_free_tx_entry_pool;

	ret = ofi_bufpool_create(&ep->rx_entry_pool,
				 sizeof(struct rxr_rx_entry),
				 RXR_BUF_POOL_ALIGNMENT,
				 RXR_MAX_RX_QUEUE_SIZE,
				 ep->rx_size, 0);
	if (ret)
		goto err_free_readrsp_tx_entry_pool;

	/* Initialize entry list */
	dlist_init(&ep->rx_list);
	dlist_init(&ep->rx_unexp_list);
	dlist_init(&ep->rx_tagged_list);
	dlist_init(&ep->rx_unexp_tagged_list);
	dlist_init(&ep->rx_posted_buf_list);
	dlist_init(&ep->rx_entry_queued_list);
	dlist_init(&ep->tx_entry_queued_list);
	dlist_init(&ep->tx_pending_list);
	dlist_init(&ep->peer_backoff_list);
	dlist_init(&ep->peer_list);
#if ENABLE_DEBUG
	dlist_init(&ep->rx_pending_list);
	dlist_init(&ep->rx_pkt_list);
	dlist_init(&ep->tx_pkt_list);
	dlist_init(&ep->rx_entry_list);
	dlist_init(&ep->tx_entry_list);
#endif

	return 0;
err_free_readrsp_tx_entry_pool:
	if (ep->readrsp_tx_entry_pool)
		ofi_bufpool_destroy(ep->readrsp_tx_entry_pool);
err_free_tx_entry_pool:
	if (ep->tx_entry_pool)
		ofi_bufpool_destroy(ep->tx_entry_pool);
err_free_rx_ooo_pool:
	if (rxr_env.rx_copy_ooo && ep->rx_ooo_pkt_pool)
		ofi_bufpool_destroy(ep->rx_ooo_pkt_pool);
err_free_rx_unexp_pool:
	if (rxr_env.rx_copy_unexp && ep->rx_unexp_pkt_pool)
		ofi_bufpool_destroy(ep->rx_unexp_pkt_pool);
err_free_rx_pool:
	if (ep->rx_pkt_pool)
		ofi_bufpool_destroy(ep->rx_pkt_pool);
err_free_tx_pool:
	if (ep->tx_pkt_pool)
		ofi_bufpool_destroy(ep->tx_pkt_pool);
err_out:
	return ret;
}

static int rxr_ep_rdm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxr_ep *ep;

	ep = container_of(fid, struct rxr_ep, util_ep.ep_fid.fid);
	return fi_setname(&ep->rdm_ep->fid, addr, addrlen);
}

static int rxr_ep_rdm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxr_ep *ep;

	ep = container_of(fid, struct rxr_ep, util_ep.ep_fid.fid);
	return fi_getname(&ep->rdm_ep->fid, addr, addrlen);
}

struct fi_ops_cm rxr_ep_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxr_ep_rdm_setname,
	.getname = rxr_ep_rdm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static inline int rxr_ep_bulk_post_recv(struct rxr_ep *ep)
{
	uint64_t flags = FI_MORE;
	int ret;

	while (ep->rx_bufs_to_post) {
		if (ep->rx_bufs_to_post == 1)
			flags = 0;
		ret = rxr_ep_post_buf(ep, flags);
		if (OFI_LIKELY(!ret))
			ep->rx_bufs_to_post--;
		else
			return ret;
	}

	return 0;
}

static inline int rxr_ep_send_queued_pkts(struct rxr_ep *ep,
					  struct dlist_entry *pkts)
{
	struct dlist_entry *tmp;
	struct rxr_pkt_entry *pkt_entry;
	int ret;

	dlist_foreach_container_safe(pkts, struct rxr_pkt_entry,
				     pkt_entry, entry, tmp) {
		ret = rxr_ep_send_pkt(ep, pkt_entry, pkt_entry->addr);
		if (ret)
			return ret;
		dlist_remove(&pkt_entry->entry);
	}
	return 0;
}

static inline void rxr_ep_check_available_data_bufs_timer(struct rxr_ep *ep)
{
	if (OFI_LIKELY(ep->available_data_bufs != 0))
		return;

	if (fi_gettime_us() - ep->available_data_bufs_ts >=
	    RXR_AVAILABLE_DATA_BUFS_TIMEOUT) {
		ep->available_data_bufs = rxr_get_rx_pool_chunk_cnt(ep);
		ep->available_data_bufs_ts = 0;
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Reset available buffers for large message receives\n");
	}
}

static inline void rxr_ep_check_peer_backoff_timer(struct rxr_ep *ep)
{
	struct rxr_peer *peer;
	struct dlist_entry *tmp;

	if (OFI_LIKELY(dlist_empty(&ep->peer_backoff_list)))
		return;

	dlist_foreach_container_safe(&ep->peer_backoff_list, struct rxr_peer,
				     peer, rnr_entry, tmp) {
		peer->rnr_state &= ~RXR_PEER_BACKED_OFF;
		if (!rxr_peer_timeout_expired(ep, peer, fi_gettime_us()))
			continue;
		peer->rnr_state = 0;
		dlist_remove(&peer->rnr_entry);
	}
}

static void rxr_ep_progress_internal(struct rxr_ep *ep)
{
	struct fi_cq_msg_entry cq_entry;
	struct rxr_rx_entry *rx_entry;
	struct rxr_tx_entry *tx_entry;
	struct dlist_entry *tmp;
	fi_addr_t src_addr;
	ssize_t ret;
	int i;

	rxr_ep_check_available_data_bufs_timer(ep);

	VALGRIND_MAKE_MEM_DEFINED(&cq_entry, sizeof(struct fi_cq_msg_entry));

	for (ret = 1, i = 0; ret > 0 && i < 100; i++) {
		if (ep->core_caps & FI_SOURCE) {
			ret = fi_cq_readfrom(ep->rdm_cq, &cq_entry, 1, &src_addr);
		} else {
			ret = fi_cq_read(ep->rdm_cq, &cq_entry, 1);
			src_addr = FI_ADDR_NOTAVAIL;
		}

		if (ret == -FI_EAGAIN)
			break;
		if (OFI_UNLIKELY(ret < 0)) {
			if (rxr_cq_handle_cq_error(ep, ret))
				assert(0 &&
				       "error writing error cq entry after reading from cq");
			rxr_ep_bulk_post_recv(ep);
			return;
		}

		if (cq_entry.flags & FI_SEND) {
#if ENABLE_DEBUG
			ep->send_comps++;
#endif
			rxr_cq_handle_pkt_send_completion(ep, &cq_entry);
		} else if (cq_entry.flags & FI_RECV) {
			rxr_cq_handle_pkt_recv_completion(ep, &cq_entry, src_addr);
#if ENABLE_DEBUG
			ep->recv_comps++;
#endif
		} else {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"Unhandled cq type\n");
			assert(0 && "Unhandled cq type");
		}
	}

	ret = rxr_ep_bulk_post_recv(ep);

	if (OFI_UNLIKELY(ret)) {
		if (rxr_cq_handle_cq_error(ep, ret))
			assert(0 &&
			       "error writing error cq entry after failed post recv");
		return;
	}

	rxr_ep_check_peer_backoff_timer(ep);

	/*
	 * Send any queued RTS/CTS packets.
	 */
	dlist_foreach_container_safe(&ep->rx_entry_queued_list,
				     struct rxr_rx_entry,
				     rx_entry, queued_entry, tmp) {
		if (rx_entry->state == RXR_RX_QUEUED_CTS)
			ret = rxr_cq_post_cts(ep, rx_entry,
					      rxr_env.rx_window_size,
					      rx_entry->total_len -
					      rx_entry->bytes_done);
		else
			ret = rxr_ep_send_queued_pkts(ep,
						      &rx_entry->queued_pkts);
		if (ret == -FI_EAGAIN)
			break;
		if (OFI_UNLIKELY(ret))
			goto rx_err;

		dlist_remove(&rx_entry->queued_entry);
		rx_entry->state = RXR_RX_RECV;
	}

	dlist_foreach_container_safe(&ep->tx_entry_queued_list,
				     struct rxr_tx_entry,
				     tx_entry, queued_entry, tmp) {
		if (tx_entry->state == RXR_TX_QUEUED_RTS)
			ret = rxr_ep_post_rts(ep, tx_entry);
		else if (tx_entry->state == RXR_TX_QUEUED_READRSP)
			ret = rxr_ep_post_readrsp(ep, tx_entry);
		else
			ret = rxr_ep_send_queued_pkts(ep,
						      &tx_entry->queued_pkts);

		if (ret == -FI_EAGAIN)
			break;
		if (OFI_UNLIKELY(ret))
			goto tx_err;

		dlist_remove(&tx_entry->queued_entry);

		if (tx_entry->state == RXR_TX_QUEUED_RTS ||
		    tx_entry->state == RXR_TX_QUEUED_RTS_RNR) {
			tx_entry->state = RXR_TX_RTS;
		} else if (tx_entry->state == RXR_TX_QUEUED_READRSP) {
			tx_entry->state = RXR_TX_SENT_READRSP;
			if (tx_entry->bytes_sent < tx_entry->total_len) {
				tx_entry->state = RXR_TX_SEND;
				dlist_insert_tail(&tx_entry->entry,
						  &ep->tx_pending_list);
			}
		} else if (tx_entry->state == RXR_TX_QUEUED_DATA_RNR) {
			tx_entry->state = RXR_TX_SEND;
			dlist_insert_tail(&tx_entry->entry,
					  &ep->tx_pending_list);
		}
	}

	/*
	 * Send data packets until window or tx queue is exhausted.
	 */
	dlist_foreach_container(&ep->tx_pending_list, struct rxr_tx_entry,
				tx_entry, entry) {
		if (tx_entry->window > 0)
			tx_entry->send_flags |= FI_MORE;
		else
			continue;

		while (tx_entry->window > 0) {
			if (ep->max_outstanding_tx - ep->tx_pending <= 1 ||
			    tx_entry->window <= ep->max_data_payload_size)
				tx_entry->send_flags &= ~FI_MORE;
			/*
			 * The core's TX queue is full so we can't do any
			 * additional work.
			 */
			if (ep->tx_pending == ep->max_outstanding_tx)
				goto out;
			ret = rxr_ep_post_data(ep, tx_entry);
			if (OFI_UNLIKELY(ret)) {
				tx_entry->send_flags &= ~FI_MORE;
				goto tx_err;
			}
		}
	}

out:
	return;
rx_err:
	if (rxr_cq_handle_rx_error(ep, rx_entry, ret))
		assert(0 &&
		       "error writing error cq entry when handling RX error");
	return;
tx_err:
	if (rxr_cq_handle_tx_error(ep, tx_entry, ret))
		assert(0 &&
		       "error writing error cq entry when handling TX error");
	return;
}

void rxr_ep_progress(struct util_ep *util_ep)
{
	struct rxr_ep *ep;

	ep = container_of(util_ep, struct rxr_ep, util_ep);

	fastlock_acquire(&ep->util_ep.lock);
	rxr_ep_progress_internal(ep);
	fastlock_release(&ep->util_ep.lock);
}

int rxr_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	struct fi_info *rdm_info;
	struct rxr_domain *rxr_domain;
	struct rxr_ep *rxr_ep;
	struct fi_cq_attr cq_attr;
	int ret, retv;

	rxr_ep = calloc(1, sizeof(*rxr_ep));
	if (!rxr_ep)
		return -FI_ENOMEM;

	rxr_domain = container_of(domain, struct rxr_domain,
				  util_domain.domain_fid);
	memset(&cq_attr, 0, sizeof(cq_attr));
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = FI_WAIT_NONE;

	ret = ofi_endpoint_init(domain, &rxr_util_prov, info, &rxr_ep->util_ep,
				context, rxr_ep_progress);
	if (ret)
		goto err_free_ep;

	ret = rxr_get_lower_rdm_info(rxr_domain->util_domain.fabric->
				     fabric_fid.api_version, NULL, NULL, 0,
				     &rxr_util_prov, info, &rdm_info);
	if (ret)
		goto err_close_ofi_ep;

	rxr_reset_rx_tx_to_core(info, rdm_info);

	ret = fi_endpoint(rxr_domain->rdm_domain, rdm_info,
			  &rxr_ep->rdm_ep, rxr_ep);
	if (ret)
		goto err_close_ofi_ep;

	rxr_ep->rx_size = info->rx_attr->size;
	rxr_ep->tx_size = info->tx_attr->size;
	rxr_ep->rx_iov_limit = info->rx_attr->iov_limit;
	rxr_ep->tx_iov_limit = info->tx_attr->iov_limit;
	rxr_ep->max_outstanding_tx = rdm_info->tx_attr->size;
	rxr_ep->core_rx_size = rdm_info->rx_attr->size;
	rxr_ep->core_iov_limit = rdm_info->tx_attr->iov_limit;
	rxr_ep->core_caps = rdm_info->caps;

	cq_attr.size = MAX(rxr_ep->rx_size + rxr_ep->tx_size,
			   rxr_env.cq_size);

	if (info->tx_attr->op_flags & FI_DELIVERY_COMPLETE)
		FI_INFO(&rxr_prov, FI_LOG_CQ, "FI_DELIVERY_COMPLETE unsupported\n");

	assert(info->tx_attr->msg_order == info->rx_attr->msg_order);
	rxr_ep->msg_order = info->rx_attr->msg_order;
	rxr_ep->core_msg_order = rdm_info->rx_attr->msg_order;
	rxr_ep->core_inject_size = rdm_info->tx_attr->inject_size;
	rxr_ep->mtu_size = rdm_info->ep_attr->max_msg_size;
	if (rxr_env.mtu_size > 0 && rxr_env.mtu_size < rxr_ep->mtu_size)
		rxr_ep->mtu_size = rxr_env.mtu_size;

	if (rxr_ep->mtu_size > RXR_MTU_MAX_LIMIT)
		rxr_ep->mtu_size = RXR_MTU_MAX_LIMIT;

	rxr_ep->max_data_payload_size = rxr_ep->mtu_size - RXR_DATA_HDR_SIZE;
	/*
	 * Assume our eager message size is the largest control header size
	 * without the source address. Use that value to set the default
	 * receive release threshold.
	 */
	rxr_ep->min_multi_recv_size = rxr_ep->mtu_size - RXR_CTRL_HDR_SIZE;

	if (rxr_env.tx_queue_size > 0 &&
	    rxr_env.tx_queue_size < rxr_ep->max_outstanding_tx)
		rxr_ep->max_outstanding_tx = rxr_env.tx_queue_size;

#if ENABLE_DEBUG
	rxr_ep->sends = 0;
	rxr_ep->send_comps = 0;
	rxr_ep->failed_send_comps = 0;
	rxr_ep->recv_comps = 0;
#endif

	rxr_ep->posted_bufs = 0;
	rxr_ep->rx_bufs_to_post = 0;
	rxr_ep->tx_pending = 0;
	rxr_ep->available_data_bufs_ts = 0;

	fi_freeinfo(rdm_info);

	ret = fi_cq_open(rxr_domain->rdm_domain, &cq_attr,
			 &rxr_ep->rdm_cq, rxr_ep);
	if (ret)
		goto err_close_core_ep;

	ret = fi_ep_bind(rxr_ep->rdm_ep, &rxr_ep->rdm_cq->fid,
			 FI_TRANSMIT | FI_RECV);
	if (ret)
		goto err_close_core_cq;

	ret = rxr_ep_init(rxr_ep);
	if (ret)
		goto err_close_core_cq;

	*ep = &rxr_ep->util_ep.ep_fid;
	(*ep)->msg = &rxr_ops_msg;
	(*ep)->rma = &rxr_ops_rma;
	(*ep)->tagged = &rxr_ops_tagged;
	(*ep)->fid.ops = &rxr_ep_fi_ops;
	(*ep)->ops = &rxr_ops_ep;
	(*ep)->cm = &rxr_ep_cm;
	return 0;

err_close_core_cq:
	retv = fi_close(&rxr_ep->rdm_cq->fid);
	if (retv)
		FI_WARN(&rxr_prov, FI_LOG_CQ, "Unable to close cq: %s\n",
			fi_strerror(-retv));
err_close_core_ep:
	retv = fi_close(&rxr_ep->rdm_ep->fid);
	if (retv)
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL, "Unable to close EP: %s\n",
			fi_strerror(-retv));
err_close_ofi_ep:
	retv = ofi_endpoint_close(&rxr_ep->util_ep);
	if (retv)
		FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
			"Unable to close util EP: %s\n",
			fi_strerror(-retv));
err_free_ep:
	free(rxr_ep);
	return ret;
}
