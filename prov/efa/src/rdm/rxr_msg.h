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

/**
 * @brief Populate the fields in fi_peer_rx_entry object
 * with the equivalences in rxr_op_entry
 *
 * @param[in,out] peer_rx_entry the fi_peer_rx_entry object to be updated
 * @param[in] rx_entry the rxr_op_entry
 * @param[in] op the ofi op code
 */
static inline
void rxr_msg_update_peer_rx_entry(struct fi_peer_rx_entry *peer_rx_entry,
				  struct rxr_op_entry *rx_entry,
				  uint32_t op)
{
	assert(op == ofi_op_msg || op == ofi_op_tagged);

	/*
	 * We cannot pass FI_MULTI_RECV flag to peer provider
	 * because it will write it to the cqe for each completed rx.
	 * However, this flag should only be written to the cqe of either
	 * the last rx entry that consumed the posted buffer, or a dummy cqe
	 * with FI_MULTI_RECV flag only that indicates the multi recv has finished.
	 * We will do the second way, see efa_rdm_srx_free_entry().
	 */
	peer_rx_entry->flags = (rx_entry->fi_flags & ~FI_MULTI_RECV);
	if (rx_entry->desc && rx_entry->desc[0] && ((struct efa_mr *)rx_entry->desc[0])->shm_mr) {
		memcpy(rx_entry->shm_desc, rx_entry->desc, rx_entry->iov_count * sizeof(void *));
		rxr_convert_desc_for_shm(rx_entry->iov_count, rx_entry->shm_desc);
		peer_rx_entry->desc = rx_entry->shm_desc;
	} else {
		peer_rx_entry->desc = NULL;
	}
	peer_rx_entry->iov = rx_entry->iov;
	peer_rx_entry->count = rx_entry->iov_count;
	peer_rx_entry->context = rx_entry->cq_entry.op_context;
	if (op == ofi_op_tagged)
		peer_rx_entry->tag = rx_entry->tag;
}

static inline
void rxr_msg_construct(struct fi_msg *msg, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t addr, void *context, uint64_t data)
{
	msg->msg_iov = iov;
	msg->desc = desc;
	msg->iov_count = count;
	msg->addr = addr;
	msg->context = context;
	msg->data = data;
}

/**
 * @brief Queue an unexp rx entry to unexp msg queues
 *
 * @param ep rxr_ep
 * @param unexp_rx_entry the unexp rx entry to be queued
 */
static inline
void rxr_msg_queue_unexp_rx_entry_for_msgrtm(struct rxr_ep *ep,
					      struct rxr_op_entry *unexp_rx_entry)
{
	struct efa_rdm_peer *peer;

	dlist_insert_tail(&unexp_rx_entry->entry, &ep->rx_unexp_list);
	peer = rxr_ep_get_peer(ep, unexp_rx_entry->addr);
	dlist_insert_tail(&unexp_rx_entry->peer_unexp_entry, &peer->rx_unexp_list);
}

/**
 * @brief Queue an unexp rx entry to unexp tag queues
 *
 * @param ep rxr_ep
 * @param unexp_rx_entry the unexp rx entry to be queued
 */
static inline
void rxr_msg_queue_unexp_rx_entry_for_tagrtm(struct rxr_ep *ep,
					      struct rxr_op_entry *unexp_rx_entry)
{
	struct efa_rdm_peer *peer;

	dlist_insert_tail(&unexp_rx_entry->entry, &ep->rx_unexp_tagged_list);
	peer = rxr_ep_get_peer(ep, unexp_rx_entry->addr);
	dlist_insert_tail(&unexp_rx_entry->peer_unexp_entry, &peer->rx_unexp_tagged_list);
}

/**
 * multi recv related functions
 */


bool rxr_msg_multi_recv_buffer_available(struct rxr_ep *ep,
					 struct rxr_op_entry *rx_entry);

void rxr_msg_multi_recv_handle_completion(struct rxr_ep *ep,
					  struct rxr_op_entry *rx_entry);

void rxr_msg_multi_recv_free_posted_entry(struct rxr_ep *ep,
					  struct rxr_op_entry *rx_entry);

/**
 * functions to allocate rx_entry for two sided operations
 */
struct rxr_op_entry *rxr_msg_alloc_rx_entry(struct rxr_ep *ep,
					    const struct fi_msg *msg,
					    uint32_t op, uint64_t flags,
					    uint64_t tag, uint64_t ignore);

struct rxr_op_entry *rxr_msg_alloc_unexp_rx_entry_for_msgrtm(struct rxr_ep *ep,
							     struct rxr_pkt_entry **pkt_entry);

struct rxr_op_entry *rxr_msg_alloc_unexp_rx_entry_for_tagrtm(struct rxr_ep *ep,
							     struct rxr_pkt_entry **pkt_entry);

void rxr_msg_queue_unexp_rx_entry_for_msgrtm(struct rxr_ep *ep,
					     struct rxr_op_entry *unexp_rx_entry);

void rxr_msg_queue_unexp_rx_entry_for_tagrtm(struct rxr_ep *ep,
					     struct rxr_op_entry *unexp_rx_entry);

struct rxr_op_entry *rxr_msg_split_rx_entry(struct rxr_ep *ep,
					    struct rxr_op_entry *posted_entry,
					    struct rxr_op_entry *consumer_entry,
					    struct rxr_pkt_entry *pkt_entry);
/*
 * The following 2 OP structures are defined in rxr_msg.c and is
 * used by rxr_endpoint()
 */
extern struct fi_ops_msg rxr_ops_msg;

extern struct fi_ops_tagged rxr_ops_tagged;

ssize_t rxr_msg_post_medium_rtm(struct rxr_ep *ep, struct rxr_op_entry *tx_entry);

ssize_t rxr_msg_post_medium_rtm_or_queue(struct rxr_ep *ep, struct rxr_op_entry *tx_entry);
