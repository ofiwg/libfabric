/*
 * Copyright (c) Amazon.com, Inc. or its affiliates.
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
 * with the equivalences in efa_rdm_ope
 *
 * @param[in,out] peer_rxe the fi_peer_rx_entry object to be updated
 * @param[in] rxe the efa_rdm_ope
 * @param[in] op the ofi op code
 */
static inline
void efa_rdm_msg_update_peer_rxe(struct fi_peer_rx_entry *peer_rxe,
				  struct efa_rdm_ope *rxe,
				  uint32_t op)
{
	assert(op == ofi_op_msg || op == ofi_op_tagged);

	/*
	 * We cannot pass FI_MULTI_RECV flag to peer provider
	 * because it will write it to the cqe for each completed rx.
	 * However, this flag should only be written to the cqe of either
	 * the last rxe that consumed the posted buffer, or a dummy cqe
	 * with FI_MULTI_RECV flag only that indicates the multi recv has finished.
	 * We will do the second way, see efa_rdm_srx_free_entry().
	 */
	peer_rxe->flags = (rxe->fi_flags & ~FI_MULTI_RECV);
	if (rxe->desc[0] && ((struct efa_mr *)rxe->desc[0])->shm_mr) {
		memcpy(rxe->shm_desc, rxe->desc, rxe->iov_count * sizeof(void *));
		efa_rdm_get_desc_for_shm(rxe->iov_count, rxe->desc, rxe->shm_desc);
		peer_rxe->desc = rxe->shm_desc;
	} else {
		peer_rxe->desc = NULL;
	}
	peer_rxe->iov = rxe->iov;
	peer_rxe->count = rxe->iov_count;
	peer_rxe->context = rxe->cq_entry.op_context;
	if (op == ofi_op_tagged)
		peer_rxe->tag = rxe->tag;
}

static inline
void efa_rdm_msg_construct(struct fi_msg *msg, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t addr, void *context, uint64_t data)
{
	msg->msg_iov = iov;
	msg->desc = desc;
	msg->iov_count = count;
	msg->addr = addr;
	msg->context = context;
	msg->data = data;
}

static inline
void rxr_tmsg_construct(struct fi_msg_tagged *msg, const struct iovec *iov, void **desc,
		       size_t count, fi_addr_t addr, void *context, uint64_t data, uint64_t tag)
{
	msg->msg_iov = iov;
	msg->desc = desc;
	msg->iov_count = count;
	msg->addr = addr;
	msg->context = context;
	msg->data = data;
	msg->tag = tag;
}

/**
 * @brief Queue an unexp rxe to unexp msg queues
 *
 * @param ep efa_rdm_ep
 * @param unexp_rxe the unexp rxe to be queued
 */
static inline
void efa_rdm_msg_queue_unexp_rxe_for_msgrtm(struct efa_rdm_ep *ep,
					      struct efa_rdm_ope *unexp_rxe)
{
	struct efa_rdm_peer *peer;

	dlist_insert_tail(&unexp_rxe->entry, &ep->rx_unexp_list);
	peer = efa_rdm_ep_get_peer(ep, unexp_rxe->addr);
	dlist_insert_tail(&unexp_rxe->peer_unexp_entry, &peer->rx_unexp_list);
}

/**
 * @brief Queue an unexp rxe to unexp tag queues
 *
 * @param ep efa_rdm_ep
 * @param unexp_rxe the unexp rxe to be queued
 */
static inline
void efa_rdm_msg_queue_unexp_rxe_for_tagrtm(struct efa_rdm_ep *ep,
					      struct efa_rdm_ope *unexp_rxe)
{
	struct efa_rdm_peer *peer;

	dlist_insert_tail(&unexp_rxe->entry, &ep->rx_unexp_tagged_list);
	peer = efa_rdm_ep_get_peer(ep, unexp_rxe->addr);
	dlist_insert_tail(&unexp_rxe->peer_unexp_entry, &peer->rx_unexp_tagged_list);
}

/**
 * multi recv related functions
 */


bool efa_rdm_msg_multi_recv_buffer_available(struct efa_rdm_ep *ep,
					 struct efa_rdm_ope *rxe);

void efa_rdm_msg_multi_recv_handle_completion(struct efa_rdm_ep *ep,
					  struct efa_rdm_ope *rxe);

void efa_rdm_msg_multi_recv_free_posted_entry(struct efa_rdm_ep *ep,
					  struct efa_rdm_ope *rxe);

/**
 * functions to allocate rxe for two sided operations
 */
struct efa_rdm_ope *efa_rdm_msg_alloc_rxe(struct efa_rdm_ep *ep,
					    const struct fi_msg *msg,
					    uint32_t op, uint64_t flags,
					    uint64_t tag, uint64_t ignore);

struct efa_rdm_ope *efa_rdm_msg_alloc_unexp_rxe_for_rtm(struct efa_rdm_ep *ep,
							struct rxr_pkt_entry **pkt_entry_ptr,
							uint32_t op);

struct efa_rdm_ope *efa_rdm_msg_split_rxe(struct efa_rdm_ep *ep,
					    struct efa_rdm_ope *posted_entry,
					    struct efa_rdm_ope *consumer_entry,
					    struct rxr_pkt_entry *pkt_entry);
/*
 * The following 2 OP structures are defined in efa_rdm_msg.c and is
 * used by rxr_endpoint()
 */
extern struct fi_ops_msg efa_rdm_msg_ops;

extern struct fi_ops_tagged efa_rdm_msg_tagged_ops;