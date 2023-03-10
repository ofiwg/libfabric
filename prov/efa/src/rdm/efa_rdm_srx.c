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

#include "efa.h"
#include "efa_rdm_srx.h"
#include "rxr_msg.h"

/**
 * @brief This call is invoked by the peer provider to obtain a
 * peer_rx_entry where an incoming message should be placed.
 * @param[in] srx the fid_peer_srx
 * @param[in] addr the source address of the incoming message
 * @param[in] size the size of the incoming message
 * @param[out] peer_rx_entry the obtained peer_rx_entry
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_get_msg(struct fid_peer_srx *srx, fi_addr_t addr,
		       size_t size, struct fi_peer_rx_entry **peer_rx_entry)
{
    return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the peer provider to obtain a
 * peer_rx_entry where an incoming tagged message should be placed.
 *
 * @param[in] srx the fid_peer_srx
 * @param[in] addr the source address of the incoming message
 * @param[in] size the size of the incoming message
 * @param[out] peer_rx_entry the obtained peer_rx_entry
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_get_tag(struct fid_peer_srx *srx, fi_addr_t addr,
			uint64_t tag, struct fi_peer_rx_entry **peer_rx_entry)
{
    return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the peer provider to queue
 * a received message to owner provider's unexpected queue,
 * when the peer calls owner_ops->get_msg() but the owner fails to
 * find a matching receive buffer.
 *
 * @param[in] peer_rx_entry the entry to be queued
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_queue_msg(struct fi_peer_rx_entry *peer_rx_entry)
{
    return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the peer provider to queue
 * a received tagged message to owner provider's unexpected queue,
 * when the peer calls owner_ops->get_tag() but the owner fails to
 * find a matching receive buffer.
 *
 * @param[in] peer_rx_entry the entry to be queued
 * @return int 0 on success, a negative integer on failure
 */
static int efa_rdm_srx_queue_tag(struct fi_peer_rx_entry *peer_rx_entry)
{
	return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the peer provider to release a peer_rx_entry
 * from owner provider's resource pool
 *
 * @param[in] peer_rx_entry the peer_rx_entry to be freed
 */
static void efa_rdm_srx_free_entry(struct fi_peer_rx_entry *peer_rx_entry)
{
}

/**
 * @brief This call is invoked by the owner provider to start progressing
 * the peer_rx_entry that matches a received message.
 *
 * @param[in] peer_rx_entry the rx entry to be progressed.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_start_msg(struct fi_peer_rx_entry *peer_rx_entry)
{
	struct rxr_op_entry *rx_op_entry;

	rx_op_entry = container_of(peer_rx_entry, struct rxr_op_entry, peer_rx_entry);

	return rxr_pkt_proc_matched_rtm(rx_op_entry->ep, rx_op_entry, peer_rx_entry->owner_context);
}

/**
 * @brief This call is invoked by the owner provider to start progressing
 * the peer_rx_entry that matches a received tagged message.
 *
 * @param[in] peer_rx_entry the fi_peer_rx_entry to be progressed.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_start_tag(struct fi_peer_rx_entry *peer_rx_entry)
{
	struct rxr_op_entry *rx_op_entry;

	rx_op_entry = container_of(peer_rx_entry, struct rxr_op_entry, peer_rx_entry);

	return rxr_pkt_proc_matched_rtm(rx_op_entry->ep, rx_op_entry, peer_rx_entry->owner_context);
}

/**
 * @brief This call is invoked by the owner provider to discard
 * the message and data associated with the specified fi_peer_rx_entry.
 * This often indicates that the application has canceled or discarded
 * the receive operation.
 *
 * @param[in] peer_rx_entry the fi_peer_rx_entry to be discarded.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_discard_msg(struct fi_peer_rx_entry *peer_rx_entry)
{
    return -FI_ENOSYS;
}

/**
 * @brief This call is invoked by the owner provider to discard
 * the tagged message and data associated with the specified fi_peer_rx_entry.
 * This often indicates that the application has canceled or discarded
 * the receive operation.
 *
 * @param[in] peer_rx_entry the fi_peer_rx_entry to be discarded.
 * @return int 0 on success, a negative integer on failure.
 */
static int efa_rdm_srx_discard_tag(struct fi_peer_rx_entry *peer_rx_entry)
{
    return -FI_ENOSYS;
}


static struct fi_ops_srx_owner efa_rdm_srx_owner_ops = {
	.size = sizeof(struct fi_ops_srx_owner),
	.get_msg = efa_rdm_srx_get_msg,
	.get_tag = efa_rdm_srx_get_tag,
	.queue_msg = efa_rdm_srx_queue_msg,
	.queue_tag = efa_rdm_srx_queue_tag,
	.free_entry = efa_rdm_srx_free_entry,
};

static struct fi_ops_srx_peer efa_rdm_srx_peer_ops = {
	.size = sizeof(struct fi_ops_srx_peer),
	.start_msg = efa_rdm_srx_start_msg,
	.start_tag = efa_rdm_srx_start_tag,
	.discard_msg = efa_rdm_srx_discard_msg,
	.discard_tag = efa_rdm_srx_discard_tag,
};

/**
 * @brief Construct peer srx
 *
 * @param[in] rxr_ep rxr_ep
 * @param[out] peer_srx the constructed peer srx
 */
void efa_rdm_peer_srx_construct(struct rxr_ep *rxr_ep, struct fid_peer_srx *peer_srx)
{
	peer_srx->owner_ops = &efa_rdm_srx_owner_ops;
	peer_srx->peer_ops = &efa_rdm_srx_peer_ops;
	/* This is required to bind this srx to peer provider's ep */
	peer_srx->ep_fid.fid.fclass = FI_CLASS_SRX_CTX;
	/* This context will be used in the ops of peer SRX to access the owner provider resources */
	peer_srx->ep_fid.fid.context = rxr_ep;
}
