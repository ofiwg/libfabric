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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>

#include "efa.h"
#include "rxr.h"
#include "rxr_msg.h"
#include "rxr_pkt_cmd.h"

#include "rxr_tp.h"

/**
 * This file define the msg ops functions.
 * It is consisted of the following sections:
 *     send functions,
 *     receive functions and
 *     ops structure
 */

/**
 *  Send function
 */

/**
 * @brief select a two-sided protocol for the send operation
 *
 * @param [in]		rxr_ep		endpoint
 * @param [in]		txe	contains information of the send operation
 * @param [in]		use_p2p		whether p2p can be used
 * @return		the RTM packet type of the two-sided protocol. Four
 *                      types of protocol can be used: eager, medium, longcts, longread.
 *                      Each protocol has tagged/non-tagged version. Some protocols has a DC version.
 * @related		rxr_ep
 */
int rxr_msg_select_rtm(struct rxr_ep *rxr_ep, struct efa_rdm_ope *txe, int use_p2p)
{
	/*
	 * For performance consideration, this function assume the tagged rtm packet type id is
	 * always the correspondent message rtm packet type id + 1, thus the assertion here.
	 */
	assert(RXR_EAGER_MSGRTM_PKT + 1 == RXR_EAGER_TAGRTM_PKT);
	assert(RXR_MEDIUM_MSGRTM_PKT + 1 == RXR_MEDIUM_TAGRTM_PKT);
	assert(RXR_LONGCTS_MSGRTM_PKT + 1 == RXR_LONGCTS_TAGRTM_PKT);
	assert(RXR_LONGREAD_MSGRTM_PKT + 1 == RXR_LONGREAD_TAGRTM_PKT);
	assert(RXR_DC_EAGER_MSGRTM_PKT + 1 == RXR_DC_EAGER_TAGRTM_PKT);
	assert(RXR_DC_MEDIUM_MSGRTM_PKT + 1 == RXR_DC_MEDIUM_TAGRTM_PKT);
	assert(RXR_DC_LONGCTS_MSGRTM_PKT + 1 == RXR_DC_LONGCTS_TAGRTM_PKT);

	int tagged;
	int eager_rtm, medium_rtm, longcts_rtm, readbase_rtm, iface;
	size_t eager_rtm_max_data_size;
	struct efa_rdm_peer *peer;
	struct efa_hmem_info *hmem_info;
	bool delivery_complete_requested;

	assert(txe->op == ofi_op_msg || txe->op == ofi_op_tagged);
	tagged = (txe->op == ofi_op_tagged);
	assert(tagged == 0 || tagged == 1);

	peer = rxr_ep_get_peer(rxr_ep, txe->addr);
	assert(peer);

	iface = txe->desc[0] ? ((struct efa_mr*) txe->desc[0])->peer.iface : FI_HMEM_SYSTEM;
	hmem_info = rxr_ep_domain(rxr_ep)->hmem_info;

	if (txe->fi_flags & FI_INJECT)
		delivery_complete_requested = false;
	else
		delivery_complete_requested = txe->fi_flags & FI_DELIVERY_COMPLETE;

	eager_rtm = (delivery_complete_requested) ? RXR_DC_EAGER_MSGRTM_PKT + tagged
						  : RXR_EAGER_MSGRTM_PKT + tagged;

	medium_rtm = (delivery_complete_requested) ? RXR_DC_MEDIUM_MSGRTM_PKT + tagged
						   :  RXR_MEDIUM_MSGRTM_PKT + tagged;

	longcts_rtm = (delivery_complete_requested) ? RXR_DC_LONGCTS_MSGRTM_PKT + tagged
						    : RXR_LONGCTS_MSGRTM_PKT + tagged;

	eager_rtm_max_data_size = efa_rdm_txe_max_req_data_capacity(rxr_ep, txe, eager_rtm);

	readbase_rtm = rxr_pkt_type_readbase_rtm(peer, txe->op, txe->fi_flags, &hmem_info[iface]);

	if (txe->total_len >= hmem_info[iface].min_read_msg_size &&
		efa_rdm_ep_support_rdma_read(rxr_ep) &&
		(txe->desc[0] || efa_is_cache_available(rxr_ep_domain(rxr_ep))))
		return readbase_rtm;

	if (txe->total_len <= eager_rtm_max_data_size)
		return eager_rtm;

	if (txe->total_len <= hmem_info[iface].max_medium_msg_size)
		return medium_rtm;

	return longcts_rtm;
}

/**
 * @brief post RTM packet(s) for a send operation
 *
 * @param[in,out]	ep		endpoint
 * @param[in,out]	txe	information of the send operation.
 * @param[in]		use_p2p		whether p2p can be used for this send operation.
 * @retval		0 if packet(s) was posted successfully.
 * @retval		-FI_ENOSUPP if the send operation requires an extra feature,
 * 			which peer does not support.
 * @retval		-FI_EAGAIN for temporary out of resources for send
 */
ssize_t rxr_msg_post_rtm(struct rxr_ep *ep, struct efa_rdm_ope *txe, int use_p2p)
{
	ssize_t err;
	int rtm_type;
	struct efa_rdm_peer *peer;

	peer = rxr_ep_get_peer(ep, txe->addr);
	assert(peer);

	rtm_type = rxr_msg_select_rtm(ep, txe, use_p2p);
	assert(rtm_type >= RXR_REQ_PKT_BEGIN);

	if (rtm_type < RXR_EXTRA_REQ_PKT_BEGIN) {
		/* rtm requires only baseline feature, which peer should always support. */
		return rxr_pkt_post_req(ep, txe, rtm_type, 0);
	}

	/*
	 * rtm_type requires an extra feature, which peer might not support.
	 *
	 * Check handshake packet from peer to verify support status.
	 */
	if (!(peer->flags & EFA_RDM_PEER_HANDSHAKE_RECEIVED)) {
		err = rxr_pkt_trigger_handshake(ep, txe->addr, peer);
		return err ? err : -FI_EAGAIN;
	}

	if (!rxr_pkt_req_supported_by_peer(rtm_type, peer))
		return -FI_EOPNOTSUPP;

	return rxr_pkt_post_req(ep, txe, rtm_type, 0);
}

ssize_t rxr_msg_generic_send(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t tag, uint32_t op, uint64_t flags)
{
	struct rxr_ep *rxr_ep;
	ssize_t err, ret, use_p2p;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	assert(msg->iov_count <= rxr_ep->tx_iov_limit);

	efa_perfset_start(rxr_ep, perf_efa_tx);
	ofi_mutex_lock(&rxr_ep->base_ep.util_ep.lock);

	peer = rxr_ep_get_peer(rxr_ep, msg->addr);
	assert(peer);

	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF) {
		err = -FI_EAGAIN;
		goto out;
	}

	txe = rxr_ep_alloc_txe(rxr_ep, msg, op, tag, flags);
	if (OFI_UNLIKELY(!txe)) {
		err = -FI_EAGAIN;
		rxr_ep_progress_internal(rxr_ep);
		goto out;
	}

	ret = rxr_ep_use_p2p(rxr_ep, txe->desc[0]);
	if (ret < 0) {
		err = ret;
		goto out;
	}

	use_p2p = ret;

	EFA_DBG(FI_LOG_EP_DATA,
	       "iov_len: %lu tag: %lx op: %x flags: %lx\n",
	       txe->total_len,
	       tag, op, flags);

	assert(txe->op == ofi_op_msg || txe->op == ofi_op_tagged);

	txe->msg_id = peer->next_msg_id++;

	rxr_tracepoint(send_begin, txe->msg_id, 
		    (size_t) txe->cq_entry.op_context, txe->total_len);
	rxr_tracepoint(send_begin_msg_context, 
		    (size_t) msg->context, (size_t) msg->addr);


	err = rxr_msg_post_rtm(rxr_ep, txe, use_p2p);
	if (OFI_UNLIKELY(err)) {
		rxr_ep_progress_internal(rxr_ep);
		efa_rdm_txe_release(txe);
		peer->next_msg_id--;
	}

out:
	ofi_mutex_unlock(&rxr_ep->base_ep.util_ep.lock);
	efa_perfset_end(rxr_ep, perf_efa_tx);
	return err;
}

/**
 *   Non-tagged send ops function
 */
static
ssize_t rxr_msg_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	struct efa_rdm_peer *peer;
	struct rxr_ep *rxr_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};
	struct fi_msg *shm_msg;
	void **efa_desc = NULL;
	fi_addr_t efa_addr;
	int ret;

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, msg->addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		shm_msg = (struct fi_msg *)msg;
		if (msg->desc) {
			efa_desc = msg->desc;
			rxr_get_desc_for_shm(msg->iov_count, msg->desc, shm_desc);
			shm_msg->desc = shm_desc;
		}
		efa_addr = msg->addr;
		shm_msg->addr = peer->shm_fiaddr;
		ret = fi_sendmsg(rxr_ep->shm_ep, shm_msg, flags);
		/* Recover the application msg */
		if (efa_desc)
			shm_msg->desc = efa_desc;
		shm_msg->addr = efa_addr;
		return ret;
	}

	return rxr_msg_generic_send(ep, msg, 0, ofi_op_msg, flags);
}

static
ssize_t rxr_msg_sendv(struct fid_ep *ep, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t dest_addr,
		      void *context)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg msg = {0};
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		rxr_get_desc_for_shm(count, desc, shm_desc);
		return fi_sendv(rxr_ep->shm_ep, iov, shm_desc, count, peer->shm_fiaddr, context);
	}

	rxr_msg_construct(&msg, iov, desc, count, dest_addr, context, 0);
	return rxr_msg_sendmsg(ep, &msg, rxr_tx_flags(rxr_ep));
}

static
ssize_t rxr_msg_send(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr, void *context)
{
	struct iovec iov;
	struct efa_rdm_peer *peer;
	struct rxr_ep *rxr_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		if (desc)
			rxr_get_desc_for_shm(1, &desc, shm_desc);
		return fi_send(rxr_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, peer->shm_fiaddr, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return rxr_msg_sendv(ep, &iov, &desc, 1, dest_addr, context);
}

static
ssize_t rxr_msg_senddata(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct rxr_ep *rxr_ep;
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		if (desc)
			rxr_get_desc_for_shm(1, &desc, shm_desc);
		return fi_senddata(rxr_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, data, peer->shm_fiaddr, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, &desc, 1, dest_addr, context, data);
	return rxr_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(rxr_ep) | FI_REMOTE_CQ_DATA);
}

static
ssize_t rxr_msg_inject(struct fid_ep *ep, const void *buf, size_t len,
		       fi_addr_t dest_addr)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	if (len > rxr_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		return fi_inject(rxr_ep->shm_ep, buf, len, peer->shm_fiaddr);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, 0);

	return rxr_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(rxr_ep) | EFA_RDM_TXE_NO_COMPLETION | FI_INJECT);
}

static
ssize_t rxr_msg_injectdata(struct fid_ep *ep, const void *buf,
			   size_t len, uint64_t data,
			   fi_addr_t dest_addr)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg msg;
	struct iovec iov;
	struct efa_rdm_peer *peer;

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	if (len > rxr_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		return fi_injectdata(rxr_ep->shm_ep, buf, len, data, peer->shm_fiaddr);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, data);

	return rxr_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(rxr_ep) | EFA_RDM_TXE_NO_COMPLETION |
				    FI_REMOTE_CQ_DATA | FI_INJECT);
}

/**
 *   Tagged send op functions
 */
static
ssize_t rxr_msg_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *tmsg,
			 uint64_t flags)
{
	struct fi_msg msg = {0};
	struct efa_rdm_peer *peer;
	struct rxr_ep *rxr_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};
	struct fi_msg_tagged *shm_tmsg;
	void **efa_desc = NULL;
	fi_addr_t efa_addr;
	int ret;

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, tmsg->addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		shm_tmsg = (struct fi_msg_tagged *)tmsg;
		if (tmsg->desc) {
			efa_desc = tmsg->desc;
			rxr_get_desc_for_shm(tmsg->iov_count, tmsg->desc, shm_desc);
			shm_tmsg->desc = shm_desc;
		}
		efa_addr = tmsg->addr;
		shm_tmsg->addr = peer->shm_fiaddr;
		ret = fi_tsendmsg(rxr_ep->shm_ep, shm_tmsg, flags);
		/* Recover the application msg */
		if (efa_desc)
			shm_tmsg->desc = efa_desc;
		shm_tmsg->addr = efa_addr;
		return ret;
	}

	rxr_msg_construct(&msg, tmsg->msg_iov, tmsg->desc, tmsg->iov_count, tmsg->addr, tmsg->context, tmsg->data);
	return rxr_msg_generic_send(ep_fid, &msg, tmsg->tag, ofi_op_tagged, flags);
}

static
ssize_t rxr_msg_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t dest_addr,
		       uint64_t tag, void *context)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg_tagged msg = {0};
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		if (desc)
			rxr_get_desc_for_shm(count, desc, shm_desc);
		return fi_tsendv(rxr_ep->shm_ep, iov, desc? shm_desc : NULL, count, peer->shm_fiaddr, tag, context);
	}

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.tag = tag;

	return rxr_msg_tsendmsg(ep_fid, &msg, rxr_tx_flags(rxr_ep));
}

static
ssize_t rxr_msg_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, fi_addr_t dest_addr, uint64_t tag,
		      void *context)
{
	struct iovec msg_iov;
	struct efa_rdm_peer *peer;
	struct rxr_ep *rxr_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		if (desc)
			rxr_get_desc_for_shm(1, &desc, shm_desc);
		return fi_tsend(rxr_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, peer->shm_fiaddr, tag, context);
	}

	msg_iov.iov_base = (void *)buf;
	msg_iov.iov_len = len;
	return rxr_msg_tsendv(ep_fid, &msg_iov, &desc, 1, dest_addr, tag,
			      context);
}

static
ssize_t rxr_msg_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct rxr_ep *rxr_ep;
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		if (desc)
			rxr_get_desc_for_shm(1, &desc, shm_desc);
		return fi_tsenddata(rxr_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, data, peer->shm_fiaddr, tag, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, &desc, 1, dest_addr, context, data);
	return rxr_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(rxr_ep) | FI_REMOTE_CQ_DATA);
}

static
ssize_t rxr_msg_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	if (len > rxr_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		return fi_tinject(rxr_ep->shm_ep, buf, len, peer->shm_fiaddr, tag);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, 0);

	return rxr_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(rxr_ep) | EFA_RDM_TXE_NO_COMPLETION | FI_INJECT);
}

static
ssize_t rxr_msg_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			    uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct rxr_ep *rxr_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	rxr_ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	if (len > rxr_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = rxr_ep_get_peer(rxr_ep, dest_addr);
	assert(peer);
	if (peer->is_local && rxr_ep->use_shm_for_tx) {
		return fi_tinjectdata(rxr_ep->shm_ep, buf, len, data, peer->shm_fiaddr, tag);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, data);

	return rxr_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(rxr_ep) | EFA_RDM_TXE_NO_COMPLETION |
				    FI_REMOTE_CQ_DATA | FI_INJECT);
}

/**
 *  Receive functions
 */

/**
 *   Utility functions and data structures
 */
struct rxr_match_info {
	uint64_t tag;
	uint64_t ignore;
};

/**
 * @brief match function for rxe in ep->unexp_tagged_list
 *
 * @param[in]	item	pointer to rxe->entry.
 * @param[in]	arg	pointer to rxr_match_info
 * @return   0 or 1 indicating wether this entry is a match
 */
static
int rxr_msg_match_ep_unexp_by_tag(struct dlist_entry *item, const void *arg)
{
	const struct rxr_match_info *match_info = arg;
	struct efa_rdm_ope *rxe;

	rxe = container_of(item, struct efa_rdm_ope, entry);

	return ofi_match_tag(rxe->tag, match_info->ignore,
			     match_info->tag);
}

/**
 * @brief match function for rxe in peer->unexp_tagged_list
 *
 * @param[in]	item	pointer to rxe->peer_unexp_entry.
 * @param[in]	arg	pointer to rxr_match_info
 * @return   0 or 1 indicating wether this entry is a match
 */
static
int rxr_msg_match_peer_unexp_by_tag(struct dlist_entry *item, const void *arg)
{
	const struct rxr_match_info *match_info = arg;
	struct efa_rdm_ope *rxe;

	rxe = container_of(item, struct efa_rdm_ope, peer_unexp_entry);

	return ofi_match_tag(rxe->tag, match_info->ignore,
			     match_info->tag);
}

int rxr_msg_handle_unexp_match(struct rxr_ep *ep,
			       struct efa_rdm_ope *rxe,
			       uint64_t tag, uint64_t ignore,
			       void *context, fi_addr_t addr,
			       uint32_t op, uint64_t flags)
{
	struct rxr_pkt_entry *pkt_entry = NULL;
	struct fid_peer_srx *srx;
	uint64_t data_len;
	int ret;

	rxe->fi_flags = flags;
	rxe->ignore = ignore;
	rxe->state = EFA_RDM_RXE_MATCHED;

	data_len = rxe->total_len;

	if (!(rxe->rxr_flags & EFA_RDM_RXE_FOR_PEER_SRX)) {
		pkt_entry = rxe->unexp_pkt;
		rxe->unexp_pkt = NULL;
	}

	rxe->cq_entry.op_context = context;
	rxe->cq_entry.buf = rxe->iov[0].iov_base;
	data_len = MIN(rxe->total_len,
		       ofi_total_iov_len(rxe->iov, rxe->iov_count));
	rxe->cq_entry.len = data_len;
	rxe->cq_entry.flags = (FI_RECV | FI_MSG);

	if (op == ofi_op_tagged) {
		rxe->cq_entry.flags |= FI_TAGGED;
		rxe->cq_entry.tag = rxe->tag;
		rxe->ignore = ignore;
	} else {
		rxe->cq_entry.tag = 0;
		rxe->ignore = ~0;
	}

	srx = rxe->peer_rxe.srx;

	rxr_msg_update_peer_rxe(&rxe->peer_rxe, rxe, op);

	rxe->peer_rxe.owner_context = pkt_entry;

	/* We need to unlock this lock first to make peer provider acquire it inside the start ops below */
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);

	if (op == ofi_op_msg)
		ret = srx->peer_ops->start_msg(&rxe->peer_rxe);
	else
		ret = srx->peer_ops->start_tag(&rxe->peer_rxe);

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);
	return ret;
}

/**
 * @brief allocate an rxe for a fi_msg.
 *        This function is used by two sided operation only.
 *
 * @param[in] ep	end point
 * @param[in] msg	fi_msg contains iov,iov_count,context for ths operation
 * @param[in] op	operation type (ofi_op_msg or ofi_op_tagged)
 * @param[in] flags	flags application used to call fi_recv/fi_trecv functions
 * @param[in] tag	tag (used only if op is ofi_op_tagged)
 * @param[in] ignore	ignore mask (used only if op is ofi_op_tagged)
 * @return		if allocation succeeded, return pointer to rxe
 * 			if allocation failed, return NULL
 */
struct efa_rdm_ope *rxr_msg_alloc_rxe(struct rxr_ep *ep,
					    const struct fi_msg *msg,
					    uint32_t op, uint64_t flags,
					    uint64_t tag, uint64_t ignore)
{
	struct efa_rdm_ope *rxe;
	fi_addr_t addr;

	if (ep->base_ep.util_ep.caps & FI_DIRECTED_RECV)
		addr = msg->addr;
	else
		addr = FI_ADDR_UNSPEC;

	rxe = rxr_ep_alloc_rxe(ep, addr, op);
	if (!rxe)
		return NULL;

	rxe->fi_flags = flags;
	if (op == ofi_op_tagged) {
		rxe->tag = tag;
		rxe->cq_entry.tag = tag;
		rxe->ignore = ignore;
	}

	/* Handle case where we're allocating an unexpected rxe */
	rxe->iov_count = msg->iov_count;
	if (rxe->iov_count) {
		assert(msg->msg_iov);
		memcpy(rxe->iov, msg->msg_iov, sizeof(*rxe->iov) * msg->iov_count);
		rxe->cq_entry.len = ofi_total_iov_len(rxe->iov, rxe->iov_count);
		rxe->cq_entry.buf = msg->msg_iov[0].iov_base;
	}

	if (msg->desc)
		memcpy(&rxe->desc[0], msg->desc, sizeof(*msg->desc) * msg->iov_count);
	else
		memset(&rxe->desc[0], 0, sizeof(rxe->desc));

	rxe->cq_entry.op_context = msg->context;

	rxr_msg_update_peer_rxe(&rxe->peer_rxe, rxe, op);

	return rxe;
}

struct efa_rdm_ope *rxr_msg_alloc_unexp_rxe_for_msgrtm(struct rxr_ep *ep,
							     struct rxr_pkt_entry **pkt_entry_ptr)
{
	struct efa_rdm_ope *rxe;
	struct rxr_pkt_entry *unexp_pkt_entry;
	enum rxr_pkt_entry_alloc_type type;

	type = (*pkt_entry_ptr)->alloc_type;

	if (type == RXR_PKT_FROM_PEER_SRX) {
		unexp_pkt_entry = *pkt_entry_ptr;
	} else {
		unexp_pkt_entry = rxr_pkt_get_unexp(ep, pkt_entry_ptr);
		if (OFI_UNLIKELY(!unexp_pkt_entry)) {
			EFA_WARN(FI_LOG_CQ, "packet entries exhausted.\n");
			return NULL;
		}
	}

	rxe = rxr_ep_alloc_rxe(ep, unexp_pkt_entry->addr, ofi_op_msg);
	if (OFI_UNLIKELY(!rxe))
		return NULL;

	rxe->rxr_flags = 0;
	rxe->state = EFA_RDM_RXE_UNEXP;
	/*
	 * The pkt entry from peer srx is transient, we cannot store it in rxe->unexp_pkt.
	 * All the required information from this pkt is already updated to rxe
	 * via rxr_pkt_rtm_update_rxe().
	 */
	rxe->unexp_pkt = (type == RXR_PKT_FROM_PEER_SRX) ? NULL : unexp_pkt_entry;
	rxr_pkt_rtm_update_rxe(unexp_pkt_entry, rxe);
	return rxe;
}

struct efa_rdm_ope *rxr_msg_alloc_unexp_rxe_for_tagrtm(struct rxr_ep *ep,
							     struct rxr_pkt_entry **pkt_entry_ptr)
{
	struct efa_rdm_ope *rxe;
	struct rxr_pkt_entry *unexp_pkt_entry;

	unexp_pkt_entry = rxr_pkt_get_unexp(ep, pkt_entry_ptr);
	if (OFI_UNLIKELY(!unexp_pkt_entry)) {
		EFA_WARN(FI_LOG_CQ, "packet entries exhausted.\n");
		return NULL;
	}

	rxe = rxr_ep_alloc_rxe(ep, unexp_pkt_entry->addr, ofi_op_tagged);
	if (OFI_UNLIKELY(!rxe))
		return NULL;

	rxe->tag = rxr_pkt_rtm_tag(unexp_pkt_entry);
	rxe->rxr_flags = 0;
	rxe->state = EFA_RDM_RXE_UNEXP;
	rxe->unexp_pkt = unexp_pkt_entry;
	rxr_pkt_rtm_update_rxe(unexp_pkt_entry, rxe);
	return rxe;
}

struct efa_rdm_ope *rxr_msg_split_rxe(struct rxr_ep *ep,
					    struct efa_rdm_ope *posted_entry,
					    struct efa_rdm_ope *consumer_entry,
					    struct rxr_pkt_entry *pkt_entry)
{
	struct efa_rdm_ope *rxe;
	size_t buf_len, consumed_len, data_len;
	uint64_t tag, ignore;
	struct fi_msg msg = {0};

	if (!consumer_entry) {
		tag = 0;
		ignore = ~0;
		msg.msg_iov = posted_entry->iov;
		msg.iov_count = posted_entry->iov_count;
		msg.addr = pkt_entry->addr;
		rxe = rxr_msg_alloc_rxe(ep, &msg,
						  ofi_op_msg,
						  posted_entry->fi_flags,
						  tag, ignore);
		if (OFI_UNLIKELY(!rxe))
			return NULL;

		EFA_DBG(FI_LOG_EP_CTRL,
		       "Splitting into new multi_recv consumer rxe %d from rxe %d\n",
		       rxe->rx_id,
		       posted_entry->rx_id);
		assert(rxr_get_base_hdr(pkt_entry->wiredata)->type >= RXR_REQ_PKT_BEGIN);
		rxr_pkt_rtm_update_rxe(pkt_entry, rxe);
	} else {
		rxe = consumer_entry;
		memcpy(rxe->iov, posted_entry->iov,
		       sizeof(*posted_entry->iov) * posted_entry->iov_count);
		rxe->iov_count = posted_entry->iov_count;
	}

	data_len = rxe->total_len;
	buf_len = ofi_total_iov_len(rxe->iov,
				    rxe->iov_count);
	consumed_len = MIN(buf_len, data_len);

	rxe->rxr_flags |= EFA_RDM_RXE_MULTI_RECV_CONSUMER;
	rxe->fi_flags |= FI_MULTI_RECV;
	rxe->master_entry = posted_entry;
	rxe->cq_entry.len = consumed_len;
	rxe->cq_entry.buf = rxe->iov[0].iov_base;
	rxe->cq_entry.op_context = posted_entry->cq_entry.op_context;
	rxe->cq_entry.flags = (FI_RECV | FI_MSG);

	ofi_consume_iov(posted_entry->iov, &posted_entry->iov_count,
			consumed_len);

	dlist_init(&rxe->multi_recv_entry);
	dlist_insert_tail(&rxe->multi_recv_entry,
			  &posted_entry->multi_recv_consumers);
	return rxe;
}

/**
 * @brief find an unexpected rxe for a receive operation.
 *
 * @param[in]	ep	endpoint
 * @param[in]	addr	fi_addr of the peer want to receive from, can be FI_ADDR_UNSPEC
 * @param[in]	tag	tag of the unexpected message, used only if op is ofi_op_tagged.
 * @param[in]	ignore	mask of the tag, used only if op is ofi_op_tagged.
 * @param[in]	op	either ofi_op_tagged or ofi_op_msg.
 * @param[in]	claim   whether to claim the rxe, e.g. remove it from unexpected queue.
 * @return	If an unexpected rxe was found, return the pointer.
 * 		Otherwise, return NULL.
 */
static inline
struct efa_rdm_ope *rxr_msg_find_unexp_rxe(struct rxr_ep *ep, fi_addr_t addr,
						 int64_t tag, uint64_t ignore, uint32_t op,
						 bool claim)
{
	struct rxr_match_info match_info;
	struct efa_rdm_ope *rxe;
	struct dlist_entry *match;
	struct efa_rdm_peer *peer;

	peer = (ep->base_ep.util_ep.caps & FI_DIRECTED_RECV) ? rxr_ep_get_peer(ep, addr) : NULL;

	switch(op) {
	case ofi_op_msg:
		if (peer) {
			match = dlist_empty(&peer->rx_unexp_list) ? NULL : peer->rx_unexp_list.next;
			rxe = match ? container_of(match, struct efa_rdm_ope, peer_unexp_entry) : NULL;
		} else {
			match = dlist_empty(&ep->rx_unexp_list) ? NULL : ep->rx_unexp_list.next;
			rxe = match ? container_of(match, struct efa_rdm_ope, entry) : NULL;
		}
		break;
	case ofi_op_tagged:
		match_info.tag = tag;
		match_info.ignore = ignore;

		if (peer) {
			match = dlist_find_first_match(&peer->rx_unexp_tagged_list,
			                               rxr_msg_match_peer_unexp_by_tag,
						       (void *)&match_info);
			rxe = match ? container_of(match, struct efa_rdm_ope, peer_unexp_entry) : NULL;
		} else {
			match = dlist_find_first_match(&ep->rx_unexp_tagged_list,
						       rxr_msg_match_ep_unexp_by_tag,
						       (void *)&match_info);
			rxe = match ? container_of(match, struct efa_rdm_ope, entry) : NULL;
		}
		break;
	default:
		EFA_WARN(FI_LOG_CQ, "Error: wrong op in rxr_msg_find_unexp_rxe()");
		abort();
	}

	if (rxe && claim) {
		dlist_remove(&rxe->entry);
		dlist_remove(&rxe->peer_unexp_entry);
	}

	return rxe;
}

/*
 *    Search unexpected list for matching message and process it if found.
 *    Returns 0 if the message is processed, -FI_ENOMSG if no match is found.
 */
static
int rxr_msg_proc_unexp_msg_list(struct rxr_ep *ep, const struct fi_msg *msg,
				uint64_t tag, uint64_t ignore, uint32_t op, uint64_t flags,
				struct efa_rdm_ope *posted_entry)
{
	struct efa_rdm_ope *rxe;
	bool claim;

	claim = true;
	rxe = rxr_msg_find_unexp_rxe(ep, msg->addr, tag, ignore, op, claim);
	if (!rxe)
		return -FI_ENOMSG;

	/* 
	 * TODO: Use a realiable way to trigger this function. Can we swap packet order with fake-rdma-core?
	 * NOTE: Cannot trigger this routine, didn't debug.
	 */
	rxr_tracepoint(msg_match_unexpected, rxe->msg_id, 
		    (size_t) rxe->cq_entry.op_context, rxe->total_len, 
		    (int) tag, msg->addr);
	/*
	 * Initialize the matched entry as a multi-recv consumer if the posted
	 * buffer is a multi-recv buffer.
	 */
	if (posted_entry) {
		/*
		 * rxr_msg_split_rxe will setup rxe iov and count
		 */
		rxe = rxr_msg_split_rxe(ep, posted_entry, rxe,
						 NULL);
		if (OFI_UNLIKELY(!rxe)) {
			EFA_WARN(FI_LOG_CQ,
				"RX entries exhausted.\n");
			return -FI_ENOBUFS;
		}
	} else {
		memcpy(rxe->iov, msg->msg_iov, sizeof(*rxe->iov) * msg->iov_count);
		rxe->iov_count = msg->iov_count;
	}

	if (msg->desc)
		memcpy(rxe->desc, msg->desc, sizeof(void*) * msg->iov_count);

	EFA_DBG(FI_LOG_EP_CTRL,
	       "Match found in unexp list for a posted recv msg_id: %" PRIu32
	       " total_len: %" PRIu64 " tag: %lx\n",
	       rxe->msg_id, rxe->total_len, rxe->tag);

	return rxr_msg_handle_unexp_match(ep, rxe, tag, ignore,
					 msg->context, msg->addr, op, flags);
}

bool rxr_msg_multi_recv_buffer_available(struct rxr_ep *ep,
					 struct efa_rdm_ope *rxe)
{
	assert(rxe->fi_flags & FI_MULTI_RECV);
	assert(rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_POSTED);

	return (ofi_total_iov_len(rxe->iov, rxe->iov_count)
		>= ep->min_multi_recv_size);
}

static inline
bool rxr_msg_multi_recv_buffer_complete(struct rxr_ep *ep,
					struct efa_rdm_ope *rxe)
{
	assert(rxe->fi_flags & FI_MULTI_RECV);
	assert(rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_POSTED);

	return (!rxr_msg_multi_recv_buffer_available(ep, rxe) &&
		dlist_empty(&rxe->multi_recv_consumers));
}

void rxr_msg_multi_recv_free_posted_entry(struct rxr_ep *ep,
					  struct efa_rdm_ope *rxe)
{
	assert(!(rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_POSTED));

	if ((rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_CONSUMER) &&
	    rxr_msg_multi_recv_buffer_complete(ep, rxe->master_entry))
		efa_rdm_rxe_release(rxe->master_entry);
}

static
ssize_t rxr_msg_multi_recv(struct rxr_ep *rxr_ep, const struct fi_msg *msg,
			   uint64_t tag, uint64_t ignore, uint32_t op, uint64_t flags)
{
	struct efa_rdm_ope *rxe;
	int ret = 0;

	/*
	 * Always get new rxe of type EFA_RDM_RXE_MULTI_RECV_POSTED when in the
	 * multi recv path. The posted entry will not be used for receiving
	 * messages but will be used for tracking the application's buffer and
	 * when to write the completion to release the buffer.
	 */
	rxe = rxr_msg_alloc_rxe(rxr_ep, msg, op, flags, tag, ignore);
	if (OFI_UNLIKELY(!rxe)) {
		rxr_ep_progress_internal(rxr_ep);
		return -FI_EAGAIN;
	}

	if (rxe->cq_entry.len < rxr_ep->min_multi_recv_size) {
		EFA_WARN(FI_LOG_EP_CTRL, "invalid size (%ld) for multi_recv! expected to be >= %ld\n",
			rxe->cq_entry.len, rxr_ep->min_multi_recv_size);
		efa_rdm_rxe_release(rxe);
		return -FI_EINVAL;
	}

	if (op == ofi_op_tagged) {
		EFA_WARN(FI_LOG_EP_CTRL, "tagged recv cannot be applied to multi_recv!\n");
		efa_rdm_rxe_release(rxe);
		return -FI_EINVAL;
	}

	rxe->rxr_flags |= EFA_RDM_RXE_MULTI_RECV_POSTED;
	dlist_init(&rxe->multi_recv_consumers);
	dlist_init(&rxe->multi_recv_entry);

	while (!dlist_empty(&rxr_ep->rx_unexp_list)) {
		ret = rxr_msg_proc_unexp_msg_list(rxr_ep, msg, tag,
						  ignore, op, flags, rxe);

		if (!rxr_msg_multi_recv_buffer_available(rxr_ep, rxe)) {
			/*
			 * Multi recv buffer consumed by short, unexp messages,
			 * free posted rxe.
			 */
			if (rxr_msg_multi_recv_buffer_complete(rxr_ep, rxe))
				efa_rdm_rxe_release(rxe);
			/*
			 * Multi recv buffer has been consumed, but waiting on
			 * long msg completion. Last msg completion will free
			 * posted rxe.
			 */
			if (ret == -FI_ENOMSG)
				return 0;
			return ret;
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

	dlist_insert_tail(&rxe->entry, &rxr_ep->rx_list);
	return ret;
}

void rxr_msg_multi_recv_handle_completion(struct rxr_ep *ep,
					  struct efa_rdm_ope *rxe)
{
	assert(!(rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_POSTED) &&
	       (rxe->rxr_flags & EFA_RDM_RXE_MULTI_RECV_CONSUMER));

	dlist_remove(&rxe->multi_recv_entry);
	rxe->rxr_flags &= ~EFA_RDM_RXE_MULTI_RECV_CONSUMER;

	if (!rxr_msg_multi_recv_buffer_complete(ep, rxe->master_entry))
		return;

	/*
	 * Buffer is consumed and all messages have been received. Update the
	 * last message to release the application buffer.
	 */
	rxe->cq_entry.flags |= FI_MULTI_RECV;
}

/*
 *     create a rxe and verify in unexpected message list
 *     else add to posted recv list
 */
static
ssize_t rxr_msg_generic_recv(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t tag, uint64_t ignore, uint32_t op,
			     uint64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *rxr_ep;
	struct dlist_entry *unexp_list;
	struct efa_rdm_ope *rxe;

	rxr_ep = container_of(ep, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	assert(msg->iov_count <= rxr_ep->rx_iov_limit);

	efa_perfset_start(rxr_ep, perf_efa_recv);

	ofi_mutex_lock(&rxr_ep->base_ep.util_ep.lock);

	if (flags & FI_MULTI_RECV) {
		ret = rxr_msg_multi_recv(rxr_ep, msg, tag, ignore, op, flags);
		goto out;
	}

	unexp_list = (op == ofi_op_tagged) ? &rxr_ep->rx_unexp_tagged_list :
		     &rxr_ep->rx_unexp_list;

	if (!dlist_empty(unexp_list)) {
		ret = rxr_msg_proc_unexp_msg_list(rxr_ep, msg, tag,
						  ignore, op, flags, NULL);

		if (ret != -FI_ENOMSG)
			goto out;
		ret = 0;
	}

	rxe = rxr_msg_alloc_rxe(rxr_ep, msg, op, flags, tag, ignore);

	if (OFI_UNLIKELY(!rxe)) {
		ret = -FI_EAGAIN;
		rxr_ep_progress_internal(rxr_ep);
		goto out;
	}

	EFA_DBG(FI_LOG_EP_DATA,
	       "%s: iov_len: %lu tag: %lx ignore: %lx op: %x flags: %lx\n",
	       __func__, rxe->total_len, tag, ignore,
	       op, flags);

	rxr_tracepoint(recv_begin, rxe->msg_id, 
		    (size_t) rxe->cq_entry.op_context, rxe->total_len);
	rxr_tracepoint(recv_begin_msg_context, (size_t) msg->context, (size_t) msg->addr);

	if (rxr_ep->use_zcpy_rx) {
		ret = rxr_ep_post_user_recv_buf(rxr_ep, rxe, flags);
		if (ret == -FI_EAGAIN)
			rxr_ep_progress_internal(rxr_ep);
	} else if (op == ofi_op_tagged) {
		dlist_insert_tail(&rxe->entry, &rxr_ep->rx_tagged_list);
	} else {
		dlist_insert_tail(&rxe->entry, &rxr_ep->rx_list);
	}

out:
	ofi_mutex_unlock(&rxr_ep->base_ep.util_ep.lock);

	efa_perfset_end(rxr_ep, perf_efa_recv);
	return ret;
}

static
ssize_t rxr_msg_discard_trecv(struct rxr_ep *ep,
			      struct efa_rdm_ope *rxe,
			      const struct fi_msg_tagged *msg,
			      int64_t flags)
{
	int ret;

	if ((flags & FI_DISCARD) && !(flags & (FI_PEEK | FI_CLAIM)))
		return -FI_EINVAL;

	rxe->fi_flags |= FI_DISCARD;
	rxe->rxr_flags |= EFA_RDM_RXE_RECV_CANCEL;
	ret = ofi_cq_write(ep->base_ep.util_ep.rx_cq, msg->context,
			   FI_TAGGED | FI_RECV | FI_MSG,
			   0, NULL, rxe->cq_entry.data,
			   rxe->cq_entry.tag);
	return ret;
}

static
ssize_t rxr_msg_claim_trecv(struct fid_ep *ep_fid,
			    const struct fi_msg_tagged *msg,
			    int64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *ep;
	struct efa_rdm_ope *rxe;
	struct fi_context *context;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);
	ofi_mutex_lock(&ep->base_ep.util_ep.lock);

	context = (struct fi_context *)msg->context;
	rxe = context->internal[0];

	if (flags & FI_DISCARD) {
		ret = rxr_msg_discard_trecv(ep, rxe, msg, flags);
		/* rxe for peer srx does not allocate unexp_pkt */
		if (!(rxe->rxr_flags & EFA_RDM_RXE_FOR_PEER_SRX))
			rxr_pkt_entry_release_rx(ep, rxe->unexp_pkt);
		efa_rdm_rxe_release(rxe);
		goto out;
	}

	/*
	 * Handle unexp match entry even for discard entry as we are sinking
	 * messages for that case
	 */
	memcpy(rxe->iov, msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	rxe->iov_count = msg->iov_count;

	ret = rxr_msg_handle_unexp_match(ep, rxe, msg->tag,
					 msg->ignore, msg->context,
					 msg->addr, ofi_op_tagged, flags);

out:
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);
	return ret;
}

static
ssize_t rxr_msg_peek_trecv(struct fid_ep *ep_fid,
			   const struct fi_msg_tagged *msg,
			   uint64_t flags)
{
	ssize_t ret = 0;
	struct rxr_ep *ep;
	struct efa_rdm_ope *rxe;
	struct fi_context *context;
	size_t data_len;
	int64_t tag;
	bool claim;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	ofi_mutex_lock(&ep->base_ep.util_ep.lock);

	rxr_ep_progress_internal(ep);

	claim = (flags & (FI_CLAIM | FI_DISCARD));
	rxe = rxr_msg_find_unexp_rxe(ep, msg->addr, msg->tag, msg->ignore, ofi_op_tagged,
					       claim);
	if (!rxe) {
		EFA_DBG(FI_LOG_EP_CTRL,
		       "Message not found addr: %" PRIu64
		       " tag: %lx ignore %lx\n", msg->addr, msg->tag,
		       msg->ignore);
		ret = ofi_cq_write_error_peek(ep->base_ep.util_ep.rx_cq, msg->tag,
					      msg->context);
		goto out;
	}

	context = (struct fi_context *)msg->context;
	if (flags & FI_CLAIM) {
		context->internal[0] = rxe;
	} else if (flags & FI_DISCARD) {
		ret = rxr_msg_discard_trecv(ep, rxe, msg, flags);
		/* rxe for peer srx does not allocate unexp_pkt */
		if (!(rxe->rxr_flags & EFA_RDM_RXE_FOR_PEER_SRX))
			rxr_pkt_entry_release_rx(ep, rxe->unexp_pkt);
		efa_rdm_rxe_release(rxe);
		goto out;
	}

	data_len = rxe->total_len;
	tag = rxe->tag;

	if (ep->base_ep.util_ep.caps & FI_SOURCE)
		ret = ofi_cq_write_src(ep->base_ep.util_ep.rx_cq, context,
				       FI_TAGGED | FI_RECV,
				       data_len, NULL,
				       rxe->cq_entry.data, tag,
				       rxe->addr);
	else
		ret = ofi_cq_write(ep->base_ep.util_ep.rx_cq, context,
				   FI_TAGGED | FI_RECV,
				   data_len, NULL,
				   rxe->cq_entry.data, tag);
out:
	ofi_mutex_unlock(&ep->base_ep.util_ep.lock);
	return ret;
}

/**
 *   Non-tagged receive ops
 */
static
ssize_t rxr_msg_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			uint64_t flags)
{
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	/*
	 * For rxr_msg_recvmsg (trecvmsg), it should pass application
	 * flags |= util_ep.rx_msg_flags, which will have NO FI_COMPLETION
	 * when application binds rx cq with FI_SELECTIVE_COMPLETION,
	 * and does not have FI_COMPLETION in the flags of fi_recvmsg.
	 */
	return rxr_msg_generic_recv(ep_fid, msg, 0, 0, ofi_op_msg, flags | ep->base_ep.util_ep.rx_msg_flags);
}

static
ssize_t rxr_msg_recv(struct fid_ep *ep_fid, void *buf, size_t len,
		     void *desc, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, &desc, 1, src_addr, context, 0);
	return rxr_msg_recvmsg(ep_fid, &msg, rxr_rx_flags(ep));
}

static
ssize_t rxr_msg_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t src_addr,
		      void *context)
{
	struct fi_msg msg = {0};
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	rxr_msg_construct(&msg, iov, desc, count, src_addr, context, 0);
	return rxr_msg_recvmsg(ep_fid, &msg, rxr_rx_flags(ep));
}

/**
 *   Tagged receive ops functions
 */
static
ssize_t rxr_msg_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		      fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		      void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rxr_msg_construct(&msg, &iov, &desc, 1, src_addr, context, 0);
	return rxr_msg_generic_recv(ep_fid, &msg, tag, ignore, ofi_op_tagged, rxr_rx_flags(ep));
}

static
ssize_t rxr_msg_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t src_addr,
		       uint64_t tag, uint64_t ignore, void *context)
{
	struct fi_msg msg = {0};
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	rxr_msg_construct(&msg, iov, desc, count, src_addr, context, 0);
	return rxr_msg_generic_recv(ep_fid, &msg, tag, ignore, ofi_op_tagged, rxr_rx_flags(ep));
}

static
ssize_t rxr_msg_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *tmsg,
			 uint64_t flags)
{
	ssize_t ret;
	struct fi_msg msg = {0};
	struct rxr_ep *ep;

	ep = container_of(ep_fid, struct rxr_ep, base_ep.util_ep.ep_fid.fid);

	/*
	 * For rxr_msg_recvmsg (trecvmsg), it should pass application
	 * flags |= util_ep.rx_msg_flags, which will have NO FI_COMPLETION
	 * when application binds rx cq with FI_SELECTIVE_COMPLETION,
	 * and does not have FI_COMPLETION in the flags of fi_recvmsg.
	 */
	if (flags & FI_PEEK) {
		ret = rxr_msg_peek_trecv(ep_fid, tmsg, flags | ep->base_ep.util_ep.rx_msg_flags);
		goto out;
	} else if (flags & FI_CLAIM) {
		ret = rxr_msg_claim_trecv(ep_fid, tmsg, flags | ep->base_ep.util_ep.rx_msg_flags);
		goto out;
	}

	rxr_msg_construct(&msg, tmsg->msg_iov, tmsg->desc, tmsg->iov_count, tmsg->addr, tmsg->context, tmsg->data);
	ret = rxr_msg_generic_recv(ep_fid, &msg, tmsg->tag, tmsg->ignore,
				   ofi_op_tagged, flags | ep->base_ep.util_ep.rx_msg_flags);

out:
	return ret;
}

/**
 * Ops structures used by rxr_endpoint()
 */
struct fi_ops_msg rxr_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.send = rxr_msg_send,
	.sendv = rxr_msg_sendv,
	.sendmsg = rxr_msg_sendmsg,
	.senddata = rxr_msg_senddata,
	.inject = rxr_msg_inject,
	.injectdata = rxr_msg_injectdata,
	.recv = rxr_msg_recv,
	.recvv = rxr_msg_recvv,
	.recvmsg = rxr_msg_recvmsg,
};

struct fi_ops_tagged rxr_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.send = rxr_msg_tsend,
	.sendv = rxr_msg_tsendv,
	.sendmsg = rxr_msg_tsendmsg,
	.senddata = rxr_msg_tsenddata,
	.inject = rxr_msg_tinject,
	.injectdata = rxr_msg_tinjectdata,
	.recv = rxr_msg_trecv,
	.recvv = rxr_msg_trecvv,
	.recvmsg = rxr_msg_trecvmsg,
};
