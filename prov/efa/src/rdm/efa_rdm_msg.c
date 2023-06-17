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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "ofi.h"
#include <ofi_util.h>
#include <ofi_iov.h>

#include "efa.h"

#include "efa_rdm_msg.h"
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
 * @param [in]		efa_rdm_ep		endpoint
 * @param [in]		txe	contains information of the send operation
 * @param [in]		use_p2p		whether p2p can be used
 * @return		the RTM packet type of the two-sided protocol. Four
 *                      types of protocol can be used: eager, medium, longcts, longread.
 *                      Each protocol has tagged/non-tagged version. Some protocols has a DC version.
 * @related		efa_rdm_ep
 */
int efa_rdm_msg_select_rtm(struct efa_rdm_ep *efa_rdm_ep, struct efa_rdm_ope *txe, int use_p2p)
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

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, txe->addr);
	assert(peer);

	iface = txe->desc[0] ? ((struct efa_mr*) txe->desc[0])->peer.iface : FI_HMEM_SYSTEM;
	hmem_info = efa_rdm_ep_domain(efa_rdm_ep)->hmem_info;

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

	eager_rtm_max_data_size = efa_rdm_txe_max_req_data_capacity(efa_rdm_ep, txe, eager_rtm);

	readbase_rtm = rxr_pkt_type_readbase_rtm(peer, txe->op, txe->fi_flags, &hmem_info[iface]);

	if (txe->total_len >= hmem_info[iface].min_read_msg_size &&
		efa_rdm_ep_support_rdma_read(efa_rdm_ep) &&
		(txe->desc[0] || efa_is_cache_available(efa_rdm_ep_domain(efa_rdm_ep))))
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
ssize_t efa_rdm_msg_post_rtm(struct efa_rdm_ep *ep, struct efa_rdm_ope *txe, int use_p2p)
{
	ssize_t err;
	int rtm_type;
	struct efa_rdm_peer *peer;

	peer = efa_rdm_ep_get_peer(ep, txe->addr);
	assert(peer);

	rtm_type = efa_rdm_msg_select_rtm(ep, txe, use_p2p);
	assert(rtm_type >= RXR_REQ_PKT_BEGIN);

	if (rtm_type < RXR_EXTRA_REQ_PKT_BEGIN) {
		/* rtm requires only baseline feature, which peer should always support. */
		return rxr_pkt_post(ep, txe, rtm_type);
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

	return rxr_pkt_post(ep, txe, rtm_type);
}

ssize_t efa_rdm_msg_generic_send(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t tag, uint32_t op, uint64_t flags)
{
	struct efa_rdm_ep *efa_rdm_ep;
	ssize_t err, ret, use_p2p;
	struct efa_rdm_ope *txe;
	struct efa_rdm_peer *peer;
	struct util_srx_ctx *srx_ctx;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);

	assert(msg->iov_count <= efa_rdm_ep->tx_iov_limit);

	efa_perfset_start(efa_rdm_ep, perf_efa_tx);
	ofi_genlock_lock(srx_ctx->lock);

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, msg->addr);
	assert(peer);

	if (peer->flags & EFA_RDM_PEER_IN_BACKOFF) {
		err = -FI_EAGAIN;
		goto out;
	}

	txe = efa_rdm_ep_alloc_txe(efa_rdm_ep, msg, op, tag, flags);
	if (OFI_UNLIKELY(!txe)) {
		err = -FI_EAGAIN;
		efa_rdm_ep_progress_internal(efa_rdm_ep);
		goto out;
	}

	ret = efa_rdm_ep_use_p2p(efa_rdm_ep, txe->desc[0]);
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


	err = efa_rdm_msg_post_rtm(efa_rdm_ep, txe, use_p2p);
	if (OFI_UNLIKELY(err)) {
		efa_rdm_ep_progress_internal(efa_rdm_ep);
		efa_rdm_txe_release(txe);
		peer->next_msg_id--;
	}

out:
	ofi_genlock_unlock(srx_ctx->lock);
	efa_perfset_end(efa_rdm_ep, perf_efa_tx);
	return err;
}

/**
 *   Non-tagged send ops function
 */
static
ssize_t efa_rdm_msg_sendmsg(struct fid_ep *ep, const struct fi_msg *msg,
			uint64_t flags)
{
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *efa_rdm_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};
	struct fi_msg *shm_msg;
	void **efa_desc = NULL;
	fi_addr_t efa_addr;
	int ret;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, msg->addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		shm_msg = (struct fi_msg *)msg;
		if (msg->desc) {
			efa_desc = msg->desc;
			efa_rdm_get_desc_for_shm(msg->iov_count, msg->desc, shm_desc);
			shm_msg->desc = shm_desc;
		}
		efa_addr = msg->addr;
		shm_msg->addr = peer->shm_fiaddr;
		ret = fi_sendmsg(efa_rdm_ep->shm_ep, shm_msg, flags);
		/* Recover the application msg */
		if (efa_desc)
			shm_msg->desc = efa_desc;
		shm_msg->addr = efa_addr;
		return ret;
	}

	return efa_rdm_msg_generic_send(ep, msg, 0, ofi_op_msg, flags);
}

static
ssize_t efa_rdm_msg_sendv(struct fid_ep *ep, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t dest_addr,
		      void *context)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg msg = {0};
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		efa_rdm_get_desc_for_shm(count, desc, shm_desc);
		return fi_sendv(efa_rdm_ep->shm_ep, iov, shm_desc, count, peer->shm_fiaddr, context);
	}

	efa_rdm_msg_construct(&msg, iov, desc, count, dest_addr, context, 0);
	return efa_rdm_msg_sendmsg(ep, &msg, rxr_tx_flags(efa_rdm_ep));
}

static
ssize_t efa_rdm_msg_send(struct fid_ep *ep, const void *buf, size_t len,
		     void *desc, fi_addr_t dest_addr, void *context)
{
	struct iovec iov;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *efa_rdm_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		if (desc)
			efa_rdm_get_desc_for_shm(1, &desc, shm_desc);
		return fi_send(efa_rdm_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, peer->shm_fiaddr, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;
	return efa_rdm_msg_sendv(ep, &iov, &desc, 1, dest_addr, context);
}

static
ssize_t efa_rdm_msg_senddata(struct fid_ep *ep, const void *buf, size_t len,
			 void *desc, uint64_t data, fi_addr_t dest_addr,
			 void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		if (desc)
			efa_rdm_get_desc_for_shm(1, &desc, shm_desc);
		return fi_senddata(efa_rdm_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, data, peer->shm_fiaddr, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, &desc, 1, dest_addr, context, data);
	return efa_rdm_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(efa_rdm_ep) | FI_REMOTE_CQ_DATA);
}

static
ssize_t efa_rdm_msg_inject(struct fid_ep *ep, const void *buf, size_t len,
		       fi_addr_t dest_addr)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	if (len > efa_rdm_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		return fi_inject(efa_rdm_ep->shm_ep, buf, len, peer->shm_fiaddr);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, 0);

	return efa_rdm_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(efa_rdm_ep) | EFA_RDM_TXE_NO_COMPLETION | FI_INJECT);
}

static
ssize_t efa_rdm_msg_injectdata(struct fid_ep *ep, const void *buf,
			   size_t len, uint64_t data,
			   fi_addr_t dest_addr)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg msg;
	struct iovec iov;
	struct efa_rdm_peer *peer;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	if (len > efa_rdm_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		return fi_injectdata(efa_rdm_ep->shm_ep, buf, len, data, peer->shm_fiaddr);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, data);

	return efa_rdm_msg_generic_send(ep, &msg, 0, ofi_op_msg,
				    rxr_tx_flags(efa_rdm_ep) | EFA_RDM_TXE_NO_COMPLETION |
				    FI_REMOTE_CQ_DATA | FI_INJECT);
}

/**
 *   Tagged send op functions
 */
static
ssize_t efa_rdm_msg_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *tmsg,
			 uint64_t flags)
{
	struct fi_msg msg = {0};
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *efa_rdm_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};
	struct fi_msg_tagged *shm_tmsg;
	void **efa_desc = NULL;
	fi_addr_t efa_addr;
	int ret;

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, tmsg->addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		shm_tmsg = (struct fi_msg_tagged *)tmsg;
		if (tmsg->desc) {
			efa_desc = tmsg->desc;
			efa_rdm_get_desc_for_shm(tmsg->iov_count, tmsg->desc, shm_desc);
			shm_tmsg->desc = shm_desc;
		}
		efa_addr = tmsg->addr;
		shm_tmsg->addr = peer->shm_fiaddr;
		ret = fi_tsendmsg(efa_rdm_ep->shm_ep, shm_tmsg, flags);
		/* Recover the application msg */
		if (efa_desc)
			shm_tmsg->desc = efa_desc;
		shm_tmsg->addr = efa_addr;
		return ret;
	}

	efa_rdm_msg_construct(&msg, tmsg->msg_iov, tmsg->desc, tmsg->iov_count, tmsg->addr, tmsg->context, tmsg->data);
	return efa_rdm_msg_generic_send(ep_fid, &msg, tmsg->tag, ofi_op_tagged, flags);
}

static
ssize_t efa_rdm_msg_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t dest_addr,
		       uint64_t tag, void *context)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg_tagged msg = {0};
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		if (desc)
			efa_rdm_get_desc_for_shm(count, desc, shm_desc);
		return fi_tsendv(efa_rdm_ep->shm_ep, iov, desc? shm_desc : NULL, count, peer->shm_fiaddr, tag, context);
	}

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;
	msg.tag = tag;

	return efa_rdm_msg_tsendmsg(ep_fid, &msg, rxr_tx_flags(efa_rdm_ep));
}

static
ssize_t efa_rdm_msg_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, fi_addr_t dest_addr, uint64_t tag,
		      void *context)
{
	struct iovec msg_iov;
	struct efa_rdm_peer *peer;
	struct efa_rdm_ep *efa_rdm_ep;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		if (desc)
			efa_rdm_get_desc_for_shm(1, &desc, shm_desc);
		return fi_tsend(efa_rdm_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, peer->shm_fiaddr, tag, context);
	}

	msg_iov.iov_base = (void *)buf;
	msg_iov.iov_len = len;
	return efa_rdm_msg_tsendv(ep_fid, &msg_iov, &desc, 1, dest_addr, tag,
			      context);
}

static
ssize_t efa_rdm_msg_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			  void *desc, uint64_t data, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_peer *peer;
	void *shm_desc[RXR_IOV_LIMIT] = {NULL};

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		if (desc)
			efa_rdm_get_desc_for_shm(1, &desc, shm_desc);
		return fi_tsenddata(efa_rdm_ep->shm_ep, buf, len, desc? shm_desc[0] : NULL, data, peer->shm_fiaddr, tag, context);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, &desc, 1, dest_addr, context, data);
	return efa_rdm_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(efa_rdm_ep) | FI_REMOTE_CQ_DATA);
}

static
ssize_t efa_rdm_msg_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			fi_addr_t dest_addr, uint64_t tag)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	if (len > efa_rdm_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		return fi_tinject(efa_rdm_ep->shm_ep, buf, len, peer->shm_fiaddr, tag);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, 0);

	return efa_rdm_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(efa_rdm_ep) | EFA_RDM_TXE_NO_COMPLETION | FI_INJECT);
}

static
ssize_t efa_rdm_msg_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			    uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct efa_rdm_ep *efa_rdm_ep;
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_peer *peer;

	efa_rdm_ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);
	if (len > efa_rdm_ep->inject_size) {
		EFA_WARN(FI_LOG_CQ, "invalid message size %ld for inject.\n", len);
		return -FI_EINVAL;
	}

	peer = efa_rdm_ep_get_peer(efa_rdm_ep, dest_addr);
	assert(peer);
	if (peer->is_local && efa_rdm_ep->use_shm_for_tx) {
		return fi_tinjectdata(efa_rdm_ep->shm_ep, buf, len, data, peer->shm_fiaddr, tag);
	}

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, NULL, 1, dest_addr, NULL, data);

	return efa_rdm_msg_generic_send(ep_fid, &msg, tag, ofi_op_tagged,
				    rxr_tx_flags(efa_rdm_ep) | EFA_RDM_TXE_NO_COMPLETION |
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
struct efa_rdm_ope *efa_rdm_msg_alloc_rxe(struct efa_rdm_ep *ep,
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

	rxe = efa_rdm_ep_alloc_rxe(ep, addr, op);
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

	return rxe;
}

struct efa_rdm_ope *efa_rdm_msg_alloc_unexp_rxe_for_rtm(struct efa_rdm_ep *ep,
							struct efa_rdm_pke **pkt_entry_ptr,
							uint32_t op)
{
	struct efa_rdm_ope *rxe;
	struct efa_rdm_pke *unexp_pkt_entry;

	assert(op == ofi_op_msg || ofi_op_tagged);

	unexp_pkt_entry = rxr_pkt_get_unexp(ep, pkt_entry_ptr);
	if (OFI_UNLIKELY(!unexp_pkt_entry)) {
		EFA_WARN(FI_LOG_CQ, "packet entries exhausted.\n");
		return NULL;
	}

	rxe = efa_rdm_ep_alloc_rxe(ep, unexp_pkt_entry->addr, op);
	if (OFI_UNLIKELY(!rxe))
		return NULL;

	if (op == ofi_op_tagged)
		rxe->tag = rxr_pkt_rtm_tag(unexp_pkt_entry);
	rxe->rxr_flags = 0;
	rxe->state = EFA_RDM_RXE_UNEXP;
	rxe->unexp_pkt = unexp_pkt_entry;
	rxr_pkt_rtm_update_rxe(unexp_pkt_entry, rxe);
	return rxe;
}

/*
 *     create a rxe and verify in unexpected message list
 *     else add to posted recv list
 */
static
ssize_t efa_rdm_msg_generic_recv(struct fid_ep *ep, const struct fi_msg *msg,
			     uint64_t tag, uint64_t ignore, uint32_t op,
			     uint64_t flags)
{
	ssize_t ret = 0;
	struct efa_rdm_ep *efa_rdm_ep;
	struct efa_rdm_ope *rxe;
	struct util_srx_ctx *srx_ctx;

	efa_rdm_ep = container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	srx_ctx = efa_rdm_ep_get_peer_srx_ctx(efa_rdm_ep);

	assert(msg->iov_count <= efa_rdm_ep->rx_iov_limit);

	efa_perfset_start(efa_rdm_ep, perf_efa_recv);

	EFA_DBG(FI_LOG_EP_DATA,
	       "%s: iov_len: %lu tag: %lx ignore: %lx op: %x flags: %lx\n",
	       __func__, ofi_total_iov_len(msg->msg_iov, msg->iov_count), tag, ignore,
	       op, flags);

	rxr_tracepoint(recv_begin_msg_context, (size_t) msg->context, (size_t) msg->addr);

	if (efa_rdm_ep->use_zcpy_rx) {
		ofi_genlock_lock(srx_ctx->lock);
		rxe = efa_rdm_msg_alloc_rxe(efa_rdm_ep, msg, op, flags, tag, ignore);
		if (OFI_UNLIKELY(!rxe)) {
			ret = -FI_EAGAIN;
			efa_rdm_ep_progress_internal(efa_rdm_ep);
			ofi_genlock_unlock(srx_ctx->lock);
			goto out;
		}

		ret = efa_rdm_ep_post_user_recv_buf(efa_rdm_ep, rxe, flags);
		if (ret == -FI_EAGAIN)
			efa_rdm_ep_progress_internal(efa_rdm_ep);
		ofi_genlock_unlock(srx_ctx->lock);
	} else if (op == ofi_op_tagged) {
		ret = util_srx_generic_trecv(efa_rdm_ep->peer_srx_ep, msg->msg_iov, msg->desc,
					     msg->iov_count, msg->addr, msg->context,
					     tag, ignore, flags);
	} else {
		ret = util_srx_generic_recv(efa_rdm_ep->peer_srx_ep, msg->msg_iov, msg->desc,
				            msg->iov_count, msg->addr, msg->context, flags);
	}

out:
	efa_perfset_end(efa_rdm_ep, perf_efa_recv);
	return ret;
}

/**
 *   Non-tagged receive ops
 */
static
ssize_t efa_rdm_msg_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			uint64_t flags)
{
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	/*
	 * For efa_rdm_msg_recvmsg (trecvmsg), it should pass application
	 * flags |= util_ep.rx_msg_flags, which will have NO FI_COMPLETION
	 * when application binds rx cq with FI_SELECTIVE_COMPLETION,
	 * and does not have FI_COMPLETION in the flags of fi_recvmsg.
	 */
	return efa_rdm_msg_generic_recv(ep_fid, msg, 0, 0, ofi_op_msg, flags | ep->base_ep.util_ep.rx_msg_flags);
}

static
ssize_t efa_rdm_msg_recv(struct fid_ep *ep_fid, void *buf, size_t len,
		     void *desc, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, &desc, 1, src_addr, context, 0);
	return efa_rdm_msg_recvmsg(ep_fid, &msg, rxr_rx_flags(ep));
}

static
ssize_t efa_rdm_msg_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
		      void **desc, size_t count, fi_addr_t src_addr,
		      void *context)
{
	struct fi_msg msg = {0};
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	efa_rdm_msg_construct(&msg, iov, desc, count, src_addr, context, 0);
	return efa_rdm_msg_recvmsg(ep_fid, &msg, rxr_rx_flags(ep));
}

/**
 *   Tagged receive ops functions
 */
static
ssize_t efa_rdm_msg_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		      fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
		      void *context)
{
	struct fi_msg msg = {0};
	struct iovec iov;
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	efa_rdm_msg_construct(&msg, &iov, &desc, 1, src_addr, context, 0);
	return efa_rdm_msg_generic_recv(ep_fid, &msg, tag, ignore, ofi_op_tagged, rxr_rx_flags(ep));
}

static
ssize_t efa_rdm_msg_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
		       void **desc, size_t count, fi_addr_t src_addr,
		       uint64_t tag, uint64_t ignore, void *context)
{
	struct fi_msg msg = {0};
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	efa_rdm_msg_construct(&msg, iov, desc, count, src_addr, context, 0);
	return efa_rdm_msg_generic_recv(ep_fid, &msg, tag, ignore, ofi_op_tagged, rxr_rx_flags(ep));
}

static
ssize_t efa_rdm_msg_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *tmsg,
			 uint64_t flags)
{
	struct fi_msg msg = {0};
	struct efa_rdm_ep *ep;

	ep = container_of(ep_fid, struct efa_rdm_ep, base_ep.util_ep.ep_fid.fid);

	/*
	 * For efa_rdm_msg_recvmsg (trecvmsg), it should pass application
	 * flags |= util_ep.rx_msg_flags, which will have NO FI_COMPLETION
	 * when application binds rx cq with FI_SELECTIVE_COMPLETION,
	 * and does not have FI_COMPLETION in the flags of fi_recvmsg.
	 */

	efa_rdm_msg_construct(&msg, tmsg->msg_iov, tmsg->desc, tmsg->iov_count, tmsg->addr, tmsg->context, tmsg->data);
	return efa_rdm_msg_generic_recv(ep_fid, &msg, tmsg->tag, tmsg->ignore,
				   ofi_op_tagged, flags | ep->base_ep.util_ep.rx_msg_flags);
}

/**
 * Ops structures used by rxr_endpoint()
 */
struct fi_ops_msg efa_rdm_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.send = efa_rdm_msg_send,
	.sendv = efa_rdm_msg_sendv,
	.sendmsg = efa_rdm_msg_sendmsg,
	.senddata = efa_rdm_msg_senddata,
	.inject = efa_rdm_msg_inject,
	.injectdata = efa_rdm_msg_injectdata,
	.recv = efa_rdm_msg_recv,
	.recvv = efa_rdm_msg_recvv,
	.recvmsg = efa_rdm_msg_recvmsg,
};

struct fi_ops_tagged efa_rdm_msg_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.send = efa_rdm_msg_tsend,
	.sendv = efa_rdm_msg_tsendv,
	.sendmsg = efa_rdm_msg_tsendmsg,
	.senddata = efa_rdm_msg_tsenddata,
	.inject = efa_rdm_msg_tinject,
	.injectdata = efa_rdm_msg_tinjectdata,
	.recv = efa_rdm_msg_trecv,
	.recvv = efa_rdm_msg_trecvv,
	.recvmsg = efa_rdm_msg_trecvmsg,
};
