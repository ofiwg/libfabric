/*
 * Copyright (c) Intel Corporation. All rights reserved
 * (C) Copyright 2021 Amazon.com, Inc. or its affiliates.
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

#include "smr.h"

static ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			   uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return util_srx_generic_recv(&ep->srx->ep_fid, msg->msg_iov, msg->desc,
				     msg->iov_count, msg->addr, msg->context,
				     flags | ep->util_ep.rx_msg_flags);
}

static ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t src_addr,
			 void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return util_srx_generic_recv(&ep->srx->ep_fid, iov, desc, count,
				     src_addr, context, smr_ep_rx_flags(ep));
}

static ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len,
			void *desc, fi_addr_t src_addr, void *context)
{
	struct iovec iov;
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return util_srx_generic_recv(&ep->srx->ep_fid, &iov, &desc, 1, src_addr,
				     context, smr_ep_rx_flags(ep));
}

static ssize_t smr_generic_sendmsg(struct smr_ep *ep, const struct iovec *iov,
				   void **desc, size_t iov_count,
				   fi_addr_t addr, uint64_t tag, uint64_t data,
				   void *context, uint32_t op,
				   uint64_t op_flags)
{
	struct smr_region *peer_smr;
	int64_t tx_id, rx_id, pos;
	ssize_t ret = -FI_EAGAIN;
	size_t total_len;
	int proto;
	struct smr_cmd_entry *ce;
	struct smr_cmd *cmd;

	assert(iov_count <= SMR_IOV_LIMIT);

	tx_id = smr_verify_peer(ep, addr);
	if (tx_id < 0)
		return -FI_EAGAIN;

	rx_id = smr_peer_data(ep->region)[tx_id].id;
	peer_smr = smr_peer_region(ep, tx_id);

	ofi_genlock_lock(&ep->util_ep.lock);
	if (smr_peer_data(ep->region)[tx_id].sar_status)
		goto unlock;

	ret = smr_cmd_queue_next(smr_cmd_queue(peer_smr), &ce, &pos);
	if (ret == -FI_ENOENT) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	total_len = ofi_total_iov_len(iov, iov_count);
	assert(!(op_flags & FI_INJECT) || total_len <= SMR_INJECT_SIZE);

	proto = smr_select_proto(desc, iov_count, smr_vma_enabled(ep, peer_smr),
	                         smr_ipc_valid(ep, peer_smr, tx_id, rx_id), op,
				 total_len, op_flags);

	if (proto != smr_proto_inline) {
		if (smr_freestack_isempty(smr_cmd_stack(ep->region))) {
			smr_cmd_queue_discard(ce, pos);
			ret = -FI_EAGAIN;
			goto unlock;
		}

		cmd = smr_freestack_pop(smr_cmd_stack(ep->region));
		assert(cmd);
		ce->ptr = smr_local_to_peer(ep, peer_smr, tx_id, rx_id,
					    (uintptr_t) cmd);
		/* Clear stale fast-inject marker from prior slot usage.
		 * Receiver's fast RX dispatch requires the marker, so zero means
		 * "not a fast inject" and the generic path handles it. */
		ce->cmd.hdr.tx_ctx = 0;
	} else {
		cmd = &ce->cmd;
		cmd->hdr.tx_ctx = 0;
	}

	ret = smr_send_ops[proto](ep, peer_smr, tx_id, rx_id, op, tag, data,
				  op_flags, (struct ofi_mr **) desc, iov,
				  iov_count, total_len, context, cmd);
	if (ret) {
		smr_cmd_queue_discard(ce, pos);
		if (proto != smr_proto_inline)
			smr_freestack_push(smr_cmd_stack(ep->region), cmd);
		goto unlock;
	}
	smr_cmd_queue_commit(ce, pos);

	if (proto != smr_proto_inline)
		goto unlock;

	ret = smr_complete_tx(ep, context, op, op_flags);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
		goto unlock;
	}

unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}


/* Fast send path for simple case: 1-IOV, no desc, peer supports V2.
 * Uses the same fast inject protocol as fast_inject_v24 but also writes
 * a CQ entry for the user context. Works for sizes SMR_MSG_DATA_LEN+1
 * to SMR_INJECT_SIZE. For ≤ SMR_MSG_DATA_LEN, use inline. */
static inline ssize_t smr_fast_tsend(struct smr_ep *ep, const void *buf,
				     size_t len, fi_addr_t dest_addr,
				     uint64_t tag, void *context, uint32_t op,
				     uint64_t op_flags)
{
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	int64_t tx_id, rx_id, pos;
	ssize_t ret;
	struct smr_cmd_entry *ce;
	struct smr_cmd *cmd, *pcmd;
	int16_t idx;

	tx_id = smr_verify_peer(ep, dest_addr);
	if (tx_id < 0)
		return -FI_EAGAIN;
	rx_id = smr_peer_data(ep->region)[tx_id].id;
	peer_smr = smr_peer_region(ep, tx_id);
	if (OFI_UNLIKELY(!(peer_smr->flags & SMR_FLAG_FAST_INJECT_V2)))
		return -FI_EOPNOTSUPP;
	ofi_genlock_lock(&ep->util_ep.lock);
	if (OFI_UNLIKELY(smr_peer_data(ep->region)[tx_id].sar_status)) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	ret = smr_cmd_queue_next(smr_cmd_queue(peer_smr), &ce, &pos);
	if (ret == -FI_ENOENT) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}

	if (len <= SMR_MSG_DATA_LEN) {
		/* Inline fast path - no cmd_stack needed */
		cmd = &ce->cmd;
		cmd->hdr.tx_ctx = 0;
		cmd->hdr.op = op;
		cmd->hdr.status = 0;
		cmd->hdr.op_flags = 0;
		cmd->hdr.tag = tag;
		cmd->hdr.tx_id = tx_id;
		cmd->hdr.rx_id = rx_id;
		cmd->hdr.cq_data = 0;
		cmd->hdr.rx_ctx = 0;
		cmd->hdr.proto = smr_proto_inline;
		cmd->hdr.size = len;
		memcpy(cmd->data.msg, buf, len);
		smr_cmd_queue_commit(ce, pos);
		ret = smr_complete_tx(ep, context, op, op_flags);
		ofi_genlock_unlock(&ep->util_ep.lock);
		return ret;
	}

	/* Inject fast path */
	if (smr_freestack_isempty(smr_cmd_stack(ep->region))) {
		smr_cmd_queue_discard(ce, pos);
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	pcmd = smr_freestack_pop(smr_cmd_stack(ep->region));
	idx = smr_freestack_get_index(smr_cmd_stack(ep->region), (char *) pcmd);
	tx_buf = &smr_inject_pool(ep->region)[idx];
	memcpy(tx_buf->data, buf, len);

	cmd = &ce->cmd;
	cmd->hdr.entry = idx;
	cmd->hdr.tx_ctx = SMR_FAST_INJECT_TX_CTX;
	cmd->hdr.op = op;
	cmd->hdr.status = 0;
	cmd->hdr.op_flags = 0;
	cmd->hdr.tag = tag;
	cmd->hdr.tx_id = tx_id;
	cmd->hdr.rx_id = rx_id;
	cmd->hdr.cq_data = 0;
	cmd->hdr.rx_ctx = 0;
	cmd->hdr.proto = smr_proto_inject;
	cmd->hdr.size = len;
	pcmd->hdr.tx_ctx = SMR_FAST_INJECT_TX_CTX;
	pcmd->hdr.proto = smr_proto_inject;
	pcmd->hdr.rx_id = rx_id;
	ce->ptr = smr_local_to_peer(ep, peer_smr, tx_id, rx_id, (uintptr_t) pcmd);

	smr_cmd_queue_commit(ce, pos);
	ret = smr_complete_tx(ep, context, op, op_flags);
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t smr_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			void *desc, fi_addr_t dest_addr, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	/* Fast path for simple send: 1-IOV, SYSTEM memory, size<=inject */
	if (OFI_LIKELY(len <= SMR_INJECT_SIZE &&
		(!desc || ((struct ofi_mr *)desc)->iface == FI_HMEM_SYSTEM))) {
		ssize_t r = smr_fast_tsend(ep, buf, len, dest_addr, 0, context, ofi_op_msg, smr_ep_tx_flags(ep));
		if (OFI_LIKELY(r != -FI_EOPNOTSUPP))
			return r;
	}
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}

static ssize_t smr_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
			 void **desc, size_t count, fi_addr_t dest_addr,
			 void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, desc, count, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}

static ssize_t smr_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			   uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg,
				   flags | ep->util_ep.tx_msg_flags);
}

static ssize_t smr_generic_inject(struct fid_ep *ep_fid, const void *buf,
				  size_t len, fi_addr_t dest_addr, uint64_t tag,
				  uint64_t data, uint32_t op, uint64_t op_flags)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	int64_t tx_id, rx_id, pos;
	ssize_t ret = 0;
	struct iovec msg_iov;
	int proto;
	struct smr_cmd_entry *ce;
	struct smr_cmd *cmd;

	assert(len <= SMR_INJECT_SIZE);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	tx_id = smr_verify_peer(ep, dest_addr);
	if (tx_id < 0)
		return -FI_EAGAIN;

	rx_id = smr_peer_data(ep->region)[tx_id].id;
	peer_smr = smr_peer_region(ep, tx_id);

	ofi_genlock_lock(&ep->util_ep.lock);
	if (smr_peer_data(ep->region)[tx_id].sar_status) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	ret = smr_cmd_queue_next(smr_cmd_queue(peer_smr), &ce, &pos);
	if (ret == -FI_ENOENT) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	if (len <= SMR_MSG_DATA_LEN) {
		proto = smr_proto_inline;
		cmd = &ce->cmd;
		cmd->hdr.tx_ctx = 0;
	} else {
		proto = smr_proto_inject;
		if (smr_freestack_isempty(smr_cmd_stack(ep->region))) {
			smr_cmd_queue_discard(ce, pos);
			ret = -FI_EAGAIN;
			goto unlock;
		}

		cmd = smr_freestack_pop(smr_cmd_stack(ep->region));
		assert(cmd);
		ce->ptr = smr_local_to_peer(ep, peer_smr, tx_id, rx_id,
					    (uintptr_t) cmd);
		/* Clear stale fast-inject marker from prior slot usage.
		 * Receiver's fast RX dispatch requires the marker, so zero means
		 * "not a fast inject" and the generic path handles it. */
		ce->cmd.hdr.tx_ctx = 0;
	}

	ret = smr_send_ops[proto](ep, peer_smr, tx_id, rx_id, op, tag, data,
				  op_flags, NULL, &msg_iov, 1, len, NULL, cmd);
	if (ret) {
		if (proto != smr_proto_inline)
			smr_freestack_push(smr_cmd_stack(ep->region), cmd);
		smr_cmd_queue_discard(ce, pos);
		ret = -FI_EAGAIN;
		goto unlock;
	}
	smr_cmd_queue_commit(ce, pos);

	if (proto == smr_proto_inline)
		ofi_ep_peer_tx_cntr_inc(&ep->util_ep, op);

unlock:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t smr_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			  fi_addr_t dest_addr)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, 0, 0,
				  ofi_op_msg, 0);
}

static ssize_t smr_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			    void *desc, uint64_t data, fi_addr_t dest_addr,
			    void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, 0, data,
				   context, ofi_op_msg,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
}

static ssize_t smr_injectdata(struct fid_ep *ep_fid, const void *buf,
			      size_t len, uint64_t data, fi_addr_t dest_addr)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, 0, data,
				  ofi_op_msg, FI_REMOTE_CQ_DATA);
}

struct fi_ops_msg smr_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = smr_recv,
	.recvv = smr_recvv,
	.recvmsg = smr_recvmsg,
	.send = smr_send,
	.sendv = smr_sendv,
	.sendmsg = smr_sendmsg,
	.inject = smr_inject,
	.senddata = smr_senddata,
	.injectdata = smr_injectdata,
};

struct fi_ops_msg smr_no_recv_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = fi_no_msg_recv,
	.recvv = fi_no_msg_recvv,
	.recvmsg = fi_no_msg_recvmsg,
	.send = smr_send,
	.sendv = smr_sendv,
	.sendmsg = smr_sendmsg,
	.inject = smr_inject,
	.senddata = smr_senddata,
	.injectdata = smr_injectdata,
};

static ssize_t smr_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
			 void *desc, fi_addr_t src_addr, uint64_t tag,
			 uint64_t ignore, void *context)
{
	struct iovec iov;
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = buf;
	iov.iov_len = len;

	return util_srx_generic_trecv(&ep->srx->ep_fid, &iov, &desc, 1,
				     src_addr, context, tag, ignore,
				     smr_ep_rx_flags(ep));
}

static ssize_t smr_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t src_addr,
			  uint64_t tag, uint64_t ignore, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return util_srx_generic_trecv(&ep->srx->ep_fid, iov, desc, count,
				      src_addr, context, tag, ignore,
				      smr_ep_rx_flags(ep));
}

static ssize_t smr_trecvmsg(struct fid_ep *ep_fid,
			    const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return util_srx_generic_trecv(&ep->srx->ep_fid, msg->msg_iov, msg->desc,
				      msg->iov_count, msg->addr, msg->context,
				      msg->tag, msg->ignore,
				      flags | ep->util_ep.rx_msg_flags);
}

static ssize_t smr_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
			 void *desc, fi_addr_t dest_addr, uint64_t tag,
			 void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	/* Fast path for simple tsend: 1-IOV, SYSTEM memory, size<=inject */
	if (OFI_LIKELY(len <= SMR_INJECT_SIZE &&
		(!desc || ((struct ofi_mr *)desc)->iface == FI_HMEM_SYSTEM))) {
		ssize_t r = smr_fast_tsend(ep, buf, len, dest_addr, tag, context, ofi_op_tagged, smr_ep_tx_flags(ep));
		if (OFI_LIKELY(r != -FI_EOPNOTSUPP))
			return r;
	}
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, &desc, 1, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

static ssize_t smr_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
			  void **desc, size_t count, fi_addr_t dest_addr,
			  uint64_t tag, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, desc, count, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

static ssize_t smr_tsendmsg(struct fid_ep *ep_fid,
			    const struct fi_msg_tagged *msg, uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->desc, msg->iov_count,
				   msg->addr, msg->tag, msg->data, msg->context,
				   ofi_op_tagged,
				   flags | ep->util_ep.tx_msg_flags);
}


/* Fast path for inline tagged inject */
static inline ssize_t smr_fast_tinject(struct smr_ep *ep, const void *buf,
				       size_t len, fi_addr_t dest_addr,
				       uint64_t tag)
{
	struct smr_region *peer_smr;
	int64_t tx_id, rx_id, pos;
	ssize_t ret;
	struct smr_cmd_entry *ce;
	struct smr_cmd *cmd;

	tx_id = smr_verify_peer(ep, dest_addr);
	if (tx_id < 0)
		return -FI_EAGAIN;
	rx_id = smr_peer_data(ep->region)[tx_id].id;
	peer_smr = smr_peer_region(ep, tx_id);
	ofi_genlock_lock(&ep->util_ep.lock);
	if (OFI_UNLIKELY(smr_peer_data(ep->region)[tx_id].sar_status)) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	ret = smr_cmd_queue_next(smr_cmd_queue(peer_smr), &ce, &pos);
	if (ret == -FI_ENOENT) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	cmd = &ce->cmd;
	cmd->hdr.tx_ctx = 0;
	cmd->hdr.op = ofi_op_tagged;
	cmd->hdr.status = 0;
	cmd->hdr.op_flags = 0;
	cmd->hdr.tag = tag;
	cmd->hdr.tx_id = tx_id;
	cmd->hdr.rx_id = rx_id;
	cmd->hdr.cq_data = 0;
	cmd->hdr.rx_ctx = 0;
	cmd->hdr.proto = smr_proto_inline;
	cmd->hdr.size = len;
	memcpy(cmd->data.msg, buf, len);
	smr_cmd_queue_commit(ce, pos);
	ofi_ep_peer_tx_cntr_inc(&ep->util_ep, ofi_op_tagged);
	ofi_genlock_unlock(&ep->util_ep.lock);
	return FI_SUCCESS;
}

/* Fast inject send: counter-based inject buf, no cmd_stack, no return queue.
 * Stores inject buf index in cmd->hdr.entry for the receiver. */
static inline ssize_t smr_fast_inject_v24(struct smr_ep *ep, const void *buf,
					  size_t len, fi_addr_t dest_addr,
					  uint64_t tag)
{
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_cmd_hdr hdr;
	int64_t tx_id, rx_id, pos;
	ssize_t ret;
	struct smr_cmd_entry *ce;
	struct smr_cmd *pcmd;
	int16_t idx;

	tx_id = smr_verify_peer(ep, dest_addr);
	if (tx_id < 0)
		return -FI_EAGAIN;
	rx_id = smr_peer_data(ep->region)[tx_id].id;
	peer_smr = smr_peer_region(ep, tx_id);
	if (OFI_UNLIKELY(!(peer_smr->flags & SMR_FLAG_FAST_INJECT_V2)))
		return -FI_EOPNOTSUPP;
	ofi_genlock_lock(&ep->util_ep.lock);
	if (OFI_UNLIKELY(smr_peer_data(ep->region)[tx_id].sar_status)) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	ret = smr_cmd_queue_next(smr_cmd_queue(peer_smr), &ce, &pos);
	if (ret == -FI_ENOENT) {
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	if (smr_freestack_isempty(smr_cmd_stack(ep->region))) {
		smr_cmd_queue_discard(ce, pos);
		ofi_genlock_unlock(&ep->util_ep.lock);
		return -FI_EAGAIN;
	}
	pcmd = smr_freestack_pop(smr_cmd_stack(ep->region));
	idx = smr_freestack_get_index(smr_cmd_stack(ep->region), (char *) pcmd);
	tx_buf = &smr_inject_pool(ep->region)[idx];
	memcpy(tx_buf->data, buf, len);

	/* Build header on stack */
	hdr.entry = idx;
	hdr.tx_ctx = SMR_FAST_INJECT_TX_CTX;
	hdr.rx_ctx = 0;
	hdr.size = len;
	hdr.status = 0;
	hdr.cq_data = 0;
	hdr.tag = tag;
	hdr.rx_id = rx_id;
	hdr.tx_id = tx_id;
	hdr.op = ofi_op_tagged;
	hdr.proto = smr_proto_inject;
	hdr.op_flags = 0;
	hdr.resv[0] = 0;

	/* Full header to pcmd (LOCAL). Receiver reads data from this */
	pcmd->hdr = hdr;

	/* Minimum to ce->cmd (CROSS-PROCESS) - only what receiver's
	 * fast RX check needs: proto, op, rx_ctx must match for dispatch.
	 * All other fields read from _hdr = &ce->cmd in fast path. */
	ce->cmd.hdr.proto = smr_proto_inject;
	ce->cmd.hdr.op = ofi_op_tagged;
	ce->cmd.hdr.rx_ctx = 0;
	ce->cmd.hdr.tx_ctx = SMR_FAST_INJECT_TX_CTX;
	/* Also need: entry (for inject_buf lookup), size, rx_id, tag, op_flags, cq_data
	 * for the fast RX path. These are all in _hdr = ce->cmd */
	ce->cmd.hdr.entry = idx;
	ce->cmd.hdr.size = len;
	ce->cmd.hdr.rx_id = rx_id;
	ce->cmd.hdr.tag = tag;
	ce->cmd.hdr.op_flags = 0;
	ce->cmd.hdr.cq_data = 0;

	ce->ptr = smr_local_to_peer(ep, peer_smr, tx_id, rx_id, (uintptr_t) pcmd);
	smr_cmd_queue_commit(ce, pos);
	ofi_ep_peer_tx_cntr_inc(&ep->util_ep, ofi_op_tagged);
	ofi_genlock_unlock(&ep->util_ep.lock);
	return FI_SUCCESS;
}

static ssize_t smr_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			   fi_addr_t dest_addr, uint64_t tag)
{
	struct smr_ep *ep;
	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	if (OFI_LIKELY(len <= SMR_MSG_DATA_LEN))
		return smr_fast_tinject(ep, buf, len, dest_addr, tag);
	if (OFI_LIKELY(len <= SMR_INJECT_SIZE)) {
		ssize_t r = smr_fast_inject_v24(ep, buf, len, dest_addr, tag);
		if (OFI_LIKELY(r != -FI_EOPNOTSUPP))
			return r;
	}
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, 0,
				  ofi_op_tagged, 0);
}

static ssize_t smr_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     void *desc, uint64_t data, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, &desc, 1, dest_addr, tag, data,
				   context, ofi_op_tagged,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
}

static ssize_t smr_tinjectdata(struct fid_ep *ep_fid, const void *buf,
			       size_t len, uint64_t data, fi_addr_t dest_addr,
			       uint64_t tag)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, data,
				  ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

struct fi_ops_tagged smr_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = smr_trecv,
	.recvv = smr_trecvv,
	.recvmsg = smr_trecvmsg,
	.send = smr_tsend,
	.sendv = smr_tsendv,
	.sendmsg = smr_tsendmsg,
	.inject = smr_tinject,
	.senddata = smr_tsenddata,
	.injectdata = smr_tinjectdata,
};

struct fi_ops_tagged smr_no_recv_tag_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = fi_no_tagged_recv,
	.recvv = fi_no_tagged_recvv,
	.recvmsg = fi_no_tagged_recvmsg,
	.send = smr_tsend,
	.sendv = smr_tsendv,
	.sendmsg = smr_tsendmsg,
	.inject = smr_tinject,
	.senddata = smr_tsenddata,
	.injectdata = smr_tinjectdata,
};