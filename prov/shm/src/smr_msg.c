/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved
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

#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>

#include "ofi_iov.h"
#include "smr.h"


static ssize_t smr_generic_recvmsg(struct smr_ep *ep, const struct iovec *iov,
				   size_t iov_count, fi_addr_t addr, uint64_t tag,
				   uint64_t ignore, void *context, uint64_t flags)
{
	struct smr_queue *recv_queue;
	struct smr_ep_entry *entry;
	ssize_t ret;

	assert(iov_count <= SMR_IOV_LIMIT);
	assert(!(flags & FI_MULTI_RECV) || iov_count == 1);

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (freestack_isempty(ep->recv_fs)) {
		ret = -FI_EAGAIN;
		goto out;
	}
	entry = freestack_pop(ep->recv_fs);
	memset(entry, 0, sizeof(*entry));

	for (entry->iov_count = 0; entry->iov_count < iov_count;
	     entry->iov_count++) {
		entry->iov[entry->iov_count] = iov[entry->iov_count];
	}

	entry->context = context;
	entry->flags = flags;
	entry->addr = addr;
	entry->tag = tag;
	entry->ignore = ignore;

	if (flags & FI_TAGGED) {
		ret = smr_progress_unexp(ep, entry);
		if (!ret || ret == -FI_EAGAIN)
			goto out;
		recv_queue = &ep->trecv_queue;
	} else {
		recv_queue = &ep->recv_queue;
	}

	dlist_insert_tail(&entry->entry, &recv_queue->list);

	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recvmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, 0, msg->context, flags);
}

ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recvmsg(ep, iov, count, src_addr,
				   0, 0, context, smr_ep_rx_flags(ep));
}

ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_recvmsg(ep, &msg_iov, 1, src_addr, 0, 0, context,
				   smr_ep_rx_flags(ep));
}

static ssize_t smr_generic_sendmsg(struct smr_ep *ep, const struct iovec *iov,
				   size_t iov_count, fi_addr_t addr, uint64_t tag,
				   uint64_t data, void *context, uint32_t op,
				   uint64_t op_flags)
{
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_resp *resp;
	struct smr_cmd *cmd, *pend;
	int peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(iov_count <= SMR_IOV_LIMIT);

	peer_id = (int) addr;

	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (!peer_smr->cmd_cnt) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto unlock_cq;
	}

	total_len = ofi_total_iov_len(iov, iov_count);

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	if (total_len <= SMR_MSG_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
				  iov_count, op, tag, data, op_flags);
	} else if (total_len <= SMR_INJECT_SIZE) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  iov, iov_count, op, tag, data, op_flags,
				  peer_smr, tx_buf);
	} else {
		assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		pend = freestack_pop(ep->pend_fs);
		smr_format_iov(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
			       iov_count, total_len, op, tag, data, op_flags,
			       context, ep->region, resp, pend);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		goto commit;
	}
	ret = ep->tx_comp(ep, context, smr_tx_comp_flags(op), 0);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
		goto unlock_cq;
	}

commit:
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;
unlock_cq:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

ssize_t smr_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, 1, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}


ssize_t smr_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, count, dest_addr, 0,
				   0, context, ofi_op_msg, smr_ep_tx_flags(ep));
}

ssize_t smr_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg, flags);
}

static ssize_t smr_generic_inject(struct fid_ep *ep_fid, const void *buf,
				  size_t len, fi_addr_t dest_addr, uint64_t tag,
				  uint64_t data, uint32_t op, uint64_t op_flags)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_cmd *cmd;
	int peer_id;
	ssize_t ret = 0;
	struct iovec msg_iov;

	assert(len <= SMR_INJECT_SIZE);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	peer_id = (int) dest_addr;

	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (!peer_smr->cmd_cnt) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	if (len <= SMR_MSG_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &msg_iov, 1, op, tag, data, op_flags);
	} else {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &msg_iov, 1, op, tag, data, op_flags,
				  peer_smr, tx_buf);
	}

	peer_smr->cmd_cnt--;
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
unlock:
	fastlock_release(&peer_smr->lock);

	return ret;
}

ssize_t smr_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, 0, 0,
				  ofi_op_msg, 0);
}

ssize_t smr_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		     void *desc, uint64_t data, fi_addr_t dest_addr,
		     void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, 1, dest_addr, 0, data, context,
				   ofi_op_msg, FI_REMOTE_CQ_DATA);
}

ssize_t smr_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
		       uint64_t data, fi_addr_t dest_addr)
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

ssize_t smr_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_recvmsg(ep, &msg_iov, 1, src_addr, tag, ignore,
				   context, FI_TAGGED | smr_ep_tx_flags(ep));
}

ssize_t smr_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recvmsg(ep, iov, count, src_addr, tag, ignore,
				   context, FI_TAGGED | smr_ep_tx_flags(ep));
}

ssize_t smr_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_recvmsg(ep, msg->msg_iov, msg->iov_count, msg->addr,
				   msg->tag, msg->ignore, msg->context,
				   flags | FI_TAGGED);
}

ssize_t smr_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct smr_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep, &msg_iov, 1, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

ssize_t smr_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
	void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, iov, count, dest_addr, tag,
				   0, context, ofi_op_tagged,
				   smr_ep_tx_flags(ep));
}

ssize_t smr_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct smr_ep *ep;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	return smr_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, msg->tag, msg->data, msg->context,
				   ofi_op_tagged, flags);
}

ssize_t smr_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
		    fi_addr_t dest_addr, uint64_t tag)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, 0,
				  ofi_op_tagged, 0);
}

ssize_t smr_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t tag, void *context)
{
	struct smr_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return smr_generic_sendmsg(ep, &iov, 1, dest_addr, tag, data, context,
				   ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

ssize_t smr_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	return smr_generic_inject(ep_fid, buf, len, dest_addr, tag, data,
				  ofi_op_tagged, FI_REMOTE_CQ_DATA);
}

struct fi_ops_tagged smr_tagged_ops = {
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
