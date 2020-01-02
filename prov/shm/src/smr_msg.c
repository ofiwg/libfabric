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


static inline uint16_t smr_convert_rx_flags(uint64_t fi_flags)
{
	uint16_t flags = 0;

	if (fi_flags & FI_COMPLETION)
		flags |= SMR_RX_COMPLETION;
	if (fi_flags & FI_MULTI_RECV)
		flags |= SMR_MULTI_RECV;

	return flags;
}

static inline struct smr_ep_entry *smr_get_recv_entry(struct smr_ep *ep,
		fi_addr_t addr, uint64_t flags)
{
	struct smr_ep_entry *entry;

	if (freestack_isempty(ep->recv_fs))
		return NULL;

	entry = freestack_pop(ep->recv_fs);

	entry->tag = 0; /* does this need to be set? */
	entry->ignore = 0; /* does this need to be set? */
	entry->err = 0;
	entry->flags = smr_convert_rx_flags(flags);
	entry->addr = ep->util_ep.caps & FI_DIRECTED_RECV ? addr : FI_ADDR_UNSPEC;

	return entry;
}

static inline ssize_t
smr_process_recv_post(struct smr_ep *ep, struct smr_ep_entry *entry)
{
	ssize_t ret;

	ret = smr_progress_unexp(ep, entry, &ep->unexp_msg_queue);
	if (!ret || ret == -FI_EAGAIN)
		return ret;

	dlist_insert_tail(&entry->entry, &ep->recv_queue.list);
	return 0;
}

ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret = 0;

	assert(msg->iov_count <= SMR_IOV_LIMIT);
	assert(!(flags & FI_MULTI_RECV) || msg->iov_count == 1);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_recv_entry(ep, msg->addr, flags | ep->util_ep.rx_msg_flags);
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = msg->iov_count;
	memcpy(&entry->iov, msg->msg_iov, sizeof(*msg->msg_iov) * msg->iov_count);

	entry->context = msg->context;

	ret = smr_process_recv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret = 0;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	assert(count <= SMR_IOV_LIMIT);
	assert(!(smr_ep_rx_flags(ep) & FI_MULTI_RECV) || count == 1);

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_recv_entry(ep, src_addr, smr_ep_rx_flags(ep));
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = count;
	memcpy(&entry->iov, iov, sizeof(*iov) * count);

	entry->context = context;

	ret = smr_process_recv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret = 0;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_recv_entry(ep, src_addr, smr_ep_rx_flags(ep));
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = 1;
	entry->iov[0].iov_base = buf;
	entry->iov[0].iov_len = len;

	entry->context = context;

	ret = smr_process_recv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
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
		if (ofi_cirque_isfull(smr_resp_queue(ep->region))) {
			ret = -FI_EAGAIN;
			goto unlock_cq;
		}
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		pend = freestack_pop(ep->pend_fs);
		smr_format_iov(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
			       iov_count, total_len, op, tag, data, op_flags,
			       context, ep->region, resp, pend);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		goto commit;
	}
	ret = smr_complete_tx(ep, context, op, cmd->msg.hdr.op_flags, 0);
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
				   ofi_op_msg, flags | ep->util_ep.tx_msg_flags);
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
	ofi_ep_tx_cntr_inc_func(&ep->util_ep, op);
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
				   ofi_op_msg,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
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

static inline struct smr_ep_entry *smr_get_trecv_entry(struct smr_ep *ep, uint64_t flags)
{
	struct smr_ep_entry *entry;

	if (freestack_isempty(ep->recv_fs))
		return NULL;

	entry = freestack_pop(ep->recv_fs);
	entry->err = 0;
	entry->flags = smr_convert_rx_flags(flags);

	return entry;
}

static inline ssize_t
smr_proccess_trecv_post(struct smr_ep *ep, struct smr_ep_entry *entry)
{
	ssize_t ret;

	ret = smr_progress_unexp(ep, entry, &ep->unexp_tagged_queue);
	if (!ret || ret == -FI_EAGAIN)
		return ret;

	dlist_insert_tail(&entry->entry, &ep->trecv_queue.list);
	return 0;
}

ssize_t smr_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_trecv_entry(ep, smr_ep_rx_flags(ep));
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = 1;
	entry->iov[0].iov_base = buf;
	entry->iov[0].iov_len = len;

	entry->context = context;
	entry->addr = src_addr;
	entry->tag = tag;
	entry->ignore = ignore;

	ret = smr_proccess_trecv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	assert(count <= SMR_IOV_LIMIT);
	assert(!(smr_ep_rx_flags(ep) & FI_MULTI_RECV) || count == 1);

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_trecv_entry(ep, smr_ep_rx_flags(ep));
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = count;
	memcpy(&entry->iov, iov, sizeof(*iov) * count);

	entry->context = context;
	entry->addr = src_addr;
	entry->tag = tag;
	entry->ignore = ignore;

	ret = smr_proccess_trecv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	struct smr_ep_entry *entry;
	struct smr_ep *ep;
	ssize_t ret;

	assert(msg->iov_count <= SMR_IOV_LIMIT);
	assert(!(flags & FI_MULTI_RECV) || msg->iov_count == 1);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	entry = smr_get_trecv_entry(ep, flags | ep->util_ep.rx_msg_flags);
	if (!entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry->iov_count = msg->iov_count;
	memcpy(&entry->iov, msg->msg_iov, sizeof(*msg->msg_iov) * msg->iov_count);

	entry->context = msg->context;
	entry->addr = msg->addr;
	entry->tag = msg->tag;
	entry->ignore = msg->ignore;

	ret = smr_proccess_trecv_post(ep, entry);
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
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
				   ofi_op_tagged, flags | ep->util_ep.tx_msg_flags);
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
				   ofi_op_tagged,
				   FI_REMOTE_CQ_DATA | smr_ep_tx_flags(ep));
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
