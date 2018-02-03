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


static void smr_format_rma_iov(struct smr_cmd *cmd, const struct fi_rma_iov *rma_iov,
			       size_t iov_count)
{
	cmd->rma.rma_count = iov_count;
	memcpy(cmd->rma.rma_iov, rma_iov, sizeof(*rma_iov) * iov_count);
}

static void smr_format_rma_resp(struct smr_cmd *cmd, fi_addr_t peer_id,
				const struct fi_rma_iov *rma_iov, size_t count,
				size_t total_len, uint32_t op)
{
	smr_generic_format(cmd, peer_id, op, 0, 0, 0, 0, 0);
	cmd->msg.hdr.size = total_len;
}

ssize_t smr_rma_fast(struct smr_ep *ep, const struct iovec *iov,
	size_t iov_count, const struct fi_rma_iov *rma_iov, size_t rma_count,
	void **desc, int peer_id, void *context, uint32_t op)
{
	struct smr_region *peer_smr;
	struct iovec rma_iovec[SMR_IOV_LIMIT];
	struct smr_cmd *cmd;
	size_t total_len;
	int ret, i;

	for (i = 0; i < rma_count; i++) {
		rma_iovec[i].iov_base = (void *) rma_iov[i].addr;
		rma_iovec[i].iov_len = rma_iov[i].len;
	}

	total_len = ofi_total_iov_len(iov, iov_count);

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

	if (op == ofi_op_write) {
		ret = process_vm_writev(peer_smr->pid, iov, iov_count,
					rma_iovec, rma_count, 0);
	} else {
		ret = process_vm_readv(peer_smr->pid, iov, iov_count,
				       rma_iovec, rma_count, 0);
	}

	if (ret != total_len) {
		if (ret < 0) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"CMA write error\n");
			ret = -errno;
		} else {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
			ret = -FI_EIO;
		}
		goto comp;
	}
	ret = 0;

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_resp(cmd, peer_id, rma_iov, rma_count, total_len,
			    (op == ofi_op_write) ? ofi_op_write_rsp :
			    ofi_op_read_rsp);

	peer_smr->cmd_cnt--;
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
comp:
	ret = ep->tx_comp(ep, context, smr_tx_comp_flags(op), ret);
unlock_cq:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

ssize_t smr_generic_rma(struct fid_ep *ep_fid, const struct iovec *iov,
	size_t iov_count, const struct fi_rma_iov *rma_iov, size_t rma_count,
	void **desc, fi_addr_t addr, void *context, uint32_t op, uint64_t data,
	uint16_t op_flags)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_resp *resp;
	struct smr_cmd *cmd, *pend;
	int peer_id, comp = 1;
	ssize_t ret = 0;
	size_t total_len;

	assert(iov_count <= SMR_IOV_LIMIT);
	assert(rma_count <= SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	peer_id = (int) addr;
	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	if (rma_count == 1 && !(op_flags & OFI_REMOTE_CQ_DATA))
		return smr_rma_fast(ep, iov, iov_count, rma_iov, rma_count,
				    desc, peer_id, context, op);

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (peer_smr->cmd_cnt < 2) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto unlock_cq;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	total_len = ofi_total_iov_len(iov, iov_count);

	if (total_len <= SMR_MSG_DATA_LEN && op == ofi_op_write) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  iov, iov_count, op, 0, data, op_flags);
	} else if (total_len <= SMR_INJECT_SIZE && op == ofi_op_write) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  iov, iov_count, op, 0, data, op_flags,
				  peer_smr, tx_buf);
	} else {
		assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		pend = freestack_pop(ep->pend_fs);
		smr_format_iov(cmd, smr_peer_addr(ep->region)[peer_id].addr,
			       iov, iov_count, total_len, op, 0, data,
			       op_flags, ep->region, resp, pend);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		comp = 0;
	}

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;
	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_iov(cmd, rma_iov, rma_count);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;

	if (!comp)
		goto unlock_cq;

	ret = ep->tx_comp(ep, context, smr_tx_comp_flags(op), 0);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process tx completion\n");
	}

unlock_cq:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

ssize_t smr_read(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return smr_generic_rma(ep_fid, &msg_iov, 1, &rma_iov, 1, &desc, 
			       src_addr, context, ofi_op_read_req, 0, 0);
}

ssize_t smr_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
	void *context)
{
	struct fi_rma_iov rma_iov;

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return smr_generic_rma(ep_fid, iov, count, &rma_iov, 1, desc,
			       src_addr, context, ofi_op_read_req, 0, 0);
}

ssize_t smr_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	return smr_generic_rma(ep_fid, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_read_req, 0, 0);
}

ssize_t smr_write(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return smr_generic_rma(ep_fid, &msg_iov, 1, &rma_iov, 1, &desc, 
			       dest_addr, context, ofi_op_write, 0, 0);
}

ssize_t smr_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		void *context)
{
	struct fi_rma_iov rma_iov;

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return smr_generic_rma(ep_fid, iov, count, &rma_iov, 1, desc,
			       dest_addr, context, ofi_op_write, 0, 0);
}


ssize_t smr_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	return smr_generic_rma(ep_fid, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_write, msg->data,
			       (flags & FI_REMOTE_CQ_DATA) ?
			       OFI_REMOTE_CQ_DATA : 0);
}

ssize_t smr_rma_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_cmd *cmd;
	struct iovec iov, rma_iovec;
	struct fi_rma_iov rma_iov;
	int peer_id;
	ssize_t ret = 0;

	assert(len <= SMR_INJECT_SIZE);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	peer_id = (int) dest_addr;
	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (!peer_smr->cmd_cnt) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iovec.iov_base = (void *) addr;
	rma_iovec.iov_len = len;

	ret = process_vm_writev(peer_smr->pid, &iov, 1,
				&rma_iovec, 1, 0);

	if (ret != len) {
		if (ret < 0) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"CMA write error\n");
			ret = -errno;
		} else {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
			ret = -FI_EIO;
		}
		goto unlock_region;
	}
	ret = 0;

	rma_iov.addr = addr;
	rma_iov.len = len;

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_resp(cmd, peer_id, &rma_iov, 1, len,
			    ofi_op_read_rsp);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;

unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

ssize_t smr_writedata(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return smr_generic_rma(ep, &iov, 1, &rma_iov, 1, &desc, dest_addr, context,
			       ofi_op_write, data, OFI_REMOTE_CQ_DATA);
}

ssize_t smr_inject_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_cmd *cmd;
	struct iovec iov;
	struct fi_rma_iov rma_iov;
	int peer_id;
	ssize_t ret = 0;

	assert(len <= SMR_INJECT_SIZE);
	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	peer_id = (int) dest_addr;
	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (peer_smr->cmd_cnt < 2) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	if (len <= SMR_MSG_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &iov, 1, ofi_op_write, 0, data,
				  OFI_REMOTE_CQ_DATA);
	} else {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &iov, 1, ofi_op_write, 0, data,
				  OFI_REMOTE_CQ_DATA, peer_smr, tx_buf);
	}

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;
	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma_iov(cmd, &rma_iov, 1);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	peer_smr->cmd_cnt--;

unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

struct fi_ops_rma smr_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = smr_read,
	.readv = smr_readv,
	.readmsg = smr_readmsg,
	.write = smr_write,
	.writev = smr_writev,
	.writemsg = smr_writemsg,
	.inject = smr_rma_inject,
	.writedata = smr_writedata,
	.injectdata = smr_inject_writedata,
};
