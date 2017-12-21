/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include "fi_iov.h"
#include "smr.h"


int smr_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct smr_ep *ep;
	char *name;

	ep = container_of(fid, struct smr_ep, util_ep.ep_fid.fid);
	name = strdup(addr);
	if (!name)
		return -FI_ENOMEM;

	if (ep->name)
		free((void *) ep->name);
	ep->name = name;
	return 0;
}

int smr_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct smr_ep *ep;
	int ret = 0;

	ep = container_of(fid, struct smr_ep, util_ep.ep_fid.fid);
	if (!ep->name)
		return -FI_EADDRNOTAVAIL;

	if (!addr || *addrlen == 0 ||
	    snprintf(addr, *addrlen, "%s", ep->name) >= *addrlen)
		ret = -FI_ETOOSMALL;
	*addrlen = sizeof(struct smr_addr);
	return ret;
}

static struct fi_ops_cm smr_cm_ops = {
	.size = sizeof(struct fi_ops_cm),
	.setname = smr_setname,
	.getname = smr_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
};

int smr_getopt(fid_t fid, int level, int optname,
		void *optval, size_t *optlen)
{
	return -FI_ENOPROTOOPT;
}

int smr_setopt(fid_t fid, int level, int optname,
		const void *optval, size_t optlen)
{
	return -FI_ENOPROTOOPT;
}

static struct fi_ops_ep smr_ep_ops = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = fi_no_cancel,
	.getopt = smr_getopt,
	.setopt = smr_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int smr_tx_comp(struct smr_ep *ep, void *context, uint64_t flags,
		       uint64_t err)
{
	struct fi_cq_tagged_entry *comp;
	struct util_cq_err_entry *entry;

	comp = ofi_cirque_tail(ep->util_ep.tx_cq->cirq);
	if (err) {
		if (!(entry = calloc(1, sizeof(*entry))))
			return -FI_ENOMEM;
		entry->err_entry.op_context = context;
		entry->err_entry.flags = flags;
		entry->err_entry.err = err;
		entry->err_entry.prov_errno = -err;
		slist_insert_tail(&entry->list_entry,
				  &ep->util_ep.tx_cq->err_list);
		comp->flags = UTIL_FLAG_ERROR;
	} else {
		comp->op_context = context;
		comp->flags = flags;
		comp->len = 0;
		comp->buf = NULL;
		comp->data = 0;
	}
	ofi_cirque_commit(ep->util_ep.tx_cq->cirq);
	return 0;
}

static int smr_tx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
			      uint64_t err)
{
	int ret;

	ret = smr_tx_comp(ep, context, flags, err);
	if (ret)
		return ret;
	ep->util_ep.tx_cq->wait->signal(ep->util_ep.tx_cq->wait);
	return 0;
}

static int smr_rx_comp(struct smr_ep *ep, void *context, uint64_t flags,
		       size_t len, void *buf, void *addr, uint64_t tag,
		       uint64_t err)
{
	struct fi_cq_tagged_entry *comp;
	struct util_cq_err_entry *entry;

	comp = ofi_cirque_tail(ep->util_ep.rx_cq->cirq);
	if (err) {
		if (!(entry = calloc(1, sizeof(*entry))))
			return -FI_ENOMEM;
		entry->err_entry.op_context = context;
		entry->err_entry.flags = FI_RECV | flags;
		entry->err_entry.tag = tag;
		entry->err_entry.err = err;
		entry->err_entry.prov_errno = -err;
		slist_insert_tail(&entry->list_entry,
				  &ep->util_ep.rx_cq->err_list);
		comp->flags = UTIL_FLAG_ERROR;
	} else {
		comp->op_context = context;
		comp->flags = FI_RECV | flags;
		comp->len = len;
		comp->buf = buf;
		comp->data = 0;
		comp->tag = tag;
	}
	ofi_cirque_commit(ep->util_ep.rx_cq->cirq);
	return 0;
}

static int smr_rx_src_comp(struct smr_ep *ep, void *context, uint64_t flags,
			   size_t len, void *buf, void *addr, uint64_t tag,
			   uint64_t err)
{
	ep->util_ep.rx_cq->src[ofi_cirque_windex(ep->util_ep.rx_cq->cirq)] =
		(uint32_t) (uintptr_t) addr;
	return smr_rx_comp(ep, context, flags, len, buf, addr, tag, err);
}

static int smr_rx_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
			      size_t len, void *buf, void *addr, uint64_t tag,
			      uint64_t err)
{
	int ret;

	ret = smr_rx_comp(ep, context, flags, len, buf, addr, tag, err);
	if (ret)
		return ret;
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
	return 0;
}

static int smr_rx_src_comp_signal(struct smr_ep *ep, void *context, uint64_t flags,
				  size_t len, void *buf, void *addr, uint64_t tag,
				  uint64_t err)
{
	int ret;

	ret = smr_rx_src_comp(ep, context, flags, len, buf, addr, tag, err);
	if (ret)
		return ret;
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
	return 0;

}
static int smr_verify_peer(struct smr_ep *ep, int peer_id)
{
	int ret;

	if (ep->region->map->peers[peer_id].peer.addr != FI_ADDR_UNSPEC)
		return 0;

	ret = smr_map_to_region(&smr_prov, &ep->region->map->peers[peer_id]);

	return (ret == -ENOENT) ? -FI_EAGAIN : ret;
}

static uint64_t smr_tx_comp_flags(uint32_t op)
{
	uint64_t flags = 0;

	switch (op) {
	case ofi_op_tagged:
		flags = FI_TAGGED;
		/* fall through */
	case ofi_op_msg:
		flags |= FI_SEND;
		break;
	case ofi_op_write:
		flags = FI_RMA | FI_WRITE;
		break;
	case ofi_op_read_req:
		flags = FI_RMA | FI_READ;
		break;
	default:
		break;
	}

	return flags;
}

void smr_progress_resp(struct smr_ep *ep)
{
	struct smr_resp *resp;
	struct dlist_entry *dlist_entry;
	struct smr_pending_cmd *pending;
	struct smr_match_attr match_attr;
	int ret;

	fastlock_acquire(&ep->region->lock);
	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	while (!ofi_cirque_isempty(smr_resp_queue(ep->region)) &&
	       !ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		resp = ofi_cirque_head(smr_resp_queue(ep->region));
		if (resp->status == FI_EBUSY)
			break;
		match_attr.ctx = resp->msg_id;;
		dlist_entry = dlist_remove_first_match(&ep->pend_queue.msg_list,
						       ep->pend_queue.match_msg,
						       &match_attr);
		if (!dlist_entry) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"no outstanding commands for found response\n");
			break;
		}
		pending = container_of(dlist_entry, struct smr_pending_cmd, entry);

		ret = ep->tx_comp(ep, (void *) resp->msg_id,
				  smr_tx_comp_flags(pending->cmd.hdr.op.op),
				  -(resp->status));
		if (ret) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unable to process tx completion\n");
			break;
		}
		freestack_push(ep->pend_fs, pending);
		ofi_cirque_discard(smr_resp_queue(ep->region));
	}
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&ep->region->lock);
}

static int smr_progress_inline(struct smr_cmd *cmd, struct iovec *iov,
			       size_t iov_count, size_t *total_len, int err)
{
	if (err)
		return err;

	*total_len = ofi_copy_to_iov(iov, iov_count, 0, cmd->data.msg,
				     cmd->hdr.op.size);
	if (*total_len != cmd->hdr.op.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		return -FI_EIO;
	}
	return 0;
}

static int smr_progress_inject(struct smr_cmd *cmd, struct iovec *iov,
			       size_t iov_count, size_t *total_len,
			       struct smr_ep *ep, int err)
{
	struct smr_inject_buf *tx_buf;
	size_t inj_offset;

	inj_offset = (size_t) cmd->hdr.op.data;
	tx_buf = (struct smr_inject_buf *) ((char **) ep->region +
					    inj_offset);
	if (err)
		goto out;

	*total_len = ofi_copy_to_iov(iov, iov_count, 0, tx_buf,
				     cmd->hdr.op.size);
	if (*total_len != cmd->hdr.op.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		err = -FI_EIO;
	}

out:
	smr_freestack_push(smr_inject_pool(ep->region), tx_buf);
	return err;
}

static int smr_progress_iov(struct smr_cmd *cmd, struct iovec *iov,
			    size_t iov_count, size_t *total_len,
			    struct smr_ep *ep, int err)
{
	struct smr_region *peer_smr;
	struct smr_resp *resp;
	int peer_id, ret;

	peer_id = (int) cmd->hdr.op.addr;
	peer_smr = smr_peer_region(ep->region, peer_id);
	resp = (struct smr_resp *) ((char **) peer_smr +
				    (size_t) cmd->hdr.op.data);

	if (err) {
		ret = -err;
		goto out;
	}

	if (cmd->hdr.op.op == ofi_op_read_req) {
		ret = process_vm_writev(peer_smr->pid, iov, iov_count,
					cmd->data.iov, SMR_IOV_LIMIT, 0);
	} else {
		ret = process_vm_readv(peer_smr->pid, iov, iov_count,
				       cmd->data.iov, SMR_IOV_LIMIT, 0);
	}

	if (ret != cmd->hdr.op.size) {
		if (ret < 0) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"CMA write error\n");
			ret = errno;
		} else { 
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"partial read occurred\n");
			ret = FI_EIO;
		}
	} else {
		*total_len = ret;
		ret = 0;
	}

out:
	resp->msg_id = cmd->hdr.msg_id;

	//Status must be set last (signals peer: op done, valid resp entry)
	resp->status = ret;

	return -ret;
}

static int smr_progress_cmd_msg(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_recv_queue *recv_queue;
	struct smr_match_attr match_attr;
	struct dlist_entry *dlist_entry;
	struct smr_ep_entry *entry;
	struct smr_pending_cmd *unexp;
	fi_addr_t addr;
	size_t total_len = 0;
	int ret = 0;

	recv_queue = (cmd->hdr.op.op == ofi_op_tagged) ?
		      &ep->trecv_queue : &ep->recv_queue;

	if (dlist_empty(&recv_queue->recv_list))
		return -FI_ENOMSG;

	match_attr.addr = cmd->hdr.op.addr;
	match_attr.tag = cmd->hdr.op.tag;

	dlist_entry = dlist_remove_first_match(&recv_queue->recv_list,
					       recv_queue->match_recv,
					       &match_attr);
	if (!dlist_entry) {
		if (freestack_isempty(ep->unexp_fs))
			return -FI_EAGAIN;
		unexp = freestack_pop(ep->unexp_fs);
		memcpy(&unexp->cmd, cmd, sizeof(*cmd));
		ofi_cirque_discard(smr_cmd_queue(ep->region));
		dlist_insert_tail(&unexp->entry, &ep->unexp_queue.msg_list);
		goto discard;
	}
	entry = container_of(dlist_entry, struct smr_ep_entry, entry);

	switch (cmd->hdr.op.op_src) {
	case smr_src_inline:
		ret = smr_progress_inline(cmd, entry->iov, entry->iov_count,
					  &total_len, 0);
		break;
	case smr_src_inject:
		ret = smr_progress_inject(cmd, entry->iov, entry->iov_count,
					  &total_len, ep, 0);
		break;
	case smr_src_iov:
		ret = smr_progress_iov(cmd, entry->iov, entry->iov_count,
				       &total_len, ep, 0);
		break;
	default:
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		ret = -FI_EINVAL;
	}
	ret = ep->rx_comp(ep, entry->context, entry->flags, total_len,
			  entry->iov[0].iov_base, &addr, cmd->hdr.op.tag, ret);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
	}
	freestack_push(ep->recv_fs, entry);

discard:
	ofi_cirque_discard(smr_cmd_queue(ep->region));
	return ret;
}

static int smr_progress_cmd_rma(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_domain *domain;
	struct iovec iov[SMR_RMA_IOV_LIMIT];
	size_t iov_count;
	size_t total_len = 0;
	int ret = 0;

	domain = container_of(ep->util_ep.domain, struct smr_domain,
			      util_domain);

	for (iov_count = 0; iov_count < cmd->hdr.op.iov_count; iov_count++) {
		ret = ofi_mr_verify(&domain->util_domain.mr_map,
				cmd->data.rma_iov[iov_count].len,
				(uintptr_t *) &(cmd->data.rma_iov[iov_count].addr),
				cmd->data.rma_iov[iov_count].key,
				cmd->hdr.op.op == ofi_op_write ?
				FI_REMOTE_WRITE : FI_REMOTE_READ);
		if (ret)
			break;

		iov[iov_count].iov_base = (void *) cmd->data.rma_iov[iov_count].addr;
		iov[iov_count].iov_len = cmd->data.rma_iov[iov_count].len;
	}

	ofi_cirque_discard(smr_cmd_queue(ep->region));
	cmd = ofi_cirque_head(smr_cmd_queue(ep->region));
	if (cmd->hdr.op.op != ofi_op_write &&
	    cmd->hdr.op.op != ofi_op_read_req) {
		ret = -FI_EINVAL;
		goto discard;
	}

	switch (cmd->hdr.op.op_src) {
	case smr_src_inline:
		ret = smr_progress_inline(cmd, iov, iov_count, &total_len, ret);
		break;
	case smr_src_inject:
		ret = smr_progress_inject(cmd, iov, iov_count, &total_len, ep, ret);
		break;
	case smr_src_iov:
		ret = smr_progress_iov(cmd, iov, iov_count, &total_len, ep, ret);
		break;
	default:
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		ret = -FI_EINVAL;
	}

discard:
	ofi_cirque_discard(smr_cmd_queue(ep->region));
	return ret;
}

void smr_progress_cmd(struct smr_ep *ep)
{
	struct smr_cmd *cmd;
	int ret = 0;

	fastlock_acquire(&ep->region->lock);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);

	while (!ofi_cirque_isempty(smr_cmd_queue(ep->region))) {
		cmd = ofi_cirque_head(smr_cmd_queue(ep->region));

		switch (cmd->hdr.op.op) {
		case ofi_op_msg:
		case ofi_op_tagged:
			ret = smr_progress_cmd_msg(ep, cmd);
			break;
		case ofi_op_write:
		case ofi_op_read_req:
			ret = smr_progress_cmd_rma(ep, cmd);
			break;
		case ofi_op_write_rsp:
		case ofi_op_read_rsp:
			ofi_cirque_discard(smr_cmd_queue(ep->region));
			break;
		default:
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
			ret = -FI_EINVAL;
		}

		if (ret) {
			if (ret != -FI_EAGAIN) {
				FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
					"error processing command\n");
			}
			break;
		}
	}
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	fastlock_release(&ep->region->lock);
}

static int smr_match_msg(struct dlist_entry *item, const void *args)
{
	struct smr_match_attr *attr = (struct smr_match_attr *)args;
	struct smr_ep_entry *recv_entry;

	recv_entry = container_of(item, struct smr_ep_entry, entry);
	return smr_match_addr(recv_entry->addr, attr->addr);
}

static int smr_match_tagged(struct dlist_entry *item, const void *args)
{
	struct smr_match_attr *attr = (struct smr_match_attr *)args;
	struct smr_ep_entry *recv_entry;

	recv_entry = container_of(item, struct smr_ep_entry, entry);
	return smr_match_addr(recv_entry->addr, attr->addr) &&
	       smr_match_tag(recv_entry->tag, recv_entry->ignore, attr->tag); 
} 

static int smr_match_unexp(struct dlist_entry *item, const void *args)
{
	struct smr_match_attr *attr = (struct smr_match_attr *)args;
	struct smr_pending_cmd *unexp_msg;

	unexp_msg = container_of(item, struct smr_pending_cmd, entry);
	return smr_match_addr(unexp_msg->cmd.hdr.op.addr, attr->addr) &&
	       smr_match_tag(unexp_msg->cmd.hdr.op.tag, attr->ignore, attr->tag);
}

static int smr_match_ctx(struct dlist_entry *item, const void *args)
{
	struct smr_match_attr *attr = (struct smr_match_attr *)args;
	struct smr_pending_cmd *pending_msg;

	pending_msg = container_of(item, struct smr_pending_cmd, entry);
	return pending_msg->cmd.hdr.msg_id == attr->ctx;
}

static void smr_init_recv_queue(struct smr_recv_queue *recv_queue,
				dlist_func_t *match_func)
{
	dlist_init(&recv_queue->recv_list);
	recv_queue->match_recv = match_func;
}

static void smr_init_pending_queue(struct smr_pending_queue *queue,
				   dlist_func_t *match_func)
{
	dlist_init(&queue->msg_list);
	queue->match_msg = match_func;
}

void smr_ep_progress(struct util_ep *util_ep)
{
	struct smr_ep *ep;

	ep = container_of(util_ep, struct smr_ep, util_ep);

	smr_progress_resp(ep);
	smr_progress_cmd(ep);
}

static int smr_check_unexp(struct smr_ep *ep, struct smr_ep_entry *entry)
{
	struct smr_match_attr match_attr;
	struct smr_pending_cmd *unexp_msg;
	struct dlist_entry *dlist_entry;
	size_t total_len = 0;
	int ret = 0;

	match_attr.addr = entry->addr;
	match_attr.ignore = entry->ignore;
	match_attr.tag = entry->tag;
	dlist_entry = dlist_remove_first_match(&ep->unexp_queue.msg_list,
					       ep->unexp_queue.match_msg,
					       &match_attr);
	if (!dlist_entry)
		return 0;

	unexp_msg = container_of(dlist_entry, struct smr_pending_cmd, entry);

	switch (unexp_msg->cmd.hdr.op.op_src){
	case smr_src_inline:
		ret = smr_progress_inline(&unexp_msg->cmd, entry->iov, entry->iov_count,
					  &total_len, 0);
		break;
	case smr_src_inject:
		ret = smr_progress_inject(&unexp_msg->cmd, entry->iov, entry->iov_count,
					  &total_len, ep, 0);
		break;
	case smr_src_iov:
		ret = smr_progress_iov(&unexp_msg->cmd, entry->iov, entry->iov_count,
				       &total_len, ep, 0);
		break;
	default:
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unidentified operation type\n");
		entry->err = FI_EINVAL;
	}
	ret = ep->rx_comp(ep, entry->context, FI_TAGGED, total_len,
			  entry->iov[0].iov_base, &entry->addr, entry->tag,
			  entry->err);
	if (ret) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unable to process rx completion\n");
	}
	freestack_push(ep->unexp_fs, unexp_msg);

	return 1;
}

static ssize_t smr_generic_recvmsg(struct fid_ep *ep_fid, const struct iovec *iov,
				   size_t iov_count, fi_addr_t addr, uint64_t tag,
				   uint64_t ignore, void *context, uint64_t flags)
{
	struct smr_ep *ep;
	struct smr_recv_queue *recv_queue;
	struct smr_ep_entry *entry;
	ssize_t ret;

	assert(iov_count <= SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
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
		if (smr_check_unexp(ep, entry)) {
			ret = 0;
			goto out;
		}
		recv_queue = &ep->trecv_queue;
	} else {
		recv_queue = &ep->recv_queue;
	}

	dlist_insert_tail(&entry->entry, &recv_queue->recv_list);

	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	return smr_generic_recvmsg(ep_fid, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, 0, msg->context, flags);
}

ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	return smr_generic_recvmsg(ep_fid, iov, count, src_addr,
				   0, 0, context, 0);
}

ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_recvmsg(ep_fid, &msg_iov, 1, src_addr, 0, 0, context, 0);
}

static void smr_post_pending(struct smr_ep *ep, struct smr_cmd *cmd)
{
	struct smr_pending_cmd *pend_cmd;

	pend_cmd = freestack_pop(ep->pend_fs);
	pend_cmd->cmd = *cmd;

	dlist_insert_tail(&pend_cmd->entry, &ep->pend_queue.msg_list);
}

static void smr_generic_format(struct smr_cmd *cmd, fi_addr_t peer_id, void *context,
			       uint32_t op, uint64_t tag)
{
	cmd->hdr.op.op = op;
	cmd->hdr.op.tag = tag;
	cmd->hdr.msg_id = (uint64_t) context;
	cmd->hdr.op.addr = peer_id;
}

static void smr_format_rma(struct smr_cmd *cmd, fi_addr_t peer_id,
			   const struct fi_rma_iov *rma_iov, size_t iov_count,
			   void *context, uint32_t op)
{
	smr_generic_format(cmd, peer_id, context, op, 0);
	cmd->hdr.op.iov_count = iov_count;
	memcpy(cmd->data.rma_iov, rma_iov, sizeof(*rma_iov) * iov_count);
}

static void smr_format_inline(struct smr_cmd *cmd, fi_addr_t peer_id,
			      const struct iovec *iov, size_t count,
			      void *context, uint32_t op, uint64_t tag)
{
	smr_generic_format(cmd, peer_id, context, op, tag);
	cmd->hdr.op.op_src = smr_src_inline;
	cmd->hdr.op.data = 0;
	cmd->hdr.op.size = ofi_copy_from_iov(cmd->data.msg, SMR_CMD_DATA_LEN,
					     iov, count, 0);
}

static void smr_format_inject(struct smr_cmd *cmd, fi_addr_t peer_id,
			      const struct iovec *iov, size_t count,
			      void *context, uint32_t op, uint64_t tag,
			      struct smr_region *smr, struct smr_inject_buf *tx_buf)
{
	smr_generic_format(cmd, peer_id, context, op, tag);
	cmd->hdr.op.op_src = smr_src_inject;
	cmd->hdr.op.data = (char **) tx_buf - (char **) smr;
	cmd->hdr.op.size = ofi_copy_from_iov(tx_buf->data, SMR_INJECT_SIZE,
					     iov, count, 0);
}

static void smr_format_iov(struct smr_cmd *cmd, fi_addr_t peer_id,
			   const struct iovec *iov, size_t count,
			   size_t total_len, void *context, uint32_t op,
			   uint64_t tag, struct smr_region *smr,
			   struct smr_resp *resp)
{
	int i;

	smr_generic_format(cmd, peer_id, context, op, tag);
	cmd->hdr.op.op_src = smr_src_iov;
	resp->status = FI_EBUSY;
	cmd->hdr.op.data = (uint64_t) ((char **) resp - (char **) smr);

	for (cmd->hdr.op.size = i = 0; i < count; i++) {
		cmd->data.iov[i].iov_base = iov[i].iov_base;
		cmd->data.iov[i].iov_len = iov[i].iov_len;
		cmd->hdr.op.size += iov[i].iov_len;
	}
	while (i < SMR_IOV_LIMIT)
		cmd->data.iov[i++].iov_len = 0;
}

static void smr_format_resp(struct smr_cmd *cmd, fi_addr_t peer_id,
			    const struct fi_rma_iov *rma_iov, size_t count,
			    size_t total_len, uint32_t op)
{
	smr_generic_format(cmd, peer_id, NULL, op, 0);
	memcpy(cmd->data.rma_iov, rma_iov, sizeof(*rma_iov) * count);
	cmd->hdr.op.size = total_len;
	cmd->hdr.op.iov_count = count;
}

static ssize_t smr_generic_sendmsg(struct fid_ep *ep_fid, const struct iovec *iov,
				   size_t iov_count, fi_addr_t addr, uint64_t tag,
				   void *context, uint32_t op)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_resp *resp;
	struct smr_cmd *cmd;
	int peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(iov_count <= SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	peer_id = (int) addr;

	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
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

	if (total_len <= SMR_CMD_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
				  iov_count, context, op, tag);
	} else if (total_len <= SMR_INJECT_SIZE) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
				  iov_count, context, op, tag, peer_smr, tx_buf);
	} else {
		assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		smr_format_iov(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
			       iov_count, total_len, context, op, tag,
			       ep->region, resp);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		smr_post_pending(ep, cmd);
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
unlock_cq:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

ssize_t smr_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep_fid, &msg_iov, 1, dest_addr, 0,
				   context, ofi_op_msg);
}


ssize_t smr_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	return smr_generic_sendmsg(ep_fid, iov, count, dest_addr, 0,
				   context, ofi_op_msg);
}

ssize_t smr_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		    uint64_t flags)
{
	return smr_generic_sendmsg(ep_fid, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, msg->context, ofi_op_msg);
}

ssize_t smr_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
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
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	if (len <= SMR_CMD_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &msg_iov, 1, NULL, 0, 0);
	} else {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  &msg_iov, 1, NULL, 0, 0, peer_smr, tx_buf);
	}

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
unlock:
	fastlock_release(&peer_smr->lock);

	return ret;
}

static struct fi_ops_msg smr_msg_ops = {
	.size = sizeof(struct fi_ops_msg),
	.recv = smr_recv,
	.recvv = smr_recvv,
	.recvmsg = smr_recvmsg,
	.send = smr_send,
	.sendv = smr_sendv,
	.sendmsg = smr_sendmsg,
	.inject = smr_inject,
	.senddata = fi_no_msg_senddata,
	.injectdata = fi_no_msg_injectdata,
};

ssize_t smr_trecv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t tag, uint64_t ignore, void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_recvmsg(ep_fid, &msg_iov, 1, src_addr, tag, ignore,
				   context, FI_TAGGED);
}

ssize_t smr_trecvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t tag, uint64_t ignore,
	void *context)
{
	return smr_generic_recvmsg(ep_fid, iov, count, src_addr, tag, ignore,
				   context, FI_TAGGED);
}

ssize_t smr_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	return smr_generic_recvmsg(ep_fid, msg->msg_iov, msg->iov_count, msg->addr,
				   msg->tag, msg->ignore, msg->context,
				   flags | FI_TAGGED);
}

ssize_t smr_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
	void *desc, fi_addr_t dest_addr, uint64_t tag, void *context)
{
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	return smr_generic_sendmsg(ep_fid, &msg_iov, 1, dest_addr, tag,
				   context, ofi_op_tagged);
}

ssize_t smr_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
	void **desc, size_t count, fi_addr_t dest_addr, uint64_t tag,
	void *context)
{
	return smr_generic_sendmsg(ep_fid, iov, count, dest_addr, tag,
				   context, ofi_op_tagged);
}

ssize_t smr_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
	uint64_t flags)
{
	return smr_generic_sendmsg(ep_fid, msg->msg_iov, msg->iov_count,
				   msg->addr, msg->tag, msg->context,
				   ofi_op_tagged);
}

ssize_t smr_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
	fi_addr_t dest_addr, uint64_t tag)
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
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	if (len <= SMR_CMD_DATA_LEN) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr, &msg_iov,
				  1, NULL, ofi_op_tagged, tag);
	} else {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr, &msg_iov,
				  1, NULL, ofi_op_tagged, tag, peer_smr, tx_buf);
	}

	ofi_cirque_commit(smr_cmd_queue(peer_smr));
unlock:
	fastlock_release(&peer_smr->lock);

	return ret;
}

static struct fi_ops_tagged smr_tagged_ops = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = smr_trecv,
	.recvv = smr_trecvv,
	.recvmsg = smr_trecvmsg,
	.send = smr_tsend,
	.sendv = smr_tsendv,
	.sendmsg = smr_tsendmsg,
	.inject = smr_tinject,
	.senddata = fi_no_tagged_senddata,
	.injectdata = fi_no_tagged_injectdata,
};

ssize_t smr_rma_fast(struct smr_ep *ep, const struct iovec *iov,
	size_t iov_count, const struct fi_rma_iov *rma_iov, size_t rma_count,
	void **desc, int peer_id, void *context, uint32_t op)
{
	struct smr_region *peer_smr;
	struct iovec rma_iovec[SMR_RMA_IOV_LIMIT];
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
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
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
	smr_format_resp(cmd, peer_id, rma_iov, rma_count, total_len,
			(op == ofi_op_write) ? ofi_op_write_rsp :
			ofi_op_read_rsp);
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
	void **desc, fi_addr_t addr, void *context, uint32_t op)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	struct smr_inject_buf *tx_buf;
	struct smr_resp *resp;
	struct smr_cmd *cmd;
	int peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(iov_count <= SMR_IOV_LIMIT);
	assert(rma_count <= SMR_RMA_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);

	peer_id = (int) addr;
	ret = smr_verify_peer(ep, peer_id);
	if (ret)
		return ret;

	if (rma_count == 1)
		return smr_rma_fast(ep, iov, iov_count, rma_iov, rma_count,
				    desc, peer_id, context, op);

	peer_smr = smr_peer_region(ep->region, peer_id);
	fastlock_acquire(&peer_smr->lock);
	if (ofi_cirque_freecnt(smr_cmd_queue(peer_smr)) < 2) {
		ret = -FI_EAGAIN;
		goto unlock_region;
	}

	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto unlock_cq;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	smr_format_rma(cmd, peer_id, rma_iov, rma_count, context, op);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));
	total_len = ofi_total_iov_len(iov, iov_count);

	if (total_len <= SMR_CMD_DATA_LEN && op == ofi_op_write) {
		smr_format_inline(cmd, smr_peer_addr(ep->region)[peer_id].addr,
				  iov, iov_count, context, op, 0);
	} else if (total_len <= SMR_INJECT_SIZE && op == ofi_op_write) {
		tx_buf = smr_freestack_pop(smr_inject_pool(peer_smr));
		smr_format_inject(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
				  iov_count, context, op, 0, peer_smr, tx_buf);
	} else {
		assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		smr_format_iov(cmd, smr_peer_addr(ep->region)[peer_id].addr, iov,
			       iov_count, total_len, context, op, 0,
			       ep->region, resp);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		smr_post_pending(ep, cmd);
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
			       src_addr, context, ofi_op_read_req);
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
			       src_addr, context, ofi_op_read_req);
}

ssize_t smr_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	return smr_generic_rma(ep_fid, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_read_req);
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
			       dest_addr, context, ofi_op_write);
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
			       dest_addr, context, ofi_op_write);
}


ssize_t smr_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	return smr_generic_rma(ep_fid, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_write);
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
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
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
	smr_format_resp(cmd, peer_id, &rma_iov, 1, len,
			ofi_op_read_rsp);
	ofi_cirque_commit(smr_cmd_queue(peer_smr));

unlock_region:
	fastlock_release(&peer_smr->lock);
	return ret;
}

static struct fi_ops_rma smr_rma_ops = {
	.size = sizeof(struct fi_ops_rma),
	.read = smr_read,
	.readv = smr_readv,
	.readmsg = smr_readmsg,
	.write = smr_write,
	.writev = smr_writev,
	.writemsg = smr_writemsg,
	.inject = smr_rma_inject,
	.writedata = fi_no_rma_writedata,
	.injectdata = fi_no_rma_injectdata,
};

static int smr_ep_close(struct fid *fid)
{
	struct smr_ep *ep;

	ep = container_of(fid, struct smr_ep, util_ep.ep_fid.fid);

	ofi_endpoint_close(&ep->util_ep);

	if (ep->region)
		smr_free(ep->region);

	smr_recv_fs_free(ep->recv_fs);
	smr_unexp_fs_free(ep->unexp_fs);
	smr_pend_fs_free(ep->pend_fs);
	free(ep);
	return 0;
}

static int smr_ep_bind_cq(struct smr_ep *ep, struct util_cq *cq, uint64_t flags)
{
	int ret = 0;

	if (flags & ~(FI_TRANSMIT | FI_RECV)) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"unsupported flags\n");
		return -FI_EBADFLAGS;
	}

	if (((flags & FI_TRANSMIT) && ep->util_ep.tx_cq) ||
	    ((flags & FI_RECV) && ep->util_ep.rx_cq)) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"duplicate CQ binding\n");
		return -FI_EINVAL;
	}

	if (flags & FI_TRANSMIT) {
		ep->util_ep.tx_cq = cq;
		ofi_atomic_inc32(&cq->ref);
		ep->tx_comp = cq->wait ? smr_tx_comp_signal : smr_tx_comp;
	}

	if (flags & FI_RECV) {
		ep->util_ep.rx_cq = cq;
		ofi_atomic_inc32(&cq->ref);

		if (cq->wait) {
			ep->rx_comp = (cq->domain->info_domain_caps & FI_SOURCE) ?
				      smr_rx_src_comp_signal :
				      smr_rx_comp_signal;
		} else {
			ep->rx_comp = (cq->domain->info_domain_caps & FI_SOURCE) ?
				      smr_rx_src_comp : smr_rx_comp;
		}
	}

	ret = fid_list_insert(&cq->ep_list,
			      &cq->ep_list_lock,
			      &ep->util_ep.ep_fid.fid);

	return ret;
}

static int smr_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct smr_ep *ep;
	struct util_av *av;
	int ret = 0;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&ep->util_ep, av);
		if (ret) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"duplicate AV binding\n");
			return -FI_EINVAL;
		}
		break;
	case FI_CLASS_CQ:
		ret = smr_ep_bind_cq(ep, container_of(bfid, struct util_cq,
						      cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int smr_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct smr_attr attr;
	struct smr_ep *ep;
	struct smr_av *av;
	int ret;

	ep = container_of(fid, struct smr_ep, util_ep.ep_fid.fid);
	av = container_of(ep->util_ep.av, struct smr_av, util_av);

	switch (command) {
	case FI_ENABLE:
		if (!ep->util_ep.rx_cq || !ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!ep->util_ep.av)
			return -FI_ENOAV;

		attr.name = ep->name;
		attr.rx_count = ep->rx_size;
		attr.tx_count = ep->tx_size;
		ret = smr_create(&smr_prov, av->smr_map, &attr, &ep->region);
		if (ret)
			return ret;
		smr_exchange_all_peers(ep->region);
		break;
	default:
		return -FI_ENOSYS;
	}
	return ret;
}

static struct fi_ops smr_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = smr_ep_close,
	.bind = smr_ep_bind,
	.control = smr_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int smr_endpoint_name(char *name, char *addr, size_t addrlen,
			     int pid, int dom_idx, int ep_idx)
{
	memset(name, 0, SMR_NAME_SIZE);
	if (addr) {
		if (addrlen > SMR_NAME_SIZE)
			return -FI_EINVAL;
		snprintf(name, addrlen, "%s", addr);
	} else {
		snprintf(name, SMR_NAME_SIZE, "%d:%d:%d", pid, dom_idx, ep_idx);
	}
	return 0;
}

int smr_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct smr_ep *ep;
	struct smr_domain *smr_domain;
	int ret, ep_idx;
	char name[SMR_NAME_SIZE];

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	smr_domain = container_of(domain, struct smr_domain, util_domain.domain_fid);

	fastlock_acquire(&smr_domain->util_domain.lock);
	ep_idx = smr_domain->ep_idx++;
	fastlock_release(&smr_domain->util_domain.lock);
	ret = smr_endpoint_name(name, info->src_addr, info->src_addrlen, getpid(),
				smr_domain->dom_idx, ep_idx);
	if (ret)
		goto err2;

	ret = smr_setname(&ep->util_ep.ep_fid.fid, name, SMR_NAME_SIZE);
	if (ret)
		goto err2;

	ep->rx_size = info->rx_attr->size;
	ep->tx_size = info->tx_attr->size;
	ret = ofi_endpoint_init(domain, &smr_util_prov, info, &ep->util_ep, context,
				smr_ep_progress);
	if (ret)
		goto err1;

	ep->recv_fs = smr_recv_fs_create(info->rx_attr->size);
	ep->unexp_fs = smr_unexp_fs_create(info->rx_attr->size);
	ep->pend_fs = smr_pend_fs_create(info->tx_attr->size);
	smr_init_recv_queue(&ep->recv_queue, smr_match_msg);
	smr_init_recv_queue(&ep->trecv_queue, smr_match_tagged);
	smr_init_pending_queue(&ep->unexp_queue, smr_match_unexp);
	smr_init_pending_queue(&ep->pend_queue, smr_match_ctx);

	ep->util_ep.ep_fid.fid.ops = &smr_ep_fi_ops;
	ep->util_ep.ep_fid.ops = &smr_ep_ops;
	ep->util_ep.ep_fid.cm = &smr_cm_ops;
	ep->util_ep.ep_fid.msg = &smr_msg_ops;
	ep->util_ep.ep_fid.tagged = &smr_tagged_ops;
	ep->util_ep.ep_fid.rma = &smr_rma_ops;

	*ep_fid = &ep->util_ep.ep_fid;
	return 0;

err1:
	free((void *)ep->name);
err2:
	free(ep);
	return ret;
}
