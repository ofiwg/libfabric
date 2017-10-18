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

static void smr_tx_comp(struct smr_ep *ep, void *context)
{
	struct fi_cq_tagged_entry *comp;

	comp = ofi_cirque_tail(ep->util_ep.tx_cq->cirq);
	comp->op_context = context;
	comp->flags = FI_SEND;
	comp->len = 0;
	comp->buf = NULL;
	comp->data = 0;
	ofi_cirque_commit(ep->util_ep.tx_cq->cirq);
}

static void smr_tx_comp_signal(struct smr_ep *ep, void *context)
{
	smr_tx_comp(ep, context);
	ep->util_ep.tx_cq->wait->signal(ep->util_ep.tx_cq->wait);
}

static void smr_rx_comp(struct smr_ep *ep, void *context, uint64_t flags,
			 size_t len, void *buf, void *addr)
{
	struct fi_cq_tagged_entry *comp;

	comp = ofi_cirque_tail(ep->util_ep.rx_cq->cirq);
	comp->op_context = context;
	comp->flags = FI_RECV | flags;
	comp->len = len;
	comp->buf = buf;
	comp->data = 0;
	ofi_cirque_commit(ep->util_ep.rx_cq->cirq);
}

static void smr_rx_src_comp(struct smr_ep *ep, void *context, uint64_t flags,
			     size_t len, void *buf, void *addr)
{
	ep->util_ep.rx_cq->src[ofi_cirque_windex(ep->util_ep.rx_cq->cirq)] =
		(uint32_t) (uintptr_t) addr;
	smr_rx_comp(ep, context, flags, len, buf, addr);
}

static void smr_rx_comp_signal(struct smr_ep *ep, void *context,
			uint64_t flags, size_t len, void *buf, void *addr)
{
	smr_rx_comp(ep, context, flags, len, buf, addr);
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);
}

static void smr_rx_src_comp_signal(struct smr_ep *ep, void *context,
			uint64_t flags, size_t len, void *buf, void *addr)
{
	smr_rx_src_comp(ep, context, flags, len, buf, addr);
	ep->util_ep.rx_cq->wait->signal(ep->util_ep.rx_cq->wait);

}

void smr_progress_resp(struct smr_ep *ep)
{
	struct smr_resp *resp;

	fastlock_acquire(&ep->region->lock);
	while (!ofi_cirque_isempty(smr_resp_queue(ep->region))) {
		resp = ofi_cirque_head(smr_resp_queue(ep->region));
		if (resp->status)
			break;
		ep->tx_comp(ep, NULL);
		ofi_cirque_discard(smr_resp_queue(ep->region));
	}
	fastlock_release(&ep->region->lock);
}

static int smr_progress_inline(struct smr_cmd *cmd, struct smr_ep_entry *entry)
{
	int ret;
	ret = ofi_copy_to_iov(entry->iov, entry->iov_count, 0,
			cmd->data.msg, cmd->hdr.op.size);
	if (ret != cmd->hdr.op.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		return -FI_EIO;
	}
	return 0;
}

static int smr_progress_inject(struct smr_cmd *cmd, struct smr_ep_entry *entry,
			       struct smr_ep *ep)
{
	struct smr_inject_buf *tx_buf;
	size_t inj_offset;
	int ret;

	inj_offset = (size_t) cmd->hdr.op.data;
	tx_buf = (struct smr_inject_buf *) ((char **) ep->region +
					    inj_offset);
	ret = ofi_copy_to_iov(entry->iov, entry->iov_count, 0,
			      tx_buf, cmd->hdr.op.size);
	if (ret != cmd->hdr.op.size) {
		FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
			"recv truncated");
		ret = -FI_EIO;
	} else {
		ret = 0;
	}
	freestack_push(smr_inject_pool(ep->region), tx_buf);

	return ret;
}

static int smr_progress_iov(struct smr_cmd *cmd, struct smr_ep_entry *entry,
			    struct smr_ep *ep)
{
	struct smr_region *peer_smr;
	struct smr_resp *resp;
	int peer_id, ret;

	peer_id = *(int *)ofi_av_get_addr(ep->util_ep.av,
					  entry->addr);
	peer_smr = smr_peer_region(ep->region, peer_id);
	resp = (struct smr_resp *) ((char **) peer_smr +
				    (size_t) cmd->hdr.op.data);
	resp->msg_id = cmd->hdr.msg_id;

	ret = process_vm_readv(peer_smr->pid, entry->iov,
			       entry->iov_count, cmd->data.iov,
			       cmd->hdr.op.iov_count, 0);

	if (ret != cmd->hdr.op.size) {
		if (ret < 0) {
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"CMA write error\n");
			ret = -errno;
		} else { 
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"partial read occurred\n");
			ret = -FI_EIO;
		}
	} else {
		ret = 0;
	}

	resp->status = ret;
	return ret;
}

void smr_progress_cmd(struct smr_ep *ep)
{
	struct smr_ep_entry *entry;
	struct smr_cmd *cmd;
	fi_addr_t addr;
	int ret = 0;

	fastlock_acquire(&ep->region->lock);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	while (!ofi_cirque_isempty(smr_cmd_queue(ep->region)) &&
	       !ofi_cirque_isempty(ep->rxq)) {
		
		entry = ofi_cirque_head(ep->rxq);
		cmd = ofi_cirque_head(smr_cmd_queue(ep->region));

		switch (cmd->hdr.op.op_data){
		case smr_src_inline:
			ret = smr_progress_inline(cmd, entry);
			break;
		case smr_src_inject:
			ret = smr_progress_inject(cmd, entry, ep);
			break;
		case smr_src_iov:
			ret = smr_progress_iov(cmd, entry, ep);
			break;
		default:
			FI_WARN(&smr_prov, FI_LOG_EP_CTRL,
				"unidentified operation type\n");
			ret = -FI_EINVAL;
		}
		ep->rx_comp(ep, entry->context, 0, ret, NULL, &addr);
		ofi_cirque_discard(ep->rxq);
		ofi_cirque_discard(smr_cmd_queue(ep->region));
	}
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	fastlock_release(&ep->region->lock);
}

void smr_ep_progress(struct util_ep *util_ep)
{
	struct smr_ep *ep;

	ep = container_of(util_ep, struct smr_ep, util_ep);

	smr_progress_resp(ep);
	smr_progress_cmd(ep);
}

ssize_t smr_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct smr_ep *ep;
	struct smr_ep_entry *entry;
	ssize_t ret;

	assert(msg->iov_count < SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->rxq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry = ofi_cirque_tail(ep->rxq);
	entry->context = msg->context;
	for (entry->iov_count = 0; entry->iov_count < msg->iov_count;
	     entry->iov_count++) {
		entry->iov[entry->iov_count] = msg->msg_iov[entry->iov_count];
	}
	entry->flags = 0;

	ofi_cirque_commit(ep->rxq);
	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

ssize_t smr_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t src_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.iov_count = count;
	msg.context = context;
	return smr_recvmsg(ep_fid, &msg, 0);
}

ssize_t smr_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
		fi_addr_t src_addr, void *context)
{
	struct smr_ep *ep;
	struct smr_ep_entry *entry;
	ssize_t ret;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->rxq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	entry = ofi_cirque_tail(ep->rxq);
	entry->context = context;
	entry->iov_count = 1;
	entry->iov[0].iov_base = buf;
	entry->iov[0].iov_len = len;
	entry->flags = 0;
	entry->addr = src_addr;

	ofi_cirque_commit(ep->rxq);
	ret = 0;
out:
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	return ret;
}

static void smr_format_send(struct smr_cmd *cmd, const struct fi_msg *msg)
{
	cmd->hdr.op.version = OFI_OP_VERSION;
	cmd->hdr.op.rx_index = 0;
	cmd->hdr.op.op = ofi_op_msg;
	cmd->hdr.op.op_data = smr_src_inline;
	cmd->hdr.op.flags = 0;

	cmd->hdr.op.data = 0;
	cmd->hdr.op.resv = 0;

	cmd->hdr.op.size = ofi_copy_from_iov(cmd->data.msg, SMR_CMD_DATA_LEN,
					     msg->msg_iov, msg->iov_count, 0);
}

static void smr_format_inject(struct smr_cmd *cmd, const struct fi_msg *msg,
			      struct smr_region *smr, size_t inj_offset)
{
	cmd->hdr.op.version = OFI_OP_VERSION;
	cmd->hdr.op.rx_index = 0;
	cmd->hdr.op.op = ofi_op_msg;
	cmd->hdr.op.op_data = smr_src_inject;
	cmd->hdr.op.flags = 0;

	cmd->hdr.op.size = 0;
	cmd->hdr.op.resv = 0;
	cmd->hdr.op.data = inj_offset;

	cmd->hdr.op.size = ofi_copy_from_iov((char **) smr + inj_offset, SMR_INJECT_SIZE,
					     msg->msg_iov, msg->iov_count, 0);
}

static void smr_format_cma(struct smr_cmd *cmd, const struct fi_msg *msg,
			   size_t total_len, struct smr_region *smr,
			   struct smr_resp *resp)
{
	int i;

	cmd->hdr.op.version = OFI_OP_VERSION;
	cmd->hdr.op.rx_index = 0;
	cmd->hdr.op.op = ofi_op_msg;
	cmd->hdr.op.op_data = smr_src_iov;
	cmd->hdr.op.flags = 0;

	cmd->hdr.op.size = 0;
	cmd->hdr.op.resv = 0;
	resp->status = -FI_EBUSY;
	cmd->hdr.op.data = (uint64_t) ((char **) resp - (char **) smr);
	cmd->hdr.op.iov_count = msg->iov_count;

	for (i = 0; i < msg->iov_count; i++) {
		cmd->data.iov[i].iov_base = msg->msg_iov[i].iov_base;
		cmd->data.iov[i].iov_len = msg->msg_iov[i].iov_len;
		cmd->hdr.op.size += msg->msg_iov[i].iov_len;
	}
}

#define peer_is_mapped(addr) (addr.name[0] == '\0')

ssize_t smr_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
		uint64_t flags)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	size_t inj_offset;
	struct smr_resp *resp;
	struct smr_cmd *cmd;
	int peer_id;
	ssize_t ret = 0;
	size_t total_len;

	assert(msg->iov_count < SMR_IOV_LIMIT);

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	peer_id = *(int *)ofi_av_get_addr(ep->util_ep.av, msg->addr);
	fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
	if (ofi_cirque_isfull(ep->util_ep.tx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	peer_smr = smr_peer_region(ep->region, peer_id);
	if (!peer_is_mapped(ep->region->map->peer_addr[peer_id])) {
		ret = smr_map_to_region(&smr_prov, &peer_smr,
					ep->region->map->peer_addr[peer_id].name);
		if (ret) {
			if (ret == -ENOENT)
				ret = -FI_EAGAIN; 
			goto out;
		}
	}

	fastlock_acquire(&peer_smr->lock);
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	total_len = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	cmd->hdr.msg_id = (ep->msg_id)++;
	if (total_len <= SMR_CMD_DATA_LEN) {
		smr_format_send(cmd, msg);
	} else if (total_len <= SMR_INJECT_SIZE) {
		inj_offset = freestack_shm_pop(smr_inject_pool(peer_smr),
					       peer_smr->inject_pool_offset);
		smr_format_inject(cmd, msg, peer_smr, inj_offset);
	} else {
		assert(!ofi_cirque_isfull(smr_resp_queue(ep->region)));
		resp = ofi_cirque_tail(smr_resp_queue(ep->region));
		smr_format_cma(cmd, msg, total_len, ep->region, resp);
		ofi_cirque_commit(smr_resp_queue(ep->region));
		goto commit;
	}
	ep->tx_comp(ep, msg->context);

commit:
	ofi_cirque_commit(smr_cmd_queue(peer_smr));
unlock:
	fastlock_release(&peer_smr->lock);
out:
	fastlock_release(&ep->util_ep.tx_cq->cq_lock);
	return ret;
}

ssize_t smr_send(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;
	struct iovec msg_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.context = context;

	return smr_sendmsg(ep_fid, &msg, 0);
}


ssize_t smr_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, void *context)
{
	struct fi_msg msg;

	msg.msg_iov = iov;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.context = context;

	return smr_sendmsg(ep_fid, &msg, 0);
}

ssize_t smr_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
		fi_addr_t dest_addr)
{
	struct smr_ep *ep;
	struct smr_region *peer_smr;
	size_t inj_offset;
	struct smr_cmd *cmd;
	int peer_id;
	ssize_t ret = 0;
	struct fi_msg msg;
	struct iovec msg_iov;

	assert(len < SMR_INJECT_SIZE);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;

	ep = container_of(ep_fid, struct smr_ep, util_ep.ep_fid.fid);
	peer_id = *(int *)ofi_av_get_addr(ep->util_ep.av, dest_addr);

	peer_smr = smr_peer_region(ep->region, peer_id);
	if (!peer_is_mapped(ep->region->map->peer_addr[peer_id])) {
		ret = smr_map_to_region(&smr_prov, &peer_smr,
					ep->region->map->peer_addr[peer_id].name);
		if (ret) {
			if (ret == -ENOENT)
				ret = -FI_EAGAIN; 
			return ret;
		}
	}

	fastlock_acquire(&peer_smr->lock);
	if (ofi_cirque_isfull(smr_cmd_queue(peer_smr))) {
		ret = -FI_EAGAIN;
		goto unlock;
	}

	cmd = ofi_cirque_tail(smr_cmd_queue(peer_smr));

	cmd->hdr.msg_id = (ep->msg_id)++;
	if (len <= SMR_CMD_DATA_LEN) {
		smr_format_send(cmd, &msg);
	} else {
		inj_offset = freestack_shm_pop(smr_inject_pool(peer_smr),
					       peer_smr->inject_pool_offset);
		smr_format_inject(cmd, &msg, peer_smr, inj_offset);
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

static int smr_ep_close(struct fid *fid)
{
	struct smr_ep *ep;

	ep = container_of(fid, struct smr_ep, util_ep.ep_fid.fid);

	ofi_endpoint_close(&ep->util_ep);

	if (ep->region)
		smr_free(ep->region);

	smr_rx_cirq_free(ep->rxq);
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
		attr.rx_count = ep->rxq->size;
		attr.tx_count = ep->tx_size;
		ret = smr_create(&smr_prov, av->smr_map, &attr, &ep->region);
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

int smr_endpoint(struct fid_domain *domain, struct fi_info *info,
		  struct fid_ep **ep_fid, void *context)
{
	struct smr_ep *ep;
	int ret;

	ep = calloc(1, sizeof(*ep));
	if (!ep)
		return -FI_ENOMEM;

	if (info->src_addr && info->src_addrlen) {
		ret = smr_setname(&ep->util_ep.ep_fid.fid, info->src_addr,
				  info->src_addrlen);
		if (ret)
			goto err;
	}

	ep->tx_size = info->tx_attr->size;
	ep->rxq = smr_rx_cirq_create(info->rx_attr->size);
	if (!ep->rxq) {
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = ofi_endpoint_init(domain, &smr_util_prov, info, &ep->util_ep, context,
				smr_ep_progress);
	if (ret)
		goto err;

	ep->util_ep.ep_fid.fid.ops = &smr_ep_fi_ops;
	ep->util_ep.ep_fid.ops = &smr_ep_ops;
	ep->util_ep.ep_fid.cm = &smr_cm_ops;
	ep->util_ep.ep_fid.msg = &smr_msg_ops;

	*ep_fid = &ep->util_ep.ep_fid;
	return 0;
err:
	free((void *)ep->name);
	free(ep);
	return ret;
}
