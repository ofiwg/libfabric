/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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
#include <ofi_mem.h>
#include "rxd.h"

ssize_t	rxd_ep_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
		       uint64_t flags)
{
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	uint64_t peer_addr;
	ssize_t ret;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid);

	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	fastlock_acquire(&rxd_ep->lock);
	if (peer->state != CMAP_CONNECTED) {
		ret = rxd_ep_connect(rxd_ep, peer, peer_addr);
		fastlock_release(&rxd_ep->lock);
		if (ret == -FI_EALREADY) {
			rxd_ep->util_ep.progress(&rxd_ep->util_ep);
			ret = -FI_EAGAIN;
		}
		return ret ? ret : -FI_EAGAIN;
	}

	tx_entry = rxd_tx_entry_alloc(rxd_ep, peer, peer_addr, flags,
				      RXD_TX_READ_REQ);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry->read_req.msg = *msg;
	memcpy(&tx_entry->read_req.dst_iov[0], msg->msg_iov,
	       sizeof(*msg->msg_iov)* msg->iov_count);
	memcpy(&tx_entry->read_req.src_iov[0], msg->rma_iov,
	       sizeof(*msg->rma_iov) * msg->rma_iov_count);
	ret = rxd_ep_start_xfer(rxd_ep, peer, ofi_op_read_req, tx_entry);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->lock);
	return ret;
}

static ssize_t rxd_ep_read(struct fid_ep *ep, void *buf, size_t len,
				 void *desc, fi_addr_t src_addr, uint64_t addr,
				 uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;
	msg.rma_iov_count = 1;
	msg.rma_iov = &rma_iov;

	msg.addr = src_addr;
	msg.context = context;

	return rxd_ep_readmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_readv(struct fid_ep *ep, const struct iovec *iov,
				void **desc, size_t count,
				fi_addr_t src_addr, uint64_t addr, uint64_t key,
				void *context)
{
	size_t len, i;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	assert(count <= RXD_IOV_LIMIT);
	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.rma_iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;

	for (i = 0, len = 0; i < count; i++)
		len += iov[i].iov_len;
	rma_iov.len = len;

	msg.rma_iov = &rma_iov;
	msg.addr = src_addr;
	msg.context = context;

	return rxd_ep_readmsg(ep, &msg, RXD_USE_OP_FLAGS);
}

ssize_t	rxd_ep_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct rxd_ep *rxd_ep;
	struct rxd_peer *peer;
	struct rxd_tx_entry *tx_entry;
	uint64_t peer_addr;
	ssize_t ret;

	rxd_ep = container_of(ep, struct rxd_ep, util_ep.ep_fid);
	peer_addr = rxd_av_dg_addr(rxd_ep_av(rxd_ep), msg->addr);
	peer = rxd_ep_getpeer_info(rxd_ep, peer_addr);

	fastlock_acquire(&rxd_ep->lock);
	if (peer->state != CMAP_CONNECTED) {
		ret = rxd_ep_connect(rxd_ep, peer, peer_addr);
		fastlock_release(&rxd_ep->lock);
		if (ret == -FI_EALREADY) {
			rxd_ep->util_ep.progress(&rxd_ep->util_ep);
			ret = -FI_EAGAIN;
		}
		return ret ? ret : -FI_EAGAIN;
	}

	tx_entry = rxd_tx_entry_alloc(rxd_ep, peer, peer_addr, flags,
				      RXD_TX_WRITE);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	tx_entry->write.msg = *msg;
	memcpy(&tx_entry->write.src_iov[0], msg->msg_iov,
	       sizeof(*msg->msg_iov) * msg->iov_count);
	memcpy(&tx_entry->write.dst_iov[0], msg->rma_iov,
	       sizeof(*msg->rma_iov) * msg->rma_iov_count);
	ret = rxd_ep_start_xfer(rxd_ep, peer, ofi_op_write, tx_entry);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->lock);
	return ret;
}

static ssize_t rxd_ep_write(struct fid_ep *ep, const void *buf,
			    size_t len, void *desc, fi_addr_t dest_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;

	msg.msg_iov = &msg_iov;
	msg.desc = &desc;
	msg.iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;

	msg.rma_iov_count = 1;
	msg.rma_iov = &rma_iov;

	msg.addr = dest_addr;
	msg.context = context;

	return rxd_ep_writemsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_writev(struct fid_ep *ep, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	size_t i;
	size_t len;
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	assert(count <= RXD_IOV_LIMIT);
	memset(&msg, 0, sizeof(msg));
	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.rma_iov_count = 1;

	for (i = 0, len = 0; i < count; i++)
		len += iov[i].iov_len;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;

	msg.rma_iov = &rma_iov;
	msg.context = context;
	msg.addr = dest_addr;

	return rxd_ep_writemsg(ep, &msg, RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_writedata(struct fid_ep *ep, const void *buf,
				size_t len, void *desc, uint64_t data,
				fi_addr_t dest_addr, uint64_t addr,
				uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.rma_iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;

	msg.rma_iov = &rma_iov;
	msg.msg_iov = &msg_iov;

	msg.addr = dest_addr;
	msg.context = context;
	msg.data = data;

	return rxd_ep_writemsg(ep, &msg, FI_REMOTE_CQ_DATA |
					RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_inject(struct fid_ep *ep, const void *buf,
			     size_t len, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.rma_iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;

	msg.rma_iov = &rma_iov;
	msg.msg_iov = &msg_iov;
	msg.addr = dest_addr;

	return rxd_ep_writemsg(ep, &msg, FI_INJECT |
				    RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
}

static ssize_t rxd_ep_injectdata(struct fid_ep *ep, const void *buf,
				 size_t len, uint64_t data,
				 fi_addr_t dest_addr, uint64_t addr,
					uint64_t key)
{
	struct fi_msg_rma msg;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	memset(&msg, 0, sizeof(msg));
	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	msg.msg_iov = &msg_iov;
	msg.iov_count = 1;
	msg.rma_iov_count = 1;

	rma_iov.addr = addr;
	rma_iov.key = key;
	rma_iov.len = len;

	msg.rma_iov = &rma_iov;
	msg.msg_iov = &msg_iov;
	msg.addr = dest_addr;
	msg.data = data;
	return rxd_ep_writemsg(ep, &msg, FI_INJECT | FI_REMOTE_CQ_DATA |
		RXD_NO_COMPLETION | RXD_USE_OP_FLAGS);
}

struct fi_ops_rma rxd_ops_rma = {
	.size = sizeof (struct fi_ops_rma),
	.read = rxd_ep_read,
	.readv = rxd_ep_readv,
	.readmsg = rxd_ep_readmsg,
	.write = rxd_ep_write,
	.writev = rxd_ep_writev,
	.writemsg = rxd_ep_writemsg,
	.inject = rxd_ep_inject,
	.writedata = rxd_ep_writedata,
	.injectdata = rxd_ep_injectdata,
};
