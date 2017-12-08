/*
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
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

#include <fi_iov.h>
#include "rxm.h"

typedef ssize_t rxm_rma_msg_fn(struct fid_ep *ep_fid,
			       const struct fi_msg_rma *msg, uint64_t flags);

static int rxm_ep_rma_common(struct fid_ep *msg_ep, struct rxm_ep *rxm_ep,
			     const struct fi_msg_rma *msg, uint64_t flags,
			     rxm_rma_msg_fn rma_msg, uint64_t comp_flags)
{
	struct rxm_domain *rxm_domain;
	struct rxm_tx_entry *tx_entry;
	struct fi_msg_rma msg_rma;
	size_t i;
	int ret;

	if (!(tx_entry = rxm_tx_entry_get(&rxm_ep->send_queue)))
		return -FI_EAGAIN;

	memset(tx_entry, 0, sizeof(*tx_entry));
	tx_entry->state = RXM_TX_NOBUF;
	tx_entry->ep = rxm_ep;
	tx_entry->context = msg->context;
	tx_entry->flags = flags;
	tx_entry->comp_flags = FI_RMA | comp_flags;

	msg_rma = *msg;
	msg_rma.context = tx_entry;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain,
			  util_domain);
	if (rxm_domain->mr_local) {
		ret = rxm_ep_msg_mr_regv(rxm_ep, msg->msg_iov,
					 msg->iov_count,
					 comp_flags & (FI_WRITE | FI_READ),
					 tx_entry->mr);
		if (ret)
			goto err;
		for (i = 0; i < msg_rma.iov_count; i++)
			msg_rma.desc[i] = fi_mr_desc(tx_entry->mr[i]);
	} else {
		/* msg_rma.desc is msg fid_mr * array */
		for (i = 0; i < msg_rma.iov_count; i++)
			msg_rma.desc[i] = fi_mr_desc(msg_rma.desc[i]);
	}
	return rma_msg(msg_ep, &msg_rma, flags);
err:
	rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	return ret;
}

ssize_t	rxm_ep_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
		       uint64_t flags)
{
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);

	ret = ofi_cmap_get_handle(rxm_ep->util_ep.cmap, msg->addr, &handle);
	if (ret)
		return ret;
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	return rxm_ep_rma_common(rxm_conn->msg_ep, rxm_ep, msg, flags,
				 fi_readmsg, FI_READ);
}

static ssize_t rxm_ep_read(struct fid_ep *ep_fid, void *buf, size_t len,
			   void *desc, fi_addr_t src_addr, uint64_t addr,
			   uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = src_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = context;
	msg.data = 0;

	return rxm_ep_readmsg(ep_fid, &msg, rxm_ep_tx_flags(ep_fid));
}

static ssize_t rxm_ep_readv(struct fid_ep *ep_fid, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t src_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = src_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = context;
	msg.data = 0;

	return rxm_ep_readmsg(ep_fid, &msg, rxm_ep_tx_flags(ep_fid));
}

static int rxm_ep_rma_inject(struct fid_ep *msg_ep, struct rxm_ep *rxm_ep,
			     const struct fi_msg_rma *msg, uint64_t flags)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;
	struct fi_msg_rma msg_rma;
	struct iovec iov;
	size_t size;
	int ret;

	size = ofi_total_iov_len(msg->msg_iov, msg->iov_count);

	if (size > rxm_ep->rxm_info->tx_attr->inject_size)
		return -FI_EMSGSIZE;

	/* Use fi_inject_write instead of fi_writemsg since the latter generates
	 * completion by default */
	if (size <= rxm_ep->msg_info->tx_attr->inject_size &&
	    !(flags & FI_COMPLETION)) {
		if (flags & FI_REMOTE_CQ_DATA)
			return fi_inject_writedata(msg_ep, msg->msg_iov->iov_base,
					       msg->msg_iov->iov_len, msg->data,
					       msg->addr, msg->rma_iov->addr,
					       msg->rma_iov->key);
		else
			return fi_inject_write(msg_ep, msg->msg_iov->iov_base,
					       msg->msg_iov->iov_len, msg->addr,
					       msg->rma_iov->addr,
					       msg->rma_iov->key);
	}

	tx_buf = (struct rxm_tx_buf *)rxm_buf_get(&rxm_ep->tx_pool);
	if (!tx_buf) {
		FI_WARN(&rxm_prov, FI_LOG_CQ, "TX queue full!\n");
		rxm_cq_progress(rxm_ep);
		return -FI_EAGAIN;
	}

	if (!(tx_entry = rxm_tx_entry_get(&rxm_ep->send_queue))) {
		rxm_cq_progress(rxm_ep);
		ret = -FI_EAGAIN;
		goto err1;
	}

	memset(tx_entry, 0, sizeof(*tx_entry));
	tx_entry->state = RXM_TX;
	tx_entry->ep = rxm_ep;
	tx_entry->flags = flags;
	tx_entry->comp_flags = FI_RMA | FI_WRITE;
	tx_entry->tx_buf = tx_buf;

	tx_buf->hdr.msg_ep = msg_ep;
	ofi_copy_from_iov(tx_buf->pkt.data, size, msg->msg_iov,
			  msg->iov_count, 0);

	iov.iov_base = &tx_buf->pkt.data;
	iov.iov_len = size;

	msg_rma.msg_iov = &iov;
	msg_rma.desc = &tx_buf->hdr.desc;
	msg_rma.iov_count = 1;
	msg_rma.addr = msg->addr;
	msg_rma.rma_iov = msg->rma_iov;
	msg_rma.rma_iov_count = msg->rma_iov_count;
	msg_rma.context = tx_entry;
	msg_rma.data = msg->data;
	flags = (flags & ~FI_INJECT) | FI_COMPLETION;

	ret = fi_writemsg(msg_ep, &msg_rma, flags);
	if (ret) {
		if (ret == -FI_EAGAIN)
			rxm_cq_progress(rxm_ep);
		goto err2;
	}
	return 0;
err2:
	rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
err1:
	rxm_buf_release(&rxm_ep->tx_pool, (struct rxm_buf *)tx_buf);
	return ret;
}

ssize_t	rxm_ep_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
			uint64_t flags)
{
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);

	ret = ofi_cmap_get_handle(rxm_ep->util_ep.cmap, msg->addr, &handle);
	if (ret)
		return ret;
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	if (flags & FI_INJECT)
		return rxm_ep_rma_inject(rxm_conn->msg_ep, rxm_ep, msg, flags);
	else
		return rxm_ep_rma_common(rxm_conn->msg_ep, rxm_ep, msg, flags,
					 fi_writemsg, FI_WRITE);
}

static ssize_t rxm_ep_write(struct fid_ep *ep_fid, const void *buf,
			    size_t len, void *desc, fi_addr_t dest_addr,
			    uint64_t addr, uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = context;
	msg.data = 0;

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(ep_fid));
}

static ssize_t rxm_ep_writev(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t addr, uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;

	rma_iov.addr = addr;
	rma_iov.len = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	msg.msg_iov = iov;
	msg.desc = desc;
	msg.iov_count = count;
	msg.addr = dest_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = context;
	msg.data = 0;

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(ep_fid));
}

static ssize_t rxm_ep_writedata(struct fid_ep *ep_fid, const void *buf,
				size_t len, void *desc, uint64_t data,
				fi_addr_t dest_addr, uint64_t addr,
				uint64_t key, void *context)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	msg.msg_iov = &iov;
	msg.desc = &desc;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = context;
	msg.data = data;

	return rxm_ep_writemsg(ep_fid, &msg, rxm_ep_tx_flags(ep_fid) |
			       FI_REMOTE_CQ_DATA);
}

static ssize_t rxm_ep_inject_write(struct fid_ep *ep_fid, const void *buf,
			     size_t len, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	msg.msg_iov = &iov;
	msg.desc = NULL;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = NULL;
	msg.data = 0;

	return rxm_ep_writemsg(ep_fid, &msg,
			       (rxm_ep_tx_flags(ep_fid) & ~FI_COMPLETION) |
			       FI_INJECT);
}

static ssize_t rxm_ep_inject_writedata(struct fid_ep *ep_fid, const void *buf,
				 size_t len, uint64_t data,
				 fi_addr_t dest_addr, uint64_t addr,
					uint64_t key)
{
	struct fi_msg_rma msg;
	struct fi_rma_iov rma_iov;
	struct iovec iov;

	iov.iov_base = (void *)buf;
	iov.iov_len = len;

	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	msg.msg_iov = &iov;
	msg.desc = NULL;
	msg.iov_count = 1;
	msg.addr = dest_addr;
	msg.rma_iov = &rma_iov;
	msg.rma_iov_count = 1;
	msg.context = NULL;
	msg.data = data;

	return rxm_ep_writemsg(ep_fid, &msg,
			       (rxm_ep_tx_flags(ep_fid) & ~FI_COMPLETION) |
			       FI_INJECT | FI_REMOTE_CQ_DATA);
}

struct fi_ops_rma rxm_ops_rma = {
	.size = sizeof (struct fi_ops_rma),
	.read = rxm_ep_read,
	.readv = rxm_ep_readv,
	.readmsg = rxm_ep_readmsg,
	.write = rxm_ep_write,
	.writev = rxm_ep_writev,
	.writemsg = rxm_ep_writemsg,
	.inject = rxm_ep_inject_write,
	.writedata = rxm_ep_writedata,
	.injectdata = rxm_ep_inject_writedata,
};
