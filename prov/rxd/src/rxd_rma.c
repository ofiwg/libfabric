/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
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
#include <ofi_iov.h>
#include "rxd.h"

static ssize_t rxd_generic_write_inject(struct rxd_ep *rxd_ep,
		const struct iovec *iov, size_t iov_count,
		const struct fi_rma_iov *rma_iov, size_t rma_count,
		fi_addr_t addr, void *context, uint32_t op, uint64_t data,
		uint32_t rxd_flags)
{
	struct rxd_x_entry *tx_entry;
	fi_addr_t rxd_addr;
	ssize_t ret = -FI_EAGAIN;

	assert(iov_count <= RXD_IOV_LIMIT && rma_count <= RXD_IOV_LIMIT);
	assert(ofi_total_iov_len(iov, iov_count) <= rxd_ep_domain(rxd_ep)->max_inline_rma);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.tx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.tx_cq->cirq))
		goto out;

	rxd_addr = rxd_ep_av(rxd_ep)->fi_addr_table[addr];
	ret = rxd_send_rts_if_needed(rxd_ep, rxd_addr);
	if (ret)
		goto out;

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, NULL, 0, rma_count, data,
				     0, context, rxd_addr, op, rxd_flags);
	if (!tx_entry)
		goto out;

	ret = rxd_ep_send_op(rxd_ep, tx_entry, rma_iov, rma_count, NULL, 0, 0, 0);
	if (ret) {
		rxd_tx_entry_free(rxd_ep, tx_entry);
		goto out;
	}

	if (tx_entry->op == RXD_READ_REQ)
		goto out;

	ret = 0;

out:
	fastlock_release(&rxd_ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

ssize_t rxd_generic_rma(struct rxd_ep *rxd_ep, const struct iovec *iov,
	size_t iov_count, const struct fi_rma_iov *rma_iov, size_t rma_count,
	void **desc, fi_addr_t addr, void *context, uint32_t op, uint64_t data,
	uint32_t rxd_flags)
{
	struct rxd_x_entry *tx_entry;
	fi_addr_t rxd_addr;
	ssize_t ret = -FI_EAGAIN;

	if (rxd_flags & RXD_INJECT)
		return rxd_generic_write_inject(rxd_ep, iov, iov_count, rma_iov,
						rma_count, addr, context, op,
						data, rxd_flags);

	assert(iov_count <= RXD_IOV_LIMIT && rma_count <= RXD_IOV_LIMIT);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.tx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.tx_cq->cirq))
		goto out;

	rxd_addr = rxd_ep_av(rxd_ep)->fi_addr_table[addr];
	ret = rxd_send_rts_if_needed(rxd_ep, rxd_addr);
	if (ret)
		goto out;

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, NULL, 0, rma_count,
				     data, 0, context, rxd_addr, op, rxd_flags);
	if (!tx_entry)
		goto out;

	ret = rxd_ep_send_op(rxd_ep, tx_entry, rma_iov, rma_count, NULL, 0, 0, 0);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

ssize_t rxd_read(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return rxd_generic_rma(ep, &msg_iov, 1, &rma_iov, 1, &desc, 
			       src_addr, context, ofi_op_read_req, 0,
			       rxd_ep_tx_flags(ep));
}

ssize_t rxd_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
	void *context)
{
	struct rxd_ep *ep;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return rxd_generic_rma(ep, iov, count, &rma_iov, 1, desc,
			       src_addr, context, ofi_op_read_req, 0,
			       rxd_ep_tx_flags(ep));
}

ssize_t rxd_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_generic_rma(ep, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_read_req, msg->data, rxd_flags(flags));
}

ssize_t rxd_write(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return rxd_generic_rma(ep, &msg_iov, 1, &rma_iov, 1, &desc, 
			       dest_addr, context, ofi_op_write, 0,
			       rxd_ep_tx_flags(ep));
}

ssize_t rxd_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		void *context)
{
	struct rxd_ep *ep;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return rxd_generic_rma(ep, iov, count, &rma_iov, 1, desc,
			       dest_addr, context, ofi_op_write, 0,
			       rxd_ep_tx_flags(ep));
}


ssize_t rxd_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_generic_rma(ep, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       ofi_op_write, msg->data, rxd_flags(flags));
}

ssize_t rxd_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	return rxd_generic_rma(ep, &iov, 1, &rma_iov, 1, &desc,
			       dest_addr, context, ofi_op_write, data,
			       rxd_ep_tx_flags(ep) | RXD_REMOTE_CQ_DATA);
}

ssize_t rxd_inject_write(struct fid_ep *ep_fid, const void *buf,
	size_t len, fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct rxd_ep *rxd_ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	rxd_ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return rxd_generic_write_inject(rxd_ep, &iov, 1, &rma_iov, 1,
					dest_addr, NULL, ofi_op_write, 0,
					RXD_NO_TX_COMP | RXD_INJECT);
}

ssize_t rxd_inject_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
			     uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			     uint64_t key)
{
	struct rxd_ep *rxd_ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	rxd_ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return rxd_generic_write_inject(rxd_ep, &iov, 1, &rma_iov, 1,
					dest_addr, NULL, ofi_op_write,
					data, RXD_NO_TX_COMP | RXD_INJECT |
					RXD_REMOTE_CQ_DATA);
}

struct fi_ops_rma rxd_ops_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = rxd_read,
	.readv = rxd_readv,
	.readmsg = rxd_readmsg,
	.write = rxd_write,
	.writev = rxd_writev,
	.writemsg = rxd_writemsg,
	.inject = rxd_inject_write,
	.writedata = rxd_writedata,
	.injectdata = rxd_inject_writedata,

};
