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
#include "uet.h"

static struct uet_x_entry *uet_tx_entry_init_rma(struct uet_ep *ep, fi_addr_t addr,
				uint32_t op, const struct iovec *iov, size_t iov_count,
				uint64_t data, uint32_t flags, void *context,
				const struct fi_rma_iov *rma_iov, size_t rma_count)
{
	struct uet_x_entry *tx_entry;
	struct uet_domain *uet_domain = uet_ep_domain(ep);
	size_t max_inline;
	struct uet_base_hdr *base_hdr;
	void *ptr;

	tx_entry = uet_tx_entry_init_common(ep, addr, op, iov, iov_count, 0,
					    data, flags, context, &base_hdr, &ptr);
	if (!tx_entry)
		return NULL;

	if (tx_entry->cq_entry.flags & FI_READ) {
		tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len,
						  uet_domain->max_seg_sz);
		uet_init_sar_hdr(&ptr, tx_entry, rma_count);
		uet_init_rma_hdr(&ptr, rma_iov, rma_count);
	} else {
		max_inline = uet_domain->max_inline_msg;
		max_inline -= sizeof(struct ofi_rma_iov) * rma_count;
		uet_check_init_cq_data(&ptr, tx_entry, &max_inline);

		if (rma_count > 1 || tx_entry->cq_entry.len > max_inline) {
			max_inline -= sizeof(struct uet_sar_hdr);
			tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len -
				max_inline, uet_domain->max_seg_sz) + 1;
			uet_init_sar_hdr(&ptr, tx_entry, rma_count);
		} else {
			tx_entry->flags |= UET_INLINE;
			base_hdr->flags = (uint16_t) tx_entry->flags;
			tx_entry->num_segs = 1;
		}
		uet_init_rma_hdr(&ptr, rma_iov, rma_count);
		tx_entry->bytes_done = uet_init_msg(&ptr, tx_entry->iov,
			tx_entry->iov_count, tx_entry->cq_entry.len,
			max_inline);
	}

	tx_entry->pkt->pkt_size = uet_pkt_size(ep, base_hdr, ptr);

	return tx_entry;
}

static ssize_t uet_generic_write_inject(struct uet_ep *uet_ep,
		const struct iovec *iov, size_t iov_count,
		const struct fi_rma_iov *rma_iov, size_t rma_count,
		fi_addr_t addr, void *context, uint32_t op, uint64_t data,
		uint32_t uet_flags)
{
	struct uet_x_entry *tx_entry;
	fi_addr_t uet_addr;
	ssize_t ret = -FI_EAGAIN;

	assert(iov_count <= UET_IOV_LIMIT && rma_count <= UET_IOV_LIMIT);
	assert(ofi_total_iov_len(iov, iov_count) <= (size_t) uet_ep_domain(uet_ep)->max_inline_rma);

	ofi_genlock_lock(&uet_ep->util_ep.lock);

	if (ofi_cirque_isfull(uet_ep->util_ep.tx_cq->cirq))
		goto out;

	uet_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(uet_ep)->fi_addr_idx),
					      UET_IDX_OFFSET((int) addr));
	if (!uet_addr)
		goto out;
	ret = uet_send_rts_if_needed(uet_ep, uet_addr);
	if (ret)
		goto out;

	tx_entry = uet_tx_entry_init_rma(uet_ep, uet_addr, op, iov, iov_count,
					 data, uet_flags, context, rma_iov,
					 rma_count);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (uet_peer(uet_ep, uet_addr)->peer_addr == UET_ADDR_INVALID)
		goto out;

	ret = uet_start_xfer(uet_ep, tx_entry);
	if (ret && tx_entry->num_segs > 1)
		(void) uet_ep_post_data_pkts(uet_ep, tx_entry);
	ret = 0;

out:
	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	return ret;
}

static ssize_t
uet_generic_rma(struct uet_ep *uet_ep, const struct iovec *iov,
	size_t iov_count, const struct fi_rma_iov *rma_iov, size_t rma_count,
	void **desc, fi_addr_t addr, void *context, uint32_t op, uint64_t data,
	uint32_t uet_flags)
{
	struct uet_x_entry *tx_entry;
	fi_addr_t uet_addr;
	ssize_t ret = -FI_EAGAIN;

	if (uet_flags & UET_INJECT)
		return uet_generic_write_inject(uet_ep, iov, iov_count, rma_iov,
						rma_count, addr, context, op,
						data, uet_flags);

	assert(iov_count <= UET_IOV_LIMIT && rma_count <= UET_IOV_LIMIT);

	ofi_genlock_lock(&uet_ep->util_ep.lock);

	if (ofi_cirque_isfull(uet_ep->util_ep.tx_cq->cirq))
		goto out;
	uet_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(uet_ep)->fi_addr_idx),
					     UET_IDX_OFFSET((int) addr));
	if (!uet_addr)
		goto out;

	ret = uet_send_rts_if_needed(uet_ep, uet_addr);
	if (ret)
		goto out;

	tx_entry = uet_tx_entry_init_rma(uet_ep, uet_addr, op, iov, iov_count,
					 data, uet_flags, context, rma_iov,
					 rma_count);
	if (!tx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (uet_peer(uet_ep, uet_addr)->peer_addr == UET_ADDR_INVALID)
		goto out;

	ret = uet_start_xfer(uet_ep, tx_entry);
	if (ret && (tx_entry->cq_entry.flags & FI_WRITE) && tx_entry->num_segs > 1)
		(void) uet_ep_post_data_pkts(uet_ep, tx_entry);
	ret = 0;

out:
	ofi_genlock_unlock(&uet_ep->util_ep.lock);
	return ret;
}

static ssize_t
uet_read(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
	fi_addr_t src_addr, uint64_t addr, uint64_t key, void *context)
{
	struct uet_ep *ep;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return uet_generic_rma(ep, &msg_iov, 1, &rma_iov, 1, &desc,
			       src_addr, context, UET_READ_REQ, 0,
			       ep->tx_flags);
}

static ssize_t
uet_readv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
	size_t count, fi_addr_t src_addr, uint64_t addr, uint64_t key,
	void *context)
{
	struct uet_ep *ep;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return uet_generic_rma(ep, iov, count, &rma_iov, 1, desc,
			       src_addr, context, UET_READ_REQ, 0,
			       ep->tx_flags);
}

static ssize_t
uet_readmsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_generic_rma(ep, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       UET_READ_REQ, msg->data, uet_tx_flags(flags |
			       ep->util_ep.tx_msg_flags));
}

static ssize_t
uet_write(struct fid_ep *ep_fid, const void *buf, size_t len, void *desc,
	fi_addr_t dest_addr, uint64_t addr, uint64_t key, void *context)
{
	struct uet_ep *ep;
	struct iovec msg_iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = (void *) buf;
	msg_iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return uet_generic_rma(ep, &msg_iov, 1, &rma_iov, 1, &desc,
			       dest_addr, context, UET_WRITE, 0,
			       ep->tx_flags);
}

static ssize_t
uet_writev(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
		size_t count, fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		void *context)
{
	struct uet_ep *ep;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	rma_iov.addr = addr;
	rma_iov.len  = ofi_total_iov_len(iov, count);
	rma_iov.key = key;

	return uet_generic_rma(ep, iov, count, &rma_iov, 1, desc,
			       dest_addr, context, UET_WRITE, 0,
			       ep->tx_flags);
}

static ssize_t
uet_writemsg(struct fid_ep *ep_fid, const struct fi_msg_rma *msg,
	uint64_t flags)
{
	struct uet_ep *ep;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	return uet_generic_rma(ep, msg->msg_iov, msg->iov_count,
			       msg->rma_iov, msg->rma_iov_count,
			       msg->desc, msg->addr, msg->context,
			       UET_WRITE, msg->data, uet_tx_flags(flags |
			       ep->util_ep.tx_msg_flags));
}

static ssize_t
uet_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context)
{
	struct uet_ep *ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len  = len;
	rma_iov.key = key;

	return uet_generic_rma(ep, &iov, 1, &rma_iov, 1, &desc,
			       dest_addr, context, UET_WRITE, data,
			       ep->tx_flags | UET_REMOTE_CQ_DATA);
}

static ssize_t
uet_inject_write(struct fid_ep *ep_fid, const void *buf,
	size_t len, fi_addr_t dest_addr, uint64_t addr, uint64_t key)
{
	struct uet_ep *uet_ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	uet_ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return uet_generic_write_inject(uet_ep, &iov, 1, &rma_iov, 1,
					dest_addr, NULL, UET_WRITE, 0,
					UET_NO_TX_COMP | UET_INJECT);
}

static ssize_t
uet_inject_writedata(struct fid_ep *ep_fid, const void *buf, size_t len,
		     uint64_t data, fi_addr_t dest_addr, uint64_t addr,
		     uint64_t key)
{
	struct uet_ep *uet_ep;
	struct iovec iov;
	struct fi_rma_iov rma_iov;

	uet_ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;
	rma_iov.addr = addr;
	rma_iov.len = len;
	rma_iov.key = key;

	return uet_generic_write_inject(uet_ep, &iov, 1, &rma_iov, 1,
					dest_addr, NULL, UET_WRITE,
					data, UET_NO_TX_COMP | UET_INJECT |
					UET_REMOTE_CQ_DATA);
}

struct fi_ops_rma uet_ops_rma = {
	.size = sizeof(struct fi_ops_rma),
	.read = uet_read,
	.readv = uet_readv,
	.readmsg = uet_readmsg,
	.write = uet_write,
	.writev = uet_writev,
	.writemsg = uet_writemsg,
	.inject = uet_inject_write,
	.writedata = uet_writedata,
	.injectdata = uet_inject_writedata,

};
