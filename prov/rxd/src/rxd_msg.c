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

static int rxd_match_unexp(struct dlist_entry *item, const void *arg)
{
	struct rxd_x_entry *rx_entry = (struct rxd_x_entry *) arg;
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_base_hdr *hdr;

	pkt_entry = container_of(item, struct rxd_pkt_entry, d_entry);
	hdr = rxd_get_base_hdr(pkt_entry);

	if (!rxd_match_addr(rx_entry->peer, hdr->peer))
		return 0;

	if (hdr->type != RXD_TAGGED)
		return 1;

	return rxd_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     rxd_get_tag_hdr(pkt_entry)->tag);
}

static int rxd_ep_check_unexp_msg_list(struct rxd_ep *ep,
					struct dlist_entry *unexp_list,
					struct dlist_entry *rx_list,
					struct rxd_x_entry *rx_entry)
{
	struct dlist_entry *match;
	struct rxd_x_entry *progress_entry, *dup_entry = NULL;
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_base_hdr *base_hdr;
	struct rxd_sar_hdr *sar_hdr = NULL;
	struct rxd_tag_hdr *tag_hdr = NULL;
	struct rxd_data_hdr *data_hdr = NULL;
	struct rxd_atom_hdr *atom_hdr = NULL;
	struct rxd_rma_hdr *rma_hdr = NULL;
	void *msg = NULL;
	size_t msg_size, total_size;

	while (!dlist_empty(unexp_list)) {
		match = dlist_remove_first_match(unexp_list, &rxd_match_unexp,
						 (void *) rx_entry);
		if (!match)
			return 0;

		FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "progressing unexp msg entry\n");
	
		pkt_entry = container_of(match, struct rxd_pkt_entry, d_entry);
		base_hdr = rxd_get_base_hdr(pkt_entry);

		rxd_unpack_hdrs(pkt_entry->pkt_size - ep->rx_prefix_size,
				base_hdr, &sar_hdr, &tag_hdr,
				&data_hdr, &rma_hdr, &atom_hdr, &msg, &msg_size);

		total_size = sar_hdr ? sar_hdr->size : msg_size;
		if (rx_entry->flags & RXD_MULTI_RECV)
			dup_entry = rxd_progress_multi_recv(ep, rx_entry, total_size);

		progress_entry = dup_entry ? dup_entry : rx_entry;
	
		progress_entry->cq_entry.len = MIN(rx_entry->cq_entry.len, total_size);

		rxd_progress_op(ep, progress_entry, pkt_entry, base_hdr, sar_hdr, tag_hdr,
				data_hdr, rma_hdr, atom_hdr, &msg, msg_size);
		rxd_release_repost_rx(ep, pkt_entry);
		rxd_ep_send_ack(ep, base_hdr->peer);

		if (!dup_entry)
			return 1;
	}

	return 0;
}

ssize_t rxd_ep_generic_recvmsg(struct rxd_ep *rxd_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t ignore, void *context, uint32_t op,
			       uint32_t rxd_flags)
{
	ssize_t ret = 0;
	struct rxd_x_entry *rx_entry;
	struct dlist_entry *unexp_list, *rx_list;

	assert(iov_count <= RXD_IOV_LIMIT);
	assert(!(rxd_flags & RXD_MULTI_RECV) || iov_count == 1);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.rx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.rx_cq->cirq)) {
		ret = -FI_EAGAIN;
		goto out;
	}

	rx_entry = rxd_rx_entry_init(rxd_ep, iov, iov_count, tag, ignore, context,
				(rxd_ep->util_ep.caps & FI_DIRECTED_RECV &&
				addr != FI_ADDR_UNSPEC) ?
				rxd_ep_av(rxd_ep)->fi_addr_table[addr] :
				FI_ADDR_UNSPEC, op, rxd_flags);
	if (!rx_entry) {
		ret = -FI_EAGAIN;
		goto out;
	}

	if (op == ofi_op_tagged) {
		unexp_list = &rxd_ep->unexp_tag_list;
		rx_list = &rxd_ep->rx_tag_list;
	} else {
		unexp_list = &rxd_ep->unexp_list;
		rx_list = &rxd_ep->rx_list;
	}

	if (!dlist_empty(unexp_list) &&
	    rxd_ep_check_unexp_msg_list(rxd_ep, unexp_list, rx_list, rx_entry))
		goto out;

	dlist_insert_tail(&rx_entry->entry, rx_list);
out:
	fastlock_release(&rxd_ep->util_ep.rx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

static ssize_t rxd_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, msg->msg_iov, msg->iov_count,
				      msg->addr, 0, ~0, msg->context, ofi_op_msg,
				      rxd_flags(flags));
}

static ssize_t rxd_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			   fi_addr_t src_addr, void *context)
{
	struct rxd_ep *ep;
	struct iovec msg_iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	msg_iov.iov_base = buf;
	msg_iov.iov_len = len;

	return rxd_ep_generic_recvmsg(ep, &msg_iov, 1, src_addr, 0, ~0, context,
				      ofi_op_msg, rxd_ep_rx_flags(ep));
}

static ssize_t rxd_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t src_addr, void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_recvmsg(ep, iov, count, src_addr,
				      0, ~0, context, ofi_op_msg, rxd_ep_rx_flags(ep));
}

ssize_t rxd_ep_generic_inject(struct rxd_ep *rxd_ep, const struct iovec *iov,
			      size_t iov_count, fi_addr_t addr, uint64_t tag,
			      uint64_t data, uint32_t op, uint32_t rxd_flags)
{
	struct rxd_x_entry *tx_entry;
	ssize_t ret = -FI_EAGAIN;
	fi_addr_t rxd_addr;

	assert(iov_count <= RXD_IOV_LIMIT);
	assert(ofi_total_iov_len(iov, iov_count) <=
	       rxd_ep_domain(rxd_ep)->max_inline_msg);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.tx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.tx_cq->cirq))
		goto out;

	rxd_addr = rxd_ep_av(rxd_ep)->fi_addr_table[addr];
	ret = rxd_send_rts_if_needed(rxd_ep, rxd_addr);
	if (ret)
		goto out;

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, NULL, 0, 0, data,
				     tag, NULL, rxd_addr, op, rxd_flags | RXD_INJECT);
	if (!tx_entry)
		goto out;

	ret = rxd_ep_send_op(rxd_ep, tx_entry, NULL, 0, NULL, 0, 0, 0);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

ssize_t rxd_ep_generic_sendmsg(struct rxd_ep *rxd_ep, const struct iovec *iov,
			       size_t iov_count, fi_addr_t addr, uint64_t tag,
			       uint64_t data, void *context, uint32_t op,
			       uint32_t rxd_flags)
{
	struct rxd_x_entry *tx_entry;
	ssize_t ret = -FI_EAGAIN;
	fi_addr_t rxd_addr;

	assert(iov_count <= RXD_IOV_LIMIT);

	if (rxd_flags & RXD_INJECT)
		return rxd_ep_generic_inject(rxd_ep, iov, iov_count, addr, tag, 0,
					     op, rxd_flags);

	fastlock_acquire(&rxd_ep->util_ep.lock);
	fastlock_acquire(&rxd_ep->util_ep.tx_cq->cq_lock);

	if (ofi_cirque_isfull(rxd_ep->util_ep.tx_cq->cirq))
		goto out;

	rxd_addr = rxd_ep_av(rxd_ep)->fi_addr_table[addr];
	ret = rxd_send_rts_if_needed(rxd_ep, rxd_addr);
	if (ret)
		goto out;

	tx_entry = rxd_tx_entry_init(rxd_ep, iov, iov_count, NULL, 0, 0,
				     data, tag, context, rxd_addr, op, rxd_flags);
	if (!tx_entry)
		goto out;

	ret = rxd_ep_send_op(rxd_ep, tx_entry, NULL, 0, NULL, 0, 0, 0);
	if (ret)
		rxd_tx_entry_free(rxd_ep, tx_entry);

out:
	fastlock_release(&rxd_ep->util_ep.tx_cq->cq_lock);
	fastlock_release(&rxd_ep->util_ep.lock);
	return ret;
}

static ssize_t rxd_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, msg->msg_iov, msg->iov_count,
				   msg->addr, 0, msg->data, msg->context,
				   ofi_op_msg, rxd_flags(flags));

}

static ssize_t rxd_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov, void **desc,
			    size_t count, fi_addr_t dest_addr, void *context)
{
	struct rxd_ep *ep;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	return rxd_ep_generic_sendmsg(ep, iov, count, dest_addr, 0,
				      0, context, ofi_op_msg,
				      rxd_ep_tx_flags(ep));
}

static ssize_t rxd_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0,
				      0, context, ofi_op_msg,
				      rxd_ep_tx_flags(ep));
}

static ssize_t rxd_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, 0, 0, ofi_op_msg,
				     RXD_NO_TX_COMP | RXD_INJECT);
}

static ssize_t rxd_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       void *context)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_sendmsg(ep, &iov, 1, dest_addr, 0, data, context,
				      ofi_op_msg, rxd_ep_tx_flags(ep) |
				      RXD_REMOTE_CQ_DATA);
}

static ssize_t rxd_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr)
{
	struct rxd_ep *ep;
	struct iovec iov;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);

	iov.iov_base = (void *) buf;
	iov.iov_len = len;

	return rxd_ep_generic_inject(ep, &iov, 1, dest_addr, 0, data, ofi_op_msg,
				     RXD_NO_TX_COMP | RXD_INJECT |
				     RXD_REMOTE_CQ_DATA);
}

struct fi_ops_msg rxd_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxd_ep_recv,
	.recvv = rxd_ep_recvv,
	.recvmsg = rxd_ep_recvmsg,
	.send = rxd_ep_send,
	.sendv = rxd_ep_sendv,
	.sendmsg = rxd_ep_sendmsg,
	.inject = rxd_ep_inject,
	.senddata = rxd_ep_senddata,
	.injectdata = rxd_ep_injectdata,
};
