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

struct rxd_pkt_entry *rxd_get_tx_pkt(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ep->do_local_mr ?
		    util_buf_alloc_ex(ep->tx_pkt_pool, &mr) :
		    util_buf_alloc(ep->tx_pkt_pool);

	pkt_entry->mr = (struct fid_mr *) mr;
	pkt_entry->retry_cnt = 0;
	rxd_set_pkt(ep, pkt_entry);

	return pkt_entry;
}

static struct rxd_pkt_entry *rxd_get_rx_pkt(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	void *mr = NULL;

	pkt_entry = ep->do_local_mr ?
		    util_buf_alloc_ex(ep->rx_pkt_pool, &mr) :
		    util_buf_alloc(ep->rx_pkt_pool);

	pkt_entry->mr = (struct fid_mr *) mr;
	pkt_entry->retry_cnt = 0;

	rxd_set_pkt(ep, pkt_entry);

	return pkt_entry;
}

void rxd_release_tx_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt)
{
	util_buf_release(ep->tx_pkt_pool, pkt);
}

void rxd_release_rx_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt)
{
	util_buf_release(ep->rx_pkt_pool, pkt);
}

static int rxd_match_ctx(struct dlist_entry *item, const void *arg)
{
	struct rxd_x_entry *x_entry;

	x_entry = container_of(item, struct rxd_x_entry, entry);

	return (x_entry->cq_entry.op_context == arg);
}

static ssize_t rxd_ep_cancel(fid_t fid, void *context)
{
	struct rxd_ep *ep;
	struct dlist_entry *entry;
	struct rxd_x_entry *rx_entry;
	struct fi_cq_err_entry err_entry;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	fastlock_acquire(&ep->util_ep.lock);

	entry = dlist_remove_first_match(&ep->rx_list,
				&rxd_match_ctx, context);
	if (!entry)
		goto out;

	rx_entry = container_of(entry, struct rxd_x_entry, entry);

	memset(&err_entry, 0, sizeof(struct fi_cq_err_entry));

	rxd_rx_entry_free(ep, rx_entry);
	err_entry.op_context = rx_entry->cq_entry.op_context;
	err_entry.flags = (FI_MSG | FI_RECV);
	err_entry.err = FI_ECANCELED;
	err_entry.prov_errno = -FI_ECANCELED;
	rxd_cq_report_error(rxd_ep_rx_cq(ep), &err_entry);

out:
	fastlock_release(&ep->util_ep.lock);
	return 0;
}

static int rxd_ep_getopt(fid_t fid, int level, int optname,
		   void *optval, size_t *optlen)
{
	return -FI_ENOSYS;
}

static int rxd_ep_setopt(fid_t fid, int level, int optname,
		   const void *optval, size_t optlen)
{
	return -FI_ENOSYS;
}

struct fi_ops_ep rxd_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = rxd_ep_cancel,
	.getopt = rxd_ep_getopt,
	.setopt = rxd_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

struct rxd_x_entry *rxd_rx_entry_init(struct rxd_ep *ep,
			const struct iovec *iov, size_t iov_count, uint64_t tag,
			uint64_t ignore, void *context, fi_addr_t addr,
			uint32_t op, uint32_t flags)
{
	struct rxd_x_entry *rx_entry;

	if (freestack_isempty(ep->rx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no-more rx entries\n");
		return NULL;
	}

	rx_entry = freestack_pop(ep->rx_fs);
	rx_entry->rx_id = rxd_x_fs_index(ep->rx_fs, rx_entry);
	rx_entry->peer = addr;
	rx_entry->flags = flags;
	rx_entry->bytes_done = 0;
	rx_entry->next_seg_no = 0;
	rx_entry->window = rxd_env.max_unacked;
	rx_entry->iov_count = iov_count;
	rx_entry->op = op;
	rx_entry->ignore = ignore;

	memcpy(rx_entry->iov, iov, sizeof(*rx_entry->iov) * iov_count);

	rx_entry->cq_entry.op_context = context;
	rx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	rx_entry->cq_entry.buf = iov[0].iov_base;
	rx_entry->cq_entry.tag = tag;

	rx_entry->cq_entry.flags = ofi_rx_cq_flags(op);
	if (rx_entry->cq_entry.flags & FI_TAGGED)
		dlist_insert_tail(&rx_entry->entry, &ep->rx_tag_list);
	else if (rx_entry->cq_entry.flags & FI_RECV)
		dlist_insert_tail(&rx_entry->entry, &ep->rx_list);
	else
		dlist_init(&rx_entry->entry);

	return rx_entry;
}

static inline void *rxd_mr_desc(struct fid_mr *mr, struct rxd_ep *ep)
{
	return (ep->do_local_mr) ? fi_mr_desc(mr) : NULL;
}

int rxd_ep_post_buf(struct rxd_ep *ep)
{
	struct rxd_pkt_entry *pkt_entry;
	ssize_t ret;

	pkt_entry = rxd_get_rx_pkt(ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	ret = fi_recv(ep->dg_ep, rxd_pkt_start(pkt_entry),
		      rxd_ep_domain(ep)->max_mtu_sz,
		      rxd_mr_desc(pkt_entry->mr, ep),
		      FI_ADDR_UNSPEC, &pkt_entry->context);
	if (ret) {
		rxd_release_rx_pkt(ep, pkt_entry);
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "failed to repost\n");
		return ret;
	}

	ep->posted_bufs++;
	slist_insert_tail(&pkt_entry->s_entry, &ep->rx_pkt_list);

	return 0;
}

static int rxd_ep_enable(struct rxd_ep *ep)
{
	size_t i;
	ssize_t ret;

	ret = fi_enable(ep->dg_ep);
	if (ret)
		return ret;

	fastlock_acquire(&ep->util_ep.lock);
	for (i = 0; i < ep->rx_size; i++) {
		ret = rxd_ep_post_buf(ep);
		if (ret)
			goto out;
	}
out:
	fastlock_release(&ep->util_ep.lock);
	return ret;
}

/*
 * Exponential back-off starting at 1ms, max 4s.
 */
void rxd_set_timeout(struct rxd_pkt_entry *pkt_entry)
{
	pkt_entry->retry_time = fi_gettime_ms() +
			      MIN(1 << (++pkt_entry->retry_cnt), 4000);
}

void rxd_init_data_pkt(struct rxd_ep *ep, struct rxd_x_entry *tx_entry,
		       struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_data_pkt *data_pkt = (struct rxd_data_pkt *) (pkt_entry->pkt);
	uint32_t seg_size;

	seg_size = tx_entry->cq_entry.len - tx_entry->bytes_done;
	seg_size = MIN(rxd_ep_domain(ep)->max_seg_sz, seg_size);

	data_pkt->base_hdr.version = RXD_PROTOCOL_VERSION;
	data_pkt->base_hdr.type = tx_entry->cq_entry.flags & FI_READ ?
				  RXD_DATA_READ : RXD_DATA;

	data_pkt->pkt_hdr.rx_id = tx_entry->rx_id;
	data_pkt->pkt_hdr.seg_no = tx_entry->next_seg_no++;
	data_pkt->pkt_hdr.tx_id = tx_entry->tx_id;
	data_pkt->pkt_hdr.msg_id = tx_entry->msg_id;
	data_pkt->pkt_hdr.peer = ep->peers[tx_entry->peer].peer_addr;

	pkt_entry->pkt_size = ofi_copy_from_iov(data_pkt->msg, seg_size,
						tx_entry->iov,
						tx_entry->iov_count,
						tx_entry->bytes_done);

	tx_entry->bytes_done += pkt_entry->pkt_size;
	data_pkt->pkt_hdr.flags = (tx_entry->bytes_done == tx_entry->cq_entry.len) ?
				   RXD_LAST : 0;

	pkt_entry->pkt_size += sizeof(struct rxd_data_pkt) + ep->prefix_size;
}

struct rxd_x_entry *rxd_tx_entry_init(struct rxd_ep *ep, const struct iovec *iov,
				      size_t iov_count, const struct iovec *res_iov,
				      size_t res_count, uint64_t data, uint64_t tag,
				      void *context, fi_addr_t addr, uint32_t op,
				      uint32_t flags)
{
	struct rxd_x_entry *tx_entry;
	struct rxd_domain *rxd_domain = rxd_ep_domain(ep);

	if (freestack_isempty(ep->tx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no-more tx entries\n");
		return NULL;
	}

	tx_entry = freestack_pop(ep->tx_fs);

	tx_entry->tx_id = rxd_x_fs_index(ep->tx_fs, tx_entry);
	tx_entry->rx_id = 0;
	tx_entry->msg_id = ep->peers[addr].tx_msg_id++;

	tx_entry->op = op;
	tx_entry->peer = addr;
	tx_entry->flags = flags;
	tx_entry->bytes_done = 0;
	tx_entry->next_seg_no = 0;
	tx_entry->iov_count = iov_count;
	memcpy(&tx_entry->iov[0], iov, sizeof(*iov) * iov_count);
	if (res_count) {
		tx_entry->res_count = res_count;
		memcpy(&tx_entry->res_iov[0], res_iov, sizeof(*res_iov) * res_count);
	}

	if (flags & RXD_REMOTE_CQ_DATA)
		tx_entry->cq_entry.data = data;

	tx_entry->cq_entry.op_context = context;
	tx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	tx_entry->cq_entry.buf = iov[0].iov_base;
	tx_entry->cq_entry.flags = ofi_tx_cq_flags(op);
	tx_entry->cq_entry.tag = tag;

	if (tx_entry->cq_entry.len <= rxd_domain->max_inline_sz ||
	    tx_entry->cq_entry.flags & FI_ATOMIC) {
		tx_entry->num_segs = 1;
	} else if (tx_entry->cq_entry.flags & FI_READ) {
		tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len,
						  rxd_domain->max_seg_sz);
	} else {
		tx_entry->num_segs = ofi_div_ceil(tx_entry->cq_entry.len -
						  rxd_domain->max_inline_sz,
						  rxd_domain->max_seg_sz) + 1;
	}

	if ((tx_entry->op == RXD_READ_REQ || tx_entry->op == RXD_ATOMIC_FETCH ||
	     tx_entry->op == RXD_ATOMIC_COMPARE) &&
	    ep->peers[tx_entry->peer].unacked_cnt < rxd_env.max_unacked &&
	    ep->peers[tx_entry->peer].peer_addr != FI_ADDR_UNSPEC &&
	    !ep->peers[tx_entry->peer].blocking)
		dlist_insert_tail(&tx_entry->entry,
				  &ep->peers[tx_entry->peer].rma_rx_list);
	else
		dlist_insert_tail(&tx_entry->entry,
				  &ep->peers[tx_entry->peer].tx_list);

	return tx_entry;
}

void rxd_tx_entry_free(struct rxd_ep *ep, struct rxd_x_entry *tx_entry)
{
	tx_entry->op = RXD_NO_OP;
	dlist_remove(&tx_entry->entry);
	freestack_push(ep->tx_fs, tx_entry);
}

void rxd_insert_unacked(struct rxd_ep *ep, fi_addr_t peer,
			struct rxd_pkt_entry *pkt_entry)
{
	dlist_insert_tail(&pkt_entry->d_entry,
			  &ep->peers[peer].unacked);
	ep->peers[peer].unacked_cnt++;
	rxd_ep_retry_pkt(ep, pkt_entry);
}

ssize_t rxd_ep_post_data_pkts(struct rxd_ep *ep, struct rxd_x_entry *tx_entry)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_data_pkt *data;

	if (ep->peers[tx_entry->peer].blocking)
		return 0;

	while (tx_entry->bytes_done != tx_entry->cq_entry.len) {
		if (ep->peers[tx_entry->peer].unacked_cnt >= rxd_env.max_unacked)
			return 0;

		pkt_entry = rxd_get_tx_pkt(ep);
		if (!pkt_entry)
			return -FI_ENOMEM;

		if (tx_entry->op == RXD_DATA_READ && !tx_entry->bytes_done) {
			tx_entry->start_seq = ep->peers[tx_entry->peer].tx_seq_no++;
			ep->peers[tx_entry->peer].tx_seq_no = tx_entry->start_seq +
							      tx_entry->num_segs;
		}

		rxd_init_data_pkt(ep, tx_entry, pkt_entry);

		if (ep->peers[tx_entry->peer].unacked_cnt < rxd_env.max_unacked) {
			data = (struct rxd_data_pkt *) (pkt_entry->pkt);
			data->pkt_hdr.seq_no = tx_entry->start_seq +
					       data->pkt_hdr.seg_no;
			if (data->base_hdr.type != RXD_DATA_READ)
				data->pkt_hdr.seq_no++;

			rxd_insert_unacked(ep, tx_entry->peer, pkt_entry);
		}
	}

	return ep->peers[tx_entry->peer].unacked_cnt < rxd_env.max_unacked;
}

int rxd_ep_retry_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	rxd_set_timeout(pkt_entry);
	return fi_send(ep->dg_ep, (const void *) rxd_pkt_start(pkt_entry),
		       pkt_entry->pkt_size, rxd_mr_desc(pkt_entry->mr, ep),
		       pkt_entry->peer, &pkt_entry->context);
}

ssize_t rxd_ep_send_rts(struct rxd_ep *rxd_ep, int dg_addr)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_rts_pkt *rts_pkt;
	ssize_t ret;
	size_t addrlen;

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	rts_pkt = (struct rxd_rts_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*rts_pkt) + rxd_ep->prefix_size;
	pkt_entry->peer = dg_addr;

	rts_pkt->base_hdr.version = RXD_PROTOCOL_VERSION;
	rts_pkt->base_hdr.type = RXD_RTS;
	rts_pkt->dg_addr = dg_addr;

	addrlen = RXD_NAME_LENGTH;
	memset(rts_pkt->source, 0, RXD_NAME_LENGTH);
	ret = fi_getname(&rxd_ep->dg_ep->fid, (void *) rts_pkt->source,
			 &addrlen);
	if (ret) {
		rxd_release_tx_pkt(rxd_ep, pkt_entry);
		return ret;
	}

	//don't insert this here, it won't get retransmitted
	dlist_insert_head(&pkt_entry->d_entry, &rxd_ep->peers[dg_addr].unacked);

	return rxd_ep_retry_pkt(rxd_ep, pkt_entry);
}

int rxd_ep_send_op(struct rxd_ep *rxd_ep, struct rxd_x_entry *tx_entry,
		   const struct fi_rma_iov *rma_iov, size_t rma_count,
		   const struct iovec *comp_iov, size_t comp_count,
		   enum fi_datatype datatype, enum fi_op atomic_op)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_domain *rxd_domain = rxd_ep_domain(rxd_ep);
	struct rxd_op_pkt *op;
	size_t len;
	int ret = 0;

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	op = (struct rxd_op_pkt *) (pkt_entry->pkt);
	pkt_entry->peer = tx_entry->peer;

	op->base_hdr.version = RXD_PROTOCOL_VERSION;
	op->base_hdr.type = tx_entry->op;

	op->tag = tx_entry->cq_entry.tag;
	op->cq_data = tx_entry->cq_entry.data;
	op->size = tx_entry->cq_entry.len;
	op->num_segs = tx_entry->num_segs;

	op->pkt_hdr.flags = tx_entry->flags;
	op->pkt_hdr.tx_id = tx_entry->tx_id;
	op->pkt_hdr.peer = rxd_ep->peers[tx_entry->peer].peer_addr;
	op->pkt_hdr.msg_id = tx_entry->msg_id;

	if (tx_entry->op != RXD_READ_REQ) {
		tx_entry->bytes_done = ofi_copy_from_iov(op->msg,
							 rxd_domain->max_inline_sz,
							 tx_entry->iov,
							 tx_entry->iov_count, 0);
		if (tx_entry->op == RXD_ATOMIC_COMPARE) {
			len = ofi_copy_from_iov(op->msg + tx_entry->cq_entry.len,
						tx_entry->cq_entry.len,
						comp_iov, comp_count, 0);
			if (len != tx_entry->cq_entry.len) {
				FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
					"compare data length mismatch\n");
			}
		}
	}

	pkt_entry->pkt_size = tx_entry->bytes_done + sizeof(*op) + rxd_ep->prefix_size;

	if (rma_count) {
		memcpy(op->rma, rma_iov, sizeof(*rma_iov) * rma_count);
		op->iov_count = rma_count;
		if (tx_entry->cq_entry.flags & FI_ATOMIC) {
			op->datatype = datatype;
			op->atomic_op = atomic_op;
		}
	}

	if (rxd_ep->peers[tx_entry->peer].unacked_cnt < rxd_env.max_unacked &&
	    rxd_ep->peers[tx_entry->peer].peer_addr != FI_ADDR_UNSPEC &&
	    !rxd_ep->peers[tx_entry->peer].blocking) {
		op->pkt_hdr.seq_no = rxd_ep->peers[tx_entry->peer].tx_seq_no++;
		tx_entry->start_seq = op->pkt_hdr.seq_no;
		if (tx_entry->op != RXD_READ_REQ && op->num_segs > 1) {
			rxd_ep->peers[tx_entry->peer].blocking = 1;
			rxd_ep->peers[tx_entry->peer].tx_seq_no = tx_entry->start_seq +
								  tx_entry->num_segs;
		}
		rxd_insert_unacked(rxd_ep, tx_entry->peer, pkt_entry);
	} else {
		tx_entry->op_pkt = pkt_entry;
	}

	if (tx_entry->op != RXD_READ_REQ && !(tx_entry->cq_entry.flags & FI_ATOMIC) &&
	    !(tx_entry->flags & RXD_INJECT))
		ret = rxd_ep_post_data_pkts(rxd_ep, tx_entry);

	return ret == -FI_ENOMEM ? ret : 0;
}

void rxd_ep_send_ack(struct rxd_ep *rxd_ep, fi_addr_t peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_ack_pkt *ack;
	int ret;

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Unable to send ack\n");
		return;
	}

	ack = (struct rxd_ack_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*ack) + rxd_ep->prefix_size;
	pkt_entry->peer = peer;

	ack->base_hdr.version = RXD_PROTOCOL_VERSION;
	ack->base_hdr.type = RXD_ACK;
	ack->pkt_hdr.peer = rxd_ep->peers[peer].peer_addr;
	ack->pkt_hdr.seq_no = rxd_ep->peers[peer].rx_seq_no;
	ack->pkt_hdr.msg_id = rxd_ep->peers[peer].rx_msg_id;
	ack->pkt_hdr.tx_id = rxd_ep->peers[peer].curr_tx_id;
	ack->pkt_hdr.rx_id = rxd_ep->peers[peer].curr_rx_id;
	rxd_ep->peers[peer].last_tx_ack = ack->pkt_hdr.seq_no;

	ret = rxd_ep_retry_pkt(rxd_ep, pkt_entry);
	if (ret)
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "Unable to send ack\n");

	rxd_release_tx_pkt(rxd_ep, pkt_entry);
}

static void rxd_ep_free_res(struct rxd_ep *ep)
{

	if (ep->tx_fs)
		rxd_x_fs_free(ep->tx_fs);

	if (ep->rx_fs)
		rxd_x_fs_free(ep->rx_fs);

	util_buf_pool_destroy(ep->tx_pkt_pool);
	util_buf_pool_destroy(ep->rx_pkt_pool);
}

static void rxd_close_peer(struct rxd_ep *ep, struct rxd_peer *peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_x_entry *x_entry;

	while (!dlist_empty(&peer->unacked)) {
		dlist_pop_front(&peer->unacked, struct rxd_pkt_entry,
				pkt_entry, d_entry);
		rxd_release_tx_pkt(ep, pkt_entry);
		peer->unacked_cnt--;
	}

	while(!dlist_empty(&peer->tx_list)) {
		dlist_pop_front(&peer->tx_list, struct rxd_x_entry,
				x_entry, entry);
		rxd_tx_entry_free(ep, x_entry);
	}

	while(!dlist_empty(&peer->rx_list)) {
		dlist_pop_front(&peer->rx_list, struct rxd_x_entry,
				x_entry, entry);
		rxd_rx_entry_free(ep, x_entry);
	}

	while(!dlist_empty(&peer->rma_rx_list)) {
		dlist_pop_front(&peer->rma_rx_list, struct rxd_x_entry,
				x_entry, entry);
		rxd_tx_entry_free(ep, x_entry);
	}
}

static int rxd_ep_close(struct fid *fid)
{
	int ret;
	struct rxd_ep *ep;
	struct rxd_pkt_entry *pkt_entry;
	struct slist_entry *entry;
	struct rxd_peer *peer;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);

	dlist_foreach_container(&ep->active_peers, struct rxd_peer, peer, entry)
		rxd_close_peer(ep, peer);

	ret = fi_close(&ep->dg_ep->fid);
	if (ret)
		return ret;

	ret = fi_close(&ep->dg_cq->fid);
	if (ret)
		return ret;

	while (!slist_empty(&ep->rx_pkt_list)) {
		entry = slist_remove_head(&ep->rx_pkt_list);
		pkt_entry = container_of(entry, struct rxd_pkt_entry, s_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	while (!dlist_empty(&ep->unexp_list)) {
		dlist_pop_front(&ep->unexp_list, struct rxd_pkt_entry,
				pkt_entry, d_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	while (!dlist_empty(&ep->unexp_tag_list)) {
		dlist_pop_front(&ep->unexp_tag_list, struct rxd_pkt_entry,
				pkt_entry, d_entry);
		rxd_release_rx_pkt(ep, pkt_entry);
	}

	if (ep->util_ep.tx_cq) {
		/* TODO: wait handling */
		fid_list_remove(&ep->util_ep.tx_cq->ep_list,
				&ep->util_ep.tx_cq->ep_list_lock,
				&ep->util_ep.ep_fid.fid);
	}

	if (ep->util_ep.rx_cq) {
		if (ep->util_ep.rx_cq != ep->util_ep.tx_cq) {
			/* TODO: wait handling */
			fid_list_remove(&ep->util_ep.rx_cq->ep_list,
					&ep->util_ep.rx_cq->ep_list_lock,
					&ep->util_ep.ep_fid.fid);
		}
	}

	rxd_ep_free_res(ep);
	ofi_endpoint_close(&ep->util_ep);
	free(ep);
	return 0;
}

static int rxd_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxd_ep *ep;
	struct rxd_av *av;
	int ret = 0;

	ep = container_of(ep_fid, struct rxd_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct rxd_av, util_av.av_fid.fid);
		ret = ofi_ep_bind_av(&ep->util_ep, &av->util_av);
		if (ret)
			return ret;

		ret = fi_ep_bind(ep->dg_ep, &av->dg_av->fid, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		ret = ofi_ep_bind_cq(&ep->util_ep, container_of(bfid,
				     struct util_cq, cq_fid.fid), flags);
		break;
	case FI_CLASS_EQ:
		break;
	case FI_CLASS_CNTR:
		return ofi_ep_bind_cntr(&ep->util_ep, container_of(bfid,
					struct util_cntr, cntr_fid.fid), flags);
	default:
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int rxd_ep_control(struct fid *fid, int command, void *arg)
{
	int ret;
	struct rxd_ep *ep;

	switch (command) {
	case FI_ENABLE:
		ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
		ret = rxd_ep_enable(ep);
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}
	return ret;
}

static struct fi_ops rxd_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_ep_close,
	.bind = rxd_ep_bind,
	.control = rxd_ep_control,
	.ops_open = fi_no_ops_open,
};

static int rxd_ep_cm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxd_ep *ep;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	return fi_setname(&ep->dg_ep->fid, addr, addrlen);
}

static int rxd_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxd_ep *ep;

	ep = container_of(fid, struct rxd_ep, util_ep.ep_fid.fid);
	return fi_getname(&ep->dg_ep->fid, addr, addrlen);
}

struct fi_ops_cm rxd_ep_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxd_ep_cm_setname,
	.getname = rxd_ep_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static void rxd_remove_from_list(struct rxd_ep *ep, struct dlist_entry *list,
				 uint32_t tx_id)
{
	struct dlist_entry *entry = list->next;
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_pkt_hdr *hdr;
	int removed = -1; 

	while (entry) {
		pkt_entry = container_of(entry, struct rxd_pkt_entry, d_entry);
		entry = entry->next;
		hdr = rxd_get_pkt_hdr(pkt_entry);
		if (hdr->tx_id != tx_id && removed == tx_id)
			break;
		if (hdr->tx_id == tx_id) {
			dlist_remove(&pkt_entry->d_entry);
			rxd_release_tx_pkt(ep, pkt_entry);
			removed = tx_id;
		}
	}
}

static void rxd_ep_remove_msg_pkts(struct rxd_ep *ep, struct rxd_peer *peer,
				   uint32_t tx_id)
{
	rxd_remove_from_list(ep, &peer->unacked, tx_id);
}

static void rxd_ep_progress(struct util_ep *util_ep)
{
	struct rxd_peer *peer;
	struct fi_cq_err_entry err_entry;
	struct rxd_x_entry *tx_entry;
	struct fi_cq_msg_entry cq_entry;
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_pkt_hdr *hdr;
	struct rxd_ep *ep;
	uint64_t current;
	ssize_t ret;
	int i;

	ep = container_of(util_ep, struct rxd_ep, util_ep);

	fastlock_acquire(&ep->util_ep.lock);
	for(ret = 1, i = 0;
	    ret > 0 && (!rxd_env.spin_count || i < rxd_env.spin_count);
	    i++) {
		ret = fi_cq_read(ep->dg_cq, &cq_entry, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (cq_entry.flags & FI_RECV)
			rxd_handle_recv_comp(ep, &cq_entry);
	}

	if (!rxd_env.retry)
		goto out;

	current = fi_gettime_ms();

	dlist_foreach_container(&ep->active_peers, struct rxd_peer, peer, entry) {
		dlist_foreach_container(&peer->unacked, struct rxd_pkt_entry,
					pkt_entry, d_entry) {
			if (current < pkt_entry->retry_time)
				continue;

			if (rxd_pkt_type(pkt_entry) != RXD_RTS) {
				hdr = rxd_get_pkt_hdr(pkt_entry);
				if (pkt_entry->retry_cnt > RXD_MAX_PKT_RETRY) {
					memset(&err_entry, 0, sizeof(struct fi_cq_err_entry));
					rxd_ep_remove_msg_pkts(ep, peer, hdr->tx_id);
					tx_entry = &ep->tx_fs->entry[hdr->tx_id].buf;
					rxd_tx_entry_free(ep, tx_entry);
					err_entry.op_context = tx_entry->cq_entry.op_context;
					err_entry.flags = (FI_MSG | FI_SEND);
					err_entry.err = FI_ECONNREFUSED;
					err_entry.prov_errno = 0;
					rxd_cq_report_error(rxd_ep_tx_cq(ep), &err_entry);
					continue;
				}
				hdr->flags |= RXD_RETRY;
			}
			ret = rxd_ep_retry_pkt(ep, pkt_entry);
			if (ret)
				break;
		}
		if (dlist_empty(&peer->unacked))
			rxd_progress_tx_list(ep, peer);
	}

out:
	while (ep->posted_bufs < ep->rx_size && !ret)
		ret = rxd_ep_post_buf(ep);

	fastlock_release(&ep->util_ep.lock);
}

static int rxd_buf_region_alloc_hndlr(void *pool_ctx, void *addr, size_t len,
				void **context)
{
	int ret;
	struct fid_mr *mr;
	struct rxd_domain *domain = pool_ctx;

	ret = fi_mr_reg(domain->dg_domain, addr, len,
			FI_SEND | FI_RECV, 0, 0, 0, &mr, NULL);
	*context = mr;
	return ret;
}

static void rxd_buf_region_free_hndlr(void *pool_ctx, void *context)
{
	fi_close((struct fid *) context);
}

int rxd_ep_init_res(struct rxd_ep *ep, struct fi_info *fi_info)
{
	struct rxd_domain *rxd_domain = rxd_ep_domain(ep);

	int ret = util_buf_pool_create_ex(
		&ep->tx_pkt_pool,
		rxd_domain->max_mtu_sz + sizeof(struct rxd_pkt_entry),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_TX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_domain);
	if (ret)
		return -FI_ENOMEM;

	ret = util_buf_pool_create_ex(
		&ep->rx_pkt_pool,
		rxd_domain->max_mtu_sz + sizeof (struct rxd_pkt_entry),
		RXD_BUF_POOL_ALIGNMENT, 0, RXD_RX_POOL_CHUNK_CNT,
	        (fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_alloc_hndlr : NULL,
		(fi_info->mode & FI_LOCAL_MR) ? rxd_buf_region_free_hndlr : NULL,
		rxd_domain);
	if (ret)
		goto err;

	//doubling sizes to handle incoming RMA operations (not included in tx and rx size)
	ep->tx_fs = rxd_x_fs_create(ep->tx_size * 2, NULL, NULL);
	if (!ep->tx_fs)
		goto err;
	ep->rx_fs = rxd_x_fs_create(ep->rx_size * 2, NULL, NULL);
	if (!ep->rx_fs)
		goto err;

	dlist_init(&ep->rx_list);
	dlist_init(&ep->rx_tag_list);
	dlist_init(&ep->active_peers);
	dlist_init(&ep->unexp_list);
	dlist_init(&ep->unexp_tag_list);
	slist_init(&ep->rx_pkt_list);

	return 0;
err:
	if (ep->tx_pkt_pool)
		util_buf_pool_destroy(ep->tx_pkt_pool);

	if (ep->rx_pkt_pool)
		util_buf_pool_destroy(ep->rx_pkt_pool);

	if (ep->tx_fs)
		rxd_x_fs_free(ep->tx_fs);

	if (ep->rx_fs)
		rxd_x_fs_free(ep->rx_fs);

	return -FI_ENOMEM;
}

static void rxd_init_peer(struct rxd_ep *ep, uint64_t dg_addr)
{
	ep->peers[dg_addr].peer_addr = FI_ADDR_UNSPEC;
	ep->peers[dg_addr].tx_seq_no = 0;
	ep->peers[dg_addr].rx_seq_no = 0;
	ep->peers[dg_addr].last_rx_ack = 0;
	ep->peers[dg_addr].last_tx_ack = 0;
	ep->peers[dg_addr].tx_msg_id = 0;
	ep->peers[dg_addr].rx_msg_id = 0;
	ep->peers[dg_addr].rx_window = rxd_env.max_unacked;
	ep->peers[dg_addr].blocking = 0;
	ep->peers[dg_addr].unacked_cnt = 0;
	dlist_init(&ep->peers[dg_addr].unacked);
	dlist_init(&ep->peers[dg_addr].tx_list);
	dlist_init(&ep->peers[dg_addr].rx_list);
	dlist_init(&ep->peers[dg_addr].rma_rx_list);
	dlist_init(&ep->peers[dg_addr].buf_ops);
	dlist_init(&ep->peers[dg_addr].buf_cq);
}

int rxd_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	struct fi_info *dg_info;
	struct rxd_domain *rxd_domain;
	struct rxd_ep *rxd_ep;
	struct fi_cq_attr cq_attr;
	int ret, i;

	rxd_ep = calloc(1, sizeof(*rxd_ep) + sizeof(struct rxd_peer) *
			rxd_env.max_peers);
	if (!rxd_ep)
		return -FI_ENOMEM;

	rxd_domain = container_of(domain, struct rxd_domain,
				  util_domain.domain_fid);
	memset(&cq_attr, 0, sizeof cq_attr);
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = FI_WAIT_FD;

	ret = ofi_endpoint_init(domain, &rxd_util_prov, info, &rxd_ep->util_ep,
				context, rxd_ep_progress);
	if (ret)
		goto err1;

	ret = ofi_get_core_info(rxd_domain->util_domain.fabric->fabric_fid.api_version,
				NULL, NULL, 0, &rxd_util_prov, info,
				rxd_info_to_core, &dg_info);
	if (ret)
		goto err2;

	rxd_ep->do_local_mr = (rxd_domain->mr_mode & FI_MR_LOCAL) ? 1 : 0;

	ret = fi_endpoint(rxd_domain->dg_domain, dg_info, &rxd_ep->dg_ep, rxd_ep);
	cq_attr.size = dg_info->tx_attr->size + dg_info->rx_attr->size;
	fi_freeinfo(dg_info);
	if (ret)
		goto err2;

	ret = fi_cq_open(rxd_domain->dg_domain, &cq_attr, &rxd_ep->dg_cq, rxd_ep);
	if (ret)
		goto err3;

	ret = fi_ep_bind(rxd_ep->dg_ep, &rxd_ep->dg_cq->fid,
			 FI_TRANSMIT | FI_RECV);
	if (ret)
		goto err4;

	rxd_ep->rx_size = info->rx_attr->size;
	rxd_ep->tx_size = info->tx_attr->size;
	rxd_ep->prefix_size = info->ep_attr->msg_prefix_size;
	ret = rxd_ep_init_res(rxd_ep, info);
	if (ret)
		goto err4;

	for (i = 0; i < rxd_env.max_peers; rxd_init_peer(rxd_ep, i++))
		;

	rxd_ep->util_ep.ep_fid.fid.ops = &rxd_ep_fi_ops;
	rxd_ep->util_ep.ep_fid.cm = &rxd_ep_cm;
	rxd_ep->util_ep.ep_fid.ops = &rxd_ops_ep;
	rxd_ep->util_ep.ep_fid.msg = &rxd_ops_msg;
	rxd_ep->util_ep.ep_fid.tagged = &rxd_ops_tagged;
	rxd_ep->util_ep.ep_fid.rma = &rxd_ops_rma;
	rxd_ep->util_ep.ep_fid.atomic = &rxd_ops_atomic;

	*ep = &rxd_ep->util_ep.ep_fid;
	return 0;

err4:
	fi_close(&rxd_ep->dg_cq->fid);
err3:
	fi_close(&rxd_ep->dg_ep->fid);
err2:
	ofi_endpoint_close(&rxd_ep->util_ep);
err1:
	free(rxd_ep);
	return ret;
}
