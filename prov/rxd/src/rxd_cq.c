/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
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
#include <inttypes.h>
#include <ofi_iov.h>
#include "rxd.h"

/*
 * All EPs use the same underlying datagram provider, so pick any and use its
 * associated CQ.
 */
static const char *rxd_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
		const void *err_data, char *buf, size_t len)
{
	struct fid_list_entry *fid_entry;
	struct util_ep *util_ep;
	struct rxd_cq *cq;
	struct rxd_ep *ep;
	const char *str;

	cq = container_of(cq_fid, struct rxd_cq, util_cq.cq_fid);

	fastlock_acquire(&cq->util_cq.ep_list_lock);
	assert(!dlist_empty(&cq->util_cq.ep_list));
	fid_entry = container_of(cq->util_cq.ep_list.next,
				struct fid_list_entry, entry);
	util_ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
	ep = container_of(util_ep, struct rxd_ep, util_ep);

	str = fi_cq_strerror(ep->dg_cq, prov_errno, err_data, buf, len);
	fastlock_release(&cq->util_cq.ep_list_lock);
	return str;
}

static int rxd_cq_write_ctx(struct rxd_cq *cq,
			     struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;

	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	ofi_cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_ctx_signal(struct rxd_cq *cq,
				    struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_ctx(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_msg(struct rxd_cq *cq,
			     struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	ofi_cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_msg_signal(struct rxd_cq *cq,
				    struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_msg(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_data(struct rxd_cq *cq,
			      struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	comp->op_context = cq_entry->op_context;
	comp->flags = cq_entry->flags;
	comp->len = cq_entry->len;
	comp->buf = cq_entry->buf;
	comp->data = cq_entry->data;
	ofi_cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_data_signal(struct rxd_cq *cq,
				     struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_data(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

static int rxd_cq_write_tagged(struct rxd_cq *cq,
				struct fi_cq_tagged_entry *cq_entry)
{
	struct fi_cq_tagged_entry *comp;
	if (ofi_cirque_isfull(cq->util_cq.cirq))
		return -FI_ENOSPC;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL,
	       "report completion: %" PRIx64 "\n", cq_entry->tag);

	comp = ofi_cirque_tail(cq->util_cq.cirq);
	*comp = *cq_entry;
	ofi_cirque_commit(cq->util_cq.cirq);
	return 0;
}

static int rxd_cq_write_tagged_signal(struct rxd_cq *cq,
				       struct fi_cq_tagged_entry *cq_entry)
{
	int ret = rxd_cq_write_tagged(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

void rxd_rx_entry_free(struct rxd_ep *ep, struct rxd_x_entry *rx_entry)
{
	rx_entry->op = RXD_NO_OP;
	dlist_remove(&rx_entry->entry);
	freestack_push(ep->rx_fs, rx_entry);
}

static int rxd_match_pkt_entry(struct slist_entry *item, const void *arg)
{
	return ((struct rxd_pkt_entry *) arg ==
		container_of(item, struct rxd_pkt_entry, s_entry));
} 

static void rxd_remove_rx_pkt(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	struct slist_entry *item;

	item = slist_remove_first_match(&ep->rx_pkt_list, rxd_match_pkt_entry,
					pkt_entry);
	if (!item) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"could not find posted rx to release\n");
	}
}

void rxd_release_repost_rx(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	rxd_release_rx_pkt(ep, pkt_entry);
	rxd_ep_post_buf(ep);
}

void rxd_cq_report_error(struct rxd_cq *cq, struct fi_cq_err_entry *err_entry)
{
	struct fi_cq_tagged_entry cq_entry = {0};
	struct util_cq_oflow_err_entry *entry = calloc(1, sizeof(*entry));
	if (!entry) {
		FI_WARN(&rxd_prov, FI_LOG_CQ,
			"out of memory, cannot report CQ error\n");
		return;
	}

	entry->comp = *err_entry;
	slist_insert_tail(&entry->list_entry, &cq->util_cq.oflow_err_list);
	cq_entry.flags = UTIL_FLAG_ERROR;
	cq->write_fn(cq, &cq_entry);
}

static void rxd_complete_rx(struct rxd_ep *ep, struct rxd_x_entry *rx_entry)
{
	struct fi_cq_err_entry err_entry;
	struct rxd_cq *rx_cq = rxd_ep_rx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.rx_cntr;

	if ((!(rx_entry->cq_entry.flags & FI_REMOTE_CQ_DATA) &&
	     (rx_entry->cq_entry.flags & FI_RMA ||
	     rx_entry->cq_entry.flags & FI_ATOMIC)) ||
	     rx_entry->flags & RXD_NO_RX_COMP)
		goto out;

	if (rx_entry->cq_entry.flags & FI_RECV)
		ep->rx_list_size--;

	/* Handle CQ comp */
	if (rx_entry->bytes_done == rx_entry->cq_entry.len) {
		rx_cq->write_fn(rx_cq, &rx_entry->cq_entry);
		/* Handle cntr */
		if (cntr)
			cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
	} else {
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = rx_entry->cq_entry.op_context;
		err_entry.flags = rx_entry->cq_entry.flags;
		err_entry.len = rx_entry->bytes_done;
		err_entry.err = FI_ETRUNC;
		err_entry.prov_errno = 0;
		rxd_cq_report_error(rx_cq, &err_entry);
	}

out:
	rxd_rx_entry_free(ep, rx_entry);
}

static void rxd_complete_tx(struct rxd_ep *ep, struct rxd_x_entry *tx_entry)
{
	struct rxd_cq *tx_cq = rxd_ep_tx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.tx_cntr;

	if (tx_entry->flags & RXD_NO_TX_COMP)
		goto out;
	tx_cq->write_fn(tx_cq, &tx_entry->cq_entry);
	if (cntr)
		cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);

out:
	rxd_tx_entry_free(ep, tx_entry);
}

static int rxd_comp_pkt_seq_no(struct dlist_entry *item, const void *arg)
{
	struct rxd_base_hdr *list_hdr;
	struct rxd_base_hdr *new_hdr;

	list_hdr = rxd_get_base_hdr(container_of(item,
				   struct rxd_pkt_entry, d_entry));

	new_hdr = rxd_get_base_hdr(container_of((struct dlist_entry *) arg,
				  struct rxd_pkt_entry, d_entry));

	return new_hdr->seq_no > list_hdr->seq_no;
}

static void rxd_ep_recv_data(struct rxd_ep *ep, struct rxd_x_entry *x_entry,
			     struct rxd_data_pkt *pkt, size_t size)
{
	struct rxd_domain *rxd_domain = rxd_ep_domain(ep);
	uint64_t done;
	struct iovec *iov;
	size_t iov_count;

	if (x_entry->cq_entry.flags & FI_ATOMIC) {
		iov = x_entry->res_iov;
		iov_count = x_entry->res_count;
	} else {
		iov = x_entry->iov;
		iov_count = x_entry->iov_count;
	}

	done = ofi_copy_to_iov(iov, iov_count, x_entry->offset +
			       (pkt->ext_hdr.seg_no * rxd_domain->max_seg_sz),
			       pkt->msg, size - sizeof(struct rxd_data_pkt) -
			       ep->prefix_size);

	x_entry->bytes_done += done;
	ep->peers[pkt->base_hdr.peer].rx_seq_no++;
	x_entry->next_seg_no++;

	if (x_entry->next_seg_no < x_entry->num_segs) {
		if (!(ep->peers[pkt->base_hdr.peer].rx_seq_no %
		    ep->peers[pkt->base_hdr.peer].rx_window))
			rxd_ep_send_ack(ep, pkt->base_hdr.peer);
		return;
	}
	rxd_ep_send_ack(ep, pkt->base_hdr.peer);

	if (x_entry->cq_entry.flags & FI_READ) {
		fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
		rxd_complete_tx(ep, x_entry);
		fastlock_release(&ep->util_ep.tx_cq->cq_lock);
	} else {
		fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
		rxd_complete_rx(ep, x_entry);
		fastlock_release(&ep->util_ep.rx_cq->cq_lock);
	}

	rxd_ep_send_ack(ep, pkt->base_hdr.peer);
}

static int rxd_verify_active(struct rxd_ep *ep, fi_addr_t addr, fi_addr_t peer_addr)
{
	if (ep->peers[addr].peer_addr == peer_addr)
		return 1;
	if (ep->peers[addr].peer_addr != FI_ADDR_UNSPEC) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"overwriting active peer - unexpected behavior\n");
		return 1;
	}

	ep->peers[addr].peer_addr = peer_addr;
	dlist_insert_tail(&ep->peers[addr].entry, &ep->active_peers);

	return 0;
}

static int rxd_move_tx_pkt(struct rxd_ep *ep, struct rxd_x_entry *tx_entry)
{
	struct rxd_base_hdr *hdr = rxd_get_base_hdr(tx_entry->pkt);

	if (ep->peers[tx_entry->peer].unacked_cnt >= rxd_env.max_unacked)
		return 0;

	tx_entry->start_seq = rxd_set_pkt_seq(&ep->peers[tx_entry->peer],
					      tx_entry->pkt);
	if (tx_entry->op != RXD_READ_REQ && tx_entry->num_segs > 1) {
		ep->peers[tx_entry->peer].tx_seq_no = tx_entry->start_seq +
						      tx_entry->num_segs;
	}
	hdr->peer = ep->peers[tx_entry->peer].peer_addr;
	rxd_insert_unacked(ep, tx_entry->peer, tx_entry->pkt);
	tx_entry->pkt = NULL;

	if (tx_entry->op == RXD_READ_REQ || tx_entry->op == RXD_ATOMIC_FETCH ||
	    tx_entry->op == RXD_ATOMIC_COMPARE) {
		dlist_remove(&tx_entry->entry);
		dlist_insert_tail(&tx_entry->entry,
				  &ep->peers[tx_entry->peer].rma_rx_list);
	} else if (tx_entry->num_segs > 1) {
		ep->peers[tx_entry->peer].blocking = 1;
		return 0;
	}

	return ep->peers[tx_entry->peer].unacked_cnt < rxd_env.max_unacked;
}

void rxd_progress_tx_list(struct rxd_ep *ep, struct rxd_peer *peer)
{
	struct dlist_entry *tmp_entry;
	struct rxd_x_entry *tx_entry;

	dlist_foreach_container_safe(&peer->tx_list, struct rxd_x_entry,
				tx_entry, entry, tmp_entry) {
		if (tx_entry->pkt) {
			if (!rxd_move_tx_pkt(ep, tx_entry) ||
			    tx_entry->op == RXD_READ_REQ)
				break;
		}

		if (tx_entry->bytes_done == tx_entry->cq_entry.len) {
			if (ofi_before(tx_entry->start_seq + (tx_entry->num_segs - 1),
			    peer->last_rx_ack)) {
				fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
				rxd_complete_tx(ep, tx_entry);
				fastlock_release(&ep->util_ep.tx_cq->cq_lock);
			}
			continue;
		}

		if (!rxd_ep_post_data_pkts(ep, tx_entry))
			break;
	}
}

static void rxd_update_peer(struct rxd_ep *ep, fi_addr_t peer, fi_addr_t dg_addr)
{
	struct rxd_pkt_entry *pkt_entry;

	if (rxd_verify_active(ep, peer, dg_addr))
		return;

	ep->peers[peer].peer_addr = dg_addr;

	if (!dlist_empty(&ep->peers[peer].unacked)) {
		pkt_entry = container_of((&ep->peers[peer].unacked)->next,
					 struct rxd_pkt_entry, d_entry);
		if (rxd_pkt_type(pkt_entry) == RXD_RTS) {
			dlist_remove(&pkt_entry->d_entry);
			rxd_release_tx_pkt(ep, pkt_entry);
		}
	}

	rxd_progress_tx_list(ep, &ep->peers[peer]);
}

static int rxd_send_cts(struct rxd_ep *rxd_ep, struct rxd_rts_pkt *rts_pkt,
			fi_addr_t peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_cts_pkt *cts;
	int ret = 0;

	rxd_update_peer(rxd_ep, peer, rts_pkt->dg_addr);

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	cts = (struct rxd_cts_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*cts) + rxd_ep->prefix_size;
	pkt_entry->retry_cnt = 0;
	pkt_entry->peer = peer;

	cts->base_hdr.version = RXD_PROTOCOL_VERSION;
	cts->base_hdr.type = RXD_CTS;
	cts->dg_addr = peer;
	cts->peer_addr = rts_pkt->dg_addr;

	ret = rxd_ep_retry_pkt(rxd_ep, pkt_entry);
	rxd_release_tx_pkt(rxd_ep, pkt_entry);

	return ret;
}

static int rxd_match_msg(struct dlist_entry *item, const void *arg)
{
	struct rxd_match_attr *attr = (struct rxd_match_attr *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, attr->peer);
}

static int rxd_match_tmsg(struct dlist_entry *item, const void *arg)
{
	struct rxd_match_attr *attr = (struct rxd_match_attr *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, attr->peer) &&
	       rxd_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     attr->tag);
}

static void rxd_handle_data(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_data_pkt *pkt = (struct rxd_data_pkt *) (pkt_entry->pkt);
	struct rxd_x_entry *x_entry;

	x_entry = pkt->base_hdr.type == RXD_DATA_READ ?
		  &ep->tx_fs->entry[pkt->ext_hdr.rx_id].buf :
		  &ep->rx_fs->entry[pkt->ext_hdr.rx_id].buf;

	if ((x_entry->op != RXD_NO_OP && pkt->base_hdr.seq_no ==
	    ep->peers[pkt->base_hdr.peer].rx_seq_no) || !rxd_env.retry)
		rxd_ep_recv_data(ep, x_entry, pkt, pkt_entry->pkt_size);
	else
		rxd_ep_send_ack(ep, pkt->base_hdr.peer);
}

static void rxd_check_post_unexp(struct rxd_ep *ep, struct dlist_entry *list,
				 struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_pkt_entry *unexp_entry;
	struct rxd_base_hdr *new_hdr = rxd_get_base_hdr(pkt_entry);
	struct rxd_base_hdr *unexp_hdr;

	if (!rxd_env.retry)
		goto insert;

	dlist_foreach_container(list, struct rxd_pkt_entry, unexp_entry, d_entry) {
		unexp_hdr = rxd_get_base_hdr(unexp_entry);
		if (unexp_hdr->seq_no == new_hdr->seq_no &&
		    unexp_hdr->peer == new_hdr->peer) {
			rxd_release_repost_rx(ep, pkt_entry);
			return;
		}
	}

insert:
	dlist_insert_tail(&pkt_entry->d_entry, list);
}

static void rxd_handle_rts(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_av *rxd_av;
	struct ofi_rbnode *node;
	fi_addr_t dg_addr;
	struct rxd_rts_pkt *pkt = (struct rxd_rts_pkt *) (pkt_entry->pkt);
	int ret;

	rxd_av = rxd_ep_av(ep);
	node = ofi_rbmap_find(&rxd_av->rbmap, pkt->source);

	if (node) {
		dg_addr = (fi_addr_t) node->data;
	} else {
		ret = rxd_av_insert_dg_addr(rxd_av, (void *) pkt->source,
					    &dg_addr, 0, NULL);
		if (ret)
			return;
	}

	if (rxd_send_cts(ep, pkt, dg_addr)) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"error posting CTS\n");
	}
}

static struct rxd_x_entry *rxd_match_rx(struct rxd_ep *ep,
					struct rxd_pkt_entry *pkt_entry,
					struct rxd_base_hdr *base,
					struct rxd_tag_hdr *tag,
					struct rxd_sar_hdr *op, size_t msg_size)
{
	struct rxd_x_entry *rx_entry;
	struct dlist_entry *unexp_list;
	struct dlist_entry *match;
	struct rxd_match_attr attr;

	attr.peer = base->peer;

	if (tag) {
		attr.tag = tag->tag;
		match = dlist_remove_first_match(&ep->rx_tag_list, &rxd_match_tmsg,
					 (void *) &attr);
		unexp_list = &ep->unexp_tag_list;
	} else {
		attr.tag = 0;
		match = dlist_remove_first_match(&ep->rx_list, &rxd_match_msg,
					 (void *) &attr);
		unexp_list = &ep->unexp_list;
	}

	if (match) {
		rx_entry = container_of(match, struct rxd_x_entry, entry);
		rx_entry->cq_entry.len = MIN(rx_entry->cq_entry.len,
			op ? op->size : msg_size);
		rx_entry->start_seq = base->seq_no;
		return rx_entry;
	}

	rxd_check_post_unexp(ep, unexp_list, pkt_entry);
	return NULL;
}

static int rxd_verify_iov(struct rxd_ep *ep, struct ofi_rma_iov *rma,
			  size_t count, uint32_t type, struct iovec *iov)
{
	struct util_domain *util_domain = &rxd_ep_domain(ep)->util_domain;
	int i, ret;

	for (i = 0; i < count; i++) {
		ret = ofi_mr_verify(&util_domain->mr_map, rma[i].len,
			(uintptr_t *)(&rma[i].addr), rma[i].key,
			ofi_rx_mr_reg_flags(type, 0));
		iov[i].iov_base = (void *) rma[i].addr;
		iov[i].iov_len = rma[i].len;
		if (ret) {
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "could not verify MR\n");
			return -FI_EACCES; 
		}
	}
	return 0;
}

static struct rxd_x_entry *rxd_rma_read_entry_init(struct rxd_ep *ep,
			struct rxd_base_hdr *base_hdr, struct rxd_sar_hdr *sar_hdr,
			struct rxd_rma_hdr *rma_hdr)
{
	struct rxd_x_entry *tx_entry;
	struct rxd_domain *rxd_domain = rxd_ep_domain(ep);
	int ret;

	if (freestack_isempty(ep->tx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no more tx entries\n");
		return NULL;
	}

	tx_entry = freestack_pop(ep->tx_fs);

	tx_entry->tx_id = rxd_x_fs_index(ep->tx_fs, tx_entry);
	tx_entry->rx_id = sar_hdr->tx_id;

	tx_entry->op = RXD_DATA_READ;
	tx_entry->peer = base_hdr->peer;
	//TODO RMA event capability + also readmsg with REMOTE_CQ_DATA
	tx_entry->flags = RXD_NO_TX_COMP;
	tx_entry->bytes_done = 0;
	tx_entry->next_seg_no = 0;
	tx_entry->num_segs = ofi_div_ceil(sar_hdr->size, rxd_domain->max_seg_sz);

 	ret = rxd_verify_iov(ep, rma_hdr->rma, sar_hdr->iov_count,
			     base_hdr->type, tx_entry->iov);
	if (ret)
		return NULL;

	tx_entry->iov_count = sar_hdr->iov_count;
	tx_entry->cq_entry.flags = FI_RMA | FI_READ;
	tx_entry->cq_entry.len = sar_hdr->size;

	dlist_insert_tail(&tx_entry->entry, &ep->peers[tx_entry->peer].tx_list);

	//TODO check ret?
	rxd_ep_send_ack(ep, base_hdr->peer);

	rxd_progress_tx_list(ep, &ep->peers[tx_entry->peer]);

	return tx_entry;
}

static struct rxd_x_entry *rxd_rma_rx_entry_init(struct rxd_ep *ep,
			struct rxd_base_hdr *base_hdr, struct rxd_sar_hdr *sar_hdr,
			struct rxd_rma_hdr *rma_hdr)
{
	struct rxd_x_entry *rx_entry;
	struct iovec iov[RXD_IOV_LIMIT];
	int ret, iov_count;

	iov_count = sar_hdr ? sar_hdr->iov_count : 1;
	ret = rxd_verify_iov(ep, rma_hdr->rma, iov_count,
			     base_hdr->type, iov);
	if (ret)
		return NULL;

	rx_entry = rxd_rx_entry_init(ep, iov, iov_count, 0, 0, NULL,
				     base_hdr->peer, base_hdr->type,
				     base_hdr->flags);
	rx_entry->start_seq = base_hdr->seq_no;

	return rx_entry;
}

static struct rxd_x_entry *rxd_rx_atomic_fetch(struct rxd_ep *ep,
			struct rxd_base_hdr *base_hdr,
			struct rxd_sar_hdr *sar_hdr,
			struct rxd_rma_hdr *rma_hdr,
			struct rxd_atom_hdr *atom_hdr)
{
	struct rxd_x_entry *tx_entry;
	int ret;

	if (freestack_isempty(ep->tx_fs)) {
		FI_INFO(&rxd_prov, FI_LOG_EP_CTRL, "no more tx entries\n");
		return NULL;
	}

	tx_entry = freestack_pop(ep->tx_fs);

	tx_entry->pkt = rxd_get_tx_pkt(ep);
	if (!tx_entry->pkt) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "could not get pkt\n");
		rxd_tx_entry_free(ep, tx_entry);
		return NULL;
	}
	tx_entry->tx_id = rxd_x_fs_index(ep->tx_fs, tx_entry);
	tx_entry->rx_id = sar_hdr->tx_id;

	tx_entry->op = RXD_DATA_READ;
	tx_entry->peer = base_hdr->peer;
	tx_entry->flags = RXD_NO_TX_COMP;
	tx_entry->bytes_done = 0;
	tx_entry->next_seg_no = 0;
	tx_entry->num_segs = 1;

	tx_entry->iov_count = sar_hdr ? sar_hdr->iov_count : 1;
 	ret = rxd_verify_iov(ep, rma_hdr->rma, tx_entry->iov_count,
			     base_hdr->type, tx_entry->iov);
	if (ret)
		return NULL;

	tx_entry->cq_entry.flags = FI_READ | FI_ATOMIC;
	tx_entry->cq_entry.len = sar_hdr->size;

	rxd_init_data_pkt(ep, tx_entry, tx_entry->pkt);
	if (tx_entry->bytes_done != tx_entry->cq_entry.len)
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL, "fetch data length mismatch\n");

	dlist_insert_tail(&tx_entry->entry, &ep->peers[tx_entry->peer].tx_list);

	rxd_ep_send_ack(ep, base_hdr->peer);

	rxd_progress_tx_list(ep, &ep->peers[tx_entry->peer]);

	return tx_entry;
}

void rxd_unpack_hdrs(size_t pkt_size, struct rxd_base_hdr *base_hdr,
		     struct rxd_sar_hdr **sar_hdr, struct rxd_tag_hdr **tag_hdr,
		     struct rxd_data_hdr **data_hdr, struct rxd_rma_hdr **rma_hdr,
		     struct rxd_atom_hdr **atom_hdr, void **msg, size_t *msg_size)
{
	char *ptr = (char *) base_hdr + sizeof(*base_hdr);
	uint8_t rma_count = 1;

	if (!(base_hdr->flags & RXD_INLINE)) {
		*sar_hdr = (struct rxd_sar_hdr *) ptr;
		rma_count = (*sar_hdr)->iov_count;
		ptr += sizeof(**sar_hdr);
	} else {
		*sar_hdr = NULL;
	}

	if (base_hdr->flags & RXD_TAG_HDR) {
		*tag_hdr = (struct rxd_tag_hdr *) ptr;
		ptr += sizeof(**tag_hdr);
	} else {
		*tag_hdr = NULL;
	}

	if (base_hdr->flags & RXD_REMOTE_CQ_DATA) {
		*data_hdr = (struct rxd_data_hdr *) ptr;
		ptr += sizeof(**data_hdr);
	} else {
		*data_hdr = NULL;
	}

	if (base_hdr->type >= RXD_READ_REQ && base_hdr->type <= RXD_ATOMIC_COMPARE) {
		*rma_hdr = (struct rxd_rma_hdr *) ptr;
		ptr += (sizeof(*(*rma_hdr)->rma) * rma_count);

		if (base_hdr->type >= RXD_ATOMIC) {
			*atom_hdr = (struct rxd_atom_hdr *) ptr;
			ptr += sizeof(**atom_hdr);
		} else {
			*atom_hdr = NULL;
		}
	} else {
		*rma_hdr = NULL;
		*atom_hdr = NULL;
	}

	*msg = ptr;
	*msg_size = pkt_size - (ptr - (char *) base_hdr);
}

static struct rxd_x_entry *rxd_unpack_init_rx(struct rxd_ep *ep,
					      struct rxd_pkt_entry *pkt_entry,
					      struct rxd_base_hdr *base_hdr,
					      struct rxd_sar_hdr **sar_hdr,
					      struct rxd_tag_hdr **tag_hdr,
					      struct rxd_data_hdr **data_hdr,
					      struct rxd_rma_hdr **rma_hdr,
					      struct rxd_atom_hdr **atom_hdr,
					      void **msg, size_t *msg_size)
{
	rxd_unpack_hdrs(pkt_entry->pkt_size - ep->prefix_size, base_hdr, sar_hdr,
			tag_hdr, data_hdr, rma_hdr, atom_hdr, msg, msg_size);

	switch (base_hdr->type) {
	case RXD_MSG:
	case RXD_TAGGED:
		return rxd_match_rx(ep, pkt_entry, base_hdr, *tag_hdr, *sar_hdr,
				    *msg_size);
	case RXD_READ_REQ:
		return rxd_rma_read_entry_init(ep, base_hdr, *sar_hdr, *rma_hdr);
	case RXD_ATOMIC_FETCH:
	case RXD_ATOMIC_COMPARE:
		return rxd_rx_atomic_fetch(ep, base_hdr, *sar_hdr, *rma_hdr, *atom_hdr);
	default:
		return rxd_rma_rx_entry_init(ep, base_hdr, *sar_hdr, *rma_hdr);
	}
}

void rxd_do_atomic(void *src, void *dst, void *cmp, enum fi_datatype datatype,
		   enum fi_op atomic_op, size_t cnt)
{
	char tmp_result[RXD_MAX_MTU_SIZE];

	if (atomic_op >= OFI_SWAP_OP_START) {
		ofi_atomic_swap_handlers[atomic_op - OFI_SWAP_OP_START][datatype](dst,
			src, cmp, tmp_result, cnt);
	} else if (atomic_op != FI_ATOMIC_READ) {
		ofi_atomic_write_handlers[atomic_op][datatype](dst, src, cnt);
	}
}

void rxd_progress_op_msg(struct rxd_ep *ep, struct rxd_x_entry *rx_entry,
			 void **msg, size_t size)
{
	rx_entry->bytes_done = ofi_copy_to_iov(rx_entry->iov,
					       rx_entry->iov_count, 0, *msg, size);
}

void rxd_progress_atom_op(struct rxd_ep *ep, struct rxd_x_entry *rx_entry,
			  struct rxd_base_hdr *base_hdr, struct rxd_sar_hdr *sar_hdr,
			  struct rxd_rma_hdr *rma_hdr, struct rxd_atom_hdr *atom_hdr,
			  void **msg, size_t msg_size)
{
	char *src, *cmp;
	size_t len;
	int i, iov_count;

	src = (char *) (*msg);
	cmp = base_hdr->type == RXD_ATOMIC_COMPARE ? (char *) (*msg) +
		(msg_size / 2) : NULL;

	iov_count = sar_hdr ? sar_hdr->iov_count : 1;
	for (i = len = 0; i < iov_count; i++) {
		rxd_do_atomic(&src[len], rx_entry->iov[i].iov_base,
			      cmp ? &cmp[len] : NULL, atom_hdr->datatype,
			      atom_hdr->atomic_op, rx_entry->iov[i].iov_len /
			      ofi_datatype_size(atom_hdr->datatype));
		len += rx_entry->iov[i].iov_len;
	}

	if (base_hdr->type == RXD_ATOMIC)
		rx_entry->bytes_done = len;
}

void rxd_progress_op(struct rxd_ep *ep, struct rxd_x_entry *rx_entry,
		     struct rxd_pkt_entry *pkt_entry,
		     struct rxd_base_hdr *base_hdr,
		     struct rxd_sar_hdr *sar_hdr,
		     struct rxd_tag_hdr *tag_hdr,
		     struct rxd_data_hdr *data_hdr,
		     struct rxd_rma_hdr *rma_hdr,
		     struct rxd_atom_hdr *atom_hdr,
		     void **msg, size_t size)
{
	if (atom_hdr)
		rxd_progress_atom_op(ep, rx_entry, base_hdr, sar_hdr,
				     rma_hdr, atom_hdr, msg, size);
	else
		rxd_progress_op_msg(ep, rx_entry, msg, size);

	rx_entry->offset = rx_entry->bytes_done;

	if (data_hdr) {
		rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rx_entry->cq_entry.data = data_hdr->cq_data;
	}

	if (sar_hdr) {
		ep->peers[base_hdr->peer].curr_tx_id = sar_hdr->tx_id;
		ep->peers[base_hdr->peer].curr_rx_id = rx_entry->rx_id;
	}
	rx_entry->peer = base_hdr->peer;

	if (!sar_hdr || sar_hdr->num_segs == 1) {
		if (!(rx_entry->cq_entry.flags & FI_READ))
			rxd_complete_rx(ep, rx_entry);
		return;
	}

	rx_entry->tx_id = sar_hdr->tx_id;
	rx_entry->num_segs = sar_hdr->num_segs;
	rx_entry->next_seg_no++;

	dlist_insert_tail(&rx_entry->entry, &ep->peers[base_hdr->peer].rx_list);
}

static void rxd_progress_buf_ops(struct rxd_ep *ep, fi_addr_t peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_base_hdr *base_hdr;
	struct rxd_sar_hdr *sar_hdr;
	struct rxd_tag_hdr *tag_hdr;
	struct rxd_data_hdr *data_hdr;
	struct rxd_rma_hdr *rma_hdr;
	struct rxd_atom_hdr *atom_hdr;
	void *msg;
	size_t msg_size;
	struct rxd_x_entry *rx_entry;

	while (!dlist_empty(&ep->peers[peer].buf_ops)) {
		pkt_entry = container_of((&ep->peers[peer].buf_ops)->next,
					struct rxd_pkt_entry, d_entry);
		base_hdr = rxd_get_base_hdr(pkt_entry);
		if (base_hdr->seq_no != ep->peers[peer].rx_seq_no)
			return;

		rx_entry = rxd_unpack_init_rx(ep, pkt_entry, base_hdr, &sar_hdr,
					      &tag_hdr, &data_hdr, &rma_hdr, &atom_hdr,
					      &msg, &msg_size);
		if (rx_entry) {
			rxd_progress_op(ep, rx_entry, pkt_entry, base_hdr,
					sar_hdr, tag_hdr, data_hdr, rma_hdr,
					atom_hdr, &msg, msg_size);
			dlist_remove(&pkt_entry->d_entry);
			rxd_release_repost_rx(ep, pkt_entry);
		}
	}
}

static void rxd_handle_op(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_x_entry *rx_entry;
	struct rxd_base_hdr *base_hdr = rxd_get_base_hdr(pkt_entry);
	struct rxd_sar_hdr *sar_hdr;
	struct rxd_tag_hdr *tag_hdr;
	struct rxd_data_hdr *data_hdr;
	struct rxd_rma_hdr *rma_hdr;
	struct rxd_atom_hdr *atom_hdr;
	void *msg;
	size_t msg_size;

	if (base_hdr->seq_no != ep->peers[base_hdr->peer].rx_seq_no) {
		if (!rxd_env.retry) {
			rxd_remove_rx_pkt(ep, pkt_entry);
			dlist_insert_order(&ep->peers[base_hdr->peer].buf_ops,
					   &rxd_comp_pkt_seq_no, &pkt_entry->d_entry);
			ep->peers[base_hdr->peer].rx_seq_no++;
			return;
		}

		if (ep->peers[base_hdr->peer].rx_seq_no !=
		    ep->peers[base_hdr->peer].last_tx_ack)
			goto ack;
		goto release;
	}


	ep->peers[base_hdr->peer].rx_seq_no++;
	rx_entry = rxd_unpack_init_rx(ep, pkt_entry, base_hdr, &sar_hdr,
				      &tag_hdr, &data_hdr, &rma_hdr, &atom_hdr,
				      &msg, &msg_size);
	if (!rx_entry) {
		rxd_remove_rx_pkt(ep, pkt_entry);
		return;
	}

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	if (base_hdr->type != RXD_READ_REQ)
		rxd_progress_op(ep, rx_entry, pkt_entry, base_hdr, sar_hdr, tag_hdr,
				data_hdr, rma_hdr, atom_hdr, &msg, msg_size);

	if (!rxd_env.retry && !dlist_empty(&ep->peers[base_hdr->peer].buf_ops))
		rxd_progress_buf_ops(ep, base_hdr->peer);

	fastlock_release(&ep->util_ep.rx_cq->cq_lock);

ack:
	rxd_ep_send_ack(ep, base_hdr->peer);
release:
	rxd_remove_rx_pkt(ep, pkt_entry);
	rxd_release_repost_rx(ep, pkt_entry);
}

static void rxd_handle_cts(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_cts_pkt *cts = (struct rxd_cts_pkt *) (pkt_entry->pkt);

	rxd_update_peer(ep, cts->peer_addr, cts->dg_addr);
}

static void rxd_handle_ack(struct rxd_ep *ep, struct rxd_pkt_entry *ack_entry)
{
	struct rxd_x_entry *tx_entry;
	struct rxd_ack_pkt *ack = (struct rxd_ack_pkt *) (ack_entry->pkt);
	struct rxd_pkt_entry *pkt_entry;
	fi_addr_t peer = ack->base_hdr.peer;
	struct rxd_base_hdr *hdr;
	struct rxd_sar_hdr *sar_hdr;

	ep->peers[peer].last_rx_ack = ack->base_hdr.seq_no;

	while (!dlist_empty(&ep->peers[peer].unacked)) {
		pkt_entry = container_of((&ep->peers[peer].unacked)->next,
					struct rxd_pkt_entry, d_entry);
		hdr = rxd_get_base_hdr(pkt_entry);
		if (ofi_after_eq(hdr->seq_no, ack->base_hdr.seq_no))
			break;

		if (rxd_pkt_type(pkt_entry) <= RXD_ATOMIC_COMPARE &&
		    !(hdr->flags & RXD_INLINE)) {
			sar_hdr = rxd_get_sar_hdr(pkt_entry);
			tx_entry = &ep->tx_fs->entry[sar_hdr->tx_id].buf;
			if (tx_entry->cq_entry.flags & (FI_WRITE | FI_SEND)) {
				if (tx_entry->num_segs > 1) {
					tx_entry->rx_id = ack->ext_hdr.rx_id;
					ep->peers[peer].blocking = 0;
				}
			}
		}

		dlist_remove(&pkt_entry->d_entry);
		rxd_release_tx_pkt(ep, pkt_entry);
	     	ep->peers[peer].unacked_cnt--;
	}

	rxd_progress_tx_list(ep, &ep->peers[ack->base_hdr.peer]);
} 

void rxd_handle_recv_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_entry *pkt_entry;
	int release = 1;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got recv completion\n");

	pkt_entry = container_of(comp->op_context, struct rxd_pkt_entry, context);
	ep->posted_bufs--;

	pkt_entry->pkt_size = comp->len;
	switch (rxd_pkt_type(pkt_entry)) {
	case RXD_RTS:
		rxd_handle_rts(ep, pkt_entry);
		break;
	case RXD_CTS:
		rxd_handle_cts(ep, pkt_entry);
		break;
	case RXD_ACK:
		rxd_handle_ack(ep, pkt_entry);
		break;
	case RXD_DATA:
	case RXD_DATA_READ:
		rxd_handle_data(ep, pkt_entry);
		break;
	default:
		rxd_handle_op(ep, pkt_entry);
		release = 0;
		break;
	}

	if (release) {
		rxd_remove_rx_pkt(ep, pkt_entry);
		rxd_release_repost_rx(ep, pkt_entry);
	}
}

static int rxd_cq_close(struct fid *fid)
{
	int ret;
	struct rxd_cq *cq;

	cq = container_of(fid, struct rxd_cq, util_cq.cq_fid.fid);
	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;
	free(cq);
	return 0;
}

static struct fi_ops rxd_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxd_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

static struct fi_ops_cq rxd_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = ofi_cq_sread,
	.sreadfrom = ofi_cq_sreadfrom,
	.signal = ofi_cq_signal,
	.strerror = rxd_cq_strerror,
};

int rxd_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct rxd_cq *cq;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&rxd_prov, domain, attr, &cq->util_cq,
			  &ofi_cq_progress, context);
	if (ret)
		goto free;

	switch (attr->format) {
	case FI_CQ_FORMAT_UNSPEC:
	case FI_CQ_FORMAT_CONTEXT:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_ctx_signal : rxd_cq_write_ctx;
		break;
	case FI_CQ_FORMAT_MSG:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_msg_signal : rxd_cq_write_msg;
		break;
	case FI_CQ_FORMAT_DATA:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_data_signal : rxd_cq_write_data;
		break;
	case FI_CQ_FORMAT_TAGGED:
		cq->write_fn = cq->util_cq.wait ?
			rxd_cq_write_tagged_signal : rxd_cq_write_tagged;
		break;
	default:
		ret = -FI_EINVAL;
		goto cleanup;
	}

	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &rxd_cq_fi_ops;
	(*cq_fid)->ops = &rxd_cq_ops;
	return 0;

cleanup:
	ofi_cq_cleanup(&cq->util_cq);
free:
	free(cq);
	return ret;
}
