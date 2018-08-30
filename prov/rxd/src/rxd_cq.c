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

void rxd_cq_report_tx_comp(struct rxd_cq *cq, struct rxd_x_entry *tx_entry)
{
	cq->write_fn(cq, &tx_entry->cq_entry);
}

static void rxd_complete_rx(struct rxd_ep *ep, struct rxd_x_entry *rx_entry)
{
	struct fi_cq_err_entry err_entry;
	struct rxd_cq *rx_cq = rxd_ep_rx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.rx_cntr;

	/* Handle CQ comp */
	if (rx_entry->bytes_done == rx_entry->cq_entry.len) {
		rx_cq->write_fn(rx_cq, &rx_entry->cq_entry);
		/* Handle cntr */
		if (cntr)
			cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
	} else {
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = rx_entry->cq_entry.op_context;
		err_entry.flags = (FI_MSG | FI_RECV);
		err_entry.len = rx_entry->bytes_done;
		err_entry.err = FI_ETRUNC;
		err_entry.prov_errno = 0;
		rxd_cq_report_error(rx_cq, &err_entry);
	}

	ep->peers[rx_entry->peer].rx_msg_id = rx_entry->msg_id;
	rxd_rx_entry_free(ep, rx_entry);
}

static int rxd_comp_pkt_msg_id(struct dlist_entry *item, const void *arg)
{
	struct rxd_op_pkt *list_op = (struct rxd_op_pkt *)
				     ((container_of(item,
				     struct rxd_pkt_entry, d_entry))->pkt);
	struct rxd_op_pkt *new_op = (struct rxd_op_pkt *)
				    ((container_of((struct dlist_entry *) arg,
				    struct rxd_pkt_entry, d_entry))->pkt);

	return new_op->pkt_hdr.msg_id > list_op->pkt_hdr.msg_id;
}

static int rxd_comp_rx_msg_id(struct dlist_entry *item, const void *arg)
{
	struct rxd_x_entry *list_rx = container_of(item, struct rxd_x_entry, entry);
	struct rxd_x_entry *new_rx = container_of((struct dlist_entry *) arg,
						  struct rxd_x_entry, entry);

	return new_rx->msg_id > list_rx->msg_id;
}

static void rxd_progress_buf_cq(struct rxd_ep *ep, fi_addr_t peer)
{
	struct rxd_x_entry *rx_entry;

	dlist_foreach_container(&ep->peers[peer].buf_cq, struct rxd_x_entry,
				rx_entry, entry) {
		if (rx_entry->msg_id != ep->peers[peer].rx_msg_id + 1)
			return;

		rxd_complete_rx(ep, rx_entry);
	}
}

static void rxd_ep_recv_data(struct rxd_ep *ep, struct rxd_x_entry *rx_entry,
			     struct rxd_data_pkt *pkt, size_t size)
{
	struct rxd_domain *rxd_domain = rxd_ep_domain(ep);
	uint64_t done;

	done = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
			       rxd_domain->max_inline_sz +
			       (pkt->pkt_hdr.seg_no * rxd_domain->max_seg_sz),
			       pkt->msg, size - sizeof(struct rxd_data_pkt) -
			       ep->prefix_size);

	rx_entry->bytes_done += done;
	ep->peers[pkt->pkt_hdr.peer].rx_seq_no++;
	rx_entry->next_seg_no++;

	if (rx_entry->next_seg_no < rx_entry->num_segs) {
		if (!(ep->peers[pkt->pkt_hdr.peer].rx_seq_no %
		    ep->peers[pkt->pkt_hdr.peer].rx_window))
			rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);
		return;
	}
	rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);

	if (!rxd_env.retry && rx_entry->msg_id !=
	    ep->peers[pkt->pkt_hdr.peer].rx_msg_id + 1) {
		dlist_insert_order(&ep->peers[pkt->pkt_hdr.peer].buf_cq,
				   &rxd_comp_rx_msg_id, &rx_entry->entry);
		return;
	}

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);

	rxd_complete_rx(ep, rx_entry);
	if (!rxd_env.retry)
		rxd_progress_buf_cq(ep, pkt->pkt_hdr.peer);

	fastlock_release(&ep->util_ep.rx_cq->cq_lock);

	rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);
}

static int rxd_matching_ids(struct rxd_x_entry *rx_entry,
			    struct rxd_pkt_hdr *pkt_hdr)
{
	return (rx_entry->tx_id == pkt_hdr->tx_id) &&
	       (rx_entry->rx_id == pkt_hdr->rx_id) &&
	       (rx_entry->msg_id == pkt_hdr->msg_id);
}

static void rxd_transfer_pending(struct rxd_ep *ep, int peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_x_entry *tx_entry;
	struct rxd_data_pkt *data;
	struct rxd_op_pkt *op;

	while (ep->peers[peer].unacked_cnt < RXD_MAX_UNACKED &&
	       !dlist_empty(&ep->peers[peer].pending) &&
	       !ep->peers[peer].blocking) {
		dlist_pop_front(&ep->peers[peer].pending,
					struct rxd_pkt_entry, pkt_entry, d_entry); 
		dlist_insert_tail(&pkt_entry->d_entry, &ep->peers[peer].unacked);
		if (rxd_pkt_type(pkt_entry) == RXD_DATA) {
			data = (struct rxd_data_pkt *) (pkt_entry->pkt);
			tx_entry = &ep->tx_fs->entry[data->pkt_hdr.tx_id].buf;
			data->pkt_hdr.rx_id = tx_entry->rx_id;
		} else {
			op = (struct rxd_op_pkt *) (pkt_entry->pkt);
			if (op->size > rxd_ep_domain(ep)->max_inline_sz)
				ep->peers[peer].blocking = 1;
		}

		rxd_ep_retry_pkt(ep, pkt_entry);
		ep->peers[peer].pending_cnt--;
		ep->peers[peer].unacked_cnt++;
	}
}

static void rxd_update_peer(struct rxd_ep *ep, fi_addr_t peer, fi_addr_t dg_addr)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_data_pkt *data;
	struct rxd_op_pkt *op;

	ep->peers[peer].peer_addr = dg_addr;

	if (!dlist_empty(&ep->peers[peer].unacked)) {
		pkt_entry = container_of((&ep->peers[peer].unacked)->next,
					 struct rxd_pkt_entry, d_entry);
		if (rxd_pkt_type(pkt_entry) == RXD_RTS) {
			dlist_remove(&pkt_entry->d_entry);
			rxd_release_tx_pkt(ep, pkt_entry);
		}
	}

	dlist_foreach_container(&ep->peers[peer].pending, struct rxd_pkt_entry,
				pkt_entry, d_entry) {
		if (rxd_pkt_type(pkt_entry) == RXD_DATA) {
			data = (struct rxd_data_pkt *) (pkt_entry->pkt);
			data->pkt_hdr.peer = dg_addr;
		} else {
			op = (struct rxd_op_pkt *) (pkt_entry->pkt);
			op->pkt_hdr.peer = dg_addr;
		}
	}
	rxd_transfer_pending(ep, peer);
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
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, pkt->pkt_hdr.peer);
}

static int rxd_match_tmsg(struct dlist_entry *item, const void *arg)
{
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, pkt->pkt_hdr.peer) &&
	       rxd_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore, pkt->tag);
}

static void rxd_handle_data(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			    struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_data_pkt *pkt = (struct rxd_data_pkt *) (pkt_entry->pkt);
	struct rxd_x_entry *rx_entry;

	rx_entry = &ep->rx_fs->entry[pkt->pkt_hdr.rx_id].buf;

	if (!rxd_matching_ids(rx_entry, rxd_get_pkt_hdr(pkt_entry)) ||
	    rx_entry->op == RXD_NO_OP) {
	    	if (pkt->pkt_hdr.flags & RXD_LAST)
			rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);
		return;
	}

	if (pkt->pkt_hdr.seq_no == ep->peers[pkt->pkt_hdr.peer].rx_seq_no ||
	    !rxd_env.retry)
		rxd_ep_recv_data(ep, rx_entry, pkt, comp->len);
	else if (ep->peers[pkt->pkt_hdr.peer].last_ack_seq_no !=
		 ep->peers[pkt->pkt_hdr.peer].rx_seq_no) {
		rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);
	}
}

static struct rxd_x_entry *rxd_check_active(struct rxd_ep *ep, struct rxd_op_pkt *pkt)
{
	struct rxd_x_entry *rx_entry;

	//TODO - improve this search
	dlist_foreach_container(&ep->peers[pkt->pkt_hdr.peer].rx_list,
				struct rxd_x_entry, rx_entry, entry) {
		if (rx_entry->tx_id == pkt->pkt_hdr.tx_id &&
		    rx_entry->msg_id == pkt->pkt_hdr.msg_id &&
		    rx_entry->peer == pkt->pkt_hdr.peer)
			return rx_entry;
	}

	return NULL;
}

static void rxd_check_post_unexp(struct rxd_ep *ep, struct dlist_entry *list,
				 struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_pkt_entry *unexp;
	struct rxd_op_pkt *unexp_pkt;
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) (pkt_entry->pkt);

	if (!rxd_env.retry)
		goto insert;

	dlist_foreach_container(list, struct rxd_pkt_entry, unexp, d_entry) {
		unexp_pkt = (struct rxd_op_pkt *) (unexp->pkt);
		if (unexp_pkt->pkt_hdr.tx_id == pkt->pkt_hdr.tx_id &&
		    unexp_pkt->pkt_hdr.msg_id == pkt->pkt_hdr.msg_id &&
		    unexp_pkt->pkt_hdr.peer == pkt->pkt_hdr.peer) {
			rxd_release_repost_rx(ep, pkt_entry);
			return;
		}
	}

insert:
	dlist_insert_tail(&pkt_entry->d_entry, list);
}

static void rxd_handle_rts(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			   struct rxd_pkt_entry *pkt_entry)
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
		dlist_insert_tail(&ep->peers[dg_addr].entry, &ep->active_peers);
	}

	if (rxd_send_cts(ep, pkt, dg_addr)) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"error posting CTS\n");
	}
}

static struct rxd_x_entry *rxd_match_rx(struct rxd_ep *ep,
					struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) (pkt_entry->pkt);
	struct dlist_entry *unexp_list;
	struct dlist_entry *match;

	if (pkt->base_hdr.type == ofi_op_tagged) {
		match = dlist_remove_first_match(&ep->rx_tag_list, &rxd_match_tmsg,
					 (void *) pkt);
		unexp_list = &ep->unexp_tag_list;
	} else {
		match = dlist_remove_first_match(&ep->rx_list, &rxd_match_msg,
					 (void *) pkt);
		unexp_list = &ep->unexp_list;
	}

	if (match)
		return container_of(match, struct rxd_x_entry, entry);

	rxd_check_post_unexp(ep, unexp_list, pkt_entry);
	return NULL;

}

static void rxd_progress_buf_ops(struct rxd_ep *ep, fi_addr_t peer)
{
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_op_pkt *pkt;
	struct rxd_x_entry *rx_entry;

	while (!dlist_empty(&ep->peers[peer].buf_ops)) {
		pkt_entry = container_of((&ep->peers[peer].buf_ops)->next,
					struct rxd_pkt_entry, d_entry);
		pkt = (struct rxd_op_pkt *) (pkt_entry->pkt);
		if (pkt->pkt_hdr.seq_no != ep->peers[peer].rx_seq_no)
			return;

		rx_entry = rxd_match_rx(ep, pkt_entry);
		if (rx_entry) {
			rxd_progress_op(ep, pkt_entry, rx_entry);
			dlist_remove(&pkt_entry->d_entry);
			rxd_release_repost_rx(ep, pkt_entry);
		}
	}
}

void rxd_progress_op(struct rxd_ep *ep, struct rxd_pkt_entry *pkt_entry,
		     struct rxd_x_entry *rx_entry)
{
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) (pkt_entry->pkt);

	rx_entry->bytes_done = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
					0, pkt->msg, pkt->num_segs == 1 ? pkt->size :
					rxd_ep_domain(ep)->max_inline_sz);

	ep->peers[pkt->pkt_hdr.peer].rx_seq_no++;
	ep->peers[pkt->pkt_hdr.peer].curr_tx_id = pkt->pkt_hdr.tx_id;
	ep->peers[pkt->pkt_hdr.peer].curr_rx_id = rx_entry->rx_id;
	rx_entry->cq_entry.len = MIN(rx_entry->cq_entry.len, pkt->size);
	rx_entry->peer = pkt->pkt_hdr.peer;
	rx_entry->msg_id = pkt->pkt_hdr.msg_id;

	if (pkt->num_segs == 1) {
		rxd_complete_rx(ep, rx_entry);
		if (!rxd_env.retry)
			rxd_progress_buf_cq(ep, pkt->pkt_hdr.peer);
		return;
	}

	rx_entry->tx_id = pkt->pkt_hdr.tx_id;
	rx_entry->num_segs = pkt->num_segs;
	rx_entry->next_seg_no++;

	dlist_insert_tail(&rx_entry->entry, &ep->peers[pkt->pkt_hdr.peer].rx_list);
}

static void rxd_handle_op(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			  struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_x_entry *rx_entry;
	struct rxd_op_pkt *pkt = (struct rxd_op_pkt *) (pkt_entry->pkt);

	if (pkt->pkt_hdr.flags & RXD_RETRY) {
		rx_entry = rxd_check_active(ep, pkt);
		if (rx_entry)
			goto ack;
	}

	if (pkt->pkt_hdr.seq_no != ep->peers[pkt->pkt_hdr.peer].rx_seq_no) {
		if (!rxd_env.retry) {
			rxd_remove_rx_pkt(ep, pkt_entry);
			dlist_insert_order(&ep->peers[pkt->pkt_hdr.peer].buf_ops,
					   &rxd_comp_pkt_msg_id, &pkt_entry->d_entry);
			return;
		}

		if (ep->peers[pkt->pkt_hdr.peer].rx_seq_no !=
		    ep->peers[pkt->pkt_hdr.peer].last_ack_seq_no)
			goto ack;
		goto release;
	}

	rx_entry = rxd_match_rx(ep, pkt_entry);
	if (!rx_entry) {
		rxd_remove_rx_pkt(ep, pkt_entry);
		return;
	}

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
	rxd_progress_op(ep, pkt_entry, rx_entry);

	if (!rxd_env.retry && !dlist_empty(&ep->peers[pkt->pkt_hdr.peer].buf_ops))
		rxd_progress_buf_ops(ep, pkt->pkt_hdr.peer);

	fastlock_release(&ep->util_ep.rx_cq->cq_lock);

ack:
	rxd_ep_send_ack(ep, pkt->pkt_hdr.peer);
release:
	rxd_remove_rx_pkt(ep, pkt_entry);
	rxd_release_repost_rx(ep, pkt_entry);
}

static void rxd_handle_cts(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			   struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_cts_pkt *cts = (struct rxd_cts_pkt *) (pkt_entry->pkt);

	ep->peers[cts->peer_addr].peer_addr = cts->dg_addr;

	rxd_update_peer(ep, cts->peer_addr, cts->dg_addr);
}

static void rxd_handle_ack(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			   struct rxd_pkt_entry *ack_entry)
{
	struct rxd_x_entry *tx_entry;
	struct rxd_ack_pkt *ack = (struct rxd_ack_pkt *) (ack_entry->pkt);
	struct rxd_pkt_entry *pkt_entry;
	struct rxd_cq *tx_cq = rxd_ep_tx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.tx_cntr;
	fi_addr_t peer = ack->pkt_hdr.peer;
	struct rxd_pkt_hdr *hdr;

	while (!dlist_empty(&ep->peers[peer].unacked)) {
		pkt_entry = container_of((&ep->peers[peer].unacked)->next,
					struct rxd_pkt_entry, d_entry);
		hdr = rxd_get_pkt_hdr(pkt_entry);
		if (hdr->seq_no >= ack->pkt_hdr.seq_no)
			break;

		if (rxd_pkt_type(pkt_entry) != RXD_DATA) {
			tx_entry = &ep->tx_fs->entry[hdr->tx_id].buf;
			if (tx_entry->num_segs > 1) {
				tx_entry->rx_id = ack->pkt_hdr.rx_id;
				ep->peers[peer].blocking = 0;
			}
		}
		dlist_remove(&pkt_entry->d_entry);
		rxd_release_tx_pkt(ep, pkt_entry);
	     	ep->peers[peer].unacked_cnt--;
	}

	dlist_foreach_container(&ep->peers[ack->pkt_hdr.peer].tx_list,
				struct rxd_x_entry, tx_entry, entry) {
		if (tx_entry->start_seq + (tx_entry->num_segs - 1) >= ack->pkt_hdr.seq_no)
			break;

		if (!(tx_entry->flags & RXD_NO_COMPLETION)) {
			fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
			tx_cq->write_fn(tx_cq, &tx_entry->cq_entry);
			fastlock_release(&ep->util_ep.tx_cq->cq_lock);
			if (cntr)
				cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
		}
		rxd_tx_entry_free(ep, tx_entry);
	}

	rxd_transfer_pending(ep, ack->pkt_hdr.peer);

	dlist_foreach_container(&ep->peers[ack->pkt_hdr.peer].tx_list,
				struct rxd_x_entry, tx_entry, entry) {
		if (!rxd_ep_post_data_pkts(ep, tx_entry))
			break;
	}
} 

void rxd_handle_recv_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_entry *pkt_entry;
	int release = 1;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got recv completion\n");

	pkt_entry = container_of(comp->op_context, struct rxd_pkt_entry, context);
	ep->posted_bufs--;

	switch (rxd_pkt_type(pkt_entry)) {
	case RXD_RTS:
		rxd_handle_rts(ep, comp, pkt_entry);
		break;
	case RXD_CTS:
		rxd_handle_cts(ep, comp, pkt_entry);
		break;
	case RXD_ACK:
		rxd_handle_ack(ep, comp, pkt_entry);
		break;
	case RXD_DATA:
		rxd_handle_data(ep, comp, pkt_entry);
		break;
	default:
		rxd_handle_op(ep, comp, pkt_entry);
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
