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
	rx_entry->state = RXD_FREE;
	rx_entry->key = ~0;
	dlist_remove(&rx_entry->entry);
	freestack_push(ep->rx_fs, rx_entry);
}

void rxd_cq_report_error(struct rxd_cq *cq, struct fi_cq_err_entry *err_entry)
{
	struct fi_cq_tagged_entry cq_entry = {0};
	struct util_cq_err_entry *entry = calloc(1, sizeof(*entry));
	if (!entry) {
		FI_WARN(&rxd_prov, FI_LOG_CQ,
			"out of memory, cannot report CQ error\n");
		return;
	}

	entry->err_entry = *err_entry;
	slist_insert_tail(&entry->list_entry, &cq->util_cq.err_list);
	cq_entry.flags = UTIL_FLAG_ERROR;
	cq->write_fn(cq, &cq_entry);
}

void rxd_cq_report_tx_comp(struct rxd_cq *cq, struct rxd_x_entry *tx_entry)
{
	cq->write_fn(cq, &tx_entry->cq_entry);
}

static void rxd_ep_recv_data(struct rxd_ep *ep, struct rxd_x_entry *rx_entry,
			     struct rxd_pkt_entry *pkt_entry, size_t size)
{
	struct fi_cq_err_entry err_entry;
	struct rxd_cq *rx_cq = rxd_ep_rx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.rx_cntr;
	uint64_t done;

	done = ofi_copy_to_iov(rx_entry->iov, rx_entry->iov_count,
			       pkt_entry->pkt->hdr.seg_no * rx_entry->seg_size,
			       &pkt_entry->pkt->data,
			       size - sizeof(struct rxd_pkt_hdr) - ep->prefix_size);

	rx_entry->bytes_done += done;
	rx_entry->next_seg_no++;

	if (rx_entry->next_seg_no < rx_entry->num_segs) {
		if (rx_entry->next_seg_no == rx_entry->next_start) {
			rx_entry->next_start += rx_entry->window;
			rxd_ep_post_ack(ep, rx_entry);
		} else {
			rx_entry->state = RXD_CTS;
		}
		return;
	}
	rxd_ep_post_ack(ep, rx_entry);

	fastlock_acquire(&ep->util_ep.rx_cq->cq_lock);
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
	fastlock_release(&ep->util_ep.rx_cq->cq_lock);

	rxd_rx_entry_free(ep, rx_entry);
}

static int rxd_check_pkt_ids(struct rxd_x_entry *rx_entry,
			     struct rxd_pkt_entry *pkt_entry)
{
	if ((rx_entry->rx_id  ==
	     pkt_entry->pkt->hdr.rx_id) &&
	    (rx_entry->tx_id ==
	     pkt_entry->pkt->hdr.tx_id) &&
	    (rx_entry->key ==
	     pkt_entry->pkt->hdr.key))
		return 0;

	return -FI_EALREADY;
}

void rxd_post_cts(struct rxd_ep *rxd_ep, struct rxd_x_entry *rx_entry,
		  struct rxd_pkt_entry *rts_pkt)
{
	struct rxd_pkt_entry *pkt_entry;
	int ret;

	if (rx_entry->state == RXD_RTS) {
		rx_entry->peer = rts_pkt->peer;
		rx_entry->key = rts_pkt->pkt->hdr.key;
		rx_entry->tx_id = rts_pkt->pkt->hdr.tx_id;
		rx_entry->seg_size = rts_pkt->pkt->ctrl.seg_size;
		rx_entry->num_segs = ofi_div_ceil(rts_pkt->pkt->ctrl.size,
						  rts_pkt->pkt->ctrl.seg_size);
		rx_entry->window = rts_pkt->pkt->ctrl.window;
		rx_entry->next_start = rx_entry->window;

		if (rts_pkt->pkt->hdr.flags & RXD_REMOTE_CQ_DATA) {
			rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
			rx_entry->cq_entry.data = rts_pkt->pkt->ctrl.data;
		}
		rx_entry->cq_entry.len = rts_pkt->pkt->ctrl.size;

		rx_entry->state = RXD_ACK;
	}

	pkt_entry = rxd_get_tx_pkt(rxd_ep);
	if (!pkt_entry)
		return;

	rxd_init_ctrl_pkt(rxd_ep, rx_entry, pkt_entry, RXD_CTS);

	ret = rxd_ep_retry_pkt(rxd_ep, pkt_entry, rx_entry);
	if (ret)
		rxd_release_tx_pkt(rxd_ep, pkt_entry);
}

static int rxd_match_msg(struct dlist_entry *item, const void *arg)
{
	const struct rxd_pkt_entry *pkt_entry = (struct rxd_pkt_entry *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, pkt_entry->peer);
}

static int rxd_match_tmsg(struct dlist_entry *item, const void *arg)
{
	struct rxd_pkt_entry *pkt_entry = (struct rxd_pkt_entry *) arg;
	struct rxd_x_entry *rx_entry;

	rx_entry = container_of(item, struct rxd_x_entry, entry);

	return rxd_match_addr(rx_entry->peer, pkt_entry->peer) &&
	       rxd_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     pkt_entry->pkt->ctrl.tag);
}

static void rxd_handle_data(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			    struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_x_entry *rx_entry, tmp_entry;
	uint32_t pkt_seg_no = pkt_entry->pkt->hdr.seg_no;

	rx_entry = &ep->rx_fs->entry[pkt_entry->pkt->hdr.rx_id].buf;

	if (!(rx_entry->state == RXD_CTS || rx_entry->state == RXD_ACK) ||
	    rxd_check_pkt_ids(rx_entry, pkt_entry)) {
	    	if (!(pkt_entry->pkt->hdr.flags & RXD_LAST))
			return;

		tmp_entry.tx_id = pkt_entry->pkt->hdr.tx_id;
		tmp_entry.rx_id = pkt_entry->pkt->hdr.rx_id;
		tmp_entry.key = pkt_entry->pkt->hdr.key;
		tmp_entry.peer = pkt_entry->pkt->hdr.peer;
		tmp_entry.next_seg_no = pkt_entry->pkt->hdr.seg_no + 1;
		tmp_entry.window = 1;
		rxd_ep_post_ack(ep, &tmp_entry);
		return;
	}

	if (pkt_seg_no == rx_entry->next_seg_no || rxd_env.ooo_rdm)
		rxd_ep_recv_data(ep, rx_entry, pkt_entry, comp->len);
	else if (rx_entry->state != RXD_ACK ||
		(rx_entry->next_seg_no == rx_entry->next_start &&
		(pkt_seg_no + 1) == rx_entry->next_start))
		rxd_ep_post_ack(ep, rx_entry);
}

static int rxd_check_active(struct rxd_ep *ep, struct rxd_pkt_entry *rts_pkt)
{
	struct rxd_x_entry *rx_entry;
	struct dlist_entry *item;

	//TODO - improve this search
	dlist_foreach(&ep->active_rx_list, item) {
		rx_entry = container_of(item, struct rxd_x_entry, entry);
		if (rx_entry->tx_id == rts_pkt->pkt->hdr.tx_id &&
		    rx_entry->key == rts_pkt->pkt->hdr.key &&
		    rx_entry->peer == rts_pkt->peer) {
			rxd_post_cts(ep, rx_entry, rts_pkt);
			return 0;
		}
	}

	return -FI_ENOMSG;
}

static int rxd_check_post_unexp(struct dlist_entry *list,
				struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_pkt_entry *unexp;
	struct dlist_entry *item;

	dlist_foreach(list, item) {
		unexp = container_of(item, struct rxd_pkt_entry, d_entry);
		if (unexp->pkt->hdr.tx_id == pkt_entry->pkt->hdr.tx_id &&
		    unexp->pkt->hdr.key == pkt_entry->pkt->hdr.key &&
		    unexp->peer == pkt_entry->peer)
			return 0;
	}
	dlist_insert_tail(&pkt_entry->d_entry, list);
	return -FI_ENOMSG;
}

static int rxd_handle_rts(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			  struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_av *rxd_av;
	struct ofi_rbnode *node;
	struct rxd_x_entry *rx_entry;
	struct dlist_entry *match;
	struct dlist_entry *unexp_list;
	fi_addr_t dg_addr;
	int ret;

	rxd_av = rxd_ep_av(ep);
	node = ofi_rbmap_find(&rxd_av->rbmap, pkt_entry->pkt->source);

	if (node) {
		dg_addr = (fi_addr_t) node->data;
	} else {
		ret = rxd_av_insert_dg_addr(rxd_av, (void *) pkt_entry->pkt->source,
					    &dg_addr, 0, NULL);
		if (ret)
			return ret;
	}
	pkt_entry->peer = dg_addr;

	if (pkt_entry->pkt->hdr.flags & RXD_RETRY) {
		ret = rxd_check_active(ep, pkt_entry);
		if (!ret)
			return 0;
	}

	if (pkt_entry->pkt->ctrl.op == ofi_op_tagged) {
		match = dlist_remove_first_match(&ep->rx_tag_list, &rxd_match_tmsg,
					 (void *) pkt_entry);
		unexp_list = &ep->unexp_tag_list;
	} else {
		match = dlist_remove_first_match(&ep->rx_list, &rxd_match_msg,
					 (void *) pkt_entry);
		unexp_list = &ep->unexp_list;
	}
	if (!match)
		return rxd_check_post_unexp(unexp_list, pkt_entry);

	rx_entry = container_of(match, struct rxd_x_entry, entry);

	dlist_insert_tail(&rx_entry->entry, &ep->active_rx_list);

	rxd_post_cts(ep, rx_entry, pkt_entry);
	return 0;
}

static void rxd_handle_cts(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			    struct rxd_pkt_entry *cts_pkt)
{
	struct slist_entry *pkt_item;
	struct rxd_pkt_entry *pkt;
	struct rxd_x_entry *tx_entry;

	tx_entry = &ep->tx_fs->entry[cts_pkt->pkt->hdr.tx_id].buf;
	if (tx_entry->state != RXD_RTS ||
	    tx_entry->key != cts_pkt->pkt->hdr.key)
		return;

	tx_entry->state = RXD_CTS;
	tx_entry->rx_id = cts_pkt->pkt->hdr.rx_id;
	tx_entry->peer_x_addr = cts_pkt->pkt->hdr.peer;
	tx_entry->window = cts_pkt->pkt->ctrl.window;

	if (tx_entry->flags & RXD_INJECT) {
		pkt = container_of(slist_remove_head(&tx_entry->pkt_list),
				   struct rxd_pkt_entry, s_entry);
		rxd_release_tx_pkt(ep, pkt);

		for (pkt_item = tx_entry->pkt_list.head; pkt_item;
		     pkt_item = pkt_item->next) {
			pkt = container_of(pkt_item, struct rxd_pkt_entry, s_entry);
			pkt->pkt->hdr.rx_id = cts_pkt->pkt->hdr.rx_id;
			pkt->pkt->hdr.peer = cts_pkt->pkt->hdr.peer;
			if (rxd_ep_retry_pkt(ep, pkt, tx_entry))
				break;
		}
		rxd_set_timeout(tx_entry);
	} else {
		rxd_ep_free_acked_pkts(ep, tx_entry, 0);
		rxd_tx_entry_progress(ep, tx_entry, 1);
	}
}


static void rxd_handle_ack(struct rxd_ep *ep, struct fi_cq_msg_entry *comp,
			    struct rxd_pkt_entry *pkt_entry)
{
	struct rxd_cq *tx_cq = rxd_ep_tx_cq(ep);
	struct util_cntr *cntr = ep->util_ep.tx_cntr;
	struct rxd_x_entry *tx_entry;

	tx_entry = &ep->tx_fs->entry[pkt_entry->pkt->hdr.tx_id].buf;
	if (rxd_check_pkt_ids(tx_entry, pkt_entry) ||
	    tx_entry->state != RXD_CTS)
		return;

	rxd_ep_free_acked_pkts(ep, tx_entry, pkt_entry->pkt->hdr.seg_no - 1);
	if (!slist_empty(&tx_entry->pkt_list))
		return;

	if (!(tx_entry->flags & RXD_INJECT)) {
		rxd_tx_entry_progress(ep, tx_entry, 1);
		if (!slist_empty(&tx_entry->pkt_list))
			return;
	}

	if (!(tx_entry->flags & RXD_NO_COMPLETION)) {
		fastlock_acquire(&ep->util_ep.tx_cq->cq_lock);
		tx_cq->write_fn(tx_cq, &tx_entry->cq_entry);
		fastlock_release(&ep->util_ep.tx_cq->cq_lock);
		if (cntr)
			cntr->cntr_fid.ops->add(&cntr->cntr_fid, 1);
	}

	rxd_tx_entry_free(ep, tx_entry);
}

void rxd_handle_recv_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_entry *pkt_entry, *pkt_entry_head;
	struct slist_entry *item, *prev;
	int ret = 0;

	FI_DBG(&rxd_prov, FI_LOG_EP_CTRL, "got recv completion\n");

	pkt_entry = container_of(comp->op_context, struct rxd_pkt_entry, context);
	ep->posted_bufs--;

	if (rxd_is_ctrl_pkt(pkt_entry)) {
		switch (pkt_entry->pkt->ctrl.type) {
		case RXD_RTS:
			ret = rxd_handle_rts(ep, comp, pkt_entry);
			break;
		case RXD_CTS:
			rxd_handle_cts(ep, comp, pkt_entry);
			break;
		case RXD_ACK:
			rxd_handle_ack(ep, comp, pkt_entry);
			break;
		default:
			FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
				"Unknown message type\n");
			goto out;
		}
	} else {
		rxd_handle_data(ep, comp, pkt_entry);
	}

	pkt_entry_head = container_of(ep->rx_pkt_list.head,
				      struct rxd_pkt_entry, s_entry);
	if (pkt_entry_head != pkt_entry) {
		FI_WARN(&rxd_prov, FI_LOG_EP_CTRL,
			"matched to incorrect receive\n");
		slist_foreach(&ep->rx_pkt_list, item, prev) {
			pkt_entry_head = container_of(item, struct rxd_pkt_entry,
						      s_entry);
			if (pkt_entry_head == pkt_entry)
				break;
		}
		slist_remove(&ep->rx_pkt_list, item, prev);
	} else {
		slist_remove_head(&ep->rx_pkt_list);
	}

out:
	if (ret != -FI_ENOMSG) {
		rxd_release_rx_pkt(ep, pkt_entry);
		rxd_ep_post_buf(ep);
	}
}

void rxd_handle_send_comp(struct rxd_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct rxd_pkt_entry *pkt_entry;

	pkt_entry = container_of(comp->op_context, struct rxd_pkt_entry, context);

	if (!rxd_is_ctrl_pkt(pkt_entry) || pkt_entry->pkt->ctrl.type == RXD_RTS)
		return;

	rxd_release_tx_pkt(ep, pkt_entry);
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
