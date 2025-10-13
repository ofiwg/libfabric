/*
 * Copyright (c) 2013-2020 Intel Corporation. All rights reserved.
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
#include "uet.h"

/*
 * All EPs use the same underlying datagram provider, so pick any and use its
 * associated CQ.
 */
static const char *uet_cq_strerror(struct fid_cq *cq_fid, int prov_errno,
		const void *err_data, char *buf, size_t len)
{
	struct fid_list_entry *fid_entry;
	struct util_ep *util_ep;
	struct uet_cq *cq;
	struct uet_ep *ep;
	const char *str;

	cq = container_of(cq_fid, struct uet_cq, util_cq.cq_fid);

	ofi_genlock_lock(&cq->util_cq.ep_list_lock);
	assert(!dlist_empty(&cq->util_cq.ep_list));
	fid_entry = container_of(cq->util_cq.ep_list.next,
				struct fid_list_entry, entry);
	util_ep = container_of(fid_entry->fid, struct util_ep, ep_fid.fid);
	ep = container_of(util_ep, struct uet_ep, util_ep);

	str = fi_cq_strerror(ep->dg_cq, prov_errno, err_data, buf, len);
	ofi_genlock_unlock(&cq->util_cq.ep_list_lock);
	return str;
}

static int uet_cq_write(struct uet_cq *cq,
			struct fi_cq_tagged_entry *cq_entry)
{
	return ofi_cq_write(&cq->util_cq, cq_entry->op_context,
			    cq_entry->flags, cq_entry->len,
			    cq_entry->buf, cq_entry->data,
			    cq_entry->tag);
}

static int uet_cq_write_signal(struct uet_cq *cq,
			       struct fi_cq_tagged_entry *cq_entry)
{
	int ret = uet_cq_write(cq, cq_entry);
	cq->util_cq.wait->signal(cq->util_cq.wait);
	return ret;
}

void uet_rx_entry_free(struct uet_ep *ep, struct uet_x_entry *rx_entry)
{
	rx_entry->op <= UET_TAGGED ? ep->rx_msg_avail++ : ep->rx_rma_avail++;
	rx_entry->op = UET_NO_OP;
	dlist_remove(&rx_entry->entry);
	ofi_ibuf_free(rx_entry);
}

static int uet_match_pkt_entry(struct slist_entry *item, const void *arg)
{
	return ((struct uet_pkt_entry *) arg ==
		container_of(item, struct uet_pkt_entry, s_entry));
}

static void uet_remove_rx_pkt(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	struct slist_entry *item;

	item = slist_remove_first_match(&ep->rx_pkt_list, uet_match_pkt_entry,
					pkt_entry);
	if (!item) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
			"could not find posted rx to release\n");
	}
}

static void uet_complete_rx(struct uet_ep *ep, struct uet_x_entry *rx_entry)
{
	struct fi_cq_err_entry err_entry;
	struct uet_cq *rx_cq = uet_ep_rx_cq(ep);
	int ret;

	if (rx_entry->bytes_done != rx_entry->cq_entry.len) {
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = rx_entry->cq_entry.op_context;
		err_entry.flags = rx_entry->cq_entry.flags;
		err_entry.len = rx_entry->bytes_done;
		err_entry.err = FI_ETRUNC;
		err_entry.prov_errno = 0;
		ret = ofi_cq_write_error(&rx_cq->util_cq, &err_entry);
		if (ret) {
			FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not write error entry\n");
			return;
		}
		goto out;
	}

	if (rx_entry->cq_entry.flags & FI_REMOTE_CQ_DATA ||
	    (!(rx_entry->flags & UET_NO_RX_COMP) &&
	      rx_entry->cq_entry.flags & FI_RECV))
		rx_cq->write_fn(rx_cq, &rx_entry->cq_entry);

	ofi_ep_rx_cntr_inc_func(&ep->util_ep, (uint8_t) rx_entry->op);

out:
	uet_rx_entry_free(ep, rx_entry);
}

static void uet_complete_tx(struct uet_ep *ep, struct uet_x_entry *tx_entry)
{
	struct uet_cq *tx_cq = uet_ep_tx_cq(ep);

	if (!(tx_entry->flags & UET_NO_TX_COMP))
		tx_cq->write_fn(tx_cq, &tx_entry->cq_entry);

	ofi_ep_tx_cntr_inc_func(&ep->util_ep, (uint8_t) tx_entry->op);

	uet_tx_entry_free(ep, tx_entry);
}

static int uet_comp_pkt_seq_no(struct dlist_entry *item, const void *arg)
{
	struct uet_base_hdr *list_hdr;
	struct uet_base_hdr *new_hdr;

	list_hdr = uet_get_base_hdr(container_of(item,
				   struct uet_pkt_entry, d_entry));

	new_hdr = uet_get_base_hdr(container_of((struct dlist_entry *) arg,
				  struct uet_pkt_entry, d_entry));

	return new_hdr->seq_no > list_hdr->seq_no;
}

void uet_ep_recv_data(struct uet_ep *ep, struct uet_x_entry *x_entry,
		      struct uet_data_pkt *pkt, size_t size)
{
	struct uet_domain *uet_domain = uet_ep_domain(ep);
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
			       (pkt->ext_hdr.seg_no * uet_domain->max_seg_sz),
			       pkt->msg, size - sizeof(struct uet_data_pkt) -
			       ep->rx_prefix_size);

	x_entry->bytes_done += done;
	x_entry->next_seg_no++;

	if (x_entry->next_seg_no < x_entry->num_segs) {
		if (!(uet_peer(ep, pkt->base_hdr.peer)->rx_seq_no %
		    uet_peer(ep, pkt->base_hdr.peer)->rx_window))
			uet_ep_send_ack(ep, pkt->base_hdr.peer);
		return;
	}
	uet_ep_send_ack(ep, pkt->base_hdr.peer);

	if (x_entry->cq_entry.flags & FI_READ)
		uet_complete_tx(ep, x_entry);
	else
		uet_complete_rx(ep, x_entry);
}

static void uet_verify_active(struct uet_ep *ep, fi_addr_t addr, fi_addr_t peer_addr)
{
	struct uet_pkt_entry *pkt_entry;

	if (uet_peer(ep, addr)->peer_addr != UET_ADDR_INVALID &&
	    uet_peer(ep, addr)->peer_addr != peer_addr)
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
			"overwriting active peer - unexpected behavior\n");

	uet_peer(ep, addr)->peer_addr = peer_addr;

	if (!dlist_empty(&(uet_peer(ep, addr)->unacked)) &&
	    uet_get_base_hdr(container_of((&(uet_peer(ep, addr)->unacked))->next,
			     struct uet_pkt_entry, d_entry))->type == UET_RTS) {
		dlist_pop_front(&(uet_peer(ep, addr)->unacked),
				struct uet_pkt_entry, pkt_entry, d_entry);
		if (pkt_entry->flags & UET_PKT_IN_USE) {
			dlist_insert_tail(&pkt_entry->d_entry, &ep->ctrl_pkts);
			pkt_entry->flags |= UET_PKT_ACKED;
		} else {
			ofi_buf_free(pkt_entry);
			uet_peer(ep, addr)->unacked_cnt--;
		}
		dlist_remove(&(uet_peer(ep, addr)->entry));
	}

	if (!uet_peer(ep, addr)->active) {
		dlist_insert_tail(&(uet_peer(ep, addr)->entry),
				  &ep->active_peers);
		uet_peer(ep, addr)->retry_cnt = 0;
		uet_peer(ep, addr)->active = 1;
	}
}

int uet_start_xfer(struct uet_ep *ep, struct uet_x_entry *tx_entry)
{
	struct uet_base_hdr *hdr = uet_get_base_hdr(tx_entry->pkt);

	if (uet_peer(ep, tx_entry->peer)->unacked_cnt >=
	    uet_peer(ep, tx_entry->peer)->tx_window)
		return 0;

	tx_entry->start_seq = uet_set_pkt_seq(uet_peer(ep, tx_entry->peer),
					      tx_entry->pkt);
	if (tx_entry->op != UET_READ_REQ && tx_entry->num_segs > 1) {
		uet_peer(ep, tx_entry->peer)->tx_seq_no = tx_entry->start_seq +
						      tx_entry->num_segs;
	}
	hdr->peer = (uint32_t) uet_peer(ep, tx_entry->peer)->peer_addr;
	uet_ep_send_pkt(ep, tx_entry->pkt);
	uet_insert_unacked(ep, tx_entry->peer, tx_entry->pkt);
	tx_entry->pkt = NULL;

	if (tx_entry->op == UET_READ_REQ || tx_entry->op == UET_ATOMIC_FETCH ||
	    tx_entry->op == UET_ATOMIC_COMPARE) {
		dlist_remove(&tx_entry->entry);
		dlist_insert_tail(&tx_entry->entry,
				  &(uet_peer(ep, tx_entry->peer)->rma_rx_list));
	}

	return uet_peer(ep, tx_entry->peer)->unacked_cnt <
	       uet_peer(ep,tx_entry->peer)->tx_window;
}

void uet_progress_tx_list(struct uet_ep *ep, struct uet_peer *peer)
{
	struct dlist_entry *tmp_entry;
	struct uet_x_entry *tx_entry;
	uint64_t head_seq = peer->last_rx_ack;
	ssize_t ret = 0;
	int inc = 0;

	if (!dlist_empty(&peer->unacked)) {
		head_seq = uet_get_base_hdr(container_of(
					    (&peer->unacked)->next,
					    struct uet_pkt_entry, d_entry))->seq_no;
	}

	if (peer->peer_addr == UET_ADDR_INVALID)
		return;

	dlist_foreach_container_safe(&peer->tx_list, struct uet_x_entry,
				tx_entry, entry, tmp_entry) {
		if (tx_entry->pkt) {
			if (!uet_start_xfer(ep, tx_entry) ||
			    tx_entry->op == UET_READ_REQ)
				break;
		}

		if (tx_entry->bytes_done == tx_entry->cq_entry.len) {
			if (ofi_before(tx_entry->start_seq + (tx_entry->num_segs - 1),
			    head_seq)) {
				if (tx_entry->op == UET_DATA_READ) {
					tx_entry->op = UET_READ_REQ;
					uet_complete_rx(ep, tx_entry);
				} else {
					uet_complete_tx(ep, tx_entry);
				}
			}
			continue;
		}

		if (tx_entry->op == UET_DATA_READ && !tx_entry->bytes_done) {
			if (uet_peer(ep, tx_entry->peer)->unacked_cnt >=
		    	    uet_peer(ep, tx_entry->peer)->tx_window) {
				break;
			}
			tx_entry->start_seq = uet_peer(ep,tx_entry->peer)->tx_seq_no;
			uet_peer(ep, tx_entry->peer)->tx_seq_no = tx_entry->start_seq +
							      tx_entry->num_segs;
			inc = 1;
		}

		ret = uet_ep_post_data_pkts(ep, tx_entry);
		if (ret) {
			if (ret == -FI_ENOMEM && inc)
				uet_peer(ep, tx_entry->peer)->tx_seq_no -=
							  tx_entry->num_segs;
			break;
		}
	}

	if (dlist_empty(&peer->tx_list))
		peer->retry_cnt = 0;
}

static void uet_update_peer(struct uet_ep *ep, fi_addr_t peer, fi_addr_t peer_addr)
{
	uet_verify_active(ep, peer, peer_addr);
	uet_progress_tx_list(ep, uet_peer(ep, peer));
}

static int uet_send_cts(struct uet_ep *uet_ep, struct uet_rts_pkt *rts_pkt,
			fi_addr_t peer)
{
	struct uet_pkt_entry *pkt_entry;
	struct uet_cts_pkt *cts;
	int ret = 0;

	uet_update_peer(uet_ep, peer, rts_pkt->rts_addr);

	pkt_entry = uet_get_tx_pkt(uet_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	cts = (struct uet_cts_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*cts) + uet_ep->tx_prefix_size;
	pkt_entry->peer = peer;

	cts->base_hdr.version = UET_PROTOCOL_VERSION;
	cts->base_hdr.type = UET_CTS;
	cts->cts_addr = peer;
	cts->rts_addr = rts_pkt->rts_addr;

	dlist_insert_tail(&pkt_entry->d_entry, &uet_ep->ctrl_pkts);
	ret = uet_ep_send_pkt(uet_ep, pkt_entry);
	if (ret)
		uet_remove_free_pkt_entry(pkt_entry);

	return ret;
}

static int uet_match_msg(struct dlist_entry *item, const void *arg)
{
	struct uet_match_attr *attr = (struct uet_match_attr *) arg;
	struct uet_x_entry *rx_entry;

	rx_entry = container_of(item, struct uet_x_entry, entry);

	return uet_match_addr(rx_entry->peer, attr->peer);
}

static int uet_match_tmsg(struct dlist_entry *item, const void *arg)
{
	struct uet_match_attr *attr = (struct uet_match_attr *) arg;
	struct uet_x_entry *rx_entry;

	rx_entry = container_of(item, struct uet_x_entry, entry);

	return uet_match_addr(rx_entry->peer, attr->peer) &&
	       uet_match_tag(rx_entry->cq_entry.tag, rx_entry->ignore,
			     attr->tag);
}

static struct uet_unexp_msg *uet_init_unexp(struct uet_ep *ep,
					    struct uet_pkt_entry *pkt_entry,
					    struct uet_base_hdr *base_hdr,
					    struct uet_sar_hdr *sar_hdr,
			 		    struct uet_tag_hdr *tag_hdr,
					    struct uet_data_hdr *data_hdr,
					    void *msg, size_t msg_size)
{
	struct uet_unexp_msg *unexp_msg;

	unexp_msg = calloc(1, sizeof(*unexp_msg));
	if (!unexp_msg)
		return NULL;

	unexp_msg->pkt_entry = pkt_entry;
	unexp_msg->base_hdr = base_hdr;
	unexp_msg->sar_hdr = sar_hdr;
	unexp_msg->tag_hdr = tag_hdr;
	unexp_msg->data_hdr = data_hdr;
	unexp_msg->msg_size = msg_size;
	unexp_msg->msg = msg;

	dlist_init(&unexp_msg->pkt_list);

	return unexp_msg;
}

static void uet_handle_rts(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	struct uet_av *uet_av;
	struct ofi_rbnode *node;
	fi_addr_t uet_addr;
	struct uet_rts_pkt *pkt = (struct uet_rts_pkt *) (pkt_entry->pkt);
	int ret;

	if (pkt->base_hdr.version != UET_PROTOCOL_VERSION) {
		FI_WARN(&uet_prov, FI_LOG_CQ,
			"ERROR: Protocol version mismatch with peer\n");
		return;
	}

	uet_av = uet_ep_av(ep);
	node = ofi_rbmap_find(&uet_av->rbmap, pkt->source);

	if (node) {
		uet_addr = (fi_addr_t) node->data;
	} else {
		ret = uet_av_insert_dg_addr(uet_av, (void *) pkt->source,
					    &uet_addr, 0, NULL);
		if (ret)
			return;
	}

	if (!uet_peer(ep, uet_addr)) {
		if (uet_create_peer(ep, uet_addr) < 0)
			return;
	}

	if (uet_send_cts(ep, pkt, uet_addr)) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
			"error posting CTS\n");
	}
}

struct uet_x_entry *uet_progress_multi_recv(struct uet_ep *ep,
					   struct uet_x_entry *rx_entry,
					   size_t total_size)
{
	struct uet_x_entry *dup_entry;
	size_t left;
	uint32_t dup_id;

	left = rx_entry->iov[0].iov_len - total_size;

	if (left < ep->min_multi_recv_size) {
		rx_entry->cq_entry.flags |= FI_MULTI_RECV;
		return NULL;
	}

	dup_entry = uet_get_rx_entry(ep, rx_entry->op);
	if (!dup_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get rx entry\n");
		return NULL;
	}
	dup_id = dup_entry->rx_id;
	memcpy(dup_entry, rx_entry, sizeof(*rx_entry));
	dup_entry->rx_id = (uint16_t) dup_id;
	dup_entry->iov[0].iov_base = rx_entry->iov[0].iov_base;
	dup_entry->iov[0].iov_len = total_size;
	dup_entry->cq_entry.len = total_size;

	rx_entry->iov[0].iov_base = (char *) rx_entry->iov[0].iov_base + total_size;
	rx_entry->cq_entry.buf = rx_entry->iov[0].iov_base;
	rx_entry->iov[0].iov_len = left;
	rx_entry->cq_entry.len = left;

	return dup_entry;
}

static struct uet_x_entry *uet_match_rx(struct uet_ep *ep,
					struct uet_pkt_entry *pkt_entry,
					struct uet_base_hdr *base,
					struct uet_tag_hdr *tag,
					struct uet_sar_hdr *op,
					struct uet_data_hdr *data,
					void *msg, size_t msg_size)
{
	struct uet_x_entry *rx_entry, *dup_entry;
	struct uet_unexp_msg *unexp_msg;
	struct dlist_entry *rx_list;
	struct dlist_entry *unexp_list;
	struct dlist_entry *match;
	struct uet_match_attr attr;
	size_t total_size;

	attr.peer = base->peer;

	if (tag) {
		attr.tag = tag->tag;
		rx_list = &ep->rx_tag_list;
		match = dlist_find_first_match(rx_list, &uet_match_tmsg,
					 (void *) &attr);
		unexp_list = &ep->unexp_tag_list;
	} else {
		attr.tag = 0;
		rx_list = &ep->rx_list;
		match = dlist_find_first_match(rx_list, &uet_match_msg,
					 (void *) &attr);
		unexp_list = &ep->unexp_list;
	}

	if (!match) {
		assert(!uet_peer(ep, base->peer)->curr_unexp);
		unexp_msg = uet_init_unexp(ep, pkt_entry, base, op,
					   tag, data, msg, msg_size);
		if (unexp_msg) {
			dlist_insert_tail(&unexp_msg->entry, unexp_list);
			uet_peer(ep, base->peer)->curr_unexp = unexp_msg;
		}
		return NULL;
	}

	rx_entry = container_of(match, struct uet_x_entry, entry);
	total_size = op ? op->size : msg_size;

	if (rx_entry->flags & UET_MULTI_RECV) {
		dup_entry = uet_progress_multi_recv(ep, rx_entry, total_size);
		if (!dup_entry)
			goto out;

		dup_entry->start_seq = base->seq_no;
		dlist_init(&dup_entry->entry);
		return dup_entry;
	}

out:
	dlist_remove(&rx_entry->entry);
	rx_entry->cq_entry.len = MIN(rx_entry->cq_entry.len, total_size);
	return rx_entry;
}

static int uet_verify_iov(struct uet_ep *ep, struct ofi_rma_iov *rma,
			  size_t count, uint32_t type, struct iovec *iov)
{
	struct util_domain *util_domain = &uet_ep_domain(ep)->util_domain;
	int i, ret;

	for (i = 0; i < count; i++) {
		ret = ofi_mr_verify(&util_domain->mr_map, rma[i].len,
			(uintptr_t *)(&rma[i].addr), rma[i].key,
			ofi_rx_mr_reg_flags(type, 0));
		iov[i].iov_base = (void *) rma[i].addr;
		iov[i].iov_len = rma[i].len;
		if (ret) {
			FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not verify MR\n");
			return -FI_EACCES;
		}
	}
	return 0;
}

static struct uet_x_entry *uet_rma_read_entry_init(struct uet_ep *ep,
			struct uet_base_hdr *base_hdr, struct uet_sar_hdr *sar_hdr,
			struct uet_rma_hdr *rma_hdr)
{
	struct uet_x_entry *rx_entry;
	struct uet_domain *uet_domain = uet_ep_domain(ep);
	int ret;

	rx_entry = uet_get_rx_entry(ep, base_hdr->type);
	if (!rx_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get rx entry\n");
		return NULL;
	}

	rx_entry->tx_id = (uint16_t)sar_hdr->tx_id;
	rx_entry->op = UET_DATA_READ;
	rx_entry->peer = base_hdr->peer;
	rx_entry->flags = UET_NO_TX_COMP;
	rx_entry->bytes_done = 0;
	rx_entry->next_seg_no = 0;
	rx_entry->num_segs = ofi_div_ceil(sar_hdr->size, uet_domain->max_seg_sz);
	rx_entry->pkt = NULL;

 	ret = uet_verify_iov(ep, rma_hdr->rma, sar_hdr->iov_count,
			     base_hdr->type, rx_entry->iov);
	if (ret)
		return NULL;

	rx_entry->iov_count = sar_hdr->iov_count;
	rx_entry->cq_entry.flags = ofi_rx_cq_flags(UET_READ_REQ);
	rx_entry->cq_entry.len = sar_hdr->size;

	dlist_insert_tail(&rx_entry->entry, &(uet_peer(ep, rx_entry->peer)->tx_list));

	uet_progress_tx_list(ep, uet_peer(ep, rx_entry->peer));

	return rx_entry;
}

static struct uet_x_entry *uet_rma_rx_entry_init(struct uet_ep *ep,
			struct uet_base_hdr *base_hdr, struct uet_sar_hdr *sar_hdr,
			struct uet_rma_hdr *rma_hdr)
{
	struct uet_x_entry *rx_entry;
	struct iovec iov[UET_IOV_LIMIT];
	int ret, iov_count;

	iov_count = sar_hdr ? sar_hdr->iov_count : 1;
	ret = uet_verify_iov(ep, rma_hdr->rma, iov_count,
			     base_hdr->type, iov);
	if (ret)
		return NULL;

	rx_entry = uet_rx_entry_init(ep, iov, iov_count, 0, 0, NULL,
				     base_hdr->peer, base_hdr->type,
				     base_hdr->flags);
	if (!rx_entry)
		return NULL;

	rx_entry->start_seq = base_hdr->seq_no;

	return rx_entry;
}

static struct uet_x_entry *uet_rx_atomic_fetch(struct uet_ep *ep,
			struct uet_base_hdr *base_hdr,
			struct uet_sar_hdr *sar_hdr,
			struct uet_rma_hdr *rma_hdr,
			struct uet_atom_hdr *atom_hdr)
{
	struct uet_x_entry *rx_entry;
	int ret;

	rx_entry = uet_get_rx_entry(ep, base_hdr->type);
	if (!rx_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get tx entry\n");
		return NULL;
	}

	rx_entry->pkt = uet_get_tx_pkt(ep);
	if (!rx_entry->pkt) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get pkt\n");
		uet_rx_entry_free(ep, rx_entry);
		return NULL;
	}
	rx_entry->tx_id = (uint16_t) sar_hdr->tx_id;

	rx_entry->op = UET_DATA_READ;
	rx_entry->peer = base_hdr->peer;
	rx_entry->flags = UET_NO_TX_COMP;
	rx_entry->bytes_done = 0;
	rx_entry->next_seg_no = 0;
	rx_entry->num_segs = 1;

	rx_entry->iov_count = sar_hdr->iov_count;
 	ret = uet_verify_iov(ep, rma_hdr->rma, rx_entry->iov_count,
			     base_hdr->type, rx_entry->iov);
	if (ret)
		return NULL;

	rx_entry->cq_entry.flags = ofi_rx_cq_flags(UET_ATOMIC_FETCH);
	rx_entry->cq_entry.len = sar_hdr->size;

	uet_init_data_pkt(ep, rx_entry, rx_entry->pkt);
	if (rx_entry->bytes_done != rx_entry->cq_entry.len)
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "fetch data length mismatch\n");

	dlist_insert_tail(&rx_entry->entry, &(uet_peer(ep, rx_entry->peer)->tx_list));

	uet_ep_send_ack(ep, base_hdr->peer);

	uet_progress_tx_list(ep, uet_peer(ep, rx_entry->peer));

	return rx_entry;
}

static int uet_unpack_hdrs(size_t pkt_size, struct uet_base_hdr *base_hdr,
			   struct uet_sar_hdr **sar_hdr, struct uet_tag_hdr **tag_hdr,
			   struct uet_data_hdr **data_hdr, struct uet_rma_hdr **rma_hdr,
			   struct uet_atom_hdr **atom_hdr, void **msg, size_t *msg_size)
{
	char *ptr = (char *) base_hdr + sizeof(*base_hdr);
	uint8_t rma_count = 1;

	if (base_hdr->flags & UET_TAG_HDR) {
		*tag_hdr = (struct uet_tag_hdr *) ptr;
		ptr += sizeof(**tag_hdr);
	} else {
		*tag_hdr = NULL;
	}

	if (base_hdr->flags & UET_REMOTE_CQ_DATA) {
		*data_hdr = (struct uet_data_hdr *) ptr;
		ptr += sizeof(**data_hdr);
	} else {
		*data_hdr = NULL;
	}

	if (!(base_hdr->flags & UET_INLINE)) {
		*sar_hdr = (struct uet_sar_hdr *) ptr;
		rma_count = (*sar_hdr)->iov_count;
		ptr += sizeof(**sar_hdr);
	} else {
		if (base_hdr->type == UET_READ_REQ ||
		    base_hdr->type == UET_ATOMIC_FETCH)
			goto err;
		*sar_hdr = NULL;
	}

	if (base_hdr->type >= UET_READ_REQ && base_hdr->type <= UET_ATOMIC_COMPARE) {
		*rma_hdr = (struct uet_rma_hdr *) ptr;
		ptr += (sizeof(*(*rma_hdr)->rma) * rma_count);

		if (base_hdr->type >= UET_ATOMIC) {
			*atom_hdr = (struct uet_atom_hdr *) ptr;
			ptr += sizeof(**atom_hdr);
		} else {
			*atom_hdr = NULL;
		}
	} else {
		*rma_hdr = NULL;
		*atom_hdr = NULL;
	}

	if (pkt_size < (size_t)(ptr - (char *) base_hdr))
		goto err;

	*msg = ptr;
	*msg_size = pkt_size - (ptr - (char *) base_hdr);

	return 0;

err:
	FI_WARN(&uet_prov, FI_LOG_CQ, "Cannot process packet\n");
	return -FI_EINVAL;
}

static int uet_unpack_init_rx(struct uet_ep *ep, struct uet_x_entry **rx_entry,
			      struct uet_pkt_entry *pkt_entry,
			      struct uet_base_hdr *base_hdr,
			      struct uet_sar_hdr **sar_hdr,
			      struct uet_tag_hdr **tag_hdr,
			      struct uet_data_hdr **data_hdr,
			      struct uet_rma_hdr **rma_hdr,
			      struct uet_atom_hdr **atom_hdr,
			      void **msg, size_t *msg_size)
{
	int ret;

	ret = uet_unpack_hdrs(pkt_entry->pkt_size - ep->rx_prefix_size, base_hdr, sar_hdr,
			      tag_hdr, data_hdr, rma_hdr, atom_hdr, msg, msg_size);
	if (ret)
		return ret;

	switch (base_hdr->type) {
	case UET_MSG:
	case UET_TAGGED:
		*rx_entry = uet_match_rx(ep, pkt_entry, base_hdr, *tag_hdr, *sar_hdr,
					*data_hdr, *msg, *msg_size);
		break;
	case UET_READ_REQ:
		*rx_entry = uet_rma_read_entry_init(ep, base_hdr, *sar_hdr, *rma_hdr);
		break;
	case UET_ATOMIC_FETCH:
	case UET_ATOMIC_COMPARE:
		*rx_entry = uet_rx_atomic_fetch(ep, base_hdr, *sar_hdr, *rma_hdr, *atom_hdr);
		break;
	default:
		*rx_entry = uet_rma_rx_entry_init(ep, base_hdr, *sar_hdr, *rma_hdr);
	}

	return 0;
}

void uet_do_atomic(void *src, void *dst, void *cmp, enum fi_datatype datatype,
		   enum fi_op atomic_op, size_t cnt)
{
	char tmp_result[UET_MAX_MTU_SIZE];

	if (ofi_atomic_isswap_op(atomic_op)) {
		ofi_atomic_swap_handler(atomic_op, datatype, dst, src, cmp,
					tmp_result, cnt);
	} else if (ofi_atomic_iswrite_op(atomic_op)) {
		ofi_atomic_write_handler(atomic_op, datatype, dst, src, cnt);
	}
}

void uet_progress_op_msg(struct uet_ep *ep, struct uet_x_entry *rx_entry,
			 void **msg, size_t size)
{
	rx_entry->bytes_done = ofi_copy_to_iov(rx_entry->iov,
					       rx_entry->iov_count, 0, *msg, size);
}

void uet_progress_atom_op(struct uet_ep *ep, struct uet_x_entry *rx_entry,
			  struct uet_base_hdr *base_hdr, struct uet_sar_hdr *sar_hdr,
			  struct uet_rma_hdr *rma_hdr, struct uet_atom_hdr *atom_hdr,
			  void **msg, size_t msg_size)
{
	char *src, *cmp;
	size_t data_size, len;
	int i, iov_count;

	src = (char *) (*msg);
	cmp = base_hdr->type == UET_ATOMIC_COMPARE ? src + (msg_size / 2) : NULL;
	iov_count = sar_hdr ? sar_hdr->iov_count : 1;

	data_size = ofi_datatype_size(atom_hdr->datatype);
	if (!data_size) {
		FI_WARN(&uet_prov, FI_LOG_EP_DATA,
			"Invalid atomic datatype received\n");
		len = ofi_total_iov_len(rx_entry->iov, iov_count);
		goto out;
	}

	for (i = 0, len = 0; i < iov_count; i++) {
		uet_do_atomic(&src[len], rx_entry->iov[i].iov_base,
			      cmp ? &cmp[len] : NULL,
			      atom_hdr->datatype, atom_hdr->atomic_op,
			      rx_entry->iov[i].iov_len / data_size);
		len += rx_entry->iov[i].iov_len;
	}

out:
	if (base_hdr->type == UET_ATOMIC)
		rx_entry->bytes_done = len;
}

void uet_progress_op(struct uet_ep *ep, struct uet_x_entry *rx_entry,
		     struct uet_pkt_entry *pkt_entry,
		     struct uet_base_hdr *base_hdr,
		     struct uet_sar_hdr *sar_hdr,
		     struct uet_tag_hdr *tag_hdr,
		     struct uet_data_hdr *data_hdr,
		     struct uet_rma_hdr *rma_hdr,
		     struct uet_atom_hdr *atom_hdr,
		     void **msg, size_t size)
{
	if (sar_hdr)
		uet_peer(ep, base_hdr->peer)->curr_tx_id =
			(uint16_t) sar_hdr->tx_id;

	uet_peer(ep, base_hdr->peer)->curr_rx_id = rx_entry->rx_id;

	if (base_hdr->type == UET_READ_REQ)
		return;

	if (atom_hdr)
		uet_progress_atom_op(ep, rx_entry, base_hdr, sar_hdr,
				     rma_hdr, atom_hdr, msg, size);
	else
		uet_progress_op_msg(ep, rx_entry, msg, size);

	rx_entry->offset = rx_entry->bytes_done;

	if (data_hdr) {
		rx_entry->cq_entry.flags |= FI_REMOTE_CQ_DATA;
		rx_entry->cq_entry.data = data_hdr->cq_data;
	}

	rx_entry->peer = base_hdr->peer;

	if (tag_hdr)
		rx_entry->cq_entry.tag = tag_hdr->tag;

	if (!sar_hdr || sar_hdr->num_segs == 1) {
		if (!(rx_entry->cq_entry.flags & FI_REMOTE_READ))
			uet_complete_rx(ep, rx_entry);
		return;
	}

	rx_entry->tx_id = (uint16_t) sar_hdr->tx_id;
	rx_entry->num_segs = sar_hdr->num_segs;
	rx_entry->next_seg_no++;
	rx_entry->start_seq = base_hdr->seq_no;

	dlist_insert_tail(&rx_entry->entry, &(uet_peer(ep, base_hdr->peer)->rx_list));
}

static struct uet_x_entry *uet_get_data_x_entry(struct uet_ep *ep,
			struct uet_data_pkt *data_pkt)
{
	if (data_pkt->base_hdr.type == UET_DATA)
		return ofi_bufpool_get_ibuf(ep->rx_entry_pool.pool,
			     uet_peer(ep, data_pkt->base_hdr.peer)->curr_rx_id);

	return ofi_bufpool_get_ibuf(ep->tx_entry_pool.pool, data_pkt->ext_hdr.tx_id);
}

static void uet_progress_buf_pkts(struct uet_ep *ep, fi_addr_t peer)
{
	struct fi_cq_err_entry err_entry;
	struct uet_pkt_entry *pkt_entry;
	struct uet_base_hdr *base_hdr;
	struct uet_sar_hdr *sar_hdr;
	struct uet_tag_hdr *tag_hdr;
	struct uet_data_hdr *data_hdr;
	struct uet_rma_hdr *rma_hdr;
	struct uet_atom_hdr *atom_hdr;
	void *msg;
	int ret;
	size_t msg_size;
	struct uet_x_entry *rx_entry = NULL;
	struct uet_data_pkt *data_pkt;
	struct dlist_entry *bufpkts;

	bufpkts = &(uet_peer(ep, peer)->buf_pkts);
	while (!dlist_empty(bufpkts)) {
		pkt_entry = container_of(bufpkts->next, struct uet_pkt_entry,
					 d_entry);
		base_hdr = uet_get_base_hdr(pkt_entry);
		if (base_hdr->seq_no != uet_peer(ep, peer)->rx_seq_no)
			return;
		if (base_hdr->type == UET_DATA || base_hdr->type == UET_DATA_READ) {
			data_pkt = (struct uet_data_pkt *) pkt_entry->pkt;
			rx_entry = uet_get_data_x_entry(ep, data_pkt);
			uet_ep_recv_data(ep, rx_entry, data_pkt, pkt_entry->pkt_size);
		} else {
			ret = uet_unpack_init_rx(ep, &rx_entry, pkt_entry, base_hdr, &sar_hdr,
					      &tag_hdr, &data_hdr, &rma_hdr, &atom_hdr,
					      &msg, &msg_size);
			if (ret) {
				memset(&err_entry, 0, sizeof(err_entry));
				err_entry.err = FI_ETRUNC;
				err_entry.prov_errno = 0;
				ret = ofi_cq_write_error(&uet_ep_rx_cq(ep)->util_cq,
							 &err_entry);
				if (ret)
					FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
						"could not write error entry\n");
				uet_peer(ep, base_hdr->peer)->rx_seq_no++;
				uet_remove_free_pkt_entry(pkt_entry);
				continue;
			}
			if (!rx_entry) {
				if (base_hdr->type == UET_MSG ||
				    base_hdr->type == UET_TAGGED) {
					uet_peer(ep, base_hdr->peer)->rx_seq_no++;
					continue;
				}
				break;
			}

			uet_progress_op(ep, rx_entry, pkt_entry, base_hdr,
					sar_hdr, tag_hdr, data_hdr, rma_hdr,
					atom_hdr, &msg, msg_size);
		}

		uet_peer(ep,base_hdr->peer)->rx_seq_no++;
		uet_remove_free_pkt_entry(pkt_entry);
	}
}

static void uet_handle_data(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	struct uet_data_pkt *pkt = (struct uet_data_pkt *) (pkt_entry->pkt);
	struct uet_x_entry *x_entry;
	struct uet_unexp_msg *unexp_msg;

	if (pkt_entry->pkt_size < sizeof(*pkt) + ep->rx_prefix_size) {
		FI_WARN(&uet_prov, FI_LOG_CQ,
			"Cannot process packet smaller than minimum header size\n");
		goto free;
	}

	if (pkt->base_hdr.seq_no == uet_peer(ep,
				    pkt->base_hdr.peer)->rx_seq_no) {
		uet_peer(ep, pkt->base_hdr.peer)->rx_seq_no++;
		if (pkt->base_hdr.type == UET_DATA &&
		    uet_peer(ep, pkt->base_hdr.peer)->curr_unexp) {
			unexp_msg = uet_peer(ep, pkt->base_hdr.peer)->curr_unexp;
			dlist_insert_tail(&pkt_entry->d_entry, &unexp_msg->pkt_list);
			if (pkt->ext_hdr.seg_no + 1 == unexp_msg->sar_hdr->num_segs - 1) {
				uet_peer(ep, pkt->base_hdr.peer)->curr_unexp = NULL;
				uet_ep_send_ack(ep, pkt->base_hdr.peer);
			}
			return;
		}
		x_entry = uet_get_data_x_entry(ep, pkt);
		uet_ep_recv_data(ep, x_entry, pkt, pkt_entry->pkt_size);
		if (!dlist_empty(&(uet_peer(ep,
				   pkt->base_hdr.peer)->buf_pkts)))
			uet_progress_buf_pkts(ep, pkt->base_hdr.peer);
	} else if (!uet_env.retry) {
		dlist_insert_order(&(uet_peer(ep,
				     pkt->base_hdr.peer)->buf_pkts),
				   &uet_comp_pkt_seq_no, &pkt_entry->d_entry);
		return;
	} else if (uet_peer(ep, pkt->base_hdr.peer)->peer_addr !=
		   UET_ADDR_INVALID) {
		uet_ep_send_ack(ep, pkt->base_hdr.peer);
	}
free:
	ofi_buf_free(pkt_entry);
}

static void uet_handle_op(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	struct uet_x_entry *rx_entry;
	struct uet_base_hdr *base_hdr = uet_get_base_hdr(pkt_entry);
	struct uet_sar_hdr *sar_hdr;
	struct uet_tag_hdr *tag_hdr;
	struct uet_data_hdr *data_hdr;
	struct uet_rma_hdr *rma_hdr;
	struct uet_atom_hdr *atom_hdr;
	void *msg;
	size_t msg_size;
	int ret;

	if (base_hdr->seq_no != uet_peer(ep, base_hdr->peer)->rx_seq_no) {
		if (!uet_env.retry) {
			dlist_insert_order(&(uet_peer(ep, base_hdr->peer)->buf_pkts),
					   &uet_comp_pkt_seq_no, &pkt_entry->d_entry);
			return;
		}

		if (uet_peer(ep, base_hdr->peer)->peer_addr != UET_ADDR_INVALID)
			goto ack;
		goto release;
	}

	if (uet_peer(ep, base_hdr->peer)->peer_addr == UET_ADDR_INVALID)
		goto release;

	ret = uet_unpack_init_rx(ep, &rx_entry, pkt_entry, base_hdr, &sar_hdr,
				 &tag_hdr, &data_hdr, &rma_hdr, &atom_hdr,
				 &msg, &msg_size);
	if (ret)
		goto ack;

	if (!rx_entry) {
		if (base_hdr->type == UET_MSG || base_hdr->type == UET_TAGGED) {
			if (!uet_peer(ep, base_hdr->peer)->curr_unexp)
				goto ack;

			uet_peer(ep, base_hdr->peer)->rx_seq_no++;

			if (!sar_hdr)
				uet_peer(ep, base_hdr->peer)->curr_unexp = NULL;

			uet_ep_send_ack(ep, base_hdr->peer);
			return;
		}
		uet_peer(ep, base_hdr->peer)->rx_window = 0;
		goto ack;
	}

	uet_peer(ep, base_hdr->peer)->rx_seq_no++;
	uet_peer(ep, base_hdr->peer)->rx_window = (uint16_t) uet_env.max_unacked;
	uet_progress_op(ep, rx_entry, pkt_entry, base_hdr, sar_hdr, tag_hdr,
			data_hdr, rma_hdr, atom_hdr, &msg, msg_size);

	if (!dlist_empty(&(uet_peer(ep, base_hdr->peer)->buf_pkts)))
		uet_progress_buf_pkts(ep, base_hdr->peer);

ack:
	uet_ep_send_ack(ep, base_hdr->peer);
release:
	ofi_buf_free(pkt_entry);
}

static void uet_handle_cts(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	struct uet_cts_pkt *cts = (struct uet_cts_pkt *) (pkt_entry->pkt);

	if (cts->base_hdr.version != UET_PROTOCOL_VERSION) {
		FI_WARN(&uet_prov, FI_LOG_CQ,
			"ERROR: Protocol version mismatch with peer\n");
		return;
	}

	uet_update_peer(ep, cts->rts_addr, cts->cts_addr);
}

static void uet_handle_ack(struct uet_ep *ep, struct uet_pkt_entry *ack_entry)
{
	struct uet_ack_pkt *ack = (struct uet_ack_pkt *) (ack_entry->pkt);
	struct uet_pkt_entry *pkt_entry;
	fi_addr_t peer = ack->base_hdr.peer;
	struct uet_base_hdr *hdr;

	uet_peer(ep, peer)->tx_window = (uint16_t) ack->ext_hdr.rx_id;

	if (uet_peer(ep, peer)->last_rx_ack == ack->base_hdr.seq_no)
		return;

	uet_peer(ep, peer)->last_rx_ack = ack->base_hdr.seq_no;

	if (dlist_empty(&(uet_peer(ep, peer)->unacked)))
		return;

	pkt_entry = container_of((&(uet_peer(ep,
				    peer)->unacked))->next,
				 struct uet_pkt_entry, d_entry);

	while (&pkt_entry->d_entry != &(uet_peer(ep,
				        peer)->unacked)) {
		hdr = uet_get_base_hdr(pkt_entry);
		if (ofi_after_eq(hdr->seq_no, ack->base_hdr.seq_no))
			break;

		if (pkt_entry->flags & UET_PKT_IN_USE) {
			pkt_entry->flags |= UET_PKT_ACKED;
			pkt_entry = container_of((&pkt_entry->d_entry)->next,
						 struct uet_pkt_entry, d_entry);
			continue;
		}
		uet_remove_free_pkt_entry(pkt_entry);
		uet_peer(ep, peer)->unacked_cnt--;
		uet_peer(ep, peer)->retry_cnt = 0;

		pkt_entry = container_of((&(uet_peer(ep, peer)->unacked))->next,
					struct uet_pkt_entry, d_entry);
	}

	uet_progress_tx_list(ep, uet_peer(ep, ack->base_hdr.peer));
}

void uet_handle_send_comp(struct uet_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct uet_pkt_entry *pkt_entry =
		container_of(comp->op_context, struct uet_pkt_entry, context);
	fi_addr_t peer;

	FI_DBG(&uet_prov, FI_LOG_EP_DATA,
	       "got send completion (type: %s)\n",
	       uet_pkt_type_str[(uet_pkt_type(pkt_entry))]);

	switch (uet_pkt_type(pkt_entry)) {
	case UET_CTS:
	case UET_ACK:
		uet_remove_free_pkt_entry(pkt_entry);
		break;
	default:
		if (pkt_entry->flags & UET_PKT_ACKED) {
			peer = pkt_entry->peer;
			uet_remove_free_pkt_entry(pkt_entry);
			uet_peer(ep, peer)->unacked_cnt--;
			uet_progress_tx_list(ep, uet_peer(ep, peer));
		} else {
			pkt_entry->flags &= ~UET_PKT_IN_USE;
		}
	}
}

void uet_handle_recv_comp(struct uet_ep *ep, struct fi_cq_msg_entry *comp)
{
	struct uet_pkt_entry *pkt_entry =
		container_of(comp->op_context, struct uet_pkt_entry, context);

	FI_DBG(&uet_prov, FI_LOG_EP_DATA,
	       "got recv completion (type: %s)\n",
	       uet_pkt_type_str[(uet_pkt_type(pkt_entry))]);

	uet_ep_post_buf(ep);
	uet_remove_rx_pkt(ep, pkt_entry);

	pkt_entry->pkt_size = comp->len;
	switch (uet_pkt_type(pkt_entry)) {
	case UET_RTS:
		uet_handle_rts(ep, pkt_entry);
		break;
	case UET_CTS:
		uet_handle_cts(ep, pkt_entry);
		break;
	case UET_ACK:
		uet_handle_ack(ep, pkt_entry);
		break;
	case UET_DATA:
	case UET_DATA_READ:
		uet_handle_data(ep, pkt_entry);
		/* don't need to perform action below:
		 * - release/repost RX packet */
		return;
	default:
		uet_handle_op(ep, pkt_entry);
		/* don't need to perform action below:
		 * - release/repost RX packet */
		return;
	}

	ofi_buf_free(pkt_entry);
}

void uet_handle_error(struct uet_ep *ep)
{
	struct fi_cq_err_entry err = {0};
	ssize_t ret;

	ret = fi_cq_readerr(ep->dg_cq, &err, 0);
	if (ret < 0) {
		FI_WARN(&uet_prov, FI_LOG_CQ,
			"Error reading CQ: %s\n", fi_strerror((int) -ret));
	} else {
		FI_WARN(&uet_prov, FI_LOG_CQ,
			"Received %s error from core provider: %s\n",
			err.flags & FI_SEND ? "tx" : "rx", fi_strerror(-err.err));
	}
}

static int uet_cq_close(struct fid *fid)
{
	int ret;
	struct uet_cq *cq;

	cq = container_of(fid, struct uet_cq, util_cq.cq_fid.fid);
	ret = ofi_cq_cleanup(&cq->util_cq);
	if (ret)
		return ret;
	free(cq);
	return 0;
}

static struct fi_ops uet_cq_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = uet_cq_close,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
};

ssize_t uet_cq_sreadfrom(struct fid_cq *cq_fid, void *buf, size_t count,
			 fi_addr_t *src_addr, const void *cond, int timeout)
{
	struct fid_list_entry *fid_entry;
	struct util_cq *cq;
	struct uet_ep *ep;
	uint64_t endtime;
	ssize_t ret;
	int ep_retry;

	cq = container_of(cq_fid, struct util_cq, cq_fid);
	assert(cq->wait && cq->internal_wait);
	endtime = ofi_timeout_time(timeout);

	do {
		ret = ofi_cq_readfrom(cq_fid, buf, count, src_addr);
		if (ret != -FI_EAGAIN)
			break;

		if (ofi_adjust_timeout(endtime, &timeout))
			return -FI_ETIMEDOUT;

		if (ofi_atomic_get32(&cq->wakeup)) {
			ofi_atomic_set32(&cq->wakeup, 0);
			return -FI_EAGAIN;
		}

		ep_retry = -1;
		ofi_genlock_lock(&cq->ep_list_lock);
		dlist_foreach_container(&cq->ep_list, struct fid_list_entry,
					fid_entry, entry) {
			ep = container_of(fid_entry->fid, struct uet_ep,
					  util_ep.ep_fid.fid);
			if (ep->next_retry == -1)
				continue;
			ep_retry = ep_retry == -1 ? ep->next_retry :
					MIN(ep_retry, ep->next_retry);
		}
		ofi_genlock_unlock(&cq->ep_list_lock);

		ret = ofi_wait(&cq->wait->wait_fid, ep_retry == -1 ?
			       timeout : uet_get_timeout(ep_retry));

		if (ep_retry != -1 && ret == -FI_ETIMEDOUT)
			ret = 0;
	} while (!ret);

	return ret == -FI_ETIMEDOUT ? -FI_EAGAIN : ret;
}

ssize_t uet_cq_sread(struct fid_cq *cq_fid, void *buf, size_t count,
		const void *cond, int timeout)
{
	return uet_cq_sreadfrom(cq_fid, buf, count, NULL, cond, timeout);
}

static struct fi_ops_cq uet_cq_ops = {
	.size = sizeof(struct fi_ops_cq),
	.read = ofi_cq_read,
	.readfrom = ofi_cq_readfrom,
	.readerr = ofi_cq_readerr,
	.sread = uet_cq_sread,
	.sreadfrom = uet_cq_sreadfrom,
	.signal = ofi_cq_signal,
	.strerror = uet_cq_strerror,
};

int uet_cq_open(struct fid_domain *domain, struct fi_cq_attr *attr,
		 struct fid_cq **cq_fid, void *context)
{
	int ret;
	struct uet_cq *cq;

	cq = calloc(1, sizeof(*cq));
	if (!cq)
		return -FI_ENOMEM;

	ret = ofi_cq_init(&uet_prov, domain, attr, &cq->util_cq,
			  &ofi_cq_progress, context);
	if (ret)
		goto free;

	cq->write_fn = cq->util_cq.wait ? uet_cq_write_signal : uet_cq_write;
	*cq_fid = &cq->util_cq.cq_fid;
	(*cq_fid)->fid.ops = &uet_cq_fi_ops;
	(*cq_fid)->ops = &uet_cq_ops;
	return 0;

free:
	free(cq);
	return ret;
}
