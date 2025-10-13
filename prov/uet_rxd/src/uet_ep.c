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

struct uet_pkt_entry *uet_get_tx_pkt(struct uet_ep *ep)
{
	struct uet_pkt_entry *pkt_entry;

	pkt_entry = ofi_buf_alloc(ep->tx_pkt_pool.pool);

	if (!pkt_entry)
		return NULL;

	pkt_entry->flags = 0;

	return pkt_entry;
}

struct uet_x_entry *uet_get_tx_entry(struct uet_ep *ep, uint32_t op)
{
	struct uet_x_entry *tx_entry;
	size_t *avail = op <= UET_TAGGED ? &ep->tx_msg_avail :
			&ep->tx_rma_avail;

	if (!(*avail)) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "not enough space to process RX\n");
		return NULL;
	}

	tx_entry = ofi_ibuf_alloc(ep->tx_entry_pool.pool);
	if (!tx_entry)
		return NULL;

	(*avail)--;

	return tx_entry;
}

struct uet_x_entry *uet_get_rx_entry(struct uet_ep *ep, uint32_t op)
{
	struct uet_x_entry *rx_entry;
	size_t *avail = op <= UET_TAGGED ? &ep->rx_msg_avail :
			&ep->rx_rma_avail;

	if (!(*avail)) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "not enough space to post TX\n");
		return NULL;
	}

	rx_entry = ofi_ibuf_alloc(ep->rx_entry_pool.pool);
	if (!rx_entry)
		return NULL;

	(*avail)--;

	return rx_entry;
}

static int uet_match_ctx(struct dlist_entry *item, const void *arg)
{
	struct uet_x_entry *x_entry;

	x_entry = container_of(item, struct uet_x_entry, entry);

	return (x_entry->cq_entry.op_context == arg);
}

static ssize_t uet_ep_cancel_recv(struct uet_ep *ep, struct dlist_entry *list,
				  void *context)
{
	struct dlist_entry *entry;
	struct uet_x_entry *rx_entry;
	struct fi_cq_err_entry err_entry;
	int ret = 0;

	ofi_genlock_lock(&ep->util_ep.lock);

	entry = dlist_remove_first_match(list, &uet_match_ctx, context);
	if (!entry)
		goto out;

	rx_entry = container_of(entry, struct uet_x_entry, entry);
	memset(&err_entry, 0, sizeof(struct fi_cq_err_entry));
	err_entry.op_context = rx_entry->cq_entry.op_context;
	err_entry.flags = rx_entry->cq_entry.flags;
	err_entry.err = FI_ECANCELED;
	err_entry.prov_errno = 0;
	ret = ofi_cq_write_error(&uet_ep_rx_cq(ep)->util_cq, &err_entry);
	if (ret) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not write error entry\n");
		goto out;
	}
	uet_rx_entry_free(ep, rx_entry);
	ret = 1;
out:
	ofi_genlock_unlock(&ep->util_ep.lock);
	return ret;
}

static ssize_t uet_ep_cancel(fid_t fid, void *context)
{
	struct uet_ep *ep;
	ssize_t ret;

	ep = container_of(fid, struct uet_ep, util_ep.ep_fid);

	ret = uet_ep_cancel_recv(ep, &ep->rx_tag_list, context);
	if (ret)
		goto out;

	ret = uet_ep_cancel_recv(ep, &ep->rx_list, context);

out:
	return 0;
}

static int uet_ep_getopt(fid_t fid, int level, int optname,
		   void *optval, size_t *optlen)
{
	struct uet_ep *uet_ep =
		container_of(fid, struct uet_ep, util_ep.ep_fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	*(size_t *)optval = uet_ep->min_multi_recv_size;
	*optlen = sizeof(size_t);

	return FI_SUCCESS;
}

static int uet_ep_setopt(fid_t fid, int level, int optname,
		   const void *optval, size_t optlen)
{
	struct uet_ep *uet_ep =
		container_of(fid, struct uet_ep, util_ep.ep_fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	uet_ep->min_multi_recv_size = *(size_t *)optval;

	return FI_SUCCESS;
}

struct fi_ops_ep uet_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = uet_ep_cancel,
	.getopt = uet_ep_getopt,
	.setopt = uet_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

struct uet_x_entry *uet_rx_entry_init(struct uet_ep *ep,
			const struct iovec *iov, size_t iov_count, uint64_t tag,
			uint64_t ignore, void *context, fi_addr_t addr,
			uint32_t op, uint32_t flags)
{
	struct uet_x_entry *rx_entry;

	rx_entry = uet_get_rx_entry(ep, op);
	if (!rx_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get rx entry\n");
		return NULL;
	}

	rx_entry->peer = addr;
	rx_entry->flags = flags;
	rx_entry->bytes_done = 0;
	rx_entry->offset = 0;
	rx_entry->next_seg_no = 0;
	rx_entry->iov_count = (uint8_t) iov_count;
	rx_entry->op = op;
	rx_entry->ignore = ignore;

	memcpy(rx_entry->iov, iov, sizeof(*rx_entry->iov) * iov_count);

	rx_entry->cq_entry.op_context = context;
	rx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	rx_entry->cq_entry.buf = iov_count ? iov[0].iov_base : NULL;
	rx_entry->cq_entry.tag = tag;

	rx_entry->cq_entry.flags = ofi_rx_cq_flags(op);
	dlist_init(&rx_entry->entry);

	return rx_entry;
}

ssize_t uet_ep_post_buf(struct uet_ep *ep)
{
	struct uet_pkt_entry *pkt_entry;
	ssize_t ret;

	pkt_entry = ofi_buf_alloc(ep->rx_pkt_pool.pool);
	if (!pkt_entry)
		return -FI_ENOMEM;

	ret = fi_recv(ep->dg_ep, uet_pkt_start(pkt_entry),
		      uet_ep_domain(ep)->max_mtu_sz,
		      pkt_entry->desc, FI_ADDR_UNSPEC,
		      &pkt_entry->context);
	if (ret) {
		ofi_buf_free(pkt_entry);
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "failed to repost\n");
		return ret;
	}

	slist_insert_tail(&pkt_entry->s_entry, &ep->rx_pkt_list);

	return 0;
}

static int uet_ep_enable(struct uet_ep *ep)
{
	size_t i;
	int ret;

	ret = fi_ep_bind(ep->dg_ep, &ep->dg_cq->fid, FI_TRANSMIT | FI_RECV);
	if (ret)
		return ret;

	ret = fi_enable(ep->dg_ep);
	if (ret)
		return ret;

	ep->tx_flags = uet_tx_flags(ep->util_ep.tx_op_flags);
	ep->rx_flags = uet_rx_flags(ep->util_ep.rx_op_flags);

	ofi_genlock_lock(&ep->util_ep.lock);
	for (i = 0; i < ep->rx_size; i++) {
		if (uet_ep_post_buf(ep))
			break;
	}
	ofi_genlock_unlock(&ep->util_ep.lock);
	return 0;
}

/*
 * Exponential back-off starting at 1ms, max 4s.
 */
int uet_get_timeout(int retry_cnt)
{
	return MIN(1 << retry_cnt, 4000);
}

uint64_t uet_get_retry_time(uint64_t start, int retry_cnt)
{
	return start + uet_get_timeout(retry_cnt);
}

void uet_init_data_pkt(struct uet_ep *ep, struct uet_x_entry *tx_entry,
		       struct uet_pkt_entry *pkt_entry)
{
	struct uet_data_pkt *data_pkt = (struct uet_data_pkt *) (pkt_entry->pkt);
	uint32_t seg_size;

	seg_size = (uint32_t) (tx_entry->cq_entry.len - tx_entry->bytes_done);
	seg_size = (uint32_t) MIN(uet_ep_domain(ep)->max_seg_sz, seg_size);

	data_pkt->base_hdr.version = UET_PROTOCOL_VERSION;
	data_pkt->base_hdr.type = (tx_entry->cq_entry.flags &
				  (FI_READ | FI_REMOTE_READ)) ?
				   UET_DATA_READ : UET_DATA;
	data_pkt->base_hdr.flags = tx_entry->flags & UET_TAG_HDR;

	data_pkt->ext_hdr.rx_id = tx_entry->rx_id;
	data_pkt->ext_hdr.tx_id = tx_entry->tx_id;
	data_pkt->ext_hdr.seg_no = tx_entry->next_seg_no++;
	data_pkt->base_hdr.peer = (uint32_t) uet_peer(ep, tx_entry->peer)->peer_addr;

	pkt_entry->pkt_size = ofi_copy_from_iov(data_pkt->msg, seg_size,
						tx_entry->iov,
						tx_entry->iov_count,
						tx_entry->bytes_done);
	pkt_entry->peer = tx_entry->peer;

	tx_entry->bytes_done += pkt_entry->pkt_size;

	pkt_entry->pkt_size += sizeof(*data_pkt) + ep->tx_prefix_size;
}

struct uet_x_entry *uet_tx_entry_init_common(struct uet_ep *ep, fi_addr_t addr,
			uint32_t op, const struct iovec *iov, size_t iov_count,
			uint64_t tag, uint64_t data, uint32_t flags, void *context,
			struct uet_base_hdr **base_hdr, void **ptr)
{
	struct uet_x_entry *tx_entry;

	tx_entry = uet_get_tx_entry(ep, op);
	if (!tx_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not get tx entry\n");
		return NULL;
	}

	tx_entry->pkt = uet_get_tx_pkt(ep);
	if (!tx_entry->pkt) {
		uet_tx_entry_free(ep, tx_entry);
		return NULL;
	}

	tx_entry->op = op;
	tx_entry->peer = addr;
	tx_entry->flags = flags;
	tx_entry->bytes_done = 0;
	tx_entry->offset = 0;
	tx_entry->next_seg_no = 0;
	tx_entry->iov_count = (uint8_t) iov_count;
	memcpy(&tx_entry->iov[0], iov, sizeof(*iov) * iov_count);

	tx_entry->cq_entry.op_context = context;
	tx_entry->cq_entry.len = ofi_total_iov_len(iov, iov_count);
	tx_entry->cq_entry.buf = iov[0].iov_base;
	tx_entry->cq_entry.flags = ofi_tx_cq_flags(op);
	tx_entry->cq_entry.tag = tag;
	tx_entry->cq_entry.data = data;

	tx_entry->pkt->peer = tx_entry->peer;

	*base_hdr = uet_get_base_hdr(tx_entry->pkt);
	*ptr = (void *) *base_hdr;
	uet_init_base_hdr(ep, &(*ptr), tx_entry);

	dlist_insert_tail(&tx_entry->entry,
			  &(uet_peer(ep, tx_entry->peer)->tx_list));

	return tx_entry;
}

void uet_tx_entry_free(struct uet_ep *ep, struct uet_x_entry *tx_entry)
{
	tx_entry->op <= UET_TAGGED ? ep->tx_msg_avail++ : ep->tx_rma_avail++;
	tx_entry->op = UET_NO_OP;
	dlist_remove(&tx_entry->entry);
	ofi_ibuf_free(tx_entry);
}

void uet_insert_unacked(struct uet_ep *ep, fi_addr_t peer,
			struct uet_pkt_entry *pkt_entry)
{
	dlist_insert_tail(&pkt_entry->d_entry,
			  &(uet_peer(ep, peer)->unacked));
	uet_peer(ep, peer)->unacked_cnt++;
}

ssize_t uet_ep_post_data_pkts(struct uet_ep *ep, struct uet_x_entry *tx_entry)
{
	struct uet_pkt_entry *pkt_entry;
	struct uet_data_pkt *data;

	while (tx_entry->bytes_done != tx_entry->cq_entry.len) {
		if (uet_peer(ep, tx_entry->peer)->unacked_cnt >=
		    uet_peer(ep, tx_entry->peer)->tx_window)
			return 0;

		pkt_entry = uet_get_tx_pkt(ep);
		if (!pkt_entry)
			return -FI_ENOMEM;

		uet_init_data_pkt(ep, tx_entry, pkt_entry);

		data = (struct uet_data_pkt *) (pkt_entry->pkt);
		data->base_hdr.seq_no = tx_entry->start_seq +
				        data->ext_hdr.seg_no;
		if (data->base_hdr.type != UET_DATA_READ)
			data->base_hdr.seq_no++;

		uet_ep_send_pkt(ep, pkt_entry);
		uet_insert_unacked(ep, tx_entry->peer, pkt_entry);
	}

	return uet_peer(ep, tx_entry->peer)->unacked_cnt >=
	       uet_peer(ep, tx_entry->peer)->tx_window;
}

ssize_t uet_ep_send_pkt(struct uet_ep *ep, struct uet_pkt_entry *pkt_entry)
{
	ssize_t ret;
	fi_addr_t dg_addr;
	pkt_entry->timestamp = ofi_gettime_ms();

	dg_addr = (intptr_t) ofi_idx_lookup(&(uet_ep_av(ep)->rxdaddr_dg_idx),
					    (int)pkt_entry->peer);
	ret = fi_send(ep->dg_ep, (const void *) uet_pkt_start(pkt_entry),
		      pkt_entry->pkt_size, pkt_entry->desc, dg_addr,
		      &pkt_entry->context);
	if (ret) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "error sending packet: %d (%s)\n",
			(int) ret, fi_strerror((int) -ret));
		return ret;
	}
	pkt_entry->flags |= UET_PKT_IN_USE;

	return 0;
}

static ssize_t uet_ep_send_rts(struct uet_ep *uet_ep, fi_addr_t uet_addr)
{
	struct uet_pkt_entry *pkt_entry;
	struct uet_rts_pkt *rts_pkt;
	ssize_t ret;
	size_t addrlen;

	pkt_entry = uet_get_tx_pkt(uet_ep);
	if (!pkt_entry)
		return -FI_ENOMEM;

	rts_pkt = (struct uet_rts_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*rts_pkt) + uet_ep->tx_prefix_size;
	pkt_entry->peer = uet_addr;

	rts_pkt->base_hdr.version = UET_PROTOCOL_VERSION;
	rts_pkt->base_hdr.type = UET_RTS;
	rts_pkt->rts_addr = uet_addr;

	addrlen = UET_NAME_LENGTH;
	memset(rts_pkt->source, 0, UET_NAME_LENGTH);
	ret = fi_getname(&uet_ep->dg_ep->fid, (void *) rts_pkt->source,
			 &addrlen);
	if (ret) {
		ofi_buf_free(pkt_entry);
		return ret;
	}

	uet_ep_send_pkt(uet_ep, pkt_entry);
	uet_insert_unacked(uet_ep, uet_addr, pkt_entry);
	dlist_insert_tail(&(uet_peer(uet_ep, uet_addr)->entry),
			  &uet_ep->rts_sent_list);

	return 0;
}

ssize_t uet_send_rts_if_needed(struct uet_ep *ep, fi_addr_t addr)
{
	if (!uet_peer(ep, addr)) {
		if (uet_create_peer(ep, addr) < 0)
			return -FI_ENOMEM;
	}

	if (uet_peer(ep, addr)->peer_addr == UET_ADDR_INVALID &&
	    dlist_empty(&(uet_peer(ep, addr)->unacked)))
		return uet_ep_send_rts(ep, addr);
	return 0;
}

void uet_init_base_hdr(struct uet_ep *uet_ep, void **ptr,
		       struct uet_x_entry *tx_entry)
{
	struct uet_base_hdr *hdr = (struct uet_base_hdr *) *ptr;

	hdr->version = UET_PROTOCOL_VERSION;
	hdr->type = (uint8_t) tx_entry->op;
	hdr->seq_no = 0;
	hdr->peer = (uint32_t) uet_peer(uet_ep, tx_entry->peer)->peer_addr;
	hdr->flags = (uint16_t) tx_entry->flags;

	*ptr = (char *) (*ptr) + sizeof(*hdr);
}

void uet_init_sar_hdr(void **ptr, struct uet_x_entry *tx_entry,
		      size_t iov_count)
{
	struct uet_sar_hdr *hdr = (struct uet_sar_hdr *) *ptr;

	hdr->size = tx_entry->cq_entry.len;
	hdr->num_segs = tx_entry->num_segs;
	hdr->tx_id = tx_entry->tx_id;
	hdr->iov_count = (uint8_t) iov_count;

	*ptr = (char *) (*ptr) + sizeof(*hdr);
}

void uet_init_tag_hdr(void **ptr, struct uet_x_entry *tx_entry)
{
	struct uet_tag_hdr *hdr = (struct uet_tag_hdr *) *ptr;

	hdr->tag = tx_entry->cq_entry.tag;

	*ptr = (char *) (*ptr) + sizeof(*hdr);
}

void uet_init_data_hdr(void **ptr, struct uet_x_entry *tx_entry)
{
	struct uet_data_hdr *hdr = (struct uet_data_hdr *) *ptr;

	hdr->cq_data = tx_entry->cq_entry.data;

	*ptr = (char *) (*ptr) + sizeof(*hdr);
}

void uet_init_rma_hdr(void **ptr, const struct fi_rma_iov *rma_iov,
		      size_t rma_count)
{
	struct uet_rma_hdr *hdr = (struct uet_rma_hdr *) *ptr;

	memcpy(hdr->rma, rma_iov, sizeof(*rma_iov) * rma_count);

	*ptr = (char *) (*ptr) + (sizeof(*rma_iov) * rma_count);
}

void uet_init_atom_hdr(void **ptr, enum fi_datatype datatype,
		       enum fi_op atomic_op)
{
	struct uet_atom_hdr *hdr = (struct uet_atom_hdr *) *ptr;

	hdr->datatype = datatype;
	hdr->atomic_op = atomic_op;

	*ptr = (char *) (*ptr) + sizeof(*hdr);
}

size_t uet_init_msg(void **ptr, const struct iovec *iov, size_t iov_count,
		    size_t total_len, size_t avail_len)
{
	size_t done;

	done = ofi_copy_from_iov(*ptr, MIN(total_len, avail_len), iov, iov_count, 0);

	*ptr = (char *) (*ptr) + done;

	return done;
}

void uet_ep_send_ack(struct uet_ep *uet_ep, fi_addr_t peer)
{
	struct uet_pkt_entry *pkt_entry;
	struct uet_ack_pkt *ack;

	pkt_entry = uet_get_tx_pkt(uet_ep);
	if (!pkt_entry) {
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "Unable to send ack\n");
		return;
	}

	ack = (struct uet_ack_pkt *) (pkt_entry->pkt);
	pkt_entry->pkt_size = sizeof(*ack) + uet_ep->tx_prefix_size;
	pkt_entry->peer = peer;

	ack->base_hdr.version = UET_PROTOCOL_VERSION;
	ack->base_hdr.type = UET_ACK;
	ack->base_hdr.peer = (uint32_t) uet_peer(uet_ep, peer)->peer_addr;
	ack->base_hdr.seq_no = uet_peer(uet_ep, peer)->rx_seq_no;
	ack->ext_hdr.rx_id = uet_peer(uet_ep, peer)->rx_window;
	uet_peer(uet_ep, peer)->last_tx_ack = ack->base_hdr.seq_no;

	dlist_insert_tail(&pkt_entry->d_entry, &uet_ep->ctrl_pkts);
	if (uet_ep_send_pkt(uet_ep, pkt_entry))
		uet_remove_free_pkt_entry(pkt_entry);
}

static void uet_ep_free_res(struct uet_ep *ep)
{
	if (ep->tx_pkt_pool.pool)
		ofi_bufpool_destroy(ep->tx_pkt_pool.pool);

	if (ep->rx_pkt_pool.pool)
		ofi_bufpool_destroy(ep->rx_pkt_pool.pool);

	if (ep->tx_entry_pool.pool)
		ofi_bufpool_destroy(ep->tx_entry_pool.pool);

	if (ep->rx_entry_pool.pool)
		ofi_bufpool_destroy(ep->rx_entry_pool.pool);
}

static void uet_close_peer(struct uet_ep *ep, struct uet_peer *peer)
{
	struct uet_pkt_entry *pkt_entry;
	struct uet_x_entry *x_entry;

	while (!dlist_empty(&peer->unacked)) {
		dlist_pop_front(&peer->unacked, struct uet_pkt_entry,
				pkt_entry, d_entry);
		ofi_buf_free(pkt_entry);
		peer->unacked_cnt--;
	}

	while (!dlist_empty(&peer->tx_list)) {
		dlist_pop_front(&peer->tx_list, struct uet_x_entry,
				x_entry, entry);
		uet_tx_entry_free(ep, x_entry);
	}

	while (!dlist_empty(&peer->rx_list)) {
		dlist_pop_front(&peer->rx_list, struct uet_x_entry,
				x_entry, entry);
		uet_rx_entry_free(ep, x_entry);
	}

	while (!dlist_empty(&peer->rma_rx_list)) {
		dlist_pop_front(&peer->rma_rx_list, struct uet_x_entry,
				x_entry, entry);
		uet_tx_entry_free(ep, x_entry);
	}

	dlist_remove(&peer->entry);
	peer->active = 0;
}

void uet_cleanup_unexp_msg(struct uet_unexp_msg *unexp_msg)
{
	struct uet_pkt_entry *pkt_entry;
	while (!dlist_empty(&unexp_msg->pkt_list)) {
		dlist_pop_front(&unexp_msg->pkt_list, struct uet_pkt_entry,
				pkt_entry, d_entry);
		ofi_buf_free(pkt_entry);
	}

	uet_free_unexp_msg(unexp_msg);
}

static void uet_cleanup_unexp_msg_list(struct dlist_entry *list)
{
	struct uet_unexp_msg *unexp_msg;

	while (!dlist_empty(list)) {
		dlist_pop_front(list, struct uet_unexp_msg,
				unexp_msg, entry);
		uet_cleanup_unexp_msg(unexp_msg);
	}
}

static int uet_ep_close(struct fid *fid)
{
	int ret;
	struct uet_ep *ep;
	struct uet_pkt_entry *pkt_entry;
	struct slist_entry *entry;
	struct uet_peer *peer;

	ep = container_of(fid, struct uet_ep, util_ep.ep_fid.fid);

	dlist_foreach_container(&ep->active_peers, struct uet_peer, peer, entry)
		uet_close_peer(ep, peer);
	dlist_foreach_container(&ep->rts_sent_list, struct uet_peer, peer, entry)
		uet_close_peer(ep, peer);
	ofi_idm_reset(&(ep->peers_idm), free);

	ret = fi_close(&ep->dg_ep->fid);
	if (ret)
		return ret;

	if (ep->dg_cq) {
		ret = fi_close(&ep->dg_cq->fid);
		if (ret)
			return ret;
	}

	while (!slist_empty(&ep->rx_pkt_list)) {
		entry = slist_remove_head(&ep->rx_pkt_list);
		pkt_entry = container_of(entry, struct uet_pkt_entry, s_entry);
		ofi_buf_free(pkt_entry);
	}

	uet_cleanup_unexp_msg_list(&ep->unexp_list);
	uet_cleanup_unexp_msg_list(&ep->unexp_tag_list);

	while (!dlist_empty(&ep->ctrl_pkts)) {
		dlist_pop_front(&ep->ctrl_pkts, struct uet_pkt_entry,
				pkt_entry, d_entry);
		ofi_buf_free(pkt_entry);
	}

	uet_ep_free_res(ep);
	ofi_endpoint_close(&ep->util_ep);
	free(ep);
	return 0;
}

static int uet_ep_trywait(void *arg)
{
	struct uet_fabric *uet_fabric;
	struct uet_ep *uet_ep = (struct uet_ep *) arg;
	struct fid *fids[1] = {&uet_ep->dg_cq->fid};

	uet_fabric = container_of(uet_ep->util_ep.domain->fabric,
				  struct uet_fabric, util_fabric);

	return fi_trywait(uet_fabric->dg_fabric, fids, 1);
}

static int uet_dg_cq_open(struct uet_ep *uet_ep, enum fi_wait_obj wait_obj)
{
	struct uet_domain *uet_domain;
	struct fi_cq_attr cq_attr = {0};
	int ret;

	assert((wait_obj == FI_WAIT_NONE) || (wait_obj == FI_WAIT_FD));

	cq_attr.size = uet_ep->tx_size + uet_ep->rx_size;
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = wait_obj;

	uet_domain = container_of(uet_ep->util_ep.domain, struct uet_domain,
				  util_domain);

	ret = fi_cq_open(uet_domain->dg_domain, &cq_attr, &uet_ep->dg_cq, uet_ep);
	if (ret)
		return ret;

	if (wait_obj == FI_WAIT_FD && (!uet_ep->dg_cq_fd)) {
		ret = fi_control(&uet_ep->dg_cq->fid, FI_GETWAIT,
				 &uet_ep->dg_cq_fd);
		if (ret) {
			FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
				"Unable to get dg CQ fd\n");
			goto err;
		}
	}

	return 0;
err:
	fi_close(&uet_ep->dg_cq->fid);
	uet_ep->dg_cq = NULL;
	return ret;
}

static int uet_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct uet_ep *ep;
	struct uet_av *av;
	struct util_cq *cq;
	struct util_cntr *cntr;
	int ret = 0;

	ep = container_of(ep_fid, struct uet_ep, util_ep.ep_fid.fid);
	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct uet_av, util_av.av_fid.fid);
		ret = ofi_ep_bind_av(&ep->util_ep, &av->util_av);
		if (ret)
			return ret;

		ret = fi_ep_bind(ep->dg_ep, &av->dg_av->fid, flags);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&ep->util_ep, cq, flags);
		if (ret)
			return ret;

		if (!ep->dg_cq) {
			ret = uet_dg_cq_open(ep, cq->wait ? FI_WAIT_FD : FI_WAIT_NONE);
			if (ret)
				return ret;
		}

		if (cq->wait)
			ret = ofi_wait_add_fd(cq->wait, ep->dg_cq_fd, POLLIN,
					      uet_ep_trywait, ep,
					      &ep->util_ep.ep_fid.fid);
		break;
	case FI_CLASS_EQ:
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&ep->util_ep, cntr, flags);
		if (ret)
			return ret;

		if (!ep->dg_cq) {
			ret = uet_dg_cq_open(ep, cntr->wait ? FI_WAIT_FD : FI_WAIT_NONE);
		} else if (!ep->dg_cq_fd && cntr->wait) {
			/* Reopen CQ with WAIT fd set */
			ret = fi_close(&ep->dg_cq->fid);
			if (ret) {
				FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
					"Unable to close dg CQ: %s\n",
					fi_strerror(-ret));
				return ret;
			}

			ep->dg_cq = NULL;
			ret = uet_dg_cq_open(ep, FI_WAIT_FD);
		}
		if (ret)
			return ret;

		if (cntr->wait)
			ret = ofi_wait_add_fd(cntr->wait, ep->dg_cq_fd,
					      POLLIN, uet_ep_trywait, ep,
					      &ep->util_ep.ep_fid.fid);
		break;
	default:
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
			"invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
}

static int uet_ep_control(struct fid *fid, int command, void *arg)
{
	int ret;
	struct uet_ep *ep;

	switch (command) {
	case FI_ENABLE:
		ep = container_of(fid, struct uet_ep, util_ep.ep_fid.fid);
		ret = uet_ep_enable(ep);
		break;
	default:
		ret = -FI_ENOSYS;
		break;
	}
	return ret;
}

static struct fi_ops uet_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = uet_ep_close,
	.bind = uet_ep_bind,
	.control = uet_ep_control,
	.ops_open = fi_no_ops_open,
};

static int uet_ep_cm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct uet_ep *ep;

	ep = container_of(fid, struct uet_ep, util_ep.ep_fid.fid);
	return fi_setname(&ep->dg_ep->fid, addr, addrlen);
}

static int uet_ep_cm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct uet_ep *ep;

	ep = container_of(fid, struct uet_ep, util_ep.ep_fid.fid);
	return fi_getname(&ep->dg_ep->fid, addr, addrlen);
}

struct fi_ops_cm uet_ep_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = uet_ep_cm_setname,
	.getname = uet_ep_cm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static void uet_peer_timeout(struct uet_ep *uet_ep, struct uet_peer *peer)
{
	struct fi_cq_err_entry err_entry;
	struct uet_x_entry *tx_entry;
	struct uet_pkt_entry *pkt_entry;
	int ret;

	while (!dlist_empty(&peer->tx_list)) {
		dlist_pop_front(&peer->tx_list, struct uet_x_entry, tx_entry, entry);
		memset(&err_entry, 0, sizeof(struct fi_cq_err_entry));
		uet_tx_entry_free(uet_ep, tx_entry);
		err_entry.op_context = tx_entry->cq_entry.op_context;
		err_entry.flags = tx_entry->cq_entry.flags;
		err_entry.err = FI_ECONNREFUSED;
		err_entry.prov_errno = 0;
		ret = ofi_cq_write_error(&uet_ep_tx_cq(uet_ep)->util_cq, &err_entry);
		if (ret)
			FI_WARN(&uet_prov, FI_LOG_EP_CTRL, "could not write error entry\n");
	}

	while (!dlist_empty(&peer->unacked)) {
		dlist_pop_front(&peer->unacked, struct uet_pkt_entry, pkt_entry,
				d_entry);
		ofi_buf_free(pkt_entry);
	     	peer->unacked_cnt--;
	}

	dlist_remove(&peer->entry);
}

static void uet_progress_pkt_list(struct uet_ep *ep, struct uet_peer *peer)
{
	struct uet_pkt_entry *pkt_entry;
	uint64_t current;
	ssize_t ret;
	int retry = 0;

	current = ofi_gettime_ms();
	if (peer->retry_cnt > UET_MAX_PKT_RETRY) {
		uet_peer_timeout(ep, peer);
		return;
	}

	dlist_foreach_container(&peer->unacked, struct uet_pkt_entry,
				pkt_entry, d_entry) {
		if (pkt_entry->flags & (UET_PKT_IN_USE | UET_PKT_ACKED) ||
		    current < uet_get_retry_time(pkt_entry->timestamp,
						 (uint8_t) peer->retry_cnt))
			break;
		retry = 1;
		ret = uet_ep_send_pkt(ep, pkt_entry);
		if (ret)
			break;
	}
	if (retry)
		peer->retry_cnt++;

	if (!dlist_empty(&peer->unacked))
		ep->next_retry = ep->next_retry == -1 ? peer->retry_cnt :
				 MIN(ep->next_retry, peer->retry_cnt);
}

void uet_ep_progress(struct util_ep *util_ep)
{
	struct uet_peer *peer;
	struct fi_cq_msg_entry cq_entry;
	struct dlist_entry *tmp;
	struct uet_ep *ep;
	ssize_t ret;
	int i;

	ep = container_of(util_ep, struct uet_ep, util_ep);

	ofi_genlock_lock(&ep->util_ep.lock);
	for(ret = 1, i = 0;
	    ret > 0 && (!uet_env.spin_count || i < uet_env.spin_count);
	    i++) {
		ret = fi_cq_read(ep->dg_cq, &cq_entry, 1);
		if (ret == -FI_EAGAIN)
			break;

		if (ret == -FI_EAVAIL) {
			uet_handle_error(ep);
			continue;
		}

		if (cq_entry.flags & FI_RECV)
			uet_handle_recv_comp(ep, &cq_entry);
		else
			uet_handle_send_comp(ep, &cq_entry);
	}

	if (!uet_env.retry)
		goto out;

	ep->next_retry = -1;
	dlist_foreach_container_safe(&ep->rts_sent_list, struct uet_peer,
				     peer, entry, tmp)
		uet_progress_pkt_list(ep, peer);

	dlist_foreach_container_safe(&ep->active_peers, struct uet_peer,
				     peer, entry, tmp) {
		uet_progress_pkt_list(ep, peer);
		if (dlist_empty(&peer->unacked))
			uet_progress_tx_list(ep, peer);
	}

out:
	ofi_genlock_unlock(&ep->util_ep.lock);
}

static int uet_buf_region_alloc_fn(struct ofi_bufpool_region *region)
{
	struct uet_buf_pool *pool = region->pool->attr.context;
	struct fid_mr *mr;
	int ret;

	if (!pool->uet_ep->do_local_mr) {
		region->context = NULL;
		return 0;
	}

	ret = fi_mr_reg(uet_ep_domain(pool->uet_ep)->dg_domain, region->mem_region,
			region->pool->region_size,
			FI_SEND | FI_RECV, 0, 0, OFI_MR_NOCACHE, &mr, NULL);

	region->context = mr;
	return ret;
}

static void uet_pkt_init_fn(struct ofi_bufpool_region *region, void *buf)
{
	struct uet_pkt_entry *pkt_entry = (struct uet_pkt_entry *) buf;
	struct uet_buf_pool *pool = (struct uet_buf_pool *) region->pool->attr.context;

 	if (pool->uet_ep->do_local_mr)
		pkt_entry->desc = fi_mr_desc((struct fid_mr *) region->context);
	else
		pkt_entry->desc = NULL;

 	pkt_entry->mr = (struct fid_mr *) region->context;
	if (pool->type == UET_BUF_POOL_RX)
		uet_set_rx_pkt(pool->uet_ep, pkt_entry);
	else
		uet_set_tx_pkt(pool->uet_ep, pkt_entry);
}

 static void uet_entry_init_fn(struct ofi_bufpool_region *region, void *buf)
{
	struct uet_x_entry *entry = (struct uet_x_entry *) buf;
	struct uet_buf_pool *pool = (struct uet_buf_pool *) region->pool->attr.context;

 	if (pool->type == UET_BUF_POOL_TX)
		entry->tx_id = (uint16_t) ofi_buf_index(entry);
	else
		entry->rx_id = (uint16_t) ofi_buf_index(entry);
}

static void uet_buf_region_free_fn(struct ofi_bufpool_region *region)
{
	struct uet_buf_pool *pool = region->pool->attr.context;

	if (pool->uet_ep->do_local_mr)
		fi_close(region->context);
}

static int uet_pool_create_attrs(struct uet_ep *ep, struct uet_buf_pool *pool,
					struct ofi_bufpool_attr attr,
					enum uet_pool_type type)
{
	int ret;
	pool->uet_ep = ep;
	pool->type = type;

	ret = ofi_bufpool_create_attr(&attr, &pool->pool);
	if (ret)
		FI_WARN(&uet_prov, FI_LOG_EP_CTRL,
			"Unable to create buf pool\n");
	return ret;
}

static int uet_pkt_pool_create(struct uet_ep *ep,
			       size_t chunk_cnt, struct uet_buf_pool *pool,
			       enum uet_pool_type type)
{
	struct ofi_bufpool_attr attr = {
		.size		= uet_ep_domain(ep)->max_mtu_sz +
				  sizeof(struct uet_pkt_entry),
		.alignment	= UET_BUF_POOL_ALIGNMENT,
		.max_cnt	= 0,
		.chunk_cnt	= chunk_cnt,
		.alloc_fn	= uet_buf_region_alloc_fn,
		.free_fn	= uet_buf_region_free_fn,
		.init_fn	= uet_pkt_init_fn,
		.context	= pool,
		.flags		= OFI_BUFPOOL_HUGEPAGES,
	};

	return uet_pool_create_attrs(ep, pool, attr, type);
}


static int uet_entry_pool_create(struct uet_ep *ep,
				 size_t chunk_cnt, struct uet_buf_pool *pool,
				 enum uet_pool_type type)
{
	struct ofi_bufpool_attr attr = {
		.size		= sizeof(struct uet_x_entry),
		.alignment	= UET_BUF_POOL_ALIGNMENT,
		.max_cnt	= (size_t) ((uint16_t) (~0)),
		.chunk_cnt	= chunk_cnt,
		.alloc_fn	= NULL,
		.free_fn	= NULL,
		.init_fn	= uet_entry_init_fn,
		.context	= pool,
		.flags		= OFI_BUFPOOL_INDEXED | OFI_BUFPOOL_NO_TRACK |
				  OFI_BUFPOOL_HUGEPAGES,
	};

	return uet_pool_create_attrs(ep, pool, attr, type);
}

int uet_ep_init_res(struct uet_ep *ep, struct fi_info *fi_info)
{
	int ret;

	ret = uet_pkt_pool_create(ep, UET_TX_POOL_CHUNK_CNT,
				  &ep->tx_pkt_pool, UET_BUF_POOL_TX);
	if (ret)
		goto err;

	ret = uet_pkt_pool_create(ep, UET_RX_POOL_CHUNK_CNT,
				  &ep->rx_pkt_pool, UET_BUF_POOL_RX);
	if (ret)
		goto err;

	ret = uet_entry_pool_create(ep, ep->tx_size,
				    &ep->tx_entry_pool, UET_BUF_POOL_TX);
	if (ret)
		goto err;

	ret = uet_entry_pool_create(ep, ep->rx_size,
				    &ep->rx_entry_pool, UET_BUF_POOL_RX);
	if (ret)
		goto err;

	dlist_init(&ep->rx_list);
	dlist_init(&ep->rx_tag_list);
	dlist_init(&ep->active_peers);
	dlist_init(&ep->rts_sent_list);
	dlist_init(&ep->unexp_list);
	dlist_init(&ep->unexp_tag_list);
	dlist_init(&ep->ctrl_pkts);
	slist_init(&ep->rx_pkt_list);

	return 0;
err:
	uet_ep_free_res(ep);

	return ret;
}

int uet_create_peer(struct uet_ep *ep, uint64_t uet_addr)
{

	struct uet_peer *peer;

	peer = calloc(1, sizeof(struct uet_peer));
	if (!peer)
		return -FI_ENOMEM;

	peer->peer_addr = UET_ADDR_INVALID;
	peer->tx_seq_no = 0;
	peer->rx_seq_no = 0;
	peer->last_rx_ack = 0;
	peer->last_tx_ack = 0;
	peer->rx_window = (uint16_t) uet_env.max_unacked;
	peer->tx_window = (uint16_t) uet_env.max_unacked;
	peer->unacked_cnt = 0;
	peer->retry_cnt = 0;
	peer->active = 0;
	dlist_init(&(peer->unacked));
	dlist_init(&(peer->tx_list));
	dlist_init(&(peer->rx_list));
	dlist_init(&(peer->rma_rx_list));
	dlist_init(&(peer->buf_pkts));

	if (ofi_idm_set(&(ep->peers_idm), (int) uet_addr, peer) < 0)
		goto err;

	return 0;
err:
	free(peer);
	return -FI_ENOMEM;
}

int uet_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep, void *context)
{
	struct fi_info *dg_info;
	struct uet_domain *uet_domain;
	struct uet_ep *uet_ep;
	int ret;

	uet_ep = calloc(1, sizeof(*uet_ep));
	if (!uet_ep)
		return -FI_ENOMEM;

	uet_domain = container_of(domain, struct uet_domain,
				  util_domain.domain_fid);

	ret = ofi_endpoint_init(domain, &uet_util_prov, info, &uet_ep->util_ep,
				context, uet_ep_progress);
	if (ret)
		goto err1;

	ret = ofi_get_core_info(uet_domain->util_domain.fabric->fabric_fid.api_version,
				NULL, NULL, 0, &uet_util_prov, info, NULL,
				uet_info_to_core, &dg_info);
	if (ret)
		goto err2;

	memcpy(dg_info->src_addr, info->src_addr, info->src_addrlen);
	uet_ep->do_local_mr = ofi_mr_local(dg_info);

	ret = fi_endpoint(uet_domain->dg_domain, dg_info, &uet_ep->dg_ep, uet_ep);
	if (ret)
		goto err2;

	uet_ep->tx_prefix_size = dg_info->tx_attr->mode & FI_MSG_PREFIX ?
				 dg_info->ep_attr->msg_prefix_size : 0;
	uet_ep->rx_prefix_size = dg_info->rx_attr->mode & FI_MSG_PREFIX ?
				 dg_info->ep_attr->msg_prefix_size : 0;
	uet_ep->rx_size = MIN(dg_info->rx_attr->size, info->rx_attr->size);
	uet_ep->tx_size = MIN(dg_info->tx_attr->size, info->tx_attr->size);
	uet_ep->tx_msg_avail = uet_ep->tx_size;
	uet_ep->rx_msg_avail = uet_ep->rx_size;
	uet_ep->tx_rma_avail = uet_ep->tx_size;
	uet_ep->rx_rma_avail = uet_ep->rx_size;
	fi_freeinfo(dg_info);

	uet_ep->next_retry = -1;
	ret = uet_ep_init_res(uet_ep, info);
	if (ret)
		goto err3;

	memset(&(uet_ep->peers_idm), 0, sizeof(uet_ep->peers_idm));

	uet_ep->util_ep.ep_fid.fid.ops = &uet_ep_fi_ops;
	uet_ep->util_ep.ep_fid.cm = &uet_ep_cm;
	uet_ep->util_ep.ep_fid.ops = &uet_ops_ep;
	uet_ep->util_ep.ep_fid.msg = &uet_ops_msg;
	uet_ep->util_ep.ep_fid.tagged = &uet_ops_tagged;
	uet_ep->util_ep.ep_fid.rma = &uet_ops_rma;
	uet_ep->util_ep.ep_fid.atomic = &uet_ops_atomic;

	*ep = &uet_ep->util_ep.ep_fid;
	return 0;

err3:
	fi_close(&uet_ep->dg_ep->fid);
err2:
	ofi_endpoint_close(&uet_ep->util_ep);
err1:
	free(uet_ep);
	return ret;
}
