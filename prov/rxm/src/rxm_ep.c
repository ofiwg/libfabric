/*
 * Copyright (c) 2013-2016 Intel Corporation. All rights reserved.
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

#include <inttypes.h>

#include "ofi.h"
#include <ofi_util.h>

#include "rxm.h"

#define RXM_EP_RECV_SANITIZE_SRC_ADDR(rxm_ep, src_addr)	\
	((rxm_ep->rxm_info->caps & FI_DIRECTED_RECV) ?	\
	 src_addr : FI_ADDR_UNSPEC)

const size_t rxm_pkt_size = sizeof(struct rxm_pkt);

static int rxm_match_recv_entry(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *) arg;
	struct rxm_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxm_recv_entry, entry);
	return rxm_match_addr(recv_entry->addr, attr->addr);
}

static int rxm_match_recv_entry_tagged(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxm_recv_entry, entry);
	return rxm_match_addr(recv_entry->addr, attr->addr) &&
		rxm_match_tag(recv_entry->tag, recv_entry->ignore, attr->tag);
}

static int rxm_match_recv_entry_context(struct dlist_entry *item, const void *context)
{
	struct rxm_recv_entry *recv_entry;

	recv_entry = container_of(item, struct rxm_recv_entry, entry);
	return recv_entry->context == context;
}

static int rxm_match_unexp_msg(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_unexp_msg *unexp_msg;

	unexp_msg = container_of(item, struct rxm_unexp_msg, entry);
	return rxm_match_addr(attr->addr, unexp_msg->addr);
}

static int rxm_match_unexp_msg_tagged(struct dlist_entry *item, const void *arg)
{
	struct rxm_recv_match_attr *attr = (struct rxm_recv_match_attr *)arg;
	struct rxm_unexp_msg *unexp_msg;

	unexp_msg = container_of(item, struct rxm_unexp_msg, entry);
	return rxm_match_addr(attr->addr, unexp_msg->addr) &&
		rxm_match_tag(attr->tag, attr->ignore, unexp_msg->tag);
}

static inline int
rxm_mr_buf_reg(struct rxm_ep *rxm_ep, void *addr, size_t len, void **context)
{
	int ret = FI_SUCCESS;
	struct fid_mr *mr;
	struct rxm_domain *rxm_domain = container_of(rxm_ep->util_ep.domain,
						     struct rxm_domain, util_domain);

	*context = NULL;
	if (rxm_ep->msg_mr_local) {
		struct fid_domain *msg_domain =
			(struct fid_domain *)rxm_domain->msg_domain;

		ret = fi_mr_reg(msg_domain, addr, len,
				FI_SEND | FI_RECV | FI_READ | FI_WRITE,
				0, 0, 0, &mr, NULL);
		*context = mr;
	}

	return ret;
}

static int rxm_buf_reg(void *pool_ctx, void *addr, size_t len, void **context)
{
	struct rxm_buf_pool *pool = (struct rxm_buf_pool *)pool_ctx;
	size_t i, entry_sz = pool->pool->entry_sz;
	int ret;
	struct rxm_tx_buf *tx_buf;
	struct rxm_rx_buf *rx_buf;
	void *mr_desc;

	ret = rxm_mr_buf_reg(pool->ep, addr, len, context);
	if (ret)
		return ret;

	mr_desc = (*context != NULL) ? fi_mr_desc((struct fid_mr *)*context) : NULL;

	for (i = 0; i < pool->pool->chunk_cnt; i++) {
		if (pool->type == RXM_BUF_POOL_RX) {
			rx_buf = (struct rxm_rx_buf *)((char *)addr + i * entry_sz);
			rx_buf->ep = pool->ep;
			rx_buf->hdr.desc = mr_desc;
		} else {
			tx_buf = (struct rxm_tx_buf *)((char *)addr + i * entry_sz);
			tx_buf->type = pool->type;
			tx_buf->pkt.ctrl_hdr.version = OFI_CTRL_VERSION;
			tx_buf->pkt.hdr.version = OFI_OP_VERSION;
			if (rxm_ep_tx_flags(pool->ep) & FI_TRANSMIT_COMPLETE)
				tx_buf->pkt.hdr.flags |= OFI_TRANSMIT_COMPLETE;
			if (rxm_ep_tx_flags(pool->ep) & FI_DELIVERY_COMPLETE)
				tx_buf->pkt.hdr.flags |= OFI_DELIVERY_COMPLETE;
			tx_buf->hdr.desc = mr_desc;

			switch (pool->type) {
			case RXM_BUF_POOL_TX_MSG:
			case RXM_BUF_POOL_RMA:
				tx_buf->pkt.ctrl_hdr.type = ofi_ctrl_data;
				tx_buf->pkt.hdr.op = ofi_op_msg;
				break;
			case RXM_BUF_POOL_TX_TAGGED:
				tx_buf->pkt.ctrl_hdr.type = ofi_ctrl_data;
				tx_buf->pkt.hdr.op = ofi_op_tagged;
				break;
			case RXM_BUF_POOL_TX_ACK:
				tx_buf->pkt.ctrl_hdr.type = ofi_ctrl_ack;
				tx_buf->pkt.hdr.op = ofi_op_msg;
				break;
			case RXM_BUF_POOL_TX_LMT:
				tx_buf->pkt.ctrl_hdr.type = ofi_ctrl_large_data;
				break;
			default:
				assert(0);
				break;
			}
		}
	}

	return FI_SUCCESS;
}

static inline void rxm_buf_close(void *pool_ctx, void *context)
{
	struct rxm_ep *rxm_ep =
		(struct rxm_ep *)((struct rxm_buf_pool *)pool_ctx)->ep;

	if (rxm_ep->msg_mr_local) {
		/* We would get a (fid_mr *) in context but
		 * it is safe to cast it into (fid *) */
		fi_close((struct fid *)context);
	}
}

static void rxm_buf_pool_destroy(struct rxm_buf_pool *pool)
{
	fastlock_destroy(&pool->lock);
	util_buf_pool_destroy(pool->pool);
}

static void rxm_ep_cleanup_post_rx_list(struct rxm_ep *rxm_ep)
{
	struct rxm_rx_buf *rx_buf;
	while (!dlist_empty(&rxm_ep->post_rx_list)) {
		dlist_pop_front(&rxm_ep->post_rx_list, struct rxm_rx_buf,
				rx_buf, entry);
		rxm_rx_buf_release(rxm_ep, rx_buf);
	}
}

static int rxm_buf_pool_create(struct rxm_ep *rxm_ep,
			       size_t chunk_count, size_t size,
			       struct rxm_buf_pool *pool,
			       enum rxm_buf_pool_type type)
{
	int ret;

	pool->ep = rxm_ep;
	pool->type = type;
	ret = util_buf_pool_create_ex(&pool->pool, size, 16, 0, chunk_count,
				      rxm_buf_reg, rxm_buf_close, pool);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to create buf pool\n");
		return -FI_ENOMEM;
	}
	fastlock_init(&pool->lock);
	return 0;
}

static int rxm_send_queue_init(struct rxm_ep *rxm_ep, struct rxm_send_queue *send_queue,
			       size_t size)
{
	ssize_t i;

	send_queue->fs = rxm_txe_fs_create(size);
	if (!send_queue->fs)
		return -FI_ENOMEM;

	for (i = send_queue->fs->size - 1; i >= 0; i--)
		send_queue->fs->buf[i].ep = rxm_ep;

	fastlock_init(&send_queue->lock);
	return 0;
}

static int rxm_recv_queue_init(struct rxm_ep *rxm_ep,  struct rxm_recv_queue *recv_queue,
			       size_t size, enum rxm_recv_queue_type type)
{
	ssize_t i;

	recv_queue->rxm_ep = rxm_ep;
	recv_queue->type = type;
	recv_queue->fs = rxm_recv_fs_create(size);
	if (!recv_queue->fs)
		return -FI_ENOMEM;

	dlist_init(&recv_queue->recv_list);
	dlist_init(&recv_queue->unexp_msg_list);
	if (type == RXM_RECV_QUEUE_MSG) {
		recv_queue->match_recv = rxm_match_recv_entry;
		recv_queue->match_unexp = rxm_match_unexp_msg;
		for (i = recv_queue->fs->size - 1; i >= 0; i--)
			recv_queue->fs->buf[i].comp_flags = FI_MSG | FI_RECV;
	} else {
		recv_queue->match_recv = rxm_match_recv_entry_tagged;
		recv_queue->match_unexp = rxm_match_unexp_msg_tagged;
		for (i = recv_queue->fs->size - 1; i >= 0; i--)
			recv_queue->fs->buf[i].comp_flags = FI_TAGGED | FI_RECV;
	}
	fastlock_init(&recv_queue->lock);
	return 0;
}

static void rxm_send_queue_close(struct rxm_send_queue *send_queue)
{
	if (send_queue->fs) {
		struct rxm_tx_entry *tx_entry;
		ssize_t i;

		for (i = send_queue->fs->size - 1; i >= 0; i--) {
			tx_entry = &send_queue->fs->buf[i];
			if (tx_entry->tx_buf) {
				rxm_tx_buf_release(tx_entry->ep, tx_entry->tx_buf);
				tx_entry->tx_buf = NULL;
			}
		}
		rxm_txe_fs_free(send_queue->fs);
	}
	fastlock_destroy(&send_queue->lock);
}

static void rxm_recv_queue_close(struct rxm_recv_queue *recv_queue)
{
	if (recv_queue->fs)
		rxm_recv_fs_free(recv_queue->fs);
	fastlock_destroy(&recv_queue->lock);
	// TODO cleanup recv_list and unexp msg list
}

static int rxm_ep_txrx_pool_create(struct rxm_ep *rxm_ep)
{
	size_t i;
	int ret;

	ret = rxm_buf_pool_create(rxm_ep,
				  rxm_ep->msg_info->rx_attr->size,
				  rxm_ep->rxm_info->tx_attr->inject_size +
				  sizeof(struct rxm_rx_buf),
				  &rxm_ep->buf_pools[RXM_BUF_POOL_RX],
				  RXM_BUF_POOL_RX);
	if (ret)
		return ret;
	dlist_init(&rxm_ep->post_rx_list);
	dlist_init(&rxm_ep->repost_ready_list);

	/* Allocates resources for TX pools */
	for (i = RXM_BUF_POOL_TX_START; i <= RXM_BUF_POOL_TX_END; i++) {
		ret = rxm_buf_pool_create(rxm_ep,
					  rxm_ep->msg_info->tx_attr->size,
					  rxm_ep->rxm_info->tx_attr->inject_size +
					  sizeof(struct rxm_tx_buf),
					  &rxm_ep->buf_pools[i], i);
		if (ret)
			goto err;
	}

	ret = rxm_buf_pool_create(rxm_ep,
				  rxm_ep->msg_info->tx_attr->size,
				  rxm_ep->rxm_info->tx_attr->inject_size +
				  sizeof(struct rxm_rma_buf),
				  &rxm_ep->buf_pools[RXM_BUF_POOL_RMA],
				  RXM_BUF_POOL_RMA);
	if (ret)
		goto err;

	return FI_SUCCESS;
err:
	while (--i >= RXM_BUF_POOL_TX_START)
		rxm_buf_pool_destroy(&rxm_ep->buf_pools[i]);
	rxm_buf_pool_destroy(&rxm_ep->buf_pools[RXM_BUF_POOL_RX]);
	return ret;
}

static void rxm_ep_txrx_pool_destroy(struct rxm_ep *rxm_ep)
{
	size_t i;

	for (i = RXM_BUF_POOL_START; i < RXM_BUF_POOL_MAX; i++)
		rxm_buf_pool_destroy(&rxm_ep->buf_pools[i]);
}

static int rxm_ep_txrx_queue_init(struct rxm_ep *rxm_ep)
{
	int ret;

	ret = rxm_send_queue_init(rxm_ep, &rxm_ep->send_queue,
				  rxm_ep->rxm_info->tx_attr->size);
	if (ret)
		return ret;

	ret = rxm_recv_queue_init(rxm_ep, &rxm_ep->recv_queue,
				  rxm_ep->rxm_info->rx_attr->size,
				  RXM_RECV_QUEUE_MSG);
	if (ret)
		goto err_recv_msg;

	ret = rxm_recv_queue_init(rxm_ep, &rxm_ep->trecv_queue,
				  rxm_ep->rxm_info->rx_attr->size,
				  RXM_RECV_QUEUE_TAGGED);
	if (ret)
		goto err_recv_tag;

	return FI_SUCCESS;
err_recv_tag:
	rxm_recv_queue_close(&rxm_ep->recv_queue);
err_recv_msg:
	rxm_send_queue_close(&rxm_ep->send_queue);
	return ret;
}

static void rxm_ep_txrx_queue_close(struct rxm_ep *rxm_ep)
{
	rxm_recv_queue_close(&rxm_ep->trecv_queue);
	rxm_recv_queue_close(&rxm_ep->recv_queue);
	rxm_send_queue_close(&rxm_ep->send_queue);
}

static int rxm_ep_txrx_res_open(struct rxm_ep *rxm_ep)
{
	int ret;

	FI_DBG(&rxm_prov, FI_LOG_EP_CTRL,
	       "MSG provider mr_mode & FI_MR_LOCAL: %d\n",
	       rxm_ep->msg_mr_local);

	ret = rxm_ep_txrx_pool_create(rxm_ep);
	if (ret)
		return ret;

	ret = rxm_ep_txrx_queue_init(rxm_ep);
	if (ret)
		goto err;

	return FI_SUCCESS;
err:
	rxm_ep_txrx_pool_destroy(rxm_ep);
	return ret;
}

static void rxm_ep_txrx_res_close(struct rxm_ep *rxm_ep)
{
	rxm_ep_txrx_queue_close(rxm_ep);

	rxm_ep_cleanup_post_rx_list(rxm_ep);
	rxm_ep_txrx_pool_destroy(rxm_ep);
}

static int rxm_setname(fid_t fid, void *addr, size_t addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_setname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

static int rxm_getname(fid_t fid, void *addr, size_t *addrlen)
{
	struct rxm_ep *rxm_ep;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	return fi_getname(&rxm_ep->msg_pep->fid, addr, addrlen);
}

static struct fi_ops_cm rxm_ops_cm = {
	.size = sizeof(struct fi_ops_cm),
	.setname = rxm_setname,
	.getname = rxm_getname,
	.getpeer = fi_no_getpeer,
	.connect = fi_no_connect,
	.listen = fi_no_listen,
	.accept = fi_no_accept,
	.reject = fi_no_reject,
	.shutdown = fi_no_shutdown,
	.join = fi_no_join,
};

static int rxm_ep_cancel_recv(struct rxm_ep *rxm_ep,
			      struct rxm_recv_queue *recv_queue, void *context)
{
	struct fi_cq_err_entry err_entry;
	struct rxm_recv_entry *recv_entry;
	struct dlist_entry *entry;

	fastlock_acquire(&recv_queue->lock);
	entry = dlist_remove_first_match(&recv_queue->recv_list,
					 rxm_match_recv_entry_context,
					 context);
	fastlock_release(&recv_queue->lock);
	if (entry) {
		recv_entry = container_of(entry, struct rxm_recv_entry, entry);
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.op_context = recv_entry->context;
		err_entry.flags |= recv_entry->comp_flags;
		err_entry.tag = recv_entry->tag;
		err_entry.err = FI_ECANCELED;
		err_entry.prov_errno = -FI_ECANCELED;
		rxm_recv_entry_release(recv_queue, recv_entry);
		return ofi_cq_write_error(rxm_ep->util_ep.rx_cq, &err_entry);
	}
	return 0;
}

static ssize_t rxm_ep_cancel(fid_t fid_ep, void *context)
{
	struct rxm_ep *rxm_ep = container_of(fid_ep, struct rxm_ep, util_ep.ep_fid);
	int ret;

	ret = rxm_ep_cancel_recv(rxm_ep, &rxm_ep->recv_queue, context);
	if (ret)
		return ret;

	ret = rxm_ep_cancel_recv(rxm_ep, &rxm_ep->trecv_queue, context);
	if (ret)
		return ret;

	return 0;
}

static int rxm_ep_getopt(fid_t fid, int level, int optname, void *optval,
			 size_t *optlen)
{
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	*(size_t *)optval = rxm_ep->min_multi_recv_size;
	*optlen = sizeof(size_t);

	return FI_SUCCESS;
}

static int rxm_ep_setopt(fid_t fid, int level, int optname,
			 const void *optval, size_t optlen)
{
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid);

	if ((level != FI_OPT_ENDPOINT) || (optname != FI_OPT_MIN_MULTI_RECV))
		return -FI_ENOPROTOOPT;

	rxm_ep->min_multi_recv_size = *(size_t *)optval;

	return FI_SUCCESS;
}

static struct fi_ops_ep rxm_ops_ep = {
	.size = sizeof(struct fi_ops_ep),
	.cancel = rxm_ep_cancel,
	.getopt = rxm_ep_getopt,
	.setopt = rxm_ep_setopt,
	.tx_ctx = fi_no_tx_ctx,
	.rx_ctx = fi_no_rx_ctx,
	.rx_size_left = fi_no_rx_size_left,
	.tx_size_left = fi_no_tx_size_left,
};

static int rxm_ep_discard_recv(struct rxm_ep *rxm_ep, struct rxm_rx_buf *rx_buf,
			       void *context)
{
	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Discarding message",
			 rx_buf->unexp_msg.addr, rx_buf->unexp_msg.tag);

	fastlock_acquire(&rxm_ep->util_ep.lock);
	dlist_insert_tail(&rx_buf->repost_entry,
			  &rx_buf->ep->repost_ready_list);
	fastlock_release(&rxm_ep->util_ep.lock);

	return ofi_cq_write(rxm_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			    0, NULL, rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
}

static int rxm_ep_peek_recv(struct rxm_ep *rxm_ep, fi_addr_t addr, uint64_t tag,
			    uint64_t ignore, void *context, uint64_t flags,
			    struct rxm_recv_queue *recv_queue)
{
	struct rxm_rx_buf *rx_buf;

	RXM_DBG_ADDR_TAG(FI_LOG_EP_DATA, "Peeking message", addr, tag);

	rxm_ep_progress_multi(&rxm_ep->util_ep);

	fastlock_acquire(&recv_queue->lock);

	rx_buf = rxm_check_unexp_msg_list(recv_queue, addr, tag, ignore);
	if (!rx_buf) {
		fastlock_release(&recv_queue->lock);
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Message not found\n");
		return ofi_cq_write_error_peek(rxm_ep->util_ep.rx_cq, tag,
					       context);
	}

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Message found\n");

	if (flags & FI_DISCARD) {
		dlist_remove(&rx_buf->unexp_msg.entry);
		fastlock_release(&recv_queue->lock);
		return rxm_ep_discard_recv(rxm_ep, rx_buf, context);
	}

	if (flags & FI_CLAIM) {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Marking message for Claim\n");
		((struct fi_context *)context)->internal[0] = rx_buf;
		dlist_remove(&rx_buf->unexp_msg.entry);
	}
	fastlock_release(&recv_queue->lock);

	return ofi_cq_write(rxm_ep->util_ep.rx_cq, context, FI_TAGGED | FI_RECV,
			    rx_buf->pkt.hdr.size, NULL,
			    rx_buf->pkt.hdr.data, rx_buf->pkt.hdr.tag);
}

static inline ssize_t
rxm_ep_format_rx_res(struct rxm_ep *rxm_ep, const struct iovec *iov,
		     void **desc, size_t count, fi_addr_t src_addr,
		     uint64_t tag, uint64_t ignore, void *context,
		     uint64_t flags, struct rxm_recv_queue *recv_queue,
		     struct rxm_recv_entry **recv_entry)
{
	size_t i;

	*recv_entry = rxm_recv_entry_get(recv_queue);
	if (OFI_UNLIKELY(!*recv_entry))
		return -FI_EAGAIN;

	(*recv_entry)->rxm_iov.count 	= (uint8_t)count;
	(*recv_entry)->addr 		= RXM_EP_RECV_SANITIZE_SRC_ADDR(rxm_ep, src_addr);
	(*recv_entry)->context 		= context;
	(*recv_entry)->flags 		= flags;
	(*recv_entry)->ignore 		= ignore;
	(*recv_entry)->tag		= tag;
	(*recv_entry)->multi_recv_buf	= iov[0].iov_base;

	for (i = 0; i < count; i++) {
		(*recv_entry)->rxm_iov.iov[i].iov_base = iov[i].iov_base;
		(*recv_entry)->total_len +=
			(*recv_entry)->rxm_iov.iov[i].iov_len = iov[i].iov_len;
		if (desc)
			(*recv_entry)->rxm_iov.desc[i] = desc[i];
	}

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_recv_common(struct rxm_ep *rxm_ep, const struct iovec *iov,
		   void **desc, size_t count, fi_addr_t src_addr,
		   uint64_t tag, uint64_t ignore, void *context,
		   uint64_t op_flags, struct rxm_recv_queue *recv_queue)
{
	struct rxm_recv_entry *recv_entry;
	ssize_t ret;

	assert(count <= rxm_ep->rxm_info->rx_attr->iov_limit);

	ret = rxm_ep_format_rx_res(rxm_ep, iov, desc, count, src_addr,
				   tag, ignore, context, op_flags,
				   recv_queue, &recv_entry);
	if (OFI_UNLIKELY(ret))
		return ret;
	return rxm_process_recv_entry(recv_queue, recv_entry);
}

static ssize_t rxm_ep_recv_common_flags(struct rxm_ep *rxm_ep, const struct iovec *iov,
					void **desc, size_t count, fi_addr_t src_addr,
					uint64_t tag, uint64_t ignore, void *context,
					uint64_t flags, uint64_t op_flags,
					struct rxm_recv_queue *recv_queue)
{
	struct rxm_recv_entry *recv_entry;
	struct rxm_rx_buf *rx_buf;

	assert(count <= rxm_ep->rxm_info->rx_attr->iov_limit);
	assert(!(flags & (FI_PEEK | FI_CLAIM | FI_DISCARD)) ||
		(recv_queue->type == RXM_RECV_QUEUE_TAGGED));
	assert(!(flags & (FI_MULTI_RECV)) ||
		(recv_queue->type == RXM_RECV_QUEUE_MSG));

	if (flags & FI_PEEK)
		return rxm_ep_peek_recv(rxm_ep,
					RXM_EP_RECV_SANITIZE_SRC_ADDR(rxm_ep, src_addr),
					tag, ignore, context,
					flags, recv_queue);

	if (flags & FI_CLAIM) {
		ssize_t ret;

		rx_buf = ((struct fi_context *)context)->internal[0];
		assert(rx_buf);
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "Claim message\n");

		if (flags & FI_DISCARD)
			return rxm_ep_discard_recv(rxm_ep, rx_buf, context);

		ret = rxm_ep_format_rx_res(rxm_ep, iov, desc, count,
					   RXM_EP_RECV_SANITIZE_SRC_ADDR(rxm_ep, src_addr),
					   tag, ignore, context,
					   flags | op_flags,
					   recv_queue, &recv_entry);
		if (OFI_UNLIKELY(ret))
			return ret;
		rx_buf->recv_entry = recv_entry;
		return rxm_cq_handle_data(rx_buf);
	} else {
		return rxm_ep_recv_common(rxm_ep, iov, desc, count, src_addr,
					  tag, ignore, context, flags | op_flags,
					  recv_queue);
	}
}

static ssize_t rxm_ep_recvmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common_flags(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
					msg->addr, 0, 0, msg->context,
					flags, (rxm_ep_rx_flags(rxm_ep) & FI_COMPLETION),
					&rxm_ep->recv_queue);
}

static ssize_t rxm_ep_recv(struct fid_ep *ep_fid, void *buf, size_t len, void *desc,
			    fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};

	return rxm_ep_recv_common(rxm_ep, &iov, &desc, 1, src_addr, 0, 0,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->recv_queue);
}

static ssize_t rxm_ep_recvv(struct fid_ep *ep_fid, const struct iovec *iov,
		void **desc, size_t count, fi_addr_t src_addr, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common(rxm_ep, iov, desc, count, src_addr, 0, 0,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->recv_queue);
}

void rxm_ep_msg_mr_closev(struct fid_mr **mr, size_t count)
{
	int ret;
	size_t i;

	for (i = 0; i < count; i++) {
		if (mr[i]) {
			ret = fi_close(&mr[i]->fid);
			if (ret)
				FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
					"Unable to close msg mr: %zu\n", i);
		}
	}
}

int rxm_ep_msg_mr_regv(struct rxm_ep *rxm_ep, const struct iovec *iov,
		       size_t count, uint64_t access, struct fid_mr **mr)
{
	struct rxm_domain *rxm_domain;
	int ret;
	size_t i;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain, util_domain);

	// TODO do fi_mr_regv if provider supports it
	for (i = 0; i < count; i++) {
		ret = fi_mr_reg(rxm_domain->msg_domain, iov[i].iov_base,
				iov[i].iov_len, access, 0, 0, 0, &mr[i], NULL);
		if (ret)
			goto err;
	}
	return 0;
err:
	rxm_ep_msg_mr_closev(mr, count);
	return ret;
}

static ssize_t rxm_rma_iov_init(struct rxm_ep *rxm_ep, void *buf,
				const struct iovec *iov, size_t count,
				struct fid_mr **mr)
{
	struct rxm_rma_iov *rma_iov = (struct rxm_rma_iov *)buf;
	size_t i;

	for (i = 0; i < count; i++) {
		rma_iov->iov[i].addr = RXM_MR_VIRT_ADDR(rxm_ep->msg_info) ?
			(uintptr_t)iov[i].iov_base : 0;
		rma_iov->iov[i].len = (uint64_t)iov[i].iov_len;
		rma_iov->iov[i].key = fi_mr_key(mr[i]);
	}
	rma_iov->count = (uint8_t)count;
	return sizeof(*rma_iov) + sizeof(*rma_iov->iov) * count;
}

static inline ssize_t
rxm_ep_format_tx_res_lightweight(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
				 size_t len, uint64_t data, uint64_t flags, uint64_t tag,
				 struct rxm_tx_buf **tx_buf, struct rxm_buf_pool *pool)
{
	*tx_buf = (struct rxm_tx_buf *)rxm_buf_get(pool);
	if (OFI_UNLIKELY(!*tx_buf)) {
		FI_WARN(&rxm_prov, FI_LOG_EP_DATA, "TX queue full!\n");
		return -FI_EAGAIN;
	}

	assert((((*tx_buf)->pkt.ctrl_hdr.type == ofi_ctrl_data) &&
		 (len <= rxm_ep->rxm_info->tx_attr->inject_size)) ||
	       ((*tx_buf)->pkt.ctrl_hdr.type == ofi_ctrl_large_data));

	(*tx_buf)->pkt.ctrl_hdr.conn_id = rxm_conn->handle.remote_key;

	(*tx_buf)->pkt.hdr.size = len;
	(*tx_buf)->pkt.hdr.tag = tag;

	if (flags & FI_REMOTE_CQ_DATA) {
		(*tx_buf)->pkt.hdr.flags = OFI_REMOTE_CQ_DATA;
		(*tx_buf)->pkt.hdr.data = data;
	}

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_format_tx_entry(struct rxm_ep *rxm_ep, void *context, uint8_t count,
		       uint64_t flags, uint64_t comp_flags,
		       struct rxm_tx_buf *tx_buf, struct rxm_tx_entry **tx_entry)
{
	*tx_entry = rxm_tx_entry_get(&rxm_ep->send_queue);
	if (OFI_UNLIKELY(!*tx_entry))
		return -FI_EAGAIN;

	(*tx_entry)->count = count;
	(*tx_entry)->context = context;
	(*tx_entry)->flags = flags;
	(*tx_entry)->tx_buf = tx_buf;
	(*tx_entry)->comp_flags = comp_flags | FI_SEND;

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_format_tx_res(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		     void *context, uint8_t count, size_t len, uint64_t data,
		     uint64_t flags, uint64_t tag, uint64_t comp_flags,
		     struct rxm_tx_buf **tx_buf, struct rxm_tx_entry **tx_entry,
		     struct rxm_buf_pool *pool)
{
	ssize_t ret;

	ret = rxm_ep_format_tx_res_lightweight(rxm_ep, rxm_conn, len, data,
					       flags, tag, tx_buf, pool);
	if (OFI_UNLIKELY(ret))
		return ret;

	ret = rxm_ep_format_tx_entry(rxm_ep, context, count, flags, comp_flags,
				     *tx_buf, tx_entry);
	if (OFI_UNLIKELY(ret))
		goto err;

	return FI_SUCCESS;
err:
	rxm_tx_buf_release(rxm_ep, *tx_buf);
	return ret;
}

static inline ssize_t
rxm_ep_normal_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   struct rxm_tx_entry *tx_entry, size_t pkt_size)
{
	ssize_t ret = fi_send(rxm_conn->msg_ep, &tx_entry->tx_buf->pkt, pkt_size,
			      tx_entry->tx_buf->hdr.desc, 0, tx_entry);
	if (OFI_UNLIKELY(ret)) {
		if (ret == -FI_EAGAIN)
			rxm_ep_progress_multi(&rxm_ep->util_ep);
		else
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"fi_send for MSG provider failed\n");
		rxm_tx_buf_release(rxm_ep, tx_entry->tx_buf);
		rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	}
	return ret;
}

static inline ssize_t
rxm_ep_alloc_lmt_tx_res(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
			void *context, uint8_t count, const struct iovec *iov,
			void **desc, size_t data_len, uint64_t data, uint64_t flags,
			uint64_t tag, uint64_t comp_flags, uint8_t op,
			struct rxm_tx_entry **tx_entry)
{
	struct rxm_tx_buf *tx_buf;
	struct fid_mr **mr_iov;
	ssize_t ret;

	/* Use LMT buf pool instead of buf pool provided to the function */
	ret = rxm_ep_format_tx_res(rxm_ep, rxm_conn, context, (uint8_t)count,
				   data_len, data, flags, tag, comp_flags,
				   &tx_buf, tx_entry,
				   &rxm_ep->buf_pools[RXM_BUF_POOL_TX_LMT]);
	if (OFI_UNLIKELY(ret))
		return ret;
	tx_buf->pkt.hdr.op = op;
	tx_buf->pkt.ctrl_hdr.msg_id = rxm_txe_fs_index(rxm_ep->send_queue.fs,
						       (*tx_entry));
	if (!rxm_ep->rxm_mr_local) {
		ret = rxm_ep_msg_mr_regv(rxm_ep, iov, (*tx_entry)->count,
					 FI_REMOTE_READ, (*tx_entry)->mr);
		if (ret)
			goto err;
		mr_iov = (*tx_entry)->mr;
	} else {
		/* desc is msg fid_mr * array */
		mr_iov = (struct fid_mr **)desc;
	}

	return rxm_rma_iov_init(rxm_ep, &(*tx_entry)->tx_buf->pkt.data, iov,
				count, mr_iov);
err:
	rxm_tx_entry_release(&rxm_ep->send_queue, (*tx_entry));
	rxm_tx_buf_release(rxm_ep, tx_buf);
	return ret;
}

static inline ssize_t
rxm_ep_lmt_tx_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   struct rxm_tx_entry *tx_entry, size_t pkt_size)
{
	ssize_t ret;

	RXM_LOG_STATE(FI_LOG_EP_DATA, tx_entry->tx_buf->pkt,
		      RXM_TX, RXM_LMT_TX);
	if (pkt_size <= rxm_ep->msg_info->tx_attr->inject_size) {
		RXM_LOG_STATE(FI_LOG_CQ, tx_entry->tx_buf->pkt,
			      RXM_LMT_TX, RXM_LMT_ACK_WAIT);
		tx_entry->state = RXM_LMT_ACK_WAIT;

		ret = fi_inject(rxm_conn->msg_ep, &tx_entry->tx_buf->pkt, pkt_size, 0);
	} else {
		tx_entry->state = RXM_LMT_TX;

		ret = rxm_ep_normal_send(rxm_ep, rxm_conn, tx_entry, pkt_size);
	}
	if (OFI_UNLIKELY(ret))
		goto err;
	return FI_SUCCESS;
err:
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
	       "Transmit for MSG provider failed\n");
	if (!rxm_ep->rxm_mr_local)
		rxm_ep_msg_mr_closev(tx_entry->mr, tx_entry->count);
	rxm_tx_buf_release(rxm_ep, tx_entry->tx_buf);
	rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	return ret;
}

static inline ssize_t
rxm_ep_inject_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		   struct rxm_tx_buf *tx_buf, size_t pkt_size)
{
	ssize_t ret = fi_inject(rxm_conn->msg_ep, &tx_buf->pkt, pkt_size, 0);
	if (OFI_LIKELY(!ret))
		rxm_cntr_inc(rxm_ep->util_ep.tx_cntr);
	/* release allocated buffer for further reuse */
	rxm_tx_buf_release(rxm_ep, tx_buf);
	return ret;
}

void rxm_ep_handle_postponed_tx_op(struct rxm_ep *rxm_ep,
				   struct rxm_conn *rxm_conn,
				   struct rxm_tx_entry *tx_entry)
{
	size_t tx_size = rxm_pkt_size + tx_entry->tx_buf->pkt.hdr.size;

	tx_entry->tx_buf->pkt.ctrl_hdr.conn_id = rxm_conn->handle.remote_key;
	FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
	       "Send deffered TX request (len - %"PRIu64") for %p conn\n",
	       tx_entry->tx_buf->pkt.hdr.size, rxm_conn);

	if ((tx_size <= rxm_ep->msg_info->tx_attr->inject_size) &&
	    (tx_entry->flags & FI_INJECT) && !(tx_entry->flags & FI_COMPLETION))  {
		(void) rxm_ep_inject_send(rxm_ep, rxm_conn,
					  tx_entry->tx_buf, tx_size);
		/* Release TX entry for futher reuse */
		rxm_tx_entry_release(&rxm_ep->send_queue, tx_entry);
	} else if (tx_entry->tx_buf->pkt.hdr.size >
			rxm_ep->rxm_info->tx_attr->inject_size) {
		struct rxm_rma_iov *rma_iov =
			(struct rxm_rma_iov *)&tx_entry->tx_buf->pkt.data;
		(void) rxm_ep_lmt_tx_send(rxm_ep, rxm_conn, tx_entry,
					  rxm_pkt_size + sizeof(*rma_iov) +
					  sizeof(*rma_iov->iov) * tx_entry->count);
	} else {
		(void) rxm_ep_normal_send(rxm_ep, rxm_conn, tx_entry, tx_size);
	}
}

static inline ssize_t
rxm_ep_postpone_send(struct rxm_ep *rxm_ep, struct rxm_conn *rxm_conn,
		     void *context, uint8_t count, const struct iovec *iov,
		     void **desc, size_t len, uint64_t data, uint64_t flags,
		     uint64_t tag, uint64_t comp_flags,
		     struct rxm_buf_pool *pool, uint8_t op)
{
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;

	FI_DBG(&rxm_prov, FI_LOG_EP_DATA,
	       "Buffer TX request (len - %zd) for %p conn\n", len, rxm_conn);

	if (len > rxm_ep->rxm_info->tx_attr->inject_size) {
		if (rxm_ep_alloc_lmt_tx_res(rxm_ep, rxm_conn, context,
					    count, iov, desc, len, data,
					    flags, tag, comp_flags,
					    op, &tx_entry) < 0)
			return -FI_EAGAIN;
	} else {
		ssize_t ret = rxm_ep_format_tx_res(rxm_ep, rxm_conn, context, count,
						   len, data, flags, tag, comp_flags,
						   &tx_buf, &tx_entry, pool);
		if (OFI_UNLIKELY(ret))
			return ret;
		ofi_copy_from_iov(tx_buf->pkt.data, tx_buf->pkt.hdr.size,
				  iov, count, 0);
		tx_entry->state = RXM_TX;
	}

	dlist_insert_tail(&tx_entry->postponed_entry, &rxm_conn->postponed_tx_list);

	return FI_SUCCESS;
}

static inline ssize_t
rxm_ep_inject_common(struct rxm_ep *rxm_ep, const void *buf, size_t len,
		     fi_addr_t dest_addr, uint64_t data, uint64_t flags,
		     uint64_t tag, uint64_t comp_flags,
		     struct rxm_buf_pool *pool)
{
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	struct rxm_tx_entry *tx_entry = NULL;
	struct rxm_tx_buf *tx_buf;
	size_t pkt_size = rxm_pkt_size + len;
	ssize_t ret;

	assert(len <= rxm_ep->rxm_info->tx_attr->inject_size);

	fastlock_acquire(&rxm_ep->util_ep.cmap->lock);
	handle = ofi_cmap_acquire_handle(rxm_ep->util_ep.cmap, dest_addr);
	if (OFI_UNLIKELY(!handle)) {
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return -FI_EAGAIN;
	} else if (OFI_UNLIKELY(handle->state != CMAP_CONNECTED)) {
		struct iovec iov = {
			.iov_base = (void *)buf,
			.iov_len = len,
		};
		ret = ofi_cmap_handle_connect(rxm_ep->util_ep.cmap,
					      dest_addr, handle);
		if (ret && (!rxm_defer_requests || OFI_UNLIKELY(ret != -FI_EAGAIN)))
			goto cmap_err;
		rxm_conn = container_of(handle, struct rxm_conn, handle);
		ret = rxm_ep_postpone_send(rxm_ep, rxm_conn, NULL, 1,
					   &iov, NULL, len, data, flags,
					   tag, comp_flags, pool, 0);
cmap_err:
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return ret;
	}
	fastlock_release(&rxm_ep->util_ep.cmap->lock);
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	assert(dlist_empty(&rxm_conn->postponed_tx_list));

	if (pkt_size <= rxm_ep->msg_info->tx_attr->inject_size) {
		ret = rxm_ep_format_tx_res_lightweight(rxm_ep, rxm_conn, len,
						       data, flags, tag,
						       &tx_buf, pool);
		if (OFI_UNLIKELY(ret))
	    		return ret;
		memcpy(tx_buf->pkt.data, buf, tx_buf->pkt.hdr.size);
		return rxm_ep_inject_send(rxm_ep, rxm_conn, tx_buf, pkt_size);
	} else {
		FI_DBG(&rxm_prov, FI_LOG_EP_DATA, "passed data (size = %zu) "
		       "is too big for MSG provider (max inject size = %zd)\n",
		       pkt_size, rxm_ep->msg_info->tx_attr->inject_size);
		ret = rxm_ep_format_tx_res(rxm_ep, rxm_conn, NULL, 1,
					   len, data, flags, tag, comp_flags,
					   &tx_buf, &tx_entry, pool);
		if (OFI_UNLIKELY(ret))
			return ret;
		memcpy(tx_buf->pkt.data, buf, tx_buf->pkt.hdr.size);
		tx_entry->state = RXM_TX;
		return rxm_ep_normal_send(rxm_ep, rxm_conn, tx_entry, pkt_size);
	}
}

// TODO handle all flags
static ssize_t
rxm_ep_send_common(struct rxm_ep *rxm_ep, const struct iovec *iov, void **desc,
		   size_t count, fi_addr_t dest_addr, void *context, uint64_t data,
		   uint64_t flags, uint64_t tag, uint64_t comp_flags,
		   struct rxm_buf_pool *pool, uint8_t op)
{
	struct util_cmap_handle *handle;
	struct rxm_conn *rxm_conn;
	struct rxm_tx_entry *tx_entry;
	struct rxm_tx_buf *tx_buf;
	size_t data_len = ofi_total_iov_len(iov, count);
	ssize_t ret;

	assert(count <= rxm_ep->rxm_info->tx_attr->iov_limit);

	fastlock_acquire(&rxm_ep->util_ep.cmap->lock);
	handle = ofi_cmap_acquire_handle(rxm_ep->util_ep.cmap, dest_addr);
	if (OFI_UNLIKELY(!handle)) {
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return -FI_EAGAIN;
	} else if (OFI_UNLIKELY(handle->state != CMAP_CONNECTED)) {
		ret = ofi_cmap_handle_connect(rxm_ep->util_ep.cmap,
					      dest_addr, handle);
		if (ret && (!rxm_defer_requests || OFI_UNLIKELY(ret != -FI_EAGAIN)))
			goto cmap_err;
		rxm_conn = container_of(handle, struct rxm_conn, handle);
		ret = rxm_ep_postpone_send(
				rxm_ep, rxm_conn, context, count, iov,
				desc, data_len, data, flags, tag, comp_flags,
				(data_len <=
					rxm_ep->rxm_info->tx_attr->inject_size ?
				 pool :
				 &rxm_ep->buf_pools[RXM_BUF_POOL_TX_LMT]), op);
cmap_err:
		fastlock_release(&rxm_ep->util_ep.cmap->lock);
		return ret;
	}
	fastlock_release(&rxm_ep->util_ep.cmap->lock);
	rxm_conn = container_of(handle, struct rxm_conn, handle);

	assert(dlist_empty(&rxm_conn->postponed_tx_list));

	if (data_len > rxm_ep->rxm_info->tx_attr->inject_size) {
		if (OFI_UNLIKELY(flags & FI_INJECT)) {
			FI_WARN(&rxm_prov, FI_LOG_EP_DATA,
				"inject size supported: %zu, msg size: %zu\n",
				rxm_tx_attr.inject_size, data_len);
			return -FI_EMSGSIZE;
		}
		ret = rxm_ep_alloc_lmt_tx_res(rxm_ep, rxm_conn, context, (uint8_t)count,
					      iov, desc, data_len, data, flags, tag,
					      comp_flags, op, &tx_entry);
		if (OFI_UNLIKELY(ret < 0))
			return ret;			
		return rxm_ep_lmt_tx_send(rxm_ep, rxm_conn, tx_entry, rxm_pkt_size + ret);
	} else {
		size_t total_len = rxm_pkt_size + data_len;

		ret = rxm_ep_format_tx_res_lightweight(rxm_ep, rxm_conn, data_len, data,
						       flags, tag, &tx_buf, pool);
		if (OFI_UNLIKELY(ret))
			return ret;
		ofi_copy_from_iov(tx_buf->pkt.data, tx_buf->pkt.hdr.size,
				  iov, count, 0);

		if ((flags & FI_INJECT) && !(flags & FI_COMPLETION) &&
		    (total_len <= rxm_ep->msg_info->tx_attr->inject_size))
			return rxm_ep_inject_send(rxm_ep, rxm_conn, tx_buf, total_len);

		ret = rxm_ep_format_tx_entry(rxm_ep, context, (uint8_t)count,
					     flags, comp_flags, tx_buf, &tx_entry);
		if (OFI_UNLIKELY(ret)) {
			rxm_tx_buf_release(rxm_ep, tx_buf);
			return ret;
		}
		tx_entry->state = RXM_TX;
		return rxm_ep_normal_send(rxm_ep, rxm_conn, tx_entry, total_len);
	}
}

#define rxm_ep_tx_flags_inject(rxm_ep) \
	((rxm_ep_tx_flags(rxm_ep) & ~FI_COMPLETION) | FI_INJECT)

static ssize_t rxm_ep_sendmsg(struct fid_ep *ep_fid, const struct fi_msg *msg,
			      uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
				  msg->addr, msg->context, msg->data,
				  flags | (rxm_ep_tx_flags(rxm_ep) & FI_COMPLETION),
				  0, FI_MSG, &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG],
				  ofi_op_msg);
}

static ssize_t rxm_ep_send(struct fid_ep *ep_fid, const void *buf, size_t len,
			   void *desc, fi_addr_t dest_addr, void *context)
{
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, &iov, &desc, 1, dest_addr, context, 0,
				  rxm_ep_tx_flags(rxm_ep), 0, FI_MSG,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG],
				  ofi_op_msg);
}

static ssize_t rxm_ep_sendv(struct fid_ep *ep_fid, const struct iovec *iov,
			    void **desc, size_t count, fi_addr_t dest_addr,
			    void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, iov, desc, count, dest_addr, context, 0,
				  rxm_ep_tx_flags(rxm_ep), 0, FI_MSG,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG], ofi_op_msg);
}

static ssize_t rxm_ep_inject(struct fid_ep *ep_fid, const void *buf, size_t len,
			     fi_addr_t dest_addr)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_inject_common(rxm_ep, buf, len, dest_addr, 0,
				    rxm_ep_tx_flags_inject(rxm_ep),
				    0, FI_MSG, &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG]);
}

static ssize_t rxm_ep_senddata(struct fid_ep *ep_fid, const void *buf, size_t len,
			       void *desc, uint64_t data, fi_addr_t dest_addr,
			       void *context)
{
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, &iov, desc, 1, dest_addr, context, data,
				  rxm_ep_tx_flags(rxm_ep) | FI_REMOTE_CQ_DATA,
				  0, FI_MSG, &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG],
				  ofi_op_msg);
}

static ssize_t rxm_ep_injectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				 uint64_t data, fi_addr_t dest_addr)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_inject_common(rxm_ep, buf, len, dest_addr, data,
				    rxm_ep_tx_flags_inject(rxm_ep) | FI_REMOTE_CQ_DATA,
				    0, FI_MSG, &rxm_ep->buf_pools[RXM_BUF_POOL_TX_MSG]);
}

static struct fi_ops_msg rxm_ops_msg = {
	.size = sizeof(struct fi_ops_msg),
	.recv = rxm_ep_recv,
	.recvv = rxm_ep_recvv,
	.recvmsg = rxm_ep_recvmsg,
	.send = rxm_ep_send,
	.sendv = rxm_ep_sendv,
	.sendmsg = rxm_ep_sendmsg,
	.inject = rxm_ep_inject,
	.senddata = rxm_ep_senddata,
	.injectdata = rxm_ep_injectdata,
};

static ssize_t rxm_ep_trecvmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common_flags(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
					msg->addr, msg->tag, msg->ignore, msg->context,
					flags, (rxm_ep_rx_flags(rxm_ep) & FI_COMPLETION),
					&rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_trecv(struct fid_ep *ep_fid, void *buf, size_t len,
			    void *desc, fi_addr_t src_addr, uint64_t tag,
			    uint64_t ignore, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);
	struct iovec iov = {
		.iov_base	= buf,
		.iov_len	= len,
	};

	return rxm_ep_recv_common(rxm_ep, &iov, &desc, 1, src_addr, tag, ignore,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_trecvv(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t src_addr,
			     uint64_t tag, uint64_t ignore, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_recv_common(rxm_ep, iov, desc, count, src_addr, tag, ignore,
				  context, rxm_ep_rx_flags(rxm_ep),
				  &rxm_ep->trecv_queue);
}

static ssize_t rxm_ep_tsendmsg(struct fid_ep *ep_fid, const struct fi_msg_tagged *msg,
			       uint64_t flags)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, msg->msg_iov, msg->desc, msg->iov_count,
				  msg->addr, msg->context, msg->data,
				  flags | (rxm_ep_tx_flags(rxm_ep) & FI_COMPLETION),
				  msg->tag, FI_TAGGED,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED],
				  ofi_op_tagged);
}

static ssize_t rxm_ep_tsend(struct fid_ep *ep_fid, const void *buf, size_t len,
			    void *desc, fi_addr_t dest_addr, uint64_t tag,
			    void *context)
{
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, &iov, &desc, 1, dest_addr, context, 0,
				  rxm_ep_tx_flags(rxm_ep), tag, FI_TAGGED,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED],
				  ofi_op_tagged);
}

static ssize_t rxm_ep_tsendv(struct fid_ep *ep_fid, const struct iovec *iov,
			     void **desc, size_t count, fi_addr_t dest_addr,
			     uint64_t tag, void *context)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, iov, desc, count, dest_addr, context, 0,
				  rxm_ep_tx_flags(rxm_ep), tag, FI_TAGGED,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED],
				  ofi_op_tagged);
}

static ssize_t rxm_ep_tinject(struct fid_ep *ep_fid, const void *buf, size_t len,
			      fi_addr_t dest_addr, uint64_t tag)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_inject_common(rxm_ep, buf, len, dest_addr, 0,
				    rxm_ep_tx_flags_inject(rxm_ep), tag, FI_TAGGED,
				    &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED]);
}

static ssize_t rxm_ep_tsenddata(struct fid_ep *ep_fid, const void *buf, size_t len,
				void *desc, uint64_t data, fi_addr_t dest_addr,
				uint64_t tag, void *context)
{
	struct iovec iov = {
		.iov_base = (void *)buf,
		.iov_len = len,
	};
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_send_common(rxm_ep, &iov, desc, 1, dest_addr, context, data,
				  rxm_ep_tx_flags(rxm_ep) | FI_REMOTE_CQ_DATA,
				  tag, FI_TAGGED,
				  &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED],
				  ofi_op_tagged);
}

static ssize_t rxm_ep_tinjectdata(struct fid_ep *ep_fid, const void *buf, size_t len,
				  uint64_t data, fi_addr_t dest_addr, uint64_t tag)
{
	struct rxm_ep *rxm_ep = container_of(ep_fid, struct rxm_ep,
					     util_ep.ep_fid.fid);

	return rxm_ep_inject_common(rxm_ep, buf, len, dest_addr, data,
				    rxm_ep_tx_flags_inject(rxm_ep) | FI_REMOTE_CQ_DATA,
				    tag, FI_TAGGED,
				    &rxm_ep->buf_pools[RXM_BUF_POOL_TX_TAGGED]);
}

struct fi_ops_tagged rxm_ops_tagged = {
	.size = sizeof(struct fi_ops_tagged),
	.recv = rxm_ep_trecv,
	.recvv = rxm_ep_trecvv,
	.recvmsg = rxm_ep_trecvmsg,
	.send = rxm_ep_tsend,
	.sendv = rxm_ep_tsendv,
	.sendmsg = rxm_ep_tsendmsg,
	.inject = rxm_ep_tinject,
	.senddata = rxm_ep_tsenddata,
	.injectdata = rxm_ep_tinjectdata,
};

static int rxm_ep_msg_res_close(struct rxm_ep *rxm_ep)
{
	int ret, retv = 0;

	if (rxm_ep->srx_ctx) {
		ret = fi_close(&rxm_ep->srx_ctx->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, \
				"Unable to close msg shared ctx\n");
			retv = ret;
		}
	}

	fi_freeinfo(rxm_ep->msg_info);
	return retv;
}

static int rxm_listener_close(struct rxm_ep *rxm_ep)
{
	int ret, retv = 0;

	if (rxm_ep->msg_pep) {
		ret = fi_close(&rxm_ep->msg_pep->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to close msg pep\n");
			retv = ret;
		}
	}
	if (rxm_ep->msg_eq) {
		ret = fi_close(&rxm_ep->msg_eq->fid);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to close msg EQ\n");
			retv = ret;
		}
	}
	return retv;
}

static int rxm_ep_close(struct fid *fid)
{
	int ret, retv = 0;
	struct rxm_ep *rxm_ep =
		container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);
	struct rxm_ep_wait_ref *wait_ref;
	struct dlist_entry *tmp_list_entry;

	dlist_foreach_container_safe(&rxm_ep->msg_cq_fd_ref_list,
				     struct rxm_ep_wait_ref,
				     wait_ref, entry, tmp_list_entry) {
		ret = ofi_wait_fd_del(wait_ref->wait,
				      rxm_ep->msg_cq_fd);
		if (ret)
			retv = ret;
		dlist_remove(&wait_ref->entry);
		free(wait_ref);
	}
	OFI_UNUSED(tmp_list_entry); /* to avoid "set, but not used" warning*/

	if (rxm_ep->util_ep.cmap)
		ofi_cmap_free(rxm_ep->util_ep.cmap);

	ret = rxm_listener_close(rxm_ep);
	if (ret)
		retv = ret;

	rxm_ep_txrx_res_close(rxm_ep);

	ret = fi_close(&rxm_ep->msg_cq->fid);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
		retv = ret;
	}

	ret = rxm_ep_msg_res_close(rxm_ep);
	if (ret)
		retv = ret;

	ofi_endpoint_close(&rxm_ep->util_ep);
	free(rxm_ep);
	return retv;
}

static int rxm_ep_msg_get_wait_cq_fd(struct rxm_ep *rxm_ep,
				     enum fi_wait_obj wait_obj)
{
	int ret = FI_SUCCESS;

	if ((wait_obj != FI_WAIT_NONE) && (!rxm_ep->msg_cq_fd)) {
		ret = fi_control(&rxm_ep->msg_cq->fid, FI_GETWAIT, &rxm_ep->msg_cq_fd);
		if (ret)
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to get MSG CQ fd\n");
	}
	return ret;
}

static int rxm_ep_msg_cq_open(struct rxm_ep *rxm_ep, enum fi_wait_obj wait_obj)
{
	struct rxm_domain *rxm_domain;
	struct fi_cq_attr cq_attr = { 0 };
	int ret;

	assert((wait_obj == FI_WAIT_NONE) || (wait_obj == FI_WAIT_FD));

	cq_attr.size = (rxm_ep->msg_info->tx_attr->size +
			rxm_ep->msg_info->rx_attr->size) * rxm_def_univ_size;
	cq_attr.format = FI_CQ_FORMAT_DATA;
	cq_attr.wait_obj = wait_obj;

	rxm_domain = container_of(rxm_ep->util_ep.domain, struct rxm_domain, util_domain);

	ret = fi_cq_open(rxm_domain->msg_domain, &cq_attr, &rxm_ep->msg_cq, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open MSG CQ\n");
		return ret;
	}

	ret = rxm_ep_msg_get_wait_cq_fd(rxm_ep, wait_obj);
	if (ret)
		goto err;

	return 0;
err:
	ret = fi_close(&rxm_ep->msg_cq->fid);
	if (ret)
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
	return ret;
}

static int rxm_ep_trywait(void *arg)
{
	struct rxm_fabric *rxm_fabric;
	struct rxm_ep *rxm_ep = (struct rxm_ep *)arg;
	struct fid *fids[1] = {&rxm_ep->msg_cq->fid};

	rxm_fabric = container_of(rxm_ep->util_ep.domain->fabric,
				  struct rxm_fabric, util_fabric);
	return fi_trywait(rxm_fabric->msg_fabric, fids, 1);
}

static int rxm_ep_bind(struct fid *ep_fid, struct fid *bfid, uint64_t flags)
{
	struct rxm_ep *rxm_ep =
		container_of(ep_fid, struct rxm_ep, util_ep.ep_fid.fid);
	struct util_cq *cq;
	struct util_av *av;
	struct util_cntr *cntr;
	struct rxm_ep_wait_ref *wait_ref = NULL;
	int ret = 0;

	switch (bfid->fclass) {
	case FI_CLASS_AV:
		av = container_of(bfid, struct util_av, av_fid.fid);
		ret = ofi_ep_bind_av(&rxm_ep->util_ep, av);
		if (ret)
			return ret;
		break;
	case FI_CLASS_CQ:
		cq = container_of(bfid, struct util_cq, cq_fid.fid);

		ret = ofi_ep_bind_cq(&rxm_ep->util_ep, cq, flags);
		if (ret)
			return ret;

		if (!rxm_ep->msg_cq) {
			ret = rxm_ep_msg_cq_open(rxm_ep, cq->wait ?
						 FI_WAIT_FD : FI_WAIT_NONE);
			if (ret)
				return ret;
		}

		if (cq->wait) {
			wait_ref = calloc(1, sizeof(struct rxm_ep_wait_ref));
			if (!wait_ref) {
				ret = -FI_ENOMEM;
				goto err1;
			}
			wait_ref->wait = cq->wait;
			dlist_insert_tail(&wait_ref->entry,
					  &rxm_ep->msg_cq_fd_ref_list);
			ret = ofi_wait_fd_add(cq->wait, rxm_ep->msg_cq_fd,
					      rxm_ep_trywait, rxm_ep,
					      &rxm_ep->util_ep.ep_fid.fid);
			if (ret)
				goto err2;
		}
		break;
	case FI_CLASS_CNTR:
		cntr = container_of(bfid, struct util_cntr, cntr_fid.fid);

		ret = ofi_ep_bind_cntr(&rxm_ep->util_ep, cntr, flags);
		if (ret)
			return ret;

		if (!rxm_ep->msg_cq) {
			ret = rxm_ep_msg_cq_open(rxm_ep, cntr->wait ?
						 FI_WAIT_FD : FI_WAIT_NONE);
			if (ret)
				return ret;
		} else if (!rxm_ep->msg_cq_fd && cntr->wait) {
			/* Reopen CQ with WAIT fd set */
			ret = fi_close(&rxm_ep->msg_cq->fid);
			if (ret)
				FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to close msg CQ\n");
			ret = rxm_ep_msg_cq_open(rxm_ep, FI_WAIT_FD);
			if (ret)
				return ret;
		}

		if (cntr->wait) {
			wait_ref = calloc(1, sizeof(struct rxm_ep_wait_ref));
			if (!wait_ref) {
				ret = -FI_ENOMEM;
				goto err1;
			}
			wait_ref->wait = cntr->wait;
			dlist_insert_tail(&wait_ref->entry,
					  &rxm_ep->msg_cq_fd_ref_list);
			ret = ofi_wait_fd_add(cntr->wait, rxm_ep->msg_cq_fd,
					      rxm_ep_trywait, rxm_ep,
					      &rxm_ep->util_ep.ep_fid.fid);
			if (ret)
				goto err2;
		}
		break;
	case FI_CLASS_EQ:
		break;
	default:
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "invalid fid class\n");
		ret = -FI_EINVAL;
		break;
	}
	return ret;
err2:
	free(wait_ref);
err1:
	if (fi_close(&rxm_ep->msg_cq->fid))
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to close msg CQ\n");
	return ret;
}

static int rxm_ep_ctrl(struct fid *fid, int command, void *arg)
{
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = container_of(fid, struct rxm_ep, util_ep.ep_fid.fid);

	switch (command) {
	case FI_ENABLE:
		if (!rxm_ep->util_ep.rx_cq || !rxm_ep->util_ep.tx_cq)
			return -FI_ENOCQ;
		if (!rxm_ep->util_ep.av)
			return -FI_EOPBADSTATE;

		ret = fi_listen(rxm_ep->msg_pep);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to set msg PEP to listen state\n");
			return ret;
		}

		rxm_ep->util_ep.cmap = rxm_conn_cmap_alloc(rxm_ep);
		if (!rxm_ep->util_ep.cmap)
			return -FI_ENOMEM;

		if (rxm_ep->srx_ctx) {
			ret = rxm_ep_prepost_buf(rxm_ep, rxm_ep->srx_ctx);
			if (ret) {
				ofi_cmap_free(rxm_ep->util_ep.cmap);
				FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
					"Unable to prepost recv bufs\n");
				return ret;
			}
		}
		break;
	default:
		return -FI_ENOSYS;
	}
	return 0;
}

static struct fi_ops rxm_ep_fi_ops = {
	.size = sizeof(struct fi_ops),
	.close = rxm_ep_close,
	.bind = rxm_ep_bind,
	.control = rxm_ep_ctrl,
	.ops_open = fi_no_ops_open,
};

static int rxm_listener_open(struct rxm_ep *rxm_ep)
{
	struct rxm_fabric *rxm_fabric;
	struct fi_eq_attr eq_attr;
	eq_attr.wait_obj = FI_WAIT_UNSPEC;
	eq_attr.flags = FI_WRITE;
	int ret;

	rxm_fabric = container_of(rxm_ep->util_ep.domain->fabric,
				  struct rxm_fabric, util_fabric);

	ret = fi_eq_open(rxm_fabric->msg_fabric, &eq_attr, &rxm_ep->msg_eq, NULL);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open msg EQ\n");
		return ret;
	}

	ret = fi_passive_ep(rxm_fabric->msg_fabric, rxm_ep->msg_info,
			    &rxm_ep->msg_pep, rxm_ep);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Unable to open msg PEP\n");
		goto err;
	}

	ret = fi_pep_bind(rxm_ep->msg_pep, &rxm_ep->msg_eq->fid, 0);
	if (ret) {
		FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to bind msg PEP to msg EQ\n");
		goto err;
	}

	return 0;
err:
	rxm_listener_close(rxm_ep);
	return ret;
}

static int rxm_info_to_core_srx_ctx(uint32_t version, const struct fi_info *rxm_hints,
				    struct fi_info *core_hints)
{
	int ret;

	ret = rxm_info_to_core(version, rxm_hints, core_hints);
	if (ret)
		return ret;
	core_hints->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;
	return 0;
}

static int rxm_ep_get_core_info(uint32_t version, const struct fi_info *hints,
				struct fi_info **info)
{
	int ret;

	ret = ofi_get_core_info(version, NULL, NULL, 0, &rxm_util_prov, hints,
				rxm_info_to_core_srx_ctx, info);
	if (!ret)
		return 0;

	FI_WARN(&rxm_prov, FI_LOG_EP_CTRL, "Shared receive context not "
		"supported by MSG provider.\n");

	return ofi_get_core_info(version, NULL, NULL, 0, &rxm_util_prov, hints,
				 rxm_info_to_core, info);
}

static int rxm_ep_msg_res_open(struct util_domain *util_domain,
			       struct rxm_ep *rxm_ep)
{
	int ret;
	size_t max_prog_val;
	struct rxm_domain *rxm_domain =
		container_of(util_domain, struct rxm_domain, util_domain);

	ret = rxm_ep_get_core_info(util_domain->fabric->fabric_fid.api_version,
				   rxm_ep->rxm_info, &rxm_ep->msg_info);
	if (ret)
		return ret;

	max_prog_val = MIN(rxm_ep->msg_info->tx_attr->size,
			   rxm_ep->msg_info->rx_attr->size) / 2;
	rxm_ep->comp_per_progress = (rxm_ep->comp_per_progress > max_prog_val) ?
				    max_prog_val : rxm_ep->comp_per_progress;

	dlist_init(&rxm_ep->msg_cq_fd_ref_list);

	if (rxm_ep->msg_info->ep_attr->rx_ctx_cnt == FI_SHARED_CONTEXT) {
		ret = fi_srx_context(rxm_domain->msg_domain, rxm_ep->msg_info->rx_attr,
				     &rxm_ep->srx_ctx, NULL);
		if (ret) {
			FI_WARN(&rxm_prov, FI_LOG_EP_CTRL,
				"Unable to open shared receive context\n");
			goto err1;
		}
	}

	ret = rxm_listener_open(rxm_ep);
	if (ret)
		goto err2;

	/* Zero out the port as we would be creating multiple MSG EPs for a single
	 * RXM EP and we don't want address conflicts. */
	if (rxm_ep->msg_info->src_addr) {
		if (((struct sockaddr *)rxm_ep->msg_info->src_addr)->sa_family == AF_INET)
			((struct sockaddr_in *)(rxm_ep->msg_info->src_addr))->sin_port = 0;
		else
			((struct sockaddr_in6 *)(rxm_ep->msg_info->src_addr))->sin6_port = 0;
	}
	return 0;
err2:
	fi_close(&rxm_ep->srx_ctx->fid);
err1:
	fi_freeinfo(rxm_ep->msg_info);
	return ret;
}

int rxm_endpoint(struct fid_domain *domain, struct fi_info *info,
		 struct fid_ep **ep_fid, void *context)
{
	struct util_domain *util_domain;
	struct rxm_ep *rxm_ep;
	int ret;

	rxm_ep = calloc(1, sizeof(*rxm_ep));
	if (!rxm_ep)
		return -FI_ENOMEM;

	rxm_ep->rxm_info = fi_dupinfo(info);
	if (!rxm_ep->rxm_info) {
		ret = -FI_ENOMEM;
		goto err1;
	}

	if (!fi_param_get_int(&rxm_prov, "comp_per_progress",
			     (int *)&rxm_ep->comp_per_progress)) {
		ret = ofi_endpoint_init(domain, &rxm_util_prov,
					info, &rxm_ep->util_ep,
					context, &rxm_ep_progress_multi);
	} else {
		rxm_ep->comp_per_progress = 1;
		ret = ofi_endpoint_init(domain, &rxm_util_prov,
					info, &rxm_ep->util_ep,
					context, &rxm_ep_progress_one);
		if (ret)
			goto err1;
	}
	if (ret)
		goto err1;


	util_domain = container_of(domain, struct util_domain, domain_fid);

	ret = rxm_ep_msg_res_open(util_domain, rxm_ep);
	if (ret)
		goto err2;

	rxm_ep->msg_mr_local = OFI_CHECK_MR_LOCAL(rxm_ep->msg_info);
	rxm_ep->rxm_mr_local = OFI_CHECK_MR_LOCAL(rxm_ep->rxm_info);

	rxm_ep->min_multi_recv_size = rxm_ep->rxm_info->tx_attr->inject_size;

	ret = rxm_ep_txrx_res_open(rxm_ep);
	if (ret)
		goto err3;

	*ep_fid = &rxm_ep->util_ep.ep_fid;
	(*ep_fid)->fid.ops = &rxm_ep_fi_ops;
	(*ep_fid)->ops = &rxm_ops_ep;
	(*ep_fid)->cm = &rxm_ops_cm;
	(*ep_fid)->msg = &rxm_ops_msg;
	(*ep_fid)->tagged = &rxm_ops_tagged;
	(*ep_fid)->rma = &rxm_ops_rma;

	return 0;
err3:
	rxm_ep_msg_res_close(rxm_ep);
err2:
	ofi_endpoint_close(&rxm_ep->util_ep);
err1:
	if (rxm_ep->rxm_info)
		fi_freeinfo(rxm_ep->rxm_info);
	free(rxm_ep);
	return ret;
}
